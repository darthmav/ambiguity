"""Console-level tests for the emergency stop and run recovery.

These exercise `serve.py` directly rather than over HTTP: the RPC methods are
plain functions, and the transport adds nothing worth testing here. What they
do cover is the bookkeeping around a run -- the flag, the snapshot, and the
`finally` that has to hold even when the graph raises.
"""

from __future__ import annotations

import json
import signal
import threading

import pytest

import serve
from langgraph_agent.control import ACTIVITY, RUN_CONTROL


@pytest.fixture(autouse=True)
def _isolate_run_state(tmp_path, monkeypatch):
    """Keep each test off the real snapshot file and out of the last one's state."""
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(serve, "LAST_RUN_PATH", tmp_path / "runs" / "last_run.json")
    monkeypatch.setattr(serve, "_last_run_snapshot", None)
    serve._shutdown_requested.clear()
    serve._exit_after_run.clear()
    with serve._run_lock:
        serve._run_progress.update(
            running=False, goal="", messages=[], node="", run_id="", stopping=False
        )
    yield
    serve._shutdown_requested.clear()
    serve._exit_after_run.clear()
    with serve._run_lock:
        serve._run_progress.update(running=False, run_id="", stopping=False)


class _FakeGraph:
    """A graph whose supersteps are driven by the test.

    `stream` yields one event per node name, calling `during_step` while each
    one is notionally running. That is where a test trips the stop, and the
    server gets its look at the flag once the event has been handed over -- the
    same shape as the real thing.
    """

    def __init__(self, nodes, during_step=None, raises=None):
        self._nodes = nodes
        self._during_step = during_step
        self._raises = raises
        self.steps = 0

    def stream(self, state, config):
        for name in self._nodes:
            if self._raises:
                raise self._raises
            self.steps += 1
            # Called while the node is "running", before its result is handed
            # over -- which is when a real Stop arrives.
            if self._during_step:
                self._during_step()
            state = dict(state)
            state["messages"] = [*state.get("messages", []), f"[{name.title()}] did work"]
            state["plan"] = "1. Something"
            yield {name: state}


def test_a_run_needs_a_goal(monkeypatch):
    """A blank goal is refused before the run is claimed or the corpus touched.

    The console's textarea already ignored one, which hid that the server did
    not: `/api/run` with no goal indexed the project and set four seats to
    work on nothing.
    """
    indexed = []
    monkeypatch.setattr(
        serve,
        "_rebuild_the_corpus_before_the_run",
        lambda: indexed.append(True) or {"source": "disabled"},
    )

    with pytest.raises(ValueError, match="needs a goal"):
        serve.rpc_run_goal({"goal": "   "})

    assert not indexed
    assert serve._run_progress["running"] is False
    assert not RUN_CONTROL.run_id()


class _Socket:
    """A handler's output stream, recording what was written to it."""

    def __init__(self, closed=False):
        self.closed = closed
        self.data = b""

    def write(self, data):
        if self.closed:
            raise BrokenPipeError(32, "Broken pipe")
        self.data += data

    def flush(self):
        pass


def _rpc_handler(socket):
    """A `Handler` with no connection behind it -- enough to reach `handle_rpc`."""
    handler = serve.Handler.__new__(serve.Handler)
    handler.wfile = socket
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /rpc HTTP/1.1"
    handler.command = "POST"
    return handler


def test_a_caller_that_went_away_is_not_a_failed_method(monkeypatch, capsys):
    """A console reloaded mid-run drops the request that started the run.

    The run still finishes; only its reply has nowhere to go. That read as
    `run_goal FAILED ... Broken pipe`, then two tracebacks, for a run that had
    ended cleanly.
    """
    monkeypatch.setitem(serve.RPC_METHODS, "probe", lambda params: {"ok": True})

    _rpc_handler(_Socket(closed=True)).handle_rpc({"method": "probe", "params": {}})

    out = capsys.readouterr().out
    assert "FAILED" not in out
    assert "disconnected before the reply" in out


def test_a_method_that_raises_still_fails_out_loud(monkeypatch, capsys):
    """Keeping the write apart must not quieten a method that really failed."""
    def boom(params):
        raise ValueError("no good")

    monkeypatch.setitem(serve.RPC_METHODS, "probe", boom)
    socket = _Socket()

    _rpc_handler(socket).handle_rpc({"method": "probe", "params": {}})

    assert "[RPC] probe FAILED" in capsys.readouterr().out
    body = json.loads(socket.data.split(b"\r\n\r\n", 1)[1])
    assert body["error"]["message"] == "no good"


def test_a_stop_ends_the_run_at_a_node_boundary(monkeypatch):
    """The stop breaks the stream loop, and the state so far comes back."""
    def trip():
        serve.rpc_stop_run({})

    graph = _FakeGraph(["architect", "planner", "builder"], during_step=trip)
    monkeypatch.setattr(serve, "graph", graph)

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    # Broken out after the first superstep, not after all three.
    assert graph.steps == 1
    assert result["stopped"] is True
    assert result["stop_reason"]
    # The work it did produce is returned, not discarded.
    assert "[Architect] did work" in result["messages"]
    assert "Stopped by the emergency stop" in result["messages"][-1]


def test_a_finished_run_is_not_reported_as_stopped(monkeypatch):
    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "builder"]))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert result["stopped"] is False
    assert result["over_budget"] is False


def test_stopping_with_no_run_in_flight_is_refused():
    answer = serve.rpc_stop_run({})

    assert answer["stopping"] is False
    assert "No run is in flight" in answer["detail"]


def test_a_stale_stop_is_refused_when_a_new_run_is_armed(monkeypatch):
    """The console can hold an id across a reload; it must not kill a new run."""
    seen = {}

    def trip():
        seen["answer"] = serve.rpc_stop_run({"run_id": "some-run-that-ended"})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "planner"], during_step=trip))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["answer"]["stopping"] is False
    assert "already ended" in seen["answer"]["detail"]
    assert result["stopped"] is False


def test_only_one_run_at_a_time(monkeypatch):
    """A second run is refused rather than left to clobber the first's progress."""
    started = threading.Event()
    release = threading.Event()

    def block():
        started.set()
        release.wait(5)

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect"], during_step=block))

    outcome: list = []
    worker = threading.Thread(
        target=lambda: outcome.append(serve.rpc_run_goal({"goal": "first"})),
        daemon=True,
    )
    worker.start()
    assert started.wait(5)

    with pytest.raises(ValueError, match="already in flight"):
        serve.rpc_run_goal({"goal": "second"})

    release.set()
    worker.join(5)
    assert outcome and outcome[0]["stopped"] is False


def test_a_run_that_raises_still_clears_the_flag_and_leaves_a_snapshot(monkeypatch):
    """Before the finally, a crashed run left the console polling a phantom.

    It also left nothing recoverable -- which is the case where the operator
    most wants the partial work back.
    """
    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect"], raises=RuntimeError("seat died"))
    )

    with pytest.raises(RuntimeError, match="seat died"):
        serve.rpc_run_goal({"goal": "Do a thing"})

    assert serve.rpc_run_progress({})["running"] is False
    assert RUN_CONTROL.run_id() == ""

    snapshot = serve.rpc_last_run({})["snapshot"]
    assert snapshot is not None
    assert snapshot["error"] == "seat died"


def test_the_snapshot_survives_the_console(monkeypatch):
    """A reloaded page reads the run's outcome back off disk."""
    def trip():
        serve.rpc_stop_run({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "builder"], during_step=trip))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    # In memory, for the page that is open now...
    assert serve.rpc_last_run({})["snapshot"]["run_id"] == result["run_id"]

    # ...and on disk, for the one that reloads.
    monkeypatch.setattr(serve, "_last_run_snapshot", None)
    written = json.loads(serve.LAST_RUN_PATH.read_text())
    assert written["stopped"] is True
    assert serve.rpc_last_run({})["snapshot"]["stopped"] is True


def test_run_progress_names_the_run_a_stop_would_target(monkeypatch):
    """The console cannot aim a Stop at a run it cannot name."""
    seen = {}

    def look():
        seen["progress"] = serve.rpc_run_progress({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect"], during_step=look))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["progress"]["running"] is True
    assert seen["progress"]["run_id"] == result["run_id"]
    # And it is released once the run is over, so a later Stop finds nothing.
    assert serve.rpc_run_progress({})["run_id"] == ""


def test_the_stop_is_visible_while_it_is_landing(monkeypatch):
    """A stop that has been asked for but not yet taken effect must show."""
    seen = {}

    def trip():
        serve.rpc_stop_run({})
        seen["progress"] = serve.rpc_run_progress({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "planner"], during_step=trip))
    serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["progress"]["stopping"] is True


def test_a_run_is_never_shown_the_last_runs_turns(monkeypatch):
    """The seat lights are driven from the turn record, so it starts empty each run.

    A turn the previous run left behind would otherwise read, to the new run's
    console, as a seat that had only just taken its turn.
    """
    ACTIVITY.begin_run()
    ACTIVITY.enter("builder")  # the last run's final turn, still on the record
    ACTIVITY.leave("builder")
    seen = {}

    def look():
        ACTIVITY.enter("architect")
        seen["progress"] = serve.rpc_run_progress({})
        ACTIVITY.leave("architect")

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect"], during_step=look))
    serve.rpc_run_goal({"goal": "Do a thing"})

    turns = seen["progress"]["turns"]
    assert [t["node"] for t in turns] == ["architect"]
    assert turns[0]["ended_ago"] is None


def test_expect_failures_rides_through_a_stopped_run(monkeypatch):
    """The safeguard setting is part of what a stopped run has to hand back.

    The recovery view says which rules were in force when the run was cut off,
    and it can only do that if the flag survives into the snapshot.
    """
    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect"], during_step=lambda: serve.rpc_stop_run({}))
    )

    result = serve.rpc_run_goal({"goal": "Write a failing fixture", "expect_failures": True})

    assert result["expect_failures"] is True
    assert serve.rpc_last_run({})["snapshot"]["expect_failures"] is True


# ---------------------------------------------------------------------------
# The exit button
# ---------------------------------------------------------------------------


def test_exiting_with_nothing_running_is_immediate():
    answer = serve.rpc_shutdown({})

    assert answer["exiting"] is True
    assert answer["running"] is False
    assert serve._shutdown_requested.is_set()


def test_exiting_will_not_kill_a_run_by_surprise(monkeypatch):
    """Asked to exit mid-run, the server says what is running instead."""
    seen = {}

    def look():
        seen["answer"] = serve.rpc_shutdown({})
        seen["armed"] = serve._shutdown_requested.is_set()

    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect", "planner"], during_step=look)
    )
    result = serve.rpc_run_goal({"goal": "a long job"})

    assert seen["answer"]["exiting"] is False
    assert seen["answer"]["goal"] == "a long job"
    assert seen["armed"] is False
    # The run was left alone: it finished on its own terms.
    assert result["stopped"] is False


def test_an_exit_through_a_run_waits_for_its_state(monkeypatch):
    """The exit is deferred to the run's finally, which is what saves the work.

    Exiting the moment the stop is requested would race the snapshot, and a run
    stopped only so the console could close is exactly the one worth keeping.
    """
    seen = {}

    def ask_to_exit():
        if "answer" in seen:
            return
        seen["answer"] = serve.rpc_shutdown({"stop_first": True})
        # Deferred, not immediate: the run has not handed anything back yet.
        seen["armed_during_run"] = serve._shutdown_requested.is_set()

    monkeypatch.setattr(
        serve,
        "graph",
        _FakeGraph(["architect", "planner", "builder"], during_step=ask_to_exit),
    )
    result = serve.rpc_run_goal({"goal": "a long job"})

    assert seen["answer"]["exiting"] is True
    assert seen["answer"]["running"] is True
    assert seen["armed_during_run"] is False

    # Stopped through the ordinary emergency stop, so the state came back...
    assert result["stopped"] is True
    assert "[Architect] did work" in result["messages"]
    # ...the snapshot is on disk...
    assert serve.LAST_RUN_PATH.exists()
    # ...and only then does the process get to go.
    assert serve._shutdown_requested.is_set()


def test_a_run_that_ends_the_instant_it_is_stopped_still_lets_the_exit_through(
    monkeypatch,
):
    """The exit must survive the run winning the race the stop itself starts.

    `RUN_CONTROL.stop()` is what makes the run head for its `finally`, and a
    run sitting at a superstep boundary gets there in microseconds -- before
    `rpc_shutdown` had claimed the exit. The run read `_exit_after_run` unset
    and declined to request the shutdown, correctly on what it could see; the
    flag was then set with no run left to honour it, and nothing ever set
    `_shutdown_requested`. The process stayed up and the console sat on
    "Closing...".

    The interleaving is forced here rather than raced: the run reaches its
    teardown inside `RUN_CONTROL.stop()`, which is the worst case and the one
    that failed.
    """
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="a long job", run_id="r1")
    RUN_CONTROL.arm("r1")

    def stop_and_finish_immediately(run_id="", reason=""):
        RUN_CONTROL.__class__.stop(RUN_CONTROL, run_id, reason)
        # The run hands its state back before rpc_shutdown gets any further.
        serve._finish_run()
        return True

    monkeypatch.setattr(serve.RUN_CONTROL, "stop", stop_and_finish_immediately)
    answer = serve.rpc_shutdown({"stop_first": True})

    assert answer["exiting"] is True
    # The process actually goes -- the whole point of the button.
    assert serve._shutdown_requested.is_set()
    # And the finished run is not left looking like it is still stopping.
    with serve._run_lock:
        assert serve._run_progress["stopping"] is False
        assert serve._run_progress["running"] is False


def test_an_exit_claimed_after_the_run_ended_is_taken_directly(monkeypatch):
    """The other side of the handshake: no run left, so nobody defers."""
    serve._finish_run()  # the run ended before the click landed

    answer = serve.rpc_shutdown({"stop_first": True})

    assert answer["exiting"] is True
    assert answer["running"] is False
    assert serve._shutdown_requested.is_set()


def test_a_termination_signal_exits_through_the_same_path(monkeypatch):
    """`docker stop` and `systemctl stop` must not cost a run its snapshot.

    Python's default SIGTERM handler ends the process where it stands, and
    `rpc_run_goal` writes its snapshot from a `finally` -- so the default would
    lose exactly the state a restarted console reattaches to, on the one exit
    path nobody is watching. The handler asks for the console's own exit, which
    means the deferral below is the run's, not the signal's.
    """
    seen = {}

    def look():
        serve._exit_the_way_the_console_would(signal.SIGTERM, None)
        seen["armed_during_run"] = serve._shutdown_requested.is_set()
        seen["exit_claimed"] = serve._exit_after_run.is_set()

    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect", "planner"], during_step=look)
    )
    result = serve.rpc_run_goal({"goal": "a long job"})

    # Deferred, not immediate: the process was still up while the run saved.
    assert seen["armed_during_run"] is False
    assert seen["exit_claimed"] is True
    assert result["stopped"] is True
    # And it really does exit, once the run has handed its state back.
    assert serve._shutdown_requested.is_set()


def test_a_termination_signal_with_no_run_exits_at_once():
    """Nothing to defer to, so the signal takes the exit itself."""
    serve._exit_the_way_the_console_would(signal.SIGTERM, None)

    assert serve._shutdown_requested.is_set()
