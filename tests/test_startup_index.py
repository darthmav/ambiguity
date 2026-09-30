"""The corpus is brought up to date when the console comes up, not only by a run.

Rebuilding it was a run's job alone, and a run is something the operator asks
for -- so between runs the header reported drift it had no way to fix, and
restarting the server did not clear it, because starting up only *opens* the
store. There is no Reindex button either: it was removed on the argument that a
run does this, which was right and left the case where a run is not what the
operator wants unattended.

Observed on 2026-09-18: a commit added `ollama_client.py` and put
`experimental/` into `PROJECT_INDEX_EXCLUDES`, the store was two days older
than both, and the header read `stale: 1 not indexed, 1 not in the walk` across
every restart -- correctly, permanently, and with nothing the operator could
press. What is pinned here is the phase that fixes it and the four things it
must not break: the creating door stays the only way a corpus comes into being,
the two phases never rebuild at once, a run always wins, and nothing may change
the corpus underneath either of them.

No test here loads the embedding model or indexes anything real; every one of
them stands in for `index_project_files`.
"""

from __future__ import annotations

import contextlib
import fcntl
import inspect
import threading
import time

import pytest

import serve
from langgraph_agent import graphrag_server

# The walk's mechanics, laid out at the top of a scratch tree.
pytestmark = pytest.mark.usefixtures("whole_root_walk")


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    """A server with no rebuild and no run in flight, and its own state dict.

    `_startup_index` is a module global that outlives a test, exactly as
    `_run_progress` is: a phase left `running` in one test refuses every upload
    in the next.
    """
    monkeypatch.setattr(serve, "_startup_index", {"running": False, "message": "", "report": {}})
    monkeypatch.setattr(serve, "_index_lock", threading.Lock())
    monkeypatch.setitem(serve._run_progress, "running", False)
    monkeypatch.setitem(serve._run_progress, "goal", "")


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """A machine where nobody has ever indexed."""
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_index(calls, report=None, watch=None):
    """Stand in for `index_project_files`, recording the corpus it was handed.

    `watch(kwargs)` is called while the phase is inside the rebuild, which is
    where the lock is held and where `should_stop` means anything.
    """
    def index(kb, **kwargs):
        calls.append(kb)
        if watch is not None:
            watch(kwargs)
        return dict(report or {"indexed": 2, "embedded": 2, "reused": 0,
                               "dropped": 0, "skipped": 0, "errors": [],
                               "total_chunks": 7})
    return index


# ---------------------------------------------------------------------------
# the phase itself
# ---------------------------------------------------------------------------


def test_starting_the_console_brings_the_corpus_up_to_date(nowhere, monkeypatch, capsys):
    """The failure this exists for: a stale corpus that survives a restart.

    The log line is part of it. The phase has no run feed to write into, so the
    server log is where it says what it did -- and it has to say it happened at
    startup: `_corpus_feed_line`'s own wording is "indexed before the run",
    which is a lie from here and the kind that reads as the phase not existing.
    """
    (nowhere / "notes.md").write_text("the planner interprets goals", encoding="utf-8")
    built = object()
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: built)
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    serve._index_the_project_at_startup()

    assert indexed == [built]  # through the creating door, once
    assert serve._startup_index_status() == {"running": False, "message": "", "source": "built"}
    logged = capsys.readouterr().out
    assert "indexed at startup" in logged
    assert "before the run" not in logged


def test_the_phase_is_started_by_main_and_never_at_import(monkeypatch):
    """This module used to fill `kb` from a thread started at import.

    Importing it -- which every test does -- then created a store on disk and
    loaded the embedding model whether or not anybody wanted a corpus. `main()`
    is the one entry point that means a person is running the console.
    """
    # The target and not merely the name: the comment beside the thread start
    # mentions the phase too, so a guard reading for the name alone passed with
    # the start deleted.
    assert "target=_index_the_project_at_startup" in inspect.getsource(serve.main)
    assert [t for t in threading.enumerate() if t.name == "startup-index"] == []


def test_switching_indexing_off_switches_this_off_too(nowhere, monkeypatch):
    """One switch, because there is one question: may this machine rebuild?"""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", False)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    serve._index_the_project_at_startup()

    assert indexed == []
    assert serve._startup_index_status()["source"] == ""


def test_nothing_to_index_leaves_no_corpus_behind(nowhere, monkeypatch):
    """The walk is counted before the creating door is opened.

    `get_knowledge_base` builds the store, so a machine with nothing to index
    would report `empty` from then on where the truth is `absent` -- and only
    one of those two means nothing has ever been built here.
    """
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)

    serve._index_the_project_at_startup()

    assert sorted(p.name for p in nowhere.iterdir()) == []
    assert serve._startup_index_status()["source"] == "nothing_to_index"


def test_the_feed_line_says_it_ran_at_startup_not_before_the_run():
    """The same five outcomes, and one of them would otherwise be a lie."""
    report = {"source": "built", "corpus": "absent", "indexed": 3, "total_chunks": 9,
              "elapsed_s": 1.2, "errors": []}

    assert "indexed at startup" in serve._corpus_feed_line(report, when="at startup")
    assert "indexed before the run" in serve._corpus_feed_line(report)


def test_an_index_that_failed_is_reported_rather_than_raised(nowhere, monkeypatch):
    """A corpus that could not be built makes for a worse run, not a crash."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())

    def boom(kb, **kwargs):
        raise RuntimeError("chroma is unhappy")
    monkeypatch.setattr(serve, "index_project_files", boom)

    serve._index_the_project_at_startup()  # never raises

    status = serve._startup_index_status()
    assert status["running"] is False
    assert status["source"] == "error"


def test_a_phase_that_raised_does_not_leave_the_flag_set(nowhere, monkeypatch):
    """The flag outlives the thread that set it, and nothing else clears it.

    A rebuild reported as in flight forever refuses every later upload and
    clear -- with no run to stop, since there is none, and no way back but
    restarting the console. The same hole `_finish_run`'s `finally` closes for a
    run. The phase may still raise: a dead thread with a traceback is a fault
    somebody can read.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)

    def boom(**kwargs):
        raise RuntimeError("nothing expected this")
    monkeypatch.setattr(serve, "_rebuild_the_corpus", boom)

    with pytest.raises(RuntimeError):
        serve._index_the_project_at_startup()

    assert serve._startup_index_status()["running"] is False


# ---------------------------------------------------------------------------
# a run wins, and the two phases never rebuild at once
# ---------------------------------------------------------------------------


def test_a_run_claiming_the_flag_stops_the_build(nowhere, monkeypatch):
    """A run started into a build takes over within one embedding batch.

    Making the *run* wait instead would hold `running` True with no node on the
    stack for as long as a whole build, which is exactly what a wedged run
    looks like from the console. Nothing is lost by stopping: what is already
    embedded keeps its vectors, and the run's own phase finishes the job.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    seen: list[bool] = []

    def watch(kwargs):
        should_stop = kwargs["should_stop"]
        seen.append(should_stop())
        serve._run_progress["running"] = True
        seen.append(should_stop())

    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index([], watch=watch))

    serve._index_the_project_at_startup()

    assert seen == [False, True]


def test_the_console_exiting_stops_the_build(nowhere, monkeypatch):
    """A daemon thread that ignored the exit would hold a build across it."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    seen: list[bool] = []

    def watch(kwargs):
        seen.append(kwargs["should_stop"]())
        serve._shutdown_requested.set()
        seen.append(kwargs["should_stop"]())

    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index([], watch=watch))
    try:
        serve._index_the_project_at_startup()
    finally:
        serve._shutdown_requested.clear()

    assert seen == [False, True]


def test_both_phases_rebuild_under_one_lock(nowhere, monkeypatch):
    """`index_project_files` prunes and clears before it re-adds anything, so
    two rebuilds interleaved produce neither caller's result: the second clear
    lands on the first's half-built graph and both report success."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    held: list[bool] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(
        serve, "index_project_files",
        _fake_index([], watch=lambda _: held.append(serve._index_lock.locked())),
    )

    serve._index_the_project_at_startup()
    serve._index_the_project_before_the_run()

    assert held == [True, True]  # both phases, the same lock, held over the work


# ---------------------------------------------------------------------------
# nothing may change the corpus underneath a rebuild
# ---------------------------------------------------------------------------


def test_a_clear_is_refused_while_the_corpus_is_being_rebuilt():
    """The run's hazard without the run: a clear empties a store that is being
    re-added to, and nothing raises where the seats can see it."""
    serve._startup_index["running"] = True

    with pytest.raises(ValueError, match="being rebuilt"):
        serve._refuse_while_a_run_is_in_flight("cleared")


def test_a_run_in_flight_is_the_wording_when_both_are_true():
    """A run started into a rebuild sets both flags, and only one of the two is
    something the operator can end. A refusal offering to stop a run that does
    not exist is the wrong instruction, not a vague one."""
    serve._startup_index["running"] = True
    serve._run_progress["running"] = True
    serve._run_progress["goal"] = "Do a thing"

    with pytest.raises(ValueError, match="Stop the run first") as caught:
        serve._refuse_while_a_run_is_in_flight("cleared")
    assert "Do a thing" in str(caught.value)


def test_an_idle_server_refuses_neither():
    assert serve._refuse_while_a_run_is_in_flight("cleared") is None


def test_the_stale_verdict_is_withheld_while_the_corpus_is_being_rebuilt(monkeypatch, tmp_path):
    """A poll landing inside a rebuild sees a corpus mid-repair of exactly what
    the verdict would accuse it of, so the accusation is withheld and the
    counts stay -- the same reading as mid-run, and for the same reason."""
    import networkx as nx

    from langgraph_agent.corpus_health import corpus_staleness

    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")

    class _KB:
        graph = nx.DiGraph()
        def stats(self):
            return {"total_documents": 1, "total_chunks": 1, "total_nodes": 1, "total_edges": 0}
    kb = _KB()
    kb.graph.add_node(str(tmp_path / "a.md"), type="document")
    monkeypatch.setattr(serve, "_open_kb", lambda: kb)
    monkeypatch.setattr(
        serve, "corpus_staleness",
        lambda docs: corpus_staleness(docs, str(tmp_path), use_cache=False),
    )

    idle = serve.rpc_rag_stats({})["staleness"]
    assert idle["stale"] and idle["missing_count"] == 1

    serve._startup_index["running"] = True
    during = serve.rpc_rag_stats({})["staleness"]
    assert not during["stale"]
    assert during["settling"] is True
    assert during["missing_count"] == 1  # still the truth about this instant


def test_the_header_is_told_what_the_rebuild_is_doing(monkeypatch):
    """A first build is tens of seconds during which the corpus reads `absent`
    and the verdict is withheld: without this the console shows a server that
    has decided to do nothing about either."""
    serve._startup_index.update(running=True, message="indexing: 3 of 77 file(s) checked")

    status = serve.rpc_status({})

    assert status["indexing"]["running"] is True
    assert "3 of 77" in status["indexing"]["message"]


# ---------------------------------------------------------------------------
# one rebuild per corpus, across processes as well as threads
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _another_process_is_rebuilding(root):
    """Hold the corpus's claim the way a second console would.

    On a separate descriptor, which is what makes this a fair stand-in: `flock`
    conflicts between two descriptors of one process exactly as it does between
    two processes, so the phase under test meets the same refusal from here as
    it would from a console on `PORT=8081`.
    """
    handle = (root / "knowledge.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        handle.close()


def test_a_corpus_another_process_is_rebuilding_is_left_alone(nowhere, monkeypatch):
    """Two consoles in one checkout used to be two rebuilds with nothing
    between them: `index_project_files` prunes and clears before it re-adds, so
    the second clear lands on the first's half-built graph and both report
    success. The port stops that on one port and stops nothing on another."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    with _another_process_is_rebuilding(nowhere):
        serve._index_the_project_at_startup()

    assert indexed == []  # the store was not touched at all
    assert serve._startup_index_status()["source"] == "busy_elsewhere"


def test_the_claim_is_released_when_the_rebuild_ends(nowhere, monkeypatch):
    """A claim the first phase kept would stop every rebuild after it -- in this
    process and in every other one, for as long as the console stayed up.

    Checked from both sides, because the descriptor is closed *and* would be
    collected: a second rebuild here, and a claim taken from outside afterwards.
    The second is what would catch a handle parked somewhere that outlives the
    phase, which is the shape this could plausibly regress into.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    serve._index_the_project_at_startup()
    serve._index_the_project_at_startup()

    assert len(indexed) == 2
    with _another_process_is_rebuilding(nowhere):
        pass  # takes it without waiting, or this raises


def test_the_lock_file_is_not_created_when_there_is_nothing_to_index(nowhere, monkeypatch):
    """The claim is taken after the walk, for the reason the creating door is
    opened after it: no work, no trace left on a machine that had none to do."""
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)

    serve._index_the_project_at_startup()

    assert sorted(p.name for p in nowhere.iterdir()) == []


def test_a_machine_without_flock_still_rebuilds(nowhere, monkeypatch):
    """The claim is an extra guarantee about a rare collision. Refusing to index
    because it could not be taken would turn that into a corpus nobody rebuilds
    -- the failure this whole phase exists to end."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "fcntl", None)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    serve._index_the_project_at_startup()

    assert len(indexed) == 1
    assert sorted(p.name for p in nowhere.iterdir()) == ["notes.md"]


def test_a_run_waits_for_a_foreign_rebuild_rather_than_searching_a_fraction(
    nowhere, monkeypatch
):
    """The run's phase is what makes the corpus whole before any seat searches
    it, and a corpus midway through a rebuild returns whatever fraction of
    itself has been re-added -- which reads to the Researcher as a corpus with
    nothing to say. So this one waits where the startup index does not, and says
    it is waiting rather than going quiet."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "CORPUS_LOCK_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(serve, "CORPUS_LOCK_POLL_SECONDS", 0.02)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index([]))

    with _another_process_is_rebuilding(nowhere):
        report = serve._index_the_project_before_the_run()

    assert report["source"] == "busy_elsewhere"
    assert any("waiting up to" in m for m in serve._run_progress["messages"])
    assert "left alone" in serve._corpus_feed_line(report)


def test_a_waiting_run_gets_the_corpus_when_the_other_process_finishes(nowhere, monkeypatch):
    """A wait that only ever times out is a delay, not a wait."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed: list[object] = []
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "CORPUS_LOCK_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(serve, "CORPUS_LOCK_POLL_SECONDS", 0.02)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index(indexed))

    released = threading.Event()
    holder = (nowhere / "knowledge.lock").open("a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release():
        time.sleep(0.1)
        holder.close()
        released.set()
    threading.Thread(target=release, daemon=True).start()

    report = serve._index_the_project_before_the_run()

    assert released.is_set()
    assert report["source"] in ("built", "updated", "current")
    assert len(indexed) == 1


def test_the_stop_interrupts_a_wait(nowhere, monkeypatch):
    """A blocking `flock` cannot be interrupted, which is why the wait is
    polled: the emergency stop has to reach a run parked here, and
    `RUN_BUDGET_SECONDS` cannot end it -- that is checked between supersteps and
    this phase runs before the first one."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", True)
    monkeypatch.setattr(serve, "CORPUS_LOCK_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(serve, "CORPUS_LOCK_POLL_SECONDS", 0.02)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_project_files", _fake_index([]))

    stopped = iter([False, True, True, True])
    monkeypatch.setattr(serve.RUN_CONTROL, "stopped", lambda: next(stopped, True))

    started = time.monotonic()
    with _another_process_is_rebuilding(nowhere):
        report = serve._index_the_project_before_the_run()

    assert report["source"] == "busy_elsewhere"
    assert time.monotonic() - started < 5.0  # not the 30s it was told to wait
