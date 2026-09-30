"""Tests for the online research phase's place in a run.

The phase itself is tested in `test_web_research.py`. What is under test here
is *when* it happens and what a run does when it fails, because both are load
bearing and neither is visible from inside the phase.

The ordering is the whole design. `_refuse_while_a_run_is_in_flight` refuses
every other corpus write while a run is live, because a corpus changing
underneath the Researcher manufactures an absence no seat can detect: a rebuild
half-done returns whatever fraction of itself has been re-added, which reads as
`no_relevant_knowledge` and routes the run around a gap created out from under
it. Researching before the graph starts is not a way around that rule, it is
the only ordering that obeys it — so "research finished before the first
superstep" is the property, and it is asserted directly rather than assumed
from the order of the lines in the function.
"""

from __future__ import annotations

from typing import Any

import pytest

import serve


class _RecordingGraph:
    """A graph that records what had already happened when it was reached."""

    def __init__(self) -> None:
        self.saw: list[str] = []
        self.streamed = False

    def stream(self, state, config):
        self.streamed = True
        self.saw = list(state.get("messages", []))
        yield {"architect": {**state, "messages": [*state.get("messages", []), "[Architect] Verdict: approved"]}}


@pytest.fixture
def graph(monkeypatch) -> _RecordingGraph:
    fake = _RecordingGraph()
    monkeypatch.setattr(serve, "graph", fake)
    return fake


def _research(monkeypatch, report: dict[str, Any], seen: list[str] | None = None):
    def fake(kb, goal, *args, **kwargs):
        if seen is not None:
            seen.append(goal)
        return report
    monkeypatch.setattr(serve, "research_online", fake)
    monkeypatch.setattr(serve, "_kb_for_indexing", lambda: object())


FOUND = {
    "goal": "g", "source": "duckduckgo", "note": None, "queries": ["g"],
    "considered": 5, "documents": 2, "chunks": 40, "stored": [], "failed": [],
    "errors": [], "elapsed_s": 3.2,
}


def test_research_finishes_before_the_first_superstep(monkeypatch, graph):
    """The property the corpus-write rule depends on."""
    seen: list[str] = []
    _research(monkeypatch, FOUND, seen)

    result = serve.rpc_run_goal({"goal": "build a retrieval gate", "research_web": True})

    assert seen == ["build a retrieval gate"]
    assert graph.streamed
    # The graph saw the research line already in state, which can only be true
    # if the phase ran first.
    assert any("[Research]" in message for message in graph.saw)
    assert result["web_research"] == FOUND


def test_the_feed_line_survives_into_the_payload(monkeypatch, graph):
    """`_run_progress["messages"]` is overwritten by every node update.

    So the line is seeded into state, not just pushed to progress: what the run
    was told is part of how its result should be read, and the snapshot is what
    a reloaded console recovers.
    """
    _research(monkeypatch, FOUND)

    result = serve.rpc_run_goal({"goal": "g", "research_web": True})

    line = next(m for m in result["messages"] if "[Research]" in m)
    assert "embedded 2" in line and "5 page(s)" in line


def test_a_research_failure_does_not_take_the_run_down(monkeypatch, graph):
    """A goal the web cannot answer is not a reason to refuse the corpus."""
    def boom(kb, goal, *args, **kwargs):
        raise RuntimeError("the search engine is on fire")
    monkeypatch.setattr(serve, "research_online", boom)
    monkeypatch.setattr(serve, "_kb_for_indexing", lambda: object())

    result = serve.rpc_run_goal({"goal": "g", "research_web": True})

    assert graph.streamed
    assert result["web_research"]["source"] == "error"
    assert "on fire" in result["web_research"]["note"]
    assert any("found nothing usable" in m for m in result["messages"])


def test_reading_pages_and_keeping_none_is_not_reported_as_a_failure(monkeypatch, graph):
    """Correct behaviour on a goal the web has nothing to say about.

    It must not read like a broken phase, or the operator starts ignoring the
    line that also reports real breakage.
    """
    _research(monkeypatch, {**FOUND, "documents": 0, "chunks": 0, "considered": 6})

    result = serve.rpc_run_goal({"goal": "g", "research_web": True})

    line = next(m for m in result["messages"] if "[Research]" in m)
    assert "kept none" in line
    assert "failed" not in line.lower()


def test_the_three_empty_outcomes_are_worded_apart():
    """"Switched off", "broke" and "found nothing" call for different actions."""
    off = serve._research_feed_line({"source": "disabled", "considered": 0, "documents": 0})
    broke = serve._research_feed_line(
        {"source": "error", "considered": 0, "documents": 0, "note": "HTTP 429"}
    )
    empty = serve._research_feed_line(
        {"source": "duckduckgo", "considered": 4, "documents": 0, "errors": []}
    )

    assert "switched off" in off
    assert "429" in broke
    assert "kept none" in empty
    assert len({off, broke, empty}) == 3


def test_a_stop_arriving_before_the_phase_skips_it(monkeypatch):
    """The stop is checked before the phase, which is the only place it can be.

    Exercised against the phase helper rather than through `rpc_run_goal`,
    because `RUN_CONTROL.arm` clears any previous stop by design — a run takes
    ownership of the flag — so a stop cannot be staged before a run begins. The
    real arrival is from the stop thread in the window between arming and the
    first superstep, and this is the check that window is guarded by.

    Nothing in flight is abandoned here either: the phase is bounded by its own
    budget, so the check that matters is the one before it starts.
    """
    called: list[str] = []
    _research(monkeypatch, FOUND, called)
    serve.RUN_CONTROL.arm("probe")
    serve.RUN_CONTROL.stop()

    report = serve._research_online_before_the_run("g", True)

    assert called == []
    assert report["source"] == "stopped"
    assert serve._research_feed_line(report) == "[Research] Stopped before online research began."


def test_a_stop_after_approval_does_not_deny_the_verdict(monkeypatch):
    """The run's own record must not contradict itself.

    Both exits are checked *between* supersteps, so a stop arriving while the
    Architect is ruling lands after that ruling is already in state. Measured
    live on 2026-09-09: the transcript read "[Architect] Verdict: approved"
    immediately above "[Graph] Stopped ... without an approved verdict" — in the
    one place anyone finds out how a run ended.
    """
    class _Approves:
        def stream(self, state, config):
            yield {"architect": {**state, "verdict": "approved",
                                 "messages": [*state.get("messages", []), "[Architect] Verdict: approved"]}}
            serve.RUN_CONTROL.stop()
            yield {"architect": {**state, "verdict": "approved",
                                 "messages": [*state.get("messages", []), "[Architect] Verdict: approved"]}}

    monkeypatch.setattr(serve, "graph", _Approves())
    _research(monkeypatch, FOUND)

    result = serve.rpc_run_goal({"goal": "g", "research_web": True})

    assert result["stopped"]
    assert result["verdict"] == "approved"
    stop_line = next(m for m in result["messages"] if "[Graph] Stopped" in m)
    assert "without an approved verdict" not in stop_line
    assert "already ruled approved" in stop_line


def test_the_phase_does_not_run_unless_the_run_asked_for_it(monkeypatch):
    """Off by default, and the caller is the operator -- never an agent.

    Three gates were built to decide relevance from the goal text and all three
    were graded wrong against thirteen hand-labelled pages; see
    `_research_online_before_the_run` for the numbers. The deciding information
    is the operator's intent, so the switch is theirs, exactly as
    `expect_failures` is.
    """
    called: list[str] = []
    _research(monkeypatch, FOUND, called)

    report = serve._research_online_before_the_run("g", False)

    assert called == [], "the phase must not reach the network unasked"
    assert report["source"] == "not_requested"
    assert report["documents"] == 0


def test_the_phase_runs_when_the_run_does_ask(monkeypatch):
    """The opt-in has to actually opt in, or the feature is merely deleted."""
    called: list[str] = []
    _research(monkeypatch, FOUND, called)

    report = serve._research_online_before_the_run("g", True)

    assert called, "asking for online research must reach the phase"
    assert report["source"] != "not_requested"


def test_not_requested_reads_differently_from_switched_off(monkeypatch):
    """"This run did not ask" and "this machine has it off" are different facts.

    The same reason `search_web` keeps its three empty outcomes apart: they call
    for different things from the operator and an empty count reads identically
    in all of them.
    """
    not_asked = serve._research_feed_line({"source": "not_requested", "considered": 0, "documents": 0})
    switched_off = serve._research_feed_line({"source": "disabled", "considered": 0, "documents": 0})

    assert not_asked != switched_off
    assert "not requested" in not_asked
    assert "switched off" in switched_off


def test_run_goal_does_not_research_online_by_default(monkeypatch, graph):
    """The default reaches through `rpc_run_goal`, not just the helper."""
    called: list[str] = []
    _research(monkeypatch, FOUND, called)

    serve.rpc_run_goal({"goal": "write me a document about all the local project data"})

    assert called == []


def test_run_goal_researches_online_when_the_caller_asks(monkeypatch, graph):
    called: list[str] = []
    _research(monkeypatch, FOUND, called)

    serve.rpc_run_goal({"goal": "something the web knows", "research_web": True})

    assert called


def test_a_discussion_run_never_researches_online(monkeypatch, graph):
    """"No actions" has to cover the corpus too.

    The phase writes pages under research/web/ and embeds them, which changes
    this machine and every later run's retrieval. So the box being ticked is
    not enough on a discussion run, and the two flags are resolved in
    `rpc_run_goal` rather than left to the operator to keep consistent.
    """
    called: list[str] = []
    _research(monkeypatch, FOUND, called)

    serve.rpc_run_goal({"goal": "g", "research_web": True, "discuss_only": True})

    assert called == []


def test_a_discussion_run_reaches_the_seats_as_state(monkeypatch, graph):
    """The flag has to arrive in AgentState; the Builder reads it from there."""
    _research(monkeypatch, FOUND)

    result = serve.rpc_run_goal({"goal": "g", "discuss_only": True})

    assert result["discuss_only"] is True


def test_a_pre_run_phase_that_raises_does_not_leave_the_run_claimed(monkeypatch, graph):
    """The phases ran between claiming the run and the `try` that releases it.

    One that raised left `running` True and the stop control armed for the life
    of the process: every later run was refused as already in flight, every
    upload and clear with it, and an exit waited on a `finally` that never came.
    """
    def unreadable(report):
        raise RuntimeError("the floor record is unreadable")

    monkeypatch.setattr(serve, "_calibrate_the_floor_before_the_run", unreadable)

    with pytest.raises(RuntimeError, match="unreadable"):
        serve.rpc_run_goal({"goal": "g"})

    assert serve._run_progress["running"] is False
    assert serve.RUN_CONTROL.run_id() == ""
    assert "unreadable" in (serve._load_snapshot() or {}).get("error", "")

    monkeypatch.setattr(
        serve, "_calibrate_the_floor_before_the_run",
        lambda report: {"source": "known", "model": "m"},
    )
    serve.rpc_run_goal({"goal": "g"})

    assert graph.streamed, "the next run was refused as already in flight"
