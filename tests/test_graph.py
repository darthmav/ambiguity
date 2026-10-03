"""Tests for the 4-Agent System.

Verifies:
- State schema matches documentation
- Architect opens the run and holds the approval gate
- Planner routes correctly based on task type
- Researcher sets proper status
- Builder reports changes and blockers
- Graph loops on the Architect's verdict and stops at the step limit
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langgraph_agent import AgentState, ResearchStatus, Verdict, create_agent_graph, initial_state
from langgraph_agent.config import StubLLM
from langgraph_agent.control import ACTIVITY, TURN_RECORD
from langgraph_agent.graph import RECURSION_LIMIT, _tracked


@pytest.fixture
def agent_graph(monkeypatch):
    """Create a fresh agent graph for testing using the deterministic StubLLM."""
    # Force every agent node to use the canned StubLLM so tests do not depend
    # on a live cloud LLM endpoint or API keys.
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    return create_agent_graph()


def test_state_schema_initialization():
    """Test that state schema matches documentation."""
    state = initial_state("Test goal")

    # Required fields per documentation
    assert state["goal"] == "Test goal"
    assert state["messages"] == []
    assert state["architecture"] == ""
    assert state["verdict"] == ""
    assert state["plan"] == ""
    assert state["research"] == ""
    assert state["builder_report"] == ""
    assert state["next_agent"] == "Researcher"
    assert state["research_status"] == ""
    assert state["blockers"] == ""
    assert state["files_changed"] == []
    assert state["unverified"] == []
    assert state["builder_cut_off"] == ""
    assert state["lint_failed"] == []
    assert state["step_count"] == 0


def test_planner_routes_to_researcher(agent_graph):
    """Test that Planner routes to Researcher for knowledge-heavy tasks."""
    state = initial_state("Research Python async best practices")

    result = agent_graph.invoke(state)

    # Planner should create a plan
    assert result["plan"] != ""
    # Should route to Researcher for research tasks
    # Note: LLM may choose Builder if task seems straightforward
    assert result["next_agent"] in ["Researcher", "Builder"]
    assert len(result["messages"]) > 0


def test_planner_routes_to_builder(agent_graph):
    """Test that Planner routes to Builder for clear tasks."""
    state = initial_state("Create a file named hello.txt with 'Hello World'")

    result = agent_graph.invoke(state)

    # Planner should create a plan
    assert result["plan"] != ""
    # Should route to Builder (task is clear, no research needed)
    # Note: LLM may still choose Researcher, so we just verify a plan exists
    assert result["plan"] != ""


def test_researcher_sets_status(agent_graph):
    """Test that Researcher sets research_status field."""
    state = initial_state("Research async error handling patterns")

    result = agent_graph.invoke(state)

    # Researcher should set research field
    assert result["research"] != ""
    # Status should be set (even if LLM simulation)
    assert "research_status" in result


def test_builder_reports_changes(agent_graph):
    """Test that Builder reports changes and files."""
    state = initial_state("Create hello.txt with 'Hello World'")

    result = agent_graph.invoke(state)

    # Builder should report something
    assert result["builder_report"] != "" or result["messages"] != []
    # Step count should increment
    assert result["step_count"] > 0


def test_full_graph_execution(agent_graph):
    """Test that the full 4-agent flow executes."""
    state = initial_state("Create a simple Python module with a greet function")

    result = agent_graph.invoke(state)

    # Verify all fields are populated
    assert result["goal"] == "Create a simple Python module with a greet function"
    assert result["plan"] != ""
    assert result["messages"] != []
    # Graph should complete (step_count incremented)
    assert result["step_count"] > 0


def test_max_steps_limit():
    """Test that graph stops at MAX_STEPS (8)."""
    # This test would require mocking to force loops
    # For now, verify the constant exists
    from langgraph_agent.graph import MAX_STEPS

    assert MAX_STEPS == 8


def test_research_status_enum():
    """Test that ResearchStatus enum has correct values."""
    assert ResearchStatus.READY_FOR_BUILDER.value == "ready_for_builder"
    assert ResearchStatus.NEED_REPLAN.value == "need_replan"
    assert ResearchStatus.NO_RELEVANT_KNOWLEDGE.value == "no_relevant_knowledge"


def test_a_fully_specified_task_still_retrieves_once(agent_graph):
    """Even a task that needs no research reads the corpus before building.

    This test asserted the opposite until 2026-09-12: a fully-specified goal
    went Planner -> Builder and `research` stayed empty, which is what the
    documentation's happy path described. What that path actually produced was
    a run that built a 77-document corpus, embedded 8 fetched web pages into
    it, and then consulted none of it -- so the assertion is inverted rather
    than deleted. The opening cycle retrieves; every later one routes as the
    Planner asks.
    """
    state = initial_state("Create a hello.txt file containing 'Hello World'")

    result = agent_graph.invoke(state)

    assert result["plan"] != ""
    assert result["research"] != ""
    assert result["research_status"] != ""
    assert result["step_count"] > 0


def test_the_forced_hop_is_only_the_first_one(agent_graph):
    """A Planner naming the Builder is obeyed once the Researcher has run.

    The override is about the corpus never being read, not about overruling the
    Planner, so it has to stop after one hop -- otherwise a revise cycle
    re-searches the same corpus with the same plan every time round the loop.
    `research_status` is the marker, and this is the test that would catch it
    being read from `research` instead: a seat that answers with nothing leaves
    findings empty and a status set.
    """
    from langgraph_agent.graph import _route_from_planner

    fresh = initial_state("Create a hello.txt file")
    fresh["next_agent"] = "Builder"
    fresh["plan"] = "1. Write the file."
    assert _route_from_planner(fresh) == "researcher"

    been_round = dict(fresh)
    been_round["research"] = ""
    been_round["research_status"] = ResearchStatus.NO_RELEVANT_KNOWLEDGE.value
    assert _route_from_planner(been_round) == "builder"


def test_a_planner_that_failed_is_not_sent_to_research(agent_graph):
    """The two placeholder plans keep their route to the Builder.

    Both are written when the Planner did not produce steps -- a stall, or a
    reply that could not be read -- and both pick the Builder deliberately: it
    is the shorter path back to the Architect, the only node that can end a run
    whose seats are already failing. Searching on a plan nobody wrote is also
    the thin retrieval that falls through to the Researcher's own model, which
    is the second slow seat such a run least needs.
    """
    from langgraph_agent.graph import _route_from_planner
    from langgraph_agent.nodes import (
        _PLANNER_NO_STEPS,
        _PLANNER_TIMED_OUT,
        NODE_DEADLINE_SECONDS,
    )

    for placeholder in (
        _PLANNER_TIMED_OUT.format(seconds=int(NODE_DEADLINE_SECONDS)),
        _PLANNER_NO_STEPS,
    ):
        state = initial_state("Create a hello.txt file")
        state["next_agent"] = "Builder"
        state["plan"] = placeholder
        assert _route_from_planner(state) == "builder"


def test_state_injection_shows_empty():
    """Test that empty fields in state injection render as (empty)."""
    from langgraph_agent.nodes import _get_state_injection

    state = initial_state("Test goal")
    injection = _get_state_injection(state)

    assert "Architecture: (empty)" in injection
    assert "Verdict: (empty)" in injection
    assert "Plan: (empty)" in injection
    assert "Research: (empty)" in injection
    assert "Builder report: (empty)" in injection
    assert "Blockers: (empty)" in injection
    assert "Files changed: (empty)" in injection


def test_architect_opens_the_run(agent_graph):
    """The Architect runs before anything is planned and sets direction."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["architecture"] != ""
    # The Architect's opening message has to precede the Planner's: it is the
    # authority that sets the constraints the plan is written against.
    roles = [m.split("]")[0].lstrip("[") for m in result["messages"]]
    assert roles[0] == "Architect"
    assert "Planner" in roles


def test_architect_gate_ends_the_run(agent_graph):
    """The run terminates on the Architect's approval, not the Builder's."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["verdict"] == Verdict.APPROVED.value
    # The Architect both opens and closes, so it appears twice and last.
    roles = [m.split("]")[0].lstrip("[") for m in result["messages"]]
    assert roles[-1] == "Architect"
    assert roles.count("Architect") == 2


def test_step_count_incremented_by_the_gate(agent_graph):
    """Every completed cycle passes the gate, so every cycle counts a step."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["step_count"] == 1


@pytest.mark.parametrize(
    "verdict,expected",
    [
        (Verdict.APPROVED.value, "__end__"),
        (Verdict.REVISE.value, "planner"),
        (Verdict.NEED_RESEARCH.value, "researcher"),
        (Verdict.PLAN.value, "planner"),
    ],
)
def test_architect_routing(verdict, expected):
    """Each verdict sends the loop where the Architect said to send it."""
    from langgraph_agent.graph import MAX_STEPS, _route_from_architect

    state = initial_state("anything")
    state["verdict"] = verdict
    assert _route_from_architect(state) == expected

    # The step ceiling outranks every verdict, including one asking to loop.
    state["step_count"] = MAX_STEPS
    assert _route_from_architect(state) == "__end__"


def test_verdict_enum():
    """Verdict values are the strings the parser and router agree on."""
    assert Verdict.PLAN.value == "plan"
    assert Verdict.APPROVED.value == "approved"
    assert Verdict.REVISE.value == "revise"
    assert Verdict.NEED_RESEARCH.value == "need_research"


def test_architect_parser_defaults_by_pass():
    """An unparseable verdict must not strand the loop.

    Before the Builder reports the only sound fallback is `plan`; after it, the
    sound fallback is `approved`, so a malformed response ends the run instead
    of looping on it until the step ceiling.
    """
    from langgraph_agent.nodes import _parse_architect_output

    assert _parse_architect_output("garbage", reviewing=False)["verdict"] == "plan"
    assert _parse_architect_output("garbage", reviewing=True)["verdict"] == "approved"

    parsed = _parse_architect_output(
        "## Architecture\nKeep it small.\n\n## Constraints\n- No new deps\n\n"
        "## Verdict\nrevise",
        reviewing=True,
    )
    assert parsed["verdict"] == "revise"
    assert "Keep it small." in parsed["architecture"]
    assert "No new deps" in parsed["architecture"]


class _ToolCallingLLM:
    """A Builder seat that calls a tool once, then reports.

    Stands in for a real tool-capable model: the first invoke asks for a
    write, the second closes the report. `bind_tools` returns self so the
    node's bind step behaves like a live seat's.
    """

    def __init__(self, path):
        self._path = path
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools, **kwargs):
        self.bound = tools
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": "written by the tool loop\n"},
                    "id": "call_1",
                }],
            )
        return AIMessage(
            content=(
                "## Changes Made\nWrote the file.\n\n"
                f"## Files Modified\n- {self._path}\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_builder_writes_through_a_tool_call(monkeypatch, tmp_path):
    """The Builder's file changes come from real tool calls, not from prose."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "written.txt"
    llm = _ToolCallingLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Write a file")
    state["plan"] = "1. Write written.txt"
    result = builder_node(state)

    # The tool actually ran: the file is on disk with the tool's content.
    assert target.read_text() == "written by the tool loop\n"
    assert result["files_changed"] == [str(target)]
    assert "filesystem_write" in result["builder_report"]
    assert result["blockers"] == ""


def test_builder_offers_only_its_own_tools(monkeypatch, tmp_path):
    """GraphRAG must never reach the Builder; retrieval is the Researcher's."""
    from langgraph_agent.nodes import builder_node

    llm = _ToolCallingLLM(tmp_path / "x.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )
    builder_node(initial_state("Write a file"))

    offered = {tool["function"]["name"] for tool in llm.bound}
    assert "filesystem_write" in offered
    assert not offered & {"search_knowledge_graph", "query_knowledge_graph"}


def test_builder_does_not_credit_unwritten_files(monkeypatch):
    """A seat that cannot call tools reports, but claims no file changes."""
    from langgraph_agent.nodes import builder_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: StubLLM()
    )
    result = builder_node(initial_state("Create app.py"))

    assert result["files_changed"] == []


def test_moving_a_seat_clears_its_recorded_failure(monkeypatch):
    """A failure belongs to the seat that produced it, not to the agent.

    The daemon is stubbed because `live` for an ollama seat is two claims at
    once: that no failure is recorded against it, and that a reachable daemon
    is holding the tag. Only the first is what moving a seat decides. Left
    unstubbed this asserted the second as well, so it passed on a laptop
    running ollama and failed everywhere else -- which is exactly what it did
    on this project's first CI run, as the one test in the suite that could
    not pass on a machine other than a developer's.
    """
    from langgraph_agent import config

    monkeypatch.setattr(config, "ollama_daemon_tags", lambda: ["kimi-k3:cloud"])

    config.set_agent_llm("architect", "anthropic", "claude-opus-5")
    config._seat_failures["architect"] = "Anthropic credit balance too low"

    # Moving the seat to a different provider retires that verdict.
    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    status = config.get_agent_status("architect")

    assert "architect" not in config._seat_failures
    assert status["live"] is True
    assert status["reason"] == ""

    config._agent_llm_overrides.pop("architect", None)
    config._seat_failures.pop("architect", None)


def test_reselecting_the_same_seat_keeps_its_failure():
    """Picking the seat you already have must not launder a real failure."""
    from langgraph_agent import config

    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    config._seat_failures["architect"] = "Provider unreachable"

    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    status = config.get_agent_status("architect")

    assert status["live"] is False
    assert status["reason"] == "Provider unreachable"
    assert status["badge"] == "FAILING"

    config._agent_llm_overrides.pop("architect", None)
    config._seat_failures.pop("architect", None)


def test_default_seats_need_no_credentials():
    """A fresh checkout must run with no provider key and no ollama.com sign-in.

    The Architect used to default to Anthropic, which made the entry node --
    and so the whole run -- depend on billable credit nobody had configured.
    A `:cloud` tag passes the provider check while doing the same through the
    daemon's ollama.com account, and README.md tells a newcomer no sign-in is
    needed.
    """
    from langgraph_agent.config import AGENTS, DEFAULT_SEATS, is_local_ollama_model

    assert set(DEFAULT_SEATS) == set(AGENTS)
    assert all(seat["provider"] == "ollama" for seat in DEFAULT_SEATS.values())
    assert all(is_local_ollama_model(seat["model"]) for seat in DEFAULT_SEATS.values())


class _WritesFileLLM(_ToolCallingLLM):
    """Writes a caller-supplied file body, then reports success."""

    def __init__(self, path, body):
        super().__init__(path)
        self._body = body

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": self._body},
                    "id": "call_1",
                }],
            )
        return AIMessage(
            content=(
                "## Changes Made\nWrote it.\n\n"
                f"## Files Modified\n- {self._path}\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_builder_runs_the_python_it_writes(monkeypatch, tmp_path):
    """A written module that raises must come back as a blocker, not success.

    This is the failure that shipped: a file written, reported complete, and
    approved -- which raised an AssertionError the first time it was run.
    """
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "broken.py"
    llm = _WritesFileLLM(target, "assert False, 'boom'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a module"))

    # The write happened, and is still reported as a change...
    assert result["files_changed"] == [str(target)]
    # ...but the run failed, so the Builder does not get to claim completion.
    assert "do not run" in result["blockers"]
    assert "boom" in result["blockers"] or "AssertionError" in result["blockers"]
    assert "FAILED" in result["builder_report"]
    assert "do not run" in result["messages"][-1]
    assert "Implementation complete" not in result["messages"][-1]


def test_builder_reports_clean_when_the_file_runs(monkeypatch, tmp_path):
    """A file that executes cleanly verifies, and sets no blocker."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "fine.py"
    llm = _WritesFileLLM(target, "print('ok')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a module"))

    assert result["blockers"] == ""
    assert "ran clean" in result["builder_report"]
    assert "Implementation complete" in result["messages"][-1]


class _NeverFinishesLLM(_ToolCallingLLM):
    """A Builder seat that asks for another read on every turn.

    It never writes the closing report, which is the shape that runs a pass
    into `MAX_BUILDER_TOOL_TURNS`.
    """

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "filesystem_read",
                "args": {"path": str(self._path)},
                "id": f"call_{self.calls}",
            }],
        )


def test_a_builder_out_of_turns_does_not_claim_completion(monkeypatch, tmp_path):
    """The turn cap is a Builder that did not finish, and the feed must say so.

    The blocker always did. The feed line beside it read "Implementation
    complete", and on 2026-09-12 the Architect approved a run on that line.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "notes.txt"
    target.write_text("read me\n")
    llm = _NeverFinishesLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )
    monkeypatch.setattr("langgraph_agent.nodes.MAX_BUILDER_TOOL_TURNS", 2)

    result = builder_node(initial_state("Read the notes"))

    assert llm.calls == 2
    assert "2 tool turns without finishing" in result["blockers"]
    assert "Implementation complete" not in result["messages"][-1]
    assert "Stopped after 2 tool turns" in result["messages"][-1]
    assert result["builder_cut_off"] == "turn_cap"
    # Not "No report produced." above a report listing every call it made.
    assert "used all 2 tool turns" in result["builder_report"]


def test_a_pass_that_finishes_clears_the_cut_off(monkeypatch, tmp_path):
    """The record is per pass: finishing the work lifts the block."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "fine.py"
    llm = _WritesFileLLM(target, "print('ok')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )
    state = initial_state("Write a module")
    state["builder_cut_off"] = "turn_cap"

    result = builder_node(state)

    assert result["builder_cut_off"] == ""


def test_builder_only_executes_runnable_files(monkeypatch, tmp_path):
    """Markdown has nothing to run; the verification pass must skip it."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "notes.md"
    llm = _WritesFileLLM(target, "# just prose\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write notes"))

    assert result["files_changed"] == [str(target)]
    assert result["blockers"] == ""
    assert "Verification" not in result["builder_report"]


class _ApprovingLLM:
    """An Architect seat that always rules approved."""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="## Architecture\nfine\n\n## Verdict\napproved\n"
        )


def test_failed_verification_blocks_approval(monkeypatch):
    """An Architect that approves work which does not run is overruled."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a module")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = ["broken.py"]

    result = architect_node(state)

    assert result["verdict"] == Verdict.REVISE.value
    assert "approval blocked" in result["messages"][-1]
    # revise routes back to the Planner rather than ending the run.
    from langgraph_agent.graph import _route_from_architect

    assert _route_from_architect(result) == "planner"


def test_approval_stands_once_verification_passes(monkeypatch):
    """With nothing failing, the Architect's approval is left alone."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a module")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = []

    result = architect_node(state)

    assert result["verdict"] == Verdict.APPROVED.value
    assert "blocked" not in result["messages"][-1]


def test_step_ceiling_still_ends_a_permanently_failing_run():
    """The block must not create a run that cannot end."""
    from langgraph_agent.graph import MAX_STEPS, _route_from_architect

    state = initial_state("anything")
    state["verdict"] = Verdict.REVISE.value
    state["failed_verification"] = ["broken.py"]
    state["step_count"] = MAX_STEPS

    assert _route_from_architect(state) == "__end__"


def _gate_with(monkeypatch, **fields):
    """Run the approval gate over an Architect that always approves."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )
    state = initial_state("Write a module")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state.update(fields)
    return architect_node(state)


def test_an_unrun_file_blocks_approval_even_under_expect_failures(monkeypatch):
    """The opt-out is for a file meant to fail, never for one nobody ran.

    The gate used to clear the whole of failed_verification under the opt-out,
    so a file the deadline left unrun was approved along with the fixture.
    """
    result = _gate_with(
        monkeypatch,
        failed_verification=["fixture.py", "unrun.py"],
        unverified=["unrun.py"],
        expect_failures=True,
    )

    assert result["verdict"] == Verdict.REVISE.value
    assert "1 file(s) never executed" in result["messages"][-1]
    assert "do not run" not in result["messages"][-1]


def test_a_builder_pass_cut_off_before_finishing_blocks_approval(monkeypatch):
    """A Builder that ran out of turns did not finish, whatever was approved."""
    result = _gate_with(monkeypatch, builder_cut_off="turn_cap")

    assert result["verdict"] == Verdict.REVISE.value
    assert "ran out of tool turns" in result["messages"][-1]


def test_a_pull_request_whose_checks_fail_blocks_approval(monkeypatch):
    """CI is the same evidence the proof is, run where the change will land."""
    result = _gate_with(
        monkeypatch, dwell={"status": "checks_failed", "checks_failed": ["CI / test (3.12)"]}
    )

    assert result["verdict"] == Verdict.REVISE.value
    assert "checks fail (CI / test (3.12))" in result["messages"][-1]


def test_a_pull_request_still_waiting_on_its_checks_does_not_block(monkeypatch):
    """Pending is the console's to finish after the run, not another cycle's."""
    result = _gate_with(monkeypatch, dwell={"status": "pending", "pending": "waiting on test"})

    assert result["verdict"] == Verdict.APPROVED.value


def test_an_unknown_cut_off_still_blocks(monkeypatch):
    """A reason nobody has wording for is still a pass that did not finish."""
    result = _gate_with(monkeypatch, builder_cut_off="something_new")

    assert result["verdict"] == Verdict.REVISE.value
    assert "something_new" in result["messages"][-1]


class _NoToolsLLM:
    """A real model the daemon says cannot call tools: it can only describe work."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        raise AttributeError("bind_tools: dolphin:latest cannot call tools")

    def invoke(self, messages: Any) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content="## Changes Made\nWrote main.py.\n\n## Files Modified\n- main.py")


def test_a_builder_whose_model_cannot_call_tools_blocks_approval(monkeypatch, tmp_path):
    """It wrote nothing, set no blocker, and said "Implementation complete",
    which the Architect was then free to approve."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: _NoToolsLLM()
    )

    result = builder_node(initial_state("Write a module"))

    assert result["builder_cut_off"] == "no_tools"
    assert "dolphin:latest cannot call tools" in result["blockers"]
    assert "Implementation complete" not in result["messages"][-1]
    assert "Could not act" in result["messages"][-1]
    assert result["files_changed"] == []

    gated = _gate_with(monkeypatch, builder_cut_off=result["builder_cut_off"])
    assert gated["verdict"] == Verdict.REVISE.value
    assert "cannot call tools" in gated["messages"][-1]


def test_a_stubbed_builder_is_not_a_model_without_tools(monkeypatch, tmp_path):
    """A keyless seat answers in canned text by design; that is not a fault."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: StubLLM()
    )

    result = builder_node(initial_state("Write a module"))

    assert result["builder_cut_off"] == ""


class _WritesNothingLLM:
    """A Builder pass that calls no tools and reports success anyway."""

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=(
                "## Changes Made\nNothing to do.\n\n"
                "## Files Modified\n\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_a_pass_that_writes_nothing_cannot_clear_a_failure(monkeypatch, tmp_path):
    """A broken file stays broken until it runs, not until it is ignored.

    The Builder could otherwise clear a failed verification by doing nothing on
    the next pass: the file stayed on disk, the failure list came back empty
    and the Architect approved.
    """
    from langgraph_agent.nodes import builder_node

    broken = tmp_path / "broken.py"
    broken.write_text("assert False, 'still broken'\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Fix it")
    state["plan"] = "1. Fix the module"
    state["failed_verification"] = [str(broken)]

    result = builder_node(state)

    assert result["files_changed"] == []            # nothing written this pass
    assert result["failed_verification"] == [str(broken)]  # still failing
    assert "do not run" in result["blockers"]


def test_a_later_pass_keeps_what_an_earlier_one_wrote(monkeypatch, tmp_path):
    """The run's record of what it produced spans passes, not just the last one.

    `files_changed` was rewritten from scratch on every Builder pass, so a run
    that wrote a file on one cycle and nothing on the next ended reporting it
    had changed nothing -- while the file sat on disk. The Architect rules on
    that record, so a build with a file to its name was approved as having
    produced none: the same false account as claiming a file that was never
    written, pointing the other way.
    """
    from langgraph_agent.nodes import builder_node

    earlier = tmp_path / "earlier.py"
    earlier.write_text("print('fine')\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Carry on")
    state["plan"] = "1. Nothing left to do"
    state["files_changed"] = [str(earlier)]

    result = builder_node(state)

    assert result["files_changed"] == [str(earlier)]
    # The feed line still describes this pass, which wrote nothing, so it has
    # to name the running total or it reads as though the run lost the file.
    assert "changed so far this run" in result["messages"][-1]


def test_a_new_file_adds_to_the_record_rather_than_replacing_it(monkeypatch, tmp_path):
    """Two passes that each write a file end with both on the record."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    earlier = tmp_path / "earlier.py"
    earlier.write_text("print('fine')\n")
    target = tmp_path / "later.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesFileLLM(target, "print('ok')\n"),
    )

    state = initial_state("Add one more")
    state["plan"] = "1. Write the second file"
    state["files_changed"] = [str(earlier)]

    result = builder_node(state)

    assert result["files_changed"] == [str(earlier), str(target)]


class _NoToolLLM:
    """A Builder pass that calls no tools and names a path it wrote earlier.

    Used for the retraction cases: the file is gone from disk, this pass did
    not write anything, and the report still mentions it.
    """

    def __init__(self, named: str = "") -> None:
        self._named = named

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=(
                "## Changes Made\nRemoved the scratch file.\n\n"
                f"## Files Modified\n- {self._named}\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_a_file_deleted_since_it_was_written_leaves_the_record(monkeypatch, tmp_path):
    """The run of 2026-09-11 ended naming three paths that did not exist.

    It wrote `gen_overview.py` and two scratch scripts under `/tmp` on one
    pass, deleted them with `rm` on a later one, and nothing retracted them --
    so the console's "changed this machine" notice, a safety notice about files
    on disk, named three files nobody could find. That is the mirror of
    "described but not written", and it was the unguarded direction.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "scratch.py"          # written by an earlier pass, since deleted
    kept = tmp_path / "kept.py"             # written by an earlier pass, still there
    kept.write_text("print('still here')\n")
    target = tmp_path / "new.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesFileLLM(target, "print('ok')\n"),
    )

    state = initial_state("Carry on")
    state["plan"] = "1. Write the new file"
    state["files_changed"] = [str(gone), str(kept)]

    result = builder_node(state)

    assert str(gone) not in result["files_changed"]
    assert result["files_changed"] == [str(kept), str(target)]


def test_a_dropped_path_is_named_rather_than_vanishing(monkeypatch, tmp_path):
    """A silent retraction is indistinguishable from an entry never made.

    The Architect rules on this report, so the removal has to be legible there
    -- the same reason `_research_snippet` announces a cut instead of trimming
    quietly.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "scratch.py"
    target = tmp_path / "new.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesFileLLM(target, "print('ok')\n"),
    )

    state = initial_state("Carry on")
    state["plan"] = "1. Write the new file"
    state["files_changed"] = [str(gone)]

    result = builder_node(state)

    assert "no longer on disk" in result["builder_report"]
    assert str(gone) in result["builder_report"]


def test_a_run_whose_every_file_was_deleted_reports_none(monkeypatch, tmp_path):
    """Empty is the accurate account here, not the false one.

    CLAUDE.md warns that an empty `files_changed` let a build with a file to
    its name be approved as having produced none. This is the other case: the
    run has no file to its name, because everything it wrote is gone, and
    saying so is correct. The report still names what went.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "scratch.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _NoToolLLM(str(gone)),
    )

    state = initial_state("Tidy up")
    state["plan"] = "1. Remove the scratch file"
    state["files_changed"] = [str(gone)]

    result = builder_node(state)

    assert result["files_changed"] == []
    assert str(gone) in result["builder_report"]


def test_a_file_written_then_removed_is_not_called_a_lie(monkeypatch, tmp_path):
    """The retraction must not manufacture the accusation it mirrors.

    A path this pass wrote through a real tool call and something then removed
    still came from a successful write, so naming it under `## Files Modified`
    is not a claim about work that never happened. `written` is computed off
    the whole record, before the retraction, precisely so this stays true --
    CLAUDE.md calls a false version of this the report's harshest claim.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "scratch.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _NoToolLLM(str(gone)),
    )

    state = initial_state("Tidy up")
    state["plan"] = "1. Remove the scratch file"
    state["files_changed"] = [str(gone)]

    result = builder_node(state)

    assert "Described but not written" not in result["builder_report"]


def test_a_rewritten_file_is_recorded_once(monkeypatch, tmp_path):
    """A path an earlier pass wrote and this one rewrote is not listed twice."""
    # The Builder writes inside the project root (`_resolve_write_path`), so
    # the root moves to the tmp dir rather than the write escaping it. Without
    # this the write is refused, the file never lands, and the assertion below
    # passes on the seeded path alone -- exercising nothing.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "again.py"

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesFileLLM(target, "print('ok')\n"),
    )

    state = initial_state("Rewrite it")
    state["plan"] = "1. Write it again"
    state["files_changed"] = [str(target)]

    result = builder_node(state)

    assert result["files_changed"] == [str(target)]


def test_a_deleted_file_clears_its_failure(monkeypatch, tmp_path):
    """Deleting the broken file is a fix, not a permanent failure.

    A carried path that no longer exists used to be re-run anyway, and
    `python <missing path>` fails forever: failed_verification never emptied,
    the gate rewrote every `approved` to `revise`, and the run could only end
    at the step ceiling.
    """
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "_tmp_scratch.py"  # never created

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Drop the scratch script")
    state["plan"] = "1. Remove it"
    state["failed_verification"] = [str(gone)]

    result = builder_node(state)

    assert result["failed_verification"] == []
    assert result["blockers"] == ""
    assert str(gone) not in result["builder_report"]


def test_a_fixed_file_clears_its_failure(monkeypatch, tmp_path):
    """Once the file actually runs, the failure is retired."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "broken.py"
    target.write_text("assert False\n")

    llm = _WritesFileLLM(target, "print('fixed')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Fix it")
    state["plan"] = "1. Fix the module"
    state["failed_verification"] = [str(target)]

    result = builder_node(state)

    assert result["failed_verification"] == []
    assert result["blockers"] == ""


def test_expect_failures_lets_the_gate_approve(monkeypatch):
    """With the opt-out set, a failing file no longer overrules the Architect."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a deliberate fixture")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = ["fixture.py"]
    state["expect_failures"] = True

    result = architect_node(state)

    assert result["verdict"] == Verdict.APPROVED.value
    assert "blocked" not in result["messages"][-1]
    # The failure is not hidden -- it stays in state for the report.
    assert result["failed_verification"] == ["fixture.py"]


def test_expect_failures_still_runs_and_reports_the_file(monkeypatch, tmp_path):
    """The opt-out suppresses the block, not the check."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "fixture.py"
    llm = _WritesFileLLM(target, "assert False, 'by design'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Write a deliberate fixture")
    state["expect_failures"] = True
    result = builder_node(state)

    # Executed and reported...
    assert "FAILED" in result["builder_report"]
    assert result["failed_verification"] == [str(target)]
    assert "expected for this run" in result["messages"][-1]
    # ...but not treated as a blocker.
    assert result["blockers"] == ""


def test_a_package_module_is_proven_by_importing_it(monkeypatch, tmp_path):
    """`python pkg/mod.py` cannot import pkg, so the module is imported instead.

    A module that imports its own package absolutely dies with
    ModuleNotFoundError under direct execution however correct it is. It used
    to be skipped, which proved nothing: a run whose only products were two
    package modules was approved with neither ever executed.
    """
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "helper.py").write_text("VALUE = 42\n")
    target = pkg / "mod.py"

    llm = _WritesFileLLM(target, "from pkg.helper import VALUE\n\nassert VALUE == 42\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a package module"))

    assert result["files_changed"] == [str(target)]
    assert result["failed_verification"] == []
    assert result["blockers"] == ""
    assert "imported clean" in result["builder_report"]
    assert "imported as `pkg.mod`" in result["builder_report"]


def test_a_broken_package_module_fails_its_import(monkeypatch, tmp_path):
    """Importing is a real check: a module that cannot import is a failure."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    target = pkg / "mod.py"

    llm = _WritesFileLLM(target, "from pkg.missing import nothing\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a package module"))

    assert result["failed_verification"] == [str(target)]
    assert "pkg.missing" in result["blockers"]
    assert "FAILED" in result["builder_report"]


def test_a_root_level_script_is_still_executed(monkeypatch, tmp_path):
    """The skip is for package modules only; a loose script still has to run."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "verify_it.py"  # no __init__.py beside it
    llm = _WritesFileLLM(target, "assert False, 'boom'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a verification script"))

    assert result["failed_verification"] == [str(target)]
    assert "do not run" in result["blockers"]


def test_an_imported_module_does_not_hide_a_failing_script(tmp_path):
    """A package module that imports clean beside a failing script: both reported."""
    from langgraph_agent.nodes import _verify_written_files

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    module = pkg / "mod.py"
    module.write_text("import pkg\n")
    script = tmp_path / "verify_it.py"
    script.write_text("assert False\n")

    log: list[str] = []
    results = _verify_written_files([str(module), str(script)], log)
    statuses = {path: status for path, status, _ in results}

    assert statuses[str(module)] == "imported"
    assert statuses[str(script)] == "failed"
    assert f"verify({module}) -> imported" in log


def test_import_target_names_the_package_and_its_root(tmp_path):
    """The walk stops above the outermost package; an __init__ is the package."""
    from langgraph_agent.nodes import _import_target

    inner = tmp_path / "src" / "outer" / "inner"
    inner.mkdir(parents=True)
    (tmp_path / "src" / "outer" / "__init__.py").write_text("")
    (inner / "__init__.py").write_text("")

    root = str((tmp_path / "src").resolve())
    assert _import_target(str(inner / "mod.py")) == (root, "outer.inner.mod")
    assert _import_target(str(inner / "__init__.py")) == (root, "outer.inner")
    assert _import_target(str(tmp_path / "loose.py")) is None


def test_a_package_directory_that_cannot_be_imported_runs_as_a_script(tmp_path):
    """`my-pkg` is not a module name, so its files are executed like any other."""
    from langgraph_agent.nodes import _import_target, _verify_written_files

    odd = tmp_path / "my-pkg"
    odd.mkdir()
    (odd / "__init__.py").write_text("")
    tool = odd / "tool.py"
    tool.write_text("print('standalone')\n")

    assert _import_target(str(tool)) is None
    assert [status for _, status, _ in _verify_written_files([str(tool)], [])] == ["ok"]


def test_a_path_with_a_space_is_still_executed(tmp_path):
    """There is no shell to split on, so the path is quoted, not word-split."""
    from langgraph_agent.nodes import _verify_written_files

    spaced = tmp_path / "my script.py"
    spaced.write_text("print('ran')\n")

    assert [status for _, status, _ in _verify_written_files([str(spaced)], [])] == ["ok"]


def test_expect_failures_defaults_off():
    """The strict behaviour is what you get without asking for otherwise."""
    assert initial_state("anything")["expect_failures"] is False


@pytest.mark.parametrize(
    "text",
    [
        "none",
        "None.",
        "n/a",
        "N/A - all good",
        "no blockers",
        "No blockers.",
        "nothing",
        "none - see the report",
        "none — Note: this file intentionally raises AssertionError by design",
        "",
    ],
)
def test_a_blockers_section_saying_none_is_not_a_blocker(text):
    """"none" followed by commentary still means there are no blockers."""
    from langgraph_agent.nodes import _clean_blockers

    assert _clean_blockers(text) == ""


@pytest.mark.parametrize(
    "text",
    [
        "none of the tests pass",
        "nothing works after the refactor",
        "Missing API key for the provider",
        "no blockers were resolved; the import still fails",
    ],
)
def test_a_real_blocker_starting_with_none_survives(text):
    """The match must not swallow a sentence that only begins with the word.

    Dropping "none of the tests pass" would be exactly the silent success the
    verification pass exists to prevent.
    """
    from langgraph_agent.nodes import _clean_blockers

    assert _clean_blockers(text) == text


# --- Timeouts -------------------------------------------------------------
#
# A stalled seat used to hang a run outright. RUN_BUDGET_SECONDS is checked
# between graph supersteps, so a node that never returns never reaches the
# check, and no provider client carried a request timeout of its own.


def test_a_hung_researcher_gives_up_and_routes_to_builder(monkeypatch):
    """The node returns on its deadline instead of blocking the run forever."""
    import threading

    from langgraph_agent import nodes

    entered = threading.Event()
    release = threading.Event()

    def _hang(state):
        entered.set()
        release.wait(30)  # released in the finally below, never on its own
        return "should never be used", "ready_for_builder"

    monkeypatch.setattr(nodes, "_gather_research", _hang)
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", 0.5)

    state = initial_state("Research something the seat will not answer")
    state["plan"] = "1. Look it up"

    try:
        result = nodes.researcher_node(state)
    finally:
        release.set()

    assert entered.is_set()                                   # it really ran
    assert result["research_status"] == "no_relevant_knowledge"
    assert result["next_agent"] == "Builder"                  # run continues
    assert "did not finish" in result["research"]
    assert any("No response within" in m for m in result["messages"])


def test_a_late_researcher_cannot_write_into_state(monkeypatch):
    """The abandoned worker must not land its result after the node returned.

    It cannot be cancelled -- Python cannot interrupt a thread blocked on a
    socket -- so the only protection is that `_gather_research` returns its
    findings rather than writing them, and the timed-out caller drops them.
    """
    import threading

    from langgraph_agent import nodes

    release = threading.Event()
    finished = threading.Event()

    def _slow(state):
        release.wait(30)
        finished.set()
        return "LATE FINDINGS", "ready_for_builder"

    monkeypatch.setattr(nodes, "_gather_research", _slow)
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", 0.3)

    state = initial_state("Research something slow")
    result = nodes.researcher_node(state)

    release.set()
    assert finished.wait(10)          # the worker did complete, just too late
    assert "LATE FINDINGS" not in result["research"]
    assert "LATE FINDINGS" not in state["research"]


def test_a_failing_researcher_still_raises(monkeypatch):
    """A seat that errors is not a timeout and must not be swallowed.

    `_SeatLLM` records the reason from the exception; converting it into a
    quiet "no research" would take the failing seat off the console.
    """
    from langgraph_agent import nodes

    def _boom(state):
        raise RuntimeError("credit balance too low")

    monkeypatch.setattr(nodes, "_gather_research", _boom)

    with pytest.raises(RuntimeError, match="credit balance"):
        nodes.researcher_node(initial_state("Research something"))


class _SilentSeat:
    """A seat whose model answers with nothing at all."""

    def __init__(self, content=""):
        self.content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self.content)


@pytest.mark.parametrize(
    "content",
    [
        "",                                  # nothing at all
        "   \n\n  ",                         # whitespace
        "## Key Findings\n\n## Status\nready_for_builder",   # headings, no content
        [],                                  # empty content blocks
    ],
)
def test_a_silent_researcher_is_not_reported_as_research(monkeypatch, content):
    """An empty answer must not route as `ready_for_builder`.

    The parsed status defaults to `ready_for_builder`, so silence was being
    announced as "Research complete" while `research` reached the Builder
    empty. The Builder then reports an empty store, the gate rules
    `need_research`, and the run goes back round to the same silent seat --
    a loop that burns a step per cycle up to the ceiling.
    """
    from langgraph_agent import nodes

    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent: _SilentSeat(content))
    monkeypatch.setattr(
        nodes, "_call_tool", lambda *a, **k: {"results": []}
    )

    state = initial_state("Research something the seat will not answer")
    state["plan"] = "1. Look it up"

    result = nodes.researcher_node(state)

    assert result["research_status"] == "no_relevant_knowledge"
    assert result["next_agent"] == "Builder"          # not back to the Planner
    assert "returned no findings" in result["research"]
    assert not any("Research complete" in m for m in result["messages"])
    assert any("check the Researcher's model" in m for m in result["messages"])


def test_real_findings_still_route_as_research(monkeypatch):
    """The guard must not swallow a seat that actually answered."""
    from langgraph_agent import nodes

    answer = (
        "## Key Findings\nThe Laplacian is D - A.\n\n"
        "## Relevant Context\nSpectral graph theory.\n\n"
        "## Recommendations for Builder\nStart from the Laplacian.\n\n"
        "## Status\nready_for_builder"
    )
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent: _SilentSeat(answer))
    monkeypatch.setattr(
        nodes, "_call_tool", lambda *a, **k: {"results": []}
    )

    state = initial_state("Research spectral graph theory")
    state["plan"] = "1. Look it up"

    result = nodes.researcher_node(state)

    assert result["research_status"] == "ready_for_builder"
    assert "Laplacian" in result["research"]
    assert any("Research complete" in m for m in result["messages"])


def test_every_provider_client_carries_a_request_timeout(monkeypatch):
    """No seat may be built without a deadline, whatever the provider.

    Each spells it differently, so a single missed keyword silently restores
    the unbounded wait for that one provider only.
    """
    from langgraph_agent import config

    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    ollama = config.get_llm(provider="ollama", model="kimi-k3:cloud")
    assert ollama._client._client.timeout.read == 42.0

    anthropic = config.get_llm(provider="anthropic", model="claude-sonnet-5")
    assert anthropic.default_request_timeout == 42.0


def test_a_timed_out_seat_reads_as_a_failure_not_a_blank(monkeypatch):
    """httpx raises ReadTimeout with an empty message.

    Left alone that produced an empty chip, which reads as a healthy seat.
    """
    import httpx

    from langgraph_agent.config import _failure_reason

    assert "No response within" in _failure_reason(httpx.ReadTimeout(""))


# --- The Builder's deadline -----------------------------------------------
#
# Unlike the Researcher, this node has side effects: abandoning a worker
# mid-write would keep writing into the project after the node returned. So the
# model's call is bounded, the tool calls underneath it always finish, and the
# budget is re-checked between turns.


class _SlowToolCallingLLM(_ToolCallingLLM):
    """A Builder seat whose *model* call hangs on the turn after its write."""

    def __init__(self, path, release):
        super().__init__(path)
        self._release = release

    def invoke(self, messages):
        if self.calls >= 1:
            self._release.wait(30)
        return super().invoke(messages)


def test_a_hung_builder_keeps_the_files_it_already_wrote(monkeypatch, tmp_path):
    """The deadline ends the turn; it does not discard completed work."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    import threading

    from langgraph_agent import nodes

    target = tmp_path / "written.py"
    release = threading.Event()
    llm = _SlowToolCallingLLM(target, release)
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: llm)
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", 0.5)

    state = initial_state("Write a file")
    state["plan"] = "1. Write it"

    try:
        result = nodes.builder_node(state)
    finally:
        release.set()

    # The write completed before the hang, so it counts.
    assert str(target) in result["files_changed"]
    assert target.exists()
    assert "deadline" in result["blockers"]
    assert not any("Implementation complete" in m for m in result["messages"])


def test_a_tool_call_is_never_abandoned_midway(monkeypatch, tmp_path):
    """A slow tool runs to completion even past the deadline.

    Only the model's own call may be abandoned. A worker cut off inside
    `filesystem_write` would go on writing into the project after the node had
    returned, which is worse than the hang the deadline exists to stop.
    """
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    import time

    from langgraph_agent import nodes

    target = tmp_path / "slow.py"
    started = []
    finished = []

    real_call = nodes._call_tool

    def _slow_call(name, args):
        if name == "filesystem_write":
            started.append(name)
            time.sleep(1.2)          # outlives the deadline set below
            result = real_call(name, args)
            finished.append(name)
            return result
        return real_call(name, args)

    monkeypatch.setattr(nodes, "_call_tool", _slow_call)
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ToolCallingLLM(target)
    )
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", 0.5)

    state = initial_state("Write a file slowly")
    result = nodes.builder_node(state)

    assert started and finished          # it ran all the way through
    assert str(target) in result["files_changed"]


def test_an_unverified_file_does_not_count_as_passing(tmp_path):
    """A file the deadline stopped us running is unproven, not working."""
    from langgraph_agent import nodes

    good = tmp_path / "fine.py"
    good.write_text("print('fine')\n")

    log = []
    spent = nodes._Deadline(0.0)         # already expired
    results = nodes._verify_written_files([str(good)], log, spent)

    assert results == [
        (str(good), "unverified", nodes.VERIFY_DEADLINE_SKIP_REASON)
    ]
    assert "not run (deadline)" in log[0]


def test_a_sliver_of_time_left_does_not_fail_a_working_file(tmp_path):
    """A slice too small to run in must read as unrun, not as broken.

    `min(60, remaining)` rounded down to a zero-second timeout, so a perfectly
    good file came back FAILED with "timed out after 0 seconds" and set the
    blocker that says the file does not run.
    """
    from langgraph_agent import nodes

    good = tmp_path / "fine.py"
    good.write_text("print('fine')\n")

    log = []
    sliver = nodes._Deadline(nodes.MIN_VERIFY_SLICE_SECONDS / 2)
    statuses = [status for _, status, _ in nodes._verify_written_files(
        [str(good)], log, sliver
    )]

    assert statuses == ["unverified"]


def test_verification_runs_headless(tmp_path, monkeypatch):
    """A script that opens a window must not wait for a human to close it.

    Every generated plotting example ends in `plt.show()`. Run with a display
    inherited from the operator's session it blocks forever, so a correct file
    burned a full VERIFY_TIMEOUT_SECONDS and came back FAILED -- and the first
    one to do it ate most of the Builder's budget, leaving the rest of the
    pass unverified.
    """
    from langgraph_agent import nodes

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os\n"
        "assert os.environ.get('MPLBACKEND') == 'Agg'\n"
        "assert 'DISPLAY' not in os.environ\n"
        "assert 'WAYLAND_DISPLAY' not in os.environ\n"
    )

    results = nodes._verify_written_files([str(probe)], [])

    assert [status for _, status, _ in results] == ["ok"]


def test_a_file_reading_stdin_fails_instead_of_hanging(tmp_path):
    """No human is at the keyboard, so a read must get EOF, not the timeout."""
    from langgraph_agent import nodes

    asks = tmp_path / "asks.py"
    asks.write_text("input('name? ')\n")

    results = nodes._verify_written_files([str(asks)], [], nodes._Deadline(30))

    (_, status, detail) = results[0]
    assert status == "failed"
    assert "EOFError" in detail        # it failed on its own, not on the clock


def test_a_timeout_reports_what_the_file_printed(tmp_path):
    """A bare timeout says a file hung but not where.

    The Builder read one as a missing dependency, installed a package that was
    already present, and spent a second full timeout on an identical retry.
    The tail of the output is what distinguishes a file that blocked on line
    one from a file that did all its work and then waited at the end.
    """
    from langgraph_agent import nodes

    stalls = tmp_path / "stalls.py"
    stalls.write_text(
        "import sys, time\n"
        "print('the real work happened')\n"
        "sys.stdout.flush()\n"
        "time.sleep(300)\n"
    )

    results = nodes._verify_written_files([str(stalls)], [], nodes._Deadline(4))

    (_, status, detail) = results[0]
    assert status == "failed"
    assert "timed out" in detail
    assert "the real work happened" in detail


def test_a_timeout_with_no_output_reports_just_the_timeout(tmp_path):
    """Nothing printed means nothing to add -- and no empty heading."""
    from langgraph_agent import nodes

    silent = tmp_path / "silent.py"
    silent.write_text("import time\ntime.sleep(300)\n")

    results = nodes._verify_written_files([str(silent)], [], nodes._Deadline(4))

    (_, status, detail) = results[0]
    assert status == "failed"
    assert "timed out" in detail
    assert nodes._TIMEOUT_OUTPUT_HEADING.strip() not in detail


def test_a_timeout_detail_stays_within_the_report_budget():
    """The tail is trimmed to fit, not allowed to bury the rest of the report."""
    from langgraph_agent import nodes

    detail = nodes._timeout_detail({
        "error": "Command 'python big.py' timed out after 60 seconds",
        "stdout": "x" * 10_000,
        "timed_out": True,
    })

    assert len(detail) <= nodes.MAX_VERIFY_DETAIL_CHARS
    assert "timed out after 60 seconds" in detail
    assert detail.endswith("x")


def _builder_with_unverified_file(monkeypatch, tmp_path, **state_overrides):
    """Run builder_node with the verification pass reporting one unrun file.

    The result is forced rather than produced by racing the real clock: making
    the tool loop finish while leaving verification under its floor takes
    timings this suite cannot hold steady. What is under test here is the
    wiring around that result, not the clock, which
    `test_an_unverified_file_does_not_count_as_passing` covers directly.
    """
    from langgraph_agent import nodes

    target = tmp_path / "written.py"
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ToolCallingLLM(target)
    )
    monkeypatch.setattr(
        nodes,
        "_verify_written_files",
        lambda paths, log, deadline=None, cwd="": [
            (str(target), "unverified", nodes.VERIFY_DEADLINE_SKIP_REASON)
        ],
    )

    state = initial_state("Write a file")
    state.update(state_overrides)
    return str(target), nodes.builder_node(state)


def test_an_unverified_file_blocks_and_is_carried(monkeypatch, tmp_path):
    """It reaches failed_verification so the next cycle re-runs it.

    Clearing by omission is the failure the verification pass exists to
    prevent, and a file nobody executed is the purest case of it.
    """
    target, result = _builder_with_unverified_file(monkeypatch, tmp_path)

    assert result["failed_verification"] == [target]
    assert result["unverified"] == [target]
    assert "Not executed before the deadline" in result["blockers"]
    assert "NOT RUN" in result["builder_report"]
    assert not any("Implementation complete" in m for m in result["messages"])


def test_expect_failures_does_not_excuse_an_unverified_file(monkeypatch, tmp_path):
    """The opt-out covers a file meant to fail, not one never executed."""
    target, result = _builder_with_unverified_file(
        monkeypatch, tmp_path, expect_failures=True
    )

    assert result["failed_verification"] == [target]
    assert "Not executed before the deadline" in result["blockers"]


# --- The Architect's and Planner's deadlines -------------------------------
#
# Both are single-call, tool-free nodes, so they can be wrapped whole the way
# the Researcher is. What needed thought was not the mechanism but the fallback
# each one leaves behind.


class _HangingLLM:
    """A seat that never answers in time, until the test releases it.

    It returns rather than raising once released: an exception would travel
    back out of `_with_deadline` on purpose, which is a different behaviour
    from a hang and would mask what these tests are checking.
    """

    def __init__(self, release):
        self._release = release
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        self._release.wait(30)
        return AIMessage(content="")   # abandoned by then; never read


def _hang(monkeypatch, seconds=0.4):
    """Point every seat at a hanging model and shorten every node deadline.

    The Builder keeps its own, larger budget, so patching only
    NODE_DEADLINE_SECONDS would leave that node waiting out the real one.
    """
    import threading

    from langgraph_agent import nodes

    release = threading.Event()
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _HangingLLM(release)
    )
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", seconds)
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", seconds * 2)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", seconds)
    return release


def test_a_hung_architect_never_approves(monkeypatch):
    """The gate ends the run, so a stalled seat must not be able to end one.

    This is the one fallback in the system that could turn a hang into a
    false success, which is why it is asserted on both passes rather than
    just the one that happens to be reachable first.
    """
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    try:
        opening = nodes.architect_node(initial_state("Do the thing"))

        reviewing = initial_state("Do the thing")
        reviewing["plan"] = "1. Do it"
        reviewing["builder_report"] = "Implementation complete"
        gate = nodes.architect_node(reviewing)
    finally:
        release.set()

    assert opening["verdict"] == Verdict.PLAN.value
    assert gate["verdict"] == Verdict.REVISE.value
    for result in (opening, gate):
        assert result["verdict"] != Verdict.APPROVED.value
        assert any("never an approval" in m for m in result["messages"])


def test_a_hung_architect_keeps_the_architecture_it_had(monkeypatch):
    """Losing it mid-run would strip the constraints out of every later prompt."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. Do it"
    state["architecture"] = "Layered, with the parser kept separate."
    try:
        result = nodes.architect_node(state)
    finally:
        release.set()

    assert result["architecture"] == "Layered, with the parser kept separate."


def test_a_hung_architect_still_counts_its_step(monkeypatch):
    """Otherwise a stalling gate loops forever instead of reaching MAX_STEPS."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. Do it"
    try:
        result = nodes.architect_node(state)
    finally:
        release.set()

    assert result["step_count"] == 1


def test_a_hung_planner_leaves_a_plan_behind(monkeypatch):
    """An empty plan would loop uncounted to the recursion limit.

    step_count is incremented only while a plan exists, so a blank fallback
    would send Planner -> Builder -> Architect round without ever counting,
    until LangGraph killed the run by exception and discarded its messages.
    """
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    try:
        result = nodes.planner_node(initial_state("Do the thing"))
    finally:
        release.set()

    assert result["plan"]                      # the load-bearing part
    assert "did not respond" in result["plan"]
    assert result["next_agent"] == "Builder"
    assert any("No response within" in m for m in result["messages"])


def test_a_hung_planner_keeps_a_real_plan_over_the_placeholder(monkeypatch):
    """On a revise cycle the existing plan is better information."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. The plan from the previous cycle"
    try:
        result = nodes.planner_node(state)
    finally:
        release.set()

    assert result["plan"] == "1. The plan from the previous cycle"


class _OffFormatPlannerLLM:
    """Answers the Planner in prose, with no `## Steps` heading to match.

    The shape a weak seat actually produces: it responds, at length, and the
    parser finds nothing in it. Every other seat keeps StubLLM.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(
            content=(
                "Sure! To do this you would want to start by looking at the "
                "indexer, then add the retry, and finally write a test for it."
            )
        )


def _off_format_planner(monkeypatch):
    """Point the Planner seat at a model that answers without the headings."""
    from langgraph_agent import nodes

    llm = _OffFormatPlannerLLM()
    real = nodes.get_agent_llm
    monkeypatch.setattr(
        nodes,
        "get_agent_llm",
        lambda agent, temperature=0.1: (
            llm if agent == "planner" else real(agent, temperature)
        ),
    )
    return llm


def test_a_planner_that_answers_off_format_still_leaves_a_plan(monkeypatch):
    """The other door onto the uncounted loop: the seat replies, unreadably.

    `plan` was whatever `## Steps` matched, so an answer without that heading
    emptied it while the feed said "Plan created" -- and step_count is
    incremented only while a plan exists, so the cycle ran uncounted. Measured
    with the guard removed: a whole run through this seat ends at step_count 0,
    which is the state that lets the loop run to LangGraph's recursion limit
    whenever the gate does not approve.
    """
    from langgraph_agent import nodes

    llm = _off_format_planner(monkeypatch)
    result = nodes.planner_node(initial_state("Do the thing"))

    assert llm.calls == 1                      # the seat really answered
    assert result["plan"]                      # the load-bearing part
    assert "never broken into steps" in result["plan"]
    assert result["next_agent"] == "Builder"
    assert any("no plan steps" in m for m in result["messages"])
    # A seat that produced no plan must not be reported as one that planned.
    assert not any("Plan created" in m for m in result["messages"])


def test_an_off_format_planner_is_not_described_as_a_hung_one(monkeypatch):
    """A seat that replied and a seat that stalled call for different fixes.

    Asserted in both directions so the guard cannot be satisfied by routing
    this case through the timeout fallback, which would report a model that
    answered in 0.2s as one that never answered.
    """
    from langgraph_agent import nodes

    _off_format_planner(monkeypatch)
    result = nodes.planner_node(initial_state("Do the thing"))

    assert any("no plan steps" in m for m in result["messages"])
    assert not any("No response within" in m for m in result["messages"])
    assert "did not respond" not in result["plan"]


def test_an_off_format_planner_keeps_a_real_plan_over_the_placeholder(monkeypatch):
    """As on the timeout path: a previous cycle's plan is better information."""
    from langgraph_agent import nodes

    _off_format_planner(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. The plan from the previous cycle"
    result = nodes.planner_node(state)

    assert result["plan"] == "1. The plan from the previous cycle"
    assert result["next_agent"] == "Builder"


def test_an_off_format_planner_never_routes_to_the_researcher(monkeypatch):
    """An empty plan is an empty search: `_gather_research` queries on `plan`.

    A Researcher hop with no plan searches for the empty string, and retrieval
    that thin is exactly what falls through to the Researcher's own model.
    """
    from langgraph_agent import nodes

    llm = _OffFormatPlannerLLM()

    def _routes_to_researcher(messages):
        from langchain_core.messages import AIMessage

        llm.calls += 1
        return AIMessage(content="## Next Agent\nResearcher\n\n## Notes\nGo look.")

    llm.invoke = _routes_to_researcher
    real = nodes.get_agent_llm
    monkeypatch.setattr(
        nodes,
        "get_agent_llm",
        lambda agent, temperature=0.1: (
            llm if agent == "planner" else real(agent, temperature)
        ),
    )

    result = nodes.planner_node(initial_state("Do the thing"))

    assert result["next_agent"] == "Builder"


def test_an_off_format_planner_run_still_terminates(monkeypatch):
    """End to end: the property every fallback above exists to give.

    Nothing hangs here -- the Planner answers instantly and unreadably, which
    is the shape that used to loop fastest of all. `step_count >= 1` is the
    assertion that fails without the guard: the run reached the end having
    counted no cycle at all.
    """
    from langgraph_agent import graph as graph_module

    _off_format_planner(monkeypatch)
    result = create_agent_graph().invoke(
        initial_state("Do the thing"),
        {"recursion_limit": graph_module.RECURSION_LIMIT},
    )

    assert result["step_count"] >= 1           # the loop was counted at all
    assert result["step_count"] <= graph_module.MAX_STEPS


def test_a_replan_counts_a_step_and_the_ceiling_ends_the_loop():
    """Planner and Researcher could alternate forever: the loop never passed
    the gate that counts steps, so MAX_STEPS never applied to it."""
    from langgraph_agent import graph as graph_module

    below = {"research_status": "need_replan", "step_count": graph_module.MAX_STEPS - 1}
    at = {"research_status": "need_replan", "step_count": graph_module.MAX_STEPS}
    assert graph_module._route_from_researcher(below) == "planner"  # type: ignore[arg-type]
    assert graph_module._route_from_researcher(at) == "__end__"  # type: ignore[arg-type]
    assert graph_module._route_from_researcher(  # type: ignore[arg-type]
        {"research_status": "ready_for_builder", "step_count": graph_module.MAX_STEPS}
    ) == "builder"


def test_the_researcher_counts_the_replan_it_asks_for(monkeypatch):
    from langgraph_agent import nodes

    monkeypatch.setattr(nodes, "_gather_research", lambda state: ("findings", "need_replan"))
    state = initial_state("g")
    state["step_count"] = 2

    result = nodes.researcher_node(state)

    assert result["next_agent"] == "Planner"
    assert result["step_count"] == 3


def test_a_researcher_that_always_wants_a_replan_still_terminates(monkeypatch):
    """End to end: the run ends at the step ceiling, not the recursion limit."""
    from langgraph_agent import graph as graph_module
    from langgraph_agent import nodes

    monkeypatch.setattr(
        nodes, "_gather_research",
        lambda state: ("## Key Findings\n- unclear\n\n## Status\nneed_replan", "need_replan"),
    )
    monkeypatch.setattr(
        nodes, "_make_plan", lambda state: {"plan": "1. Look again", "next_agent": "Researcher"}
    )

    result = create_agent_graph().invoke(
        initial_state("Do the thing"),
        {"recursion_limit": graph_module.RECURSION_LIMIT},
    )

    assert result["step_count"] == graph_module.MAX_STEPS
    assert result["verdict"] != Verdict.APPROVED.value
    assert result["builder_report"] == ""


def test_a_stalling_architect_run_still_terminates(monkeypatch):
    """End to end: every seat hangs, and the run ends at the step ceiling.

    The point of every fallback above is that the graph keeps moving and the
    counter keeps counting; this asserts the property they exist to give.
    """
    from langgraph_agent import graph as graph_module
    from langgraph_agent import nodes

    release = _hang(monkeypatch, seconds=0.05)
    try:
        result = create_agent_graph().invoke(
            initial_state("Do the thing"),
            {"recursion_limit": graph_module.RECURSION_LIMIT},
        )
    finally:
        release.set()

    assert result["step_count"] >= graph_module.MAX_STEPS
    assert result["verdict"] != Verdict.APPROVED.value
    assert nodes.NODE_DEADLINE_SECONDS == 0.05   # the fixture really applied


# ---------------------------------------------------------------------------
# The emergency stop
#
# Cooperative by construction: nothing here interrupts work in flight. The
# checks decline to start the *next* piece of work, which is why a stopped run
# never leaves a half-written file behind -- and why what it did write is still
# on disk and still reported.
# ---------------------------------------------------------------------------


class _StopsAfterWritingLLM(_ToolCallingLLM):
    """Writes a file, then trips the emergency stop before the next turn."""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        from langgraph_agent.control import RUN_CONTROL

        self.calls += 1
        if self.calls == 1:
            RUN_CONTROL.arm("run-under-test")
            RUN_CONTROL.stop("run-under-test", "Stopped from the console.")
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": "half a job\n"},
                    "id": "call_1",
                }],
            )
        raise AssertionError("the stop should have ended the turn loop")


def test_a_stopped_builder_keeps_the_files_it_already_wrote(monkeypatch, tmp_path):
    """The write in flight completes; the next turn never starts."""
    # The Builder writes inside the project root (`_resolve_write_path`),
    # so the root moves to the tmp dir rather than the write escaping it.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "half.txt"
    llm = _StopsAfterWritingLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a file"))

    # The tool call that was already issued ran to completion -- abandoning it
    # is what would leave a half-written file in the project.
    assert target.read_text() == "half a job\n"
    assert result["files_changed"] == [str(target)]
    # ...and the loop stopped rather than taking another turn.
    assert llm.calls == 1
    assert "emergency stop" in result["blockers"]
    assert "Implementation complete" not in result["messages"][-1]
    assert "Stopped by the operator" in result["messages"][-1]


def test_a_stopped_builder_is_not_described_as_out_of_time(monkeypatch, tmp_path):
    """A run the operator stopped must not be reported as one that overran.

    The two are separate flags for exactly this reason: the report is the only
    place anyone finds out which of them happened.
    """
    # Without this the write is refused by `_resolve_write_path` and the seat
    # stops after writing nothing, which is not the case this test is named for.
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import BUILDER_DEADLINE_SECONDS, builder_node

    llm = _StopsAfterWritingLLM(tmp_path / "half.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a file"))

    assert f"{int(BUILDER_DEADLINE_SECONDS)}s" not in result["blockers"]
    assert "deadline" not in result["messages"][-1]


class _StopsAsItReportsLLM(_ToolCallingLLM):
    """Writes a script, then trips the stop while writing its closing report."""

    def invoke(self, messages):
        from langgraph_agent.control import RUN_CONTROL

        reply = super().invoke(messages)
        if self.calls == 2:
            RUN_CONTROL.arm("run-under-test")
            RUN_CONTROL.stop("run-under-test", "Stopped from the console.")
        return reply


def test_a_stop_during_verification_is_not_described_as_the_deadline(
    monkeypatch, tmp_path
):
    """The seat finished; the stop landed before its file was run."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "made.py"
    llm = _StopsAsItReportsLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a script"))

    assert result["unverified"] == [str(target)]
    assert result["blockers"].startswith("Not executed before the stop")
    assert "the run was stopped" in result["builder_report"]
    assert "budget" not in result["builder_report"]
    assert result["messages"][-1].endswith("1 not run (stopped)")


def test_a_stopped_run_leaves_written_files_unproven(monkeypatch, tmp_path):
    """Files nobody executed come back unverified, not clear."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "works.py"
    target.write_text("print('fine')\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    state = initial_state("Fix the module")
    # Carried from an earlier pass: the file exists and would pass if run.
    state["failed_verification"] = [str(target)]

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = builder_node(state)

    assert result["failed_verification"] == [str(target)]
    assert "NOT RUN" in result["builder_report"]
    assert "the run was stopped" in result["builder_report"]


def test_expect_failures_does_not_clear_a_file_the_stop_left_unrun(
    monkeypatch, tmp_path
):
    """The safeguard checkbox cannot make a stopped run look finished.

    `expect_failures` is for a file the run *meant* to fail -- still executed,
    still reported. A file the stop prevented anyone from executing is a gap in
    the evidence, not an expected result, so it goes on blocking either way.
    """
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "works.py"
    target.write_text("print('fine')\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    state = initial_state("Fix the module")
    state["failed_verification"] = [str(target)]
    state["expect_failures"] = True

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = builder_node(state)

    assert result["failed_verification"] == [str(target)]
    assert "Not executed before the stop" in result["blockers"]


def test_a_stopped_seat_never_calls_its_model(monkeypatch):
    """The tool-free nodes bail before spending a cloud call."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import architect_node, planner_node, researcher_node

    class _Counting(StubLLM):
        calls = 0

        def invoke(self, messages):
            type(self).calls += 1
            return super().invoke(messages)

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _Counting(),
    )

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")

    for node, role in (
        (architect_node, "Architect"),
        (planner_node, "Planner"),
        (researcher_node, "Researcher"),
    ):
        result = node(initial_state("anything"))
        assert f"[{role}] Stopped by the emergency stop" in result["messages"][-1]

    assert _Counting.calls == 0


def test_a_stopped_architect_never_approves(monkeypatch):
    """The gate is what ends the run, so a stop must not end one successfully."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )

    state = initial_state("Ship it")
    state["plan"] = "1. Do it"
    state["builder_report"] = "Implementation complete."  # the gate pass
    before = state["step_count"]

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = architect_node(state)

    assert result["verdict"] != Verdict.APPROVED.value
    # A pass that did no work does not count as a step.
    assert result["step_count"] == before


def test_a_stale_stop_cannot_kill_the_run_that_replaced_it():
    """A Stop held over from a finished run must not hit the current one."""
    from langgraph_agent.control import RUN_CONTROL

    RUN_CONTROL.arm("run-two")
    assert RUN_CONTROL.stop("run-one") is False
    assert RUN_CONTROL.stopped() is False

    assert RUN_CONTROL.stop("run-two") is True
    assert RUN_CONTROL.stopped() is True


def test_stopping_nothing_is_refused_not_faked():
    """With no run armed there is nothing to stop, and saying so is the answer."""
    from langgraph_agent.control import RUN_CONTROL

    assert RUN_CONTROL.stop() is False
    assert RUN_CONTROL.stopped() is False


def test_an_annotated_path_is_not_an_unwritten_file():
    """`- file.py (new file)` names a file the Builder wrote, and must match.

    The "Described but not written" line is the report's harshest claim -- the
    mirror of a Builder inventing a file -- and the Architect rules on it. A
    written, executed, passing file was landing under it because the model
    annotated the bullet and nothing normalized the two sides.
    """
    from langgraph_agent.nodes import _parse_builder_output, _report_path_key

    parsed = _parse_builder_output(
        "## Files Modified\n- test_spectral_graph.py (new file)\n"
    )
    assert parsed["files_modified"] == ["test_spectral_graph.py"]

    written = {_report_path_key(p) for p in ["test_spectral_graph.py"]}
    assert [p for p in parsed["files_modified"] if p not in written] == []


def test_prose_spellings_of_a_written_path_all_match_the_tool_record():
    """Markdown, list markers and path form are decoration, not a different file."""
    from langgraph_agent.nodes import _parse_builder_output, _report_path_key

    section = "## Files Modified\n" + "\n".join(
        [
            "- `spectral_graph/operations.py`",
            "* **spectral_graph/operations.py** (modified)",
            "1. ./spectral_graph/operations.py",
            f"- {Path.cwd() / 'spectral_graph' / 'operations.py'}",
        ]
    )
    parsed = _parse_builder_output(section)
    # Every line names one and the same file.
    assert parsed["files_modified"] == ["spectral_graph/operations.py"]

    written = {_report_path_key("./spectral_graph/operations.py")}
    assert [p for p in parsed["files_modified"] if p not in written] == []


def test_a_report_of_nothing_is_not_a_file_called_none():
    """A Builder that honestly wrote nothing must not be accused over the word."""
    from langgraph_agent.nodes import _parse_builder_output

    for answer in ["None", "N/A", "-", "No files were modified in this pass."]:
        parsed = _parse_builder_output(f"## Files Modified\n{answer}\n")
        assert parsed["files_modified"] == [], answer


def test_a_file_that_was_never_written_is_still_named():
    """Normalizing must not blunt the check it exists to keep honest."""
    from langgraph_agent.nodes import _parse_builder_output, _report_path_key

    parsed = _parse_builder_output(
        "## Files Modified\n- `spectral_graph/imaginary.py` (new file)\n"
    )
    written = {_report_path_key("spectral_graph/operations.py")}
    assert [p for p in parsed["files_modified"] if p not in written] == [
        "spectral_graph/imaginary.py"
    ]


def test_a_parenthesised_filename_survives_normalization():
    """This project really has files like filter_band_pass_(40_60_hz).png."""
    from langgraph_agent.nodes import _report_path_key

    assert (
        _report_path_key("- examples/filter_band_pass_(40_60_hz).png")
        == "examples/filter_band_pass_(40_60_hz).png"
    )


# --- The seat lights: which agent is actually working ------------------------
# The console's rainbow dot is only worth anything if it means "this seat's node
# is on the stack right now". These pin that meaning, and above all that a light
# can never be left on by a node that ended some way nobody planned for.


def test_a_running_node_is_the_one_reported_as_working():
    """The light names the node executing, not the one before or after it."""
    seen = {}

    def node(state: AgentState) -> AgentState:
        seen["during"] = ACTIVITY.current()
        return state

    ACTIVITY.clear()
    assert ACTIVITY.current() == ""
    _tracked("researcher", node)(initial_state("g"))
    assert seen["during"] == "researcher"
    assert ACTIVITY.current() == ""


def test_a_node_that_raises_still_puts_its_light_out():
    """The `finally` is the whole reason the marking lives in the wrapper."""

    def node(state: AgentState) -> AgentState:
        raise RuntimeError("the seat fell over")

    ACTIVITY.clear()
    with pytest.raises(RuntimeError):
        _tracked("builder", node)(initial_state("g"))
    assert ACTIVITY.current() == ""


def test_a_late_worker_cannot_darken_the_seat_that_is_working_now():
    """`_with_deadline` abandons threads; one finishing late must not clear."""
    ACTIVITY.clear()
    ACTIVITY.enter("architect")
    ACTIVITY.leave("planner")  # a node that was abandoned turns after
    assert ACTIVITY.current() == "architect"
    # Nor may it close the running turn in the record the lights are driven from.
    assert ACTIVITY.turns()[-1]["ended_ago"] is None
    ACTIVITY.leave("architect")
    assert ACTIVITY.current() == ""


def test_no_seat_is_working_once_the_run_is_over(agent_graph):
    """A whole graph run leaves the crew idle, not stuck on its last node."""
    ACTIVITY.clear()
    agent_graph.invoke(initial_state("Add a health check endpoint"),
                       {"recursion_limit": RECURSION_LIMIT})
    assert ACTIVITY.current() == ""
    assert ACTIVITY.busy_for() == 0.0


def test_a_turn_between_two_polls_is_still_on_the_record():
    """A poll only sees the seat on the stack at the instant it lands.

    Measured through the real page with the console polling once a second, a
    0.4s Researcher turn never lit -- and the Researcher answering from
    retrieval alone is exactly that short. The record is what lets the console
    light a turn it only hears about after it has ended.
    """

    def node(state: AgentState) -> AgentState:
        return state

    ACTIVITY.begin_run()
    _tracked("researcher", node)(initial_state("g"))

    assert ACTIVITY.current() == ""  # a sample taken now sees nothing...
    (turn,) = ACTIVITY.turns()  # ...and the record still has the turn
    assert turn["node"] == "researcher"
    assert turn["ended_ago"] is not None


def test_the_record_is_the_path_the_graph_actually_took(agent_graph):
    """Every turn of a run, in order -- not a sample of them."""
    ACTIVITY.begin_run()
    ran = [
        node
        for event in agent_graph.stream(
            initial_state("Add a health check endpoint"),
            {"recursion_limit": RECURSION_LIMIT},
        )
        for node in event
    ]

    turns = ACTIVITY.turns()
    assert [t["node"] for t in turns] == ran[-TURN_RECORD:]
    assert all(t["ended_ago"] is not None for t in turns)
    numbers = [t["turn"] for t in turns]
    assert numbers == sorted(set(numbers))


def test_a_new_run_starts_an_empty_record_and_keeps_counting():
    """The last run's turns must not light a seat in this one, and its numbers
    must not hide this run's turns from a console still holding them."""
    ACTIVITY.begin_run()
    ACTIVITY.enter("builder")
    ACTIVITY.leave("builder")
    last = ACTIVITY.turns()[-1]["turn"]

    ACTIVITY.begin_run()
    assert ACTIVITY.turns() == []
    ACTIVITY.enter("architect")
    assert ACTIVITY.turns()[0]["turn"] > last
    ACTIVITY.leave("architect")


# ---------------------------------------------------------------------------
# discussion-only runs
# ---------------------------------------------------------------------------


class _TriesToWriteLLM(_ToolCallingLLM):
    """A Builder that reaches for a write tool regardless of what it was offered.

    Which is the case that matters: the guarantee cannot rest on the model
    having read its tool schema.
    """

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "filesystem_write",
                "args": {"path": str(self._path), "content": "should never land\n"},
                "id": "call_1",
            }])
        return AIMessage(content=(
            "## Changes Made\nProposed.\n\n"
            f"## Files Modified\n- {self._path}\n\n"
            "## Next Steps / Blockers\nnone\n"
        ))


def test_a_discussion_run_is_offered_no_tools_at_all(monkeypatch, tmp_path):
    """Zero, not a read-only subset.

    Nothing is bound, so the node takes the same path as a seat whose model
    cannot call tools and there is no tool loop to reason about. "Took no
    action" is then a property of the code that ran rather than of a filter
    having been complete.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    llm = _ToolCallingLLM(tmp_path / "x.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Talk it through")
    state["plan"] = "1. Consider it"
    state["discuss_only"] = True
    builder_node(state)

    assert llm.bound is None, "a discussion run must bind no tools"


def test_a_discussion_run_ignores_a_tool_call_it_was_never_given(monkeypatch, tmp_path):
    """The case that matters: a seat that asks anyway.

    The guarantee cannot rest on the model having read its tool schema, so this
    drives one that emits `filesystem_write` regardless and asserts the file is
    not there afterwards.
    """
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "never.py"
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _TriesToWriteLLM(target),
    )

    state = initial_state("Talk it through")
    state["plan"] = "1. Consider it"
    state["discuss_only"] = True
    result = builder_node(state)

    assert not target.exists(), "a discussion run must not write to disk"
    assert result["files_changed"] == []


def test_a_discussion_run_is_not_reported_as_implementation(monkeypatch, tmp_path):
    """The Architect rules on this line; nothing was implemented."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import builder_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Talk it through")
    state["plan"] = "1. Consider it"
    state["discuss_only"] = True
    result = builder_node(state)

    assert "Implementation complete" not in result["messages"][-1]
    assert "Discussion only" in result["messages"][-1]


def test_the_full_belt_is_still_offered_when_the_box_is_off(monkeypatch, tmp_path):
    """The default must not be quietly narrowed by adding the flag."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import BUILDER_TOOL_NAMES, builder_node

    llm = _ToolCallingLLM(tmp_path / "x.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    builder_node(initial_state("Do the thing"))

    assert {tool["function"]["name"] for tool in llm.bound} == BUILDER_TOOL_NAMES


def test_the_discussion_mode_reaches_every_seat_through_state(monkeypatch):
    """The Architect is the gate, and it cannot rule on a mode it is not told.

    Without this the run has no way to finish: `files_changed` is empty by
    construction, so a gate judging by build standards sends it round until the
    step ceiling -- measured at 8 cycles on a live run before the line existed.
    """
    from langgraph_agent.nodes import _get_state_injection

    state = initial_state("Talk it through")
    assert "DISCUSSION ONLY" not in _get_state_injection(state)

    state["discuss_only"] = True
    block = _get_state_injection(state)
    assert "DISCUSSION ONLY" in block
    assert "rule on" in block


# --- The lint pass ------------------------------------------------------------
#
# A file can run clean and still fail the lint CI runs over it. These use a
# ruff.toml in the tmp dir, since ruff's defaults select no whitespace rules.


def _ruff_config(directory):
    (directory / "ruff.toml").write_text('[lint]\nselect = ["E", "W", "F", "I"]\n')


def test_lint_fixes_whitespace_and_leaves_the_rest_to_the_builder(tmp_path):
    """Whitespace is fixed for the Builder; an unused import is the Builder's."""
    from langgraph_agent import nodes

    _ruff_config(tmp_path)
    target = tmp_path / "mod.py"
    target.write_text("import os\n\n\ndef f():  \n    return 1\n")

    log: list[str] = []
    failures, fixed, unavailable = nodes._lint_written_files([str(target)], log)

    assert unavailable == ""
    assert fixed == [str(target)]
    assert "def f():\n" in target.read_text()      # the trailing spaces went
    assert "import os" in target.read_text()       # the import was left alone
    ((path, findings),) = failures
    assert path == str(target)
    assert [code for _, code, _ in findings] == ["F401"]
    assert f"lint({target}) -> failed" in log


def test_a_clean_file_passes_lint_untouched(tmp_path):
    from langgraph_agent import nodes

    _ruff_config(tmp_path)
    target = tmp_path / "clean.py"
    target.write_text("def f():\n    return 1\n")

    assert nodes._lint_written_files([str(target)], []) == ([], [], "")


def test_a_missing_linter_is_reported_and_blocks_nothing(tmp_path, monkeypatch):
    """No ruff is a fact about the machine, not a defect in the file."""
    from langgraph_agent import nodes

    target = tmp_path / "mod.py"
    target.write_text("import os\n")
    monkeypatch.setattr(
        nodes,
        "_call_tool",
        lambda name, args: {
            "success": False, "stdout": "", "stderr": "python: No module named ruff",
        },
    )

    failures, fixed, unavailable = nodes._lint_written_files([str(target)], [])

    assert failures == [] and fixed == []
    assert "not installed" in unavailable


def test_a_pass_whose_file_runs_but_fails_lint_is_not_complete(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _ruff_config(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "runs_fine.py"
    llm = _WritesFileLLM(target, "import os\n\nprint('ok')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a module"))

    assert result["lint_failed"] == [str(target)]
    assert result["failed_verification"] == []        # it runs clean
    assert "1 fail lint" in result["messages"][-1]
    assert "Files that fail lint" in result["blockers"]
    assert "F401" in result["builder_report"]


def test_a_lint_failure_blocks_approval_even_under_expect_failures(monkeypatch):
    result = _gate_with(monkeypatch, lint_failed=["mod.py"], expect_failures=True)

    assert result["verdict"] == Verdict.REVISE.value
    assert "1 file(s) fail lint" in result["messages"][-1]


def test_a_lint_failure_is_carried_until_it_is_fixed(monkeypatch, tmp_path):
    """A pass that touches nothing re-lints what failed, and clears it once clean."""
    monkeypatch.chdir(tmp_path)
    _ruff_config(tmp_path)
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "mod.py"
    target.write_text("import os\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )
    state = initial_state("Tidy up")
    state["lint_failed"] = [str(target)]

    assert builder_node(dict(state, messages=[]))["lint_failed"] == [str(target)]

    target.write_text("VALUE = 1\n")
    assert builder_node(dict(state, messages=[]))["lint_failed"] == []


# --- The Planner's corpus map -------------------------------------------------


class _RecordingPlannerLLM:
    """A Planner seat that records what it was shown and answers in format."""

    def __init__(self):
        self.seen = ""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.seen = "\n".join(str(message.content) for message in messages)
        return AIMessage(content=(
            "## Goal\nx\n\n## Steps\n1. Edit src/app.py\n\n"
            "## Next Agent\nBuilder\n\n## Notes\nnone\n"
        ))


def _search_answering(results):
    def fake(name, args):
        assert name == "search_knowledge_graph"
        return {"results": results, "source": "local_graphrag"}
    return fake


def test_the_planner_is_shown_the_files_the_corpus_ranks_closest(monkeypatch):
    from langgraph_agent import nodes

    monkeypatch.setattr(nodes, "PLANNER_CORPUS_MAP", True)
    # The floor is measured per corpus and `None` until then; pin one so the
    # ranking assertion below is about the filter and not about the absence.
    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: 0.37)
    monkeypatch.setattr(nodes, "_call_tool", _search_answering([
        {"id": "src/app.py", "score": 0.61, "content": "def render_graph():\n    ...",
         "metadata": {"path": "src/app.py"}},
        {"id": "notes/unrelated.md", "score": 0.12, "content": "nothing to do with it"},
    ]))
    llm = _RecordingPlannerLLM()
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: llm)

    nodes._make_plan(initial_state("Fix the graph rendering"))

    assert "src/app.py (0.61): def render_graph(): ..." in llm.seen
    # Under the relevance floor is not a related file, so it is not shown.
    assert "notes/unrelated.md" not in llm.seen


def test_no_map_when_nothing_clears_the_floor(monkeypatch):
    from langgraph_agent import nodes

    # A pinned floor, so the empty map is the filter's doing -- with the floor
    # unmeasured (`None`) there is no map either, and that path is not this test.
    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: 0.37)
    monkeypatch.setattr(nodes, "_call_tool", _search_answering([
        {"id": "a.md", "score": 0.10, "content": "x"},
    ]))

    assert nodes._corpus_map("anything") == ""


def test_a_failed_search_leaves_the_planner_planning(monkeypatch):
    from langgraph_agent import nodes

    def boom(name, args):
        raise RuntimeError("no corpus here")

    monkeypatch.setattr(nodes, "_call_tool", boom)

    assert nodes._corpus_map("anything") == ""
    assert nodes._corpus_map("   ") == ""


def test_the_map_stays_off_when_switched_off(monkeypatch):
    """conftest switches it off; a switched-off map must not even search."""
    from langgraph_agent import nodes

    def must_not_search(name, args):
        raise AssertionError("searched with the corpus map switched off")

    monkeypatch.setattr(nodes, "_call_tool", must_not_search)
    llm = _RecordingPlannerLLM()
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: llm)

    nodes._make_plan(initial_state("Fix the graph rendering"))

    assert nodes.CORPUS_MAP_HEADING not in llm.seen


# --- Blockers -----------------------------------------------------------------


def test_the_blockers_heading_parses_in_both_spellings():
    from langgraph_agent.nodes import _parse_builder_output

    for heading in ("## Blockers", "## Next Steps / Blockers"):
        parsed = _parse_builder_output(f"{heading}\nneeds the API key\n")
        assert parsed["next_steps_blockers"] == "needs the API key", heading


class _ProposingLLM:
    """A discussion Builder that fills its Blockers section with boilerplate."""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content=(
            "## Changes Made\nWould add a not-found message to trace().\n\n"
            "## Blockers\nThis proposal requires approval from the Architect to proceed.\n"
        ))


def test_a_discussion_builder_s_blockers_stay_in_its_proposal(monkeypatch):
    """Nothing blocks a run that cannot act; the text is part of the proposal."""
    from langgraph_agent import nodes

    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ProposingLLM()
    )
    state = initial_state("Discuss the graph tab")
    state["discuss_only"] = True

    result = nodes.builder_node(state)

    assert result["blockers"] == ""
    assert "requires approval from the Architect" in result["builder_report"]


class _Answers:
    """A seat that answers every call with the same text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, messages):
        return SimpleNamespace(content=self._text)


@pytest.mark.parametrize(
    "answered",
    [Verdict.APPROVED.value, Verdict.NEED_RESEARCH.value, Verdict.REVISE.value],
)
def test_the_opening_pass_rules_plan_whatever_the_seat_answers(monkeypatch, answered):
    """Nothing has been planned or built yet, so `plan` is the only ruling.

    An opening `approved` ended the run before any other seat had worked, and
    an opening `need_research` routed around the Planner: the Researcher got an
    empty `plan` to search on, and since the gate counts a step only while a
    plan exists, that loop went uncounted until LangGraph's recursion limit
    killed the run.
    """
    from langgraph_agent import nodes

    monkeypatch.setattr(
        nodes, "get_agent_llm",
        lambda agent, temperature=0.1: _Answers(
            f"## Architecture\nOne module.\n\n## Constraints\n- none\n\n## Verdict\n{answered}"
        ),
    )

    state = nodes.architect_node(initial_state("Create hello.txt"))

    assert state["verdict"] == Verdict.PLAN.value
    assert state["step_count"] == 0
    assert "One module." in state["architecture"]
    # What the seat actually said is kept where the operator reads the run.
    assert f"`{answered}`" in state["messages"][-1]


def test_the_gate_pass_still_rules_what_the_seat_answered(monkeypatch):
    """Only the opening pass is held to `plan`; the gate's ruling stands."""
    from langgraph_agent import nodes

    monkeypatch.setattr(
        nodes, "get_agent_llm",
        lambda agent, temperature=0.1: _Answers("## Architecture\nx\n\n## Verdict\nneed_research"),
    )
    state = initial_state("Create hello.txt")
    state["plan"] = "1. Write the file."
    state["builder_report"] = "Wrote nothing yet."

    result = nodes.architect_node(state)

    assert result["verdict"] == Verdict.NEED_RESEARCH.value
    assert result["step_count"] == 1


@pytest.mark.parametrize(
    "written",
    [
        "## Verdict\n**revise**",
        "## Verdict: revise",
        "## Verdict\n`revise`",
        "## Verdict\n- revise",
        "## Verdict\n\nRevise -- the plan missed the tests.",
    ],
)
def test_a_decorated_verdict_is_read_rather_than_defaulted(written):
    """The gate's fallback is `approved`, so a verdict it could not read ended the run.

    `**revise**` is how a model emphasises one word, and reading that as no
    verdict at all turned the Architect's refusal into an approval.
    """
    from langgraph_agent.nodes import _parse_architect_output

    assert _parse_architect_output(written, reviewing=True)["verdict"] == "revise"


def test_the_value_is_not_taken_from_the_next_section():
    """An empty Verdict section is still no verdict, not the next section's word."""
    from langgraph_agent.nodes import _parse_architect_output

    parsed = _parse_architect_output("## Verdict\n\n## Notes\nrevise later", reviewing=True)

    assert parsed["verdict"] == "approved"


def test_decorated_routing_words_are_read_for_the_planner_and_researcher():
    """Both parsers default to a route, so an unread word was a silent wrong turn."""
    from langgraph_agent.nodes import _parse_planner_output, _parse_researcher_output

    planned = _parse_planner_output("## Steps\n1. x\n\n## Next Agent\n**researcher**")
    assert planned["next_agent"] == "Researcher"
    assert _parse_planner_output("## Steps\n1. x\n\n## Next Agent: Builder")["next_agent"] == "Builder"

    researched = _parse_researcher_output("## Key Findings\n- a\n\n## Status\n`need_replan`")
    assert researched["status"] == "need_replan"
