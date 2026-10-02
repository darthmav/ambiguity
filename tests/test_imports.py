"""The package's public surface, pinned."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import langgraph_agent
from langgraph_agent import AgentState, initial_state

# A symbol added to or dropped from `__all__` is a change to the package's
# public API, and has to be made here as well.
PUBLIC_API = {"AgentState", "ResearchStatus", "Verdict", "create_agent_graph", "initial_state"}


def test_the_exported_surface_is_pinned():
    assert set(langgraph_agent.__all__) == PUBLIC_API


def test_every_export_resolves_and_is_documented():
    assert langgraph_agent.__doc__
    for name in langgraph_agent.__all__:
        assert getattr(langgraph_agent, name).__doc__, f"{name} has no docstring"


def test_the_version_is_the_distribution_s_own():
    assert langgraph_agent.__version__ == version("langgraph-agent")


def test_initial_state_sets_every_field_and_only_the_caller_s_flags():
    state = initial_state("Write a module", expect_failures=True, output_dir="projects/demo")

    assert set(state) == set(AgentState.__annotations__)
    assert (state["goal"], state["expect_failures"], state["discuss_only"],
            state["output_dir"]) == ("Write a module", True, False, "projects/demo")
    written_by_agents = set(state) - {
        "goal", "expect_failures", "discuss_only", "output_dir", "next_agent", "step_count",
    }
    assert not any(state[field] for field in written_by_agents)
    assert state["step_count"] == 0


def test_each_initial_state_is_its_own():
    first, second = initial_state("a"), initial_state("b")
    first["messages"].append("x")
    first["files_changed"].append("y.py")
    assert second["messages"] == [] and second["files_changed"] == []


def test_importing_the_package_reads_no_prompt_file(tmp_path):
    """The image's build stage imports the package before prompts/ is copied in.

    A prompt read at import broke that build, so the import runs here in a fresh
    interpreter that records every file it opens.
    """
    prompts = Path(langgraph_agent.__file__).resolve().parents[2] / "prompts"
    probe = (
        "import sys\n"
        "opened = []\n"
        f"prefix = {str(prompts)!r}\n"
        "sys.addaudithook(lambda event, args: event == 'open'"
        " and str(args[0]).startswith(prefix) and opened.append(str(args[0])))\n"
        "import langgraph_agent.graphrag_server  # what the build stage imports\n"
        "print(opened)\n"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_a_missing_prompt_is_named_when_its_seat_needs_it(monkeypatch, tmp_path):
    from langgraph_agent import nodes

    monkeypatch.setattr(nodes, "PROMPTS_DIR", tmp_path)
    nodes.seat_prompt.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="The architect prompt is missing"):
            nodes.seat_prompt("architect")
    finally:
        nodes.seat_prompt.cache_clear()
