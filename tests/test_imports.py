"""The package's public surface, pinned."""

from importlib.metadata import version

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
