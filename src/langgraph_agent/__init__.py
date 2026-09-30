"""LangGraph Agent - 4-Agent System implementation.

A local-first AI system for software development experiments with four specialized
agents working in a coordinated workflow:

- **Architect** — The leading authority. Sets architectural direction before planning
  and holds the approval gate. The run ends on its approval.
- **Planner** — Creates structured plans based on architectural direction.
- **Researcher** — Gathers high-quality, relevant knowledge using GraphRAG.
- **Builder** — Implements the plan using research and creates working code.

The system uses LangGraph for orchestration with strict system prompts, state
injection on every turn, tool binding per agent, and local-first inference.

Example:
    >>> from langgraph_agent import create_agent_graph, AgentState
    >>> graph = create_agent_graph()
    >>> initial_state = AgentState(
    ...     goal="Create a Python module",
    ...     messages=[],
    ...     architecture="",
    ...     verdict="plan",
    ...     plan="",
    ...     research="",
    ...     builder_report="",
    ...     next_agent="Researcher",
    ...     research_status="ready_for_builder",
    ...     blockers="",
    ...     files_changed=[],
    ...     failed_verification=[],
    ...     expect_failures=False,
    ...     step_count=0,
    ... )
    >>> result = graph.invoke(initial_state)

Attributes:
    __version__: The package version string.
    __all__: List of public API symbols exported by this module.
"""

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from langgraph_agent.graph import create_agent_graph
from langgraph_agent.state import AgentState, ResearchStatus, Verdict

__version__ = "0.2.0"

__all__ = [
    "create_agent_graph",
    "AgentState",
    "ResearchStatus",
    "Verdict",
    "CompiledStateGraph",
    "get_agent_graph",
]


def get_agent_graph() -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Create and return a new compiled agent graph instance.

    This is a convenience wrapper around :func:`create_agent_graph` that
    returns a fresh graph instance ready for use.

    Returns:
        CompiledStateGraph: A compiled LangGraph state graph configured with
            the 4-agent system.

    Example:
        >>> graph = get_agent_graph()
        >>> result = graph.invoke(initial_state)
    """
    return create_agent_graph()
