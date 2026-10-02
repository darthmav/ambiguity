"""LangGraph Agent: a local-first four-agent system for software development.

- **Architect** sets the architectural direction and holds the approval gate;
  a run ends on its approval.
- **Planner** turns the goal and that direction into a structured plan.
- **Researcher** gathers relevant knowledge from the GraphRAG corpus.
- **Builder** implements the plan with filesystem, git, terminal and test tools.

Example:
    >>> from langgraph_agent import create_agent_graph, initial_state
    >>> from langgraph_agent.graph import RECURSION_LIMIT
    >>> graph = create_agent_graph()
    >>> result = graph.invoke(
    ...     initial_state("Create a Python module"), {"recursion_limit": RECURSION_LIMIT}
    ... )
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from langgraph_agent.graph import create_agent_graph
from langgraph_agent.state import AgentState, ResearchStatus, Verdict, initial_state

try:
    __version__ = _version("langgraph-agent")
except PackageNotFoundError:  # imported from a checkout that was never installed
    __version__ = "0+unknown"

__all__ = [
    "AgentState",
    "ResearchStatus",
    "Verdict",
    "create_agent_graph",
    "initial_state",
]
