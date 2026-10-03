"""The LangGraph wiring: Architect → Planner → (Researcher | Builder) → Architect.

The Architect opens the run and holds the gate; `MAX_STEPS` passes through the
gate end a run that never converges.
"""

from typing import Any, Literal, Protocol

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent.control import ACTIVITY
from langgraph_agent.nodes import (
    architect_node,
    builder_node,
    plan_is_placeholder,
    planner_node,
    researcher_node,
)
from langgraph_agent.state import AgentState, ResearchStatus, Verdict

# Cycles through the Architect gate before a run ends unapproved.
MAX_STEPS = 8

# Supersteps LangGraph runs before giving up: the backstop behind the gate. A
# step costs four supersteps, more when the Researcher sends the Planner back,
# so 4 * MAX_STEPS would fire before the gate could end the run itself.
RECURSION_LIMIT = 6 * MAX_STEPS + 4


# Module-level, not closures, so they can be tested without running the graph.
def _route_from_architect(state: AgentState) -> Literal["planner", "researcher", "__end__"]:
    """Route on the Architect's verdict. Opening move and terminating gate."""
    # The step ceiling is checked first so a stuck loop cannot outvote it.
    if state.get("step_count", 0) >= MAX_STEPS:
        return "__end__"

    verdict = state.get("verdict", Verdict.PLAN.value)

    if verdict == Verdict.APPROVED.value:
        return "__end__"
    if verdict == Verdict.NEED_RESEARCH.value:
        return "researcher"
    # plan and revise both go to the Planner; the difference is that on a
    # revise there is already a plan, research and a report in state for it
    # to work from.
    return "planner"


def _route_from_planner(state: AgentState) -> Literal["researcher", "builder"]:
    """Respect the Planner's routing, except for the first hop of a run.

    The opening cycle always retrieves once before the Builder acts: otherwise
    a confident plan naming the Builder means a run that rebuilt its corpus and
    never read it. Where the corpus answers, the hop costs a search and no model
    call (`_gather_research`).

    A placeholder plan is exempt (`plan_is_placeholder`): it is the Planner
    having failed, nothing worth searching on, and the Builder is the shorter
    way back to the Architect.
    """
    next_agent = state.get("next_agent", "Researcher")
    if next_agent.lower() != "builder":
        return "researcher"

    # `research_status` marks "the Researcher has run" -- not `research`, which
    # can legitimately come back empty. Every exit from `researcher_node` sets
    # it except the emergency stop, which ends the run anyway.
    if state.get("research_status"):
        return "builder"
    if plan_is_placeholder(state.get("plan", "")):
        return "builder"
    return "researcher"


def _route_from_researcher(state: AgentState) -> Literal["planner", "builder", "__end__"]:
    """Route on research_status; only a need_replan goes back to the Planner.

    A replan counts a step (`researcher_node`), since the Planner and the
    Researcher can loop without reaching the gate; at `MAX_STEPS` it ends the
    run unapproved, as the gate would.
    """
    status = state.get("research_status", ResearchStatus.READY_FOR_BUILDER.value)

    if status == ResearchStatus.NEED_REPLAN.value:
        if state.get("step_count", 0) >= MAX_STEPS:
            return "__end__"
        return "planner"
    return "builder"


class _NodeFn(Protocol):
    """A graph node: `(state) -> state`, with the parameter named `state`.

    A `Callable` will not do: LangGraph's node protocol names its parameter.
    """

    def __call__(self, state: AgentState) -> AgentState: ...


def _tracked(name: str, node: _NodeFn) -> _NodeFn:
    """Mark a seat as working for exactly as long as its node is on the stack.

    The graph's stream reports a node when it *ends*, so this is the only honest
    source for a seat light; the `finally` puts the light out however the node
    leaves.
    """

    def run(state: AgentState) -> AgentState:
        ACTIVITY.enter(name)
        try:
            return node(state)
        finally:
            ACTIVITY.leave(name)

    return run


def create_agent_graph() -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Create and compile the 4-agent system graph.

    Graph structure:
        START → Architect → Planner → (Researcher | Builder) → Architect → END
                     ^                                              |
                     +--------- revise / need_research -------------+

    Routing logic:
    - Architect opens with `plan`, then rules on the Builder's report
    - Planner chooses Researcher (needs knowledge) or Builder (task is clear),
      except on the opening cycle, which always retrieves once before the
      Builder acts -- see `_route_from_planner`
    - Researcher sets status: ready_for_builder | need_replan | no_relevant_knowledge;
      a need_replan counts a step, since that loop never reaches the gate
    - Builder always reports back to the Architect; it does not decide it is done
    - Stops on an `approved` verdict, or at MAX_STEPS
    """
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("architect", _tracked("architect", architect_node))
    graph_builder.add_node("planner", _tracked("planner", planner_node))
    graph_builder.add_node("researcher", _tracked("researcher", researcher_node))
    graph_builder.add_node("builder", _tracked("builder", builder_node))

    # Nothing is planned before the architectural direction exists.
    graph_builder.set_entry_point("architect")

    graph_builder.add_conditional_edges(
        "architect",
        _route_from_architect,
        {
            "planner": "planner",
            "researcher": "researcher",
            END: END,
        },
    )

    graph_builder.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "researcher": "researcher",
            "builder": "builder",
        },
    )

    graph_builder.add_conditional_edges(
        "researcher",
        _route_from_researcher,
        {
            "planner": "planner",
            "builder": "builder",
            END: END,
        },
    )

    # The Builder reports to the Architect, which rules on the work and counts
    # the step.
    graph_builder.add_edge("builder", "architect")

    return graph_builder.compile()
