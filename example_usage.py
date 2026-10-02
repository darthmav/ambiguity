#!/usr/bin/env python3
"""Two goals through the four-agent loop, printed section by section.

Every seat defaults to a model the local Ollama daemon serves, so no API key is
needed. The runs write into their own generated project, `projects/example/`,
never into the checkout.
"""

from langgraph_agent import AgentState, create_agent_graph, initial_state
from langgraph_agent.config import AGENTS, get_agent_status
from langgraph_agent.graph import MAX_STEPS, RECURSION_LIMIT
from langgraph_agent.projects import project_dir


def run_example(goal: str) -> AgentState:
    """Run one goal; it ends on the Architect's `approved`, or after `MAX_STEPS`."""
    for agent in AGENTS:
        seat = get_agent_status(agent)
        if not seat["live"]:
            print(f"Note: the {agent} seat cannot run -- {seat['badge']}: {seat['reason']}")

    print(f"\n{'=' * 70}\n4-Agent System - Goal: {goal}\n{'=' * 70}\n")
    result: AgentState = create_agent_graph().invoke(
        initial_state(goal, output_dir=project_dir("example")),
        {"recursion_limit": RECURSION_LIMIT},
    )

    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    for heading, field in (("Plan", "plan"), ("Research Findings", "research"),
                           ("Builder Report", "builder_report"), ("Blockers", "blockers")):
        print(f"\n## {heading}\n{result.get(field) or '(empty)'}")
    print("\n## Files Changed")
    for path in result.get("files_changed") or ["(none)"]:
        print(f"  - {path}")
    print("\n## Execution Log")
    for message in result.get("messages", []):
        print(f"  - {message}")
    print(f"\n## Steps Taken: {result.get('step_count', 0)} (ceiling {MAX_STEPS})")
    return result


if __name__ == "__main__":
    # A fully specified build task: the Planner may route straight to the
    # Builder, but the opening cycle still retrieves once first.
    run_example("Create a hello.txt file containing 'Hello World'")
    print("\n")
    run_example("Research the best practices for Python async error handling")
