#!/usr/bin/env python3
"""Example usage of the 4-Agent System.

Demonstrates:
- Architect → Planner → Researcher → Builder → Architect flow
- State injection on every turn
- Strict output format parsing
- Local-first inference: every seat defaults to a model the local Ollama
  daemon serves, so no API key is needed (Anthropic and OpenAI stay optional)
"""

from langgraph_agent import AgentState, create_agent_graph
from langgraph_agent.config import AGENTS, get_agent_status
from langgraph_agent.graph import MAX_STEPS, RECURSION_LIMIT


def run_example(goal: str) -> AgentState:
    """Run the 4-agent system with a goal.

    The run ends on the Architect's `approved`, or after `MAX_STEPS` passes
    through its gate -- a module constant, not a parameter here.

    Args:
        goal: The goal to achieve
    """
    # Asked rather than inferred from which API keys are set: the default seats
    # need none, and a seat that cannot run says why.
    for agent in AGENTS:
        seat = get_agent_status(agent)
        if not seat["live"]:
            print(f"Note: the {agent} seat cannot run -- {seat['badge']}: {seat['reason']}")

    graph = create_agent_graph()

    # Initialize state per the 4-Agent System specification. `research_status`
    # starts empty: it is what marks the Researcher as having run, so the
    # opening cycle retrieves once before the Builder acts, whatever the
    # Planner asks for.
    initial_state: AgentState = {
        "goal": goal,
        "messages": [],
        "architecture": "",
        "verdict": "",
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",  # Default, Planner will set
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "failed_verification": [],
        "unverified": [],
        "builder_cut_off": "",
        "lint_failed": [],
        "discuss_only": False,
        "output_dir": "",
        "expect_failures": False,
        "step_count": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"4-Agent System - Goal: {goal}")
    print(f"{'=' * 70}\n")

    # Run the graph. The recursion limit is the backstop behind MAX_STEPS,
    # the same one the console passes.
    result: AgentState = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n## Plan")
    print(result.get("plan", "(empty)") or "(empty)")

    print("\n## Research Findings")
    print(result.get("research", "(empty)") or "(empty)")

    print("\n## Builder Report")
    print(result.get("builder_report", "(empty)") or "(empty)")

    print("\n## Files Changed")
    files = result.get("files_changed", [])
    if files:
        for f in files:
            print(f"  - {f}")
    else:
        print("  (none)")

    print("\n## Blockers")
    blockers = result.get("blockers", "")
    print(blockers if blockers else "  (none)")

    print("\n## Execution Log")
    for msg in result.get("messages", []):
        print(f"  - {msg}")

    print(f"\n## Steps Taken: {result.get('step_count', 0)} (ceiling {MAX_STEPS})")

    return result


if __name__ == "__main__":
    # Example 1: a fully specified build task. The Planner may route straight
    # to the Builder, but the opening cycle still retrieves once first.
    run_example("Create a hello.txt file containing 'Hello World'")

    # Example 2: Research-heavy task
    print("\n\n")
    run_example("Research the best practices for Python async error handling")
