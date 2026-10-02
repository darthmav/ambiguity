#!/usr/bin/env python3
"""End-to-end run on the cloud provider: every seat on Anthropic.

Every seat defaults to a local Ollama model, so a key being present is not
enough: each seat not already configured through `{ROLE}_PROVIDER` or
`{ROLE}_MODEL` is pointed at Anthropic, on its default model there. The run
writes into its own generated project, `projects/cloud-smoke/`, never into the
checkout.

Usage:
    python scripts/cloud_smoke.py
"""

import os
import sys
from pathlib import Path

from langgraph_agent import create_agent_graph, initial_state
from langgraph_agent.config import AGENTS, get_agent_status
from langgraph_agent.graph import RECURSION_LIMIT
from langgraph_agent.projects import project_dir

OUTPUT_DIR = project_dir("cloud-smoke")
TARGET = Path(OUTPUT_DIR) / "cloud_test.txt"


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; nothing to test.")
        return 0

    for agent in AGENTS:
        if not os.getenv(f"{agent.upper()}_PROVIDER") and not os.getenv(f"{agent.upper()}_MODEL"):
            os.environ[f"{agent.upper()}_PROVIDER"] = "anthropic"
    for agent in AGENTS:
        seat = get_agent_status(agent)
        print(f"  {agent:<11}{seat['provider']:<10}{seat['model']}")

    state = initial_state(
        f"Create {TARGET} with the content 'Cloud LLM works!'", output_dir=OUTPUT_DIR
    )
    print("Running the four seats on Anthropic...")
    result = create_agent_graph().invoke(state, {"recursion_limit": RECURSION_LIMIT})

    print(f"\nPlan: {result.get('plan', '')!r}")
    print(f"Files changed: {result.get('files_changed', [])}")
    print(f"Builder report: {result.get('builder_report', '')!r}")
    if not TARGET.exists():
        print(f"\n✗ {TARGET} was not created")
        return 1
    print(f"\n✓ {TARGET}: {TARGET.read_text()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
