#!/usr/bin/env python3
"""Cloud LLM end-to-end test.

This script verifies that the 4-Agent system can run using a cloud LLM.
Set ANTHROPIC_API_KEY (preferred when both are set) or OPENAI_API_KEY in your
environment.

Every seat defaults to a local Ollama model, so a key being present is not
enough: without naming the provider, this ran the local seats and reported a
cloud test. Each seat is therefore pointed at the provider whose key is set,
through the same `{ROLE}_PROVIDER` variables `.env` uses, with each seat on
that provider's default model. A seat already configured in the environment
is left as it is.
"""

import os
from pathlib import Path

from langgraph_agent import AgentState, create_agent_graph
from langgraph_agent.config import AGENTS, get_agent_status
from langgraph_agent.graph import RECURSION_LIMIT

if os.getenv("ANTHROPIC_API_KEY"):
    provider = "anthropic"
elif os.getenv("OPENAI_API_KEY"):
    provider = "openai"
else:
    print("ANTHROPIC_API_KEY and OPENAI_API_KEY are not set. Skipping cloud LLM test.")
    print("Set one of them to run this test.")
    raise SystemExit(0)

for agent in AGENTS:
    if not os.getenv(f"{agent.upper()}_PROVIDER") and not os.getenv(f"{agent.upper()}_MODEL"):
        os.environ[f"{agent.upper()}_PROVIDER"] = provider

for agent in AGENTS:
    seat = get_agent_status(agent)
    print(f"  {agent:<11}{seat['provider']:<10}{seat['model']}")

graph = create_agent_graph()

# Simple file creation task
state: AgentState = {
    "goal": "Create cloud_test.txt with content 'Cloud LLM works!'",
    "messages": [],
    "architecture": "",
    "verdict": "",
    "plan": "",
    "research": "",
    "builder_report": "",
    "next_agent": "Researcher",
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

print("Running 4-Agent with cloud LLM...")
result = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})

print("\n=== RESULTS ===")
print(f"Full plan: {result.get('plan', 'N/A')!r}")
print(f"Files changed: {result.get('files_changed', [])}")
print(f"Builder report: {result.get('builder_report', 'N/A')!r}")
print(f"Messages: {result.get('messages', [])}")

# Verify file was created
if Path("cloud_test.txt").exists():
    print(f"\n✓ File created! Content: {Path('cloud_test.txt').read_text()!r}")
else:
    print("\n✗ File not created")
