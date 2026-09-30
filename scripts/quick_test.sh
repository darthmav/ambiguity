#!/bin/bash
# Quick verification and testing for the 4-Agent System
# Usage: ./scripts/quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# The project's packages live in .venv, which install.sh builds. Without this,
# `python` is whatever is first on PATH -- on Omarchy that is mise's
# interpreter, which has none of them -- the same reason launch_console.sh
# activates it.
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    . .venv/bin/activate
fi

echo "========================================"
echo "  4-AGENT SYSTEM - QUICK TEST"
echo "========================================"

echo -e "\n► Checking dependencies..."
python -c "import langgraph, langchain_core, chromadb, transformers, networkx; print('  ✓ Core deps OK')"

# Asked of get_agent_status rather than inferred from which API keys are set:
# every default seat runs on the local Ollama daemon and needs no key, and a
# key being present says nothing about whether its seat can run.
echo -e "\n► Checking seats..."
python - <<'PY'
from langgraph_agent.config import AGENTS, get_agent_status

for agent in AGENTS:
    seat = get_agent_status(agent)
    note = "" if seat["live"] else f"  !! {seat['badge']}: {seat['reason']}"
    print(f"  {agent:<11}{seat['provider']:<10}{seat['model']}{note}")
PY

# Opened, never created: a quick test must not be what puts a corpus on a
# machine that had none. The test suite itself always runs the StubLLM.
echo -e "\n► Testing GraphRAG..."
python - <<'PY'
from langgraph_agent.graphrag_server import open_knowledge_base

kb = open_knowledge_base()
if kb is None:
    print("  - No corpus on this machine yet; the console builds one when it starts")
else:
    results = kb.search("Planner agent", top_k=2)
    print(f"  ✓ Found {len(results)} result(s)")
PY

echo -e "\n► Running tests..."
python -m pytest tests/ -v --tb=short -q

echo -e "\n========================================"
echo "  ✓ VERIFICATION COMPLETE"
echo "========================================"
echo -e "\nRun full example: python example_usage.py"
