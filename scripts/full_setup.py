#!/usr/bin/env python3
"""Fully automated verification - no prompts required.

This script:
1. Imports the GraphRAG server, the way the console does
2. Reports what each seat will actually run
3. Tests search against whatever corpus this machine holds
4. Runs the test suite

It does not build the corpus. The console builds it when it starts, and every
run brings it up to date before the Architect opens.

It used to open by *editing* `graphrag_server.py` -- rewriting an MCP import
that a long-gone SDK version needed -- and then printed "GraphRAG imports OK"
from a check that imported nothing. A setup script has no business editing
the program's source, and a check that cannot fail tells nobody anything, so
both are gone: the import below is real, and the exit status says whether
setup succeeded.

Usage:
    python scripts/full_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_graphrag_import() -> bool:
    """Import the GraphRAG server, which is what the console's corpus runs on."""
    print("\nTesting GraphRAG import...")
    try:
        import langgraph_agent.graphrag_server  # noqa: F401
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False
    print("  ✓ GraphRAG imports OK")
    return True


def report_seats() -> None:
    """Say what each seat will run, from the one place that knows.

    Every default seat runs on the local Ollama daemon and needs no key, so
    which API keys happen to be set says nothing about whether a run can work.
    """
    print("\nSeats:")
    from langgraph_agent.config import AGENTS, get_agent_status

    for agent in AGENTS:
        seat = get_agent_status(agent)
        note = "" if seat["live"] else f"  !! {seat['badge']}: {seat['reason']}"
        print(f"  {agent:<11}{seat['provider']:<10}{seat['model']}{note}")


def test_search() -> bool:
    """Test GraphRAG search."""
    print("\nTesting GraphRAG search...")

    from langgraph_agent.graphrag_server import open_knowledge_base

    # Opened, never created. Nothing in this script builds a corpus, so an
    # absent one is reported rather than quietly made -- a store that appeared
    # because something looked at it is a corpus nobody asked for.
    kb = open_knowledge_base()
    if kb is None:
        print("  - No corpus on this machine yet; the console builds one when it starts")
        return False

    results = kb.search("Planner agent", top_k=2)

    if results:
        print(f"  ✓ Search working ({len(results)} results)")
        return True
    else:
        print("  ✗ No results found")
        return False


def run_tests() -> bool:
    """Run test suite."""
    print("\nRunning tests...")

    import pytest
    exit_code = pytest.main(["-q", "--tb=no", "tests/"])

    if exit_code == 0:
        print("  ✓ All tests passing")
        return True
    else:
        print(f"  ✗ Tests failed (exit code {exit_code})")
        return False


def main() -> None:
    """Run all setup steps; exit non-zero if a required one failed."""
    print("=" * 60)
    print("4-AGENT SYSTEM - FULL AUTOMATED SETUP")
    print("=" * 60)

    if not test_graphrag_import():
        print("\n✗ Setup failed at import stage")
        sys.exit(1)

    report_seats()

    # Informational: a machine with no corpus yet is not a failed setup.
    test_search()

    tests_ok = run_tests()

    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE" if tests_ok else "✗ SETUP INCOMPLETE: the test suite failed")
    print("=" * 60)
    if not tests_ok:
        sys.exit(1)
    print("\nReady to use:")
    print("  ./launch_console.sh")
    print("\nThe console brings the corpus up to date when it starts, and every run")
    print("checks it again before the Architect opens.")


if __name__ == "__main__":
    main()
