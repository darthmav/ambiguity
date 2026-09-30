#!/usr/bin/env python3
"""Fully automated verification - no prompts, just run and report.

Usage:
    python scripts/auto_verify.py
"""

import importlib.util
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _can_import(module: str) -> bool:
    """Check if a module is available without importing it into this namespace."""
    return importlib.util.find_spec(module) is not None


def main():
    """Run all verification steps silently."""
    results = []

    # 1. Dependencies
    required = [
        "langgraph",
        "langchain_core",
        "chromadb",
        "transformers",
        "networkx",
    ]
    missing = [m for m in required if not _can_import(m)]
    if missing:
        results.append(f"✗ Missing dependencies: {', '.join(missing)}")
        print("\n".join(results))
        sys.exit(1)
    results.append("✓ Core dependencies OK")

    # 2. Import agent modules
    try:
        import langgraph_agent  # noqa: F401

        results.append("✓ Agent modules import OK")
    except Exception as e:
        results.append(f"✗ Import error: {e}")
        print("\n".join(results))
        sys.exit(1)

    # 3. Test GraphRAG
    try:
        from langgraph_agent.graphrag_server import open_knowledge_base

        # Opened, not created: a silent verification run must not be the thing
        # that puts a corpus on a machine that had none.
        kb = open_knowledge_base()
        if kb is None:
            results.append("- GraphRAG has no corpus (a run builds one)")
        else:
            kb.search("test", top_k=1)
            results.append("✓ GraphRAG working")
    except Exception as e:
        results.append(f"✗ GraphRAG error: {e}")

    # 4. Run tests
    try:
        import pytest

        exit_code = pytest.main(["-q", "--tb=no", "tests/"])
        if exit_code == 0:
            results.append("✓ All tests passing")
        else:
            results.append("✗ Some tests failed")
    except Exception as e:
        results.append(f"✗ Test error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("4-AGENT SYSTEM STATUS")
    print("=" * 50)
    for r in results:
        print(r)
    print("=" * 50)

    # Quick usage example. The default seats run on the local Ollama daemon,
    # so a real run needs no API key.
    print("\nQuick test:")
    print("  python example_usage.py")
    print("\nThe console brings the corpus up to date when it starts, and every run")
    print("checks it again before the Architect opens.")

    # Silent, but not mute: a caller reading only the exit status -- a CI
    # step, an install check -- must not read "verified" from a run that
    # just printed a failure.
    if any(r.startswith("✗") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
