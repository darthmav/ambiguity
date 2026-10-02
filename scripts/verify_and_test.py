#!/usr/bin/env python3
"""Verification for the 4-Agent System: dependencies, seats, search, the suite.

Usage:
    python scripts/verify_and_test.py                # read-only checks
    python scripts/verify_and_test.py --all          # plus a live example run

With no flags this runs only the steps that leave the machine alone. The
example run is a live agent run that writes a generated project under
`projects/`, so it is asked for by name or with `--all`. Nothing here rebuilds
the corpus: the console does that.
"""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(text: str) -> None:
    """Print a formatted step."""
    print(f"\n► {text}")


def check_dependencies() -> bool:
    """Every dependency `pyproject.toml` declares is installed.

    Read from the declaration itself rather than a list kept here, which
    drifted: it went on requiring a package the project had dropped.
    """
    print_header("STEP 1: Checking Dependencies")

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    missing = []
    for requirement in declared["project"]["dependencies"]:
        name = re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0]
        try:
            print(f"  ✓ {name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"  ✗ {name}")
            missing.append(name)

    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with: pip install -e '.[dev]'")
        return False

    print("\n✓ All dependencies installed")
    return True


def check_seats() -> bool:
    """Report what each seat will actually run.

    Read from `get_agent_status`, because key presence is not liveness: a seat
    with no key silently becomes StubLLM while every static check still reports
    the configured model, and a stubbed seat (canned text) and a failing one
    (a dead run) are different failures.
    """
    print_step("Checking seats")

    # Imported here, so a missing dependency is reported by step 1 rather than
    # by a traceback at import.
    try:
        from langgraph_agent.config import AGENTS, get_agent_status
    except Exception as e:  # pragma: no cover - step 1 already reports this
        print(f"  ✗ Could not read the seat configuration: {e}")
        return False

    all_live = True
    for agent in AGENTS:
        seat = get_agent_status(agent)
        if seat["live"]:
            mark, note = "✓", ""
        else:
            all_live = False
            mark = "○" if seat["stubbed"] else "✗"
            note = f"   !! {seat['badge']}: {seat['reason']}"
        print(
            f"  {mark} {agent:<11}{seat['model']:<24}"
            f"{seat['provider']:<11}{seat['placement']}{note}"
        )

    if all_live:
        print("\n✓ All seats live")
    else:
        print(
            "\n  ⚠ Not every seat can run. A stubbed seat (○) completes the run "
            "with canned text; a failing seat (✗) kills it."
        )
    return all_live


def test_graphrag_search() -> bool:
    """Test GraphRAG search functionality."""
    print_header("STEP 2: Testing GraphRAG Search")

    try:
        from langgraph_agent.graphrag_server import open_knowledge_base

        print_step("Loading knowledge base")
        # Opened, never created: checking search must not leave a corpus behind.
        kb = open_knowledge_base()
        if kb is None:
            print("  \u2717 No corpus has been indexed; nothing to search.")
            print("    The corpus is the research archive: upload a document,")
            print("    research a goal online, or embed a generated project.")
            return False

        print_step("Testing search queries")

        queries = [
            "What is the Planner agent?",
            "How does GraphRAG work?",
            "What are the system requirements?",
        ]

        for query in queries:
            print(f"\n  Query: '{query}'")
            results = kb.search(query, top_k=2)

            if results:
                print(f"    Found {len(results)} result(s)")
                for result in results[:1]:
                    print(f"    - Score: {result['score']:.3f}")
                    print(f"      Source: {result['id']}")
            else:
                print("    No results found")

        print("\n✓ GraphRAG search working")
        return True

    except Exception as e:
        print(f"\n✗ GraphRAG test failed: {e}")
        return False


def run_example_usage() -> bool:
    """Run the example usage script."""
    print_header("STEP 3: Running Example Usage")

    example_path = ROOT / "example_usage.py"
    if not example_path.exists():
        print(f"  ✗ Example script not found: {example_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        print(result.stdout)
        if result.stderr:
            print("Output:", result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ Example timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def run_tests() -> bool:
    """Run the test suite."""
    print_header("STEP 4: Running Tests")

    try:
        import pytest

        # pyproject's `pythonpath` puts the root on the path for the tests that
        # import `serve` and `spectral_graph`, from here as from anywhere.
        print_step("Running pytest")
        exit_code = pytest.main(["-v", "--tb=short", str(ROOT / "tests")])

        return exit_code == 0
    except Exception as e:
        print(f"  ✗ Error running tests: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify and test the 4-Agent System"
    )
    parser.add_argument(
        "--test-graphrag",
        action="store_true",
        help="Test GraphRAG search",
    )
    parser.add_argument(
        "--run-example",
        action="store_true",
        help="Run a live agent run (writes files into the repository)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test suite",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every step, including the one that writes",
    )

    args = parser.parse_args()

    # No flags runs the read-only steps; the one that writes is asked for by
    # name or by --all.
    read_only = args.all or not any(
        [args.test_graphrag, args.run_example, args.run_tests]
    )

    # Every step runs even after one fails, so one broken step does not hide the
    # rest, and the exit code carries the result.
    failed: list[str] = []

    print_header("4-AGENT SYSTEM VERIFICATION")

    # Step 1: Always check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Reported, not judged: a stubbed seat still completes a run.
    check_seats()

    # Step 2: Test GraphRAG
    if args.test_graphrag or read_only:
        if not test_graphrag_search():
            print("\n⚠ GraphRAG test failed, continuing anyway...")
            failed.append("GraphRAG search")

    # Step 3: Run example
    if args.run_example or args.all:
        if not run_example_usage():
            print("\n⚠ Example failed, continuing anyway...")
            failed.append("example run")

    # Step 4: Run tests
    if args.run_tests or read_only:
        if not run_tests():
            print("\n⚠ Some tests failed")
            failed.append("tests")

    print_header("VERIFICATION COMPLETE")

    if failed:
        print("\n\u2717 Failed: " + ", ".join(failed))
        sys.exit(1)

    print("\nNext steps:")
    print("  - Review output above for any issues")
    print("  - Any seat not live above: start the Ollama daemon and pull its tag;")
    print("    a :cloud tag also needs `ollama signin`, and an Anthropic seat")
    print("    ANTHROPIC_API_KEY in .env")
    print("  - Run: python example_usage.py")


if __name__ == "__main__":
    main()
