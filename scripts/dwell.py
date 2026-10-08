#!/usr/bin/env python3
"""The dwell tool: carry this folder's changes from the working tree to a merged PR.

The same ordered git pipeline the Builder's `git_dwell` runs
(`langgraph_agent.dwell`) -- survey, branch, stage, commit, update, push, pr,
checks, merge, cleanup -- driven from the command line and pinned to this
folder: the working directory is always the repository root, and every named
path is refused unless it resolves inside it, so only this folder and its
subfolders can ever be staged, from wherever the tool is invoked.

    python scripts/dwell.py "fix(embedder): the thing"
    python scripts/dwell.py "wip" --stages survey branch stage commit
    python scripts/dwell.py "docs" --paths README.md tests/test_claims.py

A full run ends on a merged, cleaned-up pull request, on *pending* with the
pull request left open when CI or a review is what it waits on (the console's
monitor finishes those), or on *local* / *committed* when there is no remote.
Nothing here rebases or force-pushes, and it never commits onto the default
branch. The exit code is 0 for every ordinary ending, 1 when a stage failed,
2 for a call the tool refused outright.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph_agent.dwell import DWELL_STAGES, git_dwell  # noqa: E402


def run_vcs(*argv: str, timeout: float = 60.0, cwd: str | None = None) -> tuple[bool, str]:
    """One git/gh invocation; `(ok, output)` with stderr folded in. No shell:
    a commit message containing `;` is a message.
    """
    try:
        done = subprocess.run(
            list(argv), capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL, cwd=cwd,
        )
    except FileNotFoundError:
        return False, f"{argv[0]} is not installed on this machine"
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(argv)} timed out after {timeout:g}s"
    return done.returncode == 0, ((done.stdout or "") + (done.stderr or "")).strip()


def this_repository() -> str:
    """Why the pin cannot hold, or "": this folder must be a repository of its
    own, or git would climb upward and act on whatever repository surrounds it.
    """
    ok, top = run_vcs("git", "rev-parse", "--show-toplevel", cwd=str(ROOT))
    if not ok or Path(top.splitlines()[0]).resolve() != ROOT:
        return (
            f"{ROOT} is not a git repository of its own ({top}); the dwell "
            "tool only ever saves this folder, and will not commit the "
            "repository around it"
        )
    return ""


def confined_paths(raw: list[str]) -> tuple[list[str], str]:
    """Each path spelled from the tool, repo-relative, or the refusal naming it.
    `paths` is the pipeline's commit pathspec, so one outside this folder would
    reach outside it -- refused here, before git is asked.
    """
    out: list[str] = []
    for name in raw:
        inside = Path(name).resolve()
        if inside != ROOT and ROOT not in inside.parents:
            return [], f"{name} is outside {ROOT}; the dwell tool only saves this folder"
        out.append(str(inside.relative_to(ROOT)) or ".")
    return out, ""


def run(argv: list[str] | None = None) -> tuple[int, dict]:
    """Parse the call, confine it to this folder, and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="dwell", description=__doc__.splitlines()[0] if __doc__ else None,
        epilog="Stages, in the only order they work in: " + ", ".join(DWELL_STAGES),
    )
    parser.add_argument("message", help="the commit message; it also names the branch")
    parser.add_argument("--paths", nargs="+", metavar="PATH",
                        help="stage and commit only these (default: this whole folder)")
    parser.add_argument("--stages", nargs="+", choices=DWELL_STAGES,
                        help="which of the pipeline to run (default: all of it)")
    parser.add_argument("--branch", metavar="NAME", help="work on this branch instead of a new one")
    parser.add_argument("--checks-timeout", type=float, default=None, metavar="SECONDS",
                        help="how long the checks stage waits on CI (default: the pipeline's own)")
    parser.add_argument("--json", action="store_true", help="print the whole account as JSON")
    args = parser.parse_args(argv)

    refusal = this_repository()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2, {"success": False, "error": refusal}
    paths, refusal = confined_paths(args.paths or ["."])
    if refusal:
        print(refusal, file=sys.stderr)
        return 2, {"success": False, "error": refusal}

    dwell_args: dict = {"message": args.message, "paths": paths}
    if args.stages:
        dwell_args["stages"] = args.stages
    if args.branch:
        dwell_args["branch"] = args.branch
    if args.checks_timeout is not None:
        dwell_args["checks_timeout"] = args.checks_timeout

    result = git_dwell(run_vcs, dwell_args, str(ROOT))
    if args.json:
        text = json.dumps(result, indent=2)
    elif result.get("success"):
        text = result.get("summary", "")
    else:
        text = f"failed at {result.get('stopped_at', '?')}: {result.get('error', '')}"
    print(text)
    return 0 if result.get("success") else 1, result


if __name__ == "__main__":
    sys.exit(run()[0])
