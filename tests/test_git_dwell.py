"""Tests for `git_dwell`, the ordered git pipeline, and the Builder's side of it.

Every test drives a real temporary repository, and most of them a real bare
remote beside it. `git` is real here -- the pipeline's whole job is to run
commands in the right order and report what happened, and a mock would be
testing the mock. `gh` is not: a pull request needs GitHub, so `pr`, `checks`
and `merge` talk to a stub `gh` on PATH that keeps a small GitHub in a JSON
file -- pull requests, their checks, their review state -- and merges the way
GitHub does, squashing the head onto the base branch of the bare remote. Its
log is how a test sees what the pipeline asked for.

What matters most are the refusals: it never commits onto the default branch,
never merges past a check that failed or one it did not read, and never
overwrites someone else's push. The merge used to be the last stage of the
default and stopped there; the default now carries on to the cleanup, and the
review point a caller asks for by leaving `merge` out is pinned beside it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import langgraph_agent.dwell as dwell
from langgraph_agent.dwell import (
    DWELL_DEFAULT_STAGES,
    DWELL_STAGES,
    _branch_name_from,
    finish_pull_request,
)
from langgraph_agent.mcp_client import MCPClient


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", *argv], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _identify(repo: Path) -> None:
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A real git repository with one commit on `main`, and it is the cwd.

    Everything else a test makes -- the remote, the stub's directory -- lives
    beside it, never inside it, where `git add -A` would sweep it up.
    """
    path = tmp_path / "repo"
    path.mkdir()
    _run("git", "init", "-b", "main", str(path), cwd=path)
    _identify(path)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run("git", "add", "seed.txt", cwd=path)
    _run("git", "commit", "-m", "seed", cwd=path)
    monkeypatch.chdir(path)
    return path


@pytest.fixture
def origin(repo, tmp_path) -> Path:
    """A bare repository as `origin`, holding `main`."""
    bare = tmp_path / "origin.git"
    _run("git", "init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    _run("git", "remote", "add", "origin", str(bare), cwd=repo)
    _run("git", "push", "-u", "origin", "main", cwd=repo)
    _run("git", "remote", "set-head", "origin", "main", cwd=repo)
    return bare


def _other_clone(origin: Path, tmp_path: Path, name: str = "other") -> Path:
    """Someone else's checkout of the same remote."""
    other = tmp_path / name
    _run("git", "clone", str(origin), str(other), cwd=tmp_path)
    _identify(other)
    return other


def _hook(origin: Path, body: str) -> None:
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    hook.chmod(0o755)


_GH_STUB = r'''
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
WORLD = HERE / "world.json"
args = sys.argv[1:]
with (HERE / "gh.log").open("a") as log:
    log.write(" ".join(args) + "\n")
world = json.loads(WORLD.read_text()) if WORLD.exists() else {}
prs = world.setdefault("prs", [])


def save():
    WORLD.write_text(json.dumps(world))


def git(*argv):
    return subprocess.run(["git", *argv], capture_output=True, text=True).stdout.strip()


def tip(branch):
    found = git("ls-remote", "origin", f"refs/heads/{branch}").split()
    return found[0] if found else ""


def find(target):
    if target.isdigit():
        return next((pr for pr in prs if pr["number"] == int(target)), None)
    mine = [pr for pr in prs if pr["headRefName"] == target]
    still_open = [pr for pr in mine if pr["state"] == "OPEN"]
    return (still_open or mine or [None])[-1]


def current(pr):
    # GitHub's head is the branch's tip on the remote while the PR is open.
    if pr["state"] == "OPEN":
        pr["headRefOid"] = tip(pr["headRefName"]) or pr["headRefOid"]
    return pr


command = args[:2]
if command == ["pr", "view"]:
    pr = find(args[2])
    if pr is None:
        print("no pull requests found for branch", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(current(pr)))
    save()
elif command == ["pr", "create"]:
    flags = dict(zip(args[2::2], args[3::2]))
    number = len(prs) + 1
    pr = {
        "number": number, "url": f"https://github.com/acme/demo/pull/{number}",
        "state": "OPEN", "isDraft": False, "title": flags["--title"],
        "headRefName": flags["--head"], "baseRefName": flags["--base"], "headRefOid": "",
        "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "reviewDecision": "",
        "mergeCommit": None,
    }
    pr.update(world.get("new_pr", {}))
    prs.append(pr)
    save()
    print(pr["url"])
elif command == ["pr", "checks"]:
    polls = world.get("check_polls")
    if not polls:
        print("no checks reported on the 'x' branch", file=sys.stderr)
        sys.exit(1)
    reading = polls.pop(0) if len(polls) > 1 else polls[0]
    save()
    print(json.dumps(reading))
elif command == ["pr", "merge"]:
    pr = current(find(args[2]))
    if world.get("merge_error"):
        print(world["merge_error"], file=sys.stderr)
        sys.exit(1)
    if "--auto" in args:
        pr["autoMerge"] = True
        save()
        sys.exit(0)
    head = args[args.index("--match-head-commit") + 1]
    if head != pr["headRefOid"]:
        print("GraphQL: Head branch was modified. Review and try the merge again.",
              file=sys.stderr)
        sys.exit(1)
    tree = git("rev-parse", f"{head}^{{tree}}")
    squash = git("commit-tree", tree, "-p", tip(pr["baseRefName"]),
                 "-m", f"{pr['title']} (#{pr['number']})")
    git("push", "origin", f"{squash}:refs/heads/{pr['baseRefName']}")
    pr.update(state="MERGED", mergeCommit={"oid": squash})
    save()
elif command == ["run", "view"]:
    print(world.get("run_log", ""))
'''


class _GitHub:
    """The stub's world, and what was asked of it."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def set(self, **fields: Any) -> None:
        world = self.world()
        world.update(fields)
        (self.dir / "world.json").write_text(json.dumps(world), encoding="utf-8")

    def world(self) -> dict[str, Any]:
        path = self.dir / "world.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def prs(self) -> list[dict[str, Any]]:
        return list(self.world().get("prs", []))

    def calls(self) -> str:
        log = self.dir / "gh.log"
        return log.read_text(encoding="utf-8") if log.exists() else ""


@pytest.fixture
def github(tmp_path, monkeypatch) -> _GitHub:
    """A stub `gh` on PATH, backed by a JSON GitHub with no pull requests yet."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    script = bin_dir / "gh"
    script.write_text(f"#!{sys.executable}\n{_GH_STUB}", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return _GitHub(bin_dir)


@pytest.fixture(autouse=True)
def fast(monkeypatch) -> None:
    """GitHub's settling times, shrunk: the stub answers at once."""
    for name, seconds in {
        "CHECKS_POLL_SECONDS": 0.01,
        "CHECKS_REGISTER_SECONDS": 0.3,
        "HEAD_SETTLE_SECONDS": 0.3,
        "MERGEABILITY_SETTLE_SECONDS": 0.2,
        "SETTLE_POLL_SECONDS": 0.01,
    }.items():
        monkeypatch.setattr(dwell, name, seconds)


@pytest.fixture
def client() -> MCPClient:
    return MCPClient()


def _dwell(client: MCPClient, **args: Any) -> dict[str, Any]:
    return client.call_tool("git_dwell", args)


def _stage_of(result: dict[str, Any], stage: str) -> dict[str, Any]:
    return next(entry for entry in result["stages"] if entry["stage"] == stage)


def _check(name: str, bucket: str, link: str = "") -> dict[str, str]:
    return {"name": name, "bucket": bucket, "state": bucket.upper(), "link": link,
            "workflow": "CI"}


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------


def test_it_branches_rather_than_committing_onto_the_default(repo, client):
    """Committing onto main removes the review point before anyone can use it."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(
        client, message="feat: add a thing", stages=["survey", "branch", "stage", "commit"]
    )

    assert result["success"], result
    assert result["branch"] != "main"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == result["branch"]
    # main is untouched: its tip is still the seed commit.
    assert _git(repo, "log", "-1", "--format=%s", "main") == "seed"


def test_it_refuses_to_commit_on_the_default_when_branch_is_skipped(repo, client):
    """Dropping the branch stage must not become a way onto main.

    The guarantee cannot be "we branch for you unless you ask us not to" -- an
    agent picking its own stage list would find the gap immediately.
    """
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: sneak", stages=["stage", "commit"])

    assert not result["success"]
    assert result["stopped_at"] == "branch"
    assert "refusing to commit onto main" in result["error"]
    assert _git(repo, "log", "-1", "--format=%s", "main") == "seed"


def test_a_detached_head_is_branched_from_rather_than_committed_on(repo, client):
    """A commit on a detached HEAD belongs to no branch, and `push -u origin
    HEAD` would make one called HEAD."""
    _run("git", "checkout", "--detach", cwd=repo)
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    refused = _dwell(client, message="feat: x", stages=["stage", "commit"])
    assert refused["stopped_at"] == "branch"

    result = _dwell(client, message="feat: x", stages=["branch", "stage", "commit"])
    assert result["success"], result
    assert result["branch"] == "agent/x"
    assert "detached HEAD" in _stage_of(result, "branch")["detail"]


# ---------------------------------------------------------------------------
# the whole flow
# ---------------------------------------------------------------------------


def test_the_default_carries_the_change_onto_main_and_tidies_up(repo, origin, client, github):
    """Branch to merge, through the checks, then the branch removed there and here
    and main here brought up to what landed -- the step the user was left to do
    by hand when the flow ended at the merge."""
    assert "merge" in DWELL_DEFAULT_STAGES and "cleanup" in DWELL_DEFAULT_STAGES
    github.set(check_polls=[[_check("lint", "pass"), _check("test", "pass")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert result["merged"] and result["merge_commit"]
    assert [entry["stage"] for entry in result["stages"]] == list(DWELL_STAGES)
    assert "2 check(s) passed" in _stage_of(result, "checks")["detail"]
    # Landed on the remote's main as one squashed commit...
    assert _git(origin, "log", "-1", "--format=%s", "main") == "feat: a thing (#1)"
    # ...and here: back on main, at that commit, the branch gone everywhere.
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(repo, "rev-parse", "HEAD") == result["merge_commit"]
    assert (repo / "new.txt").exists()
    assert not _git(repo, "branch", "--list", "agent/a-thing")
    assert not _git(origin, "branch", "--list", "agent/a-thing")
    assert not result["warnings"]


def test_a_caller_can_still_stop_at_the_pull_request(repo, origin, client, github):
    """Naming stages without `merge` leaves the PR open for someone to read.

    The opt-out that replaced the old default, and the half worth pinning: a
    default can change again, while a caller that asked for the review point
    must keep getting it.
    """
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(
        client, message="feat: a thing",
        stages=["survey", "branch", "stage", "commit", "push", "pr"],
    )

    assert result["success"], result
    assert "pr create" in github.calls()
    assert "pr merge" not in github.calls(), "a stage list that omits merge must not merge"
    assert github.prs()[0]["state"] == "OPEN"


def test_an_open_pull_request_is_reused(repo, origin, client, github):
    (repo / "new.txt").write_text("work\n", encoding="utf-8")
    stages = ["survey", "branch", "stage", "commit", "push", "pr"]
    first = _dwell(client, message="feat: a thing", stages=stages)
    (repo / "more.txt").write_text("more\n", encoding="utf-8")

    second = _dwell(client, message="feat: more", stages=stages)

    assert "already open" in _stage_of(second, "pr")["detail"]
    assert second["pull_request"]["number"] == first["pull_request"]["number"]
    assert github.calls().count("pr create") == 1


def test_with_no_remote_the_commit_stays_here_and_that_is_success(repo, client):
    """A project the Builder `git init`ed has no remote. Its commit is the
    product; reporting the push as a failure sent the Builder off to repair a
    repository that was never broken."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: local")

    assert result["success"], result
    assert result["local_only"]
    assert "no remote named origin" in _stage_of(result, "update")["detail"]
    assert not any(e["stage"] in ("push", "pr", "merge") for e in result["stages"])
    assert _git(repo, "log", "-1", "--format=%s") == "feat: local"


def test_a_new_repository_s_first_commit_founds_its_branch(repo, client):
    """`git init` then git_dwell used to stop at survey: an unborn HEAD has no
    commit to resolve. Its first commit has nothing to branch from, so it
    founds the branch git init named."""
    project = repo / "projects" / "fresh"
    project.mkdir(parents=True)
    _run("git", "init", "-b", "trunk", str(project), cwd=project)
    _identify(project)
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = _dwell(client, message="feat: first", cwd="projects/fresh")

    assert result["success"], result
    assert result["branch"] == "trunk"
    assert "first commit founds it" in _stage_of(result, "branch")["detail"]
    assert _git(project, "log", "--format=%s") == "feat: first"
    assert _git(project, "branch", "--format=%(refname:short)") == "trunk"


def test_nothing_to_commit_is_not_a_failure(repo, client):
    """A clean tree is an ordinary outcome, not a repository to repair."""
    result = _dwell(
        client, message="feat: nothing", stages=["survey", "branch", "stage", "commit", "push"]
    )

    assert result["success"], result
    commit = _stage_of(result, "commit")
    assert commit["ok"] and "nothing staged" in commit["detail"]
    assert not any(entry["stage"] == "push" for entry in result["stages"])


def test_nothing_new_still_ships_a_branch_that_has_commits(repo, origin, client, github):
    """A second call after the checks were left running -- or after a fix was
    committed by hand -- has nothing to commit, and still everything else to do."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")
    _dwell(client, message="feat: a thing", stages=["branch", "stage", "commit"])

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert "already carries 1 commit(s)" in _stage_of(result, "commit")["detail"]
    assert result["merged"]


# ---------------------------------------------------------------------------
# update and push: other people's commits are merged in, never overwritten
# ---------------------------------------------------------------------------


def test_update_merges_the_default_branch_in_before_the_push(
    repo, origin, client, github, tmp_path
):
    other = _other_clone(origin, tmp_path)
    (other / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _run("git", "add", "theirs.txt", cwd=other)
    _run("git", "commit", "-m", "theirs", cwd=other)
    _run("git", "push", "origin", "main", cwd=other)
    (repo / "mine.txt").write_text("mine\n", encoding="utf-8")

    result = _dwell(client, message="feat: mine", stages=list(DWELL_STAGES[:6]))

    assert result["success"], result
    assert "merged origin/main in" in _stage_of(result, "update")["detail"]
    pushed = _git(origin, "ls-tree", "--name-only", result["branch"]).split()
    assert {"mine.txt", "theirs.txt"} <= set(pushed)


def test_an_update_that_conflicts_is_undone_and_names_the_files(
    repo, origin, client, github, tmp_path
):
    other = _other_clone(origin, tmp_path)
    (other / "seed.txt").write_text("theirs\n", encoding="utf-8")
    _run("git", "commit", "-am", "theirs", cwd=other)
    _run("git", "push", "origin", "main", cwd=other)
    (repo / "seed.txt").write_text("mine\n", encoding="utf-8")

    result = _dwell(client, message="feat: mine")

    assert not result["success"]
    assert result["stopped_at"] == "update"
    assert "conflicts in seed.txt" in result["error"]
    assert not (repo / ".git" / "MERGE_HEAD").exists(), "the merge must be undone"
    assert not _git(origin, "branch", "--list", result["branch"]), "nothing was pushed"


def test_a_push_refused_because_the_branch_moved_merges_it_in(
    repo, origin, client, github, tmp_path
):
    """Someone pushed to the same branch: their commit is kept, never forced over."""
    (repo / "mine.txt").write_text("mine\n", encoding="utf-8")
    first = _dwell(client, message="feat: mine", stages=["branch", "stage", "commit", "push"])
    other = _other_clone(origin, tmp_path)
    _run("git", "checkout", first["branch"], cwd=other)
    (other / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _run("git", "add", "theirs.txt", cwd=other)
    _run("git", "commit", "-m", "theirs", cwd=other)
    _run("git", "push", "origin", first["branch"], cwd=other)
    theirs = _git(other, "rev-parse", "HEAD")
    (repo / "more.txt").write_text("more\n", encoding="utf-8")

    result = _dwell(client, message="feat: more", stages=["stage", "commit", "push"])

    assert result["success"], result
    assert "had moved: merged it in first" in _stage_of(result, "push")["detail"]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", theirs, result["branch"]], cwd=origin
    )
    assert ancestry.returncode == 0, "their commit must still be on the branch"


def test_a_failing_stage_names_itself_and_stops_the_rest(repo, origin, client, github):
    """A push the remote refuses -- a hook, a protection rule -- is not merged
    around or sent again: it stops there and says so."""
    _hook(origin, 'echo "declined by policy" >&2\nexit 1\n')
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert not result["success"]
    assert result["stopped_at"] == "push"
    assert "declined" in result["error"]
    assert [entry["stage"] for entry in result["stages"]][-1] == "push"
    assert "pr create" not in github.calls(), "stages after the failure must not run"


# ---------------------------------------------------------------------------
# checks and merge
# ---------------------------------------------------------------------------


def test_failing_checks_stop_it_with_the_jobs_and_their_log(repo, origin, client, github):
    link = "https://github.com/acme/demo/actions/runs/7/job/42"
    github.set(
        check_polls=[[_check("lint", "pass"), _check("test (3.12)", "fail", link)]],
        run_log="FAILED tests/test_x.py::test_y - AssertionError: 2 != 3",
    )
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert not result["success"]
    assert result["stopped_at"] == "checks"
    assert result["checks_failed"] == ["CI / test (3.12)"]
    assert "AssertionError: 2 != 3" in result["failed_log"]
    assert "run view --job 42 --log-failed" in github.calls()
    assert "pr merge" not in github.calls()


def test_checks_still_running_leave_it_pending_rather_than_failed(
    repo, origin, client, github
):
    """Past its wait the pull request is left open, for the console to finish:
    CI that takes longer than a Builder's deadline is not a broken change."""
    github.set(check_polls=[[_check("lint", "pass"), _check("test", "pending")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing", checks_timeout=0.05)

    assert result["success"], result
    assert "waiting on test" in result["pending"]
    assert "pr merge" not in github.calls()
    assert not any(entry["stage"] == "cleanup" for entry in result["stages"])
    assert github.prs()[0]["state"] == "OPEN"


def test_a_merge_never_lands_on_checks_it_did_not_read(repo, origin, client, github):
    """Leaving `checks` out of the stages is not a way past a red one."""
    github.set(check_polls=[[_check("test", "fail")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(
        client, message="feat: a thing",
        stages=["branch", "stage", "commit", "push", "pr", "merge"],
    )

    assert not result["success"]
    assert result["stopped_at"] == "merge"
    assert "its checks fail" in result["error"]
    assert "pr merge" not in github.calls()


def test_the_merge_is_of_exactly_the_commit_whose_checks_were_read(
    repo, origin, client, github
):
    github.set(check_polls=[[_check("test", "pass")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert f"pr merge 1 --squash --match-head-commit {result['head']}" in github.calls()


@pytest.mark.parametrize(
    ("override", "waiting"),
    [
        ({"reviewDecision": "CHANGES_REQUESTED"}, "requested changes"),
        ({"reviewDecision": "REVIEW_REQUIRED"}, "approving review"),
        ({"isDraft": True}, "is a draft"),
    ],
)
def test_people_are_waited_on_not_merged_past(repo, origin, client, github, override, waiting):
    github.set(new_pr=override)
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert waiting in result["pending"]
    assert "pr merge" not in github.calls()


def test_a_conflicting_pull_request_is_not_merged(repo, origin, client, github):
    github.set(new_pr={"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY"})
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert not result["success"]
    assert result["stopped_at"] == "merge"
    assert "conflicts with main" in result["error"]


def test_branch_protection_hands_the_merge_to_github(repo, origin, client, github):
    github.set(new_pr={"mergeStateStatus": "BLOCKED"})
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert "auto-merge is on" in result["pending"]
    assert "--auto --squash --match-head-commit" in github.calls()


# ---------------------------------------------------------------------------
# a finished branch, and the cleanup
# ---------------------------------------------------------------------------


def test_a_branch_whose_pull_request_merged_starts_fresh(repo, origin, client, github):
    """Commits on a merged branch would reach a pull request GitHub already
    squashed away; the work moves to a new branch from the remote's main."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")
    shipped = _dwell(client, message="feat: a thing", stages=list(DWELL_STAGES[:9]))
    assert shipped["merged"], shipped
    # Still on the finished branch, as a caller who stopped short of cleanup is.
    (repo / "follow-up.txt").write_text("more\n", encoding="utf-8")

    result = _dwell(client, message="feat: a follow-up")

    assert result["success"], result
    assert result["branch"] == "agent/a-follow-up"
    assert "has already merged" in _stage_of(result, "branch")["detail"]
    assert result["merged"]
    assert _git(origin, "log", "-1", "--format=%s", "main") == "feat: a follow-up (#2)"


def test_a_remote_branch_that_cannot_be_deleted_is_a_warning_not_a_failure(
    repo, origin, client, github
):
    """The merge is what counted. This is the case of the branch that would not
    delete after the last dwell: say so, and point at the button."""
    _hook(origin, (
        "while read old new ref; do\n"
        '  case "$new" in 0000000000000000000000000000000000000000)\n'
        '    echo "deleting $ref is not allowed" >&2; exit 1;;\n'
        "  esac\n"
        "done\n"
    ))
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing")

    assert result["success"], result
    assert result["merged"]
    assert any("origin/agent/a-thing remains" in warning for warning in result["warnings"])
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_the_console_finishes_what_a_run_left_pending(repo, origin, client, github):
    github.set(check_polls=[[_check("test", "pending")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")
    left = _dwell(client, message="feat: a thing", checks_timeout=0)
    assert left["pending"], left
    _run("git", "checkout", "main", cwd=repo)  # the operator moved on meanwhile
    github.set(check_polls=[[_check("test", "pass")]])

    finished = finish_pull_request(
        client._run_vcs, cwd=None, branch=left["branch"],
        number=left["pull_request"]["number"], head=left["head"],
    )

    assert finished["success"], finished
    assert finished["merged"]
    assert _git(origin, "log", "-1", "--format=%s", "main") == "feat: a thing (#1)"


def test_the_console_never_merges_a_head_it_was_not_given(
    repo, origin, client, github, tmp_path
):
    """A push after the run is someone else's work, checked by nobody here."""
    github.set(check_polls=[[_check("test", "pending")]])
    (repo / "new.txt").write_text("work\n", encoding="utf-8")
    left = _dwell(client, message="feat: a thing", checks_timeout=0)
    other = _other_clone(origin, tmp_path)
    _run("git", "checkout", left["branch"], cwd=other)
    (other / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _run("git", "add", "theirs.txt", cwd=other)
    _run("git", "commit", "-m", "theirs", cwd=other)
    _run("git", "push", "origin", left["branch"], cwd=other)
    github.set(check_polls=[[_check("test", "pass")]])

    finished = finish_pull_request(
        client._run_vcs, cwd=None, branch=left["branch"],
        number=left["pull_request"]["number"], head=left["head"],
    )

    assert not finished["success"]
    assert finished["stopped_at"] == "checks"
    assert "pr merge" not in github.calls()


# ---------------------------------------------------------------------------
# ordering and reporting
# ---------------------------------------------------------------------------


def test_stages_run_in_pipeline_order_however_they_are_given(repo, client):
    """"push then commit" is a typo, not an instruction to do it backwards."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(
        client, message="feat: ordered", stages=["commit", "stage", "branch", "survey"]
    )

    assert result["success"], result
    assert [e["stage"] for e in result["stages"]] == ["survey", "branch", "stage", "commit"]


def test_an_unknown_stage_is_refused_by_name(repo, client):
    """A typo must not silently run a shorter pipeline than the caller meant."""
    result = _dwell(client, message="m", stages=["survey", "rebase"])

    assert not result["success"]
    assert "rebase" in result["error"]
    assert ", ".join(DWELL_STAGES) in result["error"]


def test_commit_without_a_message_is_refused(repo, client):
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, stages=["survey", "branch", "stage", "commit"])

    assert not result["success"]
    assert result["stopped_at"] == "commit"
    assert "message" in result["error"]


def test_only_the_named_paths_are_committed(repo, client):
    """A Builder sharing a tree with the operator must not sweep up their work."""
    (repo / "mine.txt").write_text("mine\n", encoding="utf-8")
    (repo / "theirs.txt").write_text("theirs\n", encoding="utf-8")

    _dwell(
        client,
        message="feat: just mine",
        paths=["mine.txt"],
        stages=["survey", "branch", "stage", "commit"],
    )

    assert _git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["mine.txt"]


def test_the_named_paths_are_all_a_commit_takes_whatever_else_is_staged(repo, client):
    """`paths` was only what got staged: the commit was a bare `git commit`,
    which takes the whole index, so whatever the operator had staged went into
    the commit, the pull request and the merge."""
    (repo / "mine.txt").write_text("mine\n", encoding="utf-8")
    (repo / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _run("git", "add", "theirs.txt", cwd=repo)  # the operator's, staged already

    result = _dwell(
        client,
        message="feat: just mine",
        paths=["mine.txt"],
        stages=["survey", "branch", "stage", "commit"],
    )

    assert result["success"], result
    assert _git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["mine.txt"]
    assert _git(repo, "diff", "--cached", "--name-only").split() == ["theirs.txt"]


def test_a_directory_inside_another_repository_is_refused(repo, client):
    """git looks upward for a repository, so a project directory without its
    own would have committed -- and merged -- the checkout around it."""
    project = repo / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = _dwell(client, message="feat: demo", cwd="projects/demo")

    assert not result["success"]
    assert "not a git repository of its own" in result["error"]
    assert len(_git(repo, "log", "--oneline").splitlines()) == 1  # the seed alone


def test_in_a_project_s_own_repository_it_commits_there_and_only_there(repo, client):
    project = repo / "projects" / "demo"
    project.mkdir(parents=True)
    _run("git", "init", "-b", "main", str(project), cwd=project)
    _identify(project)
    (project / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run("git", "add", "seed.txt", cwd=project)
    _run("git", "commit", "-m", "seed", cwd=project)
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (repo / "operator.txt").write_text("the operator's\n", encoding="utf-8")

    result = _dwell(
        client,
        message="feat: demo",
        cwd="projects/demo",
        paths=["projects/demo/main.py"],  # spelled from the root, like every tool
        stages=["survey", "branch", "stage", "commit"],
    )

    assert result["success"], result
    assert _git(project, "show", "--name-only", "--format=", "HEAD").split() == ["main.py"]
    assert "operator.txt" in _git(repo, "status", "--porcelain")  # never staged


def test_a_path_outside_the_repository_it_runs_in_is_refused(repo, client):
    project = repo / "projects" / "demo"
    project.mkdir(parents=True)
    _run("git", "init", "-b", "main", str(project), cwd=project)

    result = _dwell(client, message="feat: x", cwd="projects/demo", paths=["seed.txt"])

    assert not result["success"]
    assert "outside projects/demo" in result["error"]


# ---------------------------------------------------------------------------
# the Builder's side: the cwd, the wait, the proof, the record
# ---------------------------------------------------------------------------


class _OneCall:
    """A Builder seat that makes the given tool calls once, then reports."""

    def __init__(self, *calls: dict[str, Any]) -> None:
        self.calls = list(calls)
        self.turns = 0

    def invoke(self, messages: Any) -> Any:
        self.turns += 1
        if self.turns == 1:
            return SimpleNamespace(content="", tool_calls=self.calls)
        return SimpleNamespace(content="done", tool_calls=[])


def test_on_a_project_run_the_builder_s_git_tools_act_in_the_project(monkeypatch):
    """Whatever cwd the model asks for, or none."""
    import langgraph_agent.nodes as nodes

    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        nodes, "_call_tool", lambda name, args: seen.append((name, args)) or {"success": True}
    )
    nodes._run_builder_tools(
        _OneCall(
            {"name": "git_dwell", "args": {"message": "m"}, "id": "c1"},
            {"name": "git_status", "args": {"cwd": "."}, "id": "c2"},
        ),
        [], [], [], nodes._Deadline(30), output_dir="projects/demo",
    )

    assert [(name, args["cwd"]) for name, args in seen] == [
        ("git_dwell", "projects/demo"), ("git_status", "projects/demo"),
    ]


def test_the_checks_wait_ends_inside_the_builder_s_deadline(monkeypatch):
    """A tool call is never abandoned, so the wait must end on its own, with
    room left for the merge and the cleanup after it."""
    import langgraph_agent.nodes as nodes

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        nodes, "_call_tool", lambda name, args: seen.append(args) or {"success": True}
    )
    nodes._run_builder_tools(
        _OneCall({"name": "git_dwell", "args": {"message": "m", "checks_timeout": 9999},
                  "id": "c1"}),
        [], [], [], nodes._Deadline(100),
    )

    assert 0 < seen[0]["checks_timeout"] <= 100 - nodes.DWELL_TAIL_RESERVE_SECONDS
    assert nodes._checks_budget("soon", nodes._Deadline(1000)) == dwell.DWELL_CHECKS_WAIT_SECONDS
    assert nodes._checks_budget(float("nan"), nodes._Deadline(10)) == 0.0


def _builder_state(**fields: Any) -> Any:
    from langgraph_agent import initial_state

    state = initial_state("ship it")
    state.update(fields)  # type: ignore[typeddict-item]
    return state


def test_a_dwell_that_would_ship_unproven_code_stops_at_prove(repo, monkeypatch):
    """The pass's proof runs after its tool loop, so a dwell called mid-pass
    would push a file nobody had linted or run. It is proven first."""
    import langgraph_agent.nodes as nodes

    (repo / "bad.py").write_text("import os\n", encoding="utf-8")  # F401
    real = MCPClient()
    shipped: list[dict[str, Any]] = []

    def call(name: str, args: dict[str, Any]) -> Any:
        if name == "git_dwell":
            shipped.append(args)
            return {"success": True}
        return real.call_tool(name, args)

    monkeypatch.setattr(nodes, "_call_tool", call)
    tool_log: list[str] = []
    dwells: list[dict[str, Any]] = []
    deadline = nodes._Deadline(60)
    files = ["bad.py"]
    nodes._run_builder_tools(
        _OneCall({"name": "git_dwell", "args": {"message": "feat: bad"}, "id": "c1"}),
        [], files, tool_log, deadline,
        gate=nodes._dwell_gate(_builder_state(), files, tool_log, deadline), dwells=dwells,
    )

    assert shipped == [], "nothing may reach git_dwell"
    assert dwells[0]["stopped_at"] == "prove"
    assert "bad.py fails lint (F401)" in dwells[0]["error"]
    assert any(line.startswith("git_dwell(default) -> failed") for line in tool_log)


def test_a_dwell_that_stays_local_is_not_held_to_the_proof(monkeypatch):
    import langgraph_agent.nodes as nodes

    def never(*args: Any) -> Any:
        raise AssertionError("a local commit must not run the proof")

    monkeypatch.setattr(nodes, "_prove", never)
    gate = nodes._dwell_gate(_builder_state(), ["bad.py"], [], nodes._Deadline(60))

    assert gate({"stages": ["branch", "stage", "commit"]}) == ""


@pytest.mark.parametrize(
    ("result", "status"),
    [
        ({"success": True, "merged": True, "pull_request": {"number": 3}}, "merged"),
        ({"success": False, "checks_failed": ["CI / test"], "stopped_at": "checks"},
         "checks_failed"),
        ({"success": True, "pending": "waiting on test", "pull_request": {"number": 3}},
         "pending"),
        ({"success": True, "local_only": True}, "local"),
        ({"success": True, "pull_request": {"number": 3, "url": "u"}}, "open"),
        ({"success": True, "pull_request": {}}, "committed"),
        ({"success": False, "stopped_at": "push", "error": "push: declined"}, "failed"),
    ],
)
def test_the_run_records_where_its_pull_request_stands(result, status):
    from langgraph_agent.nodes import _dwell_record

    assert _dwell_record(result)["status"] == status


class _Seat:
    """A Builder seat that dwells once, then reports."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return _OneCall({"name": "git_dwell", "args": {"message": "feat: x"}, "id": "c1"})


def test_the_builder_reports_its_pull_request_and_red_checks_block(repo, monkeypatch):
    import langgraph_agent.nodes as nodes

    red = {
        "success": False, "stopped_at": "checks", "error": "checks: CI / test failed",
        "checks_failed": ["CI / test"], "failed_log": "AssertionError: 2 != 3",
        "branch": "agent/x", "default_branch": "main",
        "pull_request": {"number": 7, "url": "https://github.com/acme/demo/pull/7"},
        "stages": [],
    }
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: _Seat())
    monkeypatch.setattr(nodes, "_call_tool", lambda name, args: dict(red))

    state = nodes.builder_node(_builder_state(plan="1. ship"))

    assert state["dwell"]["status"] == "checks_failed"
    assert state["dwell"]["number"] == 7
    assert "#7" in state["builder_report"] and "AssertionError" in state["builder_report"]
    assert "checks fail: CI / test" in state["blockers"]
    assert "Pull request: #7" in nodes._get_state_injection(state)


# ---------------------------------------------------------------------------
# branch naming
# ---------------------------------------------------------------------------


def test_a_branch_name_drops_the_conventional_commit_prefix():
    """`fix: contain writes` is a better branch as `contain-writes`."""
    assert _branch_name_from("fix: contain writes") == "agent/contain-writes"
    assert _branch_name_from("feat(api): add thing") == "agent/add-thing"


def test_a_branch_name_survives_a_message_git_would_refuse():
    """The name reaches a remote, so it is reduced rather than cleverly kept."""
    assert _branch_name_from("Fix: the ~weird~ thing!! (again)") == "agent/the-weird-thing-again"
    assert _branch_name_from("") == "agent/change"
    assert _branch_name_from("???") == "agent/change"


def test_a_derived_branch_name_already_taken_gets_a_suffix(repo, client):
    _run("git", "branch", "agent/a-thing", cwd=repo)
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(client, message="feat: a thing", stages=["branch", "stage", "commit"])

    assert result["success"], result
    assert result["branch"] == "agent/a-thing-2"
