"""Tests for `git_dwell`, the ordered git pipeline.

Every test drives a real temporary repository. `git` is real here -- the
pipeline's whole job is to run commands in the right order and report what
happened, and a mock would be testing the mock. `gh` is not: opening a pull
request needs a remote and an account, so the `pr` and `merge` stages are
exercised through a stub `gh` placed on PATH, which records how it was called.

What matters most is the one thing the pipeline *refuses*: it never commits
onto the default branch. That is the guarantee a caller relies on, and the
reason the rest can be a default -- the work arrives on a branch, through a
pull request, with the diff and the checks attached, whatever else runs. The
merge used to be the second refusal and is now the last stage of the default;
the opt-out it became is pinned here beside it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from langgraph_agent.mcp_client import (
    DWELL_DEFAULT_STAGES,
    DWELL_STAGES,
    MCPClient,
    _branch_name_from,
)


def _run(*argv: str, cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A real git repository with one commit on `main`, and it is the cwd."""
    _run("git", "init", "-b", "main", str(tmp_path), cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=tmp_path)
    _run("git", "config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run("git", "add", "seed.txt", cwd=tmp_path)
    _run("git", "commit", "-m", "seed", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def gh(tmp_path, monkeypatch) -> Path:
    """A stub `gh` on PATH that records its arguments and reports no open PR."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    log = bin_dir / "gh.log"
    script = bin_dir / "gh"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        # `pr view` must fail, or the pipeline reads it as a PR already open.
        'case "$1 $2" in "pr view") exit 1;; esac\n'
        'echo "https://example.invalid/pr/1"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    return log


@pytest.fixture
def client() -> MCPClient:
    return MCPClient()


def _dwell(client: MCPClient, **args) -> dict:
    return client.call_tool("git_dwell", args)


# ---------------------------------------------------------------------------
# the two refusals
# ---------------------------------------------------------------------------


def test_it_branches_rather_than_committing_onto_the_default(repo, client):
    """Committing onto main removes the review point before anyone can use it."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    result = _dwell(
        client, message="feat: add a thing", stages=["survey", "branch", "stage", "commit"]
    )

    assert result["success"], result
    assert result["branch"] != "main"
    head = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.strip()
    assert head == result["branch"]
    # main is untouched: its tip is still the seed commit.
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", "main"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.strip()
    assert subject == "seed"


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
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", "main"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.strip()
    assert subject == "seed"


def test_merge_is_in_the_default_pipeline(repo, client, gh):
    """The default runs the flow to the end, merge included.

    This asserted the opposite until 2026-09-12, on the argument that a pull
    request the same agent opens and immediately merges is not a review. The
    argument holds; what it could not carry was the default. A tool that is one
    call because the flow is one act, stopping one stage short every time,
    leaves the last stage to a caller with no way to know it is outstanding --
    which is what happened, repeatedly. The review point is now something a
    caller asks for by naming its stages, which is the test below.
    """
    assert "merge" in DWELL_DEFAULT_STAGES
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    # Everything the default runs, minus the push that needs a real remote.
    _dwell(
        client, message="feat: a thing",
        stages=["survey", "branch", "stage", "commit", "pr", "merge"],
    )

    calls = gh.read_text(encoding="utf-8") if gh.exists() else ""
    assert "pr create" in calls
    assert "pr merge" in calls


def test_a_caller_can_still_stop_at_the_pull_request(repo, client, gh):
    """Naming stages without `merge` leaves the PR open for someone to read.

    This is the opt-out that replaced the old default, and it is the half worth
    pinning: a default can be changed again, while a caller that asked for the
    review point must keep getting it.
    """
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    _dwell(
        client, message="feat: a thing",
        stages=["survey", "branch", "stage", "commit", "pr"],
    )

    calls = gh.read_text(encoding="utf-8") if gh.exists() else ""
    assert "pr create" in calls
    assert "pr merge" not in calls, "a stage list that omits merge must not merge"


def test_merge_runs_only_when_named(repo, client, gh):
    """The opt-in has to actually opt in, or the stage is merely deleted."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    _dwell(
        client,
        message="feat: a thing",
        stages=["survey", "branch", "stage", "commit", "pr", "merge"],
    )

    assert "pr merge" in gh.read_text(encoding="utf-8")


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


def test_nothing_to_commit_is_not_a_failure(repo, client):
    """A clean tree is an ordinary outcome, not a repository to repair.

    Failing here would send the Builder off fixing something that was never
    broken -- the false accusation this tool belt keeps having to design
    against.
    """
    result = _dwell(
        client, message="feat: nothing", stages=["survey", "branch", "stage", "commit", "push"]
    )

    assert result["success"], result
    commit = next(e for e in result["stages"] if e["stage"] == "commit")
    assert commit["ok"] and "nothing staged" in commit["detail"]
    # push is dropped rather than attempted against a remote that is not there.
    assert not any(e["stage"] == "push" for e in result["stages"])


def test_a_failing_stage_names_itself_and_stops_the_rest(repo, client):
    """A pipeline reporting only "failed" gets retried whole."""
    (repo / "new.txt").write_text("work\n", encoding="utf-8")

    # No remote is configured, so push cannot succeed.
    result = _dwell(
        client,
        message="feat: a thing",
        stages=["survey", "branch", "stage", "commit", "push", "pr"],
    )

    assert not result["success"]
    assert result["stopped_at"] == "push"
    assert [e["stage"] for e in result["stages"]][-1] == "push"
    assert not any(e["stage"] == "pr" for e in result["stages"]), (
        "stages after the failure must not run"
    )


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

    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.split()
    assert committed == ["mine.txt"]


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
