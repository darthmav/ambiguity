"""`git_dwell`: a change carried from the working tree onto the default branch.

The Builder's one git tool that reaches a remote, and the console's way of
finishing what a run left waiting (`finish_pull_request`). Every stage is a
real git or gh command whose output is recorded, so the result is an account
of what happened; a stage that cannot go on stops the rest and names itself,
so a second call need not redo what worked.

The stages, in the only order they work in:

  survey   what the repository is: the branch, the default branch, a remote
  branch   never the default branch: a new branch off it -- or, when this
           branch's pull request has already merged, a fresh one from the
           remote's default, since a merged branch is finished
  stage    `git add`, of `paths` or of everything
  commit   what was staged (only `paths`, when named)
  update   the default branch merged in, so the checks run on what will land
  push     a push that never reached the remote is sent again; one the remote
           refused because the branch moved there merges it in first, once
  pr       the branch's open pull request, or a new one
  checks   the pull request's CI, waited on for a bounded time
  merge    a squash merge of exactly the commit whose checks were read
  cleanup  the branch deleted there and here, the default branch pulled

Three outcomes are ordinary rather than failures: nothing to commit, no remote
(the commit stays in the repository), and *pending* -- checks still running, a
review outstanding -- which leaves the pull request open and says what it is
waiting on. Nothing here rebases, force-pushes, or merges past a check that
failed or one it did not read.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph_agent.control import RUN_CONTROL
from langgraph_agent.self_healing import call_with_retry

# One git or gh invocation: `(ok, output)`, stderr folded into the output.
# `MCPClient._run_vcs`, called as run("git", *args, timeout=..., cwd=...).
VcsRunner = Callable[..., tuple[bool, str]]

DWELL_STAGES = (
    "survey", "branch", "stage", "commit", "update", "push", "pr", "checks", "merge", "cleanup",
)

# What runs when the caller names no stages: all of them. A caller who wants
# the review point names stages without `merge`.
DWELL_DEFAULT_STAGES = DWELL_STAGES

# The stages that need `origin`. Without one they are skipped, and the work
# stays in the repository -- a project the Builder `git init`ed has none.
REMOTE_STAGES = frozenset({"update", "push", "pr", "checks", "merge", "cleanup"})

# The stages that send work where other people see it. The Builder proves what
# it wrote before a call that includes any of them (`_dwell_gate` in nodes.py).
SHIPPING_STAGES = frozenset({"push", "pr", "checks", "merge"})

# What may not run on the default branch: each would put work on it directly.
_NOT_ON_THE_DEFAULT = frozenset({"commit", "update", "push", "pr"})

# How long `checks` waits for CI when the caller does not say, and the most it
# may be asked to. The Builder's node passes what is left of its own deadline
# instead, since a tool call is never abandoned; past the wait the pull request
# is left open as *pending*, for the console to finish.
DWELL_CHECKS_WAIT_SECONDS = float(os.getenv("DWELL_CHECKS_WAIT_SECONDS", "120"))
DWELL_CHECKS_MAX_WAIT_SECONDS = 600.0

# How often a wait asks GitHub again.
CHECKS_POLL_SECONDS = 10.0

# How long after a push no checks at all is read as "not registered yet" in a
# repository that has workflows. Past it, none of them runs on this pull
# request -- a path or branch filter -- and there is nothing to wait for.
CHECKS_REGISTER_SECONDS = 30.0

# How long GitHub may take to show a push as the pull request's head, and to
# work out whether the pull request can merge. Both are usually immediate.
HEAD_SETTLE_SECONDS = 10.0
MERGEABILITY_SETTLE_SECONDS = 10.0
SETTLE_POLL_SECONDS = 2.0

# The tail of a failed job's log handed back with a red check: where the
# assertion or the lint finding is.
FAILED_LOG_TAIL_CHARS = 1500

# One record's detail, cut so a long hook message cannot bury the rest.
MAX_STAGE_DETAIL_CHARS = 2000

# What git says when a push or fetch failed on the way to the remote rather
# than at it. Sending the same commit again is harmless -- the remote either
# has it or does not -- so these are retried, briefly; a rejected push is the
# remote answering and is not, and neither is a timeout, which already spent
# what a retry would.
_FAILED_IN_TRANSIT = (
    "could not resolve host", "connection reset", "connection refused",
    "failed to connect", "the remote end hung up unexpectedly", "early eof", "rpc failed",
)
PUSH_ATTEMPTS = 3


class _FailedInTransit(Exception):
    """A push or fetch that never reached the remote, carrying git's own words."""


class _Stop(Exception):
    """A stage that cannot go on: the pipeline ends there and says why."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


def _branch_name_from(message: str) -> str:
    """A branch name from a commit message's first line, for a caller that named none.

    Reduced to characters git accepts, since it reaches a remote.
    """
    head = (message.splitlines() or [""])[0].lower()
    # Drop a conventional-commit prefix: the commit carries the type.
    head = re.sub(r"^(feat|fix|docs|test|chore|refactor|ci|perf)(\([^)]*\))?:\s*", "", head)
    slug = re.sub(r"[^a-z0-9]+", "-", head).strip("-")[:48].strip("-")
    return f"agent/{slug or 'change'}"


def _moved_on_the_remote(output: str) -> bool:
    """Whether a push was refused because the branch there has commits this one lacks.

    Told apart from a hook or a protection rule refusing it (`[remote rejected]`),
    which merging cannot fix.
    """
    return "(fetch first)" in output or "(non-fast-forward)" in output


def _json_in(output: str, opener: str) -> Any:
    """The JSON value in gh's output, which arrives folded together with stderr."""
    closer = "]" if opener == "[" else "}"
    start, end = output.find(opener), output.rfind(closer)
    if start < 0 or end < start:
        return None
    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None


def _last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _gh_failure(output: str) -> str:
    """gh's own words, with what to do when the cause is gh itself."""
    lowered = output.lower()
    if "is not installed" in lowered:
        return (
            f"{output}. Install GitHub's CLI (pacman -S github-cli) and sign in "
            "with `gh auth login`."
        )
    if "gh auth login" in lowered or "not logged in" in lowered:
        return (
            f"{_last_line(output)} -- gh is not signed in here: run `gh auth login`, "
            "or in the container mount ~/.config/gh (docker-compose.yml)."
        )
    return output.strip() or "gh gave no reason"


def _wait_seconds(requested: Any) -> float:
    """A requested checks wait, clamped to [0, DWELL_CHECKS_MAX_WAIT_SECONDS].

    It comes from a model, so a missing or malformed value takes the default.
    """
    try:
        seconds = float(requested) if requested is not None else DWELL_CHECKS_WAIT_SECONDS
    except (TypeError, ValueError):
        seconds = DWELL_CHECKS_WAIT_SECONDS
    if not math.isfinite(seconds):
        seconds = DWELL_CHECKS_WAIT_SECONDS
    return max(0.0, min(seconds, DWELL_CHECKS_MAX_WAIT_SECONDS))


class _Dwell:
    """One call's pipeline: the facts it found, the stages it ran, where it stopped."""

    def __init__(
        self,
        run: VcsRunner,
        args: dict[str, Any],
        cwd: str | None,
        target: dict[str, Any] | None = None,
    ) -> None:
        self._run = run
        self.args = args
        self.cwd = cwd
        # Set when the console finishes a pull request a run left pending: the
        # branch, number and head it recorded, rather than what is checked out.
        self.target = target or {}
        self.log: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.skipped: set[str] = set()
        self.ran: set[str] = set()
        self.stages: list[str] = []
        self.paths: list[str] = []
        self.message = ""
        self.checks_wait = _wait_seconds(args.get("checks_timeout"))
        self.branch = ""
        self.default = ""
        self.unborn = False
        self.detached = False
        self.has_remote = False
        self.pr: dict[str, Any] = {}
        # The commit this call pushed, or the one the pull request must carry;
        # and the one GitHub merged, which is what cleanup measures against.
        self.head = str(self.target.get("head") or "")
        self.merged_head = ""
        self.pushed_at: float | None = None
        self.pending = ""
        self.merged = False
        self.merge_commit = ""
        self.checks_failed: list[str] = []
        self.failed_log = ""

    # -- plumbing -----------------------------------------------------------

    def git(self, *argv: str, timeout: float = 60.0) -> tuple[bool, str]:
        return self._run("git", *argv, timeout=timeout, cwd=self.cwd)

    def gh(self, *argv: str, timeout: float = 60.0) -> tuple[bool, str]:
        return self._run("gh", *argv, timeout=timeout, cwd=self.cwd)

    def record(self, stage: str, ok: bool, detail: str) -> None:
        self.log.append({"stage": stage, "ok": ok, "detail": detail[:MAX_STAGE_DETAIL_CHARS]})

    def pend(self, stage: str, reason: str) -> None:
        """End the pipeline here without failing: the pull request waits, open."""
        self.pending = reason
        self.record(stage, True, f"pending: {reason}")
        later = DWELL_STAGES[DWELL_STAGES.index(stage) + 1 :]
        self.skipped.update(later)

    def facts(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "default_branch": self.default,
            "stages": self.log,
            "local_only": not self.has_remote,
            "pull_request": dict(self.pr),
            "head": self.head,
            "pending": self.pending,
            "merged": self.merged,
            "merge_commit": self.merge_commit,
            "checks_failed": list(self.checks_failed),
            "failed_log": self.failed_log,
            "warnings": list(self.warnings),
        }

    # -- the call -------------------------------------------------------------

    def result(self) -> dict[str, Any]:
        refusal = self._parse()
        if refusal:
            return {"success": False, "error": refusal}
        steps: dict[str, Callable[[], None]] = {
            "branch": self._branch,
            "stage": self._stage,
            "commit": self._commit,
            "update": self._update,
            "push": self._push,
            "pr": self._pr,
            "checks": self._checks,
            "merge": self._merge,
            "cleanup": self._cleanup,
        }
        try:
            self._survey()
            for stage in self.stages:
                if stage == "survey" or stage in self.skipped:
                    continue
                if stage in REMOTE_STAGES and not self.has_remote:
                    self.record(stage, True, (
                        f"no remote named origin: the work stays on {self.branch} in this "
                        "repository, with nothing to push, open or merge"
                    ))
                    self.skipped.update(REMOTE_STAGES)
                    continue
                steps[stage]()
                self.ran.add(stage)
        except _Stop as stop:
            self.record(stop.stage, False, stop.detail)
            return {
                "success": False,
                "error": f"{stop.stage}: {stop.detail}",
                "stopped_at": stop.stage,
                **self.facts(),
            }
        summary = "; ".join(f"{entry['stage']}: {entry['detail']}" for entry in self.log)
        if self.warnings:
            summary += "; warnings: " + "; ".join(self.warnings)
        return {"success": True, **self.facts(), "summary": summary}

    def _parse(self) -> str:
        """Why the call cannot start, or "" -- checked before anything runs."""
        if self.cwd is not None:
            ok, top = self.git("rev-parse", "--show-toplevel")
            if not ok or Path(top).resolve() != Path(self.cwd).resolve():
                return (
                    f"{self.cwd} is not a git repository of its own, so git_dwell would act "
                    "on the repository around it. Run `git init` in it first "
                    f"(terminal_execute with cwd={self.cwd})."
                )
        for raw in [str(x) for x in (self.args.get("paths") or [])]:
            if self.cwd is None:
                self.paths.append(raw)
                continue
            # Spelled from the project root, like every other tool's paths, and
            # handed to git relative to the repository it runs in.
            inside = Path(raw).resolve()
            if not inside.is_relative_to(Path(self.cwd).resolve()):
                return f"{raw} is outside {self.cwd}, the repository this git_dwell runs in."
            self.paths.append(str(inside.relative_to(Path(self.cwd).resolve())) or ".")
        self.message = str(self.args.get("message") or "").strip()
        requested = [str(x) for x in (self.args.get("stages") or DWELL_DEFAULT_STAGES)]
        unknown = [x for x in requested if x not in DWELL_STAGES]
        if unknown:
            return (
                f"Unknown stage(s): {', '.join(unknown)}. "
                f"Valid stages, in order: {', '.join(DWELL_STAGES)}."
            )
        # Canonical order whatever the request's: "push then commit" is a typo,
        # not an instruction.
        self.stages = [x for x in DWELL_STAGES if x in requested]
        return ""

    # -- survey and branch ----------------------------------------------------

    def _current_branch(self) -> str:
        ok, out = self.git("rev-parse", "--abbrev-ref", "HEAD")
        if ok and out.strip():
            self.detached = out.strip() == "HEAD"
            return out.strip()
        # A repository with no commit yet has no HEAD to resolve, but it does
        # name the branch its first commit will found.
        ok, unborn = self.git("symbolic-ref", "--quiet", "--short", "HEAD")
        if ok and unborn.strip():
            self.unborn = True
            return unborn.strip()
        raise _Stop("survey", f"cannot read the current branch: {out}")

    def _default_branch(self) -> str:
        """The branch a pull request targets: `origin/HEAD` first, then the usual names.

        The remote's prefix is removed, not everything up to the last slash, so
        `release/2.0` survives.
        """
        ok, out = self.git("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
        if ok and out.startswith("refs/remotes/origin/"):
            return out.removeprefix("refs/remotes/origin/")
        for name in ("main", "master"):
            ok, _ = self.git("show-ref", "--verify", f"refs/heads/{name}")
            if ok:
                return name
        return "main"

    def _survey(self) -> None:
        current = self._current_branch()
        self.branch = str(self.target.get("branch") or current)
        if self.target.get("number") is not None:
            self.pr = {"number": self.target["number"], "url": str(self.target.get("url") or "")}
        self.has_remote = self.git("remote", "get-url", "origin")[0]
        # A repository's first commit has no base to branch from or open a pull
        # request against: it founds the default branch, and is pushed as that.
        self.default = self.branch if self.unborn else self._default_branch()
        if self.unborn:
            self.skipped.update({"update", "pr", "checks", "merge", "cleanup"})

        if "survey" in self.stages:
            ok, dirty = self.git("status", "--porcelain")
            if not ok:
                raise _Stop("survey", dirty)
            detail = (
                f"on {self.branch} (default {self.default}); "
                f"{len(dirty.splitlines())} path(s) changed; "
                f"origin: {'yes' if self.has_remote else 'none'}"
            )
            tracking = f"origin/{self.default}"
            if self.has_remote and self.git("rev-parse", "--verify", "--quiet", tracking)[0]:
                ok, counts = self.git("rev-list", "--left-right", "--count", f"{tracking}...HEAD")
                if ok and len(counts.split()) == 2:
                    behind, ahead = counts.split()
                    detail += f"; {ahead} ahead of and {behind} behind {tracking} as last fetched"
            self.record("survey", True, detail)

        on_the_default = not self.unborn and (self.detached or self.branch == self.default)
        if on_the_default and "branch" not in self.stages and _NOT_ON_THE_DEFAULT & set(self.stages):
            raise _Stop("branch", (
                f"refusing to commit onto {self.default}; include the 'branch' stage, "
                "or check out a branch first"
            ))

    def _branch_taken(self, name: str) -> bool:
        if self.git("show-ref", "--verify", "--quiet", f"refs/heads/{name}")[0]:
            return True
        # A branch of that name on the remote has its own history: pushing to it
        # would be refused, and merging it in would join two unrelated changes.
        return self.has_remote and self.git(
            "ls-remote", "--exit-code", "--heads", "origin", name, timeout=60
        )[0]

    def _free_name(self, name: str) -> str:
        candidate = name
        for suffix in range(2, 100):
            if candidate != self.branch and not self._branch_taken(candidate):
                return candidate
            candidate = f"{name}-{suffix}"
        return candidate

    def _branch(self) -> None:
        if self.unborn:
            self.record("branch", True, (
                f"{self.branch} has no commits yet: the first commit founds it, as there "
                "is nothing to branch from or open a pull request against"
            ))
            return
        if self.detached or self.branch == self.default:
            named = str(self.args.get("branch") or "").strip()
            # A name the caller chose is used as given, and git says if it is
            # taken; a derived one is made unique.
            wanted = named or self._free_name(_branch_name_from(self.message))
            ok, out = self.git("checkout", "-b", wanted)
            if not ok:
                raise _Stop("branch", out)
            base = "a detached HEAD" if self.detached else self.default
            self.branch, self.detached = wanted, False
            self.record("branch", True, f"created {wanted} off {base}")
            return
        merged = self._merged_pull_request()
        if merged is None:
            self.record("branch", True, f"already on {self.branch}, which is not {self.default}")
            return
        self._restart_after(merged)

    def _merged_pull_request(self) -> dict[str, Any] | None:
        """This branch's pull request when it has already merged, else None.

        Asked only with a remote; without gh, or with no pull request, the
        branch is taken as it is.
        """
        if not self.has_remote:
            return None
        view = self._pr_view(self.branch, "number,url,state,headRefOid")
        return view if view and view.get("state") == "MERGED" else None

    def _restart_after(self, merged: dict[str, Any]) -> None:
        """Carry the working tree onto a fresh branch from the remote's default.

        A merged branch is finished: commits on it would reach a pull request
        whose history GitHub already squashed away. Commits made here after the
        merge are not moved for the caller -- that is a cherry-pick, a decision.
        """
        number = merged.get("number")
        merged_head = str(merged.get("headRefOid") or "")
        if merged_head:
            ok, count = self.git("rev-list", "--count", f"{merged_head}..HEAD")
            if ok and count.strip().isdigit() and int(count) > 0:
                raise _Stop("branch", (
                    f"{self.branch}'s pull request #{number} has merged, and {count.strip()} "
                    f"commit(s) here came after it. Move them onto a branch from "
                    f"origin/{self.default} (git cherry-pick) and call git_dwell again; "
                    "nothing was changed."
                ))
        self._fetch(self.default, "branch")
        named = str(self.args.get("branch") or "").strip()
        wanted = self._free_name(named or _branch_name_from(self.message))
        ok, out = self.git("switch", "--no-track", "-c", wanted, f"origin/{self.default}")
        if not ok:
            raise _Stop("branch", (
                f"{self.branch}'s pull request #{number} has merged, so this work needs a "
                f"fresh branch from origin/{self.default}, and git would not switch: {out}"
            ))
        finished, self.branch = self.branch, wanted
        self.record("branch", True, (
            f"{finished}'s pull request #{number} has already merged; started {wanted} "
            f"from origin/{self.default}, carrying the working tree"
        ))

    # -- stage, commit, update, push ------------------------------------------

    def _pathspec(self) -> list[str]:
        return ["--", *self.paths] if self.paths else []

    def _stage(self) -> None:
        ok, out = self.git("add", *(self._pathspec() or ["-A"]))
        if not ok:
            raise _Stop("stage", out)
        self.record("stage", True, f"staged {', '.join(self.paths) if self.paths else 'all changes'}")

    def _commits_to_ship(self) -> int:
        """Commits on this branch the default branch does not have."""
        for base in (f"origin/{self.default}", self.default):
            if self.git("rev-parse", "--verify", "--quiet", base)[0]:
                ok, count = self.git("rev-list", "--count", f"{base}..HEAD")
                return int(count) if ok and count.strip().isdigit() else 0
        return 0

    def _commit(self) -> None:
        if not self.message:
            raise _Stop("commit", "no message given; pass `message`")
        # Asked of the named paths alone, when there are some: what else the
        # index holds is not this commit's.
        ok, staged = self.git("diff", "--cached", "--name-only", *self._pathspec())
        if ok and not staged.strip():
            # Nothing to commit is an ordinary outcome, not a failure. What
            # happens next depends on whether the branch has anything to ship:
            # a second call, after a fix was committed by hand or the checks
            # were left running, still has the rest to do.
            ahead = self._commits_to_ship()
            if ahead:
                self.record("commit", True, (
                    f"nothing new staged; {self.branch} already carries {ahead} commit(s) "
                    f"{self.default} does not, so the rest still runs"
                ))
            else:
                self.record("commit", True, "nothing staged to commit")
                self.skipped.update(REMOTE_STAGES)
            return
        ok, out = self.git("commit", "-m", self.message, *self._pathspec())
        if not ok:
            raise _Stop("commit", out)
        self.record("commit", True, out.splitlines()[0] if out else "committed")

    def _send(self, argv: tuple[str, ...], name: str) -> str:
        """One push or fetch, sent again only when it never reached the remote."""

        def once() -> str:
            ok, out = self.git(*argv, timeout=120)
            if ok:
                return out
            if any(mark in out.lower() for mark in _FAILED_IN_TRANSIT):
                raise _FailedInTransit(out)
            raise RuntimeError(out)

        return call_with_retry(
            once,
            max_attempts=PUSH_ATTEMPTS,
            min_wait=2.0,
            max_wait=4.0,
            exceptions=(_FailedInTransit,),
            give_up=RUN_CONTROL.stopped,
            name=name,
        )

    def _fetch(self, ref: str, stage: str) -> bool:
        """`origin/<ref>` brought up to date; False when the remote has no such branch."""
        try:
            self._send(("fetch", "origin", ref), "git fetch")
        except RuntimeError as exc:
            if "couldn't find remote ref" in str(exc).lower():
                return False
            raise _Stop(stage, f"could not fetch origin/{ref}: {exc}") from None
        except Exception as exc:
            raise _Stop(stage, f"could not fetch origin/{ref}: {exc}") from None
        return True

    def _abandon_merge(self, stage: str, ref: str, output: str) -> None:
        """Undo a merge that did not go through, and stop naming what conflicted."""
        ok, conflicted = self.git("diff", "--name-only", "--diff-filter=U")
        files = conflicted.split() if ok else []
        if self.git("rev-parse", "-q", "--verify", "MERGE_HEAD")[0]:
            self.git("merge", "--abort")
        if files:
            raise _Stop(stage, (
                f"merging {ref} conflicts in {', '.join(files)}. The merge was undone and "
                f"nothing was pushed. Merge it yourself (terminal_execute: git merge {ref}), "
                "resolve those files, commit, and call git_dwell again."
            ))
        raise _Stop(stage, f"could not merge {ref}: {output.strip() or 'git gave no reason'}")

    def _update(self) -> None:
        if not self._fetch(self.default, "update"):
            self.record("update", True, f"origin has no {self.default} yet: nothing to bring in")
            return
        upstream = f"origin/{self.default}"
        if self.git("merge-base", "--is-ancestor", upstream, "HEAD")[0]:
            self.record("update", True, f"already contains {upstream}")
            return
        ok, out = self.git("merge", "--no-edit", upstream)
        if not ok:
            self._abandon_merge("update", upstream, out)
        self.record("update", True, f"merged {upstream} in, so the checks run on what will land")

    def _push(self) -> None:
        argv = ("push", "-u", "origin", self.branch)
        moved = False
        try:
            self._send(argv, "git push")
        except RuntimeError as exc:
            if not _moved_on_the_remote(str(exc)):
                raise _Stop("push", str(exc)) from None
            # Someone else pushed to this branch: their commits are merged in,
            # never overwritten.
            self._fetch(self.branch, "push")
            ok, out = self.git("merge", "--no-edit", f"origin/{self.branch}")
            if not ok:
                self._abandon_merge("push", f"origin/{self.branch}", out)
            try:
                self._send(argv, "git push")
            except Exception as again:
                raise _Stop("push", (
                    f"origin/{self.branch} had moved; after merging it in, the push still "
                    f"failed: {again}"
                )) from None
            moved = True
        except Exception as exc:
            raise _Stop("push", str(exc)) from None
        ok, sha = self.git("rev-parse", "HEAD")
        self.head = sha.strip() if ok else ""
        self.pushed_at = time.monotonic()
        note = f" (origin/{self.branch} had moved: merged it in first)" if moved else ""
        self.record("push", True, f"pushed {self.branch} to origin{note}")

    # -- the pull request -----------------------------------------------------

    def _pr_view(self, target: Any, fields: str) -> dict[str, Any] | None:
        """`gh pr view` as a dict, or None: no pull request, no gh, or gh signed out."""
        _, out = self.gh("pr", "view", str(target), "--json", fields)
        data = _json_in(out, "{")
        return data if isinstance(data, dict) else None

    def _pr_target(self) -> Any:
        return self.pr.get("number") or self.branch

    def _require_pr(self, stage: str) -> None:
        if self.pr.get("number"):
            return
        view = self._pr_view(self.branch, "number,url,state")
        if view is None or not view.get("number"):
            raise _Stop(stage, (
                f"there is no pull request for {self.branch} (or gh cannot see it); "
                "include the 'pr' stage"
            ))
        self.pr = {"number": view["number"], "url": str(view.get("url") or "")}

    def _pr(self) -> None:
        existing = self._pr_view(self.branch, "number,url,state")
        # Only an open one is this change's: a merged or closed pull request on
        # the same branch is history.
        if existing and existing.get("state") == "OPEN" and existing.get("number"):
            self.pr = {"number": existing["number"], "url": str(existing.get("url") or "")}
            self.record("pr", True, f"already open: {self.pr['url']}")
            return
        title = (self.message.splitlines() or [""])[0]
        if not title:
            ok, subject = self.git("log", "-1", "--format=%s")
            title = subject.strip() if ok and subject.strip() else "Automated change"
        ok, out = self.gh(
            "pr", "create", "--base", self.default, "--head", self.branch,
            "--title", title, "--body", self.message or "Opened by the dwell pipeline.",
            timeout=120,
        )
        if not ok:
            raise _Stop("pr", _gh_failure(out))
        url = next(
            (line.strip() for line in reversed(out.splitlines()) if line.strip().startswith("http")),
            "",
        )
        number = url.rstrip("/").rsplit("/", 1)[-1]
        self.pr = {"number": int(number) if number.isdigit() else None, "url": url}
        if self.pr["number"] is None:
            # The URL is how the number is read; asked again rather than guessed.
            view = self._pr_view(self.branch, "number,url")
            if view and view.get("number"):
                self.pr = {"number": view["number"], "url": str(view.get("url") or url)}
        self.record("pr", True, f"opened {self.pr['url'] or 'a pull request'}")

    def _local_head(self) -> str:
        ok, sha = self.git("rev-parse", "--verify", "--quiet", self.branch)
        return sha.strip() if ok else ""

    def _await_head(self, stage: str, deadline: float) -> None:
        """Hold until the pull request carries the commit this call means, or stop.

        A check read off any other commit says nothing about this one.
        """
        expected = self.head or self._local_head()
        if not expected:
            return
        settle_until = time.monotonic() + max(0.0, min(HEAD_SETTLE_SECONDS, deadline - time.monotonic()))
        actual = ""
        while True:
            view = self._pr_view(self._pr_target(), "headRefOid")
            actual = str((view or {}).get("headRefOid") or "")
            if actual == expected:
                self.head = expected
                return
            if time.monotonic() >= settle_until or RUN_CONTROL.stopped():
                break
            RUN_CONTROL.wait(SETTLE_POLL_SECONDS)
        where = "pushed here" if self.pushed_at is not None else f"{self.branch} is at here"
        raise _Stop(stage, (
            f"the pull request's head is {actual[:12] or 'unknown'}, not {expected[:12]}, the "
            f"commit {where}. Someone else pushed to {self.branch}, or this repository has "
            "commits the pull request lacks: call git_dwell again, with 'push' among its stages."
        ))

    def _runs_workflows(self) -> bool:
        """Whether the repository has GitHub Actions workflows that could run checks."""
        ok, top = self.git("rev-parse", "--show-toplevel")
        if not ok:
            return False
        workflows = Path(top.strip()) / ".github" / "workflows"
        return workflows.is_dir() and any(
            path.suffix in (".yml", ".yaml") for path in workflows.iterdir()
        )

    def _read_checks(self, stage: str) -> list[dict[str, Any]]:
        _, out = self.gh(
            "pr", "checks", str(self._pr_target()), "--json", "name,state,bucket,link,workflow"
        )
        data = _json_in(out, "[")
        if isinstance(data, list):
            return [check for check in data if isinstance(check, dict)]
        if "no checks reported" in out.lower():
            return []
        raise _Stop(stage, f"could not read the pull request's checks: {_gh_failure(out)}")

    def _describe_failures(self, failed: list[dict[str, Any]]) -> str:
        """Name the red checks, and fetch the tail of the first one's log."""
        names: list[str] = []
        lines: list[str] = []
        for check in failed:
            workflow = str(check.get("workflow") or "").strip()
            name = str(check.get("name") or "?")
            label = f"{workflow} / {name}" if workflow and workflow != name else name
            names.append(label)
            verdict = "cancelled" if check.get("bucket") == "cancel" else "failed"
            lines.append(f"{label} {verdict} {check.get('link') or ''}".rstrip())
        self.checks_failed = names
        for check in failed:
            job = re.search(r"/actions/runs/\d+/job/(\d+)", str(check.get("link") or ""))
            if job is None:
                continue
            ok, log = self.gh("run", "view", "--job", job.group(1), "--log-failed", timeout=60)
            if ok and log.strip():
                self.failed_log = log.strip()[-FAILED_LOG_TAIL_CHARS:]
            break
        detail = "; ".join(lines)
        if self.failed_log:
            detail += ". The tail of the first failing job's log is in failed_log"
        return detail

    def _await_checks(self, stage: str, deadline: float) -> tuple[str, str]:
        """Read the pull request's checks until they settle or `deadline` passes.

        Returns ("pass" | "none" | "pending" | "fail", what to say about it).
        """
        workflows = self._runs_workflows()
        registered_by = (self.pushed_at or 0.0) + CHECKS_REGISTER_SECONDS
        started = time.monotonic()
        while True:
            checks = self._read_checks(stage)
            failed = [check for check in checks if check.get("bucket") in ("fail", "cancel")]
            if failed:
                return "fail", self._describe_failures(failed)
            running = [str(check.get("name") or "?") for check in checks
                       if check.get("bucket") == "pending"]
            now = time.monotonic()
            if checks and not running:
                passed = sum(1 for check in checks if check.get("bucket") == "pass")
                skipped = len(checks) - passed
                return "pass", f"{passed} check(s) passed" + (
                    f", {skipped} skipped" if skipped else ""
                )
            if not checks and (not workflows or now >= registered_by):
                return "none", (
                    "no checks to wait for: this repository runs no CI on pull requests"
                    if not workflows
                    else f"no check registered within {int(CHECKS_REGISTER_SECONDS)}s of the "
                    "push: none of this repository's workflows runs on this pull request"
                )
            waiting = ", ".join(running) if running else "its checks to register"
            if RUN_CONTROL.stopped():
                return "pending", f"stopped while waiting on {waiting}"
            if now >= deadline:
                return "pending", (
                    f"waiting on {waiting} after {int(now - started)}s; call git_dwell with "
                    "stages ['checks', 'merge', 'cleanup'] to finish, or let the console do it"
                )
            RUN_CONTROL.wait(min(CHECKS_POLL_SECONDS, max(0.0, deadline - now)))

    def _checks(self) -> None:
        self._require_pr("checks")
        deadline = time.monotonic() + self.checks_wait
        self._await_head("checks", deadline)
        outcome, detail = self._await_checks("checks", deadline)
        if outcome == "fail":
            raise _Stop("checks", (
                f"{detail}. Fix what failed, commit, and call git_dwell again: the push "
                "runs them again."
            ))
        if outcome == "pending":
            self.pend("checks", detail)
            return
        self.record("checks", True, detail)

    def _settled_view(self) -> dict[str, Any]:
        fields = (
            "number,url,state,isDraft,mergeable,mergeStateStatus,reviewDecision,"
            "headRefOid,baseRefName"
        )
        view = self._pr_view(self._pr_target(), fields)
        if view is None:
            raise _Stop("merge", f"gh cannot read pull request {self._pr_target()}")
        until = time.monotonic() + MERGEABILITY_SETTLE_SECONDS
        while view.get("mergeable") == "UNKNOWN" and time.monotonic() < until:
            if RUN_CONTROL.wait(SETTLE_POLL_SECONDS):
                break
            view = self._pr_view(self._pr_target(), fields) or view
        return view

    def _merge(self) -> None:
        self._require_pr("merge")
        view = self._settled_view()
        number, base = view.get("number"), str(view.get("baseRefName") or self.default)
        state = view.get("state")
        actual = str(view.get("headRefOid") or "")
        if state == "MERGED":
            self.merged, self.merged_head = True, actual
            self.record("merge", True, f"#{number} is already merged")
            return
        if state == "CLOSED":
            raise _Stop("merge", (
                f"#{number} was closed without merging; reopen it on GitHub, or open a new one"
            ))
        expected = self.head or self._local_head()
        if expected and actual != expected:
            raise _Stop("merge", (
                f"the pull request's head is {actual[:12] or 'unknown'}, not {expected[:12]}, "
                f"what {self.branch} is here: someone pushed since, or this repository has "
                "commits it lacks. Call git_dwell again with 'push' among its stages."
            ))
        self.head = actual
        if "checks" not in self.ran:
            # A merge never lands on checks it did not read: asked once, now.
            outcome, detail = self._await_checks("merge", time.monotonic())
            if outcome == "fail":
                raise _Stop("merge", f"its checks fail -- {detail}")
            if outcome == "pending":
                self.pend("merge", detail)
                return
        if view.get("isDraft"):
            self.pend("merge", f"#{number} is a draft")
            return
        review = view.get("reviewDecision") or ""
        if review == "CHANGES_REQUESTED":
            self.pend("merge", "a reviewer requested changes; their review comes before a merge")
            return
        if review == "REVIEW_REQUIRED":
            self.pend("merge", f"{base} requires an approving review first")
            return
        mergeable, status = view.get("mergeable"), view.get("mergeStateStatus")
        if mergeable == "CONFLICTING" or status == "DIRTY":
            raise _Stop("merge", (
                f"#{number} conflicts with {base}. Call git_dwell again: its update stage "
                f"merges {base} in, or stops naming the files that conflict."
            ))
        if status == "BEHIND":
            raise _Stop("merge", (
                f"{base} requires branches to be up to date, and #{number} is behind it. "
                "Call git_dwell again: its update stage merges it in and the push runs the "
                "checks again."
            ))
        if mergeable == "UNKNOWN":
            self.pend("merge", "GitHub has not finished working out whether it can merge")
            return
        if status == "BLOCKED":
            ok, out = self.gh(
                "pr", "merge", str(number), "--auto", "--squash", "--match-head-commit", actual,
                timeout=120,
            )
            self.pend("merge", (
                "auto-merge is on: GitHub merges it once branch protection is satisfied"
                if ok
                else f"blocked by branch protection ({_last_line(out)})"
            ))
            return
        # --match-head-commit: if anyone pushes between the checks being read
        # and this, GitHub refuses rather than merge a commit nobody checked.
        ok, out = self.gh(
            "pr", "merge", str(number), "--squash", "--match-head-commit", actual, timeout=120,
        )
        if not ok:
            raise _Stop("merge", _gh_failure(out))
        after = self._pr_view(number, "state,mergeCommit") or {}
        if after.get("state") != "MERGED":
            self.pend("merge", (
                "GitHub accepted the merge but has not made it yet (a merge queue?); it lands "
                "when the queue reaches it"
            ))
            return
        self.merged, self.merged_head = True, actual
        commit = after.get("mergeCommit")
        self.merge_commit = str(commit.get("oid") or "") if isinstance(commit, dict) else ""
        landed = f" as {self.merge_commit[:7]}" if self.merge_commit else ""
        self.record("merge", True, f"squash-merged #{number} into {base}{landed}")

    # -- cleanup --------------------------------------------------------------

    def _cleanup(self) -> None:
        """Remove the merged branch there and here, and bring the default branch here
        up to what landed.

        Only after GitHub says merged, which is what makes `branch -D` safe: a
        squash merge leaves the branch's commits unreachable from the default
        branch, so `-d` would refuse a branch whose work has landed. A step that
        cannot be done is a warning, never a failure: the merge is what counted.
        """
        merged_head = self.merged_head
        if not self.merged:
            view = self._pr_view(self._pr_target(), "number,state,headRefOid")
            if not view or view.get("state") != "MERGED":
                self.record("cleanup", True, (
                    f"nothing to clean up: {self.branch}'s pull request is not merged, so the "
                    "branch stays"
                ))
                return
            self.merged = True
            merged_head = str(view.get("headRefOid") or "")
        done: list[str] = []

        ok, out = self.git("push", "origin", "--delete", self.branch, timeout=60)
        if ok:
            done.append(f"deleted origin/{self.branch}")
        elif "remote ref does not exist" in out.lower():
            done.append(f"origin/{self.branch} was already gone")
        else:
            self.warnings.append(
                f"origin/{self.branch} remains ({_last_line(out)}); delete it from the pull "
                "request's page"
            )

        ok, current = self.git("rev-parse", "--abbrev-ref", "HEAD")
        current = current.strip() if ok else ""
        if current == self.branch:
            ok, out = self.git("switch", self.default)
            if not ok:
                self.warnings.append(
                    f"left on {self.branch}: git would not switch to {self.default} "
                    f"({_last_line(out)})"
                )
                self.record("cleanup", True, "; ".join(done) or "nothing else to clean up")
                return
            current = self.default
            done.append(f"switched to {self.default}")
        if current == self.default:
            fetched = self._fetch_quietly(self.default)
            ok, out = self.git("merge", "--ff-only", f"origin/{self.default}") if fetched else (
                False, "could not fetch it"
            )
            if ok:
                done.append(f"{self.default} is up to date with origin")
            else:
                self.warnings.append(f"{self.default} here was not updated ({_last_line(out)})")

        if self.git("show-ref", "--verify", "--quiet", f"refs/heads/{self.branch}")[0]:
            ok, extra = self.git("rev-list", "--count", f"{merged_head}..{self.branch}") if (
                merged_head
            ) else (False, "")
            if ok and extra.strip() == "0":
                ok, out = self.git("branch", "-D", self.branch)
                if ok:
                    done.append(f"deleted {self.branch} here")
                else:
                    self.warnings.append(f"{self.branch} here was kept ({_last_line(out)})")
            else:
                self.warnings.append(
                    f"kept {self.branch} here: it has commits the merged pull request does not"
                )
        self.record("cleanup", True, "; ".join(done) or "nothing left to clean up")

    def _fetch_quietly(self, ref: str) -> bool:
        try:
            return self._fetch(ref, "cleanup")
        except _Stop as stop:
            self.warnings.append(stop.detail)
            return False


def git_dwell(run: VcsRunner, args: dict[str, Any], cwd: str | None) -> dict[str, Any]:
    """Run the pipeline `args` asks for, in `cwd` (already resolved) or the project root."""
    return _Dwell(run, args, cwd).result()


def finish_pull_request(
    run: VcsRunner, *, cwd: str | None, branch: str, number: int, head: str
) -> dict[str, Any]:
    """Finish a pull request a run left pending: checks, merge, cleanup, without waiting.

    The console's monitor calls this. It acts on the branch, number and head
    the run recorded -- not on whatever is checked out now -- and merges only
    that head: a push since then is someone else's, and stops it.
    """
    target = {"branch": branch, "number": number, "head": head}
    args = {"stages": ["checks", "merge", "cleanup"], "checks_timeout": 0}
    return _Dwell(run, args, cwd, target).result()
