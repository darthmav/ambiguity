"""MCP client integration for GraphRAG and filesystem tools.

This module provides a unified interface to MCP (Model Context Protocol) servers.
For GraphRAG, can use the local knowledge base directly or connect to an MCP server.

Usage:
    from langgraph_agent.mcp_client import MCPClient

    async with MCPClient() as client:
        # List available tools
        tools = await client.list_tools()

        # Call GraphRAG search
        result = await client.call_tool("search_knowledge_graph", {"query": "Planner agent"})
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase



# How long one `terminal_execute` command may run before it is killed. It was
# 30, which is under what this project's own scripts take: `verify_and_test.py`
# finishes clean in ~33s and was reported FAILED for the 3s difference. That is
# the failure `MIN_VERIFY_SLICE_SECONDS` describes, one level up -- a working
# command comes back labelled broken, and the Builder spends its next turns
# repairing code that was never wrong. 60 matches `VERIFY_TIMEOUT_SECONDS`, the
# closest sibling: both bound a single command the Builder is waiting on, and
# the Builder's whole node budget (`BUILDER_DEADLINE_SECONDS`, 240) has to
# cover several of them plus the verification reserve.
TERMINAL_TIMEOUT_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_SECONDS", "60"))

# The ceiling on a *requested* timeout. A tool call is never abandoned -- the
# Builder's deadline may interrupt the model's turn but never a running tool --
# so a seat that asks for 99999 seconds does not overrun a deadline, it hangs
# the pass past every deadline there is. The ceiling is what stops an exposed
# knob from becoming that. 600 matches `_run_tests`, the longest thing the
# tool belt legitimately does.
TERMINAL_TIMEOUT_MAX_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_MAX_SECONDS", "600"))


# The largest file `filesystem_read` returns whole: fifty times what reaches the
# model (`MAX_TOOL_RESULT_CHARS`, 20,000), so any source file or document a
# Builder works on is read in one piece, and only a log or a dump is turned away.
FILESYSTEM_READ_MAX_BYTES = 1_000_000


def _resolve_timeout(requested: Any, default: float | None = None) -> float:
    """Clamp a requested command timeout to [1, TERMINAL_TIMEOUT_MAX_SECONDS].

    A missing or malformed value falls back to `default` --
    `TERMINAL_TIMEOUT_SECONDS` unless the caller names another -- instead of
    raising. The number arrives as JSON from a model, and refusing the call to
    complain about it costs a whole tool turn to say what a clamp says for
    nothing. Non-finite is rejected here rather than left to `min`:
    `min(nan, 600)` is `nan`, which reaches `subprocess.run` as no timeout at
    all.
    """
    if default is None:
        default = TERMINAL_TIMEOUT_SECONDS
    if requested is None:
        return default
    try:
        seconds = float(requested)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(seconds) or seconds <= 0:
        return default
    return min(seconds, TERMINAL_TIMEOUT_MAX_SECONDS)


# A shell builtin reaches `subprocess` as a missing executable, so
# `Command not found: 'cd'` is what the Builder gets -- true, and useless: it
# names what failed and not one of the things that would work. Measured on the
# rerun of the leviathan goal after `cwd` shipped, the Builder spent three
# turns on `cd there && ...` before finding the argument that replaces it. The
# schema says so already; an error is read at the moment the mistake is made,
# which a description consulted before the turn is not.
#
# Only the first group has a replacement to point at. The rest get the fact
# that ends the retry rather than an alternative that does not exist: `export`
# cannot set a variable for a later call here, and no wording makes it able to.
_CWD_BUILTINS = frozenset({"cd", "pushd", "popd"})
_OTHER_BUILTINS = frozenset(
    {"source", ".", "export", "set", "unset", "alias", "eval", "exec"}
)


# Shell operators, recognised only as *whole argv tokens*. That precision is
# what separates this from the character whitelist the shell removal replaced:
# the old filter scanned the raw string and so refused
# `python -c "import x; print(y)"` for a `;` that was never syntax. After
# `shlex.split` the same one-liner is three tokens with the `;` inside the
# quoted one, and only an operator the caller wrote unquoted stands alone.
#
# Redirection gets a different sentence from chaining because a replacement
# exists for it and not for them -- the same reason `cd` is answered with `cwd`
# and `export` is not answered with anything.
_REDIRECTS = frozenset({">", ">>", "<", "<<", "2>", "2>>", "1>", "&>", ">&"})
_CHAINS = frozenset({"&&", "||", ";", "&", "|"})
SHELL_OPERATORS = _REDIRECTS | _CHAINS

# A redirect written the way it is usually written: glued to its target.
# `shlex` keeps `2>/dev/null` as one token, so the whole-token check never saw
# an operator in it and the program was handed the literal string -- `find`
# answered `paths must precede expression: '2>/dev/null'` on the run of
# 2026-09-10, whose Builder wrote that suffix nine times in one pass. Output
# redirection only: an input redirect glued to its target cannot be told from
# a real argument (`grep "<div>"`), and `=` may not follow the `>`, so a
# version bound like `>=1.0` passes. The cost is the one `&&` already pays: an
# argument that really does begin with `>` cannot be passed, quoted or not.
_GLUED_REDIRECT = re.compile(r"^[0-9&]?>>?(?:&[0-9]+|[^\s=>&]\S*)$")


def _is_shell_operator(token: str) -> bool:
    return token in SHELL_OPERATORS or _GLUED_REDIRECT.match(token) is not None


def _shell_operator_error(token: str) -> str:
    """Explain a shell operator that reached argv, and name what replaces it.

    There is no shell, so the operator is inert either way; refusing it is
    about the *diagnosis*, not about safety. Left to run, the command fails
    somewhere unrelated and blames the wrong thing: `wc -l notes.md && tail
    -50 notes.md` hands `wc` the arguments `&&`, `tail` and `-50`, and comes
    back `wc: invalid option -- '5'` having never counted the file it was
    given. Nothing in that names the chain.

    That a description is not enough here is measured rather than assumed. The
    schema already said shell syntax is not interpreted, and on the rerun of
    2026-09-08 the Builder still reached for `cd there && ...` three times
    before using the `cwd` argument that replaces it. A description is
    consulted before the turn; an error is read at the moment the mistake is
    made.
    """
    if token in _REDIRECTS or _GLUED_REDIRECT.match(token):
        # Discarding or merging a stream is the commonest redirect by far, and
        # it has no replacement because it needs none: nothing is printed to a
        # terminal here, both streams come back as separate fields.
        if token.endswith("/dev/null") or re.search(r">&[0-9]+$", token):
            return (
                f"{token!r} is shell redirection, and there is no shell here. "
                "Drop it: stdout and stderr already come back to you "
                "separately, whatever the command prints."
            )
        return (
            f"{token!r} is shell redirection, and there is no shell here. Use "
            "`filesystem_write` to write a file, or `filesystem_read` to read "
            "one -- the output of this call is already returned to you."
        )
    if token == "|":
        return (
            "'|' is a shell pipe, and there is no shell here to connect two "
            "programs. Run one program per call and work on the output that "
            "comes back."
        )
    return (
        f"{token!r} chains commands in a shell, and there is no shell here. Run "
        "one program per call. To run somewhere else, pass `cwd` rather than "
        "chaining a `cd`."
    )


def _missing_program_error(program: str) -> str:
    """Say a program is missing, and say what to do when it is a builtin instead.

    Kept apart from the `FileNotFoundError` handler so the wording can be
    tested without spawning anything.
    """
    if program in _CWD_BUILTINS:
        return (
            f"Command not found: {program!r}. It is a shell builtin, not a "
            "program, and there is no shell here -- pass `cwd` to choose the "
            "directory the command runs in."
        )
    if program in _OTHER_BUILTINS:
        return (
            f"Command not found: {program!r}. It is a shell builtin, not a "
            "program, and there is no shell here -- run one program per call."
        )
    return f"Command not found: {program!r}"


# What a shell would have expanded before the program saw the argument.
_GLOB_CHARS = frozenset("*?[")


def _unexpanded_hint(argv: list[str], stderr: str) -> str | None:
    """Name a glob or `~` that reached the program literally and tripped it.

    With no shell, `cat dir/*` hands `cat` a file named `dir/*`, and `cat`
    answers "No such file or directory" about a directory that exists -- true
    of the name it got, and false of what was meant. Only raised when the
    program quoted the token back in its own error: `find . -name "*.py"`
    wants the literal and succeeds, and a regex such as `[` that fails to
    compile is reported unquoted, so neither is mistaken for an unexpanded
    glob.
    """
    for token in argv[1:]:
        if not (_GLOB_CHARS & set(token) or token.startswith("~")):
            continue
        if any(f"{quote}{token}'" in stderr for quote in ("'", "`")) or f'"{token}"' in stderr:
            return (
                f"{token!r} reached the program exactly as written: there is no "
                "shell here to expand a glob or `~`. List the directory with `ls` "
                "and name the file, or write the absolute path."
            )
    return None


# The base every filesystem tool resolves a relative path against: the server's
# working directory, which is the project root. Named here so `_resolve_cwd`,
# `filesystem_read` and `filesystem_write` cannot drift into three bases -- one
# relative path has to mean one place across the whole tool belt.
def _project_root() -> Path:
    return Path.cwd().resolve()


# Stages of the `git_dwell` pipeline, in the only order they work in. Named
# here so the tool can report which one it reached: a pipeline that fails
# somewhere has to say where, or the caller retries the whole thing and re-runs
# the parts that already succeeded.
DWELL_STAGES = ("survey", "branch", "stage", "commit", "push", "pr", "merge")

# What `git_dwell` runs when the caller names no stages: the whole pipeline,
# merge included. `merge` sat outside this tuple until 2026-09-12, on the
# argument that a pull request the same agent opens and immediately merges is
# not a review -- which is true, and was the wrong thing for the default to
# decide. The tool is one call because the flow is one act; a default that
# stopped one stage short meant the common case was a branch pushed, a PR
# opened and nothing on the default branch, with the last stage left to a
# caller that had no way to know it was missing. A caller who wants the review
# point still has it, and now has to say so: `stages` without `merge` stops at
# `pr`, which is the same opt-out the old default was, spelled by whoever
# actually wants it.
DWELL_DEFAULT_STAGES = DWELL_STAGES


def _branch_name_from(message: str) -> str:
    """Derive a branch name from a commit message's first line.

    Only used when the caller did not name one. Conservative on purpose: the
    name reaches a remote, so it is reduced to the characters git is happy with
    rather than cleverly abbreviated.
    """
    head = (message.splitlines() or [""])[0].lower()
    # Drop a conventional-commit prefix: `fix: contain writes` is a better
    # branch as `contain-writes` than as `fix-contain-writes`, and the type is
    # already carried by the commit itself.
    head = re.sub(r"^(feat|fix|docs|test|chore|refactor|ci|perf)(\([^)]*\))?:\s*", "", head)
    slug = re.sub(r"[^a-z0-9]+", "-", head).strip("-")[:48].strip("-")
    return f"agent/{slug or 'change'}"


def _resolve_write_path(requested: Any) -> tuple[Path | None, str | None]:
    """Resolve a requested write path to `(path, error)`, refusing to escape the project.

    An error is returned *instead of* a path, never alongside one, the same
    shape `_resolve_cwd` uses and for the same reason: the caller must not be
    able to act on half an answer.

    This is containment, not safety theatre. `filesystem_write` took its
    argument raw and called `Path(path).write_text()` behind a
    `parent.mkdir(parents=True)`, so an absolute path, a `..` or a symlink put
    the Builder's writes anywhere the account could reach -- and it did: the
    run of 2026-09-11 wrote `/tmp/gen_doc.py` and `/tmp/gen_overview.py` while
    working on a goal about this checkout. Two costs, and the second is the one
    that bites. The obvious cost is a file left outside the project that no
    reindex, no `corpus_staleness` and no `git status` will ever mention. The
    quiet one is that `files_changed` feeds the console's "changed this
    machine" notice, so a write outside the root makes that notice name a path
    the operator cannot find from the project -- the same false account as a
    Builder claiming a file it never wrote, which this project already guards
    in `_report_path_key`, pointing the other way.

    The path is resolved *before* the write rather than checked after, and
    `resolve()` is what does it: a check on the literal string would pass
    `project/link/x` where `link` points at `/etc`, because only resolution
    knows where a symlink lands. The whole path is resolved, leaf included:
    a leaf that already exists as a symlink out of the project is refused,
    which a check of the parent alone would miss -- `write_text` follows it.
    A leaf that does not exist yet is fine, since `strict=False` resolution
    resolves the part of the path that exists and keeps the rest as written.

    `~` is not expanded, for the reason `_resolve_cwd` gives: there is no shell
    here, and a path that quietly expanded what an argument on the same line
    would not is a worse surprise than a refusal naming the path.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None, f"Invalid path: {requested!r}. Pass a file path inside the project."
    root = _project_root()
    candidate = Path(requested)
    # The whole path, leaf included -- see the docstring for why the leaf
    # matters as much as the parent.
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, (
            f"Refusing to write outside the project: {requested!r} resolves to "
            f"{resolved}, which is not under {root}. Write inside the project; "
            "a file outside it is invisible to the reindex, to corpus staleness "
            "and to git, and makes the run's 'changed this machine' notice name "
            "a path nobody can find."
        )
    return resolved, None


def _resolve_cwd(requested: Any) -> tuple[str | None, str | None]:
    """Resolve a requested working directory to `(cwd, error)`.

    An error is returned *instead of* a directory, never alongside one; both
    are None when nothing was requested and the command inherits ours.

    The check is here rather than left to `subprocess.run` because of what
    that raises: a missing `cwd` comes back as `FileNotFoundError`, which is
    the same exception a missing *program* raises and lands in the handler
    that answers `Command not found: 'python'` -- naming the one thing that
    was fine. A `cwd` that exists but is a file raises `NotADirectoryError`,
    which is not a `FileNotFoundError` at all and falls through to a bare
    errno string. Both are the false accusation this module keeps having to
    design against, so the directory is checked while we still know it is the
    directory being complained about.

    A relative path resolves against the server's working directory, which is
    the project root -- the same base `filesystem_read` and `filesystem_write`
    use, so one relative path means one place across the whole tool belt.
    `~` is not expanded, for the reason no other shell syntax is: there is no
    shell here, and a `cwd` that quietly expanded what an argument on the same
    line would not is a worse surprise than a refusal naming the path.
    """
    if requested is None:
        return None, None
    if not isinstance(requested, str) or not requested.strip():
        return None, f"Invalid cwd: {requested!r}. Pass a directory path."
    path = Path(requested)
    if not path.is_dir():
        detail = "exists but is not a directory" if path.exists() else "does not exist"
        return None, f"Cannot run in {requested!r}: it {detail}."
    return str(path), None


class MCPClient:
    """Client for MCP servers (GraphRAG, Filesystem, Git).

    Uses local GraphRAG knowledge base when available, falls back to stubs.
    """

    def __init__(self, server_urls: list[str] | None = None):
        """Initialize MCP client.

        Args:
            server_urls: List of MCP server URLs (default from env vars)
        """
        self.server_urls = server_urls or self._default_servers()
        self._connected = False
        self._tools: dict[str, Any] = {}
        self._kb: GraphRAGKnowledgeBase | None = None

    def _default_servers(self) -> list[str]:
        """Get default MCP server URLs from environment."""
        servers: list[str] = []
        if url := os.getenv("MCP_GRAPHRAG_URL"):
            servers.append(url)
        if url := os.getenv("MCP_FILESYSTEM_URL"):
            servers.append(url)
        if url := os.getenv("MCP_GIT_URL"):
            servers.append(url)
        return servers

    async def connect(self) -> None:
        """Connect to MCP servers and initialize local GraphRAG (lazy)."""
        # Lazy init GraphRAG - only when actually needed for search
        self._kb = None
        self._connected = True
        self._tools = self._discover_tools()

    async def disconnect(self) -> None:
        """Disconnect from MCP servers."""
        self._connected = False
        self._tools = {}
        self._kb = None

    def _discover_tools(self) -> dict[str, Any]:
        """Discover available tools from connected servers.

        Returns:
            Dict mapping tool names to tool callables
        """
        tools = {}

        # Always provide GraphRAG read-only tools (local or stub).
        # GraphRAG is read-only per the 4-Agent System specification; adding
        # documents is done through indexing scripts, not the Researcher tool belt.
        tools["search_knowledge_graph"] = self._graphrag_search
        tools["query_knowledge_graph"] = self._graphrag_query_graph

        # Real filesystem, git, terminal, and test tools
        tools["filesystem_read"] = self._filesystem_read
        tools["filesystem_write"] = self._filesystem_write
        tools["git_status"] = self._git_status
        tools["git_diff"] = self._git_diff
        tools["git_dwell"] = self._git_dwell
        tools["terminal_execute"] = self._terminal_execute
        tools["run_tests"] = self._run_tests

        return tools

    async def list_tools(self) -> list[str]:
        """List all available tools."""
        if not self._connected:
            await self.connect()
        return list(self._tools.keys())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._connected:
            await self.connect()

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_fn = self._tools[tool_name]
        return await tool_fn(arguments)

    # GraphRAG implementations (use local knowledge base when available)

    def _open_kb(self) -> GraphRAGKnowledgeBase | None:
        """The corpus if one has been built, `None` otherwise.

        `open_knowledge_base`, never `get_knowledge_base`: the second one
        creates the store, and a Researcher's query is not a request for a
        knowledge base. Cached on the client so repeated calls in one run do
        not reopen Chroma. The cache is only ever filled with a real corpus,
        so a search that happens before the operator indexes does not pin
        `None` for the rest of the process.
        """
        if self._kb is None:
            try:
                from langgraph_agent.graphrag_server import open_knowledge_base

                self._kb = open_knowledge_base()
            except Exception:
                self._kb = None
        return self._kb

    async def _graphrag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search the knowledge base (cached singleton).

        With no corpus this returns *no* results. It used to return one
        fabricated row -- `[GraphRAG not indexed]`, score 0.0 -- which is a
        made-up retrieval hit sitting in the same field real ones arrive in,
        and the Builder reads that field.
        """
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import NO_CORPUS_NOTE

            return {"results": [], "source": "no_corpus", "note": NO_CORPUS_NOTE}

        return {"results": kb.search(query, top_k), "source": "local_graphrag"}

    async def _graphrag_query_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query the knowledge graph."""
        entity = args.get("entity", "")
        hops = args.get("hops", 2)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import NO_CORPUS_NOTE

            return {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "no_corpus",
                "note": NO_CORPUS_NOTE,
            }

        result = kb.query_graph(entity, hops)
        result["source"] = "local_graphrag"
        return result

    # Real filesystem implementations

    async def _filesystem_read(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read file contents.

        Not confined to the project the way `filesystem_write` is, and on
        purpose. `terminal_execute` runs any program with this server's
        permissions -- `cat` reads whatever this tool would -- so a fence here
        would be a safety claim the tool belt cannot keep. Writes are confined
        for what they feed, `files_changed` and the corpus, which are about the
        project; they are not a sandbox either. What bounds a run is where its
        seats run (local by default, so nothing read leaves this machine) and
        the console listening only on loopback.

        A file over `FILESYSTEM_READ_MAX_BYTES` is refused unread rather than
        loaded whole to be cut at `MAX_TOOL_RESULT_CHARS` on its way to the
        model -- a log or a data dump that size is memory and time for text
        nobody sees -- and the refusal says how to read part of it.
        """
        path = args.get("path", "")
        try:
            size = Path(path).stat().st_size
            if size > FILESYSTEM_READ_MAX_BYTES:
                return {
                    "success": False,
                    "path": path,
                    "error": (
                        f"{path} is {size:,} bytes, past the {FILESYSTEM_READ_MAX_BYTES:,} "
                        "this tool reads whole. Read part of it with terminal_execute: "
                        f"`head -n 200 {path}`, or `sed -n 200,400p {path}` for a range."
                    ),
                }
            content = Path(path).read_text(encoding="utf-8")
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    async def _filesystem_write(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write file contents inside the project, creating parent directories if needed.

        The path goes through `_resolve_write_path` first, which refuses
        anything landing outside the project root -- see it for why a raw
        `Path(path).write_text()` here was worth closing.
        """
        path = args.get("path", "")
        content = args.get("content", "")
        file_path, error = _resolve_write_path(path)
        if error is not None or file_path is None:
            return {"success": False, "error": error, "path": path}
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"success": True, "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    # Real git implementations

    async def _git_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Git status.

        git's exit status is read before its output is. A command that failed
        prints nothing on stdout, and "nothing" is exactly what a clean tree
        prints too -- so a directory that is not a repository came back as
        `success` with "Working tree clean".
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
        if result.returncode != 0:
            return {"success": False, "error": (result.stderr or result.stdout).strip()
                    or f"git status exited {result.returncode}"}
        return {"success": True, "status": result.stdout or "Working tree clean"}

    async def _git_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        """Git diff, of the whole tree or of one path.

        The path goes after `--`, and only when there is one. With no path this
        used to run `git diff ""`, which git refuses outright (an empty string
        is not a pathspec) -- and because the exit status went unread, the
        refusal's empty stdout came back as `success` with "No changes". So
        the Builder's plain `git_diff()` reported a clean tree on every call,
        however much it had changed. Without `--`, a path that also names a
        branch or a commit is read as a revision.
        """
        path = str(args.get("path") or "").strip()
        command = ["git", "diff"] + (["--", path] if path else [])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
        if result.returncode != 0:
            return {"success": False, "error": (result.stderr or result.stdout).strip()
                    or f"git diff exited {result.returncode}"}
        return {"success": True, "diff": result.stdout or "No changes"}

    # The ordered git pipeline (`git_dwell`)

    def _run_vcs(self, *argv: str, timeout: float = 60.0) -> tuple[bool, str]:
        """One git/gh invocation. Returns `(ok, output)` with stderr folded in.

        stderr is kept because git says the useful part there -- "nothing to
        commit", "no upstream branch", a rejected push -- and a stage that
        failed with an empty message is a stage nobody can act on. No shell,
        for the reason `_terminal_execute` has none: the arguments are argv
        entries, so a commit message containing `;` or `&&` is a message.
        """
        try:
            done = subprocess.run(
                list(argv), capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False, f"{argv[0]} is not installed on this machine"
        except subprocess.TimeoutExpired:
            return False, f"{' '.join(argv)} timed out after {timeout:g}s"
        return done.returncode == 0, ((done.stdout or "") + (done.stderr or "")).strip()

    def _default_branch(self) -> str:
        """The branch a PR targets. `origin/HEAD` first, then the usual names.

        The remote's prefix is removed rather than everything up to the last
        slash, which cut a default branch named `release/2.0` down to `2.0`.
        """
        ok, out = self._run_vcs("git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
        if ok and out.startswith("refs/remotes/origin/"):
            return out.removeprefix("refs/remotes/origin/")
        for name in ("main", "master"):
            ok, _ = self._run_vcs("git", "show-ref", "--verify", f"refs/heads/{name}")
            if ok:
                return name
        return "main"

    async def _git_dwell(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the git pipeline in order, stopping at the first stage that fails.

        Every stage is a real command whose output is recorded, so the result
        is an account of what happened rather than a claim that it did -- the
        rule `files_changed` already follows, one level up. A stage that fails
        leaves the ones after it unrun and names itself in `stopped_at`, since
        a pipeline that reports only "failed" gets retried whole and re-runs
        the parts that already worked.

        *It will not commit onto the default branch.* Committing straight onto
        `main` is exactly what the branch-then-PR flow exists to prevent, and
        an agent doing it has removed the review point before anyone could use
        it. So `branch` creates one when HEAD is the default, naming it from
        the message when the caller did not. This is the refusal that survives,
        and it is the one that matters: the work still arrives on a branch,
        through a pull request, with the diff and the checks attached.

        *It does merge by default*, and that changed on 2026-09-12. The
        argument for stopping at `pr` was that a pull request the same agent
        opens and immediately merges is not a review. That remains true -- it
        is simply not something a default can decide. A caller who wants the
        review point names its stages and stops at `pr`; what the old default
        produced instead was the opposite of a considered choice, a pipeline
        that did six sevenths of a job every time and left the seventh to
        somebody who had not been told it was outstanding. `merge` is
        `--squash --delete-branch`, so what lands on the default branch is one
        commit carrying the pull request's title and body.
        """
        message = str(args.get("message") or "").strip()
        requested = [str(x) for x in (args.get("stages") or DWELL_DEFAULT_STAGES)]
        unknown = [x for x in requested if x not in DWELL_STAGES]
        if unknown:
            return {"success": False, "error":
                    f"Unknown stage(s): {', '.join(unknown)}. "
                    f"Valid stages, in order: {', '.join(DWELL_STAGES)}."}
        # Run in the canonical order whatever order they arrived in: these are
        # pipeline phases, not a script, so "push then commit" is a typo rather
        # than an instruction to do it backwards.
        stages = [x for x in DWELL_STAGES if x in requested]

        log: list[dict[str, Any]] = []

        def record(stage: str, ok: bool, detail: str) -> None:
            log.append({"stage": stage, "ok": ok, "detail": detail[:2000]})

        def stop(stage: str, detail: str) -> dict[str, Any]:
            record(stage, False, detail)
            return {"success": False, "error": f"{stage}: {detail}",
                    "stopped_at": stage, "stages": log}

        default = self._default_branch()
        ok, branch = self._run_vcs("git", "rev-parse", "--abbrev-ref", "HEAD")
        if not ok:
            return stop("survey", f"cannot read the current branch: {branch}")

        if "survey" in stages:
            ok, dirty = self._run_vcs("git", "status", "--porcelain")
            if not ok:
                return stop("survey", dirty)
            record("survey", True, f"on {branch} (default {default}); "
                                   f"{len(dirty.splitlines())} path(s) changed")

        if "branch" in stages:
            if branch == default:
                wanted = str(args.get("branch") or "").strip() or _branch_name_from(message)
                ok, out = self._run_vcs("git", "checkout", "-b", wanted)
                if not ok:
                    return stop("branch", out)
                branch = wanted
                record("branch", True, f"created {branch} off {default}")
            else:
                record("branch", True, f"already on {branch}, which is not {default}")
        elif branch == default and {"commit", "push", "pr", "merge"} & set(stages):
            return stop("branch", f"refusing to commit onto {default}; include the "
                                  "'branch' stage, or check out a branch first")

        if "stage" in stages:
            paths = [str(x) for x in (args.get("paths") or [])]
            ok, out = self._run_vcs("git", "add", *(paths or ["-A"]))
            if not ok:
                return stop("stage", out)
            record("stage", True, f"staged {', '.join(paths) if paths else 'all changes'}")

        if "commit" in stages:
            if not message:
                return stop("commit", "no message given; pass `message`")
            ok, staged = self._run_vcs("git", "diff", "--cached", "--name-only")
            if ok and not staged.strip():
                # Not a failure. A pass with nothing to commit is an ordinary
                # outcome, and failing here would send the Builder off
                # repairing a repository that is simply already clean -- the
                # false accusation this module keeps having to design against.
                # The stages that only make sense after a commit are dropped.
                record("commit", True, "nothing staged to commit")
                stages = [x for x in stages if x not in ("push", "pr", "merge")]
            else:
                ok, out = self._run_vcs("git", "commit", "-m", message)
                if not ok:
                    return stop("commit", out)
                record("commit", True, out.splitlines()[0] if out else "committed")

        if "push" in stages:
            ok, out = self._run_vcs("git", "push", "-u", "origin", branch, timeout=120)
            if not ok:
                return stop("push", out)
            record("push", True, f"pushed {branch} to origin")

        if "pr" in stages:
            ok, existing = self._run_vcs("gh", "pr", "view", "--json", "url", "-q", ".url")
            if ok and existing.strip().startswith("http"):
                # A branch already carrying a PR is the ordinary case on the
                # second pass of a run; opening a second one would fail anyway.
                record("pr", True, f"already open: {existing.strip()}")
            else:
                title = (message.splitlines() or ["Automated change"])[0]
                ok, out = self._run_vcs(
                    "gh", "pr", "create", "--base", default, "--head", branch,
                    "--title", title,
                    "--body", message or "Opened by the dwell pipeline.",
                    timeout=120,
                )
                if not ok:
                    return stop("pr", out)
                record("pr", True, out.splitlines()[-1] if out else "pull request opened")

        if "merge" in stages:
            ok, out = self._run_vcs("gh", "pr", "merge", "--squash", "--delete-branch",
                                    timeout=120)
            if not ok:
                return stop("merge", out)
            record("merge", True, out.splitlines()[-1] if out else "merged")

        return {"success": True, "branch": branch, "default_branch": default,
                "stages": log,
                "summary": "; ".join(f"{e['stage']}: {e['detail']}" for e in log)}

    # Terminal / test tools

    async def _terminal_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run a command in the project workspace.

        There is no shell. The command is split with `shlex` and handed to
        `subprocess` as an argv list, so `;`, `|`, `>`, `&&`, `$(...)` and
        globs are inert *data* rather than syntax -- `echo hi; rm -rf /` runs
        `echo` with four literal arguments and deletes nothing.

        This replaced a whitelist of permitted characters guarding
        `shell=True`. That filter refused the ordinary way to write a
        one-liner (`python -c "import x; print(y)"` trips on `;` `(` `)`) and
        any path containing parentheses, while still admitting a bare
        `rm -rf /` -- it never guarded against a destructive command, only
        against chaining one onto another. Removing the shell removes the
        thing the chaining needed, so the characters no longer have to be
        refused to be harmless.

        The trade is that shell *features* are gone. An operator written on
        purpose -- a pipe, a chain, a redirect spaced or glued -- is refused by
        name before anything runs (`_is_shell_operator`), because a silently
        meaningless pipe is worse than a refused one. A glob or `~` cannot be
        refused, since `find -name "*.py"` needs the literal, so it reaches the
        program as written and a failure it causes is named in `hint`.

        `cwd` runs the command somewhere other than the project root, and is
        offered to the Builder because without it there is no way to express
        it at all: `cd` is a shell builtin, so `cd somewhere && python x.py`
        does not run in the wrong directory, it fails with
        `Command not found: 'cd'` -- and the Builder, having no other spelling
        to try, spends turns rediscovering absolute paths. It is the same gap
        `timeout` was: a thing the tool can do that the schema never offered.

        `env` overlays the current environment for this one command; a key
        mapped to None is removed rather than set. It is not offered to the
        Builder in BUILDER_TOOLS -- only callers inside the process set it,
        which today means the verification pass asking for a headless run.

        `timeout` is offered to the Builder, and is clamped rather than
        trusted (`_resolve_timeout`). The verification pass always passes one
        computed from its remaining deadline, so it never sees the default.
        """
        command = args.get("command", "")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            # Unbalanced quotes. Say so plainly: the caller cannot see the
            # parse, and "No closing quotation" alone reads like the program
            # failed rather than like the command was never built.
            return {
                "success": False,
                "error": f"Could not parse command ({exc}). Check the quoting.",
                "command": command,
            }
        if not argv:
            return {
                "success": False,
                "error": "Empty command.",
                "command": command,
            }

        # Refused before the spawn, like `_resolve_cwd`, and for the same
        # reason: the complaint has to be made while we still know what is
        # being complained about. Note the limit -- an operator is caught when
        # it is its own token, or a redirect glued to its target
        # (`2>/dev/null`). `echo hi; rm -rf /` splits to `['echo', 'hi;', ...]`,
        # so the `;` rides on `hi` and stays inert data, which is what the
        # canary test pins. This catches the shapes a caller writes on purpose,
        # not every shape that exists.
        operator = next((token for token in argv if _is_shell_operator(token)), None)
        if operator is not None:
            return {
                "success": False,
                "error": _shell_operator_error(operator),
                "command": command,
            }

        cwd, cwd_error = _resolve_cwd(args.get("cwd"))
        if cwd_error is not None:
            return {"success": False, "error": cwd_error, "command": command}

        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_resolve_timeout(args.get("timeout")),
                env=_child_env(args.get("env")),
                cwd=cwd,
                # No human is at the keyboard behind a Builder tool call, so a
                # command that reads stdin must get EOF and fail, never block
                # until its timeout and report as a hang.
                stdin=subprocess.DEVNULL,
            )
            outcome: dict[str, Any] = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            }
            if result.returncode != 0:
                hint = _unexpanded_hint(argv, result.stderr or "")
                if hint:
                    outcome["hint"] = hint
            return outcome
        except subprocess.TimeoutExpired as e:
            # Keep what the command managed to print. `str(e)` alone says only
            # that it timed out, and a caller with no output to look at cannot
            # tell a command that hung immediately from one that did all its
            # work and then blocked at the end -- so it guesses, and pays the
            # full timeout again on a retry that was never going to differ.
            return {
                "success": False,
                "error": _timeout_error(e),
                "timed_out": True,
                "stdout": _as_captured_text(e.stdout),
                "stderr": _as_captured_text(e.stderr),
                "command": command,
            }
        except FileNotFoundError:
            # With a shell this came back as rc=127 and a message on stderr.
            # Without one it raises, and a bare OSError repr does not say which
            # of the words was the program -- so name it, or the caller reads
            # "not found" as its file argument being missing. A builtin is
            # named along with what to do instead (`_missing_program_error`).
            return {
                "success": False,
                "error": _missing_program_error(argv[0]),
                "command": command,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "command": command}

    async def _run_tests(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the pytest test suite.

        `cwd` runs it in another directory -- a generated project under
        `projects/`, whose Builder is told to pass it (`OUTPUT_DIR_NOTE`). It
        was told that while this tool took no `cwd` at all, so every such call
        silently ran *this checkout's* suite instead. With a `cwd` and no
        `path`, pytest collects from that directory; with neither, `tests/`.

        `timeout` is clamped like `terminal_execute`'s, for its reason: a tool
        call is never abandoned, so an unclamped request hangs the pass past
        every deadline. The default is the ceiling, the longest a suite is
        allowed.
        """
        cwd, cwd_error = _resolve_cwd(args.get("cwd"))
        if cwd_error is not None:
            return {"success": False, "error": cwd_error}
        target = str(args.get("path") or ("." if cwd else "tests/"))
        timeout = _resolve_timeout(args.get("timeout"), default=TERMINAL_TIMEOUT_MAX_SECONDS)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", target, "-q"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                # Same reason as _terminal_execute: a suite that stops to ask
                # something would otherwise hang until its timeout.
                stdin=subprocess.DEVNULL,
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "error": _timeout_error(e),
                "timed_out": True,
                "stdout": _as_captured_text(e.stdout),
                "stderr": _as_captured_text(e.stderr),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def _timeout_error(exc: subprocess.TimeoutExpired) -> str:
    """Say that a command timed out, and after how long, and nothing else.

    `str(TimeoutExpired)` is `Command '[...argv...]' timed out after N
    seconds`: the fact comes last, behind a repr of the whole argv. The report
    line already names the command and cuts its reason at
    MAX_FAILURE_REASON_CHARS, so on the run of 2026-09-10 a `pip3 install`
    carrying an index URL read `-> failed: Command '['pip3', 'install', ...]'...`
    -- the timeout cut off, the Builder retrying the identical command blind.
    No advice about raising `timeout` either: verification calls this tool
    with a limit it chose itself, and the Builder cannot raise that one.
    """
    return f"timed out after {exc.timeout:g} seconds"


def _as_captured_text(captured: str | bytes | None) -> str:
    """Normalise output hung off a TimeoutExpired to text.

    `capture_output=True` with `text=True` gives str, but the attribute is
    typed to allow bytes and is None when nothing was read before the kill.
    """
    if captured is None:
        return ""
    if isinstance(captured, bytes):
        return captured.decode("utf-8", "replace")
    return captured


def _child_env(overrides: dict[str, str | None] | None) -> dict[str, str] | None:
    """Build a child environment from os.environ plus `overrides`.

    A key mapped to None is removed. Returns None when there is nothing to
    override, so the child simply inherits ours.
    """
    if not overrides:
        return None
    env = dict(os.environ)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


@asynccontextmanager
async def mcp_client(server_urls: list[str] | None = None) -> AsyncGenerator[MCPClient, None]:
    """Async context manager for MCP client.

    Usage:
        async with mcp_client() as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_knowledge_graph", {"query": "Planner"})
    """
    client = MCPClient(server_urls)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()
