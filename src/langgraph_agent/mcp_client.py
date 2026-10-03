"""The agents' tool belts, served in-process under MCP-style tool names.

Every tool takes a dict of arguments and returns a JSON-serialisable dict. The
Researcher's node calls the two read-only GraphRAG tools; the Builder is bound
to the filesystem, git, terminal and test tools (`BUILDER_TOOLS` in nodes.py).

Usage:
    result = MCPClient().call_tool("search_knowledge_graph", {"query": "Planner"})
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph_agent.dwell import git_dwell

if TYPE_CHECKING:
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


# How long one `terminal_execute` command may run before it is killed. Above
# what this project's own scripts take, or working code comes back labelled
# broken; the Builder's deadline has to cover several such commands.
TERMINAL_TIMEOUT_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_SECONDS", "60"))

# The ceiling on a *requested* timeout. A tool call is never abandoned, so an
# unbounded request would hang the pass past every deadline. 600 is the longest
# thing the tool belt legitimately does, a test suite.
TERMINAL_TIMEOUT_MAX_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_MAX_SECONDS", "600"))


# The largest file `filesystem_read` returns whole: fifty times what reaches
# the model, so only a log or a dump is turned away.
FILESYSTEM_READ_MAX_BYTES = 1_000_000


def _resolve_timeout(requested: Any, default: float | None = None) -> float:
    """Clamp a requested command timeout to (0, TERMINAL_TIMEOUT_MAX_SECONDS].

    A missing, malformed or non-finite value falls back to `default`
    (`TERMINAL_TIMEOUT_SECONDS` unless named): the number comes from a model, and
    `min(nan, 600)` is `nan`, which `subprocess.run` reads as no timeout at all.
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


# A shell builtin reaches `subprocess` as a missing executable. The error names
# what replaces it -- `cwd` for the first group -- since an error is read at
# the moment of the mistake, which a schema description is not. The rest have
# no replacement: `export` cannot set a variable for a later call here.
_CWD_BUILTINS = frozenset({"cd", "pushd", "popd"})
_OTHER_BUILTINS = frozenset(
    {"source", ".", "export", "set", "unset", "alias", "eval", "exec"}
)


# Shell operators, recognised only as *whole argv tokens* after `shlex.split`,
# so a `;` inside a quoted `python -c "..."` is never mistaken for syntax.
# Redirection gets its own wording because it has a replacement and chaining
# does not.
_REDIRECTS = frozenset({">", ">>", "<", "<<", "2>", "2>>", "1>", "&>", ">&"})
_CHAINS = frozenset({"&&", "||", ";", "&", "|"})
SHELL_OPERATORS = _REDIRECTS | _CHAINS

# A redirect glued to its target (`2>/dev/null`), which `shlex` keeps as one
# token. Output redirection only: a glued input redirect cannot be told from an
# argument (`grep "<div>"`), and `=` may not follow `>`, so `>=1.0` passes. An
# argument that really begins with `>` cannot be passed, as `&&` cannot.
_GLUED_REDIRECT = re.compile(r"^[0-9&]?>>?(?:&[0-9]+|[^\s=>&]\S*)$")


def _is_shell_operator(token: str) -> bool:
    return token in SHELL_OPERATORS or _GLUED_REDIRECT.match(token) is not None


def _shell_operator_error(token: str) -> str:
    """Explain a shell operator that reached argv, and name what replaces it.

    There is no shell, so the operator is inert either way; refusing it is about
    the diagnosis. Left to run, the command fails somewhere unrelated -- `wc -l
    notes.md && tail -50 notes.md` comes back `wc: invalid option -- '5'`.
    """
    if token in _REDIRECTS or _GLUED_REDIRECT.match(token):
        # Discarding or merging a stream needs no replacement: both streams
        # come back as separate fields.
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
    """Say a program is missing, and what to do when it is a shell builtin instead."""
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

    With no shell, `cat dir/*` hands `cat` a file named `dir/*`. Raised only when
    the program quoted the token back in its error, so `find . -name "*.py"`,
    which wants the literal, is never flagged.
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


# The base every tool resolves a relative path against -- the server's working
# directory, the project root -- so one relative path means one place.
def _project_root() -> Path:
    return Path.cwd().resolve()


def _resolve_write_path(requested: Any) -> tuple[Path | None, str | None]:
    """Resolve a requested write path to `(path, error)`, refusing to escape the project.

    An error is returned *instead of* a path, never beside one. A write outside
    the project is invisible to the corpus and to git, and makes the run's
    `files_changed` name a path nobody can find from the project.

    The whole path is resolved before the write, leaf included, because only
    resolution knows where a symlink lands: `project/link/x` with `link` pointing
    at `/etc`, or a leaf that is itself a symlink out, would pass a check of the
    literal string. A leaf that does not exist yet is fine. `~` is not expanded:
    there is no shell here.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None, f"Invalid path: {requested!r}. Pass a file path inside the project."
    root = _project_root()
    candidate = Path(requested)
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

    An error comes back *instead of* a directory; both are None when nothing was
    requested. Checked here because `subprocess.run` reports a missing `cwd` as
    `FileNotFoundError` -- the same exception as a missing program, so the error
    would name the wrong thing. Relative to the project root, like every other
    tool; `~` is not expanded.
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


_Tool = Callable[[dict[str, Any]], dict[str, Any]]


class MCPClient:
    """Every tool either seat can be handed, by name."""

    def __init__(self) -> None:
        self._tools = self._discover_tools()

    def _discover_tools(self) -> dict[str, _Tool]:
        """The tool map. A Builder tool must also have a schema in `BUILDER_TOOLS`."""
        return {
            # Read-only: the Researcher never adds to the corpus.
            "search_knowledge_graph": self._graphrag_search,
            "query_knowledge_graph": self._graphrag_query_graph,
            "filesystem_read": self._filesystem_read,
            "filesystem_write": self._filesystem_write,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_dwell": self._git_dwell,
            "terminal_execute": self._terminal_execute,
            "run_tests": self._run_tests,
        }

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Run one tool. Raises ValueError for a name no tool has."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool(arguments)

    def _open_kb(self) -> GraphRAGKnowledgeBase | None:
        """The corpus if one has been built, `None` otherwise.

        `open_knowledge_base`, never `get_knowledge_base`: a Researcher's query is not
        a request for a knowledge base to be created.
        """
        try:
            from langgraph_agent.graphrag_server import open_knowledge_base

            return open_knowledge_base()
        except Exception:
            return None

    def _graphrag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search the knowledge base. With no corpus: no results, and a note saying why."""
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import absent_corpus

            return {"results": [], "source": "no_corpus", "note": absent_corpus()[1]}

        return {"results": kb.search(query, top_k), "source": "local_graphrag"}

    def _graphrag_query_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        """One entity's neighbourhood in the knowledge graph."""
        entity = args.get("entity", "")
        hops = args.get("hops", 2)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import absent_corpus

            return {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "no_corpus",
                "note": absent_corpus()[1],
            }

        result = kb.query_graph(entity, hops)
        result["source"] = "local_graphrag"
        return result

    def _filesystem_read(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read a file's contents.

        Not confined to the project the way `filesystem_write` is: `terminal_execute`
        runs any program, so a fence here would be a safety claim the tool belt cannot
        keep. Writes are confined for what they feed (`files_changed`, the corpus).
        A file over `FILESYSTEM_READ_MAX_BYTES` is refused unread, with how to read
        part of it.
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

    def _filesystem_write(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write a file inside the project, creating parent directories as needed (see
        `_resolve_write_path`).
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

    def _git_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """`git status --porcelain`, in `cwd` when given. The exit status is read first:
        a failed command and a clean tree both print nothing.
        """
        cwd, error = _resolve_cwd(args.get("cwd"))
        if error is not None:
            return {"success": False, "error": error}
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
        if result.returncode != 0:
            return {"success": False, "error": (result.stderr or result.stdout).strip()
                    or f"git status exited {result.returncode}"}
        return {"success": True, "status": result.stdout or "Working tree clean"}

    def _git_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        """`git diff`, of the whole tree or of one path.

        The path goes after `--`, only when there is one, so a path that also names a
        branch is not read as a revision. Run in `cwd` when given.
        """
        cwd, error = _resolve_cwd(args.get("cwd"))
        if error is not None:
            return {"success": False, "error": error}
        path = str(args.get("path") or "").strip()
        command = ["git", "diff"] + (["--", path] if path else [])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
        if result.returncode != 0:
            return {"success": False, "error": (result.stderr or result.stdout).strip()
                    or f"git diff exited {result.returncode}"}
        return {"success": True, "diff": result.stdout or "No changes"}

    def _run_vcs(
        self, *argv: str, timeout: float = 60.0, cwd: str | None = None
    ) -> tuple[bool, str]:
        """One git/gh invocation, in `cwd` when given. Returns `(ok, output)` with
        stderr folded in.

        stderr says the useful part -- "nothing to commit", a rejected push. No shell:
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

    def _git_dwell(self, args: dict[str, Any]) -> dict[str, Any]:
        """The git pipeline, from the working tree to a merged pull request (see dwell.py).

        In `cwd` when given -- only that directory's own repository: git looks
        upward for one, so a directory inside another repository is refused rather
        than committing the one around it.
        """
        cwd, cwd_error = _resolve_cwd(args.get("cwd"))
        if cwd_error is not None:
            return {"success": False, "error": cwd_error}
        return git_dwell(self._run_vcs, args, cwd)

    def _terminal_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run one program in the project workspace.

        There is no shell: the command is `shlex.split` and run as argv, so `;`, `|`,
        `>`, `&&`, `$(...)` and globs are inert data -- `echo hi; rm -rf /` runs `echo`
        with four arguments. An operator written on purpose is refused by name before
        anything runs (`_is_shell_operator`); a glob or `~` cannot be refused, since
        `find -name "*.py"` needs the literal, so a failure it causes is named in
        `hint`.

        `cwd` replaces `cd`, a builtin with no meaning here. `env` overlays the
        environment for this command (None removes a key) and is set only from inside
        the process, for headless verification. `timeout` is clamped by
        `_resolve_timeout`.
        """
        command = args.get("command", "")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            # Unbalanced quotes: say so, or "No closing quotation" reads like
            # the program failed.
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

        # Refused before the spawn. Only an operator that is its own token, or
        # a redirect glued to its target, is caught: in `echo hi; rm -rf /` the
        # `;` rides on `hi` and stays inert data.
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
                # No one is at the keyboard: a command that reads stdin gets
                # EOF rather than blocking until its timeout.
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
            # Keep what the command printed, so a hang at the start can be told
            # from one at the end.
            return {
                "success": False,
                "error": _timeout_error(e),
                "timed_out": True,
                "stdout": _as_captured_text(e.stdout),
                "stderr": _as_captured_text(e.stderr),
                "command": command,
            }
        except FileNotFoundError:
            # Name the program that was missing, and what replaces a builtin.
            return {
                "success": False,
                "error": _missing_program_error(argv[0]),
                "command": command,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "command": command}

    def _run_tests(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run pytest.

        With a `cwd` -- a generated project under `projects/` -- and no `path`, pytest
        collects from that directory; with neither, `tests/`. `timeout` is clamped
        like `terminal_execute`'s and defaults to the ceiling.
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
    """Say that a command timed out, and after how long -- nothing else.

    `str(TimeoutExpired)` puts the fact behind a repr of the whole argv, where the
    report line's length cut removes it.
    """
    return f"timed out after {exc.timeout:g} seconds"


def _as_captured_text(captured: str | bytes | None) -> str:
    """Output hung off a TimeoutExpired, as text ("" when nothing was read)."""
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
