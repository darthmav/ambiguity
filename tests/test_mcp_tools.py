"""Tests for the MCP tool bindings used by the 4-Agent System.

Verifies that the documented tool belts are exposed and functional:
- Researcher: search_knowledge_graph, query_knowledge_graph
- Builder: filesystem_read, filesystem_write, git_status, git_diff,
  terminal_execute, run_tests
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from langgraph_agent.mcp_client import (
    TERMINAL_TIMEOUT_MAX_SECONDS,
    TERMINAL_TIMEOUT_SECONDS,
    MCPClient,
    _missing_program_error,
    _resolve_cwd,
    _resolve_timeout,
    _unexpanded_hint,
)
from langgraph_agent.nodes import BUILDER_TOOL_NAMES


@pytest.fixture
def client() -> MCPClient:
    return MCPClient()


def test_every_tool_a_seat_is_offered_is_served(client: MCPClient):
    """A schema in `BUILDER_TOOLS` with no tool behind it fails at call time."""
    served = set(client._discover_tools())
    assert BUILDER_TOOL_NAMES <= served
    assert {"search_knowledge_graph", "query_knowledge_graph"} <= served


def test_filesystem_write_and_read(client: MCPClient, tmp_path, monkeypatch):
    """Builder can write and read files through MCP tools.

    The write happens inside the project root, because `_resolve_write_path`
    refuses anything else -- so the test moves the root rather than writing to
    a temp directory beside it, which is what it used to do.
    """
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "test.txt"
    content = "Hello from MCP filesystem tool"

    write_result = client.call_tool(
        "filesystem_write", {"path": "test.txt", "content": content}
    )
    assert write_result["success"]
    assert path.read_text(encoding="utf-8") == content

    read_result = client.call_tool("filesystem_read", {"path": str(path)})
    assert read_result["success"]
    assert read_result["content"] == content


def test_filesystem_write_refuses_an_absolute_path_outside_the_project(
    client: MCPClient, tmp_path, monkeypatch
):
    """The run of 2026-09-11 wrote `/tmp/gen_doc.py` while working on this checkout.

    A file outside the root is invisible to the reindex, to `corpus_staleness`
    and to git, and it makes the console's "changed this machine" notice name a
    path the operator cannot find from the project.
    """
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "escaped.py"

    result = client.call_tool(
        "filesystem_write", {"path": str(outside), "content": "print('nope')"}
    )

    assert not result["success"]
    assert "outside the project" in result["error"]
    assert not outside.exists(), "the refusal has to happen before the write"


def test_filesystem_write_refuses_a_dotdot_escape(
    client: MCPClient, tmp_path, monkeypatch
):
    """`..` is the other spelling of the same escape, and `resolve()` is what sees it."""
    monkeypatch.chdir(tmp_path)

    result = client.call_tool(
        "filesystem_write", {"path": "../escaped.py", "content": "print('nope')"}
    )

    assert not result["success"]
    assert "outside the project" in result["error"]
    assert not (tmp_path.parent / "escaped.py").exists()


def test_filesystem_write_refuses_a_symlinked_parent(
    client: MCPClient, tmp_path, monkeypatch
):
    """Resolution, not string matching, is what catches this one.

    `project/link/x` is under the root as a string and outside it on disk. A
    check on the literal path would pass it.
    """
    monkeypatch.chdir(tmp_path)
    target = tmp_path.parent / "elsewhere"
    target.mkdir()
    (tmp_path / "link").symlink_to(target)

    result = client.call_tool(
        "filesystem_write", {"path": "link/escaped.py", "content": "print('nope')"}
    )

    assert not result["success"]
    assert "outside the project" in result["error"]
    assert not (target / "escaped.py").exists()


def test_filesystem_write_creates_parent_directories_inside_the_project(
    client: MCPClient, tmp_path, monkeypatch
):
    """Containment must not cost the Builder the ability to make a subdirectory."""
    monkeypatch.chdir(tmp_path)

    result = client.call_tool(
        "filesystem_write", {"path": "reports/nested/out.md", "content": "ok"}
    )

    assert result["success"]
    assert (tmp_path / "reports" / "nested" / "out.md").read_text(encoding="utf-8") == "ok"


def test_git_tools(client: MCPClient):
    """Builder can call git status and diff."""
    status = client.call_tool("git_status", {})
    assert status["success"]
    assert "status" in status

    diff = client.call_tool("git_diff", {})
    assert diff["success"]
    assert "diff" in diff


def test_terminal_execute(client: MCPClient):
    """Builder can run safe shell commands."""
    result = client.call_tool("terminal_execute", {"command": "echo hello"})
    assert result["success"]
    assert "hello" in result["stdout"]


def test_terminal_execute_neutralises_injection(client: MCPClient):
    """A chained command is inert data, never a second command.

    There is no shell, so `;` separates nothing. This replaced a character
    filter that refused the input outright -- the canary surviving is a
    stronger guarantee than that refusal was, and unlike the refusal it does
    not also block `python -c "...; ..."`. Note the old filter admitted a bare
    `rm -rf /` quite happily: it never guarded destruction, only chaining.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        canary = Path(tmpdir) / "canary.txt"
        canary.write_text("still here", encoding="utf-8")

        result = client.call_tool(
            "terminal_execute", {"command": f"echo hello; rm -rf {tmpdir}"}
        )

        # `echo` received the rest as literal arguments and printed them.
        assert result["success"]
        assert "rm -rf" in result["stdout"]

        # The part that matters: nothing was deleted.
        assert canary.exists()
        assert canary.read_text(encoding="utf-8") == "still here"


def test_terminal_execute_runs_a_python_one_liner(client: MCPClient):
    """The exact shape the old filter refused: `;` and `()` in a -c argument."""
    result = client.call_tool(
        "terminal_execute",
        {"command": 'python -c "import sys; print(sys.version_info[0])"'},
    )
    assert result["success"], result.get("error") or result.get("stderr")
    assert result["stdout"].strip() == "3"


def test_terminal_execute_refuses_a_shell_operator_and_names_it(
    client: MCPClient,
):
    """An operator written as its own word is refused, with the reason.

    This used to be accepted and inert -- `echo a | wc -l` ran `echo` and
    printed `a | wc -l` -- on the grounds that removing the shell removed the
    danger. It did, and that was never the problem: the problem is the
    *diagnosis*. `wc -l notes.md && tail -50 notes.md` hands `wc` the arguments
    `&&`, `tail` and `-50` and comes back `wc: invalid option -- '5'`, having
    never counted the file it was given, with nothing in the message naming the
    chain. The old tool description already warned about this; the Builder
    still reached for `cd there && ...` three times on the 2026-09-08 rerun
    before using `cwd`. A description is consulted before the turn, an error at
    the moment of the mistake.
    """
    chained = client.call_tool(
        "terminal_execute", {"command": "echo a && echo b"}
    )
    assert not chained["success"]
    assert "&&" in chained["error"]
    assert "cwd" in chained["error"]

    piped = client.call_tool("terminal_execute", {"command": "echo a | wc -l"})
    assert not piped["success"]
    assert "pipe" in piped["error"]

    # Redirection is answered with the tool that replaces it, the way `cd` is
    # answered with `cwd`; chaining has no replacement and is not given one.
    redirected = client.call_tool(
        "terminal_execute", {"command": "echo a > out.txt"}
    )
    assert not redirected["success"]
    assert "filesystem_write" in redirected["error"]


def test_an_operator_inside_an_argument_is_still_just_text(
    client: MCPClient,
):
    """The precision that separates this from the filter it replaced.

    The old character whitelist scanned the raw string, so it refused
    `python -c "import x; print(y)"` over a `;` that was never syntax. The
    check runs *after* `shlex.split` and matches whole tokens, so an operator
    inside a quoted argument is untouched -- and a filename containing one is
    still a filename.
    """
    quoted = client.call_tool(
        "terminal_execute",
        {"command": 'python -c "print(\'a && b | c > d\')"'},
    )
    assert quoted["success"], quoted.get("error") or quoted.get("stderr")
    assert quoted["stdout"].strip() == "a && b | c > d"


def test_a_glued_redirect_is_refused_like_a_spaced_one(client: MCPClient):
    """`2>/dev/null` is one token to `shlex`, so the whole-token check missed it.

    The Builder on the 2026-09-10 run wrote that suffix nine times in one pass,
    and the one no other operator was caught ahead of reached `find` as a
    literal argument: `paths must precede expression: '2>/dev/null'`.
    """
    discarded = client.call_tool(
        "terminal_execute", {"command": "find . -maxdepth 0 2>/dev/null"}
    )
    assert not discarded["success"]
    assert "2>/dev/null" in discarded["error"]
    # Discarding a stream needs no replacement: both come back separately.
    assert "separately" in discarded["error"]

    merged = client.call_tool("terminal_execute", {"command": "echo a 2>&1"})
    assert not merged["success"]
    assert "separately" in merged["error"]

    written = client.call_tool("terminal_execute", {"command": "echo a >x.txt"})
    assert not written["success"]
    assert "filesystem_write" in written["error"]


def test_an_argument_that_only_resembles_a_redirect_passes(client: MCPClient):
    """Only output redirection is read into a glued token, never a comparison.

    An input redirect glued to its target cannot be told from markup a grep is
    looking for, and `>=1` is a version bound.
    """
    for argument in (">=1", "<div>", "->"):
        echoed = client.call_tool(
            "terminal_execute",
            {"command": f'python -c "import sys; print(sys.argv[1])" "{argument}"'},
        )
        assert echoed["success"], echoed.get("error") or echoed.get("stderr")
        assert echoed["stdout"].strip() == argument


def test_an_unexpanded_glob_is_named_when_the_program_trips_on_it(
    client: MCPClient, tmp_path: Path
):
    """With no shell, `cat dir/*` asks `cat` for a file literally named that.

    Its answer -- no such file or directory, about a directory that exists --
    is true of the name it was given and false of what was meant.
    """
    (tmp_path / "manifest").write_text("x")

    missed = client.call_tool("terminal_execute", {"command": f"cat {tmp_path}/*"})
    assert not missed["success"]
    assert "glob" in missed["hint"]

    # A literal the program wants is not a mistake, and earns no hint.
    wanted = client.call_tool(
        "terminal_execute", {"command": f'find {tmp_path} -name "*"'}
    )
    assert wanted["success"], wanted.get("stderr")
    assert "hint" not in wanted


def test_the_glob_hint_needs_the_program_to_have_quoted_the_token():
    """Only the program's own quoted complaint about a token earns the hint."""
    assert _unexpanded_hint(["cat", "d/*"], "cat: 'd/*': No such file or directory")
    assert _unexpanded_hint(["ls", "~/x"], "ls: cannot access '~/x': No such file")
    # A regex that failed to compile is reported unquoted: not a glob problem.
    assert _unexpanded_hint(["grep", "["], "grep: Unmatched [, [^, [:, [., or [=") is None
    # A token holding nothing a shell would expand is never named.
    assert _unexpanded_hint(["cat", "plain"], "cat: 'plain': No such file") is None


def test_terminal_execute_reports_unparseable_and_missing_commands(
    client: MCPClient,
):
    """Two failures the shell used to fold into a return code."""
    unbalanced = client.call_tool(
        "terminal_execute", {"command": 'python -c "print(1)'}
    )
    assert not unbalanced["success"]
    assert "quoting" in unbalanced["error"].lower()

    empty = client.call_tool("terminal_execute", {"command": "   "})
    assert not empty["success"]

    missing = client.call_tool(
        "terminal_execute", {"command": "no-such-program-xyzzy --help"}
    )
    assert not missing["success"]
    # Names the program, not its arguments.
    assert "no-such-program-xyzzy" in missing["error"]


def test_terminal_execute_honours_a_requested_timeout(client: MCPClient):
    """A caller-supplied timeout bounds the command and reports the kill."""
    result = client.call_tool(
        "terminal_execute", {"command": "sleep 30", "timeout": 1}
    )
    assert not result["success"]
    assert result["timed_out"] is True


def test_a_timeout_survives_into_the_report_line(client: MCPClient):
    """The report line must say it timed out, however long the command was.

    `str(TimeoutExpired)` puts the fact behind a repr of the whole argv, and
    the report cuts its reason at MAX_FAILURE_REASON_CHARS -- so a long
    command, like the run of 2026-09-10's `pip3 install ... --extra-index-url
    https://...`, came back reading only its own arguments.
    """
    from langgraph_agent.nodes import _failure_reason

    padding = "x" * 200
    result = client.call_tool(
        "terminal_execute",
        {"command": f'python -c "import time; time.sleep(30)  # {padding}"', "timeout": 1},
    )

    assert result["timed_out"] is True
    assert _failure_reason(result) == "timed out after 1 seconds"


def test_terminal_timeout_clears_this_project_own_scripts():
    """The default must outlast the scripts this repo tells people to run.

    Pinned against the measurement rather than the literal number, the way
    the relevance floor's calibration is: `scripts/verify_and_test.py` completes clean
    in ~33s, and the old default of 30 reported it FAILED for the difference.
    A default at or under that is the false accusation coming back.
    """
    assert TERMINAL_TIMEOUT_SECONDS > 35
    assert TERMINAL_TIMEOUT_MAX_SECONDS >= TERMINAL_TIMEOUT_SECONDS


def test_terminal_timeout_is_clamped_not_trusted():
    """A requested timeout is bounded; a malformed one falls back."""
    # Honoured between the floor and the ceiling.
    assert _resolve_timeout(120) == 120

    # Capped: a tool call is never abandoned, so an unbounded request would
    # hang the pass past every deadline there is.
    assert _resolve_timeout(99999) == TERMINAL_TIMEOUT_MAX_SECONDS

    # Malformed, absent or non-positive falls back rather than raising.
    for bad in (None, "not-a-number", 0, -5, float("nan"), float("inf")):
        assert _resolve_timeout(bad) == TERMINAL_TIMEOUT_SECONDS


def test_terminal_execute_runs_in_a_requested_cwd(client: MCPClient):
    """`cwd` is the replacement for a `cd` that cannot exist.

    Removing the shell removed the only spelling the Builder had for "run this
    somewhere else": `cd there && python x.py` reports `Command not found:
    'cd'`, which is true and useless.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "marker.txt").write_text("here", encoding="utf-8")

        result = client.call_tool(
            "terminal_execute",
            {
                "command": 'python -c "import pathlib; print(pathlib.Path.cwd())"',
                "cwd": tmpdir,
            },
        )
        assert result["success"], result.get("error") or result.get("stderr")
        assert Path(result["stdout"].strip()).samefile(tmpdir)

        # And the command sees that directory's files by relative path.
        read = client.call_tool(
            "terminal_execute",
            {"command": "cat marker.txt", "cwd": tmpdir},
        )
        assert read["success"], read.get("error") or read.get("stderr")
        assert read["stdout"].strip() == "here"


def test_terminal_execute_blames_a_bad_cwd_not_the_program(
    client: MCPClient,
):
    """The directory is named, and the program is not accused of missing.

    An unchecked `cwd` reaches `subprocess.run`, which raises
    `FileNotFoundError` for a missing directory -- indistinguishable, at the
    handler, from a missing program, and answered `Command not found:
    'python'` while python was fine.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = str(Path(tmpdir) / "no-such-dir")
        result = client.call_tool(
            "terminal_execute", {"command": "python --version", "cwd": missing}
        )
        assert not result["success"]
        assert missing in result["error"]
        assert "python" not in result["error"].lower()

        # A file is not a directory, and raises something else again.
        a_file = Path(tmpdir) / "file.txt"
        a_file.write_text("x", encoding="utf-8")
        on_file = client.call_tool(
            "terminal_execute", {"command": "python --version", "cwd": str(a_file)}
        )
        assert not on_file["success"]
        assert str(a_file) in on_file["error"]
        assert "not a directory" in on_file["error"].lower()


def test_resolve_cwd_returns_a_directory_or_an_error_never_both():
    """Absent means inherit; anything malformed is refused rather than raised."""
    assert _resolve_cwd(None) == (None, None)

    with tempfile.TemporaryDirectory() as tmpdir:
        resolved, error = _resolve_cwd(tmpdir)
        assert error is None
        assert resolved is not None and Path(resolved).samefile(tmpdir)

    # The value arrives as JSON from a model, so a wrong type is a refusal
    # with a message, not a TypeError out of the tool call.
    for bad in ("", "   ", 5, ["/tmp"], {}):
        resolved, error = _resolve_cwd(bad)
        assert resolved is None
        assert error


def test_terminal_execute_points_a_builtin_at_its_replacement(
    client: MCPClient,
):
    """`cd` is the mistake the Builder actually makes, so the error answers it.

    Measured on the rerun after `cwd` shipped: three turns spent on
    `cd there && ...` before the Builder found the argument that replaces it.
    The schema said so; an error is read at the moment the mistake is made.
    """
    result = client.call_tool(
        "terminal_execute", {"command": "cd /tmp && python --version"}
    )
    assert not result["success"]
    assert "cd" in result["error"]
    assert "cwd" in result["error"]


def test_missing_program_error_names_a_fix_only_where_one_exists():
    """A builtin with no replacement gets the fact, not an invented alternative."""
    # Points at the argument that does the job.
    for builtin in ("cd", "pushd", "popd"):
        message = _missing_program_error(builtin)
        assert "builtin" in message
        assert "`cwd`" in message

    # No `cwd` to offer: `export` cannot set a variable for a later call here,
    # and no wording makes it able to. Say what ends the retry instead.
    export = _missing_program_error("export")
    assert "builtin" in export
    assert "cwd" not in export

    # An ordinary missing program is still named, and gains nothing.
    plain = _missing_program_error("no-such-program-xyzzy")
    assert "no-such-program-xyzzy" in plain
    assert "builtin" not in plain


def test_report_line_names_the_directory_a_command_ran_in():
    """The Architect rules on the report, so a line must be rulable on.

    `find . -type f -> ok` locates nothing once `cwd` exists: the same
    relative command means a different thing in every directory it could have
    run in, and the report was the only place anyone would find out.
    """
    from langgraph_agent.nodes import _Deadline, _run_builder_tools

    class _OneCall:
        """Asks for two commands on the first turn, then stops asking."""

        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn > 1:
                return SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "terminal_execute",
                        "args": {"command": "python --version", "cwd": tmpdir},
                        "id": "call-1",
                    },
                    {
                        "name": "terminal_execute",
                        "args": {"command": "python --version"},
                        "id": "call-2",
                    },
                ],
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        tool_log: list[str] = []
        _run_builder_tools(_OneCall(), [], [], tool_log, _Deadline(60))

    with_cwd, without_cwd = tool_log
    assert f"[cwd={tmpdir}]" in with_cwd
    assert with_cwd.endswith("-> ok")
    # A command that did not ask for one says nothing, rather than naming a
    # default the Builder never chose.
    assert "cwd" not in without_cwd


def test_report_line_says_why_a_call_failed():
    """`-> failed` alone read the same for a refused pipe and a broken machine.

    Eleven of the forty-seven calls on the 2026-09-10 run said only that, and
    eight of them were the tool correctly refusing shell syntax.
    """
    from langgraph_agent.nodes import _Deadline, _run_builder_tools

    class _TwoCalls:
        """Asks for a refused command and a working one, then stops asking."""

        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn > 1:
                return SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "terminal_execute",
                        "args": {"command": "echo a | wc -l"},
                        "id": "call-1",
                    },
                    {
                        "name": "terminal_execute",
                        "args": {"command": "python --version"},
                        "id": "call-2",
                    },
                ],
            )

    tool_log: list[str] = []
    _run_builder_tools(_TwoCalls(), [], [], tool_log, _Deadline(60))

    refused, ran = tool_log
    assert "-> failed: " in refused and "pipe" in refused
    assert ran.endswith("-> ok")


def test_the_builder_cannot_call_a_researcher_tool(monkeypatch):
    """The client serves GraphRAG too; the Builder's loop refuses it unrun."""
    from langgraph_agent import nodes

    def must_not_run(name, args):
        raise AssertionError(f"{name} ran for the Builder")

    monkeypatch.setattr(nodes, "_call_tool", must_not_run)

    class _Searches:
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn > 1:
                return SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(content="", tool_calls=[{
                "name": "search_knowledge_graph", "args": {"query": "x"}, "id": "c1",
            }])

    tool_log: list[str] = []
    nodes._run_builder_tools(_Searches(), [], [], tool_log, nodes._Deadline(60))

    assert tool_log == [
        "search_knowledge_graph() -> failed: search_knowledge_graph is not a Builder tool"
    ]


def test_a_failure_reason_is_one_line_in_the_tools_own_words():
    """The tool's own error first, then the last thing the program said."""
    from langgraph_agent.nodes import MAX_FAILURE_REASON_CHARS, _failure_reason

    traceback = "Traceback (most recent call last):\n  File \"x\"\nValueError: bad\n"
    assert _failure_reason({"success": False, "returncode": 1, "stderr": traceback}) == (
        "ValueError: bad"
    )
    assert _failure_reason({"success": False, "error": "refused\nand why"}) == "refused"
    assert _failure_reason({"success": False, "returncode": 3, "stderr": ""}) == "exit 3"
    assert len(_failure_reason({"success": False, "error": "x" * 500})) == (
        MAX_FAILURE_REASON_CHARS
    )


def test_builder_can_ask_for_a_working_directory():
    """The schema must expose `cwd`, or the tool can do what the seat cannot ask.

    Exactly the shape the `timeout` gap had: `_terminal_execute` honours the
    argument, and a BUILDER_TOOLS entry omitting it leaves the Builder with no
    way to know it exists -- reaching for `cd` instead, which cannot work.
    """
    from langgraph_agent.nodes import BUILDER_TOOLS

    tool = next(
        t for t in BUILDER_TOOLS if t["function"]["name"] == "terminal_execute"
    )
    params = tool["function"]["parameters"]
    assert "cwd" in params["properties"]
    assert "cwd" not in params.get("required", [])
    # Says why, where the seat reading the tool can see it.
    assert "cd" in tool["function"]["description"]


def test_builder_can_ask_for_a_longer_timeout():
    """The schema must expose `timeout`, or the default is a hard ceiling.

    This is what actually broke: `_terminal_execute` read
    `args.get("timeout")`, but BUILDER_TOOLS offered only `command`, so the
    Builder could not raise it and had no way to learn the limit existed.
    """
    from langgraph_agent.nodes import BUILDER_TOOLS

    tool = next(
        t for t in BUILDER_TOOLS if t["function"]["name"] == "terminal_execute"
    )
    params = tool["function"]["parameters"]
    assert "timeout" in params["properties"]
    assert "timeout" not in params.get("required", [])
    # The limit is stated where the seat reading the tool can see it.
    assert str(int(TERMINAL_TIMEOUT_SECONDS)) in tool["function"]["description"]


def test_run_tests(client: MCPClient):
    """Builder can run the pytest suite via the test tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal passing test so the tool has something to execute.
        test_file = Path(tmpdir) / "test_dummy.py"
        test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        result = client.call_tool("run_tests", {"path": str(tmpdir)})
        assert result["success"], result.get("stderr", "")
        assert "passed" in result.get("stdout", "")


def test_nodes_reach_the_tool_belt_through_one_seam(tmp_path, monkeypatch):
    """`nodes._call_tool` is the door every node uses, and what the graph tests patch."""
    monkeypatch.chdir(tmp_path)
    from langgraph_agent.nodes import _call_tool

    result = _call_tool("filesystem_write", {"path": "seam.txt", "content": "ok"})

    assert result["success"]
    assert (tmp_path / "seam.txt").read_text(encoding="utf-8") == "ok"


def _git_repo(path: Path) -> None:
    """A repository with one committed file, isolated from the user's git config."""
    import subprocess

    def git(*argv: str) -> None:
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-c", "user.email=t@example.com",
             "-c", "user.name=t", *argv],
            cwd=path, check=True, capture_output=True,
        )

    git("init", "-q")
    (path / "a.txt").write_text("one\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-qm", "init")


def test_git_diff_with_no_path_shows_the_whole_diff(client: MCPClient, tmp_path, monkeypatch):
    """`git_diff()` ran `git diff ""`, which git refuses outright.

    The exit status went unread, so the refusal's empty stdout came back as
    success with "No changes" -- on every call, however much had changed.
    """
    _git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("two\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    whole = client.call_tool("git_diff", {})
    one = client.call_tool("git_diff", {"path": "a.txt"})

    assert whole["success"] and "+two" in whole["diff"]
    assert one["success"] and "+two" in one["diff"]


def test_git_status_outside_a_repository_is_a_failure(client: MCPClient, tmp_path, monkeypatch):
    """Nothing on stdout is what a clean tree prints -- and what a failure prints."""
    monkeypatch.chdir(tmp_path)

    status = client.call_tool("git_status", {})

    assert not status["success"]
    assert "not a git repository" in status["error"]


def test_run_tests_runs_the_suite_in_the_directory_it_is_given(
    client: MCPClient, tmp_path, monkeypatch
):
    """A generated project's Builder is told to pass `cwd`, and the tool ignored it.

    It ran this checkout's own `tests/` instead, reporting on a suite the
    project never had while the project's own went unrun. Asserted on what
    `subprocess.run` is handed rather than by running pytest: under the old
    behaviour a real run is this very suite, which contains this test, and it
    recursed.
    """
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return SimpleNamespace(returncode=0, stdout="1 passed", stderr="")

    monkeypatch.setattr("langgraph_agent.mcp_client.subprocess.run", fake_run)
    project = tmp_path / "proj"
    project.mkdir()

    result = client.call_tool("run_tests", {"cwd": str(project)})
    missing = client.call_tool("run_tests", {"cwd": str(tmp_path / "nope")})
    client.call_tool("run_tests", {})

    assert result["success"]
    (in_project, in_project_kwargs), (checkout, checkout_kwargs) = calls
    assert in_project_kwargs["cwd"] == str(project)
    assert in_project[-2:] == [".", "-q"]
    assert checkout_kwargs["cwd"] is None and checkout[-2:] == ["tests/", "-q"]
    # Refused before anything is spawned, naming the directory's problem.
    assert not missing["success"] and "does not exist" in missing["error"]


def test_run_tests_clamps_the_timeout_it_is_asked_for(client: MCPClient, monkeypatch):
    """A tool call is never abandoned, so an unclamped timeout outlives every deadline."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="1 passed", stderr="")

    monkeypatch.setattr("langgraph_agent.mcp_client.subprocess.run", fake_run)

    for asked in (99999, None, "soon"):
        client.call_tool("run_tests", {} if asked is None else {"timeout": asked})
        # A malformed value falls back to the suite's own default, the ceiling,
        # rather than to the one-command default a test suite would outrun.
        assert seen["timeout"] == TERMINAL_TIMEOUT_MAX_SECONDS


def test_a_file_too_large_to_read_whole_is_refused_with_a_way_to_read_part(
    client: MCPClient, tmp_path, monkeypatch
):
    """A dump the size of the disk was read into memory whole, to be cut at 20,000."""
    from langgraph_agent import mcp_client as module

    monkeypatch.setattr(module, "FILESYSTEM_READ_MAX_BYTES", 10)
    (tmp_path / "big.log").write_text("x" * 11, encoding="utf-8")
    (tmp_path / "small.txt").write_text("0123456789", encoding="utf-8")

    refused = client.call_tool("filesystem_read", {"path": str(tmp_path / "big.log")})
    read = client.call_tool("filesystem_read", {"path": str(tmp_path / "small.txt")})

    assert not refused["success"] and "head -n 200" in refused["error"]
    assert read["success"] and read["content"] == "0123456789"


def test_builder_can_ask_for_a_directory_to_run_tests_in():
    """The schema must expose `cwd`, or OUTPUT_DIR_NOTE asks for what cannot be sent."""
    from langgraph_agent.nodes import BUILDER_TOOLS, OUTPUT_DIR_NOTE

    tool = next(t for t in BUILDER_TOOLS if t["function"]["name"] == "run_tests")

    assert "cwd" in tool["function"]["parameters"]["properties"]
    assert "run_tests" in OUTPUT_DIR_NOTE
