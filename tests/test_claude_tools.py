"""Claude Code set up to drive the console: what scripts/claude_tools.sh does.

Run against stand-ins for `claude`, `npx`, `git`, `curl` and the venv's python
on a PATH of their own, with HOME in a temporary directory, so nothing here
installs, signs in or registers anything: every call the script makes is
recorded with its arguments, the directory it ran in and which key variables
its environment still held, and the stand-in `claude` keeps its sign-in and its
MCP registration in files the tests read back.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "claude_tools.sh"
KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
PIN = "@playwright/mcp@0.0.83"
MANIFEST = "com.anthropic.claude_code_browser_extension.json"

# One stand-in for every tool, told apart by the name it is run under. Each call
# is a line in calls.jsonl; behaviour is steered by files and STANDIN_ variables.
STANDIN = r'''#!{python}
import json, os, sys
from pathlib import Path

state = Path(os.environ["STANDIN_STATE"])
tool = Path(sys.argv[0]).name
argv = sys.argv[1:]
keys = [k for k in {keys!r} if k in os.environ]
with open(state / "calls.jsonl", "a") as log:
    log.write(json.dumps({{"tool": tool, "argv": argv, "keys": keys, "cwd": os.getcwd()}}) + "\n")

if tool == "claude":
    if argv == ["--version"]:
        print("2.1.295 (Claude Code)")
    elif argv[:2] == ["auth", "status"]:
        auth = state / "auth"
        method = auth.read_text().strip() if auth.exists() else "none"
        print(json.dumps({{"loggedIn": method != "none", "authMethod": method}}, indent=2))
        sys.exit(0 if method != "none" else 1)
    elif argv[:2] == ["auth", "login"]:
        if os.environ.get("STANDIN_LOGIN_FAILS"):
            sys.exit(1)
        (state / "auth").write_text("claude.ai")
    elif argv == ["mcp", "get", "playwright"]:
        mcp = state / "mcp"
        if not mcp.exists():
            print('No MCP server named "playwright". Run `claude mcp add` to add one.', file=sys.stderr)
            sys.exit(1)
        command, args = json.loads(mcp.read_text())
        print("playwright:\n  Scope: Local config (private to you in this project)\n"
              "  Status: ✓ Connected\n  Type: stdio\n"
              f"  Command: {{command}}\n  Args: {{' '.join(args)}}\n  Environment:\n\n"
              'To remove this server, run: claude mcp remove "playwright" -s local')
    elif argv[:2] == ["mcp", "add"]:
        rest = argv[argv.index("--") + 1:]
        (state / "mcp").write_text(json.dumps([rest[0], rest[1:]]))
        print("Added stdio MCP server playwright to local config")
elif tool == "npx":
    sys.exit(int(os.environ.get("STANDIN_NPX_STATUS", "0")))
elif tool == "git":
    email = os.environ.get("STANDIN_GIT_EMAIL", "")
    if argv == ["config", "user.email"] and email:
        print(email)
    else:
        sys.exit(1)
elif tool == "curl":
    # What the native installer amounts to here: claude, in ~/.local/bin.
    print('mkdir -p "$HOME/.local/bin" && cp "$STANDIN_STATE/standin" "$HOME/.local/bin/claude"'
          ' && chmod 755 "$HOME/.local/bin/claude"')
elif tool == "python":
    print("mcp-check stand-in")
    sys.exit(int(os.environ.get("STANDIN_MCP_CHECK_STATUS", "0")))
'''

# What the script runs besides the stand-ins. Linked one by one into a
# directory of their own, so a real claude, npx or Chrome on this machine --
# CI runners ship Chrome, an Arch machine with npm has /usr/bin/npx -- is never
# on the PATH the script sees.
SYSTEM_TOOLS = ("bash", "sh", "env", "timeout", "sed", "head", "tail", "grep",
                "readlink", "dirname", "basename", "mkdir", "cp", "chmod")


@pytest.fixture
def host(tmp_path):
    """A HOME, a PATH of stand-ins and system tools, and the call log."""
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "calls.jsonl").write_text("")
    standin = state / "standin"
    standin.write_text(STANDIN.format(python=sys.executable, keys=KEYS))
    standin.chmod(0o755)

    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    for tool in ("claude", "npx", "git", "curl"):
        shutil.copy(standin, stand_ins / tool)
    venv_python = tmp_path / "venv" / "python"
    venv_python.parent.mkdir()
    shutil.copy(standin, venv_python)
    browser = tmp_path / "chromium-bin" / "chromium"
    browser.parent.mkdir()
    browser.write_text("#!/bin/sh\nexit 0\n")
    browser.chmod(0o755)

    system = tmp_path / "system"
    system.mkdir()
    for tool in SYSTEM_TOOLS:
        found = shutil.which(tool)
        assert found, f"{tool} is not on this machine's PATH"
        (system / tool).symlink_to(found)

    env = {
        "HOME": str(home),
        "PATH": f"{stand_ins}:{system}",
        "LANG": "C.UTF-8",
        "STANDIN_STATE": str(state),
        "STANDIN_GIT_EMAIL": "someone@example.org",
        "CLAUDE_TOOLS_PYTHON": str(venv_python),
        "CLAUDE_TOOLS_SYSTEM_CHROMIUM": str(tmp_path / "no-such-chromium"),
        "CLAUDE_TOOLS_INTERACTIVE": "0",
        "BROWSER_AGENT_CHROMIUM": str(browser),
        "CLAUDE_TOOLS_NOTES": str(tmp_path / "notes.txt"),
    }

    class Host:
        pass

    h = Host()
    h.env, h.home, h.state, h.bin, h.browser, h.notes = (
        env, home, state, stand_ins, browser, tmp_path / "notes.txt")
    return h


def _run(host, *args, stdin=""):
    return subprocess.run(["bash", str(SCRIPT), *args], env=host.env, input=stdin,
                          capture_output=True, text=True, timeout=60)


def _calls(host, tool=None):
    lines = (host.state / "calls.jsonl").read_text().splitlines()
    calls = [json.loads(line) for line in lines if line]
    return [c for c in calls if tool is None or c["tool"] == tool]


def _notes(host):
    return host.notes.read_text().splitlines() if host.notes.exists() else []


def _expected_mcp_args(browser):
    return ["-y", PIN, "--executable-path", str(browser), "--isolated",
            "--output-dir", str(ROOT / "reports" / "diagnostics" / "playwright-mcp")]


def _register(host, command="npx", args=None):
    (host.state / "mcp").write_text(json.dumps([command, args or _expected_mcp_args(host.browser)]))


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_install_leaves_an_installed_claude_alone(host):
    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert _calls(host, "curl") == [], "the native installer ran with claude already there"
    assert "Claude Code 2.1.295" in done.stdout


def test_yes_installs_a_missing_claude_with_the_native_installer(host):
    (host.bin / "claude").unlink()

    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert [c["argv"] for c in _calls(host, "curl")] == [["-fsSL", "https://claude.ai/install.sh"]]
    assert (host.home / ".local" / "bin" / "claude").exists()
    # Found there for the rest of the run, without touching the shell's config...
    assert any(c["tool"] == "claude" and c["argv"][:2] == ["mcp", "add"] for c in _calls(host))
    assert not list(host.home.glob(".*rc")) and not (host.home / ".profile").exists()
    # ...and the PATH that lacks it is named instead.
    assert any(".local/bin" in n and "PATH" in n for n in _notes(host))


@pytest.mark.parametrize("interactive, answer, said", [
    ("0", "y\n", "without a terminal nothing asked"),
    ("1", "n\n", "was not installed, as asked"),
    ("1", "", "was not installed, as asked"),
])
def test_a_missing_claude_is_installed_only_on_a_yes(host, interactive, answer, said):
    (host.bin / "claude").unlink()
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = interactive

    declined = _run(host, "install", stdin=answer)

    # Not a failure: nothing was tried, and the note says how to add it later.
    assert declined.returncode == 0, declined.stdout + declined.stderr
    assert _calls(host, "curl") == []
    assert any(said in n for n in _notes(host)), _notes(host)
    assert _calls(host, "npx") == [], "the MCP server was set up for a claude that is not there"


def test_an_interactive_yes_installs_it(host):
    (host.bin / "claude").unlink()
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"

    done = _run(host, "install", stdin="y\n")

    assert len(_calls(host, "curl")) == 1, done.stdout + done.stderr


def test_more_than_one_claude_on_path_is_named(host):
    shadow = host.home / "shims"
    shadow.mkdir()
    shutil.copy(host.bin / "claude", shadow / "claude")
    host.env["PATH"] = f"{shadow}:{host.env['PATH']}"

    _run(host, "install", "--yes")

    assert any("more than one claude" in n for n in _notes(host))


# ---------------------------------------------------------------------------
# the sign-in
# ---------------------------------------------------------------------------


def test_every_claude_call_runs_without_the_key_variables(host):
    for key in KEYS:
        host.env[key] = "planted-" + key.lower()
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"

    done = _run(host, "install")

    claude_calls = _calls(host, "claude")
    assert any(c["argv"][:2] == ["auth", "status"] for c in claude_calls)
    assert any(c["argv"][:2] == ["auth", "login"] for c in claude_calls), done.stdout
    leaked = [c for c in claude_calls if c["keys"]]
    assert not leaked, f"claude saw key variables: {leaked}"


def test_an_interactive_login_is_claude_ai_with_the_git_email(host):
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"

    done = _run(host, "install")

    logins = [c["argv"] for c in _calls(host, "claude") if c["argv"][:2] == ["auth", "login"]]
    assert logins == [["auth", "login", "--claudeai", "--email", "someone@example.org"]]
    assert "signed in with claude.ai" in done.stdout
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_noreply_address_is_not_filled_in(host):
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"
    host.env["STANDIN_GIT_EMAIL"] = "12345+someone@users.noreply.github.com"

    _run(host, "install")

    logins = [c["argv"] for c in _calls(host, "claude") if c["argv"][:2] == ["auth", "login"]]
    assert logins == [["auth", "login", "--claudeai"]]


def test_a_login_that_does_not_take_fails_the_install(host):
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"
    host.env["STANDIN_LOGIN_FAILS"] = "1"

    done = _run(host, "install")

    assert done.returncode == 1
    assert "still not signed in with claude.ai" in done.stdout


@pytest.mark.parametrize("interactive", ["0", "1"])
def test_yes_never_signs_in_and_leaves_a_note(host, interactive):
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = interactive

    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert not any(c["argv"][:2] == ["auth", "login"] for c in _calls(host, "claude"))
    assert any("claude auth login --claudeai" in n for n in _notes(host))


def test_an_api_key_sign_in_does_not_count(host):
    (host.state / "auth").write_text("api_key")

    _run(host, "install", "--yes")

    assert any("not signed in with claude.ai (api_key)" in n for n in _notes(host))


def test_a_claude_ai_sign_in_is_left_alone(host):
    (host.state / "auth").write_text("claude.ai")
    host.env["CLAUDE_TOOLS_INTERACTIVE"] = "1"

    done = _run(host, "install")

    assert not any(c["argv"][:2] == ["auth", "login"] for c in _calls(host, "claude"))
    assert "signed in with claude.ai" in done.stdout


# ---------------------------------------------------------------------------
# the Playwright MCP server
# ---------------------------------------------------------------------------


def test_the_server_is_registered_with_the_pinned_command_line(host):
    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    adds = [c for c in _calls(host, "claude") if c["argv"][:2] == ["mcp", "add"]]
    assert [c["argv"] for c in adds] == [
        ["mcp", "add", "--scope", "local", "playwright", "--", "npx",
         *_expected_mcp_args(host.browser)]
    ]
    # Local scope is keyed by the project's path, so it is asked from the root.
    assert {c["cwd"] for c in _calls(host, "claude") if c["argv"][0] == "mcp"} == {str(ROOT)}
    # Fetched into npx's cache first, at the same pin.
    assert [c["argv"] for c in _calls(host, "npx")] == [["-y", PIN, "--help"]]


def test_a_registered_server_is_not_added_again(host):
    _register(host)

    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert not any(c["argv"][:2] == ["mcp", "add"] for c in _calls(host, "claude"))
    assert "registered for this checkout" in done.stdout


def test_a_server_registered_otherwise_is_left_and_named(host):
    _register(host, args=["-y", "@playwright/mcp@0.0.1", "--headless"])

    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert not any(c["argv"][1:2] in (["add"], ["remove"]) for c in _calls(host, "claude"))
    note = next(n for n in _notes(host) if "already registered with other arguments" in n)
    assert 'claude mcp remove "playwright" -s local' in note
    assert PIN in note and str(host.browser) in note


def test_the_browser_is_resolved_in_order(host):
    system = host.home / "usr-lib-chromium"
    system.write_text("#!/bin/sh\n")
    system.chmod(0o755)
    on_path = host.bin / "chromium"
    shutil.copy(host.browser, on_path)

    def registered_browser():
        (host.state / "mcp").unlink(missing_ok=True)
        _run(host, "install", "--yes")
        add = next(c["argv"] for c in _calls(host, "claude") if c["argv"][:2] == ["mcp", "add"])
        (host.state / "calls.jsonl").write_text("")
        return add[add.index("--executable-path") + 1]

    assert registered_browser() == str(host.browser)
    del host.env["BROWSER_AGENT_CHROMIUM"]
    assert registered_browser() == str(on_path)
    host.env["CLAUDE_TOOLS_SYSTEM_CHROMIUM"] = str(system)
    assert registered_browser() == str(system)


def test_the_check_starts_the_server_with_the_registered_browser(host):
    done = _run(host, "install", "--yes")

    checks = [c["argv"] for c in _calls(host, "python")]
    assert checks == [["scripts/browser_agent.py", "mcp-check", "--npx", str(host.bin / "npx"),
                       "--chromium", str(host.browser)]], done.stdout
    assert "starts and drives" in done.stdout


def test_a_failed_server_check_fails_the_install(host):
    host.env["STANDIN_MCP_CHECK_STATUS"] = "1"

    done = _run(host, "install", "--yes")

    assert done.returncode == 1
    assert "did not pass its check" in done.stdout


def test_no_npx_is_a_note_not_a_failure(host):
    (host.bin / "npx").unlink()

    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    assert not any(c["argv"][:2] == ["mcp", "add"] for c in _calls(host, "claude"))
    assert any("nodejs npm" in n for n in _notes(host))


# ---------------------------------------------------------------------------
# Claude in Chrome, and check
# ---------------------------------------------------------------------------


def _all_present(host, browser_dir="chromium"):
    (host.state / "auth").write_text("claude.ai")
    _register(host)
    manifest = host.home / ".config" / browser_dir / "NativeMessagingHosts" / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return manifest


@pytest.mark.parametrize("browser_dir", ["chromium", "google-chrome", "BraveSoftware/Brave-Browser"])
def test_the_chrome_host_is_found_under_each_browser(host, browser_dir):
    manifest = _all_present(host, browser_dir)

    done = _run(host, "check")

    assert done.returncode == 0, done.stdout + done.stderr
    assert str(manifest) in done.stdout


def test_a_missing_chrome_host_is_a_note_with_the_store_link(host):
    done = _run(host, "install", "--yes")

    assert done.returncode == 0, done.stdout + done.stderr
    note = next(n for n in _notes(host) if "Claude in Chrome is not set up" in n)
    assert "https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn" in note
    assert "claude --chrome" in note and "/chrome" in note


def test_check_passes_when_everything_is_in_place(host):
    _all_present(host)

    done = _run(host, "check")

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("✓") == 4 and "✗" not in done.stdout


@pytest.mark.parametrize("missing", ["cli", "sign-in", "mcp", "chrome"])
def test_check_fails_when_one_thing_is_missing(host, missing):
    manifest = _all_present(host)
    if missing == "cli":
        (host.bin / "claude").unlink()
    elif missing == "sign-in":
        (host.state / "auth").write_text("oauth_token")
    elif missing == "mcp":
        (host.state / "mcp").unlink()
    else:
        manifest.unlink()

    done = _run(host, "check")

    assert done.returncode == 1, done.stdout
    assert "✗" in done.stdout
    # check only looks.
    assert not any(c["argv"][:2] in (["mcp", "add"], ["auth", "login"]) for c in _calls(host, "claude"))


def test_an_unknown_command_is_refused(host):
    assert _run(host, "uninstall").returncode == 2
    assert _run(host, "install", "--force").returncode == 2


# ---------------------------------------------------------------------------
# the installers
# ---------------------------------------------------------------------------


def test_both_installers_set_up_the_claude_tools():
    for installer in ("install.sh", "docker/install.sh"):
        text = (ROOT / installer).read_text(encoding="utf-8")
        assert "scripts/claude_tools.sh install" in text, installer
        assert "CLAUDE_TOOLS_NOTES" in text, f"{installer} drops the notes"
        assert "--no-claude" in text, installer


@pytest.mark.parametrize("installer, flags", [
    ("install.sh", ("--no-browser-agent", "--no-claude")),
    ("docker/install.sh", ("--no-claude",)),
])
def test_the_installers_document_their_new_flags(installer, flags):
    shown = subprocess.run(["bash", str(ROOT / installer), "--help"], capture_output=True,
                           text=True, timeout=30, env={**os.environ, "LANG": "C.UTF-8"})

    assert shown.returncode == 0, shown.stderr
    for flag in flags:
        assert flag in shown.stdout, f"{installer} --help does not list {flag}"
