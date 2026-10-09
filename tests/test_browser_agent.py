"""The browser agent: scripts/browser_agent.py.

Most of what makes the agent safe to point at a console is decided before a
browser exists -- which passes run, which RPCs leave the page, which binary is
launched and with what sandbox, what a stub console's environment holds, what
a report may contain -- so that is what CI pins, with no browser and no
Playwright. Two live tests drive a real chromium; they run only with
BROWSER_TESTS=1, since CI installs neither.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "browser_agent.py"
PAGE = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

LIVE = pytest.mark.skipif(os.getenv("BROWSER_TESTS") != "1",
                          reason="drives a real chromium; set BROWSER_TESTS=1")


def _load() -> Any:
    """Import the script by path, registered first so its dataclasses resolve."""
    spec = importlib.util.spec_from_file_location("browser_agent", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["browser_agent"] = module
    spec.loader.exec_module(module)
    return module


ba = _load()
LOOPBACK = "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# loading without Playwright
# ---------------------------------------------------------------------------


def test_the_module_imports_no_playwright_at_load():
    """CI has no Playwright, and the skill's driver loads this file: every
    import of it sits inside a function."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in ast.walk(ast.Module(body=[n for n in tree.body if not isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[])):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_level.append(node)
    names = [alias.name for node in top_level if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in top_level if isinstance(node, ast.ImportFrom)]
    assert not [n for n in names if n.split(".")[0] == "playwright"], names


def test_loading_the_module_leaves_playwright_unimported():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ba', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['ba'] = m\n"
        "spec.loader.exec_module(m)\n"
        "print('playwright' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# the page and the server, as the agent knows them
# ---------------------------------------------------------------------------


def _page_ids() -> set[str]:
    # Markup, and the elements the page builds itself (`#gone`, `#run-live`).
    return set(re.findall(r'\bid="([^"$]+)"', PAGE)) | set(re.findall(r'\.id\s*=\s*"([^"]+)"', PAGE))


def test_every_selector_the_passes_use_is_on_the_page():
    ids = _page_ids()
    missing = []
    for name, selector in ba.SELECTORS.items():
        assert re.fullmatch(r"#[A-Za-z][\w-]*", selector), f"{name}: {selector} is not one id"
        if selector[1:] not in ids:
            missing.append(f"{name} -> {selector}")
    assert not missing, f"frontend/index.html has no element for: {missing}"


def test_the_tabs_are_the_pages_tabs_in_order():
    assert ba.TABS == tuple(re.findall(r'<button class="tab" data-p="([^"]+)"', PAGE))


def test_every_method_the_agent_knows_is_served_and_every_served_one_is_classified():
    import serve

    assert set(ba.RPC_KINDS) <= set(serve.RPC_METHODS), set(ba.RPC_KINDS) - set(serve.RPC_METHODS)
    unclassified = set(serve.RPC_METHODS) - set(ba.RPC_KINDS)
    assert not unclassified, (
        f"classify {sorted(unclassified)} in RPC_KINDS: until then the agent refuses them "
        "like a mutation, and any pass whose page calls one fails")
    assert set(ba.RPC_KINDS.values()) == {"read", "heavy", "mutate"}


def test_the_polling_methods_are_the_servers_quiet_ones():
    import serve

    assert ba.POLLING_RPCS == frozenset(serve.QUIET_METHODS)


def test_loopback_names_match_the_server():
    import serve

    assert ba.LOOPBACK_NAMES == serve._LOOPBACK_NAMES


@pytest.mark.parametrize("host", [
    "localhost", "127.0.0.1", "[::1]", "LOCALHOST", "127.0.0.2", "0.0.0.0",
    "localhost.evil.example", "127.0.0.1.nip.io", "example.com", "10.0.0.5",
])
def test_loopback_is_exactly_what_the_server_accepts(host, monkeypatch):
    """`127.0.0.2` reaches the kernel's loopback, but the console refuses a Host
    it does not list -- so the agent does not call it loopback either."""
    import serve

    monkeypatch.setattr(serve, "CONSOLE_HOST", "127.0.0.1")
    server_takes_it = serve._foreign_request({"Host": f"{host}:8080"}, writes=False) is None
    assert ba.target_is_loopback(f"http://{host}:8080/") is server_takes_it


def test_the_seat_roles_are_the_configs_agents():
    from langgraph_agent.config import AGENTS

    assert ba.ROLES == AGENTS


# ---------------------------------------------------------------------------
# what leaves the page
# ---------------------------------------------------------------------------


def test_every_mutation_is_either_behind_a_key_or_never_sent():
    mutations = {m for m, kind in ba.RPC_KINDS.items() if kind == "mutate"}
    assert mutations == set(ba.ALLOW_FOR) | set(ba.NEVER_ALLOWED)
    assert not set(ba.ALLOW_FOR) & set(ba.NEVER_ALLOWED)
    assert ba.ALLOW_FOR == {"run_goal": "run", "stop_run": "run", "upload_document": "upload",
                            "reset_circuit": "circuit", "shutdown": "exit"}


def test_clear_corpus_is_refused_whatever_is_allowed():
    for allow in ([], list(ba.ALLOW_KEYS)):
        for target in (LOOPBACK, "http://localhost:8081", "http://example.com"):
            assert "never sent" in ba.rpc_refusal("clear_corpus", allow, target)


@pytest.mark.parametrize("method", ["set_seat", "set_thinking", "embed_project", "dismiss_pull_request"])
def test_the_other_never_allowed_methods_are_refused_with_every_key(method):
    assert ba.rpc_refusal(method, list(ba.ALLOW_KEYS), LOOPBACK)


def test_a_mutation_needs_its_key_and_loopback():
    assert "--allow run" in ba.rpc_refusal("run_goal", [], LOOPBACK)
    assert "--allow run" in ba.rpc_refusal("run_goal", ["upload", "exit"], LOOPBACK)
    assert ba.rpc_refusal("run_goal", ["run"], LOOPBACK) == ""
    assert ba.rpc_refusal("stop_run", ["run"], "http://[::1]:8080/rpc") == ""
    assert "loopback" in ba.rpc_refusal("run_goal", ["run"], "http://192.168.1.4:8080/rpc")
    assert ba.rpc_refusal("upload_document", ["upload"], LOOPBACK) == ""
    assert ba.rpc_refusal("reset_circuit", ["circuit"], LOOPBACK) == ""
    assert ba.rpc_refusal("shutdown", ["exit"], LOOPBACK) == ""
    assert "--allow exit" in ba.rpc_refusal("shutdown", ["run"], LOOPBACK)


def test_reads_always_go_and_unknown_methods_never_do():
    for method, kind in ba.RPC_KINDS.items():
        if kind != "mutate":
            assert ba.rpc_refusal(method, [], "http://example.com") == "", method
    assert "not a method" in ba.rpc_refusal("format_disk", list(ba.ALLOW_KEYS), LOOPBACK)
    assert "not a method" in ba.rpc_refusal("", [], LOOPBACK)


def test_the_agents_own_rpc_calls_are_under_the_same_guard():
    with pytest.raises(ba.RpcRefused, match="never sent"):
        ba.rpc_call(LOOPBACK, "clear_corpus", allow=list(ba.ALLOW_KEYS))
    with pytest.raises(ba.RpcRefused, match="--allow run"):
        ba.rpc_call(LOOPBACK, "run_goal", {"goal": "x"})


# Request targets a page can send, and whether the guard covers each. The
# server's own verdict on them is the test after this one's.
GUARDED = {
    "/rpc": True, "/rpc?x=1": True, "/rpc?": True, "/rpc;x": True, "/rpc;x?y=1": True,
    "///rpc": True, "//x/rpc": True, "//[x/rpc": True,
    "/rpcx": False, "/rpc/": False, "/a/rpc": False, "/api/status": False, "/": False,
}


@pytest.mark.parametrize("target, guarded", GUARDED.items())
def test_the_guard_reads_a_url_the_way_the_server_does(target, guarded):
    """A `**/rpc` glob matched the whole URL, so `/rpc?x` and `/rpc;x` were
    never guarded, while the server drops the query and `;params` and
    dispatched them: clear_corpus went through."""
    assert ba.is_rpc_url(f"http://127.0.0.1:8080{target}") is guarded


def test_every_target_serve_dispatches_as_an_rpc_is_guarded():
    """Grounded in the server itself rather than a reading of it: each target
    sent raw through http.server and serve.py's own `do_POST`."""
    import http.client

    import serve

    dispatched: list[str] = []

    class Recording(serve.Handler):
        def handle_rpc(self, data):
            dispatched.append(self.path)
            self.send_json({"result": {}, "elapsed_ms": 0})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        reached = []
        for target in GUARDED:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            before = len(dispatched)
            conn.request("POST", target, body=b'{"method": "status"}',
                         headers={"Content-Type": "application/json"})
            conn.getresponse().read()
            conn.close()
            if len(dispatched) > before:
                reached.append(target)
    finally:
        server.shutdown()
        server.server_close()
    assert {"/rpc", "/rpc?x=1", "/rpc;x", "///rpc"} <= set(reached), reached
    missed = [t for t in reached if not ba.is_rpc_url(f"http://127.0.0.1:{port}{t}")]
    assert not missed, f"the server ran these as RPCs and the guard would not see them: {missed}"


class _Route:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def continue_(self):
        self.calls.append(("continue", None))

    def fulfill(self, **reply):
        self.calls.append(("fulfill", json.loads(reply["body"])))


class _Request:
    method = "POST"
    url = "http://127.0.0.1:8080/rpc"

    def __init__(self, body: bytes | None) -> None:
        self.post_data_buffer = body

    @property
    def post_data(self):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


def test_the_guard_reads_the_body_the_way_the_server_does():
    """`post_data` decodes as UTF-8 and raised on a UTF-16 body the server's
    `json.loads(bytes)` reads: the handler died and the fetch hung. The bytes
    are read instead, and whatever cannot be read is refused."""
    calls: list = []
    owner = SimpleNamespace(allow=frozenset(), recorder=ba.Recorder())
    wipe = json.dumps({"method": "clear_corpus"})
    for body in (wipe.encode("utf-16"), wipe.encode("utf-32"), wipe.encode("utf-8-sig"),
                 b"\xff\xfe\xff", b"[" * 100_000, b"", None):
        ba.Session._guard(owner, _Route(calls), _Request(body))
    assert [kind for kind, _ in calls] == ["fulfill"] * 7
    assert all(reply["error"]["message"].startswith(ba.BLOCKED_PREFIX) for _, reply in calls)
    assert [b["method"] for b in owner.recorder.blocked] == ["clear_corpus"] * 3 + [""] * 4
    calls.clear()
    ba.Session._guard(owner, _Route(calls), _Request(json.dumps({"method": "status"}).encode("utf-16")))
    assert calls == [("continue", None)]


def test_the_recorder_logs_an_rpc_on_any_url_the_server_dispatches():
    rec = ba.Recorder()
    for target in ("/rpc;x", "/rpc?x=1", "///rpc"):
        rec.on_response("POST", f"http://127.0.0.1:8080{target}", 200,
                        json.dumps({"method": "status"}).encode("utf-16"), {"result": {}, "elapsed_ms": 3})
    assert [e["method"] for e in rec.rpc] == ["status"] * 3


def test_unknown_allow_keys_are_refused_by_name():
    assert ba.parse_allow("run, upload") == {"run", "upload"}
    assert ba.parse_allow(None) == frozenset()
    with pytest.raises(ba.AgentError, match="clear"):
        ba.parse_allow("run,clear")


# ---------------------------------------------------------------------------
# which passes run
# ---------------------------------------------------------------------------


def _names(passes):
    return [p.name for p in passes]


def test_the_default_check_changes_nothing():
    chosen, refused = ba.select_passes(None, set(), LOOPBACK)
    assert _names(chosen) == list(ba.DEFAULT_PASSES)
    assert not refused
    assert all(p.kind != "mutate" for p in chosen)
    chosen, _ = ba.select_passes(None, set(), LOOPBACK, quick=True)
    assert "search" not in _names(chosen) and "analyses" not in _names(chosen)
    assert "export" not in _names(chosen) and "flood" not in _names(chosen)


def test_mutating_passes_are_opt_in():
    chosen, refused = ba.select_passes(["tour", "run", "upload", "circuit", "exit"], set(), LOOPBACK)
    assert _names(chosen) == ["tour"]
    reasons = {r["name"]: r["reason"] for r in refused}
    assert set(reasons) == {"run", "upload", "circuit", "exit"}
    for name, key in (("run", "run"), ("upload", "upload"), ("circuit", "circuit"), ("exit", "exit")):
        assert f"--allow {key}" in reasons[name]
    assert all(r["status"] == "refused" for r in refused)


def test_mutating_passes_refuse_a_target_that_is_not_loopback():
    chosen, refused = ba.select_passes(["run", "tour"], {"run"}, "http://192.168.1.4:8080")
    assert _names(chosen) == ["tour"]
    assert "loopback" in refused[0]["reason"]
    chosen, refused = ba.select_passes(["run"], {"run"}, "http://127.0.0.2:8080")
    assert not chosen and refused


def test_an_allow_key_opens_its_passes_and_exit_runs_last():
    chosen, refused = ba.select_passes(None, {"run", "exit", "upload"}, LOOPBACK)
    names = _names(chosen)
    assert {"run", "stop", "reattach", "upload", "exit"} <= set(names)
    assert names[-1] == "exit" and not refused
    chosen, _ = ba.select_passes(None, {"exit"}, LOOPBACK, spawn=True)
    assert _names(chosen)[-1] == "exit"
    chosen, _ = ba.select_passes(["exit", "tour"], {"exit"}, LOOPBACK)
    assert _names(chosen) == ["tour", "exit"]


def test_an_unknown_pass_is_refused_by_name():
    chosen, refused = ba.select_passes(["tour", "teleport"], set(), LOOPBACK)
    assert _names(chosen) == ["tour"]
    assert "no pass called teleport" in refused[0]["reason"]


def test_every_pass_is_classified_and_keyed_consistently():
    for p in ba.PASSES.values():
        assert p.kind in ("read", "heavy", "mutate")
        assert (p.kind == "mutate") == bool(p.allow_key), p.name
        assert not p.allow_key or p.allow_key in ba.ALLOW_KEYS
    assert ba.PASS_ORDER[-1] == "exit"


def test_a_check_with_everything_refused_exits_2_before_any_browser(tmp_path, capsys):
    code = ba.main(["check", "--base", "http://192.168.1.4:8080", "--passes", "run,upload",
                    "--allow", "run,upload", "--out", str(tmp_path)])
    capsys.readouterr()
    assert code == 2
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["schema"] == "ambiguity-browser/1"
    assert {p["name"]: p["status"] for p in results["passes"]} == {"run": "refused", "upload": "refused"}
    assert results["browser"] is None
    assert "# browser check" in (tmp_path / "report.md").read_text()


# ---------------------------------------------------------------------------
# the browser binary and its sandbox
# ---------------------------------------------------------------------------


def test_the_sandbox_is_off_only_for_root():
    assert ba.launch_options("/x/chrome", euid=0, env={})["chromium_sandbox"] is False
    for euid in (1, 1000, 65534):
        assert ba.launch_options("/x/chrome", euid=euid, env={})["chromium_sandbox"] is True


def test_the_agent_never_asks_for_no_sandbox():
    for headed in (False, True):
        for euid in (0, 1000):
            for env in ({}, {"WAYLAND_DISPLAY": "wayland-1"}, {"DISPLAY": ":0"}):
                options = ba.launch_options("/x/chrome", headed=headed, euid=euid, env=env)
                assert "--no-sandbox" not in options["args"]
                assert options["headless"] is (not headed)
    # And nowhere in the source is it handed to a browser as an argument.
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            assert not any(isinstance(e, ast.Constant) and e.value == "--no-sandbox" for e in node.elts)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("append", "extend", "insert"):
            assert not any(isinstance(a, ast.Constant) and a.value == "--no-sandbox" for a in node.args)


def test_wayland_gets_the_ozone_hint_only_when_headed():
    assert ba.launch_options("/x", headed=True, euid=1, env={"WAYLAND_DISPLAY": "w"})["args"] == [
        "--ozone-platform-hint=auto"]
    assert ba.launch_options("/x", headed=False, euid=1, env={"WAYLAND_DISPLAY": "w"})["args"] == []
    assert ba.launch_options("/x", headed=True, euid=1, env={"DISPLAY": ":0"})["args"] == []


def _resolver(tmp_path, *, files=(), env=None, on_path=None, playwright=None):
    present = set(files)

    def exists(path):
        return path in present or (path.startswith(str(tmp_path)) and os.path.exists(path))

    return lambda explicit=None: ba.resolve_chromium(
        explicit, env=env or {}, exists=exists, which=(on_path or {}).get,
        playwright_path=lambda: playwright, home=tmp_path / "home")


def _build(root: Path, rev: str, sub: str = "chrome-linux") -> str:
    path = root / f"chromium-{rev}" / sub / "chrome"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return str(path)


def test_chromium_resolution_order(tmp_path):
    browsers = tmp_path / "pw-browsers"
    old = _build(browsers, "1194")
    new = _build(browsers, "1243", "chrome-linux64")
    cached = _build(tmp_path / "home" / ".cache" / "ms-playwright", "1100")
    system, google = "/usr/lib/chromium/chromium", "/opt/google/chrome/chrome"
    everything: dict[str, Any] = {
        "files": {"/mine/chrome", "/pw/own", system, google},
        "env": {"PLAYWRIGHT_BROWSERS_PATH": str(browsers), "BROWSER_AGENT_CHROMIUM": "/mine/chrome"},
        "on_path": {"chromium": "/bin/chromium"},
        "playwright": "/pw/own",
    }

    assert _resolver(tmp_path, **everything)("/mine/chrome") == ("/mine/chrome", "--chromium")
    assert _resolver(tmp_path, **everything)() == ("/mine/chrome", "BROWSER_AGENT_CHROMIUM")
    assert _resolver(tmp_path, **everything)("/nope")[0] is None

    env = {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}
    assert _resolver(tmp_path, files={"/pw/own", system}, env=env, playwright="/pw/own")() == (
        "/pw/own", "playwright's own build")
    # Playwright expecting a build that is not there falls through quietly.
    assert _resolver(tmp_path, files={system}, env=env, playwright="/pw/missing")() == (
        new, "PLAYWRIGHT_BROWSERS_PATH")
    assert old != new
    assert _resolver(tmp_path, files={system})() == (cached, "playwright's cache")

    (tmp_path / "home").rename(tmp_path / "elsewhere")
    assert _resolver(tmp_path, files={system, google}, on_path={"chromium": "/bin/chromium"})() == (
        system, "the system chromium")
    assert _resolver(tmp_path, files={google}, on_path={"google-chrome-stable": "/bin/gcs"})() == (
        "/bin/gcs", "google-chrome-stable on PATH")
    assert _resolver(tmp_path, files={google})() == (google, "google chrome")
    assert _resolver(tmp_path)() == (None, "no chromium found")


def test_the_real_argv_is_read_from_proc(tmp_path):
    """Driver under this process, browser under the driver, helpers under it."""
    proc = tmp_path / "proc"
    exe = tmp_path / "chrome"
    exe.write_text("")

    def process(pid, parent, argv, comm="x"):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({comm}) S {parent} 1 1 0")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")

    process(100, 1, ["python", "agent.py"])
    process(200, 100, ["node", "driver"], comm="node (x)")
    process(301, 300, [str(exe), "--type=zygote", "--no-sandbox"])
    process(300, 200, [str(exe), "--headless", "--no-sandbox", "--user-data-dir=/tmp/p"])
    process(400, 1, [str(exe), "--headless"])  # someone else's browser
    assert ba.browser_argv(str(exe), root_pid=100, proc=proc) == [
        str(exe), "--headless", "--no-sandbox", "--user-data-dir=/tmp/p"]
    assert ba.browser_argv(str(exe), root_pid=999, proc=proc) is None
    assert ba.browser_argv(str(exe), proc=tmp_path / "no-proc") is None


# ---------------------------------------------------------------------------
# the tools
# ---------------------------------------------------------------------------

# Claude in Chrome's public tool vocabulary, as this project's tool names.
VOCABULARY = {
    "navigate", "tabs", "snapshot", "find", "text", "click", "hover", "drag", "scroll", "type",
    "key", "fill", "select", "checkbox", "wait", "screenshot", "eval", "console", "network",
    "resize", "upload", "dialog", "download", "record", "trace", "wait-for-user",
}


def test_the_tool_table_covers_the_browser_vocabulary():
    assert set(ba.TOOLS) == VOCABULARY
    for tool in ba.TOOLS.values():
        assert tool.equivalent and tool.summary, tool.name
        assert callable(getattr(ba.ToolRunner, "t_" + tool.name.replace("-", "_"), None)), tool.name
    for name in ("read_page", "find", "get_page_text", "computer", "form_input", "javascript_tool",
                 "read_console_messages", "read_network_requests", "resize_window", "gif_creator"):
        assert any(name in tool.equivalent for tool in ba.TOOLS.values()), name


def test_every_tool_has_a_command_line(capsys):
    parser = ba.build_parser()
    args = parser.parse_args(["snapshot", "--interactive", "--max-chars", "500"])
    assert (args.command, args.interactive, args.max_chars) == ("snapshot", True, 500)
    args = parser.parse_args(["click", "--xy", "10", "20", "--double"])
    assert args.xy == [10, 20] and args.double
    args = parser.parse_args(["resize", "800", "600"])
    assert (args.width, args.height) == (800, 600)
    args = parser.parse_args(["wait-for-user", "--until-url", "dashboard"])
    assert args.until_url == "dashboard"
    assert ba.print_tools() == 0
    printed = capsys.readouterr().out
    for name in VOCABULARY:
        assert name in printed


def test_steps_parse_with_defaults_filled_in():
    tool, args = ba.parse_step({"tool": "click", "ref": "e12", "double": True, "continue": True})
    assert tool.name == "click"
    assert args["ref"] == "e12" and args["double"] is True and args["button"] == "left"
    _, args = ba.parse_step({"tool": "screenshot", "zoom": 2})
    assert args["zoom"] == 2.0 and args["full"] is None
    _, args = ba.parse_step({"tool": "upload", "files": "a.md", "selector": "#f"})
    assert args["files"] == ["a.md"]


@pytest.mark.parametrize("step, refusal", [
    ({"tool": "teleport"}, "there is no tool 'teleport'"),
    ({"click": "e1"}, "there is no tool None"),
    ("click e1", "a step is an object"),
    ({"tool": "click", "force": True}, "click takes no argument 'force'"),
    ({"tool": "click", "double": "yes"}, "double must be true or false"),
    ({"tool": "find"}, "find needs query"),
    ({"tool": "dialog", "action": "maybe"}, "action must be one of accept, dismiss"),
    ({"tool": "click", "xy": [1]}, "xy must be a list of 2 whole numbers"),
    ({"tool": "snapshot", "max_chars": "lots"}, "max_chars must be a whole number"),
    ({"tool": "click", "ref": "e1", "continue": "sure"}, "continue must be true or false"),
])
def test_batch_steps_are_refused_by_name(step, refusal):
    with pytest.raises(ba.ToolError) as caught:
        ba.parse_step(step)
    assert refusal in str(caught.value)


def test_a_batch_with_a_bad_step_is_refused_before_any_browser(tmp_path, capsys):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps([{"tool": "snapshot"}, {"tool": "click", "colour": "red"}]))
    assert ba.main(["batch", str(steps)]) == 2
    assert "click takes no argument 'colour'" in capsys.readouterr().err


def test_the_batch_only_tools_refuse_to_run_alone(capsys):
    assert ba.main(["record", "start"]) == 2
    assert "inside batch" in capsys.readouterr().err


# A real AI snapshot of the console, as the snapshot tool printed it.
CONSOLE_SNAPSHOT = """\
- generic [active] [ref=e1]:
  - generic [ref=e5]:
    - button "Engineer" [ref=e6] [cursor=pointer]
    - button "Graph" [ref=e7] [cursor=pointer]
    - button "Retrieval" [ref=e8] [cursor=pointer]
    - button "Corpus" [ref=e9] [cursor=pointer]
    - button "State" [ref=e10] [cursor=pointer]
    - generic [ref=e11]: "no corpus — nothing to index yet: upload a document"
    - button "×" [ref=e13] [cursor=pointer]
  - complementary [ref=e15]:
    - generic [ref=e20]:
      - generic [ref=e21]: architect
      - combobox "Reassign this seat" [ref=e23] [cursor=pointer]:
        - 'option "claude-opus-5 (no key: canned stub output)" [selected]'
      - 'generic "Think before answering: slower, and usually better." [ref=e26] [cursor=pointer]':
        - checkbox "thinking" [ref=e27]
        - text: thinking
      - generic "ANTHROPIC_API_KEY not set" [ref=e29]: NO KEY
  - main [ref=e73]:
    - textbox "node id, or blank to sweep every document" [ref=e76]
    - spinbutton "max depth" [ref=e77]: "2"
    - button "Trace" [ref=e78] [cursor=pointer]
    - button "Sweep all" [ref=e79] [cursor=pointer]
    - 'generic "Split the traced neighbourhood in two along the Fiedler vector." [ref=e81]':
      - checkbox "split" [ref=e82]
    - generic [ref=e83]: Loading graph…
"""


def test_find_ranks_refs_in_a_canned_ai_snapshot():
    def refs(query, limit=20):
        return [h["ref"] for h in ba.snapshot_find(CONSOLE_SNAPSHOT, query, limit)]

    assert refs("trace")[0] == "e78"
    assert refs("sweep all")[0] == "e79"
    assert refs("node id")[0] == "e76"
    assert refs("thinking")[0] == "e27"
    assert refs("reassign seat") == ["e23"]
    assert refs("graph button")[0] == "e7"
    assert refs("no key")[0] == "e29"  # the quoted line, and its inline text
    assert refs("e26 think")[0] == "e26"  # a YAML-quoted line still yields its ref
    assert refs("trace", limit=1) == ["e78"]
    assert refs("") == [] and refs("zebra") == []


def test_interactive_lines_keep_only_what_a_person_can_act_on():
    lines = ba.interactive_lines(CONSOLE_SNAPSHOT).splitlines()
    roles = {re.match(r"- '?(\w+)", line).group(1) for line in lines}
    assert roles == {"button", "combobox", "checkbox", "textbox", "spinbutton"}
    assert all("[ref=e" in line for line in lines)
    assert not any("Loading graph" in line for line in lines)


# ---------------------------------------------------------------------------
# the recorder and the report
# ---------------------------------------------------------------------------


def _rpc(rec, method, ms, error=None, params=None, status=200):
    body = {"elapsed_ms": ms, **({"error": {"message": error}} if error else {"result": {}})}
    rec.on_response("POST", "http://127.0.0.1:8080/rpc", status,
                    json.dumps({"method": method, "params": params or {}}), body)


def test_rpc_recorder_summary_and_envelopes():
    rec = ba.Recorder()
    rec.context = "tour"
    for ms in (10, 20, 30, 40, 1000):
        _rpc(rec, "graph_overview", ms)
    _rpc(rec, "search_documents", 50, error="the embedder is down", params={"query": "q"})
    for _ in range(3):
        _rpc(rec, "status", 5, error="database down")
    # A call the agent itself answered is in `blocked`, never in the RPC log.
    rec.on_blocked("clear_corpus", {}, "clear_corpus is never sent by the browser agent", "u")
    rec.on_response("POST", "http://127.0.0.1:8080/rpc", 200, json.dumps({"method": "clear_corpus"}),
                    {"error": {"message": f"{ba.BLOCKED_PREFIX} never"}, "elapsed_ms": 0})
    rec.on_response("GET", "http://127.0.0.1:8080/favicon.ico", 404, None, None)
    mark = rec.mark()
    rec.on_console("error", "boom", "http://127.0.0.1:8080/", 3)

    summary = rec.rpc_summary()
    assert summary["graph_overview"] == {"calls": 5, "errors": 0, "p50_ms": 30.0, "p95_ms": 1000.0,
                                         "max_ms": 1000.0}
    assert summary["status"]["errors"] == 3
    assert "clear_corpus" not in summary
    envelopes = rec.error_envelopes()
    assert [e["method"] for e in envelopes] == ["search_documents"]
    assert envelopes[0]["in"] == "tour" and "query" in envelopes[0]["params"]
    assert len(rec.error_envelopes(polling=True)) == 4
    assert [e["status"] for e in rec.http_errors] == [404]
    assert len(rec.blocked) == 1
    assert rec.since(mark)["console"][0]["text"] == "boom" and not rec.since(mark)["rpc"]


def test_a_pass_fails_on_what_else_went_wrong_on_the_page():
    rec = ba.Recorder()
    mark = rec.mark()
    rec.on_console("error", "Failed to load resource", "http://127.0.0.1:8080/favicon.ico", 0)
    _rpc(rec, "healing", 3, error="polled, counted, not noted")
    _rpc(rec, "export_corpus", 3, error="There is no corpus to export. Index one first.")
    assert ba._collateral(ba.PASSES["tour"], rec.since(mark), rec) == []
    rec.on_pageerror("TypeError: x is undefined")
    _rpc(rec, "list_documents", 3, error="database down")
    rec.on_response("GET", "http://127.0.0.1:8080/", 502, None, None)
    problems = ba._collateral(ba.PASSES["tour"], rec.since(mark), rec)
    assert len(problems) == 3
    assert any("page error" in p for p in problems) and any("list_documents" in p for p in problems)
    _rpc(rec, "run_goal", 3, error="Recursion limit")
    assert not any("run_goal" in p for p in ba._collateral(ba.PASSES["run"], rec.since(mark), rec))


def test_the_report_never_carries_a_key_an_address_or_this_home(tmp_path):
    key = "sk-ant-api03-" + "Q" * 40
    email = "someone.private@example.org"
    home = str(Path.home()) + "/private/notes.md"
    rec = ba.Recorder()
    rec.context = "tour"
    rec.on_console("error", f"leaked {key} for {email} at {home}", "http://127.0.0.1:8080/", 1)
    _rpc(rec, "upload_document", 9, error=f"cannot read {home}",
         params={"name": "x.md", "content": f"ANTHROPIC_API_KEY={key}"})
    rec.on_pageerror(f"Error: token={key}")
    results = {
        "schema": ba.REPORT_SCHEMA, "base": LOOPBACK, "options": {"out": home},
        "browser": {"path": home, "how": "--chromium", "version": "1", "headless": True,
                    "sandbox": True, "argv": [home, f"--user-data-dir={home}"]},
        "passes": [{"name": "tour", "kind": "read", "status": "fail", "duration_s": 1.0,
                    "reason": f"{email} said {key}", "observations": {"banner": f"{key} {home}"}}],
        "problems": [f"problem for {email}"],
        "rpc": {"by_method": rec.rpc_summary(), "error_envelopes": rec.error_envelopes(),
                "blocked": rec.blocked},
        "console": rec.console, "page_errors": rec.page_errors, "failed_requests": [],
        "http_errors": [], "dialogs": [], "overflow": {}, "exit_code": 1,
    }
    written = ba.write_report(tmp_path, results, rec)
    assert {"report.md", "results.json", "rpc.jsonl", "console.jsonl"} <= set(written)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert key not in text and email not in text and home not in text, path.name
    loaded = json.loads((tmp_path / "results.json").read_text())
    for name in ("schema", "base", "browser", "passes", "rpc", "console", "page_errors",
                 "failed_requests", "overflow", "artifacts"):
        assert name in loaded, name
    assert set(loaded["rpc"]) >= {"by_method", "error_envelopes", "blocked"}
    assert loaded["redaction"]["counts"]
    report = (tmp_path / "report.md").read_text()
    assert report.startswith("# browser check")
    assert all(line == line.lower() for line in report.splitlines() if line.startswith("#"))


# ---------------------------------------------------------------------------
# a console of the agent's own
# ---------------------------------------------------------------------------


def test_the_stub_seat_environment_empties_every_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-planted")
    monkeypatch.setenv("BUILDER_API_KEY", "sk-ant-api03-planted-builder")
    monkeypatch.setenv("OLLAMA_API_KEY", "planted-ollama")
    monkeypatch.setenv("ARCHITECT_PROVIDER", "ollama")
    env = ba.spawn_environment(os.environ, port=4321, runs_dir=tmp_path / "runs", stub_seats=True,
                               no_rebuild=True, dotenv_keys=["ONLY_IN_DOTENV_API_KEY", "DATABASE_URL"])
    for name, value in env.items():
        if name.endswith("_API_KEY"):
            assert value == "", name
    assert not any("planted" in value for value in env.values())
    # Empty, not removed: `.env` would fill a missing name back in.
    for name in ("ANTHROPIC_API_KEY", "ONLY_IN_DOTENV_API_KEY", *(f"{r.upper()}_API_KEY" for r in ba.ROLES)):
        assert name in env and env[name] == "", name
    assert "DATABASE_URL" not in env or env["DATABASE_URL"] == os.environ.get("DATABASE_URL")
    for role in ba.ROLES:
        assert env[f"{role.upper()}_PROVIDER"] == "anthropic"
    assert env["PORT"] == "4321" and env["CONSOLE_HOST"] == "127.0.0.1"
    assert env["RUNS_DIR"] == str(tmp_path / "runs")
    assert env["FOLLOW_PULL_REQUESTS"] == "0" and env["REBUILD_CORPUS"] == "0"


def test_without_stub_seats_the_keys_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "kept")
    env = ba.spawn_environment(os.environ, port=1, runs_dir=tmp_path, stub_seats=False, no_rebuild=False)
    assert env["ANTHROPIC_API_KEY"] == "kept"
    assert "REBUILD_CORPUS" not in env or env["REBUILD_CORPUS"] == os.environ.get("REBUILD_CORPUS")
    assert env["FOLLOW_PULL_REQUESTS"] == "0"


def test_the_stub_environment_really_stubs_every_seat(tmp_path, monkeypatch):
    """The config's own verdict, in a child that loads `.env` the way serve.py
    does: a planted key in the environment and per-seat keys stay unread.
    A file, not `-c`: under `-c` config's `load_dotenv()` searches the working
    directory, here a scratch one, and the checkout's `.env` was never read."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-planted")
    monkeypatch.setenv("PLANNER_API_KEY", "sk-ant-api03-planted")
    env = ba.spawn_environment(os.environ, port=1, runs_dir=tmp_path, stub_seats=True, no_rebuild=True,
                               dotenv_keys=ba.dotenv_names(ROOT / ".env"))
    env["PYTHONPATH"] = str(ROOT / "src")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from langgraph_agent.config import AGENTS, _resolve_seat, get_agent_status\n"
        "print(all(get_agent_status(a)['stubbed'] for a in AGENTS),"
        " any(_resolve_seat(a)['api_key'] for a in AGENTS))\n"
    )
    done = subprocess.run([sys.executable, str(probe)], env=env, cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["True", "False"]


def test_dotenv_names_are_names_only(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("# a comment\nANTHROPIC_API_KEY=sk-ant-secret\nexport OLLAMA_API_KEY = x\n"
                      "not a line\nDATABASE_URL=postgresql://u:p@h/db\n")
    assert ba.dotenv_names(dotenv) == ["ANTHROPIC_API_KEY", "OLLAMA_API_KEY", "DATABASE_URL"]
    assert ba.dotenv_names(tmp_path / "missing") == []


def test_a_serve_without_the_isolation_switches_is_not_spawned():
    assert "RUNS_DIR or FOLLOW_PULL_REQUESTS" in ba.serve_isolation_problem("RUNS_DIR = Path('runs')")
    assert "FOLLOW_PULL_REQUESTS" in ba.serve_isolation_problem('os.getenv("RUNS_DIR")')
    assert ba.serve_isolation_problem('os.getenv("RUNS_DIR") or x; os.getenv("FOLLOW_PULL_REQUESTS")') == ""


def _drop_schema(tmp_path: Path, env: dict[str, str], dotenv: Path) -> dict[str, Any]:
    """The spawned console's schema drop, run the way `drop_schema` runs it."""
    done = subprocess.run(
        [sys.executable, "-c", ba._DROP_SCHEMA, str(tmp_path / "knowledge"), str(dotenv)],
        cwd=tmp_path, env={**env, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True,
        timeout=120, stdin=subprocess.DEVNULL,
    )
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


def test_the_schema_drop_finds_the_database_the_console_found(tmp_path):
    """The console reads `.env`; a `-c` child's own `load_dotenv()` searched
    its scratch directory, so a DATABASE_URL set only in `.env` sent the drop
    to the default server and left the schema behind. Here `.env` names a
    port nothing listens on, which the drop must try, and say it could not
    reach -- not that the schema is gone."""
    dotenv = tmp_path / "dotenv"
    dotenv.write_text("DATABASE_URL=postgresql://postgres@127.0.0.1:1/nowhere\n")
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    out = _drop_schema(tmp_path, env, dotenv)
    assert out["schema"].startswith("kb_") and out["dropped"] is False, out
    assert out.get("unreachable") is True and re.search(r"port 1\b", out["error"]), out
    assert "absent" not in out


def test_the_schema_drop_never_lets_dotenv_override_the_environment(tmp_path, postgres):
    """Loaded the way the console loads it, never over a variable already set:
    the spawned console's emptied keys stay empty, and its DATABASE_URL wins."""
    dotenv = tmp_path / "dotenv"
    dotenv.write_text("DATABASE_URL=postgresql://postgres@127.0.0.1:1/nowhere\n")
    out = _drop_schema(tmp_path, {**os.environ, "DATABASE_URL": postgres}, dotenv)
    assert out.get("absent") is True and "error" not in out and "unreachable" not in out, out


def test_expected_errors_are_narrow():
    assert ba.expected_rpc_error("export_corpus", "There is no corpus to export. Index one.")
    assert not ba.expected_rpc_error("export_corpus", "database down")
    assert not ba.expected_rpc_error("search_documents", "There is no corpus to export")


# ---------------------------------------------------------------------------
# live: a real chromium (BROWSER_TESTS=1)
# ---------------------------------------------------------------------------


@LIVE
def test_a_tour_of_a_spawned_stub_console(tmp_path, capsys):
    pytest.importorskip("playwright")
    pending = ROOT / "runs" / "pull_requests.json"
    before = pending.read_bytes() if pending.exists() else None
    projects = sorted(p.name for p in (ROOT / "projects").iterdir()) if (ROOT / "projects").is_dir() else []

    code = ba.main(["check", "--spawn", "--stub-seats", "--no-rebuild", "--passes", "tour,exit",
                    "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    results = json.loads((tmp_path / "results.json").read_text())
    assert code == 0, printed + json.dumps(results["passes"], indent=1)
    assert {p["name"]: p["status"] for p in results["passes"]} == {"tour": "pass", "exit": "pass"}
    assert results["browser"]["sandbox"] is (os.geteuid() != 0)
    assert results["browser"]["argv"], "the browser's real argv was not read"
    assert ("--no-sandbox" in results["browser"]["argv"]) is (os.geteuid() == 0)
    spawned = results["spawned"]
    assert all(seat["stubbed"] for seat in spawned["seats"]) and len(spawned["seats"]) == 4
    assert spawned["workdir_removed"] is True
    tour = results["passes"][0]["observations"]
    assert len(tour["seats"]) == 4 and all("NO KEY" in s["chips"] for s in tour["seats"])
    assert (tmp_path / "console.log").exists()
    assert len(list((tmp_path / "shots").glob("*.png"))) >= 5
    assert (pending.read_bytes() if pending.exists() else None) == before
    now = sorted(p.name for p in (ROOT / "projects").iterdir()) if (ROOT / "projects").is_dir() else []
    assert now == projects
    # Last, so a machine without a database still checks everything above. A
    # server that never answered could hold no schema, and could not say so
    # either: skipped, as the suite's other database tests are, unless
    # REQUIRE_POSTGRES=1 makes that a failure.
    schema = spawned["schema"]
    if schema.get("unreachable"):
        message = f"PostgreSQL did not answer, so the schema drop went unchecked: {schema.get('error')}"
        if os.getenv("REQUIRE_POSTGRES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert schema.get("dropped") or schema.get("absent"), schema


FIXTURE = """<!doctype html><html><head><title>fixture page</title></head><body>
<h1 id="out">idle</h1>
<button id="press" onclick="console.log('pressed the button');
  document.getElementById('out').textContent='pressed'">Press me</button>
<input id="name" aria-label="your name">
<button id="ask" onclick="document.getElementById('out').textContent='answer '+confirm('sure?')">Ask me</button>
<input type="file" id="pick" aria-label="pick a file"
  onchange="document.getElementById('out').textContent='picked '+this.files[0].name">
<a id="dl" href="data:text/plain,hello%20download" download="hello.txt">Download hello</a>
<button id="read" onclick="call('status')">Read status</button>
<button id="wipe" onclick="call('clear_corpus')">Wipe</button>
<label><input type="checkbox" id="tick"> tick me</label>
<script>
function call(method){
  fetch('/rpc', {method: 'POST', headers: {'content-type': 'application/json'},
                 body: JSON.stringify({method, params: {}})})
    .then(r => r.json())
    .then(j => document.getElementById('out').textContent =
      method + ': ' + (j.error ? j.error.message : j.result.corpus));
}
</script>
</body></html>
"""


@pytest.fixture
def fixture_site(tmp_path):
    """A page and a fake /rpc on loopback, recording every method it is sent."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(FIXTURE, encoding="utf-8")
    received: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append(json.loads(self.rfile.read(length) or b"{}").get("method"))
            payload = json.dumps({"result": {"corpus": "absent"}, "elapsed_ms": 4}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/", received
    finally:
        server.shutdown()
        server.server_close()


@LIVE
def test_the_tools_round_trip_in_one_batch(tmp_path, fixture_site, capsys):
    pytest.importorskip("playwright")
    url, received = fixture_site
    upload = tmp_path / "probe.md"
    upload.write_text("probe\n")
    gif, trace = tmp_path / "out.gif", tmp_path / "trace.zip"
    steps = [
        {"tool": "snapshot", "interactive": True},
        {"tool": "find", "query": "press button"},
        {"tool": "click", "ref": "e3"},
        {"tool": "wait", "text": "pressed"},
        {"tool": "fill", "ref": "e4", "value": "ada"},
        {"tool": "click", "selector": "#name"},
        {"tool": "key", "keys": "End"},
        {"tool": "type", "text": " lovelace"},
        {"tool": "eval", "js": "document.getElementById('name').value"},
        {"tool": "dialog", "action": "accept"},
        {"tool": "click", "text": "Ask me"},
        {"tool": "wait", "text": "answer true"},
        {"tool": "upload", "selector": "#pick", "files": [str(upload)]},
        {"tool": "wait", "text": "picked probe.md"},
        {"tool": "checkbox", "selector": "#tick"},
        {"tool": "click", "selector": "#read"},
        {"tool": "wait", "text": "status: absent"},
        {"tool": "click", "selector": "#wipe"},
        {"tool": "wait", "text": "clear_corpus: refused by the browser agent"},
        {"tool": "console", "pattern": "pressed"},
        {"tool": "network", "rpc": True},
        {"tool": "download", "selector": "#dl"},
        {"tool": "record", "action": "start"},
        {"tool": "trace", "action": "start"},
        {"tool": "click", "ref": "e3"},
        {"tool": "resize", "device": "phone"},
        {"tool": "screenshot", "zoom": 2, "clip": "0,0,200,100"},
        {"tool": "screenshot", "selector": "#press", "zoom": 2},
        {"tool": "trace", "action": "stop", "path": str(trace)},
        {"tool": "record", "action": "stop", "path": str(gif)},
        {"tool": "tabs", "action": "new", "url": "about:blank"},
        {"tool": "tabs", "action": "close", "index": 2},
    ]
    file = tmp_path / "steps.json"
    file.write_text(json.dumps(steps))
    code = ba.main(["batch", str(file), "--open", url, "--images", str(tmp_path / "images")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ada lovelace" in out
    assert '[log] pressed the button' in out
    assert "status 200 4ms" in out and "clear_corpus blocked: clear_corpus is never sent" in out
    assert received == ["status"], "the blocked call reached the server"
    assert re.search(r"-shot\.png \(400x200\)", out), out
    assert gif.stat().st_size > 0 and trace.stat().st_size > 0
    downloads = list((tmp_path / "images").glob("*download-hello.txt"))
    assert downloads and downloads[0].read_text() == "hello download"
    assert "tab 1/1" in out.splitlines()[-1] or "tab 1/1" in out.splitlines()[-2]


# Every way past a `**/rpc` glob, and a body only a bytes-reading JSON parser
# reads. Each fetch gives up after five seconds, so a guard that dies on a
# body fails this rather than hanging it.
EVADE_JS = """(async () => {
  const wipe = JSON.stringify({method: 'clear_corpus'});
  const utf16 = new Uint8Array([0xff, 0xfe, ...[...wipe].flatMap(c => [c.charCodeAt(0), 0])]);
  const tries = [['/rpc', wipe], ['/rpc?x=1', wipe], ['/rpc;x', wipe], ['/rpc?', wipe],
                 ['///rpc', wipe], ['/rpc', utf16]];
  const out = [];
  for (const [target, body] of tries) {
    const stop = new AbortController();
    const timer = setTimeout(() => stop.abort(), 5000);
    try {
      const r = await fetch(location.origin + target, {method: 'POST', body, signal: stop.signal});
      out.push(target + ' -> ' + (await r.text()).slice(0, 60));
    } catch (e) {
      out.push(target + ' -> ' + e.name);
    } finally {
      clearTimeout(timer);
    }
  }
  return out.join('\\n');
})()"""


@LIVE
def test_a_page_cannot_route_a_refused_call_around_the_guard(tmp_path, fixture_site, capsys):
    """The fixture records a POST to any path, which is stricter than serve.py:
    nothing at all may reach it."""
    pytest.importorskip("playwright")
    url, received = fixture_site
    file = tmp_path / "steps.json"
    file.write_text(json.dumps([{"tool": "eval", "js": EVADE_JS}]))
    code = ba.main(["batch", str(file), "--open", url])
    out = capsys.readouterr().out
    assert code == 0, out
    assert received == [], f"refused calls reached the server: {received}\n{out}"
    answers = [line for line in out.splitlines() if re.match(r"/+rpc\S* -> ", line)]
    assert len(answers) == 6 and all(ba.BLOCKED_PREFIX in line for line in answers), out
    assert "blocked calls: 6" in out, out


# ---------------------------------------------------------------------------
# doctor and mcp-check, without a browser or a network
# ---------------------------------------------------------------------------


def test_doctor_with_no_browser_exits_2_and_says_how_to_get_one(monkeypatch, capsys):
    monkeypatch.setattr(ba, "resolve_chromium", lambda explicit=None: (None, "no chromium found"))
    assert ba.main(["doctor", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["browser"]["path"] is None
    assert report["problems"] == ["no chromium found"]
    assert any("pacman -S chromium" in fix for fix in report["fixes"])
    assert set(report) >= {"ok", "browser", "sandbox", "problems", "fixes"}


def test_doctor_without_playwright_exits_1(monkeypatch, capsys, tmp_path):
    chrome = tmp_path / "chrome"
    chrome.write_text("")
    monkeypatch.setattr(ba, "resolve_chromium", lambda explicit=None: (str(chrome), "--chromium"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert ba.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["browser"]["path"] == str(chrome)
    assert any('.[browser]' in fix for fix in report["fixes"])


def test_a_sandbox_failure_names_the_sysctl_that_blocks_it(monkeypatch):
    values = {"/proc/sys/kernel/unprivileged_userns_clone": "0",
              "/proc/sys/user/max_user_namespaces": "0",
              "/proc/sys/kernel/apparmor_restrict_unprivileged_userns": "1"}
    monkeypatch.setattr(ba, "_read_sysctl", values.get)
    problems, fixes = ba.sandbox_fixes()
    assert len(problems) == 3 and len(fixes) == 3
    assert "sudo sysctl kernel.unprivileged_userns_clone=1" in fixes
    monkeypatch.setattr(ba, "_read_sysctl", {"/proc/sys/user/max_user_namespaces": "64300"}.get)
    problems, fixes = ba.sandbox_fixes()
    assert not fixes and "look allowed" in problems[0]


# Speaks just enough MCP over stdio to stand in for the real server, and
# answers the way it was started: its argv and its sandbox variable.
FAKE_MCP = r'''
import json, os, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg["method"], msg.get("params", {})
    if method == "initialize":
        result = {"serverInfo": {"name": "stand-in", "version": "0"}, "capabilities": {},
                  "protocolVersion": params["protocolVersion"]}
    elif method == "tools/list":
        names = ["browser_navigate", "browser_snapshot", "browser_close", "browser_run_code_unsafe"]
        result = {"tools": [{"name": n} for n in names]}
    elif params.get("name") == "browser_snapshot":
        text = "argv=" + json.dumps(sys.argv[1:]) + " sandbox=" + os.environ.get("PLAYWRIGHT_MCP_SANDBOX", "unset")
        result = {"content": [{"type": "text", "text": "- heading \"browser agent mcp check\" " + text}]}
    else:
        result = {"content": [{"type": "text", "text": "ok"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
'''


def test_mcp_check_speaks_mcp_to_the_pinned_server(tmp_path, monkeypatch, capsys):
    npx = tmp_path / "npx"
    npx.write_text(f"#!{sys.executable}\n" + FAKE_MCP)
    npx.chmod(0o755)
    chrome = tmp_path / "chrome"
    chrome.write_text("")
    monkeypatch.setattr(ba, "MCP_OUTPUT_DIR", tmp_path / "mcp-out")
    code = ba.main(["mcp-check", "--npx", str(npx), "--chromium", str(chrome), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0, report
    assert report["ok"] and report["tools"] == 4 and report["snapshot_has_text"]
    assert report["command"][1:] == ["-y", "@playwright/mcp@0.0.83", "--headless", "--isolated",
                                     "--executable-path", str(chrome), "--output-dir",
                                     str(tmp_path / "mcp-out")]
    root = os.geteuid() == 0
    assert ("sandbox=false" in report["snapshot_excerpt"]) is root
    assert ("sandbox=unset" in report["snapshot_excerpt"]) is (not root)
    assert report["run_code_unsafe"]["offered"] is True


def test_mcp_check_without_npx_exits_2(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ba.shutil, "which", lambda name: None)
    assert ba.main(["mcp-check", "--chromium", str(SCRIPT), "--json"]) == 2
    assert "no npx" in json.loads(capsys.readouterr().out)["problems"][0]


def test_headed_without_a_display_is_refused_before_any_launch(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert ba.main(["wait-for-user", "--until-url", "done", "--headed"]) == 2
    assert "needs a display" in capsys.readouterr().err
    assert ba.main(["check", "--headed", "--passes", "tour", "--out", str(tmp_path)]) == 2
    capsys.readouterr()
    assert "needs a display" in json.loads((tmp_path / "results.json").read_text())["problems"][0]


def test_an_answer_is_read_without_its_title_and_badge():
    answer = {"title": "Run failed", "badge": "FAIL",
              "text": "Run failed FAIL\n\nollama-daemon is unavailable: its circuit opened\n\nWritten: nothing"}
    assert ba._answer_detail(answer) == "ollama-daemon is unavailable: its circuit opened"
    assert ba._answer_detail({"title": "Architect verdict:", "badge": "approved", "text": ""}) == ""
    assert ba._refused_run("A run is already in flight. Stop it before starting another.")
    assert not ba._refused_run("ollama-daemon is unavailable")


def test_a_gif_needs_frames_and_ffmpeg(tmp_path, monkeypatch):
    assert "no frames" in ba.encode_gif([], tmp_path / "x.gif")
    frame = tmp_path / "000000.jpg"
    frame.write_bytes(b"not really a jpeg")
    monkeypatch.setattr(ba.shutil, "which", lambda name: None)
    assert "ffmpeg is not installed" in ba.encode_gif([(frame, 0.0)], tmp_path / "x.gif")
