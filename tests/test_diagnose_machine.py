"""The machine diagnostic: what it reads, what it never carries, and that a
machine with nothing on it still gets a whole report.

Every test hands `scripts/diagnose_machine.py` a fake machine -- stand-ins for
its `run` and `http` -- so nothing here needs a GPU, a daemon, a console, a
database or a network, and nothing here changes the machine it runs on.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_machine.py"


def _load() -> Any:
    """Import the script by path. Registered in `sys.modules` first, since
    `@dataclass` resolves annotations through the module it was defined in."""
    spec = importlib.util.spec_from_file_location("diagnose_machine", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_machine"] = module
    spec.loader.exec_module(module)
    return module


dm = _load()

PLANTED_KEY = "sk-ant-test0123456789"


class FakeMachine:
    """Commands answer from a table of prefixes; anything else is not installed.

    Project snippets (`python -c CODE ROOT ...`) answer by the marker their
    code prints, so a test can say what "the project" reports.
    """

    def __init__(self, commands: dict[tuple[str, ...], Any] | None = None,
                 snippets: dict[str, Any] | None = None,
                 urls: dict[tuple[str, str], Any] | None = None) -> None:
        self.commands = commands or {}
        self.snippets = snippets or {}
        self.urls = urls or {}
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.requests: list[tuple[str, str, Any]] = []

    def run(self, cmd: list[str], timeout: float = 10.0, **kwargs: Any) -> Any:
        self.calls.append((list(cmd), kwargs))
        if len(cmd) > 2 and cmd[1] == "-c":
            marker = re.search(r'"@@(\w+) "', cmd[2])
            if marker and marker[1] in self.snippets:
                found = self.snippets[marker[1]]
                found = found(cmd) if callable(found) else found
                return dm.Ran(0, out=f"@@{marker[1]} {json.dumps(found)}\n")
        for prefix, answer in sorted(self.commands.items(), key=lambda kv: -len(kv[0])):
            if tuple(cmd[:len(prefix)]) == prefix:
                return answer(cmd, kwargs) if callable(answer) else answer
        return dm.Ran(None, err=f"{cmd[0]} is not installed", missing=True)

    def http(self, method: str, url: str, body: Any = None, timeout: float = 5.0) -> Any:
        self.requests.append((method, url, body))
        for (verb, fragment), answer in self.urls.items():
            if verb == method and fragment in url:
                return answer(body) if callable(answer) else answer
        return dm.Answer(None, error="ConnectionRefusedError: [Errno 111] Connection refused")

    def ran(self, *prefix: str) -> list[tuple[list[str], dict[str, Any]]]:
        return [(cmd, kw) for cmd, kw in self.calls if tuple(cmd[:len(prefix)]) == prefix]


def console(status: dict[str, Any], results: dict[str, Any] | None = None) -> dict:
    """URL handlers for a console answering `status` and these RPC results."""
    results = results or {}

    def rpc(body: Any) -> Any:
        method = (body or {}).get("method")
        return dm.Answer(200, json.dumps({"result": results.get(method, {}), "elapsed_ms": 1}))

    return {("GET", "/api/status"): dm.Answer(200, json.dumps(status)), ("POST", "/rpc"): rpc}


def ok(out: str = "") -> Any:
    return dm.Ran(0, out=out)


def make_ctx(tmp_path: Path, machine: FakeMachine, *argv: str, **overrides: Any) -> Any:
    args = dm.parse_args(["--base", "http://localhost:8080", *argv])
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = {
        "run": machine.run, "http": machine.http, "env": {"PATH": "/usr/bin"},
        "home": tmp_path / "home", "background": False, "tty": False, "euid": 1000,
    }
    fields.update(overrides)
    return dm.Context(out=out, args=args, **fields)


def run_main(tmp_path: Path, machine: FakeMachine, *argv: str, **overrides: Any) -> tuple[int, Path]:
    out = tmp_path / "out"
    fields: dict[str, Any] = {
        "run": machine.run, "http": machine.http,
        "env": {"PATH": "/usr/bin", "CONSOLE_LOG": str(tmp_path / "no-console.log")},
        "home": tmp_path / "home", "background": False, "tty": False, "euid": 1000,
    }
    fields.update(overrides)
    code = dm.main(["--base", "http://localhost:8080", "--out", str(out), *argv], **fields)
    return code, out


# --------------------------------------------------------------------------
# nvidia-smi, lspci and /api/ps, read
# --------------------------------------------------------------------------


def test_no_gpu_is_no_cards_not_a_crash():
    assert dm.parse_gpu_csv("") == []
    assert dm.parse_gpu_csv("No devices were found\n") == []


def test_a_field_a_card_cannot_report_is_none():
    [card] = dm.parse_gpu_csv(
        "0, NVIDIA GeForce GTX 1060 6GB, 580.95.05, 6144, 512, 6.1, Enabled, P8, [N/A], "
        "[Not Supported]\n")
    assert card["index"] == 0
    assert card["memory_total_mib"] == 6144
    assert card["compute_cap"] == "6.1"
    assert card["display_active"] is True
    assert card["temperature_c"] is None
    assert card["utilization_pct"] is None


def test_two_cards_and_a_name_with_a_comma():
    cards = dm.parse_gpu_csv(
        "0, NVIDIA GeForce GTX 1080, 580.95.05, 8192, 900, 6.1, Enabled, P2, 51, 7\n"
        "1, Tesla V100, PCIe, 570.1, 16384, 0, 7.0, Disabled, P0, 40, 0\n")
    assert [c["index"] for c in cards] == [0, 1]
    assert cards[1]["name"] == "Tesla V100, PCIe"
    assert cards[1]["memory_total_mib"] == 16384
    assert cards[1]["display_active"] is False


@pytest.mark.parametrize("cap, below", [("6.1", True), ("7.0", True), ("7.5", False),
                                        ("8.6", False), (None, False), ("[N/A]", False)])
def test_compute_capability_below_seven_five(cap, below):
    assert dm.compute_below(cap) is below


def test_lspci_finds_the_nvidia_card_and_its_driver():
    devices = dm.parse_lspci(
        "00:02.0 VGA compatible controller [0300]: Intel Corporation UHD [8086:9bc4]\n"
        "\tKernel driver in use: i915\n"
        "01:00.0 3D controller [0302]: NVIDIA Corporation GP106 [10de:1c03] (rev a1)\n"
        "\tSubsystem: Something\n"
        "\tKernel driver in use: nvidia\n"
        "02:00.0 Ethernet controller [0200]: Realtek [10ec:8168]\n")
    assert [(d["nvidia"], d["driver"]) for d in devices] == [(False, "i915"), (True, "nvidia")]


def test_placement_reads_the_cpu_share_from_api_ps():
    rows = dm.placement({"models": [
        {"name": "qwen3.8:latest", "size": 16 * 2**30, "size_vram": 4 * 2**30},
        {"name": "qwen3-embedding:latest", "size": 8 * 2**30, "size_vram": 8 * 2**30},
        {"name": "nemotron-3-ultra:cloud", "size": 0, "size_vram": 0},
    ]})
    assert [r["cpu_share"] for r in rows] == [0.75, 0.0, None]
    assert rows[0]["size_gib"] == 16.0 and rows[0]["vram_gib"] == 4.0
    assert dm.placement(None) == [] and dm.placement({}) == []


# --------------------------------------------------------------------------
# The gpu and ollama sections, against a fake machine
# --------------------------------------------------------------------------


def test_an_nvidia_card_without_nvidia_smi_is_named(tmp_path):
    machine = FakeMachine({("lspci",): ok(
        "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA106 [10de:2503]\n")})
    section = dm.section_gpu(make_ctx(tmp_path, machine))
    assert [p.kind for p in section.problems] == ["nvidia-smi-missing"]


def test_an_old_card_the_daemon_skips_points_at_the_cuda_12_build(tmp_path):
    machine = FakeMachine({
        ("nvidia-smi", f"--query-gpu={dm.GPU_QUERY}"): ok(
            "0, NVIDIA GeForce GTX 1060 6GB, 580.95.05, 6144, 300, 6.1, Enabled, P8, 40, 0\n"),
        ("nvidia-smi",): ok(""),
        ("journalctl",): ok("ollama[812]: skipping CUDA device 0: compute capability 6.1\n"),
    })
    section = dm.section_gpu(make_ctx(tmp_path, machine))
    assert [p.kind for p in section.problems] == ["cuda-old-card"]
    assert "./cuda-embed-ollama.sh" in section.problems[0].fix


def test_a_journal_the_user_cannot_read_is_not_a_quiet_one(tmp_path):
    machine = FakeMachine({
        ("nvidia-smi", f"--query-gpu={dm.GPU_QUERY}"): ok(
            "0, NVIDIA GeForce GTX 1060 6GB, 580.95.05, 6144, 300, 6.1, Enabled, P8, 40, 0\n"),
        ("nvidia-smi",): ok(""),
        ("journalctl",): ok("Hint: You are currently not seeing messages from other users and "
                            "the system.\n-- No entries --\n"),
    })
    section = dm.section_gpu(make_ctx(tmp_path, machine))
    assert section.problems == []
    assert "could not be read to confirm" in section.data["note"]


def _ollama_machine(environment: str) -> FakeMachine:
    return FakeMachine(
        commands={
            ("ollama", "--version"): ok("ollama version is 0.12.3"),
            ("systemctl", "--version"): ok("systemd 258"),
            ("systemctl", "is-enabled"): ok("enabled"),
            ("systemctl", "is-active"): ok("active"),
            ("systemctl", "show", "ollama.service", "-p", "Environment"): ok(environment),
            ("bash",): ok("  ✓ ollama.service starts at boot\n"),
            ("journalctl",): ok(""),
        },
        urls={
            ("GET", "/api/version"): dm.Answer(200, '{"version": "0.12.3"}'),
            ("GET", "/api/tags"): dm.Answer(200, json.dumps({"models": [
                {"name": "qwen3.8:latest"}, {"name": "qwen3-embedding:latest"}]})),
            ("POST", "/api/show"): dm.Answer(200, '{"capabilities": ["completion", "tools"]}'),
            ("GET", "/api/ps"): dm.Answer(200, '{"models": []}'),
        },
    )


_FACTS = {
    "seats": [{"role": "builder", "provider": "ollama", "model": "qwen3.8:latest",
               "local": True, "candidate": "qwen"}],
    "embedding_model": "qwen3-embedding:latest", "ollama_url": "http://localhost:11434",
}


def test_the_drop_in_must_hold_one_model(tmp_path):
    machine = _ollama_machine("Environment=OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_NUM_PARALLEL=1")
    ctx = make_ctx(tmp_path, machine, facts_cache=dict(_FACTS))
    section = dm.section_ollama(ctx)
    [problem] = [p for p in section.problems if p.kind == "one-model"]
    assert "OLLAMA_MAX_LOADED_MODELS is '2', not 1" in problem.evidence
    assert "OLLAMA_NUM_PARALLEL" not in problem.evidence


def test_a_daemon_with_the_drop_in_and_its_models_is_clean(tmp_path):
    machine = _ollama_machine(
        'Environment="OLLAMA_MAX_LOADED_MODELS=1" OLLAMA_NUM_PARALLEL=1 HOME=/var/lib/ollama')
    ctx = make_ctx(tmp_path, machine, facts_cache=dict(_FACTS))
    section = dm.section_ollama(ctx)
    assert section.problems == [], section.problems
    assert ctx.daemon_up is True
    assert section.data["configured"]["qwen3.8:latest"]["pulled"] is True


def test_an_unset_drop_in_variable_is_a_gap():
    assert dm.one_model_gaps(dm.one_model_settings("Environment=")) == [
        "OLLAMA_MAX_LOADED_MODELS is not set", "OLLAMA_NUM_PARALLEL is not set"]


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def test_redaction_takes_out_keys_tokens_passwords_addresses_and_paths(monkeypatch):
    monkeypatch.setenv("HOME", "/home/alice")
    text = "\n".join([
        f"ANTHROPIC_API_KEY={PLANTED_KEY}",
        "token ghp_abcdefghijklmnopqrstuvwxyz0123 and github_pat_abcdefghijklmnopqrstuv_0123",
        "hf_abcdefghijklmnopqrstuvwxyz",
        "Authorization: Bearer abc.def.ghijklmnop",
        "OLLAMA_API_KEY: plainvalue",
        "postgresql://postgres:hunter2@127.0.0.1:5432/postgres",
        "mail someone.else@example.org, remote git@github.com:owner/repo.git",
        "/home/alice/ambiguity/.env and /home/bob/x and /run/user/1000/bus",
    ])
    counts: Counter[str] = Counter()
    cleaned = dm.redact(text, counts)
    for secret in (PLANTED_KEY, "ghp_abcdef", "github_pat_", "hf_abcdef", "abc.def.ghijklmnop",
                   "plainvalue", "hunter2", "someone.else@example.org", "/home/alice", "bob",
                   "/run/user/1000"):
        assert secret not in cleaned, secret
    assert "git@github.com:owner/repo.git" in cleaned
    assert "~/ambiguity/.env" in cleaned and "/run/user/<uid>/bus" in cleaned
    assert {"anthropic-key", "github-token", "huggingface-token", "bearer", "url-password",
            "email", "home", "runtime-dir", "assigned-secret"} <= set(counts)


@pytest.mark.parametrize("prose", [
    "claude-opus-5 (no key: canned stub output)",
    "The key: this matters",
    "monkey: banana",
    "prompt_tokens: 12",
    "password: (none)",
    "API key:\nnone",
])
def test_prose_after_a_colon_is_not_a_secret(prose):
    counts: Counter[str] = Counter()
    assert dm.redact(prose, counts) == prose
    assert not counts, "a report would claim a secret that was never there"


@pytest.mark.parametrize("setting, secret", [
    ("ANTHROPIC_API_KEY=abc-def-123", "abc-def-123"),
    ("OLLAMA_API_KEY=plainvalue", "plainvalue"),
    ('"api_key": "plainvalue"', "plainvalue"),
    ('"token": "plainvalue"', "plainvalue"),
    ("PGPASSWORD: hunter2", "hunter2"),
    ("password: hunter2", "hunter2"),
    ("token=abc", "token=abc"),
    ("x-api-key: plainvalue", "plainvalue"),
    ("the key: a1b2c3d4e5f6", "a1b2c3d4e5f6"),
    ("GET /rpc?token=abc123&page=2", "abc123"),
    ('{\\"token\\": \\"plainvalue\\"}', "plainvalue"),
])
def test_a_setting_that_names_a_secret_is_redacted(setting, secret):
    counts: Counter[str] = Counter()
    cleaned = dm.redact(setting, counts)
    assert secret not in cleaned
    assert counts["assigned-secret"] == 1


def test_redacted_data_stays_valid_json():
    value = {"note": 'KEY="abc\\"def"', f"{PLANTED_KEY}": [f"x {PLANTED_KEY}"]}
    cleaned = dm.redact_data(value)
    text = json.dumps(cleaned)
    assert PLANTED_KEY not in text
    assert json.loads(text) == cleaned


def test_a_value_under_a_secret_key_is_redacted_as_data():
    cleaned = dm.redact_data({"api_key": "plainvalue", "OLLAMA_API_KEY": "x",
                              "token": "a1b2c3d4e5f6", "key": "./projects/demo#7",
                              "seat": "claude-opus-5 (no key: canned stub output)"})
    assert cleaned == {"api_key": "<redacted>", "OLLAMA_API_KEY": "<redacted>",
                       "token": "<redacted>", "key": "./projects/demo#7",
                       "seat": "claude-opus-5 (no key: canned stub output)"}


@pytest.mark.parametrize("value", [
    {"url": "http://localhost:8080/rpc?token=abc123"},
    {"said": "the key: canned"},
    {"console": "error: password=hunter2"},
    {"page": 'token="abc"'},
])
def test_json_text_is_redacted_as_json(value):
    """Text-mode redaction swallowed a string's closing quote; as data it cannot."""
    for text, lines in ((json.dumps(value, indent=2), False), (json.dumps(value) + "\n", True)):
        cleaned = dm.redact_json_text(text, lines=lines)
        assert json.loads(cleaned) == dm.redact_data(value)
        assert "abc123" not in cleaned and "hunter2" not in cleaned


def test_log_json_writes_json_that_parses(tmp_path):
    ctx = make_ctx(tmp_path, FakeMachine())
    path = ctx.log_json("planted.json", {"said": 'token="abc"', "where": Path("/x"),
                                         "api_key": PLANTED_KEY})
    written = json.loads((ctx.out / path).read_text())
    assert written == {"said": 'token="<redacted>"', "where": "/x", "api_key": "<redacted>"}


# --------------------------------------------------------------------------
# Sign-ins: names and states, never values
# --------------------------------------------------------------------------


def test_claude_auth_status_runs_without_the_key_variables(tmp_path):
    seen: dict[str, Any] = {}

    def auth_status(cmd: list[str], kwargs: dict[str, Any]) -> Any:
        seen["env"] = dict(kwargs["env"])
        return ok(json.dumps({"loggedIn": True, "authMethod": "claude.ai",
                              "email": "someone@example.com", "orgName": "Acme"}))

    home = tmp_path / "home"
    (home / ".config" / "fish").mkdir(parents=True)
    (home / ".bashrc").write_text(f"# export CLAUDE_CODE_OAUTH_TOKEN=x\nexport "
                                  f"ANTHROPIC_API_KEY={PLANTED_KEY}\n")
    (home / ".config" / "fish" / "config.fish").write_text("set -gx ANTHROPIC_AUTH_TOKEN abc\n")
    env = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": PLANTED_KEY,
           "ANTHROPIC_AUTH_TOKEN": "tok-value-1", "CLAUDE_CODE_OAUTH_TOKEN": "tok-value-2"}
    machine = FakeMachine({("claude", "auth", "status"): auth_status, ("gh", "auth"): ok()})
    section = dm.section_sign_ins(make_ctx(tmp_path, machine, env=env, home=home,
                                           facts_cache={}))

    assert not set(dm.KEY_VARIABLES) & set(seen["env"]), "the key variables reached the child"
    assert seen["env"]["PATH"] == "/usr/bin"
    assert section.data["claude"] == {"logged_in": True, "method": "claude.ai"}
    keys = section.data["key_variables"]
    assert keys["exported_here"] == list(dm.KEY_VARIABLES)
    assert keys["in_rc_files"] == {"~/.bashrc": ["ANTHROPIC_API_KEY"],
                                   "~/.config/fish/config.fish": ["ANTHROPIC_AUTH_TOKEN"]}
    written = json.dumps([section.data, [p.__dict__ for p in section.problems]])
    for value in (PLANTED_KEY, "tok-value-1", "tok-value-2", "someone@example.com", "Acme"):
        assert value not in written
    assert [p.kind for p in section.problems] == ["claude-key-env"]


def test_a_sign_in_other_than_claude_ai_is_a_problem(tmp_path):
    machine = FakeMachine({("claude", "auth", "status"): ok(
        '{"loggedIn": true, "authMethod": "api_key"}'), ("gh", "auth"): ok()})
    section = dm.section_sign_ins(make_ctx(tmp_path, machine, facts_cache={}))
    assert [p.kind for p in section.problems] == ["claude-signin"]
    assert "api_key" in section.problems[0].what


# --------------------------------------------------------------------------
# Load control
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status, progress, reason", [
    ({"run_in_flight": True}, {"running": True, "goal": "write a poem"}, "a run is in flight"),
    ({"indexing": {"running": True, "message": "12/40 files"}}, {}, "rebuild is in flight"),
    ({"pull_request_follow": {"running": True, "number": 7}}, {}, "pull request #7"),
])
def test_a_busy_console_skips_the_sections_that_load_models(tmp_path, status, progress, reason):
    machine = FakeMachine(urls=console(status, {"run_progress": progress}))
    code, out = run_main(tmp_path, machine, "--sections", "embedder,seats")
    results = json.loads((out / "results.json").read_text())
    for name in ("embedder", "seats"):
        assert results["sections"][name]["status"] == "skipped"
        assert reason in results["sections"][name]["summary"]
    # Nothing was loaded: no snippet, no seat diagnostic, no generation.
    assert not [cmd for cmd, _ in machine.calls if "-c" in cmd or "diagnose_seats" in str(cmd)]
    assert not [url for _, url, _ in machine.requests if "/api/generate" in url]
    assert code == 0


def _browser_agent(calls: list[list[str]]) -> Any:
    """A stand-in `browser_agent.py check` that writes what the real one does."""
    def answer(cmd: list[str], kwargs: dict[str, Any]) -> Any:
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "shots").mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps({
            "schema": "ambiguity-browser/1",
            "passes": [{"name": "tour", "status": "pass"}, {"name": "run", "status": "pass"}]}))
        (out / "report.md").write_text(f"# browser\nthe page said {PLANTED_KEY}\n")
        (out / "shots" / "tour.png").write_bytes(b"\x89PNG")
        (out / "trace.zip").write_bytes(b"PK")
        return ok("done")
    return answer


@pytest.mark.parametrize("running", [False, True])
def test_with_runs_sends_one_run_only_to_an_idle_console(tmp_path, running):
    calls: list[list[str]] = []
    python = make_ctx(tmp_path, FakeMachine()).python
    agent = (python, str(ROOT / "scripts" / "browser_agent.py"), "check")
    machine = FakeMachine(commands={agent: _browser_agent(calls)},
                          urls=console({"run_in_flight": running},
                                       {"run_progress": {"running": running}}))
    code, out = run_main(tmp_path, machine, "--sections", "browser", "--with-runs")
    browser = json.loads((out / "results.json").read_text())["sections"]["browser"]

    assert "--allow" not in calls[0], "the read-only pass set never allows a mutation"
    if running:
        assert len(calls) == 1
        assert browser["data"]["runs"].startswith("skipped because a run is in flight")
    else:
        assert len(calls) == 2
        assert calls[1][calls[1].index("--passes") + 1] == "run"
        assert calls[1][calls[1].index("--allow") + 1] == "run"
        assert calls[1][calls[1].index("--out") + 1].endswith("browser-run")
    assert PLANTED_KEY not in (out / "browser" / "report.md").read_text()
    with tarfile.open(out.parent / f"{out.name}-share.tar.gz") as tar:
        names = tar.getnames()
    assert f"{out.name}/browser/shots/tour.png" in names
    assert not [n for n in names if n.endswith("trace.zip")]
    assert code == 0


@pytest.mark.parametrize("status, progress, quick", [
    ({"run_in_flight": False}, {"running": False}, False),
    ({"run_in_flight": True}, {"running": True, "goal": "write a poem"}, True),
    ({"indexing": {"running": True}}, {}, True),
])
def test_a_busy_console_gets_only_the_quick_browser_passes(tmp_path, status, progress, quick):
    """The default set searches through the embedder, which would wait on the
    run's seat and then evict it."""
    calls: list[list[str]] = []
    python = make_ctx(tmp_path, FakeMachine()).python
    agent = (python, str(ROOT / "scripts" / "browser_agent.py"), "check")
    machine = FakeMachine(commands={agent: _browser_agent(calls)},
                          urls=console(status, {"run_progress": progress}))
    section = dm.section_browser(make_ctx(tmp_path, machine))
    [call] = calls
    assert ("--quick" in call) is quick
    assert ("heavy_passes" in section.data) is quick
    if quick:
        assert section.data["heavy_passes"].startswith("skipped because a ")


def test_a_browser_report_with_secret_shaped_text_is_still_read(tmp_path):
    """Page text the browser agent left alone must not cost the report its passes."""
    def agent_run(cmd: list[str], kwargs: dict[str, Any]) -> Any:
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps({
            "schema": "ambiguity-browser/1",
            "passes": [{"name": "tour", "status": "pass"}, {"name": "graph", "status": "fail",
                                                            "reason": 'the page said token="abc"'}],
            "page_errors": [{"text": "API key:\nnone"}, {"text": "auth: password=hunter2"}],
            "rpc": {"error_envelopes": [{"method": "search_documents"}]}}, indent=2))
        (out / "rpc.jsonl").write_text(json.dumps({"url": "/rpc?token=abc123"}) + "\n")
        return dm.Ran(1, out="1 pass failed")

    python = make_ctx(tmp_path, FakeMachine()).python
    agent = (python, str(ROOT / "scripts" / "browser_agent.py"), "check")
    machine = FakeMachine(commands={agent: agent_run},
                          urls=console({"run_in_flight": False}, {"run_progress": {}}))
    ctx = make_ctx(tmp_path, machine)
    section = dm.section_browser(ctx)

    assert [p["name"] for p in section.data["passes"]] == ["tour", "graph"]
    assert section.data["page_errors"] == 2 and section.data["rpc_error_envelopes"] == 1
    assert [p.kind for p in section.problems] == ["browser-pass"]
    results = json.loads((ctx.out / "browser" / "results.json").read_text())
    assert results["passes"][1]["reason"] == 'the page said token="<redacted>"'
    assert "hunter2" not in json.dumps(results)
    [row] = [json.loads(line) for line in (ctx.out / "browser" / "rpc.jsonl").read_text().splitlines()]
    assert row == {"url": "/rpc?token=<redacted>"}


def test_the_status_fields_the_busy_check_reads_are_ones_the_console_sends():
    """Both sides of one contract: a rename in serve.py, or here, fails."""
    import serve

    status = serve.rpc_status({})
    assert {"run_in_flight", "indexing", "pull_request_follow"} <= set(status), sorted(status)
    # An idle console, in the shape `busy_reasons` reads.
    assert status["pull_request_follow"] == {"running": False, "number": None}


# --------------------------------------------------------------------------
# The Playwright MCP registration, asked as scripts/claude_tools.sh asks it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("answer, state, kinds", [
    (dm.Ran(None, err="\ntimed out after 120s", timed_out=True), "no answer within 120s",
     ["mcp-failed"]),
    (dm.Ran(1, err='No MCP server named "playwright". Run `claude mcp add` to add one.'),
     "not registered", ["mcp-unregistered"]),
    (dm.Ran(1, err="Error: config file is not valid JSON"),
     "could not be read: Error: config file is not valid JSON", ["mcp-failed"]),
    (ok("playwright:\n  Scope: Local config (private to you in this project)\n"
        "  Status: ✓ Connected\n  Type: stdio\n  Command: npx\n"),
     {"status": "✓ Connected", "scope": "Local config (private to you in this project)"}, []),
])
def test_the_mcp_registration_is_read_from_the_checkout(tmp_path, answer, state, kinds):
    seen: dict[str, Any] = {}
    machine = FakeMachine({("claude", "--version"): ok("2.1.295 (Claude Code)")})

    def run(cmd: list[str], timeout: float = 10.0, **kwargs: Any) -> Any:
        if cmd[:4] == ["claude", "mcp", "get", "playwright"]:
            seen.update(timeout=timeout, cwd=kwargs.get("cwd"))
            return answer
        return machine.run(cmd, timeout, **kwargs)

    section = dm.section_tools(make_ctx(tmp_path, machine, run=run, facts_cache={}))
    # A local-scope server is keyed by the project's path, and `mcp get`
    # starts npx to health-check it.
    assert seen == {"timeout": 120.0, "cwd": str(dm.ROOT)}
    assert section.data["playwright_mcp"] == state
    assert [p.kind for p in section.problems if p.kind.startswith("mcp-")] == kinds


def test_an_idle_console_lets_the_embedder_run(tmp_path):
    machine = FakeMachine(
        snippets={"embedder": {"model": "qwen3-embedding:latest", "expected_dimensions": 4096,
                               "dimensions": 4096, "cold_s": 4.2, "warm_s": 0.1,
                               "cpu_share": 0.0, "loaded_before": False}},
        urls=console({"run_in_flight": False}, {"run_progress": {"running": False}}))
    code, out = run_main(tmp_path, machine, "--sections", "embedder")
    section = json.loads((out / "results.json").read_text())["sections"]["embedder"]
    assert section["status"] == "ok", section
    assert code == 0


# --------------------------------------------------------------------------
# The whole run: a machine with nothing on it, a section that raises, --sections
# --------------------------------------------------------------------------


def test_a_machine_with_nothing_on_it_still_gets_a_report(tmp_path):
    env = {"PATH": "/usr/bin", "CONSOLE_LOG": str(tmp_path / "no-console.log"),
           "ANTHROPIC_API_KEY": PLANTED_KEY}
    secret_log = tmp_path / "install.log"
    secret_log.write_text(f"installing\nkey {PLANTED_KEY} for someone@example.com\n")
    machine = FakeMachine()
    code, out = run_main(tmp_path, machine, "--save-log", str(secret_log), env=env)

    assert code == 1
    results = json.loads((out / "results.json").read_text())
    report = (out / "report.md").read_text()
    assert results["schema"] == "ambiguity-diagnostics/1"
    assert results["verdict"] == "problems"
    assert list(results["sections"]) == [n for n in dm.SECTIONS if n != "suite"]
    assert not [n for n, s in results["sections"].items() if s["status"] == "error"]
    kinds = {p["kind"] for s in results["sections"].values() for p in s["problems"]}
    assert {"console-down", "no-gpu", "ollama-missing", "claude-missing",
            "claude-key-env"} <= kinds
    assert "## problems and fixes" in report and "./launch_console.sh" in report
    for name in ("embedder", "seats"):
        assert results["sections"][name]["status"] == "skipped"

    bundle = out.parent / f"{out.name}-share.tar.gz"
    assert results["bundle"]["bytes"] == bundle.stat().st_size
    with tarfile.open(bundle) as tar:
        names = tar.getnames()
        contents = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isfile())
    assert f"{out.name}/report.md" in names and f"{out.name}/logs/saved-install.log" in names
    for path in out.rglob("*"):
        if path.is_file():
            assert PLANTED_KEY not in path.read_text(errors="replace"), path
            assert "someone@example.com" not in path.read_text(errors="replace"), path
    assert PLANTED_KEY.encode() not in contents


def test_a_section_that_raises_becomes_an_error_not_a_crash(tmp_path, monkeypatch):
    def boom(ctx: Any) -> Any:
        raise RuntimeError("the card caught fire")

    monkeypatch.setattr(dm, "section_gpu", boom)
    code, out = run_main(tmp_path, FakeMachine(), "--sections", "machine,gpu")
    results = json.loads((out / "results.json").read_text())
    gpu = results["sections"]["gpu"]
    assert gpu["status"] == "error" and "the card caught fire" in gpu["summary"]
    assert "RuntimeError" in (out / gpu["data"]["traceback"]).read_text()
    assert results["sections"]["machine"]["status"] in ("ok", "problem")
    assert results["verdict"] == "incomplete"
    assert code == 1


def test_nothing_diagnosed_at_all_exits_two(tmp_path, monkeypatch):
    def boom(ctx: Any) -> Any:
        raise RuntimeError("no")

    monkeypatch.setattr(dm, "section_gpu", boom)
    code, _ = run_main(tmp_path, FakeMachine(), "--sections", "gpu")
    assert code == 2


def test_sections_runs_only_the_named_ones_in_their_own_order(tmp_path):
    code, out = run_main(tmp_path, FakeMachine(), "--sections", "gpu,machine")
    assert list(json.loads((out / "results.json").read_text())["sections"]) == ["machine", "gpu"]
    assert code == 1  # no GPU on the fake machine


def test_an_unknown_section_is_refused(tmp_path, capsys):
    code, out = run_main(tmp_path, FakeMachine(), "--sections", "machine,teleporter")
    assert code == 2
    assert "teleporter" in capsys.readouterr().err
    assert not (out / "results.json").exists()


def test_quick_leaves_out_the_sections_that_load_models():
    args = dm.parse_args(["--quick"])
    names = dm.select_sections(args)
    assert "embedder" not in names and "seats" not in names and "suite" not in names
    assert dm.select_sections(dm.parse_args(["--with-tests"]))[-1] == "suite"


# --------------------------------------------------------------------------
# The bundle, the fixes, and the console's own words
# --------------------------------------------------------------------------


def test_the_bundle_leaves_out_traces_recordings_downloads_and_env(tmp_path):
    out = tmp_path / "20261009-120000"
    files = ["report.md", "results.json", "logs/nvidia-smi.txt", "logs/.env",
             "seats/report.md", "seats/results.json", "browser/report.md",
             "browser/results.json", "browser/shots/1440-tour-graph.png", "browser/trace.zip",
             "browser/passes.gif", "browser/downloads/corpus.json", "browser/rpc.jsonl",
             "browser-run/report.md", "browser-run/shots/run.png", ".env", "browser/.env"]
    for name in files:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text("x")
    members = {str(m) for m in dm.bundle_members(out)}
    assert members == {"report.md", "results.json", "logs/nvidia-smi.txt", "seats/report.md",
                       "seats/results.json", "browser/report.md", "browser/results.json",
                       "browser/shots/1440-tour-graph.png", "browser-run/report.md",
                       "browser-run/shots/run.png"}
    bundle = dm.write_bundle(out, dm.bundle_members(out))
    with tarfile.open(bundle) as tar:
        names = tar.getnames()
        owners = {(m.uname, m.gname) for m in tar.getmembers()}
    assert not [n for n in names if n.endswith((".zip", ".gif", ".env")) or "downloads" in n]
    assert owners == {("", "")}


def test_every_path_a_fix_cites_exists():
    path_like = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|md|txt|sh|html|json|toml|ini|yml|cfg)$")
    dangling = []
    for kind, fix in dm.FIXES.items():
        for word in re.split(r"[\s`]+", fix):
            word = word.strip("\"'(),;:").rstrip(".")
            if word.startswith("./"):
                cited = word[2:]
            elif "/" in word and path_like.match(word):
                cited = word
            else:
                continue
            if not (ROOT / cited).exists():
                dangling.append(f"{kind} cites {cited}")
    assert not dangling, dangling


def test_every_problem_kind_the_sections_raise_has_a_fix():
    """A kind with no entry in FIXES reaches the report with no fix under it.

    Read from the source: every string a `finding(...)` call's first argument
    can be, and every value of a table that maps onto kinds (the console's
    badges, its health components).
    """
    import ast

    kind = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
    raised: set[str] = set()
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "finding" and node.args):
            raised |= {c.value for c in ast.walk(node.args[0])
                       if isinstance(c, ast.Constant) and isinstance(c.value, str)
                       and kind.match(c.value)}
        if isinstance(node, ast.Dict) and node.values and all(
                isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.values):
            values = {v.value for v in node.values}  # type: ignore[attr-defined]
            if values & set(dm.FIXES):
                raised |= values
    assert len(raised) > 20
    assert not raised - set(dm.FIXES), sorted(raised - set(dm.FIXES))


def test_the_console_payloads_are_read_in_its_own_words():
    problems = dm._console_problems({
        "healing": {"circuits": [{"name": "embedder-load", "state": "open", "failures": 3,
                                  "retry_in_s": 40}],
                    "health": {"postgres": {"status": "unhealthy", "details": "refused"},
                               "ollama-daemon": {"status": "healthy"}}},
        "rag_stats": {"corpus": "indexed", "staleness": {"stale": True, "missing": ["a.md"]}},
        "list_seats": {"seats": [{"role": "builder", "live": False, "badge": "NOT PULLED",
                                  "reason": "qwen3.8:latest not pulled"}]},
        "status": {"pull_requests": [{"number": 4, "status": "checks_failed"}]},
    })
    assert [p.kind for p in problems] == ["circuit-open", "postgres-down", "corpus-stale",
                                          "model-not-pulled", "pr-checks-failed"]


def test_seats_down_for_one_reason_are_one_problem():
    offline = {"live": False, "badge": "OFFLINE", "reason": "Ollama daemon unreachable"}
    problems = dm._console_problems({"list_seats": {"seats": [
        {"role": role, **offline} for role in ("architect", "planner", "researcher", "builder")
    ]}})
    [problem] = problems
    assert problem.kind == "daemon-down"
    assert "architect, planner, researcher, builder seats OFFLINE" in problem.what


def test_gist_refuses_without_a_terminal(tmp_path, capsys):
    machine = FakeMachine()
    ctx = make_ctx(tmp_path, machine, "--gist", tty=False)
    (ctx.out / "report.md").write_text("x")
    assert dm.share_as_gist(ctx, [Path("report.md")], print) == 1
    assert "anyone with its URL" in capsys.readouterr().out
    assert not machine.ran("gh")


def test_gist_needs_the_word_yes(tmp_path):
    machine = FakeMachine({("gh", "gist", "create"): ok("https://gist.github.com/x")})
    ctx = make_ctx(tmp_path, machine, "--gist", tty=True, ask=lambda prompt: "y")
    assert dm.share_as_gist(ctx, [Path("report.md")], lambda text: None) == 1
    assert not machine.ran("gh")
    ctx = make_ctx(tmp_path, machine, "--gist", tty=True, ask=lambda prompt: "yes")
    assert dm.share_as_gist(ctx, [Path("report.md")], lambda text: None) == 0
    assert machine.ran("gh", "gist", "create")


def test_the_script_loads_without_the_project_or_playwright():
    """Project and browser imports are lazy, so the machine sections run before
    `pip install -e .` and in CI without Playwright."""
    import ast

    tree = ast.parse(SCRIPT.read_text())
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name.split(".")[0] for node in top for alias in node.names}
    names |= {node.module.split(".")[0] for node in top
              if isinstance(node, ast.ImportFrom) and node.module}
    assert not names & {"langgraph_agent", "serve", "playwright", "psycopg"}
