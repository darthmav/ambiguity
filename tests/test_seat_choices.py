"""The console's seat choices outlast the process, and a run records its seats.

A seat moved in the console lasted only as long as the process: every restart
put the configured seats back, and nothing said afterwards which model a past
run had used. The choice is now kept under `RUNS_DIR` and seated again when the
console starts, and the run's snapshot names what each seat held and called.

A fresh install has no record, so it runs the configured seats -- `.env`'s,
else `DEFAULT_SEATS` -- and a record never outranks a default or an `.env`
that moved after the choice was made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import serve
from langgraph_agent import config
from langgraph_agent.config import AGENTS, CLAUDE_SONNET, DEFAULT_SEATS

# What the stubbed daemon lists: two tags to move a seat onto, and every
# default, so a seat can be moved back.
TAGS = ["plain:latest", "other:latest", *sorted({s["model"] for s in DEFAULT_SEATS.values()})]


@pytest.fixture(autouse=True)
def _fresh_seats(monkeypatch):
    """No `.env` seat, no choice, no call: the defaults, as on a fresh install.

    `load_dotenv` ran at import, so a developer's own `.env` is cleared here,
    or it would decide what "configured" means. The daemon is stubbed in both
    modules that ask it: `serve` for the offer, `config` for the card.
    """
    for agent in AGENTS:
        for suffix in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY"):
            monkeypatch.delenv(f"{agent.upper()}_{suffix}", raising=False)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_agent_llm_overrides", {})
    monkeypatch.setattr(config, "_agent_thinking", {})
    monkeypatch.setattr(config, "_seat_failures", {})
    monkeypatch.setattr(config, "_seat_calls", {})
    for module in (serve, config):
        monkeypatch.setattr(module, "ollama_daemon_tags", lambda: list(TAGS))
        monkeypatch.setattr(module, "ollama_model_capabilities", lambda tag: ["completion", "thinking"])
    monkeypatch.setattr(serve, "_last_ollama_options", [])


def _restart(monkeypatch) -> None:
    """What a new process holds before it reads the record, then the reading."""
    monkeypatch.setattr(config, "_agent_llm_overrides", {})
    monkeypatch.setattr(config, "_agent_thinking", {})
    serve._restore_seat_choices()


def _seats() -> dict[str, dict[str, str]]:
    return {agent: config.get_agent_model_info(agent) for agent in AGENTS}


def test_a_fresh_install_runs_the_default_seats(monkeypatch):
    _restart(monkeypatch)

    assert _seats() == DEFAULT_SEATS
    # Reading writes nothing: no record until someone moves a seat.
    assert not serve._seats_path().exists()


def test_a_seat_chosen_in_the_console_is_seated_again_after_a_restart(monkeypatch):
    serve.rpc_set_seat({"agent": "builder", "provider": "ollama", "model": "plain:latest"})
    assert serve._seats_path().exists()

    _restart(monkeypatch)

    assert _seats() == {**DEFAULT_SEATS, "builder": {"provider": "ollama", "model": "plain:latest"}}


def test_moving_a_seat_back_leaves_nothing_to_restore(monkeypatch):
    """A seat on its configured model needs no record, and with none it keeps
    its `{ROLE}_API_KEY` and follows a later `.env` edit."""
    default = DEFAULT_SEATS["builder"]
    serve.rpc_set_seat({"agent": "builder", "provider": "ollama", "model": "plain:latest"})
    serve.rpc_set_seat({"agent": "builder", **default})

    assert json.loads(serve._seats_path().read_text())["seats"] == {}
    _restart(monkeypatch)
    assert _seats() == DEFAULT_SEATS


@pytest.mark.parametrize("moved", ["env", "default"])
def test_a_configured_seat_that_moved_since_the_choice_wins(monkeypatch, capsys, moved):
    """An `.env` edit is the operator's newer word, and a new default is what
    an update installs: a choice made against the old one does not mask it."""
    serve.rpc_set_seat({"agent": "builder", "provider": "ollama", "model": "plain:latest"})
    if moved == "env":
        monkeypatch.setenv("BUILDER_MODEL", "other:latest")
    else:
        monkeypatch.setitem(config.DEFAULT_SEATS, "builder", {"provider": "ollama", "model": "other:latest"})

    _restart(monkeypatch)

    assert config.get_agent_model_info("builder") == {"provider": "ollama", "model": "other:latest"}
    assert "changed since plain:latest was chosen" in capsys.readouterr().out


@pytest.mark.parametrize("record", [
    "not json {",
    "[]",
    json.dumps({"seats": {"builder": {"provider": "openai", "model": "gpt", "configured": {}}}}),
    json.dumps({"seats": {"cook": {"provider": "ollama", "model": "plain:latest"}}}),
    json.dumps({"seats": {"builder": "plain:latest"}, "thinking": {"builder": "yes"}}),
])
def test_a_record_this_console_cannot_read_leaves_the_configured_seats(monkeypatch, record):
    path = serve._seats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record, encoding="utf-8")

    _restart(monkeypatch)

    assert _seats() == DEFAULT_SEATS
    assert config._agent_thinking == {}


def test_a_thinking_switch_is_kept_too(monkeypatch):
    serve.rpc_set_thinking({"agent": "planner", "thinking": True})

    _restart(monkeypatch)

    assert config.get_agent_thinking("planner") is True
    assert config.get_agent_thinking("architect") is config.DEFAULT_THINKING


def test_importing_the_console_restores_nothing(tmp_path):
    """Only the console's start reads the record. Tests and scripts import
    `serve`, and must run the configured seats whatever an operator chose.

    In a process of its own, with a record waiting in its `RUNS_DIR`.
    """
    root = Path(serve.__file__).resolve().parent
    script = (
        "import json, os, pathlib, sys\n"
        f"sys.path[:0] = [{str(root)!r}, {str(root / 'src')!r}]\n"
        "from langgraph_agent import config\n"
        "configured = config._configured_info('builder')\n"
        "runs = pathlib.Path(os.environ['RUNS_DIR'])\n"
        "runs.mkdir()\n"
        "(runs / 'seats.json').write_text(json.dumps({'seats': {'builder': {\n"
        "    'provider': 'ollama', 'model': 'plain:latest', 'configured': configured}}}))\n"
        "import serve\n"
        "imported = config.get_agent_model_info('builder')\n"
        "serve._restore_seat_choices()\n"
        "print(json.dumps([configured, imported, config.get_agent_model_info('builder')]))\n"
    )
    env = {**os.environ, "RUNS_DIR": str(tmp_path / "runs"), "REBUILD_CORPUS": "0",
           "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr[-2000:]
    configured, imported, started = json.loads(done.stdout.strip().splitlines()[-1])
    assert imported == configured
    assert started == {"provider": "ollama", "model": "plain:latest"}


class _Graph:
    """A run in which the Planner's seat is called twice and no other is."""

    def stream(self, state, cfg):
        for _ in range(2):
            config.get_agent_llm("planner").invoke(["You are the Planner."])
        yield {"planner": {**state, "messages": ["[Planner] planned"]}}


def test_a_run_records_the_seats_it_held_and_what_each_called(monkeypatch):
    """Held is what the cards said at the start; called is what ran. The
    Planner is on Anthropic with no key, so what it called was the stub."""
    monkeypatch.setattr(serve, "graph", _Graph())
    config.set_agent_llm("planner", "anthropic", CLAUDE_SONNET)

    result = serve.rpc_run_goal({"goal": "g"})

    seats = result["seats"]
    assert list(seats) == list(AGENTS)
    assert seats["planner"] == {
        "provider": "anthropic", "model": CLAUDE_SONNET,
        "used": [{"provider": "anthropic", "model": CLAUDE_SONNET, "stub": True, "calls": 2}],
    }
    assert seats["architect"] == {**DEFAULT_SEATS["architect"], "used": []}
    # On disk too: the record is for the console that restarts.
    assert json.loads(serve.LAST_RUN_PATH.read_text())["seats"] == seats


def test_a_run_starts_its_record_empty(monkeypatch):
    """Calls made before the run was claimed belong to no run."""
    config.set_agent_llm("planner", "anthropic", CLAUDE_SONNET)
    config.get_agent_llm("planner").invoke(["You are the Planner."])

    class _Idle:
        def stream(self, state, cfg):
            yield {"architect": dict(state)}

    monkeypatch.setattr(serve, "graph", _Idle())
    result = serve.rpc_run_goal({"goal": "g"})

    assert all(seat["used"] == [] for seat in result["seats"].values())
