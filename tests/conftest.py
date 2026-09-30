"""Pytest configuration for the 4-Agent System test suite.

Forces the LangGraph nodes to use the deterministic StubLLM so that graph-level
unit tests run quickly and do not depend on a live cloud LLM endpoint. Tests that
exercise MCP tools still use the real tool implementations.

The online research phase is switched off here for the same reason and with the
same force. `rpc_run_goal` now searches the web before the Architect opens, so
without this every test that starts a run would fetch a dozen live pages and
embed them: slow, non-deterministic, dependent on someone else's uptime, and
quietly mutating the developer's corpus as a side effect of running the suite.
It took the suite from 28s to 138s and broke a stop test whose five-second wait
had been generous. Tests that mean to exercise the phase turn it back on and
answer it with `MockTransport`.

The corpus bootstrap is switched off here for the same reason -- see
`_no_corpus_bootstrap`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import langgraph_agent.nodes as _nodes
import langgraph_agent.web_research as _web_research
from langgraph_agent.config import StubLLM
from langgraph_agent.control import RUN_CONTROL

# Patch the LLM lookup used by agent nodes so every test gets deterministic,
# parser-friendly responses without making network calls. This has to be
# `get_agent_llm`: it is what the nodes import, and patching `get_llm` here
# only set an unused attribute on the module.
_nodes.get_agent_llm = lambda agent, temperature=0.1: StubLLM()


@pytest.fixture(autouse=True)
def _clear_run_control():
    """No test may leak a stop into the next one.

    The stop is a process-global flag by design -- the graph compiles without a
    checkpointer, so there is nowhere else for it to live -- which means a test
    that sets it and does not clear it would silently make every later test's
    nodes bail before calling their model.
    """
    RUN_CONTROL.disarm()
    yield
    RUN_CONTROL.disarm()


@pytest.fixture(autouse=True)
def _no_snapshot_clobber(monkeypatch, tmp_path):
    """A test run must not overwrite the developer's last-run snapshot.

    `runs/last_run.json` is the recovery story for a real run -- it is what a
    reloaded console shows and the only copy of a stopped run's partial work.
    Any test calling `rpc_run_goal` writes it, so running the suite replaced it
    with the outcome of a fixture goal like `"g"`, silently, and the run someone
    actually cared about was gone.
    """
    import serve
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(serve, "LAST_RUN_PATH", tmp_path / "runs" / "last_run.json")


@pytest.fixture(autouse=True)
def _no_online_research(monkeypatch):
    """No test reaches the network, and none rewrites the corpus on disk.

    Overridden in `test_web_research.py`, which turns it back on and serves
    every request from a mock transport.
    """
    monkeypatch.setattr(_web_research, "WEB_SEARCH_ENABLED", False)


@pytest.fixture(autouse=True)
def _search_backend_is_duckduckgo(monkeypatch):
    """Every test searches through the DuckDuckGo path, whatever `.env` says.

    `SEARXNG_URL` is read at import, and `config.py` loads `.env` into the
    environment, so a developer whose `install.sh` added the SearxNG it runs
    had every web research test switch backends underneath it: seven failed,
    serving DuckDuckGo markup to a parser expecting SearxNG's JSON. The tests
    that mean SearxNG set it themselves.
    """
    monkeypatch.setattr(_web_research, "SEARXNG_URL", "")


@pytest.fixture(autouse=True)
def _no_corpus_bootstrap(monkeypatch):
    """No test builds a corpus, and none rebuilds the developer's.

    `rpc_run_goal` indexes the project when there is nothing to search, which
    is what makes a fresh install work. Left on, every test that starts a run
    would index the whole checkout -- once per test, and in CI every time,
    since `knowledge/` is not committed and so every test there starts from
    `absent`. Switched off here for the same reason and with the same force as
    the online phase above; the tests that mean to exercise it turn it back on
    and hand it fakes.
    """
    import serve
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", False)


def _no_daemon() -> list:
    raise OSError("tests do not ask the Ollama daemon what it has loaded")


# The corpus a developer's own console builds in this checkout.
_CHECKOUT_STORE = (Path(__file__).resolve().parent.parent / "knowledge").resolve()


@pytest.fixture(autouse=True)
def _no_developer_corpus(monkeypatch, tmp_path_factory):
    """No test searches the developer's corpus, or touches the cards it runs on.

    `knowledge/` is the default store, and any checkout that has run the
    console has one -- so every graph test's Researcher searched it for real:
    Chroma opened, the plan embedded through the Ollama daemon, and the
    embedder's `free_the_cards_for` unloaded whatever model the operator's own
    console had resident at that moment, a run in flight included. Results
    then depended on what that corpus held, which is the reason the Planner's
    project map is switched off below. CI has no `knowledge/`, so the suite
    already has to pass without one; this makes every checkout run it that way.

    Only *this checkout's* store is hidden: a request that resolves to it is
    answered with a directory that does not exist, so the reading door
    (`open_knowledge_base`) finds no corpus there. Every other store resolves
    as before -- the many tests that `chdir` into a temporary project and use a
    relative `knowledge/` there, its floor record and its lock included. And
    asking the daemon what it has loaded fails, which every caller already
    reads as "unknown" and which `free_the_cards_for` answers by evicting
    nothing. Tests that mean to exercise either patch them themselves.
    """
    import serve
    from langgraph_agent import config, graphrag_server

    hidden = tmp_path_factory.getbasetemp() / "no-developer-corpus"
    real = graphrag_server.resolve_persist_dir

    def resolve(persist_dir=None):
        path = real(persist_dir)
        return hidden if path.resolve() == _CHECKOUT_STORE else path

    # Both names: serve imports the function rather than the module.
    monkeypatch.setattr(graphrag_server, "resolve_persist_dir", resolve)
    monkeypatch.setattr(serve, "resolve_persist_dir", resolve)
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(config, "_ollama_ps", _no_daemon)
    monkeypatch.setattr(config, "unload_ollama_model", lambda model: False)


@pytest.fixture(autouse=True)
def _no_planner_project_map(monkeypatch):
    """No planning test searches the developer's corpus.

    `_make_plan` shows the Planner the files the corpus ranks closest to the
    goal. Left on, every test that plans would search `knowledge/` -- the real
    one on a developer's machine and nothing at all in CI -- so a planning test
    would pass or fail by what the checkout it ran in had indexed. The tests
    that exercise the map turn it back on and answer the search themselves.
    """
    monkeypatch.setattr(_nodes, "PLANNER_PROJECT_MAP", False)
