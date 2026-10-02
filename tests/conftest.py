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

import os
import re
from pathlib import Path

import pytest

# The suite's own database, set before anything imports `config` (which loads
# `.env`, never over a variable already set): a test that indexes writes a
# schema, and the developer's corpus lives in the database `.env` names.
# `TEST_DATABASE_URL` names it outright; otherwise it is the configured server
# with the database swapped for `langgraph_agent_test`.
TEST_DATABASE_NAME = "langgraph_agent_test"


def _test_database_url() -> str:
    if os.getenv("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    base = os.getenv("DATABASE_URL") or "postgresql://postgres@127.0.0.1:5432/postgres"
    head, _, query = base.partition("?")
    head = re.sub(r"^(postgres(?:ql)?://[^/]*)(/[^/]*)?$", rf"\1/{TEST_DATABASE_NAME}", head)
    return head + (f"?{query}" if query else "")


def _ensure_test_database(url: str) -> str | None:
    """Create the suite's database if the server lacks it; None, or why it cannot."""
    import psycopg
    from psycopg import sql

    try:
        with psycopg.connect(url, connect_timeout=3):
            return None
    except psycopg.OperationalError as exc:
        if "does not exist" not in str(exc):
            return str(exc).strip()
    maintenance = re.sub(r"/[^/?]*(\?|$)", r"/postgres\1", url.split("://", 1)[1], count=1)
    try:
        with psycopg.connect(f"{url.split('://', 1)[0]}://{maintenance}", autocommit=True,
                             connect_timeout=3) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DATABASE_NAME)))
    except psycopg.Error as exc:
        return str(exc).strip()
    return None


os.environ["DATABASE_URL"] = _test_database_url()
_POSTGRES_PROBLEM = _ensure_test_database(os.environ["DATABASE_URL"])

import langgraph_agent.nodes as _nodes  # noqa: E402
import langgraph_agent.web_research as _web_research  # noqa: E402
from langgraph_agent.config import StubLLM  # noqa: E402
from langgraph_agent.control import RUN_CONTROL  # noqa: E402
from langgraph_agent.self_healing import reset_circuit  # noqa: E402

# Patch the LLM lookup used by agent nodes so every test gets deterministic,
# parser-friendly responses without making network calls. This has to be
# `get_agent_llm`: it is what the nodes import, and patching `get_llm` here
# only set an unused attribute on the module.
_nodes.get_agent_llm = lambda agent, temperature=0.1: StubLLM()


@pytest.fixture(scope="session", autouse=True)
def _drop_test_corpora():
    """Every schema a test indexed into is dropped when the session ends."""
    yield
    if _POSTGRES_PROBLEM is not None:
        return
    import psycopg
    from psycopg import sql

    from langgraph_agent.corpus_store import SCHEMA_PREFIX, close_databases

    close_databases()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        schemas = [row[0] for row in conn.execute(
            "SELECT nspname FROM pg_namespace WHERE starts_with(nspname, %s)", (SCHEMA_PREFIX,)
        )]
        for schema in schemas:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture(autouse=True)
def _no_waiting_for_an_absent_database(monkeypatch):
    """With no server at all, a store call fails at once rather than backing off.

    The retries ride out a database restarting under a running console; a
    suite run on a machine with none would spend seconds on every one.
    """
    if _POSTGRES_PROBLEM is not None:
        from langgraph_agent import corpus_store

        monkeypatch.setattr(corpus_store, "POSTGRES_CONNECT_ATTEMPTS", 1)


@pytest.fixture
def postgres():
    """The suite's database, for a test that stores a corpus for real.

    Skipped where no server answers, unless `REQUIRE_POSTGRES=1` (CI) makes
    that a failure: a corpus test that silently skipped would pass for nothing.
    """
    if _POSTGRES_PROBLEM is not None:
        message = f"PostgreSQL is not reachable at {os.environ['DATABASE_URL']}: {_POSTGRES_PROBLEM}"
        if os.getenv("REQUIRE_POSTGRES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return os.environ["DATABASE_URL"]


@pytest.fixture(autouse=True)
def _closed_circuits():
    """Every test starts with every circuit closed, whatever the last one tripped."""
    reset_circuit()
    yield
    reset_circuit()


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
    monkeypatch.setattr(serve, "REBUILD_CORPUS", False)


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
    corpus map is switched off below. CI has no `knowledge/`, so the suite
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
def _no_planner_corpus_map(monkeypatch):
    """No planning test searches the developer's corpus.

    `_make_plan` shows the Planner the files the corpus ranks closest to the
    goal. Left on, every test that plans would search `knowledge/` -- the real
    one on a developer's machine and nothing at all in CI -- so a planning test
    would pass or fail by what the checkout it ran in had indexed. The tests
    that exercise the map turn it back on and answer the search themselves.
    """
    monkeypatch.setattr(_nodes, "PLANNER_CORPUS_MAP", False)


@pytest.fixture
def whole_root_walk(monkeypatch):
    """Walk the whole root rather than only the archive directories.

    For the tests of the walk's *mechanics* -- pruning, vector reuse, staleness,
    the rebuild's phases -- which lay files out at the top of a scratch tree and
    do not care which directories the real corpus is confined to. Which
    directories that is has tests of its own (`test_corpus_roots.py`).
    """
    import langgraph_agent.graphrag_server as _graphrag

    monkeypatch.setattr(_graphrag, "CORPUS_ROOTS", ("",))
