"""Self-healing where the application uses it: every seam it protects, end to end.

Each test fakes the far side -- the daemon's sockets, a search engine, git --
and checks the healing the application does on top: what is retried, what
opens a circuit, what is refused once it is open, and what the console and a
run's snapshot are told.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import httpx
import networkx as nx
import psycopg
import psycopg.errors
import pytest
from store_doubles import StoreDoubleMixin

import serve
from langgraph_agent import config, corpus_store, web_research
from langgraph_agent import graphrag_server as gs
from langgraph_agent.control import RUN_CONTROL
from langgraph_agent.mcp_client import MCPClient
from langgraph_agent.self_healing import (
    CircuitOpenError,
    circuit_states,
    get_healing_logger,
)

HEALING = get_healing_logger()


def _refused(*args: Any, **kwargs: Any) -> Any:
    raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


def _circuit(name: str) -> dict[str, Any]:
    return next(c for c in circuit_states() if c["name"] == name)


def _messages_since(mark: int) -> list[str]:
    return [e["message"] for e in HEALING.events(since=mark)]


def _mark() -> int:
    events = HEALING.events()
    return events[-1]["seq"] if events else 0


@pytest.fixture
def no_waits(monkeypatch):
    monkeypatch.setattr(config, "DAEMON_RETRY_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(gs, "OLLAMA_EMBED_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(web_research, "WEB_RETRY_WAIT_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# What counts as the daemon being unreachable, and as a provider being down
# ---------------------------------------------------------------------------


def _raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as cause:
            raise outer from cause
    except BaseException as exc:
        return exc


@pytest.mark.parametrize("exc, unreachable", [
    (urllib.error.URLError(ConnectionRefusedError()), True),
    (urllib.error.HTTPError("http://x", 500, "boom", None, io.BytesIO()), False),
    (ConnectionError("Failed to connect to Ollama"), True),
    (_raised_from(httpx.ConnectError("refused"), ConnectionRefusedError()), True),
    (TimeoutError("timed out reading the reply"), False),
    (ValueError("model not found"), False),
])
def test_only_an_unreachable_daemon_counts_as_one(exc, unreachable):
    assert config.daemon_unreachable(exc) is unreachable


class _APIStatusError(Exception):
    """An SDK error carrying the HTTP status, as Anthropic's does."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("exc, down", [
    (_APIStatusError(529), True),
    (_APIStatusError(503), True),
    (_APIStatusError(401), False),
    (type("APITimeoutError", (Exception,), {})(), True),
    (ConnectionError("unreachable"), True),
])
def test_only_an_outage_counts_against_a_cloud_provider(exc, down):
    assert config.provider_unavailable(exc) is down


@pytest.mark.parametrize(
    ("exc", "unreachable"),
    [
        (psycopg.OperationalError("connection refused"), True),  # no server said anything
        (psycopg.errors.ConnectionFailure(), True),
        (psycopg.errors.AdminShutdown(), True),
        (psycopg.errors.TooManyConnections(), True),
        # OperationalError too, but the server answering over a working connection
        (psycopg.errors.DeadlockDetected(), False),
        (psycopg.errors.SerializationFailure(), False),
        (psycopg.errors.QueryCanceled(), False),
        (psycopg.errors.LockNotAvailable(), False),
        (psycopg.errors.DiskFull(), False),
        (psycopg.errors.InvalidPassword(), False),
        (psycopg.errors.UniqueViolation(), False),
    ],
)
def test_only_a_connection_failure_counts_as_the_database_unreachable(exc, unreachable):
    """Every OperationalError used to: a deadlock between two consoles opened the circuit."""
    assert corpus_store.database_unreachable(exc) is unreachable
    wrapped = RuntimeError("store call failed")
    wrapped.__cause__ = exc
    assert corpus_store.database_unreachable(wrapped) is unreachable


def test_a_deadlock_is_worth_another_attempt():
    assert corpus_store.transaction_rolled_back(psycopg.errors.DeadlockDetected())
    assert corpus_store.transaction_rolled_back(psycopg.errors.SerializationFailure())
    assert not corpus_store.transaction_rolled_back(psycopg.errors.UniqueViolation())


def test_an_unreachable_database_counts_once_per_attempt(monkeypatch):
    """The connect was counted inside the attempt and again around it, so the
    circuit opened at its second failed attempt and refused the third."""
    monkeypatch.setattr(corpus_store, "POSTGRES_CONNECT_ATTEMPTS", 3)
    monkeypatch.setattr(corpus_store, "POSTGRES_RETRY_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(corpus_store, "POSTGRES_CONNECT_TIMEOUT_SECONDS", 1)
    database = corpus_store.CorpusDatabase("postgresql://postgres@127.0.0.1:1/nothing")

    with pytest.raises(psycopg.OperationalError):
        database.run(lambda conn: None, name="probe")

    assert _circuit("postgres")["failures"] == corpus_store.POSTGRES_CIRCUIT_THRESHOLD
    assert _circuit("postgres")["state"] == "open"


def test_the_extension_is_created_only_when_its_type_is_missing(monkeypatch):
    """It was created, and committed, on every connection the console opened."""
    executed: list[str] = []
    registered: list[int] = []

    class Connection:
        def execute(self, query: str, *args: Any) -> None:
            executed.append(query)

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

    monkeypatch.setattr(corpus_store, "register_vector", lambda conn: registered.append(1))
    corpus_store._configure(Connection())  # type: ignore[arg-type]
    assert (executed, registered) == ([], [1])

    def missing_then_there(conn: Any) -> None:
        registered.append(1)
        if len(registered) == 2:
            raise psycopg.ProgrammingError("vector type not found in the database")

    monkeypatch.setattr(corpus_store, "register_vector", missing_then_there)
    corpus_store._configure(Connection())  # type: ignore[arg-type]
    assert executed == ["CREATE EXTENSION IF NOT EXISTS vector"]
    assert len(registered) == 3


# ---------------------------------------------------------------------------
# The daemon's circuit
# ---------------------------------------------------------------------------


def test_a_dead_daemon_opens_its_circuit_and_is_then_not_asked(monkeypatch):
    calls = []

    def urlopen(request: Any, timeout: float | None = None) -> Any:
        calls.append(request.full_url)
        _refused()

    monkeypatch.setattr(config.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(config, "_ollama_tags_cache", (0.0, []))

    for _ in range(config.OLLAMA_CIRCUIT_THRESHOLD):
        assert config.list_ollama_models() == []
    assert _circuit("ollama-daemon")["state"] == "open"

    assert config.list_ollama_models() == []
    assert len(calls) == config.OLLAMA_CIRCUIT_THRESHOLD, "an open circuit still asked"


def test_a_daemon_with_nothing_pulled_is_not_called_unreachable(monkeypatch):
    """Both used to be an empty list, so every seat on a fresh daemon read
    OFFLINE "daemon unreachable" and sent the operator after the wrong thing."""
    monkeypatch.setattr(config, "_seat_failures", {})
    monkeypatch.setattr(config, "_ollama_tags_cache", (0.0, None))
    monkeypatch.setattr(config, "daemon_request", lambda path, payload=None, *, timeout: {})

    assert config.get_agent_status("planner")["badge"] == "NOT PULLED"

    monkeypatch.setattr(config, "_ollama_tags_cache", (0.0, None))

    def refused(path: str, payload: Any = None, *, timeout: float) -> Any:
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(config, "daemon_request", refused)
    assert config.get_agent_status("planner")["badge"] == "OFFLINE"


def test_a_daemon_answering_with_an_error_is_up(monkeypatch):
    def urlopen(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, io.BytesIO())

    monkeypatch.setattr(config.urllib.request, "urlopen", urlopen)
    for _ in range(config.OLLAMA_CIRCUIT_THRESHOLD + 2):
        assert config.ollama_model_capabilities(f"gone-{_}:latest") is None
    assert _circuit("ollama-daemon")["state"] == "closed"


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


class _Model:
    """A chat model whose first `failures` calls raise `error`."""

    def __init__(self, failures: int, error: Exception, model: str = "dolphin:latest") -> None:
        self.model = model
        self.failures = failures
        self.error = error
        self.calls = 0

    def invoke(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "answered"


def test_a_seat_rides_out_a_daemon_restart(no_waits):
    inner = _Model(2, _raised_from(httpx.ConnectError("refused"), ConnectionRefusedError()))
    seat = config._SeatLLM("planner", inner, provider="ollama")
    mark = _mark()

    assert seat.invoke(["prompt"]) == "answered"
    assert inner.calls == 3
    assert "planner" not in config._seat_failures
    assert any("Retry succeeded for 'seat:planner' on attempt 3" in m for m in _messages_since(mark))


def test_a_seat_failure_that_is_not_an_outage_is_not_retried(no_waits):
    inner = _Model(5, ValueError("model is not a chat model"))
    seat = config._SeatLLM("planner", inner, provider="ollama")

    with pytest.raises(ValueError):
        seat.invoke(["prompt"])
    assert inner.calls == 1
    assert config._seat_failures["planner"] == "model is not a chat model"


def test_the_emergency_stop_ends_a_seat_s_wait(monkeypatch):
    monkeypatch.setattr(config, "DAEMON_RETRY_WAIT_SECONDS", 30.0)
    RUN_CONTROL.arm("run-stop")
    RUN_CONTROL.stop("run-stop", "operator")
    inner = _Model(5, ConnectionError("refused"))

    with pytest.raises(ConnectionError):
        config._SeatLLM("architect", inner, provider="ollama").invoke(["prompt"])
    assert inner.calls == 1


def test_a_cloud_seat_stands_down_once_its_provider_is_down():
    outage = type("InternalServerError", (Exception,), {"status_code": 500})("overloaded")
    inner = _Model(99, outage, model="claude-opus-5")
    seat = config._SeatLLM("architect", inner, provider="anthropic")

    for _ in range(3):
        with pytest.raises(Exception, match="overloaded"):
            seat.invoke(["prompt"])
    with pytest.raises(CircuitOpenError):
        seat.invoke(["prompt"])
    assert inner.calls == 3
    assert "anthropic-api unreachable" in config._seat_failures["architect"]


def test_the_gpu_fallback_is_journalled_as_a_recovery():
    forced = _Model(1, RuntimeError("cudaMalloc failed: out of memory"))
    unforced = _Model(0, RuntimeError("unused"))
    seat = config._SeatLLM("builder", forced, lambda force_gpu: unforced, provider="ollama")
    mark = _mark()

    assert seat.invoke(["prompt"]) == "answered"
    assert any("Recovery action 'unforced reload' on 'seat:builder': SUCCESS" in m
               for m in _messages_since(mark))


# ---------------------------------------------------------------------------
# The embedder, and the rebuild it serves
# ---------------------------------------------------------------------------


def test_an_unreachable_embedder_opens_the_daemon_circuit(monkeypatch, no_waits):
    calls = []

    def urlopen(request: Any, timeout: float | None = None) -> Any:
        calls.append(1)
        _refused()

    monkeypatch.setattr(config.urllib.request, "urlopen", urlopen)
    embedder = gs.OllamaEmbedder(gs.EMBEDDING_MODEL_NAME)

    with pytest.raises(RuntimeError, match="could not embed"):
        embedder.encode("a passage")
    assert len(calls) == config.DAEMON_CONNECT_ATTEMPTS
    with pytest.raises(CircuitOpenError):
        embedder.encode("another passage")
    assert len(calls) == config.DAEMON_CONNECT_ATTEMPTS


class _EmptyStore(StoreDoubleMixin):
    """A store with nothing in it."""

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ids": [], "metadatas": []}

    def delete(self, *args: Any, **kwargs: Any) -> None:
        pass


class _IndexingKB:
    """What `index_corpus_files` touches; the daemon goes after one document."""

    def __init__(self) -> None:
        self.collection = _EmptyStore()
        self.graph = nx.DiGraph()
        self._should_stop = None
        self._lexical_index = None
        self.added: list[str] = []

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any], **kwargs: Any
    ) -> int:
        self.added.append(doc_id)
        if len(self.added) > 1:
            raise CircuitOpenError("ollama-daemon", 12)
        return 1

    def _save_graph(self) -> None:
        pass

    def _publish_graph(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    def stats(self) -> dict[str, Any]:
        return {}


def test_a_rebuild_stops_when_the_embedder_cannot_be_reached(tmp_path):
    uploads = tmp_path / gs.UPLOADS_DIR
    uploads.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (uploads / name).write_text(f"{name} text", encoding="utf-8")
    kb = _IndexingKB()

    report = gs.index_corpus_files(kb, str(tmp_path))  # type: ignore[arg-type]

    assert len(kb.added) == 2, "the third document was tried against an open circuit"
    assert report["indexed"] == 1
    assert "ollama-daemon is unavailable" in report["unavailable"]


class _PruneFailsStore(_EmptyStore):
    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    def prune_and_fingerprint(self, keep: Any) -> Any:
        raise self.failure


def test_a_rebuild_that_cannot_reach_the_store_embeds_nothing(tmp_path):
    """A failed prune left the rebuild believing the store held nothing, so it
    re-embedded the whole archive -- on the GPU, evicting the seat -- for
    transactions that then failed."""
    uploads = tmp_path / gs.UPLOADS_DIR
    uploads.mkdir()
    (uploads / "a.md").write_text("a.md text", encoding="utf-8")
    kb = _IndexingKB()
    kb.collection = _PruneFailsStore(psycopg.OperationalError("server closed the connection"))

    report = gs.index_corpus_files(kb, str(tmp_path))  # type: ignore[arg-type]

    assert kb.added == []
    assert report["unavailable_circuit"] == "postgres"
    assert "server closed the connection" in report["unavailable"]


def test_a_rebuild_that_cannot_read_the_store_for_another_reason_raises(tmp_path):
    uploads = tmp_path / gs.UPLOADS_DIR
    uploads.mkdir()
    (uploads / "a.md").write_text("a.md text", encoding="utf-8")
    kb = _IndexingKB()
    kb.collection = _PruneFailsStore(psycopg.errors.UndefinedTable("relation does not exist"))

    with pytest.raises(psycopg.errors.UndefinedTable):
        gs.index_corpus_files(kb, str(tmp_path))  # type: ignore[arg-type]
    assert kb.added == []


def test_an_unavailable_rebuild_is_its_own_outcome(monkeypatch):
    monkeypatch.setattr(serve, "iter_corpus_files", lambda: ["uploads/a.md"])
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", gs.EMBEDDING_MODEL_NAME))
    monkeypatch.setattr(serve, "_kb_for_indexing", lambda: None)
    monkeypatch.setattr(serve, "index_corpus_files", lambda kb, **kw: {
        "stopped": False, "unavailable": "ollama-daemon is unavailable", "indexed": 0,
        "embedded": 0, "reused": 0, "dropped": 0, "skipped": 0, "errors": [],
    })

    report = serve._rebuild_the_corpus(announce=lambda n: None, progress=lambda d, t: None,
                                       should_stop=lambda: False)

    assert report["source"] == "unavailable"
    line = serve._corpus_feed_line(report)
    assert line and "could not be reached" in line and "once the daemon answers" in line
    assert serve._last_rebuild["source"] == "unavailable"


def test_a_startup_rebuild_that_meets_a_dead_database_is_redone(monkeypatch):
    """It returned `error` before recording anything, so the monitor -- which
    redoes a rebuild recorded `unavailable` -- never heard of it."""
    monkeypatch.setattr(serve, "iter_corpus_files", lambda: ["uploads/a.md"])
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", gs.EMBEDDING_MODEL_NAME))
    monkeypatch.setattr(serve, "_last_rebuild", {})

    def no_database() -> Any:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(serve, "_kb_for_indexing", no_database)

    report = serve._rebuild_the_corpus(announce=lambda n: None, progress=lambda d, t: None,
                                       should_stop=lambda: False)

    assert report["source"] == "unavailable"
    assert serve._last_rebuild["unavailable_circuit"] == "postgres"
    corpus = serve._check_health()["corpus"]
    assert corpus["status"] == "unhealthy" and "database" in corpus["details"]


def test_a_rebuild_that_fails_otherwise_is_still_recorded(monkeypatch):
    monkeypatch.setattr(serve, "iter_corpus_files", lambda: ["uploads/a.md"])
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", gs.EMBEDDING_MODEL_NAME))
    monkeypatch.setattr(serve, "_last_rebuild", {})

    def broken() -> Any:
        raise ValueError("a bug")

    monkeypatch.setattr(serve, "_kb_for_indexing", broken)

    report = serve._rebuild_the_corpus(announce=lambda n: None, progress=lambda d, t: None,
                                       should_stop=lambda: False)

    assert report == {"source": "error", "corpus": "indexed", "note": "a bug"}
    assert serve._last_rebuild == report


def test_a_corpus_whose_database_is_down_reads_unavailable_not_empty(monkeypatch):
    class _DownStore(_EmptyStore):
        def count(self) -> int:
            raise psycopg.OperationalError("connection refused")

    kb = gs.GraphRAGKnowledgeBase.__new__(gs.GraphRAGKnowledgeBase)
    kb.collection = _DownStore()
    kb.graph = nx.DiGraph()
    kb.graph.add_node("uploads/a.md", type="document")
    monkeypatch.setattr(serve, "kb", kb)

    stats = serve.rpc_rag_stats({})

    assert stats["corpus"] == "unavailable"
    assert "connection refused" in stats["note"]
    assert stats["total_documents"] == 1
    assert stats["staleness"]["stale"] is False


class _UnloadableKB(_IndexingKB):
    """The model's load fails its whole schedule on the first document."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any], **kwargs: Any
    ) -> int:
        self.added.append(doc_id)
        raise gs.EmbedderLoadFailed(
            "Ollama could not embed with the model (after 5 attempts): cudaMalloc failed: "
            "out of memory"
        )


def test_a_rebuild_stops_when_the_model_will_not_load(tmp_path):
    """One schedule per rebuild, not one per document: the rest would be refused."""
    uploads = tmp_path / gs.UPLOADS_DIR
    uploads.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (uploads / name).write_text(f"{name} text", encoding="utf-8")
    kb = _UnloadableKB()

    report = gs.index_corpus_files(kb, str(tmp_path))  # type: ignore[arg-type]

    assert len(kb.added) == 1
    assert report["unavailable_circuit"] == "embedder-load"
    assert "a.md" in report["unavailable"] and "out of memory" in report["unavailable"]
    assert report["errors"] == [], "named once, as the reason, not again as a file error"


def test_a_model_that_will_not_load_is_not_called_unreachable():
    """The fix is a freed card, not a daemon, and the line must send the operator there."""
    line = serve._corpus_feed_line({
        "source": "unavailable", "unavailable_circuit": "embedder-load", "indexed": 0,
        "unavailable": "uploads/a.md: cudaMalloc failed: out of memory", "errors": [],
    })

    assert line is not None
    assert "could not be loaded onto the cards" in line
    assert "out of memory" in line and "nvidia-smi" in line
    assert "could not be reached" not in line


def test_an_unreachable_daemon_leaves_the_embedder_circuit_closed(monkeypatch, no_waits):
    """Nothing was loaded, so nothing was learnt about whether the model fits."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _refused)

    with pytest.raises(RuntimeError, match="could not embed"):
        gs.OllamaEmbedder(gs.EMBEDDING_MODEL_NAME).encode("a passage")

    assert not gs.EMBEDDER_LOAD.is_open


# ---------------------------------------------------------------------------
# Online research
# ---------------------------------------------------------------------------

_REAL_CLIENT = httpx.Client


def _serve_web(monkeypatch, handler) -> None:
    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(web_research, "WEB_SEARCH_ENABLED", True)


def test_a_search_backend_that_is_down_is_stood_down(monkeypatch, no_waits):
    searches = []

    def handler(request: httpx.Request) -> httpx.Response:
        searches.append(request)
        raise httpx.ConnectError("refused", request=request)

    _serve_web(monkeypatch, handler)
    answer = web_research.search_web("hybrid retrieval with bm25 rerank")

    assert answer["source"] == "error"
    assert len(searches) == 3, "the circuit should open on the third failure"
    assert any("web-search:duckduckgo is unavailable" in e for e in answer["errors"])

    again = web_research.search_web("hybrid retrieval with bm25 rerank")
    assert len(searches) == 3, "an open circuit still sent a search"
    assert any("not sent" in e for e in again["errors"])


def test_a_bot_check_stands_the_engine_down_for_later_runs_too(monkeypatch):
    searches = []

    def handler(request: httpx.Request) -> httpx.Response:
        searches.append(request)
        return httpx.Response(202, text='<div class="anomaly-modal__mask"></div>')

    _serve_web(monkeypatch, handler)
    web_research.search_web("first goal")
    web_research.search_web("second goal")

    assert len(searches) == 1
    assert _circuit("web-search:duckduckgo")["state"] == "open"


def test_a_page_that_hiccups_is_fetched_again(monkeypatch, no_waits):
    served = {"https://example.com/page": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(
                200, text='<a class="result__a" href="https://example.com/page">Page</a>'
            )
        served[str(request.url)] += 1
        if served[str(request.url)] == 1:
            return httpx.Response(503)
        return httpx.Response(
            200, headers={"content-type": "text/html"},
            text="<html><body><article><p>" + "hybrid retrieval " * 60 + "</p></article></body></html>",
        )

    _serve_web(monkeypatch, handler)
    answer = web_research.search_web("hybrid retrieval")

    assert [page["url"] for page in answer["pages"]] == ["https://example.com/page"]
    assert served["https://example.com/page"] == 2


# ---------------------------------------------------------------------------
# git push
# ---------------------------------------------------------------------------


def _vcs(outcomes: list[tuple[bool, str]], calls: list[tuple[str, ...]]):
    def run(
        self: MCPClient, *argv: str, timeout: float = 60.0, cwd: str | None = None
    ) -> tuple[bool, str]:
        calls.append(argv)
        if argv[:2] == ("git", "push"):
            return outcomes.pop(0)
        if argv[:3] == ("git", "rev-parse", "--abbrev-ref"):
            return True, "feature/x"
        if argv[:2] == ("git", "symbolic-ref"):
            return True, "origin/main"
        return True, ""

    return run


def test_a_push_that_never_reached_the_remote_is_sent_again(monkeypatch):
    calls: list[tuple[str, ...]] = []
    outcomes = [(False, "fatal: unable to access: Could not resolve host: github.com"),
                (True, "branch 'feature/x' set up to track 'origin/feature/x'")]
    monkeypatch.setattr(MCPClient, "_run_vcs", _vcs(outcomes, calls))
    monkeypatch.setattr("langgraph_agent.self_healing.decorators.time.sleep", lambda s: None)

    result = MCPClient().call_tool("git_dwell", {"stages": ["push"], "branch": "feature/x"})

    pushes = [argv for argv in calls if argv[:2] == ("git", "push")]
    assert len(pushes) == 2
    assert result["success"] is True, result


def test_a_rejected_push_is_not_sent_again(monkeypatch):
    """The remote answered: a hook or a protection rule said no, and asking
    again gets the same answer. (A push refused because the branch moved there
    is merged in and sent once more -- test_git_dwell.py, against a real remote.)"""
    calls: list[tuple[str, ...]] = []
    outcomes = [(False, "! [remote rejected] feature/x -> feature/x (pre-receive hook declined)")]
    monkeypatch.setattr(MCPClient, "_run_vcs", _vcs(outcomes, calls))

    result = MCPClient().call_tool("git_dwell", {"stages": ["push"], "branch": "feature/x"})

    assert len([argv for argv in calls if argv[:2] == ("git", "push")]) == 1
    assert result["success"] is False
    assert "rejected" in result["stages"][-1]["detail"]


# ---------------------------------------------------------------------------
# The console: the monitor, the RPCs, and a run's snapshot
# ---------------------------------------------------------------------------


class _Database:
    """The corpus database as the monitor asks it: one health answer."""

    def __init__(self) -> None:
        self.answer = {"status": "healthy", "details": "PostgreSQL 18.6, pgvector 0.8.7"}

    def health(self) -> dict[str, str]:
        return dict(self.answer)


@pytest.fixture
def database(monkeypatch):
    found = _Database()
    monkeypatch.setattr(serve, "get_database", lambda: found)
    return found


@pytest.fixture
def monitor(monkeypatch, database):
    """`_heal` against a daemon and a database that answer, and no search backend."""
    monkeypatch.setattr(serve, "daemon_request", lambda path, payload=None, timeout=0: {"version": "0.33"})
    monkeypatch.setattr(serve, "search_backend_health", lambda: None)
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "_health", {})
    monkeypatch.setattr(serve, "_last_rebuild", {})
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", gs.EMBEDDING_MODEL_NAME))
    rebuilds: list[str] = []

    def rebuild(when: str = "at startup") -> dict:
        rebuilds.append(when)
        serve._last_rebuild.clear()
        serve._last_rebuild.update(source="updated")
        return {"source": "updated", "embedded": 1, "dropped": 0, "reused": 0, "elapsed_s": 0.1}

    monkeypatch.setattr(serve, "_rebuild_the_corpus_in_background", rebuild)
    return rebuilds


def test_the_monitor_rebuilds_a_corpus_the_daemon_left_half_built(monitor):
    serve._last_rebuild.update(source="unavailable")
    mark = _mark()

    serve._heal()

    assert monitor == ["after the embedder came back"]
    messages = _messages_since(mark)
    assert any("Recovery action 'rebuild corpus' on 'corpus': SUCCESS" in m for m in messages)
    assert serve._health["ollama-daemon"]["status"] == "healthy"


def test_the_monitor_waits_for_the_daemon_before_rebuilding(monitor, monkeypatch):
    serve._last_rebuild.update(source="unavailable")
    monkeypatch.setattr(serve, "daemon_request", lambda *a, **k: _refused())

    serve._heal()

    assert monitor == []
    assert serve._health["ollama-daemon"]["status"] == "unhealthy"
    assert serve._health["corpus"]["status"] == "unhealthy"


def test_the_monitor_reports_the_database(monitor, database):
    serve._heal()
    assert serve._health["postgres"]["status"] == "healthy"

    database.answer = {"status": "unhealthy", "details": "connection refused"}
    serve._heal()
    assert serve._health["postgres"] == {"status": "unhealthy", "details": "connection refused"}


def test_the_monitor_waits_for_the_database_then_rebuilds(monitor, database):
    """A rebuild the database interrupted is finished once it answers again."""
    serve._last_rebuild.update(source="unavailable", unavailable_circuit="postgres")
    database.answer = {"status": "unhealthy", "details": "connection refused"}

    serve._heal()

    assert monitor == []
    assert "database could not be reached" in serve._health["corpus"]["details"]

    database.answer = {"status": "healthy", "details": "back"}
    serve._heal()

    assert monitor == ["after the database came back"]


def test_a_rebuild_stops_at_once_when_the_database_cannot_be_reached(tmp_path):
    """Nothing could be stored, so not one document is embedded for nothing."""
    uploads = tmp_path / gs.UPLOADS_DIR
    uploads.mkdir()
    (uploads / "a.md").write_text("a text", encoding="utf-8")
    kb = _IndexingKB()

    def refused(keep: Any) -> list[str]:
        raise CircuitOpenError("postgres", 15)

    kb.collection.prune_documents = refused  # type: ignore[method-assign]

    report = gs.index_corpus_files(kb, str(tmp_path))  # type: ignore[arg-type]

    assert kb.added == []
    assert report["unavailable_circuit"] == "postgres"
    assert report["indexed"] == 0


def test_the_monitor_waits_out_the_embedder_cooldown(monitor, monkeypatch):
    """Every pass would meet the open circuit as a refusal, and log a failed repair."""
    serve._last_rebuild.update(source="unavailable", unavailable_circuit="embedder-load")
    gs.EMBEDDER_LOAD.trip("the model did not fit the cards")

    serve._heal()

    assert monitor == []
    assert "would not load" in serve._health["corpus"]["details"]

    monkeypatch.setattr(gs.EMBEDDER_LOAD._breaker, "reset_timeout", 0)
    serve._heal()

    assert monitor == ["after the embedder came back"], "the rebuild is the trial"


def test_the_monitor_logs_a_change_of_health_once(monitor):
    mark = _mark()
    serve._heal()
    serve._heal()

    checks = [m for m in _messages_since(mark) if m.startswith("Health check 'ollama-daemon'")]
    assert len(checks) == 1


def test_the_console_reads_circuits_health_and_new_events(monitor):
    serve._heal()
    first = serve.rpc_healing({})
    assert {"circuits", "health", "events", "session", "interval_s"} <= set(first)
    assert any(c["name"] == "ollama-daemon" for c in first["circuits"])
    last = first["events"][-1]["seq"]

    HEALING.info("one more")
    assert [e["message"] for e in serve.rpc_healing({"since": last})["events"]] == ["one more"]
    assert "healing" in serve.QUIET_METHODS


def test_a_circuit_can_be_reset_from_the_console():
    config.OLLAMA_DAEMON.trip("test")
    reply = serve.rpc_reset_circuit({"name": "ollama-daemon"})
    assert reply["reset"] == ["ollama-daemon"]
    assert _circuit("ollama-daemon")["state"] == "closed"
    with pytest.raises(ValueError, match="no circuit called"):
        serve.rpc_reset_circuit({"name": "nothing-by-that-name"})


class _HealingGraph:
    """A one-node graph that has a seat recover on its way through."""

    def stream(self, state: dict[str, Any], config_: dict[str, Any]):
        HEALING.log_recovery_action("unforced reload", "seat:builder", True, "in the run")
        yield {"builder": {**state, "messages": ["[Builder] done"]}}


def test_a_run_is_one_healing_session_and_its_snapshot_carries_it(monkeypatch):
    monkeypatch.setattr(serve, "graph", _HealingGraph())

    result = serve.rpc_run_goal({"goal": "Write a module"})

    session = [e for e in result["healing"] if e["session_id"] == result["run_id"]]
    assert any("Healing session started" in e["message"] for e in session)
    assert any(e.get("action") == "recovery" for e in session)
    assert HEALING.session_id is None
    saved = json.loads(serve.LAST_RUN_PATH.read_text(encoding="utf-8"))
    assert saved["healing"] == json.loads(json.dumps(result["healing"], default=str))


# ---------------------------------------------------------------------------
# Pull requests a run left waiting on their checks
# ---------------------------------------------------------------------------

_PENDING = {
    "status": "pending", "number": 12, "head": "abc123", "branch": "agent/x", "cwd": "",
    "url": "https://github.com/acme/demo/pull/12", "pending": "waiting on test",
}


@pytest.fixture
def finishes(monkeypatch):
    """`finish_pull_request` answering from a list, and what it was asked."""
    asked: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []

    def finish(run: Any, **kwargs: Any) -> dict[str, Any]:
        asked.append(kwargs)
        return answers.pop(0)

    monkeypatch.setattr(serve, "finish_pull_request", finish)
    return asked, answers


def _due_now() -> None:
    """Every followed pull request due at once, as if its interval had passed."""
    with serve._pull_requests_lock:
        entries = serve._read_pull_requests()
        for entry in entries:
            entry["next_check"] = 0.0
        serve._write_pull_requests(entries)


def test_only_a_pending_pull_request_is_followed():
    for status in ("merged", "checks_failed", "open", "local", "failed"):
        serve._track_pull_request({**_PENDING, "status": status}, "run", "goal")
    assert serve.pull_requests_snapshot() == []

    serve._track_pull_request(_PENDING, "run", "ship it")
    [entry] = serve.pull_requests_snapshot()
    assert (entry["status"], entry["number"], entry["goal"]) == ("pending", 12, "ship it")


def test_the_monitor_merges_it_once_its_checks_pass(finishes):
    asked, answers = finishes
    serve._track_pull_request(_PENDING, "run", "ship it")
    answers.append({"success": True, "pending": "waiting on test"})

    serve._follow_pull_requests()
    serve._follow_pull_requests()  # not due again until its interval is up

    assert len(asked) == 1
    assert asked[0] == {"cwd": None, "branch": "agent/x", "number": 12, "head": "abc123"}
    assert serve.pull_requests_snapshot()[0]["status"] == "pending"

    _due_now()
    answers.append({"success": True, "merged": True, "merge_commit": "def4567890",
                    "default_branch": "main", "warnings": []})
    mark = _mark()
    serve._follow_pull_requests()

    assert serve.pull_requests_snapshot()[0]["status"] == "merged"
    assert any("Pull request #12 merged into main as def4567" in m for m in _messages_since(mark))


def test_red_checks_are_reported_and_never_fixed(finishes):
    """A fix is a run, and the operator starts it."""
    asked, answers = finishes
    serve._track_pull_request(_PENDING, "run", "ship it")
    answers.append({"success": False, "stopped_at": "checks", "checks_failed": ["CI / test"],
                    "error": "checks: CI / test failed"})
    mark = _mark()

    serve._follow_pull_requests()
    _due_now()
    serve._follow_pull_requests()

    [entry] = serve.pull_requests_snapshot()
    assert entry["status"] == "checks_failed"
    assert entry["checks_failed"] == ["CI / test"]
    assert len(asked) == 1, "a red pull request is not asked about again"
    assert any("its checks fail (CI / test)" in m for m in _messages_since(mark))


def test_it_stops_following_after_repeated_trouble(finishes):
    """gh signed out, GitHub down, someone else's push: asked a few times, then
    handed back to the operator rather than asked every minute for a day."""
    _, answers = finishes
    serve._track_pull_request(_PENDING, "run", "ship it")

    for attempt in range(serve.PULL_REQUEST_FOLLOW_FAILURES):
        assert serve.pull_requests_snapshot()[0]["status"] == "pending", attempt
        answers.append({"success": False, "stopped_at": "checks",
                        "error": "checks: gh is not signed in here"})
        _due_now()
        serve._follow_pull_requests()

    [entry] = serve.pull_requests_snapshot()
    assert entry["status"] == "stopped"
    assert "not signed in" in entry["detail"]


def test_a_pull_request_closed_without_merging_is_not_asked_about_again(finishes):
    asked, answers = finishes
    serve._track_pull_request(_PENDING, "run", "ship it")
    answers.append({"success": False, "stopped_at": "merge",
                    "error": "merge: #12 was closed without merging; reopen it on GitHub"})

    serve._follow_pull_requests()
    _due_now()
    serve._follow_pull_requests()

    assert serve.pull_requests_snapshot()[0]["status"] == "stopped"
    assert len(asked) == 1


def test_nothing_is_finished_while_a_run_is_in_flight(finishes, monkeypatch):
    """The run's Builder may be in the same repository, and cleanup switches branches."""
    asked, _ = finishes
    serve._track_pull_request(_PENDING, "run", "ship it")
    monkeypatch.setitem(serve._run_progress, "running", True)

    serve._follow_pull_requests()

    assert asked == []


def test_a_run_waits_while_a_pull_request_is_being_finished(monkeypatch):
    monkeypatch.setitem(serve._pull_request_follow, "running", True)
    monkeypatch.setitem(serve._pull_request_follow, "number", 12)

    with pytest.raises(ValueError, match="finishing pull request #12"):
        serve.rpc_run_goal({"goal": "ship it"})


def test_a_followed_pull_request_can_be_dismissed():
    serve._track_pull_request(_PENDING, "run", "ship it")
    key = serve.pull_requests_snapshot()[0]["key"]

    assert serve.rpc_dismiss_pull_request({"key": key})["pull_requests"] == []
    with pytest.raises(ValueError, match="No pull request"):
        serve.rpc_dismiss_pull_request({"key": key})


class _PendingGraph:
    """A one-node graph whose Builder left its pull request waiting on CI."""

    def stream(self, state: dict[str, Any], config_: dict[str, Any]):
        yield {"builder": {**state, "dwell": dict(_PENDING),
                           "messages": ["[Builder] pull request: #12 open, waiting"]}}


def test_a_run_hands_its_pending_pull_request_to_the_monitor(monkeypatch):
    monkeypatch.setattr(serve, "graph", _PendingGraph())

    result = serve.rpc_run_goal({"goal": "ship it"})

    assert result["dwell"]["status"] == "pending"
    [entry] = serve.pull_requests_snapshot()
    assert (entry["number"], entry["status"]) == (12, "pending")
    assert serve.rpc_status({})["pull_requests"] == serve.pull_requests_snapshot()
