"""The corpus store, against PostgreSQL: what the database itself guarantees.

Every other corpus test stands a fake in for the store, which can say what the
knowledge base asks of it but not whether the SQL does it. These ask the real
thing, in the suite's own database (`conftest.py`), one schema per test:
reading creates nothing, search is exact cosine, a document is replaced in one
transaction, a clear takes chunks, graph and floor together or not at all, a
corpus belongs to one model, and one rebuild holds the claim across sessions.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import psycopg
import psycopg.errors
import pytest
from test_chunking import _FakeEmbedder

from langgraph_agent import corpus_store, graphrag_server
from langgraph_agent.corpus_store import (
    PgCorpusStore,
    corpus_schema,
    create_corpus_store,
    open_corpus_store,
    rebuild_claim,
)
from langgraph_agent.graphrag_server import EMBEDDING_DIMENSIONS, GraphRAGKnowledgeBase

pytestmark = pytest.mark.usefixtures("postgres")

DIMS = 4


def _vector(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _schemas() -> set[str]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        return {row[0] for row in conn.execute("SELECT nspname FROM pg_namespace")}


@pytest.fixture
def store(tmp_path) -> PgCorpusStore:
    return create_corpus_store(tmp_path / "knowledge", embedding_model="m", dimensions=DIMS)


def _put(store: PgCorpusStore, doc_id: str, *vectors: np.ndarray, sha: str = "s") -> None:
    store.replace_document(
        doc_id,
        ids=[f"{doc_id}#{i:04d}" for i in range(len(vectors))],
        embeddings=list(vectors),
        documents=[f"{doc_id} passage {i}" for i in range(len(vectors))],
        metadatas=[
            {"doc_id": doc_id, "chunk_index": i, "chunk_count": len(vectors), "sha": sha}
            for i in range(len(vectors))
        ],
    )


# -- the two doors -----------------------------------------------------------


def test_reading_creates_nothing(tmp_path):
    before = _schemas()

    assert open_corpus_store(tmp_path / "knowledge", embedding_model="m", dimensions=DIMS) is None

    assert _schemas() == before


def test_the_creating_door_records_what_the_corpus_belongs_to(store, tmp_path):
    assert store.schema in _schemas()
    opened = open_corpus_store(tmp_path / "knowledge", embedding_model="m", dimensions=DIMS)
    assert opened is not None and opened.schema == store.schema
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(
            f'SELECT persist_dir, embedding_model, dimensions FROM "{store.schema}".corpus'
        ).fetchone()
    assert row == (str((tmp_path / "knowledge").resolve()), "m", DIMS)


def test_one_directory_is_one_schema_and_two_are_two(tmp_path):
    a = tmp_path / "knowledge"
    assert corpus_schema(a) == corpus_schema(tmp_path / "x" / ".." / "knowledge")
    assert corpus_schema(a) != corpus_schema(tmp_path / "other" / "knowledge")
    assert corpus_schema(a).startswith(corpus_store.SCHEMA_PREFIX + "knowledge_")


def test_a_corpus_of_another_model_is_no_corpus_and_is_rebuilt_empty(store, tmp_path):
    """Vectors from two models share no space: nothing in it answers this one."""
    _put(store, "a.md", _vector(1, 0, 0, 0))
    store.set_floor_record({"model": "m", "floor": 0.4})

    assert open_corpus_store(tmp_path / "knowledge", embedding_model="m2", dimensions=DIMS) is None

    rebuilt = create_corpus_store(tmp_path / "knowledge", embedding_model="m2", dimensions=8)
    assert rebuilt.count() == 0
    assert rebuilt.floor_record() is None
    _put(rebuilt, "b.md", np.ones(8, dtype=np.float32))  # the new width
    assert open_corpus_store(tmp_path / "knowledge", embedding_model="m", dimensions=DIMS) is None


# -- chunks ------------------------------------------------------------------


def test_search_is_exact_cosine_nearest_first(store):
    _put(store, "a.md", _vector(1, 0, 0, 0), _vector(0, 1, 0, 0))
    _put(store, "b.md", _vector(1, 1, 0, 0))

    hits = store.query([[1, 0, 0, 0]], n_results=3)

    assert hits["ids"] == [["a.md#0000", "b.md#0000", "a.md#0001"]]
    assert hits["distances"][0] == pytest.approx([0.0, 1 - 1 / np.sqrt(2), 1.0])
    assert hits["metadatas"][0][0]["doc_id"] == "a.md"


def test_a_replaced_document_leaves_no_old_tail(store):
    _put(store, "a.md", _vector(1, 0, 0, 0), _vector(0, 1, 0, 0), _vector(0, 0, 1, 0))

    _put(store, "a.md", _vector(0, 0, 0, 1), sha="t")

    assert store.get(include=["documents"])["ids"] == ["a.md#0000"]
    assert store.fingerprints() == {"a.md": "t"}


def test_a_replacement_that_fails_keeps_what_the_store_held(store):
    """The delete and the insert are one transaction: a bad vector undoes both."""
    _put(store, "a.md", _vector(1, 0, 0, 0), _vector(0, 1, 0, 0))

    with pytest.raises(psycopg.Error):
        _put(store, "a.md", _vector(1, 0, 0))  # the wrong width

    assert store.get(include=[])["ids"] == ["a.md#0000", "a.md#0001"]


def test_pruning_is_a_set_difference_reported_in_documents(store):
    _put(store, "a.md", _vector(1, 0, 0, 0), _vector(0, 1, 0, 0), sha="x")
    _put(store, "b.md", _vector(0, 0, 1, 0), sha="y")
    _put(store, "c.md", _vector(0, 0, 0, 1), sha="z")

    assert store.prune_documents({"b.md", "gone.md"}) == ["a.md", "c.md"]
    assert store.fingerprints() == {"b.md": "y"}
    assert store.count() == 1


def test_get_reads_every_row_or_the_ids_asked_for_in_document_order(store):
    _put(store, "b.md", _vector(0, 0, 1, 0))
    _put(store, "a.md", _vector(1, 0, 0, 0), _vector(0, 1, 0, 0))

    assert store.get(include=[])["ids"] == ["a.md#0000", "a.md#0001", "b.md#0000"]
    assert store.get(ids=["b.md#0000", "a.md#0001"], include=["documents"]) == {
        "ids": ["a.md#0001", "b.md#0000"],
        "documents": ["a.md passage 1", "b.md passage 0"],
    }


def test_no_index_is_kept_that_nothing_queries(store):
    """A GIN index on metadata served `delete(where=...)`, which nothing called."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (store.schema,)
            )
        }
    assert "chunks_metadata" not in indexes
    assert "chunks_doc_id" in indexes


# -- a database that restarts, and one that answers ---------------------------


def _postgres_circuit() -> dict[str, Any]:
    from langgraph_agent.self_healing import circuit_states

    return next(c for c in circuit_states() if c["name"] == "postgres")


def test_a_server_restart_costs_one_retry_not_the_circuit(store, monkeypatch):
    """Idle connections outlive the server's restart. The retry took the next
    stale one, so three of them opened the circuit against a healthy server."""
    monkeypatch.setattr(corpus_store, "POSTGRES_RETRY_WAIT_SECONDS", 0.0)
    database = store.database
    database.close()
    held = [database._take() for _ in range(4)]
    for conn in held:
        database._give_back(conn, broken=False)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        for conn in held:
            admin.execute("SELECT pg_terminate_backend(%s)", (conn.info.backend_pid,))

    assert store.count() == 0

    circuit = _postgres_circuit()
    assert (circuit["state"], circuit["failures"]) == ("closed", 0)
    assert all(conn.closed for conn in held)


def test_a_deadlock_is_retried_and_never_counts_toward_the_circuit(store, monkeypatch):
    """A deadlock is the server answering over a working connection: it used
    to count toward the circuit and get that connection closed as broken."""
    monkeypatch.setattr(corpus_store, "POSTGRES_RETRY_WAIT_SECONDS", 0.0)
    backends: list[int] = []
    failures_seen: list[int] = []

    def work(conn: Any) -> str:
        backends.append(conn.info.backend_pid)
        failures_seen.append(_postgres_circuit()["failures"])
        if len(backends) < 3:
            raise psycopg.errors.DeadlockDetected()
        return "done"

    assert store.database.run(work, name="deadlocked") == "done"
    assert failures_seen == [0, 0, 0]
    assert len(set(backends)) == 1, "a healthy connection was closed as broken"
    circuit = _postgres_circuit()
    assert (circuit["state"], circuit["failures"]) == ("closed", 0)


# -- the graph, the floor, and all of it at once -------------------------------


def test_a_document_s_edges_are_replaced_not_accumulated(store):
    store.save_document_graph("a.md", {"type": "document"}, {"Planner", "Architect"})
    store.save_document_graph("b.md", {"type": "document"}, {"Planner"})

    store.save_document_graph("a.md", {"type": "document", "path": "a.md"}, {"Builder"})

    graph = store.load_graph()
    assert set(graph.successors("a.md")) == {"Builder"}
    assert graph.nodes["a.md"]["path"] == "a.md"
    assert set(graph.predecessors("Planner")) == {"b.md"}
    assert graph.edges["a.md", "Builder"]["relation"] == "mentions"


def test_a_graph_round_trips_whole(store):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("a.md", type="document", content="x")
    graph.add_node("Planner", type="entity")
    graph.add_edge("a.md", "Planner", relation="mentions")

    store.save_graph(graph)
    loaded = store.load_graph()

    assert dict(loaded.nodes(data=True)) == dict(graph.nodes(data=True))
    assert list(loaded.edges(data=True)) == list(graph.edges(data=True))


def test_clear_takes_chunks_graph_and_floor_together(store):
    _put(store, "a.md", _vector(1, 0, 0, 0))
    store.save_document_graph("a.md", {"type": "document"}, {"Planner"})
    store.set_floor_record({"model": "m", "floor": 0.4})

    assert store.clear() == {"removed_chunks": 1, "removed_floor": True}

    assert store.count() == 0
    assert store.load_graph().number_of_nodes() == 0
    assert store.floor_record() is None


def test_a_clear_that_does_not_commit_took_nothing(store):
    """Inside a transaction that fails, the clear is undone with it."""
    _put(store, "a.md", _vector(1, 0, 0, 0))
    store.save_document_graph("a.md", {"type": "document"}, {"Planner"})
    store.set_floor_record({"model": "m", "floor": 0.4})

    with pytest.raises(RuntimeError), store.transaction():
        store.clear()
        raise RuntimeError("the process ended before the commit")

    assert store.count() == 1
    assert store.load_graph().number_of_edges() == 1
    assert store.floor_record() == {"model": "m", "floor": 0.4}


# -- the knowledge base on top of it -------------------------------------------


class _WideEmbedder(_FakeEmbedder):
    """The chunking fake, answering in the store's width."""

    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        if isinstance(text, list):
            return np.ones((len(text), EMBEDDING_DIMENSIONS), dtype=np.float32)
        return np.ones(EMBEDDING_DIMENSIONS, dtype=np.float32)


@pytest.fixture
def kb(tmp_path) -> GraphRAGKnowledgeBase:
    built = GraphRAGKnowledgeBase(str(tmp_path / "knowledge"))
    built._embedder = _WideEmbedder()  # type: ignore[assignment]
    return built


def test_a_document_and_its_graph_land_in_one_transaction(kb, monkeypatch):
    """A failure writing the graph takes the chunks back with it."""
    kb.add_document("a.md", "The Planner hands the Architect a plan.")
    assert kb.collection.count() == 1

    def graph_write_fails(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the graph could not be written")

    monkeypatch.setattr(kb.collection, "save_document_graph", graph_write_fails)
    with pytest.raises(RuntimeError):
        kb.add_document("b.md", "The Builder writes the files.")

    assert kb.collection.get(include=[])["ids"] == ["a.md#0000"]
    assert "b.md" not in kb.graph  # memory follows only a commit


def test_what_was_added_is_what_a_new_process_opens(kb):
    kb.add_document("a.md", "The Planner hands the Architect a plan.")

    reopened = GraphRAGKnowledgeBase(str(kb.persist_dir))

    assert set(reopened.graph.successors("a.md")) == {"Planner", "Architect"}
    assert reopened.collection.count() == 1


def test_the_corpus_state_follows_the_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert graphrag_server.corpus_state()[0] == "absent"

    built = GraphRAGKnowledgeBase()
    assert graphrag_server.corpus_state()[0] == "empty"

    built._embedder = _WideEmbedder()  # type: ignore[assignment]
    built.add_document("a.md", "The Planner plans.")
    assert graphrag_server.corpus_state()[0] == "indexed"


def test_a_database_that_cannot_be_reached_is_unavailable_not_absent(tmp_path, monkeypatch):
    """`absent` means nobody indexed here; a dead database says nothing of that."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:1/nothing")
    monkeypatch.setattr(corpus_store, "POSTGRES_CONNECT_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(corpus_store, "POSTGRES_CONNECT_ATTEMPTS", 1)

    assert graphrag_server.corpus_state(str(tmp_path / "knowledge"))[0] == "unavailable"
    assert graphrag_server.floor_calibration(str(tmp_path / "knowledge")) is None
    corpus_store.close_databases()


# -- one rebuild at a time ---------------------------------------------------


def test_the_rebuild_claim_is_exclusive_across_sessions_and_released(tmp_path):
    corpus = tmp_path / "knowledge"
    waited: list[bool] = []

    with rebuild_claim(corpus, 0, lambda: False, lambda: None) as first:
        with rebuild_claim(corpus, 0, lambda: False, lambda: waited.append(True)) as second:
            assert (first, second) == (True, False)
        with rebuild_claim(tmp_path / "other", 0, lambda: False, lambda: None) as elsewhere:
            assert elsewhere is True  # another corpus is another lock

    with rebuild_claim(corpus, 0, lambda: False, lambda: None) as again:
        assert again is True
    assert waited == [True]
