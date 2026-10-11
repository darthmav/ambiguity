"""Tests for clearing and exporting the corpus.

Two layers, both without the embedding model. Most of what is under test --
`clear`, `export_corpus`, the graph diagnostics -- touches only `self.graph`
and `self.collection`, so the knowledge base is built field by field around a
fake collection, and the RPC layer is called directly the way
`test_console_stop.py` calls it. What has to survive a restart is asked of a
real store in PostgreSQL (`pg_kb`): the claim there is about what the database
holds, which no fake can answer.
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pytest
from store_doubles import StoreDoubleMixin

import serve
from langgraph_agent.graphrag_server import (
    EMBEDDING_DIMENSIONS,
    UPLOADS_DIR,
    WEB_RESEARCH_DIR,
    GraphRAGKnowledgeBase,
    index_corpus_files,
    iter_corpus_files,
    relevance_floor,
)
from langgraph_agent.projects import set_project_embedded

# The walk's mechanics, laid out at the top of a scratch tree.
pytestmark = pytest.mark.usefixtures("whole_root_walk")


class _FakeCollection(StoreDoubleMixin):
    """The slice of the collection API these two methods use.

    `get(include=[])` returning ids is the idiom `index_corpus_files` already
    relies on, so the fake has to honour it: ids always come back, `documents`
    and `metadatas` only when asked for.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any]]] = {}
        self.fail_on_delete = False

    def add(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        self.rows[doc_id] = (content, metadata)

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        include = include or []
        ids = list(self.rows)
        out: dict[str, Any] = {"ids": ids}
        if "documents" in include:
            out["documents"] = [self.rows[i][0] for i in ids]
        if "metadatas" in include:
            out["metadatas"] = [self.rows[i][1] for i in ids]
        return out

    def delete(self, ids: list[str]) -> None:
        if self.fail_on_delete:
            raise RuntimeError("the store is unavailable")
        for doc_id in ids:
            self.rows.pop(doc_id, None)

    def count(self) -> int:
        return len(self.rows)


def _edges(node_link: dict[str, Any]) -> list[dict[str, Any]]:
    """The edge list out of `node_link_data`, whatever this networkx calls it.

    The default key moved from `links` to `edges` in networkx 3.4. Reading both
    keeps these tests honest across the range the project can be installed
    against, rather than passing only on the version that happens to be here.
    """
    return node_link.get("edges", node_link.get("links", []))


def _make_kb(tmp_path) -> GraphRAGKnowledgeBase:
    """A knowledge base with a real graph and a fake vector store."""
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    for path, content in (
        ("serve.py", "The Architect rules on the plan."),
        ("README.md", "Ambiguity console."),
    ):
        kb.collection.add(path, content, {"path": path, "type": "python"})
        kb.graph.add_node(path, type="document", content=content[:200], path=path)
        kb.graph.add_node("Architect", type="entity")
        kb.graph.add_edge(path, "Architect", relation="mentions")

    kb._save_graph()
    return kb


@pytest.fixture
def kb(tmp_path):
    return _make_kb(tmp_path)


def _make_pg_kb(tmp_path) -> GraphRAGKnowledgeBase:
    """The same corpus as `_make_kb`, stored for real."""
    kb = GraphRAGKnowledgeBase(str(tmp_path / "knowledge"))
    for path, content in (
        ("serve.py", "The Architect rules on the plan."),
        ("README.md", "Ambiguity console."),
    ):
        kb.collection.upsert(
            ids=[path],
            embeddings=[np.ones(EMBEDDING_DIMENSIONS)],
            documents=[content],
            metadatas=[{"path": path, "type": "python", "doc_id": path}],
        )
        kb.graph.add_node(path, type="document", content=content[:200], path=path)
        kb.graph.add_node("Architect", type="entity")
        kb.graph.add_edge(path, "Architect", relation="mentions")
    kb._save_graph()
    return kb


@pytest.fixture
def pg_kb(tmp_path, postgres):
    return _make_pg_kb(tmp_path)


def _stored_graph(kb: GraphRAGKnowledgeBase) -> nx.DiGraph:
    """The graph a process starting now would load: read from the store."""
    return GraphRAGKnowledgeBase(str(kb.persist_dir)).graph


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_both_halves_and_reports_what_went(kb):
    before = kb.stats()
    assert before["total_chunks"] and before["total_nodes"]

    result = kb.clear()

    assert result["removed_chunks"] == before["total_chunks"]
    assert result["removed_nodes"] == before["total_nodes"]
    assert result["removed_edges"] == before["total_edges"]
    assert result["total_chunks"] == 0
    assert result["total_nodes"] == 0
    assert result["total_edges"] == 0
    assert result["total_documents"] == 0


def test_clear_survives_a_restart(pg_kb):
    """The clear has to reach the store, not just memory.

    Otherwise the corpus comes back at the next process start, which reloads
    the graph from the store. `index_corpus_files` had exactly this hole --
    its `graph.clear()` was persisted only as a side effect of indexing
    something afterwards -- and it is pinned next door now that it does not.
    """
    assert _stored_graph(pg_kb).number_of_nodes() == 3

    pg_kb.clear()

    reloaded = _stored_graph(pg_kb)
    assert reloaded.number_of_nodes() == 0
    assert reloaded.number_of_edges() == 0
    assert pg_kb.collection.count() == 0


def test_a_reindex_that_matches_nothing_still_reaches_the_store(pg_kb, tmp_path):
    """The other half of `test_clear_survives_a_restart`, one door along.

    `index_corpus_files` prunes Chroma and clears the graph before it indexes
    anything, but both were persisted only as a side effect of `add_document`.
    A reindex matching no files therefore emptied the graph in memory, wrote
    nothing, and left the old graph for the next process start to reload --
    reporting `indexed: 0` and success while the corpus it claimed to have
    rebuilt sat in the store intact.

    A root with no indexable file in it is the honest way to reach that: the
    same path a walk takes after every match is excluded or deleted.
    """
    kb = pg_kb
    empty_root = tmp_path / "nothing"
    empty_root.mkdir()
    kb._lexical_index = object()  # stands in for one built before the prune

    report = index_corpus_files(kb, str(empty_root))

    assert report["indexed"] == 0
    assert report["dropped"] == 2
    assert kb.collection.count() == 0  # the prune ran
    assert kb.graph.number_of_nodes() == 0  # and the graph was cleared

    reloaded = _stored_graph(kb)
    assert reloaded.number_of_nodes() == 0
    assert reloaded.number_of_edges() == 0
    assert kb._lexical_index is None  # or it answers with the pruned rows


def test_a_reindex_that_indexes_something_persists_what_it_built(pg_kb, tmp_path):
    """The ordinary path must not have been broken to fix the empty one.

    The stand-in for `add_document` does what the real one does to the graph
    and to the index -- adds the document to the graph the rebuild is building,
    mints an entity, invalidates -- and only skips the embedder this suite
    avoids throughout. A stub that merely counted would leave this test passing
    on the new unconditional save alone, which is not the claim being made.
    """
    kb = pg_kb
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.md").write_text("The Planner interprets goals.", encoding="utf-8")

    def fake_add(
        path: str, content: str, metadata: dict[str, Any], *, graph: nx.DiGraph | None = None
    ) -> int:
        kb.collection.upsert(
            ids=[path], embeddings=[np.ones(EMBEDDING_DIMENSIONS)], documents=[content],
            metadatas=[{**metadata, "doc_id": path}],
        )
        target = kb.graph if graph is None else graph
        target.add_node(path, type="document", path=path)
        target.add_node("Planner", type="entity")
        target.add_edge(path, "Planner", relation="mentions")
        kb._lexical_index = None
        return 1

    kb.add_document = fake_add
    kb._lexical_index = object()

    report = index_corpus_files(kb, str(root))

    assert report["indexed"] == 1
    reloaded = _stored_graph(kb)
    assert set(reloaded.nodes) == {str(root / "notes.md"), "Planner"}
    assert reloaded.number_of_edges()  # the rebuild is stored, not just the emptying
    assert kb._lexical_index is None


def test_a_store_failure_leaves_the_graph_alone(kb):
    """Half a wipe is worse than none: the two halves have to agree."""
    kb.collection.fail_on_delete = True
    nodes_before = kb.graph.number_of_nodes()

    with pytest.raises(RuntimeError):
        kb.clear()

    assert kb.graph.number_of_nodes() == nodes_before
    assert kb.collection.count() == 2


def test_clear_retires_the_floor_measured_on_what_it_deleted(pg_kb):
    """The floor belongs to the corpus, so it cannot outlive one.

    A floor is a measurement of *these texts* under this embedding model, and
    two things read the record after the corpus is gone: the console's embedder
    card shows a floor over an empty corpus, and
    `_calibrate_the_floor_before_the_run` treats a record's mere presence as
    `known` -- so the run that rebuilds the corpus would never measure a floor
    for it, and every search would go on being judged against a number taken on
    a corpus nobody can consult any more.
    """
    pg_kb.collection.set_floor_record({"model": "qwen3-embedding:latest", "floor": 0.41})
    assert relevance_floor(pg_kb.persist_dir) == 0.41

    result = pg_kb.clear()

    assert result["removed_floor"] is True
    assert pg_kb.collection.floor_record() is None
    assert relevance_floor(pg_kb.persist_dir) is None


def test_clear_does_not_claim_a_floor_it_never_found(pg_kb):
    """A corpus whose floor was never measured reports no removal."""
    assert pg_kb.clear()["removed_floor"] is False


def test_a_store_failure_leaves_the_floor_alone(kb):
    """The third half fails with the other two, not without them.

    The store clears chunks, graph and floor in one transaction, so a wipe
    that raised leaves the corpus and the floor measured on it still agreeing.
    """
    kb.collection.set_floor_record({"model": "qwen3-embedding:latest", "floor": 0.41})
    kb.collection.fail_on_delete = True

    with pytest.raises(RuntimeError):
        kb.clear()

    assert kb.collection.floor_record() is not None


def test_clearing_an_empty_corpus_is_not_an_error(tmp_path):
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    result = kb.clear()

    assert result["removed_chunks"] == 0
    assert result["total_nodes"] == 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_carries_the_graph_and_every_chunk(kb):
    export = kb.export_corpus()

    exported_ids = {chunk["id"] for chunk in export["chunks"]}
    assert exported_ids == set(kb.collection.rows)

    graph_nodes = {node["id"] for node in export["graph"]["nodes"]}
    assert graph_nodes == set(kb.graph.nodes())
    assert len(_edges(export["graph"])) == kb.graph.number_of_edges()

    assert export["stats"] == kb.stats()
    assert export["errors"] == []

    chunk = next(c for c in export["chunks"] if c["id"] == "serve.py")
    assert chunk["content"] == "The Architect rules on the plan."
    assert chunk["metadata"]["path"] == "serve.py"


def test_export_omits_embeddings_and_says_so(kb):
    """Asserted structurally, on the keys.

    Grepping the serialised blob for the word looks equivalent and is not: the
    corpus is source files, and `graphrag_server.py` is itself a document in it,
    so the word appears in a chunk's *text* on any real corpus. That check
    passes here only because this fixture's text happens not to say it.
    """
    export = kb.export_corpus()

    assert "embeddings" not in export
    assert all("embeddings" not in chunk for chunk in export["chunks"])
    assert all("embeddings" not in node for node in export["graph"]["nodes"])

    assert "Embeddings are omitted" in export["note"]
    assert export["embedding_model"] == "qwen3-embedding:latest"


def test_export_is_json_serialisable(kb):
    """The RPC layer will do exactly this, and a failure there is a 500."""
    payload = json.dumps(kb.export_corpus())
    assert json.loads(payload)["stats"]["total_chunks"] == 2


def test_a_store_failure_still_exports_the_graph(kb):
    """The graph lives in memory; losing the database must not cost us both."""

    def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("the store is unavailable")

    kb.collection.get = boom  # type: ignore[method-assign]

    export = kb.export_corpus()

    assert export["chunks"] == []
    assert any("the store is unavailable" in err for err in export["errors"])
    assert export["graph"]["nodes"]


def test_export_of_an_empty_corpus_is_still_a_document(tmp_path):
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    export = kb.export_corpus()

    assert export["chunks"] == []
    assert export["graph"]["nodes"] == []
    assert export["exported_at"]


# ---------------------------------------------------------------------------
# the RPC layer
# ---------------------------------------------------------------------------


@pytest.fixture
def console_kb(kb, monkeypatch, tmp_path):
    """Point the console's `_kb()` at the fake, and start with no run armed.

    Run from an empty directory: a clear deletes the corpus's sources from the
    working directory, which must never be the checkout's.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serve, "kb", kb)
    with serve._run_lock:
        serve._run_progress.update(running=False, goal="", node="", run_id="")
    yield kb
    with serve._run_lock:
        serve._run_progress.update(running=False, run_id="")


def test_rpc_export_returns_a_serialisable_corpus(console_kb):
    result = serve.rpc_export_corpus({})
    assert json.loads(json.dumps(result))["stats"]["total_chunks"] == 2


def test_rpc_clear_empties_the_corpus(console_kb):
    result = serve.rpc_clear_corpus({})

    assert result["removed_chunks"] == 2
    assert console_kb.graph.number_of_nodes() == 0
    assert console_kb.collection.count() == 0


def test_rpc_clear_deletes_the_sources_a_restart_would_reembed(console_kb, tmp_path):
    """A clear has to outlive a restart, and the restart rebuilds from the walk.

    Emptying the store alone left the fetched pages and uploads on disk, and the
    next start embedded every one of them again. A project is opted out, never
    deleted: its directory is a run's work, not the archive.
    """
    (tmp_path / WEB_RESEARCH_DIR).mkdir(parents=True)
    (tmp_path / WEB_RESEARCH_DIR / "page.md").write_text("# fetched\n")
    (tmp_path / UPLOADS_DIR / "nested").mkdir(parents=True)
    (tmp_path / UPLOADS_DIR / "nested" / "notes.md").write_text("# mine\n")
    (tmp_path / "projects" / "surfer").mkdir(parents=True)
    (tmp_path / "projects" / "surfer" / "app.py").write_text("print(1)\n")
    set_project_embedded("surfer", True, tmp_path)

    result = serve.rpc_clear_corpus({})

    assert result["removed_files"] == 2
    assert result["projects_opted_out"] == 1
    assert iter_corpus_files(str(tmp_path)) == []
    assert (tmp_path / WEB_RESEARCH_DIR).is_dir()  # a volume's mount point stays
    assert (tmp_path / "projects" / "surfer" / "app.py").exists()


def test_rpc_clear_with_no_store_still_deletes_the_sources(console_kb, monkeypatch, tmp_path):
    """Files on disk with no store yet are a corpus the next start would build."""
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda: None)
    (tmp_path / UPLOADS_DIR).mkdir()
    (tmp_path / UPLOADS_DIR / "notes.md").write_text("# mine\n")

    result = serve.rpc_clear_corpus({})

    assert result["removed_files"] == 1
    assert not (tmp_path / UPLOADS_DIR / "notes.md").exists()


def test_rpc_clear_with_nothing_at_all_is_refused(console_kb, monkeypatch):
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda: None)

    with pytest.raises(ValueError, match="no corpus to clear"):
        serve.rpc_clear_corpus({})


def test_rpc_clear_is_refused_while_a_run_is_in_flight(console_kb):
    """An emptied corpus does not fail the Researcher's search, it just answers
    nothing -- which reads as `no_relevant_knowledge`. So the operator is the
    only one who can be told."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="build the thing")

    with pytest.raises(ValueError, match="run is in flight"):
        serve.rpc_clear_corpus({})

    assert console_kb.graph.number_of_nodes() == 3
    assert console_kb.collection.count() == 2


def test_there_is_no_way_to_ask_the_console_for_a_rebuild(console_kb):
    """The Reindex button is gone, and so is the method behind it.

    A rebuild mid-run was the same hazard as a clear and did not even need to
    finish to cause it -- the graph is cleared up front and documents re-added
    one at a time, so a search landing partway through is answered from a
    corpus that is neither the old one nor the new one, and is *answered*, not
    failed. That refusal is no longer needed because the request cannot be
    made: `_rebuild_the_corpus_before_the_run` rebuilds on every run, before
    the Architect opens and while nothing is searching.

    Pinned as an absence because an RPC method is one line to add back, and
    adding it back would restore both the hazard and the thing it existed for
    -- an operator who has to remember to press something.
    """
    assert "reindex" not in serve.RPC_METHODS
    assert not hasattr(serve, "rpc_reindex")


def test_the_refusal_names_the_run_it_is_protecting(console_kb):
    """A refusal that does not say what is running leaves the operator to guess
    whether they still care about it -- the same reason `rpc_shutdown` names it."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="port the console")

    with pytest.raises(ValueError, match="port the console"):
        serve.rpc_clear_corpus({})


def test_the_refusal_survives_a_run_with_no_goal(console_kb):
    """`goal` is empty until a run sets it; the guard must not trip on that."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="")

    with pytest.raises(ValueError, match="Stop the run first."):
        serve.rpc_clear_corpus({})


def test_export_is_allowed_while_a_run_is_in_flight(console_kb):
    """Reading the corpus takes nothing away from the run using it."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="build the thing")

    assert serve.rpc_export_corpus({})["stats"]["total_chunks"] == 2


def test_both_methods_are_registered_and_not_quiet():
    """These are operator actions; the telemetry feed has to show them."""
    assert serve.RPC_METHODS["export_corpus"] is serve.rpc_export_corpus
    assert serve.RPC_METHODS["clear_corpus"] is serve.rpc_clear_corpus
    assert "export_corpus" not in serve.QUIET_METHODS
    assert "clear_corpus" not in serve.QUIET_METHODS


# ---------------------------------------------------------------------------
# connectivity (the A1 health check)
# ---------------------------------------------------------------------------


def _bipartite_corpus(docs: int, ents: int, per_doc: int = 4) -> nx.DiGraph:
    """The shape `add_document` actually mints: document -> entity, directed."""
    G = nx.DiGraph()
    for d in range(docs):
        G.add_node(f"d{d}", type="document")
        for e in range(per_doc):
            entity = f"e{(d * per_doc + e) % ents}"
            G.add_node(entity, type="entity")
            G.add_edge(f"d{d}", entity, relation="mentions")
    return G


def test_connectivity_counts_components_networkx_style_not_spectrally(kb):
    """The zero-eigenvalue identity is for `L = D - A`, not the normalized one.

    On `I - D^-1/2 A D^-1/2` an isolated node has `D^-1/2 = 0`, so the `I` term
    leaves a bare 1 on its diagonal and it contributes eigenvalue **1, not 0**.
    A component count read off the normalized spectrum therefore misses every
    orphan, which is precisely the failure this check exists to catch.
    """
    kb.graph = _bipartite_corpus(6, 12)
    kb.graph.add_node("orphan_a", type="document")
    kb.graph.add_node("orphan_b", type="document")
    kb._connectivity_cache = None

    health = kb.connectivity()

    undirected = kb.graph.to_undirected(as_view=True)
    assert health["components"] == nx.number_connected_components(undirected)
    assert health["isolated_nodes"] == 2
    assert health["components"] > 1


def test_lambda_2_is_measured_on_the_largest_component(kb):
    """On the whole graph it is identically 0 once anything is orphaned.

    A health signal that reads 0.0 on every disconnected corpus -- and a real
    one is disconnected -- carries no information. The largest component's
    lambda_2 is the number that moves.
    """
    kb.graph = _bipartite_corpus(8, 16)
    kb._connectivity_cache = None
    connected = kb.connectivity()
    assert connected["lambda_2"] is not None and connected["lambda_2"] > 0

    kb.graph.add_node("orphan", type="document")
    kb._connectivity_cache = None
    orphaned = kb.connectivity()

    # The orphan is counted, and does not drag lambda_2 to zero.
    assert orphaned["components"] == connected["components"] + 1
    assert orphaned["lambda_2"] == pytest.approx(connected["lambda_2"], rel=1e-9)


def test_dropping_edges_moves_the_health_numbers(kb):
    """The point of the check: a regression the counters cannot see.

    An entity-extraction regression in `add_document` leaves the document count
    untouched and raises nothing. It fragments the graph, and that is visible
    here and nowhere else in `stats()`.
    """
    kb.graph = _bipartite_corpus(20, 10)
    kb._connectivity_cache = None
    before = kb.connectivity()

    kb.graph.remove_edges_from(list(kb.graph.edges())[::2])
    kb._connectivity_cache = None
    after = kb.connectivity()

    assert after["components"] > before["components"]


def test_connectivity_is_defined_on_an_empty_and_a_single_node_graph(kb):
    """lambda_2 is None, never 0.0: 0.0 is a reading, not an absence.

    A graph with nothing to measure and a graph on the point of splitting in
    two are opposite diagnoses, and returning 0.0 for both would report the
    empty corpus as the alarming one.
    """
    kb.graph = nx.DiGraph()
    kb._connectivity_cache = None
    empty = kb.connectivity()
    assert empty == {
        "components": 0,
        "largest_component": 0,
        "isolated_nodes": 0,
        "lambda_2": None,
        # Pages the research phase fetched are entity-free by design, so they
        # are excluded from the component counts and reported here instead --
        # see `test_web_entities.py`. Zero of them, on an empty graph.
        "web_documents": 0,
        # Markup, script and config files are entity-free too, counted apart.
        "entity_free_sources": 0,
    }

    kb.graph = nx.DiGraph()
    kb.graph.add_node("only", type="document")
    kb._connectivity_cache = None
    single = kb.connectivity()
    assert single["components"] == 1
    assert single["lambda_2"] is None
    assert "fewer than 2 nodes" in single["lambda_2_unavailable"]


def test_stats_caches_connectivity_against_the_graph_itself(kb):
    """`stats()` is polled every five seconds; the eigendecomposition is not free."""
    kb._publish_graph(_bipartite_corpus(10, 20))
    kb._connectivity_cache = None

    first = kb.stats()
    assert kb._connectivity_cache is not None
    cached_on, _ = kb._connectivity_cache
    assert cached_on is kb.graph

    # A second call on the same graph reuses it rather than recomputing.
    marker = {"components": -1, "largest_component": -1, "isolated_nodes": -1, "lambda_2": -1.0}
    kb._connectivity_cache = (cached_on, marker)
    assert kb.stats()["components"] == -1

    # A change publishes a new graph, so the stale entry cannot survive.
    kb._place_in_graph("fresh", {"type": "document"}, set())
    assert kb.stats()["components"] == first["components"] + 1


def test_a_change_that_keeps_both_counts_is_still_a_change(kb):
    """The cache was keyed on (nodes, edges): a document whose one entity was
    swapped for another kept both counts, and the old reading with them."""
    graph = nx.DiGraph()
    for doc, entity in (("d1.md", "Alpha"), ("d2.md", "Bravo")):
        graph.add_node(doc, type="document")
        graph.add_node(entity, type="entity")
        graph.add_edge(doc, entity, relation="mentions")
    kb._publish_graph(graph)
    kb._connectivity_cache = None
    before = kb.stats()

    kb._place_in_graph("d1.md", {"type": "document"}, {"Bravo"})

    after = kb.stats()
    assert (after["total_nodes"], after["total_edges"]) == (
        before["total_nodes"], before["total_edges"]
    )
    assert after["isolated_nodes"] == before["isolated_nodes"] + 1  # Alpha, left alone


def test_a_published_graph_is_never_changed_under_a_reader(kb):
    """The rebuild emptied and refilled the graph the console's threads iterate."""
    kb._publish_graph(_bipartite_corpus(4, 8))
    held = kb.graph
    nodes_before = held.number_of_nodes()

    kb._place_in_graph("late.md", {"type": "document"}, {"Newcomer"})
    kb.clear()

    assert held.number_of_nodes() == nodes_before
    assert kb.graph is not held


def test_stats_still_reports_counters_when_lambda_2_cannot_be_computed(kb, monkeypatch):
    """The health check is a diagnostic; it must not cost the console its header."""
    import spectral_graph

    def _explode(*args, **kwargs):
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(spectral_graph, "compute_spectrum", _explode)
    kb.graph = _bipartite_corpus(6, 12)
    kb._connectivity_cache = None

    stats = kb.stats()

    assert stats["total_nodes"] == kb.graph.number_of_nodes()
    assert stats["lambda_2"] is None
    # Named, not silently dropped: "could not measure" must never read as
    # "measured 0".
    assert "solver unavailable" in stats["lambda_2_unavailable"]


def test_stats_does_not_load_the_embedder_for_the_health_check(kb):
    """The header poll still touches nothing heavyweight."""
    kb.graph = _bipartite_corpus(6, 12)
    kb._connectivity_cache = None
    kb.stats()
    assert kb._embedder is None


# ---------------------------------------------------------------------------
# bottleneck (the A3 bridge detection)
# ---------------------------------------------------------------------------


def _topic_corpus(topics: int = 2, docs: int = 30, ents: int = 40,
                  bridges: int = 2, seed: int = 0) -> nx.DiGraph:
    """Topic areas joined only through a handful of shared entities.

    The shape A3 exists to analyse: within a topic, documents and entities are
    densely interlinked; between topics the only path runs through `BRIDGE*`.
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    for i in range(bridges):
        G.add_node(f"BRIDGE{i}", type="entity")
    for t in range(topics):
        for d in range(docs):
            G.add_node(f"d{t}_{d}", type="document")
            for e in rng.choice(ents, size=6, replace=False):
                G.add_node(f"e{t}_{e}", type="entity")
                G.add_edge(f"d{t}_{d}", f"e{t}_{e}", relation="mentions")
        for d in rng.choice(docs, size=2, replace=False):
            for i in range(bridges):
                G.add_edge(f"d{t}_{int(d)}", f"BRIDGE{i}", relation="mentions")
    return G


def test_bottleneck_finds_the_planted_bridge_entities(kb):
    """The whole claim of A3: it names the nodes the topics connect through.

    Degree cannot do this. A bridge entity mentioned by two documents has
    degree 2, which is unremarkable -- what marks it is that its edges are the
    ones crossing the cut.
    """
    kb.graph = _topic_corpus(topics=2)

    result = kb.bottleneck()

    assert result["verdict"] == "found"
    assert result["conductance"] <= result["threshold"]
    named = {node["id"] for node in result["bridge_nodes"]}
    assert {"BRIDGE0", "BRIDGE1"} <= named


def test_a_well_knit_corpus_is_certified_to_have_no_bottleneck(kb):
    """A minimisation always returns something; the verdict must not.

    Asking for the narrowest cut in a corpus with no waist yields a cut anyway,
    and reporting it as a bridge would be a fabricated finding. Cheeger's lower
    bound is what refuses it -- and it refuses on a proof about the whole
    graph, not on the cut that happened to be found.
    """
    kb.graph = _topic_corpus(topics=1, docs=90, ents=45, bridges=0)

    result = kb.bottleneck()

    assert result["verdict"] == "certified_none"
    assert result["cheeger_lower"] > result["threshold"]
    # The crossing nodes still exist. The verdict is what declines to call them
    # bridges, so a caller reading `verdict` cannot be misled by the list.
    assert result["total_bridge_nodes"] > 0


def test_tied_cuts_are_reported_when_there_are_more_than_two_areas(kb):
    """mu_2 ~= mu_3 means several equally-narrow cuts, picked between arbitrarily.

    Without this flag the same corpus returns a different split on different
    runs and the diagnostic reads as broken. It is also a finding: a tie says
    there are three or more topic areas, not two.
    """
    two = object.__new__(GraphRAGKnowledgeBase)
    two.collection = kb.collection
    two.graph = _topic_corpus(topics=2)
    assert two.bottleneck()["tied_cuts"] is False

    kb.graph = _topic_corpus(topics=3)
    assert kb.bottleneck()["tied_cuts"] is True


def test_bottleneck_is_stable_in_what_it_actually_reports(kb):
    """The sides may swap between tied cuts; the findings may not.

    ARPACK starts from a random residual, so on a tied spectrum the component
    it isolates varies. What a caller acts on -- how narrow the waist is, and
    which entities carry it -- has to be the same every time or the diagnostic
    is not usable.
    """
    kb.graph = _topic_corpus(topics=3)

    runs = [kb.bottleneck() for _ in range(5)]

    assert len({round(r["conductance"], 9) for r in runs}) == 1
    entities = {
        tuple(sorted(n["id"] for n in r["bridge_nodes"] if n["type"] == "entity"))
        for r in runs
    }
    assert len(entities) == 1


def test_bottleneck_runs_on_the_largest_component_not_the_whole_graph(kb):
    """An orphan is a conductance-0 cut, and would win every time.

    "Your corpus has an orphan" is what `connectivity()` reports. If the sweep
    ran on the whole graph the orphan would crowd out the real bridge on every
    disconnected corpus -- which is every real one.
    """
    kb.graph = _topic_corpus(topics=2)
    kb.graph.add_node("orphan", type="document")

    result = kb.bottleneck()

    assert result["verdict"] == "found"
    assert "orphan" not in {node["id"] for node in result["bridge_nodes"]}
    assert result["component_size"] == kb.graph.number_of_nodes() - 1


def test_bottleneck_on_a_graph_with_nothing_to_cut(kb):
    """No verdict is better than a cut invented for a graph that has none."""
    kb.graph = nx.DiGraph()
    assert kb.bottleneck()["verdict"] == "no_graph"

    kb.graph = nx.DiGraph()
    kb.graph.add_node("solo", type="document")
    single = kb.bottleneck()
    assert single["verdict"] == "no_graph"
    assert single["bridge_nodes"] == []
    assert single["conductance"] is None


def test_bottleneck_names_its_own_failure_rather_than_reporting_no_bridge(kb):
    """"Could not measure" must never read as "there is no bottleneck"."""
    import spectral_graph

    def _explode(*args, **kwargs):
        raise RuntimeError("solver unavailable")

    kb.graph = _topic_corpus(topics=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spectral_graph, "sweep_cut", _explode)
        result = kb.bottleneck()

    assert result["verdict"] == "unavailable"
    assert "solver unavailable" in result["note"]
    assert result["bridge_nodes"] == []


def test_rpc_bottleneck_is_read_only_and_needs_a_corpus(monkeypatch):
    """Registered, serialisable, and unguarded against a run -- like export."""
    assert serve.RPC_METHODS["bottleneck"] is serve.rpc_bottleneck
    # Not in the five-second poll: it is an eigenvector plus a sweep over every
    # edge, and it answers a question the operator asks.
    assert "bottleneck" not in serve.QUIET_METHODS

    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no corpus"):
        serve.rpc_bottleneck({})


def test_rpc_bottleneck_honours_limit_and_serialises(kb, monkeypatch):
    kb.graph = _topic_corpus(topics=2)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: kb)

    result = serve.rpc_bottleneck({"limit": 2})

    assert len(result["bridge_nodes"]) == 2
    json.dumps(result)  # the console has to be able to render it


# ---------------------------------------------------------------------------
# topics (the A2 spectral clustering)
# ---------------------------------------------------------------------------


def test_topics_recovers_the_planted_topic_areas(kb):
    """Documents land with the entities that define them, one cluster per topic.

    The bipartite shape is what makes the cluster readable: it mixes documents
    with their entities, so `top_entities` names the topic rather than listing
    ids.
    """
    kb.graph = _topic_corpus(topics=3, docs=30)

    result = kb.topics()

    assert result["verdict"] == "clustered"
    assert result["k"] == 3
    assert result["k_source"] == "eigengap"
    assert len(result["clusters"]) == 3
    for cluster in result["clusters"]:
        assert cluster["documents"] > 0 and cluster["entities"] > 0
        # A real community: leaving it, few edges cross.
        assert cluster["conductance"] < 0.1
        assert cluster["top_entities"]

    # Each planted topic ends up in its own cluster: the entity prefixes do not
    # mix. BRIDGE* is shared by construction and is allowed to land anywhere.
    for cluster in result["clusters"]:
        prefixes = {
            e.split("_")[0] for e in cluster["top_entities"] if not e.startswith("BRIDGE")
        }
        assert len(prefixes) <= 1


def test_a_corpus_with_no_topics_gets_no_map(kb):
    """The eigengap always returns some k; below decisiveness it is not believed.

    A map of a corpus that has no topics is worse than no map. The heuristic's
    failures are undecided rather than merely wrong -- the winning gap barely
    beats the runner-up -- and that is what the gate reads.
    """
    kb.graph = _topic_corpus(topics=1, docs=90, ents=45, bridges=0)

    result = kb.topics()

    assert result["verdict"] == "no_clear_structure"
    assert result["clusters"] == []
    assert result["decisiveness"] < result["threshold"]
    # The rejected suggestion is still reported, so the caller can see what was
    # turned down rather than only that something was.
    assert result["suggested_k"] >= 2


def test_an_explicit_k_overrides_the_gate_but_not_the_evidence(kb):
    """A caller asking for k has made the decision; the numbers still tell them.

    Conductance is the second, independent signal: it exposes a bad k even when
    the caller insisted, and it is why the clusters are never returned bare.
    """
    kb.graph = _topic_corpus(topics=1, docs=90, ents=45, bridges=0)

    result = kb.topics(k=3)

    assert result["verdict"] == "clustered"
    assert result["k_source"] == "requested"
    assert result["decisiveness"] < result["threshold"]
    # These are not communities, and the per-cluster conductance says so.
    assert max(c["conductance"] for c in result["clusters"]) > 0.2


def test_topics_is_stable_across_runs(kb):
    """ARPACK starts from a random residual; the partition must not follow it."""
    kb.graph = _topic_corpus(topics=3, docs=30)

    partitions = set()
    for _ in range(4):
        result = kb.topics(k=3)
        partitions.add(
            frozenset(
                frozenset(c["top_entities"]) for c in result["clusters"]
            )
        )
    assert len(partitions) == 1


def test_topics_runs_on_the_largest_component(kb):
    """Components are already clusters; rediscovering them wastes every eigenvector."""
    kb.graph = _topic_corpus(topics=3, docs=30)
    kb.graph.add_node("orphan", type="document")

    result = kb.topics()

    assert result["component_size"] == kb.graph.number_of_nodes() - 1
    assert result["k"] == 3
    clustered = {node for c in result["clusters"] for node in c["sample_documents"]}
    assert "orphan" not in clustered


def test_topics_rejects_an_impossible_k_as_an_error_not_a_verdict(kb):
    """`unavailable` means the measurement failed, not that the request was bad."""
    kb.graph = _topic_corpus(topics=2, docs=30)

    with pytest.raises(ValueError, match="k must be between"):
        kb.topics(k=1)
    with pytest.raises(ValueError, match="k must be between"):
        kb.topics(k=10**6)


def test_topics_on_a_graph_too_small_to_divide(kb):
    kb.graph = nx.DiGraph()
    assert kb.topics()["verdict"] == "no_graph"

    kb.graph = nx.DiGraph()
    kb.graph.add_edge("a", "b")
    assert kb.topics()["verdict"] == "no_graph"


def test_topics_names_its_own_failure(kb):
    import spectral_graph

    def _explode(*args, **kwargs):
        raise RuntimeError("solver unavailable")

    kb.graph = _topic_corpus(topics=2, docs=30)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spectral_graph, "spectral_clustering", _explode)
        result = kb.topics(k=2)

    assert result["verdict"] == "unavailable"
    assert "solver unavailable" in result["note"]
    assert result["clusters"] == []


def test_rpc_topics_is_registered_read_only_and_serialisable(kb, monkeypatch):
    assert serve.RPC_METHODS["topics"] is serve.rpc_topics
    assert "topics" not in serve.QUIET_METHODS

    kb.graph = _topic_corpus(topics=3, docs=30)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: kb)

    json.dumps(serve.rpc_topics({}))
    assert serve.rpc_topics({"k": 2})["k"] == 2
    # An omitted or blank k means "choose one", not "k = 0".
    assert serve.rpc_topics({"k": None})["k_source"] == "eigengap"
    assert serve.rpc_topics({"k": ""})["k_source"] == "eigengap"

    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: None)
    monkeypatch.setattr(serve, "kb", None)
    with pytest.raises(ValueError, match="no corpus"):
        serve.rpc_topics({})


# ---------------------------------------------------------------------------
# neighborhood split (A4) and duplicate entities (A5)
# ---------------------------------------------------------------------------


def test_neighborhood_is_unchanged_unless_the_split_is_asked_for(kb):
    """Off by default: it triples the cost of a call the console makes per click."""
    kb.graph = _topic_corpus(topics=2, docs=30)

    plain = kb.neighborhood("BRIDGE0", max_depth=3)

    assert "split" not in plain
    assert all("side" not in record for record in plain["related_nodes"])


def test_neighborhood_returns_the_centre_as_its_own_record(kb):
    """The console draws the centre from this record, typed as the graph types it.

    It used to get only `center_node`, a bare id, and assumed a document -- so
    every entity anyone traced was drawn as a file.
    """
    kb.graph = _topic_corpus(topics=2, docs=30)

    entity = kb.neighborhood("BRIDGE0", max_depth=1)
    document = kb.neighborhood("d0_1", max_depth=1)

    assert entity["center"]["id"] == "BRIDGE0"
    assert entity["center"]["node_type"] == "entity"
    assert document["center"]["node_type"] == "document"
    # Still kept out of its own neighbourhood, so nothing is drawn twice.
    assert all(record["id"] != "BRIDGE0" for record in entity["related_nodes"])


def test_the_sign_split_separates_the_topic_areas(kb):
    """One eigenvector, and it recovers the division the corpus was built with."""
    kb.graph = _topic_corpus(topics=2, docs=30)

    result = kb.neighborhood("BRIDGE0", max_depth=4, split=True)

    assert result["split"]["available"] is True
    sides = {r["id"]: r["side"] for r in result["related_nodes"] if r["side"]}
    # Each planted topic lands wholly on one side. BRIDGE* is shared by
    # construction and may fall either way.
    for prefix in ("d0_", "e0_", "d1_", "e1_"):
        members = {side for node, side in sides.items() if node.startswith(prefix)}
        if members:
            assert len(members) == 1, f"{prefix} split across both sides"
    assert {"center", "other"} == set(sides.values())


def test_the_split_is_oriented_on_the_centre_not_on_a_sign(kb):
    """An eigenvector and its negation are equally valid; a raw sign would swap.

    So the halves are named for the centre, and the same nodes come back as
    `center` on every run. Naming them this way also answers the question the
    caller actually asked -- what clusters with the node they looked up.
    """
    kb.graph = _topic_corpus(topics=2, docs=30)

    labelled = set()
    for _ in range(4):
        result = kb.neighborhood("BRIDGE0", max_depth=4, split=True)
        labelled.add(
            frozenset(r["id"] for r in result["related_nodes"] if r["side"] == "center")
        )
    assert len(labelled) == 1, "the centre's side is not stable between runs"


def test_the_split_reports_how_real_the_division_is(kb):
    """A sign cut always exists; `mu_2` is what says whether it means anything.

    A sweep that stays inside one topic area has no boundary to find and the
    split divides it arbitrarily -- which is honest, and visible in `mu_2`. A
    sweep that spans two topic areas has a genuine one. The caller gets the
    number rather than a verdict, because the line worth drawing depends on
    what the split is for.
    """
    kb.graph = _topic_corpus(topics=2, docs=30)

    within_topic = kb.neighborhood("d0_1", max_depth=3, split=True)["split"]
    across_topics = kb.neighborhood("BRIDGE0", max_depth=5, split=True)["split"]

    assert within_topic["mu_2"] > 10 * across_topics["mu_2"]


def test_a_pruned_neighbourhood_that_disconnects_reports_detached_nodes(kb):
    """`min_degree` can disconnect the sweep, and then a stray node is not a split.

    On a disconnected graph the Fiedler vector is a component indicator, so the
    "division" would be that orphan against everything else.
    """
    kb.graph = _topic_corpus(topics=2, docs=30)
    kb.graph.add_node("stray", type="entity")
    kb.graph.add_edge("d0_0", "stray", relation="mentions")

    result = kb.neighborhood("BRIDGE0", max_depth=6, min_degree=3, split=True)

    if result["split"]["available"]:
        detached = [r for r in result["related_nodes"] if r["side"] is None]
        assert len(detached) == result["split"]["detached"]


def test_the_split_declines_on_a_subgraph_too_small_to_divide(kb):
    kb.graph = nx.DiGraph()
    kb.graph.add_node("a", type="document")
    kb.graph.add_node("b", type="entity")
    kb.graph.add_edge("a", "b", relation="mentions")

    result = kb.neighborhood("a", split=True)

    assert result["split"]["available"] is False
    assert "too few" in result["split"]["note"].lower()


def _corpus_with_known_duplicates() -> nx.DiGraph:
    """Three seeded cases, one of which string similarity gets exactly wrong."""
    rng = np.random.default_rng(1)
    G = nx.DiGraph()
    docs = [f"doc{i}" for i in range(50)]
    for d in docs:
        G.add_node(d, type="document")
    for e in range(30):
        G.add_node(f"Ent{e}", type="entity")
    for d in docs:
        for e in rng.choice(30, size=5, replace=False):
            G.add_edge(d, f"Ent{e}", relation="mentions")

    mentions = {e: list(G.predecessors(e)) for e in [f"Ent{i}" for i in range(30)]}
    # (A) same name, same structure -- "Builder" / "Builders".
    G.add_node("Ent3s", type="entity")
    for d in mentions["Ent3"]:
        G.add_edge(d, "Ent3s", relation="mentions")
    # (B) different name, same structure -- the case names cannot reach.
    G.add_node("LanguageModel", type="entity")
    for d in mentions["Ent7"]:
        G.add_edge(d, "LanguageModel", relation="mentions")
    # (C) similar name, different structure -- a string-similarity false positive.
    G.add_node("Ent11x", type="entity")
    for d in rng.choice(docs, size=6, replace=False):
        G.add_edge(str(d), "Ent11x", relation="mentions")
    # (D) the same token, shouted -- a heading in one document against an entity
    # mentioned across many. This is the commonest real duplicate in this
    # project's own corpus (thirty of them) and the most asymmetric: `BUILDER`
    # appears in one document and `Builder` in twenty-nine.
    G.add_node("ENT5", type="entity")
    G.add_edge(mentions["Ent5"][0], "ENT5", relation="mentions")
    return G


def test_duplicate_entities_finds_a_plural_against_its_singular(kb):
    """`Ent3` / `Ent3s` -- the same name and the same documents."""
    kb.graph = _corpus_with_known_duplicates()

    result = kb.duplicate_entities()

    assert result["verdict"] == "scanned"
    found = {frozenset(pair["entities"]) for pair in result["pairs"]}
    assert frozenset({"Ent3", "Ent3s"}) in found


def test_a_case_variant_is_proposed_however_lopsided_it_is(kb):
    """`ENT5` is one heading in one document; `Ent5` is mentioned across many.

    The commonest real duplicate in this project's corpus is exactly this
    shape -- thirty of them, `BUILDER` against `Builder` -- and it is the one
    the old ranking could never reach, because a pair sharing one document out
    of twenty-nine is nowhere near any symmetric measure of closeness. A case
    variant is the same token, so it is certain on the name alone and is held
    to no structural floor; any evidence threshold would drop every one.
    """
    kb.graph = _corpus_with_known_duplicates()

    result = kb.duplicate_entities()

    pair = next(p for p in result["pairs"] if set(p["entities"]) == {"ENT5", "Ent5"})
    assert pair["kind"] == "case"
    assert pair["name_similarity"] == 1.0
    assert min(pair["degrees"]) == 1, "the shouted form appears once"
    # Certain pairs are offered before merely likely ones.
    assert result["pairs"][0]["kind"] == "case"


def test_two_names_for_one_thing_sharing_no_characters_are_not_found(kb):
    """A capability given up on purpose, pinned so it is not re-added by accident.

    `LanguageModel` shares every document with `Ent7` and no characters, and
    the old design found it -- on this fixture, whose entities are assigned to
    documents **at random**, which makes an identical neighbourhood
    astronomically improbable and therefore real evidence. A corpus is not
    random: entities of one topic are mentioned in the same documents, which is
    what a topic is. Measured on this project's own corpus, structure tightened
    as far as it goes -- Jaccard 1.00 with at least three shared documents --
    returns `Oppenheim` / `Schafer`, `Nyquist` / `Frequency` and
    `BUILDER_DEADLINE_SECONDS` / `NODE_DEADLINE_SECONDS`. All collocations, no
    duplicates, 0% precision. So this pair is not found, and a structural
    generator should not be reinstated without re-running that measurement.
    """
    kb.graph = _corpus_with_known_duplicates()

    result = kb.duplicate_entities()

    assert not any(
        set(p["entities"]) == {"Ent7", "LanguageModel"} for p in result["pairs"]
    )


def test_containment_is_reported_and_jaccard_disagrees_with_it(kb):
    """The asymmetry is the finding, so both numbers are carried.

    A real duplicate is a subset, not a match: on the live corpus `Builder`
    has 29 documents to `Builders`' 4, which reads containment 1.00 and
    Jaccard 0.14. Ranking on the symmetric number is what buried every true
    duplicate under thousands of unrelated pairs.
    """
    kb.graph = _corpus_with_known_duplicates()

    result = kb.duplicate_entities()

    pair = next(p for p in result["pairs"] if set(p["entities"]) == {"ENT5", "Ent5"})
    assert pair["containment"] == pytest.approx(1.0)
    assert pair["neighbourhood_overlap"] < 0.2, "Jaccard punishes the asymmetry"


def test_duplicate_entities_excludes_the_string_similarity_false_positive(kb):
    """Similar names, different neighbourhoods. Names rank it first; structure does not."""
    kb.graph = _corpus_with_known_duplicates()

    result = kb.duplicate_entities()

    assert not any(
        set(pair["entities"]) == {"Ent11", "Ent11x"} for pair in result["pairs"]
    )


def test_duplicate_entities_only_proposes(kb):
    """It never merges. The graph is left exactly as it was found."""
    kb.graph = _corpus_with_known_duplicates()
    before = (kb.graph.number_of_nodes(), kb.graph.number_of_edges())

    kb.duplicate_entities()

    assert (kb.graph.number_of_nodes(), kb.graph.number_of_edges()) == before


def test_duplicate_entities_on_a_corpus_too_small_to_compare(kb):
    kb.graph = nx.DiGraph()
    assert kb.duplicate_entities()["verdict"] == "no_graph"

    kb.graph = nx.DiGraph()
    kb.graph.add_node("d", type="document")
    kb.graph.add_node("a", type="entity")
    kb.graph.add_edge("d", "a", relation="mentions")
    assert kb.duplicate_entities()["verdict"] == "no_graph"


def test_rpc_surface_for_the_split_and_the_duplicate_scan(kb, monkeypatch):
    assert serve.RPC_METHODS["duplicate_entities"] is serve.rpc_duplicate_entities
    assert "duplicate_entities" not in serve.QUIET_METHODS

    kb.graph = _corpus_with_known_duplicates()
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: kb)

    json.dumps(serve.rpc_duplicate_entities({}))
    assert len(serve.rpc_duplicate_entities({"limit": 1})["pairs"]) == 1
    # A blank threshold means "use the default", not "threshold 0".
    assert serve.rpc_duplicate_entities({"name_similarity": ""})["name_similarity"] > 0
    assert serve.rpc_duplicate_entities({"containment": ""})["containment"] > 0
    assert serve.rpc_duplicate_entities(
        {"name_similarity": 0.7})["name_similarity"] == 0.7
    assert serve.rpc_duplicate_entities({"containment": 0.9})["containment"] == 0.9

    kb.graph = _topic_corpus(topics=2, docs=30)
    assert "split" not in serve.rpc_query_graph({"node_id": "BRIDGE0"})
    split = serve.rpc_query_graph({"node_id": "BRIDGE0", "max_depth": 4,
                                  "min_degree": 1, "split": True})
    assert split["split"]["available"] is True
    json.dumps(split)

    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: None)
    monkeypatch.setattr(serve, "kb", None)
    with pytest.raises(ValueError, match="no corpus"):
        serve.rpc_duplicate_entities({})


# --------------------------------------------------------------------------
# What a reindex is allowed to see
# --------------------------------------------------------------------------

def test_nothing_but_a_run_and_an_upload_builds_a_corpus():
    """Two acts index, and the guard is that no third one exists.

    There were four: an install step, two indexing scripts under `scripts/`,
    and the setup and verification scripts calling them. Each looked like
    housekeeping rather than a decision, and each
    decided how fresh the corpus was on a machine nobody had run anything on --
    a question the first run answers correctly by itself. They are gone, and
    what is left is a rule worth a build failure: the creating door opens for a
    run, and for a document being embedded into the corpus.

    Recomputed from the tree rather than listed, so a script added next month
    is caught by the same test. `iter_corpus_files` is fine anywhere: reading
    the walk is not indexing.
    """
    from pathlib import Path

    builders = ("get_knowledge_base", "index_corpus_files", "GraphRAGKnowledgeBase(")
    offenders = []
    for path in [*Path("scripts").glob("*.py"), Path("install.sh"),
                 Path("launch_console.sh"), Path("example_usage.py")]:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        # A comment saying the script does not index is not the script indexing.
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("#", "//"))
        )
        hits = [name for name in builders if name in code]
        if hits:
            offenders.append(f"{path}: {', '.join(hits)}")

    assert not offenders, (
        "these build a corpus outside a run and outside an upload, which is the "
        "third writer this rule exists to refuse:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# the overview a sweep draws
# ---------------------------------------------------------------------------


def test_the_overview_is_the_union_of_every_document_s_neighbourhood(kb):
    """One call, and the same graph the console's per-document sweep assembled."""
    kb.graph = _topic_corpus(topics=2, docs=30)
    kb.graph.add_node("lonely.md", type="document")

    union_nodes: set[str] = set()
    union_edges: set[tuple[str, str]] = set()
    for node, attrs in kb.graph.nodes(data=True):
        if attrs.get("type") != "document":
            continue
        union_nodes.add(node)
        hood = kb.neighborhood(node, max_depth=2, min_degree=4)
        union_nodes |= {record["id"] for record in hood["related_nodes"]}
        union_edges |= {(edge["source_id"], edge["target_id"]) for edge in hood["edges"]}

    full = kb.overview(min_degree=4, include_isolated=True)

    assert {record["id"] for record in full["nodes"]} == union_nodes
    assert {(edge["source_id"], edge["target_id"]) for edge in full["edges"]} == union_edges


def test_the_overview_leaves_out_documents_linked_to_nothing_and_counts_them(kb):
    kb.graph = _topic_corpus(topics=2, docs=30)
    kb.graph.add_node("lonely.md", type="document")

    shown = kb.overview(min_degree=1)

    assert "lonely.md" not in {record["id"] for record in shown["nodes"]}
    assert shown["unlinked_documents"] == 1
    assert shown["total_nodes"] == len(shown["nodes"])


def test_rpc_graph_overview_is_registered_and_answers_without_a_corpus(kb, monkeypatch, postgres):
    assert serve.RPC_METHODS["graph_overview"] is serve.rpc_graph_overview

    kb.graph = _topic_corpus(topics=2, docs=30)
    monkeypatch.setattr(serve, "kb", kb)
    result = serve.rpc_graph_overview({"min_degree": 4})
    json.dumps(result)
    assert result["total_nodes"] > 0

    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: None)
    absent = serve.rpc_graph_overview({})
    assert absent["corpus"] == "absent" and absent["nodes"] == []


def test_a_save_that_dies_midway_leaves_the_graph_it_replaced(pg_kb):
    """A graph save truncates before it refills; dying in between must cost nothing.

    The process can end in between -- the console's exit and install.sh's
    start-up check both end it under a rebuild running on a daemon thread --
    and a graph cut off mid-write would be what every later start loaded. The
    save is one transaction, so a failure partway through rolls the truncate
    back with it. A node the store cannot serialise is the failure here: it
    is met after the truncate and after some rows were already copied.
    """
    before = _stored_graph(pg_kb)
    pg_kb.graph.add_node("Planner", type="entity", unserialisable=object())

    with pytest.raises(TypeError):
        pg_kb._save_graph()

    after = _stored_graph(pg_kb)
    assert set(after.nodes) == set(before.nodes)
    assert after.number_of_edges() == before.number_of_edges()
    assert "serve.py" in after
