"""Tests for traversing the knowledge graph.

The graph is a `DiGraph` whose every edge runs **document -> entity**, which
makes direction a trap rather than a detail: an entity has in-edges only, so a
directed walk starting from one reaches nothing at all. Both traversals here
have to take the undirected view, and the failure when they do not is silent —
`query_graph` returned the entity by itself for every entity in the corpus,
which reads as "this term connects to nothing" and not as a broken walk. It is
the Researcher's `query_knowledge_graph` tool, so for as long as it stood, one
half of retrieval answered every relationship question with silence.

Built field by field around a fake collection, the way `test_corpus_admin.py`
does, so the embedding model never loads.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase

DOCUMENTS = ("serve.py", "README.md", "CLAUDE.md")


class _FakeCollection:
    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        return {"ids": list(DOCUMENTS)}


@pytest.fixture
def kb(tmp_path) -> GraphRAGKnowledgeBase:
    """Three documents, all mentioning one entity. Edges document -> entity."""
    built = object.__new__(GraphRAGKnowledgeBase)
    built.persist_dir = tmp_path
    built.collection = _FakeCollection()
    built.graph = nx.DiGraph()
    built.graph.add_node("Architect", type="entity")
    for path in DOCUMENTS:
        built.graph.add_node(path, type="document", path=path)
        built.graph.add_edge(path, "Architect", relation="mentions")
    return built


def test_an_entity_finds_the_documents_that_mention_it(kb):
    """The regression: every edge points *at* an entity, never away from one.

    Traversed directed, an entity has out-degree 0 and this returns a
    neighbour list of length one — itself — however many documents mention it.
    """
    assert kb.graph.out_degree("Architect") == 0
    assert kb.graph.degree("Architect") == len(DOCUMENTS)

    result = kb.query_graph("Architect", hops=1)

    assert result["subgraph_nodes"] == len(DOCUMENTS) + 1
    found = {name for name, _ in result["neighbors"]}
    assert found == {"Architect", *DOCUMENTS}


def test_a_document_still_finds_its_entities(kb):
    """The direction that always worked must keep working."""
    result = kb.query_graph("serve.py", hops=1)

    assert "Architect" in {name for name, _ in result["neighbors"]}


def test_hops_still_bounds_the_walk(kb):
    """Undirected traversal must not mean unbounded traversal.

    At one hop from a document only its own entities are reachable; the sibling
    documents sharing that entity are two hops away.
    """
    one = kb.query_graph("serve.py", hops=1)
    two = kb.query_graph("serve.py", hops=2)

    assert one["subgraph_nodes"] == 2
    assert two["subgraph_nodes"] == len(DOCUMENTS) + 1


def test_an_unknown_entity_is_reported_rather_than_guessed(kb):
    assert "error" in kb.query_graph("no-such-entity-xyzzy", hops=1)


def test_resolution_still_accepts_a_loose_id(kb):
    """The console lets a user paste a fragment; both entry points accept it."""
    assert kb.query_graph("architect", hops=1)["entity"] == "Architect"


# ---------------------------------------------------------------------------
# Asking nothing must not be answered as though something was asked
# ---------------------------------------------------------------------------


def test_an_empty_query_returns_nothing_rather_than_arbitrary_passages(kb):
    """The empty string still embeds, and what it matches is not an answer.

    Measured on the real corpus before this guard: `search("")` returned five
    chunks with a top score of 0.412 — *above* the relevance floor of the time, so
    `_gather_research` would have formatted them as findings and announced
    "Research complete" over passages selected by nothing at all. That is the
    fabricated retrieval hit this module was fixed for once already, arriving
    through the query rather than through the corpus.

    Returning early also means the embedding model is never loaded, which is
    why this can be asserted against a knowledge base with no collection.
    """
    assert kb.search("", 5) == []
    assert kb.search("   \n\t ", 5) == []


def _resolved(kb, node_id: str) -> str | None:
    return kb._match_node(node_id)[0]


def test_a_blank_node_id_resolves_to_nothing(kb):
    """`"" in anything` is True, so a blank id matched on the first comparison.

    It resolved to whichever node the graph enumerated first and handed back a
    real document's neighbourhood — a hit, to a caller who asked about nothing.
    """
    assert _resolved(kb, "") is None
    assert _resolved(kb, "   ") is None
    assert "error" in kb.query_graph("", hops=1)


def test_a_loose_but_real_id_still_resolves(kb):
    """The guard must not cost the fuzzy match the console depends on."""
    assert _resolved(kb, "architect") == "Architect"


# ---------------------------------------------------------------------------
# resolving a loose id
# ---------------------------------------------------------------------------


def test_the_exact_name_beats_a_longer_one_that_contains_it(kb):
    """`plan` is the entity `Plan`, not `Planner`, however the graph enumerates."""
    for order in (("Planner", "Plan"), ("Plan", "Planner")):
        kb.graph.remove_nodes_from(["Plan", "Planner"])
        for node in order:
            kb.graph.add_node(node, type="entity")
        assert _resolved(kb, "plan") == "Plan"


def test_a_document_resolves_by_its_file_name(kb):
    kb.graph.add_node("src/pkg/nodes.py", type="document")
    kb.graph.add_node("NodesThing", type="entity")

    assert _resolved(kb, "nodes.py") == "src/pkg/nodes.py"
    assert _resolved(kb, "nodes") == "src/pkg/nodes.py"


def test_a_substring_match_is_the_shortest_and_the_same_every_time(kb):
    """First in enumeration order made the answer depend on insertion order."""
    long_name = "tests/test_planner_routing.py"
    for order in ((long_name, "Planner"), ("Planner", long_name)):
        kb.graph.remove_nodes_from(order)
        for node in order:
            kb.graph.add_node(node, type="document" if node.endswith(".py") else "entity")
        assert _resolved(kb, "lann") == "Planner"


def test_a_loose_trace_says_what_it_matched_and_what_else_it_could_be(kb):
    for node in ("Planner", "PlannerNode"):
        kb.graph.add_node(node, type="entity")
        kb.graph.add_edge("serve.py", node, relation="mentions")

    hood = kb.neighborhood("plann", max_depth=1, min_degree=1)

    assert hood["center_node"] == "Planner"
    assert hood["resolved_from"] == "plann"
    assert "PlannerNode" in hood["alternatives"]
    assert "resolved_from" not in kb.neighborhood("Planner", max_depth=1)


def test_the_researcher_s_graph_tool_reports_the_match_too(kb):
    kb.graph.add_node("Planner", type="entity")
    kb.graph.add_edge("serve.py", "Planner", relation="mentions")

    result = kb.query_graph("planner", hops=1)

    assert result["entity"] == "Planner"
    assert result["resolved_from"] == "planner"
