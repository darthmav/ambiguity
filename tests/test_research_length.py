"""Tests for how much retrieved evidence reaches the Builder.

Retrieval is expensive and careful — dense recall, a BM25 re-rank, a relevance
floor, a collapse back onto documents — and all of it is wasted if the passage
it selects is then cut down before the Builder reads it. Two things were doing
that: the search asked for five results and only three were forwarded, and each
was truncated to 300 characters of a passage averaging 882.

The truncation is the part worth pinning, because of *which* 300 characters it
kept. A chunk is chosen because it matched, and the matching sentence can be
anywhere in it, so keeping the opening third keeps the part with no particular
reason to be relevant. On a `research/web` document the opening is the
provenance header, and the top-scoring source of the 2026-09-09 run reached the
Builder as a title, a URL, a timestamp and the goal it was fetched for, cut
mid-word, with none of the article attached.
"""

from __future__ import annotations

import langgraph_agent.nodes as nodes
from langgraph_agent.nodes import (
    RESEARCH_RESULTS,
    RESEARCH_SNIPPET_CHARS,
    _research_snippet,
)

# The exact shape that defeated the old cut: a web document's provenance block
# is ~200 characters before a word of the article.
WEB_HEADER = (
    "# k1 Parameter in BM25: Term Frequency Saturation | Inference Systems\n\n"
    "- Source: https://inferensys.com/glossary/semantic-search/bm25/k1-parameter\n"
    "- Retrieved: 2026-09-09 16:46 UTC\n"
    "- Researched for: Explain how Okapi BM25 term saturation works via k1\n\n---\n\n"
)
ARTICLE = "The k1 parameter controls how quickly term frequency saturates. " * 6


def test_a_passage_shorter_than_the_cap_arrives_whole():
    """The common case: a chunk averages 882 characters and the cap is 1,500."""
    passage = "BM25 saturates term frequency through k1. " * 10

    assert _research_snippet(passage) == passage.strip()


def test_the_substance_of_a_web_document_now_reaches_the_builder():
    """The regression, stated as the thing that actually went wrong.

    At 300 characters this passage was its own citation and nothing else.
    """
    snippet = _research_snippet(WEB_HEADER + ARTICLE)

    assert "k1 parameter controls how quickly term frequency saturates" in snippet
    # And the old cut genuinely would not have carried it.
    assert "term frequency saturates" not in (WEB_HEADER + ARTICLE)[:300]


def test_a_truncated_passage_says_that_it_was():
    """A silent trim reads as a source that had nothing more to say."""
    snippet = _research_snippet("x" * (RESEARCH_SNIPPET_CHARS + 500))

    assert "truncated" in snippet
    assert len(snippet) < RESEARCH_SNIPPET_CHARS + 100


def test_the_cap_clears_a_typical_passage_by_a_wide_margin():
    """Measured p99 of this corpus is 1,373 characters; the cap is above it.

    Pinned so the cap cannot drift back under the population it has to clear —
    that is the mistake the hard-coded relevance floor was corrected for, one
    level along.
    """
    assert RESEARCH_SNIPPET_CHARS > 1373


def test_every_retrieved_result_is_forwarded(monkeypatch):
    """One number feeds the request and the slice, so they cannot drift.

    Asking for five and forwarding three dropped two passages that had already
    been retrieved, ranked and re-ranked — and the diverse two, since the
    search widens its window precisely to stop one file taking every hit.
    """
    asked: dict[str, object] = {}

    def fake_tool(name, args):
        asked.update(args)
        return {
            "results": [
                {"id": f"doc{n}.md", "content": f"passage {n} " + "body " * 40, "score": 0.9}
                for n in range(RESEARCH_RESULTS)
            ],
            "source": "local_graphrag",
        }

    # A floor, pinned: unmeasured (`None`) would route to the seat instead,
    # and none of the results above would be forwarded at all.
    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: 0.37)
    monkeypatch.setattr(nodes, "_call_tool", fake_tool)
    findings, status = nodes._gather_research(
        {"plan": "study bm25", "goal": "g", "research": "", "messages": []}  # type: ignore[arg-type]
    )

    assert asked["top_k"] == RESEARCH_RESULTS
    assert status == "ready_for_builder"
    for n in range(RESEARCH_RESULTS):
        assert f"passage {n}" in findings


def test_each_finding_names_the_file_and_line_it_came_from(monkeypatch):
    """Passages used to arrive with no source, so the Builder could not open the file."""
    def fake_tool(name, args):
        return {
            "results": [
                {"id": "src/app/graph.py", "line": 118, "content": "def neighbours(): " + "body " * 40, "score": 0.9},
                {"id": "notes.md", "content": "no line known " + "body " * 40, "score": 0.8},
            ],
            "source": "local_graphrag",
        }

    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: 0.37)
    monkeypatch.setattr(nodes, "_call_tool", fake_tool)
    findings, _ = nodes._gather_research(
        {"plan": "p", "goal": "g", "research": "", "messages": []}  # type: ignore[arg-type]
    )

    assert "1. src/app/graph.py:118\n" in findings
    assert "2. notes.md\n" in findings


def test_retrieval_answers_when_its_best_hit_clears_the_floor_whatever_rank_it_has(monkeypatch):
    """BM25 can rank a lower-cosine passage first. The floor was measured on a
    search's best cosine and was then read off its first hit, a different number."""
    asked_the_seat: list[object] = []

    def fake_tool(name, args):
        return {
            "results": [
                {"id": "lexical-first.md", "content": "matched the words " * 20, "score": 0.30},
                {"id": "dense-best.md", "content": "matched the meaning " * 20, "score": 0.52},
            ],
            "source": "local_graphrag",
        }

    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: 0.37)
    monkeypatch.setattr(nodes, "_call_tool", fake_tool)
    monkeypatch.setattr(nodes, "get_agent_llm", lambda seat: asked_the_seat.append(seat))
    findings, status = nodes._gather_research(
        {"plan": "p", "goal": "g", "research": "", "messages": []}  # type: ignore[arg-type]
    )

    assert status == "ready_for_builder"
    assert asked_the_seat == []
    assert "dense-best.md" in findings


def test_the_seat_is_told_the_best_score_not_the_first(monkeypatch):
    text = nodes._retrieval_for_the_seat(
        [{"id": "a.md", "content": "x", "score": 0.10}, {"id": "b.md", "content": "y", "score": 0.30}],
        0.37,
        "unused",
    )
    assert "The best scored 0.30" in text
