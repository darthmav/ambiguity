"""Tests for the lexical half of search, and for the relevance floor it feeds.

Two defects are pinned here, and neither announced itself.

The first is that dense retrieval is a poor instrument for "this passage
contains this exact rare identifier". Measured against ground truth that cannot
be argued with -- 541 identifiers each defined in exactly one project file --
dense retrieval put the defining file first 53.4% of the time and dense plus a
BM25 re-rank 65.1%, McNemar p < 0.001, measured through `search` itself. Nothing about the failure is visible
from inside: search returned five plausible files, all real, all mentioning the
constant, and the one that *defines* it ranked fourth.

The second is the relevance floor. It was 0.3 once, hard-coded beside the
comparison in `nodes.py`, and 0.3 turned out to sit inside the off-corpus score
population rather than below it -- so a question this corpus cannot answer was
formatted into the findings as though it had. The floor is now measured per
corpus by `calibrate_relevance_floor` and stored with the corpus, never
hard-coded: a cosine has no meaning across models, so no number is borrowed
from one model to judge another.

No test here talks to a daemon or loads a real embedder, for the reason
`test_chunking.py` does not: the ordering, the fusion and the invalidation are
all exercisable against a stand-in.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
import pytest
from store_doubles import StoreDoubleMixin

from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
from langgraph_agent.lexical import (
    BM25Index,
    lexical_order,
    reciprocal_rank_fusion,
    tokenize,
)

# --------------------------------------------------------------------------
# tokenize
# --------------------------------------------------------------------------


def test_an_identifier_is_indexed_whole_and_in_pieces():
    """Both, and the "whole" half is the one that does the work here.

    A question spelling `BUILDER_DEADLINE_SECONDS` exactly has to match it
    exactly. Emitting only the pieces would leave it near-identical to
    `NODE_DEADLINE_SECONDS`, which is the confusion the lexical half exists to
    resolve rather than reproduce.
    """
    terms = tokenize("BUILDER_DEADLINE_SECONDS")

    assert "builder_deadline_seconds" in terms
    assert {"builder", "deadline", "seconds"} <= set(terms)


def test_camel_case_splits_including_the_end_of_an_acronym():
    """`GraphRAGKnowledgeBase` has two kinds of seam and both have to cut.

    Without the acronym rule this splits to `graph` + `ragknowledge` + `base`.
    `ragknowledge` is a term no query will ever contain, so the word
    `knowledge` is simply absent from the index -- the name is in the corpus
    and unfindable by the obvious search for it.
    """
    terms = tokenize("GraphRAGKnowledgeBase")

    assert "graphragknowledgebase" in terms
    assert {"graph", "rag", "knowledge", "base"} <= set(terms)


def test_single_character_pieces_are_dropped():
    """`a` and `b` from `a_b_thing` are noise: they match everything."""
    assert "a" not in tokenize("a_b_thing")


def test_punctuation_and_digits_do_not_produce_terms():
    assert tokenize("--- 42 ... ???") == []


# --------------------------------------------------------------------------
# BM25Index
# --------------------------------------------------------------------------

def test_ids_and_texts_must_line_up():
    """A silent mismatch would file every score under the wrong chunk."""
    with pytest.raises(ValueError):
        BM25Index(["a", "b"], ["only one"])


def test_an_empty_index_scores_without_dividing_by_zero():
    """The mean length of no documents is a ZeroDivisionError.

    Search has to be able to ask an empty store a question -- that is the
    `no_corpus` path -- so the index must be constructible over nothing.
    """
    index = BM25Index([], [])

    assert len(index) == 0
    assert index.score("anything", ["missing"]) == {"missing": 0.0}


def test_a_chunk_the_index_never_saw_scores_zero_rather_than_raising():
    """The store is the authority on what exists.

    The index is rebuilt from the store, so the two can be a moment apart. That
    is a reason to score the stranger 0.0, not a reason to take search down.
    """
    index = BM25Index(["a#0000"], ["alpha beta"])

    assert index.score("alpha", ["b#0000"]) == {"b#0000": 0.0}


def test_a_rare_term_outweighs_a_common_one():
    """This is the whole point of the lexical half: idf, not word count.

    Every chunk mentions `seconds`; one defines the constant. A ranking that
    counted matches would prefer the chunk saying `seconds` three times.
    """
    ids = ["common#0", "common#1", "common#2", "rare#0"]
    texts = [
        "seconds seconds seconds timeout budget",
        "seconds timeout deadline budget",
        "seconds budget deadline timeout",
        "BUILDER_DEADLINE_SECONDS is the builder's own budget",
    ]
    index = BM25Index(ids, texts)

    scores = index.score("BUILDER_DEADLINE_SECONDS", ids)

    assert scores["rare#0"] == max(scores.values())
    assert scores["rare#0"] > 0


# --------------------------------------------------------------------------
# reciprocal_rank_fusion
# --------------------------------------------------------------------------

def test_two_rankings_that_agree_are_returned_unchanged():
    assert reciprocal_rank_fusion(["a", "b", "c"], ["a", "b", "c"]) == ["a", "b", "c"]


def test_an_item_both_rankings_like_beats_one_only_a_single_ranking_likes():
    """Agreement is what the fusion buys; neither half wins on its own."""
    fused = reciprocal_rank_fusion(["x", "b", "a"], ["y", "b", "a"])

    assert fused[0] == "b", "ranked second by both, against firsts by one each"


def test_a_tie_keeps_the_order_it_was_first_seen_in():
    """A lexical half with nothing to say must not reshuffle the dense order.

    Every score zero makes every lexical rank arbitrary. Without a stable tie
    break the fusion would spend real dense evidence on that noise.
    """
    dense = ["a", "b", "c"]

    assert reciprocal_rank_fusion(dense, dense) == dense
    assert reciprocal_rank_fusion(dense) == dense


# --------------------------------------------------------------------------
# lexical_order
# --------------------------------------------------------------------------

def test_without_an_index_the_order_is_untouched():
    assert lexical_order("q", ["a", "b"], None) == ["a", "b"]


def test_a_query_sharing_no_term_leaves_the_order_untouched():
    """Not an arbitrary permutation. All-zero scores are "no opinion"."""
    index = BM25Index(["a", "b"], ["alpha", "beta"])

    assert lexical_order("zzzz", ["a", "b"], index) == ["a", "b"]


def test_the_order_is_by_score_when_there_is_an_opinion():
    index = BM25Index(["a", "b"], ["alpha alpha", "beta"])

    assert lexical_order("beta", ["a", "b"], index) == ["b", "a"]


# --------------------------------------------------------------------------
# the relevance floor
# --------------------------------------------------------------------------

def test_a_measured_record_becomes_the_floor(tmp_path, monkeypatch, postgres):
    """The floor is the calibrator's record, stored with the corpus."""
    from langgraph_agent import graphrag_server

    monkeypatch.chdir(tmp_path)
    store = graphrag_server.GraphRAGKnowledgeBase().collection
    store.set_floor_record({"model": graphrag_server.EMBEDDING_MODEL_NAME, "floor": 0.5})

    assert graphrag_server.relevance_floor() == 0.5


def test_an_unmeasured_corpus_has_no_floor(tmp_path, monkeypatch):
    """None, never a borrowed number: a cosine means nothing across models.

    `nodes.py` treats a None floor as "retrieval cannot tell an answer from
    noise" and hands every search to the Researcher's model. The last hard-coded
    number earned its removal by filing a question the corpus cannot answer
    under answered -- and a missing floor must fail loud in exactly the way a
    wrong one cannot.
    """
    from langgraph_agent import graphrag_server

    monkeypatch.chdir(tmp_path)

    assert graphrag_server.relevance_floor() is None


def test_a_record_for_another_model_is_no_floor(tmp_path, monkeypatch, postgres):
    """A corpus can outlive its model; the old number must not survive that."""
    from langgraph_agent import graphrag_server

    monkeypatch.chdir(tmp_path)
    store = graphrag_server.GraphRAGKnowledgeBase().collection
    store.set_floor_record({"model": "some-other-model", "floor": 0.5})

    assert graphrag_server.relevance_floor() is None


def _corpus_kb(tmp_path, files: dict[str, str]) -> GraphRAGKnowledgeBase:
    """A corpus of real files on disk, with a fake store and no embedder."""
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        kb.graph.add_node(str(path), type="document")
        kb.collection.upsert(
            ids=[f"{path}#0000"], embeddings=[[0.0, 0.0, 0.0]], documents=[text],
            metadatas=[{"doc_id": str(path), "sha": f"sha-{name}"}],
        )
    return kb


def _page(goal: str) -> str:
    return (
        "# A page\n\n- Source: https://example.org/page\n- Retrieved: 2026-10-02 10:00 UTC\n"
        f"- Researched for: {goal}\n\n---\n\nThe article itself.\n"
    )


def test_the_questions_the_corpus_answers_come_from_its_own_documents(tmp_path):
    """They were twelve questions about this project's own code, which the
    corpus -- the research archive -- never holds."""
    from langgraph_agent.graphrag_server import answered_questions

    kb = _corpus_kb(tmp_path, {
        "research/web/a.md": _page("How should Postgres autovacuum be tuned?"),
        "research/web/b.md": _page("How should Postgres autovacuum be tuned?"),
        "uploads/deploy.md": "# Deploying the console with Docker\n\nSteps follow.\n",
        "uploads/plain.txt": "a first line with enough words\nand more\n",
        "uploads/tiny.md": "# Notes\n",
    })

    assert answered_questions(kb, 12) == [
        "How should Postgres autovacuum be tuned?",  # one goal, asked once
        "Deploying the console with Docker",
        "a first line with enough words",
    ]


def test_the_questions_are_spread_across_a_large_corpus(tmp_path):
    from langgraph_agent.graphrag_server import answered_questions

    kb = _corpus_kb(tmp_path, {
        f"research/web/{n:02d}.md": _page(f"What does research topic number {n:02d} cover?")
        for n in range(30)
    })

    picked = answered_questions(kb, 4)
    assert picked[0].endswith("00 cover?") and picked[-1].endswith("29 cover?")
    assert len(set(picked)) == 4


def test_the_floor_sits_between_the_corpus_s_own_answers_and_off_domain_questions(
    tmp_path, monkeypatch
):
    from langgraph_agent.graphrag_server import (
        answered_questions,
        calibrate_relevance_floor,
        corpus_signature,
    )

    kb = _corpus_kb(tmp_path, {
        f"research/web/{n}.md": _page(f"How does subsystem number {n} behave?")
        for n in range(3)
    })
    answered = set(answered_questions(kb, 12))
    # The best cosine of each search, not its first hit, is what is measured.
    monkeypatch.setattr(kb, "search", lambda q, k: [
        {"score": 0.1}, {"score": 0.8 if q in answered else 0.3},
    ])

    record = calibrate_relevance_floor(kb)

    assert record["answered"] == [0.8, 0.8, 0.8]
    assert set(record["unanswerable"]) == {0.3}
    assert record["floor"] == 0.55
    assert record["corpus"] == corpus_signature(kb)
    assert kb.collection.floor_record() == record


def test_a_corpus_too_small_to_answer_itself_gets_no_floor_and_no_search(tmp_path, monkeypatch):
    from langgraph_agent.graphrag_server import calibrate_relevance_floor

    kb = _corpus_kb(tmp_path, {
        "research/web/a.md": _page("How does the only topic here behave?"),
        "uploads/b.md": "# Two words\n",
    })
    searched: list[str] = []
    monkeypatch.setattr(kb, "search", lambda q, k: searched.append(q) or [])

    record = calibrate_relevance_floor(kb)

    assert (record["floor"], record["too_small"]) == (None, 1)
    assert searched == []


def test_a_changed_corpus_has_another_signature(tmp_path):
    from langgraph_agent.graphrag_server import corpus_signature

    kb = _corpus_kb(tmp_path, {"uploads/a.md": "# Some document title\n"})
    before = corpus_signature(kb)
    kb.collection.upsert(
        ids=["late.md#0000"], embeddings=[[0.0, 0.0, 0.0]], documents=["late"],
        metadatas=[{"doc_id": "late.md", "sha": "sha-late"}],
    )
    assert corpus_signature(kb) != before


def test_a_floor_measured_on_other_texts_is_measured_again(monkeypatch):
    """A record's presence was `known` for good: a no-gap record taken on a
    corpus of three pages kept retrieval off however the archive grew."""
    import serve
    from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME

    measured: list[object] = []
    stored: dict[str, Any] = {"model": EMBEDDING_MODEL_NAME, "floor": None, "corpus": "then"}
    corpus = object()
    monkeypatch.setattr(serve, "floor_calibration", lambda: stored)
    monkeypatch.setattr(serve, "_open_kb", lambda: corpus)
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", EMBEDDING_MODEL_NAME))
    monkeypatch.setattr(serve, "calibrate_relevance_floor", lambda kb: measured.append(kb) or {
        "model": EMBEDDING_MODEL_NAME, "floor": 0.5, "answered": [0.7], "unanswerable": [0.3],
    })

    monkeypatch.setattr(serve, "corpus_signature", lambda kb: "now")
    assert serve._calibrate_the_floor_before_the_run({"source": "updated"})["source"] == "calibrated"
    assert measured == [corpus]

    monkeypatch.setattr(serve, "corpus_signature", lambda kb: "then")
    assert serve._calibrate_the_floor_before_the_run({"source": "current"})["source"] == "known"
    assert measured == [corpus]

    del stored["corpus"]  # taken before records named their corpus
    assert serve._calibrate_the_floor_before_the_run({"source": "current"})["source"] == "calibrated"


def test_the_planner_map_does_not_search_without_a_floor(monkeypatch):
    """It searched -- loading the embedder and evicting the seat about to plan --
    and only then found there was no floor to read the hits against."""
    import langgraph_agent.nodes as nodes

    searched: list[object] = []
    monkeypatch.setattr("langgraph_agent.graphrag_server.relevance_floor", lambda: None)
    monkeypatch.setattr(nodes, "_call_tool", lambda name, args: searched.append(args) or {})

    assert nodes._corpus_map("a goal") == ""
    assert searched == []


def test_nodes_read_the_floor_and_never_hard_code_one():
    """The floor is a property of the model, and meaningless apart from it.

    It lives in `graphrag_server`, beside `EMBEDDING_MODEL_NAME`, and the
    assertion with teeth is on `nodes.py`: it must *read* it through
    `relevance_floor`, never carry its own copy of a number, or the two drift.
    """
    import langgraph_agent.graphrag_server as server
    import langgraph_agent.nodes as nodes

    assert hasattr(server, "EMBEDDING_MODEL_NAME")
    assert callable(server.relevance_floor)
    assert not hasattr(server, "RETRIEVAL_RELEVANCE_FLOOR"), (
        "the floor is measured per corpus, never a constant"
    )

    source = nodes.__file__ or ""
    assert source, "the module has to be on disk to read"
    text = open(source, encoding="utf-8").read()
    assert "relevance_floor()" in text, "nodes.py must read the model's floor"
    assert 'get("score", 0) > 0.' not in text, "and must not hard-code a floor"


# --------------------------------------------------------------------------
# search(): the two halves together
# --------------------------------------------------------------------------

class _FakeEmbedder:
    """A tokenizer and a vector -- enough for chunking, no model loaded."""

    placement_note: str | None = None

    class _Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            spans = [m.span() for m in re.finditer(r"\S+", text)]
            return {"offset_mapping": spans}

    def __init__(self) -> None:
        self.tokenizer = self._Tokenizer()

    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        import numpy as np

        return np.zeros((len(text), 3)) if isinstance(text, list) else np.zeros(3)


class _FakeCollection(StoreDoubleMixin):
    """Hits in insertion order with a distance that rises down the list.

    The dense ranking is fixed and known, which is what makes a re-rank
    observable: whatever comes back in a different order was reordered by the
    lexical half and by nothing else.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any], list[float]]] = {}

    def upsert(self, ids, embeddings, documents, metadatas):  # type: ignore[no-untyped-def]
        for i, chunk_id in enumerate(ids):
            self.rows[chunk_id] = (documents[i], metadatas[i], embeddings[i])

    def get(self, include: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        include = include or []
        ids = list(self.rows)
        out: dict[str, Any] = {"ids": ids}
        if "documents" in include:
            out["documents"] = [self.rows[i][0] for i in ids]
        if "metadatas" in include:
            out["metadatas"] = [self.rows[i][1] for i in ids]
        return out

    def delete(self, ids=None, where=None):  # type: ignore[no-untyped-def]
        if where is not None:
            for i in [
                i for i, (_, m, _) in self.rows.items()
                if all(m.get(k) == v for k, v in where.items())
            ]:
                self.rows.pop(i, None)
        for chunk_id in ids or []:
            self.rows.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.rows)

    def query(self, query_embeddings, n_results, include=None, **kwargs):  # type: ignore[no-untyped-def]
        ids = list(self.rows)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i][0] for i in ids]],
            "metadatas": [[self.rows[i][1] for i in ids]],
            "distances": [[0.1 * rank for rank in range(len(ids))]],
        }


@pytest.fixture
def kb(tmp_path):
    base = object.__new__(GraphRAGKnowledgeBase)
    base.persist_dir = tmp_path
    base.collection = _FakeCollection()
    base.graph = nx.DiGraph()
    base._embedder = _FakeEmbedder()  # type: ignore[assignment]
    return base


def _seed_identifier_corpus(kb):
    """Insertion order is the dense order: `other` first, `defines` second."""
    kb.add_document("other.md", "alpha beta gamma delta")
    kb.add_document("defines.md", "BUILDER_DEADLINE_SECONDS bounds the builder")
    kb.add_document("mentions.md", "the deadline and the budget and the seconds")


def test_the_exact_identifier_beats_the_document_dense_ranked_first(kb):
    """The defect this whole module exists for.

    `other.md` is first in the dense ranking and has nothing to do with the
    query; `defines.md` is where the constant is defined. On 541 real
    identifiers, dense retrieval made that class of mistake 44.7% of the time.
    """
    _seed_identifier_corpus(kb)

    results = kb.search("BUILDER_DEADLINE_SECONDS", top_k=3)

    assert [r["id"] for r in results][0] == "defines.md"


def test_two_candidates_ranked_exactly_opposite_keep_the_dense_order(kb):
    """Reciprocal rank fusion is symmetric, so an exact reversal is a tie.

    With only two candidates, "first and second" against "second and first"
    sums to the same score either way, and the tie break hands it to the dense
    ordering. That is the right way to lose: the lexical half is a re-rank of a
    result that was already correct without it, so where the two disagree with
    equal force, the half that did the retrieving keeps the call.

    Pinned because it looks like a bug when it appears in a two-document test
    corpus and never appears in a real one, where the dense window is twenty
    chunks deep.
    """
    kb.add_document("mentions.md", "the deadline and the budget and the seconds")
    kb.add_document("defines.md", "BUILDER_DEADLINE_SECONDS bounds the builder")

    results = kb.search("BUILDER_DEADLINE_SECONDS", top_k=2)

    assert [r["id"] for r in results] == ["mentions.md", "defines.md"]


def test_a_query_the_lexical_half_cannot_judge_keeps_the_dense_order(kb):
    """No shared term is no opinion, and no opinion must change nothing."""
    kb.add_document("first.md", "alpha alpha alpha")
    kb.add_document("second.md", "beta beta beta")

    results = kb.search("zzzzz", top_k=2)

    assert [r["id"] for r in results] == ["first.md", "second.md"]


def test_a_reranked_result_still_carries_its_own_dense_score(kb):
    """The gate reads `results[0]["score"]`, so it must stay a real cosine.

    Every candidate comes from the dense window and keeps the distance that
    window gave it. A fused *rank* leaking into this field would feed
    the relevance floor a number that is not a similarity at all.
    """
    _seed_identifier_corpus(kb)

    promoted = kb.search("BUILDER_DEADLINE_SECONDS", top_k=3)[0]

    assert promoted["id"] == "defines.md"
    # Second in the dense ranking: distance 0.1, so the score is 1 - 0.1. The
    # promotion moved it to the front of the results and left this untouched.
    assert promoted["score"] == pytest.approx(0.9)


def test_adding_a_document_drops_the_lexical_index(kb):
    """Otherwise the re-rank describes a corpus that no longer exists."""
    kb.add_document("a.md", "alpha")
    assert len(kb.lexical_index) == 1

    kb.add_document("b.md", "beta")
    assert kb._lexical_index is None, "stale index kept across a write"
    assert len(kb.lexical_index) == 2


def test_clearing_the_corpus_drops_the_lexical_index(kb):
    kb.add_document("a.md", "alpha")
    assert len(kb.lexical_index) == 1

    kb.clear()

    assert kb._lexical_index is None
    assert len(kb.lexical_index) == 0


def test_a_store_that_cannot_list_itself_falls_back_to_dense(kb):
    """A degraded ranking beats a broken search.

    The lexical half improves an ordering that is already correct without it,
    so a collection too old or too foreign to answer `get` is a reason to skip
    the re-rank, not to raise.
    """
    kb.add_document("a.md", "alpha")
    kb.add_document("b.md", "beta")
    kb._lexical_index = None

    def refuse(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("this collection does not support get()")

    kb.collection.get = refuse  # type: ignore[method-assign]

    assert kb.lexical_index is None
    assert [r["id"] for r in kb.search("alpha", top_k=2)] == ["a.md", "b.md"]
