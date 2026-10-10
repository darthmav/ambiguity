"""Tests for splitting a document into passages the embedder can read whole.

Before chunking, `add_document` embedded a file in one `encode()` call while
the model's window is 256 tokens, so everything past roughly the first thousand
characters was silently discarded: measured on this project's own corpus, 73 of
77 documents truncated and 91.5% of the corpus unreachable by search, with the
vector for all 46,094 characters of CLAUDE.md identical to the vector for its
first 1,000. Nothing raised, and the counters read the same either way -- which
is why these are tests and not a note in the README.

No test here touches the daemon or the real tokenizer. The suite goes out of
its way to keep the model out (`test_stats_does_not_load_the_embedder_for_the_health_check`
asserts as much), so the packing, the trimming and the collapse back onto
documents are all exercised against a stand-in tokenizer that reproduces the
one behaviour that matters: a slice re-tokenizes to a different length than it
had inside its parent.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
import pytest
from store_doubles import StoreDoubleMixin

from langgraph_agent.graphrag_server import (
    CHUNK_ID_SEPARATOR,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_BATCH_SIZE,
    ENTITY_STOPWORDS,
    GraphRAGKnowledgeBase,
    _chunk_windows,
    _content_sha,
    _document_id_of,
    index_corpus_files,
)

# The walk's mechanics, laid out at the top of a scratch tree.
pytestmark = pytest.mark.usefixtures("whole_root_walk")


class _FakeTokenizer:
    """A word-piece stand-in: runs of non-space text, split every `piece` chars.

    The splitting is what makes it a useful fake rather than a simple one. A
    real word-piece tokenizer decides on context, so a fragment cut mid-word
    encodes to a different number of tokens standalone than it did inside the
    document. Aligning the pieces from the *start of each run* reproduces
    exactly that: slice a run down the middle and the pieces re-align, and the
    count moves.
    """

    def __init__(self, piece: int = 4) -> None:
        self.piece = piece

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", text):
            start, end = match.span()
            for i in range(start, end, self.piece):
                spans.append((i, min(i + self.piece, end)))
        return spans

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(text, list):
            spans = [self._spans(one) for one in text]
            out: dict[str, Any] = {"input_ids": [[0] * len(s) for s in spans]}
            if return_offsets_mapping:
                out["offset_mapping"] = spans
            return out

        one_spans = self._spans(text)
        out = {"input_ids": [0] * len(one_spans)}
        if return_offsets_mapping:
            out["offset_mapping"] = one_spans
        return out


class _FakeEmbedder:
    """Enough of an OllamaEmbedder for chunking: a tokenizer and a vector."""

    placement_note: str | None = None

    def __init__(self, piece: int = 4) -> None:
        self.tokenizer = _FakeTokenizer(piece)

    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        import numpy as np

        if isinstance(text, list):
            return np.zeros((len(text), 3))
        return np.zeros(3)


class _FakeChunkCollection(StoreDoubleMixin):
    """The slice of the Chroma API the chunked write and read paths use.

    Supports `delete(where=...)` because that is how a document's previous
    chunks are swept before its new ones land; a collection without it is
    covered separately by `_FakeCollection` in `test_corpus_admin.py`.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any], list[float]]] = {}

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
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

    def delete(
        self, ids: list[str] | None = None, where: dict[str, Any] | None = None
    ) -> None:
        if where is not None:
            doomed = [
                i for i, (_, meta, _) in self.rows.items()
                if all(meta.get(k) == v for k, v in where.items())
            ]
            for chunk_id in doomed:
                self.rows.pop(chunk_id, None)
        for chunk_id in ids or []:
            self.rows.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.rows)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Hits in insertion order, with a distance that rises down the list.

        The ranking is not what is under test; the collapse back onto documents
        is, and that only needs hits to arrive best-first the way Chroma
        delivers them.
        """
        ids = list(self.rows)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i][0] for i in ids]],
            "metadatas": [[self.rows[i][1] for i in ids]],
            "distances": [[0.1 * rank for rank in range(len(ids))]],
        }


def _make_kb(tmp_path, piece: int = 4) -> GraphRAGKnowledgeBase:
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeChunkCollection()
    kb.graph = nx.DiGraph()
    kb._embedder = _FakeEmbedder(piece)  # type: ignore[assignment]
    return kb


@pytest.fixture
def kb(tmp_path):
    return _make_kb(tmp_path)


# ---------------------------------------------------------------------------
# the packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [1, 50, 253, 254, 255, 500, 1000, 11760])
def test_every_token_lands_in_a_window_and_none_exceeds_the_budget(n_tokens):
    """The two halves of the contract, and the bug is the first one.

    Full coverage is what stops the truncation. The budget is what stops each
    chunk from being truncated in turn, which would reintroduce the same bug
    one level down.
    """
    windows = _chunk_windows(n_tokens, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)

    covered: set[int] = set()
    for start, end in windows:
        assert start < end, "an empty window carries no tokens"
        assert end - start <= CHUNK_MAX_TOKENS
        covered.update(range(start, end))

    assert covered == set(range(n_tokens))


def test_an_empty_document_produces_no_windows():
    assert _chunk_windows(0, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS) == []


def test_an_overlap_wider_than_the_window_still_terminates():
    """A stride at or below zero would loop forever rather than fail loudly.

    A misconfigured overlap is a caller's mistake, but hanging the reindex is a
    worse answer to it than degrading to a narrower overlap.
    """
    windows = _chunk_windows(100, 10, 999)

    assert windows, "the document still has to be covered"
    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(100))


# ---------------------------------------------------------------------------
# the slicing
# ---------------------------------------------------------------------------


def test_a_document_inside_the_window_is_left_whole(kb):
    content = "one two three four five"

    assert kb.chunk_text(content) == [content]


def test_an_empty_document_yields_no_chunks(kb):
    assert kb.chunk_text("") == []
    assert kb.chunk_text("   \n  ") == []


def test_the_chunks_are_literal_substrings_that_span_the_document(kb):
    """Coverage, stated the way a reader can check it.

    The first chunk has to start the document and the last has to end it, or
    something was dropped off an end -- which is the original bug, just at a
    smaller scale.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    chunks = kb.chunk_text(content)

    assert len(chunks) > 1, "this document is far past one window"
    for chunk in chunks:
        assert chunk in content
    assert content.startswith(chunks[0])
    assert content.endswith(chunks[-1])


def test_neighbouring_chunks_overlap(kb):
    """The overlap is what keeps a passage cut down the middle findable.

    It is also what makes trimming safe in `_fit_chunks`, so a run of chunks
    that did not actually overlap would break that argument silently.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    chunks = kb.chunk_text(content)

    starts = [content.index(chunk) for chunk in chunks[:3]]
    ends = [start + len(chunk) for start, chunk in zip(starts, chunks[:3], strict=True)]
    assert starts[1] < ends[0], "chunk 2 begins after chunk 1 ended: no overlap"
    assert starts[2] < ends[1]


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------


def _over_budget_text(extra: int = 5) -> str:
    """A passage that tokenizes past the window under `_FakeTokenizer`."""
    return " ".join(["w"] * (CHUNK_MAX_TOKENS + extra))


def test_a_chunk_that_retokenizes_over_the_window_is_trimmed(kb):
    """Slicing on the parent's token boundaries does not bound the slice.

    Measured against the real tokenizer, 16 of 1,052 full-size chunks came back
    one token longer standalone than they were inside the document, which put
    them at 257 against a 256 window -- handed straight back to the truncation
    this change exists to remove. Padding the constant by the drift observed
    once is guessing; re-encoding and cutting makes the window a fact.
    """
    over = _over_budget_text()

    fitted = kb._fit_chunks([over, "a tail chunk"])

    assert len(kb.embedder.tokenizer(fitted[0])["input_ids"]) == CHUNK_MAX_TOKENS
    assert over.startswith(fitted[0]), "a middle chunk is trimmed at its tail"


def test_the_last_chunk_is_trimmed_at_the_head_so_the_documents_tail_survives(kb):
    """Which end is cut is the whole point.

    A chunk's tail is covered by the next chunk's overlap and its head by the
    previous one's, so trimming into a neighbour loses nothing. The last chunk
    has no next, and trimming *its* tail would drop the final tokens of the
    file without saying so -- the original bug in miniature.
    """
    over = _over_budget_text()

    fitted = kb._fit_chunks(["a head chunk", over])

    assert len(kb.embedder.tokenizer(fitted[-1])["input_ids"]) == CHUNK_MAX_TOKENS
    assert over.endswith(fitted[-1]), "the last chunk must keep the document's end"


def test_a_lone_chunk_is_never_trimmed(kb):
    """It is the whole document, un-sliced, so it cannot have drifted."""
    over = _over_budget_text()

    assert kb._fit_chunks([over]) == [over]


# ---------------------------------------------------------------------------
# chunk ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_id", "metadata", "expected"),
    [
        ("CLAUDE.md#0007", None, "CLAUDE.md"),
        ("CLAUDE.md#0007", {"doc_id": "CLAUDE.md"}, "CLAUDE.md"),
        # A row written before chunking is keyed by the bare path and carries
        # no doc_id. It has to resolve to itself or a reindex reads it as a
        # stranger and prunes it.
        ("CLAUDE.md", None, "CLAUDE.md"),
        # "#" in a real path is not a chunk number and must be left alone.
        ("weird#name.py", None, "weird#name.py"),
        ("a/b#3.md", None, "a/b#3.md"),
    ],
)
def test_a_stored_row_resolves_to_the_document_it_came_from(
    chunk_id, metadata, expected
):
    assert _document_id_of(chunk_id, metadata) == expected


# ---------------------------------------------------------------------------
# writing and reading
# ---------------------------------------------------------------------------


def test_a_document_is_stored_as_several_rows_but_one_graph_node(kb):
    """Chunking changes what search can find, not what the corpus is shaped like.

    The graph is what every spectral diagnostic runs on, so a document that
    started arriving as forty nodes would quietly change the meaning of every
    one of them.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    kb.add_document("big.md", content, {"path": "big.md", "type": "markdown"})

    assert kb.collection.count() > 1, "a document past the window is several rows"
    documents = [n for n, a in kb.graph.nodes(data=True) if a.get("type") == "document"]
    assert documents == ["big.md"]

    for chunk_id, (_, metadata, _) in kb.collection.rows.items():
        assert chunk_id.startswith(f"big.md{CHUNK_ID_SEPARATOR}")
        assert metadata["doc_id"] == "big.md"
        assert metadata["path"] == "big.md", "the caller's metadata is preserved"


def test_a_shrunken_document_leaves_no_leftover_chunks(kb):
    """`upsert` overwrites 0..2 and leaves 3..9 answering searches.

    Same rule as `index_corpus_files` follows for whole documents -- a reindex
    rebuilds rather than accumulates -- applied one level down. Without it a
    file that lost a section keeps matching queries with text it no longer
    contains, and nothing raises.
    """
    long_content = " ".join(f"word{i}" for i in range(2000))
    kb.add_document("shrinking.md", long_content)
    assert kb.collection.count() > 3

    kb.add_document("shrinking.md", "now it is short")

    assert kb.collection.count() == 1
    assert list(kb.collection.rows) == [f"shrinking.md{CHUNK_ID_SEPARATOR}0000"]


def test_a_failed_embed_leaves_the_stored_document_in_place(kb):
    """Embed first, delete second: a failure must not turn stale into missing.

    On 2026-09-18 a model load that ran out of GPU memory made three documents
    vanish from the corpus, because their old rows were swept before the embed
    that was meant to replace them was attempted.
    """
    kb.add_document("kept.md", "the version the store already holds")
    before = dict(kb.collection.rows)

    def fail(texts: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Ollama could not embed: out of memory")

    kb._embedder.encode = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="out of memory"):
        kb.add_document("kept.md", "a newer version that could not be embedded")

    assert kb.collection.rows == before


def test_search_collapses_chunks_onto_their_documents(kb):
    """`top_k` counts documents, as every caller already assumed.

    Without the collapse a query matching six passages of one file returns that
    file six times and crowds out five other sources -- worse than the
    behaviour chunking replaced, rather than better.
    """
    kb.add_document("a.md", " ".join(f"alpha{i}" for i in range(2000)))
    kb.add_document("b.md", " ".join(f"beta{i}" for i in range(2000)))

    results = kb.search("anything", top_k=2)

    assert [r["id"] for r in results] == ["a.md", "b.md"]
    assert all(r["chunks_matched"] > 1 for r in results), (
        "each document matched on several of its chunks"
    )


def test_search_reports_the_document_path_and_the_passage_that_matched(kb):
    """`id` stays the document, `content` becomes the passage.

    Callers index the graph with `id`, print it as a filename and derive an
    entity from it, so it has to keep being a path. `content` is what the
    Researcher forwards 300 characters of to the Builder -- which used to be
    300 characters of whichever file's header scored best.
    """
    content = " ".join(f"word{i}" for i in range(2000))
    kb.add_document("a.md", content)

    result = kb.search("anything", top_k=1)[0]

    assert result["id"] == "a.md"
    assert result["id"] in kb.graph, "the id has to resolve against the graph"
    assert result["chunk_id"].startswith(f"a.md{CHUNK_ID_SEPARATOR}")
    assert result["content"] in content
    assert len(result["content"]) < len(content), "a passage, not the whole file"


def test_search_with_no_hits_returns_nothing(kb):
    assert kb.search("anything", top_k=5) == []


def test_a_reindex_prunes_by_document_not_by_chunk_id(kb, tmp_path, monkeypatch):
    """`CLAUDE.md#0007` is not in `wanted` and never will be.

    Comparing stored ids against the wanted set directly would find every chunk
    stale and delete the whole corpus on each reindex. It happens to end in the
    same place while everything is re-added in the same pass, which is exactly
    why it would sit there unnoticed until something reindexed a subset and
    silently re-embedded the lot.
    """
    (tmp_path / "kept.md").write_text(" ".join(f"word{i}" for i in range(2000)))
    kb.add_document("gone.md", "a document that will not be found on disk")
    kept_before = kb.collection.count()
    assert kept_before == 1

    monkeypatch.chdir(tmp_path)
    kb.persist_dir = tmp_path
    report = index_corpus_files(kb, str(tmp_path))

    stored = set(kb.collection.rows)
    assert not any(cid.startswith("gone.md") for cid in stored), "stale rows go"
    assert any(cid.startswith(str(tmp_path / "kept.md")) for cid in stored)
    assert report["indexed"] == 1


# ---------------------------------------------------------------------------
# entity extraction
# ---------------------------------------------------------------------------


def test_a_capital_forced_by_position_is_not_an_entity(kb):
    """The capital rule fires on words whose capital is punctuation, not meaning.

    A sentence opener, a numpy docstring heading and a Python literal all
    arrive capitalised for reasons that say nothing about the corpus. Before
    the stopword list `False` was the 4th best-connected node in this
    project's graph and `Returns` the 6th, above `Fiedler` and `Cheeger`: an
    edge through `Returns` claims two documents are related when all it knows
    is that both contain a docstring.
    """
    kb.add_document(
        "doc.py",
        "Returns the Builder. Parameters are ignored. Nothing else. "
        "Every caller checks False against the Laplacian.",
    )

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Builder", "Laplacian"}


def test_the_stopword_match_is_the_whole_token_not_a_prefix(kb):
    """`Returns` is stopped; `Researcher` merely starts with the same letters."""
    kb.add_document("doc.md", "Returns. Researcher and Verdict and Notes.")

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Researcher", "Verdict"}


def test_the_stopword_list_holds_no_domain_term(kb):
    """A guard on the list itself, which is hand-maintained and easy to widen.

    Every one of these is a term the corpus is *about*; stopping one would
    delete a real relation from the graph and nothing would raise.

    The last four were nominated by the 2026-09-09 widening and refused. They
    are the ones a future pass is most likely to get wrong, because each looks
    generic and each is load-bearing here: `L_dense` scores zero
    position-free capitals only because an assignment starts its line, and
    `System`, `Search` and `State` are ordinary English words that this corpus
    genuinely uses as terms (`System` carries 22 free capitals on its own).
    """
    for term in (
        "Builder", "Architect", "Researcher", "Planner", "Laplacian", "Fiedler",
        "Cheeger", "GraphRAG", "AgentState", "NetworkX", "Chroma", "Verdict",
        "Spectral", "ValueError", "LangGraph", "Normalized", "Conductance",
        "L_dense", "System", "Search", "State",
    ):
        assert term.lower() not in ENTITY_STOPWORDS, f"{term} is not boilerplate"


def test_the_widened_stopwords_are_actually_stopped(kb):
    """The 2026-09-09 additions, pinned by behaviour rather than by list entry.

    `Measured` is the one that matters: this project's own notes open
    sentences with it constantly, so the document recording a measurement was
    minting an entity for the word. Asserted through `add_document` because a
    membership check would pass on a list the extractor had stopped reading.
    """
    kb.add_document(
        "note.md",
        "Measured on this corpus, the Builder ran clean. Tests pass. "
        "Initialize the Laplacian. Write the report. Degree matters.",
    )

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Builder", "Laplacian"}


def test_the_stopword_list_is_lowercase_ascii():
    """It is matched with `token.lower()`, so an entry with a capital in it is
    dead weight that reads as coverage."""
    assert all(w == w.lower() and w.isascii() for w in ENTITY_STOPWORDS)


# ---------------------------------------------------------------------------
# reuse: a rebuild re-embeds what changed and nothing else
# ---------------------------------------------------------------------------


class _CountingEmbedder(_FakeEmbedder):
    """The fake embedder, counting the calls that would cost real time."""

    def __init__(self, piece: int = 4) -> None:
        super().__init__(piece)
        self.batches = 0

    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        if isinstance(text, list):
            self.batches += 1
        return super().encode(text)


@pytest.fixture
def counting_kb(tmp_path):
    kb = _make_kb(tmp_path)
    kb._embedder = _CountingEmbedder()  # type: ignore[assignment]
    return kb


def _project(tmp_path, **files: str):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        (root / name.replace("_", ".")).write_text(text, encoding="utf-8")
    return root


def test_a_rebuild_re_embeds_only_what_changed(counting_kb, tmp_path):
    """This is what lets a rebuild happen on every run instead of on a button.

    Embedding is the only expensive part of indexing this project: measured
    warm on the real store, 52.0s to re-embed 77 files and 1,618 chunks, and
    0.09s for the same rebuild when nothing had changed -- reading every file,
    hashing it, fetching the store's metadata and rebuilding the entire entity
    graph together account for 0.1s of that. Take the reuse away and keeping
    the corpus current goes back to being something a person has to remember.
    """
    root = _project(tmp_path,
                    a_md="The Planner interprets goals and routes onward.",
                    b_md="The Architect rules on the plan it was given.")

    first = index_corpus_files(counting_kb, str(root))
    assert (first["indexed"], first["embedded"], first["reused"]) == (2, 2, 0)
    after_first = counting_kb._embedder.batches
    assert after_first  # it really did embed

    second = index_corpus_files(counting_kb, str(root))
    assert (second["indexed"], second["embedded"], second["reused"]) == (2, 0, 2)
    assert counting_kb._embedder.batches == after_first  # not one batch more

    (root / "a.md").write_text("The Planner now routes to the Researcher.",
                               encoding="utf-8")
    third = index_corpus_files(counting_kb, str(root))
    assert (third["indexed"], third["embedded"], third["reused"]) == (2, 1, 1)
    assert counting_kb._embedder.batches == after_first + 1


def test_a_document_whose_vectors_were_reused_is_still_in_the_graph(counting_kb, tmp_path):
    """`index_corpus_files` clears the graph up front, so the cheap half has
    to run for a reused document too.

    Skipping it entirely would leave a corpus that answers searches perfectly
    and has no graph at all -- and the graph is what `neighborhood`,
    `query_graph`, `topics` and every spectral diagnostic read.
    """
    root = _project(tmp_path, a_md="The Planner interprets goals.")
    index_corpus_files(counting_kb, str(root))

    report = index_corpus_files(counting_kb, str(root))

    assert report["reused"] == 1
    path = str(root / "a.md")
    assert counting_kb.graph.nodes[path]["type"] == "document"
    assert counting_kb.graph.has_edge(path, "Planner")


def test_a_document_that_left_the_project_is_dropped_and_counted_once(
    counting_kb, tmp_path
):
    """`dropped` counts documents, not rows.

    A file is many chunks, so counting the pruned ids would report "9 dropped"
    for one deleted file -- and this number is the only evidence that a
    rebuild which re-read nothing still did something.
    """
    root = _project(tmp_path,
                    a_md="The Planner interprets goals. " * 40,
                    b_md="The Architect rules on the plan.")
    first = index_corpus_files(counting_kb, str(root))
    assert first["dropped"] == 0
    assert len([i for i in counting_kb.collection.rows if "a.md" in i]) > 1

    (root / "a.md").unlink()
    report = index_corpus_files(counting_kb, str(root))

    assert report["dropped"] == 1
    assert report["indexed"] == 1
    assert str(root / "a.md") not in counting_kb.graph
    assert not [i for i in counting_kb.collection.rows if "a.md" in i]


def test_a_file_grown_past_the_limit_takes_its_old_chunks_with_it(
    counting_kb, tmp_path, monkeypatch
):
    """It stayed on the walk, so the prune kept its chunks, and the rebuild
    then skipped it: search answered from text the file no longer had."""
    from langgraph_agent import graphrag_server

    root = _project(tmp_path, a_md="The Planner interprets goals.",
                    b_md="The Architect rules.")
    index_corpus_files(counting_kb, str(root))
    path = str(root / "a.md")
    assert [i for i in counting_kb.collection.rows if i.startswith(path)]

    monkeypatch.setattr(graphrag_server, "MAX_INDEXABLE_BYTES", 40)
    (root / "a.md").write_text("The Planner interprets goals. " * 10, encoding="utf-8")
    report = index_corpus_files(counting_kb, str(root))

    assert (report["skipped"], report["dropped"], report["indexed"]) == (1, 1, 1)
    assert not [i for i in counting_kb.collection.rows if i.startswith(path)]
    assert path not in counting_kb.graph


def test_a_file_that_stopped_decoding_takes_its_old_chunks_with_it(counting_kb, tmp_path):
    root = _project(tmp_path, a_md="The Planner interprets goals.")
    index_corpus_files(counting_kb, str(root))
    path = str(root / "a.md")

    (root / "a.md").write_bytes(b"\xff\xfe not UTF-8 \x80")
    report = index_corpus_files(counting_kb, str(root))

    assert report["dropped"] == 1
    assert any("a.md" in error for error in report["errors"])
    assert not [i for i in counting_kb.collection.rows if i.startswith(path)]


def test_the_console_reads_the_previous_graph_whole_until_the_rebuild_ends(
    counting_kb, tmp_path
):
    """The rebuild emptied the published graph and refilled it document by
    document, while the console's threads iterated it."""
    root = _project(tmp_path, a_md="The Planner interprets goals.",
                    b_md="The Architect rules on plans.")
    index_corpus_files(counting_kb, str(root))
    before = counting_kb.graph
    size = before.number_of_nodes()
    seen: list[tuple[bool, int]] = []

    index_corpus_files(
        counting_kb, str(root),
        progress=lambda done, total: seen.append(
            (counting_kb.graph is before, before.number_of_nodes())
        ),
    )

    assert seen == [(True, size), (True, size)]
    assert counting_kb.graph is not before
    assert counting_kb.graph.number_of_nodes() == size


def test_the_fingerprint_is_written_by_the_store_not_by_the_caller(counting_kb):
    """Every chunk carries it, whichever door the document came in through.

    Computed inside `add_document` for the reason `doc_id` and `chunk_count`
    are: an upload, a fetched page and the walk all reach it, and one caller
    forgetting to pass a fingerprint would cost that document its reuse
    silently -- it would simply be re-embedded on every run for ever.
    """
    text = "The Architect rules on the plan. " * 20
    counting_kb.add_document("notes.md", text, {"path": "notes.md"})

    shas = {row[1]["sha"] for row in counting_kb.collection.rows.values()}
    assert shas == {_content_sha(text)}


# ---------------------------------------------------------------------------
# where a passage sits, and an embedder that stays quiet
# ---------------------------------------------------------------------------


def test_a_passage_is_anchored_to_its_line_and_reaches_back_to_its_start(tmp_path):
    from langgraph_agent.graphrag_server import _passage_location

    source = tmp_path / "mod.py"
    source.write_text("import os\n\ndef neighbours(graph):\n    return list(graph.neighbors(node))\n")

    line, text = _passage_location(str(source), "graph.neighbors(node))\n")

    assert line == 4
    assert text == "    return list(graph.neighbors(node))\n"


def test_a_passage_the_file_no_longer_holds_gets_no_line(tmp_path):
    """A guessed line would point at the wrong code."""
    from langgraph_agent.graphrag_server import _passage_location

    source = tmp_path / "mod.py"
    source.write_text("rewritten since it was indexed\n")

    assert _passage_location(str(source), "the old text") is None
    assert _passage_location(str(tmp_path / "gone.py"), "anything") is None


def test_a_minified_line_is_not_reached_back_into(tmp_path):
    from langgraph_agent.graphrag_server import MAX_PASSAGE_LEAD_CHARS, _passage_location

    source = tmp_path / "bundle.js"
    source.write_text("x" * (MAX_PASSAGE_LEAD_CHARS + 50) + "needle();\n")

    assert _passage_location(str(source), "needle();") == (1, "needle();")


class _RecordingEmbedder(_FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return super().encode(text, **kwargs)


def test_embedding_carries_the_runs_stop_check(tmp_path):
    """A stopped run waits for at most one batch, not for the rest of a document.

    Both call sites go through `_encode`, so neither a document nor a query
    reaches the daemon without the run's stop check attached.
    """
    kb = _make_kb(tmp_path)
    embedder = _RecordingEmbedder()
    kb._embedder = embedder  # type: ignore[assignment]
    kb._should_stop = lambda: False

    kb.add_document("notes.md", "Some words to embed. " * 30, {"path": "notes.md", "type": "markdown"})
    kb.search("words", 1)

    assert len(embedder.calls) >= 2
    assert all(call.get("should_stop") is kb._should_stop for call in embedder.calls)


def test_embedding_goes_in_batches_of_the_cap(tmp_path):
    """8 passages per request is the batch OLLAMA_EMBED_OPTIONS was measured at.

    Both call sites go through `_encode`, so neither a document nor a query
    reaches the daemon at any other batch size.
    """
    kb = _make_kb(tmp_path)
    embedder = _RecordingEmbedder()
    kb._embedder = embedder  # type: ignore[assignment]

    kb.add_document("notes.md", "Some words to embed. " * 30, {"path": "notes.md", "type": "markdown"})
    kb.search("words", 1)

    assert len(embedder.calls) >= 2
    assert all(call.get("batch_size") == EMBEDDING_BATCH_SIZE for call in embedder.calls)


def test_a_tokenizer_nobody_can_supply_says_why_in_one_sentence(monkeypatch):
    """transformers' own wording blamed "a local directory with the same name";
    the proxy's 403 underneath it was the cause, and the fix is the network."""
    import sys
    import types

    from langgraph_agent.graphrag_server import (
        EMBEDDING_TOKENIZER_NAME,
        OllamaEmbedder,
        TokenizerUnavailable,
    )

    class ProxyError(Exception):
        pass

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, local_files_only=False):
            if local_files_only:
                raise OSError("not in the local cache")
            try:
                raise ProxyError("403 Forbidden")
            except ProxyError as cause:
                raise OSError(f"Can't load the configuration of '{name}'. If you were "
                              "trying to load it from 'https://huggingface.co/models', make "
                              "sure you don't have a local directory with the same name.") from cause

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))

    with pytest.raises(TokenizerUnavailable) as raised:
        _ = OllamaEmbedder(EMBEDDING_TOKENIZER_NAME).tokenizer
    message = str(raised.value)
    assert "ProxyError: 403 Forbidden" in message
    assert "local directory" not in message
    assert isinstance(raised.value, OSError), "callers that catch OSError still do"


def test_a_tokenizer_failure_names_the_innermost_cause_and_keeps_only_words(monkeypatch):
    """transformers wraps the hub's LocalEntryNotFoundError, which wraps the
    HTTP client's error, which wraps the socket's: the first non-OSError was the
    hub's wrapper ("An error happened while trying to locate the file"), not the
    name that did not resolve. And the live exception -- traceback, frames, the
    client's response -- was kept for as long as the hub stayed blocked."""
    import socket
    import sys
    import types

    from langgraph_agent import graphrag_server
    from langgraph_agent.graphrag_server import (
        EMBEDDING_TOKENIZER_NAME,
        OllamaEmbedder,
        TokenizerUnavailable,
    )

    class LocalEntryNotFoundError(OSError):
        pass

    class RequestsConnectionError(OSError):
        pass

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, local_files_only=False):
            if local_files_only:
                raise OSError("not in the local cache")
            try:
                try:
                    try:
                        raise socket.gaierror(-2, "Name or service not known")
                    except socket.gaierror as dns:
                        raise RequestsConnectionError("Max retries exceeded") from dns
                except RequestsConnectionError as http:
                    raise LocalEntryNotFoundError(
                        "An error happened while trying to locate the file on the Hub"
                    ) from http
            except LocalEntryNotFoundError as hub:
                raise OSError("We couldn't connect to 'https://huggingface.co'") from hub

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))
    embedder = OllamaEmbedder(EMBEDDING_TOKENIZER_NAME)

    with pytest.raises(TokenizerUnavailable) as raised:
        _ = embedder.tokenizer
    message = str(raised.value)
    assert "gaierror: [Errno -2] Name or service not known" in message
    assert "LocalEntryNotFoundError" not in message

    when, reason = graphrag_server._tokenizer_failed
    assert reason == "gaierror: [Errno -2] Name or service not known"
    # Asked again inside the window: the same words, and no old exception behind them.
    with pytest.raises(TokenizerUnavailable) as again:
        _ = embedder.tokenizer
    assert str(again.value) == message
    assert again.value.__cause__ is None


def test_a_tokenizer_the_hub_refused_is_not_asked_for_again_file_by_file(monkeypatch):
    """A rebuild asks for the tokenizer once per file; each ask went back to the
    hub, which on a network that drops connections is a timeout per file."""
    import sys
    import types

    from langgraph_agent import graphrag_server
    from langgraph_agent.graphrag_server import (
        EMBEDDING_TOKENIZER_NAME,
        OllamaEmbedder,
        TokenizerUnavailable,
    )

    asked: list[bool] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, local_files_only=False):
            asked.append(local_files_only)
            raise OSError("no tokenizer here")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))
    embedder = OllamaEmbedder(EMBEDDING_TOKENIZER_NAME)
    for _ in range(5):
        with pytest.raises(TokenizerUnavailable):
            _ = embedder.tokenizer
    assert asked.count(False) == 1, "the hub was asked more than once"

    # Past the window the hub is asked again: a network put right is noticed.
    monkeypatch.setattr(graphrag_server, "TOKENIZER_RETRY_SECONDS", 0.0)
    with pytest.raises(TokenizerUnavailable):
        _ = embedder.tokenizer
    assert asked.count(False) == 2


def test_the_tokenizer_is_loaded_from_the_cache_before_the_network(monkeypatch):
    """Loading by name asked huggingface.co whenever the chunker first ran."""
    import sys
    import types

    calls: list[bool] = []
    cached = {"present": True}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only and not cached["present"]:
                raise OSError("not in the local cache")
            return _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))

    from langgraph_agent.graphrag_server import EMBEDDING_TOKENIZER_NAME, OllamaEmbedder

    assert isinstance(OllamaEmbedder(EMBEDDING_TOKENIZER_NAME).tokenizer, _FakeTokenizer)
    assert calls == [True]

    calls.clear()
    cached["present"] = False
    assert isinstance(OllamaEmbedder(EMBEDDING_TOKENIZER_NAME).tokenizer, _FakeTokenizer)
    assert calls == [True, False]
