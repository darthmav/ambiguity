"""Tests for uploading a document into the corpus.

The claim under test is not "the bytes reached Chroma". It is that an uploaded
document is a document *of the corpus* — same id shape, same metadata, and
inside the walk a reindex rebuilds from — because the failure this path invites
is silent in both directions: a document embedded and then pruned by the next
rebuild, or a file the embedder cannot read filed under a real filename.

Built the way `test_corpus_admin.py` builds its knowledge base: field by field
around a fake collection, so the embedding model never loads. `add_document` is
the one method here that would embed, so the fake knowledge base stands in for
it and records what it was asked to store.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

import serve
from langgraph_agent.graphrag_server import (
    INDEXABLE_SUFFIXES,
    MAX_INDEXABLE_BYTES,
    UPLOADS_DIR,
    _document_metadata,
    iter_corpus_files,
    store_uploaded_document,
)


class _RecordingKB:
    """What `store_uploaded_document` uses of a knowledge base, and nothing else.

    `add_document` returns a chunk count, so the fake returns one too: the
    report the console renders is built from it, and a fake that returned
    `None` would let a real regression there pass.
    """

    def __init__(self) -> None:
        self.added: list[tuple[str, str, dict[str, Any]]] = []
        self.graph = nx.DiGraph()

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any],
        *, on_commit: Callable[[], None] | None = None,
    ) -> int:
        self.added.append((doc_id, content, metadata))
        if on_commit is not None:
            on_commit()
        self.graph.add_node(doc_id, type="document")
        return max(1, len(content) // 100)

    def stats(self) -> dict[str, Any]:
        return {"total_documents": len(self.added), "total_chunks": len(self.added)}


@pytest.fixture
def kb() -> _RecordingKB:
    return _RecordingKB()


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path)


# ---------------------------------------------------------------------------
# The file is what makes the upload durable
# ---------------------------------------------------------------------------


def test_the_document_is_written_to_disk_before_it_is_embedded(kb, root):
    """The ordering is the design. A document that exists only in the store is
    deleted by the next reindex, which clears the graph and prunes every row
    whose document is not in the walk — silently, in a pass that reports
    success."""
    report = store_uploaded_document(kb, "notes.md", "The Architect rules.", root)

    stored = Path(root) / UPLOADS_DIR / "notes.md"
    assert stored.read_text(encoding="utf-8") == "The Architect rules."
    assert report["path"] == str(stored)
    assert kb.added[0][0] == str(stored)


def test_an_uploaded_document_survives_a_reindex(kb, tmp_path, monkeypatch):
    """The load-bearing test: the walk has to find what the upload wrote.

    Asserted through `iter_corpus_files`, so whatever decides the walk's
    contents is what is tested.
    """
    store_uploaded_document(kb, "handbook.md", "Retrieval is hybrid.", str(tmp_path))

    monkeypatch.chdir(tmp_path)
    walked = {str(path) for path in iter_corpus_files(".")}

    assert f"{UPLOADS_DIR}/handbook.md" in walked


def test_the_id_and_metadata_match_what_a_reindex_would_write(kb, tmp_path, monkeypatch):
    """Two spellings of the same document is two documents.

    The reindex keys on `str(path)` from its own walk. If an upload stores a
    different string — an absolute path, a `./` prefix — the rebuild adds the
    file again under the walk's spelling and leaves the upload's copy orphaned
    in the graph.
    """
    monkeypatch.chdir(tmp_path)
    store_uploaded_document(kb, "notes.md", "Chunking is why this is reachable.")

    walked = next(p for p in iter_corpus_files(".") if p.name == "notes.md")
    doc_id, _, metadata = kb.added[0]

    assert doc_id == str(walked)
    assert metadata == _document_metadata(walked)


class _FailingKB(_RecordingKB):
    """An embedder that cannot embed: the daemon down, the tokenizer missing."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any], **kwargs: Any
    ) -> int:
        raise OSError("the embedding tokenizer is not cached here")


def test_an_upload_that_fails_to_embed_leaves_nothing_behind(root):
    """The console says "not stored" when the call fails, so nothing may be.

    The file used to stay under uploads/ after the embed failed, and the next
    rebuild embedded the document the operator had been told was refused.
    """
    with pytest.raises(OSError, match="tokenizer"):
        store_uploaded_document(_FailingKB(), "notes.md", "The Architect rules.", root)

    assert not (Path(root) / UPLOADS_DIR / "notes.md").exists()


def test_a_re_upload_that_fails_to_embed_keeps_the_version_the_corpus_holds(kb, root):
    """A failed correction must not swap the file under the chunks of the old one."""
    store_uploaded_document(kb, "spec.md", "First draft.", root)

    with pytest.raises(OSError):
        store_uploaded_document(_FailingKB(), "spec.md", "Second draft.", root)

    assert (Path(root) / UPLOADS_DIR / "spec.md").read_text() == "First draft."


def test_a_write_that_fails_part_way_puts_the_previous_version_back(kb, root, monkeypatch):
    """The write truncates the good file before it writes; on a full disk it
    stopped part-way outside the rollback, leaving half a document for the next
    rebuild to embed while the corpus held the old one."""
    store_uploaded_document(kb, "spec.md", "First draft.", root)
    target = Path(root) / UPLOADS_DIR / "spec.md"

    def full_disk(self: Path, data: str, encoding: str | None = None) -> int:
        with open(self, "w", encoding=encoding) as handle:
            handle.write(data[:3])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full_disk)
    with pytest.raises(OSError, match="No space left"):
        store_uploaded_document(kb, "spec.md", "Second draft.", root)

    assert target.read_text() == "First draft."
    assert len(kb.added) == 1


class _StoredThenFailedKB(_RecordingKB):
    """The store committed the new version; the in-memory graph did not follow."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any],
        *, on_commit: Callable[[], None] | None = None,
    ) -> int:
        if on_commit is not None:
            on_commit()
        raise RuntimeError("placing the node failed")


def test_a_failure_after_the_store_committed_keeps_the_file_the_store_holds(kb, root):
    """Rolled back then, the file went back to the old version while the store
    kept the new chunks -- the mismatch the rollback exists to prevent."""
    store_uploaded_document(kb, "spec.md", "First draft.", root)

    with pytest.raises(RuntimeError, match="placing the node"):
        store_uploaded_document(_StoredThenFailedKB(), "spec.md", "Second draft.", root)

    assert (Path(root) / UPLOADS_DIR / "spec.md").read_text() == "Second draft."


def test_re_uploading_a_name_replaces_rather_than_accumulates(kb, root):
    """The ordinary way to correct a document, and it must not leave two.

    `add_document` deletes the document's previous chunks before the new ones
    land, so the store is clean; what this pins is that the upload keeps
    addressing it under one id and says it replaced something.
    """
    first = store_uploaded_document(kb, "spec.md", "First draft.", root)
    second = store_uploaded_document(kb, "spec.md", "Second draft.", root)

    assert first["replaced"] is False
    assert second["replaced"] is True
    assert first["path"] == second["path"]
    assert (Path(root) / UPLOADS_DIR / "spec.md").read_text() == "Second draft."
    assert kb.graph.number_of_nodes() == 1


def test_the_report_carries_the_passage_count(kb, root):
    """A document that landed whole and one the embedder read the header of are
    different outcomes, and the chunk count is the only thing that tells them
    apart. It used to be discarded by `add_document`."""
    report = store_uploaded_document(kb, "long.md", "x" * 1000, root)

    assert report["chunks"] == 10
    assert report["characters"] == 1000
    assert report["total_documents"] == 1


# ---------------------------------------------------------------------------
# What the corpus refuses, and why refusing beats accepting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../escape.md", "../../etc/passwd.md", "a/b/nested.md",
                                  "..\\..\\windows.md"])
def test_a_name_carrying_a_path_cannot_write_outside_the_upload_directory(kb, root, name):
    """Not only a filesystem concern: a file written outside `uploads/` is
    outside the walk too, so it would be embedded now and pruned by the next
    reindex."""
    store_uploaded_document(kb, name, "content", root)

    written = [p for p in Path(root).rglob("*") if p.is_file()]
    assert len(written) == 1
    assert written[0].parent == Path(root) / UPLOADS_DIR


@pytest.mark.parametrize("name", ["", ".", "..", "/", "notes"])
def test_a_name_that_is_not_a_filename_is_refused(kb, root, name):
    with pytest.raises(ValueError):
        store_uploaded_document(kb, name, "content", root)
    assert kb.added == []


def test_a_format_with_no_text_extractor_is_refused_rather_than_embedded(kb, root):
    """There is no PDF reader here. Accepting one embeds whatever its bytes
    decode to under a real filename, and it looks like a document in the corpus
    from then on — a fabricated source, which is exactly what `search` was
    fixed to stop returning."""
    with pytest.raises(ValueError, match="no text extractor"):
        store_uploaded_document(kb, "paper.pdf", "%PDF-1.7 garbage", root)

    assert kb.added == []
    assert not (Path(root) / UPLOADS_DIR).exists(), "refused before anything was written"


def test_an_upper_case_suffix_is_stored_lower_cased(kb, root):
    """The walk compares suffixes exactly, so `NOTES.MD` stored as given is a
    file the reindex cannot see."""
    report = store_uploaded_document(kb, "NOTES.MD", "Case matters to glob.", root)

    assert report["name"] == "NOTES.md"
    assert (Path(root) / UPLOADS_DIR / "NOTES.md").exists()


def test_a_file_over_the_reindex_limit_is_refused(kb, root):
    """Refused on the walk's own test, character for character. Accepting it
    would embed it now and skip it at every rebuild after — the document
    disappears from the corpus and nothing says so."""
    with pytest.raises(ValueError, match="the same limit a reindex applies"):
        store_uploaded_document(kb, "huge.md", "x" * (MAX_INDEXABLE_BYTES + 1), root)

    assert kb.added == []
    # And the boundary itself is accepted, or the two rules do not agree.
    store_uploaded_document(kb, "big.md", "x" * MAX_INDEXABLE_BYTES, root)
    assert len(kb.added) == 1


def test_binary_content_is_refused_however_it_is_named(kb, root):
    """The suffix gate catches a PDF called `paper.pdf`. This catches the same
    file renamed: a browser decodes it to text with replacement characters and
    null bytes, and no text document has a null byte in it."""
    with pytest.raises(ValueError, match="not text"):
        store_uploaded_document(kb, "paper.md", "PDF\x00\x00binary", root)
    assert kb.added == []


def test_an_empty_document_is_refused(kb, root):
    """`chunk_text` returns no chunks for whitespace, so this would add a graph
    node with nothing in the store behind it — a document that exists in the
    map and answers no search."""
    with pytest.raises(ValueError, match="nothing in it"):
        store_uploaded_document(kb, "blank.md", "   \n\t\n  ", root)
    assert kb.added == []


# ---------------------------------------------------------------------------
# The RPC layer
# ---------------------------------------------------------------------------


@pytest.fixture
def console(kb, tmp_path, monkeypatch):
    """Point the console's creating door at the fake, from a scratch root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serve, "kb", kb)
    with serve._run_lock:
        serve._run_progress.update(running=False, goal="", node="", run_id="")
    yield kb
    with serve._run_lock:
        serve._run_progress.update(running=False, run_id="")


def test_rpc_upload_stores_and_reports(console, tmp_path):
    result = serve.rpc_upload_document({"name": "notes.md", "content": "Hybrid search."})

    assert result["path"] == f"{UPLOADS_DIR}/notes.md"
    assert result["chunks"] >= 1
    assert (tmp_path / UPLOADS_DIR / "notes.md").exists()


def test_rpc_upload_is_refused_while_a_run_is_in_flight(console, tmp_path):
    """The third writer to the corpus, and it takes the same guard.

    Not for the reason the other two do — an upload only adds — but
    `add_document` mutates the graph the Researcher's search and the console's
    sweep iterate, and a run should be answered by the corpus it started
    against.
    """
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="port the console")

    with pytest.raises(ValueError, match="port the console"):
        serve.rpc_upload_document({"name": "notes.md", "content": "text"})

    assert console.added == []
    assert not (tmp_path / UPLOADS_DIR).exists()


def test_rpc_upload_refuses_params_that_are_not_a_name_and_text(console):
    with pytest.raises(ValueError, match="must be strings"):
        serve.rpc_upload_document({"name": "notes.md", "content": {"not": "text"}})
    with pytest.raises(ValueError, match="must be strings"):
        serve.rpc_upload_document({"name": ["notes.md"], "content": "text"})


def test_rpc_upload_goes_through_the_creating_door(monkeypatch, tmp_path):
    """An upload is a request for a corpus to hold the document, so it is
    allowed to bring one into being — like a reindex, and unlike every read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serve, "kb", None)
    built = _RecordingKB()
    monkeypatch.setattr(serve, "get_knowledge_base", lambda *a, **k: built)
    monkeypatch.setattr(serve, "open_knowledge_base", lambda *a, **k: None)
    with serve._run_lock:
        serve._run_progress.update(running=False, goal="")

    serve.rpc_upload_document({"name": "seed.md", "content": "The first document."})

    assert built.added, "the upload did not reach a knowledge base"


def test_the_method_is_registered_and_not_quiet():
    """An operator action that changes the corpus; the telemetry log has to
    show it, the way reindex and clear do."""
    assert serve.RPC_METHODS["upload_document"] is serve.rpc_upload_document
    assert "upload_document" not in serve.QUIET_METHODS


def test_the_console_is_told_the_accepted_suffixes():
    """The upload pickers take their list from `status` rather than the page.

    The page used to hard-code four types, and went stale the day the walk
    learned to read the console's own markup.
    """
    assert serve.rpc_status({})["indexable_suffixes"] == list(INDEXABLE_SUFFIXES)
    assert ".html" in INDEXABLE_SUFFIXES
