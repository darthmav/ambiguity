"""What `PgCorpusStore` adds to the collection API, built on a fake's own rows.

The corpus tests stand a small in-memory collection in for the store, each
holding only the slice of `count` / `get` / `delete` / `upsert` / `query` it
needs. The knowledge base also asks the store for what only a database gives
-- a transaction, a document replaced whole, a prune by set difference, the
graph and floor beside the chunks -- and this mixin answers those in terms of
the fake's own methods, so a test's rows mean the same thing either way. What
the real store does in SQL is tested against PostgreSQL in
`test_corpus_store.py`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any

import networkx as nx

from langgraph_agent.graphrag_server import _document_id_of


class StoreDoubleMixin:
    """The store's own methods, over `get` / `delete` / `upsert` / `count`."""

    saved_graph: nx.DiGraph | None = None
    floor: dict[str, Any] | None = None

    # Supplied by the fake the mixin is mixed into.
    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def upsert(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        yield None

    def replace_document(
        self,
        doc_id: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        stored = self.get(include=["metadatas"])
        doomed = [
            chunk_id
            for chunk_id, meta in zip(stored["ids"], stored.get("metadatas") or [], strict=False)
            if _document_id_of(chunk_id, meta) == doc_id
        ]
        if doomed:
            self.delete(ids=doomed)
        if ids:
            self.upsert(
                ids=list(ids), embeddings=list(embeddings), documents=list(documents),
                metadatas=list(metadatas),
            )

    def prune_documents(self, keep: Collection[str]) -> list[str]:
        stored = self.get(include=["metadatas"])
        metadatas = stored.get("metadatas") or [None] * len(stored["ids"])
        doomed: list[str] = []
        documents: set[str] = set()
        for chunk_id, meta in zip(stored["ids"], metadatas, strict=False):
            document = _document_id_of(chunk_id, meta)
            if document not in keep:
                doomed.append(chunk_id)
                documents.add(document)
        if doomed:
            self.delete(ids=doomed)
        return sorted(documents)

    def fingerprints(self) -> dict[str, str]:
        stored = self.get(include=["metadatas"])
        metadatas = stored.get("metadatas") or [None] * len(stored["ids"])
        return {
            _document_id_of(chunk_id, meta): str(meta["sha"])
            for chunk_id, meta in zip(stored["ids"], metadatas, strict=False)
            if meta and meta.get("sha")
        }

    def save_document_graph(
        self, doc_id: str, attrs: Mapping[str, Any], entities: Collection[str]
    ) -> None:
        pass

    def save_graph(self, graph: nx.DiGraph) -> None:
        self.saved_graph = graph.copy()

    def load_graph(self) -> nx.DiGraph:
        return self.saved_graph.copy() if self.saved_graph is not None else nx.DiGraph()

    def floor_record(self) -> dict[str, Any] | None:
        return self.floor

    def set_floor_record(self, record: Mapping[str, Any] | None) -> None:
        self.floor = dict(record) if record is not None else None

    def clear(self) -> dict[str, Any]:
        ids = list(self.get(include=[])["ids"])
        if ids:
            self.delete(ids=ids)
        removed_floor, self.floor = self.floor is not None, None
        self.saved_graph = nx.DiGraph()
        return {"removed_chunks": len(ids), "removed_floor": removed_floor}
