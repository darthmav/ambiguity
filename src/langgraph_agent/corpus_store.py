"""Where the corpus lives: PostgreSQL, with pgvector holding the embeddings.

One corpus is one schema. It holds the chunks with their vectors, the entity
graph's nodes and edges, and a one-row `corpus` table that says which embedding
model the vectors belong to and carries the relevance floor measured on them.
Everything that has to agree is written in one transaction: a document's
chunks and its place in the graph land together or not at all, and a clear
empties chunks, graph and floor at once.

The schema is named after the corpus directory a caller means (resolved, so the
host console and the container, which see different paths, never share one),
but nothing is written to that directory: the database is the store.

Search is exact. pgvector's approximate indexes stop at 2,000 dimensions for
`vector` and 4,000 for `halfvec`, and the embedding model answers in 4,096, so
an index would mean quantizing the vectors the relevance floor is read off.
A scan of this corpus's size is fast and returns the true nearest chunks.

Two doors, as before: `create_corpus_store` builds a schema and is reserved for
indexing; `open_corpus_store` returns None where there is none and is what every
read goes through.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypeVar

import networkx as nx
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from langgraph_agent.self_healing import Circuit, call_with_retry, exception_chain

# The database install.sh runs: postgres18, pgvector's build of postgres:18,
# published on loopback with trust authentication.
DEFAULT_DATABASE_URL = "postgresql://postgres@127.0.0.1:5432/postgres"

# Seconds a connection attempt may take before the database counts as
# unreachable.
POSTGRES_CONNECT_TIMEOUT_SECONDS = 5

# Idle connections the console keeps for reuse: a run, a rebuild, the status
# poll and a console search can each be holding one at once. More are opened
# when they are, and closed when they come back to a full pool.
POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "8"))

# The database as one circuit, opened only when it cannot be reached: a query
# the server refused is the server answering.
POSTGRES_CIRCUIT_THRESHOLD = 3
POSTGRES_CIRCUIT_COOLDOWN_SECONDS = float(os.getenv("POSTGRES_CIRCUIT_COOLDOWN_SECONDS", "15"))

# How often a transaction is attempted while the database cannot be reached,
# and the first wait. Safe to repeat: a transaction the connection dropped
# under was rolled back, so nothing of it happened.
POSTGRES_CONNECT_ATTEMPTS = 3
POSTGRES_RETRY_WAIT_SECONDS = 1.0

# How often a wait for another process's rebuild re-asks for the lock, which is
# also how quickly it notices a stop.
REBUILD_LOCK_POLL_SECONDS = 0.5

_T = TypeVar("_T")

# Prefix for every corpus schema, so a database shared with other work can
# tell them apart, and the test suite can drop its own.
SCHEMA_PREFIX = "kb_"


def database_url() -> str:
    """The database the corpus lives in: `DATABASE_URL`, else install.sh's."""
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def redacted_url(url: str) -> str:
    """`url` with any password replaced, for logs and the console."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", url)


# SQLSTATEs a server sends when it is the connection that failed, not the
# statement: class 08 (connection exception), the server shutting down or not
# yet accepting (57P01-57P03), and no connection slot left (53300).
_UNREACHABLE_SQLSTATES = ("08", "57P01", "57P02", "57P03", "53300")

# A transaction the server rolled back to break a deadlock or a serialization
# conflict: the server answering, and a second attempt can succeed.
_ROLLED_BACK_SQLSTATES = ("40001", "40P01")


def database_unreachable(exc: BaseException) -> bool:
    """Whether a failure means the database could not be reached at all.

    A connection refused, dropped or timed out: an `OperationalError` with no
    SQLSTATE (the client's own, no server said anything) or one of the
    connection states above. `OperationalError` also covers a deadlock, a
    statement timeout, a lock not granted and a full disk, which arrive over a
    working connection: the server answering, like a constraint or bad SQL.
    """
    for cause in exception_chain(exc):
        if isinstance(cause, psycopg.OperationalError):
            state = cause.sqlstate
            if state is None or state.startswith(_UNREACHABLE_SQLSTATES):
                return True
    return False


def transaction_rolled_back(exc: BaseException) -> bool:
    """Whether the server rolled a transaction back to resolve a conflict with another."""
    return any(
        isinstance(cause, psycopg.Error) and cause.sqlstate in _ROLLED_BACK_SQLSTATES
        for cause in exception_chain(exc)
    )


POSTGRES = Circuit(
    "postgres",
    failure_threshold=POSTGRES_CIRCUIT_THRESHOLD,
    recovery_timeout=POSTGRES_CIRCUIT_COOLDOWN_SECONDS,
    trips_on=database_unreachable,
)


def corpus_schema(persist_dir: str | Path) -> str:
    """The schema a corpus directory names: readable prefix, path-hash suffix.

    Resolved first, so two spellings of one directory are one corpus and two
    checkouts are two.
    """
    resolved = str(Path(persist_dir).resolve())
    slug = re.sub(r"[^a-z0-9]+", "_", Path(resolved).name.lower()).strip("_")[:24]
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{SCHEMA_PREFIX}{slug or 'corpus'}_{digest}"


def _configure(conn: psycopg.Connection[Any]) -> None:
    """Make every pooled connection speak pgvector.

    The extension is database-wide, so it is created only when the type is
    missing -- once per database, not once per connection. A role that may not
    create it still works against a database where someone else has.
    """
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.rollback()
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            register_vector(conn)
        except psycopg.Error:
            # No pgvector in this database: the health check says so by name,
            # rather than every connection failing as though it were unreachable.
            conn.rollback()
    conn.commit()


class CorpusDatabase:
    """Connections to one database, and the transaction this context is inside.

    The pool is a list of idle connections, not a library pool, for one
    reason: a library pool answers a refused connection by retrying it in the
    background while the caller waits out its timeout, so a database that is
    down cost every call seconds before the circuit could open. Here a
    connection is opened on demand, so a refusal raises at once. One that turns
    out to be broken is closed, and so is every idle one with it: they share
    the server whose restart broke it, so the retry opens a fresh connection
    rather than taking the next stale one.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._idle: list[psycopg.Connection[Any]] = []
        self._idle_lock = threading.Lock()
        # The connection of the transaction this context is inside, so nested
        # store calls join it rather than committing on their own.
        self._active: ContextVar[psycopg.Connection[Any] | None] = ContextVar(
            f"corpus-transaction-{id(self)}", default=None
        )

    def _take(self) -> psycopg.Connection[Any]:
        """An idle connection, or a new one, never waiting.

        Not through the circuit itself: every caller is already inside it, and
        a failed connect counted twice opened it at half its threshold.
        """
        with self._idle_lock:
            while self._idle:
                conn = self._idle.pop()
                if not conn.closed:
                    return conn

        conn = psycopg.connect(self.url, connect_timeout=POSTGRES_CONNECT_TIMEOUT_SECONDS)
        try:
            _configure(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _give_back(self, conn: psycopg.Connection[Any], broken: bool) -> None:
        """Keep a healthy, idle connection; close anything else."""
        if not broken and not conn.closed and conn.info.transaction_status == TransactionStatus.IDLE:
            with self._idle_lock:
                if len(self._idle) < POSTGRES_POOL_SIZE:
                    self._idle.append(conn)
                    return
        conn.close()

    def close(self) -> None:
        with self._idle_lock:
            idle, self._idle = self._idle, []
        for conn in idle:
            conn.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[psycopg.Connection[Any]]:
        """A connection inside a transaction this context now joins; committed on exit."""
        conn = self._take()
        broken = True
        try:
            with conn.transaction():
                token = self._active.set(conn)
                try:
                    yield conn
                finally:
                    self._active.reset(token)
            broken = False
        except psycopg.Error as exc:
            # Rolled back; the connection is still good unless the failure was
            # the connection itself.
            broken = database_unreachable(exc)
            raise
        except BaseException:
            broken = False
            raise
        finally:
            self._give_back(conn, broken)
            if broken:
                self.close()

    def run(self, work: Callable[[psycopg.Connection[Any]], _T], *, name: str) -> _T:
        """`work(conn)` inside one transaction, committed when it returns.

        Inside a transaction already open on this context, `work` joins it and
        commits with it. Otherwise it goes through the `POSTGRES` circuit -- one
        count per attempt -- and is retried briefly while the database cannot
        be reached, or when the server rolled it back to break a deadlock.
        """
        active = self._active.get()
        if active is not None:
            return work(active)

        def attempt() -> _T:
            with self._transaction() as conn:
                return work(conn)

        return call_with_retry(
            lambda: POSTGRES.call(attempt),
            max_attempts=POSTGRES_CONNECT_ATTEMPTS,
            min_wait=POSTGRES_RETRY_WAIT_SECONDS,
            max_wait=POSTGRES_RETRY_WAIT_SECONDS * 2,
            retry_if=lambda exc: database_unreachable(exc) or transaction_rolled_back(exc),
            name=f"postgres:{name}",
        )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[Any]]:
        """One transaction every store call inside the block joins.

        Not retried: the block is the caller's code, which may have done
        things a second attempt would repeat. Through the `POSTGRES` circuit,
        as one call.
        """
        active = self._active.get()
        if active is not None:
            yield active
            return
        with POSTGRES.guarding(), self._transaction() as conn:
            yield conn

    def health(self) -> dict[str, str]:
        """`{status, details}` for the monitor, through the circuit."""
        def ask(conn: psycopg.Connection[Any]) -> tuple[str, str | None]:
            row = conn.execute(
                "SELECT current_setting('server_version'),"
                " (SELECT extversion FROM pg_extension WHERE extname = 'vector')"
            ).fetchone()
            assert row is not None
            return str(row[0]), row[1]

        try:
            version, vector = self.run(ask, name="health")
        except Exception as exc:
            return {
                "status": "unhealthy",
                "details": f"{redacted_url(self.url)}: {type(exc).__name__}: {exc}",
            }
        if vector is None:
            return {
                "status": "unhealthy",
                "details": (
                    f"PostgreSQL {version} at {redacted_url(self.url)} has no pgvector: "
                    "run it from pgvector/pgvector:pg18-trixie (./install.sh does)"
                ),
            }
        return {
            "status": "healthy",
            "details": f"PostgreSQL {version}, pgvector {vector} at {redacted_url(self.url)}",
        }


_databases: dict[str, CorpusDatabase] = {}
_databases_lock = threading.Lock()


def get_database(url: str | None = None) -> CorpusDatabase:
    """The one pool per database URL this process holds."""
    url = url or database_url()
    with _databases_lock:
        database = _databases.get(url)
        if database is None:
            database = _databases[url] = CorpusDatabase(url)
        return database


def close_databases() -> None:
    """Close every pool, for the console's exit and the test suite's."""
    with _databases_lock:
        databases = list(_databases.values())
        _databases.clear()
    for database in databases:
        database.close()


# Idle connections are closed at exit rather than dropped on the server.
atexit.register(close_databases)


def _stored_document_id(chunk_id: str, metadata: Mapping[str, Any] | None) -> str:
    """The document a row belongs to: its `doc_id` metadata, else its id."""
    if metadata:
        doc_id = metadata.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            return doc_id
    return chunk_id


class PgCorpusStore:
    """One corpus: its chunks, its entity graph and its floor, in one schema.

    The chunk half answers the slice of a vector-collection API the knowledge
    base reads with -- `count`, `get`, `query`, `upsert`, rows shaped
    `{"ids": [...], "documents": [...], ...}` -- and adds the writes only a
    database gives: replacing a document's chunks in one transaction, pruning
    by set difference, and reading every document's fingerprint in one query.
    """

    def __init__(
        self,
        database: CorpusDatabase,
        schema: str,
        *,
        persist_dir: str | Path,
        embedding_model: str,
        dimensions: int,
    ) -> None:
        self.database = database
        self.schema = schema
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.dimensions = dimensions

    # -- plumbing -----------------------------------------------------------

    def _table(self, name: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(self.schema), sql.Identifier(name))

    def _q(self, template: str, **tables: str) -> sql.Composed:
        """`template` with `{name}` placeholders filled by this schema's tables."""
        return sql.SQL(template).format(**{k: self._table(v) for k, v in tables.items()})

    def transaction(self) -> contextlib.AbstractContextManager[psycopg.Connection[Any]]:
        """One transaction every call on this store inside the block joins."""
        return self.database.transaction()

    def _create(self, conn: psycopg.Connection[Any]) -> None:
        """The schema's DDL, serialized against another process creating it."""
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"create:{self.schema}",))
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
        conn.execute(self._q(
            """
            CREATE TABLE IF NOT EXISTS {corpus} (
                singleton       boolean PRIMARY KEY DEFAULT true CHECK (singleton),
                persist_dir     text NOT NULL,
                embedding_model text NOT NULL,
                dimensions      integer NOT NULL,
                created_at      timestamptz NOT NULL DEFAULT now(),
                floor_record    jsonb
            )
            """,
            corpus="corpus",
        ))
        conn.execute(
            self._q(
                "INSERT INTO {corpus} (persist_dir, embedding_model, dimensions)"
                " VALUES (%s, %s, %s) ON CONFLICT (singleton) DO NOTHING",
                corpus="corpus",
            ),
            (str(self.persist_dir.resolve()), self.embedding_model, self.dimensions),
        )
        conn.execute(self._q(
            f"""
            CREATE TABLE IF NOT EXISTS {{chunks}} (
                id          text PRIMARY KEY,
                doc_id      text NOT NULL,
                chunk_index integer NOT NULL,
                content     text NOT NULL,
                metadata    jsonb NOT NULL,
                embedding   vector({int(self.dimensions)}) NOT NULL
            )
            """,
            chunks="chunks",
        ))
        # A document's chunks are replaced, pruned and fingerprinted by doc_id.
        conn.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (doc_id, chunk_index)").format(
            sql.Identifier("chunks_doc_id"), self._table("chunks")))
        # Nothing queries by metadata any more; a schema built when something
        # did loses the index every chunk write was paying for.
        conn.execute(sql.SQL("DROP INDEX IF EXISTS {}.{}").format(
            sql.Identifier(self.schema), sql.Identifier("chunks_metadata")))
        conn.execute(self._q(
            "CREATE TABLE IF NOT EXISTS {nodes} ("
            " id text PRIMARY KEY, attrs jsonb NOT NULL DEFAULT '{{}}'::jsonb)",
            nodes="graph_nodes",
        ))
        conn.execute(self._q(
            """
            CREATE TABLE IF NOT EXISTS {edges} (
                source text NOT NULL REFERENCES {nodes} (id) ON DELETE CASCADE,
                target text NOT NULL REFERENCES {nodes} (id) ON DELETE CASCADE,
                attrs  jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (source, target)
            )
            """,
            edges="graph_edges", nodes="graph_nodes",
        ))
        # Entities are reached from the documents that mention them.
        conn.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (target)").format(
            sql.Identifier("graph_edges_target"), self._table("graph_edges")))

    def _identity(self, conn: psycopg.Connection[Any]) -> tuple[str, int] | None:
        """(model, dimensions) the schema was built for; None if there is no schema."""
        exists = conn.execute(
            "SELECT to_regclass(%s)", (f'"{self.schema}".corpus',)
        ).fetchone()
        if exists is None or exists[0] is None:
            return None
        row = conn.execute(
            self._q("SELECT embedding_model, dimensions FROM {corpus}", corpus="corpus")
        ).fetchone()
        return (str(row[0]), int(row[1])) if row else None

    def _matches(self, identity: tuple[str, int]) -> bool:
        return identity == (self.embedding_model, self.dimensions)

    def drop(self) -> None:
        """Remove the whole schema. For a corpus of another model, and for tests."""
        self.database.run(
            lambda conn: conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema))
            ),
            name="drop",
        )

    # -- the chunk half, in the collection API ------------------------------

    def count(self) -> int:
        def ask(conn: psycopg.Connection[Any]) -> int:
            row = conn.execute(self._q("SELECT count(*) FROM {chunks}", chunks="chunks")).fetchone()
            return int(row[0]) if row else 0

        return self.database.run(ask, name="count")

    def get(
        self,
        ids: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Rows (every one, or by id), in document order; ids always, the rest on request.

        `include` names `documents`, `metadatas` and `embeddings`; left out, it
        is documents and metadatas.
        """
        wanted = set(include if include is not None else ("documents", "metadatas"))
        params: list[Any] = []
        where = sql.SQL("")
        if ids is not None:
            where = sql.SQL(" WHERE id = ANY(%s)")
            params.append(list(ids))
        query = sql.SQL("SELECT id, content, metadata, {} FROM {}{} ORDER BY doc_id, chunk_index").format(
            sql.SQL("embedding" if "embeddings" in wanted else "NULL"),
            self._table("chunks"),
            where,
        )

        rows = self.database.run(lambda conn: conn.execute(query, params).fetchall(), name="get")
        out: dict[str, Any] = {"ids": [r[0] for r in rows]}
        if "documents" in wanted:
            out["documents"] = [r[1] for r in rows]
        if "metadatas" in wanted:
            out["metadatas"] = [r[2] for r in rows]
        if "embeddings" in wanted:
            out["embeddings"] = [r[3] for r in rows]
        return out

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """The `n_results` nearest chunks to each query, by exact cosine distance."""
        def ask(conn: psycopg.Connection[Any]) -> list[list[tuple[Any, ...]]]:
            statement = self._q(
                "SELECT id, content, metadata, embedding <=> %s AS distance"
                " FROM {chunks} ORDER BY distance, id LIMIT %s",
                chunks="chunks",
            )
            return [
                conn.execute(statement, (np.asarray(e, dtype=np.float32), n_results)).fetchall()
                for e in query_embeddings
            ]

        answers = self.database.run(ask, name="query")
        return {
            "ids": [[r[0] for r in rows] for rows in answers],
            "documents": [[r[1] for r in rows] for rows in answers],
            "metadatas": [[r[2] for r in rows] for rows in answers],
            "distances": [[float(r[3]) for r in rows] for rows in answers],
        }

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Insert rows, replacing any with the same id."""
        rows = [
            (
                chunk_id,
                _stored_document_id(chunk_id, metadatas[i]),
                int(metadatas[i].get("chunk_index", 0)),
                documents[i],
                Jsonb(dict(metadatas[i])),
                np.asarray(embeddings[i], dtype=np.float32),
            )
            for i, chunk_id in enumerate(ids)
        ]
        statement = self._q(
            "INSERT INTO {chunks} (id, doc_id, chunk_index, content, metadata, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id,"
            " chunk_index = EXCLUDED.chunk_index, content = EXCLUDED.content,"
            " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
            chunks="chunks",
        )

        def write(conn: psycopg.Connection[Any]) -> None:
            with conn.cursor() as cursor:
                cursor.executemany(statement, rows)

        if rows:
            self.database.run(write, name="upsert")

    # -- what only a database gives ----------------------------------------

    def replace_document(
        self,
        doc_id: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """A document's chunks, swapped for these in one transaction.

        A file that shrank leaves no old tail behind, and a failure anywhere
        leaves the chunks the store held.
        """
        def swap(conn: psycopg.Connection[Any]) -> None:
            conn.execute(
                self._q("DELETE FROM {chunks} WHERE doc_id = %s OR id = %s", chunks="chunks"),
                (doc_id, doc_id),
            )
            self.upsert(ids, embeddings, documents, metadatas)

        self.database.run(swap, name="replace_document")

    def prune_documents(self, keep: Collection[str]) -> list[str]:
        """Delete every chunk whose document is not in `keep`; the documents removed."""
        def prune(conn: psycopg.Connection[Any]) -> list[str]:
            rows = conn.execute(
                self._q(
                    "DELETE FROM {chunks} WHERE NOT (doc_id = ANY(%s)) RETURNING doc_id",
                    chunks="chunks",
                ),
                (list(keep),),
            ).fetchall()
            return sorted({str(r[0]) for r in rows})

        return self.database.run(prune, name="prune_documents")

    def prune_and_fingerprint(self, keep: Collection[str]) -> tuple[list[str], dict[str, str]]:
        """`prune_documents(keep)`, then `fingerprints()` of what survived, in one transaction.

        A store call like any other -- retried while the database cannot be
        reached -- since doing it twice changes nothing a first time did not.
        """
        return self.database.run(
            lambda conn: (self.prune_documents(keep), self.fingerprints()),
            name="prune_and_fingerprint",
        )

    def fingerprints(self) -> dict[str, str]:
        """Each stored document's content fingerprint, from its first chunk.

        One per document is enough: a document's chunks are only ever written
        together, so they all carry the same one.
        """
        def ask(conn: psycopg.Connection[Any]) -> dict[str, str]:
            rows = conn.execute(self._q(
                "SELECT DISTINCT ON (doc_id) doc_id, metadata->>'sha' FROM {chunks}"
                " ORDER BY doc_id, chunk_index",
                chunks="chunks",
            )).fetchall()
            return {str(r[0]): str(r[1]) for r in rows if r[1]}

        return self.database.run(ask, name="fingerprints")

    # -- the graph ----------------------------------------------------------

    def load_graph(self) -> nx.DiGraph:
        """The entity graph as stored."""
        def ask(conn: psycopg.Connection[Any]) -> nx.DiGraph:
            graph = nx.DiGraph()
            for node_id, attrs in conn.execute(
                self._q("SELECT id, attrs FROM {nodes} ORDER BY id", nodes="graph_nodes")
            ):
                graph.add_node(node_id, **attrs)
            for source, target, attrs in conn.execute(self._q(
                "SELECT source, target, attrs FROM {edges} ORDER BY source, target",
                edges="graph_edges",
            )):
                graph.add_edge(source, target, **attrs)
            return graph

        return self.database.run(ask, name="load_graph")

    def save_graph(self, graph: nx.DiGraph) -> None:
        """Replace the stored graph with `graph`, whole or not at all."""
        def write(conn: psycopg.Connection[Any]) -> None:
            conn.execute(self._q("TRUNCATE {edges}, {nodes}", edges="graph_edges", nodes="graph_nodes"))
            with conn.cursor().copy(
                self._q("COPY {nodes} (id, attrs) FROM STDIN", nodes="graph_nodes")
            ) as copy:
                for node_id, attrs in graph.nodes(data=True):
                    copy.write_row((node_id, Jsonb(dict(attrs))))
            with conn.cursor().copy(
                self._q("COPY {edges} (source, target, attrs) FROM STDIN", edges="graph_edges")
            ) as copy:
                for source, target, attrs in graph.edges(data=True):
                    copy.write_row((source, target, Jsonb(dict(attrs))))

        self.database.run(write, name="save_graph")

    def save_document_graph(
        self, doc_id: str, attrs: Mapping[str, Any], entities: Collection[str]
    ) -> None:
        """One document's node and its `mentions` edges, replacing what it had."""
        def write(conn: psycopg.Connection[Any]) -> None:
            conn.execute(
                self._q(
                    "INSERT INTO {nodes} (id, attrs) VALUES (%s, %s)"
                    " ON CONFLICT (id) DO UPDATE SET attrs = EXCLUDED.attrs",
                    nodes="graph_nodes",
                ),
                (doc_id, Jsonb(dict(attrs))),
            )
            conn.execute(
                self._q("DELETE FROM {edges} WHERE source = %s", edges="graph_edges"), (doc_id,)
            )
            names = sorted(set(entities))
            if not names:
                return
            # An id already a document keeps its attributes.
            conn.execute(
                self._q(
                    "INSERT INTO {nodes} (id, attrs)"
                    " SELECT unnest(%s::text[]), '{{\"type\": \"entity\"}}'::jsonb"
                    " ON CONFLICT (id) DO NOTHING",
                    nodes="graph_nodes",
                ),
                (names,),
            )
            conn.execute(
                self._q(
                    "INSERT INTO {edges} (source, target, attrs)"
                    " SELECT %s, unnest(%s::text[]), '{{\"relation\": \"mentions\"}}'::jsonb",
                    edges="graph_edges",
                ),
                (doc_id, names),
            )

        self.database.run(write, name="save_document_graph")

    # -- the floor ----------------------------------------------------------

    def floor_record(self) -> dict[str, Any] | None:
        """The relevance-floor measurement stored with this corpus, if any."""
        def ask(conn: psycopg.Connection[Any]) -> Any:
            row = conn.execute(
                self._q("SELECT floor_record FROM {corpus}", corpus="corpus")
            ).fetchone()
            return row[0] if row else None

        record = self.database.run(ask, name="floor_record")
        return record if isinstance(record, dict) else None

    def set_floor_record(self, record: Mapping[str, Any] | None) -> None:
        self.database.run(
            lambda conn: conn.execute(
                self._q("UPDATE {corpus} SET floor_record = %s", corpus="corpus"),
                (Jsonb(dict(record)) if record is not None else None,),
            ),
            name="set_floor_record",
        )

    # -- all of it ----------------------------------------------------------

    def clear(self) -> dict[str, Any]:
        """Empty chunks, graph and floor in one transaction; what was there."""
        def wipe(conn: psycopg.Connection[Any]) -> dict[str, Any]:
            # Locked first, so nothing lands between the count and the wipe.
            conn.execute(self._q(
                "LOCK TABLE {chunks}, {nodes}, {edges}, {corpus} IN ACCESS EXCLUSIVE MODE",
                chunks="chunks", nodes="graph_nodes", edges="graph_edges", corpus="corpus",
            ))
            counted = conn.execute(self._q(
                "SELECT (SELECT count(*) FROM {chunks}),"
                " (SELECT floor_record IS NOT NULL FROM {corpus})",
                chunks="chunks", corpus="corpus",
            )).fetchone()
            assert counted is not None
            conn.execute(self._q(
                "TRUNCATE {chunks}, {edges}, {nodes}",
                chunks="chunks", edges="graph_edges", nodes="graph_nodes",
            ))
            conn.execute(self._q("UPDATE {corpus} SET floor_record = NULL", corpus="corpus"))
            return {"removed_chunks": int(counted[0]), "removed_floor": bool(counted[1])}

        return self.database.run(wipe, name="clear")


@contextlib.contextmanager
def rebuild_claim(
    persist_dir: str | Path,
    wait_seconds: float,
    should_stop: Callable[[], bool],
    waiting: Callable[[], None],
    *,
    poll_seconds: float = REBUILD_LOCK_POLL_SECONDS,
    url: str | None = None,
) -> Iterator[bool]:
    """Hold the right to rebuild a corpus across every process that uses it.

    A session advisory lock on a connection of its own, keyed by the corpus's
    schema: the server releases it when that connection ends, so a console that
    died holds nothing, and taking it creates nothing. Polled, not blocking, so
    the emergency stop reaches a waiter; `waiting` is called once, on the first
    refusal. Yields False when another process still holds it after
    `wait_seconds`. Raises when the database cannot be reached.
    """
    key = f"rebuild:{corpus_schema(persist_dir)}"
    conn: psycopg.Connection[Any] = POSTGRES.call(
        lambda: psycopg.connect(
            url or database_url(),
            autocommit=True,
            connect_timeout=POSTGRES_CONNECT_TIMEOUT_SECONDS,
        )
    )
    try:
        deadline = time.monotonic() + wait_seconds
        announced = False
        granted = False
        while True:
            row = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (key,)).fetchone()
            if row and row[0]:
                granted = True
                break
            if not announced:
                waiting()
                announced = True
            if should_stop() or time.monotonic() >= deadline:
                break
            time.sleep(poll_seconds)
        yield granted
    finally:
        # Closing the session releases the lock with it.
        conn.close()


def _store_for(
    persist_dir: str | Path, embedding_model: str, dimensions: int, url: str | None
) -> PgCorpusStore:
    return PgCorpusStore(
        get_database(url),
        corpus_schema(persist_dir),
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )


def create_corpus_store(
    persist_dir: str | Path, *, embedding_model: str, dimensions: int, url: str | None = None
) -> PgCorpusStore:
    """The corpus's store, **created if it does not exist** -- the indexing door.

    A schema built for another embedding model is dropped and rebuilt: its
    vectors share no space with this model's, so nothing in it could be
    searched, and its floor was measured on them.
    """
    store = _store_for(persist_dir, embedding_model, dimensions, url)

    def build(conn: psycopg.Connection[Any]) -> tuple[str, int] | None:
        identity = store._identity(conn)
        if identity is not None and not store._matches(identity):
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(store.schema)))
        store._create(conn)
        return identity

    identity = store.database.run(build, name="create")
    if identity is not None and not store._matches(identity):
        from langgraph_agent.self_healing import get_healing_logger

        get_healing_logger().warning(
            f"Corpus {store.schema} was embedded with {identity[0]} ({identity[1]} dims);"
            f" rebuilt empty for {embedding_model}.",
            action="corpus_rebuilt_for_model",
            function="corpus-store",
        )
    return store


def open_corpus_store(
    persist_dir: str | Path, *, embedding_model: str, dimensions: int, url: str | None = None
) -> PgCorpusStore | None:
    """The corpus's store if one was built for this model, None if not. Never creates.

    A schema of another model reads as no corpus: nothing in it answers this
    model's queries, and the next rebuild replaces it. Raises when the database
    cannot be reached, which is not the same answer as "no corpus".
    """
    store = _store_for(persist_dir, embedding_model, dimensions, url)
    identity = store.database.run(store._identity, name="open")
    if identity is None or not store._matches(identity):
        return None
    return store
