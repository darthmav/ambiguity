"""GraphRAG: the knowledge base -- a chunked vector store beside an entity graph.

Both halves live in PostgreSQL (`corpus_store`): chunks with pgvector
embeddings, and the graph's nodes and edges, one schema per corpus, written in
transactions that keep the two agreeing. Search is hybrid (dense retrieval re-ranked with BM25, see `lexical`), and the
graph links each document to the entities it mentions. The corpus is built
only by indexing (`get_knowledge_base`); every read goes through
`open_knowledge_base`, which never creates one.
"""

import functools
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, ParamSpec, TypeVar

import networkx as nx

from langgraph_agent.control import EMBEDDER_ACTIVITY
from langgraph_agent.corpus_spectral import (
    BOTTLENECK_CONDUCTANCE,
    DUPLICATE_BLOCK_PREFIX,
    DUPLICATE_CONTAINMENT,
    DUPLICATE_NAME_SIMILARITY,
    EIGENGAP_DECISIVENESS,
    MAX_AUTO_CLUSTERS,
    CorpusSpectralMixin,
)
from langgraph_agent.corpus_store import (
    PgCorpusStore,
    create_corpus_store,
    database_unreachable,
    open_corpus_store,
)
from langgraph_agent.lexical import (
    BM25Index,
    lexical_order,
    reciprocal_rank_fusion,
)
from langgraph_agent.projects import PROJECTS_DIR, embedded_projects, held_out_of_corpus
from langgraph_agent.self_healing import Circuit, CircuitOpenError, call_with_retry

# The embedding model that builds and searches the corpus, served by the local
# daemon. There is exactly one: vectors from two models share no space, so a
# corpus is only ever built and searched by the model it was built with.
# Placement is the daemon's; nothing here touches a card.
EMBEDDING_MODEL_NAME = "qwen3-embedding:latest"

# The length of its vectors, which the store's `vector(...)` column is typed
# with: a corpus is built for one model and one dimension, and refuses others.
EMBEDDING_DIMENSIONS = 4096

# The embedding model's own tokenizer, loaded in-process (a few MB, no weights)
# because the chunker needs one here and the daemon's is not reachable.
EMBEDDING_TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"

# Passages per `/api/embed` request: the batch `OLLAMA_EMBED_OPTIONS` was
# measured at, 6.5s for 8 full passages wholly on the cards.
EMBEDDING_BATCH_SIZE = 8


# Seconds one batch may take: 23.4s for 8 passages with 30% of the model on the
# CPU, and the first batch of a run also waits for the load.
OLLAMA_EMBED_TIMEOUT_SECONDS = 600.0

# How often a batch the daemon answered with a 5xx is sent again, and the first
# wait (doubling after it). A 5xx here is nearly always the model *load*
# failing -- `num_gpu` forces every layer onto the cards, so a load that meets a
# card something else still holds errors rather than splitting -- and the load
# succeeds once that is gone: three 500s then a clean load, 12s apart, at one
# startup. A 4xx is the daemon refusing the request -- a missing tag -- and is
# never retried; a daemon that cannot be reached is `retry_unreachable`'s.
OLLAMA_EMBED_LOAD_RETRIES = 4
OLLAMA_EMBED_RETRY_SECONDS = 5.0


def _failed_model_load(exc: BaseException) -> bool:
    import urllib.error

    return isinstance(exc, urllib.error.HTTPError) and exc.code >= 500


# The embedder's load, as a circuit: a whole retry schedule that ends in a 5xx
# opens it, since what holds the cards by then is not a model the arbiter can
# evict. On 2026-10-02 it was the desktop's share of the card driving the
# display, which a browser opened minutes earlier had grown past the 75 MiB the
# load left free: fifteen loads failed in four minutes, three pages' schedules
# back to back, and a 50-file rebuild would have spent over an hour the same
# way. While it is open every embed fails at once, so a rebuild stops after one
# document; the first embed after the cooldown is the trial, and one attempt
# answers it.
EMBEDDER_LOAD_COOLDOWN_SECONDS = 120.0
EMBEDDER_LOAD = Circuit(
    "embedder-load",
    failure_threshold=1,
    recovery_timeout=EMBEDDER_LOAD_COOLDOWN_SECONDS,
    trips_on=_failed_model_load,
)


# The window and batch the embedder is loaded with, sent on every call -- the
# daemon reloads a model whose options change, so a search at another window
# would evict the runner an index is using. At the default 4,096/2,048 the
# model asked for more than the cards hold and ran a third of its layers on the
# CPU; at 512 it fits wholly, twice as fast, with identical vectors (cosine
# 1.000000) -- and 512 is still twice the chunker's passages. The batch is the
# chunker's window rather than the load's: the last card holds the output layer
# and a compute buffer sized by the batch -- nearly all of it logits an
# embedding never reads -- and here that card also drives the display. Measured
# 2026-10-02 on two 3 GB GTX 1060s, at 512 the buffer took 338 MiB and llama.cpp
# projected 64 MiB left on the display card; at 256, 169 MiB and 250 MiB, with
# every chunk (254 tokens + EOS) and short query embedding bit-identically and
# a batch of 8 no slower. A longer input still embeds, in two passes, and its
# vector moves by under 3e-4 in cosine. `num_gpu: 999` forces every layer onto
# the cards: where the model does not fit the load errors rather than spilling,
# and a corpus is built on the placement it is searched on (a split load's
# vectors differ slightly).
OLLAMA_EMBED_OPTIONS: dict[str, int] = {"num_ctx": 512, "num_batch": 256, "num_gpu": 999}


class EmbeddingStopped(RuntimeError):
    """The run was stopped between two batches of an embedding."""


class EmbedderLoadFailed(RuntimeError):
    """The daemon could not load the embedding model through a whole retry schedule.

    The model's failure, not a passage's: it opened `EMBEDDER_LOAD`, which
    refuses every embed after it, so a pass over many documents ends here.
    """


class OllamaEmbedder:
    """An Ollama embedding model behind the two things the corpus asks of one.

    `encode` sends passages to `/api/embed` in batches and returns one vector per
    passage. `tokenizer` is the model's own, loaded from the local Hugging Face
    cache first so an offline machine still chunks; its 254-token passages sit
    well inside the daemon's window, so nothing is cut twice.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._tokenizer: Any = None
        # How much of the model the daemon left on the CPU when it last
        # embedded. The reply is identical either way, so this is the only
        # place a split that quarters the speed shows. None until a call has
        # loaded the model, or when the daemon cannot say.
        self.cpu_share: float | None = None

    @property
    def placement_note(self) -> str | None:
        """Why this model embeds slowly when the daemon split it onto the CPU; None otherwise."""
        if not self.cpu_share:
            return None
        percent = max(1, round(self.cpu_share * 100))
        return (
            f"Ollama holds {percent}% of {self.model} on the CPU, which embeds at a "
            f"fraction of the speed. It fits the cards at a {OLLAMA_EMBED_OPTIONS['num_ctx']}"
            "-token window when nothing else holds them; `ollama ps` shows what does."
        )

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            # The tokenizer is all this process takes from transformers, which
            # otherwise announces on import that it found no PyTorch: printed
            # just before an embed failed, it read as the cause.
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from transformers import AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    EMBEDDING_TOKENIZER_NAME, local_files_only=True
                )
            except (OSError, ValueError):
                self._tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER_NAME)
        return self._tokenizer

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> Any:
        """One vector per passage, or a single vector for a single string.

        Each batch goes through the daemon's circuit and the embedder's own. An
        unreachable daemon is retried briefly (`retry_unreachable`); a failed model
        load (5xx) is retried on the slower load schedule, and one that outlasts it
        opens `EMBEDDER_LOAD` and raises `EmbedderLoadFailed`; a 4xx is the daemon
        refusing and raises at once. `should_stop` is asked before each batch and
        through every wait, so a stopped run waits for at most one batch.
        """
        import urllib.error

        import numpy as np

        from langgraph_agent.config import daemon_request, ollama_cpu_share, retry_unreachable

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors: list[list[float]] = []
        step = max(1, batch_size)
        for start in range(0, len(items), step):
            if should_stop is not None and should_stop():
                raise EmbeddingStopped(f"stopped after {start} of {len(items)} passages")
            body = {"model": self.model, "input": items[start : start + step],
                    "options": OLLAMA_EMBED_OPTIONS}
            attempts = 0

            def embed(body: dict[str, Any] = body) -> Any:
                nonlocal attempts
                attempts += 1
                return retry_unreachable(
                    lambda: daemon_request(
                        "/api/embed", body, timeout=OLLAMA_EMBED_TIMEOUT_SECONDS
                    ),
                    name=f"embed:{self.model}",
                    give_up=should_stop,
                )

            # While the circuit is open this call is refused, or -- once the
            # cooldown is over -- is its trial: one attempt, since the schedule
            # waits out a card being freed and the load that opened the circuit
            # already outlasted one.
            tries = 1 if EMBEDDER_LOAD.is_open else OLLAMA_EMBED_LOAD_RETRIES + 1

            def schedule(
                embed: Callable[[], Any] = embed, start: int = start, tries: int = tries
            ) -> Any:
                try:
                    return call_with_retry(
                        embed,
                        max_attempts=tries,
                        min_wait=OLLAMA_EMBED_RETRY_SECONDS,
                        max_wait=OLLAMA_EMBED_RETRY_SECONDS * 4,
                        retry_if=_failed_model_load,
                        give_up=should_stop,
                        name=f"embed:{self.model}",
                    )
                except Exception as exc:
                    # Raised inside the circuit, so a schedule the stop cut
                    # short is not counted as a load that cannot be made.
                    if should_stop is not None and should_stop():
                        raise EmbeddingStopped(
                            f"stopped after {start} of {len(items)} passages"
                        ) from exc
                    raise

            try:
                payload = EMBEDDER_LOAD.call(schedule)
            except (CircuitOpenError, EmbeddingStopped):
                # Not this batch's failure: every caller stands down together.
                raise
            except Exception as exc:
                tried = f" (after {attempts} attempts)" if attempts > 1 else ""
                detail = (
                    exc.read().decode("utf-8", "replace").strip()
                    if isinstance(exc, urllib.error.HTTPError) else ""
                )
                message = f"Ollama could not embed with {self.model}{tried}: {detail or exc}"
                if _failed_model_load(exc):
                    raise EmbedderLoadFailed(message) from exc
                raise RuntimeError(message) from exc
            batch = body["input"]
            embeddings = payload.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} vectors for {len(batch)} "
                    f"passages from {self.model}"
                )
            vectors.extend(embeddings)
            if start == 0:
                # Asked once the first batch has loaded the model: the daemon
                # fits a reload around whatever holds the cards by then.
                self.cpu_share = ollama_cpu_share(self.model)
        array = np.asarray(vectors, dtype=np.float32)
        return array[0] if single else array


# Where a corpus lives when nobody says otherwise.
DEFAULT_PERSIST_DIR = "knowledge"


def resolve_persist_dir(persist_dir: str | Path | None = None) -> Path:
    """The corpus directory a caller means, defaulting to `DEFAULT_PERSIST_DIR`."""
    return Path(persist_dir or DEFAULT_PERSIST_DIR)


# The questions the relevance floor is measured with, in JSON because the walk
# never indexes JSON: indexed, the unanswerable ones would answer themselves.
FLOOR_CALIBRATION_QUESTIONS = Path(__file__).with_name("embedding_calibration.json")


def open_store(persist_dir: str | Path | None = None) -> PgCorpusStore | None:
    """The store of the corpus a caller means, or None if none was built. Never creates."""
    return open_corpus_store(
        resolve_persist_dir(persist_dir),
        embedding_model=EMBEDDING_MODEL_NAME,
        dimensions=EMBEDDING_DIMENSIONS,
    )


def floor_calibration(persist_dir: str | Path | None = None) -> dict[str, Any] | None:
    """The stored measurement behind the corpus's floor, or None if none was taken.

    Kept in the corpus's own row, so it is cleared in the transaction that
    clears the texts it was measured on. A database that cannot be reached
    has no floor to offer, which every reader already handles.
    """
    try:
        store = open_store(persist_dir)
        record = store.floor_record() if store is not None else None
    except Exception:
        return None
    return record if isinstance(record, dict) and record.get("model") == EMBEDDING_MODEL_NAME else None


def floor_from_calibration(record: dict[str, Any] | None) -> float | None:
    """The floor a stored record carries, or None when it carries none."""
    floor = record.get("floor") if record else None
    return float(floor) if isinstance(floor, (int, float)) else None


def relevance_floor(persist_dir: str | Path | None = None) -> float | None:
    """The score over which a search counts as the corpus having answered.

    What `calibrate_relevance_floor` measured on this corpus with this model --
    None until measured, and None for good if no gap was found. None means
    retrieval cannot tell an answer from noise; a cosine is never borrowed from
    another model.
    """
    return floor_from_calibration(floor_calibration(persist_dir))


def calibrate_relevance_floor(kb: "GraphRAGKnowledgeBase") -> dict[str, Any]:
    """Take the relevance floor for this corpus's embedding model, and keep it.

    Twelve questions this corpus answers against twelve it cannot. The floor is
    the midpoint of the gap between the lowest answered score and the highest
    unanswered one; when the populations overlap there is no floor, since any
    number inside the overlap would misfile some question silently.
    """
    questions = json.loads(FLOOR_CALIBRATION_QUESTIONS.read_text(encoding="utf-8"))

    def best(question: str) -> float:
        hits = kb.search(question, 1)
        return round(float(hits[0].get("score") or 0.0), 3) if hits else 0.0

    answered = [best(question) for question in questions["answered"]]
    unanswerable = [best(question) for question in questions["unanswerable"]]
    low, high = max(unanswerable), min(answered)
    record: dict[str, Any] = {
        "model": EMBEDDING_MODEL_NAME,
        "answered": answered,
        "unanswerable": unanswerable,
        "floor": round((low + high) / 2, 3) if high > low else None,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    kb.collection.set_floor_record(record)
    return record


# What every caller says when asked to search a corpus nobody has built, so an
# absent corpus never reads as an empty one.
NO_CORPUS_NOTE = (
    "No corpus has been indexed here, so there is nothing to retrieve. The "
    "corpus is the research archive -- pages online research kept, uploaded "
    "documents, and generated projects opted in -- and the console rebuilds it "
    "from there when it starts and before every run. Reaching this note during "
    "a run means the archive is empty, or REBUILD_CORPUS is off."
)


# Capitalised tokens that are not entities. `add_document` mints an entity for
# every capitalised word over four characters, and many capitals come from
# where a word sits rather than what it means: a sentence opener (`Every`), a
# docstring header (`Returns`), a report heading (`Status`), a literal
# (`False`). Unfiltered, `False` and `Returns` rank among the best-connected
# nodes, joining every document with a docstring through a relation that means
# nothing.
#
# A hand-audited list, not a rule: dropping tokens whose every capital is
# positional was measured and severs real relations (a term introduced in a
# bulleted list never appears elsewhere in its document). Candidates are
# nominated by counting position-free capitals and ruled on by hand, and the
# vocabulary drifts as prose is written -- `tests/test_claims.py` pins the
# audit so a newcomer fails the build instead of the graph.
ENTITY_STOPWORDS = frozenset(
    word.lower()
    for word in """
    Returns Return Parameters Parameter Example Examples Provides Raises Notes
    Args Arguments Attributes Yields Warns Usage Summary Description Overview

    Where Which There These Those Their Because Without Within While Since
    Should Would Could Might Cannot Every Nothing Never Always Something
    Anything Everything Instead Otherwise Rather Before After During Between
    Against About Almost Already Still Another Other First Second Third Above
    Below Under Twice Several Given Using Used Uses Value Values Result
    Results Input Inputs Output Outputs Simple Basic Total Default Defaults

    Check Checks Create Creates Compute Computes Number Numbers Optional
    Reading Reads Writes Written Files Status Makes Taken Keeps Change Changed
    Changes Adding Added Running Choose Chooses Verify Verifies Found Finds
    Being Doing Ensure Ensures Consider Include Includes Including Following
    Follows Contains Containing

    False None Import Class Print Assert Except Finally Raise Elif Return

    Measured Initialize Dense Build Maximum Empty Skipping Dictionary Apply
    Refused Whether Split Asserted Degree Shared Based Demonstrates References
    Generate Extract Point Seconds Deliberately Named Built Asking Pinned
    Insert Tests Write Reported Computed Cached Complete Convert Dimension
    Naming Perform Useful Prose Measure Observed Nodes Spectrum

    Hello Inference Three Verification Asked Entities Local Opening
    """.split()
)

# A chunk's length in the embedding model's tokens. A passage longer than the
# model's window is not embedded badly but silently truncated: embedded whole,
# 91.5% of this corpus was once unreachable by search, and long files lost to
# short ones that merely mentioned the answer.
CHUNK_MAX_TOKENS = 254

# Tokens carried from the end of one chunk into the next. Boundaries are cut on
# token counts, not sentences, and the overlap keeps a split passage whole in
# one chunk -- which is why boundaries are not snapped to line breaks, which
# would shorten a chunk or push it past the window.
CHUNK_OVERLAP_TOKENS = 48

# Separates a document id from its chunk number: `CLAUDE.md#0003`. A filename
# may contain "#" itself, so each chunk also carries `doc_id` in its metadata.
CHUNK_ID_SEPARATOR = "#"


def _content_sha(content: str) -> str:
    """A fingerprint of a document's text, stored on every chunk it becomes.

    What lets a rebuild keep the vectors it has: embedding is the only expensive
    part of indexing, so a rebuild where nothing changed never loads the model.
    A hash, not an mtime, since a branch switch moves timestamps without changing
    a byte. Truncated: it is compared, never trusted.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# Chunks pulled per requested result before collapsing them onto documents, so
# a query matching several chunks of one document still fills top_k.
SEARCH_CHUNK_OVERSAMPLE = 4

# The second and last rung of `search`'s ladder, for a query whose first window
# collapsed to fewer documents than asked for.
SEARCH_ESCALATION = 8


def _chunk_windows(
    n_tokens: int, max_tokens: int, overlap: int
) -> list[tuple[int, int]]:
    """Token index windows `[start, end)` covering `n_tokens`, with overlap.

    Every token lands in at least one window, and no window is longer than
    `max_tokens`. Pure arithmetic, testable without loading a tokenizer.
    """
    if n_tokens <= max_tokens:
        return [(0, n_tokens)] if n_tokens else []

    # An overlap wider than the window degrades to no overlap rather than a
    # loop that never advances.
    stride = max(max_tokens - overlap, 1)

    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_tokens, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            return windows
        start += stride


def _document_id_of(
    chunk_id: str, metadata: "Mapping[str, Any] | None" = None
) -> str:
    """The document a stored row belongs to.

    The chunk's `doc_id` metadata first; failing that, the id with a numeric
    `#NNNN` suffix stripped, so a path containing "#" is left alone.
    """
    if metadata:
        doc_id = metadata.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            return doc_id

    head, sep, tail = chunk_id.rpartition(CHUNK_ID_SEPARATOR)
    if sep and head and tail.isdigit():
        return head
    return chunk_id


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _embedder_at_work(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mark the embedder busy while `method` runs, for the console's embedder light.

    On `_load_embedder` and `_encode` -- everything the embedder does.
    """

    @functools.wraps(method)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with EMBEDDER_ACTIVITY.working():
            return method(*args, **kwargs)

    return run


def _needs_the_cards(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Hold `GPU_ARBITER` and free the cards while `method` embeds.

    Makes "nothing else runs while the embedder works" a property of the code:
    a console search or a node's own search embeds at any moment, not only in the
    pre-run phases. On `_encode` alone, not `_load_embedder`, which only builds
    the handle the in-process tokenizer is reached through -- evicting a seat to
    chunk a document would be a reload for nothing. The meter is entered inside
    the arbiter, so a batch queued behind a seat reads as waiting, not working.
    """

    @functools.wraps(method)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        from langgraph_agent.config import free_the_cards_for
        from langgraph_agent.control import GPU_ARBITER

        with GPU_ARBITER.exclusive("embedder"):
            # Freed inside the arbiter, so nothing loads into the gap.
            free_the_cards_for(EMBEDDING_MODEL_NAME)
            return method(*args, **kwargs)

    return run


def _document_node(
    doc_id: str, content: str, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any], set[str]]:
    """A document's node attributes and the entities it mentions.

    Pure, so the store and the in-memory graph are written from one answer.
    """
    # `type` on a node is structural -- document vs entity; the metadata's
    # own `type` would collide with it, so it goes in as `doc_type`.
    attrs: dict[str, Any] = {k: v for k, v in (metadata or {}).items() if k != "type"}
    if metadata and "type" in metadata:
        attrs["doc_type"] = metadata["type"]
    attrs["type"] = "document"
    attrs["content"] = content[:200]  # a snippet

    # Capitalised words over four characters, stripped of prose and code
    # punctuation and filtered through `ENTITY_STOPWORDS`.
    #
    # Fetched pages and markup, script and config files mint no entities
    # (`_mints_entities`): web prose opens sentences with a vocabulary the
    # hand-audited list was never checked against -- 15 pages once minted
    # 551 entities, a fifth of the graph -- and the graph's edges are read
    # as evidence. Those documents are still chunked, embedded and
    # retrievable.
    entities: set[str] = set()
    for word in (content.split() if _mints_entities(doc_id) else []):
        token = word.strip("\"'`()[]{}<>.,!?;:*=+-/\\|")
        if (
            len(token) > 4
            and token[0].isupper()
            and token.replace("_", "").isalnum()
            and token.lower() not in ENTITY_STOPWORDS
        ):
            entities.add(token)
    entities.discard(doc_id)
    return attrs, entities


class GraphRAGKnowledgeBase(CorpusSpectralMixin):
    """The corpus: a NetworkX document/entity graph beside a pgvector chunk store.

    `collection` is the corpus's `PgCorpusStore`: the chunks, read through the
    collection API search is written against, and the graph and floor beside
    them. `graph` is the working copy of the stored graph, loaded when the corpus
    opens and written back in the same transactions as the chunks.

    Constructing one **creates the store in the database**, which is why most callers go
    through `open_knowledge_base()`, which never builds one; `get_knowledge_base()`
    is reserved for indexing.
    """

    # Declared on the class, not in `__init__`, as the attributes below are:
    # the corpus tests build instances field by field around fakes, and these
    # must read as "not loaded yet" there.
    _embedder: "OllamaEmbedder | None" = None

    # (nodes, edges) -> the connectivity computed at that shape.
    _connectivity_cache: "tuple[tuple[int, int], dict[str, Any]] | None" = None

    # The lexical half of search, built on first use and dropped on every
    # change.
    _lexical_index: "BM25Index | None" = None

    # Where the loaded embedder sits (`ollama`), and a note when the daemon
    # split it onto the CPU.
    embedding_device: str | None = None
    embedding_device_note: str | None = None

    # The model this corpus is embedded with, carried so the export can say so.
    embedding_model: str = EMBEDDING_MODEL_NAME
    # Set by `index_corpus_files` while it runs, so an embed stops between
    # batches.
    _should_stop: "Callable[[], bool] | None" = None

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = resolve_persist_dir(persist_dir)
        self.collection: Any = create_corpus_store(
            self.persist_dir,
            embedding_model=EMBEDDING_MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
        )

        self.graph = nx.DiGraph()
        self._load_graph()

    @property
    def embedder(self) -> "OllamaEmbedder":
        """The daemon's embedding endpoint, built the first time something embeds, so
        opening the corpus to read it costs nothing.
        """
        model = self._embedder
        if model is None:
            model = self._load_embedder()
        return model

    @_embedder_at_work
    def _load_embedder(self) -> "OllamaEmbedder":
        """Point the corpus at the daemon; no weights load in this process."""
        model = OllamaEmbedder(self.embedding_model)
        self._embedder = model
        self.embedding_device = "ollama"
        self.embedding_device_note = None
        return model

    @_needs_the_cards
    @_embedder_at_work
    def _encode(self, texts: str | list[str]) -> Any:
        """Embed at `EMBEDDING_BATCH_SIZE`, stopping between batches when asked.

        Every encode goes through here.
        """
        model = self.embedder
        vectors = model.encode(
            texts, batch_size=EMBEDDING_BATCH_SIZE, should_stop=self._should_stop
        )
        # The reply is identical however the daemon placed the model; the note
        # is the only place a split onto the CPU shows.
        self.embedding_device_note = model.placement_note
        return vectors

    def _load_graph(self) -> None:
        """Load the stored graph into memory."""
        self.graph = self.collection.load_graph()

    def _save_graph(self) -> None:
        """Replace the stored graph with the one in memory, whole or not at all.

        One transaction: a process ending mid-write (a rebuild runs on a daemon
        thread) leaves the graph that was stored before it.
        """
        self.collection.save_graph(self.graph)

    def chunk_text(self, content: str) -> list[str]:
        """Split a document into passages the embedder can read whole.

        One tokenizer pass with an offset mapping, then `_chunk_windows`; each
        window's span runs from its first token's start to its last token's end, so
        the text between tokens is kept and the chunks rejoin into the document bar
        the overlap. `verbose=False` silences the "sequence too long" warning this
        function exists to answer.
        """
        if not content:
            return []

        tokenizer = self.embedder.tokenizer
        encoded = tokenizer(
            content,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            # Content the tokenizer maps to nothing: there is no passage to
            # embed.
            return []

        chunks = [
            content[offsets[start][0] : offsets[end - 1][1]]
            for start, end in _chunk_windows(
                len(offsets), CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            )
        ]
        return self._fit_chunks(chunks)

    def _fit_chunks(self, chunks: list[str]) -> list[str]:
        """Trim any chunk that re-tokenizes past the window, and choose the end.

        A fragment cut mid-word can re-tokenize one token longer standalone than it
        measured inside its document, so chunks are re-encoded and the overflow cut --
        the window is then a fact, not an estimate. Every chunk is trimmed at its tail,
        which a neighbour's overlap covers, except the last, trimmed at its head:
        trimming its tail would drop the end of the file.
        """
        if len(chunks) < 2:
            # A lone chunk is the whole document, un-sliced, and cannot
            # overflow.
            return chunks

        encoded = self.embedder.tokenizer(
            chunks,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        fitted: list[str] = []
        for i, (chunk, offsets) in enumerate(zip(chunks, encoded["offset_mapping"], strict=True)):
            if len(offsets) <= CHUNK_MAX_TOKENS:
                fitted.append(chunk)
            elif i == len(chunks) - 1:
                fitted.append(chunk[offsets[-CHUNK_MAX_TOKENS][0] :])
            else:
                fitted.append(chunk[: offsets[CHUNK_MAX_TOKENS - 1][1]])
        return fitted

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a document: its chunks to the store, one node and its entities to the graph.

        Chunks are stored as `doc_id#0000`, `doc_id#0001`, ...; the graph still gets
        one node per document, with entities drawn from the whole text.

        Returns:
            How many chunks the document became -- the number that says it is
            reachable past its first passage.
        """
        chunks = self.chunk_text(content)

        # Embedded before the transaction opens: a failed or stopped embed
        # leaves the document as the store held it, and no transaction is held
        # open across the slow part. One batched call for all the chunks.
        embeddings = self._encode(chunks).tolist() if chunks else []

        base = dict(metadata or {})
        # The fingerprint rides on every chunk, so a rebuild can tell an
        # unchanged document from the metadata it already fetches.
        base["sha"] = _content_sha(content)
        attrs, entities = _document_node(doc_id, content, metadata)

        # One transaction: the document's previous chunks go (a file that shrank
        # would otherwise leave its old tail matching queries), its new ones land,
        # and its node and edges are replaced -- all of it, or none.
        with self.collection.transaction():
            self.collection.replace_document(
                doc_id,
                ids=[f"{doc_id}{CHUNK_ID_SEPARATOR}{i:04d}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[
                    {**base, "doc_id": doc_id, "chunk_index": i, "chunk_count": len(chunks)}
                    for i in range(len(chunks))
                ],
            )
            self.collection.save_document_graph(doc_id, attrs, entities)

        # Memory follows the store only once the store has committed.
        self._place_in_graph(doc_id, attrs, entities)

        # The lexical index now describes a corpus that no longer exists; the
        # next search rebuilds it.
        self._lexical_index = None
        return len(chunks)

    def _add_to_graph(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """The half of `add_document` that costs nothing: the node and its entities.

        In memory only. Separate because a rebuild that keeps a document's
        vectors still has to put it back in the graph it cleared, and writes the
        rebuilt graph once at the end.
        """
        self._place_in_graph(doc_id, *_document_node(doc_id, content, metadata))

    def _place_in_graph(
        self, doc_id: str, attrs: dict[str, Any], entities: set[str]
    ) -> None:
        """Put one document's node and `mentions` edges in the in-memory graph.

        The edges replace any the document had, as the store's do.
        """
        if doc_id in self.graph:
            self.graph.remove_edges_from(list(self.graph.out_edges(doc_id)))
        self.graph.add_node(doc_id, **attrs)
        for entity in entities:
            if entity not in self.graph:
                self.graph.add_node(entity, type="entity")
            self.graph.add_edge(doc_id, entity, relation="mentions")


    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base: matches chunks, answers in documents.

        `id` is the document's path; `content` is the passage that matched (from the
        start of its line, with `line` set, when the file still contains it). Chunks
        are oversampled, re-ranked against BM25 (see `lexical.py`) and collapsed onto
        their documents, best chunk first, so `top_k` counts documents. `score` stays
        the dense cosine of the chunk that matched -- the number the relevance floor
        is read off.

        Args:
            query: Search query; an empty one answers nothing.
            top_k: Number of documents to return.
        """
        if top_k <= 0:
            return []
        # An empty query is not a question: it still embeds, and would match
        # something.
        if not query or not query.strip():
            return []
        # Nor is a store with nothing in it -- a cleared corpus, or one whose
        # first pages all failed to embed. Embedding the query anyway loaded
        # the model and evicted the seat holding the cards, to find nothing.
        if not self.collection.count():
            return []

        query_embedding = self._encode(query).tolist()

        # Widen once if one document monopolised the first window of hits; two
        # rungs, so one query cannot walk the corpus.
        results: list[dict[str, Any]] = []
        wanted = top_k * SEARCH_CHUNK_OVERSAMPLE
        for n_results in (wanted, wanted * SEARCH_ESCALATION):
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            results = self._collapse_chunk_hits(
                raw, top_k, order=self._rerank_lexically(query, raw)
            )
            hits = len((raw.get("ids") or [[]])[0] or [])
            if len(results) >= top_k or hits < n_results:
                # Enough documents, or the collection has no more chunks to
                # give.
                break

        return results

    @property
    def lexical_index(self) -> "BM25Index | None":
        """The BM25 index over the stored chunks, built from the store on first use.

        None, falling back to dense-only search, when the store cannot list its
        documents.
        """
        if self._lexical_index is None:
            try:
                stored = self.collection.get(include=["documents"])
                ids = list(stored.get("ids") or [])
                documents = list(stored.get("documents") or [])
            except Exception:
                return None
            if len(ids) != len(documents):
                return None
            self._lexical_index = BM25Index(ids, [d or "" for d in documents])
        return self._lexical_index

    def _rerank_lexically(
        self, query: str, raw: "Mapping[str, Any]"
    ) -> list[str] | None:
        """Fuse the dense window's order with BM25's; None to keep the dense order.

        None when there is no index, fewer than two candidates, or no disagreement.
        """
        chunk_ids = [str(c) for c in ((raw.get("ids") or [[]])[0] or [])]
        if len(chunk_ids) < 2:
            return None

        index = self.lexical_index
        if index is None:
            return None

        by_lexical = lexical_order(query, chunk_ids, index)
        if by_lexical == chunk_ids:
            return None
        return reciprocal_rank_fusion(chunk_ids, by_lexical)

    def _collapse_chunk_hits(
        self, raw: "Mapping[str, Any]", top_k: int, order: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Fold chunk hits onto their documents, best chunk first.

        `order`, when given, is the hybrid ranking of the same hits; every result
        still carries the dense score its chunk was retrieved with.
        """
        ids = raw.get("ids") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        if not ids or not ids[0]:
            return []

        collapsed: list[dict[str, Any]] = []
        by_document: dict[str, dict[str, Any]] = {}

        # In the hybrid order when there is one. An id the response does not
        # hold is skipped, since rows are read positionally.
        position = {str(c): i for i, c in enumerate(ids[0])}
        walk = (
            [(position[c], c) for c in order if c in position]
            if order is not None
            else list(enumerate(ids[0]))
        )

        for i, chunk_id in walk:
            metadata = metadatas[0][i] if metadatas[0] else {}
            doc_id = _document_id_of(chunk_id, metadata)

            if doc_id in by_document:
                by_document[doc_id]["chunks_matched"] += 1
                continue

            if len(collapsed) == top_k:
                # Full: keep counting matches, admit no new documents.
                continue

            result: dict[str, Any] = {
                "id": doc_id,
                "chunk_id": chunk_id,
                "content": documents[0][i] if documents[0] else "",
                "metadata": metadata,
                "score": 1 - distances[0][i] if distances[0] else 0.0,
                "chunks_matched": 1,
            }
            location = _passage_location(doc_id, str(result["content"]))
            if location is not None:
                result["line"], result["content"] = location

            if doc_id in self.graph:
                neighbors = list(self.graph.neighbors(doc_id))[:5]
                result["related_entities"] = neighbors

            by_document[doc_id] = result
            collapsed.append(result)

        return collapsed

    def query_graph(self, entity: str, hops: int = 2) -> dict[str, Any]:
        """An entity's neighbourhood within `hops`, resolved fuzzily (`_match_node`).

        Returns:
            The entity, its neighbours with their distances, and the subgraph's size;
            `resolved_from` and `alternatives` when the id typed was not the one found.
        """
        found, alternatives = self._match_node(entity)
        if found is None:
            return {"error": f"Entity '{entity}' not found"}
        typed, entity = entity, found

        # Undirected: every edge runs document -> entity, so a directed walk
        # from an entity reaches nothing.
        undirected = self.graph.to_undirected(as_view=True)
        neighbors = nx.single_source_shortest_path_length(
            undirected, entity, cutoff=hops
        )

        subgraph = self.graph.subgraph(neighbors.keys())

        result: dict[str, Any] = {
            "entity": entity,
            "neighbors": list(neighbors.items()),
            "subgraph_nodes": len(subgraph.nodes()),
            "subgraph_edges": len(subgraph.edges()),
        }
        if entity != typed:
            # A relationship question answered about a different node says so.
            result["resolved_from"] = typed
            result["alternatives"] = alternatives
        return result

    def _match_node(self, node_id: str) -> tuple[str | None, list[str]]:
        """Resolve a typed id to a node, and name the others it could have meant.

        Ranked: the exact id ignoring case, a document by its file name or stem, an
        id starting with the text, an id containing it; within a rank the shorter id,
        then the better connected node -- deterministic across rebuilds. A blank id
        resolves to nothing.
        """
        if node_id in self.graph:
            return node_id, []

        if not node_id or not node_id.strip():
            return None, []

        needle = node_id.strip().lower()
        undirected = self.graph.to_undirected(as_view=True)

        def rank(node: Any) -> int | None:
            name = str(node)
            lowered = name.lower()
            if lowered == needle:
                return 0
            if self.graph.nodes[node].get("type") == "document":
                path = PurePosixPath(name)
                if needle in (path.name.lower(), path.stem.lower()):
                    return 1
            if lowered.startswith(needle):
                return 2
            if needle in lowered:
                return 3
            return None

        candidates = [
            (tier, len(str(node)), -undirected.degree(node), str(node))
            for node in self.graph.nodes()
            if (tier := rank(node)) is not None
        ]
        if not candidates:
            return None, []
        candidates.sort()
        return candidates[0][3], [c[3] for c in candidates[1 : 1 + MAX_MATCH_ALTERNATIVES]]

    def _node_record(self, node_id: str) -> dict[str, Any]:
        """One graph node in the shape the console draws."""
        attrs = dict(self.graph.nodes[node_id])
        node_type = attrs.pop("type", "entity")

        # The content snippet is for retrieval, not for hovering.
        attrs.pop("content", None)

        # A bare basename collides (two README.md files), so a document's label
        # keeps its parent directory.
        label = str(node_id)
        path = attrs.get("path")
        if node_type == "document" and path:
            as_path = Path(str(path))
            parent = as_path.parent.name
            label = f"{parent}/{as_path.name}" if parent else as_path.name

        return {
            "id": node_id,
            "node_type": node_type,
            "label": label,
            "properties": attrs,
        }

    def list_documents(self) -> dict[str, Any]:
        """Every document node -- the centres the console sweeps from."""
        documents = [
            {"id": node_id, "title": self._node_record(node_id)["label"], "node_type": "document"}
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "document"
        ]
        documents.sort(key=lambda doc: str(doc["title"]))
        return {"documents": documents}

    def _sign_split(self, keep: set[Any], centre: str) -> dict[str, Any]:
        """Fiedler sign bipartition of a swept neighbourhood, oriented on the centre.

        Sides are `center` and `other`, not signs, which flip arbitrarily between
        runs. `mu_2` is returned because a split always exists and is only meaningful
        when `mu_2` is small; the caller decides what counts. Runs on the largest
        component -- `min_degree` pruning can disconnect the sweep -- and reports the
        rest as `detached`.
        """
        undirected = self.graph.to_undirected(as_view=True)
        subgraph = undirected.subgraph(keep)

        if subgraph.number_of_nodes() < 4:
            return {"available": False,
                    "note": "Too few nodes to split.", "sides": {}}

        component = subgraph.subgraph(max(nx.connected_components(subgraph), key=len))
        if component.number_of_nodes() < 4 or centre not in component:
            return {"available": False,
                    "note": "The centre's component is too small to split.", "sides": {}}

        try:
            import numpy as np

            from spectral_graph import compute_spectrum, fiedler_vector

            vector = fiedler_vector(component, normalized=True)
            mu_2 = float(max(compute_spectrum(component, k=2, normalized=True, which="SM")[1], 0.0))
        except ImportError:
            return {"available": False,
                    "note": "spectral_graph is not on sys.path.", "sides": {}}
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {"available": False, "note": f"{type(exc).__name__}: {exc}", "sides": {}}

        nodes = list(component.nodes())
        centre_positive = bool(vector[nodes.index(centre)] >= 0)
        sides = {
            node: ("center" if (bool(value >= 0) == centre_positive) else "other")
            for node, value in zip(nodes, np.asarray(vector), strict=True)
        }

        with_centre = sum(1 for side in sides.values() if side == "center")
        return {
            "available": True,
            "mu_2": mu_2,
            "center_side": with_centre,
            "other_side": len(sides) - with_centre,
            "detached": subgraph.number_of_nodes() - component.number_of_nodes(),
            "sides": sides,
        }

    def neighborhood(
        self, node_id: str, max_depth: int = 2, min_degree: int = 1,
        split: bool = False,
    ) -> dict[str, Any]:
        """A node's neighbourhood as drawable nodes and edges.

        Args:
            node_id: Centre of the traversal; resolved fuzzily.
            max_depth: Hops out from the centre.
            min_degree: Drop entity nodes with fewer edges -- a sweep passes 2 so
                one-document entities do not bury the structure; a trace passes 1.
            split: Also tag every node `center` or `other` (`_sign_split`); off by
                default, since it triples the cost of the call.

        Returns:
            center_node, center, related_nodes (excluding the centre), edges and
            totals; `split` when asked for.
        """
        centre, alternatives = self._match_node(node_id)
        if centre is None:
            return {
                "error": f"Node '{node_id}' not found",
                "center_node": node_id,
                "related_nodes": [],
                "edges": [],
                "total_nodes": 0,
                "total_edges": 0,
            }

        # Undirected, so a document's walk reaches the sibling documents
        # sharing its entities.
        undirected = self.graph.to_undirected(as_view=True)
        reachable = nx.single_source_shortest_path_length(
            undirected, centre, cutoff=max_depth
        )

        keep = {
            found
            for found in reachable
            if found == centre
            or self.graph.nodes[found].get("type") != "entity"
            or undirected.degree(found) >= min_degree
        }

        edges = [
            {
                "id": f"{source}->{target}",
                "source_id": source,
                "target_id": target,
                "relationship": data.get("relation", "related_to"),
                "weight": data.get("weight", 1.0),
            }
            for source, target, data in self.graph.edges(data=True)
            if source in keep and target in keep
        ]

        related = [self._node_record(found) for found in keep if found != centre]

        result = {
            "center_node": centre,
            # Only the graph knows whether the centre is a document or an
            # entity.
            "center": self._node_record(centre),
            "related_nodes": related,
            "edges": edges,
            "total_nodes": len(related),
            "total_edges": len(edges),
        }

        if centre != node_id:
            # The id typed was not the id found: say so.
            result["resolved_from"] = node_id
            result["alternatives"] = alternatives

        if split:
            division = self._sign_split(keep, centre)
            # Folded onto the node records, so the two cannot disagree.
            sides = division.pop("sides", {})
            for record in related:
                record["side"] = sides.get(record["id"])
            result["split"] = division

        return result

    def overview(self, min_degree: int = 4, include_isolated: bool = False) -> dict[str, Any]:
        """The whole corpus as one drawable graph -- what the console's sweep draws.

        Every document plus every entity with at least `min_degree` edges, in one
        call. A document linked to nothing in it is left out unless asked for, and
        counted in `unlinked_documents`.
        """
        undirected = self.graph.to_undirected(as_view=True)
        keep = {
            node
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "document" or undirected.degree(node) >= min_degree
        }
        edges = [
            {
                "id": f"{source}->{target}",
                "source_id": source,
                "target_id": target,
                "relationship": data.get("relation", "related_to"),
                "weight": data.get("weight", 1.0),
            }
            for source, target, data in self.graph.edges(data=True)
            if source in keep and target in keep
        ]
        linked = {edge["source_id"] for edge in edges} | {edge["target_id"] for edge in edges}
        unlinked = {node for node in keep if node not in linked}
        if not include_isolated:
            keep -= unlinked
        nodes = [self._node_record(node) for node in sorted(keep, key=str)]
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "unlinked_documents": 0 if include_isolated else len(unlinked),
        }

    def stats(self) -> dict[str, Any]:
        """Counters for the console header, plus the connectivity health check."""
        documents = sum(
            1 for _, attrs in self.graph.nodes(data=True) if attrs.get("type") == "document"
        )
        try:
            chunks = self.collection.count()
        except Exception:
            # A database failure must not take the in-memory counts with it.
            chunks = 0

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()

        # Cached against (nodes, edges): the header polls every five seconds,
        # and nothing in this class changes the graph's structure without
        # changing one of the two.
        if self._connectivity_cache is not None and self._connectivity_cache[0] == (nodes, edges):
            connectivity = self._connectivity_cache[1]
        else:
            connectivity = self.connectivity()
            self._connectivity_cache = ((nodes, edges), connectivity)

        return {
            "total_documents": documents,
            "total_chunks": chunks,
            "total_nodes": nodes,
            "total_edges": edges,
            **connectivity,
        }

    def clear(self) -> dict[str, Any]:
        """Empty the knowledge base, keeping the schema that holds it.

        Chunks, graph and floor record go in one transaction, so a failure leaves
        all three intact; the floor has to go with the texts, since a floor
        measured on texts that are gone would otherwise be read as `known` by the
        run that rebuilds the corpus. The in-memory graph is emptied only once
        the store has committed.

        Returns:
            What was removed, plus the (now zeroed) stats.
        """
        removed_nodes = self.graph.number_of_nodes()
        removed_edges = self.graph.number_of_edges()

        removed = self.collection.clear()

        self.graph.clear()
        self._lexical_index = None
        removed_floor = bool(removed["removed_floor"])

        return {
            "removed_chunks": int(removed["removed_chunks"]),
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "removed_floor": removed_floor,
            **self.stats(),
        }

    def export_corpus(self) -> dict[str, Any]:
        """The whole corpus as one JSON-serialisable document.

        Embeddings are left out -- most of the bytes, and regenerated locally -- and
        the file says so. The graph half is `node_link_data`.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        try:
            stored = self.collection.get(include=["documents", "metadatas"])
            ids = stored.get("ids") or []
            documents = stored.get("documents") or []
            metadatas = stored.get("metadatas") or []
            chunks = [
                {
                    "id": chunk_id,
                    # The document, lifted out of the metadata so the export
                    # stands on its own.
                    "doc_id": _document_id_of(
                        chunk_id, metadatas[i] if i < len(metadatas) else None
                    ),
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                for i, chunk_id in enumerate(ids)
            ]
        except Exception as exc:
            # A database failure must not cost the graph half too.
            errors.append(f"reading chunks: {exc}")

        return {
            "exported_at": datetime.now(UTC).isoformat(),
            "note": (
                "Embeddings are omitted; re-indexing regenerates them locally "
                f"with {self.embedding_model}."
            ),
            "embedding_model": self.embedding_model,
            "stats": self.stats(),
            "graph": nx.readwrite.json_graph.node_link_data(self.graph),
            "chunks": chunks,
            "errors": errors,
        }


# What the walk indexes: prose and Python, plus the markup, script and config a
# generated project is made of -- retrievable, but minting no entities (see
# `ENTITY_FREE_SUFFIXES`). Compared exactly, so an upload is stored with its
# suffix lower-cased or the next rebuild would not see it.
INDEXABLE_SUFFIXES = (
    ".cfg", ".css", ".html", ".ini", ".js", ".md", ".py", ".rst",
    ".sh", ".toml", ".txt", ".yaml", ".yml",
)

# Directories the walk never enters, wherever they sit under a corpus root:
# version control, virtualenvs, tool caches and build output -- what a generated
# project accumulates without anyone writing it. Matched against whole directory
# names (plus the `.egg-info` suffix), never as substrings, so `rebuild/` is not
# `build/` and a project's own `src/` and `tests/` are indexed like any others.
CORPUS_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", "build", "dist",
})

# Above this size a file is not a document -- a dump, a minified bundle, a log.
# Chunking means a long document takes one result slot however many chunks it
# holds; the ceiling exists because `search` retrieves a window of chunks
# before collapsing them, and a large enough document could fill that window
# alone.
MAX_INDEXABLE_BYTES = 250_000

# Where a document uploaded from the console lands, relative to the project
# root. An upload is written to disk *before* it is embedded: the corpus is a
# function of what is on disk, and a rebuild prunes every stored row whose
# document is not in the walk, so a document embedded straight into the store
# would survive only until the next rebuild deleted it, silently.
UPLOADS_DIR = "uploads"

# Where the online research phase writes the pages it fetched. Defined here,
# not in `web_research`, which imports this module, because `add_document` has
# to recognise one on sight.
WEB_RESEARCH_DIR = "research/web"

# The only directories the walk reads: researched archive data, and what the
# operator embedded on purpose. The checkout itself is the program, never the
# corpus; anything else in the store is pruned by the next rebuild.
CORPUS_ROOTS = (WEB_RESEARCH_DIR, UPLOADS_DIR, PROJECTS_DIR)


def _is_web_document(doc_id: str) -> bool:
    """True for a page the online research phase fetched.

    Decided from the path, so an upload, a first index and every rebuild agree;
    matched on parent directories, so an absolute root works.
    """
    parts = PurePosixPath(str(doc_id).replace("\\", "/")).parts
    wanted = PurePosixPath(WEB_RESEARCH_DIR).parts
    return len(parts) > len(wanted) and parts[-len(wanted) - 1 : -1] == wanted


# Suffixes whose documents are retrievable but mint no entities: their capitals
# are identifiers and interface strings (`ACTIVE_SEAT`, `Copying`), not terms
# the corpus is about. Python stays out of the set; its docstrings are where
# the terms live.
ENTITY_FREE_SUFFIXES = frozenset(
    {".html", ".js", ".css", ".sh", ".toml", ".yml", ".yaml", ".ini", ".cfg"}
)

# The `type` a document is stored under, by suffix; prose is "markdown".
_DOCUMENT_TYPES = {
    ".py": "python",
    ".html": "html", ".js": "javascript", ".css": "css", ".sh": "shell",
    ".toml": "config", ".yml": "config", ".yaml": "config", ".ini": "config",
    ".cfg": "config",
}


# How many runners-up a loose node id reports beside its match.
MAX_MATCH_ALTERNATIVES = 5

# How much of a passage's line a result reaches back for: a chunk starts mid-
# line, but a minified line has no useful start.
MAX_PASSAGE_LEAD_CHARS = 240


def _passage_location(doc_id: str, passage: str) -> tuple[int, str] | None:
    """Where a retrieved passage sits in its file: (1-based line, passage from that line's start).

    Read from the file on disk; None when it cannot be read or no longer contains
    the passage, where a guessed line would point at the wrong code.
    """
    if not passage:
        return None
    try:
        text = Path(doc_id).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    start = text.find(passage)
    if start < 0:
        return None
    line_start = text.rfind("\n", 0, start) + 1
    lead = text[line_start:start]
    if len(lead) > MAX_PASSAGE_LEAD_CHARS:
        lead = ""
    return text.count("\n", 0, start) + 1, lead + passage


def _mints_entities(doc_id: str) -> bool:
    """Whether a document takes part in entity extraction, decided from its path."""
    if _is_web_document(doc_id):
        return False
    suffix = PurePosixPath(str(doc_id).replace("\\", "/")).suffix.lower()
    return suffix not in ENTITY_FREE_SUFFIXES


def _document_metadata(path: Path) -> dict[str, str]:
    """The metadata a document is stored under -- one definition for the rebuild
    and an upload, so one file is never filed twice.
    """
    return {
        "path": str(path),
        "type": _DOCUMENT_TYPES.get(path.suffix.lower(), "markdown"),
    }


def store_uploaded_document(
    kb: "GraphRAGKnowledgeBase",
    name: str,
    content: str,
    root: str = ".",
) -> dict[str, Any]:
    """Write an uploaded document under `uploads/` and embed it, in that order.

    What is left is a document like any other: the same id shape, the same
    metadata, and a place in the walk.

    Args:
        kb: The corpus to add to -- the creating door: an upload asks for a
            corpus to hold it.
        name: The filename as the browser reported it; only its last component
            is used.
        content: The document's text, already decoded.
        root: Project root, so the id stored is the id the walk produces.

    Returns:
        The path, whether it replaced a document, how many passages it became,
        and the corpus stats.

    Raises:
        ValueError: a name, type, size or content the corpus cannot take --
            anything else would embed noise under a real filename, or embed a
            file the next rebuild drops. The message says which.
    """
    # Only the last component, so a name cannot place the file outside
    # `uploads/`; `..` survives that and is refused by name.
    safe = PurePosixPath(name.replace("\\", "/")).name
    if safe in ("", ".", "..") or "\x00" in safe:
        raise ValueError(f"{name!r} is not a filename this can store.")

    suffix = PurePosixPath(safe).suffix.lower()
    if suffix not in INDEXABLE_SUFFIXES:
        accepted = ", ".join(INDEXABLE_SUFFIXES)
        raise ValueError(
            f"{safe!r} is not one of {accepted}. There is no text extractor "
            "here, so a PDF, a .docx or an image would be embedded as whatever "
            "its bytes happen to decode to rather than as what it says -- and "
            "it would look like a document in the corpus afterwards. Convert "
            "it to text first."
        )
    stored_name = PurePosixPath(safe).stem + suffix

    if "\x00" in content:
        raise ValueError(
            f"{stored_name!r} contains null bytes, so it is not text however it "
            "is named. Nothing was stored."
        )
    if not content.strip():
        raise ValueError(f"{stored_name!r} has nothing in it to embed.")
    # The walk's own limit, character for character.
    if len(content) > MAX_INDEXABLE_BYTES:
        raise ValueError(
            f"{stored_name!r} is {len(content):,} characters and the limit is "
            f"{MAX_INDEXABLE_BYTES:,} -- the same limit a reindex applies. "
            "Accepting it would embed it now and drop it at the next rebuild. "
            "Split it up."
        )

    directory = Path(root) / UPLOADS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / stored_name
    # Read before the write, so "replaced" is a fact. Re-uploading a name is
    # how a document is corrected; `add_document` removes its previous chunks.
    replaced = path.exists()
    path.write_text(content, encoding="utf-8")

    chunks = kb.add_document(str(path), content, _document_metadata(path))

    report: dict[str, Any] = {
        "path": str(path),
        "name": stored_name,
        "replaced": replaced,
        "chunks": chunks,
        "characters": len(content),
    }
    report.update(kb.stats())
    return report


def _skipped_dir(name: str) -> bool:
    return name in CORPUS_SKIP_DIRS or name.endswith(".egg-info")


def iter_corpus_files(root: str = ".", roots: tuple[str, ...] | None = None) -> list[Path]:
    """The files a rebuild indexes, spelled the way it stores them.

    Walks `CORPUS_ROOTS` alone, never the checkout, pruning `CORPUS_SKIP_DIRS`; a
    generated project is taken only once the operator has opted it in (see
    `langgraph_agent.projects`). `roots` overrides `CORPUS_ROOTS` for a caller
    auditing something other than the corpus; `("",)` is the root itself.
    """
    root_path = Path(root)
    embedded = embedded_projects(root_path)

    files: set[Path] = set()
    for corpus_root in CORPUS_ROOTS if roots is None else roots:
        for directory, subdirs, names in os.walk(root_path / corpus_root):
            subdirs[:] = [name for name in subdirs if not _skipped_dir(name)]
            for name in names:
                path = Path(directory, name)
                if name.endswith(INDEXABLE_SUFFIXES) and not held_out_of_corpus(
                    path.relative_to(root_path), embedded
                ):
                    files.add(path)
    return sorted(files)


def index_corpus_files(
    kb: "GraphRAGKnowledgeBase",
    root: str = ".",
    *,
    progress: "Callable[[int, int], None] | None" = None,
    should_stop: "Callable[[], bool] | None" = None,
) -> dict[str, Any]:
    """Rebuild the corpus from the walk, keeping the vectors of unchanged documents.

    A rebuild, not an accumulation: rows whose document left the walk are pruned
    first, and a document whose text still hashes to what the store holds keeps
    its vectors and only rejoins the graph -- so a rebuild costs what changed.

    `progress(done, total)` follows each file; `should_stop` is asked before each
    file and between embedding batches. A stop, or a circuit the embedder needs
    opening -- the daemon's, or `EMBEDDER_LOAD` -- ends the pass early: `stopped`,
    or `unavailable` with the circuit named in `unavailable_circuit`. Either
    leaves the corpus part-built for the next rebuild to finish.
    """
    files = iter_corpus_files(root)
    wanted = {str(path) for path in files}

    # One transaction: every document that left the walk is deleted by set
    # difference, and the fingerprints of the ones that stayed are read from
    # what survived it.
    stored_sha: dict[str, str] = {}
    try:
        with kb.collection.transaction():
            # Counted in documents, not rows: how many files stopped answering.
            dropped = len(kb.collection.prune_documents(wanted))
            stored_sha = kb.collection.fingerprints()
    except CircuitOpenError as exc:
        # The database is gone: nothing below could be stored either.
        kb.graph.clear()
        refused: dict[str, Any] = {
            "stopped": False, "unavailable": str(exc), "unavailable_circuit": exc.circuit,
            "indexed": 0, "embedded": 0, "reused": 0, "dropped": 0, "skipped": 0,
            "errors": [],
        }
        refused.update(kb.stats())
        return refused
    except Exception as exc:
        errors_pre = [f"pruning stale documents: {exc}"]
        dropped = 0
    else:
        errors_pre = []
    kb.graph.clear()

    indexed = embedded = reused = skipped = 0
    errors: list[str] = list(errors_pre)

    stopped = False
    unavailable = unavailable_circuit = ""
    kb._should_stop = should_stop
    for position, file_path in enumerate(files, 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        try:
            content = file_path.read_text(encoding="utf-8")
            if len(content) > MAX_INDEXABLE_BYTES:
                skipped += 1
                continue

            doc_id = str(file_path)
            metadata = _document_metadata(file_path)
            if stored_sha.get(doc_id) == _content_sha(content):
                # Unchanged: only the cleared graph needs it back.
                kb._add_to_graph(doc_id, content, metadata)
                reused += 1
            else:
                kb.add_document(doc_id, content, metadata)
                embedded += 1
            indexed += 1
        except EmbeddingStopped:
            stopped = True
            break
        except CircuitOpenError as exc:
            # The daemon or the database is gone, or the model will not load:
            # every document left would fail the same way.
            unavailable, unavailable_circuit = str(exc), exc.circuit
            break
        except EmbedderLoadFailed as exc:
            # This document's load opened the embedder's circuit, which would
            # refuse every document left.
            unavailable, unavailable_circuit = f"{file_path}: {exc}", EMBEDDER_LOAD.name
            break
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
        if progress is not None:
            progress(position, len(files))

    # Persisted unconditionally: a rebuild that embedded nothing must still
    # save the cleared graph and drop the lexical index built before the prune.
    kb._lexical_index = None
    kb._should_stop = None
    try:
        kb._save_graph()
    except Exception as exc:
        errors.append(f"saving the graph: {exc}")

    # `indexed` is how many documents the corpus holds; `embedded` and `reused`
    # split it by cost.
    report: dict[str, Any] = {
        "stopped": stopped,
        "unavailable": unavailable,
        "unavailable_circuit": unavailable_circuit,
        "indexed": indexed,
        "embedded": embedded,
        "reused": reused,
        "dropped": dropped,
        "skipped": skipped,
        "errors": errors,
    }
    report.update(kb.stats())
    return report


# The one knowledge base this process holds.
_kb_instance: GraphRAGKnowledgeBase | None = None


def get_knowledge_base(persist_dir: str | None = None) -> GraphRAGKnowledgeBase:
    """The singleton knowledge base, **built if it does not exist yet**.

    The door reserved for indexing; every read goes through
    `open_knowledge_base()`, which never creates a store. `persist_dir` is honoured
    only by the call that builds the singleton.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = GraphRAGKnowledgeBase(persist_dir)
    return _kb_instance


def embedding_device_status() -> dict[str, Any]:
    """How the embedder is placed, for the header's poll, without touching the daemon.

    `active` and `cpu_share` are None until something has embedded; reading this
    never opens a corpus.
    """
    kb = _kb_instance
    loaded = kb is not None and kb._embedder is not None
    embedder = kb._embedder if kb is not None else None
    return {
        "configured": "ollama",
        "active": kb.embedding_device if kb is not None and loaded else None,
        "note": kb.embedding_device_note if kb is not None else None,
        "cpu_share": embedder.cpu_share if isinstance(embedder, OllamaEmbedder) else None,
    }


def corpus_exists(persist_dir: str | None = None) -> bool:
    """Whether a corpus has been built, without building one.

    An emptied store still exists; `corpus_state()` tells empty from absent. A
    database that cannot be reached raises: that is not "no corpus".
    """
    return open_store(persist_dir) is not None


# Why the last `open_knowledge_base` could not ask the database, or None.
_unreachable: str | None = None


def open_knowledge_base(persist_dir: str | None = None) -> GraphRAGKnowledgeBase | None:
    """The corpus if one has been built, `None` if none has. Never builds one.

    Constructing a knowledge base creates the store, so a read wired to
    `get_knowledge_base` would leave a corpus behind on the first status poll.
    A database that cannot be reached is also `None` -- a run without a corpus
    is a worse run, not a refused one -- and `absent_corpus()` says which.
    """
    global _unreachable
    if _kb_instance is not None:
        return _kb_instance
    try:
        exists = corpus_exists(persist_dir)
    except Exception as exc:
        if not (isinstance(exc, CircuitOpenError) or database_unreachable(exc)):
            raise
        _unreachable = str(exc)
        return None
    _unreachable = None
    if not exists:
        return None
    return get_knowledge_base(persist_dir)


def absent_corpus() -> tuple[str, str]:
    """What a read that found no corpus reports: `(state, note)`.

    `unavailable` when the last open could not reach the database, which says
    nothing about whether anyone indexed; `absent` with `NO_CORPUS_NOTE`
    otherwise.
    """
    if _unreachable is not None:
        return "unavailable", (
            f"The corpus database could not be reached ({_unreachable}), so there "
            "is nothing to retrieve until it answers. The console reports it as "
            "the postgres circuit; a run goes ahead without retrieval."
        )
    return "absent", NO_CORPUS_NOTE


def corpus_state(persist_dir: str | None = None) -> tuple[str, str]:
    """Report the corpus as `absent`, `empty`, `indexed` or `unavailable`, plus the model.

    Non-creating and cheap enough to poll: two queries, never the embedder.
    `absent` and `empty` are told apart because only one of them means nothing
    was ever indexed here; `unavailable` is a database that cannot be asked,
    which is neither.

    Returns:
        (state, embedding_model_name)
    """
    try:
        store = open_store(persist_dir)
        if store is None:
            return "absent", EMBEDDING_MODEL_NAME
        return ("indexed" if store.count() > 0 else "empty"), EMBEDDING_MODEL_NAME
    except Exception:
        return "unavailable", EMBEDDING_MODEL_NAME


# Re-exported from `corpus_spectral`, where the whole-graph diagnostics live.
__all__ = [
    "BOTTLENECK_CONDUCTANCE",
    "DUPLICATE_BLOCK_PREFIX",
    "DUPLICATE_CONTAINMENT",
    "DUPLICATE_NAME_SIMILARITY",
    "EIGENGAP_DECISIVENESS",
    "MAX_AUTO_CLUSTERS",
    "CorpusSpectralMixin",
    "GraphRAGKnowledgeBase",
]
