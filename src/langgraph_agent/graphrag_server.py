"""GraphRAG MCP Server.

Provides knowledge base search with entity/relation graph + vector store.

Usage:
    python -m src.langgraph_agent.graphrag_server

Or with stdio transport for MCP:
    mcp dev src/langgraph_agent/graphrag_server.py
"""

import asyncio
import functools
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, ParamSpec, TypeVar

import chromadb
import networkx as nx
from mcp.server import MCPServer

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
from langgraph_agent.lexical import (
    BM25Index,
    lexical_order,
    reciprocal_rank_fusion,
)
from langgraph_agent.projects import embedded_projects, held_out_of_corpus

# The embedding model that builds and searches the corpus: an Ollama tag the
# local daemon runs. Named once because three places have to agree on it: the
# embedder the store is built with, the status check that reports it without
# loading it, and the export that records which model produced the corpus it
# is dumping. There is exactly one: vectors from two models share no space
# (this one's are 4,096 numbers), and a corpus is only ever built and searched
# by the model it was built with. GPU placement is the daemon's business --
# OLLAMA_EMBED_OPTIONS below has the measurements -- so nothing in this
# process touches torch or a card, and there is no EMBEDDING_DEVICE.
EMBEDDING_MODEL_NAME = "qwen3-embedding:latest"

# The tokenizer the chunker cuts passages with: the embedding model's own,
# loaded in-process from the Hugging Face cache -- the daemon's tokenizer is
# not reachable from here. `transformers` loads the tokenizer files alone, a
# few MB, and never a weight: it embeds nothing.
EMBEDDING_TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"

# Passages per `/api/embed` request. 8 is the batch the OLLAMA_EMBED_OPTIONS
# measurements were taken at: on 2026-09-16 a batch of 8 full 254-token
# passages went through in 6.5s with the model wholly on two 3 GB cards.
EMBEDDING_BATCH_SIZE = 8


# Seconds one batch of passages may take through Ollama. Measured for
# qwen3-embedding (7.6B) on two 3 GB cards on 2026-09-16: a batch of 8 took
# 23.4s with 30% of the model left on the CPU and 6.5s wholly on the cards
# (`OLLAMA_EMBED_OPTIONS`) -- and the first batch of a run waits for the load too.
OLLAMA_EMBED_TIMEOUT_SECONDS = 600.0

# How often, and how far apart, a batch the daemon answered with a 5xx is sent
# again. A 5xx here is nearly always the model *load* failing, and a load that
# fails is not a fact about the passages: `num_gpu` forces every layer onto the
# cards, so a load that meets a card still held by something else errors rather
# than splitting. Measured at console startup on 2026-09-18: the first three
# embeds came back 500 (`cudaMalloc failed: out of memory` on card 1, card 0
# with ~1 GB held by whatever else had just started) and the fourth, 12s later,
# loaded and every batch after it went through. Each of those three was a whole
# document lost to the rebuild. A 4xx is the daemon refusing the request --
# a missing tag -- and is never retried.
OLLAMA_EMBED_LOAD_RETRIES = 4
OLLAMA_EMBED_RETRY_SECONDS = 5.0

# The window and batch the embedding model is loaded with, sent on every call.
# Ollama loads one at a 4,096-token window with a 2,048-token batch by default,
# and on two 3 GB cards qwen3-embedding then asked for 7,463 MiB
# against 6,217 free -- 4,453 of weights, 576 of KV cache and 2,433 of compute
# buffers sized for that batch -- so the daemon ran 25 of its 37 layers on the
# cards and the rest on the CPU, at 0.24 passages/s. At 512 it takes 4,987 MiB,
# 37 of 37 on the cards, at 0.48 passages/s, and the vectors do not move:
# cosine 1.000000 against the default load on the corpus's eight longest
# passages, which peaked at 362 of its tokens. A longer input -- a query that
# is a whole plan -- is cut at 511 tokens rather than refused, which is still
# twice the chunker's 254-token passages. 1,024 fits as well (5,403 MiB) at
# the same speed with half the room to spare. Every call sends the same options because the
# daemon reloads a model whose options changed, so a search at another window
# would evict the runner an index is using.
#
# num_gpu puts every layer on the cards, and the window alone no longer did.
# Measured 2026-09-16 on Ollama 0.33.3's CUDA 12 build (cuda-embed-ollama.sh):
# with 5.6 GiB free across the two cards, the daemon's own estimate offloaded
# 24 of 37 layers -- 70% on the GPU, a batch of 8 in 23.4s -- and left 1.1 GB
# unused on each card. Forced, all 37 load (4,995 MiB, 100% on the GPU), a batch
# of 8 takes 6.5s, a 511-token input still runs, and the cards sit at 2,424 and
# 2,947 of 3,072 MiB. The cost is the failure mode: where the model does not
# fit, the load errors instead of spilling onto the CPU, which is what "wholly
# on the GPU" asks for. Vectors from a split load differ slightly (cosine
# 0.9986 at worst on eight passages), so a corpus is built on the placement it
# is searched on.
OLLAMA_EMBED_OPTIONS: dict[str, int] = {"num_ctx": 512, "num_batch": 512, "num_gpu": 999}


class EmbeddingStopped(RuntimeError):
    """The run was stopped between two batches of an embedding."""


class OllamaEmbedder:
    """An Ollama embedding model behind the two things the corpus asks of one.

    `encode` sends passages to `/api/embed` in batches of
    `EMBEDDING_BATCH_SIZE` and returns one numpy vector per passage, which is
    all `add_document` and `search` ask of it.

    `tokenizer` is the embedding model's own (`EMBEDDING_TOKENIZER_NAME`),
    loaded in-process. The chunker needs a tokenizer in this process and the
    daemon's is not reachable from here; it loads from the local Hugging Face
    cache first, so a machine offline after install still chunks. 254 of its
    tokens sit far inside the window the daemon is loaded with
    (`OLLAMA_EMBED_OPTIONS`), so a passage is never cut twice.
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

        `should_stop` is asked before each batch, so a stopped run waits for at
        most one batch rather than for the rest of a document.
        """
        import time
        import urllib.error
        import urllib.request

        import numpy as np

        from langgraph_agent.config import _ollama_base_url, ollama_cpu_share

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors: list[list[float]] = []
        step = max(1, batch_size)
        for start in range(0, len(items), step):
            if should_stop is not None and should_stop():
                raise EmbeddingStopped(f"stopped after {start} of {len(items)} passages")
            batch = items[start : start + step]
            request = urllib.request.Request(
                f"{_ollama_base_url()}/api/embed",
                data=json.dumps(
                    {"model": self.model, "input": batch, "options": OLLAMA_EMBED_OPTIONS}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            attempt = 0
            while True:
                try:
                    with urllib.request.urlopen(
                        request, timeout=OLLAMA_EMBED_TIMEOUT_SECONDS
                    ) as response:
                        payload = json.loads(response.read())
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", "replace").strip()
                    if exc.code >= 500 and attempt < OLLAMA_EMBED_LOAD_RETRIES:
                        attempt += 1
                        # Waited in slices, so a stop reaches a retrying
                        # build as quickly as it reaches a working one.
                        deadline = time.monotonic() + OLLAMA_EMBED_RETRY_SECONDS * attempt
                        while time.monotonic() < deadline:
                            if should_stop is not None and should_stop():
                                raise EmbeddingStopped(
                                    f"stopped after {start} of {len(items)} passages"
                                ) from exc
                            time.sleep(0.25)
                        continue
                    tried = f" (after {attempt + 1} attempts)" if attempt else ""
                    raise RuntimeError(
                        f"Ollama could not embed with {self.model}{tried}: {detail or exc}"
                    ) from exc
                except OSError as exc:
                    raise RuntimeError(
                        f"Ollama could not embed with {self.model}: {exc}"
                    ) from exc
            embeddings = payload.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} vectors for {len(batch)} "
                    f"passages from {self.model}"
                )
            vectors.extend(embeddings)
            if start == 0:
                # Asked on every call, once its first batch has loaded the
                # model: the daemon reloads a model something else evicted, and
                # fits the reload around whatever holds the cards by then.
                self.cpu_share = ollama_cpu_share(self.model)
        array = np.asarray(vectors, dtype=np.float32)
        return array[0] if single else array


# Where a corpus lives when nobody says otherwise. One constant because four
# doors resolved it by spelling the same default inline, which makes the
# project's own directory a magic string with no source of truth -- and a
# fifth door added later copies the expression rather than noticing there was
# a pattern to reuse.
DEFAULT_PERSIST_DIR = "knowledge"


def resolve_persist_dir(persist_dir: str | Path | None = None) -> Path:
    """The corpus directory a caller means, defaulting to `DEFAULT_PERSIST_DIR`."""
    return Path(persist_dir or DEFAULT_PERSIST_DIR)


# The questions the corpus's relevance floor is measured with, in a JSON file
# for one reason: the walk does not index JSON. Written anywhere the corpus
# reads, the unanswerable questions would be answered by their own text, score
# near 1.0, and leave no gap to put a floor in.
FLOOR_CALIBRATION_QUESTIONS = Path(__file__).with_name("embedding_calibration.json")
FLOOR_CALIBRATION_FILE = "floor_calibration.json"


def _floor_calibration_path(persist_dir: str | Path | None = None) -> Path:
    """Where a corpus's floor record lives: beside the store, never elsewhere.

    One function because the reader and the writer disagreeing is silent --
    a floor written under an alternate `persist_dir` and read from the default
    one leaves `relevance_floor()` None for good, which reads exactly like a
    corpus nobody has measured. The default matches every other door here
    (`corpus_exists`, `corpus_state`, `GraphRAGKnowledgeBase.__init__`).
    """
    return resolve_persist_dir(persist_dir) / FLOOR_CALIBRATION_FILE


def floor_calibration(persist_dir: str | Path | None = None) -> dict[str, Any] | None:
    """The stored measurement behind the corpus's floor, or None if none was taken."""
    path = _floor_calibration_path(persist_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("model") == EMBEDDING_MODEL_NAME else None


def floor_from_calibration(record: dict[str, Any] | None) -> float | None:
    """The floor a stored record carries, or None when it carries none.

    Split out of `relevance_floor` so a caller already holding the record reads
    the file once rather than twice: `_embedding_choice` needs both the record
    and the floor, to tell "the measurement found no gap" from "nobody has
    measured", and the console asks it on every five-second poll. One function
    because the coercion spelled twice is two answers to one question.
    """
    floor = record.get("floor") if record else None
    return float(floor) if isinstance(floor, (int, float)) else None


def relevance_floor(persist_dir: str | Path | None = None) -> float | None:
    """The score over which a search counts as the corpus having answered.

    It is what `calibrate_relevance_floor` measured on this corpus with
    `EMBEDDING_MODEL_NAME` -- None until a run finishes a corpus and takes it,
    and None for good if the measurement found no gap between the two
    populations. None means retrieval cannot tell an answer from noise, and a
    caller must treat every search as unanswered rather than borrow a number
    measured on a different model: a cosine has no meaning across models.
    """
    return floor_from_calibration(floor_calibration(persist_dir))


def calibrate_relevance_floor(kb: "GraphRAGKnowledgeBase") -> dict[str, Any]:
    """Take the relevance floor for this corpus's embedding model, and keep it.

    Twelve questions this corpus answers against twelve it cannot, each asked
    once. The floor is the midpoint of the gap between the lowest answered
    score and the highest unanswered one -- a value in open space rather than
    on an observed boundary. When the populations overlap there is no gap and
    no floor: any number inside the overlap would misfile some question, and
    nothing would say which.
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
    path = _floor_calibration_path(kb.persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1), encoding="utf-8")
    temporary.replace(path)
    return record

# What every caller says when asked to search a corpus nobody has built. One
# string because three doors report it -- the MCP tools, the Builder's tool
# belt, and the console -- and a corpus that reads as absent in one place and
# as merely empty in another is the confusion this whole path exists to avoid.
NO_CORPUS_NOTE = (
    "No corpus has been indexed here, so there is nothing to retrieve. Two "
    "things build one: a run, which indexes the project it was started from "
    "before the Architect opens, and embedding a document into the corpus from "
    "the console. Nothing else does. Reaching this note during a run means the "
    "walk found nothing to index, or INDEX_PROJECT_BEFORE_RUN is off."
)







# Capitalised tokens that are not entities. `add_document` mints an entity for
# every capitalised word over four characters, and in a corpus of prose and
# numpy-style docstrings that rule fires constantly on words whose capital is
# an artefact of where they sit rather than of what they mean: a sentence
# opener (`Every`, `Nothing`, `Without`), a docstring section header
# (`Returns`, `Parameters`, `Raises`), a report heading (`Files`, `Status`), or
# a Python literal (`False`, `None`). Measured before this list, `False` was
# the 4th best-connected node in the graph and `Returns` the 6th, above
# `Fiedler`, `Planner` and `Cheeger`: 21 documents share an edge through
# `Returns`, which says only that all 21 contain a docstring.
#
# **This list does not exist to make `topics()` decisive, and it does not.**
# That was the first hypothesis and the measurement refused it: removing every
# one of the 97 entities that bridge this corpus's two topic areas -- the
# theoretical maximum any such filter could achieve -- moves the eigengap
# decisiveness from 1.12x to 1.09x. The corpus's spectrum is a smooth
# continuum because the corpus genuinely has no decisive k, not because
# boilerplate is gluing it together. What the list is for is the graph itself:
# an edge through `Returns` is a false claim that two documents are related,
# and `neighborhood()`, `top_entities` and `duplicate_entities` all read those
# edges as evidence.
#
# Matched case-insensitively against the whole token, never as a prefix. A
# **stopword list, not a heuristic**, on purpose: the obvious alternative is to
# drop a token that only ever appears where a capital is forced (line start,
# after a full stop), and it was built and measured. It removes the same noise
# and severs real edges doing it -- `Planner` 18 documents down to 15,
# `Spectral` 19 to 15, `ValueError` 14 to 11 -- because a term introduced in a
# bulleted list (`- **Planner** -- interprets goals`) never appears anywhere
# else in that document. Silently dropping a true relation to catch a false one
# is the wrong trade here, and a list a reader can audit line by line beats a
# rule whose failures are invisible. Measured on this corpus, this list removes
# 84 entities and 437 edges and costs **no** meaningful term a single edge.
#
# **A hand-audited list drifts, and this one did.** The claim above held when
# it was written and had stopped holding by 2026-09-09, because the corpus it
# was audited against had moved on -- much of it prose written *since*, in this
# file and in CLAUDE.md. `Tests`, `Measured` and `System` were the 6th, 8th and
# 10th best-connected entities in the graph, which is the exact failure the
# list exists to prevent, one vocabulary later. `Measured` is the sharpest:
# CLAUDE.md opens sentences with it ("Measured on this corpus...") twenty-nine
# times, so the prose recording these measurements was minting the entity.
#
# The additions below were chosen by counting, per token, the capitals that
# position does **not** explain -- not at a line start, not after a full stop,
# not the first cell of a table row. A token with zero of those is recording
# where it sits. That is the same heuristic rejected above, used the way it is
# sound: to *nominate* candidates for a human to rule on, never to filter. Two
# nominations were refused on that read and are the reason the pass is a hand
# audit rather than a script. `L_dense` scores zero free capitals because an
# assignment starts its line -- it is a real identifier and stays. `Spectral`
# scores two, both marginal, and stays because it is a term this corpus is
# about; the cost of a wrong removal is a severed true relation, which is
# exactly what the paragraph above refuses. `System`, `Search`, `State` and
# `Verification` were nominated by rank and cleared by the count: `System`
# alone carries 22 free capitals, so the entity is earned.
#
# It moved again on 2026-09-12, which is the point of the guard rather than a
# surprise: `Reported` and `Computed` each crossed the four-document floor at
# zero position-free capitals, as prose was written about what a phase reports
# and about what is computed where. Both are the `Measured` case exactly -- the
# writing-up of a change minting the entity -- and both joined the list by the
# same count. The lesson is not the two words: it is that this list is a claim
# about a vocabulary, and the vocabulary grows every time someone documents
# something -- and code is documented too: the same day an agent-written
# spectral-analysis module (the quisce prototype, since moved out of the
# checkout) took `Perform` and `Useful` across
# the floor, both docstring openers at zero position-free capitals, and `Prose`
# crossed it the same day as the first word of a comment's sentence.
#
# `Observed` crossed on 2026-09-18, and it is the plainest case of the pattern
# yet: this project records what it measured, so it opens sentences with the
# word for having seen something. Four documents, zero position-free capitals --
# two line starts, two after a full stop -- and it reached the floor within the
# one change that added a fourth, which is the guard doing exactly what it is
# for. Nothing here is *about* observation.
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

    Hello Inference Three Verification
    """.split()
)

# A chunk's length in the embedding model's tokens. Far inside the 512-token
# window the daemon is loaded with (`OLLAMA_EMBED_OPTIONS`), so the cut is
# deliberate rather than window-forced. Not a tuning knob: a passage longer
# than the model's window is not embedded badly, it is **silently truncated
# and the tail discarded** -- and that failure once cost this corpus 91.5% of
# itself.
#
# A document used to be embedded whole, in one `encode()` call, with
# `MAX_INDEXABLE_BYTES` then allowing 100 KB -- so the vector for a 46 KB file
# was computed from its first ~1,000 characters and nothing else. Measured on
# this project's own corpus before chunking: 73 of 77 documents over the
# limit, 224,809 tokens present and 19,147 embedded, **91.5% of the corpus
# unreachable by search**. Two failures came out of that, and neither
# announces itself: retrieval acquired a *length bias*, because a short file
# is fully represented while a long one is represented by its preamble -- so
# the file that actually answers the query loses to a shorter one that merely
# mentions it -- and scores sat low enough that plan-shaped queries fell under
# the relevance gate in `nodes.py`, discarding retrieval and sending the run
# to the Researcher's model, which is the loop this project already knows is
# fragile.
CHUNK_MAX_TOKENS = 254

# Tokens carried from the end of one chunk into the start of the next. Chunk
# boundaries are cut on token counts, not on sentences, so a passage can be
# split down the middle; the overlap is what keeps such a passage whole in at
# least one chunk, and it is why boundaries are *not* snapped to line breaks.
# Snapping would have to either shorten a chunk (dropping tokens the model
# could have seen) or lengthen it past the window (truncating again, which is
# the bug), so the cut stays where the arithmetic puts it and the overlap
# absorbs the cosmetic cost.
CHUNK_OVERLAP_TOKENS = 48

# Separates a document id from its chunk number: `CLAUDE.md#0003`. A filename
# may itself contain "#" -- every filesystem this runs on allows it -- so the
# id alone is not what maps a chunk back to its document: each chunk carries
# `doc_id` in its metadata, `_document_id_of` reads that first, and the id is
# split only as a fallback, at the last "#" and only when a number follows it.
CHUNK_ID_SEPARATOR = "#"


def _content_sha(content: str) -> str:
    """A fingerprint of a document's text, stored on every chunk it becomes.

    This is what lets a rebuild keep the vectors it already has. Embedding is
    the only expensive part of indexing this project -- measured warm, a full
    rebuild of 77 files and 1,618 chunks takes 52.0s, of which reading every
    file, hashing it, fetching the store's metadata and rebuilding the whole
    entity graph account for 0.1s. So a rebuild that re-embeds only what
    changed costs what the change costs, and one where nothing changed costs
    nothing at all and never loads the model.

    A hash rather than an mtime: a checkout, a `git stash`, a file copied back
    into place all move the timestamp without changing a byte, and re-embedding
    a corpus because someone switched branches is the cost this exists to
    avoid. Truncated to 16 hex characters because it is compared, never
    trusted -- a collision re-uses a stale vector, which the next edit to that
    file corrects, and 64 bits of it is not a risk anyone here will meet.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

# How many chunks to pull per requested result before collapsing them onto
# their documents. A query that matches one document strongly can match several
# of its chunks, and `search` returns documents, so without oversampling a
# top_k of 5 could collapse to 1. Four is enough for the shapes measured here
# without making Chroma do meaningfully more work.
SEARCH_CHUNK_OVERSAMPLE = 4

# The second and last rung of the ladder in `search`, taken only when the first
# window of hits collapsed to fewer documents than the caller asked for. A
# focused query really can match a dozen passages of one large file before it
# matches anything else, and answering a request for five sources with one is
# narrower than the behaviour chunking replaced.
SEARCH_ESCALATION = 8


def _chunk_windows(
    n_tokens: int, max_tokens: int, overlap: int
) -> list[tuple[int, int]]:
    """Token index windows `[start, end)` covering `n_tokens`, with overlap.

    Pure arithmetic, kept out of `chunk_text` so the packing can be tested
    without loading the embedding model -- the thing every other test in this
    project goes out of its way to avoid.

    Every token lands in at least one window and no window is longer than
    `max_tokens`, which together are the whole contract: the first is what
    stops the truncation this exists to fix, and the second is what stops each
    chunk from being truncated in turn.
    """
    if n_tokens <= max_tokens:
        return [(0, n_tokens)] if n_tokens else []

    # A stride at or below zero would never advance and the loop would not
    # terminate. Clamped rather than raised on: an overlap wider than the
    # window is a caller's misconfiguration, and degrading to "no overlap" is
    # better than refusing to index at all.
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

    Prefers the `doc_id` the chunk carries in its metadata and falls back to
    parsing the id, because both shapes exist in a live store: rows written
    before chunking are keyed by the bare document path and carry no `doc_id`,
    and they must keep resolving to themselves rather than being read as
    strangers and pruned. The suffix is only stripped when it is actually a
    chunk number, so a path that happens to contain "#" is left alone.
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

    Put on the only two places the embedder does anything: `_load_embedder`,
    which builds the daemon handle, and `_encode`, which every embedding goes
    through. Loading counts as work because the first search after a start
    spends seconds there, and a light that stayed dark through it would call a
    busy embedder idle.
    """

    @functools.wraps(method)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with EMBEDDER_ACTIVITY.working():
            return method(*args, **kwargs)

    return run


def _needs_the_cards(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Hold `GPU_ARBITER` and free the cards while `method` embeds.

    This is what makes "nothing else runs while the embedder is working" a
    property of the code rather than of the phase ordering. The ordering
    already covers the big cases -- the corpus and web phases both run before
    `graph.stream` -- and covers none of the small ones: a console search or an
    upload arrives on its own thread at any moment, and a run's own Planner map
    and Researcher search embed from inside a node. Those are the embeds that
    used to meet a seat's model on the cards.

    It goes on `_encode` alone, not on `_embedder_at_work`, and the difference
    is `_load_embedder`. That builds a handle and loads no weights, but it is
    also what `self.embedder.tokenizer` goes through -- the chunker asks for a
    tokenizer that runs in this process and touches no card. Evicting a seat's
    model in order to cut a document into passages would be a reload paid for
    nothing, so the marking and the arbitration are separate decorators rather
    than one.

    The meter is entered *inside* the arbiter by the decorator below, so a
    batch queued behind a seat reads as waiting rather than working and the
    console's light keeps meaning "the model is doing something". Holding the
    cards does not delay a stop: `encode` asks `should_stop` between batches,
    which a waiting embed reaches as readily as a working one.
    """

    @functools.wraps(method)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        from langgraph_agent.config import free_the_cards_for
        from langgraph_agent.control import GPU_ARBITER

        with GPU_ARBITER.exclusive("embedder"):
            # Freed inside the arbiter, so nothing loads into the gap between
            # the eviction and the embedding that wanted the room.
            free_the_cards_for(EMBEDDING_MODEL_NAME)
            return method(*args, **kwargs)

    return run


class GraphRAGKnowledgeBase(CorpusSpectralMixin):
    """Simple GraphRAG: NetworkX graph + Chroma vector store.

    Constructing this **creates the store on disk** -- `mkdir`, plus Chroma's
    own files under `chroma/`. That is why it is not the door most callers go
    through: `open_knowledge_base()` returns the corpus only if one already
    exists, and `get_knowledge_base()` is reserved for the act of building one.
    A corpus that appeared because a status poll happened to run is not a
    corpus anyone asked for.
    """

    # Declared on the class, not assigned in `__init__`, so an instance built
    # field by field around a fake collection -- which is how the corpus tests
    # avoid the model entirely -- still reads as "not loaded yet" rather than
    # raising on the attribute.
    _embedder: "OllamaEmbedder | None" = None

    # Same reasoning, and the same construction path: (nodes, edges) -> the
    # connectivity result computed at that shape. A cache must not depend on
    # which door built the object, so it defaults on the class rather than in
    # `__init__`.
    _connectivity_cache: "tuple[tuple[int, int], dict[str, Any]] | None" = None

    # The lexical half of search, built on first use from what the store holds
    # and dropped whenever the store changes. Declared on the class for the
    # same reason as the two above: an instance assembled field by field
    # around a fake collection still reads as "not built yet".
    _lexical_index: "BM25Index | None" = None

    # Where the loaded embedder sits (`ollama` -- the daemon places it) and a
    # note when the daemon split the model onto the CPU. Declared on the class
    # for the reason the three above are: an instance built field by field
    # around a fake embedder reads as "not placed yet" rather than raising.
    embedding_device: str | None = None
    embedding_device_note: str | None = None

    # The model this corpus is embedded with. There is exactly one, but the
    # name rides on the instance so the export can say which model produced
    # the corpus it is dumping and the calibration record is not mistaken for
    # another model's.
    embedding_model: str = EMBEDDING_MODEL_NAME
    # Set by `index_project_files` while it runs, so an embedding through
    # Ollama -- ~17s a batch for qwen3-embedding here -- stops between batches
    # instead of finishing a document that can take minutes.
    _should_stop: "Callable[[], bool] | None" = None

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = resolve_persist_dir(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma vector store
        self.chroma_client = chromadb.PersistentClient(str(self.persist_dir / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self._load_graph()

    @property
    def embedder(self) -> "OllamaEmbedder":
        """The daemon's embedding endpoint, built the first time something embeds.

        Deferred because building the handle used to load a model, and it is
        only needed to add a document or to run a query. It used to load in
        `__init__`, so opening the corpus at all -- a header poll, a document
        list -- paid for it.
        """
        model = self._embedder
        if model is None:
            model = self._load_embedder()
        return model

    @_embedder_at_work
    def _load_embedder(self) -> "OllamaEmbedder":
        """Point the corpus at the daemon: no weights load in this process.

        Placement -- which GPU, how much of it -- is the daemon's decision,
        reported back through `cpu_share` on the first real encode rather than
        chosen here.
        """
        model = OllamaEmbedder(self.embedding_model)
        self._embedder = model
        self.embedding_device = "ollama"
        self.embedding_device_note = None
        return model

    @_needs_the_cards
    @_embedder_at_work
    def _encode(self, texts: str | list[str]) -> Any:
        """Embed at `EMBEDDING_BATCH_SIZE`, stopping between batches when asked.

        Every encode goes through here, so no call site reaches the daemon at
        a size of its own, and a stopped run abandons the rest of a document
        between batches rather than after it.
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
        """Load graph from disk if exists."""
        graph_path = self.persist_dir / "knowledge_graph.json"
        if graph_path.exists():
            with open(graph_path, encoding="utf-8") as handle:
                self.graph = nx.readwrite.json_graph.node_link_graph(json.load(handle))

    def _save_graph(self) -> None:
        """Save graph to disk, whole or not at all.

        Written to a temporary file and renamed over the real one, the way
        `calibrate_relevance_floor` writes its record. `json.dump` straight
        into the file truncated it first and filled it after, and the thread
        doing that can die in between: the console's exit and install.sh's
        start-up check both end the process under a rebuild that is still
        running on a daemon thread, and this is called once per document. A
        file cut off mid-write does not load (`_load_graph` raises on it), and
        a corpus whose graph does not load cannot be opened at all -- by the
        header, the Researcher, or the rebuild that would have repaired it.
        """
        graph_path = self.persist_dir / "knowledge_graph.json"
        node_link_data = nx.readwrite.json_graph.node_link_data(self.graph)
        temporary = graph_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(node_link_data, handle)
        os.replace(temporary, graph_path)

    def chunk_text(self, content: str) -> list[str]:
        """Split a document into passages the embedder can actually read whole.

        One tokenizer pass with an offset mapping, then `_chunk_windows` over
        the token indices; each window's character span runs from the start of
        its first token to the end of its last, so the text between tokens --
        whitespace, indentation, blank lines -- is carried rather than dropped.
        Concatenating the chunks therefore reproduces the document apart from
        the deliberate overlap, and the join is checked in the tests.

        `verbose=False` suppresses the tokenizer's own "sequence longer than
        the maximum" warning. It is silenced only because this function is the
        thing that answers it: the sequence *is* longer than the window, that
        is why it is being cut up, and the warning would otherwise fire once
        per document on every reindex.
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
            # Content the tokenizer maps to nothing -- whitespace, or a run of
            # characters with no token of their own. There is no passage to
            # embed, and returning the raw text would put a vector of noise in
            # the store under the document's name.
            return []

        chunks = [
            content[offsets[start][0] : offsets[end - 1][1]]
            for start, end in _chunk_windows(
                len(offsets), CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            )
        ]
        return self._fit_chunks(chunks)

    def _fit_chunks(self, chunks: list[str]) -> list[str]:
        """Trim any chunk that re-tokenizes past the window, and say which end.

        Slicing on the parent document's token boundaries does **not**
        guarantee the slice re-tokenizes to the same length. A word-piece
        tokenizer decides on context, so a fragment cut mid-word encodes
        differently standalone than it did inside the document -- measured on
        this corpus, 16 of 1,052 full-size chunks came back one token longer,
        which put them at 257 against a 256 window and handed them straight
        back to the truncation this whole change exists to remove.

        Measuring the drift and padding the constant would be guessing with an
        extra step: +1 is what this corpus does today, not a bound anybody can
        prove for the next document. So the chunks are re-encoded and the
        overflow is cut, which makes the window a fact rather than an estimate.
        Only the overflowing chunks are touched, and the batched encode of the
        rest is a few tens of milliseconds per reindex.

        **Which end is trimmed is the part that matters.** A chunk's tail is
        covered by the next chunk's overlap and its head by the previous one's,
        so trimming into a neighbour's overlap loses nothing from the corpus --
        except at the two ends of the document, which have no neighbour. The
        last chunk is therefore trimmed at the *head* and every other at the
        tail; trimming the last one's tail would drop the final tokens of the
        file, silently, which is the original bug in miniature.
        """
        if len(chunks) < 2:
            # A lone chunk is the whole document, un-sliced, so it re-tokenizes
            # to exactly what it was measured as and cannot overflow.
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
        """Add a document to the knowledge base.

        The document is embedded as several chunks and stored as several rows,
        keyed `doc_id#0000`, `doc_id#0001`, ... -- see `CHUNK_MAX_TOKENS` for
        what embedding it as one row cost. The graph is unaffected: it still
        gets exactly one node per document, with entities drawn from the whole
        text, so chunking changes what search can find and not what the corpus
        is shaped like.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (path, type, etc.)

        Returns:
            How many chunks this document became. Reported rather than
            discarded because it is the one number that says whether a
            document is *reachable*: chunking is what made a long file
            searchable past its first thousand characters, and a caller adding
            one document at a time -- an upload -- has no other way to see that
            it landed as several passages rather than as a truncated header.
        """
        chunks = self.chunk_text(content)

        # Embedded before anything is deleted. A failed or stopped embed then
        # leaves the document exactly as the store held it, rather than
        # deleting it and failing to put it back: on 2026-09-18 a load that
        # ran out of GPU memory at console startup removed CLAUDE.md,
        # frontend/index.html and pyproject.toml from the corpus outright, and
        # the header read `stale: 3 not indexed` until the next rebuild. Old
        # vectors still carry the old `sha`, so the next rebuild re-embeds it.
        #
        # Chunks are embedded in one batched call rather than one per chunk:
        # the model is the expensive thing on this machine and batching is
        # most of what makes a reindex of ~1,100 chunks finish in the time a
        # reindex of 77 documents used to take.
        embeddings = self._encode(chunks).tolist() if chunks else []

        # A document's previous chunks are deleted before the new ones land,
        # rather than left to be overwritten by `upsert`. A file that shrank
        # between reindexes -- 10 chunks down to 3 -- overwrites 0..2 and
        # leaves 3..9 in the store, still matching queries with text the file
        # no longer contains. That is the same "a reindex rebuilds rather than
        # accumulates" rule `index_project_files` follows for whole documents,
        # applied one level down.
        #
        # Keyed on the `doc_id` metadata, so it also sweeps up the single
        # unsuffixed row a pre-chunking store holds for this document.
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            # A collection that cannot filter by metadata (an older store, or
            # the fakes the corpus tests build) still gets a correct insert
            # below; what is lost is the sweep of rows this call is replacing.
            pass
        self.collection.delete(ids=[doc_id])

        if chunks:
            base = dict(metadata or {})
            # The fingerprint rides on every chunk, so `index_project_files`
            # can ask "is this document still the one I embedded?" from the
            # metadata it already fetches to prune with. Computed here rather
            # than passed in, for the reason `doc_id` and `chunk_count` are:
            # it describes what the store did with the text, not what the
            # caller knows about the file, and one caller forgetting it would
            # silently cost that document its reuse.
            base["sha"] = _content_sha(content)
            self.collection.upsert(
                ids=[
                    f"{doc_id}{CHUNK_ID_SEPARATOR}{i:04d}" for i in range(len(chunks))
                ],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[
                    {**base, "doc_id": doc_id, "chunk_index": i, "chunk_count": len(chunks)}
                    for i in range(len(chunks))
                ],
            )

        self._add_to_graph(doc_id, content, metadata)

        # The lexical index describes a corpus that no longer exists. Dropped
        # rather than amended: the next search rebuilds it from the store in
        # milliseconds, and an index maintained in parallel with Chroma is a
        # second account of the same corpus, free to disagree with it.
        self._lexical_index = None

        self._save_graph()
        return len(chunks)

    def _add_to_graph(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """The half of `add_document` that costs nothing: node and entities.

        Split out because a rebuild that keeps a document's vectors still has
        to put it back in the graph -- `index_project_files` clears the graph
        up front, so a document whose embeddings were reused would otherwise
        vanish from it while staying perfectly searchable. Entity extraction is
        a regex over the text and the whole graph rebuilds in 0.04s, so the
        cheap half is simply always done.
        """
        # `type` on a node is structural -- document vs entity -- and drives how
        # the console draws it. Metadata carries its own `type` (python,
        # markdown), which collides as a duplicate keyword and takes down the
        # whole insert, so it goes in under its own name.
        node_attrs = {k: v for k, v in (metadata or {}).items() if k != "type"}
        if metadata and "type" in metadata:
            node_attrs["doc_type"] = metadata["type"]

        self.graph.add_node(
            doc_id,
            type="document",
            content=content[:200],  # Store snippet
            **node_attrs
        )

        # Extract simple entities (words that look like important terms).
        # The strip set has to cover code punctuation as well as prose: over a
        # corpus that is mostly source files, a prose-only `.,!?;:` leaves
        # entities like `Builder")` standing as graph nodes.
        #
        # `ENTITY_STOPWORDS` is what stops the capital rule firing on words
        # whose capital comes from where they sit rather than what they mean --
        # a sentence opener, a docstring heading, a Python literal. Without it
        # `False` and `Returns` are the 4th and 6th best-connected nodes in
        # this corpus's graph, ahead of `Fiedler` and `Cheeger`, and every
        # document carrying a docstring is joined to every other one through a
        # relation that means nothing.
        # A fetched web page is a *retrieval source*, not knowledge-graph
        # material, and the capital rule is far worse on it than on this
        # project's own files. `ENTITY_STOPWORDS` is a hand-audited list, and it
        # was audited against source code and numpy docstrings, where the forced
        # capitals are `Returns`, `Every`, `False`. Web prose opens sentences
        # with a different vocabulary entirely, and none of it is on the list.
        # Measured on 2026-09-09: 15 fetched pages minted **551 entities that no
        # project document mentions -- 19% of the whole graph, 36.7 per page** --
        # `Although`, `Afterward`, `Altogether`, `Again`, `Accessed`. Skipping
        # them took the graph from 2,830 entities to 2,285 and duplicate
        # candidates from 205 to 171. Those edges are read as evidence by
        # `neighborhood`, `topics` and `duplicate_entities`, so the graph
        # degrades as the corpus grows.
        # The page is still chunked, embedded and fully retrievable; it simply
        # stops voting on what the entities of this project are. Markup, script
        # and config follow the same rule: see `ENTITY_FREE_SUFFIXES`.
        entities = []
        for word in (content.split() if _mints_entities(doc_id) else []):
            token = word.strip("\"'`()[]{}<>.,!?;:*=+-/\\|")
            if (
                len(token) > 4
                and token[0].isupper()
                and token.replace("_", "").isalnum()
                and token.lower() not in ENTITY_STOPWORDS
            ):
                entities.append(token)

        # Add entities and relationships
        for entity in set(entities):
            self.graph.add_node(entity, type="entity")
            self.graph.add_edge(doc_id, entity, relation="mentions")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Matches on chunks and answers in documents. `id` stays the document's
        path -- the thing callers index the graph with, print as a filename and
        derive an entity from -- while `content` becomes the passage that
        actually matched rather than the document's first 200 characters. That
        is the half of chunking the Researcher feels: what it forwards to the
        Builder (up to `RESEARCH_SNIPPET_CHARS` of each passage, in `nodes.py`)
        used to be the opening characters of whichever file's *header* scored
        best.

        Chunks are oversampled and then collapsed onto their documents, keeping
        each document's best-scoring chunk. Collapsing is what makes `top_k`
        mean what every caller already assumed it meant: without it a query
        that matches six passages of one file would return that file six times
        and crowd out five other sources, which is worse than the pre-chunking
        behaviour rather than better.

        The retrieval is **hybrid**: the dense window is re-ranked against BM25
        before it is collapsed. Dense similarity is a poor instrument for "this
        passage contains this exact rare identifier", which is what a plan
        naming `BUILDER_DEADLINE_SECONDS` is really asking. Measured through
        this method against the real store, on 541 identifiers each defined in
        exactly one project file, the defining file came first 53.4% of the
        time on dense alone and 65.1% with the re-rank (McNemar p < 0.001, 85
        fixed against 22 broken). The prose case improved too -- 66.0% to 68.2%
        on 400 held-out passages -- so this is not a code-search special case
        bought at the expense of ordinary questions. See `lexical.py` for why
        it re-ranks rather than retrieving in parallel, and why ranks are fused
        rather than scores.

        `score` therefore stays exactly what it was: the dense cosine of the
        chunk that matched. Every candidate comes from the dense window, so
        there is no result whose score had to be invented -- which matters
        because the relevance floor is read off the first one to decide
        whether the corpus answered at all.

        Args:
            query: Search query
            top_k: Number of documents to return

        Returns:
            List of results with content, metadata, and graph context
        """
        if top_k <= 0:
            return []
        # An empty query is not a question and must not answer like one. The
        # empty string still embeds, and on this corpus it matched five chunks
        # topping 0.412 -- over the relevance floor of the time, so those would have
        # been formatted as findings and announced as "Research complete". The
        # fabricated retrieval hit, arriving through the query this time.
        if not query or not query.strip():
            return []

        # Generate query embedding. `_encode` is what keeps it at the batch cap
        # and draws no progress bar.
        query_embedding = self._encode(query).tolist()

        # Widen once if one document monopolised the first window of hits.
        # A focused query genuinely can match a dozen passages of the same
        # large file before it matches anything else, and collapsing those
        # onto one document would answer a request for five sources with one
        # -- narrower than the behaviour chunking replaced. The ladder is two
        # rungs and stops early: there is no point asking a third time, and an
        # unbounded search for breadth would let one query walk the corpus.
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
                # Either the caller has what it asked for, or the collection
                # returned fewer chunks than requested and has no more to give.
                break

        return results

    @property
    def lexical_index(self) -> "BM25Index | None":
        """The BM25 index over the stored chunks, built on first use.

        Built from the store rather than kept alongside it, so it cannot drift:
        whatever `search` re-ranks is what `search` retrieved. `add_document`
        and `clear` drop it instead of updating it -- rebuilding over this
        corpus's ~1,200 chunks is milliseconds, and an index that edits itself
        in place is a second thing that can disagree with Chroma.

        Returns `None` rather than raising if the store cannot produce its
        documents. A collection too old or too foreign to answer `get` is a
        reason to fall back to dense-only retrieval, not a reason for search to
        stop working -- the lexical half is an improvement to the ranking, and
        the dense half is still a correct answer without it.
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
        """Fuse the dense window's order with BM25's, or leave it alone.

        Returns `None` -- meaning "use the order Chroma gave" -- when there is
        no index or fewer than two candidates to reorder. A single hit has no
        ranking to improve, and paying for an index build to discover that is
        the sort of cost that lands on the console's five-second poll.
        """
        chunk_ids = [str(c) for c in ((raw.get("ids") or [[]])[0] or [])]
        if len(chunk_ids) < 2:
            return None

        index = self.lexical_index
        if index is None:
            return None

        by_lexical = lexical_order(query, chunk_ids, index)
        if by_lexical == chunk_ids:
            # The lexical half agrees, or had nothing to say. Either way the
            # fusion would return the order we already have.
            return None
        return reciprocal_rank_fusion(chunk_ids, by_lexical)

    def _collapse_chunk_hits(
        self, raw: "Mapping[str, Any]", top_k: int, order: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Fold chunk hits onto the documents they came from, best chunk first.

        Chroma returns hits best-first, so the first chunk seen for a document
        is its best one and every later chunk of it only adds to the count.
        `order`, when given, replaces that ordering with the hybrid one --
        the same hits, re-ranked, so "best chunk first" still holds and every
        result still carries the dense score its chunk was retrieved with.
        """
        ids = raw.get("ids") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        # Guard against empty collections / no hits
        if not ids or not ids[0]:
            return []

        collapsed: list[dict[str, Any]] = []
        by_document: dict[str, dict[str, Any]] = {}

        # Walk the hits in the hybrid order when there is one, and in Chroma's
        # otherwise. A chunk id `order` names that this response does not hold
        # is skipped rather than trusted: the two come from the same query, but
        # the collapse reads `documents`/`distances` positionally and an id
        # without a row here would index the wrong one.
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
                # Full. Keep draining the hits already paid for so
                # `chunks_matched` counts every match, but admit no new
                # documents.
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

            # Add graph neighbors
            if doc_id in self.graph:
                neighbors = list(self.graph.neighbors(doc_id))[:5]
                result["related_entities"] = neighbors

            by_document[doc_id] = result
            collapsed.append(result)

        return collapsed

    def query_graph(self, entity: str, hops: int = 2) -> dict[str, Any]:
        """Query the knowledge graph for entity relationships.

        Args:
            entity: Entity name to search for
            hops: Number of hops to traverse

        Returns:
            Entity info with relationships
        """
        # One spelling of the fuzzy match, so the console and the Researcher's
        # tool cannot resolve an id differently.
        found, alternatives = self._match_node(entity)
        if found is None:
            return {"error": f"Entity '{entity}' not found"}
        typed, entity = entity, found

        # Undirected, for the reason `neighborhood` is: every edge runs
        # document -> entity, so an entity has in-edges only and a directed walk
        # from one reaches nothing. It returned the entity alone for every
        # entity in the corpus -- `Planner`, 21 edges, reported 0 against a real
        # 2-hop neighbourhood of 614 -- which reads as "this term connects to
        # nothing" rather than as a broken walk. See `test_graph_queries.py`.
        undirected = self.graph.to_undirected(as_view=True)
        neighbors = nx.single_source_shortest_path_length(
            undirected, entity, cutoff=hops
        )

        # Build subgraph
        subgraph = self.graph.subgraph(neighbors.keys())

        result: dict[str, Any] = {
            "entity": entity,
            "neighbors": list(neighbors.items()),
            "subgraph_nodes": len(subgraph.nodes()),
            "subgraph_edges": len(subgraph.edges()),
        }
        if entity != typed:
            # The Researcher reads this too: a relationship question answered
            # about a different node than the one it named should say so.
            result["resolved_from"] = typed
            result["alternatives"] = alternatives
        return result


    def _resolve_node(self, node_id: str) -> str | None:
        """Resolve a node id; see `_match_node`. Both entry points use it."""
        return self._match_node(node_id)[0]

    def _match_node(self, node_id: str) -> tuple[str | None, list[str]]:
        """Resolve a typed id to a node, and name the others it could have meant.

        The fallback used to be the first node whose id contained the text, in
        whatever order the graph enumerated its nodes -- so the same loose id
        could trace a different node after a reindex, and nobody was told there
        had been a choice. Candidates are ranked instead: the exact id ignoring
        case, a document by its file name or stem, an id starting with the text,
        an id containing it; within a rank the shorter id wins, then the better
        connected node. The runners-up come back with the match, so a caller can
        say what it chose between.
        """
        if node_id in self.graph:
            return node_id, []

        # `"" in anything` is True, so a blank id matched on the first
        # comparison and resolved to whatever the graph enumerated first. The
        # caller asked about nothing and got a real document's neighbourhood.
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
        """Render one graph node in the shape the console draws."""
        attrs = dict(self.graph.nodes[node_id])
        node_type = attrs.pop("type", "entity")

        # The stored content snippet is for retrieval, not for a properties
        # blob the user hovers over; drop it rather than ship 200 chars per node.
        attrs.pop("content", None)

        # A bare basename collides -- this project has two README.md files, and
        # two nodes labelled the same are unreadable on a graph. Keep the
        # parent directory for anything that is not at the repository root.
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
        """List every document node in the knowledge graph.

        Documents are the centres the console sweeps from. A centre never
        appears in its own neighbourhood, so the client seeds its node map from
        this list before it queries anything.
        """
        documents = [
            {"id": node_id, "title": self._node_record(node_id)["label"], "node_type": "document"}
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "document"
        ]
        documents.sort(key=lambda doc: str(doc["title"]))
        return {"documents": documents}

    def _sign_split(self, keep: set[Any], centre: str) -> dict[str, Any]:
        """Fiedler sign bipartition of a swept neighbourhood, oriented on the centre.

        The A4 application of the spectral applicability study, and the
        cheapest technique in it: one eigenvector, 1.7-3.3ms on subgraphs of
        43-291 nodes, against the 1.0ms `neighborhood()` itself costs. That is
        why it is a flag rather than always-on -- it triples the cost of a call
        the console makes on every click, and a caller who only wants the node
        list should not pay it.

        Sides are named `center` and `other` rather than by the sign of the
        eigenvector, whose direction is arbitrary: an eigenvector and its
        negation are equally valid, so a raw sign would swap the two halves
        between runs for no reason. Orienting on the centre also makes the
        answer the one the caller asked for -- "what clusters with the node I
        looked up, and what is peripheral to it".

        `mu_2` is returned because the split is always *available* and only
        sometimes *meaningful*. A sign cut exists for any connected graph; what
        says whether it corresponds to a real division is how small `mu_2` is,
        and on these subgraphs it ranges from 0.134 (a two-hop sweep, barely
        divided) to 0.006 (a five-hop sweep spanning two topic areas). The
        caller is given the number rather than a bare verdict because the
        threshold that matters depends on what the split is being used for --
        here, whether it is worth drawing.

        Runs on the largest component: `min_degree` pruning can disconnect the
        swept subgraph -- measured at depth 3, min_degree 3, which left one
        node isolated -- and on a disconnected graph the Fiedler vector is a
        component indicator, so the "split" would just be that stray node
        against everything else. Nodes outside the largest component are
        reported as `detached` and given no side, which is the truth: they were
        not part of the division.
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
        """Return a node's neighbourhood as drawable nodes and edges.

        `query_graph` answers "how big is this neighbourhood"; this answers
        "what is in it", in the record shape the console renders directly.

        Args:
            node_id: Centre of the traversal; resolved fuzzily.
            max_depth: Hops to traverse out from the centre.
            min_degree: Drop entity nodes with fewer edges than this. A sweep
                passes 2, because `add_document` mints an entity for every
                capitalised word and the one-document ones bury the structure
                worth looking at. A single trace passes 1, so a node the user
                asked for by name never has its neighbours hidden.
            split: Also compute the Fiedler sign bipartition, tagging every
                node `center` or `other`. Off by default: it triples the cost
                of this call, and only a caller that is going to draw the
                division wants it. See `_sign_split`.

        Returns:
            center_node, related_nodes (excluding the centre), edges, and
            totals; plus `split` when asked for.
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

        # Traverse undirected: every edge runs document -> entity, so a directed
        # walk from a document reaches its entities but never the sibling
        # documents that share them, which is the structure worth showing.
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
            # The centre's own record, because only the graph knows whether it
            # is a document or an entity. Without it the console assumed a
            # document and drew every traced entity as a file.
            "center": self._node_record(centre),
            "related_nodes": related,
            "edges": edges,
            "total_nodes": len(related),
            "total_edges": len(edges),
        }

        if centre != node_id:
            # Said whenever the id typed was not the id found, so a trace of
            # the wrong node reads as a choice rather than as the answer.
            result["resolved_from"] = node_id
            result["alternatives"] = alternatives

        if split:
            division = self._sign_split(keep, centre)
            # `sides` is folded onto the node records and dropped from the
            # summary: the console draws nodes, not a lookup table, and
            # shipping both would let the two disagree.
            sides = division.pop("sides", {})
            for record in related:
                record["side"] = sides.get(record["id"])
            result["split"] = division

        return result

    def overview(self, min_degree: int = 4, include_isolated: bool = False) -> dict[str, Any]:
        """The whole corpus as one drawable graph -- what the console's sweep draws.

        The console used to assemble this itself: `list_documents`, then one
        `neighborhood` call per document -- 97 round trips on this corpus, each
        a line in the server log, before anything was drawn. Every document is
        always kept and entities are kept by degree, so for any depth of one or
        more the union of those neighbourhoods is exactly every document plus
        every entity with at least `min_degree` edges. That is computed here
        once, from the same degree `neighborhood` reads.

        A document linked to nothing in that subgraph -- a fetched page, an
        entity-free source file, a document whose entities are all rarer than
        `min_degree` -- is a dot joined to nothing, and a sweep has no use for
        it. It is left out unless asked for, and counted in
        `unlinked_documents` so the drawing does not silently lose it.
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
            # Chroma is a separate store from the graph; a failure here should
            # not take out the node/edge counts that come from memory.
            chunks = 0

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()

        # The console polls this every five seconds and the eigendecomposition
        # is ~44ms on a 920-node graph, growing with the corpus. It is cached
        # against (nodes, edges) because those are what every mutation path in
        # this class moves: `add_document` only ever adds, and `clear` zeroes
        # both. Re-adding an identical document changes neither count -- and
        # changes no structure either, so the cached answer is still the right
        # one. Nothing here removes an edge without removing a node.
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
        """Empty the knowledge base, keeping the files that hold it.

        The store is emptied in place rather than deleted: Chroma has this
        directory open, and pulling it out from under a live client is a worse
        failure than an empty collection. What is left behind is the same shape
        a reindex leaves -- a real store with nothing in it.

        Chroma goes first, and the graph is only cleared once it has. The two
        halves answer different questions (search, and structure), so a run that
        wiped one and failed on the other would leave the corpus disagreeing
        with itself while reporting success. On failure this raises with both
        intact.

        The floor record goes with them, and that is a third half rather than
        tidiness: a floor is measured on *these texts* with this model, so a
        record outliving the corpus it was taken on is a number about a corpus
        that no longer exists -- and `_calibrate_the_floor_before_the_run` reads
        a record's mere presence as `known`, so the run that rebuilds the corpus
        would never measure a floor for it. The console would go on showing that
        floor over an empty corpus meanwhile. Removed last, after both halves
        are actually empty, so a failed wipe leaves the corpus and its floor
        agreeing.

        Returns:
            What was removed, plus the (now zeroed) stats.
        """
        removed_nodes = self.graph.number_of_nodes()
        removed_edges = self.graph.number_of_edges()

        existing = self.collection.get(include=[]).get("ids", [])
        if existing:
            self.collection.delete(ids=existing)

        self.graph.clear()
        self._lexical_index = None
        # Without this the clear lives only in memory: the next process start
        # reloads the old graph off disk and the corpus comes back.
        self._save_graph()

        floor_record = _floor_calibration_path(self.persist_dir)
        removed_floor = floor_record.exists()
        floor_record.unlink(missing_ok=True)

        return {
            "removed_chunks": len(existing),
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "removed_floor": removed_floor,
            **self.stats(),
        }

    def export_corpus(self) -> dict[str, Any]:
        """The whole corpus as one JSON-serialisable document.

        Embeddings are left out. They are the bulk of the store by a wide
        margin and the least useful part of a dump: the embedder is local, so
        anything reading this file back can regenerate them, and a reader
        without the same model could not use them anyway. The file says so
        itself rather than leaving the omission to be discovered.

        The graph half is `node_link_data`, which is exactly the on-disk format
        `_save_graph` writes, so it can be compared against
        `knowledge/knowledge_graph.json` directly.
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
                    # The document this passage came from, lifted out of the
                    # metadata so the export stands on its own: a row's id is
                    # `path#0007`, and a reader should not have to know how to
                    # parse that to group the file back together.
                    "doc_id": _document_id_of(
                        chunk_id, metadatas[i] if i < len(metadatas) else None
                    ),
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                for i, chunk_id in enumerate(ids)
            ]
        except Exception as exc:
            # Same posture as `stats()`: a Chroma failure must not cost us the
            # graph half of the export as well.
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


# Files worth indexing, and the directories that only add noise. One list, so
# the rebuild a run does and the walk `corpus_health` compares against cannot
# drift into describing different corpora.
#
# Markup, script and config are in it because the project is not only prose and
# Python: the console itself is `frontend/index.html`, and none of it could be
# retrieved. Measured on 2026-09-12 by embedding that file's passages against
# five questions about the console: it would rank first on three -- "how does
# the console reattach to a run after a page reload" at 0.644, against a best of
# 0.426 from the corpus as it stood -- and it lifts one question the corpus could
# not answer at all over the relevance floor, 0.308 before and 0.383 after.
# Asked about the Corpus tab without it, a discussion Builder proposed changes to
# "a React/Vue/Angular component" the project does not have. These files are
# retrievable but mint no entities: see `ENTITY_FREE_SUFFIXES`.
PROJECT_INDEX_PATTERNS = (
    "**/*.py", "**/*.md", "**/*.txt", "**/*.rst",
    "**/*.html", "**/*.js", "**/*.css",
    "**/*.sh", "**/*.toml", "**/*.yml", "**/*.yaml", "**/*.ini", "**/*.cfg",
)
# Matched as plain substrings of the path, so no globs: "*.egg-info" never
# matched anything and let build metadata (SOURCES.txt, top_level.txt) into
# the corpus as if it were project knowledge.
PROJECT_INDEX_EXCLUDES = (
    # `.git/`, not `.git`: as a plain substring it also matched `.github/`, so
    # the CI workflow went out with the repository's object store.
    "__pycache__", ".git/", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", "build/", "dist/", ".egg-info",
    "knowledge/", "scripts/", ".qwen/", ".claude/",
    # Seat-diagnostic sweeps. `.gitignore` already calls these "per-run
    # measurements against non-deterministic models, not history", but the walk
    # is a glob and not git, so being gitignored kept them out of commits and
    # not out of the corpus: five sweeps from one afternoon sat in the store as
    # indexed documents, answering questions with a week-old measurement of a
    # seating nobody runs any more. `reports/*.md` is written deliberately and
    # stays indexed; only the timestamped sweeps under it are excluded.
    "reports/diagnostics/",
    # Where an agent run's own write-ups land -- a machine inspection, an
    # integration sketch, a brainstorm nobody finished. Excluded for the reason
    # the sweeps above are: the walk is a glob and not git, so an untracked file
    # here is corpus material within seconds of being written, and these are
    # per-run notes about one afternoon rather than knowledge of the project.
    # Measured on 2026-09-18: one file here had minted 98 entities -- 43 of them
    # mentioned by no other document, 4.7% of the graph -- from a template whose
    # own body says the logic it analyses was never attached, and a heading in
    # the other took a positional token to the four documents that fail the
    # entity guard in `tests/test_claims.py`. A directory rather than an entry
    # per file, because the class recurs: four artifacts of this shape were
    # deleted on 2026-09-09 and the list grew no way to keep the fifth out.
    "experimental/",
    # Application's own source files -- these are the program itself, not
    # project knowledge to be retrieved. The operator must explicitly request
    # embedding (via upload or project opt-in) for any of these to enter the
    # corpus. This prevents the corpus from being polluted with the program's
    # own code, tests, prompts, and documentation.
    "src/", "tests/", "prompts/", "frontend/", "spectral_graph/",
)

# Above this size a file is not a document at all -- a data dump, a minified
# bundle, a log -- and reading it into the corpus indexes something nobody
# wrote. It was 100,000, and that number outlived its own justification: it
# said such a file "would dominate the embedding budget", which was true when
# `add_document` embedded a whole file as ONE vector and a long document's
# single embedding competed with everyone else's. Chunking ended that, and
# `search` collapsing chunks back onto documents ended it twice -- a document
# now takes exactly one result slot however many chunks it holds. Measured on
# this corpus: CLAUDE.md carries 7.1% of all 1,687 chunks and returned in
# exactly 1 of 5 slots on every query tried, never more.
#
# What the old number did instead was dictate the shape of the project.
# `corpus_health.py` exists as a separate module because adding the staleness
# check pushed `graphrag_server.py` from 98,920 characters to 104,582 -- the
# module that defines the corpus would have dropped out of it. By 2026-09-11
# `nodes.py` stood at 82% of the limit, `graphrag_server.py` at 75%,
# `test_graph.py` at 73%, and CLAUDE.md had 13 characters left, so the next
# paragraph anyone wrote would have silently cost the project its own
# documentation. A constant that decides how files must be split is not
# measuring anything about knowledge.
#
# The remaining real risk sets the ceiling, and it is why this is 250,000 and
# not unbounded: `search` retrieves a window of chunks *before* collapsing
# them, so a document holding a large enough share of the corpus can fill that
# window with itself and starve every other source -- the case
# `SEARCH_ESCALATION` widens the window for. At 7.1% the largest document here
# is nowhere near it; a file several times this limit would be.
MAX_INDEXABLE_BYTES = 250_000

# Where a document uploaded from the console lands, relative to the project
# root. An upload is written to disk *before* it is embedded, and that ordering
# is the design rather than a convenience: the corpus is a function of what is
# on disk, and `index_project_files` rebuilds it from there -- clearing the
# graph and pruning every stored row whose document is not in the walk. So a
# document embedded straight into the store and nowhere else survives exactly
# until the next reindex, which then deletes it silently, in a pass that
# reports success and a file count that looks right. Writing the file first is
# what makes an upload part of the corpus rather than a guest in it, and it is
# why this directory must stay out of `PROJECT_INDEX_EXCLUDES`.
UPLOADS_DIR = "uploads"

# Where the online research phase writes the pages it fetched, relative to the
# project root. It lives here rather than in `web_research` because
# `add_document` has to recognise one of these on sight -- see
# `_is_web_document` -- and because `web_research` imports this module, so the
# constant could not travel the other way.
WEB_RESEARCH_DIR = "research/web"

# The suffixes an upload may carry, derived from the walk's own patterns rather
# than restated beside them. The two have to agree exactly or an upload is
# accepted, embedded, and then dropped at the next rebuild for a reason nobody
# is told -- the same drift `PROJECT_INDEX_EXCLUDES` had while two scripts kept
# their own copy of it. Compared case-insensitively but *stored* lower-cased,
# because the walk is a glob and a glob is case-sensitive here: `NOTES.MD`
# written as given is a file the reindex cannot see.
INDEXABLE_SUFFIXES = tuple(sorted({Path(pattern).suffix for pattern in PROJECT_INDEX_PATTERNS}))


def _is_web_document(doc_id: str) -> bool:
    """True for a page the online research phase fetched.

    Decided from the **path**, never from the call site, for the same reason
    `_document_metadata` is shared between the upload and the walk: a reindex
    re-reads these files as ordinary markdown, so a decision made only where a
    document is first stored would be silently reversed the next time anyone
    rebuilt. Matching on the parent directories rather than a prefix keeps it
    working when the root is absolute, which it is in every test.
    """
    parts = PurePosixPath(str(doc_id).replace("\\", "/")).parts
    wanted = PurePosixPath(WEB_RESEARCH_DIR).parts
    return len(parts) > len(wanted) and parts[-len(wanted) - 1 : -1] == wanted


# Suffixes whose documents are retrievable but mint no entities. Their capitals
# are identifiers and interface strings, not terms the project is about:
# measured on 2026-09-12, the seven such files in this checkout would mint 72
# entities the graph does not hold, 56 of them from `frontend/index.html` alone
# -- `BRIDGE_VERDICTS`, `CLEAR_ARMED`, `ACTIVE_SEAT`, and the openers of its
# comments and messages (`Copying`, `Dimming`, `Cancelling`). The decision
# `_is_web_document` makes for fetched pages, for the same reason: they would
# vote on what this project's entities are without saying anything about them.
# Python stays out of the set, because its docstrings are where the terms live.
ENTITY_FREE_SUFFIXES = frozenset(
    {".html", ".js", ".css", ".sh", ".toml", ".yml", ".yaml", ".ini", ".cfg"}
)

# The `type` a document is stored under, by suffix. Prose falls through to
# "markdown", which is what .md, .txt and .rst have always been filed as.
_DOCUMENT_TYPES = {
    ".py": "python",
    ".html": "html", ".js": "javascript", ".css": "css", ".sh": "shell",
    ".toml": "config", ".yml": "config", ".yaml": "config", ".ini": "config",
    ".cfg": "config",
}


# How many runners-up a loose node id reports beside its match.
MAX_MATCH_ALTERNATIVES = 5

# The most of a passage's first line a search result reaches back for. A chunk
# starts wherever the token arithmetic cut it, usually mid-line; reaching back
# to the start of the line reads better, but a minified line has no useful start.
MAX_PASSAGE_LEAD_CHARS = 240


def _passage_location(doc_id: str, passage: str) -> tuple[int, str] | None:
    """Where a retrieved passage sits in its file: (1-based line, passage from that line's start).

    A chunk begins where the token arithmetic cut it -- boundaries are not
    snapped, see `CHUNK_OVERLAP_TOKENS` -- so the passage the Builder was handed
    opened mid-statement ("in self.graph: neighbors = ...") and carried no
    location at all. The line is what lets it open the file at the right place.

    Read from the file on disk, because the store holds chunks, not documents.
    None when the file cannot be read or no longer contains the passage -- a
    document edited since it was indexed, where a guessed line would point at
    the wrong code.
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
    """Whether a document takes part in entity extraction at all.

    Decided from the path alone, like `_is_web_document`, so an upload, the
    first index and every rebuild after it reach the same answer.
    """
    if _is_web_document(doc_id):
        return False
    suffix = PurePosixPath(str(doc_id).replace("\\", "/")).suffix.lower()
    return suffix not in ENTITY_FREE_SUFFIXES


def _document_metadata(path: Path) -> dict[str, str]:
    """The metadata a document is stored under. One definition, two callers.

    The reindex and an upload have to agree on this exactly: `path` is what the
    graph node is keyed by and what `search` reports as a filename, and `type`
    is what the console colours a node with. Two spellings of it would file the
    same file twice under two descriptions.
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

    See `UPLOADS_DIR` for why the file comes first. What is left after this
    call is a document indistinguishable from any other in the corpus: the same
    id shape, the same metadata, and a place in the walk, so the next reindex
    re-reads it instead of pruning it.

    Every refusal below is a `ValueError` naming what was wrong, because each
    alternative to refusing is worse than a failed upload and none of them
    announces itself. A file the embedder cannot read becomes a vector of noise
    filed under a real filename. A file over the walk's size limit is embedded
    now and dropped at the next rebuild. A name with a path in it writes
    outside the directory the reindex looks at, so the document is embedded and
    then pruned by the very next pass.

    Args:
        kb: The corpus to add to. This is the *creating* door's knowledge base:
            an upload is a request for a corpus to hold it.
        name: The filename as the browser reported it. Only its last component
            is used.
        content: The document's text, already decoded.
        root: Project root the walk runs from, so the id this stores under is
            the id the walk will produce.

    Returns:
        What was stored -- the path, whether it replaced a document already
        there, how many passages it became -- plus the corpus stats, so the
        console can repaint its header from the same reply.

    Raises:
        ValueError: the name, the type, the size or the content is one the
            corpus cannot take. The message says which, and why.
    """
    # Only the last component, so a name carrying a path cannot place the file
    # outside `uploads/`. `..` survives that (`PurePosixPath("..").name` is
    # `".."`), so it is refused by name rather than left to the suffix check.
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
    # The walk's own test, character for character. Accepting a larger file
    # here would embed it now and silently skip it at every reindex after.
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
    # Read before the write, so "replaced" is a fact rather than a guess. An
    # upload of a name already there is the ordinary way to correct a document,
    # and `add_document` deletes that document's previous chunks before the new
    # ones land, so nothing of the old version is left behind in the store.
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


def iter_project_files(
    root: str = ".", exclude_dirs: tuple[str, ...] | list[str] | None = None
) -> list[Path]:
    """Collect the project files worth indexing.

    A run's generated project under `projects/` is walked only once the
    operator has opted it in -- see `langgraph_agent.projects`.
    """
    excludes = tuple(exclude_dirs) if exclude_dirs is not None else PROJECT_INDEX_EXCLUDES
    root_path = Path(root)
    embedded = embedded_projects(root_path)

    files: list[Path] = []
    for pattern in PROJECT_INDEX_PATTERNS:
        for file_path in root_path.glob(pattern):
            if any(excluded in str(file_path) for excluded in excludes):
                continue
            if held_out_of_corpus(file_path.relative_to(root_path), embedded):
                continue
            files.append(file_path)
    return sorted(set(files))


def index_project_files(
    kb: "GraphRAGKnowledgeBase",
    root: str = ".",
    *,
    progress: "Callable[[int, int], None] | None" = None,
    should_stop: "Callable[[], bool] | None" = None,
) -> dict[str, Any]:
    """Index every project file into the knowledge base.

    Returns a report rather than printing one, so both the CLI script and the
    console's reindex button can render it their own way.

    `progress(done, total)` is called after each file, and `should_stop` is
    asked before each file and -- through `OllamaEmbedder.encode` -- between
    batches of an embedding: a batch is ~17s for qwen3-embedding here, and a
    document's worth of them can take minutes, so a phase that long has to say
    how far it has got and has to stop when asked. A stop is `stopped` in the
    report and leaves the corpus part-built, which the next run finishes rather
    than repeats, because what is already embedded keeps its vectors.
    """
    files = iter_project_files(root)
    wanted = {str(path) for path in files}

    # A reindex rebuilds rather than accumulates. A file that no longer
    # qualifies -- renamed, deleted, or newly excluded -- has to leave the
    # corpus, or it keeps answering searches and keeps its graph node long
    # after it stops existing.
    #
    # Stored rows are chunks, so staleness is a question about the *document*
    # a row belongs to, not about the row's own id: `CLAUDE.md#0007` is not in
    # `wanted` and never will be. Comparing ids directly would delete the
    # entire corpus on every reindex and rebuild it from scratch -- which
    # ends in the same place here, but would quietly become a full re-embed of
    # every document the moment anything reindexed a subset.
    #
    # The same pass reads each surviving row's fingerprint. A document whose
    # text still hashes to what the store holds keeps the vectors it has: the
    # rebuild is then proportional to what actually changed rather than to how
    # big the corpus is, which is what makes it something a run can do for
    # itself before the Architect opens. Measured warm on this project, 77
    # files and 1,618 chunks: 52.0s re-embedding everything, 0.1s when nothing
    # changed -- and in that case the embedding model is never loaded at all.
    stored_sha: dict[str, str] = {}
    try:
        existing = kb.collection.get(include=["metadatas"])
        existing_ids = existing.get("ids") or []
        existing_metadatas = existing.get("metadatas") or []
        stale: list[str] = []
        stale_documents: set[str] = set()
        for i, chunk_id in enumerate(existing_ids):
            row = existing_metadatas[i] if i < len(existing_metadatas) else None
            document = _document_id_of(chunk_id, row)
            if document not in wanted:
                stale.append(chunk_id)
                stale_documents.add(document)
            elif row and row.get("sha"):
                stored_sha[document] = str(row["sha"])
        # Counted in documents, not rows: "3 chunks pruned" is a fact about
        # the store, and the caller wants to know how many *files* stopped
        # answering searches. It is also the only half of a rebuild's work that
        # `embedded` cannot see -- a pass that deleted a document and re-read
        # nothing did something, and must not report itself as a no-op.
        dropped = len(stale_documents)
        if stale:
            kb.collection.delete(ids=stale)
    except Exception as exc:  # pragma: no cover - Chroma unavailable
        errors_pre = [f"pruning stale documents: {exc}"]
        dropped = 0
    else:
        errors_pre = []
    kb.graph.clear()

    indexed = embedded = reused = skipped = 0
    errors: list[str] = list(errors_pre)

    stopped = False
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
                # Its chunks and their vectors are still correct and were not
                # pruned above. Only the graph has to come back, because this
                # function cleared it -- and that half is free.
                kb._add_to_graph(doc_id, content, metadata)
                reused += 1
            else:
                kb.add_document(doc_id, content, metadata)
                embedded += 1
            indexed += 1
        except EmbeddingStopped:
            stopped = True
            break
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
        if progress is not None:
            progress(position, len(files))

    # Both halves of the rebuild above are otherwise persisted only as a side
    # effect of `add_document`: the `graph.clear()` reaches disk through its
    # `_save_graph()`, and the pruned rows leave the lexical index through its
    # invalidation. So a reindex that added nothing -- no matching files, or
    # every one of them oversized or unreadable -- cleared the graph in memory
    # and left the old `knowledge_graph.json` for the next process start to
    # reload, while an index built before the prune went on answering with the
    # chunks the prune had just deleted. Unconditional, the way `clear()` ends
    # with the same two lines: the work was done either way, and a reindex is
    # the one operation after which the lexical index must be rebuilt anyway.
    kb._save_graph()
    kb._lexical_index = None

    # `indexed` is how many documents the corpus now holds, which is what every
    # caller has always printed. `embedded` and `reused` split that by cost --
    # the only number that moves when a rebuild is nearly a no-op, and the one
    # thing that says whether a rebuild did any work at all.
    kb._should_stop = None
    report: dict[str, Any] = {
        "stopped": stopped,
        "indexed": indexed,
        "embedded": embedded,
        "reused": reused,
        "dropped": dropped,
        "skipped": skipped,
        "errors": errors,
    }
    report.update(kb.stats())
    return report


# Cached knowledge base instance so repeated MCP calls and test runs do not
# reload the embedding model and Chroma store every time.
_kb_instance: GraphRAGKnowledgeBase | None = None


def get_knowledge_base(persist_dir: str | None = None) -> GraphRAGKnowledgeBase:
    """The singleton knowledge base, **built if it does not exist yet**.

    This is the door for the one act that is allowed to bring a corpus into
    existence: indexing. Everything that only wants to read -- the console's
    header, the document list, the graph sweep, the Researcher's search --
    goes through `open_knowledge_base()` instead, which returns `None` rather
    than manufacturing a store. Reading is not a reason for a corpus to exist.

    `persist_dir` is honoured only on the call that builds the singleton; the
    process holds one corpus, and the argument exists so a caller that opened
    a store somewhere other than the default is not silently handed the
    default one instead.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = GraphRAGKnowledgeBase(persist_dir)
    return _kb_instance


def embedding_device_status() -> dict[str, Any]:
    """How the embedder is placed, without touching the daemon.

    For the console header, which polls: `active` is None until something has
    embedded. Placement -- which GPU, how much of the model -- is the daemon's
    decision, so what there is to report is the share it left on the CPU the
    last time it embedded, and the note when that share would slow indexing to
    a fraction. Both are None until something has embedded. It reads the
    singleton only if something already opened the corpus, so asking never
    opens one.
    """
    kb = _kb_instance
    loaded = kb is not None and kb._embedder is not None
    embedder = kb._embedder if kb is not None else None
    return {
        "configured": "ollama",
        "active": kb.embedding_device if kb is not None and loaded else None,
        "note": kb.embedding_device_note if kb is not None else None,
        # The share of the model the daemon left on the CPU when it last
        # embedded. None until something has, or when it cannot say.
        "cpu_share": embedder.cpu_share if isinstance(embedder, OllamaEmbedder) else None,
    }


def corpus_exists(persist_dir: str | None = None) -> bool:
    """Whether a corpus has been built, without building or opening one.

    A store that was emptied by `clear()` still exists -- that is the point of
    emptying it in place -- so this answers "has anyone indexed here", not "is
    there anything in it". `corpus_state()` tells those two apart.
    """
    return (resolve_persist_dir(persist_dir) / "chroma").is_dir()


def open_knowledge_base(persist_dir: str | None = None) -> GraphRAGKnowledgeBase | None:
    """The corpus if one has been built, `None` if none has. Never builds one.

    The reason this exists rather than every caller using `get_knowledge_base`:
    constructing a `GraphRAGKnowledgeBase` creates the store on disk. The
    console polls its header every five seconds, so with the creating door
    wired to a read, starting the server was enough to leave a corpus behind --
    an empty one that then reported itself as a knowledge base. A corpus should
    be there because someone indexed, or not be there at all.
    """
    if _kb_instance is not None:
        return _kb_instance
    if not corpus_exists(persist_dir):
        return None
    return get_knowledge_base(persist_dir)


def corpus_state(persist_dir: str | None = None) -> tuple[str, str]:
    """Report the corpus as `absent`, `empty` or `indexed`, plus the model name.

    Deliberately lightweight and deliberately non-creating: it opens Chroma
    read-only and never touches the embedding model, so the console can poll
    it on a timer without either calling the daemon or bringing a store into
    being as a side effect of asking about one.

    `absent` and `empty` are kept apart because they call for different things.
    Nothing has ever been indexed here, versus a corpus that exists and was
    emptied -- and a run against either finds nothing, which is exactly why the
    operator has to be told which it was.

    Returns:
        (state, embedding_model_name)
    """
    chroma_dir = resolve_persist_dir(persist_dir) / "chroma"
    if not chroma_dir.is_dir():
        return "absent", EMBEDDING_MODEL_NAME

    try:
        client = chromadb.PersistentClient(str(chroma_dir))
        # `get_collection`, not `get_or_create_collection`: asking after the
        # corpus must not create the collection it is asking about.
        collection = client.get_collection(name="knowledge")
        return ("indexed" if collection.count() > 0 else "empty"), EMBEDDING_MODEL_NAME
    except Exception:
        # A store whose collection is missing or unreadable has nothing to
        # answer with, which is what `empty` already means to every caller.
        return "empty", EMBEDDING_MODEL_NAME


def is_knowledge_base_indexed(persist_dir: str | None = None) -> tuple[bool, str]:
    """Whether the knowledge base holds any documents.

    Returns:
        (indexed, embedding_model_name)
    """
    state, model = corpus_state(persist_dir)
    return state == "indexed", model


# Create MCP server
# At WARNING, because the SDK's constructor runs `logging.basicConfig` at the
# level it is given, and this module is imported by the console. At its default
# of INFO every library in the process logged at INFO from then on: of the 1,772
# lines the server log held on 2026-09-12, 84 were HTTP requests -- each call
# to the Ollama daemon that embeds and to the seats that answer.
server = MCPServer("graphrag", log_level="WARNING")


# Register tools with the server
@server.tool(name="search_knowledge_graph")
def search_tool(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant documents and passages.

    Searching a corpus nobody built returns no results and says why. It does
    not build one: a retrieval call is not a request for a knowledge base.
    """
    kb = open_knowledge_base()
    if kb is None:
        return json.dumps({"results": [], "source": "no_corpus", "note": NO_CORPUS_NOTE}, indent=2)
    return json.dumps(kb.search(query, top_k), indent=2)


@server.tool(name="query_knowledge_graph")
def query_tool(entity: str, hops: int = 2) -> str:
    """Query the knowledge graph for entity relationships."""
    kb = open_knowledge_base()
    if kb is None:
        return json.dumps(
            {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "no_corpus",
                "note": NO_CORPUS_NOTE,
            },
            indent=2,
        )
    return json.dumps(kb.query_graph(entity, hops), indent=2)


async def main() -> None:
    """Run the GraphRAG MCP server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())


# Re-exported from `corpus_spectral`, which is where the four whole-graph
# diagnostics now live. Named here because `serve.py`, the tests and CLAUDE.md
# all reach for them through this module, and moving the code should not have
# moved the vocabulary.
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
