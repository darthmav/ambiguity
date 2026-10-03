#!/usr/bin/env python3
"""Frontend server for the 4-Agent Console.

Usage:
    python serve.py

Then open: http://localhost:8080

The console talks to a single `POST /rpc` endpoint taking {method, params}.
`GET /api/status` answers the same payload as the `status` method, for the
launcher's readiness poll and the container's health check.
"""

import contextlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit

# Upstream libraries (langsmith) emit DeprecationWarnings on Python 3.14+
# about asyncio.iscoroutinefunction. They are harmless and outside our control,
# so suppress them before importing any third-party code.
warnings.filterwarnings(
    "ignore",
    message=".*asyncio\\.iscoroutinefunction.*",
    category=DeprecationWarning,
)

# Ensure src directory is in Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langgraph.errors import GraphRecursionError  # noqa: E402

from langgraph_agent import create_agent_graph, initial_state  # noqa: E402
from langgraph_agent.config import (  # noqa: E402
    AGENTS,
    daemon_request,
    get_agent_model_info,
    get_agent_status,
    is_local_ollama_model,
    list_ollama_models,
    ollama_base_url,
    ollama_model_capabilities,
    set_agent_llm,
    set_agent_thinking,
)
from langgraph_agent.control import ACTIVITY, EMBEDDER_ACTIVITY, RUN_CONTROL  # noqa: E402
from langgraph_agent.corpus_health import (  # noqa: E402
    corpus_staleness,
    forget_cached_walk,
)
from langgraph_agent.corpus_store import (  # noqa: E402
    POSTGRES,
    database_unreachable,
    get_database,
    rebuild_claim,
)
from langgraph_agent.dwell import finish_pull_request  # noqa: E402
from langgraph_agent.graph import RECURSION_LIMIT  # noqa: E402
from langgraph_agent.graphrag_server import (  # noqa: E402
    EMBEDDER_LOAD,
    EMBEDDING_MODEL_NAME,
    INDEXABLE_SUFFIXES,
    WEB_RESEARCH_DIR,
    GraphRAGKnowledgeBase,
    absent_corpus,
    calibrate_relevance_floor,
    corpus_signature,
    corpus_state,
    embedding_device_status,
    floor_calibration,
    floor_from_calibration,
    get_knowledge_base,
    index_corpus_files,
    iter_corpus_files,
    open_knowledge_base,
    remove_corpus_sources,
    resolve_persist_dir,
    store_uploaded_document,
)
from langgraph_agent.mcp_client import MCPClient  # noqa: E402
from langgraph_agent.projects import (  # noqa: E402
    embedded_projects,
    list_projects,
    project_dir,
    project_name_error,
    set_project_embedded,
)
from langgraph_agent.self_healing import (  # noqa: E402
    CircuitOpenError,
    circuit_states,
    get_healing_logger,
    reset_circuit,
)
from langgraph_agent.web_research import research_online, search_backend_health  # noqa: E402

# Initialize graph. The knowledge base is deliberately *not* initialized here.
graph = create_agent_graph()

# The healing journal every retry, circuit, health check and recovery in this
# process writes to; a run is one healing session in it.
HEALING = get_healing_logger()

# The corpus, once something has opened one. Nothing preloads it: a corpus
# exists because someone indexed, or it does not exist.
kb: GraphRAGKnowledgeBase | None = None

# Whether this console may rebuild its own corpus: at startup, before every
# run, and for the monitor. `tests/conftest.py` turns it off, or every run a
# test starts would index a corpus.
REBUILD_CORPUS = os.getenv(
    "REBUILD_CORPUS", "1"
).strip() not in {"0", "false", "no"}


# --------------------------------------------------------------------------
# Request and parameter checking
# --------------------------------------------------------------------------

# The largest request body read. An upload is the largest thing the console
# sends -- MAX_INDEXABLE_BYTES of text, JSON-escaped -- and this sits well
# above that, so it refuses what is not a console request without reading it.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

# Bounds for the numbers a caller may ask for. Each is past anything the
# console's own controls offer, and short of what would make a call expensive.
MAX_TOP_K = 100
MAX_GRAPH_DEPTH = 10
MAX_MIN_DEGREE = 1_000_000
MAX_LIST_LIMIT = 500
MAX_TOPICS = 100
MAX_EVENT_SEQUENCE = 2**53 - 1  # the largest integer a browser holds exactly

# Parameters are parsed by these before a method touches anything, and refused
# by name: a bare `int()` answered in Python's words, a negative `top_k` came
# back as "no results", and `bool("false")` is True.


def _refusal(name: str, wanted: str, value: Any) -> ValueError:
    return ValueError(f"{name} must be {wanted}; got {value!r}.")


def _int_param(
    params: dict[str, Any], name: str, default: int, *, low: int, high: int
) -> int:
    """A whole number from `low` to `high`, or `default` when it is absent."""
    value = params.get(name)
    if value is None or value == "":
        return default
    wanted = f"a whole number from {low} to {high}"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refusal(name, wanted, value)
    try:
        number = float(value)
    except ValueError:
        raise _refusal(name, wanted, value) from None
    if not number.is_integer() or not low <= number <= high:
        raise _refusal(name, wanted, value)
    return int(number)


def _float_param(
    params: dict[str, Any], name: str, default: float | None, *, low: float, high: float
) -> float | None:
    """A number from `low` to `high`, or `default` when it is absent."""
    value = params.get(name)
    if value is None or value == "":
        return default
    wanted = f"a number from {low:g} to {high:g}"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refusal(name, wanted, value)
    try:
        number = float(value)
    except ValueError:
        raise _refusal(name, wanted, value) from None
    if not low <= number <= high:  # NaN fails this too
        raise _refusal(name, wanted, value)
    return number


def _bool_param(params: dict[str, Any], name: str, default: bool = False) -> bool:
    """A JSON boolean, or `default` when it is absent. Nothing else is coerced."""
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _refusal(name, "true or false", value)
    return value


def _str_param(params: dict[str, Any], name: str, default: str = "") -> str:
    """A string, or `default` when it is absent."""
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _refusal(name, "a string", value)
    return value


def _open_kb() -> GraphRAGKnowledgeBase | None:
    """The corpus if one has been built, `None` if not. Never builds one.

    Every read goes through here, and the console polls `rag_stats` every few
    seconds: a read that created would leave a corpus nobody asked for.
    """
    global kb
    if kb is None:
        kb = open_knowledge_base()
    return kb


def _kb_for_indexing() -> GraphRAGKnowledgeBase:
    """The corpus, created if it does not exist -- for writers only.

    Indexing, an upload and the online research phase are each a request for a
    corpus to hold something; every read goes through `_open_kb()` instead.
    """
    global kb
    if kb is None:
        kb = get_knowledge_base()
    return kb


# --------------------------------------------------------------------------
# RPC methods
# --------------------------------------------------------------------------


def rpc_rag_stats(_: dict[str, Any]) -> dict[str, Any]:
    """Counters for the console header.

    `corpus` is what the header actually renders on: zeros alone cannot tell a
    corpus that was indexed and then emptied from one that was never built,
    and only the second means "press Reindex".
    """
    kb_or_none = _open_kb()
    if kb_or_none is None:
        state, note = absent_corpus()
        return {
            "corpus": state,
            "note": note,
            "total_documents": 0,
            "total_chunks": 0,
            "total_nodes": 0,
            "total_edges": 0,
            # Same keys as an indexed corpus, so the console renders one shape.
            # `lambda_2` is None, not 0.0: 0.0 is a real reading, meaning
            # disconnected.
            "components": 0,
            "largest_component": 0,
            "isolated_nodes": 0,
            "lambda_2": None,
        }

    stats = kb_or_none.stats()
    if stats["total_chunks"] is None:
        # The store could not be asked: unavailable, never "empty", which
        # would tell the operator a rebuild is what is missing.
        state, note = "unavailable", (
            f"The corpus database could not be asked ({stats.get('store_error')}), so "
            "there is nothing to retrieve until it answers. The console reports it as "
            "the postgres circuit; a run goes ahead without retrieval."
        )
        stats.update(corpus=state, note=note, total_chunks=0,
                     staleness={"stale": False, "unavailable": note})
        return stats
    stats["corpus"] = "indexed" if stats["total_chunks"] else "empty"

    # The corpus drifts from the archive silently: every counter above stays
    # non-zero and consistent while retrieval finds less. The comparison is
    # advisory, so a walk that fails leaves the counters alone.
    try:
        documents = [
            node
            for node, attrs in kb_or_none.graph.nodes(data=True)
            if attrs.get("type") == "document"
        ]
        report = corpus_staleness(documents)
        # Mid-run and mid-rebuild the corpus is supposed to be moving -- a page
        # written but not yet embedded, a store being repaired -- so the
        # verdict is withheld and the counts, true for this instant, stay.
        if report.get("stale") and (
            _run_progress.get("running") or _background_rebuild.get("running")
        ):
            report = {**report, "stale": False, "settling": True}
        # An emptied corpus is not a drifting one: `empty` already says so, and
        # "not indexed" would accuse the operator of the Clear they just did.
        if report.get("stale") and stats["corpus"] == "empty":
            report = {**report, "stale": False, "emptied": True}
        stats["staleness"] = report
    except Exception as exc:  # pragma: no cover - a walk that cannot run
        stats["staleness"] = {"stale": False, "unavailable": str(exc)}
    return stats


def rpc_list_documents(_: dict[str, Any]) -> dict[str, Any]:
    """Every document node; the seed set for a graph sweep."""
    kb_or_none = _open_kb()
    if kb_or_none is None:
        state, note = absent_corpus()
        return {"documents": [], "corpus": state, "note": note}
    return kb_or_none.list_documents()


def rpc_query_graph(params: dict[str, Any]) -> dict[str, Any]:
    """One node's neighbourhood, as drawable nodes and edges.

    `min_degree` defaults to 2 because the common caller is a sweep, where the
    one-document entities bury the structure. A trace passes 1 explicitly.
    """
    node_id = _str_param(params, "node_id")
    max_depth = _int_param(params, "max_depth", 2, low=1, high=MAX_GRAPH_DEPTH)
    min_degree = _int_param(params, "min_degree", 2, low=1, high=MAX_MIN_DEGREE)
    split = _bool_param(params, "split")
    kb_or_none = _open_kb()
    if kb_or_none is None:
        state, note = absent_corpus()
        return {
            "error": note,
            "corpus": state,
            "center_node": node_id,
            "related_nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
        }

    return kb_or_none.neighborhood(
        node_id,
        max_depth=max_depth,
        min_degree=min_degree,
        # Off unless asked for: it triples the cost of a call the console makes
        # on every click, and only a caller drawing the division wants it.
        split=split,
    )


def rpc_graph_overview(params: dict[str, Any]) -> dict[str, Any]:
    """The whole corpus as one drawable graph, for the console's sweep."""
    min_degree = _int_param(params, "min_degree", 4, low=1, high=MAX_MIN_DEGREE)
    include_isolated = _bool_param(params, "include_isolated")
    kb_or_none = _open_kb()
    if kb_or_none is None:
        state, note = absent_corpus()
        return {
            "corpus": state,
            "note": note,
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
            "unlinked_documents": 0,
        }
    return kb_or_none.overview(
        min_degree=min_degree,
        include_isolated=include_isolated,
    )


def rpc_search_documents(params: dict[str, Any]) -> dict[str, Any]:
    """Semantic search over the corpus.

    With no corpus this returns no results and says so, rather than building
    one to search. Searching is a read like any other.
    """
    query = _str_param(params, "query")
    top_k = _int_param(params, "top_k", 5, low=1, high=MAX_TOP_K)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {"results": [], "source": "no_corpus", "note": absent_corpus()[1]}

    results = kb_or_none.search(query, top_k)
    return {"results": results, "source": "local_graphrag"}


def _why_the_corpus_is_busy(action: str) -> str | None:
    """Why the corpus cannot be `action` now, or None. The caller holds `_run_lock`."""
    running = bool(_run_progress["running"])
    goal = str(_run_progress["goal"])
    # A run takes precedence in the wording: it is the one of the two the
    # operator can end, and a run started into a rebuild sets both flags.
    if running:
        detail = f" Running: {goal}" if goal else ""
        return (
            f"A run is in flight and the Researcher is searching this corpus, so "
            f"it cannot be {action} right now. Stop the run first.{detail}"
        )
    if _background_rebuild["running"]:
        return (
            f"The corpus is being rebuilt to match the archive, so it cannot be "
            f"{action} right now. The header says how far it has got; try again "
            f"when it stops."
        )
    if _corpus_change["running"]:
        return (
            f"The corpus is being {_corpus_change['action']} from another request, so "
            f"it cannot be {action} right now; try again in a moment."
        )
    return None


def _refuse_while_a_run_is_in_flight(action: str) -> None:
    """Refuse to change the corpus underneath a run or a rebuild.

    Changing it mid-run breaks nothing visibly, which is the problem: a cleared
    or half-rebuilt corpus reads to the Researcher as one with nothing to say,
    and the run plans around an absence made out from under it. A background
    rebuild is the same hazard without the run, worded apart because the
    operator stops a run but waits out a rebuild. A change that goes on to
    happen holds `_changing_the_corpus` instead, which checks and claims in one
    hold of the lock.

    Args:
        action: Past participle of what was refused -- "cleared", "added to".

    Raises:
        ValueError: naming the goal in flight, or the rebuild.
    """
    with _run_lock:
        why = _why_the_corpus_is_busy(action)
    if why is not None:
        raise ValueError(why)


@contextlib.contextmanager
def _changing_the_corpus(action: str) -> Iterator[None]:
    """Change the corpus with no run or rebuild able to start until it is done.

    The refusal used to be checked and then let go of: a run claimed in the gap
    rebuilt the corpus under an upload, and the stored graph and the one in
    memory parted. Checked and claimed in one hold of `_run_lock`; a run asked
    for meanwhile is refused, and a background rebuild leaves it to the next.

    Raises:
        ValueError: as `_refuse_while_a_run_is_in_flight` does.
    """
    with _run_lock:
        why = _why_the_corpus_is_busy(action)
        if why is not None:
            raise ValueError(why)
        _corpus_change.update(running=True, action=action)
    try:
        yield
    finally:
        with _run_lock:
            _corpus_change.update(running=False, action="")


def rpc_list_projects(_: dict[str, Any]) -> dict[str, Any]:
    """The generated projects under `projects/`, for the console's dropdown and list."""
    return {"projects": list_projects()}


def rpc_embed_project(params: dict[str, Any]) -> dict[str, Any]:
    """Opt a generated project into the corpus, or take it back out.

    Records the choice, then starts the background rebuild, which embeds or
    prunes only what changed. Refused mid-run and mid-rebuild, like every corpus
    writer (`_refuse_while_a_run_is_in_flight`).
    """
    name = _str_param(params, "name")
    embed = _bool_param(params, "embed", default=True)
    error = project_name_error(name)
    if error:
        raise ValueError(error)
    if not Path(project_dir(name)).is_dir():
        raise ValueError(f"There is no project {name!r} under projects/.")
    with _changing_the_corpus("changed"):
        set_project_embedded(name, embed)
        forget_cached_walk()
    rebuilding = REBUILD_CORPUS
    if rebuilding:
        threading.Thread(
            target=_rebuild_the_corpus_in_background,
            args=(f"after {'embedding' if embed else 'removing'} {project_dir(name)}",),
            name="embed-project",
            daemon=True,
        ).start()
    return {
        "name": name,
        "embedded": embed,
        "rebuilding": rebuilding,
        "note": "" if rebuilding else (
            "REBUILD_CORPUS is off, so nothing will rebuild the "
            "corpus; the choice is recorded for when it is back on."
        ),
    }


# There is no `rpc_reindex`: the corpus is rebuilt when the console starts,
# before every run, when a project is opted in or out, and by the monitor after
# an outage, so keeping it current is never the operator's job.


def rpc_upload_document(params: dict[str, Any]) -> dict[str, Any]:
    """Embed one document the operator uploaded, from either upload control.

    The file is written under `uploads/` before it is embedded, which puts it
    inside the next rebuild rather than underneath it (`store_uploaded_document`).
    Refused mid-run: an upload only adds, but `add_document` mutates the graph a
    search is reading. One document per call, so a file the corpus cannot take
    fails alone instead of taking a batch down with it.
    """
    name = params.get("name", "")
    content = params.get("content", "")
    if not isinstance(name, str) or not isinstance(content, str):
        raise ValueError("An upload is a filename and its text; both must be strings.")
    with _changing_the_corpus("added to"):
        report = store_uploaded_document(_kb_for_indexing(), name, content)
        # The upload is a new file, so the cached walk is behind the corpus
        # now; forgotten, or the new document would read as `extra`.
        forget_cached_walk()
    return report


def rpc_bottleneck(params: dict[str, Any]) -> dict[str, Any]:
    """The narrowest cut in the knowledge graph, and the nodes bridging it.

    Read-only and off the poll: an eigenvector plus a sweep over every edge, for
    a question the operator asks rather than one the header needs.
    """
    limit = _int_param(params, "limit", 12, low=1, high=MAX_LIST_LIMIT)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to analyse. {absent_corpus()[1]}")
    return kb_or_none.bottleneck(limit=limit)


def rpc_topics(params: dict[str, Any]) -> dict[str, Any]:
    """Topic communities over the corpus: the whole-corpus map.

    Read-only and off the poll, for the same reasons as `bottleneck`. `k` is
    optional; omitted, the eigengap chooses it and the answer may come back
    `no_clear_structure` rather than a map of a corpus that has no topics.
    """
    # Omitted or blank means "choose one", not k = 0.
    k = None if params.get("k") in (None, "") else _int_param(
        params, "k", 0, low=2, high=MAX_TOPICS
    )
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to cluster. {absent_corpus()[1]}")
    return kb_or_none.topics(k=k)


def rpc_duplicate_entities(params: dict[str, Any]) -> dict[str, Any]:
    """Entities playing the same structural role: candidates to merge.

    Read-only and off the poll. It proposes and never merges: merging is a
    decision about meaning that the graph alone cannot make.
    """
    limit = _int_param(params, "limit", 20, low=1, high=MAX_LIST_LIMIT)
    name_similarity = _float_param(params, "name_similarity", None, low=0.0, high=1.0)
    containment = _float_param(params, "containment", None, low=0.0, high=1.0)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to scan. {absent_corpus()[1]}")
    return kb_or_none.duplicate_entities(
        limit=limit, name_similarity=name_similarity, containment=containment
    )


def rpc_export_corpus(_: dict[str, Any]) -> dict[str, Any]:
    """The whole corpus as one JSON document, for the console to save.

    Served over `/rpc` like everything else rather than as a file download, so
    a failure lands on the console's telemetry path instead of replacing the
    page with a JSON error. The browser makes the file at the other end.
    """
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to export. {absent_corpus()[1]}")
    return kb_or_none.export_corpus()


def rpc_clear_corpus(_: dict[str, Any]) -> dict[str, Any]:
    """Empty the knowledge base, and delete what it would be rebuilt from.

    Clearing the store alone did not last: the corpus is rebuilt from the walk
    when the console starts, so a restart embedded the same pages again. The
    sources go first (`remove_corpus_sources`), so a store that fails to clear
    is pruned by the next rebuild rather than refilled.

    Refused mid-run for the reason `_refuse_while_a_run_is_in_flight` sets out.
    With no corpus the sources are still deleted, but no store is created in
    order to empty it: that would leave behind exactly the thing the operator
    was asking to be rid of. Refused only when there is nothing either way.
    """
    with _changing_the_corpus("cleared"):
        kb_or_none = _open_kb()
        if kb_or_none is None and not iter_corpus_files():
            raise ValueError(f"There is no corpus to clear. {absent_corpus()[1]}")
        sources = remove_corpus_sources()
        forget_cached_walk()
        if kb_or_none is None:
            return {"removed_chunks": 0, "removed_nodes": 0, "removed_edges": 0,
                    "removed_floor": False, **sources}
        return {**kb_or_none.clear(), **sources}


def rpc_list_seats(_: dict[str, Any]) -> dict[str, Any]:
    """The four seats and whether each can actually run."""
    return {
        "seats": [{"role": agent, **get_agent_status(agent)} for agent in AGENTS]
    }


def rpc_set_seat(params: dict[str, Any]) -> dict[str, Any]:
    """Reassign one seat for the lifetime of this process."""
    agent = _str_param(params, "agent") or _str_param(params, "role")
    provider = _str_param(params, "provider")
    model = _str_param(params, "model")

    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if not provider or not model:
        raise ValueError("Provider and model are both required")

    # Only what the dropdowns offer, so a stale tab cannot seat a model the
    # daemon no longer has -- plus the seat's own current model, which the
    # console lists under "Current" when the offer lacks it, or a seat
    # configured from `.env` could never be selected back.
    allowed = {(o["provider"], o["model"]) for o in _seat_model_options()}

    # Config, not liveness: `get_agent_status` would ask the daemon for fields
    # nothing here reads.
    seated = get_agent_model_info(agent)
    allowed.add((seated["provider"], seated["model"]))

    if (provider, model) not in allowed:
        offered = ", ".join(o["model"] for o in _seat_model_options()) or "none: the daemon reported no models"
        raise ValueError(f"{model!r} is not a seat model the console offers: {offered}")

    set_agent_llm(agent, provider, model)
    return {"ok": True, "role": agent, **get_agent_status(agent)}


def rpc_set_thinking(params: dict[str, Any]) -> dict[str, Any]:
    """Switch one seat's thinking on or off for the lifetime of this process.

    `thinking` must be a JSON boolean. Anything else is refused rather than
    coerced, because the obvious coercion reads the string "false" as on.
    """
    agent = _str_param(params, "agent") or _str_param(params, "role")
    thinking = params.get("thinking")

    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be true or false")

    set_agent_thinking(agent, thinking)
    return {"ok": True, "role": agent, **get_agent_status(agent)}


def _seat_model_options() -> list[dict[str, str]]:
    """Every tag `ollama ls` reports that a seat can run, and nothing else.

    A tag whose capabilities lack `completion` (the embedder) is left out; one
    the daemon would not describe stays in -- "could not ask" is not "cannot
    run". Cloud tags are grouped apart from weights on this machine.
    """
    options = []
    for tag in list_ollama_models():
        caps = ollama_model_capabilities(tag)
        if caps is not None and "completion" not in caps:
            continue
        cloud = not is_local_ollama_model(tag)
        options.append({
            "label": tag,
            "provider": "ollama",
            "model": tag,
            "group": "Ollama Cloud" if cloud else "Ollama (local)",
        })
    # Local weights first, then cloud, each alphabetical (the daemon's list is).
    options.sort(key=lambda o: o["group"] != "Ollama (local)")
    return options


def rpc_llm_options(_: dict[str, Any]) -> dict[str, Any]:
    """Model choices for the seat dropdowns: `ollama ls`, minus embedders."""
    return {"options": _seat_model_options()}


def _embedding_choice() -> dict[str, Any]:
    """The one embedding model entry -- qwen3-embedding, served by the daemon."""
    state, _ = corpus_state()
    record = floor_calibration()
    floor = floor_from_calibration(record)
    return {
        "model": EMBEDDING_MODEL_NAME,
        "corpus": state,
        "floor": floor,
        # A corpus has no floor until a run measures one, and none for good
        # when the measurement found no gap; the console words those apart.
        "floor_source": "measured" if floor is not None else ("no_gap" if record else "not_measured"),
    }


def rpc_embedding_options(_: dict[str, Any]) -> dict[str, Any]:
    """The embedding model -- the only one, served by the Ollama daemon.

    The entry carries the state of the corpus and the relevance floor measured
    against it, `None` until the first run on a whole corpus has measured one.
    The card reports rather than offers: a corpus is only ever built and
    searched by the model it was built with.
    """
    return {
        "active": EMBEDDING_MODEL_NAME,
        "options": [_embedding_choice()],
        "device": embedding_device_status(),
    }


def rpc_status(_: dict[str, Any]) -> dict[str, Any]:
    """What the console polls for: the embedding model and the corpus.

    `corpus` is `absent` (nothing was ever indexed), `empty` or `indexed`. The
    seats are not here: the console draws them from `list_seats` alone.
    """
    try:
        state, embedding_model = corpus_state()
    except Exception:
        state, embedding_model = "absent", "unknown"

    return {
        # The embedding model, which is not loaded until something indexes or
        # searches.
        "embedding": embedding_model,
        # Where it runs, which only a loaded model can say; `active` is None
        # until something has embedded.
        "embedding_device": embedding_device_status(),
        "corpus": state,
        # Whether a run rebuilds the corpus first. Reported, not assumed:
        # `REBUILD_CORPUS=0` makes empty retrieval the expected answer.
        "indexes_on_run": REBUILD_CORPUS,
        # Whether a rebuild is in flight, and how far it has got: meanwhile the
        # corpus may read `absent` and the stale verdict is withheld, and the
        # header says why.
        "indexing": _background_rebuild_status(),
        # Whether a run is in flight, from any tab: the console keeps the Graph
        # tab blank while the embedder may load, since drawing a large SVG
        # takes GPU memory the load needs.
        "run_in_flight": _run_in_flight(),
        # What an upload may be, for the console's pickers and tooltips. From
        # the walk's own list, so the page cannot promise a different one.
        "indexable_suffixes": list(INDEXABLE_SUFFIXES),
        # Pull requests a run left waiting on their checks, which the monitor
        # merges once they pass, and the last word on each.
        "pull_requests": pull_requests_snapshot(),
    }


# What the in-flight run has done so far. The POST that started it returns only
# when the run ends, so without this a working run and a wedged one look alike.
_run_progress: dict[str, Any] = {
    "running": False,
    "goal": "",
    "messages": [],
    "node": "",
    # Identifies the run a Stop is aimed at. Without it a Stop click held over
    # from a finished run -- a stale tab, a reload -- would kill whatever
    # happened to be running when it finally landed.
    "run_id": "",
    "stopping": False,
}
_run_lock = threading.Lock()

# The last run's final state, so a stopped run is recoverable; also on disk, so
# it survives a reload and a restart.
_last_run_snapshot: dict[str, Any] | None = None
RUNS_DIR = Path(__file__).parent / "runs"
LAST_RUN_PATH = RUNS_DIR / "last_run.json"


def _save_snapshot(snapshot: dict[str, Any]) -> None:
    """Keep the run's outcome in memory and on disk.

    Written to a temp file and renamed, because the console fetches this on
    load: a reader arriving mid-write would otherwise get half a JSON document.
    A disk failure is swallowed -- the in-memory copy is what the current page
    reads, and losing the file is not worth failing a run that already finished.
    """
    global _last_run_snapshot
    _last_run_snapshot = snapshot
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=RUNS_DIR, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(snapshot, handle, default=str, indent=2)
            temp_name = handle.name
        os.replace(temp_name, LAST_RUN_PATH)
    except OSError:
        pass


def _load_snapshot() -> dict[str, Any] | None:
    """The last run's outcome, from memory or from the file a restart left."""
    global _last_run_snapshot
    if _last_run_snapshot is not None:
        return _last_run_snapshot
    try:
        with open(LAST_RUN_PATH, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict):
        _last_run_snapshot = loaded
        return _last_run_snapshot
    return None


# Wall-clock budget for one run, checked between supersteps. MAX_STEPS bounds
# how often the Architect sends work back, not how long that takes.
RUN_BUDGET_SECONDS = float(os.getenv("RUN_BUDGET_SECONDS", "300"))


def rpc_run_progress(_: dict[str, Any]) -> dict[str, Any]:
    """A snapshot of the run currently in flight, for the console to poll."""
    with _run_lock:
        # `ACTIVITY` is read under `_run_lock` because its turn record is reset
        # as a run is claimed under this lock: read beside it, a reply could
        # pair the new run's `running` with the last run's turns.
        return {
            "running": bool(_run_progress["running"]),
            "goal": str(_run_progress["goal"]),
            "node": str(_run_progress["node"]),
            # `node` is the seat that last finished (`graph.stream` yields on
            # completion); `active` is the seat working now.
            "active": ACTIVITY.current(),
            "active_for": round(ACTIVITY.busy_for(), 1),
            # What the console's seat lights are driven from. `active` is only
            # a sample, and a turn that falls between two polls is never in
            # one; the record keeps it, finished, for the console to show.
            "turns": ACTIVITY.turns(),
            # The embedder card's light, which reads a meter of its own: the
            # corpus phase and the web phase embed before any seat has a turn,
            # and a search embeds inside one.
            "embedding": EMBEDDER_ACTIVITY.snapshot(),
            "messages": list(_run_progress["messages"]),
            "budget": RUN_BUDGET_SECONDS,
            # The console needs both to reattach after a reload: the id to aim
            # a Stop at this run, and `stopping` to keep the button honest
            # about a stop that has been asked for but not yet landed.
            "run_id": str(_run_progress["run_id"]),
            "stopping": bool(_run_progress["stopping"]),
        }


def rpc_embedding_activity(_: dict[str, Any]) -> dict[str, Any]:
    """Whether the embedding model is working, for the embedder card's light.

    A run's poll already carries this as `run_progress["embedding"]`. This is
    for the console to watch between runs, while a search or an upload of its
    own is in flight -- the two things outside a run that make the embedder
    work -- without polling for a run that is not there.
    """
    return EMBEDDER_ACTIVITY.snapshot()


def rpc_stop_run(params: dict[str, Any]) -> dict[str, Any]:
    """Ask the run in flight to stop at its next safe boundary.

    Cooperative, and returns at once: work already started -- a write, a
    commit, a test run, a model call -- finishes first, so the run ends within
    one call, not instantly. Served on its own thread, since `rpc_run_goal`
    holds its request thread for the whole run.
    """
    run_id = _str_param(params, "run_id")
    reason = _str_param(params, "reason")

    if not RUN_CONTROL.stop(run_id, reason):
        armed = RUN_CONTROL.run_id()
        detail = (
            "No run is in flight."
            if not armed
            else "That Stop was for a run that has already ended."
        )
        return {"stopping": False, "run_id": armed, "detail": detail}

    with _run_lock:
        # Only while still running: the teardown the stop triggers clears this
        # flag under the same lock, and setting it blind would pin "STOPPING"
        # on an ended run.
        if _run_progress["running"]:
            _run_progress["stopping"] = True

    return {
        "stopping": True,
        "run_id": RUN_CONTROL.run_id(),
        "detail": (
            "Stopping. Nothing further will be started; the call already in "
            "flight finishes first, so this can take up to a minute."
        ),
    }


def rpc_last_run(_: dict[str, Any]) -> dict[str, Any]:
    """The last run's final state, for a console that reloaded and lost it."""
    return {"snapshot": _load_snapshot()}


# Set when the console's exit button asks the process to end; the main thread
# waits on it, exactly as it waits on Ctrl+C.
_shutdown_requested = threading.Event()

# Set when an exit has to wait for the run in flight to hand its state back
# first. The run's own `finally` is what then requests the shutdown, which is
# what guarantees the snapshot is on disk before the process goes.
_exit_after_run = threading.Event()

# A moment's grace between deciding to exit and closing the socket, so the
# reply that asked for the exit reaches the browser rather than dying with the
# connection. Small enough that the console never feels it.
SHUTDOWN_GRACE_SECONDS = float(os.getenv("SHUTDOWN_GRACE_SECONDS", "0.5"))


def _finish_run() -> None:
    """Release the run and let a deferred exit through. The run's `finally`.

    The `_exit_after_run` read happens under `_run_lock`, and `rpc_shutdown`
    decides under that same lock, because the two are one handshake: whoever
    holds the lock either sees a run still in flight and hands it the exit, or
    sees it finished and takes the exit itself. Exactly one of those happens.
    """
    RUN_CONTROL.disarm()
    HEALING.end_healing_session()
    # A run that ended has no seat working, whatever a node's own `finally`
    # did.
    ACTIVITY.clear()
    with _run_lock:
        _run_progress["running"] = False
        _run_progress["stopping"] = False
        _run_progress["run_id"] = ""
        # An exit that was waiting on this run happens now, after the snapshot
        # the caller has already written -- which is the whole reason it waited.
        if _exit_after_run.is_set():
            _shutdown_requested.set()


def rpc_shutdown(params: dict[str, Any]) -> dict[str, Any]:
    """Exit the server -- what the console's X asks for.

    A run in flight is never killed by surprise: without `stop_first` this
    refuses and names it; with it, the run is stopped and the exit waits for the
    run's own `finally`, so its snapshot is on disk before the process ends.
    """
    stop_first = _bool_param(params, "stop_first")

    # Read the run's state and claim the exit in one hold of `_run_lock`,
    # before the stop below: a run at a superstep boundary reaches its
    # `finally` in microseconds, and one that found the flag unset would never
    # request the exit.
    with _run_lock:
        running = bool(_run_progress["running"])
        goal = str(_run_progress["goal"])
        if running and stop_first:
            _exit_after_run.set()
            _run_progress["stopping"] = True

    if running and not stop_first:
        return {
            "exiting": False,
            "running": True,
            "goal": goal,
            "detail": (
                "A run is still in flight. Stop it first, or ask again with "
                "stop_first to stop it and exit once its state is saved."
            ),
        }

    if running:
        # If the run finished since the lock above, it saw the flag and
        # requested the exit itself; this stop then lands on a disarmed control
        # and is refused, correctly.
        RUN_CONTROL.stop("", "Stopped so the console could exit.")
        return {
            "exiting": True,
            "running": True,
            "detail": (
                "Stopping the run. The server exits once it has handed its "
                "state back and written its snapshot."
            ),
        }

    _shutdown_requested.set()
    return {"exiting": True, "running": False, "detail": "The server is exiting."}


def _calibrate_the_floor_before_the_run(corpus_report: dict[str, Any]) -> dict[str, Any]:
    """Measure the relevance floor whenever the corpus is whole and has changed.

    A cosine means nothing absolute and nothing across models, so the floor is
    measured, never set: questions the corpus answers -- drawn from its own
    documents -- against questions it cannot (`embedding_calibration.json`),
    and the floor sits in the gap. A corpus that leaves no gap gets no floor.
    A record taken on the texts the corpus holds now is `known`; one taken on
    other texts, or before records named their corpus, is measured again.
    Skipped unless the corpus phase left a whole corpus.
    """
    record = floor_calibration()
    kb_or_none = _open_kb()
    if record is not None and kb_or_none is not None:
        try:
            if record.get("corpus") == corpus_signature(kb_or_none):
                return {"source": "known", "model": EMBEDDING_MODEL_NAME}
        except Exception:
            pass  # the store could not say what it holds; the checks below decide
    if RUN_CONTROL.stopped() or corpus_report.get("stopped"):
        return {"source": "stopped", "model": EMBEDDING_MODEL_NAME}
    if corpus_report.get("source") in ("error", "nothing_to_index", "unavailable"):
        # The floor a record carries is still the best there is to read off.
        source = "known" if record is not None else "no_corpus"
        return {"source": source, "model": EMBEDDING_MODEL_NAME}
    if kb_or_none is None or corpus_state()[0] != "indexed":
        return {"source": "no_corpus", "model": EMBEDDING_MODEL_NAME}
    with _run_lock:
        _run_progress["messages"] = [
            f"[Embedder] Measuring a relevance floor for {EMBEDDING_MODEL_NAME} on its corpus "
            "before the run starts."
        ]
    try:
        record = calibrate_relevance_floor(kb_or_none)
    except Exception as exc:
        return {"source": "error", "model": EMBEDDING_MODEL_NAME, "note": str(exc)}
    return {"source": "calibrated", **record}


def _calibration_feed_line(report: dict[str, Any]) -> str | None:
    """Said when a floor was measured or could not be: the outcomes that change retrieval."""
    source = report.get("source")
    model = report.get("model")
    if source == "calibrated" and report.get("too_small") is not None:
        return (
            f"[Embedder] The corpus offers {report['too_small']} question(s) it answers by "
            f"construction, too few to measure a relevance floor for {model} on: the "
            "Researcher's model answers every search, and the floor is measured again "
            "once the corpus has changed."
        )
    if source == "calibrated":
        answered = report.get("answered") or [0.0]
        unanswerable = report.get("unanswerable") or [0.0]
        span = (
            f"questions the corpus answers scored from {min(answered):.3f}, questions "
            f"it cannot up to {max(unanswerable):.3f}"
        )
        if report.get("floor") is None:
            return (
                f"[Embedder] {model} left no gap between the two populations ({span}), "
                "so it has no relevance floor: the Researcher's model answers every "
                "search and the Planner gets no corpus map until the corpus changes "
                "and the floor is measured again."
            )
        return (
            f"[Embedder] Measured a relevance floor for {model}: {span}, so retrieval "
            f"over {report['floor']:.3f} counts as an answer."
        )
    if source == "error":
        return (
            f"[Embedder] The relevance floor for {model} could not be measured: "
            f"{report.get('note')}. Until it is, the Researcher's model answers every "
            "search."
        )
    return None


# One corpus, so one rebuild at a time: `index_corpus_files` prunes and clears
# before it re-adds, so two interleaved rebuilds both report success on a half-
# built corpus.
#
# Lock order: this one first, `_run_lock` briefly inside it, never the reverse.
_index_lock = threading.Lock()

# How long a run waits out a rebuild another process is running before going
# ahead against the corpus as it stands: longer than a cold build. The
# background rebuild waits 0, since the holder is rebuilding the same corpus
# from the same walk.
CORPUS_LOCK_WAIT_SECONDS = 180.0

# How often the wait above re-tries, which is also how quickly it notices a stop.
CORPUS_LOCK_POLL_SECONDS = 0.5

@contextlib.contextmanager
def _claim_the_rebuild(
    wait_seconds: float, should_stop: Callable[[], bool], waiting: Callable[[], None]
) -> "Iterator[bool]":
    """Hold the right to rebuild this corpus, across threads and processes.

    `_index_lock` serializes the phases inside one process; two consoles on one
    corpus share the database but not that lock, so the claim is also an
    advisory lock in the database, keyed by the corpus's schema
    (`rebuild_claim`). The server releases it when the process dies, so a stale
    claim cannot exist, and it is polled, so "someone else is rebuilding" is an
    answer rather than an unbounded wait.

    Yields True when the claim is held, False when another process holds it
    past `wait_seconds`. Where no claim can be taken at all -- the database
    cannot be reached -- it yields True on `_index_lock` alone, and the rebuild
    meets the same database and reports it.
    """
    with _index_lock:
        try:
            claim = rebuild_claim(
                resolve_persist_dir(), wait_seconds, should_stop, waiting,
                poll_seconds=CORPUS_LOCK_POLL_SECONDS,
            )
            granted = claim.__enter__()
        except Exception:
            claim, granted = None, True
        try:
            yield granted
        finally:
            if claim is not None:
                claim.__exit__(None, None, None)


# What the background rebuild is doing, for the console header. Not part of
# `_run_progress`: a rebuild is not a run, and the stop button, the recovery
# block and the snapshot all key off that dict. Guarded by `_run_lock`.
_background_rebuild: dict[str, Any] = {"running": False, "message": "", "report": {}}

# An upload, a clear or an opt-in under way (`_changing_the_corpus`).
_corpus_change: dict[str, Any] = {"running": False, "action": ""}


def _rebuild_the_corpus(
    *,
    announce: Callable[[int], None],
    progress: Callable[[int, int], None],
    should_stop: Callable[[], bool],
    wait_seconds: float = 0.0,
    waiting: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Make the corpus match the archive, once, serialized against every other rebuild.

    Shared by both phases so what a rebuild did is classified in one place,
    which `_corpus_feed_line` renders. `announce(total)` is called once the walk
    has found something and before any work. Not at all when there is nothing:
    the walk is counted before the creating door opens, so a machine with
    nothing to index keeps reporting `absent`, and the claim is taken after it,
    so such a machine is left without even a lock file.

    A rebuild another process holds is `busy_elsewhere` -- neither a failure nor
    a no-op. Never raises: a corpus that could not be built makes for a worse
    run, not a refused one, and the report says which.
    """
    state, _ = corpus_state()
    files = iter_corpus_files()
    # An empty walk over an existing store is still work: whatever the store
    # holds is no longer in the archive and has to be pruned.
    if not files and state == "absent":
        return {"source": "nothing_to_index", "corpus": state, "root": str(Path.cwd())}

    announce(len(files))
    started = time.monotonic()
    try:
        with _claim_the_rebuild(wait_seconds, should_stop, waiting) as claimed:
            if not claimed:
                return {
                    "source": "busy_elsewhere",
                    "corpus": state,
                    "waited_s": round(time.monotonic() - started, 1),
                }
            report = index_corpus_files(
                _kb_for_indexing(), progress=progress, should_stop=should_stop
            )
    except Exception as exc:
        if isinstance(exc, CircuitOpenError) or database_unreachable(exc):
            # The claim or the store met a database that is not answering: an
            # outage like one met midway, which the monitor redoes the rebuild
            # after, so it is recorded the same way.
            failed: dict[str, Any] = {
                "source": "unavailable", "corpus": state, "stopped": False,
                "unavailable": str(exc),
                "unavailable_circuit": getattr(exc, "circuit", POSTGRES.name),
                "indexed": 0, "embedded": 0, "reused": 0, "dropped": 0, "skipped": 0,
                "errors": [],
            }
        else:
            failed = {"source": "error", "corpus": state, "note": str(exc)}
        with _run_lock:
            _last_rebuild.clear()
            _last_rebuild.update(failed)
        return failed

    if report.get("stopped"):
        source = "stopped_midway"
    elif report.get("unavailable"):
        source = "unavailable"
    elif state != "indexed":
        source = "built"
    elif report.get("embedded") or report.get("dropped"):
        source = "updated"
    else:
        source = "current"
    report.update(
        source=source,
        corpus=state,
        model=EMBEDDING_MODEL_NAME,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    with _run_lock:
        _last_rebuild.clear()
        _last_rebuild.update(report)
    return report


def _rebuild_the_corpus_in_background(when: str = "at startup") -> dict[str, Any] | None:
    """Bring the corpus up to date without a run: at startup, and on request.

    `main()` starts it once the server is serving (never at import, which every
    test does, and never ahead of readiness, which `launch_console.sh` polls);
    `rpc_embed_project` and the self-healing monitor start it with their own
    `when`. A run wins and does not wait: `should_stop` is "a run has claimed
    the flag, or the console is exiting", and the run's own phase finishes the
    job, reusing every vector already embedded. A second console finds the
    claim taken (`_claim_the_rebuild`) and leaves the corpus to it.

    Returns the rebuild's report, or None when `REBUILD_CORPUS` is off or a
    rebuild is already under way.
    """
    if not REBUILD_CORPUS:
        return None

    # Claimed before the walk rather than with the file count, so the flag
    # covers the whole phase: `_refuse_while_a_run_is_in_flight` reads it, and a
    # rebuild under way that says it is not is the window an upload lands in.
    # Checked in the same hold, so a second caller finds one under way and
    # leaves it to finish the same walk.
    with _run_lock:
        # A change under way finishes first; the next rebuild -- before the next
        # run, or the monitor's -- catches the corpus up.
        if _background_rebuild["running"] or _corpus_change["running"]:
            return None
        _background_rebuild["running"] = True
        _background_rebuild["message"] = "checking the archive against the corpus"

    def announce(total: int) -> None:
        with _run_lock:
            _background_rebuild["message"] = (
                f"checking {total} archive file(s) against the corpus"
            )
        print(
            f"[Corpus] Checking {total} archive file(s) against the "
            f"{EMBEDDING_MODEL_NAME} corpus."
        )

    def progress(done: int, total: int) -> None:
        # Rewritten in place, so a slow build reads as moving rather than stuck.
        with _run_lock:
            _background_rebuild["message"] = f"indexing: {done} of {total} file(s) checked"

    def should_stop() -> bool:
        if _shutdown_requested.is_set():
            return True
        with _run_lock:
            return bool(_run_progress["running"])

    def waiting() -> None:
        with _run_lock:
            _background_rebuild["message"] = "another process is rebuilding this corpus"

    try:
        report = _rebuild_the_corpus(
            announce=announce,
            progress=progress,
            should_stop=should_stop,
            # Not a wait: whoever holds the claim is rebuilding the same corpus
            # from the same walk, so waiting for them buys a second copy of a
            # result that is already arriving.
            wait_seconds=0.0,
            waiting=waiting,
        )
    finally:
        # In a `finally`: a phase that raised must not leave `running` set, or
        # every later upload and clear is refused by a rebuild that is not
        # happening.
        with _run_lock:
            _background_rebuild["running"] = False
            _background_rebuild["message"] = ""
    with _run_lock:
        _background_rebuild["report"] = report

    line = _corpus_feed_line(report, when=when)
    if line is not None:
        print(line)
    return report


def _run_in_flight() -> bool:
    with _run_lock:
        return bool(_run_progress["running"])


def _background_rebuild_status() -> dict[str, Any]:
    """What the header needs: whether a rebuild is in flight, and its last word."""
    with _run_lock:
        return {
            "running": bool(_background_rebuild["running"]),
            "message": str(_background_rebuild["message"]),
            "source": str(_background_rebuild["report"].get("source", "")),
        }


def _rebuild_the_corpus_before_the_run() -> dict[str, Any]:
    """Make the corpus match the archive, before any seat searches it.

    Runs on every run, however recent the startup rebuild: that one may have
    been stopped, lost a race with a file written seconds ago, or be as old as
    the console. Rebuilding is cheap when nothing changed, since
    `index_corpus_files` keeps every vector whose text still hashes the same,
    and it compares content, so it sees an edit the staleness report cannot.

    It runs before the online research phase, which embeds its pages on top, and
    not at all once the run is stopped: `index_corpus_files` clears the graph up
    front, so an abandoned rebuild is a half-built corpus. A discussion run still
    does it -- it embeds files already on disk, and changes nothing else.

    Never raises; the report says what happened.
    """
    if not REBUILD_CORPUS:
        return {"source": "disabled", "corpus": corpus_state()[0]}
    if RUN_CONTROL.stopped():
        return {"source": "stopped", "corpus": corpus_state()[0]}

    def announce(total: int) -> None:
        # Said before the work: a first build takes tens of seconds, and a run
        # with no node on the stack and no messages looks wedged. It also
        # covers the wait for a background rebuild to notice the run.
        with _run_lock:
            _run_progress["messages"] = [
                f"[Corpus] Checking {total} archive file(s) against the {EMBEDDING_MODEL_NAME} "
                "corpus before the run starts."
            ]

    def progress(done: int, total: int) -> None:
        # Rewritten in place as the phase goes, so a slow build reads as moving
        # rather than wedged.
        with _run_lock:
            _run_progress["messages"] = [
                f"[Corpus] Indexing with {EMBEDDING_MODEL_NAME}: {done} of {total} archive file(s) "
                "checked."
            ]

    def waiting() -> None:
        with _run_lock:
            _run_progress["messages"] = [
                "[Corpus] Another process is rebuilding this corpus, so the run is "
                f"waiting up to {int(CORPUS_LOCK_WAIT_SECONDS)}s for it to finish "
                "rather than searching a corpus midway through a rebuild."
            ]

    return _rebuild_the_corpus(
        announce=announce,
        progress=progress,
        should_stop=RUN_CONTROL.stopped,
        # The run waits, where the background rebuild does not: a fraction of a
        # corpus reads to the Researcher as one with nothing to say.
        wait_seconds=CORPUS_LOCK_WAIT_SECONDS,
        waiting=waiting,
    )


def _passages(report: dict[str, Any]) -> str:
    """The report's chunk count, or what stands in for one the store would not give."""
    count = report.get("total_chunks")
    return "an unknown number of" if count is None else str(count)


def _corpus_feed_line(report: dict[str, Any], *, when: str = "before the run") -> str | None:
    """One line for the feed whenever a rebuild checked the corpus.

    `when` is all the background rebuild changes about it. The `current` line
    is said too, since silence would be indistinguishable from the phase never
    having run; only `disabled`, a setting the header already reports, says
    nothing.
    """
    source = report.get("source", "")
    if source == "disabled":
        return None
    if source == "current":
        return (
            f"[Corpus] The corpus already matched the archive, so nothing was "
            f"re-embedded: {report.get('indexed', 0)} document(s), "
            f"{_passages(report)} passage(s), checked in "
            f"{report.get('elapsed_s', 0)}s. The Researcher searches it as it "
            "stands."
        )

    errors = report.get("errors") or []
    # Named, not only counted: a count says the corpus is short and nothing
    # about whether the fix is a rebuild, a freed card or a file.
    note = (
        f" {len(errors)} file(s) failed to index: {'; '.join(errors[:3])}"
        + (f"; and {len(errors) - 3} more." if len(errors) > 3 else ".")
        if errors
        else ""
    )

    if source == "built":
        was = (
            "There was no corpus on this machine"
            if report.get("corpus") == "absent"
            else "The corpus on this machine was empty"
        )
        if not report.get("indexed"):
            return (
                f"[Corpus] {was}, and indexing produced nothing: every file the "
                f"walk offered was too large or unreadable.{note} The Researcher "
                "has nothing to retrieve."
            )
        return (
            f"[Corpus] {was}, so the archive was indexed {when}: "
            f"{report['indexed']} document(s), {_passages(report)} "
            f"passage(s) in {report.get('elapsed_s', 0)}s.{note} The Researcher "
            "searches this like any other corpus."
        )
    if source == "updated":
        changed = report.get("embedded", 0)
        dropped = report.get("dropped", 0)
        parts = []
        if changed:
            parts.append(f"{changed} document(s) re-read")
        if dropped:
            parts.append(f"{dropped} no longer in the archive (or no longer indexable) dropped")
        return (
            f"[Corpus] The corpus was behind the archive, so it was brought up "
            f"to date {when}: {', '.join(parts)}, {report.get('reused', 0)} "
            f"unchanged, in {report.get('elapsed_s', 0)}s.{note} The Researcher "
            "searches the archive as it is now."
        )
    if source == "stopped_midway":
        return (
            f"[Corpus] Stopped while indexing with {report.get('model')}: "
            f"{report.get('indexed', 0)} document(s) checked and "
            f"{report.get('embedded', 0)} embedded, in {report.get('elapsed_s', 0)}s. "
            "The next run carries on from there: what is already embedded keeps its "
            "vectors."
        )
    if source == "unavailable" and report.get("unavailable_circuit") == EMBEDDER_LOAD.name:
        return (
            f"[Corpus] {EMBEDDING_MODEL_NAME} could not be loaded onto the cards "
            f"{when}, so the rebuild stopped after {report.get('indexed', 0)} "
            f"document(s): {report.get('unavailable')}.{note} The console rebuilds "
            "the corpus once a load fits; `nvidia-smi` names what else holds the "
            "cards."
        )
    if source == "unavailable":
        return (
            f"[Corpus] The embedder could not be reached {when}, so the rebuild "
            f"stopped after {report.get('indexed', 0)} document(s): "
            f"{report.get('unavailable')}.{note} The console rebuilds the corpus "
            "once the daemon answers again."
        )
    if source == "nothing_to_index":
        return (
            "[Corpus] The archive is empty -- nothing researched online, "
            f"uploaded or embedded under {report.get('root', '.')}. The Researcher "
            "will find nothing; tick 'Research online' or upload a document to "
            "give it something."
        )
    if source == "busy_elsewhere":
        # Neither a failure nor a no-op, and it must read as neither: the work is
        # being done, by somebody else, and this caller left it alone rather than
        # pruning and clearing a store already being pruned and cleared.
        return (
            f"[Corpus] Another process is rebuilding this corpus, so it was left "
            f"alone after {report.get('waited_s', 0)}s. Nothing here changed it; "
            "what that rebuild finishes is what gets searched."
        )
    if source == "error":
        return (
            f"[Corpus] The corpus could not be brought up to date: "
            f"{report.get('note')}. The run continues against it as it stands."
        )
    return "[Corpus] Stopped before the corpus was checked."


def _research_online_before_the_run(goal: str, requested: bool) -> dict[str, Any]:
    """Search the web for the goal and embed what earns a place. Never raises.

    Only when the operator asked (`requested`), never by an agent's choice:
    three automatic gates were built and graded against hand-labelled pages,
    and all three failed -- whether the web helps is the operator's intent, and
    no comparison of goal text to page text recovers it. A kept page is a
    permanent corpus member, so guessing wrong costs every later run.

    It runs before `graph.stream`, the one ordering that keeps the corpus still
    while the Researcher searches it, and reaches the Researcher through the
    corpus alone: a page that cannot be retrieved for this goal does not reach
    the Builder because it was fetched for it. The creating door is handed over
    unopened, so a run that stores nothing leaves no corpus behind.
    """
    if not requested:
        # Kept apart from `disabled`: "not asked" and "switched off" call for
        # different things from the operator.
        return {"source": "not_requested", "documents": 0, "considered": 0, "note":
                "Online research was not requested for this run."}
    if RUN_CONTROL.stopped():
        return {"source": "stopped", "documents": 0, "considered": 0, "note":
                "Stopped before the online research phase began."}
    try:
        report = research_online(_kb_for_indexing, goal, should_stop=RUN_CONTROL.stopped)
    except Exception as exc:
        return {"source": "error", "documents": 0, "considered": 0,
                "note": f"The online research phase failed: {exc}"}
    # The phase writes files under research/web/, so the cached walk is now
    # behind what is on disk and the header would call the freshly embedded
    # pages `extra` until it expired.
    forget_cached_walk()
    return report


def _research_feed_line(report: dict[str, Any]) -> str:
    """One line for the feed, saying what the phase actually did.

    Worded so the empty outcomes cannot be mistaken for one another: "the web
    had nothing", "nobody asked" and "it broke" need different things from the
    operator, and an empty count reads the same in all three.
    """
    source = report.get("source", "")
    considered = report.get("considered", 0)
    kept = report.get("documents", 0)
    failed = report.get("failed") or []
    # Named, not only counted: a page that earned a place and was not embedded
    # once read as one the gate turned away, which sends the operator to the
    # goal when the fix is a freed card.
    unembedded = (
        f"earned a place but could not be embedded: {failed[0].get('error', '')}"
        if failed else ""
    )

    if source == "not_requested":
        return (
            "[Research] Online research was not requested; running against the "
            "corpus as it stands. Tick 'Research online' to search the web for "
            "this goal."
        )
    if source == "disabled":
        return "[Research] Online research is switched off; running against the corpus as it stands."
    if source == "stopped":
        return "[Research] Stopped before online research began."
    if source == "error" or (not considered and report.get("errors")):
        return f"[Research] Online research found nothing usable: {report.get('note') or 'the search failed'}"
    if report.get("stopped"):
        return (
            f"[Research] Stopped during online research, after embedding {kept} "
            "page(s); the pages still to embed were left alone."
        )
    if not kept and failed:
        saved = (
            f" Their text is saved under {WEB_RESEARCH_DIR}/, and the next rebuild embeds it."
            if any(page.get("saved") for page in failed) else ""
        )
        return f"[Research] Read {considered} page(s); {len(failed)} {unembedded}.{saved}"
    if not kept:
        return (
            f"[Research] Read {considered} page(s) and kept none -- none of them "
            "scored well enough against this goal to earn a place in the corpus."
        )
    more = f" {len(failed)} more {unembedded}." if failed else ""
    return (
        f"[Research] Read {considered} page(s), embedded {kept} "
        f"({report.get('chunks', 0)} passages) in {report.get('elapsed_s', 0)}s.{more} "
        "The Researcher retrieves these like any other document."
    )


def _before_the_run(goal: str, research_web: bool) -> tuple[list[str], dict[str, Any]]:
    """Ready the corpus, its floor and online research: (opening lines, research report).

    All three write to the corpus, so all three run before `graph.stream` --
    the one ordering `_refuse_while_a_run_is_in_flight` allows -- and the corpus
    goes first; see `_rebuild_the_corpus_before_the_run`.
    """
    corpus_report = _rebuild_the_corpus_before_the_run()
    corpus_line = _corpus_feed_line(corpus_report)
    if corpus_line:
        print(f"[run] corpus -> {corpus_line}")

    # The floor needs the whole corpus: measured after the rebuild, before any
    # seat searches.
    calibration_line = _calibration_feed_line(
        _calibrate_the_floor_before_the_run(corpus_report)
    )
    if calibration_line:
        print(f"[run] calibration -> {calibration_line}")

    research_report = _research_online_before_the_run(goal, research_web)
    research_line = _research_feed_line(research_report)
    print(f"[run] research -> {research_line}")
    lines = [line for line in (corpus_line, calibration_line, research_line) if line]
    return lines, research_report


def _how_it_ended(
    last: Mapping[str, Any], *, stopped: bool, over_budget: bool, elapsed: int, node_at_stop: str
) -> str | None:
    """The closing line for a run the emergency stop or the budget ended, or None."""
    # Both exits are checked between supersteps, so the Architect may already
    # have ruled `approved` when one lands: the verdict is read, not assumed.
    verdict_clause = (
        "The Architect had already ruled approved; the loop was ending anyway"
        if str(last.get("verdict", "")) == "approved"
        else "without an approved verdict"
    )
    if stopped:
        return (
            f"[Graph] Stopped by the emergency stop after {elapsed}s, at the "
            f"{node_at_stop or 'first'} boundary, {verdict_clause}. "
            "Nothing further was started. Anything already written is listed "
            "below, and anything nobody ran is unproven rather than working."
        )
    if over_budget:
        return (
            f"[Graph] Stopped after {elapsed}s, over the {int(RUN_BUDGET_SECONDS)}s "
            f"budget, {verdict_clause}. The work above is what the run "
            "produced. Raise RUN_BUDGET_SECONDS to give it longer."
        )
    return None


def _run_payload(
    last: Mapping[str, Any],
    run_id: str,
    *,
    stopped: bool,
    over_budget: bool,
    elapsed_s: int,
    web_research: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """The run's final state, plus how the run ended -- which AgentState does not hold."""
    payload = dict(last)
    payload.update(
        run_id=run_id,
        stopped=stopped,
        stop_reason=RUN_CONTROL.reason() if stopped else "",
        over_budget=over_budget,
        elapsed_s=elapsed_s,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        web_research=web_research,
        healing=HEALING.events(session_id=run_id),
        **extra,
    )
    return payload


def rpc_run_goal(params: dict[str, Any]) -> dict[str, Any]:
    """Run a goal through the four-agent loop and return the final state.

    One run at a time: it is checked, claimed and armed in one hold of
    `_run_lock`, and everything after the claim sits in a try/finally, so
    whatever raises, the run is released and its snapshot written.
    """
    run_id = uuid.uuid4().hex
    # Everything is parsed before anything is claimed or touched, so a refused
    # parameter never leaves a run armed with nothing behind it.
    goal = _str_param(params, "goal")
    if not goal.strip():
        raise ValueError("A run needs a goal, and this one was empty.")
    discuss_only = _bool_param(params, "discuss_only")
    research_web = _bool_param(params, "research_web")
    expect_failures = _bool_param(params, "expect_failures")
    # Where the Builder writes: a project under projects/, or this checkout when
    # blank. The caller's choice, never an agent's.
    project = _str_param(params, "project").strip()
    if project and (error := project_name_error(project)):
        raise ValueError(error)
    output_dir = project_dir(project) if project and not discuss_only else ""

    with _run_lock:
        if _run_progress["running"]:
            raise ValueError(
                "A run is already in flight. Stop it before starting another."
            )
        if _corpus_change["running"]:
            raise ValueError(
                f"The corpus is being {_corpus_change['action']} right now; start the "
                "run when that has finished."
            )
        if _pull_request_follow["running"]:
            # Its cleanup may switch the branch this run's Builder would work on.
            raise ValueError(
                f"The console is finishing pull request #{_pull_request_follow['number']} "
                "right now; start the run in a moment."
            )
        RUN_CONTROL.arm(run_id)
        # Under the lock `run_progress` reads the record through, so no reply
        # can hand this run's console the last run's turns.
        ACTIVITY.begin_run()
        _run_progress.update(
            running=True,
            goal=goal,
            messages=[],
            node="",
            run_id=run_id,
            stopping=False,
        )

    state = initial_state(
        goal,
        expect_failures=expect_failures,
        discuss_only=discuss_only,
        output_dir=output_dir,
    )
    last = state
    research_report: dict[str, Any] = {}
    started = time.monotonic()
    over_budget = False
    stopped = False
    node_at_stop = ""

    try:
        HEALING.start_healing_session(run_id)
        # A discussion run never researches online, whatever the box says: the
        # phase writes pages to disk and into every later run's corpus.
        opening, research_report = _before_the_run(goal, research_web and not discuss_only)
        # Into the state as well as the feed: what the run was told is part of
        # how its result should be read.
        state["messages"] = list(opening)
        with _run_lock:
            _run_progress["messages"] = list(opening)

        # The budget is the graph's; the phases above are bounded by their own
        # limits, so their time is not charged to it.
        started = time.monotonic()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Streamed rather than invoked, so the last state survives the
        # recursion ceiling: `graph.invoke` raises with no partial result.
        try:
            for event in graph.stream(state, {"recursion_limit": RECURSION_LIMIT}):
                for node, node_state in event.items():
                    if not isinstance(node_state, dict):
                        continue
                    last = node_state  # type: ignore[assignment]
                    messages = list(node_state.get("messages", []))
                    with _run_lock:
                        _run_progress["node"] = node
                        _run_progress["messages"] = messages
                    print(f"[run] {node} -> {messages[-1] if messages else '...'}")

                # Between supersteps, so the run stops at a node boundary with
                # its state intact; the nodes' own stop checks come first.
                if RUN_CONTROL.stopped():
                    stopped = True
                    node_at_stop = str(_run_progress["node"])
                    break
                if time.monotonic() - started > RUN_BUDGET_SECONDS:
                    over_budget = True
                    break
        except GraphRecursionError:
            last["messages"] = [
                *last.get("messages", []),
                "[Graph] Stopped at the recursion ceiling without an approved "
                "verdict. The work below is what the run produced before it "
                "was cut off.",
            ]

        elapsed = int(time.monotonic() - started)
        ending = _how_it_ended(
            last,
            stopped=stopped,
            over_budget=over_budget,
            elapsed=elapsed,
            node_at_stop=node_at_stop,
        )
        if ending:
            last["messages"] = [*last.get("messages", []), ending]

        payload = _run_payload(
            last,
            run_id,
            stopped=stopped,
            over_budget=over_budget,
            elapsed_s=elapsed,
            web_research=research_report,
            project_embedded=bool(project) and project in embedded_projects(),
        )
        _save_snapshot(payload)
        return payload
    except Exception as exc:
        # A run that raised still produced whatever it produced, and that is
        # when the operator most wants it back.
        _save_snapshot(_run_payload(
            last,
            run_id,
            stopped=stopped,
            over_budget=over_budget,
            elapsed_s=int(time.monotonic() - started),
            web_research=research_report,
            error=str(exc),
        ))
        raise
    finally:
        # Whatever ended the run, a pull request it left waiting on its checks
        # is the console's to finish.
        _track_pull_request(last.get("dwell") or {}, run_id, goal)
        _finish_run()


# --------------------------------------------------------------------------
# Self-healing: health checks, the monitor, and what the console reads of them
# --------------------------------------------------------------------------

# How often the console checks the services it depends on and repairs what it
# can: an open circuit is re-probed, and a corpus whose last rebuild stopped
# because the embedder could not be reached, or its model would not load, is
# rebuilt once it answers again or its cooldown is over.
HEALTH_CHECK_SECONDS = float(os.getenv("HEALTH_CHECK_SECONDS", "30"))

# The latest answer from each service, as `{status, details}` per component.
_health: dict[str, dict[str, str]] = {}
_health_lock = threading.Lock()

# The newest rebuild either phase finished, for the monitor to judge.
_last_rebuild: dict[str, Any] = {}


def _check_health() -> dict[str, dict[str, str]]:
    """Ask each service the console depends on whether it is answering."""
    results: dict[str, dict[str, str]] = {}
    # Through the daemon's circuit: while it is open this is a refusal, and
    # once the cooldown is over this probe is the trial call that closes it.
    try:
        version = daemon_request("/api/version", timeout=3.0)
        results["ollama-daemon"] = {
            "status": "healthy",
            "details": f"Ollama {version.get('version', '?')} at {ollama_base_url()}",
        }
    except CircuitOpenError as exc:
        results["ollama-daemon"] = {"status": "unhealthy", "details": str(exc)}
    except Exception as exc:
        results["ollama-daemon"] = {
            "status": "unhealthy",
            "details": f"{ollama_base_url()}: {type(exc).__name__}: {exc}",
        }

    # The database the corpus lives in, through its circuit, as the daemon is.
    results["postgres"] = get_database().health()

    search = search_backend_health()
    if search is not None:
        results["searxng"] = search

    with _run_lock:
        last = dict(_last_rebuild)
    if last.get("source") == "unavailable":
        circuit = last.get("unavailable_circuit")
        results["corpus"] = {
            "status": "unhealthy",
            "details": (
                "the last rebuild stopped because the embedding model would not load"
                if circuit == EMBEDDER_LOAD.name
                else "the last rebuild stopped because the database could not be reached"
                if circuit == POSTGRES.name
                else "the last rebuild stopped because the embedder could not be reached"
            ),
        }
    else:
        results["corpus"] = {"status": "healthy", "details": f"corpus {corpus_state()[0]}"}
    return results


def _heal() -> None:
    """One pass of the monitor: check every service, then repair what can be repaired."""
    results = _check_health()
    with _health_lock:
        previous = dict(_health)
        _health.clear()
        _health.update(results)
    for component, result in results.items():
        if result["status"] != previous.get(component, {}).get("status"):
            HEALING.log_health_check(component, result["status"], result["details"])

    # A pull request a run left waiting on its checks is merged once they pass.
    _follow_pull_requests()

    # A rebuild stopped by an unreachable embedder or database is finished once
    # what stopped it is back. Nothing else rebuilds on its own: any other
    # failure is not one waiting on a service, and a run rebuilds before it
    # starts anyway. A model that would not load is waited out for its
    # circuit's cooldown, which a rebuild tried every pass would only meet as a
    # refusal, logged each time. Every rebuild needs both services.
    with _run_lock:
        stopped_by_database = _last_rebuild.get("unavailable_circuit") == POSTGRES.name
    if (
        REBUILD_CORPUS
        and results["corpus"]["status"] == "unhealthy"
        and results["ollama-daemon"]["status"] == "healthy"
        and results["postgres"]["status"] == "healthy"
        and not EMBEDDER_LOAD.retry_in
        and not _run_in_flight()
    ):
        when = (
            "after the database came back" if stopped_by_database
            else "after the embedder came back"
        )
        report = _rebuild_the_corpus_in_background(when)
        if report is not None:  # None: another rebuild was already under way
            recovered = report.get("source") not in ("unavailable", "error", "busy_elsewhere")
            HEALING.log_recovery_action(
                "rebuild corpus", "corpus", recovered,
                _corpus_feed_line(report, when=when) or str(report.get("source", "")),
            )


# --------------------------------------------------------------------------
# Pull requests a run left waiting on their checks
# --------------------------------------------------------------------------

# CI routinely outlasts the Builder's deadline, so a run's `git_dwell` ends
# with its pull request open and its checks still running -- "pending", not
# failed. The monitor finishes it: it reads the checks without waiting, merges
# exactly the head the run pushed once they pass, and cleans up, as the run
# would have (`finish_pull_request`). A red check is reported, never fixed: a
# fix is a run, and the operator starts it.

# How often one pull request is asked about, and how long before the console
# stops: CI that has not finished in a day is not going to by itself.
PULL_REQUEST_FOLLOW_SECONDS = float(os.getenv("PULL_REQUEST_FOLLOW_SECONDS", "60"))
PULL_REQUEST_FOLLOW_HOURS = float(os.getenv("PULL_REQUEST_FOLLOW_HOURS", "24"))

# Answers in a row that were neither a verdict nor "still running" -- gh signed
# out, GitHub down, a push by someone else -- before the console stops asking.
PULL_REQUEST_FOLLOW_FAILURES = 3

# Followed pull requests live on disk only, read and written whole under this
# lock: the file is a handful of entries, and a restart goes on following them.
_pull_requests_lock = threading.Lock()

# Set under `_run_lock` while the monitor finishes one: a run is refused
# meanwhile, since the cleanup may switch the branch its Builder would use.
_pull_request_follow: dict[str, Any] = {"running": False, "number": None}

# The fields the console shows; the rest is the monitor's bookkeeping.
_PULL_REQUEST_FIELDS = (
    "key", "cwd", "branch", "number", "url", "status", "detail", "goal",
    "checks_failed", "merge_commit", "warnings", "since",
)


def _pull_requests_path() -> Path:
    return RUNS_DIR / "pull_requests.json"


def _read_pull_requests() -> list[dict[str, Any]]:
    """Every followed pull request. The caller holds `_pull_requests_lock`."""
    try:
        loaded = json.loads(_pull_requests_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [entry for entry in loaded if isinstance(entry, dict)] if isinstance(loaded, list) else []


def _write_pull_requests(entries: list[dict[str, Any]]) -> None:
    """Replace the file whole, through a rename. The caller holds `_pull_requests_lock`."""
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=RUNS_DIR, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(entries, handle, indent=2)
            temp_name = handle.name
        os.replace(temp_name, _pull_requests_path())
    except OSError as exc:
        HEALING.warning(f"Could not save the followed pull requests: {exc}",
                        action="pull_request_save_failed")


def _track_pull_request(dwell: Mapping[str, Any], run_id: str, goal: str) -> None:
    """Follow the pull request a run left pending. Never raises: the run is over."""
    try:
        if dwell.get("status") != "pending" or not dwell.get("number") or not dwell.get("head"):
            return
        entry = {
            "key": f"{dwell.get('cwd') or '.'}#{dwell['number']}",
            "cwd": str(dwell.get("cwd") or ""),
            "branch": str(dwell.get("branch") or ""),
            "number": int(dwell["number"]),
            "url": str(dwell.get("url") or ""),
            "head": str(dwell["head"]),
            "status": "pending",
            "detail": str(dwell.get("pending") or ""),
            "goal": goal[:200],
            "run_id": run_id,
            "checks_failed": [],
            "merge_commit": "",
            "warnings": [],
            "since": time.time(),
            "next_check": 0.0,
            "failures": 0,
        }
        with _pull_requests_lock:
            entries = [e for e in _read_pull_requests() if e.get("key") != entry["key"]]
            _write_pull_requests([*entries, entry])
        HEALING.info(
            f"Following pull request #{entry['number']} until its checks finish: "
            f"{entry['detail']}",
            action="pull_request_followed",
        )
    except Exception as exc:
        HEALING.warning(f"Could not follow the run's pull request: {exc}",
                        action="pull_request_save_failed")


def pull_requests_snapshot() -> list[dict[str, Any]]:
    """What the header shows: each followed pull request and where it stands."""
    with _pull_requests_lock:
        entries = _read_pull_requests()
    return [{field: entry.get(field) for field in _PULL_REQUEST_FIELDS} for entry in entries]


def _finish_one(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Ask GitHub once about one followed pull request; return what changed."""
    number, now = entry.get("number"), time.time()
    cwd = str(entry.get("cwd") or "") or None
    if cwd is not None and not Path(cwd).is_dir():
        return {"status": "stopped", "detail": f"{cwd} is gone, and its repository with it"}
    if now - float(entry.get("since") or now) > PULL_REQUEST_FOLLOW_HOURS * 3600:
        HEALING.warning(
            f"Stopped following pull request #{number}: still waiting after "
            f"{PULL_REQUEST_FOLLOW_HOURS:g}h. Finish it on GitHub.",
            action="pull_request_stopped",
        )
        return {"status": "gave_up", "detail": (
            f"still waiting after {PULL_REQUEST_FOLLOW_HOURS:g}h: {entry.get('detail')}"
        )}
    result = finish_pull_request(
        MCPClient()._run_vcs,
        cwd=cwd,
        branch=str(entry.get("branch") or ""),
        number=int(number or 0),
        head=str(entry.get("head") or ""),
    )
    warnings_ = list(result.get("warnings") or [])
    if result.get("merged"):
        landed = result.get("merge_commit") or ""
        HEALING.info(
            f"Pull request #{number} merged into {result.get('default_branch') or 'its base'}"
            + (f" as {landed[:7]}" if landed else "") + " once its checks passed"
            + (f" ({'; '.join(warnings_)})" if warnings_ else ""),
            action="pull_request_merged",
        )
        return {"status": "merged", "detail": "merged", "merge_commit": landed,
                "warnings": warnings_}
    if result.get("success"):
        return {"detail": str(result.get("pending") or "waiting"), "failures": 0,
                "next_check": now + PULL_REQUEST_FOLLOW_SECONDS}
    error = str(result.get("error") or "git_dwell gave no reason")
    if "closed without merging" in error:
        HEALING.warning(f"Stopped following pull request #{number}: it was closed "
                        "without merging.", action="pull_request_stopped")
        return {"status": "stopped", "detail": error[:500]}
    if result.get("checks_failed"):
        names = ", ".join(result["checks_failed"])
        HEALING.error(f"Pull request #{number}: its checks fail ({names}). Start a run to fix "
                      "it; nothing was merged.", action="pull_request_checks_failed")
        return {"status": "checks_failed", "checks_failed": list(result["checks_failed"]),
                "detail": error[:500]}
    failures = int(entry.get("failures") or 0) + 1
    if failures >= PULL_REQUEST_FOLLOW_FAILURES:
        HEALING.warning(f"Stopped following pull request #{number}: {error[:300]}",
                        action="pull_request_stopped")
        return {"status": "stopped", "detail": error[:500], "failures": failures}
    return {"detail": error[:500], "failures": failures,
            "next_check": now + PULL_REQUEST_FOLLOW_SECONDS}


def _follow_pull_requests() -> None:
    """Finish each followed pull request that is due, unless a run is in flight.

    Never while a run is: its Builder may be working in the same repository,
    and the cleanup switches branches. The lock is not held across GitHub: a
    finish asks gh several times, and the header reads the list meanwhile.
    """
    now = time.time()
    with _pull_requests_lock:
        due = [
            dict(entry) for entry in _read_pull_requests()
            if entry.get("status") == "pending" and float(entry.get("next_check") or 0) <= now
        ]
    for entry in due:
        with _run_lock:
            if _run_progress["running"]:
                return
            _pull_request_follow.update(running=True, number=entry.get("number"))
        try:
            change = _finish_one(entry)
        except Exception as exc:
            change = {"detail": f"{type(exc).__name__}: {exc}",
                      "next_check": time.time() + PULL_REQUEST_FOLLOW_SECONDS}
        finally:
            with _run_lock:
                _pull_request_follow.update(running=False, number=None)
        with _pull_requests_lock:
            entries = _read_pull_requests()
            for stored in entries:
                if stored.get("key") == entry.get("key"):
                    stored.update(change)
            _write_pull_requests(entries)


def rpc_dismiss_pull_request(params: dict[str, Any]) -> dict[str, Any]:
    """Stop showing -- and, if it is still pending, stop following -- one pull request."""
    key = _str_param(params, "key")
    if not key:
        raise ValueError("Name the pull request to dismiss.")
    with _pull_requests_lock:
        entries = _read_pull_requests()
        kept = [entry for entry in entries if entry.get("key") != key]
        if len(kept) == len(entries):
            raise ValueError(f"No pull request {key!r} is being followed.")
        _write_pull_requests(kept)
    return {"pull_requests": pull_requests_snapshot()}


def _self_healing_monitor() -> None:
    """Run `_heal` now and every `HEALTH_CHECK_SECONDS` until the console exits."""
    while True:
        try:
            _heal()
        except Exception as exc:  # the monitor must outlive any one bad pass
            HEALING.error(f"A self-healing pass failed: {type(exc).__name__}: {exc}",
                          action="monitor_failed")
        if _shutdown_requested.wait(HEALTH_CHECK_SECONDS):
            return


def rpc_healing(params: dict[str, Any]) -> dict[str, Any]:
    """Self-healing at a glance: circuits, service health, and the journal after `since`."""
    since = _int_param(params, "since", 0, low=0, high=MAX_EVENT_SEQUENCE)
    with _health_lock:
        health = {name: dict(result) for name, result in _health.items()}
    return {
        "circuits": circuit_states(),
        "health": health,
        "events": HEALING.events(since=since),
        "session": HEALING.session_id,
        "interval_s": HEALTH_CHECK_SECONDS,
    }


def rpc_reset_circuit(params: dict[str, Any]) -> dict[str, Any]:
    """Close a circuit now, for an operator who knows its service is back."""
    name = _str_param(params, "name")
    if not name:
        raise ValueError("Name the circuit to reset.")
    try:
        closed = reset_circuit(name)
    except KeyError:
        raise ValueError(f"There is no circuit called {name!r}.") from None
    HEALING.log_recovery_action("reset circuit", name, True, "asked for from the console")
    return {"reset": closed, "circuits": circuit_states()}


RPC_METHODS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "rag_stats": rpc_rag_stats,
    "bottleneck": rpc_bottleneck,
    "topics": rpc_topics,
    "duplicate_entities": rpc_duplicate_entities,
    "list_documents": rpc_list_documents,
    "query_graph": rpc_query_graph,
    "graph_overview": rpc_graph_overview,
    "search_documents": rpc_search_documents,
    "upload_document": rpc_upload_document,
    "list_projects": rpc_list_projects,
    "embed_project": rpc_embed_project,
    "export_corpus": rpc_export_corpus,
    "clear_corpus": rpc_clear_corpus,
    "list_seats": rpc_list_seats,
    "set_seat": rpc_set_seat,
    "set_thinking": rpc_set_thinking,
    "llm_options": rpc_llm_options,
    "embedding_options": rpc_embedding_options,
    "status": rpc_status,
    "run_goal": rpc_run_goal,
    "run_progress": rpc_run_progress,
    "embedding_activity": rpc_embedding_activity,
    "stop_run": rpc_stop_run,
    "last_run": rpc_last_run,
    "healing": rpc_healing,
    "reset_circuit": rpc_reset_circuit,
    "dismiss_pull_request": rpc_dismiss_pull_request,
    "shutdown": rpc_shutdown,
}

# Methods the console polls on a timer; logging them buries everything else.
QUIET_METHODS = {
    "status",
    "rag_stats",
    "list_seats",
    "llm_options",
    "embedding_options",
    "run_progress",
    # Polled several times a second, though only while a search or an upload of
    # the console's own is in flight; a run's poll carries the same reading.
    "embedding_activity",
    "healing",
}


# Where the console listens: loopback unless the operator says otherwise,
# because the console runs goals and a goal reaches a Builder that runs
# programs. `CONSOLE_HOST=0.0.0.0` opens it up for an operator who wants that.
# Not `HOST`, which some shells export as the machine's own name.
CONSOLE_HOST = os.getenv("CONSOLE_HOST", "127.0.0.1").strip() or "127.0.0.1"

# The names a browser gives this machine's loopback, as a Host header carries
# them once the port is gone (`urlsplit` lowercases, and strips an IPv6 address
# of its brackets).
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _listening_on_loopback() -> bool:
    return CONSOLE_HOST in _LOOPBACK_NAMES or CONSOLE_HOST.startswith("127.")


def _foreign_request(headers: Any, *, writes: bool) -> str | None:
    """Why this request did not come from the console's own page, or None.

    Each check closes a way for a page the operator merely visits to drive the
    console, and so run programs on this machine:

    - **Origin**, on anything that writes: a cross-site `text/plain` POST is a
      "simple" request, sent without asking first.
    - **Host**, while listening on loopback: a page that re-points its own
      domain at 127.0.0.1 (DNS rebinding) is same-origin by the browser's
      account, and its Host header still names it.

    Nothing without those headers is refused: curl, the launcher's readiness
    poll and the tests send no Origin.
    """
    host = str(headers.get("Host") or "")
    if host and _listening_on_loopback():
        name = urlsplit(f"//{host}").hostname or ""
        if name not in _LOOPBACK_NAMES:
            return (
                f"Host {host!r} is not this machine's loopback, and the console "
                "listens only there. Open it at http://localhost instead."
            )
    origin = headers.get("Origin")
    if writes and origin is not None:
        if urlsplit(str(origin)).netloc.lower() != host.lower():
            return (
                f"A page at {origin!r} may not drive this console: only the "
                "console's own page can. Nothing was run."
            )
    return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory="frontend", **kwargs)

    def _refused_as_foreign(self, *, writes: bool) -> bool:
        """Refuse a request from anywhere but the console's own page; True if refused."""
        reason = _foreign_request(self.headers, writes=writes)
        if reason is None:
            return False
        print(f"[API] refused {self.command} {self.path}: {reason}")
        # The reason goes in the body: `send_error`'s message is the status
        # line's reason phrase, which is no place for a sentence.
        self.send_error(403, "Refused", reason)
        return True

    def do_GET(self) -> None:
        if self._refused_as_foreign(writes=False):
            return
        if urlparse(self.path).path == "/api/status":
            self.send_json(rpc_status({}))
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self._refused_as_foreign(writes=True):
            return
        parsed = urlparse(self.path)
        # A body of unusable length is refused before it is read, and one that
        # is not a JSON object after.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._refuse("Content-Length is not a number.")
            return
        if not 0 <= length <= MAX_REQUEST_BYTES:
            self._refuse(
                f"A request body is limited to {MAX_REQUEST_BYTES:,} bytes; "
                f"this one declared {length:,}."
            )
            return
        try:
            data = json.loads(self.rfile.read(length)) if length else {}
        except ValueError:  # bad JSON, or bytes that are not UTF-8
            self._refuse("bad JSON")
            return
        if not isinstance(data, dict):
            self._refuse("The request body must be a JSON object.")
            return

        if parsed.path == "/rpc":
            self.handle_rpc(data)
            return
        self.send_error(404)

    def handle_rpc(self, data: dict[str, Any]) -> None:
        """Dispatch one {method, params} call.

        Errors come back as a 200 with an `error` member: a failed method is not a
        failed request. The reply is written apart from the method's outcome, so a
        caller gone by then -- a console reloaded mid-run, which reattaches through
        `run_progress` -- is not reported as the method failing.
        """
        method = str(data.get("method", ""))
        params = data.get("params") or {}
        if not isinstance(params, dict):
            self._refuse("params must be a JSON object.")
            return
        started = time.perf_counter()

        handler = RPC_METHODS.get(method)
        if handler is None:
            self.send_json(
                {"error": {"message": f"Unknown method: {method}"}, "elapsed_ms": 0}
            )
            return

        try:
            result = handler(params)
            elapsed = int((time.perf_counter() - started) * 1000)
            # Encoded inside the `try`, so a result that will not serialise is
            # still reported as this method failing.
            payload = self._encode({"result": result, "elapsed_ms": elapsed})
            if method not in QUIET_METHODS:
                print(f"[RPC] {method} {elapsed}ms")
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            print(f"[RPC] {method} FAILED {elapsed}ms: {exc}")
            payload = self._encode({"error": {"message": str(exc)}, "elapsed_ms": elapsed})

        try:
            self._send_payload(payload)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[RPC] {method}: the caller disconnected before the reply was sent")

    def _refuse(self, message: str) -> None:
        self.send_json({"error": {"message": message}, "elapsed_ms": 0})

    @staticmethod
    def _encode(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode()

    def send_json(self, obj: Any) -> None:
        self._send_payload(self._encode(obj))

    def _send_payload(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # No Access-Control-Allow-Origin: the console's page is this same
        # origin, and a `*` would let any other site read every reply.
        self.end_headers()
        self.wfile.write(payload)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        # RPC calls log themselves in handle_rpc, with their method name and
        # timing; the default line would add a second, less useful entry, and
        # the status poll's would drown the rest.
        if "POST /rpc" in self.requestline or "GET /api/status" in self.requestline:
            return
        print(f"[API] {self.requestline} -> {int(code) if isinstance(code, int) else code}")

    def log_error(self, format: str, *args: Any) -> None:
        # `send_error` states its code here before `log_request` repeats it
        # beside the request line; printed alone, a favicon's 404 read as the
        # page's own, on the line after `GET /`.
        if not format.startswith("code %d"):
            self.log_message(format, *args)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[API] {format % args}")


def _exit_the_way_the_console_would(signum: int, _frame: Any) -> None:
    """Handle SIGTERM by asking for the exit the console's X asks for.

    SIGTERM is how a container runtime and systemd stop a process, and Python's
    default ends it where it stands, skipping the `finally` that writes a run's
    snapshot. This stops the run and defers the exit to that `finally`. The
    signal arrives on the main thread, which never holds `_run_lock`, so the
    handshake cannot deadlock. SIGINT is left alone: `main` catches Ctrl+C.
    """
    print(f"\nReceived {signal.Signals(signum).name}; exiting through the console's own path")
    reply = rpc_shutdown({"stop_first": True})
    if reply.get("running"):
        print(f"  {reply['detail']}")


def main() -> None:
    """Serve until Ctrl+C or the console asks to exit."""
    # Line-buffered: launch_console.sh redirects this process to a log and
    # tails it, and a redirected stdout is block-buffered. `reconfigure` is on
    # the TextIOWrapper stdout is at runtime, not on the TextIO it is typed as.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    port = int(os.getenv("PORT", "8080"))
    if _listening_on_loopback():
        print(f"Serving at http://localhost:{port}")
    else:
        # Said, because it is the one setting that makes the console reachable
        # from other machines -- see CONSOLE_HOST.
        print(f"Serving at http://{CONSOLE_HOST}:{port} (CONSOLE_HOST: reachable beyond this machine)")
    print("Press Ctrl+C to stop, or use the console's exit button")

    # Threaded: a run takes as long as four seat models take, and a
    # single-threaded server would stall every status poll behind it.
    server = ThreadingHTTPServer((CONSOLE_HOST, port), Handler)
    # `serve_forever` runs off the main thread, which waits on both ways out:
    # Ctrl+C, and the console asking to exit. Shutting down from the request
    # thread that asked would deadlock.
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()

    # After the serve loop, never at import (see
    # `_rebuild_the_corpus_in_background`). A daemon thread, so Ctrl+C is not
    # held up: the rebuild checks for the exit between files and keeps what it
    # has embedded.
    threading.Thread(
        target=_rebuild_the_corpus_in_background, name="corpus-rebuild", daemon=True
    ).start()

    threading.Thread(
        target=_self_healing_monitor, name="self-healing", daemon=True
    ).start()

    signal.signal(signal.SIGTERM, _exit_the_way_the_console_would)

    asked_from_console = False
    try:
        # Waited in slices rather than indefinitely: an untimed wait swallows
        # the KeyboardInterrupt this loop exists to catch.
        while not _shutdown_requested.wait(0.5):
            pass
        asked_from_console = True
    except KeyboardInterrupt:
        pass

    print("\nShutting down (from the console)..." if asked_from_console else "\nShutting down...")
    if asked_from_console:
        # Let the reply that asked for this reach the browser.
        time.sleep(SHUTDOWN_GRACE_SECONDS)
    try:
        server.shutdown()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
