"""Online research: the web half of the corpus.

A pre-run phase. The goal is searched, the pages that earn a place are written
under `WEB_RESEARCH_DIR` and embedded, and only then does the Architect open:

* No corpus write is allowed once a run is live
  (`_refuse_while_a_run_is_in_flight` in serve.py), since a corpus changing
  under the Researcher manufactures an absence no seat can detect. Running
  before the graph is the ordering that obeys that rule.
* The file is written before it is embedded, as an upload is: the corpus is
  rebuilt from what is on disk, so `WEB_RESEARCH_DIR` stays one of
  `CORPUS_ROOTS`.

Keyless and free: results come from DuckDuckGo's HTML endpoint or an
operator-hosted SearxNG -- the one thing taken from outside. The rest is ours:

1. **Fan out** -- `expand_queries` derives several queries from the goal with
   the identifier-aware `tokenize`.
2. **Fuse** the rankings with `reciprocal_rank_fusion`, so a page several
   queries agree on outranks one that won a single query.
3. **Extract** with `html_text`, which says how much of each page it kept.
4. **Gate** on BM25 against the goal: the engine's rank decides what is
   fetched, not what is embedded. The bar is a ratio of the best page's
   score, since a BM25 score is corpus-relative (see `lexical.py`).

Searches and fetches are retried only for failures a second try can fix, and
each backend has a circuit, so a backend that is down or blocking is left
alone for `WEB_SEARCH_COOLDOWN_SECONDS`.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

# `_document_metadata` defines what a document is filed under; a second
# spelling of it would file one file as two documents.
from langgraph_agent.graphrag_server import (
    MAX_INDEXABLE_BYTES,
    WEB_RESEARCH_DIR,
    EmbeddingStopped,
    _document_metadata,
)
from langgraph_agent.html_text import extract
from langgraph_agent.lexical import BM25Index, reciprocal_rank_fusion, tokenize
from langgraph_agent.self_healing import Circuit, CircuitOpenError, call_with_retry

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


# Keyless, no account, no quota. DuckDuckGo's HTML endpoint by default, or a
# SearxNG the operator runs; every other general web search API charges.
DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"
SEARXNG_URL = os.environ.get("SEARXNG_URL", "").strip().rstrip("/")

# The machine-level off switch, for a machine with no network.
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", "1").strip() not in {"0", "false", "no"}

# How many queries the goal fans out into, and results per query; the fetch
# limit applies after fusion has ordered them.
WEB_SEARCH_QUERIES = int(os.environ.get("WEB_SEARCH_QUERIES", "3"))
WEB_RESULTS_PER_QUERY = int(os.environ.get("WEB_RESULTS_PER_QUERY", "10"))
WEB_FETCH_LIMIT = int(os.environ.get("WEB_FETCH_LIMIT", "12"))

# The longest query sent. An over-long query is refused, not searched worse:
# DuckDuckGo redirects anything past ~500 characters to an error page. 250 sits
# inside the range measured to return results.
WEB_QUERY_MAX_CHARS = int(os.environ.get("WEB_QUERY_MAX_CHARS", "250"))

# How much of a query an error message quotes.
_QUERY_LABEL_CHARS = 60

# How many pages may enter the corpus for one goal: a ceiling on how far one
# run can shift what the Researcher sees.
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "8"))

# Keep a page scoring at least this fraction of the best page's BM25 score
# against the goal. A page scoring zero is dropped whatever this says.
WEB_SELECT_RATIO = float(os.environ.get("WEB_SELECT_RATIO", "0.25"))

WEB_SEARCH_TIMEOUT_SECONDS = float(os.environ.get("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
WEB_FETCH_TIMEOUT_SECONDS = float(os.environ.get("WEB_FETCH_TIMEOUT_SECONDS", "20"))
# The whole phase, which sits before the graph and so before
# `RUN_BUDGET_SECONDS` starts.
WEB_RESEARCH_BUDGET_SECONDS = float(os.environ.get("WEB_RESEARCH_BUDGET_SECONDS", "120"))
WEB_FETCH_WORKERS = int(os.environ.get("WEB_FETCH_WORKERS", "4"))

# Attempts per search query and per page while the failure is one a second try
# can fix at once -- a refused or dropped connection, a 502/503/504 -- and the
# wait between them. A timeout is never retried: it has already spent the time
# a retry would.
WEB_ATTEMPTS = 2
WEB_RETRY_WAIT_SECONDS = 1.0

# How long a search backend that keeps failing -- down, unreachable, or behind
# a bot check -- is left alone. Every request sent into a block extends it, and
# a backend that is down will not be back in a second.
WEB_SEARCH_COOLDOWN_SECONDS = float(os.environ.get("WEB_SEARCH_COOLDOWN_SECONDS", "120"))

# Sent on every request: endpoints rate-limit a blank agent first, and an
# operator reading logs can tell what this traffic is.
USER_AGENT = os.environ.get(
    "WEB_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) langgraph-agent/1.0 (+research)",
)

WEB_SEARCH_DISABLED_NOTE = (
    "WEB_SEARCH_ENABLED is off, so nothing was researched online and nothing "
    "was added to the corpus. The run continues against the corpus as it "
    "stands."
)

# Only the scaffolding of an instruction ("please add a function that..."): a
# stopword list that reaches into the vocabulary loses terms a query needs.
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "to", "of", "in", "on", "for", "with", "is",
        "are", "be", "that", "this", "it", "as", "at", "by", "or", "from",
        "please", "can", "you", "we", "i", "should", "would", "make", "add",
        "use", "using", "how", "do", "does", "need", "want", "let", "lets",
        "our", "my", "into", "so", "if", "then", "than", "but", "not", "all",
    }
)

# Appended to the term query, in order, to aim the remaining searches at
# material worth embedding.
_QUERY_INTENTS = ("documentation reference", "example implementation", "best practices")

# Longest readable part of a stored page's filename, before its digest.
_MAX_SLUG_CHARS = 60


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _fit_query(text: str, limit: int) -> str:
    """The longest leading run of whole words in `text` that fits in `limit`.

    A first word longer than the limit is cut rather than dropped: an empty query
    is no search at all.
    """
    if limit < 1:
        return ""
    words = text.split()
    kept: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if kept else 0)
        if length + added > limit:
            break
        kept.append(word)
        length += added
    if not kept and words:
        return words[0][:limit]
    return " ".join(kept)


def _leading_sentences(text: str, limit: int) -> str:
    """As many whole leading sentences of `text` as fit in `limit`, else ""."""
    kept = ""
    for sentence in _SENTENCE_BREAK.split(text):
        candidate = f"{kept} {sentence}" if kept else sentence
        if len(candidate) > limit:
            break
        kept = candidate
    return kept


def expand_queries(goal: str, limit: int | None = None) -> list[str]:
    """Derive several search queries from one goal.

    The verbatim goal comes first when it fits -- the one query carrying the
    phrasing a person chose. The rest are the goal's content terms, through
    `tokenize`, so an identifier contributes its pieces too
    (`BUILDER_DEADLINE_SECONDS` searches `builder deadline seconds`). Every query
    fits `WEB_QUERY_MAX_CHARS`; duplicates collapse, order is kept.
    """
    wanted = max(1, limit or WEB_SEARCH_QUERIES)
    goal = " ".join(goal.split())
    if not goal:
        return []

    seen: set[str] = set()
    queries: list[str] = []

    def offer(query: str) -> None:
        cleaned = " ".join(query.split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            queries.append(cleaned)

    offer(_leading_sentences(goal, WEB_QUERY_MAX_CHARS))

    # First-appearance order: a goal's leading terms are what it is about, and
    # what survives the length cap.
    terms = [t for t in dict.fromkeys(tokenize(goal)) if t not in _QUERY_STOPWORDS and len(t) > 1]
    if terms:
        joined = " ".join(terms)
        offer(_fit_query(joined, WEB_QUERY_MAX_CHARS))
        for intent in _QUERY_INTENTS:
            if len(queries) >= wanted:
                break
            head = _fit_query(joined, WEB_QUERY_MAX_CHARS - len(intent) - 1)
            if head:
                offer(f"{head} {intent}")

    return queries[:wanted]


class _DuckDuckGoResults(HTMLParser):
    """Pull result links out of the keyless HTML endpoint.

    Parsed rather than regexed: this markup is someone else's and changes without
    notice. It fails to an empty list, reported as a search that found nothing.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result__a" in classes and attributes.get("href"):
            self._href = attributes["href"]
            self._title = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            url = _unwrap_redirect(self._href)
            if url.startswith(("http://", "https://")):
                self.results.append({"url": url, "title": " ".join("".join(self._title).split())})
            self._href = None
            self._title = []


def _unwrap_redirect(href: str) -> str:
    """Recover the real URL from DuckDuckGo's `/l/?uddg=` wrapper, which some
    results come wrapped in -- filing them under duckduckgo.com otherwise.
    """
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


class _SearchBlocked(Exception):
    """The engine answered with a challenge page instead of results.

    Unlike a failed query, a block covers every query from this address, and each
    further request prolongs it.
    """


# DuckDuckGo's bot check is a `202` carrying a challenge form and no results,
# so it is recognised by its markup, never by its status.
_DUCKDUCKGO_CHALLENGE_MARKERS = ("anomaly-modal", "/anomaly.js")


def _quick_transient(exc: BaseException) -> bool:
    """A failure a second try can fix at once: a refused or dropped connection, or a 502-504."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (502, 503, 504)
    return isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError))


def _backend_failed(exc: BaseException) -> bool:
    """A search backend that did not answer, answered 5xx, or put up its bot check."""
    if isinstance(exc, _SearchBlocked):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def _search_circuit(backend: str) -> Circuit:
    return Circuit(
        f"web-search:{backend}",
        failure_threshold=3,
        recovery_timeout=WEB_SEARCH_COOLDOWN_SECONDS,
        trips_on=_backend_failed,
    )


def search_backend_health() -> dict[str, str] | None:
    """Whether the operator's SearxNG answers, asked through its circuit.

    None when there is nothing to ask: the phase is off, or the backend is
    DuckDuckGo, which is never probed on a timer -- it rate-limits exactly that.
    Asked through the circuit, so a probe after the cooldown is the trial call
    that closes it again.
    """
    if not WEB_SEARCH_ENABLED or not SEARXNG_URL:
        return None

    def probe() -> None:
        response = httpx.get(
            f"{SEARXNG_URL}/healthz", timeout=3.0, headers={"User-Agent": USER_AGENT}
        )
        if response.status_code >= 500:
            response.raise_for_status()

    try:
        _search_circuit("searxng").call(probe)
    except Exception as exc:
        return {"status": "unhealthy", "details": f"{SEARXNG_URL}: {type(exc).__name__}: {exc}"}
    return {"status": "healthy", "details": SEARXNG_URL}


def _search_duckduckgo(query: str, count: int, client: httpx.Client) -> list[dict[str, str]]:
    response = client.post(
        DUCKDUCKGO_ENDPOINT,
        data={"q": query},
        timeout=WEB_SEARCH_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    if any(marker in response.text for marker in _DUCKDUCKGO_CHALLENGE_MARKERS):
        raise _SearchBlocked(
            f"DuckDuckGo answered with its bot check (HTTP {response.status_code}) "
            "instead of results. It lifts on its own; a SearxNG instance "
            "(SEARXNG_URL) avoids it."
        )
    parser = _DuckDuckGoResults()
    parser.feed(response.text)
    parser.close()
    return parser.results[:count]


def _search_searxng(query: str, count: int, client: httpx.Client) -> list[dict[str, str]]:
    response = client.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json"},
        timeout=WEB_SEARCH_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in (payload.get("results") or [])[:count]:
        url = (item.get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            results.append({"url": url, "title": (item.get("title") or "").strip()})
    return results


def _query_label(query: str) -> str:
    """Name a query in an error message without repeating all of it."""
    if len(query) <= _QUERY_LABEL_CHARS:
        return repr(query)
    return f"{query[:_QUERY_LABEL_CHARS].rstrip()!r}... ({len(query)} chars)"


def _rank_urls(
    client: httpx.Client,
    queries: list[str],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Search every derived query and fuse the orderings into one.

    Returns `(urls, titles, errors)`: a failed query is an error in the list, not
    an exception, so no single search decides the phase. `should_stop` is asked
    before each query and through every retry wait.
    """
    rankings: list[list[str]] = []
    titles: dict[str, str] = {}
    errors: list[str] = []
    name = "searxng" if SEARXNG_URL else "duckduckgo"
    backend = _search_searxng if SEARXNG_URL else _search_duckduckgo
    circuit = _search_circuit(name)

    for position, query in enumerate(queries):
        if should_stop is not None and should_stop():
            break
        label = _query_label(query)
        unsent = len(queries) - position - 1
        skipped = f" The other {unsent} search(es) were not sent." if unsent else ""

        def search(query: str = query) -> list[dict[str, str]]:
            return circuit.call(backend, query, WEB_RESULTS_PER_QUERY, client)

        try:
            hits = call_with_retry(
                search,
                max_attempts=WEB_ATTEMPTS,
                min_wait=WEB_RETRY_WAIT_SECONDS,
                max_wait=WEB_RETRY_WAIT_SECONDS,
                retry_if=_quick_transient,
                give_up=should_stop,
                name=f"web-search:{name}",
            )
        except _SearchBlocked as exc:
            # The rest of the fan-out would be refused too, and every request
            # sent into a block extends it: the backend is stood down for its
            # cooldown.
            circuit.trip(str(exc))
            errors.append(f"{label}: {exc}{skipped}")
            break
        except CircuitOpenError as exc:
            errors.append(f"{label}: {exc}.{skipped}")
            break
        except httpx.HTTPStatusError as exc:
            # Where a redirect points is the explanation: DuckDuckGo refuses an
            # over-long query with a `302` to its error page.
            where = exc.response.headers.get("location")
            errors.append(
                f"{label}: HTTP {exc.response.status_code}" + (f" -> {where}" if where else "")
            )
            continue
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        rankings.append([hit["url"] for hit in hits])
        for hit in hits:
            titles.setdefault(hit["url"], hit["title"])

    # The fusion `search` uses for its dense and lexical halves.
    return reciprocal_rank_fusion(*rankings), titles, errors


def _fetch_and_extract(
    url: str,
    client: httpx.Client,
    deadline: float,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Fetch one page and read it. Returns a result dict, never raises."""
    if should_stop is not None and should_stop():
        return {"url": url, "error": "skipped: the run was stopped"}
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        return {"url": url, "error": "skipped: the research budget ran out first"}

    def get() -> httpx.Response:
        response = client.get(
            url,
            timeout=max(0.5, min(WEB_FETCH_TIMEOUT_SECONDS, deadline - time.monotonic())),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return response

    try:
        response = call_with_retry(
            get,
            max_attempts=WEB_ATTEMPTS,
            min_wait=WEB_RETRY_WAIT_SECONDS,
            max_wait=WEB_RETRY_WAIT_SECONDS,
            retry_if=_quick_transient,
            give_up=lambda: deadline - time.monotonic() <= 1.0
            or (should_stop is not None and should_stop()),
            name="web-fetch",
        )
        # A PDF or an image would extract as whatever its bytes decode to,
        # filed under a real URL; there is no extractor for them here.
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return {"url": url, "error": f"not text ({content_type or 'unknown type'})"}
        page = extract(response.text)
    except httpx.HTTPStatusError as exc:
        return {"url": url, "error": f"HTTP {exc.response.status_code}"}
    except Exception as exc:
        # Not only httpx's own errors: a malformed URL raises `InvalidURL`, an
        # Exception of its own, and a page the reader chokes on raises what it
        # raises. Any of them escaping would take every other fetch with it.
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "url": url,
        "title": page.title,
        "content": page.text,
        "words": page.words,
        "fell_back": page.fell_back,
    }


def select_pages(goal: str, pages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Score fetched pages against the goal and keep the ones that earn a place.

    The engine's rank decided what was fetched; whether a page's text answers the
    goal is judged after reading it. The bar is a fraction of the best page's
    score, and a page scoring zero shares no term with the goal and is dropped.
    """
    scorable = [page for page in pages if page.get("content")]
    if not scorable:
        return []

    index = BM25Index([page["url"] for page in scorable], [page["content"] for page in scorable])
    scores = index.score(goal, [page["url"] for page in scorable])
    best = max(scores.values(), default=0.0)
    if best <= 0.0:
        # Nothing shares a term with the goal; keeping the top few anyway would
        # be the engine's ranking under this function's name.
        return []

    floor = best * WEB_SELECT_RATIO
    for page in scorable:
        page["score"] = scores[page["url"]]
    ranked = sorted(scorable, key=lambda page: -page["score"])
    return [page for page in ranked if page["score"] >= floor][:keep]


def _document_name_for(url: str) -> str:
    """A deterministic filename for a URL.

    Deterministic, so researching a topic again overwrites the earlier copy of a
    page in place rather than adding a near-duplicate. The digest separates URLs
    whose slugs coincide.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    readable = re.sub(r"[^a-z0-9]+", "-", f"{host}{parsed.path}".lower()).strip("-")
    slug = readable[:_MAX_SLUG_CHARS].strip("-") or "page"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}.md"


def _render_document(title: str, url: str, query: str, body: str) -> str:
    """Wrap an extracted page in a header naming where it came from.

    The provenance goes in the text, not the metadata: metadata is the two keys a
    rebuild writes for every document, while text survives the rebuild and comes
    back attached to whatever chunk matched.
    """
    retrieved = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"# {title or url}\n\n"
        f"- Source: {url}\n"
        f"- Retrieved: {retrieved}\n"
        f"- Researched for: {query}\n\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


def _fit_to_index_limit(document: str, url: str) -> str:
    """Trim an over-long page to the walk's limit, and say on the page that it was.

    An upload is refused instead, since the operator can split it; nobody stands
    behind a scrape. The tail goes, since a page's opening is its subject.
    """
    if len(document) <= MAX_INDEXABLE_BYTES:
        return document
    note = (
        f"\n\n---\n\n*Truncated to {MAX_INDEXABLE_BYTES:,} characters, the same "
        f"limit a reindex applies. The rest of this page was not embedded; read "
        f"it at {url}.*\n"
    )
    return document[: MAX_INDEXABLE_BYTES - len(note)].rstrip() + note


def search_web(
    goal: str,
    fetch_limit: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Search, fetch and read: everything before deciding what to keep.

    Never raises for a network failure. "The web said nothing", "we never asked"
    and "we asked and it broke" are distinct `source` values, each with a `note`.
    `should_stop` is asked before each query and each fetch, and through their
    retry waits; `stopped` says it ended the phase early.

    Returns:
        `{"goal", "queries", "pages", "source", "note", "errors", "stopped"}`,
        `source` one of `duckduckgo`, `searxng`, `disabled` or `error`.
    """
    if not WEB_SEARCH_ENABLED:
        return {
            "goal": goal,
            "queries": [],
            "pages": [],
            "source": "disabled",
            "note": WEB_SEARCH_DISABLED_NOTE,
            "errors": [],
            "stopped": False,
        }

    backend = "searxng" if SEARXNG_URL else "duckduckgo"
    deadline = time.monotonic() + WEB_RESEARCH_BUDGET_SECONDS
    queries = expand_queries(goal)

    def stopped() -> bool:
        return should_stop is not None and should_stop()

    with httpx.Client() as client:
        urls, titles, errors = _rank_urls(client, queries, should_stop)
        if not urls or stopped():
            note = (
                "No derived search succeeded; nothing was researched online. "
                + "; ".join(errors)
                if errors
                else "The search returned no results for this goal."
            )
            return {
                "goal": goal,
                "queries": queries,
                "pages": [],
                "source": "error" if errors and not urls else backend,
                "note": "Stopped before anything was fetched." if stopped() else note,
                "errors": errors,
                "stopped": stopped(),
            }

        wanted = urls[: max(1, fetch_limit or WEB_FETCH_LIMIT)]
        # Fetching is the phase's latency, all of it waiting on sockets; the
        # pool is joined, never abandoned.
        with ThreadPoolExecutor(max_workers=max(1, WEB_FETCH_WORKERS)) as pool:
            fetched = list(pool.map(
                lambda url: _fetch_and_extract(url, client, deadline, should_stop), wanted
            ))

    pages = []
    for page in fetched:
        if page.get("error"):
            errors.append(f"{page['url']}: {page['error']}")
        elif page.get("content"):
            page.setdefault("title", "")
            page["title"] = page["title"] or titles.get(page["url"], "")
            pages.append(page)
        else:
            errors.append(f"{page['url']}: nothing readable on the page")

    return {
        "goal": goal,
        "queries": queries,
        "pages": pages,
        "source": backend,
        "note": None if pages else "No page fetched for this goal had readable text.",
        "errors": errors,
        "stopped": stopped(),
    }


def store_web_document(
    kb: GraphRAGKnowledgeBase,
    result: dict[str, Any],
    query: str,
    root: str = ".",
) -> dict[str, Any]:
    """Write one fetched page under `WEB_RESEARCH_DIR` and embed it, in that order.

    Raises:
        ValueError: the result has no URL or no text -- a filename in the corpus
            that answers queries with nothing.
    """
    url = (result.get("url") or "").strip()
    body = (result.get("content") or "").strip()
    if not url:
        raise ValueError("A web result with no URL cannot be filed under one.")
    if not body:
        raise ValueError(f"{url} came back with no text to embed.")

    document = _fit_to_index_limit(
        _render_document(result.get("title") or "", url, query, body), url
    )

    directory = Path(root) / WEB_RESEARCH_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _document_name_for(url)
    # Read before the write, so "replaced" is a fact.
    replaced = path.exists()
    path.write_text(document, encoding="utf-8")

    chunks = kb.add_document(str(path), document, _document_metadata(path))
    return {
        "path": str(path),
        "url": url,
        "title": result.get("title") or "",
        "replaced": replaced,
        "chunks": chunks,
        "characters": len(document),
        "score": round(float(result.get("score", 0.0)), 3),
    }


def research_online(
    open_kb: Callable[[], GraphRAGKnowledgeBase],
    goal: str,
    root: str = ".",
    keep: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Search the web for a goal and embed what earns a place. The pre-run phase.

    Takes a *factory* for the corpus and calls it only once a page has earned a
    place, so a phase that stores nothing leaves no empty corpus behind. A failure
    storing one page is that page's alone and lands in `failed`, with `saved`
    saying whether its text reached disk for the next rebuild to embed.
    `considered` and `documents` are both reported: twelve pages read and none
    kept is a working phase on a goal the web has nothing to say about.

    `should_stop` is asked before each page and handed to the embedder, which
    asks it between batches and through every retry wait, so a stopped run waits
    for one batch rather than for every page left; `stopped` says it happened.
    """
    started = time.monotonic()
    answer = search_web(goal, should_stop=should_stop)
    # A phase stopped while searching keeps nothing it fetched: the operator
    # asked for it to end, not for what it had so far.
    selected = (
        [] if answer.get("stopped")
        else select_pages(goal, answer["pages"], keep or WEB_SEARCH_MAX_RESULTS)
    )

    stored: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    stopped = bool(answer.get("stopped"))
    kb: GraphRAGKnowledgeBase | None = None
    try:
        for page in selected:
            if should_stop is not None and should_stop():
                stopped = True
                break
            # Any failure is this page's, not the phase's.
            try:
                if kb is None:
                    kb = open_kb()
                    kb._should_stop = should_stop
                stored.append(store_web_document(kb, page, goal, root))
            except EmbeddingStopped:
                stopped = True
                break
            except Exception as exc:
                url = page.get("url", "")
                saved = bool(url) and (
                    Path(root) / WEB_RESEARCH_DIR / _document_name_for(url)
                ).is_file()
                failed.append({"url": url, "error": str(exc), "saved": saved})
    finally:
        if kb is not None:
            kb._should_stop = None

    return {
        "goal": goal,
        "queries": answer["queries"],
        "source": answer["source"],
        "note": answer["note"],
        "considered": len(answer["pages"]),
        "documents": len(stored),
        "chunks": sum(item["chunks"] for item in stored),
        "stored": stored,
        "failed": failed,
        "stopped": stopped,
        "errors": answer["errors"],
        "elapsed_s": round(time.monotonic() - started, 1),
    }
