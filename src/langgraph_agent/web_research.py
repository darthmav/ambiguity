"""Online research: the web half of the corpus.

Nothing in this process could open a socket before this module. The corpus was
a function of what sat on disk in the checkout, so a run could only ever be as
informed as the checkout was: a goal naming a library this project has never
vendored had no source of truth to retrieve, retrieval came back under
the relevance floor, and the Researcher's seat answered from whatever
the model happened to remember. That is the one path where the seat's model is
the variable, and it is the path a goal about anything external always took.

What this adds is a **pre-run** phase, and the ordering is the design rather
than a convenience. Documents are searched, written under `WEB_RESEARCH_DIR`
and embedded *before* the graph starts streaming, for two reasons that are both
rules written elsewhere in this package:

* `_refuse_while_a_run_is_in_flight` (serve.py) refuses every corpus write
  while a run is live, because a corpus changing underneath the Researcher
  manufactures an absence no seat can detect -- a rebuild half-done returns
  whatever fraction of itself has been re-added, which reads as
  `no_relevant_knowledge` and routes the run around a gap that was created out
  from under it. Embedding before the Architect's opening pass is not a way
  around that guard. It is the only ordering that respects it.
* A document embedded but never written to disk survives exactly until the next
  reindex, which then deletes it *silently*, in a pass reporting success and a
  file count that looks right. So the file is written first and embedded
  second -- the same ordering, for the same reason, as
  `store_uploaded_document` -- and `WEB_RESEARCH_DIR` has to stay inside
  `PROJECT_INDEX_PATTERNS` and out of `PROJECT_INDEX_EXCLUDES`.

**Nothing here costs money and nothing here needs an account.** The result list
comes from DuckDuckGo's keyless HTML endpoint, or from a SearxNG instance the
operator hosts. That is the only part of the pipeline that cannot be built
here -- a web index is not something a project makes for itself -- and it is
deliberately the *only* thing taken from outside. Ranking, extraction and
selection are all ours.

The pipeline has four stages, and the last two are the reason it is ours:

1. **Fan out.** One search on a raw goal asks a search engine to be good at a
   sentence. `expand_queries` derives several from the goal using this
   project's own `tokenize` -- the identifier-aware one, so a goal naming
   `BUILDER_DEADLINE_SECONDS` also searches `builder deadline seconds`.
2. **Fuse the rankings.** Each query returns its own ordering, merged with
   `reciprocal_rank_fusion` -- the same function `search` uses to merge the
   dense and lexical halves. A page several queries agree on outranks one that
   won a single query, which is what makes the fan-out worth doing rather than
   just three times the fetching.
3. **Extract with our own reader** (`html_text`), which reports how much of
   each page it kept and whether it had to fall back, so a thin page is a
   reading rather than a mystery.
4. **Gate on our own score.** The engine's rank decides what is *fetched*; it
   does not decide what is *embedded*. Every extracted page is scored with
   `BM25Index` against the goal, and a page that cannot clear the bar is
   thrown away unread. This is the answer to the problem this phase would
   otherwise create: web documents compete with the checkout's own files at
   retrieval time, under a relevance floor calibrated on a corpus containing
   none of them, so the cheapest protection is not letting a page in unless it
   is about the goal by the same measure that will later retrieve it.

The gate is a **ratio of the best page's score, never an absolute number**.
That is not a preference; `lexical.py` records the reason. A BM25 score is
unbounded and corpus-relative, so a constant threshold is a hyperparameter
tuned on one set of pages and wrong on the next -- the same objection that
stopped `search` from fusing dense and lexical scores by weighted sum.
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

# `_document_metadata` is imported rather than restated. It is module-private,
# and reaching for it is still right: it defines what a document is filed
# under, and a second spelling of that is two documents for one file, one of
# them orphaned in the graph at the next rebuild. `store_uploaded_document`
# shares it with `index_project_files` for the same reason.
from langgraph_agent.graphrag_server import (
    MAX_INDEXABLE_BYTES,
    WEB_RESEARCH_DIR,
    _document_metadata,
)
from langgraph_agent.html_text import extract
from langgraph_agent.lexical import BM25Index, reciprocal_rank_fusion, tokenize

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase

# `WEB_RESEARCH_DIR` is defined in `graphrag_server` and re-exported here: it
# is inside the walk so a reindex re-reads these pages, and `add_document` has
# to recognise one on sight to skip entity extraction for it.

# Keyless, no account, no quota. DuckDuckGo's HTML endpoint is the default
# because it answers a plain POST from anywhere; `SEARXNG_URL` points at an
# instance the operator runs, for a machine that would rather not depend on
# someone else's endpoint at all. There is deliberately no third option: every
# other general web search API charges per call.
DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"
SEARXNG_URL = os.environ.get("SEARXNG_URL", "").strip().rstrip("/")

# An off switch, so a machine with no network does not spend the phase's whole
# budget discovering that on every run.
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", "1").strip() not in {"0", "false", "no"}

# How many derived queries the goal fans out into, and how many results each
# returns. The product bounds how many URLs are considered; `WEB_FETCH_LIMIT`
# bounds how many are actually fetched, after fusion has ordered them.
WEB_SEARCH_QUERIES = int(os.environ.get("WEB_SEARCH_QUERIES", "3"))
WEB_RESULTS_PER_QUERY = int(os.environ.get("WEB_RESULTS_PER_QUERY", "10"))
WEB_FETCH_LIMIT = int(os.environ.get("WEB_FETCH_LIMIT", "12"))

# The longest query sent to a search engine. An engine does not read a long
# query badly -- it refuses it. DuckDuckGo answers `302` to its own
# `/50x.html?e=3` page, which httpx does not follow and `raise_for_status`
# reports as a bare status. Measured on 2026-09-10: 496 characters accepted,
# 592 and every length above it redirected, and a 287-character query came back
# with a full page of results. The run that found this sent the verbatim goal
# at 1,227 characters and its term list at 816 and 840, so every search the
# phase made was refused before a single page was read. The default sits inside
# the range measured to *return results*, not at the edge of the range measured
# to be *accepted*: whether results still come back between 287 and 496 is
# unmeasured, because the probe that would have said tripped the bot check.
WEB_QUERY_MAX_CHARS = int(os.environ.get("WEB_QUERY_MAX_CHARS", "250"))

# How much of a query an error message quotes. Errors are reported per query,
# and quoting each whole made the feed line for that same run ~2,900
# characters: the long goal three times over, burying the one word that said
# what had happened.
_QUERY_LABEL_CHARS = 60

# How many pages may actually enter the corpus for one goal. Every one of these
# competes with the checkout's own documents at retrieval time, so this is a
# ceiling on how far a single run can shift what the Researcher sees.
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "8"))

# Keep a page whose BM25 score against the goal is at least this fraction of
# the best page's. A ratio rather than an absolute -- see the module docstring.
# A page scoring zero shares no term with the goal and is dropped whatever this
# says, which is the only floor here that is not relative.
WEB_SELECT_RATIO = float(os.environ.get("WEB_SELECT_RATIO", "0.25"))

WEB_SEARCH_TIMEOUT_SECONDS = float(os.environ.get("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
WEB_FETCH_TIMEOUT_SECONDS = float(os.environ.get("WEB_FETCH_TIMEOUT_SECONDS", "20"))
# The whole phase, since it sits between the operator pressing Run and the
# Architect opening. Nothing downstream bounds it: `RUN_BUDGET_SECONDS` starts
# at the graph.
WEB_RESEARCH_BUDGET_SECONDS = float(os.environ.get("WEB_RESEARCH_BUDGET_SECONDS", "120"))
WEB_FETCH_WORKERS = int(os.environ.get("WEB_FETCH_WORKERS", "4"))

# Sent on every request. A blank or scripted agent is what most endpoints
# rate-limit first, and the point of naming the project is that an operator
# reading their own logs can tell what this traffic is.
USER_AGENT = os.environ.get(
    "WEB_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) langgraph-agent/1.0 (+research)",
)

WEB_SEARCH_DISABLED_NOTE = (
    "WEB_SEARCH_ENABLED is off, so nothing was researched online and nothing "
    "was added to the corpus. The run continues against the corpus as it "
    "stands."
)

# Only what a search box should not carry. Deliberately tiny: this strips the
# scaffolding of an instruction ("please add a function that...") and nothing
# domain-bearing, because a stopword list that reaches into the vocabulary is
# how a query loses the term it needed. `ENTITY_STOPWORDS` in graphrag_server
# is the same shape of decision, made by hand for the same reason.
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "to", "of", "in", "on", "for", "with", "is",
        "are", "be", "that", "this", "it", "as", "at", "by", "or", "from",
        "please", "can", "you", "we", "i", "should", "would", "make", "add",
        "use", "using", "how", "do", "does", "need", "want", "let", "lets",
        "our", "my", "into", "so", "if", "then", "than", "but", "not", "all",
    }
)

# Appended to the term-only query to aim the remaining searches at material
# worth embedding rather than at discussion of it. Ordered: the phase takes as
# many as `WEB_SEARCH_QUERIES` leaves room for.
_QUERY_INTENTS = ("documentation reference", "example implementation", "best practices")

# Longest readable part of a filename before the disambiguating digest. A
# filename is what `search` reports as the source and what the console prints,
# so it stays legible rather than becoming a bare hash.
_MAX_SLUG_CHARS = 60


def web_search_available() -> bool:
    """True when an online research phase would actually reach the network."""
    return WEB_SEARCH_ENABLED


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _fit_query(text: str, limit: int) -> str:
    """The longest leading run of whole words in `text` that fits in `limit`.

    Leading, because the terms a goal opens with are the ones it is about --
    the ordering `expand_queries` already relies on. A first word longer than
    the limit by itself is cut rather than dropped: an empty query is not a
    shorter search, it is no search, and it would be reported as the web
    having nothing to say.
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
    """As many whole leading sentences of `text` as fit in `limit`, else "".

    Whole sentences or nothing. The verbatim query is there to carry the
    phrasing a person chose, and a sentence cut off mid-clause is not that --
    the term queries already cover a goal this function has to leave out.
    """
    kept = ""
    for sentence in _SENTENCE_BREAK.split(text):
        candidate = f"{kept} {sentence}" if kept else sentence
        if len(candidate) > limit:
            break
        kept = candidate
    return kept


def expand_queries(goal: str, limit: int | None = None) -> list[str]:
    """Derive several search queries from one goal.

    The verbatim goal comes first when it fits -- it is the only query carrying
    the phrasing a person chose, and an engine's own understanding of a
    sentence is sometimes better than anything derived from it. The rest are
    built from the goal's content terms, using this project's `tokenize` rather
    than a plain split, so an identifier in the goal contributes its pieces as
    well as itself: a goal naming `BUILDER_DEADLINE_SECONDS` searches
    `builder deadline seconds` too, and a search engine has seen the second and
    never the first.

    Every query fits `WEB_QUERY_MAX_CHARS`, because an over-long query is not
    searched worse, it is refused outright. A goal too long to send whole is
    represented by its leading whole sentences, or not at all when even the
    first will not fit, and the term queries keep the goal's leading terms.

    Duplicate-free and order-preserving: a one-word goal collapses its variants
    into the verbatim query rather than searching the same string three times.
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

    # `dict.fromkeys` keeps first-appearance order, which matters: the terms a
    # goal opens with are the ones it is about, and they are what survives the
    # length cap.
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

    Parsed rather than regexed because attribute order is not ours to rely on,
    and this is the one piece of the pipeline whose input is someone else's
    markup and can therefore change without notice. It fails to an empty list,
    which `search_web` reports as a search that found nothing -- distinguishable
    from one that never ran.
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
    """Recover the real URL from DuckDuckGo's `/l/?uddg=` wrapper.

    The endpoint returns a direct href sometimes and a redirect other times,
    with nothing in the response saying which. Storing the wrapper would file
    the document under duckduckgo.com -- a provenance header naming the wrong
    site, and a filename that collides with every other wrapped result.
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

    Kept apart from an HTTP error because the right response differs. A failed
    query says nothing about the next one; a block covers every query from this
    address, and each further request only prolongs it.
    """


# DuckDuckGo's bot check arrives as `202` carrying a challenge form and no
# results, which passes `raise_for_status` and parses to an empty list -- so a
# block was reported as a web with nothing to say on the goal, the one outcome
# `search_web` exists to keep apart from "we asked and it broke". Recognised by
# its markup rather than its status, so a 202 that does carry results is read.
_DUCKDUCKGO_CHALLENGE_MARKERS = ("anomaly-modal", "/anomaly.js")


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


def _rank_urls(goal: str, client: httpx.Client, queries: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Search every derived query and fuse the orderings into one.

    Returns `(urls, titles, errors)`. A query that fails is an error in the
    list and not an exception: the fan-out exists so that no single search
    decides the phase, and that has to include a search that broke.
    """
    rankings: list[list[str]] = []
    titles: dict[str, str] = {}
    errors: list[str] = []
    backend = _search_searxng if SEARXNG_URL else _search_duckduckgo

    for position, query in enumerate(queries):
        label = _query_label(query)
        try:
            hits = backend(query, WEB_RESULTS_PER_QUERY, client)
        except _SearchBlocked as exc:
            # The rest of the fan-out would be refused the same way, and every
            # request sent into a block extends it.
            unsent = len(queries) - position - 1
            skipped = f" The other {unsent} search(es) were not sent." if unsent else ""
            errors.append(f"{label}: {exc}{skipped}")
            break
        except httpx.HTTPStatusError as exc:
            # Where a redirect points is the explanation. DuckDuckGo refuses an
            # over-long query with `302` to its `/50x.html` page, and the bare
            # status reads as a mystery rather than as a refusal.
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

    # The same fusion `search` uses to merge its dense and lexical halves. A
    # page several derived queries agree on outranks one that topped a single
    # query, which is the entire return on fanning out.
    return reciprocal_rank_fusion(*rankings), titles, errors


def _fetch_and_extract(url: str, client: httpx.Client, deadline: float) -> dict[str, Any]:
    """Fetch one page and read it. Returns a result dict, never raises."""
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        return {"url": url, "error": "skipped: the research budget ran out first"}
    try:
        response = client.get(
            url,
            timeout=min(WEB_FETCH_TIMEOUT_SECONDS, remaining),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        # A PDF or an image reaching the extractor would produce whatever its
        # bytes decode to, filed under a real URL -- the fabricated source
        # `store_uploaded_document` refuses a `.pdf` to prevent. There is no
        # text extractor for those here, so they are declined by name.
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return {"url": url, "error": f"not text ({content_type or 'unknown type'})"}
        page = extract(response.text)
    except httpx.HTTPStatusError as exc:
        return {"url": url, "error": f"HTTP {exc.response.status_code}"}
    except httpx.HTTPError as exc:
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

    The search engine's rank decided what was *fetched*. It does not decide
    what is embedded, because a result list is ordered by what a general engine
    thinks the query means, and what matters here is whether the page's actual
    text answers the goal -- which can only be judged after reading it.

    The bar is a fraction of the best page's score rather than a constant.
    `lexical.py` explains why at length: a BM25 score is unbounded and
    corpus-relative, so a fixed number is a hyperparameter fitted to one set of
    pages and meaningless on the next. A page scoring zero shares no term with
    the goal at all and is dropped regardless -- the one absolute here, and it
    is absolute because zero means "no evidence", not "a little evidence".
    """
    scorable = [page for page in pages if page.get("content")]
    if not scorable:
        return []

    index = BM25Index([page["url"] for page in scorable], [page["content"] for page in scorable])
    scores = index.score(goal, [page["url"] for page in scorable])
    best = max(scores.values(), default=0.0)
    if best <= 0.0:
        # Nothing shares a term with the goal. Returning the top few anyway
        # would be the search engine's ranking wearing this function's name.
        return []

    floor = best * WEB_SELECT_RATIO
    for page in scorable:
        page["score"] = scores[page["url"]]
    ranked = sorted(scorable, key=lambda page: -page["score"])
    return [page for page in ranked if page["score"] >= floor][:keep]


def _document_name_for(url: str) -> str:
    """A deterministic filename for a URL.

    Deterministic is the load-bearing word. Researching the same topic twice
    re-finds the same pages, and `add_document` deletes a document's previous
    chunks before the new ones land -- so a stable name overwrites the earlier
    copy in place. A name carrying a timestamp or a counter would instead grow
    the corpus by a near-identical document per run, each one competing with
    the others for the same query, and the stale copies would keep answering
    until someone noticed the store had quietly doubled.

    The digest is not decoration: two URLs differing only past
    `_MAX_SLUG_CHARS`, or only in a query string, produce the same slug, and
    the second would silently overwrite the first.
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

    The provenance is written into the *document text* rather than into the
    metadata, and that is deliberate. Metadata here is `_document_metadata`'s
    two keys, shared with `index_project_files`; a third key added only on this
    path would be gone the moment a reindex re-read the same file from the
    walk, so the corpus would describe one document two ways depending on how
    it was last written. Text on disk survives the rebuild, gets embedded with
    the passage, and comes back attached to whatever chunk matched -- which is
    what lets a Researcher finding say which page it came from.
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

    `store_uploaded_document` *refuses* an oversized file, which is right for an
    upload: the operator is standing there and can split it. Nothing is standing
    behind a scrape, and refusing would drop the source entirely over a length
    nobody chose. So it is trimmed instead -- but visibly, because a silent
    truncation at an embedding boundary is precisely the bug that left 91.5% of
    this corpus unreachable while every counter still read correctly.

    The tail goes rather than the head: a page's opening is its subject, and the
    trailing matter of a web page is navigation and comments.
    """
    if len(document) <= MAX_INDEXABLE_BYTES:
        return document
    note = (
        f"\n\n---\n\n*Truncated to {MAX_INDEXABLE_BYTES:,} characters, the same "
        f"limit a reindex applies. The rest of this page was not embedded; read "
        f"it at {url}.*\n"
    )
    return document[: MAX_INDEXABLE_BYTES - len(note)].rstrip() + note


def search_web(goal: str, fetch_limit: int | None = None) -> dict[str, Any]:
    """Search, fetch and read. Everything up to the point of deciding what to keep.

    Never raises for a network failure, and never returns a page it did not
    read. A caller has to be able to tell "the web said nothing" from "we never
    asked", and both from "we asked and it broke", so each is a distinct
    `source` with a `note` saying which -- rather than an empty list that reads
    identically in all three cases. That is the same contract `search` has: the
    empty answer carries `NO_CORPUS_NOTE` instead of a made-up row scored 0.0.

    Returns:
        `{"goal", "queries", "pages", "source", "note", "errors"}`. `source` is
        one of `duckduckgo`, `searxng`, `disabled` or `error`.
    """
    if not WEB_SEARCH_ENABLED:
        return {
            "goal": goal,
            "queries": [],
            "pages": [],
            "source": "disabled",
            "note": WEB_SEARCH_DISABLED_NOTE,
            "errors": [],
        }

    backend = "searxng" if SEARXNG_URL else "duckduckgo"
    deadline = time.monotonic() + WEB_RESEARCH_BUDGET_SECONDS
    queries = expand_queries(goal)

    with httpx.Client() as client:
        urls, titles, errors = _rank_urls(goal, client, queries)
        if not urls:
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
                "source": "error" if errors else backend,
                "note": note,
                "errors": errors,
            }

        wanted = urls[: max(1, fetch_limit or WEB_FETCH_LIMIT)]
        # Fetching is the phase's whole latency, and it is all waiting on
        # sockets. The pool is joined by the context manager rather than
        # abandoned -- the warning in CLAUDE.md is about a worker outliving the
        # thing that started it, which is a deadline's problem and not this
        # one's.
        with ThreadPoolExecutor(max_workers=max(1, WEB_FETCH_WORKERS)) as pool:
            fetched = list(pool.map(lambda url: _fetch_and_extract(url, client, deadline), wanted))

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
    }


def store_web_document(
    kb: GraphRAGKnowledgeBase,
    result: dict[str, Any],
    query: str,
    root: str = ".",
) -> dict[str, Any]:
    """Write one fetched page under `WEB_RESEARCH_DIR` and embed it, in that order.

    The file first, always. See the module docstring for why: the corpus is a
    function of what is on disk, and a document that exists only in the store
    is deleted by the next reindex without anything reporting it.

    Raises:
        ValueError: the result has no URL or no text to embed. Refused rather
            than stored, because a document with an empty body is a filename in
            the corpus that answers queries with nothing -- the shape of a
            fabricated source.
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
    # Read before the write, so "replaced" is a fact rather than a guess --
    # re-researching a topic is the ordinary way this happens.
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
) -> dict[str, Any]:
    """Search the web for a goal and embed what earns a place. The pre-run phase.

    Takes a *factory* rather than a corpus, and calls it only once a page has
    earned a place. The caller passes the creating door (`_kb_for_indexing`),
    which is right -- storing a page is indexing, and a goal researched against
    a machine with no corpus should leave one behind holding what it found.
    Resolving that door on the way in instead left one behind holding
    **nothing**: the phase is switched off in every test and on any machine
    without `WEB_SEARCH_ENABLED`, and a phase that never ran still built an
    empty store, which then reported itself as a knowledge base to everything
    that looked afterwards. `rag_stats` read `empty` where the truth was
    `absent`, and those two call for different things from the operator -- only
    one of them means "press Reindex". The same held for a goal the web has
    nothing to say about: twelve pages considered, none kept, a corpus created
    to hold them.

    Returns a report rather than raising, and the report distinguishes a phase
    that found nothing from one that never ran and one that broke -- see
    `search_web`. A per-document failure is collected into `failed` and does
    not abandon the rest: one page that came back empty should cost that page,
    not the other seven.

    `considered` and `documents` are both reported because they answer
    different questions: how much was read, and how much was good enough to
    keep. A phase that fetched twelve pages and embedded none is working
    correctly on a goal the web has nothing to say about, and it must not look
    like a phase that failed.
    """
    started = time.monotonic()
    answer = search_web(goal)
    selected = select_pages(goal, answer["pages"], keep or WEB_SEARCH_MAX_RESULTS)

    stored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    # Resolved on the first page that is actually going to be stored, and
    # reused for the rest: the factory creates the corpus, so calling it before
    # the loop creates one for a phase that stores nothing.
    kb: GraphRAGKnowledgeBase | None = None
    for page in selected:
        # Any failure is this page's, not the phase's. The embedder reports its
        # own as RuntimeError ("Ollama could not embed ..."), which the old
        # `(ValueError, OSError)` let through: the first page the daemon
        # refused abandoned every page after it, and the report of the pages
        # already embedded went with it -- the caller saw only "the phase
        # failed", with documents sitting in the corpus that nothing named.
        try:
            if kb is None:
                kb = open_kb()
            stored.append(store_web_document(kb, page, goal, root))
        except Exception as exc:
            failed.append({"url": page.get("url", ""), "error": str(exc)})

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
        "errors": answer["errors"],
        "elapsed_s": round(time.monotonic() - started, 1),
    }
