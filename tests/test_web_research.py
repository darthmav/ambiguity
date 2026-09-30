"""Tests for the online research phase.

The claim under test is not "an HTTP request was made". It is three things:

* a page fetched from the web becomes a document *of the corpus* — written to
  disk inside the walk, embedded under the same metadata as every other
  document, re-findable by the next reindex rather than pruned by it;
* the search engine's ranking decides what is fetched and *not* what is
  embedded, because a web document competes with the checkout's own files at
  retrieval time and has to earn that by our own measure;
* when nothing can be researched, the phase says which of the three ways it
  failed instead of returning an empty list that reads the same in all of them.

Built the way `test_uploads.py` builds its knowledge base: a fake that records
what it was asked to store, so the embedding model never loads. The network is
answered by `httpx.MockTransport`, so real request and response plumbing runs
while no socket does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import networkx as nx
import pytest

from langgraph_agent import web_research
from langgraph_agent.graphrag_server import (
    MAX_INDEXABLE_BYTES,
    PROJECT_INDEX_EXCLUDES,
    _document_metadata,
    iter_project_files,
)
from langgraph_agent.web_research import (
    WEB_RESEARCH_DIR,
    WEB_SEARCH_DISABLED_NOTE,
    _document_name_for,
    _unwrap_redirect,
    expand_queries,
    research_online,
    search_web,
    select_pages,
    store_web_document,
)

GOAL = "hybrid retrieval with BM25_RERANK for the knowledge graph"

ON_TOPIC = " ".join(["hybrid retrieval bm25 rerank knowledge graph ranking passages."] * 12)
OFF_TOPIC = " ".join(["sourdough bread proofing temperature and crumb structure."] * 12)


class _RecordingKB:
    """What the storing path uses of a knowledge base, and nothing else."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str, dict[str, Any]]] = []
        self.graph = nx.DiGraph()

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> int:
        self.added.append((doc_id, content, metadata))
        self.graph.add_node(doc_id, type="document")
        return max(1, len(content) // 100)

    def stats(self) -> dict[str, Any]:
        return {"total_documents": len(self.added), "total_chunks": len(self.added)}


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    """Undo conftest's suite-wide off switch; the network is mocked below."""
    monkeypatch.setattr(web_research, "WEB_SEARCH_ENABLED", True)


@pytest.fixture
def kb() -> _RecordingKB:
    return _RecordingKB()


def _result(url: str, title: str = "A page", content: str = ON_TOPIC) -> dict[str, Any]:
    return {"url": url, "title": title, "content": content}


def _results_html(*urls: str) -> str:
    links = "".join(
        f'<a class="result__a" href="{url}">Result for {url}</a>' for url in urls
    )
    return f"<html><body><div class='results'>{links}</div></body></html>"


def _page_html(body: str) -> str:
    return f"<html><head><title>A page</title></head><body><article><p>{body}</p></article></body></html>"


# Captured once, at import, and deliberately not re-read inside `_serve`.
# Reading `httpx.Client` at call time makes the patches *nest*: a second
# `_serve` in one test wraps the first, whose kwargs then overwrite the second's
# transport, so the earlier handler silently keeps answering.
_REAL_CLIENT = httpx.Client


def _serve(monkeypatch, handler):
    """Answer every request through MockTransport, with no socket involved."""

    def client(*args, **kwargs):  # noqa: ANN002, ANN003
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)


def _standard_web(monkeypatch, pages: dict[str, str]):
    """A search returning `pages`' URLs, and each URL serving its body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(200, text=_results_html(*pages))
        body = pages.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        return httpx.Response(
            200, text=_page_html(body), headers={"content-type": "text/html"}
        )

    _serve(monkeypatch, handler)


# ---------------------------------------------------------------------------
# Fanning out: several queries from one goal
# ---------------------------------------------------------------------------


def test_the_verbatim_goal_is_the_first_query_when_it_fits():
    """It is the only query carrying the phrasing a person actually chose."""
    assert expand_queries(GOAL)[0] == GOAL


def test_an_identifier_in_the_goal_is_also_searched_in_pieces():
    """A search engine has seen `bm25 rerank` and has never seen `BM25_RERANK`."""
    terms = expand_queries(GOAL)[1]

    assert "bm25" in terms and "rerank" in terms
    # Scaffolding words are not worth a search slot.
    assert " the " not in f" {terms} " and " with " not in f" {terms} "


def test_expansion_does_not_repeat_itself_or_overrun_the_limit():
    """A one-word goal must not search the same string three times."""
    queries = expand_queries("BM25")

    assert len(queries) == len({q.lower() for q in queries})
    assert len(expand_queries(GOAL, limit=2)) == 2
    assert expand_queries("   ") == []


# Paragraph-length, the shape of goal that found the limit: every sentence short
# enough to send, the whole far too long to, and more distinct terms than one
# query can carry.
LONG_GOAL = " ".join(
    f"Measure {topic} against the current baseline."
    for topic in (
        "chunk overlap", "dense recall", "lexical precision", "fusion depth",
        "relevance floors", "entity stopwords", "graph traversal", "eigengap gates",
        "cheeger brackets", "duplicate scans", "upload limits", "staleness verdicts",
        "reindex pruning", "export notes", "seat timeouts", "builder deadlines",
        "verification slices", "headless plotting", "terminal operators", "working directories",
        "research fanout", "page extraction", "selection ratios", "provenance headers",
    )
)


def test_no_query_is_longer_than_a_search_engine_will_take():
    """DuckDuckGo answers an over-long query with `302` to its own error page.

    Every search of the run that found this was refused: the verbatim goal went
    out at 1,227 characters and the term list at 816 and 840, against a limit
    measured between 496 and 592.
    """
    assert len(LONG_GOAL) > 1000
    queries = expand_queries(LONG_GOAL)

    assert len(queries) == web_research.WEB_SEARCH_QUERIES
    assert all(0 < len(q) <= web_research.WEB_QUERY_MAX_CHARS for q in queries)
    # Whole leading sentences, not a clause cut off wherever the limit fell.
    assert LONG_GOAL.startswith(queries[0]) and queries[0].endswith(".")
    # The terms that survive are the ones the goal opens with.
    assert queries[1].startswith("measure chunk overlap")
    assert queries[2].endswith("documentation reference")


def test_the_query_cap_sits_inside_the_range_measured_to_be_accepted():
    """Raising the cap past the measured limit would show up nowhere else.

    Every query would still be built and sent, each would come back `302`, and
    the phase would report having researched nothing -- the run this cap exists
    because of. 496 characters is the longest query measured to be accepted;
    592 the shortest measured to be refused.
    """
    assert 0 < web_research.WEB_QUERY_MAX_CHARS <= 496


def test_a_goal_with_no_sentence_short_enough_is_searched_by_its_terms():
    """Omitting the verbatim query must not leave the goal unsearched."""
    run_on = LONG_GOAL.replace(".", ",")

    queries = expand_queries(run_on)

    assert queries and all(0 < len(q) <= web_research.WEB_QUERY_MAX_CHARS for q in queries)
    assert queries[0].startswith("measure chunk overlap")


# ---------------------------------------------------------------------------
# Reading the engine's answer
# ---------------------------------------------------------------------------


def test_a_wrapped_redirect_recovers_the_real_url():
    """Storing the wrapper would file every result under duckduckgo.com."""
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide&rut=abc"

    assert _unwrap_redirect(wrapped) == "https://example.com/guide"
    assert _unwrap_redirect("https://example.com/x") == "https://example.com/x"


def test_results_are_fused_across_queries_rather_than_concatenated(monkeypatch):
    """A page several derived queries agree on is what fanning out buys."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            query = request.content.decode()
            seen.append(query)
            # Only the shared URL appears in every ranking.
            unique = f"https://example.com/{len(seen)}"
            return httpx.Response(200, text=_results_html(unique, "https://example.com/shared"))
        return httpx.Response(
            200, text=_page_html(ON_TOPIC), headers={"content-type": "text/html"}
        )

    _serve(monkeypatch, handler)
    answer = search_web(GOAL)

    assert len(seen) == web_research.WEB_SEARCH_QUERIES
    fetched = [page["url"] for page in answer["pages"]]
    assert fetched[0] == "https://example.com/shared"


# ---------------------------------------------------------------------------
# The gate: the engine ranks what is fetched, we rank what is kept
# ---------------------------------------------------------------------------


def test_a_page_sharing_no_term_with_the_goal_is_never_kept():
    """Zero is the one absolute floor: it means no evidence, not weak evidence."""
    kept = select_pages(GOAL, [_result("https://example.com/bread", content=OFF_TOPIC)], keep=8)

    assert kept == []


def test_the_gate_ranks_on_page_text_not_on_arrival_order():
    """The search engine put the off-topic page first; the gate must not care."""
    pages = [
        _result("https://example.com/bread", content=OFF_TOPIC),
        _result("https://example.com/retrieval", content=ON_TOPIC),
    ]

    kept = select_pages(GOAL, pages, keep=8)

    assert [page["url"] for page in kept] == ["https://example.com/retrieval"]


def test_the_bar_is_relative_to_the_best_page_not_an_absolute_score(monkeypatch):
    """A BM25 score is unbounded and corpus-relative, so a constant is wrong.

    Pinned by moving the ratio rather than by asserting a number: with the bar
    at zero-plus the weak page survives, and at 0.9 it does not, while nothing
    about either page changed.
    """
    weak = " ".join(["retrieval happens here."] * 12)
    pages = [_result("https://example.com/strong"), _result("https://example.com/weak", content=weak)]

    monkeypatch.setattr(web_research, "WEB_SELECT_RATIO", 0.01)
    assert len(select_pages(GOAL, [dict(p) for p in pages], keep=8)) == 2

    monkeypatch.setattr(web_research, "WEB_SELECT_RATIO", 0.9)
    assert len(select_pages(GOAL, [dict(p) for p in pages], keep=8)) == 1


def test_the_keep_limit_caps_what_reaches_the_corpus(monkeypatch):
    """Every kept page competes with the checkout's own files at retrieval."""
    monkeypatch.setattr(web_research, "WEB_SELECT_RATIO", 0.0)
    pages = [_result(f"https://example.com/{n}") for n in range(6)]

    assert len(select_pages(GOAL, pages, keep=2)) == 2


# ---------------------------------------------------------------------------
# The file is what makes the research durable
# ---------------------------------------------------------------------------


def test_a_fetched_page_is_written_before_it_is_embedded(kb, tmp_path):
    """The ordering is the whole design, so it is asserted directly.

    A document embedded and not written survives until the next reindex, which
    deletes it silently in a pass that reports success.
    """
    report = store_web_document(kb, _result("https://example.com/guide"), GOAL, str(tmp_path))

    written = Path(report["path"])
    assert written.exists()
    assert written.read_text(encoding="utf-8") == kb.added[0][1]
    assert kb.added[0][0] == str(written)


def test_the_research_directory_is_inside_the_walk(kb, tmp_path, monkeypatch):
    """Asserted through `iter_project_files`, never by reading the exclude list.

    The excludes are plain substrings, so the way this breaks is someone adding
    an entry that happens to match `research/web` — and only running the walk
    catches that. If it ever does break, online research keeps working and
    stops surviving reindexes, in silence.
    """
    store_web_document(kb, _result("https://example.com/guide"), GOAL, str(tmp_path))

    monkeypatch.chdir(tmp_path)
    walked = {str(path) for path in iter_project_files(".")}

    assert any(path.startswith(WEB_RESEARCH_DIR) for path in walked), walked


def test_the_research_directory_is_not_excluded_from_the_walk():
    """Stated separately because the exclude list is edited by hand."""
    assert not any(excluded in f"{WEB_RESEARCH_DIR}/page.md" for excluded in PROJECT_INDEX_EXCLUDES)


def test_a_stored_page_carries_the_same_metadata_as_any_other_document(kb, tmp_path):
    """Two spellings of a document's metadata is two documents for one file."""
    report = store_web_document(kb, _result("https://example.com/guide"), GOAL, str(tmp_path))

    assert kb.added[0][2] == _document_metadata(Path(report["path"]))
    assert kb.added[0][2]["type"] == "markdown"


# ---------------------------------------------------------------------------
# Naming: the same URL must land on the same document
# ---------------------------------------------------------------------------


def test_the_same_url_maps_to_the_same_filename():
    """Re-researching a topic must overwrite, never accumulate.

    `add_document` drops a document's previous chunks before the new ones land,
    so a stable name replaces the earlier copy. An unstable one would leave a
    near-identical document per run, all competing for the same query.
    """
    assert _document_name_for("https://example.com/a") == _document_name_for("https://example.com/a")


def test_urls_that_differ_only_past_the_slug_get_different_filenames():
    """Without the digest the second page silently overwrites the first."""
    long_path = "https://example.com/" + "x" * 200
    assert _document_name_for(long_path + "/one") != _document_name_for(long_path + "/two")
    assert _document_name_for("https://example.com/p?a=1") != _document_name_for("https://example.com/p?a=2")


def test_a_filename_stays_a_single_readable_component():
    """A URL's path must not become a directory tree outside the walk's reach."""
    name = _document_name_for("https://example.com/deep/nested/page.html?q=1")

    assert "/" not in name and "\\" not in name
    assert name.endswith(".md")
    assert "example-com" in name


def test_a_stored_page_replaces_rather_than_duplicates(kb, tmp_path):
    first = store_web_document(kb, _result("https://example.com/g"), GOAL, str(tmp_path))
    second = store_web_document(kb, _result("https://example.com/g"), GOAL, str(tmp_path))

    assert first["path"] == second["path"]
    assert not first["replaced"]
    assert second["replaced"]


# ---------------------------------------------------------------------------
# Provenance, and the size limit
# ---------------------------------------------------------------------------


def test_provenance_is_written_into_the_document_not_the_metadata(kb, tmp_path):
    """It has to survive a reindex, and only the file does.

    A metadata key set on this path alone would be gone the moment the walk
    re-read the same file, leaving the corpus describing one document two ways.
    """
    store_web_document(kb, _result("https://example.com/guide", "Hybrid Search"), GOAL, str(tmp_path))

    document = kb.added[0][1]
    assert "https://example.com/guide" in document
    assert "Hybrid Search" in document
    assert GOAL in document
    assert set(kb.added[0][2]) == {"path", "type"}


def test_an_oversized_page_is_trimmed_visibly_rather_than_refused(kb, tmp_path):
    """An upload is refused at this limit; a scrape has nobody to split it.

    Trimming is the right call, but a silent trim at an embedding boundary is
    the bug that left most of this corpus unreachable while every counter still
    read correctly — so the page says it happened.
    """
    huge = _result("https://example.com/big", content="word " * MAX_INDEXABLE_BYTES)

    report = store_web_document(kb, huge, GOAL, str(tmp_path))

    assert report["characters"] <= MAX_INDEXABLE_BYTES
    assert "Truncated" in kb.added[0][1]
    assert "https://example.com/big" in kb.added[0][1]


def test_a_page_with_no_text_is_refused(kb, tmp_path):
    """An empty document is a filename in the corpus that answers nothing."""
    with pytest.raises(ValueError):
        store_web_document(kb, _result("https://example.com/x", content="   "), GOAL, str(tmp_path))
    with pytest.raises(ValueError):
        store_web_document(kb, _result("", content="text"), GOAL, str(tmp_path))

    assert kb.added == []


# ---------------------------------------------------------------------------
# The ways a phase can come back empty, and telling them apart
# ---------------------------------------------------------------------------


def test_the_phase_can_be_switched_off_and_says_so(monkeypatch, kb, tmp_path):
    """"We never asked" must not read as "the web had nothing"."""
    monkeypatch.setattr(web_research, "WEB_SEARCH_ENABLED", False)

    report = research_online(lambda: kb, GOAL, str(tmp_path))

    assert report["source"] == "disabled"
    assert report["documents"] == 0
    assert report["note"] == WEB_SEARCH_DISABLED_NOTE
    assert kb.added == []
    assert not (Path(tmp_path) / WEB_RESEARCH_DIR).exists()


def test_a_failing_search_is_reported_as_an_error_not_as_an_empty_web(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(429))

    answer = search_web(GOAL)

    assert answer["source"] == "error"
    assert answer["pages"] == []
    assert any("429" in error for error in answer["errors"])


def test_a_redirected_search_says_where_it_was_sent(monkeypatch):
    """`HTTP 302` alone read as a mystery; where it pointed is the explanation.

    DuckDuckGo refuses an over-long query by redirecting to its own error page,
    and httpx does not follow a redirect it was not asked to.
    """
    _serve(monkeypatch, lambda request: httpx.Response(302, headers={"location": "/50x.html?e=3"}))

    answer = search_web(GOAL)

    assert answer["source"] == "error"
    assert answer["errors"]
    assert all("302 -> /50x.html?e=3" in error for error in answer["errors"])


# The shape of DuckDuckGo's challenge as captured on 2026-09-10: `202`, a modal
# asking the caller to prove a human, and not one result link.
_BOT_CHECK = (
    '<div class="anomaly-modal__mask"><div class="anomaly-modal__modal">'
    '<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>'
    '<form id="challenge-form" action="//duckduckgo.com/anomaly.js?sv=html"></form>'
    "</div></div>"
)


def test_a_bot_check_is_an_error_not_a_web_with_nothing_to_say(monkeypatch):
    """A block used to parse to zero results and be filed as a genuine empty web.

    It passes `raise_for_status` -- `202` is a success -- so nothing marked it
    as a failure, and the feed said the phase had read pages and kept none.
    Every request sent into a block extends it, so the fan-out stops at the
    first rather than spending the remaining queries on the same refusal.
    """
    searches: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        searches.append(request)
        return httpx.Response(202, text=_BOT_CHECK)

    _serve(monkeypatch, handler)
    answer = search_web(GOAL)

    assert answer["source"] == "error"
    assert "bot check" in answer["note"]
    assert "not sent" in answer["note"]
    assert len(searches) == 1


def test_a_202_carrying_results_is_still_read(monkeypatch):
    """The block is recognised by its markup, never by its status alone."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(202, text=_results_html("https://example.com/a"))
        return httpx.Response(
            200, text=_page_html(ON_TOPIC), headers={"content-type": "text/html"}
        )

    _serve(monkeypatch, handler)
    answer = search_web(GOAL)

    assert answer["source"] == "duckduckgo"
    assert [page["url"] for page in answer["pages"]] == ["https://example.com/a"]


def test_an_error_quotes_a_long_query_without_repeating_it(monkeypatch):
    """Quoting each failed query whole made one feed line ~2,900 characters."""
    _serve(monkeypatch, lambda request: httpx.Response(429))

    answer = search_web(LONG_GOAL)

    assert answer["errors"]
    for error in answer["errors"]:
        assert len(error) < 150
        assert "chars)" in error


def test_a_page_that_is_not_text_is_declined_by_name(monkeypatch):
    """A PDF through the extractor is a fabricated source under a real URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(200, text=_results_html("https://example.com/paper.pdf"))
        return httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"})

    _serve(monkeypatch, handler)
    answer = search_web(GOAL)

    assert answer["pages"] == []
    assert any("application/pdf" in error for error in answer["errors"])


def test_one_unreachable_page_does_not_cost_the_others(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(200, text=_results_html(
                "https://example.com/good", "https://example.com/gone"
            ))
        if str(request.url).endswith("/gone"):
            return httpx.Response(500)
        return httpx.Response(200, text=_page_html(ON_TOPIC), headers={"content-type": "text/html"})

    _serve(monkeypatch, handler)
    answer = search_web(GOAL)

    assert [page["url"] for page in answer["pages"]] == ["https://example.com/good"]
    assert any("500" in error for error in answer["errors"])


# ---------------------------------------------------------------------------
# The phase as a whole
# ---------------------------------------------------------------------------


def test_research_online_embeds_only_what_earned_a_place(monkeypatch, kb, tmp_path):
    """Read four, keep the two that are about the goal."""
    _standard_web(monkeypatch, {
        "https://example.com/retrieval": ON_TOPIC,
        "https://example.com/ranking": ON_TOPIC,
        "https://example.com/bread": OFF_TOPIC,
        "https://example.com/pastry": OFF_TOPIC,
    })

    report = research_online(lambda: kb, GOAL, str(tmp_path))

    assert report["source"] == "duckduckgo"
    assert report["considered"] == 4
    assert report["documents"] == 2
    assert {Path(item["path"]).name.split("-")[1] for item in report["stored"]} == {"com"}
    assert all("bread" not in item["url"] for item in report["stored"])
    assert report["chunks"] == sum(item["chunks"] for item in report["stored"])
    assert len(kb.added) == 2


def test_a_goal_the_web_cannot_answer_is_not_a_failed_phase(monkeypatch, kb, tmp_path):
    """Fetching pages and keeping none is correct behaviour, not an error."""
    _standard_web(monkeypatch, {"https://example.com/bread": OFF_TOPIC})

    report = research_online(lambda: kb, GOAL, str(tmp_path))

    assert report["considered"] == 1
    assert report["documents"] == 0
    assert report["failed"] == []
    assert kb.added == []


def test_a_page_the_embedder_refuses_costs_that_page_alone(monkeypatch, tmp_path):
    """The embedder fails with RuntimeError, which the per-page catch let through.

    The first refused page abandoned every page after it, and the report of
    the ones already embedded went with it.
    """
    _standard_web(monkeypatch, {
        "https://example.com/retrieval": ON_TOPIC,
        "https://example.com/ranking": ON_TOPIC,
    })

    class _RefusesTheFirst(_RecordingKB):
        def add_document(self, doc_id, content, metadata):
            if not self.added and not getattr(self, "refused", False):
                self.refused = True
                raise RuntimeError("Ollama could not embed with the model: out of memory")
            return super().add_document(doc_id, content, metadata)

    kb = _RefusesTheFirst()
    report = research_online(lambda: kb, GOAL, str(tmp_path))

    assert report["documents"] == 1
    assert len(report["failed"]) == 1
    assert "could not embed" in report["failed"][0]["error"]
