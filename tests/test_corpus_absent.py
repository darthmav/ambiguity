"""The corpus is built when someone asks for one, and not otherwise.

There used to be no distinction between opening a corpus and creating one:
`GraphRAGKnowledgeBase.__init__` makes the store on disk, and every read went
through it. So a corpus appeared as a side effect of asking whether there was
a corpus. `serve` started a preload thread at import, and the console polls
`rag_stats` every five seconds, which together meant that starting the server
was enough to leave a store behind -- one that then reported itself as a
knowledge base to everything that looked, having never been indexed.

What is pinned here is the split that fixes it: `open_knowledge_base` reads,
`get_knowledge_base` builds, and only an act of indexing goes through the
second one. Three acts qualify -- the Reindex button, a page the online
research phase keeps, and a run that finds nothing to search and indexes the
project before it starts -- and each is pinned to the creating door, to the
walk it must consult first, and to leaving no store behind when it turns out
to have nothing to put in one. The tests avoid the embedding model throughout,
which is also the claim of the last one in the file.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import serve
from langgraph_agent import graphrag_server
from langgraph_agent.control import EmbedderActivity
from langgraph_agent.graphrag_server import (
    GraphRAGKnowledgeBase,
    corpus_exists,
    corpus_state,
    open_knowledge_base,
)

# The walk's mechanics; and "absent" is an answer only a reachable database gives.
pytestmark = pytest.mark.usefixtures("whole_root_walk", "postgres")


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """A machine where nobody has ever indexed.

    The singleton is cleared as well as the directory: it is process-global,
    so a corpus another test built would answer for this one and hide exactly
    the failure under test.
    """
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def store(tmp_path, monkeypatch) -> str:
    """A path for a *real* store, absolute and unique to this test.

    Chroma caches its system by the path string it is handed, so two corpora
    in one process both opened as the relative "./knowledge" quietly share the
    first one's store -- and the second test then sees a directory that was
    never created under its own tmp_path.
    """
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    return str(tmp_path / "knowledge")


def _touched(path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


# ---------------------------------------------------------------------------
# the two doors
# ---------------------------------------------------------------------------


def test_no_corpus_reads_as_absent_and_leaves_nothing_behind(nowhere):
    assert corpus_exists(str(nowhere)) is False
    assert corpus_state(str(nowhere))[0] == "absent"
    assert open_knowledge_base(str(nowhere)) is None
    assert _touched(nowhere) == []


def test_indexing_is_the_act_that_creates_the_store(store):
    """`get_knowledge_base` is the one door allowed to bring a corpus into being."""
    assert corpus_exists(store) is False

    kb = graphrag_server.get_knowledge_base(store)

    assert corpus_exists(store) is True
    # And once it exists, the reading door finds it.
    assert open_knowledge_base(store) is kb


def test_a_cleared_corpus_is_empty_not_absent(store):
    """Emptying in place leaves a real store, and the two must not be confused.

    Both answer a search with nothing, and only one of them means "press
    Reindex" -- the other means the operator already did and then cleared it.
    """
    kb = graphrag_server.get_knowledge_base(store)
    kb.clear()

    assert corpus_state(store)[0] == "empty"
    assert corpus_exists(store) is True


# ---------------------------------------------------------------------------
# every read the console makes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, params",
    [
        ("status", {}),
        ("rag_stats", {}),
        ("list_documents", {}),
        ("query_graph", {"node_id": "serve.py"}),
        ("search_documents", {"query": "the planner"}),
    ],
)
def test_a_read_never_creates_a_corpus(nowhere, method, params):
    """The console polls these on a timer. A read that creates is a corpus
    nobody asked for, arriving seconds after the server starts."""
    serve.RPC_METHODS[method](params)

    assert _touched(nowhere) == []
    assert serve.kb is None


def test_a_search_with_nothing_to_search_leaves_the_embedder_idle(nowhere, monkeypatch):
    """The embedder card lights off a meter marked where the model works, so
    "there is no corpus" has to come back without a load or an encode -- or the
    light would call a search that embedded nothing busy."""
    meter = EmbedderActivity()
    monkeypatch.setattr(graphrag_server, "EMBEDDER_ACTIVITY", meter)

    serve.RPC_METHODS["search_documents"]({"query": "the planner"})

    assert meter.snapshot()["started"] == 0


def test_the_reads_report_the_absence_rather_than_zeros(nowhere):
    """Four zeros read as a knowledge base that happens to be empty."""
    assert serve.rpc_status({})["corpus"] == "absent"

    stats = serve.rpc_rag_stats({})
    assert stats["corpus"] == "absent"
    assert stats["total_chunks"] == 0
    # It has to say how a corpus comes to exist, and there are exactly two
    # ways. Naming a script or a button here is how this note went stale once
    # already: both were removed and the note went on recommending them.
    note = stats["note"].lower()
    assert "run" in note and "console" in note
    assert "reindex.py" not in note and "reindex project" not in note

    assert serve.rpc_list_documents({})["corpus"] == "absent"
    assert serve.rpc_query_graph({"node_id": "serve.py"})["corpus"] == "absent"


def test_a_search_with_no_corpus_returns_no_hits_and_says_why(nowhere):
    """Not one fabricated row. The Builder reads this field."""
    result = serve.rpc_search_documents({"query": "the planner"})

    assert result["results"] == []
    assert result["source"] == "no_corpus"
    assert result["note"]


# ---------------------------------------------------------------------------
# the writes
# ---------------------------------------------------------------------------


def test_export_and_clear_refuse_rather_than_create_one_to_act_on(nowhere):
    """Creating a store in order to empty it is the opposite of the ask."""
    with pytest.raises(ValueError, match="no corpus to export"):
        serve.rpc_export_corpus({})
    with pytest.raises(ValueError, match="no corpus to clear"):
        serve.rpc_clear_corpus({})

    assert _touched(nowhere) == []


# ---------------------------------------------------------------------------
# the Researcher's own door
# ---------------------------------------------------------------------------


def test_the_researcher_is_told_there_is_no_corpus(nowhere):
    """It used to be handed `[GraphRAG not indexed]` at score 0.0 -- a made-up
    retrieval hit sitting in the field real ones arrive in."""
    from langgraph_agent.mcp_client import MCPClient

    client = MCPClient()
    result = client.call_tool("search_knowledge_graph", {"query": "the planner"})

    assert result["results"] == []
    assert result["source"] == "no_corpus"
    assert _touched(nowhere) == []


def test_a_graph_query_with_no_corpus_is_empty_not_stubbed(nowhere):
    from langgraph_agent.mcp_client import MCPClient

    client = MCPClient()
    result = client.call_tool("query_knowledge_graph", {"entity": "Planner"})

    assert result["neighbors"] == []
    assert result["source"] == "no_corpus"
    assert _touched(nowhere) == []


# ---------------------------------------------------------------------------
# the run builds the corpus it is about to search
# ---------------------------------------------------------------------------


def _fake_index(calls, report=None):
    """Stand in for `index_corpus_files`, recording the corpus it was given."""
    def index(kb, **kwargs):
        calls.append(kb)
        return dict(report or {"indexed": 2, "embedded": 2, "reused": 0,
                               "dropped": 0, "skipped": 0, "errors": [],
                               "total_chunks": 7})
    return index


def test_a_first_run_builds_the_corpus_it_is_about_to_search(nowhere, monkeypatch):
    """A fresh install has no corpus, and a run is a request to search one.

    Nothing used to build one except the operator naming it -- an install
    step, two indexing scripts, a Reindex button -- and missing all of them
    cost nothing visible: the search answers `no_corpus`, the Researcher falls
    through to its own model, and the run reports itself finished. All four are
    gone; this is what replaced them.
    """
    (nowhere / "notes.md").write_text("the planner interprets goals", encoding="utf-8")
    built = object()  # stands in for a corpus that did not exist a moment ago
    indexed = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: built)
    monkeypatch.setattr(serve, "index_corpus_files", _fake_index(indexed))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert indexed == [built]  # through the creating door, once
    assert any("[Corpus]" in message and "indexed before the run" in message
               for message in result["messages"])
    # The fake built it and nothing else did: no second path constructs a
    # store, which is what `get_knowledge_base` being the only door means.
    assert "knowledge" not in _touched(nowhere)


def test_an_emptied_corpus_is_rebuilt_too_not_only_a_missing_one(nowhere, monkeypatch):
    """`empty` and `absent` are different states and the same problem here.

    A store an install step created and then failed to fill answers a search
    exactly as badly as no store at all, so the condition is "nothing to
    search", not "no directory".
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "corpus_state", lambda: ("empty", "m"))
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())
    monkeypatch.setattr(serve, "index_corpus_files", _fake_index(indexed))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert len(indexed) == 1
    assert any("was empty" in message for message in result["messages"])


def test_a_run_brings_a_corpus_that_is_behind_the_archive_up_to_date(nowhere, monkeypatch):
    """Drift used to be the operator's job, and the failure is silent.

    A corpus that holds documents reports `indexed` with every counter
    non-zero and self-consistent while retrieval quietly stops finding the
    project -- measured here at 8 documents against a walk of 103. The rebuild
    is affordable on every run because it keeps the vectors of documents whose
    text has not changed.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", "m"))
    monkeypatch.setattr(serve, "index_corpus_files", _fake_index(
        indexed, {"indexed": 9, "embedded": 2, "reused": 7, "dropped": 1,
                  "skipped": 0, "errors": []}))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert len(indexed) == 1
    line = next(m for m in result["messages"] if m.startswith("[Corpus]"))
    assert "2 document(s) re-read" in line and "1 no longer in the archive" in line


def test_a_rebuild_that_changed_nothing_still_says_so(nowhere, monkeypatch):
    """This asserted the silence until 2026-09-12, and the silence misread.

    The argument for saying nothing was that the phase runs on every run and
    usually does nothing, so a line about it is a line nobody reads. What it
    actually produced is a console that mentions the corpus once, on the run
    that builds it, and never again -- and a rebuild that re-embeds nothing
    takes ~0.1s and loads no model, so there is no other sign it happened. The
    operator's reading was that no embedding occurs at all, which is the
    failure `#run-live` already fixed once: a run that did the work and a run
    that skipped it must not look identical.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", "m"))
    monkeypatch.setattr(serve, "index_corpus_files", _fake_index(
        indexed, {"indexed": 9, "embedded": 0, "reused": 9, "dropped": 0,
                  "skipped": 0, "errors": []}))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert len(indexed) == 1  # it still ran; it just had nothing to do
    line = next(m for m in result["messages"] if m.startswith("[Corpus]"))
    assert "already matched the archive" in line
    assert "9 document(s)" in line


def test_indexing_switched_off_is_the_one_state_that_says_nothing(nowhere, monkeypatch):
    """`REBUILD_CORPUS=0` is the operator's own machine-level choice,
    reported by the console header on every poll rather than by a run. It is
    also the one state where the phase genuinely did not run, so the line above
    would be false here -- nothing was checked and nothing matched."""
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    indexed = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", False)
    monkeypatch.setattr(serve, "corpus_state", lambda: ("indexed", "m"))
    monkeypatch.setattr(serve, "index_corpus_files", _fake_index(indexed, {}))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert not indexed
    assert not [m for m in result["messages"] if m.startswith("[Corpus]")]


def test_the_corpus_is_built_before_the_online_phase_and_not_after(nowhere, monkeypatch):
    """The order is load-bearing, not stylistic.

    The online phase embeds the pages it keeps, so a corpus holding nothing
    but fetched pages counts above zero and reads `indexed` -- and the
    project's own files would then never be indexed at all, on this run or any
    later one.
    """
    (nowhere / "notes.md").write_text("x", encoding="utf-8")
    order: list[str] = []
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: object())

    def index(kb, **kwargs):
        order.append("corpus")
        return {"indexed": 1, "skipped": 0, "errors": [], "total_chunks": 1}

    def research(factory, goal, **kwargs):
        order.append("web")
        return {"source": "duckduckgo", "documents": 0, "considered": 0}

    monkeypatch.setattr(serve, "index_corpus_files", index)
    monkeypatch.setattr(serve, "research_online", research)

    serve.rpc_run_goal({"goal": "Do a thing", "research_web": True})

    assert order == ["corpus", "web"]


# ---------------------------------------------------------------------------
# the online research phase
# ---------------------------------------------------------------------------


def test_a_run_leaves_no_corpus_behind_when_there_is_nothing_to_put_in_one(
    nowhere, monkeypatch
):
    """The first run on a fresh machine must not invent an empty knowledge base.

    Two phases could, and both are on here. `_research_online_before_the_run`
    is allowed to build one -- storing a page is indexing -- but it used to
    resolve that door on the way in, as the argument to `research_online`,
    before it knew whether the phase would run at all: the phase is off in
    every test and on any machine without `WEB_SEARCH_ENABLED`, so a run that
    never fetched a page still created an empty store. The corpus bootstrap
    has the same shape one caller along, and counts the walk before it opens
    the door for exactly that reason -- here the walk offers nothing, so
    nothing is built.

    `rag_stats` would otherwise read `empty` from then on where the truth is
    `absent`, and only one of those two means "press Reindex".
    """
    monkeypatch.setattr(serve, "REBUILD_CORPUS", True)

    serve.rpc_run_goal({"goal": "Do a thing"})

    assert "knowledge" not in _touched(nowhere)
    assert corpus_state(str(nowhere / "knowledge"))[0] == "absent"


def test_the_phase_opens_the_door_only_for_a_page_it_keeps(monkeypatch):
    """Nothing kept, nothing built; and the corpus is built once, not per page.

    Asserted on the factory rather than on the disk so the two halves are one
    test: the same call that must not happen for an empty selection must happen
    exactly once for a non-empty one.
    """
    from langgraph_agent import web_research

    opened = []
    monkeypatch.setattr(web_research, "search_web", lambda goal: {
        "goal": goal, "pages": [], "queries": [goal], "source": "duckduckgo",
        "note": "", "errors": []})
    monkeypatch.setattr(web_research, "store_web_document",
                        lambda kb, page, goal, root: {"chunks": 1, "path": page["url"]})

    def factory():
        opened.append(1)
        # Not a bare object(): the phase hands the corpus its stop to embed under.
        return SimpleNamespace()

    monkeypatch.setattr(web_research, "select_pages", lambda *a, **k: [])
    report = web_research.research_online(factory, "a goal", ".")
    assert report["documents"] == 0
    assert opened == []  # considered nothing, so built nothing

    pages = [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]
    monkeypatch.setattr(web_research, "select_pages", lambda *a, **k: pages)
    report = web_research.research_online(factory, "a goal", ".")
    assert report["documents"] == 2
    assert opened == [1]  # one corpus for both pages, not one each


# ---------------------------------------------------------------------------
# the local model
# ---------------------------------------------------------------------------


def test_opening_a_corpus_does_not_load_the_embedding_model(store):
    """The one thing that runs on this machine loads when something embeds.

    It used to load in `__init__`, so every header poll paid for it. Counting
    documents, listing them, drawing the graph and exporting all embed nothing,
    so none of them may reach for the model -- the daemon's or otherwise.
    """
    kb = GraphRAGKnowledgeBase(store)

    assert kb._embedder is None
    kb.stats()
    kb.list_documents()
    kb.export_corpus()
    assert kb._embedder is None


def test_importing_the_server_never_imports_an_in_process_embedder():
    """`serve` is imported by this suite already; the assertion is that its
    import dragged no embedding stack in.

    This is now a guarantee rather than a laziness check: there is no
    in-process embedder at all -- the daemon embeds -- so neither package may
    appear, whatever the test order. `transformers` is allowed the moment
    something chunks (its tokenizer is the chunker's), but importing `serve`
    alone must not have paid for it.
    """
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules

    assert "serve" in sys.modules
    assert "transformers" not in sys.modules


def test_the_suite_never_sees_this_checkouts_own_corpus(monkeypatch):
    """conftest hides the developer's `knowledge/` from every test.

    Left visible, each graph test's Researcher searched it for real -- opening
    Chroma, embedding the plan through the Ollama daemon, and evicting whatever
    the developer's own console had loaded on the cards. This fails the moment
    that isolation is lost, which is otherwise invisible: CI has no corpus to
    find, and a developer's run only gets slower and less repeatable.
    """
    from pathlib import Path

    from langgraph_agent import config

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)

    assert graphrag_server.open_knowledge_base() is None
    assert graphrag_server.corpus_state()[0] == "absent"
    assert config.free_the_cards_for("hf.co/some/seat:Q4") == []
