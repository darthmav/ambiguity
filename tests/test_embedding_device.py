"""The one embedder, and the light that says when it is working.

`OllamaEmbedder` is the whole of the embedding stack this process owns: it
sends passages to the local daemon in measured batches and hands one numpy
vector per passage back. Placement -- which card, how much of the model -- is
the daemon's, and is only read back and reported. Every test here answers with
a faked `urllib` and a faked `ollama_cpu_share`, so none of them touches a
daemon, a card or a real model.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pytest

import serve
from langgraph_agent import config
from langgraph_agent import graphrag_server as gs
from langgraph_agent.control import EmbedderActivity
from langgraph_agent.graphrag_server import (
    EMBEDDING_MODEL_NAME,
    GraphRAGKnowledgeBase,
    OllamaEmbedder,
    embedding_device_status,
)


class _Reply(io.BytesIO):
    """A context manager the `with urlopen(...) as response` shape needs."""

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _Daemon:
    """`/api/embed`, one request at a time, with a fixed vector length."""

    def __init__(self, dims: int = 3) -> None:
        self.dims = dims
        self.requests: list[dict[str, Any]] = []

    def urlopen(self, request: Any, timeout: float | None = None) -> _Reply:
        payload = json.loads(request.data)
        self.requests.append(payload)
        embeddings = [[0.0] * self.dims] * len(payload["input"])
        return _Reply(json.dumps({"embeddings": embeddings}).encode())


@pytest.fixture
def daemon(monkeypatch) -> _Daemon:
    import urllib.request

    fake = _Daemon()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(config, "ollama_cpu_share", lambda model: None)
    return fake


def _kb() -> GraphRAGKnowledgeBase:
    """No store: embedding alone touches none of it."""
    return object.__new__(GraphRAGKnowledgeBase)


# ---------------------------------------------------------------------------
# one vector per passage, in measured batches
# ---------------------------------------------------------------------------


def test_every_batch_carries_the_model_and_the_measured_options(daemon):
    """8 is not a default; it is the batch OLLAMA_EMBED_OPTIONS was measured at."""
    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)

    vectors = embedder.encode([f"passage {i}" for i in range(10)], batch_size=2)

    assert vectors.shape == (10, 3)
    assert len(daemon.requests) == 5
    for i, request in enumerate(daemon.requests):
        assert request["model"] == EMBEDDING_MODEL_NAME
        assert request["input"] == [f"passage {2 * i}", f"passage {2 * i + 1}"]
        assert request["options"] == gs.OLLAMA_EMBED_OPTIONS


def test_a_single_string_earns_a_single_vector(daemon):
    vector = OllamaEmbedder(EMBEDDING_MODEL_NAME).encode("one passage")

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (3,)


def test_a_stopcheck_ends_the_encode_between_batches(daemon):
    """A stopped run waits for at most one batch, not for the rest of a document."""
    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(gs.EmbeddingStopped):
        embedder.encode([f"passage {i}" for i in range(10)], batch_size=2, should_stop=stop)

    assert len(daemon.requests) == 1


def test_a_short_reply_is_an_error_not_a_short_answer(monkeypatch):
    """A vector count that mismatches cannot be zipped away; it is a broken embed."""
    import urllib.request

    def short(request: Any, timeout: float | None = None) -> _Reply:
        return _Reply(json.dumps({"embeddings": [[0.0] * 3]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", short)

    with pytest.raises(RuntimeError, match="2 passages"):
        OllamaEmbedder(EMBEDDING_MODEL_NAME).encode(["a", "b"])


def test_a_refused_request_is_named_not_raised_bare(monkeypatch):
    """The daemon's own words reach the log: a missing tag reads as its 404 body."""
    import urllib.error
    import urllib.request

    def refuse(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", None, io.BytesIO(b'model "gone" not found')
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    with pytest.raises(RuntimeError, match='model "gone" not found'):
        OllamaEmbedder(EMBEDDING_MODEL_NAME).encode("a passage")


def _flaky(monkeypatch, failures: int, code: int = 500) -> dict[str, int]:
    """`/api/embed` answering `code` for its first `failures` requests, then vectors."""
    import urllib.error
    import urllib.request

    calls = {"n": 0}

    def urlopen(request: Any, timeout: float | None = None) -> _Reply:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise urllib.error.HTTPError(
                request.full_url, code, "Error", None,
                io.BytesIO(b"llama-server process has terminated: exit status 1"),
            )
        return _Daemon().urlopen(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(config, "ollama_cpu_share", lambda model: None)
    monkeypatch.setattr(gs, "OLLAMA_EMBED_RETRY_SECONDS", 0.0)
    return calls


def test_a_failed_model_load_is_retried_until_it_loads(monkeypatch):
    """The 2026-09-18 startup: three 500s from a load short of GPU memory, then a load."""
    calls = _flaky(monkeypatch, failures=3)

    vectors = OllamaEmbedder(EMBEDDING_MODEL_NAME).encode(["a", "b"])

    assert vectors.shape == (2, 3)
    assert calls["n"] == 4


def test_a_load_that_never_succeeds_says_how_often_it_was_tried(monkeypatch):
    calls = _flaky(monkeypatch, failures=100)

    with pytest.raises(RuntimeError, match=r"after \d+ attempts.*llama-server"):
        OllamaEmbedder(EMBEDDING_MODEL_NAME).encode("a passage")

    assert calls["n"] == gs.OLLAMA_EMBED_LOAD_RETRIES + 1


def test_a_refusal_is_not_retried(monkeypatch):
    """A 4xx is the daemon saying no to the request; asking again changes nothing."""
    calls = _flaky(monkeypatch, failures=100, code=404)

    with pytest.raises(RuntimeError) as refused:
        OllamaEmbedder(EMBEDDING_MODEL_NAME).encode("a passage")

    assert calls["n"] == 1
    assert not isinstance(refused.value, gs.EmbedderLoadFailed)
    assert not gs.EMBEDDER_LOAD.is_open, "a refused request is the daemon answering"


def test_every_chunk_fits_one_pass_of_the_load():
    """What makes the batch a free saving rather than a change of vectors.

    Measured 2026-10-02: a chunk embedded in one pass came back bit-identical at
    a batch of 256 as at 512. A batch shorter than a chunk (254 tokens, plus the
    special tokens the daemon adds) would split it into two passes and move its
    vector, and every corpus built before would be searched with others.
    """
    options = gs.OLLAMA_EMBED_OPTIONS

    assert options["num_batch"] >= gs.CHUNK_MAX_TOKENS + 2
    assert options["num_ctx"] >= options["num_batch"]


def test_a_load_that_outlasts_the_schedule_opens_the_embedder_circuit(monkeypatch):
    """On 2026-10-02 every page of a research phase spent its own whole schedule.

    Once one schedule has failed, what holds the cards is not a model being
    evicted, so the next embed is refused without asking the daemon.
    """
    calls = _flaky(monkeypatch, failures=100)
    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)

    with pytest.raises(gs.EmbedderLoadFailed, match=r"after \d+ attempts.*llama-server"):
        embedder.encode("the first page")
    assert gs.EMBEDDER_LOAD.is_open
    tried = calls["n"]

    with pytest.raises(gs.CircuitOpenError, match="embedder-load"):
        embedder.encode("the second page")
    assert calls["n"] == tried, "a refused embed must not reach the daemon"


def test_the_trial_after_the_cooldown_is_one_attempt(monkeypatch):
    """A load that already outlasted a schedule is tested once, then trusted again."""
    calls = _flaky(monkeypatch, failures=gs.OLLAMA_EMBED_LOAD_RETRIES + 2)
    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    with pytest.raises(gs.EmbedderLoadFailed):
        embedder.encode("a passage")
    monkeypatch.setattr(gs.EMBEDDER_LOAD._breaker, "reset_timeout", 0)

    with pytest.raises(gs.EmbedderLoadFailed):
        embedder.encode("a passage")
    assert calls["n"] == gs.OLLAMA_EMBED_LOAD_RETRIES + 2
    assert gs.EMBEDDER_LOAD.is_open, "a failed trial opens it again"

    embedder.encode("a passage")
    assert not gs.EMBEDDER_LOAD.is_open, "a trial that loads closes it"


def test_a_stop_during_the_schedule_does_not_open_the_circuit(monkeypatch):
    """A stopped schedule did not show that the model cannot load."""
    calls = _flaky(monkeypatch, failures=100)

    with pytest.raises(gs.EmbeddingStopped):
        OllamaEmbedder(EMBEDDING_MODEL_NAME).encode(
            "a passage", should_stop=lambda: calls["n"] >= 1
        )

    assert calls["n"] == 1
    assert not gs.EMBEDDER_LOAD.is_open


def test_an_empty_store_is_searched_without_loading_the_model(monkeypatch):
    """A cleared corpus, or one whose first pages all failed to embed, holds nothing.

    Embedding the query anyway loaded the model and evicted the seat holding the
    cards -- and it was how two run tests reached the live daemon on every suite.
    """
    from types import SimpleNamespace

    kb = _kb()
    kb.collection = SimpleNamespace(count=lambda: 0)

    def no_embed(texts: Any) -> Any:
        raise AssertionError("an empty store must not load the embedder")

    monkeypatch.setattr(kb, "_encode", no_embed)

    assert kb.search("what does the corpus say?") == []


def test_the_tokenizer_keeps_transformers_advice_out_of_the_log(monkeypatch):
    """Without torch, importing transformers announces it, which nothing here needs."""
    import os
    import sys
    from types import SimpleNamespace

    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)
    seen: list[str | None] = []

    def from_pretrained(name: str, **kwargs: Any) -> object:
        seen.append(os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"))
        return object()

    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "transformers", fake)

    assert OllamaEmbedder(EMBEDDING_MODEL_NAME).tokenizer is not None
    assert seen == ["1"]


# ---------------------------------------------------------------------------
# placement is the daemon's; this process only reads it back
# ---------------------------------------------------------------------------


def test_a_split_model_is_read_back_and_said_out_loud(monkeypatch):
    """The reply is identical either way, so a split onto the CPU shows nowhere else."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _Daemon().urlopen)
    monkeypatch.setattr(config, "ollama_cpu_share", lambda model: 0.5)

    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    embedder.encode("a passage")

    assert embedder.cpu_share == 0.5
    note = embedder.placement_note or ""
    assert "50%" in note
    assert "CPU" in note


def test_a_whole_model_carries_no_note(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _Daemon().urlopen)
    monkeypatch.setattr(config, "ollama_cpu_share", lambda model: 0.0)

    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    embedder.encode("a passage")

    assert embedder.placement_note is None


def test_the_status_reports_ollama_without_loading_anything(daemon, monkeypatch):
    """The header's device line is a report, not a load."""
    kb = _kb()
    monkeypatch.setattr(gs, "_kb_instance", kb)

    assert embedding_device_status() == {
        "configured": "ollama",
        "active": None,
        "note": None,
        "cpu_share": None,
    }
    assert daemon.requests == []

    kb._encode(["a passage"])
    status = embedding_device_status()
    assert status["active"] == "ollama"


# ---------------------------------------------------------------------------
# the embedder's light: whether the model is working
# ---------------------------------------------------------------------------


def test_work_that_falls_between_two_looks_is_still_on_the_meter():
    """A query embeds in tens of milliseconds, far inside the gap between two polls."""
    meter = EmbedderActivity()

    with meter.working():
        assert meter.snapshot()["busy"] is True

    after = meter.snapshot()
    assert after["busy"] is False
    assert after["started"] == 1
    assert after["ended_ago"] is not None


def test_overlapping_work_keeps_the_embedder_busy_until_the_last_of_it_ends():
    """A search from the console can land while a run is indexing."""
    meter = EmbedderActivity()

    with meter.working():  # the index
        with meter.working():  # the search, inside it
            pass
        assert meter.snapshot()["busy"] is True

    assert meter.snapshot()["busy"] is False
    assert meter.snapshot()["started"] == 2


def test_work_that_raises_still_puts_the_light_out():
    """A daemon that refuses an embed must not leave the embedder lit for good."""
    meter = EmbedderActivity()

    with pytest.raises(RuntimeError):
        with meter.working():
            raise RuntimeError("Ollama could not embed")

    assert meter.snapshot()["busy"] is False


class _WatchedEmbedder:
    """An embedder whose encode reports whether the light is on."""

    placement_note = None

    def __init__(self, meter: EmbedderActivity) -> None:
        self.meter = meter
        self.tok: Any = None

    def encode(self, texts: str | list[str], **kwargs: Any) -> Any:
        busy = self.meter.snapshot()["busy"]
        self.busy_while_encoding = busy
        return np.zeros((len(texts), 3)) if isinstance(texts, list) else np.zeros(3)


def test_every_encode_is_marked_where_it_happens(monkeypatch):
    """At the one place every embedding passes, so no caller can forget to say so:
    the corpus phase, a stored page, the Planner's map and a search alike."""
    meter = EmbedderActivity()
    monkeypatch.setattr(gs, "EMBEDDER_ACTIVITY", meter)
    kb = _kb()
    kb._embedder = _WatchedEmbedder(meter)  # type: ignore[assignment]
    started = meter.snapshot()["started"]

    kb._encode(["a passage"])

    assert kb._embedder.busy_while_encoding is True
    assert meter.snapshot()["started"] > started
    assert meter.snapshot()["busy"] is False


def test_loading_the_model_counts_as_work(monkeypatch):
    """The first search after a start spends seconds here, before any encode."""
    meter = EmbedderActivity()
    monkeypatch.setattr(gs, "EMBEDDER_ACTIVITY", meter)
    real = gs.OllamaEmbedder
    busy_while_loading = []

    def watched(model: str) -> Any:
        busy_while_loading.append(meter.snapshot()["busy"])
        return real(model)

    monkeypatch.setattr(gs, "OllamaEmbedder", watched)

    _kb().embedder  # noqa: B018 - the load is what is under test

    assert busy_while_loading == [True]
    assert meter.snapshot()["busy"] is False
    assert meter.snapshot()["started"] == 1


def test_the_console_reads_the_embedder_with_or_without_a_run():
    """A run's poll carries the reading; between runs the console asks for it alone."""
    fields = {"busy", "busy_for", "started", "ended_ago"}
    assert set(serve.rpc_run_progress({})["embedding"]) == fields
    assert set(serve.rpc_embedding_activity({})) == fields
    assert serve.RPC_METHODS["embedding_activity"] is serve.rpc_embedding_activity
    # Polled several times a second while a search is in flight.
    assert "embedding_activity" in serve.QUIET_METHODS
