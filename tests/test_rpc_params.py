"""RPC parameters are typed, bounded, and refused by name.

Every method used to take its parameters through a bare `int()` or `bool()`.
`top_k="abc"` came back as Python's own "invalid literal for int() with base
10", a negative `top_k` as an empty result the console showed as "no results",
and the string "false" as True -- which on `research_web` or `expect_failures`
is the opposite of what the caller asked for. And a request body that was not a
JSON object raised before the handler's error handling, dropping the connection.
"""

from __future__ import annotations

import io
import json

import pytest

import serve


def _no_corpus_touch(monkeypatch):
    """Parameters are checked before anything is opened."""
    def touched():
        raise AssertionError("opened the corpus before refusing the parameters")
    monkeypatch.setattr(serve, "_open_kb", touched)


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["abc", -1, 0, serve.MAX_TOP_K + 1, 2.5, True, [5], float("nan")])
def test_top_k_is_a_bounded_whole_number(value, monkeypatch):
    _no_corpus_touch(monkeypatch)
    with pytest.raises(ValueError, match="top_k must be a whole number"):
        serve.rpc_search_documents({"query": "x", "top_k": value})


def test_a_whole_number_sent_as_text_is_accepted(monkeypatch):
    seen = {}

    class _Corpus:
        def search(self, query, top_k):
            seen["top_k"] = top_k
            return []

    monkeypatch.setattr(serve, "_open_kb", lambda: _Corpus())
    serve.rpc_search_documents({"query": "x", "top_k": "7"})
    assert seen["top_k"] == 7


@pytest.mark.parametrize("params,name", [
    ({"node_id": "x", "max_depth": -1}, "max_depth"),
    ({"node_id": "x", "max_depth": serve.MAX_GRAPH_DEPTH + 1}, "max_depth"),
    ({"node_id": "x", "min_degree": 0}, "min_degree"),
    ({"node_id": "x", "split": "no"}, "split"),
    ({"node_id": ["x"]}, "node_id"),
])
def test_graph_parameters_are_checked(params, name, monkeypatch):
    _no_corpus_touch(monkeypatch)
    with pytest.raises(ValueError, match=name):
        serve.rpc_query_graph(params)


def test_the_overview_checks_its_parameters(monkeypatch):
    _no_corpus_touch(monkeypatch)
    with pytest.raises(ValueError, match="min_degree"):
        serve.rpc_graph_overview({"min_degree": "lots"})
    with pytest.raises(ValueError, match="include_isolated"):
        serve.rpc_graph_overview({"include_isolated": "yes"})


@pytest.mark.parametrize("params,name", [
    ({"containment": 1.5}, "containment"),
    ({"name_similarity": -0.1}, "name_similarity"),
    ({"limit": 0}, "limit"),
])
def test_duplicate_scan_thresholds_are_fractions(params, name, monkeypatch):
    _no_corpus_touch(monkeypatch)
    with pytest.raises(ValueError, match=name):
        serve.rpc_duplicate_entities(params)


def test_analysis_limits_are_checked(monkeypatch):
    _no_corpus_touch(monkeypatch)
    with pytest.raises(ValueError, match="limit"):
        serve.rpc_bottleneck({"limit": -5})
    with pytest.raises(ValueError, match="k must be"):
        serve.rpc_topics({"k": 1})


# ---------------------------------------------------------------------------
# flags and strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["discuss_only", "research_web", "expect_failures"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_run_flags_must_be_json_booleans(flag, value):
    """`bool("false")` is True, and these flags change what a run may do."""
    with pytest.raises(ValueError, match=flag):
        serve.rpc_run_goal({"goal": "g", flag: value})
    assert serve._run_progress["running"] is False


def test_shutdown_s_flag_must_be_a_boolean():
    with pytest.raises(ValueError, match="stop_first"):
        serve.rpc_shutdown({"stop_first": "false"})
    assert not serve._shutdown_requested.is_set()


def test_seat_names_must_be_strings():
    with pytest.raises(ValueError, match="provider"):
        serve.rpc_set_seat({"agent": "planner", "provider": 7, "model": "m"})


@pytest.mark.parametrize("provider,model", [
    ("ollama", "gemma4:cloud"),              # a model the console used to offer
    ("ollama", "qwen3-embedding:latest"),    # pulled, but it cannot chat
    ("anthropic", "claude-opus-5"),
    ("anthropic", "kimi-k3:cloud"),          # an offered model under the wrong provider
])
def test_a_seat_takes_only_a_model_the_console_offers(provider, model):
    """The dropdown is the whole list, so a stale tab cannot seat anything else."""
    from langgraph_agent.config import get_agent_model_info

    before = get_agent_model_info("planner")
    with pytest.raises(ValueError, match="not a seat model the console offers"):
        serve.rpc_set_seat({"agent": "planner", "provider": provider, "model": model})
    assert get_agent_model_info("planner") == before


def test_the_seat_dropdowns_query_the_daemon_for_local_models(monkeypatch):
    """The dropdowns are populated from AGENT_LLM_OPTIONS (cloud models) plus
    locally downloaded models from the daemon.

    Cloud models (tags ending in :cloud) work via the daemon proxy and don't
    need to be pulled locally. Local models are queried from the daemon and
    filtered for completion capability. Embedding models (no completion
    capability) are filtered out.
    """
    # Mock the daemon to return specific local models
    def mock_list_ollama_models():
        return ["qwen3.8:latest", "dolphin-2.9.1-yi-1.5-9b:Q4_K_M", "qwen3-embedding:latest"]

    def mock_ollama_model_capabilities(model):
        if model == "qwen3-embedding:latest":
            return ["embedding"]  # no completion
        return ["completion", "tools", "thinking"]

    # The functions are imported at module level in serve.py, so patch there
    monkeypatch.setattr(serve, "list_ollama_models", mock_list_ollama_models)
    monkeypatch.setattr(serve, "ollama_model_capabilities", mock_ollama_model_capabilities)

    result = serve.rpc_llm_options({})
    options = result["options"]

    # Should have cloud models from AGENT_LLM_OPTIONS (kimi-k3:cloud)
    # plus 2 local models (embedding model filtered out)
    models = [o["model"] for o in options]
    # Cloud models from AGENT_LLM_OPTIONS
    assert "kimi-k3:cloud" in models
    # Local models from daemon
    assert "qwen3.8:latest" in models
    assert "dolphin-2.9.1-yi-1.5-9b:Q4_K_M" in models
    # Embedding model filtered out
    assert "qwen3-embedding:latest" not in models

    # Cloud models should have "Ollama Cloud" group, local models "Ollama (local)"
    # Check for both :cloud and -cloud suffixes like the implementation does
    cloud_models = [o for o in options if o["model"].endswith((":cloud", "-cloud"))]
    local_models = [o for o in options if not o["model"].endswith((":cloud", "-cloud"))]
    for o in cloud_models:
        assert o["provider"] == "ollama"
        assert o["group"] == "Ollama Cloud"
    for o in local_models:
        assert o["provider"] == "ollama"
        assert o["group"] == "Ollama (local)"


# ---------------------------------------------------------------------------
# the request body
# ---------------------------------------------------------------------------


class _Socket:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    def flush(self):
        pass


def _post(body: bytes, headers: dict[str, str] | None = None) -> dict:
    handler = serve.Handler.__new__(serve.Handler)
    handler.path = "/rpc"
    handler.headers = {"Content-Length": str(len(body)), **(headers or {})}
    handler.rfile = io.BytesIO(body)
    handler.wfile = _Socket()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /rpc HTTP/1.1"
    handler.command = "POST"
    handler.do_POST()
    return json.loads(handler.wfile.data.split(b"\r\n\r\n", 1)[1])


@pytest.mark.parametrize("body", [b"[]", b'"status"', b"42", b"null"])
def test_a_body_that_is_not_an_object_gets_an_error_envelope(body):
    assert "JSON object" in _post(body)["error"]["message"]


def test_params_that_are_not_an_object_are_refused():
    reply = _post(json.dumps({"method": "status", "params": [1, 2]}).encode())
    assert "params must be a JSON object" in reply["error"]["message"]


def test_bad_json_still_says_bad_json():
    assert _post(b"{nope")["error"]["message"] == "bad JSON"


def test_an_oversized_body_is_refused_unread():
    reply = _post(b"{}", headers={"Content-Length": str(serve.MAX_REQUEST_BYTES + 1)})
    assert "limited" in reply["error"]["message"]


def test_a_non_numeric_content_length_is_refused():
    reply = _post(b"{}", headers={"Content-Length": "lots"})
    assert "Content-Length" in reply["error"]["message"]


def test_a_well_formed_call_still_works():
    reply = _post(json.dumps({"method": "list_seats", "params": {}}).encode())
    assert "seats" in reply["result"]
