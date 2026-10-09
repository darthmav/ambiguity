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
from http import HTTPStatus

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
    ("anthropic", "claude-3-opus-20240229"), # an Anthropic model it does not offer
    ("anthropic", "kimi-k3:cloud"),          # an offered model under the wrong provider
])
def test_a_seat_takes_only_a_model_the_console_offers(provider, model, monkeypatch):
    """The dropdown is the whole list, so a stale tab cannot seat anything else."""
    from langgraph_agent.config import get_agent_model_info

    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: None)
    monkeypatch.setattr(serve, "_last_ollama_options", [])

    before = get_agent_model_info("planner")
    with pytest.raises(ValueError, match="not a seat model the console offers"):
        serve.rpc_set_seat({"agent": "planner", "provider": provider, "model": model})
    assert get_agent_model_info("planner") == before


def test_the_seat_dropdowns_are_exactly_ollama_ls(monkeypatch):
    """The Ollama options are every tag the daemon lists that can complete, and
    nothing kept in code: no Ollama tag is offered because a list here names it.

    Cloud tags appear when the daemon lists them (pulled or used), grouped apart
    from local weights. An embedder (no `completion`) is left out; a tag the
    daemon will not describe stays in.
    """
    tags = [
        "kimi-k3:cloud",
        "nemotron-3-nano:30b-cloud",
        "qwen3-embedding:latest",
        "qwen3.8:latest",
        "mystery:latest",
    ]

    def caps(model):
        if model == "qwen3-embedding:latest":
            return ["embedding"]
        if model == "mystery:latest":
            return None
        return ["completion", "tools", "thinking"]

    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: tags)
    monkeypatch.setattr(serve, "_last_ollama_options", [])
    monkeypatch.setattr(serve, "ollama_model_capabilities", caps)

    options = [o for o in serve.rpc_llm_options({})["options"] if o["provider"] == "ollama"]
    models = [o["model"] for o in options]
    assert sorted(models) == sorted(
        ["kimi-k3:cloud", "nemotron-3-nano:30b-cloud", "qwen3.8:latest", "mystery:latest"]
    )
    groups = {o["model"]: o["group"] for o in options}
    assert groups["kimi-k3:cloud"] == groups["nemotron-3-nano:30b-cloud"] == "Ollama Cloud"
    assert groups["qwen3.8:latest"] == groups["mystery:latest"] == "Ollama (local)"
    # Local weights are listed before cloud.
    assert [o["group"] for o in options] == sorted(
        (o["group"] for o in options), key=lambda g: g != "Ollama (local)"
    )


def test_a_tag_the_daemon_does_not_list_is_not_offered_and_not_seatable(monkeypatch):
    """A cloud tag that was hardcoded once is gone the moment `ollama ls` is."""
    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: ["qwen3.8:latest"])
    monkeypatch.setattr(serve, "_last_ollama_options", [])
    monkeypatch.setattr(serve, "ollama_model_capabilities", lambda m: ["completion"])

    offered = serve.rpc_llm_options({})["options"]
    assert [o["model"] for o in offered if o["provider"] == "ollama"] == ["qwen3.8:latest"]
    with pytest.raises(ValueError, match="not a seat model the console offers"):
        serve.rpc_set_seat({"agent": "planner", "provider": "ollama", "model": "kimi-k3:cloud"})


def test_anthropic_is_offered_and_says_when_it_has_no_key(monkeypatch):
    """Moving a seat to Anthropic took a `.env` edit and a restart. The
    dropdown now offers the models `.env` would give a seat, and a model that
    would run the stub for want of a key says so before it is picked."""
    from langgraph_agent import config
    from langgraph_agent.config import ANTHROPIC_SEAT_MODELS

    monkeypatch.setattr(config, "_agent_llm_overrides", {})
    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: [])
    monkeypatch.setattr(serve, "_last_ollama_options", [])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    offered = [o for o in serve.rpc_llm_options({})["options"] if o["provider"] == "anthropic"]
    assert [o["model"] for o in offered] == list(ANTHROPIC_SEAT_MODELS)
    assert all("no key" in o["label"] and "ANTHROPIC_API_KEY" in o["group"] for o in offered)

    seat = serve.rpc_set_seat({"agent": "planner", "provider": "anthropic", "model": offered[0]["model"]})
    assert (seat["provider"], seat["badge"]) == ("anthropic", "NO KEY")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    keyed = [o for o in serve.rpc_llm_options({})["options"] if o["provider"] == "anthropic"]
    assert all(o["label"] == o["model"] and o["group"] == "Anthropic" for o in keyed)


def test_a_daemon_that_cannot_be_asked_leaves_its_last_list_offered(monkeypatch):
    """The dropdown emptied whenever the daemon was down, so no seat could be
    moved -- not even onto a provider that needs no daemon."""
    monkeypatch.setattr(serve, "_last_ollama_options", [])
    monkeypatch.setattr(serve, "ollama_model_capabilities", lambda m: ["completion"])
    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: ["qwen3.8:latest"])
    serve.rpc_llm_options({})

    monkeypatch.setattr(serve, "ollama_daemon_tags", lambda: None)
    ollama = [o for o in serve.rpc_llm_options({})["options"] if o["provider"] == "ollama"]

    assert [o["model"] for o in ollama] == ["qwen3.8:latest"]
    assert "daemon down" in ollama[0]["group"]


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


def test_a_request_is_logged_beside_its_status(capsys):
    """`[API] 404` alone, on the line after `GET /`, read as the page itself missing.

    It was the favicon's: `send_error` logs its code before the request line.
    """
    handler = serve.Handler.__new__(serve.Handler)
    handler.requestline = "GET /favicon.ico HTTP/1.1"
    handler.log_error("code %d, message %s", 404, "File not found")
    handler.log_request(HTTPStatus.NOT_FOUND)
    handler.log_error("Request timed out: %r", "slow client")
    handler.requestline = "POST /rpc HTTP/1.1"
    handler.log_request(200)

    assert capsys.readouterr().out.splitlines() == [
        "[API] GET /favicon.ico HTTP/1.1 -> 404",
        "[API] Request timed out: 'slow client'",
    ]


# ---------------------------------------------------------------------------
# Who may drive the console
# ---------------------------------------------------------------------------


def _raw_post(headers: dict[str, str], body: bytes = b'{"method": "status"}') -> bytes:
    """The whole reply, status line included: a refusal is not JSON."""
    handler = serve.Handler.__new__(serve.Handler)
    handler.path = "/rpc"
    handler.headers = {"Content-Length": str(len(body)), **headers}
    handler.rfile = io.BytesIO(body)
    handler.wfile = _Socket()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /rpc HTTP/1.1"
    handler.command = "POST"
    handler.do_POST()
    return bytes(handler.wfile.data)


def _status(reply: bytes) -> int:
    """The status code off the reply's first line, whatever HTTP version it names."""
    return int(reply.split(b" ", 2)[1])


def test_a_page_on_another_site_cannot_drive_the_console():
    """The console runs goals, and a goal reaches a Builder that runs programs.

    A page the operator merely visited could POST a `text/plain` body -- sent
    without the browser asking first -- and reach `run_goal`, and the wildcard
    CORS header let it read the reply too.
    """
    reply = _raw_post({"Host": "localhost:8080", "Origin": "https://attacker.example"})

    assert _status(reply) == 403
    assert b'"result"' not in reply


def test_a_rebound_domain_is_refused_by_its_host():
    """DNS rebinding makes the browser call a page same-origin; its Host still names it."""
    reply = _raw_post({"Host": "attacker.example:8080", "Origin": "http://attacker.example:8080"})

    assert _status(reply) == 403


def test_the_consoles_own_page_and_a_script_are_both_served():
    """Same-origin browser calls carry a matching Origin; curl carries none."""
    for headers in (
        {"Host": "localhost:8080", "Origin": "http://localhost:8080"},
        {"Host": "127.0.0.1:8080", "Origin": "http://127.0.0.1:8080"},
        {"Host": "localhost:8080"},
    ):
        reply = _raw_post(headers)
        assert _status(reply) == 200, headers
        assert b"Access-Control-Allow-Origin" not in reply


def test_the_console_listens_on_loopback_unless_told_otherwise(monkeypatch):
    assert serve._listening_on_loopback()

    monkeypatch.setattr(serve, "CONSOLE_HOST", "0.0.0.0")
    # Off loopback the operator chose other names for this machine, so the Host
    # check steps aside -- and a foreign page is still refused by its Origin.
    assert serve._foreign_request({"Host": "192.168.1.20:8080"}, writes=True) is None
    assert serve._foreign_request(
        {"Host": "192.168.1.20:8080", "Origin": "https://attacker.example"}, writes=True
    )
