"""Per-seat capability: what each seat sends, and what its card says.

The thinking switch, and the tool support beside it -- both read off the
daemon's capabilities through `ollama_model_capabilities`, on one cached
answer per tag.

Nothing here reaches a daemon or a provider. The daemon is stubbed at
`ollama_model_capabilities` -- the one place it is asked -- or at `urlopen`
when the asking itself is under test, and the Claude client is built but
never invoked, which makes no request.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_agent import config, initial_state, nodes

THINKS = ["completion", "thinking", "tools"]
PLAIN = ["completion"]


@pytest.fixture(autouse=True)
def _fresh_seats(monkeypatch):
    """Every switch, move, failure and cached answer here is a process global.

    Replaced rather than cleared, so a test cannot leak its seats into the
    next one and cannot wipe state another module set up either. The tag list
    is stubbed because `get_agent_status` asks the daemon for it, and a test
    that asserted on a live daemon would pass on a laptop and fail in CI.
    """
    monkeypatch.setattr(config, "_agent_llm_overrides", {})
    monkeypatch.setattr(config, "_agent_thinking", {})
    monkeypatch.setattr(config, "_seat_failures", {})
    monkeypatch.setattr(config, "_ollama_caps_cache", {})
    monkeypatch.setattr(
        config, "list_ollama_models", lambda: ["thinks:cloud", "plain:latest"]
    )


def _daemon(monkeypatch, answers: dict[str, list[str] | None]) -> None:
    monkeypatch.setattr(config, "ollama_model_capabilities", answers.get)


# ---------------------------------------------------------------------------
# asking the daemon
# ---------------------------------------------------------------------------


class _Reply:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_the_daemon_is_asked_once_per_half_minute(monkeypatch):
    """The status poll asks for every seat every five seconds."""
    asked: list[str] = []

    def urlopen(request, timeout):
        asked.append(json.loads(request.data)["model"])
        return _Reply({"capabilities": THINKS})

    monkeypatch.setattr(config.urllib.request, "urlopen", urlopen)

    assert config.ollama_model_capabilities("thinks:cloud") == THINKS
    assert config.ollama_model_capabilities("thinks:cloud") == THINKS
    assert asked == ["thinks:cloud"]


def test_a_daemon_that_cannot_answer_is_unknown_not_a_model_that_cannot_think(
    monkeypatch,
):
    """"Could not ask" and "cannot think" call for different things.

    An unreachable daemon and one too old to report capabilities both leave
    the question open. Reading either as an empty list would lock every card
    and tell the operator four models lost a capability they still have.
    """
    attempts: list[str] = []

    def refuse(request, timeout):
        attempts.append(json.loads(request.data)["model"])
        raise ConnectionRefusedError

    monkeypatch.setattr(config.urllib.request, "urlopen", refuse)
    assert config.ollama_model_capabilities("thinks:cloud") is None
    # The miss is remembered too: a dead daemon costs one refused connection
    # per half-minute, not one per seat per five-second poll.
    assert config.ollama_model_capabilities("thinks:cloud") is None
    assert attempts == ["thinks:cloud"]

    monkeypatch.setattr(config, "_ollama_caps_cache", {})
    monkeypatch.setattr(
        config.urllib.request, "urlopen",
        lambda request, timeout: _Reply({"modelfile": "FROM x"}),
    )
    assert config.ollama_model_capabilities("thinks:cloud") is None

    verdict, reason = config.thinking_support("ollama", "thinks:cloud")
    assert verdict == "unknown"
    assert "thinks:cloud" in reason


def test_an_ollama_tag_is_switchable_exactly_when_the_daemon_says_it_thinks(
    monkeypatch,
):
    _daemon(monkeypatch, {"thinks:cloud": THINKS, "plain:latest": PLAIN})

    assert config.thinking_support("ollama", "thinks:cloud") == ("switch", "")
    assert config.thinking_support("ollama", "plain:latest") == (
        "never", "plain:latest cannot think"
    )


# ---------------------------------------------------------------------------
# what goes on the wire
# ---------------------------------------------------------------------------


def test_ollama_is_told_either_way_and_left_alone_when_nobody_asked():
    """Off has to be said: a thinking tag given no flag thinks.

    `None` stays the model's own default, which is what every call sent
    before the switch existed.
    """
    def reasoning(thinking):
        return config.get_llm(
            provider="ollama", model="thinks:cloud", thinking=thinking
        ).reasoning

    assert reasoning(True) is True
    assert reasoning(False) is False
    assert reasoning(None) is None


@pytest.mark.parametrize("model,version", [
    ("claude-opus-5", (5, 0)),
    ("claude-sonnet-5", (5, 0)),
    ("claude-haiku-4-5", (4, 5)),
    ("claude-opus-4-6", (4, 6)),
    ("claude-opus-4-1-20250805", (4, 1)),
    # A date is not a minor version: this is Sonnet 4.0, not 4.20250514,
    # which would be read as adaptive and fail every call with a 400.
    ("claude-sonnet-4-20250514", (4, 0)),
    ("claude-3-7-sonnet-latest", (3, 7)),
    ("claude-3-5-haiku-20241022", (3, 5)),
    ("qwen3.8:latest", None),
])
def test_a_claude_version_is_read_from_either_naming_order(model, version):
    assert config._claude_version(model) == version


def _claude_payload(monkeypatch, model: str, thinking: bool | None) -> dict[str, Any]:
    """The request body langchain would send, built without sending it.

    Flattened, because langchain moves the sampling parameters the installed
    `anthropic` SDK no longer takes by name into `extra_body` -- they are
    still sent, and a check that looked only at the top level would pass on a
    temperature that was going out regardless.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = config.get_llm(provider="anthropic", model=model, thinking=thinking)
    payload = llm._get_request_payload([HumanMessage(content="hi")])
    return {**(payload.pop("extra_body", None) or {}), **payload}


def test_opus_5_is_told_off_explicitly_because_absence_means_on(monkeypatch):
    """Opus 5 and Sonnet 5 think when `thinking` is left out.

    So an unticked box that sent nothing would leave the seat thinking while
    the card said it was not, and `budget_tokens` is a 400 on these models.
    """
    for model in ("claude-opus-5", "claude-sonnet-5"):
        on = _claude_payload(monkeypatch, model, True)
        off = _claude_payload(monkeypatch, model, False)
        untouched = _claude_payload(monkeypatch, model, None)

        assert on["thinking"] == {"type": "adaptive"}
        assert off["thinking"] == {"type": "disabled"}
        assert "thinking" not in untouched
        assert "temperature" not in on and "temperature" not in off


def test_haiku_4_5_thinks_on_a_budget_and_drops_its_temperature(monkeypatch):
    """Before 4.6 a budget is the only way on, and the families that still
    accept a temperature reject anything but the default once thinking is."""
    on = _claude_payload(monkeypatch, "claude-haiku-4-5", True)
    off = _claude_payload(monkeypatch, "claude-haiku-4-5", False)

    assert on["thinking"] == {
        "type": "enabled", "budget_tokens": config.THINKING_BUDGET_TOKENS,
    }
    assert "temperature" not in on
    # Anthropic's own invariants on a budget: at least 1024, below max_tokens.
    assert 1024 <= config.THINKING_BUDGET_TOKENS < on["max_tokens"]

    assert "thinking" not in off
    assert off["temperature"] == 0.1


def test_a_model_that_always_thinks_is_locked_on_and_never_sent_disabled(
    monkeypatch,
):
    """Anthropic rejects an explicit off on these, so offering the switch
    would hand the operator a box whose off state fails every call."""
    assert config.thinking_support("anthropic", "claude-fable-5-1")[0] == "always"
    assert "thinking" not in _claude_payload(monkeypatch, "claude-fable-5-1", False)


# ---------------------------------------------------------------------------
# the seat
# ---------------------------------------------------------------------------


def test_a_seat_thinks_by_default_and_keeps_its_switch_across_a_move(monkeypatch):
    """The switch belongs to the seat, so unticking the Builder survives
    moving the Builder to another model."""
    _daemon(monkeypatch, {"thinks:cloud": THINKS, "other:cloud": THINKS})

    config.set_agent_llm("builder", "ollama", "thinks:cloud")
    assert config.get_agent_status("builder")["thinking"] is config.DEFAULT_THINKING

    config.set_agent_thinking("builder", False)
    config.set_agent_llm("builder", "ollama", "other:cloud")
    status = config.get_agent_status("builder")

    assert status["thinking"] is False
    assert status["thinking_switchable"] is True


def test_a_model_that_cannot_think_is_locked_and_refuses_the_switch():
    """Refused and said so, rather than stored and ignored: a request that
    changes nothing must not come back looking as though it worked."""
    config.set_agent_llm("planner", "anthropic", "claude-3-5-sonnet-20241022")
    status = config.get_agent_status("planner")

    assert status["thinking"] is False
    assert status["thinking_switchable"] is False
    assert status["thinking_note"] == "claude-3-5-sonnet-20241022 cannot think"
    with pytest.raises(ValueError, match="cannot be switched"):
        config.set_agent_thinking("planner", True)


def test_a_seat_nobody_can_describe_reads_as_neither_on_nor_off(monkeypatch):
    _daemon(monkeypatch, {})
    config.set_agent_llm("researcher", "ollama", "thinks:cloud")
    status = config.get_agent_status("researcher")

    assert status["thinking"] is None
    assert status["thinking_switchable"] is False
    assert status["thinking_note"]


def test_the_flag_reaches_only_a_model_that_has_the_switch(monkeypatch):
    """A tag without the capability answers a `think` flag with a 400.

    Measured against the daemon: `"... does not support thinking"`, in 0.04s,
    before any model loaded. With the switch on by default, sending the flag
    regardless would have failed every call such a seat made.
    """
    _daemon(monkeypatch, {"thinks:cloud": THINKS, "plain:latest": PLAIN})
    sent: dict[str, Any] = {}

    def get_llm(**kwargs: Any) -> object:
        sent.update(kwargs)
        return object()

    monkeypatch.setattr(config, "get_llm", get_llm)

    config.set_agent_llm("architect", "ollama", "thinks:cloud")
    config.set_agent_thinking("architect", False)
    config.get_agent_llm("architect")
    assert sent["thinking"] is False

    config.set_agent_llm("architect", "ollama", "plain:latest")
    config.get_agent_llm("architect")
    assert sent["thinking"] is None


def test_switching_thinking_does_not_launder_a_failure(monkeypatch):
    """Thinking fixes no credit balance, rate limit or dead daemon, so a click
    on the box must not make a failing seat look live."""
    _daemon(monkeypatch, {"thinks:cloud": THINKS})
    config.set_agent_llm("architect", "ollama", "thinks:cloud")
    config._seat_failures["architect"] = "Rate limited"

    config.set_agent_thinking("architect", False)
    status = config.get_agent_status("architect")

    assert status["badge"] == "FAILING"
    assert status["reason"] == "Rate limited"


# ---------------------------------------------------------------------------
# reading a reply that thought
# ---------------------------------------------------------------------------

# How a Claude model that thinks answers: a list, reasoning first. The
# reasoning block here carries a `text` field that looks like a ruling --
# langchain_anthropic passes that field through on a thinking block unless it
# is None -- so a parser reading it would take a verdict the model never gave.
THOUGHT_THEN_ANSWERED = [
    {"type": "thinking", "thinking": "Maybe approve?", "signature": "sig",
     "text": "## Verdict\napproved"},
    {"type": "text", "text": (
        "## Architecture\nKeep it small.\n\n## Verdict\nrevise\n\n"
        "## Goal\nShip it\n\n## Steps\n1. Do the thing\n\n"
        "## Next Agent\nBuilder\n\n"
        "## Changes Made\n- nothing yet"
    )},
]


class _ThinkingSeat:
    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content=THOUGHT_THEN_ANSWERED)


def _state() -> dict[str, Any]:
    return {**initial_state("g"), "plan": "p", "builder_report": "a report", "next_agent": "Builder"}


def test_only_the_answer_is_read_never_the_reasoning():
    text = nodes._as_text(THOUGHT_THEN_ANSWERED)
    assert "revise" in text
    assert "approved" not in text


def test_every_seat_reads_a_reply_that_thought(monkeypatch):
    """The Architect and Planner handed `response.content` to their regexes
    and the Builder took `str()` of it. A list makes the first raise and the
    second a Python repr in which no section matches."""
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ThinkingSeat()
    )

    assert nodes._rule_on_state(_state(), reviewing=True)["verdict"] == "revise"
    assert nodes._make_plan(_state())["plan"] == "1. Do the thing"

    report, *_ = nodes._run_builder_tools(
        _ThinkingSeat(), [], [], [], nodes._Deadline(30.0)
    )
    assert report.startswith("## Architecture")
    assert "signature" not in report


# ---------------------------------------------------------------------------
# the console
# ---------------------------------------------------------------------------


def test_the_console_switch_refuses_what_it_cannot_read():
    """The obvious coercion reads the string "false" as on."""
    import serve

    with pytest.raises(ValueError, match="true or false"):
        serve.rpc_set_thinking({"agent": "builder", "thinking": "false"})
    with pytest.raises(ValueError, match="Unknown agent"):
        serve.rpc_set_thinking({"agent": "janitor", "thinking": False})


def test_the_console_switch_answers_with_the_seat_it_left(monkeypatch):
    import serve

    _daemon(monkeypatch, {"thinks:cloud": THINKS})
    config.set_agent_llm("builder", "ollama", "thinks:cloud")

    result = serve.rpc_set_thinking({"agent": "builder", "thinking": False})

    assert result["ok"] is True
    assert result["thinking"] is False
    assert serve.RPC_METHODS["set_thinking"] is serve.rpc_set_thinking


# ---------------------------------------------------------------------------
# tools: the other capability a seat card reports
# ---------------------------------------------------------------------------


def test_a_tag_calls_tools_exactly_when_the_daemon_says_it_can(monkeypatch):
    _daemon(monkeypatch, {"thinks:cloud": THINKS, "plain:latest": PLAIN})

    assert config.tool_support("ollama", "thinks:cloud")[0] is True
    assert config.tool_support("ollama", "plain:latest")[0] is False


def test_a_daemon_that_cannot_answer_is_unknown_not_a_model_without_tools(monkeypatch):
    """`None` and `False` are different claims, as they are for thinking."""
    _daemon(monkeypatch, {})

    assert config.tool_support("ollama", "plain:latest")[0] is None


def test_a_cloud_provider_is_taken_to_call_tools(monkeypatch):
    """Anthropic refuses an unsupported tool call outright; there is no quiet failure."""
    assert config.tool_support("anthropic", "claude-opus-5")[0] is True


def test_only_the_builder_is_warned_about_a_model_that_cannot_call_tools(monkeypatch):
    """The other three seats are offered no tools, so this is the right seat for one."""
    _daemon(monkeypatch, {"plain:latest": PLAIN})
    for agent in config.AGENTS:
        config.set_agent_llm(agent, "ollama", "plain:latest")

    notes = {agent: config.get_agent_status(agent)["tools_note"] for agent in config.AGENTS}

    assert notes["builder"], "a Builder that can change nothing said nothing"
    assert "cannot call tools" in notes["builder"]
    assert [notes[a] for a in ("architect", "planner", "researcher")] == ["", "", ""]


def test_a_builder_that_can_call_tools_is_not_warned(monkeypatch):
    _daemon(monkeypatch, {"thinks:cloud": THINKS})
    config.set_agent_llm("builder", "ollama", "thinks:cloud")

    status = config.get_agent_status("builder")

    assert status["tools"] is True
    assert status["tools_note"] == ""


def test_a_builder_nobody_could_describe_is_not_accused(monkeypatch):
    """A daemon down for thirty seconds must not read as a seat that lost tools."""
    _daemon(monkeypatch, {})
    config.set_agent_llm("builder", "ollama", "plain:latest")

    status = config.get_agent_status("builder")

    assert status["tools"] is None
    assert status["tools_note"] == ""


# ---------------------------------------------------------------------------
# a seat that cannot call tools is given none
# ---------------------------------------------------------------------------


class _Bindable:
    """A chat model that binds anything, the way ChatOllama does for every tag."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.bound: list[Any] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> _Bindable:
        self.bound.append(tools)
        return self


def test_binding_tools_to_a_tag_without_them_raises_like_a_stub(monkeypatch):
    """Binding never failed for an Ollama tag: the daemon refused the call.

    `ChatOllama.bind_tools` exists for every tag, so the AttributeError the
    Builder's no-tools path waits for never came -- the belt was bound, the
    first call went out, and the daemon answered 400 "does not support tools",
    failing the run that the card said would only change nothing.
    """
    _daemon(monkeypatch, {"plain:latest": PLAIN, "thinks:cloud": THINKS})

    plain = config._SeatLLM("builder", _Bindable("plain:latest"), provider="ollama")
    with pytest.raises(AttributeError):
        plain.bind_tools(["filesystem_write"])

    able = config._SeatLLM("builder", _Bindable("thinks:cloud"), provider="ollama")
    assert able.bind_tools(["filesystem_write"])._provider == "ollama"


def test_a_tag_nobody_could_describe_is_still_offered_its_tools(monkeypatch):
    """"Could not ask" is not "cannot", so the belt is bound as it always was."""
    _daemon(monkeypatch, {})
    inner = _Bindable("plain:latest")

    config._SeatLLM("builder", inner, provider="ollama").bind_tools(["filesystem_write"])

    assert inner.bound == [["filesystem_write"]]


def test_the_builder_reports_instead_of_failing_on_a_seat_without_tools(monkeypatch):
    """End to end through `builder_node`, with the daemon's refusal stood in.

    A `:cloud` tag keeps `_SeatLLM.invoke` off the path that evicts models from
    the cards, which would reach a real daemon.
    """
    _daemon(monkeypatch, {"notools:cloud": PLAIN})

    class _RefusesTools:
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("notools:cloud does not support tools (status code: 400)")

    class _NoTools(_Bindable):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return _RefusesTools()

        def invoke(self, messages: Any) -> AIMessage:
            return AIMessage(
                content="## Changes Made\n- Proposed the edit\n\n"
                "## Files Modified\n\n## Blockers\nnone"
            )

    seat = config._SeatLLM("builder", _NoTools("notools:cloud"), provider="ollama")
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: seat)
    state: Any = {**initial_state("g"), "plan": "1. Edit it."}

    result = nodes.builder_node(state)

    assert "Proposed the edit" in result["builder_report"]
    assert result["files_changed"] == []
    assert "builder" not in config._seat_failures


# ---------------------------------------------------------------------------
# Claude 5.5
# ---------------------------------------------------------------------------


def test_opus_5_5_is_locked_on_because_it_rejects_every_off(monkeypatch):
    """Opus 5.5 answers `{"type": "disabled"}` with a 400 at every effort level.

    Left switchable, the unticked default sent exactly that, so a seat moved
    onto the current Opus failed every call it made.
    """
    assert config.thinking_support("anthropic", "claude-opus-5-5")[0] == "always"
    assert "thinking" not in _claude_payload(monkeypatch, "claude-opus-5-5", False)
    assert "thinking" not in _claude_payload(monkeypatch, "claude-opus-5-5", True)


def test_sonnet_5_5_is_switched_off_with_between_tools(monkeypatch):
    """Sonnet 5.5 rejects `disabled` too, but can still be switched off --
    `between_tools` is the word for off on it."""
    assert config.thinking_support("anthropic", "claude-sonnet-5-5")[0] == "switch"

    on = _claude_payload(monkeypatch, "claude-sonnet-5-5", True)
    off = _claude_payload(monkeypatch, "claude-sonnet-5-5", False)

    assert on["thinking"] == {"type": "adaptive"}
    assert off["thinking"] == {"type": "between_tools"}
    assert "temperature" not in on and "temperature" not in off


# ---------------------------------------------------------------------------
# resolving a seat
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_provider_env(monkeypatch):
    """`.env` is loaded at import, so its keys are cleared before each case."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_a_named_model_settles_its_provider_before_the_environment(
    monkeypatch, _no_provider_env
):
    """A model's name says whose it is before any key is read.

    The keys were read first once, so a seat named for one provider went to
    another and failed every call with a model that provider does not have. A
    name that says neither is Anthropic's when Anthropic is configured, and the
    local daemon's otherwise.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert config._detect_provider("qwen3.8:latest") == "ollama"
    assert config._detect_provider("claude-sonnet-5") == "anthropic"
    assert config._detect_provider("somemodel") == "anthropic"
    assert config._detect_provider(None) == "anthropic"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert config._detect_provider("claude-sonnet-5") == "anthropic"
    assert config._detect_provider("somemodel") == "ollama"
    assert config._detect_provider(None) == "ollama"


def test_openai_is_not_a_provider(monkeypatch, _no_provider_env):
    """Nothing here runs on OpenAI: a seat pointed there is refused by name."""
    monkeypatch.setenv("BUILDER_PROVIDER", "openai")
    monkeypatch.setenv("BUILDER_MODEL", "gpt-4o")

    assert config.get_agent_status("builder")["badge"] == "BAD PROVIDER"
    with pytest.raises(ValueError, match="Unknown provider 'openai'"):
        config.get_llm(provider="openai", model="gpt-4o")


def test_a_tag_written_without_its_version_is_the_pulled_model(monkeypatch):
    """The daemon lists `plain:latest`; a seat configured as `plain` is that model."""
    _daemon(monkeypatch, {"plain": PLAIN, "plain:latest": PLAIN})
    config.set_agent_llm("architect", "ollama", "plain")

    status = config.get_agent_status("architect")

    assert status["badge"] != "NOT PULLED"
    assert status["live"]
