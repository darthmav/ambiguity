"""Seats, model tags and LLM setup, and the Ollama daemon's circuit.

Each of the four seats can run a different model on a different provider.
Inference defaults to local: every seat runs a model the Ollama daemon on this
machine serves from its own weights, so a fresh checkout needs no API key and
no ollama.com credentials. Ollama Cloud tags and Anthropic remain available
per seat. The embedding model runs on the same daemon but belongs to GraphRAG,
never to a seat.
"""

import functools
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

from dotenv import load_dotenv
from pydantic import SecretStr

from langgraph_agent.self_healing import (
    Circuit,
    CircuitOpenError,
    call_with_retry,
    exception_chain,
    get_healing_logger,
)

load_dotenv()


# The four seats, in the order they hold the loop. Anything that iterates
# agents reads this rather than repeating the list.
AgentName = Literal["architect", "planner", "researcher", "builder"]
Provider = Literal["ollama", "anthropic"]

AGENTS: tuple[AgentName, ...] = ("architect", "planner", "researcher", "builder")
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic")

# Every model tag this project names, spelled once. Seat defaults, the tags
# install.sh pulls, the seat diagnostic and ollama_client.py all refer to these.
DOLPHIN_9B = "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"
DOLPHIN3_CYBER_8B = "hf.co/RavichandranJ/Dolphin3-Cyber-8B-GGUF:Q5_K_M"
QWEN3_8 = "qwen3.8:latest"
NEMOTRON_3_ULTRA = "nemotron-3-ultra:cloud"
CLAUDE_OPUS = "claude-opus-5"
CLAUDE_SONNET = "claude-sonnet-5"

# The model each seat takes on each provider when `{ROLE}_PROVIDER` names the
# provider and nothing names the model.
_DEFAULT_AGENT_MODELS: dict[Provider, dict[AgentName, str]] = {
    # Three seats run weights this machine's daemon holds, with no key and no
    # ollama.com credentials. The Builder's work *is* tool calls and neither
    # dolphin tag reports `tools`, so it takes the one local tag that does.
    "ollama": {
        "architect": DOLPHIN_9B, "planner": DOLPHIN_9B,
        "researcher": DOLPHIN_9B, "builder": QWEN3_8,
    },
    "anthropic": {
        "architect": CLAUDE_OPUS, "planner": CLAUDE_OPUS,
        "researcher": CLAUDE_SONNET, "builder": CLAUDE_SONNET,
    },
}

# The Anthropic models the console's seat dropdowns offer: the ones a seat takes
# when `{ROLE}_PROVIDER=anthropic` names no model, so the dropdown and `.env`
# reach the same models. Ollama's offer is whatever the daemon lists.
ANTHROPIC_SEAT_MODELS: tuple[str, ...] = tuple(dict.fromkeys(_DEFAULT_AGENT_MODELS["anthropic"].values()))

# The model a provider runs when neither the seat nor the environment
# (`OLLAMA_MODEL`, `ANTHROPIC_MODEL`) names one.
_PROVIDER_DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": DOLPHIN_9B, "anthropic": CLAUDE_OPUS,
}

# Local first: no seat needs an API key or ollama.com credentials by default.
# Point one elsewhere with {ROLE}_PROVIDER / {ROLE}_MODEL, or from the console.
DEFAULT_PROVIDER: Provider = "ollama"

DEFAULT_SEATS: dict[AgentName, dict[str, str]] = {
    agent: {"provider": DEFAULT_PROVIDER, "model": _DEFAULT_AGENT_MODELS[DEFAULT_PROVIDER][agent]}
    for agent in AGENTS
}


# The tags `install.sh` pulls so a seat can be moved onto any of them without a
# mid-run pull. It is not the offer: the console's dropdowns and `set_seat`
# read `ollama ls` (`_seat_model_options` in serve.py). qwen3-embedding is not
# here -- it is the embedder, never a seat.
AGENT_LLM_OPTIONS: list[dict[str, str]] = [
    # 17 GB against 6 GB of VRAM, so it always runs partly on the CPU -- but it
    # is the only local tag here that reports `tools`.
    {"label": "Qwen3.8", "provider": "ollama", "model": QWEN3_8,
     "group": "Ollama (local)"},
    # 5.3 GB, wholly on the GPU; `completion` only, which is all the three
    # tool-free seats need.
    {"label": "Dolphin 2.9.1 9B", "provider": "ollama", "model": DOLPHIN_9B,
     "group": "Ollama (local)"},
    # 5.7 GB, also `completion` only: a second local choice for those seats.
    {"label": "Dolphin3 Cyber 8B", "provider": "ollama", "model": DOLPHIN3_CYBER_8B,
     "group": "Ollama (local)"},
    {"label": "Nemotron 3 Ultra", "provider": "ollama", "model": NEMOTRON_3_ULTRA,
     "group": "Ollama Cloud"},
]


# How long one call to a seat may wait at the socket. Every provider client
# defaults to no deadline, and `RUN_BUDGET_SECONDS` is checked only between
# supersteps, so a stalled call would hang the run. It fires on a connection
# that goes quiet, not on a model trickling tokens forever -- the node deadline
# in nodes.py covers that.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


def ollama_base_url() -> str:
    """Where the local Ollama daemon listens: every default seat and the embedder."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def daemon_unreachable(exc: BaseException) -> bool:
    """Whether a failure means the Ollama daemon could not be reached at all.

    A refused or reset connection, a host that does not resolve, or a connect
    that timed out. An HTTP status is the daemon answering, and a reply that is
    slow to come is a busy daemon; neither counts. Read off the whole chain,
    because each client wraps it differently: urllib in `URLError`, httpx in
    `ConnectError`, the ollama client in a bare `ConnectionError`.
    """
    for cause in exception_chain(exc):
        if isinstance(cause, urllib.error.HTTPError):
            return False
        # urllib raises URLError only while connecting and sending, before any
        # response: refused, unresolvable, or a connect that timed out.
        if isinstance(cause, (urllib.error.URLError, ConnectionError)):
            return True
        if type(cause).__name__ in ("ConnectError", "ConnectTimeout"):
            return True
    return False


def provider_unavailable(exc: BaseException) -> bool:
    """Whether a cloud provider's failure says the service is down, not the request wrong.

    Unreachable, timed out, or a 5xx (Anthropic's 529 "overloaded" included).
    A 4xx is the request or the key, and is the provider answering.
    """
    for cause in exception_chain(exc):
        if daemon_unreachable(cause) or "Timeout" in type(cause).__name__:
            return True
        status = getattr(cause, "status_code", None)
        if isinstance(status, int):
            return status >= 500
    return False


# The local Ollama daemon, as one circuit: the seats, the embedder and the
# status poll's reads all count toward it, and only a daemon that cannot be
# reached trips it. While it is open every caller fails at once rather than
# each waiting out its own connection, and the first call after the cooldown
# is the probe that closes it again.
OLLAMA_CIRCUIT_THRESHOLD = 3
OLLAMA_CIRCUIT_COOLDOWN_SECONDS = float(os.getenv("OLLAMA_CIRCUIT_COOLDOWN_SECONDS", "15"))
OLLAMA_DAEMON = Circuit(
    "ollama-daemon",
    failure_threshold=OLLAMA_CIRCUIT_THRESHOLD,
    recovery_timeout=OLLAMA_CIRCUIT_COOLDOWN_SECONDS,
    trips_on=daemon_unreachable,
)

# The cloud provider's circuit, keyed by provider, opened by outages rather
# than by refusals. Its SDK already retries a failed request, so nothing here
# retries on top; the circuit is what stops the next seat from waiting out the
# same outage.
PROVIDER_CIRCUIT_COOLDOWN_SECONDS = 60.0
PROVIDER_CIRCUITS: dict[str, Circuit] = {
    "anthropic": Circuit(
        "anthropic-api",
        failure_threshold=3,
        recovery_timeout=PROVIDER_CIRCUIT_COOLDOWN_SECONDS,
        trips_on=provider_unavailable,
    ),
}


# How many times a call is attempted while the daemon cannot be reached, and
# the first wait (doubling after it): enough to ride out a daemon restart,
# short enough that a daemon that is gone fails within seconds and opens its
# circuit.
DAEMON_CONNECT_ATTEMPTS = 3
DAEMON_RETRY_WAIT_SECONDS = 1.0

_R = TypeVar("_R")


def retry_unreachable(
    func: Callable[[], _R], *, name: str, give_up: Callable[[], bool] | None = None
) -> _R:
    """`func`, retried briefly while the daemon cannot be reached.

    Safe for any daemon call: an unreachable daemon ran nothing, so asking
    again cannot repeat work. `func` goes through `OLLAMA_DAEMON` itself.
    """
    return call_with_retry(
        func,
        max_attempts=DAEMON_CONNECT_ATTEMPTS,
        min_wait=DAEMON_RETRY_WAIT_SECONDS,
        max_wait=DAEMON_RETRY_WAIT_SECONDS * 2,
        retry_if=daemon_unreachable,
        give_up=give_up,
        name=name,
    )


def daemon_request(path: str, payload: dict[str, Any] | None = None, *, timeout: float) -> Any:
    """One JSON request to the local Ollama daemon, through its circuit.

    POSTs `payload` when given and GETs otherwise; returns the decoded reply.
    Raises what urllib raises -- an `HTTPError` is the daemon answering -- and
    `CircuitOpenError` while the daemon is known to be unreachable.
    """
    request = urllib.request.Request(
        f"{ollama_base_url()}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    def send() -> Any:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    return OLLAMA_DAEMON.call(send)


_ollama_tags_cache: tuple[float, list[str] | None] = (0.0, None)


def ollama_daemon_tags() -> list[str] | None:
    """Tags the local Ollama daemon reports, cached for 30s; None if it cannot be asked.

    Both the seat dropdowns and the liveness check can ask on every status poll.
    An empty list is a daemon with nothing pulled, which is not an unreachable
    one, and is cached like any answer; a failure is not cached, so a daemon
    that comes back shows at the next poll (its circuit keeps the asking cheap).
    """
    global _ollama_tags_cache

    now = time.monotonic()
    cached_at, cached = _ollama_tags_cache
    if cached is not None and now - cached_at < 30.0:
        return cached

    try:
        payload = daemon_request("/api/tags", timeout=2.0)
        tags = sorted(str(entry["name"]) for entry in payload.get("models", []))
    except Exception:
        return None

    _ollama_tags_cache = (now, tags)
    return tags


def list_ollama_models() -> list[str]:
    """`ollama_daemon_tags()`, with an unreachable daemon as an empty list, never an exception."""
    return ollama_daemon_tags() or []


_ollama_caps_cache: dict[str, tuple[float, list[str] | None]] = {}


def ollama_model_capabilities(model: str) -> list[str] | None:
    """What the daemon says a tag can do (`thinking`, `tools`, ...), cached 30s.

    Asked of the daemon, which answers for every tag it can reach. None means no
    answer -- unreachable, or no such tag -- which is not a list without
    `thinking` in it: "could not ask" must not read as "cannot think". Failures are
    cached too, so a dead daemon costs one refused connection per half-minute.
    """
    now = time.monotonic()
    cached = _ollama_caps_cache.get(model)
    if cached and now - cached[0] < 30.0:
        return cached[1]

    caps: list[str] | None = None
    try:
        payload = daemon_request("/api/show", {"model": model}, timeout=2.0)
        # A daemon too old to report capabilities says nothing about the model.
        listed = payload.get("capabilities")
        if isinstance(listed, list):
            caps = [str(cap) for cap in listed]
    except Exception:
        pass

    _ollama_caps_cache[model] = (now, caps)
    return caps


def _ollama_ps() -> list[dict[str, Any]]:
    """Every model the daemon has loaded (`/api/ps`). Raises when it cannot be asked."""
    return list(daemon_request("/api/ps", timeout=2.0).get("models", []))


def ollama_cpu_share(model: str) -> float | None:
    """The fraction of a loaded `model` the daemon holds in system memory, not on a GPU.

    0.0 when wholly on the cards; None when not loaded or the daemon cannot be
    asked, since an unknown split is not a clean one.
    """
    try:
        loaded = _ollama_ps()
    except Exception:
        return None
    for entry in loaded:
        if _same_ollama_tag(str(entry.get("name") or entry.get("model")), model):
            size = int(entry.get("size") or 0)
            if size <= 0:
                return None
            return max(0.0, 1.0 - int(entry.get("size_vram") or 0) / size)
    return None


def _same_ollama_tag(a: str, b: str) -> bool:
    """`qwen3.8` and `qwen3.8:latest` are one model; the daemon reports the second."""

    def full(tag: str) -> str:
        return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"

    return full(a) == full(b)


# Sent with every call to a seat the daemon runs locally (none for `:cloud`
# tags): `num_gpu: 999` puts every layer on the cards. The daemon's own
# estimate left 37% of a 9B dolphin on the CPU beside 2.2 GB of idle VRAM,
# whatever `num_ctx` was asked for; forced, it loads wholly on the GPU at the
# full window. No `num_ctx` is sent -- the window is the seat's to want.
OLLAMA_SEAT_GPU_OPTIONS: dict[str, int] = {"num_gpu": 999}


def is_local_ollama_model(model: str) -> bool:
    """Whether this tag runs on the cards here rather than at ollama.com.

    A `:cloud` tag is proxied by the daemon and holds no VRAM.
    """
    return not model.endswith((":cloud", "-cloud"))


def unload_ollama_model(model: str) -> bool:
    """Ask the daemon to drop `model` from VRAM now. False if it could not be asked.

    `keep_alive: 0` with no prompt returns once the runner is gone, which is what
    makes "the cards are free from here on" true.
    """
    try:
        daemon_request("/api/generate", {"model": model, "keep_alive": 0}, timeout=30.0)
        return True
    except Exception:
        # An unload that fails costs a slower load, never a wrong answer.
        return False


def free_the_cards_for(model: str) -> list[str]:
    """Unload every locally-resident model but `model`; return what went.

    Serializing the work is not enough: the daemon keeps a runner resident for
    five minutes after its last token, so a finished seat would still hold the
    cards when the embedder loads. Evicting is what turns "one at a time" into
    "all of the cards". Reload costs ~5s for the embedder and ~25s for a 9B seat,
    once per handover.
    """
    try:
        loaded = _ollama_ps()
    except Exception:
        return []
    dropped = []
    for entry in loaded:
        tag = str(entry.get("name") or entry.get("model") or "")
        if not tag or _same_ollama_tag(tag, model) or not is_local_ollama_model(tag):
            continue
        if unload_ollama_model(tag):
            dropped.append(tag)
    return dropped


def _is_gpu_fit_failure(exc: Exception) -> bool:
    """Whether this failure is the daemon refusing to fit a forced offload.

    `num_gpu: 999` trades a graceful split for a hard failure where the model does
    not fit -- right for the embedder, whose vectors depend on placement, wrong for
    a seat, which then retries unforced. Only these shapes mean "did not fit";
    every other failure is the seat's own.
    """
    text = str(exc).lower()
    return any(
        mark in text
        for mark in (
            "out of memory",
            "cudamalloc",
            "unable to allocate",
            "no available devices",
            "timed out waiting for llama-server",
            "unable to load model",
        )
    )


# Why the last call to a seat failed, if it did. Only a real call can tell a
# present key from a working one, so the console reports this instead of
# probing every seat on every poll.
_seat_failures: dict[str, str] = {}


def _failure_reason(exc: Exception) -> str:
    """Turn a provider exception into something a seat card can show."""
    if isinstance(exc, CircuitOpenError):
        return f"{exc.circuit} unreachable; next try in {max(0, round(exc.retry_in))}s"
    text = str(exc)

    # Providers wrap their real message in a dict repr; pull it back out.
    match = re.search(r"'message': '([^']+)'", text)
    message = match.group(1) if match else text

    lowered = message.lower()
    if "credit balance" in lowered:
        return "Anthropic credit balance too low"
    if "authentication" in lowered or "invalid x-api-key" in lowered:
        return "API key rejected"
    if "rate limit" in lowered:
        return "Rate limited"
    # httpx's ReadTimeout has an empty message, so the class name identifies
    # it.
    if "timeout" in lowered or "timed out" in lowered or "Timeout" in type(exc).__name__:
        return f"No response within {int(LLM_TIMEOUT_SECONDS)}s"
    if "not found" in lowered and "model" in lowered:
        return "Model not available on this account"
    if "connect" in lowered or "connection" in lowered:
        return "Provider unreachable"

    return message[:90].strip()


# How every Ollama seat says the daemon could not be reached, whether the
# status poll found it gone or the seat's own last call did. Worded apart, the
# banner read "architect -- ollama-daemon unreachable; next try in 5s" beside
# three seats reading "Ollama daemon unreachable", over one outage.
DAEMON_UNREACHABLE = "Ollama daemon unreachable"


def _seat_failure_reason(provider: str, exc: Exception) -> str:
    """`_failure_reason`, with an unreachable daemon worded one way for every Ollama seat."""
    if provider == "ollama" and (
        (isinstance(exc, CircuitOpenError) and exc.circuit == OLLAMA_DAEMON.name)
        or daemon_unreachable(exc)
    ):
        return DAEMON_UNREACHABLE
    return _failure_reason(exc)


class SeatCallAbandoned(RuntimeError):
    """A seat call ended part-way because the node waiting for it had given up."""


@functools.cache
def _abandonment_watch() -> Any:
    """A callback that ends a streamed seat call between tokens once nobody waits for it.

    `ChatOllama.invoke` streams, and its socket timeout bounds only the gap
    between tokens: a call its node abandoned went on generating -- holding
    `GPU_ARBITER`, so every other seat and the embedder waited -- for as long as
    the model would. Raised from the token callback (`raise_error`), it closes
    the stream, and the daemon stops generating for a client that left.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    from langgraph_agent.control import abandoned

    class _StopWhenAbandoned(BaseCallbackHandler):
        raise_error = True

        def on_llm_new_token(self, token: Any, **kwargs: Any) -> None:
            if abandoned():
                raise SeatCallAbandoned("its node stopped waiting for it")

    return _StopWhenAbandoned()


class _SeatLLM:
    """A seat's chat model, wrapped so its failures reach the console.

    Transparent apart from `invoke` and `bind_tools`. Every seat call passes
    through `invoke`, which is why the cards are arbitrated here: a call takes
    `GPU_ARBITER`, and a local tag evicts whatever else is resident. `build` is
    kept so a forced load that did not fit can be rebuilt unforced. `provider`
    picks the call's circuit, and tells `bind_tools` to ask the daemon whether a
    tag can call tools.
    """

    def __init__(
        self,
        agent: str,
        inner: Any,
        build: "Callable[[bool], Any] | None" = None,
        *,
        force_gpu: bool = True,
        bound: "tuple[Any, dict[str, Any]] | None" = None,
        provider: str = "",
    ) -> None:
        self._agent = agent
        self._inner = inner
        self._build = build
        self._force_gpu = force_gpu
        self._bound = bound
        self._provider = provider

    def _model_tag(self) -> str:
        """The tag this seat calls, for the arbiter and the eviction."""
        return str(getattr(self._inner, "model", "") or "")

    def _unforce(self) -> bool:
        """Rebuild this seat with the daemon choosing the split. False if it cannot.

        Sticky for this seat object; seats are rebuilt per node turn, so the forced
        load is tried again next turn, once whatever crowded it out has gone.
        """
        if self._build is None or not self._force_gpu:
            return False
        inner = self._build(False)
        if self._bound is not None:
            tools, kwargs = self._bound
            bind = getattr(inner, "bind_tools", None)
            if bind is None:
                return False
            inner = bind(tools, **kwargs)
        self._inner = inner
        self._force_gpu = False
        return True

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        """One call to the model, through its provider's circuit.

        A daemon that cannot be reached is retried briefly -- nothing ran, so
        asking again cannot repeat any work -- and the emergency stop ends the
        waiting. Cloud SDKs retry their own requests, so only the circuit is
        added there.
        """
        kwargs = self._watched(kwargs)
        if self._provider == "ollama":
            from langgraph_agent.control import RUN_CONTROL

            return retry_unreachable(
                lambda: OLLAMA_DAEMON.call(self._inner.invoke, *args, **kwargs),
                name=f"seat:{self._agent}",
                give_up=RUN_CONTROL.stopped,
            )
        circuit = PROVIDER_CIRCUITS.get(self._provider)
        if circuit is not None:
            return circuit.call(self._inner.invoke, *args, **kwargs)
        return self._inner.invoke(*args, **kwargs)

    def _watched(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """`kwargs` with the abandonment watch added to the call's callbacks.

        Only for a LangChain model, whose `invoke` takes a config; a stub's does
        not, and has nothing to stream.
        """
        from langchain_core.runnables import Runnable

        if not isinstance(self._inner, Runnable):
            return kwargs
        config = dict(kwargs.get("config") or {})
        config["callbacks"] = [*(config.get("callbacks") or []), _abandonment_watch()]
        return {**kwargs, "config": config}

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        from langgraph_agent.control import GPU_ARBITER

        tag = self._model_tag()
        local = bool(tag) and is_local_ollama_model(tag)
        with GPU_ARBITER.exclusive(f"seat:{self._agent}"):
            if local:
                # Held inside the arbiter, so nothing loads into the gap
                # between freeing the cards and using them.
                free_the_cards_for(tag)
            try:
                result = self._call(*args, **kwargs)
            except SeatCallAbandoned:
                # Nobody is waiting for this answer, and nothing is wrong with
                # the seat: the node already said why it moved on.
                raise
            except Exception as exc:
                if local and _is_gpu_fit_failure(exc) and self._unforce():
                    healing = get_healing_logger()
                    try:
                        result = self._call(*args, **kwargs)
                    except Exception as retried:
                        healing.log_recovery_action(
                            "unforced reload", f"seat:{self._agent}", False,
                            f"{tag}: {_failure_reason(retried)}",
                        )
                        _seat_failures[self._agent] = _seat_failure_reason(self._provider, retried)
                        raise
                    healing.log_recovery_action(
                        "unforced reload", f"seat:{self._agent}", True,
                        f"{tag} did not fit the cards whole, so the daemon chose the split",
                    )
                else:
                    _seat_failures[self._agent] = _seat_failure_reason(self._provider, exc)
                    raise
        # A call that works clears an older failure: the seat recovers on its
        # own.
        _seat_failures.pop(self._agent, None)
        return result

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_SeatLLM":
        """Bind tools, keeping the wrapper so the bound model still reports.

        A seat whose model cannot call tools raises AttributeError -- `StubLLM` has no
        `bind_tools`, and an Ollama tag the daemon says lacks `tools` is refused here
        rather than failing its first call with a 400. "Could not ask" (None) binds
        as before. The bound seat carries `build` and the tools forward, so an
        unforced rebuild keeps its belt.
        """
        inner_bind = getattr(self._inner, "bind_tools", None)
        if inner_bind is None:
            raise AttributeError("bind_tools")
        tag = self._model_tag()
        if self._provider == "ollama" and tag and tool_support("ollama", tag)[0] is False:
            raise AttributeError(f"bind_tools: {tag} cannot call tools")
        return _SeatLLM(
            self._agent,
            inner_bind(tools, **kwargs),
            self._build,
            force_gpu=self._force_gpu,
            bound=(tools, kwargs),
            provider=self._provider,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _accepts_temperature(provider: str, model: str) -> bool:
    """Whether this model still accepts a sampling temperature.

    Anthropic removed it on the 4.6-and-later families, and sending one is a 400
    that reads like a credentials problem.
    """
    if provider != "anthropic":
        return True
    return model.startswith("claude-3") or "-4-5" in model


# Whether a seat thinks before it answers, until someone switches it. A
# switchable model is always sent the flag, off included: for several of them
# (and Opus 5 / Sonnet 5) omitting it means *on*, paying for reasoning nobody
# sees.
DEFAULT_THINKING = False

# The thinking budget of a pre-4.6 Claude model, the only way those can be told
# to think; spent inside the node and socket timeouts, so kept modest.
THINKING_BUDGET_TOKENS = int(os.getenv("THINKING_BUDGET_TOKENS", "4096"))

# Claude models that think on every call: an explicit "disabled" is a 400, so
# their box is ticked and locked.
_ALWAYS_THINKING_CLAUDE = ("claude-fable", "claude-mythos", "claude-opus-5-5")

# Claude models whose "off" is not spelled `disabled`.
_CLAUDE_THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}

# Both orders Anthropic has named models in (`claude-3-7-sonnet`, `claude-
# opus-4-1`); a date suffix is not read as a minor version.
_CLAUDE_VERSION = re.compile(r"claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2})(?!\d))?")

ThinkingSupport = Literal["switch", "never", "always", "unknown"]


def _claude_version(model: str) -> tuple[int, int] | None:
    """(major, minor) of a Claude model id, or None if it is not one."""
    match = _CLAUDE_VERSION.search(model)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _claude_thinking(model: str, on: bool) -> dict[str, Any] | None:
    """The `thinking` parameter that switches a Claude model on or off; None to omit it.

    From 4.6, `adaptive` is the only way on and "off" must be sent explicitly --
    `disabled`, or `_CLAUDE_THINKING_OFF`'s spelling. Before 4.6 a token budget is
    the only way on, and absence means off.
    """
    version = _claude_version(model)
    if version is None or model.startswith(_ALWAYS_THINKING_CLAUDE):
        return None
    if version >= (4, 6):
        if on:
            return {"type": "adaptive"}
        off = next(
            (spelling for prefix, spelling in _CLAUDE_THINKING_OFF.items()
             if model.startswith(prefix)),
            {"type": "disabled"},
        )
        return dict(off)
    if on:
        return {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}
    return None


def tool_support(provider: str, model: str) -> tuple[bool | None, str]:
    """Whether a model can call tools; None when the daemon could not be asked.

    Only the Builder is offered tools, and its work *is* tool calls, so a Builder
    on a model without them reports in full and changes nothing. None is kept
    apart from False, for the reason `thinking_support` keeps `unknown` apart
    from `never`.
    """
    if provider == "ollama":
        caps = ollama_model_capabilities(model)
        if caps is None:
            return None, ""
        return ("tools" in caps), ""

    # Anthropic rejects an unsupported tool call at the API, so there is no
    # quiet failure to warn about.
    return True, ""


def thinking_support(provider: str, model: str) -> tuple[ThinkingSupport, str]:
    """Whether a model can think, and whether the console may switch it.

    Returns the verdict and, when the card cannot offer the switch, the reason:

    - An Ollama tag answers for itself, through the daemon's capabilities.
    - A Claude model is read off its version: 3.7 and later can think.

    `unknown` is never folded into `never`: "could not ask" is not "cannot think".
    """
    if provider == "ollama":
        caps = ollama_model_capabilities(model)
        if caps is None:
            return "unknown", (
                f"Ollama did not describe {model}, so whether it can think is "
                "unknown"
            )
        if "thinking" in caps:
            return "switch", ""
        return "never", f"{model} cannot think"

    if provider == "anthropic":
        if model.startswith(_ALWAYS_THINKING_CLAUDE):
            return "always", f"{model} always thinks; it cannot be switched off"
        version = _claude_version(model)
        if version is None:
            return "unknown", f"{model} is not a Claude model id this can read"
        if version >= (3, 7):
            return "switch", ""
        return "never", f"{model} cannot think"

    return "unknown", f"thinking is not switchable for {provider}"


# Per-seat thinking choices from the console, for the life of the process; kept
# per seat, so they survive a model change.
_agent_thinking: dict[str, bool] = {}


def get_agent_thinking(agent: str) -> bool:
    """The thinking a seat asks for when its model can be switched."""
    return _agent_thinking.get(agent, DEFAULT_THINKING)


def set_agent_thinking(agent: str, on: bool) -> None:
    """Switch one seat's thinking on or off, for the life of the process.

    Refused for a model with no switch, rather than stored and ignored. A failure
    recorded against the seat stays: thinking fixes no credit balance.
    """
    info = get_agent_model_info(cast("AgentName", agent))
    support, reason = thinking_support(info["provider"], info["model"])
    if support != "switch":
        raise ValueError(f"Thinking cannot be switched on this seat: {reason}")
    _agent_thinking[agent] = on


def _thinking_for_call(agent: str) -> bool | None:
    """The thinking flag the seat's next call sends, or None to send none.

    Only a switchable model gets a flag: Ollama refuses `think` for a tag without
    the capability, and an unknown one is left to its own default.
    """
    info = get_agent_model_info(cast("AgentName", agent))
    support, _ = thinking_support(info["provider"], info["model"])
    return get_agent_thinking(agent) if support == "switch" else None


# Per-seat model choices from the console, for the life of the process; they
# win over the environment.
_agent_llm_overrides: dict[str, dict[str, str]] = {}


def set_agent_llm(agent: str, provider: str, model: str) -> None:
    """Seat `provider`/`model` on `agent`, in memory, for the life of the process.

    Moving a seat clears its recorded failure -- that described the old seat --
    but re-selecting the seat it already has does not, so a dead seat cannot be
    made to look live by picking it again.
    """
    current = get_agent_model_info(cast("AgentName", agent))
    if (current["provider"], current["model"]) != (provider, model):
        _seat_failures.pop(agent, None)

    _agent_llm_overrides[agent] = {"provider": provider, "model": model}


def get_llm(
    provider: Provider | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    thinking: bool | None = None,
    force_gpu: bool = True,
) -> Any:
    """Get a chat model.

    Args:
        provider: "ollama" or "anthropic"; detected from the model
            name, then the environment, when omitted. Anything else is refused.
        model: The model; the provider's default when omitted.
        temperature: Sampling temperature, where the model accepts one.
        base_url: Optional API base URL override.
        api_key: Optional API key override.
        timeout: Seconds one call may wait at the socket; `LLM_TIMEOUT_SECONDS`
            when omitted. Each provider spells this differently.
        thinking: Whether the model thinks first; None sends no flag. Pass a bool
            only for a model `thinking_support` calls switchable.
        force_gpu: Whether a local Ollama tag gets `OLLAMA_SEAT_GPU_OPTIONS`;
            `_SeatLLM` turns it off to rebuild a seat whose load did not fit.

    Returns:
        The chat model, or `StubLLM` when the provider needs a key and has none
        -- `get_agent_status()` is what says which.
    """
    provider = provider or _detect_provider(model)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}: expected one of {', '.join(PROVIDERS)}.")
    timeout = LLM_TIMEOUT_SECONDS if timeout is None else timeout

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No key: the daemon holds the ollama.com credentials for `:cloud`
        # tags. `client_kwargs` reaches the httpx client underneath.
        # `reasoning=True` rather than the model's default when thinking is on:
        # the default discards the daemon's `thinking` field, which the
        # Builder's tool loop hands back.
        tag = str(model or os.getenv("OLLAMA_MODEL", _PROVIDER_DEFAULT_MODELS["ollama"]))

        # A local tag is told to put every layer on the cards; a `:cloud` tag
        # gets nothing, since the options would describe hardware that is not
        # ollama.com's. None, not a missing keyword: langchain_ollama sends
        # only the fields that are set.
        num_gpu = (
            OLLAMA_SEAT_GPU_OPTIONS["num_gpu"]
            if force_gpu and is_local_ollama_model(tag)
            else None
        )

        return ChatOllama(
            model=tag,
            temperature=temperature,
            base_url=base_url or ollama_base_url(),
            client_kwargs={"timeout": timeout},
            reasoning=thinking,
            num_gpu=num_gpu,
        )

    # Anthropic, the one cloud provider.
    from langchain_anthropic import ChatAnthropic

    model_name = str(model or os.getenv("ANTHROPIC_MODEL", _PROVIDER_DEFAULT_MODELS["anthropic"]))
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return StubLLM()
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": SecretStr(key),
        "default_request_timeout": timeout,
    }
    thinking_param = (
        None if thinking is None else _claude_thinking(model_name, thinking)
    )
    if thinking_param is not None:
        kwargs["thinking"] = thinking_param
    # A model that is thinking takes no temperature.
    thinks = thinking_param is not None and thinking_param["type"] not in (
        "disabled", "between_tools"
    )
    if _accepts_temperature("anthropic", model_name) and not thinks:
        kwargs["temperature"] = temperature
    if base_url:
        kwargs["base_url"] = base_url
    return ChatAnthropic(**kwargs)


def _detect_provider(model: str | None) -> Provider:
    """The provider for a model name, then for the environment, then the default.

    The name decides first: an Ollama tag has a colon (`qwen3.8:latest`) and a
    Claude model says so. A name that says neither is Anthropic's when Anthropic
    is configured, and the local daemon's otherwise -- `qwen3.8` is a tag too.
    """
    if model:
        if ":" in model:
            return "ollama"
        if "claude" in model.lower():
            return "anthropic"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_MODEL"):
        return "anthropic"
    return DEFAULT_PROVIDER


def _resolve_seat(agent: str) -> dict[str, str | None]:
    """Resolve one agent's provider, model, base URL and key.

    A console override wins, then `{ROLE}_PROVIDER` / `{ROLE}_MODEL` /
    `{ROLE}_BASE_URL` / `{ROLE}_API_KEY`, then the default seat. Every reader of a
    seat routes through here, so the console and the run cannot disagree.
    """
    override = _agent_llm_overrides.get(agent)
    if override:
        chosen = override["provider"]
        return {
            "provider": chosen,
            "model": override["model"],
            # A console selection uses the provider's own credentials.
            "base_url": os.getenv(f"{chosen.upper()}_BASE_URL"),
            "api_key": None,
        }

    prefix = agent.upper()
    provider: str | None = os.getenv(f"{prefix}_PROVIDER")
    model: str | None = os.getenv(f"{prefix}_MODEL")

    if not provider and not model:
        # Not a provider-wide {PROVIDER}_MODEL: the four seats run four
        # different models on purpose.
        seat = DEFAULT_SEATS[cast("AgentName", agent)]
        provider, model = seat["provider"], seat["model"]
    elif provider and not model:
        # An unknown provider has no default model; `get_agent_status` says so.
        model = _DEFAULT_AGENT_MODELS.get(cast("Provider", provider), {}).get(
            cast("AgentName", agent)
        )
    elif model and not provider:
        provider = _detect_provider(model)

    return {
        "provider": provider,
        "model": model,
        "base_url": os.getenv(f"{prefix}_BASE_URL"),
        "api_key": os.getenv(f"{prefix}_API_KEY"),
    }


def get_agent_llm(agent: AgentName, temperature: float = 0.1) -> Any:
    """The chat model holding `agent`'s seat (see `_resolve_seat`), wrapped in `_SeatLLM`."""
    seat = _resolve_seat(agent)

    # Built through a factory, so `_SeatLLM` can rebuild it unforced with
    # everything else -- model, thinking, key -- the same.
    def build(force_gpu: bool) -> Any:
        return get_llm(
            provider=seat["provider"],  # type: ignore[arg-type]
            model=seat["model"],
            temperature=temperature,
            base_url=seat["base_url"],
            api_key=seat["api_key"],
            thinking=_thinking_for_call(agent),
            force_gpu=force_gpu,
        )

    return _SeatLLM(agent, build(True), build, provider=seat["provider"] or "")


def get_agent_model_info(agent: AgentName) -> dict[str, str]:
    """Resolve an agent's provider/model without instantiating an LLM.

    `model` is empty only for a provider this project does not know, which
    `get_agent_status` reports and `get_llm` refuses.
    """
    seat = _resolve_seat(agent)
    return {"provider": seat["provider"] or DEFAULT_PROVIDER, "model": seat["model"] or ""}


def get_agent_status(agent: AgentName) -> dict[str, Any]:
    """Resolve an agent's seat and say whether it can actually run.

    `get_llm` falls back to `StubLLM` without a key while the configured model
    name stays the same, so `live` is what says whether the seat runs -- it drives
    the console's chips and offline banner.
    """
    info = get_agent_model_info(agent)
    provider, model = info["provider"], info["model"]

    # `stubbed` (no key: the run completes on canned text) and not `live` (a
    # key that does not work: the run fails) are worded apart in the console.
    live, reason, badge, stubbed = True, "", "", False

    failure = _seat_failures.get(agent)
    if provider not in PROVIDERS:
        live, reason, badge = False, (
            f"{agent.upper()}_PROVIDER={provider!r} is not one of {', '.join(PROVIDERS)}"
        ), "BAD PROVIDER"
    # A daemon that could not be reached is the daemon's state, not the seat's,
    # so the check below says it -- OFFLINE while it lasts, nothing once the
    # daemon answers again -- rather than this seat's last call.
    elif failure and not (provider == "ollama" and failure == DAEMON_UNREACHABLE):
        live, reason, badge = False, failure, "FAILING"
    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        live, reason, badge, stubbed = False, "ANTHROPIC_API_KEY not set", "NO KEY", True
    elif provider == "ollama":
        # Asked once: an unreachable daemon is None, one with nothing pulled
        # an empty list, and only the first is OFFLINE.
        tags = ollama_daemon_tags()
        if tags is None:
            live, reason, badge = False, DAEMON_UNREACHABLE, "OFFLINE"
        # Compared as tags: `qwen3.8` is `qwen3.8:latest`.
        elif not any(_same_ollama_tag(model, tag) for tag in tags):
            live, reason, badge = False, f"{model} not pulled", "NOT PULLED"

    # Where the prompt goes: a `:cloud` tag's transport is local, but the
    # prompt leaves the machine.
    remote = provider == "anthropic" or not is_local_ollama_model(model)

    # What the next call will do; None when nobody can say.
    support, thinking_note = thinking_support(provider, model)
    thinking = {
        "switch": get_agent_thinking(agent),
        "always": True,
        "never": False,
    }.get(support)

    # Only the Builder is warned: the other seats are offered no tools.
    tools, _ = tool_support(provider, model)
    tools_note = (
        f"{model} cannot call tools, so this Builder would report its work and "
        "change nothing"
        if tools is False and agent == "builder"
        else ""
    )

    return {
        "provider": provider,
        "model": model,
        "live": live,
        "reason": reason,
        "badge": badge,
        "stubbed": stubbed,
        "placement": "REMOTE" if remote else "LOCAL",
        "thinking": thinking,
        "thinking_switchable": support == "switch",
        "thinking_note": thinking_note,
        "tools": tools,
        "tools_note": tools_note,
    }


class StubLLM:
    """Canned, parser-friendly answers in each seat's format, for tests and keyless seats."""

    # Canned text by design: a Builder seat without tools is a fault in a real
    # model, and the expected shape of a stub (see `_seat_pass`).
    is_stub = True

    def invoke(self, messages: list[Any]) -> Any:
        """Return canned responses for testing."""
        from langchain_core.messages import AIMessage

        # The role comes from the system message: retrieved documents in the
        # user message can say anything, "You are the Architect" included.
        system_content = ""
        all_content = ""
        last_content = ""
        for msg in messages:
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            all_content += content + " "
            last_content = content
            if getattr(msg, "type", "") == "system":
                system_content += content + " "

        all_lower = all_content.lower()
        last_lower = last_content.lower()
        # Fall back to the whole prompt only when no system message was given.
        role_source = system_content.lower() or all_lower

        # Every prompt names the other roles in order to route between them, so
        # the role is read off the self-identifying opening, with looser
        # phrases only when a caller wrote its own prompt.
        roles = ("architect", "planner", "researcher", "builder")
        identified = next(
            (role for role in roles if f"you are the {role}" in role_source), None
        )

        is_architect = identified == "architect"
        is_planner = identified == "planner"
        is_researcher = identified == "researcher"
        is_builder = identified == "builder"

        if identified is None:
            is_architect = "## verdict" in role_source
            is_planner = "understand the user's goal" in role_source
            is_researcher = "gather high-quality" in role_source
            is_builder = "implement the plan" in role_source

        # For the Planner: does the *user goal* ask for research? (The state
        # block has a "Research:" label of its own.)
        user_goal_match = re.search(
            r"User goal:\s*(.+)", last_content, re.IGNORECASE | re.DOTALL
        )
        user_goal = user_goal_match.group(1).lower() if user_goal_match else last_lower
        needs_research = "research" in user_goal

        if is_architect:
            # The Architect sets direction and then rules on the report; a
            # populated builder report is what separates the two.
            reviewing = (
                "builder report:" in all_lower
                and "builder report: (empty)" not in all_lower
            )
            if reviewing:
                response = """## Architecture
Change stays within the existing module boundaries.

## Constraints
- Preserve the existing state schema
- No new external services

## Verdict
approved"""
            else:
                response = """## Architecture
Single-module change against the current structure.

## Constraints
- Preserve the existing state schema
- No new external services

## Verdict
plan"""

        elif is_planner:
            # Planner response format
            if needs_research:
                response = """## Goal
Research Python best practices

## Steps
1. Search for existing documentation
2. Identify key patterns
3. Summarize findings

## Next Agent
Researcher

## Notes
Research needed for knowledge gathering"""
            else:
                response = """## Goal
Create a file with content

## Steps
1. Create the file
2. Write the content
3. Verify the file

## Next Agent
Builder

## Notes
Task is straightforward, no research needed"""

        elif is_researcher:
            # Researcher response format
            response = """## Key Findings
- Found relevant patterns in documentation
- Identified best practices

## Relevant Context
Existing code follows similar patterns

## Recommendations for Builder
Implement using the identified patterns

## Status
ready_for_builder"""

        elif is_builder:
            # Builder response format
            response = """## Changes Made
- Created file with specified content
- Verified file exists

## Files Modified
- hello.txt

## Next Steps / Blockers
none"""

        else:
            # Default fallback
            response = """## Goal
Complete the task

## Steps
1. Understand requirements
2. Implement solution

## Next Agent
Builder

## Notes
Default response"""

        return AIMessage(content=response)
