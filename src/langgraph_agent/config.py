"""Configuration and LLM setup.

Supports per-agent LLM selection so Architect, Planner, Researcher, and Builder
can each use a different model/provider. Inference defaults to local: every
seat defaults to a model the Ollama daemon on this machine serves from its own
weights, so a fresh checkout runs with no API key and no ollama.com credentials
at all. Ollama Cloud tags, Anthropic and OpenAI remain available per seat for
anyone who wants them.

The embedding model also runs on this machine, served by the same daemon --
it belongs to GraphRAG rather than to any agent seat, and is never a seat
choice itself.
"""

import functools
import json
import os
import re
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Literal, cast

from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()


# The four seats, in the order they hold the loop. Anything that iterates
# agents reads this rather than repeating the list.
AgentName = Literal["architect", "planner", "researcher", "builder"]
Provider = Literal["openai", "anthropic", "ollama"]

AGENTS: tuple[AgentName, ...] = ("architect", "planner", "researcher", "builder")


# Default model per (provider, agent) when a per-agent provider is configured
# but no model is supplied. Local-first defaults.
_DEFAULT_AGENT_MODELS: dict[tuple[str, str], str] = {
    ("anthropic", "architect"): "claude-opus-5",
    ("anthropic", "planner"): "claude-opus-5",
    ("anthropic", "researcher"): "claude-sonnet-5",
    ("anthropic", "builder"): "claude-sonnet-5",
    ("ollama", "architect"): "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M",
    ("ollama", "planner"): "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M",
    ("ollama", "researcher"): "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M",
    ("ollama", "builder"): "qwen3.8:latest",
    ("openai", "architect"): "gpt-4o",
    ("openai", "planner"): "gpt-4o",
    ("openai", "researcher"): "gpt-4o-mini",
    ("openai", "builder"): "gpt-4o-mini",
}


# The seat each agent takes when nothing overrides it. Three of the four run a
# model this machine's own daemon holds the weights for -- no API key, no
# ollama.com credentials, nothing to sign in to. The Builder is the exception:
# its work *is* tool calls, and neither dolphin tag reports `tools` (checked
# live against this daemon -- both answer `capabilities: ["completion"]` and
# nothing else), so it takes the one local tag that does. Point a seat at
# Anthropic, OpenAI or an Ollama Cloud tag with {ROLE}_PROVIDER / {ROLE}_MODEL,
# or from the console dropdown.
DEFAULT_SEATS: dict[str, dict[str, str]] = {
    "architect": {"provider": "ollama",
                  "model": "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"},
    "planner": {"provider": "ollama",
                "model": "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"},
    "researcher": {"provider": "ollama",
                   "model": "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"},
    "builder": {"provider": "ollama", "model": "qwen3.8:latest"},
}


# The tags `install.sh` pulls so a seat can be moved onto any of them without a
# mid-run pull. It is NOT the offer: the console's dropdowns and `set_seat` read
# `ollama ls` (`_seat_model_options` in serve.py), so anything the daemon carries
# is selectable and nothing here is offered unless it has been pulled. `group` is
# kept for the labels. qwen3-embedding is not a seat choice -- the daemon reports
# it with no `completion` capability -- and the embedder card offers it instead.
# Anthropic and OpenAI still work for a seat configured in .env.
AGENT_LLM_OPTIONS: list[dict[str, str]] = [
    {"label": "Qwen3.8", "provider": "ollama", "model": "qwen3.8:latest",
     "group": "Ollama (local)"},
    # qwen3.8 is 17 GB against 6 GB of VRAM and always takes the CPU fallback
    # (see the seat-placement bullet in CLAUDE.md), but it is the only local
    # tag here that reports `tools`, which is why it -- not either dolphin --
    # holds the Builder by default: that seat's work *is* tool calls.
    {"label": "Dolphin 2.9.1 9B", "provider": "ollama",
     "model": "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M",
     "group": "Ollama (local)"},
    # 5.3 GB, loads 100% on the GPU, measured 2026-09-20 at 2,958 + 2,975 MiB.
    # Holds Architect, Planner and Researcher by default -- none of the three
    # are offered tools anyway, so `completion`-only is the right seat for
    # them rather than a defect (same reasoning `get_agent_status` uses for
    # the Builder's `tools_note`, just the other three seats it doesn't fire
    # for).
    {"label": "Dolphin3 Cyber 8B", "provider": "ollama",
     "model": "hf.co/RavichandranJ/Dolphin3-Cyber-8B-GGUF:Q5_K_M",
     "group": "Ollama (local)"},
    # 5.7 GB. Also `completion`-only (checked live against this daemon, same
    # as the 2.9.1 tag above) -- not a Builder option, but a second local
    # choice for the three tool-free seats.
    {"label": "Nemotron 3 Ultra", "provider": "ollama", "model": "nemotron-3-ultra:cloud",
     "group": "Ollama Cloud"},
]


# How long one call to a seat may take before the client gives up. Every
# provider client defaults to no deadline at all, so a stalled cloud call
# blocked `llm.invoke` forever -- and RUN_BUDGET_SECONDS could not end it,
# because that is checked between graph supersteps and a node in flight never
# reaches a superstep boundary. A run could therefore hang indefinitely inside
# a single node with the console still naming the previous one.
#
# This bounds the socket, not the call: for Ollama it becomes an httpx timeout
# on a streamed response, so it fires on a connection that goes quiet, not on a
# model that trickles tokens forever. The node-level deadline in nodes.py
# covers that second case; the two are not redundant.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


def _ollama_base_url() -> str:
    """Where the local Ollama daemon listens.

    Every default seat runs through it: a local tag from weights the daemon
    holds itself, or a `:cloud` tag it forwards to ollama.com. The embedder is
    served by the same daemon.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


_ollama_tags_cache: tuple[float, list[str]] = (0.0, [])


def list_ollama_models() -> list[str]:
    """Tags the local Ollama daemon reports, cached for 30s.

    Feeds both the seat dropdowns and the liveness check, either of which can
    be hit on every status poll -- hence the cache and the short timeout. An
    unreachable daemon is an empty list, never an exception.
    """
    global _ollama_tags_cache

    now = time.monotonic()
    cached_at, cached = _ollama_tags_cache
    if cached and now - cached_at < 30.0:
        return cached

    try:
        with urllib.request.urlopen(
            f"{_ollama_base_url()}/api/tags", timeout=2.0
        ) as response:
            payload = json.loads(response.read())
        tags = sorted(str(entry["name"]) for entry in payload.get("models", []))
    except Exception:
        tags = []

    _ollama_tags_cache = (now, tags)
    return tags


_ollama_caps_cache: dict[str, tuple[float, list[str] | None]] = {}


def ollama_model_capabilities(model: str) -> list[str] | None:
    """What the daemon says a tag can do (`thinking`, `tools`, ...), cached 30s.

    Asked of the daemon rather than kept as a list here, because the daemon
    answers for every tag it can reach -- cloud tags that were never pulled
    included -- and a list kept here would be wrong about the next tag someone
    pulls.

    `None` means no answer: the daemon is unreachable or has no such tag. That
    is a different claim from a list without `thinking` in it, and the seat
    card must not turn "could not ask" into "this model cannot think". Failures
    are cached as well as answers: the status poll asks for every seat every
    five seconds, and a dead daemon should cost one refused connection per
    half-minute rather than four per poll.
    """
    now = time.monotonic()
    cached = _ollama_caps_cache.get(model)
    if cached and now - cached[0] < 30.0:
        return cached[1]

    request = urllib.request.Request(
        f"{_ollama_base_url()}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    caps: list[str] | None = None
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read())
        # A daemon too old to report capabilities answers without the field,
        # which says nothing about the model -- so it stays None, not [].
        listed = payload.get("capabilities")
        if isinstance(listed, list):
            caps = [str(cap) for cap in listed]
    except Exception:
        pass

    _ollama_caps_cache[model] = (now, caps)
    return caps


def _ollama_ps() -> list[dict[str, Any]]:
    """Every model the local daemon has loaded, as `/api/ps` lists them.

    Raises when the daemon cannot be asked; each caller decides what that means.
    """
    with urllib.request.urlopen(f"{_ollama_base_url()}/api/ps", timeout=2.0) as response:
        payload = json.loads(response.read())
    return list(payload.get("models", []))


def ollama_cpu_share(model: str) -> float | None:
    """The fraction of a loaded `model` the daemon holds in system memory, not on a GPU.

    0.0 when it is wholly on the cards. None when it is not loaded or the
    daemon cannot be asked, because an unknown split is not a clean one.
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


# Sent with every call to a seat the daemon runs locally, and with none it
# proxies to ollama.com. `num_gpu` is llama.cpp's layer-offload count, and 999
# means "every layer": the same lever `OLLAMA_EMBED_OPTIONS` pulls for the
# embedder, which seats were never given.
#
# The daemon's own estimate is the problem, and it is not a context-window
# problem. Measured on 2026-09-20, dolphin-2.9.1-yi-1.5-9b Q4_K_M on two 3 GB
# GTX 1060s, one load per row:
#
#     options sent                      placement        card 0 + card 1
#     (none -- what a seat got)         37% CPU/63% GPU  1940 + 1981 MiB
#     num_ctx 2048                      37% CPU/63% GPU  1892 + 1919 MiB
#     num_ctx 4096, num_gpu 999         100% GPU         2968 + 2975 MiB
#
# Shrinking the window moved nothing: the estimator leaves ~1.1 GB unused on
# *each* card whatever it is asked for, and 37% of the model runs on the CPU
# beside 2.2 GB of idle VRAM. Forced, the whole model lands on the cards at the
# full 4,096-token window. No `num_ctx` is sent with it, because the window is
# the seat's to want and the measurement says it was never what decided the
# split -- a seat capped here would lose context for nothing.
OLLAMA_SEAT_GPU_OPTIONS: dict[str, int] = {"num_gpu": 999}


def is_local_ollama_model(model: str) -> bool:
    """Whether this tag runs on the cards here, rather than at ollama.com.

    A `:cloud` tag is proxied by the local daemon and holds no VRAM, so it
    neither needs the GPU options nor competes for the cards. Spelled the same
    way `get_agent_status` decides `remote`.
    """
    return not model.endswith((":cloud", "-cloud"))


def unload_ollama_model(model: str) -> bool:
    """Ask the daemon to drop `model` from VRAM now. False if it could not be asked.

    `keep_alive: 0` with an empty prompt is Ollama's own spelling of an unload;
    it returns once the runner is gone rather than scheduling it, which is what
    makes it usable as "the cards are free from here on".
    """
    try:
        request = urllib.request.Request(
            f"{_ollama_base_url()}/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30.0):
            return True
    except Exception:
        # An unload that fails costs a slower load, never a wrong answer, so
        # nothing here raises: the caller is about to do its real work either
        # way and a card it could not free is the state it was already in.
        return False


def free_the_cards_for(model: str) -> list[str]:
    """Unload every locally-resident model but `model`; return what went.

    Serializing the work in time is not enough on its own, which is the whole
    reason this exists. The daemon keeps a runner resident for five minutes
    after its last token, so a seat that finished its turn still holds ~3 GB of
    a 6 GB pool while the embedder tries to load 4.7 GB beside it -- and the
    loser of that is not slowed down but failed, as the 07:02:10 journal entry
    shows. Evicting is what turns "one at a time" into "all of the cards".

    A `:cloud` tag is never unloaded because it was never loaded, and the model
    about to be used is kept so this cannot evict its own caller. Reload is the
    price: ~5s for the embedder and ~25s for a 9B seat, paid once per handover
    rather than once per call, since a model already alone on the cards finds
    nothing here to evict.
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

    `num_gpu: 999` trades a graceful split for a hard failure where the model
    does not fit, which is the right trade for the embedder -- a corpus built
    on a split placement is a corpus with different vectors -- and the wrong
    one for a seat, where it would fail the run outright. qwen3.8 is 17 GB
    against 6 GB of cards and can never fit. So the shapes that mean "it did
    not fit" are named, and the seat retries without forcing; every other
    failure is the seat's real failure and is raised as it always was.
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


# Why the last call to a seat failed, if it did. A key can be present and the
# seat still unusable -- out of credits, expired, revoked, wrong workspace --
# and only a real call finds that out. Recording the outcome here is what lets
# the console stop claiming a seat is live without spending a probe request on
# every five-second status poll.
_seat_failures: dict[str, str] = {}


def _failure_reason(exc: Exception) -> str:
    """Turn a provider exception into something a seat card can show."""
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
    # httpx raises ReadTimeout with an empty message, so the class name is the
    # only thing that identifies it -- without this a timed-out seat showed a
    # blank chip, which reads as "fine" rather than "gave up after 120s".
    if "timeout" in lowered or "timed out" in lowered or "Timeout" in type(exc).__name__:
        return f"No response within {int(LLM_TIMEOUT_SECONDS)}s"
    if "not found" in lowered and "model" in lowered:
        return "Model not available on this account"
    if "connect" in lowered or "connection" in lowered:
        return "Provider unreachable"

    return message[:90].strip()


class _SeatLLM:
    """Wraps a seat's chat model so its failures are visible in the console.

    Transparent apart from `invoke`: every other attribute passes through to
    the wrapped model.

    It is also the one place every seat call passes through, which is why the
    cards are arbitrated here rather than in each node. A call takes
    `GPU_ARBITER`, so no seat is talking to a model while the embedder is
    working and no two seats are loading models at once; a locally-run tag
    additionally evicts whatever else is resident, because the daemon holds a
    finished runner for five minutes and two models do not fit in 6 GB.

    `build` is kept beside the built model so a forced load that did not fit
    can be rebuilt unforced -- the fallback that lets `num_gpu: 999` be the
    default without making an oversized seat a failed run.

    `provider` is the seat's provider, which `bind_tools` needs: whether a
    model can call tools is asked of the Ollama daemon, and only of it.
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

        Sticky: the seat stays unforced for the rest of its life rather than
        paying a failed load per call. A seat is rebuilt per node turn, so the
        forced attempt is retried on the next turn -- which is what should
        happen, since the reason it did not fit is usually another model that
        has since been evicted.
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
                result = self._inner.invoke(*args, **kwargs)
            except Exception as exc:
                if local and _is_gpu_fit_failure(exc) and self._unforce():
                    try:
                        result = self._inner.invoke(*args, **kwargs)
                    except Exception as retried:
                        _seat_failures[self._agent] = _failure_reason(retried)
                        raise
                else:
                    _seat_failures[self._agent] = _failure_reason(exc)
                    raise
        # A call that works clears an older failure, so a seat recovers on its
        # own once credits are topped up or the daemon comes back.
        _seat_failures.pop(self._agent, None)
        return result

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_SeatLLM":
        """Bind tools, keeping the wrapper so the bound model still reports.

        Without this the bound runnable would come back unwrapped through
        `__getattr__`, and every failure the Builder hit while calling tools
        would be invisible to `get_agent_status`. Seats whose model cannot call
        tools at all (`StubLLM`) raise AttributeError here on purpose, so a
        caller wanting the no-tools path catches AttributeError around the
        call itself -- `hasattr` is always True once this method exists.

        The bound seat carries `build` and the tools forward, or an unforced
        rebuild would come back without the belt the Builder is mid-turn with.

        An Ollama tag the daemon says cannot call tools raises here too. The
        attribute check alone never caught one: `bind_tools` is ChatOllama's
        own method, so binding succeeds for every tag, and it is the daemon
        that refuses -- answering the first call with a 400 ("does not support
        tools") that failed the whole run, where a model without tools was
        meant to report its work and change nothing. Asked through
        `tool_support`, which caches the daemon's answer; "could not ask"
        (None) binds as before, since it is not a no.
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

    Anthropic removed temperature/top_p/top_k on the 4.6-and-later families
    (Opus 5, Sonnet 5, Opus 4.6+) -- sending one is rejected with a 400, which
    reads like a credentials problem and is not one. Legacy `claude-3-*` and
    the 4.5 models still take it, as do Ollama and OpenAI.
    """
    if provider != "anthropic":
        return True
    return model.startswith("claude-3") or "-4-5" in model


# Whether a seat thinks before it answers, until someone switches it. Off, so a
# fresh console starts with every thinking box unticked. That is a real change
# on the wire, not a relabelling: a switchable model is always sent the flag
# (`_thinking_for_call`), and leaving it out means *on* for several of them --
# a switchable model given no flag pays for reasoning tokens langchain_ollama
# then discards outright, so the cost is paid and nothing shows it. Opus 5 and
# Sonnet 5 likewise think when the parameter is left out. Tick a seat's box
# for the hard steps; the choice lasts until the server restarts.
DEFAULT_THINKING = False

# The ceiling on a pre-4.6 Claude model's thinking, the only way those models
# can be told to think at all. Anthropic's floor is 1024; this is kept low
# because a whole node turn is bounded by NODE_DEADLINE_SECONDS and a single
# call by LLM_TIMEOUT_SECONDS, and thinking tokens are spent inside both.
THINKING_BUDGET_TOKENS = int(os.getenv("THINKING_BUDGET_TOKENS", "4096"))

# Claude models that think on every call. Anthropic rejects an explicit
# "disabled" on them -- Opus 5.5 at every effort level -- so the card shows the
# box ticked and locked rather than offering a switch whose "off" would fail
# every call. Opus 5.5 was missing, so a seat moved onto it was sent
# `{"type": "disabled"}` by default and every call came back a 400.
_ALWAYS_THINKING_CLAUDE = ("claude-fable", "claude-mythos", "claude-opus-5-5")

# Claude models whose "off" is not spelled `disabled`. Sonnet 5.5 answers
# `{"type": "disabled"}` with a 400 and turns thinking off with `between_tools`,
# which takes no other field and is accepted at the default effort.
_CLAUDE_THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}

# Both orders Anthropic has named models in (`claude-3-7-sonnet`,
# `claude-opus-4-1`). The minor version is one or two digits, so a date suffix
# is not read as one: `claude-sonnet-4-20250514` is 4.0, not 4.20250514.
_CLAUDE_VERSION = re.compile(r"claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2})(?!\d))?")

ThinkingSupport = Literal["switch", "never", "always", "unknown"]


def _claude_version(model: str) -> tuple[int, int] | None:
    """(major, minor) of a Claude model id, or None if it is not one."""
    match = _CLAUDE_VERSION.search(model)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _claude_thinking(model: str, on: bool) -> dict[str, Any] | None:
    """The `thinking` parameter that switches a Claude model on or off.

    `None` means leave the parameter out. There are two request shapes, split
    at 4.6. From there `adaptive` is the only way on, and an explicit "off" has
    to be sent, because Opus 5 and Sonnet 5 think when the parameter is
    absent -- `disabled` for most, and `_CLAUDE_THINKING_OFF`'s spelling for a
    model that rejects that one. Before 4.6 a token budget is the only way on
    (and `budget_tokens` is a 400 on Opus 5), while absence already means off.
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


@functools.lru_cache(maxsize=64)
def _openai_reasons(model: str) -> bool | None:
    """Whether the profile langchain_openai ships for this model says it reasons.

    Building the chat model is the public way to read that profile and makes
    no request -- nothing goes on the wire before `invoke` -- so a placeholder
    key is enough. `None` when the package has no profile for the model.
    Cached because profiles are data in the installed package and the status
    poll asks every five seconds.
    """
    kwargs: dict[str, Any] = {"model": model, "api_key": SecretStr("unused")}
    try:
        from langchain_openai import ChatOpenAI

        profile = ChatOpenAI(**kwargs).profile
    except Exception:
        return None
    reasons = (profile or {}).get("reasoning_output")
    return reasons if isinstance(reasons, bool) else None


def tool_support(provider: str, model: str) -> tuple[bool | None, str]:
    """Whether a model can call tools, and why it matters when it cannot.

    Only the Builder is offered any (`BUILDER_TOOLS`), and it is the one seat
    whose work *is* the tool calls: `files_changed` is appended only when a
    write tool reports success, never from the model's prose. So a Builder on a
    model without them reports in full and changes nothing -- the `StubLLM`
    failure with a live seat behind it, which no chip in the console would
    otherwise show, because the seat *is* live and every call *does* succeed.

    It earns a place now because `AGENT_LLM_OPTIONS` offers its first such
    model: dolphin reports `completion` alone, while every tag offered before
    it reported `tools`. The daemon's answer is already cached by
    `ollama_model_capabilities`, which `thinking_support` asks on the same poll,
    so this costs no extra request.

    `None` is "could not ask", kept apart from False for the reason
    `thinking_support` keeps `unknown` apart from `never`: a daemon down for
    thirty seconds must not read as four seats that lost a capability.
    """
    if provider == "ollama":
        caps = ollama_model_capabilities(model)
        if caps is None:
            return None, ""
        return ("tools" in caps), ""

    # Anthropic and OpenAI reject a tool call at the API rather than silently
    # ignoring it, so there is no quiet failure of this shape to warn about.
    return True, ""


def thinking_support(provider: str, model: str) -> tuple[ThinkingSupport, str]:
    """Whether a model can think, and whether the console may switch it.

    Returns the verdict and, when the card cannot offer the switch, the reason
    to show on hover. Each provider is asked the question the way it can
    actually answer it:

    - An Ollama tag answers for itself, through the daemon's capabilities.
    - A Claude model is read off its version: thinking arrived with 3.7, and
      every model since has it. Not langchain's profile, which has no entry
      for 3.7 and none for a model released after the installed package, so it
      would lock the switch on exactly the models someone just added.
    - An OpenAI model is read off langchain's profile. Its reasoning is not
      wired to this switch -- effort levels, not on and off -- so a model that
      reasons reads `unknown` rather than claiming either state.

    `unknown` is its own verdict, never folded into `never`: "could not ask" is
    not "cannot think", and a daemon that is down for thirty seconds must not
    read as four models that lost a capability.
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

    if provider == "openai":
        reasons = _openai_reasons(model)
        if reasons is False:
            return "never", f"{model} cannot think"
        return "unknown", (
            f"{model}'s reasoning is not switchable from the console"
        )

    return "unknown", f"thinking is not switchable for {provider}"


# Runtime per-agent thinking choices set from the console, for the life of the
# process like `_agent_llm_overrides`. Kept per seat rather than per model, so
# unticking the Builder survives moving the Builder to another model.
_agent_thinking: dict[str, bool] = {}


def get_agent_thinking(agent: str) -> bool:
    """The thinking a seat asks for when its model can be switched."""
    return _agent_thinking.get(agent, DEFAULT_THINKING)


def set_agent_thinking(agent: str, on: bool) -> None:
    """Switch one seat's thinking on or off, for the life of the process.

    Refused for a seat whose model offers no switch, and said so, rather than
    stored and ignored: a request that changes nothing should not come back
    looking as though it worked.

    Unlike moving a seat, this keeps any failure recorded against it. Thinking
    fixes no credit balance, rate limit or unreachable daemon, and clearing
    the chip here would let a dead seat be made to look live by clicking a box.
    """
    info = get_agent_model_info(cast("AgentName", agent))
    support, reason = thinking_support(info["provider"], info["model"])
    if support != "switch":
        raise ValueError(f"Thinking cannot be switched on this seat: {reason}")
    _agent_thinking[agent] = on


def _thinking_for_call(agent: str) -> bool | None:
    """The thinking flag the seat's next call sends, or None to send none.

    Only a switchable model gets a flag. Ollama refuses `think` for a tag
    without the capability, so sending the default to one would fail every
    call; and when the capability is unknown, leaving the model to its own
    default is exactly what every call did before the switch existed.
    """
    info = get_agent_model_info(cast("AgentName", agent))
    support, _ = thinking_support(info["provider"], info["model"])
    return get_agent_thinking(agent) if support == "switch" else None


# Runtime per-agent LLM selections set from the console. These override the
# environment-variable defaults for the lifetime of the process.
_agent_llm_overrides: dict[str, dict[str, str]] = {}


def set_agent_llm(agent: str, provider: str, model: str) -> None:
    """Set the LLM for an agent at runtime.

    The selection is stored in memory only; it does not modify environment
    variables or persist across server restarts.

    Moving a seat clears any failure recorded against it. A recorded failure
    describes the seat that produced it, so leaving it in place made the new
    seat inherit the old one's verdict -- an Architect moved off Anthropic
    still read "Anthropic credit balance too low", which is exactly the
    reading that sends someone to buy credits they do not need. Re-selecting
    the seat it already has is not a move and keeps the failure, so a dead
    seat cannot be made to look live by picking it again.
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
    """Get an LLM instance.

    Args:
        provider: LLM provider ("anthropic", "ollama" or "openai").
                  Auto-detected from model name/env if not specified.
        model: Model name (default from provider-specific env var).
        temperature: Sampling temperature, where the model accepts one.
        base_url: Optional API base URL override.
        api_key: Optional API key override.
        timeout: Seconds one call may take; `LLM_TIMEOUT_SECONDS` if omitted.
                 Each provider spells this differently, hence the three
                 separate keyword names below.
        force_gpu: Whether a locally-run Ollama tag is told to put every
                   layer on the cards (`OLLAMA_SEAT_GPU_OPTIONS`). Ignored by
                   every other provider and by `:cloud` tags. `_SeatLLM` turns
                   it off to rebuild a seat whose forced load did not fit.
        thinking: Whether the model thinks before answering. `None` sends no
                  flag and leaves it to the model, which is what every call
                  did before the console could switch it. Pass a bool only for
                  a model `thinking_support` calls switchable. Not wired for
                  OpenAI, whose reasoning is set by effort, not on and off.

    Returns:
        Chat model instance, or `StubLLM` when the provider needs a key and
        none is configured. Callers that need to know which of the two they
        got should ask `get_agent_status()` rather than inspect the result.

    Environment variables:
        ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        OLLAMA_BASE_URL, OLLAMA_MODEL
        OPENAI_API_KEY, OPENAI_MODEL  (optional)
    """
    provider = provider or _detect_provider(model)
    timeout = LLM_TIMEOUT_SECONDS if timeout is None else timeout

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No key: the daemon holds the ollama.com credentials for `:cloud`
        # tags, so there is nothing for this process to authenticate with.
        # `client_kwargs` reaches the httpx client the ollama SDK builds; there
        # is no `timeout` field on ChatOllama itself.
        #
        # `reasoning=True` rather than the model's own default when thinking
        # is on: both think, but under the default langchain_ollama drops the
        # daemon's `thinking` field, so the reasoning is paid for and thrown
        # away. True keeps it in `additional_kwargs`, where the Builder's tool
        # loop hands it back to the model on the next turn.
        tag = str(model or os.getenv(
            "OLLAMA_MODEL", "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"
        ))

        # A locally-run tag is told to put every layer on the cards; a
        # `:cloud` tag is sent nothing, since the options would be forwarded
        # to ollama.com to describe hardware that is not theirs. `force_gpu`
        # is how `_SeatLLM` rebuilds this seat unforced after a load that did
        # not fit -- see `_is_gpu_fit_failure`.
        # None rather than a missing keyword: langchain_ollama builds the
        # daemon's `options` from the fields that are not None, so the two are
        # the same request and this one type-checks.
        num_gpu = (
            OLLAMA_SEAT_GPU_OPTIONS["num_gpu"]
            if force_gpu and is_local_ollama_model(tag)
            else None
        )

        return ChatOllama(
            model=tag,
            temperature=temperature,
            base_url=base_url or _ollama_base_url(),
            client_kwargs={"timeout": timeout},
            reasoning=thinking,
            num_gpu=num_gpu,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = str(model or os.getenv("ANTHROPIC_MODEL", "claude-opus-5"))
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
        # A model that is thinking takes no temperature: the families that
        # still accept one reject anything but the default once thinking is on.
        thinks = thinking_param is not None and thinking_param["type"] not in (
            "disabled", "between_tools"
        )
        if _accepts_temperature("anthropic", model_name) and not thinks:
            kwargs["temperature"] = temperature
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    # openai (optional cloud provider)
    from langchain_openai import ChatOpenAI

    model_name = str(model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return StubLLM()
    kwargs = {
        "model": str(model_name),
        "temperature": temperature,
        "api_key": SecretStr(key),
        "request_timeout": timeout,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _detect_provider(model: str | None) -> Provider:
    """Detect a provider from the model's name, then from the environment.

    The name is read first because it is the more specific fact. `gpt-4o` is an
    OpenAI model whatever keys the environment happens to hold, and consulting
    the keys first sent `BUILDER_MODEL=gpt-4o` to Anthropic on any machine with
    an ANTHROPIC_API_KEY -- a seat that then failed every call with a model
    Anthropic does not have. The environment decides only for a name no
    provider's naming claims, or no name at all: Anthropic's key or model
    first, then OpenAI's, then Anthropic for no name and OpenAI otherwise.

    An Ollama tag is recognised by its colon (`qwen3.8:latest`). A bare name
    without one cannot be told from anything else by its spelling, so name the
    provider for such a seat: `{ROLE}_PROVIDER=ollama`.
    """
    if model:
        lowered = model.lower()
        if ":" in model:
            return "ollama"
        if "claude" in lowered:
            return "anthropic"
        if lowered.startswith(("gpt-", "chatgpt", "o1", "o3", "o4")):
            return "openai"

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_MODEL"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL"):
        return "openai"
    return "anthropic" if not model else "openai"


def _resolve_seat(agent: str) -> dict[str, str | None]:
    """Resolve one agent's provider, model, base URL and key.

    Precedence: a console override, then per-agent environment variables, then
    the agent's default seat. `get_agent_llm`, `get_agent_model_info` and
    `get_agent_status` all route through here -- if they resolved seats
    independently the console could name one model while the run used another,
    which is the failure this function exists to prevent.

    Per-agent environment variables, for each of ARCHITECT, PLANNER,
    RESEARCHER and BUILDER:
        {ROLE}_PROVIDER, {ROLE}_MODEL, {ROLE}_BASE_URL, {ROLE}_API_KEY
    """
    override = _agent_llm_overrides.get(agent)
    if override:
        chosen = override["provider"]
        return {
            "provider": chosen,
            "model": override["model"],
            # A console selection carries no key of its own, so the provider's
            # own credentials apply -- the same ones the default seat uses.
            "base_url": os.getenv(f"{chosen.upper()}_BASE_URL"),
            "api_key": None,
        }

    prefix = agent.upper()
    provider: str | None = os.getenv(f"{prefix}_PROVIDER")
    model: str | None = os.getenv(f"{prefix}_MODEL")

    if not provider and not model:
        # Deliberately not consulting a provider-wide {PROVIDER}_MODEL here:
        # the four seats run four different models on purpose, and a single
        # OLLAMA_MODEL would silently collapse three of them onto one. Retune
        # a seat with {ROLE}_MODEL or the console dropdown instead.
        seat = DEFAULT_SEATS.get(agent, DEFAULT_SEATS["builder"])
        provider, model = seat["provider"], seat["model"]
    elif provider and not model:
        model = _DEFAULT_AGENT_MODELS.get((provider, agent))
    elif model and not provider:
        provider = _detect_provider(model)

    return {
        "provider": provider,
        "model": model,
        "base_url": os.getenv(f"{prefix}_BASE_URL"),
        "api_key": os.getenv(f"{prefix}_API_KEY"),
    }


def get_agent_llm(agent: AgentName, temperature: float = 0.1) -> Any:
    """Get the LLM configured for a specific agent role.

    Default seats (local first -- see DEFAULT_SEATS):
        Architect  -> Ollama    dolphin-2.9.1 9B (leading authority)
        Planner    -> Ollama    dolphin-2.9.1 9B
        Researcher -> Ollama    dolphin-2.9.1 9B
        Builder    -> Ollama    qwen3.8:latest (the local tag with tools)
    """
    seat = _resolve_seat(agent)

    # The seat is built through a factory rather than once, so `_SeatLLM` can
    # build it again unforced when a forced offload does not fit the cards.
    # Everything but the placement is captured here, which is what keeps the
    # rebuilt seat identical in every other respect -- same model, same
    # thinking flag, same key.
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
    """Resolve an agent's provider/model without instantiating an LLM."""
    seat = _resolve_seat(agent)
    return {
        "provider": seat["provider"] or "anthropic",
        "model": seat["model"] or "claude-opus-5",
    }


def get_agent_status(agent: AgentName) -> dict[str, Any]:
    """Resolve an agent's seat and say whether it can actually run.

    `get_llm` falls back to `StubLLM` when a key is missing, while
    `get_agent_model_info` keeps reporting the configured model either way --
    so without this the console shows a model name while canned text comes out
    of the run. `live` is the field that tells the truth about that, and it is
    what drives the NO KEY chip and the offline banner.
    """
    info = get_agent_model_info(agent)
    provider, model = info["provider"], info["model"]

    # `stubbed` and `live` are different failures and must not be conflated.
    # A seat with no key at all silently becomes StubLLM and the run completes
    # with canned text; a seat with a key that does not work fails the run
    # outright. The console words those two cases differently.
    live, reason, badge, stubbed = True, "", "", False

    # A real failure from the last call outranks every static check: the key
    # can be present and correct and the seat still unable to run.
    failure = _seat_failures.get(agent)
    if failure:
        live, reason, badge = False, failure, "FAILING"
    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        live, reason, badge, stubbed = False, "ANTHROPIC_API_KEY not set", "NO KEY", True
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        live, reason, badge, stubbed = False, "OPENAI_API_KEY not set", "NO KEY", True
    elif provider == "ollama":
        tags = list_ollama_models()
        if not tags:
            live, reason, badge = False, "Ollama daemon unreachable", "OFFLINE"
        # Compared as tags, not strings: the daemon lists `qwen3.8:latest`, and
        # a seat configured as `qwen3.8` is that model, not a missing one.
        elif not any(_same_ollama_tag(model, tag) for tag in tags):
            live, reason, badge = False, f"{model} not pulled", "NOT PULLED"

    # Where the prompt goes, read off the seat rather than guessed: an Ollama
    # tag ending `:cloud` is proxied to ollama.com by the local daemon, so the
    # transport is local but the prompt still leaves the machine.
    remote = provider in ("anthropic", "openai") or model.endswith((":cloud", "-cloud"))

    # `thinking` is what the next call will do, not what was asked for: a
    # switchable model is always sent the flag, so the box cannot disagree
    # with the seat. None when nobody can say -- see `thinking_support`.
    support, thinking_note = thinking_support(provider, model)
    thinking = {
        "switch": get_agent_thinking(agent),
        "always": True,
        "never": False,
    }.get(support)

    # Reported for every seat, but only the Builder is warned: the other three
    # are offered no tools at all, so a model without them is the right seat
    # for them rather than a defect. False and None are kept apart -- a daemon
    # that could not be asked must not accuse a model of anything.
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
    """Stub LLM for testing without API key.

    Returns responses in the 4-Agent System format.
    """

    def invoke(self, messages: list[Any]) -> Any:
        """Return canned responses for testing."""
        from langchain_core.messages import AIMessage

        # Role comes from the system message; everything else is data.
        # The Researcher injects retrieved documents into the Builder's prompt,
        # and a retrieved document can say anything -- an uploaded copy of a
        # prompt file puts "You are the Architect" inside a Builder call.
        # Reading the role off user content makes the stub answer as the wrong
        # agent.
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

        # Detect which agent is being called. Every prompt names the other
        # roles in order to route between them, so a bare role keyword matches
        # all four -- the Builder's prompt says "Planner" twice. Identify on
        # the self-identifying opening instead, and only fall back to the
        # looser phrases when a caller supplied its own prompt.
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

        # For the Planner, decide whether the *user goal* asks for research.
        # Ignore the state-injection block, which contains a "Research:" label.
        user_goal_match = re.search(
            r"User goal:\s*(.+)", last_content, re.IGNORECASE | re.DOTALL
        )
        user_goal = user_goal_match.group(1).lower() if user_goal_match else last_lower
        needs_research = "research" in user_goal

        if is_architect:
            # The Architect runs twice per cycle: once to set direction, and
            # again as the approval gate. A populated builder report is what
            # separates the two.
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
