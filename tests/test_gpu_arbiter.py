"""The cards: one model at a time, and every layer of it on the GPU.

Two claims, measured on 2026-09-20 on two 3 GB GTX 1060s and pinned here.

*Every layer.* Ollama's own estimator left ~1.1 GB unused on each card and ran
37% of a 9B Q4 seat on the CPU, whatever window it was asked for; forced with
`num_gpu`, the same model loaded 100% on the cards at the full 4,096-token
window. Seats were built with no options at all, so every local seat took the
estimate.

*One at a time.* qwen3-embedding is 4.7 GB and a 9B seat is 5.3 GB, against a
6 GB pool. The daemon's answer to being asked for both is a failed load, not a
split -- the journal caught one at 07:02:10, timing out after twenty seconds
spent trying to pack 48 layers onto card 1 because the embedder held card 0.
"""

import threading
import time

import pytest

from langgraph_agent import config
from langgraph_agent.config import (
    OLLAMA_SEAT_GPU_OPTIONS,
    _is_gpu_fit_failure,
    _SeatLLM,
    free_the_cards_for,
    is_local_ollama_model,
)
from langgraph_agent.control import GpuArbiter

# --- every layer on the cards ------------------------------------------------


@pytest.mark.parametrize(
    "tag, local",
    [
        ("qwen3.8:latest", True),
        ("hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M", True),
        ("hf.co/RavichandranJ/Dolphin3-Cyber-8B-GGUF:Q5_K_M", True),
        ("kimi-k3:cloud", False),
    ],
)
def test_only_a_local_tag_is_run_on_these_cards(tag: str, local: bool) -> None:
    """A `:cloud` tag is proxied to ollama.com and holds no VRAM here."""
    assert is_local_ollama_model(tag) is local


def test_a_local_seat_is_told_to_put_every_layer_on_the_cards() -> None:
    """The lever the estimator ignores. Without it, 37% of the model ran on the CPU."""
    seat = config.get_llm(provider="ollama", model="qwen3.8:latest")
    assert seat.num_gpu == OLLAMA_SEAT_GPU_OPTIONS["num_gpu"]


def test_a_cloud_seat_is_sent_no_placement_at_all() -> None:
    """Forwarding a layer count to ollama.com describes hardware that is not theirs."""
    seat = config.get_llm(provider="ollama", model="kimi-k3:cloud")
    assert seat.num_gpu is None


def test_the_window_is_left_to_the_seat() -> None:
    """Measured: `num_ctx` never moved the split, so capping it would lose context free."""
    seat = config.get_llm(provider="ollama", model="qwen3.8:latest")
    assert seat.num_ctx is None


def test_force_gpu_off_builds_the_unforced_seat() -> None:
    """The fallback's other half: the daemon chooses the split again."""
    seat = config.get_llm(provider="ollama", model="qwen3.8:latest", force_gpu=False)
    assert seat.num_gpu is None


# --- one model at a time -----------------------------------------------------


def test_the_arbiter_serializes_two_holders() -> None:
    """The whole point: a seat and the embedder are never on the cards together.

    The peak is counted rather than asserted inside the threads, because an
    assertion that fails on a worker thread kills that thread and leaves the
    test passing -- which is how the first version of this guard was asleep.
    """
    arbiter = GpuArbiter()
    counter = threading.Lock()
    concurrent = 0
    peak = 0
    inside = threading.Semaphore(0)

    def hold(name: str, seconds: float) -> None:
        nonlocal concurrent, peak
        with arbiter.exclusive(name):
            with counter:
                concurrent += 1
                peak = max(peak, concurrent)
            inside.release()
            time.sleep(seconds)
            with counter:
                concurrent -= 1

    first = threading.Thread(target=hold, args=("embedder", 0.3))
    first.start()
    inside.acquire()
    second = threading.Thread(target=hold, args=("seat:Builder", 0.05))
    second.start()
    inside.acquire()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert peak == 1, f"{peak} holders were on the cards at once"


def test_the_arbiter_reports_who_holds_the_cards() -> None:
    arbiter = GpuArbiter()
    assert arbiter.snapshot()["holder"] == ""
    with arbiter.exclusive("embedder"):
        assert arbiter.snapshot()["holder"] == "embedder"
    assert arbiter.snapshot()["holder"] == ""


def test_the_arbiter_is_reentrant_rather_than_deadlocking() -> None:
    """`_encode` reaches `_load_embedder`; a plain lock would hang the run for ever."""
    arbiter = GpuArbiter()
    with arbiter.exclusive("embedder"):
        with arbiter.exclusive("embedder"):
            assert arbiter.snapshot()["holder"] == "embedder"
        # The inner exit must not put the light out while the outer still holds.
        assert arbiter.snapshot()["holder"] == "embedder"
    assert arbiter.snapshot()["holder"] == ""


def test_a_waiter_never_runs_beside_the_holder() -> None:
    """However long the holder takes, the next model waits for the cards to be free."""
    arbiter = GpuArbiter()
    released = threading.Event()
    entered = threading.Event()
    overlapped: list[bool] = []

    def squat() -> None:
        with arbiter.exclusive("squatter"):
            entered.set()
            released.wait(5)

    def wait_for_the_cards() -> None:
        with arbiter.exclusive("seat:Builder"):
            overlapped.append(not released.is_set())

    holder = threading.Thread(target=squat, daemon=True)
    holder.start()
    entered.wait(5)
    waiter = threading.Thread(target=wait_for_the_cards, daemon=True)
    waiter.start()
    time.sleep(0.3)
    assert waiter.is_alive(), "the waiter ran while the cards were held"
    assert arbiter.snapshot()["waiting"] == 1
    released.set()
    waiter.join(5)
    holder.join(5)
    assert overlapped == [False]
    assert arbiter.snapshot() == {"holder": "", "held_for": 0.0, "waiting": 0}


# --- freeing the cards -------------------------------------------------------


def test_freeing_the_cards_drops_the_others_and_keeps_its_own(monkeypatch) -> None:
    """Serializing in time is not enough: the daemon holds a finished runner 5 minutes."""
    monkeypatch.setattr(
        config,
        "_ollama_ps",
        lambda: [
            {"name": "qwen3-embedding:latest"},
            {"name": "qwen3.8:latest"},
            {"name": "kimi-k3:cloud"},
        ],
    )
    unloaded: list[str] = []
    monkeypatch.setattr(
        config, "unload_ollama_model", lambda tag: (unloaded.append(tag), True)[1]
    )

    dropped = free_the_cards_for("qwen3-embedding:latest")

    assert unloaded == ["qwen3.8:latest"], unloaded
    assert dropped == ["qwen3.8:latest"]


def test_freeing_the_cards_survives_a_daemon_that_cannot_be_asked(monkeypatch) -> None:
    """An eviction that fails costs a slower load, never a wrong answer."""

    def refuse() -> list[dict[str, object]]:
        raise OSError("daemon down")

    monkeypatch.setattr(config, "_ollama_ps", refuse)
    assert free_the_cards_for("qwen3-embedding:latest") == []


# --- overflow to the CPU only when it truly will not fit ---------------------


@pytest.mark.parametrize(
    "message",
    [
        "cudaMalloc failed: out of memory",
        "timed out waiting for llama-server to start: context canceled",
        "unable to load model",
    ],
)
def test_a_load_that_did_not_fit_is_recognised(message: str) -> None:
    assert _is_gpu_fit_failure(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["API key rejected", "rate limit exceeded", "model not found"],
)
def test_an_ordinary_seat_failure_is_not_a_fit_failure(message: str) -> None:
    """Or a seat with no credits would be quietly retried on the CPU and still fail."""
    assert not _is_gpu_fit_failure(RuntimeError(message))


class _Inner:
    """A seat's model, standing in for ChatOllama."""

    def __init__(self, model: str, num_gpu: int | None, fails: Exception | None = None):
        self.model = model
        self.num_gpu = num_gpu
        self._fails = fails
        self.calls = 0

    def invoke(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        if self._fails is not None:
            raise self._fails
        return f"ok:{self.num_gpu}"


def _no_daemon(monkeypatch) -> None:
    monkeypatch.setattr(config, "free_the_cards_for", lambda model: [])


def test_a_seat_that_will_not_fit_falls_back_to_the_daemons_own_split(monkeypatch) -> None:
    """qwen3.8 is 17 GB against 6 GB of cards: forced, it can only ever fail."""
    _no_daemon(monkeypatch)
    forced = _Inner("qwen3.8:latest", 999, RuntimeError("cudaMalloc failed: out of memory"))
    unforced = _Inner("qwen3.8:latest", None)

    seat = _SeatLLM("Builder", forced, lambda force_gpu: unforced)

    assert seat.invoke("hello") == "ok:None"
    assert forced.calls == 1 and unforced.calls == 1


def test_the_fallback_is_sticky_for_the_life_of_the_seat(monkeypatch) -> None:
    """A seat already unforced must not be rebuilt again on the next fit failure.

    Without the `_force_gpu` check the seat rebuilds on every call that fails
    that way, so a machine that genuinely cannot run the model pays a fresh
    load per call and still fails. The second model here fails the same way the
    first did, which is the only shape that reaches the second rebuild.
    """
    _no_daemon(monkeypatch)
    forced = _Inner("qwen3.8:latest", 999, RuntimeError("out of memory"))
    unforced = _Inner("qwen3.8:latest", None, RuntimeError("out of memory"))
    builds = []

    def build(force_gpu: bool) -> _Inner:
        builds.append(force_gpu)
        return unforced

    seat = _SeatLLM("Builder", forced, build)

    with pytest.raises(RuntimeError, match="out of memory"):
        seat.invoke("one")
    with pytest.raises(RuntimeError, match="out of memory"):
        seat.invoke("two")

    assert builds == [False], f"the seat was rebuilt {len(builds)} times"
    assert forced.calls == 1, "the retired forced seat was called again"


def test_an_ordinary_failure_is_raised_rather_than_retried(monkeypatch) -> None:
    _no_daemon(monkeypatch)
    forced = _Inner("qwen3.8:latest", 999, RuntimeError("API key rejected"))
    unforced = _Inner("qwen3.8:latest", None)
    seat = _SeatLLM("Builder", forced, lambda force_gpu: unforced)

    with pytest.raises(RuntimeError, match="API key rejected"):
        seat.invoke("hello")
    assert unforced.calls == 0


def test_a_cloud_seat_never_falls_back_and_never_evicts(monkeypatch) -> None:
    """A `:cloud` failure is ollama.com's, and has nothing to do with these cards."""
    freed: list[str] = []
    monkeypatch.setattr(
        config, "free_the_cards_for", lambda model: (freed.append(model), [])[1]
    )
    forced = _Inner("kimi-k3:cloud", None, RuntimeError("out of memory"))
    unforced = _Inner("kimi-k3:cloud", None)
    seat = _SeatLLM("Planner", forced, lambda force_gpu: unforced)

    with pytest.raises(RuntimeError):
        seat.invoke("hello")
    assert unforced.calls == 0
    assert freed == []


def test_a_local_seat_frees_the_cards_before_it_calls(monkeypatch) -> None:
    freed: list[str] = []
    monkeypatch.setattr(
        config, "free_the_cards_for", lambda model: (freed.append(model), [])[1]
    )
    seat = _SeatLLM("Builder", _Inner("qwen3.8:latest", 999))

    seat.invoke("hello")

    assert freed == ["qwen3.8:latest"]


def test_a_bound_seat_keeps_its_belt_through_the_fallback(monkeypatch) -> None:
    """The Builder is mid-turn: a seat rebuilt without its tools cannot finish one."""
    _no_daemon(monkeypatch)

    class _Bindable(_Inner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bound: object = None

        def bind_tools(self, tools, **kwargs):
            clone = _Bindable(self.model, self.num_gpu, self._fails)
            clone.bound = tools
            return clone

    forced = _Bindable("qwen3.8:latest", 999, RuntimeError("out of memory"))
    unforced = _Bindable("qwen3.8:latest", None)
    seat = _SeatLLM("Builder", forced, lambda force_gpu: unforced)

    belted = seat.bind_tools(["filesystem_write"])
    assert belted.invoke("hello") == "ok:None"
    assert belted._inner.bound == ["filesystem_write"]
