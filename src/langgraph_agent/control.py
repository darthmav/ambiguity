"""The stop signal for the run currently in flight.

Deliberately a module-level object rather than a field on `AgentState`. The
graph is compiled without a checkpointer, so LangGraph's `update_state` /
`interrupt` machinery is unavailable and nothing outside a node can write into a
state the graph is already streaming. A flag the nodes read is the only channel
that reaches a run in progress.

The stop is cooperative. Nothing here interrupts anything: it sets a flag, and
the checkpoints in `nodes.py` and `serve.py` decline to start the *next* piece of
work. Work already in flight -- a `filesystem_write`, a staged commit, a test
run, a model call -- always finishes, because the alternative is a half-written
file and a worker still writing into the project after the node returned.
"""

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# Why a run ended, when the caller did not say.
DEFAULT_STOP_REASON = "Stopped from the console."


class RunControl:
    """An armed/stopped flag for one run, shared across threads.

    The run executes on the HTTP request thread that started it, so the stop
    arrives on a different thread -- which is why `ThreadingHTTPServer` is
    load-bearing and why this is guarded rather than a plain bool.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._run_id = ""
        self._reason = ""

    def arm(self, run_id: str) -> None:
        """Take ownership of a new run, clearing any previous stop."""
        with self._lock:
            self._run_id = run_id
            self._reason = ""
            self._event.clear()

    def disarm(self) -> None:
        """Release the run. A stop after this point has nothing to act on."""
        with self._lock:
            self._run_id = ""
            self._reason = ""
            self._event.clear()

    def stop(self, run_id: str = "", reason: str = "") -> bool:
        """Ask the armed run to stop. False if there is nothing to stop.

        A `run_id` that does not match the armed run is refused rather than
        applied to whatever is running now: the console can hold a stale id
        across a page reload, and a Stop meant for a finished run must not kill
        the run that replaced it. An empty `run_id` means "whatever is armed",
        which is what a caller with no id (curl, a test) wants.
        """
        with self._lock:
            if not self._run_id:
                return False
            if run_id and run_id != self._run_id:
                return False
            self._reason = reason or DEFAULT_STOP_REASON
            self._event.set()
            return True

    def stopped(self) -> bool:
        """Whether the armed run has been asked to stop.

        Lock-free on purpose: this is called at the top of every Builder tool
        turn and before every verified file.
        """
        return self._event.is_set()

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def run_id(self) -> str:
        with self._lock:
            return self._run_id


# The single control for this process. The server runs one graph object and one
# run at a time, so one control is the whole story.
RUN_CONTROL = RunControl()


# How many of a run's most recent turns `NodeActivity.turns` hands back. The
# console asks several times a second and only ever needs the last few; the
# bound is what stops a run with dozens of supersteps growing every reply.
TURN_RECORD = 16


@dataclass
class _Turn:
    """One node's time on the stack. `ended` is None while it is still there."""

    number: int
    node: str
    started: float
    ended: float | None = None


class NodeActivity:
    """Which seat is executing *right now*, for the console's seat lights.

    Deliberately separate from `_run_progress["node"]` in `serve.py`, which is
    fed by `graph.stream` and therefore names the node that has just *finished*
    -- LangGraph yields an update when a superstep completes, not when one
    starts. That is the right value for the feed, which lists what happened, and
    the wrong one for a light meaning "this seat is working": it lights the
    previous seat for the whole of the next seat's turn, so the slowest node in
    the run is the one node whose light never comes on. A stalled Architect
    would show as a busy Builder, which is precisely backwards.

    Set from the graph rather than from inside the node bodies: every node has
    several return paths (a stop, a deadline fallback, the ordinary one), so a
    wrapper with a `finally` is the only way a light cannot be left on by an
    exit nobody thought about.

    The seat working now is not enough on its own, because the console can only
    sample it: a poll sees the node on the stack at the instant it lands, and a
    turn that starts and ends between two polls is never seen at all. Measured
    through the real page with the console polling once a second, a 0.4s
    Researcher turn never lit. So every turn is also recorded -- numbered, and
    kept after it ends -- and `turns()` hands the recent ones back, which is
    what lets the console light a turn it only heard about afterwards.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node = ""
        self._since = 0.0
        self._count = 0
        self._turns: deque[_Turn] = deque(maxlen=TURN_RECORD)

    def begin_run(self) -> None:
        """Forget the previous run's turns, as a new run is claimed.

        The numbering carries on rather than starting again: a console still
        holding the last number it saw must read every turn of the new run as
        one it has not shown.
        """
        with self._lock:
            self._node = ""
            self._since = 0.0
            self._turns.clear()

    def enter(self, node: str) -> None:
        with self._lock:
            self._count += 1
            self._node = node
            self._since = time.monotonic()
            self._turns.append(_Turn(self._count, node, self._since))

    def leave(self, node: str) -> None:
        """Clear the light, unless someone else already claimed it.

        The name check matters even though the graph runs one node at a time:
        a node abandoned by `_with_deadline` keeps a worker thread alive, and
        nothing that finishes late may darken the seat that is working now --
        or close that seat's turn in the record.
        """
        with self._lock:
            if self._node == node:
                self._end_turn()

    def clear(self) -> None:
        """No seat is working. The run's `finally` calls this."""
        with self._lock:
            self._end_turn()

    def _end_turn(self) -> None:
        """Put the light out and close the turn it belonged to. Lock held."""
        latest = self._turns[-1] if self._turns else None
        if latest is not None and latest.ended is None and latest.node == self._node:
            latest.ended = time.monotonic()
        self._node = ""
        self._since = 0.0

    def current(self) -> str:
        with self._lock:
            return self._node

    def busy_for(self) -> float:
        """Seconds the current seat has been in its turn; 0.0 when idle."""
        with self._lock:
            return time.monotonic() - self._since if self._node else 0.0

    def turns(self) -> list[dict[str, object]]:
        """The recent turns, oldest first: `turn`, `node`, `seconds`, `ended_ago`.

        `ended_ago` is None while the turn is still running. Both are spans
        rather than timestamps because this clock is monotonic, and the
        console's is a different clock altogether.
        """
        with self._lock:
            now = time.monotonic()
            return [
                {
                    "turn": turn.number,
                    "node": turn.node,
                    "seconds": round((now if turn.ended is None else turn.ended) - turn.started, 2),
                    "ended_ago": None if turn.ended is None else round(now - turn.ended, 2),
                }
                for turn in self._turns
            ]


# One run at a time, so one activity light, for the same reason as RUN_CONTROL.
ACTIVITY = NodeActivity()


class EmbedderActivity:
    """Whether the embedding model is working, for the embedder card's light.

    Measured where the work is done rather than inferred from who asked for it:
    `graphrag_server` marks every encode and every load of the model, which is
    everything the embedder does -- a run's corpus phase, a fetched page stored,
    the Planner's project map, the Researcher's search, a search or an upload
    from the console. A search against no corpus embeds nothing, and must light
    nothing.

    A count rather than a flag, because the work overlaps: the console can search
    while a run is indexing, and an encode loads the model on its first call. And
    like `NodeActivity` it remembers work already finished -- how many pieces
    have started, and how long ago the last one ended -- because a query embeds
    in tens of milliseconds, far inside the gap between two polls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._working = 0
        self._started = 0
        self._since = 0.0
        self._last_end: float | None = None

    @contextmanager
    def working(self) -> Iterator[None]:
        """The embedder is busy for as long as this block runs, however it exits."""
        with self._lock:
            if not self._working:
                self._since = time.monotonic()
            self._working += 1
            self._started += 1
        try:
            yield
        finally:
            with self._lock:
                self._working -= 1
                self._last_end = time.monotonic()

    def snapshot(self) -> dict[str, object]:
        """`busy`, `busy_for`, `started` (a count that only grows), and `ended_ago`.

        `ended_ago` is None until something has finished. Spans rather than
        timestamps, for the reason `NodeActivity.turns` gives.
        """
        with self._lock:
            now = time.monotonic()
            return {
                "busy": self._working > 0,
                "busy_for": round(now - self._since, 1) if self._working else 0.0,
                "started": self._started,
                "ended_ago": None if self._last_end is None else round(now - self._last_end, 2),
            }


# The console shows one embedder, so one meter, a process-global beside
# ACTIVITY for the same reason.
EMBEDDER_ACTIVITY = EmbedderActivity()


# How often a waiter re-checks the cards. Only a polling interval: a wait is
# never cut short, because one that gave up and ran anyway was exactly the
# overlap this arbiter exists to prevent -- two models on the cards at once.
# Every holder is bounded on its own (a seat call by LLM_TIMEOUT_SECONDS at the
# socket, an embed by its request timeout), so a wait always ends.
GPU_WAIT_POLL_SECONDS = 1.0


class GpuArbiter:
    """One piece of GPU work at a time: a seat's model, or the embedder.

    This machine has two 3 GB cards and every model that matters is larger than
    one of them. qwen3-embedding is 4.7 GB and a 9B seat is 5.3 GB, so the two
    together want roughly 10 GB of a 6 GB pool: they cannot both be resident,
    and the daemon's answer to being asked anyway is not a graceful split but a
    failure. Measured from the journal on 2026-09-20 at 07:02:10 -- a seat load
    timed out with `context canceled` after the daemon spent twenty seconds
    walking `ngl_per_device_high[1].n_layer` down from 48, trying to pack every
    layer onto card 1 because the embedder held card 0. That walk is what an
    operator sees as "one of the cards isn't being touched".

    So the work is serialized rather than left to compete. This is deliberately
    *not* `EMBEDDER_ACTIVITY`, which sits beside it: that is a meter, read by
    the console to draw a light, and it counts overlapping work precisely
    because overlapping work is what it was built to display. This one is a
    lock, and its whole purpose is that the overlap stops happening.

    Reentrant on purpose. Nothing here nests today -- `_gather_research` embeds
    and *then* invokes a seat, never inside it -- but a plain lock turns a
    future nesting into a run that hangs for ever with no node on the stack,
    which is the one failure this code must not add. Reentrancy makes that
    mistake a no-op instead.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = threading.Lock()
        self._holder = ""
        self._depth = 0
        self._since = 0.0
        self._waiting = 0

    @contextmanager
    def exclusive(self, owner: str) -> Iterator[float]:
        """Hold the cards for `owner` while the block runs, however it exits.

        Waits as long as the holder takes, never less: one model on the cards
        at a time is a guarantee, not a preference. Yields the seconds spent
        waiting, which the caller may report: a wait is the whole visible cost
        of this arbiter, and one nobody can see reads as the model simply
        being slow.
        """
        start = time.monotonic()
        if not self._lock.acquire(blocking=False):
            with self._state:
                self._waiting += 1
            try:
                while not self._lock.acquire(timeout=GPU_WAIT_POLL_SECONDS):
                    pass
            finally:
                with self._state:
                    self._waiting -= 1
        waited = time.monotonic() - start
        with self._state:
            self._depth += 1
            if self._depth == 1:
                self._holder = owner
                self._since = time.monotonic()
        try:
            yield waited
        finally:
            with self._state:
                self._depth -= 1
                if self._depth == 0:
                    self._holder = ""
                    self._since = 0.0
            self._lock.release()

    def snapshot(self) -> dict[str, object]:
        """Who holds the cards, for how long, and how many are queued behind it."""
        with self._state:
            now = time.monotonic()
            return {
                "holder": self._holder,
                "held_for": round(now - self._since, 2) if self._holder else 0.0,
                "waiting": self._waiting,
            }


# One pool of cards, so one arbiter, a process-global beside ACTIVITY and
# EMBEDDER_ACTIVITY for the same reason.
GPU_ARBITER = GpuArbiter()
