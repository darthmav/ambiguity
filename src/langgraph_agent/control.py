"""Process-wide run state: the emergency stop, the activity lights, the GPU arbiter.

`RUN_CONTROL` is a module-level flag rather than a field on `AgentState`: the
graph runs without a checkpointer, so nothing outside a node can write into a
state being streamed. The stop is cooperative -- the checkpoints in `nodes.py`
and `serve.py` decline to start the next piece of work, and work in flight (a
write, a commit, a model call) always finishes rather than leave a half-written
file behind.
"""

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# Why a run ended, when the caller did not say.
DEFAULT_STOP_REASON = "Stopped from the console."


class RunControl:
    """An armed/stopped flag for one run, shared across threads.

    The stop arrives on a different request thread from the run's own.
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

        A `run_id` that does not match the armed run is refused: a stale tab's
        Stop must not kill the run that replaced it. An empty `run_id` means
        whatever is armed.
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
        """Whether the armed run has been asked to stop. Lock-free: it is asked often."""
        return self._event.is_set()

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def run_id(self) -> str:
        with self._lock:
            return self._run_id


# One run at a time, so one control.
RUN_CONTROL = RunControl()


# How many of a run's most recent turns `NodeActivity.turns` hands back.
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

    Not `_run_progress["node"]` in `serve.py`: `graph.stream` yields a node when
    it *finishes*, which would light the previous seat for the whole of the next
    one's turn. Set by a wrapper with a `finally` (`graph._tracked`), so no exit
    path leaves a light on. Every turn is also recorded, numbered, so the
    console can light a turn that began and ended between two of its polls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node = ""
        self._since = 0.0
        self._count = 0
        self._turns: deque[_Turn] = deque(maxlen=TURN_RECORD)

    def begin_run(self) -> None:
        """Forget the previous run's turns, as a new run is claimed.

        The numbering carries on, so a console holding the last number it saw
        reads every turn of the new run as new.
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
        """Clear the light, unless another seat already claimed it.

        A node abandoned by `_with_deadline` can finish late; it must not darken
        the seat working now.
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

        `ended_ago` is None while the turn runs. Spans, not timestamps: this
        clock is monotonic and the console's is another clock.
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


# One run at a time, so one set of lights.
ACTIVITY = NodeActivity()


class EmbedderActivity:
    """Whether the embedding model is working, for the embedder card's light.

    Marked where the work is done -- `graphrag_server` wraps every encode and
    every model load -- and counted, because the work overlaps (a search during a
    rebuild). It remembers finished work too, since a query embeds in tens of
    milliseconds, far inside the gap between two polls.
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

        `ended_ago` is None until something has finished.
        """
        with self._lock:
            now = time.monotonic()
            return {
                "busy": self._working > 0,
                "busy_for": round(now - self._since, 1) if self._working else 0.0,
                "started": self._started,
                "ended_ago": None if self._last_end is None else round(now - self._last_end, 2),
            }


# One embedder, so one meter.
EMBEDDER_ACTIVITY = EmbedderActivity()


# How often a waiter re-checks the cards. A wait is never cut short -- giving
# up and running anyway is the overlap the arbiter prevents -- and every holder
# ends on its own: a silent daemon at the socket timeout, and a seat call its
# node has given up on at the next token it streams (`abandoned`), however long
# the model would have gone on generating.
GPU_WAIT_POLL_SECONDS = 1.0


# Set in a worker thread to the event its caller sets on giving up on it
# (`_with_deadline` in nodes.py).
_ABANDONED: ContextVar[threading.Event | None] = ContextVar("abandoned", default=None)


@contextmanager
def abandonable(event: threading.Event) -> Iterator[None]:
    """Run the block as work whose caller says it has stopped waiting by setting `event`."""
    token = _ABANDONED.set(event)
    try:
        yield
    finally:
        _ABANDONED.reset(token)


def abandoned() -> bool:
    """Whether the caller of the work this thread is doing has stopped waiting for it.

    Python cannot cancel a thread; work that can stop part-way -- a streamed
    seat call, between tokens -- asks this and stops, releasing what it holds.
    """
    event = _ABANDONED.get()
    return event is not None and event.is_set()


class GpuArbiter:
    """One piece of GPU work at a time: a seat's model, or the embedder.

    The embedder (4.7 GB) and a 9B seat (5.3 GB) cannot both fit 6 GB of cards,
    and the daemon's answer to being asked is a failed load, not a split. So
    the work is serialized. Unlike `EMBEDDER_ACTIVITY`, a meter that counts
    overlap, this is a lock that prevents it. Reentrant, so a nested use is a
    no-op rather than a run hung with no node on the stack.
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

        Waits as long as the holder takes: one model on the cards is a
        guarantee, not a preference. Yields the seconds spent waiting.
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


# One pool of cards, so one arbiter.
GPU_ARBITER = GpuArbiter()
