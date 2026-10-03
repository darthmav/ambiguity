"""Retries with backoff and circuit breakers, as plain calls.

- `call_with_retry`: call again after a failure the policy calls transient,
  waiting exponentially longer each time, up to `max_attempts`. `give_up`
  cuts a wait short and re-raises the last failure.
- `Circuit`: stop calling a service that keeps failing. After
  `recovery_timeout` seconds one trial call is let through; if it succeeds
  the circuit closes again on its own.

Circuits are named and shared: every caller naming one counts toward it and is
protected by it, which is what lets several functions that talk to one service
stand down together. A refused call raises `CircuitOpenError`, and no retry
policy here ever retries one -- an open circuit is answered by its cooldown.

A circuit only keeps the books; it never serializes the calls it guards. Its
lock is held to read and update its state, never across the call itself, so a
seat generating for two minutes does not hold up a status read that goes
through the same circuit.
"""

import contextlib
import functools
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, ParamSpec, TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .logger import get_healing_logger

P = ParamSpec("P")
R = TypeVar("R")

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half-open"

# How finely a backoff wait is sliced, so `give_up` is asked while waiting.
_WAIT_SLICE_SECONDS = 0.25


def exception_chain(exc: BaseException) -> list[BaseException]:
    """`exc` and every exception it was raised from or during, outermost first.

    What a `trips_on` predicate reads: a library's own error is usually the
    context of the one a caller sees.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


class CircuitOpenError(Exception):
    """A call refused without being attempted, because its circuit is open."""

    def __init__(self, circuit: str, retry_in: float, *, testing: bool = False) -> None:
        self.circuit = circuit
        self.retry_in = retry_in
        if testing:
            detail = "a trial call is testing it now"
        else:
            detail = f"the next call is let through to test it in {max(0, round(retry_in))}s"
        super().__init__(
            f"{circuit} is unavailable: its circuit opened after repeated failures, and {detail}"
        )


class _Breaker:
    """One circuit's state: what counts as a failure, how many, and since when.

    Every method takes `_lock` for its own bookkeeping only. Closed, a failure
    the circuit counts adds one and a call that succeeds -- or fails in a way
    the circuit does not count, which is the service answering -- clears the
    count; at `fail_max` it opens. Open, every call is refused until
    `reset_timeout` has passed; then one caller becomes the trial and the
    circuit is half-open, refusing the rest until the trial ends. The trial
    closes it by succeeding and reopens it by failing.
    """

    def __init__(
        self,
        name: str,
        fail_max: int,
        reset_timeout: float,
        counts: Callable[[BaseException], bool],
        logger_name: str,
    ) -> None:
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._counts = counts
        self._logger = get_healing_logger(logger_name)
        self._lock = threading.Lock()
        self.state = STATE_CLOSED
        self.fail_counter = 0
        self._opened_at: float | None = None
        # Which trial is under way, so one that outlives a reset cannot
        # settle the state a later trial is testing.
        self._trial = 0
        self._trial_running = False

    def retry_in(self) -> float:
        """Seconds until an open circuit lets a trial call through; 0 if it would now."""
        if self.state != STATE_OPEN or self._opened_at is None:
            return 0.0
        return max(0.0, self.reset_timeout - (time.monotonic() - self._opened_at))

    def admit(self) -> int:
        """Let a call through: 0 for an ordinary call, a trial number for the trial.

        Raises `CircuitOpenError` for a call the circuit refuses.
        """
        with self._lock:
            if self.state == STATE_CLOSED:
                return 0
            if self.state == STATE_OPEN and self.retry_in() > 0:
                raise CircuitOpenError(self.name, self.retry_in())
            if self._trial_running:
                raise CircuitOpenError(self.name, 0.0, testing=True)
            became_half_open = self.state != STATE_HALF_OPEN
            self.state = STATE_HALF_OPEN
            self._trial += 1
            self._trial_running = True
            trial = self._trial
        if became_half_open:
            self._logger.log_circuit_half_open(self.name)
        return trial

    def settle(self, trial: int, exc: BaseException | None) -> None:
        """Record how a call admitted by `admit` ended: `exc` None for success."""
        failed = exc is not None and self._counts(exc)
        interrupted = exc is not None and not isinstance(exc, Exception)
        opened = closed = False
        with self._lock:
            ours = trial != 0 and trial == self._trial and self._trial_running
            if ours:
                self._trial_running = False
            if interrupted:
                # Neither an answer nor an outage: the trial is simply over.
                return
            if ours and self.state == STATE_HALF_OPEN:
                if failed:
                    self.state, self._opened_at, opened = STATE_OPEN, time.monotonic(), True
                else:
                    self.state, self.fail_counter, closed = STATE_CLOSED, 0, True
            elif self.state == STATE_CLOSED:
                if failed:
                    self.fail_counter += 1
                    if self.fail_counter >= self.fail_max:
                        self.state, self._opened_at, opened = STATE_OPEN, time.monotonic(), True
                else:
                    self.fail_counter = 0
            failures = self.fail_counter
        if opened:
            self._logger.log_circuit_opened(self.name, failures)
        if closed:
            self._logger.log_circuit_closed(self.name)

    def open(self) -> bool:
        """Open the circuit now; False if it already was."""
        with self._lock:
            if self.state == STATE_OPEN:
                return False
            self.state, self._opened_at = STATE_OPEN, time.monotonic()
            self._trial_running = False
        self._logger.log_circuit_opened(self.name, 0)
        return True

    def close(self) -> None:
        """Close the circuit now, clearing its count."""
        with self._lock:
            was = self.state
            self.state, self.fail_counter, self._opened_at = STATE_CLOSED, 0, None
            self._trial_running = False
        if was != STATE_CLOSED:
            self._logger.log_circuit_closed(self.name)


# Every circuit this process has defined, by name.
_circuit_breakers: dict[str, _Breaker] = {}
_registry_lock = threading.Lock()


def _get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30,
    expected_exception: type[BaseException] = Exception,
    logger_name: str = "self_healing",
    trips_on: Callable[[BaseException], bool] | None = None,
) -> _Breaker:
    """The circuit called `name`, created on first use.

    Only a failure that is an `expected_exception` -- and, when `trips_on` is
    given, one it agrees is a failure of the service -- counts toward opening
    it; any other outcome counts as the service answering. Later callers share
    the circuit as it was first defined.
    """
    with _registry_lock:
        cb = _circuit_breakers.get(name)
        if cb is None:
            def counts(exc: BaseException) -> bool:
                return isinstance(exc, expected_exception) and (
                    trips_on is None or trips_on(exc)
                )

            cb = _Breaker(name, failure_threshold, recovery_timeout, counts, logger_name)
            _circuit_breakers[name] = cb
        return cb


class Circuit:
    """A named circuit breaker, shared by every caller that names it."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30,
        expected_exception: type[BaseException] = Exception,
        trips_on: Callable[[BaseException], bool] | None = None,
        logger_name: str = "self_healing",
    ) -> None:
        self.name = name
        self._breaker = _get_circuit_breaker(
            name, failure_threshold, recovery_timeout, expected_exception, logger_name, trips_on
        )
        self._logger = get_healing_logger(logger_name)

    def call(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Call `func` through the circuit; raise `CircuitOpenError` if it is open.

        The circuit's lock is never held while `func` runs.
        """
        with self.guarding():
            return func(*args, **kwargs)

    @contextlib.contextmanager
    def guarding(self) -> Iterator[None]:
        """The circuit around a block, as `call` puts it around a function.

        Refused before the block runs while the circuit is open; how the block
        ends -- returning, or the exception it raises -- counts as one call.
        """
        try:
            trial = self._breaker.admit()
        except CircuitOpenError as refusal:
            self._logger.error(
                f"Circuit breaker preventing call to '{self.name}': {refusal}",
                action="circuit_prevented",
                function=self.name,
            )
            raise
        try:
            yield
        except BaseException as exc:
            self._breaker.settle(trial, exc)
            raise
        self._breaker.settle(trial, None)

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Use the circuit as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self.call(func, *args, **kwargs)

        return wrapper

    def trip(self, reason: str) -> None:
        """Open the circuit now, for a failure that says more than one call."""
        if self._breaker.state != STATE_OPEN:
            self._logger.warning(
                f"Circuit for '{self.name}' tripped: {reason}",
                action="circuit_tripped",
                function=self.name,
            )
            self._breaker.open()

    @property
    def is_open(self) -> bool:
        return self._breaker.state == STATE_OPEN

    @property
    def retry_in(self) -> float:
        """Seconds until an open circuit lets its trial call through; 0 if closed or due."""
        return self._breaker.retry_in()


def circuit_states() -> list[dict[str, Any]]:
    """Every circuit this process has defined: its state, failures and cooldown."""
    with _registry_lock:
        breakers = sorted(_circuit_breakers.values(), key=lambda cb: cb.name)
    states = []
    for cb in breakers:
        state = cb.state
        states.append({
            "name": cb.name,
            "state": state,
            "failures": cb.fail_counter,
            "threshold": cb.fail_max,
            "cooldown_s": cb.reset_timeout,
            "retry_in_s": round(cb.retry_in(), 1) if state == STATE_OPEN else None,
        })
    return states


def reset_circuit(name: str | None = None) -> list[str]:
    """Close one circuit (every circuit when `name` is None); return the names reset.

    For an operator who knows the service is back, and for tests. Raises
    KeyError for a name no circuit has.
    """
    with _registry_lock:
        if name is None:
            breakers = list(_circuit_breakers.values())
        else:
            breakers = [_circuit_breakers[name]]
    for cb in breakers:
        cb.close()
    return [cb.name for cb in breakers]


def call_with_retry(
    func: Callable[[], R],
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    retry_if: Callable[[BaseException], bool] | None = None,
    give_up: Callable[[], bool] | None = None,
    name: str = "",
    logger_name: str = "self_healing",
) -> R:
    """Call `func` until it succeeds, retrying the failures this policy calls transient.

    A failure is retried when it is one of `exceptions`, `retry_if` (if given)
    agrees, it is not a refused circuit, and attempts remain. Waits start at
    `min_wait` seconds and double up to `max_wait`. `give_up` is asked before
    every wait and throughout it; once it is true the last failure is re-raised
    at once. Whatever ends the retrying, the last failure is raised unchanged.
    """
    logger = get_healing_logger(logger_name)
    label = name or getattr(func, "__name__", repr(func))
    attempts = 0
    last_error: BaseException | None = None
    gave_up = False

    def transient(exc: BaseException) -> bool:
        return (
            isinstance(exc, exceptions)
            and not isinstance(exc, CircuitOpenError)
            and (retry_if is None or retry_if(exc))
        )

    def before_sleep(state: RetryCallState) -> None:
        nonlocal last_error
        last_error = state.outcome.exception() if state.outcome else None
        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        logger.log_retry_attempt(label, state.attempt_number, max_attempts, detail)

    def sleep(seconds: float) -> None:
        nonlocal gave_up
        deadline = time.monotonic() + seconds
        while True:
            if give_up is not None and give_up() and last_error is not None:
                gave_up = True
                raise last_error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, _WAIT_SLICE_SECONDS))

    def attempt() -> R:
        nonlocal attempts
        attempts += 1
        return func()

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, min=min_wait, max=max_wait),
        retry=retry_if_exception(transient),
        before_sleep=before_sleep,
        sleep=sleep,
        reraise=True,
    )
    try:
        result = retrying(attempt)
    except BaseException as exc:
        if gave_up:
            logger.log_retry_abandoned(label, attempts)
        elif attempts > 1 and attempts >= max_attempts and transient(exc):
            logger.log_retry_exhausted(label, max_attempts)
        raise
    if attempts > 1:
        logger.log_retry_success(label, attempts)
    return result
