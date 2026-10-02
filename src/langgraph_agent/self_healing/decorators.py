"""Retries with backoff and circuit breakers, as decorators or plain calls.

- `call_with_retry` / `retry_with_backoff`: call again after a failure the
  policy calls transient, waiting exponentially longer each time, up to
  `max_attempts`. `give_up` cuts a wait short and re-raises the last failure.
- `Circuit` / `circuit_breaker`: stop calling a service that keeps failing.
  After `recovery_timeout` seconds one trial call is let through; if it
  succeeds the circuit closes again on its own.
- `self_healing_wrapper`: both, with the circuit outside the retries.

Circuits are named and shared: every caller naming one counts toward it and is
protected by it, which is what lets several functions that talk to one service
stand down together. A refused call raises `CircuitOpenError`, and no retry
policy here ever retries one -- an open circuit is answered by its cooldown.
"""

import functools
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ParamSpec, TypeVar

import pybreaker
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .logger import SelfHealingLogger, get_healing_logger

P = ParamSpec("P")
R = TypeVar("R")

# Every circuit this process has defined, by name.
_circuit_breakers: dict[str, pybreaker.CircuitBreaker] = {}
_registry_lock = threading.Lock()

# How finely a backoff wait is sliced, so `give_up` is asked while waiting.
_WAIT_SLICE_SECONDS = 0.25


class CircuitOpenError(pybreaker.CircuitBreakerError):
    """A call refused without being attempted, because its circuit is open."""

    def __init__(self, circuit: str, retry_in: float) -> None:
        self.circuit = circuit
        self.retry_in = retry_in
        super().__init__(
            f"{circuit} is unavailable: its circuit opened after repeated failures, "
            f"and the next call is let through to test it in {max(0, round(retry_in))}s"
        )


class _StateChangeLogger(pybreaker.CircuitBreakerListener):
    """Logs a circuit's transitions -- only the ones that change its state."""

    def __init__(self, cb_name: str, logger: SelfHealingLogger) -> None:
        self._cb_name = cb_name
        self._logger = logger

    def state_change(self, cb: pybreaker.CircuitBreaker, old_state: object, new_state: object) -> None:
        old_name = getattr(old_state, "name", None)
        new_name = getattr(new_state, "name", str(new_state))
        if new_name == old_name:
            return
        if new_name == pybreaker.STATE_OPEN:
            self._logger.log_circuit_opened(self._cb_name, cb.fail_counter)
        elif new_name == pybreaker.STATE_CLOSED:
            self._logger.log_circuit_closed(self._cb_name)
        elif new_name == pybreaker.STATE_HALF_OPEN:
            self._logger.log_circuit_half_open(self._cb_name)


def _get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30,
    expected_exception: type[BaseException] = Exception,
    logger_name: str = "self_healing",
    trips_on: Callable[[BaseException], bool] | None = None,
) -> pybreaker.CircuitBreaker:
    """The circuit called `name`, created on first use.

    Only a failure that is an `expected_exception` -- and, when `trips_on` is
    given, one it agrees is a failure of the service -- counts toward opening
    it; any other outcome counts as the service answering. Later callers share
    the circuit as it was first defined.
    """
    with _registry_lock:
        cb = _circuit_breakers.get(name)
        if cb is None:
            def answered(exc: BaseException) -> bool:
                return not isinstance(exc, expected_exception) or (
                    trips_on is not None and not trips_on(exc)
                )

            cb = pybreaker.CircuitBreaker(
                name=name,
                fail_max=failure_threshold,
                reset_timeout=recovery_timeout,
                exclude=[answered],
                # The call that trips the circuit raises its own error, which
                # says why; only the calls refused after it raise CircuitOpenError.
                throw_new_error_on_trip=False,
            )
            cb.add_listener(
                _StateChangeLogger(name, get_healing_logger(logger_name))
            )
            _circuit_breakers[name] = cb
        return cb


def _retry_in(cb: pybreaker.CircuitBreaker) -> float:
    """Seconds until an open circuit lets a trial call through; 0 if it would now."""
    opened_at = cb._state_storage.opened_at
    if opened_at is None:
        return 0.0
    elapsed = (datetime.now(UTC) - opened_at).total_seconds()
    return max(0.0, float(cb.reset_timeout) - elapsed)


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
        """Call `func` through the circuit; raise `CircuitOpenError` if it is open."""
        try:
            result: R = self._breaker.call(func, *args, **kwargs)
        except CircuitOpenError:
            raise
        except pybreaker.CircuitBreakerError as exc:
            refusal = CircuitOpenError(self.name, _retry_in(self._breaker))
            self._logger.error(
                f"Circuit breaker preventing call to '{self.name}': {refusal}",
                action="circuit_prevented",
                function=self.name,
            )
            raise refusal from exc
        return result

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Use the circuit as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self.call(func, *args, **kwargs)

        return wrapper

    def trip(self, reason: str) -> None:
        """Open the circuit now, for a failure that says more than one call."""
        if self._breaker.current_state != pybreaker.STATE_OPEN:
            self._logger.warning(
                f"Circuit for '{self.name}' tripped: {reason}",
                action="circuit_tripped",
                function=self.name,
            )
            self._breaker.open()

    @property
    def is_open(self) -> bool:
        return bool(self._breaker.current_state == pybreaker.STATE_OPEN)

    @property
    def retry_in(self) -> float:
        """Seconds until an open circuit lets its trial call through; 0 if closed or due."""
        return _retry_in(self._breaker) if self.is_open else 0.0


def circuit_states() -> list[dict[str, Any]]:
    """Every circuit this process has defined: its state, failures and cooldown."""
    with _registry_lock:
        breakers = sorted(_circuit_breakers.values(), key=lambda cb: str(cb.name))
    states = []
    for cb in breakers:
        state = cb.current_state
        states.append({
            "name": cb.name,
            "state": state,
            "failures": cb.fail_counter,
            "threshold": cb.fail_max,
            "cooldown_s": cb.reset_timeout,
            "retry_in_s": round(_retry_in(cb), 1) if state == pybreaker.STATE_OPEN else None,
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
    return [str(cb.name) for cb in breakers]


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
            and not isinstance(exc, pybreaker.CircuitBreakerError)
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


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger_name: str = "self_healing",
    *,
    retry_if: Callable[[BaseException], bool] | None = None,
    give_up: Callable[[], bool] | None = None,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of `call_with_retry`.

    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data(): ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return call_with_retry(
                functools.partial(func, *args, **kwargs),
                max_attempts=max_attempts,
                min_wait=min_wait,
                max_wait=max_wait,
                exceptions=exceptions,
                retry_if=retry_if,
                give_up=give_up,
                name=name or func.__name__,
                logger_name=logger_name,
            )

        return wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30,
    expected_exception: type[BaseException] = Exception,
    logger_name: str = "self_healing",
    name: str | None = None,
    *,
    trips_on: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of `Circuit`; the circuit is named after the function by default.

    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def call_external_service(): ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        circuit = Circuit(
            name or func.__name__,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            trips_on=trips_on,
            logger_name=logger_name,
        )
        return circuit(func)

    return decorator


def self_healing_wrapper(
    max_attempts: int = 3,
    failure_threshold: int = 5,
    recovery_timeout: float = 30,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    circuit_exception: type[BaseException] = Exception,
    logger_name: str = "self_healing",
    name: str | None = None,
    *,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retry_if: Callable[[BaseException], bool] | None = None,
    trips_on: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retries inside a circuit: a whole run of failed retries counts as one failure.

    Example:
        @self_healing_wrapper(max_attempts=3, failure_threshold=5)
        def critical_operation(): ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        circuit = Circuit(
            name or f"{func.__module__}.{func.__name__}",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=circuit_exception,
            trips_on=trips_on,
            logger_name=logger_name,
        )
        retried = retry_with_backoff(
            max_attempts=max_attempts,
            min_wait=min_wait,
            max_wait=max_wait,
            exceptions=retry_exceptions,
            logger_name=logger_name,
            retry_if=retry_if,
            name=func.__name__,
        )(func)
        return circuit(retried)

    return decorator
