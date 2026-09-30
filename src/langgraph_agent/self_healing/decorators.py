"""
Decorator-based wrappers for self-healing functionality.

Provides decorators that wrap business logic without modifying it directly:
- retry_with_backoff: Retry logic with exponential backoff
- circuit_breaker: Circuit breaker pattern to prevent cascade failures
- self_healing_wrapper: Combined healing wrapper

All decorators implement circuit-breaker patterns to prevent infinite retry loops.
"""

import functools
from collections.abc import Callable
from typing import Any

import pybreaker
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .logger import SelfHealingLogger, get_healing_logger

# Circuit breaker instances stored by function name
_circuit_breakers: dict[str, pybreaker.CircuitBreaker] = {}


class _StateChangeLogger(pybreaker.CircuitBreakerListener):
    """Logs circuit breaker state transitions via the self-healing logger."""

    def __init__(self, cb_name: str, failure_threshold: int, logger: SelfHealingLogger) -> None:
        self._cb_name = cb_name
        self._failure_threshold = failure_threshold
        self._logger = logger

    def state_change(self, cb: pybreaker.CircuitBreaker, old_state: object, new_state: object) -> None:
        new_name = getattr(new_state, "name", str(new_state))
        if new_name == pybreaker.STATE_OPEN:
            self._logger.log_circuit_opened(self._cb_name, self._failure_threshold)
        elif new_name == pybreaker.STATE_CLOSED:
            self._logger.log_circuit_closed(self._cb_name)
        elif new_name == pybreaker.STATE_HALF_OPEN:
            self._logger.log_circuit_half_open(self._cb_name)


def _get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    expected_exception: type[Exception] = Exception,
    logger_name: str = "self_healing"
) -> pybreaker.CircuitBreaker:
    """Get or create a circuit breaker for the given function.

    `expected_exception` is accepted for API compatibility but pybreaker
    (>=1.0) has no equivalent filter: every exception trips the breaker
    unless explicitly excluded, which is the behavior every caller here
    already relies on (each `expected_exception` passed is itself a plain
    `Exception` subclass).
    """
    if name not in _circuit_breakers:
        cb = pybreaker.CircuitBreaker(
            name=name,
            fail_max=failure_threshold,
            reset_timeout=recovery_timeout,
        )
        cb.add_listener(_StateChangeLogger(name, failure_threshold, get_healing_logger(logger_name)))
        _circuit_breakers[name] = cb
    return _circuit_breakers[name]


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger_name: str = "self_healing"
) -> Callable[..., Any]:
    """
    Decorator that adds retry logic with exponential backoff.

    Implements circuit-breaker pattern to prevent infinite retry loops:
    - Maximum retry count prevents endless attempts
    - Exponential backoff reduces system load during failures
    - All retry attempts are logged with severity levels

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        min_wait: Minimum wait time between retries in seconds (default: 1.0)
        max_wait: Maximum wait time between retries in seconds (default: 60.0)
        exceptions: Tuple of exception types to retry on
        logger_name: Name of the logger to use

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data():
            # Business logic here
            pass
    """
    logger = get_healing_logger(logger_name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = func.__name__

        def _before_sleep(retry_state: RetryCallState) -> None:
            """Log retry attempt before sleeping."""
            error_msg = str(retry_state.outcome.exception()) if retry_state.outcome else "Unknown error"
            logger.log_retry_attempt(
                func_name=func_name,
                attempt=retry_state.attempt_number,
                max_attempts=max_attempts,
                error=error_msg
            )

        retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=_before_sleep,
            reraise=True,
        )

        # Success is logged here, once the call has returned, and not from
        # tenacity's `after` hook. That hook runs only after an attempt that
        # *failed* and is about to be retried, so hanging "Retry succeeded" on
        # it logged a success for the second attempt exactly when the second
        # attempt had failed -- and never logged the attempt that did succeed.
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0

            def attempt() -> Any:
                nonlocal attempts
                attempts += 1
                return func(*args, **kwargs)

            try:
                result = retrying(attempt)
            except exceptions:
                logger.log_retry_exhausted(func_name=func_name, max_attempts=max_attempts)
                raise
            if attempts > 1:
                logger.log_retry_success(func_name=func_name, attempt=attempts)
            return result

        return wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    expected_exception: type[Exception] = Exception,
    logger_name: str = "self_healing",
    name: str | None = None
) -> Callable[..., Any]:
    """
    Decorator that implements the circuit breaker pattern.

    Prevents cascade failures by opening the circuit after repeated failures:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit tripped, requests fail immediately
    - HALF-OPEN: Testing if service recovered

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Seconds to wait before testing recovery (default: 30)
        expected_exception: Exception type that triggers circuit breaker
        logger_name: Name of the logger to use
        name: Optional name for the circuit breaker (defaults to function name)

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def call_external_service():
            # Business logic here
            pass
    """
    logger = get_healing_logger(logger_name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cb_name = name or func.__name__
        cb = _get_circuit_breaker(
            name=cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            logger_name=logger_name
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return cb.call(func, *args, **kwargs)
            except pybreaker.CircuitBreakerError as e:
                logger.error(
                    f"Circuit breaker preventing call to '{cb_name}': {str(e)}",
                    action="circuit_prevented",
                    function=cb_name
                )
                raise

        return wrapper

    return decorator


def self_healing_wrapper(
    max_attempts: int = 3,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    circuit_exception: type[Exception] = Exception,
    logger_name: str = "self_healing",
    name: str | None = None
) -> Callable[..., Any]:
    """
    Combined self-healing wrapper with both retry and circuit breaker.

    Applies both retry logic and circuit breaker pattern for comprehensive
    runtime resilience. The circuit breaker wraps the retry logic, so:
    1. Retries happen first (up to max_attempts)
    2. If all retries fail, circuit breaker counts it as a failure
    3. After failure_threshold failures, circuit opens

    Args:
        max_attempts: Maximum retry attempts (default: 3)
        failure_threshold: Failures before circuit opens (default: 5)
        recovery_timeout: Seconds before testing recovery (default: 30)
        retry_exceptions: Exception types to retry on
        circuit_exception: Exception type for circuit breaker
        logger_name: Name of the logger to use

    Returns:
        Decorated function with comprehensive healing

    Example:
        @self_healing_wrapper(max_attempts=3, failure_threshold=5)
        def critical_operation():
            # Business logic here
            pass
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Apply circuit breaker first (outer layer)
        cb_name = name or f"{func.__module__}.{func.__name__}"
        cb = _get_circuit_breaker(
            name=cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=circuit_exception
        )

        # Apply retry logic (inner layer)
        retry_decorator = retry_with_backoff(
            max_attempts=max_attempts,
            exceptions=retry_exceptions,
            logger_name=logger_name
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return cb.call(retry_decorator(func), *args, **kwargs)
            except pybreaker.CircuitBreakerError as e:
                logger = get_healing_logger(logger_name)
                logger.error(
                    f"Circuit breaker preventing call to '{cb_name}': {str(e)}",
                    action="circuit_prevented",
                    function=cb_name
                )
                raise

        return wrapper

    return decorator
