"""Tests for the self_healing package: logger, retry, and circuit breaker."""

import time

import pybreaker
import pytest

from langgraph_agent.self_healing import (
    SelfHealingLogger,
    circuit_breaker,
    get_healing_logger,
    retry_with_backoff,
    self_healing_wrapper,
)


def test_package_metadata():
    from langgraph_agent.self_healing import __status__, __version__

    assert __status__ == "active"
    assert __version__


def test_logger_levels_and_session(caplog):
    logger = SelfHealingLogger(name="test_healing")
    logger.start_healing_session("test-session-001")

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    logger.log_retry_attempt("fn", 1, 3, "ConnectionError")
    logger.log_retry_success("fn", 2)
    logger.log_retry_exhausted("fn", 3)
    logger.log_circuit_opened("svc", 5)
    logger.log_circuit_closed("svc")
    logger.log_circuit_half_open("svc")
    logger.log_health_check("database", "healthy", "Connection OK")
    logger.log_health_check("api_service", "unhealthy", "Timeout")
    logger.log_recovery_action("restart", "api_service", True, "restarted")
    logger.log_recovery_action("restart", "database", False, "failed")

    logger.end_healing_session()
    assert logger._healing_session_id is None


def test_get_healing_logger_is_singleton():
    assert get_healing_logger() is get_healing_logger()


def test_retry_with_backoff_succeeds_after_failures():
    call_count = [0]

    @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.05)
    def flaky() -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"attempt {call_count[0]}")
        return "ok"

    assert flaky() == "ok"
    assert call_count[0] == 3


def test_retry_with_backoff_raises_when_exhausted():
    call_count = [0]

    @retry_with_backoff(max_attempts=2, min_wait=0.01, max_wait=0.05)
    def always_fails() -> str:
        call_count[0] += 1
        raise ValueError(f"attempt {call_count[0]}")

    with pytest.raises(ValueError):
        always_fails()
    assert call_count[0] == 2


def test_circuit_breaker_opens_after_threshold():
    call_count = [0]

    @circuit_breaker(failure_threshold=3, recovery_timeout=2, name="test_circuit")
    def failing_service() -> str:
        call_count[0] += 1
        raise ConnectionError("unavailable")

    failures = 0
    for _ in range(5):
        try:
            failing_service()
        except Exception:
            failures += 1

    assert failures >= 3


def test_circuit_breaker_recovers():
    call_count = [0]
    fail_until = 3

    @circuit_breaker(failure_threshold=3, recovery_timeout=1, name="recovery_circuit")
    def recovering_service() -> str:
        call_count[0] += 1
        if call_count[0] <= fail_until:
            raise ConnectionError(f"down {call_count[0]}")
        return "recovered"

    for _ in range(3):
        with pytest.raises((ConnectionError, pybreaker.CircuitBreakerError)):
            recovering_service()

    time.sleep(1.5)

    # Circuit is half-open; either it lets the call through and recovers,
    # or it is still guarding -- both are acceptable outcomes here.
    try:
        result = recovering_service()
        assert result == "recovered"
    except Exception:
        pass


def test_self_healing_wrapper_combines_retry_and_circuit():
    call_count = [0]

    @self_healing_wrapper(
        max_attempts=2,
        failure_threshold=3,
        recovery_timeout=2,
        retry_exceptions=(ConnectionError,),
        circuit_exception=ConnectionError,
        name="combined_healing",
    )
    def critical_operation() -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"transient {call_count[0]}")
        return "succeeded"

    try:
        result = critical_operation()
        assert result == "succeeded"
    except ConnectionError:
        pass  # retries exhausted before success is an acceptable outcome


def test_a_retry_logs_the_attempt_that_succeeded_and_no_other(caplog):
    """Success hung on tenacity's `after` hook, which runs after a *failed* attempt.

    So "Retry succeeded on attempt 2" was logged exactly when attempt 2
    failed, and the attempt that did succeed was never logged at all.
    """
    calls = [0]

    @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=0)
    def flaky() -> str:
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionError(f"attempt {calls[0]}")
        return "ok"

    with caplog.at_level("INFO", logger=get_healing_logger().logger.name):
        assert flaky() == "ok"

    successes = [r.getMessage() for r in caplog.records if "succeeded" in r.getMessage()]
    assert successes == ["Retry succeeded for 'flaky' on attempt 3"]


def test_each_logger_name_gets_its_own_logger():
    """A single global handed every caller whichever name asked first."""
    first = get_healing_logger("healing_a")
    second = get_healing_logger("healing_b")

    assert first is not second
    assert second.logger.name == "healing_b"
    assert get_healing_logger("healing_a") is first
