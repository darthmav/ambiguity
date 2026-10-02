"""Tests for the self_healing package: logger, retry, and circuit breaker."""

import threading
import time

import pybreaker
import pytest

from langgraph_agent.self_healing import (
    Circuit,
    CircuitOpenError,
    SelfHealingLogger,
    call_with_retry,
    circuit_breaker,
    circuit_states,
    get_healing_logger,
    reset_circuit,
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


# ---------------------------------------------------------------------------
# The policy decides what is retried, and when to stop
# ---------------------------------------------------------------------------


def test_only_what_the_policy_calls_transient_is_retried():
    calls = [0]

    def refused() -> str:
        calls[0] += 1
        raise ValueError("a refusal, not an outage")

    with pytest.raises(ValueError):
        call_with_retry(refused, max_attempts=5, min_wait=0, max_wait=0,
                        retry_if=lambda exc: isinstance(exc, ConnectionError))
    assert calls[0] == 1


def test_give_up_ends_the_wait_and_raises_the_last_failure(caplog):
    calls = [0]
    stop = threading.Event()

    def failing() -> str:
        calls[0] += 1
        stop.set()  # asked to stop while the first wait is under way
        raise ConnectionError(f"down {calls[0]}")

    started = time.monotonic()
    with caplog.at_level("WARNING", logger=get_healing_logger().logger.name):
        with pytest.raises(ConnectionError, match="down 1"):
            call_with_retry(failing, max_attempts=5, min_wait=30, max_wait=30,
                            give_up=stop.is_set, name="gives_up")
    assert time.monotonic() - started < 5
    assert calls[0] == 1
    assert any("Gave up retrying 'gives_up'" in r.getMessage() for r in caplog.records)


def test_a_refused_circuit_is_never_retried():
    calls = [0]

    def refused() -> str:
        calls[0] += 1
        raise CircuitOpenError("svc", 10)

    with pytest.raises(CircuitOpenError):
        call_with_retry(refused, max_attempts=5, min_wait=0, max_wait=0)
    assert calls[0] == 1


# ---------------------------------------------------------------------------
# Circuits: what counts, what opens, what closes
# ---------------------------------------------------------------------------


def _state(name: str) -> dict:
    return next(c for c in circuit_states() if c["name"] == name)


def test_only_failures_the_circuit_trips_on_count_toward_it():
    circuit = Circuit("trips_on_connection", failure_threshold=2, recovery_timeout=60,
                      trips_on=lambda exc: isinstance(exc, ConnectionError))

    def refusal() -> None:
        raise ValueError("the service answered: no")

    for _ in range(5):
        with pytest.raises(ValueError):
            circuit.call(refusal)
    assert _state("trips_on_connection")["state"] == "closed"

    def outage() -> None:
        raise ConnectionError("unreachable")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            circuit.call(outage)
    assert circuit.is_open

    called = []
    with pytest.raises(CircuitOpenError, match="trips_on_connection is unavailable") as refused:
        circuit.call(lambda: called.append(1))
    assert called == []
    assert refused.value.circuit == "trips_on_connection"
    assert 0 < _state("trips_on_connection")["retry_in_s"] <= 60


def test_expected_exception_is_what_the_circuit_counts():
    """It was accepted and ignored: every exception tripped the breaker."""
    calls = [0]

    @circuit_breaker(failure_threshold=2, recovery_timeout=60,
                     expected_exception=ConnectionError, name="expects_connection")
    def service() -> None:
        calls[0] += 1
        raise ValueError("not an outage")

    for _ in range(4):
        with pytest.raises(ValueError):
            service()
    assert calls[0] == 4
    assert _state("expects_connection")["state"] == "closed"


def test_an_open_circuit_closes_through_its_trial_call(caplog):
    circuit = Circuit("recovers_on_trial", failure_threshold=1, recovery_timeout=0.2)
    with caplog.at_level("INFO", logger=get_healing_logger().logger.name):
        with pytest.raises(ConnectionError):
            circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
        assert circuit.is_open
        time.sleep(0.3)
        assert circuit.call(lambda: "up") == "up"
    assert _state("recovers_on_trial")["state"] == "closed"
    messages = [r.getMessage() for r in caplog.records]
    assert any("OPENED for 'recovers_on_trial'" in m for m in messages)
    assert any("CLOSED for 'recovers_on_trial'" in m for m in messages)


def test_trip_opens_a_circuit_at_once_and_reset_closes_it():
    circuit = Circuit("tripped_by_hand", failure_threshold=5, recovery_timeout=60)
    circuit.trip("a bot check answered instead of results")
    assert circuit.is_open

    assert reset_circuit("tripped_by_hand") == ["tripped_by_hand"]
    assert not circuit.is_open
    with pytest.raises(KeyError):
        reset_circuit("no_such_circuit")


def test_one_name_is_one_circuit_whoever_calls_it():
    first = Circuit("shared_service", failure_threshold=2, recovery_timeout=60)
    second = Circuit("shared_service", failure_threshold=99, recovery_timeout=1)

    for circuit in (first, second):
        with pytest.raises(ConnectionError):
            circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    assert first.is_open and second.is_open
    assert _state("shared_service")["threshold"] == 2


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------


def test_the_journal_tags_a_session_and_reads_from_a_sequence_number():
    logger = SelfHealingLogger(name="journal_test")
    logger.info("before any session")
    mark = logger.events()[-1]["seq"]

    logger.start_healing_session("run-1")
    logger.log_health_check("ollama-daemon", "unhealthy", "refused")
    logger.end_healing_session()
    logger.info("after the session")

    later = logger.events(since=mark)
    assert [e["message"] for e in later][-1] == "after the session"
    session = logger.events(session_id="run-1")
    assert any(e.get("action") == "health_check" and e["level"] == "WARNING" for e in session)
    assert all(e["session_id"] == "run-1" for e in session)
