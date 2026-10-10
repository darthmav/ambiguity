"""Tests for the self_healing package: logger, retry, and circuit breaker."""

import threading
import time

import pytest

from langgraph_agent.self_healing import (
    Circuit,
    CircuitOpenError,
    SelfHealingLogger,
    call_with_retry,
    circuit_states,
    get_healing_logger,
    reset_circuit,
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


def test_call_with_retry_succeeds_after_failures():
    call_count = [0]

    def flaky() -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"attempt {call_count[0]}")
        return "ok"

    assert call_with_retry(flaky, max_attempts=3, min_wait=0.01, max_wait=0.05) == "ok"
    assert call_count[0] == 3


def test_call_with_retry_raises_when_exhausted():
    call_count = [0]

    def always_fails() -> str:
        call_count[0] += 1
        raise ValueError(f"attempt {call_count[0]}")

    with pytest.raises(ValueError):
        call_with_retry(always_fails, max_attempts=2, min_wait=0.01, max_wait=0.05)
    assert call_count[0] == 2


def test_a_circuit_opens_at_its_threshold_and_stops_calling():
    call_count = [0]
    circuit = Circuit("test_circuit", failure_threshold=3, recovery_timeout=60)

    def failing_service() -> str:
        call_count[0] += 1
        raise ConnectionError("unavailable")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            circuit.call(failing_service)
    for _ in range(2):
        with pytest.raises(CircuitOpenError):
            circuit.call(failing_service)
    assert call_count[0] == 3


def test_a_retry_logs_the_attempt_that_succeeded_and_no_other(caplog):
    """Success hung on tenacity's `after` hook, which runs after a *failed* attempt.

    So "Retry succeeded on attempt 2" was logged exactly when attempt 2
    failed, and the attempt that did succeed was never logged at all.
    """
    calls = [0]

    def flaky() -> str:
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionError(f"attempt {calls[0]}")
        return "ok"

    with caplog.at_level("INFO", logger=get_healing_logger().logger.name):
        assert call_with_retry(flaky, max_attempts=3, min_wait=0, max_wait=0) == "ok"

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
    circuit = Circuit("expects_connection", failure_threshold=2, recovery_timeout=60,
                      expected_exception=ConnectionError)

    def service() -> None:
        calls[0] += 1
        raise ValueError("not an outage")

    for _ in range(4):
        with pytest.raises(ValueError):
            circuit.call(service)
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


def test_a_circuit_never_serializes_the_calls_it_guards():
    """pybreaker held its lock across the call: a seat generating for minutes
    held up every status read through the daemon's circuit."""
    circuit = Circuit("not_serialized", failure_threshold=3, recovery_timeout=60)
    inside, release = threading.Event(), threading.Event()

    def long_call() -> str:
        inside.set()
        release.wait(5)
        return "slow"

    worker = threading.Thread(target=lambda: circuit.call(long_call))
    worker.start()
    try:
        assert inside.wait(5)
        started = time.monotonic()
        assert circuit.call(lambda: "quick") == "quick"
        assert time.monotonic() - started < 1
    finally:
        release.set()
        worker.join(5)

    # Resetting it does not wait on a call either.
    inside.clear()
    release.clear()
    worker = threading.Thread(target=lambda: circuit.call(long_call))
    worker.start()
    try:
        assert inside.wait(5)
        started = time.monotonic()
        reset_circuit("not_serialized")
        assert time.monotonic() - started < 1
    finally:
        release.set()
        worker.join(5)


def test_a_half_open_circuit_lets_one_trial_through_and_refuses_the_rest():
    circuit = Circuit("one_trial", failure_threshold=1, recovery_timeout=0.1)
    with pytest.raises(ConnectionError):
        circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    time.sleep(0.2)

    inside, release = threading.Event(), threading.Event()

    def trial() -> str:
        inside.set()
        release.wait(5)
        return "up"

    results: list[str] = []
    worker = threading.Thread(target=lambda: results.append(circuit.call(trial)))
    worker.start()
    try:
        assert inside.wait(5)
        assert _state("one_trial")["state"] == "half-open"
        called = []
        with pytest.raises(CircuitOpenError, match="a trial call is testing it now"):
            circuit.call(lambda: called.append(1))
        assert called == []
    finally:
        release.set()
        worker.join(5)
    assert results == ["up"]
    assert _state("one_trial")["state"] == "closed"


def test_a_failed_trial_reopens_the_circuit_for_another_cooldown():
    circuit = Circuit("trial_fails", failure_threshold=1, recovery_timeout=0.1)
    down = ConnectionError("down")
    with pytest.raises(ConnectionError):
        circuit.call(lambda: (_ for _ in ()).throw(down))
    time.sleep(0.2)
    with pytest.raises(ConnectionError):
        circuit.call(lambda: (_ for _ in ()).throw(down))
    assert circuit.is_open
    assert circuit.retry_in > 0
    with pytest.raises(CircuitOpenError):
        circuit.call(lambda: "refused")


def test_an_outage_is_journalled_once_however_long_it_lasts():
    """A daemon left down with the console open wrote the same refusal seventy
    times a minute, and a failed trial re-announced "OPENED after 3 failures"
    every cooldown: the 500-event journal held six minutes of one line. Now a
    spell says each thing once and counts the rest, and the close reports them."""
    journal = get_healing_logger()
    mark = journal.events()[-1]["seq"] if journal.events() else 0
    circuit = Circuit("long_outage", failure_threshold=1, recovery_timeout=0.05)

    def down() -> None:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        circuit.call(down)
    for _ in range(3):  # three cooldowns, three failed trials
        for _ in range(20):
            with pytest.raises(CircuitOpenError):
                circuit.call(lambda: "refused")
        time.sleep(0.06)
        with pytest.raises(ConnectionError):
            circuit.call(down)
    time.sleep(0.06)
    assert circuit.call(lambda: "up") == "up"

    events = [e for e in journal.events(since=mark) if e.get("function") == "long_outage"]
    actions = [e["action"] for e in events]
    assert actions == ["circuit_opened", "circuit_prevented", "circuit_half_open",
                       "circuit_trial_failed", "circuit_closed"]
    closed = events[-1]
    assert (closed["refused"], closed["failed_trials"]) == (60, 3)
    assert "60 call(s) refused, 3 trial(s) failed" in closed["message"]


def test_a_new_outage_is_journalled_afresh():
    """The counts belong to one spell: after a close, the next outage's first
    refusal and first failed trial are news again."""
    journal = get_healing_logger()
    mark = journal.events()[-1]["seq"] if journal.events() else 0
    circuit = Circuit("second_outage", failure_threshold=1, recovery_timeout=0.05)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
        with pytest.raises(CircuitOpenError):
            circuit.call(lambda: "refused")
        time.sleep(0.06)
        assert circuit.call(lambda: "up") == "up"
    actions = [e["action"] for e in journal.events(since=mark)
               if e.get("function") == "second_outage"]
    assert actions == ["circuit_opened", "circuit_prevented", "circuit_half_open",
                       "circuit_closed"] * 2


def test_a_run_that_starts_mid_outage_carries_the_refusal_in_its_own_record():
    """A spell journalled its first refusal once, so a run that began after it
    -- every call refused -- saved a snapshot whose healing record was empty.
    Each healing session the spell reaches is told once; nothing more."""
    journal = get_healing_logger()
    journal.end_healing_session()
    mark = journal.events()[-1]["seq"] if journal.events() else 0
    circuit = Circuit("outage_across_runs", failure_threshold=1, recovery_timeout=60)
    with pytest.raises(ConnectionError):
        circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(CircuitOpenError):
        circuit.call(lambda: "refused between runs")

    journal.start_healing_session("run-mid-outage")
    try:
        for _ in range(5):
            with pytest.raises(CircuitOpenError):
                circuit.call(lambda: "refused in the run")
    finally:
        journal.end_healing_session()
    with pytest.raises(CircuitOpenError):
        circuit.call(lambda: "refused after it")

    in_run = [e["action"] for e in journal.events(session_id="run-mid-outage")
              if e.get("function") == "outage_across_runs"]
    assert in_run == ["circuit_prevented"]
    every = [e["action"] for e in journal.events(since=mark)
             if e.get("function") == "outage_across_runs"]
    assert every == ["circuit_opened", "circuit_prevented", "circuit_prevented"]
    reset_circuit("outage_across_runs")


def test_a_trip_during_the_trial_keeps_the_spell_s_counts():
    """`trip` from half-open went through `open`, which started the books
    again, so the line that closed the spell under-reported it."""
    journal = get_healing_logger()
    mark = journal.events()[-1]["seq"] if journal.events() else 0
    circuit = Circuit("tripped_mid_trial", failure_threshold=1, recovery_timeout=0.05)
    with pytest.raises(ConnectionError):
        circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
    for _ in range(4):
        with pytest.raises(CircuitOpenError):
            circuit.call(lambda: "refused")
    time.sleep(0.06)
    with pytest.raises(ConnectionError):
        with circuit.guarding():
            circuit.trip("the load failed outright")
            raise ConnectionError("down")
    assert circuit.is_open
    time.sleep(0.06)
    assert circuit.call(lambda: "up") == "up"

    events = [e for e in journal.events(since=mark) if e.get("function") == "tripped_mid_trial"]
    assert [e["action"] for e in events].count("circuit_opened") == 1
    assert events[-1]["action"] == "circuit_closed"
    assert events[-1]["refused"] == 4


def test_a_success_clears_the_count_of_a_closed_circuit():
    circuit = Circuit("count_clears", failure_threshold=2, recovery_timeout=60)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            circuit.call(lambda: (_ for _ in ()).throw(ConnectionError("down")))
        assert circuit.call(lambda: "up") == "up"
    assert _state("count_clears") == {
        "name": "count_clears", "state": "closed", "failures": 0, "threshold": 2,
        "cooldown_s": 60, "retry_in_s": None,
    }


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
