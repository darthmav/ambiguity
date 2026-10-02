"""Structured, severity-levelled logging for self-healing actions, with a journal.

Every healing action -- a retry, a circuit opening or closing, a health check,
a recovery -- is logged at the severity that says what it means:

- DEBUG: detail of a healing process
- INFO: a successful recovery
- WARNING: a recovery attempt, or something unhealthy
- ERROR: a healing failure
- CRITICAL: a system-level problem healing cannot fix

Each is also kept in an in-memory journal, tagged with the healing session it
happened in, so the console can read them back (`events`) and a run's snapshot
can carry its own.
"""

import itertools
import logging
import sys
import threading
from collections import deque
from datetime import UTC, datetime
from typing import Any

# How many events one journal keeps. The console reads from the last sequence
# number it saw, and a run copies its own session's events out when it ends.
JOURNAL_SIZE = 500


class SelfHealingLogger:
    """A named healing logger: stdout lines, plus the journal behind them."""

    def __init__(self, name: str = "self_healing", level: int = logging.DEBUG) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self.logger.addHandler(handler)

        self._healing_session_id: str | None = None
        self._journal: deque[dict[str, Any]] = deque(maxlen=JOURNAL_SIZE)
        self._sequence = itertools.count(1)
        self._lock = threading.Lock()

    def start_healing_session(self, session_id: str) -> None:
        """Tag every event from here on with `session_id` -- a run's id."""
        self._healing_session_id = session_id
        self.info(f"Healing session started: {session_id}")

    def end_healing_session(self) -> None:
        """Stop tagging events with the current session."""
        if self._healing_session_id:
            self.info(f"Healing session completed: {self._healing_session_id}")
            self._healing_session_id = None

    @property
    def session_id(self) -> str | None:
        return self._healing_session_id

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug-level healing information."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: object) -> None:
        """Log info-level healing actions (successful recoveries)."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning-level healing attempts."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error-level healing failures."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical-level system healing issues."""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, fields: dict[str, object]) -> None:
        if not self.logger.isEnabledFor(level):
            return
        extra = self._build_extra(**fields)
        with self._lock:
            self._journal.append({
                "seq": next(self._sequence),
                "level": logging.getLevelName(level),
                "message": message,
                **extra,
            })
        self.logger.log(level, message, extra=extra)

    def _build_extra(self, **kwargs: object) -> dict[str, object]:
        """The structured fields every event carries, plus the caller's own."""
        extra: dict[str, object] = {"timestamp": datetime.now(UTC).isoformat()}
        if self._healing_session_id:
            extra["session_id"] = self._healing_session_id
        extra.update(kwargs)
        return extra

    def events(self, since: int = 0, session_id: str | None = None) -> list[dict[str, Any]]:
        """Journal events after sequence number `since`, oldest first.

        `session_id` narrows them to one healing session.
        """
        with self._lock:
            return [
                dict(event) for event in self._journal
                if event["seq"] > since
                and (session_id is None or event.get("session_id") == session_id)
            ]

    def log_retry_attempt(self, func_name: str, attempt: int, max_attempts: int,
                          error: str) -> None:
        """Log a retry attempt with context."""
        self.warning(
            f"Retry attempt {attempt}/{max_attempts} for '{func_name}': {error}",
            action="retry",
            function=func_name,
            attempt=attempt,
            max_attempts=max_attempts,
        )

    def log_retry_success(self, func_name: str, attempt: int) -> None:
        """Log successful retry."""
        self.info(
            f"Retry succeeded for '{func_name}' on attempt {attempt}",
            action="retry_success",
            function=func_name,
            attempt=attempt,
        )

    def log_retry_exhausted(self, func_name: str, max_attempts: int) -> None:
        """Log when all retry attempts are exhausted."""
        self.error(
            f"All {max_attempts} retry attempts exhausted for '{func_name}'",
            action="retry_exhausted",
            function=func_name,
            max_attempts=max_attempts,
        )

    def log_retry_abandoned(self, func_name: str, attempt: int) -> None:
        """Log a retry given up before its attempts ran out."""
        self.warning(
            f"Gave up retrying '{func_name}' after attempt {attempt}: told to stop",
            action="retry_abandoned",
            function=func_name,
            attempt=attempt,
        )

    def log_circuit_opened(self, func_name: str, failure_count: int) -> None:
        """Log when circuit breaker opens; `failure_count` is 0 for a manual trip."""
        cause = f" after {failure_count} failures" if failure_count else ""
        self.warning(
            f"Circuit breaker OPENED for '{func_name}'{cause}",
            action="circuit_opened",
            function=func_name,
            failure_count=failure_count,
        )

    def log_circuit_closed(self, func_name: str) -> None:
        """Log when circuit breaker closes (recovery)."""
        self.info(
            f"Circuit breaker CLOSED for '{func_name}' - service recovered",
            action="circuit_closed",
            function=func_name,
        )

    def log_circuit_half_open(self, func_name: str) -> None:
        """Log when circuit breaker enters half-open state."""
        self.info(
            f"Circuit breaker HALF-OPEN for '{func_name}' - testing recovery",
            action="circuit_half_open",
            function=func_name,
        )

    def log_health_check(self, component: str, status: str, details: str = "") -> None:
        """Log health check results."""
        level = self.info if status == "healthy" else self.warning
        level(
            f"Health check '{component}': {status}" + (f" - {details}" if details else ""),
            action="health_check",
            component=component,
            status=status,
        )

    def log_recovery_action(self, action_name: str, target: str, success: bool,
                            details: str = "") -> None:
        """Log a recovery action attempt."""
        if success:
            self.info(
                f"Recovery action '{action_name}' on '{target}': SUCCESS"
                + (f" - {details}" if details else ""),
                action="recovery",
                recovery_action=action_name,
                target=target,
                success=success,
            )
        else:
            self.error(
                f"Recovery action '{action_name}' on '{target}': FAILED"
                + (f" - {details}" if details else ""),
                action="recovery_failed",
                recovery_action=action_name,
                target=target,
                success=success,
            )


# One logger per name: every decorator takes a `logger_name`, and the same
# name must always reach the same journal.
_healing_loggers: dict[str, SelfHealingLogger] = {}
_loggers_lock = threading.Lock()


def get_healing_logger(name: str = "self_healing",
                       level: int = logging.DEBUG) -> SelfHealingLogger:
    """The self-healing logger for `name`, created on first use.

    `level` applies only to the call that creates it.
    """
    with _loggers_lock:
        logger = _healing_loggers.get(name)
        if logger is None:
            logger = _healing_loggers[name] = SelfHealingLogger(name, level)
        return logger
