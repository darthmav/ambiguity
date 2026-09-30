"""
Structured logging module for self-healing actions.

Provides logging with clear severity levels for auditability:
- DEBUG: Detailed healing process information
- INFO: Successful healing actions
- WARNING: Recovery attempts initiated
- ERROR: Healing failures
- CRITICAL: System-level healing issues requiring intervention
"""

import logging
import sys
from datetime import UTC, datetime


class SelfHealingLogger:
    """
    Structured logger for self-healing actions with severity levels.

    All healing actions are logged with timestamps and severity levels
    for full auditability.
    """

    def __init__(self, name: str = "self_healing", level: int = logging.DEBUG):
        """
        Initialize the self-healing logger.

        Args:
            name: Logger name (default: "self_healing")
            level: Logging level (default: DEBUG)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid adding multiple handlers if logger already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._healing_session_id: str | None = None

    def start_healing_session(self, session_id: str) -> None:
        """Start a new healing session for tracking related actions."""
        self._healing_session_id = session_id
        self.info(f"Healing session started: {session_id}")

    def end_healing_session(self) -> None:
        """End the current healing session."""
        if self._healing_session_id:
            self.info(f"Healing session completed: {self._healing_session_id}")
            self._healing_session_id = None

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug-level healing information."""
        extra = self._build_extra(**kwargs)
        self.logger.debug(message, extra=extra)

    def info(self, message: str, **kwargs: object) -> None:
        """Log info-level healing actions (successful recoveries)."""
        extra = self._build_extra(**kwargs)
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning-level healing attempts."""
        extra = self._build_extra(**kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error-level healing failures."""
        extra = self._build_extra(**kwargs)
        self.logger.error(message, extra=extra)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical-level system healing issues."""
        extra = self._build_extra(**kwargs)
        self.logger.critical(message, extra=extra)

    def _build_extra(self, **kwargs: object) -> dict[str, object]:
        """Build extra context for log messages."""
        extra: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if self._healing_session_id:
            extra["session_id"] = self._healing_session_id
        extra.update(kwargs)
        return extra

    def log_retry_attempt(self, func_name: str, attempt: int, max_attempts: int,
                          error: str) -> None:
        """Log a retry attempt with context."""
        self.warning(
            f"Retry attempt {attempt}/{max_attempts} for '{func_name}': {error}",
            action="retry",
            function=func_name,
            attempt=attempt,
            max_attempts=max_attempts
        )

    def log_retry_success(self, func_name: str, attempt: int) -> None:
        """Log successful retry."""
        self.info(
            f"Retry succeeded for '{func_name}' on attempt {attempt}",
            action="retry_success",
            function=func_name,
            attempt=attempt
        )

    def log_retry_exhausted(self, func_name: str, max_attempts: int) -> None:
        """Log when all retry attempts are exhausted."""
        self.error(
            f"All {max_attempts} retry attempts exhausted for '{func_name}'",
            action="retry_exhausted",
            function=func_name,
            max_attempts=max_attempts
        )

    def log_circuit_opened(self, func_name: str, failure_count: int) -> None:
        """Log when circuit breaker opens."""
        self.warning(
            f"Circuit breaker OPENED for '{func_name}' after {failure_count} failures",
            action="circuit_opened",
            function=func_name,
            failure_count=failure_count
        )

    def log_circuit_closed(self, func_name: str) -> None:
        """Log when circuit breaker closes (recovery)."""
        self.info(
            f"Circuit breaker CLOSED for '{func_name}' - service recovered",
            action="circuit_closed",
            function=func_name
        )

    def log_circuit_half_open(self, func_name: str) -> None:
        """Log when circuit breaker enters half-open state."""
        self.info(
            f"Circuit breaker HALF-OPEN for '{func_name}' - testing recovery",
            action="circuit_half_open",
            function=func_name
        )

    def log_health_check(self, component: str, status: str, details: str = "") -> None:
        """Log health check results."""
        level = self.info if status == "healthy" else self.warning
        level(
            f"Health check '{component}': {status}" + (f" - {details}" if details else ""),
            action="health_check",
            component=component,
            status=status
        )

    def log_recovery_action(self, action_name: str, target: str, success: bool,
                            details: str = "") -> None:
        """Log a recovery action attempt."""
        if success:
            self.info(
                f"Recovery action '{action_name}' on '{target}': SUCCESS" +
                (f" - {details}" if details else ""),
                action="recovery",
                recovery_action=action_name,
                target=target,
                success=success
            )
        else:
            self.error(
                f"Recovery action '{action_name}' on '{target}': FAILED" +
                (f" - {details}" if details else ""),
                action="recovery_failed",
                recovery_action=action_name,
                target=target,
                success=success
            )


# One logger per name. It was a single global, so whichever name asked first
# was the only one there ever was: every decorator takes a `logger_name`, and
# every one after the first was handed "self_healing" whatever it passed.
_healing_loggers: dict[str, SelfHealingLogger] = {}


def get_healing_logger(name: str = "self_healing",
                       level: int = logging.DEBUG) -> SelfHealingLogger:
    """
    Get or create the self-healing logger for `name`.

    The same name always returns the same instance; `level` applies only to
    the call that creates it.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        SelfHealingLogger instance
    """
    logger = _healing_loggers.get(name)
    if logger is None:
        logger = _healing_loggers[name] = SelfHealingLogger(name, level)
    return logger
