"""Runtime resilience: retries with backoff, circuit breakers, and a healing journal.

The mechanisms wrap a call rather than living inside it, so the code they
protect stays free of retry loops. Where the application uses them, and what
each protects, is listed in CLAUDE.md under "Self-healing".
"""

from .decorators import (
    Circuit,
    CircuitOpenError,
    call_with_retry,
    circuit_states,
    exception_chain,
    reset_circuit,
)
from .logger import SelfHealingLogger, get_healing_logger

__all__ = [
    "Circuit",
    "CircuitOpenError",
    "SelfHealingLogger",
    "call_with_retry",
    "circuit_states",
    "exception_chain",
    "get_healing_logger",
    "reset_circuit",
]

__version__ = "2.0.0"
__status__ = "active"
