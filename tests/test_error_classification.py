"""Database driver error classification table coverage."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if "aetherdialect" not in sys.modules:
    _pkg = types.ModuleType("aetherdialect")
    _pkg.__path__ = [str(_SRC / "aetherdialect")]
    _pkg.__package__ = "aetherdialect"
    sys.modules["aetherdialect"] = _pkg

pre_fix_failure: str | None = None

_NAMED_TRANSIENT_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("connection reset by peer", "connection reset"),
    ("connection refused", "connection refused"),
    ("server closed the connection unexpectedly", "server closed"),
    ("deadlock detected", "deadlock"),
    ("lock wait timeout exceeded", "lock wait timeout"),
    ("canceling statement due to statement timeout", "statement timeout"),
    ("too many connections", "too many connections"),
    ("rate limit exceeded", "rate limit"),
    ("temporary failure in name resolution", "temporary name resolution failure"),
)


@pytest.mark.parametrize(
    ("message", "label"),
    _NAMED_TRANSIENT_CONDITIONS,
    ids=[label for _, label in _NAMED_TRANSIENT_CONDITIONS],
)
def test_classification_table_covers_named_conditions(message: str, label: str) -> None:
    global pre_fix_failure
    from aetherdialect._utils import classify_database_error

    exc = RuntimeError(message)
    got = classify_database_error(exc)
    if got != "transient":
        pre_fix_failure = f"{label}: classify_database_error({message!r}) returned {got!r}, expected 'transient'"
    assert got == "transient", pre_fix_failure


def test_unknown_classification_is_not_retryable() -> None:
    global pre_fix_failure
    from aetherdialect._contracts_base import DatabaseExecutionError
    from aetherdialect._utils import classify_database_error

    exc = RuntimeError("syntax error at or near SELECT")
    classification = classify_database_error(exc)
    if classification != "unknown":
        pre_fix_failure = f"expected unknown for semantic error, got {classification!r}"
    assert classification == "unknown", pre_fix_failure

    wrapped = DatabaseExecutionError(
        "Database execution failed.",
        driver_class="RuntimeError",
        classification=classification,
        retryable=False,
        driver_detail={"message": str(exc)},
    )
    if wrapped.retryable:
        pre_fix_failure = "unknown classification must not be retryable"
    assert not wrapped.retryable, pre_fix_failure
    from aetherdialect._contracts_base import RetryableError

    assert not isinstance(wrapped, RetryableError), pre_fix_failure


def test_artifact_lock_timeout_error_is_retryable_subclass() -> None:
    from aetherdialect._contracts_base import ArtifactLockTimeoutError, RetryableError

    assert issubclass(ArtifactLockTimeoutError, RetryableError)
