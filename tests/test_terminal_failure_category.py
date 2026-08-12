"""Terminal SessionStep failure category mapping."""

from __future__ import annotations

import pytest

from aetherdialect._constants import SESSION_KIND_ERROR
from aetherdialect._contracts_base import Diagnostic, FailureCategory
from aetherdialect._contracts_core import SessionError, SessionOutcome, SessionStep
from aetherdialect._main_execution import MainExecutionOps


def _error_step(error: str, *, outcome: SessionOutcome = SessionOutcome.EXECUTION_FAILED) -> SessionStep:
    return SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_ERROR,
        error=SessionError(code=outcome),
        diagnostics=(
            Diagnostic(
                stage="execute",
                level="error",
                code="TEST",
                message=error,
            ),
        ),
    )


@pytest.mark.fast
def test_auth_error_is_transport_auth() -> None:
    cat = MainExecutionOps.failure_category_for_terminal_step(
        _error_step("password authentication failed for user postgres")
    )
    assert cat == FailureCategory.TRANSPORT_AUTH.value


@pytest.mark.fast
def test_sql_error_not_transport_auth() -> None:
    cat = MainExecutionOps.failure_category_for_terminal_step(_error_step("syntax error at or near SELECT"))
    assert cat != FailureCategory.TRANSPORT_AUTH.value
    assert cat == FailureCategory.EXECUTION_OTHER_ERROR.value


@pytest.mark.fast
def test_intent_parse_not_execution_other() -> None:
    cat = MainExecutionOps.failure_category_for_terminal_step(
        _error_step("intent_parse_failed: could not compose intent", outcome=SessionOutcome.PARSE_FAILED)
    )
    assert cat == FailureCategory.INTENT_ERROR.value
    assert cat != FailureCategory.EXECUTION_OTHER_ERROR.value


@pytest.mark.fast
def test_permission_not_execution_other() -> None:
    cat = MainExecutionOps.failure_category_for_terminal_step(
        _error_step("permission denied; contact your administrator", outcome=SessionOutcome.FORBIDDEN)
    )
    assert cat == FailureCategory.PERMISSION_ERROR.value
    assert cat != FailureCategory.EXECUTION_OTHER_ERROR.value
