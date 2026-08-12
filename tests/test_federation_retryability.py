"""Federation partial-failure retryability must be a RetryableError and surface on the session step."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import (
    DatabasePingFailed,
    FederationPartialFailureError,
    RetryableError,
    StatementTimeoutError,
)
from aetherdialect._contracts_core import SessionStep
from aetherdialect._contracts_schema import FederationPlanTemplate
from aetherdialect._federation_execute import federation_member_timeout_error
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_session import PipelineSession
from aetherdialect._pipeline_execute import _raise_partial_member_failure


@pytest.mark.fast
def test_retryable_partial_failure_isinstance_retryable_error() -> None:
    exc = FederationPartialFailureError(
        "member a failed",
        source_id="a",
        phase="member",
        succeeded=(),
        retryable=True,
    )
    assert isinstance(exc, RetryableError)
    assert exc.retryable is True


@pytest.mark.fast
def test_non_retryable_partial_failure_is_not_retryable_error() -> None:
    exc = FederationPartialFailureError(
        "member a failed",
        source_id="a",
        phase="member",
        succeeded=(),
        retryable=False,
    )
    assert not isinstance(exc, RetryableError)
    assert exc.retryable is False


@pytest.mark.fast
def test_raise_partial_member_failure_promotes_transient_cause_to_retryable() -> None:
    with pytest.raises(FederationPartialFailureError) as exc_info:
        _raise_partial_member_failure(
            StatementTimeoutError("statement timeout"),
            source_id="b",
            phase="member",
            succeeded=(("a", 1, "2026-01-01T00:00:00+00:00"),),
        )
    exc = exc_info.value
    assert isinstance(exc, RetryableError)
    assert exc.retryable is True
    assert exc.source_id == "b"


@pytest.mark.fast
def test_raise_partial_member_failure_non_transient_not_retryable() -> None:
    with pytest.raises(FederationPartialFailureError) as exc_info:
        _raise_partial_member_failure(
            ValueError("syntax error"),
            source_id="b",
            phase="member",
            succeeded=(),
        )
    exc = exc_info.value
    assert not isinstance(exc, RetryableError)
    assert exc.retryable is False


@pytest.mark.fast
def test_partial_failure_interactive_turn_surfaces_retryable_on_step() -> None:
    owner = MagicMock()
    port = MagicMock()
    port._pending_federation_plan_template = FederationPlanTemplate(
        plan_id="plan1",
        composite_schema_graph_id="cg",
        intent_key="ik",
        step_fingerprints=(),
        combine_hash="h",
    )
    exc = FederationPartialFailureError(
        "member b failed",
        source_id="b",
        phase="member",
        succeeded=(("a", 2, "2026-01-01T00:00:00+00:00"),),
        retryable=True,
    )
    with patch("aetherdialect._federation_execute.save_federation_plan_template"):
        MainExecutionOps._handle_federation_partial_failure_interactive(port, owner, exc)
    kwargs = port.note_turn_outcome.call_args.kwargs
    assert kwargs["retryable"] is True


@pytest.mark.fast
def test_session_step_carries_retryable_for_federation_partial_failure() -> None:
    session = PipelineSession.__new__(PipelineSession)
    session._turn_question = "q"
    session._last_turn_outcome = {
        "outcome": "federation_partial_failure",
        "error": None,
        "sql": None,
        "rows": None,
        "columns": None,
        "rejection_bucket": None,
        "intent": None,
        "matched_template": None,
        "template_history_index": None,
        "federated_bundle": None,
        "federated_plan": None,
        "generation_path": None,
        "federation_source_id": "b",
        "federation_phase": "member",
        "federation_succeeded": (("a", 2, "2026-01-01T00:00:00+00:00"),),
        "failure_kind": None,
        "retryable": True,
    }
    session._owner = MagicMock()
    session._owner._llm_config = MagicMock(provider="sandbox")
    session._owner._audit_emit = None
    session._turn_llm_usage_start = 0
    session._session_busy = True
    session._session_busy_lock = __import__("threading").Lock()
    session._reset_after_turn = MagicMock()
    session._parameters_for_completed_turn = MagicMock(return_value=())
    session._emit_turn_llm_usage = MagicMock(side_effect=lambda **kw: kw.get("diagnostics", ()))
    session._turn_accumulated_diagnostics = []
    session._data_row_cap = None
    session._apply_data_row_cap = PipelineSession._apply_data_row_cap.__get__(session, PipelineSession)
    session._mk_step = PipelineSession._mk_step.__get__(session, PipelineSession)
    session._audit_ask_emit = MagicMock()

    step = session._completed_step()
    assert isinstance(step, SessionStep)
    assert step.error is not None
    assert step.error.source_id == "b"
    assert step.error.phase == "member"


@pytest.mark.fast
def test_retryable_cause_via_member_execution_wraps_retryable() -> None:
    with pytest.raises(FederationPartialFailureError) as exc_info:
        _raise_partial_member_failure(
            DatabasePingFailed("connection reset"),
            source_id="a",
            phase="member",
            succeeded=(),
        )
    assert isinstance(exc_info.value, RetryableError)


@pytest.mark.fast
def test_member_timeout_cap_exceeded_remains_retryable_in_partial_failure() -> None:
    timeout_exc = federation_member_timeout_error("b", StatementTimeoutError("statement timeout"))
    with pytest.raises(FederationPartialFailureError) as exc_info:
        _raise_partial_member_failure(
            timeout_exc,
            source_id="b",
            phase="member",
            succeeded=(),
        )
    exc = exc_info.value
    assert isinstance(exc, RetryableError)
    assert exc.retryable is True
