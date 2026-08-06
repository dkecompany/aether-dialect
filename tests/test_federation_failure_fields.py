"""Prepare-phase federation failures must surface attribution on SessionStep."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import SessionStep
from aetherdialect._main_execution import PipelineSession


def _session_with_outcome(outcome: str, **extra: object) -> PipelineSession:
    session = PipelineSession.__new__(PipelineSession)
    session._turn_question = "how many orders"
    session._last_turn_outcome = {
        "outcome": outcome,
        "error": "member validation failed",
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
        "federation_source_id": "west",
        "federation_phase": "prepare",
        "federation_succeeded": (),
        "failure_kind": "validation_failed",
        "retryable": False,
        **extra,
    }
    session._owner = MagicMock()
    session._owner._llm_config = MagicMock(provider="mock")
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
    return session


@pytest.mark.fast
@pytest.mark.parametrize(
    "outcome",
    [
        "validation_failed",
        "schema_invalid_declined",
        "permission_denied",
    ],
)
def test_prepare_failure_outcome_carries_federation_fields_on_session_step(outcome: str) -> None:
    """Federation source/phase from prepare failures must appear on the terminal SessionStep."""
    session = _session_with_outcome(outcome)
    step = session._completed_step()
    assert isinstance(step, SessionStep)
    assert step.federation_source_id == "west"
    assert step.federation_phase == "prepare"


@pytest.mark.fast
def test_federation_partial_failure_still_carries_federation_fields() -> None:
    """Regression: partial failure attribution remains on SessionStep."""
    session = _session_with_outcome(
        "federation_partial_failure",
        error=None,
        federation_succeeded=(("east", 2, "2026-01-01T00:00:00+00:00"),),
        retryable=True,
    )
    step = session._completed_step()
    assert step.federation_source_id == "west"
    assert step.federation_phase == "prepare"
    assert step.retryable is True
    assert step.federation_succeeded == (("east", 2, "2026-01-01T00:00:00+00:00"),)
