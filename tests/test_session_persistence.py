"""Versioned serialisation for SessionStep and suspended-session state."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._constants import SESSION_PERSISTENCE_FORMAT_VERSION, SUSPEND_STATE_FORMAT_VERSION
from aetherdialect._contracts_base import ConfigError, Diagnostic, IntentSummary, SessionStep
from aetherdialect._main_execution import MainExecutionOps


def _sample_step(*, with_data: bool = False) -> SessionStep:
    data = None
    if with_data:
        data = pd.DataFrame({"id": [1, 2], "name": ["alpha", "beta"]})
    return SessionStep(
        done=False,
        prompt="Proceed?",
        kind="awaiting_intent_confirm",
        sql="SELECT id, name FROM customers",
        data=data,
        message="Intent looks correct.",
        error=None,
        intent_summary=IntentSummary(
            tables=("customers",),
            select_cols=("customers.id", "customers.name"),
            filters=(),
            group_by=(),
            order_by=(),
            limit=None,
            natural_language="show customers",
        ),
        diagnostics=(
            Diagnostic(
                stage="interpret",
                level="info",
                code="intent_ready",
                message="intent resolved",
                details=(("tables", "customers"),),
                duration_ms=12,
                source_id="primary",
            ),
        ),
        status=None,
        reply_shape="yes_no",
        semantic_warnings=("ambiguous join",),
        retryable=False,
    )


@pytest.mark.fast
def test_session_step_roundtrip_without_dataframe() -> None:
    step = _sample_step(with_data=False)
    payload = MainExecutionOps.serialize_session_step(step)
    assert payload["format_version"] == SESSION_PERSISTENCE_FORMAT_VERSION
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored == step


@pytest.mark.fast
def test_session_step_roundtrip_with_dataframe() -> None:
    step = _sample_step(with_data=True)
    payload = MainExecutionOps.serialize_session_step(step)
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored.done == step.done
    assert restored.kind == step.kind
    assert restored.sql == step.sql
    assert restored.intent_summary == step.intent_summary
    assert restored.diagnostics == step.diagnostics
    assert step.data is not None
    assert restored.data is not None
    pd.testing.assert_frame_equal(restored.data, step.data)


@pytest.mark.fast
def test_session_step_version_mismatch_refuses() -> None:
    step = _sample_step(with_data=False)
    payload = MainExecutionOps.serialize_session_step(step)
    payload["format_version"] = "9.9.9"
    with pytest.raises((ValueError, ConfigError), match=r"format_version"):
        MainExecutionOps.deserialize_session_step(payload)


@pytest.mark.fast
def test_suspended_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Payload:
        suspended_at = None

    monkeypatch.setattr(
        MainExecutionOps,
        "_serialize_pipeline_suspend_payload",
        lambda state_id, payload: {"type": "intent_confirm"},
    )
    monkeypatch.setattr(
        MainExecutionOps,
        "_deserialize_pipeline_suspend_payload",
        lambda state_id, raw, *, owner=None: _Payload(),
    )
    payload = MainExecutionOps.serialize_suspended_state(
        state_id="intent_confirm",
        message="Confirm intent?",
        choice_queue=[("intent_confirm", "y"), ("execute", "n")],
        turn_question="show customers",
        resume_choice_stage_id="intent_confirm",
        suspend_payload=_Payload(),
        policy_ttl_seconds=120,
    )
    assert payload["format_version"] == SUSPEND_STATE_FORMAT_VERSION
    assert "payload" in payload
    restored = MainExecutionOps.deserialize_suspended_state(payload)
    assert restored["state_id"] == "intent_confirm"
    assert restored["message"] == "Confirm intent?"
    assert restored["choice_queue"] == [("intent_confirm", "y"), ("execute", "n")]
    assert restored["turn_question"] == "show customers"
    assert restored["resume_choice_stage_id"] == "intent_confirm"
    assert restored["policy_ttl_seconds"] == 120
    assert restored["suspend_payload"] is not None


@pytest.mark.fast
def test_suspended_state_version_mismatch_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Payload:
        suspended_at = None

    monkeypatch.setattr(
        MainExecutionOps,
        "_serialize_pipeline_suspend_payload",
        lambda state_id, payload: {"type": "execute"},
    )
    payload = MainExecutionOps.serialize_suspended_state(
        state_id="execute",
        message="Run SQL?",
        choice_queue=[],
        turn_question=None,
        suspend_payload=_Payload(),
    )
    payload["format_version"] = "9.9.9"
    with pytest.raises((ValueError, ConfigError), match=r"format_version"):
        MainExecutionOps.deserialize_suspended_state(payload)
