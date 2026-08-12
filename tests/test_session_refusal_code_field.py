"""SessionStep.error carries structured refusal detail on terminal failures."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import SessionStep
from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
from aetherdialect._contracts_core import SessionError, SessionOutcome
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_session import PipelineSession


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner.dialect = "sqlite"
    return owner


@pytest.mark.fast
def test_session_step_exposes_error_without_refusal_fields() -> None:
    names = {f.name for f in fields(SessionStep)}
    assert "error" in names
    assert "refusal_diagnostic_code" not in names
    assert "refusal_code" not in names
    assert "status" not in names
    step = SessionStep(done=True, prompt=None, kind="result")
    assert step.error is None


@pytest.mark.fast
def test_validation_failed_sets_session_error_detail_code() -> None:
    session = PipelineSession(_session_owner())
    session._turn_question = "join orders to products"
    session._last_turn_outcome = {
        "outcome": "validation_failed",
        "error": "These tables could not be connected: orders, products.",
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
        "federation_source_id": None,
        "federation_phase": None,
        "federation_succeeded": (),
        "failure_kind": None,
        "retryable": None,
        "refusal_diagnostic_code": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    }
    with patch.object(session, "_emit_turn_llm_usage", return_value=()):
        step = session._completed_step()

    assert step.error is not None
    assert step.error.code == SessionOutcome.VALIDATION_FAILED
    assert step.error.detail_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE

    payload = MainExecutionOps.serialize_session_step(step)
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored.error is not None
    assert restored.error.detail_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    assert restored.error.code == SessionOutcome.VALIDATION_FAILED


@pytest.mark.fast
def test_session_error_roundtrip() -> None:
    step = SessionStep(
        done=True,
        prompt=None,
        kind="error",
        error=SessionError(
            code=SessionOutcome.FORBIDDEN,
            detail_code=DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
        ),
    )
    restored = MainExecutionOps.deserialize_session_step(MainExecutionOps.serialize_session_step(step))
    assert restored.error == step.error
