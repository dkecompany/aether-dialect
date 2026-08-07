"""SessionStep.refusal_diagnostic_code aligns with documented refusal field."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
from aetherdialect._main_execution import MainExecutionOps, PipelineSession


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
def test_validation_failed_sets_refusal_diagnostic_code() -> None:
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

    assert step.status == "validation_failed"
    assert step.refusal_diagnostic_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    assert step.refusal_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE

    payload = MainExecutionOps.serialize_session_step(step)
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored.refusal_diagnostic_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    assert restored.refusal_code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
