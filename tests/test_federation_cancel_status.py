"""Federation cancellation terminal SessionStep contract."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import SESSION_KIND_ERROR
from aetherdialect._contracts_base import FailureCategory
from aetherdialect._contracts_core import SessionOutcome
from aetherdialect._main_session import PipelineSession


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


@pytest.mark.fast
def test_cancel_step_kind_is_error() -> None:
    sess = PipelineSession(_session_owner())
    sess._session_busy = True
    sess._turn_question = "federated query"
    sess.note_turn_outcome(
        outcome="federation_turn_cancelled",
        federation_source_id="src_a",
        federation_phase="member",
        failure_kind=FailureCategory.FEDERATION_TURN_CANCELLED.value,
    )

    step = sess._completed_step()

    assert step.kind == SESSION_KIND_ERROR
    assert step.error is not None
    assert step.error.code == SessionOutcome.CANCELLED
    assert step.error.source_id == "src_a"
    assert step.error.phase == "member"
    assert FailureCategory.FEDERATION_TURN_CANCELLED.value  # failure kind still recorded upstream
