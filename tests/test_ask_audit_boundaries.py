"""Boundary audit events for suspend and cancel."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import AUDIT_EVENT_ASK_SUSPEND, PIPELINE_SUSPEND_ID_INTENT_CONFIRM
from aetherdialect._contracts_core import PipelineSuspended
from aetherdialect._main_session import PipelineSession


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


@pytest.mark.fast
def test_suspend_emits_ask_suspend() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner)
    sess._turn_question = "how many customers?"
    ex = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "confirm?", None)

    sess._suspend_to_step(ex)

    audit_names = [call.args[0] for call in owner._audit_emit.call_args_list]
    assert AUDIT_EVENT_ASK_SUSPEND in audit_names
