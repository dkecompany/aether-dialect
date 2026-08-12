"""Integration tests for session scope, persistence, and data row caps."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._constants import PIPELINE_SUSPEND_ID_INTENT_CONFIRM, SESSION_KIND_RESULT
from aetherdialect._contracts_base import SpaceContext
from aetherdialect._contracts_core import PipelineSuspended, SessionStep
from aetherdialect._main_session import PipelineSession
from aetherdialect._templates_ops import TemplateOps


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._artifacts_dir = None
    owner._store_by_space = {}
    owner._templates_by_space = {}

    def _open_session(**kwargs):
        return PipelineSession(
            owner,
            mode=kwargs.get("mode", "writer"),
            space_name=str(kwargs.get("space", "master")),
            data_row_cap=kwargs.get("data_row_cap"),
        )

    owner.session = MagicMock(side_effect=_open_session)
    return owner


@pytest.mark.fast
def test_session_ephemeral_scope_narrows_tables() -> None:
    from aetherdialect._main_execution import MainExecutionOps
    from aetherdialect._main_session import PipelineSession

    owner = _session_owner()
    tables, columns, deny_obj, deny_col = MainExecutionOps.intersect_space_scope(
        frozenset({"film", "customer"}),
        frozenset(),
        frozenset(),
        frozenset(),
        SpaceContext(tables=frozenset({"film"})),
    )
    sess = PipelineSession(
        owner,
        space_tables=tables,
        space_columns=columns,
        space_deny_objects=deny_obj,
        space_deny_columns=deny_col,
        visible_objects=tables,
    )
    assert sess.space_tables == frozenset({"film"})


@pytest.mark.fast
def test_session_ephemeral_scope_cannot_widen() -> None:
    from aetherdialect._main_execution import MainExecutionOps
    from aetherdialect._main_session import PipelineSession

    owner = _session_owner()
    tables, columns, deny_obj, deny_col = MainExecutionOps.intersect_space_scope(
        frozenset({"film"}),
        frozenset(),
        frozenset(),
        frozenset(),
        SpaceContext(tables=frozenset({"film", "customer"})),
    )
    sess = PipelineSession(
        owner,
        space_tables=tables,
        space_columns=columns,
        space_deny_objects=deny_obj,
        space_deny_columns=deny_col,
        visible_objects=tables,
    )
    assert sess.space_tables == frozenset({"film"})


@pytest.mark.fast
def test_pipeline_session_export_and_restore_suspended_state() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner)
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "confirm?", None)
    with patch("aetherdialect._main_init.MainInitOps.interactive_run_once", side_effect=suspended):
        sess.ask("list films")
    payload = sess.export_serialized_state()
    restored = PipelineSession.restore_serialized_state(owner, payload)
    assert restored.awaiting_prompt() is True
    assert restored._suspended is not None
    assert restored._suspended.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM


@pytest.mark.fast
def test_terminal_success_has_no_error() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner)
    sess._last_turn_outcome = {"outcome": "success", "rows": [], "columns": []}
    sess._turn_question = "q"
    step = sess._completed_step()
    assert step.kind == SESSION_KIND_RESULT
    assert step.error is None
    assert step.answer is None


@pytest.mark.fast
def test_data_row_cap_sets_truncated_flag() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner, data_row_cap=2)
    sess._last_turn_outcome = {
        "outcome": "success",
        "rows": [(1,), (2,), (3,)],
        "columns": ["id"],
    }
    sess._turn_question = "q"
    step = sess._completed_step()
    assert step.data is not None
    assert len(step.data) == 2
    assert step.data_truncated is True


@pytest.mark.fast
def test_session_step_roundtrip_includes_truncated() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    step = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        data=pd.DataFrame({"id": [1, 2, 3]}),
        data_truncated=True,
    )
    restored = MainExecutionOps.deserialize_session_step(MainExecutionOps.serialize_session_step(step))
    assert restored.data_truncated is True
