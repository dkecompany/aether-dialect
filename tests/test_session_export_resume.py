"""Export and restore suspended PipelineSession state across processes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    PIPELINE_SUSPEND_ID_SQL,
    SESSION_KIND_RESULT,
    SUSPEND_STATE_FORMAT_VERSION,
)
from aetherdialect._contracts_core import (
    GenerationPath,
    InteractiveTailSnapshot,
    PipelineSuspended,
    RuntimeIntent,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
)
from aetherdialect._main_session import PipelineSession


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._dialect = MagicMock()
    owner._dialect.name = "postgresql"
    owner._dialect.config = owner._runtime_config
    owner._audit_emit = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._artifacts_dir = None
    owner._domain_knowledge = None
    owner._phase_callback = None
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._schema_terms = set()

    def _open_session(**kwargs):
        from aetherdialect._main_session import PipelineSession

        return PipelineSession(
            owner,
            mode=kwargs.get("mode", "writer"),
            space_name=str(kwargs.get("space", "master")),
            data_row_cap=kwargs.get("data_row_cap"),
        )

    owner.session = MagicMock(side_effect=_open_session)
    return owner


def _sql_feedback_suspend(owner: MagicMock) -> PipelineSession:
    intent = RuntimeIntent(
        tables=[],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    tail = InteractiveTailSnapshot(
        q_norm="how many rows",
        intent=intent,
        schema=owner._schema_graph,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema_terms=set(),
        dialect=owner._dialect,
        semantic_warnings=(),
        has_union_match=False,
        cols_changed=False,
        matched_template=None,
        union_select_cols=None,
        structural_match_templates=(),
        ikey="k",
        intent_sim=0.0,
    )
    gen_out = SqlGenerationOutcome(
        sql="SELECT 1",
        success=True,
        generation_path=GenerationPath.EXACT_QUESTION_REUSE,
        matched_template=None,
    )
    ctx = SqlFeedbackSuspendContext(
        tail=tail,
        execution_intent=intent,
        sql="SELECT 1",
        preview_rows=((1,),),
        sql_parameters=(("p", 1),),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
    )
    sess = PipelineSession(owner, mode="reader", space_name="analytics", data_row_cap=10)
    sess._suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx)
    sess._session_busy = True
    sess._turn_question = tail.q_norm
    return sess


@pytest.mark.fast
def test_step_after_restore_completes_yes_path() -> None:
    owner = _session_owner()
    sess = _sql_feedback_suspend(owner)

    payload = sess.export_serialized_state()
    assert payload["format_version"] == SUSPEND_STATE_FORMAT_VERSION
    assert payload.get("mode") == "reader"
    assert payload.get("space_name") == "analytics"
    assert payload.get("data_row_cap") == 10
    assert payload.get("payload") is not None
    assert payload["payload"]["sql"] == "SELECT 1"
    assert payload["payload"]["sql_parameters"] == [["p", 1]]
    assert payload["payload"]["preview_rows"] == [[1]]

    restored = PipelineSession.restore_serialized_state(owner, payload)
    assert restored._session_mode == "reader"
    assert restored.space_name == "analytics"
    assert restored._data_row_cap == 10
    assert restored._suspended is not None
    assert isinstance(restored._suspended.payload, SqlFeedbackSuspendContext)

    with patch(
        "aetherdialect._main_interactive.MainInteractiveOps._reexecute_suspend_sql_rows",
        return_value=([(1,)], None),
    ):
        with patch("aetherdialect._main_interactive.handle_user_feedback"):
            step = restored.step("y")

    assert step.done is True
    assert step.kind == SESSION_KIND_RESULT
    assert step.error is None
