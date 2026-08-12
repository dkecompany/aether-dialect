"""SessionStep.template_id exposes matched or accepted template id."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_SQL,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_RESULT,
)
from aetherdialect._contracts_base import ApprovalState
from aetherdialect._contracts_core import (
    ConcreteIntent,
    DirectReuseSuspendContext,
    GenerationPath,
    InteractiveTailSnapshot,
    PipelineSuspended,
    RuntimeIntent,
    SessionStep,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._main_session import PipelineSession
from aetherdialect._pipeline_execute import complete_direct_sql_reuse_user_choice


def _tmpl(
    *,
    tid: str = "T0777",
    question: str = "how many",
    pending: bool = False,
    sql: str = "SELECT COUNT(*) FROM t1",
) -> Template:
    return Template(
        id=tid,
        intent_signature=ConcreteIntent(
            intent_id="i",
            tables=["t1"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="k",
        tables_used=["t1"],
        sql_param=sql,
        sql_fp=f"fp-{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=True, num_where=0),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{}],
            questions=[question],
            natural_language=["nl"],
            accept_counts=[0 if pending else 1],
        ),
        stats=TemplateStats(accept=0 if pending else 1, reject=0),
        approval_state=ApprovalState.PENDING if pending else ApprovalState.APPROVED,
        trust_level=0 if pending else 1,
    )


def _owner(*, templates: dict[str, Template] | None = None, store: dict | None = None) -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.tables = {}
    owner._store = store if store is not None else {}
    owner._templates = templates if templates is not None else {}
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
    return owner


def _intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["t1"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _tail(
    owner: MagicMock,
    *,
    q_norm: str,
    matched_template: Template | None,
    intent: RuntimeIntent | None = None,
) -> InteractiveTailSnapshot:
    ri = intent or _intent()
    return InteractiveTailSnapshot(
        q_norm=q_norm,
        intent=ri,
        schema=owner._schema_graph,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema_terms=set(),
        dialect=owner._dialect,
        semantic_warnings=(),
        has_union_match=False,
        cols_changed=False,
        matched_template=matched_template,
        union_select_cols=None,
        structural_match_templates=(),
        ikey="k",
        intent_sim=0.0,
    )


@pytest.mark.fast
def test_terminal_success_exposes_template_id() -> None:
    tmpl = _tmpl()
    owner = _owner(templates={tmpl.id: tmpl})
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many"
    sess._last_turn_outcome = {
        "outcome": "success",
        "sql": tmpl.sql_param,
        "rows": [(3,)],
        "columns": ["count"],
        "matched_template": tmpl,
        "intent": _intent(),
        "generation_path": GenerationPath.EXACT_QUESTION_REUSE,
    }
    step = sess._completed_step()
    assert step.kind == SESSION_KIND_RESULT
    assert step.template_id == "T0777"


@pytest.mark.fast
def test_meta_step_template_id_none() -> None:
    step = SessionStep(
        done=True,
        prompt=None,
        kind="meta",
        sql=None,
        answer="schema catalog",
        template_id=None,
    )
    assert step.template_id is None
    assert step.sql is None


@pytest.mark.fast
def test_sql_confirm_stamps_pending_matched_template() -> None:
    pending = _tmpl(tid="T0888", pending=True)
    owner = _owner(templates={pending.id: pending})
    intent = _intent()
    gen_out = SqlGenerationOutcome(
        sql=pending.sql_param,
        success=True,
        generation_path=GenerationPath.FRESH,
        matched_template=pending,
    )
    ctx = SqlFeedbackSuspendContext(
        tail=_tail(owner, q_norm="how many", matched_template=None, intent=intent),
        execution_intent=intent,
        sql=pending.sql_param,
        preview_rows=((3,),),
        sql_parameters=(),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=True,
    )
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many"
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx))
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert step.sql == pending.sql_param
    assert step.template_id == "T0888"


@pytest.mark.fast
def test_sql_confirm_finds_pending_when_matched_unset() -> None:
    pending = _tmpl(tid="T0999", pending=True)
    owner = _owner(templates={pending.id: pending})
    intent = _intent()
    gen_out = SqlGenerationOutcome(
        sql=pending.sql_param,
        success=True,
        generation_path=GenerationPath.FRESH,
        matched_template=None,
    )
    ctx = SqlFeedbackSuspendContext(
        tail=_tail(owner, q_norm="how many", matched_template=None, intent=intent),
        execution_intent=intent,
        sql=pending.sql_param,
        preview_rows=((1,),),
        sql_parameters=(),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=True,
    )
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many"
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx))
    assert step.template_id == "T0999"


@pytest.mark.fast
def test_terminal_success_after_accept_uses_matched_template() -> None:
    tmpl = _tmpl(tid="T0555")
    owner = _owner(templates={tmpl.id: tmpl})
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many"
    sess.note_turn_outcome(
        outcome="success",
        sql=tmpl.sql_param,
        rows=[(2,)],
        columns=("count",),
        intent=_intent(),
        matched_template=tmpl,
        generation_path=GenerationPath.FRESH,
    )
    step = sess._completed_step()
    assert step.template_id == "T0555"


@pytest.mark.fast
def test_complete_direct_sql_reuse_notes_matched_template() -> None:
    tmpl = _tmpl(tid="T0666")
    owner = _owner(templates={tmpl.id: tmpl})
    intent = _intent()
    ctx = DirectReuseSuspendContext(
        q_norm="how many",
        ref_tmpl=tmpl,
        dialect=owner._dialect,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema=owner._schema_graph,
        intent=intent,
        sql=tmpl.sql_param,
        rows=((4,),),
        display_sql=tmpl.sql_param,
        headers=("count",),
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    sess = PipelineSession(owner, mode="writer")
    with patch("aetherdialect._pipeline_generate.handle_user_feedback", return_value=None):
        complete_direct_sql_reuse_user_choice(ctx, "y", choice_port=sess, persist_template_learning=False)
    snap = sess._last_turn_outcome or {}
    assert snap.get("outcome") == "success"
    assert snap.get("matched_template") is tmpl
    step = sess._completed_step()
    assert step.template_id == "T0666"


@pytest.mark.fast
def test_federation_sql_confirm_stamps_composite_pending() -> None:
    pending = _tmpl(tid="T_FED1", pending=True, sql="-- federated display")
    owner = _owner(templates={pending.id: pending})
    intent = _intent()
    gen_out = SqlGenerationOutcome(
        sql=pending.sql_param,
        success=True,
        generation_path=GenerationPath.FEDERATION_PLAN,
        matched_template=pending,
        federation_plan_id="ik_fed",
        federation_dir="/tmp/fed",
    )
    ctx = SqlFeedbackSuspendContext(
        tail=_tail(owner, q_norm="how many across sources", matched_template=None, intent=intent),
        execution_intent=intent,
        sql=pending.sql_param,
        preview_rows=((9,),),
        sql_parameters=(),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=True,
    )
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many across sources"
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx))
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert step.template_id == "T_FED1"


@pytest.mark.fast
def test_direct_reuse_suspend_stamps_ref_tmpl() -> None:
    tmpl = _tmpl(tid="T_REUSE")
    owner = _owner(templates={tmpl.id: tmpl})
    intent = _intent()
    ctx = DirectReuseSuspendContext(
        q_norm="how many",
        ref_tmpl=tmpl,
        dialect=owner._dialect,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema=owner._schema_graph,
        intent=intent,
        sql=tmpl.sql_param,
        rows=((1,),),
        display_sql=tmpl.sql_param,
        headers=("count",),
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    sess = PipelineSession(owner, mode="writer")
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "Reuse?", ctx))
    assert step.template_id == "T_REUSE"
