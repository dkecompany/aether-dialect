"""Engine SessionStep.sql stays dialect-specific parameterized SQL."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import PIPELINE_SUSPEND_ID_DIRECT_REUSE, PIPELINE_SUSPEND_ID_SQL
from aetherdialect._contracts_base import PipelineSuspended
from aetherdialect._contracts_core import (
    ConcreteIntent,
    DirectReuseSuspendContext,
    GenerationPath,
    InteractiveTailSnapshot,
    RuntimeIntent,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._main_execution import PipelineSession


def _tmpl() -> Template:
    return Template(
        id="T0042",
        intent_signature=ConcreteIntent(
            intent_id="i",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="k",
        tables_used=["t1"],
        sql_param="SELECT id FROM t1 WHERE name = :p1",
        sql_fp="fp42",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "Ada"}],
            questions=["q"],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
    )


def _owner(tmpl: Template) -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.tables = {}
    owner._store = {}
    owner._templates = {tmpl.id: tmpl}
    owner._rejected = {}
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._dialect = MagicMock()
    owner._dialect.name = "postgresql"
    owner._dialect.config = owner._runtime_config
    owner._audit_emit = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._artifacts_dir = None
    owner._business_knowledge = None
    owner._ask_phase_callback = None
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._schema_terms = set()
    return owner


@pytest.mark.fast
def test_engine_step_sql_is_dialect_parameterized() -> None:
    tmpl = _tmpl()
    owner = _owner(tmpl)
    intent = RuntimeIntent(
        tables=["t1"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        param_values={"p1": "Ada"},
        sql_param=tmpl.sql_param,
    )
    ctx = DirectReuseSuspendContext(
        q_norm="q",
        ref_tmpl=tmpl,
        dialect=owner._dialect,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema=owner._schema_graph,
        intent=intent,
        sql="SELECT id FROM t1 WHERE name = 'Ada'",
        rows=((1,),),
        display_sql="SELECT id FROM t1 WHERE name = 'Ada'",
        headers=("id",),
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    sess = PipelineSession(owner, mode="reader")
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "Reuse?", ctx))
    assert isinstance(step.sql, str)
    assert step.sql == tmpl.sql_param
    assert "'Ada'" not in step.sql


@pytest.mark.fast
def test_pparams_appear_as_bind_tokens() -> None:
    tmpl = _tmpl()
    owner = _owner(tmpl)
    intent = RuntimeIntent(
        tables=["t1"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        param_values={"p1": "Ada"},
        sql_param=tmpl.sql_param,
    )
    tail = InteractiveTailSnapshot(
        q_norm="q",
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
        matched_template=tmpl,
        union_select_cols=None,
        structural_match_templates=(),
        ikey="k",
        intent_sim=0.0,
    )
    gen_out = SqlGenerationOutcome(
        sql=tmpl.sql_param,
        success=True,
        generation_path=GenerationPath.EXACT_QUESTION_REUSE,
        matched_template=tmpl,
    )
    ctx = SqlFeedbackSuspendContext(
        tail=tail,
        execution_intent=intent,
        sql=tmpl.sql_param,
        preview_rows=((1,),),
        sql_parameters=(("p1", "Ada"),),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
    )
    sess = PipelineSession(owner, mode="reader")
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "OK?", ctx))
    assert isinstance(step.sql, str)
    assert ":p1" in step.sql
    assert all(b.handle == "p1" for b in step.parameters)
