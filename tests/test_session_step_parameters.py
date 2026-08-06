"""Public SessionStep parameter bindings expose p-params only with column_expr."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import PIPELINE_SUSPEND_ID_SQL, SESSION_KIND_AWAITING_SQL_CONFIRM
from aetherdialect._contracts_base import (
    NormalizedExpr,
    ParameterBinding,
    PipelineSuspended,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
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
from aetherdialect._templates import TemplateOps


def _template_with_p_and_s() -> Template:
    intent_sig = ConcreteIntent(
        intent_id="i1",
        tables=["payment"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op="=",
                    value_type="date",
                    param_key="p1",
                )
            ]
        ),
        limit_param_key="s1",
    )
    return Template(
        id="T0009",
        intent_signature=intent_sig,
        intent_key="k9",
        tables_used=["payment"],
        sql_param="SELECT * FROM payment WHERE payment_date = :p1 LIMIT :s1",
        sql_fp="fp9",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "2024-01-01", "s1": 10}],
            questions=["payments on date"],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
        param_display_names={"p1": "payment date", "s1": "row limit"},
    )


@pytest.mark.fast
def test_display_name_and_column_expr_set() -> None:
    tmpl = _template_with_p_and_s()
    schema = MagicMock()
    schema.tables = {}
    bindings = TemplateOps.build_parameter_bindings(
        tmpl,
        history_index=0,
        schema=schema,
        persist_display_names=False,
    )
    assert len(bindings) == 1
    assert bindings[0].handle == "p1"
    assert bindings[0].display_name == "payment date"
    assert bindings[0].column_expr == "payment.payment_date"
    assert bindings[0].current_value == "2024-01-01"


@pytest.mark.fast
def test_sparams_omitted() -> None:
    tmpl = _template_with_p_and_s()
    schema = MagicMock()
    schema.tables = {}
    bindings = TemplateOps.build_parameter_bindings(
        tmpl,
        history_index=0,
        schema=schema,
        persist_display_names=False,
    )
    handles = {b.handle for b in bindings}
    assert "p1" in handles
    assert "s1" not in handles
    assert all(b.handle.startswith("p") and b.handle[1:].isdigit() for b in bindings)


@pytest.mark.fast
def test_sql_confirm_has_pparams_only() -> None:
    tmpl = _template_with_p_and_s()
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

    intent = RuntimeIntent(
        tables=["payment"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        param_values={"p1": "2024-01-01", "s1": 10},
        sql_param=tmpl.sql_param,
    )
    tail = InteractiveTailSnapshot(
        q_norm="payments on date",
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
        ikey="k9",
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
        sql_parameters=(("p1", "2024-01-01"), ("s1", 10)),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
    )
    sess = PipelineSession(owner, mode="reader", space_name="master")
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx))
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert step.sql is not None
    assert step.parameters
    handles = [b.handle for b in step.parameters]
    assert handles == ["p1"]
    assert isinstance(step.parameters[0], ParameterBinding)
    assert step.parameters[0].column_expr == "payment.payment_date"
