"""Tests for forced template reuse and parameter bindings on session steps."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aetherdialect import SessionStep
from aetherdialect._contracts_base import (
    NormalizedExpr,
    ParameterBinding,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import ConcreteIntent, Template, ValueHistory
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._templates import (
    TemplateOps,
)


def _minimal_template(*, question: str = "count of item in category x") -> Template:
    intent_sig = ConcreteIntent(
        intent_id="i1",
        tables=["t1"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t1.category"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    return Template(
        id="T0001",
        intent_signature=intent_sig,
        intent_key="k1",
        tables_used=["t1"],
        sql_param="SELECT 1 FROM t1 WHERE category = :p1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "x"}],
            questions=[question],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
        param_display_names={"p1": "category"},
    )


def test_handles_referenced_in_sql_param_orders_p_before_s() -> None:
    assert TemplateOps.handles_referenced_in_sql_param("WHERE a = :p2 AND limit :s1 AND b = :p1") == ("p1", "p2", "s1")


def test_resolve_template_for_question_exact_match() -> None:
    tmpl = _minimal_template()
    resolved = TemplateOps.resolve_template_for_question("count of item in category x", {"T0001": tmpl})
    assert resolved is not None
    found, idx = resolved
    assert found.id == "T0001"
    assert idx == 0


def test_build_parameter_bindings_uses_cached_display_names() -> None:
    tmpl = _minimal_template()
    schema = MagicMock()
    schema.tables = {}
    bindings = TemplateOps.build_parameter_bindings(
        tmpl,
        history_index=0,
        schema=schema,
        persist_display_names=False,
    )
    assert len(bindings) == 1
    assert bindings[0] == ParameterBinding(
        handle="p1",
        current_value="x",
        display_name="category",
        column_expr="t1.category",
    )


def test_session_step_accepts_parameters_default_empty() -> None:
    step = SessionStep(done=True, prompt=None, kind="result")
    assert step.parameters == ()


def test_pipeline_session_exposes_reuse_saved_question() -> None:
    from aetherdialect._main_execution import PipelineSession

    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._store = MagicMock()
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._dialect = MagicMock()
    owner._audit_emit = MagicMock()
    owner._pipeline_writer_lock = __import__("threading").Lock()
    sess = PipelineSession(owner)
    assert hasattr(sess, "reuse_saved_question")
    with patch("aetherdialect._main_execution.force_reuse_saved_question") as forced:
        forced.return_value = MagicMock(success=True)
        with patch.object(sess, "_completed_step", return_value=SessionStep(done=True, prompt=None, kind="result")):
            with patch.object(sess, "_resources", return_value=(MagicMock(), {}, {}, {}, set())):
                step = sess.reuse_saved_question("old q", "new q", {"p1": "y"})
    assert step.done is True
    forced.assert_called_once()
