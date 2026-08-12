"""Fuzzy template reuse must reject LLM-extracted params with wrong Python types."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FeedbackCounts,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._pipeline_execute import handle_direct_sql_reuse


def _orders_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "order_id": ColumnMetadata(name="order_id", data_type="integer", sensitivity="none"),
                    "status": ColumnMetadata(name="status", data_type="string", sensitivity="none"),
                },
                primary_key=["order_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )


def _template_with_integer_filter() -> Template:
    where = PredicateGroup.from_list(
        [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.order_id"),
                op="=",
                param_key="p1",
                value_type="integer",
            )
        ]
    )
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )
    return Template(
        id="T1",
        effective_structural_hash="h",
        intent_signature=concrete,
        intent_key="ik",
        tables_used=["orders"],
        sql_param="SELECT order_id FROM orders WHERE order_id = :p1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="c",
        value_history=ValueHistory(
            param_values=[{"p1": 1}],
            questions=["stored_q"],
            natural_language=["nl"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
        feedback_by_question={"new_fuzzy_q": FeedbackCounts(accepts=1, rejects=0)},
    )


@pytest.mark.fast
@patch("aetherdialect._llm_provider.LLMProvider.chat", return_value='{"aliases":{}}')
@patch(
    "aetherdialect._pipeline_generate.extract_fuzzy_reuse_params",
    return_value={"p1": "7"},
)
def test_wrong_type_aborts_reuse(mock_extract, _mock_llm) -> None:
    """String extraction for an integer exemplar slot aborts fuzzy reuse."""
    tmpl = _template_with_integer_filter()
    dialect = MagicMock()
    dialect.finalize_render.return_value = "EXEC"
    dialect.explain_validation_sql = lambda sql, _pv: sql
    dialect.execute.return_value = [("row",)]
    store: dict = {"templates": {"T1": tmpl}}
    with (
        patch("aetherdialect._pipeline_generate.validate_sql", return_value=(True, None, None, [])),
        patch("aetherdialect._templates_ops.TemplateOps.save_template_store"),
        patch("aetherdialect._templates_ops.TemplateOps.templates_to_store", side_effect=lambda s, t: s),
        patch("aetherdialect._templates_ops.TemplateOps.delete_rejected_templates_matching_question"),
        patch("aetherdialect._pipeline_execute.save_result_csv_for_store"),
        patch("aetherdialect._pipeline_execute.print_query_result"),
        patch("aetherdialect._templates_ops.TemplateOps.promote_trust"),
    ):
        result = handle_direct_sql_reuse(
            "new_fuzzy_q",
            tmpl,
            dialect,
            store,
            {"T1": tmpl},
            {},
            _orders_schema(),
            reuse_history_index=0,
        )
    assert result is None
    mock_extract.assert_called_once()
    dialect.execute.assert_not_called()
