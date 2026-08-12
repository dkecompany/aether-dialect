"""Template approval_state gates silent reuse and execute_template."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import ApprovalState, ConfigError, NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, RuntimeIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._pipeline_execute import execute_stored_template_by_ref
from aetherdialect._pipeline_generate import match_question_level_template_reuse
from aetherdialect._utils import normalize_question
from aetherdialect._utils_intent import intent_key


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                row_count=1,
            )
        },
        join_paths_multi={},
        effective_structural_hash="sh1",
        schema_graph_id="sg1",
    )


def _template(question: str, *, approval: ApprovalState) -> Template:
    concrete = ConcreteIntent(
        intent_id="x",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"id": "orders"},
    )
    runtime = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    return Template(
        id="T_APPR",
        effective_structural_hash="sh1",
        schema_graph_id="sg1",
        intent_signature=concrete,
        intent_key=intent_key(runtime),
        tables_used=["orders"],
        sql_param="SELECT orders.id FROM orders",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=[question], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
        approval_state=approval,
        footprint_tables=("orders",),
        footprint_columns=("orders.id",),
    )


@pytest.mark.fast
def test_pending_not_silently_reused() -> None:
    schema = _schema()
    q = normalize_question("list order ids")
    tmpl = _template(q, approval=ApprovalState.PENDING)
    hit = match_question_level_template_reuse(q, {tmpl.id: tmpl}, schema=schema)
    assert hit.best_template is None


@pytest.mark.fast
def test_approved_reused() -> None:
    schema = _schema()
    q = normalize_question("list order ids")
    tmpl = _template(q, approval=ApprovalState.APPROVED)
    hit = match_question_level_template_reuse(q, {tmpl.id: tmpl}, schema=schema)
    assert hit.best_template is not None
    assert hit.best_template.id == tmpl.id


@pytest.mark.fast
def test_execute_pending_raises() -> None:
    schema = _schema()
    q = normalize_question("list order ids")
    tmpl = _template(q, approval=ApprovalState.PENDING)
    dialect = MagicMock()
    with pytest.raises(ConfigError, match="pending approval"):
        execute_stored_template_by_ref(
            tmpl.id,
            {},
            question=q,
            dialect=dialect,
            store={},
            templates={tmpl.id: tmpl},
            rejected={},
            schema=schema,
        )
