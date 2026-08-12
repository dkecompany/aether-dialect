"""Template reuse matching is role-agnostic; execute applies the scope gate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants_runtime import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import EngineContext, NormalizedExpr
from aetherdialect._contracts_core import AccessError, ConcreteIntent, RuntimeIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._pipeline_execute import _execute_intent_sql_rows
from aetherdialect._pipeline_generate import match_question_level_template_reuse
from aetherdialect._utils import normalize_question
from aetherdialect._utils_intent import intent_key


def _schema() -> SchemaGraph:
    tbl = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )
    return SchemaGraph(tables={"orders": tbl}, join_paths_multi={}, effective_structural_hash="sh1")


def _runtime_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )


def _template(question: str) -> Template:
    concrete = ConcreteIntent(
        intent_id="x",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return Template(
        id="T_OWNER_1",
        effective_structural_hash="sh1",
        schema_graph_id="sg_test000000000001__abcd1234",
        intent_signature=concrete,
        intent_key=intent_key(_runtime_intent()),
        tables_used=["orders"],
        sql_param="SELECT orders.id FROM orders",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=[question], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )


@pytest.mark.fast
def test_same_question_same_template_id_across_roles() -> None:
    schema = _schema()
    q = normalize_question("how many orders")
    tmpl = _template(q)
    templates = {tmpl.id: tmpl}
    owner_hit = match_question_level_template_reuse(q, templates, schema=schema)
    consumer_hit = match_question_level_template_reuse(q, templates, schema=schema)
    assert owner_hit.best_template is not None
    assert consumer_hit.best_template is not None
    assert owner_hit.best_template.id == consumer_hit.best_template.id == tmpl.id


@pytest.mark.fast
def test_owner_template_reused_by_consumer_then_scope_denies_execute() -> None:
    schema = _schema()
    q = normalize_question("list order ids")
    tmpl = _template(q)
    templates = {tmpl.id: tmpl}
    hit = match_question_level_template_reuse(q, templates, schema=schema)
    assert hit.best_template is not None
    assert hit.best_template.id == tmpl.id

    dialect = MagicMock()
    dialect.finalize_render = MagicMock(return_value="SELECT orders.id FROM orders")
    dialect.sqlglot_dialect = "duckdb"
    dialect.name = "duckdb"
    dialect.execute = MagicMock(return_value=[(1,)])
    intent = _runtime_intent()
    intent.sql_param = tmpl.sql_param
    with pytest.raises(AccessError) as raised:
        _execute_intent_sql_rows(
            intent,
            schema,
            dialect,
            None,
            gate_kwargs={
                "schema_role": "consumer",
                "schema_context": EngineContext(deny_objects=frozenset({"orders"})),
                "visible_objects": None,
            },
        )
    assert raised.value.reason == "scope"
    assert str(raised.value) == PERMISSION_DENIED_USER_MESSAGE
    dialect.execute.assert_not_called()
