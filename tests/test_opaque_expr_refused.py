"""Opaque LLM expression text must not become raw_sql leaves."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_expr import parse_expr_string
from aetherdialect._validation_rules import validate_no_opaque_raw_sql


def _graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )


@pytest.mark.fast
def test_select_shaped_llm_expr_refused() -> None:
    with pytest.raises(ConfigError, match="opaque|expression"):
        parse_expr_string("SELECT id FROM orders")


@pytest.mark.fast
def test_unparseable_llm_expr_refused() -> None:
    with pytest.raises(ConfigError, match="opaque|expression|parse"):
        parse_expr_string("!!!not sql!!!")


@pytest.mark.fast
def test_composed_intent_with_raw_sql_refused() -> None:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(raw_sql="orders.id + 1"))],
    )
    issues = validate_no_opaque_raw_sql(intent, _graph())
    assert issues
    assert any("opaque" in (i.message or "").lower() or "raw_sql" in (i.issue_id or "").lower() for i in issues)
