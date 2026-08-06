"""Deny-only prompt payload and unqualified denied-column scope escapes."""

from __future__ import annotations

import json

import pytest

from aetherdialect._contracts_base import EngineContext, NormalizedExpr
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import assert_consumer_intent_in_scope, assert_intent_in_scope


def _table(name: str, *, extra_cols: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols: dict[str, ColumnMetadata] = {
        "id": ColumnMetadata(name="id", data_type="integer"),
    }
    if extra_cols:
        cols.update(extra_cols)
    return TableMetadata(
        name=name,
        columns=cols,
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )


def _two_table_graph() -> SchemaGraph:
    return SchemaGraph(
        join_paths_multi={},
        tables={
            "allowed": _table("allowed"),
            "denied": _table("denied"),
        },
        effective_structural_hash="eff",
    )


def _secret_graph() -> SchemaGraph:
    return SchemaGraph(
        join_paths_multi={},
        tables={
            "t": _table(
                "t",
                extra_cols={
                    "secret": ColumnMetadata(
                        name="secret",
                        data_type="text",
                        role="free_text",
                        usable_override=True,
                    ),
                },
            ),
        },
        effective_structural_hash="eff",
    )


@pytest.mark.fast
def test_resolve_payload_where_deny_only_objects_excludes_denied_tables() -> None:
    graph = _two_table_graph()
    table_filter, column_filter = graph._resolve_payload_where(deny_objects=frozenset({"denied"}))
    assert table_filter is not None
    assert "allowed" in table_filter
    assert "denied" not in table_filter
    assert column_filter is None


@pytest.mark.fast
def test_resolve_payload_where_deny_only_columns_excludes_denied_columns() -> None:
    graph = _secret_graph()
    table_filter, column_filter = graph._resolve_payload_where(deny_columns=frozenset({"t.secret"}))
    assert table_filter is not None
    assert "t" in table_filter
    assert column_filter is not None
    assert "t.id" in column_filter
    assert "t.secret" not in column_filter


@pytest.mark.fast
def test_schema_payload_interpret_deny_only_objects_omits_denied_table() -> None:
    graph = _two_table_graph()
    payload = json.loads(graph.schema_payload_interpret(deny_objects=frozenset({"denied"})))
    assert "allowed" in payload
    assert "denied" not in payload


@pytest.mark.fast
def test_schema_payload_ground_deny_only_columns_omits_denied_column() -> None:
    graph = _secret_graph()
    payload = json.loads(graph.schema_payload_ground(deny_columns=frozenset({"t.secret"})))
    assert "t" in payload
    assert "secret" not in payload["t"]["columns"]
    assert "id" in payload["t"]["columns"]


@pytest.mark.fast
def test_consumer_intent_gate_deny_only_objects_blocks_denied_table() -> None:
    graph = _two_table_graph()
    ctx = EngineContext(deny_objects=frozenset({"denied"}))
    intent = RuntimeIntent(
        tables=["denied"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("denied.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    assert assert_consumer_intent_in_scope(intent, ctx, graph, None) is False


@pytest.mark.fast
def test_consumer_intent_gate_rejects_unqualified_denied_column_in_cte() -> None:
    graph = _secret_graph()
    ctx = EngineContext(deny_columns=frozenset({"t.secret"}))
    inner = RuntimeCteStep(
        cte_name="inner_x",
        tables=["t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret"))],
        output_columns=["secret"],
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[inner],
    )
    assert assert_consumer_intent_in_scope(intent, ctx, graph, None) is False


@pytest.mark.fast
def test_intent_in_scope_rejects_unqualified_denied_column_in_cte() -> None:
    graph = _secret_graph()
    inner = RuntimeCteStep(
        cte_name="inner_x",
        tables=["t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret"))],
        output_columns=["secret"],
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[inner],
    )
    allowed_columns = frozenset({"t.id"})
    assert assert_intent_in_scope(intent, frozenset({"t"}), allowed_columns, graph) is False
