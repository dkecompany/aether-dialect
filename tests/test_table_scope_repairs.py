"""Engine table-scope adjustments are recorded with reasons and surfaced as warnings."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import CteIntent, LogicalIntent
from aetherdialect._contracts_core import NormalizedExpr, RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_process import _align_runtime_tables_to_planner
from aetherdialect._intent_repair import (
    append_table_scope_repairs,
    reconcile_tables,
    table_scope_repair_warning_messages,
    validate_table_scope_repairs,
)
from aetherdialect._pipeline import _resolve_joins_fresh


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _bridge_schema() -> SchemaGraph:
    bridge_edge = {
        "src_table": "a",
        "src_cols": ["id"],
        "dst_table": "bridge",
        "dst_cols": ["aid"],
    }
    second = {
        "src_table": "bridge",
        "src_cols": ["cid"],
        "dst_table": "c",
        "dst_cols": ["id"],
    }
    path = [bridge_edge, second]
    return SchemaGraph(
        tables={
            "a": TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "c": TableMetadata(name="c", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "bridge": TableMetadata(
                name="bridge",
                columns={"aid": _col("aid"), "cid": _col("cid")},
                primary_key=["aid"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={"a": {"c": [path]}},
        effective_structural_hash="h",
    )


def test_align_runtime_tables_to_planner_records_planner_bridge_additions() -> None:
    logical = LogicalIntent(
        tables=("film", "inventory", "rental", "payment"),
        select="",
    )
    runtime = RuntimeIntent(
        tables=["film", "payment"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    out = _align_runtime_tables_to_planner(runtime, logical)
    assert out.tables == ["film", "inventory", "rental", "payment"]
    msgs = table_scope_repair_warning_messages(out)
    assert any("inventory" in m and "planner" in m.lower() for m in msgs)
    assert any("rental" in m for m in msgs)


def test_align_runtime_tables_to_planner_records_cte_bridge_additions() -> None:
    logical = LogicalIntent(
        tables=("film",),
        select="",
        cte_steps=(CteIntent(name="step_a", tables=("a", "b", "bridge", "c")),),
    )
    runtime = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[RuntimeCteStep(cte_name="step_a", tables=["a", "c"])],
    )
    out = _align_runtime_tables_to_planner(runtime, logical)
    msgs = table_scope_repair_warning_messages(out)
    assert any("bridge" in m and "CTE 'step_a'" in m for m in msgs)


def test_reconcile_tables_records_expression_add_and_unreferenced_remove() -> None:
    intent = RuntimeIntent(
        tables=["orders", "unused_tbl"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("orders.id")),
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    out = reconcile_tables(intent)
    assert "customers" in out.tables
    assert "unused_tbl" not in out.tables
    msgs = table_scope_repair_warning_messages(out)
    assert any("customers" in m and "expression" in m.lower() for m in msgs)
    assert any("unused_tbl" in m and "not referenced" in m.lower() for m in msgs)


def test_validate_table_scope_repairs_emits_warnings() -> None:
    intent = append_table_scope_repairs(
        RuntimeIntent(
            tables=["a"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        scope_label="main query",
        added=["bridge"],
        removed=[],
        add_reason="join_bridge",
    )
    issues = validate_table_scope_repairs(intent)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "bridge" in issues[0].message
    assert "join path" in issues[0].message.lower()


def test_resolve_joins_records_join_bridge_scope_repair() -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    join_candidates = {
        "candidates": [
            {
                "candidate_id": "J01",
                "join_path_signature": sig,
                "edge_kinds": ["catalog_fk", "catalog_fk"],
                "edge_count": 2,
            }
        ]
    }
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    _resolve_joins_fresh(
        "SELECT a.id FROM a, c",
        intent,
        {},
        None,
        "list ids",
        join_candidates,
        schema=schema,
        join_preset_scope={"main": "J01"},
    )
    msgs = table_scope_repair_warning_messages(intent)
    assert any("bridge" in m and "join path" in m.lower() for m in msgs)
