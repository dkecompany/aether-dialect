"""Aggregate fan-out refusal when joins duplicate parent rows."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import AggregateJoinFanOutError, OrderByCol
from aetherdialect._contracts_core import NormalizedExpr, RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._pipeline import _resolve_joins_fresh
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import join_hints_multi
from aetherdialect._validation_execute import validate_aggregate_join_fan_out, validate_semantics


def _parent_child_schema() -> SchemaGraph:
    parent_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
    }
    child_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "parent_id": ColumnMetadata(
            name="parent_id",
            data_type="integer",
            sensitivity="none",
            is_foreign_key=True,
            fk_target=("parent", "id"),
        ),
        "qty": ColumnMetadata(name="qty", data_type="integer", sensitivity="none"),
    }
    fk = FKEdge(
        src_table="child",
        src_cols=["parent_id"],
        dst_table="parent",
        dst_cols=["id"],
    )
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns=parent_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns=child_cols,
            primary_key=["id"],
            foreign_keys=[fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="fan_out_test",
    )


def _bridge_multiply_schema() -> SchemaGraph:
    a_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
    }
    bridge_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "aid": ColumnMetadata(
            name="aid",
            data_type="integer",
            sensitivity="none",
            is_foreign_key=True,
            fk_target=("a", "id"),
        ),
        "weight": ColumnMetadata(name="weight", data_type="numeric", sensitivity="none"),
    }
    extra_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "bid": ColumnMetadata(
            name="bid",
            data_type="integer",
            sensitivity="none",
            is_foreign_key=True,
            fk_target=("bridge", "id"),
        ),
    }
    fk_a = FKEdge(src_table="bridge", src_cols=["aid"], dst_table="a", dst_cols=["id"])
    fk_b = FKEdge(src_table="extra", src_cols=["bid"], dst_table="bridge", dst_cols=["id"])
    tables = {
        "a": TableMetadata(name="a", columns=a_cols, primary_key=["id"], foreign_keys=[]),
        "bridge": TableMetadata(name="bridge", columns=bridge_cols, primary_key=["id"], foreign_keys=[fk_a]),
        "extra": TableMetadata(
            name="extra",
            columns=extra_cols,
            primary_key=["id"],
            foreign_keys=[fk_b],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="bridge_fan_out_test",
    )


def _bridge_multiply_signature(schema: SchemaGraph) -> list[str]:
    paths = (schema.join_paths_multi.get("a") or {}).get("extra") or []
    assert paths
    return [
        f"{edge['src_table']}.{','.join(edge['src_cols'])}->{edge['dst_table']}.{','.join(edge['dst_cols'])}"
        for edge in paths[0]
    ]


def _join_signature(schema: SchemaGraph) -> list[str]:
    path = (schema.join_paths_multi.get("parent") or {}).get("child") or []
    assert path
    edge = path[0][0]
    return [f"{edge['src_table']}.{','.join(edge['src_cols'])}->{edge['dst_table']}.{','.join(edge['dst_cols'])}"]


def _semantic_parent_child_schema() -> SchemaGraph:
    parent_cols = {
        "id": ColumnMetadata(
            name="id",
            data_type="integer",
            sensitivity="none",
            is_primary_key=True,
            semantic_join_neighbors=[("child", "parent_id")],
        ),
        "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
    }
    child_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "parent_id": ColumnMetadata(
            name="parent_id",
            data_type="integer",
            sensitivity="none",
            semantic_join_neighbors=[("parent", "id")],
        ),
        "qty": ColumnMetadata(name="qty", data_type="integer", sensitivity="none"),
    }
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns=parent_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns=child_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        effective_structural_hash="fan_out_semantic_test",
    )


def _profiled_non_unique_parent_schema() -> SchemaGraph:
    parent_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "ref": ColumnMetadata(
            name="ref",
            data_type="integer",
            sensitivity="none",
            is_unique=False,
            row_count=100,
            distinct_count=40,
        ),
        "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
    }
    child_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "parent_ref": ColumnMetadata(name="parent_ref", data_type="integer", sensitivity="none"),
        "qty": ColumnMetadata(name="qty", data_type="integer", sensitivity="none"),
    }
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns=parent_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns=child_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        effective_structural_hash="fan_out_profiled_non_unique_test",
    )


class TestAggregateFanOut:
    def test_parent_sum_across_one_to_many_refuses(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert any(i.severity == "error" and "parent.amount" in i.message for i in issues)

    def test_grouped_child_sum_succeeds(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "child.qty"))],
            group_by_cols=[NormalizedExpr.from_column("child.parent_id")],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert not any(i.severity == "error" for i in issues)

    def test_semi_join_form_succeeds(self) -> None:
        schema = _parent_child_schema()
        probe = RuntimeCteStep(
            cte_name="active_children",
            tables=["child"],
            emission="semi_join",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            output_columns=["parent_id"],
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[probe],
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert not any(i.severity == "error" for i in issues)

    def test_count_distinct_is_exempt(self) -> None:
        schema = _parent_child_schema()
        expr = NormalizedExpr.from_agg("count", "parent.id")
        expr.add_groups[0].distinct = True
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=expr)],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert not any(i.severity == "error" for i in issues)

    def test_many_to_one_unchanged(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["child", "parent"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "child.qty"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="child")
        assert not any(i.severity == "error" for i in issues)

    def test_non_multiplying_path_preferred(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["child", "parent"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "child.qty"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        hints = join_hints_multi(schema, ["child", "parent"], intent)
        substantive = [c for c in hints.get("candidates", []) if c.get("join_path_signature")]
        assert substantive
        assert substantive[0]["candidate_id"] != "J00"

    def test_select_distinct_does_not_exempt_sum(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            distinct_select_index=0,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert any(i.severity == "error" for i in issues)

    def test_resolve_joins_cte_fan_out_uses_cte_aggregates_not_root(self) -> None:
        """CTE fan-out guard must inspect CTE aggregates, not the root select list."""
        schema = _parent_child_schema()
        sig = _join_signature(schema)
        cte_bad = RuntimeCteStep(
            cte_name="rollup",
            tables=["parent", "child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte_bad],
        )
        join_candidates = {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]}
        cte_join_hints = {
            "rollup": {
                "candidates": [
                    {
                        "candidate_id": "J01",
                        "join_path_signature": sig,
                        "edge_kinds": ["catalog_fk"],
                    }
                ]
            }
        }
        with pytest.raises(AggregateJoinFanOutError, match="CTE 'rollup'"):
            _resolve_joins_fresh(
                "WITH rollup AS (SELECT 1) SELECT parent.id FROM parent",
                intent,
                {},
                cte_join_hints,
                "rollup parent amounts",
                join_candidates,
                schema=schema,
                join_preset_scope={"cte:rollup": "J01"},
            )

    def test_semantics_cte_parent_sum_fan_out_refuses(self) -> None:
        schema = _parent_child_schema()
        sig = _join_signature(schema)
        cte = RuntimeCteStep(
            cte_name="child_totals",
            tables=["parent", "child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            output_columns=["total_amount"],
            chosen_join_path_signature=sig,
            resolved_join_tables=["parent", "child"],
        )
        intent = RuntimeIntent(
            tables=["child_totals"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child_totals.total_amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        result = validate_semantics(intent, schema)
        assert any(
            i.severity == "error" and "parent.amount" in i.message and "CTE 'child_totals'" in i.message
            for i in result.issues
        )

    def test_resolved_join_tables_bridge_aggregate_fan_out_refuses(self) -> None:
        schema = _bridge_multiply_schema()
        sig = _bridge_multiply_signature(schema)
        intent = RuntimeIntent(
            tables=["a", "extra"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "bridge.weight"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=sig,
            resolved_join_tables=["a", "bridge", "extra"],
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="a")
        assert any(i.severity == "error" and "bridge.weight" in i.message for i in issues)

    def test_semantic_join_without_fk_refuses_parent_sum_fan_out(self) -> None:
        schema = _semantic_parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=["child.parent_id->parent.id"],
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert any(i.severity == "error" and "parent.amount" in i.message for i in issues)

    def test_order_by_aggregate_on_multiplied_anchor_refuses(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
            group_by_cols=[],
            order_by_cols=[
                OrderByCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"), direction="DESC"),
            ],
            where=None,
            chosen_join_path_signature=_join_signature(schema),
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert any(i.severity == "error" and "parent.amount" in i.message for i in issues)

    def test_profiled_non_unique_key_edge_treated_as_multiplying(self) -> None:
        schema = _profiled_non_unique_parent_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=["parent.ref->child.parent_ref"],
        )
        issues = validate_aggregate_join_fan_out(intent, schema, "main query", from_anchor="parent")
        assert any(i.severity == "error" and "parent.amount" in i.message for i in issues)

    def test_resolve_joins_raises_on_parent_sum_fan_out(self) -> None:
        schema = _parent_child_schema()
        sig = _join_signature(schema)
        join_candidates = {
            "candidates": [
                {
                    "candidate_id": "J01",
                    "join_path_signature": sig,
                    "edge_kinds": ["catalog_fk"],
                }
            ]
        }
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        with pytest.raises(AggregateJoinFanOutError):
            _resolve_joins_fresh(
                "SELECT parent.amount FROM parent, child",
                intent,
                {},
                None,
                "total parent amount",
                join_candidates,
                schema=schema,
                join_preset_scope={"main": "J01"},
            )
