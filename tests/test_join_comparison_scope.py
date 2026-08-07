"""Cross-table comparisons must not silently force long or semantic join detours."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR
from aetherdialect._contracts_base import (
    ComparisonJoinScopeExceededError,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import NormalizedExpr, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._intent_repair import reconcile_tables
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._validation_schema import (
    validate_comparison_join_scope,
    validate_comparison_join_scope_or_raise,
)


def _col(name: str, **overrides) -> ColumnMetadata:
    defaults = dict(name=name, data_type="integer", sensitivity="none")
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _fk(src: str, sc: str, dst: str, dc: str) -> FKEdge:
    return FKEdge(src_table=src, src_cols=[sc], dst_table=dst, dst_cols=[dc])


def _linear_chain_schema() -> SchemaGraph:
    tables = {
        "a": TableMetadata(
            name="a",
            columns={"id": _col("id", is_primary_key=True), "x": _col("x")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "b": TableMetadata(
            name="b",
            columns={"id": _col("id", is_primary_key=True), "aid": _col("aid")},
            primary_key=["id"],
            foreign_keys=[_fk("b", "aid", "a", "id")],
        ),
        "bridge": TableMetadata(
            name="bridge",
            columns={"id": _col("id", is_primary_key=True), "bid": _col("bid")},
            primary_key=["id"],
            foreign_keys=[_fk("bridge", "bid", "b", "id")],
        ),
        "c": TableMetadata(
            name="c",
            columns={"id": _col("id", is_primary_key=True), "bid": _col("bid"), "y": _col("y")},
            primary_key=["id"],
            foreign_keys=[_fk("c", "bid", "bridge", "id")],
        ),
        "d": TableMetadata(
            name="d",
            columns={"id": _col("id", is_primary_key=True), "cid": _col("cid"), "y": _col("y")},
            primary_key=["id"],
            foreign_keys=[_fk("d", "cid", "c", "id")],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="comparison-scope",
    )


def _comparison_intent(*, select_from_d: bool = False) -> RuntimeIntent:
    where = PredicateGroup.from_list(
        [
            WhereParam(
                left_expr=NormalizedExpr.from_column("a.x"),
                op=">",
                right_expr=NormalizedExpr.from_column("d.y"),
            )
        ]
    )
    select_cols = [SelectCol(expr=NormalizedExpr.from_column("d.y"))] if select_from_d else []
    return RuntimeIntent(
        tables=["a", "d"],
        grain="row_level",
        select_cols=select_cols or [SelectCol(expr=NormalizedExpr.from_column("a.x"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )


def _long_path_signature() -> tuple[list[str], list[str]]:
    signature = [
        "a.id->b.aid",
        "b.id->bridge.bid",
        "bridge.id->c.bid",
        "c.id->d.cid",
    ]
    kinds = ["catalog_fk"] * len(signature)
    return signature, kinds


def _bridge_chain_schema() -> SchemaGraph:
    tables = {
        "a": TableMetadata(
            name="a",
            columns={"id": _col("id", is_primary_key=True), "x": _col("x")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "bridge": TableMetadata(
            name="bridge",
            columns={"id": _col("id", is_primary_key=True), "aid": _col("aid")},
            primary_key=["id"],
            foreign_keys=[_fk("bridge", "aid", "a", "id")],
        ),
        "c": TableMetadata(
            name="c",
            columns={"id": _col("id", is_primary_key=True), "bid": _col("bid"), "y": _col("y")},
            primary_key=["id"],
            foreign_keys=[_fk("c", "bid", "bridge", "id")],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="comparison-bridge",
    )


def _short_path_signature() -> tuple[list[str], list[str]]:
    signature = ["a.id->bridge.aid", "bridge.id->c.bid"]
    return signature, ["catalog_fk", "catalog_fk"]


@pytest.mark.fast
def test_reconcile_tables_marks_right_expr_table_as_comparison_only() -> None:
    intent = reconcile_tables(_comparison_intent())
    assert intent.comparison_only_tables == ["d"]


@pytest.mark.fast
def test_comparison_only_table_beyond_hop_ceiling_refuses() -> None:
    signature, kinds = _long_path_signature()
    intent = reconcile_tables(_comparison_intent())
    issues = validate_comparison_join_scope(
        scope_label="main query",
        scope_tables=["a", "d"],
        comparison_only=list(intent.comparison_only_tables),
        signature=signature,
        edge_kinds=kinds,
        from_anchor="a",
        where_params=PredicateGroup.where_leaves(intent.where),
        having_params=[],
    )
    assert any(i.severity == "error" for i in issues)
    assert any("4 join hops" in i.message for i in issues)
    assert any("bridge" in i.message and "c" in i.message for i in issues)
    with pytest.raises(ComparisonJoinScopeExceededError):
        validate_comparison_join_scope_or_raise(
            scope_label="main query",
            scope_tables=["a", "d"],
            comparison_only=["d"],
            signature=signature,
            edge_kinds=kinds,
            from_anchor="a",
            where_params=PredicateGroup.where_leaves(intent.where),
            having_params=[],
        )


@pytest.mark.fast
def test_comparison_only_within_ceiling_emits_detour_diagnostic() -> None:
    signature, kinds = _short_path_signature()
    intent = reconcile_tables(
        RuntimeIntent(
            tables=["a", "c"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.x"))],
            group_by_cols=[],
            order_by_cols=[],
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("a.x"),
                        op=">",
                        right_expr=NormalizedExpr.from_column("c.y"),
                    )
                ]
            ),
        )
    )
    token = set_diagnostic_collector([])
    try:
        validate_comparison_join_scope_or_raise(
            scope_label="main query",
            scope_tables=["a", "c"],
            comparison_only=list(intent.comparison_only_tables),
            signature=signature,
            edge_kinds=kinds,
            from_anchor="a",
            where_params=PredicateGroup.where_leaves(intent.where),
            having_params=[],
        )
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert not any(
        i.severity == "error"
        for i in validate_comparison_join_scope(
            scope_label="main query",
            scope_tables=["a", "c"],
            comparison_only=list(intent.comparison_only_tables),
            signature=signature,
            edge_kinds=kinds,
            from_anchor="a",
            where_params=PredicateGroup.where_leaves(intent.where),
            having_params=[],
        )
    )
    assert any(d.code == DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR for d in diags)


@pytest.mark.fast
def test_projected_table_at_same_hop_count_is_unaffected() -> None:
    signature, kinds = _long_path_signature()
    intent = reconcile_tables(_comparison_intent(select_from_d=True))
    assert "d" not in intent.comparison_only_tables
    issues = validate_comparison_join_scope(
        scope_label="main query",
        scope_tables=["a", "d"],
        comparison_only=list(intent.comparison_only_tables),
        signature=signature,
        edge_kinds=kinds,
        from_anchor="a",
        where_params=PredicateGroup.where_leaves(intent.where),
        having_params=[],
    )
    assert issues == []


@pytest.mark.fast
def test_extended_semantic_path_to_comparison_only_table_refuses() -> None:
    signature = ["a.x->d.y"]
    kinds = ["semantic_profile"]
    intent = reconcile_tables(_comparison_intent())
    issues = validate_comparison_join_scope(
        scope_label="main query",
        scope_tables=["a", "d"],
        comparison_only=list(intent.comparison_only_tables),
        signature=signature,
        edge_kinds=kinds,
        from_anchor="a",
        where_params=PredicateGroup.where_leaves(intent.where),
        having_params=[],
    )
    assert any("profile-inferred" in i.message for i in issues)
