"""Tests for join-kind compensation and preserve_tables."""

from __future__ import annotations

import pytest

from aetherdialect._constants import (
    ANTI_JOIN_PRESENCE_COLUMN_SUFFIX,
    DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH,
    JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR,
)
from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._contracts_core import NoJoinPathError, RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    _join_edges_from_signature,
    _join_kind_for_edge,
    anti_join_presence_column,
    build_deterministic_sql,
    collapse_probe_edge_candidate_variation,
    emit_join_orphan_rate_diagnostics,
    inject_join_into_deterministic_sql,
)
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._validation_shape import validate_preserve_tables
from aetherdialect._validation_sql import (
    validate_cte_emission_shapes,
    validate_probe_cte_modifiers,
)
from tests.join_test_helpers import catalog_edge_kinds_for_signatures


def _col(
    name: str,
    *,
    nullable: bool = False,
    fk_target: tuple[str, str] | None = None,
    distinct_count: int = 0,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_nullable=nullable,
        is_foreign_key=fk_target is not None,
        fk_target=fk_target,
        distinct_count=distinct_count,
    )


def _parent_child_schema() -> SchemaGraph:
    parent = TableMetadata(
        name="parent",
        columns={"id": _col("id", nullable=False)},
        primary_key=["id"],
        foreign_keys=[],
        row_count=100,
    )
    child = TableMetadata(
        name="child",
        columns={
            "id": _col("id", nullable=False),
            "parent_id": _col(
                "parent_id",
                nullable=False,
                fk_target=("parent", "id"),
                distinct_count=60,
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        row_count=200,
    )
    return SchemaGraph(
        tables={"parent": parent, "child": child},
        join_paths_multi={},
        effective_structural_hash="h",
    )


def _nullable_child_schema() -> SchemaGraph:
    parent = TableMetadata(
        name="parent",
        columns={"id": _col("id", nullable=False)},
        primary_key=["id"],
        foreign_keys=[],
    )
    child = TableMetadata(
        name="child",
        columns={
            "id": _col("id", nullable=False),
            "parent_id": _col("parent_id", nullable=True, fk_target=("parent", "id")),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"parent": parent, "child": child},
        join_paths_multi={},
        effective_structural_hash="h",
    )


class TestJoinKindPrecedence:
    def test_anti_join_forces_left(self) -> None:
        assert _join_kind_for_edge("probe", "parent", ["k"], None, right_emission="anti_join") == " LEFT"

    def test_semi_join_forces_inner(self) -> None:
        assert _join_kind_for_edge("probe", "parent", ["k"], None, right_emission="semi_join") == " INNER"

    def test_preserve_tables_forces_left(self) -> None:
        schema = _parent_child_schema()
        assert _join_kind_for_edge("child", "parent", ["parent_id"], schema, left_is_preserved=True) == " LEFT"

    def test_nullable_fk_left_without_preservation(self) -> None:
        schema = _nullable_child_schema()
        assert _join_kind_for_edge("child", "parent", ["parent_id"], schema) == " LEFT"

    def test_non_nullable_fk_inner_without_preservation(self) -> None:
        schema = _parent_child_schema()
        assert _join_kind_for_edge("child", "parent", ["parent_id"], schema) == " INNER"


class TestPreserveTablesJoinPath:
    def test_preservation_propagates_left_down_chain(self) -> None:
        schema = SchemaGraph(
            tables={
                "a": TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
                "b": TableMetadata(
                    name="b",
                    columns={"id": _col("id"), "a_id": _col("a_id", fk_target=("a", "id"))},
                    primary_key=["id"],
                    foreign_keys=[],
                ),
                "c": TableMetadata(
                    name="c",
                    columns={"id": _col("id"), "b_id": _col("b_id", fk_target=("b", "id"))},
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        sig = ["a.id->b.a_id", "b.id->c.b_id"]
        resolved = _join_edges_from_signature(sig, ["catalog_fk", "catalog_fk"], "a", schema, preserve_tables=["a"])
        assert resolved is not None
        join_edges, _where, _extra, _anti = resolved
        assert all(edge.kind == "LEFT" for edge in join_edges)

    def test_multihop_preserve_tables_zero_fills_child_counts_and_sums(self) -> None:
        parent = TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[])
        mid = TableMetadata(
            name="b",
            columns={"id": _col("id"), "a_id": _col("a_id", fk_target=("a", "id"))},
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table="b", src_cols=["a_id"], dst_table="a", dst_cols=["id"]),
            ],
        )
        child = TableMetadata(
            name="c",
            columns={
                "id": _col("id"),
                "b_id": _col("b_id", fk_target=("b", "id")),
                "amount": _col("amount"),
            },
            primary_key=["id"],
            foreign_keys=[
                FKEdge(src_table="c", src_cols=["b_id"], dst_table="b", dst_cols=["id"]),
            ],
        )
        tables = {"a": parent, "b": mid, "c": child}
        schema = SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            effective_structural_hash="h",
        )
        sig = ["a.id->b.a_id", "b.id->c.b_id"]
        resolved = _join_edges_from_signature(sig, ["catalog_fk", "catalog_fk"], "a", schema, preserve_tables=["a"])
        assert resolved is not None
        join_edges, _where, _extra, _anti = resolved
        assert all(edge.kind == "LEFT" for edge in join_edges)

        intent = RuntimeIntent(
            tables=["a", "b", "c"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("a.id")),
                SelectCol(expr=NormalizedExpr(agg_func="count", add_groups=[MulGroup(multiply=["c.id"])])),
                SelectCol(expr=NormalizedExpr(agg_func="sum", add_groups=[MulGroup(multiply=["c.amount"])])),
            ],
            group_by_cols=[NormalizedExpr.from_column("a.id")],
            order_by_cols=[],
            where=None,
            preserve_tables=["a"],
            chosen_join_path_signature=sig,
        )
        sql = build_deterministic_sql(intent, schema=schema, dialect=DialectRegistry.get("sqlite"))
        assert "COALESCE(COUNT" in sql and ", 0)" in sql
        assert "COALESCE(SUM" not in sql.upper()


class TestProbeCteConstraints:
    def test_probe_cte_cannot_be_anchor(self) -> None:
        with pytest.raises(NoJoinPathError, match="cannot be the join anchor"):
            _join_edges_from_signature(
                ["probe.k->parent.id"],
                ["catalog_fk"],
                "probe",
                None,
                {"probe": "anti_join"},
                probe_cte_names=frozenset({"probe"}),
            )

    def test_probe_cte_cannot_be_left_operand(self) -> None:
        schema = SchemaGraph(
            tables={
                "parent": TableMetadata(name="parent", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
                "child": TableMetadata(
                    name="child",
                    columns={"id": _col("id"), "parent_id": _col("parent_id", fk_target=("parent", "id"))},
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        with pytest.raises(NoJoinPathError, match="cannot appear on the left"):
            _join_edges_from_signature(
                ["parent.id->probe.k", "probe.k->child.parent_id"],
                ["catalog_fk", "catalog_fk"],
                "parent",
                schema,
                {"probe": "anti_join"},
                probe_cte_names=frozenset({"probe"}),
            )


class TestAntiJoinMarkerRendering:
    def test_anti_join_cte_projects_presence_marker(self) -> None:
        cte = RuntimeCteStep(
            cte_name="missing_orders",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))],
            output_columns=["customer_id"],
            emission="anti_join",
        )
        intent = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        sql = build_deterministic_sql(intent, schema=_parent_child_schema(), dialect=DialectRegistry.get("sqlite"))
        marker = anti_join_presence_column("missing_orders")
        assert f"1 AS {marker}" in sql
        assert marker.endswith(ANTI_JOIN_PRESENCE_COLUMN_SUFFIX)

    def test_injected_anti_join_adds_renderer_owned_is_null(self) -> None:
        det = "SELECT customers.id\nFROM customers"
        sig = [["customers.id->missing_orders.customer_id"]]
        out = inject_join_into_deterministic_sql(
            det,
            sig,
            edge_kinds_ordered=catalog_edge_kinds_for_signatures(sig),
            dialect=DialectRegistry.get("sqlite"),
            cte_emissions={"missing_orders": "anti_join"},
        )
        marker = anti_join_presence_column("missing_orders")
        assert marker in out
        assert "IS NULL" in out.upper()


class TestZeroFillAggregates:
    def test_count_zero_filled_when_preservation_active(self) -> None:
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("parent.id")),
                SelectCol(expr=NormalizedExpr(agg_func="count", add_groups=[MulGroup(multiply=["child.id"])])),
            ],
            group_by_cols=[NormalizedExpr.from_column("parent.id")],
            order_by_cols=[],
            where=None,
            preserve_tables=["parent"],
        )
        sql = build_deterministic_sql(intent, schema=_parent_child_schema(), dialect=DialectRegistry.get("sqlite"))
        assert "COALESCE(COUNT" in sql and ", 0)" in sql


class TestPreserveTablesValidation:
    def test_unknown_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["other"],
            _parent_child_schema(),
            "main query",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("closed" in i.message for i in issues)

    def test_unreachable_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["child"],
            _parent_child_schema(),
            "main query",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("not reachable" in i.message for i in issues)

    def test_noop_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["parent"],
            _parent_child_schema(),
            "main query",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("would have no effect" in i.message for i in issues)

    def test_cte_unknown_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["other"],
            _parent_child_schema(),
            "CTE 'inner_cte'",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("preserve_tables_unknown" in i.issue_id for i in issues)

    def test_cte_unreachable_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["child"],
            _parent_child_schema(),
            "CTE 'inner_cte'",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("preserve_tables_unreachable" in i.issue_id for i in issues)

    def test_cte_noop_preserve_table_raises(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "child"],
            ["parent"],
            _parent_child_schema(),
            "CTE 'inner_cte'",
            join_signature=["parent.id->child.parent_id"],
        )
        assert any("preserve_tables_noop" in i.issue_id for i in issues)

    def test_preserve_tables_rejects_probe_cte_name(self) -> None:
        issues = validate_preserve_tables(
            ["parent", "probe"],
            ["probe"],
            _parent_child_schema(),
            "main query",
            join_signature=["parent.id->probe.k"],
            probe_cte_names=frozenset({"probe"}),
        )
        assert any("preserve_tables_probe" in i.issue_id for i in issues)


class TestProbeCteModifierConflicts:
    def test_distinct_select_index_on_probe_raises(self) -> None:
        cte = RuntimeCteStep(
            cte_name="probe",
            tables=["child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
            output_columns=["id"],
            emission="semi_join",
            distinct_select_index=0,
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        issues = validate_probe_cte_modifiers(intent)
        assert any("probe_cte_distinct_select_index" in i.issue_id for i in issues)

    def test_distinct_on_on_probe_raises(self) -> None:
        cte = RuntimeCteStep(
            cte_name="probe",
            tables=["child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
            output_columns=["id"],
            emission="anti_join",
            distinct_on=[NormalizedExpr.from_column("child.id")],
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        issues = validate_probe_cte_modifiers(intent)
        assert any("probe_cte_distinct_on" in i.issue_id for i in issues)

    def test_limit_on_probe_raises(self) -> None:
        cte = RuntimeCteStep(
            cte_name="probe",
            tables=["child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
            output_columns=["id"],
            emission="semi_join",
            limit=5,
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        issues = validate_probe_cte_modifiers(intent)
        assert any("probe_cte_limit" in i.issue_id for i in issues)


class TestProbeEdgeCandidateCollapse:
    def test_candidates_differing_only_on_probe_edge_collapse(self) -> None:
        probe = frozenset({"probe"})
        candidates = [
            {
                "candidate_id": "J01",
                "join_path_signature": ["parent.id->probe.k", "parent.id->child.parent_id"],
            },
            {
                "candidate_id": "J02",
                "join_path_signature": ["parent.id->probe.other_k", "parent.id->child.parent_id"],
            },
        ]
        collapsed = collapse_probe_edge_candidate_variation(candidates, probe)
        assert len(collapsed) == 1

    def test_reserved_marker_suffix_rejected(self) -> None:
        cte = RuntimeCteStep(
            cte_name="probe",
            tables=["child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
            output_columns=[f"id{ANTI_JOIN_PRESENCE_COLUMN_SUFFIX}"],
            emission="join_table",
        )
        intent = RuntimeIntent(
            tables=["parent"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
        issues = validate_cte_emission_shapes(intent, _parent_child_schema())
        assert any("reserved anti-join" in i.message for i in issues)


class TestJoinOrphanRateDiagnostic:
    def test_orphan_rate_high_emits_warning(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_path_signature=["parent.id->child.parent_id"],
        )
        token = set_diagnostic_collector([])
        try:
            emit_join_orphan_rate_diagnostics(
                intent,
                schema,
                join_signature=["parent.id->child.parent_id"],
                edge_kinds=["catalog_fk"],
                from_anchor="parent",
            )
            diags = drain_diagnostic_collector()
        finally:
            reset_diagnostic_collector(token)
        assert any(d.code == DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH for d in diags)
        assert JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR > 0

    def test_orphan_diagnostic_skipped_when_preservation_active(self) -> None:
        schema = _parent_child_schema()
        intent = RuntimeIntent(
            tables=["parent", "child"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            preserve_tables=["parent"],
            chosen_join_path_signature=["parent.id->child.parent_id"],
        )
        token = set_diagnostic_collector([])
        try:
            emit_join_orphan_rate_diagnostics(
                intent,
                schema,
                join_signature=["parent.id->child.parent_id"],
                from_anchor="parent",
                preserve_tables=["parent"],
            )
            diags = drain_diagnostic_collector()
        finally:
            reset_diagnostic_collector(token)
        assert not any(d.code == DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH for d in diags)
