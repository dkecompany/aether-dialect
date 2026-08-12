"""Tests for federation intent decomposition and planning."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    FederationContext,
    NormalizedExpr,
    PredicateGroup,
    SensitivityClassification,
    SpaceContext,
    WhereParam,
)
from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._federation_plan import (
    _build_source_sub_intent,
    apply_projected_keys_to_intent,
    federation_plan_is_degenerate,
    plan_federated_intent,
    render_federation_glue,
    resolve_federated_combine,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import union_member_graph_pair


def _graph(table: str, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_plan",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"t_a": "a", "t_b": "b"},
    "cross_source_joins": [
        {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
    ],
}


def test_single_source_fast_path() -> None:
    manifest = parse_federation_manifest({**_MANIFEST, "cross_source_joins": []}, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", ""), "b": _graph("t_b", "")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "a"
    assert plan.ineligible_reason is None
    assert plan.combine is None or plan.combine == ()
    assert plan.stages == ()
    assert plan.scope_sources == frozenset({"a"})


def test_multi_source_steps_use_manifest_when_source_id_missing() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    assert plan.ineligible_reason is None


def test_multi_source_produces_steps() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", ""), "b": _graph("t_b", "")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    assert plan.combine is not None
    assert plan.ineligible_reason is None


def test_multi_source_scalar_count_prunes_foreign_select_cols() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    by_source = {step.source_id: step for step in plan.steps}
    assert by_source["b"].sub_intent.grain == "row_level"
    assert "t_b.id" in by_source["b"].projected_keys
    assert by_source["a"].sub_intent.grain == "row_level"
    assert by_source["a"].sub_intent.select_cols
    assert "t_a.id" in str(by_source["a"].sub_intent.select_cols[0].expr)


def test_horror_style_scalar_projects_join_keys_and_aggregate_columns() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_horror",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "rental": "storefront",
                "customer": "storefront",
                "film": "catalog",
                "category": "catalog",
                "inventory": "catalog",
            },
            "cross_source_joins": [
                {
                    "left": "rental.inventory_id",
                    "right": "inventory.inventory_id",
                    "kind": "inner",
                    "logical_key": "inventory_id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={
            name: TableMetadata(
                name=name,
                columns={
                    "inventory_id": ColumnMetadata(name="inventory_id", data_type="integer", sensitivity="none"),
                    "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
                    "name": ColumnMetadata(name="name", data_type="varchar", sensitivity="none"),
                },
                primary_key=["inventory_id"] if name == "inventory" else ["customer_id"],
                foreign_keys=[],
                source_id="",
            )
            for name in ("rental", "customer", "film", "category", "inventory")
        },
        join_paths_multi={},
    )
    horror_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("category.name"),
        op="=",
        value_type="string",
        raw_value="Horror",
    )
    intent = RuntimeIntent(
        tables=["rental", "customer", "film", "category"],
        grain="scalar",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_agg("count", "rental.customer_id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([horror_filter]),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    by_source = {step.source_id: step for step in plan.steps}
    assert by_source["catalog"].sub_intent.grain == "row_level"
    assert "inventory.inventory_id" in by_source["catalog"].projected_keys
    assert by_source["storefront"].sub_intent.grain == "grouped"
    assert any(sc.is_aggregated for sc in (by_source["storefront"].sub_intent.select_cols or []))
    assert "rental.customer_id" in by_source["storefront"].projected_keys
    assert "rental.inventory_id" in by_source["storefront"].projected_keys
    assert plan.residual is not None
    assert plan.residual.select_cols
    catalog_sub = by_source["catalog"].sub_intent
    assert catalog_sub.where and catalog_sub.where.leaves()


def test_cross_source_grouped_aggregate_routes_to_residual() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_agg",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "rental": "storefront",
                "category": "catalog",
                "item": "catalog",
            },
            "cross_source_joins": [
                {
                    "left": "rental.inventory_id",
                    "right": "item.item_id",
                    "kind": "inner",
                    "logical_key": "inventory_id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {
            "storefront": SchemaGraph(
                tables={
                    "rental": TableMetadata(
                        name="rental",
                        columns={
                            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                            "inventory_id": ColumnMetadata(
                                name="inventory_id",
                                data_type="integer",
                                sensitivity="none",
                            ),
                        },
                        primary_key=["id"],
                        foreign_keys=[],
                        source_id="storefront",
                    ),
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
            "catalog": SchemaGraph(
                tables={
                    "category": TableMetadata(
                        name="category",
                        columns={"name": ColumnMetadata(name="name", data_type="text", sensitivity="none")},
                        primary_key=[],
                        foreign_keys=[],
                        source_id="catalog",
                    ),
                    "item": TableMetadata(
                        name="item",
                        columns={
                            "item_id": ColumnMetadata(name="item_id", data_type="integer", sensitivity="none"),
                        },
                        primary_key=["item_id"],
                        foreign_keys=[],
                        source_id="catalog",
                    ),
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["rental", "category", "item"],
        grain="grouped",
        select_cols=[],
        group_by_cols=[NormalizedExpr.from_column("category.name")],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.residual is not None
    assert len(plan.residual.group_by_cols) == 1
    assert plan.residual.group_by_cols[0].column_ref == "category.name"


def test_sub_intent_filters_isolated_from_parent() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", ""), "b": _graph("t_b", "")},
        manifest,
    )
    parent_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("t_a.id"),
        op="=",
        value_type="string",
        raw_value="1",
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([parent_filter]),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    parent_filter.raw_value = "mutated"
    for step in plan.steps:
        for fp in step.sub_intent.where.leaves() if step.sub_intent.where else []:
            assert fp.raw_value != "mutated"


def test_union_plan_keeps_join_combine_when_both_present() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_join",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "payment": "storefront",
                "category": "catalog",
            },
            "cross_source_joins": [
                {
                    "left": "payment.id",
                    "right": "category.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "storefront", "table": "payment", "columns": {}},
                        {"source": "catalog", "table": "payment", "columns": {}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    composite = SchemaGraph(
        tables={
            "payment": TableMetadata(
                name="payment",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
            "category": TableMetadata(
                name="category",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["payment", "category"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.union_specs
    assert isinstance(plan.combine, tuple)
    assert plan.combine
    assert len(plan.steps) == 2


def test_union_only_glue_renders_union_all() -> None:
    from aetherdialect._contracts_core import ResidualSpec, UnionSpec

    plan = FederatedPlan(
        steps=(),
        union_specs=(
            UnionSpec(
                logical_table="payment",
                member_source_ids=("storefront", "catalog"),
                semantics="union",
            ),
        ),
        residual=ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("payment.id")),),
        ),
    )
    glue = render_federation_glue(
        plan,
        {"storefront": "src_storefront", "catalog": "src_catalog"},
    )
    assert "UNION ALL" in glue.upper()
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_agg",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "payment": "storefront",
                "category": "catalog",
                "item_category": "catalog",
            },
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "storefront", "table": "payment", "columns": {}},
                        {"source": "catalog", "table": "payment", "columns": {}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    composite = compose_composite_graph(
        {
            "storefront": _graph("payment", "storefront"),
            "catalog": _graph("category", "catalog"),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["payment", "category", "item_category"],
        grain="grouped",
        select_cols=[],
        group_by_cols=[NormalizedExpr.from_column("category.name")],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    assert plan.residual.group_by_cols


def test_residual_spec_carries_ir_objects() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    cross_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("t_a.id"),
        op="=",
        right_expr=NormalizedExpr.from_column("t_b.id"),
        value_type="column",
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([cross_filter]),
        limit=5,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.residual is not None
    assert isinstance(plan.residual.where.leaves()[0], WhereParam)
    assert plan.residual.limit == 5


def test_render_federation_glue_residual_projection() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="grouped",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id"))],
        group_by_cols=[NormalizedExpr.from_column("t_b.id")],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"})
    assert "GROUP BY" in glue.upper()
    assert "COUNT" in glue.upper()
    assert "fed_base" in glue


def test_projected_keys_applied_before_sql_generation() -> None:
    intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    projected = ("t_a.id",)
    updated = apply_projected_keys_to_intent(intent, projected)
    assert len(updated.select_cols) == 1
    assert updated.select_cols[0].expr.column_ref == "t_a.id"


def test_render_residual_limit_only() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=3,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"})
    assert "LIMIT 3" in glue


def test_multiple_declared_joins_same_table_pair_picks_one() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_multi_key",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_a_alt": "a", "t_b_alt": "b"},
            "cross_source_joins": [
                {
                    "left": "t_a.alt_id",
                    "right": "t_b.alt_id",
                    "kind": "inner",
                    "logical_key": "alt_id",
                },
                {
                    "left": "t_a.id",
                    "right": "t_b.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={
            name: TableMetadata(
                name=name,
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "alt_id": ColumnMetadata(name="alt_id", data_type="integer", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            )
            for name in ("t_a", "t_b", "t_a_alt", "t_b_alt")
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.combine is not None
    assert len(plan.combine) == 1
    assert plan.combine[0].logical_key == "alt_id"


def test_resolve_federated_combine_honors_preset_choice() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_choice",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_a_alt": "a", "t_b_alt": "b"},
            "cross_source_joins": [
                {
                    "left": "t_a.alt_id",
                    "right": "t_b.alt_id",
                    "kind": "inner",
                    "logical_key": "alt_id",
                },
                {
                    "left": "t_a.id",
                    "right": "t_b.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={
            name: TableMetadata(
                name=name,
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "alt_id": ColumnMetadata(name="alt_id", data_type="integer", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            )
            for name in ("t_a", "t_b", "t_a_alt", "t_b_alt")
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    resolved = resolve_federated_combine(
        "count rows",
        plan,
        manifest,
        composite,
        preset_choices={"jc0": "J01"},
    )
    assert resolved.combine is not None
    assert len(resolved.combine) == 1
    assert resolved.combine[0].logical_key == "id"
    assert "t_a.id" in resolved.steps[0].projected_keys


def test_sensitive_cross_source_join_excluded_from_combine() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_sensitive",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {
                    "left": "t_a.secret_id",
                    "right": "t_b.secret_id",
                    "kind": "inner",
                    "logical_key": "secret_id",
                },
                {
                    "left": "t_a.id",
                    "right": "t_b.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={
            "t_a": TableMetadata(
                name="t_a",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "secret_id": ColumnMetadata(
                        name="secret_id",
                        data_type="integer",
                        sensitivity=SensitivityClassification.RESTRICTED,
                    ),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
            "t_b": TableMetadata(
                name="t_b",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "secret_id": ColumnMetadata(
                        name="secret_id",
                        data_type="integer",
                        sensitivity=SensitivityClassification.RESTRICTED,
                    ),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.combine is not None
    assert len(plan.combine) == 1
    assert plan.combine[0].logical_key == "id"
    step_a = next(step for step in plan.steps if step.source_id == "a")
    assert "t_a.id" in step_a.projected_keys
    assert "t_a.secret_id" not in step_a.projected_keys


_UNION_MANIFEST = {
    "federation_id": "fed_union_plan",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {
        "payment_a": "a",
        "payment_b": "b",
    },
    "cross_source_joins": [],
}


def _union_mappings() -> FederationMappings:
    return parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )


def _union_composite() -> SchemaGraph:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    return compose_composite_graph(union_member_graph_pair("payment_a", "payment_b"), manifest, mappings)


def test_scalar_aggregate_over_union_plans_member_steps_and_residual() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = _union_composite()
    intent = RuntimeIntent(
        tables=["payment"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert {step.source_id for step in plan.steps} == {"a", "b"}
    assert plan.union_specs
    assert plan.union_specs[0].logical_table == "payment"
    assert plan.residual is not None
    assert plan.residual.select_cols
    assert "count" in str(plan.residual.select_cols[0].expr).lower()
    assert plan.stages
    assert plan.stages[-1].kind == "coordinator"


def test_chosen_join_path_signature_crossing_sources_plans_combine() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=["t_b"],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert {step.source_id for step in plan.steps} == {"a", "b"}
    assert plan.combine is not None
    assert len(plan.combine) == 1
    assert plan.combine[0].logical_key == "id"


def test_federation_context_deny_on_collapsed_member_table_raises() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    members = union_member_graph_pair("payment_a", "payment_b")
    ctx = FederationContext(deny_objects=frozenset({"payment_a"}))
    with pytest.raises(ConfigError, match="collapsed member table 'payment_a'"):
        compose_composite_graph(members, manifest, mappings, master_context=ctx)


def test_cross_source_filter_routes_to_residual_not_gate() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    cross_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("t_a.id"),
        op="=",
        right_expr=NormalizedExpr.from_column("t_b.id"),
        value_type="column",
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([cross_filter]),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    assert plan.residual.where
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "WHERE" in glue.upper()
    assert "SELECT *" not in glue


def test_union_and_join_glue_renders_both() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_union_join_glue",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment": "a", "category": "b"},
            "cross_source_joins": [
                {
                    "left": "payment.id",
                    "right": "category.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    composite = SchemaGraph(
        tables={
            "payment": TableMetadata(
                name="payment",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
            "category": TableMetadata(
                name="category",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["payment", "category"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.union_specs
    assert plan.combine
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "WITH" in glue.upper()
    assert "JOIN" in glue.upper()
    assert "UNION ALL" in glue.upper()
    assert "SELECT *" not in glue


def test_scalar_is_aggregated_union_logical_table_is_eligible() -> None:
    from unittest.mock import patch

    from aetherdialect._federation_plan import _cross_source_aggregate_ineligible_reason, federation_table_set

    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = _union_composite()
    intent = RuntimeIntent(
        tables=["payment"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    table_set = federation_table_set(intent, composite, manifest, mappings)
    with patch("aetherdialect._federation_plan._is_sql_aggregate_select_col", return_value=True):
        reason = _cross_source_aggregate_ineligible_reason(
            intent,
            manifest,
            mappings,
            dict(table_set.source_by_table),
            schema=composite,
        )
    assert reason is None


def test_multi_member_logical_table_plan_is_not_degenerate() -> None:
    manifest = parse_federation_manifest(_UNION_MANIFEST, include_derived_roster=True)
    mappings = _union_mappings()
    composite = _union_composite()
    intent = RuntimeIntent(
        tables=["payment"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("payment.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.union_specs
    assert federation_plan_is_degenerate(plan) is False


def test_space_context_accepts_source_qualified_column_spec() -> None:
    ctx = SpaceContext(
        tables=frozenset({"t_a"}),
        columns=frozenset({"a.t_a.id"}),
    )
    assert ctx.columns == frozenset({"t_a.id"})


def test_three_part_cross_source_join_plans_multi_source() -> None:
    manifest = parse_federation_manifest(
        {
            **_MANIFEST,
            "cross_source_joins": [
                {"left": "a.t_a.id", "right": "b.t_b.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert {step.source_id for step in plan.steps} == {"a", "b"}


def test_space_excludes_table_from_federated_plan() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    space = SpaceContext(tables=frozenset({"t_a"}))
    plan = plan_federated_intent(intent, composite, manifest, space=space)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "a"
    assert plan.combine is None
    assert federation_plan_is_degenerate(plan)


@pytest.mark.fast
def test_space_excluded_table_produces_no_sub_intent_step_or_bridge() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        preserve_tables=["t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    space = SpaceContext(tables=frozenset({"t_a"}))
    mappings = FederationMappings(version="0.2.3")
    tables_all = {"t_a", "t_b"}
    source_by_table = {"t_a": "a", "t_b": "b"}

    step_b = _build_source_sub_intent(
        intent,
        "b",
        tables_all,
        source_by_table,
        mappings,
        composite,
        manifest,
        member_schema=_graph("t_b", "b"),
        space=space,
    )
    assert step_b is None

    step_a = _build_source_sub_intent(
        intent,
        "a",
        tables_all,
        source_by_table,
        mappings,
        composite,
        manifest,
        member_schema=_graph("t_a", "a"),
        space=space,
    )
    assert step_a is not None
    assert "t_b" not in (step_a.sub_intent.preserve_tables or [])

    plan_intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(plan_intent, composite, manifest, space=space)
    assert plan.ineligible_reason is None
    assert {step.source_id for step in plan.steps} == {"a"}
    assert plan.combine is None


_ONE_MEMBER_MANIFEST = {
    "federation_id": "fed_one",
    "sources": [{"source_id": "solo", "engine": "duckdb", "role": "owner"}],
    "table_namespace": {"t_solo": "solo"},
    "cross_source_joins": [],
}


def test_one_member_roster_plans_single_degenerate_step() -> None:
    manifest = parse_federation_manifest(_ONE_MEMBER_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"solo": _graph("t_solo", "solo")}, manifest)
    intent = RuntimeIntent(
        tables=["t_solo"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "solo"
    assert federation_plan_is_degenerate(plan)


_REPLICA_MANIFEST = {
    "federation_id": "fed_replica_plan",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {
        "entity_a": "a",
        "entity_b": "b",
    },
    "cross_source_joins": [],
}


def _replica_mappings() -> FederationMappings:
    return parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "replica",
                    "authoritative_source": "b",
                    "members": [
                        {"source": "a", "table": "entity_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )


def test_replica_logical_table_plans_authoritative_member_only() -> None:
    manifest = parse_federation_manifest(_REPLICA_MANIFEST, include_derived_roster=True)
    mappings = _replica_mappings()
    composite = compose_composite_graph(
        {"a": _graph("entity_a", "a"), "b": _graph("entity_b", "b")},
        manifest,
        mappings,
    )
    intent = RuntimeIntent(
        tables=["entity"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "b"


@pytest.mark.fast
def test_space_deny_columns_excludes_column_from_federated_plan() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    unscoped = plan_federated_intent(intent, composite, manifest)
    assert unscoped.ineligible_reason is None
    assert len(unscoped.steps) == 1
    space = SpaceContext(deny_columns=frozenset({"t_a.id"}))
    plan = plan_federated_intent(intent, composite, manifest, space=space)
    assert plan.steps == ()
    assert plan.ineligible_reason is not None
    assert "space" in plan.ineligible_reason.lower()


@pytest.mark.fast
def test_space_deny_objects_excludes_object_from_federated_plan() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    unscoped = plan_federated_intent(intent, composite, manifest)
    assert unscoped.ineligible_reason is None
    assert len(unscoped.steps) == 2
    space = SpaceContext(deny_objects=frozenset({"t_b"}))
    plan = plan_federated_intent(intent, composite, manifest, space=space)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "a"
    assert plan.combine is None


@pytest.mark.fast
def test_space_partial_deny_of_union_backed_logical_table_raises() -> None:
    from aetherdialect._federation_compose import validate_federation_context_against_mappings

    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    ctx = FederationContext(deny_objects=frozenset({"payment"}))
    with pytest.raises(ConfigError, match="partially denies"):
        validate_federation_context_against_mappings(ctx, mappings)


@pytest.mark.fast
def test_federation_space_for_choice_port_carries_deny_lists() -> None:
    from types import SimpleNamespace

    from aetherdialect._main_execution import MainExecutionOps

    port = SimpleNamespace(
        space_tables=frozenset({"t_a"}),
        space_columns=frozenset(),
        space_deny_objects=frozenset({"t_b"}),
        space_deny_columns=frozenset({"t_a.id"}),
    )
    space = MainExecutionOps._federation_space_for_choice_port(port)
    assert space is not None
    assert space.tables == frozenset({"t_a"})
    assert space.deny_objects == frozenset({"t_b"})
    assert space.deny_columns == frozenset({"t_a.id"})
    deny_only = SimpleNamespace(
        space_tables=frozenset(),
        space_columns=frozenset(),
        space_deny_objects=frozenset({"t_b"}),
        space_deny_columns=frozenset(),
    )
    deny_space = MainExecutionOps._federation_space_for_choice_port(deny_only)
    assert deny_space is not None
    assert deny_space.deny_objects == frozenset({"t_b"})


@pytest.mark.fast
def test_validate_space_source_qualified_column_uses_shared_resolver() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    ctx = SpaceContext(tables=frozenset({"t_a"}), columns=frozenset({"t_a.id"}))
    object.__setattr__(ctx, "columns", frozenset({"a.t_a.id"}))
    validated = MainExecutionOps.validate_space_context_against_graph(
        ctx,
        composite,
        federation_manifest=manifest,
    )
    assert validated.columns == frozenset({"t_a.id"})
    bad = SpaceContext(tables=frozenset({"t_a"}), columns=frozenset({"t_a.id"}))
    object.__setattr__(bad, "columns", frozenset({"ghost.t_a.id"}))
    with pytest.raises(ConfigError, match="unknown federation source"):
        MainExecutionOps.validate_space_context_against_graph(
            bad,
            composite,
            federation_manifest=manifest,
        )


@pytest.mark.fast
def test_dropped_member_raises_federation_invariant_error() -> None:
    from unittest.mock import patch

    from aetherdialect._contracts_base import FederationInvariantError
    from aetherdialect._federation_plan import _build_source_sub_intent

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    real_build = _build_source_sub_intent

    def _drop_b(*args: object, **kwargs: object) -> object:
        source_id = args[1] if len(args) > 1 else kwargs.get("source_id")
        if source_id == "b":
            return None
        return real_build(*args, **kwargs)

    with patch("aetherdialect._federation_plan._build_source_sub_intent", side_effect=_drop_b):
        with pytest.raises(FederationInvariantError, match=r"dropped member.*\['b'\].*scope discovery") as exc_info:
            plan_federated_intent(intent, composite, manifest)
    assert "b" in str(exc_info.value)
    assert "scope discovery" in str(exc_info.value)


@pytest.mark.fast
def test_genuinely_ineligible_still_returns_ineligible_reason() -> None:
    from aetherdialect._contracts_base import PredicateGroup

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t_a.id"),
                    op="=",
                    right_expr=NormalizedExpr.from_column("t_b.id"),
                    value_type="column",
                ),
            ),
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source OR filter is not supported: t_a.id = t_b.id"
    assert plan.steps == ()


@pytest.mark.fast
def test_residual_grouped_group_by_keeps_only_post_join_keys() -> None:
    """Local dimension keys stay; unattributable group_by keys must not widen residual grain."""
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_agg",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "rental": "storefront",
                "category": "catalog",
                "item": "catalog",
            },
            "cross_source_joins": [
                {
                    "left": "rental.inventory_id",
                    "right": "item.item_id",
                    "kind": "inner",
                    "logical_key": "inventory_id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {
            "storefront": SchemaGraph(
                tables={
                    "rental": TableMetadata(
                        name="rental",
                        columns={
                            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                            "inventory_id": ColumnMetadata(
                                name="inventory_id",
                                data_type="integer",
                                sensitivity="none",
                            ),
                        },
                        primary_key=["id"],
                        foreign_keys=[],
                        source_id="storefront",
                    ),
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
            "catalog": SchemaGraph(
                tables={
                    "category": TableMetadata(
                        name="category",
                        columns={"name": ColumnMetadata(name="name", data_type="text", sensitivity="none")},
                        primary_key=[],
                        foreign_keys=[],
                        source_id="catalog",
                    ),
                    "item": TableMetadata(
                        name="item",
                        columns={
                            "item_id": ColumnMetadata(name="item_id", data_type="integer", sensitivity="none"),
                        },
                        primary_key=["item_id"],
                        foreign_keys=[],
                        source_id="catalog",
                    ),
                },
                join_paths_multi=recompute_join_paths_multi({}),
            ),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["rental", "category", "item"],
        grain="grouped",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "rental.id"))],
        group_by_cols=[
            NormalizedExpr.from_column("category.name"),
            NormalizedExpr(),
        ],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    residual_refs = [g.column_ref for g in plan.residual.group_by_cols]
    assert residual_refs == ["category.name"]
    assert plan.grain == "grouped"
    assert plan.residual.select_cols
