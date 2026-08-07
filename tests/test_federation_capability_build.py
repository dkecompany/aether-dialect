"""Regression tests for federation capability, member build, staged execution, and semi-join."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import PredicateGroup, WhereParam
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _coordinator_promotes_spanning_windows,
    _spanning_cte_names,
    compose_composite_graph,
    inject_semijoin_where,
    intersect_member_where_ops,
    parse_federation_manifest,
    plan_federated_stages,
    stamp_federation_member_graph,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import (
    compute_database_feature_capability,
    intersect_member_database_feature_capabilities,
    recompute_join_paths_multi,
)


def _member_graph(name: str, *, source_id: str, has_array: bool = False) -> SchemaGraph:
    columns = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
    }
    if has_array:
        columns["tags"] = ColumnMetadata(
            name="tags",
            data_type="text[]",
            sensitivity="none",
            element_type="text",
        )
    table = TableMetadata(
        name=name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}",
        effective_structural_hash=f"eff_{source_id}",
    )


def test_intersect_member_database_feature_capabilities_uses_intersection() -> None:
    members = {
        "a": _member_graph("ta", source_id="a", has_array=True),
        "b": _member_graph("tb", source_id="b", has_array=False),
    }
    intersected = intersect_member_database_feature_capabilities(members)
    assert not intersected.has_array_columns
    assert members["a"].database_feature_capability.has_array_columns
    unionish = compute_database_feature_capability(
        SchemaGraph(
            tables={**members["a"].tables, **members["b"].tables},
            join_paths_multi=recompute_join_paths_multi({**members["a"].tables, **members["b"].tables}),
        ),
    )
    assert unionish.has_array_columns


def test_compose_stamps_membership_and_intersects_capability() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_generic",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": _member_graph("ta", source_id="a", has_array=True),
        "b": _member_graph("tb", source_id="b", has_array=False),
    }
    composite = compose_composite_graph(members, manifest)
    assert members["a"].federation_membership == {
        "federation_id": "fed_generic",
        "source_id": "a",
        "engine": "duckdb",
    }
    assert members["b"].federation_membership == {
        "federation_id": "fed_generic",
        "source_id": "b",
        "engine": "duckdb",
    }
    assert not composite.database_feature_capability.has_array_columns


def test_stamp_federation_member_graph_sets_membership() -> None:
    graph = _member_graph("t", source_id="west")
    stamp_federation_member_graph(graph, federation_id="fed_west", source_id="west")
    assert graph.federation_membership == {"federation_id": "fed_west", "source_id": "west"}


def test_intersect_member_where_ops_uses_engine_types() -> None:
    ops = intersect_member_where_ops(
        engine_types_by_source={"a": "postgresql", "b": "bigquery"},
    )
    assert "ilike" in ops
    assert "=" in ops


def test_apply_deny_objects_filter_on_full_build_path() -> None:
    from aetherdialect._schema_build import apply_full_build_deny_objects

    tables = {
        "orders": TableMetadata(
            name="orders",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "secret": TableMetadata(
            name="secret",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    sg = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    apply_full_build_deny_objects(sg, frozenset({"secret"}))
    assert "orders" in sg.tables
    assert "secret" not in sg.tables


def test_plan_federated_stages_emits_spanning_cte_stage() -> None:
    source_by_table = {"ta": "a", "tb": "b"}
    cte = RuntimeCteStep(
        cte_name="span_cte",
        tables=["ta", "tb"],
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    assert _spanning_cte_names(intent.cte_steps or (), source_by_table) == ("span_cte",)
    stages = plan_federated_stages(
        {"a", "b"},
        (),
        intent=intent,
        source_by_table=source_by_table,
    )
    cte_stages = [stage for stage in stages if stage.kind == "cte"]
    assert len(cte_stages) == 1
    assert cte_stages[0].spanning_cte_names == ("span_cte",)
    assert stages[-1].kind == "coordinator"


def test_inject_semijoin_where_allocates_param_key() -> None:
    intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    updated = inject_semijoin_where(intent, "id", [1])
    fp = (updated.where.leaves() if updated.where else [])[0]
    assert fp.param_key == "p1"
    assert updated.param_values == {"p1": [1]}


def test_coordinator_promotes_spanning_windows_above_cross_source_joins() -> None:
    from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec

    source_by_table = {"ta": "a", "tb": "b"}
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="grouped",
        select_cols=[],
        group_by_cols=[NormalizedExpr.from_column("ta.id")],
        order_by_cols=[],
        where=None,
        window_registry=[
            WindowRegistryStep(
                registry_id="w1",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[NormalizedExpr.from_column("ta.id"), NormalizedExpr.from_column("tb.id")],
                    order_by=[],
                ),
            ),
        ],
    )
    assert _coordinator_promotes_spanning_windows(intent, source_by_table)


_CROSS_SOURCE_MANIFEST = {
    "federation_id": "fed_eligibility",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"ta": "a", "tb": "b"},
    "cross_source_joins": [
        {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _eligibility_graph(table: str, source_id: str) -> SchemaGraph:
    table_meta = TableMetadata(
        name=table,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table: table_meta}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


def test_federation_eligibility_checked_before_intent_confirm() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    ineligible_intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("ta.id"),
                    op="=",
                    right_expr=NormalizedExpr.from_column("tb.id"),
                    value_type="column",
                ),
            ),
        ),
    )
    eligible_intent = RuntimeIntent(
        tables=["ta"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    owner = MagicMock(_is_aether_federation=True)
    owner._federation_manifest = manifest
    owner._federation_mappings = None
    choice_port = MagicMock(_owner=owner)

    with patch("aetherdialect._main_execution.MainExecutionOps._handle_federation_ineligible_plan") as mock_handle:
        assert (
            MainExecutionOps._check_federation_eligibility_before_confirm(
                ineligible_intent,
                composite,
                {},
                choice_port,
            )
            is False
        )
        mock_handle.assert_called_once()
        plan = mock_handle.call_args.args[0]
        assert plan.ineligible_reason == "cross-source OR filter is not supported: ta.id = tb.id"

    assert (
        MainExecutionOps._check_federation_eligibility_before_confirm(
            eligible_intent,
            composite,
            {},
            choice_port,
        )
        is True
    )

    non_fed_port = MagicMock(_owner=MagicMock(_is_aether_federation=False))
    assert (
        MainExecutionOps._check_federation_eligibility_before_confirm(
            ineligible_intent,
            composite,
            {},
            non_fed_port,
        )
        is True
    )


def test_predicate_renderer_refuses_unbound_contains_param() -> None:
    from unittest.mock import MagicMock

    from aetherdialect._contracts_core import WhereParam
    from aetherdialect._sql_gen import _render_predicate_clause

    pred = WhereParam(
        left_expr=NormalizedExpr.from_column("t.tags"),
        op="contains",
        value_type="string",
        raw_value="x",
    )
    parts = _render_predicate_clause(pred, MagicMock(), schema=None, cte_outputs={})
    assert parts == ""


@pytest.mark.fast
def test_cross_source_distinct_is_coordinator_residual() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_distinct",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _member_graph("ta", source_id="a"), "b": _member_graph("tb", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("ta.id")),
            SelectCol(expr=NormalizedExpr.from_column("tb.id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        distinct_select_index=0,
    )
    from aetherdialect._federation import plan_federated_intent

    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.residual is not None
    assert plan.residual.distinct_select_index == 0


@pytest.mark.fast
def test_cross_source_order_by_is_coordinator_residual() -> None:
    from aetherdialect._contracts_base import MulGroup
    from aetherdialect._contracts_core import OrderByCol
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_order",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _member_graph("ta", source_id="a"), "b": _member_graph("tb", source_id="b")},
        manifest,
    )
    cross_order = NormalizedExpr(
        add_groups=[
            MulGroup(multiply=[NormalizedExpr.from_column("ta.id"), NormalizedExpr.from_column("tb.id")]),
        ],
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[OrderByCol(expr=cross_order, direction="ASC")],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.residual is not None
    assert len(plan.residual.order_by_cols) == 1


@pytest.mark.fast
def test_cross_source_where_group_disjunction_is_ineligible() -> None:
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("ta.id"),
                    op="=",
                    value_type="integer",
                    raw_value=1,
                ),
                WhereParam(
                    left_expr=NormalizedExpr.from_column("tb.id"),
                    op="=",
                    value_type="integer",
                    raw_value=2,
                ),
            ),
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is not None
    assert plan.ineligible_reason.startswith("cross-source predicate disjunction is not supported:")
    assert "ta.id =" in plan.ineligible_reason
    assert "tb.id =" in plan.ineligible_reason


@pytest.mark.fast
def test_cross_source_window_above_join_routes_to_residual() -> None:
    from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[
            WindowRegistryStep(
                registry_id="w01",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[
                        NormalizedExpr.from_column("ta.id"),
                        NormalizedExpr.from_column("tb.id"),
                    ],
                    order_by=[],
                ),
            ),
        ],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None
    assert any(getattr(entry, "registry_id", "") == "w01" for entry in plan.residual.window_registry)


@pytest.mark.fast
def test_cross_source_window_without_combine_is_ineligible() -> None:
    from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_window_no_join",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[
            WindowRegistryStep(
                registry_id="w01",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[
                        NormalizedExpr.from_column("ta.id"),
                        NormalizedExpr.from_column("tb.id"),
                    ],
                    order_by=[],
                ),
            ),
        ],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is not None
    assert (
        plan.ineligible_reason == "cross-source window is not supported: w01 (row_number)"
        or "join path is not declared" in plan.ineligible_reason
    )


@pytest.mark.fast
def test_cross_source_scalar_subquery_cte_is_ineligible() -> None:
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[
            RuntimeCteStep(
                cte_name="lookup_span",
                tables=["ta", "tb"],
                select_cols=[
                    SelectCol(expr=NormalizedExpr.from_column("ta.id")),
                    SelectCol(expr=NormalizedExpr.from_column("tb.id")),
                ],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
                emission="scalar_subquery",
            ),
        ],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source correlated subquery is not supported: lookup_span"


@pytest.mark.fast
def test_cross_source_stddev_aggregate_is_ineligible() -> None:
    from aetherdialect._federation import plan_federated_intent

    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("stddev", "ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source aggregate not supported: stddev(ta.id)"


def test_federation_flat_predicate_conjunction_is_not_nested() -> None:
    from aetherdialect._contracts_base import DatabaseFeatureCapability
    from aetherdialect._federation import _federation_ir_capability_reason

    cap = DatabaseFeatureCapability(
        table_count=1,
        fk_edge_count=0,
        has_numeric_measures=False,
        has_date_columns=False,
        has_array_columns=False,
        has_categorical_columns=False,
        max_tables_on_any_join_path=1,
        max_fk_chain_depth=0,
        has_self_referential_fk=False,
        tables_supporting_self_join=frozenset(),
        has_window_capable_table_sets=False,
        aggregatable_columns_by_table={},
        date_columns_by_table={},
        array_columns_by_table={},
        supports_predicate_nesting=False,
    )
    where = PredicateGroup(
        op="and",
        predicates=(
            WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", raw_value=1),
            WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="=", raw_value=2),
            WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=", raw_value=3),
        ),
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )
    assert _federation_ir_capability_reason(intent, cap) is None
