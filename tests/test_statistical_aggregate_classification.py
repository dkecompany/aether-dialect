"""Statistical aggregate classification via parse_expr_string (is_aggregated)."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import (
    AGGREGATE_FUNCTION_NAMES,
    FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest, plan_federated_intent
from aetherdialect._intent_expr import parse_expr_string, parse_intent_response
from aetherdialect._schema_graph import recompute_join_paths_multi

_STATISTICAL_AGGS: tuple[str, ...] = (
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "variance",
    "var_pop",
    "var_samp",
    "median",
)

_CROSS_SOURCE_MANIFEST = {
    "federation_id": "fed_stat_agg_class",
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


@pytest.mark.fast
@pytest.mark.parametrize("func", _STATISTICAL_AGGS)
def test_parse_expr_string_statistical_agg_is_aggregated(func: str) -> None:
    expr = parse_expr_string(f"{func}(orders.amount)")
    sc = SelectCol(expr=expr)
    assert sc.is_aggregated is True
    assert expr.add_groups
    assert expr.add_groups[0].agg_func is not None
    assert expr.add_groups[0].scalar_func is None


@pytest.mark.fast
def test_parenthesized_scalar_funcs_are_not_aggregates() -> None:
    for text in (
        "coalesce(orders.amount, 0)",
        "abs(orders.amount)",
        "round(orders.amount, 2)",
        "upper(customers.name)",
    ):
        expr = parse_expr_string(text)
        sc = SelectCol(expr=expr)
        assert sc.is_aggregated is False
        assert expr.add_groups
        assert expr.add_groups[0].agg_func is None
        assert expr.add_groups[0].scalar_func is not None


@pytest.mark.fast
def test_stddev_is_aggregate_but_not_cross_source_decomposable() -> None:
    assert "stddev" in AGGREGATE_FUNCTION_NAMES
    assert "stddev" not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS
    for name in ("variance", "median", "stddev_pop", "var_pop"):
        assert name in AGGREGATE_FUNCTION_NAMES
        assert name not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS


@pytest.mark.fast
def test_cross_source_stddev_via_parse_expr_string_is_ineligible() -> None:
    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    parsed = parse_expr_string("stddev(ta.id)")
    assert SelectCol(expr=parsed).is_aggregated is True
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="scalar",
        select_cols=[SelectCol(expr=parsed)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source aggregate not supported: stddev(ta.id)"


@pytest.mark.fast
def test_cross_source_median_via_parse_expr_string_is_ineligible() -> None:
    manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
        manifest,
    )
    parsed = parse_expr_string("median(ta.id)")
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="scalar",
        select_cols=[SelectCol(expr=parsed)],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source aggregate not supported: median(ta.id)"


@pytest.mark.fast
def test_single_engine_statistical_agg_infers_scalar_grain() -> None:
    raw = json.dumps(
        {
            "tables": ["orders"],
            "select_cols": [{"expr": "stddev(orders.amount)"}],
            "group_by_cols": [],
            "order_by_cols": [],
            "where": [],
            "having_param": [],
            "natural_language": "stddev of order amount",
        }
    )
    intent = parse_intent_response(raw, "stddev of order amount")
    assert intent is not None
    assert intent.select_cols[0].is_aggregated is True
    assert intent.grain == "scalar"
    assert intent.group_by_cols == []


@pytest.mark.fast
def test_single_engine_statistical_agg_with_group_by_infers_grouped_grain() -> None:
    raw = json.dumps(
        {
            "tables": ["orders"],
            "select_cols": [
                {"expr": "orders.customer_id"},
                {"expr": "stddev(orders.amount)"},
            ],
            "group_by_cols": ["orders.customer_id"],
            "order_by_cols": [],
            "where": [],
            "having_param": [],
            "natural_language": "stddev of amount per customer",
        }
    )
    intent = parse_intent_response(raw, "stddev of amount per customer")
    assert intent is not None
    assert intent.select_cols[1].is_aggregated is True
    assert intent.grain == "grouped"
    assert len(intent.group_by_cols) == 1
