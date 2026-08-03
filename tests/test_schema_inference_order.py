"""Schema inference and join enumeration must not depend on catalog iteration order."""

from __future__ import annotations

import copy

import pytest

from aetherdialect._contracts_base import ColumnRole, TableRole
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import _merge_column_metadata_union_statistics
from aetherdialect._schema_graph import (
    infer_missing_fks,
    promote_cross_component_semantic_edges,
    recompute_join_paths_multi,
)
from aetherdialect._sql_gen import _analyze_join_topology, join_hints_multi
from aetherdialect._utils import stable_json


def _col(**overrides) -> ColumnMetadata:
    defaults = dict(
        name="col",
        data_type="varchar",
        value_type="",
        is_primary_key=False,
        is_foreign_key=False,
        fk_target=None,
        role=ColumnRole.CATEGORICAL.value,
        distinct_count=10,
        distinct_ratio=0.5,
        row_count=20,
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _table(name: str, columns: dict[str, ColumnMetadata], **overrides) -> TableMetadata:
    defaults = dict(
        name=name,
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )
    defaults.update(overrides)
    return TableMetadata(**defaults)


def _suffix_ambiguity_tables() -> dict[str, TableMetadata]:
    return {
        "customer": _table(
            "customer",
            {"customer_id": _col(name="customer_id", is_primary_key=True)},
            primary_key=["customer_id"],
        ),
        "customers": _table(
            "customers",
            {"customers_id": _col(name="customers_id", is_primary_key=True)},
            primary_key=["customers_id"],
        ),
        "orders": _table(
            "orders",
            {
                "order_id": _col(name="order_id", is_primary_key=True),
                "customer_id": _col(name="customer_id"),
            },
            primary_key=["order_id"],
        ),
    }


def _composite_tables() -> dict[str, TableMetadata]:
    return {
        "order_lines": _table(
            "order_lines",
            {
                "order_id": _col(name="order_id", is_primary_key=True),
                "line_no": _col(name="line_no", is_primary_key=True),
                "qty": _col(name="qty"),
            },
            primary_key=["order_id", "line_no"],
        ),
        "shipments": _table(
            "shipments",
            {
                "shipment_id": _col(name="shipment_id", is_primary_key=True),
                "order_id": _col(name="order_id"),
                "line_no": _col(name="line_no"),
            },
            primary_key=["shipment_id"],
        ),
        "returns": _table(
            "returns",
            {
                "return_id": _col(name="return_id", is_primary_key=True),
                "order_id": _col(name="order_id"),
                "line_no": _col(name="line_no"),
            },
            primary_key=["return_id"],
        ),
    }


def _fk_snapshot(tables: dict[str, TableMetadata]) -> str:
    edges = infer_missing_fks(copy.deepcopy(tables))
    payload = [(e.src_table, tuple(e.src_cols), e.dst_table, tuple(e.dst_cols), str(e.inference_tag)) for e in edges]
    return stable_json(sorted(payload))


def _with_inferred_edges(tables: dict[str, TableMetadata]) -> dict[str, TableMetadata]:
    tables_copy = copy.deepcopy(tables)
    for edge in infer_missing_fks(tables_copy):
        tables_copy[edge.src_table].foreign_keys.append(edge)
    return tables_copy


def _join_paths_snapshot(tables: dict[str, TableMetadata]) -> str:
    wired = _with_inferred_edges(tables)
    return stable_json(recompute_join_paths_multi(wired))


def _candidate_snapshot(tables: dict[str, TableMetadata], scope: list[str]) -> str:
    wired = _with_inferred_edges(tables)
    graph = SchemaGraph(tables=wired, join_paths_multi=recompute_join_paths_multi(wired))
    hints = join_hints_multi(graph, scope)
    payload = [c.get("join_path_signature") for c in hints.get("candidates", [])]
    return stable_json(payload)


@pytest.mark.fast
def test_suffix_fk_inference_is_independent_of_table_dict_order() -> None:
    base = _suffix_ambiguity_tables()
    orderings = [
        ["customer", "customers", "orders"],
        ["orders", "customers", "customer"],
        ["customers", "orders", "customer"],
    ]
    snapshots = [_fk_snapshot({name: copy.deepcopy(base[name]) for name in order}) for order in orderings]
    assert snapshots[0] == snapshots[1] == snapshots[2]


@pytest.mark.fast
def test_composite_fk_inference_is_independent_of_table_dict_order() -> None:
    base = _composite_tables()
    orderings = [
        ["order_lines", "shipments", "returns"],
        ["returns", "shipments", "order_lines"],
        ["shipments", "order_lines", "returns"],
    ]
    snapshots = [_fk_snapshot({name: copy.deepcopy(base[name]) for name in order}) for order in orderings]
    assert snapshots[0] == snapshots[1] == snapshots[2]


@pytest.mark.fast
def test_join_paths_are_independent_of_table_dict_order() -> None:
    base = _suffix_ambiguity_tables()
    orderings = [
        ["customer", "customers", "orders"],
        ["orders", "customers", "customer"],
    ]
    snapshots = [_join_paths_snapshot({name: copy.deepcopy(base[name]) for name in order}) for order in orderings]
    assert snapshots[0] == snapshots[1]


@pytest.mark.fast
def test_join_candidates_are_independent_of_table_dict_order() -> None:
    base = _suffix_ambiguity_tables()
    orderings = [
        ["customer", "customers", "orders"],
        ["orders", "customers", "customer"],
    ]
    snapshots = [
        _candidate_snapshot({name: copy.deepcopy(base[name]) for name in order}, ["orders", "customers"])
        for order in orderings
    ]
    assert snapshots[0] == snapshots[1]


def _semantic_promotion_tables() -> dict[str, TableMetadata]:
    shared = ["open", "closed", "pending", "done", "hold"]
    orders = _table(
        "orders",
        {
            "link_code": _col(name="link_code", value_type="string", value_overlap_sample=shared),
        },
    )
    statuses = _table(
        "statuses",
        {
            "code": _col(
                name="code",
                value_type="string",
                is_primary_key=True,
                value_overlap_sample=[*shared, "archived"],
            )
        },
        primary_key=["code"],
    )
    regions = _table(
        "regions",
        {
            "code": _col(
                name="code",
                value_type="string",
                is_primary_key=True,
                value_overlap_sample=[*shared, "archived"],
            )
        },
        primary_key=["code"],
    )
    orders.columns["link_code"].distinct_count = 10
    statuses.columns["code"].distinct_count = 10
    regions.columns["code"].distinct_count = 10
    orders.columns["link_code"].semantic_join_neighbors = [("statuses", "code"), ("regions", "code")]
    statuses.columns["code"].semantic_join_neighbors = [("orders", "link_code")]
    regions.columns["code"].semantic_join_neighbors = [("orders", "link_code")]
    return {"orders": orders, "statuses": statuses, "regions": regions}


def _semantic_promotion_snapshot(tables: dict[str, TableMetadata]) -> str:
    wired = copy.deepcopy(tables)
    graph = SchemaGraph(tables=wired, join_paths_multi={}, created_at="")
    promote_cross_component_semantic_edges(graph)
    payload = [
        (
            edge.src_table,
            tuple(edge.src_cols),
            edge.dst_table,
            tuple(edge.dst_cols),
            str(edge.inference_tag),
        )
        for edge in wired["orders"].foreign_keys
    ]
    return stable_json(sorted(payload))


@pytest.mark.fast
def test_same_name_fk_inference_sorts_table_and_column_iteration() -> None:
    shared = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def bare_table(name: str, *, pk: bool = False) -> TableMetadata:
        col = _col(
            name="region_code",
            value_type="string",
            value_overlap_sample=shared,
        )
        table = TableMetadata.__new__(TableMetadata)
        table.name = name
        table.columns = {"region_code": col}
        table.primary_key = ["region_code"] if pk else []
        table.foreign_keys = []
        table.role = TableRole.DIMENSION.value
        table.row_count = 100
        return table

    base = {
        "staging_a": bare_table("staging_a"),
        "staging_b": bare_table("staging_b"),
        "dim_a": bare_table("dim_a", pk=True),
        "dim_b": bare_table("dim_b", pk=True),
    }
    orderings = [
        ["staging_a", "staging_b", "dim_a", "dim_b"],
        ["dim_b", "dim_a", "staging_b", "staging_a"],
    ]
    snapshots = [_fk_snapshot({name: copy.deepcopy(base[name]) for name in order}) for order in orderings]
    assert snapshots[0] == snapshots[1]
    assert snapshots[0] != "[]"


@pytest.mark.fast
def test_semantic_promotion_is_independent_of_neighbor_list_order() -> None:
    base = _semantic_promotion_tables()
    forward = copy.deepcopy(base)
    reverse = copy.deepcopy(base)
    forward["orders"].columns["link_code"].semantic_join_neighbors = [("statuses", "code"), ("regions", "code")]
    reverse["orders"].columns["link_code"].semantic_join_neighbors = [("regions", "code"), ("statuses", "code")]
    assert _semantic_promotion_snapshot(forward) == _semantic_promotion_snapshot(reverse)


@pytest.mark.fast
def test_semantic_promotion_is_independent_of_table_and_neighbor_order() -> None:
    base = _semantic_promotion_tables()
    orderings = [
        ["orders", "statuses", "regions"],
        ["regions", "statuses", "orders"],
        ["statuses", "orders", "regions"],
    ]
    snapshots = [
        _semantic_promotion_snapshot({name: copy.deepcopy(base[name]) for name in order}) for order in orderings
    ]
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert snapshots[0] != "[]"


@pytest.mark.fast
def test_federation_logical_column_merge_overlap_sample_is_order_independent() -> None:
    left = _col(name="email", value_overlap_sample=["zeta", "alpha", "mango"])
    right = _col(name="email", value_overlap_sample=["bravo", "alpha", "yankee"])
    merged_ab = _merge_column_metadata_union_statistics([left, right])
    merged_ba = _merge_column_metadata_union_statistics([right, left])
    assert merged_ab.value_overlap_sample == merged_ba.value_overlap_sample
    assert merged_ab.value_overlap_sample == sorted(
        {"zeta", "alpha", "mango", "bravo", "yankee"},
        key=str.lower,
    )


@pytest.mark.fast
def test_disconnected_linear_topology_uses_lexicographic_anchor_and_sorted_leaves() -> None:
    sig = ["a.x->b.x", "c.y->d.y"]
    topo, anchor, leaves = _analyze_join_topology(sig)
    assert topo == "linear"
    assert anchor == "a"
    assert leaves == ["a", "b", "c", "d"]
