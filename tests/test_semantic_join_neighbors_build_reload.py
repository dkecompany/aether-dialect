"""Semantic join neighbours must match between fresh schema build and cache reload."""

from __future__ import annotations

import copy

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import (
    catalog_fk_graph_is_connected,
    recompute_join_paths_multi,
    run_fk_inference_if_disconnected,
)
from aetherdialect._schema_overrides import _ensure_semantic_join_neighbors
from aetherdialect._sql_gen import join_hints_multi
from aetherdialect._utils import stable_json

_CUSTOMER_SAMPLE = ["c1", "c2", "c3", "c4", "c5", "c6"]
_OVERLAP_SAMPLE = ["open", "closed", "pending", "done", "hold"]
_EXTENDED_SAMPLE = [*_OVERLAP_SAMPLE, "archived"]


def _col(name: str, **overrides) -> ColumnMetadata:
    defaults = dict(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=False,
        is_foreign_key=False,
        fk_target=None,
        distinct_count=10,
        row_count=100,
        null_ratio=0.0,
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _connected_orders_status_schema() -> SchemaGraph:
    customers = TableMetadata(
        name="customers",
        columns={"id": _col("id", is_primary_key=True, value_overlap_sample=_CUSTOMER_SAMPLE)},
        primary_key=["id"],
        foreign_keys=[],
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "id": _col("id", is_primary_key=True),
            "customer_id": _col("customer_id", value_overlap_sample=_CUSTOMER_SAMPLE),
            "status_code": _col("status_code", value_overlap_sample=_OVERLAP_SAMPLE),
        },
        primary_key=["id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            )
        ],
    )
    statuses = TableMetadata(
        name="statuses",
        columns={
            "code": _col("code", is_primary_key=True, value_overlap_sample=_EXTENDED_SAMPLE),
            "customer_id": _col("customer_id", value_overlap_sample=_CUSTOMER_SAMPLE),
        },
        primary_key=["code"],
        foreign_keys=[
            FKEdge(
                src_table="statuses",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            )
        ],
    )
    tables = {"customers": customers, "orders": orders, "statuses": statuses}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="semantic-build-reload",
    )


def _semantic_neighbors_snapshot(sg: SchemaGraph) -> dict[tuple[str, str], list[tuple[str, str]]]:
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for tbl_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            out[(tbl_name, col_name)] = list(col.semantic_join_neighbors)
    return out


def _join_candidate_snapshot(sg: SchemaGraph, scope: list[str]) -> str:
    hints = join_hints_multi(sg, scope, include_semantic=True)
    payload = [
        {
            "join_path_signature": c.get("join_path_signature"),
            "edge_kinds": c.get("edge_kinds"),
            "candidate_tier": c.get("candidate_tier"),
        }
        for c in hints.get("candidates", [])
    ]
    return stable_json(payload)


def _fresh_build_graph() -> SchemaGraph:
    sg = _connected_orders_status_schema()
    assert catalog_fk_graph_is_connected(sg)
    run_fk_inference_if_disconnected(sg)
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    return sg


def _reloaded_graph_from_cache(sg: SchemaGraph) -> SchemaGraph:
    cached = SchemaGraph.from_dict(sg.to_dict())
    _ensure_semantic_join_neighbors(cached)
    cached.join_paths_multi = recompute_join_paths_multi(cached.tables)
    return cached


@pytest.mark.fast
def test_connected_fresh_build_populates_semantic_neighbors_from_profile() -> None:
    sg = _fresh_build_graph()
    neighbors = _semantic_neighbors_snapshot(sg)
    assert neighbors[("orders", "status_code")] == [("statuses", "code")]
    assert neighbors[("statuses", "code")] == [("orders", "status_code")]


@pytest.mark.fast
def test_connected_fresh_build_and_cache_reload_agree_on_semantic_neighbors() -> None:
    built = _fresh_build_graph()
    reloaded = _reloaded_graph_from_cache(built)
    assert _semantic_neighbors_snapshot(built) == _semantic_neighbors_snapshot(reloaded)


@pytest.mark.fast
def test_connected_fresh_build_and_cache_reload_agree_on_join_candidates() -> None:
    built = _fresh_build_graph()
    reloaded = _reloaded_graph_from_cache(built)
    scope = ["orders", "statuses"]
    assert _join_candidate_snapshot(built, scope) == _join_candidate_snapshot(reloaded, scope)


@pytest.mark.fast
def test_connected_schema_round_trip_preserves_semantic_neighbors() -> None:
    built = _fresh_build_graph()
    round_tripped = SchemaGraph.from_dict(copy.deepcopy(built.to_dict()))
    _ensure_semantic_join_neighbors(round_tripped)
    assert _semantic_neighbors_snapshot(built) == _semantic_neighbors_snapshot(round_tripped)
