"""Semantic-profile edges rendered as WHERE filters must emit a visibility diagnostic."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._sql_gen import _join_edges_from_signature, emit_semantic_profile_where_diagnostics
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)

_OVERLAP = ["open", "closed", "pending", "done", "hold"]


def _col(name: str, **overrides) -> ColumnMetadata:
    defaults = dict(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=False,
        value_overlap_sample=list(_OVERLAP),
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _semantic_pair_schema() -> SchemaGraph:
    orders = TableMetadata(
        name="orders",
        columns={
            "id": _col("id", is_primary_key=True, value_overlap_sample=[]),
            "status_code": _col("status_code"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    statuses = TableMetadata(
        name="statuses",
        columns={
            "code": _col("code", is_primary_key=True, value_overlap_sample=[*_OVERLAP, "archived"]),
        },
        primary_key=["code"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"orders": orders, "statuses": statuses},
        join_paths_multi={},
        effective_structural_hash="diag",
    )


def _fk_pair_schema() -> SchemaGraph:
    customers = TableMetadata(
        name="customers",
        columns={"id": _col("id", is_primary_key=True, value_overlap_sample=[])},
        primary_key=["id"],
        foreign_keys=[],
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "id": _col("id", is_primary_key=True, value_overlap_sample=[]),
            "customer_id": _col("customer_id", value_overlap_sample=[]),
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
    return SchemaGraph(
        tables={"customers": customers, "orders": orders},
        join_paths_multi={},
        effective_structural_hash="diag",
    )


@pytest.mark.fast
def test_rendered_semantic_profile_edge_emits_overlap_diagnostic() -> None:
    schema = _semantic_pair_schema()
    sig = ["orders.status_code->statuses.code"]
    kinds = ["semantic_profile"]
    token = set_diagnostic_collector([])
    try:
        result = _join_edges_from_signature(sig, kinds, "orders", schema)
        assert result is not None
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE
    assert "orders.status_code" in diag.message
    assert "statuses.code" in diag.message
    assert "overlap intersection 5" in diag.message
    assert "ratio 100%" in diag.message
    assert "foreign_keys_add" in diag.message


@pytest.mark.fast
def test_catalog_fk_edge_emits_no_semantic_profile_diagnostic() -> None:
    schema = _fk_pair_schema()
    sig = ["orders.customer_id->customers.id"]
    kinds = ["catalog_fk"]
    token = set_diagnostic_collector([])
    try:
        result = _join_edges_from_signature(sig, kinds, "orders", schema)
        assert result is not None
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert not any(d.code == DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE for d in diags)


@pytest.mark.fast
def test_semantic_profile_virtual_edge_emits_diagnostic() -> None:
    schema = _semantic_pair_schema()
    where_segments = [(0, "orders", "statuses", ["status_code"], ["code"])]
    token = set_diagnostic_collector([])
    try:
        emit_semantic_profile_where_diagnostics(
            schema,
            where_segments=where_segments,
            edge_kinds=["semantic_profile_virtual"],
        )
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE
