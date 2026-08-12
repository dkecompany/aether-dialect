"""Structure edits record provenance and refuse replay after object recreation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import STRUCTURE_DOCUMENT_VERSION
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, RoleOwner, SchemaGraph, TableMetadata
from aetherdialect._schema_finalize import (
    apply_structure_to_graph,
    finalize_with_structure,
    load_structure_sidecar,
    reconfirm_override,
    save_schema_to_cache,
    save_structure_sidecar,
)
from aetherdialect._schema_graph import assign_schema_graph_hashes, recompute_join_paths_multi, table_structural_hash_fp
from aetherdialect._utils import reset_diagnostic_collector, set_diagnostic_collector


def _orders_table(*, role: str | None = None) -> TableMetadata:
    return TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                distinct_count=10,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                distinct_count=3,
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[],
        description="Catalog orders",
        role=role,
        row_count=10,
    )


def _graph_with_orders(table: TableMetadata) -> SchemaGraph:
    tables = {"orders": table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


@pytest.mark.fast
def test_recreated_object_does_not_inherit_overrides(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropped-and-recreated tables must not replay stale overrides without re-confirmation."""
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    cache_path = tmp_path / "schema_graph.json.gz"
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(cache_path))

    original_orders = _orders_table()
    original_graph = _graph_with_orders(original_orders)
    ctx = EngineContext()
    original_graph.notes_sha256 = ""
    assign_schema_graph_hashes(original_graph, ctx, "")
    save_schema_to_cache(original_graph, str(cache_path))

    apply_structure_to_graph(
        original_graph,
        {
            "version": STRUCTURE_DOCUMENT_VERSION,
            "tables": {"orders": {"role": "dimension"}},
        },
    )
    save_structure_sidecar(
        cache_path,
        {
            "tables": {
                "orders": {
                    "role": "dimension",
                    "authored_against_structural_hash": table_structural_hash_fp(original_orders),
                    "authored_at": "2026-01-01T00:00:00+00:00",
                },
            },
        },
        source_schema_hash=original_graph.effective_structural_hash,
        metadata_hash="meta",
    )

    previous_graph = _graph_with_orders(_orders_table())
    assign_schema_graph_hashes(previous_graph, ctx, "")
    # previous snapshot has no orders table (dropped)
    previous_graph.tables.clear()
    previous_graph.join_paths_multi = {}
    save_schema_to_cache(previous_graph, str(cache_path))

    recreated_orders = _orders_table(role="fact")
    recreated_orders.columns["amount"] = ColumnMetadata(
        name="amount",
        data_type="numeric",
        value_type="number",
        distinct_count=10,
    )
    recreated_graph = _graph_with_orders(recreated_orders)
    assign_schema_graph_hashes(recreated_graph, ctx, "")

    diags_buf: list[Any] = []
    token = set_diagnostic_collector(diags_buf)
    try:
        finalize_with_structure(recreated_graph, cache_path)
    finally:
        reset_diagnostic_collector(token)

    assert recreated_graph.tables["orders"].role == "fact"
    assert recreated_graph.tables["orders"].role_owner != RoleOwner.USER_OVERRIDE
    codes = {d.code for d in diags_buf}
    assert "STRUCTURE_NEEDS_RECONFIRMATION" in codes
    sidecar = load_structure_sidecar(cache_path)
    assert sidecar is not None
    entry = sidecar["tables"]["orders"]
    assert entry.get("needs_reconfirmation") is True


@pytest.mark.fast
def test_reconfirmation_reactivates(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit re-confirmation stamps the current hash and allows replay again."""
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    cache_path = tmp_path / "schema_graph.json.gz"
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(cache_path))

    original_orders = _orders_table()
    original_graph = _graph_with_orders(original_orders)
    ctx = EngineContext()
    original_graph.notes_sha256 = ""
    assign_schema_graph_hashes(original_graph, ctx, "")
    authored_hash = table_structural_hash_fp(original_orders)

    recreated_orders = _orders_table(role="fact")
    recreated_orders.columns["amount"] = ColumnMetadata(
        name="amount",
        data_type="numeric",
        value_type="number",
        distinct_count=10,
    )
    recreated_graph = _graph_with_orders(recreated_orders)
    assign_schema_graph_hashes(recreated_graph, ctx, "")

    save_structure_sidecar(
        cache_path,
        {
            "tables": {
                "orders": {
                    "role": "dimension",
                    "authored_against_structural_hash": authored_hash,
                    "authored_at": "2026-01-01T00:00:00+00:00",
                    "needs_reconfirmation": True,
                },
            },
        },
        source_schema_hash="stale",
        metadata_hash="stale",
    )
    cache_path.write_bytes(b"x")

    previous_graph = deepcopy(recreated_graph)
    previous_graph.tables.clear()
    previous_graph.join_paths_multi = {}

    assert reconfirm_override(cache_path, "tables.orders", recreated_graph) is True
    finalize_with_structure(recreated_graph, cache_path, previous_schema=previous_graph)

    assert recreated_graph.tables["orders"].role == "dimension"
    assert recreated_graph.tables["orders"].role_owner == RoleOwner.USER_OVERRIDE
    sidecar = load_structure_sidecar(cache_path)
    assert sidecar is not None
    entry = sidecar["tables"]["orders"]
    assert entry.get("needs_reconfirmation") is not True
    assert entry["authored_against_structural_hash"] == table_structural_hash_fp(recreated_orders)
