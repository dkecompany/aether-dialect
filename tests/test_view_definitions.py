"""Captured view definitions participate in structural drift detection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import tables_structural_payload
from aetherdialect._schema_reflect import (
    _reflect_view_definition_text,
    tables_referenced_in_view_definition,
)
from aetherdialect._utils import structural_hash_fp
from aetherdialect._validation_shape import validate_join_path_reachability_for_tables


def _column(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="INTEGER", is_nullable=True)


def _table(name: str, **overrides) -> TableMetadata:
    defaults = dict(
        name=name,
        columns={"id": _column("id")},
        primary_key=["id"],
        foreign_keys=[],
    )
    defaults.update(overrides)
    return TableMetadata(**defaults)


@pytest.mark.fast
def test_view_redefinition_is_structural_drift() -> None:
    """Different view definitions change the structural hash and are reflected from the engine."""
    old_view = _table("summary_v", kind="view", view_definition="SELECT id FROM orders")
    new_view = _table("summary_v", kind="view", view_definition="SELECT id FROM customers")
    old_hash = structural_hash_fp(tables_structural_payload({"summary_v": old_view}))
    new_hash = structural_hash_fp(tables_structural_payload({"summary_v": new_view}))
    assert old_hash != new_hash

    mock_insp = MagicMock()
    mock_insp.get_view_definition.return_value = "SELECT id FROM orders"
    assert _reflect_view_definition_text(mock_insp, "summary_v", "public") == "SELECT id FROM orders"
    assert tables_referenced_in_view_definition("SELECT id FROM orders") == frozenset({"orders"})

    orders = _table("orders")
    stores = _table("stores")
    summary = _table(
        "summary_v",
        kind="view",
        view_definition="SELECT orders.id FROM orders JOIN stores ON stores.id = orders.store_id",
    )
    schema = SchemaGraph(
        tables={"orders": orders, "stores": stores, "summary_v": summary},
        join_paths_multi={
            "orders": {"stores": [[{"src_table": "orders", "dst_table": "stores"}]]},
            "stores": {"orders": [[{"src_table": "stores", "dst_table": "orders"}]]},
        },
        effective_structural_hash="view-reachability",
    )
    issues = validate_join_path_reachability_for_tables(["summary_v", "stores"], schema, "main query")
    assert issues == []
