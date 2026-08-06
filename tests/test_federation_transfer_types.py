"""Tests for coordinator transfer column typing from member schema metadata."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import _coordinator_relation_column_types_from_names
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph_with_amount(*, precision: int = 19, scale: int = 4) -> SchemaGraph:
    tables = {
        "orders": TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "amount": ColumnMetadata(
                    name="amount",
                    data_type=f"DECIMAL({precision},{scale})",
                    sensitivity="none",
                    numeric_precision=precision,
                    numeric_scale=scale,
                    is_exact_numeric=True,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_transfer_types",
        effective_structural_hash="eff_transfer_types",
    )


@pytest.mark.fast
def test_decimal_scale_preserved_on_transfer() -> None:
    schema = _graph_with_amount(precision=19, scale=4)
    column_types = _coordinator_relation_column_types_from_names(
        ("amount",),
        "a",
        schema=schema,
        plan=None,
    )
    assert column_types == [("amount", "DECIMAL(19, 4)")]
