"""Space notes classify subset respects deny_columns."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_denied_column_absent_from_classify_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = SchemaGraph(
        tables={
            "customer": TableMetadata(
                name="customer",
                columns={
                    "customer_id": ColumnMetadata(name="customer_id", data_type="int", value_type="integer"),
                    "email": ColumnMetadata(name="email", data_type="varchar", value_type="string"),
                },
                primary_key=["customer_id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )
    space = SpaceContext(deny_columns=frozenset({"customer.email"}))
    monkeypatch.setattr(
        MainExecutionOps,
        "validate_space_context_against_graph",
        lambda ctx, sg, federation_manifest=None: ctx,
    )
    subset = MainExecutionOps.build_subset_schema_for_space_notes(graph, space)
    assert "email" not in subset.tables["customer"].columns
    assert "customer_id" in subset.tables["customer"].columns
