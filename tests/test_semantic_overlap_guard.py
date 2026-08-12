"""Numeric value overlap must not create semantic_join_neighbors."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import compute_semantic_profile_join_neighbors


@pytest.mark.fast
def test_numeric_overlap_does_not_create_neighbor() -> None:
    left = ColumnMetadata(
        name="amount_a",
        data_type="integer",
        value_type="integer",
        value_overlap_sample=[1, 2, 3, 4, 5],
        frequent_values=[1, 2, 3],
    )
    right = ColumnMetadata(
        name="amount_b",
        data_type="integer",
        value_type="integer",
        value_overlap_sample=[3, 4, 5, 6, 7],
        frequent_values=[3, 4, 5],
    )
    graph = SchemaGraph(
        tables={
            "t1": TableMetadata(name="t1", columns={"amount_a": left}, primary_key=[], foreign_keys=[]),
            "t2": TableMetadata(name="t2", columns={"amount_b": right}, primary_key=[], foreign_keys=[]),
        },
        join_paths_multi={},
    )
    compute_semantic_profile_join_neighbors(graph)
    assert left.semantic_join_neighbors == []
    assert right.semantic_join_neighbors == []
