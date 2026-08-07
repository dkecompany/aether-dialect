"""SchemaGraph stats/capability helpers must be per-instance."""

from __future__ import annotations

from typing import Any

import pytest

from aetherdialect._contracts_base import DatabaseFeatureCapability
from aetherdialect._contracts_schema import SchemaGraph


def _empty_graph() -> SchemaGraph:
    return SchemaGraph(tables={}, join_paths_multi={})


def _cap() -> DatabaseFeatureCapability:
    return DatabaseFeatureCapability(
        table_count=0,
        fk_edge_count=0,
        has_numeric_measures=False,
        has_date_columns=False,
        has_array_columns=False,
        has_categorical_columns=False,
        max_tables_on_any_join_path=0,
        max_fk_chain_depth=0,
        has_self_referential_fk=False,
        tables_supporting_self_join=frozenset(),
        has_window_capable_table_sets=False,
        aggregatable_columns_by_table={},
        date_columns_by_table={},
        array_columns_by_table={},
    )


@pytest.mark.fast
def test_two_graphs_independent_helpers() -> None:
    left = _empty_graph()
    right = _empty_graph()
    left_calls = {"n": 0}
    right_calls = {"n": 0}

    def left_stats(graph: SchemaGraph) -> dict[str, Any]:
        left_calls["n"] += 1
        return {"side": "left", "tables": len(graph.tables)}

    def right_stats(graph: SchemaGraph) -> dict[str, Any]:
        right_calls["n"] += 1
        return {"side": "right", "tables": len(graph.tables)}

    left.set_helpers(left_stats, lambda g: _cap())
    right.set_helpers(right_stats, lambda g: _cap())

    assert left.refresh_schema_stats()["side"] == "left"
    assert right.refresh_schema_stats()["side"] == "right"
    assert left_calls["n"] == 1
    assert right_calls["n"] == 1
    assert left.database_feature_capability is not None
    assert right.database_feature_capability is not None
