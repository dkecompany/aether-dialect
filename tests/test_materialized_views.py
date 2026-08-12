"""Materialized views are marked during reflection and surfaced at answer time."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_reflect import (
    _reflect_materialized_view_last_refreshed_at,
    emit_materialized_view_answer_diagnostics,
    tables_meta_to_schema_graph,
)
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


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
def test_kind_recorded_and_diagnostic_emitted() -> None:
    """Reflection records materialized views and execution emits MATERIALIZED_VIEW_ANSWER."""
    meta = {
        "sales_mv": {
            "column_names_original": ["id"],
            "column_types": ["INTEGER"],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "last_refreshed_at": "2026-01-15T10:00:00Z",
        }
    }
    graph = tables_meta_to_schema_graph(
        meta,
        row_kind_by_table={"sales_mv": "materialized_view"},
    )
    mv = graph.tables["sales_mv"]
    assert mv.kind == "materialized_view"
    assert mv.last_refreshed_at == "2026-01-15T10:00:00Z"

    mock_insp = MagicMock()
    mock_insp.get_materialized_view_last_refresh.return_value = "2026-01-15T10:00:00Z"
    assert _reflect_materialized_view_last_refreshed_at(mock_insp, "sales_mv", "public") == "2026-01-15T10:00:00Z"

    schema = SchemaGraph(
        tables={
            "sales_mv": _table(
                "sales_mv",
                kind="materialized_view",
                description="Sales snapshot",
                last_refreshed_at="2026-01-15T10:00:00Z",
            )
        },
        join_paths_multi={},
        effective_structural_hash="mv-test",
    )
    intent = RuntimeIntent(
        tables=["sales_mv"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    token = set_diagnostic_collector([])
    try:
        emit_materialized_view_answer_diagnostics(intent, schema)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER
    assert "Sales snapshot" in diags[0].message
    assert "2026-01-15T10:00:00Z" in diags[0].message
