"""A physical table may appear at most twice in one scope."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._validation_shape import validate_table_reference_counts


def _schema() -> SchemaGraph:
    tables = {
        "staff": TableMetadata(
            name="staff",
            columns={"staff_id": ColumnMetadata(name="staff_id", data_type="integer", is_primary_key=True)},
            primary_key=["staff_id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi={}, effective_structural_hash="staff-only")


@pytest.mark.fast
def test_table_referenced_three_times_refuses_at_intent() -> None:
    issues = validate_table_reference_counts(
        ["staff", "staff", "staff"],
        _schema(),
        "main query",
    )
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "staff" in issues[0].message
    assert "3 times" in issues[0].message


@pytest.mark.fast
def test_table_referenced_twice_is_allowed() -> None:
    issues = validate_table_reference_counts(
        ["staff", "staff"],
        _schema(),
        "main query",
    )
    assert issues == []


@pytest.mark.fast
def test_triple_reference_surfaces_through_semantic_validation() -> None:
    from aetherdialect._validation_sql import validate_semantics

    intent = RuntimeIntent(
        tables=["staff", "staff", "staff"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("staff.staff_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    result = validate_semantics(intent, _schema())
    errors = [i for i in result.issues if i.severity == "error"]
    assert any("staff" in e.message and "3 times" in e.message for e in errors)
