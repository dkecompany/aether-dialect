"""Federation member repair must not resolve against another member's schema objects."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr, SqlDiagnostic, SqlDiagnosticCode
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import member_schema_slice, parse_federation_manifest
from aetherdialect._intent_repair import apply_diagnostic_repairs, sanitize_table_names
from aetherdialect._schema_graph import recompute_join_paths_multi


def _lines_union_table() -> TableMetadata:
    return TableMetadata(
        name="lines",
        columns={
            "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
            "amt": ColumnMetadata(name="amt", data_type="numeric", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="",
        member_source_ids=["north", "south"],
        column_member_sources={"amount": ["north"], "amt": ["south"]},
    )


def _composite_lines_schema() -> SchemaGraph:
    tables = {"lines": _lines_union_table()}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.fast
def test_member_sub_intent_repair_resolves_same_named_table_to_own_member() -> None:
    """Unknown-column repair on a shared logical table respects the member source scope."""
    schema = _composite_lines_schema()
    runtime_intent = RuntimeIntent(
        tables=["lines"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("lines.amnt"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    diag = SqlDiagnostic(
        code=SqlDiagnosticCode.UNKNOWN_COLUMN,
        message="unknown column 'lines.amnt'",
        offending_identifier="lines.amnt",
        details={"source_id": "north"},
    )
    out, changed = apply_diagnostic_repairs(runtime_intent, schema, [diag])
    assert changed is True
    assert out.select_cols[0].expr.primary_column == "lines.amount"

    unscoped, unscoped_changed = apply_diagnostic_repairs(
        runtime_intent,
        schema,
        [
            SqlDiagnostic(
                code=SqlDiagnosticCode.UNKNOWN_COLUMN,
                message="unknown column 'lines.amnt'",
                offending_identifier="lines.amnt",
            )
        ],
    )
    assert unscoped_changed is True
    assert unscoped.select_cols[0].expr.primary_column == "lines.amt"


@pytest.mark.fast
def test_member_schema_slice_stamps_source_for_union_table_repairs() -> None:
    """Member schema slices carry a single source id so post-compose repairs stay local."""
    composite = _composite_lines_schema()
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_repair_scope",
            "sources": [
                {"source_id": "north", "engine": "duckdb", "role": "owner"},
                {"source_id": "south", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"lines": "north"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    north_schema = member_schema_slice(composite, "north", manifest=manifest)
    assert north_schema.tables["lines"].source_id == "north"

    intent = RuntimeIntent(
        tables=["FROM lines"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    sanitized = sanitize_table_names(intent, north_schema)
    assert sanitized.tables == ["lines"]
