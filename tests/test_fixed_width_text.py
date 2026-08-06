"""Tests for fixed-width CHAR/NCHAR column metadata and comparison trimming."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    WhereParam,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import telemetry_capture
from aetherdialect._dialect import DialectRegistry
from aetherdialect._schema_build import tables_meta_to_schema_graph
from aetherdialect._schema_catalog import _column_value_overlap_eligible
from aetherdialect._sql_gen import _render_predicate_clause


def _schema_with_char_column() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "codes": TableMetadata(
                name="codes",
                columns={
                    "status": ColumnMetadata(
                        name="status",
                        data_type="CHAR(5)",
                        value_type="string",
                        is_fixed_width_text=True,
                    )
                },
                primary_key=[],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )


@pytest.mark.fast
def test_char_comparison_trims_padding() -> None:
    """Comparing against a fixed-width column applies RTRIM on both sides and records the trace."""
    schema = _schema_with_char_column()
    pred = WhereParam(
        left_expr=NormalizedExpr.from_column("codes.status"),
        op="=",
        value_type="string",
        param_key="status_val",
    )
    dialect = DialectRegistry.get_class("postgresql").__new__(DialectRegistry.get_class("postgresql"))
    with telemetry_capture(force_diagnostic_flags=True) as logs:
        sql = _render_predicate_clause(pred, dialect, schema=schema, param_values={"status_val": "open"})
    assert "RTRIM(" in sql.upper()
    trace = "\n".join(logs)
    assert "fixed_width" in trace.lower() or "rtrim" in trace.lower()

    meta = {
        "codes": {
            "column_names_original": ["status", "label"],
            "column_types": ["CHAR(5)", "VARCHAR(50)"],
            "primary_keys": [],
            "foreign_keys": [],
        },
    }
    sg = tables_meta_to_schema_graph(meta, engine="postgresql")
    assert sg.tables["codes"].columns["status"].is_fixed_width_text is True
    assert sg.tables["codes"].columns["label"].is_fixed_width_text is False

    char_col = ColumnMetadata(
        name="status",
        data_type="CHAR(5)",
        value_type="string",
        distinct_count=3,
    )
    assert char_col.is_fixed_width_text is True
    assert _column_value_overlap_eligible(char_col) is False

    restored = ColumnMetadata.from_dict(char_col.to_dict())
    assert restored.is_fixed_width_text is True
