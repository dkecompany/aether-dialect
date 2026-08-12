"""Case-insensitive string comparison skips LOWER when collation is case-insensitive."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    WhereParam,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import _render_predicate_clause
from aetherdialect._utils import telemetry_capture


def _schema_with_ci_collation() -> SchemaGraph:
    tables = {
        "t": TableMetadata(
            name="t",
            columns={
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    is_case_insensitive_collation=True,
                )
            },
            primary_key=[],
            foreign_keys=[],
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.fast
def test_fold_skipped_on_case_insensitive_collation() -> None:
    schema = _schema_with_ci_collation()
    pred = WhereParam(
        left_expr=NormalizedExpr.from_column("t.name"),
        op="=",
        value_type="string",
        raw_value="Acme",
    )
    dialect = DialectRegistry.get_class("postgresql").__new__(DialectRegistry.get_class("postgresql"))
    with telemetry_capture(force_diagnostic_flags=True) as logs:
        sql = _render_predicate_clause(pred, dialect, schema=schema)
    assert "LOWER(" not in sql.upper()
    trace = "\n".join(logs)
    assert "case_fold" in trace
    assert "skipped" in trace.lower() or "case_insensitive_collation" in trace.lower()
