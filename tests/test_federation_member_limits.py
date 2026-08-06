"""Per-member cost and profile limits are enforced on federation member execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import SqlDiagnostic, SqlDiagnosticCode
from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SourceStep
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest
from aetherdialect._schema_graph import ColumnMetadata, SchemaGraph, TableMetadata, recompute_join_paths_multi
from aetherdialect._validation_execute import execute_guarded_sql


def _graph(table: str, *, source_id: str = "a") -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_limits_l29",
    "sources": [
        {
            "source_id": "a",
            "engine": "duckdb",
            "role": "owner",
            "limits": {
                "timeout_ms": 12_345,
                "max_query_cost_rows": 100.0,
                "max_query_cost_bytes": 200.0,
                "profile_timeout_ms": 9_999,
            },
        },
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
    "coordinator": {"default_source_row_cap": 2000, "default_source_timeout_ms": 9000},
}


@pytest.mark.fast
def test_member_execute_passes_resolved_cost_and_profile_limits() -> None:
    from aetherdialect._pipeline import _execute_federation_source_step

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    step = SourceStep(
        source_id="a",
        sub_intent=RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
    )
    prepared_by_source = {
        "a": type("Prep", (), {"sub_intent": step.sub_intent, "sql": "SELECT 1", "structural_defaults": {}})(),
    }
    with patch("aetherdialect._pipeline.execute_guarded_sql") as exec_mock:
        exec_mock.return_value = [{"id": 1}]
        with patch("aetherdialect._pipeline.build_result_dataframe", return_value=pd.DataFrame({"id": [1]})):
            mock_dialect = MagicMock()
            mock_dialect.finalize_render.return_value = "SELECT 1"
            _execute_federation_source_step(
                step,
                prepared_by_source=prepared_by_source,
                composite_schema=composite,
                dialect_map={"a": mock_dialect},
                dialect=mock_dialect,
                manifest=manifest,
                executed={},
                plan=FederatedPlan(steps=(step,)),
                semijoin_cap=50_000,
                q_norm="",
                join_candidates=None,
                cmap=None,
                store=None,
                gate_kwargs={},
            )
    kwargs = exec_mock.call_args.kwargs
    assert kwargs["timeout_ms"] == 12_345
    assert kwargs["max_query_cost_rows"] == 100.0
    assert kwargs["max_query_cost_bytes"] == 200.0
    assert kwargs["profile_timeout_ms"] == 9_999


@pytest.mark.fast
def test_execute_guarded_sql_enforces_member_cost_cap_not_global() -> None:
    """Member cost cap must gate EXPLAIN estimates even when global policy allows more."""
    dialect = MagicMock()
    dialect.parse_select.return_value = "SELECT"
    dialect.ast_validate_full.return_value = []
    dialect.can_explain.return_value = True
    dialect.execute.return_value = [(1,)]

    def _explain_with_cost_gate(
        _sql: str,
        _params: dict[str, object] | None = None,
        *,
        schema: object | None = None,
        intent: object | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        from aetherdialect._dialect import Dialect

        failed, why = Dialect.explain_cost_gate_violation(500.0, None, dialect=dialect)
        if failed:
            return (
                False,
                [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                why,
            )
        return True, [], ""

    dialect.explain_diagnose.side_effect = _explain_with_cost_gate

    with pytest.raises(ValueError, match="cost gate"):
        execute_guarded_sql(dialect, "SELECT 1", max_query_cost_rows=100.0)

    assert PolicyConfig.MAX_QUERY_COST_ROWS is None or 500.0 < float(PolicyConfig.MAX_QUERY_COST_ROWS)
    result = execute_guarded_sql(dialect, "SELECT 1", max_query_cost_rows=1000.0)
    assert result == [(1,)]


@pytest.mark.fast
def test_execute_guarded_sql_enforces_member_cost_bytes_cap() -> None:
    dialect = MagicMock()
    dialect.parse_select.return_value = "SELECT"
    dialect.ast_validate_full.return_value = []
    dialect.can_explain.return_value = True
    dialect.execute.return_value = [(1,)]

    def _explain_with_bytes_gate(
        _sql: str,
        _params: dict[str, object] | None = None,
        *,
        schema: object | None = None,
        intent: object | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        from aetherdialect._dialect import Dialect

        failed, why = Dialect.explain_cost_gate_violation(1.0, 500.0, dialect=dialect)
        if failed:
            return (
                False,
                [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)],
                why,
            )
        return True, [], ""

    dialect.explain_diagnose.side_effect = _explain_with_bytes_gate

    with pytest.raises(ValueError, match="cost gate"):
        execute_guarded_sql(dialect, "SELECT 1", max_query_cost_bytes=100.0)

    result = execute_guarded_sql(dialect, "SELECT 1", max_query_cost_bytes=1000.0)
    assert result == [(1,)]


@pytest.mark.fast
def test_explain_diagnose_honors_profile_timeout_ms_override() -> None:
    dialect = MagicMock()
    dialect.parse_select.return_value = "SELECT"
    dialect.ast_validate_full.return_value = []
    dialect.can_explain.return_value = True
    dialect.execute.return_value = [(1,)]
    seen: list[int | None] = []

    def _capture_validate(d: object, *_args: object, **_kwargs: object) -> tuple[bool, None, None, list[SqlDiagnostic]]:
        seen.append(getattr(d, "profile_timeout_ms", None))
        return True, None, None, []

    with patch("aetherdialect._validation_execute.validate_sql", side_effect=_capture_validate):
        execute_guarded_sql(dialect, "SELECT 1", profile_timeout_ms=42)
    assert seen == [42]
