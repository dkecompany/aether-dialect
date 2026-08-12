"""Degenerate one-member federation execution must emit the same member instrumentation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPreparedStep,
    FederatedPrepareOutcome,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import federation_plan_is_degenerate
from aetherdialect._pipeline_execute import execute_federated_prepare


def _degenerate_prepared() -> tuple[FederatedPrepareOutcome, object]:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_degen_instr",
            "sources": [{"source_id": "a", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"left_t": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
    from aetherdialect._schema_graph import recompute_join_paths_multi

    tables = {
        "left_t": TableMetadata(
            name="left_t",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
    }
    member_graph = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    composite = compose_composite_graph({"a": member_graph}, manifest)
    sub_intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    assert federation_plan_is_degenerate(plan)
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="SELECT id FROM left_t",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=sub_intent,
                sql="SELECT id FROM left_t",
                structural_defaults={},
            ),
        ),
        composite_schema_graph_id=str(composite.schema_graph_id),
    )
    return prepared, composite


@pytest.mark.fast
def test_degenerate_execution_emits_member_executed_diagnostic() -> None:
    prepared, composite = _degenerate_prepared()
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT id FROM left_t"
    diagnostics: list[tuple[str, str]] = []

    def _capture_notify(*_args: object, **kwargs: object) -> None:
        code = kwargs.get("code")
        if code:
            diagnostics.append((str(code), str(kwargs.get("source_id", ""))))

    with (
        patch("aetherdialect._pipeline_execute.execute_guarded_sql", return_value=[(1,), (2,)]),
        patch("aetherdialect._pipeline_execute.notify", side_effect=_capture_notify),
    ):
        execute_federated_prepare(
            prepared,
            composite,
            dialect=dialect,
            dialects_by_source={"a": dialect},
        )
    codes = [code for code, _source in diagnostics]
    assert DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED in codes


@pytest.mark.fast
def test_degenerate_execution_sets_statement_duration_ms() -> None:
    prepared, composite = _degenerate_prepared()
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT id FROM left_t"
    with patch("aetherdialect._pipeline_execute.execute_guarded_sql", return_value=[(1,), (2,)]):
        outcome = execute_federated_prepare(
            prepared,
            composite,
            dialect=dialect,
            dialects_by_source={"a": dialect},
        )
    assert outcome.bundle is not None
    assert len(outcome.bundle.statements) == 1
    record = outcome.bundle.statements[0]
    assert record.duration_ms is not None
    assert record.duration_ms >= 0
