"""Live federation checks on one engine with multiple schemas (optional credentials)."""

from __future__ import annotations

import os

import pytest

from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest, plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi


def _table(name: str, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _graph(table: str, source_id: str) -> SchemaGraph:
    tables = {table: _table(table, source_id)}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.live
def test_two_schema_composite_plan() -> None:
    """Cross-schema decomposition on an in-memory DuckDB-style member pair."""
    manifest = parse_federation_manifest(
        {
            "federation_id": "live_two_schema",
            "sources": [
                {"source_id": "crm", "engine": "duckdb", "role": "owner"},
                {"source_id": "wh", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"crm_accounts": "crm", "wh_accounts": "wh"},
            "cross_source_joins": [
                {
                    "left": "crm_accounts.id",
                    "right": "wh_accounts.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
    )
    composite = compose_composite_graph(
        {"crm": _graph("crm_accounts", "crm"), "wh": _graph("wh_accounts", "wh")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["crm_accounts", "wh_accounts"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    assert plan.ineligible_reason is None


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("AETHERDIALECT_LIVE_POSTGRES_URL"),
    reason="set AETHERDIALECT_LIVE_POSTGRES_URL for postgres two-schema federation",
)
def test_postgres_two_schema_placeholder() -> None:
    """Reserved hook for a real Postgres two-schema federation run."""
    pytest.skip("postgres two-schema end-to-end wiring is environment-specific")
