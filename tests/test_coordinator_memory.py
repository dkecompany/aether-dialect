"""Federation coordinator memory: single coordinator result fetch."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import execute_federation_coordinator
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, *, source_id: str) -> SchemaGraph:
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
    "federation_id": "fed_coordinator_memory",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _join_plan() -> object:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return plan_federated_intent(intent, composite, manifest)


@pytest.mark.fast
def test_result_fetched_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinator glue SQL is executed once; fan-out uses the held frame, not a second fetch."""
    from aetherdialect import _federation_execute

    plan = _join_plan()
    frames = {
        "a": pd.DataFrame({"id": [1, 2]}),
        "b": pd.DataFrame({"id": [2, 3]}),
    }
    call_count = 0
    real_execute = _federation_execute._execute_coordinator_sql_with_timeout

    def _counting_execute(
        conn: object,
        sql: str,
        bind_map: dict[str, object] | None,
        *,
        timeout_ms: int | None,
    ) -> object:
        nonlocal call_count
        call_count += 1
        return real_execute(conn, sql, bind_map or {}, timeout_ms=timeout_ms)

    monkeypatch.setattr(_federation_execute, "_execute_coordinator_sql_with_timeout", _counting_execute)
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1
    assert call_count == 1
