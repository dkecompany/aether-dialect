"""Coordinator registration from Arrow member payloads."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._federation import (
    CoordinatorMemberFrame,
    compose_composite_graph,
    distinct_semijoin_keys,
    execute_federation_coordinator,
    parse_federation_manifest,
    plan_federated_intent,
    render_federation_glue,
)
from aetherdialect._schema_graph import ColumnMetadata, SchemaGraph, TableMetadata, recompute_join_paths_multi


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
    "federation_id": "fed_arrow",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_coordinator_joins_arrow_member_tables() -> None:
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
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "JOIN" in glue.upper()
    frames = {
        "a": CoordinatorMemberFrame(kind="arrow", table=pa.table({"id": [1, 2]}), column_names=("id",)),
        "b": CoordinatorMemberFrame(kind="arrow", table=pa.table({"id": [2, 3]}), column_names=("id",)),
    }
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1
    assert int(result.iloc[0]["id"]) == 2


@pytest.mark.fast
def test_coordinator_arrow_and_pandas_members_combine() -> None:
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
    plan = plan_federated_intent(intent, composite, manifest)
    frames = {
        "a": CoordinatorMemberFrame(kind="arrow", table=pa.table({"id": [1, 2]}), column_names=("id",)),
        "b": pd.DataFrame({"id": [2, 3]}),
    }
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1


@pytest.mark.fast
def test_distinct_semijoin_keys_reads_arrow_member_column() -> None:
    member = CoordinatorMemberFrame(
        kind="arrow",
        table=pa.table({"store_id": [1, 2, 2, None]}),
        column_names=("store_id",),
    )
    keys = distinct_semijoin_keys(member, "store_id", cap=10)
    assert sorted(keys or []) == [1, 2]
