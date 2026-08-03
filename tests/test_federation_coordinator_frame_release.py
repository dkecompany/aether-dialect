"""Coordinator frame release and byte-cap enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationCapExceededError
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._federation import (
    compose_composite_graph,
    execute_federation_coordinator,
    parse_federation_manifest,
    plan_federated_intent,
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
    "federation_id": "fed_release",
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
def test_byte_cap_exceeded_names_member_that_crossed_threshold() -> None:
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
        "a": pd.DataFrame({"id": [1]}),
        "b": pd.DataFrame({"id": [2], "payload": ["x" * 5000]}),
    }
    with pytest.raises(FederationCapExceededError, match="total input byte cap exceeded") as exc_info:
        execute_federation_coordinator(frames, plan, row_cap=100, total_input_byte_cap=200)
    assert exc_info.value.limit_key == "total_input_byte_cap"
    assert exc_info.value.source_id == "b"


@pytest.mark.fast
def test_coordinator_releases_member_frames_during_registration() -> None:
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
        "a": pd.DataFrame({"id": [1, 2]}),
        "b": pd.DataFrame({"id": [2]}),
    }
    execute_federation_coordinator(frames, plan, row_cap=100)
    assert frames == {}
