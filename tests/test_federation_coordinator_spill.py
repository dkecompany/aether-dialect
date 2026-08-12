"""Coordinator spill files are cleaned up after failure with external spill_dir."""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

from aetherdialect._contracts_base import FederationCapExceededError
from aetherdialect._contracts_core import CoordinatorMemberFrame, RuntimeIntent
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
    "federation_id": "fed_spill_l26",
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
def test_external_spill_dir_cleans_spill_files_after_coordinator_failure() -> None:
    """Spill parquet files created during a run are removed even when spill_dir is external."""
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
    spill_dir = tempfile.mkdtemp(prefix="aetherdialect_ext_spill_l26_")
    sentinel = os.path.join(spill_dir, "sentinel.txt")
    with open(sentinel, "w", encoding="utf-8") as handle:
        handle.write("keep")

    # Source a spills (> spill_threshold rows); source b triggers row-cap failure after registration.
    frames = {
        "a": CoordinatorMemberFrame(
            kind="arrow",
            table=pa.table({"id": [1, 2, 3]}),
            column_names=("id",),
        ),
        "b": CoordinatorMemberFrame(
            kind="arrow",
            table=pa.table({"id": list(range(10))}),
            column_names=("id",),
        ),
    }
    with pytest.raises(FederationCapExceededError, match="total input row cap exceeded"):
        execute_federation_coordinator(
            frames,
            plan,
            row_cap=5,
            spill_row_threshold=1,
            spill_dir=spill_dir,
            schema=composite,
        )

    spill_parquets = [name for name in os.listdir(spill_dir) if name.endswith(".parquet")]
    assert spill_parquets == []
    assert os.path.isfile(sentinel)
