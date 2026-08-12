"""Per-engine QSim skeleton cache isolation and schema-change clearing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata
from aetherdialect._qsim import (
    clear_engine_skeleton_cache,
    drop_engine_skeleton_cache_owner,
    engine_skeleton_cache,
    generate_all_skeletons,
    pop_qsim_engine_owner,
    push_qsim_engine_owner,
    register_engine_skeleton_cache_owner,
)


def _roles(schema: SchemaGraph) -> dict[str, str]:
    return {
        f"{tname}.{cname}": cmeta.role
        for tname, tmeta in schema.tables.items()
        for cname, cmeta in tmeta.columns.items()
    }


def _schema(schema_graph_id: str) -> SchemaGraph:
    columns = {
        "status": ColumnMetadata(
            name="status",
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
        )
    }
    table = TableMetadata(name="orders", columns=columns, foreign_keys=[], primary_key="")
    return SchemaGraph(
        join_paths_multi={},
        structural_hash=f"struct_{schema_graph_id}",
        effective_structural_hash=f"struct_{schema_graph_id}",
        schema_graph_id=schema_graph_id,
        tables={"orders": table},
    )


@pytest.mark.fast
def test_cache_is_per_engine() -> None:
    left = SimpleNamespace(name="left")
    right = SimpleNamespace(name="right")
    register_engine_skeleton_cache_owner(left)
    register_engine_skeleton_cache_owner(right)
    schema = _schema("sg_a")
    roles = _roles(schema)
    token_left = push_qsim_engine_owner(left)
    try:
        generate_all_skeletons(["orders"], schema, roles)
        left_keys = set(engine_skeleton_cache().keys())
    finally:
        pop_qsim_engine_owner(token_left)
    token_right = push_qsim_engine_owner(right)
    try:
        assert engine_skeleton_cache() == {}
        generate_all_skeletons(["orders"], schema, roles)
        right_keys = set(engine_skeleton_cache().keys())
    finally:
        pop_qsim_engine_owner(token_right)
    assert left_keys
    assert right_keys
    drop_engine_skeleton_cache_owner(left)
    drop_engine_skeleton_cache_owner(right)


@pytest.mark.fast
def test_cache_cleared_on_schema_change() -> None:
    owner = SimpleNamespace(name="engine")
    register_engine_skeleton_cache_owner(owner)
    token = push_qsim_engine_owner(owner)
    try:
        generate_all_skeletons(["orders"], _schema("sg_old"), _roles(_schema("sg_old")))
        assert engine_skeleton_cache()
        clear_engine_skeleton_cache(owner)
        assert engine_skeleton_cache() == {}
    finally:
        pop_qsim_engine_owner(token)
        drop_engine_skeleton_cache_owner(owner)


@pytest.mark.fast
def test_cache_not_capped() -> None:
    owner = SimpleNamespace(name="engine")
    register_engine_skeleton_cache_owner(owner)
    token = push_qsim_engine_owner(owner)
    try:
        for i in range(40):
            schema = _schema(f"sg_{i}")
            generate_all_skeletons(["orders"], schema, _roles(schema))
        assert len(engine_skeleton_cache()) >= 40
    finally:
        pop_qsim_engine_owner(token)
        drop_engine_skeleton_cache_owner(owner)
