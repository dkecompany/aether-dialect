"""Warmup lattice and QSim skeleton caches must partition by space and visibility."""

from __future__ import annotations

import pytest

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata
from aetherdialect._qsim import (
    _skeleton_cache,
    generate_all_skeletons,
    pop_simulation_artifact_partition,
    push_simulation_artifact_partition,
    resolve_qsim_skeletons_path,
)
from aetherdialect._utils import (
    qsim_skeletons_filename,
    simulation_artifact_partition_fp,
    split_warmup_lattice_basename,
    warmup_lattice_filename,
)


def _orders_schema(*, schema_graph_id: str, structural_hash: str, column_names: tuple[str, ...]) -> SchemaGraph:
    columns = {
        name: ColumnMetadata(
            name=name,
            data_type="varchar",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
        )
        for name in column_names
    }
    orders = TableMetadata(name="orders", columns=columns, foreign_keys=[], primary_key="")
    return SchemaGraph(
        join_paths_multi={},
        structural_hash=structural_hash,
        effective_structural_hash=structural_hash,
        schema_graph_id=schema_graph_id,
        tables={"orders": orders},
    )


def _column_roles(schema: SchemaGraph) -> dict[str, str]:
    roles: dict[str, str] = {}
    for tname, tmeta in schema.tables.items():
        for cname, cmeta in tmeta.columns.items():
            roles[f"{tname}.{cname}"] = cmeta.role
    return roles


@pytest.mark.fast
def test_default_partition_fp_is_empty_for_owner_master() -> None:
    assert simulation_artifact_partition_fp() == ""
    assert simulation_artifact_partition_fp(scope_ctx=EngineContext()) == ""


@pytest.mark.fast
def test_partition_fp_differs_by_space_and_visibility() -> None:
    space_a = simulation_artifact_partition_fp(space_uid="aspace000000000001")
    space_b = simulation_artifact_partition_fp(space_uid="aspace000000000002")
    vis_a = simulation_artifact_partition_fp(visible_objects=frozenset({"orders"}))
    vis_b = simulation_artifact_partition_fp(visible_objects=frozenset({"orders", "customers"}))
    scoped = simulation_artifact_partition_fp(
        space_uid="aspace000000000001",
        visible_objects=frozenset({"orders"}),
    )
    assert space_a and space_b and vis_a and vis_b and scoped
    assert len({space_a, space_b, vis_a, vis_b, scoped}) == 5


@pytest.mark.fast
def test_warmup_lattice_filename_formats() -> None:
    graph_id = "sg_test000000000001__abc"
    fp = "a1b2c3d4e5f67890"
    assert warmup_lattice_filename(graph_id, "", "4") == f"lattice_{graph_id}_v4.json"
    assert warmup_lattice_filename(graph_id, fp, "4") == f"lattice_{graph_id}__{fp}_v4.json"
    parsed_id, parsed_fp, version = split_warmup_lattice_basename(warmup_lattice_filename(graph_id, fp, "4"))
    assert (parsed_id, parsed_fp, version) == (graph_id, fp, "4")
    parsed_default = split_warmup_lattice_basename(warmup_lattice_filename(graph_id, "", "4"))
    assert parsed_default == (graph_id, "", "4")


@pytest.mark.fast
def test_qsim_skeletons_filename_formats() -> None:
    assert qsim_skeletons_filename() == "qsim_skeletons.json.gz"
    assert qsim_skeletons_filename("deadbeefcafebabe") == "qsim_skeletons__deadbeefcafebabe.json.gz"


@pytest.mark.fast
def test_skeleton_cache_isolated_by_partition_fp() -> None:
    schema = _orders_schema(
        schema_graph_id="sgid_shared",
        structural_hash="struct_shared",
        column_names=("status", "category", "region"),
    )
    roles = _column_roles(schema)
    _skeleton_cache.clear()

    default_tok = push_simulation_artifact_partition("")
    default_result = generate_all_skeletons(["orders"], schema, roles)
    pop_simulation_artifact_partition(default_tok)

    scoped_tok = push_simulation_artifact_partition("partition_scope_a")
    scoped_result = generate_all_skeletons(["orders"], schema, roles)
    pop_simulation_artifact_partition(scoped_tok)

    assert max(s.num_where for s in default_result) == 4
    assert max(s.num_where for s in scoped_result) == 4
    assert default_result is not scoped_result
    assert len(_skeleton_cache) == 2


@pytest.mark.fast
def test_resolve_qsim_skeletons_path_uses_active_partition(tmp_path, monkeypatch) -> None:
    from aetherdialect._config import QSimConfig

    artifacts_dir = str(tmp_path)
    monkeypatch.setattr(QSimConfig, "SKELETONS_JSON_PATH", f"{artifacts_dir}/qsim_skeletons.json.gz")

    default_tok = push_simulation_artifact_partition("")
    assert resolve_qsim_skeletons_path().endswith("qsim_skeletons.json.gz")
    pop_simulation_artifact_partition(default_tok)

    scoped_tok = push_simulation_artifact_partition("abc123def4567890")
    assert resolve_qsim_skeletons_path().endswith("qsim_skeletons__abc123def4567890.json.gz")
    pop_simulation_artifact_partition(scoped_tok)


@pytest.mark.fast
def test_lattice_code_version_bumped() -> None:
    assert SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION == "4"
