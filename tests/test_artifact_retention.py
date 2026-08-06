"""Retention rules for prune_stale_artifact_auxiliaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aetherdialect._constants import (
    MIGRATION_MAP_FILENAME,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SEGMENT,
    TEMPLATE_STORE_SPACES_SEGMENT,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._core_utils import write_artifact_manifest
from aetherdialect._main_execution import MainExecutionOps


def _template_shard_dir(artifacts_dir: str) -> str:
    return os.path.join(
        artifacts_dir,
        TEMPLATE_STORE_SEGMENT,
        TEMPLATE_STORE_SPACES_SEGMENT,
        "master",
    )


@pytest.mark.fast
def test_prune_removes_empty_template_shards(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    store_dir = _template_shard_dir(artifacts_dir)
    os.makedirs(store_dir, exist_ok=True)
    empty_shard = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}00.json.gz")
    with open(empty_shard, "wb"):
        pass
    nonempty_shard = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}01.json.gz")
    with open(nonempty_shard, "wb") as fh:
        fh.write(
            b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xabV\xca\xcfLQ\xb2\x32\xd0Q\xb2\x32\x04\x00\x00\x00\xff\xff\x03\x00\x00\x00\x00\x00\x00\x00\x00"
        )

    MainExecutionOps.prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

    assert not os.path.exists(empty_shard)
    assert os.path.isfile(nonempty_shard)


@pytest.mark.fast
def test_prune_keeps_latest_three_applied_map_archives(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    stem = Path(MIGRATION_MAP_FILENAME).stem
    archives = []
    for idx, ts in enumerate(("20240101T000000Z", "20240201T000000Z", "20240301T000000Z", "20240401T000000Z")):
        path = tmp_path / f"{stem}.applied.{ts}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (idx + 1, idx + 1))
        archives.append(path)

    MainExecutionOps.prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

    remaining = sorted(p.name for p in tmp_path.glob(f"{stem}.applied.*.json"))
    assert remaining == [
        "schema_migration_map.applied.20240201T000000Z.json",
        "schema_migration_map.applied.20240301T000000Z.json",
        "schema_migration_map.applied.20240401T000000Z.json",
    ]


@pytest.mark.fast
def test_prune_clears_write_queue_on_manifest_id_mismatch(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="es",
        schema_graph_id="sg_stale",
    )
    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    queue_path.write_text('{"event":"template_accept"}\n', encoding="utf-8")

    MainExecutionOps.prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

    assert not queue_path.exists()


@pytest.mark.fast
def test_prune_removes_stale_warmup_lattices(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    lattice_dir = tmp_path / "anchor_lattice"
    lattice_dir.mkdir()
    stale = lattice_dir / "lattice_sg_stale000000000001__dead_v3.json"
    active = lattice_dir / "lattice_sg_active000000000001__live_v3.json"
    stale.write_text("{}", encoding="utf-8")
    active.write_text("{}", encoding="utf-8")

    MainExecutionOps.prune_stale_artifact_auxiliaries(
        artifacts_dir, active_schema_graph_id="sg_active000000000001__live"
    )

    assert not stale.exists()
    assert active.is_file()


@pytest.mark.fast
def test_prune_removes_orphaned_nonempty_template_shards(tmp_path: Path) -> None:
    import gzip

    artifacts_dir = str(tmp_path)
    stale_dir = os.path.join(
        artifacts_dir,
        TEMPLATE_STORE_SEGMENT,
        TEMPLATE_STORE_SPACES_SEGMENT,
        "stale",
    )
    active_dir = os.path.join(
        artifacts_dir,
        TEMPLATE_STORE_SEGMENT,
        TEMPLATE_STORE_SPACES_SEGMENT,
        "active",
    )
    os.makedirs(stale_dir, exist_ok=True)
    os.makedirs(active_dir, exist_ok=True)
    stale_shard = os.path.join(stale_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}02.json.gz")
    active_shard = os.path.join(active_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}03.json.gz")
    payload = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xabV\xca\xcfLQ\xb2\x32\xd0Q\xb2\x32\x04\x00\x00\x00\xff\xff\x03\x00\x00\x00\x00\x00\x00\x00\x00"
    with open(stale_shard, "wb") as fh:
        fh.write(payload)
    with open(active_shard, "wb") as fh:
        fh.write(payload)
    with gzip.open(os.path.join(stale_dir, "header.json.gz"), "wt", encoding="utf-8") as fh:
        json.dump({"schema_graph_id": "sg_stale000000000001__dead"}, fh)
    with gzip.open(os.path.join(active_dir, "header.json.gz"), "wt", encoding="utf-8") as fh:
        json.dump({"schema_graph_id": "sg_active000000000001__live"}, fh)

    MainExecutionOps.prune_stale_artifact_auxiliaries(
        artifacts_dir, active_schema_graph_id="sg_active000000000001__live"
    )

    assert not os.path.exists(stale_shard)
    assert not os.path.exists(os.path.join(stale_dir, "header.json.gz"))
    assert os.path.isfile(active_shard)


@pytest.mark.fast
def test_prune_removes_orphaned_federation_trees(tmp_path: Path) -> None:
    from aetherdialect._constants import (
        ARTIFACT_DIRECTORY_SEGMENT,
        FEDERATION_COMPOSITE_SCHEMA_FILENAME,
        FEDERATION_STORAGE_PREFIX,
    )
    from aetherdialect._federation import compute_federation_storage_dir

    root = str(tmp_path)
    active = compute_federation_storage_dir(root, "crm", tenant_slug="tenant-a")
    orphan = os.path.join(
        os.path.abspath(root),
        ARTIFACT_DIRECTORY_SEGMENT,
        "tenant-a",
        f"{FEDERATION_STORAGE_PREFIX}legacy",
    )
    os.makedirs(orphan, exist_ok=True)
    os.makedirs(active, exist_ok=True)
    (Path(active) / FEDERATION_COMPOSITE_SCHEMA_FILENAME).write_bytes(b"x")

    from aetherdialect._main_execution import MainExecutionOps

    MainExecutionOps._prune_orphaned_federation_trees(os.path.dirname(active), active_fed_dir=active)

    assert os.path.isdir(active)
    assert not os.path.exists(orphan)
