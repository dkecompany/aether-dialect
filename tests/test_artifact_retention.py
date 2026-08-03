"""Retention rules for prune_stale_artifact_auxiliaries."""

from __future__ import annotations

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
from aetherdialect._main_execution import prune_stale_artifact_auxiliaries


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

    prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

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

    prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

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

    prune_stale_artifact_auxiliaries(artifacts_dir, active_schema_graph_id="sg_active")

    assert not queue_path.exists()
