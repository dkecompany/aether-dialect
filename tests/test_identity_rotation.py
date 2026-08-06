"""Identity rotation must collect every schema-graph-keyed artifact."""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import pytest

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._constants import (
    SCHEMA_CONTEXT_CACHE_NAME,
    TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX,
    TEMPLATE_STORE_FEEDBACK_SEGMENT,
    TEMPLATE_STORE_ORPHANED_SEGMENT,
    TEMPLATE_STORE_PARTITION_PREFIX,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._core_utils import write_artifact_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates import TemplateOps


def _template_store_dir(artifacts_dir: str) -> str:
    return TemplateOps.template_store_dir_for_space(artifacts_dir)


@pytest.mark.fast
def test_every_keyed_artifact_collected_after_rotation(tmp_path: Path) -> None:
    """Rotation moves template, feedback, warmup, skeleton, and context artifacts to orphaned/."""
    artifacts_dir = str(tmp_path)
    old_id = "sg_old000000000001__deadbeef"
    new_id = "sg_new000000000001__cafebabe"
    store_dir = _template_store_dir(artifacts_dir)

    os.makedirs(store_dir, exist_ok=True)
    template_shard = os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}00.json.gz")
    with gzip.open(template_shard, "wt", encoding="utf-8") as fh:
        fh.write("{}")
    with gzip.open(os.path.join(store_dir, "header.json.gz"), "wt", encoding="utf-8") as fh:
        json.dump({"schema_graph_id": old_id}, fh)

    feedback_dir = os.path.join(store_dir, TEMPLATE_STORE_FEEDBACK_SEGMENT)
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_shard = os.path.join(feedback_dir, f"{TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX}00.json.gz")
    with gzip.open(feedback_shard, "wt", encoding="utf-8") as fh:
        json.dump({"q1": ["join hint"]}, fh)

    lattice_dir = tmp_path / SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR
    lattice_dir.mkdir()
    stale_lattice = lattice_dir / f"lattice_{old_id}_v3.json"
    stale_lattice.write_text("{}", encoding="utf-8")
    active_lattice = lattice_dir / f"lattice_{new_id}_v3.json"
    active_lattice.write_text("{}", encoding="utf-8")

    skeleton_path = tmp_path / "qsim_skeletons.json.gz"
    with gzip.open(skeleton_path, "wt", encoding="utf-8") as fh:
        json.dump({"schema_graph_id": old_id, "skeletons": {}}, fh)

    context_path = tmp_path / SCHEMA_CONTEXT_CACHE_NAME
    context_path.write_text(json.dumps({"schema_graph_id": old_id}), encoding="utf-8")

    write_artifact_manifest(
        artifacts_dir,
        structural_hash="struct",
        profiling_hash="prof",
        scope_hash="scope",
        effective_structural_hash="eff",
        schema_graph_id=old_id,
    )
    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    queue_path.write_text('{"kind":"template_accept"}\n', encoding="utf-8")

    moved = TemplateOps.orphan_superseded_identity_artifacts(
        artifacts_dir,
        previous_schema_graph_id=old_id,
        active_schema_graph_id=new_id,
    )

    orphan_root = os.path.join(store_dir, TEMPLATE_STORE_ORPHANED_SEGMENT, old_id)
    stale_mtime = time.time() - (8 * 24 * 3600)
    os.utime(orphan_root, (stale_mtime, stale_mtime))
    top_orphan = tmp_path / TEMPLATE_STORE_ORPHANED_SEGMENT / old_id
    if top_orphan.is_dir():
        os.utime(top_orphan, (stale_mtime, stale_mtime))
    assert os.path.isdir(orphan_root)
    assert os.path.isfile(os.path.join(orphan_root, os.path.basename(template_shard)))
    assert os.path.isfile(os.path.join(orphan_root, "header.json.gz"))
    assert os.path.isfile(os.path.join(orphan_root, TEMPLATE_STORE_FEEDBACK_SEGMENT, os.path.basename(feedback_shard)))
    assert not os.path.isfile(template_shard)
    assert not os.path.isfile(feedback_shard)

    warmup_orphan = tmp_path / TEMPLATE_STORE_ORPHANED_SEGMENT / old_id / SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR
    assert warmup_orphan.is_dir()
    assert (warmup_orphan / stale_lattice.name).is_file()
    assert active_lattice.is_file()

    skeleton_orphan = tmp_path / TEMPLATE_STORE_ORPHANED_SEGMENT / old_id / "qsim_skeletons.json.gz"
    assert skeleton_orphan.is_file()
    assert not skeleton_path.exists()

    context_orphan = tmp_path / TEMPLATE_STORE_ORPHANED_SEGMENT / old_id / SCHEMA_CONTEXT_CACHE_NAME
    assert context_orphan.is_file()
    assert not context_path.exists()

    assert not queue_path.exists()
    assert "template_shards" in moved
    assert "feedback_shards" in moved
    assert "warmup_lattices" in moved
    assert "qsim_skeletons" in moved
    assert "schema_context_cache" in moved
    assert "write_queue" in moved

    removed, reclaimed = TemplateOps.collect_expired_template_orphans(artifacts_dir, now=time.time())
    assert removed >= 1
    assert reclaimed > 0

    MainExecutionOps.orphan_superseded_identity_artifacts_on_rotation(
        artifacts_dir,
        previous_schema_graph_id=old_id,
        active_schema_graph_id=new_id,
    )
