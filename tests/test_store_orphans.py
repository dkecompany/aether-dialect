"""Template store shards orphaned on identity mismatch are collected, not abandoned."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import (
    TEMPLATE_STORE_ORPHANED_SEGMENT,
    TEMPLATE_STORE_PARTITION_PREFIX,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import reset_diagnostic_collector, set_diagnostic_collector
from aetherdialect._utils_artifacts import write_artifact_manifest
from tests.test_templates import _minimal_template


@pytest.mark.fast
def test_mismatched_shards_moved_not_abandoned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    artifacts_dir = str(tmp_path)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", store_dir)
    old_id = "sg_old000000000001__deadbeef"
    new_id = "sg_new000000000001__cafebabe"
    tmpl = replace(_minimal_template(), schema_graph_id=old_id, effective_structural_hash=old_id)
    store = TemplateOps.empty_template_store_for_space(old_id, artifacts_dir=artifacts_dir)
    TemplateOps.templates_to_store(store, {tmpl.id: tmpl})
    TemplateOps.save_template_store(store)
    shard_files = [
        name
        for name in os.listdir(store_dir)
        if name.startswith(TEMPLATE_STORE_PARTITION_PREFIX) and name.endswith(".json.gz")
    ]
    assert shard_files
    shard_name = shard_files[0]
    shard_path = os.path.join(store_dir, shard_name)
    assert os.path.isfile(shard_path)

    diags_buf: list = []
    token = set_diagnostic_collector(diags_buf)
    try:
        loaded = TemplateOps.load_template_store(new_id, schema=None, artifacts_dir=artifacts_dir)
    finally:
        reset_diagnostic_collector(token)

    orphan_dir = os.path.join(store_dir, TEMPLATE_STORE_ORPHANED_SEGMENT, old_id)
    assert os.path.isfile(os.path.join(orphan_dir, shard_name))
    assert not os.path.isfile(shard_path)
    assert loaded.schema_graph_id == new_id
    assert tmpl.id not in loaded.partition_map
    codes = {d.code for d in diags_buf}
    assert "TEMPLATE_STORE_ORPHANED" in codes
    orphan_diag = next(d for d in diags_buf if d.code == "TEMPLATE_STORE_ORPHANED")
    detail_map = dict(orphan_diag.details)
    assert detail_map.get("old_schema_graph_id") == old_id
    assert detail_map.get("new_schema_graph_id") == new_id


@pytest.mark.fast
def test_refresh_collects_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    artifacts_dir = str(tmp_path)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir)
    old_id = "sg_stale000000000001__dead"
    orphan_root = os.path.join(store_dir, TEMPLATE_STORE_ORPHANED_SEGMENT, old_id)
    os.makedirs(orphan_root, exist_ok=True)
    payload_path = os.path.join(orphan_root, f"{TEMPLATE_STORE_PARTITION_PREFIX}01.json.gz")
    payload = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xabV\xca\xcfLQ\xb2\x32\xd0Q\xb2\x32\x04\x00\x00\x00\xff\xff\x03\x00\x00\x00\x00\x00\x00\x00\x00"
    with open(payload_path, "wb") as fh:
        fh.write(payload)
    stale_mtime = time.time() - (8 * 24 * 3600)
    os.utime(orphan_root, (stale_mtime, stale_mtime))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
        schema_graph_id="sg_active000000000001__live",
    )

    removed, reclaimed = TemplateOps.collect_expired_template_orphans(artifacts_dir, now=time.time())

    assert removed == 1
    assert reclaimed >= len(payload)
    assert not os.path.exists(orphan_root)
