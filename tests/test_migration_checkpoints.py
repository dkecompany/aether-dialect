"""Orphaned migration checkpoints are collected or retained with diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aetherdialect._constants import (
    ARTIFACT_MANIFEST_FILENAME,
    MIGRATION_CHECKPOINT_PREFIX,
    TEMPLATE_STORE_SEGMENT,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import write_artifact_manifest


def _write_manifest(path: str, *, schema_graph_id: str, structural_hash: str = "s") -> None:
    payload = {
        "artifact_format_version": 12,
        "structural_hash": structural_hash,
        "profiling_hash": "p",
        "scope_hash": "sc",
        "effective_structural_hash": "eff",
        "schema_graph_id": schema_graph_id,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


@pytest.mark.fast
def test_completed_migration_checkpoint_collected(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
        schema_graph_id="sg_pre_migration",
    )
    checkpoint = tempfile.mkdtemp(prefix=MIGRATION_CHECKPOINT_PREFIX, dir=artifacts_dir)
    _write_manifest(
        os.path.join(checkpoint, ARTIFACT_MANIFEST_FILENAME),
        schema_graph_id="sg_pre_migration",
    )
    os.makedirs(os.path.join(checkpoint, TEMPLATE_STORE_SEGMENT), exist_ok=True)

    diags = TemplateOps.collect_orphaned_migration_checkpoints(artifacts_dir)

    assert not os.path.exists(checkpoint)
    assert not any(d.code == "MIGRATION_CHECKPOINT_ORPHANED" for d in diags)


@pytest.mark.fast
def test_ambiguous_checkpoint_retained_and_reported(tmp_path: Path) -> None:
    artifacts_dir = str(tmp_path)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s_live",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff_live",
        schema_graph_id="sg_after_partial_migration",
    )
    checkpoint = tempfile.mkdtemp(prefix=MIGRATION_CHECKPOINT_PREFIX, dir=artifacts_dir)
    _write_manifest(
        os.path.join(checkpoint, ARTIFACT_MANIFEST_FILENAME),
        schema_graph_id="sg_pre_migration",
        structural_hash="s_pre",
    )
    os.makedirs(os.path.join(checkpoint, TEMPLATE_STORE_SEGMENT), exist_ok=True)

    diags = TemplateOps.collect_orphaned_migration_checkpoints(artifacts_dir)

    assert os.path.isdir(checkpoint)
    orphan_diag = next((d for d in diags if d.code == "MIGRATION_CHECKPOINT_ORPHANED"), None)
    assert orphan_diag is not None
    assert checkpoint in orphan_diag.message or dict(orphan_diag.details).get("checkpoint_dir")
