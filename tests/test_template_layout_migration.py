"""Layout-only template store migration must not mask stale artifact format versions."""

from __future__ import annotations

import json
import os

import pytest

from aetherdialect._constants import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_MANIFEST_FILENAME,
    TEMPLATE_STORE_HEADER_FILENAME,
)
from aetherdialect._core_utils import write_gzip_json_atomic
from aetherdialect._templates import TemplateOps


def _write_stale_manifest(artifacts_dir: str, *, artifact_format_version: int) -> None:
    manifest_path = os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)
    payload = {
        "artifact_format_version": artifact_format_version,
        "created_with_package_version": "0.0.0",
        "min_compatible_package_version": "0.0.0",
        "last_action": "seed",
        "last_action_at": "2020-01-01T00:00:00+00:00",
        "structural_hash": "",
        "profiling_hash": "",
        "scope_hash": "",
        "effective_structural_hash": "",
        "schema_graph_id": "",
        "notes_hash": "",
        "semantic_edges_hash": "",
        "last_migration_tier": "",
        "last_migration_at": "",
        "last_corruption_at": "",
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


@pytest.mark.fast
def test_layout_helper_does_not_mask_incompatible_format(tmp_path) -> None:
    artifacts_dir = str(tmp_path)
    stale_version = "0.0.0"
    assert stale_version != ARTIFACT_FORMAT_VERSION
    _write_stale_manifest(artifacts_dir, artifact_format_version=stale_version)

    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir)
    os.makedirs(store_dir, exist_ok=True)
    graph_id = "sg_test000000000001__abcd1234"
    write_gzip_json_atomic(
        os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME),
        {
            "format_version": 4,
            "schema_graph_id": graph_id,
            "next_id": 1,
            "partition_map": {},
        },
        sort_keys=True,
    )

    TemplateOps.ensure_template_store_space_layout(artifacts_dir)

    with open(os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["artifact_format_version"] == stale_version
    assert manifest["artifact_format_version"] != ARTIFACT_FORMAT_VERSION
