"""Federation member artifact directories are keyed by federation identity and source_id."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aetherdialect._constants import (
    ARTIFACT_DIRECTORY_SEGMENT,
    FEDERATION_SOURCE_STORAGE_PREFIX,
)
from aetherdialect._contracts_schema import FederationMappings, FederationSourceBinding
from aetherdialect._federation_execute import (
    _raise_federation_artifact_format_version_mismatch,
    compute_federation_storage_dir,
    detect_federation_member_engine_drift,
    federation_source_artifacts_dir,
    load_federation_member_manifest,
    persist_federation_tree,
    write_federation_member_manifest,
)
from aetherdialect._federation_manifest import (
    engine_connection_federation_source_storage_slug,
    federation_artifact_paths,
    federation_source_storage_slug,
)
from tests.federation_helpers import build_two_member_federation


def _duckdb_binding(source_id: str, *, connection: str = "") -> FederationSourceBinding:
    return FederationSourceBinding(
        source_id=source_id,
        engine="duckdb",
        connection=connection or source_id,
    )


@pytest.mark.fast
def test_replacing_engine_reuses_member_directory() -> None:
    fed_id = "fed_slug"
    binding_a = _duckdb_binding("storefront", connection="storefront_conn")
    binding_b = _duckdb_binding("catalog", connection="catalog_conn")
    slug_a = federation_source_storage_slug(binding_a, federation_id=fed_id)
    slug_b = federation_source_storage_slug(binding_b, federation_id=fed_id)
    assert slug_a != slug_b
    assert slug_a.startswith(FEDERATION_SOURCE_STORAGE_PREFIX)
    assert "storefront" in slug_a
    assert "catalog" in slug_b

    replaced = FederationSourceBinding(
        source_id="storefront",
        engine="postgresql",
        connection="storefront_pg",
    )
    assert federation_source_storage_slug(replaced, federation_id=fed_id) == slug_a


@pytest.mark.fast
def test_engine_change_detected_as_drift() -> None:
    bundle = build_two_member_federation(federation_id="fed_drift")
    manifest = bundle.manifest
    binding = next(row for row in manifest.sources if row.source_id == "a")
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_root = str(Path(tmp) / "artifacts")
        member_dir = federation_source_artifacts_dir(
            artifacts_root,
            binding,
            federation_id=manifest.federation_id,
        )
        os.makedirs(member_dir, exist_ok=True)
        write_federation_member_manifest(
            member_dir,
            binding,
            federation_id=manifest.federation_id,
        )
        stored = load_federation_member_manifest(member_dir)
        assert stored is not None
        assert stored["engine"] == "duckdb"

        live = FederationSourceBinding(source_id="a", engine="postgresql", connection="a")
        assert detect_federation_member_engine_drift(live, member_dir, federation_id=manifest.federation_id)

        same = FederationSourceBinding(source_id="a", engine="duckdb", connection="a")
        assert not detect_federation_member_engine_drift(same, member_dir, federation_id=manifest.federation_id)


@pytest.mark.fast
def test_v10_member_trees_migrate_to_source_id_directories() -> None:
    from aetherdialect._contracts_base import FederationConfigError

    bundle = build_two_member_federation(federation_id="fed_migrate")
    manifest = bundle.manifest
    members = bundle.member_graphs
    composite = bundle.composite
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_root = str(Path(tmp) / "artifacts")
        fed_dir = compute_federation_storage_dir(artifacts_root, manifest.federation_id)
        prior_binding = next(row for row in manifest.sources if row.source_id == "a")
        prior_slug = engine_connection_federation_source_storage_slug(prior_binding)
        prior_dir = os.path.join(artifacts_root, ARTIFACT_DIRECTORY_SEGMENT, prior_slug)
        os.makedirs(prior_dir, exist_ok=True)
        prior_marker = os.path.join(prior_dir, "schema_graph.json.gz")
        with open(prior_marker, "wb") as handle:
            handle.write(b"{}")

        persist_federation_tree(
            fed_dir,
            manifest=manifest,
            mappings=FederationMappings(version="0.2.3"),
            composite=composite,
            member_graphs=members,
        )
        manifest_path = federation_artifact_paths(fed_dir)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["artifact_format_version"] = 10
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)

        with pytest.raises(FederationConfigError, match=r"artifact_format_version"):
            _raise_federation_artifact_format_version_mismatch(stored, manifest_path, fed_dir)
