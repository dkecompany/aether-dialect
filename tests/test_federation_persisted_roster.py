"""Re-open federation inspection from a persisted fed_<id> roster without member engines."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import FederationConfigError, FederationMappings, PersistedFederationInspection
from aetherdialect._federation import (
    compute_federation_storage_dir,
    inspect_persisted_federation,
    persist_federation_tree,
)


@pytest.mark.fast
def test_inspect_persisted_federation_returns_declaration_and_roster() -> None:
    from tests.federation_helpers import build_two_member_federation

    bundle = build_two_member_federation(federation_id="fed_reopen")
    manifest = bundle.manifest
    members = bundle.member_graphs
    mappings = FederationMappings(version=1)
    composite = bundle.composite
    with tempfile.TemporaryDirectory() as tmp:
        fed_dir = compute_federation_storage_dir(tmp, "fed_reopen")
        persist_federation_tree(
            fed_dir,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        inspection = inspect_persisted_federation(tmp, "fed_reopen")
        assert isinstance(inspection, PersistedFederationInspection)
        assert inspection.federation_id == "fed_reopen"
        assert inspection.federation_dir == fed_dir
        assert inspection.manifest.federation_id == "fed_reopen"
        assert len(inspection.manifest.cross_source_joins) == 1
        assert inspection.mappings.version == 1
        assert len(inspection.roster) == 2
        assert all(len(row) == 4 for row in inspection.roster)


@pytest.mark.fast
def test_inspect_persisted_hydrates_sources_and_table_namespace() -> None:
    from tests.federation_helpers import build_two_member_federation

    bundle = build_two_member_federation(federation_id="fed_hydrate")
    with tempfile.TemporaryDirectory() as tmp:
        fed_dir = compute_federation_storage_dir(tmp, "fed_hydrate")
        persist_federation_tree(
            fed_dir,
            manifest=bundle.manifest,
            mappings=FederationMappings(version=1),
            composite=bundle.composite,
            member_graphs=bundle.member_graphs,
        )
        inspection = inspect_persisted_federation(tmp, "fed_hydrate")
        source_ids = {binding.source_id for binding in inspection.manifest.sources}
        assert source_ids == {"a", "b"}
        assert inspection.manifest.table_namespace.get("left_t") == "a"
        assert inspection.manifest.table_namespace.get("right_t") == "b"
        roster_ids = {row[0] for row in inspection.roster}
        assert roster_ids == source_ids


@pytest.mark.fast
def test_inspect_persisted_does_not_initialize_member_engines() -> None:
    from tests.federation_helpers import build_two_member_federation

    bundle = build_two_member_federation(federation_id="fed_no_members")
    with tempfile.TemporaryDirectory() as tmp:
        fed_dir = compute_federation_storage_dir(tmp, "fed_no_members")
        persist_federation_tree(
            fed_dir,
            manifest=bundle.manifest,
            mappings=FederationMappings(version=1),
            composite=bundle.composite,
            member_graphs=bundle.member_graphs,
        )
        with patch("aetherdialect.aetherdialect.initialize_aether_federation") as init:
            inspection = AetherFederation.inspect_persisted("fed_no_members", artifacts_dir=tmp)
        init.assert_not_called()
        assert inspection.manifest.sources


@pytest.mark.fast
def test_inspect_persisted_missing_tree_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FederationConfigError, match="not found"):
            inspect_persisted_federation(tmp, "fed_missing")


@pytest.mark.fast
def test_inspect_persisted_missing_roster_raises() -> None:
    import json

    from aetherdialect._federation import federation_artifact_paths
    from tests.federation_helpers import build_two_member_federation

    bundle = build_two_member_federation(federation_id="fed_no_roster")
    with tempfile.TemporaryDirectory() as tmp:
        fed_dir = compute_federation_storage_dir(tmp, "fed_no_roster")
        persist_federation_tree(
            fed_dir,
            manifest=bundle.manifest,
            mappings=FederationMappings(version=1),
            composite=bundle.composite,
            member_graphs=bundle.member_graphs,
        )
        manifest_path = federation_artifact_paths(fed_dir)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored.pop("federation_member_roster", None)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match="roster"):
            inspect_persisted_federation(tmp, "fed_no_roster")
