"""Federation member artifact purge must target the bound directory and refuse unsafe paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from aetherdialect._constants import ARTIFACT_DIRECTORY_SEGMENT
from aetherdialect._contracts_base import FederationConfigError, FederationMappings
from aetherdialect._federation import (
    federation_source_artifacts_dir,
    persist_federation_tree,
    purge_federation_member_artifacts,
)
from tests.federation_helpers import build_two_member_federation


@pytest.mark.fast
def test_purge_removes_the_member_tree() -> None:
    bundle = build_two_member_federation(federation_id="fed_purge")
    manifest = bundle.manifest
    members = bundle.member_graphs
    composite = bundle.composite
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_root = str(Path(tmp) / "artifacts")
        fed_dir = os.path.join(artifacts_root, ARTIFACT_DIRECTORY_SEGMENT, "fed_fed_purge")
        persist_federation_tree(
            fed_dir,
            manifest=manifest,
            mappings=FederationMappings(version="0.2.1"),
            composite=composite,
            member_graphs=members,
        )
        binding = next(row for row in manifest.sources if row.source_id == "b")
        member_dir = federation_source_artifacts_dir(
            artifacts_root,
            binding,
            federation_id=manifest.federation_id,
        )
        os.makedirs(member_dir, exist_ok=True)
        marker = os.path.join(member_dir, "schema_graph.json.gz")
        with open(marker, "wb") as handle:
            handle.write(b"{}")
        assert os.path.isdir(member_dir)

        removed_path, bytes_reclaimed = purge_federation_member_artifacts(
            fed_dir,
            member_artifacts_dir=member_dir,
            artifacts_root=artifacts_root,
            source_id="b",
            manifest=manifest,
        )

        assert removed_path == os.path.abspath(member_dir)
        assert bytes_reclaimed > 0
        assert not os.path.exists(member_dir)
        assert os.path.isdir(fed_dir)


@pytest.mark.fast
def test_purge_refuses_unexpected_path() -> None:
    bundle = build_two_member_federation(federation_id="fed_purge_refuse")
    manifest = bundle.manifest
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_root = str(Path(tmp) / "artifacts")
        fed_dir = os.path.join(artifacts_root, ARTIFACT_DIRECTORY_SEGMENT, "fed_fed_purge_refuse")
        os.makedirs(fed_dir, exist_ok=True)
        rogue_dir = os.path.join(artifacts_root, ARTIFACT_DIRECTORY_SEGMENT, "conn_duckdb_main")
        os.makedirs(rogue_dir, exist_ok=True)

        with pytest.raises(FederationConfigError, match="refusing to purge"):
            purge_federation_member_artifacts(
                fed_dir,
                member_artifacts_dir=rogue_dir,
                artifacts_root=artifacts_root,
                source_id="b",
                manifest=manifest,
            )

        assert os.path.isdir(rogue_dir)
