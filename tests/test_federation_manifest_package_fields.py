"""Federation artifact manifests must carry package version floor fields."""

from __future__ import annotations

import json
import tempfile

import pytest

from aetherdialect._constants import FEDERATION_ARTIFACT_FORMAT_VERSION, MIN_COMPATIBLE_PACKAGE_VERSION
from aetherdialect._contracts_schema import FederationMappings
from aetherdialect._federation_execute import persist_federation_tree
from aetherdialect._federation_manifest import (
    federation_artifact_manifest_view,
    federation_artifact_paths,
)
from aetherdialect._utils_artifacts import artifact_package_version_string
from tests.federation_helpers import build_two_member_federation


@pytest.mark.fast
def test_persist_writes_min_compatible_package_version() -> None:
    bundle = build_two_member_federation()
    mappings = FederationMappings(version="0.2.3")
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=bundle.manifest,
            mappings=mappings,
            composite=bundle.composite,
            member_graphs=bundle.member_graphs,
        )
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        assert stored["created_with_package_version"] == artifact_package_version_string()
        assert stored["min_compatible_package_version"] == MIN_COMPATIBLE_PACKAGE_VERSION

        view = federation_artifact_manifest_view(tmp)
        assert view is not None
        assert view.artifact_format_version == FEDERATION_ARTIFACT_FORMAT_VERSION
        assert view.created_with_package_version == artifact_package_version_string()
        assert view.min_compatible_package_version == MIN_COMPATIBLE_PACKAGE_VERSION
