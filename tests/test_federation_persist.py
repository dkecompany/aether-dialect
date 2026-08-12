"""Federation persistence must keep the applied mappings sidecar consistent."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aetherdialect._contracts_base import FederationMappingsAppliedSidecarError
from aetherdialect._contracts_schema import FederationMappings
from aetherdialect._federation_execute import (
    mappings_replay_matches,
    persist_federation_tree,
    validate_federation_mappings_applied_sidecar,
)
from aetherdialect._federation_manifest import (
    federation_artifact_paths,
    load_federation_mappings_from_path,
)
from tests.federation_helpers import build_two_member_federation

_MANIFEST = {
    "federation_id": "fed_sidecar",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_applied_sidecar_rewritten_on_persist() -> None:
    bundle = build_two_member_federation(federation_id="fed_sidecar")
    manifest = bundle.manifest
    members = bundle.member_graphs
    mappings = FederationMappings(version="0.2.3")
    composite = bundle.composite
    with tempfile.TemporaryDirectory() as tmp:
        paths = federation_artifact_paths(tmp)
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        applied_path = paths["mappings_applied"]
        assert os.path.isfile(applied_path)
        applied = json.loads(open(applied_path, encoding="utf-8").read())
        live = json.loads(open(paths["mappings"], encoding="utf-8").read())
        assert applied["version"] == live["version"]
        assert applied["logical_columns"] == live["logical_columns"]
        assert applied["logical_tables"] == live["logical_tables"]

        updated = FederationMappings(version="0.2.3", logical_columns=mappings.logical_columns, logical_tables=())
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=updated,
            composite=composite,
            member_graphs=members,
        )
        applied_after = json.loads(open(applied_path, encoding="utf-8").read())
        assert applied_after["logical_tables"] == []


@pytest.mark.fast
def test_inconsistent_sidecar_refused() -> None:
    bundle = build_two_member_federation(federation_id="fed_sidecar_bad")
    manifest = bundle.manifest
    members = bundle.member_graphs
    mappings = FederationMappings(version="0.2.3")
    composite = bundle.composite
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        paths = federation_artifact_paths(tmp)
        applied_path = paths["mappings_applied"]
        applied = json.loads(open(applied_path, encoding="utf-8").read())
        applied["logical_columns"] = [
            {
                "logical": "ghost.id",
                "members": ["left_t.id"],
                "role": "key",
                "unify_in_graph": True,
            }
        ]
        with open(applied_path, "w", encoding="utf-8") as handle:
            json.dump(applied, handle)

        live_mappings = load_federation_mappings_from_path(paths["mappings"])
        with pytest.raises(FederationMappingsAppliedSidecarError, match="ghost.id"):
            validate_federation_mappings_applied_sidecar(tmp, live_mappings)

        with pytest.raises(FederationMappingsAppliedSidecarError):
            mappings_replay_matches(tmp, members, manifest, live_mappings)
