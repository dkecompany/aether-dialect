"""Init-time federation shrink must purge departed member artifact trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import ARTIFACT_DIRECTORY_SEGMENT, DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED
from aetherdialect._contracts_schema import FederationMappings
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    federation_source_artifacts_dir,
    persist_federation_tree,
)
from aetherdialect._federation_manifest import (
    binding_from_member_engine,
    parse_federation_manifest,
)
from aetherdialect._utils import drain_diagnostic_collector, reset_diagnostic_collector, set_diagnostic_collector
from tests.federation_helpers import write_federation_declaration_file
from tests.test_aether_federation_public_surface import _minimal_member
from tests.test_federation_member_removal import _three_member_manifest
from tests.test_migration_cache_drift import _member_graph, _mock_member

_TWO_MEMBER_DECL = {
    "federation_id": "fed_remove",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_departed_member_tree_removed_at_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_payload = _three_member_manifest()
    declaration_path = write_federation_declaration_file(tmp_path, _TWO_MEMBER_DECL, {"version": "0.2.3"})
    manifest = parse_federation_manifest(manifest_payload)
    members = {
        "a": _member_graph("left_t", source_id="a"),
        "b": _member_graph("right_t", source_id="b"),
        "c": _member_graph("extra_t", source_id="c"),
    }
    composite = compose_composite_graph(members, manifest)
    fed_dir = tmp_path / ARTIFACT_DIRECTORY_SEGMENT / "fed_fed_remove"
    persist_federation_tree(
        str(fed_dir),
        manifest=manifest,
        mappings=FederationMappings(version="0.2.3"),
        composite=composite,
        member_graphs=members,
    )
    binding_c = binding_from_member_engine(_minimal_member(connection="c"))
    member_tree = Path(
        federation_source_artifacts_dir(
            str(tmp_path),
            binding_c,
            federation_id=manifest.federation_id,
        )
    )
    member_tree.mkdir(parents=True, exist_ok=True)
    (member_tree / "schema_graph.json.gz").write_bytes(b"{}")

    member_a = _mock_member(members["a"], "a", tmp_path)
    member_b = _mock_member(members["b"], "b", tmp_path)

    buf: list = []
    token = set_diagnostic_collector(buf)
    try:
        AetherFederation(
            "fed_remove",
            members=(member_a, member_b),
            declaration=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert not member_tree.exists()
    removed = [d for d in diags if d.code == DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED and d.source_id == "c"]
    assert removed
    assert any(detail[0] == "bytes_reclaimed" and int(detail[1]) > 0 for detail in removed[0].details)
