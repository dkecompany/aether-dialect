"""Tests for federation artifact tree persistence and plan templates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aetherdialect._contracts_base import FederationMappings, FederationPlanTemplate
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    FederationConfigError,
    compose_composite_graph,
    compute_federation_storage_dir,
    federation_artifact_paths,
    federation_manifest_document,
    federation_plan_combine_hash,
    hydrate_persisted_federation_manifest,
    is_persisted_federation_manifest_sidecar,
    load_federation_composite_graph,
    load_federation_plan_templates,
    mappings_replay_matches,
    parse_federation_manifest,
    parse_federation_mappings,
    persist_federation_tree,
    save_federation_plan_template,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import enriched_manifest


def _graph(table: str, source_id: str = "") -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


_MANIFEST = {
    "federation_id": "fed_art",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _member_graphs() -> dict[str, SchemaGraph]:
    return {"a": _graph("left_t", "a"), "b": _graph("right_t", "b")}


def _full_manifest() -> object:
    members = _member_graphs()
    return enriched_manifest(members, _MANIFEST, member_graphs=members)


def test_compute_federation_storage_dir_uses_fed_prefix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = compute_federation_storage_dir(tmp, "crm_wh")
        assert path.endswith("fed_crm_wh")
        assert "aetherdialect" in path


def test_persist_and_load_composite_graph() -> None:
    manifest = _full_manifest()
    members = _member_graphs()
    composite = compose_composite_graph(members, manifest)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=FederationMappings(version=1),
            composite=composite,
            member_graphs=members,
        )
        paths = federation_artifact_paths(tmp)
        assert Path(paths["manifest"]).is_file()
        assert Path(paths["composite_schema"]).is_file()
        with open(paths["manifest"], encoding="utf-8") as handle:
            stored_manifest = json.load(handle)
        assert is_persisted_federation_manifest_sidecar(stored_manifest)
        assert "sources" not in stored_manifest
        assert stored_manifest["federation_id"] == "fed_art"
        assert stored_manifest["cross_source_joins"]
        hydrated = hydrate_persisted_federation_manifest(stored_manifest, manifest)
        assert len(hydrated.sources) == 2
        assert hydrated.table_namespace == manifest.table_namespace
        loaded = load_federation_composite_graph(tmp)
        assert loaded is not None
        assert "left_t" in loaded.tables
        assert loaded.tables["left_t"].source_id == "a"


def test_federation_manifest_document_include_derived() -> None:
    manifest = _full_manifest()
    minimal = federation_manifest_document(manifest)
    full = federation_manifest_document(manifest, include_derived=True)
    assert "sources" not in minimal
    assert "table_namespace" not in minimal
    assert len(full["sources"]) == 2
    assert full["table_namespace"]["left_t"] == "a"


def test_mappings_replay_matches_after_persist() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version=1)
    members = {"a": _graph("left_t"), "b": _graph("right_t")}
    composite = compose_composite_graph(members, manifest)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert mappings_replay_matches(tmp, members, manifest, mappings)


def test_federation_plan_template_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        template = FederationPlanTemplate(
            plan_id="plan_1",
            composite_schema_graph_id="sg_composite",
            intent_key="ik_1",
            step_fingerprints=(("a", "ik_a"), ("b", "ik_b")),
            combine_hash="combine_hash",
            question="join left and right",
        )
        save_federation_plan_template(tmp, template)
        loaded = load_federation_plan_templates(tmp)
        assert "plan_1" in loaded
        assert loaded["plan_1"].intent_key == "ik_1"
        assert loaded["plan_1"].step_fingerprints == (("a", "ik_a"), ("b", "ik_b"))


def test_artifact_manifest_records_member_tuple() -> None:
    manifest = _full_manifest()
    mappings = FederationMappings(version=1)
    members = _member_graphs()
    composite = compose_composite_graph(members, manifest)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        with open(federation_artifact_paths(tmp)["artifact_manifest"], encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload.get("federation_members"), list)
        assert len(payload["federation_members"]) == 2
        roster = payload.get("federation_member_roster")
        assert isinstance(roster, list)
        assert len(roster) == 2
        assert all(len(row) == 4 for row in roster)
        assert payload.get("ddl_probe_hash") == composite.ddl_probe_hash
        assert payload.get("schema_revision") == composite.schema_revision


@pytest.mark.fast
def test_load_composite_graph_validates_ddl_probe_and_schema_revision() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version=1)
    members = _member_graphs()
    composite = compose_composite_graph(members, manifest, mappings)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert load_federation_composite_graph(tmp) is not None
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["ddl_probe_hash"] = "stale_probe"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        assert load_federation_composite_graph(tmp) is None
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["ddl_probe_hash"] = composite.ddl_probe_hash
        stored["schema_revision"] = int(stored.get("schema_revision", 0)) + 1
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        assert load_federation_composite_graph(tmp) is None


def test_federation_plan_combine_hash_stable() -> None:
    from aetherdialect._contracts_core import FederatedPlan, JoinSpec

    plan = FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
    )
    assert federation_plan_combine_hash(plan) == federation_plan_combine_hash(plan)


def _collision_manifest(*, aliases: dict[str, dict[str, str]] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "federation_id": "fed_collision",
        "cross_source_joins": [],
    }
    if aliases:
        payload["aliases"] = aliases
    return payload


def test_compose_raises_on_table_name_collision() -> None:
    manifest = parse_federation_manifest(_collision_manifest())
    members = {"a": _graph("shared_t", "a"), "b": _graph("shared_t", "b")}
    with pytest.raises(FederationConfigError, match="table name collision"):
        compose_composite_graph(members, manifest)


def test_compose_lists_colliding_members_in_error() -> None:
    manifest = parse_federation_manifest(_collision_manifest())
    members = {"a": _graph("payment", "a"), "b": _graph("payment", "b")}
    with pytest.raises(FederationConfigError, match=r"a\.payment.*b\.payment"):
        compose_composite_graph(members, manifest)


def test_compose_allows_logical_tables_collision_resolution() -> None:
    manifest = parse_federation_manifest(_collision_manifest())
    members = {"a": _graph("payment", "a"), "b": _graph("payment", "b")}
    mappings = parse_federation_mappings(
        {
            "version": 1,
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    assert "payment" in composite.tables
    assert "a.payment" not in composite.tables
    assert "b.payment" not in composite.tables


def test_compose_allows_namespace_alias_for_colliding_tables() -> None:
    manifest = parse_federation_manifest(
        _collision_manifest(
            aliases={
                "payment_a": {"source": "a", "table": "payment"},
                "payment_b": {"source": "b", "table": "payment"},
            },
        ),
    )
    members = {"a": _graph("payment", "a"), "b": _graph("payment", "b")}
    composite = compose_composite_graph(members, manifest)
    assert "payment_a" in composite.tables
    assert "payment_b" in composite.tables
    assert composite.tables["payment_a"].source_id == "a"
    assert composite.tables["payment_b"].source_id == "b"


def test_compose_raises_when_alias_names_collide_across_members() -> None:
    manifest = parse_federation_manifest(
        _collision_manifest(
            aliases={
                "dup_alias": {"source": "a", "table": "dup_alias"},
            },
        ),
    )
    members = {"a": _graph("dup_alias", "a"), "b": _graph("dup_alias", "b")}
    with pytest.raises(FederationConfigError, match="dup_alias"):
        compose_composite_graph(members, manifest)


def _graph_with_notes(table: str, source_id: str, notes_sha256: str) -> SchemaGraph:
    base = _graph(table, source_id)
    return SchemaGraph(
        tables=base.tables,
        join_paths_multi=base.join_paths_multi,
        schema_graph_id=base.schema_graph_id,
        effective_structural_hash=base.effective_structural_hash,
        notes_sha256=notes_sha256,
    )


def test_federation_notes_content_changes_composite_identity() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {"a": _graph("left_t"), "b": _graph("right_t")}
    composite_v1 = compose_composite_graph(members, manifest, notes_content="fed-notes-v1")
    composite_v2 = compose_composite_graph(members, manifest, notes_content="fed-notes-v2")
    assert composite_v1.schema_graph_id
    assert composite_v2.schema_graph_id
    assert composite_v1.schema_graph_id != composite_v2.schema_graph_id


def test_member_notes_sha_changes_composite_identity() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members_a = {
        "a": _graph_with_notes("left_t", "a", "notes_sha_a"),
        "b": _graph_with_notes("right_t", "b", "notes_sha_b"),
    }
    members_b = {
        "a": _graph_with_notes("left_t", "a", "notes_sha_a_changed"),
        "b": _graph_with_notes("right_t", "b", "notes_sha_b"),
    }
    composite_a = compose_composite_graph(members_a, manifest)
    composite_b = compose_composite_graph(members_b, manifest)
    assert composite_a.schema_graph_id != composite_b.schema_graph_id


@pytest.mark.fast
def test_federation_artifact_previous_version_is_mismatch_not_missing() -> None:
    from aetherdialect._constants import FEDERATION_ARTIFACT_FORMAT_VERSION

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version=2)
    members = {"a": _graph("left_t", "a"), "b": _graph("right_t", "b")}
    with tempfile.TemporaryDirectory() as tmp:
        assert mappings_replay_matches(tmp, members, manifest, mappings) is False
        assert load_federation_composite_graph(tmp) is None

        composite = compose_composite_graph(members, manifest, mappings)
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert mappings_replay_matches(tmp, members, manifest, mappings) is True

        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        assert stored["artifact_format_version"] == FEDERATION_ARTIFACT_FORMAT_VERSION
        previous = FEDERATION_ARTIFACT_FORMAT_VERSION - 1
        stored["artifact_format_version"] = previous
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)

        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as exc_info:
            mappings_replay_matches(tmp, members, manifest, mappings)
        msg = str(exc_info.value)
        assert str(previous) in msg
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in msg
        assert "Delete" in msg

        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as load_exc:
            load_federation_composite_graph(tmp)
        load_msg = str(load_exc.value)
        assert str(previous) in load_msg
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in load_msg
