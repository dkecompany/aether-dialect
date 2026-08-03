"""Federation migration drift gates, member graph reconciliation, and map validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import FEDERATION_MIGRATION_MAP_FILENAME
from aetherdialect._contracts_base import (
    ConfigError,
    LLMConfig,
    MigrationPendingError,
    RuntimeConfig,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config, write_gzip_json_atomic
from aetherdialect._federation import (
    FederationMappings,
    apply_federation_migration_map,
    clear_federation_composite_template_store,
    compute_federation_storage_dir,
    clear_federation_plan_templates,
    compose_composite_graph,
    compute_federation_storage_dir,
    detect_broken_cross_source_joins,
    federation_artifact_paths,
    federation_source_artifacts_dir,
    load_federation_plan_templates,
    parse_federation_manifest,
    parse_federation_migration_map,
    persist_federation_tree,
    reconcile_federation_member_graphs,
    save_federation_plan_template,
    validate_federation_migration_map,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import empty_template_store_for_space, save_template_store
from tests.federation_helpers import enriched_manifest, write_federation_declaration_file


def _member_table(name: str, *, source_id: str, extra_cols: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    columns = {
        "id": ColumnMetadata(
            name="id",
            data_type="integer",
            sensitivity="none",
            row_count=1,
            valid_where_ops=["=", "!=", "in", "not in", "is null", "is not null"],
        ),
        "email": ColumnMetadata(
            name="email",
            data_type="text",
            sensitivity="none",
            is_unique=True,
            row_count=1,
            valid_where_ops=["=", "!=", "in", "not in", "is null", "is not null"],
        ),
    }
    if extra_cols:
        columns.update(extra_cols)
    return TableMetadata(
        name=name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _member_graph(
    table: str,
    *,
    source_id: str,
    schema_graph_id: str | None = None,
    effective_structural_hash: str | None = None,
    extra_cols: dict[str, ColumnMetadata] | None = None,
) -> SchemaGraph:
    tables = {table: _member_table(table, source_id=source_id, extra_cols=extra_cols)}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=schema_graph_id or f"sg_{source_id}_{table}",
        effective_structural_hash=effective_structural_hash or f"eff_{source_id}_{table}",
        profiling_hash=f"profile_{source_id}",
    )


def _mock_member(graph: SchemaGraph, connection: str, artifacts_root: Path) -> MagicMock:
    llm_exec = load_runtime_config(merged_env={})
    member = MagicMock()
    member.dialect = "duckdb"
    member._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=str(artifacts_root / connection),
        engine_context=MagicMock(),
        llm_execution=llm_exec,
    )
    member._llm_config = LLMConfig(provider="openai")
    member._schema_graph = graph
    member._dialect = MagicMock()
    member._execution_engine = MagicMock()
    member._native_connection = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._engine_identity = None
    member._connection = connection
    return member


_MANIFEST = {
    "federation_id": "fed_gate",
    "cross_source_joins": [
        {
            "left": "entity_a.email",
            "right": "entity_b.email",
            "kind": "inner",
            "logical_key": "email",
        }
    ],
}


def _write_member_schema_artifact(
    artifacts_dir: Path,
    manifest: object,
    source_id: str,
    graph: SchemaGraph,
) -> None:
    binding = next(binding for binding in manifest.sources if binding.source_id == source_id)
    member_dir = federation_source_artifacts_dir(str(artifacts_dir), binding)
    os.makedirs(member_dir, exist_ok=True)
    write_gzip_json_atomic(
        os.path.join(member_dir, "schema_graph.json.gz"),
        graph.to_dict(),
        sort_keys=True,
    )


def _bootstrap_federation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, MagicMock], Path]:
    monkeypatch.chdir(tmp_path)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    mock_members = {sid: _mock_member(graph, sid, tmp_path) for sid, graph in members.items()}
    declaration_path = write_federation_declaration_file(tmp_path, _MANIFEST)
    AetherFederation(
        "fed_gate",
        members=mock_members,
        declaration_file=str(declaration_path),
        artifacts_dir=str(tmp_path),
    )
    return mock_members, declaration_path


@pytest.mark.fast
def test_federation_owner_member_drift_raises_before_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_gate")
    paths = federation_artifact_paths(fed_dir)

    drifted = _member_graph(
        "entity_a",
        source_id="alpha",
        schema_graph_id="sg_alpha_entity_a_drift",
        effective_structural_hash="eff_alpha_drift",
    )
    mock_members["alpha"]._schema_graph = drifted

    with pytest.raises(MigrationPendingError, match="Federation migration required"):
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )

    with open(paths["artifact_manifest"], encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["federation_members"][0][1] != "sg_alpha_entity_a_drift"


@pytest.mark.fast
def test_federation_consumer_member_drift_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    drifted = _member_graph(
        "entity_a",
        source_id="alpha",
        schema_graph_id="sg_alpha_entity_a_drift",
        effective_structural_hash="eff_alpha_drift",
    )
    mock_members["alpha"]._schema_graph = drifted

    with pytest.raises(ConfigError, match="Federation member graphs have drifted"):
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
            role="consumer",
        )


@pytest.mark.fast
def test_member_drift_invalidates_plan_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_gate")
    from aetherdialect._contracts_base import FederationPlanTemplate

    save_federation_plan_template(
        fed_dir,
        FederationPlanTemplate(
            plan_id="plan_gate",
            composite_schema_graph_id="sg_composite",
            intent_key="ik_gate",
            step_fingerprints=(("alpha", "ik_a"), ("beta", "ik_b")),
            combine_hash="combine_hash",
            question="join entities",
        ),
    )
    assert load_federation_plan_templates(fed_dir)

    drifted = _member_graph(
        "entity_a",
        source_id="alpha",
        schema_graph_id="sg_alpha_entity_a_drift",
        effective_structural_hash="eff_alpha_drift",
    )
    mock_members["alpha"]._schema_graph = drifted

    with pytest.raises(MigrationPendingError):
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )

    assert not load_federation_plan_templates(fed_dir)


@pytest.mark.fast
def test_federation_destructive_migration_clears_composite_templates(tmp_path: Path) -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    composite = compose_composite_graph(members, manifest)
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_gate")
    persist_federation_tree(
        fed_dir,
        manifest=manifest,
        mappings=FederationMappings(version=1),
        composite=composite,
        member_graphs=members,
    )
    store = empty_template_store_for_space(str(composite.schema_graph_id), artifacts_dir=fed_dir)
    save_template_store(store)
    assert (Path(fed_dir) / "intent_templates").is_dir()

    migration = parse_federation_migration_map({"version": 1, "action": "destructive"})
    apply_federation_migration_map(migration, manifest, FederationMappings(version=1), fed_dir)

    assert not (Path(fed_dir) / "intent_templates").exists()
    assert clear_federation_composite_template_store(fed_dir) is False


@pytest.mark.fast
def test_validate_federation_migration_map_stale_rename() -> None:
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    manifest = enriched_manifest(members, _MANIFEST, member_graphs=members)
    cached = dict(members)
    live = dict(members)
    migration = parse_federation_migration_map(
        {
            "version": 1,
            "action": "remap",
            "qualified_column_renames": [{"from": "entity_a.missing", "to": "entity_a.email"}],
        }
    )
    with pytest.raises(MigrationPendingError, match="STALE_MAP"):
        validate_federation_migration_map(
            migration,
            cached_member_graphs=cached,
            live_member_graphs=live,
            manifest=manifest,
        )


@pytest.mark.fast
def test_missing_cross_source_join_column_exports_remap_skeleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    broken_graph = _member_graph("entity_a", source_id="alpha", extra_cols={})
    broken_graph.tables["entity_a"].columns.pop("email")
    mock_members["alpha"]._schema_graph = broken_graph

    monkeypatch.chdir(tmp_path)
    with pytest.raises(MigrationPendingError, match="Federation migration required"):
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )

    skeleton_path = Path(compute_federation_storage_dir(str(tmp_path), "fed_gate")) / FEDERATION_MIGRATION_MAP_FILENAME
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    assert skeleton["dropped_cross_source_joins"] == [
        {"left": "entity_a.email", "right": "entity_b.email"},
    ]


@pytest.mark.fast
def test_detect_broken_cross_source_joins() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha", extra_cols={}),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    members["alpha"].tables["entity_a"].columns.pop("email")
    broken = detect_broken_cross_source_joins(members, manifest)
    assert broken == (("entity_a.email", "entity_b.email"),)


@pytest.mark.fast
def test_reconcile_federation_member_graphs_prefers_live() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    live = {
        "alpha": _member_graph("entity_a", source_id="alpha", schema_graph_id="sg_live_alpha"),
        "beta": _member_graph("entity_b", source_id="beta", schema_graph_id="sg_live_beta"),
    }
    disk = {
        "alpha": _member_graph("entity_a", source_id="", schema_graph_id="sg_disk_alpha"),
        "beta": _member_graph("entity_b", source_id="", schema_graph_id="sg_disk_beta"),
    }
    merged = reconcile_federation_member_graphs(live, disk, manifest)
    assert merged["alpha"].schema_graph_id == "sg_live_alpha"
    assert merged["beta"].schema_graph_id == "sg_live_beta"


@pytest.mark.fast
def test_reconcile_federation_member_graphs_stamps_disk_fallback() -> None:
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    manifest = enriched_manifest(members, _MANIFEST, member_graphs=members)
    disk = {
        "alpha": _member_graph("entity_a", source_id="", schema_graph_id="sg_disk_alpha"),
        "beta": _member_graph("entity_b", source_id="", schema_graph_id="sg_disk_beta"),
    }
    merged = reconcile_federation_member_graphs({}, disk, manifest)
    assert merged["alpha"].tables["entity_a"].source_id == "alpha"
    assert merged["beta"].tables["entity_b"].source_id == "beta"


@pytest.mark.fast
def test_federation_init_always_recomposes_even_with_cached_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with patch("aetherdialect._main_execution.compose_composite_graph", wraps=compose_composite_graph) as compose:
        mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
        assert compose.call_count >= 2


@pytest.mark.fast
def test_disk_member_graph_does_not_mask_live_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    manifest = enriched_manifest(members, _MANIFEST, member_graphs=members)
    stale_disk = _member_graph(
        "entity_a",
        source_id="alpha",
        schema_graph_id="sg_alpha_entity_a",
        effective_structural_hash="eff_alpha_entity_a",
    )
    _write_member_schema_artifact(tmp_path, manifest, "alpha", stale_disk)
    _write_member_schema_artifact(tmp_path, manifest, "beta", members["beta"])

    live_drift = _member_graph(
        "entity_a",
        source_id="alpha",
        schema_graph_id="sg_alpha_entity_a_drift",
        effective_structural_hash="eff_alpha_drift",
    )
    mock_members["alpha"]._schema_graph = live_drift

    with pytest.raises(MigrationPendingError, match="Federation migration required"):
        AetherFederation(
            "fed_gate",
            members=mock_members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )
