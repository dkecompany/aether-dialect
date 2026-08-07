"""Post-map cache bypass and federation composite drift gating."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation, _main_execution
from aetherdialect._constants import FEDERATION_MIGRATION_MAP_FILENAME, MIGRATION_MAP_ACTION_REMAP
from aetherdialect._contracts_base import (
    EngineContext,
    FederationContext,
    LLMConfig,
    MigrationPendingError,
    RuntimeConfig,
    SchemaMigrationMap,
    SchemaMigrationMapEntry,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config, write_artifact_manifest
from aetherdialect._federation import (
    compute_federation_storage_dir,
    federation_artifact_paths,
    load_federation_composite_graph,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_overrides import save_schema_to_cache
from aetherdialect._templates import TemplateOps
from tests.federation_helpers import write_federation_declaration_file


def _member_table(name: str, *, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={
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
        },
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
) -> SchemaGraph:
    tables = {table: _member_table(table, source_id=source_id)}
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
def test_post_map_rebuild_bypasses_schema_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
        return ColumnMetadata(name=name, data_type="integer", is_primary_key=pk)

    def _table(name: str) -> TableMetadata:
        return TableMetadata(
            name=name,
            columns={"id": _col("id", pk=True), "name": _col("name")},
            primary_key=["id"],
            foreign_keys=[],
        )

    owner_tables = {"items": _table("items")}
    owner = SchemaGraph(
        tables=owner_tables,
        join_paths_multi=recompute_join_paths_multi(owner_tables),
        schema_graph_id="sg_post_map_cache",
        structural_hash="owner_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="owner_eff",
    )
    live_tables = {"products": _table("products")}
    live = SchemaGraph(
        tables=live_tables,
        join_paths_multi=recompute_join_paths_multi(live_tables),
        schema_graph_id="sg_post_map_cache",
        structural_hash="live_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="live_eff",
    )
    artifacts_dir = str(tmp_path)
    schema_path = tmp_path / "schema_graph.json.gz"
    save_schema_to_cache(owner, str(schema_path))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=owner.structural_hash,
        profiling_hash=owner.profiling_hash,
        scope_hash=owner.scope_hash,
        effective_structural_hash=owner.effective_structural_hash,
        schema_graph_id=owner.schema_graph_id,
    )
    MainExecutionOps.write_schema_context_cache(artifacts_dir, EngineContext())
    map_obj = SchemaMigrationMap(
        version=1,
        action=MIGRATION_MAP_ACTION_REMAP,
        table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="items", to_name="products"),),
        column_renames=(),
        dropped_tables=(),
        dropped_columns=(),
        added_tables=(),
        added_columns=(),
    )
    (tmp_path / "schema_migration_map.json").write_text(
        json.dumps(
            {
                "version": map_obj.version,
                "action": map_obj.action,
                "table_renames": [{"entry_type": "table", "from_name": "items", "to_name": "products"}],
                "column_renames": [],
                "dropped_tables": [],
                "dropped_columns": [],
                "added_tables": [],
                "added_columns": [],
            }
        ),
        encoding="utf-8",
    )
    rebuild_calls: list[bool] = []

    def _build_graph(*_args, **kwargs):
        rebuild_calls.append(bool(kwargs.get("force_live_schema_reflect")))
        from aetherdialect._schema_graph import diff_schemas

        return live, diff_schemas(owner, live)

    monkeypatch.setenv("AETHERDIALECT_ENGINE", "duckdb")
    monkeypatch.setenv("DUCKDB_DATABASE", ":memory:")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with (
        patch("aetherdialect._main_execution.MainExecutionOps.compute_engine_storage_dir", return_value=artifacts_dir),
        patch("aetherdialect._main_execution.DialectRegistry.get", return_value=MagicMock()),
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", side_effect=_build_graph),
        patch("aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch(
            "aetherdialect._templates.TemplateOps.apply_schema_migration_map",
            wraps=TemplateOps.apply_schema_migration_map,
        ),
    ):
        with pytest.raises(MigrationPendingError):
            _main_execution.MainExecutionOps.initialize_aether_engine(
                artifacts_dir=artifacts_dir,
                schema_role="owner",
                execution_engine=MagicMock(),
                log_sink=lambda _msg: None,
            )

    assert len(rebuild_calls) >= 2
    assert rebuild_calls[0] is True
    assert rebuild_calls[-1] is True


@pytest.mark.fast
def test_federation_composite_drift_gate_precedes_persist_in_source() -> None:
    import inspect

    source = inspect.getsource(_main_execution.MainExecutionOps.initialize_aether_federation)
    tier_idx = source.index("federation_composite_migration_tier(")
    persist_idx = source.index("persist_federation_tree(")
    gate_idx = source.index("Federation composite drift", tier_idx)
    assert tier_idx < gate_idx < persist_idx


@pytest.mark.fast
def test_federation_composite_remap_drift_raises_before_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_members, declaration_path = _bootstrap_federation(tmp_path, monkeypatch)
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_gate")
    paths = federation_artifact_paths(fed_dir)
    composite_before = Path(paths["composite_schema"]).read_bytes()
    manifest_path = Path(paths["artifact_manifest"])
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["structural_hash"] = "stale_structural_hash"
    stored["effective_structural_hash"] = "stale_effective_hash"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    persist_calls: list[str] = []
    real_persist = _main_execution.persist_federation_tree

    def _spy_persist(federation_dir: str, **kwargs) -> None:
        persist_calls.append(federation_dir)
        real_persist(federation_dir, **kwargs)

    with patch("aetherdialect._main_execution.persist_federation_tree", side_effect=_spy_persist):
        with pytest.raises(MigrationPendingError, match="Federation migration required"):
            AetherFederation(
                "fed_gate",
                members=mock_members,
                declaration_file=str(declaration_path),
                artifacts_dir=str(tmp_path),
            )

    assert persist_calls == []
    assert Path(paths["composite_schema"]).read_bytes() == composite_before
    skeleton_path = Path(fed_dir) / FEDERATION_MIGRATION_MAP_FILENAME
    assert skeleton_path.is_file()


@pytest.mark.fast
def test_federation_notes_change_soft_refreshes_when_member_replay_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    members = {
        "alpha": _member_graph("entity_a", source_id="alpha"),
        "beta": _member_graph("entity_b", source_id="beta"),
    }
    mock_members = {sid: _mock_member(graph, sid, tmp_path) for sid, graph in members.items()}
    declaration_path = write_federation_declaration_file(tmp_path, _MANIFEST)
    notes_path = tmp_path / "federation_notes.txt"
    notes_path.write_text("Customers in the west region.", encoding="utf-8")
    context = FederationContext(notes_file=str(notes_path))

    AetherFederation(
        "fed_gate",
        members=mock_members,
        declaration_file=str(declaration_path),
        artifacts_dir=str(tmp_path),
        context=context,
    )
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_gate")
    paths = federation_artifact_paths(fed_dir)
    before = load_federation_composite_graph(fed_dir)
    assert before is not None
    west_hash = before.notes_sha256

    notes_path.write_text("Customers in the east region.", encoding="utf-8")
    AetherFederation(
        "fed_gate",
        members=mock_members,
        declaration_file=str(declaration_path),
        artifacts_dir=str(tmp_path),
        context=context,
    )

    after = load_federation_composite_graph(fed_dir)
    manifest = json.loads(Path(paths["artifact_manifest"]).read_text(encoding="utf-8"))
    assert after is not None
    assert after.notes_sha256 != west_hash
    assert manifest["notes_hash"] == after.notes_hash
