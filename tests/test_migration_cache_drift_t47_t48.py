"""Post-map cache bypass and federation composite drift gating."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect import _main_execution
from aetherdialect._constants import FEDERATION_MIGRATION_MAP_FILENAME
from aetherdialect._contracts_base import (
    FederationContext,
    LLMConfig,
    MigrationPendingError,
    RuntimeConfig,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._federation import (
    compute_federation_storage_dir,
    federation_artifact_paths,
    load_federation_composite_graph,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
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
def test_post_map_rebuild_passes_force_live_schema_reflect() -> None:
    source = inspect.getsource(_main_execution.initialize_aether_engine)
    apply_idx = source.index("apply_schema_migration_map(")
    rebuild_idx = source.index("build_schema_graph_with_diff(", apply_idx)
    rebuild_block = source[rebuild_idx : rebuild_idx + 500]
    assert "force_live_schema_reflect=True" in rebuild_block


@pytest.mark.fast
def test_federation_composite_drift_gate_precedes_persist_in_source() -> None:
    source = inspect.getsource(_main_execution.initialize_aether_federation)
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
