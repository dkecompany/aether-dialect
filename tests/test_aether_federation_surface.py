"""Public-surface semantics for :class:`~aetherdialect.AetherFederation`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aetherdialect import AetherFederation
from aetherdialect._contracts_core import AetherFederationInitResult, LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FederationManifest,
    FederationMappings,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


def _graph(table: str, *, source_id: str) -> SchemaGraph:
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
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


_MANIFEST = {
    "federation_id": "fed_surface",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}

_MANIFEST_FILE = "/tmp/aether_fed_surface_declaration.json"


def _minimal_member(*, connection: str = "conn") -> MagicMock:
    llm_exec = load_runtime_config(merged_env={})
    member = MagicMock()
    member.dialect = "duckdb"
    member._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=f"/tmp/aether_fed_member_{connection}",
        engine_context=MagicMock(),
        llm_execution=llm_exec,
    )
    member._llm_config = LLMConfig(provider="openai")
    member._schema_graph = _graph("left_t", source_id="a")
    member._dialect = MagicMock()
    member._execution_engine = None
    member._native_connection = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._engine_identity = None
    member._named_connection = connection
    member._connection = connection
    return member


def _init_bundle(manifest: FederationManifest, composite: SchemaGraph) -> AetherFederationInitResult:
    store = TemplateOps.empty_template_store(str(composite.schema_graph_id))
    return AetherFederationInitResult(
        runtime_config=RuntimeConfig(
            engine="duckdb",
            artifacts_dir="/tmp/aether_fed",
            engine_context=MagicMock(),
            llm_execution=load_runtime_config(merged_env={}),
        ),
        llm_config=LLMConfig(provider="openai"),
        schema_graph=composite,
        dialect=MagicMock(),
        artifacts_dir="/tmp/aether_fed",
        store=store,
        templates={},
        rejected={},
        schema_terms=set(),
        schema_stats={"table_count": 2, "total_filterable": 2},
        federation_manifest=manifest,
        federation_mappings=FederationMappings(version="0.2.3"),
        federation_member_graphs={
            "a": _graph("left_t", source_id="a"),
            "b": _graph("right_t", source_id="b"),
        },
        members=(_minimal_member(connection="conn_a"), _minimal_member(connection="conn_b")),
    )


def test_prepared_federated_outcome_returns_staged_prepare() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    bundle = _init_bundle(manifest, composite)
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_surface",
            members=(_minimal_member(connection="conn_a"), _minimal_member(connection="conn_b")),
            declaration=_MANIFEST_FILE,
        )
    assert fed.prepared_federated_outcome() is None
