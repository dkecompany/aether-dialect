"""Tests for federation manifest and mapping load, validation, and composition."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import (
    ConfigError,
    FederationConfigError,
    FederationDeclarationError,
    FederationMemberUnprofilableError,
)
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    compute_federation_storage_dir,
    federation_source_artifacts_dir,
)
from aetherdialect._federation_manifest import (
    build_federation_manifest_from_members,
    export_federation_manifest,
    federation_artifact_paths,
    federation_manifest_document,
    federation_manifest_is_active,
    manifest_hash,
    mappings_hash,
    parse_federation_declaration,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_reflect import resolve_federation_qualified_ref
from aetherdialect._utils_artifacts import load_runtime_config, write_gzip_json_atomic
from tests.federation_helpers import (
    enriched_manifest,
    union_member_graph_pair,
    write_federation_declaration_file,
)


def _simple_graph(name: str, source_id: str = "") -> SchemaGraph:
    table = TableMetadata(
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
    tables = {name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="test-profiled",
    )


_MANIFEST = {
    "federation_id": "fed_alpha",
    "cross_source_joins": [
        {
            "left": "entity_a.email",
            "right": "entity_b.email",
            "kind": "inner",
            "logical_key": "email",
        }
    ],
    "coordinator": {"row_cap": 1000},
}


def _members() -> dict[str, SchemaGraph]:
    return {
        "alpha": _simple_graph("entity_a"),
        "beta": _simple_graph("entity_b"),
    }


def _engine_member(connection: str) -> MagicMock:
    member = MagicMock()
    member.dialect = "duckdb"
    member._connection = connection
    member._context_name = "master"
    member._schema_role = "owner"
    return member


def _engine_members() -> dict[str, MagicMock]:
    return {
        "alpha": _engine_member("alpha"),
        "beta": _engine_member("beta"),
    }


def test_federation_manifest_is_inactive_without_manifest() -> None:
    assert federation_manifest_is_active(None) is False
    assert federation_manifest_is_active({}) is False


def test_federation_manifest_is_active_with_manifest() -> None:
    assert federation_manifest_is_active(_MANIFEST) is True


def test_parse_federation_manifest_rejects_unknown_alias_source() -> None:
    bad = dict(_MANIFEST)
    bad["aliases"] = {"renamed_a": {"source": "missing", "table": "entity_a"}}
    members = {
        "alpha": MagicMock(dialect="duckdb", _connection="alpha", _context_name="master", _schema_role="owner"),
        "beta": MagicMock(dialect="duckdb", _connection="beta", _context_name="master", _schema_role="owner"),
    }
    with pytest.raises(FederationConfigError, match="unknown source_id"):
        build_federation_manifest_from_members(
            members,
            declaration=parse_federation_manifest(bad),
            member_graphs=_members(),
        )


def test_parse_federation_manifest_allows_declaration_without_sources() -> None:
    declaration = {
        "federation_id": "fed_decl",
        "aliases": {},
        "cross_source_joins": [],
        "coordinator": {"row_cap": 1000},
    }
    parsed = parse_federation_manifest(declaration)
    assert parsed.sources == ()
    assert parsed.table_namespace == {}


def test_parse_federation_manifest_order_independent_id() -> None:
    base = dict(_MANIFEST)
    m1 = parse_federation_manifest(
        {
            **base,
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "role": "owner"},
                {"source_id": "beta", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha", "entity_b": "beta"},
        },
        include_derived_roster=True,
    )
    m2 = parse_federation_manifest(
        {
            **base,
            "sources": [
                {"source_id": "beta", "engine": "duckdb", "role": "owner"},
                {"source_id": "alpha", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha", "entity_b": "beta"},
        },
        include_derived_roster=True,
    )
    assert manifest_hash(m1) == manifest_hash(m2)


def test_compose_composite_graph_stamps_source_ids() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _simple_graph("entity_a"),
        "beta": _simple_graph("entity_b"),
    }
    composite = compose_composite_graph(members, manifest)
    assert composite.tables["entity_a"].source_id == "alpha"
    assert composite.tables["entity_b"].source_id == "beta"
    assert composite.schema_graph_id


def test_sensitive_cross_source_key_rejected() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _simple_graph("entity_a"),
        "beta": _simple_graph("entity_b"),
    }
    members["beta"].tables["entity_b"].columns["email"].sensitivity = "restricted"
    with pytest.raises(FederationDeclarationError):
        compose_composite_graph(members, manifest)


def test_mappings_hash_stable() -> None:
    mappings = parse_federation_mappings({"version": "0.2.3", "logical_columns": []})
    assert mappings_hash(mappings) == mappings_hash(mappings)


def test_live_fixture_manifest_parses() -> None:
    declaration_path = (
        Path(__file__).resolve().parents[1] / "live_tests" / "fixtures" / "federation_live_declaration.json"
    )
    parsed, mappings = parse_federation_declaration(json.loads(declaration_path.read_text(encoding="utf-8")))
    assert parsed.federation_id == "live_rental_shop"
    assert parsed.sources == ()
    assert parsed.table_namespace == {}
    assert mappings.logical_tables


def test_invalid_declaration_raises_without_persisting(tmp_path: Path) -> None:
    alpha_graph = _simple_graph("entity_a", source_id="alpha")
    beta_graph = _simple_graph("entity_b", source_id="beta")
    beta_graph.tables["entity_b"].columns["email"].sensitivity = "restricted"
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_alpha")
    paths = federation_artifact_paths(fed_dir)

    def _member(graph: SchemaGraph, connection: str) -> MagicMock:
        llm_exec = load_runtime_config(merged_env={})
        member = MagicMock()
        member.dialect = "duckdb"
        member._runtime_config = RuntimeConfig(
            engine="duckdb",
            artifacts_dir=str(tmp_path / connection),
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

    with pytest.raises(FederationDeclarationError, match="cross-source join key must be sensitivity none"):
        manifest_path = write_federation_declaration_file(tmp_path, _MANIFEST)
        AetherFederation(
            "fed_alpha",
            members=(_member(alpha_graph, "alpha"), _member(beta_graph, "beta")),
            declaration=str(manifest_path),
            artifacts_dir=str(tmp_path),
        )

    assert not Path(paths["composite_schema"]).exists()
    assert not Path(paths["manifest"]).exists()
    assert not Path(paths["artifact_manifest"]).exists()


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


def _unprofiled_graph(name: str, source_id: str = "") -> SchemaGraph:
    table = TableMetadata(
        name=name,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="",
    )


def _write_member_schema_artifact(
    artifacts_dir: Path,
    manifest: object,
    source_id: str,
    graph: SchemaGraph,
) -> None:
    binding = next(binding for binding in manifest.sources if binding.source_id == source_id)
    member_dir = federation_source_artifacts_dir(
        str(artifacts_dir),
        binding,
        federation_id=str(getattr(manifest, "federation_id", "") or "") or None,
    )
    os.makedirs(member_dir, exist_ok=True)
    write_gzip_json_atomic(
        os.path.join(member_dir, "schema_graph.json.gz"),
        graph.to_dict(),
        sort_keys=True,
    )


_FOUR_MEMBER_MANIFEST = {
    "federation_id": "fed_four",
    "cross_source_joins": [
        {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id_ab"},
        {"left": "t_c.id", "right": "t_d.id", "kind": "inner", "logical_key": "id_cd"},
    ],
}


def test_unprofiled_member_raises_at_init_without_persisting_tree(tmp_path: Path) -> None:
    alpha_graph = _unprofiled_graph("entity_a", source_id="alpha")
    beta_graph = _simple_graph("entity_b", source_id="beta")
    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_alpha")
    paths = federation_artifact_paths(fed_dir)

    with pytest.raises(FederationMemberUnprofilableError, match="schema is not profiled"):
        manifest_path = write_federation_declaration_file(tmp_path, _MANIFEST)
        AetherFederation(
            "fed_alpha",
            members=(_mock_member(alpha_graph, "alpha", tmp_path), _mock_member(beta_graph, "beta", tmp_path)),
            declaration=str(manifest_path),
            artifacts_dir=str(tmp_path),
        )

    assert not Path(paths["composite_schema"]).exists()
    assert not Path(paths["manifest"]).exists()
    assert not Path(paths["artifact_manifest"]).exists()


def test_partial_member_artifact_roster_raises_without_composing(tmp_path: Path) -> None:
    members = {
        "alpha": _simple_graph("t_a", source_id="alpha"),
        "beta": _simple_graph("t_b", source_id="beta"),
        "gamma": _simple_graph("t_c", source_id="gamma"),
        "delta": _simple_graph("t_d", source_id="delta"),
    }
    mock_members = {source_id: _mock_member(graph, source_id, tmp_path) for source_id, graph in members.items()}
    manifest = build_federation_manifest_from_members(
        mock_members,
        declaration=parse_federation_manifest(_FOUR_MEMBER_MANIFEST),
        member_graphs=members,
    )
    for source_id in ("alpha", "beta", "gamma"):
        _write_member_schema_artifact(tmp_path, manifest, source_id, members[source_id])

    fed_dir = compute_federation_storage_dir(str(tmp_path), "fed_four")
    paths = federation_artifact_paths(fed_dir)
    declaration_path = write_federation_declaration_file(tmp_path, _FOUR_MEMBER_MANIFEST)

    with pytest.raises(FederationMemberUnprofilableError, match="incomplete"):
        AetherFederation(
            "fed_four",
            members=mock_members,
            declaration=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )

    assert not Path(paths["composite_schema"]).exists()
    assert not Path(paths["manifest"]).exists()
    assert not Path(paths["artifact_manifest"]).exists()


_ONE_MEMBER_MANIFEST = {
    "federation_id": "fed_one",
    "cross_source_joins": [],
}

_UNION_MANIFEST = {
    "federation_id": "fed_union_manifest",
    "cross_source_joins": [],
}


def _union_mappings() -> object:
    return parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )


def test_one_member_roster_composes() -> None:
    manifest = parse_federation_manifest(_ONE_MEMBER_MANIFEST, include_derived_roster=True)
    members = {"solo": _simple_graph("t_solo", source_id="solo")}
    composite = compose_composite_graph(members, manifest)
    assert "t_solo" in composite.tables
    assert composite.tables["t_solo"].source_id == "solo"


def test_binding_raises_when_member_key_disagrees_with_connection() -> None:
    member = MagicMock(
        spec=[
            "dialect",
            "_connection",
            "_named_connection",
            "_connection_mapping",
            "_context_name",
            "_schema_role",
            "_schema_graph",
        ]
    )
    member.dialect = "duckdb"
    member._connection = "registry_key"
    member._named_connection = "registry_key"
    member._connection_mapping = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._schema_graph = None
    member2 = MagicMock(
        spec=[
            "dialect",
            "_connection",
            "_named_connection",
            "_connection_mapping",
            "_context_name",
            "_schema_role",
            "_schema_graph",
        ]
    )
    member2.dialect = "duckdb"
    member2._connection = "other"
    member2._named_connection = "other"
    member2._connection_mapping = None
    member2._context_name = "master"
    member2._schema_role = "owner"
    member2._schema_graph = _simple_graph("t", source_id="other")
    with pytest.raises(FederationConfigError, match="does not expose a schema graph"):
        build_federation_manifest_from_members(
            {"registry_key": member, "other": member2},
            declaration=parse_federation_manifest({"federation_id": "fed_bind", "cross_source_joins": []}),
        )


def test_ambiguous_federation_reference_raises_for_collapsed_logical_table() -> None:
    members = union_member_graph_pair("payment_a", "payment_b")
    engines = {"a": _engine_member("a"), "b": _engine_member("b")}
    manifest = enriched_manifest(engines, _UNION_MANIFEST, member_graphs=members)
    mappings = _union_mappings()
    composite = compose_composite_graph(members, manifest, mappings)
    with pytest.raises(ConfigError, match="ambiguous federation reference"):
        resolve_federation_qualified_ref("payment.id", manifest=manifest, schema=composite)


def test_three_part_federation_reference_disagrees_with_namespace_raises() -> None:
    members = _members()
    manifest = enriched_manifest(_engine_members(), _MANIFEST, member_graphs=members)
    with pytest.raises(ConfigError, match="disagrees with table_namespace"):
        resolve_federation_qualified_ref("beta.entity_a.email", manifest=manifest)


def test_export_federation_manifest_writes_derived_roster(tmp_path: Path) -> None:
    members = _members()
    manifest = enriched_manifest(_engine_members(), _MANIFEST, member_graphs=members)
    target = tmp_path / "federation_manifest.json"
    export_federation_manifest(manifest, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["federation_id"] == "fed_alpha"
    assert len(payload["sources"]) == 2
    assert payload["table_namespace"]["entity_a"] == "alpha"


def test_export_federation_manifest_rejects_missing_parent(tmp_path: Path) -> None:
    members = _members()
    manifest = enriched_manifest(_engine_members(), _MANIFEST, member_graphs=members)
    missing = tmp_path / "missing" / "federation_manifest.json"
    with pytest.raises(FederationConfigError, match="export directory does not exist"):
        export_federation_manifest(manifest, missing)


def test_declaration_manifest_resolves_mapping_refs_without_sources() -> None:
    declaration = parse_federation_manifest(
        {
            "federation_id": "fed_decl",
            "cross_source_joins": [],
            "coordinator": {"row_cap": 1000},
        },
    )
    resolved = resolve_federation_qualified_ref("orders.id", manifest=declaration)
    assert resolved.table == "orders"
    assert resolved.column == "id"
    assert resolved.source_id == ""


def test_split_fk_endpoint_uses_shared_resolver_with_manifest() -> None:
    from aetherdialect._schema_reflect import split_fk_endpoint

    members = _members()
    manifest = enriched_manifest(_engine_members(), _MANIFEST, member_graphs=members)
    parsed = split_fk_endpoint("alpha.entity_a.id", manifest=manifest)
    assert parsed == ("entity_a", ["id"])


@pytest.mark.fast
def test_load_inference_block_lists_resolves_fk_blocks_with_manifest(tmp_path: Path) -> None:
    from aetherdialect._schema_reflect import load_inference_block_lists, structure_sidecar_path

    cache_path = tmp_path / "schema.json.gz"
    cache_path.write_bytes(b"")
    sidecar = structure_sidecar_path(cache_path)
    sidecar.write_text(
        json.dumps(
            {
                "_internal": {
                    "fk_block_inferred": [{"from": "alpha.entity_a.id", "to": "beta.entity_b.id"}],
                    "pk_block_inferred": [],
                }
            }
        ),
        encoding="utf-8",
    )
    members = _members()
    manifest = enriched_manifest(_engine_members(), _MANIFEST, member_graphs=members)
    _, fk_blocked = load_inference_block_lists(cache_path, manifest=manifest)
    assert ("entity_a", ("id",), "entity_b", ("id",)) in fk_blocked


@pytest.mark.fast
def test_parse_federation_manifest_rejects_authored_sources() -> None:
    bad = dict(_MANIFEST)
    bad["sources"] = [{"source_id": "alpha", "engine": "duckdb", "role": "owner"}]
    with pytest.raises(FederationDeclarationError, match="sources are derived"):
        parse_federation_manifest(bad)


@pytest.mark.fast
def test_parse_federation_manifest_rejects_authored_table_namespace() -> None:
    bad = dict(_MANIFEST)
    bad["table_namespace"] = {"entity_a": "alpha"}
    with pytest.raises(FederationDeclarationError, match="table_namespace is derived"):
        parse_federation_manifest(bad)


@pytest.mark.fast
def test_parse_federation_manifest_rejects_coordinator_engine() -> None:
    bad = dict(_MANIFEST)
    bad["coordinator"] = {"engine": "duckdb", "row_cap": 1000}
    with pytest.raises(FederationDeclarationError, match=r"unknown keys: engine"):
        parse_federation_manifest(bad)


@pytest.mark.fast
def test_parse_federation_manifest_coordinator_defaults_without_engine() -> None:
    parsed = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    assert parsed.coordinator.row_cap == 1000
    doc = federation_manifest_document(parsed)
    assert "engine" not in doc["coordinator"]
    assert doc["coordinator"]["row_cap"] == 1000


@pytest.mark.fast
def test_load_federation_manifest_from_path_reports_declarations_file(tmp_path: Path) -> None:
    from aetherdialect._federation_manifest import load_federation_manifest_from_path

    path = tmp_path / "federation_manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FederationConfigError, match="declarations file"):
        load_federation_manifest_from_path(str(path))
