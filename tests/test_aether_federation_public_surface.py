"""Lock public-surface semantics and federation guards for :class:`~aetherdialect.AetherFederation`."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import (
    AetherFederationInitResult,
    AuditEvent,
    ConfigError,
    FederationMappings,
    LLMConfig,
    RuntimeConfig,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._federation import (
    FederationManifest,
    binding_from_member_engine,
    compose_composite_graph,
    parse_federation_manifest,
)
from aetherdialect._main_execution import dispose_federation_source_runtimes
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import empty_template_store
from aetherdialect.aetherdialect import FEDERATION_METHOD_SEMANTICS


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
    "federation_id": "fed_public",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}

_MANIFEST_FILE = "/tmp/aether_fed_public_declaration.json"


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
    member._artifacts_dir = f"/tmp/aether_fed_member_{connection}"
    member._dialect = MagicMock()
    member._execution_engine = None
    member._native_connection = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._engine_identity = None
    member._named_connection = None
    member._connection = None
    member.export_schema_overrides.return_value = MagicMock()
    member.apply_schema_overrides.return_value = None
    member.clear_persisted_overrides.return_value = False
    return member


def _init_bundle(manifest: FederationManifest, composite: SchemaGraph) -> AetherFederationInitResult:
    store = empty_template_store(str(composite.schema_graph_id))
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
        federation_mappings=FederationMappings(version=2),
        federation_member_graphs={
            "a": _graph("left_t", source_id="a"),
            "b": _graph("right_t", source_id="b"),
        },
        federation_storage_dir="/tmp/aether_fed/fed_public",
        federation_source_runtimes={
            "a": MagicMock(source_id="a", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
            "b": MagicMock(source_id="b", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
        },
        members={"conn_a": _minimal_member(connection="a"), "conn_b": _minimal_member(connection="b")},
    )


def _fed() -> AetherFederation:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    bundle = _init_bundle(manifest, composite)
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        return AetherFederation(
            "fed_public",
            members={"conn_a": _minimal_member(connection="a"), "conn_b": _minimal_member(connection="b")},
            declaration_file=_MANIFEST_FILE,
        )


_EXPECTED_PUBLIC_METHODS = frozenset(FEDERATION_METHOD_SEMANTICS)


def test_federation_method_semantics_covers_public_methods() -> None:
    public = {
        name
        for name, obj in inspect.getmembers(AetherFederation)
        if not name.startswith("_")
        and callable(obj)
        and not isinstance(inspect.getattr_static(AetherFederation, name), (classmethod, staticmethod))
    }
    assert public >= _EXPECTED_PUBLIC_METHODS
    assert all(
        scope in {"composite", "member", "both", "unsupported"} for scope in FEDERATION_METHOD_SEMANTICS.values()
    )


def test_execute_sql_is_unsupported() -> None:
    fed = _fed()
    with pytest.raises(ConfigError, match="execute_sql is not available on AetherFederation"):
        fed.execute_sql("select 1")


def test_run_seed_warmup_variants_refused() -> None:
    fed = _fed()
    with patch("aetherdialect.aetherdialect.seed_warmup_run_once") as run_once:
        with patch("aetherdialect.aetherdialect.federation_stores_by_source", return_value={}):
            with patch.object(AetherFederation, "_require_open"):
                with patch.object(AetherFederation, "_ensure_llm"):
                    with patch("aetherdialect.aetherdialect.diagnostic_print_listener", return_value=MagicMock()):
                        fed.run_seed_warmup("seed.txt")
    run_once.assert_called_once()
    with pytest.raises(ConfigError, match="run_seed_warmup_from_history is not available"):
        fed.run_seed_warmup_from_history("history.sql")
    with pytest.raises(ConfigError, match="run_seed_warmup_from_query_log is not available"):
        fed.run_seed_warmup_from_query_log()


def _declaration_reload_patch():
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    return patch(
        "aetherdialect.aetherdialect.load_federation_declaration_from_path",
        return_value=(manifest, FederationMappings(version=1)),
    )


def test_schema_overrides_dispatch_to_member() -> None:
    fed = _fed()
    member = fed._members["conn_a"]
    fed.export_schema_overrides("conn_a")
    member.export_schema_overrides.assert_called_once()
    with _declaration_reload_patch():
        with patch(
            "aetherdialect.aetherdialect.initialize_aether_federation",
            return_value=_init_bundle(
                parse_federation_manifest(_MANIFEST, include_derived_roster=True),
                compose_composite_graph(
                    {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
                    parse_federation_manifest(_MANIFEST, include_derived_roster=True),
                ),
            ),
        ):
            fed.apply_schema_overrides("conn_a")
    member.apply_schema_overrides.assert_called_once()


def test_schema_overrides_unknown_member() -> None:
    fed = _fed()
    with pytest.raises(ConfigError, match="unknown federation member"):
        fed.export_schema_overrides("missing")


def test_get_schema_stats_returns_composite_snapshot() -> None:
    fed = _fed()
    stats = fed.get_schema_stats()
    assert stats.stats["table_count"] == 2


def test_show_config_includes_federation_topology() -> None:
    fed = _fed()
    snap = fed.show_config()
    assert "Federation:" in snap.text
    assert "conn_a" in snap.text
    assert "conn_b" in snap.text


def test_clear_template_store_targets_federation_scope() -> None:
    fed = _fed()
    with patch("aetherdialect.aetherdialect.drain_write_queue", return_value=0):
        with patch("aetherdialect.aetherdialect.clear_federation_template_stores", return_value=True) as clear:
            with _declaration_reload_patch():
                with patch(
                    "aetherdialect.aetherdialect.initialize_aether_federation",
                    return_value=_init_bundle(
                        parse_federation_manifest(_MANIFEST, include_derived_roster=True),
                        compose_composite_graph(
                            {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
                            parse_federation_manifest(_MANIFEST, include_derived_roster=True),
                        ),
                    ),
                ):
                    assert fed.clear_template_store() is True
    clear.assert_called_once()


def test_close_disposes_runtimes_and_emits_audit() -> None:
    events: list[AuditEvent] = []

    def sink(ev: AuditEvent) -> None:
        events.append(ev)

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    bundle = _init_bundle(manifest, composite)
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_public",
            members={"conn_a": _minimal_member(connection="a"), "conn_b": _minimal_member(connection="b")},
            declaration_file=_MANIFEST_FILE,
            audit_sink=sink,
        )
    runtimes = fed._federation_source_runtimes
    with patch("aetherdialect.aetherdialect.dispose_federation_source_runtimes") as dispose:
        fed.close()
        dispose.assert_called_once_with(runtimes, member_engines=fed._members)
    assert fed._federation_source_runtimes is None
    assert any(ev.event_type == "close" for ev in events)


def test_context_manager_closes_federation() -> None:
    fed = _fed()
    assert getattr(fed, "_closed", False) is False
    with fed:
        assert getattr(fed, "_closed", False) is False
    assert getattr(fed, "_closed", False) is True
    assert fed._federation_source_runtimes is None


def test_closed_federation_refuses_session() -> None:
    fed = _fed()
    fed.close()
    with pytest.raises(RuntimeError, match="AetherFederation is closed"):
        fed.session()


def test_member_registration_key_must_match_engine_federation_handle() -> None:
    member = _minimal_member(connection="actual_conn")
    member.dialect = "duckdb"
    member._connection = "actual_conn"
    binding = binding_from_member_engine("actual_conn", member)
    assert binding.source_id == "actual_conn"
    assert binding.connection == "actual_conn"


def test_dispose_skips_borrowed_member_connections() -> None:
    member = _minimal_member()
    borrowed_engine = object()
    member._execution_engine = borrowed_engine
    runtime = MagicMock(
        source_id="a",
        dialect=MagicMock(),
        sqlalchemy_engine=borrowed_engine,
        native_connection=None,
    )
    dispose_federation_source_runtimes({"a": runtime}, member_engines={"conn": member})
    runtime.dialect.dispose_native_connection.assert_called_once()
