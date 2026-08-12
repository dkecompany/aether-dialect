"""Scope and RBAC guards for knowledge/structure export, templates, sessions, and migration maps."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, DomainKnowledgeEntry, SpaceContext
from aetherdialect._contracts_base import (
    ConfigError,
    DomainKnowledgeHolder,
    EngineContext,
    SchemaAccessError,
    SchemaRole,
    SensitivityClassification,
)
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_finalize import build_public_structure_document
from aetherdialect._schema_graph import validate_scope_against_graph
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config
from tests.template_fixtures import _minimal_template


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", role="identifier", description="pk"),
                    "amount": ColumnMetadata(name="amount", data_type="numeric", role="measure", description="money"),
                    "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
                    "ssn": ColumnMetadata(
                        name="ssn",
                        data_type="text",
                        sensitivity=SensitivityClassification.HIDDEN,
                        description="hidden ssn",
                    ),
                },
                primary_key=["id"],
                foreign_keys=[
                    FKEdge(
                        src_table="orders",
                        src_cols=["customer_id"],
                        dst_table="customers",
                        dst_cols=["id"],
                    )
                ],
                description="customer orders",
            ),
            "customers": TableMetadata(
                name="customers",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer"),
                    "name": ColumnMetadata(name="name", data_type="text", description="display name"),
                },
                primary_key=["id"],
                foreign_keys=[],
                description="buyer roster",
            ),
            "secrets": TableMetadata(
                name="secrets",
                columns={"id": ColumnMetadata(name="id", data_type="integer", description="secret pk")},
                primary_key=["id"],
                foreign_keys=[],
                description="hidden vault",
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h-scope",
        schema_graph_id="sg-scope__h",
    )


def _minimal_engine(tmp_path: Path, **overrides: object) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env={})
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir=str(tmp_path),
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=_schema(),
        _dialect=MagicMock(),
        _artifacts_dir=tmp_path,
        _store=TemplateOps.empty_template_store("graph-scope"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_role=SchemaRole.OWNER,
        _consumer_visible_objects=None,
        _schema_stats={"table_count": 3, "total_filterable": 6},
        _phase_callback=None,
        _token_provider=None,
        _context_name="master",
        _sandbox_closed=False,
        _closed=False,
        _sandbox_mode=False,
        _credential_default_space_uid=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, str(key), value)
    obj._domain_knowledge = DomainKnowledgeHolder()
    return obj


@pytest.mark.fast
def test_consumer_export_knowledge_hides_restricted_tables(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    entries = (
        DomainKnowledgeEntry(
            key="arr",
            text="Annual recurring revenue.",
            kind="glossary",
            referenced_entities=frozenset({"orders"}),
        ),
        DomainKnowledgeEntry(
            key="vault",
            text="Secret vault metric.",
            kind="glossary",
            referenced_entities=frozenset({"secrets"}),
        ),
    )
    engine._replace_domain_knowledge(entries)
    owner = engine.export_knowledge()
    assert {e["key"] for e in owner["domain_knowledge"]} == {"arr", "vault"}
    assert "secrets" in owner["table_descriptions"]
    assert owner["table_descriptions"]["secrets"] == "hidden vault"
    assert "customers.name" in owner["column_descriptions"]
    assert any(e.get("referenced_entities") for e in owner["domain_knowledge"])

    filtered = MainExecutionOps.build_space_knowledge_export(
        engine_entries=entries,
        space="master",
        schema_graph=engine._schema_graph,
        scope_ctx=EngineContext(),
        visible_objects=frozenset({"orders", "customers"}),
    )
    assert {e["key"] for e in filtered["domain_knowledge"]} == {"arr"}
    assert "secrets" not in filtered["table_descriptions"]
    assert "orders" in filtered["table_descriptions"]
    assert "orders.ssn" not in filtered["column_descriptions"]

    snapshot = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(
            tables=frozenset({"orders", "customers"}),
            columns=frozenset({"orders.id", "orders.amount", "orders.customer_id", "customers.id", "customers.name"}),
        ),
    )
    snapshot["uid"] = "S0001"
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "S0001", snapshot)
    consumer = _minimal_engine(
        tmp_path,
        _schema_role=SchemaRole.CONSUMER,
        _consumer_visible_objects=frozenset({"orders", "customers"}),
        _credential_default_space_uid="S0001",
        _schema_graph=engine._schema_graph,
    )
    consumer._domain_knowledge = engine._domain_knowledge
    payload = consumer.export_knowledge()
    assert "vault" not in {e["key"] for e in payload["domain_knowledge"]}
    assert "secrets" not in payload["table_descriptions"]


@pytest.mark.fast
def test_consumer_export_structure_filters_fk_pk_edits(tmp_path: Path) -> None:
    engine = _minimal_engine(
        tmp_path,
        _schema_role=SchemaRole.CONSUMER,
        _consumer_visible_objects=frozenset({"orders", "customers"}),
    )
    inventory = {
        "tables": [
            {
                "name": "orders",
                "columns": [{"name": "id", "data_type": "integer"}, {"name": "customer_id", "data_type": "integer"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            },
            {
                "name": "customers",
                "columns": [{"name": "id", "data_type": "integer"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            },
        ],
        "table_count": 2,
        "relationships": [],
    }
    overrides = {
        "tables": {},
        "foreign_keys_add": [
            {"from": "orders.customer_id", "to": "customers.id", "kind": "structural"},
            {"from": "secrets.id", "to": "orders.id", "kind": "structural"},
        ],
        "foreign_keys_remove": [{"from": "secrets.id", "to": "customers.id"}],
        "primary_keys_add": [
            {"table": "orders", "column": "id"},
            {"table": "secrets", "column": "id"},
        ],
        "primary_keys_remove": [{"table": "secrets", "column": "id"}],
    }
    doc = build_public_structure_document(inventory=inventory, overrides=overrides)
    assert doc["foreign_keys_add"] == [
        {"from": "orders.customer_id", "to": "customers.id", "kind": "structural"},
    ]
    assert doc["foreign_keys_remove"] == []
    assert doc["primary_keys_add"] == [{"table": "orders", "column": "id"}]
    assert doc["primary_keys_remove"] == []

    live = engine.export_structure()
    assert {t["name"] for t in live["tables"]} == {"customers", "orders"}
    assert "secrets" not in {t["name"] for t in live["tables"]}


@pytest.mark.fast
def test_execute_template_rejects_hidden_footprint(tmp_path: Path) -> None:
    tmpl = _minimal_template()
    tmpl.tables_used = ["secrets"]
    tmpl.intent_signature.tables = ["secrets"]
    engine = _minimal_engine(
        tmp_path,
        _schema_role=SchemaRole.CONSUMER,
        _consumer_visible_objects=frozenset({"orders", "customers"}),
        _templates={tmpl.id: tmpl},
    )
    with pytest.raises(ConfigError, match=r"unknown template ref"):
        engine.execute_template("T0001", {"p1": "y"})


@pytest.mark.fast
def test_session_ephemeral_scope_out_of_rbac_raises(tmp_path: Path) -> None:
    engine = _minimal_engine(
        tmp_path,
        _schema_role=SchemaRole.CONSUMER,
        _consumer_visible_objects=frozenset({"orders", "customers"}),
    )
    with pytest.raises(ConfigError, match=r"outside visible scope|not in the schema graph"):
        engine.session(mode="reader", ephemeral_scope=SpaceContext(tables=frozenset({"secrets"})))


@pytest.mark.fast
def test_read_space_snapshot_rejects_outside_visibility(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    schema = _schema()
    snap = MainExecutionOps.subset_graph_for_space(
        schema,
        SpaceContext(tables=frozenset({"orders", "customers", "secrets"})),
    )
    snap["uid"] = "wide"
    MainExecutionOps.save_aetherspace_snapshot(engine_dir, "wide", snap)
    export_path = MainExecutionOps.write_space_snapshot(engine_dir, "wide", schema)
    MainExecutionOps.delete_aetherspace_snapshot(engine_dir, "wide")

    with pytest.raises(ConfigError, match=r"outside visible scope"):
        MainExecutionOps.read_space_snapshot(
            engine_dir,
            "imported",
            schema,
            source=export_path,
            visible_objects=frozenset({"orders", "customers"}),
        )


@pytest.mark.fast
def test_apply_knowledge_rejects_bad_referenced_entities(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    with pytest.raises(ConfigError, match=r"referenced_entities.*unknown object"):
        engine.apply_knowledge(
            "master",
            {
                "domain_knowledge": [
                    {
                        "key": "bad",
                        "kind": "glossary",
                        "text": "mentions nothing real",
                        "referenced_entities": ["no_such_table"],
                    }
                ]
            },
        )
    with pytest.raises(ConfigError, match=r"hidden or restricted column"):
        engine.apply_knowledge(
            "master",
            {
                "domain_knowledge": [
                    {
                        "key": "ssn_ref",
                        "kind": "glossary",
                        "text": "sensitive identifier policy",
                        "referenced_entities": ["orders.ssn"],
                    }
                ]
            },
        )


@pytest.mark.fast
def test_master_column_descriptions_on_hidden_column_raises(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    with patch("aetherdialect._main_spaces.save_schema_to_cache"):
        with pytest.raises(ConfigError, match=r"cannot receive a description"):
            engine.apply_knowledge(
                "master",
                {"column_descriptions": {"orders.ssn": "should not land"}},
            )


@pytest.mark.fast
def test_engine_context_unknown_allow_objects_raises() -> None:
    sg = _schema()
    ctx = EngineContext(allow_objects=frozenset({"no_such_table"}))
    with pytest.raises(SchemaAccessError, match=r"allow_objects references unknown table"):
        validate_scope_against_graph(sg, ctx)


@pytest.mark.fast
def test_cross_table_column_moves_nonempty_raises() -> None:
    with pytest.raises(Exception, match=r"cross_table_column_moves is not supported"):
        TemplateOps.parse_schema_migration_map_payload(
            {
                "version": 1,
                "action": "remap",
                "cross_table_column_moves": [
                    {
                        "from_table": "orders",
                        "from_column": "amount",
                        "to_table": "archive",
                        "to_column": "amount",
                    }
                ],
            }
        )
