"""export_knowledge, export_structure, and apply_knowledge."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, DomainKnowledgeEntry, SpaceContext
from aetherdialect._contracts_base import DomainKnowledgeHolder, EngineContext, SensitivityClassification
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", role="identifier", description="pk"),
                    "amount": ColumnMetadata(name="amount", data_type="numeric", role="measure", description="money"),
                    "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
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
                    "name": ColumnMetadata(name="name", data_type="text"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
            "secrets": TableMetadata(
                name="secrets",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h-ek",
        schema_graph_id="sg-ek__h",
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
        _store=TemplateOps.empty_template_store("graph-ek"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_role="owner",
        _consumer_visible_objects=None,
        _schema_stats={"table_count": 3, "total_filterable": 6},
        _phase_callback=None,
        _token_provider=None,
        _context_name="master",
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, str(key), value)
    obj._domain_knowledge = DomainKnowledgeHolder()
    return obj


def _entry_dict(entry: DomainKnowledgeEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "kind": entry.kind,
        "text": entry.text,
        "referenced_entities": list(entry.referenced_entities or ()),
    }


def _knowledge_document(
    *,
    domain_knowledge: Any = None,
    table_descriptions: dict[str, str] | None = None,
    column_descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    if domain_knowledge is not None:
        if (
            isinstance(domain_knowledge, (list, tuple))
            and domain_knowledge
            and isinstance(domain_knowledge[0], DomainKnowledgeEntry)
        ):
            doc["domain_knowledge"] = [_entry_dict(e) for e in domain_knowledge]
        else:
            doc["domain_knowledge"] = list(domain_knowledge)
    if table_descriptions is not None:
        doc["table_descriptions"] = dict(table_descriptions)
    if column_descriptions is not None:
        doc["column_descriptions"] = dict(column_descriptions)
    return doc


@pytest.mark.fast
def test_master_json_shape(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    entries = (DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),)
    engine._replace_domain_knowledge(entries)
    payload = engine.export_knowledge()
    assert set(payload) == {
        "uid",
        "domain_knowledge",
        "table_descriptions",
        "column_descriptions",
    }
    assert payload["uid"] == "master"
    assert payload["domain_knowledge"] == [
        {
            "key": "arr",
            "kind": "glossary",
            "text": "Annual recurring revenue.",
            "referenced_entities": [],
        },
    ]
    assert payload["table_descriptions"]["orders"] == "customer orders"
    assert payload["column_descriptions"]["orders.id"] == "pk"
    assert "tables" not in payload


@pytest.mark.fast
def test_space_overlay_is_space_local(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine._replace_domain_knowledge((DomainKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),))
    space_snap: dict[str, Any] = {
        "uid": "analytics",
        "tables": ["orders"],
        "columns": ["orders.id", "orders.amount"],
        "table_descriptions": {"orders": "space orders"},
        "column_meta": {"orders.amount": {"description": "space amount"}},
        "domain_knowledge": [
            {
                "key": "arr",
                "kind": "glossary",
                "text": "space arr wins",
                "referenced_entities": [],
            },
            {
                "key": "nrr",
                "kind": "metric",
                "text": "net recurring",
                "referenced_entities": [],
            },
        ],
    }
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "analytics", space_snap)
    master = engine.export_knowledge()
    space = engine.export_knowledge(space="analytics")
    assert master["uid"] == "master"
    assert space["uid"] == "analytics"
    assert master["domain_knowledge"][0]["text"] == "engine arr"
    merged = MainExecutionOps.merge_domain_knowledge(
        (DomainKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),),
        (
            DomainKnowledgeEntry(key="arr", text="space arr wins", kind="glossary"),
            DomainKnowledgeEntry(key="nrr", text="net recurring", kind="metric"),
        ),
    )
    assert {e["key"]: e["text"] for e in space["domain_knowledge"]} == {e.key: e.text for e in merged}
    assert space["table_descriptions"] == {"orders": "space orders"}
    assert space["column_descriptions"] == {"orders.amount": "space amount"}


@pytest.mark.fast
def test_digest_changes_when_dk_changes(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine._replace_domain_knowledge((DomainKnowledgeEntry(key="t", text="one"),))
    d1 = engine._domain_knowledge.digest()
    engine._replace_domain_knowledge((DomainKnowledgeEntry(key="t", text="two"),))
    d2 = engine._domain_knowledge.digest()
    assert d1 != d2


@pytest.mark.fast
def test_export_structure_omits_hidden_and_restricted(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    secret = ColumnMetadata(
        name="ssn",
        data_type="text",
        sensitivity=SensitivityClassification.HIDDEN,
    )
    email = ColumnMetadata(
        name="email",
        data_type="text",
        sensitivity=SensitivityClassification.RESTRICTED,
    )
    engine._schema_graph.tables["customers"].columns["ssn"] = secret
    engine._schema_graph.tables["customers"].columns["email"] = email
    meta = engine.export_structure()
    customers = next(t for t in meta["tables"] if t["name"] == "customers")
    names = {c["name"] for c in customers["columns"]}
    assert "id" in names
    assert "name" in names
    assert "ssn" not in names
    assert "email" not in names
    assert all("columns" in t and isinstance(t["columns"], list) for t in meta["tables"])


@pytest.mark.fast
def test_export_structure_has_keys_not_descriptions(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    meta = engine.export_structure()
    assert meta["table_count"] == 3
    orders = next(t for t in meta["tables"] if t["name"] == "orders")
    assert orders["primary_key"] == ["id"]
    assert "description" not in orders
    assert all("description" not in c and "role" not in c for c in orders["columns"])
    assert {c["name"] for c in orders["columns"]} == {"amount", "customer_id", "id"}
    assert any(r["src_table"] == "orders" and r["dst_table"] == "customers" for r in meta["relationships"])


@pytest.mark.fast
def test_export_structure_omits_denied_tables(tmp_path: Path) -> None:
    engine = _minimal_engine(
        tmp_path,
        _consumer_visible_objects=frozenset({"orders", "customers"}),
        _schema_role="consumer",
    )
    meta = engine.export_structure()
    assert {t["name"] for t in meta["tables"]} == {"customers", "orders"}
    assert all("description" not in t for t in meta["tables"])


@pytest.mark.fast
def test_export_apply_export_fixed_point_master(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine._replace_domain_knowledge(
        (DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),)
    )
    before = engine.export_knowledge()
    engine.apply_knowledge("master", before)
    after = engine.export_knowledge()
    assert after == before


@pytest.mark.fast
def test_export_apply_export_fixed_point_named_space(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    snapshot = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(
            tables=frozenset({"orders"}),
            columns=frozenset({"orders.id", "orders.amount"}),
        ),
    )
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "analytics", snapshot)
    engine.apply_knowledge(
        "analytics",
        _knowledge_document(
            domain_knowledge=(DomainKnowledgeEntry(key="arr", text="space arr", kind="glossary"),),
            table_descriptions={"orders": "Orders in analytics"},
            column_descriptions={"orders.amount": "Revenue amount"},
        ),
    )
    before = engine.export_knowledge(space="analytics")
    engine.apply_knowledge("analytics", before)
    after = engine.export_knowledge(space="analytics")
    assert after == before


@pytest.mark.fast
def test_apply_knowledge_round_trip(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    snapshot = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(
            tables=frozenset({"orders"}),
            columns=frozenset({"orders.id", "orders.amount"}),
        ),
    )
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "analytics", snapshot)
    engine.apply_knowledge(
        "analytics",
        _knowledge_document(
            domain_knowledge=(DomainKnowledgeEntry(key="arr", text="space arr", kind="glossary"),),
            table_descriptions={"orders": "Orders in analytics"},
            column_descriptions={"orders.amount": "Revenue amount"},
        ),
    )
    exported = engine.export_knowledge(space="analytics")
    assert exported["domain_knowledge"] == [
        {
            "key": "arr",
            "kind": "glossary",
            "text": "space arr",
            "referenced_entities": [],
        }
    ]
    assert exported["table_descriptions"] == {"orders": "Orders in analytics"}
    assert exported["column_descriptions"]["orders.amount"] == "Revenue amount"
    loaded = MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), "analytics")
    assert loaded is not None
    assert loaded["column_meta"]["orders.amount"]["description"] == "Revenue amount"
    assert "role" not in loaded["column_meta"]["orders.amount"]
    roles_before = {
        cname: str(col.role) if col.role is not None else ""
        for cname, col in engine._schema_graph.tables["orders"].columns.items()
    }
    engine.apply_knowledge(
        "analytics",
        _knowledge_document(column_descriptions={"orders.amount": "Updated amount"}),
    )
    roles_after = {
        cname: str(col.role) if col.role is not None else ""
        for cname, col in engine._schema_graph.tables["orders"].columns.items()
    }
    assert roles_before == roles_after
    assert engine.export_knowledge(space="analytics")["column_descriptions"]["orders.amount"] == "Updated amount"


@pytest.mark.fast
def test_apply_knowledge_unknown_space(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    with pytest.raises(Exception, match="unknown aetherspace"):
        engine.apply_knowledge(
            "missing",
            _knowledge_document(domain_knowledge=(DomainKnowledgeEntry(key="x", text="y"),)),
        )


@pytest.mark.fast
def test_consumer_export_omitted_space_uses_credential_default(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    engine = _minimal_engine(
        tmp_path,
        _schema_role="consumer",
        _consumer_visible_objects=frozenset({"orders", "customers"}),
        _credential_default_space_uid="S0001",
    )
    snapshot = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(
            tables=frozenset({"orders"}),
            columns=frozenset({"orders.id", "orders.amount"}),
        ),
    )
    snapshot["uid"] = "S0001"
    snapshot["domain_knowledge"] = [
        {
            "key": "arr",
            "kind": "glossary",
            "text": "consumer space arr",
            "referenced_entities": [],
        }
    ]
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "S0001", snapshot)
    payload = engine.export_knowledge()
    assert payload["uid"] == "S0001"
    assert payload["domain_knowledge"] == [
        {
            "key": "arr",
            "kind": "glossary",
            "text": "consumer space arr",
            "referenced_entities": [],
        }
    ]
    with pytest.raises(OwnerOnlyOperationError):
        engine.export_knowledge(space="master")


@pytest.mark.fast
def test_consumer_apply_named_space(tmp_path: Path) -> None:
    from aetherdialect._contracts_base import OwnerOnlyOperationError

    engine = _minimal_engine(
        tmp_path,
        _schema_role="consumer",
        _consumer_visible_objects=frozenset({"orders", "customers"}),
        _credential_default_space_uid="S0001",
    )
    snapshot = MainExecutionOps.subset_graph_for_space(
        engine._schema_graph,
        SpaceContext(
            tables=frozenset({"orders"}),
            columns=frozenset({"orders.id", "orders.amount"}),
        ),
    )
    snapshot["uid"] = "S0001"
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "S0001", snapshot)
    with pytest.raises(OwnerOnlyOperationError):
        engine.apply_knowledge(
            "S0001",
            _knowledge_document(
                domain_knowledge=(DomainKnowledgeEntry(key="arr", text="applied by consumer", kind="glossary"),),
                table_descriptions={"orders": "Orders visible to consumer"},
            ),
        )


@pytest.mark.fast
def test_export_knowledge_emits_audit(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    with patch.object(AetherEngine, "_audit_emit") as audit_emit:
        engine.export_knowledge()
    audit_emit.assert_called_once_with(
        "export_knowledge",
        schema_hash="h-ek",
        details=(("space", "master"),),
    )


@pytest.mark.fast
def test_export_structure_emits_audit(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    with patch.object(AetherEngine, "_audit_emit") as audit_emit:
        engine.export_structure()
    audit_emit.assert_called_once_with(
        "export_structure",
        schema_hash="h-ek",
        details=(("space", "master"),),
    )
