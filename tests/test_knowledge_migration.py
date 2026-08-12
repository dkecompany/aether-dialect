"""Knowledge migration for schema delete/rename and notes-hash DK invalidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._utils import load_domain_knowledge_artifact
from aetherdialect._utils_artifacts import save_domain_knowledge_artifact


def _graph(*tables: str) -> SchemaGraph:
    out: dict[str, TableMetadata] = {}
    for name in tables:
        out[name] = TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")},
            primary_key=["id"],
            foreign_keys=[],
        )
    return SchemaGraph(tables=out, join_paths_multi={})


@pytest.mark.fast
def test_structural_knowledge_fact_rejects_residual_and_empty_reference_set() -> None:
    with pytest.raises(ConfigError, match="residual"):
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind="residual",
                text="orphan note",
                referenced_entities=frozenset({"store"}),
            )
        )
    with pytest.raises(ConfigError, match="referenced_entities"):
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(kind="relation", text="store header", referenced_entities=frozenset())
        )


@pytest.mark.fast
def test_structural_knowledge_payload_kind_dispatch() -> None:
    fact = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
            text="status is closed set",
            referenced_entities=frozenset({"orders.status"}),
            payload={"values": ["open", "closed"]},
        )
    )
    assert fact.payload == {"values": ["open", "closed"]}
    with pytest.raises(ConfigError, match="payload"):
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind=StructuralKnowledgeKind.RELATION.value,
                text="orders header",
                referenced_entities=frozenset({"orders"}),
                payload={"values": ["open"]},
            )
        )


@pytest.mark.fast
def test_migrate_domain_knowledge_drops_deleted_and_remaps_renamed() -> None:
    entries = (
        DomainKnowledgeEntry(key="store", kind="glossary", text="a store", referenced_entities=frozenset({"store"})),
        DomainKnowledgeEntry(
            key="store.store_id", kind="glossary", text="pk", referenced_entities=frozenset({"store.store_id"})
        ),
        DomainKnowledgeEntry(
            key="payroll", kind="glossary", text="hidden side", referenced_entities=frozenset({"payroll"})
        ),
        DomainKnowledgeEntry(key="arr", kind="glossary", text="concept", referenced_entities=frozenset()),
    )
    migrated = MainSpaceOps.migrate_domain_knowledge_entries(
        entries,
        tmap={"store": "shop"},
        colmaps={"store": {"store_id": "shop_id"}},
        drop_tables=frozenset({"payroll"}),
        drop_columns=frozenset(),
        schema=_graph("shop"),
    )
    keys = {e.key for e in migrated}
    assert keys == {"shop", "arr"}


@pytest.mark.fast
def test_migrate_structural_knowledge_remaps_references_without_rewriting_text() -> None:
    facts = (
        StructuralKnowledgeFact(
            kind="join",
            text="store joins inventory on store.store_id",
            referenced_entities=frozenset({"store", "inventory", "store.store_id"}),
        ),
        StructuralKnowledgeFact(
            kind="relation",
            text="payroll is confidential",
            referenced_entities=frozenset({"payroll"}),
        ),
    )
    migrated = MainSpaceOps.migrate_structural_knowledge_facts(
        facts,
        tmap={"store": "shop"},
        colmaps={"store": {"store_id": "shop_id"}},
        drop_tables=frozenset({"payroll"}),
        drop_columns=frozenset(),
    )
    assert len(migrated) == 1
    assert migrated[0].text == "store joins inventory on store.store_id"
    assert migrated[0].referenced_entities == frozenset({"shop", "inventory", "shop.shop_id"})


@pytest.mark.fast
def test_domain_knowledge_artifact_notes_hash_mismatch_forces_miss(tmp_path: Path) -> None:
    from aetherdialect._knowledge_staleness import knowledge_artifact_save_stamps

    sg = _graph("store")
    sg.notes_sha256 = "aaa"
    sg_b = _graph("store")
    sg_b.notes_sha256 = "bbb"
    save_domain_knowledge_artifact(
        tmp_path,
        (
            DomainKnowledgeEntry(
                key="arr", kind="glossary", text="annual recurring revenue", referenced_entities=frozenset()
            ),
        ),
        **knowledge_artifact_save_stamps(sg_b),
    )
    assert load_domain_knowledge_artifact(tmp_path, sg) is None
    loaded = load_domain_knowledge_artifact(tmp_path, sg_b)
    assert loaded is not None
    assert len(loaded) == 1


@pytest.mark.fast
def test_space_snapshot_structural_edit_migrates_knowledge() -> None:
    snap = {
        "tables": ["store", "payroll"],
        "columns": ["store.store_id"],
        "table_descriptions": {"store": "s", "payroll": "p"},
        "column_meta": {},
        "domain_knowledge": [
            {"key": "store", "kind": "glossary", "text": "a store", "referenced_entities": ["store"]},
            {"key": "payroll", "kind": "glossary", "text": "pay", "referenced_entities": ["payroll"]},
        ],
        "structural_knowledge": [
            {"kind": "relation", "text": "store has inventory", "referenced_entities": ["store"]},
            {"kind": "relation", "text": "payroll is private", "referenced_entities": ["payroll"]},
        ],
    }
    edited = MainSpaceOps._apply_structural_edit_to_aetherspace_snapshot(
        snap,
        tmap={"store": "shop"},
        colmaps={"store": {"store_id": "shop_id"}},
        drop_tables=frozenset({"payroll"}),
        drop_columns=frozenset(),
    )
    assert edited["tables"] == ["shop"]
    dk_keys = {e["key"] for e in edited["domain_knowledge"]}
    assert dk_keys == {"shop"}
    assert len(edited["structural_knowledge"]) == 1
    assert edited["structural_knowledge"][0]["text"] == "store has inventory"
    assert edited["structural_knowledge"][0]["referenced_entities"] == ["shop"]
