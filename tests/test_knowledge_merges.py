"""Structural and domain-knowledge merge helpers."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import (
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import _base_description_from_authoritative
from aetherdialect._knowledge_merge import (
    merge_domain_knowledge_federation_peers,
    merge_domain_knowledge_space_over_engine,
    merge_structural_knowledge_space_over_engine,
)
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._schema_graph import recompute_join_paths_multi


def test_merge_domain_knowledge_space_over_engine_keeps_complementary_same_key() -> None:
    engine = (
        DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key="active", text="engine active definition", kind=DomainKnowledgeKind.GLOSSARY.value)
        ),
        DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key="fy", text="engine fy only", kind=DomainKnowledgeKind.POLICY.value)
        ),
    )
    space = (
        DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key="active", text="space active overlay", kind=DomainKnowledgeKind.GLOSSARY.value)
        ),
    )
    merged = merge_domain_knowledge_space_over_engine(engine, space)
    texts = {e.text for e in merged}
    assert "engine fy only" in texts
    assert "engine active definition" in texts
    assert "space active overlay" not in texts


def test_merge_domain_knowledge_federation_peers() -> None:
    a = (
        DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key="k", text="same", kind=DomainKnowledgeKind.GLOSSARY.value)
        ),
    )
    b = (
        DomainKnowledgeEntry.normalize(
            DomainKnowledgeEntry(key="k", text="same", kind=DomainKnowledgeKind.GLOSSARY.value)
        ),
    )
    merged = merge_domain_knowledge_federation_peers((("a", a), ("b", b)))
    assert len(merged) == 1
    assert merged[0].text == "same"


def test_merge_structural_knowledge_space_over_engine() -> None:
    engine = (
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind=StructuralKnowledgeKind.RELATION.value,
                text="engine film entity",
                referenced_entities=frozenset({"film"}),
            )
        ),
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind=StructuralKnowledgeKind.JOIN.value,
                text="engine join note",
                referenced_entities=frozenset({"film", "actor"}),
            )
        ),
    )
    space = (
        StructuralKnowledgeFact.normalize(
            StructuralKnowledgeFact(
                kind=StructuralKnowledgeKind.RELATION.value,
                text="space film entity",
                referenced_entities=frozenset({"film"}),
            )
        ),
    )
    merged = merge_structural_knowledge_space_over_engine(engine, space)
    texts = {f.text for f in merged}
    assert "engine join note" in texts
    assert "engine film entity" in texts
    assert "space film entity" not in texts


def test_base_description_prefers_authoritative_member() -> None:
    a = TableMetadata(
        name="t",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
        base_description="from a",
    )
    b = TableMetadata(
        name="t",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
        base_description="from b",
    )
    assert (
        _base_description_from_authoritative(
            (a, b),
            member_sources=("a", "b"),
            authoritative_source="b",
        )
        == "from b"
    )
    assert (
        _base_description_from_authoritative(
            (a, b),
            member_sources=("a", "b"),
            authoritative_source="",
        )
        == "from a"
    )


def test_enrich_federation_composite_knowledge_prefers_notes_and_filters() -> None:
    tables = {
        "orders": TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer"),
                "ssn": ColumnMetadata(name="ssn", data_type="text", sensitivity="hidden"),
            },
            primary_key=["id"],
            foreign_keys=[],
        )
    }
    graph = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="g_fed",
        effective_structural_hash="eff",
    )
    member_dk = (
        (
            "storefront",
            (
                DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(key="arr", text="member arr", kind=DomainKnowledgeKind.GLOSSARY.value)
                ),
                DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(
                        key="missing_table",
                        text="keyed to a table absent from the composite",
                        kind=DomainKnowledgeKind.POLICY.value,
                        referenced_entities=frozenset({"missing_table"}),
                    )
                ),
                DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(
                        key="leak",
                        text="do not expose orders.ssn values",
                        kind=DomainKnowledgeKind.CAVEAT.value,
                        referenced_entities=frozenset({"orders.ssn"}),
                    )
                ),
            ),
        ),
    )
    fed_notes = "ARR means annual recurring revenue for orders."
    fed_dk = DomainKnowledgeEntry.normalize(
        DomainKnowledgeEntry(key="arr", text="federation arr", kind=DomainKnowledgeKind.GLOSSARY.value)
    )
    fed_sk = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.RELATION.value,
            text="orders entity",
            referenced_entities=frozenset({"orders"}),
        )
    )
    with (
        patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=False),
        patch(
            "aetherdialect._main_spaces.resolve_knowledge_extraction_for_schema",
            return_value=((fed_dk,), (fed_sk,)),
        ),
    ):
        final_dk = MainSpaceOps.enrich_federation_composite_knowledge(
            graph,
            member_domain_knowledge=member_dk,
            member_structural_knowledge=(),
            notes_content=fed_notes,
            all_schema_table_names={"orders", "missing_table"},
        )
    texts = {e.key: e.text for e in final_dk}
    assert "federation arr" in texts.get("arr", "")
    assert "missing_table" not in texts
    assert "leak" not in texts
    assert graph.structural_knowledge
    assert any("orders" in f.text for f in graph.structural_knowledge)
