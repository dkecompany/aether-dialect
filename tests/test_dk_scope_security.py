"""Domain-knowledge scope security: caller-scoped domain_context and reference-set enforcement."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants_runtime import SANDBOX_MEMBER_SPACE_TABLES
from aetherdialect._contracts_base import ConfigError, DomainKnowledgeEntry, KnowledgeScope
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import domain_context_payload
from tests.support.egress_assert import assert_no_forbidden_identifiers, forbidden_identifiers_for_scope

_ALL_SANDBOX_TABLES = frozenset().union(*SANDBOX_MEMBER_SPACE_TABLES.values())


def _master_schema_graph() -> SchemaGraph:
    tables = {
        name: TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
        for name in sorted(_ALL_SANDBOX_TABLES)
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="dk_scope_security",
        effective_structural_hash="dk_scope_security",
    )


def _domain_context_export(entries: tuple[DomainKnowledgeEntry, ...]) -> str:
    payload = domain_context_payload(entries)
    return json.dumps(payload or [], sort_keys=True)


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_member_domain_context_excludes_out_of_scope_engine_dk(member: str) -> None:
    """Each member scope must drop engine DK whose reference set names foreign tables."""
    graph = _master_schema_graph()
    visible = SANDBOX_MEMBER_SPACE_TABLES[member]
    scope = KnowledgeScope.from_visible_tables(graph, visible)
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    foreign_table = sorted(_ALL_SANDBOX_TABLES - visible)[0]

    engine_entries = (
        DomainKnowledgeEntry(
            key="arr",
            text="Annual recurring revenue is a portfolio metric.",
            kind="glossary",
            referenced_entities=frozenset(),
        ),
        DomainKnowledgeEntry(
            key="foreign_fact",
            text=f"The {foreign_table} table stores cross-domain operational data.",
            kind="glossary",
            referenced_entities=frozenset({foreign_table}),
        ),
    )
    scoped = MainSpaceOps.derive_caller_scoped_domain_knowledge(
        engine_entries=engine_entries,
        schema=graph,
        space_tables=set(visible),
    )
    payload = domain_context_payload(scoped)
    assert payload is not None
    assert {row["key"] for row in payload} == {"arr"}
    assert_no_forbidden_identifiers(payload, forbidden)


@pytest.mark.fast
def test_glossary_entry_naming_forbidden_table_is_dropped() -> None:
    """Entries declaring an out-of-scope table in referenced_entities are dropped whole."""
    graph = _master_schema_graph()
    visible = SANDBOX_MEMBER_SPACE_TABLES["crm"]
    foreign_table = sorted(_ALL_SANDBOX_TABLES - visible)[0]
    entry = DomainKnowledgeEntry(
        key="leak",
        text="Operational context for reporting.",
        kind="glossary",
        referenced_entities=frozenset({foreign_table}),
    )
    scoped = MainSpaceOps.secure_domain_knowledge_for_visibility(
        (entry,),
        security_schema=graph,
        visible_table_names=set(visible),
        all_schema_table_names=set(graph.tables.keys()),
    )
    assert scoped == ()


@pytest.mark.fast
def test_reference_set_text_mismatch_is_build_error() -> None:
    from aetherdialect._contracts_base import DomainKnowledgeState

    graph = _master_schema_graph()
    entry = DomainKnowledgeEntry(
        key="bad",
        text="payment amounts are charged totals.",
        kind="glossary",
        referenced_entities=frozenset(),
    )
    with pytest.raises(ConfigError, match="not declared in referenced_entities"):
        DomainKnowledgeState.validate_entries((entry,), graph)


@pytest.mark.fast
def test_normalize_domain_knowledge_entries_requires_referenced_entities() -> None:
    with pytest.raises(ConfigError, match="referenced_entities"):
        MainSpaceOps.normalize_domain_knowledge_entries(
            [{"key": "arr", "kind": "glossary", "text": "Annual recurring revenue."}]
        )


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_egress_matrix_domain_context_clean_for_member_scope(member: str) -> None:
    """domain_context export must not name identifiers outside the member table set."""
    graph = _master_schema_graph()
    visible = SANDBOX_MEMBER_SPACE_TABLES[member]
    scope = KnowledgeScope.from_visible_tables(graph, visible)
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    allowed_table = sorted(visible)[0]
    entries = (
        DomainKnowledgeEntry(
            key="in_scope",
            text=f"Counts rows on {allowed_table}.",
            kind="metric",
            referenced_entities=frozenset({allowed_table}),
        ),
    )
    scoped = MainSpaceOps.derive_caller_scoped_domain_knowledge(
        engine_entries=entries,
        schema=graph,
        space_tables=set(visible),
    )
    export = _domain_context_export(scoped)
    assert_no_forbidden_identifiers(json.loads(export), forbidden)
