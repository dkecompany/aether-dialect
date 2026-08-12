"""KnowledgeScope derivation: factory helpers, DK subset filtering, and out-of-scope description verification."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, DomainKnowledgeEntry, KnowledgeScope, SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._schema_graph import subset_schema_graph_for_visible_tables
from aetherdialect._schema_profile import (
    out_of_scope_description_tokens,
    raise_if_flat_descriptions_name_out_of_scope_entities,
    raise_if_schema_graph_descriptions_name_out_of_scope_entities,
)


def _two_table_schema() -> SchemaGraph:
    customer = TableMetadata(
        name="customer",
        columns={
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer"),
            "email": ColumnMetadata(name="email", data_type="text"),
            "ssn": ColumnMetadata(name="ssn", data_type="text", sensitivity=SensitivityClassification.HIDDEN),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
    )
    payroll = TableMetadata(
        name="payroll",
        columns={
            "payroll_id": ColumnMetadata(name="payroll_id", data_type="integer"),
            "salary": ColumnMetadata(name="salary", data_type="numeric"),
        },
        primary_key=["payroll_id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"customer": customer, "payroll": payroll},
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g",
    )


@pytest.mark.fast
def test_scope_from_schema_graph_is_unrestricted() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_schema_graph(graph)
    assert scope.tables == frozenset({"customer", "payroll"})
    assert scope.contains("customer")
    assert scope.contains("payroll")
    assert scope.contains("customer.email")
    assert scope.contains("payroll.salary")


@pytest.mark.fast
def test_scope_from_visible_tables_excludes_hidden_columns_by_default() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"})
    assert scope.tables == frozenset({"customer"})
    assert scope.contains("customer.email")
    assert not scope.contains("customer.ssn")
    assert not scope.contains("payroll")
    assert not scope.contains("payroll.salary")


@pytest.mark.fast
def test_scope_from_visible_tables_can_keep_sensitive_columns() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"}, exclude_sensitive=False)
    assert scope.contains("customer.ssn")


@pytest.mark.fast
def test_scope_from_engine_context_none_is_unrestricted() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_engine_context(graph, None)
    assert scope.tables == frozenset({"customer", "payroll"})


@pytest.mark.fast
def test_scope_from_engine_context_narrows_to_visible_objects() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_engine_context(graph, frozenset({"customer"}))
    assert scope.tables == frozenset({"customer"})
    assert not scope.contains("payroll")


@pytest.mark.fast
def test_scope_from_space_snapshot() -> None:
    snapshot = {"tables": ["customer"], "columns": ["customer.email"]}
    scope = KnowledgeScope.from_space_snapshot(snapshot)
    assert scope.tables == frozenset({"customer"})
    assert scope.columns == frozenset({"customer.email"})


@pytest.mark.fast
def test_scope_union_combines_member_slices() -> None:
    left = KnowledgeScope(tables=frozenset({"customer"}), columns=frozenset({"customer.email"}))
    right = KnowledgeScope(tables=frozenset({"payroll"}), columns=frozenset({"payroll.salary"}))
    combined = KnowledgeScope.union((left, right))
    assert combined.tables == frozenset({"customer", "payroll"})
    assert combined.covers({"customer", "payroll", "customer.email", "payroll.salary"})


@pytest.mark.fast
def test_covers_is_subset_arithmetic() -> None:
    scope = KnowledgeScope(tables=frozenset({"customer"}))
    assert scope.covers({"customer"})
    assert scope.covers(())
    assert not scope.covers({"customer", "payroll"})


@pytest.mark.fast
def test_domain_knowledge_entry_kept_when_referenced_entities_in_scope() -> None:
    scope = KnowledgeScope(tables=frozenset({"customer"}))
    entry = DomainKnowledgeEntry(
        key="churn",
        text="Customers who cancel are marked inactive.",
        referenced_entities=frozenset({"customer"}),
    )
    assert entry.in_scope(scope)


@pytest.mark.fast
def test_domain_knowledge_entry_dropped_when_referenced_entities_outside_scope() -> None:
    scope = KnowledgeScope(tables=frozenset({"customer"}))
    entry = DomainKnowledgeEntry(
        key="comp",
        text="Compensation bands drive salary bands.",
        referenced_entities=frozenset({"payroll"}),
    )
    assert not entry.in_scope(scope)


@pytest.mark.fast
def test_domain_knowledge_entry_with_no_referenced_entities_is_in_scope() -> None:
    scope = KnowledgeScope(tables=frozenset({"customer"}))
    entry = DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", referenced_entities=frozenset())
    assert entry.in_scope(scope)


@pytest.mark.fast
def test_filter_domain_knowledge_for_visibility_uses_referenced_entities_subset() -> None:
    kept_entry = DomainKnowledgeEntry(
        key="churn", text="Customers who cancel are marked inactive.", referenced_entities=frozenset({"customer"})
    )
    dropped_entry = DomainKnowledgeEntry(
        key="comp", text="Compensation bands drive salary bands.", referenced_entities=frozenset({"payroll"})
    )
    concept_entry = DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", referenced_entities=frozenset())
    filtered = MainSpaceOps.filter_domain_knowledge_for_visibility(
        (kept_entry, dropped_entry, concept_entry),
        visible_table_names={"customer"},
        all_schema_table_names={"customer", "payroll"},
    )
    assert {e.key for e in filtered} == {"churn", "arr"}


@pytest.mark.fast
def test_filter_domain_knowledge_for_visibility_fails_closed_on_unknown_table_key() -> None:
    """An entry whose reference set names a table absent from the caller's visible set must drop."""
    stale_entry = DomainKnowledgeEntry(
        key="archived_orders_total",
        text="Legacy order total note.",
        referenced_entities=frozenset({"archived_orders"}),
    )
    concept_entry = DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", referenced_entities=frozenset())
    filtered = MainSpaceOps.filter_domain_knowledge_for_visibility(
        (stale_entry, concept_entry),
        visible_table_names={"customer"},
        all_schema_table_names={"customer"},
    )
    assert {e.key for e in filtered} == {"arr"}


@pytest.mark.fast
def test_out_of_scope_description_tokens_flags_other_table_but_not_shared_column_name() -> None:
    graph = _two_table_schema()
    graph.tables["customer"].columns["shared_id"] = ColumnMetadata(name="shared_id", data_type="integer")
    graph.tables["payroll"].columns["shared_id"] = ColumnMetadata(name="shared_id", data_type="integer")
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"})
    tokens = out_of_scope_description_tokens(graph, scope)
    assert "payroll" in tokens
    assert "salary" in tokens
    assert "shared_id" not in tokens


@pytest.mark.fast
def test_raise_if_schema_graph_descriptions_name_out_of_scope_entities_hard_fails() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"})
    tokens = out_of_scope_description_tokens(graph, scope)
    scoped = subset_schema_graph_for_visible_tables(graph, {"customer"}, prefer_base_description=False)
    scoped.tables["customer"].description = "Related to the payroll table for compensation history."
    with pytest.raises(ConfigError):
        raise_if_schema_graph_descriptions_name_out_of_scope_entities(scoped, tokens)


@pytest.mark.fast
def test_raise_if_schema_graph_descriptions_name_out_of_scope_entities_passes_clean_text() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"})
    tokens = out_of_scope_description_tokens(graph, scope)
    scoped = subset_schema_graph_for_visible_tables(graph, {"customer"}, prefer_base_description=False)
    scoped.tables["customer"].description = "People who purchase products."
    raise_if_schema_graph_descriptions_name_out_of_scope_entities(scoped, tokens)


@pytest.mark.fast
def test_raise_if_flat_descriptions_name_out_of_scope_entities_hard_fails() -> None:
    graph = _two_table_schema()
    scope = KnowledgeScope.from_visible_tables(graph, {"customer"})
    tokens = out_of_scope_description_tokens(graph, scope)
    table_descriptions = {"customer": "Linked to payroll for compensation history."}
    column_meta: dict[str, dict[str, str]] = {}
    with pytest.raises(ConfigError):
        raise_if_flat_descriptions_name_out_of_scope_entities(table_descriptions, column_meta, tokens)


@pytest.mark.fast
def test_subset_schema_graph_for_visible_tables_clears_out_of_scope_base_description() -> None:
    """No LLM path at this stage: an out-of-scope base_description is blanked, never leaked or rewritten."""
    graph = _two_table_schema()
    graph.tables["customer"].base_description = "Related to payroll compensation records."
    scoped = subset_schema_graph_for_visible_tables(graph, {"customer"})
    assert scoped.tables["customer"].description == ""


@pytest.mark.fast
def test_subset_schema_graph_for_visible_tables_keeps_clean_base_description() -> None:
    graph = _two_table_schema()
    graph.tables["customer"].base_description = "People who purchase products."
    scoped = subset_schema_graph_for_visible_tables(graph, {"customer"})
    assert scoped.tables["customer"].description == "People who purchase products."
