"""Build-time structural claim verification against profiling."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, StructuralKnowledgeFact, StructuralKnowledgeKind
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._knowledge_claims import (
    ClaimVerificationOutcome,
    finalize_structural_knowledge_claims,
    verify_structural_knowledge_claim,
)


def _schema_with_status_column() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "status": ColumnMetadata(
                        name="status",
                        data_type="text",
                        distinct_count=2,
                        distinct_from_sample=False,
                        value_overlap_sample=["open", "closed"],
                    ),
                },
                primary_key=["status"],
                foreign_keys=[],
                row_count=100,
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g",
    )


@pytest.mark.fast
def test_declared_value_set_confirmed_when_sample_subset() -> None:
    schema = _schema_with_status_column()
    fact = StructuralKnowledgeFact(
        kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
        text="status is open or closed",
        referenced_entities=frozenset({"orders.status"}),
        payload={"values": ["open", "closed"]},
    )
    result = verify_structural_knowledge_claim(fact, schema)
    assert result.outcome == ClaimVerificationOutcome.CONFIRMED


@pytest.mark.fast
def test_declared_value_set_contradicted_when_distinct_exceeds_declared() -> None:
    schema = _schema_with_status_column()
    schema.tables["orders"].columns["status"].distinct_count = 5
    fact = StructuralKnowledgeFact(
        kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
        text="status is open or closed",
        referenced_entities=frozenset({"orders.status"}),
        payload={"values": ["open", "closed"]},
    )
    result = verify_structural_knowledge_claim(fact, schema)
    assert result.outcome == ClaimVerificationOutcome.CONTRADICTED


@pytest.mark.fast
def test_unit_of_measure_is_always_unverifiable() -> None:
    schema = _schema_with_status_column()
    fact = StructuralKnowledgeFact(
        kind=StructuralKnowledgeKind.UNIT_OF_MEASURE.value,
        text="amount is USD",
        referenced_entities=frozenset({"orders.status"}),
        payload={"unit": "USD", "summable": True},
    )
    result = verify_structural_knowledge_claim(fact, schema)
    assert result.outcome == ClaimVerificationOutcome.UNVERIFIABLE


@pytest.mark.fast
def test_finalize_raises_on_contradiction() -> None:
    schema = _schema_with_status_column()
    schema.tables["orders"].columns["status"].distinct_count = 9
    fact = StructuralKnowledgeFact(
        kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
        text="status closed set",
        referenced_entities=frozenset({"orders.status"}),
        payload={"values": ["open", "closed"]},
    )
    with pytest.raises(ConfigError, match="contradicted"):
        finalize_structural_knowledge_claims(schema, (fact,))


@pytest.mark.fast
def test_sensitive_anchor_scrubbed_before_validation() -> None:
    from aetherdialect._contracts_base import SensitivityClassification

    schema = _schema_with_status_column()
    schema.tables["orders"].columns["status"].sensitivity = SensitivityClassification.HIDDEN
    fact = StructuralKnowledgeFact(
        kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
        text="status vocabulary",
        referenced_entities=frozenset({"orders.status"}),
        payload={"values": ["open"]},
    )
    finalized = finalize_structural_knowledge_claims(schema, (fact,))
    assert finalized == ()
