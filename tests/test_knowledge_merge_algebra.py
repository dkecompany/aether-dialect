"""Algebraic and federation-remap tests for deterministic knowledge merge."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from aetherdialect._knowledge_merge import (
    KnowledgeMergeAuthority,
    merge_domain_knowledge_federation_peers,
    merge_domain_knowledge_layers,
    merge_domain_knowledge_space_over_engine,
    merge_structural_knowledge_federation_peers,
    merge_structural_knowledge_layers,
)


def _dk(key: str, text: str) -> DomainKnowledgeEntry:
    return DomainKnowledgeEntry.normalize(
        DomainKnowledgeEntry(key=key, text=text, kind=DomainKnowledgeKind.GLOSSARY.value)
    )


def _sk(kind: str, text: str, *refs: str) -> StructuralKnowledgeFact:
    return StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=kind,
            text=text,
            referenced_entities=frozenset(refs),
        )
    )


def test_merge_domain_commutative_for_non_conflicting_records() -> None:
    a = (_dk("alpha", "alpha definition"),)
    b = (_dk("beta", "beta definition"),)
    left = merge_domain_knowledge_layers(
        (("engine", a), ("space", b)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )[0]
    right = merge_domain_knowledge_layers(
        (("space", b), ("engine", a)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )[0]
    assert {e.key for e in left} == {e.key for e in right}


def test_merge_domain_associative() -> None:
    layer_a = (_dk("a", "a text"),)
    layer_b = (_dk("b", "b text"),)
    layer_c = (_dk("c", "c text"),)
    ab_c = merge_domain_knowledge_layers(
        (("x", layer_a), ("y", layer_b), ("z", layer_c)),
        authority=KnowledgeMergeAuthority.PEER_EQUAL,
    )[0]
    a_bc = merge_domain_knowledge_layers(
        (("x", layer_a), ("y", layer_b + layer_c)),
        authority=KnowledgeMergeAuthority.PEER_EQUAL,
    )[0]
    assert {e.key for e in ab_c} == {e.key for e in a_bc}


def test_merge_domain_idempotent() -> None:
    layer = (_dk("k", "same"), _dk("other", "other"))
    once = merge_domain_knowledge_layers(
        (("engine", layer),),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )[0]
    twice = merge_domain_knowledge_layers(
        (("engine", layer), ("engine", layer)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )[0]
    assert {(e.key, e.text) for e in once} == {(e.key, e.text) for e in twice}


def test_master_authoritative_prefers_engine_on_same_identity() -> None:
    engine = (_dk("active", "engine active definition"), _dk("fy", "engine fy only"))
    space = (_dk("active", "space active overlay"),)
    merged = merge_domain_knowledge_space_over_engine(engine, space)
    by_key = {e.key: e.text for e in merged}
    assert by_key["active"] == "engine active definition"
    assert by_key["fy"] == "engine fy only"


def test_peer_conflict_fails_build() -> None:
    a = (_dk("arr", "member a arr"),)
    b = (_dk("arr", "member b arr"),)
    with pytest.raises(ConfigError, match="incompatible domain knowledge merge"):
        merge_domain_knowledge_federation_peers((("member_a", a), ("member_b", b)))


def test_peer_structural_conflict_fails_build() -> None:
    fact_a = (_sk(StructuralKnowledgeKind.RELATION.value, "orders entity", "orders"),)
    fact_b = (_sk(StructuralKnowledgeKind.RELATION.value, "conflicting orders entity", "orders"),)
    with pytest.raises(ConfigError, match="incompatible structural knowledge merge"):
        merge_structural_knowledge_federation_peers((("a", fact_a), ("b", fact_b)))


def test_federation_logical_collapse_remaps_reference_sets() -> None:
    fact = _sk(StructuralKnowledgeKind.RELATION.value, "orders entity", "storefront.orders")
    merged, _stats = merge_structural_knowledge_layers(
        (("storefront", (fact,)),),
        authority=KnowledgeMergeAuthority.PEER_EQUAL,
        entity_map={"storefront.orders": "orders", "storefront": "orders"},
    )
    assert len(merged) == 1
    assert merged[0].referenced_entities == frozenset({"orders"})


def test_federation_remap_failure_raises() -> None:
    fact = _sk(StructuralKnowledgeKind.FIELD.value, "id column", "missing.id")
    with pytest.raises(ConfigError, match="failed to remap"):
        merge_structural_knowledge_layers(
            (("member", (fact,)),),
            authority=KnowledgeMergeAuthority.PEER_EQUAL,
            entity_map={"other.id": "id"},
        )


def test_declared_value_set_payload_widens_reconcilably() -> None:
    left = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
            text="status vocabulary",
            referenced_entities=frozenset({"orders.status"}),
            payload={"values": ["open", "closed"]},
        )
    )
    right = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=StructuralKnowledgeKind.DECLARED_VALUE_SET.value,
            text="status vocabulary",
            referenced_entities=frozenset({"orders.status"}),
            payload={"values": ["pending"]},
        )
    )
    merged, stats = merge_structural_knowledge_layers(
        (("engine", (left,)), ("space", (right,))),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )
    assert len(merged) == 1
    assert set(merged[0].payload["values"]) == {"open", "closed", "pending"}
    assert stats.reconcilable == 1
