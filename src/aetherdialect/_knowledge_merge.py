"""Deterministic kind-dispatched knowledge merge (identity by kind+refs or kind+key)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    KnowledgeMergeAuthority,
    KnowledgeMergeDisposition,
    KnowledgeMergeStats,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)


def _merge_stats_record(
    stats: KnowledgeMergeStats,
    disposition: KnowledgeMergeDisposition,
    *,
    identity: str,
    sources: frozenset[str],
) -> KnowledgeMergeStats:
    prov = dict(stats.provenance)
    prov[identity] = sources
    return KnowledgeMergeStats(
        identical=stats.identical + (1 if disposition == KnowledgeMergeDisposition.IDENTICAL else 0),
        reconcilable=stats.reconcilable + (1 if disposition == KnowledgeMergeDisposition.RECONCILABLE else 0),
        incompatible=stats.incompatible + (1 if disposition == KnowledgeMergeDisposition.INCOMPATIBLE else 0),
        provenance=prov,
    )


def domain_knowledge_identity(entry: DomainKnowledgeEntry) -> tuple[str, str]:
    """Unanchored identity is ``(kind, key)``; anchored uses reference set (see structural)."""
    if entry.referenced_entities:
        refs = tuple(sorted(entry.referenced_entities))
        return (str(entry.kind), f"refs:{refs!r}")
    return (str(entry.kind), f"key:{entry.key}")


def structural_knowledge_identity(fact: StructuralKnowledgeFact) -> tuple[str, str]:
    """Anchored identity is ``(kind, reference_set)`` — never text."""
    refs = tuple(sorted(fact.referenced_entities))
    return (str(fact.kind), f"refs:{refs!r}")


def remap_entity_reference(entity: str, entity_map: Mapping[str, str]) -> str:
    """Remap one schema entity name; raises when *entity* is absent from *entity_map*."""
    text = str(entity).strip()
    if not text:
        raise ConfigError("knowledge merge entity remap requires non-empty entity name")
    if text not in entity_map:
        raise ConfigError(f"knowledge merge entity {text!r} failed to remap through logical collapse")
    mapped = str(entity_map[text]).strip()
    if not mapped:
        raise ConfigError(f"knowledge merge entity {text!r} remapped to empty name")
    return mapped


def remap_referenced_entities(
    referenced_entities: frozenset[str],
    entity_map: Mapping[str, str],
) -> frozenset[str]:
    """Remap every reference through *entity_map*; failed remap is an error."""
    if not referenced_entities:
        return frozenset()
    return frozenset(remap_entity_reference(ref, entity_map) for ref in referenced_entities)


def remap_domain_knowledge_entry(
    entry: DomainKnowledgeEntry,
    entity_map: Mapping[str, str] | None,
) -> DomainKnowledgeEntry:
    if not entity_map or not entry.referenced_entities:
        return entry
    return DomainKnowledgeEntry.normalize(
        DomainKnowledgeEntry(
            key=entry.key,
            kind=entry.kind,
            text=entry.text,
            referenced_entities=remap_referenced_entities(entry.referenced_entities, entity_map),
        )
    )


def remap_structural_knowledge_fact(
    fact: StructuralKnowledgeFact,
    entity_map: Mapping[str, str] | None,
) -> StructuralKnowledgeFact:
    if not entity_map or not fact.referenced_entities:
        return fact
    return StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=fact.kind,
            text=fact.text,
            referenced_entities=remap_referenced_entities(fact.referenced_entities, entity_map),
            payload=fact.payload,
        )
    )


def _reconcile_declared_value_set_payload(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_values = left.get("values")
    right_values = right.get("values")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return None
    merged = sorted({str(v).strip() for v in (*left_values, *right_values) if str(v).strip()}, key=str)
    if not merged:
        return None
    return {"values": merged}


def reconcile_structural_payload(
    kind: str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Kind-dispatched payload widen; ``None`` means incompatible."""
    if left == right:
        return left
    if left is None and right is None:
        return None
    if left is None or right is None:
        return None
    if kind == StructuralKnowledgeKind.DECLARED_VALUE_SET.value:
        return _reconcile_declared_value_set_payload(left, right)
    return None


def _merge_pair_domain(
    left: DomainKnowledgeEntry,
    right: DomainKnowledgeEntry,
    *,
    authority: KnowledgeMergeAuthority,
) -> tuple[DomainKnowledgeEntry, KnowledgeMergeDisposition]:
    if (
        left.kind == right.kind
        and left.key == right.key
        and left.text == right.text
        and left.referenced_entities == right.referenced_entities
    ):
        return left, KnowledgeMergeDisposition.IDENTICAL
    if authority == KnowledgeMergeAuthority.MASTER_AUTHORITATIVE:
        return left, KnowledgeMergeDisposition.INCOMPATIBLE
    raise ConfigError(
        f"incompatible domain knowledge merge for identity {domain_knowledge_identity(left)!r}: conflicting sources"
    )


def _merge_pair_structural(
    left: StructuralKnowledgeFact,
    right: StructuralKnowledgeFact,
    *,
    authority: KnowledgeMergeAuthority,
) -> tuple[StructuralKnowledgeFact, KnowledgeMergeDisposition]:
    if (
        left.kind == right.kind
        and left.text == right.text
        and left.referenced_entities == right.referenced_entities
        and left.payload == right.payload
    ):
        return left, KnowledgeMergeDisposition.IDENTICAL
    if left.kind != right.kind or left.text != right.text or left.referenced_entities != right.referenced_entities:
        if authority == KnowledgeMergeAuthority.MASTER_AUTHORITATIVE:
            return left, KnowledgeMergeDisposition.INCOMPATIBLE
        raise ConfigError(
            f"incompatible structural knowledge merge for identity {structural_knowledge_identity(left)!r}: "
            f"conflicting sources"
        )
    payload = reconcile_structural_payload(left.kind, left.payload, right.payload)
    if payload is None:
        if authority == KnowledgeMergeAuthority.MASTER_AUTHORITATIVE:
            return left, KnowledgeMergeDisposition.INCOMPATIBLE
        raise ConfigError(
            f"incompatible structural knowledge payload merge for kind {left.kind!r} "
            f"identity {structural_knowledge_identity(left)!r}"
        )
    merged = StructuralKnowledgeFact.normalize(
        StructuralKnowledgeFact(
            kind=left.kind,
            text=left.text,
            referenced_entities=left.referenced_entities,
            payload=payload,
        )
    )
    return merged, KnowledgeMergeDisposition.RECONCILABLE


def _source_priority(sources: Sequence[str], label: str) -> int:
    try:
        return sources.index(label)
    except ValueError:
        return len(sources)


def _fold_labeled_group_domain(
    group: Sequence[tuple[str, DomainKnowledgeEntry]],
    *,
    authority: KnowledgeMergeAuthority,
    source_order: Sequence[str],
) -> tuple[DomainKnowledgeEntry, KnowledgeMergeDisposition, frozenset[str]]:
    labels = frozenset(label for label, _ in group)
    ordered = sorted(group, key=lambda item: _source_priority(source_order, item[0]))
    acc, disposition = ordered[0][1], KnowledgeMergeDisposition.IDENTICAL
    for _label, entry in ordered[1:]:
        acc, step = _merge_pair_domain(acc, entry, authority=authority)
        if step == KnowledgeMergeDisposition.INCOMPATIBLE and disposition == KnowledgeMergeDisposition.IDENTICAL:
            disposition = KnowledgeMergeDisposition.INCOMPATIBLE
        elif step == KnowledgeMergeDisposition.RECONCILABLE:
            disposition = KnowledgeMergeDisposition.RECONCILABLE
        elif step == KnowledgeMergeDisposition.IDENTICAL and disposition == KnowledgeMergeDisposition.IDENTICAL:
            disposition = KnowledgeMergeDisposition.IDENTICAL
    return acc, disposition, labels


def _fold_labeled_group_structural(
    group: Sequence[tuple[str, StructuralKnowledgeFact]],
    *,
    authority: KnowledgeMergeAuthority,
    source_order: Sequence[str],
) -> tuple[StructuralKnowledgeFact, KnowledgeMergeDisposition, frozenset[str]]:
    labels = frozenset(label for label, _ in group)
    ordered = sorted(group, key=lambda item: _source_priority(source_order, item[0]))
    acc, disposition = ordered[0][1], KnowledgeMergeDisposition.IDENTICAL
    for _label, fact in ordered[1:]:
        acc, step = _merge_pair_structural(acc, fact, authority=authority)
        if step == KnowledgeMergeDisposition.INCOMPATIBLE and disposition == KnowledgeMergeDisposition.IDENTICAL:
            disposition = KnowledgeMergeDisposition.INCOMPATIBLE
        elif step == KnowledgeMergeDisposition.RECONCILABLE:
            disposition = KnowledgeMergeDisposition.RECONCILABLE
        elif step == KnowledgeMergeDisposition.IDENTICAL and disposition == KnowledgeMergeDisposition.IDENTICAL:
            disposition = KnowledgeMergeDisposition.IDENTICAL
    return acc, disposition, labels


def merge_domain_knowledge_layers(
    sources: Sequence[tuple[str, Sequence[DomainKnowledgeEntry]]],
    *,
    authority: KnowledgeMergeAuthority,
    entity_map: Mapping[str, str] | None = None,
) -> tuple[tuple[DomainKnowledgeEntry, ...], KnowledgeMergeStats]:
    """Merge labeled domain-knowledge layers with explicit collision dispositions."""
    if not sources:
        return (), KnowledgeMergeStats()
    source_order = [label for label, _ in sources]
    grouped: dict[tuple[str, str], list[tuple[str, DomainKnowledgeEntry]]] = defaultdict(list)
    for label, entries in sources:
        for entry in entries:
            remapped = remap_domain_knowledge_entry(entry, entity_map)
            grouped[domain_knowledge_identity(remapped)].append((label, remapped))
    merged: list[DomainKnowledgeEntry] = []
    stats = KnowledgeMergeStats()
    for identity in sorted(grouped.keys()):
        group = grouped[identity]
        if len(group) == 1:
            merged.append(group[0][1])
            continue
        record, disposition, labels = _fold_labeled_group_domain(
            group,
            authority=authority,
            source_order=source_order,
        )
        merged.append(record)
        stats = _merge_stats_record(stats, disposition, identity="|".join(identity), sources=labels)
    merged_sorted = tuple(sorted(merged, key=lambda e: (e.kind, e.key, e.text)))
    return merged_sorted, stats


def merge_structural_knowledge_layers(
    sources: Sequence[tuple[str, Sequence[StructuralKnowledgeFact]]],
    *,
    authority: KnowledgeMergeAuthority,
    entity_map: Mapping[str, str] | None = None,
) -> tuple[tuple[StructuralKnowledgeFact, ...], KnowledgeMergeStats]:
    """Merge labeled structural-knowledge layers with explicit collision dispositions."""
    if not sources:
        return (), KnowledgeMergeStats()
    source_order = [label for label, _ in sources]
    grouped: dict[tuple[str, str], list[tuple[str, StructuralKnowledgeFact]]] = defaultdict(list)
    for label, facts in sources:
        for fact in facts:
            remapped = remap_structural_knowledge_fact(fact, entity_map)
            grouped[structural_knowledge_identity(remapped)].append((label, remapped))
    merged: list[StructuralKnowledgeFact] = []
    stats = KnowledgeMergeStats()
    for identity in sorted(grouped.keys()):
        group = grouped[identity]
        if len(group) == 1:
            merged.append(group[0][1])
            continue
        record, disposition, labels = _fold_labeled_group_structural(
            group,
            authority=authority,
            source_order=source_order,
        )
        merged.append(record)
        stats = _merge_stats_record(stats, disposition, identity="|".join(identity), sources=labels)
    merged_sorted = tuple(sorted(merged, key=lambda f: (f.kind, tuple(sorted(f.referenced_entities)), f.text)))
    return merged_sorted, stats


def merge_domain_knowledge_notes_overlay(
    base_entries: Sequence[DomainKnowledgeEntry],
    overlay_entries: Sequence[DomainKnowledgeEntry],
) -> tuple[DomainKnowledgeEntry, ...]:
    """Notes overlay merge: overlay authoritative on disagreement; base additive only."""
    merged, _stats = merge_domain_knowledge_layers(
        (("overlay", overlay_entries), ("base", base_entries)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )
    return merged


def merge_structural_knowledge_notes_overlay(
    base_facts: Sequence[StructuralKnowledgeFact],
    overlay_facts: Sequence[StructuralKnowledgeFact],
) -> tuple[StructuralKnowledgeFact, ...]:
    """Notes overlay merge: overlay authoritative on disagreement; base additive only."""
    merged, _stats = merge_structural_knowledge_layers(
        (("overlay", overlay_facts), ("base", base_facts)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )
    return merged


def merge_domain_knowledge_space_over_engine(
    engine_entries: Sequence[DomainKnowledgeEntry],
    space_entries: Sequence[DomainKnowledgeEntry],
) -> tuple[DomainKnowledgeEntry, ...]:
    """Master-authoritative merge: engine wins on disagreement; space is additive."""
    merged, _stats = merge_domain_knowledge_layers(
        (("engine", engine_entries), ("space", space_entries)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )
    return merged


def merge_domain_knowledge_federation_peers(
    members: Sequence[tuple[str, Sequence[DomainKnowledgeEntry]]],
) -> tuple[DomainKnowledgeEntry, ...]:
    """Equal-peer merge; genuine disagreement fails the build."""
    merged, _stats = merge_domain_knowledge_layers(members, authority=KnowledgeMergeAuthority.PEER_EQUAL)
    return merged


def merge_structural_knowledge_space_over_engine(
    engine_facts: Sequence[StructuralKnowledgeFact],
    space_facts: Sequence[StructuralKnowledgeFact],
) -> tuple[StructuralKnowledgeFact, ...]:
    """Master-authoritative merge: engine wins on disagreement; space is additive."""
    merged, _stats = merge_structural_knowledge_layers(
        (("engine", engine_facts), ("space", space_facts)),
        authority=KnowledgeMergeAuthority.MASTER_AUTHORITATIVE,
    )
    return merged


def merge_structural_knowledge_federation_peers(
    members: Sequence[tuple[str, Sequence[StructuralKnowledgeFact]]],
) -> tuple[StructuralKnowledgeFact, ...]:
    """Equal-peer merge; genuine disagreement fails the build."""
    merged, _stats = merge_structural_knowledge_layers(members, authority=KnowledgeMergeAuthority.PEER_EQUAL)
    return merged
