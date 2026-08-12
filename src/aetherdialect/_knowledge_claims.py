"""Build-time verification of operator structural-knowledge claims against profiling."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._config import PolicyConfig
from ._constants import (
    KNOWLEDGE_EXPORT_FORMAT_VERSION,
    KNOWLEDGE_OPERATOR_REPORT_FILENAME,
    SCHEMA_CLASSIFY_ERROR_DETAIL_CAP,
)
from ._contracts_base import (
    ClaimVerificationOutcome,
    ClaimVerificationResult,
    ConfigError,
    DomainKnowledgeEntry,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from ._contracts_schema import ColumnMetadata, SchemaGraph
from ._knowledge_join import attach_structural_fanout_metadata


def filter_schema_anchored_structural_knowledge(
    facts: Sequence[StructuralKnowledgeFact],
    schema: SchemaGraph,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Security filter: drop structural facts that name schema-known sensitive columns in text."""
    return tuple(fact for fact in facts if not DomainKnowledgeEntry.sensitive_column_references(fact.text, schema))


def _column_profiled_unique(schema: SchemaGraph, table: str, column: str) -> bool | None:
    tmeta = schema.tables.get(table)
    if tmeta is None:
        return None
    pk = list(tmeta.primary_key or [])
    if pk == [column]:
        return True
    cm = tmeta.columns.get(column)
    if cm is None:
        return None
    if cm.is_unique:
        return True
    row_count = int(tmeta.row_count or 0)
    distinct = int(cm.distinct_count or 0)
    if row_count > 0 and distinct > 0:
        return distinct >= row_count
    return None


def _join_columns_profiled_unique(schema: SchemaGraph, table: str, cols: list[str]) -> bool | None:
    clean = [c.strip() for c in cols if c.strip()]
    if not clean:
        return None
    tmeta = schema.tables.get(table)
    if tmeta is None:
        return None
    pk = list(tmeta.primary_key or [])
    if pk == clean:
        return True
    if len(clean) == 1:
        return _column_profiled_unique(schema, table, clean[0])
    return None


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, sort_keys: bool = False, indent: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=sort_keys, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _qualified_column_refs(referenced_entities: frozenset[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for entity in referenced_entities:
        text = str(entity).strip()
        if "." not in text:
            continue
        table_name, column_name = text.split(".", 1)
        if table_name and column_name:
            out.append((table_name, column_name))
    return out


def _column_metadata(schema: SchemaGraph, table: str, column: str) -> ColumnMetadata | None:
    table_meta = schema.tables.get(table)
    if table_meta is None:
        return None
    return table_meta.columns.get(column)


def _verify_declared_value_set(
    fact: StructuralKnowledgeFact,
    schema: SchemaGraph,
) -> ClaimVerificationResult:
    payload = fact.payload or {}
    declared_raw = payload.get("values")
    if not isinstance(declared_raw, list):
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="declared_value_set missing values payload",
        )
    declared = frozenset(str(v).strip() for v in declared_raw if str(v).strip())
    if not declared:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="declared_value_set has no declared values",
        )
    column_refs = _qualified_column_refs(fact.referenced_entities)
    if len(column_refs) != 1:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="declared_value_set requires exactly one qualified column reference",
        )
    table_name, column_name = column_refs[0]
    col = _column_metadata(schema, table_name, column_name)
    if col is None:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence=f"column {table_name}.{column_name} not found",
        )
    if col.profile_failed or col.profile_skipped_reason:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence=f"column {table_name}.{column_name} was not profiled",
        )
    distinct_count = int(col.distinct_count or 0)
    if distinct_count > len(declared) and not col.distinct_from_sample:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.CONTRADICTED,
            evidence=(
                f"distinct_count={distinct_count} exceeds declared set size {len(declared)} "
                f"for {table_name}.{column_name}"
            ),
        )
    sample = [str(v) for v in (col.value_overlap_sample or []) if str(v)]
    cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
    if sample and distinct_count <= cap and not col.distinct_from_sample:
        observed = {str(v).strip() for v in sample if str(v).strip()}
        extra = sorted(observed - declared)
        if extra:
            return ClaimVerificationResult(
                fact=fact,
                outcome=ClaimVerificationOutcome.CONTRADICTED,
                evidence=(f"observed value(s) {extra!r} outside declared set on {table_name}.{column_name}"),
            )
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.CONFIRMED,
            evidence=f"enumerable sample contained in declared set for {table_name}.{column_name}",
        )
    return ClaimVerificationResult(
        fact=fact,
        outcome=ClaimVerificationOutcome.UNVERIFIABLE,
        evidence=f"column {table_name}.{column_name} not fully enumerable from profiling",
    )


def _verify_grain_or_uniqueness(
    fact: StructuralKnowledgeFact,
    schema: SchemaGraph,
) -> ClaimVerificationResult:
    column_refs = _qualified_column_refs(fact.referenced_entities)
    if not column_refs:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="grain claim has no qualified column reference",
        )
    tables: dict[str, list[str]] = {}
    for table_name, column_name in column_refs:
        tables.setdefault(table_name, []).append(column_name)
    if len(tables) != 1:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.UNVERIFIABLE,
            evidence="grain claim spans multiple tables",
        )
    table_name, columns = next(iter(tables.items()))
    unique = _join_columns_profiled_unique(schema, table_name, columns)
    if unique is True:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.CONFIRMED,
            evidence=f"profiled unique for {table_name}({', '.join(columns)})",
        )
    if unique is False:
        return ClaimVerificationResult(
            fact=fact,
            outcome=ClaimVerificationOutcome.CONTRADICTED,
            evidence=f"profiled not unique for {table_name}({', '.join(columns)})",
        )
    table_meta = schema.tables.get(table_name)
    if table_meta is not None and len(columns) == 1:
        col = table_meta.columns.get(columns[0])
        if col is not None and not col.profile_failed and not col.profile_skipped_reason:
            rc = int(table_meta.row_count or 0)
            dc = int(col.distinct_count or 0)
            unique_ok = bool(col.is_unique) or (rc > 0 and dc == rc and not col.distinct_from_sample)
            if unique_ok:
                return ClaimVerificationResult(
                    fact=fact,
                    outcome=ClaimVerificationOutcome.CONFIRMED,
                    evidence=f"composite-PK unique_ok for {table_name}.{columns[0]}",
                )
            if rc > 0 and dc > 0 and not col.distinct_from_sample and dc < rc:
                return ClaimVerificationResult(
                    fact=fact,
                    outcome=ClaimVerificationOutcome.CONTRADICTED,
                    evidence=(
                        f"row_count={rc} distinct_count={dc} for {table_name}.{columns[0]} "
                        "contradicts grain/uniqueness claim"
                    ),
                )
    return ClaimVerificationResult(
        fact=fact,
        outcome=ClaimVerificationOutcome.UNVERIFIABLE,
        evidence=f"uniqueness inconclusive for {table_name}({', '.join(columns)})",
    )


def _informational_claim(fact: StructuralKnowledgeFact, *, reason: str) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        fact=fact,
        outcome=ClaimVerificationOutcome.UNVERIFIABLE,
        evidence=reason,
    )


def verify_structural_knowledge_claim(
    fact: StructuralKnowledgeFact,
    schema: SchemaGraph,
) -> ClaimVerificationResult:
    """Return exactly one of confirmed, contradicted, or unverifiable for *fact*."""
    kind = str(fact.kind or "").strip().lower()
    if kind == StructuralKnowledgeKind.DECLARED_VALUE_SET.value:
        return _verify_declared_value_set(fact, schema)
    if kind == StructuralKnowledgeKind.GRAIN.value:
        return _verify_grain_or_uniqueness(fact, schema)
    if kind == StructuralKnowledgeKind.CARDINALITY.value:
        column_refs = _qualified_column_refs(fact.referenced_entities)
        if len(column_refs) == 1:
            table_name, column_name = column_refs[0]
            unique = _column_profiled_unique(schema, table_name, column_name)
            if unique is False:
                return ClaimVerificationResult(
                    fact=fact,
                    outcome=ClaimVerificationOutcome.CONFIRMED,
                    evidence=f"non-unique {table_name}.{column_name} consistent with one-to-many cardinality",
                )
            if unique is True:
                return ClaimVerificationResult(
                    fact=fact,
                    outcome=ClaimVerificationOutcome.UNVERIFIABLE,
                    evidence=f"unique {table_name}.{column_name}; cardinality direction not profile-verifiable",
                )
        return _informational_claim(fact, reason="cardinality claim not profile-verifiable")
    if kind == StructuralKnowledgeKind.UNIT_OF_MEASURE.value:
        return _informational_claim(fact, reason="unit_of_measure is informational only")
    if kind == StructuralKnowledgeKind.SENTINEL_SEMANTICS.value:
        return _informational_claim(fact, reason="sentinel_semantics is informational only")
    return _informational_claim(fact, reason=f"kind {kind!r} has no profiling oracle")


def verify_structural_knowledge_claims(
    schema: SchemaGraph,
    facts: Sequence[StructuralKnowledgeFact],
) -> tuple[ClaimVerificationResult, ...]:
    """Verify every fact; sensitivity scrub must already have removed sensitive anchors."""
    return tuple(verify_structural_knowledge_claim(fact, schema) for fact in facts)


def confirmed_declared_value_sets(
    results: Sequence[ClaimVerificationResult],
) -> dict[str, frozenset[str]]:
    """Return qualified-column -> declared values for confirmed closed- set claims (gating input)."""
    out: dict[str, frozenset[str]] = {}
    for result in results:
        if result.outcome != ClaimVerificationOutcome.CONFIRMED:
            continue
        if result.fact.kind != StructuralKnowledgeKind.DECLARED_VALUE_SET.value:
            continue
        payload = result.fact.payload or {}
        values = payload.get("values")
        if not isinstance(values, list):
            continue
        declared = frozenset(str(v).strip() for v in values if str(v).strip())
        for entity in result.fact.referenced_entities:
            if "." in entity:
                out[str(entity).strip()] = declared
                break
    return out


def finalize_structural_knowledge_claims(
    schema: SchemaGraph,
    facts: Sequence[StructuralKnowledgeFact],
) -> tuple[StructuralKnowledgeFact, ...]:
    """Scrub sensitive anchors, verify claims, and fail the build on contradictions."""
    scrubbed = filter_schema_anchored_structural_knowledge(facts, schema)
    results = verify_structural_knowledge_claims(schema, scrubbed)
    contradictions = [r for r in results if r.outcome == ClaimVerificationOutcome.CONTRADICTED]
    if contradictions:
        detail = "; ".join(f"{r.fact.kind} ({r.evidence})" for r in contradictions[:SCHEMA_CLASSIFY_ERROR_DETAIL_CAP])
        raise ConfigError(f"structural knowledge claim(s) contradicted profiling: {detail}")
    attach_structural_fanout_metadata(schema, scrubbed)
    return scrubbed


def attach_verified_declared_value_sets(schema: SchemaGraph, results: Sequence[ClaimVerificationResult]) -> None:
    """Stamp confirmed declared value sets on the schema graph for downstream gating."""
    confirmed = confirmed_declared_value_sets(results)
    if confirmed:
        object.__setattr__(schema, "_verified_declared_value_sets", confirmed)


def build_knowledge_operator_report(
    *,
    ledger: Any,
    record_stream: Sequence[tuple[str, DomainKnowledgeEntry | StructuralKnowledgeFact]],
    verification_results: Sequence[ClaimVerificationResult],
    extraction_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Owner-only build-time report: coverage ledger plus per-claim verification outcomes."""
    verification_by_fact = {id(result.fact): result for result in verification_results}
    spans: list[dict[str, Any]] = []
    for entry in getattr(ledger, "entries", ()) or ():
        span_payload: dict[str, Any] = {
            "span": entry.span,
            "coverage_disposition": entry.disposition,
        }
        if entry.record_index is not None:
            span_payload["record_index"] = entry.record_index
            if 0 <= entry.record_index < len(record_stream):
                record_kind, record = record_stream[entry.record_index]
                span_payload["record_kind"] = record_kind
                if record_kind == "structural" and isinstance(record, StructuralKnowledgeFact):
                    span_payload["structural_kind"] = record.kind
                    result = verification_by_fact.get(id(record))
                    if result is not None:
                        span_payload["verification"] = result.outcome.value
                        if result.evidence:
                            span_payload["evidence"] = result.evidence
                    else:
                        span_payload["verification"] = ClaimVerificationOutcome.UNVERIFIABLE.value
                elif record_kind == "domain" and isinstance(record, DomainKnowledgeEntry):
                    span_payload["domain_key"] = record.key
                    span_payload["verification"] = "extracted"
            else:
                span_payload["record_kind"] = "unknown"
        else:
            span_payload["verification"] = "no_fact"
        spans.append(span_payload)
    summary = {
        "confirmed": sum(1 for r in verification_results if r.outcome == ClaimVerificationOutcome.CONFIRMED),
        "unverifiable": sum(1 for r in verification_results if r.outcome == ClaimVerificationOutcome.UNVERIFIABLE),
        "contradicted": sum(1 for r in verification_results if r.outcome == ClaimVerificationOutcome.CONTRADICTED),
    }
    report: dict[str, Any] = {
        "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
        "spans": spans,
        "claim_summary": summary,
    }
    if extraction_diff is not None:
        report["extraction_diff"] = extraction_diff
    return report


def save_knowledge_operator_report(artifacts_dir: str | os.PathLike[str], report: Mapping[str, Any]) -> None:
    """Write owner-only operator report beside other build artifacts."""
    _write_json_atomic(
        Path(artifacts_dir) / KNOWLEDGE_OPERATOR_REPORT_FILENAME,
        dict(report),
        sort_keys=True,
        indent=2,
    )
