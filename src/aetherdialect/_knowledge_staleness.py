"""Staleness keys, reference-resolution, and extraction proposal artifacts for derived knowledge."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._config import EngineConfig
from ._constants import ARTIFACT_FORMAT_VERSION, KNOWLEDGE_EXPORT_FORMAT_VERSION, KNOWLEDGE_EXTRACTION_PROPOSAL_FILENAME
from ._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    KnowledgeExtractionProposal,
    NotesExtractionResult,
    StructuralKnowledgeFact,
    StructuralKnowledgeKind,
)
from ._contracts_schema import SchemaGraph
from ._knowledge_claims import (
    attach_verified_declared_value_sets,
    build_knowledge_operator_report,
    finalize_structural_knowledge_claims,
    save_knowledge_operator_report,
    verify_structural_knowledge_claims,
)
from ._knowledge_join import materialize_fk_remove_to_overrides
from ._knowledge_merge import domain_knowledge_identity, structural_knowledge_identity
from ._schema_graph import admission_report_to_dict, admit_join_fk_proposals
from ._utils import (
    debug,
    domain_knowledge_artifact_path,
    domain_knowledge_digest,
    filter_domain_knowledge_by_resolvable_references,
    format_versions_match,
    knowledge_artifact_stamp_matches,
    knowledge_scope_fingerprint,
    load_domain_knowledge_artifact,
    structural_knowledge_artifact_path,
)


def entity_resolves_on_schema(entity: str, schema: SchemaGraph) -> bool:
    """True when *entity* names an existing table or qualified column on *schema*."""
    text = str(entity).strip()
    if not text:
        return False
    if "." in text:
        table_name, column_name = text.split(".", 1)
        table = schema.tables.get(table_name)
        return table is not None and column_name in table.columns
    return text in schema.tables


def filter_structural_knowledge_by_resolvable_references(
    facts: Sequence[StructuralKnowledgeFact],
    schema: SchemaGraph,
) -> tuple[tuple[StructuralKnowledgeFact, ...], int]:
    """Drop structural facts whose reference set no longer resolves; return kept rows and drop count."""
    kept: list[StructuralKnowledgeFact] = []
    dropped = 0
    for fact in facts:
        refs = fact.referenced_entities
        if not refs or not all(entity_resolves_on_schema(ref, schema) for ref in refs):
            dropped += 1
            continue
        kept.append(fact)
    return tuple(kept), dropped


def knowledge_artifact_save_stamps(schema: SchemaGraph) -> dict[str, str | None]:
    """Return notes and scope stamps for persisting derived knowledge artifacts."""
    notes_sha = str(getattr(schema, "notes_sha256", "") or "").strip()
    return {
        "notes_sha256": notes_sha or None,
        "scope_fingerprint": knowledge_scope_fingerprint(schema),
    }


def _structural_artifact_format_ok(raw: Mapping[str, Any], *, require_stamp_match: bool) -> bool:
    """Return False when structural artifact format_version is missing or mismatched."""
    stored = raw.get("format_version")
    if require_stamp_match:
        return bool(format_versions_match(stored, ARTIFACT_FORMAT_VERSION))
    if "format_version" in raw:
        return bool(format_versions_match(stored, ARTIFACT_FORMAT_VERSION))
    return True


def knowledge_extraction_proposal_path(artifacts_dir: str | os.PathLike[str]) -> Path:
    """Return ``{artifacts_dir}/knowledge_extraction_proposal.json``."""
    return Path(artifacts_dir) / KNOWLEDGE_EXTRACTION_PROPOSAL_FILENAME


def parse_structural_items(items: Sequence[Any]) -> tuple[StructuralKnowledgeFact, ...]:
    out: list[StructuralKnowledgeFact] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip()
        text = str(raw.get("text") or "").strip()
        if not kind or not text:
            continue
        raw_referenced = raw.get("referenced_entities", [])
        if not isinstance(raw_referenced, list):
            continue
        referenced_entities = frozenset(str(r).strip() for r in raw_referenced if str(r).strip())
        payload_raw = raw.get("payload")
        payload = payload_raw if isinstance(payload_raw, dict) else None
        try:
            out.append(
                StructuralKnowledgeFact.normalize(
                    StructuralKnowledgeFact(
                        kind=kind,
                        text=text,
                        referenced_entities=referenced_entities,
                        payload=payload,
                    )
                )
            )
        except (ConfigError, ValueError, TypeError):
            continue
    return tuple(out)


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


def foreign_key_proposals_from_structural_facts(
    facts: Sequence[StructuralKnowledgeFact],
) -> tuple[dict[str, str], ...]:
    """Derive ``foreign_keys_add`` proposals from anchored join facts (graph-affecting, owner-reviewable)."""
    proposals: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        if fact.kind != StructuralKnowledgeKind.JOIN.value:
            continue
        payload = fact.payload or {}
        from_ref = str(payload.get("from") or "").strip()
        to_ref = str(payload.get("to") or "").strip()
        if not from_ref or not to_ref:
            qualified = sorted(ref for ref in fact.referenced_entities if "." in ref)
            if len(qualified) != 2:
                continue
            from_ref, to_ref = qualified[0], qualified[1]
        key = (from_ref, to_ref)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            {
                "from": from_ref,
                "to": to_ref,
                "kind": "logical",
                "provenance": "notes_structural",
            }
        )
    return tuple(proposals)


def diff_knowledge_extraction_proposals(
    prior: KnowledgeExtractionProposal | None,
    new: KnowledgeExtractionProposal,
) -> dict[str, Any]:
    """Owner-reviewable structured diff between prior and new extraction proposals."""
    if prior is None:
        return {
            "status": "initial",
            "domain_knowledge_added": [e.key for e in new.domain_knowledge],
            "structural_knowledge_added": len(new.structural_knowledge),
            "foreign_keys_add_added": list(new.foreign_keys_add),
        }

    def _dk_map(entries: Sequence[DomainKnowledgeEntry]) -> dict[str, DomainKnowledgeEntry]:
        return {"|".join(domain_knowledge_identity(e)): e for e in entries}

    def _sk_map(facts: Sequence[StructuralKnowledgeFact]) -> dict[str, StructuralKnowledgeFact]:
        return {"|".join(structural_knowledge_identity(f)): f for f in facts}

    prior_dk = _dk_map(prior.domain_knowledge)
    new_dk = _dk_map(new.domain_knowledge)
    prior_sk = _sk_map(prior.structural_knowledge)
    new_sk = _sk_map(new.structural_knowledge)
    prior_fk = {f"{item['from']}->{item['to']}": item for item in prior.foreign_keys_add}
    new_fk = {f"{item['from']}->{item['to']}": item for item in new.foreign_keys_add}

    dk_added = sorted(k for k in new_dk if k not in prior_dk)
    dk_removed = sorted(k for k in prior_dk if k not in new_dk)
    dk_changed = sorted(
        k
        for k in new_dk
        if k in prior_dk
        and (prior_dk[k].text != new_dk[k].text or prior_dk[k].referenced_entities != new_dk[k].referenced_entities)
    )
    sk_added = sorted(k for k in new_sk if k not in prior_sk)
    sk_removed = sorted(k for k in prior_sk if k not in new_sk)
    sk_changed = sorted(
        k
        for k in new_sk
        if k in prior_sk
        and (
            prior_sk[k].text != new_sk[k].text
            or prior_sk[k].payload != new_sk[k].payload
            or prior_sk[k].referenced_entities != new_sk[k].referenced_entities
        )
    )
    fk_added = sorted(k for k in new_fk if k not in prior_fk)
    fk_removed = sorted(k for k in prior_fk if k not in new_fk)
    return {
        "status": "changed",
        "domain_knowledge_added": dk_added,
        "domain_knowledge_removed": dk_removed,
        "domain_knowledge_changed": dk_changed,
        "structural_knowledge_added": sk_added,
        "structural_knowledge_removed": sk_removed,
        "structural_knowledge_changed": sk_changed,
        "foreign_keys_add_added": fk_added,
        "foreign_keys_add_removed": fk_removed,
    }


def save_knowledge_extraction_proposal(
    artifacts_dir: str | os.PathLike[str],
    proposal: KnowledgeExtractionProposal,
) -> None:
    """Persist the versioned extraction proposal artifact (single write- back point)."""
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "domain_knowledge": [
            {
                "key": e.key,
                "kind": e.kind,
                "text": e.text,
                "referenced_entities": sorted(e.referenced_entities),
            }
            for e in proposal.domain_knowledge
        ],
        "structural_knowledge": [f.to_dict() for f in proposal.structural_knowledge],
        "foreign_keys_add": [dict(item) for item in proposal.foreign_keys_add],
        "coverage": proposal.coverage,
    }
    if proposal.notes_sha256:
        payload["notes_sha256"] = proposal.notes_sha256
    if proposal.scope_fingerprint:
        payload["scope_fingerprint"] = proposal.scope_fingerprint
    if proposal.extraction_diff is not None:
        payload["extraction_diff"] = proposal.extraction_diff
    _write_json_atomic(
        knowledge_extraction_proposal_path(artifacts_dir),
        payload,
        sort_keys=True,
        indent=2,
    )


def load_knowledge_extraction_proposal(
    artifacts_dir: str | os.PathLike[str],
    schema_graph: SchemaGraph,
    *,
    require_stamp_match: bool = True,
) -> KnowledgeExtractionProposal | None:
    """Load persisted extraction proposal when derivation stamps match."""
    path = knowledge_extraction_proposal_path(artifacts_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if not _structural_artifact_format_ok(raw, require_stamp_match=require_stamp_match):
        return None
    if require_stamp_match:
        want_notes = str(getattr(schema_graph, "notes_sha256", "") or "").strip()
        scope_fp = knowledge_scope_fingerprint(schema_graph)
        if not knowledge_artifact_stamp_matches(
            raw,
            notes_sha256=want_notes,
            scope_fingerprint=scope_fp,
        ):
            return None
    domain_items = raw.get("domain_knowledge")
    structural_items = raw.get("structural_knowledge")
    coverage = raw.get("coverage")
    fk_items = raw.get("foreign_keys_add")
    if not isinstance(domain_items, list) or not isinstance(structural_items, list):
        return None
    if not isinstance(coverage, dict):
        return None
    if not isinstance(fk_items, list):
        return None
    dk_allowed = {member.value for member in DomainKnowledgeKind}
    domain_entries: list[DomainKnowledgeEntry] = []
    for item in domain_items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        text = str(item.get("text") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not key or not text or kind not in dk_allowed:
            continue
        raw_refs = item.get("referenced_entities")
        if not isinstance(raw_refs, list):
            continue
        refs = frozenset(str(r).strip() for r in raw_refs if str(r).strip())
        try:
            domain_entries.append(
                DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(key=key, kind=kind, text=text, referenced_entities=refs)
                )
            )
        except ConfigError:
            continue
    structural_facts = parse_structural_items(structural_items)
    kept_dk, _dropped_dk = filter_domain_knowledge_by_resolvable_references(domain_entries, schema_graph)
    kept_sk, _dropped_sk = filter_structural_knowledge_by_resolvable_references(structural_facts, schema_graph)
    fk_proposals: list[dict[str, str]] = []
    for item in fk_items:
        if not isinstance(item, dict):
            continue
        from_ref = str(item.get("from") or "").strip()
        to_ref = str(item.get("to") or "").strip()
        if from_ref and to_ref:
            fk_proposals.append(
                {
                    "from": from_ref,
                    "to": to_ref,
                    "kind": str(item.get("kind") or "logical").strip() or "logical",
                }
            )
    return KnowledgeExtractionProposal(
        domain_knowledge=kept_dk,
        structural_knowledge=kept_sk,
        foreign_keys_add=tuple(fk_proposals),
        coverage=coverage,
        notes_sha256=str(raw.get("notes_sha256") or "").strip() or None,
        scope_fingerprint=str(raw.get("scope_fingerprint") or "").strip() or None,
        extraction_diff=raw.get("extraction_diff") if isinstance(raw.get("extraction_diff"), dict) else None,
    )


def materialize_fk_proposals_to_overrides(
    overrides_path: str | os.PathLike[str],
    proposals: Sequence[Mapping[str, str]],
) -> bool:
    """Merge join FK proposals into overrides ``foreign_keys_add`` without duplicating edges."""
    path = Path(overrides_path)
    if not proposals:
        return False
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            doc = {}
    else:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    existing = list(doc.get("foreign_keys_add") or [])
    seen = {
        (str(item.get("from") or "").strip(), str(item.get("to") or "").strip())
        for item in existing
        if isinstance(item, dict)
    }
    changed = False
    for proposal in proposals:
        from_ref = str(proposal.get("from") or "").strip()
        to_ref = str(proposal.get("to") or "").strip()
        if not from_ref or not to_ref:
            continue
        key = (from_ref, to_ref)
        if key in seen:
            continue
        seen.add(key)
        existing.append(
            {
                "from": from_ref,
                "to": to_ref,
                "kind": str(proposal.get("kind") or "logical").strip() or "logical",
                **(
                    {"provenance": str(proposal.get("provenance")).strip()}
                    if str(proposal.get("provenance") or "").strip()
                    else {}
                ),
            }
        )
        changed = True
    if not changed:
        return False
    doc["foreign_keys_add"] = existing
    _write_json_atomic(path, doc, sort_keys=True, indent=2)
    return True


def load_structural_knowledge_artifact(
    artifacts_dir: str | os.PathLike[str],
    schema_graph: SchemaGraph,
    *,
    require_stamp_match: bool = True,
) -> tuple[StructuralKnowledgeFact, ...] | None:
    """Load persisted structural knowledge when derivation stamp matches; else ``None``."""
    path = structural_knowledge_artifact_path(artifacts_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if not _structural_artifact_format_ok(raw, require_stamp_match=require_stamp_match):
        return None
    if require_stamp_match:
        want_notes = str(getattr(schema_graph, "notes_sha256", "") or "").strip()
        scope_fp = knowledge_scope_fingerprint(schema_graph)
        if not knowledge_artifact_stamp_matches(
            raw,
            notes_sha256=want_notes,
            scope_fingerprint=scope_fp,
        ):
            return None
    items = raw.get("structural_knowledge")
    if not isinstance(items, list):
        return None
    facts = parse_structural_items(items)
    kept, dropped = filter_structural_knowledge_by_resolvable_references(facts, schema_graph)
    if dropped:
        debug(f"[load_structural_knowledge_artifact] dropped {dropped} unresolvable structural facts")
    return kept if kept else ()


def persist_structural_extraction_result(
    schema: SchemaGraph,
    extracted: NotesExtractionResult,
    *,
    artifacts_dir: str | os.PathLike[str],
    overrides_path: str | os.PathLike[str] | None = None,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Finalize, verify, and persist structural knowledge from an extraction result."""
    scope_fp = knowledge_scope_fingerprint(schema)
    notes_sha = str(getattr(schema, "notes_sha256", "") or "").strip()
    finalized = finalize_structural_knowledge_claims(schema, extracted.structural_knowledge)
    results = verify_structural_knowledge_claims(schema, finalized)
    attach_verified_declared_value_sets(schema, results)
    prior = load_knowledge_extraction_proposal(artifacts_dir, schema, require_stamp_match=False)
    fk_proposals, fk_remove, admission_entries = admit_join_fk_proposals(schema, finalized)
    fk_admission_report = admission_report_to_dict(admission_entries)
    proposal = KnowledgeExtractionProposal(
        domain_knowledge=extracted.domain_knowledge,
        structural_knowledge=finalized,
        foreign_keys_add=fk_proposals,
        coverage=extracted.ledger.to_dict(),
        notes_sha256=notes_sha or None,
        scope_fingerprint=scope_fp,
    )
    extraction_diff = diff_knowledge_extraction_proposals(prior, proposal)
    proposal = KnowledgeExtractionProposal(
        domain_knowledge=proposal.domain_knowledge,
        structural_knowledge=proposal.structural_knowledge,
        foreign_keys_add=proposal.foreign_keys_add,
        coverage=proposal.coverage,
        notes_sha256=proposal.notes_sha256,
        scope_fingerprint=proposal.scope_fingerprint,
        extraction_diff=extraction_diff,
    )
    save_knowledge_extraction_proposal(artifacts_dir, proposal)
    structural_payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "structural_knowledge": [f.to_dict() for f in finalized],
    }
    if notes_sha:
        structural_payload["notes_sha256"] = notes_sha
    if scope_fp:
        structural_payload["scope_fingerprint"] = scope_fp
    _write_json_atomic(
        structural_knowledge_artifact_path(artifacts_dir),
        structural_payload,
        sort_keys=True,
        indent=2,
    )
    domain_payload: dict[str, Any] = {
        "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
        "domain_knowledge": [
            {
                "key": e.key,
                "kind": e.kind,
                "text": e.text,
                "referenced_entities": sorted(e.referenced_entities),
            }
            for e in extracted.domain_knowledge
        ],
        "domain_knowledge_digest": domain_knowledge_digest(extracted.domain_knowledge),
    }
    if notes_sha:
        domain_payload["notes_sha256"] = notes_sha
    if scope_fp:
        domain_payload["scope_fingerprint"] = scope_fp
    _write_json_atomic(
        domain_knowledge_artifact_path(artifacts_dir),
        domain_payload,
        sort_keys=True,
        indent=2,
    )
    operator_report = build_knowledge_operator_report(
        ledger=extracted.ledger,
        record_stream=extracted.record_stream,
        verification_results=results,
        extraction_diff=extraction_diff,
    )
    if fk_admission_report:
        operator_report["fk_admission"] = fk_admission_report
    save_knowledge_operator_report(artifacts_dir, operator_report)
    if overrides_path and fk_proposals:
        materialize_fk_proposals_to_overrides(overrides_path, fk_proposals)
    if overrides_path and fk_remove:
        materialize_fk_remove_to_overrides(overrides_path, fk_remove)
    return finalized


def _finalize_loaded_structural(
    schema: SchemaGraph,
    structural_items: Sequence[StructuralKnowledgeFact],
) -> tuple[StructuralKnowledgeFact, ...]:
    finalized = finalize_structural_knowledge_claims(schema, structural_items)
    results = verify_structural_knowledge_claims(schema, finalized)
    attach_verified_declared_value_sets(schema, results)
    return finalized


def resolve_knowledge_extraction_for_schema(
    schema: SchemaGraph,
    notes_content: str | None,
    *,
    artifacts_dir: str | os.PathLike[str] | None = None,
    force_extract: bool = False,
    overrides_path: str | os.PathLike[str] | None = None,
    extract_knowledge_from_notes: Any = None,
) -> tuple[tuple[DomainKnowledgeEntry, ...], tuple[StructuralKnowledgeFact, ...]]:
    """Load or extract domain and structural knowledge via the persisted proposal artifact."""
    notes_stripped = (notes_content or "").strip()
    if not notes_stripped:
        return (), ()
    if artifacts_dir and not force_extract:
        proposal = load_knowledge_extraction_proposal(artifacts_dir, schema, require_stamp_match=True)
        if proposal is not None:
            return proposal.domain_knowledge, _finalize_loaded_structural(schema, proposal.structural_knowledge)
        structural_loaded = load_structural_knowledge_artifact(schema_graph=schema, artifacts_dir=artifacts_dir)
        domain_loaded = load_domain_knowledge_artifact(artifacts_dir, schema, require_notes_match=True)
        if structural_loaded is not None:
            finalized = _finalize_loaded_structural(schema, structural_loaded)
            return domain_loaded or (), finalized
    if extract_knowledge_from_notes is None:
        return (), ()
    if not EngineConfig.llm_credentials_configured():
        return (), ()
    extracted = extract_knowledge_from_notes(notes_content, schema)
    if artifacts_dir:
        structural = persist_structural_extraction_result(
            schema,
            extracted,
            artifacts_dir=artifacts_dir,
            overrides_path=overrides_path,
        )
        return extracted.domain_knowledge, structural
    finalized = _finalize_loaded_structural(schema, extracted.structural_knowledge)
    return extracted.domain_knowledge, finalized


def resolve_structural_knowledge_for_schema(
    schema: SchemaGraph,
    notes_content: str | None,
    *,
    artifacts_dir: str | None = None,
    force_extract: bool = False,
    overrides_path: str | os.PathLike[str] | None = None,
    extract_knowledge_from_notes: Any = None,
) -> tuple[StructuralKnowledgeFact, ...]:
    """Load or extract structural knowledge via the persisted proposal artifact."""
    _domain, structural = resolve_knowledge_extraction_for_schema(
        schema,
        notes_content,
        artifacts_dir=artifacts_dir,
        force_extract=force_extract,
        overrides_path=overrides_path,
        extract_knowledge_from_notes=extract_knowledge_from_notes,
    )
    return structural
