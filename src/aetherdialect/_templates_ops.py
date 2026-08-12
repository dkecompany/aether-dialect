"""Template learning, params, pending upserts, and TemplateOps composition."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ._config import EngineConfig, PolicyConfig, SeedWarmupConfig
from ._constants import (
    AETHERSPACES_SEGMENT,
    ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP,
    ARTIFACT_LAST_ACTION_REMAP_USER_MAP,
    ARTIFACT_MANIFEST_FILENAME,
    MIGRATION_CHECKPOINT_SCHEMA_BASENAME,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_ACTION_DESTRUCTIVE,
    MIGRATION_MAP_ACTION_REMAP,
    MIGRATION_MAP_FILENAME,
    TEMPLATE_STORE_SEGMENT,
    TRUST_AUTO_ACCEPT_THRESHOLD,
    TRUST_CEILING,
    TRUST_FLOOR,
)
from ._constants_runtime import (
    JOIN_PRIOR_FEEDBACK_PATH_LABEL,
    PARAM_QUESTION_REMAP_SYSTEM,
)
from ._contracts_base import (
    ApprovalState,
    ConfigError,
    EngineContext,
    FederationContext,
    HavingParam,
    MigrationPendingError,
    MigrationReport,
    MigrationTier,
    MockFixtureMissingError,
    ParamValue,
    PredicateGroup,
    SchemaMigrationMap,
    SchemaMigrationMapEntry,
    WhereParam,
)
from ._contracts_core import (
    ConcreteIntent,
    FeedbackCounts,
    FeedbackKind,
    GenerationPath,
    LlmJsonExhausted,
    NoJoinPathError,
    ParameterBinding,
    QuestionFeedbackEntry,
    QuestionFormStorage,
    RejectionBucket,
    RuntimeIntent,
    StoredTemplateDetail,
    StoredTemplateSummary,
    Template,
    ValueHistory,
)
from ._contracts_schema import IntentIssue, SchemaDiff, SchemaGraph, TableDiff, TemplateStats
from ._dialect import Dialect
from ._federation_manifest import member_feedback_q_norm
from ._intent_bind import (
    compute_intent_union,
)
from ._intent_expr import (
    collect_intent_referenced_param_keys,
)
from ._llm_provider import LLMProvider, SandboxRuntimeState
from ._schema_finalize import destructive_migration_execute, migrate_sidecar_for_diff, reconcile_sidecar_against_graph
from ._schema_graph import (
    apply_fk_remaps_to_graph,
    apply_pk_remaps_to_graph,
    caller_may_see_column_binding,
    rename_migration_plan_confidence,
    schema_diff_cross_table_limitation_note,
    schema_diff_is_additive_only,
)
from ._sql_gen import (
    build_display_sql,
    canonicalize_stored_join_path_signature,
    generate_col_alias,
)
from ._templates import (
    SANDBOX_PARAPHRASE_SOURCE,
    ParamSlotMeta,
    TemplateRefs,
    TemplateStoreLifecycleOps,
    TemplateStoreView,
)
from ._utils import (
    canonicalize_sql,
    colmap_signature,
    debug,
    is_structural_param_key,
    normalize_sql,
    require_exact_keys,
    safe_json_loads,
    stable_json,
)
from ._utils_artifacts import (
    artifact_lock,
    migrate_engine_knowledge_artifacts,
)
from ._utils_intent import (
    extract_tables_from_sql,
    flatten_param_values,
    generate_warmup_paraphrases_by_style,
    intent_key,
    is_exact_question_text_match,
    resolve_template_for_question,
    select_diverse_paraphrases,
    sql_shape,
    template_enumerable_by_caller,
    template_footprint_tables,
    template_visible_to_callers,
    validate_question,
)


class TemplateLearningOps:
    """Feedback, trust, param slots, insert, and pending template APIs."""

    @staticmethod
    def _coerce_rejection_bucket(raw: object) -> RejectionBucket:
        """Map LLM or wire text to :class:`RejectionBucket` (default ``OTHER``)."""
        if isinstance(raw, RejectionBucket):
            return raw
        s = str(raw or "").strip().upper()
        for m in RejectionBucket:
            if m.name == s or m.value == s:
                return m
        return RejectionBucket.OTHER

    @staticmethod
    def _compute_intent_structural_signature(intent: RuntimeIntent | None) -> tuple[str, str]:
        """Return ``(sha256_first_16_hex, stable_json_of_concrete_intent)`` for deduplication and LLM context."""
        if intent is None:
            return "", ""
        conc = intent.to_concrete("")
        payload = stable_json(conc.to_dict())
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return digest, payload

    @staticmethod
    def dedupe_prior_question_feedback_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return *rows* de-duplicated by ``(intent_structural_hash, summary prefix)``, preserving order."""
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for row in rows:
            ihash = str(row.get("intent_structural_hash", "") or "")
            summary = str(row.get("summary", "") or "")
            key = (ihash, summary[:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def in_turn_row_from_semantic_errors(
        errors: list[IntentIssue], schema_hash: str, intent: RuntimeIntent
    ) -> dict[str, str]:
        """Build one ``to_prompt_row``-shaped dict from semantic validation errors."""
        max_b = PolicyConfig.MAX_SUMMARY_BULLETS
        lines = [f"[{e.category.value}] {e.message}" for e in errors[:max_b]]
        summary = "\n".join(lines)
        ts = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ish, ipay = TemplateOps._compute_intent_structural_signature(intent)
        return {
            "kind": FeedbackKind.VALIDATION_FAILURE.value,
            "summary": summary,
            "buckets": RejectionBucket.OTHER.value,
            "effective_structural_hash": schema_hash,
            "intent_structural_hash": ish,
            "intent_payload": ipay,
            "created_at": ts,
            "updated_at": ts,
            "is_post_restart": "False",
        }

    @staticmethod
    def serialized_prior_feedback_rows(rows: list[dict[str, str]] | None) -> str:
        """Serialise merged feedback rows for Ground as compact JSON text."""
        if not rows:
            return ""
        return stable_json({"items": rows})

    @staticmethod
    def _rejected_join_path_signature_from_intent(intent: RuntimeIntent | None) -> tuple[str, ...]:
        """Return engine-verified join path signature from a runtime intent when present."""
        if intent is None:
            return ()
        sig = canonicalize_stored_join_path_signature(list(intent.chosen_join_path_signature or []))
        return tuple(str(s).strip() for s in sig if str(s).strip())

    @staticmethod
    def format_join_feedback_injection_line(
        summary: str,
        rejected_join_path_signature: Sequence[str] | tuple[str, ...] | None = None,
    ) -> str:
        """Build one join-feedback bullet for join-choice prompts, including verified FK path when stored."""
        text = (summary or "").strip()
        sig = [str(s).strip() for s in (rejected_join_path_signature or ()) if str(s).strip()]
        if not sig:
            return text
        path_text = ", ".join(sig)
        if text:
            return f"{text} ({JOIN_PRIOR_FEEDBACK_PATH_LABEL} {path_text})"
        return f"{JOIN_PRIOR_FEEDBACK_PATH_LABEL} {path_text}"

    @staticmethod
    def summarize_failure_for_memory(
        *,
        question: str,
        intent: RuntimeIntent | None,
        kind: FeedbackKind,
        schema_hash: str,
        validator_errors: list[str] | None = None,
        user_reason: str | None = None,
        is_post_restart: bool = False,
        source: Literal["engine"] = "engine",
    ) -> QuestionFeedbackEntry:
        """Build one ``QuestionFeedbackEntry`` using a single LLM summary call. Persisted rows use ``kind`` ``INTENT_REJECTED`` when the user rejects a validated intent or result. Persisted rows use ``kind`` ``VALIDATION_FAILURE`` only when :func:`aetherdialect._intent_loop.attempt_fresh_restart` records semantic exhaustion after ``semantic_oscillation`` or ``semantic_max_rounds``. Malformed LLM output is coerced to a summary string and ``OTHER`` bucket. ``llm_chat`` transport failures are not caught and propagate to the caller."""
        ish, ipay = TemplateOps._compute_intent_structural_signature(intent)
        structure_for_prompt = ipay if ipay else "{}"
        payload: dict[str, Any] = {
            "question": question,
            "kind": kind.value,
            "intent_structure_json": structure_for_prompt,
            "user_reason": user_reason or "",
        }
        if validator_errors:
            payload["validator_errors"] = list(validator_errors)
        bucket_help = (
            "MISSING_FILTER: implicit filter or constraint was missing. "
            "WRONG_GROUPING: rollup grain was wrong. "
            "WRONG_AGGREGATION: measure or aggregation was wrong. "
            "WRONG_TIME_RANGE: time window was wrong. "
            "WRONG_TABLES_OR_JOINS: wrong entities or relationships. "
            "WRONG_SORT_OR_LIMIT: ordering or top-N was wrong. "
            "OTHER: none of the above."
        )
        system = (
            "You compress one text-to-SQL failure into JSON only. "
            "Fields: summary (3-6 lines joined by newlines, ASCII, each line a structural issue; no raw SQL), "
            f"bucket (one of the categories). {bucket_help} "
            'Respond as {"summary":"...","bucket":"MISSING_FILTER"}.'
        )
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        ts = TemplateOps._feedback_iso_now()
        rej_sig = TemplateOps._rejected_join_path_signature_from_intent(intent)
        if not EngineConfig.llm_credentials_configured():
            fb = (user_reason or (validator_errors[0] if validator_errors else "") or "(unspecified failure)").strip()
            return QuestionFeedbackEntry(
                summary=fb,
                buckets=(RejectionBucket.OTHER,),
                kind=kind,
                effective_structural_hash=schema_hash,
                intent_structural_hash=ish,
                intent_payload=ipay,
                created_at=ts,
                updated_at=ts,
                is_post_restart=is_post_restart,
                source=source,
                rejected_join_path_signature=rej_sig,
            )
        try:
            raw = LLMProvider.chat(system, user, task="feedback", max_retries=1, timeout=20.0)
        except MockFixtureMissingError:
            if EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
                fb = (
                    user_reason or (validator_errors[0] if validator_errors else "") or "(unspecified failure)"
                ).strip()
                return QuestionFeedbackEntry(
                    summary=fb,
                    buckets=(RejectionBucket.OTHER,),
                    kind=kind,
                    effective_structural_hash=schema_hash,
                    intent_structural_hash=ish,
                    intent_payload=ipay,
                    created_at=ts,
                    updated_at=ts,
                    is_post_restart=is_post_restart,
                    source=source,
                    rejected_join_path_signature=rej_sig,
                )
            raise
        try:
            data = json.loads(raw)
        except Exception as exc:
            debug(f"[templates.summarize_failure_for_memory] json coerce: {exc}")
            fb = (user_reason or (validator_errors[0] if validator_errors else "") or "(unspecified failure)").strip()
            return QuestionFeedbackEntry(
                summary=fb,
                buckets=(RejectionBucket.OTHER,),
                kind=kind,
                effective_structural_hash=schema_hash,
                intent_structural_hash=ish,
                intent_payload=ipay,
                created_at=ts,
                updated_at=ts,
                is_post_restart=is_post_restart,
                source=source,
                rejected_join_path_signature=rej_sig,
            )
        compressed = str(data.get("summary", "")).strip() or (user_reason or "(unspecified failure)").strip()
        bucket = TemplateOps._coerce_rejection_bucket(data.get("bucket"))
        return QuestionFeedbackEntry(
            summary=compressed,
            buckets=(bucket,),
            kind=kind,
            effective_structural_hash=schema_hash,
            intent_structural_hash=ish,
            intent_payload=ipay,
            created_at=ts,
            updated_at=ts,
            is_post_restart=is_post_restart,
            source=source,
            rejected_join_path_signature=rej_sig,
        )

    @staticmethod
    def _iter_store_question_feedback_items(
        store: dict[str, Any] | TemplateStoreView,
    ) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        if isinstance(store, TemplateStoreView):
            return store._iter_question_feedback_items()
        qf = store.get("question_feedback")
        if not isinstance(qf, dict):
            return iter(())

        def _gen() -> Iterator[tuple[str, list[dict[str, Any]]]]:
            for qk, rows in qf.items():
                if isinstance(rows, list):
                    yield str(qk), [r for r in rows if isinstance(r, dict)]

        return _gen()

    @staticmethod
    def lookup_join_feedback_for_question(
        store: dict[str, Any] | TemplateStoreView,
        q_norm: str,
        *,
        schema_graph_id: str | None = None,
        member_source_id: str | None = None,
        visible_tables: frozenset[str] | None = None,
        schema: SchemaGraph | None = None,
    ) -> list[str]:
        """Return textual rejection summaries for prior wrong-join feedback on this question."""
        lookup_q = member_feedback_q_norm(member_source_id, q_norm) if member_source_id else q_norm
        if isinstance(store, TemplateStoreView):
            rows = store._get_feedback_rows(lookup_q)
        else:
            qf_raw = store.get("question_feedback")
            if not isinstance(qf_raw, dict):
                return []
            raw_rows = qf_raw.get(lookup_q)
            rows = raw_rows if isinstance(raw_rows, list) else []
        known_tables = frozenset(schema.tables.keys()) if schema is not None else frozenset()
        rows_with_ts: list[tuple[str, str]] = []
        for row in rows:
            ent = QuestionFeedbackEntry.from_dict(row)
            if schema_graph_id is not None:
                row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                if row_graph_id and row_graph_id != schema_graph_id:
                    continue
                if not row_graph_id and ent.effective_structural_hash != schema_graph_id:
                    continue
            if ent.kind is not FeedbackKind.INTENT_REJECTED:
                continue
            if RejectionBucket.WRONG_TABLES_OR_JOINS not in ent.buckets:
                continue
            if member_source_id is not None:
                row_member = str(row.get("member_source_id", ent.member_source_id) or "")
                if row_member != member_source_id:
                    continue
            summary = TemplateOps._filter_join_feedback_summary(
                (ent.summary or "").strip(),
                ent.rejected_join_path_signature,
                ent.intent_payload,
                visible_tables=visible_tables,
                known_tables=known_tables,
            )
            if not summary and not ent.rejected_join_path_signature:
                continue
            line = TemplateOps.format_join_feedback_injection_line(summary, ent.rejected_join_path_signature)
            if visible_tables is not None and not TemplateOps._join_feedback_line_visible_to_caller(
                line,
                ent.intent_payload,
                visible_tables=visible_tables,
                known_tables=known_tables,
            ):
                continue
            if not line:
                continue
            rows_with_ts.append((ent.updated_at or ent.created_at, line))
        rows_with_ts.sort(key=lambda t: t[0], reverse=True)
        return [s for _ts, s in rows_with_ts]

    @staticmethod
    def _tables_from_join_path_signature(signature: Sequence[str] | tuple[str, ...] | None) -> frozenset[str]:
        tables: set[str] = set()
        for seg in signature or ():
            for side in str(seg).split("->"):
                side = side.strip()
                if "." in side:
                    tbl = side.split(".", 1)[0].strip()
                    if tbl:
                        tables.add(tbl)
        return frozenset(tables)

    @staticmethod
    def _tables_from_intent_payload(intent_payload: str) -> frozenset[str]:
        payload = (intent_payload or "").strip()
        if not payload:
            return frozenset()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return frozenset()
        if not isinstance(data, dict):
            return frozenset()
        tables: set[str] = set()
        for raw in data.get("tables") or ():
            name = str(raw).strip()
            if name:
                tables.add(name)
        return frozenset(tables)

    @staticmethod
    def _tables_mentioned_in_feedback_text(text: str, known_tables: frozenset[str]) -> frozenset[str]:
        if not text or not known_tables:
            return frozenset()
        mentioned: set[str] = set()
        for tbl in known_tables:
            if re.search(rf"\b{re.escape(tbl)}\b", text, flags=re.IGNORECASE):
                mentioned.add(tbl)
        return frozenset(mentioned)

    @staticmethod
    def _join_feedback_tables_referenced(
        summary: str,
        rejected_join_path_signature: Sequence[str] | tuple[str, ...] | None,
        intent_payload: str,
        *,
        known_tables: frozenset[str],
    ) -> frozenset[str]:
        tables = set(TemplateOps._tables_from_intent_payload(intent_payload))
        tables.update(TemplateOps._tables_from_join_path_signature(rejected_join_path_signature))
        tables.update(TemplateOps._tables_mentioned_in_feedback_text(summary, known_tables))
        return frozenset(tables)

    @staticmethod
    def _join_feedback_line_visible_to_caller(
        line: str,
        intent_payload: str,
        *,
        visible_tables: frozenset[str] | None,
        known_tables: frozenset[str],
    ) -> bool:
        if visible_tables is None:
            return True
        mentioned = TemplateOps._join_feedback_tables_referenced(
            line,
            None,
            intent_payload,
            known_tables=known_tables,
        )
        mentioned |= TemplateOps._tables_mentioned_in_feedback_text(line, known_tables)
        if not mentioned:
            return True
        return mentioned <= visible_tables

    @staticmethod
    def _filter_join_feedback_summary(
        summary: str,
        rejected_join_path_signature: Sequence[str] | tuple[str, ...] | None,
        intent_payload: str,
        *,
        visible_tables: frozenset[str] | None,
        known_tables: frozenset[str],
    ) -> str:
        if not summary or visible_tables is None:
            return summary
        kept: list[str] = []
        for segment in summary.splitlines():
            seg = segment.strip()
            if not seg:
                continue
            mentioned = TemplateOps._join_feedback_tables_referenced(
                seg,
                rejected_join_path_signature,
                intent_payload,
                known_tables=known_tables,
            )
            if mentioned and not mentioned <= visible_tables:
                continue
            kept.append(segment)
        return "\n".join(kept).strip()

    @staticmethod
    def _join_avoid_candidate_id_from_intent_payload(intent_payload: str) -> str | None:
        """Return a rejected join candidate id stored in feedback intent JSON, if any."""
        payload = (intent_payload or "").strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        cid = data.get("chosen_join_candidate_id")
        if isinstance(cid, str):
            stripped = cid.strip()
            if stripped and stripped != "J00":
                return stripped
        return None

    @staticmethod
    def lookup_join_avoid_candidate_ids_for_question(
        store: dict[str, Any] | TemplateStoreView,
        q_norm: str,
        *,
        schema_graph_id: str | None = None,
        member_source_id: str | None = None,
        visible_tables: frozenset[str] | None = None,
        schema: SchemaGraph | None = None,
    ) -> frozenset[str]:
        """Return join candidate ids to hard-filter from LLM choice scopes for prior wrong-join feedback."""
        lookup_q = member_feedback_q_norm(member_source_id, q_norm) if member_source_id else q_norm
        if isinstance(store, TemplateStoreView):
            rows = store._get_feedback_rows(lookup_q)
        else:
            qf_raw = store.get("question_feedback")
            if not isinstance(qf_raw, dict):
                return frozenset()
            raw_rows = qf_raw.get(lookup_q)
            rows = raw_rows if isinstance(raw_rows, list) else []
        known_tables = frozenset(schema.tables.keys()) if schema is not None else frozenset()
        avoid: set[str] = set()
        for row in rows:
            ent = QuestionFeedbackEntry.from_dict(row)
            if schema_graph_id is not None:
                row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                if row_graph_id and row_graph_id != schema_graph_id:
                    continue
                if not row_graph_id and ent.effective_structural_hash != schema_graph_id:
                    continue
            if ent.kind is not FeedbackKind.INTENT_REJECTED:
                continue
            if RejectionBucket.WRONG_TABLES_OR_JOINS not in ent.buckets:
                continue
            if member_source_id is not None:
                row_member = str(row.get("member_source_id", ent.member_source_id) or "")
                if row_member != member_source_id:
                    continue
            if visible_tables is not None:
                mentioned = TemplateOps._join_feedback_tables_referenced(
                    (ent.summary or "").strip(),
                    ent.rejected_join_path_signature,
                    ent.intent_payload,
                    known_tables=known_tables,
                )
                if mentioned and not mentioned <= visible_tables:
                    continue
            cid = TemplateOps._join_avoid_candidate_id_from_intent_payload(ent.intent_payload)
            if cid:
                avoid.add(cid)
        return frozenset(avoid)

    @staticmethod
    def record_deterministic_join_failure_feedback(
        store: dict[str, Any] | TemplateStoreView,
        q_norm: str,
        exc: NoJoinPathError,
        *,
        intent: RuntimeIntent,
        schema: SchemaGraph,
    ) -> None:
        """Persist WRONG_TABLES_OR_JOINS feedback for a deterministic join- path refusal."""
        ish, ipay = TemplateOps._compute_intent_structural_signature(intent)
        if not ish:
            return
        ts = TemplateOps._feedback_iso_now()
        entry = QuestionFeedbackEntry(
            summary=exc.user_message,
            buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash=str(schema.effective_structural_hash or ""),
            intent_structural_hash=ish,
            intent_payload=ipay,
            created_at=ts,
            updated_at=ts,
            source="engine",
        )
        row = entry.to_dict()
        row["schema_graph_id"] = str(schema.schema_graph_id or schema.effective_structural_hash or "")
        TemplateOps.record_question_feedback(store, q_norm, QuestionFeedbackEntry.from_dict(row))

    @staticmethod
    def collect_question_feedback_for_prompt(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, effective_structural_hash: str
    ) -> list[dict[str, str]]:
        """Return feedback rows for prompts in insertion order, scoped to *effective_structural_hash*."""
        out: list[dict[str, str]] = []
        if isinstance(store, TemplateStoreView):
            for q_key in store.feedback_shard_index:
                if not is_exact_question_text_match(q_norm, str(q_key)):
                    continue
                for row in store._get_feedback_rows(str(q_key)):
                    ent = QuestionFeedbackEntry.from_dict(row)
                    row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                    if row_graph_id and row_graph_id != effective_structural_hash:
                        continue
                    if not row_graph_id and ent.effective_structural_hash != effective_structural_hash:
                        continue
                    out.append(ent.to_prompt_row())
            return out
        for q_key, rows in TemplateOps._iter_store_question_feedback_items(store):
            if not is_exact_question_text_match(q_norm, str(q_key)):
                continue
            for row in rows:
                ent = QuestionFeedbackEntry.from_dict(row)
                row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                if row_graph_id and row_graph_id != effective_structural_hash:
                    continue
                if not row_graph_id and ent.effective_structural_hash != effective_structural_hash:
                    continue
                out.append(ent.to_prompt_row())
        return out

    @staticmethod
    def compute_question_feedback_penalty(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, effective_structural_hash: str
    ) -> float:
        """Map matching feedback count to a confidence penalty capped at ``PENALTY_CAP``."""
        weighted = 0.0
        if isinstance(store, TemplateStoreView):
            for q_key in store.feedback_shard_index:
                if not is_exact_question_text_match(q_norm, str(q_key)):
                    continue
                for row in store._get_feedback_rows(str(q_key)):
                    ent = QuestionFeedbackEntry.from_dict(row)
                    row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                    if row_graph_id and row_graph_id != effective_structural_hash:
                        continue
                    if not row_graph_id and ent.effective_structural_hash != effective_structural_hash:
                        continue
                    weighted += 1.0
            return float(min(PolicyConfig.PENALTY_CAP, weighted * PolicyConfig.PEN_BY_THREE_SOURCE_UNIT))
        for q_key, rows in TemplateOps._iter_store_question_feedback_items(store):
            if not is_exact_question_text_match(q_norm, str(q_key)):
                continue
            for row in rows:
                ent = QuestionFeedbackEntry.from_dict(row)
                row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                if row_graph_id and row_graph_id != effective_structural_hash:
                    continue
                if not row_graph_id and ent.effective_structural_hash != effective_structural_hash:
                    continue
                weighted += 1.0
        return float(min(PolicyConfig.PENALTY_CAP, weighted * PolicyConfig.PEN_BY_THREE_SOURCE_UNIT))

    @staticmethod
    def has_any_rejection_history_for_question(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, schema_graph_id: str | None = None
    ) -> bool:
        """Return True when ``question_feedback`` has any row for a fuzzy-matching question key."""
        return TemplateStoreLifecycleOps.has_any_rejection_history_for_question(store, q_norm, schema_graph_id)

    @staticmethod
    def delete_rejected_templates_matching_question(store: dict[str, Any] | TemplateStoreView, q_norm: str) -> None:
        """Remove question-feedback entries whose key fuzzy-matches *q_norm*."""
        if isinstance(store, TemplateStoreView):
            for q_key in list(store.feedback_shard_index.keys()):
                if is_exact_question_text_match(q_norm, str(q_key)):
                    store._remove_feedback_question_key(str(q_key))
                    debug(f"[templates.delete_rejected_templates_matching_question] removed feedback for key={q_key!r}")
            return
        qf = store.get("question_feedback")
        if not isinstance(qf, dict):
            return
        for q_key in list(qf):
            if is_exact_question_text_match(q_norm, str(q_key)):
                del qf[q_key]
                debug(f"[templates.delete_rejected_templates_matching_question] removed feedback for key={q_key!r}")

    @staticmethod
    def promote_rejected_to_template(
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Template],
        q_norm: str,
        intent: RuntimeIntent,
        sql: str,
        schema_graph_id: str,
        *,
        effective_structural_hash: str = "",
        form_storage: QuestionFormStorage | None = None,
    ) -> Template:
        """Create an accepted template after the user accepts a result following prior question feedback. There is no separate rejected-template record in the store."""
        tid = TemplateOps._reserve_template_id(store)

        sql_canon = canonicalize_sql(sql)
        if intent.sql_param:
            sql_param = intent.sql_param
        else:
            sql_norm = normalize_sql(sql_canon)
            sql_param, _ = Dialect.parameter_abstract(sql_norm, sqlglot_dialect=Dialect.active_sqlglot_dialect())
        sql_fp_val = Dialect.compute_sql_fp(sql_param, sqlglot_dialect=Dialect.active_sqlglot_dialect())

        colmap_sig_val = colmap_signature(intent.column_map)

        intent_signature = intent.to_concrete("")
        ikey = intent_key(intent)

        all_pv = flatten_param_values(intent)
        nl0 = intent.natural_language or ""
        primary_q = form_storage.corrected if form_storage is not None else q_norm
        vh_new = ValueHistory(param_values=[all_pv], questions=[primary_q], natural_language=[nl0])
        if (
            form_storage is not None
            and not form_storage.accept_via_normalized_lookup_only
            and form_storage.normalized_optional
            and form_storage.normalized_optional != primary_q
            and not form_storage.normalized_negative_memory_dropped
        ):
            vh_new.append_question_variant(
                form_storage.normalized_optional, accept_count=0, param_values=all_pv, natural_language=nl0
            )

        sig_aliases: dict[str, str] = {}
        for sc in intent.select_cols or []:
            alias = generate_col_alias(sc)
            if alias:
                sig_aliases[sc.signature_key] = alias

        tmpl = Template(
            id=tid,
            schema_graph_id=schema_graph_id,
            effective_structural_hash=effective_structural_hash or schema_graph_id,
            intent_signature=intent_signature,
            intent_key=ikey,
            tables_used=sorted(intent.tables),
            sql_param=sql_param,
            sql_fp=sql_fp_val,
            shape=sql_shape(sql_canon, intent, sqlglot_dialect=Dialect.active_sqlglot_dialect()),
            colmap_sig=colmap_sig_val,
            value_history=vh_new,
            stats=TemplateStats(accept=1, reject=0),
            source="human",
            trust_level=TRUST_FLOOR,
            structural_defaults={k: v for k, v in all_pv.items() if is_structural_param_key(k)},
            display_alias_map=sig_aliases,
        )

        templates[tid] = tmpl

        debug(f"[templates.promote_rejected_to_template] created_template: id={tid}")
        TemplateOps.templates_to_store(store, templates)
        return tmpl

    @staticmethod
    def promote_trust(template: Template, q_norm: str = "") -> bool:
        """Promote ``template`` from trust=1 to trust=2. Promotion fires. when the per-(template, question) pair has accumulated at least :attr:`PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS` accepts AND the template's overall reject ratio is at most :attr:`PolicyConfig.TRUST_PROMOTE_MAX_REJECT_RATIO`. Trust=2 is terminal (no further promotion). Trust=1 is the floor (no demotion)."""
        if template.trust_level >= TRUST_CEILING:
            return False
        if not q_norm:
            return False
        counts = template.feedback_by_question.get(q_norm)
        if counts is None or counts.accepts < PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS:
            return False
        accept_count = template.stats.accept
        reject_count = template.stats.reject
        total_count = accept_count + reject_count
        if total_count == 0:
            return False
        reject_ratio = reject_count / total_count
        if reject_ratio > PolicyConfig.TRUST_PROMOTE_MAX_REJECT_RATIO:
            return False
        template.trust_level = TRUST_CEILING
        debug(
            f"[templates.promote_trust] promoted: id={template.id} q={q_norm[:40]!r} "
            f"pair_accepts={counts.accepts} ratio={reject_ratio:.2f}"
        )
        return True

    @staticmethod
    def record_template_feedback(template: Template, accept: bool) -> None:
        """Record accept or reject feedback on a template."""
        if accept:
            template.stats.accept += 1
        else:
            template.stats.reject += 1
        debug(f"[templates.record_template_feedback] recorded: id={template.id} accept={accept}")

    @staticmethod
    def record_per_question_feedback(template: Template, q_norm: str, accept: bool, path: int) -> FeedbackCounts:
        """Update ``template.feedback_by_question[q_norm]`` with one. accept/reject event."""
        counts = template.feedback_by_question.get(q_norm)
        if counts is None:
            counts = FeedbackCounts()
            template.feedback_by_question[q_norm] = counts
        if accept:
            counts.accepts += 1
        else:
            counts.rejects += 1
        counts.last_path = int(path)
        debug(
            f"[templates.record_per_question_feedback] id={template.id} q={q_norm[:40]!r} "
            f"accept={accept} path={path} counts={counts.to_dict()}"
        )
        return counts

    @staticmethod
    def _apply_structural_defaults_from_intent(t: Template, intent: RuntimeIntent) -> None:
        """Copy structural parameter values from *intent* onto. *t.structural_defaults*."""
        for pk, pv in flatten_param_values(intent).items():
            if is_structural_param_key(pk):
                t.structural_defaults[pk] = pv

    @staticmethod
    def should_prompt_sql_feedback(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, matched_template: Template | None
    ) -> bool:
        """Return True when the result prompt must be shown (rejection history or trust not yet earned)."""
        if TemplateOps.has_any_rejection_history_for_question(store, q_norm):
            return True
        if matched_template is None:
            return True
        return not TemplateOps.should_auto_accept_for_question(matched_template, q_norm)

    @staticmethod
    def path_bucket(path: GenerationPath | str | int | None) -> int:
        """Return the integer bucket (1-6) for a ``GenerationPath`` or its string code."""
        if path is None:
            return 0
        if isinstance(path, int):
            return int(path) if 1 <= int(path) <= 6 else 0
        code = path.code if isinstance(path, GenerationPath) else str(path)
        if not code:
            return 0
        head = code[0]
        return int(head) if head.isdigit() else 0

    @staticmethod
    def should_auto_accept_for_question(
        template: Template, q_norm: str, *, reuse_history_index: int | None = None
    ) -> bool:
        """Decide whether a generated answer can skip user confirmation for. ``q_norm``. Rule: auto-accept iff trust is at least two and the matched ``value_history`` row's ``accept_counts`` meets :data:`TRUST_AUTO_ACCEPT_THRESHOLD`."""
        if template.trust_level < TRUST_CEILING:
            return False
        if not q_norm:
            return False
        vh = template.value_history
        idx: int = reuse_history_index if reuse_history_index is not None else -1
        if reuse_history_index is None:
            for i, row_q in enumerate(vh.questions):
                if row_q == q_norm:
                    idx = i
                    break
        if idx < 0 or idx >= len(vh.accept_counts):
            return False
        return vh.accept_counts[idx] >= TRUST_AUTO_ACCEPT_THRESHOLD

    @staticmethod
    def reject_out_per_question(templates: dict[str, Template], template: Template, q_norm: str) -> tuple[bool, bool]:
        """Apply per-pair reject-out semantics when. ``feedback_by_question[q_norm].rejects`` reaches the threshold. When a (template, question) pair accumulates :attr:`PolicyConfig.PER_QUESTION_REJECT_OUT_THRESHOLD` rejects, the entry is removed from ``template.feedback_by_question``. If after removal no remaining entry has a positive accept count, the template itself is deleted from ``templates``. Caller should persist question-level rejection memory separately when needed."""
        if not q_norm:
            return False, False
        counts = template.feedback_by_question.get(q_norm)
        if counts is None or counts.rejects < PolicyConfig.PER_QUESTION_REJECT_OUT_THRESHOLD:
            return False, False
        del template.feedback_by_question[q_norm]
        debug(f"[templates.reject_out_per_question] removed pair: id={template.id} q={q_norm[:40]!r}")
        has_accepted_question = any(c.accepts > 0 for c in template.feedback_by_question.values())
        if has_accepted_question:
            return True, False
        if template.id in templates:
            del templates[template.id]
            debug(f"[templates.reject_out_per_question] deleted template: id={template.id}")
            return True, True
        return True, False

    @staticmethod
    def _active_sandbox_paraphrase_source() -> dict[str, list[str]] | None:
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None and runtime.paraphrase_source is not None:
            return runtime.paraphrase_source
        return SANDBOX_PARAPHRASE_SOURCE

    @staticmethod
    def set_sandbox_paraphrase_source(source: dict[str, list[str]] | None) -> None:
        """Register bundled paraphrase rows keyed by accepted canonical question text."""
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None:
            runtime.paraphrase_source = source
            return
        global SANDBOX_PARAPHRASE_SOURCE
        SANDBOX_PARAPHRASE_SOURCE = source

    @staticmethod
    def clear_sandbox_paraphrase_source() -> None:
        """Clear bundled paraphrase registry when a sandbox session ends."""
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None:
            runtime.paraphrase_source = None
            return
        global SANDBOX_PARAPHRASE_SOURCE
        SANDBOX_PARAPHRASE_SOURCE = None

    @staticmethod
    def _append_distinct_paraphrase_variants(
        vh: ValueHistory, paraphrases: list[str], param_values: dict[str, ParamValue], natural_language: str
    ) -> None:
        """Append zero-accept paraphrase rows that are not already in value history."""
        for p in paraphrases:
            if p in vh.questions:
                continue
            vh.append_question_variant(
                p, accept_count=0, param_values=dict(param_values), natural_language=natural_language
            )

    @staticmethod
    def _append_runtime_paraphrase_variants(
        vh: ValueHistory,
        primary_q: str,
        param_values: dict[str, ParamValue],
        natural_language: str,
        schema: SchemaGraph,
        tables_hint: list[str],
    ) -> None:
        """Generate bounded LLM paraphrases after acceptance and append distinct surviving rows."""
        if EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
            source = TemplateOps._active_sandbox_paraphrase_source()
            if source is None:
                return
            TemplateOps._append_distinct_paraphrase_variants(
                vh, list(source.get(primary_q) or []), param_values, natural_language
            )
            return
        by_style = generate_warmup_paraphrases_by_style(schema, tables_hint, seed_question=primary_q)
        if not by_style:
            return
        per_style_max = SeedWarmupConfig.WARMUP_PARAPHRASES_PER_STYLE_MAX
        for style in SeedWarmupConfig.WARMUP_QUESTION_STYLES:
            cands = by_style.get(style) or []
            if not cands:
                continue
            diverse = select_diverse_paraphrases(cands, max_count=per_style_max)
            for p in diverse:
                ok = validate_question(p).accepted
                if not ok:
                    continue
                TemplateOps._append_distinct_paraphrase_variants(vh, [p], param_values, natural_language)

    @staticmethod
    def handles_referenced_in_sql_param(sql_param: str) -> tuple[str, ...]:
        """Return ordered ``p*`` then ``s*`` handles referenced as bind tokens in SQL."""
        p_keys = sorted({m.group(1) for m in re.finditer(r":(p\d+)", sql_param)}, key=lambda x: int(x[1:]))
        s_keys = sorted({m.group(1) for m in re.finditer(r":(s\d+)", sql_param)}, key=lambda x: int(x[1:]))
        return tuple(p_keys + s_keys)

    @staticmethod
    def _ensure_structural_param_slot(
        slots: dict[str, ParamSlotMeta], key: str, *, column_expr: str, value_type: str = "number"
    ) -> None:
        pk = (key or "").strip()
        if not pk or pk in slots:
            return
        slots[pk] = ParamSlotMeta(
            handle=pk, column_expr=column_expr, op="=", value_type=value_type, upper_handle="", unit_handle=""
        )

    @staticmethod
    def _add_structural_slots_from_group(g: Any, slots: dict[str, ParamSlotMeta], *, label: str) -> None:
        TemplateOps._ensure_structural_param_slot(slots, g.coeff_param_key, column_expr=f"{label} coefficient")
        for idx, pk in enumerate(g.sarg_param_keys or []):
            TemplateOps._ensure_structural_param_slot(slots, pk, column_expr=f"{label} function arg {idx + 1}")
        for idx, pk in enumerate(g.isarg_param_keys or []):
            TemplateOps._ensure_structural_param_slot(slots, pk, column_expr=f"{label} inner function arg {idx + 1}")

    @staticmethod
    def _add_structural_slots_from_expr(expr: Any, slots: dict[str, ParamSlotMeta], *, label: str) -> None:
        if expr is None:
            return
        rendered = expr.prompt_sql() if hasattr(expr, "add_groups") else str(label)
        for g in getattr(expr, "add_groups", None) or []:
            TemplateOps._add_structural_slots_from_group(g, slots, label=rendered or label)
        for g in getattr(expr, "sub_groups", None) or []:
            TemplateOps._add_structural_slots_from_group(g, slots, label=rendered or label)
        for idx, pk in enumerate(getattr(expr, "sarg_param_keys", None) or []):
            TemplateOps._ensure_structural_param_slot(
                slots, pk, column_expr=f"{rendered or label} function arg {idx + 1}"
            )
        for idx, pk in enumerate(getattr(expr, "isarg_param_keys", None) or []):
            TemplateOps._ensure_structural_param_slot(
                slots, pk, column_expr=f"{rendered or label} inner function arg {idx + 1}"
            )
        for ev in getattr(expr, "add_values", None) or []:
            TemplateOps._ensure_structural_param_slot(slots, ev.param_key, column_expr=f"{rendered or label} offset")
        for ev in getattr(expr, "sub_values", None) or []:
            TemplateOps._ensure_structural_param_slot(slots, ev.param_key, column_expr=f"{rendered or label} offset")

    @staticmethod
    def _add_structural_slots_from_where(fp: WhereParam, slots: dict[str, ParamSlotMeta]) -> None:
        TemplateOps._add_structural_slots_from_expr(fp.left_expr, slots, label=fp.left_expr.prompt_sql())
        if fp.right_expr is not None:
            TemplateOps._add_structural_slots_from_expr(fp.right_expr, slots, label=fp.right_expr.prompt_sql())

    @staticmethod
    def _add_structural_slots_from_case_registry(registry: list[Any] | None, slots: dict[str, ParamSlotMeta]) -> None:
        for step in registry or []:
            cw = step.case_when
            if cw is None:
                continue
            for branch in cw.branches or []:
                TemplateOps._add_structural_slots_from_where(branch.condition, slots)
                TemplateOps._add_structural_slots_from_expr(branch.result, slots, label="case result")
            if cw.else_result is not None:
                TemplateOps._add_structural_slots_from_expr(cw.else_result, slots, label="case else")

    @staticmethod
    def _add_structural_slots_from_window_registry(registry: list[Any] | None, slots: dict[str, ParamSlotMeta]) -> None:
        for step in registry or []:
            spec = step.window_spec
            if spec is None:
                continue
            for expr in spec.partition_by or []:
                TemplateOps._add_structural_slots_from_expr(expr, slots, label=expr.prompt_sql())
            for obc in spec.order_by or []:
                TemplateOps._add_structural_slots_from_expr(obc.expr, slots, label=obc.expr.prompt_sql())

    @staticmethod
    def _extend_structural_param_slots(intent_sig: ConcreteIntent, slots: dict[str, ParamSlotMeta]) -> None:
        """Add structural ``s*`` slot metadata using the same traversal as ``extract_structural_params``."""
        for cte in intent_sig.cte_steps or []:
            TemplateOps._ensure_structural_param_slot(slots, cte.limit_param_key, column_expr="LIMIT")
            for sc in cte.select_cols or []:
                TemplateOps._add_structural_slots_from_expr(sc.expr, slots, label=sc.expr.prompt_sql())
            TemplateOps._add_structural_slots_from_window_registry(cte.window_registry, slots)
            TemplateOps._add_structural_slots_from_case_registry(cte.case_registry, slots)
            for g in cte.group_by_cols or []:
                TemplateOps._add_structural_slots_from_expr(g, slots, label=g.prompt_sql())
            for obc in cte.order_by_cols or []:
                TemplateOps._add_structural_slots_from_expr(obc.expr, slots, label=obc.expr.prompt_sql())
            for fp in PredicateGroup.where_leaves(cte.where) or []:
                TemplateOps._add_structural_slots_from_where(fp, slots)
            for hp in PredicateGroup.having_leaves(cte.having) or []:
                TemplateOps._add_structural_slots_from_expr(hp.left_expr, slots, label=hp.left_expr.prompt_sql())
                if hp.right_expr is not None:
                    TemplateOps._add_structural_slots_from_expr(hp.right_expr, slots, label=hp.right_expr.prompt_sql())
        for sc in intent_sig.select_cols or []:
            TemplateOps._add_structural_slots_from_expr(sc.expr, slots, label=sc.expr.prompt_sql())
        TemplateOps._add_structural_slots_from_window_registry(intent_sig.window_registry, slots)
        TemplateOps._add_structural_slots_from_case_registry(intent_sig.case_registry, slots)
        for g in intent_sig.group_by_cols or []:
            TemplateOps._add_structural_slots_from_expr(g, slots, label=g.prompt_sql())
        for obc in intent_sig.order_by_cols or []:
            TemplateOps._add_structural_slots_from_expr(obc.expr, slots, label=obc.expr.prompt_sql())
        for fp in PredicateGroup.where_leaves(intent_sig.where) or []:
            TemplateOps._add_structural_slots_from_where(fp, slots)
        for hp in PredicateGroup.having_leaves(intent_sig.having) or []:
            TemplateOps._add_structural_slots_from_expr(hp.left_expr, slots, label=hp.left_expr.prompt_sql())
            if hp.right_expr is not None:
                TemplateOps._add_structural_slots_from_expr(hp.right_expr, slots, label=hp.right_expr.prompt_sql())
        for key in collect_intent_referenced_param_keys(intent_sig):
            if is_structural_param_key(key) and key not in slots:
                TemplateOps._ensure_structural_param_slot(slots, key, column_expr=key)

    @staticmethod
    def param_keys_from_intent_signature(
        intent_sig: ConcreteIntent, *, literal_structural_only: bool
    ) -> tuple[list[str], list[str]]:
        """Return ordered ``p*`` and ``s*`` handles referenced on a stored intent signature."""
        slot_meta = TemplateOps.collect_param_slot_meta(intent_sig)
        p_keys = sorted((k for k in slot_meta if k.startswith("p") and k[1:].isdigit()), key=lambda x: int(x[1:]))
        s_keys = sorted((k for k in slot_meta if k.startswith("s") and k[1:].isdigit()), key=lambda x: int(x[1:]))
        if literal_structural_only:
            return p_keys, s_keys
        return p_keys, s_keys

    @staticmethod
    def param_slot_prompt_payload(intent_sig: ConcreteIntent, keys: Sequence[str]) -> list[dict[str, str]]:
        """Serialize intent-derived slot metadata for fuzzy reuse parameter extraction."""
        slot_meta = TemplateOps.collect_param_slot_meta(intent_sig)
        out: list[dict[str, str]] = []
        for key in keys:
            meta = slot_meta.get(key)
            if meta is None:
                continue
            out.append(
                {
                    "handle": meta.handle,
                    "column_expr": meta.column_expr,
                    "op": meta.op,
                    "value_type": meta.value_type,
                }
            )
        return out

    @staticmethod
    def collect_param_slot_meta(intent_sig: ConcreteIntent) -> dict[str, ParamSlotMeta]:
        """Map bind handles to predicate metadata from a stored concrete intent."""
        slots: dict[str, ParamSlotMeta] = {}

        def add_filter(fp: WhereParam) -> None:
            pk = (fp.param_key or "").strip()
            if not pk:
                return
            slots[pk] = ParamSlotMeta(
                handle=pk,
                column_expr=fp.left_expr.prompt_sql(),
                op=fp.op,
                value_type=fp.value_type,
                upper_handle=(fp.param_key_hi or "").strip(),
                unit_handle=(fp.param_key_unit or "").strip(),
            )

        def add_having(hp: HavingParam) -> None:
            pk = (hp.param_key or "").strip()
            if not pk:
                return
            slots[pk] = ParamSlotMeta(
                handle=pk,
                column_expr=hp.left_expr.prompt_sql(),
                op=hp.op,
                value_type=hp.value_type,
                upper_handle="",
                unit_handle=(hp.param_key_unit or "").strip(),
            )

        for fp in PredicateGroup.where_leaves(intent_sig.where) or []:
            add_filter(fp)
        for hp in PredicateGroup.having_leaves(intent_sig.having) or []:
            add_having(hp)
        for cte in intent_sig.cte_steps or []:
            for fp in PredicateGroup.where_leaves(cte.where) or []:
                add_filter(fp)
            for hp in PredicateGroup.having_leaves(cte.having) or []:
                add_having(hp)
            lpk = (cte.limit_param_key or "").strip()
            if lpk and lpk not in slots:
                slots[lpk] = ParamSlotMeta(
                    handle=lpk, column_expr="LIMIT", op="=", value_type="number", upper_handle="", unit_handle=""
                )
        lpk_main = (intent_sig.limit_param_key or "").strip()
        if lpk_main and lpk_main not in slots:
            slots[lpk_main] = ParamSlotMeta(
                handle=lpk_main, column_expr="LIMIT", op="=", value_type="number", upper_handle="", unit_handle=""
            )
        TemplateOps._extend_structural_param_slots(intent_sig, slots)
        return slots

    @staticmethod
    def resolve_template_for_question(
        question: str,
        templates: Mapping[str, Template] | list[Template],
        *,
        template_store: dict[str, Any] | TemplateStoreView | None = None,
    ) -> tuple[Template, int] | None:
        """
        Locate a trusted template and history row for *question*.

        Args:

            question: Stored or new question text; normalized before matching.
            templates: Live template map or list.
            template_store: Optional store carrying question indexes.

        Returns:

            ``(template, history_index)`` on hit, else ``None``.
        """
        store_payload: Mapping[str, Any] | None
        if template_store is None:
            store_payload = None
        elif isinstance(template_store, TemplateStoreView):
            store_payload = template_store._indexes
        else:
            store_payload = template_store
        return resolve_template_for_question(question, templates, template_store=store_payload)

    @staticmethod
    def _fallback_display_name(meta: ParamSlotMeta) -> str:
        """Derive a readable label without an LLM when credentials are unavailable."""
        expr = (meta.column_expr or meta.handle).strip()
        if expr.upper() == "LIMIT":
            return "row limit"
        tail = expr.split(".")[-1].strip() if "." in expr else expr
        return tail.replace("_", " ").strip() or meta.handle

    @staticmethod
    def resolve_param_display_names(
        template: Template,
        slots: dict[str, ParamSlotMeta],
        param_values: dict[str, Any],
        *,
        schema: SchemaGraph,
        question_nl: str,
        persist: bool,
        store: dict[str, Any] | TemplateStoreView | None = None,
        templates: dict[str, Template] | None = None,
    ) -> dict[str, str]:
        """
        Return presentation labels for bind handles, caching on the template when *persist* is True.

        Args:

            template: Accepted template whose ``param_display_names`` may already hold labels.
            slots: Handle metadata collected from the intent signature.
            param_values: Current bound values keyed by handle.
            schema: Schema graph for column descriptions in the LLM prompt.
            question_nl: Natural-language question for grounding.
            persist: When True, write newly resolved labels back to the template store.
            store: Template store for persistence.
            templates: In-memory template map for persistence.

        Returns:

            Mapping from handle to short human-readable label.
        """
        cached = dict(getattr(template, "param_display_names", None) or {})
        missing = [h for h in slots if h not in cached]
        if not missing:
            return cached
        if not EngineConfig.llm_credentials_configured():
            for h in missing:
                cached[h] = TemplateOps._fallback_display_name(slots[h])
            template.param_display_names = cached
            if persist and store is not None and templates is not None:
                templates[template.id] = template
                TemplateOps.templates_to_store(store, templates)
            return cached
        table_hints: dict[str, str] = {}
        for tname in template.tables_used or []:
            tmeta = schema.tables.get(tname)
            if tmeta is None:
                continue
            for cname, cmeta in tmeta.columns.items():
                desc = (cmeta.description or "").strip()
                if desc:
                    table_hints[f"{tname}.{cname}"] = desc
        payload_slots = []
        for h in missing:
            meta = slots[h]
            payload_slots.append(
                {
                    "handle": h,
                    "column_expr": meta.column_expr,
                    "operator": meta.op,
                    "value_type": meta.value_type,
                    "sample_value": param_values.get(h),
                }
            )
        system = (
            "You assign short human-readable labels to SQL bind parameters. "
            "Output ONLY valid JSON matching the requested format."
        )
        user = stable_json(
            {
                "task": "For each handle, produce a concise label (two to four words) describing what the parameter represents.",
                "question": question_nl,
                "column_descriptions": table_hints,
                "parameters": payload_slots,
                "rules": [
                    "Use natural domain language, not SQL syntax.",
                    "Do not include the handle id in the label.",
                    "Prefer schema descriptions when available.",
                ],
                "output_format": {"display_names": {h: "label" for h in missing}},
            }
        )
        try:
            raw = LLMProvider.chat(system, user, task="default")
            parsed = safe_json_loads(raw)
            if not parsed or not isinstance(parsed, dict):
                raw2 = LLMProvider.chat(system, user, task="default")
                parsed = safe_json_loads(raw2)
        except MockFixtureMissingError:
            for h in missing:
                cached[h] = TemplateOps._fallback_display_name(slots[h])
            template.param_display_names = cached
            if persist and store is not None and templates is not None:
                templates[template.id] = template
                TemplateOps.templates_to_store(store, templates)
            return cached
        if not isinstance(parsed, dict):
            raise ValueError("resolve_param_display_names: LLM JSON is not an object after retry")
        if "display_names" not in parsed:
            raise ValueError("resolve_param_display_names: LLM JSON missing 'display_names' key")
        names_raw = parsed["display_names"]
        if not isinstance(names_raw, dict):
            raise ValueError(
                f"resolve_param_display_names: 'display_names' must be a dict; got {type(names_raw).__name__}"
            )
        for h in missing:
            label = names_raw.get(h)
            if isinstance(label, str) and label.strip():
                cached[h] = label.strip()
            else:
                cached[h] = TemplateOps._fallback_display_name(slots[h])
        template.param_display_names = cached
        if persist and store is not None and templates is not None:
            templates[template.id] = template
            TemplateOps.templates_to_store(store, templates)
        return cached

    @staticmethod
    def _iter_all_where(intent_sig: ConcreteIntent) -> list[WhereParam]:
        """Yield every filter predicate on the main body and CTE steps."""
        out = list(PredicateGroup.where_leaves(intent_sig.where) or [])
        for cte in intent_sig.cte_steps or []:
            out.extend(PredicateGroup.where_leaves(cte.where) or [])
        return out

    @staticmethod
    def build_parameter_bindings(
        template: Template,
        *,
        history_index: int,
        schema: SchemaGraph,
        question_nl: str = "",
        persist_display_names: bool = False,
        store: dict[str, Any] | TemplateStoreView | None = None,
        templates: dict[str, Template] | None = None,
        param_values_override: dict[str, Any] | None = None,
        schema_context: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
    ) -> tuple[ParameterBinding, ...]:
        """
        Build programmatic parameter bindings for one template history row.

        Args:

            template: Accepted template with intent signature and value history.
            history_index: Row index in ``value_history`` supplying default values.
            schema: Schema graph for display-name resolution.
            question_nl: Natural-language question for display-name grounding.
            persist_display_names: Persist newly resolved labels on the template.
            store: Template store for label persistence.
            templates: In-memory template map for label persistence.
            param_values_override: Optional value map overriding the history row.

        Returns:

            Tuple of :class:`ParameterBinding` rows ordered like SQL bind tokens.
        """
        vh = template.value_history
        idx = max(0, min(history_index, len(vh.questions) - 1)) if vh.questions else 0
        row = dict(vh.param_values[idx]) if vh.param_values else {}
        if param_values_override:
            row.update(param_values_override)
        row = TemplateOps.redact_param_values_for_caller(
            template,
            row,
            schema=schema,
            schema_context=schema_context,
            visible_objects=visible_objects,
        )
        slot_meta = TemplateOps.collect_param_slot_meta(template.intent_signature)
        handles = TemplateOps.handles_referenced_in_sql_param(template.sql_param or "")
        if not handles:
            handles = tuple(slot_meta.keys())
        display_names = TemplateOps.resolve_param_display_names(
            template,
            slot_meta,
            row,
            schema=schema,
            question_nl=question_nl or (vh.natural_language[idx] if vh.natural_language else ""),
            persist=persist_display_names,
            store=store,
            templates=templates,
        )
        bindings: list[ParameterBinding] = []
        for handle in handles:
            if not re.fullmatch(r"p\d+", handle):
                continue
            meta = slot_meta.get(handle)
            current: ParamValue | None = row.get(handle)
            for fp in TemplateOps._iter_all_where(template.intent_signature):
                if (fp.param_key or "").strip() == handle:
                    resolved = fp.resolved_value(row)
                    if resolved is not None:
                        current = resolved
                    break
            label = str(display_names.get(handle) or "").strip() or handle
            col_ref = TemplateOps._param_slot_table_column(meta, template.intent_signature)
            if col_ref is not None and not caller_may_see_column_binding(
                col_ref[0],
                col_ref[1],
                schema,
                schema_context,
                visible_objects,
            ):
                current = None
            bindings.append(
                ParameterBinding(
                    handle=handle,
                    current_value=current,
                    display_name=label,
                    column_expr=str(meta.column_expr) if meta is not None else "",
                )
            )
        return tuple(bindings)

    @staticmethod
    def expand_param_bound_questions(
        vh: ValueHistory,
        *,
        old_params: Mapping[str, Any],
        new_params: Mapping[str, Any],
        donor_questions: Sequence[str],
    ) -> None:
        """Expand stored questions across param value sets using literal replace or 1:1 LLM remap."""
        if not old_params or not new_params or not donor_questions:
            return
        questions = [str(q) for q in donor_questions if str(q).strip()]
        if not questions:
            return
        old_vals = [str(v) for v in old_params.values() if v is not None and str(v)]
        new_vals = [str(v) for v in new_params.values() if v is not None and str(v)]
        if not old_vals or not new_vals or len(old_vals) != len(new_vals):
            return
        forward_literal = all(any(ov in q for q in questions) for ov in old_vals)
        if forward_literal:
            remapped = list(questions)
            for ov, nv in zip(old_vals, new_vals, strict=False):
                if ov == nv:
                    continue
                remapped = [q.replace(ov, nv) for q in remapped]
            for q in remapped:
                vh.add(dict(new_params), q, q, accept_increment=0)
            back = list(questions)
            for ov, nv in zip(old_vals, new_vals, strict=False):
                if ov == nv:
                    continue
                back = [q.replace(nv, ov) for q in back]
            for q in back:
                vh.add(dict(old_params), q, q, accept_increment=0)
            return
        if not EngineConfig.llm_credentials_configured():
            return
        payload = stable_json(
            {
                "questions": questions,
                "old_params": dict(old_params),
                "new_params": dict(new_params),
            }
        )
        try:
            raw = LLMProvider.json(PARAM_QUESTION_REMAP_SYSTEM, payload, task="default")
        except (LlmJsonExhausted, ConfigError, TypeError, ValueError, OSError):
            return
        if not isinstance(raw, dict):
            raise ValueError(f"expand_param_bound_questions: LLM JSON is not an object; got {type(raw).__name__}")
        if "questions" not in raw:
            raise ValueError("expand_param_bound_questions: LLM JSON missing 'questions' key")
        out_qs = raw["questions"]
        if not isinstance(out_qs, list):
            raise ValueError(f"expand_param_bound_questions: 'questions' must be a list; got {type(out_qs).__name__}")
        if len(out_qs) != len(questions):
            raise ValueError(f"expand_param_bound_questions: expected {len(questions)} questions, got {len(out_qs)}")
        for q in out_qs:
            qs = str(q or "").strip()
            if qs:
                vh.add(dict(new_params), qs, qs, accept_increment=0)

    @staticmethod
    def record_value_history_on_accept(
        vh: ValueHistory,
        *,
        param_values: dict[str, ParamValue],
        natural_language: str,
        form_storage: QuestionFormStorage | None,
        q_norm_fallback: str,
        schema: SchemaGraph | None = None,
        tables_hint: list[str] | None = None,
    ) -> None:
        """Append or merge accepted history using typo-corrected text and optional paraphrase rows."""
        primary_q = form_storage.corrected if form_storage is not None else q_norm_fallback
        if param_values and vh.param_values and vh.questions:
            for prior_params, prior_q in zip(vh.param_values, vh.questions, strict=False):
                if not isinstance(prior_params, dict) or not prior_q:
                    continue
                shared_keys = set(prior_params.keys()) & set(param_values.keys())
                if not shared_keys:
                    continue
                if all(prior_params.get(k) == param_values.get(k) for k in shared_keys):
                    continue
                TemplateOps.expand_param_bound_questions(
                    vh,
                    old_params={k: prior_params[k] for k in shared_keys},
                    new_params={k: param_values[k] for k in shared_keys},
                    donor_questions=[str(prior_q), primary_q],
                )
                break
        vh.add(param_values, primary_q, natural_language, accept_increment=1)
        if form_storage is not None and not form_storage.accept_via_normalized_lookup_only:
            nopt = form_storage.normalized_optional
            if nopt and nopt != primary_q and not form_storage.normalized_negative_memory_dropped:
                vh.add(param_values, nopt, natural_language, accept_increment=1)
        if schema is not None and tables_hint and EngineConfig.llm_credentials_configured():
            TemplateOps._append_runtime_paraphrase_variants(
                vh, primary_q, param_values, natural_language, schema, tables_hint
            )

    @staticmethod
    def insert_template(
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Template],
        schema: SchemaGraph,
        q_norm: str,
        intent: RuntimeIntent,
        sql: str,
        dialect: Any | None = None,
        structural_match_templates: list[Template] | None = None,
        *,
        template_source: str = "human",
        template_trust_level: int = TRUST_FLOOR,
        template_initial_stats: TemplateStats | None = None,
        template_value_history: ValueHistory | None = None,
        form_storage: QuestionFormStorage | None = None,
        record_accept: bool = False,
        member_source_id: str | None = None,
        federation_plan_id: str | None = None,
        federation_plan_only: bool = False,
        approval_state: ApprovalState = ApprovalState.APPROVED,
    ) -> Template:
        """Create and insert a template, or merge into a fingerprint- compatible match. Pass ``approval_state=ApprovalState.PENDING`` to persist a draft before user accept."""
        if member_source_id:
            q_norm = member_feedback_q_norm(member_source_id, q_norm)
        sql_canon = canonicalize_sql(sql)
        sql_param_existing = getattr(intent, "sql_param", "") or ""
        if sql_param_existing:
            sql_param = sql_param_existing
        else:
            sql_norm = normalize_sql(sql_canon)
            sg_dialect = TemplateStoreView.sqlglot_dialect_for_template_fingerprint(dialect, member_source_id)
            sql_param, _ = Dialect.parameter_abstract(sql_norm, sqlglot_dialect=sg_dialect)
        sg_dialect = TemplateStoreView.sqlglot_dialect_for_template_fingerprint(dialect, member_source_id)
        sql_fp_val = Dialect.compute_sql_fp(sql_param, sqlglot_dialect=sg_dialect)
        member_engine = ""
        if member_source_id and dialect is not None:
            engine = getattr(dialect, "engine", None)
            if engine is None:
                cfg = getattr(dialect, "config", None)
                engine = getattr(cfg, "TYPE", None) if cfg is not None else None
            if engine:
                member_engine = str(engine)
        tables_used = extract_tables_from_sql(sql_canon, list(schema.tables.keys()), sqlglot_dialect=sg_dialect)
        colmap_sig_val = colmap_signature(intent.column_map)
        ikey = intent_key(intent)

        intent_sig = intent.to_concrete("")

        exec_fp = TemplateRefs.join_fingerprint_from_runtime_intent(intent)
        structural_list = sorted(structural_match_templates or [], key=lambda x: x.id)
        pending_draft = approval_state == ApprovalState.PENDING

        def _merge_accept(t: Template) -> Template:
            debug(f"[templates.insert_template] duplicate_found: id={t.id}")
            t.stats.accept += 1
            prior_counts = t.feedback_by_question.get(q_norm)
            merge_path = int(prior_counts.last_path) if prior_counts and prior_counts.last_path else 1
            TemplateOps.record_per_question_feedback(t, q_norm, accept=True, path=merge_path)
            TemplateOps.promote_trust(t, q_norm)
            TemplateOps._apply_structural_defaults_from_intent(t, intent)
            all_pv = flatten_param_values(intent)
            TemplateOps.record_value_history_on_accept(
                t.value_history,
                param_values=all_pv,
                natural_language=intent.natural_language,
                form_storage=form_storage,
                q_norm_fallback=q_norm,
                schema=schema,
                tables_hint=sorted(intent.tables or []),
            )
            debug(f"[templates.insert_template] value_history_added: entries={len(t.value_history.questions)}")
            return t

        debug(f"[templates.insert_template] checking_duplicate: ikey={ikey[:32]} sql_fp={sql_fp_val[:16]})")

        if not pending_draft:
            if structural_list:
                for t in structural_list:
                    if TemplateOps.template_is_pending(t):
                        continue
                    if TemplateRefs.join_fingerprint_from_concrete_intent(t.intent_signature) == exec_fp:
                        merged = _merge_accept(t)
                        TemplateOps.templates_to_store(store, templates)
                        return merged
            else:
                for t in sorted(templates.values(), key=lambda x: x.id):
                    if TemplateOps.template_is_pending(t):
                        continue
                    if t.intent_key != ikey:
                        continue
                    _, cc, _ = compute_intent_union(intent, t.intent_signature)
                    if cc:
                        continue
                    if TemplateRefs.join_fingerprint_from_concrete_intent(t.intent_signature) != exec_fp:
                        continue
                    merged = _merge_accept(t)
                    TemplateOps.templates_to_store(store, templates)
                    return merged

        debug("[templates.insert_template] no_duplicate: creating_new")

        tid = TemplateOps._reserve_template_id(store)

        all_pv = flatten_param_values(intent)

        if template_initial_stats is not None:
            stats_new = template_initial_stats
        elif pending_draft:
            stats_new = TemplateStats(accept=0, reject=0)
        else:
            stats_new = TemplateStats(accept=1, reject=0)
        if template_value_history is not None:
            vh_new = template_value_history
        else:
            primary_q = form_storage.corrected if form_storage is not None else q_norm
            nl0 = intent.natural_language or ""
            vh_new = ValueHistory(param_values=[all_pv], questions=[primary_q], natural_language=[nl0])
            if (
                form_storage is not None
                and not form_storage.accept_via_normalized_lookup_only
                and form_storage.normalized_optional
                and form_storage.normalized_optional != primary_q
                and not form_storage.normalized_negative_memory_dropped
            ):
                vh_new.append_question_variant(
                    form_storage.normalized_optional, accept_count=1, param_values=all_pv, natural_language=nl0
                )
            if record_accept and intent.tables and EngineConfig.llm_credentials_configured():
                TemplateOps._append_runtime_paraphrase_variants(
                    vh_new, primary_q, all_pv, nl0, schema, sorted(intent.tables or [])
                )

        sig_aliases_insert: dict[str, str] = {}
        for sc in intent.select_cols or []:
            alias = generate_col_alias(sc)
            if alias:
                sig_aliases_insert[sc.signature_key] = alias

        tmpl = Template(
            id=tid,
            schema_graph_id=schema.schema_graph_id,
            effective_structural_hash=schema.effective_structural_hash,
            intent_signature=intent_sig,
            intent_key=ikey,
            tables_used=sorted(tables_used),
            sql_param=sql_param,
            sql_fp=sql_fp_val,
            shape=(intent.sql_shape if intent.sql_shape else sql_shape(sql_canon, intent, sqlglot_dialect=sg_dialect)),
            colmap_sig=colmap_sig_val,
            value_history=vh_new,
            stats=stats_new,
            source=template_source,
            trust_level=template_trust_level,
            structural_defaults={k: v for k, v in all_pv.items() if is_structural_param_key(k)},
            display_alias_map=sig_aliases_insert,
            member_source_id=str(member_source_id or ""),
            member_engine=member_engine,
            federation_plan_id=str(federation_plan_id or ""),
            federation_plan_only=bool(federation_plan_only),
            schema_column_types=TemplateOps._schema_column_types_for_runtime_intent(intent, schema),
        )
        ft, fc = TemplateRefs.footprint_from_refs(TemplateRefs.template_schema_refs(tmpl))
        tmpl.footprint_tables = ft
        tmpl.footprint_columns = fc
        tmpl.approval_state = approval_state

        debug(
            f"[templates.insert_template] CREATED template natural_language={tmpl.value_history.natural_language} chosen_join_candidate_id='{tmpl.chosen_join_candidate_id}' chosen_join_path_signature={tmpl.chosen_join_path_signature}"
        )

        templates[tid] = tmpl
        debug(f"[templates.insert_template] created: id={tid}")
        TemplateOps.templates_to_store(store, templates)
        return tmpl

    @staticmethod
    def _param_slot_table_column(
        meta: ParamSlotMeta | None,
        intent_sig: ConcreteIntent,
    ) -> tuple[str, str] | None:
        if meta is None:
            return None
        expr = str(meta.column_expr or "").strip()
        if not expr or expr.upper() == "LIMIT":
            return None
        if "." in expr:
            table_name, col_name = expr.rsplit(".", 1)
            table_name, col_name = table_name.strip(), col_name.strip()
            if table_name and col_name:
                return table_name, col_name
        for bare, tbl in intent_sig.column_map.items():
            if bare == expr and tbl:
                return tbl, bare
        return None

    @staticmethod
    def redact_param_values_for_caller(
        template: Template,
        param_values: Mapping[str, Any],
        *,
        schema: SchemaGraph,
        schema_context: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Return a copy of bind values with out-of-scope column literals removed."""
        if not param_values:
            return {}
        slot_meta = TemplateOps.collect_param_slot_meta(template.intent_signature)
        redacted = dict(param_values)
        for handle, value in list(param_values.items()):
            if value is None:
                continue
            col_ref = TemplateOps._param_slot_table_column(slot_meta.get(handle), template.intent_signature)
            if col_ref is None:
                continue
            if not caller_may_see_column_binding(
                col_ref[0],
                col_ref[1],
                schema,
                schema_context,
                visible_objects,
            ):
                redacted.pop(handle, None)
        return redacted

    @staticmethod
    def template_footprint_tables(template: Template) -> frozenset[str]:
        """Return table names referenced by one template footprint."""
        return template_footprint_tables(template)

    @staticmethod
    def template_enumerable_by_caller(
        template: Template,
        *,
        visible_tables: frozenset[str] | None,
    ) -> bool:
        """Return whether *template* may appear in caller-facing enumeration APIs."""
        return template_enumerable_by_caller(template, visible_tables=visible_tables)

    @staticmethod
    def template_visible_to_callers(template: Template) -> bool:
        """Return whether *template* may appear in caller-facing enumeration APIs."""
        return template_visible_to_callers(template)

    @staticmethod
    def template_is_pending(template: Template) -> bool:
        """Return True when *template* is awaiting user approval."""
        approval = getattr(template, "approval_state", None)
        if approval is None:
            return False
        return str(getattr(approval, "value", approval)).lower() == ApprovalState.PENDING.value

    @staticmethod
    def find_pending_template_for_question(
        templates: Mapping[str, Template],
        q_norm: str,
    ) -> Template | None:
        """Return the first pending template whose stored questions include *q_norm*."""
        target = str(q_norm or "").strip()
        if not target:
            return None
        for tmpl in sorted(templates.values(), key=lambda t: t.id):
            if not TemplateOps.template_is_pending(tmpl):
                continue
            for hist_q in tmpl.value_history.questions:
                if hist_q and hist_q == target:
                    return tmpl
        return None

    @staticmethod
    def upsert_pending_template(
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Template],
        schema: SchemaGraph,
        q_norm: str,
        intent: RuntimeIntent,
        sql: str,
        *,
        dialect: Any | None = None,
        form_storage: QuestionFormStorage | None = None,
        member_source_id: str | None = None,
    ) -> Template:
        """Persist or refresh a pending draft for *q_norm* before user accept."""
        existing = TemplateOps.find_pending_template_for_question(templates, q_norm)
        if existing is not None:
            templates.pop(existing.id, None)
            if isinstance(store, TemplateStoreView):
                store.remove_template_id(existing.id)
            else:
                body = store.get("templates")
                if isinstance(body, dict):
                    body.pop(existing.id, None)
        return TemplateOps.insert_template(
            store,
            templates,
            schema,
            q_norm,
            intent,
            sql,
            dialect=dialect,
            form_storage=form_storage,
            record_accept=False,
            member_source_id=member_source_id,
            approval_state=ApprovalState.PENDING,
            template_initial_stats=TemplateStats(accept=0, reject=0),
        )

    @staticmethod
    def approve_pending_template(
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Template],
        tmpl: Template,
        *,
        intent: RuntimeIntent,
        q_norm: str,
        form_storage: QuestionFormStorage | None = None,
        schema: SchemaGraph | None = None,
    ) -> Template:
        """Promote a pending draft to approved and record accept history."""
        tmpl.approval_state = ApprovalState.APPROVED
        tmpl.stats.accept = max(int(tmpl.stats.accept), 0) + 1
        TemplateOps.record_per_question_feedback(tmpl, q_norm, accept=True, path=1)
        TemplateOps.promote_trust(tmpl, q_norm)
        TemplateOps._apply_structural_defaults_from_intent(tmpl, intent)
        all_pv = flatten_param_values(intent)
        TemplateOps.record_value_history_on_accept(
            tmpl.value_history,
            param_values=all_pv,
            natural_language=intent.natural_language,
            form_storage=form_storage,
            q_norm_fallback=q_norm,
            schema=schema,
            tables_hint=sorted(intent.tables or []),
        )
        TemplateOps.templates_to_store(store, templates)
        return tmpl

    @staticmethod
    def delete_pending_templates_for_question(
        store: dict[str, Any] | TemplateStoreView,
        templates: dict[str, Template],
        q_norm: str,
    ) -> int:
        """Remove pending drafts whose stored questions include *q_norm*."""
        target = str(q_norm or "").strip()
        if not target:
            return 0
        removed = 0
        for tid in list(templates.keys()):
            tmpl = templates.get(tid)
            if tmpl is None or not TemplateOps.template_is_pending(tmpl):
                continue
            if not any(hist_q == target for hist_q in tmpl.value_history.questions if hist_q):
                continue
            templates.pop(tid, None)
            if isinstance(store, TemplateStoreView):
                store.remove_template_id(tid)
            else:
                body = store.get("templates")
                if isinstance(body, dict):
                    body.pop(tid, None)
            removed += 1
        if removed:
            TemplateOps.templates_to_store(store, templates)
        return removed

    @staticmethod
    def stored_template_use_count(template: Template) -> int:
        """Return the caller-visible reuse counter for *template*."""
        if template.stats.accept:
            return int(template.stats.accept)
        return sum(int(x) for x in template.value_history.accept_counts)

    @staticmethod
    def template_display_sql(template: Template, dialect: Any) -> str:
        """Return user-facing display SQL for one stored template."""
        rt = template.intent_signature.to_runtime_skeleton()
        return build_display_sql(template.sql_param, rt, template.display_alias_map or None, dialect=dialect)

    @staticmethod
    def resolve_template_ref(template_ref: str, templates: Mapping[str, Template]) -> Template | None:
        """Resolve *template_ref* by stable template id only."""
        ref = str(template_ref).strip()
        if not ref:
            return None
        return templates.get(ref)

    @staticmethod
    def summarize_stored_template(
        template: Template,
        *,
        space: str,
        dialect: Any,
    ) -> StoredTemplateSummary:
        """Build one summary row for caller-facing template enumeration."""
        stamped = TemplateRefs.stamp_template_footprint(template)
        approval = (
            stamped.approval_state.value
            if isinstance(stamped.approval_state, ApprovalState)
            else str(stamped.approval_state or ApprovalState.APPROVED.value)
        )
        return StoredTemplateSummary(
            id=stamped.id,
            approval_state=approval,
        )

    @staticmethod
    def build_stored_template_detail(
        template: Template,
        *,
        space: str,
        schema: SchemaGraph,
        dialect: Any,
        history_index: int = 0,
        schema_context: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
    ) -> StoredTemplateDetail:
        """Build full caller-visible detail for one stored template."""
        summary = TemplateOps.summarize_stored_template(template, space=space, dialect=dialect)
        bindings = TemplateOps.build_parameter_bindings(
            template,
            history_index=history_index,
            schema=schema,
            persist_display_names=False,
            schema_context=schema_context,
            visible_objects=visible_objects,
        )
        stamped = TemplateRefs.stamp_template_footprint(template)
        approval = (
            stamped.approval_state.value
            if isinstance(stamped.approval_state, ApprovalState)
            else str(stamped.approval_state or ApprovalState.APPROVED.value)
        )
        return StoredTemplateDetail(
            summary=summary,
            parameters=tuple(bindings),
            approval_state=approval,
        )

    @staticmethod
    def caller_scoped_templates(
        templates: Mapping[str, Template] | list[Template],
        *,
        visible_tables: frozenset[str] | None,
    ) -> list[Template]:
        """Return templates whose footprint is visible to the caller."""
        if isinstance(templates, Mapping):
            return list(TemplateOps.list_callable_templates(templates, visible_tables=visible_tables))
        return [t for t in templates if TemplateOps.template_enumerable_by_caller(t, visible_tables=visible_tables)]

    @staticmethod
    def list_callable_templates(
        templates: Mapping[str, Template],
        *,
        visible_tables: frozenset[str] | None = None,
    ) -> tuple[Template, ...]:
        """Return accepted templates visible to programmatic callers, sorted by id."""
        visible = [
            t for t in templates.values() if TemplateOps.template_enumerable_by_caller(t, visible_tables=visible_tables)
        ]
        return tuple(sorted(visible, key=lambda t: t.id))

    @staticmethod
    def list_stored_template_summaries(
        templates: Mapping[str, Template],
        *,
        space: str,
        dialect: Any,
        visible_tables: frozenset[str] | None = None,
    ) -> tuple[StoredTemplateSummary, ...]:
        """Enumerate caller-visible template summaries for one namespace."""
        return tuple(
            TemplateOps.summarize_stored_template(t, space=space, dialect=dialect)
            for t in TemplateOps.list_callable_templates(templates, visible_tables=visible_tables)
        )

    @staticmethod
    def parse_schema_migration_map_payload(payload: dict[str, Any]) -> SchemaMigrationMap:
        """Normalise a decoded JSON object into a :class:`SchemaMigrationMap`."""
        require_exact_keys(
            payload,
            allowed=frozenset(
                {
                    "version",
                    "action",
                    "table_renames",
                    "column_renames",
                    "dropped_tables",
                    "dropped_columns",
                    "added_tables",
                    "added_columns",
                    "cross_table_column_moves",
                    "fk_remaps",
                    "pk_remaps",
                    "rename_confidence",
                    "refresh_existing_descriptions_on_addition",
                    "operator_notes",
                }
            ),
            required=frozenset({"version", "action"}),
            context="schema migration map",
        )
        ver = payload.get("version")
        if not isinstance(ver, int) or ver < 1:
            raise MigrationPendingError("schema_migration_map.json: invalid or missing version")
        action_raw = str(payload.get("action") or "").strip().lower()
        if action_raw not in (MIGRATION_MAP_ACTION_REMAP, MIGRATION_MAP_ACTION_DESTRUCTIVE, MIGRATION_MAP_ACTION_ABORT):
            raise MigrationPendingError(f"schema_migration_map.json: unsupported action {action_raw!r}")
        tables_o: list[SchemaMigrationMapEntry] = []
        tr_raw = payload.get("table_renames")
        if isinstance(tr_raw, dict):
            for fk, tk in tr_raw.items():
                fo = str(fk).strip()
                tn = str(tk).strip()
                if fo and tn:
                    tables_o.append(SchemaMigrationMapEntry(entry_type="table", from_name=fo, to_name=tn))
        elif isinstance(tr_raw, list):
            for row in tr_raw:
                if not isinstance(row, dict):
                    continue
                fo = str(row.get("from") or row.get("old") or "").strip()
                tn = str(row.get("to") or row.get("new") or "").strip()
                if fo and tn:
                    tables_o.append(SchemaMigrationMapEntry(entry_type="table", from_name=fo, to_name=tn))
        cols_o: list[SchemaMigrationMapEntry] = []
        cr_raw = payload.get("column_renames")
        if isinstance(cr_raw, dict):
            for tbl, inner in cr_raw.items():
                bt = str(tbl).strip()
                if not isinstance(inner, dict) or not bt:
                    continue
                for oc, nc in inner.items():
                    ocn = str(oc).strip()
                    ncn = str(nc).strip()
                    if ocn and ncn:
                        cols_o.append(
                            SchemaMigrationMapEntry(entry_type="column", table=bt, from_name=ocn, to_name=ncn)
                        )
        elif isinstance(cr_raw, list):
            for row in cr_raw:
                if not isinstance(row, dict):
                    continue
                bt = str(row.get("table") or "").strip()
                fo = str(row.get("from") or row.get("old") or "").strip()
                tn = str(row.get("to") or row.get("new") or "").strip()
                if bt and fo and tn:
                    cols_o.append(SchemaMigrationMapEntry(entry_type="column", table=bt, from_name=fo, to_name=tn))
        dropped_t: list[str] = []
        for x in payload.get("dropped_tables") or []:
            s = str(x).strip()
            if s:
                dropped_t.append(s)
        dropped_c: list[SchemaMigrationMapEntry] = []
        for x in payload.get("dropped_columns") or []:
            if isinstance(x, str) and "." in x:
                tpart, cpart = x.split(".", 1)
                tpart, cpart = tpart.strip(), cpart.strip()
                if tpart and cpart:
                    dropped_c.append(
                        SchemaMigrationMapEntry(entry_type="dropped_column", table=tpart, from_name=cpart, to_name="")
                    )
        added_tb: list[str] = []
        for x in payload.get("added_tables") or []:
            s = str(x).strip()
            if s:
                added_tb.append(s)
        added_c: list[SchemaMigrationMapEntry] = []
        ac_raw = payload.get("added_columns")
        if isinstance(ac_raw, dict):
            for tbl, inner in ac_raw.items():
                bt = str(tbl).strip()
                if not isinstance(inner, dict) or not bt:
                    continue
                for nc in inner.keys():
                    ncn = str(nc).strip()
                    if ncn:
                        added_c.append(
                            SchemaMigrationMapEntry(entry_type="added_column", table=bt, to_name=ncn, from_name="")
                        )
        refresh_desc = payload.get("refresh_existing_descriptions_on_addition", False)
        if not isinstance(refresh_desc, bool):
            raise MigrationPendingError(
                "schema_migration_map.json: refresh_existing_descriptions_on_addition must be a boolean"
            )
        moves_raw = payload.get("cross_table_column_moves")
        if isinstance(moves_raw, list) and moves_raw:
            raise MigrationPendingError(
                "schema_migration_map.json: cross_table_column_moves is not supported; "
                "record moves under operator_notes and remap via table/column renames or drops"
            )
        fk_remaps_o: list[SchemaMigrationMapEntry] = []
        for row in payload.get("fk_remaps") or []:
            if not isinstance(row, dict):
                continue
            child = str(row.get("child_table") or row.get("table") or "").strip()
            old_parent = str(row.get("from_parent") or row.get("old_parent") or "").strip()
            new_parent = str(row.get("to_parent") or row.get("new_parent") or "").strip()
            if child and old_parent and new_parent:
                fk_remaps_o.append(
                    SchemaMigrationMapEntry(
                        entry_type="fk_remap", table=child, from_name=old_parent, to_name=new_parent
                    )
                )
        pk_remaps_o: list[SchemaMigrationMapEntry] = []
        for row in payload.get("pk_remaps") or []:
            if not isinstance(row, dict):
                continue
            tbl = str(row.get("table") or "").strip()
            old_pk = str(row.get("from_pk") or row.get("old_pk") or "").strip()
            new_pk = str(row.get("to_pk") or row.get("new_pk") or "").strip()
            if tbl and old_pk and new_pk:
                pk_remaps_o.append(
                    SchemaMigrationMapEntry(entry_type="pk_remap", table=tbl, from_name=old_pk, to_name=new_pk)
                )
        return SchemaMigrationMap(
            version=ver,
            action=action_raw,
            table_renames=tuple(tables_o),
            column_renames=tuple(cols_o),
            dropped_tables=tuple(dropped_t),
            dropped_columns=tuple(dropped_c),
            added_tables=tuple(added_tb),
            added_columns=tuple(added_c),
            fk_remaps=tuple(fk_remaps_o),
            pk_remaps=tuple(pk_remaps_o),
            refresh_existing_descriptions_on_addition=refresh_desc,
        )

    @staticmethod
    def load_schema_migration_map(cwd_path: Path) -> SchemaMigrationMap | None:
        """Read ``schema_migration_map.json`` from *cwd_path* and parse it. into a :class:`SchemaMigrationMap`. Returns None when the file is absent."""
        path = cwd_path / MIGRATION_MAP_FILENAME
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MigrationPendingError(f"schema_migration_map.json: cannot read or parse: {exc}") from exc
        if not isinstance(raw, dict):
            raise MigrationPendingError("schema_migration_map.json: root must be a JSON object")
        return TemplateOps.parse_schema_migration_map_payload(raw)

    @staticmethod
    def validate_schema_migration_map(
        map_obj: SchemaMigrationMap, cached_schema: SchemaGraph | None, live_schema: SchemaGraph
    ) -> None:
        """Check rename and drop entries against the cached and live schema. graphs. Raises:class:`MigrationPendingError` with prefix ``STALE_MAP:`` when the map was produced for a different cached snapshot so the file should be removed and init retried without it."""
        problems: list[str] = []
        stale: list[str] = []
        tmap = {
            e.from_name: e.to_name
            for e in map_obj.table_renames
            if e.entry_type == "table" and e.from_name and e.to_name
        }

        def _new_table(old: str) -> str:
            return tmap.get(old, old)

        if cached_schema is not None:
            for e in map_obj.table_renames:
                if e.entry_type != "table":
                    continue
                if e.from_name not in cached_schema.tables:
                    stale.append(f"table rename source {e.from_name!r} not in cached schema")
            for e in map_obj.column_renames:
                if e.entry_type != "column":
                    continue
                if e.table not in cached_schema.tables:
                    stale.append(f"column rename table {e.table!r} not in cached schema")
                elif e.from_name not in cached_schema.tables[e.table].columns:
                    stale.append(f"column {e.table}.{e.from_name} not in cached schema")
            for dt in map_obj.dropped_tables:
                if dt not in cached_schema.tables:
                    stale.append(f"dropped_tables entry {dt!r} not in cached schema")
            for e in map_obj.dropped_columns:
                if e.entry_type != "dropped_column":
                    continue
                if e.table not in cached_schema.tables or e.from_name not in cached_schema.tables[e.table].columns:
                    stale.append(f"dropped_columns entry {e.table}.{e.from_name} not in cached schema")
        else:
            if map_obj.action != MIGRATION_MAP_ACTION_ABORT and (
                map_obj.table_renames or map_obj.column_renames or map_obj.dropped_tables or map_obj.dropped_columns
            ):
                problems.append("cached schema snapshot missing; cannot validate migration map sources")

        for e in map_obj.table_renames:
            if e.entry_type != "table":
                continue
            if e.to_name not in live_schema.tables:
                problems.append(f"table rename target {e.to_name!r} not in live schema")
        for e in map_obj.column_renames:
            if e.entry_type != "column":
                continue
            nt = _new_table(e.table)
            if nt not in live_schema.tables:
                problems.append(f"column rename live table {nt!r} not in live schema")
            elif e.to_name not in live_schema.tables[nt].columns:
                problems.append(f"column rename target {nt}.{e.to_name} not in live schema")
        for tname in map_obj.added_tables:
            if tname not in live_schema.tables:
                problems.append(f"added_tables informational entry {tname!r} not in live schema")
        if stale and not problems:
            raise MigrationPendingError("STALE_MAP: " + "; ".join(stale))
        if stale and problems:
            problems.extend(stale)
        if problems:
            raise MigrationPendingError("schema_migration_map.json validation failed: " + "; ".join(problems))

    @staticmethod
    def _schema_diff_from_user_drops(map_obj: SchemaMigrationMap) -> SchemaDiff | None:
        """Build a :class:`SchemaDiff` describing only user-listed drops for surgical invalidation."""
        if not map_obj.dropped_tables and not map_obj.dropped_columns:
            return None
        col_drop_by_table: dict[str, set[str]] = defaultdict(set)
        for e in map_obj.dropped_columns:
            if e.entry_type != "dropped_column":
                continue
            col_drop_by_table[e.table].add(e.from_name)
        per_table: dict[str, TableDiff] = {
            t: TableDiff(dropped_columns=tuple(sorted(col_drop_by_table[t]))) for t in col_drop_by_table
        }
        return SchemaDiff(dropped_tables=tuple(sorted(set(map_obj.dropped_tables))), per_table=per_table)

    @staticmethod
    def _schema_diff_for_sidecar_renames(map_obj: SchemaMigrationMap) -> SchemaDiff | None:
        """Build a rename-only :class:`SchemaDiff` for overrides sidecar migration."""
        tr = tuple(
            (e.from_name, e.to_name)
            for e in map_obj.table_renames
            if e.entry_type == "table" and e.from_name != e.to_name
        )
        col_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        tmap = {a: b for a, b in tr}

        def _new_t(old: str) -> str:
            return tmap.get(old, old)

        for e in map_obj.column_renames:
            if e.entry_type != "column":
                continue
            nt = _new_t(e.table)
            col_groups[nt].append((e.from_name, e.to_name))
        per_table: dict[str, TableDiff] = {}
        for nt, pairs in col_groups.items():
            per_table[nt] = TableDiff(renamed_columns=tuple(sorted(set(pairs))))
        if not tr and not per_table:
            return None
        return SchemaDiff(table_renames=tr, per_table=per_table)

    @staticmethod
    def _migration_map_checkpoint_begin(artifacts_dir: str, *, schema_json_path: Path | None = None) -> str | None:
        chk = tempfile.mkdtemp(prefix=".migration_checkpoint_", dir=artifacts_dir)
        copied = False
        manifest_path = os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)
        if os.path.isfile(manifest_path):
            shutil.copy2(manifest_path, os.path.join(chk, ARTIFACT_MANIFEST_FILENAME))
            copied = True
        store_path = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
        if os.path.isdir(store_path):
            shutil.copytree(store_path, os.path.join(chk, TEMPLATE_STORE_SEGMENT))
            copied = True
        if schema_json_path is not None and schema_json_path.is_file():
            shutil.copy2(schema_json_path, os.path.join(chk, MIGRATION_CHECKPOINT_SCHEMA_BASENAME))
            copied = True
        aetherspaces_path = os.path.join(artifacts_dir, AETHERSPACES_SEGMENT)
        if os.path.isdir(aetherspaces_path):
            shutil.copytree(aetherspaces_path, os.path.join(chk, AETHERSPACES_SEGMENT))
            copied = True
        if not copied:
            shutil.rmtree(chk, ignore_errors=True)
            return None
        return chk

    @staticmethod
    def apply_schema_migration_map(
        map_obj: SchemaMigrationMap, artifacts_dir: str, schema: SchemaGraph, schema_json_path: Path
    ) -> MigrationReport:
        """Apply a user migration map to templates, simulation artifacts, and optionally the overrides sidecar. Dispatches on ``action``. ``remap`` runs rename migration plus surgical drops listed in the map; the learning-reset action clears learning artifacts. Does not delete the editor JSON; callers unlink ``schema_migration_map.json`` after success."""
        if map_obj.action == MIGRATION_MAP_ACTION_ABORT:
            raise MigrationPendingError("internal: abort action must be handled before apply_schema_migration_map")
        if map_obj.action == MIGRATION_MAP_ACTION_DESTRUCTIVE:
            with artifact_lock(artifacts_dir):
                checkpoint = TemplateOps._migration_map_checkpoint_begin(
                    artifacts_dir, schema_json_path=schema_json_path
                )
                try:
                    destroyed = TemplateOps._disk_template_row_count(artifacts_dir)
                    destructive_migration_execute(artifacts_dir, schema)
                    TemplateOps._stamp_manifest(
                        artifacts_dir,
                        schema,
                        tier=MigrationTier.DESTRUCTIVE,
                        last_action=ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP,
                    )
                except Exception:
                    if checkpoint is not None:
                        TemplateOps._migration_map_checkpoint_restore(
                            artifacts_dir, checkpoint, schema_json_path=schema_json_path
                        )
                    raise
                finally:
                    TemplateOps._migration_map_checkpoint_cleanup(checkpoint)
            return MigrationReport(tier=MigrationTier.DESTRUCTIVE, destroyed_templates=destroyed)
        renamed_tables = tuple((e.from_name, e.to_name) for e in map_obj.table_renames if e.entry_type == "table")
        renamed_columns = tuple(
            (e.table, e.from_name, e.to_name) for e in map_obj.column_renames if e.entry_type == "column"
        )
        with artifact_lock(artifacts_dir):
            schema_backup = copy.deepcopy(schema)
            checkpoint = TemplateOps._migration_map_checkpoint_begin(artifacts_dir, schema_json_path=schema_json_path)
            try:
                sidecar_diff = TemplateOps._schema_diff_for_sidecar_renames(map_obj)
                if sidecar_diff is not None or map_obj.fk_remaps or map_obj.pk_remaps:
                    migrate_sidecar_for_diff(
                        schema_json_path,
                        sidecar_diff or SchemaDiff(),
                        fk_remaps=map_obj.fk_remaps,
                        pk_remaps=map_obj.pk_remaps,
                    )
                drop_diff = TemplateOps._schema_diff_from_user_drops(map_obj)
                if drop_diff is not None:
                    migrate_sidecar_for_diff(schema_json_path, drop_diff)
                reconcile_sidecar_against_graph(schema, schema_json_path)
                if map_obj.fk_remaps:
                    apply_fk_remaps_to_graph(schema, map_obj.fk_remaps)
                if map_obj.pk_remaps:
                    apply_pk_remaps_to_graph(schema, map_obj.pk_remaps)
                remapped = 0
                destroyed = 0
                if renamed_tables or renamed_columns:
                    remapped, destroyed = TemplateOps._apply_schema_rename_migration_to_store(
                        artifacts_dir, schema, renamed_tables, renamed_columns
                    )
                surg = 0
                drop_diff = TemplateOps._schema_diff_from_user_drops(map_obj)
                if drop_diff is not None:
                    surg = TemplateOps.surgical_invalidate_templates_by_diff(artifacts_dir, schema, drop_diff)
                TemplateOps.apply_structural_migration_from_map(artifacts_dir, map_obj)
                dropped_columns = tuple(
                    f"{e.table}.{e.from_name}" for e in map_obj.dropped_columns if e.entry_type == "dropped_column"
                )
                migrate_engine_knowledge_artifacts(
                    artifacts_dir,
                    schema,
                    schema_json_path=str(schema_json_path),
                    dropped_tables=tuple(map_obj.dropped_tables),
                    dropped_columns=dropped_columns,
                    table_renames=renamed_tables,
                    column_renames=renamed_columns,
                )
                TemplateOps._stamp_manifest(
                    artifacts_dir, schema, tier=MigrationTier.REMAP, last_action=ARTIFACT_LAST_ACTION_REMAP_USER_MAP
                )
            except Exception:
                if checkpoint is not None:
                    TemplateOps._migration_map_checkpoint_restore(
                        artifacts_dir, checkpoint, schema_json_path=schema_json_path
                    )
                TemplateOps._restore_schema_graph_from_backup(schema, schema_backup)
                raise
            finally:
                TemplateOps._migration_map_checkpoint_cleanup(checkpoint)
        return MigrationReport(
            tier=MigrationTier.REMAP,
            renamed_tables=renamed_tables,
            renamed_columns=renamed_columns,
            remapped_templates=remapped,
            destroyed_templates=destroyed,
            surgically_invalidated=surg,
            dropped_tables=tuple(map_obj.dropped_tables),
        )

    @staticmethod
    def build_schema_migration_map_document(
        *,
        tier: MigrationTier,
        schema_diff: SchemaDiff | None,
        rename_plan: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None,
        previous_schema: SchemaGraph | None = None,
        schema: SchemaGraph | None = None,
    ) -> dict[str, Any]:
        """Build a schema migration map document for operator review or programmatic apply."""
        if tier == MigrationTier.DESTRUCTIVE and (schema_diff is None or not schema_diff_is_additive_only(schema_diff)):
            action = MIGRATION_MAP_ACTION_DESTRUCTIVE
        else:
            action = MIGRATION_MAP_ACTION_REMAP
        table_renames: dict[str, str] = {}
        column_renames: dict[str, dict[str, str]] = {}
        rename_confidence: float | None = None
        if rename_plan is not None:
            rt, rc = rename_plan
            table_renames = {o: n for o, n in rt}
            for ot, oc, nc in rc:
                column_renames.setdefault(ot, {})[oc] = nc
            if previous_schema is not None and schema is not None:
                rename_confidence = rename_migration_plan_confidence(previous_schema, schema)
        dropped_tables: list[str] = []
        dropped_columns: list[str] = []
        added_tables: list[str] = []
        added_columns: dict[str, list[str]] = {}
        cross_table_column_moves: list[dict[str, str]] = []
        operator_notes: list[str] = []
        if schema_diff is not None:
            dropped_tables = list(schema_diff.dropped_tables)
            for tname, td in schema_diff.per_table.items():
                for c in td.dropped_columns:
                    dropped_columns.append(f"{tname}.{c}")
            added_tables = list(schema_diff.added_tables)
            for tname, td in schema_diff.per_table.items():
                if td.added_columns:
                    added_columns[tname] = list(td.added_columns)
            for src_table, src_col, dst_table, dst_col in schema_diff.cross_table_column_moves:
                operator_notes.append(
                    f"cross-table column move detected: {src_table}.{src_col} -> {dst_table}.{dst_col} "
                    "(not auto-applied; remap manually)"
                )
            note = schema_diff_cross_table_limitation_note(schema_diff)
            if note:
                operator_notes.append(note)
        payload: dict[str, Any] = {
            "version": 1,
            "action": action,
            "table_renames": table_renames,
            "column_renames": column_renames,
            "dropped_tables": dropped_tables,
            "dropped_columns": dropped_columns,
            "added_tables": added_tables,
            "added_columns": added_columns,
            "cross_table_column_moves": cross_table_column_moves,
            "fk_remaps": [],
            "pk_remaps": [],
            "rename_confidence": rename_confidence,
            "refresh_existing_descriptions_on_addition": False,
        }
        if operator_notes:
            payload["operator_notes"] = operator_notes
        if schema_diff is not None:
            for dt in dropped_tables:
                payload["fk_remaps"].append(
                    {
                        "child_table": "<child_with_orphaned_fk>",
                        "from_parent": dt,
                        "to_parent": "<new_parent_table>",
                    }
                )
                payload["pk_remaps"].append(
                    {
                        "table": "<table_whose_pk_moved>",
                        "from_pk": "<old_pk_columns_csv>",
                        "to_pk": "<new_pk_columns_csv>",
                    }
                )
        return payload

    @staticmethod
    def export_schema_migration_map_skeleton(
        cwd_path: Path,
        *,
        tier: MigrationTier,
        schema_diff: SchemaDiff | None,
        rename_plan: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None,
        previous_schema: SchemaGraph | None = None,
        schema: SchemaGraph | None = None,
    ) -> Path:
        """Write ``schema_migration_map.json`` with auto-detected fields pre-filled. The file is written in the process working directory resolved by *cwd_path*."""
        path = cwd_path / MIGRATION_MAP_FILENAME
        payload = TemplateOps.build_schema_migration_map_document(
            tier=tier,
            schema_diff=schema_diff,
            rename_plan=rename_plan,
            previous_schema=previous_schema,
            schema=schema,
        )
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return path


class TemplateOps(TemplateStoreLifecycleOps, TemplateLearningOps):
    """Public template operations surface."""
