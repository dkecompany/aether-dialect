"""Template store I/O, trust levels, per-question feedback memory, and schema-ref helpers."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ._config import (
    ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP,
    ARTIFACT_LAST_ACTION_REMAP_USER_MAP,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_ACTION_DESTRUCTIVE,
    MIGRATION_MAP_ACTION_REMAP,
    MIGRATION_MAP_FILENAME,
    RUNTIME_PARAPHRASE_COUNT,
    SHAPE_QUESTION_INDEX_KEY,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_PARTITION_COUNT,
    TEMPLATE_STORE_PARTITION_LRU_MAX,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SEGMENT,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    TRUST_AUTO_ACCEPT_THRESHOLD,
    TRUST_CEILING,
    TRUST_FLOOR,
    EngineConfig,
    GenerationPath,
    PolicyConfig,
    SeedWarmupConfig,
    is_structural_param_key,
    llm_credentials_configured,
)
from ._contracts_base import (
    MigrationPendingError,
    MigrationReport,
    MigrationTier,
    SchemaGraph,
    SchemaMigrationMap,
    SchemaMigrationMapEntry,
    TemplateStats,
)
from ._contracts_core import (
    ConcreteCteStep,
    ConcreteIntent,
    FeedbackCounts,
    FeedbackKind,
    ParamValue,
    QuestionFeedbackEntry,
    QuestionFormStorage,
    RejectionBucket,
    RuntimeIntent,
    SeedWarmupIntent,
    Template,
    ValueHistory,
    runtime_intent_to_concrete,
)
from ._core_utils import (
    artifact_lock,
    canonicalize_sql,
    classify_migration_tier,
    colmap_signature,
    debug,
    llm_chat,
    normalize_sql,
    read_artifact_manifest,
    read_gzip_json,
    stable_json,
    try_rename_migration_plan,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from ._dialect import (
    active_sqlglot_dialect,
    compute_sql_fp,
    parameter_abstract,
)
from ._intent_process import register_templates_module
from ._intent_resolve import (
    compute_intent_union,
    join_path_key_concrete,
    join_path_key_runtime,
)
from ._schema import (
    SchemaDiff,
    TableDiff,
    _migrate_sidecar_for_diff,
    destructive_migration_execute,
)
from ._sql_gen import generate_col_alias
from ._utils import (
    body_similarity_key_for_concrete,
    build_shape_question_index,
    extract_tables_from_sql,
    flatten_param_values,
    generate_paraphrases_of_seed_question,
    intent_key,
    is_exact_question_text_match,
    question_token_fingerprint_from_raw,
    sql_shape,
    validate_question,
)


def _build_intent_key_index_for_templates(
    templates: list[Template],
) -> dict[str, list[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for t in templates:
        ik = (t.intent_key or "").strip()
        if ik:
            idx[ik].add(str(t.id))
    return {k: sorted(v) for k, v in idx.items()}


def _build_union_family_index_for_templates(
    templates: list[Template],
) -> dict[str, list[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for t in templates:
        bk = body_similarity_key_for_concrete(t.intent_signature)
        jk = join_path_key_concrete(t.intent_signature)
        idx[bk].add(str(t.id))
        idx[f"{bk}|{jk}"].add(str(t.id))
    return {k: sorted(v) for k, v in idx.items()}


def _build_question_token_index_for_templates(
    templates: list[Template],
) -> dict[str, list[list[str]]]:
    idx: dict[str, list[list[str]]] = defaultdict(list)
    for t in templates:
        tid = str(t.id)
        for hi, q in enumerate(t.value_history.questions or []):
            if not q:
                continue
            fp = question_token_fingerprint_from_raw(q)
            idx[fp].append([tid, str(hi)])
    return dict(idx)


def template_partition_number(template_id: str) -> int:
    """Return stable partition index ``0..255`` for *template_id* (SHA-256 first byte)."""

    return int(hashlib.sha256(template_id.encode("utf-8")).hexdigest()[:2], 16)


class _TemplateBodiesView(MutableMapping[str, dict[str, Any]]):
    """Mutable mapping of template id → serialised template dict backed by :class:`TemplateStoreView`."""

    __slots__ = ("_view",)

    def __init__(self, view: TemplateStoreView) -> None:
        self._view = view

    def __getitem__(self, tid: str) -> dict[str, Any]:
        raw = self._view.get_template_raw(str(tid))
        if raw is None:
            raise KeyError(tid)
        return raw

    def __setitem__(self, tid: str, value: dict[str, Any]) -> None:
        self._view.set_template_raw_dict(str(tid), dict(value))

    def __delitem__(self, tid: str) -> None:
        self._view.remove_template_id(str(tid))

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._view.partition_map))

    def __len__(self) -> int:
        return len(self._view.partition_map)

    def __contains__(self, tid: object) -> bool:
        return isinstance(tid, str) and tid in self._view.partition_map


class TemplateStoreView:
    """
    Header-backed template store with lazy partition loads and bounded in-memory partitions.

    Match indexes live in memory (eager from header). Full template payloads live in
    ``partition_<NN>.json.gz`` shards and are loaded on demand.
    """

    __slots__ = (
        "_dirty_partitions",
        "_indexes",
        "_lru_max",
        "_partition_cache",
        "_store_dir",
        "_templates_proxy",
        "effective_structural_hash",
        "next_id",
        "partition_map",
        "question_feedback",
    )

    def __init__(
        self,
        store_dir: str,
        effective_structural_hash: str,
        next_id: int,
        question_feedback: dict[str, list[dict[str, Any]]],
        partition_map: dict[str, int],
        *,
        indexes: dict[str, Any] | None = None,
    ) -> None:
        self._store_dir = store_dir
        self.effective_structural_hash = effective_structural_hash
        self.next_id = int(next_id)
        self.question_feedback = question_feedback
        self.partition_map = partition_map
        self._indexes: dict[str, Any] = (
            dict(indexes)
            if indexes is not None
            else {
                SHAPE_QUESTION_INDEX_KEY: {},
                TEMPLATE_INTENT_KEY_INDEX_KEY: {},
                TEMPLATE_UNION_FAMILY_INDEX_KEY: {},
                TEMPLATE_QUESTION_TOKEN_INDEX_KEY: {},
            }
        )
        self._partition_cache: OrderedDict[int, dict[str, dict[str, Any]]] = OrderedDict()
        self._dirty_partitions: set[int] = set()
        self._lru_max = int(TEMPLATE_STORE_PARTITION_LRU_MAX)
        self._templates_proxy = _TemplateBodiesView(self)

    @classmethod
    def empty(cls, store_dir: str, effective_structural_hash: str) -> TemplateStoreView:
        return cls(
            store_dir,
            effective_structural_hash,
            1,
            {},
            {},
        )

    @classmethod
    def from_header_payload(
        cls,
        store_dir: str,
        header: dict[str, Any],
    ) -> TemplateStoreView:
        """Build a view from a decoded header document (no template bodies)."""

        h = dict(header)
        h.pop("templates", None)
        _normalize_loaded_template_store_document(h)
        pm_raw = h.get("partition_map") or {}
        partition_map: dict[str, int] = {}
        if isinstance(pm_raw, dict):
            for tid, pv in pm_raw.items():
                try:
                    partition_map[str(tid)] = int(pv)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        indexes = {
            SHAPE_QUESTION_INDEX_KEY: h.get(SHAPE_QUESTION_INDEX_KEY) or {},
            TEMPLATE_INTENT_KEY_INDEX_KEY: h.get(TEMPLATE_INTENT_KEY_INDEX_KEY) or {},
            TEMPLATE_UNION_FAMILY_INDEX_KEY: h.get(TEMPLATE_UNION_FAMILY_INDEX_KEY) or {},
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY: h.get(TEMPLATE_QUESTION_TOKEN_INDEX_KEY) or {},
        }
        for key in indexes:
            if not isinstance(indexes[key], dict):
                indexes[key] = {}
        next_id = int(h.get("next_id", 1) or 1)
        qf = h.get("question_feedback")
        if not isinstance(qf, dict):
            qf = {}
        eff = str(h.get("effective_structural_hash", h.get("schema_hash", "")) or "")
        return cls(
            store_dir,
            eff,
            next_id,
            qf,  # type: ignore[arg-type]
            partition_map,
            indexes=indexes,
        )

    def dirty_partitions(self) -> set[int]:
        return set(self._dirty_partitions)

    def _partition_file_path(self, part: int) -> str:
        return os.path.join(
            self._store_dir,
            f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz",
        )

    def _flush_partition_to_disk(self, part: int, payload: dict[str, dict[str, Any]]) -> None:
        serial = _convert_to_json_serializable(dict(payload))
        write_gzip_json_atomic(self._partition_file_path(part), serial, sort_keys=True)

    def _evict_partition_if_needed(self) -> None:
        while len(self._partition_cache) >= self._lru_max and self._partition_cache:
            victim, victim_payload = self._partition_cache.popitem(last=False)
            if victim in self._dirty_partitions:
                self._flush_partition_to_disk(victim, victim_payload)
                self._dirty_partitions.discard(victim)

    def _load_partition_payload(self, part: int) -> dict[str, dict[str, Any]]:
        if part in self._partition_cache:
            self._partition_cache.move_to_end(part)
            return self._partition_cache[part]
        self._evict_partition_if_needed()
        path = self._partition_file_path(part)
        if os.path.isfile(path):
            try:
                raw = read_gzip_json(path)
            except (
                OSError,
                EOFError,
                gzip.BadGzipFile,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                raw = {}
        else:
            raw = {}
        payload: dict[str, dict[str, Any]] = raw if isinstance(raw, dict) else {}
        self._partition_cache[part] = payload
        return payload

    def get_template_raw(self, template_id: str) -> dict[str, Any] | None:
        tid = str(template_id)
        part = self.partition_map.get(tid)
        if part is None:
            return None
        body = self._load_partition_payload(int(part))
        raw = body.get(tid)
        return dict(raw) if isinstance(raw, dict) else None

    def get_template(self, template_id: str) -> Template | None:
        raw = self.get_template_raw(template_id)
        if raw is None:
            return None
        t = Template.from_dict({**raw, "id": str(template_id)})
        t.sql_fp = compute_sql_fp(t.sql_param or "", sqlglot_dialect=active_sqlglot_dialect())
        return t

    def iter_templates_by_ids(self, ids: Iterator[str]) -> Iterator[Template]:
        for tid in ids:
            t = self.get_template(str(tid))
            if t is not None:
                yield t

    def set_template_raw_dict(self, template_id: str, raw: dict[str, Any]) -> None:
        tid = str(template_id)
        part = template_partition_number(tid)
        self.partition_map[tid] = part
        payload = self._load_partition_payload(part)
        payload[tid] = raw
        self._dirty_partitions.add(part)

    def remove_template_id(self, template_id: str) -> None:
        tid = str(template_id)
        part = self.partition_map.pop(tid, None)
        if part is None:
            return
        p_int = int(part)
        if p_int in self._partition_cache:
            self._partition_cache[p_int].pop(tid, None)
        else:
            disk = self._load_partition_payload(p_int)
            disk.pop(tid, None)
        self._dirty_partitions.add(p_int)

    def _bulk_replace_templates_from_mapping(self, mapping: Mapping[str, Any]) -> None:
        self.partition_map.clear()
        self._partition_cache.clear()
        self._dirty_partitions.clear()
        for part in range(TEMPLATE_STORE_PARTITION_COUNT):
            path = self._partition_file_path(part)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        for tid, raw in mapping.items():
            if not isinstance(raw, dict):
                continue
            self.set_template_raw_dict(str(tid), dict(raw))

    def _header_document(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "effective_structural_hash": self.effective_structural_hash,
            "next_id": int(self.next_id),
            "question_feedback": self.question_feedback,
            "partition_map": dict(sorted(self.partition_map.items())),
            SHAPE_QUESTION_INDEX_KEY: self._indexes[SHAPE_QUESTION_INDEX_KEY],
            TEMPLATE_INTENT_KEY_INDEX_KEY: self._indexes[TEMPLATE_INTENT_KEY_INDEX_KEY],
            TEMPLATE_UNION_FAMILY_INDEX_KEY: self._indexes[TEMPLATE_UNION_FAMILY_INDEX_KEY],
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY: self._indexes[TEMPLATE_QUESTION_TOKEN_INDEX_KEY],
        }

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key == "templates":
            return True
        if key in (
            "effective_structural_hash",
            "next_id",
            "question_feedback",
            SHAPE_QUESTION_INDEX_KEY,
            TEMPLATE_INTENT_KEY_INDEX_KEY,
            TEMPLATE_UNION_FAMILY_INDEX_KEY,
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
        ):
            return True
        return False

    def __getitem__(self, key: str) -> Any:
        if key == "templates":
            return self._templates_proxy
        if key == "effective_structural_hash":
            return self.effective_structural_hash
        if key == "next_id":
            return self.next_id
        if key == "question_feedback":
            return self.question_feedback
        if key in self._indexes:
            return self._indexes[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "templates":
            if value is self._templates_proxy:
                return
            if isinstance(value, _TemplateBodiesView) and value._view is self:
                return
            if not isinstance(value, Mapping):
                raise TypeError("templates must be a mapping of serialised template dicts")
            self._bulk_replace_templates_from_mapping(value)
            return
        if key == "effective_structural_hash":
            self.effective_structural_hash = str(value)
            return
        if key == "next_id":
            self.next_id = int(value)
            return
        if key == "question_feedback":
            if not isinstance(value, dict):
                raise TypeError("question_feedback must be a dict")
            self.question_feedback = value  # type: ignore[assignment]
            return
        if key in self._indexes:
            if not isinstance(value, dict):
                raise TypeError
            self._indexes[key] = value
            return
        raise KeyError(key)

    def setdefault(self, key: str, default: Any) -> Any:
        if key == "question_feedback":
            return self.question_feedback
        if key in self._indexes:
            cur = self._indexes[key]
            if cur is None or (isinstance(cur, dict) and not cur and default is not None):
                self._indexes[key] = default if isinstance(default, dict) else {}
            return self._indexes[key]
        raise KeyError(key)

    def __deepcopy__(self, memo: dict[int, Any]) -> TemplateStoreView:
        other = TemplateStoreView.__new__(TemplateStoreView)
        other._store_dir = self._store_dir
        other.effective_structural_hash = self.effective_structural_hash
        other.next_id = self.next_id
        other.question_feedback = copy.deepcopy(self.question_feedback, memo)
        other.partition_map = dict(self.partition_map)
        other._indexes = {k: copy.deepcopy(v, memo) for k, v in self._indexes.items()}
        other._partition_cache = OrderedDict((k, copy.deepcopy(v, memo)) for k, v in self._partition_cache.items())
        other._dirty_partitions = set()
        other._lru_max = self._lru_max
        other._templates_proxy = _TemplateBodiesView(other)
        memo[id(self)] = other
        return other


def refresh_template_store_indexes(
    store: dict[str, Any] | TemplateStoreView,
    *,
    template_objs: list[Template] | None = None,
) -> None:
    """
    Recompute shape and inverted template indexes on *store* in place.

    When *template_objs* is provided (already-materialised :class:`Template` rows), avoids a round-trip
    through ``store['templates']`` dict serialisation.
    """

    if isinstance(store, TemplateStoreView):
        if template_objs is None:
            tpl_objs: list[Template] = []
            for tid in sorted(store.partition_map.keys()):
                raw = store.get_template_raw(tid)
                if raw is not None:
                    tpl_objs.append(Template.from_dict({**raw, "id": str(tid)}))
        else:
            tpl_objs = template_objs
        store._indexes[SHAPE_QUESTION_INDEX_KEY] = build_shape_question_index(tpl_objs)
        store._indexes[TEMPLATE_INTENT_KEY_INDEX_KEY] = _build_intent_key_index_for_templates(tpl_objs)
        store._indexes[TEMPLATE_UNION_FAMILY_INDEX_KEY] = _build_union_family_index_for_templates(tpl_objs)
        store._indexes[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = _build_question_token_index_for_templates(tpl_objs)
        return

    if template_objs is None:
        tpl_objs = []
        for tid, raw in (store.get("templates") or {}).items():
            if isinstance(raw, dict):
                tpl_objs.append(Template.from_dict({**raw, "id": str(tid)}))
    else:
        tpl_objs = template_objs
    store[SHAPE_QUESTION_INDEX_KEY] = build_shape_question_index(tpl_objs)
    store[TEMPLATE_INTENT_KEY_INDEX_KEY] = _build_intent_key_index_for_templates(tpl_objs)
    store[TEMPLATE_UNION_FAMILY_INDEX_KEY] = _build_union_family_index_for_templates(tpl_objs)
    store[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = _build_question_token_index_for_templates(tpl_objs)


@dataclass(frozen=True, slots=True)
class TemplateRefs:
    """Normalized schema references extracted from a template-like object."""

    tables: frozenset[str]
    columns: frozenset[tuple[str, str]]
    fk_edges: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Summary of a template-store reconciliation pass."""

    kept_template_ids: tuple[str, ...]
    dropped_template_ids: tuple[str, ...]
    kept_rejected_ids: tuple[str, ...]
    dropped_rejected_ids: tuple[str, ...]
    dropped_negative_memory_bucket_count: int
    dropped_failure_log_rows: int
    dropped_warmup_units: int
    reason_histogram: Mapping[str, int]


def _join_segment_from_edge_dict(edge: dict[str, Any]) -> str:
    """Format one join-graph edge dict as a canonical signature segment string."""

    return f"{edge['src_table']}.{','.join(edge['src_cols'])}->{edge['dst_table']}.{','.join(edge['dst_cols'])}"


def _all_join_segments_live(schema: SchemaGraph) -> frozenset[str]:
    """Collect every join path segment string still present in ``join_paths_multi``."""

    segs: set[str] = set()
    for row in schema.join_paths_multi.values():
        for paths in row.values():
            for path in paths:
                for edge in path:
                    segs.add(_join_segment_from_edge_dict(edge))
    return frozenset(segs)


def _runtime_intent_schema_refs(rt: RuntimeIntent) -> TemplateRefs:
    """Build ``TemplateRefs`` from a ``RuntimeIntent`` snapshot."""

    tables = frozenset(rt.tables or ())
    columns: set[tuple[str, str]] = set()
    for bare, tbl in (rt.column_map or {}).items():
        columns.add((tbl, bare))
    fk = frozenset(str(x) for x in (rt.chosen_join_path_signature or []) if str(x).strip())
    return TemplateRefs(tables=tables, columns=frozenset(columns), fk_edges=fk)


def template_schema_refs(template: Template) -> TemplateRefs:
    """Collect tables, column pairs, and join edge tokens referenced by *template*."""

    tables: set[str] = set(template.tables_used or ())
    if template.intent_signature.tables:
        tables.update(template.intent_signature.tables)
    columns: set[tuple[str, str]] = set()
    for bare, tbl in template.intent_signature.column_map.items():
        columns.add((tbl, bare))
    fk_edges: set[str] = set(template.chosen_join_path_signature or ())
    return TemplateRefs(
        tables=frozenset(tables),
        columns=frozenset(columns),
        fk_edges=frozenset(fk_edges),
    )


def warmup_work_unit_schema_refs(work_unit: Mapping[str, Any]) -> TemplateRefs:
    """Derive ``TemplateRefs`` from a persisted seed-warmup work unit payload."""

    er = work_unit.get("execute_result")
    if isinstance(er, dict) and isinstance(er.get("runtime"), dict):
        return _runtime_intent_schema_refs(RuntimeIntent.from_dict(er["runtime"]))
    raw_si = work_unit.get("serialized_intent")
    if isinstance(raw_si, dict):
        return _runtime_intent_schema_refs(SeedWarmupIntent.from_dict(raw_si).to_runtime_intent())
    return TemplateRefs(tables=frozenset(), columns=frozenset(), fk_edges=frozenset())


def template_is_live(refs: TemplateRefs, schema: SchemaGraph) -> tuple[bool, tuple[str, ...]]:
    """Return whether every referenced table, column, and FK segment still exists in *schema*."""

    reasons: list[str] = []
    for t in refs.tables:
        if t not in schema.tables:
            reasons.append(f"missing_table:{t}")
    for table, col in refs.columns:
        tm = schema.tables.get(table)
        if tm is None or col not in tm.columns:
            reasons.append(f"missing_column:{table}.{col}")
    if any(len(tm.foreign_keys) > 0 for tm in schema.tables.values()) and refs.fk_edges:
        live_segs = _all_join_segments_live(schema)
        for seg in refs.fk_edges:
            s = str(seg).strip()
            if not s:
                continue
            if s not in live_segs:
                reasons.append(f"missing_join_segment:{s}")
    return (len(reasons) == 0, tuple(reasons))


def join_fingerprint_from_concrete_intent(concrete: ConcreteIntent) -> str:
    """Stable hash of main and per-CTE chosen join path signatures in declaration order."""

    return join_path_key_concrete(concrete)


def join_fingerprint_from_runtime_intent(intent: RuntimeIntent) -> str:
    """Stable hash of main and per-CTE join signatures on a runtime intent after join resolution."""

    return join_path_key_runtime(intent)


def _coerce_rejection_bucket(raw: object) -> RejectionBucket:
    """Map LLM or wire text to :class:`RejectionBucket` (default ``OTHER``)."""

    if isinstance(raw, RejectionBucket):
        return raw
    s = str(raw or "").strip().upper()
    for m in RejectionBucket:
        if m.name == s or m.value == s:
            return m
    return RejectionBucket.OTHER


def _feedback_iso_now() -> str:
    """Return an ISO-8601 UTC timestamp string for feedback rows."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_intent_structural_signature(
    intent: RuntimeIntent | None,
) -> tuple[str, str]:
    """Return ``(sha256_first_16_hex, stable_json_of_concrete_intent)`` for deduplication and LLM context."""

    if intent is None:
        return "", ""
    conc = runtime_intent_to_concrete(intent, "")
    payload = stable_json(conc.to_dict())
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return digest, payload


def combine_feedback_summaries(existing: str, incoming: str, *, intent_payload: str) -> str:
    """Merge two feedback summaries for the same intent-structure key using one short LLM response."""

    a = (existing or "").strip()
    b = (incoming or "").strip()
    if not a:
        return b
    if not b:
        return a
    if not llm_credentials_configured():
        return f"{a}\n{b}".strip()
    system = (
        "You merge two brief text-to-SQL feedback summaries into one 3-6 line ASCII block. "
        "Each line states a structural issue without SQL. "
        'Respond as JSON only: {"summary":"..."}.'
    )
    user = json.dumps(
        {"existing": a, "incoming": b, "intent_structure_json": intent_payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    raw = llm_chat(system, user, task="feedback", max_retries=1, timeout=20.0)
    try:
        data = json.loads(raw)
    except Exception as exc:
        debug(f"[templates.combine_feedback_summaries] json coerce: {exc}")
        return f"{a}\n{b}".strip()
    merged = str(data.get("summary", "")).strip()
    return merged if merged else f"{a}\n{b}".strip()


def summarize_failure_for_memory(
    *,
    question: str,
    intent: RuntimeIntent | None,
    kind: FeedbackKind,
    schema_hash: str,
    validator_errors: list[str] | None = None,
    user_reason: str | None = None,
    sql: str | None = None,
    is_post_restart: bool = False,
    source: Literal["engine"] = "engine",
) -> QuestionFeedbackEntry:
    """
    Build one ``QuestionFeedbackEntry`` using a single LLM summary call.

    Persisted rows use ``kind`` ``INTENT_REJECTED`` when the user rejects a validated intent or result.

    Persisted rows use ``kind`` ``VALIDATION_FAILURE`` only when :func:`aetherdialect._intent_process._attempt_fresh_restart` records semantic exhaustion after ``semantic_oscillation`` or ``semantic_max_rounds``.

    Malformed LLM output is coerced to a summary string and ``OTHER`` bucket.

    ``llm_chat`` transport failures are not caught and propagate to the caller.

    Args:

        source: Feedback provenance; only ``engine`` is supported.
    """

    ish, ipay = compute_intent_structural_signature(intent)
    structure_for_prompt = ipay if ipay else "{}"
    payload: dict[str, Any] = {
        "question": question,
        "kind": kind.value,
        "intent_structure_json": structure_for_prompt,
        "user_reason": user_reason or "",
    }
    if sql:
        payload["sql"] = sql
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
    ts = _feedback_iso_now()
    if not llm_credentials_configured():
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
        )
    raw = llm_chat(system, user, task="feedback", max_retries=1, timeout=20.0)
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
        )
    compressed = str(data.get("summary", "")).strip() or (user_reason or "(unspecified failure)").strip()
    bucket = _coerce_rejection_bucket(data.get("bucket"))
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
    )


def record_question_feedback(
    store: dict[str, Any] | TemplateStoreView,
    q_norm: str,
    entry: QuestionFeedbackEntry,
) -> None:
    """
    Merge or append one ``QuestionFeedbackEntry`` under *q_norm* using kind-specific deduplication.

    ``VALIDATION_FAILURE`` rows deduplicate per ``intent_structural_hash`` only.

    ``INTENT_REJECTED`` rows deduplicate per ``intent_structural_hash`` and rejection bucket, merging summaries when a new bucket appears for the same hash.
    """

    if not entry.intent_structural_hash:
        return
    if isinstance(store, TemplateStoreView):
        qf = store.question_feedback
    else:
        qf = store.setdefault("question_feedback", {})
    cur = qf.setdefault(q_norm, [])
    if not isinstance(cur, list):
        cur = []
        qf[q_norm] = cur
    incoming_bucket = entry.buckets[0] if entry.buckets else RejectionBucket.OTHER
    for i, row in enumerate(cur):
        if not isinstance(row, dict):
            continue
        existing = QuestionFeedbackEntry.from_dict(row)
        if existing.intent_structural_hash != entry.intent_structural_hash:
            continue
        if entry.kind is FeedbackKind.VALIDATION_FAILURE:
            if existing.kind is FeedbackKind.VALIDATION_FAILURE:
                return
            continue
        if existing.kind is FeedbackKind.VALIDATION_FAILURE:
            break
        if existing.kind is not FeedbackKind.INTENT_REJECTED:
            break
        if incoming_bucket in existing.buckets:
            return
        merged_summary = combine_feedback_summaries(
            existing.summary,
            entry.summary,
            intent_payload=entry.intent_payload or existing.intent_payload,
        )
        cur[i] = replace(
            existing,
            summary=merged_summary,
            buckets=(*existing.buckets, incoming_bucket),
            kind=FeedbackKind.INTENT_REJECTED,
            intent_payload=entry.intent_payload or existing.intent_payload,
            updated_at=_feedback_iso_now(),
            is_post_restart=existing.is_post_restart or entry.is_post_restart,
            source=existing.source,
        ).to_dict()
        maxpq = PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION
        if len(cur) > maxpq:
            cur[:] = cur[-maxpq:]
        return
    cur.append(entry.to_dict())
    maxpq = PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION
    if len(cur) > maxpq:
        cur[:] = cur[-maxpq:]


def lookup_join_feedback_for_question(store: dict[str, Any], q_norm: str) -> list[str]:
    """
    Return textual rejection summaries for prior wrong-join feedback on this question.

    Args:

        store: Loaded template store dictionary containing question_feedback.

        q_norm: Deterministic-normalised question key.

    Returns:

        Sorted-by-recency list of summary strings, possibly empty.
    """

    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return []
    rows = qf.get(q_norm)
    if not isinstance(rows, list):
        return []
    rows_with_ts: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ent = QuestionFeedbackEntry.from_dict(row)
        if ent.kind is not FeedbackKind.INTENT_REJECTED:
            continue
        if RejectionBucket.WRONG_TABLES_OR_JOINS not in ent.buckets:
            continue
        summary = (ent.summary or "").strip()
        if not summary:
            continue
        rows_with_ts.append((ent.updated_at or ent.created_at, summary))
    rows_with_ts.sort(key=lambda t: t[0], reverse=True)
    return [s for _ts, s in rows_with_ts]


def collect_question_feedback_for_prompt(
    store: dict[str, Any],
    q_norm: str,
    schema_hash: str,
) -> list[dict[str, str]]:
    """Return feedback rows for prompts in insertion order, scoped to *schema_hash*."""

    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return []
    out: list[dict[str, str]] = []
    for q_key, rows in qf.items():
        if not is_exact_question_text_match(q_norm, str(q_key)):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            ent = QuestionFeedbackEntry.from_dict(row)
            if ent.effective_structural_hash != schema_hash:
                continue
            out.append(ent.to_prompt_row())
    return out


def compute_question_feedback_penalty(
    store: dict[str, Any],
    q_norm: str,
    schema_hash: str,
) -> float:
    """Map matching feedback count to a confidence penalty capped at ``PENALTY_CAP``."""

    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return 0.0
    weighted = 0.0
    for q_key, rows in qf.items():
        if not is_exact_question_text_match(q_norm, str(q_key)):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            ent = QuestionFeedbackEntry.from_dict(row)
            if ent.effective_structural_hash != schema_hash:
                continue
            weighted += 1.0
    return float(min(PolicyConfig.PENALTY_CAP, weighted * PolicyConfig.PEN_BY_THREE_SOURCE_UNIT))


def _build_compact_shape(intent: RuntimeIntent | ConcreteIntent) -> str:
    """Return a deterministic one-line structural fingerprint for *intent* (max 240 chars)."""

    tables = sorted(str(t).strip().lower() for t in (intent.tables or []) if str(t).strip())
    sel = len(intent.select_cols or [])
    flt = len(intent.filters_param or [])
    hav = len(intent.having_param or [])
    gb = len(intent.group_by_cols or [])
    ob = len(intent.order_by_cols or [])
    cte_n = len(intent.cte_steps or [])
    grain = str(getattr(intent, "grain", "") or "row")
    agg = False
    for sc in intent.select_cols or []:
        if getattr(sc, "is_aggregated", False):
            agg = True
            break
    if not agg:
        for hp in intent.having_param or []:
            if getattr(hp, "left_expr", None) and "agg" in str(type(hp)).lower():
                agg = True
                break
    join_ids: list[str] = []
    main_c = str(getattr(intent, "chosen_join_candidate_id", "") or "").strip()
    if main_c:
        join_ids.append(main_c)
    for step in intent.cte_steps or []:
        c = str(getattr(step, "chosen_join_candidate_id", "") or "").strip()
        if c:
            join_ids.append(c)
    joins_s = ",".join(join_ids) if join_ids else "none"
    line = (
        f"tables={tables}; grain={grain}; select={sel}; filters={flt}; having={hav}; "
        f"groupby={gb}; orderby={ob}; agg={str(agg).lower()}; joins={joins_s}; ctes={cte_n}"
    )
    return line[:240]


def has_any_rejection_history_for_question(store: dict[str, Any], q_norm: str) -> bool:
    """Return True when ``question_feedback`` has any row for a fuzzy-matching question key."""

    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return False
    for q_key in qf:
        if is_exact_question_text_match(q_norm, str(q_key)) and isinstance(qf.get(q_key), list) and qf[q_key]:
            return True
    return False


def delete_rejected_templates_matching_question(
    store: dict[str, Any],
    q_norm: str,
) -> None:
    """Remove question-feedback entries whose key fuzzy-matches *q_norm*."""

    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return
    for q_key in list(qf):
        if is_exact_question_text_match(q_norm, str(q_key)):
            del qf[q_key]
            debug(f"[templates.delete_rejected_templates_matching_question] removed feedback for key={q_key!r}")


def promote_rejected_to_template(
    store: dict[str, Any],
    templates: dict[str, Template],
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    schema_hash: str,
    *,
    form_storage: QuestionFormStorage | None = None,
) -> Template:
    """
    Create an accepted template after the user accepts a result following prior question feedback.

    The historical name is retained; there is no separate rejected-template record in the store.
    """

    tid = f"T{int(store['next_id']):04d}"
    store["next_id"] = int(store["next_id"]) + 1

    sql_canon = canonicalize_sql(sql)
    if intent.sql_param:
        sql_param = intent.sql_param
    else:
        sql_norm = normalize_sql(sql_canon)
        sql_param, _ = parameter_abstract(sql_norm, sqlglot_dialect=active_sqlglot_dialect())
    sql_fp_val = compute_sql_fp(sql_param, sqlglot_dialect=active_sqlglot_dialect())

    colmap_sig_val = colmap_signature(intent.column_map)

    intent_signature = runtime_intent_to_concrete(intent, "")
    ikey = intent_key(intent)

    all_pv = flatten_param_values(intent)
    nl0 = intent.natural_language or ""
    primary_q = form_storage.corrected if form_storage is not None else q_norm
    vh_new = ValueHistory(
        param_values=[all_pv],
        questions=[primary_q],
        natural_language=[nl0],
    )
    if (
        form_storage is not None
        and not form_storage.accept_via_normalized_lookup_only
        and form_storage.normalized_optional
        and form_storage.normalized_optional != primary_q
        and not form_storage.normalized_negative_memory_dropped
    ):
        vh_new.append_question_variant(
            form_storage.normalized_optional,
            accept_count=0,
            param_values=all_pv,
            natural_language=nl0,
        )

    sig_aliases: dict[str, str] = {}
    for sc in intent.select_cols or []:
        alias = generate_col_alias(sc)
        if alias:
            sig_aliases[sc.signature_key] = alias

    tmpl = Template(
        id=tid,
        effective_structural_hash=schema_hash,
        intent_signature=intent_signature,
        intent_key=ikey,
        tables_used=sorted(intent.tables),
        sql_param=sql_param,
        sql_fp=sql_fp_val,
        shape=sql_shape(sql_canon, intent, sqlglot_dialect=active_sqlglot_dialect()),
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
    templates_to_store(store, templates)
    return tmpl


def reconcile_template_store(store: dict[str, Any] | TemplateStoreView, schema: SchemaGraph) -> ReconcileReport:
    """Drop templates and stale ``question_feedback`` rows whose schema hash no longer matches."""

    kept_templates: list[str] = []
    dropped_templates: list[str] = []
    reason_hist: Counter[str] = Counter()
    templates = store.get("templates", {}) or {}
    for tid, raw in list(templates.items()):
        tmpl = Template.from_dict({**raw, "id": tid})
        ok, reasons = template_is_live(template_schema_refs(tmpl), schema)
        if ok:
            kept_templates.append(tid)
        else:
            dropped_templates.append(tid)
            for r in reasons:
                reason_hist[r] += 1
            del templates[tid]
    store["templates"] = templates

    eff = schema.effective_structural_hash
    dropped_qf = 0
    qf = store.get("question_feedback")
    if isinstance(qf, dict):
        for qk, rows in list(qf.items()):
            if not isinstance(rows, list):
                continue
            new_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                prev = str(row.get("effective_structural_hash", "") or "")
                if prev and prev != eff:
                    dropped_qf += 1
                    continue
                new_rows.append(row)
            if new_rows:
                qf[qk] = new_rows
            else:
                del qf[qk]
        store["question_feedback"] = qf

    refresh_template_store_indexes(store)

    return ReconcileReport(
        kept_template_ids=tuple(sorted(kept_templates)),
        dropped_template_ids=tuple(sorted(dropped_templates)),
        kept_rejected_ids=(),
        dropped_rejected_ids=(),
        dropped_negative_memory_bucket_count=dropped_qf,
        dropped_failure_log_rows=0,
        dropped_warmup_units=0,
        reason_histogram=dict(reason_hist),
    )


def _map_join_side(
    side: str,
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> str:
    side = side.strip()
    if "." not in side:
        return tmap.get(side, side)
    tbl, rest = side.split(".", 1)
    cols = [c.strip() for c in rest.split(",") if c.strip()]
    nt = tmap.get(tbl, tbl)
    cm = colmaps.get(tbl, {})
    mapped_cols = [cm.get(c, c) for c in cols]
    return f"{nt}.{','.join(mapped_cols)}"


def _rewrite_join_path_segments(
    sigs: list[str],
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> list[str]:
    out: list[str] = []
    for seg in sigs:
        s = str(seg).strip()
        if "->" not in s:
            out.append(seg)
            continue
        a, b = s.split("->", 1)
        out.append(f"{_map_join_side(a, tmap, colmaps)}->{_map_join_side(b, tmap, colmaps)}")
    return out


def _remap_column_map(
    column_map: dict[str, str],
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> dict[str, str]:
    """
    Apply table and column renames to a ``{bare_col: table}`` dict.

    ``colmaps`` is keyed by *old* table name, so we look up renames using the pre-rename table even when the table itself is being renamed.
    """

    out: dict[str, str] = {}
    for bare, tbl in column_map.items():
        new_tbl = tmap.get(tbl, tbl)
        new_bare = colmaps.get(tbl, {}).get(bare, bare)
        out[new_bare] = new_tbl
    return out


def _remap_concrete_cte(
    cte: ConcreteCteStep,
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> ConcreteCteStep:
    return replace(
        cte,
        tables=[tmap.get(x, x) for x in cte.tables],
        column_map=_remap_column_map(cte.column_map, tmap, colmaps),
        chosen_join_path_signature=_rewrite_join_path_segments(cte.chosen_join_path_signature, tmap, colmaps),
    )


def _remap_concrete_intent(
    ci: ConcreteIntent,
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> ConcreteIntent:
    return replace(
        ci,
        tables=[tmap.get(x, x) for x in ci.tables],
        column_map=_remap_column_map(ci.column_map, tmap, colmaps),
        chosen_join_path_signature=_rewrite_join_path_segments(ci.chosen_join_path_signature, tmap, colmaps),
        cte_steps=[_remap_concrete_cte(cte, tmap, colmaps) for cte in ci.cte_steps],
    )


def _remap_sql_strings(
    sql: str,
    tmap: dict[str, str],
    colmaps: dict[str, dict[str, str]],
) -> str:
    out = sql or ""
    for old_t, new_t in sorted(tmap.items(), key=lambda x: -len(x[0])):
        if old_t == new_t:
            continue
        out = re.sub(rf"(?<![\w]){re.escape(old_t)}(?![\w])", new_t, out)
    for ot, inner in colmaps.items():
        nt = tmap.get(ot, ot)
        for oc, nc in inner.items():
            if oc == nc:
                continue
            out = re.sub(rf"(?<![\w]){re.escape(ot)}\.{re.escape(oc)}(?![\w])", f"{nt}.{nc}", out)
            out = re.sub(rf"(?<![\w]){re.escape(nt)}\.{re.escape(oc)}(?![\w])", f"{nt}.{nc}", out)
    return out


def apply_schema_rename_migration_to_store(
    artifacts_dir: str,
    schema: SchemaGraph,
    renamed_tables: tuple[tuple[str, str], ...],
    renamed_columns: tuple[tuple[str, str, str], ...],
) -> tuple[int, int]:
    """Rewrite persisted template intents for schema rename migration."""

    tmap = dict(renamed_tables)
    colmaps: dict[str, dict[str, str]] = defaultdict(dict)
    for ot, oc, nc in renamed_columns:
        colmaps[ot][oc] = nc
    colmaps = {k: dict(v) for k, v in colmaps.items()}
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    if not os.path.isfile(header_path):
        return (0, 0)
    with artifact_lock(artifacts_dir):
        view = _load_partitioned_view_unlocked(store_dir)
        if view is None or not view.partition_map:
            return (0, 0)
        remapped = 0
        destroyed = 0
        for tid in list(view.partition_map.keys()):
            raw = view.get_template_raw(tid)
            if raw is None:
                continue
            tmpl = Template.from_dict({**raw, "id": tid})
            new_sig = _remap_concrete_intent(tmpl.intent_signature, tmap, colmaps)
            new_sql = _remap_sql_strings(tmpl.sql_param, tmap, colmaps)
            new_tables_used = [tmap.get(x, x) for x in (tmpl.tables_used or [])]
            rebuilt = replace(
                tmpl,
                intent_signature=new_sig,
                tables_used=new_tables_used,
                effective_structural_hash=schema.effective_structural_hash,
                sql_param=new_sql,
            )
            rebuilt.sql_fp = compute_sql_fp(rebuilt.sql_param or "", sqlglot_dialect=active_sqlglot_dialect())
            ok, _ = template_is_live(template_schema_refs(rebuilt), schema)
            if not ok:
                view.remove_template_id(tid)
                destroyed += 1
                continue
            view.set_template_raw_dict(tid, rebuilt.to_dict())
            remapped += 1
        view["effective_structural_hash"] = schema.effective_structural_hash
        refresh_template_store_indexes(view)
        save_template_store(view)
        adir = os.path.dirname(os.path.abspath(store_dir))
        prev = read_artifact_manifest(adir)
        write_artifact_manifest(
            adir,
            structural_hash=prev.structural_hash if prev else "",
            profiling_hash=prev.profiling_hash if prev else "",
            scope_hash=prev.scope_hash if prev else "",
            effective_structural_hash=(prev.effective_structural_hash if prev else view.effective_structural_hash),
            notes_hash=prev.notes_hash if prev else "",
            semantic_edges_hash=prev.semantic_edges_hash if prev else "",
            last_migration_tier=prev.last_migration_tier if prev else "",
            last_migration_at=prev.last_migration_at if prev else "",
            last_action="remap_templates",
        )
    return (remapped, destroyed)


def _disk_template_row_count(artifacts_dir: str) -> int:
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    if not os.path.isfile(header_path):
        return 0
    try:
        view = _load_partitioned_view_unlocked(store_dir)
    except (OSError, ValueError, TypeError):
        return 0
    if view is None:
        return 0
    n_tpl = len(view.partition_map)
    qf = view.question_feedback
    n_qf = 0
    for _qk, rows in qf.items():
        if isinstance(rows, list):
            n_qf += len(rows)
    return n_tpl + n_qf


def _surgical_invalidation_targets(
    schema_diff: SchemaDiff,
) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    """
    Build the tombstone sets used to detect templates invalidated by a SchemaDiff.

    Returns ``(tombstoned_tables, tombstoned_columns)``. A template is surgically deleted when its references intersect either set.

    ``tombstoned_tables`` includes both dropped tables and the *old* names of rename pairs (any template still referring to the old name is stale until a REMAP pass rewrites it; surgery never touches templates already remapped because the REMAP pass runs first and rewrites references to the new name).

    ``tombstoned_columns`` contains every ``(table, column)`` pair removed by the diff as well as columns whose ``value_type`` materially changed.
    """

    tombstoned_tables: set[str] = set(schema_diff.dropped_tables)
    tombstoned_cols: set[tuple[str, str]] = set()
    for table, td in schema_diff.per_table.items():
        for col in td.dropped_columns:
            tombstoned_cols.add((table, col))
        for col, _old_vt, _new_vt in td.value_type_changed_columns:
            tombstoned_cols.add((table, col))
    return frozenset(tombstoned_tables), frozenset(tombstoned_cols)


def surgical_invalidate_templates_by_diff(
    artifacts_dir: str,
    schema: SchemaGraph,
    schema_diff: SchemaDiff,
) -> int:
    """
    Delete persisted accepted templates whose references intersect a SchemaDiff's tombstone sets.

    A no-op (returns 0) when the store is missing or the diff yields no tombstones.
    """

    tombstoned_tables, tombstoned_cols = _surgical_invalidation_targets(schema_diff)
    if not tombstoned_tables and not tombstoned_cols:
        return 0
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    if not os.path.isfile(header_path):
        return 0
    with artifact_lock(artifacts_dir):
        view = _load_partitioned_view_unlocked(store_dir)
        if view is None or not view.partition_map:
            return 0
        deleted = 0
        for tid in list(view.partition_map.keys()):
            raw = view.get_template_raw(tid)
            if raw is None:
                continue
            tmpl = Template.from_dict({**raw, "id": tid})
            refs = template_schema_refs(tmpl)
            if refs.tables & tombstoned_tables or refs.columns & tombstoned_cols:
                view.remove_template_id(tid)
                deleted += 1
        if deleted == 0:
            return 0
        view["effective_structural_hash"] = schema.effective_structural_hash
        refresh_template_store_indexes(view)
        save_template_store(view)
    return deleted


def _diff_implies_remap(schema_diff: SchemaDiff) -> bool:
    """True when the diff carries any rename that must be propagated to templates."""

    if schema_diff.table_renames:
        return True
    return any(td.renamed_columns for td in schema_diff.per_table.values())


def _renames_from_diff(
    schema_diff: SchemaDiff,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]:
    """Project a SchemaDiff into the ``(renamed_tables, renamed_columns)`` shape used by remap."""

    renamed_tables = tuple(schema_diff.table_renames)
    renamed_columns: list[tuple[str, str, str]] = []
    for table, td in schema_diff.per_table.items():
        for old, new in td.renamed_columns:
            renamed_columns.append((table, old, new))
    return renamed_tables, tuple(renamed_columns)


def _value_type_changes_from_diff(
    schema_diff: SchemaDiff,
) -> tuple[tuple[str, str, str], ...]:
    """Flatten per-table ``value_type_changed_columns`` into ``(column, old_vt, new_vt)`` tuples."""

    out: list[tuple[str, str, str]] = []
    for _table, td in schema_diff.per_table.items():
        for col, old_vt, new_vt in td.value_type_changed_columns:
            out.append((col, old_vt, new_vt))
    return tuple(out)


def _stamp_manifest(
    artifacts_dir: str,
    schema: SchemaGraph,
    *,
    tier: MigrationTier,
    last_action: str,
) -> None:
    """Write the artifact manifest with current schema fingerprints and the given tier."""

    write_artifact_manifest(
        artifacts_dir,
        structural_hash=schema.structural_hash,
        profiling_hash=schema.profiling_hash,
        scope_hash=schema.scope_hash,
        effective_structural_hash=schema.effective_structural_hash,
        notes_hash=schema.notes_hash,
        semantic_edges_hash=schema.semantic_edges_hash,
        last_migration_tier=tier.value,
        last_action=last_action,
    )


def apply_migration_policy(
    artifacts_dir: str,
    schema: SchemaGraph,
    *,
    allow_destructive: bool = True,
    previous_schema: SchemaGraph | None = None,
    schema_diff: SchemaDiff | None = None,
) -> MigrationReport:
    """
    Reconcile the persisted template store against ``schema``.

    When ``schema_diff`` is provided and non-empty, the diff drives tier selection and surgical invalidation directly: renames feed :func:`apply_schema_rename_migration_to_store` and dropped tables, dropped columns, and ``value_type``-changed columns feed :func:`surgical_invalidate_templates_by_diff`. REMAP and SOFT_REFRESH may co-occur on a mixed diff.

    When ``schema_diff`` is ``None`` the function falls back to the legacy path that re-derives a rename plan from ``previous_schema`` via :func:`try_rename_migration_plan`; this remains the only path available after a full rebuild that did not produce a diff.
    """

    os.makedirs(artifacts_dir, exist_ok=True)
    with artifact_lock(artifacts_dir):
        return _apply_migration_policy_locked(
            artifacts_dir,
            schema,
            allow_destructive=allow_destructive,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )


def _apply_migration_policy_locked(
    artifacts_dir: str,
    schema: SchemaGraph,
    *,
    allow_destructive: bool,
    previous_schema: SchemaGraph | None,
    schema_diff: SchemaDiff | None,
) -> MigrationReport:
    """Body of :func:`apply_migration_policy` executed under the artifacts-dir lock."""

    stored = read_artifact_manifest(artifacts_dir)
    tier = classify_migration_tier(stored, schema, previous_schema=previous_schema, schema_diff=schema_diff)
    if tier == MigrationTier.NO_CHANGE:
        return MigrationReport(tier=tier)
    if schema_diff is not None and not schema_diff.is_empty:
        return _apply_diff_driven_policy(
            artifacts_dir,
            schema,
            schema_diff,
            allow_destructive=allow_destructive,
        )
    if tier == MigrationTier.SOFT_REFRESH:
        debug("[templates.apply_migration_policy] soft_refresh: updating manifest fingerprints")
        _stamp_manifest(artifacts_dir, schema, tier=tier, last_action="soft_refresh")
        return MigrationReport(tier=tier)
    if tier == MigrationTier.REMAP:
        plan = try_rename_migration_plan(previous_schema, schema) if previous_schema is not None else None
        if plan is None:
            if not allow_destructive:
                return MigrationReport(tier=MigrationTier.DESTRUCTIVE)
            destroyed = _disk_template_row_count(artifacts_dir)
            destructive_migration_execute(artifacts_dir, schema)
            return MigrationReport(
                tier=MigrationTier.DESTRUCTIVE,
                destroyed_templates=destroyed,
            )
        renamed_tables, renamed_columns = plan
        remapped, destroyed = apply_schema_rename_migration_to_store(
            artifacts_dir,
            schema,
            renamed_tables,
            renamed_columns,
        )
        _stamp_manifest(artifacts_dir, schema, tier=MigrationTier.REMAP, last_action="remap")
        return MigrationReport(
            tier=MigrationTier.REMAP,
            renamed_tables=renamed_tables,
            renamed_columns=renamed_columns,
            destroyed_templates=destroyed,
            remapped_templates=remapped,
        )
    if not allow_destructive:
        return MigrationReport(tier=MigrationTier.DESTRUCTIVE)
    destroyed = _disk_template_row_count(artifacts_dir)
    destructive_migration_execute(artifacts_dir, schema)
    return MigrationReport(
        tier=MigrationTier.DESTRUCTIVE,
        destroyed_templates=destroyed,
    )


def _apply_diff_driven_policy(
    artifacts_dir: str,
    schema: SchemaGraph,
    schema_diff: SchemaDiff,
    *,
    allow_destructive: bool,
) -> MigrationReport:
    """Diff-driven policy dispatch: REMAP for renames, SOFT_REFRESH + surgery for drops/vt-changes."""

    del allow_destructive
    renamed_tables, renamed_columns = _renames_from_diff(schema_diff)
    has_remap = bool(renamed_tables or renamed_columns)
    tombstoned_tables, tombstoned_cols = _surgical_invalidation_targets(schema_diff)
    has_surgery = bool(tombstoned_tables or tombstoned_cols)
    remapped = 0
    destroyed = 0
    surgically = 0

    if has_surgery:
        surgically = surgical_invalidate_templates_by_diff(artifacts_dir, schema, schema_diff)
    if has_remap:
        remapped, destroyed = apply_schema_rename_migration_to_store(
            artifacts_dir,
            schema,
            renamed_tables,
            renamed_columns,
        )
    if has_remap and not has_surgery:
        tier = MigrationTier.REMAP
        last_action = "remap"
    elif has_remap and has_surgery:
        tier = MigrationTier.REMAP
        last_action = "remap_and_surgical"
    else:
        tier = MigrationTier.SOFT_REFRESH
        last_action = "soft_refresh_surgical" if has_surgery else "soft_refresh"
    _stamp_manifest(artifacts_dir, schema, tier=tier, last_action=last_action)
    return MigrationReport(
        tier=tier,
        renamed_tables=renamed_tables,
        renamed_columns=renamed_columns,
        destroyed_templates=destroyed,
        remapped_templates=remapped,
        surgically_invalidated=surgically,
        dropped_tables=tuple(schema_diff.dropped_tables),
        value_type_changed_columns=_value_type_changes_from_diff(schema_diff),
    )


def demote_template_to_rejected(
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Template],
    tmpl: Template,
    schema: SchemaGraph,
    intent: RuntimeIntent,
    sql: str,
    q_norm: str,
    reason: str,
) -> None:
    """Remove an accepted template and persist one intent-rejection feedback row."""

    if tmpl.id in templates:
        del templates[tmpl.id]
    entry = summarize_failure_for_memory(
        question=q_norm,
        intent=intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=schema.effective_structural_hash,
        user_reason=reason,
        sql=sql,
    )
    record_question_feedback(store, q_norm, entry)
    templates_to_store(store, templates)
    debug(f"[templates.demote_template_to_rejected] demoted template={tmpl.id}")


def promote_trust(template: Template, q_norm: str = "") -> bool:
    """
    Promote ``template`` from trust=1 to trust=2.

    Promotion fires when the per-(template, question) pair has accumulated at least
    :attr:`PolicyConfig.TRUST_PROMOTE_PER_QUESTION_ACCEPTS` accepts AND the template's
    overall reject ratio is at most :attr:`PolicyConfig.TRUST_PROMOTE_MAX_REJECT_RATIO`.
    Trust=2 is terminal (no further promotion). Trust=1 is the floor (no demotion).

    Args:

        template: The accepted ``Template`` to potentially promote.

        q_norm: Normalized question driving this acceptance. Required for the
        per-pair gate. If empty or absent from ``feedback_by_question``, no
        promotion occurs.

    Returns:

        ``True`` if trust_level transitioned from 1 to 2, ``False`` otherwise.
    """
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


def record_template_feedback(template: Template, accept: bool) -> None:
    """
    Record accept or reject feedback on a template.

    Args:

        template: The `Template` to update in-place.

        accept: `True` to increment the accept counter, `False` for the reject counter.

    Returns:

        None.
    """
    if accept:
        template.stats.accept += 1
    else:
        template.stats.reject += 1
    debug(f"[templates.record_template_feedback] recorded: id={template.id} accept={accept}")


def record_per_question_feedback(
    template: Template,
    q_norm: str,
    accept: bool,
    path: int,
) -> FeedbackCounts:
    """
    Update ``template.feedback_by_question[q_norm]`` with one accept/reject event.

    Args:

        template: Accepted ``Template`` whose per-question dict is mutated in place.

        q_norm: Normalized question key (must be exact-match per §6.5).

        accept: ``True`` to increment ``accepts``, ``False`` to increment ``rejects``.

        path: Generation path (1-5) that produced this event; stored as ``last_path``.

    Returns:

        The updated ``FeedbackCounts`` entry for ``q_norm``.
    """
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


def _apply_structural_defaults_from_intent(t: Template, intent: RuntimeIntent) -> None:
    """
    Copy structural parameter values from *intent* onto *t.structural_defaults*.

    Args:

        t: Template whose ``structural_defaults`` mapping is updated in place.

        intent: Accepted runtime intent carrying flattened param values.
    """

    for pk, pv in flatten_param_values(intent).items():
        if is_structural_param_key(pk):
            t.structural_defaults[pk] = pv


def path_bucket(path: GenerationPath | str | int | None) -> int:
    """
    Return the integer bucket (1-5) for a ``GenerationPath`` or its string code.

    Args:

        path: A ``GenerationPath`` enum, its ``.code`` string (e.g., ``"2.1"``),
        an int already in 1-5, or ``None``.

    Returns:

        Integer bucket in 1..5, or ``0`` when the input cannot be resolved.
    """
    if path is None:
        return 0
    if isinstance(path, int):
        return int(path) if 1 <= int(path) <= 5 else 0
    code = path.code if isinstance(path, GenerationPath) else str(path)
    if not code:
        return 0
    head = code[0]
    return int(head) if head.isdigit() else 0


def should_auto_accept_for_question(
    template: Template,
    q_norm: str,
    *,
    reuse_history_index: int | None = None,
) -> bool:
    """
    Decide whether a generated answer can skip user confirmation for ``q_norm``.

    Rule: auto-accept iff trust is at least two and the matched ``value_history`` row's
    ``accept_counts`` meets :data:`TRUST_AUTO_ACCEPT_THRESHOLD`.

    Args:

        template: Candidate accepted ``Template`` reused for this turn.

        q_norm: Normalised question text for the current turn (exact-match keying when resolving an index).

        reuse_history_index: Row index from fuzzy reuse when known; otherwise *q_norm* must equal a stored ``questions`` row.

    Returns:

        ``True`` when confirmation can be skipped, ``False`` otherwise.
    """
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


def reject_out_per_question(
    templates: dict[str, Template],
    template: Template,
    q_norm: str,
) -> tuple[bool, bool]:
    """
    Apply per-pair reject-out semantics when ``feedback_by_question[q_norm].rejects`` reaches the threshold.

    When a (template, question) pair accumulates
    :attr:`PolicyConfig.PER_QUESTION_REJECT_OUT_THRESHOLD` rejects, the entry is
    removed from ``template.feedback_by_question``. If after removal no remaining
    entry has a positive accept count, the template itself is deleted from
    ``templates``. Caller should persist question-level rejection memory separately when needed.

    Args:

        templates: Mutable accepted-templates dict keyed by template id.

        template: Accepted ``Template`` to evaluate and possibly mutate.

        q_norm: Normalized question whose pair should be considered.

    Returns:

        ``(removed_pair, deleted_template)`` — both booleans.
    """
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


def empty_template_store(effective_structural_hash: str) -> TemplateStoreView:
    """Return a fresh in-memory partitioned template store."""

    return TemplateStoreView.empty(EngineConfig.TEMPLATE_STORE_DIR, effective_structural_hash)


def _normalize_loaded_template_store_document(d: dict[str, Any]) -> None:
    """Remove superseded top-level keys and ensure ``question_feedback`` exists."""

    for legacy_key in (
        "rejected_templates",
        "rejected_intents",
        "intent_failure_log",
        "next_reject_id",
        "next_rejected_intent_id",
        "next_intent_failure_id",
    ):
        d.pop(legacy_key, None)
    d.setdefault("question_feedback", {})
    d.pop("negative_memory", None)
    d.pop("semantic_rejections", None)


def _load_partitioned_view_unlocked(store_dir: str) -> TemplateStoreView | None:
    """Read header only and return a view (no hash reconciliation)."""

    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    if not os.path.isfile(header_path):
        return None
    try:
        header = read_gzip_json(header_path)
    except (
        OSError,
        EOFError,
        gzip.BadGzipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return None
    if not isinstance(header, dict):
        return None
    return TemplateStoreView.from_header_payload(store_dir, header)


def load_template_store(
    effective_structural_hash: str,
    schema: SchemaGraph | None = None,
) -> TemplateStoreView:
    """
    Load template store from disk or create an empty store.

    When the on-disk ``effective_structural_hash`` matches, returns a :class:`TemplateStoreView`
    backed by the header and lazy partition files. Otherwise, if *schema* is provided,
    reconciles rows in place, stamps the new hash, rewrites shards, and returns the mutated view.
    Without *schema*, returns a fresh empty view.

    Args:

        effective_structural_hash: Current effective structural fingerprint for the graph.

        schema: Optional live schema graph used for surgical reconciliation.

    Returns:

        The template store view with indexes and ``question_feedback`` loaded from the header.
    """

    if PolicyConfig.REGENERATE_TEMPLATE_STORE:
        debug("[templates.load_template_store] REGENERATE_TEMPLATE_STORE: returning empty store")
        return empty_template_store(effective_structural_hash)

    store_dir = EngineConfig.TEMPLATE_STORE_DIR
    adir = os.path.dirname(os.path.abspath(store_dir))

    with artifact_lock(adir):
        return _load_template_store_locked(
            store_dir,
            adir,
            effective_structural_hash,
            schema,
        )


def _load_template_store_locked(
    store_dir: str,
    adir: str,
    effective_structural_hash: str,
    schema: SchemaGraph | None,
) -> TemplateStoreView:
    """Body of :func:`load_template_store` executed under the artifacts-dir lock."""

    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    legacy_path = os.path.join(adir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)

    if not os.path.isfile(header_path):
        if os.path.isfile(legacy_path):
            debug(
                f"[templates.load_template_store] legacy_monolithic_ignored: path={legacy_path} "
                "(partitioned header missing; not migrated)",
            )
        debug(f"[templates.load_template_store] no_header: path={header_path}")
        return empty_template_store(effective_structural_hash)
    try:
        header = read_gzip_json(header_path)
    except (
        OSError,
        EOFError,
        gzip.BadGzipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        debug(f"[templates.load_template_store] corrupt_or_unreadable: path={header_path} err={exc!r}")
        write_artifact_manifest(
            adir,
            last_corruption_at=datetime.now(timezone.utc).isoformat(),
            last_action="corrupt_template_store",
        )
        return empty_template_store(effective_structural_hash)
    if not isinstance(header, dict):
        return empty_template_store(effective_structural_hash)

    index_keys = (
        SHAPE_QUESTION_INDEX_KEY,
        TEMPLATE_INTENT_KEY_INDEX_KEY,
        TEMPLATE_UNION_FAMILY_INDEX_KEY,
        TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    )
    need_index_rebuild = any(k not in header or not isinstance(header.get(k), dict) for k in index_keys)

    stored = header.get("effective_structural_hash", header.get("schema_hash", ""))
    view = TemplateStoreView.from_header_payload(store_dir, header)
    if need_index_rebuild:
        refresh_template_store_indexes(view)

    if stored == effective_structural_hash:
        qfn = 0
        qf = view.question_feedback
        for _qk, rows in qf.items():
            if isinstance(rows, list):
                qfn += len(rows)
        debug(
            f"[templates.load_template_store] loaded: templates={len(view.partition_map)} question_feedback_rows={qfn}"
        )
        return view
    if schema is None:
        debug("[templates.load_template_store] effective_structural_hash_mismatch: resetting")
        return empty_template_store(effective_structural_hash)
    debug("[templates.load_template_store] effective_structural_hash_mismatch: reconciling")
    reconcile_template_store(view, schema)
    view["effective_structural_hash"] = effective_structural_hash
    save_template_store(view)
    return view


def save_template_store(store: dict[str, Any] | TemplateStoreView) -> None:
    """
    Save template store to disk.

    Converts all non-serialisable objects (for example, sets) before writing JSON.
    For a :class:`TemplateStoreView`, writes the header and only dirty partition shards.
    """

    if not isinstance(store, TemplateStoreView):
        raise TypeError("save_template_store requires a TemplateStoreView")
    store_dir = store._store_dir
    adir = os.path.dirname(os.path.abspath(store_dir))

    qfn = 0
    qf = store.question_feedback
    for _qk, rows in qf.items():
        if isinstance(rows, list):
            qfn += len(rows)
    debug(f"[templates.save_template_store] saving: templates={len(store.partition_map)} question_feedback_rows={qfn}")
    refresh_template_store_indexes(store)
    header_doc = _convert_to_json_serializable(store._header_document())
    _debug_check_types(header_doc, "store")
    with artifact_lock(adir):
        os.makedirs(store_dir, exist_ok=True)
        header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        write_gzip_json_atomic(header_path, header_doc, sort_keys=True)
        dirty = set(store._dirty_partitions)
        for part in sorted(dirty):
            payload = store._partition_cache.get(part)
            if payload is None:
                payload = {}
                if os.path.isfile(store._partition_file_path(part)):
                    try:
                        disk = read_gzip_json(store._partition_file_path(part))
                        payload = disk if isinstance(disk, dict) else {}
                    except (
                        OSError,
                        EOFError,
                        gzip.BadGzipFile,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                    ):
                        payload = {}
            store._flush_partition_to_disk(part, payload)
        store._dirty_partitions.clear()
        prev = read_artifact_manifest(adir)
        eff = str(store.effective_structural_hash or "")
        write_artifact_manifest(
            adir,
            structural_hash=prev.structural_hash if prev else "",
            profiling_hash=prev.profiling_hash if prev else "",
            scope_hash=prev.scope_hash if prev else "",
            effective_structural_hash=eff or (prev.effective_structural_hash if prev else ""),
            notes_hash=prev.notes_hash if prev else "",
            semantic_edges_hash=prev.semantic_edges_hash if prev else "",
            last_migration_tier=prev.last_migration_tier if prev else "",
            last_migration_at=prev.last_migration_at if prev else "",
            last_action="template_store_save",
        )
    debug(f"[templates.save_template_store] complete: dir={store_dir}")


def store_to_templates(store: dict[str, Any] | TemplateStoreView) -> dict[str, Template]:
    """
    Convert store dict to `Template` objects with nested dataclass reconstruction.

    Args:

        store: The raw template store dict loaded from disk.

    Returns:

        Dict mapping template ID to `Template` dataclass instance.
    """
    if isinstance(store, TemplateStoreView):
        out: dict[str, Template] = {}
        for tid in sorted(store.partition_map.keys()):
            t = store.get_template(tid)
            if t is not None:
                out[tid] = t
        return out
    out = {}
    for tid, v in store.get("templates", {}).items():
        t = Template.from_dict(v)
        t.sql_fp = compute_sql_fp(t.sql_param or "", sqlglot_dialect=active_sqlglot_dialect())
        out[tid] = t
    return out


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert values to JSON-serialisable forms (e.g. sets to lists).

    Args:

        obj: Arbitrary nested structure of dicts, lists, tuples, sets, or scalars.

    Returns:

        A structure of the same shape with sets and frozensets coerced to lists.
    """
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_convert_to_json_serializable(item) for item in obj]
    return obj


def _debug_check_types(obj: Any, path: str = "root") -> None:
    """
    Log debug messages when non-JSON types (e.g. sets) appear in nested structures.

    Args:

        obj: Root or nested value to walk.

        path: Dot-bracket path string for log context.

    Returns:

        None.
    """
    if isinstance(obj, set):
        debug(f"[templates.debug_check_types] found_set: path={path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _debug_check_types(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        for i, item in enumerate(obj):
            _debug_check_types(item, f"{path}[{i}]")


def templates_to_store(
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Template],
) -> dict[str, Any] | TemplateStoreView:
    """
    Convert Template objects to store dict format.

    Args:

        store: The mutable template store dict to update in-place.

        templates: Dict of template ID to `Template` to serialise.

    Returns:

        The updated store dict with `templates` key populated.
    """
    debug(f"[templates.templates_to_store] converting: count={len(templates)}")
    for tid, t in templates.items():
        template_dict = t.to_dict()
        _debug_check_types(template_dict, f"template[{tid}]")
    store["templates"] = {k: _convert_to_json_serializable(v.to_dict()) for k, v in templates.items()}
    refresh_template_store_indexes(store, template_objs=list(templates.values()))
    return store


def _append_runtime_paraphrase_variants(
    vh: ValueHistory,
    primary_q: str,
    param_values: dict[str, ParamValue],
    natural_language: str,
    schema: SchemaGraph,
    tables_hint: list[str],
) -> None:
    """Generate bounded LLM paraphrases after acceptance and append distinct surviving rows."""

    styles = list(SeedWarmupConfig.WARMUP_QUESTION_STYLES)
    added = 0
    for i in range(0, len(styles) - 1, 2):
        outs = generate_paraphrases_of_seed_question(
            primary_q,
            schema,
            tables_hint,
            style_pair=(styles[i], styles[i + 1]),
        )
        if not outs:
            continue
        for p in outs:
            ok, _, _ = validate_question(p)
            if not ok:
                continue
            if p in vh.questions:
                continue
            vh.append_question_variant(
                p,
                accept_count=0,
                param_values=dict(param_values),
                natural_language=natural_language,
            )
            added += 1
            if added >= RUNTIME_PARAPHRASE_COUNT:
                return


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
    """
    Append or merge accepted history using typo-corrected text and an optional zero-accept paraphrase row.

    Args:

        vh: Target history.

        param_values: Resolved parameter dict for this acceptance.

        natural_language: Intent natural-language line.

        form_storage: Optional corrected vs normalized bundle; falls back to *q_norm_fallback* when absent.

        q_norm_fallback: Legacy single question key when *form_storage* is ``None``.

        schema: Optional schema graph for runtime paraphrase generation.

        tables_hint: Optional tables list grounding paraphrase terminology.

    Returns:

        None.
    """

    primary_q = form_storage.corrected if form_storage is not None else q_norm_fallback
    vh.add(param_values, primary_q, natural_language, accept_increment=1)
    if form_storage is not None and not form_storage.accept_via_normalized_lookup_only:
        nopt = form_storage.normalized_optional
        if nopt and nopt != primary_q and not form_storage.normalized_negative_memory_dropped:
            vh.append_question_variant(
                nopt,
                accept_count=0,
                param_values=dict(param_values),
                natural_language=natural_language,
            )
    if schema is not None and tables_hint and llm_credentials_configured():
        _append_runtime_paraphrase_variants(
            vh,
            primary_q,
            param_values,
            natural_language,
            schema,
            tables_hint,
        )


def insert_template(
    store: dict[str, Any] | TemplateStoreView,
    templates: dict[str, Template],
    schema,
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
) -> Template:
    """
    Create and insert a new accepted template, or merge into a fingerprint-compatible match.

    Mirrors *templates* into *store* on create and merge so callers can reload from *store*.

    Args:

        store: The mutable template store dict.

        templates: The accepted templates dict.

        schema: The schema graph (for `schema_hash` and table enumeration).

        q_norm: The normalised question.

        intent: The `RuntimeIntent`.

        sql: The accepted SQL.

        dialect: Optional dialect instance reserved for future template-side preparation hooks.

        structural_match_templates: Trusted same-intent-key templates for fingerprint merge (path 5).

        template_source: Stored ``Template.source`` for new rows (default ``human``).

        template_trust_level: Stored ``Template.trust_level`` for new rows.

        template_initial_stats: Optional stats for a new row; default is one accept.

        template_value_history: Optional pre-built history; default is a single row from *q_norm* or corrected text.

        form_storage: Optional typo-corrected vs canonical bundle for parallel ``ValueHistory`` rows.

    Returns:

        The existing or newly created `Template`.
    """
    sql_canon = canonicalize_sql(sql)
    if intent.sql_param:
        sql_param = intent.sql_param
    else:
        sql_norm = normalize_sql(sql_canon)
        sql_param, _ = parameter_abstract(sql_norm, sqlglot_dialect=active_sqlglot_dialect())
    sql_fp_val = compute_sql_fp(sql_param, sqlglot_dialect=active_sqlglot_dialect())
    tables_used = extract_tables_from_sql(
        sql_canon, list(schema.tables.keys()), sqlglot_dialect=active_sqlglot_dialect()
    )
    colmap_sig_val = colmap_signature(intent.column_map)
    ikey = intent_key(intent)

    intent_sig = runtime_intent_to_concrete(intent, "")

    exec_fp = join_fingerprint_from_runtime_intent(intent)
    structural_list = sorted(structural_match_templates or [], key=lambda x: x.id)

    def _merge_accept(t: Template) -> Template:
        debug(f"[templates.insert_template] duplicate_found: id={t.id}")
        t.stats.accept += 1
        prior_counts = t.feedback_by_question.get(q_norm)
        merge_path = int(prior_counts.last_path) if prior_counts and prior_counts.last_path else 1
        record_per_question_feedback(t, q_norm, accept=True, path=merge_path)
        promote_trust(t, q_norm)
        _apply_structural_defaults_from_intent(t, intent)
        all_pv = flatten_param_values(intent)
        record_value_history_on_accept(
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

    if structural_list:
        for t in structural_list:
            if join_fingerprint_from_concrete_intent(t.intent_signature) == exec_fp:
                merged = _merge_accept(t)
                templates_to_store(store, templates)
                return merged
    else:
        for t in sorted(templates.values(), key=lambda x: x.id):
            if t.intent_key != ikey:
                continue
            _, cc, _ = compute_intent_union(intent, t.intent_signature)
            if cc:
                continue
            if join_fingerprint_from_concrete_intent(t.intent_signature) != exec_fp:
                continue
            merged = _merge_accept(t)
            templates_to_store(store, templates)
            return merged

    debug("[templates.insert_template] no_duplicate: creating_new")

    tid = f"T{int(store['next_id']):04d}"
    store["next_id"] = int(store["next_id"]) + 1

    all_pv = flatten_param_values(intent)

    stats_new = template_initial_stats if template_initial_stats is not None else TemplateStats(accept=1, reject=0)
    if template_value_history is not None:
        vh_new = template_value_history
    else:
        primary_q = form_storage.corrected if form_storage is not None else q_norm
        nl0 = intent.natural_language or ""
        vh_new = ValueHistory(
            param_values=[all_pv],
            questions=[primary_q],
            natural_language=[nl0],
        )
        if (
            form_storage is not None
            and not form_storage.accept_via_normalized_lookup_only
            and form_storage.normalized_optional
            and form_storage.normalized_optional != primary_q
            and not form_storage.normalized_negative_memory_dropped
        ):
            vh_new.append_question_variant(
                form_storage.normalized_optional,
                accept_count=0,
                param_values=all_pv,
                natural_language=nl0,
            )

    sig_aliases_insert: dict[str, str] = {}
    for sc in intent.select_cols or []:
        alias = generate_col_alias(sc)
        if alias:
            sig_aliases_insert[sc.signature_key] = alias

    tmpl = Template(
        id=tid,
        effective_structural_hash=schema.effective_structural_hash,
        intent_signature=intent_sig,
        intent_key=ikey,
        tables_used=sorted(tables_used),
        sql_param=sql_param,
        sql_fp=sql_fp_val,
        shape=(
            intent.sql_shape
            if intent.sql_shape
            else sql_shape(sql_canon, intent, sqlglot_dialect=active_sqlglot_dialect())
        ),
        colmap_sig=colmap_sig_val,
        value_history=vh_new,
        stats=stats_new,
        source=template_source,
        trust_level=template_trust_level,
        structural_defaults={k: v for k, v in all_pv.items() if is_structural_param_key(k)},
        display_alias_map=sig_aliases_insert,
    )

    debug(
        f"[templates.insert_template] CREATED template natural_language={tmpl.value_history.natural_language} chosen_join_candidate_id='{tmpl.chosen_join_candidate_id}' chosen_join_path_signature={tmpl.chosen_join_path_signature}"
    )

    templates[tid] = tmpl
    debug(f"[templates.insert_template] created: id={tid}")
    templates_to_store(store, templates)
    return tmpl


def _parse_schema_migration_map_payload(payload: dict[str, Any]) -> SchemaMigrationMap:
    """Normalise a decoded JSON object into a :class:`SchemaMigrationMap`."""

    ver = payload.get("version")
    if not isinstance(ver, int) or ver < 1:
        raise MigrationPendingError("schema_migration_map.json: invalid or missing version")
    action_raw = str(payload.get("action") or "").strip().lower()
    if action_raw not in (
        MIGRATION_MAP_ACTION_REMAP,
        MIGRATION_MAP_ACTION_DESTRUCTIVE,
        MIGRATION_MAP_ACTION_ABORT,
    ):
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
                    cols_o.append(SchemaMigrationMapEntry(entry_type="column", table=bt, from_name=ocn, to_name=ncn))
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
                    SchemaMigrationMapEntry(
                        entry_type="dropped_column",
                        table=tpart,
                        from_name=cpart,
                        to_name="",
                    )
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
                        SchemaMigrationMapEntry(
                            entry_type="added_column",
                            table=bt,
                            to_name=ncn,
                            from_name="",
                        )
                    )
    refresh_desc = payload.get("refresh_existing_descriptions_on_addition", False)
    if not isinstance(refresh_desc, bool):
        raise MigrationPendingError(
            "schema_migration_map.json: refresh_existing_descriptions_on_addition must be a boolean"
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
        refresh_existing_descriptions_on_addition=refresh_desc,
    )


def _load_schema_migration_map(cwd_path: Path) -> SchemaMigrationMap | None:
    """
    Read ``schema_migration_map.json`` from *cwd_path* and parse it into a :class:`SchemaMigrationMap`.

    Returns None when the file is absent. Raises :class:`MigrationPendingError` when the file is present but not a valid object map.
    """

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
    return _parse_schema_migration_map_payload(raw)


def _validate_schema_migration_map(
    map_obj: SchemaMigrationMap,
    cached_schema: SchemaGraph | None,
    live_schema: SchemaGraph,
) -> None:
    """
    Check rename and drop entries against the cached and live schema graphs.

    Raises :class:`MigrationPendingError` with prefix ``STALE_MAP:`` when the map was produced for a different cached snapshot so the file should be removed and init retried without it. Raises :class:`MigrationPendingError` for any other validation failure.
    """

    problems: list[str] = []
    stale: list[str] = []
    tmap = {
        e.from_name: e.to_name for e in map_obj.table_renames if e.entry_type == "table" and e.from_name and e.to_name
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
    return SchemaDiff(
        dropped_tables=tuple(sorted(set(map_obj.dropped_tables))),
        per_table=per_table,
    )


def _schema_diff_for_sidecar_renames(map_obj: SchemaMigrationMap) -> SchemaDiff | None:
    """Build a rename-only :class:`SchemaDiff` for overrides sidecar migration."""

    tr = tuple(
        (e.from_name, e.to_name) for e in map_obj.table_renames if e.entry_type == "table" and e.from_name != e.to_name
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


def _apply_schema_migration_map(
    map_obj: SchemaMigrationMap,
    artifacts_dir: str,
    schema: SchemaGraph,
    schema_json_path: Path,
) -> MigrationReport:
    """
    Apply a user migration map to templates, simulation artifacts, and optionally the overrides sidecar.

    Dispatches on ``action``. ``remap`` runs rename migration plus surgical drops listed in the map; the learning-reset action clears learning artifacts. Does not delete the editor JSON; callers unlink ``schema_migration_map.json`` after success.
    """

    if map_obj.action == MIGRATION_MAP_ACTION_ABORT:
        raise MigrationPendingError("internal: abort action must be handled before _apply_schema_migration_map")
    if map_obj.action == MIGRATION_MAP_ACTION_DESTRUCTIVE:
        destroyed = _disk_template_row_count(artifacts_dir)
        destructive_migration_execute(artifacts_dir, schema)
        _stamp_manifest(
            artifacts_dir,
            schema,
            tier=MigrationTier.DESTRUCTIVE,
            last_action=ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP,
        )
        return MigrationReport(tier=MigrationTier.DESTRUCTIVE, destroyed_templates=destroyed)
    sidecar_diff = _schema_diff_for_sidecar_renames(map_obj)
    if sidecar_diff is not None:
        _migrate_sidecar_for_diff(schema_json_path, sidecar_diff)
    renamed_tables = tuple((e.from_name, e.to_name) for e in map_obj.table_renames if e.entry_type == "table")
    renamed_columns = tuple(
        (e.table, e.from_name, e.to_name) for e in map_obj.column_renames if e.entry_type == "column"
    )
    remapped = 0
    destroyed = 0
    if renamed_tables or renamed_columns:
        remapped, destroyed = apply_schema_rename_migration_to_store(
            artifacts_dir, schema, renamed_tables, renamed_columns
        )
    surg = 0
    drop_diff = _schema_diff_from_user_drops(map_obj)
    if drop_diff is not None:
        surg = surgical_invalidate_templates_by_diff(artifacts_dir, schema, drop_diff)
    _stamp_manifest(
        artifacts_dir,
        schema,
        tier=MigrationTier.REMAP,
        last_action=ARTIFACT_LAST_ACTION_REMAP_USER_MAP,
    )
    return MigrationReport(
        tier=MigrationTier.REMAP,
        renamed_tables=renamed_tables,
        renamed_columns=renamed_columns,
        remapped_templates=remapped,
        destroyed_templates=destroyed,
        surgically_invalidated=surg,
        dropped_tables=tuple(map_obj.dropped_tables),
    )


def export_schema_migration_map_skeleton(
    cwd_path: Path,
    *,
    tier: MigrationTier,
    schema_diff: SchemaDiff | None,
    rename_plan: (tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None),
) -> Path:
    """
    Write ``schema_migration_map.json`` with auto-detected fields pre-filled.

    The file is written in the process working directory resolved by *cwd_path*.
    """

    path = cwd_path / MIGRATION_MAP_FILENAME
    action = MIGRATION_MAP_ACTION_DESTRUCTIVE if tier == MigrationTier.DESTRUCTIVE else MIGRATION_MAP_ACTION_REMAP
    table_renames: dict[str, str] = {}
    column_renames: dict[str, dict[str, str]] = {}
    if rename_plan is not None:
        rt, rc = rename_plan
        table_renames = {o: n for o, n in rt}
        for ot, oc, nc in rc:
            column_renames.setdefault(ot, {})[oc] = nc
    dropped_tables: list[str] = []
    dropped_columns: list[str] = []
    added_tables: list[str] = []
    added_columns: dict[str, list[str]] = {}
    if schema_diff is not None:
        dropped_tables = list(schema_diff.dropped_tables)
        for tname, td in schema_diff.per_table.items():
            for c in td.dropped_columns:
                dropped_columns.append(f"{tname}.{c}")
        added_tables = list(schema_diff.added_tables)
        for tname, td in schema_diff.per_table.items():
            if td.added_columns:
                added_columns[tname] = list(td.added_columns)
    payload = {
        "version": 1,
        "action": action,
        "table_renames": table_renames,
        "column_renames": column_renames,
        "dropped_tables": dropped_tables,
        "dropped_columns": dropped_columns,
        "added_tables": added_tables,
        "added_columns": added_columns,
        "refresh_existing_descriptions_on_addition": False,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


register_templates_module(sys.modules[__name__])
