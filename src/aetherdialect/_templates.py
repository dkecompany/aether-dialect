"""Template store I/O, trust levels, per-question feedback memory, and schema-ref helpers. Artifacts keyed by ``schema_graph_id`` (identity rotation orphans these under ``orphaned/<old_id>/``): +----------------------+----------------------------------------------------------+ | Kind                 | Location                                                 | +======================+==========================================================+ | template_shards      | ``intent_templates/spaces/<space>/header.json.gz`` and   | |                      | ``partition_*.json.gz``                                  | | feedback_shards      | ``intent_templates/spaces/<space>/feedback/partition_*`` | | join_feedback        | feedback shard bodies via ``feedback_shard_index``       | | warmup_lattices      | ``anchor_lattice/lattice_<schema_graph_id>_v*.json``     | | qsim_skeletons       | ``qsim_skeletons.json.gz`` (``schema_graph_id`` field)   | | schema_context_cache | ``schema_context.json``                                  | | prompt_cache_refs    | ``schema_context.json`` scope for provider cache keys    | | write_queue          | ``write_queue.jsonl`` (cleared on manifest mismatch)     | | federation_pins      | federation composite manifest ``schema_graph_id``        | +----------------------+----------------------------------------------------------+ Calls ``register_templates_module`` at module load end so ``_intent_loop`` can type-check store views without importing this module."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from ._config import EngineConfig, EngineLimits, PolicyConfig, SeedWarmupConfig
from ._constants import (
    AETHERSPACES_SEGMENT,
    ARTIFACT_MANIFEST_FILENAME,
    DIAGNOSTIC_CODE_MIGRATION_CHECKPOINT_ORPHANED,
    DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED,
    DIAGNOSTIC_CODE_TEMPLATE_STORE_ORPHANED,
    FEDERATION_MANIFEST_FILENAME,
    FEEDBACK_SHARD_INDEX_KEY,
    MASTER_AETHERSPACE_NAME,
    MIGRATION_CHECKPOINT_PREFIX,
    MIGRATION_CHECKPOINT_SCHEMA_BASENAME,
    ORPHAN_RETENTION_SECONDS,
    QUESTION_NORMALIZATION_VERSION,
    QUESTION_NORMALIZATION_VERSION_KEY,
    RENAME_HISTORY_FILENAME,
    SCHEMA_CONTEXT_CACHE_NAME,
    SHAPE_QUESTION_INDEX_KEY,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX,
    TEMPLATE_STORE_FEEDBACK_SEGMENT,
    TEMPLATE_STORE_FORMAT_VERSION,
    TEMPLATE_STORE_GLOBAL_NEXT_ID_FILENAME,
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_ORPHANED_SEGMENT,
    TEMPLATE_STORE_PARTITION_COUNT,
    TEMPLATE_STORE_PARTITION_LRU_MAX,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SEGMENT,
    TEMPLATE_STORE_SPACES_SEGMENT,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    WRITE_QUEUE_FILENAME,
)
from ._contracts_base import (
    ApprovalState,
    ArtifactManifest,
    ConfigError,
    Diagnostic,
    DiagnosticSeverity,
    DomainKnowledgeEntry,
    MigrationReport,
    MigrationTier,
    PredicateGroup,
    SchemaMigrationMap,
    SensitivityClassification,
)
from ._contracts_core import (
    ConcreteCteStep,
    ConcreteIntent,
    FeedbackCounts,
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SeedWarmupIntent,
    Template,
    ValueHistory,
)
from ._contracts_schema import SchemaDiff, SchemaGraph
from ._dialect import Dialect, DialectRegistry
from ._federation_manifest import federation_artifact_manifest_view
from ._intent_bind import (
    collect_column_refs_for_cte_step,
    collect_column_refs_for_post_processing,
    join_path_key_concrete,
    join_path_segments_fingerprint_concrete,
    join_path_segments_fingerprint_runtime,
)
from ._intent_expr import replace_refs_in_expr
from ._llm_provider import LLMProvider
from ._schema_finalize import destructive_migration_execute, reconcile_sidecar_against_graph
from ._schema_graph import (
    classify_migration_tier,
    load_schema_graph_snapshot,
    schema_diff_is_additive_only,
    try_rename_migration_plan,
)
from ._schema_profile import register_sensitivity_ratchet_artifact_scrub
from ._sql_gen import (
    build_deterministic_sql,
    canonicalize_stored_join_path_signature,
)
from ._utils import (
    active_engine_limits,
    canonicalize_sql,
    coerce_format_version,
    debug,
    format_versions_match,
    normalize_sql,
    notify,
    paths_equal,
    split_warmup_lattice_basename,
    stable_json,
)
from ._utils_artifacts import (
    apply_structural_migration_to_persisted_scopes,
    artifact_lock,
    migrate_engine_knowledge_artifacts,
    read_artifact_manifest,
    read_gzip_json,
    refresh_migration_auxiliary_artifacts,
    replace_path_atomic,
    sanitize_name_segment,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from ._utils_intent import (
    body_similarity_key_for_concrete,
    build_shape_question_index,
    is_exact_question_text_match,
    question_token_fingerprint_from_raw,
    resolve_template_for_question,
)


class BrokenRenameChainError(ValueError):
    """Raised when persisted rename history cannot connect two schema graph identities."""


@dataclass(frozen=True, slots=True)
class TemplateRefs:
    """Normalized schema references extracted from a template-like object."""

    tables: frozenset[str]
    columns: frozenset[tuple[str, str]]
    column_types: frozenset[tuple[str, str, str]]
    fk_edges: frozenset[str]
    join_path_layers: tuple[tuple[str, ...], ...]

    @staticmethod
    def _join_segment_from_edge_dict(edge: dict[str, Any] | str) -> str:
        """Format one join-graph edge as a canonical signature segment string."""
        if isinstance(edge, str):
            return str(edge).strip()
        src_table = edge.get("src_table")
        if isinstance(src_table, str):
            src_cols = edge.get("src_cols") or []
            dst_table = edge.get("dst_table")
            dst_cols = edge.get("dst_cols") or []
            return f"{src_table}.{','.join(str(c) for c in src_cols)}->{dst_table}.{','.join(str(c) for c in dst_cols)}"
        src = edge.get("src")
        dst = edge.get("dst")
        if isinstance(src, str) and isinstance(dst, str):
            return f"{src.strip()}->{dst.strip()}"
        raise KeyError(f"unsupported join path edge shape: {edge!r}")

    @staticmethod
    def _canonical_join_path_layer(signature: Sequence[str]) -> tuple[str, ...]:
        """Canonicalize one stored join-path layer for currency checks."""
        cleaned = [str(seg).strip() for seg in signature if str(seg).strip()]
        if not cleaned:
            return ()
        return tuple(canonicalize_stored_join_path_signature(cleaned))

    @staticmethod
    def _path_to_segment_layer(path: list[Any]) -> tuple[str, ...]:
        """Normalize one enumerated join path to a canonical segment layer."""
        if not path:
            return ()
        if all(isinstance(edge, str) for edge in path):
            return TemplateRefs._canonical_join_path_layer([str(edge) for edge in path])
        return TemplateRefs._canonical_join_path_layer(
            [TemplateRefs._join_segment_from_edge_dict(edge) for edge in path]
        )

    @staticmethod
    def _all_join_segments_live(schema: SchemaGraph) -> frozenset[str]:
        """Collect every join path segment string still present in ``join_paths_multi``."""
        segs: set[str] = set()
        for row in schema.join_paths_multi.values():
            for paths in row.values():
                for path in paths:
                    for edge in path:
                        segs.add(TemplateRefs._join_segment_from_edge_dict(edge))
        return frozenset(segs)

    @staticmethod
    def _all_join_path_layers_live(schema: SchemaGraph) -> frozenset[tuple[str, ...]]:
        """Collect canonical multi-hop join signatures currently enumerated in the schema."""
        layers: set[tuple[str, ...]] = set()
        for row in schema.join_paths_multi.values():
            for paths in row.values():
                for path in paths:
                    layer = TemplateRefs._path_to_segment_layer(list(path))
                    if layer:
                        layers.add(layer)
        return frozenset(layers)

    @staticmethod
    def _join_layer_is_current(layer: tuple[str, ...], live_layers: frozenset[tuple[str, ...]]) -> bool:
        """Return whether *layer* matches a current join path in ``join_paths_multi``."""
        if not layer:
            return True
        return layer in live_layers

    @staticmethod
    def _runtime_intent_schema_refs(rt: RuntimeIntent) -> TemplateRefs:
        """Build ``TemplateRefs`` from a ``RuntimeIntent`` snapshot."""
        tables = frozenset(rt.tables or ())
        columns: set[tuple[str, str]] = set()
        for bare, tbl in (rt.column_map or {}).items():
            columns.add((tbl, bare))
        fk = frozenset(str(x) for x in (rt.chosen_join_path_signature or []) if str(x).strip())
        join_layers: tuple[tuple[str, ...], ...] = ()
        if rt.chosen_join_path_signature:
            join_layers = (TemplateRefs._canonical_join_path_layer(list(rt.chosen_join_path_signature)),)
        return TemplateRefs(
            tables=tables,
            columns=frozenset(columns),
            column_types=frozenset(),
            fk_edges=fk,
            join_path_layers=join_layers,
        )

    @staticmethod
    def _column_pairs_from_qualified_refs(refs: list[str]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for ref in refs:
            if "." not in ref:
                continue
            tbl, col = ref.split(".", 1)
            if tbl and col:
                out.add((tbl, col))
        return out

    @staticmethod
    def template_schema_refs(template: Template) -> TemplateRefs:
        """Collect tables, column pairs, join layers, and type snapshots referenced by *template*."""
        tables: set[str] = set(template.tables_used or ())
        concrete = template.intent_signature
        if concrete.tables:
            tables.update(concrete.tables)
        columns: set[tuple[str, str]] = set()
        for bare, tbl in concrete.column_map.items():
            columns.add((tbl, bare))
        rt = concrete.to_runtime_skeleton()
        columns.update(TemplateRefs._column_pairs_from_qualified_refs(collect_column_refs_for_post_processing(rt)))
        for step in rt.cte_steps or []:
            columns.update(TemplateRefs._column_pairs_from_qualified_refs(collect_column_refs_for_cte_step(step)))
        fk_edges: set[str] = set()
        join_layers: list[tuple[str, ...]] = []
        main_sig = list(concrete.chosen_join_path_signature or [])
        if main_sig:
            fk_edges.update(str(seg).strip() for seg in main_sig if str(seg).strip())
            join_layers.append(TemplateRefs._canonical_join_path_layer(main_sig))
        for step in cast(Any, concrete.cte_steps or []):
            cte_sig = list(step.chosen_join_path_signature or [])
            if cte_sig:
                fk_edges.update(str(seg).strip() for seg in cte_sig if str(seg).strip())
                join_layers.append(TemplateRefs._canonical_join_path_layer(cte_sig))
        column_types: set[tuple[str, str, str]] = set()
        for key, dtype in (template.schema_column_types or {}).items():
            if "." not in key:
                continue
            tbl, col = key.split(".", 1)
            if tbl and col and str(dtype or "").strip():
                column_types.add((tbl, col, str(dtype).strip().lower()))
        return TemplateRefs(
            tables=frozenset(tables),
            columns=frozenset(columns),
            column_types=frozenset(column_types),
            fk_edges=frozenset(fk_edges),
            join_path_layers=tuple(join_layers),
        )

    @staticmethod
    def footprint_from_refs(refs: TemplateRefs) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return sorted ``(footprint_tables, footprint_columns)`` from schema refs."""
        tables = tuple(sorted(refs.tables))
        columns = tuple(sorted(f"{table}.{col}" for table, col in refs.columns))
        return tables, columns

    @staticmethod
    def stamp_template_footprint(template: Template) -> Template:
        """Fill footprint fields from schema refs when absent."""
        if template.footprint_tables and template.footprint_columns:
            return template
        tables, columns = TemplateRefs.footprint_from_refs(TemplateRefs.template_schema_refs(template))
        return replace(
            template,
            footprint_tables=template.footprint_tables or tables,
            footprint_columns=template.footprint_columns or columns,
        )

    @staticmethod
    def footprint_survives(
        template: Template,
        schema: SchemaGraph,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return whether the template footprint remains compatible with *schema* (including denies)."""
        stamped = TemplateRefs.stamp_template_footprint(template)
        reasons: list[str] = []
        deny_cols = schema.deny_columns or {}
        for table in stamped.footprint_tables:
            if table not in schema.tables:
                reasons.append(f"missing_table:{table}")
        for qualified in stamped.footprint_columns:
            if "." not in qualified:
                reasons.append(f"bad_footprint_column:{qualified}")
                continue
            table, col = qualified.split(".", 1)
            if table not in schema.tables or col not in schema.tables[table].columns:
                reasons.append(f"missing_column:{qualified}")
                continue
            if col in (deny_cols.get(table) or set()) or col in (deny_cols.get("*") or set()):
                reasons.append(f"denied_column:{qualified}")
                continue
            col_meta = schema.tables[table].columns[col]
            if col_meta.sensitivity in (SensitivityClassification.HIDDEN, SensitivityClassification.RESTRICTED):
                reasons.append(f"sensitive_column:{qualified}")
                continue
            expected = str((stamped.schema_column_types or {}).get(qualified, "") or "").strip().lower()
            current = str(schema.tables[table].columns[col].data_type or "").strip().lower()
            if expected and current and expected != current:
                reasons.append(f"column_type_mismatch:{qualified}")
        return (not reasons), tuple(reasons)

    @staticmethod
    def warmup_work_unit_schema_refs(work_unit: Mapping[str, Any]) -> TemplateRefs:
        """Derive ``TemplateRefs`` from a persisted seed-warmup work unit payload."""
        er = work_unit.get("execute_result")
        if isinstance(er, dict) and isinstance(er.get("runtime"), dict):
            return TemplateRefs._runtime_intent_schema_refs(RuntimeIntent.from_dict(er["runtime"]))
        raw_si = work_unit.get("serialized_intent")
        if isinstance(raw_si, dict):
            return TemplateRefs._runtime_intent_schema_refs(SeedWarmupIntent.from_dict(raw_si).to_runtime_intent())
        return TemplateRefs(
            tables=frozenset(),
            columns=frozenset(),
            column_types=frozenset(),
            fk_edges=frozenset(),
            join_path_layers=(),
        )

    @staticmethod
    def template_is_live(refs: TemplateRefs, schema: SchemaGraph) -> tuple[bool, tuple[str, ...]]:
        """Return whether every referenced table, column, type, and join layer still exists in *schema*."""
        reasons: list[str] = []
        for t in refs.tables:
            if t not in schema.tables:
                reasons.append(f"missing_table:{t}")
        for table, col in refs.columns:
            tm = schema.tables.get(table)
            if tm is None or col not in tm.columns:
                reasons.append(f"missing_column:{table}.{col}")
        for table, col, expected_type in refs.column_types:
            tm = schema.tables.get(table)
            if tm is None or col not in tm.columns:
                reasons.append(f"missing_column:{table}.{col}")
                continue
            current_type = str(tm.columns[col].data_type or "").strip().lower()
            if current_type and expected_type and current_type != expected_type:
                reasons.append(f"column_type_mismatch:{table}.{col}")
        live_layers = TemplateRefs._all_join_path_layers_live(schema)
        for layer in refs.join_path_layers:
            if not TemplateRefs._join_layer_is_current(layer, live_layers):
                reasons.append(f"stale_join_path:{'|'.join(layer)}")
        if refs.fk_edges:
            live_segs = TemplateRefs._all_join_segments_live(schema)
            for seg in refs.fk_edges:
                s = str(seg).strip()
                if not s:
                    continue
                if s not in live_segs:
                    reasons.append(f"missing_join_segment:{s}")
        return (len(reasons) == 0, tuple(reasons))

    @staticmethod
    def join_fingerprint_from_concrete_intent(concrete: ConcreteIntent) -> str:
        """Stable hash of main and per-CTE chosen join path signatures in declaration order."""
        return join_path_segments_fingerprint_concrete(concrete)

    @staticmethod
    def join_fingerprint_from_runtime_intent(intent: RuntimeIntent) -> str:
        """Stable hash of main and per-CTE join signatures on a runtime intent after join resolution."""
        return join_path_segments_fingerprint_runtime(intent)

    @staticmethod
    def warmup_template_store_dedupe_key(tmpl: Template) -> tuple[str, str, str]:
        """Return the ``(intent_key, join_fp, sql_fp)`` key for seed- warmup template-store merge."""
        return (
            tmpl.intent_key,
            TemplateRefs.join_fingerprint_from_concrete_intent(tmpl.intent_signature),
            tmpl.sql_fp,
        )

    @staticmethod
    def merge_seed_warmup_templates_into_store(
        templates: dict[str, Template],
        new_templates: Sequence[Template],
    ) -> None:
        """Merge seed-warmup-produced templates into *templates* by intent/join/sql fingerprint key."""
        for tmpl in new_templates:
            dedupe_key = TemplateRefs.warmup_template_store_dedupe_key(tmpl)
            found = False
            for existing in templates.values():
                if TemplateRefs.warmup_template_store_dedupe_key(existing) == dedupe_key:
                    found = True
                    for i, q in enumerate(tmpl.value_history.questions):
                        pv = tmpl.value_history.param_values[i] if i < len(tmpl.value_history.param_values) else {}
                        nl = (
                            tmpl.value_history.natural_language[i]
                            if i < len(tmpl.value_history.natural_language)
                            else ""
                        )
                        existing.value_history.add(pv, q, nl)
                    break
            if not found:
                templates[tmpl.id] = tmpl


@dataclass(frozen=True, slots=True)
class _ReconcileReport:
    """Summary of a template-store reconciliation pass."""

    kept_template_ids: tuple[str, ...]
    dropped_template_ids: tuple[str, ...]
    kept_rejected_ids: tuple[str, ...]
    dropped_rejected_ids: tuple[str, ...]
    dropped_negative_memory_bucket_count: int
    dropped_failure_log_rows: int
    dropped_warmup_units: int
    reason_histogram: Mapping[str, int]


@dataclass(slots=True)
class SpaceLearningMergeCounts:
    """Disposition tallies for one space→master learning merge."""

    merged_same_identity: int = 0
    discarded_same_q_diff_join: int = 0
    discarded_same_q_diff_intent: int = 0
    folded_paraphrase: int = 0
    carried_new_id: int = 0
    id_reassigned: int = 0
    dropped_federation_plan_only: int = 0
    dropped_pending_approval: int = 0
    dropped_structural_hash_mismatch: int = 0
    dropped_entity_absent: int = 0
    discarded_template_master_rejection: int = 0
    feedback_carried: int = 0
    feedback_discarded_master_wins: int = 0
    feedback_join_rejection_dropped: int = 0
    paraphrase_mapping_dropped: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return only non-zero disposition counts."""
        return {k: int(v) for k, v in asdict(self).items() if int(v)}


@dataclass(frozen=True, slots=True)
class SpaceLearningMergeResult:
    """Outcome of promoting one space learning partition into master."""

    changed: bool
    counts: SpaceLearningMergeCounts

    def __bool__(self) -> bool:
        return self.changed


@dataclass(frozen=True, slots=True)
class ParamSlotMeta:
    """Internal predicate metadata for one bind handle."""

    handle: str
    column_expr: str
    op: str
    value_type: str
    upper_handle: str
    unit_handle: str


class _TemplateBodiesView(MutableMapping[str, dict[str, Any]]):
    """Mutable mapping of template id → serialised template dict backed by :class:`TemplateStoreView`."""

    __slots__ = "_view"

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
        for tid in sorted(self._view.partition_map):
            if self._view.get_template_raw(tid) is not None:
                yield tid

    def __len__(self) -> int:
        return len(self._view.partition_map)

    def __contains__(self, tid: object) -> bool:
        return isinstance(tid, str) and tid in self._view.partition_map


class LazyTemplateMapping(MutableMapping[str, Template]):
    """Lazy ``Mapping[str, Template]`` backed by a :class:`TemplateStoreView`."""

    __slots__ = ("_view",)

    def __init__(self, view: TemplateStoreView) -> None:
        self._view = view

    def __getitem__(self, tid: str) -> Template:
        t = self._view.get_template(str(tid))
        if t is None:
            raise KeyError(tid)
        return t

    def __setitem__(self, tid: str, value: Template) -> None:
        self._view.set_template_raw_dict(
            str(tid), cast(dict[str, Any], TemplateStoreLifecycleOps._convert_to_json_serializable(value.to_dict()))
        )

    def __delitem__(self, tid: str) -> None:
        tid_s = str(tid)
        if tid_s not in self._view.partition_map:
            raise KeyError(tid)
        self._view.remove_template_id(tid_s)

    def __iter__(self) -> Iterator[str]:
        return iter(self._view.partition_map)

    def __len__(self) -> int:
        return len(self._view.partition_map)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._view.partition_map

    def get(self, key: str, default: Any = None) -> Template | Any:
        t = self._view.get_template(str(key))
        return t if t is not None else default

    def keys(self) -> Any:
        return self._view.partition_map.keys()


class _QuestionFeedbackView:
    """Lazy shard-backed mapping for per-question feedback rows."""

    __slots__ = ("_view",)

    def __init__(self, view: TemplateStoreView) -> None:
        self._view = view

    def get(self, key: str, default: Any = None) -> Any:
        rows = self._view._get_feedback_rows(str(key))
        return rows if rows else default

    def __getitem__(self, key: str) -> list[dict[str, Any]]:
        rows = self._view._get_feedback_rows(str(key))
        if not rows:
            raise KeyError(key)
        return rows

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        qk = str(key)
        if qk in self._view.feedback_shard_index:
            rows = self._view._get_feedback_rows(qk)
            return bool(rows)
        return False

    def setdefault(self, key: str, default: Any) -> list[dict[str, Any]]:
        qk = str(key)
        self._view._ensure_feedback_shard_index(qk)
        rows = self._view._get_feedback_rows_mut(qk)
        return rows

    def keys(self) -> Any:
        return self._view.feedback_shard_index.keys()

    def items(self) -> Any:
        for qk in self._view.feedback_shard_index:
            rows = self._view._get_feedback_rows(qk)
            if rows:
                yield qk, rows

    def __iter__(self) -> Any:
        return iter(self._view.feedback_shard_index)

    def pop(self, key: str, default: Any = None) -> Any:
        qk = str(key)
        if qk not in self._view.feedback_shard_index:
            return default
        rows = self._view._get_feedback_rows_mut(qk)
        self._view._remove_feedback_question_key(qk)
        return rows if rows else default

    def __delitem__(self, key: str) -> None:
        self._view._remove_feedback_question_key(str(key))

    def clear(self) -> None:
        self._view._clear_question_feedback()


class TemplateStoreView:
    """Header-backed template store with lazy partition loads and bounded in-memory partitions. Match indexes live in memory (eager from header). Full template payloads live in ``partition_<NN>.json.gz`` shards and are loaded on demand."""

    __slots__ = (
        "_dirty_feedback_partitions",
        "_dirty_partitions",
        "_feedback_partition_cache",
        "_feedback_partition_cache_lock",
        "_indexes",
        "_lru_max",
        "_partition_cache",
        "_partition_cache_lock",
        "_question_feedback_proxy",
        "_store_dir",
        "_templates_proxy",
        "feedback_shard_index",
        "schema_graph_id",
        "next_id",
        "partition_map",
    )

    def __init__(
        self,
        store_dir: str,
        schema_graph_id: str,
        next_id: int,
        feedback_shard_index: dict[str, int],
        partition_map: dict[str, int],
        *,
        indexes: dict[str, Any] | None = None,
    ) -> None:
        self._store_dir = store_dir
        self.schema_graph_id = schema_graph_id
        self.next_id = int(next_id)
        self.feedback_shard_index = feedback_shard_index
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
        self._partition_cache_lock = threading.RLock()
        self._dirty_partitions: set[int] = set()
        self._feedback_partition_cache: OrderedDict[int, dict[str, list[dict[str, Any]]]] = OrderedDict()
        self._feedback_partition_cache_lock = threading.RLock()
        self._dirty_feedback_partitions: set[int] = set()
        self._lru_max = int(TEMPLATE_STORE_PARTITION_LRU_MAX)
        self._templates_proxy = _TemplateBodiesView(self)
        self._question_feedback_proxy = _QuestionFeedbackView(self)

    @property
    def question_feedback(self) -> _QuestionFeedbackView:
        return self._question_feedback_proxy

    @classmethod
    def empty(cls, store_dir: str, schema_graph_id: str) -> TemplateStoreView:
        return cls(store_dir, schema_graph_id, 1, {}, {})

    @classmethod
    def from_header_payload(cls, store_dir: str, header: dict[str, Any]) -> TemplateStoreView:
        """Build a view from a decoded header document (no template bodies)."""
        h = dict(header)
        h.pop("templates", None)
        TemplateStoreLifecycleOps._normalize_loaded_template_store_document(h)
        pm_raw = h.get("partition_map") or {}
        partition_map: dict[str, int] = {}
        if isinstance(pm_raw, dict):
            for tid, pv in pm_raw.items():
                if isinstance(pv, (int, float, str)):
                    try:
                        partition_map[str(tid)] = int(pv)
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
        fi_raw = h.get(FEEDBACK_SHARD_INDEX_KEY) or {}
        feedback_shard_index: dict[str, int] = {}
        if isinstance(fi_raw, dict):
            for qk, pv in fi_raw.items():
                if isinstance(pv, (int, float, str)):
                    try:
                        feedback_shard_index[str(qk)] = int(pv)
                    except (TypeError, ValueError):
                        continue
        graph_id = str(h.get("schema_graph_id", h.get("effective_structural_hash", "")) or "")
        return cls(store_dir, graph_id, next_id, feedback_shard_index, partition_map, indexes=indexes)

    def dirty_partitions(self) -> set[int]:
        return set(self._dirty_partitions)

    def _partition_file_path(self, part: int) -> str:
        return os.path.join(self._store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz")

    def _feedback_dir(self) -> str:
        return os.path.join(self._store_dir, TEMPLATE_STORE_FEEDBACK_SEGMENT)

    def _feedback_partition_file_path(self, part: int) -> str:
        return os.path.join(
            self._feedback_dir(),
            f"{TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX}{part:02x}.json.gz",
        )

    def _flush_feedback_partition_to_disk(self, part: int, payload: dict[str, list[dict[str, Any]]]) -> None:
        path = self._feedback_partition_file_path(part)
        if not payload:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serial = TemplateStoreLifecycleOps._convert_to_json_serializable(dict(payload))
        write_gzip_json_atomic(path, serial, sort_keys=True)

    def _evict_feedback_partition_if_needed(self) -> None:
        while len(self._feedback_partition_cache) >= self._lru_max and self._feedback_partition_cache:
            victim, victim_payload = self._feedback_partition_cache.popitem(last=False)
            if victim in self._dirty_feedback_partitions:
                self._flush_feedback_partition_to_disk(victim, victim_payload)
                self._dirty_feedback_partitions.discard(victim)

    def _load_feedback_partition_payload(self, part: int) -> dict[str, list[dict[str, Any]]]:
        with self._feedback_partition_cache_lock:
            if part in self._feedback_partition_cache:
                self._feedback_partition_cache.move_to_end(part)
                return self._feedback_partition_cache[part]
            self._evict_feedback_partition_if_needed()
            path = self._feedback_partition_file_path(part)
            if os.path.isfile(path):
                try:
                    raw = read_gzip_json(path)
                except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                    raw = {}
            else:
                raw = {}
            payload: dict[str, list[dict[str, Any]]] = {}
            if isinstance(raw, dict):
                for qk, rows in raw.items():
                    if isinstance(rows, list):
                        payload[str(qk)] = [r for r in rows if isinstance(r, dict)]
            self._feedback_partition_cache[part] = payload
            return payload

    def _ensure_feedback_shard_index(self, q_norm: str) -> int:
        part = self.question_feedback_partition_number(q_norm)
        self.feedback_shard_index[q_norm] = part
        return part

    def _get_feedback_rows(self, q_norm: str) -> list[dict[str, Any]]:
        part = self.feedback_shard_index.get(q_norm)
        if part is None:
            return []
        payload = self._load_feedback_partition_payload(int(part))
        rows = payload.get(q_norm)
        if not isinstance(rows, list):
            return []
        return [r for r in rows if isinstance(r, dict)]

    def _get_feedback_rows_mut(self, q_norm: str) -> list[dict[str, Any]]:
        part = self._ensure_feedback_shard_index(q_norm)
        with self._feedback_partition_cache_lock:
            payload = self._load_feedback_partition_payload(part)
            rows = payload.get(q_norm)
            if not isinstance(rows, list):
                rows = []
                payload[q_norm] = rows
            self._dirty_feedback_partitions.add(part)
            return rows

    def _remove_feedback_question_key(self, q_norm: str) -> None:
        part = self.feedback_shard_index.pop(q_norm, None)
        if part is None:
            return
        p_int = int(part)
        with self._feedback_partition_cache_lock:
            if p_int in self._feedback_partition_cache:
                self._feedback_partition_cache[p_int].pop(q_norm, None)
            else:
                disk = self._load_feedback_partition_payload(p_int)
                disk.pop(q_norm, None)
            self._dirty_feedback_partitions.add(p_int)

    def _clear_question_feedback(self) -> None:
        with self._feedback_partition_cache_lock:
            self.feedback_shard_index.clear()
            self._feedback_partition_cache.clear()
            for part in list(self._dirty_feedback_partitions):
                self._dirty_feedback_partitions.discard(part)
            for part in range(TEMPLATE_STORE_PARTITION_COUNT):
                path = self._feedback_partition_file_path(part)
                if os.path.isfile(path):
                    self._dirty_feedback_partitions.add(part)
                    self._feedback_partition_cache[part] = {}

    def _import_question_feedback_rows(self, rows_by_question: Mapping[str, Any]) -> None:
        for qk, rows in rows_by_question.items():
            if not isinstance(rows, list):
                continue
            q_norm = str(qk)
            mut_rows = self._get_feedback_rows_mut(q_norm)
            for row in rows:
                if isinstance(row, dict):
                    mut_rows.append(dict(row))

    def _iter_question_feedback_items(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for q_norm in self.feedback_shard_index:
            rows = self._get_feedback_rows(q_norm)
            if rows:
                yield q_norm, rows

    def _count_question_feedback_rows(self) -> int:
        total = 0
        for _qk, rows in self._iter_question_feedback_items():
            total += len(rows)
        return total

    def dirty_feedback_partitions(self) -> set[int]:
        return set(self._dirty_feedback_partitions)

    def _flush_partition_to_disk(self, part: int, payload: dict[str, dict[str, Any]]) -> None:
        path = self._partition_file_path(part)
        if not payload:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return
        serial = TemplateStoreLifecycleOps._convert_to_json_serializable(dict(payload))
        write_gzip_json_atomic(path, serial, sort_keys=True)

    def _evict_partition_if_needed(self) -> None:
        while len(self._partition_cache) >= self._lru_max:
            for victim in list(self._partition_cache):
                if victim in self._dirty_partitions:
                    continue
                del self._partition_cache[victim]
                break
            else:
                break

    def _load_partition_payload(self, part: int) -> dict[str, dict[str, Any]]:
        with self._partition_cache_lock:
            if part in self._partition_cache:
                self._partition_cache.move_to_end(part)
                return self._partition_cache[part]
            self._evict_partition_if_needed()
            path = self._partition_file_path(part)
            if os.path.isfile(path):
                try:
                    raw = read_gzip_json(path)
                except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
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
        return TemplateStoreView._template_from_store_dict(str(template_id), raw)

    def set_template_raw_dict(self, template_id: str, raw: dict[str, Any]) -> None:
        with self._partition_cache_lock:
            tid = str(template_id)
            part = TemplateStoreView.template_partition_number(tid)
            payload = self._load_partition_payload(part)
            payload[tid] = raw
            self._dirty_partitions.add(part)
            self.partition_map[tid] = part

    def remove_template_id(self, template_id: str) -> None:
        with self._partition_cache_lock:
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

    def _release_partition_if_clean(self, part: int) -> None:
        with self._partition_cache_lock:
            if part in self._dirty_partitions:
                return
            self._partition_cache.pop(part, None)

    def iter_templates_by_partition(self) -> Iterator[list[Template]]:
        """Yield parsed template rows one partition at a time, releasing each shard after consumption."""
        by_part: dict[int, list[str]] = defaultdict(list)
        for tid, part in self.partition_map.items():
            by_part[int(part)].append(str(tid))
        for part in sorted(by_part):
            payload = self._load_partition_payload(part)
            batch: list[Template] = []
            for tid in by_part[part]:
                raw = payload.get(tid)
                if not isinstance(raw, dict):
                    continue
                t = TemplateStoreView._template_from_store_dict(tid, raw)
                if t is not None:
                    batch.append(t)
                else:
                    self.remove_template_id(tid)
            yield batch
            self._release_partition_if_clean(part)

    def _bulk_replace_templates_from_mapping(self, mapping: Mapping[str, Any]) -> None:
        with self._partition_cache_lock:
            self.partition_map.clear()
            self._partition_cache.clear()
            self._dirty_partitions.clear()
            for tid, raw in mapping.items():
                if not isinstance(raw, dict):
                    continue
                self.set_template_raw_dict(str(tid), dict(raw))
        active_parts = {int(v) for v in self.partition_map.values()}
        for part in range(TEMPLATE_STORE_PARTITION_COUNT):
            if part in active_parts:
                continue
            path = self._partition_file_path(part)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _header_document(self) -> dict[str, Any]:
        return {
            "format_version": TEMPLATE_STORE_FORMAT_VERSION,
            QUESTION_NORMALIZATION_VERSION_KEY: QUESTION_NORMALIZATION_VERSION,
            "schema_graph_id": self.schema_graph_id,
            "next_id": int(self.next_id),
            FEEDBACK_SHARD_INDEX_KEY: dict(sorted(self.feedback_shard_index.items())),
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
            "schema_graph_id",
            "next_id",
            "question_feedback",
            FEEDBACK_SHARD_INDEX_KEY,
            QUESTION_NORMALIZATION_VERSION_KEY,
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
        if key == "schema_graph_id":
            return self.schema_graph_id
        if key == "next_id":
            return self.next_id
        if key == "question_feedback":
            return self.question_feedback
        if key == FEEDBACK_SHARD_INDEX_KEY:
            return self.feedback_shard_index
        if key == QUESTION_NORMALIZATION_VERSION_KEY:
            return QUESTION_NORMALIZATION_VERSION
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
        if key == "schema_graph_id":
            self.schema_graph_id = str(value)
            return
        if key == "next_id":
            self.next_id = int(value)
            return
        if key == "question_feedback":
            if not isinstance(value, Mapping):
                raise TypeError("question_feedback must be a mapping")
            self._clear_question_feedback()
            for qk, rows in value.items():
                if not isinstance(rows, list):
                    continue
                mut_rows = self._get_feedback_rows_mut(str(qk))
                for row in rows:
                    if isinstance(row, dict):
                        mut_rows.append(dict(row))
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
        if key == FEEDBACK_SHARD_INDEX_KEY:
            return self.feedback_shard_index
        if key in self._indexes:
            cur = self._indexes[key]
            if cur is None or (isinstance(cur, dict) and not cur and default is not None):
                self._indexes[key] = default if isinstance(default, dict) else {}
            return self._indexes[key]
        raise KeyError(key)

    def __deepcopy__(self, memo: dict[int, Any]) -> TemplateStoreView:
        other = TemplateStoreView.__new__(TemplateStoreView)
        other._store_dir = self._store_dir
        other.schema_graph_id = self.schema_graph_id
        other.next_id = self.next_id
        other.feedback_shard_index = dict(self.feedback_shard_index)
        other.partition_map = dict(self.partition_map)
        other._indexes = {k: copy.deepcopy(v, memo) for k, v in self._indexes.items()}
        other._partition_cache = OrderedDict((k, copy.deepcopy(v, memo)) for k, v in self._partition_cache.items())
        other._partition_cache_lock = threading.RLock()
        other._dirty_partitions = set()
        other._feedback_partition_cache = OrderedDict(
            (k, copy.deepcopy(v, memo)) for k, v in self._feedback_partition_cache.items()
        )
        other._feedback_partition_cache_lock = threading.RLock()
        other._dirty_feedback_partitions = set()
        other._lru_max = self._lru_max
        other._templates_proxy = _TemplateBodiesView(other)
        other._question_feedback_proxy = _QuestionFeedbackView(other)
        memo[id(self)] = other
        return other

    @staticmethod
    def _build_intent_key_index_for_templates(templates: list[Template]) -> dict[str, list[str]]:
        idx: dict[str, set[str]] = defaultdict(set)
        for t in templates:
            ik = (t.intent_key or "").strip()
            if ik:
                idx[ik].add(str(t.id))
        return {k: sorted(v) for k, v in idx.items()}

    @staticmethod
    def _build_union_family_index_for_templates(templates: list[Template]) -> dict[str, list[str]]:
        idx: dict[str, set[str]] = defaultdict(set)
        for t in templates:
            bk = body_similarity_key_for_concrete(t.intent_signature)
            jk = join_path_key_concrete(t.intent_signature)
            idx[bk].add(str(t.id))
            idx[f"{bk}|{jk}"].add(str(t.id))
        return {k: sorted(v) for k, v in idx.items()}

    @staticmethod
    def _build_question_token_index_for_templates(templates: list[Template]) -> dict[str, list[list[str]]]:
        idx: dict[str, list[list[str]]] = defaultdict(list)
        for t in templates:
            tid = str(t.id)
            for hi, q in enumerate(t.value_history.questions or []):
                if not q:
                    continue
                fp = question_token_fingerprint_from_raw(q)
                idx[fp].append([tid, str(hi)])
        return dict(idx)

    @staticmethod
    def sqlglot_dialect_for_template_fingerprint(
        dialect: Any | None,
        member_source_id: str | None,
        *,
        member_engine: str | None = None,
    ) -> str:
        """Fingerprint member templates under the member engine's sqlglot dialect."""
        engine_name = str(member_engine or "").strip()
        if not engine_name and member_source_id and dialect is not None:
            engine = getattr(dialect, "engine", None)
            if engine is None:
                cfg = getattr(dialect, "config", None)
                engine = getattr(cfg, "TYPE", None) if cfg is not None else None
            if engine:
                engine_name = str(engine)
        if engine_name:
            return DialectRegistry.sqlglot_dialect_for_engine(engine_name)
        try:
            return Dialect.active_sqlglot_dialect()
        except RuntimeError:
            return DialectRegistry.sqlglot_dialect_for_engine("duckdb")

    @staticmethod
    def _template_from_store_dict(template_id: str, raw: dict[str, Any]) -> Template | None:
        try:
            t = Template.from_dict({**raw, "id": str(template_id)})
            member_source_id = str(getattr(t, "member_source_id", "") or "") or None
            member_engine = str(raw.get("member_engine") or getattr(t, "member_engine", "") or "") or None
            t.sql_fp = Dialect.compute_sql_fp(
                t.sql_param or "",
                sqlglot_dialect=TemplateStoreView.sqlglot_dialect_for_template_fingerprint(
                    None, member_source_id, member_engine=member_engine
                ),
            )
            return t
        except Exception as exc:
            debug(f"[templates] corrupt template row id={template_id!r}: {exc!r}")
            return None

    @staticmethod
    def template_partition_number(template_id: str) -> int:
        """Return stable partition index ``0..255`` for *template_id* (SHA-256 first byte)."""
        return int(hashlib.sha256(template_id.encode("utf-8")).hexdigest()[:2], 16)

    @staticmethod
    def question_feedback_partition_number(q_norm: str) -> int:
        """Return stable partition index ``0..255`` for a normalised question (SHA-256 first byte)."""
        return int(hashlib.sha256(q_norm.encode("utf-8")).hexdigest()[:2], 16)

    _TEMPLATES_MODULE: Any = None

    @staticmethod
    def register_templates_module(module: Any) -> None:
        TemplateStoreView._TEMPLATES_MODULE = module

    @staticmethod
    def get_templates_module() -> Any:
        """Return the module registered via :meth:`register_templates_module`."""
        return TemplateStoreView._TEMPLATES_MODULE

    @staticmethod
    def is_template_store_view(store: Any) -> bool:
        """Return True when *store* is a :class:`TemplateStoreView`."""
        return isinstance(store, TemplateStoreView)

    @staticmethod
    def refresh_template_store_indexes_for_view(store: Any, *, template_objs: list[Template] | None = None) -> None:
        """Refresh matcher indexes on *store* when it is a :class:`TemplateStoreView`."""
        if not isinstance(store, TemplateStoreView):
            return
        TemplateStoreView.refresh_template_store_indexes(store, template_objs=template_objs)

    @staticmethod
    def refresh_template_store_indexes(
        store: dict[str, Any] | TemplateStoreView, *, template_objs: list[Template] | None = None
    ) -> None:
        """Recompute shape and inverted template indexes on *store* in place. When *template_objs* is provided (already-materialised :class:`Template` rows), avoids a round-trip through ``store['templates']`` dict serialisation."""
        if isinstance(store, TemplateStoreView):
            if template_objs is None:
                tpl_objs = [t for batch in store.iter_templates_by_partition() for t in batch]
            else:
                tpl_objs = template_objs
            store._indexes[SHAPE_QUESTION_INDEX_KEY] = build_shape_question_index(tpl_objs)
            store._indexes[TEMPLATE_INTENT_KEY_INDEX_KEY] = TemplateStoreView._build_intent_key_index_for_templates(
                tpl_objs
            )
            store._indexes[TEMPLATE_UNION_FAMILY_INDEX_KEY] = TemplateStoreView._build_union_family_index_for_templates(
                tpl_objs
            )
            store._indexes[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = (
                TemplateStoreView._build_question_token_index_for_templates(tpl_objs)
            )
            return

        if template_objs is None:
            tpl_objs = []
            for tid, raw in (store.get("templates") or {}).items():
                if isinstance(raw, dict):
                    t = TemplateStoreView._template_from_store_dict(str(tid), raw)
                    if t is not None:
                        tpl_objs.append(t)
        else:
            tpl_objs = template_objs
        store[SHAPE_QUESTION_INDEX_KEY] = build_shape_question_index(tpl_objs)
        store[TEMPLATE_INTENT_KEY_INDEX_KEY] = TemplateStoreView._build_intent_key_index_for_templates(tpl_objs)
        store[TEMPLATE_UNION_FAMILY_INDEX_KEY] = TemplateStoreView._build_union_family_index_for_templates(tpl_objs)
        store[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = TemplateStoreView._build_question_token_index_for_templates(tpl_objs)


SANDBOX_PARAPHRASE_SOURCE: dict[str, list[str]] | None = None


class TemplateStoreLifecycleOps:
    """Template store shard I/O, reconcile, and structural migration."""

    _manifest_reader: ClassVar[Callable[[str], Any] | None] = None
    _reconcile_template_store_until_stable: ClassVar[Callable[..., int] | None] = None

    @staticmethod
    def register_reconcile_template_store_until_stable(fn: Callable[..., int]) -> None:
        TemplateStoreLifecycleOps._reconcile_template_store_until_stable = fn

    @staticmethod
    def _invoke_reconcile_template_store_until_stable(
        templates: dict[str, Template],
        *,
        max_iterations: int = 16,
        template_store_view: Any | None = None,
    ) -> int:
        hook = TemplateStoreLifecycleOps._reconcile_template_store_until_stable
        if hook is None:
            raise RuntimeError("reconcile_template_store_until_stable hook not registered")
        return hook(templates, max_iterations=max_iterations, template_store_view=template_store_view)

    @staticmethod
    def read_store_manifest(artifacts_dir: str) -> Any:
        """Read artifact manifest, honoring an optional federation override."""
        reader = TemplateStoreLifecycleOps._manifest_reader
        if reader is not None:
            return reader(artifacts_dir)
        return read_artifact_manifest(artifacts_dir)

    @staticmethod
    def _template_id_numeric(tid: str) -> int | None:
        raw = str(tid or "")
        if len(raw) >= 2 and raw[0] == "T" and raw[1:].isdigit():
            return int(raw[1:])
        return None

    @staticmethod
    def _global_next_id_path(artifacts_dir: str) -> str:
        return os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_GLOBAL_NEXT_ID_FILENAME
        )

    @staticmethod
    def _ceiling_next_id_from_space_headers(artifacts_dir: str) -> int:
        """Return the max per-space ``next_id`` watermark (at least 1)."""
        high = 1
        spaces_root = os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT
        )
        if not os.path.isdir(spaces_root):
            return high
        for name in os.listdir(spaces_root):
            header_path = os.path.join(spaces_root, name, TEMPLATE_STORE_HEADER_FILENAME)
            if not os.path.isfile(header_path):
                continue
            try:
                header = read_gzip_json(header_path)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if not isinstance(header, dict):
                continue
            try:
                high = max(high, int(header.get("next_id", 1) or 1))
            except (TypeError, ValueError):
                continue
            partition_map = header.get("partition_map")
            if isinstance(partition_map, dict):
                for tid in partition_map:
                    num = TemplateStoreLifecycleOps._template_id_numeric(str(tid))
                    if num is not None:
                        high = max(high, num + 1)
        return high

    @staticmethod
    def _allocate_space_template_id(store: TemplateStoreView) -> tuple[str, int]:
        """Allocate the next ``T####`` id from *store*'s per-space counter."""
        store_dir = store._store_dir
        adir = TemplateStoreLifecycleOps.artifacts_dir_for_template_store(store_dir)
        with artifact_lock(adir):
            current = int(store.next_id or 1)
            header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
            if os.path.isfile(header_path):
                try:
                    header = read_gzip_json(header_path)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    header = None
                if isinstance(header, dict):
                    try:
                        current = max(current, int(header.get("next_id", 1) or 1))
                    except (TypeError, ValueError):
                        pass
                    partition_map = header.get("partition_map")
                    if isinstance(partition_map, dict):
                        for tid in partition_map:
                            num = TemplateStoreLifecycleOps._template_id_numeric(str(tid))
                            if num is not None:
                                current = max(current, num + 1)
            tid = f"T{int(current):04d}"
            nxt = int(current) + 1
            store.next_id = nxt
            return tid, nxt

    @staticmethod
    def _reserve_template_id(store: dict[str, Any] | TemplateStoreView) -> str:
        if isinstance(store, TemplateStoreView):
            adir = TemplateStoreLifecycleOps.artifacts_dir_for_template_store(store._store_dir)
            base = TemplateStoreLifecycleOps.template_store_base_dir(adir)
            if os.path.isdir(base) or os.path.isdir(adir):
                tid, _nxt = TemplateStoreLifecycleOps._allocate_space_template_id(store)
                return tid
            tid = f"T{int(store.next_id):04d}"
            store.next_id = int(store.next_id) + 1
            return tid
        next_id = int(store.get("next_id", 1) or 1)
        tid = f"T{next_id:04d}"
        store["next_id"] = next_id + 1
        return tid

    @staticmethod
    def _prune_template_value_history(vh: ValueHistory, *, limits: EngineLimits | None = None) -> None:
        """Trim ``value_history`` rows to the configured depth."""
        cap = (limits or TemplateStoreLifecycleOps._resolve_engine_limits()).template_value_history_depth
        overflow = len(vh.questions) - cap
        if overflow <= 0:
            return
        vh.questions = vh.questions[overflow:]
        vh.param_values = vh.param_values[overflow:]
        vh.natural_language = vh.natural_language[overflow:]
        vh.accept_counts = vh.accept_counts[overflow:]

    @staticmethod
    def _schema_column_types_for_runtime_intent(intent: RuntimeIntent, schema: SchemaGraph) -> dict[str, str]:
        """Snapshot ``table.column -> data_type`` for every column referenced by *intent*."""
        refs = list(collect_column_refs_for_post_processing(intent))
        for step in intent.cte_steps or []:
            refs.extend(collect_column_refs_for_cte_step(step))
        out: dict[str, str] = {}
        for ref in refs:
            if "." not in ref:
                continue
            tbl, col = ref.split(".", 1)
            tm = schema.tables.get(tbl)
            if tm is not None and col in tm.columns:
                out[f"{tbl}.{col}"] = str(tm.columns[col].data_type or "")
        for bare, tbl in (intent.column_map or {}).items():
            tm = schema.tables.get(tbl)
            if tm is not None and bare in tm.columns:
                out[f"{tbl}.{bare}"] = str(tm.columns[bare].data_type or "")
        return out

    @staticmethod
    def _template_store_disk_bytes(store_dir: str) -> int:
        total = 0
        if not os.path.isdir(store_dir):
            return 0
        for entry in os.listdir(store_dir):
            if entry == ".write_staging":
                continue
            path = os.path.join(store_dir, entry)
            if os.path.isfile(path):
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
        return total

    @staticmethod
    def _resolve_engine_limits() -> EngineLimits:
        try:
            return active_engine_limits()
        except RuntimeError:
            return EngineLimits()

    @staticmethod
    def _prune_template_store_size(view: TemplateStoreView) -> bool:
        """Enforce template-count, disk-size, and per-template value- history caps."""
        limits = TemplateStoreLifecycleOps._resolve_engine_limits()
        changed = False
        for batch in view.iter_templates_by_partition():
            for tmpl in batch:
                before = len(tmpl.value_history.questions)
                TemplateStoreLifecycleOps._prune_template_value_history(tmpl.value_history, limits=limits)
                if len(tmpl.value_history.questions) != before:
                    view.set_template_raw_dict(tmpl.id, tmpl.to_dict())
                    changed = True

        cap = limits.template_store_max_count
        if cap is not None:
            while len(view.partition_map) > cap:
                scored: list[tuple[int, str]] = []
                for batch in view.iter_templates_by_partition():
                    for tmpl in batch:
                        scored.append((tmpl.trust_level, str(tmpl.id)))
                scored.sort(key=lambda row: (row[0], row[1]))
                view.remove_template_id(scored[0][1])
                changed = True

        disk_cap = limits.template_store_max_disk_bytes
        if disk_cap is not None:
            while (
                TemplateStoreLifecycleOps._template_store_disk_bytes(view._store_dir) > disk_cap and view.partition_map
            ):
                scored = []
                for batch in view.iter_templates_by_partition():
                    for tmpl in batch:
                        scored.append((tmpl.trust_level, str(tmpl.id)))
                scored.sort(key=lambda row: (row[0], row[1]))
                view.remove_template_id(scored[0][1])
                changed = True

        if changed:
            TemplateStoreView.refresh_template_store_indexes(view)
        return changed

    @staticmethod
    def _repair_cross_shard_inconsistency(view: TemplateStoreView) -> bool:
        """Drop header ``partition_map`` rows whose shard body is missing and uncommitted shard bodies."""
        changed = TemplateStoreLifecycleOps._prune_orphan_template_partition_map(view)
        map_ids = set(view.partition_map.keys())
        for part in range(TEMPLATE_STORE_PARTITION_COUNT):
            path = view._partition_file_path(part)
            if not os.path.isfile(path):
                continue
            with view._partition_cache_lock:
                if part in view._partition_cache:
                    payload = view._partition_cache[part]
                else:
                    try:
                        raw = read_gzip_json(path)
                    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    payload = raw if isinstance(raw, dict) else {}
            stray = [tid for tid in payload if tid not in map_ids]
            if not stray:
                continue
            with view._partition_cache_lock:
                if part not in view._partition_cache:
                    view._partition_cache[part] = dict(payload)
                    payload = view._partition_cache[part]
                for tid in stray:
                    payload.pop(tid, None)
                view._dirty_partitions.add(part)
            changed = True
        if changed:
            TemplateStoreView.refresh_template_store_indexes(view)
        return changed

    @staticmethod
    def _manifest_fingerprints_for_template_save(adir: str, schema_graph_id: str) -> dict[str, str]:
        """Resolve manifest fingerprints from the live schema snapshot when ids match."""
        prev = TemplateStoreLifecycleOps.read_store_manifest(adir)
        out = {
            "structural_hash": prev.structural_hash if prev else "",
            "profiling_hash": prev.profiling_hash if prev else "",
            "scope_hash": prev.scope_hash if prev else "",
            "effective_structural_hash": prev.effective_structural_hash if prev else "",
            "schema_graph_id": schema_graph_id or (prev.schema_graph_id if prev else ""),
            "notes_hash": prev.notes_hash if prev else "",
            "semantic_edges_hash": prev.semantic_edges_hash if prev else "",
            "last_migration_tier": prev.last_migration_tier if prev else "",
            "last_migration_at": prev.last_migration_at if prev else "",
        }
        snap = load_schema_graph_snapshot(EngineConfig.SCHEMA_JSON_PATH)
        if snap is not None and str(snap.schema_graph_id or "") == str(schema_graph_id or ""):
            out.update(
                {
                    "structural_hash": snap.structural_hash,
                    "profiling_hash": snap.profiling_hash,
                    "scope_hash": snap.scope_hash,
                    "effective_structural_hash": snap.effective_structural_hash,
                    "schema_graph_id": snap.schema_graph_id,
                    "notes_hash": snap.notes_hash,
                    "semantic_edges_hash": snap.semantic_edges_hash,
                }
            )
        return out

    @staticmethod
    def _prune_orphan_template_partition_map(store: dict[str, Any] | TemplateStoreView) -> bool:
        """Drop partition_map entries whose shard body is missing and refresh indexes when needed."""
        if not isinstance(store, TemplateStoreView):
            return False
        orphans = [tid for tid in list(store.partition_map.keys()) if store.get_template_raw(tid) is None]
        if not orphans:
            return False
        for tid in orphans:
            store.partition_map.pop(tid, None)
        TemplateStoreView.refresh_template_store_indexes(store)
        return True

    @staticmethod
    def prune_orphan_template_ids(view: TemplateStoreView) -> bool:
        """Drop header ``partition_map`` rows whose shard body is missing."""
        return TemplateStoreLifecycleOps._prune_orphan_template_partition_map(view)

    @staticmethod
    def _orphan_identity_dirname(identity: str) -> str:
        safe = str(identity).strip()
        for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
            safe = safe.replace(ch, "_")
        return safe or "unknown"

    @staticmethod
    def _directory_byte_size(root: str) -> int:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
        return total

    @staticmethod
    def orphan_mismatched_template_store(
        store_dir: str,
        *,
        old_schema_graph_id: str,
        new_schema_graph_id: str,
    ) -> str:
        """Move active template shards into ``orphaned/<old_id>/`` for a superseded identity."""
        orphan_name = TemplateStoreLifecycleOps._orphan_identity_dirname(old_schema_graph_id)
        orphan_dir = os.path.join(store_dir, TEMPLATE_STORE_ORPHANED_SEGMENT, orphan_name)
        os.makedirs(orphan_dir, exist_ok=True)
        for entry in os.listdir(store_dir):
            if entry in (TEMPLATE_STORE_ORPHANED_SEGMENT,) or entry.startswith("."):
                continue
            src = os.path.join(store_dir, entry)
            dst = os.path.join(orphan_dir, entry)
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            elif os.path.isfile(dst):
                os.remove(dst)
            shutil.move(src, dst)
        notify(
            "Orphaned template store shards after schema graph identity mismatch",
            stage="artifact",
            code=DIAGNOSTIC_CODE_TEMPLATE_STORE_ORPHANED,
            details=(
                ("old_schema_graph_id", old_schema_graph_id),
                ("new_schema_graph_id", new_schema_graph_id),
                ("orphan_dir", orphan_dir),
            ),
        )
        return orphan_dir

    @staticmethod
    def orphan_superseded_identity_artifacts(
        artifacts_dir: str,
        *,
        previous_schema_graph_id: str,
        active_schema_graph_id: str,
    ) -> list[str]:
        """Move artifacts keyed to *previous_schema_graph_id* into ``orphaned/<old_id>/``."""
        previous = str(previous_schema_graph_id or "").strip()
        active = str(active_schema_graph_id or "").strip()
        if not previous or not active or previous == active:
            return []
        collected: list[str] = []
        orphan_name = TemplateStoreLifecycleOps._orphan_identity_dirname(previous)
        orphan_base = os.path.join(artifacts_dir, TEMPLATE_STORE_ORPHANED_SEGMENT, orphan_name)
        os.makedirs(orphan_base, exist_ok=True)

        spaces_root = os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT
        )
        if os.path.isdir(spaces_root):
            for space_name in os.listdir(spaces_root):
                store_dir = os.path.join(spaces_root, space_name)
                header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
                if not os.path.isfile(header_path):
                    continue
                try:
                    hdr = read_gzip_json(header_path)
                except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                    hdr = {}
                if str((hdr or {}).get("schema_graph_id", "") or "") != previous:
                    continue
                TemplateStoreLifecycleOps.orphan_mismatched_template_store(
                    store_dir,
                    old_schema_graph_id=previous,
                    new_schema_graph_id=active,
                )
                for kind in ("template_shards", "feedback_shards", "join_feedback"):
                    if kind not in collected:
                        collected.append(kind)

        lattice_root = os.path.join(artifacts_dir, SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR)
        if os.path.isdir(lattice_root):
            warmup_dest = os.path.join(orphan_base, SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR)
            os.makedirs(warmup_dest, exist_ok=True)
            for name in os.listdir(lattice_root):
                graph_id, _partition_fp, _code_version = split_warmup_lattice_basename(name)
                if graph_id != previous:
                    continue
                shutil.move(os.path.join(lattice_root, name), os.path.join(warmup_dest, name))
                if "warmup_lattices" not in collected:
                    collected.append("warmup_lattices")

        for skeleton_name in os.listdir(artifacts_dir):
            if not skeleton_name.startswith("qsim_skeletons") or not skeleton_name.endswith(".json.gz"):
                continue
            skeleton_path = os.path.join(artifacts_dir, skeleton_name)
            if not os.path.isfile(skeleton_path):
                continue
            try:
                skeleton_doc = read_gzip_json(skeleton_path)
            except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                skeleton_doc = {}
            if str((skeleton_doc or {}).get("schema_graph_id", "") or "") == previous:
                shutil.move(skeleton_path, os.path.join(orphan_base, skeleton_name))
                if "qsim_skeletons" not in collected:
                    collected.append("qsim_skeletons")

        context_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
        if os.path.isfile(context_path):
            try:
                with open(context_path, encoding="utf-8") as fh:
                    context_doc = json.load(fh)
            except (OSError, json.JSONDecodeError):
                context_doc = {}
            if str((context_doc or {}).get("schema_graph_id", "") or "") == previous:
                shutil.move(context_path, os.path.join(orphan_base, SCHEMA_CONTEXT_CACHE_NAME))
                if "schema_context_cache" not in collected:
                    collected.append("schema_context_cache")
                if "prompt_cache_references" not in collected:
                    collected.append("prompt_cache_references")

        manifest = read_artifact_manifest(artifacts_dir)
        if manifest is not None and str(manifest.schema_graph_id or "") == previous:
            queue_path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
            if os.path.isfile(queue_path):
                try:
                    os.remove(queue_path)
                except OSError:
                    pass
                collected.append("write_queue")

        fed_manifest_path = os.path.join(artifacts_dir, FEDERATION_MANIFEST_FILENAME)
        if os.path.isfile(fed_manifest_path):
            try:
                with open(fed_manifest_path, encoding="utf-8") as fh:
                    fed_doc = json.load(fh)
            except (OSError, json.JSONDecodeError):
                fed_doc = {}
            if str((fed_doc or {}).get("schema_graph_id", "") or "") == previous:
                fed_dest = os.path.join(orphan_base, FEDERATION_MANIFEST_FILENAME)
                shutil.move(fed_manifest_path, fed_dest)
                collected.append("federation_pins")

        return collected

    @staticmethod
    def collect_expired_template_orphans(
        artifacts_dir: str,
        *,
        retention_seconds: int = ORPHAN_RETENTION_SECONDS,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Remove orphaned template directories older than *retention_seconds*; return count and bytes."""
        removed = 0
        reclaimed = 0
        cutoff = (now if now is not None else time.time()) - retention_seconds
        spaces_root = os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT
        )
        if not os.path.isdir(spaces_root):
            return (0, 0)
        for space_name in os.listdir(spaces_root):
            orphan_root = os.path.join(spaces_root, space_name, TEMPLATE_STORE_ORPHANED_SEGMENT)
            if not os.path.isdir(orphan_root):
                continue
            for orphan_id in os.listdir(orphan_root):
                orphan_path = os.path.join(orphan_root, orphan_id)
                if not os.path.isdir(orphan_path):
                    continue
                try:
                    mtime = os.path.getmtime(orphan_path)
                except OSError:
                    continue
                if mtime > cutoff:
                    continue
                reclaimed += TemplateStoreLifecycleOps._directory_byte_size(orphan_path)
                shutil.rmtree(orphan_path, ignore_errors=True)
                removed += 1
        return (removed, reclaimed)

    @staticmethod
    def _checkpoint_manifest_matches_current(
        checkpoint_manifest: ArtifactManifest | None,
        current_manifest: ArtifactManifest | None,
    ) -> bool:
        if checkpoint_manifest is None or current_manifest is None:
            return False
        keys = (
            "structural_hash",
            "profiling_hash",
            "scope_hash",
            "effective_structural_hash",
            "schema_graph_id",
            "notes_hash",
            "semantic_edges_hash",
        )
        return all(getattr(checkpoint_manifest, key) == getattr(current_manifest, key) for key in keys)

    @staticmethod
    def _load_checkpoint_manifest(checkpoint_dir: str) -> ArtifactManifest | None:
        path = os.path.join(checkpoint_dir, ARTIFACT_MANIFEST_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            ver = coerce_format_version(data.get("artifact_format_version", "0") or "0")
        except (TypeError, ValueError):
            ver = "0"
        return ArtifactManifest(
            artifact_format_version=ver,
            created_with_package_version=str(data.get("created_with_package_version", "") or ""),
            min_compatible_package_version=str(data.get("min_compatible_package_version", "") or ""),
            last_action=str(data.get("last_action", "") or ""),
            last_action_at=str(data.get("last_action_at", "") or ""),
            structural_hash=str(data.get("structural_hash", "") or ""),
            profiling_hash=str(data.get("profiling_hash", "") or ""),
            scope_hash=str(data.get("scope_hash", "") or ""),
            effective_structural_hash=str(data.get("effective_structural_hash", "") or ""),
            schema_graph_id=str(data.get("schema_graph_id", "") or ""),
            notes_hash=str(data.get("notes_hash", "") or ""),
            semantic_edges_hash=str(data.get("semantic_edges_hash", "") or ""),
            last_migration_tier=str(data.get("last_migration_tier", "") or ""),
            last_migration_at=str(data.get("last_migration_at", "") or ""),
        )

    @staticmethod
    def collect_orphaned_migration_checkpoints(artifacts_dir: str) -> list[Diagnostic]:
        """Collect completed migration checkpoints or retain ambiguous ones with diagnostics."""
        diags: list[Diagnostic] = []
        try:
            names = os.listdir(artifacts_dir)
        except OSError:
            return diags
        current = read_artifact_manifest(artifacts_dir)
        for name in names:
            if not name.startswith(MIGRATION_CHECKPOINT_PREFIX):
                continue
            checkpoint_dir = os.path.join(artifacts_dir, name)
            if not os.path.isdir(checkpoint_dir):
                continue
            checkpoint_manifest = TemplateStoreLifecycleOps._load_checkpoint_manifest(checkpoint_dir)
            if TemplateStoreLifecycleOps._checkpoint_manifest_matches_current(checkpoint_manifest, current):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                continue
            diag = Diagnostic(
                stage="artifact",
                level=DiagnosticSeverity.WARNING,
                code=DIAGNOSTIC_CODE_MIGRATION_CHECKPOINT_ORPHANED,
                message=f"Retained incomplete migration checkpoint {checkpoint_dir}",
                details=(("checkpoint_dir", checkpoint_dir),),
                phase="artifact",
            )
            notify(
                diag.message,
                stage=diag.stage,
                code=diag.code,
                level=diag.level,
                details=diag.details,
            )
            diags.append(diag)
        return diags

    @staticmethod
    def restore_leftover_migration_checkpoints_on_init(
        artifacts_dir: str,
        *,
        schema_json_path: Path | str | None = None,
    ) -> None:
        """Restore artifacts from incomplete migration checkpoints before construction proceeds."""
        schema_path = Path(schema_json_path) if schema_json_path is not None else None
        try:
            names = os.listdir(artifacts_dir)
        except OSError:
            return
        current = read_artifact_manifest(artifacts_dir)
        for name in names:
            if not name.startswith(MIGRATION_CHECKPOINT_PREFIX):
                continue
            checkpoint_dir = os.path.join(artifacts_dir, name)
            if not os.path.isdir(checkpoint_dir):
                continue
            checkpoint_manifest = TemplateStoreLifecycleOps._load_checkpoint_manifest(checkpoint_dir)
            if TemplateStoreLifecycleOps._checkpoint_manifest_matches_current(checkpoint_manifest, current):
                TemplateStoreLifecycleOps._migration_map_checkpoint_cleanup(checkpoint_dir)
                continue
            with artifact_lock(artifacts_dir):
                TemplateStoreLifecycleOps._migration_map_checkpoint_restore(
                    artifacts_dir,
                    checkpoint_dir,
                    schema_json_path=schema_path,
                )
                TemplateStoreLifecycleOps._migration_map_checkpoint_cleanup(checkpoint_dir)

    @staticmethod
    def artifact_directory_byte_size(artifacts_dir: str) -> int:
        return TemplateStoreLifecycleOps._directory_byte_size(artifacts_dir)

    @staticmethod
    def artifact_growth_counts(artifacts_dir: str) -> tuple[int, int, int]:
        template_count = 0
        feedback_shard_count = 0
        orphan_count = 0
        spaces_root = os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT
        )
        if os.path.isdir(spaces_root):
            for space_name in os.listdir(spaces_root):
                space_dir = os.path.join(spaces_root, space_name)
                header_path = os.path.join(space_dir, TEMPLATE_STORE_HEADER_FILENAME)
                if os.path.isfile(header_path):
                    try:
                        header = read_gzip_json(header_path)
                    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                        header = {}
                    if isinstance(header, dict):
                        pm = header.get("partition_map") or {}
                        if isinstance(pm, dict):
                            template_count += len(pm)
                feedback_dir = os.path.join(space_dir, TEMPLATE_STORE_FEEDBACK_SEGMENT)
                if os.path.isdir(feedback_dir):
                    feedback_shard_count += sum(
                        1
                        for name in os.listdir(feedback_dir)
                        if name.startswith(TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX) and name.endswith(".json.gz")
                    )
                orphan_root = os.path.join(space_dir, TEMPLATE_STORE_ORPHANED_SEGMENT)
                if os.path.isdir(orphan_root):
                    orphan_count += sum(
                        1 for name in os.listdir(orphan_root) if os.path.isdir(os.path.join(orphan_root, name))
                    )
        return template_count, feedback_shard_count, orphan_count

    @staticmethod
    def reconcile_template_store(store: dict[str, Any] | TemplateStoreView, schema: SchemaGraph) -> _ReconcileReport:
        """Drop templates whose footprint no longer survives the live schema; stamp surviving ids onto the new graph id."""
        kept_templates: list[str] = []
        dropped_templates: list[str] = []
        reason_hist: Counter[str] = Counter()
        templates = store.get("templates", {}) or {}
        if isinstance(store, TemplateStoreView):
            TemplateStoreLifecycleOps.prune_orphan_template_ids(store)
            for batch in store.iter_templates_by_partition():
                for tmpl in batch:
                    stamped = TemplateRefs.stamp_template_footprint(tmpl)
                    ok, reasons = TemplateRefs.footprint_survives(stamped, schema)
                    if ok:
                        stamped.schema_graph_id = schema.schema_graph_id
                        stamped.effective_structural_hash = schema.effective_structural_hash
                        store.set_template_raw_dict(stamped.id, stamped.to_dict())
                        kept_templates.append(stamped.id)
                    else:
                        dropped_templates.append(tmpl.id)
                        for r in reasons:
                            reason_hist[r] += 1
                        store.remove_template_id(tmpl.id)
        else:
            for tid, raw in list(templates.items()):
                tmpl = Template.from_dict({**raw, "id": tid})
                stamped = TemplateRefs.stamp_template_footprint(tmpl)
                ok, reasons = TemplateRefs.footprint_survives(stamped, schema)
                if ok:
                    stamped.schema_graph_id = schema.schema_graph_id
                    stamped.effective_structural_hash = schema.effective_structural_hash
                    templates[tid] = stamped.to_dict()
                    kept_templates.append(tid)
                else:
                    dropped_templates.append(tid)
                    for r in reasons:
                        reason_hist[r] += 1
                    del templates[tid]
            store["templates"] = templates

        TemplateStoreView.refresh_template_store_indexes(store)

        return _ReconcileReport(
            kept_template_ids=tuple(sorted(kept_templates)),
            dropped_template_ids=tuple(sorted(dropped_templates)),
            kept_rejected_ids=(),
            dropped_rejected_ids=(),
            dropped_negative_memory_bucket_count=0,
            dropped_failure_log_rows=0,
            dropped_warmup_units=0,
            reason_histogram=dict(reason_hist),
        )

    @staticmethod
    def _rename_history_path(artifacts_dir: str) -> str:
        return os.path.join(artifacts_dir, RENAME_HISTORY_FILENAME)

    @staticmethod
    def read_rename_history(artifacts_dir: str) -> list[dict[str, Any]]:
        path = TemplateStoreLifecycleOps._rename_history_path(artifacts_dir)
        if not os.path.isfile(path):
            return []
        try:
            raw = read_gzip_json(path)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if isinstance(raw, dict):
            entries = raw.get("entries")
            if isinstance(entries, list):
                return [row for row in entries if isinstance(row, dict)]
            return []
        if isinstance(raw, list):
            return [row for row in raw if isinstance(row, dict)]
        return []

    @staticmethod
    def append_rename_history(
        artifacts_dir: str,
        *,
        from_schema_graph_id: str,
        to_schema_graph_id: str,
        renamed_tables: tuple[tuple[str, str], ...],
        renamed_columns: tuple[tuple[str, str, str], ...],
    ) -> None:
        if not from_schema_graph_id or not to_schema_graph_id or from_schema_graph_id == to_schema_graph_id:
            return
        if not renamed_tables and not renamed_columns:
            return
        os.makedirs(artifacts_dir, exist_ok=True)
        entries = TemplateStoreLifecycleOps.read_rename_history(artifacts_dir)
        entries.append(
            {
                "from_schema_graph_id": from_schema_graph_id,
                "to_schema_graph_id": to_schema_graph_id,
                "renamed_tables": [list(pair) for pair in renamed_tables],
                "renamed_columns": [list(triple) for triple in renamed_columns],
                "applied_at": datetime.now(UTC).isoformat(),
            }
        )
        write_gzip_json_atomic(
            TemplateStoreLifecycleOps._rename_history_path(artifacts_dir),
            {"entries": entries},
            sort_keys=True,
        )

    @staticmethod
    def compose_rename_chain(
        history: Sequence[dict[str, Any]],
        from_id: str,
        to_id: str,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]:
        """Compose table/column rename tuples along a persisted identity chain."""
        if from_id == to_id:
            return (), ()
        edges_by_from: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in history:
            src = str(entry.get("from_schema_graph_id", "") or "")
            dst = str(entry.get("to_schema_graph_id", "") or "")
            if src and dst:
                edges_by_from[src].append(entry)
        queue: list[tuple[str, list[dict[str, Any]]]] = [(from_id, [])]
        visited: set[str] = {from_id}
        path_entries: list[dict[str, Any]] | None = None
        while queue:
            cur, path = queue.pop(0)
            if cur == to_id:
                path_entries = path
                break
            for entry in edges_by_from.get(cur, []):
                nxt = str(entry.get("to_schema_graph_id", "") or "")
                if not nxt or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [entry]))
        if path_entries is None:
            raise BrokenRenameChainError(f"no rename chain from {from_id!r} to {to_id!r}")

        table_maps: list[dict[str, str]] = []
        column_hops: list[tuple[tuple[str, str, str], ...]] = []
        for entry in path_entries:
            tmap: dict[str, str] = {}
            for pair in entry.get("renamed_tables") or []:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                old_t, new_t = str(pair[0]), str(pair[1])
                if old_t == new_t:
                    continue
                prev = tmap.get(old_t)
                if prev is not None and prev != new_t:
                    raise BrokenRenameChainError(f"conflicting table rename for {old_t!r}")
                tmap[old_t] = new_t
            table_maps.append(tmap)
            cols: list[tuple[str, str, str]] = []
            for triple in entry.get("renamed_columns") or []:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                cols.append((str(triple[0]), str(triple[1]), str(triple[2])))
            column_hops.append(tuple(cols))

        all_tables: set[str] = set()
        for tmap in table_maps:
            all_tables.update(tmap.keys())
            all_tables.update(tmap.values())
        origin_tables: set[str] = set()
        if path_entries:
            for pair in path_entries[0].get("renamed_tables") or []:
                if isinstance(pair, (list, tuple)) and pair:
                    origin_tables.add(str(pair[0]))
        if not origin_tables:
            origin_tables = set(all_tables)
        composed_tables: dict[str, str] = {}
        for name in origin_tables:
            cur_name = name
            for tmap in table_maps:
                cur_name = tmap.get(cur_name, cur_name)
            if cur_name != name:
                composed_tables[name] = cur_name

        col_origins: set[tuple[str, str]] = set()
        if column_hops:
            for ot, oc, _nc in column_hops[0]:
                col_origins.add((ot, oc))
        if not col_origins:
            for hop in column_hops:
                for ot, oc, _nc in hop:
                    col_origins.add((ot, oc))
        composed_columns: list[tuple[str, str, str]] = []
        for orig_t, orig_c in sorted(col_origins):
            cur_t, cur_c = orig_t, orig_c
            for tmap, hop in zip(table_maps, column_hops, strict=True):
                for ot, oc, nc in hop:
                    if ot == cur_t and oc == cur_c:
                        cur_c = nc
                cur_t = tmap.get(cur_t, cur_t)
            if cur_c != orig_c:
                composed_columns.append((orig_t, orig_c, cur_c))

        return (tuple(sorted(composed_tables.items())), tuple(composed_columns))

    @staticmethod
    def _migration_sql_compare_key(sql: str) -> str:
        return normalize_sql(canonicalize_sql(sql or "")).replace('"', "").replace("'", "")

    @staticmethod
    def _rerender_migration_sql(tmpl: Template, schema: SchemaGraph) -> str | None:
        """Re-render SQL from a template intent for migration verification and persistence."""
        try:
            runtime = tmpl.intent_signature.to_runtime_skeleton()
            sig = tmpl.intent_signature
            if sig.chosen_join_path_signature:
                runtime.chosen_join_path_signature = list(sig.chosen_join_path_signature)
            if sig.chosen_join_candidate_id:
                runtime.chosen_join_candidate_id = sig.chosen_join_candidate_id
            dialect = DialectRegistry.get_dialect("duckdb")
            return build_deterministic_sql(runtime, schema=schema, dialect=dialect)
        except Exception as exc:
            debug(f"[templates._rerender_migration_sql] rerender failed: {exc!r}")
            return None

    @staticmethod
    def _verify_remapped_template_sql(tmpl: Template, schema: SchemaGraph) -> bool:
        """Return True when pattern-remapped SQL matches a normal intent re-render."""
        rerendered = TemplateStoreLifecycleOps._rerender_migration_sql(tmpl, schema)
        if rerendered is None:
            return False
        return TemplateStoreLifecycleOps._migration_sql_compare_key(
            tmpl.sql_param or ""
        ) == TemplateStoreLifecycleOps._migration_sql_compare_key(rerendered)

    @staticmethod
    def _map_join_side(side: str, tmap: dict[str, str], colmaps: dict[str, dict[str, str]]) -> str:
        side = side.strip()
        if "." not in side:
            return tmap.get(side, side)
        tbl, rest = side.split(".", 1)
        cols = [c.strip() for c in rest.split(",") if c.strip()]
        nt = tmap.get(tbl, tbl)
        cm = colmaps.get(tbl, {})
        mapped_cols = [cm.get(c, c) for c in cols]
        return f"{nt}.{','.join(mapped_cols)}"

    @staticmethod
    def _rewrite_join_path_segments(
        sigs: list[str], tmap: dict[str, str], colmaps: dict[str, dict[str, str]]
    ) -> list[str]:
        out: list[str] = []
        for seg in sigs:
            s = str(seg).strip()
            if "->" not in s:
                out.append(seg)
                continue
            a, b = s.split("->", 1)
            out.append(
                f"{TemplateStoreLifecycleOps._map_join_side(a, tmap, colmaps)}->{TemplateStoreLifecycleOps._map_join_side(b, tmap, colmaps)}"
            )
        return out

    @staticmethod
    def _remap_column_map(
        column_map: dict[str, str], tmap: dict[str, str], colmaps: dict[str, dict[str, str]]
    ) -> dict[str, str]:
        """Apply table and column renames to a ``{bare_col: table}`` dict. ``colmaps`` is keyed by *old* table name, so we look up renames using the pre-rename table even when the table itself is being renamed."""
        out: dict[str, str] = {}
        for bare, tbl in column_map.items():
            new_tbl = tmap.get(tbl, tbl)
            new_bare = colmaps.get(tbl, {}).get(bare, bare)
            out[new_bare] = new_tbl
        return out

    @staticmethod
    def _schema_migration_column_replacer(
        tmap: dict[str, str], colmaps: dict[str, dict[str, str]]
    ) -> Callable[[str], str]:
        def replacer(ref: str) -> str:
            if "." not in ref:
                return tmap.get(ref, ref)
            tbl, col = ref.split(".", 1)
            nt = tmap.get(tbl, tbl)
            nc = colmaps.get(tbl, {}).get(col, col)
            return f"{nt}.{nc}"

        return replacer

    @staticmethod
    def _remap_concrete_clause_fields(
        *,
        select_cols: list[Any],
        group_by_cols: list[Any],
        order_by_cols: list[Any],
        where: Any,
        having: Any,
        replacer: Callable[[str], str],
    ) -> tuple[list[Any], list[Any], list[Any], Any, Any]:
        def remap_expr(expr: Any) -> Any:
            return replace_refs_in_expr(expr, replacer)

        remap_where = PredicateGroup.map(
            where,
            lambda fp: replace(
                fp,
                left_expr=remap_expr(fp.left_expr),
                right_expr=(remap_expr(fp.right_expr) if fp.right_expr else None),
            ),
        )
        remap_having = PredicateGroup.map(
            having,
            lambda hp: replace(
                hp,
                left_expr=remap_expr(hp.left_expr),
                right_expr=(remap_expr(hp.right_expr) if hp.right_expr else None),
            ),
        )
        return (
            [replace(sc, expr=remap_expr(sc.expr)) for sc in (select_cols or [])],
            [remap_expr(g) for g in (group_by_cols or [])],
            [replace(obc, expr=remap_expr(obc.expr)) for obc in (order_by_cols or [])],
            remap_where,
            remap_having,
        )

    @staticmethod
    def _remap_concrete_cte(
        cte: ConcreteCteStep, tmap: dict[str, str], colmaps: dict[str, dict[str, str]]
    ) -> ConcreteCteStep:
        replacer = TemplateStoreLifecycleOps._schema_migration_column_replacer(tmap, colmaps)
        select_cols, group_by_cols, order_by_cols, where, having = (
            TemplateStoreLifecycleOps._remap_concrete_clause_fields(
                select_cols=cte.select_cols,
                group_by_cols=cte.group_by_cols,
                order_by_cols=cte.order_by_cols,
                where=cte.where,
                having=cte.having,
                replacer=replacer,
            )
        )
        return replace(
            cte,
            tables=[tmap.get(x, x) for x in cte.tables],
            column_map=TemplateStoreLifecycleOps._remap_column_map(cte.column_map, tmap, colmaps),
            chosen_join_path_signature=TemplateStoreLifecycleOps._rewrite_join_path_segments(
                cte.chosen_join_path_signature, tmap, colmaps
            ),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
        )

    @staticmethod
    def _remap_concrete_intent(
        ci: ConcreteIntent, tmap: dict[str, str], colmaps: dict[str, dict[str, str]]
    ) -> ConcreteIntent:
        replacer = TemplateStoreLifecycleOps._schema_migration_column_replacer(tmap, colmaps)
        select_cols, group_by_cols, order_by_cols, where, having = (
            TemplateStoreLifecycleOps._remap_concrete_clause_fields(
                select_cols=ci.select_cols,
                group_by_cols=ci.group_by_cols,
                order_by_cols=ci.order_by_cols,
                where=ci.where,
                having=ci.having,
                replacer=replacer,
            )
        )
        return replace(
            ci,
            tables=[tmap.get(x, x) for x in ci.tables],
            column_map=TemplateStoreLifecycleOps._remap_column_map(ci.column_map, tmap, colmaps),
            chosen_join_path_signature=TemplateStoreLifecycleOps._rewrite_join_path_segments(
                ci.chosen_join_path_signature, tmap, colmaps
            ),
            cte_steps=[TemplateStoreLifecycleOps._remap_concrete_cte(cte, tmap, colmaps) for cte in ci.cte_steps],
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
        )

    @staticmethod
    def _remap_sql_strings(
        sql: str,
        tmap: dict[str, str],
        colmaps: dict[str, dict[str, str]],
        *,
        column_map: dict[str, str] | None = None,
        renamed_columns: tuple[tuple[str, str, str], ...] = (),
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
        for ot, oc, nc in renamed_columns:
            nt = tmap.get(ot, ot)
            out = re.sub(rf"(?<![\w.]){re.escape(oc)}(?![\w])", f"{nt}.{nc}", out)
        if column_map:
            for bare, tbl in column_map.items():
                nt = tmap.get(tbl, tbl)
                out = re.sub(rf"(?<![\w.]){re.escape(bare)}(?![\w])", f"{nt}.{bare}", out)
        return out

    @staticmethod
    def _remap_templates_in_view(
        view: TemplateStoreView,
        schema: SchemaGraph,
        renamed_tables: tuple[tuple[str, str], ...],
        renamed_columns: tuple[tuple[str, str, str], ...],
        *,
        tombstoned_tables: frozenset[str] | None = None,
        tombstoned_cols: frozenset[tuple[str, str]] | None = None,
    ) -> tuple[int, int]:
        tmap = dict(renamed_tables)
        colmaps: dict[str, dict[str, str]] = defaultdict(dict)
        for ot, oc, nc in renamed_columns:
            colmaps[ot][oc] = nc
        colmaps = {k: dict(v) for k, v in colmaps.items()}
        remapped = 0
        destroyed = 0
        for tid in list(view.partition_map.keys()):
            raw = view.get_template_raw(tid)
            if raw is None:
                continue
            tmpl = Template.from_dict({**raw, "id": tid})
            new_sig = TemplateStoreLifecycleOps._remap_concrete_intent(tmpl.intent_signature, tmap, colmaps)
            pattern_sql = TemplateStoreLifecycleOps._remap_sql_strings(
                tmpl.sql_param,
                tmap,
                colmaps,
                column_map=dict(new_sig.column_map or {}),
                renamed_columns=renamed_columns,
            )
            new_tables_used = [tmap.get(x, x) for x in (tmpl.tables_used or [])]
            rerendered_sql = TemplateStoreLifecycleOps._rerender_migration_sql(
                replace(tmpl, intent_signature=new_sig, tables_used=new_tables_used),
                schema,
            )
            new_sql = rerendered_sql or pattern_sql
            new_schema_column_types: dict[str, str] = {}
            for key, vtype in (tmpl.schema_column_types or {}).items():
                ref = str(key)
                if "." in ref:
                    tbl, col = ref.split(".", 1)
                    nt = tmap.get(tbl, tbl)
                    nc = colmaps.get(tbl, {}).get(col, col)
                    new_schema_column_types[f"{nt}.{nc}"] = str(vtype)
                else:
                    new_schema_column_types[ref] = str(vtype)
            rebuilt = replace(
                tmpl,
                intent_signature=new_sig,
                tables_used=new_tables_used,
                effective_structural_hash=schema.effective_structural_hash,
                schema_graph_id=schema.schema_graph_id,
                sql_param=new_sql,
                schema_column_types=new_schema_column_types,
            )
            rebuilt.sql_fp = Dialect.compute_sql_fp(
                rebuilt.sql_param or "", sqlglot_dialect=Dialect.active_sqlglot_dialect()
            )
            ok, _ = TemplateRefs.template_is_live(TemplateRefs.template_schema_refs(rebuilt), schema)
            if not ok:
                refs = TemplateRefs.template_schema_refs(tmpl)
                join_tables = TemplateStoreLifecycleOps._tables_from_join_signature_tokens(refs.fk_edges)
                if tombstoned_tables or tombstoned_cols:
                    if (
                        refs.tables & (tombstoned_tables or frozenset())
                        or refs.columns & (tombstoned_cols or frozenset())
                        or join_tables & (tombstoned_tables or frozenset())
                    ):
                        continue
                view.remove_template_id(tid)
                destroyed += 1
                continue
            if rerendered_sql:
                sql_matches = TemplateStoreLifecycleOps._migration_sql_compare_key(
                    pattern_sql
                ) == TemplateStoreLifecycleOps._migration_sql_compare_key(rerendered_sql)
            else:
                sql_matches = TemplateStoreLifecycleOps._verify_remapped_template_sql(rebuilt, schema)
            if not sql_matches:
                notify(
                    f"Template {tid} invalidated: remapped SQL diverged from intent re-render",
                    stage="templates.remap",
                    code=DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED,
                    level=DiagnosticSeverity.WARNING,
                    details=(("template_id", tid),),
                )
                view.remove_template_id(tid)
                destroyed += 1
                continue
            view.set_template_raw_dict(tid, rebuilt.to_dict())
            remapped += 1
        view["schema_graph_id"] = schema.schema_graph_id
        TemplateStoreView.refresh_template_store_indexes(view)
        return remapped, destroyed

    @staticmethod
    def _apply_schema_rename_migration_to_store(
        artifacts_dir: str,
        schema: SchemaGraph,
        renamed_tables: tuple[tuple[str, str], ...],
        renamed_columns: tuple[tuple[str, str, str], ...],
        *,
        tombstoned_tables: frozenset[str] | None = None,
        tombstoned_cols: frozenset[tuple[str, str]] | None = None,
    ) -> tuple[int, int]:
        """Rewrite persisted template intents for schema rename migration."""
        TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
        remapped = 0
        destroyed = 0
        touched = False
        from_graph_id = ""
        with artifact_lock(artifacts_dir):
            prev_manifest = TemplateStoreLifecycleOps.read_store_manifest(artifacts_dir)
            if prev_manifest is not None and prev_manifest.schema_graph_id:
                from_graph_id = prev_manifest.schema_graph_id
            for _space_name, store_dir in TemplateStoreLifecycleOps.iter_template_store_space_dirs(artifacts_dir):
                header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
                if not os.path.isfile(header_path):
                    continue
                view = TemplateStoreLifecycleOps._load_partitioned_view_unlocked(store_dir)
                if view is None or not view.partition_map:
                    continue
                touched = True
                if not from_graph_id and view.schema_graph_id:
                    from_graph_id = view.schema_graph_id
                space_remapped, space_destroyed = TemplateStoreLifecycleOps._remap_templates_in_view(
                    view,
                    schema,
                    renamed_tables,
                    renamed_columns,
                    tombstoned_tables=tombstoned_tables,
                    tombstoned_cols=tombstoned_cols,
                )
                remapped += space_remapped
                destroyed += space_destroyed
                TemplateStoreLifecycleOps._persist_template_store_view(view, artifacts_dir)
            if touched:
                if from_graph_id and from_graph_id != schema.schema_graph_id:
                    TemplateStoreLifecycleOps.append_rename_history(
                        artifacts_dir,
                        from_schema_graph_id=from_graph_id,
                        to_schema_graph_id=schema.schema_graph_id,
                        renamed_tables=renamed_tables,
                        renamed_columns=renamed_columns,
                    )
                prev = TemplateStoreLifecycleOps.read_store_manifest(artifacts_dir)
                write_artifact_manifest(
                    artifacts_dir,
                    structural_hash=prev.structural_hash if prev else "",
                    profiling_hash=prev.profiling_hash if prev else "",
                    scope_hash=prev.scope_hash if prev else "",
                    effective_structural_hash=(prev.effective_structural_hash if prev else ""),
                    schema_graph_id=schema.schema_graph_id,
                    notes_hash=prev.notes_hash if prev else "",
                    semantic_edges_hash=prev.semantic_edges_hash if prev else "",
                    last_migration_tier=prev.last_migration_tier if prev else "",
                    last_migration_at=prev.last_migration_at if prev else "",
                    last_action="remap_templates",
                )
        return (remapped, destroyed)

    @staticmethod
    def _disk_template_row_count(artifacts_dir: str) -> int:
        TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
        total = 0
        for _space_name, store_dir in TemplateStoreLifecycleOps.iter_template_store_space_dirs(artifacts_dir):
            try:
                view = TemplateStoreLifecycleOps._load_partitioned_view_unlocked(store_dir)
            except (OSError, ValueError, TypeError):
                continue
            if view is None:
                continue
            n_tpl = len(view.partition_map)
            n_qf = view._count_question_feedback_rows()
            total += n_tpl + n_qf
        return total

    @staticmethod
    def _surgical_invalidation_targets(schema_diff: SchemaDiff) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
        """Build the tombstone sets used to detect templates invalidated by a SchemaDiff. Returns ``(tombstoned_tables, tombstoned_columns)``. A template is surgically deleted when its references intersect either set. Nullability, uniqueness, indexes, and view-definition-only changes are soft-flagged elsewhere and do not tombstone."""
        tombstoned_tables: set[str] = set(schema_diff.dropped_tables)
        tombstoned_cols: set[tuple[str, str]] = set()
        for table, td in schema_diff.per_table.items():
            for col in td.dropped_columns:
                tombstoned_cols.add((table, col))
            for col, _old_vt, _new_vt in td.value_type_changed_columns:
                tombstoned_cols.add((table, col))
            for col, _old_dt, _new_dt in td.redeclared_columns:
                tombstoned_cols.add((table, col))
            if td.fk_changed or td.pk_changed:
                tombstoned_tables.add(table)
        return frozenset(tombstoned_tables), frozenset(tombstoned_cols)

    @staticmethod
    def _surgical_soft_flag_targets(schema_diff: SchemaDiff) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
        """Return metadata-only change targets that soft-flag without deleting templates."""
        soft_tables: set[str] = set()
        soft_cols: set[tuple[str, str]] = set()
        for table, td in schema_diff.per_table.items():
            for col in td.nullability_changed_columns:
                soft_cols.add((table, col))
            for col in td.uniqueness_changed_columns:
                soft_cols.add((table, col))
            if td.indexes_changed or td.view_definition_changed:
                soft_tables.add(table)
        return frozenset(soft_tables), frozenset(soft_cols)

    @staticmethod
    def _tables_from_join_signature_tokens(fk_edges: frozenset[str]) -> set[str]:
        """Extract table names referenced in stored join path signature tokens."""
        out: set[str] = set()
        for token in fk_edges:
            s = str(token).strip()
            if "->" not in s:
                continue
            left, right = s.split("->", 1)
            out.add(left.split(".")[0].strip())
            out.add(right.split(".")[0].strip())
        return out

    @staticmethod
    def surgical_invalidate_templates_by_diff(artifacts_dir: str, schema: SchemaGraph, schema_diff: SchemaDiff) -> int:
        """Delete persisted accepted templates whose references intersect a SchemaDiff's tombstone sets. A no-op (returns 0) when the store is missing or the diff yields no tombstones."""
        tombstoned_tables, tombstoned_cols = TemplateStoreLifecycleOps._surgical_invalidation_targets(schema_diff)
        if not tombstoned_tables and not tombstoned_cols:
            return 0
        TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
        deleted = 0
        with artifact_lock(artifacts_dir):
            for _space_name, store_dir in TemplateStoreLifecycleOps.iter_template_store_space_dirs(artifacts_dir):
                header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
                if not os.path.isfile(header_path):
                    continue
                view = TemplateStoreLifecycleOps._load_partitioned_view_unlocked(store_dir)
                if view is None or not view.partition_map:
                    continue
                space_deleted = 0
                for tid in list(view.partition_map.keys()):
                    raw = view.get_template_raw(tid)
                    if raw is None:
                        continue
                    tmpl = Template.from_dict({**raw, "id": tid})
                    refs = TemplateRefs.template_schema_refs(tmpl)
                    join_tables = TemplateStoreLifecycleOps._tables_from_join_signature_tokens(refs.fk_edges)
                    if (
                        refs.tables & tombstoned_tables
                        or refs.columns & tombstoned_cols
                        or join_tables & tombstoned_tables
                    ):
                        view.remove_template_id(tid)
                        space_deleted += 1
                        continue
                    ok, _reasons = TemplateRefs.template_is_live(refs, schema)
                    if not ok:
                        view.remove_template_id(tid)
                        space_deleted += 1
                if space_deleted:
                    view["schema_graph_id"] = schema.schema_graph_id
                    TemplateStoreLifecycleOps._persist_template_store_view(view, artifacts_dir)
                    deleted += space_deleted
        return deleted

    @staticmethod
    def _dropped_columns_from_diff(schema_diff: SchemaDiff) -> tuple[str, ...]:
        dropped: list[str] = []
        for table, td in schema_diff.per_table.items():
            for col in td.dropped_columns:
                dropped.append(f"{table}.{col}")
        return tuple(sorted(set(dropped)))

    @staticmethod
    def apply_structural_migration_from_schema_diff(
        artifacts_dir: str, schema_diff: SchemaDiff, *, schema: SchemaGraph | None = None
    ) -> None:
        """Prune/remap persisted aetherspace snapshots and named context specs for *schema_diff*."""
        renamed_tables, renamed_columns = TemplateStoreLifecycleOps._renames_from_diff(schema_diff)
        column_retypes: list[tuple[str, str, str]] = []
        for table_name, td in schema_diff.per_table.items():
            for col_name, _old_dt, new_dt in td.retyped_columns:
                column_retypes.append((table_name, col_name, new_dt))
            for col_name, _old_dt, new_dt in td.redeclared_columns:
                column_retypes.append((table_name, col_name, new_dt))
        dropped_columns = TemplateStoreLifecycleOps._dropped_columns_from_diff(schema_diff)
        apply_structural_migration_to_persisted_scopes(
            artifacts_dir,
            dropped_tables=schema_diff.dropped_tables,
            dropped_columns=dropped_columns,
            table_renames=renamed_tables,
            column_renames=renamed_columns,
            column_retypes=tuple(column_retypes),
        )
        if schema is not None:
            migrate_engine_knowledge_artifacts(
                artifacts_dir,
                schema,
                dropped_tables=tuple(schema_diff.dropped_tables),
                dropped_columns=dropped_columns,
                table_renames=renamed_tables,
                column_renames=renamed_columns,
            )

    @staticmethod
    def apply_structural_migration_from_map(artifacts_dir: str, map_obj: SchemaMigrationMap) -> None:
        """Prune/remap persisted aetherspace snapshots and named context specs for a user map."""
        dropped_columns = tuple(
            f"{e.table}.{e.from_name}" for e in map_obj.dropped_columns if e.entry_type == "dropped_column"
        )
        renamed_tables = tuple(
            (e.from_name, e.to_name)
            for e in map_obj.table_renames
            if e.entry_type == "table" and e.from_name and e.to_name
        )
        renamed_columns = tuple(
            (e.table, e.from_name, e.to_name)
            for e in map_obj.column_renames
            if e.entry_type == "column" and e.from_name and e.to_name
        )
        apply_structural_migration_to_persisted_scopes(
            artifacts_dir,
            dropped_tables=tuple(map_obj.dropped_tables),
            dropped_columns=dropped_columns,
            table_renames=renamed_tables,
            column_renames=renamed_columns,
        )

    @staticmethod
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

    @staticmethod
    def _additions_from_diff(schema_diff: SchemaDiff) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Project added tables/columns from a :class:`SchemaDiff` for migration reporting."""
        added_columns: list[tuple[str, str]] = []
        for table_name, table_diff in schema_diff.per_table.items():
            for column_name in table_diff.added_columns:
                added_columns.append((table_name, column_name))
        return tuple(sorted(schema_diff.added_tables)), tuple(sorted(added_columns))

    @staticmethod
    def _restore_schema_graph_from_backup(schema: SchemaGraph, backup: SchemaGraph) -> None:
        for field_name in schema.__dataclass_fields__:
            setattr(schema, field_name, copy.deepcopy(getattr(backup, field_name)))

    @staticmethod
    def _value_type_changes_from_diff(schema_diff: SchemaDiff) -> tuple[tuple[str, str, str], ...]:
        """Flatten per-table ``value_type_changed_columns`` into ``(column, old_vt, new_vt)`` tuples."""
        out: list[tuple[str, str, str]] = []
        for _table, td in schema_diff.per_table.items():
            for col, old_vt, new_vt in td.value_type_changed_columns:
                out.append((col, old_vt, new_vt))
        return tuple(out)

    @staticmethod
    def _stamp_manifest(artifacts_dir: str, schema: SchemaGraph, *, tier: MigrationTier, last_action: str) -> None:
        """Write the artifact manifest with current schema fingerprints and the given tier."""
        write_artifact_manifest(
            artifacts_dir,
            structural_hash=schema.structural_hash,
            profiling_hash=schema.profiling_hash,
            scope_hash=schema.scope_hash,
            effective_structural_hash=schema.effective_structural_hash,
            schema_graph_id=schema.schema_graph_id,
            notes_hash=schema.notes_hash,
            semantic_edges_hash=schema.semantic_edges_hash,
            last_migration_tier=tier.value,
            last_action=last_action,
        )

    @staticmethod
    def apply_migration_policy(
        artifacts_dir: str,
        schema: SchemaGraph,
        *,
        allow_destructive: bool = True,
        previous_schema: SchemaGraph | None = None,
        schema_diff: SchemaDiff | None = None,
    ) -> MigrationReport:
        """Reconcile the persisted template store against ``schema``. When ``schema_diff`` is provided and non-empty, the diff drives tier selection and surgical invalidation directly: renames feed :func:`_apply_schema_rename_migration_to_store` and dropped tables, dropped columns, and ``value_type``-changed columns feed :func:`surgical_invalidate_templates_by_diff`. REMAP and SOFT_REFRESH may co-occur on a mixed diff. When ``schema_diff`` is ``None`` (for example after a full rebuild that did not produce a diff), re-derive a rename plan from ``previous_schema`` via :func:`try_rename_migration_plan`."""
        os.makedirs(artifacts_dir, exist_ok=True)
        with artifact_lock(artifacts_dir):
            return TemplateStoreLifecycleOps._apply_migration_policy_locked(
                artifacts_dir,
                schema,
                allow_destructive=allow_destructive,
                previous_schema=previous_schema,
                schema_diff=schema_diff,
            )

    @staticmethod
    def apply_federation_composite_migration_policy(
        federation_dir: str,
        composite: SchemaGraph,
        *,
        allow_destructive: bool = True,
        previous_composite: SchemaGraph | None = None,
    ) -> MigrationReport:
        """Apply template migration policy to a federation composite store."""
        stored = federation_artifact_manifest_view(federation_dir)
        tier = classify_migration_tier(stored, composite, previous_schema=previous_composite)
        if tier in (MigrationTier.NO_CHANGE, MigrationTier.PERMISSION_FILTERED):
            return MigrationReport(tier=tier)

        def _federation_read_manifest(adir: str) -> Any:
            if paths_equal(adir, federation_dir):
                return stored
            return read_artifact_manifest(adir)

        prev_reader = TemplateStoreLifecycleOps._manifest_reader
        TemplateStoreLifecycleOps._manifest_reader = _federation_read_manifest
        try:
            return TemplateStoreLifecycleOps.apply_migration_policy(
                federation_dir,
                composite,
                allow_destructive=allow_destructive,
                previous_schema=previous_composite,
            )
        finally:
            TemplateStoreLifecycleOps._manifest_reader = prev_reader

    @staticmethod
    def _apply_migration_policy_locked(
        artifacts_dir: str,
        schema: SchemaGraph,
        *,
        allow_destructive: bool,
        previous_schema: SchemaGraph | None,
        schema_diff: SchemaDiff | None,
    ) -> MigrationReport:
        """Body of :func:`apply_migration_policy` executed under the artifacts-dir lock."""
        stored = TemplateStoreLifecycleOps.read_store_manifest(artifacts_dir)
        tier = classify_migration_tier(stored, schema, previous_schema=previous_schema, schema_diff=schema_diff)
        if tier == MigrationTier.PERMISSION_FILTERED:
            return MigrationReport(tier=tier)
        if tier == MigrationTier.NO_CHANGE:
            return MigrationReport(tier=tier)
        if schema_diff is not None and not schema_diff.is_empty:
            if schema_diff_is_additive_only(schema_diff):
                added_tables, added_columns = TemplateStoreLifecycleOps._additions_from_diff(schema_diff)
                refresh_migration_auxiliary_artifacts(artifacts_dir, tier=MigrationTier.ADDITIVE)
                TemplateStoreLifecycleOps.apply_structural_migration_from_schema_diff(
                    artifacts_dir, schema_diff, schema=schema
                )
                TemplateStoreLifecycleOps._stamp_manifest(
                    artifacts_dir, schema, tier=MigrationTier.ADDITIVE, last_action="additive"
                )
                return MigrationReport(
                    tier=MigrationTier.ADDITIVE,
                    added_tables=added_tables,
                    added_columns=added_columns,
                )
            return TemplateStoreLifecycleOps._apply_diff_driven_policy(
                artifacts_dir, schema, schema_diff, allow_destructive=allow_destructive
            )
        if tier == MigrationTier.ADDITIVE:
            refresh_migration_auxiliary_artifacts(artifacts_dir, tier=tier)
            TemplateStoreLifecycleOps._stamp_manifest(artifacts_dir, schema, tier=tier, last_action="additive")
            return MigrationReport(tier=tier)
        if tier == MigrationTier.SOFT_REFRESH:
            debug("[templates.apply_migration_policy] soft_refresh: updating manifest fingerprints")
            refresh_migration_auxiliary_artifacts(artifacts_dir, tier=tier)
            if schema_diff is not None and not schema_diff.is_empty:
                TemplateStoreLifecycleOps.apply_structural_migration_from_schema_diff(
                    artifacts_dir, schema_diff, schema=schema
                )
            TemplateStoreLifecycleOps._stamp_manifest(artifacts_dir, schema, tier=tier, last_action="soft_refresh")
            return MigrationReport(tier=tier)
        if tier == MigrationTier.REMAP:
            plan = try_rename_migration_plan(previous_schema, schema) if previous_schema is not None else None
            if plan is None:
                if not allow_destructive:
                    return MigrationReport(tier=MigrationTier.NO_CHANGE)
                destroyed = TemplateStoreLifecycleOps._disk_template_row_count(artifacts_dir)
                destructive_migration_execute(artifacts_dir, schema)
                return MigrationReport(tier=MigrationTier.DESTRUCTIVE, destroyed_templates=destroyed)
            renamed_tables, renamed_columns = plan
            remapped, destroyed = TemplateStoreLifecycleOps._apply_schema_rename_migration_to_store(
                artifacts_dir, schema, renamed_tables, renamed_columns
            )
            TemplateStoreLifecycleOps._stamp_manifest(
                artifacts_dir, schema, tier=MigrationTier.REMAP, last_action="remap"
            )
            apply_structural_migration_to_persisted_scopes(
                artifacts_dir, table_renames=renamed_tables, column_renames=renamed_columns
            )
            return MigrationReport(
                tier=MigrationTier.REMAP,
                renamed_tables=renamed_tables,
                renamed_columns=renamed_columns,
                destroyed_templates=destroyed,
                remapped_templates=remapped,
            )
        if not allow_destructive:
            return MigrationReport(tier=MigrationTier.NO_CHANGE)
        destroyed = TemplateStoreLifecycleOps._disk_template_row_count(artifacts_dir)
        destructive_migration_execute(artifacts_dir, schema)
        return MigrationReport(tier=MigrationTier.DESTRUCTIVE, destroyed_templates=destroyed)

    @staticmethod
    def _apply_diff_driven_policy(
        artifacts_dir: str, schema: SchemaGraph, schema_diff: SchemaDiff, *, allow_destructive: bool
    ) -> MigrationReport:
        """Diff-driven policy dispatch: REMAP for renames, SOFT_REFRESH + surgery for drops/vt-changes."""
        renamed_tables, renamed_columns = TemplateStoreLifecycleOps._renames_from_diff(schema_diff)
        has_remap = bool(renamed_tables or renamed_columns)
        tombstoned_tables, tombstoned_cols = TemplateStoreLifecycleOps._surgical_invalidation_targets(schema_diff)
        has_surgery = bool(tombstoned_tables or tombstoned_cols)
        if not allow_destructive and (has_remap or has_surgery):
            return MigrationReport(tier=MigrationTier.NO_CHANGE)
        remapped = 0
        destroyed = 0
        surgically = 0

        if has_remap:
            remapped, destroyed = TemplateStoreLifecycleOps._apply_schema_rename_migration_to_store(
                artifacts_dir,
                schema,
                renamed_tables,
                renamed_columns,
                tombstoned_tables=tombstoned_tables if has_surgery else None,
                tombstoned_cols=tombstoned_cols if has_surgery else None,
            )
        if has_surgery:
            surgically = TemplateStoreLifecycleOps.surgical_invalidate_templates_by_diff(
                artifacts_dir, schema, schema_diff
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
        refresh_migration_auxiliary_artifacts(artifacts_dir, tier=tier)
        TemplateStoreLifecycleOps._stamp_manifest(artifacts_dir, schema, tier=tier, last_action=last_action)
        if has_remap or has_surgery:
            TemplateStoreLifecycleOps.apply_structural_migration_from_schema_diff(
                artifacts_dir, schema_diff, schema=schema
            )
        reconcile_sidecar_against_graph(schema, EngineConfig.SCHEMA_JSON_PATH)
        return MigrationReport(
            tier=tier,
            renamed_tables=renamed_tables,
            renamed_columns=renamed_columns,
            destroyed_templates=destroyed,
            remapped_templates=remapped,
            surgically_invalidated=surgically,
            dropped_tables=tuple(schema_diff.dropped_tables),
            value_type_changed_columns=TemplateStoreLifecycleOps._value_type_changes_from_diff(schema_diff),
        )

    @staticmethod
    def template_store_base_dir(artifacts_dir: str) -> str:
        """Return the ``intent_templates`` root under *artifacts_dir*."""
        return os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)

    @staticmethod
    def validate_space_name(space_name: str) -> str:
        """Return a normalized lowercase space segment or raise ``ValueError`` when unsafe. Accepts display names and allocator uids such as ``S0001`` (normalized to ``s0001``) so template partitions can key by space uid on case-insensitive filesystems."""
        raw = str(space_name).strip()
        if not raw or raw in (".", ".."):
            raise ValueError(f"invalid template space name: {space_name!r}")
        lower = raw.lower()
        if "/" in lower or "\\" in lower:
            raise ValueError(f"invalid template space name: {space_name!r}")
        try:
            sanitized = sanitize_name_segment(lower)
        except ValueError:
            raise ValueError(f"invalid template space name: {space_name!r}") from None
        if sanitized != lower:
            raise ValueError(f"invalid template space name: {space_name!r}")
        return lower

    @staticmethod
    def template_store_dir_for_space(artifacts_dir: str, space_name: str = MASTER_AETHERSPACE_NAME) -> str:
        """Return the partitioned template-store directory for one aetherspace namespace."""
        safe = TemplateStoreLifecycleOps.validate_space_name(space_name)
        return os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT, safe
        )

    @staticmethod
    def artifacts_dir_for_template_store(store_dir: str) -> str:
        """Resolve the engine storage directory from a nested per-space template path."""
        p = Path(store_dir).resolve()
        for parent in [p, *p.parents]:
            if parent.name == TEMPLATE_STORE_SEGMENT:
                return str(parent.parent)
        return str(p.parent)

    @staticmethod
    def _flat_template_store_dir(artifacts_dir: str) -> str:
        return TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir)

    @staticmethod
    def _migrate_unspaced_templates_to_master(artifacts_dir: str) -> bool:
        """Move a flat unspaced template store into ``spaces/master`` (idempotent)."""
        flat_dir = TemplateStoreLifecycleOps._flat_template_store_dir(artifacts_dir)
        master_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(artifacts_dir, MASTER_AETHERSPACE_NAME)
        flat_header = os.path.join(flat_dir, TEMPLATE_STORE_HEADER_FILENAME)
        master_header = os.path.join(master_dir, TEMPLATE_STORE_HEADER_FILENAME)
        if os.path.isfile(master_header) or not os.path.isfile(flat_header):
            return False
        os.makedirs(master_dir, exist_ok=True)
        for entry in os.listdir(flat_dir):
            if entry == TEMPLATE_STORE_SPACES_SEGMENT:
                continue
            if entry == TEMPLATE_STORE_HEADER_FILENAME or entry.startswith(TEMPLATE_STORE_PARTITION_PREFIX):
                src = os.path.join(flat_dir, entry)
                dst = os.path.join(master_dir, entry)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    shutil.move(src, dst)
        return True

    @staticmethod
    def _artifacts_dir_is_federation_tree(artifacts_dir: str) -> bool:
        return os.path.isfile(os.path.join(artifacts_dir, FEDERATION_MANIFEST_FILENAME))

    @staticmethod
    def ensure_template_store_space_layout(artifacts_dir: str) -> bool:
        """Ensure per-space template layout exists; migrate flat unspaced stores once."""
        return TemplateStoreLifecycleOps._migrate_unspaced_templates_to_master(artifacts_dir)

    @staticmethod
    def iter_template_store_space_dirs(artifacts_dir: str) -> Iterator[tuple[str, str]]:
        """Yield ``(space_name, store_dir)`` for every on-disk template namespace."""
        TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
        spaces_root = os.path.join(
            TemplateStoreLifecycleOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT
        )
        if os.path.isdir(spaces_root):
            for name in sorted(os.listdir(spaces_root)):
                store_dir = os.path.join(spaces_root, name)
                header = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
                if os.path.isdir(store_dir) and os.path.isfile(header):
                    yield name, store_dir
            return
        flat_dir = TemplateStoreLifecycleOps._flat_template_store_dir(artifacts_dir)
        flat_header = os.path.join(flat_dir, TEMPLATE_STORE_HEADER_FILENAME)
        if os.path.isfile(flat_header):
            yield MASTER_AETHERSPACE_NAME, flat_dir

    @staticmethod
    def purge_space_learning_partition(artifacts_dir: str, space_name: str) -> bool:
        """Remove the on-disk template-store partition for one named aetherspace."""
        norm = TemplateStoreLifecycleOps.validate_space_name(space_name)
        if norm == MASTER_AETHERSPACE_NAME:
            raise ValueError(f"cannot purge learning partition for {space_name!r}")
        space_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(artifacts_dir, norm)
        if not os.path.isdir(space_dir):
            return False
        shutil.rmtree(space_dir, ignore_errors=True)
        return True

    @staticmethod
    def _merge_value_history_row_fp(question: str, param_values: Mapping[str, Any]) -> tuple[str, str]:
        pv_items = sorted((str(k), str(v)) for k, v in param_values.items())
        return question, stable_json(pv_items)

    @staticmethod
    def _merge_template_value_histories_master_first(keeper: Template, donor: Template) -> None:
        """Union donor value history into *keeper* with master-first retention (cases 17–18)."""
        limits = TemplateStoreLifecycleOps._resolve_engine_limits()
        cap = limits.template_value_history_depth
        seen: dict[tuple[str, str], int] = {}
        for idx, q in enumerate(keeper.value_history.questions):
            pv = keeper.value_history.param_values[idx] if idx < len(keeper.value_history.param_values) else {}
            seen[TemplateStoreLifecycleOps._merge_value_history_row_fp(q, pv)] = idx
        for idx, q in enumerate(donor.value_history.questions):
            pv = donor.value_history.param_values[idx] if idx < len(donor.value_history.param_values) else {}
            nl = donor.value_history.natural_language[idx] if idx < len(donor.value_history.natural_language) else ""
            ac = donor.value_history.accept_counts[idx] if idx < len(donor.value_history.accept_counts) else 1
            fp = TemplateStoreLifecycleOps._merge_value_history_row_fp(q, pv)
            if fp in seen:
                keeper_idx = seen[fp]
                keeper.value_history.accept_counts[keeper_idx] += ac
                continue
            keeper.value_history.append_question_variant(
                q, accept_count=ac, param_values=dict(pv), natural_language=nl, dedup=False
            )
            seen[fp] = len(keeper.value_history.questions) - 1
        overflow = len(keeper.value_history.questions) - cap
        if overflow > 0:
            keeper.value_history.questions = keeper.value_history.questions[:cap]
            keeper.value_history.param_values = keeper.value_history.param_values[:cap]
            keeper.value_history.natural_language = keeper.value_history.natural_language[:cap]
            keeper.value_history.accept_counts = keeper.value_history.accept_counts[:cap]

    @staticmethod
    def _merge_template_stats_and_trust(keeper: Template, donor: Template) -> None:
        keeper.stats.accept += int(donor.stats.accept)
        keeper.stats.reject += int(donor.stats.reject)
        keeper.trust_level = max(int(keeper.trust_level), int(donor.trust_level))
        for q, counts in donor.feedback_by_question.items():
            if q in keeper.feedback_by_question:
                existing = keeper.feedback_by_question[q]
                keeper.feedback_by_question[q] = FeedbackCounts(
                    accepts=int(existing.accepts) + int(counts.accepts),
                    rejects=int(existing.rejects) + int(counts.rejects),
                    last_path=max(int(existing.last_path), int(counts.last_path)),
                )
            else:
                keeper.feedback_by_question[q] = FeedbackCounts(
                    accepts=int(counts.accepts),
                    rejects=int(counts.rejects),
                    last_path=int(counts.last_path),
                )

    @staticmethod
    def has_any_rejection_history_for_question(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, schema_graph_id: str | None = None
    ) -> bool:
        """Return True when ``question_feedback`` has any row for a fuzzy-matching question key."""
        if isinstance(store, TemplateStoreView):
            for q_key in store.feedback_shard_index:
                if not is_exact_question_text_match(q_norm, str(q_key)):
                    continue
                for row in store._get_feedback_rows(q_key):
                    if schema_graph_id is not None:
                        ent = QuestionFeedbackEntry.from_dict(row)
                        row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                        if row_graph_id and row_graph_id != schema_graph_id:
                            continue
                        if not row_graph_id and ent.effective_structural_hash != schema_graph_id:
                            continue
                    return True
            return False
        qf_raw = store.get("question_feedback")
        if not isinstance(qf_raw, dict):
            return False
        for q_key in qf_raw:
            if not is_exact_question_text_match(q_norm, str(q_key)):
                continue
            raw_rows = qf_raw.get(q_key)
            if not isinstance(raw_rows, list) or not raw_rows:
                continue
            if schema_graph_id is None:
                return True
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                ent = QuestionFeedbackEntry.from_dict(row)
                row_graph_id = str(row.get("schema_graph_id", row.get("effective_structural_hash", "")) or "")
                if row_graph_id and row_graph_id != schema_graph_id:
                    continue
                if not row_graph_id and ent.effective_structural_hash != schema_graph_id:
                    continue
                return True
        return False

    @staticmethod
    def _build_master_merge_indexes(
        master_templates: Mapping[str, Template],
    ) -> tuple[dict[str, list[Template]], dict[str, str], dict[tuple[str, str], list[Template]]]:
        q_to_templates: dict[str, list[Template]] = defaultdict(list)
        paraphrase_owner: dict[str, str] = {}
        intent_join_to_templates: dict[tuple[str, str], list[Template]] = defaultdict(list)
        for tmpl in master_templates.values():
            join_fp = TemplateRefs.join_fingerprint_from_concrete_intent(tmpl.intent_signature)
            intent_join_to_templates[(tmpl.intent_key, join_fp)].append(tmpl)
            for q in tmpl.value_history.questions:
                if not q:
                    continue
                q_to_templates[str(q)].append(tmpl)
                paraphrase_owner[str(q)] = tmpl.id
        return q_to_templates, paraphrase_owner, intent_join_to_templates

    @staticmethod
    def _space_template_merge_precheck(
        tmpl: Template,
        schema: SchemaGraph,
    ) -> str | None:
        if tmpl.federation_plan_only:
            return "federation_plan_only"
        if tmpl.approval_state == ApprovalState.PENDING:
            return "pending_approval"
        if tmpl.effective_structural_hash and tmpl.effective_structural_hash != schema.effective_structural_hash:
            return "structural_hash_mismatch"
        survives, _ = TemplateRefs.footprint_survives(tmpl, schema)
        if not survives:
            return "entity_absent"
        return None

    @staticmethod
    def _master_template_for_question(
        q_norm: str,
        master_templates: Mapping[str, Template],
        q_to_templates: Mapping[str, list[Template]],
    ) -> Template | None:
        matches = list(q_to_templates.get(q_norm, ()))
        if matches:
            return sorted(matches, key=lambda t: t.id)[0]
        resolved = resolve_template_for_question(q_norm, master_templates)
        if resolved is None:
            return None
        tmpl, _idx = resolved
        return tmpl

    @staticmethod
    def _fold_space_questions_into_master_paraphrases(
        master_tmpl: Template,
        space_tmpl: Template,
        *,
        paraphrase_owner: dict[str, str],
        counts: SpaceLearningMergeCounts,
    ) -> None:
        for idx, q in enumerate(space_tmpl.value_history.questions):
            if not q:
                continue
            owner = paraphrase_owner.get(q)
            if owner is not None and owner != master_tmpl.id:
                counts.paraphrase_mapping_dropped += 1
                continue
            pv = space_tmpl.value_history.param_values[idx] if idx < len(space_tmpl.value_history.param_values) else {}
            nl = (
                space_tmpl.value_history.natural_language[idx]
                if idx < len(space_tmpl.value_history.natural_language)
                else ""
            )
            ac = space_tmpl.value_history.accept_counts[idx] if idx < len(space_tmpl.value_history.accept_counts) else 0
            if q in master_tmpl.value_history.questions:
                continue
            master_tmpl.value_history.append_question_variant(
                q, accept_count=ac, param_values=dict(pv), natural_language=nl
            )
            paraphrase_owner[q] = master_tmpl.id

    @staticmethod
    def _merge_space_feedback_into_master(
        master_store: TemplateStoreView,
        space_store: TemplateStoreView,
        master_templates: dict[str, Template],
        q_to_templates: Mapping[str, list[Template]],
        counts: SpaceLearningMergeCounts,
    ) -> bool:
        changed = False
        for q_norm in sorted(space_store.feedback_shard_index.keys()):
            space_rows = list(space_store._get_feedback_rows(q_norm))
            if not space_rows:
                continue
            master_tmpl = TemplateStoreLifecycleOps._master_template_for_question(
                q_norm, master_templates, q_to_templates
            )
            master_rows = list(master_store._get_feedback_rows(q_norm))
            for row in space_rows:
                entry = QuestionFeedbackEntry.from_dict(row)
                if master_tmpl is not None and entry.kind is FeedbackKind.INTENT_REJECTED:
                    counts.feedback_discarded_master_wins += 1
                    continue
                duplicate_kind = False
                for existing_row in master_rows:
                    existing = QuestionFeedbackEntry.from_dict(existing_row)
                    if existing.kind == entry.kind and existing.intent_structural_hash == entry.intent_structural_hash:
                        duplicate_kind = True
                        break
                if duplicate_kind:
                    counts.feedback_discarded_master_wins += 1
                    continue
                entry_to_record = entry
                if entry.rejected_join_path_signature and master_tmpl is not None:
                    master_join_layer = TemplateRefs._canonical_join_path_layer(
                        list(master_tmpl.intent_signature.chosen_join_path_signature or [])
                    )
                    filtered_sig: list[str] = []
                    for sig in entry.rejected_join_path_signature:
                        sig_layer = TemplateRefs._canonical_join_path_layer([sig])
                        if sig_layer and master_join_layer and sig_layer == master_join_layer:
                            counts.feedback_join_rejection_dropped += 1
                            continue
                        filtered_sig.append(sig)
                    if not filtered_sig and not entry.summary.strip():
                        counts.feedback_discarded_master_wins += 1
                        continue
                    entry_to_record = replace(entry, rejected_join_path_signature=tuple(filtered_sig))
                TemplateStoreLifecycleOps.record_question_feedback(master_store, q_norm, entry_to_record)
                master_rows.append(entry_to_record.to_dict())
                counts.feedback_carried += 1
                changed = True
        return changed

    @staticmethod
    def merge_space_learning_into_master(
        artifacts_dir: str,
        space_name: str,
        schema_graph_id: str,
        schema: SchemaGraph,
    ) -> SpaceLearningMergeResult:
        """Promote space templates and question feedback into master using the exhaustive collision matrix. Multi-space deletes should process spaces in ascending uid order so each merge sees the prior result (case 21). Master is written atomically via :meth:`save_template_store`; the caller should purge the space partition only after this returns."""
        counts = SpaceLearningMergeCounts()
        norm = TemplateStoreLifecycleOps.validate_space_name(space_name)
        if norm == MASTER_AETHERSPACE_NAME:
            return SpaceLearningMergeResult(changed=False, counts=counts)
        TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
        space_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(artifacts_dir, norm)
        space_header = os.path.join(space_dir, TEMPLATE_STORE_HEADER_FILENAME)
        if not os.path.isfile(space_header):
            return SpaceLearningMergeResult(changed=False, counts=counts)

        master_store = TemplateStoreLifecycleOps.load_template_store(
            schema_graph_id, schema, artifacts_dir=artifacts_dir, space_name=MASTER_AETHERSPACE_NAME
        )
        space_store = TemplateStoreLifecycleOps.load_template_store(
            schema_graph_id, schema, artifacts_dir=artifacts_dir, space_name=norm
        )
        master_templates = dict(TemplateStoreLifecycleOps.store_to_templates(master_store))
        space_templates = dict(TemplateStoreLifecycleOps.store_to_templates(space_store))
        q_to_templates, paraphrase_owner, intent_join_to_templates = (
            TemplateStoreLifecycleOps._build_master_merge_indexes(master_templates)
        )

        changed = False
        for tmpl in sorted(space_templates.values(), key=lambda t: t.id):
            precheck = TemplateStoreLifecycleOps._space_template_merge_precheck(tmpl, schema)
            if precheck == "federation_plan_only":
                counts.dropped_federation_plan_only += 1
                continue
            if precheck == "pending_approval":
                counts.dropped_pending_approval += 1
                continue
            if precheck == "structural_hash_mismatch":
                counts.dropped_structural_hash_mismatch += 1
                continue
            if precheck == "entity_absent":
                counts.dropped_entity_absent += 1
                continue

            space_q = TemplateStoreLifecycleOps.primary_template_q_norm(tmpl)
            space_join_fp = TemplateRefs.join_fingerprint_from_concrete_intent(tmpl.intent_signature)
            eff_hash = str(schema.effective_structural_hash or "") or None
            if space_q and TemplateStoreLifecycleOps.has_any_rejection_history_for_question(
                master_store, space_q, eff_hash
            ):
                counts.discarded_template_master_rejection += 1
                continue

            master_same_q = (
                TemplateStoreLifecycleOps._master_template_for_question(space_q, master_templates, q_to_templates)
                if space_q
                else None
            )
            if master_same_q is not None:
                master_join_fp = TemplateRefs.join_fingerprint_from_concrete_intent(master_same_q.intent_signature)
                if tmpl.intent_key == master_same_q.intent_key and space_join_fp == master_join_fp:
                    TemplateStoreLifecycleOps._merge_template_value_histories_master_first(master_same_q, tmpl)
                    TemplateStoreLifecycleOps._merge_template_stats_and_trust(master_same_q, tmpl)
                    counts.merged_same_identity += 1
                    changed = True
                    continue
                if tmpl.intent_key == master_same_q.intent_key:
                    counts.discarded_same_q_diff_join += 1
                    continue
                counts.discarded_same_q_diff_intent += 1
                continue

            same_intent_join = intent_join_to_templates.get((tmpl.intent_key, space_join_fp), ())
            if same_intent_join:
                master_tmpl = sorted(same_intent_join, key=lambda t: t.id)[0]
                before = len(master_tmpl.value_history.questions)
                TemplateStoreLifecycleOps._fold_space_questions_into_master_paraphrases(
                    master_tmpl,
                    tmpl,
                    paraphrase_owner=paraphrase_owner,
                    counts=counts,
                )
                if len(master_tmpl.value_history.questions) > before or counts.paraphrase_mapping_dropped:
                    counts.folded_paraphrase += 1
                    changed = True
                continue

            adopted = tmpl
            if tmpl.id in master_templates:
                adopted = replace(tmpl, id=TemplateStoreLifecycleOps._reserve_template_id(master_store))
                counts.id_reassigned += 1
            master_templates[adopted.id] = adopted
            join_fp = TemplateRefs.join_fingerprint_from_concrete_intent(adopted.intent_signature)
            intent_join_to_templates[(adopted.intent_key, join_fp)].append(adopted)
            for q in adopted.value_history.questions:
                if q:
                    q_to_templates[str(q)].append(adopted)
                    paraphrase_owner.setdefault(str(q), adopted.id)
            counts.carried_new_id += 1
            changed = True

        if TemplateStoreLifecycleOps._merge_space_feedback_into_master(
            master_store, space_store, master_templates, q_to_templates, counts
        ):
            changed = True

        if changed:
            TemplateStoreLifecycleOps._invoke_reconcile_template_store_until_stable(
                master_templates, template_store_view=master_store
            )
            TemplateStoreLifecycleOps.templates_to_store(master_store, master_templates)
            TemplateStoreLifecycleOps.save_template_store(master_store)
            disposition = counts.to_dict()
            if disposition:
                debug(f"[templates.merge_space_learning_into_master] space={norm!r} dispositions={disposition}")
                write_artifact_manifest(artifacts_dir, last_action="space_learning_merge")
        return SpaceLearningMergeResult(changed=changed, counts=counts)

    @staticmethod
    def empty_template_store_for_space(
        schema_graph_id: str, *, artifacts_dir: str | None = None, space_name: str = MASTER_AETHERSPACE_NAME
    ) -> TemplateStoreView:
        """Return a fresh in-memory partitioned template store for one aetherspace namespace."""
        if artifacts_dir is not None:
            store_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(artifacts_dir, space_name)
        else:
            store_dir = EngineConfig.TEMPLATE_STORE_DIR
        return TemplateStoreView.empty(store_dir, schema_graph_id)

    @staticmethod
    def empty_template_store(schema_graph_id: str) -> TemplateStoreView:
        """Return a fresh in-memory partitioned template store."""
        adir = TemplateStoreLifecycleOps.artifacts_dir_for_template_store(EngineConfig.TEMPLATE_STORE_DIR)
        return TemplateStoreLifecycleOps.empty_template_store_for_space(
            schema_graph_id, artifacts_dir=adir, space_name=MASTER_AETHERSPACE_NAME
        )

    @staticmethod
    def _normalize_loaded_template_store_document(d: dict[str, Any]) -> None:
        """Remove superseded top-level keys and ensure feedback shard index exists."""
        for stale_key in (
            "rejected_templates",
            "rejected_intents",
            "intent_failure_log",
            "next_reject_id",
            "next_rejected_intent_id",
            "next_intent_failure_id",
        ):
            d.pop(stale_key, None)
        d.setdefault(FEEDBACK_SHARD_INDEX_KEY, {})
        d.pop("question_feedback", None)
        d.pop("negative_memory", None)
        d.pop("semantic_rejections", None)

    @staticmethod
    def _load_partitioned_view_unlocked(store_dir: str) -> TemplateStoreView | None:
        """Read header only and return a view (no hash reconciliation)."""
        header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        if not os.path.isfile(header_path):
            return None
        try:
            header = read_gzip_json(header_path)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(header, dict):
            return None
        return TemplateStoreView.from_header_payload(store_dir, header)

    @staticmethod
    def load_template_store(
        schema_graph_id: str,
        schema: SchemaGraph | None = None,
        *,
        space_name: str = MASTER_AETHERSPACE_NAME,
        artifacts_dir: str | None = None,
    ) -> TemplateStoreView:
        """Load template store from disk or create an empty store. When the. on-disk ``schema_graph_id`` matches, returns a :class:`TemplateStoreView` backed by the header and lazy partition files. Otherwise, if *schema* is provided, reconciles rows in place, stamps the new id, rewrites shards, and returns the mutated view. Without *schema*, returns a fresh empty view."""
        if PolicyConfig.REGENERATE_TEMPLATE_STORE:
            debug("[templates.load_template_store] REGENERATE_TEMPLATE_STORE: returning empty store")
            return TemplateStoreLifecycleOps.empty_template_store_for_space(
                schema_graph_id, artifacts_dir=artifacts_dir, space_name=space_name
            )

        if artifacts_dir is not None:
            TemplateStoreLifecycleOps.ensure_template_store_space_layout(artifacts_dir)
            store_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(artifacts_dir, space_name)
            adir = artifacts_dir
        else:
            adir = TemplateStoreLifecycleOps.artifacts_dir_for_template_store(EngineConfig.TEMPLATE_STORE_DIR)
            TemplateStoreLifecycleOps.ensure_template_store_space_layout(adir)
            store_dir = TemplateStoreLifecycleOps.template_store_dir_for_space(adir, space_name)

        with artifact_lock(store_dir):
            return TemplateStoreLifecycleOps._load_template_store_locked(store_dir, adir, schema_graph_id, schema)

    @staticmethod
    def _load_template_store_locked(
        store_dir: str, adir: str, schema_graph_id: str, schema: SchemaGraph | None
    ) -> TemplateStoreView:
        """Body of :func:`load_template_store` executed under the artifacts- dir lock."""
        header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        monolithic_path = os.path.join(adir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
        if not os.path.isfile(header_path):
            if os.path.isfile(monolithic_path):
                debug(
                    f"[templates.load_template_store] monolithic_store_ignored: path={monolithic_path} "
                    "(partitioned header missing; not migrated)"
                )
            debug(f"[templates.load_template_store] no_header: path={header_path}")
            return TemplateStoreView.empty(store_dir, schema_graph_id)
        try:
            header = read_gzip_json(header_path)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
            debug(f"[templates.load_template_store] corrupt_or_unreadable: path={header_path} err={exc!r}")
            write_artifact_manifest(
                adir, last_corruption_at=datetime.now(UTC).isoformat(), last_action="corrupt_template_store"
            )
            return TemplateStoreView.empty(store_dir, schema_graph_id)
        if not isinstance(header, dict):
            return TemplateStoreView.empty(store_dir, schema_graph_id)

        header_fmt = header.get("format_version")
        if not format_versions_match(header_fmt, TEMPLATE_STORE_FORMAT_VERSION):
            raise ConfigError(
                f"template store header at {header_path!r} has "
                f"format_version {header_fmt!r}; this build expects "
                f"{TEMPLATE_STORE_FORMAT_VERSION}. Delete the template store "
                f"directory {store_dir!r} (or the engine artifacts directory) "
                f"and re-run initialize_aether_engine so the store is rebuilt "
                f"from scratch."
            )

        index_keys = (
            SHAPE_QUESTION_INDEX_KEY,
            TEMPLATE_INTENT_KEY_INDEX_KEY,
            TEMPLATE_UNION_FAMILY_INDEX_KEY,
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
        )
        need_index_rebuild = any(k not in header or not isinstance(header.get(k), dict) for k in index_keys)
        stored_qnorm_ver = header.get(QUESTION_NORMALIZATION_VERSION_KEY)
        if not format_versions_match(stored_qnorm_ver, QUESTION_NORMALIZATION_VERSION):
            need_index_rebuild = True

        stored = str(header.get("schema_graph_id", header.get("effective_structural_hash", "")) or "")
        header_qf_raw = header.get("question_feedback")
        header_qf = header_qf_raw if isinstance(header_qf_raw, dict) and header_qf_raw else None
        view = TemplateStoreView.from_header_payload(store_dir, header)
        if header_qf:
            view._import_question_feedback_rows(header_qf)
            TemplateStoreLifecycleOps._persist_template_store_view(view, adir)
            prev = TemplateStoreLifecycleOps.read_store_manifest(adir)
            write_artifact_manifest(
                adir,
                structural_hash=prev.structural_hash if prev else "",
                profiling_hash=prev.profiling_hash if prev else "",
                scope_hash=prev.scope_hash if prev else "",
                effective_structural_hash=prev.effective_structural_hash if prev else "",
                schema_graph_id=prev.schema_graph_id if prev else "",
                notes_hash=prev.notes_hash if prev else "",
                semantic_edges_hash=prev.semantic_edges_hash if prev else "",
                last_migration_tier=prev.last_migration_tier if prev else "",
                last_migration_at=prev.last_migration_at if prev else "",
                last_action="question_feedback_sharding",
            )
        if need_index_rebuild:
            TemplateStoreView.refresh_template_store_indexes(view)
        if TemplateStoreLifecycleOps._repair_cross_shard_inconsistency(view):
            TemplateStoreLifecycleOps._persist_template_store_view(view, adir)

        if stored == schema_graph_id:
            qfn = view._count_question_feedback_rows()
            debug(
                f"[templates.load_template_store] loaded: templates={len(view.partition_map)} question_feedback_rows={qfn}"
            )
            return view
        if schema is None:
            debug("[templates.load_template_store] schema_graph_id_mismatch: orphaning shards")
            TemplateStoreLifecycleOps.orphan_mismatched_template_store(
                store_dir,
                old_schema_graph_id=stored,
                new_schema_graph_id=schema_graph_id,
            )
            return TemplateStoreView.empty(store_dir, schema_graph_id)
        history = TemplateStoreLifecycleOps.read_rename_history(adir)
        try:
            chain_tables, chain_columns = TemplateStoreLifecycleOps.compose_rename_chain(
                history, stored, schema_graph_id
            )
        except BrokenRenameChainError as exc:
            debug(f"[templates.load_template_store] broken rename chain: {exc}; reconciling live templates")
            TemplateStoreLifecycleOps.reconcile_template_store(view, schema)
            view.schema_graph_id = schema_graph_id
            TemplateStoreLifecycleOps._persist_template_store_view(view, adir)
            return view
        debug("[templates.load_template_store] schema_graph_id_mismatch: reconciling via rename chain")
        if chain_tables or chain_columns:
            remapped, destroyed = TemplateStoreLifecycleOps._remap_templates_in_view(
                view, schema, chain_tables, chain_columns
            )
            debug(
                f"[templates.load_template_store] chain_remap remapped={remapped} destroyed={destroyed} "
                f"from={stored!r} to={schema_graph_id!r}"
            )
        else:
            TemplateStoreLifecycleOps.reconcile_template_store(view, schema)
            view.schema_graph_id = schema_graph_id
        TemplateStoreLifecycleOps._persist_template_store_view(view, adir)
        return view

    @staticmethod
    def _persist_template_store_view(view: TemplateStoreView, adir: str) -> None:
        if TemplateStoreLifecycleOps._prune_template_store_size(view):
            debug("[templates._persist_template_store_view] pruned store to policy limits")
        TemplateStoreView.refresh_template_store_indexes(view)
        TemplateStoreLifecycleOps._save_template_store_unlocked(view, adir)

    @staticmethod
    def _save_template_store_unlocked(store: TemplateStoreView, adir: str) -> None:
        store_dir = store._store_dir
        os.makedirs(store_dir, exist_ok=True)
        staging_dir = os.path.join(store_dir, ".write_staging")
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)
        fb_staging_root = os.path.join(staging_dir, TEMPLATE_STORE_FEEDBACK_SEGMENT)
        os.makedirs(fb_staging_root, exist_ok=True)
        header_doc = TemplateStoreLifecycleOps._convert_to_json_serializable(store._header_document())
        TemplateStoreLifecycleOps._debug_check_types(header_doc, "store")
        dirty = set(store._dirty_partitions)
        staged_commits: list[tuple[str, str, bool]] = []
        fb_staged_commits: list[tuple[str, str, bool]] = []
        try:
            for part in sorted(dirty):
                payload = store._partition_cache.get(part)
                if payload is None:
                    payload = {}
                    live_path = store._partition_file_path(part)
                    if os.path.isfile(live_path):
                        try:
                            disk = read_gzip_json(live_path)
                            payload = disk if isinstance(disk, dict) else {}
                        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                            payload = {}
                part_name = f"{TEMPLATE_STORE_PARTITION_PREFIX}{part:02x}.json.gz"
                staging_path = os.path.join(staging_dir, part_name)
                live_path = store._partition_file_path(part)
                if not payload:
                    staged_commits.append((staging_path, live_path, True))
                    continue
                serial = TemplateStoreLifecycleOps._convert_to_json_serializable(dict(payload))
                write_gzip_json_atomic(staging_path, serial, sort_keys=True)
                staged_commits.append((staging_path, live_path, False))
            for part in sorted(store._dirty_feedback_partitions):
                fb_payload = store._feedback_partition_cache.get(part)
                if fb_payload is None:
                    fb_payload = {}
                    fb_live_path = store._feedback_partition_file_path(part)
                    if os.path.isfile(fb_live_path):
                        try:
                            disk = read_gzip_json(fb_live_path)
                            fb_payload = disk if isinstance(disk, dict) else {}
                        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                            fb_payload = {}
                fb_part_name = f"{TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX}{part:02x}.json.gz"
                fb_staging_path = os.path.join(fb_staging_root, fb_part_name)
                fb_live_path = store._feedback_partition_file_path(part)
                if not fb_payload:
                    fb_staged_commits.append((fb_staging_path, fb_live_path, True))
                    continue
                fb_serial = TemplateStoreLifecycleOps._convert_to_json_serializable(dict(fb_payload))
                write_gzip_json_atomic(fb_staging_path, fb_serial, sort_keys=True)
                fb_staged_commits.append((fb_staging_path, fb_live_path, False))
            header_staging = os.path.join(staging_dir, TEMPLATE_STORE_HEADER_FILENAME)
            write_gzip_json_atomic(header_staging, header_doc, sort_keys=True)
            header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
            replace_path_atomic(header_staging, header_path)
            for staging_path, live_path, delete_live in staged_commits:
                if delete_live:
                    if os.path.isfile(live_path):
                        try:
                            os.remove(live_path)
                        except OSError:
                            pass
                    continue
                replace_path_atomic(staging_path, live_path)
            os.makedirs(store._feedback_dir(), exist_ok=True)
            for fb_staging_path, fb_live_path, delete_live in fb_staged_commits:
                if delete_live:
                    if os.path.isfile(fb_live_path):
                        try:
                            os.remove(fb_live_path)
                        except OSError:
                            pass
                    continue
                replace_path_atomic(fb_staging_path, fb_live_path)
            store._dirty_partitions.clear()
            store._dirty_feedback_partitions.clear()
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
        manifest_fields = TemplateStoreLifecycleOps._manifest_fingerprints_for_template_save(
            adir, str(store.schema_graph_id or "")
        )
        write_artifact_manifest(
            adir,
            structural_hash=manifest_fields["structural_hash"],
            profiling_hash=manifest_fields["profiling_hash"],
            scope_hash=manifest_fields["scope_hash"],
            effective_structural_hash=manifest_fields["effective_structural_hash"],
            schema_graph_id=manifest_fields["schema_graph_id"],
            notes_hash=manifest_fields["notes_hash"],
            semantic_edges_hash=manifest_fields["semantic_edges_hash"],
            last_migration_tier=manifest_fields["last_migration_tier"],
            last_migration_at=manifest_fields["last_migration_at"],
            last_action="template_store_save",
        )

    @staticmethod
    def _merge_pending_template_store_save(disk: TemplateStoreView, pending: TemplateStoreView) -> None:
        """Overlay *pending* dirty shards onto *disk* before a locked persist."""
        if pending.schema_graph_id:
            disk.schema_graph_id = pending.schema_graph_id
        disk.next_id = max(int(disk.next_id), int(pending.next_id))

        with pending._partition_cache_lock:
            dirty_parts = set(pending._dirty_partitions)
            for part in dirty_parts:
                payload = pending._partition_cache.get(part)
                if payload is None:
                    payload = pending._load_partition_payload(part)
                pending_ids = {str(tid) for tid, raw in payload.items() if isinstance(raw, dict)}
                with disk._partition_cache_lock:
                    disk_payload = disk._load_partition_payload(part)
                    for tid in list(disk_payload):
                        if str(tid) not in pending_ids:
                            disk_payload.pop(tid, None)
                            disk.partition_map.pop(str(tid), None)
                    for tid, raw in payload.items():
                        tid_s = str(tid)
                        disk_payload[tid_s] = dict(raw)
                        disk.partition_map[tid_s] = part
                    disk._partition_cache[part] = disk_payload
                    disk._dirty_partitions.add(part)

        with pending._feedback_partition_cache_lock:
            for part in set(pending._dirty_feedback_partitions):
                fb_payload = pending._feedback_partition_cache.get(part)
                if fb_payload is None:
                    fb_payload = pending._load_feedback_partition_payload(part)
                with disk._feedback_partition_cache_lock:
                    disk_fb_payload = disk._load_feedback_partition_payload(part)
                    for q_norm, rows in fb_payload.items():
                        disk_fb_payload[q_norm] = [dict(row) for row in rows if isinstance(row, dict)]
                        disk.feedback_shard_index[str(q_norm)] = part
                    disk._feedback_partition_cache[part] = disk_fb_payload
                    disk._dirty_feedback_partitions.add(part)

    @staticmethod
    def save_template_store(store: dict[str, Any] | TemplateStoreView) -> None:
        """Save template store to disk. Converts all non-serialisable objects (for example, sets) before writing JSON. For a :class:`TemplateStoreView`, stages dirty shards and the header under a temp directory, then commits the header before shard bodies so a mid-commit crash cannot expose new bodies under an old header."""
        if not isinstance(store, TemplateStoreView):
            raise TypeError("save_template_store requires a TemplateStoreView")
        store_dir = store._store_dir

        qfn = store._count_question_feedback_rows()
        debug(
            f"[templates.save_template_store] saving: templates={len(store.partition_map)} question_feedback_rows={qfn}"
        )
        if TemplateStoreLifecycleOps._prune_template_store_size(store):
            debug("[templates.save_template_store] pruned store to policy limits")
        adir = TemplateStoreLifecycleOps.artifacts_dir_for_template_store(store_dir)
        with artifact_lock(store_dir):
            disk = TemplateStoreLifecycleOps._load_partitioned_view_unlocked(store_dir)
            if disk is None:
                disk = TemplateStoreView.empty(store_dir, str(store.schema_graph_id or ""))
            TemplateStoreLifecycleOps._merge_pending_template_store_save(disk, store)
            TemplateStoreLifecycleOps._persist_template_store_view(disk, adir)
            store._dirty_partitions.clear()
            store._dirty_feedback_partitions.clear()
        debug(f"[templates.save_template_store] complete: dir={store_dir}")

    @staticmethod
    def store_to_templates(store: dict[str, Any] | TemplateStoreView) -> dict[str, Template] | LazyTemplateMapping:
        """Convert store dict to `Template` objects with nested dataclass. reconstruction."""
        if isinstance(store, TemplateStoreView):
            return LazyTemplateMapping(store)
        out = {}
        for tid, v in store.get("templates", {}).items():
            if not isinstance(v, dict):
                continue
            t = TemplateStoreView._template_from_store_dict(str(tid), v)
            if t is not None:
                out[tid] = t
        return out

    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        """Recursively convert values to JSON-serialisable forms (e.g. sets. to lists)."""
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: TemplateStoreLifecycleOps._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [TemplateStoreLifecycleOps._convert_to_json_serializable(item) for item in obj]
        return obj

    @staticmethod
    def _debug_check_types(obj: Any, path: str = "root") -> None:
        """Log debug messages when non-JSON types (e.g. sets) appear in. nested structures."""
        if isinstance(obj, set):
            debug(f"[templates.debug_check_types] found_set: path={path}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                TemplateStoreLifecycleOps._debug_check_types(v, f"{path}.{k}")
        elif isinstance(obj, list | tuple):
            for i, item in enumerate(obj):
                TemplateStoreLifecycleOps._debug_check_types(item, f"{path}[{i}]")

    @staticmethod
    def templates_to_store(
        store: dict[str, Any] | TemplateStoreView, templates: Mapping[str, Template]
    ) -> dict[str, Any] | TemplateStoreView:
        """Convert Template objects to store dict format."""
        debug(f"[templates.templates_to_store] converting: count={len(templates)}")
        for tid, t in templates.items():
            template_dict = t.to_dict()
            TemplateStoreLifecycleOps._debug_check_types(template_dict, f"template[{tid}]")
        store["templates"] = {
            k: TemplateStoreLifecycleOps._convert_to_json_serializable(v.to_dict()) for k, v in templates.items()
        }
        TemplateStoreView.refresh_template_store_indexes(store, template_objs=list(templates.values()))
        return store

    @staticmethod
    def primary_template_q_norm(template: Template) -> str:
        """Return the most frequent stored question row for *template*."""
        vh = template.value_history
        if not vh.questions:
            return ""
        return Counter(vh.questions).most_common(1)[0][0]

    @staticmethod
    def record_question_feedback(
        store: dict[str, Any] | TemplateStoreView, q_norm: str, entry: QuestionFeedbackEntry
    ) -> None:
        """Merge or append one ``QuestionFeedbackEntry`` under *q_norm* using kind-specific deduplication. ``VALIDATION_FAILURE`` rows deduplicate per ``intent_structural_hash`` only. ``INTENT_REJECTED`` rows deduplicate per ``intent_structural_hash`` and rejection bucket, merging summaries when a new bucket appears for the same hash."""
        if not entry.intent_structural_hash:
            return
        if isinstance(store, TemplateStoreView):
            qf = store.question_feedback
        else:
            qf = store.setdefault("question_feedback", {})
        raw_cur = qf.setdefault(q_norm, [])
        cur: list[Any] = cast(list[Any], raw_cur)
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
            if incoming_bucket in existing.buckets:
                return
            merged_summary = TemplateStoreLifecycleOps._combine_feedback_summaries(
                existing.summary, entry.summary, intent_payload=entry.intent_payload or existing.intent_payload
            )
            merged_sig = tuple(
                dict.fromkeys((*existing.rejected_join_path_signature, *entry.rejected_join_path_signature))
            )
            cur[i] = replace(
                existing,
                summary=merged_summary,
                buckets=(*existing.buckets, incoming_bucket),
                kind=FeedbackKind.INTENT_REJECTED,
                intent_payload=entry.intent_payload or existing.intent_payload,
                updated_at=TemplateStoreLifecycleOps._feedback_iso_now(),
                is_post_restart=existing.is_post_restart or entry.is_post_restart,
                source=existing.source,
                rejected_join_path_signature=merged_sig,
            ).to_dict()
            maxpq = PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION
            if len(cur) > maxpq:
                cur[:] = cur[-maxpq:]
            return
        cur.append(entry.to_dict())
        maxpq = PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION
        if len(cur) > maxpq:
            cur[:] = cur[-maxpq:]

    @staticmethod
    def _migration_map_checkpoint_restore(
        artifacts_dir: str,
        checkpoint_dir: str,
        *,
        schema_json_path: Path | None = None,
    ) -> None:
        manifest_src = os.path.join(checkpoint_dir, ARTIFACT_MANIFEST_FILENAME)
        manifest_dst = os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, manifest_dst)
        store_dst = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
        store_src = os.path.join(checkpoint_dir, TEMPLATE_STORE_SEGMENT)
        if os.path.isdir(store_dst):
            shutil.rmtree(store_dst, ignore_errors=True)
        if os.path.isdir(store_src):
            shutil.copytree(store_src, store_dst)
        schema_src = os.path.join(checkpoint_dir, MIGRATION_CHECKPOINT_SCHEMA_BASENAME)
        if schema_json_path is not None:
            if os.path.isfile(schema_src):
                shutil.copy2(schema_src, schema_json_path)
            elif os.path.isfile(schema_json_path):
                os.unlink(schema_json_path)
        aetherspaces_dst = os.path.join(artifacts_dir, AETHERSPACES_SEGMENT)
        aetherspaces_src = os.path.join(checkpoint_dir, AETHERSPACES_SEGMENT)
        if os.path.isdir(aetherspaces_dst):
            shutil.rmtree(aetherspaces_dst, ignore_errors=True)
        if os.path.isdir(aetherspaces_src):
            shutil.copytree(aetherspaces_src, aetherspaces_dst)

    @staticmethod
    def _migration_map_checkpoint_cleanup(checkpoint_dir: str | None) -> None:
        if checkpoint_dir:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    @staticmethod
    def _feedback_iso_now() -> str:
        """Return an ISO-8601 UTC timestamp string for feedback rows."""
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _combine_feedback_summaries(existing: str, incoming: str, *, intent_payload: str) -> str:
        """Merge two feedback summaries for the same intent-structure key using one short LLM response."""
        a = (existing or "").strip()
        b = (incoming or "").strip()
        if not a:
            return b
        if not b:
            return a
        if not EngineConfig.llm_credentials_configured():
            return f"{a}\n{b}".strip()
        system = (
            "You merge two brief text-to-SQL feedback summaries into one 3-6 line ASCII block. "
            "Each line states a structural issue without SQL. "
            'Respond as JSON only: {"summary":"..."}.'
        )
        user = json.dumps(
            {"existing": a, "incoming": b, "intent_structure_json": intent_payload}, ensure_ascii=False, sort_keys=True
        )
        raw = LLMProvider.chat(system, user, task="feedback", max_retries=1, timeout=20.0)
        try:
            data = json.loads(raw)
        except Exception as exc:
            debug(f"[templates._combine_feedback_summaries] json coerce: {exc}")
            return f"{a}\n{b}".strip()
        merged = str(data.get("summary", "")).strip()
        return merged if merged else f"{a}\n{b}".strip()

    @staticmethod
    def _scrub_feedback_store_for_sensitivity_ratchet(store: Any, schema: SchemaGraph) -> int:
        dropped = 0
        for q_norm in list(store.feedback_shard_index.keys()):
            rows = store._get_feedback_rows_mut(q_norm)
            kept: list[dict[str, Any]] = []
            for row in rows:
                entry = QuestionFeedbackEntry.from_dict(row)
                haystacks = (entry.summary, entry.intent_payload)
                if any(
                    DomainKnowledgeEntry.sensitive_column_references(str(text), schema)
                    for text in haystacks
                    if str(text).strip()
                ):
                    dropped += 1
                    continue
                kept.append(row)
            if kept:
                rows[:] = kept
            else:
                store._remove_feedback_question_key(q_norm)
        return dropped

    @staticmethod
    def scrub_template_artifacts_for_sensitivity_ratchet(
        artifacts_dir: str, schema: SchemaGraph, graph_id: str
    ) -> tuple[int, int]:
        templates_dropped = 0
        feedback_dropped = 0
        for space_name, _store_dir in TemplateStoreLifecycleOps.iter_template_store_space_dirs(artifacts_dir):
            store = TemplateStoreLifecycleOps.load_template_store(
                graph_id,
                schema,
                artifacts_dir=artifacts_dir,
                space_name=space_name,
            )
            before_templates = sum(1 for _ in store.iter_templates_by_partition())
            reconcile = TemplateStoreLifecycleOps.reconcile_template_store(store, schema)
            space_templates_dropped = len(reconcile.dropped_template_ids)
            space_feedback_dropped = TemplateStoreLifecycleOps._scrub_feedback_store_for_sensitivity_ratchet(
                store, schema
            )
            templates_dropped += space_templates_dropped
            feedback_dropped += space_feedback_dropped
            if space_templates_dropped or space_feedback_dropped or before_templates:
                TemplateStoreLifecycleOps.save_template_store(store)
        return templates_dropped, feedback_dropped


TemplateStoreView.register_templates_module(sys.modules[__name__])
register_sensitivity_ratchet_artifact_scrub(TemplateStoreLifecycleOps.scrub_template_artifacts_for_sensitivity_ratchet)
