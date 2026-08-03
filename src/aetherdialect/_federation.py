"""Federation: manifest, composite graph, planning, and coordinator execution."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pandas as pd
from platformdirs import user_data_dir
from sqlalchemy import text

import duckdb

from ._config import CsvRuntimeConfig, EngineRuntimeConfig, PolicyConfig
from ._constants import (
    ARROW_RESULT_READER_KINDS,
    ARTIFACT_DIRECTORY_SEGMENT,
    ARTIFACT_MANIFEST_FILENAME,
    ASK_PHASE_I,
    DATA_QUALITY_SQL_RESERVED_WORDS,
    DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED,
    DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    FEDERATION_ARTIFACT_FORMAT_VERSION,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_COMPOSITE_RECONCILIATION_NOTE,
    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
    FEDERATION_COMPOSITION_PHASE_A,
    FEDERATION_COMPOSITION_PHASE_B,
    FEDERATION_COMPOSITION_PHASE_C,
    FEDERATION_COMPOSITION_PHASE_D,
    FEDERATION_COMPOSITION_PHASE_E,
    FEDERATION_COMPOSITION_PHASE_F,
    FEDERATION_COMPOSITION_PHASE_G,
    FEDERATION_COMPOSITION_PHASE_H,
    FEDERATION_CONNECTION_SLUG_NON_WORD_RE,
    FEDERATION_COORDINATOR_DUCKDB_TYPE_MAP,
    FEDERATION_CROSS_SOURCE_JOIN_KINDS,
    FEDERATION_DECLARATION_TOP_LEVEL_KEYS,
    FEDERATION_DECLARATION_VERSION,
    FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS,
    FEDERATION_ENUM_PROMPT_CAP,
    FEDERATION_MANIFEST_ALIAS_KEYS,
    FEDERATION_MANIFEST_COORDINATOR_KEYS,
    FEDERATION_MANIFEST_FILENAME,
    FEDERATION_MANIFEST_JOIN_KEYS,
    FEDERATION_MANIFEST_LIMITS_KEYS,
    FEDERATION_MANIFEST_SOURCE_KEYS,
    FEDERATION_MANIFEST_TOP_LEVEL_KEYS,
    FEDERATION_MAPPING_NAME_SCORE_FLOOR,
    FEDERATION_MAPPING_NAME_SUBSTRING_SCORE,
    FEDERATION_MAPPING_SCORE_NAME_WEIGHT,
    FEDERATION_MAPPING_SCORE_OVERLAP_WEIGHT,
    FEDERATION_MAPPING_SUGGESTIONS_CACHE_FILENAME,
    FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR,
    FEDERATION_MAPPINGS_APPLIED_FILENAME,
    FEDERATION_MAPPINGS_FILENAME,
    FEDERATION_MAPPINGS_LOGICAL_COLUMN_KEYS,
    FEDERATION_MAPPINGS_LOGICAL_TABLE_KEYS,
    FEDERATION_MAPPINGS_TABLE_MEMBER_KEYS,
    FEDERATION_MAPPINGS_TOP_LEVEL_KEYS,
    FEDERATION_MAPPINGS_MIN_VERSION,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_MAX_JOIN_CANDIDATE_CAP,
    FEDERATION_MAX_JOIN_PATH_TIE_CAP,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP,
    FEDERATION_PLAN_TEMPLATE_FILE_CAP,
    FEDERATION_PLAN_TEMPLATE_FILENAME,
    FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION,
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    FEDERATION_SENSITIVITY_RANK,
    FEDERATION_STORAGE_PREFIX,
    FEDERATION_SOURCE_STORAGE_PREFIX,
    FEDERATION_STORAGE_SLUG_NON_ALNUM_RE,
    FEDERATION_TEMPLATES_SEGMENT,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_ACTION_DESTRUCTIVE,
    MIGRATION_MAP_ACTION_REMAP,
    REPHRASE_HINT_MESSAGES,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_SEGMENT,
    SCHEMA_OVERRIDES_APPLIED_SUFFIX,
    SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT,
    VALID_GRAINS,
    VALID_HAVING_OPS,
    FederationTopologyChange,
    anti_join_presence_column,
    ineligible_answerable_hint_for_code,
    is_file_engine,
)
from ._contracts_base import (
    ConfigError,
    DatabaseFeatureCapability,
    DescriptionOwner,
    EngineContext,
    FederationCapExceededError,
    FederationConfigError,
    FederationContext,
    FederationCoordinatorConfig,
    FederationCrossSourceJoin,
    FederationDeclarationError,
    FederationIneligibleError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationManifest,
    FederationMappings,
    FederationMappingSuggestion,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationMemberUnprofilableError,
    FederationMigrationMap,
    FederationPartialFailureError,
    FederationPlanTemplate,
    FederationTurnCancelledError,
    FederationQualifiedRename,
    FederationRuntimeError,
    FederationSourceBinding,
    FederationSourceLimits,
    FederationTableAlias,
    FederationTopologyReport,
    PersistedFederationInspection,
    InferenceTag,
    LogicalColumnMapping,
    LogicalTableMapping,
    LogicalTableMember,
    MigrationPendingError,
    MigrationReport,
    MigrationTier,
    NormalizedExpr,
    OrderByCol,
    OverrideReport,
    OwnerOnlyOperationError,
    PredicateGroup,
    SchemaRole,
    SensitivityClassification,
    SpaceContext,
    coerce_cte_emission,
    default_order_by_null_placement,
    expr_registry_ref,
    partition_predicate_group,
    predicate_group_from_list,
    DescriptionOwner,
    resolve_descriptions,
    resolve_federation_qualified_ref,
    set_description,
)
from ._contracts_core import (
    AnchoredTemporalBind,
    CoordinatorMemberFrame,
    FederatedPlan,
    FederatedPrepareOutcome,
    FederatedStage,
    FederationExecutionWave,
    FederationMemberResolvedLimits,
    FederationReducingEdge,
    FederationTableSet,
    HavingParam,
    JoinSpec,
    ResidualSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    UnionSpec,
    WhereParam,
)
from ._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, SQLShape, TableMetadata
from ._core_utils import (
    ArtifactManifest,
    artifact_lock,
    cost_cap_active,
    debug,
    emit_ask_phase,
    emit_construction_phase,
    effective_structural_hash_fp,
    notify,
    pipeline_trace,
    read_gzip_json,
    reconcile_execute_bind_params,
    stable_json,
    structural_hash_fp,
    wipe_filenames,
    write_gzip_json_atomic,
)
from ._data_quality import parse_source_selections, validate_upload_sources
from ._dialect import (
    dialect_supports_ilike_semantics,
    engine_supports_array_contains,
    engine_supports_collation,
    engine_supports_median,
    engine_supports_ordered_string_agg,
    engine_supports_stddev,
    engine_supports_timestamptz_semantics,
    engine_supports_unsigned_semantics,
    engine_supports_variance,
    engine_supports_window_frames,
    extra_where_ops_for_engine,
    get_dialect,
    get_runtime_config_class,
    list_engines,
    member_supports_ilike_semantics,
    sqlglot_quote_identifier,
)
from ._intent_expr import collect_intent_referenced_param_keys, extract_columns_from_expr
from ._intent_repair import (
    collect_referenced_tables,
    expand_shared_pk_tables_for_refs,
    reconcile_tables,
    where_scope_registries_to_referenced,
)
from ._intent_resolve import check_qualified_refs_exist, join_path_segments_fingerprint_runtime
from ._schema_catalog import (
    assign_column_ops,
    description_neutrality_violations,
    llm_classify_schema,
    sanitize_schema_graph_descriptions,
)
from ._schema_graph import (
    deny_columns_by_table,
    prune_foreign_keys_after_column_removal,
    apply_deny_objects_filter,
    assert_consumer_intent_in_scope,
    classify_migration_tier,
    fk_infer_value_types_compatible,
    intersect_member_database_feature_capabilities,
    mark_canonical_duplicates,
    mint_schema_graph_id,
    raise_if_schema_unusable,
    recompute_join_paths_multi,
    redact_hidden_sensitivity_profile_values,
    schema_context_from_descriptor,
    tables_structural_payload,
    validate_scope_against_graph,
)
from ._schema_overrides import (
    user_added_fks_dump,
    user_added_pks_dump,
    apply_schema_overrides_to_graph,
    compute_metadata_hash,
    dump_schema_overrides_to_path,
    finalize_with_overrides,
    load_overrides_sidecar,
    load_schema_overrides_file,
    save_overrides_sidecar,
)
from ._sql_gen import (
    generate_col_alias,
    get_join_choice_from_llm,
    render_expr_sql,
    render_feedback_sql,
    render_predicate_clause,
    render_predicate_group_sql,
    render_select_col_sql,
    wrap_core_sql_with_distinct_on,
)
from ._utils import flatten_param_values, intent_key
from ._validation_execute import validate_semantics, validate_sql


def _predicate_where_leaves(where: PredicateGroup | None) -> list[WhereParam]:
    if where is None:
        return []
    return [param for param in where.leaves() if isinstance(param, WhereParam)]


def _predicate_having_leaves(having: PredicateGroup | None) -> list[HavingParam]:
    if having is None:
        return []
    return [param for param in having.leaves() if isinstance(param, HavingParam)]


def intersect_member_where_ops(
    dialects_by_source: Mapping[str, Any] | None = None, *, engine_types_by_source: Mapping[str, str] | None = None
) -> frozenset[str]:
    """Return WHERE operators supported by every federation member dialect."""
    allowed = set(FEDERATION_BASE_WHERE_OPS)
    extra_sets: list[frozenset[str]] = []
    if dialects_by_source:
        for dialect in dialects_by_source.values():
            if dialect is None:
                continue
            extra_fn = getattr(dialect, "extra_where_ops", None)
            if callable(extra_fn):
                extra_sets.append(frozenset(extra_fn()))
    elif engine_types_by_source:
        for engine_type in engine_types_by_source.values():
            extra_sets.append(extra_where_ops_for_engine(engine_type))
    if extra_sets:
        shared_extra = set(extra_sets[0])
        for extra in extra_sets[1:]:
            shared_extra &= set(extra)
        allowed.update(shared_extra)
    if "ilike" not in allowed:
        if dialects_by_source:
            dialects = [d for d in dialects_by_source.values() if d is not None]
            if dialects and all(dialect_supports_ilike_semantics(d) for d in dialects):
                allowed.update({"ilike", "not ilike"})
        elif engine_types_by_source:
            if all(member_supports_ilike_semantics(engine_type) for engine_type in engine_types_by_source.values()):
                allowed.update({"ilike", "not ilike"})
    if dialects_by_source:
        if all(bool(getattr(d, "supports_array_contains", True)) for d in dialects_by_source.values() if d is not None):
            allowed.add("contains")
    elif engine_types_by_source:
        if all(engine_supports_array_contains(engine_type) for engine_type in engine_types_by_source.values()):
            allowed.add("contains")
    return frozenset(allowed)


def intersect_member_dialect_capabilities(
    dialects_by_source: Mapping[str, Any] | None = None,
    *,
    engine_types_by_source: Mapping[str, str] | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
) -> dict[str, Any]:
    """Derive federation capability as the intersection of member dialect surfaces."""
    where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types_by_source)
    having_ops = frozenset(VALID_HAVING_OPS)
    for dialect in (dialects_by_source or {}).values():
        if dialect is None:
            continue
        extra_having = getattr(dialect, "extra_having_ops", None)
        if callable(extra_having):
            having_ops &= frozenset(extra_having())
    ir_cap = intersect_member_database_feature_capabilities(member_graphs) if member_graphs else None
    engine_types = list(engine_types_by_source.values()) if engine_types_by_source else []

    def _member_flag(field: str, engine_fn: Callable[[str], bool]) -> bool:
        if ir_cap is not None:
            return bool(getattr(ir_cap, field))
        if engine_types:
            return all(engine_fn(engine_type) for engine_type in engine_types)
        return True

    return {
        "where_ops": where_ops,
        "having_ops": having_ops,
        "supports_semi_join": True if ir_cap is None else ir_cap.supports_semi_join,
        "supports_anti_join": True if ir_cap is None else ir_cap.supports_anti_join,
        "supports_predicate_nesting": True if ir_cap is None else ir_cap.supports_predicate_nesting,
        "supports_preserve_tables": True if ir_cap is None else ir_cap.supports_preserve_tables,
        "supports_ordered_string_agg": _member_flag("supports_ordered_string_agg", engine_supports_ordered_string_agg),
        "supports_median": _member_flag("supports_median", engine_supports_median),
        "supports_stddev": _member_flag("supports_stddev", engine_supports_stddev),
        "supports_variance": _member_flag("supports_variance", engine_supports_variance),
        "supports_window_frames": _member_flag("supports_window_frames", engine_supports_window_frames),
        "supports_array_contains": _member_flag("supports_array_contains", engine_supports_array_contains),
        "supports_collation": _member_flag("supports_collation", engine_supports_collation),
        "supports_unsigned_semantics": _member_flag("supports_unsigned_semantics", engine_supports_unsigned_semantics),
        "supports_timestamptz_semantics": _member_flag(
            "supports_timestamptz_semantics", engine_supports_timestamptz_semantics
        ),
    }


def _member_where_ops_for_binding(
    binding: Any,
    *,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Return WHERE operators supported by one federation member."""
    allowed = set(FEDERATION_BASE_WHERE_OPS)
    dialect = (dialects_by_source or {}).get(binding.source_id)
    if dialect is not None:
        extra_fn = getattr(dialect, "extra_where_ops", None)
        if callable(extra_fn):
            allowed.update(extra_fn())
        if dialect_supports_ilike_semantics(dialect):
            allowed.update({"ilike", "not ilike"})
    else:
        allowed.update(extra_where_ops_for_engine(binding.engine))
        if member_supports_ilike_semantics(binding.engine):
            allowed.update({"ilike", "not ilike"})
    if engine_supports_array_contains(binding.engine):
        allowed.add("contains")
    return frozenset(allowed)


def _member_having_ops_for_binding(
    binding: Any,
    *,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Return HAVING operators supported by one federation member."""
    allowed = set(VALID_HAVING_OPS)
    dialect = (dialects_by_source or {}).get(binding.source_id)
    if dialect is not None:
        extra_fn = getattr(dialect, "extra_having_ops", None)
        if callable(extra_fn):
            allowed &= set(extra_fn())
    return frozenset(allowed)


def _federation_member_lacking_where_op(
    op: str,
    manifest: FederationManifest,
    *,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the first member source id that cannot express *op* in WHERE."""
    op_norm = str(op or "").strip().lower()
    if not op_norm:
        return None
    for binding in manifest.sources:
        if op_norm not in _member_where_ops_for_binding(binding, dialects_by_source=dialects_by_source):
            return binding.source_id
    return None


def _federation_member_lacking_having_op(
    op: str,
    manifest: FederationManifest,
    *,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the first member source id that cannot express *op* in HAVING."""
    op_norm = str(op or "").strip().lower()
    if not op_norm:
        return None
    for binding in manifest.sources:
        if op_norm not in _member_having_ops_for_binding(binding, dialects_by_source=dialects_by_source):
            return binding.source_id
    return None


def _federation_member_capability_operator_reason(
    clause: str,
    op: str,
    member_id: str,
    *,
    detail: str | None = None,
) -> str:
    """Build an ineligible reason that resolves to the member-capability hint."""
    text = f"member capability: {clause} operator {op!r} is not supported by federation member {member_id!r}"
    if detail:
        return f"{text}: {detail}"
    return text


def _federation_member_lacking_ilike_semantics(
    manifest: FederationManifest,
    *,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the first member source id that cannot express case- insensitive filters."""
    for binding in manifest.sources:
        dialect = (dialects_by_source or {}).get(binding.source_id)
        if dialect is not None:
            if not dialect_supports_ilike_semantics(dialect):
                return binding.source_id
            continue
        if not member_supports_ilike_semantics(binding.engine):
            return binding.source_id
    return None


def _federation_unsupported_operator_reason(
    intent: RuntimeIntent, manifest: FederationManifest, *, dialects_by_source: Mapping[str, Any] | None = None
) -> str | None:
    """Refuse operators absent from the intersection of member dialect capabilities."""
    engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
    if not engine_types and not dialects_by_source:
        return None
    caps = intersect_member_dialect_capabilities(dialects_by_source, engine_types_by_source=engine_types or None)
    allowed_where = caps.get("where_ops") or frozenset()
    allowed_having = caps["having_ops"]
    for fp in _predicate_where_leaves(intent.where) or []:
        op = str(fp.op or "").strip().lower()
        if op and op not in allowed_where:
            if op in ("ilike", "not ilike"):
                lacking = _federation_member_lacking_ilike_semantics(manifest, dialects_by_source=dialects_by_source)
                if lacking is not None:
                    return _federation_member_capability_operator_reason(
                        "where",
                        fp.op or op,
                        lacking,
                        detail="native ILIKE and case-insensitive rewrite are both unavailable",
                    )
            lacking = _federation_member_lacking_where_op(op, manifest, dialects_by_source=dialects_by_source)
            if lacking is not None:
                return _federation_member_capability_operator_reason("where", fp.op or op, lacking)
            return _federation_member_capability_operator_reason("where", fp.op or op, "unknown")
    for hp in _predicate_having_leaves(intent.having) or []:
        op = str(hp.op or "").strip().lower()
        if op and op not in allowed_having:
            lacking = _federation_member_lacking_having_op(op, manifest, dialects_by_source=dialects_by_source)
            if lacking is not None:
                return _federation_member_capability_operator_reason("having", hp.op or op, lacking)
            return _federation_member_capability_operator_reason("having", hp.op or op, "unknown")
    return None


def _expr_uses_ordered_string_agg(expr: NormalizedExpr) -> bool:
    for g in expr.add_groups + expr.sub_groups:
        if (g.agg_func or "").strip().lower() == "string_agg" and g.agg_order_by:
            return True
    return False


def _intent_uses_ordered_string_agg(intent: RuntimeIntent) -> bool:
    for sc in intent.select_cols or []:
        if _expr_uses_ordered_string_agg(sc.expr):
            return True
    for cte in intent.cte_steps or []:
        for sc in cte.select_cols or []:
            if _expr_uses_ordered_string_agg(sc.expr):
                return True
    return False


def _intent_uses_median(intent: RuntimeIntent) -> bool:
    def _check_expr(expr: NormalizedExpr) -> bool:
        for g in expr.add_groups + expr.sub_groups:
            if (g.agg_func or "").strip().lower() == "median":
                return True
        if (expr.agg_func or "").strip().lower() == "median":
            return True
        return False

    for sc in intent.select_cols or []:
        if _check_expr(sc.expr):
            return True
    for cte in intent.cte_steps or []:
        for sc in cte.select_cols or []:
            if _check_expr(sc.expr):
                return True
    return False


def _expr_uses_statistical_agg(expr: NormalizedExpr, funcs: frozenset[str]) -> bool:
    for group in expr.add_groups + expr.sub_groups:
        if (group.agg_func or "").strip().lower() in funcs:
            return True
    if (expr.agg_func or "").strip().lower() in funcs:
        return True
    return False


def _intent_uses_statistical_agg(intent: RuntimeIntent, funcs: frozenset[str]) -> bool:
    for sc in intent.select_cols or []:
        if _expr_uses_statistical_agg(sc.expr, funcs):
            return True
    for cte in intent.cte_steps or []:
        for sc in cte.select_cols or []:
            if _expr_uses_statistical_agg(sc.expr, funcs):
                return True
    return False


def _intent_uses_stddev(intent: RuntimeIntent) -> bool:
    return _intent_uses_statistical_agg(intent, frozenset({"stddev"}))


def _intent_uses_variance(intent: RuntimeIntent) -> bool:
    return _intent_uses_statistical_agg(intent, frozenset({"variance"}))


def _window_spec_uses_frames(window_spec: Any) -> bool:
    frame_kind = str(getattr(window_spec, "frame_kind", "none") or "none").strip().lower()
    return frame_kind != "none"


def _intent_uses_window_frames(intent: RuntimeIntent) -> bool:
    for entry in intent.window_registry or []:
        if _window_spec_uses_frames(getattr(entry, "window_spec", None)):
            return True
    for cte in intent.cte_steps or []:
        for entry in cte.window_registry or []:
            if _window_spec_uses_frames(getattr(entry, "window_spec", None)):
                return True
    return False


def _intent_uses_array_contains(intent: RuntimeIntent) -> bool:
    for fp in _predicate_where_leaves(intent.where):
        if (fp.op or "").strip().lower() == "contains":
            return True
    for hp in _predicate_having_leaves(intent.having):
        if (hp.op or "").strip().lower() == "contains":
            return True
    return False


def _intent_column_data_types(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    predicate: Callable[[str], bool],
) -> bool:
    refs: set[str] = set()
    for sc in intent.select_cols or []:
        refs.update(extract_columns_from_expr(sc.expr))
    for col in intent.group_by_cols or []:
        refs.update(extract_columns_from_expr(col))
    for obc in intent.order_by_cols or []:
        refs.update(extract_columns_from_expr(obc.expr))
    for fp in _predicate_where_leaves(intent.where):
        refs.update(extract_columns_from_expr(fp.left_expr))
    for hp in _predicate_having_leaves(intent.having):
        refs.update(extract_columns_from_expr(hp.left_expr))
    for cref in refs:
        if "." not in cref:
            continue
        table_name, column_name = cref.rsplit(".", 1)
        table = schema.tables.get(table_name)
        if table is None:
            continue
        col_meta = table.columns.get(column_name)
        if col_meta is not None and predicate(str(col_meta.data_type or "")):
            return True
    return False


def _federation_ir_capability_reason(
    intent: RuntimeIntent,
    cap: DatabaseFeatureCapability,
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse IR shapes absent from the intersection of member capabilities."""
    for cte in intent.cte_steps or []:
        emission = coerce_cte_emission(getattr(cte, "emission", "join_table"))
        if emission == "semi_join" and not cap.supports_semi_join:
            name = (cte.cte_name or "").strip() or "semi_join"
            return f"semi_join is not supported by all federation members: {name}"
        if emission == "anti_join" and not cap.supports_anti_join:
            name = (cte.cte_name or "").strip() or "anti_join"
            return f"anti_join is not supported by all federation members: {name}"
    if intent.preserve_tables and not cap.supports_preserve_tables:
        return "preserve_tables is not supported by all federation members"
    if _intent_uses_ordered_string_agg(intent) and not cap.supports_ordered_string_agg:
        return "ordered string_agg is not supported by all federation members"
    if _intent_uses_median(intent) and not cap.supports_median:
        return "median is not supported by all federation members"
    if _intent_uses_stddev(intent) and not cap.supports_stddev:
        return "stddev is not supported by all federation members"
    if _intent_uses_variance(intent) and not cap.supports_variance:
        return "variance is not supported by all federation members"
    if _intent_uses_window_frames(intent) and not cap.supports_window_frames:
        return "window frames are not supported by all federation members"
    if _intent_uses_array_contains(intent) and not cap.supports_array_contains:
        return "array contains is not supported by all federation members"
    if schema is not None:
        if not cap.supports_timestamptz_semantics and _intent_column_data_types(
            intent, schema, predicate=lambda dt: "timestamptz" in dt.lower()
        ):
            return "timestamptz semantics are not supported by all federation members"
        if not cap.supports_unsigned_semantics and _intent_column_data_types(
            intent, schema, predicate=lambda dt: "unsigned" in dt.lower()
        ):
            return "unsigned integer semantics are not supported by all federation members"
    nested_where = bool(intent.where and intent.where.depth() > 1)
    nested_having = bool(intent.having and intent.having.depth() > 1)
    if (nested_where or nested_having) and not cap.supports_predicate_nesting:
        return "nested predicate groups are not supported by all federation members"
    return None


def _predicate_param_sources(
    param: WhereParam | HavingParam,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    registry_kw: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return member source ids referenced by a single predicate leaf."""
    return _param_referenced_sources(param, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw)


def _predicate_group_spans_sources(
    group: PredicateGroup | None,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    registry_kw: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a reason when an OR branch in *group* connects predicates on different members."""
    if group is None:
        return None
    if group.op == "or":
        for pred in group.predicates:
            srcs = _predicate_param_sources(
                pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if len(srcs) > 1:
                return f"cross-source OR filter is not supported: {_predicate_clause_label(pred)}"
        branch_sources: list[set[str]] = []
        for pred in group.predicates:
            srcs = _predicate_param_sources(
                pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if srcs:
                branch_sources.append(srcs)
        for child in group.groups:
            child_reason = _predicate_group_spans_sources(
                child, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if child_reason:
                return child_reason
            child_srcs: set[str] = set()
            for pred in child.leaves():
                child_srcs |= _predicate_param_sources(
                    pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
                )
            if child_srcs:
                branch_sources.append(child_srcs)
        touched = {src for srcs in branch_sources for src in srcs}
        if len(touched) > 1 and len(branch_sources) > 1:
            labels = []
            for pred in group.predicates:
                labels.append(_predicate_clause_label(pred))
            for child in group.groups:
                child_labels = " AND ".join(_predicate_clause_label(pred) for pred in child.leaves())
                if child_labels:
                    labels.append(child_labels)
            joined = " OR ".join(label for label in labels if label)
            return f"cross-source predicate disjunction is not supported: {joined}"
    for child in group.groups:
        child_reason = _predicate_group_spans_sources(
            child, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
        )
        if child_reason:
            return child_reason
    return None


def _cte_probe_join_keys(cte: RuntimeCteStep) -> list[str]:
    """Return qualified join-key column refs projected by a semi/anti probe CTE."""
    keys: list[str] = []
    for sc in cte.select_cols or []:
        col = (sc.expr.column_ref or sc.expr.primary_column or sc.expr.primary_term or "").strip()
        if col and "." in col and col not in keys:
            keys.append(col)
    return keys


def _cross_source_probe_cte_steps(
    intent: RuntimeIntent, source_by_table: Mapping[str, str]
) -> tuple[RuntimeCteStep, ...]:
    """Return semi/anti probe CTEs whose body lives on a different member than the driver tables."""
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return ()
    owners = _assign_cte_sources(cte_steps, source_by_table)
    driver_tables = set(intent.tables or [])
    for cte in cte_steps:
        if cte.cte_name:
            driver_tables.discard(cte.cte_name)
    driver_sources = {source_by_table.get(table, "") for table in driver_tables if source_by_table.get(table, "")}
    lifted: list[RuntimeCteStep] = []
    for cte in cte_steps:
        emission = coerce_cte_emission(getattr(cte, "emission", "join_table"))
        if emission not in ("semi_join", "anti_join"):
            continue
        owner = owners.get(cte.cte_name or "")
        if not owner or owner in driver_sources:
            continue
        if cte.cte_name and cte.cte_name in (intent.tables or []):
            lifted.append(cte)
            continue
        refs = collect_referenced_tables(
            intent.select_cols,
            intent.order_by_cols,
            intent.group_by_cols,
            _predicate_where_leaves(intent.where),
            _predicate_having_leaves(intent.having),
            window_registry=intent.window_registry,
            case_registry=intent.case_registry,
            include_unreferenced_registries=False,
        )
        if cte.cte_name and cte.cte_name in refs:
            lifted.append(cte)
    return tuple(lifted)


def _cross_source_probe_cte_ineligible_reason(
    intent: RuntimeIntent, manifest: FederationManifest, source_by_table: Mapping[str, str]
) -> str | None:
    """Refuse cross-source semi/anti probes that cannot be lifted to the coordinator."""
    for cte in _cross_source_probe_cte_steps(intent, source_by_table):
        keys = _cte_probe_join_keys(cte)
        if not keys:
            name = (cte.cte_name or "").strip() or "probe"
            return f"cross-source {coerce_cte_emission(getattr(cte, 'emission', 'join_table'))} requires declared join keys: {name}"
        if not manifest.cross_source_joins:
            name = (cte.cte_name or "").strip() or "probe"
            return f"cross-source {coerce_cte_emission(getattr(cte, 'emission', 'join_table'))} requires declared join: {name}"
        covered = False
        key_cols = set(keys)
        for join in manifest.cross_source_joins:
            if {join.left, join.right} & key_cols:
                covered = True
                break
        if not covered:
            name = (cte.cte_name or "").strip() or "probe"
            emission = coerce_cte_emission(getattr(cte, "emission", "join_table"))
            return f"cross-source {emission} requires declared join keys: {name}"
    return None


def _distinct_on_spans_sources(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> bool:
    """Return True when any distinct_on partition column references more than one member."""
    if not intent.distinct_on:
        return False
    registry_kw = _intent_registry_kw(intent)
    refs = collect_referenced_tables([], [], [], [], [], **registry_kw)
    for expr in intent.distinct_on:
        refs |= collect_referenced_tables([], [], [expr], [], [], **registry_kw)
    srcs = _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
    return len(srcs) > 1


def stamp_federation_member_graph(graph: SchemaGraph, *, federation_id: str, source_id: str, engine: str = "") -> None:
    """Record federation membership on a member schema graph at composition time."""
    fed_id = str(federation_id or "").strip()
    sid = str(source_id or "").strip()
    if not fed_id or not sid:
        return
    payload: dict[str, str] = {"federation_id": fed_id, "source_id": sid}
    eng = str(engine or "").strip().lower()
    if eng:
        payload["engine"] = eng
    graph.federation_membership = payload
    object.__setattr__(graph, "_database_feature_capability_cache", None)


def _manifest_engine_for_source(manifest: FederationManifest, source_id: str) -> str:
    for binding in manifest.sources:
        if binding.source_id == source_id:
            return str(binding.engine or "").strip().lower()
    return ""


def stamp_federation_member_template(template: Any, *, plan_id: str, source_id: str = "") -> None:
    """Mark a member template as matchable only through its federation plan record."""
    template.federation_plan_id = str(plan_id or "")
    template.federation_plan_only = True
    if source_id:
        template.member_source_id = str(source_id)


def template_is_federation_plan_fragment(template: Any) -> bool:
    """Return True when *template* must not be reused as a standalone answer."""
    return bool(getattr(template, "federation_plan_only", False))


def _intent_registry_kw(intent: RuntimeIntent) -> dict[str, Any]:
    """Registry kwargs for intent repair collectors (mypy-safe ``**`` passthrough)."""
    return {
        "window_registry": intent.window_registry,
        "case_registry": intent.case_registry,
    }


def federation_residual_column_headers(plan: FederatedPlan) -> tuple[str, ...]:
    """Derive coordinator result column names from the federated residual projection."""
    residual = plan.residual
    if residual is None or not residual.select_cols:
        return ()
    headers: list[str] = []
    for sc in residual.select_cols:
        alias = (sc.output_alias or "").strip() or generate_col_alias(sc)
        if alias:
            headers.append(alias)
            continue
        col_ref = (sc.expr.column_ref or sc.expr.primary_column or sc.expr.primary_term or "").strip()
        if col_ref:
            headers.append(col_ref.rsplit(".", 1)[-1])
    return tuple(headers)


def federation_plan_sql_shape(plan: FederatedPlan) -> SQLShape:
    """Derive template ``sql_shape`` from a federated plan rather than display SQL."""
    num_joins = len(plan.combine) if isinstance(plan.combine, tuple) else 0
    num_where = 0
    num_having = 0
    num_cte = 0
    has_group_by = False
    has_agg = False
    has_distinct = False
    for step in plan.steps:
        sub = step.sub_intent
        num_where += len((_predicate_where_leaves(sub.where)) or [])
        num_having += len((_predicate_having_leaves(sub.having)) or [])
        num_cte += len(sub.cte_steps or [])
        if sub.group_by_cols:
            has_group_by = True
        for sc in sub.select_cols or []:
            if sc.is_aggregated:
                has_agg = True
        if (sub.distinct_select_index or -1) >= 0:
            has_distinct = True
    residual = plan.residual
    if residual is not None:
        num_where += len(_predicate_where_leaves(residual.where))
        num_having += len(_predicate_having_leaves(residual.having))
        if residual.group_by_cols:
            has_group_by = True
        for sc in residual.select_cols:
            if sc.is_aggregated:
                has_agg = True
        if (residual.distinct_select_index or -1) >= 0:
            has_distinct = True
    return SQLShape(
        num_joins=num_joins,
        has_group_by=has_group_by,
        has_agg=has_agg,
        num_cte=num_cte,
        num_where=num_where,
        num_having=num_having,
        has_distinct=has_distinct,
    )


def lookup_federation_plan_template_for_question(federation_dir: str, q_norm: str) -> FederationPlanTemplate | None:
    """Return a stored federation plan whose accepted questions include *q_norm*."""
    if not federation_dir or not q_norm:
        return None
    for template in load_federation_plan_templates(federation_dir).values():
        if q_norm in template.accepted_questions:
            return template
    return None


def federation_source_ids_on_schema(schema: SchemaGraph) -> frozenset[str]:
    """Return distinct member source ids stamped on *schema* tables."""
    sources: set[str] = set()
    for table in schema.tables.values():
        source_id = getattr(table, "source_id", None)
        if source_id:
            sources.add(str(source_id))
        member_ids = getattr(table, "member_source_ids", None) or ()
        sources.update(str(sid) for sid in member_ids)
    return frozenset(sources)


def schema_spans_multiple_sources(schema: SchemaGraph) -> bool:
    """Return True when *schema* represents a multi-member federation composite."""
    return len(federation_source_ids_on_schema(schema)) >= 2


def federation_prompt_fields_for_schema(schema: SchemaGraph) -> dict[str, str]:
    """Return no extra interpret/ground/compose fields; federation stays invisible in prompts."""
    _ = schema
    return {}


def member_feedback_q_norm(source_id: str, q_norm: str) -> str:
    """Scope member-store feedback keys so federation questions do not collide with standalone reuse."""
    sid = str(source_id or "").strip()
    q = str(q_norm or "").strip()
    if not sid or not q:
        return q
    prefix = f"{sid}::"
    if q.startswith(prefix):
        return q
    return f"{prefix}{q}"


def federation_scaled_join_path_tie_cap(member_count: int) -> int:
    """Scale shortest-path tie storage for a composite, bounded by member count."""
    base = max(1, int(PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP))
    members = max(1, int(member_count))
    return min(max(base, base * members), int(FEDERATION_MAX_JOIN_PATH_TIE_CAP))


def federation_scaled_join_candidate_cap(member_count: int) -> int:
    """Scale merged join candidate cross-product cap for a composite, bounded by member count."""
    base = max(1, int(PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP))
    members = max(1, int(member_count))
    return min(max(base, base * members), int(FEDERATION_MAX_JOIN_CANDIDATE_CAP))


def _intent_has_temporal_anchor_refs(intent: RuntimeIntent) -> bool:
    """Return True when *intent* uses relative date windows or clock keywords."""
    for fp in _predicate_where_leaves(intent.where) or ():
        if str(getattr(fp, "value_type", "") or "") == "date_window":
            return True
        right = getattr(fp, "right_expr", None)
        if right is not None and str(getattr(right, "keyword", "") or "").lower() in {
            "current_date",
            "current_timestamp",
            "localtimestamp",
            "localtime",
            "sysdate",
        }:
            return True
    for hp in _predicate_having_leaves(intent.having) or ():
        if str(getattr(hp, "value_type", "") or "") == "date_window":
            return True
    return False


def resolve_anchored_temporal_bind(
    intent: RuntimeIntent, *, anchor: datetime | None = None
) -> AnchoredTemporalBind | None:
    """Resolve a temporal reference once for federated member rendering. When the parent intent carries relative date-window or clock-keyword predicates, bind them to a single anchor at turn start so each member statement uses the same instant rather than re-evaluating per-member clock functions."""
    if not _intent_has_temporal_anchor_refs(intent):
        return None
    anchor_dt = anchor or datetime.now(timezone.utc)
    return AnchoredTemporalBind(anchor_iso=anchor_dt.isoformat())


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str], *, label: str) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise FederationDeclarationError(f"{label} contains unknown keys: {', '.join(unknown)}")


def parse_federation_manifest(
    raw: Mapping[str, Any] | str | bytes, *, include_derived_roster: bool = False
) -> FederationManifest:
    """Parse and validate a federation manifest from a mapping or JSON text."""
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FederationConfigError(f"malformed federation declaration: invalid manifest JSON: {exc}") from exc
    else:
        payload = dict(raw)
    if not isinstance(payload, dict):
        raise FederationConfigError("federation manifest must be a JSON object")
    _reject_unknown_keys(payload, FEDERATION_MANIFEST_TOP_LEVEL_KEYS, label="federation manifest")
    federation_id = str(payload.get("federation_id", "") or "").strip()
    if not federation_id:
        raise FederationConfigError("federation manifest requires federation_id")
    sources_raw = payload.get("sources")
    if sources_raw is None:
        sources_raw = []
    if not include_derived_roster and sources_raw:
        raise FederationDeclarationError("sources are derived at compose time; omit from authored federation manifest")
    if not isinstance(sources_raw, list):
        raise FederationConfigError("sources must be an array when present")
    sources: list[FederationSourceBinding] = []
    seen_ids: set[str] = set()
    for entry in sources_raw:
        if not isinstance(entry, dict):
            raise FederationConfigError("each federation source must be an object")
        _reject_unknown_keys(entry, FEDERATION_MANIFEST_SOURCE_KEYS, label="federation source")
        source_id = str(entry.get("source_id", "") or "").strip()
        if not source_id:
            raise FederationConfigError("each federation source requires source_id")
        _validate_federation_source_id_identifier(source_id)
        if source_id in seen_ids:
            raise FederationConfigError(f"duplicate federation source_id: {source_id!r}")
        seen_ids.add(source_id)
        engine = str(entry.get("engine", "") or "").strip().lower()
        if not engine:
            raise FederationConfigError(f"source {source_id!r} requires engine")
        if engine not in list_engines():
            raise FederationConfigError(f"source {source_id!r} references unknown engine {engine!r}")
        connection = str(entry.get("connection", "") or "").strip()
        context = str(entry.get("context", "master") or "master").strip().lower() or "master"
        role_raw = str(entry.get("role", "owner") or "owner").strip().lower()
        if role_raw not in ("owner", "consumer"):
            raise FederationConfigError(f"source {source_id!r} has invalid role: {role_raw!r}")
        role = cast(SchemaRole, role_raw)
        _assert_federation_member_role_is_owner(source_id, role)
        limits_obj = entry.get("limits")
        limits: FederationSourceLimits | None = None
        if isinstance(limits_obj, dict):
            _reject_unknown_keys(limits_obj, FEDERATION_MANIFEST_LIMITS_KEYS, label="federation source limits")
            row_cap_raw = limits_obj.get("row_cap")
            timeout_raw = limits_obj.get("timeout_ms")
            semijoin_raw = limits_obj.get("semijoin_enabled")
            cost_rows_raw = limits_obj.get("max_query_cost_rows")
            cost_bytes_raw = limits_obj.get("max_query_cost_bytes")
            profile_timeout_raw = limits_obj.get("profile_timeout_ms")
            limits = FederationSourceLimits(
                row_cap=int(row_cap_raw) if row_cap_raw is not None else None,
                timeout_ms=int(timeout_raw) if timeout_raw is not None else None,
                semijoin_enabled=True if semijoin_raw is None else bool(semijoin_raw),
                max_query_cost_rows=float(cost_rows_raw) if cost_rows_raw is not None else None,
                max_query_cost_bytes=float(cost_bytes_raw) if cost_bytes_raw is not None else None,
                profile_timeout_ms=int(profile_timeout_raw) if profile_timeout_raw is not None else None,
            )
        sources.append(
            FederationSourceBinding(
                source_id=source_id, engine=engine, connection=connection, context=context, role=role, limits=limits
            )
        )
    namespace_raw = payload.get("table_namespace")
    if not include_derived_roster and namespace_raw is not None:
        raise FederationDeclarationError(
            "table_namespace is derived at compose time; omit from authored federation manifest"
        )
    table_namespace: dict[str, str] = {}
    if namespace_raw is not None:
        if not isinstance(namespace_raw, dict):
            raise FederationConfigError("table_namespace must be an object when present")
        for logical, sid in namespace_raw.items():
            logical_name = str(logical).strip()
            source_name = str(sid).strip()
            if not logical_name or not source_name:
                raise FederationConfigError("table_namespace keys and values must be non-empty")
            if seen_ids and source_name not in seen_ids:
                raise FederationConfigError(f"table_namespace references unknown source_id: {source_name!r}")
            table_namespace[logical_name] = source_name
    aliases_raw = payload.get("aliases")
    if aliases_raw is None:
        aliases_raw = {}
    if not isinstance(aliases_raw, dict):
        raise FederationConfigError("aliases must be an object")
    aliases: list[FederationTableAlias] = []
    seen_alias_names: set[str] = set()
    for alias_name, entry in aliases_raw.items():
        logical = str(alias_name).strip()
        if not logical:
            raise FederationConfigError("aliases keys must be non-empty")
        if not isinstance(entry, dict):
            raise FederationConfigError(f"alias {logical!r} must be an object")
        _reject_unknown_keys(entry, FEDERATION_MANIFEST_ALIAS_KEYS, label=f"alias {logical!r}")
        source_name = str(entry.get("source", "") or "").strip()
        table_name = str(entry.get("table", "") or "").strip()
        if not source_name or not table_name:
            raise FederationConfigError(f"alias {logical!r} requires source and table")
        if logical in seen_alias_names:
            raise FederationConfigError(f"duplicate federation alias: {logical!r}")
        seen_alias_names.add(logical)
        if seen_ids and source_name not in seen_ids:
            raise FederationConfigError(f"alias {logical!r} references unknown source_id: {source_name!r}")
        aliases.append(FederationTableAlias(alias=logical, source=source_name, table=table_name))
    joins_raw = payload.get("cross_source_joins", [])
    if joins_raw is None:
        joins_raw = []
    if not isinstance(joins_raw, list):
        raise FederationConfigError("cross_source_joins must be an array")
    joins: list[FederationCrossSourceJoin] = []
    for entry in joins_raw:
        if not isinstance(entry, dict):
            raise FederationConfigError("each cross_source_join must be an object")
        _reject_unknown_keys(entry, FEDERATION_MANIFEST_JOIN_KEYS, label="cross_source_join")
        left = str(entry.get("left", "") or "").strip()
        right = str(entry.get("right", "") or "").strip()
        kind = validate_federation_cross_source_join_kind(str(entry.get("kind", "inner") or "inner"))
        logical_key = str(entry.get("logical_key", "") or "").strip()
        if not left or not right or not logical_key:
            raise FederationConfigError("cross_source_join requires left, right, and logical_key")
        joins.append(FederationCrossSourceJoin(left=left, right=right, kind=kind, logical_key=logical_key))
    coord_raw = payload.get("coordinator")
    if isinstance(coord_raw, dict):
        _reject_unknown_keys(coord_raw, FEDERATION_MANIFEST_COORDINATOR_KEYS, label="federation coordinator")
    coordinator = _parse_coordinator_config(coord_raw if isinstance(coord_raw, dict) else {})
    if not include_derived_roster:
        sources = []
        table_namespace = {}
    manifest = FederationManifest(
        federation_id=federation_id,
        sources=tuple(sorted(sources, key=lambda s: s.source_id)),
        table_namespace=table_namespace,
        cross_source_joins=tuple(joins),
        coordinator=coordinator,
        aliases=tuple(sorted(aliases, key=lambda a: (a.alias, a.source, a.table))),
    )
    validate_manifest_cross_source_joins(manifest)
    return manifest


def federation_manifest_document(manifest: FederationManifest, *, include_derived: bool = False) -> dict[str, Any]:
    """Serialize a federation manifest for persistence (joins + coordinator by default)."""
    payload: dict[str, Any] = {
        "federation_id": manifest.federation_id,
        "cross_source_joins": [
            {
                "left": join.left,
                "right": join.right,
                "kind": join.kind,
                "logical_key": join.logical_key,
            }
            for join in manifest.cross_source_joins
        ],
        "coordinator": {
            "row_cap": manifest.coordinator.row_cap,
            "default_source_row_cap": manifest.coordinator.default_source_row_cap,
            "default_source_timeout_ms": manifest.coordinator.default_source_timeout_ms,
            "coordinator_timeout_ms": manifest.coordinator.coordinator_timeout_ms,
            "plan_timeout_ms": manifest.coordinator.plan_timeout_ms,
            "semijoin_key_cap": manifest.coordinator.semijoin_key_cap,
            "spill_row_threshold": manifest.coordinator.spill_row_threshold,
            "max_parallel_members": manifest.coordinator.max_parallel_members,
            "total_input_byte_cap": manifest.coordinator.total_input_byte_cap,
        },
    }
    if manifest.aliases:
        payload["aliases"] = {
            alias.alias: {"source": alias.source, "table": alias.table}
            for alias in sorted(manifest.aliases, key=lambda a: a.alias)
        }
    if include_derived:
        payload["sources"] = [
            {
                "source_id": source.source_id,
                "engine": source.engine,
                "connection": source.connection,
                "context": source.context,
                "role": source.role,
                "limits": (
                    {
                        "row_cap": source.limits.row_cap,
                        "timeout_ms": source.limits.timeout_ms,
                        "semijoin_enabled": source.limits.semijoin_enabled,
                        "max_query_cost_rows": source.limits.max_query_cost_rows,
                        "max_query_cost_bytes": source.limits.max_query_cost_bytes,
                        "profile_timeout_ms": source.limits.profile_timeout_ms,
                    }
                    if source.limits is not None
                    else None
                ),
            }
            for source in manifest.sources
        ]
        payload["table_namespace"] = dict(manifest.table_namespace)
    return payload


def is_persisted_federation_manifest_sidecar(payload: Mapping[str, Any]) -> bool:
    """Return True when *payload* is a persisted sidecar without member roster fields."""
    return bool(payload.get("cross_source_joins") is not None or payload.get("coordinator")) and not payload.get(
        "sources"
    )


def hydrate_persisted_federation_manifest(stored: Mapping[str, Any], built: FederationManifest) -> FederationManifest:
    """Merge a persisted manifest sidecar with a member-derived roster."""
    if not is_persisted_federation_manifest_sidecar(stored):
        return parse_federation_manifest(stored)
    sidecar = federation_manifest_document(built)
    sidecar["federation_id"] = str(stored.get("federation_id", built.federation_id) or built.federation_id)
    if isinstance(stored.get("cross_source_joins"), list):
        sidecar["cross_source_joins"] = stored["cross_source_joins"]
    if isinstance(stored.get("coordinator"), dict):
        sidecar["coordinator"] = stored["coordinator"]
    if isinstance(stored.get("aliases"), dict):
        sidecar["aliases"] = stored["aliases"]
    sidecar["sources"] = federation_manifest_document(built, include_derived=True)["sources"]
    sidecar["table_namespace"] = dict(built.table_namespace)
    return parse_federation_manifest(sidecar, include_derived_roster=True)


def export_federation_manifest(manifest: FederationManifest, target: str | os.PathLike[str]) -> str:
    """Write federation manifest JSON to *target* and return the path."""
    path = os.fspath(target)
    if os.path.isdir(path):
        raise FederationConfigError(f"federation manifest export target is a directory: {path!r}")
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        raise FederationConfigError(f"federation manifest export directory does not exist: {parent!r}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            federation_manifest_document(manifest, include_derived=True),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return path


def validate_federation_cross_source_join_kind(kind: str) -> str:
    """Normalize and validate a declared cross-source join kind."""
    normalized = str(kind or "inner").strip().lower()
    if normalized not in FEDERATION_CROSS_SOURCE_JOIN_KINDS:
        accepted = ", ".join(sorted(FEDERATION_CROSS_SOURCE_JOIN_KINDS))
        raise FederationDeclarationError(f"cross_source_join kind {kind!r} is invalid; accepted values: {accepted}")
    return normalized


def validate_manifest_cross_source_joins(manifest: FederationManifest) -> None:
    """Resolve and validate declared cross-source join endpoints."""
    if not manifest.sources:
        return
    for join in manifest.cross_source_joins:
        validate_federation_cross_source_join_kind(join.kind)
        left_ref = resolve_federation_qualified_ref(join.left, manifest=manifest)
        right_ref = resolve_federation_qualified_ref(join.right, manifest=manifest)
        if left_ref.source_id == right_ref.source_id:
            raise FederationDeclarationError(f"cross_source_join must span sources: {join.left!r} and {join.right!r}")
        if manifest.table_namespace.get(left_ref.table, left_ref.source_id) != left_ref.source_id:
            raise FederationDeclarationError(
                f"cross_source_join left table {left_ref.table!r} is not owned by {left_ref.source_id!r}"
            )
        if manifest.table_namespace.get(right_ref.table, right_ref.source_id) != right_ref.source_id:
            raise FederationDeclarationError(
                f"cross_source_join right table {right_ref.table!r} is not owned by {right_ref.source_id!r}"
            )


def parse_federation_mappings(raw: Mapping[str, Any] | str | bytes | None) -> FederationMappings:
    """Parse a federation mapping sidecar; empty mappings when *raw* is None."""
    if raw is None:
        return FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FederationConfigError(f"malformed federation declaration: invalid mappings JSON: {exc}") from exc
    else:
        payload = dict(raw)
    if not isinstance(payload, dict):
        raise FederationConfigError("federation mappings must be a JSON object")
    _reject_unknown_keys(payload, FEDERATION_MAPPINGS_TOP_LEVEL_KEYS, label="federation mappings")
    version = int(payload.get("version", FEDERATION_MAPPINGS_VERSION))
    if version < FEDERATION_MAPPINGS_MIN_VERSION or version > FEDERATION_MAPPINGS_VERSION:
        raise FederationConfigError(
            f"unsupported federation mappings version {version}; "
            f"supported versions are {FEDERATION_MAPPINGS_MIN_VERSION} through {FEDERATION_MAPPINGS_VERSION}"
        )
    logical_columns: list[LogicalColumnMapping] = []
    for entry in payload.get("logical_columns", []) or []:
        if not isinstance(entry, dict):
            raise FederationConfigError("logical_columns entries must be objects")
        _reject_unknown_keys(entry, FEDERATION_MAPPINGS_LOGICAL_COLUMN_KEYS, label="logical_columns entry")
        logical = str(entry.get("logical", "") or "").strip()
        members_raw = entry.get("members", [])
        if not logical or not isinstance(members_raw, list) or not members_raw:
            raise FederationConfigError("logical_columns entry requires logical and members")
        members = tuple(str(m).strip() for m in members_raw if str(m).strip())
        role = str(entry.get("role", "join_key") or "join_key").strip()
        if role != "join_key":
            raise FederationDeclarationError(
                f"logical_columns entry {logical!r} has unsupported role {role!r}; only 'join_key' is supported"
            )
        unify = bool(entry.get("unify_in_graph", False))
        if role == "join_key" and len(members) >= 2 and not unify:
            raise FederationDeclarationError(
                f"logical_columns entry {logical!r} with role 'join_key' spanning multiple members "
                "requires unify_in_graph: true"
            )
        logical_columns.append(LogicalColumnMapping(logical=logical, members=members, role=role, unify_in_graph=unify))
    logical_tables: list[LogicalTableMapping] = []
    for entry in payload.get("logical_tables", []) or []:
        if not isinstance(entry, dict):
            raise FederationConfigError("logical_tables entries must be objects")
        _reject_unknown_keys(entry, FEDERATION_MAPPINGS_LOGICAL_TABLE_KEYS, label="logical_tables entry")
        logical = str(entry.get("logical", "") or "").strip()
        semantics_raw = str(entry.get("semantics", "") or "").strip().lower()
        if semantics_raw not in ("union", "replica"):
            raise FederationConfigError(
                f"logical_tables semantics {semantics_raw!r} is not supported for {logical!r}; "
                "supported values: union, replica"
            )
        semantics = cast(Literal["union", "replica"], semantics_raw)
        members_raw = entry.get("members", [])
        if not logical or not isinstance(members_raw, list) or not members_raw:
            raise FederationConfigError("logical_tables entry requires logical and members")
        table_members: list[LogicalTableMember] = []
        for member in members_raw:
            if not isinstance(member, dict):
                raise FederationConfigError("logical_tables member must be an object")
            _reject_unknown_keys(member, FEDERATION_MAPPINGS_TABLE_MEMBER_KEYS, label="logical_tables member")
            source = str(member.get("source", "") or "").strip()
            table = str(member.get("table", "") or "").strip()
            columns_raw = member.get("columns", {})
            if not source or not table or not isinstance(columns_raw, dict):
                raise FederationConfigError("logical_tables member requires source, table, columns")
            columns = {str(k): str(v) for k, v in columns_raw.items()}
            table_members.append(LogicalTableMember(source=source, table=table, columns=columns))
        authoritative_source = str(entry.get("authoritative_source", "") or "").strip()
        if semantics == "replica":
            if not authoritative_source:
                raise FederationConfigError(
                    f"logical_tables replica mapping requires authoritative_source: {logical!r}"
                )
            member_sources = {m.source for m in table_members}
            if authoritative_source not in member_sources:
                raise FederationConfigError(
                    f"logical_tables authoritative_source {authoritative_source!r} is not a member of {logical!r}"
                )
        elif authoritative_source:
            raise FederationConfigError(f"logical_tables union mapping must not set authoritative_source: {logical!r}")
        logical_tables.append(
            LogicalTableMapping(
                logical=logical,
                members=tuple(sorted(table_members, key=lambda m: (m.source, m.table))),
                semantics=semantics,
                authoritative_source=authoritative_source,
            )
        )
    return FederationMappings(
        version=version, logical_columns=tuple(logical_columns), logical_tables=tuple(logical_tables)
    )


def load_federation_manifest_from_path(path: str) -> FederationManifest:
    """Load ``federation_manifest.json`` from *path*."""
    if not path or not os.path.isfile(path):
        raise FederationConfigError(f"federation manifest not found: {path!r}")
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    try:
        return parse_federation_manifest(raw)
    except FederationConfigError as exc:
        raise FederationConfigError(f"malformed federation declaration in declarations file {path!r}: {exc}") from exc


def load_federation_mappings_from_path(path: str) -> FederationMappings:
    """Load ``federation_mappings.json`` from *path*."""
    if not path or not os.path.isfile(path):
        raise FederationConfigError(f"federation mappings not found: {path!r}")
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    try:
        return parse_federation_mappings(raw)
    except FederationConfigError as exc:
        raise FederationConfigError(f"malformed federation declaration in declarations file {path!r}: {exc}") from exc


def parse_federation_declaration(
    raw: Mapping[str, Any] | str | bytes | None,
    *,
    include_derived_roster: bool = False,
) -> tuple[FederationManifest, FederationMappings]:
    """Parse a unified federation declaration into manifest and mapping sections."""
    if raw is None:
        raise FederationConfigError("federation declaration must be a JSON object")
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FederationConfigError(f"malformed federation declaration: invalid JSON: {exc}") from exc
    else:
        payload = dict(raw)
    if not isinstance(payload, dict):
        raise FederationConfigError("federation declaration must be a JSON object")
    _reject_unknown_keys(payload, FEDERATION_DECLARATION_TOP_LEVEL_KEYS, label="federation declaration")
    declared_version = int(payload.get("version", FEDERATION_DECLARATION_VERSION))
    if declared_version < FEDERATION_DECLARATION_VERSION:
        raise FederationDeclarationError(
            f"unsupported federation declaration version {declared_version}; "
            f"minimum supported version is {FEDERATION_DECLARATION_VERSION}"
        )
    if declared_version > FEDERATION_DECLARATION_VERSION:
        raise FederationDeclarationError(
            f"unsupported federation declaration version {declared_version}; "
            f"maximum supported version is {FEDERATION_DECLARATION_VERSION}"
        )
    manifest_payload = {k: payload[k] for k in payload if k in FEDERATION_MANIFEST_TOP_LEVEL_KEYS}
    mappings_payload = {k: payload[k] for k in payload if k in frozenset({"logical_columns", "logical_tables"})}
    manifest = parse_federation_manifest(manifest_payload, include_derived_roster=include_derived_roster)
    mappings = (
        parse_federation_mappings(mappings_payload)
        if mappings_payload
        else FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    )
    return manifest, mappings


def load_federation_declaration_from_path(path: str) -> tuple[FederationManifest, FederationMappings]:
    """Load a unified federation declaration from *path*."""
    if not path or not os.path.isfile(path):
        raise FederationConfigError(f"federation declaration not found: {path!r}")
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    try:
        return parse_federation_declaration(raw)
    except FederationConfigError as exc:
        raise FederationConfigError(f"malformed federation declaration in declarations file {path!r}: {exc}") from exc


def federation_manifest_is_active(raw: Mapping[str, Any] | str | None) -> bool:
    """Return True when a non-empty federation manifest is supplied."""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(str(raw).strip())
    return bool(raw)


def owner_is_aether_federation(owner: Any) -> bool:
    """Return True when *owner* is a public :class:`~aetherdialect.AetherFederation` instance."""
    return getattr(owner, "_is_aether_federation", False) is True


def derive_table_namespace(
    member_graphs: Mapping[str, SchemaGraph],
    mappings: FederationMappings | None = None,
) -> dict[str, str]:
    """Build a logical table name to source id index from member graphs."""
    namespace: dict[str, str] = {}
    logical_owners: dict[str, set[str]] = {}
    for source_id in sorted(member_graphs):
        graph = member_graphs[source_id]
        for table_name, table in graph.tables.items():
            logical = str(table.name or table_name).strip()
            if not logical:
                continue
            owner = str(table.source_id or source_id).strip() or source_id
            logical_owners.setdefault(logical, set()).add(owner)
    for logical in sorted(logical_owners):
        owners = logical_owners[logical]
        if len(owners) == 1:
            namespace[logical] = next(iter(owners))
            continue
        resolved = _namespace_owner_for_duplicate_logical(logical, frozenset(owners), mappings)
        if resolved is None:
            member_desc = ", ".join(f"{sid}.{logical}" for sid in sorted(owners))
            raise FederationConfigError(
                "table name collision across federation members; "
                f"resolve with a logical_tables mapping or an explicit alias: {member_desc}"
            )
        namespace[logical] = resolved
    return namespace


def _namespace_owner_for_duplicate_logical(
    logical: str,
    owners: frozenset[str],
    mappings: FederationMappings | None,
) -> str | None:
    """Resolve namespace owner when the same logical table name appears on multiple members."""
    if mappings is None:
        return None
    for table_map in mappings.logical_tables:
        if table_map.logical != logical:
            continue
        member_sources = frozenset(member.source for member in table_map.members)
        if not owners.issubset(member_sources):
            continue
        if table_map.semantics == "replica":
            auth = (table_map.authoritative_source or "").strip()
            if auth:
                return auth
            return sorted(member_sources)[0]
        if table_map.semantics == "union":
            return sorted(member_sources)[0]
    return None


def _namespace_from_aliases_and_members(
    member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest
) -> dict[str, str]:
    """Build table_namespace when explicit aliases disambiguate colliding physical names."""
    namespace: dict[str, str] = {alias.alias: alias.source for alias in manifest.aliases}
    aliased_phys = {(alias.source, alias.table) for alias in manifest.aliases}
    for source_id in sorted(member_graphs):
        graph = member_graphs[source_id]
        for phys_name, table in graph.tables.items():
            if (source_id, phys_name) in aliased_phys:
                continue
            logical = str(table.name or phys_name).strip()
            if not logical:
                continue
            if logical in namespace and namespace[logical] != source_id:
                member_desc = ", ".join(f"{sid}.{logical}" for sid in sorted({namespace[logical], source_id}))
                raise FederationConfigError(
                    "table name collision across federation members; "
                    f"resolve with a logical_tables mapping or an explicit alias: {member_desc}"
                )
            namespace[logical] = source_id
    return namespace


def _namespace_from_composite_schema(schema: SchemaGraph) -> dict[str, str]:
    namespace: dict[str, str] = {}
    for table_name, table in schema.tables.items():
        logical = str(table.name or table_name).strip()
        source_id = str(table.source_id or "").strip()
        if logical and source_id:
            namespace[logical] = source_id
    return namespace


def _manifest_with_derived_roster(
    manifest: FederationManifest,
    *,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    composite: SchemaGraph | None = None,
    mappings: FederationMappings | None = None,
) -> FederationManifest:
    """Attach member-derived roster fields to a declaration-only manifest."""
    if manifest.sources and manifest.table_namespace:
        return manifest
    if member_graphs:
        if manifest.table_namespace:
            namespace = manifest.table_namespace
        elif manifest.aliases:
            namespace = _namespace_from_aliases_and_members(member_graphs, manifest)
        else:
            namespace = derive_table_namespace(member_graphs, mappings)
        source_ids = sorted(member_graphs)
    elif composite is not None:
        namespace = manifest.table_namespace or _namespace_from_composite_schema(composite)
        source_ids = sorted({sid for sid in namespace.values() if sid})
    else:
        return manifest
    sources = manifest.sources or tuple(
        FederationSourceBinding(
            source_id=str(source_id), engine="duckdb", connection=str(source_id), context="master", role="owner"
        )
        for source_id in source_ids
    )
    return replace(manifest, sources=sources, table_namespace=namespace)


def _stamp_member_graph_source_id(graph: SchemaGraph, source_id: str) -> SchemaGraph:
    tables = {name: replace(table, source_id=source_id or table.source_id) for name, table in graph.tables.items()}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=graph.schema_graph_id,
        effective_structural_hash=graph.effective_structural_hash,
        structural_hash=graph.structural_hash,
        scope_hash=graph.scope_hash,
        schema_stats=graph.schema_stats,
        enum_values=dict(graph.enum_values or {}),
        deny_columns=dict(graph.deny_columns or {}),
        disallowed_columns=dict(graph.disallowed_columns or {}),
        created_at=graph.created_at,
        notes_sha256=graph.notes_sha256,
        profiling_hash=graph.profiling_hash,
    )


def _member_graph_is_profiled(graph: SchemaGraph) -> bool:
    return bool(str(graph.profiling_hash or "").strip())


def assert_federation_member_graph_profiled(connection_name: str, graph: SchemaGraph) -> None:
    """Raise when a member graph was not profiled before federation composition."""
    if _member_graph_is_profiled(graph):
        return
    raise FederationMemberUnprofilableError(
        f"federation member {connection_name!r} schema is not profiled; "
        "initialize the member engine before composing the federation",
        source_id=connection_name,
    )


def member_graphs_from_engines(members: Mapping[str, Any]) -> dict[str, SchemaGraph]:
    """Collect per-member schema graphs keyed by connection name."""
    graphs: dict[str, SchemaGraph] = {}
    for connection_name, engine in sorted(members.items()):
        graph = getattr(engine, "_schema_graph", None)
        if not isinstance(graph, SchemaGraph):
            raise FederationConfigError(f"member {connection_name!r} does not expose a schema graph")
        stamped = _stamp_member_graph_source_id(graph, str(connection_name))
        assert_federation_member_graph_profiled(str(connection_name), stamped)
        grants = introspect_member_effective_grants(engine)
        if grants is not None:
            object.__setattr__(stamped, "_member_effective_grants", grants)
        graphs[str(connection_name)] = stamped
    return graphs


class _FederationMemberEngine(Protocol):
    """Minimal member-engine surface used when deriving federation source bindings."""

    dialect: str
    _connection: object
    _context_name: object
    _schema_role: object
    _runtime_config: EngineRuntimeConfig | object | None


def _assert_federation_member_role_is_owner(connection_name: str, role: SchemaRole) -> None:
    """Refuse federation members that are not owner engines."""
    if role != "owner":
        raise FederationConfigError(f"federation member {connection_name!r} must be an owner engine; got role {role!r}")


def binding_from_member_engine(connection_name: str, engine: _FederationMemberEngine) -> FederationSourceBinding:
    """Derive a federation source binding from a configured member engine."""
    engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
    if not engine_type:
        runtime_cfg = getattr(engine, "_runtime_config", None)
        engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
    source_id = str(connection_name).strip()
    federation_handle = str(getattr(engine, "_connection", None) or "").strip()
    raw_named = getattr(engine, "_named_connection", None)
    named_connection = raw_named.strip() if isinstance(raw_named, str) else ""
    if federation_handle and federation_handle != source_id:
        raise FederationConfigError(
            f"federation member key {source_id!r} is the source_id used in declarations and joins; "
            f"engine federation handle {federation_handle!r} must match that key "
            f"(named TOML connection {named_connection!r} is unrelated)"
        )
    connection = named_connection or source_id
    context = str(getattr(engine, "_context_name", "master") or "master").strip().lower() or "master"
    role_raw = str(getattr(engine, "_schema_role", "owner") or "owner").strip().lower()
    if role_raw not in ("owner", "consumer"):
        raise FederationConfigError(f"member {connection_name!r} has invalid role: {role_raw!r}")
    role = cast(SchemaRole, role_raw)
    _assert_federation_member_role_is_owner(source_id, role)
    return FederationSourceBinding(
        source_id=source_id,
        engine=engine_type or "duckdb",
        connection=connection,
        context=context,
        role=cast(SchemaRole, role_raw),
    )


def build_federation_manifest_from_members(
    members: Mapping[str, Any],
    *,
    declaration: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    mappings: FederationMappings | None = None,
) -> FederationManifest:
    """Merge an authored federation declaration with member-derived roster fields."""
    fed_id = str(declaration.federation_id or "").strip()
    if not fed_id:
        raise FederationConfigError("federation manifest requires federation_id")
    if not members:
        raise FederationConfigError("federation requires at least one member engine")
    graphs = dict(member_graphs) if member_graphs is not None else member_graphs_from_engines(members)
    if not graphs:
        raise FederationConfigError("federation requires member schema graphs")
    sources = tuple(binding_from_member_engine(name, engine) for name, engine in sorted(members.items()))
    if sources and all(is_file_engine(binding.engine) for binding in sources):
        raise FederationDeclarationError(
            "A federation whose members are all file engines is not supported; "
            "load uploads into one CSV engine instead."
        )
    validate_federation_source_slug_uniqueness(sources)
    member_ids = set(members)
    for alias in declaration.aliases:
        if alias.source not in member_ids:
            raise FederationConfigError(f"alias {alias.alias!r} references unknown source_id: {alias.source!r}")
    return _manifest_with_derived_roster(
        declaration,
        member_graphs=graphs,
        mappings=mappings,
    )


def _logical_table_mapping_document(table: LogicalTableMapping) -> dict[str, Any]:
    """Serialize one logical table mapping, omitting empty replica-only fields."""
    entry: dict[str, Any] = {
        "logical": table.logical,
        "semantics": table.semantics,
        "members": [
            {"source": m.source, "table": m.table, "columns": dict(m.columns)}
            for m in sorted(table.members, key=lambda m: (m.source, m.table))
        ],
    }
    if table.authoritative_source:
        entry["authoritative_source"] = table.authoritative_source
    return entry


def federation_mappings_document(mappings: FederationMappings) -> dict[str, Any]:
    """Serialize federation mappings for export or persistence."""
    return {
        "version": mappings.version,
        "logical_columns": [
            {
                "logical": c.logical,
                "members": sorted(c.members),
                "role": c.role,
                "unify_in_graph": c.unify_in_graph,
            }
            for c in sorted(mappings.logical_columns, key=lambda c: c.logical)
        ],
        "logical_tables": [
            _logical_table_mapping_document(t) for t in sorted(mappings.logical_tables, key=lambda t: t.logical)
        ],
    }


def export_federation_mappings_document(mappings: FederationMappings, target: str | os.PathLike[str]) -> str:
    """Write federation mappings JSON to *target* and return the path."""
    path = os.fspath(target)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(federation_mappings_document(mappings), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def federation_declaration_document(
    manifest: FederationManifest,
    mappings: FederationMappings,
) -> dict[str, Any]:
    """Serialize a unified federation declaration in authored shape (no derived roster fields)."""
    payload = federation_manifest_document(manifest, include_derived=False)
    payload["version"] = FEDERATION_DECLARATION_VERSION
    mappings_doc = federation_mappings_document(mappings)
    payload["logical_columns"] = mappings_doc.get("logical_columns") or []
    payload["logical_tables"] = mappings_doc.get("logical_tables") or []
    return payload


def export_federation_declaration(
    manifest: FederationManifest,
    mappings: FederationMappings,
    target: str | os.PathLike[str],
) -> str:
    """Write a unified federation declaration JSON to *target* and return the path."""
    path = os.fspath(target)
    if os.path.isdir(path):
        raise FederationConfigError(f"federation declaration export target is a directory: {path!r}")
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        raise FederationConfigError(f"federation declaration export directory does not exist: {parent!r}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            federation_declaration_document(manifest, mappings),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return path


def _plan_template_row_references_sources(row: Mapping[str, Any], source_ids: set[str]) -> bool:
    """Return True when a stored plan template row references any of *source_ids*."""
    steps_raw = row.get("step_fingerprints", [])
    if not isinstance(steps_raw, list):
        return False
    for entry in steps_raw:
        if isinstance(entry, (list, tuple)) and entry:
            if str(entry[0]) in source_ids:
                return True
    return False


def federation_drifted_member_source_ids(
    federation_dir: str,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
) -> set[str]:
    """Return member source ids whose pinned identity no longer matches stored artifacts."""
    stored = _load_federation_artifact_manifest_dict(federation_artifact_paths(federation_dir)["artifact_manifest"])
    if stored is None:
        return set()
    live_by_source = {row[0]: row for row in federation_member_hash_tuple(member_graphs, manifest)}
    stored_members = stored.get("federation_members")
    if not isinstance(stored_members, list):
        return set(live_by_source)
    drifted: set[str] = set()
    stored_ids: set[str] = set()
    for entry in stored_members:
        try:
            normalized = _normalize_stored_member_hash_row(entry)
        except FederationConfigError:
            return set(live_by_source)
        source_id = normalized[0]
        stored_ids.add(source_id)
        live_row = live_by_source.get(source_id)
        if live_row is None or tuple(live_row) != normalized:
            drifted.add(source_id)
    for source_id in live_by_source:
        if source_id not in stored_ids:
            drifted.add(source_id)
    if mappings is not None:
        stored_mappings_hash = str(stored.get("mappings_hash", "") or "")
        stored_manifest_hash = str(stored.get("manifest_hash", "") or "")
        if stored_mappings_hash != mappings_hash(mappings) or stored_manifest_hash != manifest_hash(manifest):
            drifted.update(live_by_source)
    return drifted


def prune_federation_plan_templates_on_drift(
    federation_dir: str,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
) -> None:
    """Drop federation plan templates that reference members whose identity has drifted."""
    drifted = federation_drifted_member_source_ids(federation_dir, member_graphs, manifest, mappings)
    if drifted:
        prune_federation_plan_templates_for_sources(federation_dir, drifted)


def prune_federation_plan_templates_for_sources(federation_dir: str, removed_source_ids: set[str]) -> None:
    """Drop federation plan templates that reference a removed member source."""
    if not removed_source_ids:
        return
    paths = federation_artifact_paths(federation_dir)
    path = paths["plan_templates"]
    if not os.path.isfile(path):
        return
    with artifact_lock(federation_dir):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise FederationConfigError(f"corrupt federation plan templates file: {path!r}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise FederationConfigError(f"federation plan templates file at {path!r} is not a JSON object")
        kept = {
            plan_id: row
            for plan_id, row in loaded.items()
            if not (isinstance(row, dict) and _plan_template_row_references_sources(row, removed_source_ids))
        }
        if len(kept) == len(loaded):
            return
        if kept:
            _write_federation_json_atomic(path, kept)
        else:
            os.remove(path)


def _engine_context_for_schema_usability(ctx: FederationContext | EngineContext) -> EngineContext:
    """Adapt federation or engine scope for composite usability checks."""
    if isinstance(ctx, EngineContext):
        return ctx
    return EngineContext(
        allow_objects=ctx.allow_objects,
        deny_objects=ctx.deny_objects,
        deny_columns=ctx.deny_columns,
        allow_columns=ctx.allow_columns,
        include=ctx.include,
        notes_file=ctx.notes_file,
    )


def collect_federation_description_forbidden_tokens(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
    composite_names: Mapping[tuple[str, str], str],
) -> frozenset[str]:
    """Collect member identifiers and physical names that must not reach composite prompts."""
    tokens: set[str] = set()
    for binding in manifest.sources:
        sid = str(binding.source_id or "").strip()
        if sid:
            tokens.add(sid)
    for source_id, graph in member_graphs.items():
        sid = str(source_id or "").strip()
        if sid:
            tokens.add(sid)
        for phys_table, table in graph.tables.items():
            composite = composite_names.get((source_id, phys_table), phys_table)
            if phys_table != composite:
                tokens.add(phys_table)
            original = (table.original_name or "").strip()
            if original and original.lower() != table.name.lower():
                tokens.add(original)
            for col in table.columns.values():
                col_original = (col.original_name or "").strip()
                if col_original and col_original.lower() != col.name.lower():
                    tokens.add(col_original)
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            for logical_col, phys_col in member.columns.items():
                if phys_col and phys_col != logical_col:
                    tokens.add(phys_col)
    return frozenset(tokens)


def raise_if_member_notes_name_federation_sources(notes_file: str | None, source_ids: Iterable[str]) -> None:
    """Reject member notes prose that names a federation source identifier."""
    path = str(notes_file or "").strip()
    if not path:
        return
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return
    try:
        content = Path(expanded).read_text(encoding="utf-8")
    except OSError:
        return
    tokens = {str(source_id or "").strip() for source_id in source_ids}
    tokens.discard("")
    for token in sorted(tokens, key=len, reverse=True):
        if token and token in content:
            raise ConfigError(f"member notes must not name a source or member; found {token!r}")


def raise_if_descriptions_name_federation_sources(
    member_graphs: Mapping[str, SchemaGraph],
    source_ids: Iterable[str],
    *,
    context: str = "federation member description",
) -> None:
    """Reject member-authored descriptions that name a federation source identifier."""
    tokens = {str(source_id or "").strip() for source_id in source_ids}
    tokens.discard("")
    if not tokens:
        return
    for graph in member_graphs.values():
        for table in graph.tables.values():
            targets = [table, *table.columns.values()]
            for target in targets:
                desc = str(getattr(target, "description", "") or "")
                if not desc:
                    continue
                for token in tokens:
                    if token in desc:
                        raise ConfigError(f"{context} must not name a source or member; found {token!r}")


def raise_if_schema_graph_descriptions_contain_member_identifiers(
    graph: SchemaGraph,
    forbidden_tokens: frozenset[str],
    *,
    context: str = "composite description",
) -> None:
    """Reject composite descriptions that name a federation member identifier."""
    if not forbidden_tokens:
        return
    for table in graph.tables.values():
        if table.description:
            hits = description_neutrality_violations(table.description, forbidden_tokens)
            if hits:
                raise ConfigError(f"{context} must not name a source or member; found {hits[0]!r}")
        for col in table.columns.values():
            if col.description:
                hits = description_neutrality_violations(col.description, forbidden_tokens)
                if hits:
                    raise ConfigError(f"{context} must not name a source or member; found {hits[0]!r}")


def _apply_composite_federation_scope(composite: SchemaGraph, scope_ctx: FederationContext | EngineContext) -> None:
    """Apply federation master-scope denials to the composite catalog."""
    usability_ctx = _engine_context_for_schema_usability(scope_ctx)
    if usability_ctx.deny_objects:
        apply_deny_objects_filter(composite, usability_ctx)
    if not usability_ctx.deny_columns:
        return
    deny_by_table = deny_columns_by_table(composite, usability_ctx)
    for canon_tbl, cols in deny_by_table.items():
        composite.deny_columns.setdefault(canon_tbl, set()).update(cols)
    for canon_tbl, cols in deny_by_table.items():
        tbl = composite.tables.get(canon_tbl)
        if tbl is None:
            continue
        for col_name in cols:
            tbl.columns.pop(col_name, None)
        tbl.primary_key = [c for c in tbl.primary_key if c in tbl.columns]
    prune_foreign_keys_after_column_removal(composite)
    composite.join_paths_multi = recompute_join_paths_multi(composite.tables)


def compose_composite_graph(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
    *,
    notes_content: str | None = None,
    llm_classify: Callable[..., Any] | None = None,
    master_context: FederationContext | EngineContext | None = None,
    member_engines: Mapping[str, Any] | None = None,
    member_effective_grants: Mapping[str, MemberEffectiveGrants] | None = None,
) -> SchemaGraph:
    """Build the disjoint union composite graph for federation intent creation."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    manifest = _manifest_with_derived_roster(manifest, member_graphs=member_graphs, mappings=mappings)
    member_graphs = _filter_member_graphs_to_allow_scope(member_graphs, manifest, mappings)
    pipeline_trace(
        FEDERATION_COMPOSITION_PHASE_A,
        lambda: f"sources={sorted(member_graphs)} federation_id={manifest.federation_id}",
    )
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_A)
    for source_id in _manifest_source_ids(manifest, member_graphs):
        if source_id not in member_graphs:
            raise FederationConfigError(f"member graph missing for source_id {source_id!r}")
        stamp_federation_member_graph(
            member_graphs[source_id],
            federation_id=manifest.federation_id,
            source_id=source_id,
            engine=_manifest_engine_for_source(manifest, source_id),
        )
    composite_names = _resolve_composite_table_names(member_graphs, manifest)
    validate_federation_mapping_members(member_graphs, mappings, composite_names, manifest)
    validate_declared_objects_against_member_grants(
        member_graphs,
        manifest,
        mappings,
        composite_names,
        member_engines=member_engines,
        member_effective_grants=member_effective_grants,
    )
    drift_messages = rescore_declared_mapping_drift(mappings, member_graphs, manifest)
    if drift_messages:
        drift_summary = "; ".join(drift_messages)
        pipeline_trace(DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT, lambda: drift_summary)
        notify(
            drift_summary,
            stage="federation",
            code=DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT,
            level="error",
            source_id=str(manifest.federation_id or ""),
            details=(("phase", "compose"), ("summary", drift_summary)),
        )
        raise FederationConfigError(f"federation declared mapping drift: {drift_summary}")
    pipeline_trace(FEDERATION_COMPOSITION_PHASE_B, lambda: f"composite_tables={len(composite_names)}")
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_B)
    merged_tables = _compose_namespaced_tables(member_graphs, manifest, mappings, composite_names)
    pipeline_trace(
        FEDERATION_COMPOSITION_PHASE_C,
        lambda: f"logical_tables={len(mappings.logical_tables)} merged_tables={len(merged_tables)}",
    )
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_C)
    _apply_logical_table_collapse(merged_tables, mappings.logical_tables, composite_names)
    pipeline_trace(
        FEDERATION_COMPOSITION_PHASE_D,
        lambda: f"logical_columns={len(mappings.logical_columns)} tables={len(merged_tables)}",
    )
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_D)
    _apply_logical_column_unification(merged_tables, mappings.logical_columns, manifest, mappings)
    pipeline_trace(FEDERATION_COMPOSITION_PHASE_E, lambda: f"cross_source_joins={len(manifest.cross_source_joins)}")
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_E)
    cross_edges = _materialize_cross_source_edges(manifest, mappings)
    for edge in cross_edges:
        src = merged_tables.get(edge.src_table)
        if src is not None:
            src.foreign_keys = list(src.foreign_keys) + [edge]
    for tbl in merged_tables.values():
        tbl.foreign_keys = sorted(tbl.foreign_keys, key=lambda e: (e.src_table, tuple(e.src_cols), e.dst_table))
    _sanitize_foreign_keys(merged_tables)
    join_paths_multi = recompute_join_paths_multi(merged_tables)
    deny_columns, disallowed_columns = _merge_member_scope_deny(member_graphs, composite_names)
    member_revision = max(
        (int(getattr(graph, "schema_revision", 0) or 0) for graph in member_graphs.values()), default=0
    )
    composite = SchemaGraph(
        tables=merged_tables,
        join_paths_multi=join_paths_multi,
        created_at=_derive_composite_created_at(member_graphs),
        deny_columns=deny_columns,
        disallowed_columns=disallowed_columns,
        enum_values=_merge_enum_values(member_graphs),
        schema_revision=member_revision,
    )
    truncated_enums = sorted(
        ename for ename, values in (composite.enum_values or {}).items() if len(values) > FEDERATION_ENUM_PROMPT_CAP
    )
    if truncated_enums:
        pipeline_trace(
            DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED,
            lambda: f"cap={FEDERATION_ENUM_PROMPT_CAP} types={','.join(truncated_enums)}",
        )
    pipeline_trace(
        FEDERATION_COMPOSITION_PHASE_F, lambda: f"tables={len(composite.tables)} join_paths={len(join_paths_multi)}"
    )
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_F)
    reconcile_composite_classifications(
        composite, member_graphs, mappings, manifest=manifest, notes_content=notes_content, llm_classify=llm_classify
    )
    forbidden_description_tokens = collect_federation_description_forbidden_tokens(
        member_graphs,
        manifest,
        mappings,
        composite_names,
    )
    object.__setattr__(composite, "_federation_description_forbidden_tokens", forbidden_description_tokens)
    raise_if_schema_graph_descriptions_contain_member_identifiers(composite, forbidden_description_tokens)
    sanitize_schema_graph_descriptions(composite, forbidden_description_tokens)
    federation_notes_sha = hashlib.sha256(notes_content.encode("utf-8")).hexdigest() if notes_content else ""
    if federation_notes_sha:
        composite.notes_sha256 = federation_notes_sha
    pipeline_trace(
        FEDERATION_COMPOSITION_PHASE_G, lambda: f"federation_id={manifest.federation_id} members={len(member_graphs)}"
    )
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_G)
    _assign_composite_identity(composite, member_graphs, manifest, mappings, federation_notes_sha=federation_notes_sha)
    scope_ctx = master_context if master_context is not None else FederationContext()
    if isinstance(scope_ctx, FederationContext):
        validate_federation_context_against_mappings(scope_ctx, mappings)
    validate_scope_against_graph(composite, scope_ctx)
    _apply_composite_federation_scope(composite, scope_ctx)
    composite.refresh_schema_stats()
    object.__setattr__(
        composite, "_database_feature_capability_cache", intersect_member_database_feature_capabilities(member_graphs)
    )
    engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
    object.__setattr__(
        composite,
        "_dialect_capability_cache",
        intersect_member_dialect_capabilities(
            engine_types_by_source=engine_types,
            member_graphs=member_graphs,
        ),
    )
    pipeline_trace(FEDERATION_COMPOSITION_PHASE_H, lambda: f"schema_graph_id={composite.schema_graph_id}")
    emit_construction_phase(FEDERATION_COMPOSITION_PHASE_H)
    assert_composite_invariants(composite, member_graphs, manifest, mappings)
    mark_canonical_duplicates(composite)
    assign_column_ops(composite)
    redact_hidden_sensitivity_profile_values(composite)
    raise_if_schema_unusable(
        composite, _engine_context_for_schema_usability(scope_ctx), federation_composite=len(member_graphs) > 1
    )
    validate_cross_source_keys_on_graph(composite, manifest, mappings)
    return composite


def assert_composite_invariants(
    composite: SchemaGraph,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
) -> None:
    """Assert composition invariants once at the end of ``compose_composite_graph``."""
    if not composite.schema_graph_id:
        raise FederationInvariantError("composite schema_graph_id is unset after composition")
    if not composite.structural_hash or not composite.scope_hash or not composite.effective_structural_hash:
        raise FederationInvariantError("composite identity hashes are incomplete after composition")
    member_sources = {binding.source_id for binding in manifest.sources} or set(member_graphs)
    if member_sources != set(member_graphs):
        missing = sorted(member_sources - set(member_graphs))
        extra = sorted(set(member_graphs) - member_sources)
        if missing:
            raise FederationInvariantError(f"composite composition missing member graphs: {missing!r}")
        if extra:
            raise FederationInvariantError(f"composite composition has undeclared member graphs: {extra!r}")
    enum_keys = set((composite.enum_values or {}).keys())
    referenced_enum_keys: set[str] = set()
    for table_name, table in composite.tables.items():
        for pk_col in table.primary_key or ():
            if pk_col not in table.columns:
                raise FederationInvariantError(
                    f"composite table {table_name!r} primary key column {pk_col!r} missing from columns"
                )
        for col_key, col_meta in table.columns.items():
            if col_meta.name != col_key:
                raise FederationInvariantError(
                    f"composite table {table_name!r} column key {col_key!r} "
                    f"does not match metadata name {col_meta.name!r}"
                )
            if col_meta.enum_type_name:
                referenced_enum_keys.add(col_meta.enum_type_name)
        for edge in table.foreign_keys:
            if edge.dst_table not in composite.tables:
                raise FederationInvariantError(
                    f"foreign key on {table_name!r} references missing table {edge.dst_table!r}"
                )
            dst = composite.tables[edge.dst_table]
            for col in edge.dst_cols:
                if col not in dst.columns:
                    raise FederationInvariantError(
                        f"foreign key on {table_name!r} references missing column {edge.dst_table}.{col}"
                    )
            for col in edge.src_cols:
                if col not in table.columns:
                    raise FederationInvariantError(
                        f"foreign key on {table_name!r} references missing source column {col!r}"
                    )
        if table.member_source_ids and not table.primary_key:
            if not table.columns:
                raise FederationInvariantError(
                    f"collapsed composite table {table_name!r} has no columns after composition"
                )
    for source_id in member_sources:
        member_graph = member_graphs.get(source_id)
        if member_graph is None or not member_graph.tables:
            raise FederationInvariantError(f"member graph for source {source_id!r} has no tables")
    for denied_table, cols in (composite.deny_columns or {}).items():
        if denied_table not in composite.tables:
            raise FederationInvariantError(f"deny_columns references missing table {denied_table!r}")
        if not cols:
            raise FederationInvariantError(f"deny_columns entry for {denied_table!r} is empty")
    for disallowed_table, cols in (composite.disallowed_columns or {}).items():
        if disallowed_table not in composite.tables:
            raise FederationInvariantError(f"disallowed_columns references missing table {disallowed_table!r}")
        if not cols:
            raise FederationInvariantError(f"disallowed_columns entry for {disallowed_table!r} is empty")
    plain_enum_keys = {key for key in enum_keys if "::" not in key}
    unused_member_enum = plain_enum_keys - referenced_enum_keys
    if unused_member_enum:
        raise FederationInvariantError(
            f"composite enum types not referenced by columns: {sorted(unused_member_enum)!r}"
        )
    for table_map in mappings.logical_tables:
        if table_map.semantics not in ("union", "replica"):
            continue
        collapsed = composite.tables.get(table_map.logical)
        if collapsed is None:
            raise FederationInvariantError(f"logical table {table_map.logical!r} missing from composite after collapse")
        if table_map.semantics == "replica" and not (table_map.authoritative_source or "").strip():
            if len(table_map.members) > 1:
                raise FederationInvariantError(
                    f"replica logical table {table_map.logical!r} missing authoritative_source"
                )


def compare_replica_member_parity(
    mappings: FederationMappings, member_graphs: Mapping[str, SchemaGraph]
) -> tuple[str, ...]:
    """Return drift messages when replica logical tables disagree across members."""
    drift: list[str] = []
    for table_map in mappings.logical_tables:
        if table_map.semantics != "replica" or len(table_map.members) < 2:
            continue
        auth = (table_map.authoritative_source or "").strip()
        col_sets: dict[str, frozenset[str]] = {}
        for member in table_map.members:
            graph = member_graphs.get(member.source)
            if graph is None:
                drift.append(f"replica {table_map.logical!r}: missing graph for source {member.source!r}")
                continue
            tbl = graph.tables.get(member.table)
            if tbl is None:
                drift.append(f"replica {table_map.logical!r}: missing table {member.table!r} on {member.source!r}")
                continue
            mapped_cols = frozenset(member.columns.keys()) if member.columns else frozenset(tbl.columns)
            col_sets[member.source] = mapped_cols
        if not col_sets:
            continue
        reference_source = auth if auth in col_sets else next(iter(sorted(col_sets)))
        reference_cols = col_sets[reference_source]
        for source_id, cols in sorted(col_sets.items()):
            if source_id == reference_source:
                continue
            if cols != reference_cols:
                drift.append(
                    f"replica {table_map.logical!r}: column coverage drift between "
                    f"{reference_source!r} and {source_id!r}"
                )
    return tuple(drift)


def select_replica_member_source(table_map: LogicalTableMapping) -> str:
    """Return the replica member named by manifest ``authoritative_source``."""
    if table_map.semantics != "replica" or not table_map.members:
        return ""
    auth = (table_map.authoritative_source or "").strip()
    if not auth:
        raise FederationConfigError(f"replica logical table {table_map.logical!r} missing authoritative_source")
    return auth


def rescore_declared_mapping_drift(
    mappings: FederationMappings, member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest | None = None
) -> tuple[str, ...]:
    """Report drift across declared replica mappings and member graphs."""
    drift: list[str] = list(compare_replica_member_parity(mappings, member_graphs))
    for col_map in mappings.logical_columns:
        if len(col_map.members) < 2:
            continue
        overlap_samples: list[tuple[str, frozenset[str]]] = []
        for member_ref in col_map.members:
            table_name, column_name = split_qualified_column(member_ref, manifest=manifest)
            source_id = _physical_table_source(table_name, mappings)
            if not source_id:
                continue
            graph = member_graphs.get(source_id)
            if graph is None:
                drift.append(f"declared {col_map.logical!r}: missing graph for source {source_id!r}")
                continue
            table = graph.tables.get(table_name)
            if table is None:
                drift.append(f"declared {col_map.logical!r}: missing table {table_name!r} on {source_id!r}")
                continue
            meta = table.columns.get(column_name)
            if meta is None:
                drift.append(f"declared {col_map.logical!r}: missing column {member_ref!r} on {source_id!r}")
                continue
            sample = frozenset(str(v) for v in (meta.value_overlap_sample or []) if str(v))
            if sample:
                overlap_samples.append((source_id, sample))
        if len(overlap_samples) < 2:
            continue
        min_overlap = 1.0
        for left_idx in range(len(overlap_samples)):
            left_source, left_sample = overlap_samples[left_idx]
            for right_source, right_sample in overlap_samples[left_idx + 1 :]:
                ratio = _value_overlap_ratio(list(left_sample), list(right_sample))
                min_overlap = min(min_overlap, ratio)
                if ratio < FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR:
                    drift.append(
                        f"declared {col_map.logical!r}: value overlap rescoring drift between "
                        f"{left_source!r} and {right_source!r}"
                    )
    return tuple(drift)


def _physical_table_source(table_name: str, mappings: FederationMappings) -> str:
    """Return the member source id that owns a physical table name."""
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            if member.table == table_name:
                return member.source
    return ""


def _validate_federation_source_id_identifier(source_id: str) -> None:
    """Require member source ids to be safe unquoted SQL identifiers."""
    if not source_id.isidentifier():
        raise FederationDeclarationError(f"federation source_id must be identifier-safe: {source_id!r}")


def _member_enum_storage_key(source_id: str, enum_name: str) -> str:
    return f"{source_id}::{enum_name}"


def collapsed_member_physical_table_names(mappings: FederationMappings) -> dict[str, str]:
    """Map collapsed-away physical member table names to their logical table."""
    out: dict[str, str] = {}
    for table_map in mappings.logical_tables:
        if table_map.semantics not in ("union", "replica"):
            continue
        for member in table_map.members:
            if member.table != table_map.logical:
                out[member.table] = table_map.logical
    return out


def validate_federation_context_against_mappings(ctx: FederationContext, mappings: FederationMappings) -> None:
    """Reject federation scope entries that name collapsed physical tables or partially deny unions."""
    collapsed = collapsed_member_physical_table_names(mappings)
    for name in sorted(ctx.allow_objects | ctx.deny_objects):
        logical = collapsed.get(name)
        if logical is not None:
            raise ConfigError(
                f"federation context names collapsed member table {name!r}; use logical table {logical!r}"
            )
    for table_map in mappings.logical_tables:
        if table_map.semantics not in ("union", "replica"):
            continue
        member_tables = {member.table for member in table_map.members}
        denied = member_tables & set(ctx.deny_objects)
        if denied and denied != member_tables:
            raise ConfigError(
                f"federation context partially denies {table_map.semantics} logical table "
                f"{table_map.logical!r}; deny all member tables or none"
            )


def composite_schema_payload_counts(schema: SchemaGraph) -> dict[str, int]:
    """Count tables, columns, and enum types exposed in composite schema payloads."""
    enum_values = schema.enum_values or {}
    enum_label_count = sum(len(values) for values in enum_values.values())
    return {
        "tables": len(schema.tables),
        "columns": sum(len(table.columns) for table in schema.tables.values()),
        "enum_types": len(enum_values),
        "enum_labels": enum_label_count,
    }


def federation_composite_semantic_edges_hash(schema: SchemaGraph) -> str:
    """Stable hash of materialized cross-source relationship edges on *schema*."""
    payload: list[dict[str, Any]] = []
    for table_name in sorted(schema.tables):
        table = schema.tables[table_name]
        for edge in table.foreign_keys:
            if edge.inference_tag != InferenceTag.CROSS_SOURCE:
                continue
            payload.append(
                {
                    "src_table": edge.src_table,
                    "src_cols": list(edge.src_cols),
                    "dst_table": edge.dst_table,
                    "dst_cols": list(edge.dst_cols),
                }
            )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_federation_sql_history_warmup_allowed(engine: Any) -> None:
    """Refuse SQL-history and query-log warmup on a composite federated engine."""
    if owner_is_aether_federation(engine):
        raise FederationConfigError(
            "SQL history and query-log warmup are not available on a federated engine; "
            "run them on each source engine individually."
        )


def assert_query_log_warmup_allowed(engine: Any) -> None:
    """Refuse credentialed query-log warmup on a composite federated engine."""
    assert_federation_sql_history_warmup_allowed(engine)


def qsim_intent_eligible_on_federation(
    tables: Sequence[str], schema: SchemaGraph, manifest: FederationManifest, mappings: FederationMappings | None = None
) -> bool:
    """Return True when a QSim table set is answerable via federation decomposition."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    intent = RuntimeIntent(
        tables=list(tables), grain="row_level", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
    )
    sources = source_ids_for_intent(intent, schema, mappings, manifest)
    if len(sources) <= 1:
        return True
    plan = plan_federated_intent(intent, schema, manifest, mappings)
    return plan.ineligible_reason is None


def member_schema_slice(
    schema: SchemaGraph,
    source_id: str,
    *,
    manifest: FederationManifest | None = None,
    member_graph: SchemaGraph | None = None,
) -> SchemaGraph:
    """Return the composite-graph tables owned by *source_id* as a single-source subgraph."""
    tables: dict[str, TableMetadata] = {}
    for name, table in schema.tables.items():
        owned = table.source_id == source_id
        if not owned and table.member_source_ids and source_id in table.member_source_ids:
            owned = True
        if not owned and not table.source_id and manifest is not None:
            owned = manifest.table_namespace.get(name, "") == source_id
        if owned:
            copied = copy.deepcopy(table)
            if not copied.source_id:
                copied = replace(copied, source_id=source_id)
            tables[name] = copied
    slice_structural = structural_hash_fp(tables_structural_payload(tables))
    if member_graph is not None:
        schema_graph_id = str(member_graph.schema_graph_id or "")
        effective_hash = str(
            member_graph.effective_structural_hash
            or effective_structural_hash_fp(member_graph.structural_hash or slice_structural, member_graph.scope_hash)
            or ""
        )
    else:
        schema_graph_id = mint_schema_graph_id(
            seed_hex=hashlib.sha256(f"{source_id}|{slice_structural}".encode()).hexdigest()[:32],
            structural_hash=slice_structural,
        )
        effective_hash = effective_structural_hash_fp(slice_structural, "")
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=schema_graph_id,
        effective_structural_hash=effective_hash,
        structural_hash=slice_structural,
    )


def _table_source_id_for_manifest(schema: SchemaGraph, manifest: FederationManifest, table_name: str) -> str:
    table = schema.tables.get(table_name)
    if table is not None and table.source_id:
        return table.source_id
    return str(manifest.table_namespace.get(table_name, "") or "")


def _join_endpoint_is_many_side_of_fk(
    schema: SchemaGraph, table_name: str, column_name: str, peer_table_name: str
) -> bool:
    """Return True when *column_name* on *table_name* references *peer_table_name* as parent."""
    if _fk_points_to_parent(schema, table_name, peer_table_name, [column_name]):
        return True
    table = schema.tables.get(table_name)
    if table is None:
        return False
    for edge in table.foreign_keys:
        if edge.dst_table == peer_table_name and column_name in edge.src_cols:
            return True
    column = table.columns.get(column_name)
    if column is not None and column.fk_target and column.fk_target[0] == peer_table_name:
        return True
    return False


def _fk_points_to_parent(schema: SchemaGraph, child_tbl: str, parent_tbl: str, cols_on_child: list[str]) -> bool:
    validation_schema = importlib.import_module("aetherdialect._validation_schema")
    return validation_schema._fk_points_to_parent(child_tbl, parent_tbl, cols_on_child, schema)


def _validate_cross_source_inner_join_keys(
    schema: SchemaGraph,
    manifest: FederationManifest,
    *,
    left_qual: str,
    right_qual: str,
    logical: str,
) -> None:
    left_tbl, left_col = split_qualified_column(left_qual)
    right_tbl, right_col = split_qualified_column(right_qual)
    left_unique = _source_join_key_is_unique(
        schema, _table_source_id_for_manifest(schema, manifest, left_tbl), left_qual, manifest=manifest
    )
    right_unique = _source_join_key_is_unique(
        schema, _table_source_id_for_manifest(schema, manifest, right_tbl), right_qual, manifest=manifest
    )
    left_many_side = _join_endpoint_is_many_side_of_fk(schema, left_tbl, left_col, right_tbl)
    right_many_side = _join_endpoint_is_many_side_of_fk(schema, right_tbl, right_col, left_tbl)
    if not left_many_side:
        if left_unique is not True:
            _validate_cross_source_join_key_unique(schema, manifest, left_qual, peer_qual=right_qual, logical=logical)
    if not right_many_side:
        if right_unique is not True:
            _validate_cross_source_join_key_unique(schema, manifest, right_qual, peer_qual=left_qual, logical=logical)


def _validate_cross_source_join_key_unique(
    schema: SchemaGraph,
    manifest: FederationManifest,
    qual: str,
    *,
    peer_qual: str,
    logical: str,
) -> None:
    tbl, col = split_qualified_column(qual)
    src = _table_source_id_for_manifest(schema, manifest, tbl)
    if not src:
        raise FederationDeclarationError(
            f"cross-source join key member could not be resolved for {qual!r} "
            f"(join from {peer_qual!r}, logical column {logical!r})"
        )
    unique = _source_join_key_is_unique(schema, src, f"{tbl}.{col}", manifest=manifest)
    if unique is True:
        return
    if unique is False:
        raise FederationDeclarationError(
            f"cross-source join key {qual!r} is not unique on member {src!r} "
            f"(join from {peer_qual!r}, logical column {logical!r})"
        )
    raise FederationDeclarationError(
        f"cross-source join key uniqueness could not be established for {qual!r} "
        f"on member {src!r} (join from {peer_qual!r}, logical column {logical!r})"
    )


def validate_cross_source_keys_on_graph(
    schema: SchemaGraph, manifest: FederationManifest, mappings: FederationMappings | None = None
) -> None:
    """Fail when a declared cross-source join key is restricted or type- incompatible. Validates every unordered pair of members that share a logical join- key column (a clique, matching ``_materialize_cross_source_edges``), plus each declared ``cross_source_joins`` endpoint pair. Pair checks run in sorted member order so the first reported incompatible pair is deterministic. Raises: FederationDeclarationError: When a referenced table/column is missing, a join-key column is not ``sensitivity == none``, a member column type cannot be determined, or any pair of shared join-key columns has incompatible value types."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    refs: set[str] = set()
    pair_refs: list[tuple[str, str, str, str]] = []
    for join in manifest.cross_source_joins:
        left_tbl, left_col = _resolve_composite_join_ref(join.left, mappings, manifest)
        right_tbl, right_col = _resolve_composite_join_ref(join.right, mappings, manifest)
        for qual, tbl_name, col_name in (
            (join.left, left_tbl, left_col),
            (join.right, right_tbl, right_col),
        ):
            table = schema.tables.get(tbl_name)
            if table is None:
                continue
            pk = list(table.primary_key or ())
            if len(pk) > 1 and col_name in pk:
                raise FederationDeclarationError(
                    f"cross_source_join endpoint {qual!r} references {col_name!r} from composite primary "
                    f"key {pk!r}; cross-source joins support single-column keys only"
                )
        left_qual = f"{left_tbl}.{left_col}"
        right_qual = f"{right_tbl}.{right_col}"
        refs.add(left_qual)
        refs.add(right_qual)
        pair_refs.append((left_qual, right_qual, join.logical_key, join.kind))
    for col_map in mappings.logical_columns:
        if col_map.role == "join_key":
            for member in col_map.members:
                tbl, col = _resolve_composite_join_ref(member, mappings, manifest)
                refs.add(f"{tbl}.{col}")
            if len(col_map.members) >= 2:
                sorted_members = sorted(col_map.members)
                for i, left_member in enumerate(sorted_members):
                    left_tbl, left_col = _resolve_composite_join_ref(left_member, mappings, manifest)
                    left_qual = f"{left_tbl}.{left_col}"
                    for right_member in sorted_members[i + 1 :]:
                        right_tbl, right_col = _resolve_composite_join_ref(right_member, mappings, manifest)
                        if (left_tbl, left_col) == (right_tbl, right_col):
                            continue
                        pair_refs.append((left_qual, f"{right_tbl}.{right_col}", col_map.logical, "inner"))
    for qual in sorted(refs):
        tbl, col = split_qualified_column(qual)
        table = schema.tables.get(tbl)
        if table is None:
            raise FederationDeclarationError(f"cross-source reference missing table: {tbl!r}")
        meta = table.columns.get(col)
        if meta is None:
            raise FederationDeclarationError(f"cross-source reference missing column: {qual!r}")
        if meta.sensitivity != SensitivityClassification.NONE:
            raise FederationDeclarationError(f"cross-source join key must be sensitivity none: {qual!r}")
    seen_pairs: set[tuple[str, str]] = set()
    for left_qual, right_qual, logical, kind in pair_refs:
        pair_key = (min(left_qual, right_qual), max(left_qual, right_qual))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        left_tbl, left_col = split_qualified_column(left_qual)
        right_tbl, right_col = split_qualified_column(right_qual)
        left_table = schema.tables.get(left_tbl)
        right_table = schema.tables.get(right_tbl)
        if left_table is None:
            raise FederationDeclarationError(f"cross-source reference missing table: {left_tbl!r}")
        if right_table is None:
            raise FederationDeclarationError(f"cross-source reference missing table: {right_tbl!r}")
        left_meta = left_table.columns.get(left_col)
        right_meta = right_table.columns.get(right_col)
        if left_meta is None:
            raise FederationDeclarationError(f"cross-source reference missing column: {left_qual!r}")
        if right_meta is None:
            raise FederationDeclarationError(f"cross-source reference missing column: {right_qual!r}")
        left_type = (left_meta.value_type or "").strip()
        right_type = (right_meta.value_type or "").strip()
        if not left_type:
            raise FederationDeclarationError(
                f"cross-source join key type could not be determined for {left_qual!r} (logical column {logical!r})"
            )
        if not right_type:
            raise FederationDeclarationError(
                f"cross-source join key type could not be determined for {right_qual!r} (logical column {logical!r})"
            )
        if not fk_infer_value_types_compatible(left_meta, right_meta):
            raise FederationDeclarationError(
                f"cross-source join key type incompatible for logical column {logical!r}: "
                f"{left_qual!r} ({left_type}) vs {right_qual!r} ({right_type})"
            )
        join_kind = (kind or "inner").strip().lower()
        if join_kind == "inner":
            _validate_cross_source_inner_join_keys(
                schema,
                manifest,
                left_qual=left_qual,
                right_qual=right_qual,
                logical=logical,
            )
        elif join_kind == "left":
            _validate_cross_source_join_key_unique(
                schema,
                manifest,
                right_qual,
                peer_qual=left_qual,
                logical=logical,
            )
        elif join_kind == "right":
            _validate_cross_source_join_key_unique(
                schema,
                manifest,
                left_qual,
                peer_qual=right_qual,
                logical=logical,
            )
        else:
            _validate_cross_source_join_key_unique(
                schema,
                manifest,
                right_qual,
                peer_qual=left_qual,
                logical=logical,
            )


def _spanning_cte_decomposition_ineligible_reason(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
) -> str | None:
    """Refuse spanning CTE bodies whose clauses cannot be replayed at the coordinator."""
    spanning = _spanning_cte_names(intent.cte_steps or (), source_by_table)
    if not spanning:
        return None
    spanning_set = set(spanning)
    for cte in intent.cte_steps or []:
        name = str(cte.cte_name or "").strip()
        if not name or name not in spanning_set:
            continue
        if cte.where or cte.having or cte.group_by_cols:
            return (
                f"cross-source CTE {name!r} cannot be decomposed with filters or aggregates; "
                "ask per member or declare a simpler shape"
            )
        if cte.select_cols:
            return (
                f"cross-source CTE {name!r} cannot be decomposed with its own projection; "
                "ask per member or declare a simpler shape"
            )
    return None


def plan_federated_intent(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
    *,
    space: SpaceContext | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> FederatedPlan:
    """Decompose a validated intent into per-source steps and a combine specification."""
    emit_ask_phase(ASK_PHASE_I)
    debug(f"[{ASK_PHASE_I}] decompose tables={list(intent.tables or ())}")
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    manifest = _manifest_with_derived_roster(manifest, member_graphs=member_graphs, composite=schema)
    table_set = federation_table_set(intent, schema, manifest, mappings)
    tables = set(table_set.tables)
    if space is not None:
        if space.tables:
            tables &= set(space.tables)
        if space.deny_objects:
            tables -= set(space.deny_objects)
        if tables and (space.columns or space.deny_columns):
            scope_ctx = EngineContext(
                allow_objects=frozenset(space.tables) if space.tables else frozenset(),
                deny_objects=space.deny_objects,
                allow_columns=space.columns,
                deny_columns=space.deny_columns,
            )
            if not assert_consumer_intent_in_scope(intent, scope_ctx, schema, frozenset(tables)):
                return FederatedPlan(steps=(), ineligible_reason="intent references columns outside the active space")
    if not tables:
        return FederatedPlan(steps=(), ineligible_reason="no tables referenced")
    capability_reason = _federation_ir_capability_reason(intent, schema.database_feature_capability, schema=schema)
    if capability_reason:
        return FederatedPlan(steps=(), ineligible_reason=capability_reason)
    capability_reason = _federation_unsupported_operator_reason(intent, manifest, dialects_by_source=dialects_by_source)
    if capability_reason:
        return FederatedPlan(steps=(), ineligible_reason=capability_reason)
    source_by_table = dict(table_set.source_by_table)
    sources: set[str] = set()
    for table in tables:
        sources.update(_planning_sources_for_table(table, manifest, mappings, source_by_table, schema))
    multi_source = len(sources) > 1
    if multi_source:
        raw_sql_reason = _unattributable_raw_sql_reason(intent)
        if raw_sql_reason:
            return FederatedPlan(steps=(), ineligible_reason=raw_sql_reason)
        if _intent_lacks_column_member_coverage(intent, schema):
            return FederatedPlan(steps=(), ineligible_reason="projection columns are not held by any single member")
        clause_reason = _federation_clause_ineligible_reason(intent, manifest, mappings, source_by_table, schema=schema)
        if clause_reason:
            return FederatedPlan(steps=(), ineligible_reason=clause_reason)
        spanning_cte_reason = _spanning_cte_decomposition_ineligible_reason(intent, source_by_table)
        if spanning_cte_reason:
            return FederatedPlan(steps=(), ineligible_reason=spanning_cte_reason)
        agg_reason = _cross_source_aggregate_ineligible_reason(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        if agg_reason:
            return FederatedPlan(steps=(), ineligible_reason=agg_reason)
    steps: list[SourceStep] = []
    union_specs = _union_specs_for_intent(tables, mappings, source_by_table)
    combine: tuple[JoinSpec, ...] | None = _join_specs_for_sources(
        manifest, mappings, frozenset(sources), schema=schema, scope_tables=frozenset(tables)
    )
    if len(sources) > 1 and combine is None and not union_specs:
        return FederatedPlan(
            steps=(), ineligible_reason="cross-source join path is not declared for referenced sources"
        )
    global _WINDOW_FINALITY_CTX
    _WINDOW_FINALITY_CTX = _WindowFinalityContext(
        manifest=manifest, schema=schema, combine=combine, source_by_table=source_by_table
    )
    try:
        for source_id in sorted(sources):
            member_schema = _member_schema_for_sub_intent_repair(
                source_id, schema, manifest=manifest, member_graphs=member_graphs
            )
            sub = _build_source_sub_intent(
                intent,
                source_id,
                tables,
                source_by_table,
                mappings,
                schema,
                manifest,
                multi_source=multi_source,
                member_schema=member_schema,
                chosen_specs=combine,
                space=space,
            )
            if sub is not None:
                steps.append(sub)
    finally:
        _WINDOW_FINALITY_CTX = None
    if multi_source and len(steps) < len(sources):
        built = {step.source_id for step in steps}
        dropped = tuple(sorted(sources - built))
        raise FederationInvariantError(
            "federation plan dropped member(s) "
            f"{list(dropped)} that scope discovery found "
            f"(scope sources={sorted(sources)})"
        )
    residual = _residual_spec_for_intent(
        intent, source_by_table, manifest, mappings, schema=schema, scope_tables=tables, combine=combine
    )
    stages = plan_federated_stages(
        sources,
        tuple(steps),
        intent=intent,
        source_by_table=source_by_table,
        manifest=manifest,
        mappings=mappings,
        residual=residual,
        schema=schema,
        combine=combine,
    )
    grain = intent.grain or "row_level"
    if grain not in VALID_GRAINS:
        grain = "row_level"
    lifted_probe_ctes = _cross_source_probe_cte_steps(intent, source_by_table)
    plan = FederatedPlan(
        steps=tuple(steps),
        union_specs=tuple(union_specs),
        combine=combine,
        residual=residual,
        stages=stages,
        grain=grain,
        scope_sources=frozenset(sources),
        lifted_probe_ctes=lifted_probe_ctes,
    )
    if multi_source:
        validate_federated_residual_aggregate_fan_out(plan, schema, manifest)
        validate_federation_coordinator_column_types(plan, schema, manifest=manifest)
        validate_federation_scalar_grain_member_frames(plan)
    return plan


def federation_plan_is_degenerate(plan: FederatedPlan) -> bool:
    """Return True when *plan* is a single-member graph with no coordinator combine work."""
    if plan.ineligible_reason or len(plan.steps) != 1:
        return False
    if plan.residual is not None:
        return False
    if effective_union_specs(plan):
        return False
    if plan.combine:
        return False
    if len(plan.scope_sources) > 1:
        return False
    return True


def resolve_federated_member_schema(
    source_id: str,
    composite_schema: SchemaGraph,
    *,
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
) -> SchemaGraph:
    """Return the member schema slice, preferring loaded member graphs when present."""
    if member_graphs is not None and source_id in member_graphs:
        return member_graphs[source_id]
    member_graph = member_graphs.get(source_id) if member_graphs is not None else None
    return member_schema_slice(composite_schema, source_id, manifest=manifest, member_graph=member_graph)


def plan_federated_stages(
    sources: set[str],
    steps: tuple[SourceStep, ...],
    *,
    intent: RuntimeIntent | None = None,
    source_by_table: Mapping[str, str] | None = None,
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    residual: ResidualSpec | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> tuple[FederatedStage, ...]:
    """Build a staged execution graph with member stages, optional spanning CTE, and coordinator."""
    if len(sources) <= 1:
        return ()
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    member_deps = (
        _member_stage_dependencies(intent, source_by_table, manifest, mappings, sources) if manifest is not None else {}
    )
    reducing_by_source = (
        _collect_member_reducing_edges(manifest, mappings, sources, intent, source_by_table, schema=schema)
        if manifest is not None
        else {}
    )
    member_stages = [
        FederatedStage(
            stage_id=f"member_{source_id}",
            kind="member",
            source_ids=(source_id,),
            depends_on=member_deps.get(source_id, ()),
            reducing_edges=reducing_by_source.get(source_id, ()),
        )
        for source_id in sorted(sources)
    ]
    stage_list: list[FederatedStage] = list(member_stages)
    coordinator_depends = tuple(stage.stage_id for stage in member_stages)
    spanning: tuple[str, ...] = ()
    if intent is not None and source_by_table is not None:
        spanning = _spanning_cte_names(intent.cte_steps or (), source_by_table)
        if spanning:
            cte_sources = _spanning_cte_source_ids(intent.cte_steps or (), spanning, source_by_table)
            if not cte_sources:
                cte_sources = tuple(sorted(sources))
            cte_source_set = set(cte_sources)
            cte_depends = tuple(f"member_{source_id}" for source_id in cte_sources)
            cte_stage = FederatedStage(
                stage_id="coordinator_cte",
                kind="cte",
                source_ids=cte_sources,
                depends_on=cte_depends,
                spanning_cte_names=spanning,
            )
            stage_list.append(cte_stage)
            remaining = tuple(f"member_{source_id}" for source_id in sorted(sources) if source_id not in cte_source_set)
            coordinator_depends = (cte_stage.stage_id,) + remaining
    promotes_windows = False
    if intent is not None and source_by_table is not None:
        promotes_windows = _coordinator_promotes_spanning_windows(intent, source_by_table, manifest=manifest)
    if promotes_windows:
        coordinator_depends = tuple(stage.stage_id for stage in stage_list)
    grain = (intent.grain or "row_level") if intent is not None else "row_level"
    coordinator_id = "coordinator_scalar" if residual is not None and grain == "scalar" else "coordinator"
    coordinator = FederatedStage(
        stage_id=coordinator_id, kind="coordinator", source_ids=tuple(sorted(sources)), depends_on=coordinator_depends
    )
    stage_list.append(coordinator)
    return tuple(stage_list)


def _spanning_cte_names(cte_steps: Sequence[RuntimeCteStep], source_by_table: Mapping[str, str]) -> tuple[str, ...]:
    """Return CTE names whose referenced base tables span more than one member."""
    if not cte_steps:
        return ()
    owners = _assign_cte_sources(cte_steps, source_by_table)
    spanning: list[str] = []
    for cte in cte_steps:
        name = cte.cte_name
        if name and name not in owners:
            spanning.append(name)
    return tuple(spanning)


def _spanning_cte_source_ids(
    cte_steps: Sequence[RuntimeCteStep], spanning_names: Sequence[str], source_by_table: Mapping[str, str]
) -> tuple[str, ...]:
    """Return member source ids that feed any spanning CTE in *spanning_names*."""
    if not cte_steps or not spanning_names:
        return ()
    spanning_set = set(spanning_names)
    owners = _assign_cte_sources(cte_steps, source_by_table)
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}
    cte_names_lower = {name.lower() for name in cte_names}
    collected: set[str] = set()
    for cte in cte_steps:
        name = cte.cte_name
        if not name or name not in spanning_set:
            continue
        refs = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            _predicate_where_leaves(cte.where),
            _predicate_having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            include_unreferenced_registries=False,
        )
        base_tables = {table for table in refs if table not in cte_names and table.lower() not in cte_names_lower}
        prior_ctes = {table for table in refs if table in cte_names or table.lower() in cte_names_lower}
        for table in base_tables:
            source_id = source_by_table.get(table, "")
            if source_id:
                collected.add(source_id)
        for prior in prior_ctes:
            canonical = next((candidate for candidate in cte_names if candidate.lower() == prior.lower()), prior)
            owner = owners.get(canonical)
            if owner:
                collected.add(owner)
            elif canonical in spanning_set:
                collected.update(
                    source_by_table.get(table, "") for table in (cte.tables or []) if source_by_table.get(table, "")
                )
        for table in cte.tables or []:
            source_id = source_by_table.get(table, "")
            if source_id:
                collected.add(source_id)
    return tuple(sorted(sid for sid in collected if sid))


def _coordinator_promotes_spanning_windows(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when any window must run at the coordinator after combine."""
    for entry in intent.window_registry or []:
        if _window_requires_coordinator(
            entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
        ):
            return True
    return False


def derive_execution_order_from_stages(plan: FederatedPlan) -> tuple[str, ...]:
    """
    Return member source ids in topological ``depends_on`` order.

    Raises:

        FederationInvariantError: When member-stage ``depends_on`` edges form a cycle.
    """
    member_stages = [stage for stage in plan.stages if stage.kind == "member"]
    if not member_stages:
        return tuple(step.source_id for step in sorted(plan.steps, key=lambda s: s.source_id))
    stage_by_id = {stage.stage_id: stage for stage in member_stages}
    ordered: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise FederationInvariantError(f"federated stage dependency cycle involving {stage_id!r}")
        stage = stage_by_id.get(stage_id)
        if stage is None:
            return
        visiting.add(stage_id)
        for dep in stage.depends_on:
            if dep in stage_by_id:
                visit(dep)
        visiting.remove(stage_id)
        visited.add(stage_id)
        if stage.source_ids:
            ordered.append(stage.source_ids[0])

    for stage in sorted(member_stages, key=lambda item: item.stage_id):
        visit(stage.stage_id)
    return tuple(ordered)


def _collect_member_reducing_edges(
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: set[str],
    intent: RuntimeIntent | None,
    source_by_table: Mapping[str, str] | None,
    *,
    schema: SchemaGraph | None = None,
) -> dict[str, tuple[FederationReducingEdge, ...]]:
    """Collect semi-join and filter-as-key reducing edges for each member stage."""
    edges: dict[str, list[FederationReducingEdge]] = {source_id: [] for source_id in sources}
    for join in manifest.cross_source_joins:
        left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
        right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if right_src in sources and source_semijoin_enabled(manifest, right_src):
            if reducing_edge_allowed_for_target(right_src, join, manifest, schema=schema):
                edges[right_src].append(
                    FederationReducingEdge(
                        driving_source_id=left_src,
                        target_source_id=right_src,
                        driving_key=left_col,
                        target_key=right_col,
                        edge_kind="semijoin",
                    )
                )
        if left_src in sources and source_semijoin_enabled(manifest, left_src):
            if reducing_edge_allowed_for_target(left_src, join, manifest, schema=schema):
                edges[left_src].append(
                    FederationReducingEdge(
                        driving_source_id=right_src,
                        target_source_id=left_src,
                        driving_key=right_col,
                        target_key=left_col,
                        edge_kind="semijoin",
                    )
                )
    if intent is not None and source_by_table is not None:
        for fp in _cross_source_where(intent, manifest, mappings, source_by_table):
            if not _cross_where_relates_to_join(fp, manifest):
                continue
            filter_cols = _param_qualified_columns(fp)
            for join in manifest.cross_source_joins:
                left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
                right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
                left_src = manifest.table_namespace.get(left_tbl, "")
                right_src = manifest.table_namespace.get(right_tbl, "")
                if not left_src or not right_src or left_src == right_src:
                    continue
                filter_tables = {split_qualified_column(col, manifest=manifest)[0] for col in filter_cols}
                filter_sources = _sources_for_refs(filter_tables, manifest, mappings, source_by_table or {})
                if right_src in filter_sources and left_src in sources and source_semijoin_enabled(manifest, right_src):
                    if reducing_edge_allowed_for_target(right_src, join, manifest, schema=schema):
                        edges[right_src].append(
                            FederationReducingEdge(
                                driving_source_id=left_src,
                                target_source_id=right_src,
                                driving_key=left_col,
                                target_key=right_col,
                                edge_kind="filter_keys",
                            )
                        )
                if left_src in filter_sources and right_src in sources and source_semijoin_enabled(manifest, left_src):
                    if reducing_edge_allowed_for_target(left_src, join, manifest, schema=schema):
                        edges[left_src].append(
                            FederationReducingEdge(
                                driving_source_id=right_src,
                                target_source_id=left_src,
                                driving_key=right_col,
                                target_key=left_col,
                                edge_kind="filter_keys",
                            )
                        )
    if intent is not None and source_by_table is not None:
        owners = _assign_cte_sources(intent.cte_steps or (), source_by_table)
        for cte in intent.cte_steps or []:
            if coerce_cte_emission(getattr(cte, "emission", "join_table")) != "semi_join":
                continue
            owner = owners.get(cte.cte_name or "")
            if not owner:
                continue
            for key in _cte_probe_join_keys(cte):
                if "." not in key:
                    continue
                left_tbl, left_col = split_qualified_column(
                    key, manifest=manifest, schema=None, source_by_table=source_by_table
                )
                left_src = source_by_table.get(left_tbl, "")
                for join in manifest.cross_source_joins:
                    j_left_tbl, j_left_col = split_qualified_column(join.left, manifest=manifest)
                    j_right_tbl, j_right_col = split_qualified_column(join.right, manifest=manifest)
                    j_left_src = manifest.table_namespace.get(j_left_tbl, "")
                    j_right_src = manifest.table_namespace.get(j_right_tbl, "")
                    if owner == j_right_src and left_src == j_left_src and j_left_col == left_col:
                        if j_right_src in sources and source_semijoin_enabled(manifest, j_right_src):
                            if reducing_edge_allowed_for_target(j_right_src, join, manifest, schema=schema):
                                edges[j_right_src].append(
                                    FederationReducingEdge(
                                        driving_source_id=j_left_src,
                                        target_source_id=j_right_src,
                                        driving_key=j_left_col,
                                        target_key=j_right_col,
                                        edge_kind="semijoin",
                                    )
                                )
                    if owner == j_left_src and left_src == j_right_src and j_right_col == left_col:
                        if j_left_src in sources and source_semijoin_enabled(manifest, j_left_src):
                            if reducing_edge_allowed_for_target(j_left_src, join, manifest, schema=schema):
                                edges[j_left_src].append(
                                    FederationReducingEdge(
                                        driving_source_id=j_right_src,
                                        target_source_id=j_left_src,
                                        driving_key=j_right_col,
                                        target_key=j_left_col,
                                        edge_kind="semijoin",
                                    )
                                )
    return {source_id: tuple(edge_list) for source_id, edge_list in edges.items() if edge_list}


def derive_federation_stages_in_order(plan: FederatedPlan) -> tuple[FederatedStage, ...]:
    """
    Return all federated stages in topological execution order.

    Raises:

        FederationInvariantError: When ``depends_on`` edges form a cycle.
    """
    if not plan.stages:
        return ()
    stage_by_id = {stage.stage_id: stage for stage in plan.stages}
    ordered: list[FederatedStage] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise FederationInvariantError(f"federated stage dependency cycle involving {stage_id!r}")
        stage = stage_by_id.get(stage_id)
        if stage is None:
            return
        visiting.add(stage_id)
        for dep in stage.depends_on:
            visit(dep)
        visiting.remove(stage_id)
        visited.add(stage_id)
        ordered.append(stage)

    for stage in sorted(
        plan.stages, key=lambda item: (0 if item.kind == "member" else 1 if item.kind == "cte" else 2, item.stage_id)
    ):
        visit(stage.stage_id)
    return tuple(ordered)


def member_stage_for_source(plan: FederatedPlan, source_id: str) -> FederatedStage | None:
    """Return the single-source member stage for *source_id*, if present. Used by the pipeline when attributing work to one federation member. Args: plan: Federated execution plan whose ``stages`` may include member, CTE, and coordinator stages. source_id: Member registration key to match against a stage whose ``source_ids`` is exactly ``(source_id)``. Returns: The matching ``kind=="member"`` :class:`~aetherdialect._contracts_core.FederatedStage`, or ``None`` when no such stage exists."""
    for stage in plan.stages:
        if stage.kind == "member" and stage.source_ids == (source_id,):
            return stage
    return None


@dataclass(frozen=True, slots=True)
class _CombineJoinTree:
    source_id: str
    children: tuple[tuple[JoinSpec, _CombineJoinTree], ...] = ()


def _combine_join_hub_source(join_specs: tuple[JoinSpec, ...], sources: set[str]) -> str:
    degree: dict[str, int] = {source_id: 0 for source_id in sources}
    for spec in join_specs:
        degree[spec.left_source] = degree.get(spec.left_source, 0) + 1
        degree[spec.right_source] = degree.get(spec.right_source, 0) + 1
    return max(sources, key=lambda source_id: (degree.get(source_id, 0), source_id))


def _build_combine_join_tree(join_specs: tuple[JoinSpec, ...], sources: set[str]) -> _CombineJoinTree:
    """Build a join tree from declared edges; refuse spanned sources without connectivity."""
    if not join_specs:
        if len(sources) == 1:
            return _CombineJoinTree(source_id=next(iter(sources)))
        raise FederationRuntimeError("federation combine requires join edges when multiple sources are spanned")
    adjacency: dict[str, list[JoinSpec]] = defaultdict(list)
    for spec in join_specs:
        adjacency[spec.left_source].append(spec)
        adjacency[spec.right_source].append(spec)
    root = _combine_join_hub_source(join_specs, sources)
    visited: set[str] = {root}

    def walk(source_id: str) -> _CombineJoinTree:
        children: list[tuple[JoinSpec, _CombineJoinTree]] = []
        for spec in sorted(adjacency.get(source_id, []), key=lambda item: (item.right_source, item.left_source)):
            other = spec.right_source if spec.left_source == source_id else spec.left_source
            if other in visited:
                continue
            visited.add(other)
            children.append((spec, walk(other)))
        return _CombineJoinTree(source_id=source_id, children=tuple(children))

    tree = walk(root)
    orphan = sources - visited
    if orphan:
        raise FederationRuntimeError(
            f"federation combine missing declared edges for sources: {', '.join(sorted(orphan))}"
        )
    return tree


def _render_combine_tree_sql(
    tree: _CombineJoinTree,
    step_ids: Mapping[str, str],
    plan: FederatedPlan,
    *,
    schema: SchemaGraph | None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
    explicit_cols: list[str] | None,
    alias_counter: list[int] | None = None,
) -> str:
    """Render combine SQL from a join tree with explicit projection on every hop."""
    counter = alias_counter if alias_counter is not None else [0]
    reg = step_ids.get(tree.source_id, "")
    if not reg:
        raise FederationRuntimeError(f"federation join missing frame for source {tree.source_id!r}")
    if not tree.children:
        select_kw = _render_combine_select_keyword(explicit_cols)
        return f"SELECT {select_kw} FROM {reg} AS s{tree.source_id}"
    sql = ""
    for idx, (spec, child) in enumerate(tree.children):
        left_reg = reg if idx == 0 else f"({sql})"
        left_alias = "l" if idx == 0 else "prev"
        right_reg = step_ids.get(child.source_id, "")
        if not right_reg:
            raise FederationRuntimeError(f"federation join missing frame for source {child.source_id!r}")
        join_kind = validate_federation_cross_source_join_kind(spec.kind).upper()
        left_table = (
            resolve_source_column_table(
                schema,
                spec.left_source,
                spec.left_key,
                manifest=manifest,
                source_by_table=source_by_table,
                declared_table=declared_table_for_source_column(
                    plan,
                    spec.left_source,
                    spec.left_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                ),
            )
            if schema
            else None
        )
        right_table = (
            resolve_source_column_table(
                schema,
                spec.right_source,
                spec.right_key,
                manifest=manifest,
                source_by_table=source_by_table,
                declared_table=declared_table_for_source_column(
                    plan,
                    spec.right_source,
                    spec.right_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                ),
            )
            if schema
            else None
        )
        left_key_source = spec.left_source if spec.left_source == tree.source_id else child.source_id
        right_key_source = spec.right_source if spec.right_source == child.source_id else tree.source_id
        left_expr = _coordinator_join_key_expr(
            left_alias,
            spec.left_key if left_key_source == tree.source_id else spec.right_key,
            schema=schema,
            table_name=left_table,
        )
        right_expr = _coordinator_join_key_expr(
            "r",
            spec.right_key if right_key_source == child.source_id else spec.left_key,
            schema=schema,
            table_name=right_table,
        )
        left_cols = _source_column_names_for_step(plan, tree.source_id if idx == 0 else child.source_id)
        right_cols = _source_column_names_for_step(plan, child.source_id)
        select_kw = _render_join_select_keyword(
            explicit_cols, left_alias=left_alias, right_alias="r", left_cols=left_cols, right_cols=right_cols
        )
        if idx == 0:
            sql = (
                f"SELECT {select_kw} FROM {left_reg} AS {left_alias} {join_kind} JOIN {right_reg} AS r "
                f"ON {left_expr} = {right_expr}"
            )
        else:
            sql = (
                f"SELECT {select_kw} FROM ({sql}) AS {left_alias} {join_kind} JOIN {right_reg} AS r "
                f"ON {left_expr} = {right_expr}"
            )
        if child.children:
            counter[0] += 1
            nested = _render_combine_tree_sql(
                child,
                step_ids,
                plan,
                schema=schema,
                manifest=manifest,
                source_by_table=source_by_table,
                explicit_cols=explicit_cols,
                alias_counter=counter,
            )
            sql = (
                f"SELECT {_render_combine_select_keyword(explicit_cols)} FROM ({sql}) AS {left_alias} "
                f"{join_kind} JOIN ({nested}) AS r ON {left_expr} = {right_expr}"
            )
    return sql


def _render_combine_sql_for_sources(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    sources: set[str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    """Render join-tree combine SQL for a subset of member sources."""
    if not sources:
        return ""
    scoped_steps = tuple(step for step in plan.steps if step.source_id in sources)
    if not scoped_steps and len(sources) == 1:
        only_source = next(iter(sources))
        reg = step_ids.get(only_source, "")
        if not reg:
            return ""
        scoped_plan = replace(
            plan,
            steps=tuple(step for step in plan.steps if step.source_id == only_source),
            combine=None,
            union_specs=(),
            residual=None,
            stages=(),
        )
        explicit_cols = _combine_select_column_names(scoped_plan)
        select_kw = _render_combine_select_keyword(explicit_cols)
        return f"SELECT {select_kw} FROM {reg}"
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    scoped_joins: tuple[JoinSpec, ...] | None = None
    if join_specs:
        scoped_joins = tuple(
            spec for spec in join_specs if spec.left_source in sources and spec.right_source in sources
        )
    scoped_plan = replace(
        plan,
        steps=scoped_steps,
        combine=scoped_joins if scoped_joins else None,
        union_specs=(),
        residual=None,
        stages=(),
    )
    scoped_ids = {source_id: reg for source_id, reg in step_ids.items() if source_id in sources}
    return _render_federation_combine_sql(
        scoped_plan, scoped_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
    )


def _render_coordinator_spanning_cte_sql(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    """Compose spanning CTE bodies through the declared join graph for contributing members."""
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is None or not cte_stage.spanning_cte_names:
        return ""
    cte_sources = set(cte_stage.source_ids)
    if not cte_sources:
        cte_sources = {step.source_id for step in plan.steps}
    body = _render_combine_sql_for_sources(
        plan, step_ids, cte_sources, schema=schema, manifest=manifest, source_by_table=source_by_table
    )
    if not body:
        return ""
    cte_defs = [f"{name} AS ({body})" for name in cte_stage.spanning_cte_names]
    first = cte_stage.spanning_cte_names[0]
    all_sources = {step.source_id for step in plan.steps}
    remaining = all_sources - cte_sources
    cte_projected: list[str] = []
    for step in plan.steps:
        if step.source_id not in cte_sources:
            continue
        cte_projected.extend(step.projected_keys)
    cte_select_cols = [name for name in dict.fromkeys(_unqualified_column_name(key) for key in cte_projected) if name]
    cte_select_kw = _render_combine_select_keyword(cte_select_cols or None)
    if not remaining:
        return f"WITH {', '.join(cte_defs)} SELECT {cte_select_kw} FROM {first}"
    remapped_joins: list[JoinSpec] = []
    join_specs = plan.combine if isinstance(plan.combine, tuple) else ()
    for spec in join_specs or ():
        left_in = spec.left_source in cte_sources
        right_in = spec.right_source in cte_sources
        if left_in and right_in:
            continue
        if not left_in and not right_in:
            if spec.left_source in remaining and spec.right_source in remaining:
                remapped_joins.append(spec)
            continue
        remapped_joins.append(
            JoinSpec(
                left_source=first if left_in else spec.left_source,
                right_source=first if right_in else spec.right_source,
                left_key=spec.left_key,
                right_key=spec.right_key,
                logical_key=spec.logical_key,
                kind=spec.kind,
            )
        )
    remaining_steps = tuple(step for step in plan.steps if step.source_id in remaining)
    cte_step = SourceStep(
        source_id=first,
        sub_intent=RuntimeIntent(
            tables=[], grain="row_level", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
        ),
        projected_keys=tuple(dict.fromkeys(cte_projected)),
    )
    outer_plan = replace(
        plan,
        steps=remaining_steps + (cte_step,),
        combine=tuple(remapped_joins) if remapped_joins else None,
        union_specs=(),
        residual=None,
        stages=(),
    )
    outer_ids = {step.source_id: step_ids[step.source_id] for step in remaining_steps if step.source_id in step_ids}
    outer_ids[first] = first
    outer_sql = _render_federation_combine_sql(
        outer_plan, outer_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
    )
    if not outer_sql:
        return f"WITH {', '.join(cte_defs)} SELECT {cte_select_kw} FROM {first}"
    return f"WITH {', '.join(cte_defs)} {outer_sql}"


def _semijoin_reduction_stage_dependencies(
    manifest: FederationManifest, sources: set[str]
) -> dict[str, tuple[str, ...]]:
    """Return member-stage depends_on edges for semi-join reduction across sources."""
    deps: dict[str, set[str]] = {source_id: set() for source_id in sources}
    for join in manifest.cross_source_joins:
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if right_src in sources and source_semijoin_enabled(manifest, right_src):
            deps.setdefault(right_src, set()).add(f"member_{left_src}")
    return {source_id: tuple(sorted(stage_ids)) for source_id, stage_ids in deps.items() if stage_ids}


def _where_pushdown_stage_dependencies(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: set[str],
) -> dict[str, tuple[str, ...]]:
    """Return member-stage depends_on edges for join-covered cross- source filter pushdown."""
    cross_filters = _cross_source_where(intent, manifest, mappings, source_by_table)
    if not cross_filters:
        return {}
    deps: dict[str, set[str]] = {source_id: set() for source_id in sources}
    for fp in cross_filters:
        if not _cross_where_relates_to_join(fp, manifest):
            continue
        filter_cols = _param_qualified_columns(fp)
        filter_tables = {split_qualified_column(col, manifest=manifest)[0] for col in filter_cols}
        filter_sources = _sources_for_refs(filter_tables, manifest, mappings, source_by_table)
        for join in manifest.cross_source_joins:
            left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
            right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
            left_src = source_by_table.get(left_tbl, manifest.table_namespace.get(left_tbl, ""))
            right_src = source_by_table.get(right_tbl, manifest.table_namespace.get(right_tbl, ""))
            if not left_src or not right_src or left_src == right_src:
                continue
            if right_src in filter_sources and left_src in sources:
                deps.setdefault(right_src, set()).add(f"member_{left_src}")
            if left_src in filter_sources and right_src in sources:
                deps.setdefault(left_src, set()).add(f"member_{right_src}")
    return {source_id: tuple(sorted(stage_ids)) for source_id, stage_ids in deps.items() if stage_ids}


def _member_stage_dependencies(
    intent: RuntimeIntent | None,
    source_by_table: Mapping[str, str] | None,
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: set[str],
) -> dict[str, tuple[str, ...]]:
    """Merge semi-join reduction and filter-pushdown stage dependencies."""
    combined: dict[str, set[str]] = {source_id: set() for source_id in sources}
    for dep_map in (
        _semijoin_reduction_stage_dependencies(manifest, sources),
        _where_pushdown_stage_dependencies(intent, source_by_table, manifest, mappings, sources)
        if intent is not None and source_by_table is not None
        else {},
    ):
        for source_id, stage_ids in dep_map.items():
            combined.setdefault(source_id, set()).update(stage_ids)
    return {source_id: tuple(sorted(stage_ids)) for source_id, stage_ids in combined.items() if stage_ids}


_federated_stages_for_plan = plan_federated_stages


@dataclass(frozen=True, slots=True)
class _WindowFinalityContext:
    """Join and schema facts used to decide whether a member window sees final rows."""

    manifest: FederationManifest
    schema: SchemaGraph
    combine: tuple[JoinSpec, ...] | None
    source_by_table: Mapping[str, str]


_WINDOW_FINALITY_CTX: _WindowFinalityContext | None = None


def effective_union_specs(plan: FederatedPlan) -> tuple[UnionSpec, ...]:
    """Return union combine specs from ``union_specs`` or the legacy ``combine`` field."""
    if plan.union_specs:
        return plan.union_specs
    if isinstance(plan.combine, UnionSpec):
        return (plan.combine,)
    return ()


def _render_federation_union_cte_defs(
    union_specs: tuple[UnionSpec, ...], step_ids: Mapping[str, str], *, explicit_cols: list[str] | None
) -> tuple[str, ...]:
    """Return ``WITH`` CTE definitions materializing each union spec before join combine."""
    cte_defs: list[str] = []
    select_kw = _render_combine_select_keyword(explicit_cols)
    for idx, spec in enumerate(union_specs):
        cte_name = f"fed_u{idx}"
        union_rel = _render_union_relation_sql(spec, step_ids, explicit_cols=explicit_cols)
        cte_defs.append(f"{cte_name} AS (SELECT {select_kw} FROM {union_rel} AS _u)")
    return tuple(cte_defs)


def _render_union_relation_sql(
    union_spec: UnionSpec, step_ids: Mapping[str, str], *, explicit_cols: list[str] | None = None
) -> str:
    members = [step_ids[sid] for sid in union_spec.member_source_ids if sid in step_ids]
    if not members:
        raise FederationRuntimeError(f"federation union {union_spec.logical_table!r} missing member frames")
    if union_spec.semantics == "replica" or len(members) == 1:
        return members[0]
    select_kw = _render_combine_select_keyword(explicit_cols)
    return "(" + " UNION ALL ".join(f"SELECT {select_kw} FROM {m}" for m in members) + ")"


def source_by_table_from_schema(schema: SchemaGraph | None) -> dict[str, str]:
    """Build table-to-member map from a composite schema graph."""
    if schema is None:
        return {}
    return {
        table_name: str(table.source_id or "")
        for table_name, table in schema.tables.items()
        if str(table.source_id or "").strip()
    }


def _apply_coordinator_probe_joins(
    base_sql: str, probe_ctes: Sequence[RuntimeCteStep], step_ids: Mapping[str, str], source_by_table: Mapping[str, str]
) -> str:
    """Lift cross-source semi/anti probes onto materialised member frames at the coordinator."""
    if not probe_ctes or not base_sql.strip():
        return base_sql
    owners = _assign_cte_sources(probe_ctes, source_by_table)
    sql = base_sql
    for cte in probe_ctes:
        owner = owners.get(cte.cte_name or "")
        if not owner or owner not in step_ids:
            continue
        probe_rel = step_ids[owner]
        keys = [_unqualified_column_name(key) for key in _cte_probe_join_keys(cte)]
        keys = [key for key in keys if key]
        if not keys:
            continue
        alias = (cte.cte_name or f"probe_{owner}").replace(".", "_")
        distinct_keys = ", ".join(_quote_ident(key) for key in keys)
        on_parts = [f"drv.{_quote_ident(key)} = {alias}.{_quote_ident(key)}" for key in keys]
        emission = coerce_cte_emission(getattr(cte, "emission", "join_table"))
        if emission == "semi_join":
            probe_subquery = f"(SELECT DISTINCT {distinct_keys} FROM {probe_rel})"
            sql = f"SELECT drv.* FROM ({sql}) AS drv INNER JOIN {probe_subquery} AS {alias} ON {' AND '.join(on_parts)}"
        elif emission == "anti_join":
            presence = anti_join_presence_column(alias)
            anti_subquery = f"(SELECT DISTINCT {distinct_keys}, 1 AS {_quote_ident(presence)} FROM {probe_rel})"
            sql = (
                f"SELECT drv.* FROM ({sql}) AS drv "
                f"LEFT JOIN {anti_subquery} AS {alias} ON {' AND '.join(on_parts)} "
                f"WHERE {alias}.{_quote_ident(presence)} IS NULL"
            )
    return sql


def render_federation_glue(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    param_values: Mapping[str, Any] | None = None,
) -> str:
    """Render deterministic DuckDB SQL glue for a federated plan."""
    if plan.ineligible_reason:
        return ""
    source_by_table = source_by_table_from_schema(schema)
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is not None and cte_stage.spanning_cte_names:
        base_sql = _render_coordinator_spanning_cte_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    else:
        base_sql = _render_federation_combine_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    if not base_sql:
        return ""
    base_sql = _apply_coordinator_probe_joins(
        base_sql, plan.lifted_probe_ctes, step_ids, source_by_table_from_schema(schema)
    )
    return render_federation_residual_sql(base_sql, plan.residual, param_values=param_values)


def _render_federation_combine_only_glue(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
) -> str:
    """Render coordinator combine SQL without residual clauses."""
    source_by_table = source_by_table_from_schema(schema)
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is not None and cte_stage.spanning_cte_names:
        base_sql = _render_coordinator_spanning_cte_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    else:
        base_sql = _render_federation_combine_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    if not base_sql:
        return ""
    return _apply_coordinator_probe_joins(
        base_sql, plan.lifted_probe_ctes, step_ids, source_by_table_from_schema(schema)
    )


def _join_signature_from_combine_plan(
    plan: FederatedPlan,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
) -> list[str]:
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return []
    source_by_table = source_by_table_from_schema(schema)
    segments: list[str] = []
    for spec in combine:
        left_table = declared_table_for_source_column(
            plan,
            spec.left_source,
            spec.left_key,
            manifest=manifest,
            schema=schema,
            source_by_table=source_by_table,
        )
        right_table = declared_table_for_source_column(
            plan,
            spec.right_source,
            spec.right_key,
            manifest=manifest,
            schema=schema,
            source_by_table=source_by_table,
        )
        if not left_table or not right_table:
            continue
        segments.append(f"{left_table}.{spec.left_key}->{right_table}.{spec.right_key}")
    return segments


def _residual_tables_for_plan(plan: FederatedPlan) -> list[str]:
    tables: list[str] = []
    for step in plan.steps:
        for key in step.projected_keys:
            if "." not in key:
                continue
            table, _ = key.split(".", 1)
            if table not in tables:
                tables.append(table)
    return tables


def validate_federated_residual_aggregate_fan_out(
    plan: FederatedPlan,
    schema: SchemaGraph,
    manifest: FederationManifest | None = None,
) -> None:
    """Refuse coordinator residual aggregates that would see join- multiplied rows."""
    residual = plan.residual
    if residual is None or not residual.select_cols:
        return
    if not any(_looks_aggregated(sc) for sc in residual.select_cols):
        return
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return
    source_by_table = source_by_table_from_schema(schema)
    agg_tables = {
        table for sc in residual.select_cols if _looks_aggregated(sc) for table in _tables_referenced_by_select_col(sc)
    }
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        for agg_table in sorted(agg_tables):
            agg_source = source_by_table.get(agg_table, "")
            if not agg_source:
                continue
            if kind == "inner":
                targets = (
                    (spec.left_source, spec.right_source, spec.right_key),
                    (spec.right_source, spec.left_source, spec.left_key),
                )
            elif kind == "left":
                targets = ((spec.left_source, spec.right_source, spec.right_key),)
            elif kind == "right":
                targets = ((spec.right_source, spec.left_source, spec.left_key),)
            else:
                continue
            for preserved_source, other_source, other_key in targets:
                if agg_source != preserved_source:
                    continue
                other_table = declared_table_for_source_column(
                    plan,
                    other_source,
                    other_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                )
                unique = _source_join_key_is_unique(
                    schema,
                    other_source,
                    f"{other_table}.{other_key}" if other_table else other_key,
                    manifest=manifest,
                )
                if unique is not False:
                    continue
                raise FederationJoinFanOutError(
                    f"federation coordinator residual: aggregate over {agg_table!r} would see rows duplicated by "
                    f"join edge {preserved_source!r}->{other_source!r} on key {other_key!r}",
                    source_id=other_source,
                    phase="coordinator",
                )
    signature = _join_signature_from_combine_plan(plan, manifest=manifest, schema=schema)
    if not signature:
        return
    tables = _residual_tables_for_plan(plan)
    if not tables:
        return
    intent = RuntimeIntent(
        tables=tables,
        grain=plan.grain or "row_level",
        select_cols=list(residual.select_cols),
        group_by_cols=list(residual.group_by_cols),
        order_by_cols=list(residual.order_by_cols),
        where=residual.where,
        having=residual.having,
        distinct_on=list(residual.distinct_on),
        distinct_select_index=residual.distinct_select_index,
        limit=residual.limit,
        limit_param_key=residual.limit_param_key,
        window_registry=list(residual.window_registry),
        case_registry=list(residual.case_registry),
        chosen_join_path_signature=signature,
    )
    validation_execute = importlib.import_module("aetherdialect._validation_execute")
    issues = validation_execute.validate_aggregate_join_fan_out(
        intent,
        schema,
        "federation coordinator residual",
        join_signature=signature,
        from_anchor=tables[0],
    )
    errors = [issue for issue in issues if getattr(issue, "severity", "") == "error"]
    if not errors:
        return
    source_id = plan.steps[0].source_id if plan.steps else ""
    raise FederationJoinFanOutError(errors[0].message, source_id=source_id, phase="coordinator")


def _residual_referenced_param_keys(residual: ResidualSpec | None) -> frozenset[str]:
    if residual is None:
        return frozenset()
    keys: set[str] = set()
    for fp in _predicate_where_leaves(residual.where):
        if fp.param_key:
            keys.add(fp.param_key)
        if fp.param_key_hi:
            keys.add(fp.param_key_hi)
        if fp.param_key_unit:
            keys.add(fp.param_key_unit)
    for hp in _predicate_having_leaves(residual.having):
        if hp.param_key:
            keys.add(hp.param_key)
        if hp.param_key_unit:
            keys.add(hp.param_key_unit)
    lpk = (residual.limit_param_key or "").strip()
    if lpk:
        keys.add(lpk)
    return frozenset(keys)


def coordinator_residual_bind_map(plan: FederatedPlan, parent_params: Mapping[str, Any]) -> dict[str, Any]:
    """Narrow *parent_params* to handles referenced by the coordinator residual."""
    residual_keys = _residual_referenced_param_keys(plan.residual)
    if not residual_keys:
        return {}
    return {k: v for k, v in parent_params.items() if k in residual_keys}


def _explicit_residual_order_col(col: OrderByCol) -> OrderByCol:
    """Make coordinator residual null placement explicit for DuckDB rendering."""
    if col.nulls in ("first", "last"):
        return col
    return OrderByCol(
        expr=col.expr,
        direction=col.direction,
        nulls=default_order_by_null_placement(col.direction),
    )


def render_federation_residual_sql(
    base_sql: str, residual: ResidualSpec | None, *, param_values: Mapping[str, Any] | None = None
) -> str:
    """Render coordinator residual clauses as DuckDB SQL wrapping *base_sql*. When *residual* is ``None`` or carries no clauses, returns *base_sql* unchanged. A limit-only residual without ``select_cols`` appends ``LIMIT`` directly; any other non-empty residual requires an explicit select projection and is rendered as ``SELECT ... FROM (<base_sql>) AS fed_base`` plus optional WHERE / GROUP BY / HAVING / ORDER BY / LIMIT. Args: base_sql: Inner SQL produced by the federation combine / member path. residual: Coordinator-spanning clauses, or ``None`` for a passthrough. param_values: Optional bind map used when rendering parameterised filter / having predicates. Defaults to an empty mapping. Returns: Either *base_sql* unchanged, *base_sql* with a trailing ``LIMIT``, or a full outer SELECT wrapping *base_sql* as ``fed_base``. Raises: FederationRuntimeError: *residual* has non-limit clauses (or a limit together with other clause kinds) but ``select_cols`` is empty."""
    if residual is None:
        return base_sql
    has_clauses = bool(
        residual.select_cols
        or residual.group_by_cols
        or residual.order_by_cols
        or residual.where
        or residual.having
        or residual.distinct_on
        or residual.distinct_select_index >= 0
        or residual.limit is not None
        or residual.window_registry
        or residual.case_registry
    )
    if not has_clauses:
        return base_sql
    dialect = get_dialect("duckdb")
    bind_values = dict(param_values or {})
    if not residual.select_cols:
        limit_only = (
            residual.limit is not None
            and not residual.group_by_cols
            and not residual.order_by_cols
            and not residual.where
            and not residual.having
            and not residual.distinct_on
            and residual.distinct_select_index < 0
            and not residual.window_registry
            and not residual.case_registry
        )
        if limit_only and residual.limit is not None:
            lpk = (residual.limit_param_key or "").strip()
            if lpk:
                return f"{base_sql} LIMIT :{lpk}"
            return f"{base_sql} LIMIT {int(residual.limit)}"
        raise FederationRuntimeError("federated residual requires explicit select_cols projection")
    select_exprs = [_render_residual_select_expr(sc, dialect) for sc in residual.select_cols]
    select_keyword = "SELECT DISTINCT" if residual.distinct_select_index >= 0 else "SELECT"
    parts = [f"{select_keyword} {', '.join(select_exprs)} FROM ({base_sql}) AS fed_base"]

    def _render_where_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect, is_having=False, param_values=bind_values)

    where_sql = render_predicate_group_sql(residual.where, _render_where_leaf)
    if where_sql:
        parts.append("WHERE " + where_sql)
    if residual.group_by_cols:
        gb_exprs = [render_expr_sql(g, dialect) for g in residual.group_by_cols]
        parts.append("GROUP BY " + ", ".join(gb_exprs))

    def _render_having_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect, is_having=True, param_values=bind_values)

    having_sql = render_predicate_group_sql(residual.having, _render_having_leaf)
    if having_sql:
        parts.append("HAVING " + having_sql)
    if residual.order_by_cols:
        ob_exprs = []
        for obc in residual.order_by_cols:
            explicit = _explicit_residual_order_col(obc)
            rendered = render_expr_sql(explicit.expr, dialect)
            direction = explicit.direction.upper() if explicit.direction else "ASC"
            ob_exprs.append(dialect.render_order_by_col(rendered, direction, explicit.nulls))
        parts.append("ORDER BY " + ", ".join(ob_exprs))
    elif select_exprs:
        parts.append(
            "ORDER BY "
            + ", ".join(
                dialect.render_order_by_col(expr, "ASC", default_order_by_null_placement("ASC"))
                for expr in select_exprs
            )
        )
    elif residual.group_by_cols:
        gb_exprs = [render_expr_sql(g, dialect) for g in residual.group_by_cols]
        parts.append(
            "ORDER BY "
            + ", ".join(
                dialect.render_order_by_col(expr, "ASC", default_order_by_null_placement("ASC")) for expr in gb_exprs
            )
        )
    if residual.limit is not None:
        lpk = (residual.limit_param_key or "").strip()
        if lpk:
            parts.append(f"LIMIT :{lpk}")
        else:
            parts.append(f"LIMIT {int(residual.limit)}")
    sql = " ".join(parts)
    if residual.distinct_on:
        order_cols = [_explicit_residual_order_col(col) for col in residual.order_by_cols]
        if not order_cols:
            order_cols = [
                OrderByCol(expr=expr, direction="ASC", nulls=default_order_by_null_placement("ASC"))
                for expr in residual.distinct_on
            ]
        sql = wrap_core_sql_with_distinct_on(
            sql,
            select_exprs=select_exprs,
            distinct_on=list(residual.distinct_on),
            order_by_cols=order_cols,
            limit=None,
            dialect=dialect,
        )
    return sql


def _coordinator_residual_agg_inner(inner: str) -> str:
    """Strip table qualifiers from aggregate column refs in coordinator residual SQL."""
    text = str(inner or "").strip()
    if not text or text == "*":
        return text
    match = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if match:
        return match.group(2)
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        return three.group(3)
    if "." in text and "(" not in text:
        return text.rsplit(".", 1)[-1]
    return text


def _cross_source_avg_decomposes_to_sum_count(
    func: str | None,
    *,
    inner: str,
    has_distinct: bool,
) -> bool:
    return (
        func == "avg"
        and func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS
        and bool(inner)
        and inner != "*"
        and not has_distinct
    )


def _render_residual_select_expr(sc: SelectCol, dialect: Any) -> str:
    """Render one residual projection, decomposing cross-source ``avg`` into sum and count."""
    func, has_distinct = _select_col_agg_meta(sc)
    inner = _aggregate_inner_column(sc) if func else ""
    if _cross_source_avg_decomposes_to_sum_count(func, inner=inner, has_distinct=has_distinct):
        inner = _coordinator_residual_agg_inner(inner)
        sum_sql = render_expr_sql(NormalizedExpr.from_column(f"sum({inner})"), dialect)
        count_sql = render_expr_sql(NormalizedExpr.from_column(f"count({inner})"), dialect)
        alias = (sc.output_alias or "").strip() or f"avg_{inner.replace('.', '_')}"
        return f"CAST({sum_sql} AS DOUBLE) / NULLIF({count_sql}, 0) AS {_quote_ident(alias)}"
    if func and inner:
        inner_sql = "*" if inner == "*" else _quote_ident(_coordinator_residual_agg_inner(inner))
        distinct_kw = "DISTINCT " if has_distinct else ""
        alias = (sc.output_alias or "").strip() or f"{func}_{inner.replace('.', '_')}"
        return f"{func}({distinct_kw}{inner_sql}) AS {_quote_ident(alias)}"
    return render_select_col_sql(sc, dialect)


def _residual_group_by_column_names(plan: FederatedPlan) -> tuple[str, ...]:
    residual = plan.residual
    if residual is None or not residual.group_by_cols:
        return ()
    headers: list[str] = []
    for expr in residual.group_by_cols:
        col_ref = (expr.column_ref or expr.primary_column or expr.primary_term or "").strip()
        if col_ref:
            headers.append(col_ref.rsplit(".", 1)[-1])
    return tuple(headers)


def _residual_is_aggregate_only(residual: ResidualSpec) -> bool:
    if not residual.select_cols:
        return False
    return all(_looks_aggregated(sc) for sc in residual.select_cols)


def aggregate_identity_row_for_residual(residual: ResidualSpec) -> tuple[Any, ...]:
    """Return the SQL aggregate identity row for an empty coordinator combine."""
    values: list[Any] = []
    for sc in residual.select_cols:
        if not _looks_aggregated(sc):
            values.append(None)
            continue
        func = _select_col_agg_func(sc)
        if func == "count":
            values.append(0)
        elif func == "sum":
            values.append(0)
        elif func in {"avg", "min", "max"}:
            values.append(None)
        else:
            values.append(None)
    return tuple(values)


def enforce_coordinator_result_grain(result_df: pd.DataFrame, plan: FederatedPlan) -> None:
    """Raise when the coordinator frame cardinality disagrees with the declared plan grain."""
    grain = plan.grain if plan.grain in VALID_GRAINS else "row_level"
    row_count = len(result_df)
    if grain == "scalar" and row_count != 1:
        raise FederationRuntimeError(f"federated scalar result has {row_count} rows, expected 1")
    if grain == "grouped" and row_count > 0:
        gb_cols = _residual_group_by_column_names(plan)
        if gb_cols and all(col in result_df.columns for col in gb_cols):
            if result_df.duplicated(subset=list(gb_cols), keep=False).any():
                raise FederationRuntimeError("federated grouped result has duplicate group keys")


def _render_federation_combine_sql(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    union_specs = effective_union_specs(plan)
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    explicit_cols = _combine_select_column_names(plan)
    if union_specs and not join_specs:
        if len(union_specs) == 1:
            select_kw = _render_combine_select_keyword(explicit_cols)
            return f"SELECT {select_kw} FROM {_render_union_relation_sql(union_specs[0], step_ids, explicit_cols=explicit_cols)} AS u0"
        union_parts = [
            f"SELECT {_render_combine_select_keyword(explicit_cols)} FROM {_render_union_relation_sql(spec, step_ids, explicit_cols=explicit_cols)} AS u{idx}"
            for idx, spec in enumerate(union_specs)
        ]
        return " UNION ALL ".join(union_parts)
    if isinstance(plan.combine, UnionSpec) and not union_specs:
        return _render_union_relation_sql(plan.combine, step_ids, explicit_cols=explicit_cols)
    if not join_specs:
        if len(step_ids) == 1:
            only = next(iter(step_ids.values()))
            select_kw = _render_combine_select_keyword(explicit_cols)
            return f"SELECT {select_kw} FROM {only}"
        return ""
    sources = {step.source_id for step in plan.steps}
    tree = _build_combine_join_tree(join_specs, sources)
    join_sql = _render_combine_tree_sql(
        tree,
        step_ids,
        plan,
        schema=schema,
        manifest=manifest,
        source_by_table=source_by_table,
        explicit_cols=explicit_cols,
    )
    if union_specs:
        cte_defs = _render_federation_union_cte_defs(union_specs, step_ids, explicit_cols=explicit_cols)
        return f"WITH {', '.join(cte_defs)} {join_sql}"
    return join_sql


def semijoin_key_is_allowed(schema: SchemaGraph, table_name: str, column_name: str) -> bool:
    """Return True when *column_name* on *table_name* may participate in semi-join reduction."""
    table = schema.tables.get(table_name)
    if table is None:
        return False
    column = table.columns.get(column_name)
    if column is None:
        return False
    return column.sensitivity == SensitivityClassification.NONE


def semijoin_key_distinct_count(schema: SchemaGraph, table_name: str, column_name: str) -> int | None:
    """Return profiled distinct count for a semi-join key column when known."""
    table = schema.tables.get(table_name)
    if table is None:
        return None
    column = table.columns.get(column_name)
    if column is None:
        return None
    if table.row_count <= 0:
        return None
    return int(column.distinct_count)


def semijoin_key_passes_distinct_floor(schema: SchemaGraph, table_name: str, column_name: str, *, floor: int) -> bool:
    """Return False when profiled cardinality is below *floor*."""
    distinct = semijoin_key_distinct_count(schema, table_name, column_name)
    if distinct is None:
        return True
    return distinct >= int(floor)


def declared_table_for_source_column(
    plan: FederatedPlan,
    source_id: str,
    column_name: str,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str | None:
    """Return a plan-declared table for *column_name* on *source_id*, when known."""
    col = str(column_name or "").strip()
    if "." in col:
        table, _ = split_qualified_column(col, manifest=manifest, schema=schema, source_by_table=source_by_table)
        return table
    step = next((item for item in plan.steps if item.source_id == source_id), None)
    if step is None:
        return None
    for key in step.projected_keys:
        if "." not in key:
            continue
        table, name = split_qualified_column(key, manifest=manifest, schema=schema, source_by_table=source_by_table)
        if name == col:
            return table
    return None


def resolve_source_column_table(
    schema: SchemaGraph,
    source_id: str,
    column_name: str,
    *,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
    declared_table: str | None = None,
) -> str | None:
    """Return the composite table name carrying *column_name* for *source_id*."""
    col = str(column_name or "").strip()
    table_hint = str(declared_table or "").strip() or None
    if "." in col:
        table_hint, col = split_qualified_column(col, manifest=manifest, schema=schema, source_by_table=source_by_table)
    if table_hint:
        table = schema.tables.get(table_hint)
        if table is not None and table.source_id == source_id and col in table.columns:
            return table_hint
        return None
    matches = [name for name, table in schema.tables.items() if table.source_id == source_id and col in table.columns]
    if len(matches) == 1:
        return matches[0]
    return None


def source_timeout_for_source(manifest: FederationManifest, source_id: str) -> int:
    """Resolve per-source execution timeout from binding limits or coordinator defaults."""
    return resolve_member_limits_for_source(manifest, source_id).timeout_ms


def member_guard_limit_kwargs(manifest: FederationManifest | None, source_id: str) -> dict[str, int | float]:
    """Resolved per-member limits for guarded member SQL execution."""
    if manifest is None:
        return {}
    resolved = resolve_member_limits_for_source(manifest, source_id)
    return {
        "timeout_ms": resolved.timeout_ms,
        "max_query_cost_rows": resolved.max_query_cost_rows,
        "max_query_cost_bytes": resolved.max_query_cost_bytes,
        "profile_timeout_ms": resolved.profile_timeout_ms,
    }


def _enforce_federation_row_cap(frame: pd.DataFrame, row_cap: int, *, source_id: str | None = None) -> pd.DataFrame:
    """Raise when *frame* exceeds *row_cap* instead of truncating."""
    if len(frame) > row_cap:
        label = f"source {source_id!r}" if source_id else "coordinator"
        raise FederationCapExceededError(
            f"federation row cap exceeded for {label}: {len(frame)} rows > cap {row_cap}",
            limit_key="row_cap",
            source_id=str(source_id or ""),
        )
    return frame


def coordinator_member_frame_from_pandas(frame: pd.DataFrame) -> CoordinatorMemberFrame:
    """Wrap a pandas member frame for coordinator registration."""
    return CoordinatorMemberFrame(
        kind="pandas",
        table=frame,
        column_names=tuple(str(col) for col in frame.columns),
    )


def normalize_coordinator_member_input(value: pd.DataFrame | CoordinatorMemberFrame) -> CoordinatorMemberFrame:
    """Return a coordinator member payload, wrapping bare pandas frames."""
    if isinstance(value, CoordinatorMemberFrame):
        return value
    return coordinator_member_frame_from_pandas(value)


def dialect_streams_arrow_to_coordinator(dialect: Any) -> bool:
    """Return True when a member dialect should stream Arrow into the coordinator."""
    reader_kind = getattr(dialect, "result_reader_kind", "sqlalchemy")
    return reader_kind in ARROW_RESULT_READER_KINDS


def member_frame_column_names(step: SourceStep) -> tuple[str, ...]:
    """Return unqualified output column names for a federated member step."""
    names: list[str] = []
    for key in step.projected_keys:
        names.append(_unqualified_column_name(key))
    return tuple(names)


def validate_member_frame_projection(
    step: SourceStep,
    frame: pd.DataFrame | CoordinatorMemberFrame | None,
) -> None:
    """Refuse when a member frame's columns do not match the prepared projection."""
    expected = member_frame_column_names(step)
    if not expected or frame is None:
        return
    member = frame if isinstance(frame, CoordinatorMemberFrame) else normalize_coordinator_member_input(frame)
    actual = tuple(member.column_names)
    if actual != expected:
        raise FederationMalformedMemberAnswerError(
            f"federation member {step.source_id!r} returned columns {list(actual)!r} "
            f"but projection requires {list(expected)!r}",
            source_id=step.source_id,
            phase="member",
        )


def coordinator_frame_required_sources(plan: FederatedPlan) -> frozenset[str]:
    """Return source ids that must register coordinator frames (``src_<id>``) for glue SQL."""
    if plan.ineligible_reason:
        return frozenset()
    if len(plan.steps) <= 1:
        return frozenset()
    sources = frozenset(step.source_id for step in plan.steps)
    union_specs = effective_union_specs(plan)
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    if join_specs:
        return sources
    if union_specs:
        return frozenset(sid for spec in union_specs for sid in spec.member_source_ids if sid in sources)
    if len(sources) > 1:
        return sources
    return frozenset()


def validate_federation_scalar_grain_member_frames(plan: FederatedPlan) -> None:
    """Refuse multi-member plans where a scalar-grain member must supply a coordinator frame."""
    required = coordinator_frame_required_sources(plan)
    if not required:
        return
    for step in plan.steps:
        grain = step.sub_intent.grain or "row_level"
        if grain not in VALID_GRAINS:
            grain = "row_level"
        if grain != "scalar":
            continue
        if step.source_id in required:
            raise FederationDeclarationError(
                f"federation member {step.source_id!r} has scalar grain and contributes no "
                f"coordinator frame, but combine requires src_{step.source_id}"
            )


def validate_federation_coordinator_column_types(
    plan: FederatedPlan,
    schema: SchemaGraph,
    *,
    manifest: FederationManifest | None = None,
) -> None:
    """Refuse when a projected coordinator column cannot be mapped to DuckDB."""
    if len(plan.steps) <= 1:
        return
    seen: set[tuple[str, str]] = set()
    for step in plan.steps:
        for key in step.projected_keys:
            pair = (step.source_id, key)
            if pair in seen:
                continue
            seen.add(pair)
            col_name = key.rsplit(".", 1)[-1] if "." in key else key
            table_names = [
                name
                for name, table in schema.tables.items()
                if table.source_id == step.source_id and col_name in table.columns
            ]
            if not table_names:
                continue
            table_name = table_names[0]
            table_meta = schema.tables.get(table_name)
            if table_meta is None:
                continue
            column = table_meta.columns.get(col_name)
            if column is None:
                continue
            data_type = str(column.data_type or "").strip()
            if not data_type:
                continue
            if _schema_column_duckdb_type(data_type) is None:
                raise FederationDeclarationError(
                    f"federation coordinator column {col_name!r} has unsupported data_type "
                    f"{data_type!r} for member {step.source_id!r}"
                )


def validate_coordinator_join_fan_out(
    plan: FederatedPlan,
    member_row_counts: Mapping[str, int],
    result_row_count: int,
    *,
    combine_row_count: int | None = None,
) -> None:
    """Refuse when a coordinator join multiplies rows past a preserved input size."""
    row_count = combine_row_count if combine_row_count is not None else result_row_count
    if row_count <= 0:
        return
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        left_n = int(member_row_counts.get(spec.left_source, 0))
        right_n = int(member_row_counts.get(spec.right_source, 0))
        if kind == "inner":
            for preserved_source, preserved_n, other_source in (
                (spec.left_source, left_n, spec.right_source),
                (spec.right_source, right_n, spec.left_source),
            ):
                if preserved_n <= 0:
                    continue
                if row_count > preserved_n:
                    raise FederationJoinFanOutError(
                        f"federation coordinator inner join produced {row_count} rows from member "
                        f"{preserved_source!r} ({preserved_n} rows) and member {other_source!r}",
                        source_id=other_source,
                        phase="coordinator",
                    )
        elif kind == "left":
            if left_n <= 0:
                continue
            if row_count > left_n:
                raise FederationJoinFanOutError(
                    f"federation coordinator left join produced {row_count} rows from preserved member "
                    f"{spec.left_source!r} ({left_n} rows) and member {spec.right_source!r} ({right_n} rows)",
                    source_id=spec.right_source,
                    phase="coordinator",
                )
        elif kind == "right":
            if right_n <= 0:
                continue
            if row_count > right_n:
                raise FederationJoinFanOutError(
                    f"federation coordinator right join produced {row_count} rows from preserved member "
                    f"{spec.right_source!r} ({right_n} rows) and member {spec.left_source!r} ({left_n} rows)",
                    source_id=spec.left_source,
                    phase="coordinator",
                )


def coordinator_member_row_count(value: pd.DataFrame | CoordinatorMemberFrame | None) -> int:
    """Return the row count for a member execution result."""
    if value is None:
        return 0
    if isinstance(value, CoordinatorMemberFrame):
        return value.row_count()
    return len(value)


def federation_member_parallelism_cap(manifest: FederationManifest | None, step_count: int) -> int:
    """Bound parallel member execution against coordinator configuration."""
    workers = max(1, int(step_count))
    if manifest is None:
        return workers
    cap = max(1, int(manifest.coordinator.max_parallel_members))
    return max(1, min(workers, cap))


def validate_federation_source_slug_uniqueness(sources: Sequence[FederationSourceBinding]) -> None:
    """Raise when two members resolve to the same artifact storage slug."""
    seen: dict[str, str] = {}
    for binding in sources:
        slug = federation_source_storage_slug(binding)
        prior = seen.get(slug)
        if prior is not None and prior != binding.source_id:
            raise FederationConfigError(
                f"federation members {prior!r} and {binding.source_id!r} resolve to the same connection slug {slug!r}"
            )
        seen[slug] = binding.source_id


def federation_member_connection_slug(manifest: FederationManifest | None, source_id: str) -> str:
    """Return the connection slug used to group member execution batches."""
    if manifest is not None:
        for binding in manifest.sources:
            if binding.source_id == source_id:
                return federation_source_storage_slug(binding)
    return str(source_id)


def _reducing_driving_sources_for_step(plan: FederatedPlan | None, step: SourceStep) -> frozenset[str]:
    """Return driving source ids whose frames *step* needs for reduction edges."""
    if plan is None:
        return frozenset()
    member_stage = member_stage_for_source(plan, step.source_id)
    if member_stage is None or not member_stage.reducing_edges:
        return frozenset()
    return frozenset(edge.driving_source_id for edge in member_stage.reducing_edges)


def federation_member_execution_batches(
    steps: Sequence[SourceStep],
    manifest: FederationManifest | None,
    *,
    plan: FederatedPlan | None = None,
) -> list[tuple[SourceStep, ...]]:
    """Partition member steps into batches that never share a connection slug or reduction driver."""
    if not steps:
        return []
    remaining = list(steps)
    batches: list[tuple[SourceStep, ...]] = []
    while remaining:
        used_slugs: set[str] = set()
        batch: list[SourceStep] = []
        next_remaining: list[SourceStep] = []
        remaining_ids = {step.source_id for step in remaining}
        for step in remaining:
            slug = federation_member_connection_slug(manifest, step.source_id)
            if slug in used_slugs:
                next_remaining.append(step)
                continue
            driving_sources = _reducing_driving_sources_for_step(plan, step)
            if driving_sources & remaining_ids:
                next_remaining.append(step)
                continue
            used_slugs.add(slug)
            batch.append(step)
        if not batch:
            batch = [remaining[0]]
            next_remaining = remaining[1:]
        batches.append(tuple(batch))
        remaining = next_remaining
    return batches


def cleanup_abandoned_federation_spill_directories() -> int:
    """Remove orphaned coordinator spill directories from prior process crashes."""
    temp_root = tempfile.gettempdir()
    removed = 0
    try:
        names = os.listdir(temp_root)
    except OSError:
        return 0
    prefix = "aetherdialect_fed_spill_"
    for name in names:
        if not name.startswith(prefix):
            continue
        path = os.path.join(temp_root, name)
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError:
            continue
    return removed


def validate_federation_file_members(members: Mapping[str, Any]) -> None:
    """Validate upload sources for every file-backed federation member."""
    for connection_name, engine in members.items():
        engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
        if not engine_type:
            runtime_cfg = getattr(engine, "_runtime_config", None)
            engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
        if not is_file_engine(engine_type):
            continue
        runtime_cfg = getattr(engine, "_runtime_config", None)
        if runtime_cfg is None or not isinstance(runtime_cfg, CsvRuntimeConfig):
            raise FederationConfigError(f"file-backed federation member {connection_name!r} requires CsvRuntimeConfig")
        upload_paths = runtime_cfg.resolve_source_files()
        report = validate_upload_sources(
            upload_paths, source_selections=parse_source_selections(runtime_cfg.SOURCE_SELECTIONS)
        )
        if not report.ok:
            raise FederationConfigError(f"file-backed federation member {connection_name!r}: {report.narrative}")


def _validate_probe_declared_schema_objects(
    members: Mapping[str, Any],
    manifest: FederationManifest,
    mappings: FederationMappings,
) -> None:
    """Verify declared federation tables and columns exist on member schema graphs."""
    mapping_sources = _mapping_member_source_by_table(mappings)
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            engine = members.get(member.source)
            if engine is None:
                raise FederationConfigError(f"federation member {member.source!r} not registered")
            graph = getattr(engine, "_schema_graph", None)
            if not isinstance(graph, SchemaGraph):
                raise FederationConfigError(f"member {member.source!r} does not expose a schema graph")
            if member.table not in graph.tables:
                raise FederationConfigError(f"declared table {member.source}.{member.table} missing from member schema")
            src_table = graph.tables[member.table]
            for _logical_col, phys_col in member.columns.items():
                if phys_col not in src_table.columns:
                    raise FederationConfigError(
                        f"declared column {member.source}.{member.table}.{phys_col} missing from member schema"
                    )
    for col_map in mappings.logical_columns:
        for qual in col_map.members:
            tbl, col = split_qualified_column(qual, manifest=manifest, source_by_table=mapping_sources)
            source_id = _physical_table_source(tbl, mappings)
            if not source_id:
                source_id = manifest.table_namespace.get(tbl, "")
            if not source_id:
                raise FederationConfigError(f"declared column member unresolved for probe: {qual!r}")
            engine = members.get(source_id)
            if engine is None:
                raise FederationConfigError(f"federation member {source_id!r} not registered")
            graph = getattr(engine, "_schema_graph", None)
            if not isinstance(graph, SchemaGraph):
                raise FederationConfigError(f"member {source_id!r} does not expose a schema graph")
            if tbl not in graph.tables:
                raise FederationConfigError(f"declared table {source_id}.{tbl} missing from member schema")
            if col not in graph.tables[tbl].columns:
                raise FederationConfigError(f"declared column {source_id}.{tbl}.{col} missing from member schema")


def probe_federation_member_connections(
    members: Mapping[str, Any],
    *,
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
) -> None:
    """Probe each database-backed member connection before federation composition."""
    if manifest is not None and mappings is not None:
        _validate_probe_declared_schema_objects(members, manifest, mappings)
    for connection_name, engine in sorted(members.items()):
        engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
        if not engine_type:
            runtime_cfg = getattr(engine, "_runtime_config", None)
            engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
        if is_file_engine(engine_type):
            continue
        sa_engine = getattr(engine, "_execution_engine", None)
        if sa_engine is None:
            raise FederationConfigError(f"federation member {connection_name!r} missing execution engine")
        try:
            with sa_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise FederationMemberProbeError(
                REPHRASE_HINT_MESSAGES["federation_member_probe_failed"],
                source_id=connection_name,
            ) from exc


def probe_federation_member_liveness(members: Mapping[str, Any]) -> None:
    """Re-check database-backed member connections before a federation turn."""
    probe_federation_member_connections(members)


def federation_user_facing_error_message(exc: BaseException) -> str:
    """Return a user-facing federation error message without physical member labels."""
    if isinstance(exc, FederationPartialFailureError):
        return REPHRASE_HINT_MESSAGES["federation_partial_failure"]
    if isinstance(exc, FederationTurnCancelledError):
        return REPHRASE_HINT_MESSAGES["federation_turn_cancelled"]
    if isinstance(exc, FederationMemberExecutionError):
        return REPHRASE_HINT_MESSAGES["federation_member_execution_failed"]
    if isinstance(exc, FederationCapExceededError):
        return REPHRASE_HINT_MESSAGES["federation_cap_exceeded"]
    if isinstance(exc, FederationMemberProbeError):
        return REPHRASE_HINT_MESSAGES["federation_member_probe_failed"]
    return str(exc)


def federation_ineligible_reason_code(reason: str | None) -> str | None:
    """Map a planner ineligible reason string to a stable reason code."""
    if not reason:
        return None
    exact_codes = {
        "no tables referenced": "no_tables",
        "intent references columns outside the active space": "space_scope",
        "projection columns are not held by any single member": "projection_not_single_member",
        "cross-source join path is not declared for referenced sources": "undeclared_join_path",
        "distinct_on requires order_by_cols": "distinct_on_requires_order_by",
        "expression contains unattributable raw_sql fragment": "unattributable_raw_sql",
    }
    if reason in exact_codes:
        return exact_codes[reason]
    prefix_codes = (
        ("cross-source aggregate not supported:", "cross_source_aggregate"),
        ("cross-source OR filter is not supported:", "cross_source_or_filter"),
        ("cross-source where_group disjunction spans sources:", "cross_source_where_group_disjunction"),
        ("cross-source window", "cross_source_window"),
        ("cross-source correlated subquery", "cross_source_correlated_subquery"),
        ("cross-source scalar subquery", "cross_source_scalar_subquery"),
        ("cross-source HAVING", "cross_source_having"),
        ("cross-source distinct_on", "cross_source_distinct_on"),
        ("cross-source semi_join", "cross_source_semijoin"),
        ("cross-source anti_join", "cross_source_antijoin"),
        ("cross-source predicate disjunction", "cross_source_predicate_disjunction"),
        ("semi_join is not supported", "semi_join_unsupported"),
        ("anti_join is not supported", "anti_join_unsupported"),
        ("distinct_on is not supported", "distinct_on_unsupported"),
        ("preserve_tables is not supported", "preserve_tables_unsupported"),
        ("nested predicate groups are not supported", "nested_predicate_groups"),
        ("member capability", "member_capability"),
    )
    for prefix, code in prefix_codes:
        if reason.startswith(prefix):
            return code
    if "raw_sql" in reason or "unattributable" in reason:
        return "unattributable_raw_sql"
    return "unknown"


def federation_ineligible_answerable_hint(reason: str | None) -> str | None:
    """Return a nearest answerable rephrase when *reason* names an ineligible federated shape."""
    code = federation_ineligible_reason_code(reason)
    if not code:
        return None
    return ineligible_answerable_hint_for_code(code)


def _dataframe_memory_bytes(frame: pd.DataFrame) -> int:
    """Return deep memory usage of *frame* in bytes for coordinator byte-cap checks."""
    usage = frame.memory_usage(deep=True)
    return int(usage.sum())


def _coordinator_member_memory_bytes(member: CoordinatorMemberFrame) -> int:
    """Measure coordinator member payload size; probe failures propagate."""
    if member.kind == "arrow":
        return int(member.table.nbytes)
    return _dataframe_memory_bytes(member.table)


def _schema_column_duckdb_type(data_type: str) -> str | None:
    """Map composite schema ``data_type`` text to a DuckDB column type."""
    raw = str(data_type or "").strip()
    if not raw:
        return "VARCHAR"
    base = raw.lower().split("(", 1)[0].strip()
    mapped = FEDERATION_COORDINATOR_DUCKDB_TYPE_MAP.get(base)
    if mapped is not None:
        return mapped
    return None


def _coordinator_column_type_lookup(
    column_name: str,
    source_id: str,
    *,
    schema: SchemaGraph | None,
    plan: FederatedPlan | None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a DuckDB type for *column_name* on *source_id* from composite metadata."""
    if schema is None:
        return None
    step = next((item for item in (plan.steps if plan is not None else ()) if item.source_id == source_id), None)
    candidates: list[str] = []
    if step is not None:
        for key in step.projected_keys:
            if _unqualified_column_name(key) == column_name:
                candidates.append(key)
    for table in schema.tables.values():
        if table.source_id and table.source_id != source_id:
            continue
        if column_name in table.columns:
            candidates.append(f"{table.name}.{column_name}")
    for qualified in candidates:
        if "." in qualified:
            tbl, col = split_qualified_column(
                qualified, manifest=manifest, schema=schema, source_by_table=source_by_table
            )
        else:
            tbl, col = "", qualified
        resolved_table: TableMetadata | None = schema.tables.get(tbl) if tbl else None
        if resolved_table is None:
            for _name, meta_table in schema.tables.items():
                if meta_table.source_id == source_id and col in meta_table.columns:
                    resolved_table = meta_table
                    break
        if resolved_table is None:
            continue
        meta = resolved_table.columns.get(col)
        if meta is not None and str(meta.data_type or "").strip():
            return _schema_column_duckdb_type(meta.data_type)
    return None


def _coordinator_relation_column_types(
    frame: pd.DataFrame,
    source_id: str,
    *,
    schema: SchemaGraph | None,
    plan: FederatedPlan | None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(column, duckdb_type)`` pairs for a coordinator input frame."""
    return _coordinator_relation_column_types_from_names(
        tuple(str(column) for column in frame.columns),
        source_id,
        schema=schema,
        plan=plan,
        manifest=manifest,
        source_by_table=source_by_table,
    )


def _coordinator_relation_column_types_from_names(
    column_names: Sequence[str],
    source_id: str,
    *,
    schema: SchemaGraph | None,
    plan: FederatedPlan | None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(column, duckdb_type)`` pairs for named coordinator columns."""
    types: list[tuple[str, str]] = []
    if source_by_table is None and schema is not None:
        source_by_table = source_by_table_from_schema(schema)
    for column in column_names:
        col_name = str(column)
        declared = _coordinator_column_type_lookup(
            col_name,
            source_id,
            schema=schema,
            plan=plan,
            manifest=manifest,
            source_by_table=source_by_table,
        )
        if declared is not None:
            types.append((col_name, declared))
            continue
        if schema is not None:
            table_name = resolve_source_column_table(
                schema,
                source_id,
                col_name,
                manifest=manifest,
                source_by_table=source_by_table,
            )
            if table_name:
                table_meta = schema.tables.get(table_name)
                lookup_col = col_name.rsplit(".", 1)[-1] if "." in col_name else col_name
                meta = table_meta.columns.get(lookup_col) if table_meta is not None else None
                data_type = str(meta.data_type or "").strip() if meta is not None else ""
                if data_type and _schema_column_duckdb_type(data_type) is None:
                    raise FederationDeclarationError(
                        f"federation coordinator column {lookup_col!r} has unsupported data_type "
                        f"{data_type!r} for member {source_id!r}"
                    )
        types.append((col_name, _schema_column_duckdb_type("") or "VARCHAR"))
    return types


def _coordinator_duckdb_type_to_pyarrow(duckdb_type: str) -> Any:
    """Map a DuckDB column type string to a PyArrow type for coordinator spill files."""
    import pyarrow as pa

    dtype = str(duckdb_type or "").strip().upper()
    if not dtype or dtype.startswith("VARCHAR") or dtype.startswith("TEXT"):
        return pa.string()
    if dtype.startswith("BIGINT"):
        return pa.int64()
    if dtype.startswith("SMALLINT") or dtype.startswith("INT2"):
        return pa.int16()
    if dtype.startswith("INTEGER") or dtype.startswith("INT"):
        return pa.int32()
    if dtype.startswith("DECIMAL") or dtype.startswith("NUMERIC"):
        return pa.decimal128(38, 9)
    if dtype.startswith("DOUBLE") or dtype.startswith("FLOAT8"):
        return pa.float64()
    if dtype.startswith("REAL") or dtype.startswith("FLOAT4"):
        return pa.float32()
    if dtype.startswith("BOOL"):
        return pa.bool_()
    if dtype.startswith("TIMESTAMP"):
        return pa.timestamp("us")
    if dtype == "DATE":
        return pa.date32()
    if dtype.startswith("TIME"):
        return pa.time64("us")
    if dtype.startswith("UUID"):
        return pa.string()
    if dtype.startswith("BLOB") or dtype.startswith("BINARY"):
        return pa.binary()
    return pa.string()


def _write_coordinator_spill_parquet(
    frame: pd.DataFrame, spill_path: str, column_types: Sequence[tuple[str, str]]
) -> None:
    """Write a coordinator spill file with explicit declared column types."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [pa.field(str(col), _coordinator_duckdb_type_to_pyarrow(dtype)) for col, dtype in column_types]
    schema = pa.schema(fields)
    arrays = [
        pa.array(frame[str(col)].tolist(), type=_coordinator_duckdb_type_to_pyarrow(dtype))
        for col, dtype in column_types
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    write_table = cast(Callable[[Any, str], None], pq.write_table)
    write_table(table, spill_path)


def _write_coordinator_spill_parquet_arrow(
    arrow_table: Any, spill_path: str, column_types: Sequence[tuple[str, str]]
) -> None:
    """Write a coordinator spill file from an Arrow table with explicit declared types."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [pa.field(str(col), _coordinator_duckdb_type_to_pyarrow(dtype)) for col, dtype in column_types]
    schema = pa.schema(fields)
    arrays = [
        pa.array(arrow_table.column(str(col)).to_pylist(), type=_coordinator_duckdb_type_to_pyarrow(dtype))
        for col, dtype in column_types
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    write_table = cast(Callable[[Any, str], None], pq.write_table)
    write_table(table, spill_path)


def _create_coordinator_typed_table_sql(reg_name: str, column_types: Sequence[tuple[str, str]]) -> str:
    col_defs = ", ".join(f"{_quote_ident(col)} {dtype}" for col, dtype in column_types)
    return f"CREATE OR REPLACE TABLE {_quote_ident(reg_name)} ({col_defs})"


def _insert_coordinator_typed_frame(
    conn: Any, reg_name: str, frame: pd.DataFrame, column_types: Sequence[tuple[str, str]]
) -> None:
    if frame.empty:
        return
    columns = [col for col, _dtype in column_types]
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(_quote_ident(col) for col in columns)
    insert_sql = f"INSERT INTO {_quote_ident(reg_name)} ({col_sql}) VALUES ({placeholders})"
    rows = frame.loc[:, columns].itertuples(index=False, name=None)
    conn.executemany(insert_sql, list(rows))


def _register_coordinator_frame(
    conn: Any,
    reg_name: str,
    frame: pd.DataFrame | CoordinatorMemberFrame,
    *,
    row_cap: int,
    spill_threshold: int,
    spill_dir: str,
    source_id: str | None = None,
    schema: SchemaGraph | None = None,
    plan: FederatedPlan | None = None,
    spill_files_created: list[str] | None = None,
) -> None:
    """Register a coordinator input frame with declared composite key types."""
    member = normalize_coordinator_member_input(frame)
    if member.kind == "arrow":
        _register_coordinator_arrow_table(
            conn,
            reg_name,
            member,
            row_cap=row_cap,
            spill_threshold=spill_threshold,
            spill_dir=spill_dir,
            source_id=source_id,
            schema=schema,
            plan=plan,
            spill_files_created=spill_files_created,
        )
        return
    bounded = _enforce_federation_row_cap(member.table, row_cap, source_id=source_id)
    column_types = _coordinator_relation_column_types(bounded, str(source_id or ""), schema=schema, plan=plan)
    typed_select = ", ".join(
        f"CAST({_quote_ident(col)} AS {dtype}) AS {_quote_ident(col)}" for col, dtype in column_types
    )
    conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(reg_name)}")
    conn.execute(_create_coordinator_typed_table_sql(reg_name, column_types))
    if len(bounded) <= spill_threshold:
        _insert_coordinator_typed_frame(conn, reg_name, bounded, column_types)
        return
    os.makedirs(spill_dir, exist_ok=True)
    spill_path = os.path.join(spill_dir, f"{reg_name}.parquet")
    _write_coordinator_spill_parquet(bounded, spill_path, column_types)
    if spill_files_created is not None:
        spill_files_created.append(spill_path)
    conn.execute(
        f"CREATE OR REPLACE VIEW {_quote_ident(reg_name)} AS "
        f"SELECT {typed_select} FROM read_parquet({_quote_sql_string(spill_path)})"
    )


def _register_coordinator_arrow_table(
    conn: Any,
    reg_name: str,
    member: CoordinatorMemberFrame,
    *,
    row_cap: int,
    spill_threshold: int,
    spill_dir: str,
    source_id: str | None = None,
    schema: SchemaGraph | None = None,
    plan: FederatedPlan | None = None,
    spill_files_created: list[str] | None = None,
) -> None:
    """Register an Arrow member table with declared composite key types."""
    import pyarrow as pa

    arrow_table = member.table
    if member.column_names and len(member.column_names) == arrow_table.num_columns:
        arrow_table = pa.table(
            [arrow_table.column(idx) for idx in range(arrow_table.num_columns)],
            names=list(member.column_names),
        )
    row_count = int(arrow_table.num_rows)
    if row_count > row_cap:
        label = f"source {source_id!r}" if source_id else "coordinator"
        raise FederationCapExceededError(
            f"federation row cap exceeded for {label}: {row_count} rows > cap {row_cap}",
            limit_key="row_cap",
            source_id=str(source_id or ""),
        )
    column_types = _coordinator_relation_column_types_from_names(
        member.column_names or tuple(str(name) for name in arrow_table.column_names),
        str(source_id or ""),
        schema=schema,
        plan=plan,
    )
    typed_select = ", ".join(
        f"CAST({_quote_ident(col)} AS {dtype}) AS {_quote_ident(col)}" for col, dtype in column_types
    )
    staging = f"__{reg_name}_arrow"
    try:
        conn.unregister(staging)
    except Exception:
        pass
    conn.register(staging, arrow_table)
    conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(reg_name)}")
    if row_count <= spill_threshold:
        conn.execute(_create_coordinator_typed_table_sql(reg_name, column_types))
        conn.execute(f"INSERT INTO {_quote_ident(reg_name)} SELECT {typed_select} FROM {staging}")
        conn.unregister(staging)
        return
    os.makedirs(spill_dir, exist_ok=True)
    spill_path = os.path.join(spill_dir, f"{reg_name}.parquet")
    _write_coordinator_spill_parquet_arrow(arrow_table, spill_path, column_types)
    if spill_files_created is not None:
        spill_files_created.append(spill_path)
    conn.unregister(staging)
    conn.execute(
        f"CREATE OR REPLACE VIEW {_quote_ident(reg_name)} AS "
        f"SELECT {typed_select} FROM read_parquet({_quote_sql_string(spill_path)})"
    )


def _quote_sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _validate_coordinator_glue_sql(
    sql: str, bind_map: Mapping[str, Any] | None, *, schema: SchemaGraph | None = None, conn: Any | None = None
) -> None:
    """Validate coordinator glue SQL through the DuckDB dialect gate."""
    dialect = get_dialect("duckdb", native_connection=conn)
    ok, err, _cat, _diags = validate_sql(dialect, sql, dict(bind_map or {}), schema=schema)
    if not ok:
        raise FederationRuntimeError(f"coordinator glue validation failed: {err or 'invalid SQL'}")


def federation_coordinator_timeout_error(timeout_ms: int, exc: Exception | None = None) -> FederationCapExceededError:
    """Wrap a coordinator glue timeout as a typed federation cap breach."""
    detail = f": {exc}" if exc is not None else ""
    return FederationCapExceededError(
        f"federation coordinator glue timeout exceeded after {int(timeout_ms)}ms{detail}",
        limit_key="coordinator_timeout_ms",
        source_id="coordinator",
    )


def federation_plan_timeout_error(elapsed_ms: int, timeout_ms: int) -> FederationCapExceededError:
    """Wrap a whole-plan wall-clock timeout as a typed federation cap breach."""
    return FederationCapExceededError(
        f"federation plan timeout exceeded after {int(elapsed_ms)}ms (limit {int(timeout_ms)}ms)",
        limit_key="plan_timeout_ms",
        source_id="",
    )


def federation_plan_timeout_deadline(plan_timeout_ms: int | None, *, started_at: float | None = None) -> float | None:
    """Return a monotonic deadline for a federated plan wall-clock budget."""
    if not cost_cap_active(plan_timeout_ms):
        return None
    start = started_at if started_at is not None else time.perf_counter()
    return start + (int(plan_timeout_ms) / 1000.0)


def enforce_federation_plan_timeout(deadline: float | None, *, started_at: float) -> None:
    """Raise when the federated plan wall-clock budget has been exhausted."""
    if deadline is None:
        return
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    timeout_ms = int((deadline - started_at) * 1000)
    if time.perf_counter() >= deadline:
        raise federation_plan_timeout_error(elapsed_ms, timeout_ms)


def _execute_coordinator_sql_with_timeout(
    conn: Any,
    sql: str,
    bind_map: Mapping[str, Any] | None,
    *,
    timeout_ms: int | None,
) -> Any:
    """Execute coordinator DuckDB SQL with an optional wall-clock timeout."""
    params = dict(bind_map or {})
    if not cost_cap_active(timeout_ms):
        return conn.execute(sql, params or {})
    deadline = time.perf_counter() + (int(timeout_ms) / 1000.0)
    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def _run() -> None:
        try:
            result_holder.append(conn.execute(sql, params or {}))
        except BaseException as exc:
            error_holder.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, deadline - time.perf_counter()))
    if worker.is_alive():
        interrupt = getattr(conn, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception:
                pass
        worker.join(timeout=1.0)
        raise federation_coordinator_timeout_error(int(timeout_ms))
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise federation_coordinator_timeout_error(int(timeout_ms))
    return result_holder[0]


def _enforce_coordinator_probe_key_caps(
    plan: FederatedPlan,
    frames: Mapping[str, pd.DataFrame | CoordinatorMemberFrame],
    *,
    source_by_table: Mapping[str, str],
    semijoin_key_cap: int,
) -> None:
    """Raise when a lifted probe member exceeds the semijoin key cap."""
    if not plan.lifted_probe_ctes:
        return
    owners = _assign_cte_sources(plan.lifted_probe_ctes, source_by_table)
    for cte in plan.lifted_probe_ctes:
        owner = owners.get(cte.cte_name or "")
        if not owner:
            continue
        frame = frames.get(owner)
        if frame is None:
            continue
        for key in _cte_probe_join_keys(cte):
            column = _unqualified_column_name(key)
            if not column:
                continue
            keys = distinct_semijoin_keys(frame, column, cap=semijoin_key_cap)
            if keys is None:
                raise FederationCapExceededError(
                    f"federation semijoin key cap exceeded for member {owner!r}: "
                    f"distinct keys on {column!r} exceed cap {semijoin_key_cap}",
                    limit_key="semijoin_key_cap",
                    source_id=owner,
                )


def execute_federation_coordinator(
    frames: MutableMapping[str, pd.DataFrame | CoordinatorMemberFrame],
    plan: FederatedPlan,
    *,
    row_cap: int | None = None,
    spill_row_threshold: int | None = None,
    spill_dir: str | None = None,
    schema: SchemaGraph | None = None,
    param_values: Mapping[str, Any] | None = None,
    total_input_byte_cap: int | None = None,
    semijoin_key_cap: int | None = None,
    coordinator_timeout_ms: int | None = None,
) -> pd.DataFrame:
    """Combine per-source frames using DuckDB."""
    if plan.ineligible_reason:
        raise FederationIneligibleError(plan.ineligible_reason)
    if not frames:
        if plan.residual is not None and _residual_is_aggregate_only(plan.residual):
            headers = federation_residual_column_headers(plan)
            identity = aggregate_identity_row_for_residual(plan.residual)
            if headers and len(headers) == len(identity):
                return pd.DataFrame([identity], columns=list(headers))
            return pd.DataFrame([identity])
        return pd.DataFrame()
    defaults = FederationCoordinatorConfig()
    cap = row_cap if row_cap is not None else defaults.row_cap
    spill_threshold = spill_row_threshold if spill_row_threshold is not None else defaults.spill_row_threshold
    byte_cap = total_input_byte_cap if total_input_byte_cap is not None else defaults.total_input_byte_cap
    key_cap = semijoin_key_cap if semijoin_key_cap is not None else defaults.semijoin_key_cap
    glue_timeout = coordinator_timeout_ms if coordinator_timeout_ms is not None else defaults.coordinator_timeout_ms
    if schema is not None and plan.lifted_probe_ctes:
        _enforce_coordinator_probe_key_caps(
            plan,
            frames,
            source_by_table=source_by_table_from_schema(schema),
            semijoin_key_cap=key_cap,
        )
    conn = duckdb.connect(":memory:")
    step_ids: dict[str, str] = {}
    owned_spill = spill_dir is None
    spill_path = spill_dir or tempfile.mkdtemp(prefix="aetherdialect_fed_spill_")
    os.makedirs(spill_path, mode=0o700, exist_ok=True)
    spill_files_created: list[str] = []
    source_count = len(frames)
    bind_map = coordinator_residual_bind_map(plan, dict(param_values or {}))
    total_rows = 0
    total_bytes = 0
    member_row_counts = {
        source_id: normalize_coordinator_member_input(frame).row_count() for source_id, frame in frames.items()
    }
    try:
        for source_id in list(frames):
            frame = frames[source_id]
            member = normalize_coordinator_member_input(frame)
            member_rows = member.row_count()
            member_bytes = _coordinator_member_memory_bytes(member)
            next_rows = total_rows + member_rows
            next_bytes = total_bytes + member_bytes
            if next_rows > cap:
                raise FederationCapExceededError(
                    f"federation coordinator total input row cap exceeded for source {source_id!r}: "
                    f"{next_rows} rows > cap {cap}",
                    limit_key="total_input_row_cap",
                    source_id=source_id,
                )
            if next_bytes > byte_cap:
                raise FederationCapExceededError(
                    f"federation coordinator total input byte cap exceeded for source {source_id!r}: "
                    f"{next_bytes} bytes > cap {byte_cap}",
                    limit_key="total_input_byte_cap",
                    source_id=source_id,
                )
            total_rows = next_rows
            total_bytes = next_bytes
            reg = f"src_{source_id}"
            _register_coordinator_frame(
                conn,
                reg,
                member,
                row_cap=cap,
                spill_threshold=spill_threshold,
                spill_dir=spill_path,
                source_id=source_id,
                schema=schema,
                plan=plan,
                spill_files_created=spill_files_created,
            )
            step_ids[source_id] = reg
            frames.pop(source_id, None)
            del frame
            del member
        glue = render_federation_glue(plan, step_ids, schema=schema, param_values=param_values)
        if not glue:
            if source_count == 1:
                only_reg = next(iter(step_ids.values()))
                explicit_cols = _combine_select_column_names(plan)
                select_kw = _render_combine_select_keyword(explicit_cols)
                single_sql = f"SELECT {select_kw} FROM {only_reg}"
                _validate_coordinator_glue_sql(single_sql, {}, schema=schema, conn=conn)
                result = _execute_coordinator_sql_with_timeout(conn, single_sql, {}, timeout_ms=glue_timeout).fetchdf()
                result = _enforce_federation_row_cap(result, cap)
                enforce_coordinator_result_grain(result, plan)
                return result
            raise FederationRuntimeError("federation glue SQL is empty")
        _assert_combine_join_plan_structure(plan)
        exec_bind = reconcile_execute_bind_params(glue, bind_map) or {}
        _validate_coordinator_glue_sql(glue, exec_bind, schema=schema, conn=conn)
        try:
            result = _execute_coordinator_sql_with_timeout(conn, glue, exec_bind, timeout_ms=glue_timeout).fetchdf()
        except FederationCapExceededError:
            raise
        except Exception as exc:
            raise FederationRuntimeError(f"coordinator glue execution failed: {exc}") from exc
        if result.empty and plan.residual is not None and _residual_is_aggregate_only(plan.residual):
            headers = federation_residual_column_headers(plan)
            identity = aggregate_identity_row_for_residual(plan.residual)
            if headers and len(headers) == len(identity):
                result = pd.DataFrame([identity], columns=list(headers))
            else:
                result = pd.DataFrame([identity])
        result = _enforce_federation_row_cap(result, cap)
        combine_row_count = len(result)
        combine_glue = _render_federation_combine_only_glue(plan, step_ids, schema=schema)
        if combine_glue:
            try:
                combine_row_count = len(
                    _execute_coordinator_sql_with_timeout(
                        conn, combine_glue, exec_bind, timeout_ms=glue_timeout
                    ).fetchdf()
                )
            except Exception:
                combine_row_count = len(result)
        validate_coordinator_join_fan_out(plan, member_row_counts, len(result), combine_row_count=combine_row_count)
        validate_federated_residual_aggregate_fan_out(plan, schema)
        enforce_coordinator_result_grain(result, plan)
        return result
    finally:
        conn.close()
        if owned_spill:
            try:
                shutil.rmtree(spill_path)
            except OSError as exc:
                raise FederationRuntimeError(f"failed to clean federation coordinator spill directory: {exc}") from exc
        elif spill_files_created:
            for created_path in spill_files_created:
                try:
                    os.remove(created_path)
                except OSError as exc:
                    raise FederationRuntimeError(
                        f"failed to clean federation coordinator spill file {created_path!r}: {exc}"
                    ) from exc


def _assert_combine_join_plan_structure(plan: FederatedPlan) -> None:
    """Assert declared combine joins form a connected join tree from plan IR."""
    if not isinstance(plan.combine, tuple) or not plan.combine:
        return
    sources = {step.source_id for step in plan.steps}
    if len(sources) < 2:
        return
    tree = _build_combine_join_tree(plan.combine, sources)
    if not tree.children:
        raise FederationRuntimeError("cross-source join plan is missing join edges for declared combine specs")


def order_federation_execution_steps(
    plan: FederatedPlan,
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
) -> tuple[SourceStep, ...]:
    """Return source steps ordered by stage dependencies, then estimated selectivity."""
    stage_order = derive_execution_order_from_stages(plan)
    stage_rank = {source_id: idx for idx, source_id in enumerate(stage_order)}
    source_by_table = source_by_table_from_schema(schema)

    def join_key_selectivity(step: SourceStep) -> float:
        if schema is None or not step.projected_keys:
            return 1.0
        ratios: list[float] = []
        for key in step.projected_keys:
            table_name = resolve_source_column_table(
                schema,
                step.source_id,
                key,
                manifest=manifest,
                source_by_table=source_by_table,
            )
            if not table_name:
                continue
            table_meta = schema.tables.get(table_name)
            if table_meta is None:
                continue
            col_name = key.rsplit(".", 1)[-1] if "." in key else key
            column = table_meta.columns.get(col_name)
            if column is not None and column.distinct_ratio is not None:
                ratios.append(float(column.distinct_ratio))
        if not ratios:
            return 1.0
        return min(ratios)

    def selectivity_score(step: SourceStep) -> tuple[int, int, float, int, int, str]:
        limit = int(step.sub_intent.limit) if step.sub_intent.limit else 0
        filters = len(_predicate_where_leaves(step.sub_intent.where))
        grain_rank = 0 if (step.sub_intent.grain or "many") == "scalar" else 1
        limit_rank = limit if limit > 0 else 10**9
        rank = stage_rank.get(step.source_id, len(stage_rank))
        return (rank, grain_rank, join_key_selectivity(step), limit_rank, -filters, step.source_id)

    return tuple(sorted(plan.steps, key=selectivity_score))


def federation_execution_wave_member_steps(waves: Sequence[FederationExecutionWave]) -> tuple[SourceStep, ...]:
    """Return member steps from *waves* in execution order."""
    return tuple(step for wave in waves for step in wave.member_steps)


def federation_stage_execution_waves(
    plan: FederatedPlan, execution_steps: Sequence[SourceStep], *, schema: SchemaGraph | None = None
) -> list[FederationExecutionWave]:
    """Derive execution waves for every federated stage, preserving member ordering."""
    steps = tuple(execution_steps)
    if not plan.stages:
        ordered = order_federation_execution_steps(plan, schema=schema)
        if not ordered:
            return []
        return [
            FederationExecutionWave(
                stage=FederatedStage(
                    stage_id="member_wave_0",
                    kind="member",
                    source_ids=tuple(step.source_id for step in ordered),
                ),
                member_steps=ordered,
            )
        ]
    ordered_stages = derive_federation_stages_in_order(plan)
    stage_by_id = {stage.stage_id: stage for stage in plan.stages}
    step_by_source = {step.source_id: step for step in steps}
    depth_cache: dict[str, int] = {}

    def stage_depth(stage_id: str) -> int:
        cached = depth_cache.get(stage_id)
        if cached is not None:
            return cached
        stage = stage_by_id.get(stage_id)
        if stage is None or not stage.depends_on:
            depth_cache[stage_id] = 0
            return 0
        deps = [dep for dep in stage.depends_on if dep in stage_by_id]
        depth = 1 + max((stage_depth(dep) for dep in deps), default=-1)
        depth_cache[stage_id] = depth
        return depth

    stage_depths = {stage.stage_id: stage_depth(stage.stage_id) for stage in plan.stages}
    if not stage_depths:
        ordered = order_federation_execution_steps(plan, schema=schema)
        if not ordered:
            return []
        return [
            FederationExecutionWave(
                stage=FederatedStage(
                    stage_id="member_wave_0",
                    kind="member",
                    source_ids=tuple(step.source_id for step in ordered),
                ),
                member_steps=ordered,
            )
        ]
    max_depth = max(stage_depths.values())
    waves: list[FederationExecutionWave] = []
    for depth in range(max_depth + 1):
        depth_stages = [stage for stage in ordered_stages if stage_depths.get(stage.stage_id) == depth]
        member_stages = [stage for stage in depth_stages if stage.kind == "member"]
        if member_stages:
            member_steps = tuple(
                step_by_source[stage.source_ids[0]]
                for stage in member_stages
                if stage.source_ids and stage.source_ids[0] in step_by_source
            )
            if member_steps:
                waves.append(
                    FederationExecutionWave(
                        stage=FederatedStage(
                            stage_id=f"member_wave_{depth}",
                            kind="member",
                            source_ids=tuple(stage.source_ids[0] for stage in member_stages if stage.source_ids),
                        ),
                        member_steps=member_steps,
                    )
                )
        for stage in depth_stages:
            if stage.kind == "member":
                continue
            waves.append(FederationExecutionWave(stage=stage, member_steps=()))
    return waves or (
        [
            FederationExecutionWave(
                stage=FederatedStage(
                    stage_id="member_wave_0", kind="member", source_ids=tuple(s.source_id for s in steps)
                ),
                member_steps=steps,
            )
        ]
        if steps
        else []
    )


def source_row_cap_for_source(manifest: FederationManifest, source_id: str) -> int:
    """Resolve per-source row cap from binding limits or coordinator defaults."""
    return resolve_member_limits_for_source(manifest, source_id).row_cap


def resolve_member_limits_for_source(manifest: FederationManifest, source_id: str) -> FederationMemberResolvedLimits:
    """Resolve per-member limits using member, coordinator, then global policy fallbacks."""
    binding_limits: FederationSourceLimits | None = None
    for binding in manifest.sources:
        if binding.source_id == source_id:
            binding_limits = binding.limits
            break
    row_cap = int(manifest.coordinator.default_source_row_cap)
    if binding_limits is not None and binding_limits.row_cap is not None:
        row_cap = int(binding_limits.row_cap)
    timeout_ms = int(manifest.coordinator.default_source_timeout_ms)
    if binding_limits is not None and binding_limits.timeout_ms is not None:
        timeout_ms = int(binding_limits.timeout_ms)
    max_query_cost_rows = PolicyConfig.MAX_QUERY_COST_ROWS
    if binding_limits is not None and binding_limits.max_query_cost_rows is not None:
        max_query_cost_rows = float(binding_limits.max_query_cost_rows)
    max_query_cost_bytes = PolicyConfig.MAX_QUERY_COST_BYTES
    if binding_limits is not None and binding_limits.max_query_cost_bytes is not None:
        max_query_cost_bytes = float(binding_limits.max_query_cost_bytes)
    profile_timeout_ms = PolicyConfig.PROFILE_TIMEOUT_MS
    if binding_limits is not None and binding_limits.profile_timeout_ms is not None:
        profile_timeout_ms = int(binding_limits.profile_timeout_ms)
    return FederationMemberResolvedLimits(
        source_id=source_id,
        row_cap=row_cap,
        timeout_ms=timeout_ms,
        max_query_cost_rows=max_query_cost_rows,
        max_query_cost_bytes=max_query_cost_bytes,
        profile_timeout_ms=profile_timeout_ms,
    )


def federation_member_resolved_limits(
    plan: FederatedPlan, manifest: FederationManifest
) -> tuple[FederationMemberResolvedLimits, ...]:
    """Return resolved per-member limits for every step in *plan*."""
    return tuple(resolve_member_limits_for_source(manifest, step.source_id) for step in plan.steps)


def federation_member_timeout_error(source_id: str, exc: Exception) -> FederationCapExceededError:
    """Wrap a member statement timeout as a typed federation cap breach."""
    return FederationCapExceededError(
        f"federation timeout exceeded for source {source_id!r}: {exc}", limit_key="timeout_ms", source_id=source_id
    )


def source_semijoin_enabled(manifest: FederationManifest, source_id: str) -> bool:
    """Return whether semi-join reduction is enabled for *source_id*."""
    for binding in manifest.sources:
        if binding.source_id == source_id:
            if binding.limits is not None:
                return bool(binding.limits.semijoin_enabled)
            return True
    return True


def column_where_value_type(schema: SchemaGraph, table_name: str, column_name: str) -> str:
    """Map a composite column type to an intent filter value_type token."""
    table = schema.tables.get(table_name)
    if table is None:
        return "string"
    meta = table.columns.get(column_name)
    if meta is None:
        return "string"
    declared = str(meta.value_type or "").strip().lower()
    if declared:
        return declared
    dtype = str(meta.data_type or "").lower()
    if any(token in dtype for token in ("int", "bigint", "smallint")):
        return "integer"
    if any(token in dtype for token in ("decimal", "numeric", "double", "float", "real")):
        return "number"
    if any(token in dtype for token in ("date", "timestamp", "time")):
        return "date"
    if "bool" in dtype:
        return "boolean"
    return "string"


def federation_coordinator_spill_dir(federation_dir: str | None) -> str | None:
    """Return the coordinator spill directory under a federation artifact tree."""
    if not federation_dir or not str(federation_dir).strip():
        return None
    path = os.path.join(str(federation_dir), "coordinator_spill")
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise FederationRuntimeError(f"failed to clean federation coordinator spill directory: {exc}") from exc
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def semijoin_key_columns(plan: FederatedPlan, driving_source: str, target_source: str) -> tuple[str, str] | None:
    """Return driving-frame and target-intent join key columns for semi- join reduction."""
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    if effective_union_specs(plan) or not join_specs:
        return None
    for spec in join_specs:
        if spec.left_source == driving_source and spec.right_source == target_source:
            return spec.left_key, spec.right_key
        if spec.right_source == driving_source and spec.left_source == target_source:
            return spec.right_key, spec.left_key
    return None


def distinct_semijoin_keys(frame: pd.DataFrame | CoordinatorMemberFrame, column: str, *, cap: int) -> list[Any] | None:
    """Return distinct non-null keys when within *cap*; otherwise None (skip reduction)."""
    if isinstance(frame, CoordinatorMemberFrame):
        return _distinct_semijoin_keys_arrow(frame, column, cap=cap)
    if column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return []
    keys = series.unique().tolist()
    if len(keys) > cap:
        return None
    return keys


def _distinct_semijoin_keys_arrow(member: CoordinatorMemberFrame, column: str, *, cap: int) -> list[Any] | None:
    """Return distinct non-null keys from an Arrow member frame when within *cap*."""
    import pyarrow.compute as pc

    col_name = _unqualified_column_name(column)
    if col_name not in member.column_names:
        return None
    arrow_table = member.table
    if member.column_names and len(member.column_names) == arrow_table.num_columns:
        import pyarrow as pa

        arrow_table = pa.table(
            [arrow_table.column(idx) for idx in range(arrow_table.num_columns)],
            names=list(member.column_names),
        )
    if col_name not in arrow_table.column_names:
        return None
    values = arrow_table.column(col_name)
    pc_ops = cast(Any, pc)
    filtered = pc_ops.drop_null(values)
    if len(filtered) == 0:
        return []
    unique = pc_ops.unique(filtered)
    if len(unique) > cap:
        return None
    return list(unique.to_pylist())


def _next_injected_param_key(intent: RuntimeIntent) -> str:
    """Allocate a ``p*`` handle above every key already referenced on *intent*."""
    max_idx = 0
    for key in collect_intent_referenced_param_keys(intent):
        if len(key) > 1 and key.startswith("p") and key[1:].isdigit():
            max_idx = max(max_idx, int(key[1:]))
    for key in intent.param_values or {}:
        if len(key) > 1 and key.startswith("p") and key[1:].isdigit():
            max_idx = max(max_idx, int(key[1:]))
    return f"p{max_idx + 1}"


def inject_semijoin_where(
    sub_intent: RuntimeIntent, key_column: str, keys: Sequence[Any], *, value_type: str = "string"
) -> RuntimeIntent:
    """Inject an ``IN`` filter on *key_column* into *sub_intent* for structural semi-join reduction."""
    return _inject_reducing_key_where(
        sub_intent,
        key_column,
        keys,
        value_type=value_type,
        empty_sentinel="__AETHERDIALECT_EMPTY_SEMIJOIN__",
    )


def inject_filter_keys_where(
    sub_intent: RuntimeIntent, key_column: str, keys: Sequence[Any], *, value_type: str = "string"
) -> RuntimeIntent:
    """Inject an ``IN`` filter on *key_column* for join-covered cross- source filter pushdown."""
    return _inject_reducing_key_where(
        sub_intent,
        key_column,
        keys,
        value_type=value_type,
        empty_sentinel="__AETHERDIALECT_EMPTY_FILTER_KEYS__",
    )


def _inject_reducing_key_where(
    sub_intent: RuntimeIntent,
    key_column: str,
    keys: Sequence[Any],
    *,
    value_type: str,
    empty_sentinel: str,
) -> RuntimeIntent:
    """Inject an ``IN`` filter on *key_column* into *sub_intent*."""
    param_key = _next_injected_param_key(sub_intent)
    bind_value = list(keys) if keys else [empty_sentinel]
    fp = WhereParam(
        left_expr=NormalizedExpr.from_column(key_column), op="in", value_type=value_type, param_key=param_key
    )
    filters = list((_predicate_where_leaves(sub_intent.where)) or [])
    filters.append(fp)
    merged_where = PredicateGroup(op="and", predicates=tuple(filters))
    param_values = dict(sub_intent.param_values or {})
    param_values[param_key] = bind_value
    return replace(sub_intent, where=merged_where, param_values=param_values)


def detect_federation_topology_change(
    recorded_source_ids: Sequence[str], manifest: FederationManifest
) -> FederationTopologyChange:
    """Compare recorded federation members against the manifest source set."""
    recorded = set(recorded_source_ids)
    declared = {binding.source_id for binding in manifest.sources}
    added = declared - recorded
    removed = recorded - declared
    if added and removed:
        return "mixed"
    if added:
        return "add"
    if removed:
        return "remove"
    return "none"


def prune_federation_aliases(manifest: FederationManifest, *, active_source_ids: set[str]) -> FederationManifest:
    """Drop table aliases that reference sources no longer in the federation."""
    kept = tuple(alias for alias in manifest.aliases if alias.source in active_source_ids)
    return replace(manifest, aliases=kept)


def reconcile_authored_declaration_for_members(
    manifest: FederationManifest,
    mappings: FederationMappings,
    *,
    active_source_ids: set[str],
) -> tuple[FederationManifest, FederationMappings]:
    """Prune authored declaration sections that reference removed federation members."""
    pruned = prune_federation_aliases(manifest, active_source_ids=active_source_ids)
    if pruned.table_namespace:
        pruned = prune_cross_source_joins(pruned, active_source_ids=active_source_ids)
    pruned_mappings = prune_federation_mappings(mappings, pruned, active_source_ids=active_source_ids)
    return pruned, pruned_mappings


def prune_cross_source_joins(manifest: FederationManifest, *, active_source_ids: set[str]) -> FederationManifest:
    """Drop cross-source joins that reference sources no longer in the federation."""
    kept: list[FederationCrossSourceJoin] = []
    for join in manifest.cross_source_joins:
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        left_sid = manifest.table_namespace.get(left_tbl, "")
        right_sid = manifest.table_namespace.get(right_tbl, "")
        if left_sid in active_source_ids and right_sid in active_source_ids:
            kept.append(join)
    return replace(manifest, cross_source_joins=tuple(kept))


def prune_federation_mappings(
    mappings: FederationMappings, manifest: FederationManifest, *, active_source_ids: set[str]
) -> FederationMappings:
    """Drop mapping members that reference sources no longer in the federation."""
    logical_columns = tuple(
        replace(
            col,
            members=tuple(
                member for member in col.members if _qualified_ref_source_id(member, manifest) in active_source_ids
            ),
        )
        for col in mappings.logical_columns
        if any(_qualified_ref_source_id(member, manifest) in active_source_ids for member in col.members)
    )
    logical_tables: list[LogicalTableMapping] = []
    for table_map in mappings.logical_tables:
        members = tuple(m for m in table_map.members if m.source in active_source_ids)
        if not members:
            continue
        logical_tables.append(replace(table_map, members=members))
    return replace(mappings, logical_columns=logical_columns, logical_tables=tuple(logical_tables))


def _qualified_ref_source_id(qualified: str, manifest: FederationManifest) -> str:
    """Resolve the manifest source id for a ``table.column`` reference."""
    return resolve_federation_qualified_ref(qualified, manifest=manifest).source_id


def recorded_federation_source_ids(federation_dir: str) -> tuple[str, ...]:
    """Return source ids from a stored federation artifact manifest."""
    paths = federation_artifact_paths(federation_dir)
    if not os.path.isfile(paths["artifact_manifest"]):
        return ()
    try:
        with open(paths["artifact_manifest"], encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationConfigError(
            f"cannot read federation artifact manifest at {paths['artifact_manifest']!r}: {exc}"
        ) from exc
    if not isinstance(stored, dict):
        return ()
    members = stored.get("federation_members")
    if not isinstance(members, list):
        return ()
    ids: list[str] = []
    for row in members:
        if isinstance(row, (list, tuple)) and row:
            ids.append(str(row[0]))
    return tuple(ids)


def reconcile_federation_topology(
    manifest: FederationManifest,
    mappings: FederationMappings,
    recorded_source_ids: Sequence[str],
    *,
    federation_dir: str | None = None,
) -> tuple[FederationManifest, FederationMappings, FederationTopologyReport]:
    """Prune dangling federation edges when the manifest source set shrinks."""
    recorded = set(recorded_source_ids)
    declared = {binding.source_id for binding in manifest.sources}
    added = tuple(sorted(declared - recorded))
    removed = tuple(sorted(recorded - declared))
    change = detect_federation_topology_change(recorded_source_ids, manifest)
    plan_templates_invalidated = False
    if change != "none" and federation_dir:
        clear_federation_plan_templates(federation_dir)
        plan_templates_invalidated = True
    if change not in ("remove", "mixed"):
        report = FederationTopologyReport(
            change=change,
            added_source_ids=added,
            removed_source_ids=removed,
            plan_templates_invalidated=plan_templates_invalidated,
        )
        return manifest, mappings, report
    active = {binding.source_id for binding in manifest.sources}
    pruned_manifest = prune_cross_source_joins(manifest, active_source_ids=active)
    pruned_manifest = prune_federation_aliases(pruned_manifest, active_source_ids=active)
    pruned_mappings = prune_federation_mappings(mappings, pruned_manifest, active_source_ids=active)
    report = FederationTopologyReport(
        change=change,
        added_source_ids=added,
        removed_source_ids=removed,
        plan_templates_invalidated=plan_templates_invalidated,
    )
    return pruned_manifest, pruned_mappings, report


def load_federation_migration_map(path: str) -> FederationMigrationMap | None:
    """Read ``federation_migration_map.json`` when present."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationPendingError(f"federation_migration_map.json: cannot read or parse: {exc}") from exc
    if not isinstance(payload, dict):
        raise MigrationPendingError("federation_migration_map.json: root must be a JSON object")
    return parse_federation_migration_map(payload)


def parse_federation_migration_map(payload: Mapping[str, Any]) -> FederationMigrationMap:
    """Parse a federation migration map payload."""
    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise MigrationPendingError("federation_migration_map.json: invalid or missing version")
    action = str(payload.get("action", "") or "").strip().lower()
    if action not in (MIGRATION_MAP_ACTION_REMAP, MIGRATION_MAP_ACTION_DESTRUCTIVE, MIGRATION_MAP_ACTION_ABORT):
        raise MigrationPendingError(f"federation_migration_map.json: unsupported action {action!r}")
    qualified: list[FederationQualifiedRename] = []
    for row in payload.get("qualified_column_renames", []) or []:
        if not isinstance(row, dict):
            raise MigrationPendingError("federation_migration_map.json: qualified_column_renames row must be an object")
        from_ref = str(row.get("from", "") or "").strip()
        to_ref = str(row.get("to", "") or "").strip()
        if from_ref and to_ref:
            qualified.append(FederationQualifiedRename(from_ref=from_ref, to_ref=to_ref))
    namespace: list[tuple[str, str]] = []
    for row in payload.get("namespace_renames", []) or []:
        if not isinstance(row, dict):
            raise MigrationPendingError("federation_migration_map.json: namespace_renames row must be an object")
        from_name = str(row.get("from", "") or "").strip()
        to_name = str(row.get("to", "") or "").strip()
        if from_name and to_name:
            namespace.append((from_name, to_name))
    dropped: list[tuple[str, str]] = []
    for row in payload.get("dropped_cross_source_joins", []) or []:
        if not isinstance(row, dict):
            raise MigrationPendingError(
                "federation_migration_map.json: dropped_cross_source_joins row must be an object"
            )
        left = str(row.get("left", "") or "").strip()
        right = str(row.get("right", "") or "").strip()
        if left and right:
            dropped.append((left, right))
    return FederationMigrationMap(
        version=version,
        action=action,
        qualified_column_renames=tuple(qualified),
        namespace_renames=tuple(namespace),
        dropped_cross_source_joins=tuple(dropped),
    )


def _rewrite_qualified_ref(ref: str, renames: Mapping[str, str]) -> str:
    return renames.get(ref, ref)


def apply_per_source_column_renames(
    manifest: FederationManifest,
    mappings: FederationMappings,
    *,
    source_id: str,
    column_renames: Sequence[tuple[str, str, str]],
) -> tuple[FederationManifest, FederationMappings]:
    """Apply a per-source column rename plan to federation manifest and mappings."""
    rename_map: dict[str, str] = {}
    namespace = dict(manifest.table_namespace)
    for table, from_col, to_col in column_renames:
        sid = namespace.get(table, "")
        if sid != source_id:
            continue
        rename_map[f"{table}.{from_col}"] = f"{table}.{to_col}"
    if not rename_map:
        return manifest, mappings
    return _apply_qualified_renames(manifest, mappings, rename_map)


def _apply_qualified_renames(
    manifest: FederationManifest, mappings: FederationMappings, rename_map: Mapping[str, str]
) -> tuple[FederationManifest, FederationMappings]:
    joins: list[FederationCrossSourceJoin] = []
    for join in manifest.cross_source_joins:
        joins.append(
            replace(
                join,
                left=_rewrite_qualified_ref(join.left, rename_map),
                right=_rewrite_qualified_ref(join.right, rename_map),
            )
        )
    namespace = dict(manifest.table_namespace)
    for old, new in rename_map.items():
        old_table, _ = split_qualified_column(old, manifest=manifest)
        new_table, _ = split_qualified_column(new, manifest=manifest)
        if old_table in namespace and old_table != new_table:
            namespace[new_table] = namespace.pop(old_table)
    logical_columns: list[LogicalColumnMapping] = []
    for col in mappings.logical_columns:
        col_members = tuple(_rewrite_qualified_ref(member, rename_map) for member in col.members)
        logical_columns.append(replace(col, members=col_members))
    logical_tables: list[LogicalTableMapping] = []
    for table_map in mappings.logical_tables:
        table_members: list[LogicalTableMember] = []
        for member in table_map.members:
            columns = {
                logical: rename_map.get(f"{member.table}.{physical}", physical)
                for logical, physical in member.columns.items()
            }
            table_name = member.table
            for old, new in rename_map.items():
                old_table, _ = split_qualified_column(old, manifest=manifest)
                new_table, _ = split_qualified_column(new, manifest=manifest)
                if table_name == old_table:
                    table_name = new_table
            table_members.append(replace(member, table=table_name, columns=columns))
        logical_tables.append(replace(table_map, members=tuple(table_members)))
    return (
        replace(manifest, cross_source_joins=tuple(joins), table_namespace=namespace),
        replace(mappings, logical_columns=tuple(logical_columns), logical_tables=tuple(logical_tables)),
    )


def archive_federation_migration_map_file(
    map_path: str | os.PathLike[str],
    *,
    archive_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Rename ``federation_migration_map.json`` to ``.applied.json`` after successful compose."""
    fed_map_path = Path(map_path)
    if not fed_map_path.is_file():
        return
    ts = datetime.now(timezone.utc).strftime(SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT)
    target_dir = Path(archive_dir) if archive_dir is not None else fed_map_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    applied_fed_map = target_dir / f"{fed_map_path.stem}.applied.json"
    try:
        if applied_fed_map.is_file():
            archive = applied_fed_map.with_name(applied_fed_map.stem + f".{ts}" + applied_fed_map.suffix)
            applied_fed_map.rename(archive)
        fed_map_path.rename(applied_fed_map)
    except OSError as exc:
        raise FederationConfigError(f"could not archive federation migration map {fed_map_path!r}: {exc}") from exc


def clear_federation_plan_templates(federation_dir: str) -> None:
    """Remove stored federation plan templates."""
    path = federation_artifact_paths(federation_dir)["plan_templates"]
    if os.path.isfile(path):
        os.remove(path)


def clear_federation_composite_template_store(federation_dir: str) -> bool:
    """Remove the composite partitioned template store under a federation tree."""
    store_dir = os.path.join(federation_dir, TEMPLATE_STORE_SEGMENT)
    legacy = os.path.join(federation_dir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
    existed = os.path.isdir(store_dir) or os.path.isfile(legacy)
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    wipe_filenames(federation_dir, (TEMPLATE_STORE_LEGACY_SINGLE_FILE,))
    return existed


def purge_federation_member_artifacts(
    federation_dir: str,
    *,
    artifacts_root: str | None,
    source_id: str,
    member_engine: Any | None = None,
    manifest: FederationManifest | None = None,
) -> None:
    """Delete on-disk artifacts for one removed federation member and clear composite template shards."""
    binding: FederationSourceBinding | None = None
    if manifest is not None:
        binding = next((row for row in manifest.sources if row.source_id == source_id), None)
    if binding is None and member_engine is not None:
        try:
            binding = binding_from_member_engine(source_id, member_engine)
        except FederationConfigError:
            binding = None
    if binding is None:
        binding = FederationSourceBinding(
            source_id=source_id,
            engine=str(getattr(member_engine, "dialect", "duckdb") or "duckdb") if member_engine else "duckdb",
            connection=source_id,
        )
    with artifact_lock(federation_dir):
        member_dir = federation_source_artifacts_dir(artifacts_root, binding)
        if os.path.isdir(member_dir):
            shutil.rmtree(member_dir, ignore_errors=True)
        extra_dir = getattr(member_engine, "_artifacts_dir", None) if member_engine is not None else None
        if (
            extra_dir
            and os.path.abspath(str(extra_dir)) != os.path.abspath(member_dir)
            and os.path.isdir(str(extra_dir))
        ):
            shutil.rmtree(str(extra_dir), ignore_errors=True)
        clear_federation_composite_template_store(federation_dir)


def _qualified_column_on_member_graphs(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    qual: str,
) -> bool:
    try:
        table_name, column_name = split_qualified_column(qual, manifest=manifest)
    except (FederationDeclarationError, ValueError):
        return False
    source_id = manifest.table_namespace.get(table_name, "")
    if not source_id:
        return False
    graph = member_graphs.get(source_id)
    if graph is None:
        return False
    table = graph.tables.get(table_name)
    if table is None:
        return False
    return column_name in table.columns


def validate_federation_migration_map(
    migration_map: FederationMigrationMap,
    *,
    cached_member_graphs: Mapping[str, SchemaGraph] | None,
    live_member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
) -> None:
    """Validate a federation migration map against cached and live member graphs."""
    stale: list[str] = []
    problems: list[str] = []
    if cached_member_graphs is not None:
        for row in migration_map.qualified_column_renames:
            if not _qualified_column_on_member_graphs(cached_member_graphs, manifest, row.from_ref):
                stale.append(f"qualified rename source {row.from_ref!r} not in cached member graphs")
        for left, right in migration_map.dropped_cross_source_joins:
            if not any(_dropped_cross_source_join_matches(join, left, right) for join in manifest.cross_source_joins):
                stale.append(f"dropped_cross_source_joins entry ({left!r}, {right!r}) not in cached manifest")
    elif migration_map.action != MIGRATION_MAP_ACTION_ABORT and (
        migration_map.qualified_column_renames or migration_map.dropped_cross_source_joins
    ):
        problems.append("cached federation member graphs missing; cannot validate migration map sources")
    for row in migration_map.qualified_column_renames:
        if not _qualified_column_on_member_graphs(live_member_graphs, manifest, row.to_ref):
            problems.append(f"qualified rename target {row.to_ref!r} not in live member graphs")
    if stale and not problems:
        raise MigrationPendingError("STALE_MAP: " + "; ".join(stale))
    if stale and problems:
        problems.extend(stale)
    if problems:
        raise MigrationPendingError("federation_migration_map.json validation failed: " + "; ".join(problems))


def detect_broken_cross_source_joins(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
) -> tuple[tuple[str, str], ...]:
    """Return declared cross-source join endpoint pairs with missing member columns."""
    broken: list[tuple[str, str]] = []
    for join in manifest.cross_source_joins:
        left_ok = _qualified_column_on_member_graphs(member_graphs, manifest, join.left)
        right_ok = _qualified_column_on_member_graphs(member_graphs, manifest, join.right)
        if not left_ok or not right_ok:
            broken.append((join.left, join.right))
    return tuple(broken)


def apply_federation_migration_map(
    migration_map: FederationMigrationMap,
    manifest: FederationManifest,
    mappings: FederationMappings,
    federation_dir: str,
) -> tuple[FederationManifest, FederationMappings]:
    """Apply a federation migration map to manifest/mapping sidecars."""
    if migration_map.action == MIGRATION_MAP_ACTION_ABORT:
        raise MigrationPendingError("user aborted via federation migration map")
    if migration_map.action == MIGRATION_MAP_ACTION_DESTRUCTIVE:
        clear_federation_plan_templates(federation_dir)
        clear_federation_composite_template_store(federation_dir)
        return manifest, mappings
    by_source: dict[str, list[tuple[str, str, str]]] = {}
    fallback_map: dict[str, str] = {}
    for row in migration_map.qualified_column_renames:
        from_ref = resolve_federation_qualified_ref(row.from_ref, manifest=manifest)
        to_ref = resolve_federation_qualified_ref(row.to_ref, manifest=manifest)
        from_table, from_col = split_qualified_column(from_ref.qualified, manifest=manifest)
        _, to_col = split_qualified_column(to_ref.qualified, manifest=manifest)
        sid = from_ref.source_id or manifest.table_namespace.get(from_table, "")
        if sid:
            by_source.setdefault(sid, []).append((from_table, from_col, to_col))
        else:
            fallback_map[from_ref.qualified] = to_ref.qualified
    for source_id, renames in by_source.items():
        manifest, mappings = apply_per_source_column_renames(
            manifest, mappings, source_id=source_id, column_renames=renames
        )
    if fallback_map:
        manifest, mappings = _apply_qualified_renames(manifest, mappings, fallback_map)
    if migration_map.namespace_renames:
        namespace = dict(manifest.table_namespace)
        table_renames: dict[str, str] = {}
        for old_name, new_name in migration_map.namespace_renames:
            if old_name in namespace:
                namespace[new_name] = namespace.pop(old_name)
                table_renames[old_name] = new_name
        manifest = replace(manifest, table_namespace=namespace)
        if table_renames:
            logical_tables: list[LogicalTableMapping] = []
            for table_map in mappings.logical_tables:
                members: list[LogicalTableMember] = []
                for member in table_map.members:
                    table_name = table_renames.get(member.table, member.table)
                    members.append(replace(member, table=table_name))
                logical_tables.append(replace(table_map, members=tuple(members)))
            mappings = replace(mappings, logical_tables=tuple(logical_tables))
    if migration_map.dropped_cross_source_joins:
        kept = tuple(
            join
            for join in manifest.cross_source_joins
            if not any(
                _dropped_cross_source_join_matches(join, drop_left, drop_right)
                for drop_left, drop_right in migration_map.dropped_cross_source_joins
            )
        )
        manifest = replace(manifest, cross_source_joins=kept)
    clear_federation_plan_templates(federation_dir)
    write_federation_mappings_applied_sidecar(federation_dir, mappings)
    return manifest, mappings


def federation_member_roster_rows(
    member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest
) -> tuple[tuple[str, str, str, str], ...]:
    """Pinned member roster: source id, connection name, storage slug, schema_graph_id."""
    rows: list[tuple[str, str, str, str]] = []
    for binding in sorted(manifest.sources, key=lambda s: s.source_id):
        graph = member_graphs.get(binding.source_id)
        if graph is None:
            continue
        rows.append(
            (
                binding.source_id,
                str(binding.connection or ""),
                federation_source_storage_slug(binding),
                str(graph.schema_graph_id or ""),
            )
        )
    return tuple(rows)


def federation_member_hash_tuple(
    member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    """Canonical member identity tuple for composite drift detection."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    source_ids = [binding.source_id for binding in manifest.sources] if manifest.sources else sorted(member_graphs)
    for source_id in sorted(source_ids):
        graph = member_graphs.get(source_id)
        if graph is None:
            continue
        eff = graph.effective_structural_hash or effective_structural_hash_fp(graph.structural_hash, graph.scope_hash)
        rows.append(
            (
                str(source_id),
                str(graph.schema_graph_id or ""),
                eff,
                str(graph.profiling_hash or ""),
                str(graph.notes_sha256 or graph.notes_hash or ""),
                compute_metadata_hash(graph),
            )
        )
    return tuple(rows)


def federation_member_tuple_hash(member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest) -> str:
    """Stable hash of the canonical member identity tuple."""
    blob = json.dumps(federation_member_hash_tuple(member_graphs, manifest), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def federation_plan_topology_identity(
    member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest
) -> tuple[str, str]:
    """Return ``(manifest_hash, member_tuple_hash)`` for plan template matching."""
    return manifest_hash(manifest), federation_member_tuple_hash(member_graphs, manifest)


def clear_federated_turn_state(session: Any | None) -> None:
    """Clear per-turn federated prepare/execute state on *session*."""
    if session is None:
        return
    pending = getattr(session, "_pending_federation_plan_template", None)
    if pending is not None:
        session._pending_federation_plan_template = None


def check_federation_member_drift_at_turn_start(owner: Any, *, manifest: FederationManifest | None = None) -> None:
    """Raise when live member graph identities drift from the pinned federation roster."""
    if not getattr(owner, "_is_aether_federation", False):
        return
    fed_manifest = manifest or getattr(owner, "_federation_manifest", None)
    if not isinstance(fed_manifest, FederationManifest):
        return
    member_graphs = getattr(owner, "_federation_member_graphs", None)
    if not isinstance(member_graphs, dict) or not member_graphs:
        return
    fed_dir = getattr(owner, "_federation_storage_dir", None)
    if not fed_dir:
        return
    fed_mappings = getattr(owner, "_federation_mappings", None)
    if not isinstance(fed_mappings, FederationMappings):
        fed_mappings = FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    if mappings_replay_matches(fed_dir, member_graphs, fed_manifest, fed_mappings):
        return
    raise FederationInvariantError(
        "federation member graphs changed since the composite was prepared; re-ask the question"
    )


def validate_federated_sub_intent(sub_intent: RuntimeIntent, member_schema: SchemaGraph) -> str | None:
    """Validate a decomposed sub-intent against its member schema slice."""
    _, schema_errors = check_qualified_refs_exist(sub_intent, member_schema)
    if schema_errors:
        return "; ".join(schema_errors)
    validation = validate_semantics(sub_intent, member_schema, post_binding=True)
    if not validation.is_valid:
        errors = [issue.message for issue in validation.issues if issue.severity == "error"]
        return errors[0] if errors else "sub-intent failed semantic validation against member slice"
    return None


def write_federation_mappings_applied_sidecar(federation_dir: str, mappings: FederationMappings) -> None:
    """Persist the applied federation mappings sidecar under *federation_dir*."""
    paths = federation_artifact_paths(federation_dir)
    payload = {
        "version": mappings.version,
        "logical_columns": [
            {
                "logical": c.logical,
                "members": list(c.members),
                "role": c.role,
                "unify_in_graph": c.unify_in_graph,
            }
            for c in mappings.logical_columns
        ],
        "logical_tables": [
            {
                "logical": t.logical,
                "semantics": t.semantics,
                "members": [{"source": m.source, "table": m.table, "columns": dict(m.columns)} for m in t.members],
            }
            for t in mappings.logical_tables
        ],
    }
    _write_federation_json_atomic(paths["mappings_applied"], payload)


def _write_federation_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON artifact under *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or None, prefix=".fed_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _normalize_stored_member_hash_row(row: Sequence[Any]) -> tuple[str, str, str, str, str, str]:
    """Normalize one stored federation member hash row; raise on corrupt rows."""
    if not isinstance(row, (list, tuple)) or len(row) < 3:
        raise FederationConfigError(f"corrupt federation member hash row: expected at least 3 fields, got {row!r}")
    source_id = str(row[0])
    schema_graph_id = str(row[1])
    eff = str(row[2])
    profiling = str(row[3]) if len(row) >= 4 else ""
    notes = str(row[4]) if len(row) >= 5 else ""
    metadata = str(row[5]) if len(row) >= 6 else ""
    return (source_id, schema_graph_id, eff, profiling, notes, metadata)


def mappings_hash(mappings: FederationMappings) -> str:
    """Stable hash of a mapping sidecar for replay short-circuit."""
    return hashlib.sha256(
        json.dumps(federation_mappings_document(mappings), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cross_source_join_hash_entry(join: FederationCrossSourceJoin) -> dict[str, str]:
    """Canonical join dict for stable manifest hashing."""
    left, right = join.left, join.right
    if join.kind == "inner" and left > right:
        left, right = right, left
    return {
        "left": left,
        "right": right,
        "kind": join.kind,
        "logical_key": join.logical_key,
    }


def _dropped_cross_source_join_matches(join: FederationCrossSourceJoin, drop_left: str, drop_right: str) -> bool:
    """Return True when a migration-map drop entry targets *join*."""
    drop_probe = FederationCrossSourceJoin(
        left=drop_left, right=drop_right, kind=join.kind, logical_key=join.logical_key
    )
    return _cross_source_join_hash_entry(drop_probe) == _cross_source_join_hash_entry(join)


def manifest_hash(manifest: FederationManifest) -> str:
    """Stable hash of a normalized manifest for composite identity."""
    payload = {
        "federation_id": manifest.federation_id,
        "aliases": [
            {"alias": alias.alias, "source": alias.source, "table": alias.table}
            for alias in sorted(manifest.aliases, key=lambda a: (a.alias, a.source, a.table))
        ],
        "cross_source_joins": [
            _cross_source_join_hash_entry(j)
            for j in sorted(
                manifest.cross_source_joins,
                key=lambda j: (*_cross_source_join_hash_entry(j).values(),),
            )
        ],
        "coordinator": {
            "row_cap": manifest.coordinator.row_cap,
            "default_source_row_cap": manifest.coordinator.default_source_row_cap,
            "default_source_timeout_ms": manifest.coordinator.default_source_timeout_ms,
            "coordinator_timeout_ms": manifest.coordinator.coordinator_timeout_ms,
            "plan_timeout_ms": manifest.coordinator.plan_timeout_ms,
            "semijoin_key_cap": manifest.coordinator.semijoin_key_cap,
            "spill_row_threshold": manifest.coordinator.spill_row_threshold,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def archive_federation_mappings_file(path: str) -> str:
    """Archive a federation mappings editor file to ``applied_federation_mappings.json``."""
    if not path.endswith(".json"):
        raise FederationConfigError(f"expected JSON editor file: {path!r}")
    directory = os.path.dirname(path) or "."
    archive = os.path.join(directory, FEDERATION_MAPPINGS_APPLIED_FILENAME)
    with open(path, encoding="utf-8") as src:
        content = src.read()
    with open(archive, "w", encoding="utf-8") as dst:
        dst.write(content)
    return archive


def archive_federation_editor_file(path: str) -> str:
    """Archive an editor JSON file to ``*.applied.json`` and return archive path."""
    if not path.endswith(".json"):
        raise FederationConfigError(f"expected JSON editor file: {path!r}")
    archive = path.replace(".json", SCHEMA_OVERRIDES_APPLIED_SUFFIX)
    with open(path, encoding="utf-8") as src:
        content = src.read()
    with open(archive, "w", encoding="utf-8") as dst:
        dst.write(content)
    return archive


def export_federation_composite_overrides(composite: SchemaGraph, target: str | os.PathLike[str]) -> Path:
    """Write composite schema overrides for review beside the federation composite graph."""
    return dump_schema_overrides_to_path(composite, target)


def _merge_override_document_with_sidecar(
    document: dict[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar_internal = sidecar.get("_internal", {}) or {}
    document_internal = document.setdefault("_internal", {})
    if not isinstance(document_internal, dict):
        document_internal = {}
        document["_internal"] = document_internal
    for block_key in ("fk_block_inferred", "pk_block_inferred"):
        existing_entries = list(sidecar_internal.get(block_key, []) or [])
        incoming_entries = list(document_internal.get(block_key, []) or [])
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for entry in existing_entries + incoming_entries:
            if not isinstance(entry, dict):
                continue
            if block_key == "pk_block_inferred":
                entry_key = (str(entry.get("table", "")), str(entry.get("column", "")))
            else:
                entry_key = (
                    (str(entry.get("from")) if not isinstance(entry.get("from"), list) else "|".join(entry["from"])),
                    (str(entry.get("to")) if not isinstance(entry.get("to"), list) else "|".join(entry["to"])),
                )
            if entry_key in seen:
                continue
            seen.add(entry_key)
            combined.append(entry)
        document_internal[block_key] = combined

    merged_pk: list[dict[str, Any]] = []
    seen_pk: set[tuple[str, str]] = set()
    for entry in list(sidecar.get("primary_keys_add", []) or []) + list(document.get("primary_keys_add", []) or []):
        if not isinstance(entry, dict):
            continue
        tnm = str(entry.get("table", ""))
        cnm = str(entry.get("column", ""))
        if not tnm or not cnm:
            continue
        pk_pair = (tnm, cnm)
        if pk_pair in seen_pk:
            continue
        seen_pk.add(pk_pair)
        merged_pk.append({"table": tnm, "column": cnm})
    document["primary_keys_add"] = merged_pk
    return document


def _refresh_federation_artifact_manifest_hashes(federation_dir: str, composite: SchemaGraph) -> None:
    paths = federation_artifact_paths(federation_dir)
    stored = _load_federation_artifact_manifest_dict(paths["artifact_manifest"])
    if stored is None:
        return
    semantic_edges_hash = federation_composite_semantic_edges_hash(composite)
    composite.semantic_edges_hash = semantic_edges_hash
    stored.update(
        {
            "schema_graph_id": str(composite.schema_graph_id or ""),
            "structural_hash": str(composite.structural_hash or ""),
            "profiling_hash": str(composite.profiling_hash or ""),
            "scope_hash": str(composite.scope_hash or ""),
            "effective_structural_hash": str(composite.effective_structural_hash or ""),
            "notes_hash": str(composite.notes_hash or ""),
            "semantic_edges_hash": semantic_edges_hash,
            "ddl_probe_hash": str(composite.ddl_probe_hash or ""),
            "schema_revision": int(getattr(composite, "schema_revision", 0) or 0),
        }
    )
    _write_federation_json_atomic(paths["artifact_manifest"], stored)


def _persist_federation_composite_schema_cache(federation_dir: str, composite: SchemaGraph) -> None:
    composite_path = federation_artifact_paths(federation_dir)["composite_schema"]
    with artifact_lock(federation_dir):
        write_gzip_json_atomic(composite_path, composite.to_dict(), sort_keys=True)
    _refresh_federation_artifact_manifest_hashes(federation_dir, composite)


def apply_federation_composite_overrides(
    composite: SchemaGraph,
    federation_dir: str,
    overrides_path: str | os.PathLike[str],
    *,
    dialect: Any | None = None,
) -> OverrideReport:
    """Apply an overrides editor file to the composite graph and persist replay state."""
    composite_path = federation_artifact_paths(federation_dir)["composite_schema"]
    document = load_schema_overrides_file(overrides_path)
    sidecar = load_overrides_sidecar(composite_path) or {}
    document = _merge_override_document_with_sidecar(document, sidecar)
    report = apply_schema_overrides_to_graph(composite, document, dialect=dialect, strict=True)
    document["foreign_keys_add"] = user_added_fks_dump(composite)
    document["primary_keys_add"] = user_added_pks_dump(composite)
    save_overrides_sidecar(
        composite_path,
        document,
        source_schema_hash=str(composite.effective_structural_hash or ""),
        metadata_hash=compute_metadata_hash(composite),
    )
    if (
        report.table_edits
        or report.column_edits
        or report.fks_added
        or report.fks_endorsed
        or report.fks_removed
        or report.pks_added
        or report.pks_endorsed
        or report.pks_blocked
        or report.coerced_columns
        or report.collapsed_inferences
        or report.descriptions_refined
    ):
        _persist_federation_composite_schema_cache(federation_dir, composite)
    return report


def finalize_federation_composite_overrides(
    composite: SchemaGraph,
    federation_dir: str,
    *,
    dialect: Any | None = None,
) -> bool:
    """Replay persisted composite overrides after composition or recomposition."""
    composite_path = federation_artifact_paths(federation_dir)["composite_schema"]
    changed = finalize_with_overrides(composite, composite_path, dialect=dialect)
    if changed:
        _persist_federation_composite_schema_cache(federation_dir, composite)
    return changed


def federation_artifact_paths(federation_dir: str) -> dict[str, str]:
    """Resolve standard federation artifact paths under *federation_dir*."""
    return {
        "manifest": os.path.join(federation_dir, FEDERATION_MANIFEST_FILENAME),
        "mappings": os.path.join(federation_dir, FEDERATION_MAPPINGS_FILENAME),
        "mappings_applied": os.path.join(federation_dir, FEDERATION_MAPPINGS_APPLIED_FILENAME),
        "composite_schema": os.path.join(federation_dir, FEDERATION_COMPOSITE_SCHEMA_FILENAME),
        "artifact_manifest": os.path.join(federation_dir, ARTIFACT_MANIFEST_FILENAME),
        "plan_templates": os.path.join(federation_dir, FEDERATION_TEMPLATES_SEGMENT, FEDERATION_PLAN_TEMPLATE_FILENAME),
        "mapping_suggestions_cache": os.path.join(federation_dir, FEDERATION_MAPPING_SUGGESTIONS_CACHE_FILENAME),
    }


def compute_federation_storage_dir(artifacts_root: str | None, federation_id: str) -> str:
    """Return the absolute federation artifact directory ``fed_<federation_id>``."""
    parent = (
        os.path.abspath(os.path.expanduser(str(artifacts_root)))
        if artifacts_root and str(artifacts_root).strip()
        else user_data_dir(appname="aetherdialect", appauthor=False)
    )
    safe_id = str(federation_id).strip()
    if not safe_id:
        raise FederationConfigError("federation_id must be non-empty")
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, f"{FEDERATION_STORAGE_PREFIX}{safe_id}")


def source_ids_for_intent(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    mappings: FederationMappings | None = None,
    manifest: FederationManifest | None = None,
) -> frozenset[str]:
    """Return the set of member source ids referenced by *intent* tables."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    source_by_table = _table_source_index(schema, mappings, manifest)
    tables = set(intent.tables or [])
    if manifest is not None:
        return frozenset(_intent_table_sources(tables, manifest, mappings, source_by_table))
    sources = {_source_for_table(table, source_by_table) for table in tables}
    sources.discard("")
    return frozenset(sources)


def _federation_artifact_format_version_from_manifest(stored: Mapping[str, Any]) -> int | None:
    stored_fmt = stored.get("artifact_format_version")
    try:
        return int(stored_fmt) if stored_fmt is not None else None
    except (TypeError, ValueError):
        return None


def _raise_federation_artifact_format_version_mismatch(
    stored: Mapping[str, Any],
    manifest_path: str,
    federation_dir: str,
) -> None:
    stored_fmt = stored.get("artifact_format_version")
    found_fmt = _federation_artifact_format_version_from_manifest(stored)
    if found_fmt != FEDERATION_ARTIFACT_FORMAT_VERSION:
        raise FederationConfigError(
            f"federation artifact manifest at {manifest_path!r} has "
            f"artifact_format_version {stored_fmt!r}; this build expects "
            f"{FEDERATION_ARTIFACT_FORMAT_VERSION}. Delete the federation artifact "
            f"directory {federation_dir!r} and re-run federation initialization "
            f"so the tree is rebuilt from scratch."
        )


def _load_federation_artifact_manifest_dict(manifest_path: str) -> dict[str, Any] | None:
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationConfigError(f"federation artifact manifest at {manifest_path!r} is unreadable: {exc}") from exc
    if not isinstance(stored, dict):
        raise FederationConfigError(f"federation artifact manifest at {manifest_path!r} is not a JSON object")
    return stored


def mappings_replay_matches(
    federation_dir: str,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
) -> bool:
    """
    Return True when stored federation artifact hashes match the live composite.

    Returns:
        ``True`` when the artifact manifest exists and its member / mapping /
        manifest hashes match the live inputs. ``False`` when the manifest is
        absent, unreadable, or the hashes diverge (caller rebuilds or treats as
        drift).

    Raises:

        FederationConfigError: When the artifact manifest exists but its
        ``artifact_format_version`` is not
        :data:`FEDERATION_ARTIFACT_FORMAT_VERSION`. Delete the federation
        artifact directory and re-run federation initialization so the tree
        is rebuilt; there is no migration path.
    """
    paths = federation_artifact_paths(federation_dir)
    manifest_path = paths["artifact_manifest"]
    stored = _load_federation_artifact_manifest_dict(manifest_path)
    if stored is None:
        return False
    _raise_federation_artifact_format_version_mismatch(stored, manifest_path, federation_dir)
    live_members = federation_member_hash_tuple(member_graphs, manifest)
    stored_members = stored.get("federation_members")
    if not isinstance(stored_members, list):
        return False
    stored_tuple = tuple(_normalize_stored_member_hash_row(entry) for entry in stored_members)
    if stored_tuple != live_members:
        return False
    stored_mappings_hash = str(stored.get("mappings_hash", "") or "")
    stored_manifest_hash = str(stored.get("manifest_hash", "") or "")
    if stored_mappings_hash != mappings_hash(mappings) or stored_manifest_hash != manifest_hash(manifest):
        return False
    return True


def federation_artifact_manifest_view(federation_dir: str) -> ArtifactManifest | None:
    """Build a migration-tier view of the federation composite artifact manifest."""
    stored_raw = _load_federation_artifact_manifest_dict(federation_artifact_paths(federation_dir)["artifact_manifest"])
    if stored_raw is None:
        return None
    return ArtifactManifest(
        artifact_format_version=0,
        structural_hash=str(stored_raw.get("structural_hash", "") or ""),
        profiling_hash=str(stored_raw.get("profiling_hash", "") or ""),
        scope_hash=str(stored_raw.get("scope_hash", "") or ""),
        effective_structural_hash=str(stored_raw.get("effective_structural_hash", "") or ""),
        schema_graph_id=str(stored_raw.get("schema_graph_id", "") or ""),
        notes_hash=str(stored_raw.get("notes_hash", "") or ""),
        semantic_edges_hash=str(stored_raw.get("semantic_edges_hash", "") or ""),
    )


def federation_composite_migration_tier(
    federation_dir: str,
    composite: SchemaGraph,
    *,
    previous_composite: SchemaGraph | None = None,
) -> MigrationTier:
    """Classify composite drift for a federation tree without format- version false positives."""
    stored = federation_artifact_manifest_view(federation_dir)
    return classify_migration_tier(stored, composite, previous_schema=previous_composite)


def persist_federation_tree(
    federation_dir: str,
    *,
    manifest: FederationManifest,
    mappings: FederationMappings,
    composite: SchemaGraph,
    member_graphs: Mapping[str, SchemaGraph],
    manifest_editor_path: str | None = None,
    mappings_editor_path: str | None = None,
) -> None:
    """Write federation manifest, mappings, composite graph, and artifact manifest."""
    os.makedirs(federation_dir, exist_ok=True)
    paths = federation_artifact_paths(federation_dir)
    manifest_payload = federation_manifest_document(manifest)
    mappings_payload = {
        "version": mappings.version,
        "logical_columns": [
            {
                "logical": c.logical,
                "members": list(c.members),
                "role": c.role,
                "unify_in_graph": c.unify_in_graph,
            }
            for c in mappings.logical_columns
        ],
        "logical_tables": [
            {
                "logical": t.logical,
                "semantics": t.semantics,
                "members": [{"source": m.source, "table": m.table, "columns": dict(m.columns)} for m in t.members],
                **(
                    {"authoritative_source": t.authoritative_source}
                    if t.semantics == "replica" and t.authoritative_source
                    else {}
                ),
            }
            for t in mappings.logical_tables
        ],
    }
    members = federation_member_hash_tuple(member_graphs, manifest)
    roster = federation_member_roster_rows(member_graphs, manifest)
    semantic_edges_hash = federation_composite_semantic_edges_hash(composite)
    composite.semantic_edges_hash = semantic_edges_hash
    artifact_payload = {
        "artifact_format_version": FEDERATION_ARTIFACT_FORMAT_VERSION,
        "federation_id": manifest.federation_id,
        "manifest_hash": manifest_hash(manifest),
        "mappings_hash": mappings_hash(mappings),
        "federation_member_roster": [list(row) for row in roster],
        "federation_members": [list(row) for row in members],
        "schema_graph_id": str(composite.schema_graph_id or ""),
        "structural_hash": str(composite.structural_hash or ""),
        "profiling_hash": str(composite.profiling_hash or ""),
        "scope_hash": str(composite.scope_hash or ""),
        "effective_structural_hash": str(composite.effective_structural_hash or ""),
        "notes_hash": str(composite.notes_hash or ""),
        "semantic_edges_hash": semantic_edges_hash,
        "ddl_probe_hash": str(composite.ddl_probe_hash or ""),
        "schema_revision": int(getattr(composite, "schema_revision", 0) or 0),
    }
    with artifact_lock(federation_dir):
        _write_federation_json_atomic(paths["manifest"], manifest_payload)
        _write_federation_json_atomic(paths["mappings"], mappings_payload)
        write_gzip_json_atomic(paths["composite_schema"], composite.to_dict(), sort_keys=True)
        _write_federation_json_atomic(paths["artifact_manifest"], artifact_payload)
    if manifest_editor_path and os.path.isfile(manifest_editor_path):
        archive_federation_editor_file(manifest_editor_path)
    if mappings_editor_path and os.path.isfile(mappings_editor_path):
        archive_federation_mappings_file(mappings_editor_path)
    os.makedirs(os.path.dirname(paths["plan_templates"]), exist_ok=True)


def load_federation_composite_graph(federation_dir: str) -> SchemaGraph | None:
    """
    Load ``composite_schema_graph.json.gz`` from a federation tree.

    Returns:
        The composite ``SchemaGraph``, or ``None`` when the composite file is
        absent, unreadable, or its fingerprints disagree with the artifact
        manifest (non-version failures).

    Raises:

        FederationConfigError: When the artifact manifest exists but its
        ``artifact_format_version`` is not
        :data:`FEDERATION_ARTIFACT_FORMAT_VERSION`. Delete the federation
        artifact directory and re-run federation initialization so the tree
        is rebuilt; there is no migration path.
    """
    path = federation_artifact_paths(federation_dir)["composite_schema"]
    manifest_path = federation_artifact_paths(federation_dir)["artifact_manifest"]
    stored_manifest = _load_federation_artifact_manifest_dict(manifest_path)
    if stored_manifest is not None:
        _raise_federation_artifact_format_version_mismatch(stored_manifest, manifest_path, federation_dir)
    if not os.path.isfile(path):
        return None
    try:
        payload = read_gzip_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if stored_manifest is not None:
            raise FederationConfigError(f"federation composite schema at {path!r} is unreadable: {exc}") from exc
        return None
    if not isinstance(payload, dict):
        if stored_manifest is not None:
            raise FederationConfigError(f"federation composite schema at {path!r} is not a JSON object")
        return None
    graph = SchemaGraph.from_dict(payload)
    if stored_manifest is None:
        return graph
    stored = stored_manifest
    stored_id = str(stored.get("schema_graph_id", "") or "")
    stored_eff = str(stored.get("effective_structural_hash", "") or "")
    if stored_id and stored_id != str(graph.schema_graph_id or ""):
        return None
    if stored_eff and stored_eff != str(graph.effective_structural_hash or ""):
        return None
    stored_semantic = str(stored.get("semantic_edges_hash", "") or "")
    if stored_semantic:
        live_semantic = federation_composite_semantic_edges_hash(graph)
        if stored_semantic != live_semantic:
            return None
    stored_probe = str(stored.get("ddl_probe_hash", "") or "")
    if stored_probe and stored_probe != str(graph.ddl_probe_hash or ""):
        return None
    stored_revision = stored.get("schema_revision")
    if stored_revision is not None:
        try:
            if int(stored_revision) != int(getattr(graph, "schema_revision", 0) or 0):
                return None
        except (TypeError, ValueError) as exc:
            raise FederationConfigError(
                f"federation composite schema revision in artifact manifest at {manifest_path!r} is invalid: {exc}"
            ) from exc
    return graph


def _engine_type_from_federation_source_slug(slug: str) -> str:
    text = str(slug or "").strip()
    if not text.startswith(FEDERATION_SOURCE_STORAGE_PREFIX):
        return "duckdb"
    rest = text[len(FEDERATION_SOURCE_STORAGE_PREFIX) :]
    engine, _, _ = rest.partition("_")
    return engine or "duckdb"


def _binding_from_persisted_roster_row(row: Sequence[str]) -> FederationSourceBinding:
    if len(row) < 4:
        raise FederationConfigError("federation roster row must have four fields")
    source_id = str(row[0] or "").strip()
    connection = str(row[1] or "").strip()
    storage_slug = str(row[2] or "").strip()
    if not source_id:
        raise FederationConfigError("federation roster row requires source_id")
    if not storage_slug:
        raise FederationConfigError(f"federation roster row for {source_id!r} requires storage slug")
    return FederationSourceBinding(
        source_id=source_id,
        engine=_engine_type_from_federation_source_slug(storage_slug),
        connection=connection or source_id,
        role="owner",
    )


def load_persisted_federation_roster_rows(federation_dir: str) -> tuple[tuple[str, str, str, str], ...]:
    """Load pinned roster rows from a federation artifact manifest."""
    paths = federation_artifact_paths(federation_dir)
    stored = _load_federation_artifact_manifest_dict(paths["artifact_manifest"])
    if stored is None:
        raise FederationConfigError(f"federation artifact tree not found at {federation_dir!r}")
    _raise_federation_artifact_format_version_mismatch(stored, paths["artifact_manifest"], federation_dir)
    raw = stored.get("federation_member_roster")
    if not isinstance(raw, list) or not raw:
        raise FederationConfigError(
            f"federation roster is missing from artifact manifest at {paths['artifact_manifest']!r}"
        )
    rows: list[tuple[str, str, str, str]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            raise FederationConfigError("federation_member_roster entries must be four-field rows")
        rows.append(
            (
                str(entry[0] or ""),
                str(entry[1] or ""),
                str(entry[2] or ""),
                str(entry[3] or ""),
            )
        )
    return tuple(rows)


def inspect_persisted_federation(
    artifacts_dir: str,
    federation_id: str,
) -> PersistedFederationInspection:
    """Load declaration and roster from a persisted ``fed_<id>`` tree. Does not construct member engines or open database connections."""
    fed_id = str(federation_id or "").strip()
    if not fed_id:
        raise FederationConfigError("federation_id must be non-empty")
    fed_dir = compute_federation_storage_dir(artifacts_dir, fed_id)
    paths = federation_artifact_paths(fed_dir)
    if not os.path.isfile(paths["manifest"]):
        raise FederationConfigError(f"federation manifest not found at {paths['manifest']!r}")
    with open(paths["manifest"], encoding="utf-8") as handle:
        stored_manifest = json.load(handle)
    if not isinstance(stored_manifest, dict):
        raise FederationConfigError(f"federation manifest at {paths['manifest']!r} is not a JSON object")
    stored_fed_id = str(stored_manifest.get("federation_id", "") or "").strip()
    if stored_fed_id and stored_fed_id != fed_id:
        raise FederationConfigError(
            f"federation_id {fed_id!r} disagrees with stored manifest federation_id {stored_fed_id!r}"
        )
    mappings = (
        load_federation_mappings_from_path(paths["mappings"])
        if os.path.isfile(paths["mappings"])
        else FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    )
    roster_rows = load_persisted_federation_roster_rows(fed_dir)
    composite = load_federation_composite_graph(fed_dir)
    sources = tuple(_binding_from_persisted_roster_row(row) for row in roster_rows)
    namespace = _namespace_from_composite_schema(composite) if composite is not None else {}
    built = replace(
        parse_federation_manifest(stored_manifest),
        sources=sources,
        table_namespace=namespace,
    )
    manifest = hydrate_persisted_federation_manifest(stored_manifest, built)
    return PersistedFederationInspection(
        federation_id=fed_id,
        federation_dir=fed_dir,
        manifest=manifest,
        mappings=mappings,
        roster=roster_rows,
    )


def federation_plan_combine_kind(plan: FederatedPlan) -> str:
    """Return a stable combine kind label for a federated plan."""
    if effective_union_specs(plan):
        return "union"
    if isinstance(plan.combine, tuple) and plan.combine:
        return "join"
    if len(plan.steps) <= 1:
        return "single"
    return "none"


def federation_plan_combine_hash(plan: FederatedPlan) -> str:
    """Stable hash of a federated combine specification."""
    union_specs = effective_union_specs(plan)
    if union_specs:
        union_payload = [
            {
                "logical_table": u.logical_table,
                "member_source_ids": list(u.member_source_ids),
                "semantics": u.semantics,
            }
            for u in union_specs
        ]
    else:
        union_payload = []
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    if join_specs:
        join_payload = {
            "kind": "join",
            "joins": [
                {
                    "left_source": j.left_source,
                    "right_source": j.right_source,
                    "left_key": j.left_key,
                    "right_key": j.right_key,
                    "logical_key": j.logical_key,
                    "kind": j.kind,
                }
                for j in join_specs
            ],
        }
    else:
        join_payload = {"kind": "none"}
    payload = {
        "unions": union_payload,
        "combine": join_payload,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _residual_spec_hash_payload(residual: ResidualSpec | None) -> dict[str, Any]:
    if residual is None:
        return {"kind": "none"}
    where_leaves = _predicate_where_leaves(residual.where)
    having_leaves = _predicate_having_leaves(residual.having)
    return {
        "kind": "residual",
        "select_cols": [sc.expr.column_ref or sc.expr.primary_term for sc in residual.select_cols],
        "group_by_cols": [g.column_ref or g.primary_term for g in residual.group_by_cols],
        "order_by_cols": [o.expr.column_ref or o.expr.primary_term for o in residual.order_by_cols],
        "where": sorted(fp.signature_key for fp in where_leaves),
        "having": sorted(hp.signature_key for hp in having_leaves),
        "distinct_select_index": residual.distinct_select_index,
        "limit": residual.limit,
    }


def federation_plan_residual_hash(plan: FederatedPlan) -> str:
    """Stable hash of a federated residual specification."""
    blob = json.dumps(_residual_spec_hash_payload(plan.residual), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def federation_member_schema_graph_ids(
    plan: FederatedPlan, member_graphs: Mapping[str, SchemaGraph] | None
) -> tuple[tuple[str, str], ...]:
    """Pinned per-member schema graph ids for a federated plan."""
    if not member_graphs:
        return ()
    out: list[tuple[str, str]] = []
    for step in plan.steps:
        graph = member_graphs.get(step.source_id)
        if graph is None:
            continue
        out.append((step.source_id, str(graph.schema_graph_id or "")))
    return tuple(out)


def revalidate_prepared_federation_plan(
    prepared: FederatedPrepareOutcome,
    composite_schema: SchemaGraph,
    *,
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    intent_key_fn: Callable[[RuntimeIntent], str] | None = None,
) -> None:
    """Re-check prepare-time pins before executing a prepared federated plan."""
    if not prepared.success:
        raise FederationInvariantError("cannot execute an unsuccessful federated prepare outcome")
    pinned_composite = str(prepared.composite_schema_graph_id or "")
    current_composite = str(composite_schema.schema_graph_id or "")
    if pinned_composite and pinned_composite != current_composite:
        raise FederationInvariantError("composite schema graph changed since preparation; re-ask the question")
    pinned_combine = str(prepared.combine_hash or "")
    current_combine = federation_plan_combine_hash(prepared.plan)
    if pinned_combine and pinned_combine != current_combine:
        raise FederationInvariantError("federated combine specification changed since preparation; re-ask the question")
    pinned_members = tuple(prepared.member_schema_graph_ids)
    if pinned_members and member_graphs:
        current_members = federation_member_schema_graph_ids(prepared.plan, member_graphs)
        if current_members != pinned_members:
            raise FederationInvariantError("member schema graph changed since preparation; re-ask the question")
    pinned_fps = tuple(prepared.step_fingerprints)
    if pinned_fps:
        key_fn = intent_key_fn or intent_key
        current_fps = federation_plan_step_fingerprints(
            prepared.plan, intent_key_fn=key_fn, manifest=manifest, member_graphs=member_graphs
        )
        if current_fps != pinned_fps:
            raise FederationInvariantError("federated step fingerprints changed since preparation; re-ask the question")


def federation_plan_step_fingerprints(
    plan: FederatedPlan,
    *,
    intent_key_fn: Callable[[RuntimeIntent], str],
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Ordered per-source sub-intent fingerprints for template matching."""
    out: list[tuple[str, str]] = []
    for step in plan.steps:
        grain = step.sub_intent.grain or "row_level"
        if grain not in VALID_GRAINS:
            grain = "row_level"
        params_fp = stable_json(flatten_param_values(step.sub_intent))
        proj_fp = stable_json({"projected_keys": list(step.projected_keys)})
        limits_fp = ""
        if manifest is not None:
            resolved = resolve_member_limits_for_source(manifest, step.source_id)
            limits_fp = stable_json(
                {
                    "row_cap": resolved.row_cap,
                    "timeout_ms": resolved.timeout_ms,
                    "max_query_cost_rows": resolved.max_query_cost_rows,
                    "max_query_cost_bytes": resolved.max_query_cost_bytes,
                    "profile_timeout_ms": resolved.profile_timeout_ms,
                }
            )
        schema_graph_id = ""
        if member_graphs is not None:
            member_graph = member_graphs.get(step.source_id)
            if member_graph is not None:
                schema_graph_id = str(member_graph.schema_graph_id or "")
        join_fp = join_path_segments_fingerprint_runtime(step.sub_intent)
        out.append(
            (
                step.source_id,
                (
                    f"{intent_key_fn(step.sub_intent)}:{grain}:{params_fp}:{proj_fp}:{limits_fp}:"
                    f"{schema_graph_id}:{join_fp}"
                ),
            )
        )
    return tuple(out)


def _federation_topology_hashes_compatible(
    template: FederationPlanTemplate,
    *,
    manifest_hash_value: str = "",
    member_tuple_hash_value: str = "",
) -> bool:
    """Return whether stored and live federation topology hashes are both present and equal."""
    tmpl_manifest = str(template.manifest_hash or "")
    tmpl_member = str(template.member_tuple_hash or "")
    live_manifest = str(manifest_hash_value or "")
    live_member = str(member_tuple_hash_value or "")
    if tmpl_manifest or tmpl_member or live_manifest or live_member:
        if not tmpl_manifest or not tmpl_member or not live_manifest or not live_member:
            return False
        if tmpl_manifest != live_manifest or tmpl_member != live_member:
            return False
    return True


def federation_plan_matches_template(
    plan: FederatedPlan,
    template: FederationPlanTemplate,
    *,
    step_fingerprints: Sequence[tuple[str, str]],
    manifest_hash_value: str = "",
    member_tuple_hash_value: str = "",
) -> bool:
    """Return whether *plan* matches a stored federation plan template."""
    if federation_plan_combine_hash(plan) != template.combine_hash:
        return False
    expected_residual = template.residual_hash or federation_plan_residual_hash(FederatedPlan(steps=(), residual=None))
    if federation_plan_residual_hash(plan) != expected_residual:
        return False
    if tuple(template.step_fingerprints) != tuple(step_fingerprints):
        return False
    if not _federation_topology_hashes_compatible(
        template,
        manifest_hash_value=manifest_hash_value,
        member_tuple_hash_value=member_tuple_hash_value,
    ):
        return False
    return True


def lookup_federation_plan_template(
    federation_dir: str,
    composite_schema_graph_id: str,
    intent_k: str,
    *,
    manifest_hash_value: str = "",
    member_tuple_hash_value: str = "",
) -> FederationPlanTemplate | None:
    """Load a federation plan template for *intent_k* when the composite id matches."""
    templates = load_federation_plan_templates(federation_dir)
    template = templates.get(intent_k)
    if template is None:
        return None
    if template.composite_schema_graph_id != composite_schema_graph_id:
        return None
    if not _federation_topology_hashes_compatible(
        template,
        manifest_hash_value=manifest_hash_value,
        member_tuple_hash_value=member_tuple_hash_value,
    ):
        return None
    return template


def _bound_federation_plan_accepted_questions(accepted: Sequence[str]) -> tuple[str, ...]:
    items = [str(x) for x in accepted if str(x).strip()]
    cap = int(FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP)
    if len(items) <= cap:
        return tuple(items)
    return tuple(items[-cap:])


def _enforce_federation_plan_template_file_cap(
    existing: dict[str, Any], *, keep_plan_id: str | None = None
) -> dict[str, Any]:
    cap = int(FEDERATION_PLAN_TEMPLATE_FILE_CAP)
    if len(existing) <= cap:
        return existing
    for plan_id in list(existing.keys()):
        if len(existing) <= cap:
            break
        if plan_id == keep_plan_id:
            continue
        row = existing.get(plan_id)
        accepted_raw = row.get("accepted_questions", []) if isinstance(row, dict) else []
        if isinstance(accepted_raw, list) and any(str(x).strip() for x in accepted_raw):
            continue
        existing.pop(plan_id, None)
    while len(existing) > cap:
        removed = False
        for plan_id in list(existing.keys()):
            if plan_id == keep_plan_id:
                continue
            existing.pop(plan_id, None)
            removed = True
            break
        if not removed:
            break
    return existing


def delete_federation_plan_template(federation_dir: str, plan_id: str, *, schema_role: SchemaRole = "owner") -> None:
    """Remove one federation plan template record when it has no accepted questions."""
    if schema_role != "owner":
        raise OwnerOnlyOperationError("delete_federation_plan_template")
    pid = str(plan_id or "").strip()
    if not federation_dir or not pid:
        return
    paths = federation_artifact_paths(federation_dir)
    path = paths["plan_templates"]
    if not os.path.isfile(path):
        return
    with artifact_lock(federation_dir):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise FederationConfigError(f"corrupt federation plan templates file: {path!r}: {exc}") from exc
        if not isinstance(loaded, dict) or pid not in loaded:
            return
        del loaded[pid]
        if loaded:
            _write_federation_json_atomic(path, loaded)
        else:
            os.remove(path)


def delete_unaccepted_federation_plan_template(
    federation_dir: str, plan_id: str, *, schema_role: SchemaRole = "owner"
) -> None:
    """Drop a plan record that was never credited with an accepted question."""
    if not federation_dir or not plan_id:
        return
    template = load_federation_plan_templates(federation_dir).get(str(plan_id))
    if template is None or template.accepted_questions:
        return
    delete_federation_plan_template(federation_dir, str(plan_id), schema_role=schema_role)


def save_federation_plan_template(
    federation_dir: str, template: FederationPlanTemplate, *, schema_role: SchemaRole = "owner"
) -> None:
    """Append or replace a federation plan template in the federation tree."""
    if schema_role != "owner":
        raise OwnerOnlyOperationError("save_federation_plan_template")
    paths = federation_artifact_paths(federation_dir)
    os.makedirs(os.path.dirname(paths["plan_templates"]), exist_ok=True)
    existing: dict[str, Any] = {}
    if os.path.isfile(paths["plan_templates"]):
        try:
            with open(paths["plan_templates"], encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError) as exc:
            raise FederationConfigError(
                f"corrupt federation plan templates file: {paths['plan_templates']!r}: {exc}"
            ) from exc
    bounded_accepted = _bound_federation_plan_accepted_questions(template.accepted_questions)
    existing[template.plan_id] = {
        "format_version": int(template.format_version),
        "composite_schema_graph_id": template.composite_schema_graph_id,
        "intent_key": template.intent_key,
        "step_fingerprints": [list(row) for row in template.step_fingerprints],
        "combine_hash": template.combine_hash,
        "question": template.question,
        "accepted_questions": list(bounded_accepted),
        "member_template_ids": [list(row) for row in template.member_template_ids],
        "residual_hash": template.residual_hash,
        "join_feedback": list(template.join_feedback),
        "manifest_hash": template.manifest_hash,
        "member_tuple_hash": template.member_tuple_hash,
    }
    existing = _enforce_federation_plan_template_file_cap(existing, keep_plan_id=template.plan_id)
    with artifact_lock(federation_dir):
        _write_federation_json_atomic(paths["plan_templates"], existing)


def load_federation_plan_templates(federation_dir: str) -> dict[str, FederationPlanTemplate]:
    """Load federation plan templates keyed by plan id."""
    path = federation_artifact_paths(federation_dir)["plan_templates"]
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationConfigError(f"corrupt federation plan templates file: {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FederationConfigError(f"federation plan templates file at {path!r} is not a JSON object")
    out: dict[str, FederationPlanTemplate] = {}
    for plan_id, row in payload.items():
        if not isinstance(row, dict):
            continue
        steps_raw = row.get("step_fingerprints", [])
        steps: list[tuple[str, str]] = []
        if isinstance(steps_raw, list):
            for entry in steps_raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    steps.append((str(entry[0]), str(entry[1])))
        accepted_raw = row.get("accepted_questions", [])
        accepted: tuple[str, ...] = ()
        if isinstance(accepted_raw, list):
            accepted = tuple(str(x) for x in accepted_raw if str(x).strip())
        member_ids_raw = row.get("member_template_ids", [])
        member_ids: list[tuple[str, str]] = []
        if isinstance(member_ids_raw, list):
            for entry in member_ids_raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    member_ids.append((str(entry[0]), str(entry[1])))
        out[str(plan_id)] = FederationPlanTemplate(
            plan_id=str(plan_id),
            composite_schema_graph_id=str(row.get("composite_schema_graph_id", "") or ""),
            intent_key=str(row.get("intent_key", "") or ""),
            step_fingerprints=tuple(steps),
            combine_hash=str(row.get("combine_hash", "") or ""),
            question=str(row.get("question", "") or ""),
            accepted_questions=accepted,
            format_version=int(
                row.get("format_version", FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION)
                or FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION
            ),
            member_template_ids=tuple(member_ids),
            residual_hash=str(row.get("residual_hash", "") or ""),
            join_feedback=tuple(str(x) for x in (row.get("join_feedback", []) or []) if str(x).strip()),
            manifest_hash=str(row.get("manifest_hash", "") or ""),
            member_tuple_hash=str(row.get("member_tuple_hash", "") or ""),
        )
    return out


def credit_federation_plan_accept(
    federation_dir: str,
    plan_id: str,
    q_norm: str,
    *,
    member_template_ids: Sequence[tuple[str, str]] | None = None,
    schema_role: SchemaRole = "owner",
    pending_plan_template: FederationPlanTemplate | None = None,
) -> None:
    """Record that *q_norm* accepted the federation plan *plan_id*."""
    if schema_role != "owner":
        raise OwnerOnlyOperationError("credit_federation_plan_accept")
    if not federation_dir or not plan_id or not q_norm:
        return
    templates = load_federation_plan_templates(federation_dir)
    template = templates.get(plan_id)
    if template is None:
        if pending_plan_template is None:
            return
        template = pending_plan_template
    if q_norm in template.accepted_questions and not member_template_ids:
        return
    accepted = template.accepted_questions
    if q_norm not in accepted:
        accepted = (*accepted, q_norm)
    accepted = _bound_federation_plan_accepted_questions(accepted)
    updated_member_ids = tuple(member_template_ids) if member_template_ids else template.member_template_ids
    updated = FederationPlanTemplate(
        plan_id=template.plan_id,
        composite_schema_graph_id=template.composite_schema_graph_id,
        intent_key=template.intent_key,
        step_fingerprints=template.step_fingerprints,
        combine_hash=template.combine_hash,
        question=template.question or q_norm,
        accepted_questions=accepted,
        member_template_ids=updated_member_ids,
        residual_hash=template.residual_hash,
        join_feedback=template.join_feedback,
        format_version=template.format_version,
        manifest_hash=template.manifest_hash,
        member_tuple_hash=template.member_tuple_hash,
    )
    save_federation_plan_template(federation_dir, updated, schema_role=schema_role)


def record_federation_join_feedback(federation_dir: str, plan_id: str, summary: str) -> None:
    """Persist cross-source join rejection feedback on the federation plan record."""
    text = str(summary or "").strip()
    if not federation_dir or not plan_id or not text:
        return
    templates = load_federation_plan_templates(federation_dir)
    template = templates.get(plan_id)
    if template is None:
        return
    if text in template.join_feedback:
        return
    updated = FederationPlanTemplate(
        plan_id=template.plan_id,
        composite_schema_graph_id=template.composite_schema_graph_id,
        intent_key=template.intent_key,
        step_fingerprints=template.step_fingerprints,
        combine_hash=template.combine_hash,
        question=template.question,
        accepted_questions=template.accepted_questions,
        member_template_ids=template.member_template_ids,
        residual_hash=template.residual_hash,
        join_feedback=(*template.join_feedback, text),
        format_version=template.format_version,
        manifest_hash=template.manifest_hash,
        member_tuple_hash=template.member_tuple_hash,
    )
    save_federation_plan_template(federation_dir, updated)


def lookup_federation_join_feedback(federation_dir: str, plan_id: str) -> list[str]:
    """Return cross-source join feedback stored on a federation plan record."""
    if not federation_dir or not plan_id:
        return []
    template = load_federation_plan_templates(federation_dir).get(plan_id)
    if template is None:
        return []
    return [str(x) for x in template.join_feedback if str(x).strip()]


def _parse_coordinator_config(raw: Mapping[str, Any]) -> FederationCoordinatorConfig:
    defaults = FederationCoordinatorConfig()
    row_cap = int(raw.get("row_cap", defaults.row_cap) or defaults.row_cap)
    default_source_row_cap = int(raw.get("default_source_row_cap", row_cap) or row_cap)
    default_source_timeout_ms = int(
        raw.get("default_source_timeout_ms", defaults.default_source_timeout_ms) or defaults.default_source_timeout_ms
    )
    coordinator_timeout_ms = int(
        raw.get("coordinator_timeout_ms", defaults.coordinator_timeout_ms) or defaults.coordinator_timeout_ms
    )
    plan_timeout_ms = int(raw.get("plan_timeout_ms", defaults.plan_timeout_ms) or defaults.plan_timeout_ms)
    semijoin_key_cap = int(raw.get("semijoin_key_cap", defaults.semijoin_key_cap) or defaults.semijoin_key_cap)
    semijoin_key_distinct_floor = int(
        raw.get("semijoin_key_distinct_floor", defaults.semijoin_key_distinct_floor)
        or defaults.semijoin_key_distinct_floor
    )
    spill_row_threshold = int(
        raw.get("spill_row_threshold", defaults.spill_row_threshold) or defaults.spill_row_threshold
    )
    max_parallel_members = int(
        raw.get("max_parallel_members", defaults.max_parallel_members) or defaults.max_parallel_members
    )
    total_input_byte_cap = int(
        raw.get("total_input_byte_cap", defaults.total_input_byte_cap) or defaults.total_input_byte_cap
    )
    return FederationCoordinatorConfig(
        row_cap=row_cap,
        default_source_row_cap=default_source_row_cap,
        default_source_timeout_ms=default_source_timeout_ms,
        coordinator_timeout_ms=coordinator_timeout_ms,
        plan_timeout_ms=plan_timeout_ms,
        semijoin_key_cap=semijoin_key_cap,
        semijoin_key_distinct_floor=semijoin_key_distinct_floor,
        spill_row_threshold=spill_row_threshold,
        max_parallel_members=max_parallel_members,
        total_input_byte_cap=total_input_byte_cap,
    )


_SENSITIVITY_RANK = FEDERATION_SENSITIVITY_RANK


def _strictest_sensitivity(*values: SensitivityClassification) -> SensitivityClassification:
    return max(values, key=lambda value: _SENSITIVITY_RANK.get(value.value, 0))


def _merge_column_metadata_strictest(candidates: Sequence[ColumnMetadata]) -> ColumnMetadata:
    if not candidates:
        raise ValueError("merge_column_metadata_strictest requires at least one column")
    merged = copy.deepcopy(candidates[0])
    for other in candidates[1:]:
        merged.sensitivity = _strictest_sensitivity(merged.sensitivity, other.sensitivity)
        if other.usable_override is False:
            merged.usable_override = False
        elif merged.usable_override is not False and not other.is_usable:
            merged.usable_override = False
        if other.role and merged.role and other.role != merged.role:
            merged.role = None
        elif other.role and not merged.role:
            merged.role = other.role
    merged_desc, merged_owner = resolve_descriptions(*((col.description, col.description_owner) for col in candidates))
    if merged_desc:
        set_description(merged, merged_desc, merged_owner or DescriptionOwner.CATALOG)
    else:
        merged.description = ""
        merged.description_owner = None
    return merged


def _assert_replica_column_data_types_agree(candidates: Sequence[ColumnMetadata], label: str) -> None:
    """Raise when replica members disagree on column data_type."""
    data_types = {str(col.data_type or "").strip().lower() for col in candidates if str(col.data_type or "").strip()}
    if len(data_types) > 1:
        raise FederationConfigError(f"{label}: replica members disagree on data_type: {sorted(data_types)!r}")


def _merge_column_metadata_union_statistics(
    candidates: Sequence[ColumnMetadata],
    *,
    composite_semantics: Literal["union", "replica", "logical_unify"] = "logical_unify",
    member_sources: Sequence[str] | None = None,
    authoritative_source: str = "",
) -> ColumnMetadata:
    """Merge column metadata and profiling statistics across *candidates*."""
    merged = _merge_column_metadata_strictest(candidates)
    if len(candidates) <= 1:
        return merged
    if composite_semantics == "replica":
        auth_source = (authoritative_source or "").strip()
        auth_col: ColumnMetadata | None = None
        if auth_source and member_sources and len(member_sources) == len(candidates):
            for source, col in zip(member_sources, candidates, strict=True):
                if source == auth_source:
                    auth_col = col
                    break
        if auth_col is None:
            auth_col = candidates[0]
        merged.frequent_values = list(auth_col.frequent_values or [])
        merged.value_overlap_sample = list(auth_col.value_overlap_sample or [])
        merged.distinct_count = int(auth_col.distinct_count or 0)
        merged.distinct_ratio = float(auth_col.distinct_ratio or 0.0)
        merged.null_ratio = float(auth_col.null_ratio or 0.0)
        merged.mode_frequency_ratio = float(auth_col.mode_frequency_ratio or 0.0)
        merged.min_val = auth_col.min_val
        merged.max_val = auth_col.max_val
        return merged
    if composite_semantics == "union":
        ordered_candidates = sorted(
            candidates,
            key=lambda col: (
                str(getattr(col, "name", "") or "").lower(),
                str(col.data_type or "").lower(),
                str(col.value_type or "").lower(),
            ),
        )
        frequent: list[str] = []
        for col in ordered_candidates:
            for value in col.frequent_values or []:
                token = str(value)
                if token and token not in frequent:
                    frequent.append(token)
        merged.frequent_values = frequent
        merged.value_overlap_sample = []
        merged.distinct_count = 0
        merged.min_val = None
        merged.max_val = None
        merged.distinct_ratio = 0.0
        merged.null_ratio = 0.0
        merged.mode_frequency_ratio = 0.0
        return merged
    ordered_candidates = sorted(
        candidates,
        key=lambda col: (
            str(getattr(col, "name", "") or "").lower(),
            str(col.data_type or "").lower(),
            str(col.value_type or "").lower(),
        ),
    )
    frequent: list[str] = []
    overlap: list[str] = []
    min_vals: list[str] = []
    max_vals: list[str] = []
    for col in ordered_candidates:
        for value in col.frequent_values or []:
            token = str(value)
            if token and token not in frequent:
                frequent.append(token)
        for value in col.value_overlap_sample or []:
            token = str(value)
            if token and token not in overlap:
                overlap.append(token)
        if col.min_val not in (None, ""):
            min_vals.append(str(col.min_val))
        if col.max_val not in (None, ""):
            max_vals.append(str(col.max_val))
    merged.frequent_values = frequent
    merged.value_overlap_sample = sorted(overlap, key=str.lower)
    merged.distinct_count = 0
    if min_vals:
        merged.min_val = min(min_vals)
    if max_vals:
        merged.max_val = max(max_vals)
    merged.distinct_ratio = 0.0
    merged.null_ratio = 0.0
    merged.mode_frequency_ratio = 0.0
    return merged


def _sanitize_foreign_keys(tables: dict[str, TableMetadata]) -> None:
    """Drop within-source foreign keys whose endpoints are absent after collapse."""
    for tbl in tables.values():
        valid: list[FKEdge] = []
        for edge in tbl.foreign_keys:
            if edge.inference_tag == InferenceTag.CROSS_SOURCE:
                valid.append(edge)
                continue
            if edge.dst_table not in tables:
                continue
            dst = tables[edge.dst_table]
            if any(col not in dst.columns for col in edge.dst_cols):
                continue
            if any(col not in tbl.columns for col in edge.src_cols):
                continue
            valid.append(edge)
        tbl.foreign_keys = valid


def _remap_fk_endpoints(tables: dict[str, TableMetadata], table_remap: Mapping[str, str]) -> None:
    if not table_remap:
        return
    for tbl in tables.values():
        updated: list[FKEdge] = []
        for edge in tbl.foreign_keys:
            dst = table_remap.get(edge.dst_table, edge.dst_table)
            src = table_remap.get(edge.src_table, edge.src_table)
            updated.append(replace(edge, src_table=src if src in tables else edge.src_table, dst_table=dst))
        tbl.foreign_keys = updated


def _partition_signature(table: TableMetadata) -> tuple[Any, ...]:
    return (
        tuple(table.partition_columns or []),
        table.partition_type,
        bool(table.require_partition_filter),
        tuple(table.clustering_fields or []),
        table.clustering_key,
    )


def _assert_partition_metadata_agrees(member_tables: Sequence[TableMetadata], logical: str) -> None:
    signatures = {_partition_signature(table) for table in member_tables}
    if len(signatures) > 1:
        raise FederationDeclarationError(
            f"logical table {logical!r} members disagree on partition or clustering metadata"
        )


def _member_logical_primary_key(member: LogicalTableMember, table: TableMetadata) -> list[str]:
    """Map a member table primary key to logical column names declared in *member*."""
    phys_to_logical = {physical: logical for logical, physical in member.columns.items() if physical}
    return [phys_to_logical.get(pk_col, pk_col) for pk_col in (table.primary_key or [])]


def _agreed_primary_key(keys: Sequence[list[str]], *, logical: str = "") -> list[str]:
    normalized = [list(pk) for pk in keys if pk]
    if not normalized:
        return []
    first = normalized[0]
    if all(pk == first for pk in normalized[1:]):
        return list(first)
    label = f"logical table {logical!r} " if logical else ""
    raise FederationDeclarationError(f"{label}members disagree on primary key")


def _merge_collapsed_foreign_keys(
    member_tables: Sequence[TableMetadata], logical_name: str, table_remap: Mapping[str, str]
) -> list[FKEdge]:
    merged: list[FKEdge] = []
    seen: set[tuple[Any, ...]] = set()
    for member_table in member_tables:
        for edge in member_table.foreign_keys:
            dst = table_remap.get(edge.dst_table, edge.dst_table)
            key = (logical_name, tuple(edge.src_cols), dst, tuple(edge.dst_cols), edge.inference_tag)
            if key in seen:
                continue
            seen.add(key)
            merged.append(replace(edge, src_table=logical_name, dst_table=dst))
    return merged


def _derive_composite_created_at(member_graphs: Mapping[str, SchemaGraph]) -> str:
    stamps = sorted(
        str(graph.created_at).strip() for graph in member_graphs.values() if str(graph.created_at or "").strip()
    )
    return stamps[-1] if stamps else ""


def _merge_member_scope_deny(
    member_graphs: Mapping[str, SchemaGraph], composite_names: Mapping[tuple[str, str], str]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    deny_columns: dict[str, set[str]] = {}
    disallowed_columns: dict[str, set[str]] = {}
    for source_id in sorted(member_graphs):
        graph = member_graphs[source_id]
        for tbl_name, cols in (graph.deny_columns or {}).items():
            composite = composite_names.get((source_id, tbl_name), tbl_name)
            deny_columns.setdefault(composite, set()).update(cols)
        for tbl_name, cols in (graph.disallowed_columns or {}).items():
            composite = composite_names.get((source_id, tbl_name), tbl_name)
            disallowed_columns.setdefault(composite, set()).update(cols)
    return deny_columns, disallowed_columns


def member_allow_tables_for_source(
    manifest: FederationManifest, mappings: FederationMappings | None, source_id: str
) -> frozenset[str]:
    """Physical table names a federation member may expose (namespace owner + union members)."""
    tables = {table_name for table_name, owner in manifest.table_namespace.items() if owner == source_id}
    suffix = f"_{source_id}"
    for table_name in list(tables):
        if table_name.endswith(suffix):
            base = table_name[: -len(suffix)]
            if base:
                tables.add(base)
    for alias in manifest.aliases:
        if alias.source == source_id and alias.table:
            tables.add(alias.table)
    if mappings is not None:
        for entry in mappings.logical_tables:
            for member in entry.members:
                if member.source == source_id and member.table:
                    tables.add(member.table)
    return frozenset(tables)


def _filter_member_graphs_to_allow_scope(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
) -> dict[str, SchemaGraph]:
    """Restrict each member graph to declaration-owned tables plus union partition members."""
    if not manifest.table_namespace and not mappings.logical_tables and not manifest.aliases:
        return dict(member_graphs)
    filtered: dict[str, SchemaGraph] = {}
    for source_id in sorted(member_graphs):
        graph = member_graphs[source_id]
        allowed = member_allow_tables_for_source(manifest, mappings, source_id)
        if not allowed:
            filtered[source_id] = graph
            continue
        tables = {name: table for name, table in graph.tables.items() if name in allowed}
        if set(tables) == set(graph.tables):
            filtered[source_id] = graph
            continue
        filtered[source_id] = SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=graph.schema_graph_id,
            effective_structural_hash=graph.effective_structural_hash,
            structural_hash=graph.structural_hash,
            scope_hash=graph.scope_hash,
            schema_stats=graph.schema_stats,
            enum_values=dict(graph.enum_values or {}),
            deny_columns={tbl: set(cols) for tbl, cols in (graph.deny_columns or {}).items() if tbl in tables},
            disallowed_columns={
                tbl: set(cols) for tbl, cols in (graph.disallowed_columns or {}).items() if tbl in tables
            },
            created_at=graph.created_at,
            notes_sha256=graph.notes_sha256,
            profiling_hash=graph.profiling_hash,
            ddl_probe_hash=graph.ddl_probe_hash,
            schema_revision=graph.schema_revision,
            scope_descriptor=graph.scope_descriptor,
        )
    return filtered


@dataclass(frozen=True)
class MemberEffectiveGrants:
    """Tables and columns a federation member role may read at composition time."""

    tables: frozenset[str]
    columns: frozenset[tuple[str, str]] | None = None


def coerce_member_effective_grants(raw: Any) -> MemberEffectiveGrants | None:
    """Normalize dialect or engine grant introspection payloads."""
    if raw is None:
        return None
    if isinstance(raw, MemberEffectiveGrants):
        return raw
    if isinstance(raw, Mapping):
        tables = frozenset(str(name) for name in (raw.get("tables") or ()) if str(name).strip())
        columns_raw = raw.get("columns")
        if columns_raw is None:
            return MemberEffectiveGrants(tables=tables)
        columns: set[tuple[str, str]] = set()
        for entry in columns_raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                table_name = str(entry[0]).strip()
                column_name = str(entry[1]).strip()
                if table_name and column_name:
                    columns.add((table_name, column_name))
                continue
            text = str(entry).strip()
            if "." in text:
                table_name, column_name = text.rsplit(".", 1)
                if table_name and column_name:
                    columns.add((table_name, column_name))
        return MemberEffectiveGrants(tables=tables, columns=frozenset(columns))
    if isinstance(raw, (frozenset, set, list, tuple)):
        if all(isinstance(entry, str) for entry in raw):
            return MemberEffectiveGrants(tables=frozenset(str(name) for name in raw if str(name).strip()))
    return None


def introspect_member_effective_grants(engine: Any) -> MemberEffectiveGrants | None:
    """Return effective grants from a member engine or dialect hook when available."""
    dialect = getattr(engine, "_dialect", None)
    if dialect is not None:
        hook = getattr(dialect, "introspect_effective_grants", None)
        if callable(hook):
            return coerce_member_effective_grants(hook())
    hook = getattr(engine, "introspect_effective_grants", None)
    if callable(hook):
        return coerce_member_effective_grants(hook())
    return None


def member_effective_grants_from_graph(graph: SchemaGraph) -> MemberEffectiveGrants:
    """Derive effective grants from a profiled member schema graph artifact."""
    desc = graph.scope_descriptor if isinstance(graph.scope_descriptor, dict) else {}
    ctx = schema_context_from_descriptor(desc) if desc else EngineContext()
    tables = set(graph.tables.keys())
    if ctx.allow_objects:
        tables &= set(ctx.allow_objects)
    tables -= set(ctx.deny_objects)
    columns: set[tuple[str, str]] = set()
    for table_name in tables:
        table = graph.tables.get(table_name)
        if table is None:
            continue
        for column_name in table.columns:
            columns.add((table_name, column_name))
    deny_by_table = deny_columns_by_table(graph, ctx)
    for table_name, denied_cols in (graph.deny_columns or {}).items():
        deny_by_table.setdefault(table_name, set()).update(denied_cols)
    for table_name, denied_cols in deny_by_table.items():
        for column_name in denied_cols:
            columns.discard((table_name, column_name))
    for table_name, denied_cols in (graph.disallowed_columns or {}).items():
        if table_name not in tables:
            continue
        for column_name in denied_cols:
            columns.discard((table_name, column_name))
    if ctx.allow_columns:
        allowed_qc = set(ctx.allow_columns)
        bare_allowed = {name for name in allowed_qc if "." not in name}
        columns = {pair for pair in columns if f"{pair[0]}.{pair[1]}" in allowed_qc or pair[1] in bare_allowed}
    return MemberEffectiveGrants(tables=frozenset(tables), columns=frozenset(columns))


def _physical_table_name(
    source_id: str,
    table_ref: str,
    composite_names: Mapping[tuple[str, str], str],
) -> str:
    for (sid, physical), composite in composite_names.items():
        if sid == source_id and composite == table_ref:
            return physical
    return table_ref


def collect_declared_member_objects(
    manifest: FederationManifest,
    mappings: FederationMappings,
    composite_names: Mapping[tuple[str, str], str],
) -> dict[str, set[tuple[str, str | None]]]:
    """Return declared physical ``(table, column|None)`` references per member source."""
    declared: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    mapping_sources = _mapping_member_source_by_table(mappings)
    namespace = manifest.table_namespace or {}
    for composite, source_id in namespace.items():
        declared[source_id].add((_physical_table_name(source_id, composite, composite_names), None))
    for alias in manifest.aliases:
        declared[alias.source].add((alias.table, None))
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            declared[member.source].add((member.table, None))
            for phys_col in member.columns.values():
                if phys_col:
                    declared[member.source].add((member.table, phys_col))
    for col_map in mappings.logical_columns:
        for qual in col_map.members:
            text = str(qual or "").strip()
            three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
            if three:
                source_id = three.group(1)
                table_name = three.group(2)
                column_name = three.group(3)
                phys_table = _physical_table_name(source_id, table_name, composite_names)
                declared[source_id].add((phys_table, column_name))
                continue
            table_name, column_name = split_qualified_column(text, manifest=manifest, source_by_table=mapping_sources)
            source_id = mapping_sources.get(table_name) or namespace.get(table_name, "")
            if not source_id:
                continue
            phys_table = _physical_table_name(source_id, table_name, composite_names)
            declared[source_id].add((phys_table, column_name))
    for join in manifest.cross_source_joins:
        for qual in (join.left, join.right):
            ref = resolve_federation_qualified_ref(qual, manifest=manifest, source_by_table=mapping_sources)
            phys_table = _physical_table_name(ref.source_id, ref.table, composite_names)
            declared[ref.source_id].add((phys_table, ref.column))
    return declared


def _member_effective_grants_include(
    grants: MemberEffectiveGrants,
    table_name: str,
    column_name: str | None,
    *,
    member_graph: SchemaGraph | None,
) -> bool:
    if table_name not in grants.tables:
        return False
    if column_name is None:
        return True
    if grants.columns is None:
        if member_graph is None:
            return True
        table = member_graph.tables.get(table_name)
        return table is not None and column_name in table.columns
    return (table_name, column_name) in grants.columns


def resolve_member_effective_grants(
    source_id: str,
    graph: SchemaGraph,
    *,
    engine: Any | None = None,
    explicit: MemberEffectiveGrants | None = None,
) -> MemberEffectiveGrants:
    """Resolve effective grants for one federation member."""
    if explicit is not None:
        return explicit
    stashed = coerce_member_effective_grants(getattr(graph, "_member_effective_grants", None))
    if stashed is not None:
        return stashed
    if engine is not None:
        introspected = introspect_member_effective_grants(engine)
        if introspected is not None:
            return introspected
    return member_effective_grants_from_graph(graph)


def validate_declared_objects_against_member_grants(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
    composite_names: Mapping[tuple[str, str], str],
    *,
    member_engines: Mapping[str, Any] | None = None,
    member_effective_grants: Mapping[str, MemberEffectiveGrants] | None = None,
) -> None:
    """Reject declared federation objects that exceed a member's effective grants."""
    declared = collect_declared_member_objects(manifest, mappings, composite_names)
    explicit_grants = dict(member_effective_grants or {})
    engines = member_engines or {}
    for source_id in sorted(declared):
        graph = member_graphs.get(source_id)
        if graph is None:
            raise FederationDeclarationError(f"logical_tables member unresolved source: {source_id!r}")
        grants = resolve_member_effective_grants(
            source_id,
            graph,
            engine=engines.get(source_id),
            explicit=explicit_grants.get(source_id),
        )
        for table_name, column_name in sorted(declared[source_id], key=lambda item: (item[0], item[1] or "")):
            if table_name not in graph.tables:
                continue
            if _member_effective_grants_include(grants, table_name, column_name, member_graph=graph):
                continue
            obj = f"{table_name}.{column_name}" if column_name else table_name
            raise FederationDeclarationError(f"federation member {source_id!r} effective grants do not include {obj!r}")


def _mapping_member_source_by_table(mappings: FederationMappings) -> dict[str, str]:
    """Map logical and physical table names to member sources from declared mappings."""
    index: dict[str, str] = {}
    for table_map in mappings.logical_tables:
        if table_map.logical:
            if table_map.authoritative_source:
                index[table_map.logical] = str(table_map.authoritative_source).strip()
            elif table_map.semantics == "replica":
                auth = select_replica_member_source(table_map)
                if auth:
                    index[table_map.logical] = auth
        for member in table_map.members:
            if member.table and member.source:
                index[member.table] = member.source
    return index


def validate_federation_mapping_members(
    member_graphs: Mapping[str, SchemaGraph],
    mappings: FederationMappings,
    composite_names: Mapping[tuple[str, str], str],
    manifest: FederationManifest,
) -> None:
    """Reject mappings whose members do not resolve in the member graphs."""
    mapping_sources = _mapping_member_source_by_table(mappings)
    for col_map in mappings.logical_columns:
        for qual in col_map.members:
            tbl, col = split_qualified_column(qual, manifest=manifest, source_by_table=mapping_sources)
            resolved = False
            for source_id, member_graph in member_graphs.items():
                phys = tbl
                for (sid, physical), composite in composite_names.items():
                    if sid == source_id and composite == tbl:
                        phys = physical
                        break
                if phys not in member_graph.tables:
                    continue
                if col not in member_graph.tables[phys].columns:
                    continue
                resolved = True
                break
            if not resolved:
                raise FederationDeclarationError(f"logical_columns member unresolved: {qual!r}")
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            table_member_graph = member_graphs.get(member.source)
            if table_member_graph is None:
                raise FederationDeclarationError(f"logical_tables member unresolved source: {member.source!r}")
            if member.table not in table_member_graph.tables:
                raise FederationDeclarationError(
                    f"logical_tables member unresolved table: {member.source}.{member.table}"
                )
            src_table = table_member_graph.tables[member.table]
            for _logical_col, phys_col in member.columns.items():
                if phys_col not in src_table.columns:
                    raise FederationDeclarationError(
                        f"logical_tables member unresolved column: {member.source}.{member.table}.{phys_col}"
                    )


def reconcile_composite_classifications(
    composite: SchemaGraph,
    member_graphs: Mapping[str, SchemaGraph],
    mappings: FederationMappings,
    *,
    manifest: FederationManifest | None = None,
    notes_content: str | None = None,
    llm_classify: Callable[..., Any] | None = None,
) -> bool:
    """Reconcile collapsed descriptions and roles deterministically; classify conflicts when needed."""
    notes_stripped = (notes_content or "").strip()
    reconcile_owner = DescriptionOwner.NOTES if notes_stripped else DescriptionOwner.LLM_REFINEMENT
    conflicts: set[str] = set()
    for table_map in mappings.logical_tables:
        table = composite.tables.get(table_map.logical)
        if table is None:
            continue
        descriptions: list[tuple[str, DescriptionOwner | None]] = []
        roles: list[str | None] = []
        for member in table_map.members:
            graph = member_graphs.get(member.source)
            if graph is None:
                continue
            member_table = graph.tables.get(member.table)
            if member_table is not None:
                descriptions.append((member_table.description, member_table.description_owner))
                roles.append(member_table.role)
        agreed_desc, agreed_owner = resolve_descriptions(*descriptions)
        if agreed_desc:
            set_description(table, agreed_desc, agreed_owner or DescriptionOwner.CATALOG)
        else:
            distinct_descriptions = {(desc or "").strip() for desc, _ in descriptions if (desc or "").strip()}
            if len(distinct_descriptions) > 1:
                conflicts.add(table_map.logical)
        agreed_role_values = {role for role in roles if role}
        if len(agreed_role_values) == 1:
            table.role = next(iter(agreed_role_values))
        elif len(agreed_role_values) > 1:
            conflicts.add(table_map.logical)
        for logical_col in table.column_member_sources:
            col = table.columns.get(logical_col)
            if col is None:
                continue
            col_desc_candidates: list[tuple[str, DescriptionOwner | None]] = []
            col_roles: list[str | None] = []
            for member in table_map.members:
                phys_col = member.columns.get(logical_col)
                if not phys_col:
                    continue
                graph = member_graphs.get(member.source)
                if graph is None:
                    continue
                member_table = graph.tables.get(member.table)
                if member_table is None:
                    continue
                member_col = member_table.columns.get(phys_col)
                if member_col is None:
                    continue
                col_desc_candidates.append((member_col.description, member_col.description_owner))
                col_roles.append(member_col.role)
            agreed_col_desc, agreed_col_owner = resolve_descriptions(*col_desc_candidates)
            if agreed_col_desc:
                set_description(col, agreed_col_desc, agreed_col_owner or DescriptionOwner.CATALOG)
            else:
                distinct_col_descriptions = {
                    (desc or "").strip() for desc, _ in col_desc_candidates if (desc or "").strip()
                }
                if len(distinct_col_descriptions) > 1:
                    conflicts.add(f"{table_map.logical}.{logical_col}")
            role_values = {role for role in col_roles if role}
            if len(role_values) == 1:
                col.role = next(iter(role_values))
            elif len(role_values) > 1:
                conflicts.add(f"{table_map.logical}.{logical_col}")
    for col_map in mappings.logical_columns:
        if not col_map.unify_in_graph:
            continue
        roles = []
        descriptions = []
        for qual in col_map.members:
            tbl, col_name = split_qualified_column(qual)
            table = composite.tables.get(tbl)
            if table is None:
                continue
            col = table.columns.get(col_name) or table.columns.get(col_map.logical)
            if col is None:
                continue
            roles.append(col.role)
            descriptions.append(col.description)
        role_values = {role for role in roles if role}
        if len(role_values) > 1:
            for qual in col_map.members:
                tbl, _ = split_qualified_column(qual)
                conflicts.add(f"{tbl}.{col_map.logical}")
    if not conflicts:
        return False
    classify = llm_classify
    if classify is None:
        notes_content = "\n".join(
            part for part in (notes_content or "", FEDERATION_COMPOSITE_RECONCILIATION_NOTE) if str(part or "").strip()
        )
        classify = llm_classify_schema
    try:
        classifications = classify(composite, notes_content)
    except Exception as exc:
        raise FederationRuntimeError(f"composite classification reconciliation failed: {exc}") from exc
    for table_name, (table_role, table_desc, col_classes) in classifications.items():
        table = composite.tables.get(table_name)
        if table is None:
            continue
        if table_name in conflicts and table_desc:
            set_description(table, table_desc, reconcile_owner)
        if table_name in conflicts and table_role:
            table.role = table_role
        for col_name, (col_role, col_desc, _sensitivity) in col_classes.items():
            col = table.columns.get(col_name)
            if col is None:
                continue
            conflict_key = f"{table_name}.{col_name}"
            if conflict_key in conflicts:
                if col_desc:
                    set_description(col, col_desc, reconcile_owner)
                if col_role:
                    col.role = col_role
    return True


def _referenced_columns_for_table(intent: RuntimeIntent, table_name: str) -> set[str]:
    refs: set[str] = set()
    for sc in intent.select_cols or []:
        tables = collect_referenced_tables(
            [sc], [], [], [], [], window_registry=intent.window_registry, case_registry=intent.case_registry
        )
        if table_name not in tables:
            continue
        col_ref = (sc.expr.column_ref or "").strip()
        if "." in col_ref:
            tbl, col = col_ref.rsplit(".", 1)
            if tbl == table_name:
                refs.add(col)
        elif col_ref:
            refs.add(col_ref)
    return refs


def _intent_lacks_column_member_coverage(intent: RuntimeIntent, schema: SchemaGraph) -> bool:
    for table_name in intent.tables or []:
        table = schema.tables.get(table_name)
        if table is None or not table.column_member_sources:
            continue
        needed = _referenced_columns_for_table(intent, table_name)
        covered_cols = [col for col in needed if col in table.column_member_sources]
        if not covered_cols:
            continue
        common: set[str] | None = None
        for col_name in covered_cols:
            holders = set(table.column_member_sources.get(col_name, []))
            common = holders if common is None else common & holders
        if common is not None and not common:
            return True
    return False


def _manifest_source_ids(manifest: FederationManifest, member_graphs: Mapping[str, SchemaGraph]) -> tuple[str, ...]:
    if manifest.sources:
        return tuple(binding.source_id for binding in manifest.sources)
    return tuple(sorted(member_graphs))


def _resolve_composite_table_names(
    member_graphs: Mapping[str, SchemaGraph], manifest: FederationManifest
) -> dict[tuple[str, str], str]:
    """Map each member ``(source_id, physical_table)`` to its composite table name."""
    namespace = manifest.table_namespace or derive_table_namespace(member_graphs)
    alias_by_member = {(alias.source, alias.table): alias.alias for alias in manifest.aliases}
    keys_by_source: dict[str, list[str]] = defaultdict(list)
    for composite, source_id in sorted(namespace.items()):
        keys_by_source[source_id].append(composite)
    names: dict[tuple[str, str], str] = {}
    for source_id in _manifest_source_ids(manifest, member_graphs):
        graph = member_graphs[source_id]
        alias_keys: list[str] = []
        for composite in keys_by_source.get(source_id, []):
            if composite in graph.tables:
                names[(source_id, composite)] = composite
            else:
                alias_keys.append(composite)
        unaliased = sorted(phys for phys in graph.tables if (source_id, phys) not in names)
        if alias_keys and unaliased:
            if len(alias_keys) > len(unaliased):
                raise FederationConfigError(
                    f"federation member {source_id!r} has {len(alias_keys)} unresolved namespace alias(es) "
                    f"but only {len(unaliased)} unaliased physical table(s) to pair"
                )
            for alias, phys in zip(sorted(alias_keys), unaliased, strict=False):
                names[(source_id, phys)] = alias
        for phys_name in graph.tables:
            member_key = (source_id, phys_name)
            if member_key in alias_by_member:
                names[member_key] = alias_by_member[member_key]
            elif member_key not in names:
                names[member_key] = phys_name
    return names


def _collision_resolved_by_logical_tables(members: frozenset[tuple[str, str]], mappings: FederationMappings) -> bool:
    for mapping in mappings.logical_tables:
        if mapping.semantics not in ("union", "replica"):
            continue
        declared = frozenset((member.source, member.table) for member in mapping.members)
        if members <= declared:
            return True
    return False


def _federation_identifier_casing_collision_errors(
    composite_names: Mapping[tuple[str, str], str],
    mappings: FederationMappings,
) -> list[str]:
    """Return declaration errors when members map to composite names that collide only by casing."""
    by_lower: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for (source_id, phys_name), composite in composite_names.items():
        by_lower[composite.lower()].append((source_id, phys_name, composite))
    errors: list[str] = []
    for entries in sorted(by_lower.values(), key=lambda items: (items[0][0], items[0][1])):
        composites = {composite for _, _, composite in entries}
        if len(composites) <= 1:
            continue
        member_set = frozenset((source_id, phys_name) for source_id, phys_name, _ in entries)
        if _collision_resolved_by_logical_tables(member_set, mappings):
            continue
        member_desc = ", ".join(f"{sid}.{phys!r} as {comp!r}" for sid, phys, comp in sorted(entries))
        errors.append(member_desc)
    return errors


def _compose_namespaced_tables(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
    composite_names: Mapping[tuple[str, str], str],
) -> dict[str, TableMetadata]:
    by_composite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (source_id, phys_name), composite in composite_names.items():
        by_composite[composite].append((source_id, phys_name))
    casing_errors = _federation_identifier_casing_collision_errors(composite_names, mappings)
    if casing_errors:
        raise FederationDeclarationError(
            "identifier casing collision across federation members; "
            "resolve with a logical_tables mapping or an explicit alias: " + "; ".join(casing_errors)
        )
    collision_errors: list[str] = []
    for composite, members in sorted(by_composite.items()):
        member_set = frozenset(members)
        if len(member_set) <= 1:
            continue
        if _collision_resolved_by_logical_tables(member_set, mappings):
            continue
        member_desc = ", ".join(f"{sid}.{phys}" for sid, phys in sorted(members))
        collision_errors.append(f"{composite!r}: {member_desc}")
    if collision_errors:
        raise FederationDeclarationError(
            "table name collision across federation members; "
            "resolve with a logical_tables mapping or an explicit alias: " + "; ".join(collision_errors)
        )
    merged: dict[str, TableMetadata] = {}
    multi_source = len(member_graphs) > 1
    for source_id in _manifest_source_ids(manifest, member_graphs):
        graph = member_graphs[source_id]
        for phys_name, table in graph.tables.items():
            final_name = composite_names[(source_id, phys_name)]
            if final_name in merged:
                member_set = frozenset(
                    (sid, phys) for (sid, phys), name in composite_names.items() if name == final_name
                )
                if _collision_resolved_by_logical_tables(member_set, mappings):
                    continue
            stamped = copy.deepcopy(table)
            stamped.name = final_name
            stamped.source_id = source_id
            if multi_source:
                stamped.original_name = ""
                for col_meta in stamped.columns.values():
                    col_meta.original_name = ""
            merged[final_name] = stamped
    return merged


def _union_key_columns_for_mapping(mapping: LogicalTableMapping, member_tables: Sequence[TableMetadata]) -> list[str]:
    member_col_sets = [set(member.columns.keys()) for member in mapping.members]
    common = set.intersection(*member_col_sets) if member_col_sets else set()
    if common:
        return sorted(common)
    cols = sorted({logical for member in mapping.members for logical in member.columns})
    if cols:
        return cols
    primary_keys = [tuple(table.primary_key or ()) for table in member_tables]
    if primary_keys and all(pk == primary_keys[0] for pk in primary_keys) and primary_keys[0]:
        return list(primary_keys[0])
    return []


def _validate_union_member_disjointness(
    mapping: LogicalTableMapping,
    member_tables: Sequence[TableMetadata],
) -> None:
    """Refuse union logical tables when profiled key overlap is detected or cannot be ruled out."""
    if mapping.semantics != "union" or len(member_tables) < 2:
        return
    key_cols = _union_key_columns_for_mapping(mapping, member_tables)
    if not key_cols:
        raise FederationDeclarationError(f"union logical table {mapping.logical!r} has no union key column")
    key_col = key_cols[0]
    member_metas: list[tuple[str, ColumnMetadata]] = []
    profiled = False
    for member, table in zip(mapping.members, member_tables):
        phys_col = member.columns.get(key_col) or key_col
        meta = table.columns.get(phys_col)
        if meta is None:
            raise FederationDeclarationError(
                f"union logical table {mapping.logical!r} missing key column {key_col!r} "
                f"on member {member.source}.{member.table}"
            )
        if int(meta.row_count or 0) > 0 or meta.value_overlap_sample:
            profiled = True
        member_metas.append((member.source, meta))
    if not profiled:
        return
    has_samples = any(bool(meta.value_overlap_sample) for _, meta in member_metas)
    if not has_samples:
        return
    for left_idx, (left_source, left_meta) in enumerate(member_metas):
        left_sample = {str(value) for value in (left_meta.value_overlap_sample or []) if str(value)}
        for right_source, right_meta in member_metas[left_idx + 1 :]:
            right_sample = {str(value) for value in (right_meta.value_overlap_sample or []) if str(value)}
            if left_sample and right_sample and (left_sample & right_sample):
                raise FederationDeclarationError(
                    f"union logical table {mapping.logical!r} members {left_source!r} and {right_source!r} "
                    f"overlap on key {key_col!r}"
                )


def _apply_logical_table_collapse(
    tables: dict[str, TableMetadata],
    logical_tables: Sequence[LogicalTableMapping],
    composite_names: Mapping[tuple[str, str], str],
) -> None:
    for mapping in logical_tables:
        if mapping.semantics not in ("union", "replica"):
            continue
        member_tables: list[TableMetadata] = []
        remove_names: list[str] = []
        for member in mapping.members:
            composite = composite_names.get((member.source, member.table), member.table)
            src_table = tables.get(composite)
            if src_table is None:
                raise FederationDeclarationError(
                    f"logical_tables member unresolved table: {member.source}.{member.table}"
                )
            member_tables.append(src_table)
            if composite != mapping.logical:
                remove_names.append(composite)
        if not member_tables:
            continue
        _assert_partition_metadata_agrees(member_tables, mapping.logical)
        if mapping.semantics == "union":
            _validate_union_member_disjointness(mapping, member_tables)
        column_names = sorted({logical for member in mapping.members for logical in member.columns})
        if not column_names:
            inferred: set[str] = set()
            for member in mapping.members:
                composite = composite_names.get((member.source, member.table), member.table)
                src_table = tables.get(composite)
                if src_table is not None:
                    inferred.update(src_table.columns.keys())
            column_names = sorted(inferred)
        merged_cols: dict[str, ColumnMetadata] = {}
        column_member_sources: dict[str, list[str]] = {}
        member_source_ids = sorted({member.source for member in mapping.members})
        authoritative = mapping.authoritative_source or (member_source_ids[0] if mapping.semantics == "replica" else "")
        for col_name in column_names:
            candidates: list[ColumnMetadata] = []
            sources: list[str] = []
            for member in mapping.members:
                phys_col = member.columns.get(col_name) or col_name
                if not phys_col:
                    continue
                composite = composite_names.get((member.source, member.table), member.table)
                src_table = tables.get(composite)
                if src_table is None:
                    continue
                col_meta = src_table.columns.get(phys_col)
                if col_meta is not None:
                    candidates.append(copy.deepcopy(col_meta))
                    sources.append(member.source)
            if not candidates:
                continue
            if mapping.semantics == "replica":
                _assert_replica_column_data_types_agree(
                    candidates, f"logical table {mapping.logical!r} column {col_name!r}"
                )
            merged_cols[col_name] = _merge_column_metadata_union_statistics(
                candidates,
                composite_semantics=mapping.semantics,
                member_sources=sources,
                authoritative_source=authoritative,
            )
            merged_cols[col_name].name = col_name
            column_member_sources[col_name] = sorted(set(sources))
        if not merged_cols:
            continue
        if mapping.semantics == "union":
            row_count = sum(int(table.row_count or 0) for table in member_tables)
            source_id = ""
        else:
            auth_table = next(
                (
                    tables.get(composite_names.get((member.source, member.table), member.table))
                    for member in mapping.members
                    if member.source == authoritative
                ),
                member_tables[0],
            )
            row_count = int((auth_table or member_tables[0]).row_count or 0)
            source_id = authoritative
        table_remap = {name: mapping.logical for name in remove_names}
        merged_foreign_keys = _merge_collapsed_foreign_keys(member_tables, mapping.logical, table_remap)
        _remap_fk_endpoints(tables, table_remap)
        primary = tables.get(mapping.logical)
        collapsed_desc, collapsed_owner = resolve_descriptions(
            *((table.description, table.description_owner) for table in member_tables)
        )
        metadata_kwargs = {
            "name": mapping.logical,
            "columns": merged_cols,
            "primary_key": _agreed_primary_key(
                [
                    _member_logical_primary_key(member, table)
                    for member, table in zip(mapping.members, member_tables, strict=True)
                ],
                logical=mapping.logical,
            ),
            "foreign_keys": merged_foreign_keys,
            "source_id": source_id,
            "member_source_ids": member_source_ids,
            "column_member_sources": column_member_sources,
            "partition_columns": list(member_tables[0].partition_columns or []),
            "partition_type": member_tables[0].partition_type,
            "require_partition_filter": bool(member_tables[0].require_partition_filter),
            "clustering_fields": list(member_tables[0].clustering_fields or []),
            "clustering_key": member_tables[0].clustering_key,
            "distkey": member_tables[0].distkey,
            "sortkey": list(member_tables[0].sortkey or []),
            "diststyle": member_tables[0].diststyle,
            "indexed_columns": list(member_tables[0].indexed_columns or []),
            "size_mb": member_tables[0].size_mb,
            "encoded": member_tables[0].encoded,
            "quote_decision": member_tables[0].quote_decision,
            "role": member_tables[0].role,
            "row_count": row_count,
            "role_owner": member_tables[0].role_owner,
            "composite_descriptive_ratios": dict(member_tables[0].composite_descriptive_ratios or {}),
            "_user_semantic_neighbors": list(member_tables[0]._user_semantic_neighbors or []),
        }
        if primary is None:
            tables[mapping.logical] = TableMetadata(
                **cast(Any, metadata_kwargs),
                description=collapsed_desc,
                description_owner=collapsed_owner,
            )
        else:
            for key, value in metadata_kwargs.items():
                setattr(primary, key, value)
            if collapsed_desc:
                set_description(primary, collapsed_desc, collapsed_owner or DescriptionOwner.CATALOG)
        for name in remove_names:
            tables.pop(name, None)


def _resolve_mapping_column_ref(
    qualified: str, mappings: FederationMappings, manifest: FederationManifest | None = None
) -> tuple[str, str]:
    """Map a mapping ``table.column`` member onto a collapsed logical table name."""
    tbl_name, col_name = split_qualified_column(qualified, manifest=manifest)
    for table_map in mappings.logical_tables:
        if table_map.semantics not in ("union", "replica"):
            continue
        for member in table_map.members:
            if member.table == tbl_name:
                tbl_name = table_map.logical
                break
    return tbl_name, col_name


def _apply_logical_column_unification(
    tables: dict[str, TableMetadata],
    logical_columns: Sequence[LogicalColumnMapping],
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
) -> None:
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    for mapping in logical_columns:
        if not mapping.unify_in_graph:
            continue
        candidates: list[ColumnMetadata] = []
        for member in mapping.members:
            tbl_name, col_name = _resolve_mapping_column_ref(member, mappings, manifest)
            table = tables.get(tbl_name)
            if table is None:
                continue
            col = table.columns.get(col_name)
            if col is not None:
                candidates.append(col)
        if not candidates:
            continue
        target_col = _merge_column_metadata_union_statistics([copy.deepcopy(col) for col in candidates])
        for member in mapping.members:
            tbl_name, col_name = _resolve_mapping_column_ref(member, mappings, manifest)
            table = tables.get(tbl_name)
            if table is None:
                continue
            if col_name in table.columns and col_name != mapping.logical:
                table.columns.pop(col_name, None)
                if col_name in (table.primary_key or []):
                    table.primary_key = [mapping.logical if pk == col_name else pk for pk in (table.primary_key or [])]
                for other in tables.values():
                    for fk in other.foreign_keys:
                        if fk.dst_table == tbl_name and col_name in fk.dst_cols:
                            fk.dst_cols = [
                                mapping.logical if dst_col == col_name else dst_col for dst_col in fk.dst_cols
                            ]
                        if fk.src_table == tbl_name and col_name in fk.src_cols:
                            fk.src_cols = [
                                mapping.logical if src_col == col_name else src_col for src_col in fk.src_cols
                            ]
            unified_col = copy.deepcopy(target_col)
            unified_col.name = mapping.logical
            table.columns[mapping.logical] = unified_col


def _resolve_composite_join_ref(
    qualified: str, mappings: FederationMappings, manifest: FederationManifest | None = None
) -> tuple[str, str]:
    """Map a manifest join reference onto collapsed logical table/column names."""
    table_name, column_name = split_qualified_column(qualified)
    for table_map in mappings.logical_tables:
        if table_map.semantics not in ("union", "replica"):
            continue
        for member in table_map.members:
            if member.table == table_name:
                table_name = table_map.logical
                break
    for col_map in mappings.logical_columns:
        if qualified in col_map.members:
            return table_name, col_map.logical
    return table_name, column_name


def _materialize_cross_source_edges(manifest: FederationManifest, mappings: FederationMappings) -> list[FKEdge]:
    edges: list[FKEdge] = []
    for join in manifest.cross_source_joins:
        left_tbl, left_col = _resolve_composite_join_ref(join.left, mappings, manifest)
        right_tbl, right_col = _resolve_composite_join_ref(join.right, mappings, manifest)
        edges.append(
            FKEdge(
                src_table=left_tbl,
                src_cols=[left_col],
                dst_table=right_tbl,
                dst_cols=[right_col],
                inference_tag=InferenceTag.CROSS_SOURCE,
                join_kind=join.kind,
            )
        )
    for col_map in mappings.logical_columns:
        if col_map.role != "join_key" or len(col_map.members) < 2:
            continue
        sorted_members = sorted(col_map.members)
        for i, left_member in enumerate(sorted_members):
            left_tbl, left_col = _resolve_composite_join_ref(left_member, mappings, manifest)
            for right_member in sorted_members[i + 1 :]:
                right_tbl, right_col = _resolve_composite_join_ref(right_member, mappings, manifest)
                if (left_tbl, left_col) == (right_tbl, right_col):
                    continue
                src_tbl, src_col, dst_tbl, dst_col = left_tbl, left_col, right_tbl, right_col
                if (dst_tbl, dst_col) < (src_tbl, src_col):
                    src_tbl, src_col, dst_tbl, dst_col = dst_tbl, dst_col, src_tbl, src_col
                edges.append(
                    FKEdge(
                        src_table=src_tbl,
                        src_cols=[src_col],
                        dst_table=dst_tbl,
                        dst_cols=[dst_col],
                        inference_tag=InferenceTag.CROSS_SOURCE,
                    )
                )
    return edges


def _assign_composite_identity(
    composite: SchemaGraph,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings,
    *,
    federation_notes_sha: str = "",
) -> None:
    st = structural_hash_fp(tables_structural_payload(composite.tables))
    composite.structural_hash = st
    manifest_fp = manifest_hash(manifest)
    composite.scope_hash = manifest_fp
    composite.effective_structural_hash = effective_structural_hash_fp(st, manifest_fp)
    composite.semantic_edges_hash = federation_composite_semantic_edges_hash(composite)
    member_blob = json.dumps(federation_member_hash_tuple(member_graphs, manifest), separators=(",", ":"))
    mapping_blob = mappings_hash(mappings)
    notes_blob = json.dumps(
        sorted((sid, str(graph.notes_sha256 or graph.notes_hash or "")) for sid, graph in member_graphs.items()),
        separators=(",", ":"),
    )
    profiling_blob = json.dumps(
        sorted((sid, str(graph.profiling_hash or "")) for sid, graph in member_graphs.items()), separators=(",", ":")
    )
    probe_blob = json.dumps(
        sorted((sid, str(graph.ddl_probe_hash or "")) for sid, graph in member_graphs.items()), separators=(",", ":")
    )
    composite.ddl_probe_hash = hashlib.sha256(probe_blob.encode()).hexdigest()[:32]
    seed = hashlib.sha256(
        f"{manifest_fp}|{member_blob}|{mapping_blob}|{notes_blob}|{profiling_blob}|{probe_blob}|{federation_notes_sha}|{composite.semantic_edges_hash}".encode()
    ).hexdigest()[:32]
    composite.schema_graph_id = mint_schema_graph_id(seed_hex=seed, structural_hash=st)


def _merge_enum_values(member_graphs: Mapping[str, SchemaGraph]) -> dict[str, list[str]] | None:
    merged: dict[str, list[str]] = {}
    for source_id in sorted(member_graphs):
        graph = member_graphs[source_id]
        if not graph.enum_values:
            continue
        for key, values in sorted(graph.enum_values.items()):
            storage_key = _member_enum_storage_key(source_id, key)
            merged.setdefault(storage_key, [])
            for value in values:
                if value not in merged[storage_key]:
                    merged[storage_key].append(value)
    return merged or None


def _table_source_index(
    schema: SchemaGraph, mappings: FederationMappings, manifest: FederationManifest | None = None
) -> dict[str, str]:
    index: dict[str, str] = {}
    logical_member_sources: dict[str, set[str]] = {}
    for lt in mappings.logical_tables:
        logical_member_sources[lt.logical] = {m.source for m in lt.members}
    for name, table in schema.tables.items():
        if table.member_source_ids:
            if len(table.member_source_ids) == 1:
                index[name] = table.member_source_ids[0]
            continue
        if table.source_id:
            index[name] = table.source_id
        elif name in logical_member_sources:
            sources = logical_member_sources[name]
            if len(sources) == 1:
                index[name] = next(iter(sources))
    if manifest is not None:
        for name, source_id in manifest.table_namespace.items():
            if name in schema.tables and name not in index:
                index[name] = source_id
    return index


def _source_for_table(table: str, index: Mapping[str, str]) -> str:
    return index.get(table, "")


def _sources_for_table(
    table: str,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> frozenset[str]:
    if schema is not None:
        table_meta = schema.tables.get(table)
        if table_meta is not None and table_meta.member_source_ids:
            return frozenset(table_meta.member_source_ids)
    src = source_by_table.get(table, "")
    if src:
        return frozenset({src})
    for lt in mappings.logical_tables:
        if lt.logical == table:
            members = frozenset(m.source for m in lt.members)
            if members:
                return members
    ns = manifest.table_namespace.get(table, "")
    if ns:
        return frozenset({ns})
    return frozenset()


def _sources_for_refs(
    refs: Iterable[str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> set[str]:
    sources: set[str] = set()
    for table in refs:
        sources.update(_sources_for_table(table, manifest, mappings, source_by_table, schema))
    return sources


def _planning_sources_for_table(
    table: str,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> frozenset[str]:
    """Return member sources that should receive a planning step for *table*."""
    all_sources = _sources_for_table(table, manifest, mappings, source_by_table, schema)
    if len(all_sources) <= 1:
        return all_sources
    for lt in mappings.logical_tables:
        if lt.logical != table:
            continue
        if lt.semantics == "replica":
            auth = select_replica_member_source(lt)
            return frozenset({auth})
        return all_sources
    return all_sources


def resolve_federation_preview_target(
    table_name: str,
    *,
    schema: SchemaGraph,
    manifest: FederationManifest,
    mappings: FederationMappings,
    members: Mapping[str, Any],
) -> tuple[Any, str, dict[str, str]]:
    """Resolve the member engine and physical table for a composite preview."""
    norm_table = str(table_name).strip()
    if norm_table not in schema.tables:
        raise ConfigError(f"unknown table {table_name!r}")
    source_by_table = _mapping_member_source_by_table(mappings)
    sources = sorted(
        _planning_sources_for_table(norm_table, manifest, mappings, source_by_table, schema),
    )
    if not sources:
        sid = _table_source_index(schema, mappings, manifest).get(norm_table, "")
        if not sid:
            raise ConfigError(f"unknown table {table_name!r}")
        sources = [sid]
    source_id = sources[0]
    member = members.get(source_id)
    if member is None:
        raise ConfigError(f"unknown federation member for table {table_name!r}")
    physical_table = norm_table
    col_map: dict[str, str] = {}
    for table_map in mappings.logical_tables:
        if table_map.logical != norm_table:
            continue
        for table_member in table_map.members:
            if table_member.source == source_id:
                physical_table = table_member.table
                col_map = dict(table_member.columns)
                break
        break
    return member, physical_table, col_map


def _table_owned_by_source(
    table: str,
    source_id: str,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> bool:
    """Return True when *source_id* should execute a sub-plan over *table*."""
    return source_id in _planning_sources_for_table(table, manifest, mappings, source_by_table, schema)


def _intent_table_sources(
    tables: Iterable[str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> set[str]:
    sources: set[str] = set()
    for table in tables:
        sources.update(_sources_for_table(table, manifest, mappings, source_by_table, schema))
    return sources


def split_qualified_column(
    qualified: str,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Split ``table.column`` or ``source.table.column`` into canonical table and column."""
    text = str(qualified).strip()
    if manifest is not None:
        resolved = resolve_federation_qualified_ref(
            text, manifest=manifest, schema=schema, source_by_table=source_by_table
        )
        return resolved.table, resolved.column
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        return three.group(2), three.group(3)
    match = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if not match:
        raise ConfigError(f"expected table.column reference: {qualified!r}")
    return match.group(1), match.group(2)


def _widen_federation_scope_tables(tables: set[str], intent: RuntimeIntent, manifest: FederationManifest) -> None:
    """Add tables from the resolved join path and cross-source join endpoints when scope is multi-source."""
    path_tables = {str(entry).strip() for entry in (intent.chosen_join_path_signature or []) if str(entry).strip()}
    tables.update(path_tables)
    scope = set(tables) | path_tables
    if not scope:
        return
    source_ids: set[str] = set()
    for table in scope:
        owner = manifest.table_namespace.get(table, "")
        if owner:
            source_ids.add(owner)
    for join in manifest.cross_source_joins:
        left_tbl, _left_col = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _right_col = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if left_tbl in scope and right_src in source_ids:
            tables.add(left_tbl)
            tables.add(right_tbl)
        elif right_tbl in scope and left_src in source_ids:
            tables.add(left_tbl)
            tables.add(right_tbl)
        elif len(source_ids) > 1 and (left_tbl in scope or right_tbl in scope):
            tables.add(left_tbl)
            tables.add(right_tbl)


def _expand_scope_along_foreign_keys(tables: set[str], schema: SchemaGraph) -> None:
    """Add tables reachable by one foreign-key hop from tables already in scope."""
    for tbl in list(tables):
        table_meta = schema.tables.get(tbl)
        if table_meta is None:
            continue
        for edge in table_meta.foreign_keys:
            if edge.inference_tag == InferenceTag.CROSS_SOURCE:
                continue
            if edge.dst_table and edge.dst_table not in tables:
                tables.add(edge.dst_table)
            if edge.src_table and edge.src_table not in tables:
                tables.add(edge.src_table)


def federation_table_set(
    intent: RuntimeIntent, schema: SchemaGraph, manifest: FederationManifest, mappings: FederationMappings | None = None
) -> FederationTableSet:
    """Return intent tables and their owning sources using the composite graph."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    tables = set(intent.tables or [])
    refs = collect_referenced_tables(
        list(intent.select_cols or []),
        list(intent.order_by_cols or []),
        list(intent.group_by_cols or []),
        list(_predicate_where_leaves(intent.where) or []),
        list(_predicate_having_leaves(intent.having) or []),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        include_unreferenced_registries=False,
    )
    tables.update(refs)
    _widen_federation_scope_tables(tables, intent, manifest)
    _expand_scope_along_foreign_keys(tables, schema)
    source_by_table = _table_source_index(schema, mappings, manifest)
    sources: set[str] = set()
    for table in tables:
        sources.update(_sources_for_table(table, manifest, mappings, source_by_table, schema))
    return FederationTableSet(tables=frozenset(tables), source_by_table=source_by_table, sources=frozenset(sources))


def _value_type_duckdb_cast(value_type: str) -> str | None:
    """Map an intent ``value_type`` token to a DuckDB cast target for join keys."""
    vt = str(value_type or "").strip().lower()
    if vt == "integer":
        return "BIGINT"
    if vt == "number":
        return "DOUBLE"
    if vt == "date":
        return "TIMESTAMP"
    if vt == "boolean":
        return "BOOLEAN"
    return None


def _coordinator_join_key_expr(alias: str, key: str, *, schema: SchemaGraph | None, table_name: str | None) -> str:
    """Render a typed coordinator join key expression for *alias*.*key*."""
    ident = f"{alias}.{_quote_ident(key)}"
    if schema is None or not table_name:
        return ident
    value_type = column_where_value_type(schema, table_name, key)
    cast_type = _value_type_duckdb_cast(value_type)
    if cast_type is not None:
        return f"CAST({ident} AS {cast_type})"
    return ident


def _join_specs_connect_sources(sources: frozenset[str], join_specs: Sequence[JoinSpec]) -> bool:
    if len(sources) <= 1:
        return True
    parent = {source_id: source_id for source_id in sources}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    for spec in join_specs:
        if spec.left_source in parent and spec.right_source in parent:
            union(spec.left_source, spec.right_source)
    return len({find(source_id) for source_id in sources}) == 1


def _group_by_tables(intent: RuntimeIntent) -> set[str]:
    return collect_referenced_tables(
        [],
        [],
        intent.group_by_cols or [],
        [],
        [],
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        include_unreferenced_registries=False,
    )


def _expr_has_unattributable_raw_sql(expr: NormalizedExpr) -> bool:
    """Return True when *expr* carries raw SQL with no recoverable column references."""
    if not (expr.raw_sql or "").strip():
        return False
    return not extract_columns_from_expr(expr)


def _unattributable_raw_sql_reason(intent: RuntimeIntent) -> str | None:
    """Return an ineligibility reason when the intent contains unattributable raw SQL."""
    for sc in intent.select_cols or []:
        if _expr_has_unattributable_raw_sql(sc.expr):
            return "expression contains unattributable raw_sql fragment"
    for obc in intent.order_by_cols or []:
        if _expr_has_unattributable_raw_sql(obc.expr):
            return "expression contains unattributable raw_sql fragment"
    for group in intent.group_by_cols or []:
        if _expr_has_unattributable_raw_sql(group):
            return "expression contains unattributable raw_sql fragment"
    for fp in _predicate_where_leaves(intent.where) or []:
        if _expr_has_unattributable_raw_sql(fp.left_expr):
            return "expression contains unattributable raw_sql fragment"
        if fp.right_expr and _expr_has_unattributable_raw_sql(fp.right_expr):
            return "expression contains unattributable raw_sql fragment"
    for hp in _predicate_having_leaves(intent.having) or []:
        if _expr_has_unattributable_raw_sql(hp.left_expr):
            return "expression contains unattributable raw_sql fragment"
        if hp.right_expr and _expr_has_unattributable_raw_sql(hp.right_expr):
            return "expression contains unattributable raw_sql fragment"
    return None


def _clause_referenced_sources(
    *,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    schema: SchemaGraph | None = None,
    window_registry: Sequence[Any] | None = None,
    case_registry: Sequence[Any] | None = None,
) -> set[str]:
    refs = collect_referenced_tables(
        list(select_cols or []),
        list(order_by_cols or []),
        list(group_by_cols or []),
        list(where_params or []),
        list(having_param or []),
        window_registry=window_registry,
        case_registry=case_registry,
    )
    if manifest is not None and mappings is not None:
        return _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
    sources: set[str] = set()
    for table in refs:
        source_id = source_by_table.get(table, "")
        if source_id:
            sources.add(source_id)
    return sources


def _clause_spans_multiple_sources(
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    schema: SchemaGraph | None = None,
    window_registry: Sequence[Any] | None = None,
    case_registry: Sequence[Any] | None = None,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
) -> bool:
    return (
        len(
            _clause_referenced_sources(
                select_cols=select_cols,
                order_by_cols=order_by_cols,
                group_by_cols=group_by_cols,
                where_params=where_params,
                having_param=having_param,
                source_by_table=source_by_table,
                manifest=manifest,
                mappings=mappings,
                schema=schema,
                window_registry=window_registry,
                case_registry=case_registry,
            )
        )
        > 1
    )


def _intent_cross_source_aggregate_shape(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> tuple[bool, bool]:
    """Return ``(has_cross_source_shape, has_decomposable_cross_source_aggregate)``."""
    intent_tables = set(intent.tables or [])
    if not intent_tables:
        return False, False
    intent_sources = _intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return False, False
    registry_kw = _intent_registry_kw(intent)
    has_shape = False
    has_decomposable = False
    group_tables = _group_by_tables(intent)
    if group_tables:
        group_sources = _intent_table_sources(group_tables, manifest, mappings, source_by_table, schema=schema)
        if len(group_sources) > 1:
            has_shape = True
        elif len(group_sources) == 1 and intent_sources - group_sources:
            has_shape = True
    for sc in intent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        agg_tables = _tables_referenced_by_select_col(sc, **registry_kw)
        agg_sources = _intent_table_sources(agg_tables, manifest, mappings, source_by_table, schema=schema)
        cross = len(agg_sources) > 1 or (len(agg_sources) == 1 and bool(intent_sources - agg_sources))
        if not cross:
            continue
        has_shape = True
        func, has_distinct = _select_col_agg_meta(sc)
        if func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS and not has_distinct:
            has_decomposable = True
    return has_shape, has_decomposable


def _intent_has_cross_source_aggregate(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> bool:
    has_shape, _ = _intent_cross_source_aggregate_shape(intent, manifest, mappings, source_by_table, schema=schema)
    return has_shape


def _cross_source_aggregate_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse cross-source aggregate shapes the coordinator cannot fold."""
    intent_tables = set(intent.tables or [])
    if not intent_tables:
        return None
    intent_sources = _intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    registry_kw = _intent_registry_kw(intent)
    group_tables = _group_by_tables(intent)
    if group_tables:
        group_sources = _intent_table_sources(group_tables, manifest, mappings, source_by_table, schema=schema)
        if len(group_sources) > 1:
            for sc in intent.select_cols or []:
                if not _is_sql_aggregate_select_col(sc):
                    continue
                func, has_distinct = _select_col_agg_meta(sc)
                if has_distinct or func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
                    return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
    for sc in intent.select_cols or []:
        if not _is_sql_aggregate_select_col(sc):
            continue
        agg_tables = _tables_referenced_by_select_col(sc, **registry_kw)
        agg_sources = _intent_table_sources(agg_tables, manifest, mappings, source_by_table, schema=schema)
        cross = len(agg_sources) > 1 or (len(agg_sources) == 1 and bool(intent_sources - agg_sources))
        if not cross:
            continue
        func, has_distinct = _select_col_agg_meta(sc)
        union_targets = {
            lt.logical for lt in mappings.logical_tables if lt.semantics == "union" and lt.logical in agg_tables
        }
        if not func:
            if union_targets:
                func = "count"
            elif sc.is_aggregated and (intent.grain or "") == "scalar":
                func = "sum"
        if has_distinct:
            return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
        if len(agg_sources) > 1:
            if union_targets and func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
                continue
        if func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
            return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
    return None


def _assign_cte_sources(cte_steps: Sequence[RuntimeCteStep], source_by_table: Mapping[str, str]) -> dict[str, str]:
    """Map each CTE name to its owning source when tables and dependencies are local."""
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}
    cte_names_lower = {name.lower() for name in cte_names}
    owners: dict[str, str] = {}
    for cte in cte_steps:
        name = cte.cte_name
        if not name:
            continue
        refs = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            _predicate_where_leaves(cte.where),
            _predicate_having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            include_unreferenced_registries=False,
        )
        base_tables = {table for table in refs if table not in cte_names and table.lower() not in cte_names_lower}
        for table in cte.tables or ():
            if table and table not in cte_names and table.lower() not in cte_names_lower:
                base_tables.add(table)
        prior_ctes = {table for table in refs if table in cte_names or table.lower() in cte_names_lower}
        sources: set[str] = set()
        for table in base_tables:
            source_id = source_by_table.get(table, "")
            if source_id:
                sources.add(source_id)
        for prior in prior_ctes:
            canonical = next((candidate for candidate in cte_names if candidate.lower() == prior.lower()), prior)
            owner = owners.get(canonical)
            if owner:
                sources.add(owner)
        if len(sources) == 1:
            owners[name] = next(iter(sources))
    return owners


def _source_join_key_is_unique(
    schema: SchemaGraph, source_id: str, key: str, *, manifest: FederationManifest | None = None
) -> bool | None:
    """Return whether *key* is unique on *source_id*, or None when schema facts are missing."""
    col = str(key or "").strip()
    table_name = ""
    if "." in col:
        table_name, col = col.rsplit(".", 1)
    if table_name:
        table = schema.tables.get(table_name)
        if table is None or (table.source_id and table.source_id != source_id):
            return None
    else:
        matches = [
            name for name, table in schema.tables.items() if table.source_id == source_id and col in table.columns
        ]
        if len(matches) != 1:
            if manifest is not None:
                matches = [
                    name
                    for name, owner in (manifest.table_namespace or {}).items()
                    if owner == source_id and name in schema.tables and col in schema.tables[name].columns
                ]
            if len(matches) != 1:
                return None
        table_name = matches[0]
        table = schema.tables.get(table_name)
    if table is None:
        return None
    column = table.columns.get(col)
    if column is None:
        return None
    pk = list(table.primary_key or [])
    if pk == [col]:
        return True
    if column.is_unique:
        return True
    row_count = int(table.row_count or 0)
    distinct = int(column.distinct_count or 0)
    if row_count > 0 and distinct > 0:
        return distinct >= row_count
    return None


def _source_is_left_combine_nullable_side(source_id: str, combine: Sequence[JoinSpec]) -> bool:
    """Return True when *source_id* is the nullable side of a declared left combine edge."""
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        if kind == "left" and source_id == spec.right_source:
            return True
    return False


def reducing_edge_allowed_for_target(
    target_source_id: str,
    join: FederationCrossSourceJoin,
    manifest: FederationManifest,
    *,
    schema: SchemaGraph | None = None,
) -> bool:
    """Return True when semi-join reduction may filter *target_source_id* for *join*."""
    spec = _cross_source_join_to_spec(join, manifest, schema=schema)
    return not _member_rows_final_after_combine(
        target_source_id,
        combine=(spec,),
        schema=schema,
        manifest=manifest,
    )


def _join_preserves_member_rows(
    source_id: str, spec: JoinSpec, *, schema: SchemaGraph | None, manifest: FederationManifest | None
) -> bool | None:
    """Return whether *spec* leaves *source_id* row membership and multiplicity unchanged."""
    if source_id not in (spec.left_source, spec.right_source):
        return True
    this_is_left = source_id == spec.left_source
    kind = (spec.kind or "inner").strip().lower()
    if kind == "inner":
        return False
    if kind == "left" and not this_is_left:
        return False
    if schema is None:
        return None
    other_source = spec.right_source if this_is_left else spec.left_source
    other_key = spec.right_key if this_is_left else spec.left_key
    return _source_join_key_is_unique(schema, other_source, other_key, manifest=manifest)


def _member_rows_final_after_combine(
    source_id: str,
    *,
    combine: tuple[JoinSpec, ...] | None,
    schema: SchemaGraph | None,
    manifest: FederationManifest | None,
) -> bool:
    """Return True only when every combine edge involving *source_id* preserves its rows."""
    if not combine:
        return True
    for spec in combine:
        if source_id not in (spec.left_source, spec.right_source):
            continue
        preserved = _join_preserves_member_rows(source_id, spec, schema=schema, manifest=manifest)
        if preserved is not True:
            return False
    return True


def _window_owner_source(entry: Any, source_by_table: Mapping[str, str]) -> str | None:
    """Return the single member owning *entry* columns, or None when unowned or spanning."""
    refs = collect_referenced_tables([], [], [], [], [], window_registry=[entry], case_registry=[])
    sources = {source_by_table.get(table, "") for table in refs if source_by_table.get(table, "")}
    sources.discard("")
    if len(sources) != 1:
        return None
    return next(iter(sources))


def _member_window_rows_are_final(
    source_id: str,
    entry: Any,
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when *entry* is local to *source_id* and that member's rows are final."""
    owner = _window_owner_source(entry, source_by_table)
    if owner != source_id:
        return False
    if len(set(source_by_table.values())) <= 1:
        return True
    if manifest is None or schema is None:
        return False
    return _member_rows_final_after_combine(source_id, combine=combine, schema=schema, manifest=manifest)


def _window_requires_coordinator(
    entry: Any,
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when *entry* must be evaluated after the cross-source combine."""
    owner = _window_owner_source(entry, source_by_table)
    if owner is None:
        return True
    return not _member_window_rows_are_final(
        owner, entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
    )


def _partition_cte_steps_for_source(
    cte_steps: Sequence[RuntimeCteStep], source_id: str, source_by_table: Mapping[str, str]
) -> list[RuntimeCteStep]:
    """Keep only CTE steps owned by *source_id*, with registries filtered to clauses."""
    if not cte_steps:
        return []
    owners = _assign_cte_sources(cte_steps, source_by_table)
    kept: list[RuntimeCteStep] = []
    for cte in cte_steps:
        name = cte.cte_name
        if not name or owners.get(name) != source_id:
            continue
        cte_copy = copy.deepcopy(cte)
        window_registry, case_registry = where_scope_registries_to_referenced(
            select_cols=cte_copy.select_cols,
            order_by_cols=cte_copy.order_by_cols,
            group_by_cols=cte_copy.group_by_cols,
            where_params=(_predicate_where_leaves(cte_copy.where)),
            having_param=(_predicate_having_leaves(cte_copy.having)),
            window_registry=cte_copy.window_registry,
            case_registry=cte_copy.case_registry,
        )
        if window_registry != list(cte_copy.window_registry or []) or case_registry != list(
            cte_copy.case_registry or []
        ):
            cte_copy = replace(cte_copy, window_registry=window_registry, case_registry=case_registry)
        kept.append(cte_copy)
    return kept


def _partition_registries_for_source(
    intent: RuntimeIntent, source_id: str, source_by_table: Mapping[str, str]
) -> RuntimeIntent:
    """Keep window/case rows local to *source_id* only when member rows are final after combine."""
    source_tables = {table for table, sid in source_by_table.items() if sid == source_id}
    ctx = _WINDOW_FINALITY_CTX

    def _registry_local(registry: Sequence[Any], *, field: str) -> list[Any]:
        kept: list[Any] = []
        for entry in registry or []:
            if field == "window":
                refs = collect_referenced_tables([], [], [], [], [], window_registry=[entry], case_registry=[])
            else:
                refs = collect_referenced_tables([], [], [], [], [], window_registry=[], case_registry=[entry])
            if not refs or not refs.issubset(source_tables):
                continue
            if field == "window":
                if not _member_window_rows_are_final(
                    source_id,
                    entry,
                    source_by_table=source_by_table,
                    manifest=ctx.manifest if ctx is not None else None,
                    schema=ctx.schema if ctx is not None else None,
                    combine=ctx.combine if ctx is not None else None,
                ):
                    continue
            kept.append(entry)
        return kept

    window_registry, case_registry = where_scope_registries_to_referenced(
        select_cols=intent.select_cols,
        order_by_cols=intent.order_by_cols,
        group_by_cols=intent.group_by_cols,
        where_params=_predicate_where_leaves(intent.where),
        having_param=_predicate_having_leaves(intent.having),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    window_registry = _registry_local(window_registry, field="window")
    case_registry = _registry_local(case_registry, field="case")
    if window_registry == list(intent.window_registry or []) and case_registry == list(intent.case_registry or []):
        return intent
    return replace(intent, window_registry=window_registry, case_registry=case_registry)


def _unqualified_column_name(qualified: str) -> str:
    text = str(qualified or "").strip()
    if not text or "(" in text:
        return ""
    match = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if match:
        return match.group(2)
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        return three.group(3)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _source_column_names_for_step(plan: FederatedPlan, source_id: str) -> set[str]:
    step = next((item for item in plan.steps if item.source_id == source_id), None)
    if step is None:
        return set()
    names: set[str] = set()
    for key in step.projected_keys:
        col = _unqualified_column_name(key)
        if col:
            names.add(col)
    for sc in step.sub_intent.select_cols or []:
        if _looks_aggregated(sc):
            continue
        col = _unqualified_column_name(_select_col_term(sc))
        if col:
            names.add(col)
    return names


def _combine_select_column_names(plan: FederatedPlan) -> list[str] | None:
    """Derive explicit coordinator column names from projected/residual keys."""
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        col = _unqualified_column_name(name)
        if col and col not in seen:
            seen.add(col)
            names.append(col)

    for step in plan.steps:
        for key in step.projected_keys:
            _add(key)
        for sc in step.sub_intent.select_cols or []:
            if _looks_aggregated(sc):
                continue
            _add(_select_col_term(sc))
    residual = plan.residual
    if residual is not None:
        for fp in _predicate_where_leaves(residual.where):
            left_ref = fp.left_expr.column_ref or fp.left_expr.primary_term or ""
            _add(left_ref)
            right_ref = getattr(fp.right_expr, "column_ref", None) or getattr(fp.right_expr, "primary_term", None) or ""
            if right_ref:
                _add(str(right_ref))
        for sc in residual.select_cols or []:
            if _looks_aggregated(sc):
                alias = (sc.output_alias or "").strip()
                if alias:
                    if alias not in seen:
                        seen.add(alias)
                        names.append(alias)
                continue
            alias = (sc.output_alias or "").strip()
            if alias:
                if alias not in seen:
                    seen.add(alias)
                    names.append(alias)
                continue
            _add(_select_col_term(sc))
    if not names and isinstance(plan.combine, tuple):
        for spec in plan.combine:
            if isinstance(spec, JoinSpec):
                _add(spec.left_key)
                _add(spec.right_key)
                if spec.logical_key:
                    _add(spec.logical_key)
    return names or None


def _render_combine_select_keyword(cols: list[str] | None) -> str:
    if not cols:
        raise FederationRuntimeError("federation combine requires explicit column projection")
    return ", ".join(_quote_ident(col) for col in cols)


def _render_join_select_keyword(
    cols: list[str] | None, *, left_alias: str, right_alias: str, left_cols: set[str], right_cols: set[str]
) -> str:
    if not cols:
        raise FederationRuntimeError("federation combine requires explicit column projection")
    exprs: list[str] = []
    for col in cols:
        ident = _quote_ident(col)
        in_left = col in left_cols
        in_right = col in right_cols
        if in_left and in_right:
            exprs.append(f"{left_alias}.{ident}, {right_alias}.{ident}")
        elif in_right:
            exprs.append(f"{right_alias}.{ident}")
        elif in_left:
            exprs.append(f"{left_alias}.{ident}")
        else:
            exprs.append(f"{left_alias}.{ident}")
    return ", ".join(exprs)


def _cross_source_where(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> list[WhereParam]:
    cross: list[WhereParam] = []
    registry_kw = _intent_registry_kw(intent)
    for fp in _predicate_where_leaves(intent.where) or []:
        refs = collect_referenced_tables([], [], [], [fp], [], **registry_kw)
        srcs = _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            cross.append(fp)
    return cross


def _cross_source_having(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> list[HavingParam]:
    cross: list[HavingParam] = []
    registry_kw = _intent_registry_kw(intent)
    for hp in _predicate_having_leaves(intent.having) or []:
        refs = collect_referenced_tables([], [], [], [], [hp], **registry_kw)
        srcs = _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            cross.append(hp)
    return cross


def _param_qualified_columns(param: WhereParam | HavingParam) -> set[str]:
    cols: set[str] = set()
    for expr in (param.left_expr, param.right_expr):
        if expr is None:
            continue
        for col in extract_columns_from_expr(expr):
            if "." in col:
                cols.add(col)
    return cols


def _cross_where_relates_to_join(fp: WhereParam, manifest: FederationManifest) -> bool:
    filter_cols = _param_qualified_columns(fp)
    if not filter_cols:
        return False
    for join in manifest.cross_source_joins:
        join_cols = {join.left, join.right}
        if filter_cols & join_cols:
            return True
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        filter_tables = {split_qualified_column(c, manifest=manifest)[0] for c in filter_cols if "." in c}
        if left_tbl in filter_tables and right_tbl in filter_tables:
            return True
    return False


def _predicate_is_literal_comparison(param: WhereParam | HavingParam) -> bool:
    """Return True when *param* compares a column to a literal rather than another column."""
    if (param.value_type or "").strip().lower() == "column":
        if param.right_expr is None:
            return False
        return not extract_columns_from_expr(param.right_expr)
    return True


def _join_covered_literal_push_allowed(
    param: WhereParam,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    registry_kw: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when a join-covered literal filter should execute on its owning member."""
    if not _predicate_is_literal_comparison(param):
        return False
    srcs = _predicate_param_sources(param, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw)
    if len(srcs) != 1:
        return False
    return _cross_where_relates_to_join(param, manifest)


def _expr_clause_label(expr: NormalizedExpr | None) -> str:
    if expr is None:
        return ""
    return (expr.primary_term or expr.column_ref or expr.primary_column or "").strip()


def _predicate_clause_label(param: WhereParam | HavingParam) -> str:
    left = _expr_clause_label(param.left_expr)
    op = (param.op or "").strip()
    right = _expr_clause_label(param.right_expr)
    if not right:
        raw = getattr(param, "raw_value", None)
        if raw is not None and raw != "":
            right = str(raw)
    parts = [part for part in (left, op, right) if part]
    return " ".join(parts) if parts else (op or "predicate")


def _param_referenced_sources(
    param: WhereParam | HavingParam,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    registry_kw: Mapping[str, Any] | None = None,
) -> set[str]:
    kw = dict(registry_kw or {})
    if isinstance(param, HavingParam):
        refs = collect_referenced_tables([], [], [], [], [param], **kw)
    else:
        refs = collect_referenced_tables([], [], [], [param], [], **kw)
    return _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)


def _cross_source_window_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse windows that need post-combine evaluation when no combine path exists."""
    intent_tables = set(intent.tables or [])
    intent_sources = _intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    combine = _join_specs_for_sources(
        manifest, mappings, frozenset(intent_sources), schema=schema, scope_tables=frozenset(intent_tables)
    )
    union_specs = _union_specs_for_intent(intent_tables, mappings, source_by_table)
    has_combine = bool(combine) or bool(union_specs)
    for entry in intent.window_registry or []:
        if not _window_requires_coordinator(
            entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
        ):
            continue
        if has_combine:
            continue
        rid = str(getattr(entry, "registry_id", "") or "").strip() or "window"
        spec = getattr(entry, "window_spec", None)
        func = str(getattr(spec, "function", "") or "").strip() or "window"
        return f"cross-source window is not supported: {rid} ({func})"
    return None


def _cross_source_scalar_subquery_ineligible_reason(
    intent: RuntimeIntent, source_by_table: Mapping[str, str]
) -> str | None:
    """Refuse scalar-subquery CTE steps whose body is not owned by a single member."""
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return None
    owners = _assign_cte_sources(cte_steps, source_by_table)
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}
    cte_names_lower = {name.lower() for name in cte_names}
    for cte in cte_steps:
        emission = getattr(cte, "emission", "join_table")
        if emission != "scalar_subquery":
            continue
        name = (cte.cte_name or "").strip()
        if not name:
            continue
        if name not in owners:
            return f"cross-source correlated subquery is not supported: {name}"
        declared = {table for table in (cte.tables or []) if table}
        refs = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            _predicate_where_leaves(cte.where),
            _predicate_having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            include_unreferenced_registries=False,
        )
        base_tables = {
            table for table in (declared | refs) if table not in cte_names and table.lower() not in cte_names_lower
        }
        sources = {source_by_table[table] for table in base_tables if table in source_by_table}
        if len(sources) > 1:
            return f"cross-source correlated subquery is not supported: {name}"
    return None


def _federation_clause_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse clause shapes federation cannot compose instead of silently dropping them."""
    registry_kw = _intent_registry_kw(intent)
    predicate_reason = _predicate_group_spans_sources(
        intent.where, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
    )
    if predicate_reason:
        return predicate_reason
    having_predicate_reason = _predicate_group_spans_sources(
        intent.having, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
    )
    if having_predicate_reason:
        return having_predicate_reason
    window_reason = _cross_source_window_ineligible_reason(intent, manifest, mappings, source_by_table, schema=schema)
    if window_reason:
        return window_reason
    subquery_reason = _cross_source_scalar_subquery_ineligible_reason(intent, source_by_table)
    if subquery_reason:
        return subquery_reason
    probe_reason = _cross_source_probe_cte_ineligible_reason(intent, manifest, source_by_table)
    if probe_reason:
        return probe_reason
    if intent.distinct_on:
        if _distinct_on_spans_sources(intent, manifest, mappings, source_by_table, schema=schema):
            combine = _join_specs_for_sources(
                manifest,
                mappings,
                frozenset(
                    _intent_table_sources(set(intent.tables or []), manifest, mappings, source_by_table, schema=schema)
                ),
                schema=schema,
                scope_tables=frozenset(intent.tables or []),
            )
            union_specs = _union_specs_for_intent(set(intent.tables or []), mappings, source_by_table)
            if not combine and not union_specs:
                return "cross-source distinct_on requires declared join"
        elif not intent.order_by_cols:
            return "distinct_on requires order_by_cols"
    cross_having = _cross_source_having(intent, manifest, mappings, source_by_table, schema=schema)
    if cross_having:
        uncovered = [hp for hp in cross_having if not _cross_having_covered(hp, manifest)]
        if uncovered:
            labels = ", ".join(_predicate_clause_label(hp) for hp in uncovered)
            return f"cross-source HAVING requires declared join: {labels}"
    return None


def _cross_having_covered(hp: HavingParam, manifest: FederationManifest) -> bool:
    if not manifest.cross_source_joins:
        return False
    cols = _param_qualified_columns(hp)
    if not cols:
        return False
    for join in manifest.cross_source_joins:
        join_cols = {join.left, join.right}
        if cols & join_cols:
            return True
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        filter_tables = {split_qualified_column(c, manifest=manifest)[0] for c in cols if "." in c}
        if left_tbl in filter_tables and right_tbl in filter_tables:
            return True
    return False


def _union_specs_for_intent(
    tables: set[str], mappings: FederationMappings, source_by_table: Mapping[str, str]
) -> list[UnionSpec]:
    del source_by_table
    specs: list[UnionSpec] = []
    for lt in mappings.logical_tables:
        if lt.logical not in tables:
            continue
        if lt.semantics == "replica":
            auth = select_replica_member_source(lt)
            member_sources: tuple[str, ...] = (auth,)
        else:
            member_sources = tuple(sorted({m.source for m in lt.members}))
        specs.append(UnionSpec(logical_table=lt.logical, member_source_ids=member_sources, semantics=lt.semantics))
    return specs


def _copy_runtime_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Return a deep copy of *intent* so sub-intents do not share mutable IR nodes."""
    return copy.deepcopy(intent)


def _isolate_sub_intent_decisions(intent: RuntimeIntent) -> RuntimeIntent:
    """Clear parent join and validation decisions a member sub-intent cannot honour."""
    return replace(
        intent,
        chosen_join_candidate_id="",
        chosen_join_path_signature=[],
        sql_shape=None,
        schema_invalid=False,
        planner_cte_names=[],
        grain="row_level",
    )


def _member_schema_for_sub_intent_repair(
    source_id: str,
    composite_schema: SchemaGraph,
    *,
    manifest: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph] | None,
) -> SchemaGraph:
    """Return the schema a sub-intent repair must be judged against. Prefers the loaded member graph. When that graph is absent, uses the per-source composite slice — never the full composite graph."""
    if member_graphs is not None and source_id in member_graphs:
        return member_graphs[source_id]
    return member_schema_slice(composite_schema, source_id, manifest=manifest, member_graph=None)


def _finalize_member_sub_intent(sub: RuntimeIntent, member_schema: SchemaGraph) -> RuntimeIntent:
    """Run shared-key expansion and post-compose processing against *member_schema*."""
    intent_process = importlib.import_module("aetherdialect._intent_process")
    expanded = expand_shared_pk_tables_for_refs(sub, member_schema)
    question_fallback = (sub.natural_language or "").strip()
    processed, post_issues = cast(
        tuple[RuntimeIntent | None, list[Any]],
        intent_process.apply_runtime_post_processing(expanded, member_schema, question_fallback=question_fallback),
    )
    if processed is None:
        raise FederationRuntimeError("federated member sub-intent post-processing incomplete")
    blocking = [issue for issue in post_issues if getattr(issue, "severity", "") == "error"]
    if blocking:
        messages = "; ".join(str(getattr(issue, "message", issue)) for issue in blocking)
        raise FederationRuntimeError(f"federated member sub-intent post-processing failed: {messages}")
    return processed


def _tables_in_federation_space(tables: set[str], space: SpaceContext | None) -> set[str]:
    """Return *tables* restricted to the active federation space allow/deny lists."""
    if space is None:
        return set(tables)
    scoped = set(tables)
    if space.tables:
        scoped &= set(space.tables)
    if space.deny_objects:
        scoped -= set(space.deny_objects)
    return scoped


def _build_source_sub_intent(
    intent: RuntimeIntent,
    source_id: str,
    tables: set[str],
    source_by_table: Mapping[str, str],
    mappings: FederationMappings,
    schema: SchemaGraph,
    manifest: FederationManifest,
    *,
    multi_source: bool = False,
    member_schema: SchemaGraph,
    chosen_specs: Sequence[JoinSpec] | None = None,
    space: SpaceContext | None = None,
) -> SourceStep | None:
    intent = _isolate_sub_intent_decisions(_copy_runtime_intent(intent))
    source_tables = {
        t for t in tables if _table_owned_by_source(t, source_id, manifest, mappings, source_by_table, schema)
    }
    source_tables = _tables_in_federation_space(source_tables, space)
    if not source_tables:
        return None
    cte_steps = _partition_cte_steps_for_source(intent.cte_steps or [], source_id, source_by_table)
    if space is not None:
        allowed_cte_names = _tables_in_federation_space(
            {cte.cte_name for cte in cte_steps if cte.cte_name},
            space,
        )
        cte_steps = [cte for cte in cte_steps if cte.cte_name in allowed_cte_names]
    partitioned_cte_steps = copy.deepcopy(cte_steps)
    source_tables |= {cte.cte_name for cte in cte_steps if cte.cte_name}
    source_tables = _tables_in_federation_space(source_tables, space)
    if not source_tables:
        return None

    def _predicate_local(param: WhereParam | HavingParam) -> bool:
        if isinstance(param, HavingParam):
            refs = collect_referenced_tables([], [], [], [], [param])
        else:
            refs = collect_referenced_tables([], [], [], [param], [])
        if not refs.issubset(source_tables):
            return False
        if chosen_specs and _source_is_left_combine_nullable_side(
            source_id,
            chosen_specs,
        ):
            srcs = _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
            if len(srcs) == 1 and source_id in srcs:
                if isinstance(param, WhereParam) and _join_covered_literal_push_allowed(
                    param, manifest, mappings, source_by_table, schema=schema
                ):
                    return True
                return False
        return True

    local_where, _ = partition_predicate_group(intent.where, _predicate_local)
    local_having, _ = partition_predicate_group(intent.having, _predicate_local)
    local_preserve_tables = sorted(
        {str(table).strip() for table in (intent.preserve_tables or []) if str(table).strip() in source_tables}
    )
    sub = replace(
        intent,
        tables=sorted(source_tables),
        where=local_where,
        having=local_having,
        preserve_tables=local_preserve_tables,
        cte_steps=cte_steps,
    )
    sub = _rewrite_logical_references(sub, source_id, mappings, schema, manifest)
    if multi_source:
        parent_cross_agg = _intent_has_cross_source_aggregate(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        _cross_shape, has_decomposable_partial = _intent_cross_source_aggregate_shape(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        sub = _intent_exprs_local_to_tables(sub, source_tables, multi_source=True, residual_fold=parent_cross_agg)
        if has_decomposable_partial:
            sub = _apply_member_partial_aggregation(
                sub,
                intent,
                source_id,
                source_tables,
                manifest,
                chosen_specs=chosen_specs,
            )
        sub = _strip_coordinator_clauses_from_sub_intent(sub)
        if chosen_specs:
            sub = replace(sub, order_by_cols=[])
    sub = _partition_registries_for_source(sub, source_id, source_by_table)
    sub = reconcile_tables(sub)
    sub_grain = sub.grain or "row_level"
    if sub_grain not in VALID_GRAINS:
        sub = replace(sub, grain="row_level")
    sub = _finalize_member_sub_intent(sub, member_schema)
    if partitioned_cte_steps:
        kept_names = {cte.cte_name for cte in (sub.cte_steps or []) if cte.cte_name}
        merged_ctes = list(sub.cte_steps or [])
        for cte in partitioned_cte_steps:
            if cte.cte_name and cte.cte_name not in kept_names:
                merged_ctes.append(cte)
        if merged_ctes != list(sub.cte_steps or []):
            sub = replace(sub, cte_steps=merged_ctes)
    source_tables = set(sub.tables or []) or source_tables
    keys = _projected_keys_for_step(
        source_id,
        manifest,
        sub,
        parent=intent,
        source_tables=source_tables,
        chosen_specs=chosen_specs,
    )
    return SourceStep(source_id=source_id, sub_intent=sub, projected_keys=keys)


def _looks_aggregated(select_col: SelectCol) -> bool:
    """Return whether *select_col* carries structured aggregate metadata."""
    return select_col.is_aggregated


def _select_col_agg_meta(select_col: SelectCol) -> tuple[str | None, bool]:
    """Return ``(agg_func, has_distinct)`` from structured select-column IR."""
    expr = select_col.expr
    if expr.agg_func:
        return str(expr.agg_func).lower(), False
    for group in expr.add_groups:
        if group.agg_func:
            return str(group.agg_func).lower(), bool(group.distinct)
    for group in expr.sub_groups:
        if group.agg_func:
            return str(group.agg_func).lower(), bool(group.distinct)
    return None, False


def _select_col_agg_func(select_col: SelectCol) -> str | None:
    """Return the structured aggregate function name from select-column IR metadata."""
    func, _ = _select_col_agg_meta(select_col)
    return func


def _select_col_agg_label(select_col: SelectCol) -> str:
    """Human-readable aggregate label derived from structured IR (not raw SQL text)."""
    func, has_distinct = _select_col_agg_meta(select_col)
    term = _select_col_term(select_col).strip()
    if func and term:
        distinct_kw = "distinct " if has_distinct else ""
        return f"{func}({distinct_kw}{term})"
    return term or func or "aggregate"


def _is_sql_aggregate_select_col(select_col: SelectCol) -> bool:
    """Return True for SQL aggregate select columns, excluding bare registry refs."""
    if expr_registry_ref(select_col.expr) is not None:
        return False
    return _looks_aggregated(select_col)


def _tables_referenced_by_select_col(
    sc: SelectCol, *, window_registry: Sequence[Any] | None = None, case_registry: Sequence[Any] | None = None
) -> set[str]:
    tables: set[str] = set()
    if _looks_aggregated(sc):
        inner = _aggregate_inner_column(sc)
        if inner and inner != "*":
            if "." in inner:
                tables.add(inner.split(".", 1)[0])
            return tables
    return collect_referenced_tables([sc], [], [], [], [], window_registry=window_registry, case_registry=case_registry)


def _strip_coordinator_clauses_from_sub_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Remove limit, distinct, and distinct_on from per-source sub- intents; coordinator applies them."""
    unchanged = intent.limit is None and intent.distinct_select_index < 0 and not intent.distinct_on
    if unchanged:
        return intent
    return replace(intent, limit=None, distinct_select_index=-1, distinct_on=[])


def _join_key_columns_for_source(
    source_id: str, manifest: FederationManifest, *, chosen_specs: Sequence[JoinSpec] | None = None
) -> list[str]:
    """Return qualified join-key columns declared on *source_id* for cross-source joins."""
    cols: list[str] = []
    for join in manifest.cross_source_joins:
        spec = _cross_source_join_to_spec(join, manifest)
        if chosen_specs is not None and spec not in chosen_specs:
            continue
        for qualified in (join.left, join.right):
            tbl, col = split_qualified_column(qualified, manifest=manifest)
            if manifest.table_namespace.get(tbl, "") != source_id:
                continue
            ref = f"{tbl}.{col}"
            if ref not in cols:
                cols.append(ref)
    return cols


def _aggregate_inner_column(select_col: SelectCol) -> str:
    """Return the aggregated column/star target from structured select- column IR."""
    if not select_col.is_aggregated:
        return ""
    col = select_col.expr.primary_column
    return col if col else ""


def _aggregate_columns_for_source(parent: RuntimeIntent, source_tables: set[str]) -> list[str]:
    cols: list[str] = []
    for sc in parent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        col_ref = _aggregate_inner_column(sc)
        if not col_ref or "." not in col_ref:
            continue
        tbl = col_ref.split(".", 1)[0]
        if tbl in source_tables:
            cols.append(col_ref)
    return cols


def _select_col_term(sc: SelectCol) -> str:
    expr = sc.expr
    term = (expr.primary_term or "").strip()
    if term:
        return term
    col = (expr.primary_column or "").strip()
    if col:
        return col
    return str(expr)


def _remap_distinct_select_index(old_cols: Sequence[SelectCol], new_cols: Sequence[SelectCol], old_index: int) -> int:
    """Translate ``distinct_select_index`` after select-col append/reorder."""
    if old_index < 0:
        return old_index
    if old_index >= len(old_cols):
        return -1
    target = old_cols[old_index]
    for idx, sc in enumerate(new_cols):
        if sc is target:
            return idx
    return -1


def apply_projected_keys_to_intent(intent: RuntimeIntent, projected_keys: tuple[str, ...]) -> RuntimeIntent:
    """Ensure *intent* projects every coordinator-required column."""
    if not projected_keys:
        return intent
    old_cols = list(intent.select_cols or [])
    existing = {_select_col_term(sc) for sc in old_cols}
    new_cols = [SelectCol(expr=NormalizedExpr.from_column(ref)) for ref in projected_keys if ref not in existing]
    if not new_cols:
        return intent
    merged = old_cols + new_cols
    distinct_idx = _remap_distinct_select_index(old_cols, merged, intent.distinct_select_index)
    return replace(intent, select_cols=merged, distinct_select_index=distinct_idx)


def _normalized_expr_identity(expr: NormalizedExpr) -> str:
    return (expr.column_ref or expr.primary_column or expr.primary_term or "").strip()


def _apply_member_partial_aggregation(
    sub: RuntimeIntent,
    parent: RuntimeIntent,
    source_id: str,
    source_tables: set[str],
    manifest: FederationManifest,
    *,
    chosen_specs: Sequence[JoinSpec] | None = None,
) -> RuntimeIntent:
    """Pre-aggregate decomposable member aggregates before the coordinator folds them."""
    group_exprs: list[NormalizedExpr] = []
    seen_group: set[str] = set()
    for key in _join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs):
        expr = NormalizedExpr.from_column(key)
        ident = _normalized_expr_identity(expr)
        if ident and ident not in seen_group:
            seen_group.add(ident)
            group_exprs.append(expr)
    for col in parent.group_by_cols or []:
        refs = collect_referenced_tables([], [], [col], [], [])
        if refs and refs.issubset(source_tables):
            ident = _normalized_expr_identity(col)
            if ident and ident not in seen_group:
                seen_group.add(ident)
                group_exprs.append(col)

    select_cols: list[SelectCol] = []
    seen_select: set[str] = set()
    for expr in group_exprs:
        ident = _normalized_expr_identity(expr)
        if ident and ident not in seen_select:
            seen_select.add(ident)
            select_cols.append(SelectCol(expr=expr))

    for sc in parent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        inner = _aggregate_inner_column(sc)
        if not inner or inner == "*":
            continue
        if "." in inner and inner.split(".", 1)[0] not in source_tables:
            continue
        func, has_distinct = _select_col_agg_meta(sc)
        if has_distinct or func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
            continue
        if func == "avg":
            for partial_func in ("sum", "count"):
                partial_expr = NormalizedExpr.from_agg(partial_func, inner)
                ident = f"{partial_func}:{inner}"
                if ident not in seen_select:
                    seen_select.add(ident)
                    select_cols.append(SelectCol(expr=partial_expr))
            continue
        partial_expr = NormalizedExpr.from_agg(func, inner)
        ident = f"{func}:{inner}"
        if ident not in seen_select:
            seen_select.add(ident)
            select_cols.append(SelectCol(expr=partial_expr))

    if not any(_looks_aggregated(sc) for sc in select_cols):
        return sub
    grain = "grouped" if group_exprs else (parent.grain or "row_level")
    return replace(
        sub,
        tables=sorted(source_tables),
        grain=grain,
        select_cols=select_cols,
        group_by_cols=group_exprs,
        order_by_cols=[],
    )


def _intent_exprs_local_to_tables(
    intent: RuntimeIntent, source_tables: set[str], *, multi_source: bool = False, residual_fold: bool = False
) -> RuntimeIntent:
    """Drop parent aggregates/projections that reference tables outside *source_tables*."""
    allowed = set(source_tables)

    def _expr_refs_local(
        select_cols: Sequence[SelectCol],
        order_by_cols: Sequence[OrderByCol],
        group_by_cols: Sequence[NormalizedExpr],
        where_params: Sequence[WhereParam],
        having_param: Sequence[HavingParam],
    ) -> bool:
        refs = collect_referenced_tables(
            list(select_cols),
            list(order_by_cols),
            list(group_by_cols),
            list(where_params),
            list(having_param),
            window_registry=intent.window_registry,
            case_registry=intent.case_registry,
            include_unreferenced_registries=False,
        )
        return bool(refs) and refs.issubset(allowed)

    select_cols = [
        sc
        for sc in (intent.select_cols or [])
        if not _is_sql_aggregate_select_col(sc) and _expr_refs_local([sc], [], [], [], [])
    ]
    group_by_cols = [col for col in (intent.group_by_cols or []) if _expr_refs_local([], [], [col], [], [])]
    order_by_cols = [col for col in (intent.order_by_cols or []) if _expr_refs_local([], [col], [], [], [])]
    having_leaves = [
        hp for hp in (_predicate_having_leaves(intent.having) or []) if _expr_refs_local([], [], [], [], [hp])
    ]
    having_group = predicate_group_from_list(having_leaves)
    parent_refs = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        _predicate_where_leaves(intent.where),
        _predicate_having_leaves(intent.having),
    )
    parent_had_agg = any(_is_sql_aggregate_select_col(sc) for sc in (intent.select_cols or []))
    fold_to_residual = multi_source and (residual_fold or (intent.grain or "") == "scalar" or parent_had_agg)
    if fold_to_residual:
        return replace(
            intent,
            grain="row_level",
            select_cols=select_cols,
            group_by_cols=[] if residual_fold else group_by_cols,
            order_by_cols=order_by_cols,
            having=having_group,
            limit=None,
            distinct_select_index=-1,
        )
    if parent_refs - allowed and not select_cols and not group_by_cols:
        return replace(
            intent, grain="row_level", select_cols=[], group_by_cols=[], order_by_cols=[], having=None, limit=None
        )
    return replace(
        intent, select_cols=select_cols, group_by_cols=group_by_cols, order_by_cols=order_by_cols, having=having_group
    )


def _member_logical_column_map(
    source_id: str,
    mappings: FederationMappings,
    schema: SchemaGraph,
    base_map: Mapping[str, str] | None = None,
    manifest: FederationManifest | None = None,
) -> dict[str, str]:
    """Build logical-to-physical column aliases for one federation member."""
    column_map = dict(base_map or {})
    for col_map in mappings.logical_columns:
        for member in col_map.members:
            tbl, col = split_qualified_column(member, manifest=manifest)
            if schema.tables.get(tbl) and schema.tables[tbl].source_id == source_id:
                column_map[col_map.logical] = col
    for lt in mappings.logical_tables:
        for table_member in lt.members:
            if table_member.source != source_id:
                continue
            for logical, physical in table_member.columns.items():
                column_map[logical] = physical
    return column_map


def _rewrite_logical_references(
    intent: RuntimeIntent,
    source_id: str,
    mappings: FederationMappings,
    schema: SchemaGraph,
    manifest: FederationManifest | None = None,
) -> RuntimeIntent:
    column_map = _member_logical_column_map(source_id, mappings, schema, intent.column_map, manifest)
    if not column_map:
        return intent
    cte_steps_out: list[RuntimeCteStep] = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        merged = dict(cte.column_map or {})
        for logical, physical in column_map.items():
            merged[logical] = physical
        if merged != dict(cte.column_map or {}):
            cte = replace(cte, column_map=merged)
            cte_changed = True
        cte_steps_out.append(cte)
    if column_map == dict(intent.column_map or {}) and not cte_changed:
        return intent
    return replace(intent, column_map=column_map, cte_steps=cte_steps_out if cte_changed else intent.cte_steps)


def _projected_keys_for_step(
    source_id: str,
    manifest: FederationManifest,
    sub_intent: RuntimeIntent,
    *,
    parent: RuntimeIntent | None = None,
    source_tables: set[str] | None = None,
    chosen_specs: Sequence[JoinSpec] | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    keys.extend(_join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs))
    if parent is not None and source_tables:
        keys.extend(_aggregate_columns_for_source(parent, source_tables))
    for sc in sub_intent.select_cols or []:
        term = _select_col_term(sc)
        if term:
            keys.append(term)
    return tuple(dict.fromkeys(keys))


def _residual_spec_for_intent(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    *,
    schema: SchemaGraph | None = None,
    scope_tables: Iterable[str] | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> ResidualSpec | None:
    intent_tables = set(scope_tables or intent.tables or [])
    if not intent_tables:
        return None
    intent_sources = _intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    registry_kw = _intent_registry_kw(intent)
    source_kw: dict[str, Any] = {
        "source_by_table": source_by_table,
        "manifest": manifest,
        "mappings": mappings,
        "schema": schema,
        **registry_kw,
    }
    has_cross_agg = _intent_has_cross_source_aggregate(intent, manifest, mappings, source_by_table, schema=schema)

    def _cross_source(**clause: Any) -> bool:
        return _clause_spans_multiple_sources(**source_kw, **clause)

    if has_cross_agg:
        select_cols = tuple(
            sc for sc in (intent.select_cols or []) if _looks_aggregated(sc) or _cross_source(select_cols=[sc])
        )
        parent_cross_select_agg = any(_looks_aggregated(sc) for sc in (intent.select_cols or []))

        def _group_by_needed_post_join(col: NormalizedExpr) -> bool:
            if _cross_source(group_by_cols=[col]):
                return True
            col_sources = _clause_referenced_sources(**source_kw, group_by_cols=[col])
            if not col_sources:
                return False
            return bool(intent_sources - col_sources)

        if parent_cross_select_agg and (intent.grain or "") != "grouped":
            group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _cross_source(group_by_cols=[col]))
        else:
            group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _group_by_needed_post_join(col))
    else:
        select_cols = tuple(sc for sc in (intent.select_cols or []) if _cross_source(select_cols=[sc]))
        group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _cross_source(group_by_cols=[col]))
    order_by_cols = (
        tuple(_explicit_residual_order_col(col) for col in (intent.order_by_cols or []))
        if combine
        else tuple(
            _explicit_residual_order_col(col)
            for col in (intent.order_by_cols or [])
            if _cross_source(order_by_cols=[col])
        )
    )

    def _predicate_spans_sources(param: WhereParam | HavingParam) -> bool:
        if isinstance(param, HavingParam):
            refs = collect_referenced_tables([], [], [], [], [param], **registry_kw)
        else:
            refs = collect_referenced_tables([], [], [], [param], [], **registry_kw)
        srcs = _sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            return True
        if combine and len(srcs) == 1:
            if isinstance(param, WhereParam) and _join_covered_literal_push_allowed(
                param,
                manifest,
                mappings,
                source_by_table,
                schema=schema,
                registry_kw=registry_kw,
            ):
                return False
            return _source_is_left_combine_nullable_side(next(iter(srcs)), combine)
        return False

    _, residual_where = partition_predicate_group(intent.where, lambda param: not _predicate_spans_sources(param))
    _, residual_having = partition_predicate_group(intent.having, lambda param: not _predicate_spans_sources(param))
    distinct_on: tuple[NormalizedExpr, ...] = ()
    if intent.distinct_on and _distinct_on_spans_sources(intent, manifest, mappings, source_by_table, schema=schema):
        distinct_on = tuple(intent.distinct_on)
    distinct_select_index = intent.distinct_select_index if intent.distinct_select_index >= 0 else -1
    limit = intent.limit

    def _registry_entry_cross_source(entry: Any, *, field: str) -> bool:
        if field == "window":
            return _window_requires_coordinator(
                entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
            )
        return _clause_spans_multiple_sources(
            source_by_table=source_by_table, manifest=manifest, mappings=mappings, schema=schema, case_registry=[entry]
        )

    window_registry = tuple(
        entry for entry in (intent.window_registry or []) if _registry_entry_cross_source(entry, field="window")
    )
    case_registry = tuple(
        entry for entry in (intent.case_registry or []) if _registry_entry_cross_source(entry, field="case")
    )
    if window_registry:
        window_ids = {
            str(getattr(entry, "registry_id", "") or "")
            for entry in window_registry
            if getattr(entry, "registry_id", "")
        }
        promoted: list[SelectCol] = list(select_cols)
        seen_refs: set[str] = set()
        for sc in select_cols:
            ref = expr_registry_ref(sc.expr)
            if ref:
                seen_refs.add(ref)
        for sc in intent.select_cols or []:
            ref = expr_registry_ref(sc.expr)
            if ref and ref in window_ids and ref not in seen_refs:
                promoted.append(sc)
                seen_refs.add(ref)
        select_cols = tuple(promoted)
    if not select_cols and (residual_where or window_registry):
        select_cols = tuple(intent.select_cols or [])
    if not any(
        (
            select_cols,
            group_by_cols,
            order_by_cols,
            residual_where,
            residual_having,
            distinct_on,
            distinct_select_index >= 0,
            limit is not None,
            window_registry,
            case_registry,
        )
    ):
        return None
    return ResidualSpec(
        select_cols=select_cols,
        group_by_cols=group_by_cols,
        order_by_cols=order_by_cols,
        where=residual_where,
        having=residual_having,
        distinct_on=distinct_on,
        distinct_select_index=distinct_select_index,
        limit=limit,
        limit_param_key=(intent.limit_param_key or "").strip(),
        window_registry=window_registry,
        case_registry=case_registry,
    )


def _cross_source_join_to_spec(
    join: FederationCrossSourceJoin, manifest: FederationManifest, *, schema: SchemaGraph | None = None
) -> JoinSpec:
    if schema is not None:
        left_ref = resolve_federation_qualified_ref(join.left, manifest=manifest, schema=schema)
        right_ref = resolve_federation_qualified_ref(join.right, manifest=manifest, schema=schema)
        return JoinSpec(
            left_source=left_ref.source_id,
            right_source=right_ref.source_id,
            left_key=left_ref.column,
            right_key=right_ref.column,
            logical_key=join.logical_key,
            kind=join.kind,
        )
    left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
    right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
    left_source = manifest.table_namespace.get(left_tbl, "")
    right_source = manifest.table_namespace.get(right_tbl, "")
    return JoinSpec(
        left_source=left_source,
        right_source=right_source,
        left_key=left_col,
        right_key=right_col,
        logical_key=join.logical_key,
        kind=join.kind,
    )


def _opaque_join_choice_scope_key(scope_index: int) -> str:
    return f"jc{scope_index}"


def _cross_source_join_path_signature(join: FederationCrossSourceJoin) -> list[str]:
    return [f"{join.left}->{join.right}"]


def _cross_source_join_table_pair(
    join: FederationCrossSourceJoin, manifest: FederationManifest | None = None
) -> tuple[str, str]:
    left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
    right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
    return (min(left_tbl, right_tbl), max(left_tbl, right_tbl))


def _cross_source_join_in_scope(
    join: FederationCrossSourceJoin, scope_tables: frozenset[str] | None, manifest: FederationManifest | None = None
) -> bool:
    if not scope_tables:
        return True
    left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
    right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
    return left_tbl in scope_tables and right_tbl in scope_tables


def _is_cross_source_join_key_nonsensitive(
    join: FederationCrossSourceJoin, schema: SchemaGraph, manifest: FederationManifest | None = None
) -> bool:
    for qualified in (join.left, join.right):
        tbl, col = split_qualified_column(qualified, manifest=manifest)
        table = schema.tables.get(tbl)
        if table is None:
            continue
        cm = table.columns.get(col)
        if cm is not None and cm.sensitivity != SensitivityClassification.NONE:
            return False
    return True


def _eligible_cross_source_joins(
    manifest: FederationManifest, sources: frozenset[str], schema: SchemaGraph | None = None
) -> list[tuple[FederationCrossSourceJoin, JoinSpec]]:
    eligible: list[tuple[FederationCrossSourceJoin, JoinSpec]] = []
    for join in manifest.cross_source_joins:
        spec = _cross_source_join_to_spec(join, manifest, schema=schema)
        if spec.left_source not in sources or spec.right_source not in sources:
            continue
        if schema is not None and not _is_cross_source_join_key_nonsensitive(join, schema, manifest):
            continue
        eligible.append((join, spec))
    return eligible


def _deterministic_cross_source_choice(entries: Sequence[tuple[FederationCrossSourceJoin, JoinSpec, str]]) -> str:
    return sorted(entries, key=lambda row: row[2])[0][2]


def _join_specs_for_sources(
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: frozenset[str],
    *,
    schema: SchemaGraph | None = None,
    join_choices: Mapping[str, str] | None = None,
    scope_tables: frozenset[str] | None = None,
) -> tuple[JoinSpec, ...] | None:
    _ = mappings
    if len(sources) < 2:
        return None
    eligible = [
        row
        for row in _eligible_cross_source_joins(manifest, sources, schema)
        if _cross_source_join_in_scope(row[0], scope_tables, manifest)
    ]
    by_pair: dict[tuple[str, str], list[tuple[FederationCrossSourceJoin, JoinSpec, str]]] = {}
    for join, spec in sorted(eligible, key=lambda row: (row[0].logical_key, row[0].left, row[0].right)):
        pair = _cross_source_join_table_pair(join, manifest)
        candidate_id = f"J{len(by_pair.get(pair, [])):02d}"
        by_pair.setdefault(pair, []).append((join, spec, candidate_id))
    if not by_pair:
        return None
    choices = dict(join_choices or {})
    chosen: list[JoinSpec] = []
    for scope_index, pair in enumerate(sorted(by_pair.keys())):
        entries = by_pair[pair]
        scope = _opaque_join_choice_scope_key(scope_index)
        if len(entries) == 1:
            chosen.append(entries[0][1])
            continue
        chosen_id: str | None = choices.get(scope)
        if chosen_id is None:
            chosen_id = _deterministic_cross_source_choice(entries)
        picked: JoinSpec | None = None
        for _join, spec, cid in entries:
            if cid == chosen_id:
                picked = spec
                break
        if picked is None:
            fallback_cid = _deterministic_cross_source_choice(entries)
            for _join, spec, cid in entries:
                if cid == fallback_cid:
                    picked = spec
                    break
        if picked is None:
            picked = entries[0][1]
        chosen.append(picked)
    if chosen and not _join_specs_connect_sources(sources, chosen):
        return None
    return tuple(chosen)


def resolve_federated_combine(
    q_norm: str,
    plan: FederatedPlan,
    manifest: FederationManifest,
    composite_schema: SchemaGraph,
    *,
    preset_choices: Mapping[str, str] | None = None,
    temporal_bind: AnchoredTemporalBind | None = None,
) -> FederatedPlan:
    """Disambiguate declared cross-source joins and refresh coordinator projections."""
    if plan.ineligible_reason or not plan.steps:
        return plan
    if temporal_bind is None and plan.steps:
        parent_intent = plan.steps[0].sub_intent
        temporal_bind = resolve_anchored_temporal_bind(parent_intent)
    sources = frozenset(step.source_id for step in plan.steps)
    eligible = [
        row
        for row in _eligible_cross_source_joins(manifest, sources, composite_schema)
        if _cross_source_join_in_scope(
            row[0], frozenset(table for step in plan.steps for table in (step.sub_intent.tables or [])), manifest
        )
    ]
    by_pair: dict[tuple[str, str], list[tuple[FederationCrossSourceJoin, JoinSpec, str]]] = {}
    for join, spec in sorted(eligible, key=lambda row: (row[0].logical_key, row[0].left, row[0].right)):
        pair = _cross_source_join_table_pair(join, manifest)
        candidate_id = f"J{len(by_pair.get(pair, [])):02d}"
        by_pair.setdefault(pair, []).append((join, spec, candidate_id))
    if not by_pair:
        return replace(plan, combine=None)
    join_choices = dict(preset_choices or {})
    llm_scopes: list[dict[str, Any]] = []
    for scope_index, pair in enumerate(sorted(by_pair.keys())):
        entries = by_pair[pair]
        scope = _opaque_join_choice_scope_key(scope_index)
        if len(entries) == 1:
            join_choices[scope] = entries[0][2]
            continue
        if scope in join_choices:
            continue
        tables = sorted(
            {split_qualified_column(join.left, manifest=manifest)[0] for join, _spec, _cid in entries}
            | {split_qualified_column(join.right, manifest=manifest)[0] for join, _spec, _cid in entries}
        )
        llm_scopes.append(
            {
                "scope": scope,
                "tables": tables,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "join_path_signature": _cross_source_join_path_signature(join),
                    }
                    for join, _spec, candidate_id in entries
                ],
            }
        )
    if llm_scopes:
        resolved = get_join_choice_from_llm(
            q_norm, "SELECT 1", llm_scopes=llm_scopes, preset_choices=join_choices, schema=composite_schema
        )
        join_choices.update(resolved)
    combine = _join_specs_for_sources(
        manifest,
        FederationMappings(version=FEDERATION_MAPPINGS_VERSION),
        sources,
        schema=composite_schema,
        join_choices=join_choices,
        scope_tables=frozenset(table for step in plan.steps for table in (step.sub_intent.tables or [])),
    )
    refreshed_steps: list[SourceStep] = []
    for step in plan.steps:
        join_keys = _join_key_columns_for_source(step.source_id, manifest, chosen_specs=combine)
        output_keys = [key for key in step.projected_keys if key not in join_keys]
        keys = tuple(dict.fromkeys([*join_keys, *output_keys]))
        refreshed_steps.append(replace(step, projected_keys=keys))
    return replace(plan, combine=combine, steps=tuple(refreshed_steps))


def _federation_storage_slug_fragment(raw: str, *, fallback: str) -> str:
    """Return a filesystem-friendly lowercase token for one slug component."""
    token = FEDERATION_STORAGE_SLUG_NON_ALNUM_RE.sub("_", str(raw).strip()).strip("_").lower()
    return token if token else fallback


def _federation_duckdb_schema_from_connection(connection: str) -> str:
    """Map a federation source connection label to the DuckDB schema used for qualification."""
    conn = str(connection or "").strip().lower()
    if conn in {"", "memory", "main"}:
        return "main"
    return conn


def _federation_connection_slug_fields(
    runtime_cls: type[EngineRuntimeConfig], *, engine: str, connection: str
) -> dict[str, str]:
    """Resolve slug fields for a federation binding, honouring an explicit connection handle."""
    conn = str(connection or "").strip()
    fields = dict(runtime_cls.connection_slug_fields())
    if not conn:
        return fields
    slug_keys = runtime_cls.connection_slug_keys()
    if engine == "duckdb" and "schema" in fields:
        fields["schema"] = _federation_duckdb_schema_from_connection(conn)
    elif slug_keys:
        fields[slug_keys[0]] = conn
    return fields


def federation_source_storage_slug(binding: FederationSourceBinding) -> str:
    """Resolve the per-source artifact directory slug for a federation binding."""
    engine = str(binding.engine or "duckdb").strip().lower()
    connection = str(binding.connection or "").strip()
    try:
        runtime_cls = cast(type[EngineRuntimeConfig], get_runtime_config_class(engine))
    except ValueError:
        conn = (connection or engine).strip().lower()
        safe = FEDERATION_CONNECTION_SLUG_NON_WORD_RE.sub("_", conn).strip("_") or "source"
        return f"{FEDERATION_SOURCE_STORAGE_PREFIX}{safe}"[:ENGINE_STORAGE_SLUG_MAX_CHARS]
    fields = _federation_connection_slug_fields(runtime_cls, engine=engine, connection=connection)
    parts = [
        _federation_storage_slug_fragment(fields[key], fallback=key[0]) for key in runtime_cls.connection_slug_keys()
    ]
    slug = f"{FEDERATION_SOURCE_STORAGE_PREFIX}{engine}_" + "_".join(parts)
    if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
        return f"{FEDERATION_SOURCE_STORAGE_PREFIX}{engine}_{digest}"
    return slug


def federation_source_artifacts_dir(artifacts_root: str | None, binding: FederationSourceBinding) -> str:
    """Return the artifact directory for one federation member source."""
    parent = (
        os.path.abspath(os.path.expanduser(str(artifacts_root)))
        if artifacts_root and str(artifacts_root).strip()
        else user_data_dir(appname="aetherdialect", appauthor=False)
    )
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, federation_source_storage_slug(binding))


def _federation_member_schema_graph_path(artifacts_root: str | None, binding: FederationSourceBinding) -> str:
    return os.path.join(federation_source_artifacts_dir(artifacts_root, binding), "schema_graph.json.gz")


def _load_federation_member_schema_graph(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
) -> SchemaGraph:
    """Load one stored member schema graph, surfacing unreadable or unprofiled artifacts."""
    source_id = str(binding.source_id or "").strip()
    path = _federation_member_schema_graph_path(artifacts_root, binding)
    if not os.path.isfile(path):
        raise FederationMemberUnprofilableError(
            f"federation member {source_id!r} stored schema graph is missing at {path!r}",
            source_id=source_id,
        )
    try:
        payload = read_gzip_json(path)
        tables_raw = payload.get("tables", {})
        if not isinstance(tables_raw, dict):
            raise ValueError("tables payload is missing or not an object")
        model: dict[str, Any] = dict(payload)
        if "join_paths_multi" not in model and "join_paths" in model:
            jp = payload.get("join_paths", {})
            join_paths_multi: dict[str, Any] = {}
            for a in jp:
                join_paths_multi[a] = {}
                for b in jp[a]:
                    join_paths_multi[a][b] = [jp[a][b]] if jp[a][b] is not None else []
            model["join_paths_multi"] = join_paths_multi
        graph = SchemaGraph.from_dict(model)
    except FederationMemberUnprofilableError:
        raise
    except Exception as exc:
        raise FederationMemberUnprofilableError(
            f"federation member {source_id!r} stored schema graph at {path!r} could not be loaded: {exc}",
            source_id=source_id,
        ) from exc
    assert_federation_member_graph_profiled(source_id, graph)
    return graph


def load_federation_member_graphs(artifacts_root: str | None, manifest: FederationManifest) -> dict[str, SchemaGraph]:
    """Load per-source schema graphs from member artifact trees when present."""
    graphs: dict[str, SchemaGraph] = {}
    for binding in manifest.sources:
        path = _federation_member_schema_graph_path(artifacts_root, binding)
        if not os.path.isfile(path):
            continue
        graphs[binding.source_id] = _load_federation_member_schema_graph(artifacts_root, binding)
    return graphs


def reconcile_federation_member_graphs(
    live_graphs: Mapping[str, SchemaGraph],
    disk_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
) -> dict[str, SchemaGraph]:
    """Prefer live engine graphs and stamp disk snapshots only as fallback."""
    merged: dict[str, SchemaGraph] = {}
    source_ids = [binding.source_id for binding in manifest.sources] if manifest.sources else sorted(live_graphs)
    for source_id in source_ids:
        live = live_graphs.get(source_id)
        disk = disk_graphs.get(source_id)
        if live is not None:
            merged[source_id] = live
        elif disk is not None:
            merged[source_id] = _stamp_member_graph_source_id(disk, source_id)
    return merged


def assert_federation_member_graph_roster_complete(
    manifest: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph],
) -> None:
    """Raise when stored member graphs cover only part of the declared roster."""
    declared_ids = {binding.source_id for binding in manifest.sources}
    loaded_ids = set(member_graphs)
    if loaded_ids and loaded_ids != declared_ids:
        missing = sorted(declared_ids - loaded_ids)
        raise FederationMemberUnprofilableError(
            f"federation stored member graphs are incomplete; missing graphs for {missing!r}",
            source_id=missing[0] if missing else "",
        )


def _name_similarity(left: str, right: str) -> float:
    a = left.strip().lower()
    b = right.strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return FEDERATION_MAPPING_NAME_SUBSTRING_SCORE
    return 0.0


def _types_compatible(left_type: str, right_type: str) -> bool:
    """Return whether two raw data-type strings are compatible for mapping suggestions. An empty or missing type on either side is incompatible: unknown is not a positive compatibility signal. Callers that need a hard failure (declared join-key validation) must raise themselves; this helper only returns bool."""
    lt = str(left_type or "").strip().lower()
    rt = str(right_type or "").strip().lower()
    if not lt or not rt:
        return False
    if lt == rt:
        return True
    stringish = {"text", "varchar", "string", "char", "character varying"}
    if lt in stringish and rt in stringish:
        return True
    numeric = {"int", "integer", "bigint", "smallint", "number", "numeric", "decimal", "float", "double"}
    if any(token in lt for token in numeric) and any(token in rt for token in numeric):
        return True
    return False


def _value_overlap_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    s1 = {str(v) for v in left if str(v)}
    s2 = {str(v) for v in right if str(v)}
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    return inter / float(min(len(s1), len(s2)))


def _mapping_suggestion_cutoff(*, same_source: bool) -> float:
    if same_source:
        return PolicyConfig.FEDERATION_MAPPING_SUGGESTION_WITHIN_SOURCE_CUTOFF
    return PolicyConfig.FEDERATION_MAPPING_SUGGESTION_CROSS_SOURCE_CUTOFF


def _cross_source_column_suggestion_score(
    left: ColumnMetadata, right: ColumnMetadata, left_name: str, right_name: str
) -> float:
    if not _types_compatible(left.data_type, right.data_type):
        return 0.0
    if left.sensitivity != SensitivityClassification.NONE or right.sensitivity != SensitivityClassification.NONE:
        return 0.0
    name_score = _name_similarity(left_name, right_name)
    overlap = _value_overlap_ratio(left.value_overlap_sample, right.value_overlap_sample)
    if overlap < FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR and name_score < FEDERATION_MAPPING_NAME_SCORE_FLOOR:
        return 0.0
    return min(
        1.0, FEDERATION_MAPPING_SCORE_NAME_WEIGHT * name_score + FEDERATION_MAPPING_SCORE_OVERLAP_WEIGHT * overlap
    )


def suggest_cross_source_mappings(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    *,
    existing_mappings: FederationMappings | None = None,
    max_suggestions: int = 20,
) -> tuple[FederationMappingSuggestion, ...]:
    """Propose cross-source column equivalences; advisory only, never auto-applied."""
    mappings = existing_mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    if mappings.logical_columns or mappings.logical_tables:
        return ()
    suggestions: list[FederationMappingSuggestion] = []
    seen: set[tuple[str, ...]] = set()
    refs: list[tuple[str, str, str, ColumnMetadata]] = []
    source_ids = [binding.source_id for binding in manifest.sources] or sorted(member_graphs)
    for source_id in source_ids:
        graph = member_graphs.get(source_id)
        if graph is None:
            continue
        for phys_name, table in graph.tables.items():
            logical = phys_name
            for col_name, col in table.columns.items():
                if col.sensitivity != SensitivityClassification.NONE:
                    continue
                refs.append((source_id, logical, col_name, col))
    parent = list(range(len(refs)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left_idx, (sid_a, _, c_a, m_a) in enumerate(refs):
        for right_idx in range(left_idx + 1, len(refs)):
            sid_b, _, c_b, m_b = refs[right_idx]
            same_source = sid_a == sid_b
            score = _cross_source_column_suggestion_score(m_a, m_b, c_a, c_b)
            if score < _mapping_suggestion_cutoff(same_source=same_source):
                continue
            _union(left_idx, right_idx)

    clusters: dict[int, list[int]] = {}
    for index in range(len(refs)):
        clusters.setdefault(_find(index), []).append(index)

    for indices in clusters.values():
        by_source: dict[str, int] = {}
        for index in indices:
            source_id = refs[index][0]
            if source_id not in by_source:
                by_source[source_id] = index
        if len(by_source) < 2:
            continue
        chosen = list(by_source.values())
        min_score = 1.0
        for left_pos, left_idx in enumerate(chosen):
            _, _, c_a, m_a = refs[left_idx]
            for right_idx in chosen[left_pos + 1 :]:
                _, _, c_b, m_b = refs[right_idx]
                min_score = min(min_score, _cross_source_column_suggestion_score(m_a, m_b, c_a, c_b))
        if min_score < _mapping_suggestion_cutoff(same_source=False):
            continue
        members = tuple(sorted(f"{refs[index][1]}.{refs[index][2]}" for index in chosen))
        if members in seen:
            continue
        seen.add(members)
        col_names = [refs[index][2] for index in chosen]
        logical = min(col_names, key=lambda name: (len(name), name))
        suggestions.append(
            FederationMappingSuggestion(
                logical=logical, members=members, kind="column", score=min_score, role="join_key"
            )
        )
    suggestions.sort(key=lambda row: (-row.score, row.logical, row.members))
    return tuple(suggestions[: max(1, int(max_suggestions))])


def load_cached_federation_mapping_suggestions(
    federation_dir: str, *, member_tuple_hash_value: str
) -> tuple[FederationMappingSuggestion, ...] | None:
    """Load cached mapping suggestions when *member_tuple_hash_value* matches."""
    path = federation_artifact_paths(federation_dir).get("mapping_suggestions_cache", "")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationConfigError(f"corrupt federation mapping suggestions cache: {path!r}: {exc}") from exc
    if str(payload.get("member_tuple_hash", "") or "") != member_tuple_hash_value:
        return None
    rows = payload.get("suggestions")
    if not isinstance(rows, list):
        return None
    out: list[FederationMappingSuggestion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            FederationMappingSuggestion(
                logical=str(row.get("logical", "") or ""),
                members=tuple(str(m) for m in row.get("members", []) or ()),
                kind=str(row.get("kind", "column") or "column"),
                score=float(row.get("score", 0.0) or 0.0),
                role=str(row.get("role", "") or ""),
            )
        )
    return tuple(out)


def persist_federation_mapping_suggestions_cache(
    federation_dir: str, *, member_tuple_hash_value: str, suggestions: Sequence[FederationMappingSuggestion]
) -> None:
    """Write cached mapping suggestions keyed by member tuple hash."""
    path = federation_artifact_paths(federation_dir).get("mapping_suggestions_cache", "")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "member_tuple_hash": member_tuple_hash_value,
        "suggestions": [
            {
                "logical": row.logical,
                "members": list(row.members),
                "kind": row.kind,
                "score": row.score,
                "role": row.role,
            }
            for row in suggestions
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def cached_or_suggest_cross_source_mappings(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    federation_dir: str | None,
    *,
    existing_mappings: FederationMappings | None = None,
) -> tuple[FederationMappingSuggestion, ...]:
    """Return mapping suggestions, reusing a federation-tree cache when unchanged."""
    member_tuple_hash_value = federation_member_tuple_hash(member_graphs, manifest)
    if federation_dir:
        cached = load_cached_federation_mapping_suggestions(
            federation_dir, member_tuple_hash_value=member_tuple_hash_value
        )
        if cached is not None:
            return cached
    suggestions = suggest_cross_source_mappings(member_graphs, manifest, existing_mappings=existing_mappings)
    if federation_dir:
        persist_federation_mapping_suggestions_cache(
            federation_dir, member_tuple_hash_value=member_tuple_hash_value, suggestions=suggestions
        )
    return suggestions


def export_federation_migration_map_skeleton(
    cwd_path: str, *, dropped_joins: Sequence[tuple[str, str]] | None = None
) -> str:
    """Write ``federation_migration_map.json`` skeleton into *cwd_path*."""
    path = os.path.join(cwd_path, FEDERATION_MIGRATION_MAP_FILENAME)
    payload: dict[str, Any] = {
        "version": 1,
        "action": MIGRATION_MAP_ACTION_REMAP,
        "qualified_column_renames": [],
        "namespace_renames": [],
        "dropped_cross_source_joins": [{"left": left, "right": right} for left, right in (dropped_joins or ())],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.write("\n")
    return path


_quote_ident = sqlglot_quote_identifier
