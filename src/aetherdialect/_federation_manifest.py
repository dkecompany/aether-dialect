"""Federation manifest parse/load/export, capabilities, temporal, and mapping suggestions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from ._config import EngineRuntimeConfig, PolicyConfig
from ._constants import (
    ARTIFACT_DIR_MODE,
    ARTIFACT_MANIFEST_FILENAME,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
    FEDERATION_CONNECTION_SLUG_NON_WORD_RE,
    FEDERATION_CROSS_SOURCE_JOIN_KINDS,
    FEDERATION_DECLARATION_TOP_LEVEL_KEYS,
    FEDERATION_DECLARATION_VERSION,
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
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_MAX_JOIN_CANDIDATE_CAP,
    FEDERATION_MAX_JOIN_PATH_TIE_CAP,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FEDERATION_PLAN_TEMPLATE_FILENAME,
    FEDERATION_SOURCE_STORAGE_PREFIX,
    FEDERATION_STORAGE_SLUG_NON_ALNUM_RE,
    FEDERATION_TEMPLATES_SEGMENT,
    FILE_ENGINE_NAMES,
    MIGRATION_MAP_ACTION_REMAP,
    VALID_HAVING_OPS,
)
from ._contracts_base import (
    ArtifactManifest,
    ConfigError,
    CteEmissionKind,
    DatabaseFeatureCapability,
    EngineContext,
    FederationConfigError,
    FederationContext,
    FederationDeclarationError,
    FederationMemberEngine,
    FederationMemberUnprofilableError,
    HavingParam,
    MemberEffectiveGrants,
    NormalizedExpr,
    PredicateGroup,
    SchemaRole,
    SensitivityClassification,
    WhereParam,
)
from ._contracts_core import (
    AnchoredTemporalBind,
    FederatedPlan,
    RuntimeCteStep,
    RuntimeIntent,
)
from ._contracts_schema import (
    ColumnMetadata,
    FederationCoordinatorConfig,
    FederationCrossSourceJoin,
    FederationManifest,
    FederationMappings,
    FederationMappingSuggestion,
    FederationSourceBinding,
    FederationSourceLimits,
    FederationTableAlias,
    LogicalColumnMapping,
    LogicalTableMapping,
    LogicalTableMember,
    SchemaGraph,
    SQLShape,
)
from ._dialect import (
    DialectRegistry,
)
from ._intent_expr import extract_columns_from_expr
from ._intent_normalize import (
    collect_referenced_tables,
)
from ._schema_finalize import (
    compute_metadata_hash,
)
from ._schema_graph import (
    apply_deny_objects_filter,
    compute_database_feature_capability,
    deny_columns_by_table,
    prune_foreign_keys_after_column_removal,
    recompute_join_paths_multi,
)
from ._schema_profile import (
    description_neutrality_violations,
    value_overlap_ratio_for_columns,
)
from ._schema_reflect import resolve_federation_qualified_ref
from ._sql_gen import (
    generate_col_alias,
)
from ._utils import (
    coerce_format_version,
    effective_structural_hash_fp,
    format_versions_match,
    notify,
)


def _intersect_column_sets_by_table(
    caps: Sequence[DatabaseFeatureCapability],
    field: Literal["aggregatable_columns_by_table", "date_columns_by_table", "array_columns_by_table"],
) -> dict[str, frozenset[str]]:
    """Intersect per-table column name sets across member capability snapshots."""
    if not caps:
        return {}
    common_tables = set(getattr(caps[0], field).keys())
    for cap in caps[1:]:
        common_tables &= set(getattr(cap, field).keys())
    merged: dict[str, set[str]] = {}
    for cap in caps:
        table_map = getattr(cap, field)
        for table_name in common_tables:
            cols = table_map.get(table_name)
            if cols is None:
                continue
            merged.setdefault(table_name, set(cols)).intersection_update(cols)
    return {table_name: frozenset(cols) for table_name, cols in merged.items()}


def intersect_member_database_feature_capabilities(
    member_graphs: Mapping[str, SchemaGraph],
) -> DatabaseFeatureCapability:
    """Derive federation structural capability as the intersection of member graphs."""
    caps = [graph.database_feature_capability for graph in member_graphs.values()]
    if not caps:
        return compute_database_feature_capability(SchemaGraph(tables={}, join_paths_multi={}))
    if len(caps) == 1:
        return caps[0]
    self_join_tables = caps[0].tables_supporting_self_join
    for cap in caps[1:]:
        self_join_tables &= cap.tables_supporting_self_join
    return DatabaseFeatureCapability(
        table_count=min(cap.table_count for cap in caps),
        fk_edge_count=min(cap.fk_edge_count for cap in caps),
        has_numeric_measures=all(cap.has_numeric_measures for cap in caps),
        has_date_columns=all(cap.has_date_columns for cap in caps),
        has_array_columns=all(cap.has_array_columns for cap in caps),
        has_categorical_columns=all(cap.has_categorical_columns for cap in caps),
        max_tables_on_any_join_path=min(cap.max_tables_on_any_join_path for cap in caps),
        max_fk_chain_depth=min(cap.max_fk_chain_depth for cap in caps),
        has_self_referential_fk=all(cap.has_self_referential_fk for cap in caps),
        tables_supporting_self_join=frozenset(self_join_tables),
        has_window_capable_table_sets=all(cap.has_window_capable_table_sets for cap in caps),
        aggregatable_columns_by_table=_intersect_column_sets_by_table(caps, "aggregatable_columns_by_table"),
        date_columns_by_table=_intersect_column_sets_by_table(caps, "date_columns_by_table"),
        array_columns_by_table=_intersect_column_sets_by_table(caps, "array_columns_by_table"),
        supports_semi_join=all(cap.supports_semi_join for cap in caps),
        supports_anti_join=all(cap.supports_anti_join for cap in caps),
        supports_predicate_nesting=all(cap.supports_predicate_nesting for cap in caps),
        supports_preserve_tables=all(cap.supports_preserve_tables for cap in caps),
        supports_ordered_string_agg=all(cap.supports_ordered_string_agg for cap in caps),
        supports_median=all(cap.supports_median for cap in caps),
        supports_stddev=all(cap.supports_stddev for cap in caps),
        supports_variance=all(cap.supports_variance for cap in caps),
        supports_window_frames=all(cap.supports_window_frames for cap in caps),
        supports_array_contains=all(cap.supports_array_contains for cap in caps),
        supports_collation=all(cap.supports_collation for cap in caps),
        supports_unsigned_semantics=all(cap.supports_unsigned_semantics for cap in caps),
        supports_timestamptz_semantics=all(cap.supports_timestamptz_semantics for cap in caps),
    )


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
            extra_sets.append(DialectRegistry.extra_where_ops_for_engine(engine_type))
    if extra_sets:
        shared_extra = set(extra_sets[0])
        for extra in extra_sets[1:]:
            shared_extra &= set(extra)
        allowed.update(shared_extra)
    if "ilike" not in allowed:
        if dialects_by_source:
            dialects = [d for d in dialects_by_source.values() if d is not None]
            if dialects and all(DialectRegistry.dialect_supports_ilike_semantics(d) for d in dialects):
                allowed.update({"ilike", "not ilike"})
        elif engine_types_by_source:
            if all(
                DialectRegistry.member_supports_ilike_semantics(engine_type)
                for engine_type in engine_types_by_source.values()
            ):
                allowed.update({"ilike", "not ilike"})
    if dialects_by_source:
        if all(bool(getattr(d, "supports_array_contains", True)) for d in dialects_by_source.values() if d is not None):
            allowed.add("contains")
    elif engine_types_by_source:
        if all(
            DialectRegistry.engine_supports_array_contains(engine_type)
            for engine_type in engine_types_by_source.values()
        ):
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
        "supports_ordered_string_agg": _member_flag(
            "supports_ordered_string_agg", DialectRegistry.engine_supports_ordered_string_agg
        ),
        "supports_median": _member_flag("supports_median", DialectRegistry.engine_supports_median),
        "supports_stddev": _member_flag("supports_stddev", DialectRegistry.engine_supports_stddev),
        "supports_variance": _member_flag("supports_variance", DialectRegistry.engine_supports_variance),
        "supports_window_frames": _member_flag("supports_window_frames", DialectRegistry.engine_supports_window_frames),
        "supports_array_contains": _member_flag(
            "supports_array_contains", DialectRegistry.engine_supports_array_contains
        ),
        "supports_collation": _member_flag("supports_collation", DialectRegistry.engine_supports_collation),
        "supports_unsigned_semantics": _member_flag(
            "supports_unsigned_semantics", DialectRegistry.engine_supports_unsigned_semantics
        ),
        "supports_timestamptz_semantics": _member_flag(
            "supports_timestamptz_semantics", DialectRegistry.engine_supports_timestamptz_semantics
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
        if DialectRegistry.dialect_supports_ilike_semantics(dialect):
            allowed.update({"ilike", "not ilike"})
    else:
        allowed.update(DialectRegistry.extra_where_ops_for_engine(binding.engine))
        if DialectRegistry.member_supports_ilike_semantics(binding.engine):
            allowed.update({"ilike", "not ilike"})
    if DialectRegistry.engine_supports_array_contains(binding.engine):
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
            if not DialectRegistry.dialect_supports_ilike_semantics(dialect):
                return binding.source_id
            continue
        if not DialectRegistry.member_supports_ilike_semantics(binding.engine):
            return binding.source_id
    return None


def federation_unsupported_operator_reason(
    intent: RuntimeIntent, manifest: FederationManifest, *, dialects_by_source: Mapping[str, Any] | None = None
) -> str | None:
    """Refuse operators absent from the intersection of member dialect capabilities."""
    engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
    if not engine_types and not dialects_by_source:
        return None
    caps = intersect_member_dialect_capabilities(dialects_by_source, engine_types_by_source=engine_types or None)
    allowed_where = caps.get("where_ops") or frozenset()
    allowed_having = caps["having_ops"]
    for fp in PredicateGroup.where_leaves(intent.where) or []:
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
    for hp in PredicateGroup.having_leaves(intent.having) or []:
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
    for fp in PredicateGroup.where_leaves(intent.where):
        if (fp.op or "").strip().lower() == "contains":
            return True
    for hp in PredicateGroup.having_leaves(intent.having):
        if (hp.op or "").strip().lower() == "contains":
            return True
    return False


def _expr_mentions_collation(expr: NormalizedExpr | None) -> bool:
    if expr is None:
        return False
    if (expr.scalar_func or "").strip().lower() == "collate":
        return True
    raw = str(expr.raw_sql or "")
    if "collate" in raw.lower():
        return True
    for group in (*expr.add_groups, *expr.sub_groups):
        for mult in (*group.multiply, *group.divide):
            if _expr_mentions_collation(mult):
                return True
    return False


def _intent_uses_collation(intent: RuntimeIntent) -> bool:
    for obc in intent.order_by_cols or []:
        if _expr_mentions_collation(obc.expr):
            return True
    for sc in intent.select_cols or []:
        if _expr_mentions_collation(sc.expr):
            return True
    for col in intent.group_by_cols or []:
        if _expr_mentions_collation(col):
            return True
    for cte in intent.cte_steps or []:
        for obc in cte.order_by_cols or []:
            if _expr_mentions_collation(obc.expr):
                return True
    return False


def _intent_column_data_types(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    *,
    predicate: Callable[[str], bool] | None = None,
    column_predicate: Callable[[ColumnMetadata], bool] | None = None,
) -> bool:
    refs: set[str] = set()
    for sc in intent.select_cols or []:
        refs.update(extract_columns_from_expr(sc.expr))
    for col in intent.group_by_cols or []:
        refs.update(extract_columns_from_expr(col))
    for obc in intent.order_by_cols or []:
        refs.update(extract_columns_from_expr(obc.expr))
    for fp in PredicateGroup.where_leaves(intent.where):
        refs.update(extract_columns_from_expr(fp.left_expr))
    for hp in PredicateGroup.having_leaves(intent.having):
        refs.update(extract_columns_from_expr(hp.left_expr))
    for cref in refs:
        if "." not in cref:
            continue
        table_name, column_name = cref.rsplit(".", 1)
        table = schema.tables.get(table_name)
        if table is None:
            continue
        col_meta = table.columns.get(column_name)
        if col_meta is None:
            continue
        if column_predicate is not None and column_predicate(col_meta):
            return True
        if predicate is not None and predicate(str(col_meta.data_type or "")):
            return True
    return False


def federation_ir_capability_reason(
    intent: RuntimeIntent,
    cap: DatabaseFeatureCapability,
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse IR shapes absent from the intersection of member capabilities."""
    for cte in intent.cte_steps or []:
        emission = CteEmissionKind.coerce(getattr(cte, "emission", "join_table"))
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
    if _intent_uses_collation(intent) and not cap.supports_collation:
        return "collation is not supported by all federation members"
    if schema is not None:
        if not cap.supports_timestamptz_semantics and _intent_column_data_types(
            intent, schema, column_predicate=lambda col: col.is_timezone_aware
        ):
            return "timestamptz semantics are not supported by all federation members"
        if not cap.supports_unsigned_semantics and _intent_column_data_types(
            intent, schema, column_predicate=lambda col: col.is_unsigned
        ):
            return "unsigned integer semantics are not supported by all federation members"
    nested_where = bool(intent.where and intent.where.depth() > 1)
    nested_having = bool(intent.having and intent.having.depth() > 1)
    if (nested_where or nested_having) and not cap.supports_predicate_nesting:
        return "nested predicate groups are not supported by all federation members"
    return None


def predicate_param_sources(
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


def predicate_group_spans_sources(
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
            srcs = predicate_param_sources(
                pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if len(srcs) > 1:
                return f"cross-source OR filter is not supported: {predicate_clause_label(pred)}"
        branch_sources: list[set[str]] = []
        for pred in group.predicates:
            srcs = predicate_param_sources(
                pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if srcs:
                branch_sources.append(srcs)
        for child in group.groups:
            child_reason = predicate_group_spans_sources(
                child, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
            )
            if child_reason:
                return child_reason
            child_srcs: set[str] = set()
            for pred in child.leaves():
                child_srcs |= predicate_param_sources(
                    pred, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
                )
            if child_srcs:
                branch_sources.append(child_srcs)
        touched = {src for srcs in branch_sources for src in srcs}
        if len(touched) > 1 and len(branch_sources) > 1:
            labels = []
            for pred in group.predicates:
                labels.append(predicate_clause_label(pred))
            for child in group.groups:
                child_labels = " AND ".join(predicate_clause_label(pred) for pred in child.leaves())
                if child_labels:
                    labels.append(child_labels)
            joined = " OR ".join(label for label in labels if label)
            return f"cross-source predicate disjunction is not supported: {joined}"
    for child in group.groups:
        child_reason = predicate_group_spans_sources(
            child, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
        )
        if child_reason:
            return child_reason
    return None


def cte_probe_join_keys(cte: RuntimeCteStep) -> list[str]:
    """Return qualified join-key column refs projected by a semi/anti probe CTE."""
    keys: list[str] = []
    for sc in cte.select_cols or []:
        col = (sc.expr.column_ref or sc.expr.primary_column or sc.expr.primary_term or "").strip()
        if col and "." in col and col not in keys:
            keys.append(col)
    return keys


def cross_source_probe_cte_steps(
    intent: RuntimeIntent, source_by_table: Mapping[str, str]
) -> tuple[RuntimeCteStep, ...]:
    """Return semi/anti probe CTEs whose body lives on a different member than the driver tables."""
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return ()
    owners = assign_cte_sources(cte_steps, source_by_table)
    driver_tables = set(intent.tables or [])
    for cte in cte_steps:
        if cte.cte_name:
            driver_tables.discard(cte.cte_name)
    driver_sources = {source_by_table.get(table, "") for table in driver_tables if source_by_table.get(table, "")}
    lifted: list[RuntimeCteStep] = []
    for cte in cte_steps:
        emission = CteEmissionKind.coerce(getattr(cte, "emission", "join_table"))
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
            PredicateGroup.where_leaves(intent.where),
            PredicateGroup.having_leaves(intent.having),
            window_registry=intent.window_registry,
            case_registry=intent.case_registry,
            include_unreferenced_registries=False,
        )
        if cte.cte_name and cte.cte_name in refs:
            lifted.append(cte)
    return tuple(lifted)


def cross_source_probe_cte_ineligible_reason(
    intent: RuntimeIntent, manifest: FederationManifest, source_by_table: Mapping[str, str]
) -> str | None:
    """Refuse cross-source semi/anti probes that cannot be lifted to the coordinator."""
    for cte in cross_source_probe_cte_steps(intent, source_by_table):
        keys = cte_probe_join_keys(cte)
        if not keys:
            name = (cte.cte_name or "").strip() or "probe"
            return f"cross-source {CteEmissionKind.coerce(getattr(cte, 'emission', 'join_table'))} requires declared join keys: {name}"
        if not manifest.cross_source_joins:
            name = (cte.cte_name or "").strip() or "probe"
            return f"cross-source {CteEmissionKind.coerce(getattr(cte, 'emission', 'join_table'))} requires declared join: {name}"
        covered = False
        key_cols = set(keys)
        for join in manifest.cross_source_joins:
            if {join.left, join.right} & key_cols:
                covered = True
                break
        if not covered:
            name = (cte.cte_name or "").strip() or "probe"
            emission = CteEmissionKind.coerce(getattr(cte, "emission", "join_table"))
            return f"cross-source {emission} requires declared join keys: {name}"
    return None


def distinct_on_spans_sources(
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
    registry_kw = intent_registry_kw(intent)
    refs = collect_referenced_tables([], [], [], [], [], **registry_kw)
    for expr in intent.distinct_on:
        refs |= collect_referenced_tables([], [], [expr], [], [], **registry_kw)
    srcs = sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
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


def manifest_engine_for_source(manifest: FederationManifest, source_id: str) -> str:
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


def intent_registry_kw(intent: RuntimeIntent) -> dict[str, Any]:
    """Return window/case registry kwargs for intent repair collectors."""
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
        num_where += len((PredicateGroup.where_leaves(sub.where)) or [])
        num_having += len((PredicateGroup.having_leaves(sub.having)) or [])
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
        num_where += len(PredicateGroup.where_leaves(residual.where))
        num_having += len(PredicateGroup.having_leaves(residual.having))
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
    for fp in PredicateGroup.where_leaves(intent.where) or ():
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
    for hp in PredicateGroup.having_leaves(intent.having) or ():
        if str(getattr(hp, "value_type", "") or "") == "date_window":
            return True
    return False


def resolve_anchored_temporal_bind(
    intent: RuntimeIntent, *, anchor: datetime | None = None
) -> AnchoredTemporalBind | None:
    """Resolve a temporal reference once for federated member rendering. When the parent intent carries relative date-window or clock-keyword predicates, bind them to a single anchor at turn start so each member statement uses the same instant rather than re-evaluating per-member clock functions."""
    if not _intent_has_temporal_anchor_refs(intent):
        return None
    anchor_dt = anchor or datetime.now(UTC)
    return AnchoredTemporalBind(anchor_iso=anchor_dt.isoformat())


def _manifest_binding_timezones(manifest: FederationManifest, source_ids: frozenset[str]) -> dict[str, str | None]:
    return {
        binding.source_id: binding.session_timezone for binding in manifest.sources if binding.source_id in source_ids
    }


def _distinct_member_timezones(tz_by_source: Mapping[str, str | None]) -> set[str]:
    return {tz for tz in tz_by_source.values() if tz}


def _temporal_predicate_columns(intent: RuntimeIntent) -> list[str]:
    columns: list[str] = []
    for fp in PredicateGroup.where_leaves(intent.where) or ():
        if str(getattr(fp, "value_type", "") or "") != "date_window":
            continue
        left = getattr(fp, "left_expr", None)
        if left is None:
            continue
        col = str(getattr(left, "column_ref", "") or getattr(left, "column", "") or "").strip()
        if col:
            columns.append(col.rsplit(".", 1)[-1])
    for hp in PredicateGroup.having_leaves(intent.having) or ():
        if str(getattr(hp, "value_type", "") or "") != "date_window":
            continue
        left = getattr(hp, "left_expr", None)
        if left is None:
            continue
        col = str(getattr(left, "column_ref", "") or getattr(left, "column", "") or "").strip()
        if col:
            columns.append(col.rsplit(".", 1)[-1])
    return columns


def _cross_source_temporal_join_columns(
    manifest: FederationManifest,
    plan: FederatedPlan,
    schema: SchemaGraph | None,
) -> list[str]:
    if schema is None or not plan.combine:
        return []
    source_ids = frozenset(step.source_id for step in plan.steps)
    if len(source_ids) < 2:
        return []
    columns: list[str] = []
    for join in manifest.cross_source_joins:
        left_ref = resolve_federation_qualified_ref(join.left, manifest=manifest)
        right_ref = resolve_federation_qualified_ref(join.right, manifest=manifest)
        if left_ref.source_id not in source_ids or right_ref.source_id not in source_ids:
            continue
        temporal = False
        for ref in (left_ref, right_ref):
            table = schema.tables.get(ref.table)
            if table is None:
                continue
            meta = table.columns.get(ref.column)
            if meta is None:
                continue
            value_type = (meta.value_type or "").lower()
            base = (meta.data_type or "").split("(", 1)[0].strip().lower()
            if value_type in {"date", "timestamp", "temporal"} or base in {
                "timestamp",
                "timestamptz",
                "datetime",
                "datetimeoffset",
                "date",
            }:
                temporal = True
                break
        if temporal:
            columns.append(str(join.logical_key or left_ref.column or right_ref.column))
    return columns


def emit_federation_member_timezone_mismatch_diagnostics(
    manifest: FederationManifest | None,
    plan: FederatedPlan,
    *,
    schema: SchemaGraph | None = None,
) -> None:
    """Emit diagnostics when temporal work spans members with different session time zones."""
    if manifest is None or plan.ineligible_reason or not plan.steps:
        return
    source_ids = frozenset(step.source_id for step in plan.steps)
    tz_by_source = _manifest_binding_timezones(manifest, source_ids)
    distinct_tz = _distinct_member_timezones(tz_by_source)
    if len(distinct_tz) < 2:
        return
    parent_intent = plan.steps[0].sub_intent
    logical_columns: list[str] = []
    if _intent_has_temporal_anchor_refs(parent_intent):
        logical_columns.extend(_temporal_predicate_columns(parent_intent))
    logical_columns.extend(_cross_source_temporal_join_columns(manifest, plan, schema))
    seen: set[str] = set()
    for logical_column in logical_columns:
        key = str(logical_column or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        notify(
            (
                f"federation members use different session time zones for temporal column {key!r}; "
                "coordinator transfer normalises timestamps to UTC"
            ),
            stage="federation",
            code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH,
            level="info",
            source_id="composite",
            details=(
                ("logical_column", key),
                ("phase", "prepare"),
                ("timezones", ",".join(sorted(distinct_tz))),
            ),
        )


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
        if engine not in DialectRegistry.list_engines():
            raise FederationConfigError(f"source {source_id!r} references unknown engine {engine!r}")
        connection = str(entry.get("connection", "") or "").strip()
        context = str(entry.get("context", "master") or "master").strip().lower() or "master"
        role_val = entry.get("role", "owner")
        try:
            role = SchemaRole.coerce(role_val)
        except ValueError as exc:
            raise FederationConfigError(f"source {source_id!r} has invalid role: {role_val!r}") from exc
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
        session_timezone_raw = entry.get("session_timezone")
        session_timezone = str(session_timezone_raw).strip() if session_timezone_raw else None
        sources.append(
            FederationSourceBinding(
                source_id=source_id,
                engine=engine,
                connection=connection,
                context=context,
                role=role,
                limits=limits,
                session_timezone=session_timezone,
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
                "role": source.role.value if isinstance(source.role, SchemaRole) else str(source.role),
                "session_timezone": source.session_timezone,
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
    version = coerce_format_version(payload.get("version", FEDERATION_MAPPINGS_VERSION))
    if not format_versions_match(version, FEDERATION_MAPPINGS_VERSION):
        raise FederationConfigError(
            f"unsupported federation mappings version {version!r}; this build expects {FEDERATION_MAPPINGS_VERSION!r}"
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
    declared_version = coerce_format_version(payload.get("version", FEDERATION_DECLARATION_VERSION))
    if not format_versions_match(declared_version, FEDERATION_DECLARATION_VERSION):
        raise FederationDeclarationError(
            f"unsupported federation declaration version {declared_version!r}; "
            f"this build expects {FEDERATION_DECLARATION_VERSION!r}"
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


def namespace_from_composite_schema(schema: SchemaGraph) -> dict[str, str]:
    namespace: dict[str, str] = {}
    for table_name, table in schema.tables.items():
        logical = str(table.name or table_name).strip()
        source_id = str(table.source_id or "").strip()
        if logical and source_id:
            namespace[logical] = source_id
    return namespace


def manifest_with_derived_roster(
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
        namespace = manifest.table_namespace or namespace_from_composite_schema(composite)
        source_ids = sorted({sid for sid in namespace.values() if sid})
    else:
        return manifest
    sources = manifest.sources or tuple(
        FederationSourceBinding(
            source_id=str(source_id),
            engine="duckdb",
            connection=str(source_id),
            context="master",
            role=SchemaRole.OWNER,
        )
        for source_id in source_ids
    )
    return replace(manifest, sources=sources, table_namespace=namespace)


def stamp_member_graph_source_id(graph: SchemaGraph, source_id: str) -> SchemaGraph:
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
        stamped = stamp_member_graph_source_id(graph, str(connection_name))
        assert_federation_member_graph_profiled(str(connection_name), stamped)
        grants = introspect_member_effective_grants(engine)
        if grants is not None:
            object.__setattr__(stamped, "_member_effective_grants", grants)
        graphs[str(connection_name)] = stamped
    return graphs


def _assert_federation_member_role_is_owner(connection_name: str, role: SchemaRole) -> None:
    """Refuse federation members that are not owner engines."""
    if role != SchemaRole.OWNER:
        raise FederationConfigError(f"federation member {connection_name!r} must be an owner engine; got role {role!r}")


def member_connection_name_from_engine(engine: FederationMemberEngine) -> str:
    """Return the connection name that identifies a federation member (TOML sub-block or mapping ``name``)."""
    raw_named = getattr(engine, "_named_connection", None)
    if isinstance(raw_named, str) and raw_named.strip():
        return raw_named.strip()
    raw_connection = getattr(engine, "_connection", None)
    if isinstance(raw_connection, str) and raw_connection.strip():
        return raw_connection.strip()
    mapping = getattr(engine, "_connection_mapping", None)
    if isinstance(mapping, Mapping):
        raw_name = mapping.get("name")
        if raw_name is not None and str(raw_name).strip():
            return str(raw_name).strip()
    raise FederationConfigError("federation member engine has no connection name")


def federation_members_mapping(members: Sequence[FederationMemberEngine] | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize federation members to a ``source_id -> engine`` mapping."""
    if isinstance(members, Mapping):
        member_dict = {str(key): engine for key, engine in members.items()}
    else:
        member_dict = {}
        for engine in members:
            source_id = member_connection_name_from_engine(engine)
            if source_id in member_dict:
                raise FederationConfigError(f"duplicate federation member connection name: {source_id!r}")
            member_dict[source_id] = engine
    if len(member_dict) < 2:
        raise ConfigError("AetherFederation requires at least two member engines")
    return member_dict


def binding_from_member_engine(
    engine: FederationMemberEngine,
    *,
    require_owner: bool = True,
) -> FederationSourceBinding:
    """Derive a federation source binding from a configured member engine. Owner federation create/persist requires owner member engines. Consumer federation open may pass consumer members for privilege probe and execution; pass ``require_owner=False``."""
    engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
    if not engine_type:
        runtime_cfg = getattr(engine, "_runtime_config", None)
        engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
    source_id = member_connection_name_from_engine(engine)
    raw_named = getattr(engine, "_named_connection", None)
    named_connection = raw_named.strip() if isinstance(raw_named, str) else ""
    connection = named_connection or source_id
    context = str(getattr(engine, "_context_name", "master") or "master").strip().lower() or "master"
    try:
        role = SchemaRole.coerce(getattr(engine, "_schema_role", "owner"))
    except ValueError as exc:
        raise FederationConfigError(
            f"member {source_id!r} has invalid role: {getattr(engine, '_schema_role', None)!r}"
        ) from exc
    if require_owner:
        _assert_federation_member_role_is_owner(source_id, role)
    session_timezone_raw = getattr(engine, "_session_timezone", None)
    session_timezone = str(session_timezone_raw).strip() if session_timezone_raw else None
    return FederationSourceBinding(
        source_id=source_id,
        engine=engine_type or "duckdb",
        connection=connection,
        context=context,
        role=role,
        session_timezone=session_timezone or None,
    )


def build_federation_manifest_from_members(
    members: Mapping[str, Any],
    *,
    declaration: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    mappings: FederationMappings | None = None,
    require_owner_members: bool = True,
) -> FederationManifest:
    """Merge an authored federation declaration with member-derived roster fields."""
    fed_id = str(declaration.federation_id or "").strip()
    if not fed_id:
        raise FederationConfigError("federation manifest requires federation_id")
    if len(members) < 2:
        raise FederationConfigError("federation requires at least two member engines")
    graphs = dict(member_graphs) if member_graphs is not None else member_graphs_from_engines(members)
    if not graphs:
        raise FederationConfigError("federation requires member schema graphs")
    sources = tuple(
        binding_from_member_engine(engine, require_owner=require_owner_members)
        for engine in sorted(members.values(), key=member_connection_name_from_engine)
    )
    if sources and all((binding.engine or "").strip().lower() in FILE_ENGINE_NAMES for binding in sources):
        raise FederationDeclarationError(
            "A federation whose members are all file engines is not supported; load uploads into one CSV engine instead."
        )
    validate_federation_source_slug_uniqueness(sources, federation_id=fed_id)
    member_ids = set(members)
    for alias in declaration.aliases:
        if alias.source not in member_ids:
            raise FederationConfigError(f"alias {alias.alias!r} references unknown source_id: {alias.source!r}")
    return manifest_with_derived_roster(
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


def plan_template_row_steps_reference_sources(row: Mapping[str, Any], source_ids: set[str]) -> bool:
    """Return True when any stored step fingerprint references a removed source."""
    steps_raw = row.get("step_fingerprints", [])
    if isinstance(steps_raw, list):
        for entry in steps_raw:
            if isinstance(entry, (list, tuple)) and entry and str(entry[0]) in source_ids:
                return True
    return False


def _plan_template_row_references_sources(row: Mapping[str, Any], source_ids: set[str]) -> bool:
    """Return True when a stored plan template row should be dropped for *source_ids*."""
    if plan_template_row_steps_reference_sources(row, source_ids):
        return True
    member_ids_raw = row.get("member_template_ids", [])
    if isinstance(member_ids_raw, list):
        for entry in member_ids_raw:
            if isinstance(entry, (list, tuple)) and entry and str(entry[0]) in source_ids:
                return True
    return False


def sanitize_plan_template_row_member_template_ids(
    row: dict[str, Any], removed_source_ids: set[str]
) -> dict[str, Any] | None:
    """Drop removed-source entries from ``member_template_ids`` on a surviving plan row."""
    member_ids_raw = row.get("member_template_ids", [])
    if not isinstance(member_ids_raw, list) or not member_ids_raw:
        return None
    kept = [
        list(entry)
        for entry in member_ids_raw
        if not (isinstance(entry, (list, tuple)) and entry and str(entry[0]) in removed_source_ids)
    ]
    if len(kept) == len(member_ids_raw):
        return None
    updated = dict(row)
    updated["member_template_ids"] = kept
    return updated


def federation_drifted_member_source_ids(
    federation_dir: str,
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
) -> set[str]:
    """Return member source ids whose pinned identity does not match stored artifacts."""
    stored = load_federation_artifact_manifest_dict(federation_artifact_paths(federation_dir)["artifact_manifest"])
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
            normalized = normalize_stored_member_hash_row(entry)
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


def engine_context_for_schema_usability(ctx: FederationContext | EngineContext) -> EngineContext:
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
        notes=ctx.notes,
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
    tokens = frozenset(str(source_id or "").strip() for source_id in source_ids) - frozenset({""})
    if not tokens:
        return
    for graph in member_graphs.values():
        for table in graph.tables.values():
            targets = [table, *table.columns.values()]
            for target in targets:
                desc = str(getattr(target, "description", "") or "")
                if not desc:
                    continue
                hits = description_neutrality_violations(desc, tokens)
                if hits:
                    raise ConfigError(f"{context} must not name a source or member; found {hits[0]!r}")


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


def apply_composite_federation_scope(composite: SchemaGraph, scope_ctx: FederationContext | EngineContext) -> None:
    """Apply federation master-scope denials to the composite catalog."""
    usability_ctx = engine_context_for_schema_usability(scope_ctx)
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


def _validate_federation_source_id_identifier(source_id: str) -> None:
    """Require member source ids to be safe unquoted SQL identifiers."""
    if not source_id.isidentifier():
        raise FederationDeclarationError(f"federation source_id must be identifier-safe: {source_id!r}")


def validate_federation_source_slug_uniqueness(
    sources: Sequence[FederationSourceBinding],
    *,
    federation_id: str | None = None,
) -> None:
    """Raise when two members resolve to the same artifact storage slug."""
    seen: dict[str, str] = {}
    for binding in sources:
        slug = federation_source_storage_slug(binding, federation_id=federation_id)
        prior = seen.get(slug)
        if prior is not None and prior != binding.source_id:
            raise FederationConfigError(
                f"federation members {prior!r} and {binding.source_id!r} resolve to the same connection slug {slug!r}"
            )
        seen[slug] = binding.source_id


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


def normalize_stored_member_hash_row(row: Sequence[Any]) -> tuple[str, str, str, str, str, str]:
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


def cross_source_join_hash_entry(join: FederationCrossSourceJoin) -> dict[str, str]:
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


def manifest_hash(manifest: FederationManifest) -> str:
    """Stable hash of a normalized manifest for composite identity."""
    payload = {
        "federation_id": manifest.federation_id,
        "aliases": [
            {"alias": alias.alias, "source": alias.source, "table": alias.table}
            for alias in sorted(manifest.aliases, key=lambda a: (a.alias, a.source, a.table))
        ],
        "cross_source_joins": [
            cross_source_join_hash_entry(j)
            for j in sorted(
                manifest.cross_source_joins,
                key=lambda j: (*cross_source_join_hash_entry(j).values(),),
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


def load_federation_artifact_manifest_dict(manifest_path: str) -> dict[str, Any] | None:
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


def sources_for_table(
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


def sources_for_refs(
    refs: Iterable[str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> set[str]:
    sources: set[str] = set()
    for table in refs:
        sources.update(sources_for_table(table, manifest, mappings, source_by_table, schema))
    return sources


def assign_cte_sources(cte_steps: Sequence[RuntimeCteStep], source_by_table: Mapping[str, str]) -> dict[str, str]:
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
            PredicateGroup.where_leaves(cte.where),
            PredicateGroup.having_leaves(cte.having),
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


def _expr_clause_label(expr: NormalizedExpr | None) -> str:
    if expr is None:
        return ""
    return (expr.primary_term or expr.column_ref or expr.primary_column or "").strip()


def predicate_clause_label(param: WhereParam | HavingParam) -> str:
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
    return sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)


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
    fields = dict(runtime_cls().connection_slug_fields())
    if not conn:
        return fields
    slug_keys = runtime_cls.connection_slug_keys()
    if engine == "duckdb" and "schema" in fields:
        fields["schema"] = _federation_duckdb_schema_from_connection(conn)
    elif slug_keys:
        fields[slug_keys[0]] = conn
    return fields


def federation_source_storage_slug(
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
) -> str:
    """Resolve the per-source artifact directory slug for a federation binding."""
    fed_id = str(federation_id or "").strip()
    source_id = str(binding.source_id or "").strip()
    if fed_id and source_id:
        safe_fed = _federation_storage_slug_fragment(fed_id, fallback="fed")
        safe_src = _federation_storage_slug_fragment(source_id, fallback="src")
        slug = f"{FEDERATION_SOURCE_STORAGE_PREFIX}{safe_fed}_{safe_src}"
        if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
            digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
            slug = f"{FEDERATION_SOURCE_STORAGE_PREFIX}{safe_fed}_{digest}"
        return slug
    return engine_connection_federation_source_storage_slug(binding)


def engine_connection_federation_source_storage_slug(binding: FederationSourceBinding) -> str:
    """Resolve the engine/connection artifact slug for a federation binding."""
    engine = str(binding.engine or "duckdb").strip().lower()
    connection = str(binding.connection or "").strip()
    try:
        runtime_cls = cast(type[EngineRuntimeConfig], DialectRegistry.get_runtime_config_class(engine))
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


def _value_overlap_ratio_for_columns(left: ColumnMetadata, right: ColumnMetadata) -> float:
    return value_overlap_ratio_for_columns(left, right)


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
    overlap = _value_overlap_ratio_for_columns(left, right)
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
    os.makedirs(os.path.dirname(path), mode=ARTIFACT_DIR_MODE, exist_ok=True)
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


def build_federation_migration_map_document(
    *,
    dropped_joins: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a federation migration map document for operator review or programmatic apply."""
    return {
        "version": 1,
        "action": MIGRATION_MAP_ACTION_REMAP,
        "qualified_column_renames": [],
        "namespace_renames": [],
        "dropped_cross_source_joins": [{"left": left, "right": right} for left, right in (dropped_joins or ())],
    }


def export_federation_migration_map_skeleton(
    cwd_path: str, *, dropped_joins: Sequence[tuple[str, str]] | None = None
) -> str:
    """Write ``federation_migration_map.json`` skeleton into *cwd_path*."""
    path = os.path.join(cwd_path, FEDERATION_MIGRATION_MAP_FILENAME)
    payload = build_federation_migration_map_document(dropped_joins=dropped_joins)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.write("\n")
    return path


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


def federation_persist_quad_coherent(federation_dir: str) -> bool:
    """Return True when the four core federation artifacts form one committed set."""
    paths = federation_artifact_paths(federation_dir)
    keys = ("manifest", "mappings", "composite_schema", "artifact_manifest")
    present = [key for key in keys if os.path.isfile(paths[key])]
    if not present:
        return True
    if len(present) != len(keys):
        return False
    stored = load_federation_artifact_manifest_dict(paths["artifact_manifest"])
    if stored is None:
        return False
    try:
        with open(paths["manifest"], encoding="utf-8") as handle:
            manifest_payload = json.load(handle)
        with open(paths["mappings"], encoding="utf-8") as handle:
            mappings_payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest_payload, dict) or not isinstance(mappings_payload, dict):
        return False
    try:
        manifest = parse_federation_manifest(manifest_payload)
        mappings = parse_federation_mappings(mappings_payload)
    except FederationConfigError:
        return False
    if str(stored.get("manifest_hash", "") or "") != manifest_hash(manifest):
        return False
    if str(stored.get("mappings_hash", "") or "") != mappings_hash(mappings):
        return False
    return True


def federation_artifact_manifest_view(federation_dir: str) -> ArtifactManifest | None:
    """Build a migration-tier view of the federation composite artifact manifest."""
    if not federation_persist_quad_coherent(federation_dir):
        return None
    stored_raw = load_federation_artifact_manifest_dict(federation_artifact_paths(federation_dir)["artifact_manifest"])
    if stored_raw is None:
        return None
    try:
        ver = coerce_format_version(stored_raw.get("artifact_format_version", "0") or "0")
    except (TypeError, ValueError):
        ver = "0"
    return ArtifactManifest(
        artifact_format_version=ver,
        created_with_package_version=str(stored_raw.get("created_with_package_version", "") or ""),
        min_compatible_package_version=str(stored_raw.get("min_compatible_package_version", "") or ""),
        structural_hash=str(stored_raw.get("structural_hash", "") or ""),
        profiling_hash=str(stored_raw.get("profiling_hash", "") or ""),
        scope_hash=str(stored_raw.get("scope_hash", "") or ""),
        effective_structural_hash=str(stored_raw.get("effective_structural_hash", "") or ""),
        schema_graph_id=str(stored_raw.get("schema_graph_id", "") or ""),
        notes_hash=str(stored_raw.get("notes_hash", "") or ""),
        semantic_edges_hash=str(stored_raw.get("semantic_edges_hash", "") or ""),
    )
