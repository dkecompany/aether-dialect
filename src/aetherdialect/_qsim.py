"""QSim: skeleton cache, schema context, value sampling, and LLM-backed intent generation."""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, cast

from ._config import PolicyConfig, QSimConfig, SeedWarmupConfig
from ._constants import (
    AGG_PATTERN,
    HAVING_COUNT_VALUES,
    HAVING_MIN_MAX_VALUES,
    HAVING_SUM_AVG_VALUES,
    MASTER_AETHERSPACE_NAME,
    MASTER_AETHERSPACE_UID,
    TABLE_COL_PATTERN,
    VALID_HAVING_OPS,
    VALID_HAVING_VALUE_TYPES,
    VALID_WHERE_VALUE_TYPES,
)
from ._constants_runtime import (
    QSIM_FILL_SYSTEM,
    QSIM_PHASE_B,
    QSIM_PHASE_C,
    QSIM_PHASE_D,
    QSIM_PHASE_E,
    QSIM_PHASE_F,
    QSIM_PHASE_G,
    QSIM_PHASE_H,
    QSIM_PHASE_J,
)
from ._contracts_base import (
    QSIM_COMPLEXITY_TIER_SPECS,
    QSIM_SUPPORTED_ADVANCED_FEATURES,
    ComplexityTier,
    DatabaseFeatureCapability,
)
from ._contracts_core import LlmJsonExhausted, PipelineFeatureSpec
from ._contracts_schema import (
    ColumnRole,
    QSimHaving,
    QSimIntent,
    QSimSkeleton,
    QSimWhereParam,
    RetryFailureContext,
    SchemaGraph,
    SkeletonLimits,
    SkeletonPool,
    ValueDomain,
)
from ._llm_provider import LLMProvider
from ._main_spaces import MainSpaceOps
from ._utils import (
    debug,
    get_aggregatable_columns,
    get_groupable_columns,
    intent_id,
    qsim_skeletons_filename,
    simulation_artifact_partition_fp,
    stable_bucket,
)
from ._utils_artifacts import artifact_lock, read_gzip_json, write_gzip_json_atomic
from ._utils_intent import generate_question

_active_qsim_engine_owner: ContextVar[object | None] = ContextVar("aetherdialect_qsim_engine_owner", default=None)
_active_simulation_partition_fp: ContextVar[str] = ContextVar("aetherdialect_simulation_partition_fp", default="")
_skeleton_cache: dict[tuple[str, str, frozenset[str]], list[QSimSkeleton]] = {}
_engine_skeleton_caches: dict[int, dict[tuple[str, str, frozenset[str]], list[QSimSkeleton]]] = {0: _skeleton_cache}


def register_engine_skeleton_cache_owner(owner: object) -> None:
    """Attach an empty in-memory skeleton cache to *owner*."""
    _engine_skeleton_caches[id(owner)] = {}


def drop_engine_skeleton_cache_owner(owner: object) -> None:
    """Drop the in-memory skeleton cache for *owner*."""
    _engine_skeleton_caches.pop(id(owner), None)


def push_qsim_engine_owner(owner: object) -> Token[object | None]:
    """Bind *owner* as the active QSim cache scope for nested skeleton generation."""
    return _active_qsim_engine_owner.set(owner)


def pop_qsim_engine_owner(token: Token[object | None]) -> None:
    """Restore the prior QSim cache owner after :func:`push_qsim_engine_owner`."""
    _active_qsim_engine_owner.reset(token)


def push_simulation_artifact_partition(partition_fp: str) -> Token[str]:
    """Bind *partition_fp* for nested warmup/QSim artifact reads and writes."""
    return _active_simulation_partition_fp.set(str(partition_fp or ""))


def pop_simulation_artifact_partition(token: Token[str]) -> None:
    """Restore the prior simulation partition after :func:`push_simulation_artifact_partition`."""
    _active_simulation_partition_fp.reset(token)


def active_simulation_artifact_partition_fp() -> str:
    """Return the active warmup/QSim scope partition fingerprint (empty for owner default)."""
    return str(_active_simulation_partition_fp.get() or "")


def resolve_simulation_artifact_partition_from_owner(owner: object | None) -> str:
    """Derive the simulation artifact partition for an engine owner."""
    if owner is None:
        return ""

    space_uid = ""
    space_tables: set[str] | None = None
    context_name = str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
    norm_context = context_name.strip().lower()
    if norm_context not in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID.lower()):
        artifacts_dir = getattr(owner, "_artifacts_dir", None)
        if artifacts_dir is not None:
            try:
                resolved_uid = MainSpaceOps.resolve_aetherspace_identity(str(artifacts_dir), norm_context)
            except (AttributeError, TypeError, ValueError):
                resolved_uid = None
            if resolved_uid and resolved_uid not in (MASTER_AETHERSPACE_UID, MASTER_AETHERSPACE_NAME):
                space_uid = resolved_uid
                snap = MainSpaceOps.load_aetherspace_snapshot(str(artifacts_dir), resolved_uid)
                if isinstance(snap, dict):
                    raw_tables = snap.get("tables")
                    if isinstance(raw_tables, (list, tuple)):
                        space_tables = {str(t) for t in raw_tables}
    scope_ctx = MainSpaceOps.resolve_preview_scope_context(owner)
    visible = getattr(owner, "_consumer_visible_objects", None)
    if visible is not None and not isinstance(visible, frozenset):
        visible = frozenset(str(v) for v in visible)
    return simulation_artifact_partition_fp(
        space_uid=space_uid,
        scope_ctx=scope_ctx,
        visible_objects=visible,
        space_tables=space_tables,
    )


def push_simulation_artifact_scope_from_owner(owner: object | None) -> Token[str]:
    """Bind the simulation artifact partition derived from *owner*."""
    return push_simulation_artifact_partition(resolve_simulation_artifact_partition_from_owner(owner))


def engine_skeleton_cache() -> dict[tuple[str, str, frozenset[str]], list[QSimSkeleton]]:
    """Return the skeleton cache for the active engine owner (or process default)."""
    owner = _active_qsim_engine_owner.get()
    if owner is None:
        return _skeleton_cache
    return _engine_skeleton_caches.setdefault(id(owner), {})


def clear_engine_skeleton_cache(owner: object | None = None) -> None:
    """Clear skeleton cache entries for *owner*, the active owner, or the process default."""
    target = owner if owner is not None else _active_qsim_engine_owner.get()
    if target is None:
        _skeleton_cache.clear()
        return
    _engine_skeleton_caches[id(target)] = {}


def _tier_feasible_for_capability(tier_key: str, cap: DatabaseFeatureCapability) -> bool:
    """Return whether a complexity tier remains achievable on this database snapshot."""
    if cap.table_count <= 0:
        return False
    if tier_key == ComplexityTier.SIMPLE.value:
        return True
    if tier_key == ComplexityTier.MODERATE.value:
        return cap.table_count >= 1
    if tier_key == ComplexityTier.COMPLEX.value:
        return (
            cap.max_tables_on_any_join_path >= 3
            or (cap.table_count >= 2 and cap.fk_edge_count >= 1)
            or (cap.has_numeric_measures and cap.table_count >= 1)
        )
    if tier_key == ComplexityTier.HIGHLY_COMPLEX.value:
        return (
            cap.max_fk_chain_depth >= 2
            or cap.has_self_referential_fk
            or cap.max_tables_on_any_join_path >= 4
            or (cap.max_tables_on_any_join_path >= 3 and cap.has_window_capable_table_sets)
        )
    return False


def rebalance_complexity_target_proportions(
    proportions: Mapping[str, float], cap: DatabaseFeatureCapability
) -> dict[str, float]:
    """Zero unreachable tier mass and renormalize remaining targets for QSim and warmup budgets."""
    keys = [
        ComplexityTier.SIMPLE.value,
        ComplexityTier.MODERATE.value,
        ComplexityTier.COMPLEX.value,
        ComplexityTier.HIGHLY_COMPLEX.value,
    ]
    feas = {k: _tier_feasible_for_capability(k, cap) for k in keys}
    raw_mass = sum(max(0.0, float(proportions.get(k, 0.0))) for k in keys if feas[k])
    if raw_mass <= 0.0:
        active = [k for k in keys if feas[k]]
        if not active:
            return {k: 0.25 for k in keys}
        u = 1.0 / float(len(active))
        return {k: (u if k in active else 0.0) for k in keys}
    out: dict[str, float] = {}
    for k in keys:
        if feas[k]:
            out[k] = max(0.0, float(proportions.get(k, 0.0))) / raw_mass
        else:
            out[k] = 0.0
    s = sum(out.values())
    if s <= 0.0:
        active = [k for k in keys if feas[k]]
        u = 1.0 / float(len(active))
        return {k: (u if k in active else 0.0) for k in keys}
    return {k: (v / s) for k, v in out.items()}


def _skeleton_schema_key(schema: SchemaGraph) -> str:
    """Return the schema-graph identity used to partition skeleton cache entries."""
    graph_id = str(schema.schema_graph_id or "").strip()
    if graph_id:
        return graph_id
    structural = str(schema.structural_hash or "").strip()
    if structural:
        return structural
    return str(schema.effective_structural_hash or "")


def _skeleton_cache_key(schema: SchemaGraph, tables: list[str]) -> tuple[str, str, frozenset[str]]:
    """Build the in-memory skeleton cache key for a schema graph, scope partition, and table set."""
    return (
        _skeleton_schema_key(schema),
        active_simulation_artifact_partition_fp(),
        frozenset(tables),
    )


def _skeleton_cache_for_schema(schema: SchemaGraph) -> dict[frozenset[str], list[QSimSkeleton]]:
    """Return cached skeleton lists keyed by table set for one schema graph and active partition."""
    schema_key = _skeleton_schema_key(schema)
    partition_fp = active_simulation_artifact_partition_fp()
    return {
        table_key: skeletons
        for (cached_schema_key, cached_partition_fp, table_key), skeletons in engine_skeleton_cache().items()
        if cached_schema_key == schema_key and cached_partition_fp == partition_fp
    }


def _deserialize_skeleton_entries(skel_list: list[dict[str, Any]]) -> list[QSimSkeleton]:
    """Rebuild ``QSimSkeleton`` rows from on-disk cache payloads."""
    return [
        QSimSkeleton(
            tables=s["tables"],
            has_aggregation=s["has_aggregation"],
            num_where=s.get("num_where", 0),
            num_groupby=s["num_groupby"],
            has_orderby=s["has_orderby"],
            num_having=(int(s["num_having"]) if s.get("num_having") is not None else (1 if s.get("has_having") else 0)),
            has_distinct=s.get("has_distinct", False),
            has_expr_comparison=s.get("has_expr_comparison", s.get("has_column_comparison", False)),
            advanced_slot=s.get("advanced_slot"),
        )
        for s in skel_list
    ]


def _serialize_skeleton_cache(schema: SchemaGraph) -> dict[str, list[dict[str, Any]]]:
    """Serialize in-memory skeleton cache entries for one schema graph."""
    return {
        "|".join(sorted(table_key)): [asdict(s) for s in skeletons]
        for table_key, skeletons in _skeleton_cache_for_schema(schema).items()
    }


def _store_skeleton_cache_entries(schema: SchemaGraph, skeletons_data: dict[str, list[dict[str, Any]]]) -> None:
    """Load serialized skeleton rows into the in-memory cache for one schema graph."""
    for table_key_str, skel_list in skeletons_data.items():
        table_key = frozenset(table_key_str.split("|"))
        cache_key = _skeleton_cache_key(schema, list(table_key))
        engine_skeleton_cache()[cache_key] = _deserialize_skeleton_entries(skel_list)


def build_fk_adjacency(schema: SchemaGraph) -> dict[str, set[str]]:
    """Build an undirected FK adjacency map for tables in the schema."""
    adj: dict[str, set[str]] = {t: set() for t in schema.tables}

    for table in schema.tables.values():
        for fk in table.foreign_keys:
            adj[fk.src_table].add(fk.dst_table)
            adj[fk.dst_table].add(fk.src_table)

    return adj


def is_connected(tables: list[str], adj: dict[str, set[str]]) -> bool:
    """Return whether all given tables are mutually reachable via FK edges."""
    if len(tables) <= 1:
        return True

    table_set = set(tables)
    visited = set()
    queue = [tables[0]]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for neighbor in adj.get(current, set()):
            if neighbor in table_set and neighbor not in visited:
                queue.append(neighbor)

    return visited == table_set


def enumerate_table_sets(schema: SchemaGraph, max_tables: int | None = None) -> list[list[str]]:
    """Enumerate all valid FK-connected table combinations up to max_tables tables."""
    if max_tables is None:
        max_tables = QSimConfig.MAX_TABLES_PER_INTENT

    adj = build_fk_adjacency(schema)
    table_names = sorted(schema.tables.keys())
    valid_sets: list[list[str]] = []

    for t in table_names:
        valid_sets.append([t])

    for size in range(2, max_tables + 1):
        for combo in combinations(table_names, size):
            combo_list = list(combo)
            if is_connected(combo_list, adj):
                valid_sets.append(combo_list)

    debug(f"[{QSIM_PHASE_C}] enumerate_table_sets found {len(valid_sets)} valid table combinations")
    return valid_sets


def get_filterable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[tuple[str, str]]:
    """Return filterable columns for a table, excluding denied and sensitive columns."""
    result: list[tuple[str, str]] = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        if not col_meta.is_filterable or not col_meta.is_selectable:
            continue
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        result.append((col_key, role))

    return result


def get_comparable_column_pairs(
    table_set: list[str], schema: SchemaGraph, column_roles: dict[str, str]
) -> list[tuple[str, str, str, str, str]]:
    """Return cross-table column pairs that can be semantically compared."""
    comparable_pairs = []

    numeric_roles = {
        ColumnRole.NUMERIC_MEASURE.value,
        ColumnRole.NUMERIC_CATEGORICAL.value,
    }
    temporal_roles = {ColumnRole.TEMPORAL.value}

    all_numeric = []
    all_temporal = []

    for table in table_set:
        table_ir = schema.tables.get(table)
        if not table_ir:
            continue
        for col_name, col_meta in table_ir.columns.items():
            col_key = f"{table}.{col_name}"
            role = column_roles.get(col_key, col_meta.role or "unknown")
            if role in numeric_roles:
                all_numeric.append((table, col_name, role))
            elif role in temporal_roles:
                all_temporal.append((table, col_name, role))

    for i, (t1, c1, r1) in enumerate(all_numeric):
        for t2, c2, r2 in all_numeric[i + 1 :]:
            if t1 != t2 and r1 == r2:
                comparable_pairs.append((t1, c1, t2, c2, r1))

    for i, (t1, c1, r1) in enumerate(all_temporal):
        for t2, c2, _r2 in all_temporal[i + 1 :]:
            if t1 != t2:
                comparable_pairs.append((t1, c1, t2, c2, r1))

    return comparable_pairs


def _compute_skeleton_limits(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> SkeletonLimits:
    """Compute schema-derived limits for skeleton enumeration."""
    all_filterable = []
    all_groupable = []
    all_aggregatable = []
    for table in tables:
        all_filterable.extend(get_filterable_columns(table, schema, column_roles))
        all_groupable.extend(get_groupable_columns(table, schema, column_roles))
        all_aggregatable.extend(get_aggregatable_columns(table, schema, column_roles))

    num_whereable = len(set(col for col, _ in all_filterable))
    max_where_cols = min(QSimConfig.MAX_WHERE_COLUMNS, num_whereable)
    max_where_predicates = min(QSimConfig.MAX_WHERE_PREDICATES_PER_INTENT, max_where_cols * 2)
    max_groupby = min(len(all_groupable), QSimConfig.MAX_GROUP_BY_COLUMNS)
    max_having = min(SeedWarmupConfig.MAX_HAVING_CONDITIONS, 1 + len(all_aggregatable))

    return SkeletonLimits(max_where_predicates=max_where_predicates, max_groupby=max_groupby, max_having=max_having)


def compute_intent_id(intent_dict: dict[str, Any]) -> str:
    """Compute a hash-based intent ID from structural intent fields."""
    structural = {
        "tables": sorted(intent_dict.get("tables", [])),
        "grain": intent_dict.get("grain", "row_level"),
        "select_cols": sorted(intent_dict.get("select_cols", [])),
        "group_by_cols": sorted(intent_dict.get("group_by_cols", [])),
        "where": sorted(
            intent_dict.get("where", []),
            key=lambda x: str(x.get("column", "")) if isinstance(x, dict) else "",
        ),
        "having_param": sorted(
            intent_dict.get("having_param", []),
            key=lambda x: str(x.get("expression", "")) if isinstance(x, dict) else "",
        ),
    }
    return intent_id(structural)


def generate_all_skeletons(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> list[QSimSkeleton]:
    """Generate all valid structural `QSimSkeleton` instances for a table set."""
    cache_key = _skeleton_cache_key(schema, tables)
    if cache_key in engine_skeleton_cache():
        debug(f"[{QSIM_PHASE_B}]  cache_hit: {len(engine_skeleton_cache()[cache_key])} skeletons")
        return engine_skeleton_cache()[cache_key]

    limits = _compute_skeleton_limits(tables, schema, column_roles)
    max_where_predicates = limits.max_where_predicates
    max_groupby = limits.max_groupby
    max_having = limits.max_having

    is_single_table = len(tables) == 1
    has_comparable_pairs = len(get_comparable_column_pairs(tables, schema, column_roles)) > 0

    skeletons = []

    for has_agg in [True, False]:
        for num_where in range(0, max_where_predicates + 1):
            groupby_options = [0] + list(range(1, max_groupby + 1)) if has_agg else [0]
            for num_groupby in groupby_options:
                for has_orderby in [True, False]:
                    if has_agg and num_groupby == 0:
                        having_opts = [0]
                    elif has_agg and num_groupby > 0:
                        having_opts = list(range(0, max_having + 1))
                    else:
                        having_opts = [0]
                    for num_having in having_opts:
                        distinct_options = [True, False] if not has_agg and is_single_table else [False]
                        for has_distinct in distinct_options:
                            expr_cmp_options = [True, False] if has_comparable_pairs and num_where > 0 else [False]
                            for has_expr_cmp in expr_cmp_options:
                                skeletons.append(
                                    QSimSkeleton(
                                        tables=tables,
                                        has_aggregation=has_agg,
                                        num_where=num_where,
                                        num_groupby=num_groupby,
                                        has_orderby=has_orderby,
                                        num_having=num_having,
                                        has_distinct=has_distinct,
                                        has_expr_comparison=has_expr_cmp,
                                    )
                                )

    engine_skeleton_cache()[cache_key] = skeletons

    debug(
        f"[{QSIM_PHASE_B}]  created {len(skeletons)} skeletons for tables={tables}, max_where_predicates={max_where_predicates}, max_groupby={max_groupby}, max_having={max_having}"
    )
    return skeletons


def _skeleton_cache_payload(
    schema: SchemaGraph, schema_cache: dict[frozenset[str], list[QSimSkeleton]]
) -> dict[str, Any]:
    """Build the on-disk skeleton cache document for one schema graph and active partition."""
    return {
        "schema_graph_id": str(schema.schema_graph_id or ""),
        "structural_hash": schema.structural_hash,
        "partition_fp": active_simulation_artifact_partition_fp(),
        "num_table_sets": len(schema_cache),
        "skeletons": _serialize_skeleton_cache(schema),
    }


def resolve_qsim_skeletons_path(partition_fp: str | None = None) -> str:
    """Return the on-disk skeleton cache path for the active or explicit scope partition."""
    fp = active_simulation_artifact_partition_fp() if partition_fp is None else str(partition_fp or "")
    artifacts_dir = os.path.dirname(os.path.abspath(QSimConfig.SKELETONS_JSON_PATH))
    return os.path.join(artifacts_dir, qsim_skeletons_filename(fp))


def load_or_create_skeletons(
    schema: SchemaGraph, column_roles: dict[str, str]
) -> dict[frozenset[str], list[QSimSkeleton]]:
    """Load the skeleton cache from disk or generate and persist it."""
    skeleton_path = resolve_qsim_skeletons_path()
    adir = os.path.dirname(os.path.abspath(skeleton_path))
    with artifact_lock(adir):
        return _load_or_create_skeletons_locked(schema, column_roles, skeleton_path)


def _load_or_create_skeletons_locked(
    schema: SchemaGraph, column_roles: dict[str, str], skeleton_path: str
) -> dict[frozenset[str], list[QSimSkeleton]]:
    """Body of :func:`load_or_create_skeletons` executed under the artifacts-dir lock."""
    if not PolicyConfig.REGENERATE_SKELETON_CACHE and os.path.exists(skeleton_path):
        try:
            cache_data = read_gzip_json(skeleton_path)

            cached_hash = cache_data.get("structural_hash", "")
            if cached_hash != schema.structural_hash:
                debug(
                    f"[{QSIM_PHASE_B}]  structural_hash mismatch: {cached_hash} != {schema.structural_hash}, attempting surgical prune"
                )
                skeletons_data = cache_data.get("skeletons", {})
                live_tables = set(schema.tables.keys())
                pruned: dict[str, list[dict[str, Any]]] = {}
                for table_key_str, skel_list in skeletons_data.items():
                    table_key = frozenset(table_key_str.split("|"))
                    if table_key <= live_tables:
                        pruned[table_key_str] = skel_list
                if pruned:
                    _store_skeleton_cache_entries(schema, pruned)
                    schema_cache = _skeleton_cache_for_schema(schema)
                    cache_data = _skeleton_cache_payload(schema, schema_cache)
                    debug(f"[{QSIM_PHASE_B}]  surgical prune retained {len(schema_cache)} table sets; rewriting cache")
                    write_gzip_json_atomic(skeleton_path, cache_data, sort_keys=True)
                    return schema_cache
                debug(f"[{QSIM_PHASE_B}]  surgical prune empty; full regeneration")
            else:
                skeletons_data = cache_data.get("skeletons", {})
                _store_skeleton_cache_entries(schema, skeletons_data)
                schema_cache = _skeleton_cache_for_schema(schema)
                debug(f"[{QSIM_PHASE_B}]  loaded {len(schema_cache)} table sets from cache")
                return schema_cache
        except Exception as e:
            debug(f"[{QSIM_PHASE_B}]  cache_load_failed: {e}")

    debug(f"[{QSIM_PHASE_B}]  generating new skeletons")
    table_sets = enumerate_table_sets(schema, QSimConfig.MAX_TABLES_PER_INTENT)

    for table_set in table_sets:
        generate_all_skeletons(table_set, schema, column_roles)

    schema_cache = _skeleton_cache_for_schema(schema)
    cache_data = _skeleton_cache_payload(schema, schema_cache)

    debug(f"[{QSIM_PHASE_B}]  saving {len(schema_cache)} table sets to cache")
    write_gzip_json_atomic(skeleton_path, cache_data, sort_keys=True)

    return schema_cache


def decompose_between_filter(f: QSimWhereParam) -> list[QSimWhereParam]:
    """Decompose a `BETWEEN` `QSimWhereParam` into `>=` and `<=` filters."""
    if f.op != "between":
        return [f]
    return [
        replace(f, op=">="),
        replace(f, op="<="),
    ]


def build_schema_context(tables: list[str], schema: SchemaGraph) -> str:
    """Build a schema context string for LLM prompts using the Compose field set under master scope."""
    payload = json.loads(schema.schema_payload_compose(tables, owner_master_scope=True))
    context_parts: list[str] = []
    for table in tables:
        table_body = payload.get(table)
        if not isinstance(table_body, dict):
            continue
        col_map = table_body.get("columns") or {}
        col_descriptions: list[str] = []
        for col_name, col_obj in sorted(col_map.items()):
            if not isinstance(col_obj, dict):
                continue
            col_type = str(col_obj.get("type") or "unknown")
            col_desc = f"{col_name} ({col_type})"
            if col_obj.get("pk"):
                col_desc += " [PK]"
            if col_obj.get("fk"):
                col_desc += f" [FK -> {col_obj['fk']}]"
            role = col_obj.get("role")
            if role:
                col_desc += f" [{role}]"
            col_descriptions.append(col_desc)
        table_desc = table_body.get("description") or f"{table} table"
        context_parts.append(f"TABLE {table} ({table_desc}):\n  " + "\n  ".join(col_descriptions))
    return "\n\n".join(context_parts)


def _domain_prefers_integer_samples(domain: ValueDomain) -> bool:
    """Return whether numeric samples for *domain* should use integer- like literals."""
    vt = (domain.value_type or "").strip().lower()
    if vt == "integer":
        return True
    if vt in ("number", "numeric", "float", "double"):
        return False
    if domain.value_type:
        return False
    return _is_integer_type(domain.data_type)


def validate_column_exists(col_ref: str, tables: list[str], schema: SchemaGraph) -> bool:
    """Return whether a `table.column` reference is valid for the given tables."""
    if "." not in col_ref:
        return False
    table, col = col_ref.split(".", 1)
    if table not in tables:
        return False
    table_ir = schema.tables.get(table)
    if not table_ir:
        return False
    return col in table_ir.columns


def _is_integer_type(data_type: str | None) -> bool:
    """Return whether `data_type` is an integer-like column type."""
    if not data_type:
        return False
    dtype_lower = data_type.lower()
    if dtype_lower in ("integer", "int", "bigint", "smallint", "tinyint", "long", "short"):
        return True
    if "int" in dtype_lower or dtype_lower in ("long", "short"):
        if "interval" not in dtype_lower:
            return True
    return False


def _parse_date(val: str) -> datetime | None:
    """Parse a date substring from `val` into a datetime."""
    if "T" in val:
        val = val.split("T")[0]
    elif " " in val:
        val = val.split(" ")[0]

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def _format_date(dt: datetime) -> str:
    """Format `dt` as an ISO date string."""
    return dt.strftime("%Y-%m-%d")


def _extract_date_part(val: str) -> str:
    """Return the calendar-date portion of `val`."""
    if "T" in val:
        return val.split("T")[0]
    if " " in val:
        return val.split(" ")[0]
    return val


def _sample_categorical(domain: ValueDomain, variant_idx: int) -> str | None:
    """Pick a categorical value from `domain` by `variant_idx`."""
    values_list = domain.values
    if values_list:
        base = stable_bucket(repr(tuple(values_list)), len(values_list))
        idx = (base + variant_idx) % len(values_list)
        return values_list[idx]
    if domain.min_val is not None and domain.max_val is not None:
        try:
            min_v = int(float(domain.min_val))
            max_v = int(float(domain.max_val))
            range_size = max(1, max_v - min_v + 1)
            value = min_v + (variant_idx % range_size)
            return str(value)
        except (ValueError, TypeError):
            return str(domain.min_val)
    return None


def _sample_boolean(domain: ValueDomain, variant_idx: int) -> str | None:
    """Pick a boolean literal string from `domain` or defaults."""
    values_list = domain.values
    if values_list:
        normalized: list[str] = []
        for v in cast(list[Any], values_list):
            if isinstance(v, bool):
                normalized.append("true" if v else "false")
                continue
            s = str(v)
            low = s.lower()
            normalized.append(low if low in ("true", "false") else s)
        idx = variant_idx % len(normalized)
        return normalized[idx]
    default_bools = ["true", "false"]
    idx = variant_idx % len(default_bools)
    return default_bools[idx]


def _sample_numeric_categorical(domain: ValueDomain, variant_idx: int) -> str | None:
    """Pick a discrete numeric string from `domain`."""
    values_list = domain.values
    if values_list:
        idx = variant_idx % len(values_list)
        val = values_list[idx]
        return str(int(float(val)))
    if domain.min_val is not None and domain.max_val is not None:
        try:
            min_v = int(float(domain.min_val))
            max_v = int(float(domain.max_val))
            range_size = max(1, max_v - min_v + 1)
            value = min_v + (variant_idx % range_size)
            return str(value)
        except (ValueError, TypeError):
            return str(int(float(domain.min_val)))
    return None


def _sample_numeric(domain: ValueDomain, op: str, variant_idx: int) -> str | None:
    """Sample a numeric literal suited to comparison operator `op`."""
    if domain.min_val is not None and domain.max_val is not None:
        try:
            min_v = float(domain.min_val)
            max_v = float(domain.max_val)
            range_size = max_v - min_v
            is_integer = _domain_prefers_integer_samples(domain)
            value: int | float

            if op == "=":
                if is_integer:
                    int_range = max(1, int(range_size + 1))
                    value = int(min_v + (variant_idx % int_range))
                else:
                    segment = (variant_idx % 10) / 10.0
                    value = min_v + segment * range_size
                    value = round(value, 2) if abs(value) >= 1 else round(value, 4)
            elif op in (">", ">="):
                lower_bound = min_v + range_size * 0.2
                upper_bound = min_v + range_size * 0.5
                value = lower_bound + (variant_idx % 5) * (upper_bound - lower_bound) / 5
                value = int(round(value)) if is_integer else (round(value, 2) if abs(value) >= 1 else round(value, 4))
            elif op in ("<", "<="):
                lower_bound = min_v + range_size * 0.5
                upper_bound = min_v + range_size * 0.8
                value = lower_bound + (variant_idx % 5) * (upper_bound - lower_bound) / 5
                value = int(round(value)) if is_integer else (round(value, 2) if abs(value) >= 1 else round(value, 4))
            else:
                if range_size > 0:
                    segment = (variant_idx % 10) / 10.0
                    value = min_v + segment * range_size
                else:
                    value = min_v
                value = int(round(value)) if is_integer else (round(value, 2) if abs(value) >= 1 else round(value, 4))

            return str(value)
        except (ValueError, TypeError):
            pass

    values_list = domain.values
    if values_list:
        idx = variant_idx % len(values_list)
        return values_list[idx]

    return None


def _sample_temporal(domain: ValueDomain, op: str, variant_idx: int) -> str | None:
    """Sample a date string suited to comparison operator `op`."""
    if domain.min_val is not None and domain.max_val is not None:
        try:
            min_dt = _parse_date(str(domain.min_val))
            max_dt = _parse_date(str(domain.max_val))

            if min_dt is None or max_dt is None:
                return _extract_date_part(str(domain.min_val))

            total_days = (max_dt - min_dt).days
            if total_days <= 0:
                return _format_date(min_dt)

            if op in (">", ">="):
                segment = 0.2 + ((variant_idx % 5) / 5.0) * 0.15
            elif op in ("<", "<="):
                segment = 0.65 + ((variant_idx % 5) / 5.0) * 0.15
            else:
                segment = (variant_idx % 10) / 10.0

            offset_days = int(total_days * segment)
            result_dt = min_dt + timedelta(days=offset_days)
            return _format_date(result_dt)
        except (ValueError, TypeError):
            pass

    values_list = domain.values
    if values_list:
        idx = variant_idx % len(values_list)
        return _extract_date_part(values_list[idx])

    return None


def _sample_in_values(domain: ValueDomain, value_type: str, variant_idx: int) -> str | None:
    """Build a comma-separated literal list for `in` / `not in`."""
    if value_type == "categorical":
        values_list = domain.values
        if values_list:
            n_values = min(3 + (variant_idx % 3), len(values_list))
            start_idx = variant_idx % max(1, len(values_list) - n_values + 1)
            values = values_list[start_idx : start_idx + n_values]
            return "'" + "','".join(values) + "'"

    elif value_type == "numeric_categorical":
        values_list = domain.values
        if values_list:
            n_values = min(3 + (variant_idx % 3), len(values_list))
            start_idx = variant_idx % max(1, len(values_list) - n_values + 1)
            values = values_list[start_idx : start_idx + n_values]
            int_values = [str(int(float(v))) for v in values]
            return ",".join(int_values)
        if domain.min_val is not None and domain.max_val is not None:
            try:
                min_v = int(float(domain.min_val))
                max_v = int(float(domain.max_val))
                range_size = max(1, max_v - min_v + 1)
                n_values = min(3 + (variant_idx % 3), range_size)
                values = []
                for i in range(n_values):
                    value = min_v + ((variant_idx + i) % range_size)
                    values.append(str(value))
                return ",".join(values)
            except (ValueError, TypeError):
                pass

    elif value_type == "boolean":
        values_list = domain.values
        if values_list:
            normalized = [v.lower() if v.lower() in ("true", "false") else v for v in values_list]
            return ",".join(normalized)
        return "true,false"

    elif value_type in ("numeric", "temporal"):
        if domain.min_val is not None and domain.max_val is not None:
            try:
                min_bound = float(domain.min_val)
                max_bound = float(domain.max_val)
                float_range = max_bound - min_bound
                is_integer = _domain_prefers_integer_samples(domain)
                n_values = 2 + (variant_idx % 3)
                values = []
                for i in range(n_values):
                    segment = ((variant_idx + i) % 10) / 10.0
                    val: int | float = min_bound + segment * float_range
                    val = int(round(val)) if is_integer else (round(val, 2) if abs(val) >= 1 else round(val, 4))
                    values.append(str(val))
                return ",".join(values)
            except (ValueError, TypeError):
                pass

    return None


def sample_value_from_domain(domain: ValueDomain, value_type: str, op: str = "=", variant_idx: int = 0) -> str | None:
    """Sample one concrete filter value from `domain`."""
    if value_type == "null" or op in ("is null", "is not null"):
        return None

    if op in ("in", "not in"):
        return _sample_in_values(domain, value_type, variant_idx)

    if value_type == "categorical":
        return _sample_categorical(domain, variant_idx)

    if value_type == "numeric_categorical":
        return _sample_numeric_categorical(domain, variant_idx)

    if value_type == "numeric":
        return _sample_numeric(domain, op, variant_idx)

    if value_type == "temporal":
        return _sample_temporal(domain, op, variant_idx)

    if value_type == "boolean":
        return _sample_boolean(domain, variant_idx)

    return None


def _identify_range_pairs(filters: list[QSimWhereParam]) -> dict[str, dict[str, int]]:
    """Find columns with both lower and upper bound filters."""
    column_ops: dict[str, dict[str, int]] = {}
    for idx, f in enumerate(filters):
        if f.is_expr_comparison:
            continue
        if f.op in (">", ">="):
            column_ops.setdefault(f.column, {})["lower_idx"] = idx
        elif f.op in ("<", "<="):
            column_ops.setdefault(f.column, {})["upper_idx"] = idx
    return {col: ops for col, ops in column_ops.items() if "lower_idx" in ops and "upper_idx" in ops}


def _sample_numeric_range(domain: ValueDomain, variant_idx: int) -> tuple[str | None, str | None]:
    """Sample a consistent lower and upper numeric bound pair."""
    if domain.min_val is None or domain.max_val is None:
        return None, None

    try:
        min_v = float(domain.min_val)
        max_v = float(domain.max_val)
        range_size = max_v - min_v
        if range_size <= 0:
            return None, None

        is_integer = _domain_prefers_integer_samples(domain)

        lower_segment = 0.15 + ((variant_idx % 5) / 5.0) * 0.2
        upper_segment = 0.65 + ((variant_idx % 5) / 5.0) * 0.2

        lower_val = min_v + lower_segment * range_size
        upper_val = min_v + upper_segment * range_size

        if is_integer:
            lower_val = int(round(lower_val))
            upper_val = int(round(upper_val))
            if lower_val >= upper_val:
                upper_val = min(lower_val + 1, int(max_v))
        else:
            lower_val = round(lower_val, 2) if abs(lower_val) >= 1 else round(lower_val, 4)
            upper_val = round(upper_val, 2) if abs(upper_val) >= 1 else round(upper_val, 4)

        return str(lower_val), str(upper_val)
    except (ValueError, TypeError):
        return None, None


def _sample_temporal_range(domain: ValueDomain, variant_idx: int) -> tuple[str | None, str | None]:
    """Sample a consistent lower and upper date bound pair."""
    if domain.min_val is None or domain.max_val is None:
        return None, None

    try:
        min_dt = _parse_date(str(domain.min_val))
        max_dt = _parse_date(str(domain.max_val))

        if min_dt is None or max_dt is None:
            lower_val = _extract_date_part(str(domain.min_val))
            upper_val = _extract_date_part(str(domain.max_val))
            return lower_val, upper_val

        total_days = (max_dt - min_dt).days
        if total_days <= 0:
            return _format_date(min_dt), _format_date(max_dt)

        lower_segment = 0.15 + ((variant_idx % 5) / 5.0) * 0.2
        upper_segment = 0.65 + ((variant_idx % 5) / 5.0) * 0.2

        lower_days = int(total_days * lower_segment)
        upper_days = int(total_days * upper_segment)

        lower_dt = min_dt + timedelta(days=lower_days)
        upper_dt = min_dt + timedelta(days=upper_days)

        return _format_date(lower_dt), _format_date(upper_dt)
    except (ValueError, TypeError):
        return None, None


def sample_coordinated_range(domain: ValueDomain, value_type: str, variant_idx: int) -> tuple[str | None, str | None]:
    """Sample coordinated lower and upper values for range filters."""
    if value_type not in ("numeric", "temporal"):
        return None, None

    if value_type == "numeric":
        return _sample_numeric_range(domain, variant_idx)

    if value_type == "temporal":
        return _sample_temporal_range(domain, variant_idx)

    return None, None


def deterministic_having_value(agg_func: str, variant_idx: int, having_idx: int = 0) -> str:
    """Pick a HAVING threshold from built-in pools."""
    offset = variant_idx * 3 + having_idx
    value: int | float

    if agg_func == "count":
        value = HAVING_COUNT_VALUES[offset % len(HAVING_COUNT_VALUES)]
        return str(value)

    if agg_func in {"sum", "avg"}:
        value = HAVING_SUM_AVG_VALUES[offset % len(HAVING_SUM_AVG_VALUES)]
        return str(value)

    if agg_func in {"min", "max"}:
        value = HAVING_MIN_MAX_VALUES[offset % len(HAVING_MIN_MAX_VALUES)]
        return str(value)

    idx = offset % len(HAVING_COUNT_VALUES)
    return str(HAVING_COUNT_VALUES[idx])


def _compute_intent_variance(intent: QSimIntent, value_domains: dict[str, ValueDomain]) -> int:
    """Score how much sampling diversity an intent allows."""
    variance_score = 0

    for f in intent.where:
        if f.is_expr_comparison:
            continue
        col_key = f.column
        domain = value_domains.get(col_key)
        if domain:
            if domain.values:
                variance_score += len(domain.values)
            elif domain.min_val is not None and domain.max_val is not None:
                variance_score += 10

    if intent.where:
        variance_score += 10 * len(intent.having_param)
    else:
        variance_score += 5 * len(intent.having_param)

    return variance_score


def _instantiate_intent(
    intent: QSimIntent, value_domains: dict[str, ValueDomain], variant_idx: int = 0
) -> QSimIntent | None:
    """Fill `param_values` for filters and HAVING from domains."""
    decomposed_filters: list[QSimWhereParam] = []
    for f in intent.where:
        decomposed_filters.extend(decompose_between_filter(f))

    range_pairs = _identify_range_pairs(decomposed_filters)
    range_values: dict[str, tuple[str, str]] = {}

    for col_key, pair_indices in range_pairs.items():
        domain = value_domains.get(col_key)
        if domain is None:
            continue
        lower_idx = pair_indices["lower_idx"]
        value_type = decomposed_filters[lower_idx].value_type
        lower_val, upper_val = sample_coordinated_range(domain, value_type, variant_idx)
        if lower_val is not None and upper_val is not None:
            range_values[col_key] = (lower_val, upper_val)

    new_filters: list[QSimWhereParam] = []
    new_param_values: dict[str, Any] = {}

    for filter_idx, f in enumerate(decomposed_filters):
        param_key = f"f{filter_idx}"

        if f.is_expr_comparison:
            new_filters.append(f)
            debug(f"[{QSIM_PHASE_H}]  expr_comparison: {f.column} {f.op} {f.right_column}")
            continue

        col_key = f.column
        value_type = f.value_type
        op = f.op

        if value_type == "null" or op in ("is null", "is not null"):
            new_filters.append(replace(f, value_type="null"))
            debug(f"[{QSIM_PHASE_H}]  null_filter: {col_key} {op}")
            continue

        domain = value_domains.get(col_key)

        if domain is None:
            debug(f"[{QSIM_PHASE_H}]  no_domain_skip_variant: {col_key}")
            return None

        value: str | None
        if col_key in range_values:
            lower_val, upper_val = range_values[col_key]
            if f.op in (">", ">="):
                value = lower_val
            elif f.op in ("<", "<="):
                value = upper_val
            else:
                combined_idx = variant_idx * len(decomposed_filters) + filter_idx
                value = sample_value_from_domain(domain, value_type, f.op, combined_idx)
        else:
            combined_idx = variant_idx * len(decomposed_filters) + filter_idx
            value = sample_value_from_domain(domain, value_type, f.op, combined_idx)

        if value is not None:
            new_param_values[param_key] = value

        new_filters.append(f)

    new_having: list[QSimHaving] = []
    for having_idx, h in enumerate(intent.having_param):
        param_key = f"h{having_idx}"

        agg_match = AGG_PATTERN.match(h.expression)
        agg_func = agg_match.group(1).lower() if agg_match else "count"
        value = deterministic_having_value(agg_func, variant_idx, having_idx)
        new_param_values[param_key] = value

        new_having.append(h)

    return QSimIntent(
        intent_id=intent.intent_id,
        tables=intent.tables,
        grain=intent.grain,
        select_cols=intent.select_cols,
        group_by_cols=intent.group_by_cols,
        order_by_cols=intent.order_by_cols,
        where=new_filters,
        having_param=new_having,
        param_values=new_param_values,
        question="",
        variant_idx=variant_idx,
        limit=intent.limit,
        distinct=intent.distinct,
    )


def instantiate_all(
    intents: list[QSimIntent],
    schema: SchemaGraph,
    num_questions: int | None = None,
    *,
    rng_seed: int | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> list[QSimIntent]:
    """Instantiate intents with proportional variant counts."""
    if num_questions is None:
        num_questions = QSimConfig.QUESTIONS_COUNT

    random.seed(rng_seed if rng_seed is not None else QSimConfig.RANDOM_SEED)

    avg_variants = num_questions / len(intents) if intents else 0
    if avg_variants < QSimConfig.MIN_AVG_VARIANTS_PER_INTENT:
        debug(
            f"[{QSIM_PHASE_H}]  WARNING: avg_variants={avg_variants:.2f} below MIN={QSimConfig.MIN_AVG_VARIANTS_PER_INTENT}"
        )
    if avg_variants > QSimConfig.MAX_AVG_VARIANTS_PER_INTENT:
        raise ValueError(
            f"Intent/variant ratio unrealistic: {len(intents)} intents cannot generate {num_questions} diverse questions (avg={avg_variants:.1f} > max={QSimConfig.MAX_AVG_VARIANTS_PER_INTENT})"
        )

    value_domains: dict[str, ValueDomain] = {}
    for table_name, table_meta in schema.tables.items():
        for col_name, col_meta in table_meta.columns.items():
            col_key = f"{table_name}.{col_name}"
            if not getattr(col_meta, "is_visible", True):
                value_domains[col_key] = ValueDomain(
                    values=[],
                    min_val=None,
                    max_val=None,
                    data_type=col_meta.data_type,
                    value_type=col_meta.value_type or "",
                )
                continue
            value_domains[col_key] = ValueDomain(
                values=col_meta.frequent_values or [],
                min_val=col_meta.min_val,
                max_val=col_meta.max_val,
                data_type=col_meta.data_type,
                value_type=col_meta.value_type or "",
            )
    debug(f"[{QSIM_PHASE_H}]  value_domains: {len(value_domains)} columns")

    variances: dict[str, float] = {}
    for intent in intents:
        variances[intent.intent_id] = _compute_intent_variance(intent, value_domains)

    total_variance = sum(v for v in variances.values() if v > 0)

    allocations: dict[str, int] = {}
    if total_variance == 0:
        for intent in intents:
            allocations[intent.intent_id] = 1
    else:
        for intent in intents:
            v = variances[intent.intent_id]
            if v == 0:
                allocations[intent.intent_id] = 1
            else:
                share = v / total_variance
                allocations[intent.intent_id] = max(1, round(num_questions * share))

    debug(f"[{QSIM_PHASE_H}]  total_variance={total_variance:.2f}, allocations_sum={sum(allocations.values())}")
    if trace_summary is not None:
        trace_summary["avg_variants"] = avg_variants
        trace_summary["total_variance"] = total_variance
        trace_summary["allocations_sum"] = sum(allocations.values())

    instantiated: list[QSimIntent] = []

    for intent in intents:
        max_variants = allocations[intent.intent_id]
        if trace_rows is not None:
            trace_rows.append(
                {
                    "stage": "instantiation_plan",
                    "intent_id": intent.intent_id,
                    "tables": list(intent.tables),
                    "filters_count": len(intent.where),
                    "having_count": len(intent.having_param),
                    "allocated_variants": max_variants,
                    "variance_score": variances[intent.intent_id],
                }
            )
        for variant_idx in range(max_variants):
            result = _instantiate_intent(intent, value_domains, variant_idx)
            if result is not None:
                instantiated.append(result)
                if trace_rows is not None:
                    trace_rows.append(
                        {
                            "stage": "instantiation_variant",
                            "status": "accepted",
                            "intent_id": intent.intent_id,
                            "variant_idx": variant_idx,
                            "param_keys": sorted(result.param_values.keys()),
                        }
                    )
            elif trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "instantiation_variant",
                        "status": "failed",
                        "intent_id": intent.intent_id,
                        "variant_idx": variant_idx,
                    }
                )

    if len(instantiated) > num_questions:
        random.shuffle(instantiated)
        instantiated = instantiated[:num_questions]
        debug(f"[{QSIM_PHASE_H}]  truncated: {len(instantiated)}/{num_questions}")
        if trace_summary is not None:
            trace_summary["truncated_to_num_questions"] = True
    elif len(instantiated) < num_questions:
        debug(f"[{QSIM_PHASE_H}]  limit_reached: {len(instantiated)}/{num_questions}")
    else:
        debug(f"[{QSIM_PHASE_H}]  created: {len(instantiated)} intents")

    if trace_summary is not None:
        trace_summary["produced_variants"] = len(instantiated)
        trace_summary["requested_questions"] = num_questions

    return instantiated


def _allocate_tier_int_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    """Convert fractional tier weights into nonnegative integers summing to *total*."""
    keys = [
        ComplexityTier.SIMPLE.value,
        ComplexityTier.MODERATE.value,
        ComplexityTier.COMPLEX.value,
        ComplexityTier.HIGHLY_COMPLEX.value,
    ]
    raw = [max(0.0, float(weights.get(k, 0.0))) * float(total) for k in keys]
    floors = [int(math.floor(r + 1e-12)) for r in raw]
    rem = int(total) - sum(floors)
    order = sorted(range(len(keys)), key=lambda i: raw[i] - floors[i], reverse=True)
    for j in range(max(0, rem)):
        floors[order[j % len(order)]] += 1
    return dict(zip(keys, floors, strict=False))


def _skeleton_bucket_lookup_key(target: ComplexityTier) -> str:
    """Map a sampled target tier to skeleton buckets derived from. :func:`classify_qsim_skeleton_complexity`."""
    if target == ComplexityTier.HIGHLY_COMPLEX:
        return ComplexityTier.COMPLEX.value
    return str(target.value)


def _tier_spec_lines(target: ComplexityTier) -> tuple[str, str]:
    """Resolve summary and example sketch strings for a tier target."""
    for spec in QSIM_COMPLEXITY_TIER_SPECS:
        if spec.tier == target:
            return spec.summary, spec.example_sketch
    return "", ""


def _advanced_feature_allowed(feature_id: str, cap: DatabaseFeatureCapability) -> bool:
    """Return whether an advanced feature remains plausible on this schema snapshot."""
    return feature_id in PipelineFeatureSpec.feasible_features_for_capability(cap)


def _advanced_feature_prompt_block(cap: DatabaseFeatureCapability) -> str:
    """Format capability-filtered advanced feature bullets for the skeleton-fill prompt."""
    lines: list[str] = []
    for spec in QSIM_SUPPORTED_ADVANCED_FEATURES:
        if not _advanced_feature_allowed(spec.feature_id, cap):
            continue
        lines.append(f"- {spec.feature_id}: {spec.summary} Example: {spec.example_fragment}")
    return "\n".join(lines) if lines else "(none feasible on this schema)"


def _sample_select_col_count_geometric(p: float, rng: random.Random, cap_cols: int = 12) -> int:
    """Sample a SELECT-column count biased toward small integers."""
    n = 1
    while n < cap_cols and rng.random() > p:
        n += 1
    return n


def _skeleton_suitable_for_advanced(skeleton: QSimSkeleton, feature_id: str) -> bool:
    """Return whether a base skeleton can host the requested advanced feature slot."""
    if feature_id == "distinct_select":
        return not skeleton.has_aggregation and skeleton.num_groupby == 0
    if feature_id in ("window_partition_order", "case_when_select"):
        return skeleton.has_aggregation and skeleton.num_groupby > 0
    if feature_id in ("date_window_filter", "date_diff_shapes"):
        return skeleton.num_where > 0
    if feature_id in ("scalar_cte_bridge", "multi_cte_chain", "self_join_via_cte"):
        return skeleton.has_aggregation or skeleton.num_where > 0
    if feature_id == "unnest_array_column":
        return skeleton.num_where >= 0 and not skeleton.has_aggregation
    return skeleton.num_where > 0 or skeleton.has_aggregation


def append_advanced_skeleton_variants(
    skeletons: list[QSimSkeleton], cap: DatabaseFeatureCapability
) -> list[QSimSkeleton]:
    """Append skeleton clones tagged with schema-feasible advanced feature slots."""
    qsim_ids = {spec.feature_id for spec in QSIM_SUPPORTED_ADVANCED_FEATURES}
    feasible = PipelineFeatureSpec.feasible_features_for_capability(cap) & qsim_ids
    if not feasible:
        return skeletons
    out = list(skeletons)
    for feature_id in sorted(feasible):
        base = next((s for s in skeletons if _skeleton_suitable_for_advanced(s, feature_id)), None)
        if base is not None:
            out.append(replace(base, advanced_slot=feature_id))
    return out


def _advanced_slot_prompt_line(skeleton: QSimSkeleton) -> str:
    """Format a required advanced-feature instruction when the skeleton carries a slot."""
    if not skeleton.advanced_slot:
        return ""
    label = skeleton.advanced_slot.replace("_", " ")
    return (
        f"REQUIRED ADVANCED FEATURE ({skeleton.advanced_slot}): "
        f"The structured intent MUST implement {label}. "
        "Use only columns and filters compatible with this skeleton."
    )


def _qsim_advanced_slot_detected(intent: QSimIntent, feature_id: str) -> bool:
    """Heuristic compliance check for advanced slots on string-based QSim intents."""
    if feature_id == "distinct_select":
        return bool(intent.distinct)
    if feature_id == "date_window_filter":
        return any(
            (f.value_type or "").lower() in ("temporal", "date", "datetime") or "date" in (f.column or "").lower()
            for f in intent.where
        )
    if feature_id == "date_diff_shapes":
        return any("date" in (f.column or "").lower() and f.op in (">", "<", ">=", "<=") for f in intent.where)
    if feature_id in ("window_partition_order", "case_when_select"):
        return bool(intent.having_param) or any("CASE" in sc.upper() for sc in intent.select_cols)
    return bool(intent.where or intent.having_param or intent.distinct)


def _build_merged_tier_buckets(
    schema: SchemaGraph, column_roles: dict[str, str]
) -> dict[str, list[tuple[QSimSkeleton, list[str]]]]:
    """Flatten A/B/C skeleton tiers into complexity buckets for weighted sampling."""
    merged: dict[str, list[tuple[QSimSkeleton, list[str]]]] = {
        ComplexityTier.SIMPLE.value: [],
        ComplexityTier.MODERATE.value: [],
        ComplexityTier.COMPLEX.value: [],
        ComplexityTier.HIGHLY_COMPLEX.value: [],
    }
    for nt in (1, 2, 3):
        pool = _build_skeleton_pool(schema, column_roles, num_tables=nt)
        for tk in pool.table_set_keys:
            ts = tk.split("|")
            for tier_dict in (pool.tier_a_by_table_set, pool.tier_b_by_table_set, pool.tier_c_by_table_set):
                for skel in tier_dict[tk]:
                    ct = skel.complexity_tier()
                    merged[ct.value].append((skel, ts))
    for k in merged:
        random.shuffle(merged[k])
    return merged


def _pop_matching_skeleton(
    bucket: list[tuple[QSimSkeleton, list[str]]], need_where: bool, need_having: bool
) -> tuple[QSimSkeleton, list[str]] | None:
    """Pop the next skeleton from *bucket* honoring filter and HAVING coverage needs."""
    for i, (sk, ts) in enumerate(bucket):
        if need_where and sk.num_where == 0:
            continue
        if need_having and sk.num_having == 0:
            continue
        bucket.pop(i)
        return sk, ts
    return None


def _pick_weighted_tier(tier_remaining: dict[str, int], rng: random.Random) -> str | None:
    """Sample the next tier to fill using remaining quota counts as weights."""
    active = [(k, v) for k, v in tier_remaining.items() if v > 0]
    if not active:
        return None
    keys = [a[0] for a in active]
    weights = [a[1] for a in active]
    return str(rng.choices(keys, weights=weights, k=1)[0])


def _has_aggregation(select_cols: list[str]) -> bool:
    """Return True if any select column string matches an aggregation pattern."""
    return any(AGG_PATTERN.match(sc) for sc in select_cols)


def _extract_agg_info(expr: str) -> tuple[str, str] | None:
    """Extract aggregation function and inner column from a SQL aggregation expression."""
    m = AGG_PATTERN.match(expr.strip())
    if m:
        return (m.group(1).lower(), m.group(2).strip())
    return None


def _extract_tables_from_expr(expr: str) -> set[str]:
    """Extract table names from a SQL expression containing. `table.column` references."""
    return {m.group(1) for m in TABLE_COL_PATTERN.finditer(expr)}


def _validate_skeleton_constraints(response: dict[str, Any], skeleton: QSimSkeleton) -> tuple[bool, list[str]]:
    """Validate an LLM response dict against structural skeleton constraints."""
    violations = []
    select_cols_raw = response.get("select_cols", [])
    has_agg = any(AGG_PATTERN.match(sc) for sc in select_cols_raw if isinstance(sc, str))

    if skeleton.has_aggregation and not has_agg:
        violations.append("skeleton requires aggregation but no aggregated select_cols found")
    if not skeleton.has_aggregation and has_agg:
        violations.append("skeleton forbids aggregation but aggregated select_cols found")

    filters = response.get("filters", [])
    if skeleton.num_where > 0 and len(filters) == 0 and not skeleton.has_expr_comparison:
        violations.append(f"skeleton requires {skeleton.num_where} filters but got 0")

    groupby = response.get("groupby_cols", [])
    if skeleton.num_groupby > 0 and len(groupby) == 0:
        violations.append(f"skeleton requires {skeleton.num_groupby} groupby but got 0")
    if skeleton.num_groupby == 0 and len(groupby) > 0:
        violations.append(f"skeleton forbids groupby but got {len(groupby)}")

    having = response.get("having", [])
    if len(having) != skeleton.num_having:
        violations.append(f"skeleton requires {skeleton.num_having} having clause(s) but got {len(having)}")

    has_distinct = response.get("distinct", False)
    if skeleton.has_distinct and not has_distinct:
        violations.append(f"skeleton requires distinct but got distinct={has_distinct}")
    if not skeleton.has_distinct and has_distinct:
        violations.append(f"skeleton forbids distinct but got distinct={has_distinct}")

    expr_comparison = response.get("expr_comparison") or response.get("column_comparison")
    if skeleton.has_expr_comparison and not expr_comparison:
        violations.append("skeleton requires expr_comparison but got none")

    orderby_cols = response.get("orderby_cols", [])
    if skeleton.has_orderby and len(orderby_cols) == 0:
        violations.append("skeleton requires orderby but got none")
    if not skeleton.has_orderby and len(orderby_cols) > 0:
        violations.append("skeleton forbids orderby but got orderby_cols")

    return (len(violations) == 0, violations)


def _build_retry_guidance(failure_ctx: RetryFailureContext, schema: SchemaGraph, column_roles: dict[str, str]) -> str:
    """Build retry guidance text for the LLM from a previous failure context."""
    guidance_parts = []
    guidance_parts.append(f"\n\n    RETRY GUIDANCE (Attempt {failure_ctx.attempt_number + 2}):")
    guidance_parts.append(f"    Previous attempt failed: {failure_ctx.failure_type}")
    guidance_parts.append(f"    Required tables: {failure_ctx.required_tables}")
    guidance_parts.append(f"    Tables you used: {list(failure_ctx.used_tables)}")
    guidance_parts.append(f"    Tables you MUST include: {list(failure_ctx.missing_tables)}")

    for missing_table in failure_ctx.missing_tables:
        table_ir = schema.tables.get(missing_table)
        if table_ir:
            cols = list(table_ir.columns.keys())[:5]
            guidance_parts.append(f"    Available columns in {missing_table}: {cols}")

    guidance_parts.append(
        f"    FIX: Add filters, select_cols, groupby_cols, or aggregation from {list(failure_ctx.missing_tables)}"
    )

    return "\n".join(guidance_parts)


def _llm_fill_intent(
    skeleton: QSimSkeleton,
    schema: SchemaGraph,
    column_roles: dict[str, str],
    *,
    target_tier: ComplexityTier | None = None,
    select_col_target: int | None = None,
    advanced_feature_lines: str | None = None,
) -> QSimIntent | None:
    """Fill a structural skeleton via the LLM; validate and retry with guidance on failure."""
    context = build_schema_context(skeleton.tables, schema)

    all_filterable = []
    all_groupable = []
    all_aggregatable = []
    for table in skeleton.tables:
        all_filterable.extend(get_filterable_columns(table, schema, column_roles))
        all_groupable.extend(get_groupable_columns(table, schema, column_roles))
        all_aggregatable.extend(get_aggregatable_columns(table, schema, column_roles))

    filterable_list = [col_key for col_key, _ in all_filterable]

    effective_filters = skeleton.num_where
    if skeleton.has_expr_comparison:
        effective_filters = max(0, skeleton.num_where - 1)

    if skeleton.has_aggregation:
        agg_instruction = (
            "MUST include at least one aggregated select column (COUNT/SUM/AVG/MIN/MAX wrapping table.column)"
        )
    else:
        agg_instruction = "NO aggregation - all select_cols must be plain table.column references"

    filter_instruction = (
        f"MUST include {skeleton.num_where} filter conditions" if skeleton.num_where > 0 else "DO NOT include filters"
    )
    groupby_instruction = (
        f"MUST include {skeleton.num_groupby} GROUP BY columns"
        if skeleton.num_groupby > 0
        else "DO NOT include GROUP BY"
    )
    orderby_instruction = (
        "MUST include ORDER BY clause (non-empty orderby_cols)" if skeleton.has_orderby else "DO NOT include ORDER BY"
    )

    if skeleton.num_having > 0:
        having_instruction = (
            f"MUST include exactly {skeleton.num_having} HAVING condition(s) with aggregation. "
            "The having array length must match."
        )
    else:
        having_instruction = "DO NOT include HAVING (having must be an empty array)"

    distinct_instruction = "Use SELECT DISTINCT (no aggregations)" if skeleton.has_distinct else ""

    expr_comparison_instruction = ""
    comparable_pairs = []
    if skeleton.has_expr_comparison:
        comparable_pairs = get_comparable_column_pairs(skeleton.tables, schema, column_roles)
        if comparable_pairs:
            pairs_str = ", ".join([f"{t1}.{c1} vs {t2}.{c2}" for t1, c1, t2, c2, _ in comparable_pairs[:5]])
            expr_comparison_instruction = (
                f"MUST include an expr-vs-expr comparison (e.g., {pairs_str}). "
                "Choose columns and operator that make logical sense. "
                "DO NOT set expr_comparison to null."
            )

    filterable_constraint = (
        f"\n        FILTERABLE COLUMNS (MUST use ONLY these for filters): {filterable_list}"
        if effective_filters > 0
        else ""
    )
    aggregatable_constraint = (
        f"\n        AGGREGATABLE COLUMNS (use for SUM/AVG/MIN/MAX): {all_aggregatable}"
        if skeleton.has_aggregation and all_aggregatable
        else ""
    )
    groupable_constraint = (
        f"\n        GROUPABLE COLUMNS (MUST use for GROUP BY): {all_groupable}" if skeleton.num_groupby > 0 else ""
    )

    optional_instructions = []
    if distinct_instruction:
        optional_instructions.append(distinct_instruction)
    if expr_comparison_instruction:
        optional_instructions.append(expr_comparison_instruction)
    optional_str = "\n        - ".join([""] + optional_instructions) if optional_instructions else ""

    tier_extra = ""
    if target_tier is not None:
        tsumm, tex = _tier_spec_lines(target_tier)
        sel_hint = int(select_col_target) if select_col_target is not None else 1
        feat_blk = advanced_feature_lines or ""
        slot_line = _advanced_slot_prompt_line(skeleton)
        tier_extra = f"""
        TARGET COMPLEXITY BAND: {target_tier.value}
        BAND GUIDANCE: {tsumm}
        EXAMPLE SKETCH: {tex}
        AIM FOR APPROXIMATELY {sel_hint} DISTINCT SELECT LIST ENTRIES (respect aggregation rules above).
        DATABASE-SUPPORTED ADVANCED SHAPES (only where compatible with this skeleton):
        {feat_blk}
        """
        if slot_line:
            tier_extra += f"\n        {slot_line}\n        "

    user_prompt = f"""
        Schema:
        {context}
        {filterable_constraint}{aggregatable_constraint}{groupable_constraint}

        CRITICAL REQUIREMENTS (MUST follow exactly):
        - Tables: {skeleton.tables}
        - {agg_instruction}
        - {filter_instruction}
        - {groupby_instruction}
        - {orderby_instruction}
        - {having_instruction}{optional_str}
        {tier_extra}

        Return JSON:
        {{
        "select_cols": ["table.column" | "COUNT(table.column)" | "SUM(table.column)" | "AVG(table.column)" | "MIN(table.column)" | "MAX(table.column)", ...],
        "filters": [{{"column": "table.column", "op": "=" | ">" | "<" | ">=" | "<=" | "!=" | "like" | "between" | "in" | "not in" | "is null" | "is not null", "value_type": "categorical" | "numeric_categorical" | "numeric" | "temporal" | "boolean" | "null"}}],
        "groupby_cols": ["table.column", ...],
        "orderby_cols": ["table.column ASC" | "table.column DESC" | "COUNT(table.column) DESC", ...],
        "having": [{{"expression": "COUNT(table.column)" | "SUM(table.column)" | "AVG(table.column)" | "MIN(table.column)" | "MAX(table.column)", "op": "=" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "not in" | "between", "value_type": "number" | "integer"}}],
        "expr_comparison": {{"left_column": "table.column", "op": "=" | ">" | "<" | ">=" | "<=" | "!=", "right_column": "table.column"}} | null,
        "distinct": true | false
        }}
    """

    last_failure_reason = None
    failure_context = None
    for attempt in range(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1):
        prompt_with_context = user_prompt
        if failure_context and attempt > 0:
            retry_guidance = _build_retry_guidance(failure_context, schema, column_roles)
            prompt_with_context = f"{user_prompt}{retry_guidance}"
        elif last_failure_reason and attempt > 0:
            prompt_with_context = f"{user_prompt}\n\n    PREVIOUS ATTEMPT FAILED: {last_failure_reason}\n    Please fix this issue in your response."

        try:
            result = LLMProvider.json(QSIM_FILL_SYSTEM, prompt_with_context, task="synth")
        except LlmJsonExhausted as exc:
            last_failure_reason = f"LLM returned no parseable JSON ({exc})"
            failure_context = None
            debug(
                f"[{QSIM_PHASE_E}] attempt {attempt + 1}/{(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} exhausted: {last_failure_reason} for skeleton tables={skeleton.tables}"
            )
            continue

        debug(
            f"[{QSIM_PHASE_E}] attempt {attempt + 1} LLM returned: select_cols={len(result.get('select_cols', []))}, filters_count={len(result.get('filters', []))}, groupby_count={len(result.get('groupby_cols', []))}, having_count={len(result.get('having', []))}, expr_comparison={result.get('expr_comparison') or result.get('column_comparison')}, distinct={result.get('distinct')}"
        )

        is_valid, violations = _validate_skeleton_constraints(result, skeleton)
        if not is_valid:
            last_failure_reason = "; ".join(violations)
            failure_context = None
            debug(
                f"[{QSIM_PHASE_E}] attempt {attempt + 1}/{(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} SKELETON_CONSTRAINT_VIOLATION: {violations}"
            )
            continue

        parse_result = _parse_llm_response(result, skeleton, schema, column_roles)

        if isinstance(parse_result, tuple) and len(parse_result) == 3:
            failure_type, used_tables, missing_tables = parse_result
            failure_context = RetryFailureContext(
                failure_type=failure_type,
                required_tables=skeleton.tables,
                used_tables=used_tables,
                missing_tables=missing_tables,
                attempt_number=attempt,
            )
            last_failure_reason = None
            debug(
                f"[{QSIM_PHASE_E}] attempt {attempt + 1}/{(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} failed: {failure_type} for skeleton tables={skeleton.tables}, missing={missing_tables}"
            )
            continue

        if isinstance(parse_result, QSimIntent):
            if target_tier is not None:
                classified = parse_result.complexity_tier()
                if not QSimIntent.matches_target_tier(classified, target_tier):
                    last_failure_reason = f"tier_conformance: classified={classified.value} target={target_tier.value}"
                    failure_context = None
                    debug(
                        f"[{QSIM_PHASE_E}] attempt {attempt + 1}/{(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} "
                        f"TIER_MISMATCH classified={classified.value} target={target_tier.value}"
                    )
                    continue
            debug(
                f"[{QSIM_PHASE_E}] SUCCESS: intent_id={parse_result.intent_id}, grain={parse_result.grain}, filters={len(parse_result.where)}, groupby={len(parse_result.group_by_cols)}, distinct={parse_result.distinct}"
            )
            return parse_result

        last_failure_reason = "Response validation failed (filters/columns rejected)"
        failure_context = None
        debug(
            f"[{QSIM_PHASE_E}] attempt {attempt + 1}/{(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} failed: parse_llm_response returned None for skeleton tables={skeleton.tables}, LLM response keys={list(result.keys())}"
        )

    debug(
        f"[{QSIM_PHASE_E}] FINAL_FAILURE: exhausted {(PolicyConfig.MAX_ASK_COMPOSE_REPAIRS + 1)} attempts for skeleton tables={skeleton.tables}, has_agg={skeleton.has_aggregation}, num_where={skeleton.num_where}"
    )
    return None


def _parse_llm_response(
    response: dict[str, Any], skeleton: QSimSkeleton, schema: SchemaGraph, column_roles: dict[str, str]
) -> QSimIntent | tuple[str, set[str], set[str]] | None:
    """Parse and validate LLM JSON into a `QSimIntent` or retry context tuple."""
    select_cols_raw = response.get("select_cols", [])
    where_dicts = response.get("filters", [])
    groupby_cols = response.get("groupby_cols", [])
    orderby_cols_raw = response.get("orderby_cols", [])
    having_dicts = response.get("having", [])
    expr_comparison_dict = response.get("expr_comparison") or response.get("column_comparison")
    has_distinct = response.get("distinct", False)

    has_agg = _has_aggregation(select_cols_raw)

    if skeleton.has_aggregation and not has_agg:
        debug(f"[{QSIM_PHASE_F}] REJECTED: skeleton requires aggregation but none in select_cols")
        return None
    if skeleton.num_where > 0 and len(where_dicts) == 0 and not skeleton.has_expr_comparison:
        debug(f"[{QSIM_PHASE_F}] REJECTED: skeleton requires {skeleton.num_where} filters but none provided")
        return None
    if skeleton.num_groupby > 0 and len(groupby_cols) == 0:
        debug(f"[{QSIM_PHASE_F}] REJECTED: skeleton requires {skeleton.num_groupby} groupby cols but none provided")
        return None

    if skeleton.has_orderby and len(orderby_cols_raw) == 0:
        debug(f"[{QSIM_PHASE_F}] REJECTED: skeleton requires orderby but none provided")
        return None
    if not skeleton.has_orderby and len(orderby_cols_raw) > 0:
        debug(f"[{QSIM_PHASE_F}] REJECTED: skeleton forbids orderby but orderby_cols provided")
        return None

    if skeleton.has_distinct and skeleton.has_aggregation:
        debug(f"[{QSIM_PHASE_F}] REJECTED: DISTINCT not allowed with aggregation")
        return None

    select_cols: list[str] = []
    for sc in select_cols_raw:
        if not isinstance(sc, str) or not sc.strip():
            continue
        sc = sc.strip()
        agg_info = _extract_agg_info(sc)
        if agg_info:
            agg_func, agg_inner = agg_info
            if agg_inner != "*":
                if not validate_column_exists(agg_inner, skeleton.tables, schema):
                    debug(f"[{QSIM_PHASE_F}] REJECTED_SELECT: {sc}, reason=agg_column_not_found")
                    continue
            select_cols.append(f"{agg_func.upper()}({agg_inner})")
        else:
            if not validate_column_exists(sc, skeleton.tables, schema):
                debug(f"[{QSIM_PHASE_F}] REJECTED_SELECT: {sc}, reason=column_not_found")
                continue
            select_cols.append(sc)

    if not select_cols:
        debug(f"[{QSIM_PHASE_F}] REJECTED: no valid select_cols remaining")
        return None

    aggregated_tables: set[str] = set()
    if has_agg and groupby_cols:
        for gcol in groupby_cols:
            if "." in gcol:
                aggregated_tables.add(gcol.split(".")[0])

    filters: list[QSimWhereParam] = []
    where_columns_used: set[str] = set()
    for _where_idx, fd in enumerate(where_dicts):
        col = fd.get("column", "")
        if not validate_column_exists(col, skeleton.tables, schema):
            debug(f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=column_not_found")
            continue

        table, col_name = col.split(".", 1)
        col_meta = schema.tables[table].columns.get(col_name)
        if not col_meta or not col_meta.is_filterable or not col_meta.is_visible:
            debug(f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=not_filterable")
            continue

        if col not in where_columns_used and len(where_columns_used) >= QSimConfig.MAX_WHERE_COLUMNS + 1:
            debug(
                f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=max_where_columns_exceeded (>{QSimConfig.MAX_WHERE_COLUMNS + 1})"
            )
            continue

        if col not in where_columns_used and len(where_columns_used) >= QSimConfig.MAX_WHERE_COLUMNS:
            debug(
                f"[{QSIM_PHASE_F}] WARNING_WHERE: col={col}, using {len(where_columns_used) + 1} distinct columns (preferred max={QSimConfig.MAX_WHERE_COLUMNS})"
            )

        op = fd.get("op", "=")
        valid_ops = col_meta.get_valid_where_ops()

        if op not in valid_ops:
            debug(f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=invalid_operator_{op}_for_type")
            continue

        if has_agg and col_meta.is_foreign_key and op == "=":
            fk_target_table = col_meta.fk_target[0] if col_meta.fk_target else None

            if fk_target_table and fk_target_table in aggregated_tables:
                debug(f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=circular_fk_to_aggregated_table")
                continue

            if table in aggregated_tables:
                debug(f"[{QSIM_PHASE_F}] REJECTED_WHERE: col={col}, reason=fk_filter_on_aggregated_source_table")
                continue

        value_type = fd.get("value_type", "categorical")
        if op in ("is null", "is not null"):
            value_type = "null"
        elif value_type not in VALID_WHERE_VALUE_TYPES and value_type != "null":
            value_type = "categorical"

        where_columns_used.add(col)

        qf = QSimWhereParam(column=col, op=op, value_type=value_type)
        if op == "between":
            decomposed = decompose_between_filter(qf)
            filters.extend(decomposed)
            debug(f"[{QSIM_PHASE_F}] DECOMPOSED_BETWEEN: col={col} into >= and <=")
        else:
            filters.append(qf)
            debug(f"[{QSIM_PHASE_F}] ACCEPTED_FILTER: col={col}, op={op}, value_type={value_type}")

    if skeleton.has_expr_comparison and expr_comparison_dict:
        left_col_full = expr_comparison_dict.get("left_column", "")
        right_col_full = expr_comparison_dict.get("right_column", "")
        cmp_op = expr_comparison_dict.get("op", "=")

        if left_col_full and right_col_full and "." in left_col_full and "." in right_col_full:
            left_table, left_col_name = left_col_full.split(".", 1)
            right_table, right_col_name = right_col_full.split(".", 1)

            left_valid = validate_column_exists(left_col_full, skeleton.tables, schema)
            right_valid = validate_column_exists(right_col_full, skeleton.tables, schema)

            if left_valid and right_valid:
                left_meta = schema.tables[left_table].columns.get(left_col_name)
                right_meta = schema.tables[right_table].columns.get(right_col_name)

                if left_meta and right_meta:
                    left_is_numeric = left_meta.value_type in ("integer", "number")
                    right_is_numeric = right_meta.value_type in ("integer", "number")
                    left_is_temporal = left_meta.value_type == "date"
                    right_is_temporal = right_meta.value_type == "date"

                    left_role = column_roles.get(f"{left_table}.{left_col_name}", left_meta.role or "unknown")
                    right_role = column_roles.get(f"{right_table}.{right_col_name}", right_meta.role or "unknown")

                    semantic_compatible = False
                    rejection_reason = None

                    if left_role == right_role and left_role != "unknown":
                        semantic_compatible = True
                    elif left_is_temporal and right_is_temporal:
                        semantic_compatible = True
                    elif left_is_numeric and right_is_numeric and left_role == right_role:
                        semantic_compatible = True
                    else:
                        rejection_reason = f"role_mismatch: left={left_col_full}(role={left_role}) vs right={right_col_full}(role={right_role})"

                    if semantic_compatible:
                        value_type = "temporal" if left_is_temporal else "numeric"
                        filters.append(
                            QSimWhereParam(
                                column=left_col_full, op=cmp_op, value_type=value_type, right_column=right_col_full
                            )
                        )
                        debug(
                            f"[{QSIM_PHASE_F}] ACCEPTED_COLUMN_COMPARISON: {left_col_full} {cmp_op} {right_col_full}, roles={left_role}={right_role}"
                        )
                    else:
                        debug(
                            f"[{QSIM_PHASE_F}] DISCARDED_EXPR_COMPARISON: {left_col_full} {cmp_op} {right_col_full}, reason={rejection_reason}"
                        )
                else:
                    debug(f"[{QSIM_PHASE_F}] DISCARDED_EXPR_COMPARISON: column metadata not found")
            else:
                debug(
                    f"[{QSIM_PHASE_F}] DISCARDED_EXPR_COMPARISON: column validation failed left={left_valid} right={right_valid}"
                )
        else:
            debug(f"[{QSIM_PHASE_F}] DISCARDED_EXPR_COMPARISON: invalid column format")

    total_where_elements = len(filters)
    if skeleton.num_where > 0 and total_where_elements == 0:
        debug(
            f"[{QSIM_PHASE_F}] INSUFFICIENT_WHERE: requested={skeleton.num_where}, validated_filters={len(filters)}, rejecting_intent"
        )
        return None

    if has_agg and groupby_cols:
        for sc in select_cols:
            agg_info = _extract_agg_info(sc)
            if agg_info:
                _, agg_inner = agg_info
                agg_inner_base = agg_inner.split(".")[-1] if "." in agg_inner else agg_inner
                for gcol in groupby_cols:
                    gother_columnase = gcol.split(".")[-1] if "." in gcol else gcol
                    if agg_inner == gcol:
                        debug(
                            f"[{QSIM_PHASE_F}] REJECTED: agg_inner={agg_inner} matches groupby_col={gcol}, reason=exact_self_grouping"
                        )
                        return None
                    if agg_inner_base == gother_columnase:
                        debug(
                            f"[{QSIM_PHASE_F}] REJECTED: agg_inner={agg_inner} matches groupby_col={gcol}, reason=base_name_self_grouping"
                        )
                        return None

    having: list[QSimHaving] = []
    for hd in having_dicts:
        h_expression = hd.get("expression", "")
        h_op = hd.get("op", ">")
        if h_op not in VALID_HAVING_OPS:
            h_op = ">"
        h_value_type = hd.get("value_type", "number")
        if h_value_type not in VALID_HAVING_VALUE_TYPES:
            h_value_type = "number"
        right_expr = hd.get("right_expression", "")

        h_agg_info = _extract_agg_info(h_expression)
        if not h_agg_info:
            debug(f"[{QSIM_PHASE_F}] REJECTED_HAVING: expression={h_expression}, reason=no_aggregation_pattern")
            continue

        h_agg_func, h_agg_inner = h_agg_info
        if h_agg_inner != "*" and not validate_column_exists(h_agg_inner, skeleton.tables, schema):
            debug(f"[{QSIM_PHASE_F}] REJECTED_HAVING: expression={h_expression}, reason=column_not_found")
            continue

        if right_expr:
            right_agg_info = _extract_agg_info(right_expr)
            if not right_agg_info:
                debug(f"[{QSIM_PHASE_F}] REJECTED_HAVING: right_expression={right_expr}, reason=no_aggregation_pattern")
                continue
            right_agg_func, right_agg_inner = right_agg_info
            if right_agg_inner != "*" and not validate_column_exists(right_agg_inner, skeleton.tables, schema):
                debug(f"[{QSIM_PHASE_F}] REJECTED_HAVING: right_expression={right_expr}, reason=column_not_found")
                continue
            having.append(
                QSimHaving(
                    expression=f"{h_agg_func.upper()}({h_agg_inner})",
                    op=h_op,
                    value_type="expression",
                    right_expression=f"{right_agg_func.upper()}({right_agg_inner})",
                )
            )
        else:
            having.append(
                QSimHaving(expression=f"{h_agg_func.upper()}({h_agg_inner})", op=h_op, value_type=h_value_type)
            )

    validated_groupby: list[str] = []
    for gcol in groupby_cols:
        if validate_column_exists(gcol, skeleton.tables, schema):
            validated_groupby.append(gcol)
        else:
            debug(f"[{QSIM_PHASE_F}] REJECTED_GROUPBY: col={gcol}, reason=column_not_found")

    order_by_cols: list[str] = []
    for ob in orderby_cols_raw:
        ob_clean = ob.strip()
        direction = "ASC"
        if ob_clean.upper().endswith(" DESC"):
            direction = "DESC"
            ob_clean = ob_clean[:-5].strip()
        elif ob_clean.upper().endswith(" ASC"):
            ob_clean = ob_clean[:-4].strip()

        agg_info = _extract_agg_info(ob_clean)
        if agg_info:
            agg_func, agg_inner = agg_info
            if agg_inner != "*" and not validate_column_exists(agg_inner, skeleton.tables, schema):
                debug(f"[{QSIM_PHASE_F}] REJECTED_ORDERBY: {ob}, reason=column_not_found")
                continue
            order_by_cols.append(f"{agg_func.upper()}({agg_inner}) {direction}")
        else:
            if not validate_column_exists(ob_clean, skeleton.tables, schema):
                debug(f"[{QSIM_PHASE_F}] REJECTED_ORDERBY: {ob}, reason=column_not_found")
                continue
            order_by_cols.append(f"{ob_clean} {direction}")

    grain = "row_level"
    if has_agg:
        grain = "grouped" if validated_groupby else "scalar"

    use_distinct = skeleton.has_distinct and has_distinct and grain == "row_level"

    if skeleton.has_distinct and not use_distinct:
        if not has_distinct:
            debug("[{QSIM_PHASE_F}] DISTINCT_REJECTED: LLM returned distinct=false despite skeleton.has_distinct=True")
        elif grain != "row_level":
            debug(f"[{QSIM_PHASE_F}] DISTINCT_REJECTED: grain={grain} incompatible with DISTINCT (requires row_level)")

    if len(skeleton.tables) >= 3:
        tables_used: set[str] = set()
        for sc in select_cols:
            tables_used.update(_extract_tables_from_expr(sc))
        for col in validated_groupby:
            tables_used.update(_extract_tables_from_expr(col))
        for f in filters:
            tables_used.update(_extract_tables_from_expr(f.column))
            if f.right_column:
                tables_used.update(_extract_tables_from_expr(f.right_column))
        for ob in order_by_cols:
            tables_used.update(_extract_tables_from_expr(ob))

        missing_tables = set(skeleton.tables) - tables_used
        if missing_tables:
            debug(
                f"[{QSIM_PHASE_F}] REJECTED_THREE_TABLE: tables={skeleton.tables}, used={tables_used}, missing={missing_tables}"
            )
            return ("three_table_violation", tables_used, missing_tables)

    intent_id_val = compute_intent_id(
        {
            "tables": skeleton.tables,
            "grain": grain,
            "select_cols": select_cols,
            "group_by_cols": validated_groupby,
            "where": [f.to_dict() for f in filters],
            "having_param": [h.to_dict() for h in having],
            "distinct": use_distinct,
        }
    )

    return QSimIntent(
        intent_id=intent_id_val,
        tables=skeleton.tables,
        grain=grain,
        select_cols=select_cols,
        group_by_cols=validated_groupby,
        order_by_cols=order_by_cols,
        where=filters,
        having_param=having,
        param_values={},
        distinct=use_distinct,
    )


def _generate_question_from_intent(intent: QSimIntent, schema: SchemaGraph) -> str | None:
    """Produce natural language for an intent using `generate_question`."""
    filter_descriptions = []
    for idx, f in enumerate(intent.where):
        if f.is_expr_comparison:
            cond = f"{f.op} {f.right_column}"
        else:
            cond = f"{f.op} {intent.param_values.get(f'f{idx}', '?')}"
        filter_descriptions.append({"column": f.column, "condition": cond})

    having_descriptions = []
    for hidx, h in enumerate(intent.having_param):
        if h.is_expression_comparison:
            cond = f"{h.op} {h.right_expression}"
        else:
            cond = f"{h.op} {intent.param_values.get(f'h{hidx}', '?')}"
        having_descriptions.append({"expression": h.expression, "condition": cond})

    return generate_question(
        intent.tables,
        intent.select_cols,
        filter_descriptions,
        intent.group_by_cols,
        having_descriptions,
        schema,
    )


def generate_all_questions(
    intents: list[QSimIntent],
    schema: SchemaGraph,
    *,
    trace_rows: list[dict[str, Any]] | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> list[QSimIntent]:
    """Return intents with generated NL `question` fields where generation succeeds."""
    debug(f"[{QSIM_PHASE_J}] generating: {len(intents)} questions")

    results: list[QSimIntent] = []

    for i, intent in enumerate(intents):
        if i > 0 and i % 10 == 0:
            debug(f"[{QSIM_PHASE_J}] progress: {i}/{len(intents)}")

        question = _generate_question_from_intent(intent, schema)
        if question:
            intent_with_question = QSimIntent(
                intent_id=intent.intent_id,
                tables=intent.tables,
                grain=intent.grain,
                select_cols=intent.select_cols,
                group_by_cols=intent.group_by_cols,
                order_by_cols=intent.order_by_cols,
                where=intent.where,
                having_param=intent.having_param,
                param_values=intent.param_values,
                question=question,
                variant_idx=intent.variant_idx,
                limit=intent.limit,
                distinct=intent.distinct,
            )
            results.append(intent_with_question)
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "question_generation",
                        "status": "accepted",
                        "intent_id": intent.intent_id,
                        "variant_idx": intent.variant_idx,
                        "question": question,
                        "tables": list(intent.tables),
                        "grain": intent.grain,
                        "select_cols": list(intent.select_cols),
                        "group_by_cols": list(intent.group_by_cols),
                        "order_by_cols": list(intent.order_by_cols),
                        "where": [f.to_dict() for f in intent.where],
                        "having_param": [h.to_dict() for h in intent.having_param],
                        "param_values": dict(intent.param_values),
                        "distinct": intent.distinct,
                    }
                )
        else:
            debug(f"[{QSIM_PHASE_J}] failed: {intent.intent_id}")
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "question_generation",
                        "status": "failed",
                        "intent_id": intent.intent_id,
                        "variant_idx": intent.variant_idx,
                    }
                )

    debug(f"[{QSIM_PHASE_J}] complete: {len(results)} questions")
    if trace_summary is not None:
        trace_summary["accepted_questions"] = len(results)
        trace_summary["failed_questions"] = max(0, len(intents) - len(results))
    return results


def _is_no_variance_skeleton(skeleton: QSimSkeleton) -> bool:
    """Return True if the skeleton has no filters and no HAVING (no value-sampling variance)."""
    return skeleton.num_where == 0 and skeleton.num_having == 0


def _compute_skeleton_complexity_tier(skeleton: QSimSkeleton) -> str:
    """Assign complexity tier A, B, or C from skeleton features for stratified pools."""
    score = 0
    score += skeleton.num_where * 2
    score += 3 if skeleton.has_aggregation else 0
    score += skeleton.num_groupby * 2
    score += 3 if skeleton.num_having > 0 else 0
    score += 1 if skeleton.has_orderby else 0
    score += 2 if skeleton.has_distinct else 0
    score += 3 if skeleton.has_expr_comparison else 0

    if score >= 8:
        return "A"
    elif score >= 4:
        return "B"
    else:
        return "C"


def _compute_table_set_richness(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> int:
    """Score a table set from filterable, aggregatable, groupable, and comparable-column counts."""
    filterable_count = 0
    aggregatable_count = 0
    groupable_count = 0

    for table in tables:
        filterable_count += len(get_filterable_columns(table, schema, column_roles))
        aggregatable_count += len(get_aggregatable_columns(table, schema, column_roles))
        groupable_count += len(get_groupable_columns(table, schema, column_roles))

    comparable_pairs = len(get_comparable_column_pairs(tables, schema, column_roles))

    return filterable_count * 2 + aggregatable_count * 3 + groupable_count * 2 + comparable_pairs * 2


def _build_skeleton_pool(
    schema: SchemaGraph, column_roles: dict[str, str], num_tables: int | None = None
) -> SkeletonPool:
    """Build a tiered `SkeletonPool` from enumerated table sets and generated skeletons."""
    table_sets = enumerate_table_sets(schema)

    if num_tables is not None:
        table_sets = [ts for ts in table_sets if len(ts) == num_tables]

    scored_sets = [(ts, _compute_table_set_richness(ts, schema, column_roles)) for ts in table_sets]
    scored_sets.sort(key=lambda x: x[1], reverse=True)

    tier_a_by_table_set: dict[str, list[QSimSkeleton]] = {}
    tier_b_by_table_set: dict[str, list[QSimSkeleton]] = {}
    tier_c_by_table_set: dict[str, list[QSimSkeleton]] = {}

    for table_set, _ in scored_sets:
        table_key = "|".join(sorted(table_set))
        tier_a_by_table_set[table_key] = []
        tier_b_by_table_set[table_key] = []
        tier_c_by_table_set[table_key] = []

        skeletons = generate_all_skeletons(table_set, schema, column_roles)
        skeletons = append_advanced_skeleton_variants(skeletons, schema.database_feature_capability)
        for skel in skeletons:
            tier = _compute_skeleton_complexity_tier(skel)
            if tier == "A":
                tier_a_by_table_set[table_key].append(skel)
            elif tier == "B":
                tier_b_by_table_set[table_key].append(skel)
            else:
                tier_c_by_table_set[table_key].append(skel)

    for table_key in tier_a_by_table_set:
        random.shuffle(tier_a_by_table_set[table_key])
        random.shuffle(tier_b_by_table_set[table_key])
        random.shuffle(tier_c_by_table_set[table_key])

    table_set_keys = list(tier_a_by_table_set.keys())
    tier_a_indices = {k: 0 for k in table_set_keys}
    tier_b_indices = {k: 0 for k in table_set_keys}
    tier_c_indices = {k: 0 for k in table_set_keys}

    total_a = sum(len(v) for v in tier_a_by_table_set.values())
    total_b = sum(len(v) for v in tier_b_by_table_set.values())
    total_c = sum(len(v) for v in tier_c_by_table_set.values())

    debug(f"[{QSIM_PHASE_B}] built pool: tier_a={total_a}, tier_b={total_b}, tier_c={total_c}")
    return SkeletonPool(
        tier_a_by_table_set=tier_a_by_table_set,
        tier_b_by_table_set=tier_b_by_table_set,
        tier_c_by_table_set=tier_c_by_table_set,
        table_set_keys=table_set_keys,
        tier_a_indices=tier_a_indices,
        tier_b_indices=tier_b_indices,
        tier_c_indices=tier_c_indices,
    )


def _normalize_qsim_intent(intent: QSimIntent, schema: SchemaGraph) -> QSimIntent:
    """Return a canonical `QSimIntent`: grain, deduped columns, pruned tables, new `intent_id`."""
    grain = intent.grain
    has_agg = _has_aggregation(intent.select_cols)

    if grain == "grouped":
        if not intent.group_by_cols:
            grain = "row_level"
    else:
        if has_agg:
            grain = "grouped" if intent.group_by_cols else "scalar"

    normalized_select = sorted(set(intent.select_cols))
    normalized_orderby = sorted(intent.order_by_cols)

    tables_used: set[str] = set()
    for sc in normalized_select:
        tables_used.update(_extract_tables_from_expr(sc))
    for col in intent.group_by_cols:
        tables_used.update(_extract_tables_from_expr(col))
    for ob in normalized_orderby:
        tables_used.update(_extract_tables_from_expr(ob))
    for f in intent.where:
        tables_used.update(_extract_tables_from_expr(f.column))
        if f.right_column:
            tables_used.update(_extract_tables_from_expr(f.right_column))
    for h in intent.having_param:
        tables_used.update(_extract_tables_from_expr(h.expression))

    tables_used.discard("")

    normalized_tables = intent.tables
    if tables_used and len(tables_used) < len(intent.tables):
        adj = build_fk_adjacency(schema)
        if is_connected(list(tables_used), adj):
            normalized_tables = sorted(tables_used)
            debug(f"[{QSIM_PHASE_G}] removed unnecessary tables: {set(intent.tables) - tables_used}")

    table_prefixed_group_by = []
    for col in intent.group_by_cols:
        if "." not in col:
            if normalized_tables:
                col = f"{normalized_tables[0]}.{col}"
        table_prefixed_group_by.append(col)

    intent_id_val = compute_intent_id(
        {
            "tables": normalized_tables,
            "grain": grain,
            "select_cols": normalized_select,
            "group_by_cols": table_prefixed_group_by,
            "where": [f.to_dict() for f in intent.where],
            "having_param": [h.to_dict() for h in intent.having_param],
            "distinct": intent.distinct,
        }
    )

    return QSimIntent(
        intent_id=intent_id_val,
        tables=normalized_tables,
        grain=grain,
        select_cols=normalized_select,
        group_by_cols=table_prefixed_group_by,
        order_by_cols=normalized_orderby,
        where=intent.where,
        having_param=intent.having_param,
        param_values=intent.param_values,
        question=intent.question,
        variant_idx=intent.variant_idx,
        limit=intent.limit,
        distinct=intent.distinct,
    )


def generate_all_intents(
    schema: SchemaGraph,
    column_roles: dict[str, str],
    num_intents: int | None = None,
    *,
    rng_seed: int | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> list[QSimIntent]:
    """Generate diverse ``QSimIntent`` rows using tier-balanced skeleton sampling and coverage targets."""
    seed_val = rng_seed if rng_seed is not None else QSimConfig.RANDOM_SEED
    random.seed(seed_val)
    rng = random.Random(seed_val)
    load_or_create_skeletons(schema, column_roles)

    if num_intents is None:
        num_intents = QSimConfig.INTENT_TYPES

    cap = schema.database_feature_capability
    weights = rebalance_complexity_target_proportions(QSimConfig.COMPLEXITY_TARGET_PROPORTIONS, cap)
    quotas = _allocate_tier_int_quotas(weights, num_intents)
    tier_remaining: dict[str, int] = dict(quotas)
    adv_txt = _advanced_feature_prompt_block(cap)

    min_with_filters = int(num_intents * QSimConfig.MIN_FILTER_RATIO)
    min_with_having = int(num_intents * QSimConfig.MIN_HAVING_RATIO)
    min_three_table = int(num_intents * QSimConfig.MIN_THREE_TABLE_RATIO)
    max_no_variance = int(num_intents * QSimConfig.MAX_NO_VARIANCE_RATIO)

    debug(
        f"[{QSIM_PHASE_D}] targeting {num_intents} intents tier_quotas={tier_remaining} "
        f"rebalanced_weights={weights} schema_tables={cap.table_count}"
    )
    if trace_summary is not None:
        trace_summary["requested_intents"] = num_intents
        trace_summary["tier_quotas"] = dict(quotas)
        trace_summary["rebalanced_weights"] = dict(weights)
        trace_summary["min_with_filters"] = min_with_filters
        trace_summary["min_with_having"] = min_with_having
        trace_summary["min_three_table"] = min_three_table
        trace_summary["max_no_variance"] = max_no_variance

    merged_buckets = _build_merged_tier_buckets(schema, column_roles)

    intents: list[QSimIntent] = []
    seen_ids: set[str] = set()
    table_set_usage: dict[str, int] = {}
    no_variance_count = 0

    consecutive_duplicates = 0
    consecutive_failures = 0

    while len(intents) < num_intents:
        if consecutive_duplicates >= QSimConfig.MAX_CONSECUTIVE_DUPLICATES:
            debug(f"[{QSIM_PHASE_D}] EARLY_EXIT: consecutive duplicate cap")
            if trace_summary is not None:
                trace_summary["stop_reason"] = "consecutive_duplicate_cap"
            break
        if consecutive_failures >= QSimConfig.MAX_CONSECUTIVE_FAILURES:
            debug(f"[{QSIM_PHASE_D}] EARLY_EXIT: consecutive failure cap")
            if trace_summary is not None:
                trace_summary["stop_reason"] = "consecutive_failure_cap"
            break
        if sum(tier_remaining.values()) <= 0:
            debug(f"[{QSIM_PHASE_D}] STOP: tier quotas exhausted")
            if trace_summary is not None:
                trace_summary["stop_reason"] = "tier_quotas_exhausted"
            break

        tier_key = _pick_weighted_tier(tier_remaining, rng)
        if tier_key is None:
            break

        target_enum = ComplexityTier(str(tier_key))
        bucket_key = _skeleton_bucket_lookup_key(target_enum)
        bucket = merged_buckets.setdefault(bucket_key, [])

        current_with_filters = len([i for i in intents if i.where])
        current_with_having = len([i for i in intents if i.having_param])
        need_where = current_with_filters < min_with_filters
        need_having = current_with_having < min_with_having

        selection = _pop_matching_skeleton(bucket, need_where, need_having)
        if selection is None:
            tier_remaining[str(tier_key)] = 0
            debug(f"[{QSIM_PHASE_D}] exhausted skeleton bucket for tier={tier_key}")
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "intent_generation",
                        "status": "bucket_exhausted",
                        "target_tier": str(tier_key),
                        "need_where": need_where,
                        "need_having": need_having,
                    }
                )
            continue

        skeleton, table_set = selection

        if _is_no_variance_skeleton(skeleton) and no_variance_count >= max_no_variance:
            bucket.append((skeleton, table_set))
            debug(f"[{QSIM_PHASE_D}] SKIPPING: no-variance budget exceeded ({no_variance_count}/{max_no_variance})")
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "intent_generation",
                        "status": "skipped_no_variance_budget",
                        "target_tier": str(tier_key),
                        "tables": list(table_set),
                        "skeleton": asdict(skeleton),
                    }
                )
            continue

        sel_target = _sample_select_col_count_geometric(QSimConfig.SELECT_COL_GEOMETRIC_P, rng)

        intent = _llm_fill_intent(
            skeleton,
            schema,
            column_roles,
            target_tier=target_enum,
            select_col_target=sel_target,
            advanced_feature_lines=adv_txt,
        )

        if not intent:
            consecutive_failures += 1
            bucket.append((skeleton, table_set))
            debug(f"[{QSIM_PHASE_D}] LLM failed, consecutive_failures={consecutive_failures}")
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "intent_generation",
                        "status": "llm_fill_failed",
                        "target_tier": str(tier_key),
                        "tables": list(table_set),
                        "skeleton": asdict(skeleton),
                        "need_where": need_where,
                        "need_having": need_having,
                        "select_col_target": sel_target,
                    }
                )
            continue

        consecutive_failures = 0

        normalized = _normalize_qsim_intent(intent, schema)

        if skeleton.advanced_slot and not _qsim_advanced_slot_detected(normalized, skeleton.advanced_slot):
            consecutive_failures += 1
            bucket.append((skeleton, table_set))
            debug(f"[{QSIM_PHASE_D}] advanced slot {skeleton.advanced_slot} not detected after fill; retrying")
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "intent_generation",
                        "status": "advanced_slot_rejected",
                        "target_tier": str(tier_key),
                        "tables": list(table_set),
                        "advanced_slot_requested": skeleton.advanced_slot,
                        "skeleton": asdict(skeleton),
                    }
                )
            continue

        if normalized.intent_id in seen_ids:
            consecutive_duplicates += 1
            bucket.append((skeleton, table_set))
            debug(
                f"[{QSIM_PHASE_D}] DUPLICATE intent_id={normalized.intent_id}, "
                f"consecutive_duplicates={consecutive_duplicates}"
            )
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "stage": "intent_generation",
                        "status": "duplicate_intent",
                        "target_tier": str(tier_key),
                        "tables": list(table_set),
                        "intent_id": normalized.intent_id,
                        "skeleton": asdict(skeleton),
                    }
                )
            continue

        consecutive_duplicates = 0

        if _is_no_variance_skeleton(skeleton):
            no_variance_count += 1

        table_set_key = "|".join(sorted(table_set))
        table_set_usage[table_set_key] = table_set_usage.get(table_set_key, 0) + 1

        intents.append(normalized)
        seen_ids.add(normalized.intent_id)
        tier_remaining[str(tier_key)] -= 1
        debug(
            f"[{QSIM_PHASE_D}] ADDED intent_id={normalized.intent_id}, tier={tier_key}, "
            f"tables={normalized.tables}, filters={len(normalized.where)}, "
            f"having={len(normalized.having_param)}, total={len(intents)}/{num_intents}"
        )
        if trace_rows is not None:
            trace_rows.append(
                {
                    "stage": "intent_generation",
                    "status": "accepted",
                    "target_tier": str(tier_key),
                    "tables": list(table_set),
                    "intent_id": normalized.intent_id,
                    "grain": normalized.grain,
                    "filters_count": len(normalized.where),
                    "having_count": len(normalized.having_param),
                    "group_by_count": len(normalized.group_by_cols),
                    "distinct": normalized.distinct,
                    "skeleton": asdict(skeleton),
                    "select_col_target": sel_target,
                    "need_where": need_where,
                    "need_having": need_having,
                    "advanced_slot_requested": skeleton.advanced_slot,
                    "advanced_slot_detected": skeleton.advanced_slot
                    if skeleton.advanced_slot and _qsim_advanced_slot_detected(normalized, skeleton.advanced_slot)
                    else None,
                }
            )

    final_with_filters = len([i for i in intents if i.where])
    final_with_having = len([i for i in intents if i.having_param])
    single_count = len([i for i in intents if len(i.tables) == 1])
    two_count = len([i for i in intents if len(i.tables) == 2])
    three_count = len([i for i in intents if len(i.tables) >= 3])

    debug(
        f"[{QSIM_PHASE_D}] generated {len(intents)} intents: single={single_count}, two={two_count}, three={three_count}"
    )
    debug(
        f"[{QSIM_PHASE_D}] coverage: with_filters={final_with_filters}/{min_with_filters}, "
        f"with_having={final_with_having}/{min_with_having}, three_table={three_count}/{min_three_table}, "
        f"no_variance={no_variance_count}/{max_no_variance}"
    )
    debug(
        f"[{QSIM_PHASE_D}] table_set_usage: "
        f"{dict(sorted(table_set_usage.items(), key=lambda x: x[1], reverse=True)[:10])}"
    )
    if trace_summary is not None:
        trace_summary["generated_intents"] = len(intents)
        trace_summary["final_with_filters"] = final_with_filters
        trace_summary["final_with_having"] = final_with_having
        trace_summary["single_count"] = single_count
        trace_summary["two_count"] = two_count
        trace_summary["three_count"] = three_count
        trace_summary["no_variance_count"] = no_variance_count
        trace_summary["table_set_usage_top10"] = dict(
            sorted(table_set_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        trace_summary.setdefault(
            "stop_reason", "requested_count_reached" if len(intents) >= num_intents else "natural_stop"
        )

    return intents


def greedy_cover_indices_by_atoms(atoms_per_row: list[frozenset[str]], universe: frozenset[str]) -> list[int]:
    """Greedy set-cover ordering of row indices over a discrete atom universe."""
    uncovered = set(universe)
    picked: list[int] = []
    available = list(range(len(atoms_per_row)))
    while uncovered and available:
        best_i = -1
        best_gain = -1
        for idx in available:
            gain = len(atoms_per_row[idx] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = idx
        if best_i < 0 or best_gain <= 0:
            break
        picked.append(best_i)
        uncovered -= atoms_per_row[best_i]
        available.remove(best_i)
    return picked
