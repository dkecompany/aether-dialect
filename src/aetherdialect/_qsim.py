"""
QSim: skeleton cache, schema context, and value sampling for simulated intents.

When ``QSimSkeleton`` fields or enumeration rules change, invalidate the on-disk cache (``QSimConfig.SKELETONS_JSON_PATH``) or set ``PolicyConfig.REGENERATE_SKELETON_CACHE`` so loaded skeletons match the current code.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

from ._config import (
    AGG_PATTERN,
    HAVING_COUNT_VALUES,
    HAVING_MIN_MAX_VALUES,
    HAVING_SUM_AVG_VALUES,
    PolicyConfig,
    QSimConfig,
    SeedWarmupConfig,
)
from ._contracts_base import (
    ColumnRole,
    QSimSkeleton,
    SchemaGraph,
    SkeletonLimits,
    ValueDomain,
    data_type_to_value_type,
)
from ._contracts_core import QSimFilter, QSimHaving, QSimIntent
from ._core_utils import (
    artifact_lock,
    debug,
    intent_id,
    read_gzip_json,
    write_gzip_json_atomic,
)

_SKELETON_CACHE: dict[frozenset[str], list[QSimSkeleton]] = {}


def build_fk_adjacency(schema: SchemaGraph) -> dict[str, set[str]]:
    """
    Build an undirected FK adjacency map for tables in the schema.

    Args:

        schema: Schema graph whose foreign-key definitions are traversed.

    Returns:

        relationship with (bidirectional).
    """
    adj: dict[str, set[str]] = {t: set() for t in schema.tables}

    for table in schema.tables.values():
        for fk in table.foreign_keys:
            adj[fk.src_table].add(fk.dst_table)
            adj[fk.dst_table].add(fk.src_table)

    return adj


def is_connected(tables: list[str], adj: dict[str, set[str]]) -> bool:
    """
    Return whether all given tables are mutually reachable via FK edges.

    Args:

        tables: List of table names to test for connectivity.

        adj: Undirected FK adjacency map from `build_fk_adjacency`.

    Returns:

        has zero or one entry.
    """
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


def enumerate_table_sets(schema: SchemaGraph, max_tables: int = None) -> list[list[str]]:
    """
    Enumerate all valid FK-connected table combinations up to max_tables tables.

    Args:

        schema: Schema graph to derive tables and FK adjacency from.

        max_tables: Maximum tables per combination; defaults to `QSimConfig.MAX_TABLES_PER_INTENT`.

    Returns:

        List of table-name lists for each valid set.
    """
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

    debug(f"[qsim_struct.enumerate_table_sets] found {len(valid_sets)} valid table combinations")
    return valid_sets


def _is_excluded_filter_column(col_name: str) -> bool:
    """
    Return whether a column name matches any excluded filter pattern.

    Args:

        col_name: Column name string to test.

    Returns:

        case-insensitive substring of col_name.
    """
    for pattern in QSimConfig.EXCLUDED_FILTER_PATTERNS:
        if pattern in col_name.lower():
            return True
    return False


def get_filterable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[tuple[str, str]]:
    """
    Return filterable columns for a table, excluding audit and system columns.

    Args:

        table: Table name to inspect.

        schema: Schema graph containing column metadata.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        matching excluded patterns.
    """
    result = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        if not col_meta.is_filterable or _is_excluded_filter_column(col_name):
            continue
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        result.append((col_key, role))

    return result


def get_aggregatable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    """
    Return column keys that can be aggregated with SUM, AVG, MIN, or MAX.

    Args:

        table: Table name to inspect.

        schema: Schema graph containing column metadata.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        List of `table.column` keys with `NUMERIC_MEASURE` role.
    """
    result = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        if role == ColumnRole.NUMERIC_MEASURE.value:
            result.append(col_key)

    return result


def get_groupable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    """
    Return column keys usable in GROUP BY clauses.

    Args:

        table: Table name to inspect.

        schema: Schema graph containing column metadata.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        `NUMERIC_CATEGORICAL` role.
    """
    result = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        if role in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
        ):
            result.append(col_key)

    return result


def get_comparable_column_pairs(
    table_set: list[str], schema: SchemaGraph, column_roles: dict[str, str]
) -> list[tuple[str, str, str, str, str]]:
    """
    Return cross-table column pairs that can be semantically compared.

    Args:

        table_set: List of table names to consider.

        schema: Schema graph containing column metadata.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        comparable pair.
    """
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


def compute_skeleton_limits(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> SkeletonLimits:
    """
    Compute schema-derived limits for skeleton enumeration.

    Args:

        tables: List of table names in the intent.

        schema: Schema graph for column metadata.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        derived from column capabilities and config caps.
    """
    all_filterable = []
    all_groupable = []
    all_aggregatable = []
    for table in tables:
        all_filterable.extend(get_filterable_columns(table, schema, column_roles))
        all_groupable.extend(get_groupable_columns(table, schema, column_roles))
        all_aggregatable.extend(get_aggregatable_columns(table, schema, column_roles))

    num_filterable = len(set(col for col, _ in all_filterable))
    max_filter_cols = min(QSimConfig.MAX_FILTER_COLUMNS, num_filterable)
    max_filters = min(QSimConfig.MAX_FILTERS_PER_INTENT, max_filter_cols * 2)
    max_groupby = min(len(all_groupable), QSimConfig.MAX_GROUP_BY_COLUMNS)
    max_having = min(SeedWarmupConfig.MAX_HAVING_CONDITIONS, 1 + len(all_aggregatable))

    return SkeletonLimits(max_filters=max_filters, max_groupby=max_groupby, max_having=max_having)


def compute_intent_id(intent_dict: dict[str, Any]) -> str:
    """
    Compute a hash-based intent ID from structural intent fields.

    Args:

        intent_dict: Description.

    Returns:

        Short hash string suitable as a deduplicated intent identifier.
    """
    structural = {
        "tables": sorted(intent_dict.get("tables", [])),
        "grain": intent_dict.get("grain", "row_level"),
        "select_cols": sorted(intent_dict.get("select_cols", [])),
        "group_by_cols": sorted(intent_dict.get("group_by_cols", [])),
        "filters_param": sorted(
            intent_dict.get("filters_param", []),
            key=lambda x: str(x.get("column", "")) if isinstance(x, dict) else "",
        ),
        "having_param": sorted(
            intent_dict.get("having_param", []),
            key=lambda x: str(x.get("expression", "")) if isinstance(x, dict) else "",
        ),
    }
    return intent_id(structural)


def generate_all_skeletons(tables: list[str], schema: SchemaGraph, column_roles: dict[str, str]) -> list[QSimSkeleton]:
    """
    Generate all valid structural `QSimSkeleton` instances for a table set.

    Args:

        tables: Ordered list of table names defining the skeleton table set.

        schema: Schema graph for filterable, groupable, and aggregatable columns.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        results are cached in `_SKELETON_CACHE`.
    """
    global _SKELETON_CACHE

    table_key = frozenset(tables)
    if table_key in _SKELETON_CACHE:
        debug(f"[qsim_struct.generate_all_skeletons] cache_hit: {len(_SKELETON_CACHE[table_key])} skeletons")
        return _SKELETON_CACHE[table_key]

    limits = compute_skeleton_limits(tables, schema, column_roles)
    max_filters = limits.max_filters
    max_groupby = limits.max_groupby
    max_having = limits.max_having

    is_single_table = len(tables) == 1
    has_comparable_pairs = len(get_comparable_column_pairs(tables, schema, column_roles)) > 0

    skeletons = []

    for has_agg in [True, False]:
        for num_filters in range(0, max_filters + 1):
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
                            expr_cmp_options = [True, False] if has_comparable_pairs and num_filters > 0 else [False]
                            for has_expr_cmp in expr_cmp_options:
                                skeletons.append(
                                    QSimSkeleton(
                                        tables=tables,
                                        has_aggregation=has_agg,
                                        num_filters=num_filters,
                                        num_groupby=num_groupby,
                                        has_orderby=has_orderby,
                                        num_having=num_having,
                                        has_distinct=has_distinct,
                                        has_expr_comparison=has_expr_cmp,
                                    )
                                )

    _SKELETON_CACHE[table_key] = skeletons

    debug(
        f"[qsim_struct.generate_all_skeletons] created {len(skeletons)} skeletons for tables={tables}, max_filters={max_filters}, max_groupby={max_groupby}, max_having={max_having}"
    )
    return skeletons


def load_or_create_skeletons(
    schema: SchemaGraph, column_roles: dict[str, str]
) -> dict[frozenset[str], list[QSimSkeleton]]:
    """
    Load the skeleton cache from disk or generate and persist it.

    Args:

        schema: Description.

        column_roles: Map of `table.column` keys to role strings.

    Returns:

        stored in `_SKELETON_CACHE`).
    """
    global _SKELETON_CACHE

    skeleton_path = QSimConfig.SKELETONS_JSON_PATH
    adir = os.path.dirname(os.path.abspath(skeleton_path))
    with artifact_lock(adir):
        return _load_or_create_skeletons_locked(schema, column_roles, skeleton_path)


def _load_or_create_skeletons_locked(
    schema: SchemaGraph,
    column_roles: dict[str, str],
    skeleton_path: str,
) -> dict[frozenset[str], list[QSimSkeleton]]:
    """Body of :func:`load_or_create_skeletons` executed under the artifacts-dir lock."""

    global _SKELETON_CACHE

    if not PolicyConfig.REGENERATE_SKELETON_CACHE and os.path.exists(skeleton_path):
        try:
            cache_data = read_gzip_json(skeleton_path)

            cached_hash = cache_data.get("structural_hash", cache_data.get("schema_hash", ""))
            if cached_hash != schema.structural_hash:
                debug(
                    f"[qsim_struct.load_or_create_skeletons] structural_hash mismatch: {cached_hash} != {schema.structural_hash}, attempting surgical prune"
                )
                skeletons_data = cache_data.get("skeletons", {})
                live_tables = set(schema.tables.keys())
                for table_key_str, skel_list in skeletons_data.items():
                    table_key = frozenset(table_key_str.split("|"))
                    if not table_key <= live_tables:
                        continue
                    _SKELETON_CACHE[table_key] = [
                        QSimSkeleton(
                            tables=s["tables"],
                            has_aggregation=s["has_aggregation"],
                            num_filters=s["num_filters"],
                            num_groupby=s["num_groupby"],
                            has_orderby=s["has_orderby"],
                            num_having=(
                                int(s["num_having"])
                                if s.get("num_having") is not None
                                else (1 if s.get("has_having") else 0)
                            ),
                            has_distinct=s.get("has_distinct", False),
                            has_expr_comparison=s.get(
                                "has_expr_comparison",
                                s.get("has_column_comparison", False),
                            ),
                        )
                        for s in skel_list
                    ]
                if _SKELETON_CACHE:
                    cache_data = {
                        "structural_hash": schema.structural_hash,
                        "num_table_sets": len(_SKELETON_CACHE),
                        "skeletons": {"|".join(sorted(k)): [asdict(s) for s in v] for k, v in _SKELETON_CACHE.items()},
                    }
                    debug(
                        f"[qsim_struct.load_or_create_skeletons] surgical prune retained {len(_SKELETON_CACHE)} table sets; rewriting cache"
                    )
                    write_gzip_json_atomic(skeleton_path, cache_data, sort_keys=True)
                    return _SKELETON_CACHE
                debug("[qsim_struct.load_or_create_skeletons] surgical prune empty; full regeneration")
            else:
                skeletons_data = cache_data.get("skeletons", {})
                for table_key_str, skel_list in skeletons_data.items():
                    table_key = frozenset(table_key_str.split("|"))
                    _SKELETON_CACHE[table_key] = [
                        QSimSkeleton(
                            tables=s["tables"],
                            has_aggregation=s["has_aggregation"],
                            num_filters=s["num_filters"],
                            num_groupby=s["num_groupby"],
                            has_orderby=s["has_orderby"],
                            num_having=(
                                int(s["num_having"])
                                if s.get("num_having") is not None
                                else (1 if s.get("has_having") else 0)
                            ),
                            has_distinct=s.get("has_distinct", False),
                            has_expr_comparison=s.get(
                                "has_expr_comparison",
                                s.get("has_column_comparison", False),
                            ),
                        )
                        for s in skel_list
                    ]
                debug(f"[qsim_struct.load_or_create_skeletons] loaded {len(_SKELETON_CACHE)} table sets from cache")
                return _SKELETON_CACHE
        except Exception as e:
            debug(f"[qsim_struct.load_or_create_skeletons] cache_load_failed: {e}")

    debug("[qsim_struct.load_or_create_skeletons] generating new skeletons")
    table_sets = enumerate_table_sets(schema, QSimConfig.MAX_TABLES_PER_INTENT)

    for table_set in table_sets:
        generate_all_skeletons(table_set, schema, column_roles)

    cache_data = {
        "structural_hash": schema.structural_hash,
        "num_table_sets": len(_SKELETON_CACHE),
        "skeletons": {"|".join(sorted(k)): [asdict(s) for s in v] for k, v in _SKELETON_CACHE.items()},
    }

    debug(f"[qsim_struct.load_or_create_skeletons] saving {len(_SKELETON_CACHE)} table sets to cache")
    write_gzip_json_atomic(skeleton_path, cache_data, sort_keys=True)

    return _SKELETON_CACHE


def decompose_between_filter(f: QSimFilter) -> list[QSimFilter]:
    """
    Decompose a `BETWEEN` `QSimFilter` into `>=` and `<=` filters.

    Args:

        f: Filter to decompose.

    Returns:

        one-element list with `f` unchanged.
    """
    if f.op != "between":
        return [f]
    return [
        replace(f, op=">="),
        replace(f, op="<="),
    ]


def build_schema_context(tables: list[str], schema: SchemaGraph) -> str:
    """
    Build a schema context string for LLM prompts.

    Columns whose ``is_visible`` is false are omitted so the LLM never sees audit-only, near-empty, single-value, denied, or hidden-sensitivity columns; primary and foreign key columns remain visible only when not denied and not tagged as hidden sensitivity.

    Args:

        tables: Ordered list of table names to include.

        schema: Schema graph with table and column metadata.

    Returns:

        blank lines.
    """
    context_parts = []

    for table in tables:
        table_ir = schema.tables.get(table)
        if not table_ir:
            continue

        col_descriptions = []
        for col_name, col_meta in table_ir.columns.items():
            if not col_meta.is_visible:
                continue
            col_type = (col_meta.value_type or "").strip() or (
                data_type_to_value_type(col_meta.data_type) if col_meta.data_type else "unknown"
            )
            col_desc = f"{col_name} ({col_type})"
            if col_meta.is_primary_key:
                col_desc += " [PK]"
            if col_meta.is_foreign_key:
                fk_target = f"{col_meta.fk_target[0]}.{col_meta.fk_target[1]}" if col_meta.fk_target else "?"
                col_desc += f" [FK -> {fk_target}]"
            if col_meta.is_filterable:
                col_desc += " [filter]"
            col_descriptions.append(col_desc)

        table_desc = table_ir.description or f"{table} table"
        context_parts.append(f"TABLE {table} ({table_desc}):\n  " + "\n  ".join(col_descriptions))

    return "\n\n".join(context_parts)


def _domain_prefers_integer_samples(domain: ValueDomain) -> bool:
    """Return whether numeric samples for *domain* should use integer-like literals."""

    vt = (domain.value_type or "").strip().lower()
    if vt == "integer":
        return True
    if vt in ("number", "numeric", "float", "double"):
        return False
    if domain.value_type:
        return False
    return _is_integer_type(domain.data_type)


def validate_column_exists(col_ref: str, tables: list[str], schema: SchemaGraph) -> bool:
    """
    Return whether a `table.column` reference is valid for the given tables.

    Args:

        col_ref: Fully qualified reference in `table.column` form.

        tables: Allowed table names; the reference table must be in this list.

        schema: Schema graph for column existence.

    Returns:

        otherwise.
    """
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
    """
    Return whether `data_type` is an integer-like column type.

    Args:

        data_type: Declared SQL type string, or None.

    Returns:

        True if treated as integer; otherwise False.
    """
    if not data_type:
        return False
    dtype_lower = data_type.lower()
    if dtype_lower in (
        "integer",
        "int",
        "bigint",
        "smallint",
        "tinyint",
        "long",
        "short",
    ):
        return True
    if "int" in dtype_lower or dtype_lower in ("long", "short"):
        if "interval" not in dtype_lower:
            return True
    return False


def _parse_date(val: str) -> datetime | None:
    """
    Parse a date substring from `val` into a datetime.

    Args:

        val: Date or datetime string.

    Returns:

        Parsed date at midnight, or None if no format matches.
    """
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
    """
    Format `dt` as an ISO date string.

    Args:

        dt: Datetime to format.

    Returns:

        `YYYY-MM-DD` string.
    """
    return dt.strftime("%Y-%m-%d")


def _extract_date_part(val: str) -> str:
    """
    Return the calendar-date portion of `val`.

    Args:

        val: Datetime or date string.

    Returns:

        Date part before `T` or first space, or `val` unchanged.
    """
    if "T" in val:
        return val.split("T")[0]
    if " " in val:
        return val.split(" ")[0]
    return val


def _sample_categorical(domain: ValueDomain, variant_idx: int) -> str | None:
    """
    Pick a categorical value from `domain` by `variant_idx`.

    Args:

        domain: Column value domain.

        variant_idx: Deterministic variant index.

    Returns:

        Sampled string, or None.
    """
    values_list = domain.values
    if values_list:
        base = hash(tuple(values_list)) % len(values_list)
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
    """
    Pick a boolean literal string from `domain` or defaults.

    Args:

        domain: Column value domain.

        variant_idx: Deterministic variant index.

    Returns:

        `"true"` or `"false"`.
    """
    values_list = domain.values
    if values_list:
        normalized = []
        for v in values_list:
            if isinstance(v, bool):
                normalized.append("true" if v else "false")
            elif isinstance(v, str):
                normalized.append(v.lower() if v.lower() in ("true", "false") else v)
            else:
                normalized.append(str(v))
        idx = variant_idx % len(normalized)
        return normalized[idx]
    default_bools = ["true", "false"]
    idx = variant_idx % len(default_bools)
    return default_bools[idx]


def _sample_numeric_categorical(domain: ValueDomain, variant_idx: int) -> str | None:
    """
    Pick a discrete numeric string from `domain`.

    Args:

        domain: Column value domain.

        variant_idx: Deterministic variant index.

    Returns:

        Integer-like string, or None.
    """
    values_list = domain.values
    if values_list:
        idx = variant_idx % len(values_list)
        val = values_list[idx]
        return str(int(float(val))) if isinstance(val, int | float) else str(val)
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
    """
    Sample a numeric literal suited to comparison operator `op`.

    Args:

        domain: Column value domain.

        op: Filter operator (`=`, `>`, etc.).

        variant_idx: Deterministic variant index.

    Returns:

        Numeric string, list element, or None.
    """
    if domain.min_val is not None and domain.max_val is not None:
        try:
            min_v = float(domain.min_val)
            max_v = float(domain.max_val)
            range_size = max_v - min_v
            is_integer = _domain_prefers_integer_samples(domain)

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
    """
    Sample a date string suited to comparison operator `op`.

    Args:

        domain: Column value domain.

        op: Filter operator.

        variant_idx: Deterministic variant index.

    Returns:

        Date string, or None.
    """
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
    """
    Build a comma-separated literal list for `in` / `not in`.

    Args:

        domain: Column value domain.

        value_type: Semantic type (`categorical`, `numeric`, etc.).

        variant_idx: Deterministic variant index.

    Returns:

        SQL fragment of values, or None.
    """
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
            int_values = [str(int(float(v))) if isinstance(v, int | float) else str(v) for v in values]
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
            normalized = []
            for v in values_list:
                if isinstance(v, bool):
                    normalized.append("true" if v else "false")
                elif isinstance(v, str):
                    normalized.append(v.lower() if v.lower() in ("true", "false") else v)
                else:
                    normalized.append(str(v))
            return ",".join(normalized)
        return "true,false"

    elif value_type in ("numeric", "temporal"):
        if domain.min_val is not None and domain.max_val is not None:
            try:
                min_v = float(domain.min_val)
                max_v = float(domain.max_val)
                range_size = max_v - min_v
                is_integer = _domain_prefers_integer_samples(domain)
                n_values = 2 + (variant_idx % 3)
                values = []
                for i in range(n_values):
                    segment = ((variant_idx + i) % 10) / 10.0
                    val = min_v + segment * range_size
                    val = int(round(val)) if is_integer else (round(val, 2) if abs(val) >= 1 else round(val, 4))
                    values.append(str(val))
                return ",".join(values)
            except (ValueError, TypeError):
                pass

    return None


def sample_value_from_domain(domain: ValueDomain, value_type: str, op: str = "=", variant_idx: int = 0) -> str | None:
    """
    Sample one concrete filter value from `domain`.

    Args:

        domain: Column value domain.

        value_type: Semantic type (`categorical`, `numeric`, etc.).

        op: Filter operator.

        variant_idx: Deterministic variant index.

    Returns:

        Literal string, or None for null predicates.
    """
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


def _identify_range_pairs(filters: list[QSimFilter]) -> dict[str, dict[str, int]]:
    """
    Find columns with both lower and upper bound filters.

    Args:

        filters: Decomposed filter list.

    Returns:

        Map column key to indices `lower_idx` and `upper_idx`.
    """
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
    """
    Sample a consistent lower and upper numeric bound pair.

    Args:

        domain: Column value domain.

        variant_idx: Deterministic variant index.

    Returns:

        `(lower, upper)` strings, or `(None, None)` if unavailable.
    """
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
    """
    Sample a consistent lower and upper date bound pair.

    Args:

        domain: Column value domain.

        variant_idx: Deterministic variant index.

    Returns:

        `(lower, upper)` date strings, or `(None, None)` on failure.
    """
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
    """
    Sample coordinated lower and upper values for range filters.

    Args:

        domain: Column value domain.

        value_type: `numeric` or `temporal`.

        variant_idx: Deterministic variant index.

    Returns:

        Bound pair, or `(None, None)` if type unsupported.
    """
    if value_type not in ("numeric", "temporal"):
        return None, None

    if value_type == "numeric":
        return _sample_numeric_range(domain, variant_idx)

    if value_type == "temporal":
        return _sample_temporal_range(domain, variant_idx)

    return None, None


def deterministic_having_value(agg_func: str, variant_idx: int, having_idx: int = 0) -> str:
    """
    Pick a HAVING threshold from built-in pools.

    Args:

        agg_func: Aggregate name (`count`, `sum`, etc.).

        variant_idx: Variant offset.

        having_idx: Index within HAVING clauses.

    Returns:

        Threshold string.
    """
    offset = variant_idx * 3 + having_idx

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
    """
    Score how much sampling diversity an intent allows.

    Args:

        intent: Intent skeleton.

        value_domains: Column key to domain map.

    Returns:

        Non-negative variance score.
    """
    variance_score = 0

    for f in intent.filters_param:
        if f.is_expr_comparison:
            continue
        col_key = f.column
        domain = value_domains.get(col_key)
        if domain:
            if domain.values:
                variance_score += len(domain.values)
            elif domain.min_val is not None and domain.max_val is not None:
                variance_score += 10

    if intent.filters_param:
        variance_score += 10 * len(intent.having_param)
    else:
        variance_score += 5 * len(intent.having_param)

    return variance_score


def _instantiate_intent(
    intent: QSimIntent, value_domains: dict[str, ValueDomain], variant_idx: int = 0
) -> QSimIntent | None:
    """
    Fill `param_values` for filters and HAVING from domains.

    Args:

        intent: Intent skeleton.

        value_domains: Column key to domain map.

        variant_idx: Variant index for sampling.

    Returns:

        New `QSimIntent` with values and empty `question`.
    """
    decomposed_filters: list[QSimFilter] = []
    for f in intent.filters_param:
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

    new_filters: list[QSimFilter] = []
    new_param_values: dict[str, any] = {}

    for filter_idx, f in enumerate(decomposed_filters):
        param_key = f"f{filter_idx}"

        if f.is_expr_comparison:
            new_filters.append(f)
            debug(f"[qsim_sample.instantiate_intent] expr_comparison: {f.column} {f.op} {f.right_column}")
            continue

        col_key = f.column
        value_type = f.value_type
        op = f.op

        if value_type == "null" or op in ("is null", "is not null"):
            new_filters.append(replace(f, value_type="null"))
            debug(f"[qsim_sample.instantiate_intent] null_filter: {col_key} {op}")
            continue

        domain = value_domains.get(col_key)

        if domain is None:
            debug(f"[qsim_sample.instantiate_intent] no_domain_skip_variant: {col_key}")
            return None

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
        filters_param=new_filters,
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
) -> list[QSimIntent]:
    """
    Instantiate intents with proportional variant counts.

    Args:

        intents: Intent skeletons.

        schema: Schema graph for column domains.

        num_questions: Target count; defaults to `QSimConfig.QUESTIONS_COUNT`.

    Returns:

        List of instantiated intents, possibly truncated or shuffled.
    """
    if num_questions is None:
        num_questions = QSimConfig.QUESTIONS_COUNT

    random.seed(rng_seed if rng_seed is not None else QSimConfig.RANDOM_SEED)

    avg_variants = num_questions / len(intents) if intents else 0
    if avg_variants < QSimConfig.MIN_AVG_VARIANTS_PER_INTENT:
        debug(
            f"[qsim_sample.instantiate_all] WARNING: avg_variants={avg_variants:.2f} below MIN={QSimConfig.MIN_AVG_VARIANTS_PER_INTENT}"
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
                values=col_meta.top_k_values or [],
                min_val=col_meta.min_val,
                max_val=col_meta.max_val,
                data_type=col_meta.data_type,
                value_type=col_meta.value_type or "",
            )
    debug(f"[qsim_sample.instantiate_all] value_domains: {len(value_domains)} columns")

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

    debug(
        f"[qsim_sample.instantiate_all] total_variance={total_variance:.2f}, allocations_sum={sum(allocations.values())}"
    )

    instantiated: list[QSimIntent] = []

    for intent in intents:
        max_variants = allocations[intent.intent_id]
        for variant_idx in range(max_variants):
            result = _instantiate_intent(intent, value_domains, variant_idx)
            if result is not None:
                instantiated.append(result)

    if len(instantiated) > num_questions:
        random.shuffle(instantiated)
        instantiated = instantiated[:num_questions]
        debug(f"[qsim_sample.instantiate_all] truncated: {len(instantiated)}/{num_questions}")
    elif len(instantiated) < num_questions:
        debug(f"[qsim_sample.instantiate_all] limit_reached: {len(instantiated)}/{num_questions}")
    else:
        debug(f"[qsim_sample.instantiate_all] created: {len(instantiated)} intents")

    return instantiated
