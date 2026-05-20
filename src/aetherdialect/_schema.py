"""
Reflect schema, infer FKs, build join graph, profile columns, cache by hash, and attach adaptive limits.

``pyspark.sql.SparkSession`` is imported at module load when available so Databricks catalog paths do not use deferred imports elsewhere in this file.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import UniqueConstraint

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

from . import _core_utils
from ._config import (
    COMPATIBLE_TYPE_PAIRS,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP,
    FK_INFERENCE_SUFFIX_STEMS,
    INFERRED_PK_VALUE_TYPES,
    INTEGER_VALUE_TYPES,
    JSON_COMPACT_SEPARATORS,
    MIGRATION_DATA_OVERLAP_MIN,
    MIGRATION_TABLE_RENAME_COLUMN_FRACTION,
    OVERRIDES_EDITABLE_ENUMS,
    PK_STYLE_FK_STEMS,
    ROLE_VALUE_TYPE_COMPAT,
    SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS,
    SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER,
    SCHEMA_OVERRIDES_SIDECAR_FILENAME,
    SCHEMA_OVERRIDES_VERSION,
    STRING_VALUE_TYPES,
    VALID_COLUMN_OVERRIDE_KEYS,
    VALID_FK_ADD_KEYS,
    VALID_FK_KINDS,
    VALID_FK_REMOVE_KEYS,
    VALID_PK_ADD_KEYS,
    VALID_PK_REMOVE_KEYS,
    VALID_SENSITIVITY_LEVELS,
    VALID_TABLE_OVERRIDE_KEYS,
    VALID_TOP_LEVEL_OVERRIDE_KEYS,
    EngineConfig,
    PolicyConfig,
    llm_credentials_configured,
)
from ._contracts_base import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    ColumnRole,
    DatabaseFeatureCapability,
    DescriptionOwner,
    FKEdge,
    InferenceTag,
    MigrationTier,
    OverrideReport,
    OverrideSkip,
    PkInferenceTag,
    RoleOwner,
    SchemaAccessError,
    SchemaContext,
    SchemaGraph,
    SchemaInclude,
    SchemaInvariantError,
    SchemaLimits,
    SensitivityClassification,
    SidecarReconcileReport,
    TableMetadata,
    TableRole,
    can_overwrite_role,
    column_sensitivity_from_dict,
    data_type_to_value_type,
    is_date_type,
    is_numeric_type,
    sensitivity_classification_from_legacy_fields,
    set_description,
    set_sensitivity,
    set_schema_helpers,
)
from ._core_utils import (
    artifact_lock,
    debug,
    effective_structural_hash_fp,
    llm_chat,
    notify,
    profiling_hash_fp,
    read_gzip_json,
    schema_hash_fp,
    scope_hash_fp,
    stable_json,
    structural_hash_fp,
    wipe_versioned_artifacts,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from ._qsim import get_aggregatable_columns, get_groupable_columns
from ._schema_profiling import (
    apply_boolean_coercion_pass,
    apply_column_roles_llm,
    assign_column_ops,
    collect_profiling_topk_values,
    compute_semantic_profile_join_neighbors,
    extract_tables_from_catalog,
    extract_tables_from_catalog_sql_connector,
    parse_sql_file,
    replay_user_semantic_neighbors_to_columns,
    _llm_classify_schema,
)

if TYPE_CHECKING:
    from ._dialect import Dialect


def _notes_content_sha256(notes_content: str | None) -> str:
    """Return SHA-256 hex digest of UTF-8 notes text, or of empty bytes when None."""

    body = (notes_content or "").encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sql_file_content_sha256(sql_file: str | None) -> str:
    """Return SHA-256 of *sql_file* contents (UTF-8) or empty-string digest when missing."""

    if not sql_file:
        return hashlib.sha256(b"").hexdigest()
    expanded = os.path.expanduser(str(sql_file))
    if not os.path.isfile(expanded):
        return hashlib.sha256(b"").hexdigest()
    try:
        with open(expanded, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        debug(f"[schema._sql_file_content_sha256] read failed for {sql_file!r}: {exc!r}")
        return hashlib.sha256(b"").hexdigest()


def compute_dialect_probe(dialect: Dialect, schema_context: SchemaContext) -> str:
    """
    Return the combined DDL probe: dialect ``information_schema`` digest XOR'd with the ``sql_file`` content digest.

    The combination is a SHA-256 over the two hex digests joined by ``|``. Returns ``""`` when the dialect probe itself is empty (so the caller falls back to fingerprint validation); otherwise always returns a non-empty digest, even when ``sql_file`` is absent.

    Note on collision risk: the joined-digest construction inherits SHA-256 collision resistance, but a hypothetical adversary who can simultaneously alter both the catalog DDL and the local ``sql_file`` in offsetting ways could in theory produce the same final digest. This is negligible in practice (no adversarial input is involved during cache validation), and the only consequence would be a false cache-hit that the downstream structural fingerprint check is expected to surface; documented here for future auditors.
    """

    dialect_part = ""
    try:
        dialect_part = dialect.compute_ddl_probe(schema_context) or ""
    except Exception as exc:
        debug(f"[schema.compute_dialect_probe] dialect probe raised, treating as empty: {exc!r}")
        dialect_part = ""
    if not dialect_part:
        return ""
    file_part = _sql_file_content_sha256(getattr(schema_context, "sql_file", None))
    return hashlib.sha256(f"{dialect_part}|{file_part}".encode()).hexdigest()


def rerun_column_classifier(
    sg: SchemaGraph,
    notes_content: str | None,
    *,
    skip_columns: set[tuple[str, str]] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """
    Re-run the LLM column-role classifier and the deterministic boolean coercion pass over *sg* in place.

    Used by the notes-only refresh fast path in :func:`build_schema_graph` so a domain-notes edit can update roles, descriptions, and sensitivity without re-reflecting or re-profiling the database. ``skip_columns`` (table, column) pairs are not overwritten by the LLM so user-pinned roles/sensitivities survive a notes-driven reclassification.
    """

    debug(f"[schema.rerun_column_classifier] reclassifying {len(sg.tables)} tables")
    apply_column_roles_llm(sg, notes_content=notes_content, skip_columns=skip_columns, log_sink=log_sink)
    apply_boolean_coercion_pass(sg)
    assign_column_ops(sg)


def _user_pinned_columns_from_sidecar(
    schema_json_path: str | Path,
) -> set[tuple[str, str]]:
    """Return ``(table, column)`` pairs pinned by explicit role, sensitivity, or user-owned description."""

    sidecar = load_overrides_sidecar(schema_json_path)
    if sidecar is None:
        return set()
    pinned: set[tuple[str, str]] = set()
    tables_block = sidecar.get("tables") or {}
    if not isinstance(tables_block, dict):
        return pinned
    for tname, tval in tables_block.items():
        if not isinstance(tval, dict):
            continue
        cols = tval.get("columns") or {}
        if not isinstance(cols, dict):
            continue
        for cname, cval in cols.items():
            if not isinstance(cval, dict):
                continue
            if "role" in cval or "sensitivity" in cval or "pii" in cval:
                pinned.add((str(tname), str(cname)))
            elif "description" in cval:
                desc = cval["description"]
                if isinstance(desc, dict) and desc.get("owner") == DescriptionOwner.USER_OVERRIDE.value:
                    pinned.add((str(tname), str(cname)))
    return pinned


def compute_schema_stats(schema: SchemaGraph) -> dict[str, Any]:
    """
    Compute schema-wide column availability statistics for adaptive limit calculation.

    Args:

        schema: Populated `SchemaGraph` with profiled column metadata.

    Returns:

        `filterable_per_table` (list of per-table counts).
    """
    debug("[schema.compute_schema_stats] computing schema statistics for adaptive limits")

    stats = {
        "total_filterable": 0,
        "total_groupable": 0,
        "total_aggregatable": 0,
        "min_filterable_per_table": float("inf"),
        "max_filterable_per_table": 0,
        "min_groupable_per_table": float("inf"),
        "max_groupable_per_table": 0,
        "table_count": len(schema.tables),
        "filterable_per_table": [],
    }

    table_details = []
    for table_name, table in schema.tables.items():
        filterable_count = sum(1 for col in table.columns.values() if col.is_filterable)
        groupable_count = sum(1 for col in table.columns.values() if col.is_groupable)
        aggregatable_count = sum(1 for col in table.columns.values() if col.is_aggregatable)

        table_details.append(
            {
                "table": table_name,
                "filterable": filterable_count,
                "groupable": groupable_count,
                "aggregatable": aggregatable_count,
            }
        )

        stats["total_filterable"] += filterable_count
        stats["total_groupable"] += groupable_count
        stats["total_aggregatable"] += aggregatable_count

        if filterable_count > 0:
            stats["min_filterable_per_table"] = min(stats["min_filterable_per_table"], filterable_count)
            stats["max_filterable_per_table"] = max(stats["max_filterable_per_table"], filterable_count)
            stats["filterable_per_table"].append(filterable_count)

        if groupable_count > 0:
            stats["min_groupable_per_table"] = min(stats["min_groupable_per_table"], groupable_count)
            stats["max_groupable_per_table"] = max(stats["max_groupable_per_table"], groupable_count)

    if stats["min_filterable_per_table"] == float("inf"):
        stats["min_filterable_per_table"] = 0
    if stats["min_groupable_per_table"] == float("inf"):
        stats["min_groupable_per_table"] = 0

    debug("[schema.compute_schema_stats] per-table column counts:")
    for td in table_details:
        debug(
            f"  {td['table']}: filterable={td['filterable']}, groupable={td['groupable']}, aggregatable={td['aggregatable']}"
        )

    debug("[schema.compute_schema_stats] schema-wide statistics:")
    debug(f"  table_count: {stats['table_count']}")
    debug(f"  total_filterable: {stats['total_filterable']}")
    debug(f"  total_groupable: {stats['total_groupable']}")
    debug(f"  total_aggregatable: {stats['total_aggregatable']}")
    debug(f"  min_filterable_per_table: {stats['min_filterable_per_table']}")
    debug(f"  max_filterable_per_table: {stats['max_filterable_per_table']}")
    debug(f"  min_groupable_per_table: {stats['min_groupable_per_table']}")
    debug(f"  max_groupable_per_table: {stats['max_groupable_per_table']}")
    debug(f"  filterable_per_table distribution: {stats['filterable_per_table']}")

    return stats


def compute_schema_limits(schema_stats: dict[str, Any]) -> SchemaLimits:
    """
    Compute adaptive pipeline limits from schema statistics.

    Args:

        schema_stats: Dictionary as returned by `compute_schema_stats`.

    Returns:

        `SchemaLimits` with `max_filters`, `max_groupby`, and `max_tables`.
    """
    table_count = schema_stats.get("table_count", 1)
    total_filterable = schema_stats.get("total_filterable", 0)
    total_groupable = schema_stats.get("total_groupable", 0)

    max_filters = max(1, total_filterable // table_count) if table_count > 0 else 1
    max_groupby = max(1, total_groupable // table_count) if table_count > 0 else 1

    if table_count <= 3:
        max_tables = table_count
    elif table_count <= 10:
        max_tables = 3
    else:
        max_tables = 4

    return SchemaLimits(
        max_filters=max_filters,
        max_groupby=max_groupby,
        max_tables=max_tables,
    )


def _edge_key(e: FKEdge) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """
    Generate a stable, sortable tuple key for an FK edge.

    Args:

        e: `FKEdge` to key.

    Returns:

        Tuple of `(src_table, src_cols_tuple, dst_table, dst_cols_tuple)`.
    """
    return (e.src_table, tuple(e.src_cols), e.dst_table, tuple(e.dst_cols))


def _table_to_dict(table: TableMetadata) -> dict[str, Any]:
    """
    Serialize a TableMetadata instance to a plain dictionary.

    Args:

        table: `TableMetadata` to serialize.

    Returns:

        `role`, `row_count`, and `description` fields.
    """
    return {
        "name": table.name,
        "kind": table.kind,
        "columns": {k: asdict(v) for k, v in table.columns.items()},
        "primary_key": table.primary_key,
        "foreign_keys": [asdict(fk) for fk in table.foreign_keys],
        "partition_columns": table.partition_columns,
        "role": table.role,
        "role_owner": table.role_owner.value if table.role_owner is not None else None,
        "row_count": table.row_count,
        "description": table.description,
        "description_owner": (table.description_owner.value if table.description_owner is not None else None),
        "composite_descriptive_ratios": {
            f"{c1}|{c2}": ratio for (c1, c2), ratio in table.composite_descriptive_ratios.items()
        },
    }


def load_schema_graph_snapshot(path: str) -> SchemaGraph | None:
    """Load ``SchemaGraph`` from a gzip JSON cache at *path*, or ``None`` when unavailable."""

    if not path or not os.path.isfile(path):
        return None
    try:
        d = read_gzip_json(path)
        tables_raw = d.get("tables", {})
        if not isinstance(tables_raw, dict):
            return None
        model: dict[str, Any] = dict(d)
        if "join_paths_multi" not in model and "join_paths" in model:
            jp = d.get("join_paths", {})
            join_paths_multi: dict[str, Any] = {}
            for a in jp:
                join_paths_multi[a] = {}
                for b in jp[a]:
                    join_paths_multi[a][b] = [jp[a][b]] if jp[a][b] is not None else []
            model["join_paths_multi"] = join_paths_multi
        return SchemaGraph.from_dict(model)
    except (
        OSError,
        EOFError,
        gzip.BadGzipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def _table_from_dict(d: dict[str, Any]) -> TableMetadata:
    """
    Deserialize a TableMetadata instance from a plain dictionary.

    Args:

        d: Dictionary with keys matching `TableMetadata` fields.

    Returns:

        objects.
    """
    columns = {k: ColumnMetadata.from_dict(v) for k, v in d["columns"].items()}
    kind_raw = d.get("kind", "table")
    kind: Literal["table", "view"] = "table" if kind_raw == "table" else "view"
    return TableMetadata(
        name=d["name"],
        columns=columns,
        primary_key=d["primary_key"],
        foreign_keys=[FKEdge(**fk) for fk in d["foreign_keys"]],
        kind=kind,
        partition_columns=d.get("partition_columns", []),
        role=d.get("role"),
        row_count=d.get("row_count", 0),
        description=d.get("description", ""),
        composite_descriptive_ratios={
            tuple(k.split("|", 1)): v for k, v in (d.get("composite_descriptive_ratios") or {}).items() if "|" in k
        },
    )


def _load_pg_enum_values(engine: Any) -> dict[str, list[str]]:
    """Load PostgreSQL enum labels keyed by lowercased type name."""

    buckets: dict[str, list[str]] = {}
    stmt = text(
        """
        SELECT lower(t.typname::text) AS typname, e.enumlabel::text AS lbl
        FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY typname, e.enumsortorder
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt)
        for typname, lbl in rows:
            buckets.setdefault(str(typname), []).append(str(lbl))
    return buckets


def _graph_tables_lower_index(tables: dict[str, TableMetadata]) -> dict[str, str]:
    """Map lowercased relation name to the graph's canonical table key."""

    return {name.lower(): name for name in tables}


def _column_names_lower_index(columns: dict[str, ColumnMetadata]) -> dict[str, str]:
    """Map lowercased column name to the canonical column key."""

    return {col.lower(): col for col in columns}


def _semantic_neighbor_edge_count(sg: SchemaGraph) -> int:
    """Count undirected semantic neighbor pairs recorded on column metadata."""

    total = 0
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            total += len(col.semantic_join_neighbors)
    return total


def _effective_reflect_include(ctx: SchemaContext) -> SchemaInclude:
    """Use catalog-wide reflection when an explicit allow-list is provided."""

    if ctx.allow_objects:
        return "both"
    return ctx.include


def _allow_objects_lower_set(
    allow_objects: frozenset[str] | None,
) -> frozenset[str] | None:
    """Return lowercase relation names for filtering, or ``None`` when no allow-list."""

    if not allow_objects:
        return None
    return frozenset(str(x).lower() for x in allow_objects)


def _merge_reflected_schema_graphs(a: SchemaGraph, b: SchemaGraph) -> SchemaGraph:
    """Merge two reflected graphs keyed by relation name."""

    merged_tables = dict(a.tables)
    for name, meta in b.tables.items():
        if name in merged_tables:
            existing = merged_tables[name]
            if existing.kind != meta.kind:
                raise SchemaAccessError(
                    f"ambiguous relation name {name!r} resolves to both a table and a view in the catalog",
                )
        merged_tables[name] = meta
    join_paths_multi = _recompute_join_paths_multi(merged_tables)
    return SchemaGraph(
        tables=merged_tables,
        join_paths_multi=join_paths_multi,
        created_at=datetime.now().isoformat(),
        enum_values=a.enum_values or b.enum_values,
    )


def _apply_allow_objects_filter(sg: SchemaGraph, ctx: SchemaContext) -> None:
    """Restrict the graph to explicitly allowed relation names."""

    if not ctx.allow_objects:
        return
    tindex = _graph_tables_lower_index(sg.tables)
    want = {str(x).lower() for x in ctx.allow_objects}
    kept: dict[str, TableMetadata] = {}
    for low in sorted(want):
        canon = tindex.get(low)
        if not canon or canon not in sg.tables:
            raise SchemaAccessError(f"allow_objects references unknown relation: {low!r}")
        kept[canon] = sg.tables[canon]
    sg.tables = kept
    sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)


def _semantic_edges_fingerprint(tables: dict[str, TableMetadata]) -> str:
    """Stable digest of semantic join neighbor tuples for migration tiering."""

    edges: list[list[str]] = []
    for tn in sorted(tables):
        for cn in sorted(tables[tn].columns):
            col = tables[tn].columns[cn]
            for nb in sorted(col.semantic_join_neighbors, key=lambda x: tuple(str(p) for p in x)):
                edges.append([tn, cn, *[str(p) for p in nb]])
    return hashlib.sha256(stable_json({"semantic_edges": edges}).encode("utf-8")).hexdigest()


def _validate_scope_against_graph(sg: SchemaGraph, ctx: SchemaContext) -> None:
    """Ensure deny_columns reference relations and columns present in the graph."""

    reasons: list[str] = []
    offending: list[str] = []
    tindex = _graph_tables_lower_index(sg.tables)
    for tbl_raw, col_raw in ctx.qualified_denies():
        if tbl_raw not in tindex:
            reasons.append(f"deny_columns references unknown table: {tbl_raw}")
            offending.append(tbl_raw)
            continue
        canon = tindex[tbl_raw]
        cix = _column_names_lower_index(sg.tables[canon].columns)
        if col_raw not in cix:
            reasons.append(f"deny_columns references unknown column: {tbl_raw}.{col_raw}")
            offending.append(f"{canon}.{col_raw}")
    for gcol in ctx.glob_column_denies():
        any_hit = False
        for _tbl, tab in sg.tables.items():
            if gcol in _column_names_lower_index(tab.columns):
                any_hit = True
                break
        if not any_hit:
            reasons.append(f"deny_columns glob '*.{gcol}' matches no column in scope")
            offending.append(f"*.{gcol}")
    for tbl_raw, col_raw in ctx.qualified_allows():
        if tbl_raw not in tindex:
            reasons.append(f"allow_columns references unknown table: {tbl_raw}")
            offending.append(tbl_raw)
            continue
        canon = tindex[tbl_raw]
        cix = _column_names_lower_index(sg.tables[canon].columns)
        if col_raw not in cix:
            reasons.append(f"allow_columns references unknown column: {tbl_raw}.{col_raw}")
            offending.append(f"{canon}.{col_raw}")
    for gcol in ctx.glob_column_allows():
        any_hit = False
        for _tbl, tab in sg.tables.items():
            if gcol in _column_names_lower_index(tab.columns):
                any_hit = True
                break
        if not any_hit:
            reasons.append(f"allow_columns glob '*.{gcol}' matches no column in scope")
            offending.append(f"*.{gcol}")
    if reasons:
        raise SchemaAccessError("; ".join(reasons))


def _deny_columns_by_table(sg: SchemaGraph, ctx: SchemaContext) -> dict[str, set[str]]:
    """Resolve ``ctx`` deny specs to canonical ``{table: {column, ...}}`` against *sg*."""

    deny_by_table: dict[str, set[str]] = {}
    tindex = _graph_tables_lower_index(sg.tables)
    for tbl_raw, col_raw in ctx.qualified_denies():
        canon_tbl = tindex.get(tbl_raw)
        if canon_tbl is None:
            continue
        cix = _column_names_lower_index(sg.tables[canon_tbl].columns)
        canon_col = cix.get(col_raw)
        if canon_col is None:
            continue
        deny_by_table.setdefault(canon_tbl, set()).add(canon_col)
    for gcol in ctx.glob_column_denies():
        for canon_tbl, tbl in sg.tables.items():
            cix = _column_names_lower_index(tbl.columns)
            canon_col = cix.get(gcol)
            if canon_col is None:
                continue
            deny_by_table.setdefault(canon_tbl, set()).add(canon_col)
    return deny_by_table


def _prune_foreign_keys_after_column_removal(sg: SchemaGraph) -> None:
    """Drop FK edges whose source or destination columns were removed from the graph."""

    tindex = _graph_tables_lower_index(sg.tables)
    for canon_tbl, tbl in sg.tables.items():
        kept: list[FKEdge] = []
        for fk in tbl.foreign_keys:
            if any(c not in tbl.columns for c in fk.src_cols):
                continue
            dst_res = tindex.get(str(fk.dst_table).lower())
            if dst_res is None:
                continue
            dst_tbl = sg.tables.get(dst_res)
            if dst_tbl is None or any(c not in dst_tbl.columns for c in fk.dst_cols):
                continue
            kept.append(fk)
        tbl.foreign_keys = kept


def _strip_schema_context_denied_columns(sg: SchemaGraph, ctx: SchemaContext) -> None:
    """
    Remove denied columns from ``TableMetadata.columns`` before profiling.

    Prunes foreign keys that referenced removed endpoints, clears ``SchemaGraph.deny_columns`` because denied names no longer exist as rows, and leaves the authoritative deny specification on the frozen ``SchemaContext`` passed into the build.
    """

    deny_by_table = _deny_columns_by_table(sg, ctx)
    for canon_tbl, cols in deny_by_table.items():
        tbl = sg.tables.get(canon_tbl)
        if tbl is None:
            continue
        for col_name in cols:
            tbl.columns.pop(col_name, None)
        tbl.primary_key = [c for c in tbl.primary_key if c in tbl.columns]
    _prune_foreign_keys_after_column_removal(sg)
    sg.deny_columns = {}


def _apply_schema_context_allow_columns(sg: SchemaGraph, ctx: SchemaContext) -> None:
    """
    Restrict each table to its ``allow_columns`` subset; PK and FK columns are always retained.

    No-op when ``ctx.allow_columns`` is empty. Glob ``*.column`` entries match that column name on every table where it exists. Qualified ``table.column`` entries scope to one table. Primary key columns and any column appearing in a foreign key edge (source or destination) are auto-included so the join graph survives a narrow allow list.
    """

    if not ctx.allow_columns:
        sg.disallowed_columns = {}
        return
    sg.disallowed_columns = {}
    tindex = _graph_tables_lower_index(sg.tables)
    qualified: dict[str, set[str]] = {}
    for tbl_raw, col_raw in ctx.qualified_allows():
        canon_tbl = tindex.get(tbl_raw)
        if canon_tbl is None:
            continue
        cix = _column_names_lower_index(sg.tables[canon_tbl].columns)
        canon_col = cix.get(col_raw)
        if canon_col is None:
            continue
        qualified.setdefault(canon_tbl, set()).add(canon_col)
    bare = set(ctx.glob_column_allows())
    fk_columns_by_table: dict[str, set[str]] = {}
    for canon_tbl, tbl in sg.tables.items():
        for fk in tbl.foreign_keys:
            for c in fk.src_cols:
                fk_columns_by_table.setdefault(canon_tbl, set()).add(c)
            dst_canon = tindex.get(str(fk.dst_table).lower())
            if dst_canon is not None:
                for c in fk.dst_cols:
                    fk_columns_by_table.setdefault(dst_canon, set()).add(c)
    for canon_tbl, tbl in sg.tables.items():
        keep: set[str] = set(qualified.get(canon_tbl, set()))
        cix = _column_names_lower_index(tbl.columns)
        for bare_col in bare:
            canon_col = cix.get(bare_col)
            if canon_col is not None:
                keep.add(canon_col)
        for pk_col in tbl.primary_key:
            if pk_col in tbl.columns:
                keep.add(pk_col)
        for col_name, col in tbl.columns.items():
            if col.is_primary_key or col.is_foreign_key:
                keep.add(col_name)
        for fk_col in fk_columns_by_table.get(canon_tbl, set()):
            if fk_col in tbl.columns:
                keep.add(fk_col)
        removed = set(tbl.columns.keys()) - keep
        if removed:
            sg.disallowed_columns[canon_tbl] = set(removed)
        tbl.columns = {name: col for name, col in tbl.columns.items() if name in keep}


def _scope_is_subset_or_equal(narrow: SchemaContext, wide: SchemaContext) -> bool:
    """Return True iff every (table, column) visible under *narrow* is also visible under *wide*."""

    if narrow.include != wide.include and wide.include != "both":
        return False

    if wide.allow_objects:
        if not narrow.allow_objects:
            return False
        if not narrow.allow_objects.issubset(wide.allow_objects):
            return False

    if not narrow.deny_columns.issuperset(wide.deny_columns):
        return False

    if wide.allow_columns:
        if not narrow.allow_columns:
            return False
        if not narrow.allow_columns.issubset(wide.allow_columns):
            return False
    elif narrow.allow_columns:
        return True
    return True


def classify_scope_change(
    old: SchemaContext, new: SchemaContext
) -> Literal["identical", "subset", "superset", "orthogonal"]:
    """
    Classify the relationship between *old* and *new* scope contexts.

    Returns:
        ``"identical"``  - all four scope-relevant fields (allow_objects, deny_columns, allow_columns, include) compare equal.
        ``"subset"``     - every (table, column) visible under *new* is also visible under *old* (new is strictly narrower).
        ``"superset"``   - every (table, column) visible under *old* is also visible under *new* (new is strictly broader).
        ``"orthogonal"`` - neither containment direction holds.
    """

    if (
        old.allow_objects == new.allow_objects
        and old.deny_columns == new.deny_columns
        and old.allow_columns == new.allow_columns
        and old.include == new.include
    ):
        return "identical"
    new_le_old = _scope_is_subset_or_equal(new, old)
    old_le_new = _scope_is_subset_or_equal(old, new)
    if new_le_old and not old_le_new:
        return "subset"
    if old_le_new and not new_le_old:
        return "superset"
    return "orthogonal"


def _filter_schema_graph_by_scope(sg: SchemaGraph, new_ctx: SchemaContext) -> SchemaGraph:
    """
    Pure (no-I/O) deepcopy of *sg* with *new_ctx*'s allow/deny rules applied.

    Drops tables not in ``new_ctx.allow_objects``, removes ``deny_columns`` targets from column maps (pruning foreign keys), restricts each table's columns to ``new_ctx.allow_columns`` (auto-keeping PK/FK columns), and recomputes ``join_paths_multi``. Used by the scope-subset fast path in :func:`build_schema_graph`.
    """

    new_sg = copy.deepcopy(sg)

    new_sg.deny_columns = {}
    _apply_allow_objects_filter(new_sg, new_ctx)
    _strip_schema_context_denied_columns(new_sg, new_ctx)
    _apply_schema_context_allow_columns(new_sg, new_ctx)
    new_sg.join_paths_multi = _recompute_join_paths_multi(new_sg.tables)
    return new_sg


@dataclass(frozen=True)
class TableDiff:
    """
    Per-table delta between a cached and a freshly-reflected ``SchemaGraph``.

    Entries are sorted tuples to keep equality + hashing deterministic in tests.

    ``retyped_columns`` records catalog type changes where the normalized ``value_type`` changes (profile must be refreshed). ``redeclared_columns`` holds pure ``data_type`` widenings (for example ``varchar(50)`` to ``text``) where ``value_type`` is unchanged; those updates merge metadata without clearing profiling samples.

    ``value_type_changed_columns`` mirrors the ``(column, old_vt, new_vt)`` entries implied by ``retyped_columns``.

    ``renamed_columns`` is populated by :func:`resolve_column_renames` after profile overlap matching; columns appearing here are removed from ``added_columns`` / ``dropped_columns``.
    """

    added_columns: tuple[str, ...] = ()
    dropped_columns: tuple[str, ...] = ()
    redeclared_columns: tuple[tuple[str, str, str], ...] = ()
    retyped_columns: tuple[tuple[str, str, str], ...] = ()
    value_type_changed_columns: tuple[tuple[str, str, str], ...] = ()
    renamed_columns: tuple[tuple[str, str], ...] = ()
    fk_changed: bool = False
    pk_changed: bool = False

    @property
    def is_empty(self) -> bool:
        return (
            not self.added_columns
            and not self.dropped_columns
            and not self.redeclared_columns
            and not self.retyped_columns
            and not self.renamed_columns
            and not self.fk_changed
            and not self.pk_changed
        )

    @property
    def needs_profile(self) -> bool:
        """
        True when applying this diff requires re-profiling the table.

        Pure-rename tables keep cached profiles. Adds and value-type retypes always need profiling; pure ``redeclared_columns`` (same ``value_type``) do not. Tables whose catalog PK or FK edge sets changed are pulled into :meth:`SchemaDiff.changed_table_names` so subset reprofiling refreshes statistics on those relations even when no columns were added or retyped.
        """
        return bool(self.added_columns or self.retyped_columns)


@dataclass
class SchemaDiff:
    """Whole-graph delta consumed by :func:`apply_diff` and downstream invalidation."""

    added_tables: tuple[str, ...] = ()
    dropped_tables: tuple[str, ...] = ()
    table_renames: tuple[tuple[str, str], ...] = ()
    per_table: dict[str, TableDiff] = field(default_factory=dict)
    dropped_user_fks: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    dropped_catalog_fks: list[tuple[str, str, str, str]] = field(default_factory=list)
    ported_user_fks: list[tuple[str, str, str, str, str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.added_tables and not self.dropped_tables and not self.table_renames and not self.per_table

    def implies_rename_remapping(self) -> bool:
        """True when template rename migration should treat this diff as a REMAP-tier rename."""

        if self.table_renames:
            return True
        return any(td.renamed_columns for td in self.per_table.values())

    def changed_table_names(self) -> set[str]:
        """Tables in the *new* graph that need subset profiling (adds, retypes, catalog PK/FK shape changes)."""
        out: set[str] = set(self.added_tables)
        for _old, new in self.table_renames:
            out.add(new)
        for tname, td in self.per_table.items():
            if td.needs_profile or td.pk_changed or td.fk_changed:
                out.add(tname)
        return out


def _table_column_typed_set(t: TableMetadata) -> frozenset[tuple[str, str]]:
    """Structural multiset for rename detection using normalized ``value_type``, not raw ``data_type``."""

    out: list[tuple[str, str]] = []
    for c in t.columns.values():
        vt = (c.value_type or "").strip().lower()
        if not vt:
            vt = data_type_to_value_type(c.data_type)
        out.append((c.name, vt))
    return frozenset(out)


def _fk_edge_set(
    t: TableMetadata,
) -> frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    return frozenset(_edge_key(fk) for fk in t.foreign_keys)


def _catalog_fk_edge_set(
    t: TableMetadata,
) -> frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    """
    Return only the catalog-declared FK edges (``inference_tag is None``).

    The diff uses this rather than the full edge set so that inferred or user-override edges sitting only on the cached graph do not register as a structural change between the cache and a fresh reflection (which never carries non-catalog tags).
    """

    return frozenset(_edge_key(fk) for fk in t.foreign_keys if fk.inference_tag is None)


def diff_schemas(old_sg: SchemaGraph, new_sg: SchemaGraph) -> SchemaDiff:
    """
    Pure structural diff between two :class:`SchemaGraph` instances.

    Detects column-stable table renames (same column-name+value-type multiset) before per-table column diffing. FK changes are reported as a single boolean per table (handlers re-copy the FK list wholesale; finer-grained join-path repair is left to later migration handling).
    """

    old_names = set(old_sg.tables.keys())
    new_names = set(new_sg.tables.keys())
    dropped_only = old_names - new_names
    added_only = new_names - old_names

    renames: list[tuple[str, str]] = []
    used_added: set[str] = set()
    for old_name in sorted(dropped_only):
        old_sig = _table_column_typed_set(old_sg.tables[old_name])
        if not old_sig:
            continue
        match: str | None = None
        for new_name in sorted(added_only - used_added):
            if _table_column_typed_set(new_sg.tables[new_name]) == old_sig:
                match = new_name
                break
        if match is not None:
            renames.append((old_name, match))
            used_added.add(match)

    renamed_old = {o for o, _n in renames}
    renamed_new = {n for _o, n in renames}
    final_dropped = tuple(sorted(dropped_only - renamed_old))
    final_added = tuple(sorted(added_only - renamed_new))

    per_table: dict[str, TableDiff] = {}
    surviving = old_names & new_names

    for name in sorted(surviving):
        old_t = old_sg.tables[name]
        new_t = new_sg.tables[name]
        old_cols = set(old_t.columns.keys())
        new_cols = set(new_t.columns.keys())
        added_cols = tuple(sorted(new_cols - old_cols))
        dropped_cols = tuple(sorted(old_cols - new_cols))
        retyped: list[tuple[str, str, str]] = []
        redeclared: list[tuple[str, str, str]] = []
        vt_changed: list[tuple[str, str, str]] = []
        for c in sorted(old_cols & new_cols):
            old_dt = old_t.columns[c].data_type
            new_dt = new_t.columns[c].data_type
            if old_dt == new_dt:
                continue
            old_vt = data_type_to_value_type(old_dt)
            new_vt = data_type_to_value_type(new_dt)
            if old_vt != new_vt:
                retyped.append((c, old_dt, new_dt))
                vt_changed.append((c, old_vt, new_vt))
            else:
                redeclared.append((c, old_dt, new_dt))
        fk_changed = _catalog_fk_edge_set(old_t) != _catalog_fk_edge_set(new_t)
        pk_changed = sorted(old_t.primary_key) != sorted(new_t.primary_key)
        td = TableDiff(
            added_columns=added_cols,
            dropped_columns=dropped_cols,
            redeclared_columns=tuple(redeclared),
            retyped_columns=tuple(retyped),
            value_type_changed_columns=tuple(vt_changed),
            fk_changed=fk_changed,
            pk_changed=pk_changed,
        )
        if not td.is_empty:
            per_table[name] = td

    result = SchemaDiff(
        added_tables=final_added,
        dropped_tables=final_dropped,
        table_renames=tuple(renames),
        per_table=per_table,
    )
    debug(
        "[schema.diff_schemas] "
        f"+tables={len(result.added_tables)} -tables={len(result.dropped_tables)} "
        f"renames={len(result.table_renames)} per_table={len(result.per_table)}"
    )
    return result


def _profile_subset(
    dialect: Dialect,
    target_sg: SchemaGraph,
    table_names: set[str],
    notes_content: str | None,
) -> None:
    """
    Run profiling + LLM classifier on a *subset* of tables in *target_sg*, in-place.

    Builds a temporary :class:`SchemaGraph` containing only the named tables (sharing the same :class:`TableMetadata` objects so mutations propagate to *target_sg*) and asks the dialect to profile it. The same temp graph is fed through the LLM column classifier so only the changed tables incur LLM cost.
    """
    if not table_names:
        return
    subset_tables = {n: target_sg.tables[n] for n in table_names if n in target_sg.tables}
    if not subset_tables:
        return
    tmp_sg = SchemaGraph(tables=subset_tables, join_paths_multi={})
    dialect.profile_schema(tmp_sg)
    apply_column_roles_llm(tmp_sg, notes_content=notes_content)
    apply_boolean_coercion_pass(tmp_sg)
    assign_column_ops(tmp_sg)


def _merge_llm_descriptions_and_roles_for_tables(
    sg: SchemaGraph,
    *,
    table_names: set[str],
    classifications: dict[str, tuple[str, str, dict[str, tuple[str, str, str | None]]]],
) -> None:
    """Apply table/column descriptions and LLM-owned roles for *table_names* only; never writes sensitivity."""

    for tname in table_names:
        table = sg.tables.get(tname)
        if table is None or tname not in classifications:
            continue
        table_role, description, column_classifications = classifications[tname]
        if can_overwrite_role(table.role_owner, RoleOwner.LLM):
            table.role = table_role
            table.role_owner = RoleOwner.LLM
        set_description(table, description, DescriptionOwner.LLM_REFINEMENT)
        for col in table.columns.values():
            if col.name not in column_classifications:
                continue
            role, hint, _sensitivity = column_classifications[col.name]
            if can_overwrite_role(col.role_owner, RoleOwner.LLM):
                col.role = role
                col.role_owner = RoleOwner.LLM
            set_description(col, hint, DescriptionOwner.LLM_REFINEMENT)


def _refresh_existing_descriptions_after_addition(
    cached_sg: SchemaGraph,
    diff: SchemaDiff,
    notes_content: str | None,
) -> None:
    """Full-graph classify, then merge descriptions/roles for tables outside ``diff.changed_table_names()``."""

    changed = diff.changed_table_names()
    unchanged = set(cached_sg.tables) - changed
    if not unchanged:
        return
    try:
        classifications = _llm_classify_schema(cached_sg, notes_content)
    except Exception as exc:
        debug(f"[schema._refresh_existing_descriptions_after_addition] full-graph classify failed: {exc!r}")
        return
    _merge_llm_descriptions_and_roles_for_tables(
        cached_sg,
        table_names=unchanged,
        classifications=classifications,
    )
    apply_boolean_coercion_pass(cached_sg)
    assign_column_ops(cached_sg)


def _column_topk_set(col: ColumnMetadata) -> frozenset[str]:
    """Top-K profiling values normalised for Jaccard comparison."""
    vals = col.top_k_values or []
    cleaned = {str(v).strip() for v in vals if v is not None and str(v).strip() != ""}
    return frozenset(cleaned)


def _column_jaccard(a: ColumnMetadata, b: ColumnMetadata) -> float:
    """Jaccard overlap of two columns' Top-K profiling sets (0.0 when both are empty)."""
    sa = _column_topk_set(a)
    sb = _column_topk_set(b)
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return len(sa & sb) / union


def _profile_table_clone(
    dialect: Dialect,
    table: TableMetadata,
    notes_content: str | None,
) -> TableMetadata | None:
    """
    Deep-copy *table* and run profiling/classification against it; return the clone.

    Returns ``None`` when profiling raises so callers can fall back to drop+add.
    """
    clone = copy.deepcopy(table)
    tmp_sg = SchemaGraph(tables={clone.name: clone}, join_paths_multi={})
    try:
        dialect.profile_schema(tmp_sg)
        apply_column_roles_llm(tmp_sg, notes_content=notes_content)
        apply_boolean_coercion_pass(tmp_sg)
        assign_column_ops(tmp_sg)
    except Exception as exc:
        debug(f"[schema._profile_table_clone] profile failed for {table.name!r}: {exc!r}")
        return None
    return clone


def resolve_column_renames(
    diff: SchemaDiff,
    cached_sg: SchemaGraph,
    new_sg: SchemaGraph,
    dialect: Dialect,
    notes_content: str | None = None,
    *,
    threshold: float = MIGRATION_DATA_OVERLAP_MIN,
) -> SchemaDiff:
    """
    Detect per-table column renames by Top-K profile overlap.

    For each table in ``diff.per_table`` with both added *and* dropped columns, profile the new (added) columns and greedily match dropped→added pairs whose Jaccard overlap clears *threshold*. Confirmed pairs move from ``added_columns`` / ``dropped_columns`` into ``renamed_columns`` on a fresh :class:`TableDiff`. Unrelated tables are passed through unchanged. The resulting :class:`SchemaDiff` is independent of *diff*.
    """

    if not diff.per_table:
        return diff

    new_per_table: dict[str, TableDiff] = {}
    for tname, td in diff.per_table.items():
        if not td.added_columns or not td.dropped_columns:
            new_per_table[tname] = td
            continue
        if tname not in cached_sg.tables or tname not in new_sg.tables:
            new_per_table[tname] = td
            continue

        profiled_clone = _profile_table_clone(dialect, new_sg.tables[tname], notes_content)
        if profiled_clone is None:
            new_per_table[tname] = td
            continue

        cached_t = cached_sg.tables[tname]
        remaining_added = list(td.added_columns)
        remaining_dropped = list(td.dropped_columns)
        confirmed: list[tuple[str, str]] = []

        for old_col_name in list(td.dropped_columns):
            old_col = cached_t.columns.get(old_col_name)
            if old_col is None:
                continue
            best_name: str | None = None
            best_score = -1.0
            for new_col_name in remaining_added:
                new_col = profiled_clone.columns.get(new_col_name)
                if new_col is None:
                    continue
                score = _column_jaccard(old_col, new_col)
                if score > best_score:
                    best_score = score
                    best_name = new_col_name
            if best_name is not None and best_score >= threshold:
                new_col = profiled_clone.columns.get(best_name)
                if new_col is None:
                    continue
                old_vt = (old_col.value_type or "").strip().lower() or data_type_to_value_type(old_col.data_type)
                new_vt = (new_col.value_type or "").strip().lower() or data_type_to_value_type(new_col.data_type)
                if old_vt != new_vt:
                    top_intersection = len(_column_topk_set(old_col) & _column_topk_set(new_col))
                    if top_intersection < int(PolicyConfig.SEMANTIC_JOIN_MIN_INTERSECTION):
                        continue
                confirmed.append((old_col_name, best_name))
                remaining_added.remove(best_name)
                remaining_dropped.remove(old_col_name)

        if not confirmed:
            new_per_table[tname] = td
            continue

        debug(f"[schema.resolve_column_renames] {tname!r}: detected {len(confirmed)} column rename(s): {confirmed!r}")
        new_per_table[tname] = TableDiff(
            added_columns=tuple(remaining_added),
            dropped_columns=tuple(remaining_dropped),
            redeclared_columns=td.redeclared_columns,
            retyped_columns=td.retyped_columns,
            value_type_changed_columns=td.value_type_changed_columns,
            renamed_columns=tuple(sorted(confirmed)),
            fk_changed=td.fk_changed,
            pk_changed=td.pk_changed,
        )

    return SchemaDiff(
        added_tables=diff.added_tables,
        dropped_tables=diff.dropped_tables,
        table_renames=diff.table_renames,
        per_table=new_per_table,
    )


def resolve_table_renames(
    diff: SchemaDiff,
    cached_sg: SchemaGraph,
    new_sg: SchemaGraph,
    dialect: Dialect,
    notes_content: str | None = None,
    *,
    overlap_threshold: float = MIGRATION_DATA_OVERLAP_MIN,
    column_fraction: float = MIGRATION_TABLE_RENAME_COLUMN_FRACTION,
) -> SchemaDiff:
    """
    Detect simultaneous table renames (with optional column renames) via profile overlap.

    For each ``(dropped_table, added_table)`` pair with the same column count, profile the candidate added table, greedily match its columns to the dropped table's cached columns by Top-K Jaccard overlap, and accept the pair as a table rename when the matched-column overlap clears ``overlap_threshold`` for at least ``column_fraction`` of columns.

    Confirmed renames are removed from ``added_tables`` / ``dropped_tables`` and pushed into ``table_renames`` (plus per-table ``renamed_columns`` for any column renames).
    """

    if not diff.dropped_tables or not diff.added_tables:
        return diff

    remaining_dropped = list(diff.dropped_tables)
    remaining_added = list(diff.added_tables)
    new_table_renames = list(diff.table_renames)
    per_table: dict[str, TableDiff] = dict(diff.per_table)

    profiled_added: dict[str, TableMetadata] = {}

    def _candidate_score(
        old_table: TableMetadata,
        new_clone: TableMetadata,
    ) -> tuple[float, list[tuple[str, str]]]:
        """Return ``(matched_fraction, column_pairings)`` for an old↔new table pair."""

        old_cols = list(old_table.columns.values())
        new_col_names = list(new_clone.columns.keys())
        if not old_cols or not new_col_names:
            return (0.0, [])
        used: set[str] = set()
        pairings: list[tuple[str, str]] = []
        matched = 0
        for old_col in old_cols:
            best_name: str | None = None
            best_score = -1.0
            for nc_name in new_col_names:
                if nc_name in used:
                    continue
                nc = new_clone.columns[nc_name]
                score = _column_jaccard(old_col, nc)
                if score > best_score:
                    best_score = score
                    best_name = nc_name
            if best_name is not None:
                pairings.append((old_col.name, best_name))
                used.add(best_name)
                if best_score >= overlap_threshold:
                    matched += 1
        fraction = matched / float(len(old_cols))
        return (fraction, pairings)

    for old_name in list(diff.dropped_tables):
        if old_name not in cached_sg.tables or old_name not in remaining_dropped:
            continue
        old_t = cached_sg.tables[old_name]
        old_col_count = len(old_t.columns)
        best_match: tuple[str, float, list[tuple[str, str]]] | None = None
        for new_name in remaining_added:
            if new_name not in new_sg.tables:
                continue
            new_t_struct = new_sg.tables[new_name]
            if len(new_t_struct.columns) != old_col_count:
                continue
            if new_name not in profiled_added:
                clone = _profile_table_clone(dialect, new_t_struct, notes_content)
                if clone is None:
                    continue
                profiled_added[new_name] = clone
            fraction, pairings = _candidate_score(old_t, profiled_added[new_name])
            if fraction < column_fraction:
                continue
            if best_match is None or fraction > best_match[1]:
                best_match = (new_name, fraction, pairings)

        if best_match is None:
            continue

        new_name, fraction, pairings = best_match
        debug(
            f"[schema.resolve_table_renames] {old_name!r} -> {new_name!r} "
            f"(matched_fraction={fraction:.2f}, pairings={pairings!r})"
        )
        new_table_renames.append((old_name, new_name))
        remaining_dropped.remove(old_name)
        remaining_added.remove(new_name)

        col_renames = tuple(sorted((o, n) for o, n in pairings if o != n))
        if col_renames:
            existing = per_table.get(new_name, TableDiff())
            per_table[new_name] = TableDiff(
                added_columns=existing.added_columns,
                dropped_columns=existing.dropped_columns,
                redeclared_columns=existing.redeclared_columns,
                retyped_columns=existing.retyped_columns,
                value_type_changed_columns=existing.value_type_changed_columns,
                renamed_columns=col_renames,
                fk_changed=existing.fk_changed,
                pk_changed=existing.pk_changed,
            )

    return SchemaDiff(
        added_tables=tuple(sorted(remaining_added)),
        dropped_tables=tuple(sorted(remaining_dropped)),
        table_renames=tuple(sorted(new_table_renames)),
        per_table=per_table,
    )


def _merge_fk_layers(
    cached_fks: list[FKEdge],
    fresh_catalog_fks: list[FKEdge],
    *,
    surviving_columns: dict[str, set[str]],
    src_table: str,
) -> tuple[list[FKEdge], list[FKEdge], list[FKEdge], list[FKEdge]]:
    """
    Combine fresh catalog FKs with cached non-catalog (inferred + user override) FKs.

    The fresh catalog snapshot wins for catalog-declared edges (``inference_tag is None``); cached non-catalog edges are kept as long as both endpoints still exist in the new graph (``surviving_columns`` maps table name to its surviving column-name set). The function returns ``(merged, dropped_inferred, dropped_user, dropped_catalog)`` where each *dropped* list holds the original :class:`FKEdge` objects so the caller can record provenance in a migration report.
    """

    new_catalog = [copy.deepcopy(e) for e in fresh_catalog_fks if e.inference_tag is None]
    new_keys = {_edge_key(e) for e in new_catalog}
    cached_catalog_keys = {_edge_key(e): e for e in cached_fks if e.inference_tag is None}

    merged: list[FKEdge] = list(new_catalog)
    dropped_inferred: list[FKEdge] = []
    dropped_user: list[FKEdge] = []
    dropped_catalog: list[FKEdge] = [edge for key, edge in cached_catalog_keys.items() if key not in new_keys]

    for edge in cached_fks:
        tag = edge.inference_tag
        if tag is None:
            continue
        key = _edge_key(edge)
        if key in new_keys:
            continue
        dst_cols_ok = edge.dst_table in surviving_columns and all(
            c in surviving_columns[edge.dst_table] for c in edge.dst_cols
        )
        src_cols_ok = edge.src_table in surviving_columns and all(
            c in surviving_columns[edge.src_table] for c in edge.src_cols
        )
        if dst_cols_ok and src_cols_ok:
            merged.append(copy.deepcopy(edge))
            continue
        if isinstance(tag, str) and tag.startswith("user_override_"):
            dropped_user.append(edge)
        else:
            dropped_inferred.append(edge)

    return merged, dropped_inferred, dropped_user, dropped_catalog


def apply_diff(
    cached_sg: SchemaGraph,
    new_sg: SchemaGraph,
    diff: SchemaDiff,
    dialect: Dialect,
    notes_content: str | None = None,
    *,
    schema_json_path: str | Path | None = None,
    refresh_existing_descriptions_on_addition: bool = False,
) -> SchemaGraph:
    """
    Mutate *cached_sg* in place to reflect *diff*, profiling only the affected tables.

    Profiling/classification is run only on tables that appear in :meth:`SchemaDiff.changed_table_names`; everything else keeps its cached profile and LLM-assigned roles. When *refresh_existing_descriptions_on_addition* is true and the diff adds tables, a follow-up full-graph classifier pass may refresh descriptions and roles on tables that were otherwise unchanged. Caller is responsible for updating hashes and persisting.
    """

    for t in diff.dropped_tables:
        cached_sg.tables.pop(t, None)

    rename_map = dict(diff.table_renames)
    for old, new in diff.table_renames:
        if old not in cached_sg.tables:
            continue
        tbl = cached_sg.tables.pop(old)
        tbl.name = new
        cached_sg.tables[new] = tbl
    if rename_map:
        for tbl in cached_sg.tables.values():
            for fk in tbl.foreign_keys:
                if fk.dst_table in rename_map:
                    fk.dst_table = rename_map[fk.dst_table]
                if fk.src_table in rename_map:
                    fk.src_table = rename_map[fk.src_table]

    column_rename_lookup: dict[str, dict[str, str]] = {}
    for tname, td in diff.per_table.items():
        if td.renamed_columns:
            column_rename_lookup[tname] = {old: new for old, new in td.renamed_columns}
    if column_rename_lookup:
        for tbl in cached_sg.tables.values():
            for fk in tbl.foreign_keys:
                if fk.inference_tag is None:
                    continue
                src_map = column_rename_lookup.get(fk.src_table) or {}
                dst_map = column_rename_lookup.get(fk.dst_table) or {}
                if not src_map and not dst_map:
                    continue
                old_src_cols = list(fk.src_cols)
                old_dst_cols = list(fk.dst_cols)
                fk.src_cols = [src_map.get(c, c) for c in fk.src_cols]
                fk.dst_cols = [dst_map.get(c, c) for c in fk.dst_cols]
                tag = fk.inference_tag
                if (
                    isinstance(tag, str)
                    and tag.startswith("user_override_")
                    and (old_src_cols != fk.src_cols or old_dst_cols != fk.dst_cols)
                ):
                    diff.ported_user_fks.append(
                        (
                            fk.src_table,
                            ",".join(old_src_cols),
                            ",".join(fk.src_cols),
                            fk.dst_table,
                            ",".join(old_dst_cols),
                            ",".join(fk.dst_cols),
                        )
                    )

    for tname, td in diff.per_table.items():
        if tname not in cached_sg.tables or tname not in new_sg.tables:
            continue
        cached_t = cached_sg.tables[tname]
        new_t = new_sg.tables[tname]

        for old_col, new_col in td.renamed_columns:
            if old_col not in cached_t.columns:
                continue
            col = cached_t.columns.pop(old_col)
            col.name = new_col
            if new_col in new_t.columns:
                nt_col = new_t.columns[new_col]
                if col.data_type != nt_col.data_type:
                    col.data_type = nt_col.data_type
                    col.value_type = data_type_to_value_type(nt_col.data_type)
                col.is_nullable = nt_col.is_nullable
                col.is_unique = nt_col.is_unique
            cached_t.columns[new_col] = col
        for c in td.dropped_columns:
            cached_t.columns.pop(c, None)
        for c in td.added_columns:
            if c in new_t.columns:
                cached_t.columns[c] = copy.deepcopy(new_t.columns[c])
        for col_name, _old_dt, new_dt in td.redeclared_columns:
            if col_name not in cached_t.columns or col_name not in new_t.columns:
                continue
            cur = cached_t.columns[col_name]
            nt_col = new_t.columns[col_name]
            cur.data_type = new_dt
            cur.value_type = data_type_to_value_type(new_dt)
            cur.is_nullable = nt_col.is_nullable
            cur.is_unique = nt_col.is_unique
        for col_name, _old_dt, new_dt in td.retyped_columns:
            if col_name not in cached_t.columns:
                continue
            cur = cached_t.columns[col_name]
            data_type_changed = cur.data_type != new_dt
            cur.data_type = new_dt

            new_value_type = data_type_to_value_type(new_dt)
            if data_type_changed or new_value_type != cur.value_type:
                cur.top_k_values = []
                cur.min_val = None
                cur.max_val = None
                cur.distinct_count = 0
                cur.distinct_ratio = 0.0
                cur.null_ratio = 0.0
                cur.row_count = 0
                cur.mode_frequency_ratio = 0.0
                cur.semantic_distinct_values = []
                cur.semantic_join_neighbors = []
                cur.valid_filter_ops = []
                cur.valid_aggregations = []
                cur.valid_having_ops = []
                cur.distinct_from_sample = False
            cur.value_type = new_value_type
            if col_name in new_t.columns:
                cur.is_nullable = new_t.columns[col_name].is_nullable
                cur.is_unique = new_t.columns[col_name].is_unique
        if td.fk_changed:
            surviving = {tname: set(cached_sg.tables[tname].columns.keys()) for tname in cached_sg.tables}
            merged, dropped_inferred, dropped_user, dropped_catalog = _merge_fk_layers(
                cached_t.foreign_keys,
                new_t.foreign_keys,
                surviving_columns=surviving,
                src_table=tname,
            )
            cached_t.foreign_keys = merged
            if dropped_inferred:
                debug(
                    f"[schema.apply_diff] {tname}: dropped {len(dropped_inferred)} inferred FK(s) "
                    f"with missing endpoints: {dropped_inferred[:3]!r}"
                )
            if dropped_user:
                debug(
                    f"[schema.apply_diff] {tname}: dropped {len(dropped_user)} user-override FK(s) "
                    f"with missing endpoints: {dropped_user[:3]!r}"
                )
                for edge in dropped_user:
                    diff.dropped_user_fks.append(
                        (
                            edge.src_table,
                            ",".join(edge.src_cols),
                            edge.dst_table,
                            ",".join(edge.dst_cols),
                            (str(edge.inference_tag) if edge.inference_tag is not None else ""),
                        )
                    )
            for edge in dropped_catalog:
                diff.dropped_catalog_fks.append(
                    (
                        edge.src_table,
                        ",".join(edge.src_cols),
                        edge.dst_table,
                        ",".join(edge.dst_cols),
                    )
                )
            cached_t.primary_key = list(new_t.primary_key)
        if td.pk_changed:
            new_pk_set = set(new_t.primary_key)
            old_pk_set = set(cached_t.primary_key)
            for ex_pk in old_pk_set - new_pk_set:
                ex_col = cached_t.columns.get(ex_pk)
                if ex_col is not None:
                    ex_col.pk_inference_tag = None
            for new_pk in new_pk_set - old_pk_set:
                new_col = cached_t.columns.get(new_pk)
                if new_col is not None:
                    new_col.pk_inference_tag = None
            cached_t.primary_key = list(new_t.primary_key)

    for tname in diff.added_tables:
        if tname in new_sg.tables:
            cached_sg.tables[tname] = copy.deepcopy(new_sg.tables[tname])

    surviving_columns_global = {tn: set(cached_sg.tables[tn].columns.keys()) for tn in cached_sg.tables}

    table_renames_map = {old: new for old, new in diff.table_renames}
    column_renames_by_table: dict[str, dict[str, str]] = {}
    for tname, td in diff.per_table.items():
        col_map = {old: new for old, new in td.renamed_columns}
        if col_map:
            column_renames_by_table[tname] = col_map
    for _tname, tbl in cached_sg.tables.items():
        cleaned: list[tuple[str, str, str, str]] = []
        for quad in list(getattr(tbl, "_user_semantic_neighbors", []) or []):
            if not isinstance(quad, tuple) or len(quad) != 4:
                continue
            src_t, src_c, dst_t, dst_c = quad
            new_src_t = table_renames_map.get(src_t, src_t)
            new_dst_t = table_renames_map.get(dst_t, dst_t)
            new_src_c = (column_renames_by_table.get(new_src_t) or {}).get(src_c, src_c)
            new_dst_c = (column_renames_by_table.get(new_dst_t) or {}).get(dst_c, dst_c)
            if new_src_t not in surviving_columns_global or new_dst_t not in surviving_columns_global:
                continue
            if new_src_c not in surviving_columns_global[new_src_t]:
                continue
            if new_dst_c not in surviving_columns_global[new_dst_t]:
                continue
            cleaned.append((new_src_t, new_src_c, new_dst_t, new_dst_c))
        tbl._user_semantic_neighbors = cleaned

    for _tname, tbl in cached_sg.tables.items():
        kept: list[FKEdge] = []
        for edge in tbl.foreign_keys:
            if edge.inference_tag is None:
                kept.append(edge)
                continue
            dst_ok = edge.dst_table in surviving_columns_global and all(
                c in surviving_columns_global[edge.dst_table] for c in edge.dst_cols
            )
            src_ok = edge.src_table in surviving_columns_global and all(
                c in surviving_columns_global[edge.src_table] for c in edge.src_cols
            )
            if dst_ok and src_ok:
                kept.append(edge)
            else:
                debug(
                    f"[schema.apply_diff] post-sweep dropped {edge.inference_tag} FK "
                    f"{edge.src_table}.{','.join(edge.src_cols)}->{edge.dst_table}.{','.join(edge.dst_cols)} "
                    "(endpoint no longer exists)"
                )
        tbl.foreign_keys = kept

    _profile_subset(dialect, cached_sg, diff.changed_table_names(), notes_content)

    if refresh_existing_descriptions_on_addition and diff.added_tables:
        _refresh_existing_descriptions_after_addition(cached_sg, diff, notes_content)

    pk_blocked, fk_blocked = _load_inference_block_lists(schema_json_path)
    _infer_missing_pks_from_profile(cached_sg.tables, dialect=dialect, blocked=pk_blocked)
    _pair_targeted_fk_inference(cached_sg, blocked=fk_blocked)
    _redact_hidden_sensitivity_profile_values(cached_sg)
    compute_semantic_profile_join_neighbors(cached_sg)
    pc = _promote_cross_component_semantic_edges(cached_sg)
    if pc:
        debug(f"[schema.apply_diff] promoted {pc} cross-component semantic edges to FKs")
    ps = _promote_same_component_semantic_edges(cached_sg)
    if ps:
        debug(f"[schema.apply_diff] promoted {ps} same-component semantic edges to FKs")
    _mark_canonical_duplicates(cached_sg)

    _coerce_pk_fk_columns_to_identifier(cached_sg)
    cached_sg.join_paths_multi = _recompute_join_paths_multi(cached_sg.tables)
    cached_sg.refresh_schema_stats()

    return cached_sg


def _raise_if_schema_unusable(sg: SchemaGraph, schema_context: SchemaContext) -> None:
    """Raise :class:`SchemaAccessError` when the graph fails pipeline invariants."""

    reasons: list[str] = []
    offending: list[str] = []
    if not sg.tables:
        reasons.append("no relations reflected after applying include mode, allow_objects, and deny lists")
    for name, tbl in sg.tables.items():
        if not tbl.columns:
            reasons.append(
                f"table {name} reflected with zero columns; likely a catalog misconfiguration",
            )
            offending.append(name)
        if schema_context.include == "tables" and tbl.kind != "table":
            reasons.append(f"internal invariant: non-table relation in tables-only scope: {name}")
            offending.append(name)
        if schema_context.include == "views" and tbl.kind != "view":
            reasons.append(f"internal invariant: non-view relation in views-only scope: {name}")
            offending.append(name)
    seen_lower: dict[str, str] = {}
    for n in sorted(sg.tables.keys()):
        low = n.lower()
        if low in seen_lower:
            reasons.append(f"ambiguous relation names after deny: {seen_lower[low]!r}, {n!r}")
            offending.extend([seen_lower[low], n])
        else:
            seen_lower[low] = n
    fk_ct = sum(len(x.foreign_keys) for x in sg.tables.values())
    sem_ct = _semantic_neighbor_edge_count(sg)
    if len(sg.tables) > 1:
        if schema_context.include in ("tables", "both"):
            if fk_ct + sem_ct == 0:
                reasons.append(
                    "graph has multiple relations but no FK edges and no semantic join neighbors; "
                    "multi-table questions cannot be answered",
                )
        elif sem_ct == 0:
            reasons.append(
                "graph has multiple views but no semantic join neighbors; multi-view questions cannot be routed",
            )
    if reasons:
        raise SchemaAccessError("; ".join(reasons))


def _fk_edge_stable_dict(edge: FKEdge) -> dict[str, Any]:
    """Serialize an FK edge with stable key order for hashing."""

    return {
        "dst_cols": list(edge.dst_cols),
        "dst_table": edge.dst_table,
        "inference_tag": edge.inference_tag,
        "src_cols": list(edge.src_cols),
        "src_table": edge.src_table,
    }


def _column_structural_dict(col: ColumnMetadata) -> dict[str, Any]:
    """DDL-stable subset of column metadata for structural hashing."""

    fk = [col.fk_target[0], col.fk_target[1]] if col.fk_target else None
    return {
        "data_type": col.data_type,
        "fk_target": fk,
        "is_foreign_key": col.is_foreign_key,
        "is_nullable": col.is_nullable,
        "is_primary_key": col.is_primary_key,
        "is_unique": col.is_unique,
        "name": col.name,
    }


def _column_profiling_dict(col: ColumnMetadata) -> dict[str, Any]:
    """Profiling-only column payload for profiling hashing."""

    return {
        "description": col.description,
        "distinct_count": col.distinct_count,
        "distinct_from_sample": col.distinct_from_sample,
        "distinct_ratio": col.distinct_ratio,
        "element_type": col.element_type,
        "is_aggregatable_override": col.is_aggregatable_override,
        "is_filterable_override": col.is_filterable_override,
        "is_groupable_override": col.is_groupable_override,
        "is_selectable": col.is_selectable,
        "max_val": col.max_val,
        "min_val": col.min_val,
        "null_ratio": col.null_ratio,
        "role": col.role,
        "row_count": col.row_count,
        "semantic_distinct_values": col.semantic_distinct_values,
        "semantic_join_neighbors": [list(p) for p in col.semantic_join_neighbors],
        "sensitivity": col.sensitivity,
        "top_k_values": collect_profiling_topk_values(col.top_k_values),
        "valid_aggregations": col.valid_aggregations,
        "valid_filter_ops": col.valid_filter_ops,
        "valid_having_ops": col.valid_having_ops,
        "value_type": col.value_type,
    }


def _table_structural_dict(table: TableMetadata) -> dict[str, Any]:
    """DDL-stable subset of table metadata for structural hashing."""

    cols = {k: _column_structural_dict(table.columns[k]) for k in sorted(table.columns)}
    fkeys = sorted(
        (_fk_edge_stable_dict(e) for e in table.foreign_keys),
        key=lambda d: (
            d["src_table"],
            tuple(d["src_cols"]),
            d["dst_table"],
            tuple(d["dst_cols"]),
        ),
    )
    return {
        "columns": cols,
        "foreign_keys": fkeys,
        "kind": table.kind,
        "primary_key": list(table.primary_key),
    }


def _table_profiling_dict(table: TableMetadata) -> dict[str, Any]:
    """Profiling-only table payload including nested column profiles."""

    cdr = {f"{a}|{b}": v for (a, b), v in sorted(table.composite_descriptive_ratios.items())}
    cols = {k: _column_profiling_dict(table.columns[k]) for k in sorted(table.columns)}
    return {
        "columns": cols,
        "composite_descriptive_ratios": cdr,
        "description": table.description,
        "role": table.role,
        "row_count": table.row_count,
    }


def tables_structural_payload(tables: dict[str, TableMetadata]) -> dict[str, Any]:
    """Build sorted structural table dict for :func:`structural_hash_fp`."""

    return {name: _table_structural_dict(tables[name]) for name in sorted(tables)}


def tables_profiling_payload(tables: dict[str, TableMetadata]) -> dict[str, Any]:
    """Build sorted profiling table dict for :func:`profiling_hash_fp`."""

    return {name: _table_profiling_dict(tables[name]) for name in sorted(tables)}


def assign_schema_graph_hashes(sg: SchemaGraph, schema_context: SchemaContext, notes_sha256: str) -> None:
    """Compute structural, profiling, scope, effective, notes, and semantic-edge hashes on *sg* in place."""

    st = structural_hash_fp(tables_structural_payload(sg.tables))
    pr = profiling_hash_fp(tables_profiling_payload(sg.tables))
    sc = scope_hash_fp(schema_context)
    ef = effective_structural_hash_fp(st, sc)
    sg.structural_hash = st
    sg.profiling_hash = pr
    sg.scope_hash = sc
    sg.effective_structural_hash = ef
    sg.include = schema_context.include
    sg.notes_hash = notes_sha256
    sg.semantic_edges_hash = _semantic_edges_fingerprint(sg.tables)
    sg.scope_descriptor = scope_descriptor_for(schema_context)


def scope_descriptor_for(ctx: SchemaContext) -> dict[str, Any]:
    """Return a JSON-serialisable descriptor of *ctx*'s scope-relevant fields for cache persistence."""

    return {
        "allow_objects": sorted(ctx.allow_objects),
        "deny_columns": sorted(ctx.deny_columns),
        "allow_columns": sorted(ctx.allow_columns),
        "include": ctx.include,
    }


def schema_context_from_descriptor(desc: dict[str, Any]) -> SchemaContext:
    """Reconstruct a :class:`SchemaContext` from a cached scope descriptor (4 scope fields)."""

    inc_raw = desc.get("include", "tables")
    if inc_raw not in ("tables", "views", "both"):
        inc_raw = "tables"
    return SchemaContext(
        allow_objects=frozenset(str(x) for x in (desc.get("allow_objects") or [])),
        deny_columns=frozenset(str(x) for x in (desc.get("deny_columns") or [])),
        allow_columns=frozenset(str(x) for x in (desc.get("allow_columns") or [])),
        include=inc_raw,
    )


def _reverse_fk_path(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reverse a FK path by flipping each edge's direction and reversing list order.

    Args:

        path: Description.

    Returns:

        New list of edge dicts with src/dst swapped and order reversed.
    """
    reversed_path = []
    for e in reversed(path):
        flipped = {
            "src_table": e["dst_table"],
            "src_cols": e["dst_cols"],
            "dst_table": e["src_table"],
            "dst_cols": e["src_cols"],
        }
        if "inference_tag" in e:
            flipped["inference_tag"] = e["inference_tag"]
        reversed_path.append(flipped)
    return reversed_path


def _analyze_fk_path_topology(path: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """
    Analyze an FK path to determine its topology type, anchor table, and leaf tables.

    Args:

        path: List of FK edge dicts representing a join path.

    Returns:

        `'tree'`.
    """
    if not path:
        return ("none", "", [])
    table_counts: dict[str, int] = {}
    for e in path:
        src = e["src_table"]
        dst = e["dst_table"]
        table_counts[src] = table_counts.get(src, 0) + 1
        table_counts[dst] = table_counts.get(dst, 0) + 1
    if not table_counts:
        return ("none", "", [])
    leaves = sorted([t for t, c in table_counts.items() if c == 1])
    hubs = sorted(
        [t for t, c in table_counts.items() if c > 1],
        key=lambda t: (-table_counts[t], t),
    )
    if len(leaves) == 2 and len(hubs) == len(table_counts) - 2:
        return ("linear", min(leaves), leaves)
    if len(hubs) == 1:
        return ("star", hubs[0], leaves)
    if hubs:
        return ("tree", hubs[0], leaves)
    return ("linear", min(table_counts.keys()), list(table_counts.keys()))


def _compute_join_paths_multi_from_adj(
    adj: dict[str, list[FKEdge]],
    tlist: list[str],
) -> dict[str, dict[str, list[list[dict[str, Any]]]]]:
    """
    All shortest FK-edge paths per ordered table pair, capped per pair for storage size.

    Args:

        adj: Undirected FK adjacency lists keyed by table name.

        tlist: Sorted table names defining iteration order.

    Returns:

        ``join_paths_multi[source][target]`` lists of normalized edge dict paths (empty when unreachable).
    """

    cap = max(1, int(PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP))
    join_paths_multi: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}
    for s in tlist:
        row: dict[str, list[list[dict[str, Any]]]] = {s: [[]]}
        dist: dict[str, int] = {s: 0}
        preds: dict[str, list[tuple[str, FKEdge]]] = {s: []}
        frontier = [s]
        while frontier:
            next_frontier: list[str] = []
            in_next: set[str] = set()
            for cur in frontier:
                d_cur = dist[cur]
                for e in adj[cur]:
                    nxt = e.dst_table
                    nd = d_cur + 1
                    if nxt not in dist:
                        dist[nxt] = nd
                        preds[nxt] = [(cur, e)]
                        if nxt not in in_next:
                            in_next.add(nxt)
                            next_frontier.append(nxt)
                    elif dist[nxt] == nd:
                        preds[nxt].append((cur, e))
                        if nxt not in in_next:
                            in_next.add(nxt)
                            next_frontier.append(nxt)
            frontier = next_frontier
            if not frontier:
                break

        for t in tlist:
            if t == s:
                continue
            if t not in dist:
                row[t] = []
                continue
            paths_edges: list[list[FKEdge]] = []

            def collect_paths(
                node: str,
                stack: list[FKEdge],
                _paths_edges: list[list[FKEdge]] = paths_edges,
                _preds: dict[str, list[tuple[str, FKEdge]]] = preds,
                _target: str = s,
            ) -> None:
                """Depth-first FK-path enumeration bounded by ``cap`` for one source/target pair."""
                if len(_paths_edges) >= cap:
                    return
                if node == _target:
                    _paths_edges.append(list(reversed(stack)))
                    return
                for pr, ed in _preds.get(node, ()):
                    collect_paths(pr, stack + [ed])

            collect_paths(t, [])
            seen_keys: set[str] = set()
            out_paths: list[list[dict[str, Any]]] = []
            for pedges in paths_edges:
                sp = [asdict(ed) for ed in pedges]
                norm = _normalize_fk_path(sp)
                key = stable_json(norm)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out_paths.append(norm)
            row[t] = out_paths
        join_paths_multi[s] = row
    return join_paths_multi


def _normalize_fk_path(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize an FK join path to a canonical form based on its topology.

    Args:

        path: List of FK edge dicts to normalize.

    Returns:

        Reordered and/or flipped list of edge dicts in canonical form.
    """
    if not path:
        return path
    topology_type, anchor, leaves = _analyze_fk_path_topology(path)
    if topology_type == "none":
        return path
    if topology_type == "linear":
        start_table = path[0]["src_table"]
        if start_table == anchor:
            return path
        return _reverse_fk_path(path)
    edge_map: dict[str, list[dict[str, Any]]] = {}
    for e in path:
        src = e["src_table"]
        dst = e["dst_table"]
        if src == anchor:
            edge_map.setdefault(dst, []).append(e)
        elif dst == anchor:
            flipped = {
                "src_table": dst,
                "src_cols": e["dst_cols"],
                "dst_table": src,
                "dst_cols": e["src_cols"],
            }
            if "inference_tag" in e:
                flipped["inference_tag"] = e["inference_tag"]
            edge_map.setdefault(src, []).append(flipped)
        else:
            for branch_key in sorted(edge_map.keys()):
                branch_tables = set()
                for be in edge_map[branch_key]:
                    branch_tables.add(be["src_table"])
                    branch_tables.add(be["dst_table"])
                if src in branch_tables or dst in branch_tables:
                    edge_map[branch_key].append(e)
                    break
    normalized = []
    for branch_key in sorted(edge_map.keys()):
        normalized.extend(edge_map[branch_key])
    return normalized if normalized else path


def _fk_tables_lower_index(tables: dict[str, TableMetadata]) -> dict[str, str]:
    """Map lowercased table name to the first canonical table key with that spelling."""
    index: dict[str, str] = {}
    for name in tables:
        lo = name.lower()
        index.setdefault(lo, name)
    return index


def _fk_match_suffix_stem(col_lower: str) -> str | None:
    """Return the first configured stem that *col_lower* ends with, or None."""
    for stem in FK_INFERENCE_SUFFIX_STEMS:
        if col_lower.endswith(stem):
            return stem
    return None


def _fk_candidate_prefixes(col_lower: str, stem: str) -> list[str]:
    """
    Return plural-tolerant prefix candidates derived from stripping *stem* from *col_lower*.

    The output is the deduplicated, length-descending list of plausible target-table-name prefixes derived from a foreign-key-shaped column name. Includes the full prefix, every right-anchored sub-segment when the prefix is snake_case, and singular/plural variants of each. Returns an empty list when stripping the stem leaves no usable prefix.

    Args:

        col_lower: Lower-cased column name.

        stem: Suffix already matched by :func:`_fk_match_suffix_stem`.

    Returns:

        Deduplicated, length-descending prefix candidate list.
    """

    if not col_lower or not stem:
        return []
    if not col_lower.endswith(stem):
        return []
    prefix_full = col_lower[: -len(stem)]
    if not prefix_full:
        return []
    candidate_prefixes: list[str] = [prefix_full]
    if "_" in prefix_full:
        parts = prefix_full.split("_")
        candidate_prefixes.append(parts[-1])
        for i in range(len(parts) - 1, 0, -1):
            candidate_prefixes.append("_".join(parts[i:]))
    expanded: list[str] = []
    for p in candidate_prefixes:
        if not p:
            continue
        expanded.append(p)
        if p.endswith("s") and len(p) > 1:
            expanded.append(p[:-1])
        else:
            expanded.append(p + "s")
    seen: set[str] = set()
    ordered: list[str] = []
    for p in sorted(expanded, key=len, reverse=True):
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    return ordered


def _fk_name_shape_matches_table(col_lower: str, dst_table_lower: str) -> bool:
    """
    Return True when *col_lower* has a recognised FK-style suffix and one of its prefix candidates equals *dst_table_lower*.

    Used by both suffix FK inference (layer 2) and the semantic→FK promoter (layer 5) so the two layers agree on what a "FK-shaped" column name pointing at a given table looks like.
    """

    if not col_lower or not dst_table_lower:
        return False
    stem = _fk_match_suffix_stem(col_lower)
    if not stem:
        return False
    return dst_table_lower in _fk_candidate_prefixes(col_lower, stem)


def _fk_infer_value_types_compatible(src: ColumnMetadata, dst_col: ColumnMetadata | None) -> bool:
    """Return True when profiling types are absent or compatible for inferred FK endpoints."""
    if dst_col is None:
        return True
    st = (src.value_type or "").strip()
    dt = (dst_col.value_type or "").strip()
    if not st or not dt:
        return True
    if st == dt:
        return True
    if (st, dt) in COMPATIBLE_TYPE_PAIRS or (dt, st) in COMPATIBLE_TYPE_PAIRS:
        return True
    if _fk_string_int_compatible(src, dst_col):
        return True
    return False


def _fk_string_int_compatible(a: ColumnMetadata, b: ColumnMetadata) -> bool:
    """
    Allow string↔integer FK candidates when the string-side samples are all digit strings.

    Symmetric: accepts ``a`` string / ``b`` integer or vice versa. When the string side has no samples it cannot be judged digit-only and the helper returns False; the conservative answer keeps spurious string→int FKs from being promoted on naming alone. Coercion only widens the inference compatibility check; downstream value-type semantics are unchanged.
    """

    at = (a.value_type or "").strip().lower()
    bt = (b.value_type or "").strip().lower()
    if at in STRING_VALUE_TYPES and bt in INTEGER_VALUE_TYPES:
        string_side = a
    elif bt in STRING_VALUE_TYPES and at in INTEGER_VALUE_TYPES:
        string_side = b
    else:
        return False
    samples = list(string_side.top_k_values or [])
    if not samples:
        return False
    return all(str(v).strip().lstrip("-").isdigit() for v in samples if v is not None and str(v).strip() != "")


def _fk_overlap_validates(src: ColumnMetadata, dst: ColumnMetadata) -> bool:
    """
    Return True when sampled values overlap enough to support an inferred FK.

    Compares ``top_k_values`` from both sides after normalizing via ``str()`` and stripping. When either side has fewer than ``PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE`` non-empty samples the helper returns True (insufficient evidence to reject — fall back to the naming-only signal). Otherwise the overlap ratio is computed against the smaller sample set, and the candidate is accepted when the ratio is at least ``PolicyConfig.FK_INFER_OVERLAP_MIN_RATIO``. The helper is symmetric and treats integer / digit-string pairs as equal after string normalization so it cooperates with ``_fk_string_int_compatible``.
    """

    def _norm_set(col: ColumnMetadata) -> set[str]:
        out: set[str] = set()
        for v in col.top_k_values or []:
            if v is None:
                continue
            s = str(v).strip()
            if s == "":
                continue
            out.add(s)
        return out

    a_set = _norm_set(src)
    b_set = _norm_set(dst)
    min_sample = int(PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE)
    if len(a_set) < min_sample or len(b_set) < min_sample:
        return True
    overlap = len(a_set & b_set)
    smaller = min(len(a_set), len(b_set))
    ratio = overlap / smaller if smaller else 0.0
    return ratio >= float(PolicyConfig.FK_INFER_OVERLAP_MIN_RATIO)


_INFERRED_PK_NAME_SUFFIXES: tuple[str, ...] = tuple(s for s in FK_INFERENCE_SUFFIX_STEMS if s in PK_STYLE_FK_STEMS)


def _infer_missing_pks_from_profile(
    tables: dict[str, TableMetadata],
    *,
    blocked: frozenset[tuple[str, str]] = frozenset(),
    dialect: Any | None = None,
) -> list[tuple[str, str]]:
    """
    Infer single-column primary keys from constraint signals and profiling statistics.

    A column qualifies when its containing table has no declared primary key, the column is non-nullable (or has a zero null ratio), the column either carries a single-column ``UNIQUE`` constraint (``is_unique=True``) or its profiled distinct count equals the table row count, and the value type is integer, number, or string. When the qualifying signal is statistical (distinct == row_count) the table row count must also meet ``PolicyConfig.INFERRED_PK_MIN_ROW_COUNT``; the row-count floor is bypassed for ``UNIQUE``-constrained columns because the database has already guaranteed uniqueness. When ``ColumnMetadata.distinct_from_sample`` is true (large-table sampling), statistical uniqueness is confirmed via :meth:`Dialect.refresh_full_table_distinct_for_pk_inference` before inference; without a dialect or when that refresh fails, sampled statistics are not treated as globally unique. When more than one candidate qualifies the helper prefers ``id``, then ``<table>_id``, then names ending in ``_id``, ``_key``, ``_uuid``, or ``_pk``, then the first remaining candidate by sorted name. Inferred primary keys are written to ``ColumnMetadata.is_primary_key`` and ``TableMetadata.primary_key`` and stamped with ``ColumnMetadata.pk_inference_tag = "profile"`` so downstream code (cache merge, override export) can distinguish them from catalog-declared keys. Pairs in ``blocked`` are skipped as candidates so user-rejected inferences do not reappear on a rebuild.

    Args:

        tables: Table metadata keyed by reflected table name.

        blocked: ``(table, column)`` pairs the user has marked as suppressed via overrides.

        dialect: Active dialect for full-table distinct checks after sampled profiling; omit only in tests.

    Returns:

        Sorted list of ``(table, column)`` pairs marked as primary keys by this pass.
    """
    inferred: list[tuple[str, str]] = []
    if not tables:
        return inferred
    min_rows = int(PolicyConfig.INFERRED_PK_MIN_ROW_COUNT)
    for table_name, table in tables.items():
        if table.primary_key:
            continue
        candidates: list[str] = []
        for col_name, col in table.columns.items():
            rc = int(table.row_count or 0)
            if (table_name, col_name) in blocked:
                continue
            if col.is_nullable and (col.null_ratio is None or col.null_ratio > 0.0):
                continue
            vt = (col.value_type or "").strip().lower()
            if vt and vt not in INFERRED_PK_VALUE_TYPES:
                continue
            unique_constraint = bool(col.is_unique)
            if unique_constraint:
                statistical_unique = False
            else:
                if col.distinct_from_sample:
                    if dialect is None:
                        statistical_unique = False
                    else:
                        ft = dialect.refresh_full_table_distinct_for_pk_inference(
                            table_name,
                            col_name,
                            table_kind=table.kind,
                        )
                        if ft is None:
                            statistical_unique = False
                        else:
                            dist_ft, cnt_ft, nr_ft = ft
                            col.distinct_count = dist_ft
                            col.null_ratio = nr_ft
                            col.distinct_from_sample = False
                            if cnt_ft > 0:
                                table.row_count = cnt_ft
                            rc = int(table.row_count or 0)
                            statistical_unique = rc > 0 and dist_ft == rc and rc >= min_rows
                else:
                    statistical_unique = (
                        col.distinct_count is not None and rc > 0 and col.distinct_count == rc and rc >= min_rows
                    )
            if not (unique_constraint or statistical_unique):
                continue
            candidates.append(col_name)
        if not candidates:
            continue
        chosen = _select_inferred_pk_candidate(table_name, candidates)
        if chosen is None:
            continue
        col_meta = table.columns[chosen]
        col_meta.pk_inference_tag = PkInferenceTag.PROFILE
        if chosen not in table.primary_key:
            table.primary_key.append(chosen)
        debug(f"[schema.infer_missing_pks] inferred PK {table_name}.{chosen}")
        inferred.append((table_name, chosen))
    inferred.sort()
    return inferred


def _select_inferred_pk_candidate(table_name: str, candidates: list[str]) -> str | None:
    """
    Pick a single primary-key candidate from a deterministic name preference list.

    When only one candidate is provided it is returned as-is so user-named primary keys are accepted regardless of naming convention. With multiple candidates the helper prefers ``id``, then ``<table_name>_id``, then names ending in any of ``_id``, ``_key``, ``_uuid``, or ``_pk``, then the lexicographically first remaining name.

    Args:

        table_name: Name of the owning table for ``<table>_id`` matching.

        candidates: Column names that already passed the structural checks.

    Returns:

        The chosen column name or ``None`` when *candidates* is empty.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    sorted_candidates = sorted(candidates)
    table_lower = table_name.lower()
    for name in sorted_candidates:
        if name.lower() == "id":
            return name
    for name in sorted_candidates:
        if name.lower() == f"{table_lower}_id":
            return name
    for suffix in _INFERRED_PK_NAME_SUFFIXES:
        for name in sorted_candidates:
            if name.lower().endswith(suffix):
                return name
    return sorted_candidates[0]


def _apply_inferred_fks_to_graph(sg: SchemaGraph, edges: list[FKEdge]) -> int:
    """
    Append inferred FK edges to *sg* in-place, marking source columns as foreign keys.

    Skips edges whose endpoint columns no longer exist or whose canonical edge key already appears in the source table. Returns the number of edges appended.

    Args:

        sg: Schema graph mutated in place.

        edges: Candidate inferred FK edges produced by ``_infer_missing_fks``.

    Returns:

        Count of newly added edges.
    """
    if not edges:
        return 0
    added = 0
    for e in edges:
        src_tbl = sg.tables.get(e.src_table)
        dst_tbl = sg.tables.get(e.dst_table)
        if src_tbl is None or dst_tbl is None:
            continue
        existing = {_edge_key(x) for x in src_tbl.foreign_keys}
        if _edge_key(e) in existing:
            continue
        if any(c not in src_tbl.columns for c in e.src_cols):
            continue
        if any(c not in dst_tbl.columns for c in e.dst_cols):
            continue
        src_tbl.foreign_keys.append(e)
        added += 1
        debug(f"[schema.apply_inferred_fks] {e.src_table}.{e.src_cols[0]} -> {e.dst_table}.{e.dst_cols[0]}")
    return added


class _FkUnionFind:
    """Disjoint-set union for table names (FK graph connectivity)."""

    __slots__ = ("_parent", "_rank")

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def make_set(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        if x not in self._parent:
            self.make_set(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while x != root:
            px = self._parent[x]
            self._parent[x] = root
            x = px
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            self._parent[ra] = rb
        elif self._rank[ra] > self._rank[rb]:
            self._parent[rb] = ra
        else:
            self._parent[rb] = ra
            self._rank[ra] += 1


def _union_find_with_tag_filter(
    sg: SchemaGraph,
    exclude_tags: frozenset[InferenceTag],
) -> _FkUnionFind:
    """Connect tables using FK edges whose inference_tag is not in *exclude_tags* (catalog edges always union)."""

    uf = _FkUnionFind()
    for t in sg.tables:
        uf.make_set(t)
    for tbl_name, tbl in sg.tables.items():
        for e in tbl.foreign_keys:
            if e.src_table != tbl_name:
                continue
            tag = e.inference_tag
            if tag is not None and tag in exclude_tags:
                continue
            uf.union(e.src_table, e.dst_table)
    return uf


def _union_find_truth_fk_edges(sg: SchemaGraph) -> _FkUnionFind:
    """Tables connected by catalog FKs plus user-declared structural/semantic override FK edges."""

    uf = _FkUnionFind()
    for t in sg.tables:
        uf.make_set(t)
    for tbl_name, tbl in sg.tables.items():
        for e in tbl.foreign_keys:
            if e.src_table != tbl_name:
                continue
            tag = e.inference_tag
            if tag is None:
                uf.union(e.src_table, e.dst_table)
            elif tag in (InferenceTag.USER_STRUCTURAL, InferenceTag.USER_SEMANTIC):
                uf.union(e.src_table, e.dst_table)
    return uf


_UF_EXCLUDE_SEMANTIC_INFERENCE_ONLY: frozenset[InferenceTag] = frozenset({InferenceTag.SEMANTIC})

_INFERRED_COLLAPSE_TAGS: frozenset[InferenceTag] = frozenset(
    {
        InferenceTag.SUFFIX,
        InferenceTag.SELF,
        InferenceTag.COMPOSITE,
        InferenceTag.SEMANTIC,
        InferenceTag.SEMANTIC_PROMOTED,
    }
)


def _pair_targeted_fk_inference(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> int:
    """
    Infer FK candidates between table pairs until fixed point.

    Runs naming-convention inference with ``restrict_tables`` set to each table (self-FKs) and to each pair of tables that lie in different union-find components, merging components as new edges are applied.
    """

    total = 0
    names = sorted(sg.tables.keys())
    while True:
        round_added = 0
        for t in names:
            if t not in sg.tables:
                continue
            edges = _infer_missing_fks(sg.tables, blocked=blocked, restrict_tables=frozenset({t}))
            round_added += _apply_inferred_fks_to_graph(sg, edges)
        uf = _union_find_with_tag_filter(sg, frozenset())
        for i, a in enumerate(names):
            if a not in sg.tables:
                continue
            for b in names[i + 1 :]:
                if b not in sg.tables:
                    continue
                if uf.find(a) == uf.find(b):
                    continue
                edges = _infer_missing_fks(sg.tables, blocked=blocked, restrict_tables=frozenset({a, b}))
                n = _apply_inferred_fks_to_graph(sg, edges)
                round_added += n
                if n:
                    uf = _union_find_with_tag_filter(sg, frozenset())
        total += round_added
        if round_added == 0:
            break
    return total


def _collapse_redundant_inferences(sg: SchemaGraph, skipped: list[OverrideSkip]) -> int:
    """
    Remove inferred FK edges and semantic neighbor pairs made redundant by catalog/user truth FKs.

    Truth connectivity is catalog FKs plus ``user_override_*`` FK edges. Returns a removal count.
    """

    uf_truth = _union_find_truth_fk_edges(sg)
    removed = 0
    for tbl_name, tbl in list(sg.tables.items()):
        kept_edges: list[FKEdge] = []
        for e in tbl.foreign_keys:
            if e.src_table != tbl_name:
                continue
            tag = e.inference_tag
            if tag in _INFERRED_COLLAPSE_TAGS and uf_truth.find(e.src_table) == uf_truth.find(e.dst_table):
                skipped.append(
                    OverrideSkip(
                        path=f"foreign_keys.inferred.{e.src_table}->{e.dst_table}",
                        reason="superseded_by_user_fk",
                    )
                )
                removed += 1
                continue
            kept_edges.append(e)
        tbl.foreign_keys = kept_edges
    for tbl_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            new_neigh: list[tuple[str, str]] = []
            for neigh_tbl, neigh_col in col.semantic_join_neighbors:
                if uf_truth.find(tbl_name) == uf_truth.find(neigh_tbl):
                    skipped.append(
                        OverrideSkip(
                            path=f"tables.{tbl_name}.columns.{col_name}.semantic_join_neighbors",
                            reason="superseded_by_user_fk",
                        )
                    )
                    removed += 1
                    continue
                new_neigh.append((neigh_tbl, neigh_col))
            col.semantic_join_neighbors = new_neigh
    for tbl_name, tbl in sg.tables.items():
        kept_quads: list[tuple[str, str, str, str]] = []
        for quad in list(getattr(tbl, "_user_semantic_neighbors", []) or []):
            if not isinstance(quad, tuple) or len(quad) != 4:
                kept_quads.append(quad)
                continue
            st, _sc, dt, _dc = quad
            if st in sg.tables and dt in sg.tables and uf_truth.find(st) == uf_truth.find(dt):
                skipped.append(
                    OverrideSkip(
                        path=f"tables.{tbl_name}._user_semantic_neighbors",
                        reason="superseded_by_user_fk",
                    )
                )
                removed += 1
                continue
            kept_quads.append(quad)
        tbl._user_semantic_neighbors = kept_quads
    return removed


def _mark_canonical_duplicates(sg: SchemaGraph) -> int:
    """
    Recompute the canonical-bearer index on *sg* for every duplicated column name.

    Args:

        sg: Schema graph whose tables have already been profiled and PK/FK-augmented.

    Returns:

        Number of column entries demoted (i.e., losers among the duplicates).

        For every column name that appears (case-insensitive) in two or more tables, one bearer is chosen using the deterministic ranking ``(primary_key_first, distinct_count_descending, lex_smallest)``. Inferred and declared primary keys are treated identically. The result is stored on ``SchemaGraph._canonical_bearers`` (the single source of truth read by :attr:`ColumnMetadata.is_canonical_duplicate`); singletons are not recorded so they trivially read as canonical.
    """
    by_name: dict[str, list[tuple[str, str, ColumnMetadata]]] = {}
    for table_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            by_name.setdefault(col_name.lower(), []).append((table_name, col_name, col))
    bearers: dict[str, tuple[str, str]] = {}
    demoted = 0
    for key, entries in by_name.items():
        if len(entries) < 2:
            continue
        ordered = sorted(
            entries,
            key=lambda triple: (
                0 if triple[2].is_primary_key else 1,
                -int(triple[2].distinct_count or 0),
                triple[0],
                triple[1],
            ),
        )
        winner_t, winner_c, _ = ordered[0]
        bearers[key] = (winner_t, winner_c)
        demoted += len(ordered) - 1
    object.__setattr__(sg, "_canonical_bearers", bearers)
    return demoted


def _promote_semantic_neighbor_pairs(
    sg: SchemaGraph,
    *,
    cross_component_only: bool,
    inference_tag: InferenceTag,
) -> int:
    """
    Promote semantic profile neighbor pairs to inferred FK edges when gates pass.

    Connectivity for filtering uses all FK edges except pure ``semantic`` inference tags so ``semantic_promoted`` bridges participate once emitted.
    """

    if not sg.tables:
        return 0
    seen_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    promotions: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for tbl_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            for neigh_tbl, neigh_col in col.semantic_join_neighbors:
                a = (tbl_name, col_name)
                b = (neigh_tbl, neigh_col)
                pair = tuple(sorted([a, b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                promotions.append(pair)
    promoted = 0
    uf = _union_find_with_tag_filter(sg, _UF_EXCLUDE_SEMANTIC_INFERENCE_ONLY)
    for left, right in promotions:
        left_tbl = sg.tables.get(left[0])
        right_tbl = sg.tables.get(right[0])
        if left_tbl is None or right_tbl is None:
            continue
        left_col = left_tbl.columns.get(left[1])
        right_col = right_tbl.columns.get(right[1])
        if left_col is None or right_col is None:
            continue
        left_is_pk = left_col.is_primary_key
        right_is_pk = right_col.is_primary_key
        if not left_is_pk and not right_is_pk:
            continue
        if not _fk_infer_value_types_compatible(left_col, right_col):
            continue
        left_vt = (left_col.value_type or "").strip().lower()
        right_vt = (right_col.value_type or "").strip().lower()
        if left_vt != "string" or right_vt != "string":
            debug(
                f"[schema.promote_semantic_to_fk] string-only gate reject "
                f"{left[0]}.{left[1]} ({left_vt}) <-> {right[0]}.{right[1]} ({right_vt})"
            )
            continue
        if not _fk_overlap_validates(left_col, right_col):
            debug(f"[schema.promote_semantic_to_fk] overlap fail {left[0]}.{left[1]} <-> {right[0]}.{right[1]}")
            continue
        if right_is_pk and not left_is_pk:
            src_tbl_name, src_col_name = left
            dst_tbl_name, dst_col_name = right
        elif left_is_pk and not right_is_pk:
            src_tbl_name, src_col_name = right
            dst_tbl_name, dst_col_name = left
        else:
            src_tbl_name, src_col_name = left
            dst_tbl_name, dst_col_name = right
        if src_tbl_name == dst_tbl_name and src_col_name == dst_col_name:
            continue
        bridge_tbl_a, bridge_tbl_b = left[0], right[0]
        connected = uf.find(bridge_tbl_a) == uf.find(bridge_tbl_b)
        if cross_component_only:
            if connected:
                continue
        elif not connected:
            continue
        src_col_meta = sg.tables[src_tbl_name].columns[src_col_name]
        src_distinct = int(src_col_meta.distinct_count or 0)
        if src_distinct < int(PolicyConfig.SEMANTIC_JOIN_MIN_DISTINCT):
            debug(
                f"[schema.promote_semantic_to_fk] distinct floor reject "
                f"{src_tbl_name}.{src_col_name} distinct={src_distinct}"
            )
            continue
        edge = FKEdge(
            src_table=src_tbl_name,
            src_cols=[src_col_name],
            dst_table=dst_tbl_name,
            dst_cols=[dst_col_name],
            inference_tag=inference_tag,
        )
        added = _apply_inferred_fks_to_graph(sg, [edge])
        if added == 0:
            continue
        promoted += added
        uf = _union_find_with_tag_filter(sg, _UF_EXCLUDE_SEMANTIC_INFERENCE_ONLY)
        for tbl_name_c, col_name_c, other in (
            (src_tbl_name, src_col_name, (dst_tbl_name, dst_col_name)),
            (dst_tbl_name, dst_col_name, (src_tbl_name, src_col_name)),
        ):
            tbl_c = sg.tables.get(tbl_name_c)
            if tbl_c is None:
                continue
            col_c = tbl_c.columns.get(col_name_c)
            if col_c is None:
                continue
            col_c.semantic_join_neighbors = [n for n in col_c.semantic_join_neighbors if tuple(n) != other]
        debug(f"[schema.promote_semantic_to_fk] {src_tbl_name}.{src_col_name} -> {dst_tbl_name}.{dst_col_name}")
    return promoted


def _promote_cross_component_semantic_edges(sg: SchemaGraph) -> int:
    """Prefer semantic promotions that bridge structural FK islands."""

    return _promote_semantic_neighbor_pairs(
        sg,
        cross_component_only=True,
        inference_tag=InferenceTag.SEMANTIC_PROMOTED,
    )


def _promote_same_component_semantic_edges(sg: SchemaGraph) -> int:
    """Emit semantic FK shortcuts within an already-connected structural component."""

    return _promote_semantic_neighbor_pairs(
        sg,
        cross_component_only=False,
        inference_tag=InferenceTag.SEMANTIC,
    )


def _promote_semantic_edges_to_fks(sg: SchemaGraph) -> int:
    """Run cross-component then same-component semantic promotions (backward-compatible aggregate)."""

    return _promote_cross_component_semantic_edges(sg) + _promote_same_component_semantic_edges(sg)


def _infer_missing_fks_suffix(
    tables: dict[str, TableMetadata],
    tables_lower: dict[str, str],
) -> list[FKEdge]:
    """Infer FK edges from ``*_id`` / ``*_key`` style names using case-insensitive matching."""
    inferred: list[FKEdge] = []
    for table_name, table in tables.items():
        for col_name, col in table.columns.items():
            if col.is_foreign_key or col.is_primary_key:
                continue
            col_lower = col_name.lower()
            matched_suffix = _fk_match_suffix_stem(col_lower)
            if not matched_suffix:
                continue
            ordered = _fk_candidate_prefixes(col_lower, matched_suffix)
            if not ordered:
                continue
            for pref_lower in ordered:
                dst_table = tables_lower.get(pref_lower)
                if not dst_table:
                    continue
                target = tables[dst_table]
                if len(target.primary_key) != 1:
                    continue
                target_pk = target.primary_key[0]
                pk_lower = target_pk.lower()
                target_ok = False
                for ts in FK_INFERENCE_SUFFIX_STEMS:
                    if pk_lower.endswith(ts):
                        if pk_lower[: -len(ts)] == pref_lower:
                            target_ok = True
                            break
                if target_ok:
                    dst_meta_col = target.columns.get(target_pk)
                    if not _fk_infer_value_types_compatible(col, dst_meta_col):
                        debug(
                            f"[schema.infer_missing_fks] suffix skip type mismatch "
                            f"{table_name}.{col_name} -> {dst_table}.{target_pk}"
                        )
                        continue
                    if dst_meta_col is not None and not _fk_overlap_validates(col, dst_meta_col):
                        debug(
                            f"[schema.infer_missing_fks] suffix skip overlap fail "
                            f"{table_name}.{col_name} -> {dst_table}.{target_pk}"
                        )
                        continue
                    debug(f"[schema.infer_missing_fks] suffix: {table_name}.{col_name} -> {dst_table}.{target_pk}")
                    inferred.append(
                        FKEdge(
                            src_table=table_name,
                            src_cols=[col_name],
                            dst_table=dst_table,
                            dst_cols=[target_pk],
                            inference_tag=(InferenceTag.SELF if dst_table == table_name else InferenceTag.SUFFIX),
                        )
                    )
                    break
    return inferred


def _infer_missing_fks(
    tables: dict[str, TableMetadata],
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
    restrict_tables: frozenset[str] | None = None,
) -> list[FKEdge]:
    """
    Infer missing foreign keys from column naming conventions.

    Uses suffix-based heuristics (case-insensitive ``*_id`` / ``*_key``), including
    self-referential edges when the inferred destination table is the source table.

    Args:

        tables: Table metadata keyed by reflected table name.

        blocked: Canonical edge keys ``(src_table, tuple(src_cols), dst_table, tuple(dst_cols))`` recorded in the overrides sidecar's ``_internal.fk_block_inferred`` envelope. Candidates whose key matches an entry are dropped before being returned so an inference the operator rejected is never re-emitted on the next rebuild.

        restrict_tables: When non-empty, only tables whose names appear in this set participate in inference.

    Returns:

        Inferred edges not declared as foreign keys in the catalog.
    """
    if not tables:
        return []
    if restrict_tables is not None:
        tables = {k: v for k, v in tables.items() if k in restrict_tables}
        if not tables:
            return []
    tables_lower = _fk_tables_lower_index(tables)
    suffix_inferred = _infer_missing_fks_suffix(tables, tables_lower)
    composite_inferred = _infer_missing_fks_composite(tables, tables_lower, existing=suffix_inferred)
    candidates = suffix_inferred + composite_inferred
    if not blocked:
        return candidates
    return [e for e in candidates if (e.src_table, tuple(e.src_cols), e.dst_table, tuple(e.dst_cols)) not in blocked]


def _infer_missing_fks_composite(
    tables: dict[str, TableMetadata],
    tables_lower: dict[str, str],
    *,
    existing: list[FKEdge],
) -> list[FKEdge]:
    """
    Infer composite foreign keys when a source table contains every column of a target's composite PK.

    A candidate composite FK requires:

    - The target table has a primary key with two or more columns. - The source table contains every PK column by exact case-insensitive name and none of those columns are already declared (or just inferred via the suffix pass) as part of a foreign key. - Every per-column pair passes ``_fk_infer_value_types_compatible``; the string ↔ digit-only-string relaxation participates here too. - The source table is not the target table (self-referential composite FKs are rejected).

    Sample-overlap validation is intentionally omitted because per-column ``top_k_values`` are not row-aligned and a faithful tuple overlap would require additional profiling queries.
    """

    if not tables:
        return []
    inferred: list[FKEdge] = []
    existing_src_cols: set[tuple[str, str]] = set()
    for e in existing:
        for c in e.src_cols:
            existing_src_cols.add((e.src_table, c.lower()))
    for src_name, src_tbl in tables.items():
        for _dst_name_lower, dst_real in tables_lower.items():
            if dst_real == src_name:
                continue
            dst_tbl = tables[dst_real]
            pk = dst_tbl.primary_key
            if len(pk) < 2:
                continue
            src_cols_lower = {c.lower(): c for c in src_tbl.columns.keys()}
            mapped: list[tuple[str, str]] = []
            ok = True
            for pk_col in pk:
                pkl = pk_col.lower()
                if pkl not in src_cols_lower:
                    ok = False
                    break
                src_real = src_cols_lower[pkl]
                src_col_meta = src_tbl.columns[src_real]
                if src_col_meta.is_foreign_key:
                    ok = False
                    break
                if (src_name, src_real.lower()) in existing_src_cols:
                    ok = False
                    break
                dst_col_meta = dst_tbl.columns.get(pk_col)
                if not _fk_infer_value_types_compatible(src_col_meta, dst_col_meta):
                    ok = False
                    break
                mapped.append((src_real, pk_col))
            if not ok or not mapped:
                continue
            src_cols = [m[0] for m in mapped]
            dst_cols = [m[1] for m in mapped]
            debug(
                f"[schema.infer_missing_fks] composite: {src_name}.({', '.join(src_cols)}) -> "
                f"{dst_real}.({', '.join(dst_cols)})"
            )
            inferred.append(
                FKEdge(
                    src_table=src_name,
                    src_cols=src_cols,
                    dst_table=dst_real,
                    dst_cols=dst_cols,
                    inference_tag=InferenceTag.COMPOSITE,
                )
            )
    return inferred


def _collect_unique_columns_from_reflected_table(t: Any) -> set[str]:
    """
    Single-source aggregator for single-column uniqueness signals on a reflected SQLAlchemy table.

    Merges hits from both ``UniqueConstraint`` declarations and unique indexes so neither source can independently mark a column unique without going through this helper. Composite (multi-column) constraints/indexes do not imply per-column uniqueness and are skipped.
    """
    unique: set[str] = set()
    for constr in t.constraints:
        if isinstance(constr, UniqueConstraint):
            ucols = [c.name for c in constr.columns]
            if len(ucols) == 1:
                unique.add(ucols[0])
    for idx in t.indexes:
        if not getattr(idx, "unique", False):
            continue
        icols = [c.name for c in idx.columns]
        if len(icols) == 1:
            unique.add(icols[0])
    return unique


def _reflect_schema(
    engine: Any,
    schema_name: str | None = None,
    *,
    object_kind: Literal["table", "view"] = "table",
    allow_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
) -> SchemaGraph:
    """
    Reflect a database schema using SQLAlchemy and build a join-path graph.

    Args:

        engine: SQLAlchemy `Engine` connected to the target database.

        schema_name: Database schema to reflect. Defaults to `EngineConfig.RUNTIME.SCHEMA` or `'public'` if not set.

        object_kind: Reflect base tables and materialized views, or ordinary views only.

    Returns:

        Populated ``SchemaGraph`` without fingerprints (assigned after profiling and scope).
    """
    if schema_name is None:
        schema_name = EngineConfig.RUNTIME.SCHEMA if hasattr(EngineConfig.RUNTIME, "SCHEMA") else "public"

    debug(f"[schema.reflect_schema] reflecting schema '{schema_name}' object_kind={object_kind}")
    insp = inspect(engine)
    md = MetaData(schema=schema_name)
    allow_lower = _allow_objects_lower_set(allow_objects)
    _, fk_blocked = _load_inference_block_lists(schema_json_path)
    if object_kind == "table":
        names: set[str] = set(insp.get_table_names(schema=schema_name))
        gmv = getattr(insp, "get_materialized_view_names", None)
        if callable(gmv):
            try:
                names |= set(gmv(schema=schema_name))
            except Exception:
                pass
        ordered = sorted(names)
        if allow_lower is not None:
            ordered = [n for n in ordered if str(n).lower() in allow_lower]
        if ordered:
            md.reflect(bind=engine, schema=schema_name, only=ordered, views=False)
    else:
        ordered = sorted(insp.get_view_names(schema=schema_name))
        if allow_lower is not None:
            ordered = [n for n in ordered if str(n).lower() in allow_lower]
        if ordered:
            md.reflect(bind=engine, schema=schema_name, only=ordered, views=True)

    row_kind: Literal["table", "view"] = "view" if object_kind == "view" else "table"
    tables: dict[str, TableMetadata] = {}

    for t in md.tables.values():
        columns: dict[str, ColumnMetadata] = {}
        for c in t.columns:
            columns[c.name] = ColumnMetadata(
                name=c.name,
                data_type=str(c.type),
                is_primary_key=c.name in [pk.name for pk in t.primary_key.columns],
                is_foreign_key=False,
                fk_target=None,
                is_nullable=getattr(c, "nullable", True),
            )

        tables[t.name] = TableMetadata(
            name=t.name,
            columns=columns,
            primary_key=[c.name for c in t.primary_key.columns],
            foreign_keys=[],
            kind=row_kind,
        )

    debug(f"[schema.reflect_schema] found {len(tables)} relations")

    for t in md.tables.values():
        unique_cols = _collect_unique_columns_from_reflected_table(t)
        for col_name in unique_cols:
            if col_name in tables[t.name].columns:
                tables[t.name].columns[col_name].is_unique = True

    if object_kind == "table":
        for t in md.tables.values():
            for fk in t.foreign_key_constraints:
                e = FKEdge(
                    src_table=t.name,
                    src_cols=[el.parent.name for el in fk.elements],
                    dst_table=fk.elements[0].column.table.name,
                    dst_cols=[el.column.name for el in fk.elements],
                )
                tables[t.name].foreign_keys.append(e)

                debug(
                    f"[schema.reflect_schema] explicit FK: {e.src_table}.{e.src_cols[0]} -> "
                    f"{e.dst_table}.{e.dst_cols[0]}",
                )

        fk_count = sum(len(tbl.foreign_keys) for tbl in tables.values())
        debug(f"[schema.reflect_schema] found {fk_count} foreign key edges")

        tmp_sg = SchemaGraph(tables=tables, join_paths_multi={})
        inferred_ct = _pair_targeted_fk_inference(tmp_sg, blocked=fk_blocked)
        if inferred_ct:
            debug(
                f"[schema.reflect_schema] pair-targeted inference added {inferred_ct} inferred FK edge(s) "
                "from naming conventions"
            )

    else:
        tmp_sg = SchemaGraph(tables=tables, join_paths_multi={})
        inferred_ct = _pair_targeted_fk_inference(tmp_sg, blocked=fk_blocked)
        if inferred_ct:
            debug(f"[schema.reflect_schema] pair-targeted inference added {inferred_ct} inferred FK edge(s) on views")

    enum_values = _load_pg_enum_values(engine)

    adj: dict[str, list[FKEdge]] = {t: [] for t in tables}
    for tbl in tables.values():
        for e in tbl.foreign_keys:
            if e.src_table not in tables or e.dst_table not in tables:
                continue
            adj[e.src_table].append(e)
            adj[e.dst_table].append(
                FKEdge(
                    src_table=e.dst_table,
                    src_cols=e.dst_cols,
                    dst_table=e.src_table,
                    dst_cols=e.src_cols,
                    inference_tag=e.inference_tag,
                )
            )
    for t in adj:
        adj[t] = sorted(adj[t], key=lambda x: _edge_key(x))

    debug("[schema.reflect_schema] computing shortest join paths")
    tlist = sorted(tables.keys())
    join_paths_multi = _compute_join_paths_multi_from_adj(adj, tlist)

    sg = SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths_multi,
        created_at=datetime.now().isoformat(),
        enum_values=enum_values,
    )

    return sg


def build_schema_graph_with_diff(
    dialect: Dialect,
    schema_context: SchemaContext,
    notes_content: str | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
    refresh_existing_descriptions_on_addition: bool = False,
) -> tuple[SchemaGraph, SchemaDiff | None]:
    """
    Load or build the schema graph and report the structural diff if a partial rebuild ran.

    Args:

        dialect: Active database dialect with reflection and profiling behavior.

        schema_context: Scope, deny lists, required filters, and object kind for reflection and hashing.

        notes_content: Optional domain notes text passed to LLM column classification when the graph is built (not when served from cache).

        log_sink: Optional callback for user-facing schema status lines; defaults to :func:`_core_utils.notify`.

        refresh_existing_descriptions_on_addition: When true and a partial rebuild adds tables, run an extra full-graph classifier pass to refresh descriptions/roles on unchanged tables (see :func:`apply_diff`).

    Returns:

        Tuple of ``(SchemaGraph, SchemaDiff | None)``. The diff is non-``None`` only when
        the partial-rebuild branch ran (``ddl_probe_hash`` mismatch with a usable cache);
        all cache-hit, notes-refresh, scope-subset, and full-rebuild branches return ``None``.
    """

    sink: Callable[[str], None] = log_sink if log_sink is not None else _core_utils.notify
    schema_json_path = EngineConfig.SCHEMA_JSON_PATH

    debug(f"[schema.build_schema_graph] engine_type={dialect.name}")

    if not PolicyConfig.REGENERATE_SCHEMA_GRAPH and os.path.exists(schema_json_path):
        debug(f"[schema.build_schema_graph] loading from cache '{schema_json_path}'")
        try:
            d = read_gzip_json(schema_json_path)
        except (
            OSError,
            EOFError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            debug(f"[schema.build_schema_graph] corrupt cache unreadable: {exc!r}")
            adir = os.path.dirname(os.path.abspath(schema_json_path))
            write_artifact_manifest(
                adir,
                last_corruption_at=datetime.now(timezone.utc).isoformat(),
                last_action="corrupt_schema_cache",
            )
            try:
                os.remove(schema_json_path)
            except OSError:
                pass
        else:
            incoming_notes_hash = _notes_content_sha256(notes_content)
            incoming_scope_hash = scope_hash_fp(schema_context)
            incoming_probe = compute_dialect_probe(dialect, schema_context)
            cached_probe = str(d.get("ddl_probe_hash", "") or "")
            cached_scope = str(d.get("scope_hash", "") or "")
            cached_notes = str(d.get("notes_hash", "") or "")

            if incoming_probe and incoming_probe == cached_probe and incoming_scope_hash == cached_scope:
                tables_fast = {k: _table_from_dict(v) for k, v in d["tables"].items()}
                join_paths_multi = d.get("join_paths_multi")
                if not join_paths_multi:
                    join_paths_multi = {}
                    jp = d.get("join_paths", {})
                    for a in jp:
                        join_paths_multi[a] = {}
                        for b in jp[a]:
                            join_paths_multi[a][b] = [jp[a][b]] if jp[a][b] is not None else []
                model = dict(d)
                model["join_paths_multi"] = join_paths_multi
                model["tables"] = {k: _table_to_dict(v) for k, v in tables_fast.items()}
                sg = SchemaGraph.from_dict(model)
                _ensure_semantic_join_neighbors(sg)

                stamped_pr = str(d.get("profiling_hash", "") or "")
                pr_live = profiling_hash_fp(tables_profiling_payload(sg.tables))
                if stamped_pr and pr_live != stamped_pr:
                    sink("  Schema: profiling fingerprint drift — refreshing column statistics...")
                    _add_profiling_data(
                        dialect,
                        sg,
                        notes_content=notes_content,
                        schema_json_path=schema_json_path,
                        log_sink=sink,
                    )
                    sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
                    sg.refresh_schema_stats()

                if incoming_notes_hash == cached_notes:
                    debug(
                        "[schema.build_schema_graph] cache_hit_via_probe "
                        f"({len(sg.tables)} relations, probe={incoming_probe[:16]!r})"
                    )
                    sg.notes_sha256 = incoming_notes_hash
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    sg.ddl_probe_hash = incoming_probe
                    _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
                    notify_schema_path_health(sg)
                    sink(f"  Schema: cache hit ({len(sg.tables)} relations).")
                    return sg, None

                debug(
                    "[schema.build_schema_graph] notes_refresh_only "
                    f"(was={cached_notes[:16]!r} incoming={incoming_notes_hash[:16]!r})"
                )
                pinned = _user_pinned_columns_from_sidecar(schema_json_path)
                rerun_column_classifier(sg, notes_content, skip_columns=pinned, log_sink=sink)
                _redact_hidden_sensitivity_profile_values(sg)
                sg.notes_sha256 = incoming_notes_hash
                assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                sg.ddl_probe_hash = incoming_probe
                _save_schema_to_cache(sg, schema_json_path)
                _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
                notify_schema_path_health(sg)
                sink(f"  Schema: notes-only refresh ({len(sg.tables)} relations).")
                return sg, None

            cached_descriptor = d.get("scope_descriptor")
            if incoming_probe and incoming_probe == cached_probe and isinstance(cached_descriptor, dict):
                old_ctx = schema_context_from_descriptor(cached_descriptor)
                change = classify_scope_change(old_ctx, schema_context)
                if change == "subset":
                    debug(f"[schema.build_schema_graph] scope_subset_filter (cached_tables={len(d.get('tables', {}))})")
                    tables_fast = {k: _table_from_dict(v) for k, v in d["tables"].items()}
                    join_paths_multi = d.get("join_paths_multi") or {}
                    model = dict(d)
                    model["join_paths_multi"] = join_paths_multi
                    model["tables"] = {k: _table_to_dict(v) for k, v in tables_fast.items()}
                    cached_sg = SchemaGraph.from_dict(model)
                    _ensure_semantic_join_neighbors(cached_sg)
                    sg = _filter_schema_graph_by_scope(cached_sg, schema_context)
                    if incoming_notes_hash != cached_notes:
                        pinned = _user_pinned_columns_from_sidecar(schema_json_path)
                        rerun_column_classifier(sg, notes_content, skip_columns=pinned, log_sink=sink)
                        _redact_hidden_sensitivity_profile_values(sg)
                    sg.notes_sha256 = incoming_notes_hash
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    sg.ddl_probe_hash = incoming_probe
                    _save_schema_to_cache(sg, schema_json_path)
                    _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
                    notify_schema_path_health(sg)
                    sink(f"  Schema: scope-subset filter ({len(sg.tables)} relations).")
                    return sg, None
                if change == "superset":
                    debug(
                        "[schema.build_schema_graph] scope_superset_falling_through_to_rebuild "
                        "(partial-add not yet implemented)"
                    )
                elif change == "orthogonal":
                    debug("[schema.build_schema_graph] scope_orthogonal_falling_through_to_rebuild")

            if incoming_probe and cached_probe and incoming_probe != cached_probe:
                debug(
                    "[schema.build_schema_graph] probe_mismatch_reflecting "
                    f"(was={cached_probe[:16]!r} incoming={incoming_probe[:16]!r})"
                )

                try:
                    tables_cached = {k: _table_from_dict(v) for k, v in d["tables"].items()}
                    join_paths_cached = d.get("join_paths_multi") or {}
                    cached_model = dict(d)
                    cached_model["join_paths_multi"] = join_paths_cached
                    cached_model["tables"] = {k: _table_to_dict(v) for k, v in tables_cached.items()}
                    cached_sg = SchemaGraph.from_dict(cached_model)
                    _ensure_semantic_join_neighbors(cached_sg)

                    new_struct = dialect.reflect_only(schema_context)
                    _validate_scope_against_graph(new_struct, schema_context)
                    _apply_allow_objects_filter(new_struct, schema_context)
                    _strip_schema_context_denied_columns(new_struct, schema_context)
                    _apply_schema_context_allow_columns(new_struct, schema_context)

                    diff = diff_schemas(cached_sg, new_struct)
                    debug(
                        "[schema.build_schema_graph] schema_diff: "
                        f"+tables={len(diff.added_tables)} -tables={len(diff.dropped_tables)} "
                        f"renames={len(diff.table_renames)} per_table={len(diff.per_table)}"
                    )

                    diff = resolve_table_renames(
                        diff,
                        cached_sg,
                        new_struct,
                        dialect,
                        notes_content=notes_content,
                    )

                    diff = resolve_column_renames(
                        diff,
                        cached_sg,
                        new_struct,
                        dialect,
                        notes_content=notes_content,
                    )
                    debug(
                        "[schema.build_schema_graph] schema_diff_resolved: "
                        f"+tables={len(diff.added_tables)} -tables={len(diff.dropped_tables)} "
                        f"renames={len(diff.table_renames)} per_table={len(diff.per_table)}"
                    )
                    sg = apply_diff(
                        cached_sg,
                        new_struct,
                        diff,
                        dialect,
                        notes_content=notes_content,
                        schema_json_path=schema_json_path,
                        refresh_existing_descriptions_on_addition=refresh_existing_descriptions_on_addition,
                    )
                    sg.notes_sha256 = incoming_notes_hash
                    _raise_if_schema_unusable(sg, schema_context)
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    sg.ddl_probe_hash = incoming_probe
                    _save_schema_to_cache(sg, schema_json_path)
                    _migrate_sidecar_for_diff(schema_json_path, diff)
                    _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
                    col_renames_total = sum(len(td.renamed_columns) for td in diff.per_table.values())
                    sink(
                        f"  Schema: partial rebuild — +{len(diff.added_tables)} "
                        f"-{len(diff.dropped_tables)} tables, "
                        f"{len(diff.table_renames)} table-rename, "
                        f"{col_renames_total} column-rename, "
                        f"{len(diff.per_table)} column-changed "
                        f"({len(sg.tables)} relations)."
                    )
                    for (
                        src_t,
                        old_src_c,
                        new_src_c,
                        dst_t,
                        old_dst_c,
                        new_dst_c,
                    ) in diff.ported_user_fks:
                        sink(
                            "  Schema: user FK auto-ported across rename: "
                            f"{src_t}.{old_src_c}->{dst_t}.{old_dst_c} => "
                            f"{src_t}.{new_src_c}->{dst_t}.{new_dst_c}"
                        )
                    for src_t, src_c, dst_t, dst_c, tag in diff.dropped_user_fks:
                        sink(f"  Schema: user FK dropped (endpoint missing) [{tag}]: {src_t}.{src_c}->{dst_t}.{dst_c}")
                    for src_t, src_c, dst_t, dst_c in diff.dropped_catalog_fks:
                        sink(f"  Schema: catalog FK removed by upstream DDL change: {src_t}.{src_c}->{dst_t}.{dst_c}")
                    notify_schema_path_health(sg)
                    return sg, diff
                except SchemaAccessError:
                    raise
                except Exception as exc:
                    debug(
                        f"[schema.build_schema_graph] partial_rebuild_failed: {exc!r}; falling through to full rebuild"
                    )

            tables = {k: _table_from_dict(v) for k, v in d["tables"].items()}

            required_fp_keys = (
                "structural_hash",
                "profiling_hash",
                "scope_hash",
                "effective_structural_hash",
                "notes_hash",
                "semantic_edges_hash",
            )
            if any(k not in d for k in required_fp_keys):
                debug("[schema.build_schema_graph] cache missing fingerprint fields, removing")
                os.remove(schema_json_path)
            else:
                rest = structural_hash_fp(tables_structural_payload(tables))
                prt = profiling_hash_fp(tables_profiling_payload(tables))
                scp = incoming_scope_hash
                eff = effective_structural_hash_fp(rest, scp)
                sem = _semantic_edges_fingerprint(tables)
                if (
                    rest != d["structural_hash"]
                    or prt != d["profiling_hash"]
                    or scp != d["scope_hash"]
                    or eff != d["effective_structural_hash"]
                    or incoming_notes_hash != str(d.get("notes_hash", ""))
                    or sem != str(d.get("semantic_edges_hash", ""))
                ):
                    cached_hash = d.get("effective_structural_hash", d.get("schema_hash", ""))
                    _debug_schema_cache_hash_mismatch(
                        schema_json_path=schema_json_path,
                        stamped_hash=cached_hash,
                        json_tables=d["tables"],
                    )
                    debug(
                        "[schema.build_schema_graph] fingerprint mismatch: "
                        f"structural={rest[:16]!r} vs {str(d.get('structural_hash', ''))[:16]!r}",
                    )
                    debug(
                        "[schema.build_schema_graph] fingerprint mismatch detail: "
                        f"profiling={prt[:16]!r} vs {str(d.get('profiling_hash', ''))[:16]!r}; "
                        f"scope={scp[:16]!r} vs {str(d.get('scope_hash', ''))[:16]!r}; "
                        f"effective={eff[:16]!r} vs {str(d.get('effective_structural_hash', ''))[:16]!r}; "
                        f"notes={incoming_notes_hash[:16]!r} vs {str(d.get('notes_hash', ''))[:16]!r}; "
                        f"semantic={sem[:16]!r} vs {str(d.get('semantic_edges_hash', ''))[:16]!r}"
                    )
                    os.remove(schema_json_path)
                else:
                    join_paths_multi = d.get("join_paths_multi")
                    if not join_paths_multi:
                        join_paths_multi = {}
                        jp = d.get("join_paths", {})
                        for a in jp:
                            join_paths_multi[a] = {}
                            for b in jp[a]:
                                join_paths_multi[a][b] = [jp[a][b]] if jp[a][b] is not None else []

                    debug(f"[schema.build_schema_graph] loaded {len(tables)} relations from cache")

                    model = dict(d)
                    model["join_paths_multi"] = join_paths_multi
                    sg = SchemaGraph.from_dict(model)
                    _ensure_semantic_join_neighbors(sg)
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    if incoming_probe and not cached_probe:
                        sg.ddl_probe_hash = incoming_probe
                        debug("[schema.build_schema_graph] backfilling ddl_probe_hash to cache")
                        _save_schema_to_cache(sg, schema_json_path)
                    elif incoming_probe:
                        sg.ddl_probe_hash = incoming_probe
                    _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
                    notify_schema_path_health(sg)
                    sink(f"  Schema: cache hit ({len(sg.tables)} relations).")
                    return sg, None

    debug("[schema.build_schema_graph] building schema")

    sink("  Schema: building from database (this can take a while)...")
    profiling_started = time.monotonic()
    allow_kw = schema_context.allow_objects if schema_context.allow_objects else None
    sg = dialect.reflect_schema_graph(
        include=_effective_reflect_include(schema_context),
        allow_objects=allow_kw,
    )
    sink(f"  Schema: reflected {len(sg.tables)} relations; applying scope...")
    debug("[schema.build_schema_graph] validating scope against reflected graph")
    _validate_scope_against_graph(sg, schema_context)
    _apply_allow_objects_filter(sg, schema_context)
    _strip_schema_context_denied_columns(sg, schema_context)
    _apply_schema_context_allow_columns(sg, schema_context)
    col_total = sum(len(t.columns) for t in sg.tables.values())
    sink(f"  Schema: profiling {col_total} columns...")
    _add_profiling_data(
        dialect,
        sg,
        notes_content=notes_content,
        schema_json_path=schema_json_path,
        log_sink=sink,
    )

    sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)

    debug("[schema.build_schema_graph] computing schema stats after profiling")
    sg.refresh_schema_stats()
    sg.notes_sha256 = _notes_content_sha256(notes_content)
    _raise_if_schema_unusable(sg, schema_context)
    assign_schema_graph_hashes(sg, schema_context, sg.notes_sha256)
    sg.ddl_probe_hash = compute_dialect_probe(dialect, schema_context)

    _save_schema_to_cache(sg, schema_json_path)
    debug("[schema.build_schema_graph] cache saved with profiling data")
    _finalize_with_overrides(sg, schema_json_path, dialect=dialect)
    notify_schema_path_health(sg)
    sink(
        f"  Schema: built and cached {len(sg.tables)} relations in {time.monotonic() - profiling_started:.1f}s.",
    )

    return sg, None


def build_schema_graph(
    dialect: Dialect,
    schema_context: SchemaContext,
    notes_content: str | None = None,
) -> SchemaGraph:
    """Load or build the schema graph; thin wrapper that discards the diff."""

    sg, _ = build_schema_graph_with_diff(dialect, schema_context, notes_content)
    return sg


def _schema_context_from_graph(sg: SchemaGraph) -> SchemaContext:
    """Reconstruct a :class:`SchemaContext` fingerprint input from a loaded graph."""

    desc = sg.scope_descriptor
    if isinstance(desc, dict) and desc:
        return schema_context_from_descriptor(desc)
    specs: list[str] = []
    for tbl, cols in sg.deny_columns.items():
        for c in cols:
            specs.append(f"{tbl}.{c}")
    return SchemaContext(
        include=sg.include,
        deny_columns=frozenset(specs),
    )


def _databricks_row_kind(table_type: str) -> Literal["table", "view"]:
    """Map an information_schema ``table_type`` string to a graph relation kind."""

    u = table_type.upper()
    if "VIEW" in u and "MATERIALIZED" not in u:
        return "view"
    return "table"


def _tables_meta_to_schema_graph(
    tables_meta: dict[str, dict],
    *,
    object_kind: Literal["table", "view"] = "table",
    row_kind_by_table: dict[str, Literal["table", "view"]] | None = None,
) -> SchemaGraph:
    """
    Convert a raw table metadata dictionary to a fully connected `SchemaGraph`.

    Args:

        tables_meta: Description.

    Returns:

        `created_at`.
    """
    tables: dict[str, TableMetadata] = {}

    for table_name, meta in tables_meta.items():
        row_kind = object_kind
        if row_kind_by_table is not None:
            row_kind = row_kind_by_table.get(str(table_name).lower(), object_kind)
        columns: dict[str, ColumnMetadata] = {}
        col_names = meta.get("column_names_original", [])
        col_types = meta.get("column_types", [])
        pk_cols = meta.get("primary_keys", [])
        pk_set = set(pk_cols)
        nullable_list = meta.get("column_is_nullable")
        use_parsed_nullable = (
            isinstance(nullable_list, list)
            and len(nullable_list) == len(col_names)
            and all(isinstance(x, bool) for x in nullable_list)
        )

        uq_cols = set(meta.get("unique_columns", []) or [])
        for i, col_name in enumerate(col_names):
            col_type = col_types[i] if i < len(col_types) else "UNKNOWN"
            if use_parsed_nullable:
                is_nullable = bool(nullable_list[i])
            else:
                is_nullable = col_name not in pk_set
            if col_name in pk_set:
                is_nullable = False
            columns[col_name] = ColumnMetadata(
                name=col_name,
                data_type=col_type,
                is_primary_key=col_name in pk_cols,
                is_foreign_key=False,
                fk_target=None,
                is_unique=col_name in uq_cols,
                is_nullable=is_nullable,
            )

        fk_edges = []
        for fk in meta.get("foreign_keys", []):
            edge = FKEdge(
                src_table=table_name,
                src_cols=fk["src_cols"],
                dst_table=fk["dst_table"],
                dst_cols=fk["dst_cols"],
            )
            fk_edges.append(edge)

        partition_cols = meta.get("partition_columns", [])
        tables[table_name] = TableMetadata(
            name=table_name,
            columns=columns,
            primary_key=pk_cols,
            foreign_keys=fk_edges,
            partition_columns=partition_cols,
            kind=row_kind,
        )

    fk_count = sum(len(tbl.foreign_keys) for tbl in tables.values())
    debug(f"[schema.tables_meta_to_schema_graph] {len(tables)} tables, {fk_count} FK edges")

    adj: dict[str, list[FKEdge]] = {t: [] for t in tables}
    for tbl in tables.values():
        for e in tbl.foreign_keys:
            if e.src_table not in tables or e.dst_table not in tables:
                continue
            adj[e.src_table].append(e)
            adj[e.dst_table].append(
                FKEdge(
                    src_table=e.dst_table,
                    src_cols=e.dst_cols,
                    dst_table=e.src_table,
                    dst_cols=e.src_cols,
                    inference_tag=e.inference_tag,
                )
            )
    for t in adj:
        adj[t] = sorted(adj[t], key=lambda x: _edge_key(x))

    debug("[schema.tables_meta_to_schema_graph] computing shortest join paths")

    tlist = sorted(tables.keys())
    join_paths_multi = _compute_join_paths_multi_from_adj(adj, tlist)

    sg = SchemaGraph(
        tables=tables,
        join_paths_multi=join_paths_multi,
        created_at=datetime.now().isoformat(),
        enum_values={},
    )
    assign_schema_graph_hashes(sg, SchemaContext(), "")
    assert_schema_invariants(sg)

    return sg


def compute_database_feature_capability(sg: SchemaGraph) -> DatabaseFeatureCapability:
    """
    Derive a once-per-graph capability snapshot for tier and QSim feature gating.

    Args:

        sg: Fully wired schema graph with join path metadata.

    Returns:

        Immutable :class:`DatabaseFeatureCapability` for rebalance and prompt filtering.
    """

    roles: dict[str, str] = {}
    tc = len(sg.tables)
    fk_edge_count = len(sg.fk_edges)
    max_tables_on_path = 1
    max_edge_depth = 0
    for row in sg.join_paths_multi.values():
        for paths in row.values():
            for p in paths:
                if not p:
                    continue
                ec = len(p)
                max_edge_depth = max(max_edge_depth, ec)
                max_tables_on_path = max(max_tables_on_path, ec + 1)

    self_tables: set[str] = set()
    has_self_ref = False
    for e in sg.fk_edges:
        if e.src_table == e.dst_table:
            has_self_ref = True
            self_tables.add(e.src_table)

    agg_by: dict[str, set[str]] = {}
    date_by: dict[str, set[str]] = {}
    arr_by: dict[str, set[str]] = {}
    has_num = False
    has_date = False
    has_arr = False
    has_cat = False

    for tn, tbl in sg.tables.items():
        for cn, col in tbl.columns.items():
            dt = str(col.data_type or "")
            dtl = dt.lower()
            if "array" in dtl or dtl.endswith("[]") or ("[" in dtl and "]" in dtl):
                has_arr = True
                arr_by.setdefault(tn, set()).add(cn)
            role_v = str(col.role or "")
            if is_date_type(dt) or role_v == ColumnRole.TEMPORAL.value:
                has_date = True
                date_by.setdefault(tn, set()).add(cn)
            if is_numeric_type(dt) or role_v == ColumnRole.NUMERIC_MEASURE.value:
                has_num = True
                agg_by.setdefault(tn, set()).add(cn)
            if role_v in (
                ColumnRole.CATEGORICAL.value,
                ColumnRole.NUMERIC_CATEGORICAL.value,
                ColumnRole.BOOLEAN.value,
                ColumnRole.FREE_TEXT.value,
            ):
                has_cat = True

    has_window = False
    for tn in sg.tables:
        if get_groupable_columns(tn, sg, roles) and get_aggregatable_columns(tn, sg, roles):
            has_window = True
            break

    return DatabaseFeatureCapability(
        table_count=tc,
        fk_edge_count=fk_edge_count,
        has_numeric_measures=has_num,
        has_date_columns=has_date,
        has_array_columns=has_arr,
        has_categorical_columns=has_cat,
        max_tables_on_any_join_path=max_tables_on_path,
        max_fk_chain_depth=max_edge_depth,
        has_self_referential_fk=has_self_ref,
        tables_supporting_self_join=frozenset(self_tables),
        has_window_capable_table_sets=has_window,
        aggregatable_columns_by_table={k: frozenset(v) for k, v in agg_by.items()},
        date_columns_by_table={k: frozenset(v) for k, v in date_by.items()},
        array_columns_by_table={k: frozenset(v) for k, v in arr_by.items()},
    )


def assert_schema_invariants(sg: SchemaGraph) -> None:
    """
    Verify the canonical containers on *sg* remain consistent with their derived properties.

    Raises:class:`SchemaInvariantError` when any of the following violations is found:

        1. A primary-key column listed in ``TableMetadata.primary_key`` is missing from ``TableMetadata.columns``.
        2. A foreign-key edge references a source column that is not present on the source table. (Edges that point to a destination table absent from the graph are tolerated as a normal consequence of scope filtering and skipped.)
        3. A column's ``_owner_table`` back-reference is unwired or points to a different table than the one that owns it.
        4. A table's ``_owner_graph`` back-reference is unwired or points to a different graph instance.
        5. ``SchemaGraph.deny_columns`` references an unknown table or column.
        6. ``SchemaGraph._canonical_bearers`` records a bearer that is not present in the schema or names a column whose lower-cased name does not match the index key.
        7. A column has ``role`` set without a matching :class:`RoleOwner` provenance.
    """
    for tname, tbl in sg.tables.items():
        if getattr(tbl, "_owner_graph", None) is not sg:
            raise SchemaInvariantError(f"table {tname!r} owner_graph back-reference is not the enclosing SchemaGraph")
        for pk_col in tbl.primary_key:
            if pk_col not in tbl.columns:
                raise SchemaInvariantError(f"primary-key column {tname}.{pk_col} not present in columns")
        for fk in tbl.foreign_keys:
            for sc in fk.src_cols:
                if sc not in tbl.columns:
                    raise SchemaInvariantError(f"FK src column missing: {fk.src_table}.{sc}")
            dst_tbl = sg.tables.get(fk.dst_table)
            if dst_tbl is None:
                continue
            for dc in fk.dst_cols:
                if dc not in dst_tbl.columns:
                    raise SchemaInvariantError(f"FK dst column missing: {fk.dst_table}.{dc}")
        for cname, col in tbl.columns.items():
            if getattr(col, "_owner_table", None) is not tbl:
                raise SchemaInvariantError(
                    f"column {tname}.{cname} owner_table back-reference is not the enclosing table"
                )
            if col.role and col.role_owner is None:
                raise SchemaInvariantError(f"column {tname}.{cname} has role={col.role!r} without role_owner")
    for dtbl, dcols in (sg.deny_columns or {}).items():
        tbl = sg.tables.get(dtbl)
        if tbl is None:
            raise SchemaInvariantError(f"deny_columns references unknown table: {dtbl}")
        for dc in dcols:
            if dc not in tbl.columns:
                raise SchemaInvariantError(f"deny_columns references unknown column: {dtbl}.{dc}")
    bearers = getattr(sg, "_canonical_bearers", {}) or {}
    for key, (btbl, bcol) in bearers.items():
        tbl = sg.tables.get(btbl)
        if tbl is None or bcol not in tbl.columns:
            raise SchemaInvariantError(f"canonical bearer {btbl}.{bcol} is missing from schema")
        if bcol.lower() != key:
            raise SchemaInvariantError(f"canonical bearer index key {key!r} does not match column name {bcol!r}")


def _recompute_join_paths_multi(
    tables: dict[str, TableMetadata],
) -> dict[str, dict[str, list[list[dict[str, Any]]]]]:
    """
    Recompute ``join_paths_multi`` from current ``TableMetadata`` FK edges.

    Args:

        tables: All tables in the graph.

    Returns:

        Fresh join path map between every ordered table pair.
    """
    adj: dict[str, list[FKEdge]] = {t: [] for t in tables}
    for tbl in tables.values():
        for e in tbl.foreign_keys:
            if e.src_table not in tables or e.dst_table not in tables:
                continue
            adj[e.src_table].append(e)
            adj[e.dst_table].append(
                FKEdge(
                    src_table=e.dst_table,
                    src_cols=e.dst_cols,
                    dst_table=e.src_table,
                    dst_cols=e.src_cols,
                    inference_tag=e.inference_tag,
                )
            )
    for t in adj:
        adj[t] = sorted(adj[t], key=lambda x: _edge_key(x))

    tlist = sorted(tables.keys())
    return _compute_join_paths_multi_from_adj(adj, tlist)


def _resolve_graph_table_name(raw_name: str, graph_tables: set[str]) -> str | None:
    """
    Map a DDL or catalog table name to a key present in *graph_tables*.

    Args:

        raw_name: Description.

        graph_tables: Description.

    Returns:

        Return value.
    """

    if raw_name in graph_tables:
        return raw_name
    lower_index = {t.lower(): t for t in graph_tables}
    return lower_index.get(raw_name.lower())


def merge_ddl_foreign_keys_into_schema_graph(
    sg: SchemaGraph,
    ddl_tables: dict[str, dict[str, Any]],
) -> None:
    """
    Add FK edges from parsed DDL into *sg* when endpoints exist, then refresh paths.

    Args:

        sg: Live schema graph mutated in place.

        ddl_tables: Output of :func:`schema_profiling.parse_sql_file`.

    Returns:

        None.
    """
    if not ddl_tables or not sg.tables:
        return
    graph_names = set(sg.tables.keys())
    for ddl_table, meta in ddl_tables.items():
        src_resolved = _resolve_graph_table_name(ddl_table, graph_names)
        if not src_resolved:
            continue
        src_tbl = sg.tables[src_resolved]
        existing = {_edge_key(e) for e in src_tbl.foreign_keys}
        for fk in meta.get("foreign_keys", []) or []:
            dst_raw = fk.get("dst_table", "")
            dst_resolved = _resolve_graph_table_name(str(dst_raw), graph_names)
            if not dst_resolved:
                continue
            src_cols = list(fk.get("src_cols", []) or [])
            dst_cols = list(fk.get("dst_cols", []) or [])
            if len(src_cols) != len(dst_cols) or not src_cols:
                continue
            if any(c not in src_tbl.columns for c in src_cols):
                continue
            dst_tbl = sg.tables[dst_resolved]
            if any(c not in dst_tbl.columns for c in dst_cols):
                continue
            edge = FKEdge(
                src_table=src_resolved,
                src_cols=src_cols,
                dst_table=dst_resolved,
                dst_cols=dst_cols,
            )
            ek = _edge_key(edge)
            if ek in existing:
                continue
            existing.add(ek)
            src_tbl.foreign_keys.append(edge)

    sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
    assign_schema_graph_hashes(sg, _schema_context_from_graph(sg), sg.notes_sha256)


def load_or_create_schema_postgresql(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
) -> SchemaGraph:
    """
    Build a `SchemaGraph` for PostgreSQL from a live database or SQL file fallback.

    Args:

        engine: SQLAlchemy `Engine` connected to the PostgreSQL database.

        include: Which relation kinds to reflect from the catalog.

    Returns:

        `SchemaGraph` built from the database or SQL file.
    """
    try:
        debug("[schema.load_or_create_schema_postgresql] reflecting_database")
        sidecar_path = schema_json_path if schema_json_path is not None else EngineConfig.SCHEMA_JSON_PATH
        if include == "both":
            sg = _merge_reflected_schema_graphs(
                _reflect_schema(
                    engine,
                    object_kind="table",
                    allow_objects=allow_objects,
                    schema_json_path=sidecar_path,
                ),
                _reflect_schema(
                    engine,
                    object_kind="view",
                    allow_objects=allow_objects,
                    schema_json_path=sidecar_path,
                ),
            )
        else:
            sg = _reflect_schema(
                engine,
                object_kind="table" if include == "tables" else "view",
                allow_objects=allow_objects,
                schema_json_path=sidecar_path,
            )
        debug(f"[schema.load_or_create_schema_postgresql] reflected: {len(sg.tables)} tables")
        sql_file_path = getattr(EngineConfig.RUNTIME, "SQL_FILE_PATH", None)
        if include in ("tables", "both") and sql_file_path and os.path.exists(sql_file_path) and sg.tables:
            ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
            if ddl_tables:
                merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
        return sg
    except Exception as e:
        debug(f"[schema.load_or_create_schema_postgresql] reflection_failed: {e}")
        sql_file_path = getattr(EngineConfig.RUNTIME, "SQL_FILE_PATH", None)

        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[schema.load_or_create_schema_postgresql] parsing_sql_file: {sql_file_path}")
            tables_meta = parse_sql_file(Path(sql_file_path))

            if not tables_meta or len(tables_meta) == 0:
                raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e

            ok: Literal["table", "view"] = "table" if include != "views" else "view"
            filtered: dict[str, dict] = tables_meta
            allow_lower = _allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
            return _tables_meta_to_schema_graph(filtered, object_kind=ok)
        raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def _ensure_semantic_join_neighbors(sg: SchemaGraph) -> None:
    """Recompute semantic profile edges from cached or freshly profiled ``semantic_distinct_values``."""

    compute_semantic_profile_join_neighbors(sg)


def _redact_hidden_sensitivity_profile_values(sg: SchemaGraph) -> int:
    """
    Null out concrete profile values on every column whose :attr:`SensitivityClassification` is not :attr:`SensitivityClassification.NONE`.

    Clears ``top_k_values``, ``min_val``, and ``max_val`` in place. Distinct counts and ratios are statistical, not value-bearing, and remain so the downstream operation-assignment and qsim layers can still gate behaviour on cardinality. Returns the number of columns that were redacted.
    """

    redacted = 0
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            if col.sensitivity == SensitivityClassification.NONE:
                continue
            if not col.top_k_values and col.min_val is None and col.max_val is None:
                continue
            col.top_k_values = []
            col.min_val = None
            col.max_val = None
            redacted += 1
    return redacted


def _add_profiling_data(
    dialect: Dialect,
    sg: SchemaGraph,
    notes_content: str | None = None,
    *,
    schema_json_path: str | Path | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """
    Add column profiling data to a SchemaGraph in-place.

    Args:

        dialect: Active dialect owning engine or Spark/warehouse connections.

        sg: `SchemaGraph` to enrich in-place.

        notes_content: Optional human-written notes for LLM role and sensitivity hints.

        schema_json_path: When provided, the overrides sidecar at the matching path is read so its ``_internal.pk_block_inferred`` and ``_internal.fk_block_inferred`` envelopes can suppress re-inference of pairs the operator rejected (the inference helpers receive these as ``blocked`` arguments).

        log_sink: Optional callback for user-facing LLM classification status; defaults to :func:`_core_utils.notify`.

    Returns:

        None.
    """
    pk_blocked, fk_blocked = _load_inference_block_lists(schema_json_path)
    debug("[schema.add_profiling_data] Step 1: profiling columns (statistics)")

    dialect.profile_schema(sg)

    debug("[schema.add_profiling_data] Step 1b: inferring missing primary keys from profile")
    inferred_pks = _infer_missing_pks_from_profile(sg.tables, dialect=dialect, blocked=pk_blocked)
    if inferred_pks:
        debug(
            f"[schema.add_profiling_data] inferred {len(inferred_pks)} primary keys; "
            "re-running pair-targeted FK inference"
        )
        added_fks = _pair_targeted_fk_inference(sg, blocked=fk_blocked)
        if added_fks:
            debug(f"[schema.add_profiling_data] added {added_fks} FKs from pair-targeted inference after new PKs")

    debug("[schema.add_profiling_data] Step 2: inferring column and table roles via LLM")
    apply_column_roles_llm(sg, notes_content=notes_content, log_sink=log_sink)

    debug("[schema.add_profiling_data] Step 2b: boolean coercion pass")
    apply_boolean_coercion_pass(sg)

    debug("[schema.add_profiling_data] Step 2c: redacting profile values for hidden-sensitivity columns")
    _redact_hidden_sensitivity_profile_values(sg)

    debug("[schema.add_profiling_data] Step 3: assigning column operations (deterministic)")
    assign_column_ops(sg)

    debug("[schema.add_profiling_data] Step 4: semantic profile join neighbors")
    compute_semantic_profile_join_neighbors(sg)

    debug("[schema.add_profiling_data] Step 4b: cross-component semantic promotions (bridging islands)")
    pc = _promote_cross_component_semantic_edges(sg)
    if pc:
        debug(f"[schema.add_profiling_data] promoted {pc} cross-component semantic edges to FKs")
    debug("[schema.add_profiling_data] Step 4b2: same-component semantic promotions")
    ps = _promote_same_component_semantic_edges(sg)
    if ps:
        debug(f"[schema.add_profiling_data] promoted {ps} same-component semantic edges to FKs")

    debug("[schema.add_profiling_data] Step 4c: coercing PK and FK columns to identifier role")
    coerced = _coerce_pk_fk_columns_to_identifier(sg)
    if coerced:
        debug(f"[schema.add_profiling_data] coerced {len(coerced)} PK/FK columns to identifier role")

    debug("[schema.add_profiling_data] Step 4d: marking canonical bearer per duplicated column name")
    demoted = _mark_canonical_duplicates(sg)
    if demoted:
        debug(f"[schema.add_profiling_data] demoted {demoted} non-canonical duplicate columns")


def _schema_cache_json_blob(cache_data: dict[str, Any]) -> str:
    """
    Serialize schema cache data as one compact JSON document.

    Args:

        cache_data: Schema cache payload (tables, join_paths_multi, etc.).

    Returns:

        UTF-8 JSON text with sorted object keys.
    """
    return json.dumps(
        cache_data,
        ensure_ascii=False,
        separators=JSON_COMPACT_SEPARATORS,
        sort_keys=True,
    )


def _tables_payload_through_model_round_trip(
    tables_json: dict[str, Any],
) -> dict[str, Any]:
    """
    Rebuild table dicts by parsing into TableMetadata and serializing back to plain dicts.

    Args:

        tables_json: The ``tables`` object produced by ``json.load`` of a cache file.

    Returns:

        Mapping of table name to dict suitable for ``schema_hash_fp``.
    """
    return {name: _table_to_dict(_table_from_dict(blob)) for name, blob in tables_json.items()}


def _fingerprint_tables_after_document_round_trip(cache_data: dict[str, Any]) -> str:
    """
    Compute ``schema_hash_fp`` for ``tables`` after an in-memory write/parse of the full document.

    Args:

        cache_data: Full on-disk cache structure before writing to a physical path.

    Returns:

        Hex digest for the reparsed ``tables`` value.
    """
    reparsed = json.loads(_schema_cache_json_blob(cache_data))
    return schema_hash_fp(reparsed["tables"])


def _first_table_where_stable_json_differs(
    left_tables: dict[str, Any],
    right_tables: dict[str, Any],
) -> str | None:
    """
    Return the first table name whose single-slot stable JSON differs between mappings.

    Args:

        left_tables: First ``tables`` mapping.

        right_tables: Second ``tables`` mapping.

    Returns:

        A table name, or ``None`` when every shared slot matches.
    """
    names = sorted(set(left_tables) | set(right_tables))
    for name in names:
        left_json = stable_json({name: left_tables.get(name)})
        right_json = stable_json({name: right_tables.get(name)})
        if left_json != right_json:
            return name
    return None


def _debug_clip_stable_json(label: str, payload: Any) -> None:
    """
    Emit one debug line with a clipped ``stable_json`` rendering of payload.

    Args:

        label: Log prefix tag.

        payload: JSON-serialisable value.

    Returns:

        None.
    """
    clip = PolicyConfig.SCHEMA_CACHE_HASH_DEBUG_CLIP_CHARS
    text = stable_json(payload)
    if len(text) <= clip:
        debug(f"[schema.cache_hash_debug] {label} chars={len(text)} body={text}")
    else:
        debug(f"[schema.cache_hash_debug] {label} chars={len(text)} head={text[:clip]!r}")


def _debug_schema_cache_hash_mismatch(
    *,
    schema_json_path: str,
    stamped_hash: str,
    json_tables: dict[str, Any],
) -> None:
    """
    Emit phased diagnostics when the file ``schema_hash`` disagrees with ``schema_hash_fp(tables)``.

    Args:

        schema_json_path: Absolute path to the cache file.

        stamped_hash: On-disk ``schema_hash`` field.

        json_tables: On-disk ``tables`` object from ``json.load``.

    Returns:

        None.
    """
    fp_json = schema_hash_fp(json_tables)
    normalized_tables = _tables_payload_through_model_round_trip(json_tables)
    fp_norm = schema_hash_fp(normalized_tables)
    debug(
        "[schema.cache_hash_debug] mismatch_summary "
        f"path={schema_json_path!r} "
        f"stamped_prefix={stamped_hash[:16]!r} "
        f"fp_json_prefix={fp_json[:16]!r} "
        f"fp_after_model_round_trip_prefix={fp_norm[:16]!r}"
    )
    if fp_json != fp_norm:
        debug(
            "[schema.cache_hash_debug] phase=model_round_trip_drift "
            "fingerprint(json tables) != fingerprint(after _table_from_dict/_table_to_dict)"
        )
        diverge = _first_table_where_stable_json_differs(json_tables, normalized_tables)
        if diverge is not None:
            debug(f"[schema.cache_hash_debug] first_diverging_table={diverge!r}")
            _debug_clip_stable_json("json_table_slot", {diverge: json_tables.get(diverge)})
            _debug_clip_stable_json("normalized_table_slot", {diverge: normalized_tables.get(diverge)})
    else:
        debug("[schema.cache_hash_debug] phase=json_vs_model_round_trip_ok raw json tables and model round-trip agree")
    if stamped_hash != fp_json:
        debug(
            "[schema.cache_hash_debug] phase=stamped_vs_json_tables "
            "on-disk schema_hash is not schema_hash_fp(tables); writer/stamp desync or foreign edit"
        )


def _debug_verify_schema_cache_write(cache_data: dict[str, Any], schema_json_path: str) -> None:
    """
    Emit debug lines when stamped hash or document round-trip disagrees with table fingerprints.

    Args:

        cache_data: Payload about to be written.

        schema_json_path: Target path for context.

    Returns:

        None.
    """
    tables_payload = cache_data["tables"]
    fp_tables = schema_hash_fp(tables_payload)
    stamped = cache_data.get("effective_structural_hash", cache_data.get("schema_hash", ""))
    if fp_tables != stamped:
        debug(
            "[schema.cache_hash_debug] write_path_stamp_mismatch "
            f"path={schema_json_path!r} "
            f"fp_tables_prefix={fp_tables[:16]!r} "
            f"effective_structural_hash_field_prefix={stamped[:16]!r}"
        )
    fp_round_trip = _fingerprint_tables_after_document_round_trip(cache_data)
    if fp_round_trip != fp_tables:
        debug(
            "[schema.cache_hash_debug] write_document_round_trip_drift "
            f"path={schema_json_path!r} "
            f"fp_tables_prefix={fp_tables[:16]!r} "
            f"fp_after_stringio_round_trip_prefix={fp_round_trip[:16]!r}"
        )
    _debug_verify_profiling_round_trip(cache_data, schema_json_path)


def _debug_verify_profiling_round_trip(cache_data: dict[str, Any], schema_json_path: str) -> None:
    """
    Detect drift between the stamped ``profiling_hash`` and the value recomputed after a
    JSON round-trip of the written tables. On drift, log the first column whose profiling
    dict differs and dump both slots so the offending field is identifiable.
    """
    stamped_pr = str(cache_data.get("profiling_hash", ""))
    if not stamped_pr:
        return
    tables_json = cache_data["tables"]
    reparsed = json.loads(_schema_cache_json_blob(cache_data))["tables"]
    reloaded_tables = {k: _table_from_dict(v) for k, v in reparsed.items()}
    pr_after = profiling_hash_fp(tables_profiling_payload(reloaded_tables))
    if pr_after == stamped_pr:
        return
    debug(
        "[schema.cache_hash_debug] write_profiling_round_trip_drift "
        f"path={schema_json_path!r} "
        f"stamped_profiling_prefix={stamped_pr[:16]!r} "
        f"recomputed_after_round_trip_prefix={pr_after[:16]!r}"
    )
    pre_tables = {k: _table_from_dict(v) for k, v in tables_json.items()}
    pre_payload = tables_profiling_payload(pre_tables)
    post_payload = tables_profiling_payload(reloaded_tables)
    for tname in sorted(set(pre_payload) | set(post_payload)):
        pre_tbl = pre_payload.get(tname, {})
        post_tbl = post_payload.get(tname, {})
        pre_cols = pre_tbl.get("columns", {})
        post_cols = post_tbl.get("columns", {})
        for cname in sorted(set(pre_cols) | set(post_cols)):
            pre_col = pre_cols.get(cname)
            post_col = post_cols.get(cname)
            if pre_col == post_col:
                continue
            pre_keys = set(pre_col or {})
            post_keys = set(post_col or {})
            differing_fields = sorted(
                {k for k in (pre_keys | post_keys) if (pre_col or {}).get(k) != (post_col or {}).get(k)}
            )
            debug(
                "[schema.cache_hash_debug] profiling_first_diverging_column "
                f"table={tname!r} column={cname!r} differing_fields={differing_fields}"
            )
            _debug_clip_stable_json("profiling_pre_round_trip", {cname: pre_col})
            _debug_clip_stable_json("profiling_post_round_trip", {cname: post_col})
            return


def _save_schema_to_cache(sg: SchemaGraph, schema_json_path: str) -> None:
    """
    Save a SchemaGraph to a JSON cache file.

    Args:

        sg: `SchemaGraph` to persist.

        schema_json_path: Absolute path to the output JSON file.

    Returns:

        None.
    """
    cache_data = {
        "tables": {k: _table_to_dict(v) for k, v in sg.tables.items()},
        "join_paths_multi": sg.join_paths_multi,
        "structural_hash": sg.structural_hash,
        "profiling_hash": sg.profiling_hash,
        "scope_hash": sg.scope_hash,
        "effective_structural_hash": sg.effective_structural_hash,
        "include": sg.include,
        "notes_hash": sg.notes_hash,
        "semantic_edges_hash": sg.semantic_edges_hash,
        "ddl_probe_hash": sg.ddl_probe_hash,
        "created_at": sg.created_at,
        "enum_values": sg.enum_values or {},
        "schema_stats": sg.schema_stats or {},
        "deny_columns": {k: sorted(v) for k, v in sg.deny_columns.items()},
        "disallowed_columns": {k: sorted(v) for k, v in sg.disallowed_columns.items()},
        "notes_sha256": sg.notes_sha256,
        "scope_descriptor": sg.scope_descriptor,
    }

    _debug_verify_schema_cache_write(cache_data, schema_json_path)

    debug(f"[schema.save_schema_to_cache] saving to '{schema_json_path}'")
    adir = os.path.dirname(os.path.abspath(schema_json_path))
    with artifact_lock(adir):
        write_gzip_json_atomic(schema_json_path, cache_data, sort_keys=True)
        write_artifact_manifest(
            adir,
            structural_hash=sg.structural_hash,
            profiling_hash=sg.profiling_hash,
            scope_hash=sg.scope_hash,
            effective_structural_hash=sg.effective_structural_hash,
            notes_hash=sg.notes_hash,
            semantic_edges_hash=sg.semantic_edges_hash,
            last_migration_tier=MigrationTier.NO_CHANGE.value,
            last_action="reconcile",
        )


def _filter_databricks_tables_meta(
    tables_meta: dict[str, dict],
    *,
    object_kind: Literal["table", "view"],
    table_types: dict[str, str],
) -> dict[str, dict]:
    """Keep only relations whose Unity ``table_type`` matches *object_kind*."""

    if not table_types:
        return tables_meta
    out: dict[str, dict] = {}
    for name, meta in tables_meta.items():
        typ = table_types.get(name.lower(), "")
        u = typ.upper()
        if object_kind == "view":
            if "VIEW" in u and "MATERIALIZED" not in u:
                out[name] = meta
        else:
            if "VIEW" in u and "MATERIALIZED" not in u:
                continue
            out[name] = meta
    return out


def _filter_databricks_for_include(
    tables_meta: dict[str, dict],
    *,
    include: SchemaInclude,
    table_types: dict[str, str],
) -> dict[str, dict]:
    """Filter catalog metadata to tables, views, or both."""

    if include == "both":
        a = _filter_databricks_tables_meta(tables_meta, object_kind="table", table_types=table_types)
        b = _filter_databricks_tables_meta(tables_meta, object_kind="view", table_types=table_types)
        return {**a, **b}
    return _filter_databricks_tables_meta(
        tables_meta,
        object_kind="table" if include == "tables" else "view",
        table_types=table_types,
    )


def load_or_create_schema_databricks(
    spark_session=None,
    connection=None,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    unity_table_types: dict[str, str],
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> SchemaGraph:
    """
    Build a `SchemaGraph` for Databricks from catalog introspection or SQL DDL fallback.

    Args:

        spark_session: Optional Spark session for catalog introspection.

        connection: Active `databricks.sql` connection for connector-based extraction.

        include: Which relation kinds to retain from the catalog.

        allow_objects: When set, restrict catalog extraction to these relation names (case-insensitive).

        unity_table_types: Lowercased relation name to ``information_schema.tables.table_type`` map from the active dialect.

        structural_constraints_index: PK, FK, and single-column UNIQUE index from the active dialect.

    Returns:

        `SchemaGraph` built from the catalog or SQL file.
    """
    catalog = EngineConfig.RUNTIME.CATALOG
    schema_name = EngineConfig.RUNTIME.SCHEMA
    tables_meta: dict[str, dict] = {}
    spark_used: Any = None
    try:
        if connection is not None:
            tables_meta = extract_tables_from_catalog_sql_connector(
                connection,
                catalog,
                schema_name,
                allow_objects=allow_objects,
                structural_constraints_index=structural_constraints_index,
            )
        else:
            if SparkSession is None:
                raise SchemaAccessError("Cannot build Databricks schema: pyspark is not installed.")
            spark_used = spark_session if spark_session else SparkSession.builder.getOrCreate()
            tables_meta = extract_tables_from_catalog(
                spark_used,
                catalog,
                schema_name,
                allow_objects=allow_objects,
                structural_constraints_index=structural_constraints_index,
            )
    except Exception as e:
        debug(f"[schema.load_or_create_schema_databricks] catalog extraction error: {e}")
        tables_meta = {}

    if not tables_meta:
        sql_file_path = getattr(EngineConfig.RUNTIME, "SQL_FILE_PATH", None)
        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[schema.load_or_create_schema_databricks] catalog empty; parsing SQL file '{sql_file_path}'")
            tables_meta = parse_sql_file(Path(sql_file_path))
            allow_lower = _allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                tables_meta = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
        if not tables_meta:
            raise SchemaAccessError(
                "Cannot build Databricks schema: catalog returned no tables and "
                "SQL file is missing, empty, or unparsable.",
            )

    tables_meta = _filter_databricks_for_include(
        tables_meta,
        include=include,
        table_types=unity_table_types,
    )

    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    row_by: dict[str, Literal["table", "view"]] | None = None
    if include == "both":
        row_by = {n.lower(): _databricks_row_kind(unity_table_types.get(n.lower(), "")) for n in tables_meta}
    sg = _tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=row_by)
    sql_file_path = getattr(EngineConfig.RUNTIME, "SQL_FILE_PATH", None)
    if include in ("tables", "both") and sql_file_path and os.path.exists(sql_file_path) and tables_meta:
        ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
        if ddl_tables:
            merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
    return sg


_DESCRIPTION_REFINER_SYSTEM = (
    "You refine human-written database descriptions so a downstream text-to-SQL LLM can use them effectively. "
    "When previous_text is non-empty, mirror its prose style, length, and structural pattern (sentence shape, "
    "role mentions, qualifier ordering). Preserve every keyword and identifier the human wrote in text "
    "(column names, table names, units, values, conditions, references). Tighten phrasing, remove fluff, make role "
    "and business meaning explicit, and keep wording in plain prose. Do not invent facts the human did not state. "
    "Output ONLY valid JSON."
)


def _description_owner_export_token(owner: DescriptionOwner | None) -> str:
    """Return the canonical ``description_owner`` string stored in overrides JSON."""

    if owner is None:
        return DescriptionOwner.CATALOG.value
    return owner.value


def _role_owner_export_token(owner: RoleOwner | None) -> str:
    """Return the canonical ``role_owner`` string stored in overrides JSON."""

    if owner is None:
        return RoleOwner.CATALOG.value
    return owner.value


def _parse_editable_description_json(raw: Any, path: str) -> tuple[Any, DescriptionOwner]:
    """
    Parse editable ``description`` JSON.

    Accepts a bare string or null, or ``{\"value\": ...}`` with an optional ``owner`` field set only to the schema-overrides export default token for catalog round-trips.

    Provenance is engine-managed for user-authored files; user edits without an export owner apply as ``user_override``.
    """

    if isinstance(raw, dict):
        export_owner: str | None = None
        if "owner" in raw:
            export_owner = str(raw.get("owner") or "").strip()
            if export_owner != SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER:
                raise ValueError(f"{path}: owner is engine-managed and not user-editable")
        extra = set(raw.keys()) - {"value", "owner"}
        if extra:
            raise ValueError(f"{path}: unsupported keys {sorted(extra)!r}")
        if "value" not in raw:
            raise ValueError(f"{path}: object must contain key 'value'")
        val = raw["value"]
        if export_owner == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER:
            if val is not None and not isinstance(val, str):
                raise ValueError(f"{path}: description must be a string or null")
            if isinstance(val, str) and len(val) > SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS:
                raise ValueError(f"{path}: exceeds {SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS} chars")
            return val, DescriptionOwner.CATALOG
    elif raw is None or isinstance(raw, str):
        val = raw
    else:
        raise ValueError(f"{path}: must be a string, null, or object with key 'value'")
    if val is not None and not isinstance(val, str):
        raise ValueError(f"{path}: description must be a string or null")
    if isinstance(val, str) and len(val) > SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS:
        raise ValueError(f"{path}: exceeds {SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS} chars")
    return val, DescriptionOwner.USER_OVERRIDE


def _override_json_null_sentinel(raw: Any) -> bool:
    """Return True when *raw* is JSON null or an export envelope whose ``value`` is null."""

    if raw is None:
        return True
    if isinstance(raw, dict) and "value" in raw and raw.get("value") is None:
        if set(raw.keys()) <= {"value", "owner"}:
            return True
    return False


def _validate_owned_description_json(raw: Any, path: str) -> None:
    """Raise ``ValueError`` when *raw* is not valid editable description JSON."""

    _parse_editable_description_json(raw, path)


def _parse_editable_role_json(
    raw: Any,
    path: str,
    allowed_values: set[str],
    *,
    value_type: str | None = None,
) -> tuple[str | None, RoleOwner]:
    """Parse editable ``role`` JSON (bare token, null, or ``{\"value\": ...}`` with optional export-only ``owner``)."""

    if isinstance(raw, dict):
        export_owner: str | None = None
        if "owner" in raw:
            export_owner = str(raw.get("owner") or "").strip()
            if export_owner != SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER:
                raise ValueError(f"{path}: owner is engine-managed and not user-editable")
        extra = set(raw.keys()) - {"value", "owner"}
        if extra:
            raise ValueError(f"{path}: unsupported keys {sorted(extra)!r}")
        if "value" not in raw:
            raise ValueError(f"{path}: object must contain key 'value'")
        val = raw["value"]
        if export_owner == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER:
            if val is not None and val not in allowed_values:
                raise ValueError(f"{path}: {val!r} not in {sorted(allowed_values)!r}")
            if val is not None and value_type:
                vt = value_type.strip().lower()
                if vt:
                    rv = str(val).strip().lower()
                    allowed_vt = ROLE_VALUE_TYPE_COMPAT.get(rv)
                    if allowed_vt is not None and vt not in allowed_vt:
                        raise ValueError(f"{path}: role {val!r} is incompatible with column value_type {value_type!r}")
            return val, RoleOwner.CATALOG
    elif raw is None or isinstance(raw, str):
        val = raw
    else:
        raise ValueError(f"{path}: must be a string, null, or object with key 'value'")
    if val is not None and val not in allowed_values:
        raise ValueError(f"{path}: {val!r} not in {sorted(allowed_values)!r}")
    if val is not None and value_type:
        vt = value_type.strip().lower()
        if vt:
            rv = str(val).strip().lower()
            allowed_vt = ROLE_VALUE_TYPE_COMPAT.get(rv)
            if allowed_vt is not None and vt not in allowed_vt:
                raise ValueError(f"{path}: role {val!r} is incompatible with column value_type {value_type!r}")
    return val, RoleOwner.USER_OVERRIDE


def _validate_owned_role_json(
    raw: Any,
    path: str,
    allowed_values: set[str],
    *,
    value_type: str | None = None,
) -> None:
    """Raise ``ValueError`` when *raw* is not valid editable role JSON."""

    _parse_editable_role_json(raw, path, allowed_values, value_type=value_type)


def _v4_value_owner_envelope(raw: Any) -> dict[str, str]:
    """
    Wrap a catalog-derived string or enum-like value for schema-overrides v4 export.

    Args:

        raw: Description text, role label, or other metadata coerced with ``str``.

    Returns:

        Dict with ``value`` and ``owner`` keys using the default export owner constant.
    """

    text = "" if raw is None else str(raw)
    return {"value": text, "owner": SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER}


def _column_override_value_dict(col: ColumnMetadata) -> dict[str, Any]:
    """
    Return the editable subset of a ``ColumnMetadata`` for an overrides dump using v4 owner envelopes.

    Args:

        col: Column metadata row from the live graph.

    Returns:

        Dictionary suitable for nested inclusion under table overrides with v4-shaped description and role fields.
    """

    vt = (col.value_type or "").strip().lower() or (
        data_type_to_value_type(col.data_type).lower() if col.data_type else ""
    )
    out: dict[str, Any] = {
        "description": _v4_value_owner_envelope(col.description or ""),
        "role": _v4_value_owner_envelope(col.role),
        "sensitivity": col.sensitivity.value,
    }
    if vt == "boolean":
        out["boolean_truth_value"] = col.boolean_truth_value
    return out


def _table_override_value_dict(table: TableMetadata) -> dict[str, Any]:
    """
    Return the editable subset of a ``TableMetadata`` for an overrides dump using v4 owner envelopes.

    Args:

        table: Table metadata row from the live graph.

    Returns:

        Dictionary with v4-shaped table description and role plus nested column override payloads.
    """

    return {
        "description": _v4_value_owner_envelope(table.description or ""),
        "role": _v4_value_owner_envelope(table.role),
        "columns": {cname: _column_override_value_dict(table.columns[cname]) for cname in table.columns},
    }


def _fk_endpoint_string(table: str, cols: list[str]) -> Any:
    """Render an FK endpoint as the dotted shorthand for single-column edges or the list form for composites."""

    if len(cols) == 1:
        return f"{table}.{cols[0]}"
    return [f"{table}.{c}" for c in cols]


def _foreign_keys_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """
    Snapshot every FK currently bound to the graph as ``{from, to, inference_tag, removable, declared}`` records.

    The ``inference_tag`` field exposes which inference layer produced each edge: ``None`` denotes a catalog FK declared by the database itself, ``"suffix"``/``"self"``/``"composite"`` denote suffix-name inference variants, ``"semantic"`` denotes a value-overlap promotion, and ``"user_override_*"`` denotes an FK added through ``foreign_keys_add``. The ``removable`` boolean tells the editor whether the edge can be cited under ``foreign_keys_remove`` (true for inferred and user-override edges; false for catalog edges, which the database itself declares). ``declared`` is true exactly for catalog edges (``inference_tag is None``). Catalog edges remain visible so editors can see them without ambiguity.
    """

    records: list[dict[str, Any]] = []
    for tname in sg.tables:
        tbl = sg.tables[tname]
        for edge in tbl.foreign_keys:
            records.append(
                {
                    "from": _fk_endpoint_string(edge.src_table, list(edge.src_cols)),
                    "to": _fk_endpoint_string(edge.dst_table, list(edge.dst_cols)),
                    "inference_tag": edge.inference_tag,
                    "removable": edge.inference_tag is not None,
                    "declared": edge.inference_tag is None,
                }
            )
    records.sort(
        key=lambda r: (
            str(r["from"]) if isinstance(r["from"], str) else "|".join(r["from"]),
            str(r["to"]) if isinstance(r["to"], str) else "|".join(r["to"]),
        )
    )
    return records


def _primary_keys_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """
    Snapshot every PK currently bound to the graph with provenance.

    Each entry exposes the owning table, the ordered PK column list, and ``pk_inference_tag`` (``None`` for catalog-declared keys, ``"profile"`` for keys promoted by ``_infer_missing_pks_from_profile``).
    """

    records: list[dict[str, Any]] = []
    for tname in sg.tables:
        tbl = sg.tables[tname]
        if not tbl.primary_key:
            continue
        tag: str | None = None
        for col_name in tbl.primary_key:
            col = tbl.columns.get(col_name)
            if col is not None and col.pk_inference_tag is not None:
                tag = col.pk_inference_tag
                break
        records.append(
            {
                "table": tname,
                "columns": list(tbl.primary_key),
                "pk_inference_tag": tag,
                "declared": all(
                    tbl.columns.get(cn) is not None and tbl.columns[cn].pk_inference_tag is None
                    for cn in tbl.primary_key
                ),
            }
        )
    records.sort(key=lambda r: r["table"])
    return records


def _tables_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Snapshot every table's role and description for the read-only export envelope."""

    return sorted(
        (
            {
                "name": tname,
                "role": sg.tables[tname].role,
                "role_owner": (sg.tables[tname].role_owner.value if sg.tables[tname].role_owner is not None else None),
                "description": sg.tables[tname].description or "",
                "description_owner": (
                    sg.tables[tname].description_owner.value if sg.tables[tname].description_owner is not None else None
                ),
            }
            for tname in sg.tables
        ),
        key=lambda r: r["name"],
    )


def _columns_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Snapshot every column's editable state for the read-only export envelope."""

    records: list[dict[str, Any]] = []
    for tname in sg.tables:
        tbl = sg.tables[tname]
        for cname, col in tbl.columns.items():
            vt = (col.value_type or "").strip() or (data_type_to_value_type(col.data_type) if col.data_type else "")
            records.append(
                {
                    "table": tname,
                    "column": cname,
                    "role": col.role,
                    "role_owner": (col.role_owner.value if col.role_owner is not None else None),
                    "sensitivity": col.sensitivity,
                    "is_selectable": bool(col.is_selectable),
                    "description": col.description or "",
                    "description_owner": (col.description_owner.value if col.description_owner is not None else None),
                    "value_type": vt,
                    "boolean_truth_value": col.boolean_truth_value,
                }
            )
    records.sort(key=lambda r: (r["table"], r["column"]))
    return records


def dump_schema_overrides_dict(sg: SchemaGraph) -> dict[str, Any]:
    """
    Build the editable overrides JSON document from a built schema graph.

    The document contains the editable input surface the user is allowed to mutate (``tables``, ``foreign_keys_add``, ``foreign_keys_remove``, ``primary_keys_remove``) and a ``_readonly`` envelope showing the *current* graph state for reference (FKs with ``removable``, PKs with provenance, table/column metadata). Editors should never modify ``_readonly`` — those entries are ignored on apply. Existing user-added FKs are surfaced under ``foreign_keys_add`` so editors can re-export and re-apply without losing them. Internal block lists (used by the system to suppress re-inference of FKs/PKs the user removed) live in the persisted sidecar under ``_internal`` and are never surfaced in the editable JSON.
    """

    return {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {tname: _table_override_value_dict(sg.tables[tname]) for tname in sg.tables},
        "foreign_keys_add": _user_added_fks_dump(sg),
        "foreign_keys_remove": [],
        "primary_keys_add": _user_added_pks_dump(sg),
        "primary_keys_remove": [],
        "_readonly": {
            "foreign_keys_current": _foreign_keys_current_dump(sg),
            "primary_keys_current": _primary_keys_current_dump(sg),
            "tables_current": _tables_current_dump(sg),
            "columns_current": _columns_current_dump(sg),
        },
    }


def _catalog_fk_keys(sg: SchemaGraph) -> set[tuple[str, str, str, str]]:
    """Return canonical keys for every catalog-declared FK edge on *sg*."""

    keys: set[tuple[str, str, str, str]] = set()
    for tbl in sg.tables.values():
        for edge in tbl.foreign_keys:
            if edge.inference_tag is None:
                keys.add(_edge_key(edge))
    return keys


def is_user_asserted_pk_column(col: ColumnMetadata) -> bool:
    """Return True when *col* carries a user-promoted primary-key provenance tag."""

    return col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE


def is_user_asserted_fk_edge(edge: FKEdge, catalog_keys: frozenset[tuple[str, ...]]) -> bool:
    """Return True when *edge* is a user override edge that is not redundant with a catalog FK key."""

    tag = edge.inference_tag or ""
    if not tag.startswith("user_override_"):
        return False
    return _edge_key(edge) not in catalog_keys


def compute_metadata_hash(sg: SchemaGraph) -> str:
    """Return a stable digest of descriptions, roles, and sensitivities across *sg*."""

    tables_payload: dict[str, Any] = {}
    for tname in sorted(sg.tables):
        tbl = sg.tables[tname]
        cols_payload: dict[str, Any] = {}
        for cname in sorted(tbl.columns):
            col = tbl.columns[cname]
            cols_payload[cname] = {
                "description": col.description or "",
                "description_owner": (col.description_owner.value if col.description_owner is not None else None),
                "role": col.role,
                "role_owner": (col.role_owner.value if col.role_owner is not None else None),
                "sensitivity": col.sensitivity,
            }
        tables_payload[tname] = {
            "description": tbl.description or "",
            "description_owner": (tbl.description_owner.value if tbl.description_owner is not None else None),
            "role": tbl.role,
            "role_owner": tbl.role_owner.value if tbl.role_owner is not None else None,
            "columns": cols_payload,
        }
    return hashlib.sha256(stable_json(tables_payload).encode("utf-8")).hexdigest()


def _user_added_pks_dump(sg: SchemaGraph) -> list[dict[str, str]]:
    """Serialize user-promoted primary keys for overrides round-trip."""

    rows: list[dict[str, str]] = []
    for tname, tbl in sg.tables.items():
        for pkc in tbl.primary_key:
            col = tbl.columns.get(pkc)
            if col is not None and is_user_asserted_pk_column(col):
                rows.append({"table": tname, "column": pkc})
    rows.sort(key=lambda r: (r["table"], r["column"]))
    return rows


def _user_added_fks_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Re-emit currently-applied user-asserted FKs (structural FKEdges and semantic neighbor pairs) in ``foreign_keys_add`` shape so a fresh export is a faithful round-trip."""

    out: list[dict[str, Any]] = []
    catalog_keys = _catalog_fk_keys(sg)
    for tname in sg.tables:
        for edge in sg.tables[tname].foreign_keys:
            if not is_user_asserted_fk_edge(edge, catalog_keys):
                continue
            tag = edge.inference_tag or ""
            kind = tag[len("user_override_") :] or "structural"
            out.append(
                {
                    "from": _fk_endpoint_string(edge.src_table, list(edge.src_cols)),
                    "to": _fk_endpoint_string(edge.dst_table, list(edge.dst_cols)),
                    "kind": kind,
                }
            )
    seen_sem: set[tuple[str, str, str, str]] = set()
    for tname in sg.tables:
        for quad in sg.tables[tname]._user_semantic_neighbors:
            canon = tuple(sorted([(quad[0], quad[1]), (quad[2], quad[3])]))
            key = (canon[0][0], canon[0][1], canon[1][0], canon[1][1])
            if key in seen_sem:
                continue
            seen_sem.add(key)
            out.append(
                {
                    "from": _fk_endpoint_string(key[0], [key[1]]),
                    "to": _fk_endpoint_string(key[2], [key[3]]),
                    "kind": "semantic",
                }
            )
    out.sort(
        key=lambda r: (
            str(r["from"]) if isinstance(r["from"], str) else "|".join(r["from"]),
            str(r["to"]) if isinstance(r["to"], str) else "|".join(r["to"]),
            str(r.get("kind", "")),
        )
    )
    return out


def dump_overrides_json_schema(path: str | Path) -> Path:
    """
    Write a JSON Schema describing the editable overrides surface alongside the editable file at *path*.

    The companion file is named ``<stem>.schema.json`` and lists the v4 top-level shape (owned ``description`` / ``role`` objects under ``tables``, ``foreign_keys_add``, ``primary_keys_add``, ``foreign_keys_remove``, ``primary_keys_remove``, ``_readonly``, ``_internal``) plus the editable enum vocabularies sourced from :data:`OVERRIDES_EDITABLE_ENUMS`. Editor tooling (e.g. VS Code) can use this to power autocomplete and inline validation without re-deriving the contract from code.
    """

    target = Path(path).expanduser().resolve()
    schema_path = target.with_name(target.stem + ".schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    enums = OVERRIDES_EDITABLE_ENUMS
    tr_enum = list(enums.get("table_role", []))
    cr_enum = list(enums.get("column_role", []))
    editable_description_schema: dict[str, Any] = {
        "oneOf": [
            {"type": "string", "maxLength": SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS},
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "maxLength": SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS,
                    },
                },
            },
        ],
    }
    editable_table_role_schema: dict[str, Any] = {
        "oneOf": [
            {"type": "string", "enum": tr_enum},
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {
                    "value": {"type": ["string", "null"], "enum": [None, *tr_enum]},
                },
            },
        ],
    }
    editable_column_role_schema: dict[str, Any] = {
        "oneOf": [
            {"type": "string", "enum": cr_enum},
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {
                    "value": {"type": ["string", "null"], "enum": [None, *cr_enum]},
                },
            },
        ],
    }
    schema_doc: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"applied_overrides v{SCHEMA_OVERRIDES_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": ["version"],
        "properties": {
            "version": {"const": SCHEMA_OVERRIDES_VERSION},
            "tables": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "description": editable_description_schema,
                        "role": editable_table_role_schema,
                        "columns": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "description": editable_description_schema,
                                    "role": editable_column_role_schema,
                                    "sensitivity": {
                                        "type": ["string", "null"],
                                        "enum": [
                                            None,
                                            *enums.get("column_sensitivity", []),
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "foreign_keys_add": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["from", "to"],
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": enums.get("foreign_key_kind", []),
                        },
                    },
                },
            },
            "foreign_keys_remove": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["from", "to"],
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                    },
                },
            },
            "primary_keys_add": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["table", "column"],
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                    },
                },
            },
            "primary_keys_remove": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["table", "column"],
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                    },
                },
            },
            "_readonly": {
                "type": "object",
                "description": "system-supplied snapshot; do not edit",
            },
            "_internal": {
                "type": "object",
                "description": "system-managed envelope; do not edit",
            },
        },
    }
    text = json.dumps(schema_doc, indent=2, sort_keys=True)
    tmp = schema_path.with_suffix(schema_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, schema_path)
    return schema_path


def dump_schema_overrides_to_path(sg: SchemaGraph, path: str | Path) -> Path:
    """
    Write the overrides editor document to *path*, replacing any existing file atomically.
    """

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dump_schema_overrides_dict(sg)
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def _validate_overrides_structure(d: Any) -> dict[str, Any]:
    """Structurally validate the overrides JSON document; raise ``ValueError`` with a path on the first problem."""

    if not isinstance(d, dict):
        raise ValueError("overrides JSON must be a top-level object")
    if "foreign_keys_block_inferred" in d:
        raise ValueError(
            "overrides 'foreign_keys_block_inferred' is no longer accepted in the editable JSON; "
            "use 'foreign_keys_remove' instead (the system manages re-inference suppression internally)"
        )
    if "primary_keys_block_inferred" in d:
        raise ValueError(
            "overrides 'primary_keys_block_inferred' is no longer accepted in the editable JSON; "
            "use 'primary_keys_remove' instead"
        )
    allowed = VALID_TOP_LEVEL_OVERRIDE_KEYS | {"_internal"}
    extra = set(d.keys()) - allowed
    if extra:
        raise ValueError(f"unsupported top-level keys in overrides: {sorted(extra)!r}")
    if "version" not in d:
        raise ValueError("overrides JSON missing 'version' field")
    if d["version"] != SCHEMA_OVERRIDES_VERSION:
        raise ValueError(f"overrides 'version' is {d['version']!r}; this build expects {SCHEMA_OVERRIDES_VERSION}")
    tables = d.get("tables", {}) or {}
    if not isinstance(tables, dict):
        raise ValueError("overrides 'tables' must be an object keyed by qualified table name")
    valid_col_roles = {r.value for r in ColumnRole}
    valid_table_roles = {r.value for r in TableRole}
    for tname, tval in tables.items():
        if not isinstance(tval, dict):
            raise ValueError(f"tables.{tname}: must be an object")
        bad = set(tval.keys()) - VALID_TABLE_OVERRIDE_KEYS
        if bad:
            raise ValueError(f"tables.{tname}: unsupported keys {sorted(bad)!r}")
        if "description" in tval:
            _validate_owned_description_json(tval["description"], f"tables.{tname}.description")
        if "role" in tval:
            _validate_owned_role_json(tval["role"], f"tables.{tname}.role", valid_table_roles)
        cols = tval.get("columns", {}) or {}
        if not isinstance(cols, dict):
            raise ValueError(f"tables.{tname}.columns: must be an object")
        for cname, cval in cols.items():
            if not isinstance(cval, dict):
                raise ValueError(f"tables.{tname}.columns.{cname}: must be an object")
            forbidden = {"is_selectable", "is_usable"} & set(cval.keys())
            if forbidden:
                raise ValueError(
                    f"tables.{tname}.columns.{cname}: keys {sorted(forbidden)!r} are system-derived "
                    "and not user-editable; set 'sensitivity' to hygiene, strict, or forbidden to hide a column"
                )
            cbad = set(cval.keys()) - VALID_COLUMN_OVERRIDE_KEYS
            if cbad:
                raise ValueError(f"tables.{tname}.columns.{cname}: unsupported keys {sorted(cbad)!r}")
            if "description" in cval:
                _validate_owned_description_json(cval["description"], f"tables.{tname}.columns.{cname}.description")
            if (
                "sensitivity" in cval
                and cval["sensitivity"] is not None
                and cval["sensitivity"] not in VALID_SENSITIVITY_LEVELS
            ):
                raise ValueError(
                    f"tables.{tname}.columns.{cname}.sensitivity: {cval['sensitivity']!r} "
                    f"not in {sorted(VALID_SENSITIVITY_LEVELS)!r} or null"
                )
            if "role" in cval:
                _validate_owned_role_json(
                    cval["role"],
                    f"tables.{tname}.columns.{cname}.role",
                    valid_col_roles,
                )
            if "boolean_truth_value" in cval:
                bt = cval["boolean_truth_value"]
                if bt is not None and (not isinstance(bt, str) or not str(bt).strip()):
                    raise ValueError(
                        f"tables.{tname}.columns.{cname}.boolean_truth_value: must be null or a non-empty string"
                    )
    fks = d.get("foreign_keys_add", []) or []
    if not isinstance(fks, list):
        raise ValueError("overrides 'foreign_keys_add' must be a list")
    for i, fk in enumerate(fks):
        if not isinstance(fk, dict):
            raise ValueError(f"foreign_keys_add[{i}]: must be an object")
        fbad = set(fk.keys()) - VALID_FK_ADD_KEYS
        if fbad:
            raise ValueError(f"foreign_keys_add[{i}]: unsupported keys {sorted(fbad)!r}")
        if "from" not in fk or "to" not in fk:
            raise ValueError(f"foreign_keys_add[{i}]: missing 'from' or 'to'")
        kind = fk.get("kind", "structural")
        if kind == "logical":
            kind = "structural"
        if kind not in VALID_FK_KINDS:
            raise ValueError(f"foreign_keys_add[{i}].kind: {kind!r} not in {sorted(VALID_FK_KINDS)!r}")
    rem = d.get("foreign_keys_remove", []) or []
    if not isinstance(rem, list):
        raise ValueError("overrides 'foreign_keys_remove' must be a list")
    for i, fk in enumerate(rem):
        if not isinstance(fk, dict):
            raise ValueError(f"foreign_keys_remove[{i}]: must be an object")
        rbad = set(fk.keys()) - VALID_FK_REMOVE_KEYS
        if rbad:
            raise ValueError(f"foreign_keys_remove[{i}]: unsupported keys {sorted(rbad)!r}")
        if "from" not in fk or "to" not in fk:
            raise ValueError(f"foreign_keys_remove[{i}]: missing 'from' or 'to'")
    blk = d.get("primary_keys_remove", []) or []
    if not isinstance(blk, list):
        raise ValueError("overrides 'primary_keys_remove' must be a list")
    for i, pk in enumerate(blk):
        if not isinstance(pk, dict):
            raise ValueError(f"primary_keys_remove[{i}]: must be an object")
        pbad = set(pk.keys()) - VALID_PK_REMOVE_KEYS
        if pbad:
            raise ValueError(f"primary_keys_remove[{i}]: unsupported keys {sorted(pbad)!r}")
        if "table" not in pk or "column" not in pk:
            raise ValueError(f"primary_keys_remove[{i}]: missing 'table' or 'column'")
        if not isinstance(pk["table"], str) or not isinstance(pk["column"], str):
            raise ValueError(f"primary_keys_remove[{i}]: 'table'/'column' must be strings")
    add_pk = d.get("primary_keys_add", []) or []
    if not isinstance(add_pk, list):
        raise ValueError("overrides 'primary_keys_add' must be a list")
    for i, pk in enumerate(add_pk):
        if not isinstance(pk, dict):
            raise ValueError(f"primary_keys_add[{i}]: must be an object")
        pbad = set(pk.keys()) - VALID_PK_ADD_KEYS
        if pbad:
            raise ValueError(f"primary_keys_add[{i}]: unsupported keys {sorted(pbad)!r}")
        if "table" not in pk or "column" not in pk:
            raise ValueError(f"primary_keys_add[{i}]: missing 'table' or 'column'")
        if not isinstance(pk["table"], str) or not isinstance(pk["column"], str):
            raise ValueError(f"primary_keys_add[{i}]: 'table'/'column' must be strings")
    if "_readonly" in d and not isinstance(d["_readonly"], dict):
        raise ValueError("overrides '_readonly' must be an object (read-only envelope)")
    if "_internal" in d and not isinstance(d["_internal"], dict):
        raise ValueError("overrides '_internal' must be an object (system-managed envelope)")
    return d


def _refine_descriptions_via_llm(changes: list[dict[str, Any]]) -> dict[str, str]:
    """
    Send a batch of description edits to the LLM and return ``{path: refined_text}``.

    Each change item includes ``path``, ``kind``, ``text``, and optional ``previous_text`` (the description
    string on the graph before this apply pass). Falls back to the original text for any path missing from
    the LLM response or when LLM credentials are absent.
    """

    if not changes:
        return {}
    if not llm_credentials_configured():
        debug("[schema.refine_descriptions] llm not configured; skipping refinement")
        return {str(c["path"]): str(c["text"]) for c in changes}
    user_payload = stable_json({"items": changes})
    instructions = (
        "For each item, rewrite 'text' into a concise, role-aware description that (a) keeps every keyword the "
        "human added, and (b) matches the style of 'previous_text' when that field is non-empty. "
        "Output JSON of the form "
        '{"items": [{"path": "...", "text": "<refined>"}]} with one entry per input item, in the same order.'
    )
    try:
        raw = llm_chat(
            system=_DESCRIPTION_REFINER_SYSTEM,
            user=instructions + "\n" + user_payload,
            task="schema",
        )
        parsed = json.loads(raw)
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        refined: dict[str, str] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            p = it.get("path")
            t = it.get("text")
            if isinstance(p, str) and isinstance(t, str) and t.strip():
                refined[p] = t.strip()[:SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS]
        for c in changes:
            refined.setdefault(c["path"], c["text"])
        return refined
    except (ValueError, RuntimeError, OSError) as exc:
        debug(f"[schema.refine_descriptions] llm refinement failed; using originals: {exc!r}")
        return {c["path"]: c["text"] for c in changes}


def load_schema_overrides_file(path: str | Path) -> dict[str, Any]:
    """Read *path* as UTF-8 JSON and return the structurally validated overrides document."""

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"overrides file not found: {p!s}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            d = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"overrides JSON parse failed: {exc.msg} (line {exc.lineno})") from exc
    return _validate_overrides_structure(d)


def _split_fk_endpoint(endpoint: Any) -> tuple[str, list[str]] | None:
    """Split a ``"schema.table.col"`` shorthand or ``["schema.table.col", ...]`` into ``(table, [cols])``."""

    if isinstance(endpoint, str):
        parts = endpoint.rsplit(".", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return parts[0], [parts[1]]
    if isinstance(endpoint, list) and endpoint:
        table_name: str | None = None
        cols: list[str] = []
        for ep in endpoint:
            if not isinstance(ep, str):
                return None
            parts = ep.rsplit(".", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return None
            if table_name is None:
                table_name = parts[0]
            elif table_name != parts[0]:
                return None
            cols.append(parts[1])
        if table_name is None:
            return None
        return table_name, cols
    return None


def compute_fk_connected_components(sg: SchemaGraph) -> list[set[str]]:
    """
    Return connected components of the undirected FK+semantic-neighbor join graph as sets of table names.

    Singletons (tables with neither catalog/inferred FKs nor user semantic neighbors) become their own component. The result is sorted: components are ordered by descending size, then by lexicographically smallest member; each set is returned as-is (callers should sort for display).
    """

    table_names = list(sg.tables)
    parent: dict[str, str] = {t: t for t in table_names}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for tname, tbl in sg.tables.items():
        for edge in tbl.foreign_keys:
            if edge.src_table in parent and edge.dst_table in parent:
                _union(edge.src_table, edge.dst_table)
        for col in tbl.columns.values():
            for nbr_tbl, _nbr_col in col.semantic_join_neighbors:
                if nbr_tbl in parent:
                    _union(tname, nbr_tbl)

    groups: dict[str, set[str]] = {}
    for t in table_names:
        r = _find(t)
        groups.setdefault(r, set()).add(t)
    components = list(groups.values())
    components.sort(key=lambda s: (-len(s), min(s) if s else ""))
    return components


def format_disconnected_components_message(components: list[set[str]], sg: SchemaGraph) -> str:
    """
    Format a one-paragraph operator warning describing disconnected FK components in *sg*.

    Returned string is intended for :func:`_core_utils.notify`. Lists each connected group (table island), smallest first, and reminds operators that multi-hop joins require bridging foreign keys.
    """

    if len(components) <= 1:
        return ""
    sized = sorted(components, key=lambda s: (len(s), sorted(s)[0] if s else ""))
    summary_parts = []
    for comp in sized:
        sample = sorted(comp)[:3]
        more = "" if len(comp) <= 3 else f" (+{len(comp) - 3} more)"
        summary_parts.append("{" + ", ".join(sample) + more + "}")
    return (
        f"  Schema warning: join graph splits into {len(components)} disconnected groups: "
        + "; ".join(summary_parts)
        + ". Tables in different groups have no inferable join path; add bridging "
        "`foreign_keys_add` entries (structural or semantic) in the overrides editor."
    )


def format_disconnected_components_warning(components: list[set[str]], sg: SchemaGraph) -> str:
    """Alias for :func:`format_disconnected_components_message` (backward compatibility)."""

    return format_disconnected_components_message(components, sg)


def notify_schema_path_health(sg: SchemaGraph) -> None:
    """Emit a notify line when the FK join graph has more than one connected component."""

    components = compute_fk_connected_components(sg)
    if len(components) > 1:
        msg = format_disconnected_components_message(components, sg)
        if msg:
            notify(msg, stage="schema", code=DIAGNOSTIC_CODE_ENGINE_INFO)


def _coerce_pk_fk_columns_to_identifier(sg: SchemaGraph) -> list[tuple[str, str, str]]:
    """
    Force every PK and FK source column's role to ``IDENTIFIER`` and return the coercion records.

    Returns a list of ``(table, column, prev_role)`` tuples for columns whose role changed.

    Idempotent: re-running on a graph already coerced returns ``[]``. Called at the tail of override apply (and after diff/cache merges) so a user-asserted PK or FK never lingers as ``categorical``/``numeric_categorical``/``free_text``.
    """
    records: list[tuple[str, str, str]] = []
    ident = ColumnRole.IDENTIFIER.value
    for tbl_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            if not (col.is_primary_key or col.is_foreign_key):
                continue
            if col.role_owner == RoleOwner.USER_OVERRIDE:
                debug(
                    f"[schema._coerce_pk_fk_columns_to_identifier] preserving user role for {tbl_name}.{col_name}",
                )
                continue
            if not can_overwrite_role(col.role_owner, RoleOwner.PK_FK_COERCION):
                continue
            prev = col.role or ""
            if prev == ident:
                continue
            col.role = ident
            col.role_owner = RoleOwner.PK_FK_COERCION
            records.append((tbl_name, col_name, prev))
    return records


def apply_schema_overrides_to_graph(
    sg: SchemaGraph,
    overrides: dict[str, Any],
) -> OverrideReport:
    """
    Apply a validated overrides document to *sg* in place and return an ``OverrideReport``.

    Skips entries that reference unknown tables/columns or that would break PK/FK joins, recording each skip in the report. Description fields that differ from the current value are sent through a single batched LLM refinement pass when LLM credentials are configured (the call falls through cleanly with the raw text otherwise).

    The internal envelope ``overrides["_internal"]`` (system-managed; never round-tripped into the editable JSON) carries persistent block lists for inferred FKs and PKs that the user removed. ``foreign_keys_remove`` and ``primary_keys_remove`` always remove the requested edge/PK *and* (when the removed item was inferred, not catalog) auto-promote it into the matching internal block list so subsequent rebuilds suppress re-inference.
    """

    skipped: list[OverrideSkip] = []
    table_edits = 0
    column_edits = 0
    description_changes: list[dict[str, Any]] = []

    description_targets: dict[str, tuple[str, str | None, DescriptionOwner]] = {}
    direct_descriptions_refined = 0

    valid_table_roles = {r.value for r in TableRole}
    valid_col_roles = {r.value for r in ColumnRole}
    tables = overrides.get("tables", {}) or {}
    stale_table_names: list[str] = []
    for tname, tval in list(tables.items()):
        tbl = sg.tables.get(tname)
        if tbl is None:
            skipped.append(OverrideSkip(path=f"tables.{tname}", reason="unknown table"))
            continue
        if not isinstance(tval, dict):
            continue
        touched_table = False
        if "description" in tval:
            desc_raw = tval["description"]
            if _override_json_null_sentinel(desc_raw):
                tval.pop("description", None)
            else:
                new_desc, want_owner = _parse_editable_description_json(
                    desc_raw,
                    f"tables.{tname}.description",
                )
                new_desc = (new_desc or "").strip()
                cur_desc = (tbl.description or "").strip()
                cur_owner_norm = (
                    tbl.description_owner if tbl.description_owner is not None else DescriptionOwner.CATALOG
                )
                desc_changed = new_desc != cur_desc
                owner_changed = want_owner != cur_owner_norm
                if desc_changed or (owner_changed and new_desc != ""):
                    path = f"tables.{tname}.description"
                    previous_text = tbl.description or ""
                    description_changes.append(
                        {
                            "path": path,
                            "kind": "table",
                            "text": new_desc,
                            "previous_text": previous_text,
                        },
                    )
                    description_targets[path] = (tname, None, want_owner)
                    touched_table = True
        if "role" in tval:
            role_raw = tval["role"]
            if _override_json_null_sentinel(role_raw):
                tval.pop("role", None)
            else:
                tr_path = f"tables.{tname}.role"
                try:
                    r_val, r_own = _parse_editable_role_json(
                        role_raw,
                        tr_path,
                        valid_table_roles,
                    )
                except ValueError as exc:
                    skipped.append(
                        OverrideSkip(
                            path=tr_path,
                            reason=str(exc),
                            code="invalid_role_override",
                        ),
                    )
                    notify(
                        f"Schema override skipped {tr_path}: {exc}",
                        stage="schema",
                        code=DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP,
                        details=(("path", tr_path), ("reason", str(exc))),
                    )
                    tval.pop("role", None)
                else:
                    cur_r_own = tbl.role_owner if tbl.role_owner is not None else RoleOwner.CATALOG
                    if tbl.role != r_val:
                        if can_overwrite_role(tbl.role_owner, r_own):
                            tbl.role = r_val
                            tbl.role_owner = r_own
                            touched_table = True
                        else:
                            skipped.append(
                                OverrideSkip(
                                    path=tr_path,
                                    reason=f"role_owner_blocked:{tbl.role_owner!s}",
                                ),
                            )
        if touched_table:
            table_edits += 1
        col_map = tval.get("columns")
        if not isinstance(col_map, dict):
            col_map = None
        if col_map is not None:
            empty_cols: list[str] = []
            for cname, cval in list(col_map.items()):
                if not isinstance(cval, dict):
                    continue
                col = tbl.columns.get(cname)
                if col is None:
                    skipped.append(
                        OverrideSkip(
                            path=f"tables.{tname}.columns.{cname}",
                            reason="unknown column",
                        )
                    )
                    continue
                touched_col = False
                if "sensitivity" in cval or "pii" in cval:
                    if "sensitivity" in cval and "pii" in cval:
                        merged = sensitivity_classification_from_legacy_fields(cval["sensitivity"], cval["pii"])
                    elif "pii" in cval:
                        merged = sensitivity_classification_from_legacy_fields("pii", cval["pii"])
                    else:
                        merged = column_sensitivity_from_dict({"sensitivity": cval["sensitivity"], "pii": None})
                    if merged != col.sensitivity:
                        set_sensitivity(col, merged)
                        touched_col = True
                if "description" in cval:
                    desc_raw = cval["description"]
                    if _override_json_null_sentinel(desc_raw):
                        cval.pop("description", None)
                    else:
                        new_desc, want_owner = _parse_editable_description_json(
                            desc_raw,
                            f"tables.{tname}.columns.{cname}.description",
                        )
                        new_desc = (new_desc or "").strip()
                        cur_desc = (col.description or "").strip()
                        cur_owner_norm = (
                            col.description_owner if col.description_owner is not None else DescriptionOwner.CATALOG
                        )
                        desc_changed = new_desc != cur_desc
                        owner_changed = want_owner != cur_owner_norm
                        if desc_changed or (owner_changed and new_desc != ""):
                            path = f"tables.{tname}.columns.{cname}.description"
                            previous_text = col.description or ""
                            if col.sensitivity != SensitivityClassification.NONE:
                                if set_description(col, new_desc, want_owner):
                                    direct_descriptions_refined += 1
                                    touched_col = True
                            else:
                                description_changes.append(
                                    {
                                        "path": path,
                                        "kind": "column",
                                        "text": new_desc,
                                        "previous_text": previous_text,
                                    },
                                )
                                description_targets[path] = (tname, cname, want_owner)
                                touched_col = True
                if "role" in cval:
                    role_raw = cval["role"]
                    if _override_json_null_sentinel(role_raw):
                        cval.pop("role", None)
                    else:
                        cr_path = f"tables.{tname}.columns.{cname}.role"
                        try:
                            r_val, r_own = _parse_editable_role_json(
                                role_raw,
                                cr_path,
                                valid_col_roles,
                                value_type=col.value_type,
                            )
                        except ValueError as exc:
                            skipped.append(
                                OverrideSkip(
                                    path=cr_path,
                                    reason=str(exc),
                                    code="invalid_role_override",
                                ),
                            )
                            notify(
                                f"Schema override skipped {cr_path}: {exc}",
                                stage="schema",
                                code=DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP,
                                details=(("path", cr_path), ("reason", str(exc))),
                            )
                            cval.pop("role", None)
                        else:
                            cur_r_own = col.role_owner if col.role_owner is not None else RoleOwner.CATALOG
                            if col.role != r_val:
                                if can_overwrite_role(col.role_owner, r_own):
                                    col.role = r_val
                                    col.role_owner = r_own
                                    touched_col = True
                                else:
                                    skipped.append(
                                        OverrideSkip(
                                            path=cr_path,
                                            reason=f"role_owner_blocked:{col.role_owner!s}",
                                        ),
                                    )
                if "boolean_truth_value" in cval:
                    raw_bt = cval["boolean_truth_value"]
                    vt = (col.value_type or "").strip().lower() or data_type_to_value_type(col.data_type).lower()
                    if vt != "boolean":
                        skipped.append(
                            OverrideSkip(
                                path=f"tables.{tname}.columns.{cname}.boolean_truth_value",
                                reason="boolean_truth_value applies only when column value_type is boolean",
                            )
                        )
                    elif raw_bt is None:
                        if col.boolean_truth_value is not None:
                            col.boolean_truth_value = None
                            touched_col = True
                    elif not isinstance(raw_bt, str) or not raw_bt.strip():
                        skipped.append(
                            OverrideSkip(
                                path=f"tables.{tname}.columns.{cname}.boolean_truth_value",
                                reason="boolean_truth_value must be null or a non-empty string",
                            )
                        )
                    else:
                        tops = {
                            str(x).strip().lower() for x in (col.top_k_values or []) if x is not None and str(x).strip()
                        }
                        allowed = set(tops)
                        if col.boolean_truth_value:
                            allowed.add(str(col.boolean_truth_value).strip().lower())
                        wl = raw_bt.strip().lower()
                        if wl not in allowed:
                            skipped.append(
                                OverrideSkip(
                                    path=f"tables.{tname}.columns.{cname}.boolean_truth_value",
                                    reason="boolean_truth_value must match an observed literal (top_k) for this column",
                                )
                            )
                        else:
                            canon = raw_bt.strip()
                            for x in col.top_k_values or []:
                                if str(x).strip().lower() == wl:
                                    canon = str(x).strip()
                                    break
                            if col.boolean_truth_value != canon:
                                col.boolean_truth_value = canon
                                touched_col = True
                if touched_col:
                    column_edits += 1
                    br_after = col.visibility_block_reason()
                    if br_after is not None:
                        skipped.append(
                            OverrideSkip(
                                path=f"tables.{tname}.columns.{cname}",
                                reason=f"column hidden ({br_after.value}); override still applied",
                                code="hidden_column_override",
                            )
                        )
                if not cval:
                    empty_cols.append(cname)
            for ec in empty_cols:
                col_map.pop(ec, None)
            if "columns" in tval and isinstance(tval["columns"], dict) and not tval["columns"]:
                tval.pop("columns", None)
        if isinstance(tval, dict) and not tval:
            stale_table_names.append(tname)
    for stn in stale_table_names:
        tables.pop(stn, None)

    refined_map: dict[str, str] = {}
    if description_changes:
        refined_map = _refine_descriptions_via_llm(description_changes)
    descriptions_refined = 0
    for path, (tname, cname, desc_owner) in description_targets.items():
        new_text = refined_map.get(path)
        if new_text is None:
            continue
        cleaned = new_text.strip()
        if cname is None:
            meta = sg.tables[tname]
        else:
            meta = sg.tables[tname].columns[cname]
        if set_description(meta, cleaned, desc_owner):
            descriptions_refined += 1
    descriptions_refined += direct_descriptions_refined

    fks_added = 0
    fks_endorsed = 0
    raw_fk_add = list(overrides.get("foreign_keys_add", []) or [])
    seen_fk_canonical: set[tuple[str, str]] = set()
    deduped_fk_add: list[tuple[int, dict[str, Any]]] = []
    for original_index, fk in enumerate(raw_fk_add):
        if not isinstance(fk, dict):
            skipped.append(
                OverrideSkip(
                    path=f"foreign_keys_add[{original_index}]",
                    reason="malformed_fk_entry",
                ),
            )
            continue
        from_str = "|".join(str(x) for x in fk["from"]) if isinstance(fk.get("from"), list) else str(fk.get("from", ""))
        to_str = "|".join(str(x) for x in fk["to"]) if isinstance(fk.get("to"), list) else str(fk.get("to", ""))
        canonical = (from_str, to_str)
        if canonical in seen_fk_canonical:
            skipped.append(
                OverrideSkip(
                    path=f"foreign_keys_add[{original_index}]",
                    reason=f"duplicate endpoint pair already added at earlier index (canonical={canonical!r})",
                )
            )
            continue
        seen_fk_canonical.add(canonical)
        deduped_fk_add.append((original_index, fk))
    for i, fk in deduped_fk_add:
        path = f"foreign_keys_add[{i}]"
        src = _split_fk_endpoint(fk.get("from"))
        dst = _split_fk_endpoint(fk.get("to"))
        if src is None or dst is None:
            skipped.append(OverrideSkip(path=path, reason="malformed 'from' or 'to' endpoint"))
            continue
        src_table, src_cols = src
        dst_table, dst_cols = dst
        if len(src_cols) != len(dst_cols):
            skipped.append(OverrideSkip(path=path, reason="from/to column counts differ"))
            continue
        if src_table not in sg.tables:
            skipped.append(OverrideSkip(path=path, reason=f"unknown source table {src_table!r}"))
            continue
        if dst_table not in sg.tables:
            skipped.append(OverrideSkip(path=path, reason=f"unknown destination table {dst_table!r}"))
            continue
        src_tbl = sg.tables[src_table]
        dst_tbl = sg.tables[dst_table]
        missing_src = [c for c in src_cols if c not in src_tbl.columns]
        if missing_src:
            skipped.append(OverrideSkip(path=path, reason=f"unknown source columns {missing_src!r}"))
            continue
        missing_dst = [c for c in dst_cols if c not in dst_tbl.columns]
        if missing_dst:
            skipped.append(OverrideSkip(path=path, reason=f"unknown destination columns {missing_dst!r}"))
            continue
        kind_raw = fk.get("kind", "structural")
        kind = "structural" if kind_raw == "logical" else kind_raw
        if kind == "semantic":
            if len(src_cols) != 1 or len(dst_cols) != 1:
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason="kind:semantic must be single-column (composite not supported)",
                    )
                )
                continue
            src_col_name = src_cols[0]
            dst_col_name = dst_cols[0]
            src_col_meta = src_tbl.columns[src_col_name]
            dst_col_meta = dst_tbl.columns[dst_col_name]
            src_vt = (src_col_meta.value_type or "").strip().lower()
            dst_vt = (dst_col_meta.value_type or "").strip().lower()
            if src_vt != "string" or dst_vt != "string":
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason=f"kind:semantic requires string columns (src={src_vt!r}, dst={dst_vt!r})",
                    )
                )
                continue
            anchor_ok = (
                src_col_meta.is_primary_key
                or dst_col_meta.is_primary_key
                or bool(src_col_meta.is_unique)
                or bool(dst_col_meta.is_unique)
            )
            if not anchor_ok:
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason="kind:semantic requires at least one endpoint to be PK or UNIQUE",
                    )
                )
                continue
            quad = (src_table, src_col_name, dst_table, dst_col_name)
            mirror = (dst_table, dst_col_name, src_table, src_col_name)
            existing_user_pairs = set(src_tbl._user_semantic_neighbors) | set(dst_tbl._user_semantic_neighbors)
            if quad in existing_user_pairs or mirror in existing_user_pairs:
                skipped.append(OverrideSkip(path=path, reason="duplicate_semantic_quad"))
                continue
            src_tbl._user_semantic_neighbors.append(quad)
            dst_tbl._user_semantic_neighbors.append(mirror)
            fks_added += 1
            continue
        edge = FKEdge(
            src_table=src_table,
            src_cols=list(src_cols),
            dst_table=dst_table,
            dst_cols=list(dst_cols),
            inference_tag=InferenceTag.USER_STRUCTURAL,
        )
        existing_by_key = {_edge_key(e): e for e in src_tbl.foreign_keys}
        ek = _edge_key(edge)
        if ek in existing_by_key:
            match_edge = existing_by_key[ek]
            if match_edge.inference_tag is None:
                match_edge.inference_tag = InferenceTag.USER_STRUCTURAL
                fks_endorsed += 1
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason=(
                            "catalog FK already present; user structural assertion recorded as "
                            "user_override_structural (endorsement)"
                        ),
                    )
                )
                continue
            skipped.append(OverrideSkip(path=path, reason="duplicate_structural_edge"))
            continue
        src_tbl.foreign_keys.append(edge)
        fks_added += 1

    fks_removed = 0
    pks_blocked = 0

    internal = overrides.setdefault("_internal", {})
    if not isinstance(internal, dict):
        internal = {}
        overrides["_internal"] = internal
    fk_block_list: list[dict[str, Any]] = list(internal.get("fk_block_inferred", []) or [])
    pk_block_list: list[dict[str, Any]] = list(internal.get("pk_block_inferred", []) or [])

    def _endpoint_to_str(ep: Any) -> str:
        if isinstance(ep, str):
            return ep
        if isinstance(ep, list):
            return "|".join(str(x) for x in ep)
        return str(ep)

    fk_block_seen: set[tuple[str, str]] = {
        (_endpoint_to_str(e.get("from")), _endpoint_to_str(e.get("to"))) for e in fk_block_list
    }
    pk_block_seen: set[tuple[str, str]] = {(str(e.get("table", "")), str(e.get("column", ""))) for e in pk_block_list}

    pk_block_pre_sig = frozenset(
        (str(e.get("table", "")), str(e.get("column", ""))) for e in pk_block_list if isinstance(e, dict)
    )
    fk_block_pre_sig = frozenset(
        (_endpoint_to_str(e.get("from")), _endpoint_to_str(e.get("to"))) for e in fk_block_list if isinstance(e, dict)
    )

    for i, fk in enumerate(overrides.get("foreign_keys_remove", []) or []):
        path = f"foreign_keys_remove[{i}]"
        src = _split_fk_endpoint(fk.get("from"))
        dst = _split_fk_endpoint(fk.get("to"))
        if src is None or dst is None:
            skipped.append(OverrideSkip(path=path, reason="malformed 'from' or 'to' endpoint"))
            continue
        src_table, src_cols = src
        dst_table, dst_cols = dst
        if src_table not in sg.tables:
            skipped.append(OverrideSkip(path=path, reason=f"unknown source table {src_table!r}"))
            continue
        if dst_table not in sg.tables:
            skipped.append(OverrideSkip(path=path, reason=f"unknown destination table {dst_table!r}"))
            continue
        src_tbl = sg.tables[src_table]
        target = FKEdge(
            src_table=src_table,
            src_cols=list(src_cols),
            dst_table=dst_table,
            dst_cols=list(dst_cols),
        )
        target_key = _edge_key(target)
        match_idx = -1
        match_edge: FKEdge | None = None
        for j, existing_edge in enumerate(src_tbl.foreign_keys):
            if _edge_key(existing_edge) == target_key:
                match_idx = j
                match_edge = existing_edge
                break
        if match_idx < 0 or match_edge is None:
            skipped.append(OverrideSkip(path=path, reason="no matching foreign key on source table"))
            continue
        if match_edge.inference_tag is None:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog FK (inference_tag is null) cannot be removed via overrides",
                )
            )
            continue
        del src_tbl.foreign_keys[match_idx]
        fks_removed += 1
        if not (match_edge.inference_tag or "").startswith("user_override_"):
            entry = {"from": fk.get("from"), "to": fk.get("to")}
            key = (_endpoint_to_str(entry["from"]), _endpoint_to_str(entry["to"]))
            if key not in fk_block_seen:
                fk_block_list.append(entry)
                fk_block_seen.add(key)

    for i, entry in enumerate(list(fk_block_list)):
        path = f"_internal.fk_block_inferred[{i}]"
        src = _split_fk_endpoint(entry.get("from"))
        dst = _split_fk_endpoint(entry.get("to"))
        if src is None or dst is None:
            skipped.append(OverrideSkip(path=path, reason="malformed 'from' or 'to' endpoint; entry dropped"))
            try:
                fk_block_list.remove(entry)
            except ValueError:
                pass
            continue
        src_table, src_cols = src
        dst_table, dst_cols = dst
        if src_table not in sg.tables:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=f"unknown source table {src_table!r}; entry dropped",
                )
            )
            try:
                fk_block_list.remove(entry)
            except ValueError:
                pass
            continue
        if dst_table not in sg.tables:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=f"unknown destination table {dst_table!r}; entry dropped",
                )
            )
            try:
                fk_block_list.remove(entry)
            except ValueError:
                pass
            continue
        src_tbl = sg.tables[src_table]
        dst_tbl = sg.tables[dst_table]
        missing_src = [c for c in src_cols if c not in src_tbl.columns]
        missing_dst = [c for c in dst_cols if c not in dst_tbl.columns]
        if missing_src or missing_dst:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=(
                        "missing endpoint columns "
                        f"(src_missing={missing_src!r}, dst_missing={missing_dst!r}); entry dropped"
                    ),
                )
            )
            try:
                fk_block_list.remove(entry)
            except ValueError:
                pass
            continue
        target_key = _edge_key(
            FKEdge(
                src_table=src_table,
                src_cols=list(src_cols),
                dst_table=dst_table,
                dst_cols=list(dst_cols),
            )
        )
        match_idx = -1
        match_edge = None
        for j, existing_edge in enumerate(src_tbl.foreign_keys):
            if _edge_key(existing_edge) == target_key:
                match_idx = j
                match_edge = existing_edge
                break
        if match_idx < 0 or match_edge is None:
            skipped.append(OverrideSkip(path=path, reason="fk_block_no_match"))
            continue
        if match_edge.inference_tag is None:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog FK cannot be blocked (block applies only to inferred edges)",
                )
            )
            continue
        if (match_edge.inference_tag or "").startswith("user_override_"):
            skipped.append(OverrideSkip(path=path, reason="fk_block_user_override_edge"))
            continue
        del src_tbl.foreign_keys[match_idx]
        fks_removed += 1

    pks_added = 0
    pks_endorsed = 0
    for i, entry in enumerate(overrides.get("primary_keys_add", []) or []):
        path = f"primary_keys_add[{i}]"
        tname = entry.get("table")
        cname = entry.get("column")
        if not isinstance(tname, str) or not isinstance(cname, str):
            skipped.append(OverrideSkip(path=path, reason="malformed 'table' or 'column'"))
            continue
        tbl = sg.tables.get(tname)
        if tbl is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown table {tname!r}"))
            continue
        col = tbl.columns.get(cname)
        if col is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown column {tname}.{cname}"))
            continue
        if col.is_primary_key:
            if col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE:
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason="pk_already_user_override",
                    ),
                )
                continue
            col.pk_inference_tag = PkInferenceTag.USER_OVERRIDE
            pks_endorsed += 1
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=(
                        f"column {tname}.{cname} already in primary key; "
                        "user assertion recorded as user_override (endorsement)"
                    ),
                )
            )
            continue
        null_ratio = float(col.null_ratio or 0.0)
        if null_ratio > 0.0:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=f"column {tname}.{cname} has null_ratio={null_ratio} > 0; cannot be a primary key",
                )
            )
            continue
        rc = int(col.row_count or 0)
        dc = int(col.distinct_count or 0)
        unique_ok = bool(col.is_unique) or (rc > 0 and dc == rc and not col.distinct_from_sample)
        if not unique_ok:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=(
                        f"column {tname}.{cname} not unique "
                        f"(row_count={rc}, distinct_count={dc}, is_unique={col.is_unique})"
                    ),
                )
            )
            continue
        col.pk_inference_tag = PkInferenceTag.USER_OVERRIDE
        if cname not in tbl.primary_key:
            tbl.primary_key.append(cname)
        pks_added += 1

    for i, entry in enumerate(overrides.get("primary_keys_remove", []) or []):
        path = f"primary_keys_remove[{i}]"
        tname = entry.get("table")
        cname = entry.get("column")
        if not isinstance(tname, str) or not isinstance(cname, str):
            skipped.append(OverrideSkip(path=path, reason="malformed 'table' or 'column'"))
            continue
        tbl = sg.tables.get(tname)
        if tbl is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown table {tname!r}"))
            continue
        col = tbl.columns.get(cname)
        if col is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown column {tname}.{cname}"))
            continue
        if col.pk_inference_tag is None and col.is_primary_key:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog PK cannot be removed (only inferred PKs are removable)",
                )
            )
            continue
        if not col.is_primary_key:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="not_primary_key",
                ),
            )
            continue
        remaining_pk = [c for c in tbl.primary_key if c != cname]
        if not remaining_pk:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason=(
                        f"removing {tname}.{cname} would leave table {tname!r} with an empty primary key; "
                        "supply a replacement column via 'primary_keys_add' for the same table"
                    ),
                )
            )
            continue
        col.pk_inference_tag = None
        if cname in tbl.primary_key:
            tbl.primary_key.remove(cname)
        pks_blocked += 1
        key = (tname, cname)
        if key not in pk_block_seen:
            pk_block_list.append({"table": tname, "column": cname})
            pk_block_seen.add(key)

    for i, entry in enumerate(list(pk_block_list)):
        path = f"_internal.pk_block_inferred[{i}]"
        tname = entry.get("table")
        cname = entry.get("column")
        if not isinstance(tname, str) or not isinstance(cname, str):
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="malformed_pk_block: expected string table and column",
                ),
            )
            continue
        tbl = sg.tables.get(tname)
        if tbl is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown table {tname!r}"))
            continue
        col = tbl.columns.get(cname)
        if col is None:
            skipped.append(OverrideSkip(path=path, reason=f"unknown column {tname}.{cname}"))
            continue
        if col.pk_inference_tag is None and col.is_primary_key:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog PK cannot be blocked (block applies only to inferred PKs)",
                )
            )
            continue
        if not col.is_primary_key:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="pk_block_not_pk",
                ),
            )
            continue
        col.pk_inference_tag = None
        if cname in tbl.primary_key:
            tbl.primary_key.remove(cname)
        pks_blocked += 1

    if pks_added > 0 or pks_blocked > 0:
        fk_blocked_keys: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
        for blk_entry in fk_block_list:
            src_ep = _split_fk_endpoint(blk_entry.get("from"))
            dst_ep = _split_fk_endpoint(blk_entry.get("to"))
            if src_ep is None or dst_ep is None:
                continue
            st, scols = src_ep
            dt, dcols = dst_ep
            fk_blocked_keys.add((st, tuple(scols), dt, tuple(dcols)))
        reinferred = _pair_targeted_fk_inference(sg, blocked=frozenset(fk_blocked_keys))
        fks_added += reinferred

    internal["fk_block_inferred"] = fk_block_list
    internal["pk_block_inferred"] = pk_block_list

    replay_user_semantic_neighbors_to_columns(sg)

    collapsed = _collapse_redundant_inferences(sg, skipped)

    coerced = _coerce_pk_fk_columns_to_identifier(sg)
    coerced_n = len(coerced)

    if (
        fks_added
        or fks_endorsed
        or fks_removed
        or pks_added
        or pks_endorsed
        or pks_blocked
        or table_edits
        or column_edits
        or coerced_n
        or collapsed
    ):
        sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
        assign_schema_graph_hashes(sg, _schema_context_from_graph(sg), sg.notes_sha256)
        sg.refresh_schema_stats()
        sg.schema_revision = int(getattr(sg, "schema_revision", 0)) + 1

    pk_block_post_sig = frozenset(
        (str(e.get("table", "")), str(e.get("column", ""))) for e in pk_block_list if isinstance(e, dict)
    )
    fk_block_post_sig = frozenset(
        (_endpoint_to_str(e.get("from")), _endpoint_to_str(e.get("to"))) for e in fk_block_list if isinstance(e, dict)
    )
    changed_pk_blocks = pk_block_pre_sig != pk_block_post_sig
    changed_fk_blocks = fk_block_pre_sig != fk_block_post_sig

    return OverrideReport(
        table_edits=table_edits,
        column_edits=column_edits,
        fks_added=fks_added,
        fks_endorsed=fks_endorsed,
        fks_removed=fks_removed,
        pks_added=pks_added,
        pks_endorsed=pks_endorsed,
        pks_blocked=pks_blocked,
        changed_pk_blocks=changed_pk_blocks,
        changed_fk_blocks=changed_fk_blocks,
        coerced_columns=coerced_n,
        collapsed_inferences=collapsed,
        descriptions_refined=descriptions_refined,
        skipped=tuple(skipped),
    )


def _overrides_sidecar_path(schema_json_path: str | Path) -> Path:
    """
    Return the canonical sidecar location for *schema_json_path*'s overrides document.

    Sidecar lives next to the gzip schema cache so artifact directory cleanup removes both atomically. The filename is fixed by ``SCHEMA_OVERRIDES_SIDECAR_FILENAME`` so a single context only ever has one sidecar.
    """

    return Path(schema_json_path).expanduser().resolve().parent / SCHEMA_OVERRIDES_SIDECAR_FILENAME


def load_overrides_sidecar(schema_json_path: str | Path) -> dict[str, Any] | None:
    """
    Read the persisted overrides sidecar; return ``None`` when missing or unreadable.

    A corrupt sidecar is logged and treated as missing so a single bad write never blocks schema rebuilds. The returned document is structurally validated against the current ``SCHEMA_OVERRIDES_VERSION``; mismatched versions also yield ``None`` (and a debug entry) so a future bump can recover by re-applying user input rather than crashing.
    """

    path = _overrides_sidecar_path(schema_json_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        meta_source_hash = d.pop("source_schema_hash", None)
        meta_applied_at = d.pop("applied_at", None)
        meta_metadata_hash = d.pop("metadata_hash", None)
        validated = _validate_overrides_structure(d)
        if meta_source_hash is not None:
            validated["source_schema_hash"] = meta_source_hash
        if meta_applied_at is not None:
            validated["applied_at"] = meta_applied_at
        if meta_metadata_hash is not None:
            validated["metadata_hash"] = meta_metadata_hash
        return validated
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        debug(f"[schema.load_overrides_sidecar] ignoring corrupt sidecar {path!s}: {exc!r}")
        return None


def _migrate_sidecar_for_diff(schema_json_path: str | Path, diff: SchemaDiff) -> bool:
    """
    Rewrite the persisted overrides sidecar in place so user-edited entries survive table/column renames.

    Loads the sidecar, applies *diff*'s table renames to top-level ``tables`` keys and any FK endpoints in ``foreign_keys_add`` / ``_internal.fk_block_inferred`` / ``_internal.pk_block_inferred``, then applies each ``per_table`` column rename to the column-keyed ``tables[<name>].columns`` map and to FK ``from`` / ``to`` endpoints that reference the renamed pair. Saves only when something actually changed. Returns True when a write occurred.
    """

    sidecar = load_overrides_sidecar(schema_json_path)
    if sidecar is None:
        return False
    table_renames: dict[str, str] = {old: new for old, new in diff.table_renames if old != new}
    column_renames_by_new_table: dict[str, dict[str, str]] = {}
    for tname, td in diff.per_table.items():
        col_renames = {old: new for old, new in td.renamed_columns if old != new}
        if col_renames:
            column_renames_by_new_table[tname] = col_renames
    if not table_renames and not column_renames_by_new_table:
        return False
    changed = False
    tables_block = sidecar.get("tables") or {}
    if isinstance(tables_block, dict) and table_renames:
        new_tables_block: dict[str, Any] = {}
        for tname, tval in tables_block.items():
            new_name = table_renames.get(tname, tname)
            if new_name != tname:
                changed = True
            new_tables_block[new_name] = tval
        sidecar["tables"] = new_tables_block
        tables_block = new_tables_block
    if isinstance(tables_block, dict):
        for tname, tval in list(tables_block.items()):
            if not isinstance(tval, dict):
                continue
            cols_block = tval.get("columns")
            if not isinstance(cols_block, dict):
                continue
            col_renames = column_renames_by_new_table.get(tname) or {}
            if not col_renames:
                continue
            new_cols_block: dict[str, Any] = {}
            for cname, cval in cols_block.items():
                new_cname = col_renames.get(cname, cname)
                if new_cname != cname:
                    changed = True
                new_cols_block[new_cname] = cval
            tval["columns"] = new_cols_block

    def _remap_endpoint(endpoint: Any) -> tuple[Any, bool]:
        if not isinstance(endpoint, str) or "." not in endpoint:
            return endpoint, False
        tbl, _, rest = endpoint.partition(".")
        new_tbl = table_renames.get(tbl, tbl)
        col_renames = column_renames_by_new_table.get(new_tbl) or {}
        cols = rest.split(",") if "," in rest else [rest]
        new_cols = [col_renames.get(c, c) for c in cols]
        rebuilt = f"{new_tbl}." + ",".join(new_cols)
        return rebuilt, rebuilt != endpoint

    fk_add = sidecar.get("foreign_keys_add") or []
    if isinstance(fk_add, list):
        for entry in fk_add:
            if not isinstance(entry, dict):
                continue
            for key in ("from", "to"):
                if key in entry:
                    new_val, did_change = _remap_endpoint(entry[key])
                    if did_change:
                        entry[key] = new_val
                        changed = True
    pk_add_top = sidecar.get("primary_keys_add") or []
    if isinstance(pk_add_top, list):
        for entry in pk_add_top:
            if not isinstance(entry, dict):
                continue
            tbl = str(entry.get("table", ""))
            col = str(entry.get("column", ""))
            new_tbl = table_renames.get(tbl, tbl)
            col_renames = column_renames_by_new_table.get(new_tbl) or {}
            new_col = col_renames.get(col, col)
            if new_tbl != tbl or new_col != col:
                entry["table"] = new_tbl
                entry["column"] = new_col
                changed = True
    internal = sidecar.get("_internal") or {}
    if isinstance(internal, dict):
        for blk_key in ("fk_block_inferred",):
            blk = internal.get(blk_key) or []
            if not isinstance(blk, list):
                continue
            for entry in blk:
                if not isinstance(entry, dict):
                    continue
                for key in ("from", "to"):
                    if key in entry:
                        new_val, did_change = _remap_endpoint(entry[key])
                        if did_change:
                            entry[key] = new_val
                            changed = True
        pk_blk = internal.get("pk_block_inferred") or []
        if isinstance(pk_blk, list):
            for entry in pk_blk:
                if not isinstance(entry, dict):
                    continue
                tbl = str(entry.get("table", ""))
                col = str(entry.get("column", ""))
                new_tbl = table_renames.get(tbl, tbl)
                col_renames = column_renames_by_new_table.get(new_tbl) or {}
                new_col = col_renames.get(col, col)
                if new_tbl != tbl or new_col != col:
                    entry["table"] = new_tbl
                    entry["column"] = new_col
                    changed = True
    if not changed:
        return False
    source_hash = sidecar.get("source_schema_hash") or ""
    resolved_cache = str(Path(schema_json_path).expanduser().resolve())
    snap = load_schema_graph_snapshot(resolved_cache)
    meta_hash = compute_metadata_hash(snap) if snap is not None else str(sidecar.get("metadata_hash") or "")
    save_overrides_sidecar(
        schema_json_path,
        sidecar,
        source_schema_hash=source_hash,
        metadata_hash=meta_hash,
    )
    return True


def _snapshot_schema_graph(sg: SchemaGraph) -> SchemaGraph:
    """Return a deep copy of *sg* for rolling back failed in-place mutations."""

    return copy.deepcopy(sg)


def _restore_schema_graph_inplace(target: SchemaGraph, snapshot: SchemaGraph) -> None:
    """Replace *target* state with a copy of *snapshot* and re-wire table and column owners."""

    target.tables = copy.deepcopy(snapshot.tables)
    target.join_paths_multi = copy.deepcopy(snapshot.join_paths_multi)
    target.structural_hash = snapshot.structural_hash
    target.profiling_hash = snapshot.profiling_hash
    target.scope_hash = snapshot.scope_hash
    target.effective_structural_hash = snapshot.effective_structural_hash
    target.notes_hash = snapshot.notes_hash
    target.semantic_edges_hash = snapshot.semantic_edges_hash
    target.ddl_probe_hash = snapshot.ddl_probe_hash
    target.include = snapshot.include
    target.created_at = snapshot.created_at
    target.enum_values = copy.deepcopy(snapshot.enum_values) if snapshot.enum_values else None
    target.schema_stats = copy.deepcopy(snapshot.schema_stats) if snapshot.schema_stats else None
    target.deny_columns = {k: set(v) for k, v in snapshot.deny_columns.items()}
    target.disallowed_columns = {k: set(v) for k, v in snapshot.disallowed_columns.items()}
    target.notes_sha256 = snapshot.notes_sha256
    target.scope_descriptor = copy.deepcopy(snapshot.scope_descriptor) if snapshot.scope_descriptor else None
    target.schema_revision = snapshot.schema_revision
    target._stats_dirty = snapshot._stats_dirty
    object.__setattr__(target, "_canonical_bearers", dict(getattr(snapshot, "_canonical_bearers", {})))
    for tbl in target.tables.values():
        object.__setattr__(tbl, "_owner_graph", target)
        for col in tbl.columns.values():
            col._owner_table = tbl


def _sidecar_fk_endpoints_exist(sg: SchemaGraph, entry: dict[str, Any]) -> bool:
    """Return True when both FK endpoints reference existing tables and columns on *sg*."""

    src = _split_fk_endpoint(entry.get("from"))
    dst = _split_fk_endpoint(entry.get("to"))
    if src is None or dst is None:
        return False
    st, scols = src
    dt, dcols = dst
    if st not in sg.tables or dt not in sg.tables:
        return False
    stbl = sg.tables[st]
    dtbl = sg.tables[dt]
    return all(c in stbl.columns for c in scols) and all(c in dtbl.columns for c in dcols)


def _reconcile_sidecar_against_graph(sg: SchemaGraph, schema_json_path: str | Path) -> SidecarReconcileReport:
    """Drop persisted overrides whose endpoints are missing from *sg*; rewrite the sidecar when needed."""

    sidecar = load_overrides_sidecar(schema_json_path)
    if sidecar is None:
        return SidecarReconcileReport(pruned_paths=(), wrote_disk=False)
    pruned: list[str] = []
    src_hash = str(sidecar.get("source_schema_hash", "") or "")
    tables_block = sidecar.get("tables")
    if isinstance(tables_block, dict):
        for tname in list(tables_block.keys()):
            if tname not in sg.tables:
                pruned.append(f"tables.{tname}")
                del tables_block[tname]
                continue
            tval = tables_block[tname]
            if not isinstance(tval, dict):
                continue
            cols = tval.get("columns")
            if not isinstance(cols, dict):
                continue
            tbl = sg.tables[tname]
            for cname in list(cols.keys()):
                if cname not in tbl.columns:
                    pruned.append(f"tables.{tname}.columns.{cname}")
                    del cols[cname]
    fk_add = sidecar.get("foreign_keys_add")
    if isinstance(fk_add, list):
        indices_to_drop: list[int] = []
        for i, entry in enumerate(fk_add):
            if not isinstance(entry, dict):
                indices_to_drop.append(i)
                pruned.append(f"foreign_keys_add[{i}]")
                continue
            if not _sidecar_fk_endpoints_exist(sg, entry):
                indices_to_drop.append(i)
                pruned.append(f"foreign_keys_add[{i}]")
        for j in reversed(indices_to_drop):
            fk_add.pop(j)
    pk_add = sidecar.get("primary_keys_add")
    if isinstance(pk_add, list):
        pk_drop: list[int] = []
        for i, entry in enumerate(pk_add):
            if not isinstance(entry, dict):
                pk_drop.append(i)
                pruned.append(f"primary_keys_add[{i}]")
                continue
            tnm = str(entry.get("table", ""))
            cnm = str(entry.get("column", ""))
            if tnm not in sg.tables or cnm not in sg.tables[tnm].columns:
                pk_drop.append(i)
                pruned.append(f"primary_keys_add[{i}]")
        for j in reversed(pk_drop):
            pk_add.pop(j)
    internal = sidecar.get("_internal")
    if isinstance(internal, dict):
        fk_blk = internal.get("fk_block_inferred")
        if isinstance(fk_blk, list):
            to_drop_fk: list[int] = []
            for i, entry in enumerate(fk_blk):
                if not isinstance(entry, dict):
                    to_drop_fk.append(i)
                    pruned.append(f"_internal.fk_block_inferred[{i}]")
                    continue
                if not _sidecar_fk_endpoints_exist(sg, entry):
                    to_drop_fk.append(i)
                    pruned.append(f"_internal.fk_block_inferred[{i}]")
            for j in reversed(to_drop_fk):
                fk_blk.pop(j)
        pk_blk = internal.get("pk_block_inferred")
        if isinstance(pk_blk, list):
            to_drop_pk: list[int] = []
            for i, entry in enumerate(pk_blk):
                if not isinstance(entry, dict):
                    to_drop_pk.append(i)
                    pruned.append(f"_internal.pk_block_inferred[{i}]")
                    continue
                tnm = str(entry.get("table", ""))
                cnm = str(entry.get("column", ""))
                if tnm not in sg.tables or cnm not in sg.tables[tnm].columns:
                    to_drop_pk.append(i)
                    pruned.append(f"_internal.pk_block_inferred[{i}]")
            for j in reversed(to_drop_pk):
                pk_blk.pop(j)
    if not pruned:
        return SidecarReconcileReport(pruned_paths=(), wrote_disk=False)
    save_overrides_sidecar(
        schema_json_path,
        sidecar,
        source_schema_hash=src_hash,
        metadata_hash=compute_metadata_hash(sg),
    )
    return SidecarReconcileReport(pruned_paths=tuple(sorted(pruned)), wrote_disk=True)


def destructive_migration_execute(artifacts_dir: str, schema: SchemaGraph) -> None:
    """Clear template and simulation caches and stamp the manifest after a learning-reset tier decision."""

    debug("[schema.destructive_migration_execute] clearing template and simulation caches")
    wipe_versioned_artifacts(artifacts_dir)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=schema.structural_hash,
        profiling_hash=schema.profiling_hash,
        scope_hash=schema.scope_hash,
        effective_structural_hash=schema.effective_structural_hash,
        notes_hash=schema.notes_hash,
        semantic_edges_hash=schema.semantic_edges_hash,
        last_migration_tier=MigrationTier.DESTRUCTIVE.value,
        last_action="destructive",
    )
    sidecar_report = _reconcile_sidecar_against_graph(schema, EngineConfig.SCHEMA_JSON_PATH)
    if sidecar_report.pruned_paths:
        notify(
            f"  Overrides sidecar pruned {len(sidecar_report.pruned_paths)} stale "
            f"entr{'y' if len(sidecar_report.pruned_paths) == 1 else 'ies'} after learning reset migration.",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("pruned_count", str(len(sidecar_report.pruned_paths))),),
        )


def save_overrides_sidecar(
    schema_json_path: str | Path,
    doc: dict[str, Any],
    *,
    source_schema_hash: str,
    metadata_hash: str,
) -> Path:
    """
    Atomically write *doc* as the persisted overrides sidecar for *schema_json_path*.

    The sidecar represents the *resolved* state: it carries the editable surface (``tables``, ``foreign_keys_add``) and the system-managed ``_internal`` envelope (``fk_block_inferred``, ``pk_block_inferred``) used to suppress re-inference on rebuilds. Transient input lists (``foreign_keys_remove``, ``primary_keys_remove``) are not persisted. Stamps ``source_schema_hash``, ``metadata_hash``, and ``applied_at`` so a later rebuild can decide whether to replay.
    """

    path = _overrides_sidecar_path(schema_json_path)
    adir = os.path.dirname(os.path.abspath(str(path)))
    with artifact_lock(adir):
        path.parent.mkdir(parents=True, exist_ok=True)
        internal = doc.get("_internal", {}) or {}
        payload: dict[str, Any] = {
            "version": SCHEMA_OVERRIDES_VERSION,
            "source_schema_hash": source_schema_hash,
            "metadata_hash": metadata_hash,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "tables": doc.get("tables", {}) or {},
            "foreign_keys_add": doc.get("foreign_keys_add", []) or [],
            "primary_keys_add": doc.get("primary_keys_add", []) or [],
            "_internal": {
                "fk_block_inferred": list(internal.get("fk_block_inferred", []) or []),
                "pk_block_inferred": list(internal.get("pk_block_inferred", []) or []),
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return path


def clear_persisted_overrides(schema_json_path: str | Path) -> bool:
    """
    Delete the overrides sidecar and the schema cache for *schema_json_path*; returns True when the sidecar existed.

    Removing the schema cache forces the next rebuild to perform a full reflect, profile, and infer pass so user-added FKs, user PK overrides, and the persisted ``_internal`` block lists are all dropped from the resulting graph. The caller is responsible for triggering that rebuild (``Text2SQL.clear_persisted_overrides`` re-runs ``initialize_text2sql`` immediately).
    """

    sidecar_path = _overrides_sidecar_path(schema_json_path)
    cache_path = Path(schema_json_path).expanduser().resolve()
    sidecar_existed = sidecar_path.is_file()
    if sidecar_existed:
        try:
            sidecar_path.unlink()
        except OSError as exc:
            debug(f"[schema.clear_persisted_overrides] failed to remove {sidecar_path!s}: {exc!r}")
            sidecar_existed = False
    if cache_path.is_file():
        try:
            cache_path.unlink()
        except OSError as exc:
            debug(f"[schema.clear_persisted_overrides] failed to remove {cache_path!s}: {exc!r}")
    return sidecar_existed


def _load_inference_block_lists(
    schema_json_path: str | Path | None,
) -> tuple[
    frozenset[tuple[str, str]],
    frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
]:
    """
    Read the overrides sidecar and return parsed PK/FK inference block sets.

    The sidecar's ``_internal.pk_block_inferred`` envelope is normalized to ``(table, column)`` pairs and ``_internal.fk_block_inferred`` is normalized to canonical edge keys ``(src_table, tuple(src_cols), dst_table, tuple(dst_cols))`` so :func:`_infer_missing_pks_from_profile` and :func:`_infer_missing_fks` can drop blocked candidates before they are re-emitted. Returns two empty frozensets when *schema_json_path* is ``None``, when the sidecar file is absent, or when the envelope is missing or malformed; the post-hoc removal in :func:`apply_schema_overrides_to_graph` remains the safety net for entries that survive any pre-filter gap.
    """
    if schema_json_path is None:
        return frozenset(), frozenset()
    sidecar = load_overrides_sidecar(schema_json_path)
    if sidecar is None:
        return frozenset(), frozenset()
    internal = sidecar.get("_internal", {}) or {}
    pk_pairs: set[tuple[str, str]] = set()
    for entry in internal.get("pk_block_inferred", []) or []:
        if not isinstance(entry, dict):
            continue
        table = str(entry.get("table", "") or "")
        column = str(entry.get("column", "") or "")
        if table and column:
            pk_pairs.add((table, column))
    fk_keys: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for entry in internal.get("fk_block_inferred", []) or []:
        if not isinstance(entry, dict):
            continue
        src = _split_fk_endpoint(entry.get("from"))
        dst = _split_fk_endpoint(entry.get("to"))
        if src is None or dst is None:
            continue
        fk_keys.add((src[0], tuple(src[1]), dst[0], tuple(dst[1])))
    return frozenset(pk_pairs), frozenset(fk_keys)


def _finalize_with_overrides(
    sg: SchemaGraph,
    schema_json_path: str | Path,
    *,
    dialect: Any | None = None,
) -> bool:
    """
    Replay the persisted overrides sidecar onto *sg* if one exists.

    Called at the tail of every ``build_schema_graph_with_diff`` branch so user-applied metadata, user-added FKs, and inference block lists survive a cache hit, a notes-only refresh, a scope-subset filter, a partial rebuild, and a full rebuild alike. Replay is idempotent so it always runs when a sidecar exists; the sidecar's ``source_schema_hash`` and ``metadata_hash`` are then refreshed to the freshly stamped ``effective_structural_hash`` and :func:`compute_metadata_hash` output. Skipped override entries (unknown tables/columns, malformed FKs, etc.) are emitted via :func:`_core_utils.notify` and stashed on ``sg._last_overrides_skipped`` for programmatic inspection. Returns True iff a replay actually executed.
    """

    sidecar = load_overrides_sidecar(schema_json_path)
    if sidecar is None:
        sg._last_overrides_skipped = ()
        return False
    stored_hash = sidecar.get("source_schema_hash")
    stored_meta = sidecar.get("metadata_hash")
    curr_meta = compute_metadata_hash(sg)
    if (
        isinstance(stored_hash, str)
        and stored_hash == sg.effective_structural_hash
        and isinstance(stored_meta, str)
        and stored_meta == curr_meta
    ):
        sg._last_overrides_skipped = ()
        return False
    debug(f"[schema.finalize_with_overrides] replaying sidecar (curr_hash={sg.effective_structural_hash[:16]!r})")
    document: dict[str, Any] = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": sidecar.get("tables", {}) or {},
        "foreign_keys_add": sidecar.get("foreign_keys_add", []) or [],
        "foreign_keys_remove": [],
        "primary_keys_add": sidecar.get("primary_keys_add", []) or [],
        "primary_keys_remove": [],
        "_internal": {
            "fk_block_inferred": list((sidecar.get("_internal", {}) or {}).get("fk_block_inferred", []) or []),
            "pk_block_inferred": list((sidecar.get("_internal", {}) or {}).get("pk_block_inferred", []) or []),
        },
    }
    report = apply_schema_overrides_to_graph(sg, document)
    sg._last_overrides_skipped = report.skipped
    if report.skipped:
        _core_utils.notify(
            f"Schema overrides replay skipped {len(report.skipped)} entr{'y' if len(report.skipped) == 1 else 'ies'}:",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("skipped_count", str(len(report.skipped))),),
        )
        for entry in report.skipped:
            _core_utils.notify(
                f"  - {entry.path}: {entry.reason}",
                stage="schema",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                details=(("path", entry.path), ("reason", entry.reason)),
            )
    if report.changed_pk_blocks or report.changed_fk_blocks:
        pk_blocked, fk_blocked = _load_inference_block_lists(schema_json_path)
        if report.changed_pk_blocks:
            _infer_missing_pks_from_profile(sg.tables, dialect=dialect, blocked=pk_blocked)
        if report.changed_fk_blocks:
            _pair_targeted_fk_inference(sg, blocked=fk_blocked)
        _coerce_pk_fk_columns_to_identifier(sg)
        sg.join_paths_multi = _recompute_join_paths_multi(sg.tables)
        sg.refresh_schema_stats()
        assign_schema_graph_hashes(sg, _schema_context_from_graph(sg), sg.notes_sha256)
    document["foreign_keys_add"] = _user_added_fks_dump(sg)
    document["primary_keys_add"] = _user_added_pks_dump(sg)
    adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
    with artifact_lock(adir):
        save_overrides_sidecar(
            schema_json_path,
            document,
            source_schema_hash=sg.effective_structural_hash,
            metadata_hash=compute_metadata_hash(sg),
        )
        _save_schema_to_cache(sg, str(schema_json_path))
    return True


def apply_overrides_and_persist(
    sg: SchemaGraph,
    overrides_path: str | Path,
    *,
    schema_json_path: str,
) -> OverrideReport:
    """
    Load overrides from *overrides_path*, apply them to *sg*, persist the schema cache, and update the sidecar.

    The sidecar at ``_overrides_sidecar_path(schema_json_path)`` is rewritten with the *resolved* state (existing user-added FKs surfaced from the in-memory graph plus the merged ``_internal`` block lists) so the next rebuild can replay without re-reading the user's editor file. The ``source_schema_hash`` and ``metadata_hash`` fields are stamped from the freshly stamped ``effective_structural_hash`` and :func:`compute_metadata_hash` so subsequent cache hits can short-circuit replay when neither structure nor metadata drifted. Description refinement always runs through the LLM batch helper (which falls through cleanly when credentials are absent). Cache and sidecar writes share a single ``artifact_lock`` critical section.
    """

    document = load_schema_overrides_file(overrides_path)
    sidecar = load_overrides_sidecar(schema_json_path) or {}
    sidecar_internal = sidecar.get("_internal", {}) or {}
    document_internal = document.setdefault("_internal", {})
    if not isinstance(document_internal, dict):
        document_internal = {}
        document["_internal"] = document_internal
    for key in ("fk_block_inferred", "pk_block_inferred"):
        existing_entries = list(sidecar_internal.get(key, []) or [])
        incoming_entries = list(document_internal.get(key, []) or [])
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for entry in existing_entries + incoming_entries:
            if not isinstance(entry, dict):
                continue
            if key == "pk_block_inferred":
                k = (str(entry.get("table", "")), str(entry.get("column", "")))
            else:
                k = (
                    (str(entry.get("from")) if not isinstance(entry.get("from"), list) else "|".join(entry["from"])),
                    (str(entry.get("to")) if not isinstance(entry.get("to"), list) else "|".join(entry["to"])),
                )
            if k in seen:
                continue
            seen.add(k)
            combined.append(entry)
        document_internal[key] = combined

    merged_pk: list[dict[str, Any]] = []
    seen_pk: set[tuple[str, str]] = set()
    for entry in list(sidecar.get("primary_keys_add", []) or []) + list(document.get("primary_keys_add", []) or []):
        if not isinstance(entry, dict):
            continue
        tnm = str(entry.get("table", ""))
        cnm = str(entry.get("column", ""))
        if not tnm or not cnm:
            continue
        key = (tnm, cnm)
        if key in seen_pk:
            continue
        seen_pk.add(key)
        merged_pk.append({"table": tnm, "column": cnm})
    document["primary_keys_add"] = merged_pk

    report = apply_schema_overrides_to_graph(sg, document)
    document["foreign_keys_add"] = _user_added_fks_dump(sg)
    document["primary_keys_add"] = _user_added_pks_dump(sg)

    adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
    with artifact_lock(adir):
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
            _save_schema_to_cache(sg, str(schema_json_path))
        save_overrides_sidecar(
            schema_json_path,
            document,
            source_schema_hash=sg.effective_structural_hash,
            metadata_hash=compute_metadata_hash(sg),
        )
    notify_schema_path_health(sg)
    return report


set_schema_helpers(compute_schema_stats, compute_database_feature_capability)
