"""Schema graph lifecycle: snapshots, diff, scope, FK inference, join paths, and hashes."""

from __future__ import annotations

import copy
import glob
import gzip
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any, Literal
from uuid import uuid4

import sqlglot
from sqlalchemy import text

from ._config import (
    ConfigError,
    PolicyConfig,
)
from ._constants import (
    COMPATIBLE_TYPE_PAIRS,
    DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT,
    FK_INFERENCE_SUFFIX_STEMS,
    INFERRED_PK_VALUE_TYPES,
    INTEGER_VALUE_TYPES,
    MIGRATION_DATA_OVERLAP_MIN,
    MIGRATION_TABLE_RENAME_COLUMN_FRACTION,
    PK_STYLE_FK_STEMS,
    SCHEMA_GRAPH_ID_DETERMINISTIC_SEED_V1,
    SCHEMA_GRAPH_ID_PREFIX,
    STRING_VALUE_TYPES,
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SEGMENT,
    WRITE_QUEUE_FILENAME,
)
from ._contracts_base import (
    ColumnRole,
    DatabaseFeatureCapability,
    EngineContext,
    InferenceTag,
    OverrideSkip,
    PkInferenceTag,
    SchemaAccessError,
    SchemaInclude,
    SchemaRole,
    SensitivityClassification,
    data_type_to_value_type,
    is_date_type,
    is_numeric_type,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    SchemaLimits,
    TableMetadata,
    set_schema_helpers,
)
from ._core_utils import (
    debug,
    effective_structural_hash_fp,
    notify,
    profiling_hash_fp,
    read_artifact_manifest,
    read_gzip_json,
    scope_hash_fp,
    stable_json,
    structural_hash_fp,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from ._dialect import Dialect
from ._intent_expr import extract_columns_from_expr
from ._qsim import get_aggregatable_columns, get_groupable_columns
from ._schema_catalog import (
    apply_boolean_coercion_pass,
    apply_column_roles_llm,
    assign_column_ops,
    collect_profiling_frequent_values,
    compute_semantic_profile_join_neighbors,
)


def notes_content_sha256(notes_content: str | None) -> str:
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


def expanded_scope_sql_file(sql_file: str | None) -> str | None:
    """Return an expanded filesystem path for a ``EngineContext.sql_file`` value."""
    if sql_file is None:
        return None
    expanded = os.path.expanduser(str(sql_file).strip())
    return expanded if expanded else None


def compute_dialect_probe(dialect: Dialect, schema_context: EngineContext) -> str:
    """Return the combined DDL probe: dialect ``information_schema`` digest XOR'd with the ``sql_file`` content digest. The combination is a SHA-256 over the two hex digests joined by ``|``. Returns ``""`` when the dialect probe itself is empty (so the caller falls back to fingerprint validation); otherwise always returns a non-empty digest, even when ``sql_file`` is absent. Note on collision risk: the joined-digest construction inherits SHA-256 collision resistance, but a hypothetical adversary who can simultaneously alter both the catalog DDL and the local ``sql_file`` in offsetting ways could in theory produce the same final digest. This is negligible in practice (no adversarial input is involved during cache validation), and the only consequence would be a false cache-hit that the downstream structural fingerprint check is expected to surface; documented here for future auditors."""
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
    """Re-run the LLM column-role classifier and the deterministic boolean coercion pass over *sg* in place. Used by the notes-only refresh fast path in :func:`_build_schema_graph` so a domain-notes edit can update roles, descriptions, and sensitivity without re-reflecting or re-profiling the database. ``skip_columns`` (table, column) pairs are not overwritten by the LLM so user-pinned roles/sensitivities survive a notes-driven reclassification."""
    debug(f"[schema.rerun_column_classifier] reclassifying {len(sg.tables)} tables")
    apply_column_roles_llm(sg, notes_content=notes_content, skip_columns=skip_columns, log_sink=log_sink)
    apply_boolean_coercion_pass(sg)
    assign_column_ops(sg)


def compute_schema_stats(schema: SchemaGraph) -> dict[str, Any]:
    """Compute schema-wide column availability statistics for adaptive. limit calculation."""
    debug("[schema._compute_schema_stats] computing schema statistics for adaptive limits")

    stats: dict[str, Any] = {
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

    debug("[schema._compute_schema_stats] per-table column counts:")
    for td in table_details:
        debug(
            f"  {td['table']}: filterable={td['filterable']}, groupable={td['groupable']}, aggregatable={td['aggregatable']}"
        )

    debug("[schema._compute_schema_stats] schema-wide statistics:")
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
    """Compute adaptive pipeline limits from schema statistics."""
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


def edge_key(e: FKEdge) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """Generate a stable, sortable tuple key for an FK edge."""
    return (e.src_table, tuple(e.src_cols), e.dst_table, tuple(e.dst_cols))


def table_to_dict(table: TableMetadata) -> dict[str, Any]:
    """Serialize a TableMetadata instance to a plain dictionary."""
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


def table_from_dict(d: dict[str, Any]) -> TableMetadata:
    """Deserialize a TableMetadata instance from a plain dictionary."""
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


def load_pg_enum_values(engine: Any) -> dict[str, list[str]]:
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


def effective_reflect_include(ctx: EngineContext) -> SchemaInclude:
    """Use catalog-wide reflection when an explicit allow-list is provided."""
    if ctx.allow_objects:
        return "both"
    return ctx.include


def allow_objects_lower_set(
    allow_objects: frozenset[str] | None,
) -> frozenset[str] | None:
    """Return lowercase relation names for filtering, or ``None`` when no allow-list."""
    if not allow_objects:
        return None
    return frozenset(str(x).lower() for x in allow_objects)


def merge_reflected_schema_graphs(a: SchemaGraph, b: SchemaGraph) -> SchemaGraph:
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
    join_paths_multi = recompute_join_paths_multi(merged_tables)
    return SchemaGraph(
        tables=merged_tables,
        join_paths_multi=join_paths_multi,
        created_at=datetime.now().isoformat(),
        enum_values=a.enum_values or b.enum_values,
    )


def apply_deny_objects_filter(sg: SchemaGraph, ctx: EngineContext) -> None:
    """Remove relations listed in ``ctx.deny_objects`` from the graph."""
    if not ctx.deny_objects:
        return
    tindex = _graph_tables_lower_index(sg.tables)
    deny = {str(x).lower() for x in ctx.deny_objects}
    for low in sorted(deny):
        canon = tindex.get(low)
        if canon and canon in sg.tables:
            del sg.tables[canon]
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)


def apply_allow_objects_filter(sg: SchemaGraph, ctx: EngineContext) -> None:
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

    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)


def semantic_edges_fingerprint(tables: dict[str, TableMetadata]) -> str:
    """Stable digest of semantic join neighbor tuples for migration tiering."""
    edges: list[list[str]] = []
    for tn in sorted(tables):
        for cn in sorted(tables[tn].columns):
            col = tables[tn].columns[cn]
            for nb in sorted(col.semantic_join_neighbors, key=lambda x: tuple(str(p) for p in x)):
                edges.append([tn, cn, *[str(p) for p in nb]])
    return hashlib.sha256(stable_json({"semantic_edges": edges}).encode("utf-8")).hexdigest()


def validate_scope_against_graph(sg: SchemaGraph, ctx: EngineContext) -> None:
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


def _deny_columns_by_table(sg: SchemaGraph, ctx: EngineContext) -> dict[str, set[str]]:
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
    for _canon_tbl, tbl in sg.tables.items():
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


def strip_schema_context_denied_columns(sg: SchemaGraph, ctx: EngineContext) -> None:
    """Remove denied columns from ``TableMetadata.columns`` before profiling. Prunes foreign keys that referenced removed endpoints, clears ``SchemaGraph.deny_columns`` because denied names no longer exist as rows, and leaves the authoritative deny specification on the frozen ``EngineContext`` passed into the build."""
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


def apply_schema_context_allow_columns(sg: SchemaGraph, ctx: EngineContext) -> None:
    """Restrict each table to its ``allow_columns`` subset; PK and FK columns are always retained. No-op when ``ctx.allow_columns`` is empty. Glob ``*.column`` entries match that column name on every table where it exists. Qualified ``table.column`` entries scope to one table. Primary key columns and any column appearing in a foreign key edge (source or destination) are auto-included so the join graph survives a narrow allow list."""
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


def _scope_is_subset_or_equal(narrow: EngineContext, wide: EngineContext) -> bool:
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

    if not narrow.deny_objects.issuperset(wide.deny_objects):
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
    old: EngineContext, new: EngineContext
) -> Literal["identical", "subset", "superset", "orthogonal"]:
    """Classify the relationship between *old* and *new* scope contexts."""
    if (
        old.allow_objects == new.allow_objects
        and old.deny_objects == new.deny_objects
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


def filter_schema_graph_by_scope(sg: SchemaGraph, new_ctx: EngineContext) -> SchemaGraph:
    """Pure (no-I/O) deepcopy of *sg* with *new_ctx* deny rules applied at graph build."""
    new_sg = copy.deepcopy(sg)

    new_sg.deny_columns = {}
    apply_deny_objects_filter(new_sg, new_ctx)
    strip_schema_context_denied_columns(new_sg, new_ctx)

    new_sg.join_paths_multi = recompute_join_paths_multi(new_sg.tables)
    return new_sg


@dataclass(frozen=True)
class TableDiff:
    """Per-table delta between a cached and a freshly-reflected ``SchemaGraph``. Entries are sorted tuples to keep equality + hashing deterministic in tests. ``retyped_columns`` records catalog type changes where the normalized ``value_type`` changes (profile must be refreshed). ``redeclared_columns`` holds pure ``data_type`` widenings (for example ``varchar(50)`` to ``text``) where ``value_type`` is unchanged; those updates merge metadata without clearing profiling samples. ``value_type_changed_columns`` mirrors the ``(column, old_vt, new_vt)`` entries implied by ``retyped_columns``. ``renamed_columns`` is populated by :func:`resolve_column_renames` after profile overlap matching; columns appearing here are removed from ``added_columns`` / ``dropped_columns``."""

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
        """True when applying this diff requires re-profiling the table. Pure-rename tables keep cached profiles. Adds and value-type retypes always need profiling; pure ``redeclared_columns`` (same ``value_type``) do not. Tables whose catalog PK or FK edge sets changed are pulled into :meth:`SchemaDiff.changed_table_names` so subset reprofiling refreshes statistics on those relations even when no columns were added or retyped."""
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
    return frozenset(edge_key(fk) for fk in t.foreign_keys)


def _catalog_fk_edge_set(
    t: TableMetadata,
) -> frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    """Return only the catalog-declared FK edges (``inference_tag is None``). The diff uses this rather than the full edge set so that inferred or user-override edges sitting only on the cached graph do not register as a structural change between the cache and a fresh reflection (which never carries non-catalog tags)."""
    return frozenset(edge_key(fk) for fk in t.foreign_keys if fk.inference_tag is None)


def diff_schemas(old_sg: SchemaGraph, new_sg: SchemaGraph) -> SchemaDiff:
    """Pure structural diff between two :class:`SchemaGraph` instances. Detects column-stable table renames (same column-name+value-type multiset) before per-table column diffing. FK changes are reported as a single boolean per table (handlers re-copy the FK list wholesale; finer-grained join-path repair is left to later migration handling)."""
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


def _column_topk_set(col: ColumnMetadata) -> frozenset[str]:
    """Top-K profiling values normalised for Jaccard comparison."""
    vals = col.frequent_values or []
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
    """Deep-copy *table* and run profiling/classification against it; return the clone. Returns ``None`` when profiling raises so callers can fall back to drop+add."""
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
    """Detect per-table column renames by Top-K profile overlap. For each table in ``diff.per_table`` with both added *and* dropped columns, profile the new (added) columns and greedily match dropped→added pairs whose Jaccard overlap clears *threshold*. Confirmed pairs move from ``added_columns`` / ``dropped_columns`` into ``renamed_columns`` on a fresh :class:`TableDiff`. Unrelated tables are passed through unchanged. The resulting :class:`SchemaDiff` is independent of *diff*."""
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
    """Detect simultaneous table renames (with optional column renames) via profile overlap. For each ``(dropped_table, added_table)`` pair with the same column count, profile the candidate added table, greedily match its columns to the dropped table's cached columns by Top-K Jaccard overlap, and accept the pair as a table rename when the matched-column overlap clears ``overlap_threshold`` for at least ``column_fraction`` of columns. Confirmed renames are removed from ``added_tables`` / ``dropped_tables`` and pushed into ``table_renames`` (plus per-table ``renamed_columns`` for any column renames)."""
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


def redact_hidden_sensitivity_profile_values(sg: SchemaGraph) -> int:
    """Null out concrete profile values on every column whose :attr:`SensitivityClassification` is not :attr:`SensitivityClassification.NONE`. Clears ``frequent_values``, ``min_val``, and ``max_val`` in place. Distinct counts and ratios are statistical, not value-bearing, and remain so the downstream operation-assignment and qsim layers can still gate behaviour on cardinality. Returns the number of columns that were redacted."""
    redacted = 0
    for tbl in sg.tables.values():
        for col in tbl.columns.values():
            if col.sensitivity == SensitivityClassification.NONE:
                continue
            if not col.frequent_values and not col.value_overlap_sample and col.min_val is None and col.max_val is None:
                continue
            col.frequent_values = []
            col.value_overlap_sample = []
            col.min_val = None
            col.max_val = None
            redacted += 1
    return redacted


def apply_fk_remaps_to_graph(
    sg: SchemaGraph,
    remaps: tuple[Any, ...] | list[Any],
) -> int:
    """Rewire FK parent tables according to migration-map ``fk_remap`` entries."""
    count = 0
    for entry in remaps:
        if getattr(entry, "entry_type", None) != "fk_remap":
            continue
        child = str(getattr(entry, "table", "") or "").strip()
        old_parent = str(getattr(entry, "from_name", "") or "").strip()
        new_parent = str(getattr(entry, "to_name", "") or "").strip()
        if not child or not old_parent or not new_parent:
            continue
        if child not in sg.tables or new_parent not in sg.tables:
            continue
        for fk in sg.tables[child].foreign_keys:
            if fk.dst_table == old_parent:
                fk.dst_table = new_parent
                count += 1
    return count


def apply_pk_remaps_to_graph(
    sg: SchemaGraph,
    remaps: tuple[Any, ...] | list[Any],
) -> int:
    """Rewire PK columns and dependent FK dst endpoints per migration- map ``pk_remap`` entries."""
    count = 0
    for entry in remaps:
        if getattr(entry, "entry_type", None) != "pk_remap":
            continue
        tname = str(getattr(entry, "table", "") or "").strip()
        if tname not in sg.tables:
            continue
        old_cols = [c.strip() for c in str(getattr(entry, "from_name", "") or "").split(",") if c.strip()]
        new_cols = [c.strip() for c in str(getattr(entry, "to_name", "") or "").split(",") if c.strip()]
        if not old_cols or not new_cols or len(old_cols) != len(new_cols):
            continue
        tbl = sg.tables[tname]
        old_set = set(old_cols)
        for cname in list(tbl.primary_key):
            if cname in old_set and cname in tbl.columns:
                col = tbl.columns[cname]
                if col.pk_inference_tag is not None:
                    col.pk_inference_tag = None
        tbl.primary_key = list(new_cols)
        for cname in new_cols:
            if cname in tbl.columns:
                col = tbl.columns[cname]
                if col.pk_inference_tag is None:
                    col.pk_inference_tag = PkInferenceTag.DDL
        for src_tbl in sg.tables.values():
            for fk in src_tbl.foreign_keys:
                if fk.dst_table == tname and list(fk.dst_cols) == old_cols:
                    fk.dst_cols = list(new_cols)
                    count += 1
    return count


def raise_if_schema_unusable(sg: SchemaGraph, schema_context: EngineContext) -> None:
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
        "value_overlap_sample": col.value_overlap_sample,
        "semantic_join_neighbors": [list(p) for p in col.semantic_join_neighbors],
        "sensitivity": col.sensitivity,
        "frequent_values": collect_profiling_frequent_values(col.frequent_values),
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


def _fresh_schema_graph_seed() -> str:
    """Return a random 16-hex seed for a newly minted schema-graph identity."""
    return uuid4().hex[:16]


def _derive_deterministic_schema_graph_seed(effective_structural_hash: str) -> str:
    """Return a deterministic 16-hex seed derived from a legacy effective structural hash."""
    payload = (SCHEMA_GRAPH_ID_DETERMINISTIC_SEED_V1 + effective_structural_hash).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def mint_schema_graph_id(*, seed_hex: str, structural_hash: str) -> str:
    """Format a schema-graph identity string from a seed half and structural hash breadcrumb."""
    return f"{SCHEMA_GRAPH_ID_PREFIX}{seed_hex}__{structural_hash[:8]}"


def derive_deterministic_schema_graph_id(
    effective_structural_hash: str,
    structural_hash: str,
) -> str:
    """Build a stable schema-graph identity for upgrading legacy artifact directories."""
    seed = _derive_deterministic_schema_graph_seed(effective_structural_hash)
    return mint_schema_graph_id(seed_hex=seed, structural_hash=structural_hash)


def unify_reflected_schema_graph(sg: SchemaGraph) -> None:
    """Normalize reflection metadata for cross-engine schema-graph comparison."""
    for tbl in sg.tables.values():
        pk_set = set(tbl.primary_key or [])
        for col in tbl.columns.values():
            if col.name in pk_set:
                col.is_nullable = False
        tbl.foreign_keys = sorted(tbl.foreign_keys, key=edge_key)


def assign_schema_graph_hashes(
    sg: SchemaGraph,
    schema_context: EngineContext,
    notes_sha256: str,
    *,
    schema_role: SchemaRole = "owner",
    pinned_schema_graph_id: str | None = None,
) -> None:
    """Compute structural, profiling, scope, effective, notes, semantic- edge hashes and schema-graph identity on *sg* in place."""
    prior_structural = sg.structural_hash
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
    sg.semantic_edges_hash = semantic_edges_fingerprint(sg.tables)
    sg.scope_descriptor = scope_descriptor_for(schema_context)
    if schema_role == "consumer":
        pin = pinned_schema_graph_id or sg.schema_graph_id
        if not pin:
            raise ConfigError("Consumer role requires a pinned schema_graph_id from the owner snapshot artifacts.")
        sg.schema_graph_id = pin
        return
    if not sg.schema_graph_id:
        sg.schema_graph_id = mint_schema_graph_id(seed_hex=_fresh_schema_graph_seed(), structural_hash=st)
        return
    if prior_structural and prior_structural != st:
        sg.schema_graph_id = mint_schema_graph_id(seed_hex=_fresh_schema_graph_seed(), structural_hash=st)


def consumer_graph_is_permission_subset(owner: SchemaGraph, consumer: SchemaGraph) -> bool:
    """Return True when *consumer* tables are a subset of *owner* with matching DDL on overlap."""
    consumer_tables = set(consumer.tables)
    owner_tables = set(owner.tables)
    if not consumer_tables <= owner_tables:
        return False
    owner_payload = tables_structural_payload(owner.tables)
    consumer_payload = tables_structural_payload(consumer.tables)
    for table_name in consumer_tables:
        if owner_payload.get(table_name) != consumer_payload.get(table_name):
            return False
    return True


def assert_intent_in_scope(
    intent: Any,
    allowed_tables: frozenset[str],
    allowed_columns: frozenset[str],
    schema_graph: SchemaGraph,
) -> bool:
    """Return True when every intent table/column reference lies in the space allowlists."""
    if not isinstance(intent, RuntimeIntent):
        return True

    graph_tables = set(schema_graph.tables.keys())
    effective_tables = set(graph_tables)
    if allowed_tables:
        effective_tables &= set(allowed_tables)
    restrict_columns = bool(allowed_columns)
    allowed_column_set = set(allowed_columns)

    def _column_allowed(table_name: str, col_name: str) -> bool:
        if not restrict_columns:
            return True
        qc = f"{table_name}.{col_name}"
        return qc in allowed_column_set

    cte_names = {cte.cte_name for cte in (intent.cte_steps or [])}
    cte_output_cols = {cte.cte_name: set(cte.output_columns or []) for cte in (intent.cte_steps or [])}

    def _check_table(table_name: str) -> bool:
        if table_name in cte_names:
            return True
        return table_name in effective_tables

    def _check_column_ref(col: str) -> bool:
        if "." not in col:
            return True
        table_name, col_name = col.rsplit(".", 1)
        if table_name in cte_output_cols:
            return col_name in cte_output_cols[table_name]
        if not _check_table(table_name):
            return False
        if table_name not in schema_graph.tables:
            return False
        if col_name not in schema_graph.tables[table_name].columns:
            return False
        return _column_allowed(table_name, col_name)

    def _check_scope_block(
        tables: list[str] | None,
        select_cols: list[Any] | None,
        filters_param: list[Any] | None,
        having_param: list[Any] | None,
        order_by_cols: list[Any] | None,
        group_by_cols: list[Any] | None,
        window_registry: list[Any] | None,
    ) -> bool:
        for table_name in tables or []:
            if not _check_table(table_name):
                return False
        for sc in select_cols or []:
            for col in extract_columns_from_expr(sc.expr):
                if not _check_column_ref(col):
                    return False
        for fp in filters_param or []:
            for col in extract_columns_from_expr(fp.left_expr):
                if not _check_column_ref(col):
                    return False
            if fp.right_expr is not None:
                for col in extract_columns_from_expr(fp.right_expr):
                    if not _check_column_ref(col):
                        return False
        for hp in having_param or []:
            for col in extract_columns_from_expr(hp.left_expr):
                if not _check_column_ref(col):
                    return False
            if hp.right_expr is not None:
                for col in extract_columns_from_expr(hp.right_expr):
                    if not _check_column_ref(col):
                        return False
        for ob in order_by_cols or []:
            for col in extract_columns_from_expr(ob.expr):
                if not _check_column_ref(col):
                    return False
        for gb in group_by_cols or []:
            col = gb.primary_term if hasattr(gb, "primary_term") else str(gb)
            if not _check_column_ref(col):
                return False
        for step in window_registry or []:
            ws = step.window_spec
            for part in ws.partition_by or []:
                for col in extract_columns_from_expr(part):
                    if not _check_column_ref(col):
                        return False
            for obc in ws.order_by or []:
                for col in extract_columns_from_expr(obc.expr):
                    if not _check_column_ref(col):
                        return False
            if ws.argument is not None:
                for col in extract_columns_from_expr(ws.argument):
                    if not _check_column_ref(col):
                        return False
        return True

    if not _check_scope_block(
        list(intent.tables or []),
        intent.select_cols,
        intent.filters_param,
        intent.having_param,
        intent.order_by_cols,
        intent.group_by_cols,
        intent.window_registry,
    ):
        return False
    for cte in intent.cte_steps or []:
        if not _check_scope_block(
            list(cte.tables or []),
            cte.select_cols,
            cte.filters_param,
            cte.having_param,
            cte.order_by_cols,
            cte.group_by_cols,
            cte.window_registry,
        ):
            return False
    return True


def assert_consumer_intent_in_scope(
    intent: Any,
    schema_context: EngineContext,
    schema_graph: SchemaGraph,
    visible_objects: frozenset[str] | None,
) -> bool:
    """Return True when every intent table and column reference lies in consumer scope."""
    if not isinstance(intent, RuntimeIntent):
        return True

    allowed_tables = set(schema_graph.tables.keys())
    if schema_context.allow_objects:
        allowed_tables &= set(schema_context.allow_objects)
    if schema_context.deny_objects:
        allowed_tables -= set(schema_context.deny_objects)
    if visible_objects is not None:
        allowed_tables &= set(visible_objects)

    qualified_denies = schema_context.qualified_denies()
    glob_denies = schema_context.glob_column_denies()
    qualified_allows = schema_context.qualified_allows()
    glob_allows = schema_context.glob_column_allows()
    restrict_columns = bool(schema_context.allow_columns)

    if not restrict_columns and not qualified_denies and not glob_denies:
        if visible_objects is not None:
            if not allowed_tables:
                return False
            return assert_intent_in_scope(
                intent,
                frozenset(allowed_tables),
                frozenset(),
                schema_graph,
            )
        return assert_intent_in_scope(
            intent,
            frozenset(allowed_tables) if schema_context.allow_objects else frozenset(),
            frozenset(),
            schema_graph,
        )

    def _column_allowed(table_name: str, col_name: str) -> bool:
        if (table_name, col_name) in qualified_denies:
            return False
        if col_name in glob_denies:
            return False
        if not restrict_columns:
            return True
        if (table_name, col_name) in qualified_allows:
            return True
        if col_name in glob_allows:
            return True
        return False

    cte_names = {cte.cte_name for cte in (intent.cte_steps or [])}
    cte_output_cols = {cte.cte_name: set(cte.output_columns or []) for cte in (intent.cte_steps or [])}

    def _check_table(table_name: str) -> bool:
        if table_name in cte_names:
            return True
        return table_name in allowed_tables

    def _check_column_ref(col: str) -> bool:
        if "." not in col:
            return True
        table_name, col_name = col.rsplit(".", 1)
        if table_name in cte_output_cols:
            return col_name in cte_output_cols[table_name]
        if not _check_table(table_name):
            return False
        if table_name not in schema_graph.tables:
            return False
        if col_name not in schema_graph.tables[table_name].columns:
            return False
        return _column_allowed(table_name, col_name)

    def _check_scope_block(
        tables: list[str] | None,
        select_cols: list[Any] | None,
        filters_param: list[Any] | None,
        having_param: list[Any] | None,
        order_by_cols: list[Any] | None,
        group_by_cols: list[Any] | None,
        window_registry: list[Any] | None,
    ) -> bool:
        for table_name in tables or []:
            if not _check_table(table_name):
                return False
        for sc in select_cols or []:
            for col in extract_columns_from_expr(sc.expr):
                if not _check_column_ref(col):
                    return False
        for fp in filters_param or []:
            for col in extract_columns_from_expr(fp.left_expr):
                if not _check_column_ref(col):
                    return False
            if fp.right_expr is not None:
                for col in extract_columns_from_expr(fp.right_expr):
                    if not _check_column_ref(col):
                        return False
        for hp in having_param or []:
            for col in extract_columns_from_expr(hp.left_expr):
                if not _check_column_ref(col):
                    return False
            if hp.right_expr is not None:
                for col in extract_columns_from_expr(hp.right_expr):
                    if not _check_column_ref(col):
                        return False
        for ob in order_by_cols or []:
            for col in extract_columns_from_expr(ob.expr):
                if not _check_column_ref(col):
                    return False
        for gb in group_by_cols or []:
            col = gb.primary_term if hasattr(gb, "primary_term") else str(gb)
            if not _check_column_ref(col):
                return False
        for step in window_registry or []:
            ws = step.window_spec
            for part in ws.partition_by or []:
                for col in extract_columns_from_expr(part):
                    if not _check_column_ref(col):
                        return False
            for obc in ws.order_by or []:
                for col in extract_columns_from_expr(obc.expr):
                    if not _check_column_ref(col):
                        return False
            if ws.argument is not None:
                for col in extract_columns_from_expr(ws.argument):
                    if not _check_column_ref(col):
                        return False
        return True

    if not _check_scope_block(
        list(intent.tables or []),
        intent.select_cols,
        intent.filters_param,
        intent.having_param,
        intent.order_by_cols,
        intent.group_by_cols,
        intent.window_registry,
    ):
        return False
    for cte in intent.cte_steps or []:
        if not _check_scope_block(
            list(cte.tables or []),
            cte.select_cols,
            cte.filters_param,
            cte.having_param,
            cte.order_by_cols,
            cte.group_by_cols,
            cte.window_registry,
        ):
            return False
    return True


def assert_consumer_sql_in_scope(
    sql: str,
    dialect: Any,
    schema_context: EngineContext,
    schema_graph: SchemaGraph,
    visible_objects: frozenset[str] | None,
) -> bool:
    """Return True when physical table/column references in *sql* lie in consumer scope."""
    read_dialect = getattr(dialect, "sqlglot_dialect", None) or getattr(dialect, "name", "postgres")
    try:
        tree = sqlglot.parse_one(sql, read=read_dialect)
    except Exception:
        return False

    allowed_tables = set(schema_graph.tables.keys())
    if schema_context.allow_objects:
        allowed_tables &= set(schema_context.allow_objects)
    if schema_context.deny_objects:
        allowed_tables -= set(schema_context.deny_objects)
    if visible_objects is not None:
        allowed_tables &= set(visible_objects)

    qualified_denies = schema_context.qualified_denies()
    glob_denies = schema_context.glob_column_denies()
    qualified_allows = schema_context.qualified_allows()
    glob_allows = schema_context.glob_column_allows()
    restrict_columns = bool(schema_context.allow_columns)

    cte_names: set[str] = set()
    wc = tree.args.get("with_")
    if wc is not None:
        for cte in wc.expressions or ():
            alias = cte.alias_or_name
            if isinstance(alias, str) and alias:
                cte_names.add(alias.lower())

    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(sqlglot.exp.Table):
        real = (table.name or "").lower()
        if not real:
            continue
        alias_to_table[real] = real
        alias_node = table.args.get("alias")
        if alias_node is not None:
            alias = alias_node.name if hasattr(alias_node, "name") else None
            if isinstance(alias, str) and alias:
                alias_to_table[alias.lower()] = real

    for table in tree.find_all(sqlglot.exp.Table):
        name = (table.name or "").lower()
        if not name or name in cte_names:
            continue
        if name not in allowed_tables:
            return False

    for col in tree.find_all(sqlglot.exp.Column):
        col_name = col.name or ""
        if not col_name or col_name == "*":
            continue
        tbl = (col.table or "").lower() if col.table else ""
        table_name = alias_to_table.get(tbl, tbl) if tbl else ""
        if not table_name or table_name in cte_names:
            continue
        if table_name not in allowed_tables:
            return False
        if table_name not in schema_graph.tables:
            return False
        if col_name not in schema_graph.tables[table_name].columns:
            continue
        if (table_name, col_name) in qualified_denies or col_name in glob_denies:
            return False
        if restrict_columns and (table_name, col_name) not in qualified_allows and col_name not in glob_allows:
            return False
    return True


def upgrade_artifacts_schema_graph_id(artifacts_dir: str) -> dict[str, int]:
    """Backfill ``schema_graph_id`` on legacy artifact directories under *artifacts_dir*."""
    counts = {"template_rows": 0, "queue_lines": 0, "lattices_renamed": 0}
    manifest = read_artifact_manifest(artifacts_dir)
    if manifest is not None and manifest.schema_graph_id:
        return counts
    eff = str(manifest.effective_structural_hash if manifest else "")
    structural = str(manifest.structural_hash if manifest else "")
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    header_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
    if not eff and os.path.isfile(header_path):
        try:
            header = read_gzip_json(header_path)
            if isinstance(header, dict):
                eff = str(header.get("effective_structural_hash", header.get("schema_hash", "")) or "")
        except (
            OSError,
            EOFError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            eff = ""
    if not structural and eff:
        structural = eff
    if not eff:
        return counts
    graph_id = derive_deterministic_schema_graph_id(eff, structural or eff)
    if os.path.isfile(header_path):
        try:
            header = read_gzip_json(header_path)
        except (
            OSError,
            EOFError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            header = {}
        if isinstance(header, dict):
            header["schema_graph_id"] = graph_id
            header.pop("effective_structural_hash", None)
            write_gzip_json_atomic(header_path, header, sort_keys=True)
        for part_path in glob.glob(os.path.join(store_dir, f"{TEMPLATE_STORE_PARTITION_PREFIX}*.json.gz")):
            try:
                part_doc = read_gzip_json(part_path)
            except (
                OSError,
                EOFError,
                gzip.BadGzipFile,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                continue
            if not isinstance(part_doc, dict):
                continue
            for _tid, row in part_doc.items():
                if isinstance(row, dict):
                    row["schema_graph_id"] = graph_id
                    counts["template_rows"] += 1
            write_gzip_json_atomic(part_path, part_doc, sort_keys=True)
    schema_path = os.path.join(artifacts_dir, "schema_graph.json.gz")
    if os.path.isfile(schema_path):
        try:
            cache = read_gzip_json(schema_path)
            if isinstance(cache, dict):
                cache["schema_graph_id"] = graph_id
                write_gzip_json_atomic(schema_path, cache, sort_keys=True)
        except (
            OSError,
            EOFError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            pass
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=manifest.structural_hash if manifest else structural,
        profiling_hash=manifest.profiling_hash if manifest else "",
        scope_hash=manifest.scope_hash if manifest else "",
        effective_structural_hash=eff,
        schema_graph_id=graph_id,
        notes_hash=manifest.notes_hash if manifest else "",
        semantic_edges_hash=manifest.semantic_edges_hash if manifest else "",
        last_migration_tier=manifest.last_migration_tier if manifest else "",
        last_migration_at=manifest.last_migration_at if manifest else "",
        last_action="schema_graph_id_upgrade",
    )
    queue_path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    if os.path.isfile(queue_path):
        new_lines: list[str] = []
        with open(queue_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(doc, dict):
                    continue
                doc["schema_graph_id"] = graph_id
                new_lines.append(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
                counts["queue_lines"] += 1
        if new_lines:
            with open(queue_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(new_lines) + "\n")
    safe_id = graph_id.replace("__", "_")
    for old_path in glob.glob(os.path.join(artifacts_dir, "**", f"lattice_{eff}_*.json"), recursive=True):
        base = os.path.basename(old_path)
        new_base = base.replace(f"lattice_{eff}_", f"lattice_{safe_id}_", 1)
        new_path = os.path.join(os.path.dirname(old_path), new_base)
        try:
            os.rename(old_path, new_path)
            counts["lattices_renamed"] += 1
        except OSError:
            pass
    return counts


def scope_descriptor_for(ctx: EngineContext) -> dict[str, Any]:
    """Return a JSON-serialisable descriptor of *ctx*'s scope-relevant fields for cache persistence."""
    return {
        "allow_objects": sorted(ctx.allow_objects),
        "deny_objects": sorted(ctx.deny_objects),
        "deny_columns": sorted(ctx.deny_columns),
        "allow_columns": sorted(ctx.allow_columns),
        "include": ctx.include,
    }


def schema_context_from_descriptor(desc: dict[str, Any]) -> EngineContext:
    """Reconstruct a :class:`EngineContext` from a cached scope descriptor (4 scope fields)."""
    inc_raw = desc.get("include", "tables")
    if inc_raw not in ("tables", "views", "both"):
        inc_raw = "tables"
    return EngineContext(
        name="master",
        allow_objects=frozenset(str(x) for x in (desc.get("allow_objects") or [])),
        deny_objects=frozenset(str(x) for x in (desc.get("deny_objects") or [])),
        deny_columns=frozenset(str(x) for x in (desc.get("deny_columns") or [])),
        allow_columns=frozenset(str(x) for x in (desc.get("allow_columns") or [])),
        include=inc_raw,
    )


def _reverse_fk_path(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reverse a FK path by flipping each edge's direction and. reversing. list order."""
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
    """Analyze an FK path to determine its topology type, anchor table, and leaf tables."""
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


def compute_join_paths_multi_from_adj(
    adj: dict[str, list[FKEdge]],
    tlist: list[str],
) -> dict[str, dict[str, list[list[dict[str, Any]]]]]:
    """All shortest FK-edge paths per ordered table pair, capped per. pair for storage size."""
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
    """Normalize an FK join path to a canonical form based on its. topology."""
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


def fk_tables_lower_index(tables: dict[str, TableMetadata]) -> dict[str, str]:
    """Map lowercased table name to the first canonical table key with that spelling."""
    index: dict[str, str] = {}
    for name in tables:
        lo = name.lower()
        index.setdefault(lo, name)
    return index


def fk_match_suffix_stem(col_lower: str) -> str | None:
    """Return the first configured stem that *col_lower* ends with, or None."""
    for stem in FK_INFERENCE_SUFFIX_STEMS:
        if col_lower.endswith(stem):
            return str(stem)
    return None


def fk_candidate_prefixes(col_lower: str, stem: str) -> list[str]:
    """Return plural-tolerant prefix candidates derived from stripping. *stem* from *col_lower*. The output is the deduplicated, length- descending list of plausible target-table-name prefixes derived from a foreign-key-shaped column name. Includes the full prefix, every right-anchored sub-segment when the prefix is snake_case, and singular/plural variants of each. Returns an empty list when stripping the stem leaves no usable prefix."""
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


def fk_name_shape_matches_table(col_lower: str, dst_table_lower: str) -> bool:
    """Return True when *col_lower* has a recognised FK-style suffix and one of its prefix candidates equals *dst_table_lower*. Used by both suffix FK inference (layer 2) and the semantic→FK promoter (layer 5) so the two layers agree on what a "FK-shaped" column name pointing at a given table looks like."""
    if not col_lower or not dst_table_lower:
        return False
    stem = fk_match_suffix_stem(col_lower)
    if not stem:
        return False
    return dst_table_lower in fk_candidate_prefixes(col_lower, stem)


def fk_infer_value_types_compatible(src: ColumnMetadata, dst_col: ColumnMetadata | None) -> bool:
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
    """Allow string↔integer FK candidates when the string-side samples are all digit strings. Symmetric: accepts ``a`` string / ``b`` integer or vice versa. When the string side has no samples it cannot be judged digit-only and the helper returns False; the conservative answer keeps spurious string→int FKs from being promoted on naming alone. Coercion only widens the inference compatibility check; downstream value-type semantics are unchanged."""
    at = (a.value_type or "").strip().lower()
    bt = (b.value_type or "").strip().lower()
    if at in STRING_VALUE_TYPES and bt in INTEGER_VALUE_TYPES:
        string_side = a
    elif bt in STRING_VALUE_TYPES and at in INTEGER_VALUE_TYPES:
        string_side = b
    else:
        return False
    samples = list(string_side.frequent_values or [])
    if not samples:
        return False
    return all(str(v).strip().lstrip("-").isdigit() for v in samples if v is not None and str(v).strip() != "")


def _fk_overlap_sample_norm_set(col: ColumnMetadata) -> set[str]:
    out: set[str] = set()
    for v in col.value_overlap_sample or []:
        s = str(v).strip()
        if s == "":
            continue
        out.add(s)
    return out


def fk_overlap_validates(src: ColumnMetadata, dst: ColumnMetadata) -> bool:
    """Return True when sampled values overlap enough to support an inferred FK. Compares ``value_overlap_sample`` from both sides after normalizing via ``str()`` and stripping. When either side has fewer than ``PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE`` non-empty samples the helper returns True (insufficient evidence to reject — fall back to the naming-only signal). Otherwise the overlap ratio is computed against the smaller sample set, and the candidate is accepted when the ratio is at least ``PolicyConfig.FK_INFER_OVERLAP_MIN_RATIO``. The helper is symmetric and treats integer / digit-string pairs as equal after string normalization so it cooperates with ``_fk_string_int_compatible``."""
    a_set = _fk_overlap_sample_norm_set(src)
    b_set = _fk_overlap_sample_norm_set(dst)
    min_sample = int(PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE)
    if len(a_set) < min_sample or len(b_set) < min_sample:
        return True
    overlap = len(a_set & b_set)
    smaller = min(len(a_set), len(b_set))
    ratio = overlap / smaller if smaller else 0.0
    return ratio >= float(PolicyConfig.FK_INFER_OVERLAP_MIN_RATIO)


def _fk_containment_validates(child: ColumnMetadata, parent: ColumnMetadata) -> bool:
    """Return True when the child sample is sufficiently contained in the parent sample. Directional FK semantics require child values to appear in the parent key domain. ASC-ordered LIMIT-N samples are not guaranteed to be subsets, so acceptance uses ``|child ∩ parent| / |child|`` against ``PolicyConfig.FK_INFER_CONTAINMENT_MIN_RATIO``."""
    child_set = _fk_overlap_sample_norm_set(child)
    parent_set = _fk_overlap_sample_norm_set(parent)
    min_sample = int(PolicyConfig.FK_INFER_OVERLAP_MIN_SAMPLE)
    if len(child_set) < min_sample or len(parent_set) < min_sample:
        return True
    if not child_set:
        return True
    containment = len(child_set & parent_set) / len(child_set)
    return containment >= float(PolicyConfig.FK_INFER_CONTAINMENT_MIN_RATIO)


def revalidate_named_fks_with_overlap(sg: SchemaGraph) -> int:
    """Drop name-inferred FK edges whose child values fail containment against the parent PK."""
    revalidate_tags = frozenset(
        {
            InferenceTag.SUFFIX,
            InferenceTag.COMPOSITE,
            InferenceTag.SELF,
            InferenceTag.SEMANTIC,
            InferenceTag.SEMANTIC_PROMOTED,
        }
    )
    removed = 0
    for tbl_name, tbl in sg.tables.items():
        kept: list[FKEdge] = []
        for edge in tbl.foreign_keys:
            if edge.src_table != tbl_name:
                continue
            tag = edge.inference_tag
            if tag not in revalidate_tags:
                kept.append(edge)
                continue
            dst_tbl = sg.tables.get(edge.dst_table)
            if dst_tbl is None:
                kept.append(edge)
                continue
            valid = True
            for src_col, dst_col in zip(edge.src_cols, edge.dst_cols, strict=False):
                child_col = tbl.columns.get(src_col)
                parent_col = dst_tbl.columns.get(dst_col)
                if child_col is None or parent_col is None:
                    valid = False
                    break
                if not fk_infer_value_types_compatible(child_col, parent_col):
                    valid = False
                    break
                if not _fk_containment_validates(child_col, parent_col):
                    valid = False
                    break
            if valid:
                kept.append(edge)
            else:
                removed += 1
        tbl.foreign_keys = kept
    return removed


_INFERRED_PK_NAME_SUFFIXES: tuple[str, ...] = tuple(s for s in FK_INFERENCE_SUFFIX_STEMS if s in PK_STYLE_FK_STEMS)
_PK_NAME_SUFFIXES_FOR_LONGEST: tuple[str, ...] = ("_id", "_key", "_uuid", "_pk")


def _pk_null_gate_passes(col: ColumnMetadata) -> bool:
    """Return True when a column satisfies the not-null gate for PK inference."""
    if col.null_ratio > 0.0:
        return False
    if col.is_nullable:
        return False
    return True


def _statistical_pk_unique(
    table_name: str,
    table: TableMetadata,
    col_name: str,
    col: ColumnMetadata,
    *,
    dialect: Any | None,
    min_rows: int,
) -> bool:
    """Return True when profiling statistics support a tier-2 PK candidate."""
    rc = int(table.row_count or 0)
    if col.distinct_from_sample:
        if dialect is None:
            return False
        ft = dialect.refresh_full_table_distinct_for_pk_inference(
            table_name,
            col_name,
            table_kind=table.kind,
        )
        if ft is None:
            return False
        dist_ft, cnt_ft, nr_ft = ft
        col.distinct_count = dist_ft
        col.null_ratio = nr_ft
        col.distinct_from_sample = False
        if cnt_ft > 0:
            table.row_count = cnt_ft
        rc = int(table.row_count or 0)
        return rc > 0 and dist_ft == rc and rc >= min_rows
    return col.distinct_count is not None and rc > 0 and col.distinct_count == rc and rc >= min_rows


def _apply_inferred_pk_columns(table: TableMetadata, col_names: list[str]) -> None:
    """Stamp profile-inferred PK membership on *table* for *col_names*."""
    for col_name in col_names:
        col_meta = table.columns[col_name]
        col_meta.pk_inference_tag = PkInferenceTag.PROFILE
        if col_name not in table.primary_key:
            table.primary_key.append(col_name)


def _infer_composite_pk_from_profile(
    table_name: str,
    table: TableMetadata,
    not_null_cols: list[str],
    *,
    dialect: Any | None,
    min_rows: int,
) -> list[str] | None:
    """Probe NOT NULL column combinations until one matches row-count distinct tuples."""
    rc = int(table.row_count or 0)
    if rc <= 0 or len(not_null_cols) < 2:
        return None
    max_width = int(PolicyConfig.INFERRED_PK_COMPOSITE_MAX_COLUMNS)
    ordered = sorted(not_null_cols)
    refresh = getattr(dialect, "refresh_composite_distinct_for_pk_inference", None) if dialect else None
    if refresh is None:
        return None
    for width in range(2, min(len(ordered), max_width) + 1):
        for combo in combinations(ordered, width):
            cols = list(combo)
            ft = refresh(table_name, cols, table_kind=table.kind)
            if ft is None:
                continue
            dist_ft, cnt_ft, _nr_ft = ft
            if cnt_ft > 0:
                table.row_count = cnt_ft
            if dist_ft == cnt_ft and cnt_ft >= min_rows:
                return cols
    return None


def infer_missing_pks_from_profile(
    tables: dict[str, TableMetadata],
    *,
    blocked: frozenset[tuple[str, str]] = frozenset(),
    dialect: Any | None = None,
) -> list[tuple[str, str]]:
    """Infer single- or composite-column primary keys from catalog UNIQUE and profiling statistics."""
    inferred: list[tuple[str, str]] = []
    if not tables:
        return inferred
    min_rows = int(PolicyConfig.INFERRED_PK_MIN_ROW_COUNT)
    for table_name, table in tables.items():
        if table.primary_key:
            continue
        tier1: list[str] = []
        tier2: list[str] = []
        not_null_cols: list[str] = []
        for col_name, col in table.columns.items():
            if (table_name, col_name) in blocked:
                continue
            if not _pk_null_gate_passes(col):
                continue
            vt = (col.value_type or "").strip().lower()
            if vt and vt not in INFERRED_PK_VALUE_TYPES:
                continue
            not_null_cols.append(col_name)
            if col.is_unique:
                tier1.append(col_name)
                continue
            if _statistical_pk_unique(
                table_name,
                table,
                col_name,
                col,
                dialect=dialect,
                min_rows=min_rows,
            ):
                tier2.append(col_name)
        chosen_cols: list[str] | None = None
        if tier1:
            chosen = _select_inferred_pk_candidate(table_name, tier1)
            if chosen is not None:
                chosen_cols = [chosen]
        elif tier2:
            chosen = _select_inferred_pk_candidate(table_name, tier2)
            if chosen is not None:
                chosen_cols = [chosen]
        elif not_null_cols:
            composite = _infer_composite_pk_from_profile(
                table_name,
                table,
                not_null_cols,
                dialect=dialect,
                min_rows=min_rows,
            )
            if composite:
                chosen_cols = composite
        if chosen_cols:
            _apply_inferred_pk_columns(table, chosen_cols)
            for col_name in chosen_cols:
                debug(f"[schema.infer_missing_pks] inferred PK {table_name}.{col_name}")
                inferred.append((table_name, col_name))
        else:
            if not_null_cols:
                notify(
                    f"No primary key could be inferred for table {table_name!r}; declare one via schema overrides.",
                    stage="schema",
                    code=DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT,
                )
    inferred.sort()
    return inferred


def _select_inferred_pk_candidate(table_name: str, candidates: list[str]) -> str | None:
    """Pick a single primary-key candidate using tier-local name heuristics."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    sorted_candidates = sorted(candidates)
    table_lower = table_name.lower()
    for name in sorted_candidates:
        lower = name.lower()
        if lower == "id" or lower == "key":
            return name
    for name in sorted_candidates:
        lower = name.lower()
        if lower == f"{table_lower}_id" or lower == f"{table_lower}_key":
            return name
    best: str | None = None
    best_len = -1
    for name in sorted_candidates:
        lower = name.lower()
        for suffix in _PK_NAME_SUFFIXES_FOR_LONGEST:
            if lower.endswith(suffix) and len(suffix) > best_len:
                best = name
                best_len = len(suffix)
    if best is not None:
        return best
    return sorted_candidates[0]


def catalog_fk_union_find(sg: SchemaGraph) -> _FkUnionFind:
    """Connect tables using only catalog-declared FK edges."""
    uf = _FkUnionFind()
    for t in sg.tables:
        uf.make_set(t)
    for tbl_name, tbl in sg.tables.items():
        for e in tbl.foreign_keys:
            if e.src_table != tbl_name:
                continue
            if e.inference_tag is None:
                uf.union(e.src_table, e.dst_table)
    return uf


def catalog_fk_graph_is_connected(sg: SchemaGraph) -> bool:
    """Return True when every table lies in one catalog-FK connected component."""
    if len(sg.tables) <= 1:
        return True
    uf = catalog_fk_union_find(sg)
    root: str | None = None
    for t in sg.tables:
        r = uf.find(t)
        if root is None:
            root = r
        elif r != root:
            return False
    return True


def structural_fk_table_pairs(sg: SchemaGraph) -> frozenset[frozenset[str]]:
    """Return undirected table pairs that already have at least one FK edge."""
    pairs: set[frozenset[str]] = set()
    for tbl_name, tbl in sg.tables.items():
        for e in tbl.foreign_keys:
            if e.src_table != tbl_name:
                continue
            pairs.add(frozenset({e.src_table, e.dst_table}))
    return frozenset(pairs)


def table_pair_has_structural_fk(sg: SchemaGraph, a: str, b: str) -> bool:
    """Return True when *a* and *b* already share a structural FK edge."""
    return frozenset({a, b}) in structural_fk_table_pairs(sg)


def bridge_disjoint_graph_by_value_overlap(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
) -> int:
    """Bridge catalog-FK islands using type-compatible column value overlap."""
    if catalog_fk_graph_is_connected(sg):
        return 0
    uf = catalog_fk_union_find(sg)
    promoted = 0
    cols: list[tuple[str, str, ColumnMetadata]] = []
    for tname, tbl in sg.tables.items():
        for cname, cmeta in tbl.columns.items():
            if cmeta.value_overlap_sample:
                cols.append((tname, cname, cmeta))
    for i, (t1, c1, m1) in enumerate(cols):
        for t2, c2, m2 in cols[i + 1 :]:
            if t1 == t2:
                continue
            if uf.find(t1) == uf.find(t2):
                continue
            if frozenset({t1, t2}) in structural_fk_table_pairs(sg):
                continue
            if not fk_infer_value_types_compatible(m1, m2):
                continue
            if not fk_overlap_validates(m1, m2):
                continue
            left_is_pk = m1.is_primary_key
            right_is_pk = m2.is_primary_key
            if left_is_pk and not right_is_pk:
                src_tbl, src_col, dst_tbl, dst_col = t2, c2, t1, c1
            elif right_is_pk and not left_is_pk:
                src_tbl, src_col, dst_tbl, dst_col = t1, c1, t2, c2
            elif left_is_pk and right_is_pk:
                src_tbl, src_col, dst_tbl, dst_col = t1, c1, t2, c2
            else:
                m1.semantic_join_neighbors = sorted(
                    set(m1.semantic_join_neighbors) | {(t2, c2)},
                    key=lambda p: (p[0], p[1]),
                )
                m2.semantic_join_neighbors = sorted(
                    set(m2.semantic_join_neighbors) | {(t1, c1)},
                    key=lambda p: (p[0], p[1]),
                )
                continue
            child = sg.tables[src_tbl].columns[src_col]
            parent = sg.tables[dst_tbl].columns[dst_col]
            if not _fk_containment_validates(child, parent):
                continue
            edge_key_fwd = (src_tbl, (src_col,), dst_tbl, (dst_col,))
            if edge_key_fwd in blocked:
                continue
            edge = FKEdge(
                src_table=src_tbl,
                src_cols=[src_col],
                dst_table=dst_tbl,
                dst_cols=[dst_col],
                inference_tag=InferenceTag.SEMANTIC_PROMOTED,
            )
            n = _apply_inferred_fks_to_graph(sg, [edge])
            if n:
                promoted += n
                uf = catalog_fk_union_find(sg)
    return promoted


def run_fk_inference_if_disconnected(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
    semantic_neighbors: bool = True,
    semantic_promotion: bool = True,
) -> int:
    """Run FK inference, semantic discovery, and overlap bridging only when catalog FKs leave islands."""
    if catalog_fk_graph_is_connected(sg):
        debug("[schema.fk_inference] catalog FK graph connected; skipping inference")
        return 0
    total = pair_targeted_fk_inference(sg, blocked=blocked)
    if semantic_neighbors:
        compute_semantic_profile_join_neighbors(sg)
    revalidated = revalidate_named_fks_with_overlap(sg)
    if revalidated:
        debug(f"[schema.fk_inference] removed {revalidated} name-inferred FK(s) failing overlap")
    if semantic_promotion:
        total += promote_cross_component_semantic_edges(sg, blocked=blocked)
        total += promote_same_component_semantic_edges(sg, blocked=blocked)
    total += bridge_disjoint_graph_by_value_overlap(sg, blocked=blocked)
    return total


def _apply_inferred_fks_to_graph(sg: SchemaGraph, edges: list[FKEdge]) -> int:
    """Append inferred FK edges to *sg* in-place, marking source. columns. as foreign keys. Skips edges whose endpoint columns no longer exist or whose canonical edge key already appears in the source table. Returns the number of edges appended."""
    if not edges:
        return 0
    added = 0
    for e in edges:
        src_tbl = sg.tables.get(e.src_table)
        dst_tbl = sg.tables.get(e.dst_table)
        if src_tbl is None or dst_tbl is None:
            continue
        existing = {edge_key(x) for x in src_tbl.foreign_keys}
        if edge_key(e) in existing:
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


def pair_targeted_fk_inference(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> int:
    """Infer FK candidates between table pairs until fixed point when the catalog FK graph is disconnected."""
    if catalog_fk_graph_is_connected(sg):
        debug("[schema.pair_targeted_fk_inference] catalog graph connected; skipping")
        return 0
    total = 0
    names = sorted(sg.tables.keys())
    while True:
        round_added = 0
        skip_pairs = structural_fk_table_pairs(sg)
        for t in names:
            if t not in sg.tables:
                continue
            edges = infer_missing_fks(
                sg.tables,
                blocked=blocked,
                restrict_tables=frozenset({t}),
                skip_table_pairs=skip_pairs,
            )
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
                skip_pairs = structural_fk_table_pairs(sg)
                edges = infer_missing_fks(
                    sg.tables,
                    blocked=blocked,
                    restrict_tables=frozenset({a, b}),
                    skip_table_pairs=skip_pairs,
                )
                n = _apply_inferred_fks_to_graph(sg, edges)
                round_added += n
                if n:
                    uf = _union_find_with_tag_filter(sg, frozenset())
        total += round_added
        if round_added == 0:
            break
    return total


def collapse_redundant_inferences(sg: SchemaGraph, skipped: list[OverrideSkip]) -> int:
    """Remove inferred FK edges and semantic neighbor pairs made redundant by catalog/user truth FKs. Truth connectivity is catalog FKs plus ``user_override_*`` FK edges. Returns a removal count."""
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


def mark_canonical_duplicates(sg: SchemaGraph) -> int:
    """Recompute the canonical-bearer index on *sg* for every. duplicated. column name."""
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


def promote_semantic_neighbor_pairs(
    sg: SchemaGraph,
    *,
    cross_component_only: bool,
    inference_tag: InferenceTag,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
) -> int:
    """Promote semantic profile neighbor pairs to inferred FK edges when gates pass. Connectivity for filtering uses all FK edges except pure ``semantic`` inference tags so ``semantic_promoted`` bridges participate once emitted."""
    if not sg.tables:
        return 0
    seen_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    promotions: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for tbl_name, tbl in sg.tables.items():
        for col_name, col in tbl.columns.items():
            for neigh_tbl, neigh_col in col.semantic_join_neighbors:
                a = (tbl_name, col_name)
                b = (neigh_tbl, neigh_col)
                ordered_pair: list[tuple[str, str]] = sorted([a, b])
                pair: tuple[tuple[str, str], tuple[str, str]] = (ordered_pair[0], ordered_pair[1])
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
        if not fk_infer_value_types_compatible(left_col, right_col):
            continue
        left_vt = (left_col.value_type or "").strip().lower()
        right_vt = (right_col.value_type or "").strip().lower()
        if left_vt != "string" or right_vt != "string":
            debug(
                f"[schema.promote_semantic_to_fk] string-only gate reject "
                f"{left[0]}.{left[1]} ({left_vt}) <-> {right[0]}.{right[1]} ({right_vt})"
            )
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
        child_col = sg.tables[src_tbl_name].columns[src_col_name]
        parent_col = sg.tables[dst_tbl_name].columns[dst_col_name]
        if not _fk_containment_validates(child_col, parent_col):
            debug(f"[schema.promote_semantic_to_fk] overlap fail {left[0]}.{left[1]} <-> {right[0]}.{right[1]}")
            continue
        edge_key = (src_tbl_name, (src_col_name,), dst_tbl_name, (dst_col_name,))
        rev_key = (dst_tbl_name, (dst_col_name,), src_tbl_name, (src_col_name,))
        if edge_key in blocked or rev_key in blocked:
            continue
        if src_tbl_name == dst_tbl_name and src_col_name == dst_col_name:
            continue
        bridge_table, bridge_other_table = left[0], right[0]
        connected = uf.find(bridge_table) == uf.find(bridge_other_table)
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


def _infer_missing_fks_suffix(
    tables: dict[str, TableMetadata],
    tables_lower: dict[str, str],
    *,
    skip_table_pairs: frozenset[frozenset[str]] = frozenset(),
) -> list[FKEdge]:
    """Infer FK edges from ``*_id`` / ``*_key`` style names using case- insensitive matching."""
    inferred: list[FKEdge] = []
    for table_name, table in tables.items():
        for col_name, col in table.columns.items():
            if col.is_foreign_key or col.is_primary_key:
                continue
            col_lower = col_name.lower()
            matched_suffix = fk_match_suffix_stem(col_lower)
            if not matched_suffix:
                continue
            ordered = fk_candidate_prefixes(col_lower, matched_suffix)
            if not ordered:
                continue
            for pref_lower in ordered:
                if not fk_name_shape_matches_table(col_lower, pref_lower):
                    continue
                dst_table = tables_lower.get(pref_lower)
                if not dst_table:
                    continue
                if frozenset({table_name, dst_table}) in skip_table_pairs:
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
                    if not fk_infer_value_types_compatible(col, dst_meta_col):
                        debug(
                            f"[schema.infer_missing_fks] suffix skip type mismatch "
                            f"{table_name}.{col_name} -> {dst_table}.{target_pk}"
                        )
                        continue
                    if dst_meta_col is not None and not fk_overlap_validates(col, dst_meta_col):
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


def promote_cross_component_semantic_edges(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
) -> int:
    """Prefer semantic promotions that bridge structural FK islands."""
    return promote_semantic_neighbor_pairs(
        sg,
        cross_component_only=True,
        inference_tag=InferenceTag.SEMANTIC_PROMOTED,
        blocked=blocked,
    )


def compute_database_feature_capability(sg: SchemaGraph) -> DatabaseFeatureCapability:
    """Derive a once-per-graph capability snapshot for tier and QSim. feature gating."""
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
    other_date_columny: dict[str, set[str]] = {}
    arr_by: dict[str, set[str]] = {}
    has_num = False
    has_date = False
    has_arr = False
    has_cat = False

    for tn, tbl in sg.tables.items():
        for cn, col in tbl.columns.items():
            dt = str(col.data_type or "")
            dtl = dt.lower()
            if col.element_type or "array" in dtl or dtl.endswith("[]") or ("[" in dtl and "]" in dtl):
                has_arr = True
                arr_by.setdefault(tn, set()).add(cn)
            role_v = str(col.role or "")
            if is_date_type(dt) or role_v == ColumnRole.TEMPORAL.value:
                has_date = True
                other_date_columny.setdefault(tn, set()).add(cn)
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
        date_columns_by_table={k: frozenset(v) for k, v in other_date_columny.items()},
        array_columns_by_table={k: frozenset(v) for k, v in arr_by.items()},
    )


def schema_context_from_graph(sg: SchemaGraph) -> EngineContext:
    """Reconstruct a :class:`EngineContext` fingerprint input from a loaded graph."""
    desc = sg.scope_descriptor
    if isinstance(desc, dict) and desc:
        return schema_context_from_descriptor(desc)
    specs: list[str] = []
    for tbl, cols in sg.deny_columns.items():
        for c in cols:
            specs.append(f"{tbl}.{c}")
    return EngineContext(
        name="master",
        include=sg.include,
        deny_columns=frozenset(specs),
    )


def _infer_missing_fks_same_name(
    tables: dict[str, TableMetadata],
    tables_lower: dict[str, str],
    *,
    skip_table_pairs: frozenset[frozenset[str]] = frozenset(),
) -> list[FKEdge]:
    """Infer FK edges from exact same-name columns with value overlap."""
    inferred: list[FKEdge] = []
    by_lower: dict[str, list[tuple[str, str, ColumnMetadata]]] = {}
    for tname, tbl in tables.items():
        for cname, col in tbl.columns.items():
            if col.is_foreign_key or col.is_primary_key:
                continue
            by_lower.setdefault(cname.lower(), []).append((tname, cname, col))
    for entries in by_lower.values():
        if len(entries) < 2:
            continue
        for i, (t1, c1, col1) in enumerate(entries):
            for t2, c2, col2 in entries[i + 1 :]:
                if t1 == t2:
                    continue
                if frozenset({t1, t2}) in skip_table_pairs:
                    continue
                dst_tbl = tables.get(t2)
                if dst_tbl is None or len(dst_tbl.primary_key) != 1 or dst_tbl.primary_key[0] != c2:
                    dst_tbl = tables.get(t1)
                    if dst_tbl is None or len(dst_tbl.primary_key) != 1 or dst_tbl.primary_key[0] != c1:
                        continue
                    src_name, src_col, dst_name, dst_col = t2, c2, t1, c1
                    src_meta, dst_meta = col2, col1
                else:
                    src_name, src_col, dst_name, dst_col = t1, c1, t2, c2
                    src_meta, dst_meta = col1, col2
                if not fk_infer_value_types_compatible(src_meta, dst_meta):
                    continue
                if not fk_overlap_validates(src_meta, dst_meta):
                    continue
                inferred.append(
                    FKEdge(
                        src_table=src_name,
                        src_cols=[src_col],
                        dst_table=dst_name,
                        dst_cols=[dst_col],
                        inference_tag=InferenceTag.SUFFIX,
                    )
                )
    return inferred


def _infer_missing_fks_composite(
    tables: dict[str, TableMetadata],
    tables_lower: dict[str, str],
    *,
    existing: list[FKEdge],
    skip_table_pairs: frozenset[frozenset[str]] = frozenset(),
) -> list[FKEdge]:
    """Infer composite foreign keys when a source table contains every column of a target's composite PK. A candidate composite FK requires: - The target table has a primary key with two or more columns. - The source table contains every PK column by exact case-insensitive name and none of those columns are already declared (or just inferred via the suffix pass) as part of a foreign key. - Every per-column pair passes ``fk_infer_value_types_compatible``; the string ↔ digit-only-string relaxation participates here too. - The source table is not the target table (self-referential composite FKs are rejected). Sample-overlap validation is intentionally omitted because per-column ``top_k_values`` are not row-aligned and a faithful tuple overlap would require additional profiling queries."""
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
            if frozenset({src_name, dst_real}) in skip_table_pairs:
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
                if not fk_infer_value_types_compatible(src_col_meta, dst_col_meta):
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


def recompute_join_paths_multi(
    tables: dict[str, TableMetadata],
) -> dict[str, dict[str, list[list[dict[str, Any]]]]]:
    """Recompute ``join_paths_multi`` from current ``TableMetadata`` FK. edges."""
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
        adj[t] = sorted(adj[t], key=lambda x: edge_key(x))

    tlist = sorted(tables.keys())
    return compute_join_paths_multi_from_adj(adj, tlist)


def promote_same_component_semantic_edges(
    sg: SchemaGraph,
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
) -> int:
    """Emit semantic FK shortcuts within an already-connected structural component."""
    return promote_semantic_neighbor_pairs(
        sg,
        cross_component_only=False,
        inference_tag=InferenceTag.SEMANTIC,
        blocked=blocked,
    )


def infer_missing_fks(
    tables: dict[str, TableMetadata],
    *,
    blocked: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = frozenset(),
    restrict_tables: frozenset[str] | None = None,
    skip_table_pairs: frozenset[frozenset[str]] = frozenset(),
) -> list[FKEdge]:
    """Infer missing foreign keys from column naming conventions and same-name overlap."""
    if not tables:
        return []
    if restrict_tables is not None:
        tables = {k: v for k, v in tables.items() if k in restrict_tables}
        if not tables:
            return []
    tables_lower = fk_tables_lower_index(tables)
    suffix_inferred = _infer_missing_fks_suffix(tables, tables_lower, skip_table_pairs=skip_table_pairs)
    same_name_inferred = _infer_missing_fks_same_name(tables, tables_lower, skip_table_pairs=skip_table_pairs)
    composite_inferred = _infer_missing_fks_composite(
        tables, tables_lower, existing=suffix_inferred + same_name_inferred, skip_table_pairs=skip_table_pairs
    )
    candidates = suffix_inferred + same_name_inferred + composite_inferred
    if not blocked:
        return candidates
    return [e for e in candidates if (e.src_table, tuple(e.src_cols), e.dst_table, tuple(e.dst_cols)) not in blocked]


set_schema_helpers(compute_schema_stats, compute_database_feature_capability)
