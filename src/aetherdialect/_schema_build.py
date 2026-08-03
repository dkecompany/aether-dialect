"""Reflect catalogs, build schema graphs with diff, and dialect load entrypoints."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._contracts_base import FederationManifest
    from ._contracts_schema import SchemaGraph

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import UniqueConstraint

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    JSON_COMPACT_SEPARATORS,
    MYSQL_INDEX_STATISTICS_SQL,
    MYSQL_PARTITION_EXPRESSIONS_SQL,
    POSTGRESQL_PARTITION_KEY_COLUMNS_SQL,
    REDSHIFT_INFORMATION_SCHEMA_UNIQUE_COLUMNS_SQL,
    REDSHIFT_SVV_FOREIGN_KEYS_SQL,
    SCHEMA_BUILD_PHASE_A,
    SCHEMA_BUILD_PHASE_C,
    SCHEMA_OVERRIDES_SIDECAR_FILENAME,
    SQLSERVER_PARTITION_KEY_COLUMNS_SQL,
    SQLSERVER_UNIQUE_INDEX_COLUMNS_SQL,
)
from ._contracts_base import (
    ConfigError,
    EngineContext,
    FederationManifest,
    PkInferenceTag,
    SchemaAccessError,
    SchemaInclude,
    SchemaInvariantError,
    resolve_federation_qualified_ref,
)
from ._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from ._core_utils import debug, schema_hash_fp, stable_json
from ._schema_catalog import apply_catalog_descriptions_from_tables_meta, parse_sql_file
from ._schema_graph import (
    allow_objects_lower_set,
    apply_deny_objects_filter,
    assign_schema_graph_hashes,
    catalog_fk_graph_is_connected,
    compute_join_paths_multi_from_adj,
    edge_key,
    expanded_scope_sql_file,
    load_pg_enum_values,
    pair_targeted_fk_inference,
    recompute_join_paths_multi,
    schema_context_from_graph,
    table_from_dict,
    table_to_dict,
    unify_reflected_schema_graph,
)


def apply_full_build_deny_objects(sg: SchemaGraph, deny_objects: frozenset[str] | None) -> SchemaGraph:
    """Remove denied relations on the structural reflection path before profiling."""
    if deny_objects:
        apply_deny_objects_filter(sg, EngineContext(deny_objects=deny_objects))
    return sg


def overrides_sidecar_path(schema_json_path: str | Path) -> Path:
    """Return the canonical sidecar location for *schema_json_path*'s overrides document."""
    parent = Path(schema_json_path).expanduser().resolve().parent
    return Path(parent / SCHEMA_OVERRIDES_SIDECAR_FILENAME)


def _split_qualified_endpoint_text(
    text: str,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> tuple[str, str] | None:
    """Split one ``table.column`` or ``source.table.column`` endpoint into table and column."""
    raw = str(text).strip()
    if not raw:
        return None
    if manifest is not None:
        try:
            resolved = resolve_federation_qualified_ref(
                raw, manifest=manifest, schema=schema, source_by_table=source_by_table
            )
        except ConfigError:
            return None
        return resolved.table, resolved.column
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(raw)
    if three:
        return three.group(2), three.group(3)
    two = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(raw)
    if two:
        return two.group(1), two.group(2)
    parts = raw.rsplit(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def split_fk_endpoint(
    endpoint: Any,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> tuple[str, list[str]] | None:
    """Split a ``table.column`` or ``source.table.column`` shorthand into ``(table, [cols])``."""
    if isinstance(endpoint, str):
        split = _split_qualified_endpoint_text(
            endpoint, manifest=manifest, schema=schema, source_by_table=source_by_table
        )
        if split is None:
            return None
        return split[0], [split[1]]
    if isinstance(endpoint, list) and endpoint:
        table_name: str | None = None
        cols: list[str] = []
        for ep in endpoint:
            if not isinstance(ep, str):
                return None
            split = _split_qualified_endpoint_text(
                ep, manifest=manifest, schema=schema, source_by_table=source_by_table
            )
            if split is None:
                return None
            if table_name is None:
                table_name = split[0]
            elif table_name != split[0]:
                return None
            cols.append(split[1])
        if table_name is None:
            return None
        return table_name, cols
    if isinstance(endpoint, dict):
        tbl = str(endpoint.get("table", "") or "")
        cols_raw = endpoint.get("columns") or endpoint.get("column") or []
        if isinstance(cols_raw, str):
            cols = [cols_raw]
        elif isinstance(cols_raw, list):
            cols = [str(c) for c in cols_raw if c]
        else:
            return None
        if not tbl or not cols:
            return None
        return tbl, cols
    return None


def read_sidecar_internal(schema_json_path: str | Path) -> dict[str, Any] | None:
    path = overrides_sidecar_path(schema_json_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        debug(f"[schema_build.read_sidecar_internal] ignoring sidecar {path!s}: {exc!r}")
        return None
    if not isinstance(payload, dict):
        return None
    internal = payload.get("_internal", {}) or {}
    return internal if isinstance(internal, dict) else None


def load_inference_block_lists(
    schema_json_path: str | Path | None,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> tuple[
    frozenset[tuple[str, str]],
    frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
]:
    """Read the overrides sidecar and return parsed PK/FK inference block sets."""
    if schema_json_path is None:
        return frozenset(), frozenset()
    internal = read_sidecar_internal(schema_json_path)
    if internal is None:
        return frozenset(), frozenset()
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
        src = split_fk_endpoint(entry.get("from"), manifest=manifest, schema=schema, source_by_table=source_by_table)
        dst = split_fk_endpoint(entry.get("to"), manifest=manifest, schema=schema, source_by_table=source_by_table)
        if src is None or dst is None:
            continue
        fk_keys.add((src[0], tuple(src[1]), dst[0], tuple(dst[1])))
    return frozenset(pk_pairs), frozenset(fk_keys)


def _collect_unique_columns_from_reflected_table(t: Any) -> set[str]:
    """Single-source aggregator for single-column uniqueness signals on a reflected SQLAlchemy table. Merges hits from both ``UniqueConstraint`` declarations and unique indexes so neither source can independently mark a column unique without going through this helper. Composite (multi-column) constraints/indexes do not imply per-column uniqueness and are skipped."""
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
    load_pg_enums: bool | None = None,
) -> SchemaGraph:
    """Reflect a database schema using SQLAlchemy and build a join-path. graph."""
    if schema_name is None:
        schema_name = EngineConfig.RUNTIME.SCHEMA if hasattr(EngineConfig.RUNTIME, "SCHEMA") else "public"

    debug(f"[{SCHEMA_BUILD_PHASE_C}]  reflecting schema '{schema_name}' object_kind={object_kind}")
    insp = inspect(engine)
    md = MetaData(schema=schema_name)
    allow_lower = allow_objects_lower_set(allow_objects)
    _, fk_blocked = load_inference_block_lists(schema_json_path)
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

    debug(f"[{SCHEMA_BUILD_PHASE_C}]  found {len(tables)} relations")

    for t in md.tables.values():
        unique_cols = _collect_unique_columns_from_reflected_table(t)
        for col_name in unique_cols:
            if col_name in tables[t.name].columns:
                tables[t.name].columns[col_name].is_unique = True

    if object_kind == "table":
        for t in md.tables.values():
            existing_keys = {edge_key(e) for e in tables[t.name].foreign_keys}
            for fk in t.foreign_key_constraints:
                e = FKEdge(
                    src_table=t.name,
                    src_cols=[el.parent.name for el in fk.elements],
                    dst_table=fk.elements[0].column.table.name,
                    dst_cols=[el.column.name for el in fk.elements],
                )
                ek = edge_key(e)
                if ek in existing_keys:
                    continue
                existing_keys.add(ek)
                tables[t.name].foreign_keys.append(e)

                debug(
                    f"[{SCHEMA_BUILD_PHASE_C}]  explicit FK: {e.src_table}.{e.src_cols[0]} -> "
                    f"{e.dst_table}.{e.dst_cols[0]}"
                )

        fk_count = sum(len(tbl.foreign_keys) for tbl in tables.values())
        debug(f"[{SCHEMA_BUILD_PHASE_C}]  found {fk_count} foreign key edges")

        tmp_sg = SchemaGraph(tables=tables, join_paths_multi={})
        if not catalog_fk_graph_is_connected(tmp_sg):
            inferred_ct = pair_targeted_fk_inference(tmp_sg, blocked=fk_blocked)
            if inferred_ct:
                debug(
                    f"[{SCHEMA_BUILD_PHASE_C}]  pair-targeted inference added {inferred_ct} inferred FK edge(s) "
                    "from naming conventions"
                )

    else:
        tmp_sg = SchemaGraph(tables=tables, join_paths_multi={})
        if not catalog_fk_graph_is_connected(tmp_sg):
            inferred_ct = pair_targeted_fk_inference(tmp_sg, blocked=fk_blocked)
            if inferred_ct:
                debug(
                    f"[{SCHEMA_BUILD_PHASE_C}]  pair-targeted inference added {inferred_ct} inferred FK edge(s) on views"
                )

    enum_values: dict[str, list[str]] = {}
    if load_pg_enums is None:
        load_pg_enums = (EngineConfig.TYPE or "").strip().lower() in ("postgresql", "redshift")
    if load_pg_enums:
        enum_values = load_pg_enum_values(engine)

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
    for adj_key in adj:
        adj[adj_key] = sorted(adj[adj_key], key=lambda x: edge_key(x))

    debug(f"[{SCHEMA_BUILD_PHASE_C}] reflect_schema computing shortest join paths")
    tlist = sorted(tables.keys())
    join_paths_multi = compute_join_paths_multi_from_adj(adj, tlist)

    sg = SchemaGraph(
        tables=tables, join_paths_multi=join_paths_multi, created_at=datetime.now().isoformat(), enum_values=enum_values
    )

    return sg


def databricks_row_kind(table_type: str) -> Literal["table", "view"]:
    """Map an information_schema ``table_type`` string to a graph relation kind."""
    u = table_type.upper()
    if "VIEW" in u and "MATERIALIZED" not in u:
        return "view"
    return "table"


def tables_meta_to_schema_graph(
    tables_meta: dict[str, dict[str, Any]],
    *,
    object_kind: Literal["table", "view"] = "table",
    row_kind_by_table: dict[str, Literal["table", "view"]] | None = None,
) -> SchemaGraph:
    """Convert a raw table metadata dictionary to a ``SchemaGraph``."""
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
        enum_types = meta.get("column_enum_type_names", [])
        enum_labels = meta.get("column_enum_labels", [])
        generated_flags = meta.get("column_is_generated", [])
        identity_flags = meta.get("column_is_identity", [])
        original_labels = meta.get("original_column_labels", col_names)
        for i, col_name in enumerate(col_names):
            col_type = col_types[i] if i < len(col_types) else "UNKNOWN"
            if use_parsed_nullable and nullable_list is not None:
                is_nullable = bool(nullable_list[i])
            else:
                is_nullable = col_name not in pk_set
            if col_name in pk_set:
                is_nullable = False
            enum_type = enum_types[i] if i < len(enum_types) else None
            labels = enum_labels[i] if i < len(enum_labels) else []
            is_generated = generated_flags[i] if i < len(generated_flags) else False
            is_identity = identity_flags[i] if i < len(identity_flags) else False
            original_label = original_labels[i] if i < len(original_labels) else col_name
            columns[col_name] = ColumnMetadata(
                name=col_name,
                original_name=str(original_label) if str(original_label) != col_name else "",
                data_type=col_type,
                is_primary_key=col_name in pk_cols,
                is_foreign_key=False,
                fk_target=None,
                is_unique=col_name in uq_cols,
                is_generated=bool(is_generated),
                is_identity=bool(is_identity),
                is_nullable=is_nullable,
                enum_type_name=enum_type,
                value_overlap_sample=list(labels or []),
            )

        fk_edges = []
        for fk in meta.get("foreign_keys", []):
            edge = FKEdge(
                src_table=table_name, src_cols=fk["src_cols"], dst_table=fk["dst_table"], dst_cols=fk["dst_cols"]
            )
            fk_edges.append(edge)

        partition_cols = meta.get("partition_columns", [])
        table_original = str(meta.get("original_table_label", table_name) or table_name)
        tables[table_name] = TableMetadata(
            name=table_name,
            original_name=table_original if table_original != table_name else "",
            columns=columns,
            primary_key=pk_cols,
            foreign_keys=fk_edges,
            partition_columns=partition_cols,
            partition_type=meta.get("partition_type"),
            require_partition_filter=bool(meta.get("require_partition_filter", False)),
            clustering_fields=list(meta.get("clustering_fields", []) or []),
            clustering_key=meta.get("clustering_key"),
            distkey=meta.get("distkey"),
            sortkey=list(meta.get("sortkey", []) or []),
            diststyle=meta.get("diststyle"),
            indexed_columns=list(meta.get("indexed_columns", []) or []),
            size_mb=meta.get("size_mb"),
            encoded=meta.get("encoded"),
            kind=row_kind,
        )

    fk_count = sum(len(tbl.foreign_keys) for tbl in tables.values())
    debug(f"[{SCHEMA_BUILD_PHASE_C}]  {len(tables)} tables, {fk_count} FK edges")

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
    for adj_key in adj:
        adj[adj_key] = sorted(adj[adj_key], key=lambda x: edge_key(x))

    debug(f"[{SCHEMA_BUILD_PHASE_C}] tables_meta_to_schema_graph computing shortest join paths")

    tlist = sorted(tables.keys())
    join_paths_multi = compute_join_paths_multi_from_adj(adj, tlist)

    sg = SchemaGraph(
        tables=tables, join_paths_multi=join_paths_multi, created_at=datetime.now().isoformat(), enum_values={}
    )
    apply_catalog_descriptions_from_tables_meta(sg, tables_meta)
    unify_reflected_schema_graph(sg)
    assign_schema_graph_hashes(sg, EngineContext(), "")
    _assert_schema_invariants(sg)

    return sg


def _assert_schema_invariants(sg: SchemaGraph) -> None:
    """Verify the canonical containers on *sg* remain consistent with. their derived properties."""
    graph_tables = sg.tables
    for tname, tbl in graph_tables.items():
        if getattr(tbl, "_owner_graph", None) is not sg:
            raise SchemaInvariantError(f"table {tname!r} owner_graph back-reference is not the enclosing SchemaGraph")
        for pk_col in tbl.primary_key:
            if pk_col not in tbl.columns:
                raise SchemaInvariantError(f"primary-key column {tname}.{pk_col} not present in columns")
        for fk in tbl.foreign_keys:
            for sc in fk.src_cols:
                if sc not in tbl.columns:
                    raise SchemaInvariantError(f"FK src column missing: {fk.src_table}.{sc}")
            dst_tbl = graph_tables.get(fk.dst_table)
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
        deny_tbl = graph_tables.get(dtbl)
        if deny_tbl is None:
            raise SchemaInvariantError(f"deny_columns references unknown table: {dtbl}")
        for dc in dcols:
            if dc not in deny_tbl.columns:
                raise SchemaInvariantError(f"deny_columns references unknown column: {dtbl}.{dc}")
    bearers = getattr(sg, "_canonical_bearers", {}) or {}
    for key, (btbl, bcol) in bearers.items():
        bearer_tbl = graph_tables.get(btbl)
        if bearer_tbl is None or bcol not in bearer_tbl.columns:
            raise SchemaInvariantError(f"canonical bearer {btbl}.{bcol} is missing from schema")
        if bcol.lower() != key:
            raise SchemaInvariantError(f"canonical bearer index key {key!r} does not match column name {bcol!r}")


def resolve_graph_table_name(raw_name: str, graph_tables: set[str]) -> str | None:
    """Map a DDL or catalog table name to a key present in. *graph_tables*."""
    if raw_name in graph_tables:
        return raw_name
    lower_index = {t.lower(): t for t in graph_tables}
    return lower_index.get(raw_name.lower())


def _merge_ddl_primary_keys_into_schema_graph(sg: SchemaGraph, ddl_tables: dict[str, dict[str, Any]]) -> None:
    """Add primary-key columns from parsed DDL into *sg* when columns exist."""
    if not ddl_tables or not sg.tables:
        return
    graph_names = set(sg.tables.keys())
    for ddl_table, meta in ddl_tables.items():
        src_resolved = resolve_graph_table_name(ddl_table, graph_names)
        if not src_resolved:
            continue
        src_tbl = sg.tables[src_resolved]
        for pk_col in meta.get("primary_keys", []) or []:
            pk_name = str(pk_col)
            if pk_name not in src_tbl.columns:
                continue
            if pk_name not in src_tbl.primary_key:
                src_tbl.primary_key.append(pk_name)
            col_meta = src_tbl.columns[pk_name]
            col_meta.is_nullable = False
            if col_meta.pk_inference_tag is None:
                col_meta.pk_inference_tag = PkInferenceTag.DDL


def _merge_ddl_column_constraints_into_schema_graph(sg: SchemaGraph, ddl_tables: dict[str, dict[str, Any]]) -> None:
    """Apply UNIQUE and NOT NULL signals from parsed DDL onto *sg* when columns exist."""
    if not ddl_tables or not sg.tables:
        return
    graph_names = set(sg.tables.keys())
    for ddl_table, meta in ddl_tables.items():
        src_resolved = resolve_graph_table_name(ddl_table, graph_names)
        if not src_resolved:
            continue
        src_tbl = sg.tables[src_resolved]
        col_names = list(meta.get("column_names_original", []) or [])
        nullable_list = meta.get("column_is_nullable")
        if (
            isinstance(nullable_list, list)
            and len(nullable_list) == len(col_names)
            and all(isinstance(x, bool) for x in nullable_list)
        ):
            for col_name, is_nullable in zip(col_names, nullable_list, strict=False):
                if col_name not in src_tbl.columns:
                    continue
                if not is_nullable:
                    src_tbl.columns[col_name].is_nullable = False
        for uq_col in meta.get("unique_columns", []) or []:
            uq_name = str(uq_col)
            if uq_name in src_tbl.columns:
                src_tbl.columns[uq_name].is_unique = True


def merge_ddl_partition_columns_into_schema_graph(sg: SchemaGraph, ddl_tables: dict[str, dict[str, Any]]) -> None:
    """Merge partition column names from parsed DDL into *sg* when columns exist."""
    if not ddl_tables or not sg.tables:
        return
    graph_names = set(sg.tables.keys())
    for ddl_table, meta in ddl_tables.items():
        src_resolved = resolve_graph_table_name(ddl_table, graph_names)
        if not src_resolved:
            continue
        part_cols = meta.get("partition_columns") or []
        if not isinstance(part_cols, list) or not part_cols:
            continue
        src_tbl = sg.tables[src_resolved]
        validated: list[str] = []
        for col in part_cols:
            col_name = str(col)
            if col_name in src_tbl.columns and col_name not in validated:
                validated.append(col_name)
        if validated:
            src_tbl.partition_columns = validated


def merge_ddl_foreign_keys_into_schema_graph(sg: SchemaGraph, ddl_tables: dict[str, dict[str, Any]]) -> None:
    """Add FK edges from parsed DDL into *sg* when endpoints exist, then. refresh paths."""
    if not ddl_tables or not sg.tables:
        return
    graph_names = set(sg.tables.keys())
    for ddl_table, meta in ddl_tables.items():
        src_resolved = resolve_graph_table_name(ddl_table, graph_names)
        if not src_resolved:
            continue
        src_tbl = sg.tables[src_resolved]
        existing = {edge_key(e) for e in src_tbl.foreign_keys}
        for fk in meta.get("foreign_keys", []) or []:
            dst_raw = fk.get("dst_table", "")
            dst_resolved = resolve_graph_table_name(str(dst_raw), graph_names)
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
            edge = FKEdge(src_table=src_resolved, src_cols=src_cols, dst_table=dst_resolved, dst_cols=dst_cols)
            ek = edge_key(edge)
            if ek in existing:
                continue
            existing.add(ek)
            src_tbl.foreign_keys.append(edge)

    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    assign_schema_graph_hashes(sg, schema_context_from_graph(sg), sg.notes_sha256)


def enrich_postgresql_partition_columns(engine: Any, sg: SchemaGraph, *, schema_name: str | None = None) -> None:
    """Populate ``partition_columns`` on reflected PostgreSQL tables from ``pg_catalog``."""
    if not sg.tables or engine is None:
        return
    effective_schema = schema_name
    if effective_schema is None:
        runtime_schema = getattr(EngineConfig.RUNTIME, "SCHEMA", None)
        effective_schema = str(runtime_schema) if runtime_schema else "public"
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(POSTGRESQL_PARTITION_KEY_COLUMNS_SQL), {"s": effective_schema}).fetchall()
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}]  catalog_query_failed: {e}")
        return
    part_by_table: dict[str, list[str]] = {}
    for tname, cname in rows:
        name = str(tname)
        col = str(cname)
        cols = part_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    graph_names = set(sg.tables.keys())
    for table_name, part_cols in part_by_table.items():
        resolved = resolve_graph_table_name(table_name, graph_names)
        if resolved and part_cols:
            sg.tables[resolved].partition_columns = list(part_cols)


def load_or_create_schema_postgresql(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a `SchemaGraph` for PostgreSQL from a live database or SQL. file fallback."""
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] load_or_create_schema_postgresql reflecting_database")
        sidecar_path = schema_json_path if schema_json_path is not None else EngineConfig.SCHEMA_JSON_PATH
        sg = _reflect_schema(
            engine,
            object_kind="table" if include == "tables" else "view",
            allow_objects=allow_objects,
            schema_json_path=sidecar_path,
            load_pg_enums=True,
        )
        debug(f"[{SCHEMA_BUILD_PHASE_C}]  reflected: {len(sg.tables)} tables")
        if engine is not None and include == "tables":
            enrich_postgresql_partition_columns(engine, sg)
        sql_file_path = expanded_scope_sql_file(sql_file)
        if include == "tables" and sql_file_path and os.path.exists(sql_file_path) and sg.tables:
            ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
            if ddl_tables:
                merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
                merge_ddl_partition_columns_into_schema_graph(sg, ddl_tables)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}]  reflection_failed: {e}")
        sql_file_path = expanded_scope_sql_file(sql_file)

        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  parsing_sql_file: {sql_file_path}")
            tables_meta = parse_sql_file(Path(sql_file_path))

            if not tables_meta or len(tables_meta) == 0:
                raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e

            ok: Literal["table", "view"] = "table" if include != "views" else "view"
            filtered: dict[str, dict[str, Any]] = tables_meta
            allow_lower = allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
            return apply_full_build_deny_objects(tables_meta_to_schema_graph(filtered, object_kind=ok), deny_objects)
        raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def _effective_runtime_schema_name() -> str:
    """Return the schema or database name from the active runtime config."""
    runtime = EngineConfig.RUNTIME
    schema = getattr(runtime, "SCHEMA", None)
    if schema:
        return str(schema)
    database = getattr(runtime, "DATABASE", None)
    if database:
        return str(database)
    dataset = getattr(runtime, "DATASET", None)
    if dataset:
        return str(dataset)
    return "public"


def _schema_graph_from_sql_file_fallback(
    e: Exception,
    *,
    include: SchemaInclude,
    allow_objects: frozenset[str] | None,
    log_prefix: str,
    sql_file: str | None = None,
    deny_objects: frozenset[str] | None = None,
) -> SchemaGraph:
    """Parse ``EngineContext.sql_file`` when live catalog reflection fails."""
    sql_file_path = expanded_scope_sql_file(sql_file)
    if sql_file_path and os.path.exists(sql_file_path):
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} parsing_sql_file: {sql_file_path}")
        tables_meta = parse_sql_file(Path(sql_file_path))
        if not tables_meta:
            raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e
        ok: Literal["table", "view"] = "table" if include != "views" else "view"
        filtered: dict[str, dict[str, Any]] = tables_meta
        allow_lower = allow_objects_lower_set(allow_objects)
        if allow_lower is not None:
            filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
        return apply_full_build_deny_objects(tables_meta_to_schema_graph(filtered, object_kind=ok), deny_objects)
    raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def parse_mysql_enum_or_set_labels(column_type: str) -> tuple[str | None, list[str]]:
    """Extract enum or set kind and labels from a MySQL ``COLUMN_TYPE`` string."""
    raw = str(column_type or "")
    lower = raw.lower()
    if lower.startswith("enum("):
        return "enum", re.findall(r"'([^']*)'", raw)
    if lower.startswith("set("):
        return "set", re.findall(r"'([^']*)'", raw)
    return None, []


def parse_mysql_partition_columns(partition_expression: str) -> list[str]:
    """Return bare column names extracted from a MySQL ``PARTITION_EXPRESSION``."""
    raw = str(partition_expression or "").strip()
    if not raw:
        return []
    paren_match = re.search(r"\(([^)]+)\)\s*$", raw)
    if paren_match:
        inner = paren_match.group(1)
        cols = re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", inner)
        return list(dict.fromkeys(cols))
    single = re.search(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", raw)
    return [single.group(1)] if single else []


def _parse_mysql_partition_column(partition_expression: str) -> str | None:
    """Return the first bare column identifier from a MySQL partition expression."""
    cols = parse_mysql_partition_columns(partition_expression)
    return cols[0] if cols else None


def _fk_tables_meta_edge_key(
    table_name: str, src_cols: list[str], dst_table: str, dst_cols: list[str]
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """Return a dedupe key for a ``tables_meta`` foreign-key dict entry."""
    return (table_name, tuple(src_cols), dst_table, tuple(dst_cols))


def _append_tables_meta_foreign_key(
    tables_meta: dict[str, dict[str, Any]],
    table_name: str,
    *,
    src_cols: list[str],
    dst_table: str,
    dst_cols: list[str],
    seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> None:
    """Append one FK edge to ``tables_meta`` when *table_name* exists and the edge is new."""
    if table_name not in tables_meta:
        return
    key = _fk_tables_meta_edge_key(table_name, src_cols, dst_table, dst_cols)
    if key in seen:
        return
    seen.add(key)
    tables_meta[table_name]["foreign_keys"].append(
        {"src_cols": list(src_cols), "dst_table": dst_table, "dst_cols": list(dst_cols)}
    )


def _information_schema_nullable_flag(raw: Any) -> bool:
    """Return True when a catalog ``is_nullable`` token means NULL is allowed."""
    token = str(raw or "").strip().upper()
    if token in ("YES", "TRUE", "Y", "T"):
        return True
    if token in ("NO", "FALSE", "N", "F"):
        return False
    return True


def _accumulate_information_schema_fk_bucket(
    buckets: dict[tuple[str, str], dict[str, Any]],
    *,
    table_name: str,
    constraint_name: str,
    ordinal_position: int,
    src_col: str,
    dst_table: str,
    dst_col: str,
) -> None:
    """Accumulate one FK column pair into a composite constraint bucket."""
    key = (table_name, constraint_name)
    bucket = buckets.setdefault(key, {"dst_table": dst_table, "pairs": []})
    bucket["pairs"].append((ordinal_position, src_col, dst_col))


def _flush_information_schema_fk_buckets(
    tables_meta: dict[str, dict[str, Any]],
    buckets: dict[tuple[str, str], dict[str, Any]],
    *,
    seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> None:
    """Emit grouped composite FK edges from accumulated constraint buckets."""
    for (table_name, _constraint_name), data in buckets.items():
        pairs = sorted(data["pairs"], key=lambda x: x[0])
        if not pairs:
            continue
        src_cols = [p[1] for p in pairs]
        dst_cols = [p[2] for p in pairs]
        dst_table = str(data.get("dst_table") or "")
        if not dst_table:
            continue
        _append_tables_meta_foreign_key(
            tables_meta, table_name, src_cols=src_cols, dst_table=dst_table, dst_cols=dst_cols, seen=seen
        )


def _merge_svv_foreign_keys_into_tables_meta(tables_meta: dict[str, dict[str, Any]], svv_rows: list[Any]) -> None:
    """Merge Redshift ``svv_foreign_keys`` rows into ``tables_meta`` without duplicating ``information_schema`` edges."""
    seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tname, meta in tables_meta.items():
        for fk in meta.get("foreign_keys", []):
            seen.add(
                _fk_tables_meta_edge_key(
                    tname,
                    list(fk.get("src_cols", []) or []),
                    str(fk.get("dst_table", "")),
                    list(fk.get("dst_cols", []) or []),
                )
            )
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for row in svv_rows:
        constraint_name = str(row[0] or "")
        child_table = str(row[2] or "")
        child_col = str(row[3] or "")
        parent_table = str(row[5] or "")
        parent_col = str(row[6] or "")
        if not child_table or not child_col or not parent_table or not parent_col:
            continue
        buckets.setdefault((child_table, constraint_name), []).append((child_col, parent_table, parent_col))
    for (child_table, _constraint_name), pairs in buckets.items():
        src_cols = [p[0] for p in pairs]
        dst_table = pairs[0][1]
        dst_cols = [p[2] for p in pairs]
        _append_tables_meta_foreign_key(
            tables_meta, child_table, src_cols=src_cols, dst_table=dst_table, dst_cols=dst_cols, seen=seen
        )


def parse_redshift_sortkey_columns(sortkey1: str) -> list[str]:
    """Parse Redshift ``sortkey1`` into one or more sort-key column names."""
    raw = str(sortkey1 or "").strip()
    if not raw:
        return []
    upper = raw.upper()
    if upper.startswith("INTERLEAVED:") or upper.startswith("COMPOUND:"):
        tail = raw.split(":", 1)[1]
        return [part.strip() for part in tail.split(",") if part.strip()]
    return [raw]


def _parse_mysql_enum_labels(column_type: str) -> list[str]:
    """Extract enum labels from a MySQL ``COLUMN_TYPE`` string."""
    _, labels = parse_mysql_enum_or_set_labels(column_type)
    return labels


def _reflect_mysql_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect MySQL schema via ``information_schema`` queries."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT TABLE_NAME, TABLE_TYPE FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = :s AND TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
                "ORDER BY TABLE_NAME"
            ),
            {"s": schema_name},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE, IS_NULLABLE, "
                "COLUMN_TYPE, EXTRA, COLUMN_KEY, GENERATION_EXPRESSION "
                "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = :s "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            ),
            {"s": schema_name},
        ).fetchall()
        constraint_rows = conn.execute(
            text(
                "SELECT kcu.TABLE_NAME, tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION, "
                "kcu.COLUMN_NAME, tc.CONSTRAINT_TYPE, "
                "kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME "
                "FROM information_schema.TABLE_CONSTRAINTS tc "
                "JOIN information_schema.KEY_COLUMN_USAGE kcu "
                "  ON tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA "
                " AND tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
                "WHERE tc.TABLE_SCHEMA = :s "
                "AND tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE') "
                "ORDER BY kcu.TABLE_NAME, tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION"
            ),
            {"s": schema_name},
        ).fetchall()
        partition_rows = conn.execute(text(MYSQL_PARTITION_EXPRESSIONS_SQL), {"s": schema_name}).fetchall()
        statistics_rows = conn.execute(text(MYSQL_INDEX_STATISTICS_SQL), {"s": schema_name}).fetchall()
    enum_values: dict[str, list[str]] = {}
    partition_by_table: dict[str, list[str]] = {}
    for tname, part_expr, _part_method in partition_rows:
        name = str(tname)
        for part_col in parse_mysql_partition_columns(str(part_expr or "")):
            cols = partition_by_table.setdefault(name, [])
            if part_col not in cols:
                cols.append(part_col)
    indexed_by_table: dict[str, list[str]] = {}
    for tname, cname in statistics_rows:
        name = str(tname)
        col = str(cname)
        cols = indexed_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "column_enum_type_names": [],
            "column_enum_labels": [],
            "column_is_generated": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "partition_columns": list(partition_by_table.get(name, [])),
            "indexed_columns": list(indexed_by_table.get(name, [])),
        }
    for tname, cname, _ord, dtype, nullable, col_type, extra, column_key, generation_expr in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(_information_schema_nullable_flag(nullable))
        enum_kind, labels = parse_mysql_enum_or_set_labels(str(col_type or ""))
        meta["column_enum_type_names"].append(enum_kind)
        meta["column_enum_labels"].append(labels)
        extra_text = str(extra or "").lower()
        is_generated = bool(str(generation_expr or "").strip()) or "generated" in extra_text
        meta["column_is_generated"].append(is_generated)
        if labels:
            enum_values[str(cname).lower()] = labels
        if "auto_increment" in extra_text and str(column_key or "").upper() == "PRI":
            pk_col = str(cname)
            if pk_col not in meta["primary_keys"]:
                meta["primary_keys"].append(pk_col)
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tname, cname, ord_pos, col, ctype, ref_t, ref_c in constraint_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        col_name = str(col)
        if str(ctype) == "PRIMARY KEY" and col_name not in meta["primary_keys"]:
            meta["primary_keys"].append(col_name)
        elif str(ctype) == "UNIQUE" and col_name not in meta["unique_columns"]:
            meta["unique_columns"].append(col_name)
        elif str(ctype) == "FOREIGN KEY" and ref_t and ref_c:
            _accumulate_information_schema_fk_bucket(
                fk_buckets,
                table_name=name,
                constraint_name=str(cname),
                ordinal_position=int(ord_pos or 0),
                src_col=col_name,
                dst_table=str(ref_t),
                dst_col=str(ref_c),
            )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    row_by = None
    sg = tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=row_by)
    if enum_values:
        sg.enum_values = enum_values
    return sg


def _reflect_redshift_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect Redshift schema via ``information_schema``, ``SVV_TABLE_INFO``, and ``svv_foreign_keys``."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_schema = :s ORDER BY table_name"
            ),
            {"s": schema_name},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT table_name, column_name, ordinal_position, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_schema = :s "
                "ORDER BY table_name, ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        fk_rows = conn.execute(
            text(
                "SELECT tc.table_name, tc.constraint_name, kcu.ordinal_position, "
                "kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                " AND tc.table_schema = kcu.table_schema "
                "JOIN information_schema.constraint_column_usage ccu "
                "  ON ccu.constraint_name = tc.constraint_name "
                " AND ccu.table_schema = tc.table_schema "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'FOREIGN KEY' "
                "ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        pk_rows = conn.execute(
            text(
                "SELECT kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                " AND tc.table_schema = kcu.table_schema "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'PRIMARY KEY' "
                "ORDER BY kcu.table_name, kcu.ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        try:
            svv_rows = conn.execute(
                text('SELECT "table", diststyle, sortkey1, size, encoded FROM svv_table_info WHERE "schema" = :s'),
                {"s": schema_name},
            ).fetchall()
        except Exception:
            svv_rows = []
        try:
            uq_rows = conn.execute(text(REDSHIFT_INFORMATION_SCHEMA_UNIQUE_COLUMNS_SQL), {"s": schema_name}).fetchall()
        except Exception:
            uq_rows = []
        try:
            svv_fk_rows = conn.execute(text(REDSHIFT_SVV_FOREIGN_KEYS_SQL), {"s": schema_name}).fetchall()
        except Exception:
            svv_fk_rows = []
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    svv_by_table: dict[str, tuple[Any, ...]] = {str(r[0]): r for r in svv_rows}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = "VIEW" in str(ttype).upper()
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        svv = svv_by_table.get(name)
        distkey = None
        sortkey: list[str] = []
        diststyle = None
        size_mb = None
        encoded = None
        if svv:
            diststyle = str(svv[1] or "") or None
            if diststyle and diststyle.upper().startswith("KEY") and len(diststyle.split()) > 1:
                distkey = diststyle.split()[1]
            sk = str(svv[2] or "")
            sortkey = parse_redshift_sortkey_columns(sk)
            try:
                size_mb = float(svv[3]) if svv[3] is not None else None
            except (TypeError, ValueError):
                size_mb = None
            encoded = bool(svv[4]) if svv[4] is not None else None
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "distkey": distkey,
            "diststyle": diststyle,
            "sortkey": sortkey,
            "size_mb": size_mb,
            "encoded": encoded,
        }
    for tname, cname, _ord, dtype, nullable in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(_information_schema_nullable_flag(nullable))
    for tname, cname in pk_rows:
        name = str(tname)
        if name in tables_meta and str(cname) not in tables_meta[name]["primary_keys"]:
            tables_meta[name]["primary_keys"].append(str(cname))
    for tname, cname in uq_rows:
        name = str(tname)
        if name in tables_meta and str(cname) not in tables_meta[name]["unique_columns"]:
            tables_meta[name]["unique_columns"].append(str(cname))
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tname, constraint_name, ord_pos, cname, ref_t, ref_c in fk_rows:
        _accumulate_information_schema_fk_bucket(
            fk_buckets,
            table_name=str(tname),
            constraint_name=str(constraint_name),
            ordinal_position=int(ord_pos or 0),
            src_col=str(cname),
            dst_table=str(ref_t),
            dst_col=str(ref_c),
        )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    _merge_svv_foreign_keys_into_tables_meta(tables_meta, list(svv_fk_rows))
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    sg = tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)
    sg.enum_values = load_pg_enum_values(engine)
    return sg


def _reflect_duckdb_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect DuckDB schema via ``information_schema`` including ``KEY_COLUMN_USAGE`` FK edges."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_schema = :s AND table_type IN ('BASE TABLE', 'VIEW') "
                "ORDER BY table_name"
            ),
            {"s": schema_name},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT table_name, column_name, ordinal_position, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_schema = :s "
                "ORDER BY table_name, ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        pk_rows = conn.execute(
            text(
                "SELECT kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_schema = kcu.constraint_schema "
                " AND tc.constraint_name = kcu.constraint_name "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'PRIMARY KEY' "
                "ORDER BY kcu.table_name, kcu.ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        uq_rows = conn.execute(
            text(
                "SELECT kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_schema = kcu.constraint_schema "
                " AND tc.constraint_name = kcu.constraint_name "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'UNIQUE' "
                "ORDER BY kcu.table_name, kcu.ordinal_position"
            ),
            {"s": schema_name},
        ).fetchall()
        fk_rows: list[Any] = []
        try:
            fk_rows = conn.execute(
                text(
                    "SELECT kcu.table_name, tc.constraint_name, kcu.ordinal_position, "
                    "kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column "
                    "FROM information_schema.table_constraints tc "
                    "JOIN information_schema.key_column_usage kcu "
                    "  ON tc.constraint_name = kcu.constraint_name "
                    " AND tc.table_schema = kcu.table_schema "
                    "JOIN information_schema.constraint_column_usage ccu "
                    "  ON ccu.constraint_name = tc.constraint_name "
                    " AND ccu.table_schema = tc.table_schema "
                    "WHERE tc.table_schema = :s AND tc.constraint_type = 'FOREIGN KEY' "
                    "ORDER BY kcu.table_name, tc.constraint_name, kcu.ordinal_position"
                ),
                {"s": schema_name},
            ).fetchall()
        except Exception as exc:
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  FK reflection skipped: {exc!r}")
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
        }
    for tname, cname, _ord, dtype, nullable in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(_information_schema_nullable_flag(nullable))
    for tname, cname in pk_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        col = str(cname)
        if col not in tables_meta[name]["primary_keys"]:
            tables_meta[name]["primary_keys"].append(col)
    for tname, cname in uq_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        col = str(cname)
        if col not in tables_meta[name]["unique_columns"]:
            tables_meta[name]["unique_columns"].append(col)
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tname, constraint_name, ord_pos, cname, ref_t, ref_c in fk_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        if ref_t and ref_c:
            _accumulate_information_schema_fk_bucket(
                fk_buckets,
                table_name=name,
                constraint_name=str(constraint_name),
                ordinal_position=int(ord_pos or 0),
                src_col=str(cname),
                dst_table=str(ref_t),
                dst_col=str(ref_c),
            )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def _reflect_sqlite_catalog(
    engine: Any, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect SQLite schema from ``sqlite_master`` and ``PRAGMA foreign_key_list`` when FK enforcement is on."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        fk_enabled = bool(conn.execute(text("PRAGMA foreign_keys")).scalar())
        table_rows = conn.execute(
            text(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ).fetchall()
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).lower() == "view"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
        }
    with engine.connect() as conn:
        for tname in list(tables_meta.keys()):
            info_rows = conn.execute(text(f'PRAGMA table_info("{tname}")')).fetchall()
            meta = tables_meta[tname]
            for info in info_rows:
                cname = str(info[1])
                meta["column_names_original"].append(cname)
                meta["column_types"].append(str(info[2]))
                meta["column_is_nullable"].append(int(info[3] or 0) == 0)
                if int(info[5] or 0) == 1 and cname not in meta["primary_keys"]:
                    meta["primary_keys"].append(cname)
        if fk_enabled:
            fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
            for tname in tables_meta:
                fk_rows = conn.execute(text(f'PRAGMA foreign_key_list("{tname}")')).fetchall()
                buckets: dict[int, list[tuple[int, str, str, str]]] = {}
                for fk in fk_rows:
                    fk_id = int(fk[0])
                    seq = int(fk[1] or 0)
                    buckets.setdefault(fk_id, []).append((seq, str(fk[3]), str(fk[2]), str(fk[4])))
                for pairs in buckets.values():
                    pairs.sort(key=lambda x: x[0])
                    src_cols = [p[1] for p in pairs]
                    dst_table = pairs[0][2]
                    dst_cols = [p[3] for p in pairs]
                    _append_tables_meta_foreign_key(
                        tables_meta, tname, src_cols=src_cols, dst_table=dst_table, dst_cols=dst_cols, seen=fk_seen
                    )
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def _reflect_sqlserver_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect SQL Server schema via ``sys.*`` catalog views."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        rel_rows = conn.execute(
            text(
                "SELECT t.name, 'TABLE' AS kind FROM sys.tables t "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id WHERE s.name = :s "
                "UNION ALL "
                "SELECT v.name, 'VIEW' AS kind FROM sys.views v "
                "JOIN sys.schemas s ON v.schema_id = s.schema_id WHERE s.name = :s "
                "ORDER BY 1"
            ),
            {"s": schema_name},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT o.name, c.name, ty.name, c.is_nullable, c.is_identity, c.is_computed, "
                "cc.definition "
                "FROM sys.columns c "
                "JOIN sys.objects o ON c.object_id = o.object_id "
                "JOIN sys.schemas s ON o.schema_id = s.schema_id "
                "JOIN sys.types ty ON c.user_type_id = ty.user_type_id "
                "LEFT JOIN sys.computed_columns cc "
                "  ON c.object_id = cc.object_id AND c.column_id = cc.column_id "
                "WHERE s.name = :s ORDER BY o.name, c.column_id"
            ),
            {"s": schema_name},
        ).fetchall()
        pk_rows = conn.execute(
            text(
                "SELECT t.name, c.name FROM sys.indexes i "
                "JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
                "JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
                "JOIN sys.tables t ON i.object_id = t.object_id "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "WHERE i.is_primary_key = 1 AND s.name = :s ORDER BY t.name, ic.key_ordinal"
            ),
            {"s": schema_name},
        ).fetchall()
        fk_rows = conn.execute(
            text(
                "SELECT fk.name, tp.name, fkc.constraint_column_id, cp.name, tr.name, cr.name "
                "FROM sys.foreign_keys fk "
                "JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id "
                "JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id "
                "JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id "
                " AND fkc.parent_column_id = cp.column_id "
                "JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id "
                "JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id "
                " AND fkc.referenced_column_id = cr.column_id "
                "JOIN sys.schemas s ON tp.schema_id = s.schema_id "
                "WHERE s.name = :s "
                "ORDER BY tp.name, fk.name, fkc.constraint_column_id"
            ),
            {"s": schema_name},
        ).fetchall()
        index_rows = conn.execute(
            text(
                "SELECT t.name, c.name, i.type_desc, i.filter_definition "
                "FROM sys.indexes i "
                "JOIN sys.index_columns ic "
                "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
                "JOIN sys.columns c "
                "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
                "JOIN sys.tables t ON i.object_id = t.object_id "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "WHERE s.name = :s AND i.type IN (1, 2, 5, 6) "
                "ORDER BY t.name, i.name, ic.key_ordinal"
            ),
            {"s": schema_name},
        ).fetchall()
        uq_rows = conn.execute(text(SQLSERVER_UNIQUE_INDEX_COLUMNS_SQL), {"s": schema_name}).fetchall()
        try:
            partition_rows = conn.execute(text(SQLSERVER_PARTITION_KEY_COLUMNS_SQL), {"s": schema_name}).fetchall()
        except Exception:
            partition_rows = []
    indexed_by_table: dict[str, list[str]] = {}
    for tname, cname, _type_desc, _filter_def in index_rows:
        name = str(tname)
        col = str(cname)
        cols = indexed_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    partition_by_table: dict[str, list[str]] = {}
    for tname, cname, _param_id in partition_rows:
        name = str(tname)
        col = str(cname)
        cols = partition_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, kind in rel_rows:
        name = str(tname)
        is_view = str(kind).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "column_is_generated": [],
            "column_is_identity": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "partition_columns": list(partition_by_table.get(name, [])),
            "indexed_columns": list(indexed_by_table.get(name, [])),
        }
    for tname, cname, dtype, nullable, is_identity, is_computed, _definition in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(bool(nullable))
        meta["column_is_generated"].append(bool(is_computed))
        meta["column_is_identity"].append(bool(is_identity))
    for tname, cname in pk_rows:
        name = str(tname)
        if name in tables_meta and str(cname) not in tables_meta[name]["primary_keys"]:
            tables_meta[name]["primary_keys"].append(str(cname))
    for tname, cname in uq_rows:
        name = str(tname)
        if name in tables_meta and str(cname) not in tables_meta[name]["unique_columns"]:
            tables_meta[name]["unique_columns"].append(str(cname))
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for fk_name, tname, ord_id, cname, ref_t, ref_c in fk_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        _accumulate_information_schema_fk_bucket(
            fk_buckets,
            table_name=name,
            constraint_name=str(fk_name),
            ordinal_position=int(ord_id or 0),
            src_col=str(cname),
            dst_table=str(ref_t),
            dst_col=str(ref_c),
        )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def _reflect_snowflake_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect Snowflake schema via ``INFORMATION_SCHEMA`` queries."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT TABLE_NAME, TABLE_TYPE FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = :s ORDER BY TABLE_NAME"
            ),
            {"s": schema_name.upper()},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE, IS_NULLABLE "
                "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :s "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            ),
            {"s": schema_name.upper()},
        ).fetchall()
        constraint_rows = conn.execute(
            text(
                "SELECT tc.TABLE_NAME, tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION, "
                "kcu.COLUMN_NAME, tc.CONSTRAINT_TYPE, "
                "kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME "
                "FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
                "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
                "  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
                " AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA "
                "WHERE tc.TABLE_SCHEMA = :s "
                "AND tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE') "
                "ORDER BY tc.TABLE_NAME, tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION"
            ),
            {"s": schema_name.upper()},
        ).fetchall()
        try:
            cluster_rows = conn.execute(
                text("SELECT TABLE_NAME, CLUSTERING_KEY FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = :s"),
                {"s": schema_name.upper()},
            ).fetchall()
        except Exception:
            cluster_rows = []
    cluster_by_table = {str(r[0]): str(r[1]) if r[1] is not None else None for r in cluster_rows}
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        ck = cluster_by_table.get(name)
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "clustering_key": ck,
        }
    for tname, cname, _ord, dtype, nullable in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(_information_schema_nullable_flag(nullable))
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tname, cname, ord_pos, col, ctype, ref_t, ref_c in constraint_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        col_name = str(col)
        if str(ctype) == "PRIMARY KEY" and col_name not in meta["primary_keys"]:
            meta["primary_keys"].append(col_name)
        elif str(ctype) == "UNIQUE" and col_name not in meta["unique_columns"]:
            meta["unique_columns"].append(col_name)
        elif str(ctype) == "FOREIGN KEY" and ref_t and ref_c:
            _accumulate_information_schema_fk_bucket(
                fk_buckets,
                table_name=name,
                constraint_name=str(cname),
                ordinal_position=int(ord_pos or 0),
                src_col=col_name,
                dst_table=str(ref_t),
                dst_col=str(ref_c),
            )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def _reflect_bigquery_catalog(
    engine: Any, dataset: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect BigQuery schema via ``INFORMATION_SCHEMA``. BigQuery does not expose enforced FK metadata; join edges for multi- table queries must come from ``EngineContext.sql_file`` DDL and/or operator ``foreign_keys_add`` overrides (suffix/composite/semantic inference still runs during profiling when PK anchors exist)."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT table_name, table_type FROM INFORMATION_SCHEMA.TABLES "
                "WHERE table_schema = :s ORDER BY table_name"
            ),
            {"s": dataset},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT table_name, column_name, ordinal_position, data_type, is_nullable "
                "FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = :s "
                "ORDER BY table_name, ordinal_position"
            ),
            {"s": dataset},
        ).fetchall()
        try:
            part_rows = conn.execute(
                text(
                    "SELECT table_name, partitioning_column, partition_type "
                    "FROM INFORMATION_SCHEMA.PARTITIONS WHERE table_schema = :s "
                    "GROUP BY table_name, partitioning_column, partition_type"
                ),
                {"s": dataset},
            ).fetchall()
        except Exception:
            part_rows = []
        try:
            req_rows = conn.execute(
                text(
                    "SELECT table_name, require_partition_filter, clustering_fields "
                    "FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = :s"
                ),
                {"s": dataset},
            ).fetchall()
        except Exception:
            req_rows = []
    part_by_table: dict[str, tuple[str | None, str | None]] = {}
    for tname, pcol, ptype in part_rows:
        part_by_table[str(tname)] = (str(pcol) if pcol else None, str(ptype) if ptype else None)
    req_by_table: dict[str, tuple[bool, list[str]]] = {}
    for tname, req, cluster in req_rows:
        fields = [str(x).strip() for x in str(cluster or "").split(",") if str(x).strip()]
        req_by_table[str(tname)] = (bool(req), fields)
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, Literal["table", "view"]] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = "view" if is_view else "table"
        pcol, ptype = part_by_table.get(name, (None, None))
        req, cluster_fields = req_by_table.get(name, (False, []))
        partition_columns = [pcol] if pcol else []
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "partition_columns": partition_columns,
            "partition_type": ptype,
            "require_partition_filter": req,
            "clustering_fields": cluster_fields,
        }
    for tname, cname, _ord, dtype, nullable in col_rows:
        name = str(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        meta["column_names_original"].append(str(cname))
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(_information_schema_nullable_flag(nullable))
    row_kind: Literal["table", "view"] = "table" if include != "views" else "view"
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def _merge_schema_graph_sql_file_fks(
    sg: SchemaGraph, *, include: SchemaInclude, schema_json_path: str | Path | None, sql_file: str | None = None
) -> None:
    """Merge PK, column constraints, and FK hints from ``EngineContext.sql_file`` when tables are present."""
    sql_file_path = expanded_scope_sql_file(sql_file)
    if include == "tables" and sql_file_path and os.path.exists(sql_file_path) and sg.tables:
        ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
        if ddl_tables:
            _merge_ddl_primary_keys_into_schema_graph(sg, ddl_tables)
            _merge_ddl_column_constraints_into_schema_graph(sg, ddl_tables)
            merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
            merge_ddl_partition_columns_into_schema_graph(sg, ddl_tables)


def _load_or_create_schema_sqlalchemy(
    engine: Any,
    *,
    schema_name: str | None = None,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    log_prefix: str,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` via SQLAlchemy reflection with SQL file fallback."""
    effective_schema = schema_name if schema_name is not None else _effective_runtime_schema_name()
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sidecar_path = schema_json_path if schema_json_path is not None else EngineConfig.SCHEMA_JSON_PATH
        sg = _reflect_schema(
            engine,
            schema_name=effective_schema,
            object_kind="table" if include == "tables" else "view",
            allow_objects=allow_objects,
            schema_json_path=sidecar_path,
        )
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        sql_file_path = expanded_scope_sql_file(sql_file)
        if include == "tables" and sql_file_path and os.path.exists(sql_file_path) and sg.tables:
            ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
            if ddl_tables:
                merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        sql_file_path = expanded_scope_sql_file(sql_file)
        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} parsing_sql_file: {sql_file_path}")
            tables_meta = parse_sql_file(Path(sql_file_path))
            if not tables_meta:
                raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e
            ok: Literal["table", "view"] = "table" if include != "views" else "view"
            filtered: dict[str, dict[str, Any]] = tables_meta
            allow_lower = allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
            return apply_full_build_deny_objects(tables_meta_to_schema_graph(filtered, object_kind=ok), deny_objects)
        raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def load_or_create_schema_mysql(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for MySQL from live reflection or SQL file fallback."""
    effective_schema = _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_mysql"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_mysql_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_duckdb(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a DuckDB schema graph from ``information_schema.KEY_COLUMN_USAGE`` or SQL-file DDL fallback."""
    effective_schema = _effective_runtime_schema_name() or "main"
    log_prefix = "load_or_create_schema_duckdb"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_duckdb_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_sqlite(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a SQLite schema graph from ``sqlite_master`` and ``PRAGMA foreign_key_list`` when FK enforcement is on."""
    log_prefix = "load_or_create_schema_sqlite"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_sqlite_catalog(engine, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_redshift(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for Redshift from live reflection or SQL file fallback."""
    effective_schema = _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_redshift"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_redshift_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_sqlserver(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for SQL Server from live reflection or SQL file fallback."""
    effective_schema = _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_sqlserver"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_sqlserver_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_snowflake(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for Snowflake from live reflection or SQL file fallback."""
    effective_schema = _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_snowflake"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_snowflake_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def load_or_create_schema_bigquery(
    engine: Any,
    *,
    include: SchemaInclude = "tables",
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for BigQuery from live reflection or SQL file fallback. Catalog reflection never yields FK edges; supply ``EngineContext.sql_file`` and/or ``foreign_keys_add`` overrides for multi-table join graphs."""
    effective_schema = _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_bigquery"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_bigquery_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflected: {len(sg.tables)} tables")
        _merge_schema_graph_sql_file_fks(sg, include=include, schema_json_path=schema_json_path, sql_file=sql_file)
        return apply_full_build_deny_objects(sg, deny_objects)
    except Exception as e:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflection_failed: {e}")
        return _schema_graph_from_sql_file_fallback(
            e,
            include=include,
            allow_objects=allow_objects,
            log_prefix=log_prefix,
            sql_file=sql_file,
            deny_objects=deny_objects,
        )


def schema_cache_json_blob(cache_data: dict[str, Any]) -> str:
    """Serialize schema cache data as one compact JSON document."""
    return json.dumps(cache_data, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=True)


def tables_payload_through_model_round_trip(tables_json: dict[str, Any]) -> dict[str, Any]:
    """Rebuild table dicts by parsing into TableMetadata and. serializing. back to plain dicts."""
    return {name: table_to_dict(table_from_dict(blob)) for name, blob in tables_json.items()}


def fingerprint_tables_after_document_round_trip(cache_data: dict[str, Any]) -> str:
    """Compute ``schema_hash_fp`` for ``tables`` after an in-memory. write/parse of the full document."""
    reparsed = json.loads(schema_cache_json_blob(cache_data))
    return str(schema_hash_fp(reparsed["tables"]))


def first_table_where_stable_json_differs(left_tables: dict[str, Any], right_tables: dict[str, Any]) -> str | None:
    """Return the first table name whose single-slot stable JSON. differs. between mappings."""
    names = sorted(set(left_tables) | set(right_tables))
    for name in names:
        left_json = stable_json({name: left_tables.get(name)})
        right_json = stable_json({name: right_tables.get(name)})
        if left_json != right_json:
            return name
    return None


def debug_clip_stable_json(label: str, payload: Any) -> None:
    """Emit one debug line with a clipped ``stable_json`` rendering of. payload."""
    clip = PolicyConfig.SCHEMA_CACHE_HASH_DEBUG_CLIP_CHARS
    text = stable_json(payload)
    if len(text) <= clip:
        debug(f"[{SCHEMA_BUILD_PHASE_A}] cache_hash_debug {label} chars={len(text)} body={text}")
    else:
        debug(f"[{SCHEMA_BUILD_PHASE_A}] cache_hash_debug {label} chars={len(text)} head={text[:clip]!r}")
