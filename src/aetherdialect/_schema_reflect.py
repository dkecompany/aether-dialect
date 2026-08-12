"""Reflect catalogs, build schema graphs with diff, and dialect load entrypoints."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import sqlglot
from sqlglot import expressions as exp

from ._schema_graph import (
    allow_objects_lower_set,
    apply_allow_objects_filter,
    apply_deny_objects_filter,
    assign_schema_graph_hashes,
    catalog_fk_graph_is_connected,
    compute_join_paths_multi_from_adj,
    compute_semantic_profile_join_neighbors,
    edge_key,
    effective_reflect_mode,
    expanded_scope_sql_file,
    load_pg_enum_values,
    merge_reflected_schema_graphs,
    pair_targeted_fk_inference,
    recompute_join_paths_multi,
    schema_context_from_graph,
    table_from_dict,
    table_to_dict,
    tables_profiling_payload,
    unify_reflected_schema_graph,
)
from ._schema_profile import (
    apply_catalog_descriptions_from_tables_meta,
    cursor_rows_as_dicts,
    extract_partition_columns_from_describe_detail_spark,
    extract_partition_columns_from_describe_detail_sql_connector,
    parse_catalog_constraints_from_ddl,
    parse_partition_columns_from_create_stmt,
    parse_sql_file,
    tables_meta_foreign_key_dicts_from_edges,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._contracts_schema import FederationManifest, SchemaGraph

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import UniqueConstraint

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    ARTIFACT_MANIFEST_FILENAME,
    DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH,
    DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    JSON_COMPACT_SEPARATORS,
    MYSQL_CONNECTION_CHARSET,
    MYSQL_INDEX_STATISTICS_SQL,
    MYSQL_PARTITION_EXPRESSIONS_SQL,
    POSTGRESQL_PARTITION_KEY_COLUMNS_SQL,
    REDSHIFT_INFORMATION_SCHEMA_UNIQUE_COLUMNS_SQL,
    REDSHIFT_SVV_FOREIGN_KEYS_SQL,
    SCHEMA_JOIN_PATH_ENUMERATION_VERSION,
    SQLSERVER_PARTITION_KEY_COLUMNS_SQL,
    SQLSERVER_UNIQUE_INDEX_COLUMNS_SQL,
    STRUCTURE_SIDECAR_FILENAME,
)
from ._constants_runtime import (
    SCHEMA_BUILD_PHASE_A,
    SCHEMA_BUILD_PHASE_C,
)
from ._contracts_base import (
    ConfigError,
    EngineContext,
    ReflectMode,
    ResolvedQualifiedRef,
    SchemaAccessError,
    SchemaInclude,
    SchemaInvariantError,
    TableKind,
)
from ._contracts_schema import (
    CatalogStructuralConstraintsIndex,
    ColumnMetadata,
    DescriptionOwner,
    FederationManifest,
    FederationMappings,
    FKEdge,
    InferenceTag,
    PkInferenceTag,
    SchemaGraph,
    TableMetadata,
)
from ._intent_expr import (
    intent_join_reachability_tables,
)
from ._utils import (
    active_engine_identity,
    bound_engine_runtime_config,
    build_case_folded_index,
    collation_name_is_case_insensitive,
    column_is_unsigned_from_data_type,
    column_numeric_metadata_from_data_type,
    column_timezone_aware_from_data_type,
    debug,
    notify,
    profiling_hash_fp,
    schema_hash_fp,
    stable_json,
)
from ._utils_artifacts import (
    artifact_lock,
    read_artifact_manifest,
    write_artifact_manifest,
    write_gzip_json_atomic,
)


def table_user_display_name(table: TableMetadata) -> str:
    """Return the user-facing label for a reflected relation."""
    description = (table.description or "").strip()
    if description:
        return description
    original = (table.original_name or "").strip()
    if original:
        return original
    return table.name


def column_user_display_name(col: ColumnMetadata) -> str:
    """Return the user-facing label for a reflected column."""
    description = (col.description or "").strip()
    if description:
        return description
    original = (col.original_name or "").strip()
    if original:
        return original
    return col.name


def _federation_table_source_index(
    schema: Any, mappings: FederationMappings, manifest: FederationManifest | None = None
) -> dict[str, str]:
    """Map composite table names to a single owning member source when unambiguous."""
    index: dict[str, str] = {}
    logical_member_sources: dict[str, set[str]] = {}
    for lt in mappings.logical_tables:
        logical_member_sources[lt.logical] = {m.source for m in lt.members}
    tables = getattr(schema, "tables", None) or {}
    for name, table in tables.items():
        member_source_ids = getattr(table, "member_source_ids", None) or ()
        if member_source_ids:
            if len(member_source_ids) == 1:
                index[name] = member_source_ids[0]
            continue
        source_id = getattr(table, "source_id", "") or ""
        if source_id:
            index[name] = source_id
        elif name in logical_member_sources:
            sources = logical_member_sources[name]
            if len(sources) == 1:
                index[name] = next(iter(sources))
    if manifest is not None:
        for name, source_id in manifest.table_namespace.items():
            if name in tables and name not in index:
                index[name] = source_id
    return index


def tables_referenced_in_view_definition(definition: str | None) -> frozenset[str]:
    """Return bare relation names referenced in a view definition SQL string."""
    text_def = (definition or "").strip()
    if not text_def:
        return frozenset()
    try:
        tree = sqlglot.parse_one(text_def, read="postgres")
    except Exception:
        return frozenset()
    names: set[str] = set()
    for node in tree.walk():
        if isinstance(node, exp.Table) and node.name:
            names.add(str(node.name))
    return frozenset(names)


def project_base_descriptions_onto_views(tables_graph: SchemaGraph, views_graph: SchemaGraph) -> int:
    """Project base-table ``base_description`` values onto matching view columns by name."""
    copied = 0
    for view_name, view in views_graph.tables.items():
        referenced: tuple[str, ...] = ()
        definition = str(getattr(view, "view_definition", "") or "").strip()
        if definition:
            referenced = tuple(tables_referenced_in_view_definition(definition))
        base_candidates = [tables_graph.tables[n] for n in referenced if n in tables_graph.tables]
        if not base_candidates:
            base_candidates = list(tables_graph.tables.values())
        view_base = (view.base_description or "").strip()
        if not view_base and view.description_owner != DescriptionOwner.USER_OVERRIDE:
            view.base_description = f"View {view_name}"
            view.description = view.base_description
            view.description_owner = DescriptionOwner.LLM_REFINEMENT
            copied += 1
        for col_name, view_col in view.columns.items():
            if view_col.description_owner == DescriptionOwner.USER_OVERRIDE:
                continue
            matched = ""
            for base_table in base_candidates:
                base_col = base_table.columns.get(col_name)
                if base_col is None:
                    continue
                text = (base_col.base_description or "").strip()
                if text:
                    matched = text
                    break
            if not matched:
                continue
            view_col.base_description = matched
            view_col.description = matched
            view_col.description_owner = DescriptionOwner.LLM_REFINEMENT
            copied += 1
    return copied


def _view_column_index(table: TableMetadata) -> dict[str, str]:
    """Map lowercased column name to canonical column key on *table*."""
    return {str(c).lower(): str(c) for c in table.columns}


def view_fully_covers_table(view: TableMetadata, base: TableMetadata) -> bool:
    """Return True when *view* references *base* and exposes every base column name."""
    refs = {str(r).lower() for r in tables_referenced_in_view_definition(view.view_definition)}
    if base.name.lower() not in refs:
        return False
    view_cols = {c.lower() for c in view.columns}
    return all(str(c).lower() in view_cols for c in base.columns)


def reflect_private_base_catalog(
    dialect: Any,
    public_sg: SchemaGraph,
    ctx: EngineContext,
) -> dict[str, TableMetadata]:
    """Reflect private base-table metadata for view lineage (not inserted into the public graph)."""
    public_table_names = {n.lower() for n, t in public_sg.tables.items() if t.kind == TableKind.TABLE}
    need: set[str] = set()
    for tbl in public_sg.tables.values():
        if tbl.kind not in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW):
            continue
        for ref in tables_referenced_in_view_definition(tbl.view_definition):
            if ref.lower() not in public_table_names:
                need.add(ref)
    if not need:
        return {}
    try:
        base_sg = dialect.reflect_schema_graph(
            include=SchemaInclude.TABLES,
            allow_objects=frozenset(need),
            deny_objects=None,
            sql_file=ctx.sql_file,
        )
    except Exception:
        return {}
    return dict(base_sg.tables)


def apply_cover_reduction(sg: SchemaGraph) -> int:
    """Remove base tables fully covered by a view in the public graph (mixed-kind scopes only)."""
    has_table = any(t.kind == TableKind.TABLE for t in sg.tables.values())
    has_view = any(t.kind in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW) for t in sg.tables.values())
    if not (has_table and has_view):
        return 0
    views = sorted(
        (t for t in sg.tables.values() if t.kind in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW)),
        key=lambda t: t.name,
    )
    tables = sorted((t for t in sg.tables.values() if t.kind == TableKind.TABLE), key=lambda t: t.name)
    rep_for: dict[str, str] = {}
    for tbl in tables:
        covering = [v.name for v in views if view_fully_covers_table(v, tbl)]
        if covering:
            rep_for[tbl.name] = sorted(covering)[0]
    if not rep_for:
        return 0
    for tbl_name, rep_view in rep_for.items():
        rep_tbl = sg.tables.get(rep_view)
        if rep_tbl is None:
            continue
        for src_tbl in sg.tables.values():
            kept: list[FKEdge] = []
            for fk in src_tbl.foreign_keys:
                if fk.dst_table == tbl_name:
                    dst_cols = list(fk.dst_cols)
                    resolved_dst = []
                    for dc in dst_cols:
                        canon = _view_column_index(rep_tbl).get(str(dc).lower())
                        if canon is None:
                            break
                        resolved_dst.append(canon)
                    else:
                        kept.append(
                            FKEdge(
                                src_table=fk.src_table,
                                src_cols=list(fk.src_cols),
                                dst_table=rep_view,
                                dst_cols=resolved_dst,
                                inference_tag=fk.inference_tag,
                            )
                        )
                        continue
                if fk.src_table == tbl_name:
                    src_cols = list(fk.src_cols)
                    resolved_src = []
                    for sc in src_cols:
                        canon = _view_column_index(rep_tbl).get(str(sc).lower())
                        if canon is None:
                            break
                        resolved_src.append(canon)
                    else:
                        kept.append(
                            FKEdge(
                                src_table=rep_view,
                                src_cols=resolved_src,
                                dst_table=fk.dst_table,
                                dst_cols=list(fk.dst_cols),
                                inference_tag=fk.inference_tag,
                            )
                        )
                        continue
                kept.append(fk)
            src_tbl.foreign_keys = kept
        del sg.tables[tbl_name]
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    return len(rep_for)


def _resolve_fk_destination(
    sg: SchemaGraph,
    dst_table: str,
    dst_cols: list[str],
    all_bases: dict[str, TableMetadata],
) -> tuple[str, list[str]] | None:
    """Resolve a projected FK destination against the public graph or a covering view."""
    tindex = {n.lower(): n for n in sg.tables}
    dst_lower = str(dst_table).lower()
    dst_canon = tindex.get(dst_lower)
    if dst_canon is not None:
        pub = sg.tables[dst_canon]
        if pub.kind == TableKind.TABLE:
            resolved_dst: list[str] = []
            for dc in dst_cols:
                canon = _view_column_index(pub).get(str(dc).lower())
                if canon is None:
                    return None
                resolved_dst.append(canon)
            return dst_canon, resolved_dst
    dst_base = all_bases.get(dst_lower)
    if dst_base is None:
        return None
    pk_cols = list(dst_base.primary_key) or list(dst_cols)
    views = sorted(
        (t for t in sg.tables.values() if t.kind in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW)),
        key=lambda t: t.name,
    )
    for view in views:
        if not view_fully_covers_table(view, dst_base):
            continue
        resolved_dst = []
        for pk in pk_cols:
            canon = _view_column_index(view).get(str(pk).lower())
            if canon is None:
                break
            resolved_dst.append(canon)
        else:
            return view.name, resolved_dst
    return None


def project_view_keys_from_bases(sg: SchemaGraph, private_bases: dict[str, TableMetadata]) -> int:
    """Project base-table PK/FK metadata onto views using view- definition lineage."""
    all_bases: dict[str, TableMetadata] = {n.lower(): t for n, t in private_bases.items()}
    for name, tbl in sg.tables.items():
        if tbl.kind == TableKind.TABLE:
            all_bases[name.lower()] = tbl
    added = 0
    for vname, view in sg.tables.items():
        if view.kind not in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW):
            continue
        if not (view.view_definition or "").strip():
            continue
        ref_names = tables_referenced_in_view_definition(view.view_definition)
        bases: list[TableMetadata] = []
        for ref in ref_names:
            base = all_bases.get(ref.lower())
            if base is not None and base.kind == TableKind.TABLE:
                bases.append(base)
        pk_union: set[str] = set()
        for base in bases:
            pk_cols = list(base.primary_key)
            if not pk_cols:
                continue
            resolved_pk: list[str] = []
            for pk in pk_cols:
                canon = _view_column_index(view).get(str(pk).lower())
                if canon is None:
                    break
                resolved_pk.append(canon)
            else:
                pk_union.update(resolved_pk)
        for pk_col in sorted(pk_union):
            if pk_col not in view.primary_key:
                view.primary_key.append(pk_col)
        for base in bases:
            for fk in base.foreign_keys:
                if fk.src_table != base.name:
                    continue
                resolved_src: list[str] = []
                for sc in fk.src_cols:
                    canon = _view_column_index(view).get(str(sc).lower())
                    if canon is None:
                        break
                    resolved_src.append(canon)
                else:
                    dst = _resolve_fk_destination(sg, fk.dst_table, list(fk.dst_cols), all_bases)
                    if dst is None:
                        continue
                    dst_table, dst_cols = dst
                    if any(
                        e.dst_table == dst_table and e.src_cols == resolved_src and e.dst_cols == dst_cols
                        for e in view.foreign_keys
                    ):
                        continue
                    view.foreign_keys.append(
                        FKEdge(
                            src_table=vname,
                            src_cols=resolved_src,
                            dst_table=dst_table,
                            dst_cols=dst_cols,
                            inference_tag=InferenceTag.VIEW_LINEAGE,
                        )
                    )
                    added += 1
    if added:
        sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    return added


def apply_view_scope_postprocess(sg: SchemaGraph, dialect: Any, ctx: EngineContext) -> None:
    """Run private-base reflection, cover reduction, and PK/FK projection for view scopes."""
    has_views = any(t.kind in (TableKind.VIEW, TableKind.MATERIALIZED_VIEW) for t in sg.tables.values())
    if not has_views:
        return
    private_bases = reflect_private_base_catalog(dialect, sg, ctx)
    apply_cover_reduction(sg)
    project_view_keys_from_bases(sg, private_bases)


def reflect_schema_graph_for_context(dialect: Any, ctx: EngineContext) -> SchemaGraph:
    """Build a structural schema graph for *ctx* using the effective reflection mode."""
    active_engine_identity()
    mode = effective_reflect_mode(ctx)
    sql_file = ctx.sql_file
    if mode == ReflectMode.SINGLE_KIND:
        return cast(
            SchemaGraph,
            dialect.reflect_schema_graph(
                include=ctx.include,
                allow_objects=None,
                deny_objects=None,
                sql_file=sql_file,
            ),
        )
    sg_tables = dialect.reflect_schema_graph(
        include=SchemaInclude.TABLES,
        allow_objects=None,
        deny_objects=None,
        sql_file=sql_file,
    )
    sg_views = dialect.reflect_schema_graph(
        include=SchemaInclude.VIEWS,
        allow_objects=None,
        deny_objects=None,
        sql_file=sql_file,
    )
    sg = merge_reflected_schema_graphs(sg_tables, sg_views)
    if mode == ReflectMode.ALLOW_LIST:
        apply_allow_objects_filter(sg, ctx)
    apply_deny_objects_filter(sg, ctx, strict=bool(ctx.deny_objects))
    return sg


def _normalize_refresh_timestamp(value: Any) -> str | None:
    """Coerce an engine-provided refresh timestamp to a stable ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _reflect_view_definition_text(insp: Any, table_name: str, schema_name: str | None) -> str:
    """Read a view definition from the SQLAlchemy inspector when exposed."""
    getter = getattr(insp, "get_view_definition", None)
    if not callable(getter):
        return ""
    try:
        return str(getter(table_name, schema=schema_name) or "")
    except Exception:
        return ""


def _reflect_materialized_view_last_refreshed_at(insp: Any, table_name: str, schema_name: str | None) -> str | None:
    """Read a materialized-view last-refresh timestamp when the engine exposes one."""
    for attr in ("get_materialized_view_last_refresh", "get_matview_last_refresh"):
        getter = getattr(insp, attr, None)
        if not callable(getter):
            continue
        try:
            return _normalize_refresh_timestamp(getter(table_name, schema=schema_name))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            continue
    return None


def emit_materialized_view_answer_diagnostics(intent: Any, schema: SchemaGraph) -> None:
    """Emit ``MATERIALIZED_VIEW_ANSWER`` when a turn reads from a materialized view."""
    seen: set[str] = set()
    for table_name in intent_join_reachability_tables(intent):
        if table_name in seen:
            continue
        table_meta = schema.tables.get(table_name)
        if table_meta is None or table_meta.kind != "materialized_view":
            continue
        seen.add(table_name)
        label = table_user_display_name(table_meta)
        message = f"Answer uses materialized view {label!r}."
        if table_meta.last_refreshed_at:
            message = f"Answer uses materialized view {label!r} (last refreshed {table_meta.last_refreshed_at})."
        notify(
            message,
            stage="execution",
            code=DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER,
            level="info",
            details=(
                ("table", table_name),
                ("label", label),
                ("last_refreshed_at", table_meta.last_refreshed_at or ""),
            ),
        )


def _emit_column_charset_mismatch_if_needed(
    *,
    table_name: str,
    column_name: str,
    character_set: str | None,
    connection_charset: str = MYSQL_CONNECTION_CHARSET,
) -> None:
    """Emit ``COLUMN_CHARSET_MISMATCH`` when a reflected charset differs from the connection default."""
    if not character_set:
        return
    if str(character_set).casefold() == str(connection_charset).casefold():
        return
    notify(
        (
            f"column {table_name}.{column_name} uses character set {character_set!r}, "
            f"which differs from the connection charset {connection_charset!r}"
        ),
        stage="schema",
        code=DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH,
        level="warning",
        details=(
            ("table", table_name),
            ("column", column_name),
            ("character_set", str(character_set)),
            ("connection_charset", str(connection_charset)),
        ),
    )


def _column_collation_overlap_fields(collation: str | None) -> tuple[bool, str]:
    """Derive overlap comparison fields from a reflected collation name."""
    if not collation:
        return False, "exact"
    is_case_insensitive = collation_name_is_case_insensitive(str(collation))
    return is_case_insensitive, "case_folded" if is_case_insensitive else "exact"


def apply_full_build_deny_objects(sg: SchemaGraph, deny_objects: frozenset[str] | None) -> SchemaGraph:
    """Remove denied relations on the structural reflection path before profiling."""
    if deny_objects:
        apply_deny_objects_filter(sg, EngineContext(deny_objects=deny_objects))
    return sg


def structure_sidecar_path(schema_json_path: str | Path) -> Path:
    """Return the canonical sidecar location for *schema_json_path*'s overrides document."""
    parent = Path(schema_json_path).expanduser().resolve().parent
    return Path(parent / STRUCTURE_SIDECAR_FILENAME)


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
    path = structure_sidecar_path(schema_json_path)
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
    object_kind: TableKind = TableKind.TABLE,
    allow_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    load_pg_enums: bool | None = None,
    relation_name_filter: Callable[[str, Sequence[str]], Sequence[str]] | None = None,
) -> SchemaGraph:
    """Reflect a database schema using SQLAlchemy and build a join-path graph. Credential-scoped reflects use ``only=`` so MetaData does not pull FK targets outside the allowlist. Foreign-key edges whose destination table is outside this credential-visible reflect are omitted."""
    if schema_name is None:
        runtime_cfg = bound_engine_runtime_config()
        schema_name = getattr(runtime_cfg, "SCHEMA", None) or "public"

    debug(f"[{SCHEMA_BUILD_PHASE_C}]  reflecting schema '{schema_name}' object_kind={object_kind}")
    insp = inspect(engine)
    md = MetaData(schema=schema_name)
    allow_lower = allow_objects_lower_set(allow_objects)
    _, fk_blocked = load_inference_block_lists(schema_json_path)
    matview_names: frozenset[str] = frozenset()
    if object_kind == "table":
        names: set[str] = set(insp.get_table_names(schema=schema_name))
        gmv = getattr(insp, "get_materialized_view_names", None)
        if callable(gmv):
            try:
                matview_names = frozenset(str(n) for n in gmv(schema=schema_name))
                names |= set(matview_names)
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                pass
        ordered = sorted(names)
        if allow_lower is not None:
            ordered = [n for n in ordered if str(n).lower() in allow_lower]
        if ordered and relation_name_filter is not None:
            before = len(ordered)
            ordered = [str(n) for n in relation_name_filter(str(schema_name), ordered)]
            matview_names = frozenset(n for n in matview_names if n in ordered)
            if len(ordered) != before:
                debug(
                    f"[{SCHEMA_BUILD_PHASE_C}]  selectable-relation filter "
                    f"{before}->{len(ordered)} relations in schema {schema_name!r}"
                )
        if ordered:
            md.reflect(bind=engine, schema=schema_name, only=ordered, views=False, resolve_fks=False)
    else:
        ordered = sorted(insp.get_view_names(schema=schema_name))
        if allow_lower is not None:
            ordered = [n for n in ordered if str(n).lower() in allow_lower]
        if ordered and relation_name_filter is not None:
            before = len(ordered)
            ordered = [str(n) for n in relation_name_filter(str(schema_name), ordered)]
            if len(ordered) != before:
                debug(
                    f"[{SCHEMA_BUILD_PHASE_C}]  selectable-relation filter "
                    f"{before}->{len(ordered)} views in schema {schema_name!r}"
                )
        if ordered:
            md.reflect(bind=engine, schema=schema_name, only=ordered, views=True, resolve_fks=False)

    tables: dict[str, TableMetadata] = {}

    for t in md.tables.values():
        if t.name in matview_names:
            row_kind: TableKind = TableKind.MATERIALIZED_VIEW
        elif object_kind == "view":
            row_kind = TableKind.VIEW
        else:
            row_kind = TableKind.TABLE
        view_definition = _reflect_view_definition_text(insp, t.name, schema_name) if row_kind == "view" else ""
        last_refreshed_at = (
            _reflect_materialized_view_last_refreshed_at(insp, t.name, schema_name)
            if row_kind == "materialized_view"
            else None
        )
        columns: dict[str, ColumnMetadata] = {}
        engine_name = str(getattr(getattr(engine, "dialect", None), "name", "") or "")
        for c in t.columns:
            col_type_str = str(c.type)
            reflected_precision = getattr(c.type, "precision", None)
            reflected_scale = getattr(c.type, "scale", None)
            reflected_unsigned = getattr(c.type, "unsigned", None)
            numeric_precision, numeric_scale, is_exact_numeric = column_numeric_metadata_from_data_type(
                col_type_str,
                reflected_precision=reflected_precision,
                reflected_scale=reflected_scale,
            )
            is_unsigned = column_is_unsigned_from_data_type(
                col_type_str,
                reflected_unsigned=reflected_unsigned if reflected_unsigned is not None else None,
            )
            is_timezone_aware = column_timezone_aware_from_data_type(col_type_str, engine=engine_name)
            columns[c.name] = ColumnMetadata(
                name=c.name,
                data_type=col_type_str,
                is_primary_key=c.name in [pk.name for pk in t.primary_key.columns],
                is_foreign_key=False,
                fk_target=None,
                is_nullable=getattr(c, "nullable", True),
                numeric_precision=numeric_precision,
                numeric_scale=numeric_scale,
                is_exact_numeric=is_exact_numeric,
                is_unsigned=is_unsigned,
                is_timezone_aware=is_timezone_aware,
            )

        tables[t.name] = TableMetadata(
            name=t.name,
            columns=columns,
            primary_key=[c.name for c in t.primary_key.columns],
            foreign_keys=[],
            kind=row_kind,
            view_definition=view_definition,
            last_refreshed_at=last_refreshed_at,
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
            try:
                reflected_fks = insp.get_foreign_keys(t.name, schema=schema_name)
            except (AttributeError, NotImplementedError, TypeError, ValueError, OSError):
                reflected_fks = []
            for fk in reflected_fks:
                src_cols = [str(c) for c in list(fk.get("constrained_columns") or []) if c]
                dst_table = str(fk.get("referred_table") or "").strip()
                dst_cols = [str(c) for c in list(fk.get("referred_columns") or []) if c]
                if not src_cols or not dst_table or not dst_cols:
                    continue
                if dst_table not in tables:
                    continue
                e = FKEdge(
                    src_table=t.name,
                    src_cols=src_cols,
                    dst_table=dst_table,
                    dst_cols=dst_cols,
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


def databricks_row_kind(table_type: str) -> TableKind:
    """Map an information_schema ``table_type`` string to a graph relation kind."""
    u = table_type.upper()
    if "MATERIALIZED" in u and "VIEW" in u:
        return TableKind.MATERIALIZED_VIEW
    if "VIEW" in u:
        return TableKind.VIEW
    return TableKind.TABLE


def tables_meta_to_schema_graph(
    tables_meta: dict[str, dict[str, Any]],
    *,
    object_kind: TableKind = TableKind.TABLE,
    row_kind_by_table: dict[str, TableKind] | None = None,
    engine: str | None = None,
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
        character_sets = meta.get("column_character_sets", [])
        collations = meta.get("column_collations", [])
        original_labels = meta.get("original_column_labels", col_names)
        for i, col_name in enumerate(col_names):
            col_type = col_types[i] if i < len(col_types) else "UNKNOWN"
            numeric_precision, numeric_scale, is_exact_numeric = column_numeric_metadata_from_data_type(col_type)
            is_unsigned = column_is_unsigned_from_data_type(col_type)
            is_timezone_aware = column_timezone_aware_from_data_type(col_type, engine=engine)
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
            character_set = character_sets[i] if i < len(character_sets) else None
            collation = collations[i] if i < len(collations) else None
            is_case_insensitive_collation, overlap_comparison = _column_collation_overlap_fields(collation)
            original_label = original_labels[i] if i < len(original_labels) else col_name
            if engine == "mysql":
                _emit_column_charset_mismatch_if_needed(
                    table_name=table_name,
                    column_name=col_name,
                    character_set=character_set,
                )
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
                character_set=character_set,
                collation=collation,
                is_case_insensitive_collation=is_case_insensitive_collation,
                overlap_comparison=overlap_comparison,
                numeric_precision=numeric_precision,
                numeric_scale=numeric_scale,
                is_exact_numeric=is_exact_numeric,
                is_unsigned=is_unsigned,
                is_timezone_aware=is_timezone_aware,
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
            view_definition=str(meta.get("view_definition", "") or ""),
            last_refreshed_at=(str(meta.get("last_refreshed_at")).strip() if meta.get("last_refreshed_at") else None),
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
    """Verify the canonical containers on *sg* remain consistent with their derived properties."""
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
    lower_index = build_case_folded_index(graph_tables, kind="table")
    if raw_name in graph_tables:
        return raw_name
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
    """Add FK edges from parsed DDL into *sg* when endpoints exist, then refresh paths."""
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
        runtime_schema = getattr(bound_engine_runtime_config(), "SCHEMA", None)
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
    include: SchemaInclude = SchemaInclude.TABLES,
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
    schema_name: str | None = None,
    relation_name_filter: Callable[[str, Sequence[str]], Sequence[str]] | None = None,
) -> SchemaGraph:
    """Build a `SchemaGraph` for PostgreSQL from a live database or SQL file fallback."""
    resolved_schema = str(schema_name or "").strip() or _effective_runtime_schema_name()
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] load_or_create_schema_postgresql reflecting_database")
        sidecar_path = schema_json_path if schema_json_path is not None else EngineConfig.SCHEMA_JSON_PATH
        sg = _reflect_schema(
            engine,
            resolved_schema,
            object_kind=TableKind.TABLE if include == "tables" else TableKind.VIEW,
            allow_objects=allow_objects,
            schema_json_path=sidecar_path,
            load_pg_enums=True,
            relation_name_filter=relation_name_filter,
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
        _reraise_if_missing_engine_identity(e)
        sql_file_path = expanded_scope_sql_file(sql_file)

        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  parsing_sql_file: {sql_file_path}")
            tables_meta = parse_sql_file(Path(sql_file_path))

            if not tables_meta or len(tables_meta) == 0:
                raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e

            ok: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
            filtered: dict[str, dict[str, Any]] = tables_meta
            allow_lower = allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
            if relation_name_filter is not None and filtered:
                keep = {str(n) for n in relation_name_filter(resolved_schema, list(filtered))}
                if keep:
                    before = len(filtered)
                    filtered = {k: v for k, v in filtered.items() if k in keep}
                    debug(
                        f"[{SCHEMA_BUILD_PHASE_C}]  sql fallback selectable-relation filter "
                        f"{before}->{len(filtered)} relations"
                    )
            return apply_full_build_deny_objects(tables_meta_to_schema_graph(filtered, object_kind=ok), deny_objects)
        raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def _reraise_if_missing_engine_identity(exc: BaseException) -> None:
    if isinstance(exc, RuntimeError) and "no active engine identity" in str(exc):
        raise exc


def _effective_runtime_schema_name() -> str:
    """Return the schema or database name from the active runtime config."""
    runtime = bound_engine_runtime_config()
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
    _reraise_if_missing_engine_identity(e)
    sql_file_path = expanded_scope_sql_file(sql_file)
    if sql_file_path and os.path.exists(sql_file_path):
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} parsing_sql_file: {sql_file_path}")
        tables_meta = parse_sql_file(Path(sql_file_path))
        if not tables_meta:
            raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e
        ok: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
                "COLUMN_TYPE, EXTRA, COLUMN_KEY, GENERATION_EXPRESSION, "
                "CHARACTER_SET_NAME, COLLATION_NAME "
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
        view_def_rows: list[Any] = []
        try:
            view_def_rows = conn.execute(
                text("SELECT TABLE_NAME, VIEW_DEFINITION FROM information_schema.VIEWS WHERE TABLE_SCHEMA = :s"),
                {"s": schema_name},
            ).fetchall()
        except Exception as exc:
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  view definition reflection skipped: {exc!r}")
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
    table_kinds: dict[str, TableKind] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "column_enum_type_names": [],
            "column_enum_labels": [],
            "column_character_sets": [],
            "column_collations": [],
            "column_is_generated": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "partition_columns": list(partition_by_table.get(name, [])),
            "indexed_columns": list(indexed_by_table.get(name, [])),
            "view_definition": "",
        }
    for vname, vdef in view_def_rows:
        name = str(vname)
        if name in tables_meta:
            tables_meta[name]["view_definition"] = str(vdef or "")
    for (
        tname,
        cname,
        _ord,
        dtype,
        nullable,
        col_type,
        extra,
        column_key,
        generation_expr,
        charset,
        collation,
    ) in col_rows:
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
        meta["column_character_sets"].append(str(charset) if charset else None)
        meta["column_collations"].append(str(collation) if collation else None)
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
                "kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema WHERE tc.table_schema = :s AND tc.constraint_type = 'FOREIGN KEY' ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position"
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
    table_kinds: dict[str, TableKind] = {}
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
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
                    "kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema WHERE tc.table_schema = :s AND tc.constraint_type = 'FOREIGN KEY' ORDER BY kcu.table_name, tc.constraint_name, kcu.ordinal_position"
                ),
                {"s": schema_name},
            ).fetchall()
        except Exception as exc:
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  FK reflection skipped: {exc!r}")
        view_def_rows: list[Any] = []
        try:
            view_def_rows = conn.execute(
                text("SELECT view_name, sql FROM duckdb_views() WHERE schema_name = :s"),
                {"s": schema_name},
            ).fetchall()
        except Exception as exc:
            debug(f"[{SCHEMA_BUILD_PHASE_C}]  view definition reflection skipped: {exc!r}")
    tables_meta: dict[str, dict[str, Any]] = {}
    table_kinds: dict[str, TableKind] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
        tables_meta[name] = {
            "column_names_original": [],
            "column_types": [],
            "column_is_nullable": [],
            "primary_keys": [],
            "unique_columns": [],
            "foreign_keys": [],
            "view_definition": "",
        }
    for vname, vsql in view_def_rows:
        name = str(vname)
        if name in tables_meta:
            tables_meta[name]["view_definition"] = str(vsql or "")
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
    table_kinds: dict[str, TableKind] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).lower() == "view"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
                "  ON c.object_id = cc.object_id AND c.column_id = cc.column_id WHERE s.name = :s ORDER BY o.name, c.column_id"
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
                "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id JOIN sys.tables t ON i.object_id = t.object_id JOIN sys.schemas s ON t.schema_id = s.schema_id WHERE s.name = :s AND i.type IN (1, 2, 5, 6) ORDER BY t.name, i.name, ic.key_ordinal"
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
    table_kinds: dict[str, TableKind] = {}
    for tname, kind in rel_rows:
        name = str(tname)
        is_view = str(kind).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
    table_kinds: dict[str, TableKind] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
    table_kinds: dict[str, TableKind] = {}
    for tname, ttype in table_rows:
        name = str(tname)
        is_view = str(ttype).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name.lower() not in allow_lower:
            continue
        table_kinds[name.lower()] = TableKind.VIEW if is_view else TableKind.TABLE
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
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
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
    include: SchemaInclude = SchemaInclude.TABLES,
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    log_prefix: str,
    sql_file: str | None = None,
    relation_name_filter: Callable[[str, Sequence[str]], Sequence[str]] | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` via SQLAlchemy reflection with SQL file fallback."""
    effective_schema = schema_name if schema_name is not None else _effective_runtime_schema_name()
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sidecar_path = schema_json_path if schema_json_path is not None else EngineConfig.SCHEMA_JSON_PATH
        sg = _reflect_schema(
            engine,
            schema_name=effective_schema,
            object_kind=TableKind.TABLE if include == "tables" else TableKind.VIEW,
            allow_objects=allow_objects,
            schema_json_path=sidecar_path,
            relation_name_filter=relation_name_filter,
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
        _reraise_if_missing_engine_identity(e)
        sql_file_path = expanded_scope_sql_file(sql_file)
        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} parsing_sql_file: {sql_file_path}")
            tables_meta = parse_sql_file(Path(sql_file_path))
            if not tables_meta:
                raise SchemaAccessError("Both database reflection and SQL file parsing failed") from e
            ok: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
            filtered: dict[str, dict[str, Any]] = tables_meta
            allow_lower = allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                filtered = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
            if relation_name_filter is not None and filtered:
                keep = {str(n) for n in relation_name_filter(str(effective_schema), list(filtered))}
                if keep:
                    filtered = {k: v for k, v in filtered.items() if k in keep}
            return apply_full_build_deny_objects(tables_meta_to_schema_graph(filtered, object_kind=ok), deny_objects)
        raise SchemaAccessError(f"Database reflection failed and no SQL file available: {e}") from e


def load_or_create_schema_mysql(
    engine: Any,
    *,
    include: SchemaInclude = SchemaInclude.TABLES,
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
    include: SchemaInclude = SchemaInclude.TABLES,
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
    schema_name: str | None = None,
) -> SchemaGraph:
    """Build a DuckDB schema graph from ``information_schema.KEY_COLUMN_USAGE`` or SQL-file DDL fallback."""
    effective_schema = schema_name or _effective_runtime_schema_name() or "main"
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
    include: SchemaInclude = SchemaInclude.TABLES,
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
    include: SchemaInclude = SchemaInclude.TABLES,
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
    include: SchemaInclude = SchemaInclude.TABLES,
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


def _reflect_oracle_catalog(
    engine: Any, schema_name: str, *, include: SchemaInclude, allow_objects: frozenset[str] | None
) -> SchemaGraph:
    """Reflect Oracle schema via ``ALL_*`` dictionary views."""
    allow_lower = allow_objects_lower_set(allow_objects)
    want_views = include == "views"
    want_tables = include == "tables"
    owner = str(schema_name or "").upper()
    with engine.connect() as conn:
        rel_rows = conn.execute(
            text(
                "SELECT table_name AS name, 'TABLE' AS kind FROM all_tables WHERE owner = :s "
                "UNION ALL "
                "SELECT view_name AS name, 'VIEW' AS kind FROM all_views WHERE owner = :s "
                "ORDER BY 1"
            ),
            {"s": owner},
        ).fetchall()
        col_rows = conn.execute(
            text(
                "SELECT c.table_name, c.column_name, c.data_type, c.nullable, "
                "CASE WHEN i.column_name IS NOT NULL THEN 1 ELSE 0 END AS is_identity, CASE WHEN c.virtual_column = 'YES' THEN 1 ELSE 0 END AS is_generated FROM all_tab_cols c LEFT JOIN all_tab_identity_cols i ON i.owner = c.owner AND i.table_name = c.table_name AND i.column_name = c.column_name WHERE c.owner = :s AND c.hidden_column = 'NO' ORDER BY c.table_name, c.column_id"
            ),
            {"s": owner},
        ).fetchall()
        pk_rows = conn.execute(
            text(
                "SELECT cc.table_name, cc.column_name "
                "FROM all_constraints cons "
                "JOIN all_cons_columns cc "
                "  ON cons.owner = cc.owner AND cons.constraint_name = cc.constraint_name WHERE cons.owner = :s AND cons.constraint_type = 'P' ORDER BY cc.table_name, cc.position"
            ),
            {"s": owner},
        ).fetchall()
        uq_rows = conn.execute(
            text(
                "SELECT cc.table_name, cc.column_name "
                "FROM all_constraints cons "
                "JOIN all_cons_columns cc "
                "  ON cons.owner = cc.owner AND cons.constraint_name = cc.constraint_name WHERE cons.owner = :s AND cons.constraint_type = 'U' ORDER BY cc.table_name, cc.position"
            ),
            {"s": owner},
        ).fetchall()
        fk_rows = conn.execute(
            text(
                "SELECT cons.constraint_name, cc.table_name, cc.position, cc.column_name, "
                "rcc.table_name AS ref_table, rcc.column_name AS ref_column FROM all_constraints cons JOIN all_cons_columns cc ON cons.owner = cc.owner AND cons.constraint_name = cc.constraint_name JOIN all_constraints rcons ON cons.r_owner = rcons.owner AND cons.r_constraint_name = rcons.constraint_name JOIN all_cons_columns rcc ON rcons.owner = rcc.owner AND rcons.constraint_name = rcc.constraint_name AND cc.position = rcc.position WHERE cons.owner = :s AND cons.constraint_type = 'R' ORDER BY cc.table_name, cons.constraint_name, cc.position"
            ),
            {"s": owner},
        ).fetchall()
        index_rows = conn.execute(
            text(
                "SELECT ic.table_name, ic.column_name "
                "FROM all_indexes i "
                "JOIN all_ind_columns ic "
                "  ON i.owner = ic.index_owner AND i.index_name = ic.index_name WHERE i.table_owner = :s AND i.index_type IN ('NORMAL', 'NORMAL/REV', 'FUNCTION-BASED NORMAL') ORDER BY ic.table_name, i.index_name, ic.column_position"
            ),
            {"s": owner},
        ).fetchall()
        try:
            partition_rows = conn.execute(
                text(
                    "SELECT table_name, column_name "
                    "FROM all_part_key_columns "
                    "WHERE owner = :s AND object_type = 'TABLE' "
                    "ORDER BY table_name, column_position"
                ),
                {"s": owner},
            ).fetchall()
        except Exception:
            partition_rows = []

    def _oracle_fold(ident: str) -> str:
        return str(ident).lower()

    indexed_by_table: dict[str, list[str]] = {}
    for tname, cname in index_rows:
        name = _oracle_fold(tname)
        col = _oracle_fold(cname)
        cols = indexed_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    partition_by_table: dict[str, list[str]] = {}
    for tname, cname in partition_rows:
        name = _oracle_fold(tname)
        col = _oracle_fold(cname)
        cols = partition_by_table.setdefault(name, [])
        if col not in cols:
            cols.append(col)
    tables_meta: dict[str, dict[str, Any]] = {}
    for tname, kind in rel_rows:
        physical = str(tname)
        name = _oracle_fold(physical)
        is_view = str(kind).upper() == "VIEW"
        if is_view and not want_views:
            continue
        if not is_view and not want_tables:
            continue
        if allow_lower is not None and name not in allow_lower:
            continue
        tables_meta[name] = {
            "original_table_label": physical,
            "column_names_original": [],
            "original_column_labels": [],
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
    for tname, cname, dtype, nullable, is_identity, is_generated in col_rows:
        name = _oracle_fold(tname)
        if name not in tables_meta:
            continue
        meta = tables_meta[name]
        physical_col = str(cname)
        meta["column_names_original"].append(_oracle_fold(physical_col))
        meta["original_column_labels"].append(physical_col)
        meta["column_types"].append(str(dtype))
        meta["column_is_nullable"].append(str(nullable).upper() in {"Y", "YES", "1", "TRUE"})
        meta["column_is_generated"].append(bool(int(is_generated or 0)))
        meta["column_is_identity"].append(bool(int(is_identity or 0)))
    for tname, cname in pk_rows:
        name = _oracle_fold(tname)
        col = _oracle_fold(cname)
        if name in tables_meta and col not in tables_meta[name]["primary_keys"]:
            tables_meta[name]["primary_keys"].append(col)
    for tname, cname in uq_rows:
        name = _oracle_fold(tname)
        col = _oracle_fold(cname)
        if name in tables_meta and col not in tables_meta[name]["unique_columns"]:
            tables_meta[name]["unique_columns"].append(col)
    fk_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    fk_seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for fk_name, tname, ord_id, cname, ref_t, ref_c in fk_rows:
        name = _oracle_fold(tname)
        if name not in tables_meta:
            continue
        _accumulate_information_schema_fk_bucket(
            fk_buckets,
            table_name=name,
            constraint_name=str(fk_name),
            ordinal_position=int(ord_id or 0),
            src_col=_oracle_fold(cname),
            dst_table=_oracle_fold(ref_t),
            dst_col=_oracle_fold(ref_c),
        )
    _flush_information_schema_fk_buckets(tables_meta, fk_buckets, seen=fk_seen)
    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
    return tables_meta_to_schema_graph(tables_meta, object_kind=row_kind, row_kind_by_table=None)


def load_or_create_schema_oracle(
    engine: Any,
    *,
    include: SchemaInclude = SchemaInclude.TABLES,
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    schema_json_path: str | Path | None = None,
    sql_file: str | None = None,
    schema_name: str | None = None,
) -> SchemaGraph:
    """Build a ``SchemaGraph`` for Oracle from live reflection or SQL file fallback."""
    effective_schema = schema_name or _effective_runtime_schema_name()
    log_prefix = "load_or_create_schema_oracle"
    try:
        debug(f"[{SCHEMA_BUILD_PHASE_C}] {log_prefix} reflecting_database")
        sg = _reflect_oracle_catalog(engine, effective_schema, include=include, allow_objects=allow_objects)
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
    include: SchemaInclude = SchemaInclude.TABLES,
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
    include: SchemaInclude = SchemaInclude.TABLES,
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
    """Rebuild table dicts by parsing into TableMetadata and serializing back to plain dicts."""
    return {name: table_to_dict(table_from_dict(blob)) for name, blob in tables_json.items()}


def fingerprint_tables_after_document_round_trip(cache_data: dict[str, Any]) -> str:
    """Compute ``schema_hash_fp`` for ``tables`` after an in-memory write/parse of the full document."""
    reparsed = json.loads(schema_cache_json_blob(cache_data))
    return str(schema_hash_fp(reparsed["tables"]))


def first_table_where_stable_json_differs(left_tables: dict[str, Any], right_tables: dict[str, Any]) -> str | None:
    """Return the first table name whose single-slot stable JSON differs between mappings."""
    names = sorted(set(left_tables) | set(right_tables))
    for name in names:
        left_json = stable_json({name: left_tables.get(name)})
        right_json = stable_json({name: right_tables.get(name)})
        if left_json != right_json:
            return name
    return None


def debug_clip_stable_json(label: str, payload: Any) -> None:
    """Emit one debug line with a clipped ``stable_json`` rendering of payload."""
    clip = PolicyConfig.SCHEMA_CACHE_HASH_DEBUG_CLIP_CHARS
    text = stable_json(payload)
    if len(text) <= clip:
        debug(f"[{SCHEMA_BUILD_PHASE_A}] cache_hash_debug {label} chars={len(text)} body={text}")
    else:
        debug(f"[{SCHEMA_BUILD_PHASE_A}] cache_hash_debug {label} chars={len(text)} head={text[:clip]!r}")


def resolve_federation_qualified_ref(
    ref: str,
    *,
    manifest: FederationManifest,
    schema: Any | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> ResolvedQualifiedRef:
    """Resolve ``table.column`` or ``source.table.column``, erroring on ambiguity. Replica and authoritative composite tables pin ``source_id``; when that pin is set, member_source_ids are not re-expanded into the candidate set."""
    text = str(ref or "").strip()
    if not text:
        raise ConfigError("empty federation qualified reference")
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        source_id, table, column = three.group(1), three.group(2), three.group(3)
        declared_sources = {s.source_id for s in manifest.sources}
        if not declared_sources:
            declared_sources = set(manifest.table_namespace.values())
            if schema is not None:
                declared_sources |= {
                    str(getattr(table_meta, "source_id", "") or "").strip()
                    for table_meta in getattr(schema, "tables", {}).values()
                    if str(getattr(table_meta, "source_id", "") or "").strip()
                }
        if declared_sources and source_id not in declared_sources:
            raise ConfigError(f"unknown federation source in reference: {text!r}")
        ns_table = manifest.table_namespace.get(table, "")
        if ns_table and ns_table != source_id:
            raise ConfigError(
                f"qualified reference {text!r} disagrees with table_namespace ({table!r} -> {ns_table!r})"
            )
        if schema is not None:
            table_meta = tables.get(table) if (tables := getattr(schema, "tables", None)) else None
            if table_meta is not None:
                composite_source = getattr(table_meta, "source_id", "") or ""
                if composite_source and composite_source != source_id:
                    raise ConfigError(f"qualified reference {text!r} disagrees with composite source_id")
        return ResolvedQualifiedRef(source_id=source_id, table=table, column=column, qualified=f"{table}.{column}")
    two = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if not two:
        raise ConfigError(f"expected table.column or source.table.column: {text!r}")
    table, column = two.group(1), two.group(2)
    index = dict(source_by_table or {})
    if not index and schema is not None:
        index = _federation_table_source_index(
            schema, FederationMappings(version=FEDERATION_MAPPINGS_VERSION), manifest
        )
    candidates: set[str] = set()
    ns = manifest.table_namespace.get(table, "")
    if ns:
        candidates.add(ns)
    mapped = index.get(table, "")
    if mapped:
        candidates.add(mapped)
    if schema is not None:
        composite_table = getattr(schema, "tables", {}).get(table)
        if composite_table is not None:
            composite_source = getattr(composite_table, "source_id", "") or ""
            if composite_source:
                candidates.add(composite_source)
            else:
                candidates.update(getattr(composite_table, "member_source_ids", None) or ())
    candidates.discard("")
    if len(candidates) > 1:
        raise ConfigError(f"ambiguous federation reference {text!r}: sources {sorted(candidates)}")
    if not candidates:
        if not manifest.sources and not manifest.table_namespace:
            return ResolvedQualifiedRef(source_id="", table=table, column=column, qualified=f"{table}.{column}")
        raise ConfigError(f"cannot attribute federation reference: {text!r}")
    source_id = next(iter(candidates))
    return ResolvedQualifiedRef(source_id=source_id, table=table, column=column, qualified=f"{table}.{column}")


def extract_tables_from_catalog_sql_connector(
    connection: Any,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict[str, Any]]:
    """Extract full table metadata from a Databricks Unity Catalog schema via SQL connector."""
    tables = {}
    allow_lower = frozenset(str(x).lower() for x in allow_objects) if allow_objects else None
    with connection.cursor() as cursor:
        cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
        table_rows = cursor_rows_as_dicts(cursor)
    table_col = "tableName"
    if table_rows and table_rows[0]:
        row0 = table_rows[0]
        if "tableName" in row0:
            table_col = "tableName"
        elif "tablename" in row0:
            table_col = "tablename"
        else:
            table_col = list(row0.keys())[0]
    for row in table_rows or []:
        table_name = row.get(table_col) if row else None
        if not table_name:
            continue
        if allow_lower is not None and str(table_name).lower() not in allow_lower:
            continue
        full_table = f"{catalog}.{schema}.{table_name}"
        debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] extracting: {full_table}")
        with connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE TABLE {full_table}")
            cols = cursor_rows_as_dicts(cursor)
        column_names = []
        column_types = []
        column_is_nullable: list[bool] = []
        null_map = structural_constraints_index.column_nullability.get(str(table_name).lower(), {})
        for col in cols:
            cname = col.get("col_name") or col.get("colname")
            if not cname or str(cname).startswith("#"):
                break
            column_names.append(cname)
            column_types.append(col.get("data_type") or "STRING")
            if str(cname) in null_map:
                column_is_nullable.append(bool(null_map[str(cname)]))
            else:
                column_is_nullable.append(True)
        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema and bundle is not None:
            primary_keys = list(bundle.primary_keys)
            foreign_keys = tables_meta_foreign_key_dicts_from_edges(bundle.foreign_keys)
            unique_columns = list(bundle.unique_columns)
        else:
            primary_keys = []
            foreign_keys = []
            unique_columns = []
        partition_columns: list[str] = []
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW CREATE TABLE {full_table}")
                create_rows = cursor_rows_as_dicts(cursor)
            if create_rows:
                row0 = create_rows[0] or {}
                stmt = row0.get("createtab_stmt") or (list(row0.values())[0] if row0 else None)
                if stmt:
                    partition_columns = parse_partition_columns_from_create_stmt(stmt)
                    if not use_info_schema:
                        pk_dd, fk_dd, uq_dd = parse_catalog_constraints_from_ddl(stmt)
                        primary_keys = pk_dd
                        foreign_keys = fk_dd
                        unique_columns = uq_dd
                    debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] ddl_found: {full_table}")
        except Exception as e:
            debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] ddl_error: {full_table} {e}")

        if not partition_columns:
            partition_columns = extract_partition_columns_from_describe_detail_sql_connector(connection, full_table)

        properties = {}
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW TBLPROPERTIES {full_table}")
                prop_rows = cursor_rows_as_dicts(cursor)
            for r in prop_rows or []:
                k = r.get("key")
                v = r.get("value")
                if k is not None:
                    properties[k] = v
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        table_comment = None
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE TABLE EXTENDED {full_table}")
                ext_rows = cursor_rows_as_dicts(cursor)
            for r in ext_rows or []:
                cname = r.get("col_name") or r.get("colname")
                if cname == "Comment":
                    table_comment = r.get("data_type")
                    break
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        tables[table_name] = {
            "table_name_original": table_name,
            "column_names_original": column_names,
            "column_types": column_types,
            "column_is_nullable": column_is_nullable,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }
    debug(f"[schema_profiling.extract_tables_from_catalog_sql_connector] complete: {len(tables)} tables")
    return tables


def extract_tables_from_catalog(
    spark: Any,
    catalog: str,
    schema: str,
    *,
    allow_objects: frozenset[str] | None = None,
    structural_constraints_index: CatalogStructuralConstraintsIndex,
) -> dict[str, dict[str, Any]]:
    """Extract full table metadata from a Databricks Unity Catalog schema."""
    tables = {}

    table_list = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()

    allow_lower = frozenset(str(x).lower() for x in allow_objects) if allow_objects else None
    for row in table_list:
        table_name = row["tableName"]
        if allow_lower is not None and str(table_name).lower() not in allow_lower:
            continue
        full_table = f"{catalog}.{schema}.{table_name}"

        debug(f"[schema_profiling.extract_tables_from_catalog] extracting: {full_table}")

        cols = spark.sql(f"DESCRIBE TABLE {full_table}").collect()

        column_names = []
        column_types = []
        column_is_nullable: list[bool] = []
        null_map = structural_constraints_index.column_nullability.get(str(table_name).lower(), {})

        for col in cols:
            col_name = col["col_name"]

            if col_name.startswith("#"):
                break

            column_names.append(col_name)
            column_types.append(col["data_type"])
            if col_name in null_map:
                column_is_nullable.append(bool(null_map[col_name]))
            else:
                column_is_nullable.append(True)

        bundle = structural_constraints_index.tables.get(str(table_name).lower())
        use_info_schema = bundle is not None and bool(
            bundle.primary_keys or bundle.foreign_keys or bundle.unique_columns
        )
        if use_info_schema and bundle is not None:
            primary_keys = list(bundle.primary_keys)
            foreign_keys = tables_meta_foreign_key_dicts_from_edges(bundle.foreign_keys)
            unique_columns = list(bundle.unique_columns)
        else:
            primary_keys = []
            foreign_keys = []
            unique_columns = []
        partition_columns: list[str] = []
        try:
            create_result = spark.sql(f"SHOW CREATE TABLE {full_table}").collect()
            if create_result:
                create_stmt = create_result[0]["createtab_stmt"]
                partition_columns = parse_partition_columns_from_create_stmt(create_stmt)
                if not use_info_schema:
                    pk_dd, fk_dd, uq_dd = parse_catalog_constraints_from_ddl(create_stmt)
                    primary_keys = pk_dd
                    foreign_keys = fk_dd
                    unique_columns = uq_dd
                debug(f"[schema_profiling.extract_tables_from_catalog] ddl_found: {full_table}")
        except Exception as e:
            debug(f"[schema_profiling.extract_tables_from_catalog] ddl_error: {full_table} {e}")

        if not partition_columns:
            partition_columns = extract_partition_columns_from_describe_detail_spark(spark, full_table)

        try:
            props = spark.sql(f"SHOW TBLPROPERTIES {full_table}").collect()
            properties = {p["key"]: p["value"] for p in props}
        except Exception:
            properties = {}

        try:
            extended = spark.sql(f"DESCRIBE TABLE EXTENDED {full_table}").collect()
            table_comment = None
            for row in extended:
                if row["col_name"] == "Comment":
                    table_comment = row["data_type"]
                    break
        except Exception:
            table_comment = None

        tables[table_name] = {
            "table_name_original": table_name,
            "column_names_original": column_names,
            "column_types": column_types,
            "column_is_nullable": column_is_nullable,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "unique_columns": unique_columns,
            "partition_columns": partition_columns,
            "table_comment": table_comment,
            "properties": properties,
        }

    debug(f"[schema_profiling.extract_tables_from_catalog] complete: {len(tables)} tables")
    return tables


def debug_schema_cache_hash_mismatch(
    *,
    schema_json_path: str,
    stamped_hash: str,
    json_tables: dict[str, Any],
) -> None:
    """Emit phased diagnostics when the file ``schema_hash`` disagrees with ``schema_hash_fp(tables)``."""
    fp_json = schema_hash_fp(json_tables)
    normalized_tables = tables_payload_through_model_round_trip(json_tables)
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
            "[schema.cache_hash_debug] stage=model_round_trip_drift "
            "fingerprint(json tables) != fingerprint(after table_from_dict/table_to_dict)"
        )
        diverge = first_table_where_stable_json_differs(json_tables, normalized_tables)
        if diverge is not None:
            debug(f"[schema.cache_hash_debug] first_diverging_table={diverge!r}")
            debug_clip_stable_json("json_table_slot", {diverge: json_tables.get(diverge)})
            debug_clip_stable_json("normalized_table_slot", {diverge: normalized_tables.get(diverge)})
    else:
        debug("[schema.cache_hash_debug] stage=json_vs_model_round_trip_ok raw json tables and model round-trip agree")
    if stamped_hash != fp_json:
        debug(
            "[schema.cache_hash_debug] stage=stamped_vs_json_tables "
            "on-disk schema_hash is not schema_hash_fp(tables); writer/stamp desync or foreign edit"
        )


def invalidate_corrupt_schema_cache(
    schema_json_path: str,
    exc: BaseException,
    *,
    stage: str = "deserialize",
) -> None:
    debug(f"[schema.build_schema_graph] corrupt cache {stage}: {exc!r}")
    adir = os.path.dirname(os.path.abspath(schema_json_path))
    write_artifact_manifest(
        adir,
        last_corruption_at=datetime.now(UTC).isoformat(),
        last_action="corrupt_schema_cache",
    )
    try:
        os.remove(schema_json_path)
    except OSError:
        pass


def schema_graph_from_cache_dict(d: dict[str, Any], schema_json_path: str) -> SchemaGraph | None:
    try:
        tables_raw = d.get("tables")
        if not isinstance(tables_raw, dict):
            raise ValueError("cache missing tables dict")
        tables_fast = {k: table_from_dict(v) for k, v in tables_raw.items()}
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
        model["tables"] = {k: table_to_dict(v) for k, v in tables_fast.items()}
        sg = SchemaGraph.from_dict(model)
        ensure_semantic_join_neighbors(sg)
        return sg
    except Exception as exc:
        invalidate_corrupt_schema_cache(schema_json_path, exc)
        return None


def _debug_verify_schema_cache_write(cache_data: dict[str, Any], schema_json_path: str) -> None:
    """Emit debug lines when stamped hash or document round-trip disagrees with table fingerprints."""
    tables_payload = cache_data["tables"]
    fp_tables = schema_hash_fp(tables_payload)
    stamped = cache_data.get("effective_structural_hash", "")
    if fp_tables != stamped:
        debug(
            "[schema.cache_hash_debug] write_path_stamp_mismatch "
            f"path={schema_json_path!r} "
            f"fp_tables_prefix={fp_tables[:16]!r} "
            f"effective_structural_hash_field_prefix={stamped[:16]!r}"
        )
    fp_round_trip = fingerprint_tables_after_document_round_trip(cache_data)
    if fp_round_trip != fp_tables:
        debug(
            "[schema.cache_hash_debug] write_document_round_trip_drift "
            f"path={schema_json_path!r} "
            f"fp_tables_prefix={fp_tables[:16]!r} "
            f"fp_after_stringio_round_trip_prefix={fp_round_trip[:16]!r}"
        )
    debug_verify_profiling_round_trip(cache_data, schema_json_path)


def _commit_schema_cache_and_manifest_pair(
    adir: str,
    schema_json_path: str,
    cache_data: dict[str, Any],
    *,
    manifest_kwargs: dict[str, Any],
) -> None:
    """Stage schema cache and artifact manifest, then replace both live paths atomically."""
    graph_path = os.path.abspath(schema_json_path)
    manifest_path = os.path.join(adir, ARTIFACT_MANIFEST_FILENAME)
    staging_dir = os.path.join(adir, ".schema_pair_staging")
    graph_backup: str | None = None
    try:
        if os.path.isfile(graph_path):
            fd, graph_backup = tempfile.mkstemp(prefix=".rollback_", suffix=".json.gz", dir=adir)
            os.close(fd)
            shutil.copy2(graph_path, graph_backup)

        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

        graph_staging = os.path.join(staging_dir, os.path.basename(graph_path))
        write_gzip_json_atomic(graph_staging, cache_data, sort_keys=True)
        write_artifact_manifest(staging_dir, **manifest_kwargs)
        manifest_staging = os.path.join(staging_dir, ARTIFACT_MANIFEST_FILENAME)

        os.replace(graph_staging, graph_path)
        try:
            os.replace(manifest_staging, manifest_path)
        except BaseException:
            if graph_backup and os.path.isfile(graph_backup):
                os.replace(graph_backup, graph_path)
                graph_backup = None
            elif os.path.isfile(graph_path):
                try:
                    os.remove(graph_path)
                except OSError:
                    pass
            raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if graph_backup and os.path.isfile(graph_backup):
            try:
                os.unlink(graph_backup)
            except OSError:
                pass


def save_schema_to_cache(sg: SchemaGraph, schema_json_path: str) -> None:
    """Save a SchemaGraph to a JSON cache file."""
    cache_data = {
        "tables": {k: table_to_dict(v) for k, v in sg.tables.items()},
        "join_paths_multi": sg.join_paths_multi,
        "structural_hash": sg.structural_hash,
        "profiling_hash": sg.profiling_hash,
        "descriptions_hash": sg.descriptions_hash,
        "scope_hash": sg.scope_hash,
        "effective_structural_hash": sg.effective_structural_hash,
        "schema_graph_id": sg.schema_graph_id,
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
        "join_path_enumeration_version": SCHEMA_JOIN_PATH_ENUMERATION_VERSION,
    }

    _debug_verify_schema_cache_write(cache_data, schema_json_path)

    debug(f"[schema.save_schema_to_cache] saving to '{schema_json_path}'")
    adir = os.path.dirname(os.path.abspath(schema_json_path))
    prev_manifest = read_artifact_manifest(adir)
    with artifact_lock(adir):
        _commit_schema_cache_and_manifest_pair(
            adir,
            schema_json_path,
            cache_data,
            manifest_kwargs={
                "structural_hash": sg.structural_hash,
                "profiling_hash": sg.profiling_hash,
                "scope_hash": sg.scope_hash,
                "effective_structural_hash": sg.effective_structural_hash,
                "schema_graph_id": sg.schema_graph_id,
                "notes_hash": sg.notes_hash,
                "semantic_edges_hash": sg.semantic_edges_hash,
                "last_migration_tier": prev_manifest.last_migration_tier if prev_manifest else "",
                "last_action": "reconcile",
            },
        )


def _scope_databricks_tables_meta(
    tables_meta: dict[str, dict[str, Any]],
    *,
    object_kind: TableKind,
    table_types: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Keep only relations whose Unity ``table_type`` matches *object_kind*."""
    if not table_types:
        return tables_meta
    out: dict[str, dict[str, Any]] = {}
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


def _scope_databricks_for_include(
    tables_meta: dict[str, dict[str, Any]],
    *,
    include: SchemaInclude,
    table_types: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Filter catalog metadata to tables or views."""
    return _scope_databricks_tables_meta(
        tables_meta,
        object_kind=TableKind.TABLE if include == "tables" else TableKind.VIEW,
        table_types=table_types,
    )


def load_or_create_schema_databricks(
    spark_session: Any | None = None,
    connection: Any | None = None,
    *,
    include: SchemaInclude = SchemaInclude.TABLES,
    allow_objects: frozenset[str] | None = None,
    deny_objects: frozenset[str] | None = None,
    table_kinds_map: dict[str, str],
    structural_constraints_index: CatalogStructuralConstraintsIndex,
    sql_file: str | None = None,
) -> SchemaGraph:
    """Build a `SchemaGraph` for Databricks from catalog introspection or SQL DDL fallback."""
    runtime_cfg = bound_engine_runtime_config()
    catalog = str(getattr(runtime_cfg, "CATALOG", None) or "")
    schema_name = str(getattr(runtime_cfg, "SCHEMA", None) or "")
    tables_meta: dict[str, dict[str, Any]] = {}
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
            try:
                from pyspark.sql import SparkSession
            except ImportError as exc:
                raise SchemaAccessError("Cannot build Databricks schema: pyspark is not installed.") from exc
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
        sql_file_path = expanded_scope_sql_file(sql_file)
        if sql_file_path and os.path.exists(sql_file_path):
            debug(f"[schema.load_or_create_schema_databricks] catalog empty; parsing SQL file '{sql_file_path}'")
            tables_meta = parse_sql_file(Path(sql_file_path))
            allow_lower = allow_objects_lower_set(allow_objects)
            if allow_lower is not None:
                tables_meta = {k: v for k, v in tables_meta.items() if str(k).lower() in allow_lower}
        if not tables_meta:
            raise SchemaAccessError(
                "Cannot build Databricks schema: catalog returned no tables and SQL file is missing, empty, or unparsable.",
            )

    tables_meta = _scope_databricks_for_include(
        tables_meta,
        include=include,
        table_types=table_kinds_map,
    )

    row_kind: TableKind = TableKind.TABLE if include != "views" else TableKind.VIEW
    sg = tables_meta_to_schema_graph(tables_meta, object_kind=row_kind)
    sql_file_path = expanded_scope_sql_file(sql_file)
    if include == "tables" and sql_file_path and os.path.exists(sql_file_path) and tables_meta:
        ddl_tables = parse_sql_file(Path(sql_file_path), reflected_schema=sg)
        if ddl_tables:
            merge_ddl_foreign_keys_into_schema_graph(sg, ddl_tables)
    return apply_full_build_deny_objects(sg, deny_objects)


def ensure_semantic_join_neighbors(sg: SchemaGraph) -> None:
    """Recompute semantic profile edges from cached or freshly profiled ``value_overlap_sample``."""
    compute_semantic_profile_join_neighbors(sg)


def debug_verify_profiling_round_trip(cache_data: dict[str, Any], schema_json_path: str) -> None:
    """Detect drift between the stamped ``profiling_hash`` and the value recomputed after a JSON round-trip of the written tables. On drift, debug the first column whose profiling dict differs and dump both slots so the offending field is identifiable."""
    stamped_pr = str(cache_data.get("profiling_hash", ""))
    if not stamped_pr:
        return
    tables_json = cache_data["tables"]
    reparsed = json.loads(schema_cache_json_blob(cache_data))["tables"]
    reloaded_tables = {k: table_from_dict(v) for k, v in reparsed.items()}
    pr_after = profiling_hash_fp(tables_profiling_payload(reloaded_tables))
    if pr_after == stamped_pr:
        return
    debug(
        "[schema.cache_hash_debug] write_profiling_round_trip_drift "
        f"path={schema_json_path!r} "
        f"stamped_profiling_prefix={stamped_pr[:16]!r} "
        f"recomputed_after_round_trip_prefix={pr_after[:16]!r}"
    )
    pre_tables = {k: table_from_dict(v) for k, v in tables_json.items()}
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
            debug_clip_stable_json("profiling_pre_round_trip", {cname: pre_col})
            debug_clip_stable_json("profiling_post_round_trip", {cname: post_col})
            return
