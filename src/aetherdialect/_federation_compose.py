"""Federation composite graph compose/collapse, grants, invariants, and description scrub."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, cast

from ._constants import (
    DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED,
    DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT,
    FEDERATION_ENUM_PROMPT_CAP,
    FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    FEDERATION_SENSITIVITY_RANK,
    FEDERATION_TIMEZONE_AWARE_DATA_TYPES,
)
from ._constants_runtime import (
    FEDERATION_COMPOSITE_RECONCILIATION_NOTE,
    FEDERATION_COMPOSITION_PHASE_A,
    FEDERATION_COMPOSITION_PHASE_B,
    FEDERATION_COMPOSITION_PHASE_C,
    FEDERATION_COMPOSITION_PHASE_D,
    FEDERATION_COMPOSITION_PHASE_E,
    FEDERATION_COMPOSITION_PHASE_F,
    FEDERATION_COMPOSITION_PHASE_G,
    FEDERATION_COMPOSITION_PHASE_H,
)
from ._contracts_base import (
    ConfigError,
    EngineContext,
    FederationConfigError,
    FederationContext,
    FederationDeclarationError,
    FederationInvariantError,
    FederationRuntimeError,
    MemberEffectiveGrants,
    SensitivityClassification,
)
from ._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
)
from ._contracts_schema import (
    ColumnMetadata,
    DescriptionOwner,
    FederationManifest,
    FederationMappings,
    FKEdge,
    InferenceTag,
    LogicalColumnMapping,
    LogicalTableMapping,
    LogicalTableMember,
    SchemaGraph,
    TableMetadata,
)
from ._federation_manifest import (
    apply_composite_federation_scope,
    assign_cte_sources,
    coerce_member_effective_grants,
    collect_federation_description_forbidden_tokens,
    derive_table_namespace,
    engine_context_for_schema_usability,
    federation_member_hash_tuple,
    intersect_member_database_feature_capabilities,
    intersect_member_dialect_capabilities,
    introspect_member_effective_grants,
    manifest_engine_for_source,
    manifest_hash,
    manifest_with_derived_roster,
    mappings_hash,
    owner_is_aether_federation,
    raise_if_schema_graph_descriptions_contain_member_identifiers,
    sources_for_table,
    stamp_federation_member_graph,
)
from ._intent_normalize import (
    collect_referenced_tables,
)
from ._schema_graph import (
    deny_columns_by_table,
    fk_infer_value_types_compatible,
    mark_canonical_duplicates,
    mint_schema_graph_id,
    raise_if_schema_unusable,
    recompute_join_paths_multi,
    redact_hidden_sensitivity_profile_values,
    schema_context_from_descriptor,
    tables_structural_payload,
    validate_scope_against_graph,
)
from ._schema_profile import (
    assign_column_ops,
    llm_classify_schema,
    sanitize_schema_graph_descriptions,
    value_overlap_ratio_for_columns,
)
from ._schema_reflect import resolve_federation_qualified_ref
from ._utils import (
    effective_structural_hash_fp,
    emit_construction_phase,
    notify,
    pipeline_trace,
    structural_hash_fp,
)
from ._validation_shape import fk_points_to_parent


def scrub_federation_member_description_source_tokens(
    member_graphs: Mapping[str, SchemaGraph],
    source_ids: Iterable[str],
) -> None:
    """Strip federation source identifiers from member table/column descriptions in-place."""
    tokens = frozenset(str(source_id or "").strip() for source_id in source_ids) - frozenset({""})
    if not tokens:
        return
    for graph in member_graphs.values():
        sanitize_schema_graph_descriptions(graph, tokens)


def schema_graph_classification_payload(
    graph: SchemaGraph,
) -> dict[str, tuple[Any, Any, dict[str, tuple[Any, Any, Any]]]]:
    """Build an ``llm_classify_schema``-shaped payload from an existing composite graph."""
    payload: dict[str, tuple[Any, Any, dict[str, tuple[Any, Any, Any]]]] = {}
    for table_name, table in graph.tables.items():
        col_classes = {
            col_name: (col.role, col.description, getattr(col, "sensitivity", None))
            for col_name, col in table.columns.items()
        }
        payload[table_name] = (table.role, table.description, col_classes)
    return payload


_bundled_composite_classifications: dict[str, tuple[Any, Any, dict[str, tuple[Any, Any, Any]]]] | None = None


def set_bundled_composite_classifications(
    payload: dict[str, tuple[Any, Any, dict[str, tuple[Any, Any, Any]]]] | None,
) -> None:
    """Pin offline/bundled composite classifications used when reconcile would otherwise call the LLM."""
    global _bundled_composite_classifications
    _bundled_composite_classifications = payload


def clear_bundled_composite_classifications() -> None:
    """Clear any pin installed by :func:`set_bundled_composite_classifications`."""
    set_bundled_composite_classifications(None)


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
    manifest = manifest_with_derived_roster(manifest, member_graphs=member_graphs, mappings=mappings)
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
            engine=manifest_engine_for_source(manifest, source_id),
        )
    composite_names = _resolve_composite_table_names(member_graphs, manifest)
    _assign_collapse_staging_composite_names(composite_names, mappings)
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
    sanitize_schema_graph_descriptions(composite, forbidden_description_tokens)
    raise_if_schema_graph_descriptions_contain_member_identifiers(composite, forbidden_description_tokens)
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
        validate_federation_scope_against_member_visibility(
            scope_ctx, member_graphs, composite, composite_names, mappings, manifest
        )
    validate_scope_against_graph(composite, scope_ctx)
    apply_composite_federation_scope(composite, scope_ctx)
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
        composite, engine_context_for_schema_usability(scope_ctx), federation_composite=len(member_graphs) > 1
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
    """Return drift messages when replica logical tables disagree across members. When any member declares a column projection, that logical key set is the replica contract: every projected member must declare the same keys, and members with an empty map must physically cover those keys. When no member declares a projection, physical column names must match across members."""
    drift: list[str] = []
    for table_map in mappings.logical_tables:
        if table_map.semantics != "replica" or len(table_map.members) < 2:
            continue
        member_specs: list[tuple[str, TableMetadata, dict[str, str] | None]] = []
        for member in table_map.members:
            graph = member_graphs.get(member.source)
            if graph is None:
                drift.append(f"replica {table_map.logical!r}: missing graph for source {member.source!r}")
                continue
            tbl = graph.tables.get(member.table)
            if tbl is None:
                drift.append(f"replica {table_map.logical!r}: missing table {member.table!r} on {member.source!r}")
                continue
            projection = dict(member.columns) if member.columns else None
            member_specs.append((member.source, tbl, projection))
            if projection is not None:
                for logical, physical in projection.items():
                    if physical not in tbl.columns:
                        drift.append(
                            f"replica {table_map.logical!r}: missing physical column "
                            f"{physical!r} for logical {logical!r} on {member.source!r}"
                        )
        if len(member_specs) < 2:
            continue
        projected_key_sets = [frozenset(projection.keys()) for _, _, projection in member_specs if projection]
        if projected_key_sets:
            reference_keys = projected_key_sets[0]
            for keys in projected_key_sets[1:]:
                if keys != reference_keys:
                    drift.append(f"replica {table_map.logical!r}: column coverage drift between projected members")
                    break
            for source_id, tbl, projection in member_specs:
                if projection is not None:
                    continue
                missing = reference_keys - frozenset(tbl.columns)
                if missing:
                    drift.append(
                        f"replica {table_map.logical!r}: column coverage drift between "
                        f"projected members and {source_id!r}"
                    )
            continue
        col_sets = {source_id: frozenset(tbl.columns) for source_id, tbl, _ in member_specs}
        reference_source = next(iter(sorted(col_sets)))
        auth = (table_map.authoritative_source or "").strip()
        if auth in col_sets:
            reference_source = auth
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
        overlap_samples: list[tuple[str, ColumnMetadata]] = []
        for member_ref in col_map.members:
            table_name, column_name = split_qualified_column(member_ref, manifest=manifest)
            source_id = physical_table_source(table_name, mappings)
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
            if meta.value_overlap_sample:
                overlap_samples.append((source_id, meta))
        if len(overlap_samples) < 2:
            continue
        min_overlap = 1.0
        for left_idx in range(len(overlap_samples)):
            left_source, left_meta = overlap_samples[left_idx]
            for right_source, right_meta in overlap_samples[left_idx + 1 :]:
                ratio = value_overlap_ratio_for_columns(left_meta, right_meta)
                min_overlap = min(min_overlap, ratio)
                if ratio < FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR:
                    drift.append(
                        f"declared {col_map.logical!r}: value overlap rescoring drift between "
                        f"{left_source!r} and {right_source!r}"
                    )
    return tuple(drift)


def physical_table_source(table_name: str, mappings: FederationMappings) -> str:
    """Return the member source id that owns a physical table name."""
    for table_map in mappings.logical_tables:
        for member in table_map.members:
            if member.table == table_name:
                return member.source
    return ""


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


def _member_scope_context_from_graph(graph: SchemaGraph) -> EngineContext:
    desc = graph.scope_descriptor if isinstance(graph.scope_descriptor, dict) else {}
    return schema_context_from_descriptor(desc) if desc else EngineContext()


def _member_visible_physical_tables(graph: SchemaGraph) -> frozenset[str]:
    member_ctx = _member_scope_context_from_graph(graph)
    tables = set(graph.tables.keys())
    if member_ctx.allow_objects:
        tables &= set(member_ctx.allow_objects)
    tables -= set(member_ctx.deny_objects)
    return frozenset(tables)


def _federation_scope_member_refs(
    name: str,
    composite_names: Mapping[tuple[str, str], str],
    mappings: FederationMappings,
    manifest: FederationManifest,
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for member_key, composite in composite_names.items():
        if composite != name or member_key in seen:
            continue
        refs.append(member_key)
        seen.add(member_key)
    for table_map in mappings.logical_tables:
        if table_map.logical != name:
            continue
        for member in table_map.members:
            key = (member.source, member.table)
            if member.source and member.table and key not in seen:
                refs.append(key)
                seen.add(key)
    if refs:
        return sorted(refs)
    source_id = (manifest.table_namespace or {}).get(name, "")
    if source_id:
        return [(source_id, name)]
    return []


def _classify_federation_member_scope_violation(
    source_id: str,
    phys: str,
    federation_name: str,
    graph: SchemaGraph,
) -> str | None:
    member_ctx = _member_scope_context_from_graph(graph)
    if phys not in graph.tables:
        return "not_on_member_graph"
    if phys in member_ctx.deny_objects or federation_name in member_ctx.deny_objects:
        return "member_denied"
    if member_ctx.allow_objects and phys not in member_ctx.allow_objects:
        return "member_allow_omitted"
    if phys not in _member_visible_physical_tables(graph):
        if phys in member_ctx.deny_objects or federation_name in member_ctx.deny_objects:
            return "member_denied"
        if member_ctx.allow_objects:
            return "member_allow_omitted"
        return "not_on_member_graph"
    return None


def validate_federation_scope_against_member_visibility(
    ctx: FederationContext,
    member_graphs: Mapping[str, SchemaGraph],
    composite: SchemaGraph,
    composite_names: Mapping[tuple[str, str], str],
    mappings: FederationMappings,
    manifest: FederationManifest,
) -> None:
    """Reject federation scope that exceeds any member engine's effective object visibility."""
    composite_tables = frozenset(composite.tables.keys())
    for name in sorted(ctx.deny_objects):
        if name not in composite_tables:
            raise ConfigError(f"federation deny_objects references unknown table: {name!r}")
    for name in sorted(ctx.allow_objects):
        if name not in composite_tables:
            raise ConfigError(f"federation allow_objects references unknown table: {name!r}")
        refs = _federation_scope_member_refs(name, composite_names, mappings, manifest)
        if not refs:
            raise FederationDeclarationError(
                f"federation allow_objects names {name!r} with no resolvable member source (not_on_member_graph)"
            )
        for source_id, phys in refs:
            graph = member_graphs.get(source_id)
            if graph is None:
                raise FederationDeclarationError(
                    f"federation allow_objects names {name!r} on unresolved member {source_id!r} (not_on_member_graph)"
                )
            reason = _classify_federation_member_scope_violation(source_id, phys, name, graph)
            if reason is not None:
                raise FederationDeclarationError(
                    f"federation allow_objects names {name!r} on member {source_id!r} ({reason})"
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
    if fk_points_to_parent(table_name, peer_table_name, [column_name], schema):
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
    left_unique = source_join_key_is_unique(
        schema, _table_source_id_for_manifest(schema, manifest, left_tbl), left_qual, manifest=manifest
    )
    right_unique = source_join_key_is_unique(
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
    unique = source_join_key_is_unique(schema, src, f"{tbl}.{col}", manifest=manifest)
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


def _cross_source_join_key_exactness_mismatch(left_meta: ColumnMetadata, right_meta: ColumnMetadata) -> bool:
    """Return True when two numeric join-key columns disagree on exactness."""
    if not _column_metadata_is_numeric(left_meta) or not _column_metadata_is_numeric(right_meta):
        return False
    return left_meta.is_exact_numeric != right_meta.is_exact_numeric


def column_data_type_is_timezone_aware(data_type: str) -> bool:
    raw = str(data_type or "").strip().lower()
    if not raw:
        return False
    base = raw.split("(", 1)[0].strip()
    if base in FEDERATION_TIMEZONE_AWARE_DATA_TYPES:
        return True
    return "with time zone" in raw


def _column_metadata_is_timestamp_type(meta: ColumnMetadata) -> bool:
    dtype = str(meta.data_type or "").strip().lower()
    if not dtype:
        return False
    base = dtype.split("(", 1)[0].strip()
    if base in {
        "timestamp",
        "timestamptz",
        "datetime",
        "datetime2",
        "smalldatetime",
        "datetimeoffset",
        "timestamp_ntz",
        "timestamp_tz",
        "timestamp_ltz",
    }:
        return True
    if base == "date":
        return False
    return "timestamp" in dtype or "datetime" in dtype


def _column_metadata_timezone_awareness_mismatch(left_meta: ColumnMetadata, right_meta: ColumnMetadata) -> bool:
    if not _column_metadata_is_timestamp_type(left_meta) or not _column_metadata_is_timestamp_type(right_meta):
        return False
    return left_meta.is_timezone_aware != right_meta.is_timezone_aware


def _assert_union_column_timestamp_awareness_agrees(
    candidates: Sequence[ColumnMetadata],
    sources: Sequence[str],
    label: str,
) -> None:
    for left_idx, left_meta in enumerate(candidates):
        for right_idx in range(left_idx + 1, len(candidates)):
            right_meta = candidates[right_idx]
            if not _column_metadata_timezone_awareness_mismatch(left_meta, right_meta):
                continue
            left_qual = f"{sources[left_idx]}.{left_meta.name}"
            right_qual = f"{sources[right_idx]}.{right_meta.name}"
            raise FederationDeclarationError(
                f"timestamp timezone awareness incompatible for {label}: "
                f"{left_qual} ({left_meta.data_type}) vs {right_qual} ({right_meta.data_type})"
            )


def _column_metadata_is_numeric(meta: ColumnMetadata) -> bool:
    value_type = (meta.value_type or "").strip().lower()
    if value_type == "number":
        return True
    dtype = str(meta.data_type or "").lower()
    return any(token in dtype for token in ("decimal", "numeric", "double", "float", "real", "number"))


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
        if _cross_source_join_key_exactness_mismatch(left_meta, right_meta):
            left_dtype = str(left_meta.data_type or left_type).strip()
            right_dtype = str(right_meta.data_type or right_type).strip()
            raise FederationDeclarationError(
                f"cross-source join key exactness incompatible for logical column {logical!r}: "
                f"{left_qual!r} ({left_dtype}) vs {right_qual!r} ({right_dtype})"
            )
        if _column_metadata_timezone_awareness_mismatch(left_meta, right_meta):
            left_dtype = str(left_meta.data_type or left_type).strip()
            right_dtype = str(right_meta.data_type or right_type).strip()
            raise FederationDeclarationError(
                f"timestamp timezone awareness incompatible for logical column {logical!r}: "
                f"{left_qual!r} ({left_dtype}) vs {right_qual!r} ({right_dtype})"
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


def spanning_cte_decomposition_ineligible_reason(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
) -> str | None:
    """Refuse spanning CTE bodies whose clauses cannot be replayed at the coordinator."""
    spanning = spanning_cte_names(intent.cte_steps or (), source_by_table)
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


def spanning_cte_names(cte_steps: Sequence[RuntimeCteStep], source_by_table: Mapping[str, str]) -> tuple[str, ...]:
    """Return CTE names whose referenced base tables span more than one member."""
    if not cte_steps:
        return ()
    owners = assign_cte_sources(cte_steps, source_by_table)
    spanning: list[str] = []
    for cte in cte_steps:
        name = cte.cte_name
        if name and name not in owners:
            spanning.append(name)
    return tuple(spanning)


def source_ids_for_intent(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    mappings: FederationMappings | None = None,
    manifest: FederationManifest | None = None,
) -> frozenset[str]:
    """Return the set of member source ids referenced by *intent* tables."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    source_by_table = table_source_index(schema, mappings, manifest)
    tables = set(intent.tables or [])
    if manifest is not None:
        return frozenset(intent_table_sources(tables, manifest, mappings, source_by_table))
    sources = {_source_for_table(table, source_by_table) for table in tables}
    sources.discard("")
    return frozenset(sources)


def _strictest_sensitivity(*values: SensitivityClassification) -> SensitivityClassification:
    return max(values, key=lambda value: FEDERATION_SENSITIVITY_RANK.get(value.value, 0))


def _base_description_from_authoritative(
    candidates: Sequence[Any],
    *,
    member_sources: Sequence[str] | None,
    authoritative_source: str,
) -> str:
    """Prefer authoritative member ``base_description``, else first non- empty."""
    auth = (authoritative_source or "").strip()
    if auth and member_sources and len(member_sources) == len(candidates):
        for source, item in zip(member_sources, candidates, strict=True):
            if source == auth:
                text = str(getattr(item, "base_description", "") or "").strip()
                if text:
                    return text
                break
    for item in candidates:
        text = str(getattr(item, "base_description", "") or "").strip()
        if text:
            return text
    return ""


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
        if other.enum_type_name and merged.enum_type_name and other.enum_type_name != merged.enum_type_name:
            merged.enum_type_name = None
        elif other.enum_type_name and not merged.enum_type_name:
            merged.enum_type_name = other.enum_type_name
        if other.element_type and merged.element_type and other.element_type != merged.element_type:
            merged.element_type = None
        elif other.element_type and not merged.element_type:
            merged.element_type = other.element_type
        if (
            other.boolean_truth_value
            and merged.boolean_truth_value
            and other.boolean_truth_value != merged.boolean_truth_value
        ):
            merged.boolean_truth_value = None
        elif other.boolean_truth_value and not merged.boolean_truth_value:
            merged.boolean_truth_value = other.boolean_truth_value
        for neighbor in other.semantic_join_neighbors or []:
            if neighbor not in merged.semantic_join_neighbors:
                merged.semantic_join_neighbors.append(neighbor)
    merged_desc, merged_owner = DescriptionOwner.resolve(
        *((col.description, col.description_owner) for col in candidates)
    )
    if merged_desc:
        DescriptionOwner.set_on(merged, merged_desc, merged_owner or DescriptionOwner.CATALOG)
    else:
        merged.description = ""
        merged.description_owner = None
    return merged


def _apply_authoritative_base_description(
    merged: ColumnMetadata,
    candidates: Sequence[ColumnMetadata],
    *,
    member_sources: Sequence[str] | None,
    authoritative_source: str,
) -> None:
    merged.base_description = _base_description_from_authoritative(
        candidates,
        member_sources=member_sources,
        authoritative_source=authoritative_source,
    )


def _scalar_field_values(candidates: Sequence[ColumnMetadata], field: str) -> set[Any]:
    return {getattr(col, field) for col in candidates}


def _normalized_string_values(candidates: Sequence[ColumnMetadata], field: str) -> set[str]:
    return {
        str(getattr(col, field) or "").strip().lower() for col in candidates if str(getattr(col, field) or "").strip()
    }


def _assert_collapsed_column_metadata_agrees(
    candidates: Sequence[ColumnMetadata],
    label: str,
    *,
    semantics: Literal["union", "replica"],
) -> None:
    """Raise when collapsed members disagree on mergeable column metadata fields."""
    if len(candidates) <= 1:
        return
    scalar_fields: tuple[str, ...] = (
        "value_type",
        "is_nullable",
        "is_unique",
        "is_generated",
        "is_identity",
        "enum_type_name",
        "element_type",
        "boolean_truth_value",
    )
    if semantics == "replica":
        scalar_fields = ("data_type",) + scalar_fields
    for field in scalar_fields:
        if field in {"data_type", "value_type", "enum_type_name", "element_type", "boolean_truth_value"}:
            values = _normalized_string_values(candidates, field)
            if not values:
                continue
        else:
            values = _scalar_field_values(candidates, field)
        if len(values) > 1:
            raise FederationConfigError(f"{label}: collapsed members disagree on {field}: {sorted(values)!r}")
    if semantics == "replica":
        for field in ("is_aggregatable_override", "is_groupable_override", "is_filterable_override"):
            values = _scalar_field_values(candidates, field)
            if len(values) > 1:
                raise FederationConfigError(f"{label}: collapsed members disagree on {field}: {sorted(values)!r}")


def _assert_replica_column_data_types_agree(candidates: Sequence[ColumnMetadata], label: str) -> None:
    """Raise when replica members disagree on column data_type."""
    _assert_collapsed_column_metadata_agrees(candidates, label, semantics="replica")


def _merge_column_metadata_union_statistics(
    candidates: Sequence[ColumnMetadata],
    *,
    composite_semantics: Literal["union", "replica", "logical_unify"] = "logical_unify",
    member_sources: Sequence[str] | None = None,
    authoritative_source: str = "",
) -> ColumnMetadata:
    """Merge column metadata and profiling statistics across *candidates*."""
    merged = _merge_column_metadata_strictest(candidates)
    _apply_authoritative_base_description(
        merged,
        candidates,
        member_sources=member_sources,
        authoritative_source=authoritative_source,
    )
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
        merged_frequent: list[str] = []
        for col in ordered_candidates:
            for value in col.frequent_values or []:
                token = str(value)
                if token and token not in merged_frequent:
                    merged_frequent.append(token)
        merged.frequent_values = merged_frequent
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
    frequent_values: list[str] = []
    overlap: list[str] = []
    min_vals: list[str] = []
    max_vals: list[str] = []
    for col in ordered_candidates:
        for value in col.frequent_values or []:
            token = str(value)
            if token and token not in frequent_values:
                frequent_values.append(token)
        for value in col.value_overlap_sample or []:
            token = str(value)
            if token and token not in overlap:
                overlap.append(token)
        if col.min_val not in (None, ""):
            min_vals.append(str(col.min_val))
        if col.max_val not in (None, ""):
            max_vals.append(str(col.max_val))
    merged.frequent_values = frequent_values
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


def _assert_collapsed_table_metadata_agrees(member_tables: Sequence[TableMetadata], logical: str) -> None:
    for field in ("kind", "role", "distkey", "diststyle", "clustering_key", "quote_decision", "encoded"):
        values = {getattr(table, field) for table in member_tables}
        if len(values) > 1:
            raise FederationDeclarationError(
                f"logical table {logical!r} members disagree on {field}: {sorted(values, key=str)!r}"
            )
    for field in ("sortkey", "indexed_columns", "clustering_fields"):
        signatures = {tuple(getattr(table, field) or []) for table in member_tables}
        if len(signatures) > 1:
            raise FederationDeclarationError(f"logical table {logical!r} members disagree on {field}")


def _table_grain_signature(table: TableMetadata, logical_pk: Sequence[str]) -> str | None:
    """Classify whether *table* rows are entity- or event-grained for *logical_pk*."""
    pk = [col for col in logical_pk if col]
    if not pk:
        return "none"
    if len(pk) == 1:
        col_name = pk[0]
        column = table.columns.get(col_name)
        if column is None:
            return None
        if column.is_unique:
            return "entity"
        row_count = int(table.row_count or 0)
        distinct = int(column.distinct_count or 0)
        if row_count > 0 and distinct > 0:
            return "entity" if distinct >= row_count else "event"
        return None
    if list(table.primary_key or []) != list(pk):
        return None
    row_count = int(table.row_count or 0)
    if row_count <= 0:
        return None
    for col_name in pk:
        column = table.columns.get(col_name)
        if column is None:
            return None
        if column.is_unique:
            continue
        distinct = int(column.distinct_count or 0)
        if distinct > 0 and distinct < row_count:
            return "event"
    return "entity"


def _assert_member_grain_equivalence(
    mapping: LogicalTableMapping,
    member_tables: Sequence[TableMetadata],
) -> None:
    """Refuse replica/union collapse when members disagree on row grain."""
    if len(member_tables) < 2:
        return
    signatures: dict[str, str] = {}
    for member, table in zip(mapping.members, member_tables, strict=True):
        logical_pk = _member_logical_primary_key(member, table)
        signature = _table_grain_signature(table, logical_pk)
        if signature is not None:
            signatures[member.source] = signature
    if len(set(signatures.values())) > 1:
        detail = ", ".join(f"{source}={signature}" for source, signature in sorted(signatures.items()))
        raise FederationDeclarationError(f"logical table {mapping.logical!r} members disagree on row grain: {detail}")


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
    mapping_sources = mapping_member_source_by_table(mappings)
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
    allow_profiled_graph_fallback: bool = True,
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
    if allow_profiled_graph_fallback:
        return member_effective_grants_from_graph(graph)
    sid = str(source_id or "").strip() or "member"
    raise FederationDeclarationError(
        f"federation member {sid!r} live effective grants are unavailable; "
        "profiled schema graphs cannot substitute for runtime read permissions"
    )


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
        engine = engines.get(source_id)
        grants = resolve_member_effective_grants(
            source_id,
            graph,
            engine=engine,
            explicit=explicit_grants.get(source_id),
            allow_profiled_graph_fallback=engine is None,
        )
        for table_name, column_name in sorted(declared[source_id], key=lambda item: (item[0], item[1] or "")):
            if table_name not in graph.tables:
                continue
            if _member_effective_grants_include(grants, table_name, column_name, member_graph=graph):
                continue
            obj = f"{table_name}.{column_name}" if column_name else table_name
            raise FederationDeclarationError(f"federation member {source_id!r} effective grants do not include {obj!r}")


def mapping_member_source_by_table(mappings: FederationMappings) -> dict[str, str]:
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
    mapping_sources = mapping_member_source_by_table(mappings)
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
        agreed_desc, agreed_owner = DescriptionOwner.resolve(*descriptions)
        if agreed_desc:
            DescriptionOwner.set_on(table, agreed_desc, agreed_owner or DescriptionOwner.CATALOG)
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
            agreed_col_desc, agreed_col_owner = DescriptionOwner.resolve(*col_desc_candidates)
            if agreed_col_desc:
                DescriptionOwner.set_on(col, agreed_col_desc, agreed_col_owner or DescriptionOwner.CATALOG)
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
        member_col_roles: list[Any] = []
        member_col_descriptions: list[Any] = []
        for qual in col_map.members:
            tbl, col_name = split_qualified_column(qual)
            table = composite.tables.get(tbl)
            if table is None:
                continue
            col = table.columns.get(col_name) or table.columns.get(col_map.logical)
            if col is None:
                continue
            member_col_roles.append(col.role)
            member_col_descriptions.append((col.description, col.description_owner))
        role_values = {role for role in member_col_roles if role}
        if len(role_values) > 1:
            for qual in col_map.members:
                tbl, _ = split_qualified_column(qual)
                conflicts.add(f"{tbl}.{col_map.logical}")
    if not conflicts:
        return False
    classify = llm_classify
    if classify is None and _bundled_composite_classifications is not None:
        pinned = _bundled_composite_classifications

        def classify(_composite: SchemaGraph, _notes: str | None = None) -> dict[str, Any]:
            return pinned

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
            DescriptionOwner.set_on(table, table_desc, reconcile_owner)
        if table_name in conflicts and table_role:
            table.role = table_role
        for col_name, (col_role, col_desc, _sensitivity) in col_classes.items():
            col = table.columns.get(col_name)
            if col is None:
                continue
            conflict_key = f"{table_name}.{col_name}"
            if conflict_key in conflicts:
                if col_desc:
                    DescriptionOwner.set_on(col, col_desc, reconcile_owner)
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


def intent_column_member_coverage_ineligible_reason(intent: RuntimeIntent, schema: SchemaGraph) -> str | None:
    """Return an ineligible reason when referenced columns are not jointly held on one member."""
    for table_name in intent.tables or []:
        table = schema.tables.get(table_name)
        if table is None or not table.column_member_sources:
            continue
        needed = _referenced_columns_for_table(intent, table_name)
        covered_cols = [col for col in needed if col in table.column_member_sources]
        if not covered_cols:
            continue
        all_members: set[str] = set(table.member_source_ids or ())
        if not all_members:
            for member_holders in table.column_member_sources.values():
                all_members.update(member_holders)
        common: set[str] | None = None
        for col_name in covered_cols:
            holders = set(table.column_member_sources.get(col_name, []))
            common = holders if common is None else common & holders
        if common is not None and not common:
            for col_name in sorted(covered_cols):
                holders = set(table.column_member_sources.get(col_name, []))
                lacking = sorted(all_members - holders)
                if lacking:
                    return (
                        f"union logical column {col_name!r} on {table_name!r} "
                        f"is not present on members: {', '.join(lacking)}"
                    )
            return "projection columns are not held by any single member"
    return None


def _intent_lacks_column_member_coverage(intent: RuntimeIntent, schema: SchemaGraph) -> bool:
    return intent_column_member_coverage_ineligible_reason(intent, schema) is not None


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
            if len(unaliased) > len(alias_keys):
                raise FederationConfigError(
                    f"federation member {source_id!r} has {len(unaliased)} unaliased physical table(s) "
                    f"but only {len(alias_keys)} namespace alias(es) to pair"
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


def _assign_collapse_staging_composite_names(
    composite_names: dict[tuple[str, str], str],
    mappings: FederationMappings,
) -> None:
    """Give colliding members unique staging composite names before logical-table collapse."""
    by_composite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for member_key, composite in composite_names.items():
        by_composite[composite].append(member_key)
    for member_keys in by_composite.values():
        if len(member_keys) <= 1:
            continue
        member_set = frozenset(member_keys)
        if not _collision_resolved_by_logical_tables(member_set, mappings):
            continue
        for source_id, phys_name in sorted(member_keys):
            composite_names[(source_id, phys_name)] = f"__federation_stage__{source_id}__{phys_name}"


def _collision_resolved_by_logical_tables(members: frozenset[tuple[str, str]], mappings: FederationMappings) -> bool:
    for mapping in mappings.logical_tables:
        if mapping.semantics not in ("union", "replica"):
            continue
        declared = frozenset((member.source, member.table) for member in mapping.members)
        if members <= declared:
            return True
    return False


def composite_physical_member_refs(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map each final composite table name to contributing ``(source_id, physical_table)`` pairs. Used by consumer federation open to privilege-probe member engines against owner physical names, then subset the owner composite the same way a single engine subsets owner cache."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    composite_names = _resolve_composite_table_names(member_graphs, manifest)
    _assign_collapse_staging_composite_names(composite_names, mappings)
    logical_member_keys: set[tuple[str, str]] = set()
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for mapping in mappings.logical_tables:
        refs = tuple((str(m.source), str(m.table)) for m in mapping.members if m.source and m.table)
        if not refs:
            continue
        logical_member_keys.update(refs)
        out[str(mapping.logical)] = refs
    for (source_id, phys_name), cname in composite_names.items():
        key = (str(source_id), str(phys_name))
        if key in logical_member_keys:
            continue
        name = str(cname)
        if name.startswith("__federation_stage__"):
            continue
        out.setdefault(name, (key,))
    return out


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
            "identifier casing collision across federation members; resolve with a logical_tables mapping or an explicit alias: "
            + "; ".join(casing_errors)
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
            "table name collision across federation members; resolve with a logical_tables mapping or an explicit alias: "
            + "; ".join(collision_errors)
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
    """Return logical union-key columns, preferring primary-key identity when shared."""
    logical_sets: list[set[str]] = []
    for member, table in zip(mapping.members, member_tables, strict=True):
        if member.columns:
            logical_sets.append(set(member.columns.keys()))
        else:
            logical_sets.append(set(table.columns.keys()))
    common = set.intersection(*logical_sets) if logical_sets else set()
    if not common:
        cols = sorted({logical for member in mapping.members for logical in member.columns})
        if cols:
            return cols
        primary_keys = [tuple(table.primary_key or ()) for table in member_tables]
        if primary_keys and all(pk == primary_keys[0] for pk in primary_keys) and primary_keys[0]:
            return list(primary_keys[0])
        return []

    def _logical_resolves_to_pk(logical: str) -> bool:
        for member, table in zip(mapping.members, member_tables, strict=True):
            physical = member.columns.get(logical) if member.columns else logical
            if not physical:
                physical = logical
            pk = tuple(table.primary_key or ())
            if physical not in pk and logical not in pk:
                return False
        return True

    preferred = sorted(logical for logical in common if _logical_resolves_to_pk(logical))
    if preferred:
        return preferred + sorted(common - set(preferred))
    return sorted(common)


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
    for member, table in zip(mapping.members, member_tables, strict=True):
        phys_col = member.columns.get(key_col) or key_col
        meta = table.columns.get(phys_col)
        if meta is None:
            raise FederationDeclarationError(
                f"union logical table {mapping.logical!r} missing key column {key_col!r} "
                f"on member {member.source}.{member.table}"
            )
        effective_row_count = max(int(meta.row_count or 0), int(table.row_count or 0))
        if effective_row_count > 0 or meta.value_overlap_sample:
            profiled = True
        member_metas.append((member.source, meta))
    if not profiled:
        raise FederationDeclarationError(
            f"union logical table {mapping.logical!r} disjointness could not be established for key "
            f"{key_col!r}: members lack row_count or value_overlap_sample profiling"
        )
    missing_samples = [source for source, meta in member_metas if not meta.value_overlap_sample]
    if missing_samples:
        raise FederationDeclarationError(
            f"union logical table {mapping.logical!r} disjointness could not be established for key "
            f"{key_col!r}: members lack value_overlap_sample profiling: {', '.join(sorted(missing_samples))}"
        )
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
        _assert_collapsed_table_metadata_agrees(member_tables, mapping.logical)
        _assert_member_grain_equivalence(mapping, member_tables)
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
            label = f"logical table {mapping.logical!r} column {col_name!r}"
            _assert_collapsed_column_metadata_agrees(candidates, label, semantics=mapping.semantics)
            if mapping.semantics == "union":
                _assert_union_column_timestamp_awareness_agrees(candidates, sources, label)
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
        collapsed_desc, collapsed_owner = DescriptionOwner.resolve(
            *((table.description, table.description_owner) for table in member_tables)
        )
        collapsed_base = _base_description_from_authoritative(
            member_tables,
            member_sources=[member.source for member in mapping.members],
            authoritative_source=authoritative,
        )
        template_table = member_tables[0]
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
            "kind": template_table.kind,
            "partition_columns": list(template_table.partition_columns or []),
            "partition_type": template_table.partition_type,
            "require_partition_filter": bool(template_table.require_partition_filter),
            "clustering_fields": list(template_table.clustering_fields or []),
            "clustering_key": template_table.clustering_key,
            "distkey": template_table.distkey,
            "sortkey": list(template_table.sortkey or []),
            "diststyle": template_table.diststyle,
            "indexed_columns": list(template_table.indexed_columns or []),
            "size_mb": template_table.size_mb,
            "encoded": template_table.encoded,
            "quote_decision": template_table.quote_decision,
            "role": template_table.role,
            "row_count": row_count,
            "role_owner": template_table.role_owner,
            "composite_descriptive_ratios": dict(template_table.composite_descriptive_ratios or {}),
            "_user_semantic_neighbors": list(template_table._user_semantic_neighbors or []),
            "base_description": collapsed_base,
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
                DescriptionOwner.set_on(primary, collapsed_desc, collapsed_owner or DescriptionOwner.CATALOG)
            primary.base_description = collapsed_base
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


def table_source_index(
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


def intent_table_sources(
    tables: Iterable[str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> set[str]:
    sources: set[str] = set()
    for table in tables:
        sources.update(sources_for_table(table, manifest, mappings, source_by_table, schema))
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


def source_join_key_is_unique(
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
