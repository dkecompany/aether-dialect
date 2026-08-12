"""Federation probe, coordinator execution, frames, topology persist, and plan-template CRUD."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from platformdirs import user_data_dir
from sqlalchemy import text

from ._config import CsvRuntimeConfig, FederationLimits
from ._constants import (
    AETHERSPACES_SEGMENT,
    ARROW_RESULT_READER_KINDS,
    ARTIFACT_DIR_MODE,
    ARTIFACT_DIRECTORY_SEGMENT,
    ARTIFACT_FILE_MODE,
    DIAGNOSTIC_CODE_COORDINATOR_LIMITS,
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK,
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED,
    DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS,
    FEDERATION_ARTIFACT_FORMAT_VERSION,
    FEDERATION_COMBINE_SEMI_KIND,
    FEDERATION_JOIN_FEEDBACK_PREFIX,
    FEDERATION_JOIN_FEEDBACK_SEGMENT,
    FEDERATION_MAPPINGS_APPLIED_FILENAME,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_MEMBER_MANIFEST_FILENAME,
    FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP,
    FEDERATION_PLAN_TEMPLATE_FILE_CAP,
    FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION,
    FEDERATION_SOURCE_STORAGE_PREFIX,
    FEDERATION_STORAGE_PREFIX,
    FEDERATION_TEMPLATES_SEGMENT,
    FILE_ENGINE_NAMES,
    MASTER_AETHERSPACE_NAME,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_ACTION_DESTRUCTIVE,
    MIGRATION_MAP_ACTION_REMAP,
    MIN_COMPATIBLE_PACKAGE_VERSION,
    STRUCTURE_APPLIED_SUFFIX,
    STRUCTURE_APPLIED_TIMESTAMP_FORMAT,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_SEGMENT,
    VALID_GRAINS,
)
from ._constants_runtime import (
    INELIGIBLE_ANSWERABLE_HINTS_BY_CODE,
    REPHRASE_HINT_MESSAGES,
)
from ._contracts_base import (
    ArrayStorageKind,
    ConfigError,
    FederationCapExceededError,
    FederationConfigError,
    FederationDeclarationError,
    FederationIneligibleError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationMalformedMemberAnswerError,
    FederationMappingsAppliedSidecarError,
    FederationMemberExecutionError,
    FederationMemberProbeError,
    FederationMemberUnprofilableError,
    FederationPartialFailureError,
    FederationRuntimeError,
    FederationTopologyChange,
    FederationTopologyReport,
    FederationTurnCancelledError,
    MigrationPendingError,
    MigrationTier,
    NormalizedExpr,
    OwnerOnlyOperationError,
    PredicateGroup,
    SchemaRole,
    StructureReport,
    WhereParam,
)
from ._contracts_core import (
    AnchoredTemporalBind,
    CoordinatorMemberFrame,
    FederatedPlan,
    FederatedPrepareOutcome,
    FederatedStage,
    FederationExecutionWave,
    FederationMemberResolvedLimits,
    ResidualSpec,
    RuntimeIntent,
    SourceStep,
)
from ._contracts_schema import (
    FederationCoordinatorConfig,
    FederationCrossSourceJoin,
    FederationManifest,
    FederationMappings,
    FederationMigrationMap,
    FederationPlanTemplate,
    FederationQualifiedRename,
    FederationSourceBinding,
    LogicalColumnMapping,
    LogicalTableMapping,
    LogicalTableMember,
    PersistedFederationInspection,
    SchemaGraph,
    TableMetadata,
)
from ._data_quality import parse_source_selections, validate_upload_sources
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._dialect_sqlglot_helper import SqlglotEngineDialect
from ._federation_compose import (
    federation_composite_semantic_edges_hash,
    mapping_member_source_by_table,
    physical_table_source,
    split_qualified_column,
)
from ._federation_manifest import (
    assert_federation_member_graph_profiled,
    assign_cte_sources,
    binding_from_member_engine,
    cross_source_join_hash_entry,
    cte_probe_join_keys,
    derive_table_namespace,
    engine_connection_federation_source_storage_slug,
    federation_artifact_manifest_view,
    federation_artifact_paths,
    federation_drifted_member_source_ids,
    federation_manifest_document,
    federation_member_hash_tuple,
    federation_member_tuple_hash,
    federation_persist_quad_coherent,
    federation_residual_column_headers,
    federation_source_storage_slug,
    hydrate_persisted_federation_manifest,
    load_federation_artifact_manifest_dict,
    load_federation_mappings_from_path,
    manifest_hash,
    mappings_hash,
    namespace_from_composite_schema,
    normalize_stored_member_hash_row,
    parse_federation_manifest,
    plan_template_row_steps_reference_sources,
    sanitize_plan_template_row_member_template_ids,
    stamp_member_graph_source_id,
)
from ._federation_plan import (
    aggregate_identity_row_for_residual,
    build_combine_join_tree,
    combine_select_column_names,
    coordinator_residual_bind_map,
    derive_execution_order_from_stages,
    derive_federation_stages_in_order,
    effective_union_specs,
    enforce_coordinator_result_grain,
    member_stage_for_source,
    render_combine_select_keyword,
    render_federation_glue,
    residual_is_aggregate_only,
    resolve_member_limits_for_source,
    resolve_source_column_table,
    rewrite_federated_residual_aggregate_fan_out,
    schema_column_duckdb_type,
    source_by_table_from_schema,
    unqualified_column_name,
    validate_federated_residual_aggregate_fan_out,
)
from ._intent_bind import check_qualified_refs_exist, join_path_segments_fingerprint_runtime
from ._intent_expr import collect_intent_referenced_param_keys
from ._schema_finalize import (
    apply_structure_to_graph,
    compute_metadata_hash,
    dump_structure_to_path,
    finalize_with_structure,
    load_structure_document_file,
    load_structure_sidecar,
    save_structure_sidecar,
    user_added_fks_dump,
    user_added_pks_dump,
)
from ._schema_graph import (
    classify_migration_tier,
)
from ._schema_reflect import resolve_federation_qualified_ref
from ._utils import (
    active_federation_limits,
    coerce_format_version,
    cost_cap_active,
    format_versions_match,
    normalize_question,
    notify,
    reconcile_execute_bind_params,
    require_driver,
    require_exact_keys,
    stable_json,
)
from ._utils_artifacts import (
    artifact_lock,
    artifact_package_version_string,
    mark_connection_poisoned,
    read_gzip_json,
    refresh_migration_simulation_caches,
    wipe_filenames,
    write_gzip_json_atomic,
    write_text_atomic,
)
from ._utils_intent import flatten_param_values, intent_key
from ._validation_sql import assert_residual_execution_parameters_validated, validate_semantics, validate_sql


def lookup_federation_plan_template_for_question(federation_dir: str, q_norm: str) -> FederationPlanTemplate | None:
    """Return a stored federation plan whose accepted questions include *q_norm*."""
    if not federation_dir or not q_norm:
        return None
    for template in load_federation_plan_templates(federation_dir).values():
        if q_norm in template.accepted_questions:
            return template
    return None


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
        kept: dict[str, Any] = {}
        changed = False
        for plan_id, row in loaded.items():
            if not isinstance(row, dict):
                kept[plan_id] = row
                continue
            if plan_template_row_steps_reference_sources(row, removed_source_ids):
                changed = True
                continue
            sanitized = sanitize_plan_template_row_member_template_ids(row, removed_source_ids)
            if sanitized is not None:
                kept[plan_id] = sanitized
                changed = True
            else:
                kept[plan_id] = row
        if not changed:
            return
        if kept:
            _write_federation_json_atomic(path, kept)
        else:
            os.remove(path)


def member_guard_limit_kwargs(manifest: FederationManifest | None, source_id: str) -> dict[str, int | float | None]:
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
        names.append(unqualified_column_name(key))
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
        if kind == FEDERATION_COMBINE_SEMI_KIND:
            continue
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


def federation_member_connection_slug(manifest: FederationManifest | None, source_id: str) -> str:
    """Return the connection slug used to group member execution batches."""
    if manifest is not None:
        for binding in manifest.sources:
            if binding.source_id == source_id:
                return engine_connection_federation_source_storage_slug(binding)
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
        if (engine_type or "").strip().lower() not in FILE_ENGINE_NAMES:
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
    mapping_sources = mapping_member_source_by_table(mappings)
    live_checks: list[tuple[str, str, str]] = []
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
            live_checks.append((member.source, member.table, ""))
            for _logical_col, phys_col in member.columns.items():
                if phys_col not in src_table.columns:
                    raise FederationConfigError(
                        f"declared column {member.source}.{member.table}.{phys_col} missing from member schema"
                    )
                live_checks.append((member.source, member.table, phys_col))
    for col_map in mappings.logical_columns:
        for qual in col_map.members:
            tbl, col = split_qualified_column(qual, manifest=manifest, source_by_table=mapping_sources)
            source_id = physical_table_source(tbl, mappings)
            if not source_id:
                source_id = manifest.table_namespace.get(tbl, "")
            if not source_id:
                for candidate_id, engine in members.items():
                    graph = getattr(engine, "_schema_graph", None)
                    if isinstance(graph, SchemaGraph) and tbl in graph.tables:
                        source_id = candidate_id
                        break
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
            live_checks.append((source_id, tbl, col))
    _probe_live_declared_schema_objects(members, live_checks)


def _member_dialect_for_probe(engine: Any) -> Any | None:
    runtime_cfg = getattr(engine, "_runtime_config", None)
    engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
    if not engine_type and runtime_cfg is not None:
        engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
    if engine_type:
        stub = DialectRegistry.dialect_stub_for_engine(engine_type)
        if stub is not None:
            return stub
    dialect = getattr(engine, "_dialect", None)
    if dialect is not None:
        return dialect
    return None


def _quote_probe_relation(dialect: Any | None, name: str) -> str:
    if dialect is not None and hasattr(dialect, "quote_schema_qualified"):
        return cast(str, dialect.quote_schema_qualified(name))
    parts = [p for p in str(name).split(".") if p]
    if not parts:
        return str(name)
    return ".".join(f'"{part}"' for part in parts)


def _probe_live_declared_schema_objects(
    members: Mapping[str, Any],
    checks: Sequence[tuple[str, str, str]],
) -> None:
    """Verify declared tables/columns are readable from each member database."""
    seen: set[tuple[str, str, str]] = set()
    for source_id, table_name, column_name in checks:
        key = (source_id, table_name, column_name)
        if key in seen:
            continue
        seen.add(key)
        engine = members.get(source_id)
        if engine is None:
            continue
        engine_type = str(getattr(engine, "dialect", "") or "").strip().lower()
        if not engine_type:
            runtime_cfg = getattr(engine, "_runtime_config", None)
            engine_type = str(getattr(runtime_cfg, "engine", "") or "").strip().lower()
        if (engine_type or "").strip().lower() in FILE_ENGINE_NAMES:
            continue
        sa_engine = getattr(engine, "_execution_engine", None)
        if sa_engine is None:
            continue
        dialect = _member_dialect_for_probe(engine)
        quoted_table = _quote_probe_relation(dialect, table_name)
        if column_name:
            quoted_col = (
                dialect.quote_identifier(column_name)
                if dialect is not None and hasattr(dialect, "quote_identifier")
                else f'"{column_name}"'
            )
            probe_sql = f"SELECT {quoted_col} FROM {quoted_table} WHERE FALSE"
        else:
            probe_sql = f"SELECT 1 FROM {quoted_table} WHERE FALSE"
        try:
            with sa_engine.connect() as conn:
                conn.execute(text(probe_sql))
        except Exception as exc:
            raise FederationMemberProbeError(
                REPHRASE_HINT_MESSAGES["federation_member_probe_failed"],
                source_id=source_id,
            ) from exc


def _probe_member_session_timezone(conn: Any, engine: Any) -> str | None:
    dialect = _member_dialect_for_probe(engine)
    if dialect is None or not hasattr(dialect, "session_timezone_sql"):
        return None
    tz_sql = dialect.session_timezone_sql()
    if not tz_sql:
        return None
    try:
        row = conn.execute(text(tz_sql)).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    tz = str(row[0]).strip()
    return tz or None


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
        if (engine_type or "").strip().lower() in FILE_ENGINE_NAMES:
            continue
        sa_engine = getattr(engine, "_execution_engine", None)
        if sa_engine is None:
            sa_engine = getattr(getattr(engine, "_dialect", None), "engine", None)
        if sa_engine is None:
            raise FederationConfigError(f"federation member {connection_name!r} missing execution engine")
        try:
            with sa_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                session_timezone = _probe_member_session_timezone(conn, engine)
        except Exception as exc:
            raise FederationMemberProbeError(
                REPHRASE_HINT_MESSAGES["federation_member_probe_failed"],
                source_id=connection_name,
            ) from exc
        engine._session_timezone = session_timezone


def probe_federation_member_liveness(members: Mapping[str, Any]) -> None:
    """Re-check database-backed member connections before a federation turn."""
    probe_federation_member_connections(members)


def federation_user_facing_ineligible_message(reason: str) -> str:
    """Return neutral user-facing text for a compose ineligible reason."""
    _ = reason
    return REPHRASE_HINT_MESSAGES["restricted_question"]


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
    """Map a compose ineligible reason string to a stable reason code."""
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
        ("union logical column", "union_column_missing"),
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
    return ArrayStorageKind.UNKNOWN


def federation_ineligible_answerable_hint(reason: str | None) -> str | None:
    """Return a nearest answerable rephrase when *reason* names an ineligible federated shape."""
    code = federation_ineligible_reason_code(reason)
    if not code:
        return None
    return ineligible_answerable_hint_for_code(code)


def federation_coordinator_decimal_duckdb_type(precision: int, scale: int) -> str:
    """Render a DuckDB ``DECIMAL`` type for coordinator transfer from member metadata."""
    return f"DECIMAL({precision}, {scale})"


def ineligible_answerable_hint_for_code(code: str) -> str | None:
    """Return the nearest answerable rephrase hint for a federation ineligibility code."""
    return INELIGIBLE_ANSWERABLE_HINTS_BY_CODE.get(code)


def _dataframe_memory_bytes(frame: pd.DataFrame) -> int:
    """Return deep memory usage of *frame* in bytes for coordinator byte-cap checks."""
    usage = frame.memory_usage(deep=True)
    return int(usage.sum())


def _coordinator_member_memory_bytes(member: CoordinatorMemberFrame) -> int:
    """Measure coordinator member payload size; probe failures propagate."""
    if member.kind == "arrow":
        return int(member.table.nbytes)
    return _dataframe_memory_bytes(member.table)


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
            if unqualified_column_name(key) == column_name:
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
            return schema_column_duckdb_type(
                meta.data_type,
                column_meta=meta,
                column_name=col,
                source_id=source_id,
            )
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
                if data_type and schema_column_duckdb_type(data_type, column_meta=meta, column_name=lookup_col) is None:
                    raise FederationDeclarationError(
                        f"federation coordinator column {lookup_col!r} has unsupported data_type "
                        f"{data_type!r} for member {source_id!r}"
                    )
                if data_type:
                    mapped = schema_column_duckdb_type(
                        data_type, column_meta=meta, column_name=lookup_col, source_id=source_id
                    )
                    if mapped is not None:
                        types.append((col_name, mapped))
                        continue
        types.append((col_name, "VARCHAR"))
    return types


def _import_coordinator_duckdb() -> Any:
    require_driver("duckdb")
    return importlib.import_module("duckdb")


def _coordinator_pyarrow_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _emit_coordinator_arrow_spill_fallback(*, reg_name: str, row_count: int) -> None:
    notify(
        "federation coordinator PyArrow unavailable; using in-memory coordinator transfer",
        stage="federation",
        code=DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK,
        level="info",
        source_id="coordinator",
        details=(
            ("phase", "transfer"),
            ("relation", reg_name),
            ("row_count", str(row_count)),
        ),
    )


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
        match = re.search(r"\((\d+)\s*,\s*(\d+)\)", dtype)
        if match:
            return pa.decimal128(int(match.group(1)), int(match.group(2)))
        return pa.decimal128(38, 9)
    if dtype.startswith("DOUBLE") or dtype.startswith("FLOAT8"):
        return pa.float64()
    if dtype.startswith("REAL") or dtype.startswith("FLOAT4"):
        return pa.float32()
    if dtype.startswith("BOOL"):
        return pa.bool_()
    if dtype.startswith("TIMESTAMP WITH TIME ZONE"):
        return pa.timestamp("us", tz="UTC")
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
    col_defs = ", ".join(f"{Dialect.sqlglot_quote_identifier(col)} {dtype}" for col, dtype in column_types)
    return f"CREATE OR REPLACE TABLE {Dialect.sqlglot_quote_identifier(reg_name)} ({col_defs})"


def _insert_coordinator_typed_frame(
    conn: Any, reg_name: str, frame: pd.DataFrame, column_types: Sequence[tuple[str, str]]
) -> None:
    if frame.empty:
        return
    columns = [col for col, _dtype in column_types]
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(Dialect.sqlglot_quote_identifier(col) for col in columns)
    insert_sql = f"INSERT INTO {Dialect.sqlglot_quote_identifier(reg_name)} ({col_sql}) VALUES ({placeholders})"
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
        f"CAST({Dialect.sqlglot_quote_identifier(col)} AS {dtype}) AS {Dialect.sqlglot_quote_identifier(col)}"
        for col, dtype in column_types
    )
    conn.execute(f"DROP TABLE IF EXISTS {Dialect.sqlglot_quote_identifier(reg_name)}")
    conn.execute(_create_coordinator_typed_table_sql(reg_name, column_types))
    if len(bounded) <= spill_threshold or not _coordinator_pyarrow_available():
        if len(bounded) > spill_threshold and not _coordinator_pyarrow_available():
            _emit_coordinator_arrow_spill_fallback(reg_name=reg_name, row_count=len(bounded))
        _insert_coordinator_typed_frame(conn, reg_name, bounded, column_types)
        return
    os.makedirs(spill_dir, exist_ok=True)
    spill_path = os.path.join(spill_dir, f"{reg_name}.parquet")
    _write_coordinator_spill_parquet(bounded, spill_path, column_types)
    if spill_files_created is not None:
        spill_files_created.append(spill_path)
    conn.execute(
        f"CREATE OR REPLACE VIEW {Dialect.sqlglot_quote_identifier(reg_name)} AS "
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
        f"CAST({Dialect.sqlglot_quote_identifier(col)} AS {dtype}) AS {Dialect.sqlglot_quote_identifier(col)}"
        for col, dtype in column_types
    )
    staging = f"__{reg_name}_arrow"
    try:
        conn.unregister(staging)
    except (OSError, AttributeError, TypeError):
        pass
    conn.register(staging, arrow_table)
    conn.execute(f"DROP TABLE IF EXISTS {Dialect.sqlglot_quote_identifier(reg_name)}")
    if row_count <= spill_threshold or not _coordinator_pyarrow_available():
        if row_count > spill_threshold and not _coordinator_pyarrow_available():
            _emit_coordinator_arrow_spill_fallback(reg_name=reg_name, row_count=row_count)
        conn.execute(_create_coordinator_typed_table_sql(reg_name, column_types))
        conn.execute(f"INSERT INTO {Dialect.sqlglot_quote_identifier(reg_name)} SELECT {typed_select} FROM {staging}")
        conn.unregister(staging)
        return
    os.makedirs(spill_dir, exist_ok=True)
    spill_path = os.path.join(spill_dir, f"{reg_name}.parquet")
    _write_coordinator_spill_parquet_arrow(arrow_table, spill_path, column_types)
    if spill_files_created is not None:
        spill_files_created.append(spill_path)
    conn.unregister(staging)
    conn.execute(
        f"CREATE OR REPLACE VIEW {Dialect.sqlglot_quote_identifier(reg_name)} AS "
        f"SELECT {typed_select} FROM read_parquet({_quote_sql_string(spill_path)})"
    )


def _quote_sql_string(value: str) -> str:
    duckdb_cls = DialectRegistry.get_dialect_class("duckdb")
    return duckdb_cls.__new__(duckdb_cls).quote_string_literal(value)


def _validate_coordinator_glue_sql(
    sql: str, bind_map: Mapping[str, Any] | None, *, schema: SchemaGraph | None = None, conn: Any | None = None
) -> None:
    """Validate coordinator glue SQL through the DuckDB dialect gate."""
    dialect = DialectRegistry.get_dialect("duckdb", native_connection=conn)
    ok, err, _cat, _diags = validate_sql(dialect, sql, dict(bind_map or {}), schema=schema, allow_union=True)
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
    return start + (int(plan_timeout_ms or 0) / 1000.0)


def enforce_federation_plan_timeout(deadline: float | None, *, started_at: float) -> None:
    """Raise when the federated plan wall-clock budget has been exhausted."""
    if deadline is None:
        return
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    timeout_ms = int((deadline - started_at) * 1000)
    if time.perf_counter() >= deadline:
        raise federation_plan_timeout_error(elapsed_ms, timeout_ms)


def _coordinator_result_to_dataframe(result: Any) -> pd.DataFrame:
    """Materialize coordinator DuckDB results without widening exact numerics to float. ``SessionStep.data`` may use object dtype for DECIMAL columns so Python :class:`decimal.Decimal` values and SQL NULL (``None``) survive egress."""
    description = getattr(result, "description", None)
    columns = [str(col[0]) for col in description] if description else []
    if columns:
        seen: dict[str, int] = {}
        unique_columns: list[str] = []
        for name in columns:
            n = seen.get(name, 0)
            seen[name] = n + 1
            unique_columns.append(name if n == 0 else f"{name}_{n}")
        columns = unique_columns
    rows = result.fetchall() if hasattr(result, "fetchall") else []
    if not rows:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    frame = pd.DataFrame([tuple(row) for row in rows], columns=columns or None)
    return frame


def _execute_coordinator_sql_with_timeout(
    conn: Any,
    sql: str,
    bind_map: Mapping[str, Any] | None,
    *,
    timeout_ms: int | None,
) -> Any:
    """Execute coordinator DuckDB SQL with an optional wall-clock timeout."""
    params = dict(bind_map or {})
    exec_sql = sql
    exec_args: dict[str, Any] | list[Any] = params
    if params:
        exec_sql, exec_args = SqlglotEngineDialect.bind_colon_parameters_for_duckdb(sql, params)
    if timeout_ms is None or not cost_cap_active(timeout_ms):
        return conn.execute(exec_sql, exec_args or {})
    resolved_timeout_ms = int(timeout_ms)
    deadline = time.perf_counter() + (resolved_timeout_ms / 1000.0)
    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def _run() -> None:
        try:
            result_holder.append(conn.execute(exec_sql, exec_args or {}))
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
            except (OSError, AttributeError, RuntimeError, TypeError):
                pass
        mark_connection_poisoned(conn)
        worker.join(timeout=1.0)
        raise federation_coordinator_timeout_error(resolved_timeout_ms)
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise federation_coordinator_timeout_error(resolved_timeout_ms)
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
    owners = assign_cte_sources(plan.lifted_probe_ctes, source_by_table)
    for cte in plan.lifted_probe_ctes:
        owner = owners.get(cte.cte_name or "")
        if not owner:
            continue
        frame = frames.get(owner)
        if frame is None:
            continue
        for key in cte_probe_join_keys(cte):
            column = unqualified_column_name(key)
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
    federation_dir: str | None = None,
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
        if plan.residual is not None and residual_is_aggregate_only(plan.residual):
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
    conn = _import_coordinator_duckdb().connect(":memory:")
    owned_coordinator_temp = False
    coordinator_temp_directory = ""
    owned_spill = False
    spill_path = spill_dir or ""
    spill_files_created: list[str] = []
    try:
        _, coordinator_temp_directory, _, owned_coordinator_temp = _configure_federation_coordinator_connection(
            conn,
            federation_dir=federation_dir,
            spill_dir=spill_dir,
        )
        step_ids: dict[str, str] = {}
        owned_spill = spill_dir is None
        if spill_dir is None:
            temp_root = tempfile.gettempdir()
            _require_coordinator_temp_directory_writable(temp_root)
            spill_path = tempfile.mkdtemp(prefix="aetherdialect_fed_spill_")
        else:
            spill_path = spill_dir
        os.makedirs(spill_path, mode=0o700, exist_ok=True)
        source_count = len(frames)
        bind_map = coordinator_residual_bind_map(plan, dict(param_values or {}))
        if schema is not None and plan.residual is not None:
            assert_residual_execution_parameters_validated(plan.residual, bind_map, schema)
        total_rows = 0
        total_bytes = 0
        member_row_counts = {
            source_id: normalize_coordinator_member_input(frame).row_count() for source_id, frame in frames.items()
        }
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
        if schema is not None:
            plan = rewrite_federated_residual_aggregate_fan_out(plan, schema)
        glue = render_federation_glue(plan, step_ids, schema=schema, param_values=param_values)
        if not glue:
            if source_count == 1:
                only_reg = next(iter(step_ids.values()))
                explicit_cols = combine_select_column_names(plan)
                select_kw = render_combine_select_keyword(explicit_cols)
                single_sql = f"SELECT {select_kw} FROM {only_reg}"
                _validate_coordinator_glue_sql(single_sql, {}, schema=schema, conn=conn)
                result = _coordinator_result_to_dataframe(
                    _execute_coordinator_sql_with_timeout(conn, single_sql, {}, timeout_ms=glue_timeout)
                )
                result = _enforce_federation_row_cap(result, cap)
                enforce_coordinator_result_grain(result, plan)
                return result
            raise FederationRuntimeError("federation glue SQL is empty")
        _assert_combine_join_plan_structure(plan)
        exec_bind = reconcile_execute_bind_params(glue, bind_map) or {}
        _validate_coordinator_glue_sql(glue, exec_bind, schema=schema, conn=conn)
        try:
            result = _coordinator_result_to_dataframe(
                _execute_coordinator_sql_with_timeout(conn, glue, exec_bind, timeout_ms=glue_timeout)
            )
        except FederationCapExceededError:
            raise
        except Exception as exc:
            raise FederationRuntimeError(f"coordinator glue execution failed: {exc}") from exc
        if result.empty and plan.residual is not None and residual_is_aggregate_only(plan.residual):
            headers = federation_residual_column_headers(plan)
            identity = aggregate_identity_row_for_residual(plan.residual)
            if headers and len(headers) == len(identity):
                result = pd.DataFrame([identity], columns=list(headers))
            else:
                result = pd.DataFrame([identity])
        result = _enforce_federation_row_cap(result, cap)
        combine_row_count = len(result)
        validate_coordinator_join_fan_out(plan, member_row_counts, len(result), combine_row_count=combine_row_count)
        if schema is not None:
            validate_federated_residual_aggregate_fan_out(plan, schema)
        enforce_coordinator_result_grain(result, plan)
        return result
    finally:
        conn.close()
        if owned_spill and spill_path:
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
        if owned_coordinator_temp and coordinator_temp_directory:
            try:
                shutil.rmtree(coordinator_temp_directory)
            except OSError as exc:
                raise FederationRuntimeError(f"failed to clean federation coordinator temp directory: {exc}") from exc


def _assert_combine_join_plan_structure(plan: FederatedPlan) -> None:
    """Assert declared combine joins form a connected join tree from plan IR."""
    if not isinstance(plan.combine, tuple) or not plan.combine:
        return
    sources = {step.source_id for step in plan.steps}
    if len(sources) < 2:
        return
    tree = build_combine_join_tree(plan.combine, sources)
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
        filters = len(PredicateGroup.where_leaves(step.sub_intent.where))
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


def resolve_member_row_cap(
    manifest: Any,
    source_id: str,
    limits: FederationLimits | None,
) -> int | None:
    """Resolve one member row cap: member declaration, then coordinator, then federation limits."""
    member_cap: int | None = None
    sources = getattr(manifest, "sources", None)
    if isinstance(sources, Mapping):
        binding = sources.get(source_id)
        binding_limits = getattr(binding, "limits", None) if binding is not None else None
        raw = getattr(binding_limits, "row_cap", None) if binding_limits is not None else None
        if raw is not None:
            member_cap = int(raw)
    else:
        for binding in sources or ():
            if getattr(binding, "source_id", None) != source_id:
                continue
            binding_limits = getattr(binding, "limits", None)
            raw = getattr(binding_limits, "row_cap", None) if binding_limits is not None else None
            if raw is not None:
                member_cap = int(raw)
            break
    if member_cap is not None:
        return member_cap
    coordinator = getattr(manifest, "coordinator", None)
    if coordinator is not None:
        for attr in ("default_source_row_cap", "row_cap"):
            raw = getattr(coordinator, attr, None)
            if raw is not None:
                return int(raw)
    if limits is not None:
        member_row_cap = getattr(limits, "member_row_cap", None)
        if member_row_cap is not None:
            return int(member_row_cap)
    return None


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
    _require_coordinator_temp_directory_writable(path)
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise FederationRuntimeError(f"failed to clean federation coordinator spill directory: {exc}") from exc
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def federation_coordinator_temp_dir(federation_dir: str | None) -> str | None:
    """Return the DuckDB temp directory under a federation artifact tree."""
    if not federation_dir or not str(federation_dir).strip():
        return None
    path = os.path.join(str(federation_dir), "coordinator_temp")
    _require_coordinator_temp_directory_writable(path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _atomic_write_directory(path: str) -> tuple[str, str]:
    """Resolve *path* absolutely and return ``(absolute_path, parent_directory)``."""
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    return abs_path, directory


def _format_duckdb_byte_setting(byte_count: int) -> str:
    """Format a byte count for DuckDB ``SET`` settings that accept size literals."""
    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    gib = 1024**3
    mib = 1024**2
    if byte_count % gib == 0:
        return f"{byte_count // gib}GB"
    if byte_count % mib == 0:
        return f"{byte_count // mib}MB"
    return f"{byte_count}B"


def _duckdb_sql_string_literal(value: str) -> str:
    escaped = value.replace("\\", "/").replace("'", "''")
    return f"'{escaped}'"


def _resolve_coordinator_temp_directory(
    limits: FederationLimits,
    *,
    federation_dir: str | None = None,
    spill_dir: str | None = None,
) -> tuple[str, bool]:
    """Return ``(temp_directory, owned)`` where *owned* means the caller must delete it."""
    if limits.coordinator_temp_dir:
        path = os.path.abspath(limits.coordinator_temp_dir)
        _require_coordinator_temp_directory_writable(path)
        os.makedirs(path, mode=0o700, exist_ok=True)
        return path, False
    resolved_federation_dir = federation_dir
    if resolved_federation_dir is None and spill_dir:
        spill_abs = os.path.abspath(spill_dir)
        if os.path.basename(os.path.normpath(spill_abs)) == "coordinator_spill":
            resolved_federation_dir = os.path.dirname(spill_abs)
    if resolved_federation_dir:
        maybe_path = federation_coordinator_temp_dir(resolved_federation_dir)
        if maybe_path:
            return maybe_path, False
    temp_root = tempfile.gettempdir()
    _require_coordinator_temp_directory_writable(temp_root)
    path = tempfile.mkdtemp(prefix="aetherdialect_coordinator_temp_")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path, True


def _effective_federation_limits() -> FederationLimits:
    try:
        return active_federation_limits()
    except RuntimeError:
        return FederationLimits()


def _configure_federation_coordinator_connection(
    conn: Any,
    *,
    federation_dir: str | None = None,
    spill_dir: str | None = None,
) -> tuple[str, str, int, bool]:
    """Apply coordinator DuckDB limits; return memory, temp dir, threads, and temp ownership."""
    limits = _effective_federation_limits()
    temp_directory, owned_temp = _resolve_coordinator_temp_directory(
        limits,
        federation_dir=federation_dir,
        spill_dir=spill_dir,
    )
    memory_report = "default"
    if limits.coordinator_memory_limit_bytes is not None:
        memory_report = _format_duckdb_byte_setting(int(limits.coordinator_memory_limit_bytes))
        conn.execute(f"SET memory_limit={_duckdb_sql_string_literal(memory_report)}")
    conn.execute(f"SET temp_directory={_duckdb_sql_string_literal(temp_directory)}")
    threads = int(limits.coordinator_threads)
    conn.execute(f"SET threads={threads}")
    if limits.coordinator_spill_max_bytes is not None:
        spill_literal = _format_duckdb_byte_setting(int(limits.coordinator_spill_max_bytes))
        conn.execute(f"SET max_temp_directory_size={_duckdb_sql_string_literal(spill_literal)}")
    notify(
        "federation coordinator DuckDB limits configured",
        stage="federation",
        code=DIAGNOSTIC_CODE_COORDINATOR_LIMITS,
        level="info",
        source_id="coordinator",
        details=(
            ("phase", "plan"),
            ("memory_limit", memory_report),
            ("temp_directory", temp_directory),
            ("threads", str(threads)),
        ),
    )
    return memory_report, temp_directory, threads, owned_temp


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


def emit_federation_reduction_null_keys_diagnostic(
    dropped_count: int,
    *,
    column: str,
    source_id: str | None = None,
) -> None:
    """Emit when equality reduction drops rows whose join key is unknown."""
    if dropped_count <= 0:
        return
    notify(
        (f"federation reduction dropped {dropped_count} row(s) with unknown key on {column!r} before key transfer"),
        stage="federation",
        code=DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS,
        level="info",
        source_id=source_id,
        details=(
            ("phase", "prepare"),
            ("dropped_count", str(dropped_count)),
            ("column", column),
        ),
    )


def distinct_semijoin_keys(frame: pd.DataFrame | CoordinatorMemberFrame, column: str, *, cap: int) -> list[Any] | None:
    """Return distinct non-null keys when within *cap*; otherwise None (skip reduction). Rows whose key is unknown cannot participate in equality reduction; null keys are dropped before transfer."""
    if isinstance(frame, CoordinatorMemberFrame):
        return _distinct_semijoin_keys_arrow(frame, column, cap=cap)
    if column not in frame.columns:
        return None
    series = frame[column]
    null_count = int(series.isna().sum())
    if null_count:
        emit_federation_reduction_null_keys_diagnostic(null_count, column=column)
    series = series.dropna()
    if series.empty:
        return []
    keys = series.unique().tolist()
    if len(keys) > cap:
        return None
    return keys


def _distinct_semijoin_keys_arrow(member: CoordinatorMemberFrame, column: str, *, cap: int) -> list[Any] | None:
    """Return distinct non-null keys from an Arrow member frame when within *cap*."""
    import pyarrow.compute as pc

    col_name = unqualified_column_name(column)
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
    null_count = int(len(values) - len(pc_ops.drop_null(values)))
    if null_count:
        emit_federation_reduction_null_keys_diagnostic(null_count, column=column)
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
    filters = list((PredicateGroup.where_leaves(sub_intent.where)) or [])
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
        return FederationTopologyChange.MIXED
    if added:
        return FederationTopologyChange.ADD
    if removed:
        return FederationTopologyChange.REMOVE
    return FederationTopologyChange.NONE


def prune_federation_aliases(manifest: FederationManifest, *, active_source_ids: set[str]) -> FederationManifest:
    """Drop table aliases that reference sources absent from the federation."""
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
    """Drop cross-source joins that reference sources absent from the federation."""
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
    """Drop mapping members that reference sources absent from the federation."""
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
    if not isinstance(payload, Mapping):
        raise MigrationPendingError("federation_migration_map.json: root must be a JSON object")
    require_exact_keys(
        payload,
        allowed=frozenset(
            {
                "version",
                "action",
                "qualified_column_renames",
                "namespace_renames",
                "dropped_cross_source_joins",
            }
        ),
        required=frozenset({"version", "action"}),
        context="federation migration map",
    )
    version = coerce_format_version(payload.get("version", 0) or 0)
    if version != "1" and not format_versions_match(version, "1"):
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
        version=1,
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
    ts = datetime.now(UTC).strftime(STRUCTURE_APPLIED_TIMESTAMP_FORMAT)
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
    fb_dir = _federation_join_feedback_dir(federation_dir)
    if os.path.isdir(fb_dir):
        shutil.rmtree(fb_dir, ignore_errors=True)


def clear_federation_composite_template_store(federation_dir: str) -> bool:
    """Remove the composite partitioned template store under a federation tree."""
    store_dir = os.path.join(federation_dir, TEMPLATE_STORE_SEGMENT)
    legacy = os.path.join(federation_dir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
    existed = os.path.isdir(store_dir) or os.path.isfile(legacy)
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    wipe_filenames(federation_dir, (TEMPLATE_STORE_LEGACY_SINGLE_FILE,))
    return existed


def _member_logical_table_names(manifest: FederationManifest, source_id: str) -> tuple[str, ...]:
    """Return composite logical table names owned by *source_id*."""
    names: set[str] = set()
    for table_name, owner in manifest.table_namespace.items():
        if owner == source_id:
            names.add(table_name)
    for alias in manifest.aliases:
        if alias.source == source_id:
            names.add(alias.alias)
    return tuple(sorted(names))


def _purge_dropped_tables_from_aetherspace_snapshots(
    engine_dir: str,
    dropped_tables: tuple[str, ...],
) -> int:
    """Remove dropped table references from persisted aetherspace snapshot JSON files."""
    if not dropped_tables:
        return 0
    drop_tables = frozenset(dropped_tables)
    root = os.path.join(engine_dir, AETHERSPACES_SEGMENT)
    if not os.path.isdir(root):
        return 0
    updated = 0
    for entry in os.listdir(root):
        if Path(entry).suffix.lower() != ".json":
            continue
        stem = entry[: -len(".json")]
        if not stem or stem == MASTER_AETHERSPACE_NAME:
            continue
        path = os.path.join(root, entry)
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        tables = sorted({str(t) for t in (payload.get("tables") or ()) if str(t) not in drop_tables})
        columns = sorted(
            {
                str(c)
                for c in (payload.get("columns") or ())
                if not (str(c).count(".") == 1 and str(c).split(".", 1)[0] in drop_tables)
            }
        )
        table_descriptions = {
            str(k): v
            for k, v in dict(payload.get("table_descriptions") or {}).items()
            if str(k) not in drop_tables and isinstance(v, str) and v.strip()
        }
        column_meta = {
            str(k): v
            for k, v in dict(payload.get("column_meta") or {}).items()
            if not (str(k).count(".") == 1 and str(k).split(".", 1)[0] in drop_tables)
        }
        deny_objects = sorted({str(t) for t in (payload.get("deny_objects") or ()) if str(t) not in drop_tables})
        deny_columns = sorted(
            {
                str(c)
                for c in (payload.get("deny_columns") or ())
                if not (str(c).count(".") == 1 and str(c).split(".", 1)[0] in drop_tables)
            }
        )
        edited = {
            **payload,
            "tables": tables,
            "columns": columns,
            "table_descriptions": table_descriptions,
            "column_meta": column_meta,
            "deny_objects": deny_objects,
            "deny_columns": deny_columns,
        }
        if edited == payload:
            continue
        abs_path, directory = _atomic_write_directory(path)
        os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                delete=False,
            ) as tmp:
                json.dump(edited, tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, abs_path)
            updated += 1
        except OSError:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return updated


def _directory_tree_byte_size(path: str) -> int:
    """Return the total byte size of a file or directory tree."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _directory_is_writable(path: str) -> bool:
    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    target = abs_path if os.path.isdir(abs_path) else (os.path.dirname(abs_path) or abs_path)
    if not os.path.isdir(target):
        try:
            os.makedirs(target, mode=0o700, exist_ok=True)
        except OSError:
            return False
    if not os.access(target, os.W_OK):
        return False
    probe = os.path.join(target, ".aetherdialect_write_probe")
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("")
        os.remove(probe)
    except OSError:
        return False
    return True


def _require_writable_directory(path: str, *, message: str) -> None:
    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    if not _directory_is_writable(abs_path):
        raise ConfigError(message.format(path=abs_path))


def _require_default_artifacts_root_writable(path: str) -> None:
    _require_writable_directory(
        path,
        message=("default artifacts directory {path!r} is not writable; set an explicit artifacts_dir"),
    )


def _require_coordinator_temp_directory_writable(path: str) -> None:
    _require_writable_directory(
        path,
        message=(
            "coordinator temporary directory {path!r} is not writable; set FederationLimits.coordinator_temp_dir or ensure the system temporary directory is writable"
        ),
    )


def _federation_member_artifacts_root(artifacts_root: str | None) -> str:
    if artifacts_root and str(artifacts_root).strip():
        parent = os.path.abspath(os.path.expanduser(str(artifacts_root)))
    else:
        parent = user_data_dir(appname="aetherdialect", appauthor=False)
        _require_default_artifacts_root_writable(parent)
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT)


def _expected_member_artifacts_dir(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
    storage_slug: str | None = None,
) -> str:
    if storage_slug:
        return os.path.join(_federation_member_artifacts_root(artifacts_root), str(storage_slug))
    return federation_source_artifacts_dir(artifacts_root, binding, federation_id=federation_id)


def _assert_member_artifacts_dir_is_purgeable(
    member_artifacts_dir: str,
    *,
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    federation_id: str | None = None,
    storage_slug: str | None = None,
) -> None:
    """Refuse to purge directories that are not federation member artifact trees."""
    abs_member = os.path.abspath(os.path.expanduser(str(member_artifacts_dir)))
    expected_slug = str(storage_slug or "").strip() or federation_source_storage_slug(
        binding, federation_id=federation_id
    )
    if os.path.basename(abs_member) != expected_slug:
        raise FederationConfigError(
            f"refusing to purge unexpected federation member artifacts directory {member_artifacts_dir!r}; "
            f"expected slug {expected_slug!r}"
        )
    if not expected_slug.startswith(FEDERATION_SOURCE_STORAGE_PREFIX):
        raise FederationConfigError(
            f"refusing to purge federation member artifacts directory without {FEDERATION_SOURCE_STORAGE_PREFIX!r} prefix: "
            f"{member_artifacts_dir!r}"
        )
    if os.path.basename(os.path.dirname(abs_member)) != ARTIFACT_DIRECTORY_SEGMENT:
        raise FederationConfigError(
            f"refusing to purge federation member artifacts outside {ARTIFACT_DIRECTORY_SEGMENT!r} segment: "
            f"{member_artifacts_dir!r}"
        )
    candidate_roots = [_federation_member_artifacts_root(artifacts_root)]
    inferred_artifacts_root = os.path.dirname(os.path.dirname(abs_member))
    if inferred_artifacts_root:
        candidate_roots.append(_federation_member_artifacts_root(inferred_artifacts_root))
    if not any(abs_member.startswith(root + os.sep) for root in candidate_roots):
        raise FederationConfigError(
            f"refusing to purge federation member artifacts outside expected root: {member_artifacts_dir!r}"
        )


def federation_member_artifacts_dir_for_purge(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
    member_artifacts_dir: str | None = None,
) -> str:
    """Resolve the member artifact directory to purge, preferring a validated explicit path."""
    binding_dir = federation_source_artifacts_dir(artifacts_root, binding, federation_id=federation_id)
    candidate = str(member_artifacts_dir or "").strip()
    if not candidate:
        return binding_dir
    try:
        _assert_member_artifacts_dir_is_purgeable(
            candidate,
            artifacts_root=artifacts_root,
            binding=binding,
            federation_id=federation_id,
        )
    except FederationConfigError:
        return binding_dir
    return candidate


def purge_federation_member_artifacts(
    federation_dir: str,
    *,
    member_artifacts_dir: str,
    artifacts_root: str | None,
    source_id: str,
    member_engine: Any | None = None,
    manifest: FederationManifest | None = None,
    federation_id: str | None = None,
    storage_slug: str | None = None,
) -> tuple[str, int]:
    """Delete on-disk artifacts for one removed federation member and clear composite template shards."""
    binding: FederationSourceBinding | None = None
    if manifest is not None:
        binding = next((row for row in manifest.sources if row.source_id == source_id), None)
    if binding is None and member_engine is not None:
        try:
            binding = binding_from_member_engine(member_engine)
        except FederationConfigError:
            binding = None
    if binding is None:
        binding = FederationSourceBinding(
            source_id=source_id,
            engine=str(getattr(member_engine, "dialect", "duckdb") or "duckdb") if member_engine else "duckdb",
            connection=source_id,
        )
    fed_id = federation_id or (str(manifest.federation_id) if manifest is not None else None)
    _assert_member_artifacts_dir_is_purgeable(
        member_artifacts_dir,
        artifacts_root=artifacts_root,
        binding=binding,
        federation_id=fed_id,
        storage_slug=storage_slug,
    )
    abs_member = os.path.abspath(os.path.expanduser(str(member_artifacts_dir)))
    bytes_reclaimed = _directory_tree_byte_size(abs_member)
    with artifact_lock(federation_dir):
        extra_dir = getattr(member_engine, "_artifacts_dir", None) if member_engine is not None else None
        if extra_dir and os.path.isdir(str(extra_dir)):
            refresh_migration_simulation_caches(str(extra_dir))
        if os.path.isdir(abs_member):
            shutil.rmtree(abs_member, ignore_errors=True)
        if extra_dir and os.path.abspath(str(extra_dir)) != abs_member and os.path.isdir(str(extra_dir)):
            shutil.rmtree(str(extra_dir), ignore_errors=True)
        clear_federation_composite_template_store(federation_dir)
        dropped_tables = _member_logical_table_names(manifest, source_id) if manifest is not None else ()
        if dropped_tables:
            _purge_dropped_tables_from_aetherspace_snapshots(
                federation_dir,
                dropped_tables,
            )
        refresh_migration_simulation_caches(federation_dir)
    return abs_member, bytes_reclaimed


def purge_departed_federation_member_trees(
    federation_dir: str,
    *,
    artifacts_root: str | None,
    removed_source_ids: Sequence[str],
) -> None:
    """Remove artifact trees for federation members pruned during init- time shrink."""
    if not removed_source_ids:
        return
    stored = load_federation_artifact_manifest_dict(federation_artifact_paths(federation_dir)["artifact_manifest"])
    federation_id = str((stored or {}).get("federation_id", "") or "").strip() or None
    roster_by_source: dict[str, tuple[str, str, str, str]] = {}
    try:
        for row in load_persisted_federation_roster_rows(federation_dir):
            roster_by_source[str(row[0])] = row
    except FederationConfigError:
        pass
    for source_id in removed_source_ids:
        sid = str(source_id)
        roster_row = roster_by_source.get(sid)
        if roster_row is not None:
            connection = str(roster_row[1] or "")
            storage_slug = str(roster_row[2] or "")
            binding = FederationSourceBinding(
                source_id=sid,
                engine=_engine_type_from_federation_source_slug(storage_slug),
                connection=connection or sid,
            )
        else:
            storage_slug = ""
            binding = FederationSourceBinding(source_id=sid, engine="duckdb", connection=sid)
        member_dir = _expected_member_artifacts_dir(
            artifacts_root,
            binding,
            federation_id=federation_id,
            storage_slug=storage_slug or None,
        )
        removed_path, bytes_reclaimed = purge_federation_member_artifacts(
            federation_dir,
            member_artifacts_dir=member_dir,
            artifacts_root=artifacts_root,
            source_id=sid,
            federation_id=federation_id,
            storage_slug=storage_slug,
        )
        notify(
            f"Removed federation member {sid!r} artifacts from {removed_path!r}",
            stage="artifact",
            code=DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED,
            source_id=sid,
            details=(
                ("phase", "shrink"),
                ("bytes_reclaimed", str(bytes_reclaimed)),
            ),
        )


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


def _member_column_overlap_sample(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    qualified: str,
    *,
    mappings: FederationMappings | None = None,
) -> frozenset[str]:
    table_name, column_name = split_qualified_column(qualified, manifest=manifest)
    namespace = manifest.table_namespace or derive_table_namespace(member_graphs, mappings)
    source_id = namespace.get(table_name, "")
    if not source_id and mappings is not None:
        source_id = physical_table_source(table_name, mappings)
    if not source_id:
        return frozenset()
    graph = member_graphs.get(source_id)
    if graph is None:
        return frozenset()
    table = graph.tables.get(table_name)
    if table is None:
        return frozenset()
    meta = table.columns.get(column_name)
    if meta is None:
        return frozenset()
    return frozenset(str(value) for value in (meta.value_overlap_sample or []) if str(value))


def detect_unmapped_cross_source_references(
    member_graphs: Mapping[str, SchemaGraph],
    manifest: FederationManifest,
    *,
    mappings: FederationMappings | None = None,
) -> tuple[str, ...]:
    """Report declared cross-source join child values absent from the parent value sample."""
    messages: list[str] = []
    for join in manifest.cross_source_joins:
        left_sample = _member_column_overlap_sample(member_graphs, manifest, join.left, mappings=mappings)
        right_sample = _member_column_overlap_sample(member_graphs, manifest, join.right, mappings=mappings)
        if not left_sample or not right_sample:
            continue
        unmapped = sorted(left_sample - right_sample, key=str)
        if not unmapped:
            continue
        preview = ", ".join(unmapped[:5])
        if len(unmapped) > 5:
            preview = f"{preview}, ..."
        messages.append(f"cross_source_join {join.left!r} -> {join.right!r} has unmapped child values: {preview}")
    return tuple(messages)


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
                federation_source_storage_slug(binding, federation_id=manifest.federation_id),
                str(graph.schema_graph_id or ""),
            )
        )
    return tuple(rows)


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


def validate_federation_mappings_applied_sidecar(federation_dir: str, mappings: FederationMappings) -> None:
    """Refuse when the applied mappings sidecar references mappings absent from the live file."""
    paths = federation_artifact_paths(federation_dir)
    applied_path = paths["mappings_applied"]
    if not os.path.isfile(applied_path):
        return
    try:
        with open(applied_path, encoding="utf-8") as handle:
            applied = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationMappingsAppliedSidecarError(
            f"applied federation mappings sidecar at {applied_path!r} is unreadable: {exc}"
        ) from exc
    if not isinstance(applied, dict):
        raise FederationMappingsAppliedSidecarError(
            f"applied federation mappings sidecar at {applied_path!r} is not a JSON object"
        )
    live_columns = {col.logical for col in mappings.logical_columns}
    for entry in applied.get("logical_columns", []) or []:
        if not isinstance(entry, dict):
            continue
        logical = str(entry.get("logical", "") or "").strip()
        if logical and logical not in live_columns:
            raise FederationMappingsAppliedSidecarError(
                f"applied federation mappings sidecar references logical column {logical!r} "
                f"absent from {paths['mappings']!r}"
            )
    live_tables = {table.logical for table in mappings.logical_tables}
    for entry in applied.get("logical_tables", []) or []:
        if not isinstance(entry, dict):
            continue
        logical = str(entry.get("logical", "") or "").strip()
        if logical and logical not in live_tables:
            raise FederationMappingsAppliedSidecarError(
                f"applied federation mappings sidecar references logical table {logical!r} "
                f"absent from {paths['mappings']!r}"
            )


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
    abs_path, directory = _atomic_write_directory(path)
    os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".fed_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, abs_path)
        try:
            os.chmod(abs_path, ARTIFACT_FILE_MODE)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _backup_federation_live_artifact(path: str, federation_dir: str) -> str | None:
    if not os.path.isfile(path):
        return None
    fd, backup = tempfile.mkstemp(prefix=".rollback_", dir=federation_dir)
    os.close(fd)
    shutil.copy2(path, backup)
    return backup


def _commit_federation_staged_replaces(
    federation_dir: str,
    staged_pairs: Sequence[tuple[str, str]],
) -> None:
    """Replace live federation artifacts from pre-staged paths; roll back on failure."""
    backups: list[tuple[str | None, str]] = []
    committed: list[str] = []
    try:
        for _staging, live in staged_pairs:
            backups.append((_backup_federation_live_artifact(live, federation_dir), live))
        for staging, live in staged_pairs:
            if not os.path.isfile(staging):
                raise FederationConfigError(f"federation staging file missing at {staging!r}")
            os.replace(staging, live)
            committed.append(live)
            try:
                os.chmod(live, ARTIFACT_FILE_MODE)
            except OSError:
                pass
    except BaseException:
        for backup, live in reversed(backups):
            if live not in committed:
                continue
            if backup and os.path.isfile(backup):
                os.replace(backup, live)
            else:
                try:
                    os.remove(live)
                except OSError:
                    pass
        raise
    finally:
        for backup, _live in backups:
            if backup and os.path.isfile(backup):
                try:
                    os.unlink(backup)
                except OSError:
                    pass


def _commit_federation_persist_quad(
    federation_dir: str,
    *,
    manifest_path: str,
    manifest_payload: Mapping[str, Any],
    mappings_path: str,
    mappings_payload: Mapping[str, Any],
    composite_path: str,
    composite_payload: Mapping[str, Any],
    artifact_manifest_path: str,
    artifact_manifest_payload: Mapping[str, Any],
) -> None:
    """Stage federation manifest, mappings, composite, and artifact manifest, then commit atomically."""
    staging_dir = os.path.join(federation_dir, ".fed_quad_staging")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    try:
        manifest_staging = os.path.join(staging_dir, os.path.basename(manifest_path))
        mappings_staging = os.path.join(staging_dir, os.path.basename(mappings_path))
        composite_staging = os.path.join(staging_dir, os.path.basename(composite_path))
        artifact_staging = os.path.join(staging_dir, os.path.basename(artifact_manifest_path))
        _write_federation_json_atomic(manifest_staging, manifest_payload)
        _write_federation_json_atomic(mappings_staging, mappings_payload)
        write_gzip_json_atomic(composite_staging, composite_payload, sort_keys=True)
        _write_federation_json_atomic(artifact_staging, artifact_manifest_payload)
        _commit_federation_staged_replaces(
            federation_dir,
            (
                (manifest_staging, manifest_path),
                (mappings_staging, mappings_path),
                (composite_staging, composite_path),
                (artifact_staging, artifact_manifest_path),
            ),
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _federation_composite_hash_manifest_payload(stored: Mapping[str, Any], composite: SchemaGraph) -> dict[str, Any]:
    semantic_edges_hash = federation_composite_semantic_edges_hash(composite)
    composite.semantic_edges_hash = semantic_edges_hash
    updated = dict(stored)
    updated.update(
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
    return updated


def _commit_federation_composite_hash_pair(federation_dir: str, composite: SchemaGraph) -> None:
    paths = federation_artifact_paths(federation_dir)
    stored = load_federation_artifact_manifest_dict(paths["artifact_manifest"])
    if stored is None:
        return
    staging_dir = os.path.join(federation_dir, ".fed_quad_staging")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    try:
        composite_staging = os.path.join(staging_dir, os.path.basename(paths["composite_schema"]))
        artifact_staging = os.path.join(staging_dir, os.path.basename(paths["artifact_manifest"]))
        write_gzip_json_atomic(composite_staging, composite.to_dict(), sort_keys=True)
        _refresh_federation_artifact_manifest_hashes(federation_dir, composite)
        if not os.path.isfile(artifact_staging):
            payload = _federation_composite_hash_manifest_payload(stored, composite)
            _write_federation_json_atomic(artifact_staging, payload)
        _commit_federation_staged_replaces(
            federation_dir,
            (
                (composite_staging, paths["composite_schema"]),
                (artifact_staging, paths["artifact_manifest"]),
            ),
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _dropped_cross_source_join_matches(join: FederationCrossSourceJoin, drop_left: str, drop_right: str) -> bool:
    """Return True when a migration-map drop entry targets *join*."""
    drop_probe = FederationCrossSourceJoin(
        left=drop_left, right=drop_right, kind=join.kind, logical_key=join.logical_key
    )
    return cross_source_join_hash_entry(drop_probe) == cross_source_join_hash_entry(join)


def archive_federation_mappings_file(path: str) -> str:
    """Archive a federation mappings editor file to ``applied_federation_mappings.json``."""
    if Path(path).suffix.lower() != ".json":
        raise FederationConfigError(f"expected JSON editor file: {path!r}")
    directory = os.path.dirname(path) or "."
    archive = os.path.join(directory, FEDERATION_MAPPINGS_APPLIED_FILENAME)
    with open(path, encoding="utf-8") as src:
        content = src.read()
    write_text_atomic(archive, content)
    return archive


def archive_federation_editor_file(path: str) -> str:
    """Archive an editor JSON file to ``*.applied.json`` and return archive path."""
    if Path(path).suffix.lower() != ".json":
        raise FederationConfigError(f"expected JSON editor file: {path!r}")
    archive = path.replace(".json", STRUCTURE_APPLIED_SUFFIX)
    with open(path, encoding="utf-8") as src:
        content = src.read()
    write_text_atomic(archive, content)
    return archive


def export_federation_composite_overrides(
    composite: SchemaGraph,
    target: str | os.PathLike[str],
) -> Path:
    """Write composite schema overrides for review beside the federation composite graph."""
    return dump_structure_to_path(composite, Path(target))


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
    stored = load_federation_artifact_manifest_dict(paths["artifact_manifest"])
    if stored is None:
        return
    payload = _federation_composite_hash_manifest_payload(stored, composite)
    staging_dir = os.path.join(federation_dir, ".fed_quad_staging")
    staged_manifest = os.path.join(staging_dir, os.path.basename(paths["artifact_manifest"]))
    if os.path.isdir(staging_dir):
        _write_federation_json_atomic(staged_manifest, payload)
        return
    _write_federation_json_atomic(paths["artifact_manifest"], payload)


def _persist_federation_composite_schema_cache(federation_dir: str, composite: SchemaGraph) -> None:
    with artifact_lock(federation_dir):
        _commit_federation_composite_hash_pair(federation_dir, composite)


def apply_federation_composite_overrides(
    composite: SchemaGraph,
    federation_dir: str,
    overrides_path: str | os.PathLike[str],
    *,
    dialect: Any | None = None,
) -> StructureReport:
    """Apply an overrides editor file to the composite graph and persist replay state."""
    composite_path = federation_artifact_paths(federation_dir)["composite_schema"]
    document = load_structure_document_file(Path(overrides_path))
    sidecar = load_structure_sidecar(composite_path) or {}
    document = _merge_override_document_with_sidecar(document, sidecar)
    report = apply_structure_to_graph(composite, document, dialect=dialect, strict=True)
    document["foreign_keys_add"] = user_added_fks_dump(composite)
    document["primary_keys_add"] = user_added_pks_dump(composite)
    save_structure_sidecar(
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
    changed = finalize_with_structure(composite, composite_path, dialect=dialect)
    if changed:
        _persist_federation_composite_schema_cache(federation_dir, composite)
    return changed


def compute_federation_storage_dir(
    artifacts_root: str | None,
    federation_id: str,
) -> str:
    """Return the absolute federation artifact directory ``fed_<federation_id>``."""
    if artifacts_root and str(artifacts_root).strip():
        parent = os.path.abspath(os.path.expanduser(str(artifacts_root)))
    else:
        parent = user_data_dir(appname="aetherdialect", appauthor=False)
        _require_default_artifacts_root_writable(parent)
    safe_id = str(federation_id).strip()
    if not safe_id:
        raise FederationConfigError("federation_id must be non-empty")
    return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, f"{FEDERATION_STORAGE_PREFIX}{safe_id}")


def _federation_artifact_format_version_from_manifest(stored: Mapping[str, Any]) -> str | None:
    stored_fmt = coerce_format_version(stored.get("artifact_format_version"))
    return stored_fmt or None


def _raise_federation_artifact_format_version_mismatch(
    stored: Mapping[str, Any],
    manifest_path: str,
    federation_dir: str,
) -> None:
    stored_fmt = coerce_format_version(stored.get("artifact_format_version"))
    found_fmt = _federation_artifact_format_version_from_manifest(stored)
    if not format_versions_match(found_fmt, FEDERATION_ARTIFACT_FORMAT_VERSION):
        raise FederationConfigError(
            f"federation artifact manifest at {manifest_path!r} has "
            f"artifact_format_version {stored_fmt!r}; this build expects "
            f"{FEDERATION_ARTIFACT_FORMAT_VERSION}. Delete the federation artifact "
            f"directory {federation_dir!r} and re-run federation initialization "
            f"so the tree is rebuilt from scratch."
        )


def _federation_join_feedback_dir(federation_dir: str) -> str:
    return os.path.join(federation_dir, FEDERATION_TEMPLATES_SEGMENT, FEDERATION_JOIN_FEEDBACK_SEGMENT)


def _federation_join_feedback_shard_path(federation_dir: str, part: int) -> str:
    return os.path.join(
        _federation_join_feedback_dir(federation_dir),
        f"{FEDERATION_JOIN_FEEDBACK_PREFIX}{part:02x}.json.gz",
    )


def _federation_join_feedback_partition_number(q_norm: str) -> int:
    return int(hashlib.sha256(q_norm.encode("utf-8")).hexdigest()[:2], 16)


def _load_federation_join_feedback_shard(federation_dir: str, part: int) -> dict[str, list[str]]:
    path = _federation_join_feedback_shard_path(federation_dir, part)
    if not os.path.isfile(path):
        return {}
    try:
        raw = read_gzip_json(path)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {}
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for qk, rows in raw.items():
            if isinstance(rows, list):
                out[str(qk)] = [str(x) for x in rows if str(x).strip()]
    return out


def _write_federation_join_feedback_shard(federation_dir: str, part: int, payload: dict[str, list[str]]) -> None:
    path = _federation_join_feedback_shard_path(federation_dir, part)
    if not payload:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return
    os.makedirs(os.path.dirname(path), mode=ARTIFACT_DIR_MODE, exist_ok=True)
    write_gzip_json_atomic(path, payload, sort_keys=True)


def _lookup_federation_join_feedback_for_question(federation_dir: str, q_norm: str) -> list[str]:
    part = _federation_join_feedback_partition_number(q_norm)
    shard = _load_federation_join_feedback_shard(federation_dir, part)
    return list(shard.get(q_norm, []))


def _append_federation_join_feedback_for_question(federation_dir: str, q_norm: str, summary: str) -> None:
    text = str(summary or "").strip()
    if not text or not q_norm:
        return
    part = _federation_join_feedback_partition_number(q_norm)
    shard = _load_federation_join_feedback_shard(federation_dir, part)
    rows = list(shard.get(q_norm, []))
    if text in rows:
        return
    rows.append(text)
    shard[q_norm] = rows
    _write_federation_join_feedback_shard(federation_dir, part, shard)


def _migrate_federation_join_feedback_from_plans(federation_dir: str) -> dict[str, tuple[str, ...]]:
    path = federation_artifact_paths(federation_dir)["plan_templates"]
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    migrated_rows: dict[str, tuple[str, ...]] = {}
    cleaned: dict[str, Any] = {}
    for plan_id, row in loaded.items():
        if not isinstance(row, dict):
            continue
        join_fb = row.get("join_feedback")
        if isinstance(join_fb, list) and join_fb:
            question = str(row.get("question", "") or "")
            q_norm = normalize_question(question) if question else ""
            texts = tuple(str(item).strip() for item in join_fb if str(item).strip())
            if texts:
                migrated_rows[str(plan_id)] = texts
            if q_norm:
                for text in texts:
                    _append_federation_join_feedback_for_question(federation_dir, q_norm, text)
        cleaned_row = {k: v for k, v in row.items() if k != "join_feedback"}
        cleaned[str(plan_id)] = cleaned_row
    if migrated_rows:
        with artifact_lock(federation_dir):
            if cleaned:
                _write_federation_json_atomic(path, cleaned)
            elif os.path.isfile(path):
                os.remove(path)
    return migrated_rows


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
    if not federation_persist_quad_coherent(federation_dir):
        return False
    stored = load_federation_artifact_manifest_dict(manifest_path)
    if stored is None:
        return False
    _raise_federation_artifact_format_version_mismatch(stored, manifest_path, federation_dir)
    validate_federation_mappings_applied_sidecar(federation_dir, mappings)
    live_members = federation_member_hash_tuple(member_graphs, manifest)
    stored_members = stored.get("federation_members")
    if not isinstance(stored_members, list):
        return False
    active_ids = {binding.source_id for binding in manifest.sources}
    if not active_ids:
        active_ids = set(member_graphs.keys())
    for entry in stored_members:
        if isinstance(entry, (list, tuple)) and entry:
            normalize_stored_member_hash_row(entry)
    stored_member_ids = {str(entry[0]) for entry in stored_members if isinstance(entry, (list, tuple)) and entry}
    topology_shrink_only = bool(stored_member_ids - active_ids) and not (active_ids - stored_member_ids)
    stored_tuple = tuple(
        normalize_stored_member_hash_row(entry)
        for entry in stored_members
        if isinstance(entry, (list, tuple)) and entry and str(entry[0]) in active_ids
    )
    if stored_tuple != live_members:
        return False
    if not topology_shrink_only:
        stored_mappings_hash = str(stored.get("mappings_hash", "") or "")
        if stored_mappings_hash != mappings_hash(mappings):
            return False
        stored_manifest_hash = str(stored.get("manifest_hash", "") or "")
        if stored_manifest_hash != manifest_hash(manifest):
            return False
    return True


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
    os.makedirs(federation_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
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
        "created_with_package_version": artifact_package_version_string(),
        "min_compatible_package_version": MIN_COMPATIBLE_PACKAGE_VERSION,
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
        _commit_federation_persist_quad(
            federation_dir,
            manifest_path=paths["manifest"],
            manifest_payload=manifest_payload,
            mappings_path=paths["mappings"],
            mappings_payload=mappings_payload,
            composite_path=paths["composite_schema"],
            composite_payload=composite.to_dict(),
            artifact_manifest_path=paths["artifact_manifest"],
            artifact_manifest_payload=artifact_payload,
        )
        write_federation_mappings_applied_sidecar(federation_dir, mappings)
        fed_id = str(manifest.federation_id or "").strip()
        for binding in manifest.sources:
            member_dir = federation_source_artifacts_dir(
                os.path.dirname(os.path.dirname(federation_dir)),
                binding,
                federation_id=fed_id or None,
            )
            if os.path.isdir(member_dir):
                write_federation_member_manifest(member_dir, binding, federation_id=fed_id)
    if manifest_editor_path and os.path.isfile(manifest_editor_path):
        archive_federation_editor_file(manifest_editor_path)
    if mappings_editor_path and os.path.isfile(mappings_editor_path):
        archive_federation_mappings_file(mappings_editor_path)
    os.makedirs(os.path.dirname(paths["plan_templates"]), mode=ARTIFACT_DIR_MODE, exist_ok=True)


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
    if not federation_persist_quad_coherent(federation_dir):
        return None
    stored_manifest = load_federation_artifact_manifest_dict(manifest_path)
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
    candidate = engine or "duckdb"
    if candidate in DialectRegistry.list_engines():
        return candidate
    return "duckdb"


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
        role=SchemaRole.OWNER,
    )


def load_persisted_federation_roster_rows(federation_dir: str) -> tuple[tuple[str, str, str, str], ...]:
    """Load pinned roster rows from a federation artifact manifest."""
    paths = federation_artifact_paths(federation_dir)
    if not federation_persist_quad_coherent(federation_dir):
        raise FederationConfigError(f"federation artifact tree at {federation_dir!r} is incomplete or torn")
    stored = load_federation_artifact_manifest_dict(paths["artifact_manifest"])
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
    *,
    schema_role: SchemaRole = SchemaRole.OWNER,
) -> PersistedFederationInspection:
    """Load declaration and roster from a persisted ``fed_<id>`` tree. Does not construct member engines or open database connections."""
    if schema_role == SchemaRole.CONSUMER:
        raise PermissionError("inspect_persisted requires owner role")
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
    namespace = namespace_from_composite_schema(composite) if composite is not None else {}
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
    where_leaves = PredicateGroup.where_leaves(residual.where)
    having_leaves = PredicateGroup.having_leaves(residual.having)
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
    temporal_bind: AnchoredTemporalBind | None = None,
) -> tuple[tuple[str, str], ...]:
    """Ordered per-source sub-intent fingerprints for template matching."""
    out: list[tuple[str, str]] = []
    anchor_fp = temporal_bind.anchor_iso if temporal_bind is not None else ""
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
                    f"{schema_graph_id}:{join_fp}:{anchor_fp}"
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


def delete_federation_plan_template(
    federation_dir: str, plan_id: str, *, schema_role: SchemaRole = SchemaRole.OWNER
) -> None:
    """Remove one federation plan template record when it has no accepted questions."""
    if schema_role != SchemaRole.OWNER:
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
    federation_dir: str, plan_id: str, *, schema_role: SchemaRole = SchemaRole.OWNER
) -> None:
    """Drop a plan record that was never credited with an accepted question."""
    if not federation_dir or not plan_id:
        return
    template = load_federation_plan_templates(federation_dir).get(str(plan_id))
    if template is None or template.accepted_questions:
        return
    delete_federation_plan_template(federation_dir, str(plan_id), schema_role=schema_role)


def save_federation_plan_template(
    federation_dir: str, template: FederationPlanTemplate, *, schema_role: SchemaRole = SchemaRole.OWNER
) -> None:
    """Append or replace a federation plan template in the federation tree."""
    if schema_role != SchemaRole.OWNER:
        raise OwnerOnlyOperationError("save_federation_plan_template")
    paths = federation_artifact_paths(federation_dir)
    os.makedirs(os.path.dirname(paths["plan_templates"]), mode=ARTIFACT_DIR_MODE, exist_ok=True)
    bounded_accepted = _bound_federation_plan_accepted_questions(template.accepted_questions)
    row: dict[str, Any] = {
        "format_version": coerce_format_version(template.format_version),
        "composite_schema_graph_id": template.composite_schema_graph_id,
        "intent_key": template.intent_key,
        "step_fingerprints": [list(part) for part in template.step_fingerprints],
        "combine_hash": template.combine_hash,
        "question": template.question,
        "accepted_questions": list(bounded_accepted),
        "member_template_ids": [list(part) for part in template.member_template_ids],
        "residual_hash": template.residual_hash,
        "manifest_hash": template.manifest_hash,
        "member_tuple_hash": template.member_tuple_hash,
    }
    if template.join_feedback:
        row["join_feedback"] = list(template.join_feedback)
    with artifact_lock(federation_dir):
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
        existing[template.plan_id] = row
        existing = _enforce_federation_plan_template_file_cap(existing, keep_plan_id=template.plan_id)
        _write_federation_json_atomic(paths["plan_templates"], existing)


def load_federation_plan_templates(federation_dir: str) -> dict[str, FederationPlanTemplate]:
    """Load federation plan templates keyed by plan id."""
    migrated_join_feedback = _migrate_federation_join_feedback_from_plans(federation_dir)
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
        try:
            row_fmt = coerce_format_version(row.get("format_version", FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION))
        except (TypeError, ValueError):
            continue
        if not format_versions_match(row_fmt, FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION):
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
        question_str = str(row.get("question", "") or "")
        join_feedback = migrated_join_feedback.get(str(plan_id), ())
        out[str(plan_id)] = FederationPlanTemplate(
            plan_id=str(plan_id),
            composite_schema_graph_id=str(row.get("composite_schema_graph_id", "") or ""),
            intent_key=str(row.get("intent_key", "") or ""),
            step_fingerprints=tuple(steps),
            combine_hash=str(row.get("combine_hash", "") or ""),
            question=question_str,
            accepted_questions=accepted,
            format_version=row_fmt,
            member_template_ids=tuple(member_ids),
            residual_hash=str(row.get("residual_hash", "") or ""),
            join_feedback=join_feedback,
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
    schema_role: SchemaRole = SchemaRole.OWNER,
    pending_plan_template: FederationPlanTemplate | None = None,
) -> None:
    """Record that *q_norm* accepted the federation plan *plan_id*."""
    if schema_role != SchemaRole.OWNER:
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


def _federation_plan_question_norm(template: FederationPlanTemplate) -> str:
    """Resolve the normalised question key used for plan-scoped join feedback."""
    q_norm = normalize_question(template.question) if template.question else ""
    if q_norm:
        return q_norm
    for raw in template.accepted_questions:
        q_norm = normalize_question(raw)
        if q_norm:
            return q_norm
    return ""


def mirror_federation_plan_join_feedback(
    federation_dir: str,
    plan_id: str,
    summary: str,
) -> None:
    """Mirror join feedback onto the federation plan template record."""
    text = str(summary or "").strip()
    if not federation_dir or not plan_id or not text:
        return
    template = load_federation_plan_templates(federation_dir).get(plan_id)
    if template is None:
        return
    existing = tuple(template.join_feedback)
    if text in existing:
        return
    save_federation_plan_template(
        federation_dir,
        replace(template, join_feedback=existing + (text,)),
    )


def record_federation_join_feedback(
    federation_dir: str,
    plan_id: str,
    summary: str,
    *,
    q_norm: str | None = None,
) -> None:
    """Persist cross-source join rejection feedback for the plan's question."""
    text = str(summary or "").strip()
    if not federation_dir or not plan_id or not text:
        return
    template = load_federation_plan_templates(federation_dir).get(plan_id)
    if template is None:
        return
    resolved_q = normalize_question(q_norm or "") if q_norm else ""
    if not resolved_q:
        resolved_q = _federation_plan_question_norm(template)
    if not resolved_q:
        return
    with artifact_lock(federation_dir):
        if not template.question:
            save_federation_plan_template(
                federation_dir,
                replace(template, question=resolved_q),
            )
        _append_federation_join_feedback_for_question(federation_dir, resolved_q, text)


def lookup_federation_join_feedback(federation_dir: str, plan_id: str) -> list[str]:
    """Return cross-source join feedback stored for the federation plan's question."""
    if not federation_dir or not plan_id:
        return []
    template = load_federation_plan_templates(federation_dir).get(plan_id)
    if template is None:
        return []
    q_norm = _federation_plan_question_norm(template)
    if not q_norm:
        return []
    return _lookup_federation_join_feedback_for_question(federation_dir, q_norm)


def federation_source_artifacts_dir(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
) -> str:
    """Return the artifact directory for one federation member source."""
    parent = _federation_member_artifacts_root(artifacts_root)
    return os.path.join(parent, federation_source_storage_slug(binding, federation_id=federation_id))


def _federation_member_manifest_path(member_dir: str) -> str:
    return os.path.join(member_dir, FEDERATION_MEMBER_MANIFEST_FILENAME)


def write_federation_member_manifest(
    member_dir: str,
    binding: FederationSourceBinding,
    *,
    federation_id: str,
) -> None:
    """Persist engine identity for a federation member artifact tree."""
    os.makedirs(member_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    payload = {
        "federation_id": str(federation_id or "").strip(),
        "source_id": str(binding.source_id or "").strip(),
        "engine": str(binding.engine or "").strip().lower(),
        "connection": str(binding.connection or "").strip(),
    }
    _write_federation_json_atomic(_federation_member_manifest_path(member_dir), payload)


def load_federation_member_manifest(member_dir: str) -> dict[str, Any] | None:
    """Load a federation member manifest when present."""
    path = _federation_member_manifest_path(member_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FederationConfigError(f"federation member manifest at {path!r} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FederationConfigError(f"federation member manifest at {path!r} is not a JSON object")
    return payload


def detect_federation_member_engine_drift(
    binding: FederationSourceBinding,
    member_dir: str,
    *,
    federation_id: str | None = None,
) -> bool:
    """Return True when the live binding disagrees with the stored member manifest."""
    stored = load_federation_member_manifest(member_dir)
    if stored is None:
        return False
    fed_id = str(federation_id or "").strip()
    if fed_id and str(stored.get("federation_id", "") or "").strip() not in ("", fed_id):
        return True
    if str(stored.get("source_id", "") or "").strip() != str(binding.source_id or "").strip():
        return True
    stored_engine = str(stored.get("engine", "") or "").strip().lower()
    live_engine = str(binding.engine or "").strip().lower()
    if stored_engine and live_engine and stored_engine != live_engine:
        return True
    stored_connection = str(stored.get("connection", "") or "").strip()
    live_connection = str(binding.connection or "").strip()
    if stored_connection and live_connection and stored_connection != live_connection:
        return True
    return False


def _federation_member_schema_graph_path(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
) -> str:
    return os.path.join(
        federation_source_artifacts_dir(artifacts_root, binding, federation_id=federation_id),
        "schema_graph.json.gz",
    )


def _load_federation_member_schema_graph(
    artifacts_root: str | None,
    binding: FederationSourceBinding,
    *,
    federation_id: str | None = None,
) -> SchemaGraph:
    """Load one stored member schema graph, surfacing unreadable or unprofiled artifacts."""
    source_id = str(binding.source_id or "").strip()
    path = _federation_member_schema_graph_path(artifacts_root, binding, federation_id=federation_id)
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
    fed_id = str(manifest.federation_id or "").strip() or None
    for binding in manifest.sources:
        path = _federation_member_schema_graph_path(artifacts_root, binding, federation_id=fed_id)
        if not os.path.isfile(path):
            continue
        member_dir = federation_source_artifacts_dir(artifacts_root, binding, federation_id=fed_id)
        with artifact_lock(member_dir):
            graphs[binding.source_id] = _load_federation_member_schema_graph(
                artifacts_root,
                binding,
                federation_id=fed_id,
            )
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
            merged[source_id] = stamp_member_graph_source_id(disk, source_id)
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


def _value_overlap_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    s1 = {str(v) for v in left if str(v)}
    s2 = {str(v) for v in right if str(v)}
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    return inter / float(min(len(s1), len(s2)))
