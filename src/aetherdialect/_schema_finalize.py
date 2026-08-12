"""Structure documents, migration, persistence, and profiling SQL helpers."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ._config import (
    ConfigError,
    EngineConfig,
    PolicyConfig,
)
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
    DIAGNOSTIC_CODE_STRUCTURE_NEEDS_RECONFIRMATION,
    PUBLIC_STRUCTURE_DOCUMENT_KEYS,
    ROLE_VALUE_TYPE_COMPAT,
    SCHEMA_JOIN_PATH_ENUMERATION_VERSION,
    STRUCTURE_COLUMN_EDITABLE_KEYS,
    STRUCTURE_DOCUMENT_VERSION,
    STRUCTURE_EXPORT_DEFAULT_OWNER,
    STRUCTURE_MAX_DESCRIPTION_CHARS,
    STRUCTURE_PROSE_KEYS,
    STRUCTURE_PROSE_REDIRECT_HINT,
    STRUCTURE_TABLE_EDIT_KEYS,
    STRUCTURE_TOP_LEVEL_EDIT_KEYS,
    VALID_FK_ADD_KEYS,
    VALID_FK_KINDS,
    VALID_FK_REMOVE_KEYS,
    VALID_PK_ADD_KEYS,
    VALID_PK_REMOVE_KEYS,
    VALID_SENSITIVITY_LEVELS,
)
from ._constants_runtime import (
    DESCRIPTION_REFINER_SYSTEM,
    DOMAIN_KNOWLEDGE_REFINER_SYSTEM,
    SCHEMA_BUILD_PHASE_A,
    SCHEMA_BUILD_PHASE_B,
    SCHEMA_BUILD_PHASE_C,
    SCHEMA_BUILD_PHASE_D,
    SCHEMA_BUILD_PHASE_E,
    SCHEMA_BUILD_PHASE_F,
    SCHEMA_BUILD_PHASE_G,
    SCHEMA_BUILD_PHASE_H,
    SCHEMA_BUILD_PHASE_I,
    SCHEMA_BUILD_PHASE_J,
    SCHEMA_BUILD_PHASE_K,
    STRUCTURE_EDITABLE_ENUMS,
)
from ._contracts_base import (
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    EngineContext,
    MigrationTier,
    OverrideSkip,
    SchemaAccessError,
    SensitivityClassification,
    SidecarReconcileReport,
    StructureReport,
    TableKind,
)
from ._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    DescriptionOwner,
    FederationManifest,
    FKEdge,
    InferenceTag,
    PkInferenceTag,
    RoleOwner,
    SchemaGraph,
    TableMetadata,
    TableRole,
)
from ._dialect import Dialect
from ._knowledge_staleness import resolve_structural_knowledge_for_schema
from ._llm_provider import LLMProvider
from ._schema_graph import (
    SchemaDiff,
    apply_deny_objects_filter,
    apply_fk_remaps_to_graph,
    apply_pk_remaps_to_graph,
    apply_schema_context_allow_columns,
    assign_schema_graph_hashes,
    catalog_fk_graph_is_connected,
    classify_scope_change,
    coerce_pk_fk_columns_to_identifier,
    collapse_redundant_inferences,
    compute_dialect_probe,
    diff_schemas,
    edge_key,
    filter_schema_graph_by_scope,
    infer_missing_pks_from_profile,
    load_schema_graph_snapshot,
    mark_canonical_duplicates,
    notes_content_sha256,
    notify_schema_path_health,
    raise_if_schema_unusable,
    recompute_join_paths_multi,
    redact_hidden_sensitivity_profile_values,
    refuse_incompatible_catalog_foreign_keys,
    resolve_column_renames,
    resolve_table_renames,
    run_fk_inference_if_disconnected,
    schema_context_from_descriptor,
    schema_context_from_graph,
    semantic_edges_fingerprint,
    strip_schema_context_denied_columns,
    table_from_dict,
    table_structural_hash_fp,
    table_to_dict,
    tables_profiling_payload,
    tables_row_count_fingerprint,
    tables_structural_payload,
    validate_scope_against_graph,
)
from ._schema_profile import (
    apply_boolean_coercion_pass,
    apply_column_roles_llm,
    assign_column_ops,
    emit_description_enrichment_failed,
    emit_description_enrichment_noop,
    emit_schema_fk_catalog_absent_warning,
    emit_schema_unknown_type_unusable_warnings,
    extract_knowledge_from_notes,
    filter_schema_anchored_domain_knowledge,
    infer_view_same_name_key_edges,
    llm_classify_schema,
    on_sensitivity_classification_change,
    replay_user_semantic_neighbors_to_columns,
    rerun_column_classifier,
    sensitivity_increased_columns,
    snapshot_column_sensitivities,
)
from ._schema_reflect import (
    apply_view_scope_postprocess,
    debug_clip_stable_json,
    ensure_semantic_join_neighbors,
    first_table_where_stable_json_differs,
    load_inference_block_lists,
    reflect_schema_graph_for_context,
    save_schema_to_cache,
    split_fk_endpoint,
    structure_sidecar_path,
    tables_payload_through_model_round_trip,
)
from ._utils import (
    coerce_format_version,
    data_type_to_value_type,
    debug,
    effective_structural_hash_fp,
    emit_construction_phase,
    format_versions_match,
    llm_usage_build_scope,
    notify,
    profiling_hash_fp,
    require_exact_keys,
    schema_hash_fp,
    scope_hash_fp,
    stable_json,
    structural_hash_fp,
)
from ._utils_artifacts import (
    artifact_lock,
    read_gzip_json,
    wipe_versioned_artifacts,
    write_artifact_manifest,
    write_json_atomic,
)


def _add_profiling_data(
    dialect: Dialect,
    sg: SchemaGraph,
    notes_content: str | None = None,
    *,
    schema_json_path: str | Path | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """Add column profiling data to a SchemaGraph in-place."""
    pk_blocked, fk_blocked = load_inference_block_lists(schema_json_path, schema=sg)
    emit_construction_phase(SCHEMA_BUILD_PHASE_E)
    debug(f"[{SCHEMA_BUILD_PHASE_E}] profiling columns (statistics)")

    dialect.profile_schema(sg)
    emit_schema_unknown_type_unusable_warnings(sg)

    debug(f"[{SCHEMA_BUILD_PHASE_E}] inferring missing primary keys from profile")
    infer_missing_pks_from_profile(sg.tables, dialect=dialect, blocked=pk_blocked)

    if dialect.name == "bigquery":
        catalog_fk_count = sum(
            1 for table in sg.tables.values() for fk in table.foreign_keys if fk.inference_tag is None
        )
        if catalog_fk_count == 0 and len(sg.tables) >= 2:
            emit_schema_fk_catalog_absent_warning(dialect.name)

    debug(f"[{SCHEMA_BUILD_PHASE_E}] FK inference when catalog graph is disconnected")
    added_fks = run_fk_inference_if_disconnected(sg, blocked=fk_blocked)
    if added_fks:
        debug(f"[{SCHEMA_BUILD_PHASE_E}] added {added_fks} inferred FK edge(s)")

    emit_construction_phase(SCHEMA_BUILD_PHASE_F)
    debug(f"[{SCHEMA_BUILD_PHASE_F}] inferring column and table roles via LLM")
    sensitivity_before = snapshot_column_sensitivities(sg)
    adir = os.path.dirname(os.path.abspath(str(schema_json_path))) if schema_json_path else None
    apply_column_roles_llm(sg, notes_content=notes_content, log_sink=log_sink, artifacts_dir=adir)
    increased = sensitivity_increased_columns(sensitivity_before, sg)
    if increased and schema_json_path:
        on_sensitivity_classification_change(sg, increased, artifacts_dir=adir)

    emit_construction_phase(SCHEMA_BUILD_PHASE_G)
    debug(f"[{SCHEMA_BUILD_PHASE_G}] boolean coercion pass")
    apply_boolean_coercion_pass(sg)

    debug(f"[{SCHEMA_BUILD_PHASE_G}] redacting profile values for hidden-sensitivity columns")
    redact_hidden_sensitivity_profile_values(sg)

    unusable_count = sum(
        1
        for tbl in sg.tables.values()
        for col in tbl.columns.values()
        if not col.is_usable and not col.is_primary_key and not col.is_foreign_key
    )
    sensitive_count = sum(
        1
        for tbl in sg.tables.values()
        for col in tbl.columns.values()
        if col.sensitivity != SensitivityClassification.NONE
    )
    if unusable_count:
        notify(
            f"Schema build: {unusable_count} column(s) marked unusable (excluded from classification and LLM-facing schema).",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
    if sensitive_count:
        notify(
            f"Schema build: {sensitive_count} column(s) marked sensitive (restricted or hidden).",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )

    debug(f"[{SCHEMA_BUILD_PHASE_G}] assigning column operations (deterministic)")
    assign_column_ops(sg)

    debug(f"[{SCHEMA_BUILD_PHASE_G}] coercing PK and FK columns to identifier role")
    coerced = coerce_pk_fk_columns_to_identifier(sg)
    if coerced:
        debug(f"[{SCHEMA_BUILD_PHASE_G}] coerced {len(coerced)} PK/FK columns to identifier role")

    debug(f"[{SCHEMA_BUILD_PHASE_K}] marking canonical bearer per duplicated column name")
    demoted = mark_canonical_duplicates(sg)
    if demoted:
        debug(f"[{SCHEMA_BUILD_PHASE_K}] demoted {demoted} non-canonical duplicate columns")


def _debug_schema_cache_hash_mismatch(
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


def _invalidate_corrupt_schema_cache(
    schema_json_path: str,
    exc: BaseException,
    *,
    stage: str = "deserialize",
) -> None:
    debug(f"[schema._build_schema_graph] corrupt cache {stage}: {exc!r}")
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


def _schema_graph_from_cache_dict(d: dict[str, Any], schema_json_path: str) -> SchemaGraph | None:
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
        _invalidate_corrupt_schema_cache(schema_json_path, exc)
        return None


def _refresh_join_paths_multi_if_enumeration_stale(
    sg: SchemaGraph,
    cache_data: dict[str, Any],
    *,
    sink: Callable[[str], None] | None = None,
) -> bool:
    cached_ver = coerce_format_version(cache_data.get("join_path_enumeration_version", "0") or "0")
    if format_versions_match(cached_ver, SCHEMA_JOIN_PATH_ENUMERATION_VERSION):
        return False
    if sink is not None:
        sink("  Schema: join-path enumeration policy changed — recomputing join paths...")
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    sg.refresh_schema_stats()
    return True


def _role_owner_export_map() -> dict[RoleOwner, str]:
    return {
        RoleOwner.CATALOG: "catalog",
        RoleOwner.PROFILE: "profile",
        RoleOwner.LLM: STRUCTURE_EXPORT_DEFAULT_OWNER,
        RoleOwner.BOOLEAN_COERCION: "boolean_coercion",
        RoleOwner.USER_OVERRIDE: "user",
        RoleOwner.PK_FK_COERCION: "pk_fk_coercion",
    }


def _role_owner_import_map() -> dict[str, RoleOwner]:
    return {token: owner for owner, token in _role_owner_export_map().items()}


def _role_owner_export_token(owner: RoleOwner | None) -> str:
    resolved = owner if owner is not None else RoleOwner.CATALOG
    return _role_owner_export_map()[resolved]


def _parse_export_owner_token(raw: Any, path: str, allowed: dict[str, Any]) -> Any:
    export_owner = str(raw or "").strip()
    if export_owner not in allowed:
        raise ValueError(f"{path}: owner is engine-managed and not user-editable")
    return allowed[export_owner]


def _override_json_null_sentinel(raw: Any) -> bool:
    """Return True when *raw* is JSON null or an export envelope whose ``value`` is null."""
    if raw is None:
        return True
    if isinstance(raw, dict) and "value" in raw and raw.get("value") is None:
        if set(raw.keys()) <= {"value", "owner"}:
            return True
    return False


def _parse_editable_role_json(
    raw: Any,
    path: str,
    allowed_values: set[str],
    *,
    value_type: str | None = None,
) -> tuple[str | None, RoleOwner]:
    """Parse editable ``role`` JSON (bare token, null, or ``{"value": ...}`` with optional export ``owner`` token)."""
    if isinstance(raw, dict):
        if "owner" in raw:
            want_owner = _parse_export_owner_token(raw.get("owner"), f"{path}.owner", _role_owner_import_map())
        else:
            want_owner = RoleOwner.USER_OVERRIDE
        extra = set(raw.keys()) - {"value", "owner"}
        if extra:
            raise ValueError(f"{path}: unsupported keys {sorted(extra)!r}")
        if "value" not in raw:
            raise ValueError(f"{path}: object must contain key 'value'")
        val = raw["value"]
        if val is not None and val not in allowed_values:
            raise ValueError(f"{path}: {val!r} not in {sorted(allowed_values)!r}")
        if val is not None and value_type:
            vt = value_type.strip().lower()
            if vt:
                rv = str(val).strip().lower()
                allowed_vt = ROLE_VALUE_TYPE_COMPAT.get(rv)
                if allowed_vt is not None and vt not in allowed_vt:
                    raise ValueError(f"{path}: role {val!r} is incompatible with column value_type {value_type!r}")
        return val, want_owner
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


def _role_envelope(value: str | None, owner: RoleOwner | None) -> dict[str, str]:
    """Wrap a role token for schema-overrides export with provenance."""
    return {"value": "" if value is None else str(value), "owner": _role_owner_export_token(owner)}


def _column_override_value_dict(col: ColumnMetadata) -> dict[str, Any]:
    """Return the structural editable subset of a ``ColumnMetadata`` for an overrides dump."""
    vt = (col.value_type or "").strip().lower() or (
        data_type_to_value_type(col.data_type).lower() if col.data_type else ""
    )
    out: dict[str, Any] = {
        "role": _role_envelope(col.role, col.role_owner),
        "sensitivity": col.sensitivity.value,
    }
    if vt == "boolean":
        out["boolean_truth_value"] = col.boolean_truth_value
    if col.usable_override is True:
        out["usable"] = True
    return out


def _table_override_value_dict(table: TableMetadata) -> dict[str, Any]:
    """Return the structural editable subset of a ``TableMetadata`` for an overrides dump."""
    columns_out: dict[str, Any] = {}
    for cname, col in table.columns.items():
        if _column_has_exportable_override(col):
            columns_out[cname] = _column_override_value_dict(col)
    out: dict[str, Any] = {}
    if table.role_owner == RoleOwner.USER_OVERRIDE:
        out["role"] = _role_envelope(table.role, table.role_owner)
    if columns_out:
        out["columns"] = columns_out
    return out


def _column_has_exportable_override(col: ColumnMetadata) -> bool:
    """Return True when *col* carries a user-edited structural override worth exporting."""
    if col.role_owner == RoleOwner.USER_OVERRIDE:
        return True
    if col.usable_override is True:
        return True
    if col.sensitivity != SensitivityClassification.NONE:
        return True
    return False


def _column_structural_hash_fp(table: TableMetadata, column_name: str) -> str:
    """Return the structural fingerprint for one column within *table*."""
    col = table.columns[column_name]
    payload = {
        table.name: {
            "columns": {
                column_name: {
                    "data_type": col.data_type,
                    "is_nullable": col.is_nullable,
                    "is_primary_key": col.is_primary_key,
                    "is_foreign_key": col.is_foreign_key,
                }
            },
            "foreign_keys": [],
            "indexed_columns": [],
            "kind": table.kind,
            "primary_key": list(table.primary_key),
            "view_definition": "",
        }
    }
    return structural_hash_fp(payload)


def _override_object_recreated(
    object_name: str,
    *,
    previous_schema: SchemaGraph | None,
    current_schema: SchemaGraph,
    authored_hash: str,
    current_hash: str,
) -> bool:
    """Return True when *object_name* was absent before and is present now with a new hash."""
    if not authored_hash or authored_hash == current_hash:
        return False
    if previous_schema is None:
        return False
    if "." in object_name:
        table_name, column_name = object_name.split(".", 1)
        if table_name not in current_schema.tables:
            return False
        prev_tbl = previous_schema.tables.get(table_name)
        if prev_tbl is None:
            return True
        return column_name not in prev_tbl.columns
    return object_name not in previous_schema.tables


def _notify_override_needs_reconfirmation(path: str, object_name: str) -> None:
    notify(
        f"Schema override for {object_name} requires re-confirmation after object recreation",
        stage="schema",
        code=DIAGNOSTIC_CODE_STRUCTURE_NEEDS_RECONFIRMATION,
        details=(("path", path), ("object", object_name)),
    )


def _override_entry_blocked_by_recreation(
    path: str,
    object_name: str,
    entry: dict[str, Any],
    *,
    current_hash: str,
    previous_schema: SchemaGraph | None,
    current_schema: SchemaGraph,
    skipped: list[OverrideSkip],
) -> bool:
    """Skip replay when a recreated object still carries stale override provenance."""
    authored_hash = str(entry.get("authored_against_structural_hash", "") or "")
    if entry.get("needs_reconfirmation"):
        skipped.append(OverrideSkip(path=path, reason="needs_reconfirmation", code="needs_reconfirmation"))
        _notify_override_needs_reconfirmation(path, object_name)
        return True
    if not _override_object_recreated(
        object_name,
        previous_schema=previous_schema,
        current_schema=current_schema,
        authored_hash=authored_hash,
        current_hash=current_hash,
    ):
        return False
    entry["needs_reconfirmation"] = True
    skipped.append(OverrideSkip(path=path, reason="needs_reconfirmation", code="needs_reconfirmation"))
    _notify_override_needs_reconfirmation(path, object_name)
    return True


def _stamp_override_entry_provenance(entry: dict[str, Any], structural_hash: str) -> None:
    """Record what structural state an override entry was authored against."""
    entry["authored_against_structural_hash"] = structural_hash
    entry["authored_at"] = datetime.now(UTC).isoformat()
    entry.pop("needs_reconfirmation", None)


def _stamp_sidecar_provenance(document: dict[str, Any], sg: SchemaGraph) -> None:
    """Stamp provenance on every persisted override entry in *document*."""
    tables = document.get("tables")
    if isinstance(tables, dict):
        for tname, tval in tables.items():
            if not isinstance(tval, dict) or tname not in sg.tables:
                continue
            if tval.get("needs_reconfirmation"):
                continue
            _stamp_override_entry_provenance(tval, table_structural_hash_fp(sg.tables[tname]))
            cols = tval.get("columns")
            if isinstance(cols, dict):
                for cname, cval in cols.items():
                    if not isinstance(cval, dict) or cname not in sg.tables[tname].columns:
                        continue
                    if cval.get("needs_reconfirmation"):
                        continue
                    _stamp_override_entry_provenance(
                        cval,
                        _column_structural_hash_fp(sg.tables[tname], cname),
                    )
    for block_key in ("foreign_keys_add", "primary_keys_add"):
        entries = document.get(block_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("needs_reconfirmation"):
                continue
            _stamp_override_entry_provenance(entry, sg.effective_structural_hash)


def _resolve_sidecar_entry(document: dict[str, Any], object_path: str) -> dict[str, Any] | None:
    """Return the override dict at *object_path* like ``tables.orders`` or ``tables.orders.columns.status``."""
    parts = object_path.split(".")
    if len(parts) < 2 or parts[0] != "tables":
        return None
    tables = document.get("tables")
    if not isinstance(tables, dict):
        return None
    entry = tables.get(parts[1])
    if not isinstance(entry, dict):
        return None
    if len(parts) == 2:
        return entry
    if len(parts) == 4 and parts[2] == "columns":
        cols = entry.get("columns")
        if not isinstance(cols, dict):
            return None
        col_entry = cols.get(parts[3])
        return col_entry if isinstance(col_entry, dict) else None
    return None


def _structural_hash_for_override_path(sg: SchemaGraph, object_path: str) -> str | None:
    """Return the current structural hash for an override object path on *sg*."""
    parts = object_path.split(".")
    if len(parts) == 2 and parts[0] == "tables":
        tbl = sg.tables.get(parts[1])
        return table_structural_hash_fp(tbl) if tbl is not None else None
    if len(parts) == 4 and parts[0] == "tables" and parts[2] == "columns":
        tbl = sg.tables.get(parts[1])
        if tbl is None or parts[3] not in tbl.columns:
            return None
        return _column_structural_hash_fp(tbl, parts[3])
    return None


def reconfirm_override(schema_json_path: str | Path, object_path: str, sg: SchemaGraph) -> bool:
    """Stamp the current structural hash onto a sidecar override entry so replay can apply it again."""
    sidecar = load_structure_sidecar(schema_json_path)
    if sidecar is None:
        return False
    entry = _resolve_sidecar_entry(sidecar, object_path)
    if entry is None:
        return False
    current_hash = _structural_hash_for_override_path(sg, object_path)
    if not current_hash:
        return False
    _stamp_override_entry_provenance(entry, current_hash)
    save_structure_sidecar(
        schema_json_path,
        sidecar,
        source_schema_hash=str(sidecar.get("source_schema_hash", "") or sg.effective_structural_hash),
        metadata_hash=str(sidecar.get("metadata_hash", "") or compute_metadata_hash(sg)),
    )
    return True


def _fk_block_to_remove_entries(block: list[Any]) -> list[dict[str, Any]]:
    """Convert persisted FK block-list entries into editable ``foreign_keys_remove`` records."""
    out: list[dict[str, Any]] = []
    for entry in block:
        if isinstance(entry, dict) and entry.get("from") and entry.get("to"):
            out.append({"from": entry["from"], "to": entry["to"]})
    return out


def _pk_block_to_remove_entries(block: list[Any]) -> list[dict[str, Any]]:
    """Convert persisted PK block-list entries into editable ``primary_keys_remove`` records."""
    out: list[dict[str, Any]] = []
    for entry in block:
        if isinstance(entry, dict) and entry.get("table") and entry.get("column"):
            out.append({"table": entry["table"], "column": entry["column"]})
    return out


def _catalog_fk_revoked_pairs(internal: dict[str, Any]) -> set[tuple[str, str]]:
    """Normalize revoked catalog FK endpoint pairs from the sidecar ``_internal`` envelope."""
    pairs: set[tuple[str, str]] = set()
    for entry in internal.get("catalog_fk_revoked") or []:
        if not isinstance(entry, dict):
            continue
        frm = entry.get("from")
        to = entry.get("to")
        if frm is None or to is None:
            continue
        pairs.add((str(frm), str(to)))
    return pairs


def _append_catalog_fk_revoked(
    schema_json_path: str | Path,
    edges: list[FKEdge],
    *,
    sg: SchemaGraph | None = None,
) -> None:
    """Record catalog FK edges removed upstream so replay cannot resurrect them."""
    if not edges:
        return
    sidecar = load_structure_sidecar(schema_json_path) or {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "primary_keys_add": [],
        "_internal": {},
    }
    internal = sidecar.setdefault("_internal", {})
    if not isinstance(internal, dict):
        internal = {}
        sidecar["_internal"] = internal
    revoked = list(internal.get("catalog_fk_revoked") or [])
    seen = _catalog_fk_revoked_pairs(internal)
    for edge in edges:
        entry = {
            "from": _fk_endpoint_string(edge.src_table, list(edge.src_cols)),
            "to": _fk_endpoint_string(edge.dst_table, list(edge.dst_cols)),
        }
        key = (str(entry["from"]), str(entry["to"]))
        if key in seen:
            continue
        revoked.append(entry)
        seen.add(key)
    internal["catalog_fk_revoked"] = revoked
    src_hash = str(sidecar.get("source_schema_hash") or "")
    meta_hash = compute_metadata_hash(sg) if sg is not None else str(sidecar.get("metadata_hash") or "")
    save_structure_sidecar(schema_json_path, sidecar, source_schema_hash=src_hash, metadata_hash=meta_hash)


def _fk_endpoint_string(table: str, cols: list[str]) -> Any:
    """Render an FK endpoint as the dotted shorthand for single-column edges or the list form for composites."""
    if len(cols) == 1:
        return f"{table}.{cols[0]}"
    return [f"{table}.{c}" for c in cols]


def _foreign_keys_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Snapshot every FK currently bound to the graph as ``{from, to, inference_tag, removable, declared}`` records. The ``inference_tag`` field exposes which inference layer produced each edge: ``None`` denotes a catalog FK declared by the database itself, ``"suffix"``/``"self"``/``"composite"`` denote suffix-name inference variants, ``"semantic"`` denotes a value-overlap promotion, and ``"user_override_*"`` denotes an FK added through ``foreign_keys_add``. The ``removable`` boolean tells the editor whether the edge can be cited under ``foreign_keys_remove`` (true for inferred and user-override edges; false for catalog edges, which the database itself declares). ``declared`` is true exactly for catalog edges (``inference_tag is None``). Catalog edges remain visible so editors can see them without ambiguity."""
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
    """Snapshot every PK currently bound to the graph with provenance. Each entry exposes the owning table, the ordered PK column list, and ``pk_inference_tag`` (``None`` for catalog-declared keys, ``"profile"`` for keys promoted by ``_infer_missing_pks_from_profile``)."""
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
    """Snapshot every table's structural role state for the read-only export envelope."""
    records: list[dict[str, Any]] = []
    for tname in sg.tables:
        tbl = sg.tables[tname]
        record: dict[str, Any] = {
            "name": tname,
            "role": tbl.role,
            "role_owner": (tbl.role_owner.value if tbl.role_owner is not None else None),
        }
        original_name = (tbl.original_name or "").strip()
        if original_name and original_name != tname:
            record["original_name"] = original_name
        records.append(record)
    records.sort(key=lambda r: r["name"])
    return records


def _columns_current_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Snapshot every column's structural editable state for the read- only export envelope."""
    records: list[dict[str, Any]] = []
    for tname in sg.tables:
        tbl = sg.tables[tname]
        for cname, col in tbl.columns.items():
            vt = (col.value_type or "").strip() or (data_type_to_value_type(col.data_type) if col.data_type else "")
            record: dict[str, Any] = {
                "table": tname,
                "column": cname,
                "role": col.role,
                "role_owner": (col.role_owner.value if col.role_owner is not None else None),
                "sensitivity": col.sensitivity,
                "is_selectable": bool(col.is_selectable),
                "value_type": vt,
                "boolean_truth_value": col.boolean_truth_value,
            }
            original_name = (col.original_name or "").strip()
            if original_name and original_name != cname:
                record["original_name"] = original_name
            records.append(record)
    records.sort(key=lambda r: (r["table"], r["column"]))
    return records


def build_public_structure_document(
    *,
    inventory: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge read-only inventory with editable structural overrides for public export."""
    doc = {k: v for k, v in inventory.items() if k not in ("format_version", "version")}
    override_tables = dict(overrides.get("tables") or {})
    tables_out: list[dict[str, Any]] = []
    visible_tables: set[str] = set()
    visible_columns: set[str] = set()
    for table_entry in doc.get("tables") or []:
        if not isinstance(table_entry, dict):
            continue
        merged = dict(table_entry)
        tname = str(merged.get("name") or "").strip()
        tbl_ov = override_tables.get(tname) if tname else None
        if isinstance(tbl_ov, dict) and "role" in tbl_ov:
            merged["role"] = tbl_ov["role"]
        col_ov_map = dict(tbl_ov.get("columns") or {}) if isinstance(tbl_ov, dict) else {}
        cols_out: list[dict[str, Any]] = []
        for col_entry in merged.get("columns") or []:
            if not isinstance(col_entry, dict):
                continue
            col_merged = dict(col_entry)
            cname = str(col_merged.get("name") or "").strip()
            col_ov = col_ov_map.get(cname) if cname else None
            if isinstance(col_ov, dict):
                for key in STRUCTURE_COLUMN_EDITABLE_KEYS | frozenset({"role"}):
                    if key in col_ov:
                        col_merged[key] = col_ov[key]
            cols_out.append(col_merged)
            if tname and cname:
                visible_columns.add(f"{tname}.{cname}")
        merged["columns"] = cols_out
        if tname:
            visible_tables.add(tname)
        tables_out.append(merged)
    doc["tables"] = tables_out
    doc["table_count"] = len(tables_out)
    for key in ("foreign_keys_add", "foreign_keys_remove", "primary_keys_add", "primary_keys_remove"):
        if key not in overrides:
            continue
        raw = overrides[key]
        if not isinstance(raw, list):
            doc[key] = raw
            continue
        if key.startswith("foreign_keys_"):
            doc[key] = [
                entry
                for entry in raw
                if isinstance(entry, dict) and _fk_override_entry_visible(entry, visible_tables, visible_columns)
            ]
        else:
            doc[key] = [
                entry
                for entry in raw
                if isinstance(entry, dict) and _pk_override_entry_visible(entry, visible_tables, visible_columns)
            ]
    return doc


def _qualified_refs_from_fk_endpoint(endpoint: Any) -> list[str]:
    """Return ``table.column`` strings named by an FK endpoint value."""
    if isinstance(endpoint, str):
        text = endpoint.strip()
        if not text:
            return []
        if "." in text:
            return [text]
        return []
    if isinstance(endpoint, list):
        out: list[str] = []
        for item in endpoint:
            out.extend(_qualified_refs_from_fk_endpoint(item))
        return out
    if isinstance(endpoint, dict):
        tbl = str(endpoint.get("table") or "").strip()
        cols_raw = endpoint.get("columns") or endpoint.get("column") or []
        if isinstance(cols_raw, str):
            cols = [cols_raw]
        elif isinstance(cols_raw, list):
            cols = [str(c) for c in cols_raw if c]
        else:
            cols = []
        if not tbl or not cols:
            return []
        return [f"{tbl}.{c}" for c in cols]
    return []


def _fk_override_entry_visible(
    entry: Mapping[str, Any],
    visible_tables: set[str],
    visible_columns: set[str],
) -> bool:
    """True when every FK endpoint table/column survives the filtered inventory."""
    refs = _qualified_refs_from_fk_endpoint(entry.get("from")) + _qualified_refs_from_fk_endpoint(entry.get("to"))
    if not refs:
        return False
    for ref in refs:
        if "." not in ref:
            return False
        tname, _cname = ref.split(".", 1)
        if tname not in visible_tables:
            return False
        if ref not in visible_columns:
            return False
    return True


def _pk_override_entry_visible(
    entry: Mapping[str, Any],
    visible_tables: set[str],
    visible_columns: set[str],
) -> bool:
    """True when the PK table/column survives the filtered inventory."""
    tname = str(entry.get("table") or "").strip()
    cname = str(entry.get("column") or "").strip()
    if not tname or not cname:
        return False
    if tname not in visible_tables:
        return False
    return f"{tname}.{cname}" in visible_columns


def validate_public_structure_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a caller-supplied structure document before apply."""
    if not isinstance(document, Mapping):
        raise ConfigError("structure document must be a JSON object")
    require_exact_keys(
        document,
        allowed=PUBLIC_STRUCTURE_DOCUMENT_KEYS,
        required=frozenset({"tables"}),
        context="structure document",
    )
    tables_raw = document.get("tables")
    if not isinstance(tables_raw, list):
        raise ConfigError("structure document tables must be an array")
    for table_entry in tables_raw:
        if not isinstance(table_entry, dict):
            raise ConfigError("structure document tables entries must be objects")
        require_exact_keys(
            table_entry,
            allowed=frozenset({"name", "columns", "primary_key", "foreign_keys", "role"}),
            required=frozenset({"name", "columns"}),
            context="structure document table",
        )
        columns_raw = table_entry.get("columns")
        if not isinstance(columns_raw, list):
            raise ConfigError(f"structure document table {table_entry.get('name')!r}: columns must be an array")
        for col_entry in columns_raw:
            if not isinstance(col_entry, dict):
                raise ConfigError("structure document column entries must be objects")
            require_exact_keys(
                col_entry,
                allowed=frozenset({"name", "data_type", "role", "sensitivity", "usable", "boolean_truth_value"}),
                required=frozenset({"name", "data_type"}),
                context="structure document column",
            )
    for list_key in ("foreign_keys_add", "foreign_keys_remove", "primary_keys_add", "primary_keys_remove"):
        raw = document.get(list_key)
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise ConfigError(f"structure document {list_key} must be an array")
    return dict(document)


def structure_document_to_overrides_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a validated public structure document into an overrides apply payload."""
    validated = validate_public_structure_document(document)
    tables_override: dict[str, Any] = {}
    for table_entry in validated.get("tables") or []:
        if not isinstance(table_entry, dict):
            continue
        tname = str(table_entry.get("name") or "").strip()
        if not tname:
            continue
        tbl_override: dict[str, Any] = {}
        if "role" in table_entry:
            tbl_override["role"] = table_entry["role"]
        cols_override: dict[str, Any] = {}
        for col_entry in table_entry.get("columns") or []:
            if not isinstance(col_entry, dict):
                continue
            cname = str(col_entry.get("name") or "").strip()
            if not cname:
                continue
            col_override: dict[str, Any] = {}
            for key in STRUCTURE_COLUMN_EDITABLE_KEYS | frozenset({"role"}):
                if key in col_entry:
                    col_override[key] = col_entry[key]
            if col_override:
                cols_override[cname] = col_override
        if cols_override:
            tbl_override["columns"] = cols_override
        if tbl_override:
            tables_override[tname] = tbl_override
    payload: dict[str, Any] = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": tables_override,
        "foreign_keys_add": list(validated.get("foreign_keys_add") or []),
        "foreign_keys_remove": list(validated.get("foreign_keys_remove") or []),
        "primary_keys_add": list(validated.get("primary_keys_add") or []),
        "primary_keys_remove": list(validated.get("primary_keys_remove") or []),
    }
    return _validate_structure_edits(payload)


def dump_structure_edits(sg: SchemaGraph) -> dict[str, Any]:
    """Build the structural overrides JSON document from a built schema graph. The document contains the editable structural surface the user is allowed to mutate (``tables`` roles / sensitivity / boolean truth / usable, ``foreign_keys_add``, ``foreign_keys_remove``, ``primary_keys_add``, ``primary_keys_remove``) and a ``_readonly`` envelope showing the *current* structural graph state for reference (FKs with ``removable``, PKs with provenance, table/column structural metadata). Prose — table and column descriptions plus domain knowledge — is not part of this document; it round-trips through ``export_knowledge`` / ``apply_knowledge``. Editors should never modify ``_readonly`` — those entries are ignored on apply. Existing user-added FKs are surfaced under ``foreign_keys_add`` so editors can re-export and re-apply without losing them. Internal block lists (used by the system to suppress re-inference of FKs/PKs the user removed) live in the persisted sidecar under ``_internal`` and are never surfaced in the editable JSON."""
    sidecar = load_structure_sidecar(EngineConfig.SCHEMA_JSON_PATH)
    internal = dict((sidecar or {}).get("_internal") or {})
    stashed = getattr(sg, "_override_internal_blocks", None)
    if isinstance(stashed, dict):
        for key in ("fk_block_inferred", "pk_block_inferred", "catalog_fk_revoked"):
            if stashed.get(key):
                internal[key] = stashed[key]
    tables_out = {
        tname: tbl_doc for tname in sorted(sg.tables) if (tbl_doc := _table_override_value_dict(sg.tables[tname]))
    }
    return {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": tables_out,
        "foreign_keys_add": user_added_fks_dump(sg),
        "foreign_keys_remove": _fk_block_to_remove_entries(list(internal.get("fk_block_inferred") or [])),
        "primary_keys_add": user_added_pks_dump(sg),
        "primary_keys_remove": _pk_block_to_remove_entries(list(internal.get("pk_block_inferred") or [])),
        "_readonly": {
            "foreign_keys_current": _foreign_keys_current_dump(sg),
            "primary_keys_current": _primary_keys_current_dump(sg),
            "tables_current": _tables_current_dump(sg),
            "columns_current": _columns_current_dump(sg),
        },
    }


def _catalog_fk_keys(sg: SchemaGraph) -> set[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    """Return canonical keys for every catalog-declared FK edge on *sg*."""
    keys: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for tbl in sg.tables.values():
        for edge in tbl.foreign_keys:
            if edge.inference_tag is None:
                keys.add(edge_key(edge))
    return keys


def _is_user_asserted_pk_column(col: ColumnMetadata) -> bool:
    """Return True when *col* carries a user-promoted primary-key provenance tag."""
    return bool(col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE)


def _table_pk_catalog_locked(tbl: TableMetadata) -> bool:
    """Return True when any current PK column is an engine catalog key (``pk_inference_tag`` is ``None``)."""
    if not tbl.primary_key:
        return False
    for cname in tbl.primary_key:
        col = tbl.columns.get(cname)
        if col is not None and col.is_primary_key and col.pk_inference_tag is None:
            return True
    return False


def _validate_composite_pk_tuple(
    table_name: str,
    table: TableMetadata,
    col_names: list[str],
    *,
    dialect: Any | None,
) -> tuple[bool, str]:
    """Validate that *col_names* form a null-free uniquely identifying PK tuple for *table*."""
    if not col_names:
        return False, "empty primary key column list"
    rc = int(table.row_count or 0)
    for cname in col_names:
        col = table.columns.get(cname)
        if col is None:
            return False, f"unknown column {table_name}.{cname}"
        if float(col.null_ratio or 0.0) > 0.0:
            return False, f"column {table_name}.{cname} has null_ratio > 0; cannot be a primary key"
    if len(col_names) == 1:
        cname = col_names[0]
        col = table.columns[cname]
        dc = int(col.distinct_count or 0)
        unique_ok = bool(col.is_unique) or (rc > 0 and dc == rc and not col.distinct_from_sample)
        if not unique_ok:
            return (
                False,
                (
                    f"column {table_name}.{cname} not unique "
                    f"(row_count={rc}, distinct_count={dc}, is_unique={col.is_unique})"
                ),
            )
        return True, ""
    refresh = getattr(dialect, "refresh_composite_distinct_for_pk_inference", None) if dialect else None
    if refresh is None:
        return False, "composite primary key validation requires live dialect profiling"
    ft = refresh(table_name, col_names, table_kind=table.kind)
    if ft is None:
        return False, f"could not profile composite PK tuple for {table_name!r}"
    dist_ft, cnt_ft, _nr_ft = ft
    if cnt_ft > 0:
        table.row_count = cnt_ft
    if dist_ft != cnt_ft or cnt_ft <= 0:
        return False, (
            f"composite PK tuple for {table_name!r} not unique (distinct_tuples={dist_ft}, row_count={cnt_ft})"
        )
    return True, ""


def _clear_overridable_pk_columns(tbl: TableMetadata) -> None:
    """Drop overridable PK membership from *tbl* without touching catalog-locked PK columns."""
    kept: list[str] = []
    for cname in list(tbl.primary_key):
        col = tbl.columns.get(cname)
        if col is not None and col.pk_inference_tag is None:
            kept.append(cname)
            continue
        if col is not None:
            col.pk_inference_tag = None
    tbl.primary_key = kept


def _is_user_asserted_fk_edge(
    edge: FKEdge,
    catalog_keys: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> bool:
    """Return True when *edge* is a user override edge that is not redundant with a catalog FK key."""
    tag = edge.inference_tag or ""
    if not tag.startswith("user_override_"):
        return False
    return edge_key(edge) not in catalog_keys


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


def user_added_pks_dump(sg: SchemaGraph) -> list[dict[str, str]]:
    """Serialize user-promoted primary keys for overrides round-trip."""
    rows: list[dict[str, str]] = []
    for tname, tbl in sg.tables.items():
        for pkc in tbl.primary_key:
            col = tbl.columns.get(pkc)
            if col is not None and _is_user_asserted_pk_column(col):
                rows.append({"table": tname, "column": pkc})
    rows.sort(key=lambda r: (r["table"], r["column"]))
    return rows


def user_added_fks_dump(sg: SchemaGraph) -> list[dict[str, Any]]:
    """Re-emit currently-applied user-asserted FKs (structural FKEdges and semantic neighbor pairs) in ``foreign_keys_add`` shape so a fresh export is a faithful round-trip."""
    out: list[dict[str, Any]] = []
    catalog_keys = frozenset(_catalog_fk_keys(sg))
    for tname in sg.tables:
        for edge in sg.tables[tname].foreign_keys:
            if not _is_user_asserted_fk_edge(edge, catalog_keys):
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


def _dump_overrides_json_schema(path: str | Path) -> Path:
    """Write a JSON Schema describing the editable overrides surface alongside the editable file at *path*. The companion file is named ``<stem>.schema.json`` and lists the structural top-level shape (owned ``role`` objects under ``tables``, ``foreign_keys_add``, ``primary_keys_add``, ``foreign_keys_remove``, ``primary_keys_remove``, ``_readonly``, ``_internal``) plus the editable enum vocabularies sourced from :data:`STRUCTURE_EDITABLE_ENUMS`. Prose (descriptions and domain knowledge) is omitted here: it travels through the space-knowledge surface, not the overrides editor. Editor tooling (e.g. VS Code) can use this to power autocomplete and inline validation without re- deriving the contract from code."""
    target = Path(path).expanduser().resolve()
    schema_path = target.with_name(target.stem + ".schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    enums = STRUCTURE_EDITABLE_ENUMS
    tr_enum = list(enums.get("table_role", []))
    cr_enum = list(enums.get("column_role", []))
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
        "title": f"applied_structure v{STRUCTURE_DOCUMENT_VERSION}",
        "type": "object",
        "additionalProperties": False,
        "required": ["version"],
        "properties": {
            "version": {"const": STRUCTURE_DOCUMENT_VERSION},
            "tables": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": editable_table_role_schema,
                        "columns": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
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
    write_json_atomic(schema_path, schema_doc, sort_keys=True, indent=2)
    return schema_path


def dump_structure_to_path(
    sg: SchemaGraph,
    path: str | Path,
) -> Path:
    """Write the overrides editor document to *path*, replacing any existing file atomically."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dump_structure_edits(sg)
    write_json_atomic(target, payload, sort_keys=True, indent=2)
    _dump_overrides_json_schema(target)
    return target


def _validate_structure_edits(d: Any) -> dict[str, Any]:
    """Structurally validate the overrides JSON document; raise ``ValueError`` with a path on the first problem."""
    if not isinstance(d, dict):
        raise ValueError("overrides JSON must be a top-level object")
    if "foreign_keys_block_inferred" in d:
        raise ValueError(
            "overrides 'foreign_keys_block_inferred' is not accepted in the editable JSON; use 'foreign_keys_remove' instead (the system manages re-inference suppression internally)"
        )
    if "primary_keys_block_inferred" in d:
        raise ValueError(
            "overrides 'primary_keys_block_inferred' is not accepted in the editable JSON; use 'primary_keys_remove' instead"
        )
    allowed = STRUCTURE_TOP_LEVEL_EDIT_KEYS | {"_internal"}
    extra = set(d.keys()) - allowed
    if extra:
        prose = sorted(extra & STRUCTURE_PROSE_KEYS)
        if prose:
            raise ValueError(f"unsupported top-level keys in overrides: {prose!r}; {STRUCTURE_PROSE_REDIRECT_HINT}")
        raise ValueError(f"unsupported top-level keys in overrides: {sorted(extra)!r}")
    if "version" not in d:
        raise ValueError("overrides JSON missing 'version' field")
    if not format_versions_match(d["version"], STRUCTURE_DOCUMENT_VERSION):
        raise ValueError(f"overrides 'version' is {d['version']!r}; this build expects {STRUCTURE_DOCUMENT_VERSION}")
    tables = d.get("tables", {}) or {}
    if not isinstance(tables, dict):
        raise ValueError("overrides 'tables' must be an object keyed by qualified table name")
    valid_col_roles = {r.value for r in ColumnRole}
    valid_table_roles = {r.value for r in TableRole}
    for tname, tval in tables.items():
        if not isinstance(tval, dict):
            raise ValueError(f"tables.{tname}: must be an object")
        bad = set(tval.keys()) - STRUCTURE_TABLE_EDIT_KEYS
        if bad:
            prose = sorted(bad & STRUCTURE_PROSE_KEYS)
            if prose:
                raise ValueError(f"tables.{tname}: unsupported keys {prose!r}; {STRUCTURE_PROSE_REDIRECT_HINT}")
            raise ValueError(f"tables.{tname}: unsupported keys {sorted(bad)!r}")
        if "role" in tval:
            _validate_owned_role_json(tval["role"], f"tables.{tname}.role", valid_table_roles)
        cols = tval.get("columns", {}) or {}
        if not isinstance(cols, dict):
            raise ValueError(f"tables.{tname}.columns: must be an object")
        for cname, cval in cols.items():
            if not isinstance(cval, dict):
                raise ValueError(f"tables.{tname}.columns.{cname}: must be an object")
            cbad = set(cval.keys()) - STRUCTURE_COLUMN_EDITABLE_KEYS
            if cbad:
                prose = sorted(cbad & STRUCTURE_PROSE_KEYS)
                if prose:
                    raise ValueError(
                        f"tables.{tname}.columns.{cname}: unsupported keys {prose!r}; {STRUCTURE_PROSE_REDIRECT_HINT}"
                    )
                raise ValueError(f"tables.{tname}.columns.{cname}: unsupported keys {sorted(cbad)!r}")
            if "usable" in cval and cval["usable"] is False:
                raise ValueError(
                    f"tables.{tname}.columns.{cname}.usable: false is not allowed; use 'sensitivity' to hide a column"
                )
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


def _parse_document_domain_knowledge(
    document: dict[str, Any],
) -> tuple[DomainKnowledgeEntry, ...] | None:
    """Return normalized DK entries when the overrides document carries ``domain_knowledge``, else ``None``."""
    if "domain_knowledge" not in document:
        return None
    section = document.get("domain_knowledge") or {}
    raw_entries = list(section.get("entries") or []) if isinstance(section, dict) else []
    out: list[DomainKnowledgeEntry] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or "").strip()
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("kind") or DomainKnowledgeKind.GLOSSARY.value).strip() or DomainKnowledgeKind.GLOSSARY.value
        if not key or not text or key in seen:
            continue
        try:
            entry = DomainKnowledgeEntry.normalize(DomainKnowledgeEntry(key=key, text=text, kind=kind))
        except ConfigError:
            continue
        seen.add(entry.key)
        out.append(entry)
    return tuple(out)


def refine_domain_knowledge_via_llm(
    entries: Sequence[DomainKnowledgeEntry],
    sg: SchemaGraph,
) -> tuple[DomainKnowledgeEntry, ...]:
    """Refine applied domain-knowledge entries; never invent keys outside the applied set. Runs under :func:`llm_usage_build_scope` so the call attributes to engine/build usage."""
    applied = tuple(entries)
    if not applied:
        return ()
    allowed_keys = {e.key for e in applied}
    if not EngineConfig.llm_credentials_configured():
        return filter_schema_anchored_domain_knowledge(applied, sg)
    user_payload = stable_json(
        {
            "entries": [{"key": e.key, "kind": e.kind, "text": e.text} for e in applied],
        }
    )
    try:
        with llm_usage_build_scope():
            raw = LLMProvider.chat(
                system=DOMAIN_KNOWLEDGE_REFINER_SYSTEM,
                user=user_payload,
                task="domain_knowledge",
            )
    except (RuntimeError, OSError, TypeError) as exc:
        debug(f"[schema.refine_domain_knowledge] llm refinement failed; using originals: {exc!r}")
        return filter_schema_anchored_domain_knowledge(applied, sg)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"[schema.refine_domain_knowledge] LLM returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"[schema.refine_domain_knowledge] LLM JSON is not an object; got {type(parsed).__name__}")
    if "entries" not in parsed:
        raise ValueError("[schema.refine_domain_knowledge] LLM JSON missing 'entries' key")
    items = parsed["entries"]
    if not isinstance(items, list):
        raise ValueError(f"[schema.refine_domain_knowledge] 'entries' must be a list; got {type(items).__name__}")
    refined: list[DomainKnowledgeEntry] = []
    seen: set[str] = set()
    allowed_kinds = {member.value for member in DomainKnowledgeKind}
    by_key = {e.key: e for e in applied}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(
                f"[schema.refine_domain_knowledge] entry item must be an object; got {type(item).__name__}"
            )
        key = str(item.get("key") or "").strip()
        text = str(item.get("text") or "").strip()
        kind = str(item.get("kind") or DomainKnowledgeKind.GLOSSARY.value).strip() or DomainKnowledgeKind.GLOSSARY.value
        if not key or not text or key not in allowed_keys or key in seen:
            continue
        if kind not in allowed_kinds:
            kind = DomainKnowledgeKind.GLOSSARY.value
        referenced_entities = by_key[key].referenced_entities if key in by_key else frozenset()
        try:
            entry = DomainKnowledgeEntry.normalize(
                DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced_entities)
            )
        except ConfigError:
            continue
        seen.add(entry.key)
        refined.append(entry)
    if not refined:
        refined = list(applied)
    else:
        for e in applied:
            if e.key not in seen:
                refined.append(e)
                seen.add(e.key)
    return filter_schema_anchored_domain_knowledge(tuple(refined), sg)


def apply_document_domain_knowledge(
    document: dict[str, Any],
    sg: SchemaGraph,
) -> tuple[tuple[DomainKnowledgeEntry, ...] | None, int]:
    """Parse and refine ``domain_knowledge`` from an overrides document when present. Returns ``(None, 0)`` when the key is absent. When present, returns refined entries (possibly empty) and the count of entries after refine."""
    parsed = _parse_document_domain_knowledge(document)
    if parsed is None:
        return None, 0
    refined = refine_domain_knowledge_via_llm(parsed, sg)
    return refined, len(refined)


def _refine_descriptions_via_llm(changes: list[dict[str, Any]]) -> dict[str, str]:
    """Send a batch of description edits to the LLM and return ``{path: refined_text}``. Each change item includes ``path``, ``kind``, ``text``, and optional ``previous_text`` (the description string on the graph before this apply pass). Falls back to the original text for any path missing from the LLM response or when LLM credentials are absent."""
    if not changes:
        return {}
    if not EngineConfig.llm_credentials_configured():
        debug("[schema.refine_descriptions] llm not configured; skipping refinement")
        return {str(c["path"]): str(c["text"]) for c in changes}
    user_payload = stable_json({"items": changes})
    instructions = (
        "For each item, rewrite 'text' into a concise, role-aware description that (a) keeps every keyword the human added, and (b) matches the style of 'previous_text' when that field is non-empty. "
        'Output JSON of the form {"items": [{"path": "...", "text": "<refined>"}]} with one entry per input item, in the same order.'
    )
    try:
        raw = LLMProvider.chat(
            system=DESCRIPTION_REFINER_SYSTEM,
            user=instructions + "\n" + user_payload,
            task="default",
        )
    except (RuntimeError, OSError) as exc:
        debug(f"[schema.refine_descriptions] llm refinement failed; using originals: {exc!r}")
        return {c["path"]: c["text"] for c in changes}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"[schema.refine_descriptions] LLM returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"[schema.refine_descriptions] LLM JSON is not an object; got {type(parsed).__name__}")
    if "items" not in parsed:
        raise ValueError("[schema.refine_descriptions] LLM JSON missing 'items' key")
    items = parsed["items"]
    if not isinstance(items, list):
        raise ValueError(f"[schema.refine_descriptions] 'items' must be a list; got {type(items).__name__}")
    refined: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            raise ValueError(f"[schema.refine_descriptions] item must be an object; got {type(it).__name__}")
        p = it.get("path")
        t = it.get("text")
        if not isinstance(p, str) or not isinstance(t, str):
            raise ValueError(
                f"[schema.refine_descriptions] item path/text must be strings; "
                f"got path={type(p).__name__} text={type(t).__name__}"
            )
        if t.strip():
            refined[p] = t.strip()[:STRUCTURE_MAX_DESCRIPTION_CHARS]
    for c in changes:
        refined.setdefault(c["path"], c["text"])
    return refined


def load_structure_document_file(path: str | Path) -> dict[str, Any]:
    """Read *path* as UTF-8 JSON and return the structurally validated overrides document."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"overrides file not found: {p!s}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            d = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"overrides JSON parse failed: {exc.msg} (line {exc.lineno})") from exc
    return _validate_structure_edits(d)


def _column_override_snapshot(col: ColumnMetadata) -> dict[str, Any]:
    """Capture editable column override fields for rollback when a hidden column is touched."""
    return {
        "description": col.description,
        "description_owner": col.description_owner,
        "role": col.role,
        "role_owner": col.role_owner,
        "sensitivity": col.sensitivity,
        "boolean_truth_value": col.boolean_truth_value,
        "usable_override": col.usable_override,
    }


def _restore_column_override_snapshot(col: ColumnMetadata, snapshot: dict[str, Any]) -> None:
    """Restore editable column override fields from :func:`_column_override_snapshot`."""
    col.description = snapshot["description"]
    col.description_owner = snapshot["description_owner"]
    col.role = snapshot["role"]
    col.role_owner = snapshot["role_owner"]
    col.sensitivity = snapshot["sensitivity"]
    col.boolean_truth_value = snapshot["boolean_truth_value"]
    col.usable_override = snapshot["usable_override"]


def _override_endpoint_skip_or_raise(
    *,
    strict: bool,
    path: str,
    reason: str,
    skipped: list[OverrideSkip],
) -> None:
    unknown = (
        reason == "unknown table"
        or reason == "unknown column"
        or reason.startswith("unknown table")
        or reason.startswith("unknown column")
        or reason.startswith("unknown source table")
        or reason.startswith("unknown destination table")
        or reason.startswith("unknown source columns")
        or reason.startswith("unknown destination columns")
    )
    if strict and unknown:
        raise ConfigError(f"schema override at {path}: {reason}")
    skipped.append(OverrideSkip(path=path, reason=reason))


def _publish_schema_graph_override(live: SchemaGraph, working: SchemaGraph) -> None:
    """Atomically publish override mutations from *working* onto the live graph."""
    live.tables = working.tables
    live.join_paths_multi = working.join_paths_multi
    live.structural_hash = working.structural_hash
    live.profiling_hash = working.profiling_hash
    live.scope_hash = working.scope_hash
    live.effective_structural_hash = working.effective_structural_hash
    live.semantic_edges_hash = working.semantic_edges_hash
    live.schema_graph_id = working.schema_graph_id
    live.include = working.include
    live.notes_hash = working.notes_hash
    live.scope_descriptor = working.scope_descriptor
    live.schema_revision = working.schema_revision
    live.schema_stats = working.schema_stats
    live._stats_dirty = working._stats_dirty
    live.deny_columns = working.deny_columns
    live.disallowed_columns = working.disallowed_columns
    live.enum_values = working.enum_values
    for tbl in live.tables.values():
        object.__setattr__(tbl, "_owner_graph", live)


def apply_structure_to_graph(
    sg: SchemaGraph,
    overrides: dict[str, Any],
    *,
    dialect: Any | None = None,
    strict: bool = False,
    notes_content: str | None = None,
    manifest: FederationManifest | None = None,
    previous_schema: SchemaGraph | None = None,
) -> StructureReport:
    """Apply a validated overrides document to *sg* in place and return a ``StructureReport``. Skips entries that reference unknown tables/columns or that would break PK/FK joins, recording each skip in the report. Description fields that differ from the current value are sent through a single batched LLM refinement pass when LLM credentials are configured (the call falls through cleanly with the raw text otherwise). The internal envelope ``overrides["_internal"]`` (system-managed; never round-tripped into the editable JSON) carries persistent block lists for inferred FKs and PKs that the user removed. ``foreign_keys_remove`` and ``primary_keys_remove`` always remove the requested edge/PK *and* (when the removed item was inferred, not catalog) auto-promote it into the matching internal block list so subsequent rebuilds suppress re-inference."""
    live = sg
    sg = copy.deepcopy(sg)
    skipped: list[OverrideSkip] = []
    table_edits = 0
    column_edits = 0
    description_changes: list[dict[str, Any]] = []

    description_targets: dict[str, tuple[str, str | None, DescriptionOwner]] = {}
    direct_descriptions_refined = 0
    sensitivity_lowered_tables: set[str] = set()
    sensitivity_increased_columns_set: set[str] = set()
    newly_usable_columns: set[tuple[str, str]] = set()

    valid_table_roles = {r.value for r in TableRole}
    valid_col_roles = {r.value for r in ColumnRole}
    tables = overrides.get("tables", {}) or {}
    stale_table_names: list[str] = []
    for tname, tval in list(tables.items()):
        tbl = sg.tables.get(tname)
        if tbl is None:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=f"tables.{tname}",
                reason="unknown table",
                skipped=skipped,
            )
            continue
        if not isinstance(tval, dict):
            continue
        table_path = f"tables.{tname}"
        if _override_entry_blocked_by_recreation(
            table_path,
            tname,
            tval,
            current_hash=table_structural_hash_fp(tbl),
            previous_schema=previous_schema,
            current_schema=sg,
            skipped=skipped,
        ):
            continue
        touched_table = False
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
                        code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                        details=(("path", tr_path), ("reason", str(exc))),
                    )
                    tval.pop("role", None)
                else:
                    if tbl.role != r_val:
                        if RoleOwner.can_overwrite(tbl.role_owner, r_own):
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
                    elif r_own != (tbl.role_owner if tbl.role_owner is not None else RoleOwner.CATALOG):
                        if RoleOwner.can_overwrite(tbl.role_owner, r_own):
                            tbl.role_owner = r_own
                            touched_table = True
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
                    _override_endpoint_skip_or_raise(
                        strict=strict,
                        path=f"tables.{tname}.columns.{cname}",
                        reason="unknown column",
                        skipped=skipped,
                    )
                    continue
                column_path = f"tables.{tname}.columns.{cname}"
                if _override_entry_blocked_by_recreation(
                    column_path,
                    f"{tname}.{cname}",
                    cval,
                    current_hash=_column_structural_hash_fp(tbl, cname),
                    previous_schema=previous_schema,
                    current_schema=sg,
                    skipped=skipped,
                ):
                    continue
                col_snapshot = _column_override_snapshot(col)
                touched_col = False
                if "sensitivity" in cval:
                    sens_raw = cval["sensitivity"]
                    if _override_json_null_sentinel(sens_raw):
                        cval.pop("sensitivity", None)
                    elif SensitivityClassification.coerce(sens_raw) is None:
                        skipped.append(
                            OverrideSkip(
                                path=f"tables.{tname}.columns.{cname}.sensitivity",
                                reason="invalid sensitivity value",
                            )
                        )
                    else:
                        merged = SensitivityClassification.from_dict({"sensitivity": sens_raw})
                        prev_sens = col.sensitivity
                        if merged != prev_sens:
                            SensitivityClassification.apply_to(col, merged)
                            if prev_sens != SensitivityClassification.NONE and merged == SensitivityClassification.NONE:
                                sensitivity_lowered_tables.add(tname)
                            prev_rank = (
                                2
                                if prev_sens == SensitivityClassification.HIDDEN
                                else 1
                                if prev_sens == SensitivityClassification.RESTRICTED
                                else 0
                            )
                            new_rank = (
                                2
                                if merged == SensitivityClassification.HIDDEN
                                else 1
                                if merged == SensitivityClassification.RESTRICTED
                                else 0
                            )
                            if new_rank > prev_rank:
                                sensitivity_increased_columns_set.add(f"{tname}.{cname}")
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
                                code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                                details=(("path", cr_path), ("reason", str(exc))),
                            )
                            cval.pop("role", None)
                        else:
                            if col.role != r_val:
                                if RoleOwner.can_overwrite(col.role_owner, r_own):
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
                            str(x).strip().lower()
                            for x in (col.frequent_values or [])
                            if x is not None and str(x).strip()
                        }
                        allowed = set(tops)
                        if col.boolean_truth_value:
                            allowed.add(str(col.boolean_truth_value).strip().lower())
                        wl = raw_bt.strip().lower()
                        if wl not in allowed:
                            skipped.append(
                                OverrideSkip(
                                    path=f"tables.{tname}.columns.{cname}.boolean_truth_value",
                                    reason="boolean_truth_value must match an observed literal (frequent_values) for this column",
                                )
                            )
                        else:
                            canon = raw_bt.strip()
                            for x in col.frequent_values or []:
                                if str(x).strip().lower() == wl:
                                    canon = str(x).strip()
                                    break
                            if col.boolean_truth_value != canon:
                                col.boolean_truth_value = canon
                                touched_col = True
                if "usable" in cval and cval["usable"] is True:
                    if col.usable_override is not True:
                        col.usable_override = True
                        newly_usable_columns.add((tname, cname))
                        touched_col = True
                if touched_col:
                    br_after = col.visibility_block_reason()
                    if br_after is not None:
                        _restore_column_override_snapshot(col, col_snapshot)
                        touched_col = False
                        skipped.append(
                            OverrideSkip(
                                path=f"tables.{tname}.columns.{cname}",
                                reason=f"column hidden ({br_after.value}); override skipped",
                                code="hidden_column_override",
                            )
                        )
                if touched_col:
                    column_edits += 1
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
            desc_meta: TableMetadata | ColumnMetadata = sg.tables[tname]
        else:
            desc_meta = sg.tables[tname].columns[cname]
        if DescriptionOwner.set_on(desc_meta, cleaned, desc_owner):
            descriptions_refined += 1
    descriptions_refined += direct_descriptions_refined

    if sensitivity_lowered_tables:
        if dialect is not None:
            _profile_values_only(dialect, sg, sensitivity_lowered_tables)
        else:
            notify(
                "Schema override lowered sensitivity on one or more columns; profile samples stay empty until a full rebuild with a live dialect.",
                stage="schema",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                level="info",
            )

    if newly_usable_columns:
        skip_columns = {
            (tbl.name, col.name)
            for tbl in sg.tables.values()
            for col in tbl.columns.values()
            if (tbl.name, col.name) not in newly_usable_columns
        }
        apply_column_roles_llm(sg, notes_content=notes_content, skip_columns=skip_columns)

    fks_added = 0
    fks_endorsed = 0

    def _endpoint_to_str(ep: Any) -> str:
        if isinstance(ep, str):
            return ep
        if isinstance(ep, list):
            return "|".join(str(x) for x in ep)
        return str(ep)

    internal = overrides.get("_internal")
    if not isinstance(internal, dict):
        internal = {}
    overrides["_internal"] = internal
    revoked_catalog_pairs = _catalog_fk_revoked_pairs(internal)

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
        frm_cmp = _endpoint_to_str(fk.get("from"))
        to_cmp = _endpoint_to_str(fk.get("to"))
        if (frm_cmp, to_cmp) in revoked_catalog_pairs:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog FK revoked by upstream DDL; cannot resurrect via foreign_keys_add",
                )
            )
            continue
        src = split_fk_endpoint(fk.get("from"), manifest=manifest, schema=sg)
        dst = split_fk_endpoint(fk.get("to"), manifest=manifest, schema=sg)
        if src is None or dst is None:
            skipped.append(OverrideSkip(path=path, reason="malformed 'from' or 'to' endpoint"))
            continue
        src_table, src_cols = src
        dst_table, dst_cols = dst
        if len(src_cols) != len(dst_cols):
            skipped.append(OverrideSkip(path=path, reason="from/to column counts differ"))
            continue
        if src_table not in sg.tables:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown source table {src_table!r}",
                skipped=skipped,
            )
            continue
        if dst_table not in sg.tables:
            if src_table in sg.tables:
                reason = (
                    "cross-source foreign keys must be declared in the federation manifest, "
                    f"not member schema overrides (unknown destination table {dst_table!r})"
                )
            else:
                reason = f"unknown destination table {dst_table!r}"
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=reason,
                skipped=skipped,
            )
            continue
        src_tbl = sg.tables[src_table]
        dst_tbl = sg.tables[dst_table]
        missing_src = [c for c in src_cols if c not in src_tbl.columns]
        if missing_src:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown source columns {missing_src!r}",
                skipped=skipped,
            )
            continue
        missing_dst = [c for c in dst_cols if c not in dst_tbl.columns]
        if missing_dst:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown destination columns {missing_dst!r}",
                skipped=skipped,
            )
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
        notes_structural = str(fk.get("provenance") or "").strip() == "notes_structural"
        inference_tag = InferenceTag.NOTES_STRUCTURAL if notes_structural else InferenceTag.USER_STRUCTURAL
        edge = FKEdge(
            src_table=src_table,
            src_cols=list(src_cols),
            dst_table=dst_table,
            dst_cols=list(dst_cols),
            inference_tag=inference_tag,
        )
        existing_by_key = {edge_key(e): e for e in src_tbl.foreign_keys}
        ek = edge_key(edge)
        if ek in existing_by_key:
            existing_edge = existing_by_key[ek]
            if existing_edge.inference_tag is None:
                existing_edge.inference_tag = inference_tag
                fks_endorsed += 1
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason=(
                            "catalog FK already present; user structural assertion recorded as user_override_structural (endorsement)"
                        ),
                    )
                )
                continue
            skipped.append(OverrideSkip(path=path, reason="duplicate_structural_edge"))
            continue
        dst_is_pk = all(dst_tbl.columns[c].is_primary_key for c in dst_cols)
        src_is_non_pk = all(not src_tbl.columns[c].is_primary_key for c in src_cols)
        if not (dst_is_pk and src_is_non_pk):
            if catalog_fk_graph_is_connected(sg):
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason="structural FK requires child=non-PK and parent=PK when graph is connected",
                        code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                    )
                )
                notify(
                    f"Skipped {path}: structural FK parent must be PK and child non-PK on a connected graph.",
                    stage="structure",
                    code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                )
                continue
            if len(src_cols) == 1 and len(dst_cols) == 1:
                quad = (src_table, src_cols[0], dst_table, dst_cols[0])
                mirror = (dst_table, dst_cols[0], src_table, src_cols[0])
                src_tbl._user_semantic_neighbors.append(quad)
                dst_tbl._user_semantic_neighbors.append(mirror)
                fks_added += 1
                skipped.append(
                    OverrideSkip(
                        path=path,
                        reason="demoted to semantic edge (parent/child PK gate failed on disconnected graph)",
                        code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                    )
                )
                continue
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="structural FK parent/child PK gate failed",
                    code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                )
            )
            continue
        src_tbl.foreign_keys.append(edge)
        fks_added += 1

    fks_removed = 0
    pks_blocked = 0

    fk_block_list: list[dict[str, Any]] = list(internal.get("fk_block_inferred", []) or [])
    pk_block_list: list[dict[str, Any]] = list(internal.get("pk_block_inferred", []) or [])

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
        if not isinstance(fk, dict):
            skipped.append(OverrideSkip(path=path, reason="malformed_fk_entry"))
            continue
        src = split_fk_endpoint(fk.get("from"), manifest=manifest, schema=sg)
        dst = split_fk_endpoint(fk.get("to"), manifest=manifest, schema=sg)
        if src is None or dst is None:
            skipped.append(OverrideSkip(path=path, reason="malformed 'from' or 'to' endpoint"))
            continue
        src_table, src_cols = src
        dst_table, dst_cols = dst
        if src_table not in sg.tables:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown source table {src_table!r}",
                skipped=skipped,
            )
            continue
        if dst_table not in sg.tables:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown destination table {dst_table!r}",
                skipped=skipped,
            )
            continue
        src_tbl = sg.tables[src_table]
        target = FKEdge(
            src_table=src_table,
            src_cols=list(src_cols),
            dst_table=dst_table,
            dst_cols=list(dst_cols),
        )
        target_key = edge_key(target)
        match_idx = -1
        removed_edge: FKEdge | None = None
        for j, existing_edge in enumerate(src_tbl.foreign_keys):
            if edge_key(existing_edge) == target_key:
                match_idx = j
                removed_edge = existing_edge
                break
        if match_idx < 0 or removed_edge is None:
            skipped.append(OverrideSkip(path=path, reason="no matching foreign key on source table"))
            continue
        if removed_edge.inference_tag is None:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog FK (inference_tag is null) cannot be removed via overrides",
                )
            )
            continue
        del src_tbl.foreign_keys[match_idx]
        fks_removed += 1
        if not (removed_edge.inference_tag or "").startswith("user_override_"):
            entry = {"from": fk.get("from"), "to": fk.get("to")}
            key = (_endpoint_to_str(entry["from"]), _endpoint_to_str(entry["to"]))
            if key not in fk_block_seen:
                fk_block_list.append(entry)
                fk_block_seen.add(key)

    for i, entry in enumerate(list(fk_block_list)):
        path = f"_internal.fk_block_inferred[{i}]"
        src = split_fk_endpoint(entry.get("from"), manifest=manifest, schema=sg)
        dst = split_fk_endpoint(entry.get("to"), manifest=manifest, schema=sg)
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
        target_key = edge_key(
            FKEdge(
                src_table=src_table,
                src_cols=list(src_cols),
                dst_table=dst_table,
                dst_cols=list(dst_cols),
            )
        )
        match_idx = -1
        blocked_edge: FKEdge | None = None
        for j, existing_edge in enumerate(src_tbl.foreign_keys):
            if edge_key(existing_edge) == target_key:
                match_idx = j
                blocked_edge = existing_edge
                break
        if match_idx < 0 or blocked_edge is None:
            skipped.append(OverrideSkip(path=path, reason="fk_block_no_match"))
            continue
        if blocked_edge.inference_tag is None:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="catalog FK cannot be blocked (block applies only to inferred edges)",
                )
            )
            continue
        if (blocked_edge.inference_tag or "").startswith("user_override_"):
            skipped.append(OverrideSkip(path=path, reason="fk_block_user_override_edge"))
            continue
        del src_tbl.foreign_keys[match_idx]
        fks_removed += 1

    pks_added = 0
    pks_endorsed = 0
    pk_add_by_table: dict[str, list[tuple[int, str, str]]] = {}
    for i, entry in enumerate(overrides.get("primary_keys_add", []) or []):
        path = f"primary_keys_add[{i}]"
        tname = entry.get("table")
        cname = entry.get("column")
        if not isinstance(tname, str) or not isinstance(cname, str):
            skipped.append(OverrideSkip(path=path, reason="malformed 'table' or 'column'"))
            continue
        tbl = sg.tables.get(tname)
        if tbl is None:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown table {tname!r}",
                skipped=skipped,
            )
            continue
        col = tbl.columns.get(cname)
        if col is None:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown column {tname}.{cname}",
                skipped=skipped,
            )
            continue
        pk_add_by_table.setdefault(tname, []).append((i, path, cname))

    for tname, entries in pk_add_by_table.items():
        tbl = sg.tables[tname]
        target_cols: list[str] = []
        paths: list[str] = []
        for _idx, path, cname in entries:
            if cname not in target_cols:
                target_cols.append(cname)
                paths.append(path)
        catalog_locked = _table_pk_catalog_locked(tbl)
        if catalog_locked:
            for _idx, path, cname in entries:
                col = tbl.columns[cname]
                if not col.is_primary_key:
                    reason = (
                        f"engine catalog primary key on {tname!r} cannot be extended; "
                        f"column {cname!r} is not in the current PK"
                    )
                    skipped.append(
                        OverrideSkip(path=path, reason=reason, code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP),
                    )
                    notify(
                        f"Skipped {path}: {reason}",
                        stage="structure",
                        code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                    )
                    continue
                if col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE:
                    skipped.append(OverrideSkip(path=path, reason="pk_already_user_override"))
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
                    ),
                )
            continue

        endorse_only = all(c in tbl.primary_key for c in target_cols) and len(target_cols) == len(entries)
        if endorse_only:
            for _idx, path, cname in entries:
                col = tbl.columns[cname]
                if col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE:
                    skipped.append(OverrideSkip(path=path, reason="pk_already_user_override"))
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
                    ),
                )
            continue

        prev_pk = list(tbl.primary_key)
        ok, reason = _validate_composite_pk_tuple(tname, tbl, target_cols, dialect=dialect)
        if not ok:
            for path in paths:
                skipped.append(
                    OverrideSkip(path=path, reason=reason, code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP),
                )
                notify(
                    f"Skipped {path}: {reason}",
                    stage="structure",
                    code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                )
            tbl.primary_key = prev_pk
            continue

        if set(target_cols) != set(prev_pk):
            _clear_overridable_pk_columns(tbl)
        for cname in target_cols:
            col = tbl.columns[cname]
            col.pk_inference_tag = PkInferenceTag.USER_OVERRIDE
            if cname not in tbl.primary_key:
                tbl.primary_key.append(cname)
        pks_added += len(target_cols)

    for i, entry in enumerate(overrides.get("primary_keys_remove", []) or []):
        path = f"primary_keys_remove[{i}]"
        tname = entry.get("table")
        cname = entry.get("column")
        if not isinstance(tname, str) or not isinstance(cname, str):
            skipped.append(OverrideSkip(path=path, reason="malformed 'table' or 'column'"))
            continue
        tbl = sg.tables.get(tname)
        if tbl is None:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown table {tname!r}",
                skipped=skipped,
            )
            continue
        col = tbl.columns.get(cname)
        if col is None:
            _override_endpoint_skip_or_raise(
                strict=strict,
                path=path,
                reason=f"unknown column {tname}.{cname}",
                skipped=skipped,
            )
            continue
        if col.pk_inference_tag is None and col.is_primary_key:
            skipped.append(
                OverrideSkip(
                    path=path,
                    reason="engine catalog PK cannot be removed (only inferred or DDL-declared PKs are removable)",
                    code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
                )
            )
            notify(
                f"Skipped {path}: engine catalog primary key is locked.",
                stage="structure",
                code=DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP,
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
            src_ep = split_fk_endpoint(blk_entry.get("from"), manifest=manifest, schema=sg)
            dst_ep = split_fk_endpoint(blk_entry.get("to"), manifest=manifest, schema=sg)
            if src_ep is None or dst_ep is None:
                continue
            st, scols = src_ep
            dt, dcols = dst_ep
            fk_blocked_keys.add((st, tuple(scols), dt, tuple(dcols)))
        reinferred = run_fk_inference_if_disconnected(sg, blocked=frozenset(fk_blocked_keys))
        fks_added += reinferred

    internal["fk_block_inferred"] = fk_block_list
    internal["pk_block_inferred"] = pk_block_list

    cast(Any, live)._override_internal_blocks = {
        "fk_block_inferred": list(fk_block_list),
        "pk_block_inferred": list(pk_block_list),
        "catalog_fk_revoked": list(internal.get("catalog_fk_revoked") or []),
    }

    replay_user_semantic_neighbors_to_columns(sg)

    collapsed = collapse_redundant_inferences(sg, skipped)

    coerced = coerce_pk_fk_columns_to_identifier(sg)
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
        sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
        assign_schema_graph_hashes(sg, schema_context_from_graph(sg), sg.notes_sha256)
        sg.refresh_schema_stats()
        sg.schema_revision = int(getattr(sg, "schema_revision", 0)) + 1
        _publish_schema_graph_override(live, sg)

    pk_block_post_sig = frozenset(
        (str(e.get("table", "")), str(e.get("column", ""))) for e in pk_block_list if isinstance(e, dict)
    )
    fk_block_post_sig = frozenset(
        (_endpoint_to_str(e.get("from")), _endpoint_to_str(e.get("to"))) for e in fk_block_list if isinstance(e, dict)
    )
    changed_pk_blocks = pk_block_pre_sig != pk_block_post_sig
    changed_fk_blocks = fk_block_pre_sig != fk_block_post_sig

    return StructureReport(
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
        sensitivity_increased_columns=frozenset(sensitivity_increased_columns_set),
        skipped=tuple(skipped),
    )


def load_structure_sidecar(schema_json_path: str | Path) -> dict[str, Any] | None:
    """Read the persisted overrides sidecar; return ``None`` when missing or unreadable. A corrupt sidecar is logged and treated as missing so a single bad write never blocks schema rebuilds. The returned document is structurally validated against the current ``STRUCTURE_DOCUMENT_VERSION``; mismatched versions raise :class:`~aetherdialect._config.ConfigError` so callers can distinguish drift from absence."""
    path = structure_sidecar_path(schema_json_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        debug(f"[schema.load_structure_sidecar] ignoring corrupt sidecar {path!s}: {exc!r}")
        return None
    if not isinstance(d, dict):
        debug(f"[schema.load_structure_sidecar] ignoring non-object sidecar {path!s}")
        return None
    sidecar_version = d.get("version")
    if not format_versions_match(sidecar_version, STRUCTURE_DOCUMENT_VERSION):
        raise ConfigError(
            f"overrides sidecar at {path!r} has version {sidecar_version!r}; "
            f"this build expects {STRUCTURE_DOCUMENT_VERSION}. "
            f"Delete {path!r} (or the engine artifacts directory) and re-run "
            f"initialize_aether_engine so the sidecar is rebuilt from scratch."
        )
    try:
        meta_source_hash = d.pop("source_schema_hash", None)
        meta_applied_at = d.pop("applied_at", None)
        meta_metadata_hash = d.pop("metadata_hash", None)
        validated = _validate_structure_edits(d)
        if meta_source_hash is not None:
            validated["source_schema_hash"] = meta_source_hash
        if meta_applied_at is not None:
            validated["applied_at"] = meta_applied_at
        if meta_metadata_hash is not None:
            validated["metadata_hash"] = meta_metadata_hash
        return validated
    except ValueError as exc:
        debug(f"[schema.load_structure_sidecar] ignoring corrupt sidecar {path!s}: {exc!r}")
        return None


def user_pinned_columns_from_sidecar(
    schema_json_path: str | Path,
) -> set[tuple[str, str]]:
    """Return ``(table, column)`` pairs pinned by explicit role, sensitivity, or user-owned description."""
    sidecar = load_structure_sidecar(schema_json_path)
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
            if "role" in cval or "sensitivity" in cval or "usable" in cval:
                pinned.add((str(tname), str(cname)))
            elif "description" in cval:
                desc = cval["description"]
                if isinstance(desc, dict) and desc.get("owner") == DescriptionOwner.USER_OVERRIDE.value:
                    pinned.add((str(tname), str(cname)))
    return pinned


def migrate_sidecar_for_diff(
    schema_json_path: str | Path,
    diff: SchemaDiff,
    *,
    fk_remaps: tuple[Any, ...] = (),
    pk_remaps: tuple[Any, ...] = (),
) -> bool:
    """Rewrite the persisted overrides sidecar in place so user-edited entries survive table/column renames. Loads the sidecar, applies *diff*'s table renames to top-level ``tables`` keys and any FK endpoints in ``foreign_keys_add`` / ``_internal.fk_block_inferred`` / ``_internal.pk_block_inferred``, then applies each ``per_table`` column rename to the column-keyed ``tables[<name>].columns`` map and to FK ``from`` / ``to`` endpoints that reference the renamed pair. Saves only when something actually changed. Returns True when a write occurred."""
    sidecar = load_structure_sidecar(schema_json_path)
    if sidecar is None:
        return False
    table_renames: dict[str, str] = {old: new for old, new in diff.table_renames if old != new}
    column_renames_by_new_table: dict[str, dict[str, str]] = {}
    for tname, td in diff.per_table.items():
        col_renames = {old: new for old, new in td.renamed_columns if old != new}
        if col_renames:
            column_renames_by_new_table[tname] = col_renames
    has_drops = bool(diff.dropped_tables) or any(td.dropped_columns for td in diff.per_table.values())
    if not table_renames and not column_renames_by_new_table and not fk_remaps and not pk_remaps and not has_drops:
        return False
    changed = False
    tables_block = sidecar.get("tables") or {}
    if isinstance(tables_block, dict) and diff.dropped_tables:
        for tname in diff.dropped_tables:
            if tname in tables_block:
                del tables_block[tname]
                changed = True
    if isinstance(tables_block, dict):
        for tname, td in diff.per_table.items():
            if not td.dropped_columns:
                continue
            tval = tables_block.get(tname)
            if not isinstance(tval, dict):
                continue
            cols_block = tval.get("columns")
            if not isinstance(cols_block, dict):
                continue
            for cname in td.dropped_columns:
                if cname in cols_block:
                    del cols_block[cname]
                    changed = True
    if isinstance(tables_block, dict) and table_renames:
        new_tables_block: dict[str, Any] = {}
        for tname, tval in tables_block.items():
            new_name = str(table_renames.get(tname, tname))
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
                new_cname = str(col_renames.get(cname, cname))
                if new_cname != cname:
                    changed = True
                new_cols_block[new_cname] = cval
            tval["columns"] = new_cols_block

    def _remap_endpoint(endpoint: Any) -> tuple[Any, bool]:
        if isinstance(endpoint, dict):
            tbl = str(endpoint.get("table", "") or "")
            cols_raw = endpoint.get("columns") or endpoint.get("column") or []
            if isinstance(cols_raw, str):
                cols = [cols_raw]
            elif isinstance(cols_raw, list):
                cols = [str(c) for c in cols_raw]
            else:
                return endpoint, False
            new_tbl = table_renames.get(tbl, tbl)
            col_renames = column_renames_by_new_table.get(new_tbl) or {}
            new_cols = [col_renames.get(c, c) for c in cols]
            rebuilt_dict = {"table": new_tbl, "columns": new_cols}
            return rebuilt_dict, rebuilt_dict != endpoint
        if isinstance(endpoint, list) and endpoint:
            remapped: list[Any] = []
            changed = False
            for ep in endpoint:
                new_ep, did = _remap_endpoint(ep)
                remapped.append(new_ep)
                changed = changed or did
            return remapped, changed
        if isinstance(endpoint, str) and "." in endpoint:
            tbl, _, rest = endpoint.partition(".")
            new_tbl = table_renames.get(tbl, tbl)
            col_renames = column_renames_by_new_table.get(new_tbl) or {}
            cols = rest.split(",") if "," in rest else [rest]
            new_cols = [col_renames.get(c, c) for c in cols]
            rebuilt_str = f"{new_tbl}." + ",".join(new_cols)
            return rebuilt_str, rebuilt_str != endpoint
        return endpoint, False

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
    pk_col_remap: dict[str, dict[str, str]] = {}
    for entry in pk_remaps:
        if getattr(entry, "entry_type", None) != "pk_remap":
            continue
        tbl = str(getattr(entry, "table", "") or "")
        old_cols = [c.strip() for c in str(getattr(entry, "from_name", "") or "").split(",") if c.strip()]
        new_cols = [c.strip() for c in str(getattr(entry, "to_name", "") or "").split(",") if c.strip()]
        if len(old_cols) != len(new_cols):
            continue
        pk_col_remap.setdefault(tbl, {}).update(dict(zip(old_cols, new_cols, strict=True)))
    if pk_col_remap:
        pk_add_top = sidecar.get("primary_keys_add") or []
        if isinstance(pk_add_top, list):
            for entry in pk_add_top:
                if not isinstance(entry, dict):
                    continue
                tbl = str(entry.get("table", ""))
                col = str(entry.get("column", ""))
                new_col = (pk_col_remap.get(tbl) or {}).get(col, col)
                if new_col != col:
                    entry["column"] = new_col
                    changed = True
    for entry in fk_remaps:
        if getattr(entry, "entry_type", None) != "fk_remap":
            continue
        child = str(getattr(entry, "table", "") or "")
        old_parent = str(getattr(entry, "from_name", "") or "")
        new_parent = str(getattr(entry, "to_name", "") or "")
        if not child or not old_parent or not new_parent:
            continue
        fk_add = sidecar.get("foreign_keys_add") or []
        if not isinstance(fk_add, list):
            continue
        for fk_entry in fk_add:
            if not isinstance(fk_entry, dict):
                continue
            src_ep = fk_entry.get("from")
            if isinstance(src_ep, dict) and str(src_ep.get("table", "")) == child:
                dst_ep = fk_entry.get("to")
                if isinstance(dst_ep, dict) and str(dst_ep.get("table", "")) == old_parent:
                    dst_ep["table"] = new_parent
                    changed = True
    if not changed:
        return False
    source_hash = sidecar.get("source_schema_hash") or ""
    resolved_cache = str(Path(schema_json_path).expanduser().resolve())
    snap = load_schema_graph_snapshot(resolved_cache)
    meta_hash = compute_metadata_hash(snap) if snap is not None else str(sidecar.get("metadata_hash") or "")
    save_structure_sidecar(
        schema_json_path,
        sidecar,
        source_schema_hash=source_hash,
        metadata_hash=meta_hash,
    )
    return True


def _sidecar_fk_endpoints_exist(sg: SchemaGraph, entry: dict[str, Any]) -> bool:
    """Return True when both FK endpoints reference existing tables and columns on *sg*."""
    src = split_fk_endpoint(entry.get("from"), schema=sg)
    dst = split_fk_endpoint(entry.get("to"), schema=sg)
    if src is None or dst is None:
        return False
    st, scols = src
    dt, dcols = dst
    if st not in sg.tables or dt not in sg.tables:
        return False
    stbl = sg.tables[st]
    dtbl = sg.tables[dt]
    return all(c in stbl.columns for c in scols) and all(c in dtbl.columns for c in dcols)


def reconcile_sidecar_against_graph(sg: SchemaGraph, schema_json_path: str | Path) -> SidecarReconcileReport:
    """Drop persisted overrides whose endpoints are missing from *sg*; rewrite the sidecar when needed."""
    sidecar = load_structure_sidecar(schema_json_path)
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
    save_structure_sidecar(
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
        schema_graph_id=schema.schema_graph_id,
        notes_hash=schema.notes_hash,
        semantic_edges_hash=schema.semantic_edges_hash,
        last_migration_tier=MigrationTier.DESTRUCTIVE.value,
        last_action="destructive",
    )
    sidecar_report = reconcile_sidecar_against_graph(schema, EngineConfig.SCHEMA_JSON_PATH)
    if sidecar_report.pruned_paths:
        notify(
            f"  Overrides sidecar pruned {len(sidecar_report.pruned_paths)} stale "
            f"entr{'y' if len(sidecar_report.pruned_paths) == 1 else 'ies'} after learning reset migration.",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("pruned_count", str(len(sidecar_report.pruned_paths))),),
        )


def _write_overrides_sidecar_payload(
    path: Path,
    doc: dict[str, Any],
    *,
    source_schema_hash: str,
    metadata_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    internal = doc.get("_internal", {}) or {}
    payload: dict[str, Any] = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "source_schema_hash": source_schema_hash,
        "metadata_hash": metadata_hash,
        "applied_at": datetime.now(UTC).isoformat(),
        "tables": doc.get("tables", {}) or {},
        "foreign_keys_add": doc.get("foreign_keys_add", []) or [],
        "primary_keys_add": doc.get("primary_keys_add", []) or [],
        "_internal": {
            "fk_block_inferred": list(internal.get("fk_block_inferred", []) or []),
            "pk_block_inferred": list(internal.get("pk_block_inferred", []) or []),
            "catalog_fk_revoked": list(internal.get("catalog_fk_revoked", []) or []),
        },
    }
    write_json_atomic(path, payload, sort_keys=True, indent=2)


def save_structure_sidecar(
    schema_json_path: str | Path,
    doc: dict[str, Any],
    *,
    source_schema_hash: str,
    metadata_hash: str,
) -> Path:
    """Atomically write *doc* as the persisted overrides sidecar for *schema_json_path*. The sidecar represents the *resolved* state: it carries the editable surface (``tables``, ``foreign_keys_add``) and the system-managed ``_internal`` envelope (``fk_block_inferred``, ``pk_block_inferred``) used to suppress re-inference on rebuilds. Transient input lists (``foreign_keys_remove``, ``primary_keys_remove``) are not persisted. Stamps ``source_schema_hash``, ``metadata_hash``, and ``applied_at`` so a later rebuild can decide whether to replay."""
    path = structure_sidecar_path(schema_json_path)
    adir = os.path.dirname(os.path.abspath(str(path)))
    with artifact_lock(adir):
        _write_overrides_sidecar_payload(
            path,
            doc,
            source_schema_hash=source_schema_hash,
            metadata_hash=metadata_hash,
        )
    return path


def delete_persisted_structure_artifacts(schema_json_path: str | Path) -> bool:
    """Delete the overrides sidecar and schema cache for *schema_json_path* (internal)."""
    sidecar_path = structure_sidecar_path(schema_json_path)
    cache_path = Path(schema_json_path).expanduser().resolve()
    sidecar_existed = sidecar_path.is_file()
    if sidecar_existed:
        try:
            sidecar_path.unlink()
        except OSError as exc:
            debug(f"[schema.delete_persisted_structure_artifacts] failed to remove {sidecar_path!s}: {exc!r}")
            sidecar_existed = False
    if cache_path.is_file():
        try:
            cache_path.unlink()
        except OSError as exc:
            debug(f"[schema.delete_persisted_structure_artifacts] failed to remove {cache_path!s}: {exc!r}")
    return sidecar_existed


def finalize_with_structure(
    sg: SchemaGraph,
    schema_json_path: str | Path,
    *,
    dialect: Any | None = None,
    previous_schema: SchemaGraph | None = None,
) -> bool:
    """Replay the persisted overrides sidecar onto *sg* if one exists. Called at the tail of every ``build_schema_graph_with_diff`` branch so user-applied metadata, user-added FKs, and inference block lists survive a cache hit, a notes-only refresh, a scope-subset filter, a partial rebuild, and a full rebuild alike. Replay is idempotent so it always runs when a sidecar exists; the sidecar's ``source_schema_hash`` and ``metadata_hash`` are then refreshed to the freshly stamped ``effective_structural_hash`` and :func:`compute_metadata_hash` output. Skipped override entries (unknown tables/columns, malformed FKs, etc.) are emitted via :func:`notify` and stashed on ``sg._last_structure_skipped`` for programmatic inspection. Returns True iff a replay actually executed."""
    emit_construction_phase(SCHEMA_BUILD_PHASE_K)
    sidecar = load_structure_sidecar(schema_json_path)
    if sidecar is None:
        sg._last_structure_skipped = ()
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
        sg._last_structure_skipped = ()
        return False
    debug(f"[schema.finalize_with_structure] replaying sidecar (curr_hash={sg.effective_structural_hash[:16]!r})")
    prev_schema = previous_schema
    if prev_schema is None:
        prev_schema = load_schema_graph_snapshot(str(schema_json_path))
    document: dict[str, Any] = {
        "version": STRUCTURE_DOCUMENT_VERSION,
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
    tables_doc = document.get("tables")
    if isinstance(tables_doc, dict):
        for stale_tname in [t for t in list(tables_doc) if t not in sg.tables]:
            tables_doc.pop(stale_tname, None)
    report = apply_structure_to_graph(
        sg,
        document,
        dialect=dialect,
        strict=True,
        previous_schema=prev_schema,
    )
    sg._last_structure_skipped = report.skipped
    if report.skipped:
        notify(
            f"Schema overrides replay skipped {len(report.skipped)} entr{'y' if len(report.skipped) == 1 else 'ies'}:",
            stage="schema",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            details=(("skipped_count", str(len(report.skipped))),),
        )
        for entry in report.skipped:
            notify(
                f"  - {entry.path}: {entry.reason}",
                stage="schema",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
                details=(("path", entry.path), ("reason", entry.reason)),
            )
    if report.changed_pk_blocks or report.changed_fk_blocks:
        pk_blocked, fk_blocked = load_inference_block_lists(schema_json_path, schema=sg)
        if report.changed_pk_blocks:
            infer_missing_pks_from_profile(sg.tables, dialect=dialect, blocked=pk_blocked)
        if report.changed_fk_blocks:
            run_fk_inference_if_disconnected(sg, blocked=fk_blocked)
        coerce_pk_fk_columns_to_identifier(sg)
        sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
        sg.refresh_schema_stats()
        assign_schema_graph_hashes(sg, schema_context_from_graph(sg), sg.notes_sha256)
    document["foreign_keys_add"] = user_added_fks_dump(sg)
    document["primary_keys_add"] = user_added_pks_dump(sg)
    _stamp_sidecar_provenance(document, sg)
    adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
    with artifact_lock(adir):
        save_structure_sidecar(
            schema_json_path,
            document,
            source_schema_hash=sg.effective_structural_hash,
            metadata_hash=compute_metadata_hash(sg),
        )
        save_schema_to_cache(sg, str(schema_json_path))
    return True


def apply_structure_document(
    sg: SchemaGraph,
    document: Mapping[str, Any],
    *,
    schema_json_path: str,
    dialect: Any | None = None,
    domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
) -> StructureReport:
    """Apply a structure document (or compact tables-keyed edit doc) to *sg* and persist artifacts."""
    if not isinstance(document, Mapping):
        raise ConfigError("structure document must be a JSON object")
    tables = document.get("tables")
    if isinstance(tables, list):
        overrides_payload = structure_document_to_overrides_payload(document)
    else:
        overrides_payload = _validate_structure_edits(dict(document))
    sidecar = load_structure_sidecar(schema_json_path) or {}
    sidecar_internal = sidecar.get("_internal", {}) or {}
    document_internal = overrides_payload.setdefault("_internal", {})
    if not isinstance(document_internal, dict):
        document_internal = {}
        overrides_payload["_internal"] = document_internal
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
    for entry in list(sidecar.get("primary_keys_add", []) or []) + list(
        overrides_payload.get("primary_keys_add", []) or []
    ):
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
    overrides_payload["primary_keys_add"] = merged_pk

    report = apply_structure_to_graph(sg, overrides_payload, dialect=dialect, strict=True)
    sg._last_structure_skipped = report.skipped
    domain_knowledge_entries = report.domain_knowledge_entries
    if report.sensitivity_increased_columns:
        adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
        ratchet = on_sensitivity_classification_change(
            sg,
            report.sensitivity_increased_columns,
            artifacts_dir=adir,
            domain_knowledge=domain_knowledge,
        )
        domain_knowledge_entries = ratchet.domain_knowledge_entries
    overrides_payload["foreign_keys_add"] = user_added_fks_dump(sg)
    overrides_payload["primary_keys_add"] = user_added_pks_dump(sg)

    adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
    with artifact_lock(adir):
        if (
            report.table_edits
            or report.column_edits
            or report.fks_added
            or report.fks_removed
            or report.pks_added
            or report.pks_endorsed
            or report.pks_blocked
            or report.coerced_columns
            or report.collapsed_inferences
        ):
            save_schema_to_cache(sg, str(schema_json_path))
        save_structure_sidecar(
            schema_json_path,
            overrides_payload,
            source_schema_hash=sg.effective_structural_hash,
            metadata_hash=compute_metadata_hash(sg),
        )
    if domain_knowledge_entries is not report.domain_knowledge_entries:
        report = StructureReport(
            table_edits=report.table_edits,
            column_edits=report.column_edits,
            fks_added=report.fks_added,
            fks_removed=report.fks_removed,
            fks_endorsed=report.fks_endorsed,
            pks_added=report.pks_added,
            pks_endorsed=report.pks_endorsed,
            pks_blocked=report.pks_blocked,
            coerced_columns=report.coerced_columns,
            collapsed_inferences=report.collapsed_inferences,
            descriptions_refined=report.descriptions_refined,
            skipped=report.skipped,
            sensitivity_increased_columns=report.sensitivity_increased_columns,
            domain_knowledge_entries=domain_knowledge_entries,
        )
    return report


def apply_structure_from_path(
    sg: SchemaGraph,
    overrides_path: str | Path,
    *,
    schema_json_path: str,
    dialect: Any | None = None,
    domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
) -> StructureReport:
    """Load overrides from *overrides_path*, apply them to *sg*, persist the schema cache, and update the sidecar. The sidecar at ``structure_sidecar_path(schema_json_path)`` is rewritten with the *resolved* state (existing user-added FKs surfaced from the in-memory graph plus the merged ``_internal`` block lists) so the next rebuild can replay without re-reading the user's editor file. The ``source_schema_hash`` and ``metadata_hash`` fields are stamped from the freshly stamped ``effective_structural_hash`` and :func:`compute_metadata_hash` so subsequent cache hits can short-circuit replay when neither structure nor metadata drifted. Overrides carry structure only; descriptions and domain knowledge travel through the space- knowledge surface and are never read from this document. Cache and sidecar writes share a single ``artifact_lock`` critical section."""
    document = load_structure_document_file(overrides_path)
    sidecar = load_structure_sidecar(schema_json_path) or {}
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

    report = apply_structure_to_graph(sg, document, dialect=dialect, strict=True)
    sg._last_structure_skipped = report.skipped
    domain_knowledge_entries = report.domain_knowledge_entries
    if report.sensitivity_increased_columns:
        adir = os.path.dirname(os.path.abspath(str(schema_json_path)))
        ratchet = on_sensitivity_classification_change(
            sg,
            report.sensitivity_increased_columns,
            artifacts_dir=adir,
            domain_knowledge=domain_knowledge,
        )
        domain_knowledge_entries = ratchet.domain_knowledge_entries
    document["foreign_keys_add"] = user_added_fks_dump(sg)
    document["primary_keys_add"] = user_added_pks_dump(sg)

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
            save_schema_to_cache(sg, str(schema_json_path))
        _write_overrides_sidecar_payload(
            structure_sidecar_path(schema_json_path),
            document,
            source_schema_hash=sg.effective_structural_hash,
            metadata_hash=compute_metadata_hash(sg),
        )
    notify_schema_path_health(sg)
    if domain_knowledge_entries is not report.domain_knowledge_entries:
        report = StructureReport(
            table_edits=report.table_edits,
            column_edits=report.column_edits,
            fks_added=report.fks_added,
            fks_endorsed=report.fks_endorsed,
            fks_removed=report.fks_removed,
            pks_added=report.pks_added,
            pks_endorsed=report.pks_endorsed,
            pks_blocked=report.pks_blocked,
            changed_pk_blocks=report.changed_pk_blocks,
            changed_fk_blocks=report.changed_fk_blocks,
            coerced_columns=report.coerced_columns,
            collapsed_inferences=report.collapsed_inferences,
            descriptions_refined=report.descriptions_refined,
            domain_knowledge_refined=report.domain_knowledge_refined,
            domain_knowledge_entries=domain_knowledge_entries,
            sensitivity_increased_columns=report.sensitivity_increased_columns,
            skipped=report.skipped,
        )
    return report


def _merge_fk_layers(
    cached_fks: list[FKEdge],
    fresh_catalog_fks: list[FKEdge],
    *,
    surviving_columns: dict[str, set[str]],
    src_table: str,
) -> tuple[list[FKEdge], list[FKEdge], list[FKEdge], list[FKEdge]]:
    """Combine fresh catalog FKs with cached non-catalog (inferred + user override) FKs. The fresh catalog snapshot wins for catalog-declared edges (``inference_tag is None``); cached non-catalog edges are kept as long as both endpoints still exist in the new graph (``surviving_columns`` maps table name to its surviving column-name set). The function returns ``(merged, dropped_inferred, dropped_user, dropped_catalog)`` where each *dropped* list holds the original :class:`FKEdge` objects so the caller can record provenance in a migration report."""
    new_catalog = [copy.deepcopy(e) for e in fresh_catalog_fks if e.inference_tag is None]
    new_keys = {edge_key(e) for e in new_catalog}
    cached_catalog_keys = {edge_key(e): e for e in cached_fks if e.inference_tag is None}

    merged: list[FKEdge] = list(new_catalog)
    dropped_inferred: list[FKEdge] = []
    dropped_user: list[FKEdge] = []
    dropped_catalog: list[FKEdge] = [edge for key, edge in cached_catalog_keys.items() if key not in new_keys]

    for edge in cached_fks:
        tag = edge.inference_tag
        if tag is None:
            continue
        key = edge_key(edge)
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


def _profile_values_only(
    dialect: Dialect,
    target_sg: SchemaGraph,
    table_names: set[str],
) -> None:
    """Re-profile value samples on *table_names* without LLM classification."""
    if not table_names:
        return
    subset_tables = {n: target_sg.tables[n] for n in table_names if n in target_sg.tables}
    if not subset_tables:
        return
    tmp_sg = SchemaGraph(tables=subset_tables, join_paths_multi={})
    dialect.profile_schema(tmp_sg)


def _profile_subset(
    dialect: Dialect,
    target_sg: SchemaGraph,
    table_names: set[str],
    notes_content: str | None,
) -> None:
    """Run profiling + LLM classifier on a *subset* of tables in *target_sg*, in-place. Builds a temporary :class:`SchemaGraph` containing only the named tables (sharing the same :class:`TableMetadata` objects so mutations propagate to *target_sg*) and asks the dialect to profile it. The same temp graph is fed through the LLM column classifier so only the changed tables incur LLM cost."""
    if not table_names:
        return
    subset_tables = {n: target_sg.tables[n] for n in table_names if n in target_sg.tables}
    if not subset_tables:
        return
    tmp_sg = SchemaGraph(tables=subset_tables, join_paths_multi={})
    dialect.profile_schema(tmp_sg)
    apply_column_roles_llm(
        tmp_sg,
        notes_content=notes_content,
        skip_structural_extraction=True,
        structural_knowledge=target_sg.structural_knowledge,
    )
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
        if RoleOwner.can_overwrite(table.role_owner, RoleOwner.LLM):
            table.role = table_role
            table.role_owner = RoleOwner.LLM
        DescriptionOwner.set_on(table, description, DescriptionOwner.LLM_REFINEMENT)
        for col in table.columns.values():
            if col.name not in column_classifications:
                continue
            role, col_description, _sensitivity = column_classifications[col.name]
            if RoleOwner.can_overwrite(col.role_owner, RoleOwner.LLM):
                col.role = role
                col.role_owner = RoleOwner.LLM
            DescriptionOwner.set_on(col, col_description, DescriptionOwner.LLM_REFINEMENT)


def _descriptions_fingerprint(sg: SchemaGraph, table_names: set[str]) -> tuple[tuple[str, str], ...]:
    """Stable table/column description snapshot for unchanged-table refresh checks."""
    parts: list[tuple[str, str]] = []
    for tname in sorted(table_names):
        table = sg.tables.get(tname)
        if table is None:
            continue
        parts.append((f"table:{tname}", str(table.description or "").strip()))
        for col_name in sorted(table.columns):
            col = table.columns[col_name]
            parts.append((f"col:{tname}.{col_name}", str(col.description or "").strip()))
    return tuple(parts)


def _refresh_existing_descriptions_after_addition(
    cached_sg: SchemaGraph,
    diff: SchemaDiff,
    notes_content: str | None,
    *,
    artifacts_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Full-graph classify, then merge descriptions/roles for tables outside ``diff.changed_table_names()``."""
    changed = diff.changed_table_names()
    unchanged = set(cached_sg.tables) - changed
    if not unchanged:
        return
    before = _descriptions_fingerprint(cached_sg, unchanged)
    try:
        structural = resolve_structural_knowledge_for_schema(
            cached_sg,
            notes_content,
            artifacts_dir=str(artifacts_dir) if artifacts_dir is not None else None,
            extract_knowledge_from_notes=extract_knowledge_from_notes,
        )
        classifications = llm_classify_schema(
            cached_sg,
            notes_content,
            structural_knowledge=structural,
        )
    except Exception as exc:
        debug(f"[schema._refresh_existing_descriptions_after_addition] full-graph classify failed: {exc!r}")
        emit_description_enrichment_failed("schema_migration_refresh", exc)
        return
    _merge_llm_descriptions_and_roles_for_tables(
        cached_sg,
        table_names=unchanged,
        classifications=classifications,
    )
    apply_boolean_coercion_pass(cached_sg)
    assign_column_ops(cached_sg)
    if _descriptions_fingerprint(cached_sg, unchanged) == before:
        emit_description_enrichment_noop("schema_migration_refresh")


def _resync_column_key_flags(sg: SchemaGraph) -> None:
    """Reconcile PK lists and clear stale inference tags so column key properties stay consistent."""
    for _tbl_name, tbl in sg.tables.items():
        tbl.primary_key = [c for c in tbl.primary_key if c in tbl.columns]
        pk_set = set(tbl.primary_key)
        for col_name, col in tbl.columns.items():
            if col_name not in pk_set and col.pk_inference_tag is not None:
                col.pk_inference_tag = None
        kept: list[FKEdge] = []
        for edge in tbl.foreign_keys:
            if edge.src_table != tbl.name:
                kept.append(edge)
                continue
            dst_tbl = sg.tables.get(edge.dst_table)
            if dst_tbl is None:
                continue
            if any(c not in tbl.columns for c in edge.src_cols):
                continue
            if any(c not in dst_tbl.columns for c in edge.dst_cols):
                continue
            kept.append(edge)
        tbl.foreign_keys = kept


def apply_diff(
    cached_sg: SchemaGraph,
    new_sg: SchemaGraph,
    diff: SchemaDiff,
    dialect: Dialect,
    notes_content: str | None = None,
    *,
    schema_json_path: str | Path | None = None,
    refresh_existing_descriptions_on_addition: bool = False,
    fk_remaps: tuple[Any, ...] = (),
    pk_remaps: tuple[Any, ...] = (),
) -> SchemaGraph:
    """Mutate *cached_sg* in place to reflect *diff*, profiling only the affected tables. Profiling/classification is run only on tables that appear in :meth:`SchemaDiff.changed_table_names`; everything else keeps its cached profile and LLM-assigned roles. When *refresh_existing_descriptions_on_addition* is true and the diff adds tables, an additional full-graph classifier pass may refresh descriptions and roles on tables that were otherwise unchanged. Callers update hashes and persist afterward."""
    dropped_catalog_edges: list[FKEdge] = []
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
                cur.frequent_values = []
                cur.value_overlap_sample = []
                cur.min_val = None
                cur.max_val = None
                cur.distinct_count = 0
                cur.distinct_ratio = 0.0
                cur.null_ratio = 0.0
                cur.row_count = 0
                cur.mode_frequency_ratio = 0.0
                cur.semantic_join_neighbors = []
                cur.valid_where_ops = []
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
                dropped_catalog_edges.append(edge)
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
                pk_col_meta = cached_t.columns.get(new_pk)
                if pk_col_meta is not None:
                    pk_col_meta.pk_inference_tag = None
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
            src_t, src_c, dst_t, dst_c = (str(quad[0]), str(quad[1]), str(quad[2]), str(quad[3]))
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

    if fk_remaps:
        apply_fk_remaps_to_graph(cached_sg, fk_remaps)
    if pk_remaps:
        apply_pk_remaps_to_graph(cached_sg, pk_remaps)

    _profile_subset(dialect, cached_sg, diff.changed_table_names(), notes_content)

    if refresh_existing_descriptions_on_addition and diff.added_tables:
        adir = os.path.dirname(os.path.abspath(str(schema_json_path))) if schema_json_path else None
        _refresh_existing_descriptions_after_addition(cached_sg, diff, notes_content, artifacts_dir=adir)

    if schema_json_path is not None:
        migrate_sidecar_for_diff(
            schema_json_path,
            diff,
            fk_remaps=fk_remaps,
            pk_remaps=pk_remaps,
        )
        if dropped_catalog_edges:
            _append_catalog_fk_revoked(schema_json_path, dropped_catalog_edges, sg=cached_sg)
        reconcile_sidecar_against_graph(cached_sg, schema_json_path)

    pk_blocked, fk_blocked = load_inference_block_lists(schema_json_path, schema=cached_sg)
    infer_missing_pks_from_profile(cached_sg.tables, dialect=dialect, blocked=pk_blocked)
    run_fk_inference_if_disconnected(cached_sg, blocked=fk_blocked)
    redact_hidden_sensitivity_profile_values(cached_sg)
    mark_canonical_duplicates(cached_sg)

    coerce_pk_fk_columns_to_identifier(cached_sg)
    skipped: list[OverrideSkip] = []
    collapse_redundant_inferences(cached_sg, skipped)
    _resync_column_key_flags(cached_sg)
    cached_sg.join_paths_multi = recompute_join_paths_multi(cached_sg.tables)
    cached_sg.refresh_schema_stats()

    return cached_sg


def build_schema_graph(
    dialect: Any,
    schema_context: EngineContext,
    notes_content: str | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
    refresh_existing_descriptions_on_addition: bool = False,
) -> SchemaGraph:
    sg, _ = build_schema_graph_with_diff(
        dialect,
        schema_context,
        notes_content,
        log_sink=log_sink,
        refresh_existing_descriptions_on_addition=refresh_existing_descriptions_on_addition,
    )
    return sg


def _live_row_count_fingerprint(dialect: Dialect, sg: SchemaGraph) -> str:
    """Return the dialect live row-count probe, or empty when unavailable."""
    probe = getattr(dialect, "compute_row_count_probe", None)
    if probe is None:
        return ""
    try:
        return str(probe(sg) or "")
    except Exception as exc:
        debug(f"[schema._live_row_count_fingerprint] probe raised, treating as empty: {exc!r}")
        return ""


def _profiling_cache_stale(dialect: Dialect, sg: SchemaGraph) -> bool:
    """Return True when live row counts disagree with the cached graph statistics."""
    live_fp = _live_row_count_fingerprint(dialect, sg)
    if not live_fp:
        return False
    cached_fp = tables_row_count_fingerprint(sg.tables)
    return live_fp != cached_fp


def _maybe_refresh_stale_profiling_cache(
    dialect: Dialect,
    sg: SchemaGraph,
    *,
    schema_json_path: str | Path,
    notes_content: str | None,
    log_sink: Callable[[str], None] | None,
) -> bool:
    """Re-profile *sg* when live row counts drift from the cached statistics."""
    if not _profiling_cache_stale(dialect, sg):
        return False
    sink: Callable[[str], None] = log_sink if log_sink is not None else notify
    sink("  Schema: live data drift — refreshing column statistics...")
    _add_profiling_data(
        dialect,
        sg,
        notes_content=notes_content,
        schema_json_path=schema_json_path,
        log_sink=sink,
    )
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)
    sg.refresh_schema_stats()
    return True


def build_schema_graph_with_diff(
    dialect: Dialect,
    schema_context: EngineContext,
    notes_content: str | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
    refresh_existing_descriptions_on_addition: bool = False,
    force_live_schema_reflect: bool = False,
    trust_bundled_baseline: bool | None = None,
    schema_json_path: str | None = None,
    persist_schema_cache: bool = True,
) -> tuple[SchemaGraph, SchemaDiff | None]:
    """Load or build the schema graph and report the structural diff if a partial rebuild ran. When ``persist_schema_cache`` is false (consumer privilege reflects), the shared owner ``schema_graph.json.gz`` is consulted for comparison only and never rewritten. When the live catalog probe disagrees with a cached owner blob, treat it as a cache miss rather than a fingerprint hit (avoids leaking the full owner schema to consumers)."""
    sink: Callable[[str], None] = log_sink if log_sink is not None else notify
    resolved_path = str(schema_json_path or EngineConfig.SCHEMA_JSON_PATH or "").strip()
    if not resolved_path or os.path.isdir(resolved_path):
        raise ConfigError(
            "schema graph cache path is unset or points at a directory; pass schema_json_path or construct the engine with artifacts_dir"
        )
    schema_json_path = resolved_path

    def _persist_schema_cache(sg: SchemaGraph) -> None:
        if persist_schema_cache:
            save_schema_to_cache(sg, str(schema_json_path))

    debug(f"[schema._build_schema_graph] engine_type={dialect.name}")
    emit_construction_phase(SCHEMA_BUILD_PHASE_A)

    cache_miss_reason: str | None = None
    if force_live_schema_reflect:
        cache_miss_reason = "force_live_schema_reflect"
    if PolicyConfig.REGENERATE_SCHEMA_GRAPH:
        cache_miss_reason = "REGENERATE_SCHEMA_GRAPH enabled"
    elif not os.path.exists(schema_json_path):
        cache_miss_reason = f"cache file missing ({schema_json_path})"

    if not force_live_schema_reflect and not PolicyConfig.REGENERATE_SCHEMA_GRAPH and os.path.exists(schema_json_path):
        debug(f"[schema._build_schema_graph] loading from cache '{schema_json_path}'")
        try:
            d = read_gzip_json(schema_json_path)
        except (
            OSError,
            EOFError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            cache_miss_reason = f"cache unreadable ({exc.__class__.__name__})"
            sink(f"  Schema: cache miss — {cache_miss_reason}.")
            _invalidate_corrupt_schema_cache(schema_json_path, exc, stage="unreadable")
        else:
            incoming_notes_hash = notes_content_sha256(notes_content)
            incoming_scope_hash = scope_hash_fp(schema_context)
            incoming_probe = compute_dialect_probe(dialect, schema_context)
            cached_probe = str(d.get("ddl_probe_hash", "") or "")
            cached_scope = str(d.get("scope_hash", "") or "")
            cached_notes = str(d.get("notes_hash", "") or "")
            cache_viable = True
            trust_baseline = (
                trust_bundled_baseline
                if trust_bundled_baseline is not None
                else PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE
            )
            probe_ok = bool(incoming_probe) and incoming_probe == cached_probe
            if trust_baseline and not probe_ok:
                debug("[schema._build_schema_graph] trust_baseline bypassing probe gate")

            if cache_viable and incoming_scope_hash == cached_scope and (probe_ok or trust_baseline):
                sg = _schema_graph_from_cache_dict(d, schema_json_path)
                if sg is None:
                    cache_viable = False
                    cache_miss_reason = "cached schema graph not viable"
                    sink(f"  Schema: cache miss — {cache_miss_reason}.")
                else:
                    enum_refreshed = _refresh_join_paths_multi_if_enumeration_stale(sg, d, sink=sink)
                    profiling_refreshed = _maybe_refresh_stale_profiling_cache(
                        dialect,
                        sg,
                        schema_json_path=schema_json_path,
                        notes_content=notes_content,
                        log_sink=sink,
                    )

                    if incoming_notes_hash == cached_notes:
                        debug(
                            "[schema._build_schema_graph] cache_hit_via_probe "
                            f"({len(sg.tables)} relations, probe={incoming_probe[:16]!r})"
                        )
                        sg.notes_sha256 = incoming_notes_hash
                        finalize_with_structure(sg, schema_json_path, dialect=dialect)
                        assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                        sg.ddl_probe_hash = incoming_probe
                        if enum_refreshed or profiling_refreshed:
                            _persist_schema_cache(sg)
                        notify_schema_path_health(sg)
                        sink(f"  Schema: cache hit ({len(sg.tables)} relations).")
                        validate_scope_against_graph(sg, schema_context)
                        raise_if_schema_unusable(sg, schema_context)
                        return sg, None

                    debug(
                        "[schema._build_schema_graph] notes_refresh_only "
                        f"(was={cached_notes[:16]!r} incoming={incoming_notes_hash[:16]!r})"
                    )
                    pinned = user_pinned_columns_from_sidecar(schema_json_path)
                    rerun_column_classifier(sg, notes_content, skip_columns=pinned, log_sink=sink)
                    redact_hidden_sensitivity_profile_values(sg)
                    sg.notes_sha256 = incoming_notes_hash
                    finalize_with_structure(sg, schema_json_path, dialect=dialect)
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    sg.ddl_probe_hash = incoming_probe
                    _persist_schema_cache(sg)
                    notify_schema_path_health(sg)
                    sink(f"  Schema: notes-only refresh ({len(sg.tables)} relations).")
                    return sg, None

            cached_descriptor = d.get("scope_descriptor")
            if cache_viable and isinstance(cached_descriptor, dict) and (probe_ok or trust_baseline):
                old_ctx = schema_context_from_descriptor(cached_descriptor)
                change = classify_scope_change(old_ctx, schema_context)
                if change == "subset":
                    debug(
                        f"[schema._build_schema_graph] scope_subset_filter (cached_tables={len(d.get('tables', {}))})"
                    )
                    cached_sg = _schema_graph_from_cache_dict(d, schema_json_path)
                    if cached_sg is None:
                        cache_viable = False
                    else:
                        validate_scope_against_graph(cached_sg, schema_context)
                        sg = filter_schema_graph_by_scope(cached_sg, schema_context)
                        if incoming_notes_hash != cached_notes:
                            pinned = user_pinned_columns_from_sidecar(schema_json_path)
                            rerun_column_classifier(sg, notes_content, skip_columns=pinned, log_sink=sink)
                            redact_hidden_sensitivity_profile_values(sg)
                        sg.notes_sha256 = incoming_notes_hash
                        finalize_with_structure(sg, schema_json_path, dialect=dialect)
                        assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                        sg.ddl_probe_hash = incoming_probe
                        _persist_schema_cache(sg)
                        notify_schema_path_health(sg)
                        sink(f"  Schema: scope-subset filter ({len(sg.tables)} relations).")
                        raise_if_schema_unusable(sg, schema_context)
                        return sg, None
                if change == "superset":
                    debug("[schema._build_schema_graph] scope_superset_partial_rebuild")
                    try:
                        cached_sg = _schema_graph_from_cache_dict(d, schema_json_path)
                        if cached_sg is None:
                            cache_viable = False
                        else:
                            reflect_only = getattr(dialect, "reflect_only", None)
                            if reflect_only is None:
                                new_struct = reflect_schema_graph_for_context(dialect, schema_context)
                            else:
                                new_struct = reflect_only(schema_context)
                                apply_deny_objects_filter(
                                    new_struct,
                                    schema_context,
                                    strict=bool(schema_context.deny_objects),
                                )
                            validate_scope_against_graph(new_struct, schema_context)
                            apply_view_scope_postprocess(new_struct, dialect, schema_context)
                            strip_schema_context_denied_columns(new_struct, schema_context)
                            apply_schema_context_allow_columns(new_struct, schema_context)

                            diff = diff_schemas(cached_sg, new_struct)
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
                            emit_construction_phase(SCHEMA_BUILD_PHASE_B)
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
                            raise_if_schema_unusable(sg, schema_context)
                            finalize_with_structure(sg, schema_json_path, dialect=dialect)
                            assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                            sg.ddl_probe_hash = incoming_probe
                            _persist_schema_cache(sg)
                            notify_schema_path_health(sg)
                            sink(
                                f"  Schema: scope-superset partial rebuild — +{len(diff.added_tables)} "
                                f"tables ({len(sg.tables)} relations)."
                            )
                            return sg, diff
                    except SchemaAccessError:
                        raise
                    except Exception as exc:
                        cache_miss_reason = f"scope superset partial rebuild failed ({exc!r})"
                        sink(f"  Schema: cache miss — {cache_miss_reason}.")
                        debug(f"[schema._build_schema_graph] scope_superset_falling_through_to_rebuild ({exc!r})")
                elif change == "orthogonal":
                    cache_miss_reason = "scope orthogonal to cached descriptor"
                    sink(f"  Schema: cache miss — {cache_miss_reason}.")
                    debug("[schema._build_schema_graph] scope_orthogonal_falling_through_to_rebuild")

            if cache_viable and incoming_probe and (not cached_probe or incoming_probe != cached_probe):
                if not cached_probe:
                    debug(f"[schema._build_schema_graph] probe_missing_reflecting (incoming={incoming_probe[:16]!r})")
                else:
                    debug(
                        "[schema._build_schema_graph] probe_mismatch_reflecting "
                        f"(was={cached_probe[:16]!r} incoming={incoming_probe[:16]!r})"
                    )

                try:
                    tables_cached = {k: table_from_dict(v) for k, v in d["tables"].items()}
                    join_paths_cached = d.get("join_paths_multi") or {}
                    cached_model = dict(d)
                    cached_model["join_paths_multi"] = join_paths_cached
                    cached_model["tables"] = {k: table_to_dict(v) for k, v in tables_cached.items()}
                    cached_sg = SchemaGraph.from_dict(cached_model)
                    ensure_semantic_join_neighbors(cached_sg)

                    new_struct = dialect.reflect_only(schema_context)
                    apply_deny_objects_filter(
                        new_struct,
                        schema_context,
                        strict=bool(schema_context.deny_objects),
                    )
                    validate_scope_against_graph(new_struct, schema_context)
                    apply_view_scope_postprocess(new_struct, dialect, schema_context)
                    strip_schema_context_denied_columns(new_struct, schema_context)
                    apply_schema_context_allow_columns(new_struct, schema_context)

                    diff = diff_schemas(cached_sg, new_struct)
                    debug(
                        "[schema._build_schema_graph] schema_diff: "
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
                        "[schema._build_schema_graph] schema_diff_resolved: "
                        f"+tables={len(diff.added_tables)} -tables={len(diff.dropped_tables)} "
                        f"renames={len(diff.table_renames)} per_table={len(diff.per_table)}"
                    )
                    emit_construction_phase(SCHEMA_BUILD_PHASE_B)
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
                    raise_if_schema_unusable(sg, schema_context)
                    migrate_sidecar_for_diff(schema_json_path, diff)
                    finalize_with_structure(sg, schema_json_path, dialect=dialect)
                    assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                    sg.ddl_probe_hash = incoming_probe
                    _persist_schema_cache(sg)
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
                        f"[schema._build_schema_graph] partial_rebuild_failed: {exc!r}; falling through to full rebuild"
                    )

            if cache_viable:
                try:
                    tables = {k: table_from_dict(v) for k, v in d["tables"].items()}
                except Exception as exc:
                    _invalidate_corrupt_schema_cache(schema_json_path, exc)
                    cache_viable = False

            if cache_viable:
                required_fp_keys = (
                    "structural_hash",
                    "profiling_hash",
                    "scope_hash",
                    "effective_structural_hash",
                    "notes_hash",
                    "semantic_edges_hash",
                )
                if any(k not in d for k in required_fp_keys):
                    debug("[schema._build_schema_graph] cache missing fingerprint fields, removing")
                    os.remove(schema_json_path)
                elif incoming_probe and cached_probe and incoming_probe != cached_probe:
                    cache_miss_reason = (
                        f"ddl probe mismatch (cached={cached_probe[:16]!r} incoming={incoming_probe[:16]!r})"
                    )
                    sink(f"  Schema: cache miss — {cache_miss_reason}.")
                    debug(
                        "[schema._build_schema_graph] fingerprint_hit_blocked_by_probe_mismatch "
                        f"(was={cached_probe[:16]!r} incoming={incoming_probe[:16]!r})"
                    )
                else:
                    rest = structural_hash_fp(tables_structural_payload(tables))
                    prt = profiling_hash_fp(tables_profiling_payload(tables))
                    scp = incoming_scope_hash
                    eff = effective_structural_hash_fp(rest, scp)
                    sem = semantic_edges_fingerprint(tables)
                    if (
                        rest != d["structural_hash"]
                        or prt != d["profiling_hash"]
                        or scp != d["scope_hash"]
                        or eff != d["effective_structural_hash"]
                        or incoming_notes_hash != str(d.get("notes_hash", ""))
                        or sem != str(d.get("semantic_edges_hash", ""))
                    ):
                        cached_hash = d.get("effective_structural_hash", "")
                        _debug_schema_cache_hash_mismatch(
                            schema_json_path=schema_json_path,
                            stamped_hash=cached_hash,
                            json_tables=d["tables"],
                        )
                        debug(
                            "[schema._build_schema_graph] fingerprint mismatch: "
                            f"structural={rest[:16]!r} vs {str(d.get('structural_hash', ''))[:16]!r}",
                        )
                        debug(
                            "[schema._build_schema_graph] fingerprint mismatch detail: "
                            f"profiling={prt[:16]!r} vs {str(d.get('profiling_hash', ''))[:16]!r}; "
                            f"scope={scp[:16]!r} vs {str(d.get('scope_hash', ''))[:16]!r}; "
                            f"effective={eff[:16]!r} vs {str(d.get('effective_structural_hash', ''))[:16]!r}; "
                            f"notes={incoming_notes_hash[:16]!r} vs {str(d.get('notes_hash', ''))[:16]!r}; "
                            f"semantic={sem[:16]!r} vs {str(d.get('semantic_edges_hash', ''))[:16]!r}"
                        )
                        os.remove(schema_json_path)
                    else:
                        debug(f"[schema._build_schema_graph] loaded {len(tables)} relations from cache")

                        sg = _schema_graph_from_cache_dict(d, schema_json_path)
                        if sg is not None:
                            enum_refreshed = _refresh_join_paths_multi_if_enumeration_stale(sg, d, sink=sink)
                            profiling_refreshed = _maybe_refresh_stale_profiling_cache(
                                dialect,
                                sg,
                                schema_json_path=schema_json_path,
                                notes_content=notes_content,
                                log_sink=sink,
                            )
                            finalize_with_structure(sg, schema_json_path, dialect=dialect)
                            assign_schema_graph_hashes(sg, schema_context, incoming_notes_hash)
                            if incoming_probe and not cached_probe:
                                sg.ddl_probe_hash = incoming_probe
                                debug("[schema._build_schema_graph] backfilling ddl_probe_hash to cache")
                            elif incoming_probe:
                                sg.ddl_probe_hash = incoming_probe
                            if enum_refreshed or profiling_refreshed or (incoming_probe and not cached_probe):
                                _persist_schema_cache(sg)
                            notify_schema_path_health(sg)
                            sink(f"  Schema: cache hit ({len(sg.tables)} relations).")
                            validate_scope_against_graph(sg, schema_context)
                            raise_if_schema_unusable(sg, schema_context)
                            return sg, None

            if cache_miss_reason is None and cache_viable:
                if incoming_scope_hash != cached_scope:
                    cache_miss_reason = (
                        f"scope-hash mismatch (cached={cached_scope[:16]!r} incoming={incoming_scope_hash[:16]!r})"
                    )
                else:
                    cache_miss_reason = "cache present but no hit path matched"
                sink(f"  Schema: cache miss — {cache_miss_reason}.")

    sink(
        f"  Schema: building from database — {cache_miss_reason or 'full build'} (this can take a while)...",
    )
    profiling_started = time.monotonic()
    emit_construction_phase(SCHEMA_BUILD_PHASE_C)
    sg = reflect_schema_graph_for_context(dialect, schema_context)
    apply_view_scope_postprocess(sg, dialect, schema_context)
    sink(f"  Schema: reflected {len(sg.tables)} relations; applying scope...")
    debug("[schema._build_schema_graph] validating scope against reflected graph")
    emit_construction_phase(SCHEMA_BUILD_PHASE_D)
    validate_scope_against_graph(sg, schema_context)
    strip_schema_context_denied_columns(sg, schema_context)
    apply_schema_context_allow_columns(sg, schema_context)
    col_total = sum(len(t.columns) for t in sg.tables.values())
    sink(f"  Schema: profiling {col_total} columns...")
    _add_profiling_data(
        dialect,
        sg,
        notes_content=notes_content,
        schema_json_path=schema_json_path,
        log_sink=sink,
    )

    refuse_incompatible_catalog_foreign_keys(sg)

    views_only = sg.tables and all(t.kind == TableKind.VIEW for t in sg.tables.values())
    if views_only and sum(len(x.foreign_keys) for x in sg.tables.values()) == 0:
        infer_view_same_name_key_edges(sg)

    emit_construction_phase(SCHEMA_BUILD_PHASE_H)
    sg.join_paths_multi = recompute_join_paths_multi(sg.tables)

    debug("[schema._build_schema_graph] computing schema stats after profiling")
    sg.refresh_schema_stats()
    sg.notes_sha256 = notes_content_sha256(notes_content)
    raise_if_schema_unusable(sg, schema_context)
    emit_construction_phase(SCHEMA_BUILD_PHASE_I)
    finalize_with_structure(sg, schema_json_path, dialect=dialect)
    assign_schema_graph_hashes(sg, schema_context, sg.notes_sha256)
    sg.ddl_probe_hash = compute_dialect_probe(dialect, schema_context)

    emit_construction_phase(SCHEMA_BUILD_PHASE_J)
    _persist_schema_cache(sg)
    debug("[schema._build_schema_graph] cache saved with profiling data")
    notify_schema_path_health(sg)
    sink(
        f"  Schema: built and cached {len(sg.tables)} relations in {time.monotonic() - profiling_started:.1f}s.",
    )

    return sg, None
