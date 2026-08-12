"""Meta Q&A, interactive_run_once, env/config, initialize/refresh/dispose."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import jsonschema

from ._config import (
    CsvRuntimeConfig,
    DatabricksRuntimeConfig,
    DuckDBRuntimeConfig,
    EngineConfig,
    EngineLimits,
    EngineRuntimeConfig,
    PolicyConfig,
    QSimConfig,
    SeedWarmupConfig,
)
from ._constants import (
    ARTIFACT_DIRECTORY_SEGMENT,
    AZURE_OPENAI_ENV_REQUIRED,
    DIAGNOSTIC_CODE_ARTIFACT_GROWTH,
    DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    EMBEDDED_ENGINE_NAMES,
    ENGINE_STORAGE_SLUG_MAX_CHARS,
    FEDERATION_MIGRATION_MAP_FILENAME,
    FILE_ENGINE_NAMES,
    MASTER_AETHERSPACE_NAME,
    META_ANSWER_FORMAT_VERSION,
    META_ANSWERS_FILENAME,
    META_DEFAULT_SOURCE_ID,
    MIGRATION_MAP_ACTION_ABORT,
    MIGRATION_MAP_FILENAME,
    OPENAI_ENV_REQUIRED,
    SCHEMA_CONTEXT_CACHE_NAME,
    SCHEMA_CONTEXT_CACHE_VERSION,
    SCHEMA_CONTEXT_CACHED_DDL,
    SCHEMA_CONTEXT_CACHED_NOTES,
    SESSION_KIND_ERROR,
    SESSION_KIND_META,
    SIMULATION_CACHE_EXACT_FILENAMES,
    SIMULATION_CACHE_GLOB_PATTERNS,
    STRUCTURE_APPLIED_TIMESTAMP_FORMAT,
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    TEMPLATE_STORE_SEGMENT,
    TOML_ENGINE_FIELD_MAPS,
    TOML_SECTION_TO_ENGINE,
)
from ._constants_runtime import (
    META_DOMAIN_KNOWLEDGE_SYSTEM,
    META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE,
    META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND,
    META_KNOWLEDGE_ANSWER_SCHEMA,
    META_SCHEMA_AND_KNOWLEDGE_ANSWER_SCHEMA,
    META_SCHEMA_AND_KNOWLEDGE_SYSTEM,
    META_SCHEMA_ANSWER_SCHEMA,
    META_SCHEMA_CATALOG_PROMPT_KEY_ORDER,
    META_SCHEMA_CATALOG_REPAIR_PROMPT_KEY_ORDER,
    META_SCHEMA_CATALOG_SYSTEM,
)
from ._contracts_base import (
    ConfigError,
    DatabaseConnectionError,
    DataQualityReport,
    Diagnostic,
    DiagnosticSeverity,
    DomainKnowledgeEntry,
    DomainKnowledgeHolder,
    EngineContext,
    EngineIdentity,
    FederationConfigError,
    FederationContext,
    FederationTopologyReport,
    MigrationPendingError,
    MigrationReport,
    MigrationTier,
    RefreshReport,
    SchemaRole,
)
from ._contracts_core import (
    AetherEngineInitResult,
    AetherFederationInitResult,
    GenerationPath,
    InteractiveChoicePort,
    LLMConfig,
    MigrationPreview,
    QuestionFormStorage,
    QuestionRoute,
    RefinementContext,
    RefinementRetry,
    RephraseHint,
    RuntimeConfig,
    SessionError,
    SessionOutcome,
    SessionStep,
)
from ._contracts_schema import (
    FederationManifest,
    FederationMappings,
    QSimSummary,
    SchemaGraph,
    SeedWarmupSummary,
    SourceRuntime,
)
from ._data_quality import (
    parse_source_selections,
    validate_upload_sources,
)
from ._dialect import (
    DialectRegistry,
)
from ._federation_compose import (
    compose_composite_graph,
    composite_physical_member_refs,
    composite_schema_payload_counts,
    scrub_federation_member_description_source_tokens,
    validate_cross_source_keys_on_graph,
)
from ._federation_execute import (
    apply_federation_migration_map,
    archive_federation_migration_map_file,
    assert_federation_member_graph_roster_complete,
    cleanup_abandoned_federation_spill_directories,
    clear_federation_plan_templates,
    compute_federation_storage_dir,
    detect_broken_cross_source_joins,
    detect_federation_topology_change,
    federation_composite_migration_tier,
    federation_source_artifacts_dir,
    load_federation_composite_graph,
    load_federation_member_graphs,
    load_federation_migration_map,
    mappings_replay_matches,
    persist_federation_tree,
    probe_federation_member_connections,
    prune_cross_source_joins,
    prune_federation_aliases,
    prune_federation_mappings,
    prune_federation_plan_templates_on_drift,
    purge_departed_federation_member_trees,
    reconcile_authored_declaration_for_members,
    reconcile_federation_member_graphs,
    reconcile_federation_topology,
    recorded_federation_source_ids,
    validate_federation_file_members,
    validate_federation_migration_map,
)
from ._federation_manifest import (
    build_federation_manifest_from_members,
    build_federation_migration_map_document,
    cached_or_suggest_cross_source_mappings,
    federation_artifact_paths,
    federation_members_mapping,
    federation_residual_column_headers,
    intersect_member_database_feature_capabilities,
    load_federation_declaration_from_path,
    member_graphs_from_engines,
    raise_if_descriptions_name_federation_sources,
    raise_if_member_notes_name_federation_sources,
    stamp_federation_member_graph,
    validate_manifest_cross_source_joins,
)
from ._llm_provider import (
    LLMProvider,
    MockProvider,
)
from ._main_interactive import MainInteractiveOps
from ._main_spaces import MainSpaceOps
from ._pipeline_execute import handle_direct_sql_reuse, try_federation_plan_inplace_reuse
from ._pipeline_generate import load_pipeline_resources, match_question_level_template_reuse
from ._schema_finalize import (
    build_schema_graph_with_diff,
    finalize_with_structure,
)
from ._schema_graph import (
    assign_schema_graph_hashes,
    classify_migration_tier,
    diff_schemas,
    effective_execution_visible_tables,
    load_schema_graph_snapshot,
    raise_if_schema_unusable,
    subset_schema_graph_for_visible_tables,
    try_rename_migration_plan,
    upgrade_artifacts_schema_graph_id,
)
from ._schema_profile import (
    llm_classify_schema,
)
from ._templates import TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    active_domain_knowledge,
    active_domain_knowledge_digest,
    active_engine_identity,
    bind_construction_orphan_identity,
    debug,
    emit_session_refusal_diagnostic,
    format_versions_match,
    invalid_input,
    normalize_question,
    note_interactive_turn,
    notes_content_from_context,
    notify,
    pop_engine_identity,
    print_rephrase_hint,
    progress,
    prompt,
    prompt_json,
    push_engine_identity,
    refusal_user_text_for_code,
    release_construction_orphan_identity,
    scope_hash_fp,
    stable_json,
    terminated,
)
from ._utils_artifacts import (
    artifact_lock,
    artifact_manifest_incompatible_with_package,
    detect_legacy_artifacts,
    load_runtime_config,
    read_artifact_manifest,
    unregister_dialect_live_handles,
    warn_if_artifacts_dir_not_local,
    wipe_filenames,
    wipe_globs,
    wipe_versioned_artifacts,
)
from ._utils_intent import (
    normalize_question_via_llm,
    validate_question,
)


class MainInitOps:
    """Meta Q&A, interactive_run_once, env/config, initialize/refresh/dispose."""

    @staticmethod
    def _owner_has_federation(owner: Any | None) -> bool:
        return owner is not None and getattr(owner, "_federation_manifest", None) is not None

    @staticmethod
    def filter_domain_knowledge_for_visibility(
        entries: Sequence[DomainKnowledgeEntry],
        *,
        visible_table_names: set[str] | None,
        all_schema_table_names: set[str] | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Drop DK entries keyed to schema tables outside the caller's visible table set."""
        return MainSpaceOps.filter_domain_knowledge_for_visibility(
            entries,
            visible_table_names=visible_table_names,
            all_schema_table_names=all_schema_table_names,
        )

    @staticmethod
    def build_meta_schema_dump(
        schema: SchemaGraph,
        *,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        space_tables: set[str] | None = None,
        exclude_restricted: bool = True,
        table_descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build a filtered schema grounding dump for schema_catalog answers. Optional *table_descriptions* / *column_descriptions* (space overlays) replace master-graph prose so metadata answers stay inside the active space view."""
        table_desc_overlay = {
            str(k): str(v).strip() for k, v in (table_descriptions or {}).items() if str(v or "").strip()
        }
        col_desc_overlay = {
            str(k): str(v).strip() for k, v in (column_descriptions or {}).items() if str(v or "").strip()
        }
        tables_out: list[dict[str, Any]] = []
        columns_per_table: dict[str, int] = {}
        tables_per_member: dict[str, int] = {}
        member_table_counts: dict[str, int] = {}
        relationships: list[dict[str, str]] = []
        seen_rel: set[tuple[str, str, str]] = set()
        total_columns = 0
        for table_name in sorted(schema.tables):
            if space_tables is not None and table_name not in space_tables:
                continue
            if not MainSpaceOps.table_allowed_for_visibility(table_name, schema, scope_ctx, visible_objects):
                continue
            tbl = schema.tables[table_name]
            source_id = str(tbl.source_id or "").strip() or META_DEFAULT_SOURCE_ID
            cols_out: list[dict[str, Any]] = []
            for col_name in sorted(tbl.columns):
                col = tbl.columns[col_name]
                if not MainSpaceOps.column_allowed_for_visibility(
                    table_name,
                    col_name,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=exclude_restricted,
                    col=col,
                ):
                    continue
                fk_target = None
                if col.is_foreign_key and col.fk_target is not None:
                    fk_target = f"{col.fk_target[0]}.{col.fk_target[1]}"
                qc = f"{table_name}.{col_name}"
                cols_out.append(
                    {
                        "name": col_name,
                        "data_type": str(col.data_type or ""),
                        "value_type": str(col.value_type or ""),
                        "role": str(col.role or ""),
                        "description": col_desc_overlay.get(qc) or str(col.description or ""),
                        "is_primary_key": bool(col.is_primary_key),
                        "is_foreign_key": bool(col.is_foreign_key),
                        "fk_target": fk_target,
                    }
                )
            if not cols_out:
                continue
            tables_out.append(
                {
                    "name": table_name,
                    "source_id": source_id,
                    "description": table_desc_overlay.get(table_name) or str(tbl.description or ""),
                    "columns": cols_out,
                    "primary_key": list(tbl.primary_key or []),
                    "foreign_keys": [
                        {
                            "src_cols": list(fk.src_cols),
                            "dst_table": fk.dst_table,
                            "dst_cols": list(fk.dst_cols),
                        }
                        for fk in (tbl.foreign_keys or [])
                    ],
                }
            )
            columns_per_table[table_name] = len(cols_out)
            total_columns += len(cols_out)
            member_table_counts[source_id] = member_table_counts.get(source_id, 0) + 1
            for fk in tbl.foreign_keys or []:
                for src_c, dst_c in zip(fk.src_cols, fk.dst_cols, strict=False):
                    left = f"{fk.src_table}.{src_c}"
                    right = f"{fk.dst_table}.{dst_c}"
                    kind = "semantic" if str(fk.join_kind or "").lower() == "semantic" else "fk"
                    key = (left, right, kind)
                    if key in seen_rel:
                        continue
                    seen_rel.add(key)
                    relationships.append({"left": left, "right": right, "kind": kind})
            for col_name, col in tbl.columns.items():
                if not MainSpaceOps.column_allowed_for_visibility(
                    table_name,
                    col_name,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=exclude_restricted,
                    col=col,
                ):
                    continue
                for nb_table, nb_col in col.semantic_join_neighbors or ():
                    left = f"{table_name}.{col_name}"
                    right = f"{nb_table}.{nb_col}"
                    key = (left, right, "semantic")
                    if key in seen_rel:
                        continue
                    seen_rel.add(key)
                    relationships.append({"left": left, "right": right, "kind": "semantic"})
        tables_per_member = dict(sorted(member_table_counts.items()))
        members = [{"source_id": sid, "table_count": ct} for sid, ct in tables_per_member.items()]
        if not members:
            members = [{"source_id": META_DEFAULT_SOURCE_ID, "table_count": 0}]
            tables_per_member = {META_DEFAULT_SOURCE_ID: 0}
        return {
            "inventory": {
                "table_count": len(tables_out),
                "column_count": total_columns,
                "member_count": len(tables_per_member),
                "columns_per_table": columns_per_table,
                "tables_per_member": tables_per_member,
            },
            "members": members,
            "tables": tables_out,
            "relationships": relationships,
        }

    @staticmethod
    def _meta_terminal_error_step(*, detail_code: str, last_error: str | None = None) -> SessionStep:
        detail = last_error or detail_code
        return SessionStep(
            done=True,
            prompt=None,
            kind=SESSION_KIND_ERROR,
            sql=None,
            answer=None,
            error=SessionError(code=SessionOutcome.INTERNAL_ERROR, detail_code=detail_code),
            diagnostics=(
                Diagnostic(
                    stage="meta",
                    level=DiagnosticSeverity.ERROR,
                    code=detail_code,
                    message=detail,
                    phase="meta",
                ),
            ),
        )

    @staticmethod
    def _meta_insufficient_knowledge_step() -> SessionStep:
        """Return a terminal meta refusal when schema or glossary cannot answer."""
        return SessionStep(
            done=True,
            prompt=None,
            kind=SESSION_KIND_META,
            sql=None,
            data=None,
            answer=None,
            diagnostics=(
                Diagnostic(
                    stage="meta",
                    level=DiagnosticSeverity.INFO,
                    code=DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
                    message="",
                    phase="meta",
                ),
            ),
            intent_summary=None,
            semantic_warnings=(),
            error=SessionError(
                code=SessionOutcome.INSUFFICIENT_KNOWLEDGE,
                detail_code=DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
            ),
            parameters=(),
        )

    @staticmethod
    def validate_meta_schema_answer(answer: dict[str, Any], dump: dict[str, Any]) -> None:
        """Validate a schema_catalog LLM answer against JSON Schema and dump grounding rules."""
        try:
            jsonschema.validate(instance=answer, schema=META_SCHEMA_ANSWER_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"meta schema answer failed JSON Schema: {exc.message}") from exc
        if answer.get("response_kind") == META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND:
            return
        if answer.get("response_kind") != "schema_catalog":
            raise ValueError("response_kind must be schema_catalog")
        inventory = dump.get("inventory") or {}
        dump_tables = {str(t["name"]): t for t in dump.get("tables") or [] if isinstance(t, dict)}
        dump_members = {str(m["source_id"]) for m in dump.get("members") or [] if isinstance(m, dict)}
        dump_rels = {
            (str(r.get("left")), str(r.get("right")), str(r.get("kind")))
            for r in dump.get("relationships") or []
            if isinstance(r, dict)
        }
        for tbl in answer.get("tables") or []:
            name = str(tbl.get("name") or "")
            if name not in dump_tables:
                raise ValueError(f"invented table not in schema dump: {name}")
            dump_tbl = dump_tables[name]
            if str(tbl.get("source_id") or "") != str(dump_tbl.get("source_id") or ""):
                raise ValueError(f"source_id mismatch for table {name}")
            dump_cols = {str(c["name"]): c for c in dump_tbl.get("columns") or []}
            for col in tbl.get("columns") or []:
                cname = str(col.get("name") or "")
                if cname not in dump_cols:
                    raise ValueError(f"invented column not in schema dump: {name}.{cname}")
        for rel in answer.get("relationships") or []:
            kind = str(rel.get("kind") or "")
            if kind not in ("fk", "semantic"):
                raise ValueError(f"invalid relationship kind: {kind}")
            left = str(rel.get("left") or "")
            right = str(rel.get("right") or "")
            if (left, right, kind) not in dump_rels and (right, left, kind) not in dump_rels:
                ok_endpoints = False
                for dl, dr, dk in dump_rels:
                    if dk != kind:
                        continue
                    if {dl, dr} == {left, right}:
                        ok_endpoints = True
                        break
                if not ok_endpoints:
                    raise ValueError(f"invented relationship not in schema dump: {left}->{right}")
        counts = answer.get("counts") or {}
        if counts.get("tables") is not None and int(counts["tables"]) != int(inventory.get("table_count") or 0):
            raise ValueError("counts.tables must equal inventory.table_count")
        if counts.get("columns") is not None and int(counts["columns"]) != int(inventory.get("column_count") or 0):
            raise ValueError("counts.columns must equal inventory.column_count")
        if counts.get("members") is not None and int(counts["members"]) != int(inventory.get("member_count") or 0):
            raise ValueError("counts.members must equal inventory.member_count")
        cit = counts.get("columns_in_table")
        if isinstance(cit, dict):
            tname = str(cit.get("table") or "")
            if tname not in (inventory.get("columns_per_table") or {}):
                raise ValueError(f"columns_in_table.table not in dump: {tname}")
            if int(cit.get("columns") or -1) != int((inventory.get("columns_per_table") or {})[tname]):
                raise ValueError("columns_in_table.columns must equal inventory.columns_per_table")
        tim = counts.get("tables_in_member")
        if isinstance(tim, dict):
            sid = str(tim.get("source_id") or "")
            if sid not in dump_members:
                raise ValueError(f"tables_in_member.source_id not in dump: {sid}")
            if int(tim.get("tables") or -1) != int((inventory.get("tables_per_member") or {}).get(sid, -2)):
                raise ValueError("tables_in_member.tables must equal inventory.tables_per_member")

    @staticmethod
    def format_meta_schema_message(answer: dict[str, Any]) -> str:
        """Render a schema_catalog answer as deterministic plain text."""
        lines: list[str] = [str(answer.get("headline") or "").strip()]
        counts = answer.get("counts") or {}
        count_lines: list[str] = []
        if counts.get("tables") is not None:
            count_lines.append(f"tables: {counts['tables']}")
        if counts.get("columns") is not None:
            count_lines.append(f"columns: {counts['columns']}")
        if counts.get("members") is not None:
            count_lines.append(f"members: {counts['members']}")
        cit = counts.get("columns_in_table")
        if isinstance(cit, dict):
            count_lines.append(f"columns in {cit.get('table')}: {cit.get('columns')}")
        tim = counts.get("tables_in_member")
        if isinstance(tim, dict):
            count_lines.append(f"tables in {tim.get('source_id')}: {tim.get('tables')}")
        if count_lines:
            lines.append("")
            lines.extend(count_lines)
        tables = answer.get("tables") or []
        if tables:
            lines.append("")
            lines.append("tables:")
            for tbl in tables:
                lines.append(f"- {tbl.get('name')} ({tbl.get('source_id')}): {tbl.get('description') or ''}".rstrip())
                for col in tbl.get("columns") or []:
                    lines.append(
                        f"  - {col.get('name')} {col.get('data_type')} "
                        f"[{col.get('role')}] {col.get('description') or ''}".rstrip()
                    )
        rels = answer.get("relationships") or []
        if rels:
            lines.append("")
            lines.append("relationships:")
            for rel in rels:
                lines.append(f"- {rel.get('left')} -> {rel.get('right')} ({rel.get('kind')})")
        notes = [str(n) for n in (answer.get("notes") or []) if str(n).strip()]
        if notes:
            lines.append("")
            lines.append("notes:")
            for note in notes:
                lines.append(f"- {note}")
        return "\n".join(lines).strip()

    @staticmethod
    def _meta_cache_space_name(space_overlay: Any) -> str:
        if space_overlay is None:
            return ""
        if isinstance(space_overlay, str):
            return space_overlay.strip()
        name = getattr(space_overlay, "name", None) or getattr(space_overlay, "space_name", None)
        return str(name or "").strip()

    @staticmethod
    def _meta_cache_federation_id(owner: Any, schema: SchemaGraph | None) -> str:
        if schema is not None and isinstance(schema.federation_membership, dict):
            fed = str(schema.federation_membership.get("federation_id") or "").strip()
            if fed:
                return fed
        manifest = getattr(owner, "_federation_manifest", None) if owner is not None else None
        if manifest is not None:
            fed = str(getattr(manifest, "federation_id", "") or "").strip()
            if fed:
                return fed
        return ""

    @staticmethod
    def _meta_cache_schema_graph_id(schema: SchemaGraph | None) -> str:
        if schema is None:
            return ""
        return str(getattr(schema, "schema_graph_id", "") or "").strip()

    @staticmethod
    def _meta_cache_dk_digest(owner: Any) -> str:
        digest = active_domain_knowledge_digest()
        if digest:
            return digest
        holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
        if isinstance(holder, DomainKnowledgeHolder):
            return str(holder.digest() or "").strip()
        return ""

    @staticmethod
    def _meta_answer_visibility_fingerprint(
        *,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        space_tables: set[str] | None,
    ) -> str:
        """Fingerprint the caller's effective schema visibility for metadata cache keys."""
        parts: list[str] = []
        if scope_ctx is not None:
            parts.append(scope_hash_fp(scope_ctx))
        else:
            parts.append("")
        if visible_objects is not None:
            parts.append(MainInitOps.credential_visibility_fingerprint(visible_objects))
        else:
            parts.append("")
        if space_tables is not None:
            parts.append(hashlib.sha256(",".join(sorted(space_tables)).encode("utf-8")).hexdigest()[:16])
        else:
            parts.append("")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _meta_forbidden_schema_identifiers(
        schema: SchemaGraph,
        *,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        space_tables: set[str] | None,
    ) -> frozenset[str]:
        """Return table and table.column names that must not appear in a cached metadata answer."""
        forbidden: set[str] = set()
        for table_name in schema.tables:
            if space_tables is not None and table_name not in space_tables:
                forbidden.add(table_name)
                for col_name in schema.tables[table_name].columns:
                    forbidden.add(f"{table_name}.{col_name}")
                continue
            if not MainSpaceOps.table_allowed_for_visibility(table_name, schema, scope_ctx, visible_objects):
                forbidden.add(table_name)
                for col_name in schema.tables[table_name].columns:
                    forbidden.add(f"{table_name}.{col_name}")
                continue
            tbl = schema.tables[table_name]
            for col_name in tbl.columns:
                if not MainSpaceOps.column_allowed_for_visibility(
                    table_name,
                    col_name,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=True,
                    col=tbl.columns[col_name],
                ):
                    forbidden.add(f"{table_name}.{col_name}")
        return frozenset(forbidden)

    @staticmethod
    def _meta_message_references_forbidden_identifiers(message: str, forbidden: frozenset[str]) -> bool:
        """Return True when *message* mentions a schema identifier outside the caller's scope."""
        if not forbidden or not message:
            return False
        lower = message.lower()
        for ident in forbidden:
            pattern = r"(?<![\w.])" + re.escape(ident.lower()) + r"(?![\w.])"
            if re.search(pattern, lower):
                return True
        return False

    @staticmethod
    def _meta_validate_cached_step(
        step: SessionStep,
        *,
        route: QuestionRoute,
        schema: SchemaGraph | None,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        space_tables: set[str] | None,
        space_snapshot: dict[str, Any] | None,
        pipeline_session: Any = None,
        schema_payload: dict[str, Any] | None = None,
    ) -> SessionStep | None:
        """Re-check a cached metadata answer against the caller's current visibility."""
        if not isinstance(step.answer, str) or not step.answer.strip():
            return None
        if route == QuestionRoute.SCHEMA_CATALOG and schema is not None:
            table_descriptions, column_descriptions = MainInitOps._space_description_overlays(
                space_snapshot, pipeline_session=pipeline_session
            )
            dump = MainInitOps.build_meta_schema_dump(
                schema,
                scope_ctx=scope_ctx,
                visible_objects=visible_objects,
                space_tables=space_tables,
                exclude_restricted=True,
                table_descriptions=table_descriptions,
                column_descriptions=column_descriptions,
            )
            if not isinstance(schema_payload, dict):
                return None
            try:
                MainInitOps.validate_meta_schema_answer(schema_payload, dump)
            except ValueError:
                return None
            return replace(step, answer=MainInitOps.format_meta_schema_message(schema_payload))
        if schema is not None and route in (
            QuestionRoute.SCHEMA_AND_KNOWLEDGE,
            QuestionRoute.DOMAIN_KNOWLEDGE,
        ):
            forbidden = MainInitOps._meta_forbidden_schema_identifiers(
                schema,
                scope_ctx=scope_ctx,
                visible_objects=visible_objects,
                space_tables=space_tables,
            )
            if MainInitOps._meta_message_references_forbidden_identifiers(step.answer, forbidden):
                return None
        return step

    @staticmethod
    def meta_answer_cache_key(
        *,
        schema_graph_id: str,
        federation_id: str,
        space_name: str,
        domain_knowledge_digest: str,
        corrected_question: str,
        route: str,
        visibility_fingerprint: str,
    ) -> str:
        """Return the sha256 hex cache key for a metadata answer."""
        material = "|".join(
            (
                str(schema_graph_id or ""),
                str(federation_id or ""),
                str(space_name or ""),
                str(domain_knowledge_digest or ""),
                str(corrected_question or ""),
                str(route or ""),
                str(visibility_fingerprint or ""),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _meta_answers_path(artifacts_dir: str) -> str:
        return os.path.join(os.path.abspath(artifacts_dir), META_ANSWERS_FILENAME)

    @staticmethod
    def load_meta_answer_cache(artifacts_dir: str | None) -> dict[str, Any]:
        """Load ``meta_answers.json`` or return an empty versioned document."""
        empty: dict[str, Any] = {"meta_answer_format_version": META_ANSWER_FORMAT_VERSION, "entries": {}}
        if not artifacts_dir:
            return empty
        path = MainInitOps._meta_answers_path(artifacts_dir)
        if not os.path.isfile(path):
            return empty
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return empty
        if not isinstance(payload, dict):
            return empty
        if not format_versions_match(payload.get("meta_answer_format_version"), META_ANSWER_FORMAT_VERSION):
            return empty
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return empty
        return {"meta_answer_format_version": META_ANSWER_FORMAT_VERSION, "entries": dict(entries)}

    @staticmethod
    def save_meta_answer_cache(artifacts_dir: str | None, cache: dict[str, Any]) -> None:
        """Persist ``meta_answers.json`` when *artifacts_dir* is set."""
        if not artifacts_dir:
            return
        payload = {
            "meta_answer_format_version": META_ANSWER_FORMAT_VERSION,
            "entries": dict(cache.get("entries") or {}),
        }
        MainSpaceOps.write_json_atomic(MainInitOps._meta_answers_path(artifacts_dir), payload)

    @staticmethod
    def _lookup_meta_answer_cache(
        artifacts_dir: str | None,
        *,
        schema: SchemaGraph | None,
        owner: Any,
        space_overlay: Any,
        corrected: str,
        route: QuestionRoute,
        visibility_fingerprint: str,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        space_tables: set[str] | None = None,
        space_snapshot: dict[str, Any] | None = None,
        pipeline_session: Any = None,
    ) -> SessionStep | None:
        if not artifacts_dir:
            return None
        key = MainInitOps.meta_answer_cache_key(
            schema_graph_id=MainInitOps._meta_cache_schema_graph_id(schema),
            federation_id=MainInitOps._meta_cache_federation_id(owner, schema),
            space_name=MainInitOps._meta_cache_space_name(space_overlay),
            domain_knowledge_digest=MainInitOps._meta_cache_dk_digest(owner),
            corrected_question=corrected,
            route=route.value,
            visibility_fingerprint=visibility_fingerprint,
        )
        cache = MainInitOps.load_meta_answer_cache(artifacts_dir)
        entry = (cache.get("entries") or {}).get(key)
        if not isinstance(entry, dict):
            return None
        message = entry.get("answer")
        kind = str(entry.get("kind") or SESSION_KIND_META)
        schema_payload = entry.get("schema_payload")
        if not isinstance(message, str) or not message.strip():
            return None
        if schema_payload is not None and not isinstance(schema_payload, dict):
            return None
        candidate = SessionStep(
            done=True,
            prompt=None,
            kind=kind,
            sql=None,
            data=None,
            answer=message,
            diagnostics=(),
            intent_summary=None,
            semantic_warnings=(),
            error=None,
            parameters=(),
        )
        validated = MainInitOps._meta_validate_cached_step(
            candidate,
            route=route,
            schema=schema,
            scope_ctx=scope_ctx,
            visible_objects=visible_objects,
            space_tables=space_tables,
            space_snapshot=space_snapshot,
            pipeline_session=pipeline_session,
            schema_payload=dict(schema_payload) if isinstance(schema_payload, dict) else None,
        )
        if validated is None:
            return None
        notify("Metadata cache hit", stage="meta", code="meta.cache.hit", level="info")
        return validated

    @staticmethod
    def _store_meta_answer_cache(
        artifacts_dir: str | None,
        *,
        schema: SchemaGraph | None,
        owner: Any,
        space_overlay: Any,
        corrected: str,
        route: QuestionRoute,
        visibility_fingerprint: str,
        step: SessionStep,
        schema_payload: dict[str, Any] | None = None,
    ) -> None:
        if not artifacts_dir or step.kind != SESSION_KIND_META or step.error is not None:
            return
        if not isinstance(step.answer, str) or not step.answer.strip():
            return
        key = MainInitOps.meta_answer_cache_key(
            schema_graph_id=MainInitOps._meta_cache_schema_graph_id(schema),
            federation_id=MainInitOps._meta_cache_federation_id(owner, schema),
            space_name=MainInitOps._meta_cache_space_name(space_overlay),
            domain_knowledge_digest=MainInitOps._meta_cache_dk_digest(owner),
            corrected_question=corrected,
            route=route.value,
            visibility_fingerprint=visibility_fingerprint,
        )
        cache = MainInitOps.load_meta_answer_cache(artifacts_dir)
        entries = dict(cache.get("entries") or {})
        stored: dict[str, Any] = {
            "answer": step.answer,
            "kind": step.kind,
        }
        if isinstance(schema_payload, dict):
            stored["schema_payload"] = dict(schema_payload)
        entries[key] = stored
        cache["entries"] = entries
        MainInitOps.save_meta_answer_cache(artifacts_dir, cache)

    @staticmethod
    def _resolve_active_domain_knowledge_entries(owner: Any) -> tuple[DomainKnowledgeEntry, ...]:
        """Return scoped domain knowledge, falling back to the owner's holder."""
        active = active_domain_knowledge()
        if active:
            return active
        holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
        if isinstance(holder, DomainKnowledgeHolder):
            return holder.entries()
        return ()

    @staticmethod
    def meta_visibility_knobs(
        owner: Any,
        schema: SchemaGraph | None,
        space_overlay: Any,
        *,
        pipeline_session: Any = None,
    ) -> tuple[
        EngineContext | FederationContext | None,
        frozenset[str] | None,
        set[str] | None,
        dict[str, Any] | None,
    ]:
        """Resolve scope context, credential visible set, optional space table subset, and space snapshot."""
        scope_ctx: EngineContext | FederationContext | None = None
        if owner is not None:
            try:
                resolved = MainSpaceOps.resolve_preview_scope_context(owner)
                if isinstance(resolved, (EngineContext, FederationContext)):
                    scope_ctx = resolved
            except (AttributeError, TypeError, ConfigError):
                scope_ctx = None
        visible = getattr(owner, "_consumer_visible_objects", None) if owner is not None else None
        if isinstance(visible, frozenset):
            pass
        elif isinstance(visible, (set, list, tuple)):
            visible = frozenset(str(v) for v in visible)
        else:
            visible = None
        space_tables: set[str] | None = None
        space_snapshot: dict[str, Any] | None = None
        raw_space = ""
        if space_overlay is not None:
            raw_space = str(space_overlay).strip()
        elif pipeline_session is not None:
            raw_space = str(getattr(pipeline_session, "_space_name", "") or "").strip()
        if raw_space and schema is not None:
            space_uid = ""
            try:
                space_uid = MainSpaceOps.validate_space_uid(raw_space)
            except ValueError:
                space_uid = raw_space
            if space_uid and space_uid.lower() != MASTER_AETHERSPACE_NAME:
                art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
                if art is not None:
                    snap = MainSpaceOps.load_aetherspace_snapshot(str(art), space_uid)
                    if isinstance(snap, dict):
                        space_snapshot = snap
                        raw_tables = snap.get("tables")
                        if isinstance(raw_tables, (list, tuple)):
                            space_tables = {str(t) for t in raw_tables}
        if space_tables is None and pipeline_session is not None:
            sess_tables = getattr(pipeline_session, "_space_tables", None)
            if isinstance(sess_tables, (set, frozenset, list, tuple)) and sess_tables:
                space_tables = {str(t) for t in sess_tables}
        return scope_ctx, visible, space_tables, space_snapshot

    @staticmethod
    def _space_description_overlays(
        space_snapshot: dict[str, Any] | None,
        *,
        pipeline_session: Any = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Collect table/column description overlays from a space snapshot and session overlay."""
        table_descriptions, column_descriptions = MainSpaceOps.description_overlays_from_snapshot(space_snapshot)
        session_overlay = None
        if pipeline_session is not None:
            session_overlay = getattr(pipeline_session, "space_description_overlay", None)
        if isinstance(session_overlay, Mapping):
            overlay_tables, overlay_cols = MainSpaceOps.description_overlays_from_snapshot(session_overlay)
            table_descriptions = {**table_descriptions, **overlay_tables}
            column_descriptions = {**column_descriptions, **overlay_cols}
        return table_descriptions, column_descriptions

    @staticmethod
    def _answer_domain_knowledge_question(
        owner: Any,
        corrected: str,
        *,
        schema: SchemaGraph | None = None,
        space_overlay: Any = None,
        artifacts_dir: str | None = None,
        pipeline_session: Any = None,
    ) -> SessionStep:
        """Answer a domain_knowledge route from the active knowledge list."""
        route = QuestionRoute.DOMAIN_KNOWLEDGE
        scope_ctx, visible, space_tables, space_snapshot = MainInitOps.meta_visibility_knobs(
            owner, schema, space_overlay, pipeline_session=pipeline_session
        )
        visibility_fp = MainInitOps._meta_answer_visibility_fingerprint(
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
        )
        cached = MainInitOps._lookup_meta_answer_cache(
            artifacts_dir,
            schema=schema,
            owner=owner,
            space_overlay=space_overlay,
            corrected=corrected,
            route=route,
            visibility_fingerprint=visibility_fp,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
            space_snapshot=space_snapshot,
            pipeline_session=pipeline_session,
        )
        if cached is not None:
            return cached
        active = active_domain_knowledge()
        if active:
            entries = active
        elif schema is not None:
            holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
            engine_entries = holder.entries() if isinstance(holder, DomainKnowledgeHolder) else ()
            entries = MainSpaceOps.derive_caller_scoped_domain_knowledge(
                engine_entries=engine_entries,
                schema=schema,
                scope_ctx=scope_ctx,
                visible_objects=visible,
                space_snapshot=space_snapshot,
                space_tables=space_tables,
            )
        else:
            entries = MainInitOps._resolve_active_domain_knowledge_entries(owner)
        payload_entries = [{"key": e.key, "kind": e.kind, "text": e.text} for e in entries]
        if not payload_entries:
            step = SessionStep(
                done=True,
                prompt=None,
                kind=SESSION_KIND_META,
                sql=None,
                data=None,
                answer=META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE,
                diagnostics=(),
                intent_summary=None,
                semantic_warnings=(),
                error=None,
                parameters=(),
            )
            MainInitOps._store_meta_answer_cache(
                artifacts_dir,
                schema=schema,
                owner=owner,
                space_overlay=space_overlay,
                corrected=corrected,
                route=route,
                visibility_fingerprint=visibility_fp,
                step=step,
            )
            return step
        notify("Metadata cache miss", stage="meta", code="meta.cache.miss", level="info")
        user = stable_json({"question": corrected, "domain_knowledge": payload_entries})
        answer: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    raw = LLMProvider.json(META_DOMAIN_KNOWLEDGE_SYSTEM, user, task="meta_dk")
                else:
                    notify("Metadata answer repair", stage="meta", code="meta.answer.repair", level="info")
                    repair_user = stable_json(
                        {
                            "question": corrected,
                            "domain_knowledge": payload_entries,
                            "previous_answer": answer,
                            "error": last_error,
                        }
                    )
                    raw = LLMProvider.json(META_DOMAIN_KNOWLEDGE_SYSTEM, repair_user, task="meta_dk")
                if not isinstance(raw, dict):
                    raise ValueError("domain knowledge answer must be a JSON object")
                answer = raw
                jsonschema.validate(instance=answer, schema=META_KNOWLEDGE_ANSWER_SCHEMA)
                if answer.get("response_kind") == META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND:
                    step = MainInitOps._meta_insufficient_knowledge_step()
                    MainInitOps._store_meta_answer_cache(
                        artifacts_dir,
                        schema=schema,
                        owner=owner,
                        space_overlay=space_overlay,
                        corrected=corrected,
                        route=route,
                        visibility_fingerprint=visibility_fp,
                        step=step,
                    )
                    return step
                if answer.get("response_kind") != "domain_knowledge":
                    raise ValueError("response_kind must be domain_knowledge")
                message = str(answer.get("message") or "").strip()
                if not message:
                    raise ValueError("domain knowledge message must be non-empty")
                notify("Metadata answer validated", stage="meta", code="meta.answer.validated", level="info")
                step = SessionStep(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_META,
                    sql=None,
                    data=None,
                    answer=message,
                    diagnostics=(),
                    intent_summary=None,
                    semantic_warnings=(),
                    error=None,
                    parameters=(),
                )
                MainInitOps._store_meta_answer_cache(
                    artifacts_dir,
                    schema=schema,
                    owner=owner,
                    space_overlay=space_overlay,
                    corrected=corrected,
                    route=route,
                    visibility_fingerprint=visibility_fp,
                    step=step,
                )
                return step
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                last_error = str(exc)
                continue
        notify(
            f"Metadata answer failed: {last_error or 'unknown'}",
            stage="meta",
            code="meta.answer.failed",
            level="error",
        )
        return MainInitOps._meta_terminal_error_step(
            detail_code="meta.answer.failed",
            last_error=last_error,
        )

    @staticmethod
    def _answer_schema_and_knowledge_question(
        owner: Any,
        corrected: str,
        schema: SchemaGraph,
        *,
        space_overlay: Any = None,
        artifacts_dir: str | None = None,
        pipeline_session: Any = None,
    ) -> SessionStep:
        """Answer a combined schema_and_knowledge route from filtered schema + DK payloads."""
        route = QuestionRoute.SCHEMA_AND_KNOWLEDGE
        scope_ctx, visible, space_tables, space_snapshot = MainInitOps.meta_visibility_knobs(
            owner, schema, space_overlay, pipeline_session=pipeline_session
        )
        visibility_fp = MainInitOps._meta_answer_visibility_fingerprint(
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
        )
        cached = MainInitOps._lookup_meta_answer_cache(
            artifacts_dir,
            schema=schema,
            owner=owner,
            space_overlay=space_overlay,
            corrected=corrected,
            route=route,
            visibility_fingerprint=visibility_fp,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
            space_snapshot=space_snapshot,
            pipeline_session=pipeline_session,
        )
        if cached is not None:
            return cached
        table_descriptions, column_descriptions = MainInitOps._space_description_overlays(
            space_snapshot, pipeline_session=pipeline_session
        )
        dump = MainInitOps.build_meta_schema_dump(
            schema,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
            exclude_restricted=True,
            table_descriptions=table_descriptions,
            column_descriptions=column_descriptions,
        )
        active = active_domain_knowledge()
        if active:
            entries = active
        else:
            holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
            engine_entries = holder.entries() if isinstance(holder, DomainKnowledgeHolder) else ()
            entries = MainSpaceOps.derive_caller_scoped_domain_knowledge(
                engine_entries=engine_entries,
                schema=schema,
                scope_ctx=scope_ctx,
                visible_objects=visible,
                space_snapshot=space_snapshot,
                space_tables=space_tables,
            )
        payload_entries = [{"key": e.key, "kind": e.kind, "text": e.text} for e in entries]
        notify("Metadata cache miss", stage="meta", code="meta.cache.miss", level="info")
        user_payload = {"question": corrected, "schema": dump, "domain_knowledge": payload_entries}
        user = stable_json(user_payload)
        answer: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    raw = LLMProvider.json(META_SCHEMA_AND_KNOWLEDGE_SYSTEM, user, task="meta_both")
                else:
                    notify("Metadata answer repair", stage="meta", code="meta.answer.repair", level="info")
                    repair_user = stable_json(
                        {
                            **user_payload,
                            "previous_answer": answer,
                            "error": last_error,
                        }
                    )
                    raw = LLMProvider.json(META_SCHEMA_AND_KNOWLEDGE_SYSTEM, repair_user, task="meta_both")
                if not isinstance(raw, dict):
                    raise ValueError("combined metadata answer must be a JSON object")
                answer = raw
                jsonschema.validate(instance=answer, schema=META_SCHEMA_AND_KNOWLEDGE_ANSWER_SCHEMA)
                if answer.get("response_kind") == META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND:
                    step = MainInitOps._meta_insufficient_knowledge_step()
                    MainInitOps._store_meta_answer_cache(
                        artifacts_dir,
                        schema=schema,
                        owner=owner,
                        space_overlay=space_overlay,
                        corrected=corrected,
                        route=route,
                        visibility_fingerprint=visibility_fp,
                        step=step,
                    )
                    return step
                if answer.get("response_kind") != "schema_and_knowledge":
                    raise ValueError("response_kind must be schema_and_knowledge")
                message = str(answer.get("message") or "").strip()
                if not message:
                    raise ValueError("combined metadata message must be non-empty")
                notify("Metadata answer validated", stage="meta", code="meta.answer.validated", level="info")
                step = SessionStep(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_META,
                    sql=None,
                    data=None,
                    answer=message,
                    diagnostics=(),
                    intent_summary=None,
                    semantic_warnings=(),
                    error=None,
                    parameters=(),
                )
                MainInitOps._store_meta_answer_cache(
                    artifacts_dir,
                    schema=schema,
                    owner=owner,
                    space_overlay=space_overlay,
                    corrected=corrected,
                    route=route,
                    visibility_fingerprint=visibility_fp,
                    step=step,
                )
                return step
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                last_error = str(exc)
                continue
        notify(
            f"Metadata answer failed: {last_error or 'unknown'}",
            stage="meta",
            code="meta.answer.failed",
            level="error",
        )
        return MainInitOps._meta_terminal_error_step(
            detail_code="meta.answer.failed",
            last_error=last_error,
        )

    @staticmethod
    def answer_metadata_question(
        owner: Any,
        corrected: str,
        route: QuestionRoute | str,
        schema: SchemaGraph | None,
        space_overlay: Any = None,
        artifacts_dir: str | None = None,
        pipeline_session: Any = None,
    ) -> SessionStep:
        """Answer a schema_catalog, domain_knowledge, or schema_and_knowledge question without SQL generation."""
        route_enum = route if isinstance(route, QuestionRoute) else QuestionRoute(str(route))
        if route_enum == QuestionRoute.DOMAIN_KNOWLEDGE:
            return MainInitOps._answer_domain_knowledge_question(
                owner,
                corrected,
                schema=schema,
                space_overlay=space_overlay,
                artifacts_dir=artifacts_dir,
                pipeline_session=pipeline_session,
            )
        if schema is None:
            notify("Metadata answer failed: schema missing", stage="meta", code="meta.answer.failed", level="error")
            return MainInitOps._meta_terminal_error_step(
                detail_code="meta.answer.failed",
                last_error="schema missing for metadata answer",
            )
        if route_enum == QuestionRoute.SCHEMA_AND_KNOWLEDGE:
            return MainInitOps._answer_schema_and_knowledge_question(
                owner,
                corrected,
                schema,
                space_overlay=space_overlay,
                artifacts_dir=artifacts_dir,
                pipeline_session=pipeline_session,
            )
        scope_ctx, visible, space_tables, space_snapshot = MainInitOps.meta_visibility_knobs(
            owner, schema, space_overlay, pipeline_session=pipeline_session
        )
        visibility_fp = MainInitOps._meta_answer_visibility_fingerprint(
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
        )
        cached = MainInitOps._lookup_meta_answer_cache(
            artifacts_dir,
            schema=schema,
            owner=owner,
            space_overlay=space_overlay,
            corrected=corrected,
            route=route_enum,
            visibility_fingerprint=visibility_fp,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
            space_snapshot=space_snapshot,
            pipeline_session=pipeline_session,
        )
        if cached is not None:
            return cached
        table_descriptions, column_descriptions = MainInitOps._space_description_overlays(
            space_snapshot, pipeline_session=pipeline_session
        )
        dump = MainInitOps.build_meta_schema_dump(
            schema,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_tables=space_tables,
            exclude_restricted=True,
            table_descriptions=table_descriptions,
            column_descriptions=column_descriptions,
        )
        notify("Metadata cache miss", stage="meta", code="meta.cache.miss", level="info")
        user_payload = {"schema": dump, "question": corrected}
        user = prompt_json(user_payload, META_SCHEMA_CATALOG_PROMPT_KEY_ORDER)
        answer: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    raw = LLMProvider.json(META_SCHEMA_CATALOG_SYSTEM, user, task="meta_schema")
                else:
                    notify("Metadata answer repair", stage="meta", code="meta.answer.repair", level="info")
                    repair_user = prompt_json(
                        {
                            "schema": dump,
                            "question": corrected,
                            "previous_answer": answer,
                            "error": last_error,
                        },
                        META_SCHEMA_CATALOG_REPAIR_PROMPT_KEY_ORDER,
                    )
                    raw = LLMProvider.json(META_SCHEMA_CATALOG_SYSTEM, repair_user, task="meta_schema")
                if not isinstance(raw, dict):
                    raise ValueError("metadata answer must be a JSON object")
                answer = raw
                MainInitOps.validate_meta_schema_answer(answer, dump)
                if answer.get("response_kind") == META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND:
                    step = MainInitOps._meta_insufficient_knowledge_step()
                    MainInitOps._store_meta_answer_cache(
                        artifacts_dir,
                        schema=schema,
                        owner=owner,
                        space_overlay=space_overlay,
                        corrected=corrected,
                        route=route_enum,
                        visibility_fingerprint=visibility_fp,
                        step=step,
                    )
                    return step
                notify("Metadata answer validated", stage="meta", code="meta.answer.validated", level="info")
                message = MainInitOps.format_meta_schema_message(answer)
                step = SessionStep(
                    done=True,
                    prompt=None,
                    kind=SESSION_KIND_META,
                    sql=None,
                    data=None,
                    answer=message,
                    diagnostics=(),
                    intent_summary=None,
                    semantic_warnings=(),
                    error=None,
                    parameters=(),
                )
                MainInitOps._store_meta_answer_cache(
                    artifacts_dir,
                    schema=schema,
                    owner=owner,
                    space_overlay=space_overlay,
                    corrected=corrected,
                    route=route_enum,
                    visibility_fingerprint=visibility_fp,
                    step=step,
                    schema_payload=dict(answer),
                )
                return step
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                last_error = str(exc)
                continue
        notify(
            f"Metadata answer failed: {last_error or 'unknown'}",
            stage="meta",
            code="meta.answer.failed",
            level="error",
        )
        return MainInitOps._meta_terminal_error_step(
            detail_code="meta.answer.failed",
            last_error=last_error,
        )

    @staticmethod
    def _routing_domain_knowledge_keys(
        owner: Any,
        schema: SchemaGraph | None,
        *,
        pipeline_session: Any = None,
    ) -> tuple[str, ...]:
        """Return caller-scoped domain-knowledge concept keys for question routing inventory."""
        if schema is None:
            return ()
        scope_ctx, visible, space_tables, space_snapshot = MainInitOps.meta_visibility_knobs(
            owner, schema, None, pipeline_session=pipeline_session
        )
        holder = getattr(owner, "_domain_knowledge", None) if owner is not None else None
        engine_entries = holder.entries() if isinstance(holder, DomainKnowledgeHolder) else ()
        entries = MainSpaceOps.derive_caller_scoped_domain_knowledge(
            engine_entries=engine_entries,
            schema=schema,
            scope_ctx=scope_ctx,
            visible_objects=visible,
            space_snapshot=space_snapshot,
            space_tables=space_tables,
        )
        return tuple(sorted({str(e.key).strip() for e in entries if str(e.key).strip()}))

    @staticmethod
    def interactive_run_once(
        schema: SchemaGraph | None = None,
        store: dict[str, Any] | TemplateStoreView | None = None,
        templates: dict[str, Any] | None = None,
        rejected: dict[str, Any] | None = None,
        schema_terms: Any | None = None,
        question: str | None = None,
        pipeline_session: Any | None = None,
    ) -> dict[str, Any] | None:
        """Execute a single interactive pipeline iteration. Reads a question from stdin or uses the supplied `question`, validates it, checks for template reuse, parses intent via LLM if needed, generates SQL, executes it, and handles user feedback."""
        if question is None:
            notify("Enter question", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
            try:
                question = prompt("").strip()
            except (EOFError, KeyboardInterrupt):
                terminated()
                return None

        if not question:
            if pipeline_session is not None:
                note_interactive_turn(
                    choice_port=pipeline_session, outcome="parse_failed", error="Question must not be empty."
                )
                return None
            invalid_input()
            return None
        MainSpaceOps.raise_if_session_turn_cancelled()
        progress("\nValidating question...")

        raw_question = question

        owner_dialect = None
        owner = getattr(pipeline_session, "_owner", None) if pipeline_session is not None else None
        if owner is not None:
            owner_dialect = getattr(owner, "_dialect", None)
        fed_reuse_kwargs = MainInitOps.federation_reuse_kwargs(owner, pipeline_session)

        dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
            schema, store, templates, rejected, schema_terms, dialect=owner_dialect
        )
        choice_port: InteractiveChoicePort | None = pipeline_session
        persist_tl = MainInteractiveOps.persist_template_learning_for_pipeline_session(choice_port)
        gate_kwargs = MainSpaceOps.consumer_sql_gate_kwargs(choice_port)
        caller_visible_tables = effective_execution_visible_tables(
            schema,
            gate_kwargs.get("schema_context"),
            gate_kwargs.get("visible_objects"),
        )

        pending_pre = TemplateOps.find_pending_template_for_question(
            templates, normalize_question(raw_question.strip())
        )
        if pending_pre is not None and not MainInitOps._owner_has_federation(owner):
            debug(f"pending template confirmation short-circuit (template='{pending_pre.id}')")
            reuse_pending = handle_direct_sql_reuse(
                normalize_question(raw_question.strip()),
                pending_pre,
                dialect,
                store,
                templates,
                rejected,
                schema,
                existing_nl=None,
                choice_port=choice_port,
                form_storage=QuestionFormStorage(corrected=raw_question.strip()),
                persist_template_learning=persist_tl,
                **gate_kwargs,
                **fed_reuse_kwargs,
            )
            if reuse_pending is not None and reuse_pending.success:
                return None

        tmpl_pre = match_question_level_template_reuse(
            raw_question,
            templates,
            template_store=store,
            schema=schema,
            visible_tables=caller_visible_tables,
        )
        if tmpl_pre.reuse_type == "direct_reuse" and not MainInitOps._owner_has_federation(owner):
            best_template_pre = tmpl_pre.best_template
            if best_template_pre is None:
                return None
            debug(f"direct SQL reuse via question match pre-validation (trust>=1, template='{best_template_pre.id}')")
            debug("[main_execution.interactive_run_once] direct_reuse_pre: question_match")
            assert tmpl_pre.reuse_candidate_normalized is not None
            reuse_pre = handle_direct_sql_reuse(
                tmpl_pre.reuse_candidate_normalized,
                best_template_pre,
                dialect,
                store,
                templates,
                rejected,
                schema,
                existing_nl=None,
                choice_port=choice_port,
                reuse_history_index=tmpl_pre.reuse_history_index,
                form_storage=QuestionFormStorage(corrected=raw_question.strip()),
                persist_template_learning=persist_tl,
                **gate_kwargs,
                **fed_reuse_kwargs,
            )
            if reuse_pre is not None and reuse_pre.success:
                return None

        validation = validate_question(
            raw_question,
            table_names=caller_visible_tables,
            domain_knowledge_keys=MainInitOps._routing_domain_knowledge_keys(
                owner,
                schema,
                pipeline_session=pipeline_session,
            ),
        )
        if not validation.accepted:
            if validation.route == QuestionRoute.RESTRICTED:
                restricted_message = refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED)
                emit_session_refusal_diagnostic(
                    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
                    f"\n{restricted_message}",
                    stage="rephrase_hint",
                )
                note_interactive_turn(
                    choice_port,
                    outcome="restricted",
                    error=restricted_message,
                    refusal_diagnostic_code=DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
                )
            elif validation.invalid_kind == "conversational":
                conv_message = refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY)
                emit_session_refusal_diagnostic(
                    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
                    f"\n{conv_message}",
                    stage="rephrase_hint",
                )
                note_interactive_turn(
                    choice_port,
                    outcome="conversational_deny",
                    error=conv_message,
                    refusal_diagnostic_code=DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
                )
            else:
                print_rephrase_hint(RephraseHint.VAGUE_QUESTION)
                note_interactive_turn(choice_port, outcome="invalid_question", error="Question failed validation.")
            return None
        corrected_text = validation.corrected
        if corrected_text != raw_question:
            debug(f"[main_execution.interactive_run_once] typo_corrected: '{raw_question}' -> '{corrected_text}'")

        if validation.route in (
            QuestionRoute.SCHEMA_CATALOG,
            QuestionRoute.DOMAIN_KNOWLEDGE,
            QuestionRoute.SCHEMA_AND_KNOWLEDGE,
        ):
            route_code = f"meta.route.{validation.route.value}"
            notify(
                f"Metadata route: {validation.route.value}",
                stage="meta",
                code=route_code,
                level="info",
            )
            art = getattr(owner, "_artifacts_dir", None) if owner is not None else None
            adir: str | None = None
            if art is not None:
                try:
                    adir = os.path.abspath(os.fspath(art))
                except (TypeError, OSError, ValueError):
                    adir = None
            space_overlay = getattr(pipeline_session, "_space_name", None) if pipeline_session is not None else None
            meta_step = MainInitOps.answer_metadata_question(
                owner,
                corrected_text,
                validation.route,
                schema,
                space_overlay,
                adir,
                pipeline_session=pipeline_session,
            )
            if pipeline_session is not None:
                pipeline_session._pending_terminal_step = meta_step
            return None

        tmpl_typo = match_question_level_template_reuse(
            corrected_text,
            templates,
            template_store=store,
            schema=schema,
            visible_tables=caller_visible_tables,
        )
        if tmpl_typo.reuse_type == "direct_reuse" and not MainInitOps._owner_has_federation(owner):
            best_template_typo = tmpl_typo.best_template
            if best_template_typo is None:
                return None
            debug(f"direct SQL reuse via question match (trust>=1, template='{best_template_typo.id}')")
            debug("[main_execution.interactive_run_once] direct_reuse: question_match")
            assert tmpl_typo.reuse_candidate_normalized is not None
            reuse_result = handle_direct_sql_reuse(
                tmpl_typo.reuse_candidate_normalized,
                best_template_typo,
                dialect,
                store,
                templates,
                rejected,
                schema,
                existing_nl=None,
                choice_port=choice_port,
                reuse_history_index=tmpl_typo.reuse_history_index,
                form_storage=QuestionFormStorage(corrected=corrected_text),
                persist_template_learning=persist_tl,
                **gate_kwargs,
                **fed_reuse_kwargs,
            )
            if reuse_result is not None and reuse_result.success:
                return None

        neg_drop = False
        normalized_canonical = normalize_question_via_llm(corrected_text, raw_original=raw_question)
        if normalized_canonical != corrected_text and TemplateOps.has_any_rejection_history_for_question(
            store, corrected_text
        ):
            debug(
                f"[main_execution.interactive_run_once] dropped_normalized_due_to_negative_memory {normalized_canonical!r}"
            )
            neg_drop = True
            normalized_canonical = corrected_text

        tmpl_norm = None
        if normalized_canonical != corrected_text:
            tmpl_norm = match_question_level_template_reuse(
                normalized_canonical,
                templates,
                template_store=store,
                schema=schema,
                visible_tables=caller_visible_tables,
            )
            if tmpl_norm.reuse_type == "direct_reuse" and not MainInitOps._owner_has_federation(owner):
                best_template_norm = tmpl_norm.best_template
                if best_template_norm is None:
                    return None
                debug(f"direct SQL reuse via normalized question match (trust>=1, template='{best_template_norm.id}')")
                assert tmpl_norm.reuse_candidate_normalized is not None
                fs_norm = QuestionFormStorage(
                    corrected=corrected_text,
                    normalized_optional=normalized_canonical,
                    normalized_negative_memory_dropped=neg_drop,
                    accept_via_normalized_lookup_only=True,
                )
                reuse_norm = handle_direct_sql_reuse(
                    tmpl_norm.reuse_candidate_normalized,
                    best_template_norm,
                    dialect,
                    store,
                    templates,
                    rejected,
                    schema,
                    existing_nl=None,
                    choice_port=choice_port,
                    reuse_history_index=tmpl_norm.reuse_history_index,
                    form_storage=fs_norm,
                    persist_template_learning=persist_tl,
                    **gate_kwargs,
                    **fed_reuse_kwargs,
                )
                if reuse_norm is not None and reuse_norm.success:
                    return None

        norm_opt = normalized_canonical if normalized_canonical != corrected_text else None
        form_storage = QuestionFormStorage(
            corrected=corrected_text,
            normalized_optional=norm_opt,
            normalized_negative_memory_dropped=neg_drop,
            accept_via_normalized_lookup_only=False,
        )

        q_norm = normalize_question(corrected_text)
        debug(f"[main_execution.interactive_run_once] q_norm: {q_norm}")

        if MainInitOps._owner_has_federation(owner):
            fed_kwargs = MainInitOps.federation_reuse_kwargs(owner, choice_port)
            intake_reuse = try_federation_plan_inplace_reuse(
                q_norm,
                schema,
                dialect,
                federation_dir=fed_kwargs.get("federation_dir"),
                federation_manifest=fed_kwargs.get("federation_manifest"),
                federation_mappings=fed_kwargs.get("federation_mappings"),
                stores_by_source=fed_kwargs.get("stores_by_source"),
                dialects_by_source=fed_kwargs.get("dialects_by_source"),
                source_runtimes=fed_kwargs.get("source_runtimes"),
                member_graphs=fed_kwargs.get("member_graphs"),
                gate_kwargs_by_source=fed_kwargs.get("gate_kwargs_by_source"),
            )
            if intake_reuse is not None and intake_reuse.success:
                return None

        conv_hints: tuple[str, ...] = ()
        if pipeline_session is not None:
            raw_h = getattr(pipeline_session, "_pending_conversation_rejection_hints", None)
            if isinstance(raw_h, tuple):
                conv_hints = raw_h
                pipeline_session._pending_conversation_rejection_hints = ()

        refinement_ctx = RefinementContext(corrected_text, form_storage, conversation_rejection_hints=conv_hints)
        MainInteractiveOps.interactive_attach_refinement_ctx(choice_port, refinement_ctx)

        while True:
            MainSpaceOps.raise_if_session_turn_cancelled()
            try:
                completed = MainInteractiveOps.interactive_run_intent_pass(
                    corrected_text=corrected_text,
                    q_norm=q_norm,
                    dialect=dialect,
                    schema=schema,
                    store=store,
                    templates=templates,
                    rejected=rejected,
                    schema_terms=schema_terms,
                    choice_port=choice_port,
                    form_storage=form_storage,
                    refinement_ctx=refinement_ctx,
                    persist_template_learning=persist_tl,
                )
                if not completed:
                    return None
                break
            except RefinementRetry:
                continue
        return None

    @staticmethod
    def get_seed_warmup_summary_from_dir(artifacts_dir: str, version: int) -> SeedWarmupSummary:
        """Build a ``SeedWarmupSummary`` from a persisted ``seed_warmup_report_v{version}.json`` file."""
        report_path = os.path.join(artifacts_dir, SeedWarmupConfig.SEED_WARMUP_REPORT_PATTERN.format(version=version))
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Seed warmup report v{version} not found")

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        total = report.get("total", 0)
        success = report.get("success", 0)
        failed = report.get("failed", 0)
        success_rate = round(success / total, 3) if total > 0 else 0.0

        return SeedWarmupSummary(
            version=version,
            total=total,
            success=success,
            failed=failed,
            success_rate=success_rate,
            seed_questions_loaded=int(report.get("seed_questions_loaded", 0)),
            gold_intents_total=int(report.get("gold_intents_total", 0)),
            unique_prompts=int(
                report.get("unique_prompts", report.get("synthetic_runnable_count", report.get("unique_synthetic", 0)))
            ),
            gold_new=int(report.get("gold_new", 0)),
            gold_skipped=int(report.get("gold_skipped", 0)),
            gold_failed=int(report.get("gold_failed", 0)),
            gold_user_rejected=int(report.get("gold_user_rejected", 0)),
            deduped_prompts_count=int(
                report.get(
                    "deduped_prompts_count",
                    report.get("synthetic_unique_body_keys", report.get("deduped_synthetic_count", 0)),
                )
            ),
            gold_prompts_count=int(report.get("gold_prompts_count", report.get("seed_questions_loaded", 0))),
            templates_added=int(report.get("templates_added", 0)),
            validation_drop=int(report.get("validation_drop", 0)),
            realism_drop=int(report.get("realism_drop", 0)),
            question_generation_failed=int(report.get("question_generation_failed", 0)),
            early_pipeline_failed=int(report.get("early_pipeline_failed", 0)),
        )

    @staticmethod
    def _toml_claim_put_scalar(
        block: dict[str, Any], subkey: str, target_key: str, output: dict[str, str], claimed: set[str]
    ) -> None:
        if subkey not in block:
            return
        claimed.add(target_key)
        raw_value = block.get(subkey)
        if raw_value is None:
            return
        text = str(raw_value).strip()
        if text:
            output[target_key] = text

    @staticmethod
    def _toml_claim_put_csv_files(block: dict[str, Any], output: dict[str, str], claimed: set[str]) -> None:
        files_raw = block.get("files")
        if files_raw is None:
            return
        claimed.add("CSV_FILES")
        if isinstance(files_raw, list):
            parts = [str(item).strip() for item in files_raw if str(item).strip()]
            if parts:
                output["CSV_FILES"] = ",".join(parts)
        else:
            text = str(files_raw).strip()
            if text:
                output["CSV_FILES"] = text

    @staticmethod
    def _flatten_scalar_engine_fields(
        block: dict[str, Any],
        field_specs: tuple[tuple[str, str], ...],
        output: dict[str, str],
        claimed: set[str],
        *,
        section_name: str,
    ) -> None:
        for subkey, target_key in field_specs:
            MainInitOps._toml_claim_put_scalar(block, subkey, target_key, output, claimed)
        if section_name in {"csv", "excel"}:
            MainInitOps._toml_claim_put_csv_files(block, output, claimed)

    @staticmethod
    def _flatten_engine_block(
        section_name: str,
        block: dict[str, Any],
        field_specs: tuple[tuple[str, str], ...],
        connection_name: str | None = None,
    ) -> tuple[dict[str, str], set[str], frozenset[str]]:
        """Flatten one engine TOML block to env-style keys. Scalar keys define a single unnamed connection. Nested dicts define named connections; when only sub-tables are present there is no unnamed default."""
        named_blocks = {key: value for key, value in block.items() if isinstance(value, dict)}
        scalar_keys = {key for key in block if not isinstance(block.get(key), dict)}
        if named_blocks and scalar_keys:
            raise ConfigError(
                f"config_file [{section_name}] mixes scalar keys with named connection sub-tables; "
                "use either a flat block or named sub-tables, not both."
            )
        output: dict[str, str] = {}
        claimed: set[str] = set()
        if not named_blocks:
            MainInitOps._flatten_scalar_engine_fields(block, field_specs, output, claimed, section_name=section_name)
            return output, claimed, frozenset()
        connection_names = frozenset(str(name) for name in named_blocks)
        selected = connection_name
        if selected is None and len(named_blocks) == 1:
            selected = next(iter(named_blocks))
        if selected is None:
            return output, claimed, connection_names
        if selected not in named_blocks:
            options = ", ".join(sorted(connection_names))
            raise ConfigError(
                f"config_file [{section_name}] has no connection {selected!r}; expected one of: {options}."
            )
        MainInitOps._flatten_scalar_engine_fields(
            named_blocks[selected], field_specs, output, claimed, section_name=section_name
        )
        return output, claimed, connection_names

    @staticmethod
    def _select_connection_name(
        env: Mapping[str, str],
        named_connections_by_engine: Mapping[str, frozenset[str]],
        engine: str,
        *,
        explicit_connection: str | None = None,
    ) -> str | None:
        """Resolve the named connection handle for *engine*, if any."""
        names = named_connections_by_engine.get(engine, frozenset())
        if not names:
            return None
        explicit = str(explicit_connection or env.get("AETHERDIALECT_CONNECTION", "") or "").strip()
        if explicit:
            if explicit not in names:
                options = ", ".join(sorted(names))
                raise ConfigError(
                    f"Unknown AETHERDIALECT_CONNECTION {explicit!r} for {engine}; expected one of: {options}."
                )
            return explicit
        if len(names) == 1:
            return next(iter(names))
        options = ", ".join(sorted(names))
        raise ConfigError(
            f"Multiple named connections configured for {engine} ({options}); "
            "set AETHERDIALECT_CONNECTION or pass connection= to AetherEngine."
        )

    @staticmethod
    def _load_config_file(
        path: str | os.PathLike[str] | None, *, connection: str | None = None
    ) -> tuple[dict[str, str], frozenset[str], dict[str, frozenset[str]]]:
        """Parse a TOML configuration file into flat environment-style string keys."""
        if path is None:
            return {}, frozenset(), {}
        path_str = str(path).strip()
        if not path_str:
            return {}, frozenset(), {}
        expanded = os.path.expanduser(path_str)
        try:
            with open(expanded, "rb") as file_handle:
                document = tomllib.load(file_handle)
        except OSError as exc:
            raise ConfigError(f"config_file cannot be opened: {expanded}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"config_file TOML parse error in {expanded}: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigError(f"config_file root must be a table: {expanded}")
        output: dict[str, str] = {}
        claimed: set[str] = set()
        named_connections_by_engine: dict[str, frozenset[str]] = {}

        def _claim_put(block: dict[str, Any], subkey: str, target_key: str) -> None:
            MainInitOps._toml_claim_put_scalar(block, subkey, target_key, output, claimed)

        openai_block = document.get("openai")
        if isinstance(openai_block, dict):
            _claim_put(openai_block, "api_key", "OPENAI_API_KEY")
            _claim_put(openai_block, "base_url", "OPENAI_BASE_URL")
        azure_block = document.get("azure_openai")
        if isinstance(azure_block, dict):
            _claim_put(azure_block, "endpoint", "AZURE_OPENAI_ENDPOINT")
            _claim_put(azure_block, "api_key", "AZURE_OPENAI_API_KEY")
            _claim_put(azure_block, "api_version", "AZURE_OPENAI_API_VERSION")
            _claim_put(azure_block, "base_url", "AZURE_OPENAI_BASE_URL")
            deployments_block = azure_block.get("deployments")
            if isinstance(deployments_block, dict):
                _claim_put(deployments_block, "light", "AZURE_OPENAI_DEPLOYMENT_LIGHT")
                _claim_put(deployments_block, "heavy", "AZURE_OPENAI_DEPLOYMENT_HEAVY")
        for section_name, field_specs in TOML_ENGINE_FIELD_MAPS.items():
            engine_block = document.get(section_name)
            if not isinstance(engine_block, dict):
                continue
            if section_name == "excel":
                field_specs = TOML_ENGINE_FIELD_MAPS["csv"]
            flat, section_claimed, named = MainInitOps._flatten_engine_block(
                section_name, engine_block, field_specs, connection
            )
            output.update(flat)
            claimed.update(section_claimed)
            if named:
                engine_name = TOML_SECTION_TO_ENGINE[section_name]
                existing = named_connections_by_engine.get(engine_name, frozenset())
                named_connections_by_engine[engine_name] = existing | named
        engine_block = document.get("engine")
        if isinstance(engine_block, dict):
            _claim_put(engine_block, "selected", "AETHERDIALECT_ENGINE")
            _claim_put(engine_block, "connection", "AETHERDIALECT_CONNECTION")
        llm_block = document.get("llm")
        if isinstance(llm_block, dict):
            _claim_put(llm_block, "provider", "AETHERDIALECT_LLM_PROVIDER")
        sandbox_block = document.get("sandbox")
        if isinstance(sandbox_block, dict):
            _claim_put(sandbox_block, "fixtures_file", "AETHERDIALECT_SANDBOX_FIXTURES_FILE")
        mock_block = document.get("mock")
        if isinstance(mock_block, dict):
            _claim_put(mock_block, "fixtures_file", "AETHERDIALECT_MOCK_FIXTURES_FILE")
        return output, frozenset(claimed), named_connections_by_engine

    @staticmethod
    def _merge_configuration_environment(
        config_file_values: Mapping[str, str], *, toml_claimed_keys: frozenset[str] | None = None
    ) -> tuple[dict[str, str], frozenset[str]]:
        """Build the effective environment mapping used for engine configuration reads. When *toml_claimed_keys* is ``None`` (no ``config_file`` in use), non-empty TOML values overlay ``os.environ`` for matching keys only. When *toml_claimed_keys* is provided (a ``config_file`` was loaded), the file is the single source of truth for every key in that set: non-empty flattened values replace ``os.environ``, and keys present in the file with empty or absent string values remove the variable from the effective mapping so environment defaults cannot leak past an explicit TOML field. This function never mutates ``os.environ``."""
        baseline = {str(k): str(v) for k, v in os.environ.items()}
        merged = dict(baseline)
        if toml_claimed_keys is None:
            config_effect_candidates: set[str] = set()
            for raw_key, raw_val in config_file_values.items():
                key = str(raw_key)
                value_string = str(raw_val).strip()
                if not value_string:
                    continue
                baseline_value = str(baseline.get(key, "") or "").strip()
                if value_string != baseline_value:
                    config_effect_candidates.add(key)
                merged[key] = value_string
            final_diag: set[str] = set()
            for key in config_effect_candidates:
                toml_value = str(config_file_values.get(key, "")).strip()
                if toml_value and merged.get(key) == toml_value:
                    final_diag.add(key)
            return merged, frozenset(final_diag)

        for key in toml_claimed_keys:
            sk = str(key)
            if sk in config_file_values:
                value_string = str(config_file_values[sk]).strip()
                if value_string:
                    merged[sk] = value_string
                else:
                    merged.pop(sk, None)
            else:
                merged.pop(sk, None)

        config_effect_candidates = set()
        for sk in config_file_values:
            value_string = str(config_file_values[sk]).strip()
            if not value_string:
                continue
            baseline_value = str(baseline.get(sk, "") or "").strip()
            if value_string != baseline_value:
                config_effect_candidates.add(sk)
        final_diag_ssot: set[str] = set()
        for key in config_effect_candidates:
            toml_value = str(config_file_values.get(key, "")).strip()
            if toml_value and merged.get(key) == toml_value:
                final_diag_ssot.add(key)
        return merged, frozenset(final_diag_ssot)

    @staticmethod
    def _engine_storage_slug_fragment(raw: str, *, fallback: str) -> str:
        """Return a filesystem-friendly lowercase token for a single slug component."""
        t = re.sub(r"[^0-9A-Za-z]+", "_", str(raw).strip()).strip("_").lower()
        return t if t else fallback

    @staticmethod
    def compute_connection_storage_slug(engine: str, runtime: EngineRuntimeConfig | None = None) -> str:
        """Return a stable connection slug derived from the active engine runtime configuration. When the composed slug is longer than :data:`ENGINE_STORAGE_SLUG_MAX_CHARS`, a deterministic hash suffix is used instead."""
        runtime_cls = DialectRegistry.get_runtime_config_class(engine)
        runtime_cfg = runtime if runtime is not None else runtime_cls()
        fields = dict(runtime_cfg.connection_slug_fields())
        slug_keys = runtime_cfg.connection_slug_keys()
        if runtime is None:
            for key in slug_keys:
                attr = key.upper()
                if hasattr(runtime_cls, attr):
                    class_val = getattr(runtime_cls, attr)
                    if class_val is not None and str(class_val).strip():
                        fields[key] = str(class_val)
        parts = [MainInitOps._engine_storage_slug_fragment(fields[key], fallback=key[0]) for key in slug_keys]
        slug = f"conn_{engine}_" + "_".join(parts)
        if len(slug) > int(ENGINE_STORAGE_SLUG_MAX_CHARS):
            digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:24]
            return f"conn_{engine}_{digest}"
        return slug

    @staticmethod
    def compute_engine_storage_dir(
        artifacts_root: str | None,
        engine: str,
        *,
        runtime: EngineRuntimeConfig | None = None,
        storage_dir: str | None = None,
    ) -> str:
        """Return the absolute engine storage directory for persisted artifacts. When *storage_dir* is set, return its absolute expanded path. Otherwise the parent directory is :meth:`EngineConfig.default_artifacts_root` when *artifacts_root* is ``None`` or blank, or the absolute expanded *artifacts_root*. The final directory is ``os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, connection_slug)``."""
        if storage_dir is not None and str(storage_dir).strip():
            return os.path.abspath(os.path.expanduser(str(storage_dir)))
        parent = (
            os.path.abspath(os.path.expanduser(str(artifacts_root)))
            if artifacts_root and str(artifacts_root).strip()
            else str(EngineConfig.default_artifacts_root())
        )
        slug = MainInitOps.compute_connection_storage_slug(engine, runtime=runtime)
        return os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, slug)

    @staticmethod
    def _prepare_schema_context_for_init(
        schema_context: EngineContext, engine_storage_dir: str, sink: Callable[[str], None]
    ) -> EngineContext:
        """Merge an explicit ``EngineContext`` with any compatible on- disk cache under *engine_storage_dir*."""
        try:
            cached = MainInitOps.load_schema_context_cache(engine_storage_dir)
        except ConfigError as exc:
            sink(str(exc))
            cached = None
        if cached is not None and (
            cached.include != schema_context.include
            or cached.allow_objects != schema_context.allow_objects
            or cached.deny_columns != schema_context.deny_columns
            or cached.allow_columns != schema_context.allow_columns
        ):
            sink("Schema scope changed since last run — caches will be rebuilt where needed.")
        notes_use = schema_context.notes_file
        sql_use = schema_context.sql_file
        if cached is not None:
            if notes_use is None and cached.notes_file:
                notes_use = cached.notes_file
                sink("  Schema context: reusing cached notes file.")
            if sql_use is None and cached.sql_file:
                sql_use = cached.sql_file
                sink("  Schema context: reusing cached SQL file.")
            cache_payload_path = os.path.join(engine_storage_dir, SCHEMA_CONTEXT_CACHE_NAME)
            try:
                with open(cache_payload_path, encoding="utf-8") as fh:
                    prev_ctx = json.load(fh)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                prev_ctx = None
            if isinstance(prev_ctx, dict):
                if schema_context.notes_file:
                    old_notes = prev_ctx.get("notes_text")
                    new_notes = MainInitOps._read_text_if_file(schema_context.notes_file)
                    if isinstance(old_notes, str) and isinstance(new_notes, str) and new_notes != old_notes:
                        sink("  Schema context: notes file changed since last run.")
                if schema_context.sql_file:
                    old_sql = prev_ctx.get("sql_text")
                    new_sql = MainInitOps._read_text_if_file(schema_context.sql_file)
                    if isinstance(old_sql, str) and isinstance(new_sql, str) and new_sql != old_sql:
                        sink("  Schema context: SQL file changed since last run.")
        if notes_use != schema_context.notes_file or sql_use != schema_context.sql_file:
            return EngineContext(
                allow_objects=schema_context.allow_objects,
                include=schema_context.include,
                deny_columns=schema_context.deny_columns,
                allow_columns=schema_context.allow_columns,
                notes_file=notes_use,
                sql_file=sql_use,
            )
        return schema_context

    @staticmethod
    def _env_all_non_empty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
        """Return True when every key maps to a non-blank string."""
        return all(str(env.get(k, "") or "").strip() for k in keys)

    @staticmethod
    def _env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
        """Return the first non-blank value among *keys*, else an empty string."""
        return EngineConfig.env_first_nonempty(env, *keys)

    @staticmethod
    def _env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
        """True when at least one key maps to a non-blank string."""
        return EngineConfig.env_any_nonempty(env, keys)

    @staticmethod
    def _env_role_hint(label: str, keys: tuple[str, ...]) -> str:
        return EngineConfig.env_role_hint(label, keys)

    @staticmethod
    def _runtime_config_for_engine(engine: str) -> type[EngineRuntimeConfig]:
        return cast(type[EngineRuntimeConfig], DialectRegistry.get_runtime_config_class(engine))

    @staticmethod
    def _apply_runtime_environments(env: Mapping[str, str]) -> None:
        """Load every registered runtime config whose partial env scope is present."""
        for engine in DialectRegistry.list_engines():
            runtime_cls = MainInitOps._runtime_config_for_engine(engine)
            if runtime_cls.should_apply_environment(env):
                runtime_cls.load_process_default_from_environment(env)

    @staticmethod
    def _select_engine_name(
        env: Mapping[str, str], named_connections_by_engine: Mapping[str, frozenset[str]] | None = None
    ) -> str:
        named = named_connections_by_engine or {}
        engines = DialectRegistry.list_engines()
        explicit = str(env.get("AETHERDIALECT_ENGINE", "") or "").strip().lower()
        if explicit:
            if explicit not in engines:
                raise ConfigError(f"Unsupported AETHERDIALECT_ENGINE: {explicit!r}. Expected one of {engines}.")
            blockers = MainInitOps._runtime_config_for_engine(explicit).selection_blockers(env)
            if blockers and not named.get(explicit):
                raise ConfigError(f"Cannot select {explicit} engine: {'; '.join(blockers)}")
            return explicit
        ready: list[str] = []
        for engine in engines:
            if not MainInitOps._runtime_config_for_engine(engine).selection_blockers(env):
                ready.append(engine)
            elif named.get(engine):
                ready.append(engine)
        if len(ready) > 1:
            labels = ", ".join(ready)
            raise ConfigError(
                f"Multiple database engines are configured and available ({labels}); set AETHERDIALECT_ENGINE "
                "or [engine] selected in the config file to one of them."
            )
        if len(ready) == 1:
            return ready[0]
        missing: list[str] = []
        for engine in engines:
            missing.extend(MainInitOps._runtime_config_for_engine(engine).selection_blockers(env))
        raise ConfigError("Cannot select database engine: " + "; ".join(missing))

    @staticmethod
    def _activate_engine(name: str) -> None:
        """Bind :attr:`EngineConfig.TYPE` and :attr:`EngineConfig.RUNTIME` to the chosen engine."""
        if name not in DialectRegistry.list_engines():
            raise ConfigError(f"Unsupported engine activation: {name!r}.")
        EngineConfig.TYPE = name
        EngineConfig.RUNTIME = MainInitOps._runtime_config_for_engine(name)

    @staticmethod
    def configure_runtime_from_environment(
        engine_context: EngineContext, merged_env: Mapping[str, str]
    ) -> tuple[str, EngineRuntimeConfig]:
        env: dict[str, str] = dict(merged_env)
        selected = MainInitOps._select_engine_name(env)
        MainInitOps._apply_runtime_environments(env)
        PolicyConfig.apply_environment(env)
        runtime = MainInitOps._runtime_config_for_engine(selected).from_environment(env)
        MainInitOps._activate_engine(selected)
        if selected == "databricks" and not cast(DatabricksRuntimeConfig, runtime).has_native_connection():
            if not DatabricksRuntimeConfig.pyspark_session_reachable():
                raise ConfigError(
                    "Databricks requires either all SQL warehouse connection variables or an active PySpark session."
                )
        MainInitOps._configure_llm_from_environment(env)
        if selected not in DialectRegistry.list_engines():
            raise ConfigError(f"Unsupported engine resolved: {selected!r}")
        return selected, runtime

    @staticmethod
    def validate_azure_llm_execution(llm_exec: Any) -> None:
        """Raise ``ConfigError`` when Azure provider is missing required deployment fields."""
        missing = [
            n
            for n, v in (
                ("azure_endpoint", getattr(llm_exec, "azure_endpoint", None)),
                ("azure_api_key", getattr(llm_exec, "azure_api_key", None)),
                ("azure_api_version", getattr(llm_exec, "azure_api_version", None)),
                ("deployment_light", getattr(llm_exec, "deployment_light", None)),
                ("deployment_heavy", getattr(llm_exec, "deployment_heavy", None)),
            )
            if not (isinstance(v, str) and v.strip())
        ]
        if missing:
            raise ConfigError("Azure OpenAI requires non-empty runtime configuration for: " + ", ".join(missing))

    @staticmethod
    def _apply_logical_model_env_overrides(env: Mapping[str, str]) -> None:
        """Override ``OPENAI_MODEL*`` ClassVars when matching environment keys are set."""
        for attr in (
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA_BASE",
            "OPENAI_MODEL_DDL",
            "OPENAI_MODEL_SCHEMA",
            "OPENAI_MODEL_DOMAIN_KNOWLEDGE",
            "OPENAI_MODEL_SYNTH",
            "OPENAI_MODEL_SYNTH_VARIETY",
            "OPENAI_MODEL_INTENT_FORMAT",
            "OPENAI_MODEL_INTENT_SCHEMA_REPAIR",
            "OPENAI_MODEL_UPLOAD_SUMMARY",
            "OPENAI_MODEL_UPLOAD_INTERPRET",
        ):
            raw = str(env.get(attr, "") or "").strip()
            if raw:
                setattr(EngineConfig, attr, raw)

    @staticmethod
    def _configure_openai_from_environment(env: Mapping[str, str]) -> None:
        """Populate :class:`EngineConfig` with OpenAI credentials and clear Azure fields."""
        EngineConfig.LLM_PROVIDER = "openai"
        EngineConfig.API_TOKEN = str(env["OPENAI_API_KEY"]).strip()
        EngineConfig.AZURE_API_TOKEN = None
        EngineConfig.OPENAI_MODEL = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
        EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-nano"
        EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5-mini"
        EngineConfig.OPENAI_MODEL_DOMAIN_KNOWLEDGE = "gpt-5.4-mini"
        EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-nano"
        EngineConfig.OPENAI_MODEL_SYNTH = "gpt-5-mini"
        EngineConfig.OPENAI_MODEL_SYNTH_VARIETY = "gpt-5-nano"
        EngineConfig.OPENAI_MODEL_INTENT_FORMAT = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_INTENT_SCHEMA_REPAIR = "gpt-5.4-nano"
        bu = str(env.get("OPENAI_BASE_URL", "") or "").strip()
        EngineConfig.OPENAI_BASE_URL = bu or "https://api.openai.com/v1"
        MainInitOps._apply_logical_model_env_overrides(env)

    @staticmethod
    def _configure_azure_from_environment(env: Mapping[str, str]) -> None:
        """Populate :class:`EngineConfig` with Azure OpenAI credentials and clear OpenAI token."""
        EngineConfig.LLM_PROVIDER = "azure"
        EngineConfig.AZURE_API_TOKEN = str(env["AZURE_OPENAI_API_KEY"]).strip()
        EngineConfig.API_TOKEN = None
        EngineConfig.AZURE_OPENAI_ENDPOINT = str(env["AZURE_OPENAI_ENDPOINT"]).strip()
        EngineConfig.AZURE_OPENAI_API_VERSION = str(env["AZURE_OPENAI_API_VERSION"]).strip()
        base = str(env.get("AZURE_OPENAI_BASE_URL", "") or "").strip()
        EngineConfig.AZURE_OPENAI_BASE_URL = base or None
        EngineConfig.OPENAI_MODEL = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_INTENT = "gpt-5.4-mini"
        EngineConfig.OPENAI_MODEL_JOIN = "gpt-5.4-nano"
        EngineConfig.OPENAI_MODEL_SCHEMA = "gpt-5-mini"
        EngineConfig.OPENAI_MODEL_DOMAIN_KNOWLEDGE = "gpt-5.4-mini"
        EngineConfig.OPENAI_MODEL_SCHEMA_BASE = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_DDL = "gpt-4.1-nano"
        EngineConfig.OPENAI_MODEL_SYNTH = "gpt-5-mini"
        EngineConfig.OPENAI_MODEL_SYNTH_VARIETY = "gpt-5-nano"
        EngineConfig.OPENAI_MODEL_INTENT_FORMAT = "gpt-4.1-mini"
        EngineConfig.OPENAI_MODEL_INTENT_SCHEMA_REPAIR = "gpt-5.4-nano"
        MainInitOps._apply_logical_model_env_overrides(env)

    @staticmethod
    def _openai_direct_env_complete(env: Mapping[str, str]) -> bool:
        return MainInitOps._env_any_nonempty(env, ("OPENAI_API_KEY",))

    @staticmethod
    def _configure_mock_from_environment(env: Mapping[str, str]) -> None:
        """Bind sandbox LLM replay from a fixtures JSON file."""
        path = str(
            env.get("AETHERDIALECT_SANDBOX_FIXTURES_FILE", "") or env.get("AETHERDIALECT_MOCK_FIXTURES_FILE", "") or ""
        ).strip()
        if not path:
            raise ConfigError(
                "Sandbox LLM requires AETHERDIALECT_SANDBOX_FIXTURES_FILE, "
                "AETHERDIALECT_MOCK_FIXTURES_FILE, or [sandbox]/[mock] fixtures_file in the config file."
            )
        EngineConfig.LLM_PROVIDER = "sandbox"
        EngineConfig.MOCK_FIXTURES_FILE = path
        EngineConfig.API_TOKEN = None
        EngineConfig.AZURE_API_TOKEN = None

    @staticmethod
    def _configure_llm_from_environment(env: Mapping[str, str]) -> None:
        explicit = EngineConfig.normalize_llm_provider(str(env.get("AETHERDIALECT_LLM_PROVIDER", "") or ""))
        if explicit == "sandbox":
            MainInitOps._configure_mock_from_environment(env)
            LLMProvider.clear_llm_clients()
            MockProvider.reset_mock_provider()
            return
        openai_ready = MainInitOps._openai_direct_env_complete(env)
        azure_ready = MainInitOps._env_all_non_empty(env, AZURE_OPENAI_ENV_REQUIRED)
        if not (openai_ready or azure_ready):
            raise ConfigError(
                "LLM is not configured. Set "
                + ", ".join(OPENAI_ENV_REQUIRED)
                + " for OpenAI, or "
                + ", ".join(AZURE_OPENAI_ENV_REQUIRED)
                + " for Azure OpenAI."
            )
        if explicit:
            if explicit not in ("openai", "azure"):
                raise ConfigError(
                    f"Unsupported AETHERDIALECT_LLM_PROVIDER: {explicit!r}. Expected 'openai', 'azure', or 'sandbox'."
                )
            if explicit == "openai":
                if not openai_ready:
                    raise ConfigError(
                        "AETHERDIALECT_LLM_PROVIDER is 'openai' but the OpenAI environment is incomplete."
                    )
                MainInitOps._configure_openai_from_environment(env)
            else:
                if not azure_ready:
                    raise ConfigError(
                        "AETHERDIALECT_LLM_PROVIDER is 'azure' but the Azure OpenAI environment is incomplete."
                    )
                MainInitOps._configure_azure_from_environment(env)
            return
        if openai_ready and azure_ready:
            raise ConfigError(
                "Both OpenAI and Azure OpenAI credentials are available; "
                "set AETHERDIALECT_LLM_PROVIDER or [llm] provider in the config file to 'openai' or 'azure'."
            )
        if openai_ready:
            MainInitOps._configure_openai_from_environment(env)
            return
        if azure_ready:
            MainInitOps._configure_azure_from_environment(env)
            return
        raise ConfigError("LLM is not configured.")

    @staticmethod
    def federation_reuse_kwargs(owner: Any | None, choice_port: InteractiveChoicePort | None) -> dict[str, Any]:
        """Optional federation context for question-level reuse paths."""
        if owner is None or getattr(owner, "_federation_manifest", None) is None:
            return {}
        manifest = getattr(owner, "_federation_manifest", None)
        member_graphs = getattr(owner, "_federation_member_graphs", None)
        stores_by_source: dict[str, TemplateStoreView] = {}
        gate_kwargs_by_source: dict[str, dict[str, Any]] | None = None
        if isinstance(member_graphs, dict) and member_graphs:
            stores_by_source = MainSpaceOps.federation_stores_by_source(
                owner, member_graphs, space_name=MainSpaceOps.session_space_name_for_federation(owner, choice_port)
            )
            if manifest is not None:
                gate_kwargs_by_source = MainSpaceOps.federation_gate_kwargs_by_source(
                    owner, choice_port, manifest, getattr(owner, "_federation_dialects", None)
                )
        return {
            "federation_dir": getattr(owner, "_federation_storage_dir", None),
            "federation_manifest": manifest,
            "federation_mappings": getattr(owner, "_federation_mappings", None),
            "stores_by_source": stores_by_source or None,
            "dialects_by_source": getattr(owner, "_federation_dialects", None),
            "source_runtimes": getattr(owner, "_federation_source_runtimes", None),
            "member_graphs": member_graphs if isinstance(member_graphs, dict) else None,
            "gate_kwargs_by_source": gate_kwargs_by_source,
        }

    @staticmethod
    def federation_contract_kwargs_from_snap(snap: Mapping[str, Any]) -> dict[str, Any]:
        """Derive federation column contract kwargs stored on a completed turn snapshot."""
        federated_bundle = snap.get("federated_bundle")
        federated_plan = snap.get("federated_plan")
        generation_path = snap.get("generation_path")
        if (
            federated_bundle is None
            and federated_plan is None
            and generation_path is not GenerationPath.FEDERATION_PLAN
        ):
            return {}
        kwargs: dict[str, Any] = {"generation_path": GenerationPath.FEDERATION_PLAN}
        if federated_plan is not None:
            kwargs["federated_plan"] = federated_plan
        if federated_bundle is not None:
            kwargs["federated_bundle"] = federated_bundle
        column_names: Sequence[str] | None = None
        if federated_bundle is not None and getattr(federated_bundle, "column_names", None):
            column_names = federated_bundle.column_names
        elif federated_plan is not None:
            residual = federation_residual_column_headers(federated_plan)
            if residual:
                column_names = residual
        if column_names:
            kwargs["column_names"] = column_names
        return kwargs

    @staticmethod
    def _federation_duckdb_schema_for_connection(connection: str) -> str:
        """Map a federation source connection label to the DuckDB schema used for qualification."""
        conn = str(connection or "").strip().lower()
        if conn in {"", "memory", "main", "storefront"}:
            return "main"
        return conn

    @staticmethod
    def _duckdb_runtime_config_for_schema(base_cls: type[EngineRuntimeConfig], schema: str) -> EngineRuntimeConfig:
        """Return a DuckDB runtime config with ``SCHEMA`` pinned to *schema*."""
        runtime_cfg = EngineRuntimeConfig.process_default_for_class(base_cls)
        if schema != "main":
            runtime_cfg = copy.copy(runtime_cfg)
            cast(DuckDBRuntimeConfig, runtime_cfg).SCHEMA = schema
        return runtime_cfg

    @staticmethod
    def _build_federation_source_runtimes(
        manifest: FederationManifest,
        artifacts_root: str | None,
        default_dialect: Any,
        *,
        default_identity: EngineIdentity | None = None,
        native_connection: Any = None,
        sqlalchemy_engine: Any = None,
        engines_by_source: Mapping[str, Any] | None = None,
        native_connections_by_source: Mapping[str, Any] | None = None,
        existing_runtimes: Mapping[str, SourceRuntime] | None = None,
        members_by_source: Mapping[str, Any] | None = None,
    ) -> dict[str, SourceRuntime]:
        """Bind per-source dialect handles for federated SQL generation and execution."""
        runtimes: dict[str, SourceRuntime] = {}
        fallback_identity = default_identity or active_engine_identity()
        sa_by_source = dict(engines_by_source or {})
        native_by_source = dict(native_connections_by_source or {})
        prior_runtimes = dict(existing_runtimes or {})
        members = dict(members_by_source or {})
        for binding in manifest.sources:
            adir = federation_source_artifacts_dir(
                artifacts_root,
                binding,
                federation_id=str(manifest.federation_id or "") or None,
            )
            with artifact_lock(adir):
                engine_type = str(binding.engine or fallback_identity.engine_type).strip().lower()
                member_engine = members.get(binding.source_id)
                member_dialect = getattr(member_engine, "_dialect", None) if member_engine is not None else None
                if member_dialect is not None:
                    schema_path = os.path.join(adir, "schema_graph.json.gz")
                    if hasattr(member_dialect, "_schema_json_path"):
                        member_dialect._schema_json_path = schema_path
                    runtimes[binding.source_id] = SourceRuntime(
                        source_id=binding.source_id,
                        engine=engine_type,
                        connection=str(binding.connection or ""),
                        artifacts_dir=adir,
                        dialect=member_dialect,
                        sqlglot_dialect=DialectRegistry.sqlglot_dialect_for_engine(engine_type),
                        native_connection=native_by_source.get(binding.source_id),
                        sqlalchemy_engine=sa_by_source.get(binding.source_id),
                    )
                    continue
                try:
                    runtime_cfg_cls = MainInitOps._runtime_config_for_engine(engine_type)
                    runtime_cfg = EngineRuntimeConfig.process_default_for_class(runtime_cfg_cls)
                except Exception:
                    identity_runtime = fallback_identity.runtime_config
                    if isinstance(identity_runtime, type):
                        runtime_cfg_cls = identity_runtime
                        runtime_cfg = EngineRuntimeConfig.process_default_for_class(runtime_cfg_cls)
                    else:
                        runtime_cfg = identity_runtime
                        runtime_cfg_cls = type(runtime_cfg)
                if engine_type == "duckdb":
                    runtime_cfg = MainInitOps._duckdb_runtime_config_for_schema(
                        runtime_cfg_cls,
                        MainInitOps._federation_duckdb_schema_for_connection(str(binding.connection or "")),
                    )
                source_sa = sa_by_source.get(binding.source_id, sqlalchemy_engine)
                source_native = native_by_source.get(binding.source_id, native_connection)
                prior = prior_runtimes.get(binding.source_id)
                if (
                    prior is not None
                    and prior.engine == engine_type
                    and prior.connection == str(binding.connection or "")
                ):
                    if prior.native_connection is not None:
                        source_native = prior.native_connection
                    if prior.sqlalchemy_engine is not None:
                        source_sa = prior.sqlalchemy_engine
                try:
                    bound_dialect = DialectRegistry.get_dialect(
                        engine_type, runtime_cfg, sqlalchemy_engine=source_sa, native_connection=source_native
                    )
                except Exception:
                    bound_dialect = default_dialect
                runtimes[binding.source_id] = SourceRuntime(
                    source_id=binding.source_id,
                    engine=engine_type,
                    connection=str(binding.connection or ""),
                    artifacts_dir=adir,
                    dialect=bound_dialect,
                    sqlglot_dialect=DialectRegistry.sqlglot_dialect_for_engine(engine_type),
                    native_connection=source_native,
                    sqlalchemy_engine=source_sa,
                )
        return runtimes

    @staticmethod
    def _read_text_if_file(path: str | None) -> str | None:
        """Return the text content of *path* if it exists and is a regular file, else None."""
        if not path:
            return None
        expanded = os.path.expanduser(str(path))
        if not os.path.isfile(expanded):
            return None
        with open(expanded, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def write_schema_context_cache(artifacts_dir: str, schema_context: EngineContext) -> str:
        """Persist *schema_context* (with sql_file/notes_file text inlined) to *artifacts_dir*. Returns the path of the written cache file."""
        payload: dict[str, Any] = {
            "version": SCHEMA_CONTEXT_CACHE_VERSION,
            "include": schema_context.include,
            "allow_objects": sorted(schema_context.allow_objects),
            "deny_objects": sorted(schema_context.deny_objects),
            "deny_columns": sorted(schema_context.deny_columns),
            "allow_columns": sorted(schema_context.allow_columns),
            "sql_file_original": schema_context.sql_file,
            "notes_file_original": schema_context.notes_file,
            "notes_inline": schema_context.notes is not None,
            "sql_text": MainInitOps._read_text_if_file(schema_context.sql_file),
            "notes_text": notes_content_from_context(schema_context),
        }
        os.makedirs(artifacts_dir, exist_ok=True)
        cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
        MainSpaceOps.write_json_atomic(cache_path, payload)
        return cache_path

    @staticmethod
    def load_schema_context_cache(artifacts_dir: str) -> EngineContext | None:
        """Reload a persisted ``EngineContext`` from *artifacts_dir*. Inlined ``sql_text`` / ``notes_text`` are materialised back to disk inside *artifacts_dir* so downstream consumers that expect file paths continue to work. Returns: The restored ``EngineContext``, or ``None`` when no cache file exists or the file is unreadable / not a JSON object. Raises: ConfigError: When the cache file exists but its ``version`` is not :data:`SCHEMA_CONTEXT_CACHE_VERSION` (including version 3). Delete the cache file (or the engine artifacts directory) and re-run initialization so the cache is rewritten; there is no migration path."""
        cache_path = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHE_NAME)
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        cache_version = payload.get("version")
        if not format_versions_match(cache_version, SCHEMA_CONTEXT_CACHE_VERSION):
            raise ConfigError(
                f"schema context cache at {cache_path!r} has version {cache_version!r}; "
                f"this build expects {SCHEMA_CONTEXT_CACHE_VERSION}. "
                f"Delete {cache_path!r} (or the engine artifacts directory) and re-run "
                f"initialize_aether_engine so the cache is rebuilt from scratch."
            )
        MainSpaceOps.validate_scope_list_fields(payload)
        sql_text = payload.get("sql_text")
        notes_text = payload.get("notes_text")
        sql_file: str | None = None
        notes_file: str | None = None
        notes_inline: str | None = None
        if isinstance(sql_text, str):
            sql_file = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHED_DDL)
            with open(sql_file, "w", encoding="utf-8") as fh:
                fh.write(sql_text)
        if isinstance(notes_text, str):
            if payload.get("notes_inline"):
                notes_inline = notes_text
            else:
                notes_file = os.path.join(artifacts_dir, SCHEMA_CONTEXT_CACHED_NOTES)
                with open(notes_file, "w", encoding="utf-8") as fh:
                    fh.write(notes_text)
        include_raw = payload.get("include", "tables")
        if include_raw not in ("tables", "views"):
            raise ConfigError(f"schema context cache include must be tables or views; got {include_raw!r}")
        return EngineContext(
            allow_objects=frozenset(payload.get("allow_objects") or ()),
            include=include_raw,
            deny_objects=frozenset(payload.get("deny_objects") or ()),
            deny_columns=frozenset(payload.get("deny_columns") or ()),
            allow_columns=frozenset(payload.get("allow_columns") or ()),
            sql_file=sql_file,
            notes_file=notes_file,
            notes=notes_inline,
        )

    @staticmethod
    def credential_visible_object_set(schema_graph: SchemaGraph) -> frozenset[str]:
        """Return table names plus ``table.column`` quals present in a reflected consumer graph."""
        names: set[str] = set(schema_graph.tables.keys())
        for tname, tbl in schema_graph.tables.items():
            for cname in tbl.columns:
                names.add(f"{tname}.{cname}")
        return frozenset(names)

    @staticmethod
    def credential_visibility_fingerprint(visible_tables: frozenset[str] | set[str] | Sequence[str]) -> str:
        """Stable short fingerprint of a sorted credential-visible table set."""
        tables = sorted({str(t).strip() for t in visible_tables if str(t).strip()})
        return hashlib.sha256(",".join(tables).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def probe_credential_visible_tables(
        dialect: Any,
        owner: SchemaGraph,
        *,
        execution_ctx: EngineContext | None = None,
    ) -> frozenset[str]:
        """SQL privilege probe over owner table names (no reclassify)."""
        owner_names = list(owner.tables.keys())
        schema_name = ""
        if hasattr(dialect, "schema_name"):
            try:
                schema_name = str(dialect.schema_name() or "")
            except (AttributeError, TypeError, ValueError, RuntimeError):
                schema_name = ""
        kept = list(dialect.filter_selectable_relation_names(schema_name, owner_names))
        allowed = {str(n) for n in kept if str(n).strip()}
        tindex = {str(n).lower(): n for n in owner.tables}
        canon = {tindex[n.lower()] for n in allowed if n.lower() in tindex}
        if execution_ctx is not None:
            if execution_ctx.allow_objects:
                allow_l = {str(x).lower() for x in execution_ctx.allow_objects}
                canon = {n for n in canon if n.lower() in allow_l}
            if execution_ctx.deny_objects:
                deny_l = {str(x).lower() for x in execution_ctx.deny_objects}
                canon = {n for n in canon if n.lower() not in deny_l}
        return frozenset(canon)

    @staticmethod
    def open_consumer_schema_from_owner_cache(
        dialect: Any,
        owner_snapshot: SchemaGraph,
        *,
        execution_ctx: EngineContext | None = None,
    ) -> tuple[SchemaGraph, frozenset[str]]:
        """Build the consumer working graph as an owner-cache subset via privilege probe."""
        visible_tables = MainInitOps.probe_credential_visible_tables(
            dialect, owner_snapshot, execution_ctx=execution_ctx
        )
        subset = subset_schema_graph_for_visible_tables(
            owner_snapshot,
            visible_tables,
            prefer_base_description=True,
        )
        consumer_visible = MainInitOps.credential_visible_object_set(subset)
        return subset, consumer_visible

    @staticmethod
    def open_consumer_federation_from_owner_cache(
        members: Mapping[str, Any],
        owner_composite: SchemaGraph,
        owner_member_graphs: Mapping[str, SchemaGraph],
        *,
        manifest: FederationManifest,
        mappings: FederationMappings,
        execution_ctx: FederationContext | None = None,
    ) -> tuple[SchemaGraph, frozenset[str]]:
        """Build the consumer federation working graph as an owner- composite subset via member privilege probes. Members may be consumer engines (execution credentials). Physical table visibility is probed on each member against the owner member-graph table list; a composite table is kept only when every contributing physical member table is privilege- visible. Federation context allow/deny then intersects on composite names. Prefers ``base_description`` like single-engine consumer open."""
        refs_by_composite = composite_physical_member_refs(owner_member_graphs, manifest, mappings)
        visible_physical: dict[str, frozenset[str]] = {}
        for source_id, eng in members.items():
            owner_mg = owner_member_graphs.get(str(source_id))
            dialect = getattr(eng, "_dialect", None)
            if owner_mg is None or dialect is None:
                visible_physical[str(source_id)] = frozenset()
                continue
            visible_physical[str(source_id)] = MainInitOps.probe_credential_visible_tables(
                dialect,
                owner_mg,
                execution_ctx=None,
            )
        visible_composite: set[str] = set()
        for cname, refs in refs_by_composite.items():
            if cname not in owner_composite.tables:
                continue
            if not refs:
                continue
            if all(phys in visible_physical.get(src, frozenset()) for src, phys in refs):
                visible_composite.add(cname)
        for cname in owner_composite.tables:
            if cname in visible_composite:
                continue
            if cname in refs_by_composite:
                continue
            table = owner_composite.tables[cname]
            source_id = str(getattr(table, "source_id", "") or "").strip()
            if not source_id:
                continue
            phys = str(getattr(table, "original_name", "") or "").strip() or cname
            if phys in visible_physical.get(source_id, frozenset()):
                visible_composite.add(cname)
        if execution_ctx is not None:
            if execution_ctx.allow_objects:
                allow_l = {str(x).lower() for x in execution_ctx.allow_objects}
                visible_composite = {n for n in visible_composite if n.lower() in allow_l}
            if execution_ctx.deny_objects:
                deny_l = {str(x).lower() for x in execution_ctx.deny_objects}
                visible_composite = {n for n in visible_composite if n.lower() not in deny_l}
        subset = subset_schema_graph_for_visible_tables(
            owner_composite,
            frozenset(visible_composite),
            prefer_base_description=True,
        )
        return subset, MainInitOps.credential_visible_object_set(subset)

    @staticmethod
    def _purge_schema_context_cache(artifacts_dir: str) -> None:
        """Remove the persisted ``schema_context.json`` and any materialised cache files. Used when clearing pre-manifest artifacts so a stale schema context cannot be silently reloaded after a learning-reset rebuild."""
        for name in (SCHEMA_CONTEXT_CACHE_NAME, SCHEMA_CONTEXT_CACHED_DDL, SCHEMA_CONTEXT_CACHED_NOTES):
            fp = os.path.join(artifacts_dir, name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError as exc:
                    debug(f"[main_execution._purge_schema_context_cache] {fp}: {exc}")

    @staticmethod
    def _notify_schema_context_warnings(schema_context: EngineContext, sink: Callable[[str], None]) -> None:
        """Emit non-fatal notices for ambiguous or ineffective scope entries."""
        overlap = schema_context.allow_columns & schema_context.deny_columns
        if overlap:
            n = len(overlap)
            sink(
                f"  Schema scope: allow_columns ∩ deny_columns has {n} duplicate "
                f"entr{'ies' if n != 1 else 'y'}; deny_columns wins for those keys."
            )
        allow = schema_context.allow_objects
        for spec in sorted(schema_context.deny_columns):
            if "." not in spec:
                continue
            tbl, _, _rest = spec.partition(".")
            if tbl == "*":
                continue
            if allow and tbl not in allow:
                sink(
                    f"  Schema scope: deny_columns entry {spec!r} references table {tbl!r} "
                    "outside allow_objects; it never applies under the current scope."
                )

    @staticmethod
    def _upload_validation_config_error(message: str, data_quality_report: object) -> ConfigError:
        """Attach upload validation context to a configuration error."""
        exc = ConfigError(message)
        cast(Any, exc).data_quality_report = data_quality_report
        return exc

    @staticmethod
    def _emit_runtime_config_override_diagnostics(overridden: frozenset[str]) -> None:
        """Emit one diagnostic per runtime-config field whose effective value came from the TOML file over env."""
        for key in sorted(overridden):
            notify(
                f"Runtime config file overrides environment for {key}",
                stage="config",
                code=DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED,
                details=(("key", key),),
            )

    @staticmethod
    def migration_report_for_init(
        artifacts_dir: str,
        prompt_schema: SchemaGraph,
        *,
        schema_role: SchemaRole,
        previous_schema: SchemaGraph | None,
        schema_diff: Any | None,
    ) -> MigrationReport:
        """Resolve template migration during single-engine init; consumers never mutate artifacts."""
        if schema_role == "consumer":
            return MigrationReport(tier=MigrationTier.NO_CHANGE)
        return TemplateOps.apply_migration_policy(
            artifacts_dir,
            prompt_schema,
            allow_destructive=True,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )

    @staticmethod
    def preview_schema_migration(
        *,
        artifacts_dir: str | os.PathLike[str],
        schema_graph: Any,
    ) -> MigrationPreview:
        """Return a read-only migration preview for the live schema graph against stored artifacts."""
        adir = Path(os.fspath(artifacts_dir))
        schema_path = adir / "schema_graph.json.gz"
        previous_schema = load_schema_graph_snapshot(str(schema_path)) if schema_path.is_file() else None
        schema_diff = diff_schemas(previous_schema, schema_graph) if previous_schema is not None else None
        stored = read_artifact_manifest(str(adir))
        tier = classify_migration_tier(stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff)
        if tier in (
            MigrationTier.NO_CHANGE,
            MigrationTier.ADDITIVE,
            MigrationTier.SOFT_REFRESH,
            MigrationTier.PERMISSION_FILTERED,
        ):
            preview_tier: Literal["compatible", "remap", "destructive"] = "compatible"
        elif tier == MigrationTier.REMAP:
            preview_tier = "remap"
        else:
            preview_tier = "destructive"
        affected_tables: tuple[str, ...] = ()
        affected_columns: tuple[tuple[str, str], ...] = ()
        skeleton_document: dict[str, Any] = {}
        if schema_diff is not None:
            affected_tables = tuple(sorted(set(schema_diff.dropped_tables) | set(schema_diff.added_tables)))
            column_pairs: list[tuple[str, str]] = []
            for table_name, table_diff in schema_diff.per_table.items():
                for column_name in table_diff.dropped_columns:
                    column_pairs.append((table_name, column_name))
                for column_name in table_diff.added_columns:
                    column_pairs.append((table_name, column_name))
            affected_columns = tuple(sorted(column_pairs))
        if tier in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            rename_plan = (
                try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
            )
            skeleton_document = TemplateOps.build_schema_migration_map_document(
                tier=tier,
                schema_diff=schema_diff,
                rename_plan=rename_plan,
                previous_schema=previous_schema,
                schema=schema_graph,
            )
            skeleton_document = {k: v for k, v in skeleton_document.items() if k != "version"}
        return MigrationPreview(
            tier=preview_tier,
            affected_tables=affected_tables,
            affected_columns=affected_columns,
            skeleton_document=skeleton_document,
        )

    @staticmethod
    def _emit_artifact_growth_diagnostics(artifacts_dir: str, limits: EngineLimits) -> list[Diagnostic]:
        """Emit growth snapshot and optional near-limit warnings for artifact storage."""
        artifact_bytes = TemplateOps.artifact_directory_byte_size(artifacts_dir)
        template_count, feedback_shard_count, orphan_count = TemplateOps.artifact_growth_counts(artifacts_dir)
        growth = Diagnostic(
            stage="artifact",
            level="info",
            code=DIAGNOSTIC_CODE_ARTIFACT_GROWTH,
            message="Artifact directory growth snapshot",
            details=(
                ("artifact_bytes", str(artifact_bytes)),
                ("template_count", str(template_count)),
                ("feedback_shard_count", str(feedback_shard_count)),
                ("orphan_count", str(orphan_count)),
            ),
            phase="artifact",
        )
        notify(
            growth.message,
            stage=growth.stage,
            code=growth.code,
            level=growth.level,
            details=growth.details,
        )
        diags: list[Diagnostic] = [growth]
        if limits.template_store_max_count is not None:
            cap = int(limits.template_store_max_count)
            if cap > 0 and template_count >= int(cap * 0.9):
                near = Diagnostic(
                    stage="artifact",
                    level=DiagnosticSeverity.WARNING,
                    code=DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
                    message="Template count is within ten percent of template_store_max_count",
                    details=(
                        ("limit", "template_store_max_count"),
                        ("cap", str(cap)),
                        ("current", str(template_count)),
                    ),
                    phase="artifact",
                )
                notify(near.message, stage=near.stage, code=near.code, level=near.level, details=near.details)
                diags.append(near)
        if limits.template_store_max_disk_bytes is not None:
            cap = int(limits.template_store_max_disk_bytes)
            if cap > 0 and artifact_bytes >= int(cap * 0.9):
                near = Diagnostic(
                    stage="artifact",
                    level=DiagnosticSeverity.WARNING,
                    code=DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR,
                    message="Artifact directory size is within ten percent of template_store_max_disk_bytes",
                    details=(
                        ("limit", "template_store_max_disk_bytes"),
                        ("cap", str(cap)),
                        ("current", str(artifact_bytes)),
                    ),
                    phase="artifact",
                )
                notify(near.message, stage=near.stage, code=near.code, level=near.level, details=near.details)
                diags.append(near)
        return diags

    @staticmethod
    def refresh_aether_engine(
        owner: Any,
        *,
        reflect: bool = True,
        log_sink: Callable[[str], None] | None = None,
    ) -> RefreshReport:
        """Re-run post-connection artifact reconciliation for an existing engine. For consumers, credential reflect is the RBAC allowlist for API and execution; the owner snapshot is kept only for LLM/prompt internals when permission-filtered."""
        sink: Callable[[str], None] = log_sink if log_sink is not None else notify
        adir = str(owner._artifacts_dir)
        dialect = owner._dialect
        runtime_cfg = owner._runtime_config
        master_ctx = runtime_cfg.engine_context
        if not isinstance(master_ctx, EngineContext):
            raise ConfigError("refresh requires a single-engine context")
        schema_role = getattr(owner, "_schema_role", SchemaRole.OWNER)
        trust_bundled_baseline = getattr(owner, "_trust_bundled_baseline", False)
        limits = getattr(owner, "_limits", EngineLimits())
        schema_json_path = MainSpaceOps.engine_schema_json_path(adir)
        diagnostics: list[Diagnostic] = list(TemplateOps.collect_orphaned_migration_checkpoints(adir))
        notes_content: str | None = None
        if master_ctx.notes is not None or master_ctx.notes_file:
            notes_content = notes_content_from_context(master_ctx)
        previous_schema = load_schema_graph_snapshot(schema_json_path)
        artifacts_root = Path(adir)
        map_path = artifacts_root / MIGRATION_MAP_FILENAME
        pending_migration_map = None
        consumer_visible_early: frozenset[str] | None = None
        if reflect:
            identity = getattr(owner, "_engine_identity", None)
            if not isinstance(identity, EngineIdentity):
                engine_type = str(getattr(dialect, "name", getattr(owner, "dialect", "")) or "")
                identity = EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg)
            identity_token = push_engine_identity(identity)
            try:
                pending_migration_map = (
                    TemplateOps.load_schema_migration_map(artifacts_root)
                    if map_path.is_file() and schema_role == SchemaRole.OWNER
                    else None
                )
                is_consumer = schema_role in (SchemaRole.CONSUMER, "consumer")
                if is_consumer:
                    if previous_schema is None:
                        raise ConfigError(
                            "Owner schema_graph.json.gz is required before consumer refresh; "
                            "an owner must initialize artifacts first."
                        )
                    schema_graph, consumer_visible_early = MainInitOps.open_consumer_schema_from_owner_cache(
                        dialect,
                        previous_schema,
                        execution_ctx=None,
                    )
                    schema_diff = None
                else:
                    schema_graph, schema_diff = build_schema_graph_with_diff(
                        dialect,
                        master_ctx,
                        notes_content=notes_content,
                        log_sink=sink,
                        refresh_existing_descriptions_on_addition=(
                            pending_migration_map.refresh_existing_descriptions_on_addition
                            if pending_migration_map is not None
                            else False
                        ),
                        force_live_schema_reflect=pending_migration_map is not None,
                        trust_bundled_baseline=trust_bundled_baseline,
                        schema_json_path=schema_json_path,
                        persist_schema_cache=True,
                    )
                if map_path.is_file() and schema_role == SchemaRole.OWNER:
                    loaded = (
                        pending_migration_map
                        if pending_migration_map is not None
                        else TemplateOps.load_schema_migration_map(artifacts_root)
                    )
                    if loaded is not None:
                        try:
                            TemplateOps.validate_schema_migration_map(loaded, previous_schema, schema_graph)
                        except MigrationPendingError as exc:
                            msg = str(exc)
                            if msg.startswith("STALE_MAP:"):
                                try:
                                    map_path.unlink()
                                except OSError:
                                    pass
                                sink("  Removed stale schema_migration_map.json for this snapshot.")
                            else:
                                raise
                        else:
                            if loaded.action == MIGRATION_MAP_ACTION_ABORT:
                                try:
                                    map_path.unlink()
                                except OSError:
                                    pass
                                raise MigrationPendingError("user aborted via migration map")
                            TemplateOps.apply_schema_migration_map(loaded, adir, schema_graph, Path(schema_json_path))
                            ts = datetime.now(UTC).strftime(STRUCTURE_APPLIED_TIMESTAMP_FORMAT)
                            applied_map = map_path.with_name(map_path.stem + ".applied.json")
                            try:
                                if applied_map.is_file():
                                    archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                                    applied_map.rename(archive)
                                map_path.rename(applied_map)
                            except OSError as exc:
                                debug(f"[main_execution.refresh_aether_engine] could not archive migration map: {exc}")
                            previous_schema = load_schema_graph_snapshot(schema_json_path)
                            pending_migration_map = None
                            schema_graph, schema_diff = build_schema_graph_with_diff(
                                dialect,
                                master_ctx,
                                notes_content=notes_content,
                                log_sink=sink,
                                refresh_existing_descriptions_on_addition=False,
                                force_live_schema_reflect=True,
                                trust_bundled_baseline=trust_bundled_baseline,
                                persist_schema_cache=schema_role not in (SchemaRole.CONSUMER, "consumer"),
                            )
            finally:
                pop_engine_identity(identity_token)
        else:
            loaded_graph = load_schema_graph_snapshot(schema_json_path)
            if loaded_graph is None:
                raise ConfigError("artifact-only refresh requires a cached schema graph")
            schema_graph = loaded_graph
            schema_diff = None
            finalize_with_structure(schema_graph, schema_json_path, dialect=dialect)
        owner_snapshot = previous_schema
        stored = read_artifact_manifest(adir)
        if schema_role == SchemaRole.OWNER and stored is not None and not stored.schema_graph_id:
            upgrade_artifacts_schema_graph_id(adir)
            stored = read_artifact_manifest(adir)
        pinned_id = None
        if owner_snapshot is not None:
            pinned_id = str(owner_snapshot.schema_graph_id or "") or None
        if pinned_id is None and stored is not None:
            pinned_id = str(stored.schema_graph_id or "") or None
        assign_schema_graph_hashes(
            schema_graph,
            master_ctx,
            str(getattr(schema_graph, "notes_sha256", "") or ""),
            schema_role=schema_role,
            pinned_schema_graph_id=pinned_id if schema_role == SchemaRole.CONSUMER else None,
        )
        consumer_visible: frozenset[str] | None = None
        prompt_schema = schema_graph
        tier_preview = classify_migration_tier(
            stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff
        )
        if schema_role == SchemaRole.CONSUMER:
            consumer_visible = consumer_visible_early
            if consumer_visible is None:
                consumer_visible = MainInitOps.credential_visible_object_set(schema_graph)
            if owner_snapshot is not None and stored is not None:
                tier_preview = MigrationTier.PERMISSION_FILTERED
        if schema_role == SchemaRole.CONSUMER and consumer_visible is None:
            consumer_visible = frozenset()
        if (
            schema_role == SchemaRole.CONSUMER
            and stored is not None
            and artifact_manifest_incompatible_with_package(stored)
        ):
            raise ConfigError(
                "Artifact manifest is incompatible with this package version; an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == SchemaRole.CONSUMER and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            raise ConfigError(
                "Schema has drifted since artifacts were published; an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == SchemaRole.OWNER and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            rename_plan = (
                try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
            )
            skeleton_document = TemplateOps.build_schema_migration_map_document(
                tier=tier_preview,
                schema_diff=schema_diff,
                rename_plan=rename_plan,
                previous_schema=previous_schema,
                schema=schema_graph,
            )
            skeleton_document = {k: v for k, v in skeleton_document.items() if k != "version"}
            raise MigrationPendingError(
                "Schema migration required: supply a migration map and restart init.",
                skeleton_document=skeleton_document,
            )
        migration_report = MainInitOps.migration_report_for_init(
            adir,
            prompt_schema,
            schema_role=schema_role,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )
        if migration_report.tier != MigrationTier.NO_CHANGE:
            MainInteractiveOps.print_migration_applied(migration_report, sink)
        previous_graph_id = ""
        if stored is not None:
            previous_graph_id = str(stored.schema_graph_id or "")
        elif owner_snapshot is not None:
            previous_graph_id = str(owner_snapshot.schema_graph_id or "")
        active_graph_id = str(prompt_schema.schema_graph_id or "")
        if (
            schema_role == SchemaRole.OWNER
            and previous_graph_id
            and active_graph_id
            and previous_graph_id != active_graph_id
        ):
            MainSpaceOps.orphan_superseded_identity_artifacts_on_rotation(
                adir,
                previous_schema_graph_id=previous_graph_id,
                active_schema_graph_id=active_graph_id,
            )
        if schema_role == SchemaRole.OWNER:
            MainSpaceOps.prune_stale_artifact_auxiliaries(adir, active_schema_graph_id=active_graph_id)
        if schema_role == SchemaRole.CONSUMER:
            owner._consumer_visible_objects = (
                frozenset(consumer_visible) if consumer_visible is not None else frozenset()
            )
            dk = None
            holder = getattr(owner, "_domain_knowledge", None)
            if holder is not None:
                try:
                    dk = holder.entries()
                except Exception:
                    dk = None
            owner._credential_default_space_uid = MainSpaceOps.ensure_credential_default_aetherspace(
                adir,
                prompt_schema,
                owner._consumer_visible_objects,
                engine_domain_knowledge=dk,
            )
        elif consumer_visible is not None:
            owner._consumer_visible_objects = consumer_visible
        MainSpaceOps.bind_owner_default_template_store(owner, prompt_schema, adir, schema_role=schema_role)
        store = owner._store
        reconcile_report = TemplateOps.reconcile_template_store(store, prompt_schema)
        if reconcile_report.dropped_template_ids:
            TemplateOps.save_template_store(store)
        templates = TemplateOps.store_to_templates(store)
        owner._templates = templates
        orphans_removed, bytes_reclaimed = TemplateOps.collect_expired_template_orphans(adir)
        diagnostics.extend(MainInitOps._emit_artifact_growth_diagnostics(adir, limits))
        if schema_role == SchemaRole.OWNER:
            try:
                MainInitOps.write_schema_context_cache(adir, master_ctx)
            except OSError as exc:
                debug(f"[main_execution.refresh_aether_engine] schema_context cache write failed: {exc}")
        owner._schema_graph = schema_graph
        owner._store = store
        owner._templates = templates
        if schema_role == SchemaRole.OWNER:
            load_dk = getattr(owner, "_load_persisted_domain_knowledge", None)
            ingest_dk = getattr(owner, "_ingest_notes_domain_knowledge", None)
            if callable(load_dk) and callable(ingest_dk):
                if not load_dk():
                    ingest_dk()
        schema_terms: set[str] = set(schema_graph.tables.keys())
        for tinfo in schema_graph.tables.values():
            schema_terms.update(tinfo.columns)
            for col in tinfo.columns:
                schema_terms.add(col.lower())
        owner._schema_terms = schema_terms
        owner._schema_stats = schema_graph.schema_stats or {}
        tables_added = tuple(sorted(set(migration_report.added_tables)))
        tables_removed = tuple(sorted(set(migration_report.dropped_tables)))
        columns_added = tuple(sorted(set(migration_report.added_columns)))
        columns_removed: list[tuple[str, str]] = []
        if schema_diff is not None:
            for table_name, table_diff in schema_diff.per_table.items():
                for column_name in table_diff.dropped_columns:
                    columns_removed.append((table_name, column_name))
        columns_removed_t = tuple(sorted(set(columns_removed)))
        schema_changed = schema_diff is not None and not schema_diff.is_empty
        return RefreshReport(
            migration_tier=migration_report.tier,
            schema_changed=schema_changed,
            tables_added=tables_added,
            tables_removed=tables_removed,
            columns_added=columns_added,
            columns_removed=columns_removed_t,
            templates_invalidated=len(reconcile_report.dropped_template_ids),
            orphans_removed=orphans_removed,
            bytes_reclaimed=bytes_reclaimed,
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def overlay_programmatic_connection(merged: dict[str, str], connection: Mapping[str, Any]) -> dict[str, str]:
        """Merge programmatic connection parameters into *merged* without writing ``os.environ``. The reserved ``name`` key (connection identity for federation ``source_id``) is never forwarded as an environment override."""
        for raw_key, raw_val in connection.items():
            key = str(raw_key).strip()
            if not key or key == "name":
                continue
            if raw_val is None:
                continue
            value = str(raw_val).strip()
            if not value:
                continue
            merged[key] = value
        return merged

    _CONNECTION_MAPPING_UNIVERSAL_KEYS = frozenset({"AETHERDIALECT_ENGINE", "AETHERDIALECT_CONNECTION", "name"})

    @staticmethod
    def _validate_connection_mapping_keys(connection: Mapping[str, Any], engine: str) -> None:
        """Raise :class:`ConfigError` when *connection* carries a key outside *engine*'s accepted connection key set."""
        allowed = MainInitOps._runtime_config_for_engine(engine).accepted_connection_keys()
        universal = MainInitOps._CONNECTION_MAPPING_UNIVERSAL_KEYS
        unknown = sorted(
            str(k) for k in connection if str(k).strip() and str(k) not in allowed and str(k) not in universal
        )
        if unknown:
            raise ConfigError(f"connection= contains key(s) not accepted by engine {engine!r}: {', '.join(unknown)}")

    @staticmethod
    def _validate_engine_specific_construction_args(
        active_engine: str,
        *,
        native_connection: Any | None,
        source_selections: Mapping[str, Mapping[str, Any]] | None,
        execution_engine: Any | None,
    ) -> None:
        """Raise :class:`ConfigError` when a constructor argument is supplied for an engine that cannot use it, instead of silently dropping it."""
        engine = (active_engine or "").strip().lower()
        if native_connection is not None and engine not in EMBEDDED_ENGINE_NAMES:
            raise ConfigError(
                f"native_connection is only accepted for embedded engines ({', '.join(sorted(EMBEDDED_ENGINE_NAMES))}); got {active_engine!r}"
            )
        if source_selections and engine not in FILE_ENGINE_NAMES:
            raise ConfigError(
                f"source_selections is only accepted for file engines ({', '.join(sorted(FILE_ENGINE_NAMES))}); got {active_engine!r}"
            )
        if execution_engine is not None:
            dialect_cls = DialectRegistry.get_dialect_class(engine)
            if "sqlalchemy_engine" not in inspect.signature(dialect_cls.__init__).parameters:
                raise ConfigError(f"execution_engine is not accepted for engine {active_engine!r}")

    @staticmethod
    def _split_connection_argument(
        connection: str | Mapping[str, Any] | None,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Return ``(named_connection, programmatic_mapping)`` for engine construction."""
        if connection is None:
            return None, None
        if isinstance(connection, Mapping):
            return None, connection
        named = str(connection).strip()
        return (named or None), None

    @staticmethod
    def initialize_aether_engine(
        engine_context: EngineContext | str | None = None,
        *,
        artifacts_dir: str | None = None,
        config_file: str | os.PathLike[str] | None = None,
        connection: str | Mapping[str, Any] | None = None,
        log_sink: Callable[[str], None] | None = None,
        execution_engine: Any | None = None,
        native_connection: Any | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
        source_selections: Mapping[str, Mapping[str, Any]] | None = None,
        trust_bundled_baseline: bool = False,
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
        limits: EngineLimits | None = None,
        storage_dir: str | None = None,
    ) -> AetherEngineInitResult:
        """Configure the process environment, build the schema graph, migrate templates, and load stores. For consumers, credential reflect is the RBAC allowlist for API and execution; the owner snapshot is kept only for LLM/prompt internals when permission-filtered."""
        sink: Callable[[str], None] = log_sink if log_sink is not None else notify
        sink("Initialising AetherEngine.")
        named_connection, programmatic_connection = MainInitOps._split_connection_argument(connection)
        config_file_values, toml_claimed_keys, named_by_engine = MainInitOps._load_config_file(config_file)
        ssot = config_file is not None and bool(str(config_file).strip())
        merged, toml_diagnostic_keys = MainInitOps._merge_configuration_environment(
            config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
        )
        if programmatic_connection is not None:
            MainInitOps.overlay_programmatic_connection(merged, programmatic_connection)
        selected_preview = MainInitOps._select_engine_name(merged, named_by_engine)
        if programmatic_connection is not None:
            MainInitOps._validate_connection_mapping_keys(programmatic_connection, selected_preview)
        resolved_connection = MainInitOps._select_connection_name(
            merged, named_by_engine, selected_preview, explicit_connection=named_connection
        )
        if resolved_connection and named_by_engine.get(selected_preview):
            connection_values, connection_claimed, _ = MainInitOps._load_config_file(
                config_file, connection=resolved_connection
            )
            config_file_values.update(connection_values)
            toml_claimed_keys = toml_claimed_keys | connection_claimed
            merged, toml_diagnostic_keys = MainInitOps._merge_configuration_environment(
                config_file_values, toml_claimed_keys=toml_claimed_keys if ssot else None
            )
            if programmatic_connection is not None:
                MainInitOps.overlay_programmatic_connection(merged, programmatic_connection)
            merged["AETHERDIALECT_CONNECTION"] = resolved_connection
        MainInitOps._apply_runtime_environments(merged)
        preview_runtime = MainInitOps._runtime_config_for_engine(selected_preview).from_environment(merged)
        adir = MainInitOps.compute_engine_storage_dir(
            artifacts_dir,
            selected_preview,
            runtime=preview_runtime,
            storage_dir=storage_dir,
        )
        warn_if_artifacts_dir_not_local(adir)
        try:
            cached_master = MainInitOps.load_schema_context_cache(adir)
        except ConfigError as exc:
            sink(str(exc))
            cached_master = None
        prepare_master: EngineContext | None = None
        if isinstance(engine_context, FederationContext):
            raise ConfigError(
                "initialize_aether_engine does not accept FederationContext; use AetherFederation instead"
            )
        if isinstance(engine_context, EngineContext):
            prepare_master = MainInitOps._prepare_schema_context_for_init(engine_context, adir, sink)
        master_ctx, active_ctx, context_name = MainSpaceOps.resolve_engine_context_plan(
            engine_context, adir, schema_role=schema_role, load_master=cached_master, prepare_master=prepare_master
        )
        MainInitOps._notify_schema_context_warnings(master_ctx, sink)
        active_engine, active_runtime = MainInitOps.configure_runtime_from_environment(master_ctx, merged)
        MainInitOps._validate_engine_specific_construction_args(
            active_engine,
            native_connection=native_connection,
            source_selections=source_selections,
            execution_engine=execution_engine,
        )
        engine_identity = EngineIdentity(engine_type=active_engine, runtime_config=active_runtime)
        construction_orphan_token = bind_construction_orphan_identity(engine_identity)
        try:
            llm_exec = load_runtime_config(merged_env=merged)
        except ValueError as exc:
            release_construction_orphan_identity(construction_orphan_token)
            raise ConfigError(str(exc)) from exc
        MainInitOps._emit_runtime_config_override_diagnostics(toml_diagnostic_keys)
        if EngineConfig.LLM_PROVIDER == "azure":
            MainInitOps.validate_azure_llm_execution(llm_exec)
        _rt = active_runtime
        _rt_name = type(_rt).__name__.lower()
        if _rt_name.endswith("runtimeconfig"):
            _rt_name = _rt_name[: -len("runtimeconfig")]
        runtime_label = _rt_name or "default"
        sink(f"  Engine: {active_engine} ({runtime_label}).")
        os.makedirs(adir, exist_ok=True)
        legacy_files = detect_legacy_artifacts(adir)
        if legacy_files:
            sink(f"  Detected pre-manifest artifacts (no manifest): {', '.join(legacy_files)}. Rebuilding caches.")
            wipe_versioned_artifacts(adir)
            MainInitOps._purge_schema_context_cache(adir)
        schema_json_path = os.path.join(adir, "schema_graph.json.gz")
        template_store_dir = TemplateOps.template_store_dir_for_space(adir, MASTER_AETHERSPACE_NAME)
        MainSpaceOps.register_engine_artifact_state(
            adir,
            schema_json_path=schema_json_path,
            template_store_dir=template_store_dir,
        )
        TemplateOps.ensure_template_store_space_layout(adir)
        QSimConfig.SKELETONS_JSON_PATH = os.path.join(adir, "qsim_skeletons.json.gz")
        data_quality_report: DataQualityReport | None = None
        if (active_engine or "").strip().lower() in FILE_ENGINE_NAMES:
            csv_runtime = cast(CsvRuntimeConfig, active_runtime)
            upload_paths = csv_runtime.resolve_source_files()
            selections = parse_source_selections(source_selections or csv_runtime.SOURCE_SELECTIONS)
            data_quality_report = validate_upload_sources(upload_paths, log_sink=sink, source_selections=selections)
            if data_quality_report.requires_review and not selections:
                raise MainInitOps._upload_validation_config_error(
                    f"{data_quality_report.narrative} "
                    "Call inspect_tabular_upload and pass source_selections with the accepted interpretation.",
                    data_quality_report,
                )
            if not data_quality_report.ok:
                raise MainInitOps._upload_validation_config_error(data_quality_report.narrative, data_quality_report)
            if source_selections:
                csv_runtime.set_source_selections(source_selections)
                data_quality_report = DataQualityReport(
                    ok=data_quality_report.ok,
                    issues=data_quality_report.issues,
                    narrative=data_quality_report.narrative,
                    suggested_selections=data_quality_report.suggested_selections,
                    confirmed_selections=cast(Any, {k: dict(v) for k, v in source_selections.items()}),
                )
        if token_provider is not None:
            active_runtime = MainInitOps.apply_connection_credentials_for_engine(
                active_engine,
                MainInitOps.resolve_connection_credentials(None, token_provider),
                runtime=active_runtime,
            )
        try:
            dialect = DialectRegistry.get_dialect(
                active_engine,
                active_runtime,
                sqlalchemy_engine=execution_engine,
                native_connection=native_connection,
                limits=limits if limits is not None else EngineLimits(),
            )
        except DatabaseConnectionError:
            raise
        except Exception as exc:
            raise DatabaseConnectionError(str(exc)) from exc
        notes_content: str | None = None
        if master_ctx.notes is not None or master_ctx.notes_file:
            notes_content = notes_content_from_context(master_ctx)
        previous_schema = load_schema_graph_snapshot(schema_json_path)
        TemplateOps.restore_leftover_migration_checkpoints_on_init(adir, schema_json_path=Path(schema_json_path))
        artifacts_root = Path(adir)
        map_path = artifacts_root / MIGRATION_MAP_FILENAME
        pending_migration_map = (
            TemplateOps.load_schema_migration_map(artifacts_root)
            if map_path.is_file() and schema_role == "owner"
            else None
        )
        engine_identity = EngineIdentity(engine_type=active_engine, runtime_config=active_runtime)
        identity_token = push_engine_identity(engine_identity)
        is_consumer = schema_role in (SchemaRole.CONSUMER, "consumer")
        schema_diff = None
        consumer_visible_early: frozenset[str] | None = None
        try:
            if is_consumer:
                if previous_schema is None:
                    raise ConfigError(
                        "Owner schema_graph.json.gz is required before consumer init; "
                        "an owner must initialize artifacts first."
                    )
                schema_graph, consumer_visible_early = MainInitOps.open_consumer_schema_from_owner_cache(
                    dialect,
                    previous_schema,
                    execution_ctx=None,
                )
            else:
                schema_graph, schema_diff = build_schema_graph_with_diff(
                    dialect,
                    master_ctx,
                    notes_content=notes_content,
                    log_sink=sink,
                    refresh_existing_descriptions_on_addition=(
                        pending_migration_map.refresh_existing_descriptions_on_addition
                        if pending_migration_map is not None
                        else False
                    ),
                    force_live_schema_reflect=pending_migration_map is not None,
                    trust_bundled_baseline=trust_bundled_baseline,
                    schema_json_path=schema_json_path,
                    persist_schema_cache=True,
                )
        finally:
            pop_engine_identity(identity_token)
        stored = read_artifact_manifest(adir)
        if map_path.is_file() and schema_role == "owner":
            loaded = (
                pending_migration_map
                if pending_migration_map is not None
                else TemplateOps.load_schema_migration_map(artifacts_root)
            )
            if loaded is not None:
                try:
                    TemplateOps.validate_schema_migration_map(loaded, previous_schema, schema_graph)
                except MigrationPendingError as exc:
                    msg = str(exc)
                    if msg.startswith("STALE_MAP:"):
                        try:
                            map_path.unlink()
                        except OSError:
                            pass
                        sink("  Removed stale schema_migration_map.json for this snapshot.")
                    else:
                        raise
                else:
                    if loaded.action == MIGRATION_MAP_ACTION_ABORT:
                        try:
                            map_path.unlink()
                        except OSError:
                            pass
                        raise MigrationPendingError("user aborted via migration map")
                    TemplateOps.apply_schema_migration_map(loaded, adir, schema_graph, Path(schema_json_path))
                    ts = datetime.now(UTC).strftime(STRUCTURE_APPLIED_TIMESTAMP_FORMAT)
                    applied_map = map_path.with_name(map_path.stem + ".applied.json")
                    try:
                        if applied_map.is_file():
                            archive = applied_map.with_name(applied_map.stem + f".{ts}" + applied_map.suffix)
                            applied_map.rename(archive)
                        map_path.rename(applied_map)
                    except OSError as exc:
                        debug(f"[main_execution.initialize_aether_engine] could not archive migration map: {exc}")
                    previous_schema = load_schema_graph_snapshot(schema_json_path)
                    pending_migration_map = None
                    identity_token = push_engine_identity(engine_identity)
                    try:
                        schema_graph, schema_diff = build_schema_graph_with_diff(
                            dialect,
                            master_ctx,
                            notes_content=notes_content,
                            log_sink=sink,
                            refresh_existing_descriptions_on_addition=False,
                            force_live_schema_reflect=True,
                            trust_bundled_baseline=trust_bundled_baseline,
                            persist_schema_cache=schema_role not in (SchemaRole.CONSUMER, "consumer"),
                        )
                    finally:
                        pop_engine_identity(identity_token)
                    stored = read_artifact_manifest(adir)
        owner_snapshot = previous_schema
        stored = read_artifact_manifest(adir)
        if schema_role == "owner" and stored is not None and not stored.schema_graph_id:
            upgrade_artifacts_schema_graph_id(adir)
            stored = read_artifact_manifest(adir)
        pinned_id = None
        if owner_snapshot is not None:
            pinned_id = str(owner_snapshot.schema_graph_id or "") or None
        if pinned_id is None and stored is not None:
            pinned_id = str(stored.schema_graph_id or "") or None
        assign_schema_graph_hashes(
            schema_graph,
            master_ctx,
            str(getattr(schema_graph, "notes_sha256", "") or ""),
            schema_role=schema_role,
            pinned_schema_graph_id=pinned_id if schema_role == "consumer" else None,
        )
        consumer_visible: frozenset[str] | None = None
        prompt_schema = schema_graph
        tier_preview = classify_migration_tier(
            stored, schema_graph, previous_schema=previous_schema, schema_diff=schema_diff
        )
        if schema_role == "consumer":
            consumer_visible = consumer_visible_early
            if consumer_visible is None:
                consumer_visible = MainInitOps.credential_visible_object_set(schema_graph)
            if owner_snapshot is not None and stored is not None:
                tier_preview = MigrationTier.PERMISSION_FILTERED
        if schema_role == "consumer" and stored is not None and artifact_manifest_incompatible_with_package(stored):
            raise ConfigError(
                "Artifact manifest is incompatible with this package version; an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == "consumer" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            raise ConfigError(
                "Schema has drifted since artifacts were published; an owner must refresh artifacts before consumer init can proceed."
            )
        if schema_role == "owner" and tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE):
            rename_plan = (
                try_rename_migration_plan(previous_schema, schema_graph) if previous_schema is not None else None
            )
            skeleton_document = TemplateOps.build_schema_migration_map_document(
                tier=tier_preview,
                schema_diff=schema_diff,
                rename_plan=rename_plan,
                previous_schema=previous_schema,
                schema=schema_graph,
            )
            skeleton_document = {k: v for k, v in skeleton_document.items() if k != "version"}
            raise MigrationPendingError(
                "Schema migration required: supply a migration map and restart init.",
                skeleton_document=skeleton_document,
            )
        migration_report = MainInitOps.migration_report_for_init(
            adir,
            prompt_schema,
            schema_role=schema_role,
            previous_schema=previous_schema,
            schema_diff=schema_diff,
        )
        if migration_report.tier != MigrationTier.NO_CHANGE:
            MainInteractiveOps.print_migration_applied(migration_report, sink)
        if schema_role == "owner":
            MainSpaceOps.prune_stale_artifact_auxiliaries(
                adir, active_schema_graph_id=str(prompt_schema.schema_graph_id)
            )
        store = TemplateOps.load_template_store(
            prompt_schema.schema_graph_id, prompt_schema, space_name=MASTER_AETHERSPACE_NAME, artifacts_dir=adir
        )
        templates = TemplateOps.store_to_templates(store)
        rejected: dict[str, Any] = {}
        sink(f"  Templates: {len(templates)} reusable, {len(rejected)} rejected.")
        schema_terms: set[str] = set(schema_graph.tables.keys())
        for tinfo in schema_graph.tables.values():
            schema_terms.update(tinfo.columns)
            for col in tinfo.columns:
                schema_terms.add(col.lower())
        schema_stats = schema_graph.schema_stats or {}
        if EngineConfig.LLM_PROVIDER == "azure":
            prov: Literal["openai", "azure", "sandbox"] = "azure"
        elif EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
            prov = "sandbox"
        else:
            prov = "openai"
        llm_config = LLMConfig(provider=prov)
        if context_name != MASTER_AETHERSPACE_NAME:
            MainSpaceOps.validate_named_context_subset(master_ctx, active_ctx, schema_graph)
        execution_ctx = MainSpaceOps.effective_execution_context(master_ctx, active_ctx, context_name)
        if schema_role == "consumer" and isinstance(execution_ctx, EngineContext) and previous_schema is not None:
            tables = set(schema_graph.tables.keys())
            if execution_ctx.allow_objects:
                allow_l = {str(x).lower() for x in execution_ctx.allow_objects}
                tables = {n for n in tables if n.lower() in allow_l}
            if execution_ctx.deny_objects:
                deny_l = {str(x).lower() for x in execution_ctx.deny_objects}
                tables = {n for n in tables if n.lower() not in deny_l}
            if tables != set(schema_graph.tables.keys()):
                schema_graph = subset_schema_graph_for_visible_tables(
                    previous_schema,
                    frozenset(tables),
                    prefer_base_description=True,
                )
                prompt_schema = schema_graph
                consumer_visible = MainInitOps.credential_visible_object_set(schema_graph)
                schema_terms = set(schema_graph.tables.keys())
                for tinfo in schema_graph.tables.values():
                    schema_terms.update(tinfo.columns)
                    for col in tinfo.columns:
                        schema_terms.add(col.lower())
                schema_stats = schema_graph.schema_stats or {}
        if schema_role == "consumer" and consumer_visible is None and getattr(execution_ctx, "allow_objects", None):
            consumer_visible = frozenset(execution_ctx.allow_objects)
        if schema_role == "consumer" and consumer_visible is None:
            consumer_visible = frozenset()
        runtime_config = RuntimeConfig(
            engine=active_engine,
            artifacts_dir=adir,
            engine_context=master_ctx,
            llm_execution=llm_exec,
            execution_context=execution_ctx,
        )
        if schema_role == "owner":
            try:
                MainInitOps.write_schema_context_cache(adir, master_ctx)
            except OSError as exc:
                debug(f"[main_execution.initialize_aether_engine] schema_context cache write failed: {exc}")
        sink("Ready.")
        release_construction_orphan_identity(construction_orphan_token)
        return AetherEngineInitResult(
            runtime_config=runtime_config,
            llm_config=llm_config,
            schema_graph=schema_graph,
            dialect=dialect,
            artifacts_dir=adir,
            store=store,
            templates=cast(dict[str, Any], templates),
            rejected=rejected,
            schema_terms=schema_terms,
            schema_stats=schema_stats,
            schema_role=schema_role,
            consumer_visible_objects=consumer_visible,
            context_name=context_name,
            execution_context=execution_ctx,
            data_quality_report=data_quality_report,
            federation_manifest=None,
            federation_mappings=None,
            federation_member_graphs=None,
            federation_storage_dir=None,
            federation_source_runtimes=None,
            federation_mapping_suggestions=(),
            federation_dialects_by_source=None,
            engine_identity=engine_identity,
        )

    @staticmethod
    def initialize_aether_federation(
        name: str,
        *,
        members: Sequence[Any] | Mapping[str, Any],
        declaration_file: str | None = None,
        declaration: tuple[FederationManifest, FederationMappings] | None = None,
        artifacts_dir: str | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
        master_context: FederationContext | None = None,
        log_sink: Callable[[str], None] | None = None,
    ) -> AetherFederationInitResult:
        """Compose a federated schema graph from member engines and persist the federation tree."""
        sink: Callable[[str], None] = log_sink if log_sink is not None else notify
        sink(f"Initialising AetherFederation {name!r}.")
        cleanup_abandoned_federation_spill_directories()
        member_dict = federation_members_mapping(members)
        validate_federation_file_members(member_dict)
        if declaration is not None:
            authored_manifest, fed_mappings = declaration
        elif declaration_file is not None and str(declaration_file).strip():
            authored_manifest, fed_mappings = load_federation_declaration_from_path(declaration_file)
        else:
            raise ConfigError("AetherFederation requires declaration or declaration_file")
        probe_federation_member_connections(member_dict, manifest=authored_manifest, mappings=fed_mappings)
        fed_id = str(name).strip()
        if not fed_id:
            raise ConfigError("AetherFederation name must be non-empty")
        if authored_manifest.federation_id != fed_id:
            raise ConfigError(
                f"federation name {fed_id!r} disagrees with manifest federation_id {authored_manifest.federation_id!r}"
            )
        fed_member_graphs_dict = member_graphs_from_engines(member_dict)
        member_source_ids = set(member_dict)
        authored_manifest, fed_mappings = reconcile_authored_declaration_for_members(
            authored_manifest,
            fed_mappings,
            active_source_ids=member_source_ids,
        )
        fed_manifest = build_federation_manifest_from_members(
            member_dict,
            declaration=authored_manifest,
            member_graphs=fed_member_graphs_dict,
            mappings=fed_mappings,
            require_owner_members=schema_role != "consumer",
        )
        active_source_ids = {binding.source_id for binding in fed_manifest.sources}
        fed_manifest = prune_federation_aliases(fed_manifest, active_source_ids=active_source_ids)
        fed_manifest = prune_cross_source_joins(fed_manifest, active_source_ids=active_source_ids)
        fed_mappings = prune_federation_mappings(fed_mappings, fed_manifest, active_source_ids=active_source_ids)
        validate_manifest_cross_source_joins(fed_manifest)
        fed_storage_dir = compute_federation_storage_dir(
            artifacts_dir,
            fed_manifest.federation_id,
        )
        if artifacts_dir:
            os.makedirs(fed_storage_dir, exist_ok=True)
            MainSpaceOps.prune_orphaned_federation_trees(
                os.path.dirname(fed_storage_dir), active_fed_dir=fed_storage_dir
            )
        with artifact_lock(fed_storage_dir):
            loaded_member_graphs = load_federation_member_graphs(artifacts_dir, fed_manifest)
            if loaded_member_graphs:
                recorded_ids = recorded_federation_source_ids(fed_storage_dir)
                topology_change = (
                    detect_federation_topology_change(recorded_ids, fed_manifest) if recorded_ids else "none"
                )
                if topology_change != "add":
                    assert_federation_member_graph_roster_complete(fed_manifest, loaded_member_graphs)
            fed_member_graphs_dict = reconcile_federation_member_graphs(
                fed_member_graphs_dict, loaded_member_graphs, fed_manifest
            )
            for source_id, member_graph in fed_member_graphs_dict.items():
                member_engine = member_dict.get(source_id)
                member_ctx = EngineContext()
                if member_engine is not None:
                    runtime_cfg = getattr(member_engine, "_runtime_config", None)
                    ctx = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
                    if isinstance(ctx, EngineContext):
                        member_ctx = ctx
                raise_if_schema_unusable(member_graph, member_ctx, federation_composite=False)
            try:
                coord_dialect = DialectRegistry.get_dialect("duckdb", DuckDBRuntimeConfig)
            except Exception as exc:
                raise FederationConfigError(
                    f"federation coordinator dialect resolution failed for engine 'duckdb': {exc}"
                ) from exc
            llm_exec = load_runtime_config(merged_env=dict(os.environ))
            if EngineConfig.LLM_PROVIDER == "azure":
                prov: Literal["openai", "azure", "sandbox"] = "azure"
            elif EngineConfig.is_sandbox_llm_provider(EngineConfig.LLM_PROVIDER):
                prov = "sandbox"
            else:
                prov = "openai"
            llm_config = LLMConfig(provider=prov)
            fed_master_ctx = master_context or FederationContext()
            master_ctx = fed_master_ctx
            execution_ctx = fed_master_ctx
            context_name = MASTER_AETHERSPACE_NAME
            engine_identity = EngineIdentity(engine_type="duckdb", runtime_config=DuckDBRuntimeConfig)
            consumer_visible: frozenset[str] | None = None
            if schema_role == "consumer":
                stored = read_artifact_manifest(fed_storage_dir)
                if stored is None:
                    raise ConfigError("Owner federation artifacts are required before consumer init can proceed.")
                if artifact_manifest_incompatible_with_package(stored):
                    raise ConfigError(
                        "Federation artifact manifest is incompatible with this package version; "
                        "an owner must refresh artifacts before consumer init can proceed."
                    )
                owner_composite = load_federation_composite_graph(fed_storage_dir)
                if owner_composite is None:
                    raise ConfigError("Owner federation composite is required before consumer init can proceed.")
                if not loaded_member_graphs:
                    raise ConfigError("Owner federation member graphs are required before consumer init can proceed.")
                fed_member_graphs_dict = dict(loaded_member_graphs)
                schema_graph, consumer_visible = MainInitOps.open_consumer_federation_from_owner_cache(
                    member_dict,
                    owner_composite,
                    fed_member_graphs_dict,
                    manifest=fed_manifest,
                    mappings=fed_mappings,
                    execution_ctx=fed_master_ctx,
                )
                for source_id, member_graph in fed_member_graphs_dict.items():
                    engine = ""
                    for binding in fed_manifest.sources:
                        if binding.source_id == source_id:
                            engine = str(binding.engine or "").strip().lower()
                            break
                    stamp_federation_member_graph(
                        member_graph,
                        federation_id=fed_manifest.federation_id,
                        source_id=source_id,
                        engine=engine,
                    )
                object.__setattr__(
                    schema_graph,
                    "_database_feature_capability_cache",
                    intersect_member_database_feature_capabilities(fed_member_graphs_dict),
                )
            else:
                notes_content: str | None = None
                if fed_master_ctx.notes is not None or fed_master_ctx.notes_file:
                    notes_content = notes_content_from_context(fed_master_ctx)
                if notes_content:
                    for binding in fed_manifest.sources:
                        token = str(binding.source_id or "").strip()
                        if token and token in notes_content:
                            raise ConfigError(f"federation notes must not name a source or member; found {token!r}")
                source_token_ids = [binding.source_id for binding in fed_manifest.sources]
                scrub_federation_member_description_source_tokens(fed_member_graphs_dict, source_token_ids)
                for source_id, member_engine in member_dict.items():
                    engine_graph = getattr(member_engine, "_schema_graph", None)
                    if engine_graph is not None:
                        scrub_federation_member_description_source_tokens({source_id: engine_graph}, source_token_ids)
                raise_if_descriptions_name_federation_sources(
                    fed_member_graphs_dict,
                    source_token_ids,
                )
                source_ids = source_token_ids
                for _, member_engine in member_dict.items():
                    member_ctx = EngineContext()
                    runtime_cfg = getattr(member_engine, "_runtime_config", None)
                    ctx = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
                    if isinstance(ctx, EngineContext):
                        member_ctx = ctx
                    raise_if_member_notes_name_federation_sources(member_ctx.notes_file, source_ids)
                    member_notes = notes_content_from_context(member_ctx)
                    if member_notes:
                        for token in source_ids:
                            sid = str(token or "").strip()
                            if sid and sid in member_notes:
                                raise ConfigError(f"federation notes must not name a source or member; found {sid!r}")
                federation_artifacts_root = Path(fed_storage_dir)
                recorded_source_ids = recorded_federation_source_ids(fed_storage_dir)
                topo_report: FederationTopologyReport | None = None
                topology_shrink_only = False
                if recorded_source_ids:
                    fed_manifest, fed_mappings, topo_report = reconcile_federation_topology(
                        fed_manifest, fed_mappings, recorded_source_ids, federation_dir=fed_storage_dir
                    )
                    if topo_report.change != "none":
                        added = ", ".join(topo_report.added_source_ids) or "none"
                        removed = ", ".join(topo_report.removed_source_ids) or "none"
                        sink(
                            "  Federation topology change "
                            f"{topo_report.change!r}: added=[{added}] removed=[{removed}] "
                            f"plan_templates_invalidated={topo_report.plan_templates_invalidated}"
                        )
                    if topo_report.removed_source_ids:
                        purge_departed_federation_member_trees(
                            fed_storage_dir,
                            artifacts_root=artifacts_dir,
                            removed_source_ids=topo_report.removed_source_ids,
                        )
                topology_shrink_only = (
                    topo_report is not None and topo_report.change == "remove" and not topo_report.added_source_ids
                )
                fed_map_path = federation_artifacts_root / FEDERATION_MIGRATION_MAP_FILENAME
                pending_fed_map_archive: Path | None = None
                if fed_map_path.is_file():
                    fed_loaded = load_federation_migration_map(str(fed_map_path))
                    if fed_loaded is not None:
                        try:
                            validate_federation_migration_map(
                                fed_loaded,
                                cached_member_graphs=loaded_member_graphs,
                                live_member_graphs=fed_member_graphs_dict,
                                manifest=fed_manifest,
                            )
                        except MigrationPendingError as exc:
                            msg = str(exc)
                            if msg.startswith("STALE_MAP:"):
                                try:
                                    fed_map_path.unlink()
                                except OSError:
                                    pass
                                sink("  Removed stale federation_migration_map.json for this snapshot.")
                            else:
                                raise
                        else:
                            if fed_loaded.action == MIGRATION_MAP_ACTION_ABORT:
                                try:
                                    fed_map_path.unlink()
                                except OSError:
                                    pass
                                raise MigrationPendingError("user aborted via federation migration map")
                            fed_manifest, fed_mappings = apply_federation_migration_map(
                                fed_loaded, fed_manifest, fed_mappings, fed_storage_dir
                            )
                            pending_fed_map_archive = fed_map_path
                fed_mapping_suggestions = cached_or_suggest_cross_source_mappings(
                    fed_member_graphs_dict, fed_manifest, fed_storage_dir, existing_mappings=fed_mappings
                )
                sa_by_source = {
                    source_id: getattr(engine, "_execution_engine", None) for source_id, engine in member_dict.items()
                }
                native_by_source = {
                    source_id: getattr(engine, "_native_connection", None) for source_id, engine in member_dict.items()
                }
                default_dialect = coord_dialect
                fed_source_runtimes = MainInitOps._build_federation_source_runtimes(
                    fed_manifest,
                    artifacts_dir,
                    default_dialect,
                    default_identity=engine_identity,
                    engines_by_source=sa_by_source,
                    native_connections_by_source=native_by_source,
                    members_by_source=member_dict,
                )
                fed_dialects_by_source = {
                    source_id: runtime.dialect for source_id, runtime in fed_source_runtimes.items()
                }
                llm_classify = llm_classify_schema if notes_content else None
                federation_format_stale = False
                try:
                    replay_ok = mappings_replay_matches(
                        fed_storage_dir, fed_member_graphs_dict, fed_manifest, fed_mappings
                    )
                except FederationConfigError as exc:
                    sink(str(exc))
                    replay_ok = False
                    federation_format_stale = True
                broken_joins = detect_broken_cross_source_joins(fed_member_graphs_dict, fed_manifest)
                has_persisted_federation = os.path.isfile(
                    federation_artifact_paths(fed_storage_dir)["artifact_manifest"]
                )
                if broken_joins and has_persisted_federation:
                    skeleton_document = {
                        k: v
                        for k, v in build_federation_migration_map_document(dropped_joins=broken_joins).items()
                        if k != "version"
                    }
                    sink("  Federation cross-source join columns missing; supply a federation migration map.")
                    raise MigrationPendingError(
                        "Federation migration required: supply a migration map and restart init.",
                        skeleton_document=skeleton_document,
                    )
                if not replay_ok and not federation_format_stale and has_persisted_federation:
                    prune_federation_plan_templates_on_drift(
                        fed_storage_dir, fed_member_graphs_dict, fed_manifest, fed_mappings
                    )
                    skeleton_document = {
                        k: v for k, v in build_federation_migration_map_document().items() if k != "version"
                    }
                    sink("  Federation drift detected; supply a federation migration map.")
                    raise MigrationPendingError(
                        "Federation migration required: supply a migration map and restart init.",
                        skeleton_document=skeleton_document,
                    )
                schema_graph = compose_composite_graph(
                    fed_member_graphs_dict,
                    fed_manifest,
                    fed_mappings,
                    notes_content=notes_content,
                    llm_classify=llm_classify,
                    master_context=fed_master_ctx,
                )
                validate_cross_source_keys_on_graph(schema_graph, fed_manifest, fed_mappings)
                for source_id, member_graph in fed_member_graphs_dict.items():
                    engine = ""
                    for binding in fed_manifest.sources:
                        if binding.source_id == source_id:
                            engine = str(binding.engine or "").strip().lower()
                            break
                    stamp_federation_member_graph(
                        member_graph,
                        federation_id=fed_manifest.federation_id,
                        source_id=source_id,
                        engine=engine,
                    )
                object.__setattr__(
                    schema_graph,
                    "_database_feature_capability_cache",
                    intersect_member_database_feature_capabilities(fed_member_graphs_dict),
                )
                notes_sha = str(getattr(schema_graph, "notes_sha256", "") or "")
                assign_schema_graph_hashes(
                    schema_graph,
                    master_ctx,
                    notes_sha,
                    schema_role=schema_role,
                    federation_scope_hash=schema_graph.scope_hash or None,
                )
                stored = read_artifact_manifest(fed_storage_dir)
                previous_composite = load_federation_composite_graph(fed_storage_dir) if stored is not None else None
                tier_preview = federation_composite_migration_tier(
                    fed_storage_dir, schema_graph, previous_composite=previous_composite
                )
                if tier_preview in (MigrationTier.REMAP, MigrationTier.DESTRUCTIVE) and not topology_shrink_only:
                    skeleton_document = {
                        k: v for k, v in build_federation_migration_map_document().items() if k != "version"
                    }
                    sink(f"  Federation composite drift ({tier_preview.value}); supply a federation migration map.")
                    raise MigrationPendingError(
                        "Federation migration required: supply a migration map and restart init.",
                        skeleton_document=skeleton_document,
                    )
                migration_report = TemplateOps.apply_federation_composite_migration_policy(
                    fed_storage_dir,
                    schema_graph,
                    allow_destructive=True,
                    previous_composite=previous_composite,
                )
                if migration_report.tier != MigrationTier.NO_CHANGE:
                    MainInteractiveOps.print_migration_applied(migration_report, sink)
                MainSpaceOps.prune_stale_artifact_auxiliaries(
                    fed_storage_dir, active_schema_graph_id=str(schema_graph.schema_graph_id)
                )
                persist_federation_tree(
                    fed_storage_dir,
                    manifest=fed_manifest,
                    mappings=fed_mappings,
                    composite=schema_graph,
                    member_graphs=fed_member_graphs_dict,
                )
                if pending_fed_map_archive is not None:
                    archive_federation_migration_map_file(pending_fed_map_archive, archive_dir=fed_storage_dir)
            sa_by_source = {
                source_id: getattr(engine, "_execution_engine", None) for source_id, engine in member_dict.items()
            }
            native_by_source = {
                source_id: getattr(engine, "_native_connection", None) for source_id, engine in member_dict.items()
            }
            fed_source_runtimes = MainInitOps._build_federation_source_runtimes(
                fed_manifest,
                artifacts_dir,
                coord_dialect,
                default_identity=engine_identity,
                engines_by_source=sa_by_source,
                native_connections_by_source=native_by_source,
                members_by_source=member_dict,
            )
            fed_dialects_by_source = {source_id: runtime.dialect for source_id, runtime in fed_source_runtimes.items()}
            if schema_role == "consumer":
                fed_mapping_suggestions = ()

        store = TemplateOps.load_template_store(
            schema_graph.schema_graph_id,
            schema_graph,
            space_name=MASTER_AETHERSPACE_NAME,
            artifacts_dir=fed_storage_dir,
        )
        templates = TemplateOps.store_to_templates(store)
        rejected: dict[str, Any] = {}
        sink(f"  Templates: {len(templates)} reusable, {len(rejected)} rejected.")
        schema_terms: set[str] = set(schema_graph.tables.keys())
        for tinfo in schema_graph.tables.values():
            schema_terms.update(tinfo.columns)
            for col in tinfo.columns:
                schema_terms.add(col.lower())
        schema_stats = schema_graph.schema_stats or {}
        composite_tables = frozenset(schema_graph.tables.keys())
        if composite_tables:
            execution_ctx = replace(
                fed_master_ctx,
                allow_objects=MainSpaceOps.federation_execution_allow_objects(fed_master_ctx, composite_tables),
            )
        runtime_config = RuntimeConfig(
            engine="federation",
            artifacts_dir=fed_storage_dir,
            engine_context=master_ctx,
            llm_execution=llm_exec,
            execution_context=execution_ctx,
        )
        drain_owner = SimpleNamespace(
            _is_aether_federation=True,
            _schema_graph=schema_graph,
            _store=store,
            _templates=templates,
            _rejected=rejected,
            _dialect=coord_dialect,
            _federation_source_runtimes=fed_source_runtimes,
            _federation_member_graphs=fed_member_graphs_dict,
        )
        MainSpaceOps.drain_write_queue(drain_owner, fed_storage_dir)
        sink(f"  Federation: {fed_manifest.federation_id} ({len(member_dict)} members).")
        payload_counts = composite_schema_payload_counts(schema_graph)
        sink(
            "  Composite schema payload: "
            f"{payload_counts['tables']} tables, "
            f"{payload_counts['columns']} columns, "
            f"{payload_counts['enum_types']} enum types "
            f"({payload_counts['enum_labels']} labels)."
        )
        sink("Ready.")
        return AetherFederationInitResult(
            runtime_config=runtime_config,
            llm_config=llm_config,
            schema_graph=schema_graph,
            dialect=coord_dialect,
            artifacts_dir=fed_storage_dir,
            store=store,
            templates=cast(dict[str, Any], templates),
            rejected=rejected,
            schema_terms=schema_terms,
            schema_stats=schema_stats,
            schema_role=schema_role,
            consumer_visible_objects=consumer_visible,
            context_name=context_name,
            execution_context=execution_ctx,
            data_quality_report=None,
            federation_manifest=fed_manifest,
            federation_mappings=fed_mappings,
            federation_member_graphs=fed_member_graphs_dict,
            federation_storage_dir=fed_storage_dir,
            federation_source_runtimes=fed_source_runtimes,
            federation_mapping_suggestions=fed_mapping_suggestions,
            federation_dialects_by_source=fed_dialects_by_source,
            engine_identity=engine_identity,
            members=member_dict,
        )

    @staticmethod
    def clear_template_store_only(
        artifacts_dir: str,
        schema_graph: SchemaGraph,
        *,
        space: str | None = None,
    ) -> bool:
        """Remove template learning. ``space=None`` wipes the whole store; otherwise one partition (including master)."""
        assert isinstance(schema_graph, SchemaGraph)
        if space is not None:
            TemplateOps.ensure_template_store_space_layout(artifacts_dir)
            space_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, space)
            if not os.path.isdir(space_dir):
                return False
            shutil.rmtree(space_dir, ignore_errors=True)
            return True
        store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
        legacy = os.path.join(artifacts_dir, TEMPLATE_STORE_LEGACY_SINGLE_FILE)
        existed = os.path.isdir(store_dir) or os.path.isfile(legacy)
        if os.path.isdir(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)
        wipe_filenames(artifacts_dir, (TEMPLATE_STORE_LEGACY_SINGLE_FILE,))
        return existed

    @staticmethod
    def resolve_connection_credentials(
        credentials: str | Mapping[str, str] | None,
        token_provider: Callable[[], str | Mapping[str, str]] | None,
    ) -> str | Mapping[str, str]:
        """Return explicit credentials or consult *token_provider*."""
        if credentials is not None:
            return credentials
        if token_provider is not None:
            resolved = token_provider()
            if resolved is None or (isinstance(resolved, str) and not str(resolved).strip()):
                raise ConfigError("token_provider returned an empty credential value")
            return resolved
        raise ConfigError(
            "refresh requires explicit credentials or a token_provider callable configured on the engine."
        )

    @staticmethod
    def apply_connection_credentials_for_engine(
        engine_type: str,
        credentials: str | Mapping[str, str],
        *,
        runtime: EngineRuntimeConfig | None = None,
    ) -> EngineRuntimeConfig:
        """Apply rotatable secrets on the runtime config for *engine_type*."""
        runtime_cfg = runtime
        if runtime_cfg is None:
            runtime_cfg = EngineRuntimeConfig.process_default_for_class(EngineConfig.RUNTIME)
        try:
            runtime_cfg.apply_connection_credentials(credentials)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return runtime_cfg

    @staticmethod
    def dispose_engine_dialect(
        dialect: Any,
        *,
        borrowed_execution_engine: Any | None = None,
        borrowed_native_connection: Any | None = None,
    ) -> None:
        """Release dialect-owned database handles without closing borrowed caller handles."""
        unregister_dialect_live_handles(
            dialect,
            borrowed_execution_engine=borrowed_execution_engine,
            borrowed_native_connection=borrowed_native_connection,
        )
        dispose_native = getattr(dialect, "dispose_native_connection", None)
        if callable(dispose_native):
            try:
                dispose_native()
            except (OSError, AttributeError, TypeError):
                pass
            return
        connection = getattr(dialect, "connection", None)
        if connection is not None and connection is not borrowed_native_connection:
            close = getattr(connection, "close", None)
            if callable(close):
                try:
                    close()
                except (OSError, AttributeError, TypeError):
                    pass
        sa_engine = getattr(dialect, "engine", None)
        if sa_engine is not None and sa_engine is not borrowed_execution_engine:
            dispose = getattr(sa_engine, "dispose", None)
            if callable(dispose):
                try:
                    dispose()
                except (OSError, AttributeError, TypeError):
                    pass

    @staticmethod
    def refresh_engine_connection(
        *,
        engine_type: str,
        dialect: Any,
        credentials: str | Mapping[str, str] | None = None,
        token_provider: Callable[[], str | Mapping[str, str]] | None = None,
        execution_engine: Any | None = None,
        native_connection: Any | None = None,
        runtime: EngineRuntimeConfig | None = None,
    ) -> Any:
        """Dispose the live dialect, apply fresh credentials, and open a replacement handle."""
        resolved = MainInitOps.resolve_connection_credentials(credentials, token_provider)
        MainInitOps.dispose_engine_dialect(
            dialect,
            borrowed_execution_engine=execution_engine,
            borrowed_native_connection=native_connection,
        )
        runtime_cfg = runtime or getattr(dialect, "config", None)
        runtime_cfg = MainInitOps.apply_connection_credentials_for_engine(engine_type, resolved, runtime=runtime_cfg)
        try:
            return DialectRegistry.get_dialect(
                engine_type,
                runtime_cfg,
                sqlalchemy_engine=execution_engine,
                native_connection=native_connection,
            )
        except DatabaseConnectionError:
            raise
        except Exception as exc:
            raise DatabaseConnectionError(str(exc)) from exc

    @staticmethod
    def dispose_federation_source_runtimes(
        runtimes: Mapping[str, SourceRuntime] | None, *, member_engines: Mapping[str, Any] | None = None
    ) -> None:
        """Release dialect-owned resources for federation source runtimes without closing borrowed member handles."""
        if not runtimes:
            return
        borrowed_sa: set[int] = set()
        borrowed_native: set[int] = set()
        for engine in (member_engines or {}).values():
            sa = getattr(engine, "_execution_engine", None)
            if sa is not None:
                borrowed_sa.add(id(sa))
            native = getattr(engine, "_native_connection", None)
            if native is not None:
                borrowed_native.add(id(native))
        for runtime in runtimes.values():
            dialect = getattr(runtime, "dialect", None)
            dispose_dialect = getattr(dialect, "dispose_native_connection", None)
            if callable(dispose_dialect):
                try:
                    dispose_dialect()
                except (OSError, AttributeError, TypeError):
                    pass
            sa = getattr(runtime, "sqlalchemy_engine", None)
            if sa is not None and id(sa) not in borrowed_sa:
                dispose_sa = getattr(sa, "dispose", None)
                if callable(dispose_sa):
                    try:
                        dispose_sa()
                    except (OSError, AttributeError, TypeError):
                        pass
            native = getattr(runtime, "native_connection", None)
            if native is not None and id(native) not in borrowed_native:
                close_native = getattr(native, "close", None)
                if callable(close_native):
                    try:
                        close_native()
                    except (OSError, AttributeError, TypeError):
                        pass

    @staticmethod
    def clear_federation_template_stores(
        federation_dir: str | None,
        composite_artifacts_dir: str,
        composite_graph: SchemaGraph,
        member_engines: Mapping[str, Any],
        *,
        space: str | None = None,
    ) -> bool:
        """Clear composite, plan-record, and member template stores for a federation. When ``space`` is set, only that learning partition is removed (plan templates are left alone)."""
        existed = MainInitOps.clear_template_store_only(composite_artifacts_dir, composite_graph, space=space)
        if space is None and federation_dir:
            clear_federation_plan_templates(federation_dir)
        for engine in member_engines.values():
            graph = getattr(engine, "_schema_graph", None)
            adir = getattr(engine, "_artifacts_dir", None)
            if graph is not None and adir is not None:
                existed = MainInitOps.clear_template_store_only(str(adir), graph, space=space) or existed
        return existed

    @staticmethod
    def describe_federation_config(
        federation_name: str,
        runtime: RuntimeConfig,
        llm: LLMConfig,
        *,
        members: Mapping[str, Any],
        federation_storage_dir: str | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
    ) -> str:
        """Build a redacted config snapshot including federation topology."""
        lines = [
            MainInteractiveOps.describe_runtime_config(runtime, llm, schema_role=schema_role),
            "",
            "Federation:",
        ]
        lines.append(f"  name:          {federation_name}")
        if federation_storage_dir:
            lines.append(f"  storage dir:   {os.path.abspath(federation_storage_dir)}")
        lines.append(f"  member count:  {len(members)}")
        for connection_name, engine in sorted(members.items()):
            member_engine = str(getattr(engine, "dialect", "") or "")
            member_dir = os.path.abspath(str(getattr(engine, "_artifacts_dir", "") or ""))
            lines.append(f"  {connection_name}: engine={member_engine!r} artifacts_dir={member_dir}")
        return "\n".join(lines)

    @staticmethod
    def clear_simulation_caches_only(artifacts_dir: str) -> int:
        """Remove QSim and seed-warmup simulation artifacts; return count of files removed."""
        count = wipe_filenames(artifacts_dir, SIMULATION_CACHE_EXACT_FILENAMES)
        count += wipe_globs(artifacts_dir, SIMULATION_CACHE_GLOB_PATTERNS)
        return count

    @staticmethod
    def load_qsim_summaries(artifacts_dir: str) -> list[QSimSummary]:
        """Load every ``QSimSummary`` from per-run files under ``qsim/``, oldest first."""
        qsim_dir = os.path.join(artifacts_dir, "qsim")
        index_path = os.path.join(qsim_dir, "index.jsonl")
        if os.path.isfile(index_path):
            summaries: list[QSimSummary] = []
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    run_id = str(row.get("run_id") or row.get("version") or "").strip()
                    if not run_id:
                        continue
                    summary_path = os.path.join(qsim_dir, f"summary_{run_id}.json")
                    if not os.path.isfile(summary_path):
                        continue
                    try:
                        with open(summary_path, encoding="utf-8") as sf:
                            payload = json.load(sf)
                    except (json.JSONDecodeError, OSError):
                        continue
                    if isinstance(payload, dict):
                        summaries.append(QSimSummary.from_dict(payload))
            return summaries
        qsim_summary_path = os.path.join(artifacts_dir, "qsim_summary.json")
        if not os.path.exists(qsim_summary_path):
            return []
        with open(qsim_summary_path, encoding="utf-8") as f:
            summaries_raw: Any = json.load(f)
        if not isinstance(summaries_raw, list):
            return []
        return [QSimSummary.from_dict(s) for s in summaries_raw if isinstance(s, dict)]

    @staticmethod
    def validate_yes_no_reply_token(token: str, *, param: str) -> None:
        if token not in ("y", "n"):
            raise ValueError(f"{param} must be 'y' or 'n'")

    @staticmethod
    def normalise_yes_no(raw: str, options: list[str]) -> str | None:
        """Map free text to ``y`` or ``n`` when present in *options*."""
        token = raw.strip().lower()
        if token in ("y", "yes") and "y" in options:
            return "y"
        if token in ("n", "no") and "n" in options:
            return "n"
        return None

    @staticmethod
    def find_latest_seed_warmup_summary(artifacts_dir: str) -> SeedWarmupSummary | None:
        """Return the newest ``SeedWarmupSummary`` under *artifacts_dir*, or ``None`` when absent."""
        if not os.path.isdir(artifacts_dir):
            return None
        best_ver = -1
        for name in os.listdir(artifacts_dir):
            if not name.startswith("seed_warmup_report_v") or not name.endswith(".json"):
                continue
            mid = name[len("seed_warmup_report_v") : -len(".json")]
            if not mid.isdigit():
                continue
            best_ver = max(best_ver, int(mid))
        if best_ver < 0:
            return None
        return MainInitOps.get_seed_warmup_summary_from_dir(artifacts_dir, best_ver)
