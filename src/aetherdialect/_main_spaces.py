"""Aetherspace, visibility, knowledge/structure export, and write-queue drain ops."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ._config import (
    EngineConfig,
    SeedWarmupConfig,
)
from ._constants import (
    AETHERSPACE_ARTIFACT_VERSION,
    AETHERSPACE_NEXT_ID_FILENAME,
    AETHERSPACE_UID_PREFIX,
    AETHERSPACES_SEGMENT,
    APPLIED_MAP_ARCHIVE_RETENTION_COUNT,
    APPLIED_MAP_ARCHIVE_TIMESTAMP_RE,
    ARTIFACT_DIR_MODE,
    ARTIFACT_FILE_MODE,
    ARTIFACT_MANIFEST_FILENAME,
    AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD,
    AUDIT_EVENT_WRITE_QUEUE_STRUCTURE_PROPOSAL,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT,
    AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
    CREDENTIAL_DEFAULT_AETHERSPACE_NAME,
    CREDENTIAL_DEFAULT_FINGERPRINT_KEY,
    CREDENTIAL_DEFAULT_SNAPSHOT_FLAG,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT,
    FEDERATION_BASE_WHERE_OPS,
    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
    FEDERATION_STORAGE_PREFIX,
    JSON_COMPACT_SEPARATORS,
    KNOWLEDGE_EXPORT_FORMAT_VERSION,
    MASTER_AETHERSPACE_NAME,
    MASTER_AETHERSPACE_UID,
    NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
    NAMED_SCHEMA_CONTEXT_PREFIX,
    QSIM_QUESTIONS_PATTERN,
    SCHEMA_CONTEXT_NAMED_SPEC_GLOB,
    STRUCTURE_APPLIED_TIMESTAMP_FORMAT,
    STRUCTURE_DOCUMENT_VERSION,
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_PARTITION_PREFIX,
    TEMPLATE_STORE_SPACES_SEGMENT,
    WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS,
    WRITE_QUEUE_FILENAME,
    WRITE_QUEUE_MAX_BYTES_PER_DRAIN,
)
from ._contracts_base import (
    AetherSpace,
    AetherspaceDeleteResult,
    ConfigError,
    DomainKnowledgeEntry,
    DomainKnowledgeState,
    EngineContext,
    FederationConfigError,
    FederationContext,
    KnowledgeScope,
    OwnerOnlyOperationError,
    SchemaRole,
    SensitivityClassification,
    SessionTurnCancelledError,
    SpaceContext,
    StructuralKnowledgeFact,
)
from ._contracts_core import (
    FederatedPlan,
    FederatedPrepareOutcome,
    FederatedSqlBundle,
    GenerationPath,
    InteractiveChoicePort,
    QuestionFeedbackEntry,
    QuestionFormStorage,
    RuntimeIntent,
    SqlGenerationOutcome,
    Template,
    UserFeedbackRejectSuspendContext,
    WriteQueueEvent,
)
from ._contracts_schema import (
    DescriptionOwner,
    FederationCoordinatorConfig,
    FederationManifest,
    FederationMappings,
    QSimSummary,
    SchemaGraph,
)
from ._dialect import (
    DialectRegistry,
)
from ._federation_compose import (
    source_ids_for_intent,
)
from ._federation_execute import (
    federation_source_artifacts_dir,
)
from ._federation_manifest import (
    federation_residual_column_headers,
    intersect_member_where_ops,
)
from ._federation_plan import resolve_federated_member_schema
from ._knowledge_merge import (
    merge_domain_knowledge_federation_peers,
    merge_domain_knowledge_notes_overlay,
    merge_domain_knowledge_space_over_engine,
    merge_structural_knowledge_federation_peers,
    merge_structural_knowledge_notes_overlay,
)
from ._knowledge_staleness import knowledge_artifact_save_stamps, resolve_knowledge_extraction_for_schema
from ._pipeline_generate import complete_user_feedback_reject, handle_user_feedback
from ._schema_finalize import (
    apply_structure_from_path,
    finalize_with_structure,
)
from ._schema_graph import (
    apply_deny_objects_filter,
    load_schema_graph_snapshot,
    strip_schema_context_denied_columns,
    validate_scope_against_graph,
)
from ._schema_profile import (
    emit_description_enrichment_noop,
    extract_knowledge_from_notes,
    filter_schema_anchored_domain_knowledge,
    llm_enrich_schema_from_structural_knowledge,
    out_of_scope_description_tokens,
    raise_if_flat_descriptions_name_out_of_scope_entities,
)
from ._schema_reflect import (
    resolve_federation_qualified_ref,
    save_schema_to_cache,
)
from ._templates import LazyTemplateMapping, TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    debug,
    format_versions_match,
    load_domain_knowledge_artifact,
    norm_schema_identifier,
    notify,
    require_exact_keys,
    session_turn_cancelled,
    split_warmup_lattice_basename,
)
from ._utils_artifacts import (
    artifact_lock,
    decode_write_queue_event,
    manifest_matches_schema,
    read_artifact_manifest,
    save_domain_knowledge_artifact,
    write_queue_event_space_name,
)


@dataclass
class EngineArtifactState:
    """Per-engine artifact paths registered during construction."""

    schema_json_path: str
    template_store_dir: str


_ENGINE_ARTIFACT_STATES: dict[str, EngineArtifactState] = {}


@dataclass
class _WriteQueueDrainTarget:
    """Live graph and template state for one write-queue drain pass."""

    schema_graph: SchemaGraph
    store: dict[str, Any] | TemplateStoreView
    templates: dict[str, Template] | LazyTemplateMapping
    rejected: dict[str, Any]
    dialect: Any


class MainSpaceOps:
    """Aetherspace, visibility, knowledge/structure export, and write- queue drain ops."""

    @staticmethod
    def register_engine_artifact_state(
        artifacts_dir: str,
        *,
        schema_json_path: str,
        template_store_dir: str,
    ) -> None:
        """Record artifact paths for *artifacts_dir* without mutating global EngineConfig."""
        _ENGINE_ARTIFACT_STATES[os.path.abspath(artifacts_dir)] = EngineArtifactState(
            schema_json_path=schema_json_path,
            template_store_dir=template_store_dir,
        )

    @staticmethod
    def engine_schema_json_path(artifacts_dir: str) -> str:
        """Return the schema graph path for a constructed engine directory."""
        state = _ENGINE_ARTIFACT_STATES.get(os.path.abspath(artifacts_dir))
        if state is not None:
            return state.schema_json_path
        return os.path.join(artifacts_dir, "schema_graph.json.gz")

    @staticmethod
    def engine_template_store_dir(artifacts_dir: str) -> str:
        """Return the template store path for a constructed engine directory."""
        state = _ENGINE_ARTIFACT_STATES.get(os.path.abspath(artifacts_dir))
        if state is not None:
            return state.template_store_dir
        return TemplateOps.template_store_dir_for_space(artifacts_dir, MASTER_AETHERSPACE_NAME)

    @staticmethod
    def _remove_empty_template_shard_files(artifacts_dir: str) -> None:
        spaces_root = os.path.join(TemplateOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT)
        if not os.path.isdir(spaces_root):
            return
        for root, _dirs, files in os.walk(spaces_root):
            for name in files:
                if not name.startswith(TEMPLATE_STORE_PARTITION_PREFIX):
                    continue
                path = os.path.join(root, name)
                if os.path.isfile(path) and os.path.getsize(path) == 0:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    @staticmethod
    def _prune_stale_template_shards(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        import gzip

        spaces_root = os.path.join(TemplateOps.template_store_base_dir(artifacts_dir), TEMPLATE_STORE_SPACES_SEGMENT)
        if not os.path.isdir(spaces_root):
            return
        active = str(active_schema_graph_id or "")
        if not active:
            return
        for root, _dirs, files in os.walk(spaces_root):
            header_path = os.path.join(root, TEMPLATE_STORE_HEADER_FILENAME)
            if TEMPLATE_STORE_HEADER_FILENAME not in files:
                continue
            header_id = ""
            try:
                with gzip.open(header_path, "rt", encoding="utf-8") as fh:
                    hdr = json.load(fh)
                if isinstance(hdr, dict):
                    header_id = str(hdr.get("schema_graph_id", "") or "")
            except (OSError, json.JSONDecodeError):
                header_id = ""
            if header_id and header_id != active:
                for name in files:
                    if name.startswith(TEMPLATE_STORE_PARTITION_PREFIX):
                        try:
                            os.remove(os.path.join(root, name))
                        except OSError:
                            pass
                try:
                    os.remove(header_path)
                except OSError:
                    pass

    @staticmethod
    def _prune_stale_warmup_lattices(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        lattice_root = os.path.join(artifacts_dir, SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR)
        if not os.path.isdir(lattice_root):
            return
        active = str(active_schema_graph_id or "")
        if not active:
            return
        prefix = "lattice_"
        for name in os.listdir(lattice_root):
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            graph_id, _partition_fp, code_version = split_warmup_lattice_basename(name)
            if code_version != SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION:
                try:
                    os.remove(os.path.join(lattice_root, name))
                except OSError:
                    pass
                continue
            if graph_id and graph_id != active:
                try:
                    os.remove(os.path.join(lattice_root, name))
                except OSError:
                    pass

    @staticmethod
    def prune_orphaned_federation_trees(federation_parent_dir: str, *, active_fed_dir: str) -> None:
        keep = os.path.abspath(active_fed_dir)
        parent = os.path.abspath(federation_parent_dir)
        if not os.path.isdir(parent):
            return
        for name in os.listdir(parent):
            if not name.startswith(FEDERATION_STORAGE_PREFIX):
                continue
            path = os.path.join(parent, name)
            if not os.path.isdir(path) or os.path.abspath(path) == keep:
                continue
            composite = os.path.join(path, FEDERATION_COMPOSITE_SCHEMA_FILENAME)
            manifest = os.path.join(path, ARTIFACT_MANIFEST_FILENAME)
            if os.path.isfile(composite) or os.path.isfile(manifest):
                continue
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _prune_applied_map_archives(artifacts_dir: str, *, keep: int = APPLIED_MAP_ARCHIVE_RETENTION_COUNT) -> None:
        archives: list[tuple[str, str]] = []
        try:
            names = os.listdir(artifacts_dir)
        except OSError:
            return
        for name in names:
            if APPLIED_MAP_ARCHIVE_TIMESTAMP_RE.search(name):
                ts = name.split(".applied.", 1)[1].rsplit(".json", 1)[0]
                archives.append((name, ts))
        archives.sort(key=lambda item: item[1])
        for name, _ts in archives[:-keep]:
            try:
                os.remove(os.path.join(artifacts_dir, name))
            except OSError:
                pass

    @staticmethod
    def _clear_stale_write_queue(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        manifest = read_artifact_manifest(artifacts_dir)
        if manifest is None:
            return
        if str(manifest.schema_graph_id or "") == str(active_schema_graph_id or ""):
            return
        path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def prune_stale_artifact_auxiliaries(artifacts_dir: str, *, active_schema_graph_id: str) -> None:
        """Prune stale template shards, warmup lattices, old applied-map archives, and stale write queues."""
        MainSpaceOps._remove_empty_template_shard_files(artifacts_dir)
        MainSpaceOps._prune_stale_template_shards(artifacts_dir, active_schema_graph_id=active_schema_graph_id)
        MainSpaceOps._prune_stale_warmup_lattices(artifacts_dir, active_schema_graph_id=active_schema_graph_id)
        MainSpaceOps._prune_applied_map_archives(artifacts_dir)
        MainSpaceOps._clear_stale_write_queue(artifacts_dir, active_schema_graph_id=active_schema_graph_id)

    @staticmethod
    def orphan_superseded_identity_artifacts_on_rotation(
        artifacts_dir: str,
        *,
        previous_schema_graph_id: str,
        active_schema_graph_id: str,
    ) -> list[str]:
        """Move every artifact keyed to a superseded schema graph identity into ``orphaned/<id>/``."""
        return TemplateOps.orphan_superseded_identity_artifacts(
            artifacts_dir,
            previous_schema_graph_id=previous_schema_graph_id,
            active_schema_graph_id=active_schema_graph_id,
        )

    @staticmethod
    def _aetherspace_dir(engine_dir: str) -> str:
        return os.path.join(engine_dir, AETHERSPACES_SEGMENT)

    @staticmethod
    def validate_space_uid(space_uid: str) -> str:
        """Return a normalized space uid or raise ``ValueError`` when unsafe. Accepted forms: ``master``, ``S####`` allocator ids, and slug uids derived from the display-name stem."""
        raw = str(space_uid).strip()
        if not raw or raw in (".", "..") or "/" in raw or "\\" in raw:
            raise ValueError(f"invalid aetherspace uid: {space_uid!r}")
        lower = raw.lower()
        if lower == MASTER_AETHERSPACE_UID:
            return MASTER_AETHERSPACE_UID
        if len(raw) > 1 and raw[0] in ("S", "s") and raw[1:].isdigit():
            return f"{AETHERSPACE_UID_PREFIX}{raw[1:]}"
        try:
            return TemplateOps.validate_space_name(lower)
        except ValueError as exc:
            raise ValueError(f"invalid aetherspace uid: {space_uid!r}") from exc

    @staticmethod
    def is_allocated_space_uid(token: str) -> bool:
        """Return True when *token* is ``master`` or an allocator ``S####`` uid."""
        try:
            safe = MainSpaceOps.validate_space_uid(token)
        except ValueError:
            return False
        if safe == MASTER_AETHERSPACE_UID:
            return True
        return bool(safe.startswith(AETHERSPACE_UID_PREFIX) and safe[len(AETHERSPACE_UID_PREFIX) :].isdigit())

    @staticmethod
    def is_space_uid_token(token: str) -> bool:
        """Return True when *token* is a legal space uid form."""
        try:
            MainSpaceOps.validate_space_uid(token)
            return True
        except ValueError:
            return False

    @staticmethod
    def _aetherspace_path(engine_dir: str, uid: str) -> str:
        try:
            safe = MainSpaceOps.validate_space_uid(uid)
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace uid: {uid!r}") from exc
        if safe == MASTER_AETHERSPACE_UID:
            raise ConfigError("master is the implicit full-scope space and has no snapshot file")
        return os.path.join(MainSpaceOps._aetherspace_dir(engine_dir), f"{safe}.json")

    @staticmethod
    def _aetherspace_next_id_path(engine_dir: str) -> str:
        return os.path.join(MainSpaceOps._aetherspace_dir(engine_dir), AETHERSPACE_NEXT_ID_FILENAME)

    @staticmethod
    def _space_uid_numeric(uid: str) -> int | None:
        try:
            safe = MainSpaceOps.validate_space_uid(uid)
        except ValueError:
            return None
        if safe == MASTER_AETHERSPACE_UID:
            return None
        digits = safe[len(AETHERSPACE_UID_PREFIX) :]
        try:
            return int(digits)
        except ValueError:
            return None

    @staticmethod
    def _ceiling_next_space_id(engine_dir: str) -> int:
        high = 1
        root = MainSpaceOps._aetherspace_dir(engine_dir)
        if not os.path.isdir(root):
            return high
        for entry in os.listdir(root):
            if not entry.endswith(".json"):
                continue
            stem = entry[: -len(".json")]
            num = MainSpaceOps._space_uid_numeric(stem)
            if num is not None:
                high = max(high, num + 1)
            path = os.path.join(root, entry)
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                num = MainSpaceOps._space_uid_numeric(str(payload.get("uid") or ""))
                if num is not None:
                    high = max(high, num + 1)
        return high

    @staticmethod
    def allocate_aetherspace_uid(engine_dir: str) -> str:
        """Allocate the next ``S####`` uid under *engine_dir*."""
        path = MainSpaceOps._aetherspace_next_id_path(engine_dir)
        with artifact_lock(engine_dir):
            MainSpaceOps.ensure_aetherspace_catalog_upgraded(engine_dir, _holding_lock=True)
            current = MainSpaceOps._ceiling_next_space_id(engine_dir)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        raw = json.load(fh)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    raw = None
                if isinstance(raw, dict):
                    try:
                        current = max(current, int(raw.get("next_id", 1) or 1))
                    except (TypeError, ValueError):
                        pass
            uid = f"{AETHERSPACE_UID_PREFIX}{int(current):04d}"
            nxt = int(current) + 1
            os.makedirs(os.path.dirname(path), mode=ARTIFACT_DIR_MODE, exist_ok=True)
            MainSpaceOps.write_json_atomic(path, {"next_id": nxt})
            return uid

    @staticmethod
    def is_credential_default_snapshot(snapshot: Mapping[str, Any] | None) -> bool:
        """Return True when *snapshot* is a library-managed credential- default space."""
        if not isinstance(snapshot, Mapping):
            return False
        return bool(snapshot.get(CREDENTIAL_DEFAULT_SNAPSHOT_FLAG))

    @staticmethod
    def reclaim_stale_credential_default_spaces(engine_dir: str, current_fingerprint: str) -> int:
        """Remove credential-default snapshots whose grant fingerprint does not match."""
        want = str(current_fingerprint or "").strip()
        if not want:
            return 0
        removed = 0
        for uid, _label in MainSpaceOps.list_saved_aetherspace_entries(engine_dir):
            snap = MainSpaceOps.load_aetherspace_snapshot(engine_dir, uid)
            if snap is None or not MainSpaceOps.is_credential_default_snapshot(snap):
                continue
            fp = str(snap.get(CREDENTIAL_DEFAULT_FINGERPRINT_KEY) or "").strip()
            if fp == want:
                continue
            try:
                TemplateOps.purge_space_learning_partition(engine_dir, uid)
            except ValueError:
                pass
            path = MainSpaceOps._aetherspace_path(engine_dir, uid)
            if os.path.isfile(path):
                os.unlink(path)
                removed += 1
        return removed

    @staticmethod
    def find_credential_default_uid(engine_dir: str, fingerprint: str) -> str | None:
        """Return the uid of an existing credential-default space for *fingerprint*, if any."""
        want = str(fingerprint or "").strip()
        if not want:
            return None
        for uid, _label in MainSpaceOps.list_saved_aetherspace_entries(engine_dir):
            snap = MainSpaceOps.load_aetherspace_snapshot(engine_dir, uid)
            if snap is None or not MainSpaceOps.is_credential_default_snapshot(snap):
                continue
            if str(snap.get(CREDENTIAL_DEFAULT_FINGERPRINT_KEY) or "").strip() == want:
                return uid
        return None

    @staticmethod
    def _tables_named_by_domain_knowledge_entry(
        entry: DomainKnowledgeEntry, *, schema_table_names_l: frozenset[str] | None = None
    ) -> frozenset[str]:
        """Return the lowercased table names an entry declares via ``referenced_entities``. Empty means the entry names no table."""
        _ = schema_table_names_l
        return frozenset(r.strip().lower().split(".", 1)[0] for r in entry.referenced_entities if r.strip())

    @staticmethod
    def filter_domain_knowledge_for_visibility(
        entries: Sequence[DomainKnowledgeEntry],
        *,
        visible_table_names: set[str] | None,
        all_schema_table_names: set[str] | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Drop DK entries naming a table outside the caller's visible table set. Fails closed by construction: a named table absent from *visible_table_names* is dropped whether it is a table this caller has never heard of or one it knows about but cannot see — subset arithmetic never needs to distinguish the two. ``all_schema_table_names`` widens which bare (non-dotted) keys are recognized as naming a table at all; it defaults to *visible_table_names*."""
        if visible_table_names is None:
            return tuple(entries)
        visible_l = {t.lower() for t in visible_table_names}
        schema_l = frozenset(t.lower() for t in (all_schema_table_names or visible_table_names))
        kept: list[DomainKnowledgeEntry] = []
        for entry in entries:
            named_tables = MainSpaceOps._tables_named_by_domain_knowledge_entry(entry, schema_table_names_l=schema_l)
            if named_tables and not named_tables <= visible_l:
                continue
            kept.append(entry)
        return tuple(kept)

    @staticmethod
    def ensure_credential_default_aetherspace(
        engine_dir: str,
        schema_graph: SchemaGraph,
        visible_objects: frozenset[str] | None,
        *,
        engine_domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
    ) -> str:
        """Ensure a notes-free system credential-default space for this visibility grant. Returns the durable space uid. Creates once per fingerprint; reuses when present."""
        table_names = frozenset(
            n for n in (visible_objects or frozenset()) if isinstance(n, str) and n.strip() and "." not in n
        )
        if not table_names:
            table_names = frozenset(schema_graph.tables.keys())
        tindex = {str(n).lower(): n for n in schema_graph.tables}
        table_names = frozenset(tindex[n.lower()] for n in table_names if n.lower() in tindex)
        fingerprint = hashlib.sha256(",".join(sorted(table_names)).encode("utf-8")).hexdigest()[:16]
        MainSpaceOps.reclaim_stale_credential_default_spaces(engine_dir, fingerprint)
        existing = MainSpaceOps.find_credential_default_uid(engine_dir, fingerprint)
        if existing is not None:
            return existing
        uid = MainSpaceOps.allocate_aetherspace_uid(engine_dir)
        space_context = SpaceContext(tables=table_names, columns=frozenset())
        validated = MainSpaceOps.validate_space_context_against_graph(space_context, schema_graph)
        snapshot = MainSpaceOps.subset_graph_for_space(schema_graph, validated)
        snapshot["uid"] = uid
        snapshot["name"] = CREDENTIAL_DEFAULT_AETHERSPACE_NAME
        snapshot[CREDENTIAL_DEFAULT_SNAPSHOT_FLAG] = True
        snapshot[CREDENTIAL_DEFAULT_FINGERPRINT_KEY] = fingerprint
        snapshot = MainSpaceOps.enrich_space_snapshot_with_notes(
            snapshot,
            schema_graph,
            validated,
            engine_domain_knowledge=engine_domain_knowledge,
        )
        space_tables = {str(t) for t in (snapshot.get("tables") or ())}
        dk_raw = snapshot.get("domain_knowledge") or []
        if isinstance(dk_raw, list) and space_tables:
            entries: list[DomainKnowledgeEntry] = []
            for item in dk_raw:
                if not isinstance(item, Mapping):
                    continue
                try:
                    entries.append(
                        DomainKnowledgeEntry(
                            key=str(item.get("key") or ""),
                            kind=str(item.get("kind") or "glossary"),
                            text=str(item.get("text") or ""),
                            referenced_entities=MainSpaceOps._domain_knowledge_referenced_entities_from_item(item),
                        )
                    )
                except (TypeError, ValueError, ConfigError):
                    continue
            filtered = MainSpaceOps.filter_domain_knowledge_for_visibility(
                entries,
                visible_table_names=space_tables,
                all_schema_table_names=set(schema_graph.tables.keys()),
            )
            snapshot["domain_knowledge"] = [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in filtered]
            snapshot["domain_knowledge_digest"] = DomainKnowledgeState.digest_for(filtered)
        snapshot["uid"] = uid
        snapshot["name"] = CREDENTIAL_DEFAULT_AETHERSPACE_NAME
        snapshot[CREDENTIAL_DEFAULT_SNAPSHOT_FLAG] = True
        snapshot[CREDENTIAL_DEFAULT_FINGERPRINT_KEY] = fingerprint
        snapshot = MainSpaceOps.filter_space_snapshot_sensitive_columns(snapshot, schema_graph)
        MainSpaceOps.save_aetherspace_snapshot(engine_dir, uid, snapshot)
        return uid

    @staticmethod
    def write_json_atomic(path: str, obj: Any) -> None:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json.tmp", prefix=".aetherspace_", dir=directory, delete=False
            ) as tf:
                tmp_path = tf.name
                json.dump(obj, tf, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=True)
            os.replace(tmp_path, path)
            tmp_path = None
            try:
                os.chmod(path, ARTIFACT_FILE_MODE)
            except OSError:
                pass
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def write_jsonl_atomic(path: str, rows: list[dict[str, Any]]) -> None:
        """Write JSONL rows atomically."""
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".jsonl.tmp", prefix=".aetherdialect_", dir=directory, delete=False
            ) as tf:
                tmp_path = tf.name
                for row in rows:
                    tf.write(json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")
            os.replace(tmp_path, path)
            tmp_path = None
            try:
                os.chmod(path, ARTIFACT_FILE_MODE)
            except OSError:
                pass
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def column_eligible_for_space_allowlist(col: Any) -> bool:
        """Return False for columns that must never appear on an aetherspace allow-list. HIDDEN / RESTRICTED / denied columns stay on the schema graph for redaction and classification, but they are not selectable space membership and must not be persisted on space snapshots or offered in structure checklists."""
        if col is None:
            return False
        if getattr(col, "is_denied", False):
            return False
        sens = getattr(col, "sensitivity", None)
        if sens == SensitivityClassification.HIDDEN:
            return False
        if sens == SensitivityClassification.RESTRICTED:
            return False
        return True

    @staticmethod
    def validate_space_context_against_graph(
        space_context: SpaceContext, schema_graph: SchemaGraph, *, federation_manifest: FederationManifest | None = None
    ) -> SpaceContext:
        """Normalize *space_context* and verify every table/column exists in *schema_graph*."""
        graph_tables = set(schema_graph.tables.keys())
        if space_context.tables:
            for tbl in space_context.tables:
                if tbl not in graph_tables:
                    raise ConfigError(f"SpaceContext tables entry {tbl!r} is not in the schema graph")
        scope_tables = frozenset(space_context.tables) if space_context.tables else frozenset(graph_tables)
        resolve_manifest = federation_manifest or FederationManifest(
            federation_id="",
            sources=(),
            table_namespace={},
            cross_source_joins=(),
            coordinator=FederationCoordinatorConfig(),
        )
        if space_context.columns:
            normalized_cols: set[str] = set()
            for qc in space_context.columns:
                resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=schema_graph)
                tbl = resolved.table
                col = resolved.column
                normalized = resolved.qualified
                if tbl not in graph_tables:
                    raise ConfigError(f"SpaceContext columns entry {qc!r} references unknown table {tbl!r}")
                if tbl not in scope_tables:
                    raise ConfigError(
                        f"SpaceContext columns entry {qc!r} references table {tbl!r} outside tables scope"
                    )
                tm = schema_graph.tables.get(tbl)
                if tm is None or col not in tm.columns:
                    raise ConfigError(f"SpaceContext columns entry {qc!r} is not in the schema graph")
                col_meta = tm.columns[col]
                if not MainSpaceOps.column_eligible_for_space_allowlist(col_meta):
                    raise ConfigError(
                        f"SpaceContext columns entry {qc!r} cannot be included in an aetherspace "
                        f"(hidden, restricted, or denied columns are excluded from space scope)"
                    )
                normalized_cols.add(normalized)
            return SpaceContext(
                tables=space_context.tables,
                columns=frozenset(normalized_cols),
                deny_objects=space_context.deny_objects,
                deny_columns=space_context.deny_columns,
                notes_file=space_context.notes_file,
                notes=space_context.notes,
            )
        return space_context

    @staticmethod
    def filter_space_snapshot_sensitive_columns(
        snapshot: Mapping[str, Any],
        schema_graph: SchemaGraph,
        *,
        federation_manifest: FederationManifest | None = None,
    ) -> dict[str, Any]:
        """Drop HIDDEN/RESTRICTED/denied columns from a space snapshot's allow-list and column_meta."""
        out = dict(snapshot)
        resolve_manifest = MainSpaceOps._space_column_resolve_manifest(federation_manifest)
        raw_columns = out.get("columns") or ()
        kept_columns: list[str] = []
        if isinstance(raw_columns, (list, tuple)):
            for spec in raw_columns:
                raw = str(spec).strip()
                if not raw:
                    continue
                try:
                    resolved = resolve_federation_qualified_ref(raw, manifest=resolve_manifest, schema=schema_graph)
                except ConfigError:
                    continue
                tm = schema_graph.tables.get(resolved.table)
                col_meta = tm.columns.get(resolved.column) if tm is not None else None
                if not MainSpaceOps.column_eligible_for_space_allowlist(col_meta):
                    continue
                kept_columns.append(resolved.qualified)
        out["columns"] = sorted(set(kept_columns))
        raw_meta = out.get("column_meta")
        if isinstance(raw_meta, Mapping):
            filtered_meta: dict[str, Any] = {}
            for key, value in raw_meta.items():
                raw = str(key).strip()
                try:
                    resolved = resolve_federation_qualified_ref(raw, manifest=resolve_manifest, schema=schema_graph)
                except ConfigError:
                    continue
                tm = schema_graph.tables.get(resolved.table)
                col_meta = tm.columns.get(resolved.column) if tm is not None else None
                if not MainSpaceOps.column_eligible_for_space_allowlist(col_meta):
                    continue
                filtered_meta[resolved.qualified] = value
            out["column_meta"] = filtered_meta
        return out

    @staticmethod
    def build_master_space_descriptor(schema_graph: SchemaGraph) -> AetherSpace:
        """Return the implicit full-scope ``master`` descriptor derived from *schema_graph*."""
        tables = tuple(sorted(schema_graph.tables.keys()))
        columns: list[str] = []
        for tname in tables:
            tm = schema_graph.tables[tname]
            for col_name in sorted(tm.columns.keys()):
                if not MainSpaceOps.column_eligible_for_space_allowlist(tm.columns[col_name]):
                    continue
                columns.append(f"{tname}.{col_name}")
        return AetherSpace(
            uid=MASTER_AETHERSPACE_UID,
            name=MASTER_AETHERSPACE_NAME,
            tables=tables,
            columns=tuple(columns),
            notes=None,
        )

    @staticmethod
    def _space_column_resolve_manifest(federation_manifest: FederationManifest | None) -> FederationManifest:
        return federation_manifest or FederationManifest(
            federation_id="",
            sources=(),
            table_namespace={},
            cross_source_joins=(),
            coordinator=FederationCoordinatorConfig(),
        )

    @staticmethod
    def subset_graph_for_space(
        master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
    ) -> dict[str, Any]:
        """Build a versioned snapshot dict for persistence from *master_graph* and *space_context*. Tables-only spaces materialize only allowlist-eligible columns (never HIDDEN / RESTRICTED / denied). Empty ``validated.columns`` means all eligible columns on these tables, not every physical column."""
        validated = MainSpaceOps.validate_space_context_against_graph(
            space_context, master_graph, federation_manifest=federation_manifest
        )
        resolve_manifest = MainSpaceOps._space_column_resolve_manifest(federation_manifest)
        graph_tables = set(master_graph.tables.keys())
        if validated.tables:
            scope_tables = sorted(validated.tables)
        elif validated.deny_objects or validated.deny_columns:
            scope_tables = []
        else:
            scope_tables = sorted(graph_tables)
        frozenset(scope_tables)
        scope_columns: list[str] = []
        table_descriptions: dict[str, str] = {}
        column_meta: dict[str, dict[str, Any]] = {}
        for tname in scope_tables:
            tm = master_graph.tables[tname]
            desc = (tm.description or "").strip()
            if desc:
                table_descriptions[tname] = desc
            if validated.columns:
                allowed_cols: list[str] = []
                for qc in validated.columns:
                    resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=master_graph)
                    if resolved.table == tname:
                        allowed_cols.append(resolved.column)
                allowed_cols = sorted(set(allowed_cols))
            else:
                allowed_cols = sorted(
                    cname for cname, col in tm.columns.items() if MainSpaceOps.column_eligible_for_space_allowlist(col)
                )
            for col_name in allowed_cols:
                col = tm.columns[col_name]
                if not MainSpaceOps.column_eligible_for_space_allowlist(col):
                    continue
                qc = f"{tname}.{col_name}"
                scope_columns.append(qc)
                entry: dict[str, Any] = {}
                cdesc = (col.description or "").strip()
                if cdesc:
                    entry["description"] = cdesc
                sens = getattr(col, "sensitivity", None)
                if sens is not None and str(getattr(sens, "value", sens)) != "none":
                    entry["sensitivity"] = str(getattr(sens, "value", sens))
                if entry:
                    column_meta[qc] = entry
        return {
            "version": AETHERSPACE_ARTIFACT_VERSION,
            "uid": "",
            "name": "",
            "tables": scope_tables,
            "columns": sorted(scope_columns),
            "deny_objects": sorted(validated.deny_objects),
            "deny_columns": sorted(validated.deny_columns),
            "table_descriptions": table_descriptions,
            "column_meta": column_meta,
            "notes": None,
            "notes_hash": "",
        }

    @staticmethod
    def ensure_aetherspace_catalog_upgraded(engine_dir: str, *, _holding_lock: bool = False) -> None:
        """Validate on-disk aetherspace snapshots match the current artifact version and uid layout."""

        def _upgrade() -> None:
            root = MainSpaceOps._aetherspace_dir(engine_dir)
            if not os.path.isdir(root):
                return
            seen_uids: dict[str, str] = {}
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".json") or entry == AETHERSPACE_NEXT_ID_FILENAME:
                    continue
                if entry.startswith("."):
                    continue
                path = os.path.join(root, entry)
                if not os.path.isfile(path):
                    continue
                stem = entry[: -len(".json")]
                try:
                    with open(path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                found = payload.get("version")
                uid_raw = payload.get("uid")
                current = format_versions_match(found, AETHERSPACE_ARTIFACT_VERSION)
                has_uid = isinstance(uid_raw, str) and bool(str(uid_raw).strip())
                if not current:
                    raise ConfigError(
                        f"aetherspace snapshot at {path!r} has version {found!r}; "
                        f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
                        f"Delete {path!r} and redefine the aetherspace so it is rewritten "
                        f"at the current version."
                    )
                if not has_uid:
                    raise ConfigError(
                        f"aetherspace snapshot at {path!r} is missing uid; "
                        f"delete {path!r} and redefine the aetherspace."
                    )
                try:
                    uid = MainSpaceOps.validate_space_uid(str(uid_raw))
                except ValueError as exc:
                    raise ConfigError(f"aetherspace snapshot at {path!r} has invalid uid {uid_raw!r}") from exc
                if uid == MASTER_AETHERSPACE_UID:
                    raise ConfigError(f"aetherspace snapshot at {path!r} must not use master uid")
                if stem != uid:
                    dest = MainSpaceOps._aetherspace_path(engine_dir, uid)
                    if os.path.abspath(dest) != os.path.abspath(path):
                        if os.path.isfile(dest):
                            raise ConfigError(f"duplicate aetherspace uid {uid!r} under {root!r}")
                        os.replace(path, dest)
                        path = dest
                if uid in seen_uids and seen_uids[uid] != path:
                    raise ConfigError(f"duplicate aetherspace uid {uid!r} under {root!r}")
                seen_uids[uid] = path

        if _holding_lock:
            _upgrade()
        else:
            with artifact_lock(engine_dir):
                _upgrade()

    @staticmethod
    def load_aetherspace_snapshot(engine_dir: str, uid: str) -> dict[str, Any] | None:
        """
        Load a persisted space snapshot by uid.

        Returns:
            The snapshot dict, or ``None`` when the file is absent, unreadable, or
            structurally invalid (non-version failures).

        Raises:

            ConfigError: When the file exists but its ``version`` does not match
            :data:`AETHERSPACE_ARTIFACT_VERSION`.
        """
        MainSpaceOps.ensure_aetherspace_catalog_upgraded(engine_dir)
        try:
            path = MainSpaceOps._aetherspace_path(engine_dir, uid)
        except ConfigError:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        found = payload.get("version")
        if not format_versions_match(found, AETHERSPACE_ARTIFACT_VERSION):
            raise ConfigError(
                f"aetherspace snapshot at {path!r} has version {found!r}; "
                f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
                f"Delete {path!r} and redefine the aetherspace so it is rewritten "
                f"at the current version."
            )
        if not MainSpaceOps._aetherspace_snapshot_payload_valid(payload):
            return None
        try:
            payload_uid = MainSpaceOps.validate_space_uid(str(payload.get("uid") or ""))
            expect_uid = MainSpaceOps.validate_space_uid(uid)
        except ValueError:
            return None
        if payload_uid != expect_uid:
            return None
        return payload

    @staticmethod
    def _aetherspace_snapshot_payload_valid(payload: dict[str, Any]) -> bool:
        """Return True when *payload* has the expected structural shape (version checked by the loader)."""
        for key in ("tables", "columns"):
            raw = payload.get(key)
            if raw is None:
                continue
            if not isinstance(raw, (list, tuple)):
                return False
            if not all(isinstance(x, str) for x in raw):
                return False
        table_descriptions = payload.get("table_descriptions")
        if table_descriptions is not None and not isinstance(table_descriptions, dict):
            return False
        column_meta = payload.get("column_meta")
        if column_meta is not None and not isinstance(column_meta, dict):
            return False
        notes = payload.get("notes")
        if notes is not None and not isinstance(notes, str):
            return False
        notes_hash = payload.get("notes_hash")
        if notes_hash is not None and not isinstance(notes_hash, str):
            return False
        uid = payload.get("uid")
        if uid is not None and not isinstance(uid, str):
            return False
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            return False
        return True

    @staticmethod
    def save_aetherspace_snapshot(engine_dir: str, uid: str, snapshot: dict[str, Any]) -> str:
        """Persist *snapshot* under *uid* atomically; return the written path."""
        safe_uid = MainSpaceOps.validate_space_uid(uid)
        if safe_uid == MASTER_AETHERSPACE_UID:
            raise ConfigError("master is the implicit full-scope space and cannot be persisted")
        snap = dict(snapshot)
        snap["version"] = AETHERSPACE_ARTIFACT_VERSION
        snap["uid"] = safe_uid
        display = str(snap.get("name") or "").strip().lower() or safe_uid
        try:
            snap["name"] = TemplateOps.validate_space_name(display)
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace name: {display!r}") from exc
        if snap["name"] == MASTER_AETHERSPACE_NAME and safe_uid != MASTER_AETHERSPACE_UID:
            raise ConfigError("master is reserved and cannot be used as a display name for a persisted space")
        path = MainSpaceOps._aetherspace_path(engine_dir, safe_uid)
        MainSpaceOps.write_json_atomic(path, snap)
        return path

    @staticmethod
    def list_saved_aetherspace_entries(engine_dir: str) -> tuple[tuple[str, str], ...]:
        """Return sorted ``(uid, name)`` pairs for persisted spaces (excluding ``master``)."""
        MainSpaceOps.ensure_aetherspace_catalog_upgraded(engine_dir)
        root = MainSpaceOps._aetherspace_dir(engine_dir)
        if not os.path.isdir(root):
            return ()
        entries: list[tuple[str, str]] = []
        for entry in os.listdir(root):
            if not entry.endswith(".json") or entry == AETHERSPACE_NEXT_ID_FILENAME:
                continue
            stem = entry[: -len(".json")]
            if not stem or stem == MASTER_AETHERSPACE_UID:
                continue
            snap = MainSpaceOps.load_aetherspace_snapshot(engine_dir, stem)
            if snap is None:
                continue
            name = str(snap.get("name") or stem).strip().lower()
            entries.append((str(snap.get("uid") or stem), name))
        entries.sort(key=lambda item: (item[1], item[0]))
        return tuple(entries)

    @staticmethod
    def list_saved_aetherspace_names(engine_dir: str) -> tuple[str, ...]:
        """Return sorted saved space display names (excluding ``master``); duplicates preserved by scan order of entries."""
        return tuple(name for _uid, name in MainSpaceOps.list_saved_aetherspace_entries(engine_dir))

    @staticmethod
    def list_saved_aetherspace_uids(engine_dir: str) -> tuple[str, ...]:
        """Return sorted saved space uids (excluding ``master``)."""
        return tuple(uid for uid, _name in MainSpaceOps.list_saved_aetherspace_entries(engine_dir))

    @staticmethod
    def find_aetherspace_uids_by_name(engine_dir: str, name: str) -> tuple[str, ...]:
        """Return all uids whose display name equals *name* (normalized)."""
        try:
            norm = TemplateOps.validate_space_name(str(name).strip().lower())
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace name: {name!r}") from exc
        return tuple(uid for uid, label in MainSpaceOps.list_saved_aetherspace_entries(engine_dir) if label == norm)

    @staticmethod
    def resolve_aetherspace_identity(engine_dir: str, token: str) -> str:
        """Resolve *token* to a space uid. Prefer an on-disk snapshot whose uid equals *token*; otherwise resolve by display name when exactly one match exists."""
        raw = str(token).strip().lower()
        if not raw:
            raise ConfigError("aetherspace identity must be non-empty")
        if raw == MASTER_AETHERSPACE_UID:
            return MASTER_AETHERSPACE_UID
        MainSpaceOps.ensure_aetherspace_catalog_upgraded(engine_dir)
        if MainSpaceOps.is_space_uid_token(raw):
            snap = MainSpaceOps.load_aetherspace_snapshot(engine_dir, raw)
            if snap is not None:
                return MainSpaceOps.validate_space_uid(raw)
        matches = MainSpaceOps.find_aetherspace_uids_by_name(engine_dir, raw)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ConfigError(f"ambiguous aetherspace name {raw!r}; matches uids {', '.join(matches)}")
        raise ConfigError(f"unknown aetherspace {token!r}")

    @staticmethod
    def _aetherspace_export_path(engine_dir: str, uid: str) -> str:
        try:
            safe = MainSpaceOps.validate_space_uid(uid)
        except ValueError as exc:
            raise ConfigError(f"invalid aetherspace uid: {uid!r}") from exc
        return os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports", f"{safe}.export.json")

    @staticmethod
    def _parse_aetherspace_export_payload(payload: Any, *, source_path: str) -> dict[str, Any]:
        """Validate an exported aetherspace JSON document and return a persistable snapshot dict."""
        if not isinstance(payload, dict):
            raise ConfigError(f"malformed aetherspace export at {source_path!r}: expected a JSON object")
        found = payload.get("version")
        if not format_versions_match(found, AETHERSPACE_ARTIFACT_VERSION):
            raise ConfigError(
                f"aetherspace export at {source_path!r} has version {found!r}; "
                f"this build expects {AETHERSPACE_ARTIFACT_VERSION}. "
                f"Delete the export file and re-export at the current version."
            )
        snap = dict(payload)
        if not MainSpaceOps._aetherspace_snapshot_payload_valid(snap):
            raise ConfigError(f"malformed aetherspace export at {source_path!r}")
        uid_raw = snap.get("uid")
        name_raw = snap.get("name")
        if not isinstance(uid_raw, str) or not str(uid_raw).strip():
            raise ConfigError(f"malformed aetherspace export at {source_path!r}: missing uid")
        if not isinstance(name_raw, str) or not str(name_raw).strip():
            raise ConfigError(f"malformed aetherspace export at {source_path!r}: missing name")
        try:
            snap["uid"] = MainSpaceOps.validate_space_uid(uid_raw)
            snap["name"] = TemplateOps.validate_space_name(str(name_raw).strip().lower())
        except ValueError as exc:
            raise ConfigError(f"malformed aetherspace export at {source_path!r}: {exc}") from exc
        return snap

    @staticmethod
    def validate_aetherspace_snapshot_against_graph(
        snapshot: dict[str, Any],
        schema_graph: SchemaGraph,
        *,
        federation_manifest: FederationManifest | None = None,
    ) -> None:
        """Raise :class:`ConfigError` when *snapshot* scope references objects outside *schema_graph*."""
        space_context = SpaceContext(
            tables=frozenset(str(t) for t in (snapshot.get("tables") or ())),
            columns=frozenset(str(c) for c in (snapshot.get("columns") or ())),
            deny_objects=frozenset(str(t) for t in (snapshot.get("deny_objects") or ())),
            deny_columns=frozenset(str(c) for c in (snapshot.get("deny_columns") or ())),
        )
        MainSpaceOps.validate_space_context_against_graph(
            space_context,
            schema_graph,
            federation_manifest=federation_manifest,
        )

    @staticmethod
    def write_space_snapshot(engine_dir: str, uid: str, master_graph: SchemaGraph) -> Path:
        """Write a JSON export for *uid* and return its path (pair with :func:`read_space_snapshot`)."""
        if uid == MASTER_AETHERSPACE_UID:
            snap = MainSpaceOps.subset_graph_for_space(
                master_graph, SpaceContext(tables=frozenset(), columns=frozenset())
            )
            snap["uid"] = MASTER_AETHERSPACE_UID
            snap["name"] = MASTER_AETHERSPACE_NAME
        else:
            loaded = MainSpaceOps.load_aetherspace_snapshot(engine_dir, uid)
            if loaded is None:
                raise ConfigError(f"unknown aetherspace {uid!r}")
            snap = dict(loaded)
            snap["uid"] = str(loaded.get("uid") or uid)
            snap["name"] = str(loaded.get("name") or "")
        export_dir = os.path.join(engine_dir, AETHERSPACES_SEGMENT, "_exports")
        os.makedirs(export_dir, exist_ok=True)
        out_path = MainSpaceOps._aetherspace_export_path(engine_dir, snap["uid"])
        MainSpaceOps.write_json_atomic(out_path, snap)
        return Path(out_path)

    @staticmethod
    def read_space_snapshot(
        engine_dir: str,
        uid: str | None,
        master_graph: SchemaGraph,
        *,
        source: str | os.PathLike[str] | None = None,
        federation_manifest: FederationManifest | None = None,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        mappings: FederationMappings | None = None,
    ) -> AetherSpace:
        """Persist one aetherspace from an exported JSON document under *uid* (or the export uid)."""
        target_uid = None
        if uid is not None:
            try:
                target_uid = MainSpaceOps.validate_space_uid(str(uid).strip())
            except ValueError as exc:
                raise ConfigError(f"invalid aetherspace uid: {uid!r}") from exc
        if target_uid == MASTER_AETHERSPACE_UID or target_uid == MASTER_AETHERSPACE_NAME:
            raise ConfigError(
                "master is the implicit full-scope space; it cannot be created or overwritten",
            )
        source_path = (
            os.fspath(source)
            if source is not None
            else MainSpaceOps._aetherspace_export_path(engine_dir, target_uid or "")
        )
        if source is None and not target_uid:
            raise ConfigError("read_space_snapshot requires uid or source")
        if not os.path.isfile(source_path):
            raise ConfigError(f"aetherspace export file not found: {source_path}")
        try:
            with open(source_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigError(f"could not read aetherspace export at {source_path!r}: {exc}") from exc
        snap = MainSpaceOps._parse_aetherspace_export_payload(payload, source_path=source_path)
        MainSpaceOps.validate_aetherspace_snapshot_against_graph(
            snap,
            master_graph,
            federation_manifest=federation_manifest,
        )
        if target_uid:
            try:
                persist_uid = MainSpaceOps.validate_space_uid(target_uid)
            except ValueError as exc:
                raise ConfigError(f"invalid aetherspace uid: {target_uid!r}") from exc
            if persist_uid == MASTER_AETHERSPACE_UID:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            if not MainSpaceOps.is_allocated_space_uid(persist_uid):
                snap["name"] = persist_uid
        else:
            persist_uid = str(snap["uid"])
            if persist_uid == MASTER_AETHERSPACE_UID:
                raise ConfigError(
                    "master is the implicit full-scope space; it cannot be created or overwritten",
                )
            try:
                persist_uid = MainSpaceOps.validate_space_uid(persist_uid)
            except ValueError as exc:
                raise ConfigError(f"invalid aetherspace uid: {persist_uid!r}") from exc
        snap["uid"] = persist_uid
        snap = MainSpaceOps.filter_space_snapshot_sensitive_columns(
            snap,
            master_graph,
            federation_manifest=federation_manifest,
        )
        space_tables, space_columns = MainSpaceOps.space_allowed_sets_from_snapshot(snap)
        MainSpaceOps.validate_aetherspace_define_within_visibility(
            space_tables,
            space_columns,
            master_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
            federation_manifest=federation_manifest,
        )
        MainSpaceOps.save_aetherspace_snapshot(engine_dir, persist_uid, snap)
        return MainSpaceOps.aetherspace_descriptor_from_snapshot(persist_uid, snap)

    @staticmethod
    def delete_aetherspace_snapshot(engine_dir: str, uid: str) -> bool:
        """Delete one persisted aetherspace snapshot. Returns ``True`` when the snapshot file existed and was deleted."""
        safe = MainSpaceOps.validate_space_uid(uid)
        if safe == MASTER_AETHERSPACE_UID:
            raise ConfigError("master is the implicit full-scope space and cannot be deleted")
        path = MainSpaceOps._aetherspace_path(engine_dir, safe)
        if not os.path.isfile(path):
            raise ConfigError(f"unknown aetherspace {uid!r}")
        os.unlink(path)
        return True

    @staticmethod
    def delete_aetherspace(
        engine_dir: str,
        uid: str,
        *,
        persist_learning: bool = True,
        schema_graph: SchemaGraph,
    ) -> AetherspaceDeleteResult:
        """Delete one persisted aetherspace, optionally merging its learning into master first."""
        safe = MainSpaceOps.validate_space_uid(uid)
        if safe == MASTER_AETHERSPACE_UID:
            raise ConfigError("master is the implicit full-scope space and cannot be deleted")
        path = MainSpaceOps._aetherspace_path(engine_dir, safe)
        if not os.path.isfile(path):
            raise ConfigError(f"unknown aetherspace {uid!r}")
        graph_id = str(schema_graph.schema_graph_id or "")
        merge_counts: dict[str, int] = {}
        with artifact_lock(engine_dir):
            if persist_learning:
                merge_result = TemplateOps.merge_space_learning_into_master(engine_dir, safe, graph_id, schema_graph)
                merge_counts = merge_result.counts.to_dict()
                if merge_counts:
                    debug(f"[spaces.delete_aetherspace] learning_merge space={safe!r} dispositions={merge_counts}")
            TemplateOps.purge_space_learning_partition(engine_dir, safe)
        os.unlink(path)
        return AetherspaceDeleteResult(deleted=True, merge_counts=merge_counts)

    @staticmethod
    def aetherspace_descriptor_from_snapshot(uid: str, snapshot: dict[str, Any]) -> AetherSpace:
        """Build an :class:`AetherSpace` read-only view from a stored snapshot dict."""
        tables_raw = snapshot.get("tables") or ()
        cols_raw = snapshot.get("columns") or ()
        tables = tuple(str(t) for t in tables_raw)
        columns = tuple(str(c) for c in cols_raw)
        notes_raw = snapshot.get("notes")
        notes = str(notes_raw).strip() if isinstance(notes_raw, str) and notes_raw.strip() else None
        snap_uid = str(snapshot.get("uid") or uid).strip() or uid
        try:
            snap_uid = MainSpaceOps.validate_space_uid(snap_uid)
        except ValueError:
            snap_uid = str(uid).strip()
        snap_name = str(snapshot.get("name") or snap_uid).strip().lower()
        return AetherSpace(
            uid=snap_uid,
            name=snap_name,
            tables=tables,
            columns=columns,
            notes=notes,
        )

    @staticmethod
    def space_allowed_sets_from_snapshot(snapshot: dict[str, Any] | None) -> tuple[frozenset[str], frozenset[str]]:
        """Return ``(allowed_tables, allowed_columns)`` for enforcement; empty frozensets mean unrestricted."""
        if snapshot is None:
            return frozenset(), frozenset()
        tables_raw = snapshot.get("tables") or ()
        cols_raw = snapshot.get("columns") or ()
        tables = frozenset(norm_schema_identifier(str(t), what="aetherspace table") for t in tables_raw)
        columns: set[str] = set()
        for spec in cols_raw:
            raw = str(spec).strip()
            if raw.count(".") != 1:
                continue
            tbl, col = raw.split(".", 1)
            columns.add(
                f"{norm_schema_identifier(tbl, what='aetherspace table')}.{norm_schema_identifier(col, what='aetherspace column')}"
            )
        return tables, frozenset(columns)

    @staticmethod
    def space_deny_sets_from_snapshot(snapshot: dict[str, Any] | None) -> tuple[frozenset[str], frozenset[str]]:
        """Return ``(deny_objects, deny_columns)`` from a persisted aetherspace snapshot."""
        if snapshot is None:
            return frozenset(), frozenset()
        deny_obj_raw = snapshot.get("deny_objects") or ()
        deny_col_raw = snapshot.get("deny_columns") or ()
        deny_objects = frozenset(norm_schema_identifier(str(t), what="aetherspace deny_objects") for t in deny_obj_raw)
        deny_columns: set[str] = set()
        for spec in deny_col_raw:
            raw = str(spec).strip()
            if raw.count(".") != 1:
                continue
            tbl, col = raw.split(".", 1)
            deny_columns.add(
                f"{norm_schema_identifier(tbl, what='aetherspace deny table')}.{norm_schema_identifier(col, what='aetherspace deny column')}"
            )
        return deny_objects, frozenset(deny_columns)

    @staticmethod
    def intersect_space_scope(
        base_tables: frozenset[str],
        base_columns: frozenset[str],
        base_deny_objects: frozenset[str],
        base_deny_columns: frozenset[str],
        ephemeral: SpaceContext | None,
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        """Intersect ephemeral session scope with base aetherspace scope. Ephemeral allows never widen base allows. Deny lists from both layers are unioned. When either allow side is empty it is treated as unrestricted at that layer, matching engine-context composition rules."""
        if ephemeral is None or not (
            ephemeral.tables or ephemeral.columns or ephemeral.deny_objects or ephemeral.deny_columns
        ):
            return base_tables, base_columns, base_deny_objects, base_deny_columns

        if base_tables:
            if ephemeral.tables:
                tables = frozenset(t for t in ephemeral.tables if t in base_tables)
            else:
                tables = base_tables
        else:
            tables = ephemeral.tables

        if base_columns:
            if ephemeral.columns:
                columns = frozenset(c for c in ephemeral.columns if c in base_columns)
            else:
                columns = base_columns
        else:
            columns = ephemeral.columns

        deny_objects = base_deny_objects | ephemeral.deny_objects
        deny_columns = base_deny_columns | ephemeral.deny_columns

        overlap_obj = tables & deny_objects
        if overlap_obj:
            raise ConfigError(f"SpaceContext tables and deny_objects overlap: {sorted(overlap_obj)!r}")

        for table_name in deny_objects:
            for spec in deny_columns:
                denied_table, _, _rest = spec.partition(".")
                if denied_table != "*" and denied_table == table_name:
                    raise ConfigError(f"deny_objects entry {table_name!r} conflicts with deny_columns entry {spec!r}")

        return tables, columns, deny_objects, deny_columns

    @staticmethod
    def resolve_preview_scope_context(owner: Any) -> EngineContext | FederationContext:
        runtime_cfg = getattr(owner, "_runtime_config", None)
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        if execution_context is not None:
            return cast(EngineContext | FederationContext, execution_context)
        if runtime_cfg is not None:
            ctx = getattr(runtime_cfg, "engine_context", None)
            if ctx is not None:
                return cast(EngineContext | FederationContext, ctx)
        return EngineContext()

    @staticmethod
    def build_subset_schema_for_space_notes(
        master_graph: SchemaGraph, space_context: SpaceContext, *, federation_manifest: FederationManifest | None = None
    ) -> SchemaGraph:
        """Return a deep-copied in-scope schema graph for notes-aware LLM classification."""
        validated = MainSpaceOps.validate_space_context_against_graph(
            space_context, master_graph, federation_manifest=federation_manifest
        )
        resolve_manifest = MainSpaceOps._space_column_resolve_manifest(federation_manifest)
        graph_tables = set(master_graph.tables.keys())
        scope_tables = sorted(validated.tables) if validated.tables else sorted(graph_tables)
        subset_tables: dict[str, Any] = {}
        for tname in scope_tables:
            tm = copy.deepcopy(master_graph.tables[tname])
            if validated.columns:
                allowed: set[str] = set()
                for qc in validated.columns:
                    resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=master_graph)
                    if resolved.table == tname:
                        allowed.add(resolved.column)
                for col_name in list(tm.columns.keys()):
                    if col_name not in allowed:
                        del tm.columns[col_name]
            subset_tables[tname] = tm
        subset = SchemaGraph(tables=subset_tables, join_paths_multi={})
        deny_ctx = EngineContext(
            deny_objects=frozenset(validated.deny_objects or ()),
            deny_columns=frozenset(validated.deny_columns or ()),
        )
        apply_deny_objects_filter(subset, deny_ctx)
        strip_schema_context_denied_columns(subset, deny_ctx)
        return subset

    @staticmethod
    def merge_domain_knowledge(
        engine_entries: Sequence[DomainKnowledgeEntry],
        space_entries: Sequence[DomainKnowledgeEntry],
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Merge space DK over engine DK by key; space replaces on collision."""
        return merge_domain_knowledge_space_over_engine(engine_entries, space_entries)

    @staticmethod
    def _domain_knowledge_entry_to_dict(entry: DomainKnowledgeEntry) -> dict[str, Any]:
        """Serialize an entry for persistence, always carrying ``referenced_entities``."""
        return {
            "key": entry.key,
            "kind": entry.kind,
            "text": entry.text,
            "referenced_entities": sorted(entry.referenced_entities),
        }

    @staticmethod
    def _domain_knowledge_referenced_entities_from_item(item: Mapping[str, Any]) -> frozenset[str]:
        if "referenced_entities" not in item:
            raise ConfigError("domain knowledge entry requires referenced_entities")
        raw = item.get("referenced_entities")
        if not isinstance(raw, list) or not all(isinstance(r, str) for r in raw):
            raise ConfigError("domain knowledge entry referenced_entities must be a list of strings")
        return frozenset(str(r).strip() for r in raw if str(r).strip())

    @staticmethod
    def entries_from_snapshot_domain_knowledge(
        snapshot: Mapping[str, Any] | None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        if not isinstance(snapshot, Mapping):
            return ()
        raw = snapshot.get("domain_knowledge")
        if not isinstance(raw, list):
            return ()
        out: list[DomainKnowledgeEntry] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            text = str(item.get("text") or "").strip()
            kind = str(item.get("kind") or "glossary").strip() or "glossary"
            if not key or not text or key in seen:
                continue
            referenced_entities = MainSpaceOps._domain_knowledge_referenced_entities_from_item(item)
            try:
                entry = DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced_entities)
                )
            except ConfigError:
                continue
            seen.add(entry.key)
            out.append(entry)
        return tuple(out)

    @staticmethod
    def description_overlays_from_snapshot(
        space_snapshot: Mapping[str, Any] | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        table_descriptions: dict[str, str] = {}
        column_descriptions: dict[str, str] = {}
        if not isinstance(space_snapshot, Mapping):
            return table_descriptions, column_descriptions
        for tname, desc in dict(space_snapshot.get("table_descriptions") or {}).items():
            if isinstance(desc, str) and desc.strip():
                table_descriptions[str(tname)] = desc.strip()
        for qc, meta in dict(space_snapshot.get("column_meta") or {}).items():
            if not isinstance(meta, dict):
                continue
            desc = meta.get("description")
            if isinstance(desc, str) and desc.strip():
                column_descriptions[str(qc)] = desc.strip()
        return table_descriptions, column_descriptions

    @staticmethod
    def _live_graph_description_maps(
        schema_graph: SchemaGraph | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        table_descriptions: dict[str, str] = {}
        column_descriptions: dict[str, str] = {}
        if schema_graph is None:
            return table_descriptions, column_descriptions
        for tname, tbl in schema_graph.tables.items():
            desc = (tbl.description or "").strip()
            if desc:
                table_descriptions[str(tname)] = desc
            for cname, col in tbl.columns.items():
                cdesc = (col.description or "").strip()
                if cdesc:
                    column_descriptions[f"{tname}.{cname}"] = cdesc
        return table_descriptions, column_descriptions

    @staticmethod
    def build_space_knowledge_export(
        *,
        engine_entries: Sequence[DomainKnowledgeEntry],
        space: str | None = None,
        space_snapshot: Mapping[str, Any] | None = None,
        schema_graph: SchemaGraph | None = None,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        space_tables: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build the slim space-knowledge payload for master or one named space."""
        space_uid = MASTER_AETHERSPACE_UID
        entries = tuple(engine_entries)
        table_descriptions: dict[str, str] = {}
        column_descriptions: dict[str, str] = {}
        resolved_space_tables = space_tables
        if space is not None:
            norm = str(space).strip().lower()
            if not norm:
                raise ConfigError("space identity must be non-empty")
            if norm in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID):
                space_uid = MASTER_AETHERSPACE_UID
                table_descriptions, column_descriptions = MainSpaceOps._live_graph_description_maps(schema_graph)
            else:
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
                space_uid = str(space_snapshot.get("uid") or norm).strip() or norm
                try:
                    space_uid = MainSpaceOps.validate_space_uid(space_uid)
                except ValueError:
                    space_uid = norm
                space_entries = MainSpaceOps.entries_from_snapshot_domain_knowledge(space_snapshot)
                entries = MainSpaceOps.merge_domain_knowledge(engine_entries, space_entries)
                table_descriptions, column_descriptions = MainSpaceOps.description_overlays_from_snapshot(
                    space_snapshot
                )
                if resolved_space_tables is None:
                    raw_tables = space_snapshot.get("tables")
                    if isinstance(raw_tables, (list, tuple)) and raw_tables:
                        resolved_space_tables = {str(t) for t in raw_tables}
        else:
            table_descriptions, column_descriptions = MainSpaceOps._live_graph_description_maps(schema_graph)
        if schema_graph is not None:
            use_snapshot = space_snapshot if space is not None and space_uid != MASTER_AETHERSPACE_UID else None
            entries = MainSpaceOps.derive_caller_scoped_domain_knowledge(
                engine_entries=engine_entries,
                schema=schema_graph,
                scope_ctx=scope_ctx,
                visible_objects=visible_objects,
                space_snapshot=use_snapshot,
                space_tables=resolved_space_tables,
            )
            filtered_tables: dict[str, str] = {}
            for tname, desc in table_descriptions.items():
                if MainSpaceOps.table_allowed_for_visibility(tname, schema_graph, scope_ctx, visible_objects):
                    if resolved_space_tables is None or tname in resolved_space_tables:
                        filtered_tables[tname] = desc
            filtered_columns: dict[str, str] = {}
            for qc, desc in column_descriptions.items():
                if "." not in qc:
                    continue
                tname, cname = qc.split(".", 1)
                if resolved_space_tables is not None and tname not in resolved_space_tables:
                    continue
                tbl = schema_graph.tables.get(tname)
                col = tbl.columns.get(cname) if tbl is not None else None
                if col is None:
                    continue
                if not MainSpaceOps.column_allowed_for_visibility(
                    tname,
                    cname,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=True,
                    col=col,
                ):
                    continue
                filtered_columns[qc] = desc
            table_descriptions = filtered_tables
            column_descriptions = filtered_columns
        return {
            "uid": space_uid,
            "domain_knowledge": [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in entries],
            "table_descriptions": dict(sorted(table_descriptions.items())),
            "column_descriptions": dict(sorted(column_descriptions.items())),
        }

    _KNOWLEDGE_DOCUMENT_KEYS = frozenset({"domain_knowledge", "table_descriptions", "column_descriptions", "uid"})
    _KNOWLEDGE_ENTRY_KEYS = frozenset({"key", "kind", "text", "referenced_entities"})

    @staticmethod
    def validate_knowledge_document(document: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a caller-supplied knowledge document before apply."""
        if not isinstance(document, Mapping):
            raise ConfigError("knowledge document must be a JSON object")
        require_exact_keys(
            document,
            allowed=MainSpaceOps._KNOWLEDGE_DOCUMENT_KEYS,
            required=frozenset(),
            context="knowledge document",
        )
        dk_raw = document.get("domain_knowledge")
        if dk_raw is not None:
            if not isinstance(dk_raw, list):
                raise ConfigError("knowledge document domain_knowledge must be an array")
            for item in dk_raw:
                if not isinstance(item, dict):
                    raise ConfigError("knowledge document domain_knowledge entries must be objects")
                require_exact_keys(
                    item,
                    allowed=MainSpaceOps._KNOWLEDGE_ENTRY_KEYS,
                    required=frozenset({"key", "text", "referenced_entities"}),
                    context="knowledge document domain_knowledge entry",
                )
        for map_key in ("table_descriptions", "column_descriptions"):
            raw = document.get(map_key)
            if raw is not None and not isinstance(raw, Mapping):
                raise ConfigError(f"knowledge document {map_key} must be an object")
        return dict(document)

    @staticmethod
    def knowledge_document_apply_fields(document: Mapping[str, Any]) -> dict[str, Any]:
        """Extract apply fields from a validated knowledge document."""
        validated = MainSpaceOps.validate_knowledge_document(document)
        return {
            "domain_knowledge": validated.get("domain_knowledge"),
            "table_descriptions": validated.get("table_descriptions"),
            "column_descriptions": validated.get("column_descriptions"),
        }

    @staticmethod
    def build_knowledge_export(
        *,
        space_payloads: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build the visible-spaces knowledge aggregate (same payload shape as per-space export)."""
        spaces_out: dict[str, dict[str, Any]] = {}
        for name in sorted(space_payloads):
            payload = dict(space_payloads[name])
            payload.pop("format_version", None)
            spaces_out[name] = payload
        return {
            "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
            "spaces": spaces_out,
        }

    @staticmethod
    def build_structure_export(
        *,
        schema_graph: SchemaGraph,
        space: str | None = None,
        space_snapshot: Mapping[str, Any] | None = None,
        federation_members: Mapping[str, Any] | None = None,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Build read-only structural inventory (tables/columns/types/keys/relationships)."""
        allowed_tables: set[str] | None = None
        allowed_columns: set[str] | None = None
        if space is not None:
            norm = str(space).strip().lower()
            if norm and norm != MASTER_AETHERSPACE_NAME:
                if space_snapshot is None:
                    raise ConfigError(f"unknown aetherspace {space!r}")
                raw_tables = space_snapshot.get("tables")
                if isinstance(raw_tables, (list, tuple)):
                    allowed_tables = {str(t) for t in raw_tables}
                raw_columns = space_snapshot.get("columns")
                if isinstance(raw_columns, (list, tuple)) and raw_columns:
                    allowed_columns = {str(c) for c in raw_columns}
        visible_table_names: set[str] = set()
        tables_out: list[dict[str, Any]] = []
        for tname in sorted(schema_graph.tables):
            if allowed_tables is not None and tname not in allowed_tables:
                continue
            if not MainSpaceOps.table_allowed_for_visibility(tname, schema_graph, scope_ctx, visible_objects):
                continue
            tbl = schema_graph.tables[tname]
            columns_out: list[dict[str, Any]] = []
            visible_col_names: set[str] = set()
            for cname in sorted(tbl.columns):
                qc = f"{tname}.{cname}"
                if allowed_columns is not None and qc not in allowed_columns:
                    continue
                col = tbl.columns[cname]
                if not MainSpaceOps.column_allowed_for_visibility(
                    tname,
                    cname,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=True,
                    col=col,
                ):
                    continue
                visible_col_names.add(cname)
                columns_out.append({"name": cname, "data_type": str(col.data_type or "")})
            if not columns_out:
                continue
            visible_table_names.add(tname)
            primary_key = [c for c in list(tbl.primary_key or []) if c in visible_col_names]
            foreign_keys: list[dict[str, Any]] = []
            for fk in list(tbl.foreign_keys or []):
                src_cols = [c for c in list(fk.src_cols or []) if c in visible_col_names]
                dst_table = str(fk.dst_table or "")
                dst_cols = [str(c) for c in list(fk.dst_cols or [])]
                if not src_cols or not dst_table or not dst_cols:
                    continue
                foreign_keys.append(
                    {
                        "src_cols": src_cols,
                        "dst_table": dst_table,
                        "dst_cols": dst_cols,
                    }
                )
            tables_out.append(
                {
                    "name": tname,
                    "columns": columns_out,
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                }
            )
        relationships: list[dict[str, Any]] = []
        for table_entry in tables_out:
            src_table = str(table_entry["name"])
            for fk in table_entry["foreign_keys"]:
                dst_table = str(fk["dst_table"])
                if dst_table not in visible_table_names:
                    continue
                relationships.append(
                    {
                        "src_table": src_table,
                        "src_cols": list(fk["src_cols"]),
                        "dst_table": dst_table,
                        "dst_cols": list(fk["dst_cols"]),
                    }
                )
            table_entry["foreign_keys"] = [
                fk for fk in table_entry["foreign_keys"] if str(fk["dst_table"]) in visible_table_names
            ]
        out: dict[str, Any] = {
            "table_count": len(tables_out),
            "tables": tables_out,
            "relationships": relationships,
        }
        if federation_members is not None:
            out["members"] = {
                str(sid): {
                    "tables": sorted(
                        str(t)
                        for t in (info if isinstance(info, (list, tuple, set, frozenset)) else [])
                        if str(t) in visible_table_names
                    )
                }
                for sid, info in sorted(federation_members.items(), key=lambda kv: str(kv[0]))
            }
            out["member_count"] = len(out["members"])
        return out

    @staticmethod
    def apply_knowledge_to_snapshot(
        snapshot: Mapping[str, Any],
        *,
        domain_knowledge: Sequence[DomainKnowledgeEntry | Mapping[str, Any]] | None = None,
        table_descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, str] | None = None,
        schema_graph: SchemaGraph | None = None,
    ) -> dict[str, Any]:
        """Return a copy of *snapshot* with space DK and/or description overlays updated."""
        if domain_knowledge is None and table_descriptions is None and column_descriptions is None:
            raise ConfigError(
                "apply_knowledge requires domain_knowledge and/or table_descriptions and/or column_descriptions"
            )
        out = dict(snapshot)
        scope_tables = {str(t) for t in (out.get("tables") or ())}
        scope_columns = {str(c) for c in (out.get("columns") or ())}
        if domain_knowledge is not None:
            normalized: list[DomainKnowledgeEntry] = []
            seen: set[str] = set()
            for item in domain_knowledge:
                if isinstance(item, DomainKnowledgeEntry):
                    entry = DomainKnowledgeEntry.normalize(item)
                elif isinstance(item, Mapping):
                    key = str(item.get("key") or "").strip()
                    text = str(item.get("text") or "").strip()
                    kind = str(item.get("kind") or "glossary").strip() or "glossary"
                    if not key or not text:
                        continue
                    referenced_entities = MainSpaceOps._domain_knowledge_referenced_entities_from_item(item)
                    entry = DomainKnowledgeEntry.normalize(
                        DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced_entities)
                    )
                else:
                    raise ConfigError("domain_knowledge entries must be DomainKnowledgeEntry or mappings")
                if entry.key in seen:
                    continue
                seen.add(entry.key)
                normalized.append(entry)
            if schema_graph is not None:
                normalized = list(DomainKnowledgeState.validate_entries(tuple(normalized), schema_graph))
            out["domain_knowledge"] = [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in normalized]
            out["domain_knowledge_digest"] = DomainKnowledgeState.digest_for(tuple(normalized))
        if table_descriptions is not None:
            current = dict(out.get("table_descriptions") or {})
            owners = dict(out.get("_table_description_owners") or {})
            for tname, desc in table_descriptions.items():
                name = str(tname)
                if scope_tables and name not in scope_tables:
                    raise ConfigError(f"table {name!r} is outside aetherspace scope")
                text = str(desc or "").strip()
                if text:
                    current[name] = text
                    owners[name] = DescriptionOwner.SPACE_NOTES.value
                else:
                    current.pop(name, None)
                    owners.pop(name, None)
            out["table_descriptions"] = current
            if owners:
                out["_table_description_owners"] = owners
            else:
                out.pop("_table_description_owners", None)
        if column_descriptions is not None:
            column_meta = dict(out.get("column_meta") or {})
            for qc, desc in column_descriptions.items():
                name = str(qc)
                if scope_columns and name not in scope_columns:
                    raise ConfigError(f"column {name!r} is outside aetherspace scope")
                text = str(desc or "").strip()
                meta_entry = dict(column_meta.get(name) or {})
                if text:
                    meta_entry["description"] = text
                    meta_entry["description_owner"] = DescriptionOwner.SPACE_NOTES.value
                    column_meta[name] = meta_entry
                else:
                    meta_entry.pop("description", None)
                    meta_entry.pop("description_owner", None)
                    if meta_entry:
                        column_meta[name] = meta_entry
                    else:
                        column_meta.pop(name, None)
            out["column_meta"] = column_meta
        return out

    @staticmethod
    def normalize_domain_knowledge_entries(
        domain_knowledge: Sequence[DomainKnowledgeEntry | Mapping[str, Any]],
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Normalize caller DK entries into a deduplicated tuple."""
        normalized: list[DomainKnowledgeEntry] = []
        seen: set[str] = set()
        for item in domain_knowledge:
            if isinstance(item, DomainKnowledgeEntry):
                entry = DomainKnowledgeEntry.normalize(item)
            elif isinstance(item, Mapping):
                require_exact_keys(
                    item,
                    allowed=MainSpaceOps._KNOWLEDGE_ENTRY_KEYS,
                    required=frozenset({"key", "text", "referenced_entities"}),
                    context="domain_knowledge entry",
                )
                key = str(item.get("key") or "").strip()
                text = str(item.get("text") or "").strip()
                kind = str(item.get("kind") or "glossary").strip() or "glossary"
                if not key or not text:
                    raise ConfigError("domain_knowledge entry requires non-empty key and text")
                referenced_entities = MainSpaceOps._domain_knowledge_referenced_entities_from_item(item)
                entry = DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced_entities)
                )
            else:
                raise ConfigError("domain_knowledge entries must be DomainKnowledgeEntry or mappings")
            if entry.key in seen:
                continue
            seen.add(entry.key)
            normalized.append(entry)
        return tuple(normalized)

    @staticmethod
    def apply_master_space_knowledge_to_graph(
        schema_graph: SchemaGraph,
        *,
        schema_json_path: str,
        table_descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Upsert master description prose onto the live schema graph and persist the cache."""
        if table_descriptions is None and column_descriptions is None:
            return
        if table_descriptions is not None:
            for tname, desc in table_descriptions.items():
                name = str(tname)
                tbl = schema_graph.tables.get(name)
                if tbl is None:
                    raise ConfigError(f"unknown table {name!r}")
                text = str(desc or "").strip()
                DescriptionOwner.set_on(tbl, text, DescriptionOwner.USER_OVERRIDE)
        if column_descriptions is not None:
            for qc, desc in column_descriptions.items():
                name = str(qc)
                if "." not in name:
                    raise ConfigError(f"column {name!r} must be table.column")
                tname, cname = name.split(".", 1)
                tbl = schema_graph.tables.get(tname)
                if tbl is None or cname not in tbl.columns:
                    raise ConfigError(f"unknown column {name!r}")
                col = tbl.columns[cname]
                if not MainSpaceOps.column_eligible_for_space_allowlist(col):
                    raise ConfigError(
                        f"column {name!r} cannot receive a description "
                        f"(hidden, restricted, or denied columns are excluded)"
                    )
                text = str(desc or "").strip()
                DescriptionOwner.set_on(col, text, DescriptionOwner.USER_OVERRIDE)
        save_schema_to_cache(schema_graph, schema_json_path)

    @staticmethod
    def _master_graph_has_notes_enrichment(
        master_graph: SchemaGraph,
        engine_domain_knowledge: Sequence[DomainKnowledgeEntry],
    ) -> bool:
        """True when master already carries notes-derived DK, structural facts, or NOTES-owned descriptions."""
        if engine_domain_knowledge:
            return True
        if tuple(getattr(master_graph, "structural_knowledge", ()) or ()):
            return True
        for table in master_graph.tables.values():
            if table.description_owner == DescriptionOwner.NOTES:
                return True
            for col in table.columns.values():
                if col.description_owner == DescriptionOwner.NOTES:
                    return True
        return False

    @staticmethod
    def _resolve_space_over_engine_knowledge(
        engine_items: Sequence[Any],
        space_items: Sequence[Any],
        *,
        merge_both: Callable[[Sequence[Any], Sequence[Any]], Sequence[Any]],
    ) -> tuple[Any, ...]:
        """Carry one side when the other is empty; LLM-merge only when both contribute."""
        engine_t = tuple(engine_items)
        space_t = tuple(space_items)
        if not space_t:
            return engine_t
        if not engine_t:
            return space_t
        return tuple(merge_both(engine_t, space_t))

    @staticmethod
    def merge_overlay_knowledge_layers(
        *,
        base_domain_knowledge: Sequence[DomainKnowledgeEntry],
        base_structural_knowledge: Sequence[StructuralKnowledgeFact],
        notes_content: str | None,
        extract_schema: SchemaGraph,
        artifacts_dir: str | os.PathLike[str] | None = None,
        overrides_path: str | os.PathLike[str] | None = None,
    ) -> tuple[tuple[DomainKnowledgeEntry, ...], tuple[StructuralKnowledgeFact, ...], bool]:
        """Shared space/federation overlay: extract notes once, then space-over-engine merge. Returns ``(final_dk, final_structural, has_overlay_notes)``."""
        notes_text = notes_content.strip() if notes_content and str(notes_content).strip() else None
        overlay_dk: tuple[DomainKnowledgeEntry, ...] = ()
        overlay_structural: tuple[StructuralKnowledgeFact, ...] = ()
        if notes_text and notes_content is not None:
            overlay_dk, overlay_structural = resolve_knowledge_extraction_for_schema(
                extract_schema,
                notes_content,
                artifacts_dir=artifacts_dir,
                overrides_path=overrides_path,
                extract_knowledge_from_notes=extract_knowledge_from_notes,
            )
        final_dk = MainSpaceOps._resolve_space_over_engine_knowledge(
            tuple(base_domain_knowledge),
            overlay_dk,
            merge_both=merge_domain_knowledge_notes_overlay,
        )
        final_structural = MainSpaceOps._resolve_space_over_engine_knowledge(
            tuple(base_structural_knowledge),
            overlay_structural,
            merge_both=merge_structural_knowledge_notes_overlay,
        )
        return final_dk, final_structural, bool(notes_text)

    @staticmethod
    def derive_caller_scoped_domain_knowledge(
        *,
        engine_entries: Sequence[DomainKnowledgeEntry],
        schema: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        space_snapshot: Mapping[str, Any] | None = None,
        space_tables: set[str] | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Merge engine and space DK, then filter to the caller's visible entity set."""
        space_dk = MainSpaceOps.entries_from_snapshot_domain_knowledge(space_snapshot)
        merged = MainSpaceOps.merge_domain_knowledge(engine_entries, space_dk) if space_dk else tuple(engine_entries)
        all_tables = set(schema.tables.keys())
        visible_tables = {
            t
            for t in schema.tables
            if MainSpaceOps.table_allowed_for_visibility(t, schema, scope_ctx, visible_objects)
            and (space_tables is None or t in space_tables)
        }
        return MainSpaceOps.secure_domain_knowledge_for_visibility(
            merged,
            security_schema=schema,
            visible_table_names=visible_tables,
            all_schema_table_names=all_tables,
        )

    @staticmethod
    def secure_domain_knowledge_for_visibility(
        entries: Sequence[DomainKnowledgeEntry],
        *,
        security_schema: SchemaGraph,
        visible_table_names: set[str] | None,
        all_schema_table_names: set[str] | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Visibility key filter, then scope-subset check on ``referenced_entities``, then drop entries that name schema- known HIDDEN columns in text."""
        filtered = MainSpaceOps.filter_domain_knowledge_for_visibility(
            entries,
            visible_table_names=visible_table_names,
            all_schema_table_names=all_schema_table_names,
        )
        scope = KnowledgeScope.from_visible_tables(security_schema, visible_table_names)
        filtered = tuple(e for e in filtered if e.in_scope(scope))
        return filter_schema_anchored_domain_knowledge(filtered, security_schema)

    @staticmethod
    def enrich_descriptions_from_structural_knowledge(
        enrich_graph: SchemaGraph,
        structural: Sequence[StructuralKnowledgeFact],
        *,
        prefer_base_descriptions: bool,
        description_owner: DescriptionOwner,
        noop_label: str | None = None,
        has_notes: bool = False,
    ) -> bool:
        """LLM-scope/refine descriptions onto *enrich_graph*. Returns True when any description was written."""
        if not EngineConfig.llm_credentials_configured():
            return False
        classifications = llm_enrich_schema_from_structural_knowledge(
            enrich_graph,
            structural,
            prefer_base_descriptions=prefer_base_descriptions,
        )
        enriched_any = False
        for tname, table in enrich_graph.tables.items():
            if tname not in classifications:
                continue
            _table_role, desc, col_classes = classifications[tname]
            if desc and str(desc).strip():
                DescriptionOwner.set_on(table, str(desc).strip(), description_owner)
                enriched_any = True
            for col_name, (_col_role, col_description, _sensitivity) in col_classes.items():
                col = table.columns.get(col_name)
                if col is None:
                    continue
                if col_description and str(col_description).strip():
                    DescriptionOwner.set_on(col, str(col_description).strip(), description_owner)
                    enriched_any = True
        if not enriched_any and has_notes and noop_label:
            emit_description_enrichment_noop(noop_label)
        return enriched_any

    @staticmethod
    def enrich_space_snapshot_with_notes(
        snapshot: dict[str, Any],
        master_graph: SchemaGraph,
        space_context: SpaceContext,
        notes_file: str | None = None,
        *,
        notes: str | None = None,
        engine_domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
    ) -> dict[str, Any]:
        """Attach space/master knowledge and scope descriptions for the space subset. Space notes are optional. When absent, master structural knowledge and domain knowledge are carried through without merging. When only one side has knowledge, that side is final. Merge runs only when both sides contribute. Description scoping always runs when LLM credentials are configured: with space notes, start from profile base descriptions; without space notes, inherit master notes-enriched prose when present, otherwise profile base — then remove out-of-scope mentions."""
        if notes is not None and notes_file is not None:
            raise ConfigError("set at most one of notes and notes_file")
        notes_content: str | None = None
        if notes is not None:
            notes_content = str(notes)
        elif notes_file is not None and str(notes_file).strip():
            path = os.path.expanduser(str(notes_file).strip())
            if not os.path.isfile(path):
                raise ConfigError(f"notes_file not found: {notes_file!r}")
            with open(path, encoding="utf-8") as fh:
                notes_content = fh.read()
        notes_text = notes_content.strip() if notes_content and notes_content.strip() else None
        out = dict(snapshot)
        out["notes"] = notes_text
        out["notes_hash"] = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest() if notes_content else ""
        subset_sg = MainSpaceOps.build_subset_schema_for_space_notes(master_graph, space_context)
        engine_dk = tuple(engine_domain_knowledge or ())
        engine_structural = tuple(getattr(master_graph, "structural_knowledge", ()) or ())
        final_dk, final_structural, has_space_notes = MainSpaceOps.merge_overlay_knowledge_layers(
            base_domain_knowledge=engine_dk,
            base_structural_knowledge=engine_structural,
            notes_content=notes_content,
            extract_schema=subset_sg,
        )
        space_table_names = {str(t) for t in (out.get("tables") or ())}
        if space_table_names:
            final_dk = MainSpaceOps.secure_domain_knowledge_for_visibility(
                final_dk,
                security_schema=master_graph,
                visible_table_names=space_table_names,
                all_schema_table_names=set(master_graph.tables.keys()),
            )
        else:
            final_dk = filter_schema_anchored_domain_knowledge(final_dk, master_graph)
        out["domain_knowledge"] = [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in final_dk]
        out["domain_knowledge_digest"] = DomainKnowledgeState.digest_for(final_dk)
        out["structural_knowledge"] = [f.to_dict() for f in final_structural]
        if not EngineConfig.llm_credentials_configured():
            return out
        has_master_notes = MainSpaceOps._master_graph_has_notes_enrichment(master_graph, engine_dk)
        prefer_base = True if has_space_notes else (not has_master_notes)
        classifications = llm_enrich_schema_from_structural_knowledge(
            subset_sg,
            final_structural,
            prefer_base_descriptions=prefer_base,
        )
        table_descriptions = dict(out.get("table_descriptions") or {})
        column_meta = dict(out.get("column_meta") or {})
        scope_cols = frozenset(str(c) for c in (out.get("columns") or ()))
        enriched_any = False
        description_owner = DescriptionOwner.SPACE_NOTES if has_space_notes else DescriptionOwner.LLM_REFINEMENT
        for tname in out.get("tables") or ():
            if tname not in classifications:
                continue
            _table_role, desc, col_classes = classifications[tname]
            if desc and str(desc).strip():
                table_descriptions[str(tname)] = str(desc).strip()
                enriched_any = True
            tm = subset_sg.tables.get(str(tname))
            if tm is None:
                continue
            for col_name, (_col_role, col_description, sensitivity) in col_classes.items():
                if col_name not in tm.columns:
                    continue
                qc = f"{tname}.{col_name}"
                if scope_cols and qc not in scope_cols:
                    continue
                entry = dict(column_meta.get(qc) or {})
                if col_description and str(col_description).strip():
                    entry["description"] = str(col_description).strip()
                    entry["description_owner"] = description_owner.value
                    enriched_any = True
                if sensitivity is not None and str(sensitivity) not in ("", "none"):
                    entry["sensitivity"] = str(sensitivity)
                if entry:
                    column_meta[qc] = entry
        if table_descriptions:
            out["_table_description_owners"] = {str(t): description_owner.value for t in table_descriptions}
        out["table_descriptions"] = table_descriptions
        out["column_meta"] = column_meta
        verify_scope = KnowledgeScope.from_visible_tables(master_graph, space_table_names or None)
        verify_tokens = out_of_scope_description_tokens(master_graph, verify_scope)
        raise_if_flat_descriptions_name_out_of_scope_entities(
            table_descriptions, column_meta, verify_tokens, context="aetherspace description"
        )
        if not enriched_any and notes_text:
            emit_description_enrichment_noop("aetherspace_notes")
        return out

    @staticmethod
    def enrich_federation_composite_knowledge(
        schema_graph: SchemaGraph,
        *,
        member_domain_knowledge: Sequence[tuple[str, Sequence[DomainKnowledgeEntry]]],
        member_structural_knowledge: Sequence[tuple[str, Sequence[StructuralKnowledgeFact]]],
        notes_content: str | None,
        all_schema_table_names: set[str] | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Merge member DK/structural with federation-notes overlays and enrich composite descriptions. Same overlay/security path as spaces (``merge_overlay_knowledge_layers`` + ``secure_domain_knowledge_for_visibility`` + structural description enrich). Federation notes prefer over peer-merged member knowledge; hidden-column DK refs are dropped against the composite schema."""
        peer_dk = merge_domain_knowledge_federation_peers(member_domain_knowledge)
        peer_structural = merge_structural_knowledge_federation_peers(member_structural_knowledge)
        final_dk, final_structural, has_fed_notes = MainSpaceOps.merge_overlay_knowledge_layers(
            base_domain_knowledge=peer_dk,
            base_structural_knowledge=peer_structural,
            notes_content=notes_content,
            extract_schema=schema_graph,
        )
        composite_tables = set(schema_graph.tables.keys())
        schema_universe = set(all_schema_table_names or ()) | composite_tables
        final_dk = MainSpaceOps.secure_domain_knowledge_for_visibility(
            final_dk,
            security_schema=schema_graph,
            visible_table_names=composite_tables,
            all_schema_table_names=schema_universe,
        )
        schema_graph.structural_knowledge = final_structural
        has_member_notes = MainSpaceOps._master_graph_has_notes_enrichment(schema_graph, peer_dk)
        prefer_base = True if has_fed_notes else (not has_member_notes)
        MainSpaceOps.enrich_descriptions_from_structural_knowledge(
            schema_graph,
            final_structural,
            prefer_base_descriptions=prefer_base,
            description_owner=DescriptionOwner.NOTES if has_fed_notes else DescriptionOwner.LLM_REFINEMENT,
            noop_label="federation_notes",
            has_notes=has_fed_notes,
        )
        return final_dk

    @staticmethod
    def _remap_qualified_column(
        spec: str, tmap: Mapping[str, str], colmaps: Mapping[str, Mapping[str, str]]
    ) -> str | None:
        raw = str(spec).strip()
        if raw.count(".") != 1:
            return None
        tbl, col = raw.split(".", 1)
        nt = tmap.get(tbl, tbl)
        nc = colmaps.get(tbl, {}).get(col, colmaps.get(nt, {}).get(col, col))
        return f"{nt}.{nc}"

    @staticmethod
    def _prune_remap_string_list(
        values: list[str],
        *,
        tmap: Mapping[str, str],
        colmaps: Mapping[str, Mapping[str, str]],
        drop_tables: frozenset[str],
        drop_columns: frozenset[str],
        column_specs: bool,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            spec = str(raw).strip()
            if not spec:
                continue
            if column_specs:
                if spec.count(".") != 1:
                    continue
                tbl, col = spec.split(".", 1)
                if tbl in drop_tables or spec in drop_columns or f"{tbl}.{col}" in drop_columns:
                    continue
                remapped = MainSpaceOps._remap_qualified_column(spec, tmap, colmaps)
                if remapped is None:
                    continue
                spec = remapped
            else:
                if spec in drop_tables:
                    continue
                spec = tmap.get(spec, spec)
            if spec not in seen:
                seen.add(spec)
                out.append(spec)
        return sorted(out)

    @staticmethod
    def _apply_structural_edit_to_aetherspace_snapshot(
        snapshot: dict[str, Any],
        *,
        tmap: Mapping[str, str],
        colmaps: Mapping[str, Mapping[str, str]],
        drop_tables: frozenset[str],
        drop_columns: frozenset[str],
        column_retypes: tuple[tuple[str, str, str], ...] = (),
    ) -> dict[str, Any]:
        out = dict(snapshot)
        tables = MainSpaceOps._prune_remap_string_list(
            [str(t) for t in (out.get("tables") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=False,
        )
        columns = MainSpaceOps._prune_remap_string_list(
            [str(c) for c in (out.get("columns") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=True,
        )
        table_descriptions: dict[str, str] = {}
        for key, val in dict(out.get("table_descriptions") or {}).items():
            tbl = str(key).strip()
            if tbl in drop_tables:
                continue
            nt = tmap.get(tbl, tbl)
            if isinstance(val, str) and val.strip():
                table_descriptions[nt] = val.strip()
        column_meta: dict[str, dict[str, Any]] = {}
        for key, meta in dict(out.get("column_meta") or {}).items():
            remapped = MainSpaceOps._remap_qualified_column(str(key), tmap, colmaps)
            if remapped is None:
                continue
            tbl = remapped.split(".", 1)[0]
            if tbl in drop_tables or remapped in drop_columns:
                continue
            if isinstance(meta, dict):
                entry = dict(meta)
                if column_retypes:
                    tbl, col = remapped.split(".", 1)
                    for rt_tbl, rt_col, new_dt in column_retypes:
                        if rt_tbl == tbl and rt_col == col:
                            entry["value_type"] = new_dt
                column_meta[remapped] = entry
        out["tables"] = tables
        out["columns"] = columns
        out["table_descriptions"] = table_descriptions
        out["column_meta"] = column_meta
        out["deny_objects"] = MainSpaceOps._prune_remap_string_list(
            [str(t) for t in (out.get("deny_objects") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=False,
        )
        out["deny_columns"] = MainSpaceOps._prune_remap_string_list(
            [str(c) for c in (out.get("deny_columns") or ())],
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            column_specs=True,
        )
        dk_entries = MainSpaceOps.entries_from_snapshot_domain_knowledge(out)
        if dk_entries or out.get("domain_knowledge") is not None:
            migrated_dk = MainSpaceOps.migrate_domain_knowledge_entries(
                dk_entries,
                tmap=tmap,
                colmaps=colmaps,
                drop_tables=drop_tables,
                drop_columns=drop_columns,
            )
            out["domain_knowledge"] = [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in migrated_dk]
        raw_structural = out.get("structural_knowledge")
        if isinstance(raw_structural, list):
            facts: list[StructuralKnowledgeFact] = []
            for item in raw_structural:
                if not isinstance(item, Mapping):
                    continue
                kind = str(item.get("kind") or "").strip()
                text = str(item.get("text") or "").strip()
                if not text or not kind:
                    continue
                if kind.lower() == "residual":
                    raise ConfigError("structural knowledge kind residual is invalid; re-derive from notes")
                raw_referenced = item.get("referenced_entities", [])
                if not isinstance(raw_referenced, list):
                    raise ConfigError("structural_knowledge referenced_entities must be a list")
                referenced_entities = frozenset(str(r).strip() for r in raw_referenced if str(r).strip())
                payload_raw = item.get("payload")
                payload = payload_raw if isinstance(payload_raw, dict) else None
                facts.append(
                    StructuralKnowledgeFact.normalize(
                        StructuralKnowledgeFact(
                            kind=kind,
                            text=text,
                            referenced_entities=referenced_entities,
                            payload=payload,
                        )
                    )
                )
            migrated_facts = MainSpaceOps.migrate_structural_knowledge_facts(
                facts,
                tmap=tmap,
                colmaps=colmaps,
                drop_tables=drop_tables,
                drop_columns=drop_columns,
            )
            out["structural_knowledge"] = [f.to_dict() for f in migrated_facts]
        return out

    @staticmethod
    def _migrate_single_entity_name(
        name: str,
        *,
        tmap: Mapping[str, str],
        colmaps: Mapping[str, Mapping[str, str]],
        drop_tables_l: set[str],
        drop_cols_l: set[str],
    ) -> str | None:
        """Remap one bare table or qualified ``table.column`` name across a rename migration; ``None`` when it names something dropped."""
        raw = str(name or "").strip()
        if not raw:
            return None
        raw_l = raw.lower()
        if "." in raw_l:
            remapped = MainSpaceOps._remap_qualified_column(raw, tmap, colmaps)
            if remapped is None:
                return None
            parent, _col = remapped.split(".", 1)
            if parent.lower() in drop_tables_l or remapped.lower() in drop_cols_l:
                return None
            return remapped
        if raw_l in drop_tables_l:
            return None
        new_name = tmap.get(raw, tmap.get(raw_l, raw))
        for old, new in tmap.items():
            if old.lower() == raw_l:
                new_name = new
                break
        if str(new_name).lower() in drop_tables_l:
            return None
        return str(new_name)

    @staticmethod
    def migrate_domain_knowledge_entries(
        entries: Sequence[DomainKnowledgeEntry],
        *,
        tmap: Mapping[str, str],
        colmaps: Mapping[str, Mapping[str, str]],
        drop_tables: frozenset[str],
        drop_columns: frozenset[str],
        schema: SchemaGraph | None = None,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Prune deleted table/column DK keys and remap renamed keys and referenced_entities; then HIDDEN-scrub when *schema* is set. An entry whose key survives but whose referenced_entities name something dropped is itself dropped — re-deriving from stale references would leak, so the entry does not carry forward."""
        drop_tables_l = {t.lower() for t in drop_tables}
        drop_cols_l = {c.lower() for c in drop_columns}
        kept: list[DomainKnowledgeEntry] = []
        seen: set[str] = set()
        for entry in entries:
            new_key = MainSpaceOps._migrate_single_entity_name(
                entry.key, tmap=tmap, colmaps=colmaps, drop_tables_l=drop_tables_l, drop_cols_l=drop_cols_l
            )
            if new_key is None:
                continue
            if entry.referenced_entities:
                migrated_refs: set[str] = set()
                refs_dropped = False
                for ref in entry.referenced_entities:
                    new_ref = MainSpaceOps._migrate_single_entity_name(
                        ref, tmap=tmap, colmaps=colmaps, drop_tables_l=drop_tables_l, drop_cols_l=drop_cols_l
                    )
                    if new_ref is None:
                        refs_dropped = True
                        break
                    migrated_refs.add(new_ref)
                if refs_dropped:
                    continue
                referenced_entities = frozenset(migrated_refs)
            else:
                referenced_entities = frozenset()
            try:
                migrated = DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(
                        key=new_key, kind=entry.kind, text=entry.text, referenced_entities=referenced_entities
                    )
                )
            except ConfigError:
                continue
            if migrated.key in seen:
                continue
            seen.add(migrated.key)
            kept.append(migrated)
        result = tuple(kept)
        if schema is not None:
            remaining = set(schema.tables.keys())
            result = MainSpaceOps.secure_domain_knowledge_for_visibility(
                result,
                security_schema=schema,
                visible_table_names=remaining,
                all_schema_table_names=remaining | {str(t) for t in drop_tables},
            )
        return result

    @staticmethod
    def migrate_structural_knowledge_facts(
        facts: Sequence[StructuralKnowledgeFact],
        *,
        tmap: Mapping[str, str],
        colmaps: Mapping[str, Mapping[str, str]],
        drop_tables: frozenset[str],
        drop_columns: frozenset[str],
    ) -> tuple[StructuralKnowledgeFact, ...]:
        """Drop structural facts that name deleted objects; remap reference sets only (no prose rewrite)."""
        drop_tables_l = {t.lower() for t in drop_tables}
        drop_cols_l = {c.lower() for c in drop_columns}

        out: list[StructuralKnowledgeFact] = []
        for fact in facts:
            migrated_refs: set[str] = set()
            refs_dropped = False
            for ref in fact.referenced_entities:
                new_ref = MainSpaceOps._migrate_single_entity_name(
                    ref,
                    tmap=tmap,
                    colmaps=colmaps,
                    drop_tables_l=drop_tables_l,
                    drop_cols_l=drop_cols_l,
                )
                if new_ref is None:
                    refs_dropped = True
                    break
                migrated_refs.add(new_ref)
            if refs_dropped:
                continue
            referenced_entities = frozenset(migrated_refs)
            try:
                out.append(
                    StructuralKnowledgeFact.normalize(
                        StructuralKnowledgeFact(
                            kind=fact.kind,
                            text=str(fact.text or "").strip(),
                            referenced_entities=referenced_entities,
                            payload=fact.payload,
                        )
                    )
                )
            except ConfigError:
                continue
        return tuple(out)

    @staticmethod
    def migrate_engine_knowledge_artifacts(
        engine_dir: str,
        schema: SchemaGraph,
        *,
        schema_json_path: str | None = None,
        dropped_tables: tuple[str, ...] = (),
        dropped_columns: tuple[str, ...] = (),
        table_renames: tuple[tuple[str, str], ...] = (),
        column_renames: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        """Migrate on-disk DK + in-graph structural knowledge for delete/rename, then persist."""
        tmap = {old: new for old, new in table_renames if old and new and old != new}
        colmaps: dict[str, dict[str, str]] = defaultdict(dict)
        for ot, oc, nc in column_renames:
            colmaps[ot][oc] = nc
            nt = tmap.get(ot, ot)
            if nt != ot:
                colmaps.setdefault(nt, {})[oc] = nc
        drop_tables = frozenset(dropped_tables)
        drop_columns = frozenset(dropped_columns)
        if not (drop_tables or drop_columns or tmap or colmaps):
            return
        schema.structural_knowledge = MainSpaceOps.migrate_structural_knowledge_facts(
            tuple(getattr(schema, "structural_knowledge", ()) or ()),
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
        )
        cache_path = schema_json_path or str(MainSpaceOps.engine_schema_json_path(engine_dir))
        if cache_path:
            try:
                save_schema_to_cache(schema, cache_path)
            except OSError:
                pass
        loaded = load_domain_knowledge_artifact(engine_dir, schema, require_notes_match=False)
        if loaded is None:
            return
        migrated = MainSpaceOps.migrate_domain_knowledge_entries(
            loaded,
            tmap=tmap,
            colmaps=colmaps,
            drop_tables=drop_tables,
            drop_columns=drop_columns,
            schema=schema,
        )
        stamps = knowledge_artifact_save_stamps(schema)
        try:
            save_domain_knowledge_artifact(engine_dir, migrated, **stamps)
        except OSError:
            pass

    @staticmethod
    def apply_structural_migration_to_aetherspace_snapshots(
        engine_dir: str,
        *,
        dropped_tables: tuple[str, ...] = (),
        dropped_columns: tuple[str, ...] = (),
        table_renames: tuple[tuple[str, str], ...] = (),
        column_renames: tuple[tuple[str, str, str], ...] = (),
        column_retypes: tuple[tuple[str, str, str], ...] = (),
    ) -> int:
        """Prune or remap table/column references inside every persisted aetherspace snapshot."""
        tmap = {old: new for old, new in table_renames if old and new and old != new}
        colmaps: dict[str, dict[str, str]] = defaultdict(dict)
        for ot, oc, nc in column_renames:
            colmaps[ot][oc] = nc
            nt = tmap.get(ot, ot)
            if nt != ot:
                colmaps.setdefault(nt, {})[oc] = nc
        drop_tables = frozenset(dropped_tables)
        drop_columns = frozenset(dropped_columns)
        if not (drop_tables or drop_columns or tmap or colmaps or column_retypes):
            return 0
        updated = 0
        root = MainSpaceOps._aetherspace_dir(engine_dir)
        if not os.path.isdir(root):
            return 0
        for entry in os.listdir(root):
            if not entry.endswith(".json"):
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
            edited = MainSpaceOps._apply_structural_edit_to_aetherspace_snapshot(
                payload,
                tmap=tmap,
                colmaps=colmaps,
                drop_tables=drop_tables,
                drop_columns=drop_columns,
                column_retypes=column_retypes,
            )
            if edited != payload:
                MainSpaceOps.write_json_atomic(path, edited)
                updated += 1
        if updated:
            schema_path = MainSpaceOps.engine_schema_json_path(engine_dir)
            master_graph = load_schema_graph_snapshot(schema_path)
            if master_graph is not None:
                MainSpaceOps.reenrich_aetherspace_snapshots_with_notes(engine_dir, master_graph)
        return updated

    @staticmethod
    def reenrich_aetherspace_snapshots_with_notes(
        engine_dir: str,
        master_graph: SchemaGraph,
        *,
        engine_domain_knowledge: Sequence[DomainKnowledgeEntry] | None = None,
    ) -> int:
        """Re-run notes enrichment on persisted snapshots that carry inline space notes."""
        updated = 0
        for uid, label in MainSpaceOps.list_saved_aetherspace_entries(engine_dir):
            snap = MainSpaceOps.load_aetherspace_snapshot(engine_dir, uid)
            if snap is None:
                continue
            notes_raw = snap.get("notes")
            if not isinstance(notes_raw, str) or not notes_raw.strip():
                continue
            space_context = SpaceContext(
                tables=frozenset(str(t) for t in (snap.get("tables") or ())),
                columns=frozenset(str(c) for c in (snap.get("columns") or ())),
                deny_objects=frozenset(str(t) for t in (snap.get("deny_objects") or ())),
                deny_columns=frozenset(str(c) for c in (snap.get("deny_columns") or ())),
            )
            enriched = MainSpaceOps.enrich_space_snapshot_with_notes(
                snap,
                master_graph,
                space_context,
                notes=notes_raw,
                engine_domain_knowledge=engine_domain_knowledge,
            )
            enriched["uid"] = uid
            enriched["name"] = str(snap.get("name") or label)
            enriched = MainSpaceOps.filter_space_snapshot_sensitive_columns(enriched, master_graph)
            MainSpaceOps.save_aetherspace_snapshot(engine_dir, uid, enriched)
            updated += 1
        return updated

    @staticmethod
    def apply_structural_migration_to_named_context_specs(
        engine_dir: str,
        *,
        dropped_tables: tuple[str, ...] = (),
        dropped_columns: tuple[str, ...] = (),
        table_renames: tuple[tuple[str, str], ...] = (),
        column_renames: tuple[tuple[str, str, str], ...] = (),
    ) -> int:
        """Prune or remap allow/deny lists inside named ``schema_context.<name>.json`` specs."""
        tmap = {old: new for old, new in table_renames if old and new and old != new}
        colmaps: dict[str, dict[str, str]] = defaultdict(dict)
        for ot, oc, nc in column_renames:
            colmaps[ot][oc] = nc
            nt = tmap.get(ot, ot)
            if nt != ot:
                colmaps.setdefault(nt, {})[oc] = nc
        drop_tables = frozenset(dropped_tables)
        drop_columns = frozenset(dropped_columns)
        if not (drop_tables or drop_columns or tmap or colmaps):
            return 0
        updated = 0
        for path in glob.glob(os.path.join(engine_dir, SCHEMA_CONTEXT_NAMED_SPEC_GLOB)):
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            edited = dict(payload)
            for key, column_specs in (("allow_columns", True), ("deny_columns", True), ("allow_objects", False)):
                raw_vals = payload.get(key)
                if not isinstance(raw_vals, list):
                    continue
                edited[key] = MainSpaceOps._prune_remap_string_list(
                    [str(v) for v in raw_vals],
                    tmap=tmap,
                    colmaps=colmaps,
                    drop_tables=drop_tables,
                    drop_columns=drop_columns,
                    column_specs=column_specs,
                )
            if edited != payload:
                MainSpaceOps.write_json_atomic(path, edited)
                updated += 1
        return updated

    @staticmethod
    def apply_structural_migration_to_persisted_scopes(
        engine_dir: str,
        *,
        dropped_tables: tuple[str, ...] = (),
        dropped_columns: tuple[str, ...] = (),
        table_renames: tuple[tuple[str, str], ...] = (),
        column_renames: tuple[tuple[str, str, str], ...] = (),
        column_retypes: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        """Apply table/column delete/remap migration to aetherspace snapshots and named context specs."""
        MainSpaceOps.apply_structural_migration_to_aetherspace_snapshots(
            engine_dir,
            dropped_tables=dropped_tables,
            dropped_columns=dropped_columns,
            table_renames=table_renames,
            column_renames=column_renames,
            column_retypes=column_retypes,
        )
        MainSpaceOps.apply_structural_migration_to_named_context_specs(
            engine_dir,
            dropped_tables=dropped_tables,
            dropped_columns=dropped_columns,
            table_renames=table_renames,
            column_renames=column_renames,
        )

    @staticmethod
    def _normalize_context_name(name: str) -> str:
        norm = str(name).strip().lower()
        if not norm:
            raise ConfigError("engine context name must be non-empty")
        if "/" in norm or "\\" in norm:
            raise ConfigError(f"invalid engine context name: {name!r}")
        return norm

    @staticmethod
    def _named_schema_context_path(engine_dir: str, name: str) -> str:
        safe = MainSpaceOps._normalize_context_name(name)
        return os.path.join(engine_dir, f"{NAMED_SCHEMA_CONTEXT_PREFIX}{safe}.json")

    @staticmethod
    def validate_scope_list_fields(payload: dict[str, Any]) -> None:
        for key in ("allow_objects", "deny_objects", "deny_columns", "allow_columns"):
            if key not in payload:
                continue
            val = payload[key]
            if val is None:
                continue
            if not isinstance(val, list):
                raise ConfigError(f"{key} must be a list or null, got {type(val).__name__}")

    @staticmethod
    def _schema_context_from_named_payload(payload: dict[str, Any]) -> EngineContext:
        """Reconstruct a named :class:`EngineContext` from a persisted sidecar."""
        MainSpaceOps.validate_scope_list_fields(payload)
        return EngineContext(
            allow_objects=frozenset(str(x) for x in (payload.get("allow_objects") or ())),
            deny_objects=frozenset(str(x) for x in (payload.get("deny_objects") or ())),
            deny_columns=frozenset(str(x) for x in (payload.get("deny_columns") or ())),
            allow_columns=frozenset(str(x) for x in (payload.get("allow_columns") or ())),
        )

    @staticmethod
    def load_named_schema_context(engine_dir: str, name: str) -> EngineContext | None:
        """
        Load a persisted named context spec, or ``None`` when absent.

        Raises:

            ConfigError: When the sidecar exists but its ``version`` is not
            :data:`NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION`. Delete the stale
            file and re-save the named context; there is no migration path.
        """
        if MainSpaceOps._normalize_context_name(name) == MASTER_AETHERSPACE_NAME:
            return None
        path = MainSpaceOps._named_schema_context_path(engine_dir, name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        found = payload.get("version")
        if not format_versions_match(found, NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION):
            raise ConfigError(
                f"named schema context at {path!r} has version {found!r}; "
                f"this build expects {NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION}. "
                f"Delete {path!r} and re-save the named context so it is rewritten "
                f"at the current version."
            )
        return MainSpaceOps._schema_context_from_named_payload(payload)

    @staticmethod
    def save_named_schema_context(engine_dir: str, name: str, ctx: EngineContext) -> str:
        """Persist a named allow/deny spec atomically; return the written path."""
        norm = MainSpaceOps._normalize_context_name(name)
        if norm == MASTER_AETHERSPACE_NAME:
            raise ConfigError("master engine context is derived live and is not persisted as a named sidecar")
        payload: dict[str, Any] = {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "allow_objects": sorted(ctx.allow_objects),
            "deny_objects": sorted(ctx.deny_objects),
            "deny_columns": sorted(ctx.deny_columns),
            "allow_columns": sorted(ctx.allow_columns),
        }
        path = MainSpaceOps._named_schema_context_path(engine_dir, norm)
        MainSpaceOps.write_json_atomic(path, payload)
        return path

    @staticmethod
    def list_named_schema_context_names(engine_dir: str) -> tuple[str, ...]:
        """Return sorted saved named engine-context names (excluding ``master``)."""
        if not os.path.isdir(engine_dir):
            return ()
        prefix = NAMED_SCHEMA_CONTEXT_PREFIX
        suffix = ".json"
        names: list[str] = []
        for entry in os.listdir(engine_dir):
            if not entry.startswith(prefix) or not entry.endswith(suffix):
                continue
            stem = entry[len(prefix) : -len(suffix)]
            if stem and stem != MASTER_AETHERSPACE_NAME:
                names.append(stem)
        return tuple(sorted(names))

    @staticmethod
    def engine_context_references_out_of_scope(
        ctx: EngineContext | FederationContext,
        *,
        visible_tables: frozenset[str],
        visible_columns: frozenset[str],
    ) -> bool:
        """Return True when any allow/deny field names an object outside caller visibility."""
        for obj in ctx.allow_objects | ctx.deny_objects:
            if obj not in visible_tables:
                return True
        for qc in ctx.deny_columns | ctx.allow_columns:
            if qc not in visible_columns:
                return True
        return False

    @staticmethod
    def consumer_safe_scope_context_fields(
        ctx: EngineContext | FederationContext,
        *,
        visible_tables: frozenset[str],
        visible_columns: frozenset[str],
    ) -> dict[str, Any]:
        """Return consumer-safe allow/deny fields without out-of-scope object names."""
        fields: dict[str, Any] = {
            "allow_objects": sorted(obj for obj in ctx.allow_objects if obj in visible_tables),
            "allow_columns": sorted(qc for qc in ctx.allow_columns if qc in visible_columns),
            "deny_objects_count": len(ctx.deny_objects),
            "deny_columns_count": len(ctx.deny_columns),
        }
        if isinstance(ctx, FederationContext):
            fields["include"] = str(ctx.include)
        return fields

    @staticmethod
    def consumer_safe_schema_context_export(
        ctx: EngineContext,
        *,
        name: str,
        visible_tables: frozenset[str],
        visible_columns: frozenset[str],
    ) -> dict[str, Any]:
        """Build a consumer-facing engine-context export without out-of- scope object names."""
        return {
            "version": NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION,
            "name": name,
            **MainSpaceOps.consumer_safe_scope_context_fields(
                ctx,
                visible_tables=visible_tables,
                visible_columns=visible_columns,
            ),
        }

    @staticmethod
    def list_consumer_visible_named_context_names(
        engine_dir: str,
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
    ) -> tuple[str, ...]:
        """Return named engine-context names that do not reference out- of-scope objects."""
        visible_tables = MainSpaceOps.effective_visible_tables(schema_graph, scope_ctx, visible_objects)
        visible_columns = MainSpaceOps.effective_visible_columns(schema_graph, scope_ctx, visible_objects)
        out: list[str] = []
        for name in MainSpaceOps.list_named_schema_context_names(engine_dir):
            loaded = MainSpaceOps.load_named_schema_context(engine_dir, name)
            if loaded is None:
                continue
            if MainSpaceOps.engine_context_references_out_of_scope(
                loaded,
                visible_tables=visible_tables,
                visible_columns=visible_columns,
            ):
                continue
            out.append(name)
        return tuple(out)

    @staticmethod
    def build_named_schema_context_export(
        engine_dir: str,
        name: str,
        master_context: EngineContext,
        *,
        schema_graph: SchemaGraph | None = None,
        scope_ctx: EngineContext | FederationContext | None = None,
        visible_objects: frozenset[str] | None = None,
        schema_role: SchemaRole = SchemaRole.OWNER,
    ) -> dict[str, Any]:
        """Build a read-only export document for one engine context (no file I/O)."""
        norm = MainSpaceOps._normalize_context_name(name)
        if norm == MASTER_AETHERSPACE_NAME:
            ctx = master_context
        else:
            loaded = MainSpaceOps.load_named_schema_context(engine_dir, norm)
            if loaded is None:
                raise ConfigError(f"unknown engine context {name!r}")
            ctx = loaded
        if schema_role == SchemaRole.CONSUMER:
            if schema_graph is None:
                raise ConfigError("consumer export_context requires a schema graph")
            visible_tables = MainSpaceOps.effective_visible_tables(schema_graph, scope_ctx, visible_objects)
            visible_columns = MainSpaceOps.effective_visible_columns(schema_graph, scope_ctx, visible_objects)
            snap = MainSpaceOps.consumer_safe_schema_context_export(
                ctx,
                name=norm,
                visible_tables=visible_tables,
                visible_columns=visible_columns,
            )
        elif norm == MASTER_AETHERSPACE_NAME:
            snap = {
                "name": MASTER_AETHERSPACE_NAME,
                "allow_objects": sorted(master_context.allow_objects),
                "deny_columns": sorted(master_context.deny_columns),
                "allow_columns": sorted(master_context.allow_columns),
            }
        else:
            snap = {
                "name": norm,
                "allow_objects": sorted(ctx.allow_objects),
                "deny_columns": sorted(ctx.deny_columns),
                "allow_columns": sorted(ctx.allow_columns),
            }
        snap.pop("version", None)
        return snap

    @staticmethod
    def validate_named_engine_context_spec(ctx: EngineContext) -> None:
        """Reject master-only fields on a named engine-context registration spec."""
        if ctx.sql_file is not None:
            raise ConfigError("named engine context cannot set sql_file; only master defines DDL")
        if ctx.notes_file is not None:
            raise ConfigError("named engine context cannot set notes_file; only master defines notes")
        if getattr(ctx, "notes", None) is not None:
            raise ConfigError("named engine context cannot set notes; only master defines notes")
        if ctx.include != "tables":
            raise ConfigError("named engine context cannot set include; only master defines include mode")

    @staticmethod
    def validate_named_context_subset(master: EngineContext, named: EngineContext, schema_graph: SchemaGraph) -> None:
        """Ensure *named* is a subset-only refinement of *master* over *schema_graph*."""
        if named.include != master.include:
            raise ConfigError("named EngineContext cannot set include; only master defines include mode")
        graph_tables = set(schema_graph.tables.keys())
        if named.allow_objects:
            if master.allow_objects:
                extra = named.allow_objects - master.allow_objects
                if extra:
                    raise ConfigError(f"named context allow_objects widens master scope: {sorted(extra)!r}")
            else:
                extra = named.allow_objects - graph_tables
                if extra:
                    raise ConfigError(f"named context allow_objects references unknown tables: {sorted(extra)!r}")
        if not named.deny_columns.issuperset(master.deny_columns):
            missing = master.deny_columns - named.deny_columns
            raise ConfigError(f"named context must inherit all master deny_columns; missing {sorted(missing)!r}")
        if named.allow_columns:
            if master.allow_columns:
                extra_cols = named.allow_columns - master.allow_columns
                if extra_cols:
                    raise ConfigError(f"named context allow_columns widens master scope: {sorted(extra_cols)!r}")
        validate_scope_against_graph(schema_graph, named)

    @staticmethod
    def federation_execution_allow_objects(
        master_ctx: FederationContext, composite_tables: frozenset[str]
    ) -> frozenset[str]:
        """Return federation execution allow_objects (validated against members at compose time)."""
        if master_ctx.allow_objects:
            return master_ctx.allow_objects
        return composite_tables

    @staticmethod
    def effective_execution_context(master: EngineContext, active: EngineContext, active_name: str) -> EngineContext:
        """Combine master and active named context into the execution- time RBAC scope."""
        if MainSpaceOps._normalize_context_name(active_name) == MASTER_AETHERSPACE_NAME:
            return EngineContext(
                allow_objects=master.allow_objects,
                include=master.include,
                deny_objects=master.deny_objects,
                deny_columns=master.deny_columns,
                allow_columns=master.allow_columns,
                notes_file=master.notes_file,
                notes=master.notes,
                sql_file=master.sql_file,
            )
        if master.allow_objects:
            if active.allow_objects:
                eff_allow = frozenset(t for t in active.allow_objects if t in master.allow_objects)
            else:
                eff_allow = master.allow_objects
        else:
            eff_allow = active.allow_objects
        eff_deny = master.deny_columns | active.deny_columns
        eff_deny_obj = master.deny_objects | active.deny_objects
        if master.allow_columns:
            if active.allow_columns:
                eff_allow_cols = frozenset(c for c in active.allow_columns if c in master.allow_columns)
            else:
                eff_allow_cols = master.allow_columns
        else:
            eff_allow_cols = active.allow_columns
        return EngineContext(
            allow_objects=eff_allow,
            include=master.include,
            deny_objects=eff_deny_obj,
            deny_columns=eff_deny,
            allow_columns=eff_allow_cols,
        )

    @staticmethod
    def context_allowed_table_set(
        ctx: EngineContext | FederationContext, schema_graph: SchemaGraph, *, mappings: FederationMappings | None = None
    ) -> frozenset[str]:
        """Return tables visible under *ctx* against *schema_graph*."""
        tables = set(schema_graph.tables.keys())
        if ctx.allow_objects:
            tables &= set(ctx.allow_objects)
        if ctx.deny_objects:
            if mappings is not None and isinstance(ctx, FederationContext):
                denied = set(ctx.deny_objects)
                for table_map in mappings.logical_tables:
                    if table_map.semantics not in ("union", "replica"):
                        continue
                    member_tables = {member.table for member in table_map.members}
                    if member_tables & denied == member_tables:
                        denied.discard(table_map.logical)
                tables -= denied
            else:
                tables -= set(ctx.deny_objects)
        return frozenset(tables)

    @staticmethod
    def effective_visible_tables(
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        *,
        mappings: FederationMappings | None = None,
    ) -> frozenset[str]:
        """Return tables visible under execution context ∩ credential objects."""
        if scope_ctx is None:
            tables = set(schema_graph.tables.keys())
        else:
            tables = set(MainSpaceOps.context_allowed_table_set(scope_ctx, schema_graph, mappings=mappings))
        if visible_objects is not None:
            table_keys = {v for v in visible_objects if "." not in v}
            col_parents = {v.split(".", 1)[0] for v in visible_objects if "." in v}
            tables &= table_keys | col_parents
        return frozenset(tables)

    @staticmethod
    def effective_visible_columns(
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        *,
        mappings: FederationMappings | None = None,
        exclude_restricted: bool = True,
    ) -> frozenset[str]:
        """Return ``table.column`` names visible under context ∩ credentials ∩ sensitivity."""
        visible_tables = MainSpaceOps.effective_visible_tables(
            schema_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
        )
        out: set[str] = set()
        for tname in visible_tables:
            tm = schema_graph.tables.get(tname)
            if tm is None:
                continue
            for cname, col in tm.columns.items():
                if MainSpaceOps.column_allowed_for_visibility(
                    tname,
                    cname,
                    scope_ctx=scope_ctx,
                    visible_objects=visible_objects,
                    exclude_restricted=exclude_restricted,
                    col=col,
                ):
                    out.add(f"{tname}.{cname}")
        return frozenset(out)

    @staticmethod
    def aetherspace_within_effective_visibility(
        space_tables: frozenset[str],
        space_columns: frozenset[str],
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        *,
        mappings: FederationMappings | None = None,
        federation_manifest: FederationManifest | None = None,
    ) -> bool:
        """Return True when space allow-lists are ⊆ effective visibility (empty allow = unrestricted)."""
        visible_tables = MainSpaceOps.effective_visible_tables(
            schema_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
        )
        if not space_tables and not space_columns:
            return visible_tables >= frozenset(schema_graph.tables.keys())
        if space_tables - visible_tables:
            return False
        if not space_columns:
            return True
        visible_cols = MainSpaceOps.effective_visible_columns(
            schema_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
        )
        resolve_manifest = MainSpaceOps._space_column_resolve_manifest(federation_manifest)
        for qc in space_columns:
            resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=schema_graph)
            tbl, col = resolved.table, resolved.column
            qualified = f"{tbl}.{col}"
            if tbl not in visible_tables:
                return False
            tm = schema_graph.tables.get(tbl)
            col_meta = tm.columns.get(col) if tm is not None else None
            if not MainSpaceOps.column_eligible_for_space_allowlist(col_meta):
                continue
            if qualified not in visible_cols and qc not in visible_cols:
                return False
        return True

    @staticmethod
    def validate_aetherspace_define_within_visibility(
        space_tables: frozenset[str],
        space_columns: frozenset[str],
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        *,
        mappings: FederationMappings | None = None,
        federation_manifest: FederationManifest | None = None,
    ) -> None:
        """Raise :class:`ConfigError` when a define/overwrite names objects outside visible scope."""
        if not space_tables and not space_columns:
            return
        visible_tables = MainSpaceOps.effective_visible_tables(
            schema_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
        )
        extra_tables = space_tables - visible_tables
        if extra_tables:
            raise ConfigError(f"aetherspace cannot be defined outside visible scope: tables {sorted(extra_tables)!r}")
        if not space_columns:
            return
        visible_cols = MainSpaceOps.effective_visible_columns(
            schema_graph,
            scope_ctx,
            visible_objects,
            mappings=mappings,
        )
        resolve_manifest = MainSpaceOps._space_column_resolve_manifest(federation_manifest)
        blocked: list[str] = []
        extras: list[str] = []
        for qc in sorted(space_columns):
            resolved = resolve_federation_qualified_ref(qc, manifest=resolve_manifest, schema=schema_graph)
            tbl, col = resolved.table, resolved.column
            qualified = f"{tbl}.{col}"
            tm = schema_graph.tables.get(tbl)
            col_meta = tm.columns.get(col) if tm is not None else None
            if not MainSpaceOps.column_eligible_for_space_allowlist(col_meta):
                blocked.append(qc)
                continue
            if tbl not in visible_tables or (qualified not in visible_cols and qc not in visible_cols):
                extras.append(qc)
        if blocked:
            raise ConfigError(
                f"SpaceContext columns entry cannot be included in an aetherspace "
                f"(hidden, restricted, or denied): {blocked!r}"
            )
        if extras:
            raise ConfigError(f"aetherspace cannot be defined outside visible scope: columns {extras!r}")

    @staticmethod
    def _column_allowed_in_context(
        table_name: str, col_name: str, ctx: EngineContext | FederationContext, schema_graph: SchemaGraph
    ) -> bool:
        if (table_name, col_name) in ctx.qualified_denies():
            return False
        if col_name in ctx.glob_column_denies():
            return False
        if not ctx.allow_columns:
            return True
        if (table_name, col_name) in ctx.qualified_allows():
            return True
        return col_name in ctx.glob_column_allows()

    @staticmethod
    def resolve_engine_context_plan(
        engine_context: EngineContext | str | None,
        engine_dir: str,
        *,
        schema_role: SchemaRole,
        load_master: EngineContext | None,
        prepare_master: EngineContext | None,
    ) -> tuple[EngineContext, EngineContext, str]:
        """Resolve construction input into master, active, and registration name. *load_master* is the on-disk master cache (may be ``None``). *prepare_master* is an explicit master object after ``_prepare_schema_context_for_init`` (may be ``None``)."""
        if isinstance(engine_context, str):
            name = MainSpaceOps._normalize_context_name(engine_context)
            if schema_role == "consumer" and name != MASTER_AETHERSPACE_NAME:
                pass
            master = load_master
            if master is None:
                raise ConfigError("create master engine context first; no cached schema_context.json was found")
            if name == MASTER_AETHERSPACE_NAME:
                return master, master, MASTER_AETHERSPACE_NAME
            named = MainSpaceOps.load_named_schema_context(engine_dir, name)
            if named is None:
                raise ConfigError(f"unknown engine context {engine_context!r}")
            return master, named, name

        if engine_context is None:
            master = load_master
            if master is None:
                raise ConfigError(
                    "schema_context is required on first initialisation. No cached "
                    f"schema_context.json was found in {engine_dir!r}. Pass an explicit "
                    "EngineContext (use EngineContext() to scope to the whole database)."
                )
            return master, master, MASTER_AETHERSPACE_NAME

        if schema_role == "consumer":
            raise OwnerOnlyOperationError("EngineContext(engine context definition)")
        master = prepare_master if prepare_master is not None else engine_context
        return master, master, MASTER_AETHERSPACE_NAME

    @staticmethod
    def bind_template_store_for_space(owner: Any, space_name: str) -> None:
        """Load and cache the template-store partition for one aetherspace namespace."""
        norm_space = TemplateOps.validate_space_name(space_name)
        schema_graph = getattr(owner, "_schema_graph", None)
        if schema_graph is None:
            return
        graph_id = str(getattr(schema_graph, "schema_graph_id", "") or "")
        artifacts_dir = getattr(owner, "_artifacts_dir", None)
        if artifacts_dir is None:
            return
        store = TemplateOps.load_template_store(
            graph_id,
            schema_graph,
            space_name=norm_space,
            artifacts_dir=str(artifacts_dir),
        )
        MainSpaceOps.sync_owner_template_cache(owner, store, space_name=norm_space)

    @staticmethod
    def bind_owner_default_template_store(
        owner: Any,
        schema_graph: SchemaGraph,
        artifacts_dir: str,
        *,
        schema_role: SchemaRole,
    ) -> None:
        """Load the template-store partition for the owner's default aetherspace."""
        space_name = MASTER_AETHERSPACE_NAME
        if schema_role == SchemaRole.CONSUMER:
            default_uid = getattr(owner, "_credential_default_space_uid", None)
            if default_uid:
                space_name = str(default_uid)
        graph_id = str(schema_graph.schema_graph_id or "")
        store = TemplateOps.load_template_store(
            graph_id, schema_graph, space_name=space_name, artifacts_dir=artifacts_dir
        )
        templates = TemplateOps.store_to_templates(store)
        owner._store = store
        owner._templates = templates
        MainSpaceOps.sync_owner_template_cache(owner, store, space_name=space_name)

    @staticmethod
    def sync_owner_template_cache(owner: Any, store: Any, *, space_name: str = MASTER_AETHERSPACE_NAME) -> None:
        """Keep the facade template cache aligned with the in-memory store view per aetherspace."""
        norm_space = TemplateOps.validate_space_name(space_name)
        stores_by_space = getattr(owner, "_store_by_space", None)
        if not isinstance(stores_by_space, dict):
            stores_by_space = {}
            owner._store_by_space = stores_by_space
        templates_by_space = getattr(owner, "_templates_by_space", None)
        if not isinstance(templates_by_space, dict):
            templates_by_space = {}
            owner._templates_by_space = templates_by_space
        stores_by_space[norm_space] = store
        templates_by_space[norm_space] = TemplateOps.store_to_templates(store)
        if norm_space == MASTER_AETHERSPACE_NAME:
            owner._store = store
            owner._templates = templates_by_space[norm_space]

    @staticmethod
    def owner_template_store_for_space(owner: Any, space_name: str) -> Any | None:
        """Return a cached template store for *space_name*, if one is loaded on *owner*."""
        norm_space = TemplateOps.validate_space_name(space_name)
        stores_by_space = getattr(owner, "_store_by_space", None)
        if isinstance(stores_by_space, dict):
            cached = stores_by_space.get(norm_space)
            if cached is not None:
                return cached
        if norm_space == MASTER_AETHERSPACE_NAME:
            return getattr(owner, "_store", None)
        return None

    @staticmethod
    def persist_template_store(owner: Any | None, store: Any, *, space_name: str = MASTER_AETHERSPACE_NAME) -> None:
        """Flush *store* to disk and refresh *owner*'s cached template map when present."""
        TemplateOps.save_template_store(store)
        if owner is not None:
            MainSpaceOps.sync_owner_template_cache(owner, store, space_name=space_name)

    @staticmethod
    def reload_reader_learning_if_manifest_drift(owner: Any) -> None:
        """Reload partitioned template store and replay overrides when disk manifest drifts from the live graph."""
        manifest = read_artifact_manifest(str(owner._artifacts_dir))
        if manifest is None:
            return
        live_graph = getattr(owner, "_schema_graph", None)
        if not isinstance(live_graph, SchemaGraph):
            return
        if manifest_matches_schema(manifest, live_graph):
            return
        store = TemplateOps.load_template_store(live_graph.schema_graph_id, live_graph)
        templates = TemplateOps.store_to_templates(store)
        owner._store = store
        owner._templates = templates
        finalize_with_structure(
            owner._schema_graph,
            MainSpaceOps.engine_schema_json_path(str(owner._artifacts_dir)),
            dialect=getattr(owner, "_dialect", None),
        )

    @staticmethod
    def _emit_write_queue_audit(owner: Any, event_type: str, details: tuple[tuple[str, str], ...]) -> None:
        """Forward write-queue drain outcomes to ``owner._audit_emit`` when an audit sink is configured."""
        fn = getattr(owner, "_audit_emit", None)
        if not callable(fn):
            return
        sg = getattr(owner, "_schema_graph", None)
        sh = str(getattr(sg, "effective_structural_hash", "") or "") or None
        fn(event_type, schema_hash=sh, details=details)

    @staticmethod
    def _owner_write_queue_drain_target(owner: Any) -> _WriteQueueDrainTarget:
        return _WriteQueueDrainTarget(
            schema_graph=owner._schema_graph,
            store=owner._store,
            templates=owner._templates,
            rejected=owner._rejected,
            dialect=getattr(owner, "_dialect", None),
        )

    @staticmethod
    def _federation_member_write_queue_targets(owner: Any) -> list[tuple[str, _WriteQueueDrainTarget]]:
        """Return per-member artifact dirs and drain targets for a federation owner."""
        if not getattr(owner, "_is_aether_federation", False):
            return []
        runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
        member_graphs = getattr(owner, "_federation_member_graphs", None) or {}
        if not isinstance(runtimes, dict) or not isinstance(member_graphs, dict):
            return []
        targets: list[tuple[str, _WriteQueueDrainTarget]] = []
        for source_id in sorted(runtimes):
            runtime = runtimes.get(source_id)
            if runtime is None or not getattr(runtime, "artifacts_dir", None):
                raise FederationConfigError(
                    f"federation member store missing for source_id {source_id!r}; "
                    "each member must have its own artifact tree"
                )
            artifacts_dir = getattr(runtime, "artifacts_dir", None)
            graph = member_graphs.get(source_id)
            if graph is None:
                raise FederationConfigError(
                    f"federation member store missing for source_id {source_id!r}; "
                    "each member must have its own artifact tree"
                )
            graph_id = str(graph.schema_graph_id or "")
            store = TemplateOps.load_template_store(graph_id, graph, artifacts_dir=str(artifacts_dir))
            targets.append(
                (
                    str(artifacts_dir),
                    _WriteQueueDrainTarget(
                        schema_graph=graph,
                        store=store,
                        templates=TemplateOps.store_to_templates(store),
                        rejected={},
                        dialect=getattr(runtime, "dialect", None),
                    ),
                )
            )
        return targets

    @staticmethod
    def _drain_dispatch_write_queue_event(
        owner: Any, event: WriteQueueEvent, *, target: _WriteQueueDrainTarget | None = None
    ) -> bool:
        """Apply one queue event to *target* stores. Returns True when the template store should be saved."""
        tgt = target or MainSpaceOps._owner_write_queue_drain_target(owner)
        live = str(getattr(tgt.schema_graph, "schema_graph_id", "") or "")
        if not live or event.schema_graph_id != live:
            return False
        store = tgt.store
        templates: dict[str, Template] | LazyTemplateMapping = tgt.templates
        rejected: dict[str, Any] = tgt.rejected
        schema = tgt.schema_graph
        dialect = tgt.dialect
        pl = dict(event.payload)

        if event.kind == "feedback_record":
            q_norm = str(pl.get("q_norm") or "")
            raw_entry = pl.get("entry_json") or "{}"
            try:
                entry_doc = json.loads(raw_entry)
            except json.JSONDecodeError:
                notify("write_queue: malformed entry_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                return False
            if not isinstance(entry_doc, dict):
                return False
            entry = QuestionFeedbackEntry.from_dict(entry_doc)
            TemplateOps.record_question_feedback(store, q_norm, entry)
            MainSpaceOps._emit_write_queue_audit(
                owner, AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD, (("kind", "feedback_record"), ("q_norm", q_norm))
            )
            return True

        if event.kind == "template_reject":
            raw_ctx = pl.get("ctx_json") or "{}"
            try:
                ctx_doc = json.loads(raw_ctx)
            except json.JSONDecodeError:
                notify("write_queue: malformed ctx_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                return False
            if not isinstance(ctx_doc, dict):
                return False
            try:
                intent = RuntimeIntent.from_dict(ctx_doc.get("intent") or {})
            except (KeyError, TypeError, ValueError):
                notify("write_queue: malformed intent in ctx_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                return False
            mid = str(ctx_doc.get("matched_template_id") or "")
            mrej_id = str(ctx_doc.get("matched_rejected_template_id") or "")
            mt = templates.get(mid) if mid else None
            mrej = rejected.get(mrej_id) if mrej_id else None
            try:
                gpath = GenerationPath.parse(str(ctx_doc.get("generation_path") or ""))
            except (KeyError, ValueError, TypeError):
                gpath = GenerationPath.FRESH
            ctx = UserFeedbackRejectSuspendContext(
                intent=intent,
                sql=str(ctx_doc.get("sql") or ""),
                schema=schema,
                store=cast(dict[str, Any], store),
                templates=cast(dict[str, Any], templates),
                rejected=rejected,
                q_norm=str(ctx_doc.get("q_norm") or ""),
                generation_path=gpath,
                matched_template=mt,
                matched_rejected_template=mrej,
                dialect=dialect,
                structural_match_templates=None,
            )
            complete_user_feedback_reject(
                ctx,
                needs_reason=bool(ctx_doc.get("needs_reason")),
                reject_reason=str(ctx_doc.get("reject_reason") or ""),
                choice_port=None,
                persist_template_learning=True,
            )
            MainSpaceOps._emit_write_queue_audit(
                owner,
                AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT,
                (("kind", "template_reject"), ("q_norm", str(ctx_doc.get("q_norm") or ""))),
            )
            return False

        if event.kind == "template_accept":
            raw = pl.get("replay_json") or "{}"
            try:
                rep = json.loads(raw)
            except json.JSONDecodeError:
                notify("write_queue: malformed replay_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                return False
            if not isinstance(rep, dict):
                return False
            try:
                intent = RuntimeIntent.from_dict(rep.get("intent") or {})
            except (KeyError, TypeError, ValueError):
                notify(
                    "write_queue: malformed intent in replay_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO
                )
                return False
            sql = str(rep.get("sql") or "")
            q_norm = str(rep.get("q_norm") or "")
            try:
                gpath = GenerationPath.parse(str(rep.get("generation_path") or ""))
            except (KeyError, ValueError, TypeError):
                gpath = GenerationPath.FRESH
            mid = str(rep.get("matched_template_id") or "")
            mrej_id = str(rep.get("matched_rejected_id") or "")
            mt = templates.get(mid) if mid else None
            mrej = rejected.get(mrej_id) if mrej_id else None
            join_matches = bool(rep.get("join_matches", True))
            sm_ids = [x for x in str(rep.get("structural_ids") or "").split(",") if x]
            sm_list = [templates[x] for x in sm_ids if x in templates]
            fs_raw = rep.get("form_storage")
            fs: QuestionFormStorage | None = None
            if isinstance(fs_raw, dict):
                fs = QuestionFormStorage(
                    corrected=str(fs_raw.get("corrected") or ""),
                    normalized_optional=fs_raw.get("normalized_optional"),
                    normalized_negative_memory_dropped=bool(fs_raw.get("normalized_negative_memory_dropped")),
                    accept_via_normalized_lookup_only=bool(fs_raw.get("accept_via_normalized_lookup_only")),
                )
            handle_user_feedback(
                "y",
                intent,
                sql,
                schema,
                store,
                cast(dict[str, Any], templates),
                rejected,
                q_norm,
                gpath,
                mt,
                mrej,
                dialect=dialect,
                structural_match_templates=sm_list or None,
                choice_port=None,
                join_matches_template=join_matches,
                form_storage=fs,
                persist_template_learning=True,
            )
            MainSpaceOps._emit_write_queue_audit(
                owner, AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT, (("kind", "template_accept"), ("q_norm", q_norm))
            )
            return False

        if event.kind == "override_proposal":
            raw_doc = pl.get("document_json") or "{}"
            try:
                document = json.loads(raw_doc)
            except json.JSONDecodeError:
                notify("write_queue: malformed document_json", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                return False
            if not isinstance(document, dict):
                return False
            document.setdefault("version", STRUCTURE_DOCUMENT_VERSION)
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as tf:
                    json.dump(document, tf, ensure_ascii=False)
                    tmp_path = tf.name
                apply_structure_from_path(
                    schema,
                    tmp_path,
                    schema_json_path=MainSpaceOps.engine_schema_json_path(str(owner._artifacts_dir)),
                )
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                notify(
                    f"write_queue: override_proposal apply failed: {exc}",
                    stage="pipeline",
                    code=DIAGNOSTIC_CODE_ENGINE_INFO,
                )
                return False
            finally:
                if tmp_path and os.path.isfile(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            MainSpaceOps._emit_write_queue_audit(
                owner,
                AUDIT_EVENT_WRITE_QUEUE_STRUCTURE_PROPOSAL,
                (("kind", "override_proposal"),),
            )
            return False

        if event.kind == "paraphrase_emit":
            notify(
                "write_queue: paraphrase_emit is reserved; line skipped",
                stage="pipeline",
                code=DIAGNOSTIC_CODE_ENGINE_INFO,
            )
            return False

    @staticmethod
    def _write_queue_store_checkpoint(store: dict[str, Any] | TemplateStoreView) -> Any:
        """Capture mutable template-store state before write-queue drain mutations."""
        if isinstance(store, TemplateStoreView):
            return (
                copy.deepcopy(store.feedback_shard_index),
                {pid: copy.deepcopy(part) for pid, part in store._feedback_partition_cache.items()},
                set(store._dirty_feedback_partitions),
                copy.deepcopy(store.partition_map),
                {pid: copy.deepcopy(part) for pid, part in store._partition_cache.items()},
                set(store._dirty_partitions),
                copy.deepcopy(store._indexes),
                int(store.next_id),
            )
        return copy.deepcopy(store)

    @staticmethod
    def _write_queue_store_restore(store: dict[str, Any] | TemplateStoreView, checkpoint: Any) -> None:
        """Restore *store* from a checkpoint produced by :meth:`_write_queue_store_checkpoint`."""
        if isinstance(store, TemplateStoreView):
            (
                feedback_shard_index,
                feedback_cache,
                dirty_feedback,
                partition_map,
                partition_cache,
                dirty_partitions,
                indexes,
                next_id,
            ) = checkpoint
            store.feedback_shard_index = feedback_shard_index
            store._feedback_partition_cache = OrderedDict(feedback_cache)
            store._dirty_feedback_partitions = dirty_feedback
            store.partition_map = partition_map
            store._partition_cache = OrderedDict(partition_cache)
            store._dirty_partitions = dirty_partitions
            store._indexes = indexes
            store.next_id = next_id
            return
        store.clear()
        if isinstance(checkpoint, dict):
            store.update(checkpoint)

    @staticmethod
    def _persist_write_queue_stores(
        owner: Any,
        stores: set[dict[str, Any] | TemplateStoreView],
        *,
        store_spaces: Mapping[int, str] | None = None,
    ) -> None:
        """Flush dirty template stores after a successful write-queue drain batch."""
        for store in stores:
            if isinstance(store, TemplateStoreView):
                space_name = (store_spaces or {}).get(id(store), MASTER_AETHERSPACE_NAME)
                MainSpaceOps.persist_template_store(owner, store, space_name=space_name)
            else:
                TemplateOps.save_template_store(store)
                if store is getattr(owner, "_store", None):
                    MainSpaceOps.sync_owner_template_cache(owner, store)

    @staticmethod
    def _archive_corrupt_write_queue(artifacts_dir: str, path: str) -> str:
        """Move an unparseable write queue aside and return the archive path."""
        ts = datetime.now(UTC).strftime(STRUCTURE_APPLIED_TIMESTAMP_FORMAT)
        corrupt_name = f"write_queue.corrupt.{ts}.jsonl"
        corrupt_path = os.path.join(artifacts_dir, corrupt_name)
        os.replace(path, corrupt_path)
        return corrupt_path

    @staticmethod
    def _drain_write_queue_at_path(
        owner: Any, artifacts_dir: str, *, target: _WriteQueueDrainTarget | None = None
    ) -> int:
        """Drain one artifact tree's write queue under the artifact lock."""
        path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
        applied = 0
        with artifact_lock(artifacts_dir, timeout=WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS):
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                return 0
            with open(path, "rb") as fh:
                body = fh.read()
            if not body:
                return 0
            limit = WRITE_QUEUE_MAX_BYTES_PER_DRAIN
            if len(body) > limit:
                head = body[:limit]
                cut = head.rfind(b"\n")
                if cut == -1:
                    corrupt_path = MainSpaceOps._archive_corrupt_write_queue(artifacts_dir, path)
                    notify(
                        f"write queue archived as corrupt: {corrupt_path}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT,
                        details=(("path", corrupt_path),),
                    )
                    return 0
                to_process = head[: cut + 1]
                tail = head[cut + 1 :] + body[limit:]
            else:
                to_process = body
                tail = b""
            text = to_process.decode("utf-8", errors="replace")
            tgt = target or MainSpaceOps._owner_write_queue_drain_target(owner)
            raw_lines = text.splitlines(keepends=True)
            stores_to_save: set[dict[str, Any] | TemplateStoreView] = set()
            store_checkpoints: dict[int, Any] = {}
            store_spaces: dict[int, str] = {}
            pending_suffix: list[str] = []
            for idx, raw_line in enumerate(raw_lines):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    doc = json.loads(stripped)
                except json.JSONDecodeError:
                    notify("write_queue: malformed line skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                    continue
                if not isinstance(doc, dict):
                    continue
                evt = decode_write_queue_event(doc)
                if evt is None:
                    notify("write_queue: unknown event skipped", stage="pipeline", code=DIAGNOSTIC_CODE_ENGINE_INFO)
                    continue
                explicit_space = write_queue_event_space_name(doc)
                if explicit_space is None:
                    notify(
                        "write_queue: event missing space_name; skipped",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    continue
                event_space = explicit_space
                if target is not None and evt.schema_graph_id == str(
                    getattr(target.schema_graph, "schema_graph_id", "") or ""
                ):
                    event_store = target.store
                else:
                    cached_event_store = MainSpaceOps.owner_template_store_for_space(owner, event_space)
                    if cached_event_store is None:
                        graph = tgt.schema_graph
                        graph_id = str(getattr(graph, "schema_graph_id", "") or "")
                        event_store = TemplateOps.load_template_store(
                            graph_id, graph, artifacts_dir=artifacts_dir, space_name=event_space
                        )
                        MainSpaceOps.sync_owner_template_cache(owner, event_store, space_name=event_space)
                    else:
                        event_store = cached_event_store
                event_templates = TemplateOps.store_to_templates(event_store)
                event_target = _WriteQueueDrainTarget(
                    schema_graph=tgt.schema_graph,
                    store=event_store,
                    templates=event_templates,
                    rejected=tgt.rejected,
                    dialect=tgt.dialect,
                )
                store = event_target.store
                store_key = id(store)
                store_spaces[store_key] = event_space
                if store_key not in store_checkpoints:
                    store_checkpoints[store_key] = MainSpaceOps._write_queue_store_checkpoint(store)
                try:
                    if MainSpaceOps._drain_dispatch_write_queue_event(owner, evt, target=event_target):
                        stores_to_save.add(store)
                    applied += 1
                except Exception as exc:
                    notify(
                        f"write_queue: event dispatch failed: {exc!r}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    pending_suffix = list(raw_lines[idx:])
                    break
            if stores_to_save:
                try:
                    MainSpaceOps._persist_write_queue_stores(owner, stores_to_save, store_spaces=store_spaces)
                except Exception as exc:
                    notify(
                        f"write_queue: store persist failed: {exc!r}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_ENGINE_INFO,
                    )
                    for store_key, checkpoint in store_checkpoints.items():
                        for store in stores_to_save:
                            if id(store) == store_key:
                                MainSpaceOps._write_queue_store_restore(store, checkpoint)
                                break
                    return 0
            with open(path, "wb") as out:
                if pending_suffix:
                    out.write("".join(pending_suffix).encode("utf-8"))
                out.write(tail)
        return applied

    @staticmethod
    def drain_write_queue(owner: Any, artifacts_dir: str) -> int:
        """Drain deferred reader events under the artifact lock; returns the number of events applied."""
        applied = MainSpaceOps._drain_write_queue_at_path(owner, artifacts_dir)
        seen_dirs = {os.path.abspath(os.fspath(artifacts_dir))}
        for member_dir, member_target in MainSpaceOps._federation_member_write_queue_targets(owner):
            member_abs = os.path.abspath(os.fspath(member_dir))
            if member_abs in seen_dirs:
                continue
            seen_dirs.add(member_abs)
            applied += MainSpaceOps._drain_write_queue_at_path(owner, member_dir, target=member_target)
        return applied

    @staticmethod
    def column_allowed_for_visibility(
        table_name: str,
        col_name: str,
        *,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
        exclude_restricted: bool,
        col: Any,
    ) -> bool:
        """Return True when a column may appear in meta/export payloads under effective visibility."""
        if col.is_denied:
            return False
        if col.sensitivity == SensitivityClassification.HIDDEN:
            return False
        if exclude_restricted and col.sensitivity == SensitivityClassification.RESTRICTED:
            return False
        if visible_objects is not None:
            qualified = f"{table_name}.{col_name}"
            has_table = table_name in visible_objects
            has_qualified = qualified in visible_objects
            has_any_qualified_for_table = any(v.startswith(f"{table_name}.") for v in visible_objects if "." in v)
            if has_any_qualified_for_table:
                if not has_qualified and not has_table:
                    return False
            elif not has_table and not has_qualified:
                return False
        if scope_ctx is not None:
            if (table_name, col_name) in scope_ctx.qualified_denies():
                return False
            if col_name in scope_ctx.glob_column_denies():
                return False
            if scope_ctx.allow_columns:
                if (
                    table_name,
                    col_name,
                ) not in scope_ctx.qualified_allows() and col_name not in scope_ctx.glob_column_allows():
                    return False
        return True

    @staticmethod
    def federation_feedback_kwargs(
        owner: Any | None,
        gen_out: SqlGenerationOutcome,
        choice_port: InteractiveChoicePort | None = None,
        *,
        federated_prepare: FederatedPrepareOutcome | None = None,
    ) -> dict[str, Any]:
        """Build optional federation accept kwargs for :func:`handle_user_feedback`."""
        if gen_out.generation_path is not GenerationPath.FEDERATION_PLAN:
            return {}
        member_graphs = getattr(owner, "_federation_member_graphs", None) if owner is not None else None
        stores_by_source: dict[str, TemplateStoreView] = {}
        schemas_by_source: dict[str, SchemaGraph] = {}
        if owner is not None and isinstance(member_graphs, dict):
            stores_by_source = MainSpaceOps.federation_stores_by_source(
                owner, member_graphs, space_name=MainSpaceOps.session_space_name_for_federation(owner, choice_port)
            )
            schemas_by_source = dict(member_graphs)
        federated_plan = federated_prepare.plan if federated_prepare is not None else None
        pending_plan_template = None
        if choice_port is not None:
            pending_plan_template = getattr(choice_port, "_pending_federation_plan_template", None)
        return {
            "federated_steps": tuple(gen_out.federated_steps),
            "federation_dir": gen_out.federation_dir,
            "federation_plan_id": gen_out.federation_plan_id,
            "stores_by_source": stores_by_source,
            "schemas_by_source": schemas_by_source,
            "federated_plan": federated_plan,
            "pending_plan_template": pending_plan_template,
        }

    @staticmethod
    def federation_gate_kwargs_by_source(
        owner: Any,
        choice_port: InteractiveChoicePort | None,
        manifest: FederationManifest,
        dialects_by_source: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compose per-source execution gate kwargs from manifest bindings."""
        engine_types = {binding.source_id: binding.engine for binding in manifest.sources}
        allowed_where_ops = intersect_member_where_ops(dialects_by_source, engine_types_by_source=engine_types)
        gates: dict[str, dict[str, Any]] = {}
        runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
        artifacts_root = MainSpaceOps._federation_artifacts_root(owner)
        for binding in manifest.sources:
            base = dict(MainSpaceOps.consumer_sql_gate_kwargs(choice_port))
            base["schema_role"] = binding.role
            member_ops = DialectRegistry.extra_where_ops_for_engine(binding.engine)
            base["allowed_where_ops"] = allowed_where_ops & (member_ops | set(FEDERATION_BASE_WHERE_OPS))
            if binding.context and binding.context != "master":
                runtime = runtimes.get(binding.source_id)
                member_dir = (
                    str(runtime.artifacts_dir)
                    if runtime is not None and runtime.artifacts_dir
                    else federation_source_artifacts_dir(artifacts_root, binding)
                )
                named = MainSpaceOps.load_named_schema_context(member_dir, binding.context)
                if named is None:
                    raise ConfigError(
                        f"federation source {binding.source_id!r} declared context {binding.context!r} "
                        f"not found in member artifacts at {member_dir}"
                    )
                base["schema_context"] = named
                base["context_name"] = binding.context
            else:
                base["schema_context"] = EngineContext()
                base["context_name"] = MASTER_AETHERSPACE_NAME
            gates[binding.source_id] = base
        return gates

    @staticmethod
    def federation_result_contract_kwargs(
        gen_out: SqlGenerationOutcome,
        *,
        federated_prepare: FederatedPrepareOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
    ) -> dict[str, Any]:
        """Column and shape kwargs derived from a federated plan rather than display SQL."""
        if gen_out.generation_path is not GenerationPath.FEDERATION_PLAN:
            return {}
        prep = federated_prepare
        plan = prep.plan if prep is not None else None
        kwargs: dict[str, Any] = {
            "generation_path": gen_out.generation_path,
        }
        if plan is not None:
            kwargs["federated_plan"] = plan
        if federated_bundle is not None:
            kwargs["federated_bundle"] = federated_bundle
        column_names: Sequence[str] | None = None
        if federated_bundle is not None and federated_bundle.column_names:
            column_names = federated_bundle.column_names
        elif prep is not None and prep.bundle is not None and prep.bundle.column_names:
            column_names = prep.bundle.column_names
        elif plan is not None:
            residual = federation_residual_column_headers(plan)
            if residual:
                column_names = residual
        if column_names:
            kwargs["column_names"] = column_names
        return kwargs

    @staticmethod
    def federation_single_source_sql_context(
        owner: Any,
        intent: Any,
        schema: SchemaGraph,
        fed_manifest: FederationManifest,
        fed_mappings: FederationMappings | None,
        default_dialect: Any,
    ) -> tuple[Any, SchemaGraph] | None:
        """Return per-source dialect and schema when *intent* references exactly one federation source."""
        source_ids = source_ids_for_intent(intent, schema, fed_mappings, fed_manifest)
        if len(source_ids) != 1:
            return None
        source_id = next(iter(source_ids))
        member_graphs = getattr(owner, "_federation_member_graphs", None) if owner is not None else None
        member_schema = resolve_federated_member_schema(
            source_id,
            schema,
            manifest=fed_manifest,
            member_graphs=member_graphs if isinstance(member_graphs, dict) else None,
        )
        dialects_by_source = getattr(owner, "_federation_dialects", None) if owner is not None else None
        source_dialect = (
            dialects_by_source.get(source_id)
            if isinstance(dialects_by_source, dict) and source_id in dialects_by_source
            else default_dialect
        )
        return source_dialect, member_schema

    @staticmethod
    def raise_if_session_turn_cancelled() -> None:
        if session_turn_cancelled():
            raise SessionTurnCancelledError("Turn cancelled.")

    @staticmethod
    def resolved_session_step_sql(
        sql: str | dict[str, str] | None,
        *,
        gen_out: SqlGenerationOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
    ) -> str | dict[str, str] | None:
        """Resolve ``SessionStep.sql`` for single-engine and federated turns."""
        fed_sql = MainSpaceOps._federation_session_step_sql(
            gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        )
        if fed_sql is not None:
            return fed_sql
        if MainSpaceOps._federation_turn_active(
            gen_out=gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        ):
            return None
        return sql

    @staticmethod
    def session_space_name_for_federation(owner: Any, choice_port: InteractiveChoicePort | None) -> str:
        """Return the active AetherSpace name for federated per-source template learning."""
        if choice_port is not None:
            sn = getattr(choice_port, "space_name", None)
            if callable(sn):
                value = sn()
                if value:
                    return str(value)
            elif isinstance(sn, str) and sn.strip():
                return sn.strip()
        return str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)

    @staticmethod
    def table_allowed_for_visibility(
        table_name: str,
        schema_graph: SchemaGraph,
        scope_ctx: EngineContext | FederationContext | None,
        visible_objects: frozenset[str] | None,
    ) -> bool:
        """Return True when a table may appear under effective visibility."""
        if table_name not in schema_graph.tables:
            return False
        allowed = set(schema_graph.tables.keys())
        if scope_ctx is not None:
            if scope_ctx.allow_objects:
                allowed &= set(scope_ctx.allow_objects)
            if scope_ctx.deny_objects:
                allowed -= set(scope_ctx.deny_objects)
        if visible_objects is not None:
            table_keys = {v for v in visible_objects if "." not in v}
            col_parents = {v.split(".", 1)[0] for v in visible_objects if "." in v}
            allowed &= table_keys | col_parents
        return table_name in allowed

    @staticmethod
    def federation_stores_by_source(
        owner: Any, member_graphs: Mapping[str, SchemaGraph], *, space_name: str = MASTER_AETHERSPACE_NAME
    ) -> dict[str, TemplateStoreView]:
        """Load per-source template stores from federation member artifact trees."""
        runtimes = getattr(owner, "_federation_source_runtimes", None) or {}
        stores: dict[str, TemplateStoreView] = {}
        for source_id, graph in member_graphs.items():
            runtime = runtimes.get(source_id)
            if runtime is None or not getattr(runtime, "artifacts_dir", None):
                raise FederationConfigError(
                    f"federation member store missing for source_id {source_id!r}; "
                    "each member must have its own artifact tree"
                )
            artifacts_dir = str(runtime.artifacts_dir)
            graph_id = str(graph.schema_graph_id or "")
            stores[source_id] = TemplateOps.load_template_store(
                graph_id, graph, artifacts_dir=artifacts_dir, space_name=space_name
            )
        return stores

    @staticmethod
    def resolve_qsim_path(version_or_result: int | QSimSummary, artifacts_dir: str) -> str:
        """Resolve the full file path for a QSim questions text artifact."""
        if isinstance(version_or_result, QSimSummary):
            ver = version_or_result.version
        else:
            ver = int(version_or_result)
        return os.path.join(artifacts_dir, QSIM_QUESTIONS_PATTERN.format(version=ver))

    @staticmethod
    def consumer_sql_gate_kwargs(choice_port: InteractiveChoicePort | None) -> dict[str, Any]:
        """Collect execution-scope parameters from the active programmatic session owner."""
        owner = getattr(choice_port, "_owner", None)
        schema_role = str(getattr(owner, "_schema_role", "owner") or "owner")
        execution_visible_objects = getattr(choice_port, "execution_visible_objects", None)
        runtime_cfg = getattr(owner, "_runtime_config", None)
        master_context = getattr(runtime_cfg, "engine_context", None) if runtime_cfg is not None else None
        execution_context = getattr(runtime_cfg, "execution_context", None) if runtime_cfg is not None else None
        scope_ctx = execution_context if execution_context is not None else master_context
        space_tables = getattr(choice_port, "space_tables", None)
        space_columns = getattr(choice_port, "space_columns", None)
        space_deny_tables = getattr(choice_port, "space_deny_objects", None)
        space_deny_columns = getattr(choice_port, "space_deny_columns", None)
        context_name = str(getattr(owner, "_context_name", MASTER_AETHERSPACE_NAME) or MASTER_AETHERSPACE_NAME)
        return {
            "schema_role": schema_role,
            "visible_objects": execution_visible_objects,
            "schema_context": scope_ctx,
            "context_name": context_name,
            "space_allowed_tables": space_tables,
            "space_allowed_columns": space_columns,
            "space_deny_tables": space_deny_tables,
            "space_deny_columns": space_deny_columns,
        }

    @staticmethod
    def _federation_artifacts_root(owner: Any) -> str | None:
        """Resolve the artifacts parent directory that holds member ``conn_*`` trees."""
        root = getattr(owner, "_artifacts_root", None)
        if root is not None:
            return str(root)
        fed_dir = getattr(owner, "_federation_storage_dir", None)
        if fed_dir:
            return str(Path(fed_dir).parent)
        adir = getattr(owner, "_artifacts_dir", None)
        if adir is not None:
            return str(Path(adir).parent)
        return None

    @staticmethod
    def _federation_session_step_sql(
        gen_out: SqlGenerationOutcome | None = None,
        *,
        federated_bundle: FederatedSqlBundle | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
    ) -> str | dict[str, str] | None:
        """Return member SQL for a federated turn: ``str`` for one member, ``dict`` for many."""
        if not MainSpaceOps._federation_turn_active(
            gen_out=gen_out,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
        ):
            return None
        if federated_bundle is None:
            return None
        member_statements = [
            rec
            for rec in federated_bundle.statements
            if str(getattr(rec, "phase", "member") or "member") == "member"
            and str(getattr(rec, "source_id", "") or "").strip()
            and str(getattr(rec, "statement", "") or "").strip()
        ]
        if not member_statements:
            return None
        if len(member_statements) == 1:
            return str(member_statements[0].statement).strip() or None
        mapping: dict[str, str] = {}
        for rec in member_statements:
            sid = str(rec.source_id).strip()
            statement = str(rec.statement).strip()
            if sid and statement and sid not in mapping:
                mapping[sid] = statement
        return mapping or None

    @staticmethod
    def _federation_turn_active(
        *,
        gen_out: SqlGenerationOutcome | None = None,
        federated_bundle: FederatedSqlBundle | None = None,
        federated_plan: FederatedPlan | None = None,
        generation_path: GenerationPath | None = None,
    ) -> bool:
        return (
            generation_path is GenerationPath.FEDERATION_PLAN
            or (gen_out is not None and gen_out.generation_path is GenerationPath.FEDERATION_PLAN)
            or federated_bundle is not None
            or federated_plan is not None
        )
