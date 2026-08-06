"""Seed warmup: gold intents, expansion, joins, SQL validation, NL questions, cache, and artifact I/O."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ._config import PolicyConfig, SeedWarmupConfig
from ._constants import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    EMPTY_JOIN_CANDIDATES,
    JOIN_CHOICE_SCOPE_MAIN,
    JSON_COMPACT_SEPARATORS,
    REALISM_DROP_REASON_CATEGORIES,
    SEED_FAILURE_CODE_REALISM_DROPPED,
    SEED_NORMALIZATION_BATCH_SIZE,
    SEED_QUESTION_CLARIFY_SYSTEM,
    SEED_WARMUP_DROP_CODES,
    SEED_WARMUP_FAILURE_CODES,
    WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY,
)
from ._contracts_base import (
    HavingParam,
    LlmBatchRequest,
    LlmJsonExhausted,
    PredicateGroup,
    WhereParam,
)
from ._contracts_core import (
    AnchorLattice,
    AnchorLatticeCell,
    AnchorLatticeKey,
    GenerationPath,
    QuestionFormStorage,
    RuntimeCteStep,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    Template,
    ValueHistory,
)
from ._contracts_schema import SchemaGraph, TemplateStats, ValueDomain
from ._core_utils import (
    StepResult,
    append_failure_trace,
    artifact_lock,
    ask_user_choice,
    bind_params_for_sql,
    debug,
    normalize_question,
    notify,
    pipeline_capture,
    reconcile_execute_bind_params,
    seed_warmup_failure_code_from_validate_sql_error,
    sha256,
    stable_json,
    telemetry_capture,
    write_json_atomic,
)
from ._dialect import Dialect
from ._federation import schema_spans_multiple_sources
from ._intent_expr import apply_default_structural_values
from ._intent_process import (
    apply_deterministic_repairs,
    apply_runtime_post_processing_lite,
    collect_structural_match_templates,
    full_intent_parse,
    structural_compare,
)
from ._intent_repair import apply_diagnostic_repairs
from ._intent_resolve import (
    check_qualified_refs_exist,
    join_path_key_runtime,
    prune_unused_cte_steps,
)
from ._llm_provider import LLMProvider
from ._pipeline import (
    execute_federated_warmup_intent,
    finalize_substitute_sql,
    other_template_owns_question_string,
    persist_federated_warmup_learning,
)
from ._qsim import (
    deterministic_having_value,
    greedy_cover_indices_by_atoms,
    sample_coordinated_range,
    sample_value_from_domain,
)
from ._sql_gen import (
    build_deterministic_sql,
    canonicalize_stored_join_path_signature,
    edge_kinds_for_join_candidate,
    get_join_choice_from_llm,
    inject_join_into_deterministic_sql,
    join_candidate_map,
    join_hints_multi,
    physical_tables_for_join_hints,
)
from ._templates import TemplateOps, TemplateRefs
from ._utils import (
    body_similarity_key,
    exact_question_match,
    flatten_warmup_paraphrases_by_style,
    generate_warmup_paraphrases_by_style,
    generate_warmup_questions_freeform,
    intent_key,
    template_instance_key_for_concrete,
    template_instance_key_for_runtime,
)
from ._validation_execute import curated_warmup_post_binding_issues, curated_warmup_semantic_issues, validate_sql
from ._validation_schema import assert_execution_parameters_validated

JoinCacheEntry = tuple[str, list[str], dict[str, Any]]
JoinCacheKey = tuple[frozenset[str], str]


class _WarmupCapDefaultSentinel:
    """Resolve warmup budget cap from SeedWarmupConfig at call time."""


_WARMUP_CAP_DEFAULT = _WarmupCapDefaultSentinel()


@dataclass
class SeedWarmupCacheSession:
    """Mutable in-memory view of the seed warmup cache for one orchestration run."""

    manifest: dict[str, Any]
    work_units: dict[str, dict[str, Any]]
    fp_to_wid: dict[str, str] = field(default_factory=dict)
    execute_hits: int = 0
    touched_work_unit_ids: list[str] = field(default_factory=list)

    @staticmethod
    def _warmup_join_cache_key(
        tables: list[str],
        question_hint: str,
        *,
        hint_is_natural_language: bool,
    ) -> JoinCacheKey:
        """Cache key for warmup join resolution: table set plus hint identity."""
        if hint_is_natural_language:
            hint_id = normalize_question(question_hint) or question_hint.strip()
        else:
            hint_id = f"intent:{question_hint.strip()}"
        return (frozenset(tables), hint_id)

    def ensure_work_unit_id(self, fingerprint: str) -> str:
        """Return stable content-addressed id for *fingerprint*."""
        wid = self.fp_to_wid.get(fingerprint)
        if wid is None:
            wid = fingerprint
            self.fp_to_wid[fingerprint] = wid
        return wid

    def get_cached_execute(self, fingerprint: str) -> dict[str, Any] | None:
        """Return packed execute payload when cache matches *fingerprint*."""
        wid = self.fp_to_wid.get(fingerprint)
        if not wid:
            return None
        wu = self.work_units.get(wid)
        if not wu or wu.get("intent_fingerprint") != fingerprint:
            return None
        er = wu.get("execute_result")
        if not isinstance(er, dict):
            return None
        return er

    def write_work_unit(
        self, fingerprint: str, intent: SeedWarmupIntent, packed_execute: dict[str, Any], *, report_version: int
    ) -> None:
        """Persist one work unit record and append *report_version* to session id lists."""
        wid = self.ensure_work_unit_id(fingerprint)
        bk = packed_execute.get("body_key") or ""
        jk = packed_execute.get("join_path_key") or ""
        tik = packed_execute.get("template_instance_key") or ""
        prev = self.work_units.get(wid, {})
        sids = list(prev.get("run_session_ids", []))
        if report_version not in sids:
            sids.append(report_version)
        fc_exec = packed_execute.get("failure_code") if not packed_execute.get("ok") else None
        rec: dict[str, Any] = {
            "work_unit_id": wid,
            "body_key": bk,
            "join_path_key": jk,
            "template_instance_key": tik,
            "intent_fingerprint": fingerprint,
            "serialized_intent": intent.to_dict(),
            "lifecycle_state": ("execute_recorded" if packed_execute.get("ok") else "failed"),
            "execute_result": packed_execute,
            "failure_code": fc_exec,
            "drop_reason_code": fc_exec,
            "question_llm": prev.get("question_llm"),
            "run_session_ids": sids,
        }
        self.work_units[wid] = rec
        if wid not in self.touched_work_unit_ids:
            self.touched_work_unit_ids.append(wid)

    def mark_sampled_in(self, fingerprint: str) -> str | None:
        """Set *lifecycle_state* to ``sampled_in`` after a successful cached or fresh execute."""
        wid = self.fp_to_wid.get(fingerprint)
        if not wid:
            return None
        prev = self.work_units.get(wid)
        if not prev:
            return None
        er = prev.get("execute_result")
        if not isinstance(er, dict) or not er.get("ok"):
            return None
        if prev.get("intent_fingerprint") != fingerprint:
            return None
        rec = dict(prev)
        if rec.get("lifecycle_state") == "execute_recorded":
            rec["lifecycle_state"] = "sampled_in"
        self.work_units[wid] = rec
        if wid not in self.touched_work_unit_ids:
            self.touched_work_unit_ids.append(wid)
        return wid

    def record_question_llm(self, fingerprint: str, payload: dict[str, Any], *, ok: bool) -> None:
        """Attach question/realism LLM payload and advance lifecycle after sampling."""
        wid = self.fp_to_wid.get(fingerprint)
        if not wid:
            return
        prev = self.work_units.get(wid)
        if not prev:
            return
        rec = dict(prev)
        rec["question_llm"] = payload
        rec["lifecycle_state"] = "llm_done" if ok else "failed"
        if not ok and payload.get("failure_code"):
            rec["failure_code"] = payload["failure_code"]
            rec["drop_reason_code"] = payload["failure_code"]
        self.work_units[wid] = rec

    @staticmethod
    def warmup_intent_fingerprint(intent: SeedWarmupIntent) -> str:
        """Stable SHA-256 hex of the serialized seed warmup intent (pre- execute)."""
        return hashlib.sha256(stable_json(intent.to_dict()).encode("utf-8")).hexdigest()

    @staticmethod
    def warmup_pool_operator_feature_stats(intents: list[SeedWarmupIntent]) -> dict[str, Any]:
        """Summarize operator-vector diversity for seed-warmup funnel. reporting."""
        vectors = [i.operator_feature_vector() for i in intents]
        distinct_vectors = len(set(vectors))
        bits = {(v.has_aggregate, v.has_grouping, v.has_having, v.window_kind != "none") for v in vectors}
        max_4 = WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY
        distinct_bits = len(bits)
        union_atoms: set[str] = set()
        for intent in intents:
            union_atoms |= intent.coverage_atoms()
        return {
            "warmup_queue_distinct_operator_vectors": distinct_vectors,
            "warmup_queue_operator_feature_4bit_tuple_distinct": distinct_bits,
            "warmup_queue_operator_feature_4bit_tuple_max": max_4,
            "warmup_queue_operator_feature_4bit_tuple_coverage_ratio": (
                round(distinct_bits / max_4, 4) if max_4 else 0.0
            ),
            "warmup_queue_coverage_atom_union_size": len(union_atoms),
        }

    @staticmethod
    def _warmup_anchor_lattice_json_path(output_root: str, schema: SchemaGraph) -> str:
        """Absolute path to persisted anchor-lattice JSON for *schema* under. *output_root*."""
        base = os.path.join(output_root, SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR)
        fn = f"lattice_{schema.schema_graph_id}_v{SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION}.json"
        return os.path.join(base, fn)

    @staticmethod
    def _load_warmup_anchor_lattice(path: str) -> dict[str, list[str]]:
        """Load lattice cell keys to anchor phrase lists from disk."""
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        cells = raw.get("cells")
        if not isinstance(cells, dict):
            return {}
        out: dict[str, list[str]] = {}
        for k, v in cells.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            anch = v.get("anchors")
            if isinstance(anch, list):
                out[k] = [str(x) for x in anch if isinstance(x, str)]
        return out

    @staticmethod
    def _warmup_artifacts_lock_dir(path: str) -> str:
        """Return the artifacts directory that owns warmup siblings at *path*."""
        lock_dir = os.path.abspath(os.path.dirname(os.path.abspath(path)))
        if os.path.basename(lock_dir) == SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR:
            lock_dir = os.path.dirname(lock_dir)
        return lock_dir

    @staticmethod
    def _save_warmup_anchor_lattice(path: str, schema: SchemaGraph, cells: dict[str, list[str]]) -> None:
        """Persist anchor lattice cells atomically next to other warmup. artifacts."""
        lock_dir = SeedWarmupCacheSession._warmup_artifacts_lock_dir(path)
        with artifact_lock(lock_dir):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "schema_fp": schema.schema_graph_id,
                "code_version": SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION,
                "cells": {k: {"anchors": v} for k, v in sorted(cells.items())},
            }
            write_json_atomic(path, payload, sort_keys=False)

    @staticmethod
    def load_seed_warmup_cache_zip(output_dir: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Read ``seed_warmup_cache.zip`` from *output_dir*."""
        path = os.path.join(output_dir, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
        if not os.path.isfile(path):
            return {}, {}
        manifest: dict[str, Any] = {}
        work_units: dict[str, dict[str, Any]] = {}
        with zipfile.ZipFile(path, "r") as zf:
            if SeedWarmupConfig.WARMUP_CACHE_MANIFEST in zf.namelist():
                manifest = json.loads(zf.read(SeedWarmupConfig.WARMUP_CACHE_MANIFEST).decode("utf-8"))
            for name in zf.namelist():
                if not name.startswith(SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX) or not name.endswith(".json"):
                    continue
                rec = json.loads(zf.read(name).decode("utf-8"))
                wid = str(rec.get("work_unit_id") or os.path.basename(name)[:-5])
                work_units[wid] = rec
        return manifest, work_units

    @staticmethod
    def save_seed_warmup_cache_zip(
        output_dir: str,
        manifest: dict[str, Any],
        work_units: dict[str, dict[str, Any]],
        *,
        gold_intent_dicts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Atomically write ``seed_warmup_cache.zip`` under *output_dir*."""
        path = os.path.join(output_dir, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
        lock_dir = os.path.abspath(output_dir)
        with artifact_lock(lock_dir):
            os.makedirs(output_dir, exist_ok=True)
            out_manifest = {
                **manifest,
                "last_updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if gold_intent_dicts is not None:
                out_manifest["gold_intent_count"] = len(gold_intent_dicts)
            directory = os.path.dirname(os.path.abspath(path)) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".zip", dir=directory)
            os.close(fd)
            try:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(
                        SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
                        json.dumps(out_manifest, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS),
                    )
                    if gold_intent_dicts is not None:
                        zf.writestr(
                            SeedWarmupConfig.WARMUP_CACHE_GOLD_INTENTS_JSON,
                            json.dumps(gold_intent_dicts, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS),
                        )
                    for wid, rec in work_units.items():
                        zf.writestr(
                            f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}{wid}.json",
                            json.dumps(rec, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS),
                        )
                os.replace(tmp_path, path)
                tmp_path = ""
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        debug(
            f"save_seed_warmup_cache_zip: wrote {len(work_units)} work units"
            f"{' and gold snapshot' if gold_intent_dicts is not None else ''} to {path}"
        )

    @staticmethod
    def _prune_stale_warmup_work_units(
        work_units: dict[str, dict[str, Any]],
        schema: SchemaGraph,
    ) -> dict[str, dict[str, Any]]:
        """Drop cached work units whose schema refs are no longer live in *schema*."""
        pruned: dict[str, dict[str, Any]] = {}
        for wid, wu in work_units.items():
            refs = TemplateRefs.warmup_work_unit_schema_refs(wu)
            if not refs.tables:
                continue
            ok, _ = TemplateRefs.template_is_live(refs, schema)
            if ok:
                pruned[wid] = wu
        return pruned

    @staticmethod
    def open_seed_warmup_cache_session(
        output_dir: str,
        schema: SchemaGraph,
        seed_content_sha256: str | None = None,
        *,
        sql_history_content_sha256: str | None = None,
    ) -> SeedWarmupCacheSession:
        """Load disk cache when seed content matches; prune work units whose schema refs are stale. Warmup units survive profiling-only drift when structural and effective hashes still match the manifest; structural drift triggers surgical drops keyed by ``warmup_work_unit_schema_refs``."""
        manifest, work_units = SeedWarmupCacheSession.load_seed_warmup_cache_zip(output_dir)
        identity_ok = False
        if seed_content_sha256 is not None and manifest.get("seed_content_hash") == seed_content_sha256:
            identity_ok = True
        if (
            sql_history_content_sha256 is not None
            and manifest.get("sql_history_content_hash") == sql_history_content_sha256
        ):
            identity_ok = True
        seed_ok = identity_ok
        prev_id = str(
            manifest.get("schema_graph_id")
            or manifest.get("effective_structural_hash")
            or manifest.get("schema_hash")
            or ""
        )
        eff_ok = prev_id == schema.schema_graph_id
        prev_prof = str(manifest.get("profiling_hash") or "")

        if not seed_ok:
            work_units = {}
        elif not eff_ok:
            work_units = SeedWarmupCacheSession._prune_stale_warmup_work_units(work_units, schema)
        elif prev_prof and prev_prof != schema.profiling_hash:
            work_units = SeedWarmupCacheSession._prune_stale_warmup_work_units(work_units, schema)

        fp_to_wid = {
            str(wu["intent_fingerprint"]): str(wu["work_unit_id"])
            for wu in work_units.values()
            if wu.get("intent_fingerprint") and wu.get("work_unit_id")
        }
        manifest = {
            **manifest,
            "schema_hash": schema.schema_hash,
            "schema_graph_id": schema.schema_graph_id,
            "effective_structural_hash": schema.effective_structural_hash,
            "structural_hash": schema.structural_hash,
            "profiling_hash": schema.profiling_hash,
            "policy_version": SeedWarmupConfig.WARMUP_SAMPLING_POLICY_VERSION,
            "code_version": SeedWarmupConfig.SEED_WARMUP_CODE_VERSION,
        }
        if seed_content_sha256 is not None:
            manifest["seed_content_hash"] = seed_content_sha256
        if sql_history_content_sha256 is not None:
            manifest["sql_history_content_hash"] = sql_history_content_sha256
        return SeedWarmupCacheSession(manifest=manifest, work_units=work_units, fp_to_wid=fp_to_wid)

    @staticmethod
    def _warmup_pack_execute(
        runtime: RuntimeIntent,
        *,
        ok: bool,
        final_sql: str | None,
        failure_code: str | None,
        error: str | None,
        body_key: str,
        join_path_key: str,
        template_instance_key: str,
    ) -> dict[str, Any]:
        return {
            "ok": ok,
            "runtime": runtime.to_dict(),
            "final_sql": final_sql,
            "failure_code": failure_code,
            "error": error,
            "body_key": body_key,
            "join_path_key": join_path_key,
            "template_instance_key": template_instance_key,
        }

    @staticmethod
    def execute_warmup_sql_rows(
        runtime: RuntimeIntent,
        schema: SchemaGraph,
        dialect: Any,
        exec_sql: str,
        all_params: dict[str, Any],
    ) -> list[Any]:
        """Validate bind parameters, then execute one warmup SQL statement."""
        assert_execution_parameters_validated(runtime, schema)
        return list(dialect.execute(exec_sql, reconcile_execute_bind_params(exec_sql, all_params)))

    @staticmethod
    def _warmup_synthetic_store_path_blocks(
        intent: SeedWarmupIntent, runtime: RuntimeIntent, templates: dict[str, Template]
    ) -> str | None:
        """Return drop code when a synthetic row matches the store only via warmup-forbidden path 4.1 or 4.2."""
        if (intent.source or "gold") == "gold":
            return None
        for tmpl in templates.values():
            if tmpl.trust_level < 1:
                continue
            cr = structural_compare(runtime, tmpl)
            if cr.union_eligible and cr.union_sql_path == GenerationPath.UNION_TEMPLATE_WIDEN:
                return "warmup_path41_not_allowed"
            if cr.union_eligible and cr.union_sql_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN:
                return "warmup_path42_not_allowed"
        return None

    @staticmethod
    def get_next_seed_warmup_version(output_dir: str) -> int:
        """Return the next auto-incrementing version for seed warmup run. reports and bundle zips."""
        versions: list[int] = []
        for pattern in (
            os.path.join(output_dir, "seed_warmup_report_v*.json"),
            os.path.join(output_dir, "seed_warmup_v*.zip"),
        ):
            for fpath in glob.glob(pattern):
                base = os.path.basename(fpath)
                try:
                    if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
                        num = base[len("seed_warmup_report_v") : -len(".json")]
                    elif base.startswith("seed_warmup_v") and base.endswith(".zip"):
                        num = base[len("seed_warmup_v") : -len(".zip")]
                    else:
                        continue
                    versions.append(int(num))
                except (IndexError, ValueError):
                    continue
        return max(versions) + 1 if versions else 1

    @staticmethod
    def run_seed_question_normalization(seeds: list[dict[str, Any]]) -> tuple[dict[int, dict[str, str]], str, str]:
        """Batch LLM normalization for seed lines; return phrases plus. compact JSON and `.txt` payloads."""
        by_num: dict[int, str] = {int(s["number"]): str(s["question"]).strip() for s in seeds}
        sorted_nums = sorted(by_num.keys())
        out: dict[int, dict[str, str]] = {}
        batch_chunks: list[tuple[str, list[int], str]] = []
        for i in range(0, len(sorted_nums), SEED_NORMALIZATION_BATCH_SIZE):
            batch = sorted_nums[i : i + SEED_NORMALIZATION_BATCH_SIZE]
            payload = [{"index": n, "source": by_num[n]} for n in batch]
            user = stable_json({"batch": payload})
            batch_chunks.append((f"seed-normalize-{i}", batch, user))

        parsed_by_id: dict[str, dict[str, Any]] = {}
        if LLMProvider.batch_enabled() and batch_chunks:
            requests = [
                LlmBatchRequest(custom_id=custom_id, system=SEED_QUESTION_CLARIFY_SYSTEM, user=user, task="default")
                for custom_id, _batch, user in batch_chunks
            ]
            try:
                parsed_by_id = LLMProvider.batch_json(requests)
            except (RuntimeError, OSError, ValueError) as exc:
                debug(f"[seed_warmup.run_seed_question_normalization] batch failed: {exc}")
                parsed_by_id = {}

        for custom_id, batch, user in batch_chunks:
            parsed = parsed_by_id.get(custom_id, {})
            if not parsed:
                try:
                    parsed = LLMProvider.json(SEED_QUESTION_CLARIFY_SYSTEM, user, retries=2, task="default")
                except LlmJsonExhausted as exc:
                    debug(f"[seed_warmup.run_seed_question_normalization] llm_json exhausted on {custom_id}: {exc}")
                    parsed = {}
            lines = parsed.get("lines")
            if not isinstance(lines, list):
                lines = []
            got: dict[int, str] = {}
            for row in lines:
                if not isinstance(row, dict):
                    continue
                idx = row.get("index")
                nm = str(row.get("clarified") or row.get("normalized") or "").strip()
                if idx is not None and nm:
                    got[int(idx)] = nm
            for n in batch:
                out[n] = {"original": by_num[n], "normalized": got.get(n, by_num[n])}
        serial = [{"index": k, **v} for k, v in sorted(out.items(), key=lambda x: x[0])]
        json_body = json.dumps(serial, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)
        lines_txt = "".join(f"{k}. {v['normalized']}\n" for k, v in sorted(out.items(), key=lambda x: x[0]))
        debug(f"run_seed_question_normalization: normalized {len(out)} seed lines")
        return out, json_body, lines_txt

    @staticmethod
    def _load_seed_questions(filepath: str) -> list[dict[str, Any]]:
        """Load seed questions from a text file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Seed questions file not found: {filepath}")

        questions: list[dict[str, Any]] = []
        current_phase = "unknown"
        auto_number = 1

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    phase_match = re.search(r"Phase\s+(\d+)", line, re.IGNORECASE)
                    if phase_match:
                        current_phase = f"phase_{phase_match.group(1)}"
                    continue

                question_text = None
                number = None

                match = re.match(r"^(\d+)\.\s*(.+)$", line)
                if match:
                    number = int(match.group(1))
                    question_text = match.group(2).strip()

                if not question_text:
                    match = re.match(r"^(\d+)\)\s*(.+)$", line)
                    if match:
                        number = int(match.group(1))
                        question_text = match.group(2).strip()

                if not question_text:
                    match = re.match(r'^["\'](.+)["\']$', line)
                    if match:
                        question_text = match.group(1).strip()
                        number = auto_number
                        auto_number += 1

                if not question_text:
                    question_text = line
                    number = auto_number
                    auto_number += 1

                question_text = question_text.strip("\"'")
                if question_text:
                    questions.append(
                        {
                            "number": number,
                            "question": question_text,
                            "phase": current_phase,
                        }
                    )

        debug(f"[seed_warmup.load_seed_questions] loaded: {len(questions)} questions")
        mx = SeedWarmupConfig.MAX_SEED_QUESTIONS
        if len(questions) > mx:
            debug(f"[seed_warmup.load_seed_questions] truncating to MAX_SEED_QUESTIONS={mx}")
            questions = questions[:mx]
        return questions

    @staticmethod
    def _parse_gold_intent_strict(question: str, schema: SchemaGraph) -> tuple[RuntimeIntent | None, list[str]]:
        """Parse a seed question with optional retry when the first parse. returns no intent."""
        q_norm = normalize_question(question)
        last_warns: list[str] = []
        attempts = max(1, int(PolicyConfig.GOLD_INTENT_PARSE_ATTEMPTS))
        for attempt in range(attempts):
            intent, warns, _calls, _plan = full_intent_parse(q_norm, schema)
            last_warns = list(warns or [])
            if intent is not None:
                return intent, last_warns
            debug(f"[seed_warmup.parse_gold_intent_strict] attempt {attempt + 1}/{attempts} returned no intent")
        debug(f"[seed_warmup.parse_gold_intent_strict] failed after {attempts} attempts: {q_norm}")
        return None, last_warns

    @staticmethod
    def _replay_gold_intent_parse_for_telemetry(question: str, schema: SchemaGraph) -> None:
        """Run two `full_intent_parse` attempts for diagnostic replay when. gold parse failed."""
        q_norm = normalize_question(question)
        full_intent_parse(q_norm, schema)
        full_intent_parse(q_norm, schema)

    @staticmethod
    def _gold_failure_trace_text(seed_warmup_version: int, sections: list[str]) -> str:
        """Build concatenated gold-parse failure sections with header. metadata."""
        header = (
            f"seed_warmup_version={seed_warmup_version}\n"
            f"failed_seed_count={len(sections)}\n"
            "interactive_gold=false\n"
            "Telemetry blocks are from a diagnostic replay (two full_intent_parse calls per seed).\n\n"
        )
        return header + "\n\n".join(sections)

    @staticmethod
    def _confirm_gold_intent(question: str, intent: RuntimeIntent) -> tuple[bool, RuntimeIntent | None]:
        """Interactively confirm a parsed gold intent with the user."""
        nl_summary = intent.natural_language or ""
        if not nl_summary:
            nl_summary = f"Query {', '.join(intent.tables or [])} for {intent.grain or 'data'}"
        grain = intent.grain or "row_level"
        if grain == "scalar":
            expected_display = "scalar"
        else:
            expected_display = f"{intent.expected_rows or 'many'} row(s)"

        agg_ops = []
        for sc in intent.select_cols or []:
            if sc.is_aggregated:
                e = sc.expr
                agg = e.agg_func or (e.add_groups[0].agg_func if e.add_groups and e.add_groups[0].agg_func else None)
                term = e.primary_term
                if agg:
                    agg_ops.append(f"{agg.upper()}({term})")
                else:
                    agg_ops.append(term)

        notify(
            f"Question: {question}\n\nI understood: {nl_summary}\n"
            f"Tables: {intent.tables or []}\n"
            f"Aggregations: {agg_ops or ['none']}\n"
            f"Expected: {expected_display}",
            stage="cli",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        choice = ask_user_choice("Is this correct?", ["y", "n"])
        if choice is None:
            return False, None
        elif choice == "y":
            notify("\nIntent accepted.", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
            return True, intent
        else:
            notify("\nIntent rejected.", stage="cli", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
            return False, None

    @staticmethod
    def _abstract_values(intent: RuntimeIntent) -> RuntimeIntent:
        """Return `intent` with `param_values` cleared."""
        return RuntimeIntent(
            tables=intent.tables,
            grain=intent.grain,
            select_cols=intent.select_cols,
            group_by_cols=intent.group_by_cols,
            order_by_cols=intent.order_by_cols,
            where=intent.where,
            having=intent.having,
            param_values={},
            cte_steps=intent.cte_steps,
            column_map=intent.column_map,
            natural_language=intent.natural_language,
            limit=intent.limit,
        )

    @staticmethod
    def run_gold_intent_generation(
        schema: SchemaGraph,
        seed_filepath: str,
        interactive: bool = True,
        seed_phrases: dict[int, dict[str, str]] | None = None,
        seed_warmup_version: int = 1,
    ) -> tuple[list[dict[str, Any]], dict[str, int], str | None, tuple[str, str] | None]:
        """Run the full gold intent generation pipeline from seed questions."""
        debug("gold_intent_generation: starting")
        seeds = SeedWarmupCacheSession._load_seed_questions(seed_filepath)
        debug(f"gold_intent_generation: {len(seeds)} seed questions loaded")

        norm_bundle: tuple[str, str] | None = None
        phrases = seed_phrases
        if phrases is None:
            phrases, norm_json, norm_txt = SeedWarmupCacheSession.run_seed_question_normalization(seeds)
            norm_bundle = (norm_json, norm_txt)

        existing_questions: set[str] = set()
        gold_intents: list[dict[str, Any]] = []
        new_count = 0
        skip_count = 0
        fail_count = 0
        user_rejected_count = 0
        failure_trace_sections: list[str] = []

        for seed in seeds:
            num = int(seed["number"])
            pack = phrases.get(num) or {
                "original": seed["question"].strip(),
                "normalized": seed["question"].strip(),
            }
            q_norm = normalize_question(pack["normalized"])
            if q_norm in existing_questions:
                skip_count += 1
                continue

            debug(f"gold_intent_generation: processing [{num}] {pack['normalized']}")
            intent, warns = SeedWarmupCacheSession._parse_gold_intent_strict(pack["normalized"], schema)
            if intent is None:
                debug(f"gold_intent_generation: FAILED to parse [{num}]")
                fail_count += 1
                if not interactive:
                    phase = seed.get("phase") or ""
                    with telemetry_capture(suppress_console=True, force_diagnostic_flags=True) as cap_buf:
                        SeedWarmupCacheSession._replay_gold_intent_parse_for_telemetry(pack["normalized"], schema)
                    warn_lines = "\n".join(f"  - {w}" for w in warns) if warns else "  (none)"
                    block = (
                        f"===== seed_index={num} phase={phase} =====\n"
                        f"original: {pack['original']!r}\n"
                        f"normalized: {pack['normalized']!r}\n"
                        f"warnings ({len(warns)}):\n{warn_lines}\n\n"
                        f"--- telemetry ({len(cap_buf)} lines) ---\n" + "\n".join(cap_buf)
                    )
                    failure_trace_sections.append(block)
                continue
            if warns:
                debug(f"gold_intent_generation: parsed [{num}] with {len(warns)} semantic warning(s)")

            if interactive:
                confirmed, final_intent = SeedWarmupCacheSession._confirm_gold_intent(pack["original"], intent)
                if not confirmed or final_intent is None:
                    debug(f"gold_intent_generation: user rejected [{num}]")
                    user_rejected_count += 1
                    continue
                intent = final_intent

            intent = SeedWarmupCacheSession._abstract_values(intent)
            gold_dict = intent.to_dict()
            gold_dict["normalized_question"] = q_norm
            gold_dict["intent_id"] = intent_key(intent)
            gold_dict["seed_index"] = num
            gold_dict["seed_prompt_original"] = pack["original"]
            gold_dict["seed_prompt_normalized"] = pack["normalized"]
            gold_dict["natural_language"] = intent.natural_language or ""
            gold_intents.append(gold_dict)
            existing_questions.add(q_norm)
            new_count += 1

        trace_body: str | None = None
        if not interactive and failure_trace_sections:
            trace_body = SeedWarmupCacheSession._gold_failure_trace_text(seed_warmup_version, failure_trace_sections)
            debug(f"gold_intent_generation: failure trace built for {len(failure_trace_sections)} seeds")
        debug(
            "gold_intent_generation: complete. "
            f"new={new_count}, skipped={skip_count}, failed={fail_count}, "
            f"user_rejected={user_rejected_count}, total_saved={len(gold_intents)}"
        )
        stats = {
            "seed_questions_loaded": len(seeds),
            "gold_new": new_count,
            "gold_skipped": skip_count,
            "gold_failed": fail_count,
            "gold_user_rejected": user_rejected_count,
            "gold_intents_total": len(gold_intents),
        }
        return gold_intents, stats, trace_body, norm_bundle

    @staticmethod
    def _collect_resolved_table_names(
        tables: list[str], cte_steps: list[RuntimeCteStep] | list[dict[str, object]] | list[object]
    ) -> set[str]:
        """Return top-level and CTE-referenced table names in lowercase."""
        names = {str(t).lower() for t in tables}
        for cte in cte_steps:
            cte_tables: list[object]
            if isinstance(cte, dict):
                raw_cte_tables = cte.get("tables")
                cte_tables = list(raw_cte_tables) if isinstance(raw_cte_tables, list) else []
            else:
                cte_tables = list(getattr(cte, "tables", []) or [])
            for tbl in cte_tables:
                names.add(str(tbl).lower())
        return names

    @staticmethod
    def _where_hint_search_blob(
        filters: list[WhereParam] | list[dict[str, object]] | list[object], param_values: Mapping[str, object]
    ) -> str:
        """Build lowercase searchable text for ``must_where`` hint matching."""
        parts: list[str] = []
        for filt in filters:
            if isinstance(filt, dict):
                left_expr = filt.get("left_expr", "")
                op = filt.get("op", "")
                value = filt.get("value", "")
                param_key = str(filt.get("param_key") or "")
            else:
                left_expr = getattr(filt, "left_expr", "")
                op = getattr(filt, "op", "")
                value = getattr(filt, "value", "")
                param_key = str(getattr(filt, "param_key", "") or "")
            parts.extend([str(left_expr), str(op), str(value), param_key])
            if param_key and param_key in param_values:
                parts.append(str(param_values[param_key]))
        return " ".join(parts).lower()

    @staticmethod
    def _object_list(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _object_dict(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return {str(k): cast(object, v) for k, v in value.items()}

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        return value if isinstance(value, int) else None

    @staticmethod
    def check_intent_against_expectation(
        intent: dict[str, object] | RuntimeIntent, expectation: dict[str, object]
    ) -> list[str]:
        """Return human-readable failure reasons when *intent* violates *expectation*."""
        tables: list[str]
        grain: str
        select_cols: list[object]
        group_by: list[object]
        filters: list[object]
        limit: int | None
        cte_steps: list[object]
        window_registry: list[object]
        param_values: dict[str, object]
        if isinstance(intent, RuntimeIntent):
            tables = list(intent.tables or [])
            grain = str(intent.grain or "")
            select_cols = [cast(object, sc) for sc in (intent.select_cols or [])]
            group_by = [cast(object, g) for g in (intent.group_by_cols or [])]
            filters = [cast(object, f) for f in (PredicateGroup.where_leaves(intent.where) or [])]
            limit = intent.limit
            cte_steps = [cast(object, c) for c in (intent.cte_steps or [])]
            window_registry = [cast(object, w) for w in (intent.window_registry or [])]
            param_values = SeedWarmupCacheSession._object_dict(intent.param_values or {})
        else:
            tables = [str(t) for t in SeedWarmupCacheSession._object_list(intent.get("tables"))]
            grain = str(intent.get("grain") or "")
            select_cols = SeedWarmupCacheSession._object_list(intent.get("select_cols"))
            group_by = SeedWarmupCacheSession._object_list(intent.get("group_by_cols"))
            filters = SeedWarmupCacheSession._object_list(intent.get("where", intent.get("where_param")))
            limit = SeedWarmupCacheSession._optional_int(intent.get("limit"))
            cte_steps = SeedWarmupCacheSession._object_list(intent.get("cte_steps"))
            window_registry = SeedWarmupCacheSession._object_list(intent.get("window_registry"))
            param_values = SeedWarmupCacheSession._object_dict(intent.get("param_values"))

        failures: list[str] = []
        allowed_grains = SeedWarmupCacheSession._object_list(expectation.get("allowed_grains"))
        if allowed_grains:
            allowed = {str(g) for g in allowed_grains}
            if grain not in allowed:
                failures.append(f"grain expected one of {sorted(allowed)!r}, got {grain!r}")
        else:
            exp_grain = expectation.get("grain")
            if exp_grain and grain != str(exp_grain):
                failures.append(f"grain expected {exp_grain!r}, got {grain!r}")

        table_set = SeedWarmupCacheSession._collect_resolved_table_names(tables, cte_steps)
        for tbl in SeedWarmupCacheSession._object_list(expectation.get("must_tables")):
            if str(tbl).lower() not in table_set:
                failures.append(f"missing required table {tbl!r}")
        for tbl in SeedWarmupCacheSession._object_list(expectation.get("forbid_tables")):
            if str(tbl).lower() in table_set:
                failures.append(f"forbidden table present {tbl!r}")

        agg_ops: set[str] = set()
        for sc in select_cols:
            if isinstance(sc, dict):
                expr = sc.get("expr") or ""
                if sc.get("is_aggregated"):
                    for token in ("count", "sum", "avg", "min", "max"):
                        if token in str(expr).lower():
                            agg_ops.add(token)
            elif getattr(sc, "is_aggregated", False):
                e = getattr(sc, "expr", None)
                fn = getattr(e, "agg_func", None) or ""
                if fn:
                    agg_ops.add(str(fn).lower())

        for agg in SeedWarmupCacheSession._object_list(expectation.get("must_aggregate")):
            if str(agg).lower() not in agg_ops:
                failures.append(f"missing aggregate {agg!r}")

        for hint in SeedWarmupCacheSession._object_list(expectation.get("must_group_by")):
            hint_l = str(hint).lower()
            if not any(hint_l in str(g).lower() for g in group_by):
                failures.append(f"missing group_by hint {hint!r}")

        for hint in SeedWarmupCacheSession._object_list(expectation.get("must_where", expectation.get("must_filter"))):
            blob = SeedWarmupCacheSession._where_hint_search_blob(filters, param_values)
            if str(hint).lower() not in blob:
                failures.append(f"missing where hint {hint!r}")

        exp_limit = expectation.get("limit")
        if exp_limit is not None and limit != exp_limit:
            failures.append(f"limit expected {exp_limit!r}, got {limit!r}")

        if expectation.get("must_have_cte") and not cte_steps:
            failures.append("expected at least one CTE")
        if expectation.get("must_have_window") and not window_registry:
            failures.append("expected at least one window function")

        return failures

    @staticmethod
    def _seed_warmup_intent_sort_key(intent: SeedWarmupIntent) -> tuple[int, str]:
        """Order intents so shallower expansion layers resolve joins before deeper same-table children."""
        depth = intent.expansion_metadata.depth if intent.expansion_metadata else 0
        return (depth, intent.intent_id or "")

    @staticmethod
    def _ambiguous_join_reuse_from_parent(
        intent: SeedWarmupIntent,
        join_cache: dict[JoinCacheKey, JoinCacheEntry],
        id_to_intent: dict[str, SeedWarmupIntent],
    ) -> JoinCacheEntry | None:
        """Reuse a cached join decision when the parent intent shares the same table set."""
        key = frozenset(intent.tables or [])
        em = intent.expansion_metadata
        if not em or not em.parent_intent_id:
            return None
        parent = id_to_intent.get(em.parent_intent_id)
        if parent is None or frozenset(parent.tables or []) != key:
            return None
        nl_hint = (parent.natural_language or parent.seed_prompt_normalized or "").strip()
        if nl_hint:
            parent_key = SeedWarmupCacheSession._warmup_join_cache_key(
                parent.tables or [], nl_hint, hint_is_natural_language=True
            )
        else:
            parent_key = SeedWarmupCacheSession._warmup_join_cache_key(
                parent.tables or [], parent.intent_id or "", hint_is_natural_language=False
            )
        return join_cache.get(parent_key)

    @staticmethod
    def resolve_joins_for_table_set(
        tables: list[str],
        schema: SchemaGraph,
        question_hint: str,
        join_cache: dict[JoinCacheKey, JoinCacheEntry],
        *,
        ambiguous_reuse_entry: JoinCacheEntry | None = None,
        hint_is_natural_language: bool = False,
    ) -> JoinCacheEntry:
        """Resolve join path for a table set, using cache when available."""
        key = SeedWarmupCacheSession._warmup_join_cache_key(
            tables, question_hint, hint_is_natural_language=hint_is_natural_language
        )
        if key in join_cache:
            return join_cache[key]

        if len(tables) <= 1:
            entry: JoinCacheEntry = ("J00", [], EMPTY_JOIN_CANDIDATES)
            join_cache[key] = entry
            return entry

        join_tables = physical_tables_for_join_hints(tables, schema)
        candidates = join_hints_multi(schema, join_tables, virtual_specs={}, include_semantic=False)
        cmap = join_candidate_map(candidates)

        if not candidates:
            entry = ("J00", [], EMPTY_JOIN_CANDIDATES)
            join_cache[key] = entry
            return entry

        non_trivial = [c for c in cmap if c != "J00"]
        if len(non_trivial) <= 1:
            chosen = non_trivial[0] if non_trivial else "J00"
            sig_list = list(cmap.get(chosen, []))
            entry = (chosen, sig_list, candidates)
            join_cache[key] = entry
            return entry

        if ambiguous_reuse_entry is not None:
            chosen = ambiguous_reuse_entry[0]
            if chosen in cmap:
                sig_list = list(cmap.get(chosen, []))
                entry = (chosen, sig_list, candidates)
                join_cache[key] = entry
                return entry

        if hint_is_natural_language:
            choices = get_join_choice_from_llm(
                question_hint,
                "",
                llm_scopes=[
                    {
                        "scope": JOIN_CHOICE_SCOPE_MAIN,
                        "tables": [],
                        "candidates": list(candidates.get("candidates") or []),
                    }
                ],
                preset_choices={},
                accept_na_by_scope={JOIN_CHOICE_SCOPE_MAIN: False},
                require_final=False,
                schema=schema,
            )
            chosen = choices.get(JOIN_CHOICE_SCOPE_MAIN, "J00")
        else:
            chosen = sorted(non_trivial)[0]
        if chosen not in cmap:
            chosen = "J00"
        sig_list = list(cmap.get(chosen, []))
        entry = (chosen, sig_list, candidates)
        join_cache[key] = entry
        return entry

    @staticmethod
    def _decompose_between_where_param(f: WhereParam) -> list[WhereParam]:
        """Decompose a `between` filter into `>=` and `<=` filters."""
        if f.op != "between":
            return [f]
        base_param_key = (f.param_key or "").strip()
        if base_param_key:
            lower_key: str | None = f"{base_param_key}_lower"
            upper_key: str | None = f"{base_param_key}_upper"
        else:
            lower_key = None
            upper_key = None
        return [
            replace(f, op=">=", param_key=lower_key),
            replace(f, op="<=", param_key=upper_key),
        ]

    @staticmethod
    def _identify_range_pairs(filters: list[WhereParam]) -> dict[str, dict[str, int]]:
        """Identify columns with paired lower and upper bound filters."""
        column_ops: dict[str, dict[str, int]] = {}
        for idx, f in enumerate(filters):
            if f.right_expr:
                continue
            if f.op in (">", ">="):
                column_ops.setdefault(f.left_expr.primary_column, {})["lower_idx"] = idx
            elif f.op in ("<", "<="):
                column_ops.setdefault(f.left_expr.primary_column, {})["upper_idx"] = idx
        return {col: ops for col, ops in column_ops.items() if "lower_idx" in ops and "upper_idx" in ops}

    @staticmethod
    def instantiate_intent(intent: SeedWarmupIntent, value_domains: dict[str, ValueDomain]) -> SeedWarmupIntent | None:
        """Populate filter and HAVING values from profiling data."""
        decomposed: list[WhereParam] = []
        for f in PredicateGroup.where_leaves(intent.where) if intent.where else []:
            decomposed.extend(SeedWarmupCacheSession._decompose_between_where_param(f))

        range_pairs = SeedWarmupCacheSession._identify_range_pairs(decomposed)
        range_values: dict[str, tuple[str, str]] = {}
        for col_key, pair_indices in range_pairs.items():
            domain = value_domains.get(col_key)
            if domain is None:
                continue
            lower_idx = pair_indices["lower_idx"]
            vtype = decomposed[lower_idx].value_type
            lower_val, upper_val = sample_coordinated_range(domain, vtype, 0)
            if lower_val is not None and upper_val is not None:
                range_values[col_key] = (lower_val, upper_val)

        new_filters: list[WhereParam] = []
        new_param_values: dict[str, Any] = {}

        for filter_idx, f in enumerate(decomposed):
            col_key = f.left_expr.primary_column
            op = f.op
            param_key = f.param_key or f"filter_{filter_idx}"
            value: Any | None = None

            if f.right_expr:
                new_filters.append(f)
                continue

            if f.value_type == "null" or op in ("is null", "is not null"):
                new_filters.append(replace(f, op=op, value_type="null", param_key=param_key))
                continue

            if f.value_type in ("date_window", "date_diff"):
                new_filters.append(replace(f, param_key=param_key))
                continue

            domain = value_domains.get(col_key)
            if domain is None:
                new_filters.append(replace(f, param_key=param_key))
                continue

            if col_key in range_values:
                lower_val, upper_val = range_values[col_key]
                if f.op in (">", ">="):
                    value = lower_val
                elif f.op in ("<", "<="):
                    value = upper_val
                else:
                    value = sample_value_from_domain(domain, f.value_type, f.op, 0)
            else:
                value = sample_value_from_domain(domain, f.value_type, f.op, 0)

            if value is not None:
                new_param_values[param_key] = value
            new_filters.append(replace(f, param_key=param_key))

        new_having: list[HavingParam] = []
        for having_idx, h in enumerate(PredicateGroup.having_leaves(intent.having)):
            param_key = h.param_key or f"having_{having_idx}"
            if h.right_expr is not None:
                new_having.append(replace(h, param_key=param_key))
                continue
            value = deterministic_having_value(h.left_expr.primary_term, 0, having_idx)
            new_param_values[param_key] = value
            new_having.append(replace(h, param_key=param_key))

        return SeedWarmupIntent(
            intent_id=intent.intent_id,
            tables=intent.tables,
            grain=intent.grain,
            select_cols=intent.select_cols,
            group_by_cols=intent.group_by_cols,
            order_by_cols=intent.order_by_cols,
            where=PredicateGroup.from_list(new_filters),
            having=PredicateGroup.from_list(new_having),
            param_values=new_param_values,
            cte_steps=intent.cte_steps,
            question="",
            natural_language=getattr(intent, "natural_language", "") or "",
            expansion_metadata=intent.expansion_metadata,
            limit=intent.limit,
            distinct_select_index=intent.distinct_select_index,
            seed_prompt_original=getattr(intent, "seed_prompt_original", "") or "",
            seed_prompt_normalized=getattr(intent, "seed_prompt_normalized", "") or "",
            seed_index=getattr(intent, "seed_index", None),
            source=getattr(intent, "source", "gold") or "gold",
            window_registry=list(intent.window_registry or []),
            case_registry=list(intent.case_registry or []),
        )

    @staticmethod
    def accepted_template_instance_keys(templates: dict[str, Template]) -> set[str]:
        """Return ``template_instance_key`` values for all accepted templates in *templates*."""
        keys: set[str] = set()
        for tmpl in templates.values():
            conc = tmpl.intent_signature
            keys.add(template_instance_key_for_concrete(conc, tmpl.sql_fp))
        return keys

    @staticmethod
    def _warmup_stratum_key(warmup_intent: SeedWarmupIntent) -> str:
        """Deterministic stratum id for seed-warmup downsampling."""
        origin = "gold" if (warmup_intent.source or "gold") == "gold" else "synthetic"
        em = warmup_intent.expansion_metadata
        depth = em.depth if em else 0
        op = (em.operator if em else "") or ""
        tables_t = tuple(sorted(warmup_intent.tables or []))
        tables_fp = stable_json(tables_t)
        tier = warmup_intent.complexity_tier().value
        return f"{origin}|d{depth}|{op}|{tables_fp}|t{tier}"

    @staticmethod
    def _allocate_stratum_quotas(counts: dict[str, int], budget: int, floor: int) -> dict[str, int]:
        """Assign integer quotas per stratum, at least *floor* each when possible, total *budget*."""
        keys = sorted(counts.keys())
        if not keys:
            return {}
        q: dict[str, int] = {k: 0 for k in keys}
        rem = budget
        changed = True
        while changed and rem > 0:
            changed = False
            for k in keys:
                if rem <= 0:
                    break
                if q[k] < floor and q[k] < counts[k]:
                    q[k] += 1
                    rem -= 1
                    changed = True
        while rem > 0:
            best = max(keys, key=lambda k: (counts[k] - q[k], k))
            if counts[best] - q[best] <= 0:
                break
            q[best] += 1
            rem -= 1
        return q

    @staticmethod
    def _warmup_jaccard_signature(intent: SeedWarmupIntent) -> frozenset[str]:
        """Feature tokens for MMR deduplication over synthetic warmup survivors."""
        v = intent.operator_feature_vector()
        parts: set[str] = set()
        for t in intent.tables or []:
            parts.add(f"tbl:{t}")
        parts.add(f"body:{body_similarity_key(intent.to_runtime_intent())}")
        parts.add(f"ofv:agg:{int(v.has_aggregate)}")
        parts.add(f"ofv:gb:{int(v.has_grouping)}")
        parts.add(f"ofv:hav:{int(v.has_having)}")
        parts.add(f"ofv:win:{v.window_kind}")
        parts.add(f"ofv:sj:{int(v.has_self_join_via_cte)}")
        parts.add(f"ofv:scte:{int(v.has_scalar_cte)}")
        parts.add(f"ofv:un:{int(v.has_unnest)}")
        parts.add(f"ofv:cw:{int(v.has_case_when)}")
        parts.add(f"ofv:dw:{int(v.has_date_window)}")
        parts.add(f"ofv:dd:{int(v.has_date_diff)}")
        parts.add(f"ofv:cteb:{v.cte_depth_bucket}")
        parts.add(f"ofv:jbb:{v.join_breadth_bucket}")
        parts.add(f"ofv:fam:{v.workload_family.value}")
        return frozenset(parts)

    @staticmethod
    def _warmup_body_footprint(intent: SeedWarmupIntent) -> int:
        """Structural size heuristic for tie-breaking duplicate Jaccard signatures."""
        return (
            len(intent.select_cols or [])
            + len(PredicateGroup.where_leaves(intent.where) or [])
            + len(PredicateGroup.having_leaves(intent.having) or [])
            + len(intent.cte_steps or [])
        )

    @staticmethod
    def _warmup_min_gold_fraction() -> float:
        """Return the active gold floor fraction for the current warmup sampling profile."""
        profile = str(SeedWarmupConfig.WARMUP_SAMPLING_PROFILE or "default").strip().lower()
        if profile in ("benchmark", "benchmark_corpus"):
            return float(SeedWarmupConfig.WARMUP_MIN_GOLD_FRACTION_BENCHMARK)
        return float(SeedWarmupConfig.WARMUP_MIN_GOLD_FRACTION)

    @staticmethod
    def _warmup_normalize_result_cell(value: Any) -> str:
        """Render one query result cell for benchmark-aligned result- signature hashing."""
        if value is None:
            return "__NULL__"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{round(value, 8):.8f}"
        if isinstance(value, int):
            return str(value)
        if hasattr(value, "isoformat"):
            try:
                return str(value.isoformat())
            except (TypeError, ValueError):
                pass
        return str(value)

    @staticmethod
    def _warmup_canonical_result_signature(rows: list[Any] | None) -> str | None:
        """Return SHA-256 over sorted canonical rows for post-execute warmup atoms."""
        if not rows:
            return None
        canon_rows: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                keys = sorted(row.keys())
                cells = [SeedWarmupCacheSession._warmup_normalize_result_cell(row[k]) for k in keys]
            elif isinstance(row, (list, tuple)):
                cells = [SeedWarmupCacheSession._warmup_normalize_result_cell(c) for c in row]
            else:
                cells = [SeedWarmupCacheSession._warmup_normalize_result_cell(row)]
            canon_rows.append("\t".join(cells))
        canon_rows.sort()
        payload = "\n".join(canon_rows)
        return sha256(payload)

    @staticmethod
    def _warmup_scenario_atom(intent: SeedWarmupIntent) -> str:
        """Business-shape scenario atom replacing raw seed-index bucketing."""
        fam = intent.workload_family()
        tier = intent.complexity_tier().value
        tables = ",".join(sorted(intent.tables or []))
        digest = hashlib.sha256(f"{tables}|{fam.value}|{tier}".encode()).hexdigest()
        return f"scenario:{int(digest[:8], 16) % 128}"

    @staticmethod
    def _warmup_semantic_coverage_atoms(intent: SeedWarmupIntent) -> frozenset[str]:
        """Semantic coverage atoms layered on structural warmup tags."""
        v = intent.operator_feature_vector()
        atoms: set[str] = set()
        if v.has_aggregate:
            atoms.add("metric:aggregate")
        grain = "month"
        if v.window_kind and v.window_kind != "none":
            grain = v.window_kind
        atoms.add(f"grain:{grain}")
        entity = (intent.tables or ["unknown"])[0]
        atoms.add(f"entity:{entity}")
        atoms.add("time:absolute" if v.window_kind == "none" else "time:relative")
        if v.has_grouping:
            atoms.add("result_shape:table")
        elif v.has_aggregate:
            atoms.add("result_shape:scalar")
        else:
            atoms.add("result_shape:table")
        return frozenset(atoms)

    @staticmethod
    def _warmup_low_volume_cap_positions(positions: list[int], ordered_intents: list[SeedWarmupIntent]) -> set[int]:
        """Apply body-key caps when the low-volume bypass keeps all survivors."""
        body_counts: dict[str, int] = {}
        tier_body_counts: dict[tuple[str, str], int] = {}
        kept: list[int] = []
        for pos in positions:
            intent = ordered_intents[pos]
            bk = body_similarity_key(intent.to_runtime_intent())
            tier = intent.complexity_tier().value
            if body_counts.get(bk, 0) >= 3:
                continue
            tb = (bk, tier)
            if tier_body_counts.get(tb, 0) >= 2:
                continue
            body_counts[bk] = body_counts.get(bk, 0) + 1
            tier_body_counts[tb] = tier_body_counts.get(tb, 0) + 1
            kept.append(pos)
        deduped, _ = SeedWarmupCacheSession._warmup_dedupe_jaccard_positions(kept, ordered_intents)
        return set(deduped)

    @staticmethod
    def _warmup_llm_uncertainty_score(intent: SeedWarmupIntent) -> float:
        """Deterministic uncertainty score in [0, 1] for LLM diversity routing."""
        rule_count = (
            len(PredicateGroup.where_leaves(intent.where) or [])
            + len(PredicateGroup.having_leaves(intent.having) or [])
            + len(intent.select_cols or [])
        )
        score = 0.0
        if rule_count < 4:
            score += 0.40
        sk = SeedWarmupCacheSession._warmup_stratum_key(intent)
        if sk.endswith(":rare") or ":tail" in sk:
            score += 0.25
        em = intent.expansion_metadata
        depth = int(em.depth) if em and em.depth is not None else 0
        if depth >= 2:
            score += 0.20
        if not intent.tables:
            score += 0.15
        return min(score, 1.0)

    @staticmethod
    def _warmup_submodular_atoms_for_row(
        ordered_intents: list[SeedWarmupIntent], pos: int, *, position_rsig: dict[int, str] | None = None
    ) -> frozenset[str]:
        """Coverage atoms for greedy set-cover over synthetic execute indices."""
        intent = ordered_intents[pos]
        base = set(intent.coverage_atoms())
        base |= set(SeedWarmupCacheSession._warmup_semantic_coverage_atoms(intent))
        base.add(f"body:{body_similarity_key(intent.to_runtime_intent())}")
        base.add(SeedWarmupCacheSession._warmup_scenario_atom(intent))
        rsig = (position_rsig or {}).get(pos)
        if rsig:
            base.add(f"rsig:{rsig}")
        return frozenset(base)

    @staticmethod
    def _warmup_jaccard_similarity_frozen(a: frozenset[str], b: frozenset[str]) -> float:
        """Jaccard similarity on string-token sets."""
        if not a and not b:
            return 1.0
        union_n = len(a | b)
        if union_n == 0:
            return 0.0
        return len(a & b) / union_n

    @staticmethod
    def _warmup_mmr_order(
        positions: list[int], ordered_intents: list[SeedWarmupIntent], lambda_mmr: float
    ) -> list[int]:
        """Maximum marginal relevance ordering for diversified survivor ordering."""
        if len(positions) <= 1:
            return list(positions)
        sigs = {p: SeedWarmupCacheSession._warmup_jaccard_signature(ordered_intents[p]) for p in positions}
        remaining = set(positions)
        first = min(positions)
        selected: list[int] = [first]
        remaining.remove(first)
        lam = float(lambda_mmr)
        while remaining:
            best_p = -1
            best_score = -1e9
            for p in remaining:
                sp = sigs[p]
                max_sim = 0.0
                for q in selected:
                    max_sim = max(max_sim, SeedWarmupCacheSession._warmup_jaccard_similarity_frozen(sp, sigs[q]))
                novelty = 1.0 - max_sim
                score = lam * novelty - (1.0 - lam) * max_sim
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_p < 0:
                best_p = min(remaining)
            selected.append(best_p)
            remaining.remove(best_p)
        return selected

    @staticmethod
    def _warmup_dedupe_jaccard_positions(
        positions: list[int], ordered_intents: list[SeedWarmupIntent]
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """Keep one survivor per Jaccard signature with smallest structural footprint."""
        by_sig: dict[frozenset[str], list[int]] = {}
        for pos in positions:
            sig = SeedWarmupCacheSession._warmup_jaccard_signature(ordered_intents[pos])
            by_sig.setdefault(sig, []).append(pos)
        kept: list[int] = []
        drop_records: list[dict[str, Any]] = []
        for plist in by_sig.values():
            best = min(plist, key=lambda p: (SeedWarmupCacheSession._warmup_body_footprint(ordered_intents[p]), p))
            kept.append(best)
            for p in plist:
                if p == best:
                    continue
                oi = ordered_intents[p]
                sk = SeedWarmupCacheSession._warmup_stratum_key(oi)
                drop_records.append(
                    {
                        "drop_phase": "sampling",
                        "intent_index": p,
                        "intent_id": getattr(oi, "intent_id", None),
                        "failure_code": "redundant_cover_representative",
                        "origin": "synthetic",
                        "stratum_id": sk,
                        "quota": None,
                        "rank_in_stratum": None,
                        "detail": (f"redundant_cover_representative stratum_id={sk!r} kept_index={best}"),
                    }
                )
        return sorted(kept), drop_records

    @staticmethod
    def build_anchor_lattice(
        synthetic_pending: list[tuple[SeedWarmupIntent, str]],
        schema: SchemaGraph,
        lattice_disk_by_sig: dict[str, list[str]],
    ) -> AnchorLattice:
        """Precompute shared NL anchors per lattice cell for synthetic warmup rows."""
        groups: dict[AnchorLatticeKey, list[tuple[SeedWarmupIntent, str]]] = {}
        for intent, sql in synthetic_pending:
            k = intent.anchor_lattice_key()
            groups.setdefault(k, []).append((intent, sql))
        cells: dict[AnchorLatticeKey, AnchorLatticeCell] = {}
        for key, rows in groups.items():
            lk = key.signature(schema.schema_graph_id)
            disk_a = lattice_disk_by_sig.get(lk)
            if disk_a:
                cells[key] = AnchorLatticeCell(
                    key=key, representative_intent_id=rows[0][0].intent_id, anchors=tuple(str(x) for x in disk_a)
                )
                continue
            best_intent, best_sql = rows[0]
            by_style = generate_warmup_paraphrases_by_style(schema, best_intent.tables or [], sql=best_sql)
            if not by_style:
                continue
            raw_phrases = flatten_warmup_paraphrases_by_style(by_style)
            if not raw_phrases:
                continue
            cells[key] = AnchorLatticeCell(
                key=key, representative_intent_id=best_intent.intent_id, anchors=tuple(raw_phrases)
            )
        return AnchorLattice(cells=cells)

    @staticmethod
    def _warmup_collect_phrases_for_intent(
        intent: SeedWarmupIntent,
        final_sql: str,
        schema: SchemaGraph,
        *,
        anchor: str | None = None,
        lattice_anchors: list[str] | None = None,
    ) -> tuple[list[str], dict[str, list[str]] | None]:
        """Return candidate NL phrases and optional per-style buckets for one warmup intent."""
        origin_sql_history = (intent.source or "") == "sql_history"
        origin_gold = (intent.source or "gold") == "gold"
        if origin_gold:
            seed = (
                (anchor or "").strip()
                or (intent.seed_prompt_original or "").strip()
                or (intent.seed_prompt_normalized or "").strip()
                or (intent.natural_language or "").strip()
                or (intent.question or "").strip()
            )
            if not seed:
                return [], None
            by_style = generate_warmup_paraphrases_by_style(schema, intent.tables or [], seed_question=seed)
            phrases = [seed]
            if by_style:
                for p in flatten_warmup_paraphrases_by_style(by_style):
                    if normalize_question(p) != normalize_question(seed):
                        phrases.append(p)
            return phrases, by_style
        if origin_sql_history:
            by_style = generate_warmup_paraphrases_by_style(schema, intent.tables or [], sql=final_sql)
            phrases = flatten_warmup_paraphrases_by_style(by_style) if by_style else []
            if not phrases:
                freeform = generate_warmup_questions_freeform(schema, intent.tables or [], sql=final_sql)
                phrases = list(freeform or [])
            typed = {str(k): list(v) for k, v in by_style.items() if isinstance(v, list)} if by_style else None
            return phrases, typed
        if lattice_anchors:
            return list(lattice_anchors), None
        by_style = generate_warmup_paraphrases_by_style(schema, intent.tables or [], sql=final_sql)
        phrases = flatten_warmup_paraphrases_by_style(by_style) if by_style else []
        if not phrases:
            freeform = generate_warmup_questions_freeform(schema, intent.tables or [], sql=final_sql)
            phrases = list(freeform or [])
        return phrases, by_style

    @staticmethod
    def resolve_warmup_max_kept_intents(
        max_kept_intents: int | None | _WarmupCapDefaultSentinel,
    ) -> tuple[int | None, bool]:
        """Return ``(cap, uncapped)`` for warmup sampling budget resolution."""
        if isinstance(max_kept_intents, _WarmupCapDefaultSentinel):
            return SeedWarmupConfig.WARMUP_TARGET_CAP, False
        if max_kept_intents is None:
            return None, True
        return max_kept_intents, False

    @staticmethod
    def _work_unit_seed_intent(wu: dict[str, Any]) -> SeedWarmupIntent | None:
        raw = wu.get("serialized_intent")
        if not isinstance(raw, dict):
            return None
        try:
            return SeedWarmupIntent.from_dict(raw)
        except (TypeError, ValueError, KeyError):
            return None

    @staticmethod
    def _work_unit_question_llm_succeeded(wu: dict[str, Any]) -> bool:
        if str(wu.get("lifecycle_state") or "") not in {"sampled_in", "llm_done"}:
            return False
        ql = wu.get("question_llm")
        if not isinstance(ql, dict):
            return False
        if ql.get("failure_code") or ql.get("filtered_empty"):
            return False
        sel = str(ql.get("selected_question") or ql.get("question") or "").strip()
        if sel:
            return True
        qs = ql.get("questions") or ql.get("raw_phrases") or []
        return bool(isinstance(qs, list) and any(str(x).strip() for x in qs))

    @staticmethod
    def derive_capped_warmup_view_from_uncapped(
        work_units: list[dict[str, Any]],
        *,
        max_kept_intents: int | None | _WarmupCapDefaultSentinel = _WARMUP_CAP_DEFAULT,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Replay capped sampling and fillback on cached uncapped work units without new LLM calls."""
        fp_to_wu: dict[str, dict[str, Any]] = {}
        intents: list[SeedWarmupIntent] = []
        for wu_in in work_units:
            intent = SeedWarmupCacheSession._work_unit_seed_intent(wu_in)
            fp = str(wu_in.get("intent_fingerprint") or "")
            if intent is None or not fp:
                continue
            fp_to_wu[fp] = wu_in
            intents.append(intent)
        ordered_intents = sorted(intents, key=SeedWarmupCacheSession._seed_warmup_intent_sort_key)
        cap_n = SeedWarmupConfig.MAX_WARMUP_EXECUTE_UNITS
        if len(ordered_intents) > cap_n:
            ordered_intents = ordered_intents[:cap_n]

        execute_positions: list[int] = []
        position_rsig: dict[int, str] = {}
        pending_meta: dict[int, tuple[SeedWarmupIntent, str, dict[str, Any]]] = {}
        for idx, intent in enumerate(ordered_intents):
            fp = SeedWarmupCacheSession.warmup_intent_fingerprint(intent)
            wu_row = fp_to_wu.get(fp)
            if wu_row is None:
                continue
            er = wu_row.get("execute_result")
            if not isinstance(er, dict) or not er.get("ok"):
                continue
            execute_positions.append(idx)
            final_sql = str(er.get("final_sql") or "")
            pending_meta[idx] = (intent, final_sql, wu_row)

        fillback_cap, _uncapped_exec = SeedWarmupCacheSession.resolve_warmup_max_kept_intents(max_kept_intents)
        sampled, gold_dropped_pos, sampling_detail = SeedWarmupCacheSession._warmup_submodular_cover_select(
            ordered_intents, execute_positions, position_rsig=position_rsig, max_kept_intents=max_kept_intents
        )
        sampling_detail["result_signature_positions"] = len(position_rsig)
        sampling_detail["selection_order"] = sorted(sampled)
        sampling_detail["max_kept_intents"] = None if _uncapped_exec else fillback_cap
        sampling_detail["derived_from_uncapped"] = True

        replay_traces: list[dict[str, Any]] = []
        for idx in execute_positions:
            intent, final_sql, _wu = pending_meta[idx]
            em = intent.expansion_metadata
            replay_traces.append(
                {
                    "intent_index": idx,
                    "intent_id": intent.intent_id,
                    "seed_index": intent.seed_index,
                    "origin": intent.source or "gold",
                    "selected": idx in sampled,
                    "uncertainty_score": round(SeedWarmupCacheSession._warmup_llm_uncertainty_score(intent), 4),
                    "result_signature": position_rsig.get(idx),
                    "stratum_id": SeedWarmupCacheSession._warmup_stratum_key(intent),
                    "expansion_operator": (em.operator if em else None),
                    "expansion_depth": (em.depth if em else None),
                    "expansion_path": list(em.expansion_path) if em and em.expansion_path else [],
                    "parent_intent_id": (em.parent_intent_id if em else None),
                    "generated_sql": final_sql,
                    "complexity_tier": intent.complexity_tier().value,
                    "body_key": body_similarity_key(intent.to_runtime_intent()),
                    "coverage_atoms": sorted(
                        SeedWarmupCacheSession._warmup_submodular_atoms_for_row(
                            ordered_intents, idx, position_rsig=position_rsig
                        )
                    ),
                }
            )
        sampling_detail["replay_traces"] = replay_traces
        sampling_drop_by_index: dict[int, dict[str, Any]] = {
            int(r["intent_index"]): r
            for r in sampling_detail.get("sampling_drop_records", [])
            if isinstance(r, dict) and r.get("intent_index") is not None
        }

        final_sampled = set(sampled)
        success_positions: set[int] = {
            idx
            for idx in final_sampled
            if SeedWarmupCacheSession._work_unit_question_llm_succeeded(pending_meta[idx][2])
        }
        unsampled_ordered = [i for i in execute_positions if i not in final_sampled]
        for _ in range(SeedWarmupConfig.WARMUP_MAX_FILLBACK_ROUNDS):
            if fillback_cap is not None and len(success_positions) >= fillback_cap:
                break
            shortfall = (fillback_cap - len(success_positions)) if fillback_cap is not None else len(unsampled_ordered)
            if shortfall <= 0 or not unsampled_ordered:
                break
            pool = [i for i in unsampled_ordered if i not in success_positions]
            candidates = [
                i for i in pool if SeedWarmupCacheSession._work_unit_question_llm_succeeded(pending_meta[i][2])
            ]
            if not candidates:
                break
            atoms_rows = [
                SeedWarmupCacheSession._warmup_submodular_atoms_for_row(ordered_intents, p, position_rsig=position_rsig)
                for p in candidates
            ]
            universe: set[str] = set()
            for row_a in atoms_rows:
                universe |= set(row_a)
            if universe:
                order_local = greedy_cover_indices_by_atoms(atoms_rows, frozenset(universe))
                batch = [candidates[i] for i in order_local[:shortfall]]
            else:
                batch = candidates[:shortfall]
            for idx in batch:
                final_sampled.add(idx)
                success_positions.add(idx)
            unsampled_ordered = [p for p in unsampled_ordered if p not in set(batch)]

        sampling_detail["fillback_derived_success_count"] = len(success_positions)

        capped_units: list[dict[str, Any]] = []
        for idx, intent in enumerate(ordered_intents):
            fp = SeedWarmupCacheSession.warmup_intent_fingerprint(intent)
            wu_row = fp_to_wu.get(fp)
            if wu_row is None:
                continue
            rec = dict(wu_row)
            si = rec.get("serialized_intent")
            if isinstance(si, dict) and "runtime" not in rec:
                rec["runtime"] = si
            if idx not in execute_positions:
                capped_units.append(rec)
                continue
            if idx in success_positions:
                if SeedWarmupCacheSession._work_unit_question_llm_succeeded(wu_row):
                    rec["lifecycle_state"] = "llm_done"
                capped_units.append(rec)
                continue
            rec["lifecycle_state"] = "failed"
            if idx in final_sampled:
                rec["failure_code"] = "missing_question_data"
                rec["drop_reason_code"] = "missing_question_data"
            else:
                drop_rec = sampling_drop_by_index.get(idx)
                if drop_rec:
                    fc = str(drop_rec.get("failure_code") or "stratum_quota_exceeded")
                    rec["failure_code"] = fc
                    rec["drop_reason_code"] = fc
                else:
                    fc_fallback = "gold_cap_exceeded" if idx in gold_dropped_pos else "stratum_quota_exceeded"
                    rec["failure_code"] = fc_fallback
                    rec["drop_reason_code"] = fc_fallback
            capped_units.append(rec)

        for trace in replay_traces:
            idx = int(trace.get("intent_index", -1))
            trace["selected"] = idx in success_positions
            drop_rec = sampling_drop_by_index.get(idx)
            if drop_rec is not None:
                trace["drop_failure_code"] = drop_rec.get("failure_code")
                trace["drop_detail"] = drop_rec.get("detail")
        sampling_detail["selection_order"] = sorted(success_positions)
        sampling_detail["coverage_order"] = SeedWarmupCacheSession._warmup_positions_coverage_order(
            sorted(success_positions), ordered_intents, position_rsig=position_rsig
        )

        return capped_units, sampling_detail

    @staticmethod
    def _warmup_submodular_cover_select(
        ordered_intents: list[SeedWarmupIntent],
        eligible_positions: list[int],
        *,
        position_rsig: dict[int, str] | None = None,
        max_kept_intents: int | None | _WarmupCapDefaultSentinel = _WARMUP_CAP_DEFAULT,
    ) -> tuple[set[int], set[int], dict[str, Any]]:
        """Return ``(sampled_positions, gold_cap_dropped_positions, sampling_detail)`` for *eligible_positions*. Sampling picks whole intents only; it never perturbs structural ``s*`` parameters inside an intent row."""
        cfg = SeedWarmupConfig
        target, uncapped = SeedWarmupCacheSession.resolve_warmup_max_kept_intents(max_kept_intents)
        drop_records: list[dict[str, Any]] = []
        detail: dict[str, Any] = {
            "skipped_due_to_low_volume": False,
            "strata": [],
            "policy_version": cfg.WARMUP_SAMPLING_POLICY_VERSION,
            "counts": {},
            "sampling_drops_by_code": {},
            "sampling_drop_records": drop_records,
            "target_cap": None if uncapped else target,
            "uncapped": uncapped,
        }
        e_list = list(eligible_positions)
        if uncapped:
            if not e_list:
                detail["counts"] = {"eligible_total": 0}
                return set(), set(), detail
            mmr_all = SeedWarmupCacheSession._warmup_mmr_order(e_list, ordered_intents, cfg.WARMUP_MMR_LAMBDA)
            ded, ded_drops = SeedWarmupCacheSession._warmup_dedupe_jaccard_positions(mmr_all, ordered_intents)
            drop_records.extend(ded_drops)
            kept_set = set(ded)
            dropped = set(e_list) - kept_set
            detail["counts"] = {
                "eligible_total": len(e_list),
                "gold_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") == "gold"),
                "synthetic_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") != "gold"),
                "target_cap": None,
                "gold_kept": len(kept_set),
                "gold_dropped": len(dropped),
                "synthetic_budget": 0,
                "synthetic_kept": len(kept_set),
                "synthetic_dropped": len(dropped),
            }
            return kept_set, dropped, detail
        if len(e_list) <= cfg.WARMUP_KEEP_ALL_BELOW:
            detail["skipped_due_to_low_volume"] = True
            kept_set = SeedWarmupCacheSession._warmup_low_volume_cap_positions(e_list, ordered_intents)
            dropped = set(e_list) - kept_set
            detail["counts"] = {
                "eligible_total": len(e_list),
                "gold_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") == "gold"),
                "synthetic_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") != "gold"),
                "target_cap": target,
                "gold_kept": len(kept_set),
                "gold_dropped": len(dropped),
                "synthetic_budget": 0,
                "synthetic_kept": 0,
                "synthetic_dropped": 0,
            }
            detail["sampling_drops_by_code"] = {}
            return kept_set, dropped, detail

        g_positions = [i for i in e_list if (ordered_intents[i].source or "gold") == "gold"]
        s_positions = [i for i in e_list if i not in set(g_positions)]
        assert target is not None
        min_gold = 0
        if g_positions:
            pool_n = len(e_list)
            gr = len(g_positions) / max(pool_n, 1)
            min_gold = min(
                len(g_positions),
                max(
                    int(math.ceil(SeedWarmupCacheSession._warmup_min_gold_fraction() * target)),
                    int(math.ceil(gr * target)),
                ),
            )
        gold_stratum: dict[str, list[int]] = {}
        for pos in g_positions:
            gsk = SeedWarmupCacheSession._warmup_stratum_key(ordered_intents[pos])
            gold_stratum.setdefault(gsk, []).append(pos)
        counts_g = {k: len(v) for k, v in gold_stratum.items()}
        g_kept: set[int] = set()
        if min_gold > 0 and counts_g:
            g_floor = cfg.WARMUP_STRATUM_MIN if min_gold >= len(counts_g) * cfg.WARMUP_STRATUM_MIN else 0
            gold_budget = min(min_gold, len(g_positions))
            quotas_g = SeedWarmupCacheSession._allocate_stratum_quotas(counts_g, gold_budget, g_floor)
            for gsk in sorted(gold_stratum.keys()):
                lst = gold_stratum[gsk]
                take_n = min(quotas_g.get(gsk, 0), len(lst))
                taken = lst[:take_n]
                g_kept.update(taken)
                for j, pos in enumerate(lst):
                    if j < take_n:
                        continue
                    oi = ordered_intents[pos]
                    drop_records.append(
                        {
                            "drop_phase": "sampling",
                            "intent_index": pos,
                            "intent_id": getattr(oi, "intent_id", None),
                            "failure_code": "gold_stratum_quota_exceeded",
                            "origin": "gold",
                            "stratum_id": gsk,
                            "quota": take_n,
                            "rank_in_stratum": j,
                            "detail": (
                                f"gold_stratum_quota_exceeded stratum_id={gsk!r} quota={take_n} "
                                f"rank_in_stratum={j} pool_size={len(lst)}"
                            ),
                        }
                    )
        budget = max(0, target - len(g_kept))
        stratum_to_positions: dict[str, list[int]] = {}
        for pos in s_positions:
            sk = SeedWarmupCacheSession._warmup_stratum_key(ordered_intents[pos])
            stratum_to_positions.setdefault(sk, []).append(pos)
        syn_kept: set[int] = set()
        if budget == 0 and s_positions:
            for pos in s_positions:
                oi = ordered_intents[pos]
                sk = SeedWarmupCacheSession._warmup_stratum_key(oi)
                drop_records.append(
                    {
                        "drop_phase": "sampling",
                        "intent_index": pos,
                        "intent_id": getattr(oi, "intent_id", None),
                        "failure_code": "global_cap_after_gold",
                        "origin": "synthetic",
                        "stratum_id": sk,
                        "quota": 0,
                        "rank_in_stratum": None,
                        "detail": (
                            f"global_cap_after_gold stratum_id={sk!r} "
                            f"synthetic_budget=0 target_cap={target} gold_kept={len(g_kept)}"
                        ),
                    }
                )
            detail["strata"] = [
                {
                    "stratum_id": sk,
                    "pool_size": len(lst),
                    "quota": 0,
                    "kept": 0,
                    "dropped": len(lst),
                }
                for sk, lst in sorted(stratum_to_positions.items(), key=lambda x: x[0])
            ]
        elif s_positions:
            rem_syn = budget
            for sk in sorted(stratum_to_positions.keys()):
                lst = stratum_to_positions[sk]
                take_floor = min(cfg.WARMUP_STRATUM_MIN, len(lst), rem_syn) if rem_syn > 0 else 0
                for pos in lst[:take_floor]:
                    syn_kept.add(pos)
                rem_syn = budget - len(syn_kept)
            rest_syn = [p for p in s_positions if p not in syn_kept]
            rem_syn = budget - len(syn_kept)
            if rem_syn > 0 and rest_syn:
                atoms_rows = [
                    SeedWarmupCacheSession._warmup_submodular_atoms_for_row(
                        ordered_intents, p, position_rsig=position_rsig
                    )
                    for p in rest_syn
                ]
                universe_syn: set[str] = set()
                for row_a in atoms_rows:
                    universe_syn |= set(row_a)
                if universe_syn:
                    order_local = greedy_cover_indices_by_atoms(atoms_rows, frozenset(universe_syn))
                    for li in order_local:
                        if len(syn_kept) >= budget:
                            break
                        syn_kept.add(rest_syn[li])
                ii = 0
                while len(syn_kept) < budget and ii < len(rest_syn):
                    syn_kept.add(rest_syn[ii])
                    ii += 1
            syn_only_sorted = sorted(p for p in syn_kept if p in set(s_positions))
            mmr_syn = SeedWarmupCacheSession._warmup_mmr_order(syn_only_sorted, ordered_intents, cfg.WARMUP_MMR_LAMBDA)
            ded_syn, red_drops = SeedWarmupCacheSession._warmup_dedupe_jaccard_positions(mmr_syn, ordered_intents)
            syn_kept = set(ded_syn)
            drop_records.extend(red_drops)
            for sk in sorted(stratum_to_positions.keys()):
                lst = stratum_to_positions[sk]
                kept_here = [p for p in lst if p in syn_kept]
                take_n = len(kept_here)
                detail["strata"].append(
                    {
                        "stratum_id": sk,
                        "pool_size": len(lst),
                        "quota": take_n,
                        "kept": take_n,
                        "dropped": len(lst) - take_n,
                    }
                )
                for j, pos in enumerate(lst):
                    if pos in syn_kept:
                        continue
                    oi = ordered_intents[pos]
                    drop_records.append(
                        {
                            "drop_phase": "sampling",
                            "intent_index": pos,
                            "intent_id": getattr(oi, "intent_id", None),
                            "failure_code": "stratum_quota_exceeded",
                            "origin": "synthetic",
                            "stratum_id": sk,
                            "quota": take_n,
                            "rank_in_stratum": j,
                            "detail": (
                                f"stratum_quota_exceeded stratum_id={sk!r} quota={take_n} "
                                f"rank_in_stratum={j} pool_size={len(lst)}"
                            ),
                        }
                    )
        combined: set[int] = set(g_kept) | set(syn_kept)
        if len(combined) < target:
            for pos in g_positions:
                if len(combined) >= target:
                    break
                if pos not in combined:
                    combined.add(pos)
        if len(combined) < target:
            for pos in s_positions:
                if len(combined) >= target:
                    break
                if pos not in combined:
                    combined.add(pos)
        gold_dropped = {p for p in g_positions if p not in combined}
        by_code: dict[str, int] = {}
        for row in drop_records:
            c = str(row.get("failure_code") or "")
            if c:
                by_code[c] = by_code.get(c, 0) + 1
        detail["sampling_drops_by_code"] = by_code
        detail["counts"] = {
            "eligible_total": len(e_list),
            "gold_eligible": len(g_positions),
            "synthetic_eligible": len(s_positions),
            "target_cap": target,
            "gold_kept": len(g_kept),
            "gold_dropped": len(gold_dropped),
            "synthetic_budget": budget,
            "synthetic_kept": len(syn_kept),
            "synthetic_dropped": len(s_positions) - len(syn_kept),
        }
        return combined, gold_dropped, detail

    @staticmethod
    def _warmup_positions_coverage_order(
        positions: list[int],
        ordered_intents: list[SeedWarmupIntent],
        *,
        position_rsig: dict[int, str] | None = None,
    ) -> list[int]:
        """Return a deterministic coverage-first ordering for selected positions."""
        if len(positions) <= 1:
            return list(positions)
        atoms_rows = [
            SeedWarmupCacheSession._warmup_submodular_atoms_for_row(ordered_intents, pos, position_rsig=position_rsig)
            for pos in positions
        ]
        universe: set[str] = set()
        for row_atoms in atoms_rows:
            universe |= set(row_atoms)
        ordered: list[int] = []
        if universe:
            greedy_order = greedy_cover_indices_by_atoms(atoms_rows, frozenset(universe))
            ordered.extend(positions[i] for i in greedy_order)
        remaining = [pos for pos in positions if pos not in set(ordered)]
        if remaining:
            ordered.extend(
                SeedWarmupCacheSession._warmup_mmr_order(remaining, ordered_intents, SeedWarmupConfig.WARMUP_MMR_LAMBDA)
            )
        return ordered

    @staticmethod
    def _create_template_from_result(
        result: SeedWarmupResult,
        schema: SchemaGraph,
        next_id: int,
        dialect: Any | None = None,
        source: str = "synthetic",
        trust_level: int = 1,
        seed_warmup_intent: SeedWarmupIntent | None = None,
        question_phrases: list[str] | None = None,
        *,
        store: dict[str, Any] | None = None,
        templates: dict[str, Template] | None = None,
        structural_match_templates: list[Template] | None = None,
    ) -> Template | None:
        """Create a `Template` from a successful seed warmup execution. result."""
        if not result.success or not result.sql:
            return None

        raw_intent = result.intent
        intent = raw_intent.to_runtime_intent() if hasattr(raw_intent, "to_runtime_intent") else raw_intent
        param_values = intent.param_values if hasattr(intent, "param_values") else {}
        nl_row = (intent.natural_language or "") if hasattr(intent, "natural_language") else ""
        nl_row = nl_row or (result.question or "")

        phrases = list(question_phrases) if question_phrases else [result.question or ""]
        phrases = [p for p in phrases if p and str(p).strip()]
        if not phrases:
            phrases = [result.question or nl_row or "?"]

        first_q = phrases[0].strip()
        q_norm = normalize_question(first_q) or normalize_question(result.question or "") or "?"
        norm_opt = q_norm if q_norm != first_q else None
        form_storage = QuestionFormStorage(corrected=first_q, normalized_optional=norm_opt)

        vh: ValueHistory | None = None
        if seed_warmup_intent and (
            seed_warmup_intent.seed_prompt_original or seed_warmup_intent.seed_prompt_normalized
        ):
            vh = ValueHistory(param_values=[], questions=[], natural_language=[])
            orig = seed_warmup_intent.seed_prompt_original or seed_warmup_intent.seed_prompt_normalized
            norm = seed_warmup_intent.seed_prompt_normalized or orig
            vh.add(param_values, orig, nl_row)
            vh.add(param_values, norm, nl_row)
            for phrase in phrases:
                if phrase.strip() in {orig.strip(), norm.strip()}:
                    continue
                vh.add(param_values, phrase, nl_row)
        else:
            vh = None

        store_use = store if store is not None else {"next_id": next_id}
        templates_use = templates if templates is not None else {}
        structural_list = structural_match_templates
        if structural_list is None and templates_use:
            structural_list = collect_structural_match_templates(intent, templates_use, schema=schema)

        tmpl = TemplateOps.insert_template(
            store_use,
            templates_use,
            schema,
            q_norm,
            intent,
            result.sql,
            dialect=dialect,
            structural_match_templates=structural_list,
            template_source=source,
            template_trust_level=trust_level,
            template_initial_stats=TemplateStats(accept=0, reject=0),
            template_value_history=vh,
            form_storage=form_storage,
        )
        debug(f"[seed_warmup.create_template] created: id={tmpl.id}")
        return tmpl

    @staticmethod
    def _build_value_domains(schema: SchemaGraph) -> dict[str, ValueDomain]:
        """Build `ValueDomain` objects from schema column metadata."""
        domains: dict[str, ValueDomain] = {}
        for table_name, table_meta in schema.tables.items():
            for col_name, col_meta in table_meta.columns.items():
                col_key = f"{table_name}.{col_name}"
                domains[col_key] = ValueDomain(
                    values=col_meta.frequent_values or [],
                    min_val=col_meta.min_val,
                    max_val=col_meta.max_val,
                    data_type=col_meta.data_type or None,
                )
        return domains

    @staticmethod
    def run_seed_warmup_execution(
        intents: list[SeedWarmupIntent],
        schema: SchemaGraph,
        dialect: Any,
        next_id: int,
        join_cache: dict[JoinCacheKey, JoinCacheEntry] | None = None,
        join_resolver_intent_index: dict[str, SeedWarmupIntent] | None = None,
        *,
        store_instance_keys: set[str] | None = None,
        accepted_templates: dict[str, Template] | None = None,
        warmup_cache: SeedWarmupCacheSession | None = None,
        warmup_report_version: int = 1,
        warmup_lattice_root: str | None = None,
        max_kept_intents: int | None | _WarmupCapDefaultSentinel = _WARMUP_CAP_DEFAULT,
        federation_manifest: Any | None = None,
        federation_mappings: Any | None = None,
        stores_by_source: dict[str, Any] | None = None,
        dialects_by_source: Mapping[str, Any] | None = None,
        source_runtimes: Mapping[str, Any] | None = None,
        member_graphs: Mapping[str, SchemaGraph] | None = None,
        federation_dir: str | None = None,
        persist_template_learning: bool = True,
    ) -> tuple[list[SeedWarmupResult], list[Template], int, dict[str, Any]]:
        """Execute SQL for each intent, stratify successes, then run question LLM on the sample."""
        if join_cache is None:
            join_cache = {}

        store_keys = store_instance_keys or set()
        tmpl_for_q = accepted_templates or {}

        id_to_intent = (
            join_resolver_intent_index
            if join_resolver_intent_index is not None
            else {i.intent_id: i for i in intents if getattr(i, "intent_id", "")}
        )

        value_domains = SeedWarmupCacheSession._build_value_domains(schema)
        results: list[SeedWarmupResult] = []
        new_templates_collected: list[Template] = []
        batch_templates: dict[str, Template] = dict(tmpl_for_q)
        warmup_store: dict[str, Any] = {"next_id": next_id}

        success_count = 0
        fail_count = 0
        validation_drop = 0
        realism_drop = 0
        question_generation_failed = 0
        join_resolution_failed = 0
        sql_build_failed = 0
        instantiation_failed = 0
        substitution_failed = 0
        empty_sql_failed = 0
        semantic_precheck_failed = 0
        template_instance_exists_count = 0
        not_sampled_after_execute = 0
        all_questions_dropped = 0
        warmup_path41_drop = 0
        warmup_path42_drop = 0
        drop_audit: list[dict[str, Any]] = []
        sampled_work_unit_ids: list[str] = []

        debug(f"run_seed_warmup_execution: processing {len(intents)} intents")
        debug("Warmup union pre-align: validate, execute, then sampling")

        if not warmup_lattice_root or not str(warmup_lattice_root).strip():
            raise ValueError("warmup_lattice_root is required for warmup failure logging and artifact paths")
        results_root = Path(os.path.abspath(warmup_lattice_root))
        cwd = Path.cwd().resolve()
        results_file: Path | None = None
        if results_root.resolve() != cwd:
            results_file = results_root / "live_tests" / "results.txt"

        ordered_intents = sorted(intents, key=SeedWarmupCacheSession._seed_warmup_intent_sort_key)
        cap_n = SeedWarmupConfig.MAX_WARMUP_EXECUTE_UNITS
        cap_drop: list[SeedWarmupIntent] = []
        if len(ordered_intents) > cap_n:
            cap_drop = ordered_intents[cap_n:]
            ordered_intents = ordered_intents[:cap_n]

        pending_success: list[
            tuple[
                int,
                SeedWarmupIntent,
                RuntimeIntent,
                str,
                list[Any],
                dict[str, Any],
                SeedWarmupResult,
                Any | None,
            ]
        ] = []

        for idx, intent in enumerate(ordered_intents):
            ifp = SeedWarmupCacheSession.warmup_intent_fingerprint(intent)

            with pipeline_capture(auto_responses=["y", "y", "y"]) as capture:
                runtime = intent.to_runtime_intent()
                runtime = apply_deterministic_repairs(
                    runtime, schema, intent.natural_language or intent.intent_id or ""
                )
                result = SeedWarmupResult(runtime, "")

                def _wu_record(
                    rt: RuntimeIntent,
                    *,
                    ok: bool,
                    final_sql: str | None,
                    fc: str | None,
                    err: str | None,
                    tik: str = "",
                    unit_fp: str = ifp,
                    unit_intent: SeedWarmupIntent = intent,
                ) -> None:
                    if not ok:
                        step_res = StepResult(
                            scenario_id=f"warmup:{unit_fp[:8]}",
                            question=unit_intent.natural_language or unit_intent.intent_id or "warmup intent",
                            status="failed",
                            error=err or fc or "unknown failure",
                            captured_logs=capture["logs"],
                            intent=rt,
                            sql=final_sql,
                        )
                        if results_file is not None:
                            append_failure_trace(step_res, results_file)

                    if warmup_cache is None:
                        return
                    bk = body_similarity_key(rt)
                    jkr = join_path_key_runtime(rt)
                    pack = SeedWarmupCacheSession._warmup_pack_execute(
                        rt,
                        ok=ok,
                        final_sql=final_sql,
                        failure_code=fc,
                        error=err,
                        body_key=bk,
                        join_path_key=jkr,
                        template_instance_key=tik,
                    )
                    warmup_cache.write_work_unit(unit_fp, unit_intent, pack, report_version=warmup_report_version)

                runtime, qual_msgs = check_qualified_refs_exist(runtime, schema)
                if qual_msgs:
                    result.error = qual_msgs[0]
                    result.failure_code = "warmup_qualified_refs"
                    results.append(result)
                    fail_count += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="warmup_qualified_refs", err=result.error)
                    continue

                lit_rt, pp_issues = apply_runtime_post_processing_lite(
                    runtime, schema, question_fallback=intent.intent_id or ""
                )
                if lit_rt is None:
                    result.error = "warmup_post_processing_lite_failed"
                    result.failure_code = "warmup_post_processing_lite_failed"
                    results.append(result)
                    fail_count += 1
                    _wu_record(
                        runtime, ok=False, final_sql=None, fc="warmup_post_processing_lite_failed", err=result.error
                    )
                    continue
                if any((i.severity or "").lower() == "error" for i in pp_issues):
                    result.error = f"warmup_post_processing_lite_failed: {pp_issues[0].message}"
                    result.failure_code = "warmup_post_processing_lite_failed"
                    results.append(result)
                    fail_count += 1
                    _wu_record(
                        lit_rt, ok=False, final_sql=None, fc="warmup_post_processing_lite_failed", err=result.error
                    )
                    continue
                runtime = lit_rt

                sem_msgs = curated_warmup_semantic_issues(runtime, schema)
                if sem_msgs:
                    result.error = f"warmup_semantic_precheck: {sem_msgs[0]}"
                    result.failure_code = "warmup_semantic_precheck"
                    results.append(result)
                    fail_count += 1
                    semantic_precheck_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="warmup_semantic_precheck", err=result.error)
                    continue

                if warmup_cache is not None:
                    er_hit = warmup_cache.get_cached_execute(ifp)
                    if er_hit is not None:
                        warmup_cache.execute_hits += 1
                        rt_hit = RuntimeIntent.from_dict(er_hit["runtime"])
                        res_hit = SeedWarmupResult(rt_hit, "")
                        res_hit.sql = er_hit.get("final_sql")
                        res_hit.rows = None
                        res_hit.failure_code = er_hit.get("failure_code")
                        res_hit.error = er_hit.get("error") or ""
                        res_hit.execute_ok = bool(er_hit.get("ok"))
                        if not er_hit.get("ok"):
                            results.append(res_hit)
                            fail_count += 1
                            fch = er_hit.get("failure_code") or ""
                            if fch == "join_resolution_failed":
                                join_resolution_failed += 1
                            elif fch == "sql_build_failed":
                                sql_build_failed += 1
                            elif fch == "instantiation_failed":
                                instantiation_failed += 1
                            elif fch == "substitution_failed":
                                substitution_failed += 1
                            elif fch == "empty_sql_after_substitution":
                                empty_sql_failed += 1
                            elif fch == "validation_exception_unexpected" or (
                                isinstance(fch, str) and fch.startswith("ast_validate_")
                            ):
                                validation_drop += 1
                            elif fch == "explain_failed":
                                validation_drop += 1
                            elif fch == "execution_failed":
                                validation_drop += 1
                            elif fch == "template_instance_exists":
                                template_instance_exists_count += 1
                            elif fch == "warmup_path41_not_allowed":
                                warmup_path41_drop += 1
                            elif fch == "warmup_path42_not_allowed":
                                warmup_path42_drop += 1
                            continue
                        tik_hit = str(er_hit.get("template_instance_key") or "")
                        if tik_hit in store_keys:
                            res_hit.failure_code = "template_instance_exists"
                            res_hit.error = "template_instance_exists"
                            results.append(res_hit)
                            fail_count += 1
                            template_instance_exists_count += 1
                            continue
                        res_hit.success = False
                        fs_sql = str(er_hit.get("final_sql") or "")
                        all_pv_hit = dict(rt_hit.param_values or {})
                        pending_success.append((idx, intent, rt_hit, fs_sql, [], all_pv_hit, res_hit, None))
                        results.append(res_hit)
                        continue

                try:
                    reuse_entry = SeedWarmupCacheSession._ambiguous_join_reuse_from_parent(
                        intent, join_cache, id_to_intent
                    )
                    nl_hint = (intent.natural_language or intent.seed_prompt_normalized or "").strip()
                    if nl_hint:
                        join_id, join_sig, candidates = SeedWarmupCacheSession.resolve_joins_for_table_set(
                            intent.tables or [],
                            schema,
                            nl_hint,
                            join_cache,
                            ambiguous_reuse_entry=reuse_entry,
                            hint_is_natural_language=True,
                        )
                    else:
                        join_id, join_sig, candidates = SeedWarmupCacheSession.resolve_joins_for_table_set(
                            intent.tables or [],
                            schema,
                            intent.intent_id or "",
                            join_cache,
                            ambiguous_reuse_entry=reuse_entry,
                            hint_is_natural_language=False,
                        )
                except Exception as e:
                    result.error = f"join_resolution_failed: {e}"
                    result.failure_code = "join_resolution_failed"
                    results.append(result)
                    fail_count += 1
                    join_resolution_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="join_resolution_failed", err=result.error)
                    continue

                try:
                    runtime = apply_default_structural_values(runtime)
                    runtime = prune_unused_cte_steps(runtime)

                    det_sql = build_deterministic_sql(runtime, None, schema, dialect)
                    if join_id != "J00" and join_sig:
                        join_kinds = edge_kinds_for_join_candidate(candidates, join_id)
                        det_sql = inject_join_into_deterministic_sql(
                            det_sql,
                            [join_sig],
                            schema=schema,
                            edge_kinds_ordered=[join_kinds],
                            dialect=dialect,
                        )
                    runtime.sql_param = det_sql
                    runtime.chosen_join_candidate_id = join_id
                    runtime.chosen_join_path_signature = canonicalize_stored_join_path_signature(
                        list(join_sig or []),
                        from_anchor=runtime.tables[0] if runtime.tables else None,
                    )
                    syn_drop = SeedWarmupCacheSession._warmup_synthetic_store_path_blocks(intent, runtime, tmpl_for_q)
                    if syn_drop:
                        result.failure_code = syn_drop
                        result.error = syn_drop
                        results.append(result)
                        fail_count += 1
                        if syn_drop == "warmup_path41_not_allowed":
                            warmup_path41_drop += 1
                        else:
                            warmup_path42_drop += 1
                        _wu_record(runtime, ok=False, final_sql=None, fc=syn_drop, err=syn_drop, tik="")
                        continue
                except Exception as e:
                    result.error = f"sql_build_failed: {e}"
                    result.failure_code = "sql_build_failed"
                    results.append(result)
                    fail_count += 1
                    sql_build_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="sql_build_failed", err=result.error)
                    continue

                instantiated = SeedWarmupCacheSession.instantiate_intent(intent, value_domains)
                if instantiated is None:
                    result.error = "instantiation_failed"
                    result.failure_code = "instantiation_failed"
                    results.append(result)
                    fail_count += 1
                    instantiation_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="instantiation_failed", err=result.error)
                    continue

                all_params = dict(instantiated.param_values)
                all_params.update(runtime.param_values or {})
                try:
                    runtime.sql_param = det_sql
                    final_sql = finalize_substitute_sql(runtime, structural_defaults_src=None, params=all_params)
                except Exception as e:
                    result.error = f"substitution_failed: {e}"
                    result.failure_code = "substitution_failed"
                    results.append(result)
                    fail_count += 1
                    substitution_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="substitution_failed", err=result.error)
                    continue

                if not final_sql or not final_sql.strip():
                    result.error = "empty_sql_after_substitution"
                    result.failure_code = "empty_sql_after_substitution"
                    results.append(result)
                    fail_count += 1
                    empty_sql_failed += 1
                    _wu_record(runtime, ok=False, final_sql=None, fc="empty_sql_after_substitution", err=result.error)
                    continue

                post_msgs_pb = curated_warmup_post_binding_issues(runtime, schema, final_sql)
                if post_msgs_pb:
                    result.error = f"warmup_post_binding_semantics: {post_msgs_pb[0]}"
                    result.failure_code = "warmup_post_binding_semantics"
                    results.append(result)
                    fail_count += 1
                    _wu_record(
                        runtime, ok=False, final_sql=final_sql, fc="warmup_post_binding_semantics", err=result.error
                    )
                    continue

                rows: list[Any] | None = None
                federation_executed = False
                federated_prepared: Any | None = None
                if (
                    federation_manifest is not None
                    and schema_spans_multiple_sources(schema)
                    and stores_by_source is not None
                ):
                    qn = normalize_question(intent.natural_language or intent.intent_id or "warmup")
                    ok_fed, err_fed, rows_fed, fed_sql, fed_prep = execute_federated_warmup_intent(
                        qn,
                        runtime,
                        schema,
                        dialect,
                        federation_manifest=federation_manifest,
                        federation_mappings=federation_mappings,
                        stores_by_source=stores_by_source,
                        dialects_by_source=dialects_by_source,
                        source_runtimes=source_runtimes,
                        member_graphs=member_graphs,
                        federation_dir=federation_dir,
                        persist_template_learning=persist_template_learning,
                    )
                    if not ok_fed:
                        result.error = err_fed or "federated_warmup_failed"
                        result.failure_code = "federated_warmup_failed"
                        results.append(result)
                        validation_drop += 1
                        fail_count += 1
                        _wu_record(
                            runtime,
                            ok=False,
                            final_sql=fed_sql or final_sql,
                            fc="federated_warmup_failed",
                            err=result.error,
                        )
                        continue
                    final_sql = fed_sql or final_sql
                    rows = rows_fed
                    federation_executed = True
                    federated_prepared = fed_prep

                if not federation_executed:
                    try:
                        ok, err, vcat, vdiags = validate_sql(
                            dialect,
                            final_sql,
                            bind_params_for_sql(final_sql, all_params),
                            schema=schema,
                            intent=runtime,
                        )
                    except Exception as e:
                        result.error = f"validation_exception: {e}"
                        result.failure_code = "validation_exception_unexpected"
                        results.append(result)
                        validation_drop += 1
                        fail_count += 1
                        _wu_record(
                            runtime,
                            ok=False,
                            final_sql=final_sql,
                            fc="validation_exception_unexpected",
                            err=result.error,
                        )
                        continue

                    skip_after_diag = False
                    if not ok:
                        for _rep in range(SeedWarmupConfig.WARMUP_DIAGNOSTIC_REPAIR_MAX_ROUNDS):
                            runtime_r, chg = apply_diagnostic_repairs(runtime, schema, vdiags)
                            if not chg:
                                break
                            try:
                                runtime = runtime_r
                                runtime = apply_default_structural_values(runtime)
                                runtime = prune_unused_cte_steps(runtime)
                                det_sql = build_deterministic_sql(runtime, None, schema, dialect)
                                if join_id != "J00" and join_sig:
                                    join_kinds = edge_kinds_for_join_candidate(candidates, join_id)
                                    det_sql = inject_join_into_deterministic_sql(
                                        det_sql,
                                        [join_sig],
                                        schema=schema,
                                        edge_kinds_ordered=[join_kinds],
                                        dialect=dialect,
                                    )
                                runtime.sql_param = det_sql
                                runtime.chosen_join_candidate_id = join_id
                                runtime.chosen_join_path_signature = canonicalize_stored_join_path_signature(
                                    list(join_sig or []),
                                    from_anchor=runtime.tables[0] if runtime.tables else None,
                                )
                                instantiated2 = SeedWarmupCacheSession.instantiate_intent(intent, value_domains)
                                if instantiated2 is None:
                                    break
                                all_params2 = dict(instantiated2.param_values)
                                all_params2.update(runtime.param_values or {})
                                final_sql = finalize_substitute_sql(
                                    runtime, structural_defaults_src=None, params=all_params2
                                )
                                all_params = all_params2
                                post_retry = curated_warmup_post_binding_issues(runtime, schema, final_sql)
                                if post_retry:
                                    result.error = f"warmup_post_binding_semantics: {post_retry[0]}"
                                    result.failure_code = "warmup_post_binding_semantics"
                                    results.append(result)
                                    validation_drop += 1
                                    fail_count += 1
                                    _wu_record(
                                        runtime,
                                        ok=False,
                                        final_sql=final_sql,
                                        fc="warmup_post_binding_semantics",
                                        err=result.error,
                                    )
                                    skip_after_diag = True
                                    break
                                if not final_sql or not final_sql.strip():
                                    break
                                ok, err, vcat, vdiags = validate_sql(
                                    dialect,
                                    final_sql,
                                    bind_params_for_sql(final_sql, runtime.param_values),
                                    schema=schema,
                                    intent=runtime,
                                )
                                if ok:
                                    break
                            except Exception:
                                break

                    if skip_after_diag:
                        continue

                    if not ok:
                        vcode = seed_warmup_failure_code_from_validate_sql_error(
                            err, failure_category=vcat.value if vcat is not None else None
                        )
                        result.error = f"validation_failed: {err}"
                        result.failure_code = vcode
                        results.append(result)
                        validation_drop += 1
                        fail_count += 1
                        _wu_record(runtime, ok=False, final_sql=final_sql, fc=vcode, err=result.error)
                        continue

                    try:
                        exec_sql = dialect.finalize_render(
                            runtime.sql_param or "",
                            all_params,
                            schema=schema,
                            intent=runtime,
                            execution_sql_override=None,
                            structural_defaults=None,
                        )
                        if dialect.can_explain():
                            ok_ex, _diags_ex, err_ex = dialect.explain_diagnose(
                                exec_sql, all_params, schema=schema, intent=runtime
                            )
                            if not ok_ex:
                                result.error = f"explain_failed: {err_ex}"
                                result.failure_code = "explain_failed"
                                results.append(result)
                                validation_drop += 1
                                fail_count += 1
                                _wu_record(
                                    runtime, ok=False, final_sql=final_sql, fc="explain_failed", err=result.error
                                )
                                continue
                        rows = SeedWarmupCacheSession.execute_warmup_sql_rows(
                            runtime, schema, dialect, exec_sql, all_params
                        )
                    except Exception as e:
                        result.error = f"execution_failed: {e}"
                        result.failure_code = "execution_failed"
                        results.append(result)
                        validation_drop += 1
                        fail_count += 1
                        _wu_record(runtime, ok=False, final_sql=final_sql, fc="execution_failed", err=result.error)
                        continue

            sfp = Dialect.compute_sql_fp(final_sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())
            tik = template_instance_key_for_runtime(runtime, sfp)
            if not federation_executed and tik in store_keys:
                result.failure_code = "template_instance_exists"
                result.error = "template_instance_exists"
                results.append(result)
                fail_count += 1
                template_instance_exists_count += 1
                _wu_record(
                    runtime, ok=False, final_sql=final_sql, fc="template_instance_exists", err=result.error, tik=tik
                )
                continue

            runtime.param_values = all_params
            result.intent = runtime
            result.sql = final_sql
            result.rows = rows
            result.execute_ok = True
            result.success = False
            _wu_record(runtime, ok=True, final_sql=final_sql, fc=None, err=None, tik=tik)
            pending_success.append(
                (idx, intent, runtime, final_sql, rows or [], all_params, result, federated_prepared)
            )
            results.append(result)

        pre_cap_drop = 0
        for dropped_intent in cap_drop:
            rt = dropped_intent.to_runtime_intent()
            r = SeedWarmupResult(rt, "")
            r.failure_code = "pre_execute_absolute_cap"
            r.error = "pre_execute_absolute_cap"
            results.append(r)
            pre_cap_drop += 1

        execute_positions = [t[0] for t in pending_success]
        assert len(execute_positions) == len(pending_success)
        position_rsig: dict[int, str] = {}
        for idx, _intent, _runtime, _final_sql, rows, _all_params, _res, _fed_prep in pending_success:
            sig = SeedWarmupCacheSession._warmup_canonical_result_signature(rows)
            if sig:
                position_rsig[idx] = sig
        fillback_cap, _uncapped_exec = SeedWarmupCacheSession.resolve_warmup_max_kept_intents(max_kept_intents)
        sampled, gold_dropped_pos, sampling_detail = SeedWarmupCacheSession._warmup_submodular_cover_select(
            ordered_intents, execute_positions, position_rsig=position_rsig, max_kept_intents=max_kept_intents
        )
        sampling_detail["result_signature_positions"] = len(position_rsig)
        sampling_detail["selection_order"] = sorted(sampled)
        sampling_detail["max_kept_intents"] = None if _uncapped_exec else fillback_cap
        replay_traces: list[dict[str, Any]] = []
        for idx, intent, _runtime, final_sql, _rows, _all_params, _res, _fed_prep in pending_success:
            em = intent.expansion_metadata
            replay_traces.append(
                {
                    "intent_index": idx,
                    "intent_id": intent.intent_id,
                    "seed_index": intent.seed_index,
                    "origin": intent.source or "gold",
                    "selected": idx in sampled,
                    "uncertainty_score": round(SeedWarmupCacheSession._warmup_llm_uncertainty_score(intent), 4),
                    "result_signature": position_rsig.get(idx),
                    "stratum_id": SeedWarmupCacheSession._warmup_stratum_key(intent),
                    "expansion_operator": (em.operator if em else None),
                    "expansion_depth": (em.depth if em else None),
                    "expansion_path": list(em.expansion_path) if em and em.expansion_path else [],
                    "parent_intent_id": (em.parent_intent_id if em else None),
                    "generated_sql": final_sql,
                    "complexity_tier": intent.complexity_tier().value,
                    "body_key": body_similarity_key(intent.to_runtime_intent()),
                    "coverage_atoms": sorted(
                        SeedWarmupCacheSession._warmup_submodular_atoms_for_row(
                            ordered_intents, idx, position_rsig=position_rsig
                        )
                    ),
                    "selected_after_sampling": idx in sampled,
                }
            )
        sampling_detail["replay_traces"] = replay_traces
        trace_by_index = {
            int(t["intent_index"]): t
            for t in replay_traces
            if isinstance(t, dict) and t.get("intent_index") is not None
        }
        sampling_drop_by_index: dict[int, dict[str, Any]] = {
            int(r["intent_index"]): r
            for r in sampling_detail.get("sampling_drop_records", [])
            if isinstance(r, dict) and r.get("intent_index") is not None
        }

        synthetic_pairs_for_lattice: list[tuple[SeedWarmupIntent, str]] = []
        for idx, intent, _runtime, final_sql, _rows, _all_params, _res, _fed_prep in pending_success:
            if idx not in sampled:
                continue
            if (intent.source or "gold") == "gold":
                continue
            synthetic_pairs_for_lattice.append((intent, final_sql))

        seen_nl_norm: set[str] = set()
        lattice_path: str | None = None
        lattice_disk: dict[str, list[str]] = {}
        if warmup_lattice_root:
            lattice_path = SeedWarmupCacheSession._warmup_anchor_lattice_json_path(warmup_lattice_root, schema)
            lattice_disk = SeedWarmupCacheSession._load_warmup_anchor_lattice(lattice_path)
        anchor_lattice = SeedWarmupCacheSession.build_anchor_lattice(synthetic_pairs_for_lattice, schema, lattice_disk)
        lattice_runtime_persist: dict[str, list[str]] = {
            k.signature(schema.schema_graph_id): list(cell.anchors) for k, cell in anchor_lattice.cells.items()
        }
        pending_by_idx = {t[0]: t for t in pending_success}
        fillback_batches = 0
        success_positions: set[int] = set()

        def _warmup_question_llm_branch(
            idx: int,
            intent: SeedWarmupIntent,
            final_sql: str,
            res: SeedWarmupResult,
            *,
            runtime_intent: RuntimeIntent | None = None,
            federated_prepared: Any | None = None,
        ) -> bool:
            nonlocal success_count, fail_count, question_generation_failed, realism_drop, all_questions_dropped
            nonlocal sampled_work_unit_ids, seen_nl_norm, new_templates_collected

            if warmup_cache is not None:
                sw = warmup_cache.mark_sampled_in(SeedWarmupCacheSession.warmup_intent_fingerprint(intent))
                if sw:
                    sampled_work_unit_ids.append(sw)

            origin_gold = (intent.source or "gold") == "gold"
            lattice_anchors: list[str] | None = None
            if not origin_gold and (intent.source or "") != "sql_history":
                akey = intent.anchor_lattice_key()
                cell = anchor_lattice.cells.get(akey)
                if cell is not None and cell.anchors:
                    lattice_anchors = list(cell.anchors)
            raw_phrases, by_style = SeedWarmupCacheSession._warmup_collect_phrases_for_intent(
                intent, final_sql, schema, lattice_anchors=lattice_anchors
            )
            if not raw_phrases:
                res.error = "question_generation_failed"
                res.failure_code = "question_generation_failed"
                fail_count += 1
                question_generation_failed += 1
                trace = trace_by_index.get(idx)
                if trace is not None:
                    trace["question_generation_status"] = "failed"
                    trace["question_failure_code"] = "question_generation_failed"
                    trace["question_candidate_count"] = 0
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        SeedWarmupCacheSession.warmup_intent_fingerprint(intent),
                        {"questions": [], "failure_code": "question_generation_failed"},
                        ok=False,
                    )
                return False
            for trace in replay_traces:
                if trace.get("intent_index") == idx:
                    trace["candidate_phrases"] = list(raw_phrases)
                    trace["paraphrases_by_style"] = by_style or {}
                    trace["question_candidate_count"] = len(raw_phrases)
                    break

            filtered: list[str] = []
            for phrase in raw_phrases:
                qn = normalize_question(phrase)
                if not qn:
                    continue
                if other_template_owns_question_string(tmpl_for_q, "__warmup_new__", qn):
                    continue
                collides = False
                for prev in seen_nl_norm:
                    if exact_question_match(qn, prev, label="warmup_seen_nl"):
                        collides = True
                        break
                if collides:
                    continue
                filtered.append(phrase.strip())
                seen_nl_norm.add(qn)

            if not filtered:
                res.error = "all_questions_collided_or_empty"
                res.failure_code = "all_questions_collided_or_empty"
                fail_count += 1
                all_questions_dropped += 1
                trace = trace_by_index.get(idx)
                if trace is not None:
                    trace["question_generation_status"] = "failed"
                    trace["question_failure_code"] = "all_questions_collided_or_empty"
                    trace["filtered_question_count"] = 0
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        SeedWarmupCacheSession.warmup_intent_fingerprint(intent),
                        {
                            "questions": raw_phrases,
                            "filtered_empty": True,
                            "failure_code": "all_questions_collided_or_empty",
                        },
                        ok=False,
                    )
                return False

            selected = filtered[0]
            res.question = selected
            res.questions = list(filtered)
            res.success = True
            success_positions.add(idx)

            if warmup_cache is not None:
                warmup_cache.record_question_llm(
                    SeedWarmupCacheSession.warmup_intent_fingerprint(intent),
                    {
                        "questions": filtered,
                        "raw_phrases": raw_phrases,
                        "paraphrases_by_style": by_style or {},
                        "selected_question": selected,
                        "selection_reason": "first_filtered",
                        "is_realistic": True,
                    },
                    ok=True,
                )
            for trace in replay_traces:
                if trace.get("intent_index") == idx:
                    trace["selected_question"] = selected
                    trace["selection_reason"] = "first_filtered"
                    trace["filtered_question_count"] = len(filtered)
                    trace["question_generation_status"] = "success"
                    break

            if (
                federated_prepared is not None
                and persist_template_learning
                and stores_by_source is not None
                and federation_manifest is not None
            ):
                qn = normalize_question(selected) or selected
                parent_rt = runtime_intent if runtime_intent is not None else intent.to_runtime_intent()
                created = persist_federated_warmup_learning(
                    qn,
                    parent_rt,
                    federated_prepared,
                    schema,
                    stores_by_source=stores_by_source,
                    dialects_by_source=dialects_by_source,
                    member_graphs=member_graphs,
                    federation_dir=federation_dir,
                    federation_manifest=federation_manifest,
                    question_phrases=filtered,
                )
                new_templates_collected.extend(created)
            else:
                tmpl = SeedWarmupCacheSession._create_template_from_result(
                    res,
                    schema,
                    int(warmup_store["next_id"]),
                    dialect,
                    seed_warmup_intent=intent,
                    question_phrases=filtered,
                    store=warmup_store,
                    templates=batch_templates,
                )
                if tmpl:
                    new_templates_collected.append(tmpl)
            success_count += 1
            return True

        for idx, intent, runtime, final_sql, _rows, _all_params, res, fed_prep in pending_success:
            if idx not in sampled:
                not_sampled_after_execute += 1
                rec = sampling_drop_by_index.get(idx)
                trace = trace_by_index.get(idx)
                if rec:
                    fc = str(rec.get("failure_code") or "stratum_quota_exceeded")
                    res.failure_code = fc
                    res.error = str(rec.get("detail") or fc)
                    drop_audit.append(rec)
                    if trace is not None:
                        trace["drop_failure_code"] = fc
                        trace["drop_detail"] = rec.get("detail")
                else:
                    oi = ordered_intents[idx]
                    origin = "gold" if (oi.source or "gold") == "gold" else "synthetic"
                    fc_fallback = "gold_cap_exceeded" if idx in gold_dropped_pos else "stratum_quota_exceeded"
                    res.failure_code = fc_fallback
                    res.error = fc_fallback
                    drop_audit.append(
                        {
                            "drop_phase": "sampling",
                            "intent_index": idx,
                            "intent_id": oi.intent_id,
                            "failure_code": fc_fallback,
                            "origin": origin,
                            "stratum_id": (
                                SeedWarmupCacheSession._warmup_stratum_key(oi) if origin == "synthetic" else None
                            ),
                            "detail": fc_fallback,
                        }
                    )
                    if trace is not None:
                        trace["drop_failure_code"] = fc_fallback
                        trace["drop_detail"] = fc_fallback
                fail_count += 1
                continue
            _warmup_question_llm_branch(
                idx, intent, final_sql, res, runtime_intent=runtime, federated_prepared=fed_prep
            )

        unsampled_ordered = [i for i in execute_positions if i not in sampled]
        fillback_cap, _uncapped_exec = SeedWarmupCacheSession.resolve_warmup_max_kept_intents(max_kept_intents)
        for _ in range(SeedWarmupConfig.WARMUP_MAX_FILLBACK_ROUNDS):
            if fillback_cap is not None and success_count >= fillback_cap:
                break
            shortfall = (fillback_cap - success_count) if fillback_cap is not None else len(unsampled_ordered)
            if shortfall <= 0 or not unsampled_ordered:
                break
            pool = list(unsampled_ordered)
            atoms_rows = [
                SeedWarmupCacheSession._warmup_submodular_atoms_for_row(ordered_intents, p, position_rsig=position_rsig)
                for p in pool
            ]
            universe: set[str] = set()
            for row_a in atoms_rows:
                universe |= set(row_a)
            if universe:
                order_local = greedy_cover_indices_by_atoms(atoms_rows, frozenset(universe))
                batch = [pool[i] for i in order_local[:shortfall]]
            else:
                batch = pool[:shortfall]
            unsampled_ordered = [p for p in unsampled_ordered if p not in set(batch)]
            fillback_batches += 1
            for idx in batch:
                row = pending_by_idx.get(idx)
                if row is None:
                    continue
                _i, intent, runtime, final_sql, _rows, _all_params, res, fed_prep = row
                fail_count -= 1
                res.failure_code = None
                res.error = None
                res.success = False
                res.question = ""
                trace = trace_by_index.get(idx)
                if trace is not None:
                    trace["selected_via_fillback"] = True
                _warmup_question_llm_branch(
                    idx, intent, final_sql, res, runtime_intent=runtime, federated_prepared=fed_prep
                )

        debug(
            f"run_seed_warmup_execution: "
            f"{success_count} success, {fail_count} failed "
            f"(validation_drop={validation_drop}, "
            f"realism_drop={realism_drop}), "
            f"{len(new_templates_collected)} templates"
        )

        early = (
            join_resolution_failed
            + sql_build_failed
            + instantiation_failed
            + substitution_failed
            + empty_sql_failed
            + semantic_precheck_failed
        )
        warmup_funnel: dict[str, Any] = {
            "validation_drop": validation_drop,
            "realism_drop": realism_drop,
            "question_generation_failed": question_generation_failed,
            "early_pipeline_failed": early,
            "join_resolution_failed": join_resolution_failed,
            "sql_build_failed": sql_build_failed,
            "instantiation_failed": instantiation_failed,
            "substitution_failed": substitution_failed,
            "empty_sql_failed": empty_sql_failed,
            "warmup_semantic_precheck_failed": semantic_precheck_failed,
            "template_instance_exists": template_instance_exists_count,
            "not_sampled_after_execute": not_sampled_after_execute,
            "pre_execute_absolute_cap": pre_cap_drop,
            "all_questions_collided_or_empty": all_questions_dropped,
            "warmup_sampling": sampling_detail,
            "execute_ok_count": len(pending_success),
            "cache_execute_hits": warmup_cache.execute_hits if warmup_cache else 0,
            "warmup_drop_audit": drop_audit,
            "warmup_path41_not_allowed": warmup_path41_drop,
            "warmup_path42_not_allowed": warmup_path42_drop,
            "warmup_touched_work_unit_ids": (list(warmup_cache.touched_work_unit_ids) if warmup_cache else []),
            "warmup_sampled_work_unit_ids": sampled_work_unit_ids,
            "warmup_fillback_batches": fillback_batches,
        }
        sampling_detail["selection_order"] = sorted(success_positions)
        sampling_detail["coverage_order"] = SeedWarmupCacheSession._warmup_positions_coverage_order(
            sorted(success_positions), ordered_intents, position_rsig=position_rsig
        )
        for trace in replay_traces:
            idx = int(trace.get("intent_index", -1))
            trace["selected"] = idx in success_positions
            trace["selected_after_fillback"] = idx in success_positions
        if lattice_path is not None:
            SeedWarmupCacheSession._save_warmup_anchor_lattice(lattice_path, schema, lattice_runtime_persist)

        return results, new_templates_collected, int(warmup_store["next_id"]), warmup_funnel

    @staticmethod
    def seed_warmup_drops_jsonl_path_for_report(report_filepath: str) -> str | None:
        """Return sibling drops JSONL path for a seed warmup report filename."""
        d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
        if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
            ver = base[len("seed_warmup_report_v") : -len(".json")]
            return os.path.join(d, f"seed_warmup_drops_v{ver}.jsonl")
        return None

    @staticmethod
    def seed_warmup_drops_detail_jsonl_path_for_report(report_filepath: str) -> str | None:
        """Return path for per-row sampling ``drops_detail`` JSONL next to the report file."""
        d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
        if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
            ver = base[len("seed_warmup_report_v") : -len(".json")]
            return os.path.join(d, f"seed_warmup_drops_detail_v{ver}.jsonl")
        return None

    @staticmethod
    def seed_warmup_replay_manifest_path_for_report(report_filepath: str) -> str | None:
        """Return sibling replay-manifest JSON path for a seed warmup report filename."""
        d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
        if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
            ver = base[len("seed_warmup_report_v") : -len(".json")]
            return os.path.join(d, f"warmup_replay_manifest_v{ver}.json")
        return None

    @staticmethod
    def seed_warmup_provenance_path_for_report(report_filepath: str) -> str | None:
        """Return sibling provenance JSON path for a seed warmup report filename."""
        d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
        if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
            ver = base[len("seed_warmup_report_v") : -len(".json")]
            return os.path.join(d, f"seed_warmup_provenance_v{ver}.json")
        return None

    @staticmethod
    def seed_warmup_provenance_rows_path_for_report(report_filepath: str) -> str | None:
        """Return sibling provenance JSONL path for per-row seed warmup traces."""
        d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
        if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
            ver = base[len("seed_warmup_report_v") : -len(".json")]
            return os.path.join(d, f"seed_warmup_provenance_rows_v{ver}.jsonl")
        return None

    @staticmethod
    def _warmup_frontier_summary(replay_traces: list[dict[str, Any]], selection_order: list[int]) -> dict[str, Any]:
        """Summarize marginal coverage gain across the selected frontier."""
        atom_map: dict[int, frozenset[str]] = {}
        pool_atoms: set[str] = set()
        selected_traces = 0
        for row in replay_traces:
            idx = row.get("intent_index")
            if idx is None:
                continue
            atoms = frozenset(str(a) for a in (row.get("coverage_atoms") or []) if str(a))
            atom_map[int(idx)] = atoms
            pool_atoms |= set(atoms)
            if row.get("selected"):
                selected_traces += 1

        ordered = [idx for idx in selection_order if idx in atom_map]
        if not ordered:
            return {
                "selected_total": selected_traces,
                "pool_atom_union_size": len(pool_atoms),
                "selected_atom_union_size": 0,
                "selected_atom_coverage_ratio": 0.0,
                "checkpoints": [],
                "head_avg_new_atoms": 0.0,
                "tail_avg_new_atoms": 0.0,
                "tail_vs_head_ratio": 0.0,
            }

        checkpoints: list[dict[str, Any]] = []
        seen_atoms: set[str] = set()
        marginal_gains: list[int] = []
        checkpoint_sizes = sorted(
            {
                1,
                len(ordered),
                max(1, int(math.ceil(len(ordered) * 0.10))),
                max(1, int(math.ceil(len(ordered) * 0.25))),
                max(1, int(math.ceil(len(ordered) * 0.50))),
                max(1, int(math.ceil(len(ordered) * 0.75))),
            }
        )
        for rank, idx in enumerate(ordered, start=1):
            atoms = atom_map.get(idx, frozenset())
            new_atoms = len(set(atoms) - seen_atoms)
            marginal_gains.append(new_atoms)
            seen_atoms |= set(atoms)
            if rank not in checkpoint_sizes:
                continue
            checkpoints.append(
                {
                    "kept": rank,
                    "atom_union_size": len(seen_atoms),
                    "atom_coverage_ratio": round(len(seen_atoms) / max(1, len(pool_atoms)), 4),
                    "new_atoms_last_step": new_atoms,
                    "avg_new_atoms_per_intent": round(sum(marginal_gains) / len(marginal_gains), 4),
                }
            )

        head_n = max(1, len(marginal_gains) // 4)
        tail_n = max(1, len(marginal_gains) // 4)
        head_avg = sum(marginal_gains[:head_n]) / head_n
        tail_avg = sum(marginal_gains[-tail_n:]) / tail_n
        return {
            "selected_total": len(ordered),
            "pool_atom_union_size": len(pool_atoms),
            "selected_atom_union_size": len(seen_atoms),
            "selected_atom_coverage_ratio": round(len(seen_atoms) / max(1, len(pool_atoms)), 4),
            "checkpoints": checkpoints,
            "head_avg_new_atoms": round(head_avg, 4),
            "tail_avg_new_atoms": round(tail_avg, 4),
            "tail_vs_head_ratio": round(tail_avg / head_avg, 4) if head_avg > 0 else 0.0,
        }

    @staticmethod
    def save_warmup_replay_manifest(report_filepath: str, funnel: dict[str, Any] | None) -> None:
        """Persist intermediate warmup scores, selection order, and removal reasons."""
        path = SeedWarmupCacheSession.seed_warmup_replay_manifest_path_for_report(report_filepath)
        if not path:
            return
        fn = funnel or {}
        ws_raw = fn.get("warmup_sampling")
        ws = cast(dict[str, Any], ws_raw) if isinstance(ws_raw, dict) else {}
        manifest: dict[str, Any] = {
            "policy_version": SeedWarmupConfig.WARMUP_SAMPLING_POLICY_VERSION,
            "code_version": SeedWarmupConfig.SEED_WARMUP_CODE_VERSION,
            "anchor_lattice_code_version": SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION,
            "sampling_profile": SeedWarmupConfig.WARMUP_SAMPLING_PROFILE,
            "selection_order": list(ws.get("selection_order") or []),
            "result_signature_positions": int(ws.get("result_signature_positions") or 0),
            "counts": ws.get("counts") or {},
            "sampling_drops_by_code": ws.get("sampling_drops_by_code") or {},
            "removal_records": list(ws.get("sampling_drop_records") or []),
            "intent_traces": list(ws.get("replay_traces") or []),
        }
        lock_dir = SeedWarmupCacheSession._warmup_artifacts_lock_dir(path)
        with artifact_lock(lock_dir):
            write_json_atomic(path, manifest, sort_keys=False)
        debug(f"save_warmup_replay_manifest: wrote replay manifest to {path}")

    @staticmethod
    def save_seed_warmup_provenance(report_filepath: str, funnel: dict[str, Any] | None) -> None:
        """Persist private seed warmup provenance summaries and per-row traces."""
        path = SeedWarmupCacheSession.seed_warmup_provenance_path_for_report(report_filepath)
        rows_path = SeedWarmupCacheSession.seed_warmup_provenance_rows_path_for_report(report_filepath)
        if not path or not rows_path:
            return
        fn = funnel or {}
        ws_raw = fn.get("warmup_sampling")
        ws = cast(dict[str, Any], ws_raw) if isinstance(ws_raw, dict) else {}
        replay_traces = list(ws.get("replay_traces") or [])
        selection_order = list(ws.get("coverage_order") or ws.get("selection_order") or [])
        payload = {
            "policy_version": SeedWarmupConfig.WARMUP_SAMPLING_POLICY_VERSION,
            "code_version": SeedWarmupConfig.SEED_WARMUP_CODE_VERSION,
            "sampling_profile": SeedWarmupConfig.WARMUP_SAMPLING_PROFILE,
            "target_cap": ws.get("target_cap"),
            "max_kept_intents": ws.get("max_kept_intents"),
            "derived_from_uncapped": bool(ws.get("derived_from_uncapped")),
            "counts": ws.get("counts") or {},
            "sampling_drops_by_code": ws.get("sampling_drops_by_code") or {},
            "selection_order": selection_order,
            "frontier": SeedWarmupCacheSession._warmup_frontier_summary(replay_traces, selection_order),
        }
        lock_dir = SeedWarmupCacheSession._warmup_artifacts_lock_dir(path)
        with artifact_lock(lock_dir):
            write_json_atomic(path, payload, sort_keys=False)
            with open(rows_path, "w", encoding="utf-8", newline="") as rf:
                for row in replay_traces:
                    if isinstance(row, dict):
                        rf.write(json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")
        debug(f"save_seed_warmup_provenance: wrote provenance summary to {path}")
        debug(f"save_seed_warmup_provenance: wrote {len(replay_traces)} row traces to {rows_path}")

    @staticmethod
    def save_seed_warmup_report(
        results: list[SeedWarmupResult], filepath: str, funnel: dict[str, Any] | None = None
    ) -> None:
        """Save compact aggregate seed warmup metrics and optional funnel. counters to a JSON file."""
        fn = funnel or {}
        ws = fn.get("warmup_sampling") or {}
        unrealistic_by_category: dict[str, int] = {}
        failure_histogram: dict[str, int] = {}
        for r in results:
            code = (r.failure_code or "").strip()
            if not code:
                code = "success" if r.success else "unknown"
            failure_histogram[code] = failure_histogram.get(code, 0) + 1
            if r.success:
                continue
            err = r.error or ""
            if SEED_FAILURE_CODE_REALISM_DROPPED in err:
                cat = (r.drop_reason_category or "other").strip() or "other"
                if cat not in REALISM_DROP_REASON_CATEGORIES:
                    cat = "other"
                unrealistic_by_category[cat] = unrealistic_by_category.get(cat, 0) + 1
        by_sqlstate: dict[str, int] = {}
        ast_validate_subcounts: dict[str, int] = {}
        for r in results:
            st = (r.sqlstate or "").strip()
            if st:
                by_sqlstate[st] = by_sqlstate.get(st, 0) + 1
        for fk, fv in failure_histogram.items():
            if fk.startswith("ast_validate_"):
                ast_validate_subcounts[fk] = fv
        drops_by_code: dict[str, int] = {}
        for row in fn.get("warmup_drop_audit") or []:
            if not isinstance(row, dict):
                continue
            c = str(row.get("failure_code") or "")
            if c in SEED_WARMUP_DROP_CODES:
                drops_by_code[c] = drops_by_code.get(c, 0) + 1
        sampling_for_report = (
            {k: v for k, v in ws.items() if k != "sampling_drop_records"} if isinstance(ws, dict) else {}
        )
        known_pipeline_failure_histogram = {
            k: v for k, v in failure_histogram.items() if k in SEED_WARMUP_FAILURE_CODES
        }
        report: dict[str, Any] = {
            "total": len(results),
            "success": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "failure_histogram": failure_histogram,
            "known_pipeline_failure_histogram": known_pipeline_failure_histogram,
            "drops_by_code": drops_by_code,
            "sampling": sampling_for_report,
            "failure_summary": {
                "by_stage": {
                    "validation": int(fn.get("validation_drop", 0)),
                    "join": int(fn.get("join_resolution_failed", 0)),
                    "unrealism_drop": int(fn.get("realism_drop", 0)),
                },
                "by_sqlstate": by_sqlstate,
                "ast_validate_subcounts": ast_validate_subcounts,
                "unrealistic_by_category": unrealistic_by_category,
            },
        }
        for k, v in fn.items():
            if k not in report:
                report[k] = v
        lock_dir = SeedWarmupCacheSession._warmup_artifacts_lock_dir(filepath)
        with artifact_lock(lock_dir):
            write_json_atomic(filepath, report, sort_keys=False)
            drops_path = SeedWarmupCacheSession.seed_warmup_drops_jsonl_path_for_report(filepath)
            if drops_path:
                audit = fn.get("warmup_drop_audit") or []
                with open(drops_path, "w", encoding="utf-8", newline="") as jf:
                    for row in audit:
                        if isinstance(row, dict):
                            jf.write(json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")
                debug(f"save_seed_warmup_report: wrote {len(audit)} sampling drop rows to {drops_path}")
            detail_path = SeedWarmupCacheSession.seed_warmup_drops_detail_jsonl_path_for_report(filepath)
            if detail_path and isinstance(ws, dict):
                detail_rows = list(ws.get("sampling_drop_records") or [])
                with open(detail_path, "w", encoding="utf-8", newline="") as df:
                    for row in detail_rows:
                        if isinstance(row, dict):
                            df.write(json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n")
                debug(f"save_seed_warmup_report: wrote {len(detail_rows)} sampling drops_detail rows to {detail_path}")
            SeedWarmupCacheSession.save_warmup_replay_manifest(filepath, fn)
            SeedWarmupCacheSession.save_seed_warmup_provenance(filepath, fn)
        debug(f"save_seed_warmup_report: wrote aggregate report for {len(results)} results to {filepath}")
