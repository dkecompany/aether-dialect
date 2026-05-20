"""Seed warmup: gold intents, expansion, joins, SQL validation, NL questions, cache, and artifact I/O."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from ._config import (
    DIAGNOSTIC_CODE_ENGINE_INFO,
    EMPTY_JOIN_CANDIDATES,
    JOIN_CHOICE_SCOPE_MAIN,
    JSON_COMPACT_SEPARATORS,
    REALISM_DROP_REASON_CATEGORIES,
    SEED_FAILURE_CODE_REALISM_DROPPED,
    SEED_NORMALIZATION_BATCH_SIZE,
    SEED_WARMUP_DROP_CODES,
    SEED_WARMUP_FAILURE_CODES,
    GenerationPath,
    SeedWarmupConfig,
    WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY,
    WARMUP_PARAPHRASE_COUNT_FROM_SQL,
    seed_warmup_failure_code_from_validate_sql_error,
)
from ._contracts_base import LlmJsonExhausted, SchemaGraph, TemplateStats, ValueDomain
from ._contracts_core import (
    AnchorLattice,
    AnchorLatticeCell,
    AnchorLatticeKey,
    FilterParam,
    HavingParam,
    QuestionFormStorage,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    Template,
    ValueHistory,
    anchor_lattice_key_for_seed_intent,
    anchor_lattice_signature,
    classify_seed_warmup_intent_complexity,
    operator_feature_vector_for_seed_intent,
    warmup_coverage_atoms_for_seed_intent,
)
from ._core_utils import (
    ask_user_choice,
    debug,
    llm_json,
    log,
    normalize_question,
    notify,
    print_info,
    stable_json,
    telemetry_capture,
)
from ._dialect import active_sqlglot_dialect, compute_sql_fp
from ._intent_expr import apply_default_structural_values
from ._intent_process import (
    apply_deterministic_repairs,
    apply_runtime_post_processing_lite,
    full_intent_parse,
    join_path_key_concrete,
    join_path_key_runtime,
    structural_compare,
)
from ._intent_repair import apply_diagnostic_repairs
from ._intent_resolve import check_qualified_refs_exist, prune_unused_cte_steps
from ._pipeline import finalize_substitute_sql, other_template_owns_question_string
from ._qsim_ops import greedy_cover_indices_by_atoms
from ._qsim import (
    deterministic_having_value,
    sample_coordinated_range,
    sample_value_from_domain,
)
from ._sql_gen import (
    build_deterministic_sql,
    get_join_choice_from_llm,
    inject_join_into_deterministic_sql,
    join_candidate_map,
    join_hints_multi,
    physical_tables_for_join_hints,
)
from ._templates import insert_template, template_is_live, warmup_work_unit_schema_refs
from ._utils import (
    body_similarity_key,
    body_similarity_key_for_concrete,
    exact_question_match,
    generate_bulk_anchors,
    generate_paraphrases_of_seed_question,
    generate_question_from_sql,
    intent_key,
    schema_context_enriched_lines_for_tables,
    select_best_question_via_judge,
    select_three_warmup_styles,
    template_instance_key_from_parts,
)
from ._validation_execute import bind_params_for_sql, validate_sql
from ._validation_execute import curated_warmup_post_binding_issues, curated_warmup_semantic_issues

_SEED_LINE_NORMALIZE_SYSTEM = (
    "You rephrase database analyst questions for clarity only. Do not answer them. "
    "Preserve all entities, filters, metrics, grouping, ordering, and limits implied by each source line. "
    "Do not add or remove requirements. Do not use SQL or qualified identifiers unless the source already does. "
    'Output only valid JSON: {"lines":[{"index":<int>,"normalized":"<string>"}]} with exactly one object '
    "per input index, indices matching the batch, no extra keys, no markdown."
)


def warmup_intent_fingerprint(intent: SeedWarmupIntent) -> str:
    """Stable SHA-256 hex of the serialized seed warmup intent (pre-execute)."""

    return hashlib.sha256(
        stable_json(intent.to_dict()).encode("utf-8"),
    ).hexdigest()


def warmup_pool_operator_feature_stats(intents: list[SeedWarmupIntent]) -> dict[str, Any]:
    """
    Summarize operator-vector diversity for seed-warmup funnel reporting.

    Args:

        intents: Candidate warmup intents after pool dedupe and store classification.

    Returns:

        Plain counters suitable for merging into the warmup JSON funnel.
    """

    vectors = [operator_feature_vector_for_seed_intent(i) for i in intents]
    distinct_vectors = len(set(vectors))
    bits = {
        (v.has_aggregate, v.has_grouping, v.has_having, v.window_kind != "none")
        for v in vectors
    }
    max_4 = WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY
    distinct_bits = len(bits)
    union_atoms: set[str] = set()
    for intent in intents:
        union_atoms |= warmup_coverage_atoms_for_seed_intent(intent)
    return {
        "warmup_queue_distinct_operator_vectors": distinct_vectors,
        "warmup_queue_operator_feature_4bit_tuple_distinct": distinct_bits,
        "warmup_queue_operator_feature_4bit_tuple_max": max_4,
        "warmup_queue_operator_feature_4bit_tuple_coverage_ratio": (
            round(distinct_bits / max_4, 4) if max_4 else 0.0
        ),
        "warmup_queue_coverage_atom_union_size": len(union_atoms),
    }


def _warmup_anchor_lattice_json_path(output_root: str, schema: SchemaGraph) -> str:
    """
    Absolute path to persisted anchor-lattice JSON for *schema* under *output_root*.

    Args:

        output_root: Warmup artifact directory (same root as seed warmup reports).

        schema: Schema graph carrying ``effective_structural_hash``.

    Returns:

        Path to the lattice sidecar file.
    """

    base = os.path.join(
        output_root,
        SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR,
    )
    fn = (
        f"lattice_{schema.effective_structural_hash}_"
        f"v{SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION}.json"
    )
    return os.path.join(base, fn)


def _load_warmup_anchor_lattice(path: str) -> dict[str, list[str]]:
    """
    Load lattice cell keys to anchor phrase lists from disk.

    Args:

        path: JSON file path from :func:`_warmup_anchor_lattice_json_path`.

    Returns:

        Mapping of cell key strings to phrase lists; empty when missing or invalid.
    """

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


def _save_warmup_anchor_lattice(
    path: str,
    schema: SchemaGraph,
    cells: dict[str, list[str]],
) -> None:
    """
    Persist anchor lattice cells atomically next to other warmup artifacts.

    Args:

        path: Destination JSON path.

        schema: Schema graph for fingerprint validation in the payload.

        cells: Mapping of :func:`anchor_lattice_signature` keys to anchor lists.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_fp": schema.effective_structural_hash,
        "code_version": SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_CODE_VERSION,
        "cells": {k: {"anchors": v} for k, v in sorted(cells.items())},
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)
    os.replace(tmp, path)


def load_seed_warmup_cache_zip(
    output_dir: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """
    Read ``seed_warmup_cache.zip`` from *output_dir*.

    Returns:

        ``(manifest, work_units_by_id)``; empty dicts when the zip is missing.
    """

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


def save_seed_warmup_cache_zip(
    output_dir: str,
    manifest: dict[str, Any],
    work_units: dict[str, dict[str, Any]],
    *,
    gold_intent_dicts: list[dict[str, Any]] | None = None,
) -> None:
    """Atomically write ``seed_warmup_cache.zip`` under *output_dir*."""

    path = os.path.join(output_dir, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
    os.makedirs(output_dir, exist_ok=True)
    out_manifest = {
        **manifest,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if gold_intent_dicts is not None:
        out_manifest["gold_intent_count"] = len(gold_intent_dicts)
    tmp = f"{path}.{os.getpid()}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
            json.dumps(out_manifest, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS),
        )
        if gold_intent_dicts is not None:
            zf.writestr(
                SeedWarmupConfig.WARMUP_CACHE_GOLD_INTENTS_JSON,
                json.dumps(
                    gold_intent_dicts,
                    ensure_ascii=False,
                    separators=JSON_COMPACT_SEPARATORS,
                ),
            )
        for wid, rec in work_units.items():
            zf.writestr(
                f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}{wid}.json",
                json.dumps(rec, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS),
            )
    os.replace(tmp, path)
    log(
        f"save_seed_warmup_cache_zip: wrote {len(work_units)} work units"
        f"{' and gold snapshot' if gold_intent_dicts is not None else ''} to {path}",
    )


@dataclass
class SeedWarmupCacheSession:
    """Mutable in-memory view of the seed warmup cache for one orchestration run."""

    manifest: dict[str, Any]
    work_units: dict[str, dict[str, Any]]
    fp_to_wid: dict[str, str] = field(default_factory=dict)
    execute_hits: int = 0
    touched_work_unit_ids: list[str] = field(default_factory=list)

    def ensure_work_unit_id(self, fingerprint: str) -> str:
        """Return stable id for *fingerprint*, allocating a UUID on first sight."""

        wid = self.fp_to_wid.get(fingerprint)
        if wid is None:
            wid = str(uuid.uuid4())
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
        self,
        fingerprint: str,
        intent: SeedWarmupIntent,
        packed_execute: dict[str, Any],
        *,
        report_version: int,
        is_preflight: bool,
    ) -> None:
        """Persist one work unit record and append *report_version* to session id lists."""

        wid = self.ensure_work_unit_id(fingerprint)
        bk = packed_execute.get("body_key") or ""
        jk = packed_execute.get("join_path_key") or ""
        tik = packed_execute.get("template_instance_key") or ""
        prev = self.work_units.get(wid, {})
        sid_key = "preflight_session_ids" if is_preflight else "run_session_ids"
        other_key = "run_session_ids" if is_preflight else "preflight_session_ids"
        sids = list(prev.get(sid_key, []))
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
            "preflight_session_ids": list(prev.get("preflight_session_ids", [])),
            "run_session_ids": list(prev.get("run_session_ids", [])),
        }
        rec[sid_key] = sids
        rec[other_key] = list(prev.get(other_key, []))
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

    def record_question_llm(
        self,
        fingerprint: str,
        payload: dict[str, Any],
        *,
        ok: bool,
    ) -> None:
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


def open_seed_warmup_cache_session(
    output_dir: str,
    schema: SchemaGraph,
    seed_content_sha256: str | None = None,
    *,
    sql_history_content_sha256: str | None = None,
) -> SeedWarmupCacheSession:
    """
    Load disk cache when seed content matches; prune work units whose schema refs are stale.

    Warmup units survive profiling-only drift when structural and effective hashes still match the manifest; structural drift triggers surgical drops keyed by ``warmup_work_unit_schema_refs``.
    """

    manifest, work_units = load_seed_warmup_cache_zip(output_dir)
    identity_ok = False
    if seed_content_sha256 is not None and manifest.get("seed_content_hash") == seed_content_sha256:
        identity_ok = True
    if sql_history_content_sha256 is not None and manifest.get(
        "sql_history_content_hash",
    ) == sql_history_content_sha256:
        identity_ok = True
    seed_ok = identity_ok
    prev_eff = str(manifest.get("effective_structural_hash") or manifest.get("schema_hash") or "")
    eff_ok = prev_eff == schema.effective_structural_hash
    prev_prof = str(manifest.get("profiling_hash") or "")

    if not seed_ok:
        work_units = {}
    elif not eff_ok:
        pruned: dict[str, dict[str, Any]] = {}
        for wid, wu in work_units.items():
            refs = warmup_work_unit_schema_refs(wu)
            if not refs.tables:
                continue
            ok, _ = template_is_live(refs, schema)
            if ok:
                pruned[wid] = wu
        work_units = pruned
    elif prev_prof and prev_prof != schema.profiling_hash:
        pass

    fp_to_wid = {
        str(wu["intent_fingerprint"]): str(wu["work_unit_id"])
        for wu in work_units.values()
        if wu.get("intent_fingerprint") and wu.get("work_unit_id")
    }
    manifest = {
        **manifest,
        "schema_hash": schema.schema_hash,
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


def _warmup_synthetic_store_path_blocks(
    intent: SeedWarmupIntent,
    runtime: RuntimeIntent,
    templates: dict[str, Template],
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


def get_next_seed_warmup_version(output_dir: str) -> int:
    """
    Return the next auto-incrementing version for seed warmup run reports and bundle zips.

    Args:

        output_dir: Directory containing prior ``seed_warmup_report_v*.json`` or ``seed_warmup_v*.zip``.

    Returns:

        One greater than the highest existing version, or ``1`` when none exist.
    """
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


def get_next_warmup_preflight_version(output_dir: str) -> int:
    """Return the next version for ``warmup_preflight_report_v*.json`` files in *output_dir*."""
    versions: list[int] = []
    pattern = os.path.join(output_dir, "warmup_preflight_report_v*.json")
    for fpath in glob.glob(pattern):
        base = os.path.basename(fpath)
        try:
            if base.startswith("warmup_preflight_report_v") and base.endswith(".json"):
                num = base[len("warmup_preflight_report_v") : -len(".json")]
                versions.append(int(num))
        except (IndexError, ValueError):
            continue
    return max(versions) + 1 if versions else 1


def run_seed_question_normalization(
    seeds: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, str]], str, str]:
    """
    Batch LLM normalization for seed lines; return phrases plus compact JSON and `.txt` payloads.

    Args:

        seeds: Seed dicts with `number` and `question`.

    Returns:

        Phrase map, compact JSON array text, and numbered normalized-lines text for bundling.
    """
    by_num: dict[int, str] = {int(s["number"]): str(s["question"]).strip() for s in seeds}
    sorted_nums = sorted(by_num.keys())
    out: dict[int, dict[str, str]] = {}
    for i in range(0, len(sorted_nums), SEED_NORMALIZATION_BATCH_SIZE):
        batch = sorted_nums[i : i + SEED_NORMALIZATION_BATCH_SIZE]
        payload = [{"index": n, "source": by_num[n]} for n in batch]
        user = stable_json({"batch": payload})
        try:
            parsed = llm_json(_SEED_LINE_NORMALIZE_SYSTEM, user, retries=2, task="intent")
        except LlmJsonExhausted as exc:
            debug(f"[seed_warmup.run_seed_question_normalization] llm_json exhausted on batch {i}: {exc}")
            parsed = {}
        lines = parsed.get("lines")
        if not isinstance(lines, list):
            lines = []
        got: dict[int, str] = {}
        for row in lines:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            nm = str(row.get("normalized", "")).strip()
            if idx is not None and nm:
                got[int(idx)] = nm
        for n in batch:
            out[n] = {"original": by_num[n], "normalized": got.get(n, by_num[n])}
    serial = [{"index": k, **v} for k, v in sorted(out.items(), key=lambda x: x[0])]
    json_body = json.dumps(serial, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)
    lines_txt = "".join(f"{k}. {v['normalized']}\n" for k, v in sorted(out.items(), key=lambda x: x[0]))
    log(f"run_seed_question_normalization: normalized {len(out)} seed lines")
    return out, json_body, lines_txt


def _load_seed_questions(filepath: str) -> list[dict[str, Any]]:
    """
    Load seed questions from a text file.

    Args:

        filepath: Description.

    Returns:

        Return value.
    """
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
                phase_match = re.search(
                    r"Phase\s+(\d+)",
                    line,
                    re.IGNORECASE,
                )
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


def _parse_gold_intent_strict(
    question: str,
    schema: SchemaGraph,
) -> tuple[RuntimeIntent | None, list[str]]:
    """
    Parse a seed question with optional retry when the first parse returns no intent.

    Args:

        question: Raw or normalized question text.

        schema: Schema graph.

    Returns:

        ``(intent, warning_messages)``. ``intent`` is ``None`` only when all attempts fail.
        Semantic warnings do not suppress a non-``None`` intent.
    """
    q_norm = normalize_question(question)
    intent, warns, _ = full_intent_parse(q_norm, schema)
    if intent is not None:
        return intent, list(warns)
    intent2, warns2, _ = full_intent_parse(q_norm, schema)
    if intent2 is not None:
        return intent2, list(warns2)
    debug(f"[seed_warmup.parse_gold_intent_strict] failed after retry: {q_norm}")
    return None, list(warns2 or warns)


def _replay_gold_intent_parse_for_telemetry(question: str, schema: SchemaGraph) -> None:
    """
    Run two `full_intent_parse` attempts for diagnostic replay when gold parse failed.

    Args:

        question: Seed question text.

        schema: Schema graph.

    Returns:

        None.
    """
    q_norm = normalize_question(question)
    full_intent_parse(q_norm, schema)
    full_intent_parse(q_norm, schema)


def _gold_failure_trace_text(seed_warmup_version: int, sections: list[str]) -> str:
    """
    Build concatenated gold-parse failure sections with header metadata.

    Args:

        seed_warmup_version: Run version stamp.

        sections: Per-seed diagnostic blocks.

    Returns:

        Full trace body for bundling into the seed warmup bundle zip.
    """
    header = (
        f"seed_warmup_version={seed_warmup_version}\n"
        f"failed_seed_count={len(sections)}\n"
        "interactive_gold=false\n"
        "Telemetry blocks are from a diagnostic replay (two full_intent_parse calls per seed).\n\n"
    )
    return header + "\n\n".join(sections)


def _confirm_gold_intent(
    question: str,
    intent: RuntimeIntent,
) -> tuple[bool, RuntimeIntent | None]:
    """
    Interactively confirm a parsed gold intent with the user.

    Args:

        question: Description.

        intent: Description.

    Returns:

        Return value.
    """
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

    print_info(
        f"Question: {question}\n\nI understood: {nl_summary}",
        items={
            "Tables": intent.tables or [],
            "Aggregations": agg_ops or ["none"],
            "Expected": expected_display,
        },
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


def _abstract_values(intent: RuntimeIntent) -> RuntimeIntent:
    """
    Return `intent` with `param_values` cleared.

    Args:

        intent: Description.

    Returns:

        Return value.
    """
    return RuntimeIntent(
        tables=intent.tables,
        grain=intent.grain,
        select_cols=intent.select_cols,
        group_by_cols=intent.group_by_cols,
        order_by_cols=intent.order_by_cols,
        filters_param=intent.filters_param,
        having_param=intent.having_param,
        param_values={},
        cte_steps=intent.cte_steps,
        column_map=intent.column_map,
        natural_language=intent.natural_language,
        limit=intent.limit,
    )


def run_gold_intent_generation(
    schema: SchemaGraph,
    seed_filepath: str,
    interactive: bool = True,
    seed_phrases: dict[int, dict[str, str]] | None = None,
    seed_warmup_version: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None, tuple[str, str] | None]:
    """
    Run the full gold intent generation pipeline from seed questions.

    Args:

        schema: Schema graph.

        seed_filepath: Seed text file.

        interactive: Confirm each intent when True.

        seed_phrases: Optional map seed number to original/normalized text; when None, normalization runs in memory.

        seed_warmup_version: Version stamp for failure trace metadata.

    Returns:

        Gold intent dicts, stats map, optional failure trace body for the seed warmup bundle zip, and optional
        ``(normalization_json, normalized_txt)`` when normalization ran here.
    """
    log("gold_intent_generation: starting")
    seeds = _load_seed_questions(seed_filepath)
    log(f"gold_intent_generation: {len(seeds)} seed questions loaded")

    norm_bundle: tuple[str, str] | None = None
    phrases = seed_phrases
    if phrases is None:
        phrases, norm_json, norm_txt = run_seed_question_normalization(seeds)
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

        log(f"gold_intent_generation: processing [{num}] {pack['normalized']}")
        intent, warns = _parse_gold_intent_strict(pack["normalized"], schema)
        if intent is None:
            log(f"gold_intent_generation: FAILED to parse [{num}]")
            fail_count += 1
            if not interactive:
                phase = seed.get("phase") or ""
                with telemetry_capture(suppress_console=True, force_diagnostic_flags=True) as cap_buf:
                    _replay_gold_intent_parse_for_telemetry(pack["normalized"], schema)
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
            log(f"gold_intent_generation: parsed [{num}] with {len(warns)} semantic warning(s)")

        if interactive:
            confirmed, final_intent = _confirm_gold_intent(pack["original"], intent)
            if not confirmed or final_intent is None:
                log(f"gold_intent_generation: user rejected [{num}]")
                user_rejected_count += 1
                continue
            intent = final_intent

        intent = _abstract_values(intent)
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
        trace_body = _gold_failure_trace_text(seed_warmup_version, failure_trace_sections)
        log(f"gold_intent_generation: failure trace built for {len(failure_trace_sections)} seeds")
    log(
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


JoinCacheEntry = tuple[str, list[str], dict[str, Any]]


def _seed_warmup_intent_sort_key(intent: SeedWarmupIntent) -> tuple[int, str]:
    """Order intents so shallower expansion layers resolve joins before deeper same-table children."""

    depth = intent.expansion_metadata.depth if intent.expansion_metadata else 0
    return (depth, intent.intent_id or "")


def _ambiguous_join_reuse_from_parent(
    intent: SeedWarmupIntent,
    join_cache: dict[frozenset[str], JoinCacheEntry],
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
    return join_cache.get(key)


def resolve_joins_for_table_set(
    tables: list[str],
    schema: SchemaGraph,
    question_hint: str,
    join_cache: dict[frozenset[str], JoinCacheEntry],
    *,
    ambiguous_reuse_entry: JoinCacheEntry | None = None,
) -> JoinCacheEntry:
    """
    Resolve join path for a table set, using cache when available.

    Args:

        tables: Description.

        schema: Description.

        question_hint: Description.

        join_cache: Description.

    Returns:

        Return value.
    """
    key = frozenset(tables)
    if key in join_cache:
        return join_cache[key]

    if len(tables) <= 1:
        entry: JoinCacheEntry = ("J00", [], EMPTY_JOIN_CANDIDATES)
        join_cache[key] = entry
        return entry

    join_tables = physical_tables_for_join_hints(tables, schema)
    candidates = join_hints_multi(
        schema,
        join_tables,
        virtual_specs={},
        include_semantic=False,
    )
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
    if chosen not in cmap:
        chosen = "J00"
    sig_list = list(cmap.get(chosen, []))
    entry = (chosen, sig_list, candidates)
    join_cache[key] = entry
    return entry


def _decompose_between_filter_param(f: FilterParam) -> list[FilterParam]:
    """
    Decompose a `between` filter into `>=` and `<=` filters.

    Args:

        f: Description.

    Returns:

        Return value.
    """
    if f.op != "between":
        return [f]
    return [
        replace(
            f,
            op=">=",
            param_key=f"{f.param_key}_lower" if f.param_key else None,
        ),
        replace(
            f,
            op="<=",
            param_key=f"{f.param_key}_upper" if f.param_key else None,
        ),
    ]


def _identify_range_pairs(
    filters: list[FilterParam],
) -> dict[str, dict[str, int]]:
    """
    Identify columns with paired lower and upper bound filters.

    Args:

        filters: Description.

    Returns:

        Return value.
    """
    column_ops: dict[str, dict[str, int]] = {}
    for idx, f in enumerate(filters):
        if f.right_expr:
            continue
        if f.op in (">", ">="):
            column_ops.setdefault(f.left_expr.primary_column, {})["lower_idx"] = idx
        elif f.op in ("<", "<="):
            column_ops.setdefault(f.left_expr.primary_column, {})["upper_idx"] = idx
    return {col: ops for col, ops in column_ops.items() if "lower_idx" in ops and "upper_idx" in ops}


def instantiate_intent(
    intent: SeedWarmupIntent,
    value_domains: dict[str, ValueDomain],
) -> SeedWarmupIntent | None:
    """
    Populate filter and HAVING values from profiling data.

    Args:

        intent: Description.

        value_domains: Description.

    Returns:

        Return value.
    """
    decomposed: list[FilterParam] = []
    for f in intent.filters_param:
        decomposed.extend(_decompose_between_filter_param(f))

    range_pairs = _identify_range_pairs(decomposed)
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

    new_filters: list[FilterParam] = []
    new_param_values: dict[str, Any] = {}

    for filter_idx, f in enumerate(decomposed):
        col_key = f.left_expr.primary_column
        op = f.op
        param_key = f.param_key or f"filter_{filter_idx}"

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
                value = sample_value_from_domain(
                    domain,
                    f.value_type,
                    f.op,
                    0,
                )
        else:
            value = sample_value_from_domain(
                domain,
                f.value_type,
                f.op,
                0,
            )

        if value is not None:
            new_param_values[param_key] = value
        new_filters.append(replace(f, param_key=param_key))

    new_having: list[HavingParam] = []
    for having_idx, h in enumerate(intent.having_param):
        param_key = h.param_key or f"having_{having_idx}"
        if h.right_expr is not None:
            new_having.append(replace(h, param_key=param_key))
            continue
        value = deterministic_having_value(
            h.left_expr.primary_term,
            0,
            having_idx,
        )
        new_param_values[param_key] = value
        new_having.append(replace(h, param_key=param_key))

    return SeedWarmupIntent(
        intent_id=intent.intent_id,
        tables=intent.tables,
        grain=intent.grain,
        select_cols=intent.select_cols,
        group_by_cols=intent.group_by_cols,
        order_by_cols=intent.order_by_cols,
        filters_param=new_filters,
        having_param=new_having,
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


def accepted_template_instance_keys(templates: dict[str, Template]) -> set[str]:
    """Return ``template_instance_key`` values for all accepted templates in *templates*."""

    keys: set[str] = set()
    for tmpl in templates.values():
        conc = tmpl.intent_signature
        bk = body_similarity_key_for_concrete(conc)
        jk = join_path_key_concrete(conc)
        keys.add(template_instance_key_from_parts(bk, jk, tmpl.sql_fp))
    return keys


def _warmup_stratum_key(warmup_intent: SeedWarmupIntent) -> str:
    """Deterministic stratum id for seed-warmup downsampling."""

    origin = "gold" if (warmup_intent.source or "gold") == "gold" else "synthetic"
    em = warmup_intent.expansion_metadata
    depth = em.depth if em else 0
    op = (em.operator if em else "") or ""
    tables_t = tuple(sorted(warmup_intent.tables or []))
    tables_fp = stable_json(tables_t)
    tier = classify_seed_warmup_intent_complexity(warmup_intent).value
    return f"{origin}|d{depth}|{op}|{tables_fp}|t{tier}"


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


def _warmup_jaccard_signature(intent: SeedWarmupIntent) -> frozenset[str]:
    """Feature tokens for MMR deduplication over synthetic warmup survivors."""

    v = operator_feature_vector_for_seed_intent(intent)
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


def _warmup_body_footprint(intent: SeedWarmupIntent) -> int:
    """Structural size heuristic for tie-breaking duplicate Jaccard signatures."""

    return (
        len(intent.select_cols or [])
        + len(intent.filters_param or [])
        + len(intent.having_param or [])
        + len(intent.cte_steps or [])
    )


def _warmup_submodular_atoms_for_row(
    ordered_intents: list[SeedWarmupIntent],
    pos: int,
) -> frozenset[str]:
    """Coverage atoms for greedy set-cover over synthetic execute indices."""

    intent = ordered_intents[pos]
    base = set(warmup_coverage_atoms_for_seed_intent(intent))
    base.add(f"body:{body_similarity_key(intent.to_runtime_intent())}")
    sid = intent.seed_index if intent.seed_index is not None else 0
    base.add(f"seed:{sid % 64}")
    return frozenset(base)


def _warmup_jaccard_similarity_frozen(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity on string-token sets."""

    if not a and not b:
        return 1.0
    union_n = len(a | b)
    if union_n == 0:
        return 0.0
    return len(a & b) / union_n


def _warmup_mmr_order(
    positions: list[int],
    ordered_intents: list[SeedWarmupIntent],
    lambda_mmr: float,
) -> list[int]:
    """Maximum marginal relevance ordering for diversified survivor ordering."""

    if len(positions) <= 1:
        return list(positions)
    sigs = {p: _warmup_jaccard_signature(ordered_intents[p]) for p in positions}
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
                max_sim = max(max_sim, _warmup_jaccard_similarity_frozen(sp, sigs[q]))
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


def _warmup_dedupe_jaccard_positions(
    positions: list[int],
    ordered_intents: list[SeedWarmupIntent],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Keep one survivor per Jaccard signature with smallest structural footprint."""

    by_sig: dict[frozenset[str], list[int]] = {}
    for pos in positions:
        sig = _warmup_jaccard_signature(ordered_intents[pos])
        by_sig.setdefault(sig, []).append(pos)
    kept: list[int] = []
    drop_records: list[dict[str, Any]] = []
    for plist in by_sig.values():
        best = min(
            plist,
            key=lambda p: (_warmup_body_footprint(ordered_intents[p]), p),
        )
        kept.append(best)
        for p in plist:
            if p == best:
                continue
            oi = ordered_intents[p]
            sk = _warmup_stratum_key(oi)
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
                    "detail": (
                        f"redundant_cover_representative stratum_id={sk!r} kept_index={best}"
                    ),
                }
            )
    return sorted(kept), drop_records


def build_anchor_lattice(
    synthetic_pending: list[tuple[SeedWarmupIntent, str]],
    schema: SchemaGraph,
    lattice_disk_by_sig: dict[str, list[str]],
) -> AnchorLattice:
    """
    Precompute shared NL anchors per lattice cell for synthetic warmup rows.

    Args:

        synthetic_pending: Successful synthetic rows as ``(intent, final_sql)`` pairs.

        schema: Active schema graph.

        lattice_disk_by_sig: On-disk anchors keyed by :func:`anchor_lattice_signature`.

    Returns:

        Populated :class:`AnchorLattice` for per-row NL attachment.
    """

    groups: dict[AnchorLatticeKey, list[tuple[SeedWarmupIntent, str]]] = {}
    for intent, sql in synthetic_pending:
        k = anchor_lattice_key_for_seed_intent(intent)
        groups.setdefault(k, []).append((intent, sql))
    cells: dict[AnchorLatticeKey, AnchorLatticeCell] = {}
    for key, rows in groups.items():
        lk = anchor_lattice_signature(key, schema.effective_structural_hash)
        disk_a = lattice_disk_by_sig.get(lk)
        if disk_a:
            cells[key] = AnchorLatticeCell(
                key=key,
                representative_intent_id=rows[0][0].intent_id,
                anchors=tuple(str(x) for x in disk_a),
            )
            continue

        def _distinct_rule_n(si: SeedWarmupIntent) -> tuple[int, str]:
            rt_loc = si.to_runtime_intent()
            phrases = generate_bulk_anchors(
                rt_loc,
                schema,
                SeedWarmupConfig.RULE_NLG_ANCHOR_COUNT,
            )
            norms = {normalize_question(str(p)) for p in phrases if p}
            return (len(norms), si.intent_id)

        best_intent, best_sql = max(rows, key=lambda t: _distinct_rule_n(t[0]))
        rt_best = best_intent.to_runtime_intent()
        rule_first = list(
            generate_bulk_anchors(
                rt_best,
                schema,
                SeedWarmupConfig.RULE_NLG_ANCHOR_COUNT,
            ),
        )
        raw_phrases: list[str] = list(rule_first)
        if _synthetic_warmup_llm_diversity_enabled(best_intent):
            triple = select_three_warmup_styles(
                best_intent.seed_index if best_intent.seed_index is not None else 0,
                best_intent.intent_id,
            )
            qresult = generate_question_from_sql(
                best_sql,
                schema,
                best_intent.tables or [],
                warmup_style_triple=triple,
            )
            if qresult is not None and qresult.get("is_realistic", False):
                raw_all = list(qresult.get("questions") or [])
                llm_phrases = raw_all[: SeedWarmupConfig.WARMUP_QUESTIONS_MAX]
                if not llm_phrases and qresult.get("question"):
                    llm_phrases = [str(qresult["question"])]
                raw_phrases = rule_first + llm_phrases
        cells[key] = AnchorLatticeCell(
            key=key,
            representative_intent_id=best_intent.intent_id,
            anchors=tuple(raw_phrases),
        )
    return AnchorLattice(cells=cells)


def _synthetic_warmup_llm_diversity_enabled(intent: SeedWarmupIntent) -> bool:
    """Deterministic subsample gate for synthetic ``generate_question_from_sql`` diversity passes."""

    div = SeedWarmupConfig.WARMUP_LLM_DIVERSITY_SUBSAMPLE_DIVISOR
    if div <= 0:
        return True
    key = (intent.intent_id or "x").encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:12], 16)
    return (h % div) == 0


def _warmup_submodular_cover_select(
    ordered_intents: list[SeedWarmupIntent],
    eligible_positions: list[int],
) -> tuple[set[int], set[int], dict[str, Any]]:
    """
    Return ``(sampled_positions, gold_cap_dropped_positions, sampling_detail)`` for *eligible_positions*.

    Sampling picks whole intents only; it never perturbs structural ``s*`` parameters inside an intent row.
    """

    cfg = SeedWarmupConfig
    drop_records: list[dict[str, Any]] = []
    detail: dict[str, Any] = {
        "skipped_due_to_low_volume": False,
        "strata": [],
        "policy_version": cfg.WARMUP_SAMPLING_POLICY_VERSION,
        "counts": {},
        "sampling_drops_by_code": {},
        "sampling_drop_records": drop_records,
    }
    e_list = list(eligible_positions)
    if len(e_list) <= cfg.WARMUP_KEEP_ALL_BELOW:
        detail["skipped_due_to_low_volume"] = True
        detail["counts"] = {
            "eligible_total": len(e_list),
            "gold_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") == "gold"),
            "synthetic_eligible": sum(1 for i in e_list if (ordered_intents[i].source or "gold") != "gold"),
            "target_cap": cfg.WARMUP_TARGET_CAP,
            "gold_kept": len(e_list),
            "gold_dropped": 0,
            "synthetic_budget": 0,
            "synthetic_kept": 0,
            "synthetic_dropped": 0,
        }
        detail["sampling_drops_by_code"] = {}
        return set(e_list), set(), detail

    g_positions = [i for i in e_list if (ordered_intents[i].source or "gold") == "gold"]
    s_positions = [i for i in e_list if i not in set(g_positions)]
    target = cfg.WARMUP_TARGET_CAP
    min_gold = 0
    if g_positions:
        pool_n = len(e_list)
        gr = len(g_positions) / max(pool_n, 1)
        min_gold = min(
            len(g_positions),
            max(
                int(math.ceil(cfg.WARMUP_MIN_GOLD_FRACTION * target)),
                int(math.ceil(gr * target)),
            ),
        )
    gold_stratum: dict[str, list[int]] = {}
    for pos in g_positions:
        gsk = _warmup_stratum_key(ordered_intents[pos])
        gold_stratum.setdefault(gsk, []).append(pos)
    counts_g = {k: len(v) for k, v in gold_stratum.items()}
    g_kept: set[int] = set()
    if min_gold > 0 and counts_g:
        g_floor = cfg.WARMUP_STRATUM_MIN if min_gold >= len(counts_g) * cfg.WARMUP_STRATUM_MIN else 0
        gold_budget = min(min_gold, len(g_positions))
        quotas_g = _allocate_stratum_quotas(counts_g, gold_budget, g_floor)
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
        sk = _warmup_stratum_key(ordered_intents[pos])
        stratum_to_positions.setdefault(sk, []).append(pos)
    syn_kept: set[int] = set()
    if budget == 0 and s_positions:
        for pos in s_positions:
            oi = ordered_intents[pos]
            sk = _warmup_stratum_key(oi)
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
            atoms_rows = [_warmup_submodular_atoms_for_row(ordered_intents, p) for p in rest_syn]
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
        mmr_syn = _warmup_mmr_order(syn_only_sorted, ordered_intents, cfg.WARMUP_MMR_LAMBDA)
        ded_syn, red_drops = _warmup_dedupe_jaccard_positions(mmr_syn, ordered_intents)
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
    """
    Create a `Template` from a successful seed warmup execution result.

    Args:

        result: Warmup row including runtime intent and final SQL.

        schema: Active schema graph.

        next_id: Template id counter when *store* is omitted (tests and one-off callers).

        dialect: Optional dialect for ``execution_sql``.

        source: Stored ``Template.source``.

        trust_level: Stored ``Template.trust_level``.

        seed_warmup_intent: Optional gold provenance for multi-row ``ValueHistory``.

        question_phrases: Optional LLM question variants.

        store: Mutable store dict with ``next_id``; ephemeral when omitted.

        templates: Accepted-template dict to insert into; ephemeral when omitted.

        structural_match_templates: Optional explicit merge list; defaults to sorted *templates* values.

    Returns:

        New or merged ``Template``, or ``None`` when the result is not template-ready.
    """
    if not result.success or not result.sql:
        return None

    raw_intent = result.intent
    if isinstance(raw_intent, SeedWarmupIntent):
        intent = raw_intent.to_runtime_intent()
    else:
        intent = raw_intent
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
    form_storage = QuestionFormStorage(
        corrected=first_q,
        normalized_optional=norm_opt,
    )

    vh: ValueHistory | None = None
    if seed_warmup_intent and (seed_warmup_intent.seed_prompt_original or seed_warmup_intent.seed_prompt_normalized):
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
        structural_list = sorted(templates_use.values(), key=lambda x: x.id)

    tmpl = insert_template(
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


def _build_value_domains(
    schema: SchemaGraph,
) -> dict[str, ValueDomain]:
    """
    Build `ValueDomain` objects from schema column metadata.

    Args:

        schema: Description.

    Returns:

        Return value.
    """
    domains: dict[str, ValueDomain] = {}
    for table_name, table_meta in schema.tables.items():
        for col_name, col_meta in table_meta.columns.items():
            col_key = f"{table_name}.{col_name}"
            domains[col_key] = ValueDomain(
                values=col_meta.top_k_values or [],
                min_val=col_meta.min_val,
                max_val=col_meta.max_val,
                data_type=col_meta.data_type or None,
            )
    return domains


def run_seed_warmup_execution(
    intents: list[SeedWarmupIntent],
    schema: SchemaGraph,
    dialect: Any,
    next_id: int,
    join_cache: dict[frozenset[str], JoinCacheEntry] | None = None,
    join_resolver_intent_index: dict[str, SeedWarmupIntent] | None = None,
    *,
    store_instance_keys: set[str] | None = None,
    accepted_templates: dict[str, Template] | None = None,
    warmup_run_mode: Literal["full", "preflight"] = "full",
    warmup_cache: SeedWarmupCacheSession | None = None,
    warmup_report_version: int = 1,
    warmup_dry_run_session: bool = False,
    warmup_lattice_root: str | None = None,
) -> tuple[list[SeedWarmupResult], list[Template], int, dict[str, Any]]:
    """
    Execute SQL for each intent, optionally stratify successes, then run question LLM only on the sample.

    When *warmup_run_mode* is ``preflight``, stops after execute (no question LLM or templates).

    When *warmup_lattice_root* is set and *warmup_run_mode* is ``full``, synthetic warmup reuses
    cached NL anchors per lattice cell under that directory and refreshes the JSON sidecar after the run.
    """

    if join_cache is None:
        join_cache = {}

    store_keys = store_instance_keys or set()
    tmpl_for_q = accepted_templates or {}

    id_to_intent = (
        join_resolver_intent_index
        if join_resolver_intent_index is not None
        else {i.intent_id: i for i in intents if getattr(i, "intent_id", "")}
    )

    value_domains = _build_value_domains(schema)
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

    log(f"run_seed_warmup_execution: processing {len(intents)} intents")
    log("[P6] Warmup union pre-align (4.3 only), validate, execute, then sampling")

    ordered_intents = sorted(intents, key=_seed_warmup_intent_sort_key)
    cap_drop: list[SeedWarmupIntent] = []
    cap_n = SeedWarmupConfig.MAX_WARMUP_EXECUTE_UNITS
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
        ]
    ] = []

    for idx, intent in enumerate(ordered_intents):
        ifp = warmup_intent_fingerprint(intent)
        runtime = intent.to_runtime_intent()
        runtime = apply_deterministic_repairs(
            runtime,
            schema,
            intent.natural_language or intent.intent_id or "",
        )
        result = SeedWarmupResult(runtime, "")

        runtime, qual_msgs = check_qualified_refs_exist(runtime, schema)
        if qual_msgs:
            result.error = qual_msgs[0]
            result.failure_code = "warmup_qualified_refs"
            results.append(result)
            fail_count += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="warmup_qualified_refs",
                err=result.error,
            )
            continue

        lit_rt, pp_issues = apply_runtime_post_processing_lite(
            runtime,
            schema,
            question_fallback=intent.intent_id or "",
        )
        if lit_rt is None:
            result.error = "warmup_post_processing_lite_failed"
            result.failure_code = "warmup_post_processing_lite_failed"
            results.append(result)
            fail_count += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="warmup_post_processing_lite_failed",
                err=result.error,
            )
            continue
        if any((i.severity or "").lower() == "error" for i in pp_issues):
            result.error = f"warmup_post_processing_lite_failed: {pp_issues[0].message}"
            result.failure_code = "warmup_post_processing_lite_failed"
            results.append(result)
            fail_count += 1
            _wu_record(
                lit_rt,
                ok=False,
                final_sql=None,
                fc="warmup_post_processing_lite_failed",
                err=result.error,
            )
            continue
        runtime = lit_rt

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
            if warmup_cache is None:
                return
            bk = body_similarity_key(rt)
            jkr = join_path_key_runtime(rt)
            pack = _warmup_pack_execute(
                rt,
                ok=ok,
                final_sql=final_sql,
                failure_code=fc,
                error=err,
                body_key=bk,
                join_path_key=jkr,
                template_instance_key=tik,
            )
            warmup_cache.write_work_unit(
                unit_fp,
                unit_intent,
                pack,
                report_version=warmup_report_version,
                is_preflight=warmup_dry_run_session,
            )

        sem_msgs = curated_warmup_semantic_issues(runtime, schema)
        if sem_msgs:
            result.error = f"warmup_semantic_precheck: {sem_msgs[0]}"
            result.failure_code = "warmup_semantic_precheck"
            results.append(result)
            fail_count += 1
            semantic_precheck_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="warmup_semantic_precheck",
                err=result.error,
            )
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
                res_hit.preflight_execute_ok = bool(er_hit.get("ok"))
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
                pending_success.append((idx, intent, rt_hit, fs_sql, [], all_pv_hit, res_hit))
                results.append(res_hit)
                continue

        try:
            reuse_entry = _ambiguous_join_reuse_from_parent(intent, join_cache, id_to_intent)
            join_id, join_sig, candidates = resolve_joins_for_table_set(
                intent.tables or [],
                schema,
                intent.intent_id,
                join_cache,
                ambiguous_reuse_entry=reuse_entry,
            )
        except Exception as e:
            result.error = f"join_resolution_failed: {e}"
            result.failure_code = "join_resolution_failed"
            results.append(result)
            fail_count += 1
            join_resolution_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="join_resolution_failed",
                err=result.error,
            )
            continue

        try:
            runtime = apply_default_structural_values(runtime)
            runtime = prune_unused_cte_steps(runtime)

            det_sql = build_deterministic_sql(runtime, None, schema, dialect)
            if join_id != "J00" and join_sig:
                det_sql = inject_join_into_deterministic_sql(
                    det_sql,
                    [join_sig],
                    schema=schema,
                    dialect=dialect,
                )
            runtime.sql_param = det_sql
            runtime.chosen_join_candidate_id = join_id
            runtime.chosen_join_path_signature = join_sig
            syn_drop = _warmup_synthetic_store_path_blocks(intent, runtime, tmpl_for_q)
            if syn_drop:
                result.failure_code = syn_drop
                result.error = syn_drop
                results.append(result)
                fail_count += 1
                if syn_drop == "warmup_path41_not_allowed":
                    warmup_path41_drop += 1
                else:
                    warmup_path42_drop += 1
                _wu_record(
                    runtime,
                    ok=False,
                    final_sql=None,
                    fc=syn_drop,
                    err=syn_drop,
                    tik="",
                )
                continue
        except Exception as e:
            result.error = f"sql_build_failed: {e}"
            result.failure_code = "sql_build_failed"
            results.append(result)
            fail_count += 1
            sql_build_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="sql_build_failed",
                err=result.error,
            )
            continue

        instantiated = instantiate_intent(intent, value_domains)
        if instantiated is None:
            result.error = "instantiation_failed"
            result.failure_code = "instantiation_failed"
            results.append(result)
            fail_count += 1
            instantiation_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="instantiation_failed",
                err=result.error,
            )
            continue

        all_params = dict(instantiated.param_values)
        all_params.update(runtime.param_values or {})
        try:
            runtime.sql_param = det_sql
            final_sql = finalize_substitute_sql(
                runtime,
                structural_defaults_src=None,
                params=all_params,
            )
        except Exception as e:
            result.error = f"substitution_failed: {e}"
            result.failure_code = "substitution_failed"
            results.append(result)
            fail_count += 1
            substitution_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="substitution_failed",
                err=result.error,
            )
            continue

        if not final_sql or not final_sql.strip():
            result.error = "empty_sql_after_substitution"
            result.failure_code = "empty_sql_after_substitution"
            results.append(result)
            fail_count += 1
            empty_sql_failed += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=None,
                fc="empty_sql_after_substitution",
                err=result.error,
            )
            continue

        post_msgs_pb = curated_warmup_post_binding_issues(runtime, schema, final_sql)
        if post_msgs_pb:
            result.error = f"warmup_post_binding_semantics: {post_msgs_pb[0]}"
            result.failure_code = "warmup_post_binding_semantics"
            results.append(result)
            fail_count += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=final_sql,
                fc="warmup_post_binding_semantics",
                err=result.error,
            )
            continue

        try:
            ok, err, vcat, vdiags = validate_sql(
                dialect,
                final_sql,
                bind_params_for_sql(final_sql, runtime.param_values),
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
                        det_sql = inject_join_into_deterministic_sql(
                            det_sql,
                            [join_sig],
                            schema=schema,
                            dialect=dialect,
                        )
                    runtime.sql_param = det_sql
                    runtime.chosen_join_candidate_id = join_id
                    runtime.chosen_join_path_signature = join_sig
                    instantiated2 = instantiate_intent(intent, value_domains)
                    if instantiated2 is None:
                        break
                    all_params2 = dict(instantiated2.param_values)
                    all_params2.update(runtime.param_values or {})
                    final_sql = finalize_substitute_sql(
                        runtime,
                        structural_defaults_src=None,
                        params=all_params2,
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
                err,
                failure_category=vcat.value if vcat is not None else None,
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
                    exec_sql,
                    all_params,
                    schema=schema,
                    intent=runtime,
                )
                if not ok_ex:
                    result.error = f"explain_failed: {err_ex}"
                    result.failure_code = "explain_failed"
                    results.append(result)
                    validation_drop += 1
                    fail_count += 1
                    _wu_record(
                        runtime,
                        ok=False,
                        final_sql=final_sql,
                        fc="explain_failed",
                        err=result.error,
                    )
                    continue
            rows = dialect.execute(exec_sql)
        except Exception as e:
            result.error = f"execution_failed: {e}"
            result.failure_code = "execution_failed"
            results.append(result)
            validation_drop += 1
            fail_count += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=final_sql,
                fc="execution_failed",
                err=result.error,
            )
            continue

        sfp = compute_sql_fp(final_sql, sqlglot_dialect=active_sqlglot_dialect())
        bk = body_similarity_key(runtime)
        jkr = join_path_key_runtime(runtime)
        tik = template_instance_key_from_parts(bk, jkr, sfp)
        if tik in store_keys:
            result.failure_code = "template_instance_exists"
            result.error = "template_instance_exists"
            results.append(result)
            fail_count += 1
            template_instance_exists_count += 1
            _wu_record(
                runtime,
                ok=False,
                final_sql=final_sql,
                fc="template_instance_exists",
                err=result.error,
                tik=tik,
            )
            continue

        runtime.param_values = all_params
        result.intent = runtime
        result.sql = final_sql
        result.rows = rows
        result.preflight_execute_ok = True
        result.success = False
        _wu_record(
            runtime,
            ok=True,
            final_sql=final_sql,
            fc=None,
            err=None,
            tik=tik,
        )
        pending_success.append((idx, intent, runtime, final_sql, rows, all_params, result))
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
    sampled, gold_dropped_pos, sampling_detail = _warmup_submodular_cover_select(ordered_intents, execute_positions)
    sampling_drop_by_index: dict[int, dict[str, Any]] = {
        int(r["intent_index"]): r
        for r in sampling_detail.get("sampling_drop_records", [])
        if isinstance(r, dict) and r.get("intent_index") is not None
    }

    synthetic_pairs_for_lattice: list[tuple[SeedWarmupIntent, str]] = []
    for idx, intent, _runtime, final_sql, _rows, _all_params, _res in pending_success:
        if idx not in sampled:
            continue
        if (intent.source or "gold") == "gold":
            continue
        synthetic_pairs_for_lattice.append((intent, final_sql))

    seen_nl_norm: set[str] = set()
    lattice_path: str | None = None
    lattice_disk: dict[str, list[str]] = {}
    if warmup_lattice_root:
        lattice_path = _warmup_anchor_lattice_json_path(warmup_lattice_root, schema)
        lattice_disk = _load_warmup_anchor_lattice(lattice_path)
    anchor_lattice = build_anchor_lattice(synthetic_pairs_for_lattice, schema, lattice_disk)
    lattice_runtime_persist: dict[str, list[str]] = {
        anchor_lattice_signature(k, schema.effective_structural_hash): list(cell.anchors)
        for k, cell in anchor_lattice.cells.items()
    }
    pending_by_idx = {t[0]: t for t in pending_success}
    fillback_batches = 0

    def _warmup_question_llm_branch(
        idx: int,
        intent: SeedWarmupIntent,
        final_sql: str,
        res: SeedWarmupResult,
    ) -> bool:
        nonlocal success_count, fail_count, question_generation_failed, realism_drop, all_questions_dropped
        nonlocal sampled_work_unit_ids, seen_nl_norm, new_templates_collected

        if warmup_cache is not None:
            sw = warmup_cache.mark_sampled_in(warmup_intent_fingerprint(intent))
            if sw:
                sampled_work_unit_ids.append(sw)

        seed_idx = intent.seed_index if intent.seed_index is not None else 0
        judge_ctx = schema_context_enriched_lines_for_tables(schema, intent.tables or [])
        origin_sql_history = (intent.source or "") == "sql_history"
        origin_gold = (intent.source or "gold") == "gold"
        lk = anchor_lattice_signature(
            anchor_lattice_key_for_seed_intent(intent),
            schema.effective_structural_hash,
        )
        lattice_hit = False

        if origin_sql_history:
            triple = select_three_warmup_styles(seed_idx, intent.intent_id)
            qresult = generate_question_from_sql(
                final_sql,
                schema,
                intent.tables or [],
                warmup_style_triple=triple,
                intent_source="sql_history",
            )
            if qresult is None:
                res.error = "question_generation_failed"
                res.failure_code = "question_generation_failed"
                fail_count += 1
                question_generation_failed += 1
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        warmup_intent_fingerprint(intent),
                        {"questions": [], "failure_code": "question_generation_failed"},
                        ok=False,
                    )
                return False
            llm_qs = list(qresult.get("questions") or [])
            anchor = (llm_qs[0] if llm_qs else "").strip()
            if not anchor:
                res.error = "question_generation_failed"
                res.failure_code = "question_generation_failed"
                fail_count += 1
                question_generation_failed += 1
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        warmup_intent_fingerprint(intent),
                        {"questions": [], "failure_code": "question_generation_failed"},
                        ok=False,
                    )
                return False
            extras = generate_paraphrases_of_seed_question(
                anchor,
                schema,
                intent.tables or [],
                style_pair=(triple[1], triple[2]),
            )
            budget = int(WARMUP_PARAPHRASE_COUNT_FROM_SQL)
            raw_phrases = [anchor]
            for q in llm_qs[1:]:
                if len(raw_phrases) >= budget:
                    break
                if q and str(q).strip():
                    raw_phrases.append(str(q).strip())
            if extras:
                for x in extras:
                    if len(raw_phrases) >= budget:
                        break
                    raw_phrases.append(x)
            if len(raw_phrases) > SeedWarmupConfig.WARMUP_QUESTIONS_MAX:
                raw_phrases = raw_phrases[: SeedWarmupConfig.WARMUP_QUESTIONS_MAX]
        elif origin_gold:
            anchor = (
                (intent.seed_prompt_original or "").strip()
                or (intent.seed_prompt_normalized or "").strip()
                or (intent.natural_language or "").strip()
                or (intent.question or "").strip()
            )
            if not anchor:
                res.error = "question_generation_failed"
                res.failure_code = "question_generation_failed"
                fail_count += 1
                question_generation_failed += 1
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        warmup_intent_fingerprint(intent),
                        {"questions": [], "failure_code": "question_generation_failed"},
                        ok=False,
                    )
                return False

            triple = select_three_warmup_styles(seed_idx, intent.intent_id)
            extras = generate_paraphrases_of_seed_question(
                anchor,
                schema,
                intent.tables or [],
                style_pair=(triple[1], triple[2]),
            )
            raw_phrases = [anchor]
            if extras:
                for x in extras:
                    if len(raw_phrases) >= SeedWarmupConfig.WARMUP_QUESTIONS_MAX:
                        break
                    raw_phrases.append(x)
        else:
            akey = anchor_lattice_key_for_seed_intent(intent)
            cell = anchor_lattice.cells.get(akey)
            if cell is None or not cell.anchors:
                res.error = "question_generation_failed"
                res.failure_code = "question_generation_failed"
                fail_count += 1
                question_generation_failed += 1
                if warmup_cache is not None:
                    warmup_cache.record_question_llm(
                        warmup_intent_fingerprint(intent),
                        {"questions": [], "failure_code": "question_generation_failed"},
                        ok=False,
                    )
                return False
            raw_phrases = list(cell.anchors)
            lattice_hit = lk in lattice_disk and bool(lattice_disk.get(lk))

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
            if warmup_cache is not None:
                warmup_cache.record_question_llm(
                    warmup_intent_fingerprint(intent),
                    {
                        "questions": raw_phrases,
                        "filtered_empty": True,
                        "failure_code": "all_questions_collided_or_empty",
                    },
                    ok=False,
                )
            return False

        pick = select_best_question_via_judge(
            final_sql,
            judge_ctx,
            filtered,
        )
        res.question = filtered[pick]
        res.questions = list(filtered)
        res.confidence = 1.0
        res.success = True

        if warmup_cache is not None:
            warmup_cache.record_question_llm(
                warmup_intent_fingerprint(intent),
                {
                    "questions": filtered,
                    "raw_phrases_truncated": raw_phrases,
                    "is_realistic": True,
                },
                ok=True,
            )

        tmpl = _create_template_from_result(
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

    for idx, intent, _runtime, final_sql, _rows, _all_params, res in pending_success:
        if idx not in sampled:
            not_sampled_after_execute += 1
            rec = sampling_drop_by_index.get(idx)
            if rec:
                fc = str(rec.get("failure_code") or "stratum_quota_exceeded")
                res.failure_code = fc
                res.error = str(rec.get("detail") or fc)
                drop_audit.append(rec)
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
                        "stratum_id": (_warmup_stratum_key(oi) if origin == "synthetic" else None),
                        "detail": fc_fallback,
                    }
                )
            fail_count += 1
            continue
        if warmup_run_mode == "preflight":
            res.failure_code = "preflight_skipped"
            res.error = "preflight_skipped"
            res.preflight_execute_ok = True
            continue

        _warmup_question_llm_branch(idx, intent, final_sql, res)

    if warmup_run_mode != "preflight":
        unsampled_ordered = [i for i in execute_positions if i not in sampled]
        for _ in range(SeedWarmupConfig.WARMUP_MAX_FILLBACK_ROUNDS):
            if success_count >= SeedWarmupConfig.WARMUP_TARGET_CAP:
                break
            shortfall = SeedWarmupConfig.WARMUP_TARGET_CAP - success_count
            if shortfall <= 0 or not unsampled_ordered:
                break
            batch = unsampled_ordered[:shortfall]
            unsampled_ordered = unsampled_ordered[shortfall:]
            fillback_batches += 1
            for idx in batch:
                row = pending_by_idx.get(idx)
                if row is None:
                    continue
                _i, intent, _rt, final_sql, _rows, _all_params, res = row
                fail_count -= 1
                res.failure_code = None
                res.error = None
                res.success = False
                res.question = ""
                _warmup_question_llm_branch(idx, intent, final_sql, res)

    log(
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
        "dry_run_execute_ok_count": len(pending_success),
        "cache_execute_hits": warmup_cache.execute_hits if warmup_cache else 0,
        "warmup_drop_audit": drop_audit,
        "warmup_path41_not_allowed": warmup_path41_drop,
        "warmup_path42_not_allowed": warmup_path42_drop,
        "warmup_touched_work_unit_ids": (list(warmup_cache.touched_work_unit_ids) if warmup_cache else []),
        "warmup_sampled_work_unit_ids": sampled_work_unit_ids,
        "warmup_fillback_batches": fillback_batches,
    }
    if warmup_run_mode != "preflight" and lattice_path is not None:
        _save_warmup_anchor_lattice(lattice_path, schema, lattice_runtime_persist)

    return results, new_templates_collected, int(warmup_store["next_id"]), warmup_funnel


def seed_warmup_drops_jsonl_path_for_report(report_filepath: str) -> str | None:
    """Return sibling drops JSONL path for a seed warmup or preflight report filename."""

    d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
    if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
        ver = base[len("seed_warmup_report_v") : -len(".json")]
        return os.path.join(d, f"seed_warmup_drops_v{ver}.jsonl")
    if base.startswith("warmup_preflight_report_v") and base.endswith(".json"):
        ver = base[len("warmup_preflight_report_v") : -len(".json")]
        return os.path.join(d, f"warmup_preflight_drops_v{ver}.jsonl")
    return None


def seed_warmup_drops_detail_jsonl_path_for_report(report_filepath: str) -> str | None:
    """Return path for per-row sampling ``drops_detail`` JSONL next to the report file."""

    d, base = os.path.dirname(report_filepath), os.path.basename(report_filepath)
    if base.startswith("seed_warmup_report_v") and base.endswith(".json"):
        ver = base[len("seed_warmup_report_v") : -len(".json")]
        return os.path.join(d, f"seed_warmup_drops_detail_v{ver}.jsonl")
    if base.startswith("warmup_preflight_report_v") and base.endswith(".json"):
        ver = base[len("warmup_preflight_report_v") : -len(".json")]
        return os.path.join(d, f"warmup_preflight_drops_detail_v{ver}.jsonl")
    return None


def save_seed_warmup_report(
    results: list[SeedWarmupResult],
    filepath: str,
    funnel: dict[str, Any] | None = None,
) -> None:
    """
    Save compact aggregate seed warmup metrics and optional funnel counters to a JSON file.

    Args:

        results: Per-intent warmup outcomes used for aggregate and category counts.

        filepath: Output JSON path.

        funnel: Optional extra scalar fields (gold counters, dedupe counts, phase drops).

    Returns:

        None.
    """
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
    sampling_for_report = {k: v for k, v in ws.items() if k != "sampling_drop_records"} if isinstance(ws, dict) else {}
    known_pipeline_failure_histogram = {k: v for k, v in failure_histogram.items() if k in SEED_WARMUP_FAILURE_CODES}
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
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS)
    drops_path = seed_warmup_drops_jsonl_path_for_report(filepath)
    if drops_path:
        audit = fn.get("warmup_drop_audit") or []
        with open(drops_path, "w", encoding="utf-8") as jf:
            for row in audit:
                if isinstance(row, dict):
                    jf.write(
                        json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n",
                    )
        log(f"save_seed_warmup_report: wrote {len(audit)} sampling drop rows to {drops_path}")
    detail_path = seed_warmup_drops_detail_jsonl_path_for_report(filepath)
    if detail_path and isinstance(ws, dict):
        detail_rows = list(ws.get("sampling_drop_records") or [])
        with open(detail_path, "w", encoding="utf-8") as df:
            for row in detail_rows:
                if isinstance(row, dict):
                    df.write(
                        json.dumps(row, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS) + "\n",
                    )
        log(f"save_seed_warmup_report: wrote {len(detail_rows)} sampling drops_detail rows to {detail_path}")
    log(f"save_seed_warmup_report: wrote aggregate report for {len(results)} results to {filepath}")
