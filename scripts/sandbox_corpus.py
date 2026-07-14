"""Shared helpers for sandbox corpus assembly, recording, validation, and packing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aetherdialect._live_testing
import aetherdialect._llm_provider
import aetherdialect._main_execution
import aetherdialect._pipeline
import aetherdialect._sandbox
import aetherdialect._schema_overrides
from aetherdialect._config import (
    ConfigError,
    DuckDBRuntimeConfig,
    EngineConfig,
    PolicyConfig,
    llm_credentials_configured,
)
from aetherdialect._constants import (
    INTENT_COMPOSE_SYSTEM,
    INTENT_GROUND_SYSTEM,
    INTENT_INTERPRET_SYSTEM,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_QUESTION_TIERS,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
)
from aetherdialect._contracts_core import TemplateMatch
from aetherdialect._core_utils import (
    StepResult,
    append_failure_trace,
    build_session_step_trace,
    pipeline_capture,
    run_with_pipeline_capture,
)
from aetherdialect._llm_provider import (
    reset_mock_provider,
)
from aetherdialect._main_execution import (
    _configure_llm_from_environment,
    compute_engine_storage_dir,
)
from aetherdialect._sandbox import (
    _sandbox_doctor_verbose,
    assert_sandbox_complete,
    check_sandbox_faithfulness,
    question_ok,
    validate_sandbox_corpus,
)
from aetherdialect._utils import generate_paraphrases_of_seed_question, normalize_question
from aetherdialect.aetherdialect import AetherEngine

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
_BUILD_VERBOSE = True
_SMOKE_BUILD = False
SMOKE_TOUR_QUESTION = "How many rentals happened in 2025?"
SMOKE_FLAKY_QUESTION = "Which games support English?"
SCRIPTS = REPO / "scripts"
DATA = SCRIPTS / "data"
STAGING = SCRIPTS / "sandbox_staging"
STAGING_ZIP = SCRIPTS / "sandbox_staging.zip"
OUT_ZIP = REPO / "src" / "aetherdialect" / "sandbox" / "data.zip"
ENV_FILE = REPO / "env.env"
QUESTIONS_SOURCE = DATA / "sandbox_questions.txt"
MIGRATION_DEMO_SOURCE = DATA / "sandbox_migration_demo.json"
OVERRIDES_DEMO_SOURCE = DATA / "sandbox_overrides_demo.json"
FIXTURES_PATH = STAGING / "fixtures" / "rental_shop_mock.json"
BUILD_FINGERPRINT_PATH = STAGING / "sandbox_build_fingerprint.json"
RECORDING_MAX_ATTEMPTS = 2
RECORDING_RESULTS_PATH = SCRIPTS / "logs" / "sandbox_results.txt"
SANDBOX_REPLAY_TRACE_PATH = SCRIPTS / "logs" / "sandbox_replay_trace.txt"
RECORDING_MANIFEST_PATH = STAGING / "recording_manifest.json"
EXPECTATIONS_SOURCE = DATA / "sandbox_expectations.json"
SCENARIOS_SOURCE = DATA / "sandbox_scenarios.json"
HANDCRAFTED_FIXTURES_SOURCE = DATA / "sandbox_handcrafted_fixtures.json"
SPACE_CATALOG_NOTES_SOURCE = DATA / "sandbox_space_catalog_notes.txt"
VALIDATE_OUT = SCRIPTS / "logs" / "validate_out.txt"
VALIDATE_TRACE_OUT = SCRIPTS / "logs" / "validate_trace.txt"
SKIP_ZIP_NAMES = frozenset(
    {
        "rental_shop_post_migration.sql",
        "rental_shop_mock.pre_repair.json",
        "sandbox_build_fingerprint.json",
    },
)

BASELINE_OWNER_SUBDIR = "owner"
BASELINE_CONSUMER_SUBDIR = "consumer"
LLM_PATCH_MODULES = (
    "aetherdialect._intent_process",
    "aetherdialect._pipeline",
    "aetherdialect._schema_catalog",
    "aetherdialect._templates",
    "aetherdialect._utils",
    "aetherdialect._schema_overrides",
    "aetherdialect._live_testing",
    "aetherdialect._main_execution",
)


def slot_id_for(slot: RecordingSlot) -> str:
    if slot.kind == "feedback":
        return f"feedback:{slot.label}"
    mode_part = slot.mode or "default"
    return f"{slot.preset}:{mode_part}:{slot.tier}:{slot.label}"


def _load_recording_manifest(path: Path | None = None) -> dict[str, object]:
    target = path
    if target is None:
        target = STAGING / "recording_manifest.json"
    if not target.is_file():
        return {"slots": []}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {"slots": []}


def write_recording_manifest(rows: list[dict[str, object]], *, staging_dir: Path | None = None) -> None:
    target = (staging_dir or STAGING) / "recording_manifest.json"
    write_text_atomic(target, json.dumps({"slots": rows}, ensure_ascii=False, indent=2) + "\n")


def _load_manifest_rows(path: Path | None = None) -> list[dict[str, object]]:
    payload = _load_recording_manifest(path)
    rows = payload.get("slots")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _upsert_manifest_row(
    rows: list[dict[str, object]],
    entry: dict[str, object],
) -> None:
    slot_id = str(entry.get("slot_id", ""))
    if slot_id:
        for idx, row in enumerate(rows):
            if str(row.get("slot_id", "")) == slot_id:
                rows[idx] = entry
                return
    rows.append(entry)


def committed_slot_ids(manifest: dict[str, object]) -> set[str]:
    rows = manifest.get("slots")
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row.get("committed") and row.get("slot_id"):
            out.add(str(row["slot_id"]))
    return out


def _check_slot_recording(
    step: object,
    slot: RecordingSlot,
    *,
    profile: str | None = None,
    tier: str | None = None,
) -> tuple[bool, str]:
    sid = slot_id_for(slot)
    check_profile = profile if profile is not None else slot.preset
    check_tier = tier if tier is not None else slot.tier
    if question_ok(
        step,
        slot.label,
        slot_id=sid,
        profile=check_profile,
        tier=check_tier,
    ):
        return True, ""
    detail = check_sandbox_faithfulness(
        step,
        slot.label,
        slot_id=sid,
        profile=check_profile,
        tier=check_tier,
    )
    if detail:
        return False, detail
    err = str(getattr(step, "error", "") or "").strip()
    return False, err or "expectation not met"


def _mock_verify_targets_for_slot(
    slot: RecordingSlot,
) -> list[tuple[str, bool, str | None, bool, str, str]]:
    """Return mock-verify runs that must pass before a slot commits (mirrors pack validate)."""
    preset, restricted_consumer, mode, apply_overrides = _mock_preset_for_slot(slot)
    targets: list[tuple[str, bool, str | None, bool, str, str]] = [
        (preset, restricted_consumer, mode, apply_overrides, slot.preset, slot.tier),
    ]
    if (
        slot.kind == "question"
        and slot.tier == "questions"
        and slot.preset == "owner_writer"
        and slot.mode in (None, "writer")
    ):
        targets.append(
            ("consumer_reader", False, "reader", False, "consumer_reader", "consumer_reader"),
        )
    return targets


def baseline_dir_for_preset(bundle_dir: Path, preset: str) -> Path:
    """Return the preset-scoped baseline directory under *bundle_dir*."""
    del preset
    root = bundle_dir / "artifacts_baseline"
    owner = root / BASELINE_OWNER_SUBDIR
    if owner.is_dir() and (owner / "schema_graph.json.gz").is_file():
        return owner
    if (root / "schema_graph.json.gz").is_file():
        return root
    scoped = root / BASELINE_CONSUMER_SUBDIR
    if scoped.is_dir() and (scoped / "schema_graph.json.gz").is_file():
        return scoped
    return scoped if scoped.is_dir() else root


_BASELINE_CACHE_FILES = (
    "schema_graph.json.gz",
    "artifact_manifest.json",
    "schema_context.json",
)


def _copy_baseline_cache_files(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in _BASELINE_CACHE_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for sidecar in source.glob("schema_context*.json"):
        if sidecar.name == "schema_context.json":
            continue
        dst = dest / sidecar.name
        if not dst.is_file():
            shutil.copy2(sidecar, dst)


def _reset_sandbox_duckdb_runtime() -> None:
    DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
    DuckDBRuntimeConfig.SCHEMA = "main"


def _sandbox_memory_engine_dir(artifacts_dir: str) -> Path:
    saved_path = DuckDBRuntimeConfig.DATABASE_PATH
    saved_schema = DuckDBRuntimeConfig.SCHEMA
    try:
        _reset_sandbox_duckdb_runtime()
        return Path(compute_engine_storage_dir(artifacts_dir, "duckdb"))
    finally:
        DuckDBRuntimeConfig.DATABASE_PATH = saved_path
        DuckDBRuntimeConfig.SCHEMA = saved_schema


def _seed_engine_baseline(*, artifacts_dir: str, bundle_dir: Path, preset: str) -> None:
    baseline = baseline_dir_for_preset(bundle_dir, preset)
    graph_path = baseline / "schema_graph.json.gz"
    if not baseline.is_dir():
        verbose_message(f"[build] seed baseline skip: missing baseline dir {baseline}")
        return
    if not graph_path.is_file():
        verbose_message(f"[build] seed baseline skip: no schema_graph.json.gz under {baseline}")
        return
    engine_dir = _sandbox_memory_engine_dir(artifacts_dir)
    if (engine_dir / "schema_graph.json.gz").is_file():
        verbose_message(f"[build] seed baseline skip: cache already present at {engine_dir}")
        return
    _copy_baseline_cache_files(baseline, engine_dir)
    copied = [name for name in _BASELINE_CACHE_FILES if (engine_dir / name).is_file()]
    verbose_message(
        f"[build] seed baseline: copied {baseline} -> {engine_dir} ({', '.join(copied) or 'no files'})",
    )


def _recording_error_retryable(detail: str) -> bool:
    lowered = detail.lower()
    if "no mock fixture" in lowered:
        return False
    if "mock replay" in lowered:
        return False
    if "add an entry to the fixture corpus" in lowered:
        return False
    if "no openai/azure openai api key configured" in lowered:
        return False
    if "env file not found" in lowered:
        return False
    if "llm is not configured" in lowered:
        return False
    if isinstance(detail, str) and detail.startswith("ConfigError"):
        return False
    return True


def _short_retry_reason(detail: str, *, mock: bool = False) -> str:
    text = detail.strip()
    if mock and "add an entry to the fixture corpus" in text.lower():
        text = text.split("\n", 1)[0].strip()
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _begin_eval_results(path: Path) -> None:
    import live_tests.conftest as lt_conftest

    path.parent.mkdir(parents=True, exist_ok=True)
    lt_conftest._RESULTS_FILE = path
    lt_conftest._results_trace_pending_sep = False
    lt_conftest._step_results.clear()
    lt_conftest._NODEID_SCENARIO_IDS.clear()
    lt_conftest._clear_results_file()


def _reset_replay_trace() -> None:
    SANDBOX_REPLAY_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SANDBOX_REPLAY_TRACE_PATH.write_text("", encoding="utf-8")


def _format_trace_rows(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["    (no LLM calls captured)"]
    return [
        f"    #{idx} task={row.get('task', '')!r} stage={row.get('system_id', '')} "
        f"user_sha={row.get('user_sha', '')} result={row.get('result', '')}\n"
        f"        user_head={row.get('user_head', '')!r}"
        for idx, row in enumerate(rows, 1)
    ]


def _append_replay_trace(
    *,
    slot_id: str,
    label: str,
    tier: str,
    attempt: int,
    outcome: str,
    live_rows: list[dict[str, str]],
    mock_rows: list[dict[str, str]],
    detail: str = "",
) -> None:
    """Append paired live-record and mock-replay traces for one slot to the replay trace file."""
    SANDBOX_REPLAY_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if SANDBOX_REPLAY_TRACE_PATH.is_file() and SANDBOX_REPLAY_TRACE_PATH.stat().st_size > 0:
        lines.append("\n" + "=" * 80)
    lines.append(f"[{outcome}] slot={slot_id} tier={tier} attempt={attempt}")
    lines.append(f"  question: {label}")
    if detail:
        lines.append(f"  detail: {detail}")
    lines.append(f"  LIVE record trace ({len(live_rows)} call(s)):")
    lines.extend(_format_trace_rows(live_rows))
    lines.append(f"  MOCK replay trace ({len(mock_rows)} call(s)):")
    lines.extend(_format_trace_rows(mock_rows))
    live_keys = {(r.get("task", ""), r.get("system_id", ""), r.get("user_sha", "")) for r in live_rows}
    misses = [r for r in mock_rows if str(r.get("result", "")).startswith("MISS")]
    unmatched = [
        r
        for r in mock_rows
        if (r.get("task", ""), r.get("system_id", ""), r.get("user_sha", "")) not in live_keys
    ]
    if misses:
        lines.append(f"  >>> {len(misses)} mock MISS(es): cached fixture not found during replay")
    if unmatched:
        lines.append(
            f"  >>> {len(unmatched)} mock call(s) requested a key the live record never produced "
            "(prompt nondeterminism or key drift)",
        )
    with open(SANDBOX_REPLAY_TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


@dataclass
class RecordingEnvironment:
    """Patches applied for live sandbox fixture recording."""

    orig_write_toml: Callable[..., str]
    orig_refine: Callable[..., dict[str, str]]
    orig_template_match: Callable[..., object]
    orig_persist_template_learning: Callable[[object | None], bool]
    prev_provider: str
    openai_toml: Callable[..., str]
    skip_template_reuse: Callable[..., object]
    llm_patch_modules: tuple[str, ...]


def prepare_recording_environment() -> RecordingEnvironment:
    """Load credentials, sync ``EngineConfig``, and apply recording patches."""
    from live_tests.conftest import write_sandbox_recording_toml

    from load_rental_shop_engines import load_env_file

    env_path = load_env_file(ENV_FILE, override=True)
    merged_env = dict(os.environ)
    merged_env["AETHERDIALECT_LLM_PROVIDER"] = "openai"
    try:
        _configure_llm_from_environment(merged_env)
    except ConfigError as exc:
        raise RuntimeError(
            f"LLM credentials are not configured (check {env_path}). {exc}",
        ) from exc
    if not llm_credentials_configured():
        raise RuntimeError(
            f"LLM credentials are not configured after loading {env_path}. "
            "Set OPENAI_API_KEY or the full Azure OpenAI variable set.",
        )

    def skip_template_reuse(*args: object, **kwargs: object) -> TemplateMatch:
        del args, kwargs
        return TemplateMatch(
            intent=None,
            best_template=None,
            similarity_score=0.0,
            reuse_type="none",
            reuse_candidate_normalized=None,
        )

    def openai_toml(*, fixtures_file: str) -> str:
        del fixtures_file
        return write_sandbox_recording_toml(str(ENV_FILE))

    orig_refine = aetherdialect._schema_overrides._refine_descriptions_via_llm
    orig_write_toml = aetherdialect._sandbox._write_sandbox_toml
    orig_template_match = aetherdialect._pipeline.match_question_level_template_reuse
    aetherdialect._sandbox._write_sandbox_toml = openai_toml
    prev_provider = EngineConfig.LLM_PROVIDER
    reset_mock_provider()
    EngineConfig.LLM_PROVIDER = "openai"
    aetherdialect._pipeline.match_question_level_template_reuse = skip_template_reuse
    aetherdialect._main_execution.match_question_level_template_reuse = skip_template_reuse
    aetherdialect._live_testing.match_question_level_template_reuse = skip_template_reuse
    orig_persist = aetherdialect._main_execution._persist_template_learning_for_pipeline_session

    def skip_template_learning(_port: object | None) -> bool:
        del _port
        return False

    aetherdialect._main_execution._persist_template_learning_for_pipeline_session = skip_template_learning

    return RecordingEnvironment(
        orig_write_toml=orig_write_toml,
        orig_refine=orig_refine,
        orig_template_match=orig_template_match,
        orig_persist_template_learning=orig_persist,
        prev_provider=prev_provider,
        openai_toml=openai_toml,
        skip_template_reuse=skip_template_reuse,
        llm_patch_modules=LLM_PATCH_MODULES,
    )


def teardown_recording_environment(env: RecordingEnvironment) -> None:
    """Restore module patches applied by :func:`prepare_recording_environment`."""
    aetherdialect._schema_overrides._refine_descriptions_via_llm = env.orig_refine
    aetherdialect._sandbox._write_sandbox_toml = env.orig_write_toml
    aetherdialect._pipeline.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._main_execution.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._live_testing.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._main_execution._persist_template_learning_for_pipeline_session = env.orig_persist_template_learning
    EngineConfig.LLM_PROVIDER = env.prev_provider


SANDBOX_SAMPLE_SEED = 2202
SMALL_TABLES_WHOLE = frozenset(
    {
        "category",
        "country",
        "language",
        "staff",
        "store",
        "promotion",
        "courier",
        "supplier",
        "warehouse",
        "author",
        "publisher",
    }
)

def corpus_message(message: str) -> None:
    print(message, flush=True)


def verbose_message(message: str) -> None:
    if _BUILD_VERBOSE:
        corpus_message(message)


@dataclass(frozen=True)
class RecordingSlot:
    tier: str
    label: str
    preset: str = "owner_writer"
    mode: str | None = None
    kind: str = "question"


def _paraphrase_catalog_ready(*, staging_dir: Path = STAGING) -> bool:
    path = staging_dir / "sandbox_catalog.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("paraphrase_pairs")
    if not isinstance(pairs, list) or not pairs:
        return False
    return any(
        isinstance(row, dict) and str(row.get("canonical", "")).strip()
        for row in pairs
    )


def _aetherspace_snapshots_ready(*, staging_dir: Path = STAGING) -> bool:
    from aetherdialect._constants import AETHERSPACES_SEGMENT

    dest = staging_dir / "artifacts_baseline" / AETHERSPACES_SEGMENT
    return dest.is_dir() and any(dest.glob("*.json"))


def _paraphrase_gatekeeper_ready(corpus: FixtureCorpus, paraphrase: str) -> bool:
    needle = paraphrase.strip()
    if not needle:
        return False
    for row in corpus.fixtures:
        if str(row.get("task", "")) != "default":
            continue
        if str(row.get("user", "")).strip() != needle:
            continue
        system = str(row.get("system", "")).lower()
        if "valid_database_question" in system or "decide if user input" in system:
            return True
    return False


def _reuse_fixtures_ready(corpus: FixtureCorpus) -> bool:
    paraphrase = "How many rentals happened in 2026?"
    if not _paraphrase_gatekeeper_ready(corpus, paraphrase):
        return False
    canonical = "How many rentals happened in 2025?"
    swaps = INLINE_REUSE_PARAM_COPY_RULES.get((canonical, paraphrase), ())
    if not swaps:
        return True
    forward_rows = _param_extraction_forward_rows_from_corpus(corpus.fixtures, canonical)
    return _reuse_reverse_param_rows_committed(
        corpus,
        forward_rows=forward_rows,
        swaps=swaps,
        llm_mod=aetherdialect._llm_provider,
    )


def _post_migration_record_question(questions: dict[str, list[str]]) -> str:
    practice = questions.get("questions", [])
    if practice:
        return str(practice[0])
    return "How many films are in the Rental Shop catalog?"


def _migration_fixtures_ready(corpus: FixtureCorpus) -> bool:
    """True when migration-demo schema_base fixtures exist (post-migration ``item_title`` column)."""
    for row in corpus.fixtures:
        if str(row.get("task", "")) != "schema_base":
            continue
        if "item_title" in str(row.get("user", "")):
            return True
    return False


def _catalog_paraphrase_canonicals(*, staging_dir: Path = STAGING) -> set[str]:
    path = staging_dir / "sandbox_catalog.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("paraphrase_pairs")
    if not isinstance(pairs, list):
        return set()
    out: set[str] = set()
    for row in pairs:
        if isinstance(row, dict):
            canonical = str(row.get("canonical", "")).strip()
            if canonical:
                out.add(canonical)
    return out


def _all_recording_slots_committed(
    questions: dict[str, list[str]],
    manifest: dict[str, object],
) -> bool:
    committed = committed_slot_ids(manifest)
    return all(slot_id_for(slot) in committed for slot in iter_recording_slots(questions))


def _missing_paraphrase_canonicals(
    questions: dict[str, list[str]],
    manifest: dict[str, object],
    *,
    staging_dir: Path = STAGING,
) -> list[RecordingSlot]:
    catalog = _catalog_paraphrase_canonicals(staging_dir=staging_dir)
    out: list[RecordingSlot] = []
    for slot in _committed_paraphrase_source_slots(questions, manifest):
        if slot.label not in catalog:
            out.append(slot)
    return out


def _paraphrase_catalog_complete(
    questions: dict[str, list[str]],
    manifest: dict[str, object],
    *,
    staging_dir: Path = STAGING,
) -> bool:
    return not _missing_paraphrase_canonicals(questions, manifest, staging_dir=staging_dir)


def _recording_pipeline_ready(
    *,
    questions: dict[str, list[str]] | None = None,
    manifest: dict[str, object] | None = None,
    corpus: FixtureCorpus | None = None,
    staging_dir: Path = STAGING,
) -> tuple[bool, list[str]]:
    questions = questions or load_staging_questions()
    manifest = manifest if manifest is not None else _load_recording_manifest()
    if corpus is None:
        corpus = FixtureCorpus(staging_dir / "fixtures" / "rental_shop_mock.json")
    reasons: list[str] = []
    if not _all_recording_slots_committed(questions, manifest):
        reasons.append("uncommitted recording slots")
    if not _paraphrase_catalog_complete(questions, manifest, staging_dir=staging_dir):
        missing = _missing_paraphrase_canonicals(questions, manifest, staging_dir=staging_dir)
        reasons.append(f"missing paraphrase catalog entries ({len(missing)})")
    if not _reuse_fixtures_ready(corpus):
        reasons.append("missing reuse fixtures")
    if not _aetherspace_snapshots_ready(staging_dir=staging_dir):
        reasons.append("missing aetherspace snapshots")
    if not _migration_fixtures_ready(corpus):
        reasons.append("missing migration schema_base fixtures")
    return (not reasons, reasons)


def _paraphrase_seeds_for_missing(
    session: RecordingSession,
    missing: list[RecordingSlot],
    collected: list[tuple[RecordingSlot, object]],
) -> tuple[list[tuple[RecordingSlot, object]], str]:
    by_label = {slot.label: (slot, step) for slot, step in collected}
    out: list[tuple[RecordingSlot, object]] = []
    need_live: list[RecordingSlot] = []
    for slot in missing:
        hit = by_label.get(slot.label)
        if hit is not None:
            out.append(hit)
        else:
            need_live.append(slot)
    if need_live:
        out.extend(session.collect_paraphrase_seeds(need_live))
    seeded = {slot.label for slot, _ in out}
    not_seeded = [slot.label for slot in missing if slot.label not in seeded]
    if not_seeded:
        preview = ", ".join(not_seeded[:3])
        suffix = "..." if len(not_seeded) > 3 else ""
        return out, f"paraphrase seed collection failed for {len(not_seeded)} slot(s): {preview}{suffix}"
    return out, ""


def _committed_paraphrase_source_slots(
    questions: dict[str, list[str]],
    manifest: dict[str, object],
) -> list[RecordingSlot]:
    committed = committed_slot_ids(manifest)
    out: list[RecordingSlot] = []
    for slot in iter_recording_slots(questions):
        if slot_id_for(slot) not in committed:
            continue
        if slot.kind != "question":
            continue
        if slot.tier not in PARAPHRASE_SOURCE_TIERS:
            continue
        if slot.preset != "owner_writer":
            continue
        if slot.mode not in (None, "writer"):
            continue
        if not _paraphrase_eligible_question(slot.label, kind=slot.kind):
            continue
        out.append(slot)
    return out


def parse_questions_file(path: Path) -> dict[str, list[str]]:
    tiers: dict[str, list[str]] = {
        "questions": [],
        "validation_failures": [],
        "feedback_samples": [],
    }
    current = "questions"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip().lower().replace(" ", "_")
            if header in tiers:
                current = header
            continue
        if current:
            tiers[current].append(stripped)
    return tiers


def smoke_questions(*, source: Path | None = None) -> dict[str, list[str]]:
    """Minimal question set: two practice questions plus all validation/feedback sections."""
    full = parse_questions_file(source or QUESTIONS_SOURCE)
    practice: list[str] = []
    for candidate in (SMOKE_TOUR_QUESTION, SMOKE_FLAKY_QUESTION):
        if candidate in full["questions"] and candidate not in practice:
            practice.append(candidate)
    if not practice:
        practice = list(full["questions"][:2])
    return {
        "questions": practice,
        "validation_failures": [
            question
            for question in full["validation_failures"]
            if question != "How many rentals were made in total?"
        ],
        "feedback_samples": list(full["feedback_samples"]),
    }


def _smoke_expectations_from_source() -> list[dict[str, object]]:
    """Subset hand-maintained expectations for smoke recording + consumer validation slots."""
    payload = json.loads(EXPECTATIONS_SOURCE.read_text(encoding="utf-8"))
    slots = payload.get("slots")
    if not isinstance(slots, list):
        raise SystemExit(f"Invalid expectations file: {EXPECTATIONS_SOURCE}")
    smoke_q = smoke_questions()
    smoke_slot_ids = {slot_id_for(slot) for slot in iter_recording_slots(smoke_q)}
    smoke_slot_ids.update(slot_id_for(slot) for slot in iter_consumer_validation_slots(smoke_q))
    return [row for row in slots if isinstance(row, dict) and row.get("slot_id") in smoke_slot_ids]


def write_questions_file(path: Path, questions: dict[str, list[str]]) -> None:
    lines: list[str] = []
    for tier in ("questions", "validation_failures", "feedback_samples"):
        lines.append(f"# {tier.replace('_', ' ')}")
        lines.extend(questions.get(tier, []))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _feedback_anchor_question(questions: dict[str, list[str]]) -> str:
    practice = questions.get("questions", [])
    if len(practice) >= 2:
        return practice[1]
    if practice:
        return practice[0]
    return ""


def load_staging_questions() -> dict[str, list[str]]:
    path = STAGING / "questions.txt"
    if not path.is_file():
        path = QUESTIONS_SOURCE
    return parse_questions_file(path)


def iter_recording_slots(questions: dict[str, list[str]]) -> list[RecordingSlot]:
    """Live LLM fixture slots — owner writer only (consumer is validated, not re-recorded)."""
    slots: list[RecordingSlot] = []
    for tier in SANDBOX_QUESTION_TIERS:
        for question in questions[tier]:
            slots.append(RecordingSlot(tier=tier, label=question))
    feedback = questions["feedback_samples"]
    if feedback:
        for sample in feedback:
            slots.append(RecordingSlot(tier="feedback", label=sample, kind="feedback"))
    slots.append(RecordingSlot(tier="spaces", label="catalog", kind="space"))
    return slots


def iter_consumer_validation_slots(questions: dict[str, list[str]]) -> list[RecordingSlot]:
    """Consumer reader replay slots — validated at pack time, not live-recorded."""
    return [
        RecordingSlot(
            tier="consumer_reader",
            label=question,
            preset="consumer_reader",
            mode="reader",
        )
        for question in questions.get("questions", [])
    ]


def _expectation_row_for_slot(slot: RecordingSlot) -> dict[str, object]:
    profile = slot.preset if slot.kind == "question" else "feedback"
    return {
        "slot_id": slot_id_for(slot),
        "question": slot.label,
        "kind": slot.kind,
        "profile": profile,
        "mode": slot.mode,
        "tier": slot.tier,
        "expect": _expect_for_question(slot.label, kind=slot.kind),
    }


def build_expectations() -> list[dict[str, object]]:
    questions = load_staging_questions()
    rows: list[dict[str, object]] = []
    for slot in iter_recording_slots(questions):
        rows.append(_expectation_row_for_slot(slot))
    for slot in iter_consumer_validation_slots(questions):
        rows.append(_expectation_row_for_slot(slot))
    return rows


INVALID_QUESTIONS = frozenset(
    {
        "What's the weather today?",
        "What is the best pizza topping?",
    },
)
NO_SQL_QUESTIONS = frozenset(
    {
        "What's the weather today?",
        "Show payroll deductions by employee SSN.",
        "How many rentals happened on 2025-01-01?",
    },
)
VALIDATION_FAILURE_QUESTIONS = frozenset(
    {
        "Show payroll deductions by employee SSN.",
        "Show me all staff salaries.",
        "How many rentals happened on 2025-01-01?",
    },
)
FAITHFULNESS: dict[str, dict[str, object]] = {
    "Which games support English?": {
        "must_tables": [],
        "sql_contains": ["game_supported_language"],
        "contains_join": True,
    },
    "Which city has the most customers?": {
        "must_tables": ["city", "customer"],
        "contains_join": True,
    },
    "How many customers are in each country?": {
        "must_tables": ["country", "customer"],
        "contains_join": True,
    },
    "Film title and replacement cost minus rental rate as profit margin": {
        "forbidden_sql_tokens": ["interval"],
    },
    "How many rentals were made in total?": {
        "must_tables": ["film"],
        "sql_contains": ["item_id"],
    },
}
SANDBOX_ZIP_BUILD_TIME_ONLY = frozenset(
    {
    },
)
PARAPHRASE_SOURCE_TIERS = frozenset({"questions"})
INLINE_PARAPHRASE_COPY_RULES: dict[str, tuple[str, ...]] = {
    "How many rentals happened in 2025?": ("How many rentals happened in 2026?",),
}
INLINE_REUSE_PARAM_COPY_RULES: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("How many rentals happened in 2025?", "How many rentals happened in 2026?"): (("2025", "2026"),),
}
PARAM_EXTRACTION_SYSTEM_MARKER = "deterministic parameter value extractor"


def _paraphrase_eligible_question(question: str, *, kind: str = "question") -> bool:
    """Return True when a committed slot should appear in the paraphrase catalog."""
    if kind != "question":
        return False
    if question in INVALID_QUESTIONS:
        return False
    expect = _expect_for_question(question, kind=kind)
    if expect.get("terminal_status") in {"invalid_question", "error"}:
        return False
    if expect.get("sql_required") is False:
        return False
    return True


def _expect_for_question(question: str, *, kind: str) -> dict[str, object]:
    if question in INVALID_QUESTIONS:
        return {
            "terminal_status": "invalid_question",
            "sql_required": False,
            "grain": "none",
            "must_tables": [],
            "must_filter": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
        }
    if question in NO_SQL_QUESTIONS or question in VALIDATION_FAILURE_QUESTIONS:
        return {
            "terminal_status": "error",
            "sql_required": False,
            "grain": "none",
            "must_tables": [],
            "must_filter": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
            "validation_failure": question in VALIDATION_FAILURE_QUESTIONS,
        }
    if kind == "feedback":
        return {
            "terminal_status": "ok",
            "sql_required": True,
            "grain": "scalar",
            "must_tables": [],
            "must_filter": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
        }
    extra = FAITHFULNESS.get(question, {})
    return {
        "terminal_status": "ok",
        "sql_required": True,
        "grain": "scalar",
        "must_tables": list(extra.get("must_tables", [])),
        "must_filter": [],
        "sql_contains": list(extra.get("sql_contains", [])),
        "forbidden_sql_tokens": list(extra.get("forbidden_sql_tokens", [])),
        "contains_join": extra.get("contains_join"),
    }


def build_scenarios() -> list[dict[str, object]]:
    questions = load_staging_questions()
    feedback = questions.get("feedback_samples", [])
    anchor = _feedback_anchor_question(questions) or "How many books do we have?"
    rows: list[dict[str, object]] = [
        {
            "id": "hidden_column",
            "question": "Show payroll deductions by employee SSN.",
            "mechanism": "bundled_overrides_hide_staff_ssn",
        },
        {
            "id": "schema_invalid",
            "question": "Show me all staff salaries.",
            "mechanism": "schema_validation_failure",
        },
        {
            "id": "fail_everywhere",
            "question": "How many rentals happened on 2025-01-01?",
            "mechanism": "schema_validation_failure",
        },
        {
            "id": "pass_but_wrong",
            "question": "How many rentals were made in total?",
        },
    ]
    for sample in feedback[:1]:
        rows.append(
            {
                "id": "feedback_demo",
                "kind": "feedback",
                "anchor_question": anchor,
                "allowed_rejection_text": sample,
                "flow": "reject_then_accept",
            },
        )
    return rows


def build_sandbox_catalog(*, paraphrase_pairs: list[dict[str, object]] | None = None) -> dict[str, object]:
    """User-facing discovery catalog (shipped in data.zip)."""
    pairs_out: list[dict[str, object]] = []
    for row in paraphrase_pairs or []:
        canonical = str(row.get("canonical", "")).strip()
        paraphrases = row.get("paraphrases")
        if not canonical or not isinstance(paraphrases, list):
            continue
        pairs_out.append(
            {
                "canonical": canonical,
                "paraphrases": [str(item).strip() for item in paraphrases if str(item).strip()],
            },
        )
    validation_demo: list[dict[str, str]] = []
    for row in _load_scenarios():
        if str(row.get("mechanism", "")) == "schema_validation_failure":
            question = str(row.get("question", "")).strip()
            if question:
                validation_demo.append(
                    {
                        "question": question,
                        "description": "Expected terminal error: schema validation failure (no SQL).",
                    },
                )
    feedback_row = _feedback_scenario()
    feedback_demo: dict[str, str] = {}
    if feedback_row:
        anchor = str(feedback_row.get("anchor_question", "")).strip()
        rejection = str(feedback_row.get("allowed_rejection_text", "")).strip()
        if anchor and rejection:
            feedback_demo = {
                "anchor_question": anchor,
                "allowed_rejection_text": rejection,
                "description": "Reject the anchor intent with the allowed text, then accept the retry.",
            }
    return {
        "version": 1,
        "paraphrase_pairs": pairs_out,
        "validation_failure_demo": validation_demo,
        "feedback_demo": feedback_demo,
    }


def write_sandbox_catalog_file(
    *,
    target_dir: Path = STAGING,
    paraphrase_pairs: list[dict[str, object]] | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = build_sandbox_catalog(paraphrase_pairs=paraphrase_pairs)
    write_text_atomic(
        target_dir / "sandbox_catalog.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def fixture_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("task", "")),
        str(row.get("system", "")),
        str(row.get("user", "")).strip(),
    )


def _intent_system_label(system: str) -> str:
    """Short, stable label for a system prompt (for readable trace lines)."""
    if system == INTENT_INTERPRET_SYSTEM:
        return "interpret"
    if system == INTENT_GROUND_SYSTEM:
        return "ground"
    if system == INTENT_COMPOSE_SYSTEM:
        return "compose"
    if PARAM_EXTRACTION_SYSTEM_MARKER in system.lower():
        return "param_extraction"
    return "default"


def write_text_atomic(path: Path, text: str, *, attempts: int = 5) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_err = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    if last_err is not None:
        raise last_err


class FixtureCorpus:
    """Committed fixture store with per-slot buffering."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fixtures: list[dict[str, str]] = []
        self.seen: set[tuple[str, str, str]] = set()
        self._buffer: dict[tuple[str, str, str], dict[str, str]] = {}
        self._buffer_order: list[tuple[str, str, str]] = []
        self._load()
        self.committed_keys: set[tuple[str, str, str]] = set(self.seen)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        entries = raw.get("fixtures", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return
        self.fixtures = [dict(row) for row in entries if isinstance(row, dict)]
        self.seen = {fixture_key(row) for row in self.fixtures}

    def flush(self) -> None:
        payload = {"version": 1, "fixtures": self.fixtures}
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.path, text)

    def start_slot(self) -> None:
        self._buffer.clear()
        self._buffer_order.clear()

    def record(self, *, task: str, system: str, user_key: str, output_text: str) -> None:
        key = (task, system, user_key)
        entry = {
            "task": task,
            "system": system,
            "user": user_key,
            "output_text": output_text,
        }
        self._buffer[key] = entry
        if key not in self._buffer_order:
            self._buffer_order.append(key)

    def discard_slot(self) -> None:
        self._buffer.clear()
        self._buffer_order.clear()

    def collapsed_slot_fixtures(self) -> list[dict[str, str]]:
        """One entry per (task, system, user_key); identical keys keep last output."""
        return [dict(self._buffer[key]) for key in self._buffer_order]

    def buffered_keys(self) -> set[tuple[str, str, str]]:
        """Return the (task, system, user) keys currently staged in the slot buffer."""
        return {fixture_key(row) for row in self.collapsed_slot_fixtures()}

    def commit_slot(self, *, freeze: bool = True) -> set[tuple[str, str, str]]:
        """Merge the slot buffer into the corpus by exact key without overwriting frozen keys."""
        frozen = self.committed_keys
        written: set[tuple[str, str, str]] = set()
        existing = {fixture_key(row): idx for idx, row in enumerate(self.fixtures)}
        for entry in self.collapsed_slot_fixtures():
            key = fixture_key(entry)
            idx = existing.get(key)
            if idx is not None:
                prior = self.fixtures[idx]
                if str(prior.get("output_text", "")) != str(entry.get("output_text", "")):
                    if key in frozen:
                        continue
                    self.fixtures[idx] = entry
                    written.add(key)
                continue
            self.fixtures.append(entry)
            existing[key] = len(self.fixtures) - 1
            self.seen.add(key)
            written.add(key)
        self.discard_slot()
        self.flush()
        if freeze:
            self.committed_keys |= written
        return written

    def rollback_keys(self, keys: set[tuple[str, str, str]]) -> None:
        """Remove the given keys from the corpus (used to undo a failed slot's tentative merge)."""
        if not keys:
            return
        self.fixtures = [row for row in self.fixtures if fixture_key(row) not in keys]
        self.seen = {fixture_key(row) for row in self.fixtures}
        self.flush()

    def snapshot(self) -> list[dict[str, str]]:
        """Deep-copy the committed corpus rows so a failed tentative merge can be fully undone."""
        return [dict(row) for row in self.fixtures]

    def restore(self, rows: list[dict[str, str]]) -> None:
        """Restore corpus rows from a snapshot (frozen keys are unaffected) and reflush to disk."""
        self.fixtures = [dict(row) for row in rows]
        self.seen = {fixture_key(row) for row in self.fixtures}
        self.flush()


def _digest(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _remove_tree(path: Path) -> None:
    def _on_rm_error(func: Any, p: str, _exc_info: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_rm_error)


def _prepare_sqlite_from_csvs() -> None:
    import argparse

    import load_rental_shop_engines as load_mod
    from load_rental_shop_engines import (
        DEFAULT_ENV_FILE,
        _load_sqlite,
        default_csv_dir,
        default_ddl_path,
        load_env_file,
    )
    from source_rental_shop import OUT_DIR, ZIP_PATH, ensure_csv_bundle

    source = ensure_csv_bundle(OUT_DIR, ZIP_PATH)
    verbose_message(f"CSV bundle source: {source}")
    load_env_file(DEFAULT_ENV_FILE, override=True)
    original_log_progress = load_mod._log_progress
    if not _BUILD_VERBOSE:
        load_mod._log_progress = lambda *_args, **_kwargs: None
    load_args = argparse.Namespace(
        engine="sqlite",
        csv_dir=default_csv_dir(),
        ddl=default_ddl_path(),
        env_file=DEFAULT_ENV_FILE,
        schema=None,
        drop_first=True,
        recreate_schema=False,
        allow_public_schema_recreate=False,
        database=None,
    )
    if not load_args.csv_dir.is_dir():
        raise SystemExit(f"Missing CSV directory: {load_args.csv_dir}")
    try:
        _load_sqlite(load_args)
    finally:
        load_mod._log_progress = original_log_progress


def _compute_sandbox_subset(conn: sqlite3.Connection) -> dict[str, set[int]]:
    customers = [int(row[0]) for row in conn.execute("SELECT customer_id FROM customer ORDER BY customer_id")]
    films = [int(row[0]) for row in conn.execute("SELECT item_id FROM film ORDER BY item_id")]
    books = [int(row[0]) for row in conn.execute("SELECT item_id FROM book ORDER BY item_id")]
    games = [int(row[0]) for row in conn.execute("SELECT item_id FROM game ORDER BY item_id")]
    customer_ids = {
        customers[_digest(SANDBOX_SAMPLE_SEED, "cust", index) % len(customers)]
        for index in range(min(80, len(customers)))
    }
    film_ids = {
        films[_digest(SANDBOX_SAMPLE_SEED, "film", index) % len(films)] for index in range(min(100, len(films)))
    }
    book_ids = {books[_digest(SANDBOX_SAMPLE_SEED, "book", index) % len(books)] for index in range(min(40, len(books)))}
    game_ids = {games[_digest(SANDBOX_SAMPLE_SEED, "game", index) % len(games)] for index in range(min(20, len(games)))}
    item_ids = film_ids | book_ids | game_ids
    address_ids: set[int] = set()
    for customer_id in customer_ids:
        row = conn.execute("SELECT address_id FROM customer WHERE customer_id=?", (customer_id,)).fetchone()
        if row and row[0] is not None:
            address_ids.add(int(row[0]))
    for staff_row in conn.execute("SELECT staff_id FROM staff"):
        row = conn.execute("SELECT address_id FROM staff WHERE staff_id=?", (staff_row[0],)).fetchone()
        if row and row[0] is not None:
            address_ids.add(int(row[0]))
    city_ids: set[int] = set()
    for address_id in address_ids:
        row = conn.execute("SELECT city_id FROM address WHERE address_id=?", (address_id,)).fetchone()
        if row and row[0] is not None:
            city_ids.add(int(row[0]))
    country_ids: set[int] = set()
    for city_id in city_ids:
        row = conn.execute("SELECT country_id FROM city WHERE city_id=?", (city_id,)).fetchone()
        if row and row[0] is not None:
            country_ids.add(int(row[0]))
    language_ids: set[int] = set()
    for item_id in film_ids:
        row = conn.execute(
            "SELECT original_language_id FROM film WHERE item_id=?",
            (item_id,),
        ).fetchone()
        if row and row[0] is not None:
            language_ids.add(int(row[0]))
    inventory_ids: set[int] = set()
    store_ids: set[int] = set()
    for item_id in item_ids:
        for inv_id, store_id in conn.execute(
            "SELECT inventory_id, store_id FROM inventory WHERE item_id=?",
            (item_id,),
        ):
            inventory_ids.add(int(inv_id))
            store_ids.add(int(store_id))
    rental_ids: set[int] = set()
    for customer_id in customer_ids:
        for (rental_id,) in conn.execute("SELECT rental_id FROM rental WHERE customer_id=?", (customer_id,)):
            rental_ids.add(int(rental_id))
    payment_ids: set[int] = set()
    for rental_id in rental_ids:
        for (payment_id,) in conn.execute("SELECT payment_id FROM payment WHERE rental_id=?", (rental_id,)):
            payment_ids.add(int(payment_id))
    delivery_ids: set[int] = set()
    for rental_id in rental_ids:
        for (delivery_id,) in conn.execute("SELECT delivery_id FROM delivery WHERE rental_id=?", (rental_id,)):
            delivery_ids.add(int(delivery_id))
    actor_ids: set[int] = set()
    category_ids: set[int] = set()
    for item_id in film_ids:
        for (actor_id,) in conn.execute(
            "SELECT actor_id FROM film_actor WHERE film_item_id=?",
            (item_id,),
        ):
            actor_ids.add(int(actor_id))
        for (category_id,) in conn.execute(
            "SELECT category_id FROM item_category WHERE item_id=?",
            (item_id,),
        ):
            category_ids.add(int(category_id))
    redemption_ids: set[int] = set()
    for rental_id in rental_ids:
        for (redemption_id,) in conn.execute(
            "SELECT redemption_id FROM promotion_redemption WHERE rental_id=?",
            (rental_id,),
        ):
            redemption_ids.add(int(redemption_id))
    po_ids: set[int] = set()
    for store_id in store_ids:
        for (po_id,) in conn.execute(
            "SELECT po_id FROM purchase_order WHERE store_id=?",
            (store_id,),
        ):
            po_ids.add(int(po_id))
    line_ids: set[int] = set()
    for po_id in po_ids:
        for (line_id,) in conn.execute(
            "SELECT line_id FROM purchase_line WHERE po_id=?",
            (po_id,),
        ):
            line_ids.add(int(line_id))
    transfer_ids: set[int] = set()
    for item_id in item_ids:
        for (transfer_id,) in conn.execute(
            "SELECT transfer_id FROM stock_transfer WHERE item_id=?",
            (item_id,),
        ):
            transfer_ids.add(int(transfer_id))
    return {
        "customer": customer_ids,
        "film": film_ids,
        "book": book_ids,
        "game": game_ids,
        "item": item_ids,
        "address": address_ids,
        "city": city_ids,
        "country": country_ids,
        "language": language_ids,
        "inventory": inventory_ids,
        "store": store_ids,
        "rental": rental_ids,
        "payment": payment_ids,
        "delivery": delivery_ids,
        "actor": actor_ids,
        "category": category_ids,
        "film_actor": set(),
        "item_category": set(),
        "promotion_redemption": redemption_ids,
        "purchase_order": po_ids,
        "purchase_line": line_ids,
        "stock_transfer": transfer_ids,
    }


def _row_allowed(table_name: str, cols: list[str], row: tuple[object, ...], subset: dict[str, set[int]]) -> bool:
    if table_name in SMALL_TABLES_WHOLE:
        return True
    row_map = dict(zip(cols, row, strict=True))
    if table_name == "actor":
        return int(row_map["actor_id"]) in subset["actor"]
    if table_name == "address":
        return int(row_map["address_id"]) in subset["address"]
    if table_name == "city":
        return int(row_map["city_id"]) in subset["city"]
    if table_name == "country":
        return int(row_map["country_id"]) in subset["country"]
    if table_name == "film":
        return int(row_map["item_id"]) in subset["film"]
    if table_name == "item":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "book":
        return int(row_map["item_id"]) in subset["book"]
    if table_name == "game":
        return int(row_map["item_id"]) in subset["game"]
    if table_name == "language":
        return int(row_map["language_id"]) in subset["language"]
    if table_name == "film_actor":
        return int(row_map["film_item_id"]) in subset["film"]
    if table_name == "item_category":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "item_feature":
        return int(row_map["item_id"]) in subset["item"]
    if table_name == "game_supported_language":
        return int(row_map["item_id"]) in subset["game"]
    if table_name == "reservation":
        return int(row_map["customer_id"]) in subset["customer"]
    if table_name == "inventory_status_history":
        return int(row_map["inventory_id"]) in subset["inventory"]
    if table_name == "damage_report":
        return int(row_map["rental_id"]) in subset["rental"]
    if table_name == "inventory":
        return int(row_map["inventory_id"]) in subset["inventory"]
    if table_name == "customer":
        return int(row_map["customer_id"]) in subset["customer"]
    if table_name == "rental":
        return int(row_map["rental_id"]) in subset["rental"]
    if table_name == "payment":
        return int(row_map["payment_id"]) in subset["payment"]
    if table_name == "delivery":
        return int(row_map["delivery_id"]) in subset["delivery"]
    if table_name == "purchase_order":
        return int(row_map["po_id"]) in subset["purchase_order"]
    if table_name == "purchase_line":
        return int(row_map["line_id"]) in subset["purchase_line"]
    if table_name == "stock_transfer":
        return int(row_map["transfer_id"]) in subset["stock_transfer"]
    if table_name == "promotion_redemption":
        return int(row_map["redemption_id"]) in subset["promotion_redemption"]
    return False


def export_seed_sql(path: Path) -> None:
    sqlite_path = REPO / "scripts" / "sqlite" / "rental_shop.sqlite"
    if not sqlite_path.is_file():
        raise SystemExit(f"Missing seed source: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    try:
        subset = _compute_sandbox_subset(conn)
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        ).fetchall()
        lines: list[str] = []
        for (table_name,) in table_rows:
            if str(table_name).startswith("sqlite_"):
                continue
            create_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if create_row and create_row[0]:
                lines.append(str(create_row[0]).rstrip(";") + ";;")
            cols = [desc[0] for desc in conn.execute(f"SELECT * FROM [{table_name}] LIMIT 0").description]
            col_list = ", ".join(cols)
            for row in conn.execute(f"SELECT * FROM [{table_name}]"):
                if not _row_allowed(str(table_name), cols, row, subset):
                    continue
                vals = []
                for value in row:
                    if value is None or value == "":
                        vals.append("NULL")
                    elif isinstance(value, bool):
                        vals.append("1" if value else "0")
                    elif value in ("t", "f", "true", "false"):
                        vals.append("1" if str(value).lower() in ("t", "true") else "0")
                    elif isinstance(value, str):
                        vals.append("'" + value.replace("'", "''") + "'")
                    elif isinstance(value, bytes):
                        vals.append("X'" + value.hex() + "'")
                    else:
                        vals.append(repr(value))
                lines.append(f"INSERT INTO {table_name} ({col_list}) VALUES ({', '.join(vals)});")
    finally:
        conn.close()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verbose_message(f"Wrote {path} ({path.stat().st_size} bytes)")


def assemble_staging(*, reset_fixtures: bool = True, smoke: bool = False) -> None:
    """Populate ``scripts/sandbox_staging`` from canonical pipeline data."""
    if smoke:
        corpus_message(
            "[smoke] subsetting questions from scripts/data/sandbox_questions.txt",
        )
    if STAGING.is_dir():
        _remove_tree(STAGING)
    STAGING.mkdir(parents=True)
    _prepare_sqlite_from_csvs()
    export_seed_sql(STAGING / "rental_shop_seed.sql")
    for name, src in (
        ("rental_shop.sql", DATA / "rental_shop.sql"),
        ("rental_shop_views.sql", DATA / "rental_shop_views.sql"),
        ("rental_shop_notes.txt", DATA / "rental_shop_notes.txt"),
        ("questions.txt", QUESTIONS_SOURCE),
        ("sandbox_expectations.json", EXPECTATIONS_SOURCE),
        ("sandbox_scenarios.json", SCENARIOS_SOURCE),
        ("sandbox_handcrafted_fixtures.json", HANDCRAFTED_FIXTURES_SOURCE),
        ("sandbox_space_catalog_notes.txt", SPACE_CATALOG_NOTES_SOURCE),
    ):
        if not src.is_file():
            raise SystemExit(f"Missing source file: {src}")
        if smoke and name == "questions.txt":
            write_questions_file(STAGING / name, smoke_questions())
            continue
        if smoke and name == "sandbox_expectations.json":
            expectations = _smoke_expectations_from_source()
            (STAGING / name).write_text(
                json.dumps({"version": 1, "slots": expectations}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            verbose_message(f"Wrote smoke expectations ({len(expectations)} slots from scripts/data)")
            continue
        shutil.copy2(src, STAGING / name)
    write_sandbox_catalog_file(target_dir=STAGING)
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if reset_fixtures:
        FIXTURES_PATH.write_text('{"version": 1, "fixtures": []}\n', encoding="utf-8")
    corpus = FixtureCorpus(FIXTURES_PATH)
    print(f"Staged fixtures corpus ({len(corpus.fixtures)} entries)")


def write_overrides_demo() -> None:
    if not OVERRIDES_DEMO_SOURCE.is_file():
        raise SystemExit(f"Missing overrides demo source: {OVERRIDES_DEMO_SOURCE}")
    (STAGING / "schema_overrides_demo.json").write_text(
        OVERRIDES_DEMO_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print("Wrote schema_overrides_demo.json")


def write_migration_demo() -> None:
    if not MIGRATION_DEMO_SOURCE.is_file():
        raise SystemExit(f"Missing migration demo source: {MIGRATION_DEMO_SOURCE}")
    demo_root = STAGING / "migration_demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    (demo_root / "schema_migration_map.json").write_text(
        MIGRATION_DEMO_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Wrote migration demo map under {demo_root}")


def build_artifacts_baseline() -> None:
    """Build fresh schema artifacts and bundled schema literals from staging."""
    from live_tests.conftest import write_sandbox_recording_toml

    import aetherdialect._sandbox
    import aetherdialect.aetherdialect
    from aetherdialect.aetherdialect import AetherEngine
    from load_rental_shop_engines import DEFAULT_ENV_FILE, load_env_file

    load_env_file(DEFAULT_ENV_FILE, override=True)
    baseline_root = STAGING / "artifacts_baseline"
    owner_baseline = baseline_root / BASELINE_OWNER_SUBDIR
    consumer_baseline = baseline_root / BASELINE_CONSUMER_SUBDIR
    if baseline_root.is_dir():
        _remove_tree(baseline_root)
    orig_write_toml = aetherdialect._sandbox._write_sandbox_toml
    prev_provider = EngineConfig.LLM_PROVIDER
    prev_regen = PolicyConfig.REGENERATE_SCHEMA_GRAPH

    def openai_toml(*, fixtures_file: str) -> str:
        del fixtures_file
        return write_sandbox_recording_toml(str(DEFAULT_ENV_FILE))

    def build_log_sink(line: str) -> None:
        del line

    aetherdialect._sandbox._write_sandbox_toml = openai_toml
    reset_mock_provider()
    merged_env = dict(os.environ)
    merged_env["AETHERDIALECT_LLM_PROVIDER"] = "openai"
    _configure_llm_from_environment(merged_env)
    PolicyConfig.REGENERATE_SCHEMA_GRAPH = True
    original_log_sink = aetherdialect.aetherdialect._init_log_sink
    aetherdialect.aetherdialect._init_log_sink = build_log_sink
    seed_sql = str(STAGING / "rental_shop_seed.sql")
    try:
        corpus_message("[build] artifacts baseline: building schema graph...")
        with AetherEngine.offline_sandbox(
            cleanup_artifacts=True,
            bundle_dir=str(STAGING),
            seed_sql=seed_sql,
        ) as owner_sb:
            owner_literal = owner_sb.engine._schema_graph.schema_literal_json
            schema_graph = owner_sb.engine._schema_graph
            table_count = len(schema_graph.tables)
            column_count = sum(len(table.columns) for table in schema_graph.tables.values())
            corpus_message(
                f"[build] artifacts baseline: schema graph ready ({table_count} tables, {column_count} columns)",
            )
            owner_engine_dir = Path(EngineConfig.SCHEMA_JSON_PATH).parent
            owner_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(owner_engine_dir, owner_baseline)
            consumer_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(owner_engine_dir, consumer_baseline)
            written = [
                f"{name}={ (owner_baseline / name).stat().st_size }B"
                for name in _BASELINE_CACHE_FILES
                if (owner_baseline / name).is_file()
            ]
            verbose_message(f"[build] artifacts baseline files: {', '.join(written) or 'none'}")
        literals_payload = {"owner": owner_literal, "consumer": owner_literal}
        (STAGING / SANDBOX_SCHEMA_LITERALS_FILENAME).write_text(
            json.dumps(literals_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        interpret_domain = json.loads(owner_sb.engine._schema_graph.schema_payload_interpret())
        (STAGING / SANDBOX_INTERPRET_DOMAIN_FILENAME).write_text(
            json.dumps(interpret_domain, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        demo_root = STAGING / "migration_demo" / "artifacts_v1"
        if demo_root.is_dir():
            _remove_tree(demo_root)
        demo_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(owner_baseline, demo_root)
    finally:
        aetherdialect.aetherdialect._init_log_sink = original_log_sink
        PolicyConfig.REGENERATE_SCHEMA_GRAPH = prev_regen
        aetherdialect._sandbox._write_sandbox_toml = orig_write_toml
        reset_mock_provider()
        EngineConfig.LLM_PROVIDER = prev_provider
    aetherdialect._sandbox._pin_bundled_schema_literals(STAGING)
    verbose_message(f"Wrote artifacts baseline under {baseline_root}")


def ensure_schema_literals(staging_dir: Path = STAGING) -> None:
    """No-op when bundled schema literals already exist (written during baseline build)."""
    target = staging_dir / SANDBOX_SCHEMA_LITERALS_FILENAME
    if target.is_file():
        return
    corpus_message(
        f"Warning: {target.name} missing under {staging_dir}; run build_artifacts_baseline() first.",
    )


def pin_staging_mock_fixture_keys(staging_dir: Path = STAGING) -> None:
    """Pin mock-fixture lookup keys from staged bundle files (same path as offline_sandbox)."""
    aetherdialect._sandbox._pin_bundled_schema_literals(staging_dir)


def _assert_pinning_parity(staging_dir: Path = STAGING) -> None:
    """Fail fast when pinned mock-lookup context does not byte-match staged bundle files."""
    llm = aetherdialect._llm_provider
    domain_path = staging_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if domain_path.is_file():
        staged_domain = json.loads(domain_path.read_text(encoding="utf-8"))
        if llm.load_canonical_interpret_domain() != staged_domain:
            raise SystemExit(
                "sandbox build: pinned interpret domain does not match staged "
                f"{SANDBOX_INTERPRET_DOMAIN_FILENAME}; mock keys would drift from replay.",
            )
    literals_path = staging_dir / SANDBOX_SCHEMA_LITERALS_FILENAME
    if literals_path.is_file():
        payload = json.loads(literals_path.read_text(encoding="utf-8"))
        pinned = llm.load_canonical_schema_literals()
        expected_owner = llm.stable_schema_literal(str(payload.get("owner", "")))
        expected_consumer = llm.stable_schema_literal(str(payload.get("consumer", "")))
        if pinned.get("owner") != expected_owner or pinned.get("consumer") != expected_consumer:
            raise SystemExit(
                "sandbox build: pinned schema literals do not match staged "
                f"{SANDBOX_SCHEMA_LITERALS_FILENAME}; mock keys would drift from replay.",
            )


def ensure_interpret_domain(staging_dir: Path = STAGING) -> None:
    """Write schema_interpret_domain.json to staging when missing."""
    target = staging_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if target.is_file():
        return
    seed_sql = str(staging_dir / "rental_shop_seed.sql")
    if not Path(seed_sql).is_file():
        return
    with AetherEngine.offline_sandbox(
        cleanup_artifacts=True,
        bundle_dir=str(staging_dir),
        seed_sql=seed_sql,
    ) as handle:
        domain = json.loads(handle.engine._schema_graph.schema_payload_interpret())
    target.write_text(json.dumps(domain, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class WarmRecordingPool:
    """Reuse one DuckDB connection; isolate mock-verify artifacts per run."""

    def __init__(self, bundle_dir: Path) -> None:
        self._bundle_dir = bundle_dir
        self._seed_sql = str(bundle_dir / "rental_shop_seed.sql")
        self._live_artifacts_owner = tempfile.mkdtemp(prefix="aetherdialect_record_live_owner_")
        self._live_artifacts_consumer = tempfile.mkdtemp(prefix="aetherdialect_record_live_consumer_")
        self._connection = aetherdialect._sandbox._load_memory_connection(self._seed_sql)
        self._engine_cls = AetherEngine
        self._live_handles: dict[str, object] = {}
        self._closed = False
        self._literals_pinned = False
        self._baseline_bytes_cache: dict[str, dict[str, bytes]] = {}

    def _artifacts_for_preset(self, preset: str) -> str:
        if preset == "consumer_reader":
            return self._live_artifacts_consumer
        return self._live_artifacts_owner

    def _pin_schema_literals(self) -> None:
        if self._literals_pinned:
            return
        aetherdialect._sandbox._pin_bundled_schema_literals(self._bundle_dir)
        self._literals_pinned = True

    def _seed_engine_baseline_cached(self, *, artifacts_dir: str, preset: str) -> None:
        """Seed an ephemeral engine dir from the baseline, reading source files at most once per pool."""
        baseline = baseline_dir_for_preset(self._bundle_dir, preset)
        if not baseline.is_dir() or not (baseline / "schema_graph.json.gz").is_file():
            _seed_engine_baseline(artifacts_dir=artifacts_dir, bundle_dir=self._bundle_dir, preset=preset)
            return
        engine_dir = _sandbox_memory_engine_dir(artifacts_dir)
        if (engine_dir / "schema_graph.json.gz").is_file():
            return
        cache = self._baseline_bytes_cache.get(str(baseline))
        if cache is None:
            cache = {}
            for name in _BASELINE_CACHE_FILES:
                src = baseline / name
                if src.is_file():
                    cache[name] = src.read_bytes()
            for sidecar in baseline.glob("schema_context*.json"):
                if sidecar.name == "schema_context.json":
                    continue
                cache[sidecar.name] = sidecar.read_bytes()
            self._baseline_bytes_cache[str(baseline)] = cache
        engine_dir.mkdir(parents=True, exist_ok=True)
        for name, data in cache.items():
            dest = engine_dir / name
            if name != "schema_context.json" and name.startswith("schema_context") and dest.is_file():
                continue
            dest.write_bytes(data)

    def live_handle(self, *, preset: str = "owner_writer", restricted_consumer: bool = False) -> object:
        if self._closed:
            raise RuntimeError("Warm recording pool is closed")
        cache_key = f"{preset}:restricted={restricted_consumer}"
        cached = self._live_handles.get(cache_key)
        if cached is not None:
            return cached
        _reset_sandbox_duckdb_runtime()
        artifacts_dir = self._artifacts_for_preset(preset)
        self._seed_engine_baseline_cached(artifacts_dir=artifacts_dir, preset=preset)
        kwargs: dict[str, object] = {
            "bundle_dir": str(self._bundle_dir),
            "seed_sql": self._seed_sql,
            "connection": self._connection,
            "owns_connection": False,
            "artifacts_dir": artifacts_dir,
            "cleanup_artifacts": False,
        }
        if preset != "owner_writer":
            kwargs["preset"] = preset
        if restricted_consumer:
            kwargs["restricted_consumer"] = True
        handle = self._engine_cls.offline_sandbox(**kwargs)
        self._live_handles[cache_key] = handle
        return handle

    def evict_live_handle(self, *, preset: str = "owner_writer", restricted_consumer: bool = False) -> None:
        """Drop a cached live handle so bundled overrides cannot pollute later recordings."""
        cache_key = f"{preset}:restricted={restricted_consumer}"
        handle = self._live_handles.pop(cache_key, None)
        if handle is not None:
            handle.close()

    def _ephemeral_handle(
        self,
        *,
        preset: str,
        fixtures_file: str,
        provider: str,
        restricted_consumer: bool = False,
    ) -> tuple[object, str]:
        del provider
        verify_artifacts = tempfile.mkdtemp(prefix="aetherdialect_record_verify_")
        _reset_sandbox_duckdb_runtime()
        self._seed_engine_baseline_cached(artifacts_dir=verify_artifacts, preset=preset)
        orig_write = aetherdialect._sandbox._write_sandbox_toml

        def mock_toml(*, fixtures_file: str) -> str:
            return orig_write(fixtures_file=fixtures_file)

        aetherdialect._sandbox._write_sandbox_toml = mock_toml
        kwargs: dict[str, object] = {
            "bundle_dir": str(self._bundle_dir),
            "seed_sql": self._seed_sql,
            "connection": self._connection,
            "owns_connection": False,
            "artifacts_dir": verify_artifacts,
            "cleanup_artifacts": True,
        }
        if preset != "owner_writer":
            kwargs["preset"] = preset
        if restricted_consumer:
            kwargs["restricted_consumer"] = True
        try:
            handle = self._engine_cls.offline_sandbox(**kwargs)
        finally:
            aetherdialect._sandbox._write_sandbox_toml = orig_write
        return handle, verify_artifacts

    def run_live(self, question: str, *, preset: str = "owner_writer", mode: str | None = None) -> object:
        self._pin_schema_literals()
        handle = self.live_handle(preset=preset)
        session_cm = handle.engine.session(mode=mode) if mode else handle.engine.session()
        with session_cm as session:
            return session.accept_until_done(question)

    def run_mock(
        self,
        question: str,
        *,
        preset: str = "owner_writer",
        mode: str | None = None,
        fixtures_file: str,
        restricted_consumer: bool = False,
        apply_overrides: bool = False,
    ) -> tuple[object | None, str]:
        """Run one mock-provider question, flattening handcrafted slot fixtures when needed."""
        from aetherdialect._llm_provider import reset_mock_provider

        self._pin_schema_literals()

        effective_fixtures = fixtures_file
        cleanup_fixtures = False
        try:
            with open(fixtures_file, encoding="utf-8") as f:
                data = json.load(f)
            if "slots" in data:
                flat = []
                for block in data["slots"].values():
                    if isinstance(block, dict) and "fixtures" in block:
                        flat.extend(block["fixtures"])
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                json.dump({"fixtures": flat}, tmp)
                tmp.close()
                effective_fixtures = tmp.name
                cleanup_fixtures = True
        except Exception:
            pass

        handle, artifacts_dir = self._ephemeral_handle(
            preset=preset,
            fixtures_file=effective_fixtures,
            provider="mock",
            restricted_consumer=restricted_consumer,
        )
        prev_provider = EngineConfig.LLM_PROVIDER
        prev_fixtures = EngineConfig.MOCK_FIXTURES_FILE
        EngineConfig.LLM_PROVIDER = "mock"
        EngineConfig.MOCK_FIXTURES_FILE = effective_fixtures
        try:
            if apply_overrides:
                handle.apply_bundled_schema_overrides()
            session_cm = handle.engine.session(mode=mode) if mode else handle.engine.session()
            with session_cm as session:
                step = session.accept_until_done(question)
            return step, ""
        except Exception as exc:
            return None, _short_retry_reason(str(exc), mock=True)
        finally:
            handle.close()
            reset_mock_provider()
            EngineConfig.LLM_PROVIDER = prev_provider
            EngineConfig.MOCK_FIXTURES_FILE = prev_fixtures

            aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)
            if cleanup_fixtures:
                os.remove(effective_fixtures)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self._live_handles.values():
            handle.close()
        self._live_handles.clear()
        try:
            self._connection.close()
        except Exception:
            pass

        for artifacts_dir in (self._live_artifacts_owner, self._live_artifacts_consumer):
            aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)



def _load_scenarios_by_question() -> dict[str, dict[str, object]]:
    if not SCENARIOS_SOURCE.is_file():
        return {}
    payload = json.loads(SCENARIOS_SOURCE.read_text(encoding="utf-8"))
    rows = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    index: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict):
            question = str(row.get("question", "")).strip()
            if question:
                index[question.lower()] = row
    return index


def _load_scenarios() -> list[dict[str, object]]:
    if not SCENARIOS_SOURCE.is_file():
        return []
    payload = json.loads(SCENARIOS_SOURCE.read_text(encoding="utf-8"))
    rows = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _feedback_scenario() -> dict[str, object]:
    for row in _load_scenarios():
        if str(row.get("kind", "")).strip().lower() == "feedback":
            return row
    return {}


def _load_handcrafted_entries() -> list[dict[str, object]]:
    path = HANDCRAFTED_FIXTURES_SOURCE
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _handcrafted_entries_for_question(question: str) -> list[dict[str, object]]:
    target = question.strip()
    if not target:
        return []
    out: list[dict[str, object]] = []
    for row in _load_handcrafted_entries():
        if str(row.get("question", "")).strip() == target:
            out.append(row)
    return out


def _handcrafted_stage_for_system(system: str) -> str:
    if system == INTENT_INTERPRET_SYSTEM:
        return "interpret"
    if system == INTENT_GROUND_SYSTEM:
        return "ground"
    if system == INTENT_COMPOSE_SYSTEM:
        return "compose"
    return "default"


def _clean_generated_paraphrases(question: str, paraphrases: list[str] | None) -> list[str]:
    rows = list(paraphrases or [])
    rows.extend(INLINE_PARAPHRASE_COPY_RULES.get(question, ()))
    question_norm = normalize_question(question)
    out: list[str] = []
    seen: set[str] = set()
    for item in rows:
        text = str(item).strip()
        if not text:
            continue
        norm = normalize_question(text)
        if not norm or norm == question_norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _swap_copy_tokens(text: str, swaps: tuple[tuple[str, str], ...]) -> str:
    out = str(text)
    placeholders: list[tuple[str, str]] = []
    for idx, (left, right) in enumerate(swaps):
        marker = f"__AETHER_SWAP_{idx}__"
        out = out.replace(left, marker)
        placeholders.append((marker, right))
    for left, right in swaps:
        out = out.replace(right, left)
    for marker, right in placeholders:
        out = out.replace(marker, right)
    return out


def _is_param_extraction_fixture_row(row: dict[str, str]) -> bool:
    return (
        str(row.get("task", "")) == "default"
        and PARAM_EXTRACTION_SYSTEM_MARKER in str(row.get("system", "")).lower()
    )


def _param_extraction_forward_rows_from_corpus(
    fixtures: list[dict[str, str]],
    canonical: str,
) -> list[dict[str, str]]:
    needle = canonical.strip()
    if not needle:
        return []
    out: list[dict[str, str]] = []
    for row in fixtures:
        if not _is_param_extraction_fixture_row(row):
            continue
        if needle not in str(row.get("user", "")):
            continue
        out.append(dict(row))
    return out


def _reuse_reverse_param_rows_committed(
    corpus: FixtureCorpus,
    *,
    forward_rows: list[dict[str, str]],
    swaps: tuple[tuple[str, str], ...],
    llm_mod: Any,
) -> bool:
    reverse_rows = _build_reverse_param_extraction_rows(forward_rows, swaps=swaps, llm_mod=llm_mod)
    if not reverse_rows:
        return False
    return all(fixture_key(row) in corpus.seen for row in reverse_rows)


def _build_reverse_param_extraction_rows(
    rows: list[dict[str, str]],
    *,
    swaps: tuple[tuple[str, str], ...],
    llm_mod: Any,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if not _is_param_extraction_fixture_row(row):
            continue
        swapped_user = _swap_copy_tokens(str(row.get("user", "")), swaps)
        swapped_output = _swap_copy_tokens(str(row.get("output_text", "")), swaps)
        out.append(
            {
                "task": str(row.get("task", "")),
                "system": str(row.get("system", "")),
                "user": llm_mod.mock_fixture_user_key(swapped_user),
                "output_text": swapped_output,
            }
        )
    return out


def _recording_handle_for_slot(pool: WarmRecordingPool, slot: RecordingSlot) -> tuple[object, bool, str | None]:
    """Return (handle, apply_overrides, mode) for a recording slot."""
    scenario = _load_scenarios_by_question().get(slot.label.lower(), {})
    mechanism = str(scenario.get("mechanism", ""))
    if mechanism == "bundled_overrides_hide_staff_ssn":
        return pool.live_handle(preset=slot.preset), True, slot.mode
    if mechanism == "schema_validation_failure":
        return pool.live_handle(preset="consumer_reader", restricted_consumer=True), False, "reader"
    return pool.live_handle(preset=slot.preset), False, slot.mode


def _mock_preset_for_slot(slot: RecordingSlot) -> tuple[str, bool, str | None, bool]:
    """Return (preset, restricted_consumer, mode, apply_overrides) for mock replay."""
    scenario = _load_scenarios_by_question().get(slot.label.lower(), {})
    mechanism = str(scenario.get("mechanism", ""))
    if mechanism == "bundled_overrides_hide_staff_ssn":
        return slot.preset, False, slot.mode, True
    if mechanism == "schema_validation_failure":
        return "consumer_reader", True, "reader", False
    return slot.preset, False, slot.mode, False


def _slot_requires_mock_verify(slot: RecordingSlot) -> bool:
    return slot.kind in {"question", "feedback"}


class RecordingSession:
    """Live fixture capture with per-slot buffering and deterministic expectation checks."""

    def __init__(
        self,
        *,
        corpus: FixtureCorpus,
        pool: WarmRecordingPool,
        questions: dict[str, list[str]],
        env: RecordingEnvironment,
    ) -> None:
        self.corpus = corpus
        self.pool = pool
        self.questions = questions
        self.env = env
        self.max_attempts = RECORDING_MAX_ATTEMPTS
        self.failed_slots: list[RecordingSlot] = []
        self.manifest_rows = _load_manifest_rows()
        self.paraphrase_catalog_rows: list[dict[str, object]] = []
        self.paraphrase_seeds_collected: list[tuple[RecordingSlot, object]] = []
        self._llm_mod = aetherdialect._llm_provider
        self._orig_chat = aetherdialect._llm_provider.llm_chat
        self._orig_json = aetherdialect._llm_provider.llm_json
        self._trace_phase: str | None = None
        self._live_trace: list[dict[str, str]] = []
        self._mock_trace: list[dict[str, str]] = []
        self._mock_read_keys: set[tuple[str, str, str]] = set()
        self._set_llm_chat(self._recording_chat)
        self._set_llm_json(self._recording_json)

    def _set_llm_chat(self, hook: Callable[..., str]) -> None:
        self._llm_mod.llm_chat = hook
        for mod_name in self.env.llm_patch_modules:
            mod = __import__(mod_name, fromlist=["llm_chat"])
            mod.llm_chat = hook

    def _set_llm_json(self, hook: Callable[..., dict[str, Any]]) -> None:
        self._llm_mod.llm_json = hook
        for mod_name in self.env.llm_patch_modules:
            mod = __import__(mod_name, fromlist=["llm_json"])
            if hasattr(mod, "llm_json"):
                mod.llm_json = hook

    @contextmanager
    def _patched_handcrafted_entries(self, question: str) -> Any:
        rows = _handcrafted_entries_for_question(question)
        if not rows:
            yield False
            return
        call_counts: dict[tuple[str, str], int] = {}

        def hook(
            system: str,
            user: str,
            max_retries: int = 3,
            timeout: Any = None,
            task: str = "default",
            **kwargs: object,
        ) -> str:
            del kwargs
            stage = _handcrafted_stage_for_system(system)
            for row in rows:
                if str(row.get("task", "") or "").strip() != task:
                    continue
                if str(row.get("stage", "") or "").strip() != stage:
                    continue
                key = (task, stage)
                next_attempt = call_counts.get(key, 0) + 1
                wanted = row.get("attempt")
                if wanted is not None and int(wanted) != next_attempt:
                    continue
                call_counts[key] = next_attempt
                result = json.dumps(row.get("response", {}), ensure_ascii=False)
                user_for_llm = self._llm_mod._llm_user_text_without_sensitivity_classification(user)
                user_key = self._llm_mod.mock_fixture_user_key(user_for_llm)
                lookup_key = (task, system, user_key)
                self.corpus.record(task=task, system=system, user_key=user_key, output_text=result)
                if self._trace_phase == "mock":
                    self._mock_read_keys.add(lookup_key)
                return result
            return self._recording_chat(
                system,
                user,
                max_retries=max_retries,
                timeout=timeout,
                task=task,
            )

        self._set_llm_chat(hook)
        try:
            yield True
        finally:
            self._set_llm_chat(self._recording_chat)

    def _trace_key(self, task: str, system: str, user_key: str) -> dict[str, str]:
        return {
            "task": task,
            "system_id": _intent_system_label(system),
            "user_sha": hashlib.sha256(user_key.encode("utf-8")).hexdigest()[:16],
            "user_head": user_key[:160],
        }

    def _recording_chat(
        self,
        system: str,
        user: str,
        max_retries: int = 3,
        timeout: Any = None,
        task: str = "default",
        **kwargs: object,
    ) -> str:
        del kwargs
        if timeout is None:
            timeout = self._llm_mod._DEFAULT_LLM_CHAT_TIMEOUT
        user_for_llm = self._llm_mod._llm_user_text_without_sensitivity_classification(user)
        user_key = self._llm_mod.mock_fixture_user_key(user_for_llm)
        lookup_key = (task, system, user_key)
        if self._trace_phase == "mock":
            entry = self._trace_key(task, system, user_key)
            try:
                result = self._orig_chat(system, user, max_retries=max_retries, timeout=timeout, task=task)
            except Exception as exc:
                entry["result"] = f"MISS:{exc}"
                self._mock_trace.append(entry)
                raise
            entry["result"] = "hit:" + hashlib.sha256(result.encode("utf-8")).hexdigest()[:16]
            self._mock_trace.append(entry)
            self._mock_read_keys.add(lookup_key)
            return result
        result = self._orig_chat(system, user, max_retries=max_retries, timeout=timeout, task=task)
        self.corpus.record(task=task, system=system, user_key=user_key, output_text=result)
        if self._trace_phase == "live":
            entry = self._trace_key(task, system, user_key)
            entry["result"] = "live:" + hashlib.sha256(result.encode("utf-8")).hexdigest()[:16]
            self._live_trace.append(entry)
        return result

    def _recording_json(
        self,
        system: str,
        user: str,
        *,
        retries: int = 3,
        task: str = "default",
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        user_for_llm = self._llm_mod._llm_user_text_without_sensitivity_classification(user)
        user_key = self._llm_mod.mock_fixture_user_key(user_for_llm)
        lookup_key = (task, system, user_key)
        if self._trace_phase == "mock":
            entry = self._trace_key(task, system, user_key)
            try:
                result = self._orig_json(system, user, retries=retries, task=task)
            except Exception as exc:
                entry["result"] = f"MISS:{exc}"
                self._mock_trace.append(entry)
                raise
            output_text = json.dumps(result, ensure_ascii=False)
            entry["result"] = "hit:" + hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16]
            self._mock_trace.append(entry)
            self._mock_read_keys.add(lookup_key)
            return result
        result = self._orig_json(system, user, retries=retries, task=task)
        output_text = json.dumps(result, ensure_ascii=False)
        self.corpus.record(task=task, system=system, user_key=user_key, output_text=output_text)
        if self._trace_phase == "live":
            entry = self._trace_key(task, system, user_key)
            entry["result"] = "live:" + hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16]
            self._live_trace.append(entry)
        return result

    def _run_live_slot(self, slot: RecordingSlot) -> tuple[object | None, str]:
        if slot.kind == "question":
            preset, restricted_consumer, mode, apply_overrides = _mock_preset_for_slot(slot)
            handle, artifacts_dir = self.pool._ephemeral_handle(
                preset=preset,
                fixtures_file=str(FIXTURES_PATH),
                provider="openai",
                restricted_consumer=restricted_consumer,
            )
            try:
                if apply_overrides:
                    handle.apply_bundled_schema_overrides()
                session_cm = handle.engine.session(mode=mode) if mode else handle.engine.session()
                try:
                    with session_cm as session:
                        step = session.accept_until_done(slot.label)
                except Exception as exc:
                    return None, str(exc)
                ok, detail = _check_slot_recording(step, slot)
                return step, ("" if ok else detail)
            finally:
                handle.close()
                aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
                shutil.rmtree(artifacts_dir, ignore_errors=True)
        if slot.kind == "feedback":
            return self._run_live_feedback(slot)
        if slot.kind == "space":
            return self._run_live_space(slot)
        return None, f"unknown slot kind {slot.kind!r}"

    def _run_live_space(self, slot: RecordingSlot) -> tuple[object | None, str]:
        from aetherdialect._contracts_base import SpaceContext

        handle = self.pool.live_handle(preset="owner_writer")
        notes_path = STAGING / "sandbox_space_catalog_notes.txt"
        notes_arg = str(notes_path) if notes_path.is_file() else None

        try:
            catalog = SpaceContext(
                tables=frozenset({"item", "film", "category", "item_category"}),
                columns=frozenset(),
            )
            handle.engine.aetherspace("catalog", space_context=catalog, notes_file=notes_arg)
            return handle.engine, ""
        except Exception as exc:
            return None, str(exc)

    def _run_live_feedback(self, slot: RecordingSlot) -> tuple[object | None, str]:
        scenario = _feedback_scenario()
        anchor = str(scenario.get("anchor_question", "")).strip()
        allowed_rejection = str(scenario.get("allowed_rejection_text", "")).strip()
        if not anchor:
            anchor = _feedback_anchor_question(self.questions)
        if allowed_rejection and slot.label.strip() != allowed_rejection:
            return None, f"feedback slot label must match scenario allowed_rejection_text ({allowed_rejection!r})"
        handle = self.pool.live_handle(preset="owner_writer")
        try:
            with handle.engine.session() as session:
                step = session.ask(anchor)
                if not step.done and step.reply_shape == "yes_no":
                    step = session.step("n")
                if not step.done and step.reply_shape == "free_text":
                    step = session.step(slot.label)
                while not step.done and step.reply_shape == "yes_no":
                    step = session.step("y")
        except Exception as exc:
            return None, str(exc)
        if step.done:
            return step, ""
        err = str(getattr(step, "error", "") or step.message or "feedback flow incomplete")
        return step, err

    @contextmanager
    def _template_match_enabled(self) -> Any:
        saved_pipeline_match = aetherdialect._pipeline.match_question_level_template_reuse
        saved_main_match = aetherdialect._main_execution.match_question_level_template_reuse
        saved_live_match = aetherdialect._live_testing.match_question_level_template_reuse
        aetherdialect._pipeline.match_question_level_template_reuse = self.env.orig_template_match
        aetherdialect._main_execution.match_question_level_template_reuse = self.env.orig_template_match
        aetherdialect._live_testing.match_question_level_template_reuse = self.env.orig_template_match
        try:
            yield
        finally:
            aetherdialect._pipeline.match_question_level_template_reuse = saved_pipeline_match
            aetherdialect._main_execution.match_question_level_template_reuse = saved_main_match
            aetherdialect._live_testing.match_question_level_template_reuse = saved_live_match

    def _verify_mock_feedback(self, slot: RecordingSlot, fixtures_file: str) -> tuple[bool, str]:
        scenario = _feedback_scenario()
        anchor = str(scenario.get("anchor_question", "")).strip()
        allowed_rejection = str(scenario.get("allowed_rejection_text", "")).strip()
        if not anchor:
            anchor = _feedback_anchor_question(self.questions)
        if allowed_rejection and slot.label.strip() != allowed_rejection:
            return False, f"feedback slot label must match scenario allowed_rejection_text ({allowed_rejection!r})"
        handle, artifacts_dir = self.pool._ephemeral_handle(
            preset="owner_writer",
            fixtures_file=fixtures_file,
            provider="mock",
        )
        prev_provider = EngineConfig.LLM_PROVIDER
        prev_fixtures = EngineConfig.MOCK_FIXTURES_FILE
        EngineConfig.LLM_PROVIDER = "mock"
        EngineConfig.MOCK_FIXTURES_FILE = fixtures_file
        try:
            with handle.engine.session() as session:
                step = session.ask(anchor)
                if not step.done and step.reply_shape == "yes_no":
                    step = session.step("n")
                if not step.done and step.reply_shape == "free_text":
                    step = session.step(slot.label)
                while not step.done and step.reply_shape == "yes_no":
                    step = session.step("y")
            if not step.done:
                return False, "feedback flow incomplete"
            return True, ""
        except Exception as exc:
            return False, _short_retry_reason(str(exc), mock=True)
        finally:
            handle.close()
            reset_mock_provider()
            EngineConfig.LLM_PROVIDER = prev_provider
            EngineConfig.MOCK_FIXTURES_FILE = prev_fixtures
            aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)

    def _verify_mock_slot(self, slot: RecordingSlot) -> tuple[bool, str]:
        """Replay the slot against the full in-progress corpus exactly like validate/pack."""
        if not _slot_requires_mock_verify(slot):
            return True, ""
        self._mock_trace = []
        self._mock_read_keys = set()
        prev_phase = self._trace_phase
        self._trace_phase = "mock"
        fixtures_file = str(FIXTURES_PATH)
        try:
            if slot.kind == "feedback":
                return self._verify_mock_feedback(slot, fixtures_file)
            for preset, restricted_consumer, mode, apply_overrides, profile, tier in _mock_verify_targets_for_slot(
                slot,
            ):
                logs: list[str] = []
                step: object | None = None
                with self._patched_handcrafted_entries(slot.label):
                    step, err, logs = run_with_pipeline_capture(
                        lambda preset=preset, restricted_consumer=restricted_consumer, mode=mode, apply_overrides=apply_overrides: self.pool.run_mock(
                            slot.label,
                            preset=preset,
                            mode=mode,
                            fixtures_file=fixtures_file,
                            restricted_consumer=restricted_consumer,
                            apply_overrides=apply_overrides,
                        ),
                    )
                if err:
                    append_failure_trace(
                        build_session_step_trace(
                            scenario_id=slot_id_for(slot),
                            question=slot.label,
                            step=step,
                            error=err,
                            captured_logs=logs,
                        ),
                        RECORDING_RESULTS_PATH,
                    )
                    return False, f"mock replay ({tier}): {err}"
                if step is None:
                    append_failure_trace(
                        build_session_step_trace(
                            scenario_id=slot_id_for(slot),
                            question=slot.label,
                            step=None,
                            error="mock replay failed",
                            captured_logs=logs,
                        ),
                        RECORDING_RESULTS_PATH,
                    )
                    return False, f"mock replay ({tier}): failed"
                ok, detail = _check_slot_recording(step, slot, profile=profile, tier=tier)
                if ok:
                    continue
                append_failure_trace(
                    build_session_step_trace(
                        scenario_id=slot_id_for(slot),
                        question=slot.label,
                        step=step,
                        error=detail or "expectation not met",
                        captured_logs=logs,
                    ),
                    RECORDING_RESULTS_PATH,
                )
                return False, f"mock replay ({tier}): {detail or 'expectation not met'}"
            return True, ""
        finally:
            self._trace_phase = prev_phase

    def record_slot(self, slot: RecordingSlot) -> tuple[bool, str, int, object | None]:
        last_detail = ""
        attempts_used = 0
        last_step_result: object | None = None
        sid = slot_id_for(slot)

        for attempt in range(1, self.max_attempts + 1):
            attempts_used = attempt
            self.corpus.start_slot()

            auto = ["y"]
            if slot.kind == "feedback":
                auto = ["n", slot.label, "y"]

            self._live_trace = []
            self._trace_phase = "live"
            try:
                with self._patched_handcrafted_entries(slot.label) as handcrafted_active:
                    with pipeline_capture(auto_responses=auto) as capture:
                        _step, live_detail = self._run_live_slot(slot)
            finally:
                self._trace_phase = None

            if _step is not None:
                last_step_result = _step

            if live_detail:
                last_detail = live_detail
                self.corpus.discard_slot()
                _append_replay_trace(
                    slot_id=sid,
                    label=slot.label,
                    tier=slot.tier,
                    attempt=attempt,
                    outcome="VARIANCE",
                    live_rows=self._live_trace,
                    mock_rows=[],
                    detail=live_detail,
                )
                if _BUILD_VERBOSE:
                    corpus_message(
                        f"  retry {attempt}/{self.max_attempts} [{slot.tier}] {slot.label[:60]} "
                        f"({_short_retry_reason(live_detail)})",
                    )

                if _step is not None:
                    step_res = StepResult(
                        scenario_id=sid,
                        question=slot.label,
                        status="failed",
                        error=live_detail,
                        captured_logs=capture.get("logs", []),
                        duration_seconds=0.0,
                    )
                    append_failure_trace(step_res, RECORDING_RESULTS_PATH)

                if not _recording_error_retryable(live_detail):
                    break
                continue

            buffered_keys = self.corpus.buffered_keys()
            snapshot = self.corpus.snapshot()
            pre_frozen = set(self.corpus.committed_keys)
            self.corpus.commit_slot(freeze=False)
            mock_ok, mock_detail = self._verify_mock_slot(slot)
            if mock_ok:
                orphan_keys = (buffered_keys - self._mock_read_keys) - pre_frozen
                self.corpus.rollback_keys(orphan_keys)
                self.corpus.committed_keys |= buffered_keys & self._mock_read_keys
                _append_replay_trace(
                    slot_id=sid,
                    label=slot.label,
                    tier=slot.tier,
                    attempt=attempt,
                    outcome="COMMIT",
                    live_rows=self._live_trace,
                    mock_rows=self._mock_trace,
                )
                _upsert_manifest_row(
                    self.manifest_rows,
                    {
                        "slot_id": sid,
                        "tier": slot.tier,
                        "kind": slot.kind,
                        "label": slot.label,
                        "committed": True,
                        "attempts": attempts_used,
                        "detail": "handcrafted" if handcrafted_active else "",
                    },
                )
                write_recording_manifest(self.manifest_rows)
                return True, "", attempts_used, last_step_result

            self.corpus.restore(snapshot)
            last_detail = f"determinism-divergence: {mock_detail or 'mock replay failed'}"
            _append_replay_trace(
                slot_id=sid,
                label=slot.label,
                tier=slot.tier,
                attempt=attempt,
                outcome="DETERMINISM-DIVERGENCE",
                live_rows=self._live_trace,
                mock_rows=self._mock_trace,
                detail=last_detail,
            )
            corpus_message(
                f"  [DETERMINISM BUG] [{slot.tier}] {slot.label[:60]} passed live but failed mock "
                f"replay of its own cached fixtures ({_short_retry_reason(mock_detail, mock=True)}). "
                f"See {SANDBOX_REPLAY_TRACE_PATH}",
            )
            if _step is not None:
                step_res = StepResult(
                    scenario_id=sid,
                    question=slot.label,
                    status="failed",
                    error=last_detail,
                    captured_logs=capture.get("logs", []),
                    duration_seconds=0.0,
                )
                append_failure_trace(step_res, RECORDING_RESULTS_PATH)
            break

        self.failed_slots.append(slot)
        _upsert_manifest_row(
            self.manifest_rows,
            {
                "slot_id": sid,
                "tier": slot.tier,
                "kind": slot.kind,
                "label": slot.label,
                "committed": False,
                "attempts": attempts_used,
                "detail": last_detail,
            },
        )
        write_recording_manifest(self.manifest_rows)
        return False, last_detail, attempts_used, last_step_result

    def record_bundled_paraphrases(
        self,
        recorded_slots: list[tuple[RecordingSlot, object]],
        *,
        require_seeds: bool = False,
        existing_rows: list[dict[str, object]] | None = None,
    ) -> tuple[bool, str]:
        if not recorded_slots:
            if require_seeds:
                return False, "no paraphrase seeds"
            if existing_rows:
                write_sandbox_catalog_file(target_dir=STAGING, paraphrase_pairs=existing_rows)
                self.paraphrase_catalog_rows = list(existing_rows)
            return True, ""
        merged_by_canonical: dict[str, dict[str, object]] = {}
        for row in existing_rows or []:
            if isinstance(row, dict):
                canonical = str(row.get("canonical", "")).strip()
                if canonical:
                    merged_by_canonical[canonical] = row
        new_rows: list[dict[str, object]] = []
        for slot, step in recorded_slots:
            intent_summary = getattr(step, "intent_summary", None)
            tables = [str(item) for item in getattr(intent_summary, "tables", ()) if str(item).strip()]
            if not tables:
                return False, f"missing tables for paraphrase generation on {slot.label!r}"
            preset, restricted_consumer, _mode, apply_overrides = _mock_preset_for_slot(slot)
            handle, artifacts_dir = self.pool._ephemeral_handle(
                preset=preset,
                fixtures_file=str(FIXTURES_PATH),
                provider="openai",
                restricted_consumer=restricted_consumer,
            )
            self.corpus.start_slot()
            try:
                if apply_overrides:
                    handle.apply_bundled_schema_overrides()
                try:
                    raw = generate_paraphrases_of_seed_question(
                        slot.label,
                        handle.engine._schema_graph,
                        tables,
                    )
                    paraphrases = _clean_generated_paraphrases(slot.label, raw)
                    self.corpus.commit_slot()
                finally:
                    handle.close()
                    aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
                    shutil.rmtree(artifacts_dir, ignore_errors=True)
            except Exception as exc:
                self.corpus.discard_slot()
                return False, str(exc)
            new_rows.append({"canonical": slot.label, "paraphrases": paraphrases})
            for paraphrase in paraphrases:
                swaps = INLINE_REUSE_PARAM_COPY_RULES.get((slot.label, paraphrase))
                if swaps is None:
                    continue
                ok, detail = self.record_inline_reuse_param_fixtures(slot.label, paraphrase, swaps=swaps)
                if not ok:
                    return False, detail
        for row in new_rows:
            merged_by_canonical[str(row["canonical"])] = row
        final_rows = list(merged_by_canonical.values())
        self.paraphrase_catalog_rows = final_rows
        write_sandbox_catalog_file(target_dir=STAGING, paraphrase_pairs=final_rows)
        return True, ""

    def record_reuse_param_fixtures(self) -> tuple[bool, str]:
        for key, swaps in INLINE_REUSE_PARAM_COPY_RULES.items():
            canonical, paraphrase = key
            ok, detail = self.record_inline_reuse_param_fixtures(canonical, paraphrase, swaps=swaps)
            if not ok:
                return False, detail
        return True, ""

    def collect_paraphrase_seeds(self, slots: list[RecordingSlot]) -> list[tuple[RecordingSlot, object]]:
        """Collect intent tables for paraphrase generation from committed mock fixtures."""
        seeds: list[tuple[RecordingSlot, object]] = []
        for slot in slots:
            preset, restricted_consumer, mode, apply_overrides = _mock_preset_for_slot(slot)
            step, err = self.pool.run_mock(
                slot.label,
                preset=preset,
                mode=mode,
                fixtures_file=str(FIXTURES_PATH),
                restricted_consumer=restricted_consumer,
                apply_overrides=apply_overrides,
            )
            if step is None or err:
                continue
            intent_summary = getattr(step, "intent_summary", None)
            tables = [str(item) for item in getattr(intent_summary, "tables", ()) if str(item).strip()]
            if not tables:
                continue
            seeds.append((slot, step))
        return seeds

    def record_inline_reuse_param_fixtures(
        self,
        canonical: str,
        paraphrase: str,
        *,
        swaps: tuple[tuple[str, str], ...],
    ) -> tuple[bool, str]:
        canonical = canonical.strip()
        paraphrase = paraphrase.strip()
        pe_forward_existing = _param_extraction_forward_rows_from_corpus(self.corpus.fixtures, canonical)
        if _reuse_reverse_param_rows_committed(
            self.corpus,
            forward_rows=pe_forward_existing,
            swaps=swaps,
            llm_mod=self._llm_mod,
        ):
            return True, ""

        if not _paraphrase_gatekeeper_ready(self.corpus, paraphrase):
            handle, artifacts_dir = self.pool._ephemeral_handle(
                preset="owner_writer",
                fixtures_file=str(FIXTURES_PATH),
                provider="openai",
            )
            self.corpus.start_slot()
            try:
                with handle.engine.session() as session:
                    step = session.accept_until_done(paraphrase)
                    if getattr(step, "error", None):
                        self.corpus.discard_slot()
                        return False, f"reuse standalone capture failed for {paraphrase!r}: {step.error}"
                self.corpus.commit_slot()
            except Exception as exc:
                self.corpus.discard_slot()
                return False, str(exc)
            finally:
                handle.close()
                aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
                shutil.rmtree(artifacts_dir, ignore_errors=True)

        pe_forward_existing = _param_extraction_forward_rows_from_corpus(self.corpus.fixtures, canonical)
        if _reuse_reverse_param_rows_committed(
            self.corpus,
            forward_rows=pe_forward_existing,
            swaps=swaps,
            llm_mod=self._llm_mod,
        ):
            return True, ""

        self.pool.evict_live_handle(preset="owner_writer")
        handle, artifacts_dir = self.pool._ephemeral_handle(
            preset="owner_writer",
            fixtures_file=str(FIXTURES_PATH),
            provider="openai",
        )
        forward_rows: list[dict[str, str]] = []
        try:
            with self._template_match_enabled():
                with handle.engine.session() as session:
                    self.corpus.start_slot()
                    session.accept_until_done(canonical)
                    step = session.accept_until_done(paraphrase)
                    if getattr(step, "error", None):
                        self.corpus.discard_slot()
                        return False, f"reuse capture failed for {canonical!r} -> {paraphrase!r}: {step.error}"
                    forward_rows = self.corpus.collapsed_slot_fixtures()
                    pe_forward = [row for row in forward_rows if _is_param_extraction_fixture_row(row)]
                    if not pe_forward:
                        self.corpus.discard_slot()
                        return False, f"missing parameter extraction fixture for {canonical!r} -> {paraphrase!r}"
                    self.corpus.commit_slot()
        except Exception as exc:
            self.corpus.discard_slot()
            return False, str(exc)
        finally:
            handle.close()
            aetherdialect._sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)

        pe_forward = [row for row in forward_rows if _is_param_extraction_fixture_row(row)]
        if not pe_forward:
            pe_forward = _param_extraction_forward_rows_from_corpus(self.corpus.fixtures, canonical)
        if _reuse_reverse_param_rows_committed(
            self.corpus,
            forward_rows=pe_forward,
            swaps=swaps,
            llm_mod=self._llm_mod,
        ):
            return True, ""
        reverse_rows = _build_reverse_param_extraction_rows(pe_forward, swaps=swaps, llm_mod=self._llm_mod)
        if not reverse_rows:
            return False, f"missing parameter extraction fixture for {canonical!r} -> {paraphrase!r}"
        self.corpus.start_slot()
        try:
            for row in reverse_rows:
                self.corpus.record(
                    task=str(row.get("task", "")),
                    system=str(row.get("system", "")),
                    user_key=str(row.get("user", "")),
                    output_text=str(row.get("output_text", "")),
                )
            self.corpus.commit_slot()
        except Exception:
            self.corpus.discard_slot()
            raise
        return True, ""

    def record_migration_demo_fixtures(self) -> tuple[bool, str]:
        from aetherdialect._contracts_base import MigrationPendingError
        from aetherdialect._dialect_sqlglot_engines import create_duckdb_sqlalchemy_engine
        from aetherdialect._sandbox import (
            _load_memory_connection,
            _owner_writer_schema_context,
            _post_migration_seed_sql,
        )

        demo_root = STAGING / "migration_demo"
        artifacts_src = demo_root / "artifacts_v1"
        map_path = demo_root / "schema_migration_map.json"
        seed_path = STAGING / "rental_shop_seed.sql"
        if not artifacts_src.is_dir() or not map_path.is_file() or not seed_path.is_file():
            return False, f"migration demo assets incomplete under {demo_root}"
        work = Path(tempfile.mkdtemp(prefix="aetherdialect_record_migration_"))
        config_file = ""
        self.corpus.start_slot()
        try:
            post_sql = work / "rental_shop_post_migration.sql"
            post_sql.write_text(_post_migration_seed_sql(seed_path, map_path), encoding="utf-8")
            artifacts_dir = str(work / "artifacts")
            shutil.copytree(artifacts_src, artifacts_dir)
            connection = _load_memory_connection(str(post_sql))
            execution_engine = create_duckdb_sqlalchemy_engine(connection)
            notes_file = STAGING / "rental_shop_notes.txt"
            notes_arg = str(notes_file) if notes_file.is_file() else None
            sql_file = STAGING / "rental_shop.sql"
            sql_arg = str(sql_file) if sql_file.is_file() else None
            schema_context = _owner_writer_schema_context(notes_file=notes_arg, sql_file=sql_arg)
            config_file = self.env.openai_toml(fixtures_file=str(FIXTURES_PATH))
            try:
                self.pool._engine_cls(
                    schema_context,
                    artifacts_dir=artifacts_dir,
                    config_file=config_file,
                    execution_engine=execution_engine,
                    native_connection=connection,
                    role="owner",
                )
            except MigrationPendingError:
                pass
            t2s = self.pool._engine_cls.apply_migration_map(
                str(map_path),
                engine_context=schema_context,
                artifacts_dir=artifacts_dir,
                config_file=config_file,
                execution_engine=execution_engine,
                native_connection=connection,
                role="owner",
            )
            t2s._sandbox_mode = True
            with t2s.session() as session:
                session.accept_until_done(_post_migration_record_question(self.questions))
            self.corpus.commit_slot()
            return True, ""
        except Exception as exc:
            self.corpus.discard_slot()
            return False, str(exc)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            if config_file:
                try:
                    Path(config_file).unlink(missing_ok=True)
                except OSError:
                    pass

    def record_all(self, *, slots: list[RecordingSlot] | None = None, record_reuse_pairs: bool = False) -> bool:
        import live_tests.conftest as lt_conftest

        del record_reuse_pairs
        slot_list = slots if slots is not None else iter_recording_slots(self.questions)
        total = len(slot_list)
        self.paraphrase_seeds_collected = []
        import aetherdialect.aetherdialect

        original_log_sink = aetherdialect.aetherdialect._init_log_sink
        aetherdialect.aetherdialect._init_log_sink = lambda _line: None
        _begin_eval_results(RECORDING_RESULTS_PATH)
        _reset_replay_trace()
        try:
            if _BUILD_VERBOSE:
                corpus_message(f"Recording sandbox fixtures ({total} slots)...")
            for idx, slot in enumerate(slot_list, 1):
                ok, detail, attempts, step_result = self.record_slot(slot)
                label = slot.label[:70]
                if ok:
                    suffix = f" — retry {attempts}/{self.max_attempts}" if attempts > 1 else ""
                    corpus_message(f"[recording] {idx}/{total} OK [{slot.tier}] {label}{suffix}")
                    if (
                        slot.kind == "question"
                        and slot.tier in PARAPHRASE_SOURCE_TIERS
                        and slot.preset == "owner_writer"
                        and slot.mode in (None, "writer")
                        and _paraphrase_eligible_question(slot.label, kind=slot.kind)
                        and step_result is not None
                    ):
                        self.paraphrase_seeds_collected.append((slot, step_result))
                else:
                    reason = _short_retry_reason(detail)
                    retry_note = f" — retry {attempts}/{self.max_attempts}" if attempts > 1 else ""
                    corpus_message(
                        f"[recording] {idx}/{total} FAIL [{slot.tier}] {label}{retry_note} ({reason})",
                    )
                    if step_result is not None:
                        lt_conftest._append_failure_trace(step_result)
        finally:
            aetherdialect.aetherdialect._init_log_sink = original_log_sink
        slot_failures = bool(self.failed_slots)
        if slot_failures:
            lines = [f"  [{slot.tier}] {slot.label}" for slot in self.failed_slots]
            corpus_message(
                f"{len(self.failed_slots)} slot(s) failed recording. "
                f"See {RECORDING_RESULTS_PATH}:\n" + "\n".join(lines),
            )
        return not self.failed_slots

    def close(self) -> None:
        self._llm_mod.llm_chat = self._orig_chat
        self._llm_mod.llm_json = self._orig_json


def pack_bundled_aetherspace_snapshots(pool: WarmRecordingPool) -> None:
    """Copy recorded aetherspace snapshots into staging for data.zip."""
    from aetherdialect._constants import AETHERSPACES_SEGMENT
    from aetherdialect._contracts_base import SpaceContext

    handle = pool.live_handle(preset="owner_writer")
    artifacts_root = Path(str(handle.engine._artifacts_dir))
    src_dir = artifacts_root / AETHERSPACES_SEGMENT
    if not src_dir.is_dir() or not any(src_dir.glob("*.json")):
        notes_path = STAGING / "sandbox_space_catalog_notes.txt"
        notes_arg = str(notes_path) if notes_path.is_file() else None
        catalog = SpaceContext(
            tables=frozenset({"item", "film", "category", "item_category"}),
            columns=frozenset(),
        )
        handle.engine.aetherspace("catalog", space_context=catalog, notes_file=notes_arg)
        src_dir = artifacts_root / AETHERSPACES_SEGMENT
    dest = STAGING / "artifacts_baseline" / AETHERSPACES_SEGMENT
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(src_dir.glob("*.json")):
        shutil.copy2(path, dest / path.name)
        copied += 1
    if not copied:
        raise RuntimeError(f"no aetherspace snapshots found under {src_dir}")
    verbose_message(f"Packed {copied} aetherspace snapshot(s) from {src_dir} -> {dest}")


def finalize_recording_tail(
    session: RecordingSession,
    pool: WarmRecordingPool,
    corpus: FixtureCorpus,
    *,
    paraphrase_seeds: list[tuple[RecordingSlot, object]] | None = None,
    capture_paraphrases: bool = False,
    require_paraphrase_seeds: bool = False,
    capture_reuse: bool = False,
    capture_aetherspace: bool = False,
    capture_migration: bool = False,
) -> bool:
    ok = True
    if capture_paraphrases:
        existing_rows: list[dict[str, object]] | None = None
        catalog_path = STAGING / "sandbox_catalog.json"
        if catalog_path.is_file():
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            pairs = payload.get("paraphrase_pairs")
            if isinstance(pairs, list):
                existing_rows = [row for row in pairs if isinstance(row, dict)]
        paraphrase_ok, detail = session.record_bundled_paraphrases(
            paraphrase_seeds or [],
            require_seeds=require_paraphrase_seeds,
            existing_rows=existing_rows,
        )
        if not paraphrase_ok:
            corpus_message(f"[build] paraphrase capture failed: {detail}")
            ok = False
    if capture_reuse:
        reuse_ok, detail = session.record_reuse_param_fixtures()
        if not reuse_ok:
            corpus_message(f"[build] reuse capture failed: {detail}")
            ok = False
    if capture_aetherspace:
        pack_bundled_aetherspace_snapshots(pool)
    if capture_migration:
        migration_ok, detail = session.record_migration_demo_fixtures()
        if not migration_ok:
            corpus_message(f"[build] migration fixture capture failed: {detail}")
            ok = False
    write_build_fingerprint()
    corpus.flush()
    print(
        f"Fixture corpus at {FIXTURES_PATH} now has {len(corpus.fixtures)} entr"
        f"{'y' if len(corpus.fixtures) == 1 else 'ies'} (total committed rows, not per-slot delta)",
        flush=True,
    )
    return ok


def record_corpus(record_reuse_pairs: bool = False) -> bool:
    del record_reuse_pairs
    ensure_schema_literals()
    ensure_interpret_domain()
    pin_staging_mock_fixture_keys()
    _assert_pinning_parity()
    literals_path = STAGING / SANDBOX_SCHEMA_LITERALS_FILENAME
    if not literals_path.is_file():
        corpus_message(
            f"[build] warning: {SANDBOX_SCHEMA_LITERALS_FILENAME} missing; "
            "run build_artifacts_baseline() for stable mock keys",
        )
    questions = load_staging_questions()
    corpus = FixtureCorpus(FIXTURES_PATH)
    env = prepare_recording_environment()
    pool = WarmRecordingPool(STAGING)
    session = RecordingSession(
        corpus=corpus,
        pool=pool,
        questions=questions,
        env=env,
    )
    with _staging_sandbox_bundle():
        try:
            session.record_all()
            manifest = _load_recording_manifest()
            missing_para = _missing_paraphrase_canonicals(questions, manifest)
            need_para = bool(missing_para)
            need_reuse = not _reuse_fixtures_ready(corpus)
            need_aetherspace = not _aetherspace_snapshots_ready()
            need_migration = not _migration_fixtures_ready(corpus)
            paraphrase_seeds: list[tuple[RecordingSlot, object]] | None = None
            if need_para:
                paraphrase_seeds, seed_detail = _paraphrase_seeds_for_missing(
                    session,
                    missing_para,
                    session.paraphrase_seeds_collected,
                )
                if seed_detail:
                    corpus_message(f"[build] paraphrase capture failed: {seed_detail}")
                    need_para = False
            finalize_recording_tail(
                session,
                pool,
                corpus,
                paraphrase_seeds=paraphrase_seeds,
                capture_paraphrases=need_para,
                require_paraphrase_seeds=need_para,
                capture_reuse=need_reuse,
                capture_aetherspace=need_aetherspace,
                capture_migration=need_migration,
            )
        finally:
            session.close()
            pool.close()
            teardown_recording_environment(env)
    ready, reasons = _recording_pipeline_ready(questions=questions, corpus=corpus)
    if not ready:
        for reason in reasons:
            corpus_message(f"[build] recording incomplete: {reason}")
    return ready


def _fingerprint_file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_build_fingerprint(*, staging_dir: Path = STAGING) -> dict[str, object]:
    """Hash staging inputs that should stay stable across fixture repair."""
    file_paths = (
        staging_dir / "rental_shop_seed.sql",
        staging_dir / "rental_shop.sql",
        staging_dir / "rental_shop_views.sql",
        staging_dir / "rental_shop_notes.txt",
        staging_dir / "questions.txt",
        staging_dir / "schema_overrides_demo.json",
        staging_dir / SANDBOX_SCHEMA_LITERALS_FILENAME,
        staging_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME,
        staging_dir / "migration_demo" / "schema_migration_map.json",
    )
    files: dict[str, str | None] = {path.name: _fingerprint_file_digest(path) for path in file_paths}
    baseline_root = staging_dir / "artifacts_baseline"
    baseline_files: dict[str, str] = {}
    if baseline_root.is_dir():
        for path in sorted(baseline_root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(staging_dir).as_posix()
                baseline_files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"files": files, "baseline_files": baseline_files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"fingerprint": digest, **payload}


def write_build_fingerprint(*, staging_dir: Path = STAGING) -> None:
    payload = compute_build_fingerprint(staging_dir=staging_dir)
    write_text_atomic(
        staging_dir / "sandbox_build_fingerprint.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def staging_fingerprint_matches(*, staging_dir: Path = STAGING) -> bool:
    stored_path = staging_dir / "sandbox_build_fingerprint.json"
    if not stored_path.is_file():
        return False
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    current = compute_build_fingerprint(staging_dir=staging_dir)
    return str(stored.get("fingerprint", "")) == str(current.get("fingerprint", ""))


def repair_failing_slots() -> bool:
    _ensure_script_import_paths()
    if not STAGING.is_dir():
        raise SystemExit(f"Missing staging directory: {STAGING}")
    if not staging_fingerprint_matches():
        raise SystemExit(
            "Staging fingerprint mismatch or missing; run full scripts/sandbox_corpus.py before --repair.",
        )
    ensure_schema_literals()
    ensure_interpret_domain()
    pin_staging_mock_fixture_keys()
    _assert_pinning_parity()
    literals_path = STAGING / SANDBOX_SCHEMA_LITERALS_FILENAME
    if not literals_path.is_file():
        corpus_message(
            f"[repair] warning: {SANDBOX_SCHEMA_LITERALS_FILENAME} missing; "
            "run build_artifacts_baseline() for stable mock keys",
        )
    questions = load_staging_questions()
    manifest = _load_recording_manifest()
    committed = committed_slot_ids(manifest)
    all_slots = iter_recording_slots(questions)
    slots_to_repair = [slot for slot in all_slots if slot_id_for(slot) not in committed]
    corpus = FixtureCorpus(FIXTURES_PATH)
    missing_para = _missing_paraphrase_canonicals(questions, manifest)
    need_paraphrases = bool(missing_para)
    need_reuse = not _reuse_fixtures_ready(corpus)
    need_aetherspace = not _aetherspace_snapshots_ready()
    need_migration = not _migration_fixtures_ready(corpus)
    if not slots_to_repair and not need_paraphrases and not need_reuse and not need_aetherspace and not need_migration:
        print("All recording slots committed in manifest.", flush=True)
        return _recording_pipeline_ready(questions=questions, manifest=manifest, corpus=corpus)[0]
    env = prepare_recording_environment()
    pool = WarmRecordingPool(STAGING)
    session = RecordingSession(
        corpus=corpus,
        pool=pool,
        questions=questions,
        env=env,
    )
    with _staging_sandbox_bundle():
        try:
            if slots_to_repair:
                corpus_message(f"Repairing {len(slots_to_repair)} uncommitted slot(s)...")
                session.record_all(slots=slots_to_repair)
            else:
                print("All recording slots committed in manifest.", flush=True)
            manifest = _load_recording_manifest()
            missing_para = _missing_paraphrase_canonicals(questions, manifest)
            need_paraphrases = bool(missing_para)
            paraphrase_seeds: list[tuple[RecordingSlot, object]] | None = None
            if need_paraphrases:
                paraphrase_seeds, seed_detail = _paraphrase_seeds_for_missing(
                    session,
                    missing_para,
                    session.paraphrase_seeds_collected,
                )
                if seed_detail:
                    corpus_message(f"[build] paraphrase capture failed: {seed_detail}")
                    need_paraphrases = False
            finalize_recording_tail(
                session,
                pool,
                corpus,
                paraphrase_seeds=paraphrase_seeds,
                capture_paraphrases=need_paraphrases,
                require_paraphrase_seeds=need_paraphrases,
                capture_reuse=need_reuse,
                capture_aetherspace=need_aetherspace,
                capture_migration=need_migration,
            )
        finally:
            session.close()
            pool.close()
            teardown_recording_environment(env)
    manifest = _load_recording_manifest()
    return _recording_pipeline_ready(questions=questions, manifest=manifest, corpus=corpus)[0]


def should_skip_zip_member(path: Path) -> bool:
    name = path.name
    if name in SKIP_ZIP_NAMES or name in SANDBOX_ZIP_BUILD_TIME_ONLY:
        return True
    return name.endswith(".__write.lock") or name.endswith(".lock")


def write_zip_from_staging(*, dest: Path | None = None) -> Path:
    if not STAGING.is_dir():
        raise SystemExit(f"Missing staging directory: {STAGING}")
    target = dest or STAGING_ZIP
    tmp = target.with_suffix(".zip.tmp") if target.suffix else target.with_name(target.name + ".tmp")
    if tmp.is_file():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file() and not should_skip_zip_member(path):
                zf.write(path, path.relative_to(STAGING).as_posix())
    os.replace(tmp, target)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Wrote {target} ({size_mb:.2f} MB)")
    return target


def promote_staging_zip(staging_zip: Path) -> None:
    tmp = OUT_ZIP.with_suffix(".zip.tmp")
    if tmp.is_file():
        tmp.unlink()
    shutil.copy2(staging_zip, tmp)
    os.replace(tmp, OUT_ZIP)
    print(f"Promoted {staging_zip} -> {OUT_ZIP}")


@contextmanager
def _staging_sandbox_bundle():
    """Point sandbox bundle reads at the in-progress staging directory."""
    prev = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP")
    os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = str(STAGING)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev


def validate_staging_dir(*, staging_dir: Path | None = None, smoke: bool = False) -> list[dict[str, str]]:
    from aetherdialect.aetherdialect import AetherEngine

    target = staging_dir or STAGING
    if not target.is_dir():
        raise SystemExit(f"Missing staging directory: {target}")
    prev = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP")
    os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = str(target)
    try:
        return validate_sandbox_corpus(AetherEngine, smoke=smoke)
    finally:
        if prev is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev


def validate_staging_zip(*, data_zip: Path | None = None) -> list[dict[str, str]]:
    from aetherdialect.aetherdialect import AetherEngine

    target = data_zip or STAGING_ZIP
    if target.is_dir():
        return validate_staging_dir(staging_dir=target)
    prev = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP")
    os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = str(target)
    try:
        return validate_sandbox_corpus(AetherEngine)
    finally:
        if prev is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev


def _uncommitted_slot_labels() -> list[str]:
    """List still-uncommitted recording slots (LLM variance / determinism divergence)."""
    manifest = _load_recording_manifest()
    committed = committed_slot_ids(manifest)
    out: list[str] = []
    for slot in iter_recording_slots(load_staging_questions()):
        if slot_id_for(slot) not in committed:
            out.append(f"[{slot.tier}] {slot.label}")
    return out


def finalize_and_pack(*, smoke: bool = False) -> None:
    """Finalize when all slots are committed: validate once, then pack without uncommitting."""
    prefix = "[smoke] " if smoke else "[build] "
    ready, reasons = _recording_pipeline_ready()
    if not ready:
        for reason in reasons:
            corpus_message(f"{prefix}recording incomplete: {reason}")
        uncommitted = _uncommitted_slot_labels()
        if uncommitted:
            corpus_message(
                f"{prefix}{len(uncommitted)} recording slot(s) still uncommitted (LLM variance). "
                "Run 'python scripts/sandbox_corpus.py --repair' to retry only these:",
            )
            for label in uncommitted:
                corpus_message(f"    - {label}")
        raise SystemExit(1)
    failures = validate_staging_dir(smoke=smoke)
    if failures:
        report_lines = [
            f"FAIL [{row['kind']}] {row.get('tier', '')} {row['name'][:70]}: {row['detail']}"
            for row in failures
        ]
        VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
        VALIDATE_OUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        for line in report_lines:
            print(line)
        raise SystemExit(
            f"DETERMINISM REGRESSION: {len(failures)} committed slot(s) failed offline validate "
            "even though they were recorded green against this corpus. This is a pipeline "
            "nondeterminism bug and must be fixed, not retried. "
            f"See {VALIDATE_TRACE_OUT} and {SANDBOX_REPLAY_TRACE_PATH}.",
        )
    pack_and_promote(smoke=smoke, revalidate=False)


def pack_and_promote(*, smoke: bool = False, revalidate: bool = True) -> None:
    from aetherdialect.aetherdialect import AetherEngine

    ensure_schema_literals()
    if revalidate:
        failures = validate_staging_dir(smoke=smoke)
        if failures:
            report_lines = [
                f"FAIL [{row['kind']}] {row.get('tier', '')} {row['name'][:70]}: {row['detail']}" for row in failures
            ]
            VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
            VALIDATE_OUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            for line in report_lines:
                print(line)
            raise SystemExit(f"{len(failures)} sandbox validation failures")
    write_zip_from_staging(dest=OUT_ZIP if not smoke else STAGING_ZIP)
    if smoke:
        issues: list[str] = []
    else:
        issues = _sandbox_doctor_verbose()
    if issues:
        raise SystemExit(f"sandbox doctor failed after promote: {issues}")
    if smoke:
        msg = f"Sandbox smoke OK ({len(iter_recording_slots(load_staging_questions()))} recording slots)"
        print(msg)
        VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
        VALIDATE_OUT.write_text(msg + "\n", encoding="utf-8")
        if STAGING.is_dir():
            _remove_tree(STAGING)
        verbose_message(f"Wrote {STAGING_ZIP} (data.zip not updated); removed ephemeral staging workspace")
        return
    prev_bundle = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP")
    os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = str(OUT_ZIP)
    try:
        assert_sandbox_complete(AetherEngine)
    finally:
        if prev_bundle is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev_bundle
    msg = "Sandbox corpus OK (all questions, recipes, feedback flows)"
    print(msg)
    VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATE_OUT.write_text(msg + "\n", encoding="utf-8")
    if STAGING.is_dir():
        _remove_tree(STAGING)
    if STAGING_ZIP.is_file():
        STAGING_ZIP.unlink()
    verbose_message("Removed ephemeral staging workspace after successful pack.")

def _run_full_build(record_reuse_pairs: bool = False, *, smoke: bool = False) -> None:
    global _SMOKE_BUILD
    _SMOKE_BUILD = smoke
    prefix = "[smoke] " if smoke else "[build] "
    corpus_message(f"{prefix}assembling staging (sqlite seed + subset SQL)...")
    assemble_staging(reset_fixtures=True, smoke=smoke)
    corpus_message(f"{prefix}writing overrides demo...")
    write_overrides_demo()
    corpus_message(f"{prefix}building artifacts baseline...")
    build_artifacts_baseline()
    corpus_message(f"{prefix}writing migration demo...")
    write_migration_demo()
    write_build_fingerprint()
    corpus_message(f"{prefix}recording fixtures...")
    recording_ok = record_corpus(record_reuse_pairs=record_reuse_pairs)
    if not recording_ok:
        corpus_message(
            f"{prefix}recording incomplete after fresh build (LLM variance on some slots). "
            "Re-run 'python scripts/sandbox_corpus.py --repair' to retry only the uncommitted slots.",
        )
    corpus_message(f"{prefix}validating and packing sandbox zip...")
    finalize_and_pack(smoke=smoke)
    corpus_message(f"{prefix}done.")


def _ensure_script_import_paths() -> None:
    import sys

    for path in (REPO, SCRIPTS):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def main() -> None:
    import argparse

    _ensure_script_import_paths()

    parser = argparse.ArgumentParser(
        description="Build the sandbox corpus from scratch, or repair failing slots and repack.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-record failing slots under scripts/sandbox_staging, then validate and pack",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the full build pipeline with two practice questions (including the flaky "
            "games-language slot) plus all validation, feedback, and space slots; runs "
            "paraphrase catalog, reuse PE, aetherspace, and migration tail capture; writes "
            "scripts/sandbox_staging.zip only (not src/aetherdialect/sandbox/data.zip). "
            "Combine with --repair to repack sandbox_staging.zip without touching data.zip."
        ),
    )
    parser.add_argument(
        "--record-reuse-pairs",
        action="store_true",
        help="Record paraphrase pairs to test intent reuse within the same session.",
    )
    args = parser.parse_args()
    if args.repair:
        prefix = "[smoke] " if args.smoke else "[repair] "
        corpus_message(f"{prefix}re-recording only uncommitted (LLM-variance) slots...")
        repair_failing_slots()
        corpus_message(f"{prefix}validating and packing sandbox zip...")
        finalize_and_pack(smoke=args.smoke)
        corpus_message(f"{prefix}done.")
        return
    _run_full_build(record_reuse_pairs=args.record_reuse_pairs, smoke=args.smoke)



if __name__ == "__main__":
    main()
