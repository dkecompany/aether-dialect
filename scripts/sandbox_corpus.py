"""Shared helpers for sandbox corpus assembly, recording, validation, and packing."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aetherdialect._live_testing
import aetherdialect._llm_provider
import aetherdialect._main_execution
import aetherdialect._sandbox
import aetherdialect._schema_finalize
from aetherdialect._config import (
    ConfigError,
    DuckDBRuntimeConfig,
    EngineConfig,
    PolicyConfig,
)
from aetherdialect._constants import FEDERATION_DECLARATION_FILENAME
from aetherdialect._constants_runtime import (
    INTENT_COMPOSE_SYSTEM,
    INTENT_GROUND_SYSTEM,
    INTENT_INTERPRET_SYSTEM,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_MEMBER_SPACE_NOTES_FILES,
    SANDBOX_MEMBER_SPACE_QUESTIONS,
    SANDBOX_MEMBER_SPACE_TABLES,
    SANDBOX_QUESTION_TIERS,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
)
from aetherdialect._contracts_base import EngineContext, SchemaRole
from aetherdialect._contracts_core import TemplateMatch
from aetherdialect._llm_provider import MockProvider
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._sandbox import Sandbox
from aetherdialect._utils import (
    StepResult,
    build_session_step_trace,
    llm_usage_question_scope,
    llm_usage_session_scope,
    pipeline_capture,
    stable_json,
)
from aetherdialect._utils_artifacts import append_failure_trace
from aetherdialect._utils_intent import generate_paraphrases_of_seed_question, normalize_question
from aetherdialect.aetherdialect import AetherEngine

REPO = Path(__file__).resolve().parents[1]

_BUILD_VERBOSE = True
_SMOKE_BUILD = False
SMOKE_TOUR_QUESTION = "How many rentals happened in 2025?"
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
RECORDING_MAX_VALIDATE_PASSES = 25
RECORDING_RESULTS_PATH = SCRIPTS / "logs" / "sandbox_results.txt"
RECORDING_INVOICE_PATH = SCRIPTS / "logs" / "sandbox_invoice.txt"
RECORDING_MANIFEST_PATH = STAGING / "recording_manifest.json"
EXPECTATIONS_SOURCE = DATA / "sandbox_expectations.json"
SANDBOX_ARTIFACTS_ROOT = SCRIPTS / "logs" / "_sandbox_artifacts"
PASS_BUT_WRONG_QUESTION = "How many rentals were made in total?"
SCENARIOS_SOURCE = DATA / "sandbox_scenarios.json"
HANDCRAFTED_FIXTURES_SOURCE = DATA / "sandbox_handcrafted_fixtures.json"
VALIDATE_OUT = SCRIPTS / "logs" / "validate_out.txt"
VALIDATE_TRACE_OUT = SCRIPTS / "logs" / "validate_trace.txt"
SKIP_ZIP_NAMES = frozenset(
    {
        "rental_shop_post_migration.sql",
        "rental_shop_mock.pre_repair.json",
        "sandbox_build_fingerprint.json",
    },
)

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_source_rental_shop = importlib.import_module("source_rental_shop")
PAYMENT_UNION_SPLIT_STORE_THRESHOLD = _source_rental_shop.PAYMENT_UNION_SPLIT_STORE_THRESHOLD
payment_store_id_by_payment_id = _source_rental_shop.payment_store_id_by_payment_id
payment_store_id_by_rental_id = _source_rental_shop.payment_store_id_by_rental_id

BASELINE_OWNER_SUBDIR = "owner"
BASELINE_CONSUMER_SUBDIR = "consumer"
BASELINE_OWNER_VIEWS_SUBDIR = "owner_views"
BASELINE_CONSUMER_VIEWS_SUBDIR = "consumer_views"
BASELINE_FEDERATION_SUBDIR = "federation"
FEDERATION_DECLARATION_PATH = DATA / FEDERATION_DECLARATION_FILENAME
FEDERATION_STOREFRONT_PG_SCHEMA = "rental_shop_fed_storefront"
FEDERATION_CATALOG_MYSQL_DATABASE = "rental_shop_fed_catalog"
FEDERATION_LOGISTICS_PG_SCHEMA = "rental_shop_fed_logistics"
FEDERATION_CRM_MARIADB_DATABASE = "rental_shop_fed_crm"
_CROSS_PARTITION_FK_RE = re.compile(
    r"\bREFERENCES\s+(\w+)\s*\(",
    re.IGNORECASE,
)


def load_federation_declaration_data() -> dict[str, Any]:
    """Return the parsed federation declaration from ``scripts/data``."""
    if not FEDERATION_DECLARATION_PATH.is_file():
        raise FileNotFoundError(f"Missing federation declaration: {FEDERATION_DECLARATION_PATH}")
    payload = json.loads(FEDERATION_DECLARATION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("federation declaration must be a JSON object")
    return payload


def _stage_demo_schema_structure(handle: Any) -> None:
    """Load bundled structure demo JSON and call production ``apply_structure(document)``."""
    from aetherdialect._sandbox import Sandbox

    extract = Sandbox._sandbox_extract_path_for_engine(handle.engine)
    if extract is None and STAGING.is_dir():
        extract = STAGING
    if extract is None:
        raise FileNotFoundError("Missing sandbox extract path for structure staging")
    source = extract / "schema_structure_demo.json"
    if not source.is_file():
        source = STAGING / "schema_structure_demo.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing bundled schema structure demo: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"schema structure demo must be a JSON object: {source}")
    handle.engine.apply_structure(document)

FEDERATION_PARTITION_TABLES: dict[str, frozenset[str]] = {
    "storefront": frozenset(
        {
            "country",
            "city",
            "address",
            "store",
            "staff",
            "customer",
            "rental",
            "reservation",
            "payment",
        },
    ),
    "catalog": frozenset(
        {
            "language",
            "item",
            "film",
            "book",
            "game",
            "actor",
            "film_actor",
            "category",
            "item_category",
            "item_feature",
            "game_supported_language",
            "author",
            "publisher",
            "inventory",
            "payment",
            "country",
            "city",
        },
    ),
    "logistics": frozenset(
        {
            "warehouse",
            "stock_transfer",
            "supplier",
            "purchase_order",
            "purchase_line",
            "courier",
            "delivery",
            "damage_report",
            "inventory_status_history",
            "receipts",
        },
    ),
    "crm": frozenset({"promotion", "promotion_redemption", "customer", "staff"}),
}


def load_federation_partition_map() -> dict[str, frozenset[str]]:
    """Return the authoritative federation partition map from ``scripts/data`` or the in-code constant."""
    path = Path(__file__).resolve().parent / "data" / "federation_partition.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("federation_partition.json must be a JSON object")
        return {
            str(source_id): frozenset(str(table) for table in tables if str(table).strip())
            for source_id, tables in payload.items()
            if isinstance(tables, list)
        }
    return dict(FEDERATION_PARTITION_TABLES)


def federation_partition_tables(source_id: str) -> frozenset[str]:
    """Return physical table names that belong to *source_id* (union members included)."""
    from aetherdialect._federation_manifest import parse_federation_declaration

    partition_map = load_federation_partition_map()
    tables: set[str] = set(partition_map.get(source_id, frozenset()))
    _, mappings = parse_federation_declaration(load_federation_declaration_data())
    for entry in mappings.logical_tables:
        for member in entry.members:
            if str(member.source).strip() == source_id:
                tables.add(str(member.table).strip())
    return frozenset(name for name in tables if name)


def federation_member_column_projections(source_id: str) -> dict[str, frozenset[str]]:
    """Return per-table column projections declared for *source_id*."""
    from aetherdialect._federation_manifest import parse_federation_declaration

    _, mappings = parse_federation_declaration(load_federation_declaration_data())
    out: dict[str, frozenset[str]] = {}
    for entry in mappings.logical_tables:
        for member in entry.members:
            if str(member.source).strip() != source_id:
                continue
            if member.columns:
                out[str(member.table).strip()] = frozenset(str(key).strip() for key in member.columns)
    return out


def _strip_disallowed_create_table_foreign_keys(
    create_sql: str,
    table: str,
    partition_tables: frozenset[str],
) -> str:
    pattern = re.compile(r"\s+REFERENCES\s+\w+\s*\([^)]*\)", re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        clause = match.group(0).strip()
        if federation_foreign_key_allowed(table, clause, partition_tables):
            return match.group(0)
        return ""

    return pattern.sub(_replace, create_sql)


def _create_table_sql_for_projection(
    conn: sqlite3.Connection,
    table: str,
    columns: frozenset[str],
) -> str:
    info = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    col_defs: list[str] = []
    for _cid, name, ctype, _notnull, _dflt, pk in info:
        if str(name) not in columns:
            continue
        part = f"{name} {ctype or 'TEXT'}"
        if pk:
            part += " PRIMARY KEY"
        col_defs.append(part)
    if not col_defs:
        raise ValueError(f"no projected columns for table {table!r}")
    return f"CREATE TABLE {table} ({', '.join(col_defs)});"


def federation_foreign_key_allowed(table: str, fk_clause: str, partition_tables: frozenset[str]) -> bool:
    """Return True when both ends of *fk_clause* stay inside *partition_tables*."""
    if table not in partition_tables:
        return False
    match = _CROSS_PARTITION_FK_RE.search(fk_clause)
    if match is None:
        return True
    return match.group(1) in partition_tables


def corpus_message(message: str) -> None:
    print(message, flush=True)


def verbose_message(message: str) -> None:
    if _BUILD_VERBOSE:
        corpus_message(message)


LLM_PATCH_MODULES = (
    "aetherdialect._intent_loop",
    "aetherdialect._pipeline_generate",
    "aetherdialect._pipeline_execute",
    "aetherdialect._schema_profile",
    "aetherdialect._templates",
    "aetherdialect._utils_intent",
    "aetherdialect._schema_finalize",
    "aetherdialect._live_testing",
    "aetherdialect._main_execution",
)


def slot_id_for(slot: RecordingSlot) -> str:
    if slot.kind == "feedback":
        return f"feedback:{slot.label}"
    role = "consumer" if slot.preset.startswith("consumer") else "owner"
    mode = slot.mode or ("reader" if role == "consumer" else "writer")
    return f"{role}:{mode}:{slot.label}"


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
    if Sandbox.question_ok(
        step,
        slot.label,
        slot_id=sid,
        profile=check_profile,
        tier=check_tier,
    ):
        return True, ""
    detail = Sandbox.check_sandbox_faithfulness(
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


def _intent_fixture_question(row: dict[str, str]) -> str | None:
    user = str(row.get("user", "")).strip()
    if not user.startswith("{"):
        return None
    try:
        body = json.loads(user)
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    question = body.get("question")
    if isinstance(question, str) and question.strip():
        return normalize_question(question)
    return None


def _mock_verify_targets_for_slot(
    slot: RecordingSlot,
) -> list[tuple[SlotConstruction, bool, str, str, str]]:
    """Return mock-verify runs that must pass before a slot commits (mirrors pack validate).

    Each target carries the expectation ``slot_id`` it must be judged under, so replay
    resolves the same expectation row the live capture check used.

    Owner writer question slots also replay under a member-scoped consumer (pack validate
    does the same). Federation slots use federation construction as their primary target.
    """
    construction = _construction_for_slot(slot)
    targets: list[tuple[SlotConstruction, bool, str, str, str]] = [
        (construction, construction.apply_structure, slot.preset, slot_id_for(slot), slot.tier),
    ]
    if (
        slot.kind == "question"
        and slot.tier == "questions"
        and slot.preset == "owner_writer"
        and slot.mode in (None, "writer")
        and construction.surface != "federation"
    ):
        member = _member_for_question(slot.label)
        consumer_construction = SlotConstruction(
            surface="single",
            role="consumer",
            member=member,
            mode="reader",
        )
        consumer_slot = RecordingSlot(
            tier="consumer_reader",
            label=slot.label,
            preset="consumer_reader",
            mode="reader",
        )
        targets.append(
            (
                consumer_construction,
                False,
                "consumer_reader",
                slot_id_for(consumer_slot),
                "consumer_reader",
            ),
        )
    return targets


def baseline_dir_for_preset(
    bundle_dir: Path,
    preset: str,
    *,
    include: Literal["tables", "views"] = "tables",
) -> Path:
    """Return the preset-scoped baseline directory under *bundle_dir*."""
    from aetherdialect._constants import FEDERATION_COMPOSITE_SCHEMA_FILENAME

    root = bundle_dir / "artifacts_baseline"
    if include == "views":
        if preset == "consumer_reader":
            views = root / BASELINE_CONSUMER_VIEWS_SUBDIR
        else:
            views = root / BASELINE_OWNER_VIEWS_SUBDIR
        if views.is_dir() and (views / "schema_graph.json.gz").is_file():
            return views
        return views if views.is_dir() else root / BASELINE_OWNER_VIEWS_SUBDIR
    if preset == "federation":
        fed = root / BASELINE_FEDERATION_SUBDIR
        if fed.is_dir() and (fed / FEDERATION_COMPOSITE_SCHEMA_FILENAME).is_file():
            return fed
        if fed.is_dir() and (fed / "schema_graph.json.gz").is_file():
            return fed
    if preset == "consumer_reader":
        consumer = root / BASELINE_CONSUMER_SUBDIR
        if consumer.is_dir() and (consumer / "schema_graph.json.gz").is_file():
            return consumer
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
    from aetherdialect._constants import FEDERATION_COMPOSITE_SCHEMA_FILENAME

    dest.mkdir(parents=True, exist_ok=True)
    for name in _BASELINE_CACHE_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    composite = source / FEDERATION_COMPOSITE_SCHEMA_FILENAME
    schema_dest = dest / "schema_graph.json.gz"
    if composite.is_file() and not schema_dest.is_file():
        shutil.copy2(composite, schema_dest)
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
        return Path(MainExecutionOps.compute_engine_storage_dir(artifacts_dir, "duckdb"))
    finally:
        DuckDBRuntimeConfig.DATABASE_PATH = saved_path
        DuckDBRuntimeConfig.SCHEMA = saved_schema


def _seed_engine_baseline(
    *,
    artifacts_dir: str,
    bundle_dir: Path,
    preset: str,
    include: Literal["tables", "views"] = "tables",
) -> None:
    from aetherdialect._constants import FEDERATION_COMPOSITE_SCHEMA_FILENAME

    if preset == "federation":
        verbose_message("[build] seed baseline skip: federation uses member/composite seeders")
        return
    baseline = baseline_dir_for_preset(bundle_dir, preset, include=include)
    graph_path = baseline / "schema_graph.json.gz"
    composite_path = baseline / FEDERATION_COMPOSITE_SCHEMA_FILENAME
    if not baseline.is_dir():
        verbose_message(f"[build] seed baseline skip: missing baseline dir {baseline}")
        return
    if not graph_path.is_file() and not composite_path.is_file():
        verbose_message(f"[build] seed baseline skip: no schema graph under {baseline}")
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


def _begin_eval_results(path: Path, *, invoice_path: Path | None = None) -> Path:
    from sandbox_recording import begin_eval_results

    return begin_eval_results(path, invoice_path=invoice_path)


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
    skip_template_learning: Callable[[object | None], bool]
    llm_patch_modules: tuple[str, ...]


def prepare_recording_environment() -> RecordingEnvironment:
    """Load credentials, sync ``EngineConfig``, and apply recording patches."""
    from load_rental_shop_engines import load_env_file
    from sandbox_recording import write_sandbox_recording_toml

    env_path = load_env_file(ENV_FILE, override=True)
    merged_env = dict(os.environ)
    merged_env["AETHERDIALECT_LLM_PROVIDER"] = "openai"
    try:
        MainExecutionOps._configure_llm_from_environment(merged_env)
    except ConfigError as exc:
        raise RuntimeError(
            f"LLM credentials are not configured (check {env_path}). {exc}",
        ) from exc
    if not EngineConfig.llm_credentials_configured():
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

    orig_refine = aetherdialect._schema_finalize._refine_descriptions_via_llm
    orig_write_toml = aetherdialect._sandbox.Sandbox._write_sandbox_toml
    orig_template_match = aetherdialect._pipeline_generate.match_question_level_template_reuse
    aetherdialect._sandbox.Sandbox._write_sandbox_toml = openai_toml
    prev_provider = EngineConfig.LLM_PROVIDER
    MockProvider.reset_mock_provider()
    EngineConfig.LLM_PROVIDER = "openai"
    aetherdialect._pipeline_generate.match_question_level_template_reuse = skip_template_reuse
    aetherdialect._main_execution.match_question_level_template_reuse = skip_template_reuse
    aetherdialect._live_testing.match_question_level_template_reuse = skip_template_reuse
    orig_persist = aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session

    def skip_template_learning(_port: object | None) -> bool:
        del _port
        return False

    aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session = (
        skip_template_learning
    )

    return RecordingEnvironment(
        orig_write_toml=orig_write_toml,
        orig_refine=orig_refine,
        orig_template_match=orig_template_match,
        orig_persist_template_learning=orig_persist,
        prev_provider=prev_provider,
        openai_toml=openai_toml,
        skip_template_reuse=skip_template_reuse,
        skip_template_learning=skip_template_learning,
        llm_patch_modules=LLM_PATCH_MODULES,
    )


def teardown_recording_environment(env: RecordingEnvironment) -> None:
    """Restore module patches applied by :func:`prepare_recording_environment`."""
    aetherdialect._schema_finalize._refine_descriptions_via_llm = env.orig_refine
    aetherdialect._sandbox.Sandbox._write_sandbox_toml = env.orig_write_toml
    aetherdialect._pipeline_generate.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._main_execution.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._live_testing.match_question_level_template_reuse = env.orig_template_match
    aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session = (
        env.orig_persist_template_learning
    )
    EngineConfig.LLM_PROVIDER = env.prev_provider


@dataclass(frozen=True)
class RecordingSlot:
    tier: str
    label: str
    preset: str = "owner_writer"
    mode: str | None = None
    kind: str = "question"


@dataclass(frozen=True)
class SlotConstruction:
    """Production-shaped sandbox construction axes for corpus record/replay."""

    surface: Literal["single", "federation", "views"] = "single"
    role: Literal["owner", "consumer"] = "owner"
    member: str | None = None
    mode: str | None = None
    apply_structure: bool = False
    include: Literal["tables", "views"] = "tables"


_SCHEMA_BODY_KEYS = ("schema_literal_json", "schema_summary", "schema_info")
_GATEKEEPER_MARKER = "valid_database_question"
_SCOPE_REFUSAL_OUTPUT = json.dumps(
    {
        "valid_database_question": "no",
        "query_type": "unspecified",
        "corrected": "",
        "reason": "out of scope for this member space",
    },
    ensure_ascii=False,
)


@dataclass(frozen=True)
class FixtureSurface:
    """One replay surface that may need derived fixtures for a recorded question."""

    construction: SlotConstruction
    profile: str
    tier: str
    schema_literal_slot: Literal["owner", "consumer"]


def question_home_member(question: str) -> str | None:
    """Return the sole member space that owns *question*, or ``None`` when full-only."""
    norm = normalize_question(question)
    matches = [
        member
        for member, questions in SANDBOX_MEMBER_SPACE_QUESTIONS.items()
        if any(normalize_question(candidate) == norm for candidate in questions)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def is_full_only_question(question: str) -> bool:
    """True when *question* is not owned by exactly one member space."""
    return question_home_member(question) is None


def fixture_question_for_row(row: dict[str, str]) -> str | None:
    """Best-effort question text associated with a fixture row."""
    user = str(row.get("user", "")).strip()
    if user.startswith("{"):
        try:
            body = json.loads(user)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            question = body.get("question")
            if isinstance(question, str) and question.strip():
                return normalize_question(question)
            matched = body.get("matched_question")
            if isinstance(matched, str) and matched.strip():
                return normalize_question(matched)
    if _GATEKEEPER_MARKER in str(row.get("system", "")) and user and not user.startswith("{"):
        return normalize_question(user)
    return None


def fixture_rows_for_question(fixtures: list[dict[str, str]], question: str) -> list[dict[str, str]]:
    """Return fixture rows whose payload references *question*."""
    target = normalize_question(question)
    return [row for row in fixtures if fixture_question_for_row(row) == target]


def _pin_literals(literals: dict[str, str] | None) -> dict[str, str]:
    if literals is not None:
        MockProvider._pin_mock_schema_literals(literals)
        return literals
    return MockProvider.load_canonical_schema_literals()


def adapt_fixture_user_for_surface(
    user: str,
    *,
    source_slot: Literal["owner", "consumer"],
    target_slot: Literal["owner", "consumer"],
    literals: dict[str, str] | None = None,
) -> str:
    """Re-key a stored fixture user payload for *target_slot* schema stubs."""
    pinned = _pin_literals(literals)
    if source_slot == target_slot:
        return MockProvider.mock_fixture_user_key(user, literals=pinned)
    stripped = user.strip()
    if not stripped.startswith("{"):
        return MockProvider.mock_fixture_user_key(user, literals=pinned)
    try:
        body = json.loads(stripped)
    except json.JSONDecodeError:
        return MockProvider.mock_fixture_user_key(user, literals=pinned)
    if not isinstance(body, dict):
        return MockProvider.mock_fixture_user_key(user, literals=pinned)
    canonical_literal = pinned[target_slot]
    for key in _SCHEMA_BODY_KEYS:
        if key not in body:
            continue
        try:
            body[key] = json.loads(canonical_literal)
        except json.JSONDecodeError:
            body[key] = canonical_literal
    return MockProvider.mock_fixture_user_key(stable_json(body), literals=pinned)


def adapt_fixture_output_for_surface(
    output_text: str,
    *,
    source: FixtureSurface,
    target: FixtureSurface,
) -> str:
    """Apply declarative output rewrites when the target surface needs different naming."""
    if source.construction.surface == target.construction.surface:
        return output_text
    if target.construction.surface != "federation":
        return output_text
    return output_text


def scope_refusal_question_label(question: str, member: str) -> str:
    """Stable gatekeeper label for an out-of-scope question inside *member*."""
    return f"{question} @member:{member}"


def scope_refusal_rows_for_question(
    question: str,
    *,
    member: str,
    literals: dict[str, str] | None = None,
    gatekeeper_system: str = _GATEKEEPER_MARKER,
    interpret_system: str = "intent-interpret",
) -> list[dict[str, str]]:
    """Deterministic refusal fixtures for *question* inside member *member* when out-of-scope."""
    pinned = _pin_literals(literals)
    gatekeeper_user = scope_refusal_question_label(question, member)
    user_key = MockProvider.mock_fixture_user_key(gatekeeper_user, literals=pinned)
    rows: list[dict[str, str]] = [
        {
            "task": "default",
            "system": gatekeeper_system,
            "user": user_key,
            "output_text": _SCOPE_REFUSAL_OUTPUT,
        },
    ]
    refusal_intent = {
        "interpret_plan_schema": {},
        "question": question,
        "member_space": member,
        "schema_literal_json": json.loads(pinned["consumer"]),
    }
    rows.append(
        {
            "task": "intent",
            "system": interpret_system,
            "user": MockProvider.mock_fixture_user_key(stable_json(refusal_intent), literals=pinned),
            "output_text": json.dumps(
                {
                    "interpret_plan": {
                        "approach": "Refuse out-of-scope question for member space.",
                        "tables": [],
                        "grounding": [],
                    }
                },
                ensure_ascii=False,
            ),
        },
    )
    return rows


def fan_out_surfaces_for_slot(
    slot: RecordingSlot,
    *,
    construction_for_slot: Any,
    recipe_for_slot: Any,
    federation_ineligible: frozenset[str],
) -> list[FixtureSurface]:
    """Return wider/sibling surfaces that should share fixtures for *slot*."""
    if slot.kind != "question":
        return []
    recipe = recipe_for_slot(slot)
    member = question_home_member(slot.label)
    surfaces: list[FixtureSurface] = []

    if (
        slot.tier == "questions"
        and slot.preset == "owner_writer"
        and slot.mode in (None, "writer")
        and recipe != "federation"
        and member is not None
    ):
        surfaces.append(
            FixtureSurface(
                construction=SlotConstruction(
                    surface="single",
                    role="consumer",
                    member=member,
                    mode="reader",
                ),
                profile="consumer_reader",
                tier="consumer_reader",
                schema_literal_slot="consumer",
            ),
        )

    if (
        recipe != "federation"
        and member is not None
        and slot.label not in federation_ineligible
        and slot.tier == "questions"
    ):
        surfaces.append(
            FixtureSurface(
                construction=SlotConstruction(surface="federation", role="owner"),
                profile="owner_writer",
                tier="federation",
                schema_literal_slot="owner",
            ),
        )

    if recipe == "federation":
        surfaces.append(
            FixtureSurface(
                construction=SlotConstruction(surface="single", role="owner", mode="writer"),
                profile="owner_writer",
                tier="questions",
                schema_literal_slot="owner",
            ),
        )

    canonical = construction_for_slot(slot)

    def _construction_identity(construction: SlotConstruction) -> tuple[object, ...]:
        return (
            construction.surface,
            construction.role,
            construction.member,
            construction.mode,
            construction.include,
            construction.apply_structure,
        )

    canonical_id = _construction_identity(canonical)
    deduped: list[FixtureSurface] = []
    seen: set[tuple[object, ...]] = set()
    for surface in surfaces:
        target_id = _construction_identity(surface.construction)
        if target_id == canonical_id:
            continue
        if target_id in seen:
            continue
        seen.add(target_id)
        deduped.append(surface)
    return deduped


def canonical_surface_for_slot(
    slot: RecordingSlot,
    *,
    construction_for_slot: Any,
) -> FixtureSurface:
    """Surface where *slot* was live-recorded (owner writer by default)."""
    construction = construction_for_slot(slot)
    schema_slot: Literal["owner", "consumer"] = "consumer" if construction.role == "consumer" else "owner"
    profile = slot.preset if slot.kind == "question" else "feedback"
    return FixtureSurface(
        construction=construction,
        profile=profile,
        tier=slot.tier,
        schema_literal_slot=schema_slot,
    )


def derive_fixture_rows_for_surface(
    canonical_rows: list[dict[str, str]],
    *,
    question: str,
    source: FixtureSurface,
    target: FixtureSurface,
    fixture_key: Any,
    literals: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Derive target-surface fixture rows from canonical rows without calling the LLM."""
    del question
    derived: list[dict[str, str]] = []
    for row in canonical_rows:
        adapted_user = adapt_fixture_user_for_surface(
            str(row.get("user", "")),
            source_slot=source.schema_literal_slot,
            target_slot=target.schema_literal_slot,
            literals=literals,
        )
        new_row = {
            "task": str(row.get("task", "")),
            "system": str(row.get("system", "")),
            "user": adapted_user,
            "output_text": adapt_fixture_output_for_surface(
                str(row.get("output_text", "")),
                source=source,
                target=target,
            ),
        }
        if fixture_key(new_row) == fixture_key(row):
            continue
        derived.append(new_row)
    return derived


def plan_fixture_fan_out(
    fixtures: list[dict[str, str]],
    slots: list[RecordingSlot],
    seen: set[tuple[str, str, str]],
    *,
    construction_for_slot: Any,
    recipe_for_slot: Any,
    federation_ineligible: frozenset[str],
    fixture_key: Any,
    literals: dict[str, str] | None = None,
    include_scope_refusals: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    """Return fixture rows to add and human-readable notes (pure function)."""
    to_add: list[dict[str, str]] = []
    notes: list[str] = []
    pending_keys = set(seen)

    for slot in slots:
        if slot.kind != "question":
            continue
        canonical = canonical_surface_for_slot(slot, construction_for_slot=construction_for_slot)
        canonical_rows = fixture_rows_for_question(fixtures, slot.label)
        if not canonical_rows:
            notes.append(f"skip fan-out for {slot.label!r}: no canonical fixtures")
            continue
        for target in fan_out_surfaces_for_slot(
            slot,
            construction_for_slot=construction_for_slot,
            recipe_for_slot=recipe_for_slot,
            federation_ineligible=federation_ineligible,
        ):
            derived = derive_fixture_rows_for_surface(
                canonical_rows,
                question=slot.label,
                source=canonical,
                target=target,
                fixture_key=fixture_key,
                literals=literals,
            )
            added = 0
            for row in derived:
                key = fixture_key(row)
                if key in pending_keys:
                    continue
                pending_keys.add(key)
                to_add.append(row)
                added += 1
            if added:
                notes.append(f"fan-out {slot.label!r} -> {target.tier} ({added} row(s))")

        if include_scope_refusals and is_full_only_question(slot.label):
            for member in SANDBOX_MEMBER_SPACE_QUESTIONS:
                refusal_rows = scope_refusal_rows_for_question(
                    slot.label,
                    member=member,
                    literals=literals,
                )
                for row in refusal_rows:
                    key = fixture_key(row)
                    if key in pending_keys:
                        continue
                    pending_keys.add(key)
                    to_add.append(row)
                if refusal_rows:
                    notes.append(f"scope refusal {slot.label!r} @ {member}")

    return to_add, notes


def apply_fixture_fan_out(
    corpus: FixtureCorpus,
    slots: list[RecordingSlot],
    *,
    construction_for_slot: Any,
    recipe_for_slot: Any,
    federation_ineligible: frozenset[str],
    fixture_key: Any,
    literals: dict[str, str] | None = None,
) -> int:
    """Merge fan-out rows into *corpus* and flush. Returns number of rows added."""
    rows, notes = plan_fixture_fan_out(
        corpus.fixtures,
        slots,
        corpus.seen,
        construction_for_slot=construction_for_slot,
        recipe_for_slot=recipe_for_slot,
        federation_ineligible=federation_ineligible,
        fixture_key=fixture_key,
        literals=literals,
    )
    for note in notes:
        print(f"[fan-out] {note}", flush=True)
    if not rows:
        return 0
    for row in rows:
        key = fixture_key(row)
        corpus.fixtures.append(row)
        corpus.seen.add(key)
    corpus.flush()
    return len(rows)


def _member_for_question(question: str) -> str:
    norm = normalize_question(question)
    for member, questions in SANDBOX_MEMBER_SPACE_QUESTIONS.items():
        for candidate in questions:
            if normalize_question(candidate) == norm:
                return member
    return "catalog"


def _baseline_preset_for_construction(construction: SlotConstruction) -> str:
    if construction.surface == "federation":
        return "federation"
    if construction.role == "consumer":
        return "consumer_reader"
    return "owner_writer"


def _construction_cache_key(construction: SlotConstruction) -> str:
    return (
        f"{construction.surface}:role={construction.role}:member={construction.member}:"
        f"include={construction.include}:structure={construction.apply_structure}"
    )


def _engine_context_for_construction(construction: SlotConstruction) -> EngineContext | None:
    if construction.role != "consumer":
        return None
    member = construction.member or _member_for_question("")
    tables = SANDBOX_MEMBER_SPACE_TABLES.get(member)
    if tables is None:
        raise ValueError(f"unknown sandbox member {member!r}")
    return EngineContext(allow_objects=tables)


def _construction_for_slot(
    slot: RecordingSlot,
    scenario: dict[str, object] | None = None,
) -> SlotConstruction:
    """Return production-shaped construction axes for *slot*."""
    if scenario is None:
        scenario = _load_scenarios_by_question().get(slot.label.lower(), {})
    mechanism = str(scenario.get("mechanism", ""))
    recipe = _recipe_for_slot(slot, scenario)
    include: Literal["tables", "views"] = "views" if slot.tier == "views_questions" or recipe == "views" else "tables"
    if mechanism == "bundled_overrides_hide_staff_ssn":
        return SlotConstruction(
            surface="single",
            role="owner",
            mode=slot.mode,
            apply_structure=True,
            include=include,
        )
    if mechanism == "schema_validation_failure":
        return SlotConstruction(
            surface="single",
            role="consumer",
            member="catalog",
            mode="reader",
            include=include,
        )
    if recipe == "federation":
        return SlotConstruction(surface="federation", role="owner", mode=slot.mode, include=include)
    if slot.preset.startswith("consumer") or slot.tier == "consumer_reader":
        return SlotConstruction(
            surface="single",
            role="consumer",
            member=_member_for_question(slot.label),
            mode=slot.mode or "reader",
            include=include,
        )
    resolved_recipe: Literal["single", "federation", "views"] = "views" if recipe == "views" else "single"
    return SlotConstruction(surface=resolved_recipe, role="owner", mode=slot.mode, include=include)


def _paraphrase_catalog_ready(*, staging_dir: Path = STAGING) -> bool:
    path = staging_dir / "sandbox_catalog.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("paraphrase_pairs")
    if not isinstance(pairs, list) or not pairs:
        return False
    return any(isinstance(row, dict) and str(row.get("canonical", "")).strip() for row in pairs)


def _aetherspace_snapshots_ready(*, staging_dir: Path = STAGING) -> bool:
    from aetherdialect._constants import AETHERSPACES_SEGMENT

    dest = staging_dir / "artifacts_baseline" / AETHERSPACES_SEGMENT
    return dest.is_dir() and any(dest.glob("*.json"))


def _reuse_fixtures_ready(corpus: FixtureCorpus) -> bool:
    paraphrase = "How many rentals happened in 2026?"
    for row in corpus.fixtures:
        user = str(row.get("user", ""))
        output = str(row.get("output_text", ""))
        if paraphrase in user or paraphrase in output:
            return True
        if PARAM_EXTRACTION_SYSTEM_MARKER in str(row.get("system", "")).lower() and "2026" in user:
            return True
    return False


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
) -> list[tuple[RecordingSlot, object]]:
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
    return out


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
        "federation": [],
        "views_questions": [],
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
    tiers["questions"].extend(tiers.pop("federation", []))
    return tiers


FEDERATION_SMOKE_QUESTION = "How many rentals are linked to film titles?"


def smoke_questions(*, source: Path | None = None) -> dict[str, list[str]]:
    """Minimal question set: one practice question, one cross-source question, plus validation sections."""
    full = parse_questions_file(source or QUESTIONS_SOURCE)
    practice: list[str] = []
    if SMOKE_TOUR_QUESTION in full["questions"]:
        practice.append(SMOKE_TOUR_QUESTION)
    elif full["questions"]:
        practice.append(full["questions"][0])
    if FEDERATION_SMOKE_QUESTION in full["questions"] and FEDERATION_SMOKE_QUESTION not in practice:
        practice.append(FEDERATION_SMOKE_QUESTION)
    return {
        "questions": practice,
        "validation_failures": [
            question for question in full["validation_failures"] if question != "How many rentals were made in total?"
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
    for tier in SANDBOX_QUESTION_TIERS + ("feedback_samples",):
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
    """Live LLM fixture slots — owner writer only (consumer is validated, not re-recorded).

    Appends one packing slot that defines/caches all four member-aligned spaces
    (not natural-language questions).
    """
    slots: list[RecordingSlot] = []
    for tier in SANDBOX_QUESTION_TIERS:
        for question in questions.get(tier, []):
            slots.append(RecordingSlot(tier=tier, label=question))
    feedback = questions["feedback_samples"]
    if feedback:
        for sample in feedback:
            slots.append(RecordingSlot(tier="feedback", label=sample, kind="feedback"))
    slots.append(RecordingSlot(tier="spaces", label="member_spaces", kind="space"))
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
FEDERATION_INELIGIBLE_QUESTIONS = frozenset(
    {
        "What is the average payment amount grouped by film category?",
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
    "What is the average rental duration?": {
        "must_tables": ["item"],
        "sql_contains": ["rental_duration"],
    },
    "How many active customers do we have?": {
        "must_tables": ["active_customer_v"],
    },
    "What is the total revenue by store?": {
        "must_tables": ["store_revenue_v"],
    },
    "Which films are in the catalog view?": {
        "must_tables": ["film_catalog_v"],
    },
}
SANDBOX_ZIP_BUILD_TIME_ONLY = frozenset(
    {},
)
PARAPHRASE_SOURCE_TIERS = frozenset({"questions"})
INLINE_PARAPHRASE_COPY_RULES: dict[str, tuple[str, ...]] = {
    "How many rentals happened in 2025?": ("How many rentals happened in 2026?",),
    "Which games support English?": ("Which games have English language support?",),
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
            "must_where": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
        }
    if question in NO_SQL_QUESTIONS or question in VALIDATION_FAILURE_QUESTIONS:
        return {
            "terminal_status": "error",
            "sql_required": False,
            "grain": "none",
            "must_tables": [],
            "must_where": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
            "validation_failure": question in VALIDATION_FAILURE_QUESTIONS,
        }
    if question in FEDERATION_INELIGIBLE_QUESTIONS:
        return {
            "terminal_status": "error",
            "sql_required": False,
            "grain": "none",
            "must_tables": [],
            "must_where": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
        }
    if kind == "feedback":
        return {
            "terminal_status": "ok",
            "sql_required": True,
            "grain": "scalar",
            "must_tables": [],
            "must_where": [],
            "sql_contains": [],
            "forbidden_sql_tokens": [],
        }
    extra = FAITHFULNESS.get(question, {})
    return {
        "terminal_status": "ok",
        "sql_required": True,
        "grain": "scalar",
        "must_tables": list(extra.get("must_tables", [])),
        "must_where": [],
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

    def commit_slot(self, *, prune_question: str = "") -> int:
        buffer_rows = self.collapsed_slot_fixtures()
        buffer_intent_keys = {fixture_key(row) for row in buffer_rows if str(row.get("task", "")) == "intent"}
        for entry in buffer_rows:
            key = fixture_key(entry)
            replaced = False
            for idx, row in enumerate(self.fixtures):
                if fixture_key(row) == key:
                    self.fixtures[idx] = entry
                    replaced = True
                    break
            if not replaced:
                self.fixtures.append(entry)
            self.seen.add(key)
        removed = 0
        if prune_question.strip() and buffer_intent_keys:
            keep_keys = _expand_intent_chain_keep_keys(self.fixtures, buffer_intent_keys)
            self.fixtures, removed = _prune_orphan_intent_rows(
                self.fixtures,
                prune_question,
                keep_keys,
            )
            self.seen = {fixture_key(row) for row in self.fixtures}
        self.discard_slot()
        self.flush()
        return removed


def _corpus_snapshot(corpus: FixtureCorpus) -> tuple[list[dict[str, str]], set[tuple[str, str, str]]]:
    return [dict(row) for row in corpus.fixtures], set(corpus.seen)


def _restore_corpus_snapshot(
    corpus: FixtureCorpus,
    snap: tuple[list[dict[str, str]], set[tuple[str, str, str]]],
) -> None:
    corpus.fixtures, corpus.seen = snap[0], snap[1]
    corpus.flush()


@contextmanager
def _staging_bundle_env():
    """Point sandbox bundle resolution at the staging workspace."""
    prev = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP")
    os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = str(STAGING)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev


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


def assemble_staging(*, reset_fixtures: bool = True, smoke: bool = False) -> None:
    """Populate ``scripts/sandbox_staging`` from canonical pipeline data."""
    from source_rental_shop import (
        export_sandbox_federation_partition_data_dirs,
        export_sandbox_federation_partition_schemas,
        export_sandbox_main_data_dir,
        set_export_log_callback,
    )

    set_export_log_callback(verbose_message)
    if smoke:
        corpus_message(
            "[smoke] subsetting questions from scripts/data/sandbox_questions.txt",
        )
    if STAGING.is_dir():
        _remove_tree(STAGING)
    STAGING.mkdir(parents=True)
    _prepare_sqlite_from_csvs()
    sqlite_path = REPO / "scripts" / "sqlite" / "rental_shop.sqlite"
    if not sqlite_path.is_file():
        raise SystemExit(f"Missing seed source: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    try:
        export_sandbox_main_data_dir(STAGING, conn)
        export_sandbox_federation_partition_schemas(STAGING, conn)
        export_sandbox_federation_partition_data_dirs(STAGING, conn)
    finally:
        conn.close()
    for member in ("storefront", "catalog", "logistics", "crm"):
        notes = DATA / f"federation_{member}_notes.txt"
        if notes.is_file():
            shutil.copy2(notes, STAGING / notes.name)
    for name, src in (
        ("rental_shop.sql", DATA / "rental_shop.sql"),
        ("rental_shop_views.sql", DATA / "rental_shop_views.sql"),
        ("rental_shop_notes.txt", DATA / "rental_shop_notes.txt"),
        ("questions.txt", QUESTIONS_SOURCE),
        ("sandbox_expectations.json", EXPECTATIONS_SOURCE),
        ("sandbox_scenarios.json", SCENARIOS_SOURCE),
        ("sandbox_handcrafted_fixtures.json", HANDCRAFTED_FIXTURES_SOURCE),
        ("federation_declaration.json", DATA / FEDERATION_DECLARATION_FILENAME),
        ("federation_partition.json", DATA / "federation_partition.json"),
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
    (STAGING / "schema_structure_demo.json").write_text(
        OVERRIDES_DEMO_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print("Wrote schema_structure_demo.json")


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
    import aetherdialect._sandbox
    import aetherdialect.aetherdialect
    from aetherdialect.aetherdialect import AetherEngine
    from load_rental_shop_engines import DEFAULT_ENV_FILE, load_env_file
    from sandbox_recording import write_sandbox_recording_toml

    load_env_file(DEFAULT_ENV_FILE, override=True)
    baseline_root = STAGING / "artifacts_baseline"
    owner_baseline = baseline_root / BASELINE_OWNER_SUBDIR
    consumer_baseline = baseline_root / BASELINE_CONSUMER_SUBDIR
    if baseline_root.is_dir():
        _remove_tree(baseline_root)
    orig_write_toml = aetherdialect._sandbox.Sandbox._write_sandbox_toml
    prev_provider = EngineConfig.LLM_PROVIDER
    prev_regen = PolicyConfig.REGENERATE_SCHEMA_GRAPH

    def openai_toml(*, fixtures_file: str) -> str:
        del fixtures_file
        return write_sandbox_recording_toml(str(DEFAULT_ENV_FILE))

    def build_log_sink(line: str) -> None:
        del line

    aetherdialect._sandbox.Sandbox._write_sandbox_toml = openai_toml
    MockProvider.reset_mock_provider()
    merged_env = dict(os.environ)
    merged_env["AETHERDIALECT_LLM_PROVIDER"] = "openai"
    MainExecutionOps._configure_llm_from_environment(merged_env)
    PolicyConfig.REGENERATE_SCHEMA_GRAPH = True
    original_log_sink = aetherdialect.aetherdialect._init_log_sink
    aetherdialect.aetherdialect._init_log_sink = build_log_sink
    owner_tables_graph: Any | None = None
    try:
        corpus_message("[build] artifacts baseline: building schema graph...")
        with AetherEngine.offline_sandbox(
            cleanup_artifacts=True,
            bundle_dir=str(STAGING),
            maintainer_access=True,
        ) as owner_sb:
            owner_literal = owner_sb.engine._schema_graph.schema_literal_json
            schema_graph = owner_sb.engine._schema_graph
            owner_tables_graph = schema_graph
            table_count = len(schema_graph.tables)
            column_count = sum(len(table.columns) for table in schema_graph.tables.values())
            corpus_message(
                f"[build] artifacts baseline: schema graph ready ({table_count} tables, {column_count} columns)",
            )
            owner_engine_dir = Path(str(owner_sb.engine._schema_json_path)).parent
            owner_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(owner_engine_dir, owner_baseline)
            consumer_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(owner_engine_dir, consumer_baseline)
            written = [
                f"{name}={(owner_baseline / name).stat().st_size}B"
                for name in _BASELINE_CACHE_FILES
                if (owner_baseline / name).is_file()
            ]
            verbose_message(f"[build] artifacts baseline files: {', '.join(written) or 'none'}")
        owner_views_baseline = baseline_root / BASELINE_OWNER_VIEWS_SUBDIR
        consumer_views_baseline = baseline_root / BASELINE_CONSUMER_VIEWS_SUBDIR
        corpus_message("[build] artifacts baseline: building views schema graph...")
        with AetherEngine.offline_sandbox(
            cleanup_artifacts=True,
            bundle_dir=str(STAGING),
            maintainer_access=True,
            include="views",
        ) as owner_views_sb:
            from aetherdialect._schema_reflect import project_base_descriptions_onto_views

            views_schema_graph = owner_views_sb.engine._schema_graph
            projected = project_base_descriptions_onto_views(schema_graph, views_schema_graph)
            if projected:
                verbose_message(f"[build] projected {projected} base descriptions onto views graph")
                schema_json = getattr(owner_views_sb.engine, "_schema_json_path", None)
                if schema_json:
                    from aetherdialect._utils_artifacts import write_gzip_json_atomic

                    write_gzip_json_atomic(str(schema_json), views_schema_graph.to_dict(), sort_keys=True)
            views_table_count = len(views_schema_graph.tables)
            views_column_count = sum(len(table.columns) for table in views_schema_graph.tables.values())
            corpus_message(
                f"[build] artifacts baseline: views schema graph ready "
                f"({views_table_count} views, {views_column_count} columns)",
            )
            views_engine_dir = Path(str(owner_views_sb.engine._schema_json_path)).parent
            owner_views_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(views_engine_dir, owner_views_baseline)
            consumer_views_baseline.mkdir(parents=True, exist_ok=True)
            _copy_baseline_cache_files(views_engine_dir, consumer_views_baseline)
            views_written = [
                f"{name}={(owner_views_baseline / name).stat().st_size}B"
                for name in _BASELINE_CACHE_FILES
                if (owner_views_baseline / name).is_file()
            ]
            verbose_message(f"[build] artifacts views baseline files: {', '.join(views_written) or 'none'}")
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
        aetherdialect._sandbox.Sandbox._write_sandbox_toml = orig_write_toml
        MockProvider.reset_mock_provider()
        EngineConfig.LLM_PROVIDER = prev_provider
    aetherdialect._sandbox.Sandbox._pin_bundled_schema_literals(STAGING)
    verbose_message(f"Wrote artifacts baseline under {baseline_root}")
    build_federation_artifacts_baseline(STAGING, owner_tables_graph=owner_tables_graph)


def build_federation_artifacts_baseline(
    staging_dir: Path = STAGING,
    *,
    owner_tables_graph: Any | None = None,
) -> None:
    """Stage federation composite artifacts and per-member trees when partition schemas are present."""
    from aetherdialect._constants import (
        ARTIFACT_DIRECTORY_SEGMENT,
        ARTIFACT_MANIFEST_FILENAME,
        FEDERATION_COMPOSITE_SCHEMA_FILENAME,
        FEDERATION_DECLARATION_FILENAME,
        FEDERATION_MANIFEST_FILENAME,
        FEDERATION_MAPPINGS_FILENAME,
    )
    from aetherdialect._federation_manifest import (
        federation_source_storage_slug,
        parse_federation_declaration,
    )
    from load_rental_shop_engines import DEFAULT_ENV_FILE, load_env_file
    from sandbox_recording import write_sandbox_recording_toml

    storefront_schema = staging_dir / "federation_storefront_schema.sql"
    catalog_schema = staging_dir / "federation_catalog_schema.sql"
    if not (storefront_schema.is_file() and catalog_schema.is_file()):
        corpus_message("[build] federation baseline skipped: partition schemas not staged")
        return
    fed_baseline = staging_dir / "artifacts_baseline" / BASELINE_FEDERATION_SUBDIR
    if fed_baseline.is_dir():
        shutil.rmtree(fed_baseline, ignore_errors=True)
    fed_baseline.mkdir(parents=True, exist_ok=True)
    declaration_src = staging_dir / FEDERATION_DECLARATION_FILENAME
    if not declaration_src.is_file():
        corpus_message("[build] federation baseline skipped: declaration not staged")
        return
    if declaration_src.is_file():
        shutil.copy2(declaration_src, fed_baseline / FEDERATION_DECLARATION_FILENAME)
    manifest_src = staging_dir / FEDERATION_MANIFEST_FILENAME
    mappings_src = staging_dir / FEDERATION_MAPPINGS_FILENAME
    if manifest_src.is_file():
        shutil.copy2(manifest_src, fed_baseline / FEDERATION_MANIFEST_FILENAME)
    if mappings_src.is_file():
        shutil.copy2(mappings_src, fed_baseline / FEDERATION_MAPPINGS_FILENAME)
    load_env_file(DEFAULT_ENV_FILE, override=True)
    orig_write_toml = aetherdialect._sandbox.Sandbox._write_sandbox_toml
    prev_provider = EngineConfig.LLM_PROVIDER

    def openai_toml(*, fixtures_file: str) -> str:
        del fixtures_file
        return write_sandbox_recording_toml(str(DEFAULT_ENV_FILE))

    aetherdialect._sandbox.Sandbox._write_sandbox_toml = openai_toml
    merged_env = dict(os.environ)
    merged_env["AETHERDIALECT_LLM_PROVIDER"] = "openai"
    MainExecutionOps._configure_llm_from_environment(merged_env)
    parsed_manifest, _ = parse_federation_declaration(json.loads(declaration_src.read_text(encoding="utf-8")))
    try:
        with aetherdialect._sandbox.Sandbox._federation_offline_handle_cm(
            AetherEngine,
            cleanup_artifacts=True,
            bundle_dir=str(staging_dir),
            maintainer_access=True,
        ) as fed_sb:
            fed_manifest = getattr(fed_sb.engine, "_federation_manifest", parsed_manifest)
            fed_schema_path = getattr(fed_sb.engine, "_schema_json_path", None)
            fed_engine_dir = (
                Path(str(fed_schema_path)).parent
                if fed_schema_path
                else Path(str(getattr(fed_sb.engine, "_artifacts_dir", fed_sb.artifacts_dir)))
            )
            _copy_baseline_cache_files(fed_engine_dir, fed_baseline)
            fed_storage = getattr(fed_sb.engine, "_federation_storage_dir", None)
            if fed_storage:
                storage = Path(str(fed_storage))
                for name in (
                    FEDERATION_COMPOSITE_SCHEMA_FILENAME,
                    ARTIFACT_MANIFEST_FILENAME,
                    FEDERATION_MANIFEST_FILENAME,
                    FEDERATION_MAPPINGS_FILENAME,
                ):
                    src = storage / name
                    if src.is_file():
                        shutil.copy2(src, fed_baseline / name)
            artifacts_parent = Path(str(fed_sb.artifacts_dir))
            for source in fed_manifest.sources:
                source_id = str(source.source_id or "").strip()
                member_src = Path(
                    aetherdialect._sandbox.Sandbox._sandbox_federation_member_artifacts_dir(
                        str(artifacts_parent),
                        source_id,
                    )
                )
                if not member_src.is_dir():
                    slug = federation_source_storage_slug(source)
                    member_src = artifacts_parent / ARTIFACT_DIRECTORY_SEGMENT / slug
                member_dest = fed_baseline / source_id
                if member_src.is_dir():
                    if member_dest.exists():
                        shutil.rmtree(member_dest)
                    shutil.copytree(member_src, member_dest)
            composite_dest = fed_baseline / FEDERATION_COMPOSITE_SCHEMA_FILENAME
            schema_dest = fed_baseline / "schema_graph.json.gz"
            if composite_dest.is_file() and not schema_dest.is_file():
                shutil.copy2(composite_dest, schema_dest)
            elif schema_dest.is_file() and not composite_dest.is_file():
                shutil.copy2(schema_dest, composite_dest)
    except Exception as exc:
        raise RuntimeError(f"federation baseline build failed: {exc}") from exc
    finally:
        aetherdialect._sandbox.Sandbox._write_sandbox_toml = orig_write_toml
        EngineConfig.LLM_PROVIDER = prev_provider
    verbose_message(f"Wrote federation artifacts baseline under {fed_baseline}")


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
    aetherdialect._sandbox.Sandbox._pin_bundled_schema_literals(staging_dir)


def ensure_interpret_domain(staging_dir: Path = STAGING) -> None:
    """Write schema_interpret_domain.json to staging when missing."""
    target = staging_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if target.is_file():
        return
    if not (staging_dir / "rental_shop.sql").is_file():
        return
    if not (staging_dir / "rental_shop_data").is_dir():
        return
    with AetherEngine.offline_sandbox(
        cleanup_artifacts=True,
        bundle_dir=str(staging_dir),
        maintainer_access=True,
    ) as handle:
        domain = json.loads(handle.engine._schema_graph.schema_payload_interpret())
    target.write_text(json.dumps(domain, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


_QUESTION_GATEKEEPER_MARKER = "valid_database_question"
_QUESTION_NORMALIZE_MARKER = "rewritten canonical short query"


def sync_gatekeeper_normalization_fixture_questions(corpus: FixtureCorpus) -> int:
    """Align normalization and intent question fields with gatekeeper corrected text."""
    corrected_by_raw: dict[str, str] = {}
    for row in corpus.fixtures:
        if str(row.get("task", "")) != "default":
            continue
        system = str(row.get("system", ""))
        user = str(row.get("user", "")).strip()
        if _QUESTION_GATEKEEPER_MARKER not in system or user.startswith("{"):
            continue
        try:
            payload = json.loads(str(row.get("output_text", "")))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        corrected = str(payload.get("corrected", "") or user).strip()
        if not corrected:
            continue
        corrected_by_raw[user] = corrected
        corrected_by_raw[normalize_question(user)] = corrected

    if not corrected_by_raw:
        return 0

    replacements: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in corpus.fixtures:
        task = str(row.get("task", ""))
        system = str(row.get("system", ""))
        user = str(row.get("user", "")).strip()
        if not user.startswith("{"):
            continue
        try:
            body = json.loads(user)
        except json.JSONDecodeError:
            continue
        if not isinstance(body, dict):
            continue
        question = str(body.get("question", "") or "").strip()
        if not question:
            continue
        if task == "default" and _QUESTION_NORMALIZE_MARKER not in system:
            continue
        corrected = corrected_by_raw.get(question) or corrected_by_raw.get(normalize_question(question))
        if not corrected or corrected == question:
            continue
        body["question"] = corrected
        new_user = MockProvider.mock_fixture_user_key(stable_json(body))
        if new_user == user:
            continue
        replacements[fixture_key(row)] = {
            "task": task,
            "system": system,
            "user": new_user,
            "output_text": str(row.get("output_text", "")),
        }

    if not replacements:
        return 0
    updated: list[dict[str, str]] = []
    for row in corpus.fixtures:
        updated.append(replacements.get(fixture_key(row), row))
    corpus.fixtures = updated
    corpus.seen = {fixture_key(row) for row in corpus.fixtures}
    corpus.flush()
    return len(replacements)


def repair_federation_intent_schema_literals(corpus: FixtureCorpus, staging_dir: Path = STAGING) -> int:
    """Backfill federation intent fixture schema_literal_json from the bundled composite graph."""
    from aetherdialect._llm_provider import MockProvider
    from aetherdialect._sandbox import Sandbox

    fixtures_path = staging_dir / "fixtures" / "rental_shop_mock.json"
    if not fixtures_path.is_file():
        return 0

    markers = sorted(
        {
            str(row.get("question", "")).strip().lower()
            for row in _load_scenarios_by_question().values()
            if str(row.get("recipe", "")) == "federation" and str(row.get("question", "")).strip()
        },
    )
    if not markers:
        return 0

    with Sandbox._federation_offline_handle_cm(
        AetherEngine,
        bundle_dir=str(staging_dir),
        maintainer_access=True,
        cleanup_artifacts=True,
    ) as handle:
        schema_literal = MockProvider.stable_schema_literal(handle.engine._schema_graph.schema_literal_json)

    def _is_federation_intent_fixture(user: str) -> bool:
        lowered = user.lower()
        return any(marker in lowered for marker in markers)

    replacements: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in corpus.fixtures:
        if str(row.get("task", "")) != "intent":
            continue
        user = str(row.get("user", "")).strip()
        if not user.startswith("{") or not _is_federation_intent_fixture(user):
            continue
        try:
            body = json.loads(user)
        except json.JSONDecodeError:
            continue
        if not isinstance(body, dict) or "schema_literal_json" not in body:
            continue
        embedded = body.get("schema_literal_json")
        if isinstance(embedded, dict):
            embedded_text = MockProvider.stable_schema_literal(json.dumps(embedded, ensure_ascii=False))
        else:
            embedded_text = MockProvider.stable_schema_literal(str(embedded or "{}"))
        if embedded_text == schema_literal:
            continue
        body["schema_literal_json"] = schema_literal
        new_user = MockProvider.mock_fixture_user_key(stable_json(body))
        if new_user == user:
            continue
        replacements[fixture_key(row)] = {
            "task": str(row.get("task", "")),
            "system": str(row.get("system", "")),
            "user": new_user,
            "output_text": str(row.get("output_text", "")),
        }

    if not replacements:
        return 0
    updated: list[dict[str, str]] = []
    for row in corpus.fixtures:
        updated.append(replacements.get(fixture_key(row), row))
    corpus.fixtures = updated
    corpus.seen = {fixture_key(row) for row in corpus.fixtures}
    corpus.flush()
    return len(replacements)


def recanonicalize_mock_fixture_user_keys(corpus: FixtureCorpus) -> int:
    """Re-key fixture rows to the stable-json mock lookup form."""
    replacements: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in corpus.fixtures:
        user = str(row.get("user", "")).strip()
        if not user:
            continue
        new_user = MockProvider.mock_fixture_user_key(user)
        if new_user == user:
            continue
        key = fixture_key(row)
        replacements[key] = {
            "task": str(row.get("task", "")),
            "system": str(row.get("system", "")),
            "user": new_user,
            "output_text": str(row.get("output_text", "")),
        }
    if not replacements:
        return 0
    updated: list[dict[str, str]] = []
    for row in corpus.fixtures:
        key = fixture_key(row)
        updated.append(replacements.get(key, row))
    corpus.fixtures = updated
    corpus.seen = {fixture_key(row) for row in corpus.fixtures}
    corpus.flush()
    return len(replacements)


def normalize_fixture_corpus_schema_domains(corpus: FixtureCorpus) -> int:
    """Re-key intent fixtures so schema_domain matches the bundled interpret payload."""
    domain_path = STAGING / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if not domain_path.is_file():
        return 0
    payload = json.loads(domain_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    domain = payload
    replacements: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in corpus.fixtures:
        if str(row.get("task", "")) != "intent":
            continue
        user = str(row.get("user", "")).strip()
        if not user.startswith("{"):
            continue
        try:
            body = json.loads(user)
        except json.JSONDecodeError:
            continue
        if not isinstance(body, dict) or "schema_domain" not in body:
            continue
        body["schema_domain"] = domain
        new_user = MockProvider.mock_fixture_user_key(stable_json(body))
        if new_user == user:
            continue
        key = fixture_key(row)
        replacements[key] = {
            "task": str(row.get("task", "")),
            "system": str(row.get("system", "")),
            "user": new_user,
            "output_text": str(row.get("output_text", "")),
        }
    if not replacements:
        return 0
    updated: list[dict[str, str]] = []
    for row in corpus.fixtures:
        key = fixture_key(row)
        updated.append(replacements.get(key, row))
    corpus.fixtures = updated
    corpus.seen = {fixture_key(row) for row in corpus.fixtures}
    corpus.flush()
    return len(replacements)


def _prepare_artifact_dir(path: str) -> str:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


class WarmRecordingPool:
    """Reuse one DuckDB connection; isolate mock-verify artifacts per run."""

    def __init__(self, bundle_dir: Path) -> None:
        self._bundle_dir = bundle_dir
        SANDBOX_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
        self._live_artifacts_owner = _prepare_artifact_dir(str(SANDBOX_ARTIFACTS_ROOT / "live_owner"))
        self._live_artifacts_consumer = _prepare_artifact_dir(str(SANDBOX_ARTIFACTS_ROOT / "live_consumer"))
        self._live_artifacts_owner_views = _prepare_artifact_dir(str(SANDBOX_ARTIFACTS_ROOT / "live_owner_views"))
        self._live_artifacts_consumer_views = _prepare_artifact_dir(
            str(SANDBOX_ARTIFACTS_ROOT / "live_consumer_views"),
        )
        self._live_artifacts_federation = _prepare_artifact_dir(str(SANDBOX_ARTIFACTS_ROOT / "live_federation"))
        self._connection = aetherdialect._sandbox.Sandbox._load_main_memory_connection(bundle_dir)
        self._engine_cls = AetherEngine
        self._live_handles: dict[str, object] = {}
        self._closed = False
        self._literals_pinned = False

    def _artifacts_for_construction(
        self,
        construction: SlotConstruction,
    ) -> str:
        if construction.include == "views":
            if construction.role == "consumer":
                return self._live_artifacts_consumer_views
            return self._live_artifacts_owner_views
        if construction.surface == "federation":
            return self._live_artifacts_federation
        if construction.role == "consumer":
            return self._live_artifacts_consumer
        return self._live_artifacts_owner

    def _pin_schema_literals(self) -> None:
        if self._literals_pinned:
            return
        aetherdialect._sandbox.Sandbox._pin_bundled_schema_literals(self._bundle_dir)
        self._literals_pinned = True

    def _open_construction_handle(
        self,
        *,
        construction: SlotConstruction,
        artifacts_dir: str,
        cleanup_artifacts: bool,
    ) -> object:
        """Open a production-shaped offline handle for recording."""
        role = SchemaRole.CONSUMER if construction.role == "consumer" else SchemaRole.OWNER
        engine_context = _engine_context_for_construction(construction)
        if construction.surface == "federation":
            cm = aetherdialect._sandbox.Sandbox._federation_offline_handle_cm(
                self._engine_cls,
                artifacts_dir=artifacts_dir,
                cleanup_artifacts=cleanup_artifacts,
                bundle_dir=str(self._bundle_dir),
                maintainer_access=True,
            )
        else:
            cm = aetherdialect._sandbox.Sandbox._offline_handle_cm(
                self._engine_cls,
                role=role,
                engine_context=engine_context,
                include=construction.include,
                artifacts_dir=artifacts_dir,
                cleanup_artifacts=cleanup_artifacts,
                bundle_dir=str(self._bundle_dir),
                maintainer_access=True,
                connection=self._connection,
                owns_connection=False,
            )
        handle = cm.__enter__()
        handle._lifecycle_cm = cm
        return handle

    def live_handle(
        self,
        *,
        construction: SlotConstruction | None = None,
    ) -> object:
        if self._closed:
            raise RuntimeError("Warm recording pool is closed")
        resolved = construction or SlotConstruction()
        cache_key = _construction_cache_key(resolved)
        cached = self._live_handles.get(cache_key)
        if cached is not None:
            return cached
        _reset_sandbox_duckdb_runtime()
        artifacts_dir = self._artifacts_for_construction(resolved)
        _seed_engine_baseline(
            artifacts_dir=artifacts_dir,
            bundle_dir=self._bundle_dir,
            preset=_baseline_preset_for_construction(resolved),
            include=resolved.include,
        )
        if resolved.surface == "federation" or resolved.role == "consumer":
            handle = self._open_construction_handle(
                construction=resolved,
                artifacts_dir=artifacts_dir,
                cleanup_artifacts=False,
            )
        else:
            handle = self._engine_cls.offline_sandbox(
                bundle_dir=str(self._bundle_dir),
                maintainer_access=True,
                connection=self._connection,
                owns_connection=False,
                artifacts_dir=artifacts_dir,
                cleanup_artifacts=False,
                include=resolved.include,
            )
        self._live_handles[cache_key] = handle
        return handle

    def evict_live_handle(
        self,
        *,
        construction: SlotConstruction | None = None,
    ) -> None:
        """Drop a cached live handle so bundled overrides cannot pollute later recordings."""
        resolved = construction or SlotConstruction()
        cache_key = _construction_cache_key(resolved)
        handle = self._live_handles.pop(cache_key, None)
        if handle is not None:
            handle.close()

    def _ephemeral_handle(
        self,
        *,
        construction: SlotConstruction,
        fixtures_file: str,
        provider: str,
        live_recording: bool = False,
    ) -> tuple[object, str]:
        del provider, fixtures_file
        self._pin_schema_literals()
        verify_artifacts = tempfile.mkdtemp(prefix="aetherdialect_record_verify_")
        _reset_sandbox_duckdb_runtime()
        _seed_engine_baseline(
            artifacts_dir=verify_artifacts,
            bundle_dir=self._bundle_dir,
            preset=_baseline_preset_for_construction(construction),
            include=construction.include,
        )
        if construction.surface == "federation" or construction.role == "consumer":
            handle = self._open_construction_handle(
                construction=construction,
                artifacts_dir=verify_artifacts,
                cleanup_artifacts=True,
            )
            return handle, verify_artifacts
        kwargs: dict[str, object] = {
            "bundle_dir": str(self._bundle_dir),
            "maintainer_access": True,
            "connection": self._connection,
            "owns_connection": False,
            "artifacts_dir": verify_artifacts,
            "cleanup_artifacts": True,
            "include": construction.include,
        }
        if live_recording:
            handle = self._engine_cls.offline_sandbox(**kwargs)
            return handle, verify_artifacts
        orig_write = aetherdialect._sandbox.Sandbox._write_sandbox_toml

        def mock_toml(*, fixtures_file: str) -> str:
            return orig_write(fixtures_file=fixtures_file)

        aetherdialect._sandbox.Sandbox._write_sandbox_toml = mock_toml
        try:
            handle = self._engine_cls.offline_sandbox(**kwargs)
        finally:
            aetherdialect._sandbox.Sandbox._write_sandbox_toml = orig_write
        return handle, verify_artifacts

    def run_live(
        self,
        question: str,
        *,
        construction: SlotConstruction | None = None,
        mode: str | None = None,
    ) -> object:
        self._pin_schema_literals()
        resolved = construction or SlotConstruction()
        handle = self.live_handle(construction=resolved)
        session_cm = handle.engine.session(mode=mode) if mode else handle.engine.session()
        with session_cm as session:
            return session.accept_until_done(question)

    def run_mock(
        self,
        question: str,
        *,
        construction: SlotConstruction,
        mode: str | None = None,
        fixtures_file: str,
        apply_structure: bool = False,
    ) -> tuple[object | None, str]:
        """Run one mock-provider question, flattening handcrafted slot fixtures when needed."""

        self._pin_schema_literals()
        MockProvider.reset_mock_provider()

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
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
                json.dump({"fixtures": flat}, tmp)
                tmp.close()
                effective_fixtures = tmp.name
                cleanup_fixtures = True
        except Exception:
            pass

        handle, artifacts_dir = self._ephemeral_handle(
            construction=construction,
            fixtures_file=effective_fixtures,
            provider="mock",
        )
        prev_provider = EngineConfig.LLM_PROVIDER
        prev_fixtures = EngineConfig.MOCK_FIXTURES_FILE
        EngineConfig.LLM_PROVIDER = "mock"
        EngineConfig.MOCK_FIXTURES_FILE = effective_fixtures
        try:
            if apply_structure:
                _stage_demo_schema_structure(handle)
            from aetherdialect._sandbox import Sandbox

            with Sandbox.federation_scenario_session(handle.engine, question, mode=mode) as session:
                step = session.accept_until_done(question)
            return step, ""
        except Exception as exc:
            return None, _short_retry_reason(str(exc), mock=True)
        finally:
            handle.close()
            MockProvider.reset_mock_provider()
            EngineConfig.LLM_PROVIDER = prev_provider
            EngineConfig.MOCK_FIXTURES_FILE = prev_fixtures

            aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
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

        for artifacts_dir in (
            self._live_artifacts_owner,
            self._live_artifacts_consumer,
            self._live_artifacts_owner_views,
            self._live_artifacts_consumer_views,
            self._live_artifacts_federation,
        ):
            aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
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


def _is_pass_but_wrong_question(question: str) -> bool:
    return question.strip() == PASS_BUT_WRONG_QUESTION


def _handcrafted_question_for_slot(slot: RecordingSlot) -> str:
    """Return the question key used to match sandbox_handcrafted_fixtures rows."""
    if slot.kind == "question" and _is_pass_but_wrong_question(slot.label):
        return PASS_BUT_WRONG_QUESTION
    return slot.label


@contextmanager
def _patched_pass_but_wrong_ground_parse() -> Any:
    """Accept handcrafted wrong-film ground IR during pass_but_wrong recording only.

    Live ask uses :mod:`aetherdialect._intent_loop`'s imported
    ``parse_logical_intent_response`` binding, so both that module and
    ``_intent_expr`` are patched. Handcrafted payloads with a ``filter``
    key are remapped to ``where`` before schema validation.
    """
    import aetherdialect._intent_expr
    import aetherdialect._intent_loop
    from aetherdialect._intent_expr import (
        _logical_intent_schema_issues,
        logical_intent_from_parsed,
        safe_json_loads,
    )

    saved_expr = aetherdialect._intent_expr.parse_logical_intent_response
    saved_loop = aetherdialect._intent_loop.parse_logical_intent_response

    def _remap_filter_key_to_where(obj: dict[str, Any]) -> dict[str, Any]:
        if "filter" not in obj or "where" in obj:
            return obj
        remapped = dict(obj)
        remapped["where"] = remapped.pop("filter")
        return remapped

    def _parse(raw: str, schema_graph: Any) -> tuple[Any, list[Any]]:
        result, issues = saved_expr(raw, schema_graph)
        if result is not None:
            return result, issues
        obj = safe_json_loads(raw.strip())
        if not isinstance(obj, dict):
            return result, issues
        obj = _remap_filter_key_to_where(obj)
        schema_issues = _logical_intent_schema_issues(obj)
        if schema_issues:
            return result, issues
        logical = logical_intent_from_parsed(obj)
        if not logical.tables or not logical.select.strip():
            return result, issues
        tables = tuple(str(t).strip().lower() for t in logical.tables if str(t).strip())
        if tables == ("film",) and "film" in logical.select.lower():
            return logical, []
        return result, issues

    aetherdialect._intent_expr.parse_logical_intent_response = _parse
    aetherdialect._intent_loop.parse_logical_intent_response = _parse
    try:
        yield
    finally:
        aetherdialect._intent_expr.parse_logical_intent_response = saved_expr
        aetherdialect._intent_loop.parse_logical_intent_response = saved_loop


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
        str(row.get("task", "")) == "default" and PARAM_EXTRACTION_SYSTEM_MARKER in str(row.get("system", "")).lower()
    )


def _intent_fixture_refs_question(row: dict[str, str], question: str) -> bool:
    if str(row.get("task", "")) != "intent":
        return False
    label = question.strip().lower()
    if not label:
        return False
    return label in str(row.get("user", "")).lower()


def _interpret_outputs_for_question(fixtures: list[dict[str, str]], question: str) -> list[str]:
    label = question.strip().lower()
    if not label:
        return []
    out: list[str] = []
    for row in fixtures:
        if str(row.get("task", "")) != "intent":
            continue
        if str(row.get("system", "")) != INTENT_INTERPRET_SYSTEM:
            continue
        if label not in str(row.get("user", "")).lower():
            continue
        text = str(row.get("output_text", ""))
        if text:
            out.append(text)
    return out


def _row_chains_from_interpret_outputs(row: dict[str, str], interpret_outputs: list[str]) -> bool:
    if not interpret_outputs:
        return False
    user = str(row.get("user", ""))
    for text in interpret_outputs:
        if not text:
            continue
        if text in user:
            return True
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        plan = parsed.get("interpret_plan") if isinstance(parsed, dict) else None
        if not isinstance(plan, dict):
            plan = parsed if isinstance(parsed, dict) else None
        if not isinstance(plan, dict):
            continue
        approach = str(plan.get("approach", ""))
        if approach and approach in user:
            return True
        plan_json = json.dumps(plan, ensure_ascii=False)
        if plan_json in user:
            return True
    return False


def _expand_intent_chain_keep_keys(
    fixtures: list[dict[str, str]],
    seed_keys: set[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    keep = set(seed_keys)
    outputs = {
        str(row.get("output_text", ""))
        for row in fixtures
        if fixture_key(row) in keep and str(row.get("output_text", ""))
    }
    changed = True
    while changed:
        changed = False
        for row in fixtures:
            if str(row.get("task", "")) != "intent":
                continue
            key = fixture_key(row)
            if key in keep:
                continue
            if str(row.get("system", "")) not in {INTENT_GROUND_SYSTEM, INTENT_COMPOSE_SYSTEM}:
                continue
            if not _row_chains_from_interpret_outputs(row, list(outputs)):
                continue
            keep.add(key)
            text = str(row.get("output_text", ""))
            if text and text not in outputs:
                outputs.add(text)
                changed = True
    return keep


def _active_intent_keep_keys_for_question(
    fixtures: list[dict[str, str]],
    question: str,
) -> set[tuple[str, str, str]]:
    label = question.strip().lower()
    if not label:
        return set()
    active_interpret: dict[str, str] | None = None
    for row in fixtures:
        if str(row.get("task", "")) != "intent":
            continue
        if str(row.get("system", "")) != INTENT_INTERPRET_SYSTEM:
            continue
        if label not in str(row.get("user", "")).lower():
            continue
        active_interpret = row
    if active_interpret is None:
        return set()
    return _expand_intent_chain_keep_keys(fixtures, {fixture_key(active_interpret)})


def _prune_orphan_intent_rows(
    fixtures: list[dict[str, str]],
    question: str,
    keep_keys: set[tuple[str, str, str]],
) -> tuple[list[dict[str, str]], int]:
    label = question.strip().lower()
    if not label or not keep_keys:
        return fixtures, 0
    interpret_outputs = _interpret_outputs_for_question(fixtures, question)
    pruned: list[dict[str, str]] = []
    removed = 0
    for row in fixtures:
        if str(row.get("task", "")) != "intent":
            pruned.append(row)
            continue
        key = fixture_key(row)
        if key in keep_keys:
            pruned.append(row)
            continue
        user = str(row.get("user", ""))
        system = str(row.get("system", ""))
        belongs = label in user.lower()
        if not belongs and system in {INTENT_GROUND_SYSTEM, INTENT_COMPOSE_SYSTEM}:
            belongs = _row_chains_from_interpret_outputs(row, interpret_outputs)
        if belongs:
            removed += 1
            continue
        pruned.append(row)
    return pruned, removed


def prune_orphan_intent_fixtures(
    corpus: FixtureCorpus,
    questions: dict[str, list[str]],
    *,
    manifest: dict[str, object] | None = None,
    committed_only: bool = True,
) -> int:
    """Drop superseded intent fixtures for committed question slots."""
    manifest = manifest or _load_recording_manifest()
    committed = committed_slot_ids(manifest) if committed_only else None
    total_removed = 0
    for slot in iter_recording_slots(questions):
        if slot.kind != "question":
            continue
        if committed_only and slot_id_for(slot) not in committed:
            continue
        keep_keys = _active_intent_keep_keys_for_question(corpus.fixtures, slot.label)
        if not keep_keys:
            continue
        corpus.fixtures, removed = _prune_orphan_intent_rows(corpus.fixtures, slot.label, keep_keys)
        total_removed += removed
    if total_removed:
        corpus.seen = {fixture_key(row) for row in corpus.fixtures}
        corpus.flush()
    return total_removed


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
                "user": llm_mod.MockProvider.mock_fixture_user_key(swapped_user),
                "output_text": swapped_output,
            }
        )
    return out


def _recipe_for_slot(slot: RecordingSlot, scenario: dict[str, object] | None = None) -> str:
    """Return the internal sandbox recipe for *slot* from scenario metadata."""
    if scenario is None:
        scenario = _load_scenarios_by_question().get(slot.label.lower(), {})
    mechanism = str(scenario.get("mechanism", ""))
    recipe = str(scenario.get("recipe", ""))
    if mechanism.startswith("federation_") or recipe == "federation":
        return "federation"
    if mechanism == "views_scope" or recipe == "views":
        return "views"
    return "single"


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
        self._orig_chat = aetherdialect._llm_provider.LLMProvider.chat
        self._orig_json = aetherdialect._llm_provider.LLMProvider.json
        self._set_llm_chat(self._recording_chat)
        self._set_llm_json(self._recording_json)

    def _set_llm_chat(self, hook: Callable[..., str]) -> None:
        self._llm_mod.LLMProvider.chat = hook
        for mod_name in self.env.llm_patch_modules:
            mod = __import__(mod_name, fromlist=["LLMProvider"])
            if hasattr(mod, "LLMProvider"):
                mod.LLMProvider.chat = hook

    def _set_llm_json(self, hook: Callable[..., dict[str, Any]]) -> None:
        self._llm_mod.LLMProvider.json = hook
        for mod_name in self.env.llm_patch_modules:
            mod = __import__(mod_name, fromlist=["LLMProvider"])
            if hasattr(mod, "LLMProvider"):
                mod.LLMProvider.json = hook

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
                user_for_llm = self._llm_mod.LLMProvider._llm_user_text_without_sensitivity_classification(user)
                user_key = self._llm_mod.MockProvider.mock_fixture_user_key(user_for_llm)
                self.corpus.record(task=task, system=system, user_key=user_key, output_text=result)
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
        result = self._orig_chat(system, user, max_retries=max_retries, timeout=timeout, task=task)
        user_for_llm = self._llm_mod.LLMProvider._llm_user_text_without_sensitivity_classification(user)
        user_key = self._llm_mod.MockProvider.mock_fixture_user_key(user_for_llm)
        self.corpus.record(task=task, system=system, user_key=user_key, output_text=result)
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
        result = self._orig_json(system, user, retries=retries, task=task)
        user_for_llm = self._llm_mod.LLMProvider._llm_user_text_without_sensitivity_classification(user)
        user_key = self._llm_mod.MockProvider.mock_fixture_user_key(user_for_llm)
        output_text = json.dumps(result, ensure_ascii=False)
        self.corpus.record(task=task, system=system, user_key=user_key, output_text=output_text)
        return result

    def _run_live_slot(self, slot: RecordingSlot) -> tuple[object | None, str]:
        if slot.kind == "question":
            self.pool._pin_schema_literals()
            construction = _construction_for_slot(slot)
            handle, artifacts_dir = self.pool._ephemeral_handle(
                construction=construction,
                fixtures_file=str(FIXTURES_PATH),
                provider="openai",
                live_recording=True,
            )
            try:
                if construction.apply_structure:
                    _stage_demo_schema_structure(handle)
                from aetherdialect._sandbox import Sandbox

                try:
                    with Sandbox.federation_scenario_session(
                        handle.engine,
                        slot.label,
                        mode=construction.mode,
                    ) as session:
                        step = session.accept_until_done(slot.label)
                        ok, detail = _check_slot_recording(step, slot)
                        return step, ("" if ok else detail)
                except Exception as exc:
                    return None, str(exc)
            finally:
                handle.close()
                aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
                shutil.rmtree(artifacts_dir, ignore_errors=True)
        if slot.kind == "feedback":
            return self._run_live_feedback(slot)
        if slot.kind == "space":
            return self._run_live_space(slot)
        return None, f"unknown slot kind {slot.kind!r}"

    def _run_live_space(self, slot: RecordingSlot) -> tuple[object | None, str]:
        from aetherdialect._contracts_base import ConfigError, SpaceContext

        if slot.label.strip() != "member_spaces":
            return None, f"unknown space packing slot {slot.label!r}"

        handle = self.pool.live_handle()
        try:
            for name, tables in SANDBOX_MEMBER_SPACE_TABLES.items():
                notes_name = SANDBOX_MEMBER_SPACE_NOTES_FILES[name]
                notes_path = STAGING / notes_name
                notes_arg = str(notes_path) if notes_path.is_file() else None
                context = SpaceContext(
                    tables=tables,
                    columns=frozenset(),
                    notes_file=notes_arg,
                )
                existing_uid: str | None = None
                try:
                    existing_uid = str(handle.engine.aetherspace(name).uid)
                except ConfigError:
                    existing_uid = None
                if existing_uid is not None:
                    handle.engine.aetherspace(name, space_context=context, uid=existing_uid)
                else:
                    handle.engine.aetherspace(name, space_context=context)
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
        handle = self.pool.live_handle()
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
    def _mock_replay_env(self) -> Any:
        """Serve LLM calls from the recorded corpus instead of the live provider.

        Recording installs global patches (``llm_chat``/``llm_json`` hooks plus an OpenAI
        sandbox TOML). Without undoing them the verify pass re-samples the live LLM and
        records those answers, so a slot that just passed capture can fail verification for
        reasons unrelated to the fixtures it produced.
        """
        self.corpus.flush()
        saved_toml = aetherdialect._sandbox.Sandbox._write_sandbox_toml
        prev_provider = EngineConfig.LLM_PROVIDER
        prev_fixtures = EngineConfig.MOCK_FIXTURES_FILE
        saved_pipeline_match = aetherdialect._pipeline_generate.match_question_level_template_reuse
        saved_main_match = aetherdialect._main_execution.match_question_level_template_reuse
        saved_live_match = aetherdialect._live_testing.match_question_level_template_reuse
        saved_persist = aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session
        self._set_llm_chat(self._orig_chat)
        self._set_llm_json(self._orig_json)
        aetherdialect._sandbox.Sandbox._write_sandbox_toml = self.env.orig_write_toml
        EngineConfig.LLM_PROVIDER = "mock"
        EngineConfig.MOCK_FIXTURES_FILE = str(FIXTURES_PATH)
        aetherdialect._pipeline_generate.match_question_level_template_reuse = self.env.skip_template_reuse
        aetherdialect._main_execution.match_question_level_template_reuse = self.env.skip_template_reuse
        aetherdialect._live_testing.match_question_level_template_reuse = self.env.skip_template_reuse
        aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session = (
            self.env.skip_template_learning
        )
        MockProvider.reset_mock_provider()
        try:
            yield
        finally:
            MockProvider.reset_mock_provider()
            EngineConfig.LLM_PROVIDER = prev_provider
            EngineConfig.MOCK_FIXTURES_FILE = prev_fixtures
            aetherdialect._pipeline_generate.match_question_level_template_reuse = saved_pipeline_match
            aetherdialect._main_execution.match_question_level_template_reuse = saved_main_match
            aetherdialect._live_testing.match_question_level_template_reuse = saved_live_match
            aetherdialect._main_execution.MainExecutionOps.persist_template_learning_for_pipeline_session = (
                saved_persist
            )
            aetherdialect._sandbox.Sandbox._write_sandbox_toml = saved_toml
            self._set_llm_chat(self._recording_chat)
            self._set_llm_json(self._recording_json)

    @contextmanager
    def _template_match_enabled(self) -> Any:
        saved_pipeline_match = aetherdialect._pipeline_generate.match_question_level_template_reuse
        saved_main_match = aetherdialect._main_execution.match_question_level_template_reuse
        saved_live_match = aetherdialect._live_testing.match_question_level_template_reuse
        aetherdialect._pipeline_generate.match_question_level_template_reuse = self.env.orig_template_match
        aetherdialect._main_execution.match_question_level_template_reuse = self.env.orig_template_match
        aetherdialect._live_testing.match_question_level_template_reuse = self.env.orig_template_match
        try:
            yield
        finally:
            aetherdialect._pipeline_generate.match_question_level_template_reuse = saved_pipeline_match
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
            construction=SlotConstruction(),
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
            MockProvider.reset_mock_provider()
            EngineConfig.LLM_PROVIDER = prev_provider
            EngineConfig.MOCK_FIXTURES_FILE = prev_fixtures
            aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)

    def _verify_mock_slot(self, slot: RecordingSlot) -> tuple[bool, str]:
        if not _slot_requires_mock_verify(slot):
            return True, ""
        if slot.kind == "feedback":
            with self._mock_replay_env():
                return self._verify_mock_feedback(slot, str(FIXTURES_PATH))
        with self._mock_replay_env():
            ok, detail = _verify_slot_with_validate(slot, pool=self.pool)
        if not ok:
            import sandbox_recording as lt_conftest

            append_failure_trace(
                build_session_step_trace(
                    scenario_id=slot_id_for(slot),
                    question=slot.label,
                    step=None,
                    error=detail or "validate failed",
                    captured_logs=[],
                ),
                lt_conftest._RESULTS_FILE,
            )
        return ok, detail

    def record_slot(self, slot: RecordingSlot) -> tuple[bool, str, int, object | None]:
        import sandbox_recording as lt_conftest

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

            with self._patched_handcrafted_entries(_handcrafted_question_for_slot(slot)) as handcrafted_active:
                with (
                    _patched_pass_but_wrong_ground_parse() if _is_pass_but_wrong_question(slot.label) else nullcontext()
                ):
                    with llm_usage_question_scope():
                        with pipeline_capture(auto_responses=auto) as capture:
                            _step, live_detail = self._run_live_slot(slot)

            if _step is not None:
                last_step_result = _step

            if live_detail:
                last_detail = live_detail
                self.corpus.discard_slot()
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
                    append_failure_trace(step_res, lt_conftest._RESULTS_FILE)

                if not _recording_error_retryable(live_detail):
                    break
                continue

            corpus_snap = _corpus_snapshot(self.corpus)
            self.corpus.commit_slot(prune_question="")
            mock_ok, mock_detail = self._verify_mock_slot(slot)
            if not mock_ok:
                last_detail = mock_detail or "validate failed"
                _restore_corpus_snapshot(self.corpus, corpus_snap)
                self.corpus.discard_slot()
                if _BUILD_VERBOSE:
                    corpus_message(
                        f"  retry {attempt}/{self.max_attempts} [{slot.tier}] {slot.label[:60]} "
                        f"({_short_retry_reason(last_detail, mock=True)})",
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
                    append_failure_trace(step_res, lt_conftest._RESULTS_FILE)
                if not _recording_error_retryable(last_detail):
                    break
                continue

            _finalize_fixture_corpus_repair(self.corpus)
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
        failed: list[str] = []
        from aetherdialect._utils_intent import generate_warmup_questions_freeform

        for slot, step in recorded_slots:
            intent_summary = getattr(step, "intent_summary", None)
            tables = [str(item) for item in getattr(intent_summary, "tables", ()) if str(item).strip()]
            if not tables:
                failed.append(f"{slot.label!r}: missing tables")
                continue
            construction = _construction_for_slot(slot)
            handle, artifacts_dir = self.pool._ephemeral_handle(
                construction=construction,
                fixtures_file=str(FIXTURES_PATH),
                provider="openai",
            )
            self.corpus.start_slot()
            try:
                if construction.apply_structure:
                    _stage_demo_schema_structure(handle)
                try:
                    raw = generate_paraphrases_of_seed_question(
                        slot.label,
                        handle.engine._schema_graph,
                        tables,
                    )
                    paraphrases = _clean_generated_paraphrases(slot.label, raw)
                    if not paraphrases:
                        raw_freeform = generate_warmup_questions_freeform(
                            handle.engine._schema_graph,
                            tables,
                            seed_question=slot.label,
                        )
                        paraphrases = _clean_generated_paraphrases(slot.label, raw_freeform)
                    if not paraphrases:
                        self.corpus.discard_slot()
                        failed.append(f"{slot.label!r}: no paraphrases generated")
                        continue
                    self.corpus.commit_slot()
                finally:
                    handle.close()
                    aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
                    shutil.rmtree(artifacts_dir, ignore_errors=True)
            except Exception as exc:
                self.corpus.discard_slot()
                failed.append(f"{slot.label!r}: {exc}")
                continue
            new_rows.append({"canonical": slot.label, "paraphrases": paraphrases})
            for paraphrase in paraphrases:
                swaps = INLINE_REUSE_PARAM_COPY_RULES.get((slot.label, paraphrase))
                if swaps is None:
                    continue
                ok, detail = self.record_inline_reuse_param_fixtures(slot.label, paraphrase, swaps=swaps)
                if not ok:
                    failed.append(f"{slot.label!r} reuse: {detail}")
        for row in new_rows:
            merged_by_canonical[str(row["canonical"])] = row
        final_rows = list(merged_by_canonical.values())
        self.paraphrase_catalog_rows = final_rows
        if final_rows:
            write_sandbox_catalog_file(target_dir=STAGING, paraphrase_pairs=final_rows)
        if failed:
            return False, "; ".join(failed[:3]) + (f" (+{len(failed) - 3} more)" if len(failed) > 3 else "")
        return True, ""

    def record_reuse_param_fixtures(self) -> tuple[bool, str]:
        for key, swaps in INLINE_REUSE_PARAM_COPY_RULES.items():
            canonical, paraphrase = key
            ok, detail = self.record_inline_reuse_param_fixtures(canonical, paraphrase, swaps=swaps)
            if not ok:
                return False, detail
        return True, ""

    def collect_paraphrase_seeds(self, slots: list[RecordingSlot]) -> list[tuple[RecordingSlot, object]]:
        seeds: list[tuple[RecordingSlot, object]] = []
        for slot in slots:
            self.corpus.start_slot()
            try:
                step, detail = self._run_live_slot(slot)
                if step is None or detail:
                    continue
                ok, _check_detail = _check_slot_recording(step, slot)
                if not ok:
                    continue
                seeds.append((slot, step))
            finally:
                self.corpus.discard_slot()
        return seeds

    def record_inline_reuse_param_fixtures(
        self,
        canonical: str,
        paraphrase: str,
        *,
        swaps: tuple[tuple[str, str], ...],
    ) -> tuple[bool, str]:
        handle = self.pool.live_handle()
        try:
            with self._template_match_enabled():
                with handle.engine.session() as session:
                    self.corpus.start_slot()
                    session.accept_until_done(canonical)
                    self.corpus.discard_slot()
                    self.corpus.start_slot()
                    step = session.accept_until_done(paraphrase)
                    if getattr(step, "error", None):
                        self.corpus.discard_slot()
                        return False, f"reuse capture failed for {canonical!r} -> {paraphrase!r}: {step.error}"
                    forward_rows = self.corpus.collapsed_slot_fixtures()
                    self.corpus.commit_slot()
        except Exception as exc:
            self.corpus.discard_slot()
            return False, str(exc)

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
        from aetherdialect._dialect_sqlglot_engines import DuckDBDialect

        demo_root = STAGING / "migration_demo"
        artifacts_src = demo_root / "artifacts_v1"
        map_path = demo_root / "schema_migration_map.json"
        if not artifacts_src.is_dir() or not map_path.is_file():
            return False, f"migration demo assets incomplete under {demo_root}"
        if not (STAGING / "rental_shop.sql").is_file() or not (STAGING / "rental_shop_data").is_dir():
            return False, "migration demo requires staged rental_shop.sql and rental_shop_data/"
        work = Path(tempfile.mkdtemp(prefix="aetherdialect_record_migration_"))
        config_file = ""
        self.corpus.start_slot()
        try:
            artifacts_dir = str(work / "artifacts")
            shutil.copytree(artifacts_src, artifacts_dir)
            connection = Sandbox._load_main_memory_connection(STAGING)
            Sandbox._apply_migration_column_renames(connection, map_path)
            execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
            notes_file = STAGING / "rental_shop_notes.txt"
            notes_arg = str(notes_file) if notes_file.is_file() else None
            sql_file = STAGING / "rental_shop.sql"
            sql_arg = str(sql_file) if sql_file.is_file() else None
            schema_context = Sandbox._owner_writer_schema_context(notes_file=notes_arg, sql_file=sql_arg)
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
            questions = self.questions.get("questions", [])
            post_q = questions[0] if questions else "How many films are in the Rental Shop catalog?"
            with t2s.session() as session:
                session.accept_until_done(post_q)
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
        import sandbox_recording as _invoice
        import sandbox_recording as lt_conftest

        slot_list = slots if slots is not None else iter_recording_slots(self.questions)
        total = len(slot_list)
        self.paraphrase_seeds_collected = []
        import aetherdialect.aetherdialect

        original_log_sink = aetherdialect.aetherdialect._init_log_sink
        aetherdialect.aetherdialect._init_log_sink = lambda _line: None
        _begin_eval_results(RECORDING_RESULTS_PATH, invoice_path=RECORDING_INVOICE_PATH)
        reuse_pair_failures: list[str] = []
        with llm_usage_session_scope():
            try:
                if _BUILD_VERBOSE:
                    corpus_message(f"Recording sandbox fixtures ({total} slots)...")
                for idx, slot in enumerate(slot_list, 1):
                    ok, detail, attempts, step_result = self.record_slot(slot)
                    label = slot.label[:70]
                    if ok:
                        suffix = f" — retry {attempts}/{self.max_attempts}" if attempts > 1 else ""
                        corpus_message(f"[recording] {idx}/{total} OK [{slot.tier}] {label}{suffix}")
                        lt_conftest.append_results_summary_line(f"OK [{slot.tier}] {slot.label}")
                        if (
                            slot.kind == "question"
                            and slot.tier in PARAPHRASE_SOURCE_TIERS
                            and slot.preset == "owner_writer"
                            and slot.mode in (None, "writer")
                            and _paraphrase_eligible_question(slot.label, kind=slot.kind)
                            and step_result is not None
                        ):
                            self.paraphrase_seeds_collected.append((slot, step_result))
                        if record_reuse_pairs and (
                            slot.kind == "question"
                            and slot.tier in PARAPHRASE_SOURCE_TIERS
                            and slot.preset == "owner_writer"
                            and slot.mode in (None, "writer")
                            and _paraphrase_eligible_question(slot.label, kind=slot.kind)
                        ):
                            for paraphrase in INLINE_PARAPHRASE_COPY_RULES.get(slot.label, ()):
                                swaps = INLINE_REUSE_PARAM_COPY_RULES.get((slot.label, paraphrase))
                                if swaps is None:
                                    continue
                                reuse_ok, reuse_detail = self.record_inline_reuse_param_fixtures(
                                    slot.label,
                                    paraphrase,
                                    swaps=swaps,
                                )
                                if not reuse_ok:
                                    reuse_pair_failures.append(
                                        f"{slot.label!r} -> {paraphrase!r}: {reuse_detail}",
                                    )
                    else:
                        reason = _short_retry_reason(detail)
                        retry_note = f" — retry {attempts}/{self.max_attempts}" if attempts > 1 else ""
                        corpus_message(
                            f"[recording] {idx}/{total} FAIL [{slot.tier}] {label}{retry_note} ({reason})",
                        )
                        lt_conftest.append_results_summary_line(f"FAIL [{slot.tier}] {slot.label}: {reason}")
                        if step_result is not None:
                            lt_conftest.append_live_failure_trace(step_result)
                    _invoice.flush_invoice_file()
            finally:
                _invoice.append_run_total_invoice()
                aetherdialect.aetherdialect._init_log_sink = original_log_sink
        slot_failures = bool(self.failed_slots)
        if slot_failures:
            lines = [f"  [{slot.tier}] {slot.label}" for slot in self.failed_slots]
            corpus_message(
                f"{len(self.failed_slots)} slot(s) failed recording. "
                f"See {lt_conftest._RESULTS_FILE}:\n" + "\n".join(lines),
            )
        if reuse_pair_failures:
            corpus_message(
                f"{len(reuse_pair_failures)} reuse-pair capture(s) failed:\n"
                + "\n".join(f"  {line}" for line in reuse_pair_failures),
            )
        return not self.failed_slots and not reuse_pair_failures

    def close(self) -> None:
        self._llm_mod.LLMProvider.chat = self._orig_chat
        self._llm_mod.LLMProvider.json = self._orig_json


def pack_bundled_aetherspace_snapshots(pool: WarmRecordingPool) -> None:
    """Copy recorded aetherspace snapshots into staging for data.zip."""
    from aetherdialect._constants import AETHERSPACES_SEGMENT
    from aetherdialect._contracts_base import SpaceContext

    handle = pool.live_handle()
    artifacts_root = Path(str(handle.engine._artifacts_dir))
    src_dir = artifacts_root / AETHERSPACES_SEGMENT

    def _named_spaces(root: Path) -> set[str]:
        names: set[str] = set()
        if not root.is_dir():
            return names
        for path in root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                name = str(payload.get("name") or "").strip().lower()
                if name:
                    names.add(name)
        return names

    missing = [name for name in SANDBOX_MEMBER_SPACE_TABLES if name not in _named_spaces(src_dir)]
    if missing:
        from aetherdialect._contracts_base import ConfigError

        for name, tables in SANDBOX_MEMBER_SPACE_TABLES.items():
            notes_name = SANDBOX_MEMBER_SPACE_NOTES_FILES[name]
            notes_path = STAGING / notes_name
            notes_arg = str(notes_path) if notes_path.is_file() else None
            context = SpaceContext(
                tables=tables,
                columns=frozenset(),
                notes_file=notes_arg,
            )
            existing_uid: str | None = None
            try:
                existing_uid = str(handle.engine.aetherspace(name).uid)
            except ConfigError:
                existing_uid = None
            if existing_uid is not None:
                handle.engine.aetherspace(name, space_context=context, uid=existing_uid)
            else:
                handle.engine.aetherspace(name, space_context=context)
        src_dir = artifacts_root / AETHERSPACES_SEGMENT
    dest = STAGING / "artifacts_baseline" / AETHERSPACES_SEGMENT
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(src_dir.glob("*.json")):
        shutil.copy2(path, dest / path.name)
        copied += 1
    if not copied:
        raise RuntimeError(f"no aetherspace snapshots found under {src_dir}")
    packed_names = _named_spaces(dest)
    missing_packed = [name for name in SANDBOX_MEMBER_SPACE_TABLES if name not in packed_names]
    if missing_packed:
        raise RuntimeError(f"missing bundled aetherspace snapshot(s) for member spaces: {missing_packed!r}")
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
    _post_record_corpus_hygiene(corpus, session.questions)
    _finalize_fixture_corpus_repair(corpus)
    run_fixture_fan_out(corpus, questions=session.questions, manifest=_load_recording_manifest())
    write_build_fingerprint()
    print(f"Wrote {len(corpus.fixtures)} fixtures to {FIXTURES_PATH}", flush=True)
    return ok


def record_corpus(record_reuse_pairs: bool = False) -> bool:
    ensure_schema_literals()
    ensure_interpret_domain()
    pin_staging_mock_fixture_keys()
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
            session.record_all(record_reuse_pairs=record_reuse_pairs)
            manifest = _load_recording_manifest()
            missing_para = _missing_paraphrase_canonicals(questions, manifest)
            need_para = bool(missing_para)
            need_reuse = not _reuse_fixtures_ready(corpus)
            need_aetherspace = not _aetherspace_snapshots_ready()
            need_migration = not _migration_fixtures_ready(corpus)
            paraphrase_seeds: list[tuple[RecordingSlot, object]] | None = None
            if need_para:
                paraphrase_seeds = _paraphrase_seeds_for_missing(
                    session,
                    missing_para,
                    session.paraphrase_seeds_collected,
                )
            finalize_recording_tail(
                session,
                pool,
                corpus,
                paraphrase_seeds=paraphrase_seeds,
                capture_paraphrases=need_para,
                require_paraphrase_seeds=need_para,
                capture_reuse=need_reuse and not need_para,
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
        staging_dir / "rental_shop.sql",
        staging_dir / "rental_shop_views.sql",
        staging_dir / "rental_shop_notes.txt",
        staging_dir / "questions.txt",
        staging_dir / "schema_structure_demo.json",
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


def run_fixture_fan_out(
    corpus: FixtureCorpus,
    questions: dict[str, list[str]] | None = None,
    manifest: dict[str, object] | None = None,
    *,
    pool: WarmRecordingPool | None = None,
) -> int:
    """Derive cross-surface fixture keys from canonical owner-writer recordings (zero LLM)."""
    questions = questions or load_staging_questions()
    manifest = manifest if manifest is not None else _load_recording_manifest()
    committed = committed_slot_ids(manifest)
    slots = [slot for slot in iter_recording_slots(questions) if slot_id_for(slot) in committed]
    if not slots:
        return 0
    pin_staging_mock_fixture_keys()
    literals_path = STAGING / SANDBOX_SCHEMA_LITERALS_FILENAME
    literals: dict[str, str] | None = None
    if literals_path.is_file():
        payload = json.loads(literals_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            owner = str(payload.get("owner", "")).strip()
            consumer = str(payload.get("consumer", "")).strip()
            if owner and consumer:
                literals = {"owner": owner, "consumer": consumer}
    added = apply_fixture_fan_out(
        corpus,
        slots,
        construction_for_slot=_construction_for_slot,
        recipe_for_slot=_recipe_for_slot,
        federation_ineligible=FEDERATION_INELIGIBLE_QUESTIONS,
        fixture_key=fixture_key,
        literals=literals,
    )
    owns_pool = pool is None and STAGING.is_dir()
    replay_pool = pool
    if owns_pool:
        replay_pool = WarmRecordingPool(STAGING)
    try:
        if replay_pool is not None:
            added += _harvest_fixture_fan_out_via_replay(corpus, slots, replay_pool)
    finally:
        if owns_pool and replay_pool is not None:
            replay_pool.close()
    if added:
        corpus_message(f"[fan-out] added {added} derived fixture row(s)")
    return added


def _canonical_fixture_output_index(
    fixtures: list[dict[str, str]],
) -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}
    for row in fixtures:
        question = fixture_question_for_row(row)
        if not question:
            continue
        key = (str(row.get("task", "")), str(row.get("system", "")), question)
        index[key] = str(row.get("output_text", ""))
    return index


def _fixture_output_for_key(corpus: FixtureCorpus, task: str, system: str, user_key: str) -> str | None:
    key = (task, system, user_key)
    for row in corpus.fixtures:
        if fixture_key(row) == key:
            return str(row.get("output_text", ""))
    return None


def _harvest_fixture_fan_out_via_replay(
    corpus: FixtureCorpus,
    slots: list[RecordingSlot],
    pool: WarmRecordingPool,
) -> int:
    """Replay fan-out targets under mock to harvest exact missed fixture keys."""
    import aetherdialect._llm_provider
    from aetherdialect._sandbox import Sandbox

    canonical_index = _canonical_fixture_output_index(corpus.fixtures)
    before = len(corpus.fixtures)
    orig_chat = aetherdialect._llm_provider.LLMProvider.chat
    orig_json = aetherdialect._llm_provider.LLMProvider.json

    def _lookup_canonical(task: str, system: str, user: str) -> str | None:
        probe = fixture_question_for_row({"task": task, "system": system, "user": user})
        if probe is None:
            return None
        return canonical_index.get((task, system, probe))

    def _fan_out_chat(
        system: str,
        user: str,
        max_retries: int = 3,
        timeout: Any = None,
        task: str = "default",
        **kwargs: object,
    ) -> str:
        del kwargs
        user_for_llm = aetherdialect._llm_provider.LLMProvider._llm_user_text_without_sensitivity_classification(user)
        user_key = aetherdialect._llm_provider.MockProvider.mock_fixture_user_key(user_for_llm)
        hit = _fixture_output_for_key(corpus, task, system, user_key)
        if hit is not None:
            return hit
        output = _lookup_canonical(task, system, user)
        if output is None:
            return orig_chat(system, user, max_retries=max_retries, timeout=timeout, task=task)
        corpus.record(task=task, system=system, user_key=user_key, output_text=output)
        return output

    def _fan_out_json(
        system: str,
        user: str,
        *,
        retries: int = 3,
        task: str = "default",
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        raw = _fan_out_chat(system, user, max_retries=retries, task=task)
        return json.loads(raw)

    aetherdialect._llm_provider.LLMProvider.chat = _fan_out_chat
    aetherdialect._llm_provider.LLMProvider.json = _fan_out_json
    for mod_name in LLM_PATCH_MODULES:
        mod = __import__(mod_name, fromlist=["LLMProvider"])
        if hasattr(mod, "LLMProvider"):
            mod.LLMProvider.chat = _fan_out_chat
            mod.LLMProvider.json = _fan_out_json

    try:
        with _staging_bundle_env():
            for slot in slots:
                if slot.kind != "question":
                    continue
                targets = fan_out_surfaces_for_slot(
                    slot,
                    construction_for_slot=_construction_for_slot,
                    recipe_for_slot=_recipe_for_slot,
                    federation_ineligible=FEDERATION_INELIGIBLE_QUESTIONS,
                )
                if not targets:
                    continue
                corpus.start_slot()
                for surface in targets:
                    construction = surface.construction
                    handle, artifacts_dir = pool._ephemeral_handle(
                        construction=construction,
                        fixtures_file=str(corpus.path),
                        provider="mock",
                    )
                    try:
                        if construction.apply_structure:
                            _stage_demo_schema_structure(handle)
                        if construction.surface == "federation":
                            with Sandbox.federation_scenario_session(
                                handle.engine,
                                slot.label,
                                mode=construction.mode,
                            ) as session:
                                session.accept_until_done(slot.label)
                        else:
                            session_cm = (
                                handle.engine.session(mode=construction.mode)
                                if construction.mode
                                else handle.engine.session()
                            )
                            with session_cm as session:
                                session.accept_until_done(slot.label)
                    finally:
                        handle.close()
                        aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
                        shutil.rmtree(artifacts_dir, ignore_errors=True)
                corpus.commit_slot()
    finally:
        aetherdialect._llm_provider.LLMProvider.chat = orig_chat
        aetherdialect._llm_provider.LLMProvider.json = orig_json
        for mod_name in LLM_PATCH_MODULES:
            mod = __import__(mod_name, fromlist=["LLMProvider"])
            if hasattr(mod, "LLMProvider"):
                mod.LLMProvider.chat = orig_chat
                mod.LLMProvider.json = orig_json

    return len(corpus.fixtures) - before


def _finalize_fixture_corpus_repair(corpus: FixtureCorpus) -> None:
    normalized = normalize_fixture_corpus_schema_domains(corpus)
    if normalized:
        corpus_message(f"[repair] normalized schema_domain on {normalized} intent fixture(s)")
    synced = sync_gatekeeper_normalization_fixture_questions(corpus)
    if synced:
        corpus_message(
            f"[repair] synced gatekeeper corrected text on {synced} normalization/intent fixture(s)",
        )
    federation_literals = repair_federation_intent_schema_literals(corpus)
    if federation_literals:
        corpus_message(
            f"[repair] backfilled schema_literal_json on {federation_literals} federation intent fixture(s)",
        )
    rekeyed = recanonicalize_mock_fixture_user_keys(corpus)
    if rekeyed:
        corpus_message(f"[repair] re-keyed {rekeyed} mock fixture user payload(s)")


def _post_record_corpus_hygiene(corpus: FixtureCorpus, questions: dict[str, list[str]]) -> int:
    """Drop superseded intent rows after committed slots are finalized and verified."""
    removed = prune_orphan_intent_fixtures(corpus, questions, committed_only=True)
    if removed:
        corpus_message(f"[build] pruned {removed} orphan intent fixture(s) from corpus")
    return removed


def _replay_sandbox_factory(pool: WarmRecordingPool) -> Callable[..., Any]:
    """Build sandboxes for replay the same way live capture builds them.

    Live capture seeds the preset baseline artifacts and reuses the warm connection and
    staging bundle. Rebuilding the engine any other way re-derives the schema graph, which
    shifts the schema context embedded in every intent prompt and misses recorded fixtures.
    """

    @contextmanager
    def factory(
        *,
        role: SchemaRole = SchemaRole.OWNER,
        engine_context: EngineContext | None = None,
        include: Literal["tables", "views"] = "tables",
        federation: bool = False,
    ) -> Any:
        if federation:
            construction = SlotConstruction(surface="federation", include=include)
        elif role == SchemaRole.CONSUMER:
            member = "catalog"
            if engine_context is not None and engine_context.allow_objects:
                for name, tables in SANDBOX_MEMBER_SPACE_TABLES.items():
                    if frozenset(engine_context.allow_objects) == tables:
                        member = name
                        break
            construction = SlotConstruction(
                surface="single",
                role="consumer",
                member=member,
                include=include,
            )
        else:
            construction = SlotConstruction(surface="single", role="owner", include=include)
        handle, artifacts_dir = pool._ephemeral_handle(
            construction=construction,
            fixtures_file=str(FIXTURES_PATH),
            provider="mock",
        )
        try:
            yield handle
        finally:
            handle.close()
            aetherdialect._sandbox.Sandbox._unlink_artifact_lock_files(artifacts_dir)
            shutil.rmtree(artifacts_dir, ignore_errors=True)

    return factory


def _verify_slot_with_validate(
    slot: RecordingSlot,
    *,
    pool: WarmRecordingPool | None = None,
) -> tuple[bool, str]:
    """Run the same offline validation predicate used at pack time against staging fixtures."""
    from aetherdialect._sandbox import Sandbox
    from aetherdialect.aetherdialect import AetherEngine

    if slot.kind == "feedback":
        return True, ""

    sandbox_factory = _replay_sandbox_factory(pool) if pool is not None else None
    with _staging_bundle_env():
        for (
            construction,
            apply_structure,
            _profile,
            target_slot_id,
            tier,
        ) in _mock_verify_targets_for_slot(slot):
            if construction.surface == "federation" or tier == "federation":
                row = Sandbox._validate_federation_slot(
                    AetherEngine,
                    slot.label,
                    slot_id=target_slot_id,
                    sandbox_factory=sandbox_factory,
                )
            else:
                role = SchemaRole.CONSUMER if construction.role == "consumer" else SchemaRole.OWNER
                row = Sandbox._validate_question_slot(
                    AetherEngine,
                    slot.label,
                    tier=tier,
                    role=role,
                    engine_context=_engine_context_for_construction(construction),
                    mode=construction.mode,
                    apply_structure=apply_structure,
                    slot_id=target_slot_id,
                    sandbox_factory=sandbox_factory,
                    include=construction.include,
                )
            if row is not None:
                detail = str(row.get("detail", "") or "expectation not met")
                return False, f"validate ({tier}): {detail}"
    return True, ""


def repair_failing_slots(*, force: bool = False) -> bool:
    if not STAGING.is_dir():
        raise SystemExit(f"Missing staging directory: {STAGING}")
    if not staging_fingerprint_matches():
        raise SystemExit(
            "Staging fingerprint mismatch or missing; run full scripts/sandbox_corpus.py before --repair.",
        )
    ensure_schema_literals()
    ensure_interpret_domain()
    pin_staging_mock_fixture_keys()
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
    if force:
        slots_to_repair = list(all_slots)
        corpus_message(f"[repair] force re-recording all {len(slots_to_repair)} slot(s)...")
    else:
        slots_to_repair = [slot for slot in all_slots if slot_id_for(slot) not in committed]
    corpus = FixtureCorpus(FIXTURES_PATH)
    missing_para = _missing_paraphrase_canonicals(questions, manifest)
    need_paraphrases = bool(missing_para)
    need_reuse = not _reuse_fixtures_ready(corpus)
    need_aetherspace = not _aetherspace_snapshots_ready()
    need_migration = not _migration_fixtures_ready(corpus)
    if not slots_to_repair and not need_paraphrases and not need_reuse and not need_aetherspace and not need_migration:
        print("All recording slots committed in manifest.", flush=True)
        _post_record_corpus_hygiene(corpus, questions)
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
                paraphrase_seeds = _paraphrase_seeds_for_missing(
                    session,
                    missing_para,
                    session.paraphrase_seeds_collected,
                )
            finalize_recording_tail(
                session,
                pool,
                corpus,
                paraphrase_seeds=paraphrase_seeds,
                capture_paraphrases=need_paraphrases,
                require_paraphrase_seeds=need_paraphrases,
                capture_reuse=need_reuse and not need_paraphrases,
                capture_aetherspace=need_aetherspace,
                capture_migration=need_migration,
            )
        finally:
            session.close()
            pool.close()
            teardown_recording_environment(env)
    manifest = _load_recording_manifest()
    _post_record_corpus_hygiene(corpus, questions)
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
        return Sandbox.validate_sandbox_corpus(AetherEngine, smoke=smoke)
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
        return Sandbox.validate_sandbox_corpus(AetherEngine)
    finally:
        if prev is None:
            os.environ.pop("AETHERDIALECT_SANDBOX_DATA_ZIP", None)
        else:
            os.environ["AETHERDIALECT_SANDBOX_DATA_ZIP"] = prev


def _practice_questions_from_validate_failures(failures: list[dict[str, str]]) -> set[str]:
    """Return owner practice questions that failed pack-time validate and need re-record."""
    out: set[str] = set()
    for row in failures:
        tier = str(row.get("tier", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if tier not in {"questions", "consumer_reader"}:
            continue
        if kind not in {"question", "faithfulness"}:
            continue
        name = str(row.get("name", "")).strip()
        if name:
            out.add(name)
    return out


def _federation_questions_from_validate_failures(failures: list[dict[str, str]]) -> set[str]:
    """Return federation questions that failed pack-time validate and need re-record."""
    out: set[str] = set()
    for row in failures:
        tier = str(row.get("tier", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if tier != "federation":
            continue
        if kind not in {"question", "faithfulness"}:
            continue
        name = str(row.get("name", "")).strip()
        if name:
            out.add(name)
    return out


def _uncommit_practice_question_slots(questions: set[str]) -> int:
    """Mark owner practice question slots uncommitted so repair will re-record them."""
    if not questions:
        return 0
    rows = _load_manifest_rows()
    touched = 0
    for slot in iter_recording_slots(load_staging_questions()):
        if slot.kind != "question" or slot.tier != "questions":
            continue
        if slot.label not in questions:
            continue
        sid = slot_id_for(slot)
        for row in rows:
            if str(row.get("slot_id", "")) != sid:
                continue
            if row.get("committed"):
                row["committed"] = False
                touched += 1
    if touched:
        write_recording_manifest(rows)
    return touched


def _uncommit_federation_question_slots(questions: set[str]) -> int:
    """Mark federation question slots uncommitted so repair will re-record them."""
    if not questions:
        return 0
    rows = _load_manifest_rows()
    touched = 0
    for slot in iter_recording_slots(load_staging_questions()):
        if slot.kind != "question" or slot.tier != "federation":
            continue
        if slot.label not in questions:
            continue
        sid = slot_id_for(slot)
        for row in rows:
            if str(row.get("slot_id", "")) != sid:
                continue
            if row.get("committed"):
                row["committed"] = False
                touched += 1
    if touched:
        write_recording_manifest(rows)
    return touched


def _sensitive_qualified_names(schema_graph: Any) -> tuple[str, ...]:
    from aetherdialect._contracts_base import SensitivityClassification

    names: list[str] = []
    for table_name, table in sorted(schema_graph.tables.items()):
        for column_name, column in sorted(table.columns.items()):
            sensitivity = getattr(column, "sensitivity", None)
            if sensitivity in (SensitivityClassification.HIDDEN, SensitivityClassification.RESTRICTED):
                names.append(f"{table_name}.{column_name}")
    return tuple(names)


def _domain_knowledge_sensitive_references(text: str, schema_graph: Any) -> list[str]:
    from aetherdialect._contracts_base import DomainKnowledgeEntry

    hidden = DomainKnowledgeEntry.hidden_column_references(text, schema_graph)
    sensitive = _sensitive_qualified_names(schema_graph)
    restricted_only = [
        name for name in sensitive if name not in DomainKnowledgeEntry.hidden_qualified_names(schema_graph)
    ]
    extra = [
        qualified for qualified in restricted_only if DomainKnowledgeEntry._qualified_token_appears(text, qualified)
    ]
    return sorted(set(hidden + extra))


def assert_staging_notes_parity(*, staging_dir: Path | None = None) -> None:
    """Assert member notes match space notes and federation composite notes match full notes."""
    from aetherdialect._constants_runtime import SANDBOX_BUNDLED_MEMBER_NOTES, SANDBOX_MEMBER_SPACE_NOTES_FILES

    target = staging_dir or STAGING
    main_notes_path = target / "rental_shop_notes.txt"
    if not main_notes_path.is_file():
        raise AssertionError(f"missing main notes file: {main_notes_path}")
    main_notes = main_notes_path.read_text(encoding="utf-8")
    for member, notes_name in SANDBOX_BUNDLED_MEMBER_NOTES:
        member_path = target / notes_name
        if not member_path.is_file():
            raise AssertionError(f"missing federation member notes for {member!r}: {member_path}")
        space_notes_name = SANDBOX_MEMBER_SPACE_NOTES_FILES.get(member)
        if space_notes_name is None:
            raise AssertionError(f"no sandbox space notes mapping for federation member {member!r}")
        if space_notes_name != notes_name:
            raise AssertionError(
                f"member notes filename {notes_name!r} must match space notes {space_notes_name!r} for {member!r}",
            )
        space_path = target / space_notes_name
        if not space_path.is_file():
            raise AssertionError(f"missing space notes file for {member!r}: {space_path}")
        member_notes = member_path.read_text(encoding="utf-8")
        if member_notes != space_path.read_text(encoding="utf-8"):
            raise AssertionError(f"member notes differ from space notes for {member!r}")
    fed_baseline = target / "artifacts_baseline" / BASELINE_FEDERATION_SUBDIR / "schema_graph.json.gz"
    if fed_baseline.is_file():
        from aetherdialect._schema_graph import load_schema_graph_snapshot

        fed_graph = load_schema_graph_snapshot(str(fed_baseline))
        if fed_graph is not None and fed_graph.notes_sha256:
            import hashlib

            main_hash = hashlib.sha256(main_notes.encode("utf-8")).hexdigest()
            if fed_graph.notes_sha256 != main_hash:
                raise AssertionError(
                    "federation composite notes hash must match rental_shop_notes.txt "
                    f"(expected {main_hash}, got {fed_graph.notes_sha256})",
                )


def assert_staging_domain_knowledge_no_sensitive_columns(*, staging_dir: Path | None = None) -> None:
    """Assert shipped domain knowledge names no HIDDEN or RESTRICTED columns."""
    from aetherdialect._constants import DOMAIN_KNOWLEDGE_FILENAME
    from aetherdialect._contracts_base import DomainKnowledgeEntry
    from aetherdialect._schema_graph import load_schema_graph_snapshot

    target = staging_dir or STAGING
    baseline_root = target / "artifacts_baseline"
    if not baseline_root.is_dir():
        raise AssertionError(f"missing artifacts baseline under {baseline_root}")
    violations: list[str] = []
    scan_dirs = [baseline_root / BASELINE_OWNER_SUBDIR, baseline_root / BASELINE_FEDERATION_SUBDIR]
    for member in ("storefront", "catalog", "logistics", "crm"):
        scan_dirs.append(baseline_root / BASELINE_FEDERATION_SUBDIR / member)
    for artifact_dir in scan_dirs:
        schema_path = artifact_dir / "schema_graph.json.gz"
        dk_path = artifact_dir / DOMAIN_KNOWLEDGE_FILENAME
        if not schema_path.is_file() or not dk_path.is_file():
            continue
        schema_graph = load_schema_graph_snapshot(str(schema_path))
        if schema_graph is None:
            continue
        payload = json.loads(dk_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            continue
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            try:
                entry = DomainKnowledgeEntry.normalize(
                    DomainKnowledgeEntry(
                        key=str(raw.get("key", "")),
                        kind=str(raw.get("kind", "")),
                        text=str(raw.get("text", "")),
                    ),
                )
            except Exception:
                continue
            refs = _domain_knowledge_sensitive_references(entry.text, schema_graph)
            if refs:
                violations.append(f"{artifact_dir.name}: DK key {entry.key!r} references sensitive columns {refs!r}")
    if violations:
        raise AssertionError("shipped domain knowledge references sensitive columns:\n" + "\n".join(violations))


def run_staging_pack_assertions(*, staging_dir: Path | None = None) -> None:
    """Run pack-time parity and DK safety checks without creating data.zip."""
    assert_staging_notes_parity(staging_dir=staging_dir)
    assert_staging_domain_knowledge_no_sensitive_columns(staging_dir=staging_dir)


def finalize_validate(*, smoke: bool = False, force: bool = False) -> None:
    """Repair/validate loop until staging passes the corpus gate (no zip pack)."""
    prefix = "[smoke] " if smoke else "[build] "
    for pass_num in range(1, RECORDING_MAX_VALIDATE_PASSES + 1):
        if pass_num == 1 and force:
            repair_failing_slots(force=True)
        else:
            repair_failing_slots(force=False)
        ready, reasons = _recording_pipeline_ready()
        if not ready:
            for reason in reasons:
                corpus_message(f"{prefix}recording incomplete: {reason}")
            raise SystemExit(1)
        corpus = FixtureCorpus(FIXTURES_PATH)
        run_fixture_fan_out(corpus)
        failures = validate_staging_dir(smoke=smoke)
        if not failures:
            run_staging_pack_assertions()
            msg = "Sandbox corpus OK (all questions, recipes, feedback flows)"
            print(msg)
            VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
            VALIDATE_OUT.write_text(msg + "\n", encoding="utf-8")
            return
        questions = _practice_questions_from_validate_failures(failures)
        federation_questions = _federation_questions_from_validate_failures(failures)
        if not questions and not federation_questions:
            report_lines = [
                f"FAIL [{row['kind']}] {row.get('tier', '')} {row['name'][:70]}: {row['detail']}" for row in failures
            ]
            VALIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
            VALIDATE_OUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            for line in report_lines:
                print(line)
            raise SystemExit(f"{len(failures)} sandbox validation failures")
        if questions:
            corpus_message(
                f"{prefix}validate pass {pass_num}/{RECORDING_MAX_VALIDATE_PASSES}: "
                f"re-record {len(questions)} practice question(s) after expectation mismatch",
            )
            _uncommit_practice_question_slots(questions)
        if federation_questions:
            corpus_message(
                f"{prefix}validate pass {pass_num}/{RECORDING_MAX_VALIDATE_PASSES}: "
                f"re-record {len(federation_questions)} federation question(s) after expectation mismatch",
            )
            _uncommit_federation_question_slots(federation_questions)
    raise SystemExit(
        f"sandbox validate did not pass after {RECORDING_MAX_VALIDATE_PASSES} repair passes",
    )


def pack_and_promote(*, smoke: bool = False) -> None:
    from aetherdialect.aetherdialect import AetherEngine

    ensure_schema_literals()
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
        issues = Sandbox._sandbox_doctor_verbose()
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
        Sandbox.assert_sandbox_complete(AetherEngine)
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
    corpus_message(f"{prefix}assembling staging (sqlite subset + CSV/DDL)...")
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
        corpus_message(f"{prefix}recording incomplete; running repair/validate passes...")
    corpus_message(f"{prefix}validating sandbox corpus...")
    try:
        finalize_validate(smoke=smoke)
    except SystemExit:
        raise SystemExit(1) from None
    corpus_message(f"{prefix}done.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the sandbox corpus from scratch, or repair failing slots and repack.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-record failing slots under scripts/sandbox_staging, then validate and pack",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --repair, re-record every recording slot (not only uncommitted ones)",
    )
    parser.add_argument(
        "--record-reuse-pairs",
        action="store_true",
        help="Record paraphrase pairs to test intent reuse within the same session.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the full build pipeline with one practice question plus all validation, "
            "feedback, and scenario slots; writes sandbox_staging.zip only (not data.zip). "
            "Paraphrases are still LLM-generated from the recorded practice slot."
        ),
    )
    args = parser.parse_args()
    if args.force and not args.repair:
        parser.error("--force only applies with --repair")
    if args.repair and (args.smoke or args.record_reuse_pairs):
        parser.error("--repair cannot be combined with --smoke or --record-reuse-pairs")
    if args.repair:
        if args.force:
            corpus_message("[repair] force re-recording all committed slots...")
        else:
            corpus_message("[repair] re-recording uncommitted slots...")
        corpus_message("[repair] validating sandbox corpus...")
        try:
            finalize_validate(force=args.force)
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                raise SystemExit(code) from None
            if code:
                raise SystemExit(code) from None
            raise SystemExit(1) from None
        corpus_message("[repair] done.")
        return
    _run_full_build(record_reuse_pairs=args.record_reuse_pairs, smoke=args.smoke)


if __name__ == "__main__":
    main()
