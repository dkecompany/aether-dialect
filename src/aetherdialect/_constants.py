"""Static package data: allow-lists, UI strings, and short error templates."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from platformdirs import user_data_dir
from sqlglot import exp


def env_first_nonempty(env: Mapping[str, str], *keys: str) -> str:
    """Return the first non-blank value among *keys*, else an empty string."""
    for key in keys:
        value = str(env.get(key, "") or "").strip()
        if value:
            return value
    return ""


def env_any_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    """Return True when at least one key maps to a non-blank string."""
    return any(str(env.get(key, "") or "").strip() for key in keys)


def env_role_hint(label: str, keys: tuple[str, ...]) -> str:
    """Return a human-readable hint listing acceptable environment variable names."""
    return f"{label}: {' or '.join(keys)}"


def package_importable(name: str) -> bool:
    """Return True when *name* can be imported as a top-level module."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


class GenerationPath(str, Enum):
    """Canonical SQL generation path codes (1, 2.1, 2.2, 3, 4.1, 4.2, 4.3, 5)."""

    EXACT_QUESTION_REUSE = "1"
    FUZZY_REUSE_LITERAL_STRUCTURAL = "2.1"
    FUZZY_REUSE_FULL_PARAMS = "2.2"
    INTENT_DIRECT_MATCH = "3"
    UNION_TEMPLATE_WIDEN = "4.1"
    UNION_TEMPLATE_AND_RUNTIME_WIDEN = "4.2"
    RUNTIME_SUBSET_TEMPLATE_WIDE = "4.3"
    FRESH = "5"

    @property
    def code(self) -> str:
        """Return the path code string."""
        return str(self.value)

    @property
    def label(self) -> str:
        """Return a stable readable path label."""
        if self is GenerationPath.EXACT_QUESTION_REUSE:
            return "exact_question_reuse"
        if self is GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL:
            return "fuzzy_reuse_literal_structural"
        if self is GenerationPath.FUZZY_REUSE_FULL_PARAMS:
            return "fuzzy_reuse_full_params"
        if self is GenerationPath.INTENT_DIRECT_MATCH:
            return "intent_direct_match"
        if self is GenerationPath.UNION_TEMPLATE_WIDEN:
            return "union_template_widen"
        if self is GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN:
            return "union_template_and_runtime_widen"
        if self is GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE:
            return "runtime_subset_template_wide"
        return "fresh"

    @classmethod
    def parse(
        cls,
        value: str | GenerationPath,
    ) -> GenerationPath:
        """Parse enum from code string or enum value. Legacy persisted ``"2"`` maps to :attr:`FUZZY_REUSE_FULL_PARAMS`. Legacy ``"4"`` maps to :attr:`UNION_TEMPLATE_AND_RUNTIME_WIDEN` when disambiguation metadata is absent."""
        if isinstance(value, GenerationPath):
            return value
        s = str(value).strip()
        if s == "2":
            return cls.FUZZY_REUSE_FULL_PARAMS
        if s == "4":
            return cls.UNION_TEMPLATE_AND_RUNTIME_WIDEN
        return cls(s)


ResultReaderKind = Literal[
    "sqlalchemy",
    "spark",
    "connector",
    "bq_client",
    "bq_storage",
    "snowflake_arrow",
]

_ARTIFACT_USER_PARENT: str = user_data_dir(appname="aetherdialect", appauthor=False)
ENGINE_STORAGE_PLACEHOLDER_DIR: str = os.path.join(
    _ARTIFACT_USER_PARENT,
    "aetherdialect",
    "__placeholder__",
)

SESSION_PROMPT_YESNO: str = "Is this correct? (y/n): "
SESSION_PROMPT_REASON: str = "Please provide a reason: "

MIGRATION_HEADER_BY_TIER: dict[str, str] = {
    "soft_refresh": "Refreshing cached metadata. Existing learning is kept.",
    "remap": "Schema renames detected. Mapping existing learning to the new names.",
    "destructive": "Learning reset: cache rebuilt from scratch (schema changed in ways that cannot be remapped).",
}

SAVED_LINE: str = "Saved."

FEEDBACK_NOTED_LINE: str = "Feedback noted. Try rephrasing your question for a better match."

QUERY_RESULTS_HEADER: str = "Query Results"

USER_ERROR_PREFIX: str = "Error: "
USER_WARN_PREFIX: str = "! "
USER_TERMINATED_LINE: str = "\nUser terminated."
USER_INVALID_INPUT_LINE: str = "\nInvalid input."

REPHRASE_HINT_MESSAGES: dict[str, str] = {
    "intent_parse_failed": (
        "Please rephrase your question.\n\n"
        "Tips: mention specific tables or columns, keep filters simple, and avoid ambiguous references.\n"
    ),
    "schema_invalid_declined": (
        "Please rephrase your question.\n"
        "Tips: use tables and columns that exist in this database, or ask about a related concept."
    ),
    "sql_validation_failed": (
        "Please rephrase or retry.\n\n"
        "Tips: simplify filters, be explicit about columns, or split a complex question into smaller ones.\n"
    ),
    "user_rejected_intent": (
        "Please rephrase your question.\n"
        "Tips: be more specific about which columns, filters, grouping, or time range you want."
    ),
    "user_rejected_result": (
        "Please retry or rephrase your question.\n\n"
        "Tips: be more specific about columns, filters, grouping, or time range.\n"
    ),
    "restricted_question": (
        "This question references columns or tables outside the visible schema. Try rephrasing using only "
        "the table and column names you see in the schema notes, or update `deny_columns` / `allow_columns` "
        "if you want them in scope."
    ),
    "vague_question": (
        "I could not pin this question to specific tables or columns.\n\n"
        "Try naming the entity (a table or business object), the metric you want, and any filter (date range, "
        "status, region) so I have something concrete to map.\n"
    ),
}

USER_REJECTED_RESULT_BUCKET_TIPS: dict[str, str] = {
    "MISSING_FILTER": "Tips: name the filter or dimension you care about (time range, status, category).",
    "WRONG_GROUPING": "Tips: say whether you want totals per entity, per period, or overall.",
    "WRONG_AGGREGATION": "Tips: specify sum, average, count, or another metric clearly.",
    "WRONG_TIME_RANGE": "Tips: give an explicit date range or relative window.",
    "WRONG_TABLES_OR_JOINS": "Tips: name the tables or relationships that should connect your answer.",
    "WRONG_SORT_OR_LIMIT": "Tips: say how results should be ordered or how many rows you need.",
    "OTHER": "Tips: be more specific about columns, filters, grouping, or time range.",
}

JOIN_PRIOR_FEEDBACK_HEADING: str = "Previously rejected joins for this question (avoid these table sets / FK paths):"

NORMALIZATION_ALLOWED_INTRODUCED_TOKENS: frozenset[str] = frozenset(
    {"list", "count", "sum", "average", "max", "min", "total", "of"},
)

INSTRUCTIONAL_TABLE_PLACEHOLDER: str = "table"
INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER: str = "other_table"
INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER: str = "table.column"
INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER: str = "other_table.other_column"

INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR: str = (
    "Instructional placeholder tokens appear in expression strings. Replace each with exact table.column "
    "names from schema_info. Do not leave angle-bracket markup, table_N or column_N instructional tokens, "
    "or synthetic shape tokens from the prompt (table, other_table, table.column, other_table.other_column)."
)

NATURAL_LANGUAGE_REFUSAL_PARSE_ERROR: str = (
    "natural_language contains refusal or permission prose while select_cols remain populated"
)

ARTIFACT_DIRECTORY_SEGMENT: str = "aetherdialect"

ENGINE_STORAGE_SLUG_MAX_CHARS: int = 180

AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV: str = "AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL"

SUPPORTED_ENGINES: frozenset[str] = frozenset()


def set_supported_engines(engines: frozenset[str]) -> None:
    """Update the runtime registry of dialect names after ``register_dialect`` calls."""
    global SUPPORTED_ENGINES
    SUPPORTED_ENGINES = engines


ARTIFACT_LOCK_TIMEOUT_SECONDS: float = 30.0
ARTIFACT_LOCK_POLL_INTERVAL_SECONDS: float = 0.05

NORMALIZATION_JACCARD_FLOOR: float = 0.4

TRUST_FLOOR: int = 1
TRUST_CEILING: int = 2
TRUST_AUTO_ACCEPT_THRESHOLD: int = 1

WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE: float = 0.25

WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS: float = 30.0
WRITE_QUEUE_MAX_BYTES_PER_DRAIN: int = 4194304

SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS: int = 2000

MAX_NON_AGG_COL_DIFF = 2

RUNTIME_PARAPHRASE_COUNT: int = 4

BQ_DEFAULT_PARTITION_LOOKBACK_DAYS: int = 90

SEED_NORMALIZATION_BATCH_SIZE: int = 20

MIGRATION_DATA_OVERLAP_MIN: float = 0.15
MIGRATION_TABLE_RENAME_COLUMN_FRACTION: float = 0.60

WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY: int = 16

WARMUP_ROUND_TRIP_LIMIT: int = 100

WARMUP_PARAPHRASE_COUNT_FROM_SQL: int = 5

EMPTY_JOIN_CANDIDATES: dict[str, Any] = {"candidates": []}

SCHEMA_GRAPH_ID_PREFIX: str = "sg_"
SCHEMA_GRAPH_ID_DETERMINISTIC_SEED_V1: str = "aetherdialect-sg-v1|"

PERMISSION_DENIED_USER_MESSAGE: str = (
    "You do not have access to one or more tables required by this answer. Please contact your administrator."
)

JSON_COMPACT_SEPARATORS: tuple[str, str] = (",", ":")

SCHEMA_FIELD_DESCRIPTION: str = "description"
SCHEMA_FIELD_ROLE: str = "role"
SCHEMA_FIELD_TYPE: str = "type"
SCHEMA_FIELD_TRUTH_VALUE: str = "truth_value"
SCHEMA_FIELD_KEYS: str = "keys"
SCHEMA_FIELD_ENUM: str = "enum"

INTERPRET_FIELDS: frozenset[str] = frozenset({SCHEMA_FIELD_DESCRIPTION, SCHEMA_FIELD_ENUM})
GROUND_FIELDS: frozenset[str] = frozenset(
    {SCHEMA_FIELD_DESCRIPTION, SCHEMA_FIELD_ROLE, SCHEMA_FIELD_TYPE, SCHEMA_FIELD_TRUTH_VALUE, SCHEMA_FIELD_ENUM}
)
COMPOSE_FIELDS: frozenset[str] = frozenset(
    {SCHEMA_FIELD_ROLE, SCHEMA_FIELD_TYPE, SCHEMA_FIELD_TRUTH_VALUE, SCHEMA_FIELD_KEYS, SCHEMA_FIELD_ENUM}
)
FULL_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_DESCRIPTION,
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_KEYS,
        SCHEMA_FIELD_ENUM,
    }
)

INTERPRET_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["approach", "tables"],
    "additionalProperties": False,
    "properties": {
        "approach": {"type": "string", "minLength": 1},
        "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "grounding": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ref", "used_for"],
                "additionalProperties": False,
                "properties": {"ref": {"type": "string"}, "used_for": {"type": "string"}},
            },
        },
        "schema_invalid": {"type": "boolean"},
        "missing": {"type": "string"},
    },
}

INTERPRET_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Identify the semantic entities, measures, conditions, grouping, ordering, ranking, row cap, conditional labeling, and time reasoning needed to answer the question.",
    "Reformulate unsupported full-SQL constructs into supported analysis shapes in plain language without naming IR or SQL operators.",
    "Infer whether the question needs row-level output, grouped output, a scalar answer, staged intermediate computation, a windowed comparison, or a conditional bucketed result.",
    "Use only the domain schema descriptions and enum heads to ground business concepts; capture any missing or ambiguous binding as internal planning uncertainty rather than refusing.",
    "Record grounding traceability for tables, enum values, and filter, having, or group_by constraints only; do not enumerate select output columns in grounding.",
)

GROUND_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Bind the interpret pathway to real schema identifiers and natural-language clause descriptions without emitting runtime IR.",
    "Populate select, filter, group_by, having, order_by, limit, window, and case prose fields; copy every literal into the matching prose field.",
    "List semantic base tables in tables; omit junction_table from tables when its columns appear in prose.",
    "Keep a table in join scope only via qualified table.column tokens in select, filter, group_by, having, order_by, or registry prose — not from the tables list alone.",
    "When membership or existence requires link_table or junction_table, name junction_table.column or bridge_table.column in select prose, not only join-equality narration in filter prose.",
    "Use cte_steps when staged computation is needed; each step tables list may name base schema tables and prior cte_steps names this step reads from.",
    "Never author join paths; the engine discovers foreign-key paths after structural encoding.",
)

COMPOSE_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Return columns or computed expressions; optionally return only distinct rows.",
    "Aggregate with count, sum, avg, min, or max; sum and avg require numeric columns, count applies to any column.",
    "Group by one or more columns (grouped grain), or compute a single aggregate over all rows (scalar grain); a query with no aggregation is row-level.",
    "Filter rows with =, !=, <, <=, >, >=, like, not like, ilike, not ilike, in, not in, between, is null, is not null, and array contains.",
    "Combine filters as AND within a group and OR across groups.",
    "Restrict aggregated results with a HAVING comparison using =, !=, <, <=, >, >=, in, not in, or between.",
    "Order by any column or expression ascending or descending, and cap rows with a limit.",
    "Compute arithmetic with +, -, *, / and wrap arithmetic in an aggregate.",
    "Apply scalar functions upper, lower, trim, ltrim, rtrim, length, abs, round, floor, ceil, date_trunc, date_part, extract (never epoch), coalesce, concat, year, month, day.",
    "Concatenate text with concat, never the || operator.",
    "Rank or number rows per group and compute running or offset values with window functions row_number, rank, dense_rank, sum, avg, lag, lead, first_value, last_value, using partitioning, ordering, and row or range frames.",
    "Produce conditional labels or buckets with CASE.",
    "Filter on a relative time window (last N days, weeks, months, quarters, or years) or on the difference between two date columns.",
    "Compare or shift a date by an integer number of days, and reference the current date or current timestamp.",
    "Break a computation into intermediate steps (WITH steps) for staged aggregation, self-comparison, per-entity ranking, or reuse; a later step reads an earlier step's named outputs.",
    "Not expressible - reformulate instead: set difference or EXCEPT, UNION, INTERSECT, EXISTS or NOT EXISTS, anti-joins, correlated or scalar subqueries beyond the intermediate-step shapes, and LATERAL. Express absence as a left-joined source filtered by is null; express existence as an inner join returning distinct rows; express per-entity top-one as row_number partitioned by the entity filtered to one; compare a row to an aggregate via an extra intermediate step or a window function.",
    "Join paths are discovered by the engine from foreign keys; never author joins or name junction tables except when the many-to-many set itself is requested; list only the base tables whose columns are used.",
)

IR_SUPPORTED_CAPABILITIES: tuple[str, ...] = COMPOSE_SUPPORTED_CAPABILITIES

ASK_PHASE_A: str = "A:intake.reuse"
ASK_PHASE_B: str = "B:intent.interpret"
ASK_PHASE_C: str = "C:intent.ground"
ASK_PHASE_D: str = "D:intent.compose"
ASK_PHASE_E: str = "E:intent.repair_structural"
ASK_PHASE_F: str = "F:intent.validate_schema"
ASK_PHASE_G: str = "G:intent.validate_semantic"
ASK_PHASE_H: str = "H:intent.finalize"
ASK_PHASE_I: str = "I:sql.build_joins"
ASK_PHASE_J: str = "J:sql.validate_scope"
ASK_PHASE_K: str = "K:sql.execute"
ASK_PHASE_L: str = "L:feedback"

SCHEMA_BUILD_PHASE_A: str = "A:cache_gate"
SCHEMA_BUILD_PHASE_B: str = "B:diff"
SCHEMA_BUILD_PHASE_C: str = "C:reflect"
SCHEMA_BUILD_PHASE_D: str = "D:scope_filter"
SCHEMA_BUILD_PHASE_E: str = "E:profile"
SCHEMA_BUILD_PHASE_F: str = "F:classify"
SCHEMA_BUILD_PHASE_G: str = "G:coerce_ops"
SCHEMA_BUILD_PHASE_H: str = "H:join_stats"
SCHEMA_BUILD_PHASE_I: str = "I:viability_hash"
SCHEMA_BUILD_PHASE_J: str = "J:persist"
SCHEMA_BUILD_PHASE_K: str = "K:overrides_health"

QSIM_PHASE_A: str = "A:bootstrap"
QSIM_PHASE_B: str = "B:skeletons"
QSIM_PHASE_C: str = "C:targets"
QSIM_PHASE_D: str = "D:select"
QSIM_PHASE_E: str = "E:fill"
QSIM_PHASE_F: str = "F:validate"
QSIM_PHASE_G: str = "G:normalize_dedup"
QSIM_PHASE_H: str = "H:instantiate"
QSIM_PHASE_I: str = "I:nl_synthesis"
QSIM_PHASE_J: str = "J:emit"

WARMUP_PHASE_A: str = "A:setup"
WARMUP_PHASE_B: str = "B:ingest"
WARMUP_PHASE_C: str = "C:expand"
WARMUP_PHASE_D: str = "D:dedupe_classify"
WARMUP_PHASE_E: str = "E:join_prep"
WARMUP_PHASE_F: str = "F:cache_open"
WARMUP_PHASE_G: str = "G:execute"
WARMUP_PHASE_H: str = "H:sample"
WARMUP_PHASE_I: str = "I:nl_generate"
WARMUP_PHASE_J: str = "J:template_learn"
WARMUP_PHASE_K: str = "K:persist"

DIAGNOSTIC_CODE_REUSE_HIT: str = "REUSE_HIT"
DIAGNOSTIC_CODE_REUSE_MISS: str = "REUSE_MISS"
DIAGNOSTIC_CODE_LOW_CONFIDENCE: str = "LOW_CONFIDENCE"
DIAGNOSTIC_CODE_LARGE_RESULT_WARNING: str = "LARGE_RESULT_WARNING"
DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT: str = "SENSITIVITY_GATE_HIT"
DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY: str = "INTERPRET_GROUND_RETRY"
DIAGNOSTIC_CODE_COMPOSE_REPAIR: str = "COMPOSE_REPAIR"
DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART: str = "FALLBACK_FRESH_RESTART"
DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED: str = "CONFIG_FILE_VALUE_APPLIED"
DIAGNOSTIC_CODE_ENGINE_INFO: str = "ENGINE_INFO"
DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP: str = "SCHEMA_OVERRIDE_SKIP"
DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT: str = "PK_INFERENCE_PROMPT"
DIAGNOSTIC_CODE_ZERO_ROW_FILTER_SUGGESTION: str = "ZERO_ROW_FILTER_SUGGESTION"
DIAGNOSTIC_CODE_ZERO_ROW_FILTER_AUTO_FIXED: str = "ZERO_ROW_FILTER_AUTO_FIXED"

AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD: str = "write_queue_feedback_record"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT: str = "write_queue_template_reject"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT: str = "write_queue_template_accept"
AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL: str = "write_queue_override_proposal"

CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC: str = "deny_columns and allow_columns entries must be qualified as 'table.column' or '*.column'; bare column names are not permitted; got {spec!r}"

INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_BY_SENSITIVITY_VALUE: MappingProxyType[str, str] = MappingProxyType(
    {
        "restricted": "{location}: column {table}.{column} cannot appear in {surface} (restricted classification).",
        "hidden": "{location}: column {table}.{column} cannot appear in {surface} (hidden classification).",
    }
)
INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_DEFAULT: str = (
    "{location}: column {table}.{column} cannot appear in {surface} (sensitivity policy)."
)

ARTIFACT_FORMAT_VERSION: int = 4
MIN_COMPATIBLE_PACKAGE_VERSION: str = "0.1.8"
ARTIFACT_MANIFEST_FILENAME: str = "artifact_manifest.json"
ARTIFACT_LOCK_FILENAME: str = ".aetherdialect_engine.lock"

SCHEMA_OVERRIDES_DEFAULT_FILENAME: str = "schema_overrides.json"
MIGRATION_MAP_FILENAME: str = "schema_migration_map.json"
WRITE_QUEUE_FILENAME: str = "write_queue.jsonl"

MIGRATION_MAP_ACTION_REMAP: str = "remap"
MIGRATION_MAP_ACTION_DESTRUCTIVE: str = "destructive"
MIGRATION_MAP_ACTION_ABORT: str = "abort"

ARTIFACT_LAST_ACTION_REMAP_USER_MAP: str = "remap_user_map"
ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP: str = "destructive_user_map"

SCHEMA_OVERRIDES_APPLIED_SUFFIX: str = ".applied.json"
SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"
SCHEMA_OVERRIDES_SIDECAR_FILENAME: str = "applied_overrides.json"
SCHEMA_OVERRIDES_VERSION: int = 1
SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER: str = "ai"

INTERACTIVE_STAGE_DIRECT_REUSE: str = "direct_reuse_confirm"
INTERACTIVE_STAGE_INTENT_CONFIRM: str = "intent_confirm"
INTERACTIVE_STAGE_SQL_FEEDBACK: str = "sql_result_confirm"

PIPELINE_SUSPEND_ID_DIRECT_REUSE: str = "awaiting_direct_reuse_confirmation"
PIPELINE_SUSPEND_ID_INTENT_CONFIRM: str = "awaiting_intent_confirmation"
PIPELINE_SUSPEND_ID_INTENT_FEEDBACK: str = "awaiting_intent_rejection_feedback"
PIPELINE_SUSPEND_ID_SQL: str = "awaiting_sql_result_confirmation"
PIPELINE_SUSPEND_ID_EXECUTE: str = "awaiting_execute_confirmation"
PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT: str = "awaiting_user_feedback_reject_reason"
PIPELINE_BUG_SQL_VALIDATION: str = "pipeline_bug_sql_validation"

SESSION_KIND_IDLE: str = "idle"
SESSION_KIND_AWAITING_INTENT_CONFIRM: str = "awaiting_intent_confirm"
SESSION_KIND_AWAITING_INTENT_FEEDBACK: str = "awaiting_intent_feedback"
SESSION_KIND_AWAITING_SQL_CONFIRM: str = "awaiting_sql_confirm"
SESSION_KIND_AWAITING_SQL_FEEDBACK: str = "awaiting_sql_feedback"
SESSION_KIND_EXECUTE: str = "execute"
SESSION_KIND_RESULT: str = "result"
SESSION_KIND_ERROR: str = "error"

YES_NO_SESSION_KINDS: frozenset[str] = frozenset(
    {
        SESSION_KIND_AWAITING_INTENT_CONFIRM,
        SESSION_KIND_AWAITING_SQL_CONFIRM,
        SESSION_KIND_EXECUTE,
    }
)

SUSPEND_ID_TO_SESSION_KIND: dict[str, str] = {
    PIPELINE_SUSPEND_ID_DIRECT_REUSE: SESSION_KIND_AWAITING_SQL_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM: SESSION_KIND_AWAITING_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK: SESSION_KIND_AWAITING_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_EXECUTE: SESSION_KIND_EXECUTE,
    PIPELINE_SUSPEND_ID_SQL: SESSION_KIND_AWAITING_SQL_CONFIRM,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT: SESSION_KIND_AWAITING_SQL_FEEDBACK,
}

SEED_NORMALIZATION_JSON: str = "seed_question_normalization.json"
NORMALIZED_SEEDS_TXT: str = "seed_questions_normalized.txt"
QSIM_QUESTIONS_PATTERN: str = "qsim_questions_v{version}.txt"

JOIN_CHOICE_SCOPE_MAIN: str = "main"

SCHEMA_CONTEXT_CACHE_NAME: str = "schema_context.json"
SCHEMA_CONTEXT_CACHED_DDL: str = "_cached_schema_context.sql"
SCHEMA_CONTEXT_CACHED_NOTES: str = "_cached_schema_context_notes.txt"
SCHEMA_CONTEXT_CACHE_VERSION: int = 3

AETHERSPACE_ARTIFACT_VERSION: int = 1
AETHERSPACES_SEGMENT: str = "aetherspaces"
MASTER_AETHERSPACE_NAME: str = "master"
CANONICAL_FEEDBACK_DIALECT: str = "duckdb"

TEMPLATE_STORE_SEGMENT: str = "intent_templates"
TEMPLATE_STORE_SPACES_SEGMENT: str = "spaces"
SCHEMA_CONTEXT_NAMED_SPEC_GLOB: str = "schema_context.*.json"
TEMPLATE_STORE_HEADER_FILENAME: str = "header.json.gz"
TEMPLATE_STORE_PARTITION_PREFIX: str = "partition_"
TEMPLATE_STORE_PARTITION_COUNT: int = 256
TEMPLATE_STORE_PARTITION_LRU_MAX: int = 32
TEMPLATE_STORE_LEGACY_SINGLE_FILE: str = "intent_templates.json.gz"

SEED_WARMUP_CACHE_ZIP: str = "seed_warmup_cache.zip"
WARMUP_ANCHOR_LATTICE_SUBDIR: str = "anchor_lattice"

LEGACY_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "schema_graph.json.gz",
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    "qsim_skeletons.json.gz",
    SEED_WARMUP_CACHE_ZIP,
)

LEGACY_ARTIFACT_GLOBS: tuple[str, ...] = (
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
)

SIMULATION_CACHE_EXACT_FILENAMES: tuple[str, ...] = (
    "qsim_skeletons.json.gz",
    "qsim_summary.json",
    SEED_WARMUP_CACHE_ZIP,
)

SIMULATION_CACHE_GLOB_PATTERNS: tuple[str, ...] = (
    "qsim_questions_v*.txt",
    "seed_warmup_report_v*.json",
    "seed_warmup_v*.zip",
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
    f"{WARMUP_ANCHOR_LATTICE_SUBDIR}/*",
)

SOFT_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        "explain_seq_scan_indexed",
        "explain_zero_estimate",
    }
)

ROLE_VALUE_TYPE_COMPAT: dict[str, frozenset[str]] = {
    "boolean": frozenset({"boolean", "integer", "string"}),
    "numeric_measure": frozenset({"integer", "number"}),
    "numeric_categorical": frozenset({"integer", "number"}),
    "temporal": frozenset({"date", "integer"}),
    "audit": frozenset({"date"}),
    "free_text": frozenset({"string"}),
    "categorical": frozenset({"string", "integer", "number", "boolean"}),
    "identifier": frozenset({"string", "integer", "number"}),
}

DURATION_COLUMN_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "duration",
        "period",
        "lead_time",
        "leadtime",
        "lag",
        "tenure",
        "offset",
        "days",
        "day_count",
        "hours",
        "hour_count",
    }
)

YEAR_LIKE_COLUMN_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "year",
        "release_year",
        "birth_year",
        "fiscal_year",
    }
)

CANONICAL_ENGINE_ORDER: tuple[str, ...] = (
    "sqlite",
    "duckdb",
    "csv",
    "mysql",
    "mariadb",
    "sqlserver",
    "postgresql",
    "redshift",
    "databricks",
    "snowflake",
    "bigquery",
)

RESULT_READER_KINDS: tuple[ResultReaderKind, ...] = (
    "sqlalchemy",
    "spark",
    "connector",
    "bq_client",
    "bq_storage",
    "snowflake_arrow",
)

_INTENT_DATE_UNIT_AMOUNT_VALUE: dict[str, Any] = {
    "type": "object",
    "required": ["unit", "amount"],
    "properties": {
        "unit": {"enum": ["day", "week", "month", "quarter", "year"]},
        "amount": {"type": "integer", "minimum": 1},
    },
}

_INTENT_FILTER_ITEM_ALLOF: list[dict[str, Any]] = [
    {
        "if": {
            "allOf": [
                {"required": ["value_type"]},
                {"properties": {"value_type": {"const": "date_diff"}}},
            ]
        },
        "then": {"properties": {"value": _INTENT_DATE_UNIT_AMOUNT_VALUE}},
    },
    {
        "if": {
            "allOf": [
                {"required": ["value_type"]},
                {"properties": {"value_type": {"const": "date_window"}}},
            ]
        },
        "then": {"properties": {"value": _INTENT_DATE_UNIT_AMOUNT_VALUE}},
    },
]

INTENT_SCHEMA = {
    "type": "object",
    "required": ["tables"],
    "properties": {
        "tables": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "string"},
            ]
        },
        "select_cols": {
            "type": "array",
            "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
        },
        "group_by_cols": {"type": "array", "items": {"type": "string"}},
        "order_by_cols": {
            "type": "array",
            "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
        },
        "filters_param": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "left_expr": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "left_col": {"type": "string"},
                    "op": {"type": "string"},
                    "right_expr": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "right_col": {"type": "string"},
                    "value_type": {"type": "string"},
                    "value": {},
                    "bool_op": {"type": "string"},
                    "filter_group": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                },
                "allOf": _INTENT_FILTER_ITEM_ALLOF,
            },
        },
        "having_param": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "left_expr": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "left_agg": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "op": {"type": "string"},
                    "right_expr": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "right_agg": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "value_type": {"type": "string"},
                    "value": {},
                    "bool_op": {"type": "string"},
                    "filter_group": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                },
                "allOf": _INTENT_FILTER_ITEM_ALLOF,
            },
        },
        "cte_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cte_name", "select_cols", "output_columns"],
                "properties": {
                    "cte_name": {"type": "string"},
                    "description": {"type": "string"},
                    "tables": {"type": "array", "items": {"type": "string"}},
                    "grain": {
                        "type": "string",
                        "enum": ["row_level", "grouped", "scalar"],
                    },
                    "select_cols": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "required": ["expr"],
                                    "properties": {
                                        "expr": {"type": "string"},
                                        "alias": {"type": "string"},
                                    },
                                },
                            ]
                        },
                    },
                    "group_by_cols": {"type": "array"},
                    "order_by_cols": {"type": "array"},
                    "filters_param": {"type": "array"},
                    "having_param": {"type": "array"},
                    "output_columns": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                    },
                    "window_registry": {"type": "array"},
                    "case_registry": {"type": "array"},
                },
            },
        },
        "window_registry": {"type": "array"},
        "case_registry": {"type": "array"},
        "limit": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "natural_language": {"type": "string"},
    },
}


def _build_logical_intent_schema() -> dict[str, Any]:
    """Return JSON Schema for planner :class:`LogicalIntent` LLM output."""
    prose: dict[str, Any] = {"type": "string"}
    limit_null: dict[str, Any] = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    cte_item: dict[str, Any] = {
        "type": "object",
        "required": ["name", "tables", "select"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "select": {"type": "string", "minLength": 1},
            "filter": prose,
            "group_by": prose,
            "having": prose,
            "order_by": prose,
            "limit": limit_null,
            "window": prose,
            "case": prose,
        },
    }
    return {
        "type": "object",
        "required": ["tables", "select"],
        "additionalProperties": False,
        "properties": {
            "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "select": {"type": "string", "minLength": 1},
            "filter": prose,
            "group_by": prose,
            "having": prose,
            "order_by": prose,
            "limit": limit_null,
            "window": prose,
            "case": prose,
            "cte_steps": {"type": "array", "items": cte_item},
        },
    }


LOGICAL_INTENT_SCHEMA: dict[str, Any] = _build_logical_intent_schema()

PLANNER_PROSE_FIELDS: tuple[str, ...] = (
    "select",
    "filter",
    "group_by",
    "having",
    "order_by",
    "window",
    "case",
)

_AGGREGATION_FUNCTION_NAMES_ORDERED: tuple[str, ...] = (
    "count",
    "sum",
    "avg",
    "min",
    "max",
)

VALID_AGGREGATION_FUNCTIONS = frozenset(_AGGREGATION_FUNCTION_NAMES_ORDERED)

WINDOW_RANKING_FUNCTIONS = frozenset({"row_number", "rank", "dense_rank"})

WINDOW_AGG_FUNCTIONS = frozenset({"sum", "avg"})

WINDOW_OFFSET_FUNCTIONS = frozenset({"lag", "lead"})

WINDOW_VALUE_FUNCTIONS = frozenset({"first_value", "last_value"})

VALID_WINDOW_FUNCTIONS = (
    WINDOW_RANKING_FUNCTIONS | WINDOW_AGG_FUNCTIONS | WINDOW_OFFSET_FUNCTIONS | WINDOW_VALUE_FUNCTIONS
)

VALID_SENSITIVITY_LEVELS = frozenset({"none", "restricted", "hidden"})

HIDDEN_SENSITIVITIES = frozenset({"hidden", "restricted"})

VALID_SCALAR_FUNCTIONS = {
    "upper",
    "lower",
    "trim",
    "ltrim",
    "rtrim",
    "length",
    "abs",
    "round",
    "floor",
    "ceil",
    "date_trunc",
    "date_part",
    "extract",
    "coalesce",
    "concat",
    "year",
    "month",
    "day",
}

SCALAR_FUNCTIONS_STRING = {
    "upper",
    "lower",
    "trim",
    "ltrim",
    "rtrim",
    "length",
    "concat",
}

SCALAR_FUNCTIONS_VARIADIC = frozenset({"concat"})

SCALAR_FUNCTIONS_NUMERIC = {"abs", "round", "floor", "ceil"}

SCALAR_FUNCTIONS_TEMPORAL = {
    "date_trunc",
    "date_part",
    "extract",
    "year",
    "month",
    "day",
}

SCALAR_FUNCTIONS_LEADING_ARG = {"date_trunc", "date_part", "extract"}

DISALLOWED_EXTRACT_UNITS = {"epoch"}

VALID_GRAINS = {"scalar", "grouped", "row_level"}

VALID_EXPECTED_ROWS = {"one", "few", "many"}

REGISTRY_TOKEN_PATTERN = r"^[wc]\d+$"

VALID_HAVING_OPS = {"=", "!=", "<", "<=", ">", ">=", "in", "not in", "between"}

QUALIFY_SKIP_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "avg",
        "case",
        "cast",
        "coalesce",
        "count",
        "date_part",
        "date_trunc",
        "extract",
        "lateral",
        "lower",
        "max",
        "min",
        "nullif",
        "replace",
        "substring",
        "sum",
        "trim",
        "try_cast",
        "unnest",
        "upper",
        "values",
    }
)

NATIVE_BACKEND_ENGINES: frozenset[str] = frozenset(
    {
        "databricks",
        "snowflake",
        "bigquery",
        "sqlserver",
        "mysql",
        "mariadb",
        "redshift",
        "duckdb",
        "sqlite",
        "csv",
    }
)

EMBEDDED_ENGINE_NAMES: frozenset[str] = frozenset({"duckdb", "sqlite", "csv"})

SQLGLOT_DIALECT_HOOK_ENGINES: frozenset[str] = frozenset(
    name for name in CANONICAL_ENGINE_ORDER if name != "postgresql"
)

QUALIFIED_TABLE_REF_ENGINES: frozenset[str] = frozenset(
    {
        "mysql",
        "mariadb",
        "sqlserver",
        "snowflake",
        "bigquery",
        "databricks",
        "redshift",
    }
)

STRUCTURAL_INDEX_ENGINES: frozenset[str] = frozenset(
    {
        "mysql",
        "mariadb",
        "redshift",
        "snowflake",
        "bigquery",
        "databricks",
        "duckdb",
        "sqlite",
        "csv",
        "sqlserver",
    }
)

VALID_VALUE_TYPES = {
    "integer",
    "string",
    "date",
    "number",
    "null",
    "boolean",
    "date_window",
    "date_diff",
}

VALID_RELATIVE_DATE_UNITS = frozenset(
    {"day", "week", "month", "quarter", "half_year", "year", "hour", "minute", "second"}
)

YEAR_LITERAL_COMPARISON_OPS: frozenset[str] = frozenset({"=", ">", "<", ">=", "<="})
YEAR_LITERAL_RE = re.compile(r"^(19|20)\d{2}$")

DATE_UNIT_ALIAS_TO_CANONICAL: dict[str, str] = {
    "d": "day",
    "day": "day",
    "days": "day",
    "daily": "day",
    "w": "week",
    "wk": "week",
    "wks": "week",
    "week": "week",
    "weeks": "week",
    "weekly": "week",
    "mo": "month",
    "mon": "month",
    "mos": "month",
    "month": "month",
    "months": "month",
    "monthly": "month",
    "q": "quarter",
    "qtr": "quarter",
    "qtrs": "quarter",
    "quarter": "quarter",
    "quarters": "quarter",
    "quarterly": "quarter",
    "h1": "half_year",
    "h2": "half_year",
    "halfyear": "half_year",
    "half_year": "half_year",
    "half-year": "half_year",
    "halfyears": "half_year",
    "half_years": "half_year",
    "semester": "half_year",
    "semesters": "half_year",
    "semiannual": "half_year",
    "semi_annual": "half_year",
    "semi-annual": "half_year",
    "semiannually": "half_year",
    "y": "year",
    "yr": "year",
    "yrs": "year",
    "yyyy": "year",
    "year": "year",
    "years": "year",
    "yearly": "year",
    "annual": "year",
    "annually": "year",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "hourly": "hour",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "s": "second",
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
}

DATE_UNIT_PLURAL_TO_SINGULAR: dict[str, str] = {
    alias: canonical
    for alias, canonical in DATE_UNIT_ALIAS_TO_CANONICAL.items()
    if alias.endswith("s") and alias != canonical
}

DATE_INTERVAL_EXPR_SUBSTRINGS: tuple[str, ...] = (
    "current_date",
    "current_timestamp",
    "now()",
    "sysdate",
    "interval",
    "localtimestamp",
    "localtime",
    "utc_timestamp",
)

DESCRIPTIVE_ALLOWED_VALUE_TYPES = frozenset({"string", "integer"})

DESCRIPTIVE_EXCLUDED_VALUE_TYPES = frozenset({"date", "boolean", "number"})

VALID_FILTER_VALUE_TYPES = {
    "categorical",
    "numeric",
    "numeric_categorical",
    "temporal",
    "boolean",
    "null",
}

VALID_HAVING_VALUE_TYPES = {"number", "integer"}

VALUE_TYPE_NORMALIZATION = {
    "timestamp": "date",
    "datetime": "date",
    "timestamptz": "date",
    "time": "date",
    "numeric": "number",
    "decimal": "number",
    "float": "number",
    "double": "number",
    "real": "number",
    "money": "number",
    "bigint": "integer",
    "smallint": "integer",
    "int": "integer",
    "int2": "integer",
    "int4": "integer",
    "int8": "integer",
    "serial": "integer",
    "varchar": "string",
    "char": "string",
    "text": "string",
    "bpchar": "string",
    "uuid": "string",
    "bool": "boolean",
    "enum": "string",
    "integer": "integer",
    "string": "string",
    "date": "date",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "date_window": "date_window",
    "date_diff": "date_diff",
}

_BOOLEAN_FILTER_OPS = {"=", "!=", "in", "not in", "is null", "is not null"}

_CATEGORICAL_FILTER_OPS = {
    "=",
    "!=",
    "like",
    "ilike",
    "not like",
    "not ilike",
    "in",
    "not in",
    "is null",
    "is not null",
}

_NUMERIC_CATEGORICAL_FILTER_OPS = {
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not in",
    "between",
    "is null",
    "is not null",
}

_NUMERIC_FILTER_OPS = frozenset(
    {
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "in",
        "not in",
        "between",
        "is null",
        "is not null",
    }
)

CTE_NUMERIC_FILTER_OPS = list(_NUMERIC_FILTER_OPS)

ROLE_ALLOWED_AGGREGATIONS = {
    "IDENTIFIER": {"count"},
    "CATEGORICAL": {"count", "min", "max"},
    "NUMERIC_CATEGORICAL": {"count", "min", "max"},
    "NUMERIC_MEASURE": {"count", "sum", "avg", "min", "max"},
    "TEMPORAL": {"count", "min", "max"},
    "BOOLEAN": {"count"},
    "FREE_TEXT": {"count"},
    "AUDIT": set(),
}

VALID_FILTER_OPS: frozenset[str] = frozenset(
    _BOOLEAN_FILTER_OPS | _CATEGORICAL_FILTER_OPS | _NUMERIC_CATEGORICAL_FILTER_OPS | frozenset({"contains"})
)

NUMERIC_ONLY_AGGREGATIONS = {"sum", "avg"}

COLUMN_TYPE_TO_VALUE_TYPE = {
    "int": "integer",
    "integer": "integer",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "int2": "integer",
    "int4": "integer",
    "int8": "integer",
    "long": "integer",
    "short": "integer",
    "serial": "integer",
    "bigserial": "integer",
    "smallserial": "integer",
    "float": "number",
    "double": "number",
    "decimal": "number",
    "numeric": "number",
    "real": "number",
    "float4": "number",
    "float8": "number",
    "double precision": "number",
    "money": "number",
    "varchar": "string",
    "text": "string",
    "char": "string",
    "string": "string",
    "character varying": "string",
    "bpchar": "string",
    "nchar": "string",
    "nvarchar": "string",
    "ntext": "string",
    "clob": "string",
    "date": "date",
    "timestamp": "date",
    "timestamptz": "date",
    "datetime": "date",
    "time": "date",
    "timestamp without time zone": "date",
    "timestamp with time zone": "date",
    "boolean": "boolean",
    "bool": "boolean",
    "number": "number",
    "byteint": "integer",
    "int64": "integer",
    "int32": "integer",
    "int16": "integer",
    "float64": "number",
    "float32": "number",
    "bignumeric": "number",
    "bytes": "string",
    "blob": "string",
    "timestamp_ntz": "date",
    "timestamp_ltz": "date",
    "timestamp_tz": "date",
    "datetime2": "date",
    "smalldatetime": "date",
    "datetimeoffset": "date",
}

AGGREGATION_ALLOWED_COLUMN_TYPES = {
    "count": ["integer", "string", "date", "number", "boolean"],
    "sum": ["integer", "number"],
    "avg": ["integer", "number"],
    "min": ["integer", "number", "string", "date"],
    "max": ["integer", "number", "string", "date"],
}

EXCLUDED_FILTER_PATTERNS = [
    r"password",
    r"picture",
    r"photo",
    r"image",
    r"blob",
    r"address.?2",
    r"address_line.?2",
]

BOOLEAN_TRUTH_PATTERN_MAP: Mapping[frozenset[str], str] = MappingProxyType(
    {
        frozenset({"0", "1"}): "1",
        frozenset({"true", "false"}): "true",
        frozenset({"yes", "no"}): "yes",
        frozenset({"y", "n"}): "y",
        frozenset({"t", "f"}): "t",
        frozenset({"on", "off"}): "on",
        frozenset({"active", "inactive"}): "active",
        frozenset({"enabled", "disabled"}): "enabled",
        frozenset({"pass", "fail"}): "pass",
        frozenset({"passed", "failed"}): "passed",
        frozenset({"pass", "failed"}): "pass",
        frozenset({"passed", "fail"}): "passed",
        frozenset({"open", "closed"}): "open",
        frozenset({"opened", "closed"}): "opened",
        frozenset({"open", "close"}): "open",
        frozenset({"opened", "close"}): "opened",
        frozenset({"success", "failure"}): "success",
        frozenset({"succeeded", "failed"}): "succeeded",
        frozenset({"success", "failed"}): "success",
        frozenset({"present", "absent"}): "present",
    }
)

BOOLEAN_NEGATION_PREFIXES: tuple[str, ...] = (
    "no ",
    "no_",
    "not ",
    "not_",
    "non ",
    "non_",
    "non-",
    "un",
    "in",
    "a",
    "dis",
    "de",
    "mis",
    "ir",
    "il",
    "im",
)

BOOLEAN_NEGATION_SUFFIXES: tuple[str, ...] = ("less",)

BOOLEAN_ANTONYM_MIN_STEM_LEN: int = 3

BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES: tuple[str, ...] = ("a ", "an ")

FK_INFERENCE_SUFFIX_STEMS: tuple[str, ...] = (
    "_id",
    "_key",
    "_uuid",
    "_pk",
    "_fk",
    "_ref",
    "_no",
    "_num",
    "_code",
    "id",
    "key",
    "uuid",
    "no",
    "num",
    "code",
)

DO_NOT_LEMMATIZE: frozenset[str] = frozenset(
    {
        "status",
        "address",
        "business",
        "process",
        "axis",
        "basis",
        "news",
        "analysis",
    },
)

STOPWORDS_GRAMMATICAL_PARTICLES: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "and",
        "or",
        "as",
        "that",
        "this",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "it",
        "me",
        "us",
        "my",
        "our",
        "your",
        "its",
        "please",
        "kindly",
        "just",
        "simply",
        "also",
        "then",
        "want",
        "wanted",
        "need",
        "needed",
    },
)

SELF_JOIN_CTE_NAME_PREFIX: str = "sj_"

NUMERIC_TYPE_TOKENS = frozenset(
    {
        "int",
        "integer",
        "float",
        "double",
        "decimal",
        "numeric",
        "real",
        "number",
        "serial",
        "bigint",
        "smallint",
        "tinyint",
        "money",
        "long",
        "short",
    }
)

STRING_TYPE_TOKENS = frozenset(
    {
        "char",
        "varchar",
        "text",
        "string",
        "clob",
        "nchar",
        "nvarchar",
        "ntext",
        "bpchar",
    }
)

DATE_TYPE_TOKENS = frozenset(
    {
        "date",
        "time",
        "timestamp",
        "timestamptz",
        "interval",
    }
)

DATE_FRIENDLY_VALUE_TYPES = frozenset({"date", "date_window", "timestamp"})

OP_FLIP: dict[str, str] = {">": "<", "<": ">", ">=": "<=", "<=": ">="}

COMPATIBLE_TYPE_PAIRS = {
    ("int", "int"),
    ("int", "integer"),
    ("int", "bigint"),
    ("int", "smallint"),
    ("int", "tinyint"),
    ("int", "long"),
    ("int", "short"),
    ("int", "numeric"),
    ("int", "decimal"),
    ("integer", "integer"),
    ("integer", "int"),
    ("integer", "bigint"),
    ("integer", "smallint"),
    ("integer", "tinyint"),
    ("integer", "long"),
    ("integer", "short"),
    ("bigint", "bigint"),
    ("bigint", "int"),
    ("bigint", "integer"),
    ("bigint", "smallint"),
    ("bigint", "tinyint"),
    ("bigint", "long"),
    ("bigint", "numeric"),
    ("smallint", "smallint"),
    ("smallint", "int"),
    ("smallint", "integer"),
    ("smallint", "bigint"),
    ("smallint", "tinyint"),
    ("tinyint", "tinyint"),
    ("tinyint", "int"),
    ("tinyint", "integer"),
    ("tinyint", "smallint"),
    ("tinyint", "bigint"),
    ("long", "long"),
    ("long", "int"),
    ("long", "integer"),
    ("long", "bigint"),
    ("short", "short"),
    ("short", "int"),
    ("short", "integer"),
    ("short", "smallint"),
    ("short", "tinyint"),
    ("numeric", "numeric"),
    ("numeric", "decimal"),
    ("numeric", "int"),
    ("numeric", "integer"),
    ("numeric", "bigint"),
    ("decimal", "decimal"),
    ("decimal", "numeric"),
    ("decimal", "int"),
    ("decimal", "integer"),
    ("float", "float"),
    ("float", "double"),
    ("float", "real"),
    ("float", "numeric"),
    ("double", "double"),
    ("double", "float"),
    ("double", "real"),
    ("real", "real"),
    ("real", "float"),
    ("real", "double"),
    ("varchar", "varchar"),
    ("varchar", "text"),
    ("varchar", "char"),
    ("varchar", "string"),
    ("text", "text"),
    ("text", "varchar"),
    ("text", "char"),
    ("text", "string"),
    ("char", "char"),
    ("char", "varchar"),
    ("char", "text"),
    ("char", "string"),
    ("string", "string"),
    ("string", "varchar"),
    ("string", "text"),
    ("string", "char"),
    ("date", "date"),
    ("date", "timestamp"),
    ("date", "timestamptz"),
    ("timestamp", "timestamp"),
    ("timestamp", "date"),
    ("timestamp", "timestamptz"),
    ("timestamptz", "timestamptz"),
    ("timestamptz", "timestamp"),
    ("timestamptz", "date"),
    ("boolean", "boolean"),
    ("boolean", "bool"),
    ("bool", "bool"),
    ("bool", "boolean"),
    ("number", "number"),
    ("number", "integer"),
    ("number", "numeric"),
    ("number", "decimal"),
    ("number", "float"),
    ("number", "double"),
    ("number", "real"),
    ("integer", "number"),
}

SCALAR_FUNC_DEFAULTS: dict[str, list[int | str]] = {
    "round": [2],
    "trunc": [0],
    "truncate": [0],
    "coalesce": [0],
    "date_trunc": ["month"],
    "date_part": ["month"],
    "extract": ["year"],
}

DATE_UNIT_KEYWORDS = [
    ("month", "month"),
    ("day", "day"),
    ("week", "week"),
    ("quarter", "quarter"),
    ("year", "year"),
    ("date", "year"),
]

BOOLEAN_TRUTHY_VALUES = frozenset(
    {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "active",
        "enabled",
        "pass",
        "passed",
        "open",
        "opened",
        "success",
        "succeeded",
        "present",
    }
)

BOOLEAN_FALSY_VALUES = frozenset(
    {
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "inactive",
        "disabled",
        "fail",
        "failed",
        "closed",
        "close",
        "failure",
        "absent",
    }
)

NUMERIC_DATA_TYPES = frozenset(
    {
        "integer",
        "int",
        "int2",
        "int4",
        "int8",
        "smallint",
        "bigint",
        "serial",
        "bigserial",
        "numeric",
        "decimal",
        "real",
        "double precision",
        "float",
        "float4",
        "float8",
        "money",
    }
)

REVERSE_OP_MAP: dict[str, str] = {
    **OP_FLIP,
    "=": "=",
    "!=": "!=",
    "like": "like",
    "not like": "not like",
    "ilike": "ilike",
    "not ilike": "not ilike",
    "in": "in",
    "not in": "not in",
    "is null": "is null",
    "is not null": "is not null",
}

QUESTION_STARTS_AGG = [
    "How many",
    "What is the total",
    "What is the average",
    "What is the minimum",
    "What is the maximum",
    "Find the sum of",
    "Calculate the",
    "Show the count of",
    "Get the total",
]

QUESTION_STARTS_LIST = [
    "List all",
    "Show me",
    "What are",
    "Which",
    "Find",
    "Display",
    "Get",
    "Return",
    "Retrieve",
]

QUESTION_STARTS_GROUP = [
    "Show me",
    "What is",
    "Group",
    "Break down",
    "Summarize",
    "Calculate",
    "Find the",
    "Get the",
]

SQL_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "distinct",
        "where",
        "group",
        "order",
        "having",
        "limit",
        "join",
        "inner",
        "outer",
        "left",
        "right",
        "cross",
        "on",
        "as",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "table",
        "index",
        "view",
        "into",
        "values",
        "set",
        "and",
        "or",
        "not",
        "in",
    }
)

JOIN_EDGE_KIND_RANK: dict[str, int] = {
    "catalog_fk": 0,
    "self_fk": 1,
    "inferred_suffix_fk": 2,
    "composite_fk": 2,
    "inferred_semantic_fk": 3,
    "virtual_pk_bridge": 4,
    "virtual_fk_bridge": 5,
    "virtual_fk_shadow_path": 6,
    "virtual_shared_pk": 7,
    "virtual_shared_base": 8,
    "virtual_shared_lineage": 9,
    "virtual_shared_fk_target": 9,
    "semantic_profile": 10,
    "semantic_profile_virtual": 11,
    "semantic_distinct_overlap": 12,
}

WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER: str = "rows"

WINDOW_DEFAULT_FRAME_START_WITH_ORDER: str = "UNBOUNDED PRECEDING"

WINDOW_DEFAULT_FRAME_END_WITH_ORDER: str = "CURRENT ROW"

WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER: str = "rows"

WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER: str = "UNBOUNDED PRECEDING"

WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER: str = "UNBOUNDED FOLLOWING"

SQL_WINDOW_FUNCTION_UPPER: dict[str, str] = {
    "row_number": "ROW_NUMBER",
    "rank": "RANK",
    "dense_rank": "DENSE_RANK",
    "sum": "SUM",
    "avg": "AVG",
    "lag": "LAG",
    "lead": "LEAD",
    "first_value": "FIRST_VALUE",
    "last_value": "LAST_VALUE",
}

IRREGULAR_PLURALS_MAP: dict[str, str] = {
    "data": "datum",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "indices": "index",
    "matrices": "matrix",
    "people": "person",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
}


class ExpansionOperatorId:
    """Stable expansion operator ids; registry keys and expansion metadata stamps."""

    FILTER_ADD = "FILTER_ADD"
    FILTER_EXPR_ADD = "FILTER_EXPR_ADD"
    AGG_CHANGE = "AGG_CHANGE"
    GROUPBY_ADD = "GROUPBY_ADD"
    ORDERBY_ADD = "ORDERBY_ADD"
    HAVING_VALUE_ADD = "HAVING_VALUE_ADD"
    HAVING_EXPR_ADD = "HAVING_EXPR_ADD"
    FILTER_REMOVE = "FILTER_REMOVE"
    GROUPBY_REMOVE = "GROUPBY_REMOVE"
    HAVING_REMOVE = "HAVING_REMOVE"
    JOIN_DIMENSION_ADD = "JOIN_DIMENSION_ADD"
    JOIN_FACT_ADD = "JOIN_FACT_ADD"
    DIMENSION_SWAP = "DIMENSION_SWAP"
    TABLE_REMOVE = "TABLE_REMOVE"
    BRIDGE_INTERMEDIATE_ADD = "BRIDGE_INTERMEDIATE_ADD"
    INCLUDE_GOLD = "INCLUDE_GOLD"
    TEMP_EXTRACT_GROUPBY = "TEMP_EXTRACT_GROUPBY"
    TEMP_DATE_TRUNC_GROUPBY = "TEMP_DATE_TRUNC_GROUPBY"
    TEMP_DATE_WINDOW_FILTER = "TEMP_DATE_WINDOW_FILTER"
    TEMP_DATE_DIFF_FILTER = "TEMP_DATE_DIFF_FILTER"
    NUM_ROUND_SELECT = "NUM_ROUND_SELECT"
    NUM_ABS_FILTER = "NUM_ABS_FILTER"
    DISTINCT_ADD = "DISTINCT_ADD"
    LIMIT_ADD = "LIMIT_ADD"
    FILTER_OR_GROUP = "FILTER_OR_GROUP"
    SELECT_EXPR_PAIR_MULTIPLY = "SELECT_EXPR_PAIR_MULTIPLY"
    WINDOW_RANK_ADD = "WINDOW_RANK_ADD"
    WINDOW_SUM_PARTITION_ADD = "WINDOW_SUM_PARTITION_ADD"
    SELECT_CASE_LABEL_ADD = "SELECT_CASE_LABEL_ADD"
    WINDOW_LAG_ADD = "WINDOW_LAG_ADD"
    WINDOW_LEAD_ADD = "WINDOW_LEAD_ADD"
    FILTER_ILIKE_ADD = "FILTER_ILIKE_ADD"
    FILTER_ARRAY_CONTAINS_ADD = "FILTER_ARRAY_CONTAINS_ADD"
    ORDERBY_REMOVE = "ORDERBY_REMOVE"
    LIMIT_REMOVE = "LIMIT_REMOVE"
    SELECT_COL_TRIM = "SELECT_COL_TRIM"
    WINDOW_STRIP = "WINDOW_STRIP"
    DISTINCT_REMOVE = "DISTINCT_REMOVE"
    SPLICE_SUBTREE = "SPLICE_SUBTREE"
    EMI_MUTATE = "EMI_MUTATE"
    CTE_WRAP_GROUPED = "CTE_WRAP_GROUPED"
    CTE_SCALAR_THRESHOLD = "CTE_SCALAR_THRESHOLD"
    CASE_CATEGORICAL_ADD = "CASE_CATEGORICAL_ADD"
    FILTER_IN_LIST_ADD = "FILTER_IN_LIST_ADD"
    FILTER_NULL_ADD = "FILTER_NULL_ADD"
    FILTER_NOT_NULL_ADD = "FILTER_NOT_NULL_ADD"
    HAVING_MATCH_SELECT_AGG = "HAVING_MATCH_SELECT_AGG"
    COUNT_DISTINCT_ADD = "COUNT_DISTINCT_ADD"
    WINDOW_DENSE_RANK_ADD = "WINDOW_DENSE_RANK_ADD"
    WINDOW_RANK_FUNC_ADD = "WINDOW_RANK_FUNC_ADD"
    WINDOW_AVG_PARTITION_ADD = "WINDOW_AVG_PARTITION_ADD"
    ORDERBY_WINDOW_COL_ADD = "ORDERBY_WINDOW_COL_ADD"
    FILTER_LIKE_ADD = "FILTER_LIKE_ADD"
    SELECT_COALESCE_ADD = "SELECT_COALESCE_ADD"
    SELECT_STRING_SCALAR_ADD = "SELECT_STRING_SCALAR_ADD"
    TEMP_EXTRACT_FILTER = "TEMP_EXTRACT_FILTER"
    CTE_UNNEST_ADD = "CTE_UNNEST_ADD"
    SELF_JOIN_CTE_ADD = "SELF_JOIN_CTE_ADD"
    MULTI_CTE_CHAIN_ADD = "MULTI_CTE_CHAIN_ADD"
    SPLICE_HAVING_SUBTREE = "SPLICE_HAVING_SUBTREE"
    SPLICE_WINDOW_SUBTREE = "SPLICE_WINDOW_SUBTREE"


POSTGRES_ENV_HOST: tuple[str, ...] = (
    "POSTGRES_HOST",
    "POSTGRES_SERVER",
    "POSTGRES_HOSTNAME",
    "PGHOST",
    "PGHOSTADDR",
)

POSTGRES_ENV_PORT: tuple[str, ...] = ("POSTGRES_PORT", "PGPORT")

POSTGRES_ENV_USER: tuple[str, ...] = ("POSTGRES_USER", "POSTGRES_USERNAME", "PGUSER")

POSTGRES_ENV_PASSWORD: tuple[str, ...] = ("POSTGRES_PASSWORD", "POSTGRES_PWD", "PGPASSWORD")

POSTGRES_ENV_DATABASE: tuple[str, ...] = ("POSTGRES_DATABASE", "POSTGRES_DB", "PGDATABASE")

POSTGRES_ENV_SCHEMA: tuple[str, ...] = ("POSTGRES_SCHEMA", "PGSCHEMA")

DATABRICKS_ENV_SERVER_HOSTNAME: tuple[str, ...] = (
    "DATABRICKS_HOST",
    "DATABRICKS_SERVER",
    "DATABRICKS_SERVER_HOSTNAME",
)

DATABRICKS_ENV_HTTP_PATH: tuple[str, ...] = (
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_SQL_HTTP_PATH",
    "DATABRICKS_WAREHOUSE_HTTP_PATH",
)

DATABRICKS_ENV_TOKEN: tuple[str, ...] = (
    "DATABRICKS_TOKEN",
    "DATABRICKS_ACCESS_TOKEN",
    "DATABRICKS_PAT",
    "ACCESS_TOKEN",
)

DATABRICKS_ENV_CATALOG: tuple[str, ...] = (
    "DATABRICKS_CATALOG",
    "SPARK_DEFAULT_CATALOG",
)

DATABRICKS_ENV_SCHEMA: tuple[str, ...] = (
    "DATABRICKS_SCHEMA",
    "DATABRICKS_DEFAULT_SCHEMA",
    "SPARK_DEFAULT_SCHEMA",
)

MYSQL_ENV_HOST: tuple[str, ...] = ("MYSQL_HOST", "MYSQL_SERVER", "MYSQL_HOSTNAME")

MYSQL_ENV_PORT: tuple[str, ...] = ("MYSQL_PORT", "MYSQL_TCP_PORT")

MYSQL_ENV_USER: tuple[str, ...] = ("MYSQL_USER", "MYSQL_USERNAME")

MYSQL_ENV_PASSWORD: tuple[str, ...] = ("MYSQL_PASSWORD", "MYSQL_PWD")

MYSQL_ENV_DATABASE: tuple[str, ...] = ("MYSQL_DATABASE", "MYSQL_DB")

MARIADB_ENV_HOST: tuple[str, ...] = ("MARIADB_HOST", "MARIADB_SERVER", "MARIADB_HOSTNAME")

MARIADB_ENV_PORT: tuple[str, ...] = ("MARIADB_PORT", "MARIADB_TCP_PORT")

MARIADB_ENV_USER: tuple[str, ...] = ("MARIADB_USER", "MARIADB_USERNAME")

MARIADB_ENV_PASSWORD: tuple[str, ...] = ("MARIADB_PASSWORD", "MARIADB_PWD")

MARIADB_ENV_DATABASE: tuple[str, ...] = ("MARIADB_DATABASE", "MARIADB_DB")

SQLSERVER_ENV_HOST: tuple[str, ...] = ("SQLSERVER_HOST", "SQLSERVER_SERVER", "MSSQL_HOST", "MSSQL_SERVER")

SQLSERVER_ENV_PORT: tuple[str, ...] = ("SQLSERVER_PORT", "MSSQL_PORT")

SQLSERVER_ENV_USER: tuple[str, ...] = ("SQLSERVER_USER", "SQLSERVER_USERNAME", "MSSQL_USER")

SQLSERVER_ENV_PASSWORD: tuple[str, ...] = (
    "SQLSERVER_PASSWORD",
    "SQLSERVER_PWD",
    "MSSQL_SA_PASSWORD",
    "MSSQL_PASSWORD",
)

SQLSERVER_ENV_DATABASE: tuple[str, ...] = (
    "SQLSERVER_DATABASE",
    "SQLSERVER_DB",
    "MSSQL_DATABASE",
    "MSSQL_DB",
)

SQLSERVER_ENV_SCHEMA: tuple[str, ...] = ("SQLSERVER_SCHEMA", "MSSQL_SCHEMA", "SQLSERVER_DEFAULT_SCHEMA")

SQLSERVER_ENV_DRIVER: tuple[str, ...] = ("SQLSERVER_DRIVER", "MSSQL_DRIVER", "ODBC_DRIVER")

SQLSERVER_ENV_AUTH_MODE: tuple[str, ...] = ("SQLSERVER_AUTH_MODE", "MSSQL_AUTH_MODE")

SQLSERVER_ENV_TENANT_ID: tuple[str, ...] = ("SQLSERVER_TENANT_ID", "MSSQL_TENANT_ID", "AZURE_TENANT_ID")

SQLSERVER_ENV_CLIENT_ID: tuple[str, ...] = ("SQLSERVER_CLIENT_ID", "MSSQL_CLIENT_ID", "AZURE_CLIENT_ID")

SQLSERVER_ENV_CLIENT_SECRET: tuple[str, ...] = (
    "SQLSERVER_CLIENT_SECRET",
    "MSSQL_CLIENT_SECRET",
    "AZURE_CLIENT_SECRET",
)

SNOWFLAKE_ENV_ACCOUNT: tuple[str, ...] = ("SNOWFLAKE_ACCOUNT", "SNOWSQL_ACCOUNT", "SF_ACCOUNT")

SNOWFLAKE_ENV_USER: tuple[str, ...] = ("SNOWFLAKE_USER", "SNOWFLAKE_USERNAME", "SNOWSQL_USER")

SNOWFLAKE_ENV_PASSWORD: tuple[str, ...] = ("SNOWFLAKE_PASSWORD", "SNOWFLAKE_PWD", "SNOWSQL_PWD")

SNOWFLAKE_ENV_DATABASE: tuple[str, ...] = ("SNOWFLAKE_DATABASE", "SNOWFLAKE_DB", "SNOWSQL_DATABASE")

SNOWFLAKE_ENV_SCHEMA: tuple[str, ...] = ("SNOWFLAKE_SCHEMA", "SNOWSQL_SCHEMA", "SNOWFLAKE_DEFAULT_SCHEMA")

SNOWFLAKE_ENV_WAREHOUSE: tuple[str, ...] = ("SNOWFLAKE_WAREHOUSE", "SNOWSQL_WAREHOUSE")

SNOWFLAKE_ENV_ROLE: tuple[str, ...] = ("SNOWFLAKE_ROLE", "SNOWSQL_ROLE")

SNOWFLAKE_ENV_PRIVATE_KEY_PATH: tuple[str, ...] = (
    "SNOWFLAKE_PRIVATE_KEY_PATH",
    "SNOWFLAKE_PRIVATE_KEY",
    "SNOWSQL_PRIVATE_KEY_PATH",
)

SNOWFLAKE_ENV_PRIVATE_KEY_PASSPHRASE: tuple[str, ...] = (
    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
    "SNOWSQL_PRIVATE_KEY_PASSPHRASE",
)

SNOWFLAKE_ENV_AUTHENTICATOR: tuple[str, ...] = ("SNOWFLAKE_AUTHENTICATOR", "SNOWSQL_AUTHENTICATOR")

SNOWFLAKE_ENV_OAUTH_TOKEN: tuple[str, ...] = ("SNOWFLAKE_OAUTH_TOKEN", "SNOWFLAKE_OAUTH", "SNOWSQL_OAUTH_TOKEN")

BIGQUERY_ENV_PROJECT: tuple[str, ...] = (
    "BIGQUERY_PROJECT",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_PROJECT",
)

BIGQUERY_ENV_DATASET: tuple[str, ...] = (
    "BIGQUERY_DATASET",
    "BIGQUERY_DB",
    "GCP_DATASET",
    "BIGQUERY_SCHEMA",
    "BQ_DATASET",
)

BIGQUERY_ENV_CREDENTIALS_PATH: tuple[str, ...] = (
    "BIGQUERY_CREDENTIALS_PATH",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GCP_CREDENTIALS_PATH",
    "BQ_CREDENTIALS_PATH",
)

BIGQUERY_ENV_LOCATION: tuple[str, ...] = (
    "BIGQUERY_LOCATION",
    "GCP_LOCATION",
    "BQ_LOCATION",
    "GOOGLE_CLOUD_LOCATION",
)

REDSHIFT_ENV_HOST: tuple[str, ...] = ("REDSHIFT_HOST", "REDSHIFT_SERVER")

REDSHIFT_ENV_PORT: tuple[str, ...] = ("REDSHIFT_PORT", "REDSHIFT_TCP_PORT")

REDSHIFT_ENV_USER: tuple[str, ...] = ("REDSHIFT_USER", "REDSHIFT_USERNAME")

REDSHIFT_ENV_PASSWORD: tuple[str, ...] = ("REDSHIFT_PASSWORD", "REDSHIFT_PWD")

REDSHIFT_ENV_DATABASE: tuple[str, ...] = ("REDSHIFT_DATABASE", "REDSHIFT_DB")

REDSHIFT_ENV_SCHEMA: tuple[str, ...] = ("REDSHIFT_SCHEMA",)

REDSHIFT_ENV_USE_IAM: tuple[str, ...] = ("REDSHIFT_USE_IAM", "REDSHIFT_IAM")

REDSHIFT_ENV_CLUSTER_IDENTIFIER: tuple[str, ...] = ("REDSHIFT_CLUSTER_IDENTIFIER", "REDSHIFT_CLUSTER_ID")

REDSHIFT_ENV_WORKGROUP: tuple[str, ...] = ("REDSHIFT_WORKGROUP", "REDSHIFT_SERVERLESS_WORKGROUP")

REDSHIFT_ENV_REGION: tuple[str, ...] = ("REDSHIFT_REGION", "REDSHIFT_AWS_REGION")

DUCKDB_ENV_PATH: tuple[str, ...] = (
    "DUCKDB_PATH",
    "DUCKDB_DATABASE",
    "DUCKDB_DATABASE_PATH",
    "DUCKDB_FILE",
    "DUCKDB_DB",
    "DUCKDB_DSN",
)

DUCKDB_ENV_SCHEMA: tuple[str, ...] = ("DUCKDB_SCHEMA", "DUCKDB_DEFAULT_SCHEMA")

CSV_ENV_DIRECTORY: tuple[str, ...] = ("CSV_DIRECTORY",)

CSV_ENV_FILES: tuple[str, ...] = ("CSV_FILES",)

SQLITE_ENV_PATH: tuple[str, ...] = (
    "SQLITE_PATH",
    "SQLITE_DATABASE",
    "SQLITE_DATABASE_PATH",
    "SQLITE_FILE",
    "SQLITE_DB",
    "SQLITE_DSN",
    "SQLITE3_DATABASE",
)

OPENAI_ENV_REQUIRED: tuple[str, ...] = ("OPENAI_API_KEY",)

AZURE_OPENAI_ENV_REQUIRED: tuple[str, ...] = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)

AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT: str = "AZURE_OPENAI_DEPLOYMENT_LIGHT"

AZURE_OPENAI_ENV_DEPLOYMENT_MEDIUM: str = "AZURE_OPENAI_DEPLOYMENT_MEDIUM"

AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY: str = "AZURE_OPENAI_DEPLOYMENT_HEAVY"

UNIT_TO_DAYS: dict[str, int] = {
    "day": 1,
    "week": 7,
    "month": 30,
    "quarter": 91,
    "half_year": 182,
    "year": 365,
}

NULL_CHECK_OPS: frozenset[str] = frozenset({"is null", "is not null"})

NULL_OP_NEGATED_ALIASES: frozenset[str] = frozenset(
    {
        "not is null",
        "not isnull",
        "not is_null",
        "is not_null",
        "isnotnull",
        "is_not_null",
    }
)

NULL_OP_DOUBLE_NEGATED_ALIASES: frozenset[str] = frozenset({"not is not null", "not isnotnull", "not is_not_null"})

NULL_OP_PLAIN_ALIASES: frozenset[str] = frozenset({"isnull", "is_null"})

DATE_RESULT_SCALARS: frozenset[str] = frozenset(
    {
        "date_trunc",
        "date_part",
        "extract",
        "current_date",
        "current_timestamp",
        "now",
        "make_date",
        "make_timestamp",
        "to_date",
        "to_timestamp",
    }
)

DATE_COLUMN_VALUE_TYPES: frozenset[str] = frozenset({"date", "datetime", "timestamp", "date_time", "time"})

STRING_COLUMN_VALUE_TYPES: frozenset[str] = frozenset({"string", "text", "varchar", "char"})

NON_NUMERIC_AGGS_FOR_DATES: frozenset[str] = frozenset({"min", "max"})

STRING_OPS: frozenset[str] = frozenset({"like", "not like", "ilike", "not ilike", "contains"})

UNKNOWN_DATEPART_TO_EXTRACT_UNIT: dict[str, str] = {
    "year": "year",
    "month": "month",
    "day": "day",
    "quarter": "quarter",
    "dayofweek": "dow",
    "dow": "dow",
    "weekday": "dow",
}

ARRAY_REWRITABLE_OPS: frozenset[str] = frozenset({"=", "!=", "like", "not like", "ilike", "not ilike"})

COLUMN_DEFINITION_STOP_WORDS: frozenset[str] = frozenset(
    {
        "NOT",
        "DEFAULT",
        "NULL",
        "PRIMARY",
        "UNIQUE",
        "REFERENCES",
        "CHECK",
        "CONSTRAINT",
        "GENERATED",
        "AUTO_INCREMENT",
        "AUTOINCREMENT",
        "COMMENT",
        "AS",
        "ROW",
        "FORMAT",
        "SIGNED",
        "UNSIGNED",
        "ZEROFILL",
        "ENCODE",
        "COLLATE",
    }
)

INTEGER_VALUE_TYPES: frozenset[str] = frozenset({"int", "integer", "bigint", "smallint", "tinyint", "long", "short"})

STRING_VALUE_TYPES: frozenset[str] = frozenset({"string", "text", "varchar", "char"})

INFERRED_PK_VALUE_TYPES: frozenset[str] = frozenset({"integer", "number", "string"})

PK_STYLE_FK_STEMS: frozenset[str] = frozenset({"_id", "_key", "_uuid", "_pk"})

VALID_FK_ADD_KEYS: frozenset[str] = frozenset({"from", "to", "kind"})

VALID_FK_REMOVE_KEYS: frozenset[str] = frozenset({"from", "to"})

VALID_PK_ADD_KEYS: frozenset[str] = frozenset({"table", "column"})

VALID_PK_REMOVE_KEYS: frozenset[str] = frozenset({"table", "column"})

VALID_FK_KINDS: frozenset[str] = frozenset({"structural", "semantic"})

OVERRIDE_DESCRIPTION_OWNER_STRINGS: frozenset[str] = frozenset(
    {"catalog", "profile", "notes", "llm_refinement", "user_override"},
)

OVERRIDE_ROLE_OWNER_STRINGS: frozenset[str] = frozenset(
    {
        "catalog",
        "profile",
        "llm",
        "boolean_coercion",
        "user_override",
        "pk_fk_coercion",
    },
)

OVERRIDES_EDITABLE_ENUMS: dict[str, list[str]] = {
    "table_role": ["dimension", "fact", "bridge", "unknown"],
    "column_role": [
        "identifier",
        "categorical",
        "numeric_categorical",
        "numeric_measure",
        "temporal",
        "boolean",
        "free_text",
        "audit",
    ],
    "column_sensitivity": [
        "none",
        "restricted",
        "hidden",
    ],
    "foreign_key_kind": ["structural", "semantic"],
}

DIAG_TO_FAILURE_CATEGORY: dict[str, str] = {
    "ast_parse_failed": "schema_validation",
    "multiple_statements": "schema_validation",
    "not_select": "schema",
    "subquery_not_allowed": "structural",
    "lateral_not_allowed": "structural",
    "cross_join_not_allowed": "structural",
    "using_not_allowed": "structural",
    "self_join_not_allowed": "structural",
    "exists_not_allowed": "structural",
    "forbidden_structure": "structural",
    "no_root": "schema_validation",
    "unknown_table": "unknown_table",
    "unknown_column": "unknown_column",
    "ambiguous_column": "column_ambiguous",
    "unknown_cte": "cte_table_reference",
    "cte_unreferenced": "cte_usage",
    "param_unbound": "unbound_placeholder",
    "param_undeclared": "unbound_placeholder",
    "non_grouped_select_col": "group_by_membership",
    "agg_in_where": "filter_aggregation",
    "having_without_group": "having_validity",
    "explain_cartesian_join": "wrong_join",
    "explain_zero_estimate": "semantic_contradiction",
    "explain_seq_scan_indexed": "other",
    "explain_type_mismatch": "type_mismatch",
    "explain_permission_denied": "access_policy",
    "explain_other": "execution_explain_failed",
    "explain_cost_exceeded": "execution_cost_exceeded",
}

DIAGNOSTIC_REPAIR_HANDLER_KEYS: dict[str, str] = {
    "unknown_column": "_repair_unknown_column",
    "ambiguous_column": "_repair_ambiguous_column",
    "unknown_table": "_repair_unknown_table",
    "non_grouped_select_col": "_repair_grain_consistency",
    "agg_in_where": "_repair_agg_in_where",
    "explain_cartesian_join": "_repair_cartesian",
    "explain_zero_estimate": "_repair_filter_overlap",
    "param_unbound": "_repair_param_binding",
}

SANDBOX_QUESTION_TIERS: tuple[str, ...] = (
    "questions",
    "validation_failures",
)
SANDBOX_RECIPES: tuple[str, ...] = (
    "chat_basics",
    "rejections",
    "reader_writer",
    "overrides",
    "migration",
    "validation_failures",
    "maintenance",
    "errors",
    "column_security",
    "full_session",
    "partition_pruning",
    "views",
    "aetherspace",
)
SANDBOX_TOUR_EXPECT_NO_SQL: frozenset[str] = frozenset(
    {
        "What's the weather today?",
        "Show payroll deductions by employee SSN.",
        "How many rentals happened on 2025-01-01?",
    }
)
SANDBOX_VALIDATION_FAILURE_QUESTIONS: frozenset[str] = frozenset(
    {
        "Show payroll deductions by employee SSN.",
        "Show me all staff salaries.",
        "How many rentals happened on 2025-01-01?",
        "How many rentals were made in total?",
        "How many items are there?",
    }
)
SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL: frozenset[str] = frozenset()
SANDBOX_DOCTOR_REQUIRED_MEMBERS: tuple[str, ...] = (
    "rental_shop_seed.sql",
    "rental_shop.sql",
    "rental_shop_views.sql",
    "rental_shop_notes.txt",
    "questions.txt",
    "fixtures/rental_shop_mock.json",
    "artifacts_baseline/owner/schema_graph.json.gz",
    "artifacts_baseline/consumer/schema_graph.json.gz",
    "schema_literals.json",
    "schema_overrides_demo.json",
    "sandbox_catalog.json",
    "sandbox_expectations.json",
    "sandbox_scenarios.json",
    "sandbox_handcrafted_fixtures.json",
    "sandbox_space_catalog_notes.txt",
    "migration_demo/schema_migration_map.json",
    "artifacts_baseline/aetherspaces/catalog.json",
)
SANDBOX_MIN_FIXTURE_COUNT = 100
SANDBOX_MIN_INTENT_FIXTURE_COUNT = 50
SANDBOX_SCHEMA_LITERALS_FILENAME = "schema_literals.json"
SANDBOX_INTERPRET_DOMAIN_FILENAME = "schema_interpret_domain.json"
MOCK_FIXTURE_STUB_SCHEMA_LITERALS: dict[str, str] = {"owner": "{}", "consumer": "{}"}
MOCK_FIXTURE_RETRY_CONTEXT_KEYS: frozenset[str] = frozenset(
    {"prior_attempt_failures", "prior_grounding_failures"},
)

CONSUMER_ALLOW_OBJECTS: frozenset[str] = frozenset(
    {
        "actor",
        "address",
        "author",
        "book",
        "category",
        "city",
        "country",
        "courier",
        "customer",
        "damage_report",
        "delivery",
        "film",
        "film_actor",
        "game",
        "game_supported_language",
        "inventory",
        "inventory_status_history",
        "item",
        "item_category",
        "item_feature",
        "language",
        "payment",
        "promotion",
        "promotion_redemption",
        "publisher",
        "purchase_line",
        "purchase_order",
        "rental",
        "reservation",
        "staff",
        "stock_transfer",
        "store",
        "supplier",
        "warehouse",
    }
)

CONSUMER_RESTRICTED_ALLOW_OBJECTS: frozenset[str] = frozenset(
    {
        "customer",
        "payment",
        "rental",
        "address",
        "city",
        "country",
    }
)

RENTAL_SHOP_VIEW_NAMES = ("active_customer_v", "store_revenue_v", "film_catalog_v")

WINDOW_ADD_OPS = frozenset(
    {
        ExpansionOperatorId.WINDOW_RANK_ADD,
        ExpansionOperatorId.WINDOW_DENSE_RANK_ADD,
        ExpansionOperatorId.WINDOW_RANK_FUNC_ADD,
        ExpansionOperatorId.WINDOW_SUM_PARTITION_ADD,
        ExpansionOperatorId.WINDOW_AVG_PARTITION_ADD,
        ExpansionOperatorId.WINDOW_LAG_ADD,
        ExpansionOperatorId.WINDOW_LEAD_ADD,
        ExpansionOperatorId.ORDERBY_WINDOW_COL_ADD,
    }
)

CASE_ADD_OPS = frozenset(
    {
        ExpansionOperatorId.SELECT_CASE_LABEL_ADD,
        ExpansionOperatorId.CASE_CATEGORICAL_ADD,
    }
)

CTE_ADD_OPS = frozenset(
    {
        ExpansionOperatorId.CTE_WRAP_GROUPED,
        ExpansionOperatorId.CTE_SCALAR_THRESHOLD,
        ExpansionOperatorId.CTE_UNNEST_ADD,
        ExpansionOperatorId.SELF_JOIN_CTE_ADD,
        ExpansionOperatorId.MULTI_CTE_CHAIN_ADD,
    }
)

HAVING_ADD_OPS = frozenset(
    {
        ExpansionOperatorId.HAVING_VALUE_ADD,
        ExpansionOperatorId.HAVING_EXPR_ADD,
        ExpansionOperatorId.HAVING_MATCH_SELECT_AGG,
    }
)

SIMPLE_AGG_NAMES: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})

AGG_NODE_TO_NAME: dict[type[exp.Expression], str] = {
    exp.Sum: "sum",
    exp.Count: "count",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
}

ALLOWED_JOIN_KINDS: frozenset[str | None] = frozenset({None, "INNER", "LEFT", "RIGHT", "FULL"})

DEFAULT_FILTER_OP_MAP: dict[str, str] = {
    "=": "=",
    "==": "=",
    "<>": "<>",
    "!=": "<>",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "like": "like",
    "not like": "not like",
    "ilike": "ilike",
    "not ilike": "not ilike",
}

CSV_SUFFIXES = frozenset({".csv", ".xlsx"})

BOOL_LITERALS = frozenset({"1", "0", "true", "false", "t", "f", "yes", "no"})

TASK_MODEL_TO_DEPLOYMENT_FIELD: dict[str, str] = {
    "gpt-4o-mini": "deployment_light",
    "gpt-4.1-mini": "deployment_medium",
    "gpt-5.4-mini": "deployment_heavy",
}

TASK_PROFILES: dict[str, dict[str, Any]] = {
    "intent": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "feedback": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "schema": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "schema_base": {
        "temperature": 0,
    },
    "ddl": {
        "temperature": 0,
    },
    "join": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "judge": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "conversation": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "default": {
        "temperature": 0,
    },
}

LLM_SENSITIVITY_STRIP_KEYS: frozenset[str] = frozenset({"sensitivity"})

INFERENCE_TAG_VALUES: frozenset[str] = frozenset(
    {
        "suffix",
        "self",
        "composite",
        "semantic",
        "semantic_promoted",
        "user_override_structural",
        "user_override_semantic",
    }
)

PK_INFERENCE_TAG_VALUES: frozenset[str] = frozenset({"ddl", "profile", "user_override"})

ROLE_OWNER_VALUES: frozenset[str] = frozenset(
    {
        "catalog",
        "profile",
        "llm",
        "boolean_coercion",
        "user_override",
        "pk_fk_coercion",
    }
)

RAW_SQL_AGG_OR_WINDOW_RE = re.compile(
    r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(|OVER\s*\(",
    re.IGNORECASE,
)

STAGE_ATTRIBUTION_TABLE: Mapping[str, Literal["ground", "compose"]] = MappingProxyType(
    {
        "column_not_found_in_chosen_tables": "compose",
        "chosen_table_lacks_required_column": "compose",
        "filter_targets_missing_column": "compose",
        "joinpath_does_not_exist": "ground",
        "grain_inconsistent_with_chosen_tables": "ground",
        "cte_chosen_tables_inconsistent": "ground",
        "window_partition_column_missing": "compose",
        "encoder_added_or_removed_tables": "compose",
        "json_schema_violation": "compose",
        "missing_required_field": "compose",
        "invalid_operator": "compose",
        "invalid_value_type": "compose",
        "unqualified_column_reference": "compose",
        "cte_dependency_cycle": "compose",
        "window_frame_syntax_invalid": "compose",
        "existence_filter_encoded_as_subquery": "compose",
        "self_reference_encoded_as_inline_self_join": "compose",
        "correlated_lookup_encoded_as_lateral": "compose",
    }
)

LITERAL_BEARING_CATEGORIES: frozenset[str] = frozenset({"missing_numeric_filter", "missing_temporal_column"})

LOGICAL_FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "unknown_table",
        "wrong_tables",
        "wrong_join",
        "grain_consistency",
        "grain_validity",
        "cte_table_reference",
        "cte_grain_consistency",
        "cte_grain_compatibility",
        "wrong_column_selection",
        "wrong_filter_logic",
    }
)

KEPT_ISSUE_SEVERITIES: frozenset[str] = frozenset({"error", "warning"})

CASE_WHEN_QUALIFIED_COLUMN_REF_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$",
)

DESCRIPTION_OWNER_VALUES: frozenset[str] = frozenset({"catalog", "profile", "notes", "llm_refinement", "user_override"})

AST_AGG_NODE_TO_NAME: dict[type[exp.Expression], str] = {
    exp.Sum: "sum",
    exp.Count: "count",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
}

INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES = (
    "Semantically-required join intermediates. The join enumerator only considers the SHORTEST FK paths "
    "connecting the tables in `tables`, and any table listed in `tables` whose columns are not referenced "
    "elsewhere in the intent (in `select_cols`, filters, `group_by_cols`, `having_param`, `order_by_cols`, "
    "or registry expressions) is pruned before enumeration. Therefore, listing an extra table in `tables` "
    "alone is never enough to force the join through it. "
    "When the descriptions on the endpoint tables (or on tables related to them) indicate that two "
    "semantically distinct FK paths connect the same pair of tables, and one path is strictly LONGER than "
    "the other, the shorter path will always be chosen unless you force the longer one explicitly. "
    "To force the longer path, reference at least one column from each required intermediate table inside "
    "the intent. Adding such a column to `select_cols` is preferred — typically the intermediate table's "
    "primary key or its most descriptive column — because it both keeps the table from being pruned and "
    "surfaces context to the user that this specific semantic was intended. "
    "Concretely, suppose the intent connects `table` to `other_table` and the descriptions indicate two semantic "
    "paths: `table -> link_table -> other_table` for one semantic and `table -> junction_table -> bridge_table -> other_table` for another. "
    "If the question requires the second semantic, add at least one column from `junction_table` or one column from "
    "`bridge_table` (preferably to `select_cols`, ideally a primary key or the most descriptive column from each). "
    "If the question requires the first semantic, no extra columns are needed because the shorter path is "
    "already what the resolver will pick. When descriptions are silent about distinct paths, or when only "
    "one FK path exists between the endpoints, no extra columns are needed. "
    "COUNT(*) does not count as a column reference for prune purposes; when more than one physical "
    "table is in scope, emit COUNT(table.primary_key_column) on the table being counted (and reference "
    "any other required tables via qualified columns) instead of COUNT(*)."
)

INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE = (
    "`natural_language` field shape. The `natural_language` field must read like a question a non-technical "
    "user would ask, not a description of how the SQL is computed. Even when the intent uses CTEs, derived "
    "steps, or window functions, compress the overall information need into a single conversational sentence. "
    "Never enumerate steps, never reference CTE names, never mention SQL operators. The reader of this field "
    'is the end user being asked "I understood: ; is this what you wanted?"'
)

REPAIR_INSTRUCTIONS: dict[str, str] = {
    "extract_epoch": (
        "Do not use EXTRACT(EPOCH FROM ...). Use date column subtraction "
        "(table.date_column - other_table.other_date_column) or other supported "
        "date functions for time differences."
    ),
    "grain_validity": "Use one of the allowed grain values: 'scalar', 'grouped', or 'row_level'.",
    "grain_consistency": "Ensure grain matches the query structure. 'grouped' requires group_by_cols and aggregation in select_cols. 'row_level' means no GROUP BY and no aggregation. 'scalar' means a single aggregated value with no GROUP BY.",
    "schema_validation": "One or more tables or columns do not exist in the schema. Check allowed_tables and use only exact column names from each table.",
    "unknown_table": "The table does not exist in the schema. Remove it from tables and rewrite any references to use only tables that appear in allowed_tables.",
    "unknown_column": "The column does not exist in its table. Check the schema for available columns with a similar name or meaning and replace the reference.",
    "semantic_contradiction": "The intent contains contradictory operations. Keep only the aggregation or pattern that matches the question.",
    "expression_type": "Arithmetic expressions require numeric columns. Ensure all operands in arithmetic and all comparison sides have compatible types.",
    "filter_aggregation": "Conditions with aggregation functions (COUNT, SUM, AVG, MIN, MAX) belong in having_param, not filters_param. Move the condition.",
    "having_aggregation": "HAVING conditions must have an aggregation function in left_expr. Conditions without aggregation belong in filters_param.",
    "filter_semantic": "Fix the filter comparison: remove self-comparisons and ensure type compatibility between left and right expressions.",
    "having_semantic": "Fix the HAVING comparison: remove self-comparisons and ensure type compatibility between aggregated expressions.",
    "nested_aggregation": "Nested aggregation is not allowed. Use a CTE: compute the inner aggregation in a CTE step, then aggregate the CTE output in the main query.",
    "mixed_aggregation": "An expression cannot mix aggregated and bare column terms. Either wrap all terms in an aggregation function or add bare columns to group_by_cols.",
    "group_by_membership": "Every non-aggregated column in select_cols must appear in group_by_cols when grain is 'grouped'. Add the missing column to group_by_cols or wrap it in an aggregation.",
    "order_by_aggregation": "ORDER BY cannot contain aggregation when grain is 'row_level'. Change grain to 'grouped' or remove the aggregation from order_by_cols.",
    "aggregation_hint": "The question contains a quantity-comparison phrase that typically requires COUNT or SUM aggregation with GROUP BY and HAVING. Add aggregation in select_cols, group_by_cols on the entity, having_param with the threshold, and set grain to 'grouped'.",
    "column_schema": "A referenced column does not exist in its table. Check the schema for the correct column name.",
    "table_schema": "A referenced table does not exist in the schema. Use only tables from allowed_tables.",
    "filter_type_ops": "The filter operator is not compatible with the column data type. Use an appropriate operator for the column type.",
    "null_filter": "NULL checks must use 'is null' or 'is not null' operator with no value or value_type field.",
    "between_filter": (
        "Use op 'between' with value as a two-element array [lower, upper]. "
        "The pipeline decomposes it into >= and <= automatically."
    ),
    "date_window": (
        "For relative time-window filters, "
        "use value_type 'date_window' with the date column in left_expr, "
        'op \'>=\' and value as {"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. The amount is interpreted as an ISO half-open window: "
        "the start is N units before today, the end is exclusive. Prefer unit 'day' whenever "
        "the question phrases the window in days, weeks, or fortnights (convert weeks to days) "
        "so the start/end boundary matches the schema's daily granularity."
    ),
    "date_diff": (
        "For date-difference filters comparing two date columns "
        "(table.other_date_column - table.date_column compared to a duration), use value_type 'date_diff' with "
        "left_expr as the date subtraction expression (later minus earlier per the question), "
        "op as the comparison operator, and value as "
        '{"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. Prefer unit 'day' whenever the question phrases the duration in "
        "days or weeks. Do NOT use date_diff for relative date-window "
        "filters; use date_window instead."
    ),
    "date_integer_days": (
        "For date-shift arithmetic comparing a date column shifted by an integer day count to another date column, express "
        "table.date_column + table.integer_column or table.date_column - table.integer_column directly in left_expr/right_expr using "
        "+/- between the date column and the integer day count (literal or column). Do not use "
        "date_diff when comparing a shifted date to another date column."
    ),
    "agg_role": "SUM and AVG should only be applied to numeric measure columns. Use COUNT for non-measure columns, or select a numeric column.",
    "agg_type": "SUM and AVG require a numeric column. The referenced column is not numeric. Use COUNT instead or choose a numeric column.",
    "for_each_grouping": "The question contains a 'for each', 'per', or 'by' phrase implying a GROUP BY on the referenced entity. Add the entity's identifying column to group_by_cols, include it as a non-aggregated entry in select_cols, and set grain to 'grouped'.",
    "scalar_func_type": "The scalar function is applied to an incompatible column type. Ensure the column type matches what the function expects.",
    "threshold_missing_having": "The question contains a threshold phrase and the intent already has aggregation, but no HAVING condition is defined. Add a HAVING clause that compares the aggregated column to the numeric threshold in the question.",
    "cte_structure": "CTE steps require a cte_name string, an output_columns list of alias strings, and valid tables.",
    "cte_grain_consistency": "CTE grain must match its structure: same rules as the main query regarding grain, group_by_cols, and aggregation.",
    "cte_table_reference": "A CTE references an unknown table. A CTE can only reference schema tables or CTEs defined earlier in the same WITH list.",
    "cte_grain_compatibility": "A row_level query or CTE depends on an aggregated CTE. Ensure the grain is compatible with upstream CTE grains.",
    "cte_aggregation": "A CTE has HAVING conditions but no aggregation in its select_cols. Add aggregation or remove the HAVING.",
    "agg_keyword_missing": "The question asks for an aggregation (total, count, average, sum, etc.) but the intent has no aggregated column and no HAVING condition. Add the appropriate aggregation function to select_cols, include all tables needed to compute the aggregated value, and set grain to 'grouped' with the correct group_by_cols.",
}

PLANNER_NL_CONVENTIONS_BODY: dict[str, Any] = {
    "mandatory": [
        "Copy every literal that constrains rows or ordering into the matching prose field; the encoder never sees the original question.",
        "List only semantic base tables in top-level tables; omit junction_table from tables when its columns appear in prose.",
        "Reference window and case output names from select when you define them in window or case using as <registry_name>.",
        "Never emit SQL set operators, EXISTS, NOT EXISTS, LATERAL, param_key, raw_value, wNN, cNN, filter_group, or IR vocabulary.",
        "Never use as <name> inside select, filter, having, group_by, order_by, or limit; select output aliases are assigned later by the pipeline.",
        "Leave group_by, order_by, and limit empty unless the question explicitly asks for grouping, ordering, or a row cap; do not invent presentation-layer sorting or grouping for context.",
        "Project select prose to primary keys, primary human-readable label columns such as names or titles, and every column referenced by stated filters, ordering, grouping, having, or limits; never enumerate every physical column unless the question explicitly asks for all columns, every column, or complete row dumps.",
        "After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit table.column in the appropriate prose field.",
        "The tables list alone never keeps a table in join scope; qualified table.column tokens must appear in select, filter, group_by, having, order_by, or registry prose.",
        "When membership or existence requires link_table or junction_table, name junction_table.column or bridge_table.column in select prose, not only join-equality narration in filter prose.",
        "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
        "Existence or membership conditions belong in filter prose with binding columns named.",
        "Copy relative time unit and amount into filter prose; copy two-date duration comparisons into filter prose naming both columns.",
        "For cte_steps, tables lists base schema tables and names of prior cte_steps this step reads from; do not use a separate dependency field.",
    ],
    "recommended": [
        "Qualify columns as table.column when multiple listed tables share a column name.",
        "Describe aggregates with phrases such as sum of other_table.other_column.",
        "Describe anti-existence with plain language or other_table.pk is null style wording.",
    ],
}

PLANNER_NL_CONVENTIONS: Mapping[str, Any] = MappingProxyType(PLANNER_NL_CONVENTIONS_BODY)

ENCODER_NL_PHRASE_MAPPINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "=": ("equals", "equal to", "is", "are", "="),
        "!=": ("not equal", "not equals", "different from", "!="),
        ">": ("greater than", "more than", "above", "over", ">"),
        ">=": ("at least", "greater than or equal to", ">="),
        "<": ("less than", "below", "under", "<"),
        "<=": ("at most", "less than or equal to", "<="),
        "like": ("like", "matches", "contains pattern"),
        "not like": ("not like", "does not match"),
        "in": ("in", "one of", "any of", "among"),
        "not in": ("not in", "none of", "not among"),
        "between": ("between", "from", "to"),
        "is null": ("is null", "is absent", "is missing", "has no", "without"),
        "is not null": ("is not null", "has", "with", "present"),
        "sum": ("sum of", "total of", "sum", "total"),
        "avg": ("average of", "avg of", "mean of", "average"),
        "min": ("minimum of", "min of", "smallest", "lowest"),
        "max": ("maximum of", "max of", "largest", "highest"),
        "count": ("count of", "number of", "how many"),
        "asc": ("ascending", "asc", "lowest first"),
        "desc": ("descending", "desc", "highest first"),
        "date_window": ("last", "past", "within the last", "in the last", "recent", "since"),
        "date_diff": (
            "more than N days after",
            "less than N days before",
            "at least N days between",
        ),
    }
)

ENCODER_NL_TO_IR_GUIDANCE: tuple[str, ...] = (
    "Match planner prose phrases against nl_phrase_mappings case-insensitively to pick IR operators and aggregates.",
    "Infer value_type from literal form: single-quoted text is string; bare digits are integer or number; ISO dates are date; true or false are boolean; is null operator uses value_type null without a value.",
    "Emit AggregateCol with alias empty string; never invent select-column aliases.",
    "WindowRegistryStep.name and CaseRegistryStep.name come from as <registry_name> inside planner window and case prose; CTE names come from cte_steps[].name.",
    "logical_intent is the sole semantic source; translate select, filter, group_by, having, and order_by prose mechanically without inventing or dropping columns named in prose.",
    "Emit a qualified column reference in the IR slot matching each clause for every table.column named in that clause's prose at that scope.",
    'Phrases such as last N days, past N weeks, or within the last N months map to value_type date_window with {"unit", "amount"}.',
    "Elapsed time between two date columns maps to value_type date_diff with left_expr as a subtraction expression.",
    "CURRENT_DATE and CURRENT_TIMESTAMP map to keyword right_expr leaves, not string raw_value literals.",
    "Integer columns with schema role temporal and type integer are day-count durations; compare them to elapsed-day expressions, not as calendar dates.",
    "Date shifted by an integer duration column uses table.date_column + table.integer_column in left_expr or right_expr, not date_diff.",
)

ENCODER_IR_ASSEMBLY_RULES: tuple[str, ...] = (
    "filter_group is OR-of-AND groups parsed from planner prose.",
    "Emit raw_value for every Filter and Having literal; never emit param_key or param_values.",
    "Every column_ref is table.column using only structural_schema_for_chosen_tables identifiers.",
    "Downstream repair keeps only tables with column references in each scope's IR; emit refs only for table.column tokens named in the matching clause prose.",
    "Do not invent filters, grouping, or columns absent from logical_intent prose fields.",
    "For CONCAT expressions, place every concat argument into the same MulGroup.multiply list under scalar_func='concat'. Do not introduce divisors or coefficients for CONCAT groups. "
    "COUNT(DISTINCT CONCAT(a, b)) is Shape A: an outer MulGroup with agg_func='count', distinct=true, and a single multiply child whose add_groups[0] is the CONCAT MulGroup (scalar_func='concat'); do not set agg_func and scalar_func='concat' on the same MulGroup (Shape B). "
    "Use only COUNT as the outer aggregation wrapper for a CONCAT MulGroup; SUM, AVG, MIN, and MAX are not valid as that outer aggregation.",
)

INTENT_CRITICAL_RULES: tuple[str, ...] = (
    "Every JSON object uses only keys listed in structural_json_keys for its structural type "
    "(the root object follows RuntimeIntent; nested rows follow their named type; SQL expressions "
    "are always a single string field such as expr or left_expr per structural_json_keys.sql_expression). "
    "Do not emit extra sibling keys at any level.",
    "Qualify every column reference as table.column using names from the schema text and allowed_tables; "
    "never emit bare column names except the bare wNN and cNN registry tokens on select_cols that point at window_registry and case_registry entries. "
    "Qualify columns inside every window_registry.window_spec.partition_by, order_by, and argument, and inside every case_registry case_when branch (condition sides, result, else_result).",
    "Use only tables and columns from the provided schema text and allowed_tables; do not invent identifiers.",
    "Join path discovery, foreign-key traversal, and bridge or junction tables are handled only by the downstream engine after this JSON is parsed; never refuse or shrink the intent because tables look disconnected in the structural payload.",
    "Do not judge whether the question is answerable from schema connectivity; translate the planner prose into IR using only the listed planner tables and their columns.",
    "Grain must match structure: grouped requires group_by_cols and aggregation in select_cols; "
    "row_level means no GROUP BY and no aggregation in select_cols; "
    "scalar is a single aggregated result with no GROUP BY.",
    "Row-level predicates belong in filters_param; predicates on aggregates belong in having_param. "
    "Never put join predicates in filters_param or having_param.",
    "SUM and AVG apply only to numeric measure columns; use COUNT for non-measure columns.",
    "Nested aggregation is forbidden; compute inner aggregates in a CTE step, then aggregate in the main query.",
    "Do not use EXTRACT(EPOCH FROM ...) for time differences; subtract date columns directly or use supported date functions.",
    "CTE output_columns are snake_case alias tokens matching ^[a-z_][a-z0-9_]*$; never qualified table.column, never function call text, never AS clauses; align positionally with select_cols. Reference CTE outputs only via cteN.<output_columns_token> in window_registry, filters_param, having_param, order_by_cols, and select_cols.",
    'Relative date-window filters use value_type date_window with value {"unit", "amount"}; '
    'column-to-column date spans use value_type date_diff with value {"unit", "amount"}. Use singular unit names.',
    "Integer columns with schema role temporal represent day-count durations; compare them to elapsed day expressions "
    "(date subtraction or keyword minus date), not as calendar dates.",
    "BETWEEN uses op between with value [lower, upper]. NULL checks use op is null or is not null without a value field.",
    "filter_group (integer) labels OR-of-AND blocks: predicates sharing a filter_group are joined by AND; "
    "distinct filter_group values are joined by OR. Use bool_op only when every row has filter_group unset "
    "(a single AND chain or a flat AND/OR chain). Do not put bool_op on rows that carry filter_group.",
    "window_registry defines registry_id and window_spec; select_cols reference entries with bare wNN tokens. "
    "Never put a window_spec key on a select_cols entry.",
    "case_registry defines registry_id and case_when with non-empty branches; select_cols reference entries with bare cNN tokens. "
    "When the question asks for conditional labels or buckets over columns, populate case_registry rather than dropping the derived column.",
    "SELECT DISTINCT prefixes the column expr with the bare token DISTINCT and a space "
    "('DISTINCT table.column'). Use COUNT(DISTINCT table.column) for distinct counts; "
    "for COUNT(DISTINCT CONCAT(table.column, other_table.other_column)) emit Shape A: COUNT MulGroup with distinct=true whose single multiply child is a CONCAT MulGroup (scalar_func='concat'); "
    "do not wrap DISTINCT around arbitrary expressions except as COUNT(DISTINCT ...). "
    "Do not embed COUNT(*) inside arithmetic subexpressions—use COUNT(*) only as a top-level aggregate where appropriate.",
    "Arithmetic combines expressions with +, -, *, /; aggregations may wrap arithmetic (SUM(table.column * table.other_column)). "
    "Subtract date columns directly (table.date_column - table.other_date_column) for day differences. "
    "Add or subtract integer day counts from date columns (table.date_column + table.integer_column or table.date_column + 7) for due-date comparisons.",
    "String concatenation uses CONCAT(expr1, ' ', expr2, ...) in expr strings; do not use the SQL || operator (pipe-pipe).",
    "Apply scalar functions such as ROUND after aggregates when needed (ROUND(SUM(table.column), 2)).",
    "Use exact identifiers from the provided schema text; never leave synthetic shape tokens from this prompt "
    "(table, other_table, column, date_column, other_date_column), generic instructional tokens (table_N, column_N), or angle-bracket markup in expressions.",
)

INTENT_PARSE_RULES_APPEND: tuple[str, ...] = (
    "output_format lists every required top-level key; use [] for empty arrays and null for unused scalars.",
    "natural_language is required: describe exactly what the emitted IR returns (tables, columns, filters, aggregates) using real entity names; never include refusal, unavailability, permission, or denial language.",
    (
        "Constructs the intermediate representation cannot emit directly "
        "(set difference, EXISTS / NOT EXISTS, anti-joins, correlated subqueries, lateral joins) "
        "should be reformulated using available primitives such as IS NULL / IS NOT NULL filters on "
        "left-joined sources or conditional aggregations inside GROUP BY."
    ),
    (
        "For (predicate_A OR predicate_B) emit two filters_param objects with different filter_group integers "
        "and no bool_op fields. "
        "For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit four objects with "
        "filter_group values 1, 1, 2, 2 and no bool_op. "
        "Each disjunct gets its own integer filter_group; predicates inside the same disjunct share the same "
        "filter_group. Do not emit bool_op on rows that have a filter_group. "
        "Use qualified column references from the schema for left_expr; bind literals via value placeholders as elsewhere in these rules. "
        "Do not nest filter_group as an array; each filters_param element must include op."
    ),
)

PLANNER_CARDINALITY_RELATIONSHIP_RULE: str = (
    "When table links to other_table through both a direct FK and a junction table, use the direct FK when the "
    "question concerns one other_table value per table row; use the junction when it concerns the set of other_table "
    "values per table row."
)

PLANNER_SHARED_PK_TABLE_SCOPE_RULE: str = (
    "When referenced columns belong to a table that shares its primary key with another table you list, "
    "include every such table whose columns appear in the intent; do not assume same-key columns are "
    "reachable without listing that table."
)

PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE: str = (
    "Do not introduce a CTE for a question answerable by a single SELECT with ordinary joins, filters, or "
    "a window function on the primary table. Use CTEs only for reuse across multiple SELECT bodies, staged "
    "aggregation, self-references, or per-entity ranking."
)

INTENT_FORMAT_REPAIR_JSON_RULES: tuple[str, ...] = (
    "Return JSON only with no prose, markdown fences, or trailing commas.",
    "Preserve intent content while correcting syntax; ensure all required fields are present.",
    "Use [] for empty array fields and null for absent optional scalars.",
)

INTENT_INTERPRET_SYSTEM = (
    "You are the Interpret stage. Your role is to lay out a thinking pathway for answering the data question, not to "
    "author a query structure, join plan, or runtime representation. Output ONLY valid JSON matching "
    "interpret_plan_schema in the user payload. "
    "approach: plain-language steps describing entities to return, row conditions, grouping or per-entity breakdown, "
    "ordering or ranking, row caps, aggregates, conditional labels, and time-window or duration reasoning. Do not use "
    "table.column syntax, SQL, IR tokens, join paths, or set operators. "
    "tables: semantic base tables whose concepts are needed; omit junction tables unless the many-to-many set itself "
    "is the answer. "
    "grounding: traceability only. Record each table in tables, each enum head or value used to resolve business "
    "terms, and any column or enum specifically driving filter, having, or group_by reasoning named in approach. Do "
    "not enumerate select output columns in grounding. Each entry is ref plus used_for. "
    "When schema binding is incomplete, set schema_invalid true as a UI signal only; still complete approach and "
    "tables. Use only names from the payload. Express only computations in supported_capabilities; reformulate "
    "unsupported constructs in plain language."
)

INTENT_GROUND_SYSTEM = (
    "You are the Ground stage. Your role is to convert interpret_plan into logical intent JSON: referenced tables plus "
    "natural-language descriptions of what belongs in each clause field. Output ONLY valid JSON matching "
    "logical_intent_json_schema in the user payload. "
    "interpret_plan is the semantic source; schema_literal_json supplies identifier descriptions, roles, and types. "
    "Follow nl_conventions. Never emit SQL, IR operators, EXISTS, NOT EXISTS, UNION, INTERSECT, EXCEPT, join paths, "
    "or join types. The Compose stage never sees the question or schema payload; your JSON is the sole semantic "
    "contract for literals, table choice, and clause content. "
    "Populate select, filter, group_by, having, order_by, limit, window, and case as natural language using qualified "
    "table.column where needed. Copy every literal into the matching prose field. "
    "Scope preservation: after structural encoding, a table remains in join scope only when a qualified column from "
    "that table appears in clause prose at that scope; name columns in the clause that uses them, not in select when "
    "only needed for join reachability. "
    "Bridge tables: name bridge columns in filter prose when existence or membership is required. "
    "Per-entity breakdown maps to group_by, not row-level DISTINCT. "
    "cte_steps: each step has name, tables, and the same prose fields. tables may list base schema tables and prior "
    "cte_steps names this step reads from. Express only supported_capabilities."
)

INTENT_COMPOSE_SYSTEM = (
    "You are the Compose stage. Your role is to encode logical_intent natural language into runtime intermediate "
    "representation JSON. You do not re-read the question, re-plan semantics, add tables, or author joins. Output ONLY "
    "valid JSON matching output_format. "
    "logical_intent prose fields are the sole semantic source. Map each populated prose field to its IR slot using "
    "logical_to_ir_field_map in the user payload. Use nl_phrase_mappings and only identifiers from "
    "structural_schema_for_chosen_tables. "
    "Translate select, filter, group_by, having, order_by, limit, window, and case prose mechanically. Emit a "
    "qualified column reference in IR for every table.column named in the matching clause prose at that scope. "
    "For cte_steps, set cte_name from name and tables from tables; tables may contain base schema tables and prior "
    "planner step names matching earlier cte_steps names. "
    "Emit only operators in operator_reference, value types in value_type_reference, and constructs in "
    "supported_capabilities. NEVER emit param_key, param_values, or harvested-literal mappings; emit raw_value for "
    "Filter and Having literals; leave select and aggregate aliases empty strings."
)

LOGICAL_DECOMPOSITION_GUIDANCE: tuple[str, ...] = (
    "Describe each prose field thoroughly enough that a structural converter can build the IR without re-reading the question.",
    "Only tables may name real schema tables; every other field is natural language.",
    "Use cte_steps when the question needs a reusable intermediate; each step lists name, tables, and the same prose fields as the top level; tables may name base schema tables and prior step names.",
    "Put window definitions in the window prose field and case definitions in the case prose field; use as <registry_name> only inside those two fields.",
    "Never describe joins as explicit paths; the engine discovers FK paths.",
    "After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit table.column in the appropriate prose field.",
    "Omitting junction_table from tables is correct when its columns appear in prose; the tables list alone never keeps a table in join scope.",
    "When membership or existence requires link_table or junction_table, name junction_table.column or bridge_table.column in select prose, not only join-equality narration in filter prose.",
    "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
    "Existence or membership conditions belong in filter prose with binding columns named.",
    PLANNER_CARDINALITY_RELATIONSHIP_RULE,
    PLANNER_SHARED_PK_TABLE_SCOPE_RULE,
    PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE,
)

FORMAT_STRUCTURAL_GUIDANCE: tuple[str, ...] = (
    "Window semantics belong in window_registry only; map logical_intent.window and each cte_steps[].window prose into WindowRegistryStep rows with wNN ids.",
    "Case semantics belong in case_registry only; map logical_intent.case and each cte_steps[].case prose into CaseRegistryStep rows with cNN ids.",
    "Reference registry outputs from select_cols using window_ref or case_ref tokens alongside other projected columns.",
    "Do not encode row filters as CASE branches; use filters_param for row membership.",
    "Never put AVG, SUM, COUNT, MIN, MAX calls or OVER (...) frames inside filters_param.right_expr as raw_sql. "
    "Compare to aggregates using an extra cte_steps row, a scalar subquery shape allowed by the IR, or window_registry references.",
)

INTENT_ANSWER_STYLE_GUIDANCE: tuple[str, ...] = LOGICAL_DECOMPOSITION_GUIDANCE + FORMAT_STRUCTURAL_GUIDANCE

SQL_AGG_FUNC_CALL_RE = re.compile(
    r"\b(?:count|sum|avg|min|max)\s*\(",
    re.IGNORECASE,
)

INTENT_PLACEHOLDER_ANGLE_RE = re.compile(
    r"<(table_\d+|table\d+|column_\d+|col\d+|date_column|other_date_column|value_from_question|measure_\d+|count_rows)>",
    re.IGNORECASE,
)

EXPR_TABLE_COLUMN_REF_RE = re.compile(r"\w+\.\w+")

REGISTRY_WINDOW_ID_RE = re.compile(r"^w\d{2}$")

REGISTRY_CASE_ID_RE = re.compile(r"^c\d{2}$")

AGG_PREFIXES = frozenset({"COUNT(", "SUM(", "AVG(", "MIN(", "MAX("})

NUMERIC_RESULT_SCALARS = frozenset(
    {
        "abs",
        "round",
        "floor",
        "ceil",
        "extract",
        "date_part",
        "year",
        "month",
        "day",
        "length",
    }
)

INTEGER_SCALARS = frozenset({"extract", "date_part", "year", "month", "day", "length"})

NUMERIC_RESULT_AGGS = frozenset({"count", "sum", "avg"})

_NUMERIC_COMPARE_OPS_ORDERED: tuple[str, ...] = ("=", "!=", "<", "<=", ">", ">=")

NUMERIC_RESULT_OPS = frozenset(_NUMERIC_COMPARE_OPS_ORDERED)

ARITHMETIC_ROLES = frozenset({"numeric_measure", "numeric_categorical"})

AGG_QUANTITY_RE = re.compile(
    r"\b(?:more\s+than|greater\s+than|at\s+least|fewer\s+than|less\s+than|"
    r"no\s+more\s+than|no\s+fewer\s+than|over|under|exceeding|"
    r"above|below|a\s+minimum\s+of|a\s+maximum\s+of)\s+\d+\b",
    re.IGNORECASE,
)

COUNT_THRESHOLD_TABLE_RE = re.compile(
    r"\b(?:in\s+(?:exactly\s+)?|exactly\s+)(\d+)\s+(\w+)\b",
    re.IGNORECASE,
)

CTE_FULL_AGGS = list(_AGGREGATION_FUNCTION_NAMES_ORDERED)

CTE_DEFAULT_AGGS = ["count", "min", "max"]

CTE_HAVING_COMPARE_OPS = list(_NUMERIC_COMPARE_OPS_ORDERED)

STRUCTURAL_IDENTITY_VALUES = frozenset({0, 1})

STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE = re.compile(r":(s\d+)\b")

STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)*$")

IN_OPS = frozenset({"in", "not in"})

IN_STRING_SEPARATORS = re.compile(r"['\"]?\s*,\s*['\"]?")

AGG_KEYWORDS_RE = re.compile(r"\b(?:total|count|number\s+of|average|avg|sum|how\s+many)\b", re.IGNORECASE)

AGG_PATTERN = re.compile(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(.+?)\s*\)$", re.IGNORECASE)

TABLE_COL_PATTERN = re.compile(r"(\w+)\.(\w+)")

HAVING_COUNT_VALUES = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]

HAVING_SUM_AVG_VALUES = [10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]

HAVING_MIN_MAX_VALUES = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]

DEFAULT_RANDOM_SEED = 2202

RANGE_OPS = frozenset({">", "<", ">=", "<="})

IMPOSSIBLE_HAVING_RE = re.compile(
    r"^COUNT\b.*",
    re.IGNORECASE,
)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PG_LAST_WINDOW_FRAME_OPTIONS_INLINE_DEFAULT: int = 1058

PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_UNBOUNDED_PAIR: int = 309

PG_LAST_WINDOW_FRAME_OPTIONS_RANGE_UNBOUNDED_CURRENT: int = 1075

PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_OFFSET_CURRENT: int = 3093

QUESTION_NUMERIC_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

QUESTION_YEAR_IN_STRING_RE = re.compile(r"\b(19|20)\d{2}\b")

QUESTION_TOP_N_PHRASE_RE = re.compile(
    r"\b(?:top|first|bottom|last|least|most)\s+\d+\b",
    re.IGNORECASE,
)

QUESTION_DISTINCT_KEYWORD_RE = re.compile(
    r"\b(?:distinct|unique)\b",
    re.IGNORECASE,
)

QUESTION_FOR_EACH_OR_PER_RE = re.compile(
    r"\b(?:for\s+each|per|each)\s+(\w+(?:\s+\w+)?)\b",
    re.IGNORECASE,
)

QUESTION_AGGREGATION_RATE_PREFIX_RE = re.compile(
    r"\b(?:average|avg|sum|total|count|min|max|mean|number\s+of|amount)\b",
    re.IGNORECASE,
)

SHAPE_FORM_NUM_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")

SHAPE_FORM_DATE_REGEX = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
)

SHAPE_FORM_STR_REGEX = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")

SHAPE_QUESTION_INDEX_KEY: str = "shape_question_index"

TEMPLATE_INTENT_KEY_INDEX_KEY: str = "intent_key_index"

TEMPLATE_UNION_FAMILY_INDEX_KEY: str = "union_family_index"

TEMPLATE_QUESTION_TOKEN_INDEX_KEY: str = "question_token_index"

SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PGPASSWORD",
        "POSTGRES_PASSWORD",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "DATABRICKS_TOKEN",
        "DATABRICKS_ACCESS_TOKEN",
        "MYSQL_PASSWORD",
        "MYSQL_PWD",
        "REDSHIFT_PASSWORD",
        "SQLSERVER_PASSWORD",
        "MSSQL_SA_PASSWORD",
        "SQLSERVER_CLIENT_SECRET",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_OAUTH_TOKEN",
        "BIGQUERY_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
    },
)

MYSQL_PROFILING_SAMPLE_PREDICATE = "RAND() < {ratio}"

REDSHIFT_PROFILING_SAMPLE_PREDICATE = "RANDOM() < {ratio}"

DUCKDB_PROFILING_SAMPLE_PREDICATE: str = "USING SAMPLE {pct:.4f} PERCENT (bernoulli)"

SQLITE_PROFILING_SAMPLE_PREDICATE: str = "abs(random()) / 9223372036854775807.0 < {ratio}"

DUCKDB_EXPLAIN_ESTIMATED_CARDINALITY_RE: str = r"(?i)EC[:=]\s*(\d+)"

DUCKDB_EXPLAIN_CARTESIAN_TOKENS: tuple[str, ...] = ("CROSS_PRODUCT", "NESTED_LOOP_JOIN")

SQLITE_EXPLAIN_FULL_SCAN_TOKENS: tuple[str, ...] = ("SCAN TABLE", "SCAN ")

REGISTRY_REF_TOKEN_RE = re.compile(r"^[wc]\d{2}$")

CASE_RESULT_BARE_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CASE_RESULT_REGISTRY_TOKEN_RE = re.compile(r"^[wc]\d{2}$")

REALISM_DROP_REASON_CATEGORIES: frozenset[str] = frozenset(
    {
        "nonsensical_sql",
        "tautology",
        "overly_narrow_filter",
        "sensitive_smell",
        "unmeasurable_metric",
        "other",
    }
)

WINDOW_REGISTRY_RANK_KIND_HINTS: frozenset[str] = frozenset(
    {"row_number", "rank", "dense_rank", "ntile"},
)

WINDOW_REGISTRY_AGG_KIND_HINTS: frozenset[str] = frozenset(
    {
        "sum",
        "avg",
        "mean",
        "count",
        "max",
        "min",
        "stddev_pop",
        "stddev_samp",
        "var_pop",
        "var_samp",
    },
)

WINDOW_REGISTRY_NAV_KIND_HINTS: frozenset[str] = frozenset(
    {"lag", "lead", "first_value", "last_value", "nth_value"},
)

FILTER_VALUE_TYPE_DATE_WINDOW: frozenset[str] = frozenset({"temporal", "date_window"})

FILTER_VALUE_TYPE_DATE_DIFF: frozenset[str] = frozenset({"date_diff"})

SEED_FAILURE_CODE_REALISM_DROPPED: str = "realism_dropped"

WINDOW_FRAME_BOUNDS: frozenset[str] = frozenset(
    {
        "unbounded_preceding",
        "current_row",
        "n_preceding",
        "n_following",
        "unbounded_following",
    }
)

WINDOW_FRAME_BOUND_DEFAULT_START: str = "unbounded_preceding"

WINDOW_FRAME_BOUND_DEFAULT_END: str = "current_row"

SEED_WARMUP_FAILURE_CODES: frozenset[str] = frozenset(
    {
        "join_resolution_failed",
        "sql_build_failed",
        "instantiation_failed",
        "substitution_failed",
        "empty_sql_after_substitution",
        "ast_validate_pglast_syntax",
        "ast_validate_unsupported_construct",
        "ast_validate_missing_from_clause",
        "ast_valiother_date_columnad_identifier",
        "ast_validate_cte_error",
        "ast_validate_other",
        "explain_failed",
        "explain_schema",
        "explain_semantic",
        "explain_transient",
        "execution_failed",
        "question_generation_failed",
        SEED_FAILURE_CODE_REALISM_DROPPED,
        "validation_exception_unexpected",
        "warmup_semantic_precheck",
        "warmup_qualified_refs",
        "warmup_post_processing_lite_failed",
        "warmup_post_binding_semantics",
    }
)

SQL_TO_INTENT_FAILURE_CODES: frozenset[str] = frozenset(
    {
        "SQL_PARSE_FAILED",
        "INTENT_VALIDATION_FAILED",
        "ROUND_TRIP_SHAPE_MISMATCH",
        "ROUND_TRIP_INTENT_DRIFT",
        "ROUND_TRIP_EXECUTE_MISMATCH",
    }
)

SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM: str = "<num>"

SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR: str = "<str>"

SQL_TO_INTENT_LITERAL_PLACEHOLDER_DATE: str = "<date>"

SQL_TO_INTENT_PARAM_KEY_PREFIX: str = "sql_hist_lit_"

SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY: str = "sql_hist_limit_offset"

SEED_WARMUP_DROP_CODES: frozenset[str] = frozenset(
    {
        "warmup_path41_not_allowed",
        "warmup_path42_not_allowed",
        "warmup_would_mutate_store",
        "gold_warmup_blocked_path41_or_42",
        "template_instance_exists",
        "ledger_already_success",
        "question_owned_by_store_template",
        "all_questions_collided_or_empty",
        "question_duplicate_in_warmup_batch",
        "union_agg_or_case_mismatch",
        "pre_execute_absolute_cap",
        "stratum_quota_exceeded",
        "gold_cap_exceeded",
        "gold_stratum_quota_exceeded",
        "global_cap_after_gold",
        "cache_reuse_skipped_unchanged",
        "union_would_exceed_select_cap",
        "union_reconcile_skipped_cap",
        "reconcile_merged_duplicate_template",
        "reconcile_deleted_subset_template",
        "redundant_cover_representative",
    }
)

NAMED_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")

DOLLAR_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\$(\d+)")

PG_NAMED_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s")

DBR_ZERO_ROW_RE: re.Pattern[str] = re.compile(r"\b(?:rows|rowCount|Statistics\(rowCount)\s*[=:]\s*0\b")

NAME_COLUMN_PATTERN: re.Pattern[str] = re.compile(r"(first.?name|last.?name|given.?name|family.?name)", re.IGNORECASE)

CTE_OUTPUT_ALIAS_RE: re.Pattern[str] = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)

CUMULATIVE_PHRASING_RE: re.Pattern[str] = re.compile(
    r"\b(?:running\s+(?:total|sum|count|average|avg|mean)|cumulative(?:\s+(?:total|sum|count|average|avg|mean))?|"
    r"year[\s\-]?to[\s\-]?date|ytd|month[\s\-]?to[\s\-]?date|mtd|quarter[\s\-]?to[\s\-]?date|qtd|"
    r"rolling\s+\d+|moving\s+(?:total|sum|average|avg|mean))\b",
    re.IGNORECASE,
)

SQL_BIND_TOKEN_RE: re.Pattern[str] = re.compile(r"[:@$](p\d+|s\d+)\b")

UNBOUND_PYFORMAT_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)s|(?<!\w)%s(?!\w)")

PG_JOIN_NODE_TYPES: frozenset[str] = frozenset({"Nested Loop", "Hash Join", "Merge Join"})

PG_JOIN_CONDITION_KEYS: tuple[str, ...] = ("Join Filter", "Hash Cond", "Merge Cond")

PG_INNER_CONDITION_KEYS: tuple[str, ...] = ("Index Cond", "Recheck Cond", "Filter")

DBR_CARTESIAN_TOKENS: tuple[str, ...] = ("CartesianProduct", "BroadcastNestedLoopJoin")

INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable "
    "FROM information_schema.columns WHERE table_schema = :s "
    "ORDER BY table_schema, table_name, ordinal_position"
)

INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT kcu.table_schema, kcu.table_name, kcu.column_name "
    "FROM information_schema.table_constraints tc "
    "JOIN information_schema.key_column_usage kcu "
    "  ON tc.constraint_schema = kcu.constraint_schema "
    " AND tc.constraint_name = kcu.constraint_name "
    "WHERE tc.table_schema = :s AND tc.constraint_type = 'UNIQUE' "
    "ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name"
)

REDSHIFT_INFORMATION_SCHEMA_UNIQUE_COLUMNS_SQL: str = (
    "SELECT kcu.table_name, kcu.column_name "
    "FROM information_schema.table_constraints tc "
    "JOIN information_schema.key_column_usage kcu "
    "  ON tc.constraint_schema = kcu.constraint_schema "
    " AND tc.constraint_name = kcu.constraint_name "
    "WHERE tc.table_schema = :s AND tc.constraint_type = 'UNIQUE' "
    "ORDER BY kcu.table_name, kcu.column_name"
)

INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL: str = (
    "SELECT constraint_schema, constraint_name, table_schema, table_name, constraint_type "
    "FROM information_schema.table_constraints "
    "WHERE table_schema = :s AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE') "
    "ORDER BY table_name, constraint_name"
)

INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL: str = (
    "SELECT constraint_schema, constraint_name, table_schema, table_name, column_name, "
    "ordinal_position "
    "FROM information_schema.key_column_usage "
    "WHERE table_schema = :s "
    "ORDER BY constraint_name, ordinal_position"
)

INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL: str = (
    "SELECT constraint_schema, constraint_name, unique_constraint_schema, unique_constraint_name "
    "FROM information_schema.referential_constraints "
    "WHERE constraint_schema = :s"
)

REDSHIFT_SVV_FOREIGN_KEYS_SQL: str = (
    "SELECT constraint_name, table_schema, table_name, column_name, "
    "referenced_table_schema, referenced_table_name, referenced_column_name "
    "FROM svv_foreign_keys "
    "WHERE table_schema = :s "
    "ORDER BY table_name, constraint_name"
)

SQLSERVER_UNIQUE_INDEX_COLUMNS_SQL: str = (
    "SELECT t.name, c.name "
    "FROM sys.indexes i "
    "JOIN sys.index_columns ic "
    "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
    "JOIN sys.columns c "
    "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
    "JOIN sys.tables t ON i.object_id = t.object_id "
    "JOIN sys.schemas s ON t.schema_id = s.schema_id "
    "WHERE s.name = :s AND i.is_unique = 1 AND i.is_primary_key = 0 "
    "ORDER BY t.name, i.name, ic.key_ordinal"
)

UNITY_INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, table_catalog, "
    "table_schema, table_name, constraint_type "
    "FROM `{catalog_esc}`.information_schema.table_constraints "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "AND upper(constraint_type) IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE') "
    "ORDER BY table_name, constraint_name"
)

UNITY_INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, table_catalog, "
    "table_schema, table_name, column_name, ordinal_position "
    "FROM `{catalog_esc}`.information_schema.key_column_usage "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "ORDER BY constraint_name, ordinal_position"
)

UNITY_INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL: str = (
    "SELECT constraint_catalog, constraint_schema, constraint_name, "
    "unique_constraint_catalog, unique_constraint_schema, unique_constraint_name "
    "FROM `{catalog_esc}`.information_schema.referential_constraints "
    "WHERE lower(constraint_schema) = lower('{schema_lit}')"
)

UNITY_INFORMATION_SCHEMA_TABLES_TABLE_TYPE_SQL: str = (
    "SELECT lower(table_name) AS t, table_type "
    "FROM `{catalog_esc}`.information_schema.tables "
    "WHERE lower(table_schema) = lower('{schema_lit}')"
)

UNITY_INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable "
    "FROM `{catalog_esc}`.information_schema.columns "
    "WHERE lower(table_schema) = lower('{schema_lit}') "
    "ORDER BY table_schema, table_name, ordinal_position"
)

UNITY_INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL: str = (
    "SELECT kcu.table_schema, kcu.table_name, kcu.column_name "
    "FROM `{catalog_esc}`.information_schema.table_constraints tc "
    "JOIN `{catalog_esc}`.information_schema.key_column_usage kcu "
    "  ON tc.constraint_schema = kcu.constraint_schema "
    " AND tc.constraint_name = kcu.constraint_name "
    "WHERE lower(tc.table_schema) = lower('{schema_lit}') "
    "  AND upper(tc.constraint_type) = 'UNIQUE' "
    "ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name"
)

MYSQL_QUERY_LOG_AVAILABILITY_SQL: str = (
    "SELECT 1 FROM performance_schema.setup_consumers "
    "WHERE NAME = 'events_statements_history' AND ENABLED = 'YES' LIMIT 1"
)

MYSQL_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT DIGEST_TEXT AS sql_text "
    "FROM performance_schema.events_statements_history "
    "WHERE TIMER_START >= (SELECT MAX(TIMER_START) FROM performance_schema.events_statements_history) "
    "  - :lookback_microseconds "
    "ORDER BY TIMER_START DESC LIMIT :max_queries"
)

REDSHIFT_QUERY_LOG_AVAILABILITY_SQL: str = "SELECT 1 FROM stl_query LIMIT 1"

REDSHIFT_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT querytxt AS sql_text FROM svl_qlog "
    "WHERE starttime >= DATEADD(day, -:lookback_days, GETDATE()) "
    "ORDER BY starttime DESC LIMIT :max_queries"
)

SQLSERVER_QUERY_LOG_AVAILABILITY_SQL: str = "SELECT 1 FROM sys.dm_exec_query_stats WHERE 1=0"

SQLSERVER_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT SUBSTRING(st.text, (qs.statement_start_offset/2)+1, "
    "((CASE qs.statement_end_offset WHEN -1 THEN DATALENGTH(st.text) "
    "ELSE qs.statement_end_offset END - qs.statement_start_offset)/2)+1) AS sql_text "
    "FROM sys.dm_exec_query_stats qs "
    "CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st "
    "ORDER BY qs.last_execution_time DESC"
)

SQLSERVER_QUERY_STORE_AVAILABILITY_SQL: str = "SELECT 1 FROM sys.database_query_store_options WHERE desired_state = 2"

SQLSERVER_QUERY_STORE_FETCH_SQL: str = (
    "SELECT DISTINCT qt.query_sql_text AS sql_text "
    "FROM sys.query_store_query q "
    "JOIN sys.query_store_query_text qt ON q.query_text_id = qt.query_text_id "
    "JOIN sys.query_store_plan p ON q.query_id = p.query_id "
    "JOIN sys.query_store_runtime_stats rs ON p.plan_id = rs.plan_id "
    "WHERE rs.last_execution_time >= DATEADD(day, -:lookback_days, GETDATE()) "
    "ORDER BY rs.last_execution_time DESC"
)

MYSQL_PARTITION_EXPRESSIONS_SQL: str = (
    "SELECT TABLE_NAME, PARTITION_EXPRESSION, PARTITION_METHOD "
    "FROM information_schema.PARTITIONS "
    "WHERE TABLE_SCHEMA = :s AND PARTITION_NAME IS NOT NULL "
    "GROUP BY TABLE_NAME, PARTITION_EXPRESSION, PARTITION_METHOD"
)

POSTGRESQL_PARTITION_KEY_COLUMNS_SQL: str = (
    "SELECT c.relname AS table_name, a.attname AS column_name "
    "FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "JOIN pg_partitioned_table pt ON pt.partrelid = c.oid "
    "JOIN LATERAL unnest(pt.partattrs) WITH ORDINALITY AS u(attnum, ord) ON true "
    "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = u.attnum "
    "WHERE n.nspname = :s AND c.relkind = 'p' "
    "ORDER BY c.relname, u.ord"
)

SQLSERVER_PARTITION_KEY_COLUMNS_SQL: str = (
    "SELECT DISTINCT t.name AS table_name, c.name AS column_name, pp.parameter_id "
    "FROM sys.tables t "
    "JOIN sys.schemas s ON t.schema_id = s.schema_id "
    "JOIN sys.indexes i ON t.object_id = i.object_id AND i.index_id IN (0, 1) "
    "JOIN sys.partition_schemes ps ON i.data_space_id = ps.data_space_id "
    "JOIN sys.partition_functions pf ON ps.function_id = pf.function_id "
    "JOIN sys.partition_parameters pp ON pf.function_id = pp.function_id "
    "JOIN sys.columns c ON c.object_id = t.object_id AND c.column_id = pp.parameter_id "
    "WHERE s.name = :s "
    "ORDER BY t.name, pp.parameter_id"
)

MYSQL_INDEX_STATISTICS_SQL: str = (
    "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.STATISTICS "
    "WHERE TABLE_SCHEMA = :s ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX"
)

SNOWFLAKE_QUERY_LOG_AVAILABILITY_SQL: str = (
    "SELECT 1 FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(RESULT_LIMIT => 1)) LIMIT 1"
)

SNOWFLAKE_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT QUERY_TEXT AS sql_text "
    "FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY("
    "END_TIME_RANGE_START => DATEADD(day, -:lookback_days, CURRENT_TIMESTAMP()), "
    "RESULT_LIMIT => :max_queries)) "
    "WHERE EXECUTION_STATUS = 'SUCCESS' ORDER BY START_TIME DESC"
)

BIGQUERY_QUERY_LOG_AVAILABILITY_SQL: str = "SELECT 1 FROM `{project}`.INFORMATION_SCHEMA.JOBS LIMIT 1"

BIGQUERY_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT query AS sql_text "
    "FROM `{project}`.INFORMATION_SCHEMA.JOBS "
    "WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL :lookback_days DAY) "
    "AND state = 'DONE' "
    "ORDER BY creation_time DESC LIMIT :max_queries"
)

EXPLAIN_PERMISSION_DENIED_PATTERNS: tuple[str, ...] = (
    "permission denied",
    "insufficient privilege",
    "access denied",
    "not authorized",
    "does not have permission",
    "does not have access",
    "operation not permitted",
    "undefinedtable",
    "42501",
    "42p01",
)

SQLGLOT_DIALECT_BY_ENGINE: dict[str, str] = {}

AGGREGATE_FUNCTION_NAMES: frozenset[str] = frozenset({"sum", "count", "avg", "min", "max", "stddev", "variance"})

PG_AGG_FUNCNAMES: frozenset[str] = frozenset(
    {
        "sum",
        "count",
        "avg",
        "min",
        "max",
        "stddev",
        "variance",
        "array_agg",
        "string_agg",
        "bool_and",
        "bool_or",
    }
)

VALID_COLUMN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"description", "sensitivity", "role", "boolean_truth_value", "usable"},
)

VALID_TABLE_OVERRIDE_KEYS: frozenset[str] = frozenset({"description", "role", "columns"})

VALID_TOP_LEVEL_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "version",
        "tables",
        "foreign_keys_add",
        "foreign_keys_remove",
        "primary_keys_add",
        "primary_keys_remove",
        "_readonly",
    }
)

MAX_REPAIR_ATTEMPTS_PER_CODE: int = 1

DIAGNOSTIC_FUZZY_CUTOFF: float = 0.6
