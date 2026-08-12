"""Shared package constants: versions, vocab, policy, regex, and structural tokens."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from sqlglot import exp

SESSION_PERSISTENCE_FORMAT_VERSION: str = "0.2.3"

SUSPEND_STATE_FORMAT_VERSION: str = "0.2.3"

ARTIFACT_FORMAT_VERSION: str = "0.2.3"

KNOWLEDGE_EXPORT_FORMAT_VERSION: str = "0.2.3"

META_ANSWER_FORMAT_VERSION: str = "0.2.3"

NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION: str = "0.2.3"

MIN_COMPATIBLE_PACKAGE_VERSION: str = "0.2.3"

STRUCTURE_DOCUMENT_VERSION: str = "0.2.3"

SCHEMA_CONTEXT_CACHE_VERSION: str = "0.2.3"

SCHEMA_JOIN_PATH_ENUMERATION_VERSION: str = "0.2.3"

AETHERSPACE_ARTIFACT_VERSION: str = "0.2.3"

QUESTION_NORMALIZATION_VERSION: str = "0.2.3"

QUESTION_NORMALIZATION_VERSION_KEY: str = "question_normalization_version"

TEMPLATE_STORE_FORMAT_VERSION: str = "0.2.3"

FEDERATION_ARTIFACT_FORMAT_VERSION: str = "0.2.3"

FEDERATION_MAPPINGS_VERSION: str = "0.2.3"

FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION: str = "0.2.3"

FEDERATION_DECLARATION_VERSION: str = "0.2.3"

ENGINE_MODULE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "sqlite": "aetherdialect._dialect_sqlglot_engines",
        "duckdb": "aetherdialect._dialect_sqlglot_engines",
        "csv": "aetherdialect._dialect_sqlglot_engines",
        "mysql": "aetherdialect._dialect_sqlglot_engines",
        "mariadb": "aetherdialect._dialect_sqlglot_engines",
        "sqlserver": "aetherdialect._dialect_sqlglot_engines",
        "postgresql": "aetherdialect._dialect_postgres",
        "redshift": "aetherdialect._dialect_sqlglot_engines",
        "databricks": "aetherdialect._dialect_sqlglot_engines",
        "snowflake": "aetherdialect._dialect_sqlglot_engines",
        "bigquery": "aetherdialect._dialect_sqlglot_engines",
        "oracle": "aetherdialect._dialect_sqlglot_engines",
    }
)

SUPPORTED_ENGINES: frozenset[str] = frozenset()

CLASS_DELEGATED_METHODS: frozenset[str] = frozenset(
    {
        "db_url",
        "connect_args",
        "connection_slug_fields",
        "sqlalchemy_url",
        "validate",
        "has_native_connection",
        "has_iam_credentials",
        "has_password_auth",
        "has_keypair_auth",
        "has_oauth_auth",
        "has_sql_auth",
        "has_windows_auth",
        "has_aad_password_auth",
        "has_aad_service_principal_auth",
        "has_service_account",
        "has_wallet_auth",
        "has_token_auth",
        "ensure_driver_mode",
        "resolve_source_files",
        "set_source_selections",
        "apply_connection_credentials",
    }
)

ENGINE_DRIVER_REQUIREMENTS: Mapping[str, tuple[str | tuple[str, ...], str | tuple[str, ...], str]] = MappingProxyType(
    {
        "postgresql": ("psycopg", "psycopg", "postgresql"),
        "mysql": ("mysql.connector", "mysql-connector-python", "mysql"),
        "mariadb": ("mysql.connector", "mysql-connector-python", "mysql"),
        "duckdb": ("duckdb", "duckdb", "duckdb"),
        "csv": ("openpyxl", "openpyxl", "csv"),
        "sqlite": ("sqlite3", "sqlite", "sqlite"),
        "sqlserver": ("pyodbc", "pyodbc", "sqlserver"),
        "snowflake": ("snowflake.connector", "snowflake-connector-python", "snowflake"),
        "bigquery": ("google.cloud.bigquery", "google-cloud-bigquery", "bigquery"),
        "databricks": ("databricks", "databricks-sql-connector", "databricks"),
        "redshift": ("redshift_connector", "redshift-connector", "redshift"),
        "oracle": ("oracledb", "oracledb", "oracle"),
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
    "oracle",
)

FILE_ENGINE_NAMES: frozenset[str] = frozenset({"csv"})

NATIVE_BACKEND_ENGINES: frozenset[str] = frozenset(
    {
        "databricks",
        "snowflake",
        "bigquery",
        "sqlserver",
        "oracle",
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
        "postgresql",
        "mysql",
        "mariadb",
        "sqlserver",
        "oracle",
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
        "oracle",
    }
)

SQLGLOT_DIALECT_BY_ENGINE: dict[str, str] = {}

STATISTICAL_AGG_EXCLUDED_ENGINES: frozenset[str] = frozenset({"sqlite", "csv"})

WINDOW_FRAMES_EXCLUDED_ENGINES: frozenset[str] = frozenset({"csv"})

ARRAY_CONTAINS_EXCLUDED_ENGINES: frozenset[str] = frozenset({"csv"})

ENGINE_STORAGE_PLACEHOLDER_DIR: str = ".aetherdialect/__placeholder__"

ARTIFACT_DIRECTORY_SEGMENT: str = "aetherdialect"

ENGINE_STORAGE_SLUG_MAX_CHARS: int = 180

TEMPLATE_STORE_PARTITION_PREFIX: str = "partition_"

TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX: str = "partition_"

FEDERATION_STORAGE_SLUG_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^0-9A-Za-z]+")

META_ANSWERS_FILENAME: str = "meta_answers.json"

ARTIFACT_MANIFEST_FILENAME: str = "artifact_manifest.json"

RENAME_HISTORY_FILENAME: str = "rename_history.json.gz"

ARTIFACT_LOCK_FILENAME: str = ".aetherdialect_engine.lock"

UPLOAD_STORE_FILENAME: str = "upload_store.duckdb"

STRUCTURE_DOCUMENT_FILENAME: str = "schema_structure.json"

DOMAIN_KNOWLEDGE_FILENAME: str = "domain_knowledge.json"

STRUCTURAL_KNOWLEDGE_FILENAME: str = "structural_knowledge.json"

KNOWLEDGE_EXTRACTION_PROPOSAL_FILENAME: str = "knowledge_extraction_proposal.json"

KNOWLEDGE_OPERATOR_REPORT_FILENAME: str = "knowledge_operator_report.json"

MIGRATION_MAP_FILENAME: str = "schema_migration_map.json"

WRITE_QUEUE_FILENAME: str = "write_queue.jsonl"

STRUCTURE_APPLIED_SUFFIX: str = ".applied.json"

STRUCTURE_SIDECAR_FILENAME: str = "applied_structure.json"

AETHERSPACE_NEXT_ID_FILENAME: str = "next_id.json"

TEMPLATE_STORE_HEADER_FILENAME: str = "header.json.gz"

TEMPLATE_STORE_GLOBAL_NEXT_ID_FILENAME: str = "next_id.json"

ANTI_JOIN_PRESENCE_COLUMN_SUFFIX: str = "__present"

FEDERATION_MANIFEST_FILENAME: str = "federation_manifest.json"

FEDERATION_MAPPINGS_FILENAME: str = "federation_mappings.json"

FEDERATION_MAPPINGS_APPLIED_FILENAME: str = "applied_federation_mappings.json"

FEDERATION_COMPOSITE_SCHEMA_FILENAME: str = "composite_schema_graph.json.gz"

FEDERATION_MIGRATION_MAP_FILENAME: str = "federation_migration_map.json"

FEDERATION_PLAN_TEMPLATE_FILENAME: str = "federation_plan_templates.json"

FEDERATION_MAPPING_SUGGESTIONS_CACHE_FILENAME: str = "federation_mapping_suggestions_cache.json"

FEDERATION_MEMBER_MANIFEST_FILENAME: str = "federation_member_manifest.json"

FEDERATION_DECLARATION_FILENAME: str = "federation_declaration.json"

ARTIFACT_DIR_MODE: int = 0o700

ARTIFACT_FILE_MODE: int = 0o600

ARTIFACT_LOCK_TIMEOUT_SECONDS: float = 30.0

ARTIFACT_LOCK_POLL_INTERVAL_SECONDS: float = 0.05

DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK: str = "STALE_ARTIFACT_LOCK"

DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING: str = "DATA_QUALITY_BLOCKING"

ORPHAN_RETENTION_SECONDS: int = 7 * 24 * 3600

APPLIED_MAP_ARCHIVE_RETENTION_COUNT: int = 3

DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL: str = "ARTIFACTS_DIR_NOT_LOCAL"

DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT: str = "WRITE_QUEUE_CORRUPT"

DIAGNOSTIC_CODE_WRITE_QUEUE_FULL: str = "WRITE_QUEUE_FULL"

DIAGNOSTIC_CODE_REUSE_HIT: str = "REUSE_HIT"

DIAGNOSTIC_CODE_REUSE_MISS: str = "REUSE_MISS"

DIAGNOSTIC_CODE_LARGE_RESULT_WARNING: str = "LARGE_RESULT_WARNING"

DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT: str = "SENSITIVITY_GATE_HIT"

DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY: str = "INTERPRET_GROUND_RETRY"

DIAGNOSTIC_CODE_COMPOSE_REPAIR: str = "COMPOSE_REPAIR"

DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART: str = "FALLBACK_FRESH_RESTART"

DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED: str = "CONFIG_FILE_VALUE_APPLIED"

DIAGNOSTIC_CODE_ENGINE_INFO: str = "ENGINE_INFO"

DIAGNOSTIC_CODE_LLM_TURN_COST: str = "LLM_TURN_COST"

DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP: str = "STRUCTURE_EDIT_SKIP"

DIAGNOSTIC_CODE_STRUCTURE_NEEDS_RECONFIRMATION: str = "STRUCTURE_NEEDS_RECONFIRMATION"

DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT: str = "PK_INFERENCE_PROMPT"

DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION: str = "ZERO_ROW_WHERE_SUGGESTION"

DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED: str = "ZERO_ROW_WHERE_AUTO_FIXED"

DIAGNOSTIC_CODE_DATA_QUALITY_ADVISORY: str = "DATA_QUALITY_ADVISORY"

DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ: str = "DATA_QUALITY_AUTO_READ"

DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED: str = "DATA_QUALITY_AUTO_CORRECTED"

DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED: str = "UPLOAD_UNIT_AFFIX_STRIPPED"

DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED: str = "UPLOAD_TRANSFORM_REJECTED"

DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_APPLIED: str = "UPLOAD_TRANSFORM_APPLIED"

DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED: str = "COMPOSITE_DESCRIPTIVE_PROFILE_FAILED"

DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED: str = "COLUMN_PROFILE_FAILED"

DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH: str = "COLUMN_CHARSET_MISMATCH"

DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER: str = "MATERIALIZED_VIEW_ANSWER"

DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED: str = "PROFILE_TABLE_CLONE_FAILED"

DIAGNOSTIC_CODE_TEMPLATE_STORE_ORPHANED: str = "TEMPLATE_STORE_ORPHANED"

DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED: str = "TEMPLATE_REMAP_DIVERGED"

DIAGNOSTIC_CODE_MIGRATION_CHECKPOINT_ORPHANED: str = "MIGRATION_CHECKPOINT_ORPHANED"

DIAGNOSTIC_CODE_ARTIFACT_GROWTH: str = "ARTIFACT_GROWTH"

DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR: str = "ARTIFACT_LIMIT_NEAR"

DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH: str = "JOIN_ORPHAN_RATE_HIGH"

DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY: str = "JOIN_NULLABLE_KEY"

DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED: str = "JOIN_PATH_TIE_CEILING_EXCEEDED"

DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP: str = "JOIN_CANDIDATE_CAP"

DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE: str = "SEMANTIC_PROFILE_WHERE_EDGE"

DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED: str = "REDUNDANT_JOIN_WHERE_DROPPED"

DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_ELIMINATED: str = "REDUNDANT_KEY_JOIN_ELIMINATED"

DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_CAP_REACHED: str = "REDUNDANT_KEY_JOIN_CAP_REACHED"

DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR: str = "COMPARISON_JOIN_DETOUR"

DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED: str = "ENUM_PROMPT_TRUNCATED"

DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED: str = "DESCRIPTION_ENRICHMENT_FAILED"

DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP: str = "DESCRIPTION_ENRICHMENT_NOOP"

DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT: str = "SCHEMA_FK_CATALOG_ABSENT"

DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED: str = "SCHEMA_ROLE_TYPE_COERCED"

DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE: str = "SCHEMA_UNKNOWN_TYPE_UNUSABLE"

DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE: str = "FEDERATION_INELIGIBLE"

DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE: str = "FEDERATION_PARTIAL_FAILURE"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED: str = "FEDERATION_MEMBER_FAILED"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED: str = "FEDERATION_MEMBER_PROBE_FAILED"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_GENERATED: str = "FEDERATION_MEMBER_GENERATED"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED: str = "FEDERATION_MEMBER_EXECUTED"

DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_EXECUTED: str = "FEDERATION_COORDINATOR_EXECUTED"

DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED: str = "FEDERATION_SOURCES_QUERIED"

DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT: str = "FEDERATION_MAPPING_DRIFT"

DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED: str = "FEDERATION_SEMIJOIN_SKIPPED"

DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS: str = "FEDERATION_REDUCTION_NULL_KEYS"

DIAGNOSTIC_CODE_FEDERATION_JOIN_CANDIDATE_CAP: str = "FEDERATION_JOIN_CANDIDATE_CAP"

DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY: str = "FEDERATION_PLAN_REPLAY"

DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED: str = "FEDERATION_CAP_EXCEEDED"

DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED: str = "FEDERATION_TURN_CANCELLED"

DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED: str = "CANCEL_NOT_SUPPORTED"

DIAGNOSTIC_CODE_SQL_PARSE_FAILED: str = "SQL_PARSE_FAILED"

DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER: str = "FEDERATION_MALFORMED_MEMBER_ANSWER"

DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT: str = "FEDERATION_JOIN_FAN_OUT"

DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_DECIMAL_FALLBACK: str = "FEDERATION_COORDINATOR_DECIMAL_FALLBACK"

DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK: str = "FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK"

DIAGNOSTIC_CODE_FEDERATION_TIME_ANCHOR: str = "FEDERATION_TIME_ANCHOR"

DIAGNOSTIC_CODE_FEDERATION_TIMESTAMP_NORMALISED: str = "FEDERATION_TIMESTAMP_NORMALISED"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH: str = "FEDERATION_MEMBER_TIMEZONE_MISMATCH"

DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED: str = "FEDERATION_MEMBER_REMOVED"

DIAGNOSTIC_CODE_FEDERATION_POOL_UNDERSIZED: str = "FEDERATION_POOL_UNDERSIZED"

DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED: str = "MEMBER_LIMIT_NARROWED"

DIAGNOSTIC_CODE_COORDINATOR_LIMITS: str = "COORDINATOR_LIMITS"

DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED: str = "ROUNDING_MODE_MIXED"

DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE: str = "REFUSAL_JOIN_PATH_UNAVAILABLE"

DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT: str = "REFUSAL_AGGREGATE_FAN_OUT"

DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING: str = "REFUSAL_HOP_CEILING"

DIAGNOSTIC_CODE_REFUSAL_CTE_CAP: str = "REFUSAL_CTE_CAP"

DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP: str = "REFUSAL_CAPABILITY_GAP"

DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST: str = "REFUSAL_NULL_IN_NEGATED_LIST"

DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR: str = "REFUSAL_OPAQUE_EXPR"

DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN: str = "REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN"

DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL: str = "REFUSAL_AMBIGUOUS_DATE_LITERAL"

DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING: str = "REFUSAL_UNION_COLUMN_MISSING"

DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE: str = "REFUSAL_UNSUPPORTED_COLUMN_TYPE"

DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT: str = "REFUSAL_NOT_AVAILABLE_IN_CONTEXT"

DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED: str = "REFUSAL_PERMISSION_DENIED"

DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION: str = "REFUSAL_SCOPE_VIOLATION"

DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION: str = "REFUSAL_INVALID_QUESTION"

DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY: str = "REFUSAL_CONVERSATIONAL_DENY"

DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION: str = "REFUSAL_UNMAPPABLE_QUESTION"

DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED: str = "REFUSAL_OPERATION_NOT_SUPPORTED"

DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE: str = "REFUSAL_INSUFFICIENT_KNOWLEDGE"

DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE: str = "REFUSAL_PARSE_FAILURE"

DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA: str = "REFUSAL_DECLINED_SCHEMA"

DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP: str = "REFUSAL_JOIN_PATH_TIE_CAP"

DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET: str = "REFUSAL_CLAUSE_WIDENED_ROWSET"

DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT: str = "REFUSAL_PROBE_CTE_PLACEMENT"

AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD: str = "write_queue_feedback_record"

AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT: str = "write_queue_template_reject"

AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT: str = "write_queue_template_accept"

AUDIT_EVENT_WRITE_QUEUE_STRUCTURE_PROPOSAL: str = "write_queue_structure_proposal"

AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER: str = "federation_semijoin_key_transfer"

AUDIT_EVENT_SQL_EXECUTION: str = "sql_execution"

AUDIT_EVENT_ASK_SUSPEND: str = "ask_suspend"

AUDIT_EVENT_ASK_CANCELLED: str = "ask_cancelled"

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

SESSION_KIND_AWAITING_REUSE_CONFIRM: str = "awaiting_reuse_confirm"

SESSION_KIND_AWAITING_SQL_CONFIRM: str = "awaiting_sql_confirm"

SESSION_KIND_AWAITING_SQL_FEEDBACK: str = "awaiting_sql_feedback"

SESSION_KIND_EXECUTE: str = "execute"

SESSION_KIND_RESULT: str = "result"

SESSION_KIND_META: str = "meta"

SESSION_KIND_ERROR: str = "error"

YES_NO_SESSION_KINDS: frozenset[str] = frozenset(
    {
        SESSION_KIND_AWAITING_INTENT_CONFIRM,
        SESSION_KIND_AWAITING_REUSE_CONFIRM,
        SESSION_KIND_AWAITING_SQL_CONFIRM,
        SESSION_KIND_EXECUTE,
    }
)

SUSPEND_ID_TO_SESSION_KIND: Mapping[str, str] = MappingProxyType(
    {
        PIPELINE_SUSPEND_ID_DIRECT_REUSE: SESSION_KIND_AWAITING_REUSE_CONFIRM,
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM: SESSION_KIND_AWAITING_INTENT_CONFIRM,
        PIPELINE_SUSPEND_ID_INTENT_FEEDBACK: SESSION_KIND_AWAITING_INTENT_FEEDBACK,
        PIPELINE_SUSPEND_ID_EXECUTE: SESSION_KIND_EXECUTE,
        PIPELINE_SUSPEND_ID_SQL: SESSION_KIND_AWAITING_SQL_CONFIRM,
        PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT: SESSION_KIND_AWAITING_SQL_FEEDBACK,
    }
)

FAILURE_TRACE_ROTATE_BYTES: int = 8388608

SCHEMA_CLASSIFY_ERROR_DETAIL_CAP: int = 50

NORMALIZATION_JACCARD_FLOOR: float = 0.4

TRUST_FLOOR: int = 1

TRUST_CEILING: int = 2

TRUST_AUTO_ACCEPT_THRESHOLD: int = 1

WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS: float = 30.0

WRITE_QUEUE_MAX_BYTES_PER_DRAIN: int = 4194304

STRUCTURE_MAX_DESCRIPTION_CHARS: int = 2000

MAX_NON_AGG_COL_DIFF: int = 2

WARMUP_ROUND_TRIP_LIMIT: int = 100

UPLOAD_SAMPLE_MAX_ROWS: int = 5

UPLOAD_INTERPRET_MAX_ROWS: int = 25

UPLOAD_BAND_VALUE_MAP_MAX_DISTINCT: int = 25

TEMPLATE_STORE_PARTITION_COUNT: int = 256

TEMPLATE_VALUE_HISTORY_MAX_ROWS: int = 64

BOOLEAN_ANTONYM_MIN_STEM_LEN: int = 3

LIMIT_ADD: str = "LIMIT_ADD"

LIMIT_REMOVE: str = "LIMIT_REMOVE"

CTE_SCALAR_THRESHOLD: str = "CTE_SCALAR_THRESHOLD"

COUNT_DISTINCT_ADD: str = "COUNT_DISTINCT_ADD"

SNOWFLAKE_ENV_ACCOUNT: tuple[str, ...] = ("SNOWFLAKE_ACCOUNT", "SNOWSQL_ACCOUNT", "SF_ACCOUNT")

MYSQL_NO_BACKSLASH_ESCAPES_SQL_MODE_TOKEN: str = "NO_BACKSLASH_ESCAPES"

LIKE_ESCAPE_CHAR: str = "/"

HAVING_COUNT_VALUES: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20, 25, 50, 100)

HAVING_MIN_MAX_VALUES: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0)

SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY: str = "sql_hist_limit_offset"

MAX_REPAIR_ATTEMPTS_PER_CODE: int = 1

MAX_PREDICATE_NESTING_DEPTH: int = 3

MAX_PREDICATE_DISTRIBUTE_LEAVES: int = 32

JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR: float = 0.05

JOIN_PATH_TIE_REFUSAL_CEILING: int = 64

ELIMINATE_REDUNDANT_KEY_JOINS_MAX_ITERATIONS: int = 8

KNOWLEDGE_NOTES_EXTRACT_MAX_ATTEMPTS: int = 3

FEDERATION_PLAN_TEMPLATE_FILE_CAP: int = 256

FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP: int = 64

FEDERATION_MAX_JOIN_PATH_TIE_CAP: int = 256

FEDERATION_MAX_JOIN_CANDIDATE_CAP: int = 1024

FEDERATION_ENUM_PROMPT_CAP: int = 10

FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION: int = 38

SCHEMA_ENRICHED_LINES_MAX_CHARS: int = 16384

FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR: float = 0.1

FEDERATION_MAPPING_NAME_SCORE_FLOOR: float = 0.8

FEDERATION_MANIFEST_LIMITS_KEYS: frozenset[str] = frozenset(
    {
        "row_cap",
        "timeout_ms",
        "semijoin_enabled",
        "max_query_cost_rows",
        "max_query_cost_bytes",
        "profile_timeout_ms",
    }
)

MAX_FLOAT_SAFE_INTEGER: int = 9007199254740992

REFUSAL_TIMING_FLOOR_MS: int = 50

REFUSAL_CAPABILITY_GAP_REASON_CODES: frozenset[str] = frozenset(
    {
        "member_capability",
        "semi_join_unsupported",
        "anti_join_unsupported",
        "distinct_on_unsupported",
        "preserve_tables_unsupported",
        "nested_predicate_groups",
    }
)

REFUSAL_CAPABILITY_GAP_REASON_PREFIXES: tuple[str, ...] = (
    "semi_join is not supported",
    "anti_join is not supported",
    "distinct_on is not supported",
    "preserve_tables is not supported",
    "nested predicate groups are not supported",
    "member capability",
)

REFUSAL_CTE_CAP_ISSUE_IDS: frozenset[str] = frozenset(
    {
        "cte_step_count_exceeded",
        "cte_reference_depth_exceeded",
    }
)

UNUSABLE_NULL_RATIO_THRESHOLD: float = 0.99

SENTINEL_MODE_FREQUENCY_THRESHOLD: float = 0.99

EMPTY_JOIN_CANDIDATES: dict[str, Any] = {"candidates": []}

UNKNOWN_VALUE_TYPE: str = "unknown"

JSON_COLUMN_TYPE_TOKENS: frozenset[str] = frozenset({"json", "jsonb"})

JOIN_CHOICE_SCOPE_MAIN: str = "main"

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

AGGREGATION_FUNCTION_NAMES_ORDERED: tuple[str, ...] = (
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "string_agg",
    "stddev",
    "variance",
    "median",
)

VALID_AGGREGATION_FUNCTIONS: frozenset[str] = frozenset(AGGREGATION_FUNCTION_NAMES_ORDERED)

WINDOW_RANKING_FUNCTIONS: frozenset[str] = frozenset(
    {"row_number", "rank", "dense_rank", "ntile", "percent_rank", "cume_dist"}
)

WINDOW_AGG_FUNCTIONS: frozenset[str] = frozenset({"sum", "avg"})

WINDOW_OFFSET_FUNCTIONS: frozenset[str] = frozenset({"lag", "lead"})

WINDOW_VALUE_FUNCTIONS: frozenset[str] = frozenset({"first_value", "last_value", "nth_value"})

VALID_WINDOW_FUNCTIONS: frozenset[str] = (
    WINDOW_RANKING_FUNCTIONS | WINDOW_AGG_FUNCTIONS | WINDOW_OFFSET_FUNCTIONS | WINDOW_VALUE_FUNCTIONS
)

VALID_SENSITIVITY_LEVELS: frozenset[str] = frozenset({"none", "restricted", "hidden"})

VALID_SCALAR_FUNCTIONS: frozenset[str] = frozenset(
    {
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
)

SCALAR_FUNCTIONS_STRING: frozenset[str] = frozenset(
    {
        "upper",
        "lower",
        "trim",
        "ltrim",
        "rtrim",
        "length",
        "concat",
    }
)

SCALAR_FUNCTIONS_VARIADIC: frozenset[str] = frozenset({"concat"})

SCALAR_FUNCTIONS_NUMERIC: frozenset[str] = frozenset({"abs", "round", "floor", "ceil"})

SCALAR_FUNCTIONS_TEMPORAL: frozenset[str] = frozenset(
    {
        "date_trunc",
        "date_part",
        "extract",
        "year",
        "month",
        "day",
    }
)

SCALAR_FUNCTIONS_LEADING_ARG: frozenset[str] = frozenset({"date_trunc", "date_part", "extract"})

VALID_GRAINS: frozenset[str] = frozenset({"scalar", "grouped", "row_level"})

VALID_EXPECTED_ROWS: frozenset[str] = frozenset({"one", "few", "many"})

VALID_HAVING_OPS: frozenset[str] = frozenset({"=", "!=", "<", "<=", ">", ">=", "in", "not in", "between"})

VALID_VALUE_TYPES: frozenset[str] = frozenset(
    {
        "integer",
        "string",
        "date",
        "number",
        "null",
        "boolean",
        "binary",
        "unknown",
        "date_window",
        "date_diff",
    }
)

VALID_RELATIVE_DATE_UNITS: frozenset[str] = frozenset(
    {"day", "week", "month", "quarter", "half_year", "year", "hour", "minute", "second"}
)

SUBDAY_RELATIVE_DATE_UNITS: frozenset[str] = frozenset({"hour", "minute", "second"})

MYSQL_DATE_WINDOW_TRUNC_FORMAT: Mapping[str, str] = MappingProxyType(
    {
        "month": "%Y-%m-01",
        "year": "%Y-01-01",
    }
)

MYSQL_DATE_WINDOW_SUBDAY_TRUNC_FORMAT: Mapping[str, str] = MappingProxyType(
    {
        "hour": "%Y-%m-%d %H:00:00",
        "minute": "%Y-%m-%d %H:%i:00",
        "second": "%Y-%m-%d %H:%i:%s",
    }
)

DATE_UNIT_ALIAS_TO_CANONICAL: Mapping[str, str] = MappingProxyType(
    {
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
)

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

DESCRIPTIVE_ALLOWED_VALUE_TYPES: frozenset[str] = frozenset({"string", "integer"})

DESCRIPTIVE_EXCLUDED_VALUE_TYPES: frozenset[str] = frozenset({"date", "boolean", "number"})

VALID_WHERE_VALUE_TYPES: frozenset[str] = frozenset(
    {
        "categorical",
        "numeric",
        "numeric_categorical",
        "temporal",
        "boolean",
        "null",
    }
)

VALID_HAVING_VALUE_TYPES: frozenset[str] = frozenset({"number", "integer"})

VALUE_TYPE_NORMALIZATION: Mapping[str, str] = MappingProxyType(
    {
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
        "binary": "binary",
        "null": "null",
        "date_window": "date_window",
        "date_diff": "date_diff",
        "unknown": "unknown",
    }
)

_BOOLEAN_WHERE_OPS: frozenset[str] = frozenset({"=", "!=", "in", "not in", "is null", "is not null"})

_CATEGORICAL_WHERE_OPS: frozenset[str] = frozenset(
    {
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
)

_NUMERIC_CATEGORICAL_WHERE_OPS: frozenset[str] = frozenset(
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

_NUMERIC_WHERE_OPS: frozenset[str] = frozenset(
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

CTE_NUMERIC_WHERE_OPS: tuple[str, ...] = tuple(_NUMERIC_WHERE_OPS)

ROLE_ALLOWED_AGGREGATIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "IDENTIFIER": frozenset({"count"}),
        "CATEGORICAL": frozenset({"count", "min", "max", "string_agg"}),
        "NUMERIC_CATEGORICAL": frozenset({"count", "min", "max", "string_agg"}),
        "NUMERIC_MEASURE": frozenset({"count", "sum", "avg", "min", "max", "stddev", "variance", "median"}),
        "TEMPORAL": frozenset({"count", "min", "max", "string_agg"}),
        "BOOLEAN": frozenset({"count"}),
        "FREE_TEXT": frozenset({"count", "string_agg"}),
        "AUDIT": frozenset({"count"}),
    }
)

VALID_WHERE_OPS: frozenset[str] = frozenset(
    _BOOLEAN_WHERE_OPS | _CATEGORICAL_WHERE_OPS | _NUMERIC_CATEGORICAL_WHERE_OPS | frozenset({"contains"})
)

NUMERIC_ONLY_AGGREGATIONS: frozenset[str] = frozenset({"sum", "avg", "stddev", "variance", "median"})

COLUMN_TYPE_TO_VALUE_TYPE: Mapping[str, str] = MappingProxyType(
    {
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
        "bytes": "binary",
        "blob": "binary",
        "bytea": "binary",
        "binary": "binary",
        "varbinary": "binary",
        "image": "binary",
        "timestamp_ntz": "date",
        "timestamp_ltz": "date",
        "timestamp_tz": "date",
        "datetime2": "date",
        "smalldatetime": "date",
        "datetimeoffset": "date",
    }
)

AGGREGATION_ALLOWED_COLUMN_TYPES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "count": ("integer", "string", "date", "number", "boolean"),
        "sum": ("integer", "number"),
        "avg": ("integer", "number"),
        "min": ("integer", "number", "string", "date"),
        "max": ("integer", "number", "string", "date"),
        "string_agg": ("string", "date", "integer", "number"),
        "stddev": ("integer", "number"),
        "variance": ("integer", "number"),
        "median": ("integer", "number"),
    }
)

SELF_JOIN_CTE_NAME_PREFIX: str = "sj_"

NUMERIC_TYPE_TOKENS: frozenset[str] = frozenset(
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

EXACT_NUMERIC_BASE_TYPES: frozenset[str] = frozenset(
    {
        "decimal",
        "numeric",
        "money",
        "int",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "int2",
        "int4",
        "int8",
        "long",
        "short",
        "serial",
        "bigserial",
        "smallserial",
        "byteint",
        "int64",
    }
)

INEXACT_NUMERIC_BASE_TYPES: frozenset[str] = frozenset(
    {
        "float",
        "real",
        "double",
        "double precision",
        "float4",
        "float8",
    }
)

DATE_TYPE_TOKENS: frozenset[str] = frozenset(
    {
        "date",
        "time",
        "timestamp",
        "timestamptz",
        "interval",
    }
)

DATE_FRIENDLY_VALUE_TYPES: frozenset[str] = frozenset({"date", "date_window", "timestamp"})

OP_FLIP: Mapping[str, str] = MappingProxyType({">": "<", "<": ">", ">=": "<=", "<=": ">="})

SCALAR_FUNC_DEFAULTS: Mapping[str, tuple[int | str, ...]] = MappingProxyType(
    {
        "round": (2,),
        "trunc": (0,),
        "truncate": (0,),
        "coalesce": (0,),
        "date_trunc": ("month",),
        "date_part": ("month",),
        "extract": ("year",),
    }
)

DATE_UNIT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("month", "month"),
    ("day", "day"),
    ("week", "week"),
    ("quarter", "quarter"),
    ("year", "year"),
    ("date", "year"),
)

NUMERIC_DATA_TYPES: frozenset[str] = frozenset(
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

SQL_KEYWORDS: frozenset[str] = frozenset(
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
}

JOIN_PATH_EDGE_KINDS: frozenset[str] = frozenset(JOIN_EDGE_KIND_RANK)

JOIN_PATH_EDGE_KIND_WHERE_BUCKET: frozenset[str] = frozenset(
    {
        "semantic_profile",
        "semantic_profile_virtual",
    }
)

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
    "ntile": "NTILE",
    "percent_rank": "PERCENT_RANK",
    "cume_dist": "CUME_DIST",
    "sum": "SUM",
    "avg": "AVG",
    "lag": "LAG",
    "lead": "LEAD",
    "first_value": "FIRST_VALUE",
    "last_value": "LAST_VALUE",
    "nth_value": "NTH_VALUE",
}

WINDOW_IMPORT_FUNC_ALIASES: dict[str, str] = {
    "rownumber": "row_number",
    "denserank": "dense_rank",
    "firstvalue": "first_value",
    "lastvalue": "last_value",
    "nthvalue": "nth_value",
    "percentrank": "percent_rank",
    "cumedist": "cume_dist",
}

WINDOW_NUMERIC_ARG_FUNCTIONS: frozenset[str] = frozenset({"ntile", "nth_value"})

WINDOW_FUNCTIONS_WITHOUT_COLUMN_ARG: frozenset[str] = frozenset(
    {"row_number", "rank", "dense_rank", "ntile", "percent_rank", "cume_dist"}
)

WHERE_ADD: str = "WHERE_ADD"

WHERE_EXPR_ADD: str = "WHERE_EXPR_ADD"

AGG_CHANGE: str = "AGG_CHANGE"

HAVING_VALUE_ADD: str = "HAVING_VALUE_ADD"

HAVING_EXPR_ADD: str = "HAVING_EXPR_ADD"

WHERE_REMOVE: str = "WHERE_REMOVE"

HAVING_REMOVE: str = "HAVING_REMOVE"

JOIN_DIMENSION_ADD: str = "JOIN_DIMENSION_ADD"

JOIN_FACT_ADD: str = "JOIN_FACT_ADD"

TEMP_DATE_TRUNC_GROUPBY: str = "TEMP_DATE_TRUNC_GROUPBY"

TEMP_DATE_WINDOW_WHERE: str = "TEMP_DATE_WINDOW_WHERE"

TEMP_DATE_DIFF_WHERE: str = "TEMP_DATE_DIFF_WHERE"

NUM_ABS_WHERE: str = "NUM_ABS_WHERE"

WHERE_OR_GROUP: str = "WHERE_OR_GROUP"

WINDOW_RANK_ADD: str = "WINDOW_RANK_ADD"

WINDOW_SUM_PARTITION_ADD: str = "WINDOW_SUM_PARTITION_ADD"

WINDOW_LAG_ADD: str = "WINDOW_LAG_ADD"

WINDOW_LEAD_ADD: str = "WINDOW_LEAD_ADD"

WHERE_ILIKE_ADD: str = "WHERE_ILIKE_ADD"

WHERE_ARRAY_CONTAINS_ADD: str = "WHERE_ARRAY_CONTAINS_ADD"

WINDOW_STRIP: str = "WINDOW_STRIP"

CTE_WRAP_GROUPED: str = "CTE_WRAP_GROUPED"

WHERE_IN_LIST_ADD: str = "WHERE_IN_LIST_ADD"

WHERE_NULL_ADD: str = "WHERE_NULL_ADD"

WHERE_NOT_NULL_ADD: str = "WHERE_NOT_NULL_ADD"

HAVING_MATCH_SELECT_AGG: str = "HAVING_MATCH_SELECT_AGG"

WINDOW_DENSE_RANK_ADD: str = "WINDOW_DENSE_RANK_ADD"

WINDOW_RANK_FUNC_ADD: str = "WINDOW_RANK_FUNC_ADD"

WINDOW_AVG_PARTITION_ADD: str = "WINDOW_AVG_PARTITION_ADD"

ORDERBY_WINDOW_COL_ADD: str = "ORDERBY_WINDOW_COL_ADD"

WHERE_LIKE_ADD: str = "WHERE_LIKE_ADD"

SELECT_STRING_SCALAR_ADD: str = "SELECT_STRING_SCALAR_ADD"

TEMP_EXTRACT_WHERE: str = "TEMP_EXTRACT_WHERE"

CTE_UNNEST_ADD: str = "CTE_UNNEST_ADD"

SELF_JOIN_CTE_ADD: str = "SELF_JOIN_CTE_ADD"

MULTI_CTE_CHAIN_ADD: str = "MULTI_CTE_CHAIN_ADD"

SPLICE_HAVING_SUBTREE: str = "SPLICE_HAVING_SUBTREE"

SPLICE_WINDOW_SUBTREE: str = "SPLICE_WINDOW_SUBTREE"

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

INTEGER_VALUE_TYPES: frozenset[str] = frozenset({"int", "integer", "bigint", "smallint", "tinyint", "long", "short"})

STRING_VALUE_TYPES: frozenset[str] = frozenset({"string", "text", "varchar", "char"})

INFERRED_PK_VALUE_TYPES: frozenset[str] = frozenset({"integer", "number", "string"})

VALID_FK_ADD_KEYS: frozenset[str] = frozenset(
    {"from", "to", "kind", "authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

VALID_FK_REMOVE_KEYS: frozenset[str] = frozenset({"from", "to"})

VALID_PK_ADD_KEYS: frozenset[str] = frozenset(
    {"table", "column", "authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

VALID_PK_REMOVE_KEYS: frozenset[str] = frozenset({"table", "column"})

VALID_FK_KINDS: frozenset[str] = frozenset({"structural", "semantic"})

RENTAL_SHOP_BUNDLE_MEMBERS: tuple[str, ...] = (
    "rental_shop.sql",
    "rental_shop_views.sql",
    "rental_shop_notes.txt",
    "fixtures/rental_shop_mock.json",
)

WINDOW_ADD_OPS: frozenset[str] = frozenset(
    {
        WINDOW_RANK_ADD,
        WINDOW_DENSE_RANK_ADD,
        WINDOW_RANK_FUNC_ADD,
        WINDOW_SUM_PARTITION_ADD,
        WINDOW_AVG_PARTITION_ADD,
        WINDOW_LAG_ADD,
        WINDOW_LEAD_ADD,
        ORDERBY_WINDOW_COL_ADD,
    }
)

CTE_ADD_OPS: frozenset[str] = frozenset(
    {
        CTE_WRAP_GROUPED,
        CTE_SCALAR_THRESHOLD,
        CTE_UNNEST_ADD,
        SELF_JOIN_CTE_ADD,
        MULTI_CTE_CHAIN_ADD,
    }
)

HAVING_ADD_OPS: frozenset[str] = frozenset(
    {
        HAVING_VALUE_ADD,
        HAVING_EXPR_ADD,
        HAVING_MATCH_SELECT_AGG,
    }
)

SIMPLE_AGG_NAMES: frozenset[str] = frozenset(
    {"count", "sum", "avg", "min", "max", "string_agg", "stddev", "variance", "median"}
)

ALLOWED_JOIN_KINDS: frozenset[str | None] = frozenset({None, "INNER", "LEFT", "RIGHT", "FULL"})

DEFAULT_WHERE_OP_MAP: dict[str, str] = {
    "=": "=",
    "==": "=",
    "<>": "!=",
    "!=": "!=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "like": "like",
    "not like": "not like",
    "ilike": "ilike",
    "not ilike": "not ilike",
}

LLM_SENSITIVITY_STRIP_KEYS: frozenset[str] = frozenset({"sensitivity"})

AST_AGG_NODE_TO_NAME: dict[type[exp.Expression], str] = {
    exp.Sum: "sum",
    exp.Count: "count",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
    exp.GroupConcat: "string_agg",
    exp.Stddev: "stddev",
    exp.StddevSamp: "stddev",
    exp.StddevPop: "stddev",
    exp.Variance: "variance",
    exp.VariancePop: "variance",
    exp.Median: "median",
}

INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER: str = "date_column"

INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER: str = "other_date_column"

UPLOAD_SCALAR_AFFIX_TOKENS: tuple[str, ...] = (
    "US$",
    "USD",
    "AUD",
    "EUR",
    "GBP",
    "CAD",
    "NZD",
    "CHF",
    "JPY",
    "CNY",
    "€",
    "£",
    "$",
    "%",
)

UPLOAD_SCALAR_AFFIX_TOKENS_SORTED: tuple[str, ...] = tuple(sorted(UPLOAD_SCALAR_AFFIX_TOKENS, key=len, reverse=True))

AGG_PREFIXES: frozenset[str] = frozenset(
    {
        "COUNT(",
        "SUM(",
        "AVG(",
        "MIN(",
        "MAX(",
        "STRING_AGG(",
        "STDDEV(",
        "STDDEV_SAMP(",
        "VARIANCE(",
        "VAR_SAMP(",
        "MEDIAN(",
        "PERCENTILE_CONT(",
    }
)

NUMERIC_RESULT_AGGS: frozenset[str] = frozenset({"count", "sum", "avg", "stddev", "variance", "median"})

NUMERIC_RESULT_SCALARS: frozenset[str] = frozenset(
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

INTEGER_SCALARS: frozenset[str] = frozenset({"extract", "date_part", "year", "month", "day", "length"})

NUMERIC_COMPARE_OPS_ORDERED: tuple[str, ...] = ("=", "!=", "<", "<=", ">", ">=")

NUMERIC_RESULT_OPS: frozenset[str] = frozenset(NUMERIC_COMPARE_OPS_ORDERED)

CTE_FULL_AGGS: tuple[str, ...] = AGGREGATION_FUNCTION_NAMES_ORDERED

CTE_DEFAULT_AGGS: tuple[str, ...] = ("count", "min", "max")

CTE_HAVING_COMPARE_OPS: tuple[str, ...] = NUMERIC_COMPARE_OPS_ORDERED

SQL_STRING_LITERAL_STATEMENT_TERMINATOR: str = ";"

HAVING_SUM_AVG_VALUES: tuple[float, ...] = (10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0)

PG_LAST_WINDOW_FRAME_OPTIONS_INLINE_DEFAULT: int = 1058

PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_UNBOUNDED_PAIR: int = 309

PG_LAST_WINDOW_FRAME_OPTIONS_RANGE_UNBOUNDED_CURRENT: int = 1075

PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_OFFSET_CURRENT: int = 3093

MYSQL_PROFILING_SAMPLE_PREDICATE: str = "RAND({seed}) < {ratio}"

PG_SIMPLE_AGG_NAMES: frozenset[str] = frozenset(
    {"count", "sum", "avg", "min", "max", "stddev", "variance", "median", "string_agg"}
)

WHERE_VALUE_TYPE_DATE_WINDOW: frozenset[str] = frozenset({"temporal", "date_window"})

WHERE_VALUE_TYPE_DATE_DIFF: frozenset[str] = frozenset({"date_diff"})

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

SEED_WARMUP_DROP_CODES: frozenset[str] = frozenset(
    {
        "warmup_union_template_widen_not_allowed",
        "warmup_union_template_and_runtime_widen_not_allowed",
        "warmup_would_mutate_store",
        "gold_warmup_blocked_union_template_widen",
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

PG_JOIN_NODE_TYPES: frozenset[str] = frozenset({"Nested Loop", "Hash Join", "Merge Join"})

PG_JOIN_CONDITION_KEYS: tuple[str, ...] = ("Join Filter", "Hash Cond", "Merge Cond")

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

MYSQL_INDEX_STATISTICS_SQL: str = (
    "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.STATISTICS "
    "WHERE TABLE_SCHEMA = :s ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX"
)

MYSQL_CONNECTION_CHARSET: str = "utf8mb4"

AGGREGATE_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        "sum",
        "count",
        "avg",
        "min",
        "max",
        "string_agg",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "variance",
        "var_pop",
        "var_samp",
        "median",
    }
)

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

DISTINCT_ON_CTE_NAME_PREFIX: str = "don_"

FAN_OUT_SENSITIVE_AGG_FUNCS: frozenset[str] = frozenset({"sum", "avg", "count"})

SQLGLOT_AGG_FUNC_KEY_ALIASES: dict[str, str] = {
    "stddevpop": "stddev",
    "stddevsamp": "stddev",
    "stddev": "stddev",
    "variancepop": "var_pop",
    "variancesamp": "variance",
    "variance": "variance",
    "varsamp": "variance",
    "groupconcat": "string_agg",
    "listagg": "string_agg",
    "median": "median",
}

MYSQL_TIMESTAMP_ENGINES: frozenset[str] = frozenset({"mysql", "mariadb"})

QSIM_QUESTIONS_PATTERN: str = "qsim_questions_v{version}.txt"

REGISTRY_TOKEN_PATTERN: str = r"^[wc]\d{2}$"

ISO_DATE_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ISO_TIMESTAMP_RE: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)

BARE_SCALAR_NUMBER_RE: re.Pattern[str] = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")

YEAR_LITERAL_RE: re.Pattern[str] = re.compile(r"^(19|20)\d{2}$")

NUMERIC_TYPE_ARGUMENTS_RE: re.Pattern[str] = re.compile(r"\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", re.IGNORECASE)

SQL_INTEGER_LITERAL_RE: re.Pattern[str] = re.compile(r"^[+-]?\d+$")

SQL_FIXED_POINT_LITERAL_RE: re.Pattern[str] = re.compile(r"^[+-]?\d+\.\d+$")

SQL_EXPONENT_LITERAL_RE: re.Pattern[str] = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")

RAW_SQL_AGG_OR_WINDOW_RE: re.Pattern[str] = re.compile(
    r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(|OVER\s*\(",
    re.IGNORECASE,
)

CASE_WHEN_QUALIFIED_COLUMN_REF_RE: re.Pattern[str] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$",
)

SQL_AGG_FUNC_CALL_RE: re.Pattern[str] = re.compile(
    r"\b(?:count|sum|avg|min|max)\s*\(",
    re.IGNORECASE,
)

UPLOAD_SCALAR_BAND_PATTERN_STRINGS: tuple[str, ...] = (
    r"(?i)\bto\b",
    r"(?i)\bor\s+more\b",
    r"(?i)\bless\s+than\b",
    r"(?i)\d+\s*[-–—]\s*\d+\s*%",
)

UPLOAD_SCALAR_BAND_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in UPLOAD_SCALAR_BAND_PATTERN_STRINGS
)

INTENT_PLACEHOLDER_ANGLE_RE: re.Pattern[str] = re.compile(
    rf"<(table_\d+|table\d+|column_\d+|col\d+|{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER}|{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}|value_from_question|measure_\d+|count_rows)>",
    re.IGNORECASE,
)

EXPR_TABLE_COLUMN_REF_RE: re.Pattern[str] = re.compile(r"\w+\.\w+")

REGISTRY_WINDOW_ID_RE: re.Pattern[str] = re.compile(r"^w\d{2}$")

REGISTRY_CASE_ID_RE: re.Pattern[str] = re.compile(r"^c\d{2}$")

STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE: re.Pattern[str] = re.compile(r":(s\d+)\b")

STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE: re.Pattern[str] = re.compile(r"^-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)*$")

PRE_QUOTED_IN_LIST_INLINE_RE: re.Pattern[str] = re.compile(r"^'(?:[^']|'')*'(?:,'(?:[^']|'')*')+$")

SQL_STRING_LITERAL_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")

AGG_PATTERN: re.Pattern[str] = re.compile(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(.+?)\s*\)$", re.IGNORECASE)

TABLE_COL_PATTERN: re.Pattern[str] = re.compile(r"(\w+)\.(\w+)")

IMPOSSIBLE_HAVING_RE: re.Pattern[str] = re.compile(
    r"^COUNT\b.*",
    re.IGNORECASE,
)

IDENTIFIER_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RENDERED_SELECT_ALIAS_RE: re.Pattern[str] = re.compile(
    r'\s+AS\s+(?:"([^"]+)"|([A-Za-z_][\w$]*))\s*$',
    re.IGNORECASE,
)

QUESTION_NUMERIC_LITERAL_RE: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?\b")

QUESTION_YEAR_IN_STRING_RE: re.Pattern[str] = re.compile(r"\b(19|20)\d{2}\b")

SHAPE_FORM_NUM_REGEX: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?\b")

SHAPE_FORM_DATE_REGEX: re.Pattern[str] = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
)

SHAPE_FORM_STR_REGEX: re.Pattern[str] = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")

DUCKDB_EXPLAIN_ESTIMATED_CARDINALITY_RE: str = r"(?i)EC[:=]\s*(\d+)"

REGISTRY_REF_TOKEN_RE: re.Pattern[str] = re.compile(r"^[wc]\d{2}$")

CASE_RESULT_BARE_LABEL_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CASE_RESULT_REGISTRY_TOKEN_RE: re.Pattern[str] = re.compile(r"^[wc]\d{2}$")

NAMED_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")

DOLLAR_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\$(\d+)")

DOLLAR_NAMED_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

PG_NAMED_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s")

DBR_ZERO_ROW_RE: re.Pattern[str] = re.compile(r"\b(?:rows|rowCount|Statistics\(rowCount)\s*[=:]\s*0\b")

CTE_OUTPUT_ALIAS_RE: re.Pattern[str] = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)

SQL_BIND_TOKEN_RE: re.Pattern[str] = re.compile(r"[:@$](p\d+|s\d+)\b")

UNBOUND_PYFORMAT_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)s|(?<!\w)%s(?!\w)")

EXPLAIN_PERMISSION_DENIED_PATTERNS: tuple[str, ...] = (
    "permission denied",
    "insufficient privilege",
    "access denied",
    "not authorized",
    "does not have permission",
    "does not have access",
    "operation not permitted",
    "42501",
)

APPLIED_MAP_ARCHIVE_TIMESTAMP_RE: re.Pattern[str] = re.compile(r"\.applied\.\d{8}T\d{6}Z\.json$")

FEDERATION_QUALIFIED_COLUMN_REF_RE: re.Pattern[str] = re.compile(r"^([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)$")

FEDERATION_QUALIFIED_THREE_PART_REF_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)$",
)

FEDERATION_CONNECTION_SLUG_NON_WORD_RE: re.Pattern[str] = re.compile(r"[^\w]+")

ISO_DATE_ONLY_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ISO_DATETIME_RE: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)

STRUCTURE_APPLIED_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"

STRUCTURE_EXPORT_DEFAULT_OWNER: str = "ai"

STRUCTURE_DESCRIPTION_OWNER_STRINGS: frozenset[str] = frozenset(
    {"catalog", "profile", "notes", "llm_refinement", "space_notes", "user_override"},
)

STRUCTURE_ROLE_OWNER_STRINGS: frozenset[str] = frozenset(
    {
        "catalog",
        "profile",
        "llm",
        "boolean_coercion",
        "user_override",
        "pk_fk_coercion",
    },
)

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

DESCRIPTION_OWNER_VALUES: frozenset[str] = frozenset(
    {"catalog", "profile", "notes", "llm_refinement", "space_notes", "user_override"}
)

STRUCTURE_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {"authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

PUBLIC_STRUCTURE_DOCUMENT_KEYS: frozenset[str] = frozenset(
    {
        "table_count",
        "tables",
        "relationships",
        "foreign_keys_add",
        "foreign_keys_remove",
        "primary_keys_add",
        "primary_keys_remove",
        "members",
        "member_count",
    }
)

STRUCTURE_COLUMN_EDITABLE_KEYS: frozenset[str] = frozenset(
    {"sensitivity", "role", "boolean_truth_value", "usable", *STRUCTURE_PROVENANCE_KEYS},
)

STRUCTURE_TABLE_EDIT_KEYS: frozenset[str] = frozenset({"role", "columns", *STRUCTURE_PROVENANCE_KEYS})

STRUCTURE_TOP_LEVEL_EDIT_KEYS: frozenset[str] = frozenset(
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

STRUCTURE_PROSE_KEYS: frozenset[str] = frozenset({"description", "domain_knowledge"})

STRUCTURE_PROSE_REDIRECT_HINT: str = "prose lives in knowledge; pass it to apply_knowledge(space, document) where document carries domain_knowledge / table_descriptions / column_descriptions"

DOMAIN_KNOWLEDGE_DEFAULT_KIND: str = "glossary"

DOMAIN_KNOWLEDGE_ENTRY_KEYS: frozenset[str] = frozenset({"key", "kind", "text", "referenced_entities"})

STRUCTURAL_KNOWLEDGE_FACT_KEYS: frozenset[str] = frozenset({"kind", "text", "referenced_entities", "payload"})

STRUCTURAL_KNOWLEDGE_LEGACY_KINDS: frozenset[str] = frozenset(
    {"relation", "field", "grain", "cardinality", "lifecycle"}
)

STRUCTURAL_KNOWLEDGE_PAYLOAD_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "join": frozenset({"from", "to", "path_signature", "negative"}),
        "declared_value_set": frozenset({"values"}),
        "sentinel_semantics": frozenset({"sentinel_value", "meaning"}),
        "unit_of_measure": frozenset({"unit", "summable"}),
        "relation_shape": frozenset({"shape"}),
        "term_binding": frozenset({"term", "binds_to"}),
        "period_convention": frozenset({"boundary"}),
        "concept_absence": frozenset({"term"}),
    }
)

DOMAIN_KNOWLEDGE_TOP_KEYS: frozenset[str] = frozenset({"entries"})

STRUCTURAL_KNOWLEDGE_TOP_KEYS: frozenset[str] = frozenset({"facts"})

KNOWLEDGE_NOTES_RECORD_KEYS: frozenset[str] = frozenset({"key", "kind", "text", "referenced_entities", "payload"})

KNOWLEDGE_NOTES_TOP_KEYS: frozenset[str] = frozenset({"records", "coverage"})

KNOWLEDGE_NOTES_COVERAGE_ENTRY_KEYS: frozenset[str] = frozenset({"span", "disposition", "record_index"})

FEDERATION_JOIN_FEEDBACK_SEGMENT: str = "feedback"

FEDERATION_JOIN_FEEDBACK_PREFIX: str = "join_"

FEDERATION_STORAGE_PREFIX: str = "fed_"

FEDERATION_SOURCE_STORAGE_PREFIX: str = "fedsrc_"

FEDERATION_TEMPLATES_SEGMENT: str = "federation_templates"

FEDERATION_AVERAGE_SCALE_HEADROOM: int = 6

FEDERATION_COORDINATOR_DECIMAL_FALLBACK_SCALE: int = 9

FEDERATION_COORDINATOR_DECIMAL_FALLBACK: str = (
    f"DECIMAL({FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION}, {FEDERATION_COORDINATOR_DECIMAL_FALLBACK_SCALE})"
)

FEDERATION_MAPPING_NAME_SUBSTRING_SCORE: float = 0.85

FEDERATION_MAPPING_SCORE_NAME_WEIGHT: float = 0.5

FEDERATION_MAPPING_SCORE_OVERLAP_WEIGHT: float = 0.5

FEDERATION_BASE_WHERE_OPS: frozenset[str] = frozenset(
    {
        "=",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "like",
        "not like",
        "in",
        "not in",
        "is null",
        "is not null",
        "between",
    },
)

FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS: frozenset[str] = frozenset(
    {"avg", "count", "sum", "min", "max"},
)

FEDERATION_SENSITIVITY_RANK: dict[str, int] = {
    "none": 0,
    "restricted": 1,
    "hidden": 2,
}

FEDERATION_MANIFEST_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"federation_id", "sources", "table_namespace", "aliases", "cross_source_joins", "coordinator"},
)

FEDERATION_MANIFEST_ALIAS_KEYS: frozenset[str] = frozenset({"source", "table"})

FEDERATION_MANIFEST_SOURCE_KEYS: frozenset[str] = frozenset(
    {"source_id", "engine", "connection", "context", "role", "limits", "session_timezone"},
)

FEDERATION_MANIFEST_JOIN_KEYS: frozenset[str] = frozenset({"left", "right", "kind", "logical_key"})

FEDERATION_CROSS_SOURCE_JOIN_KINDS: frozenset[str] = frozenset({"inner", "left"})

FEDERATION_COMBINE_SEMI_KIND: str = "semi"

FEDERATION_MANIFEST_COORDINATOR_KEYS: frozenset[str] = frozenset(
    {
        "row_cap",
        "default_source_row_cap",
        "default_source_timeout_ms",
        "coordinator_timeout_ms",
        "plan_timeout_ms",
        "semijoin_key_cap",
        "spill_row_threshold",
        "max_parallel_members",
        "total_input_byte_cap",
    },
)

FEDERATION_MAPPINGS_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"version", "logical_columns", "logical_tables"})

FEDERATION_DECLARATION_TOP_LEVEL_KEYS: frozenset[str] = (
    (FEDERATION_MANIFEST_TOP_LEVEL_KEYS | FEDERATION_MAPPINGS_TOP_LEVEL_KEYS) - {"version"}
) | {"version"}

FEDERATION_MAPPINGS_LOGICAL_COLUMN_KEYS: frozenset[str] = frozenset(
    {"logical", "members", "role", "unify_in_graph"},
)

FEDERATION_MAPPINGS_LOGICAL_TABLE_KEYS: frozenset[str] = frozenset(
    {"logical", "semantics", "members", "authoritative_source"},
)

FEDERATION_MAPPINGS_TABLE_MEMBER_KEYS: frozenset[str] = frozenset({"source", "table", "columns"})

FEDERATION_COORDINATOR_DUCKDB_TYPE_MAP: Mapping[str, str | Callable[[int, int], str]] = MappingProxyType(
    {
        "bigint": "BIGINT",
        "int8": "BIGINT",
        "long": "BIGINT",
        "bigserial": "BIGINT",
        "smallint": "SMALLINT",
        "int2": "SMALLINT",
        "tinyint": "SMALLINT",
        "short": "SMALLINT",
        "smallserial": "SMALLINT",
        "int": "INTEGER",
        "integer": "INTEGER",
        "int4": "INTEGER",
        "serial": "INTEGER",
        "decimal": lambda precision, scale: f"DECIMAL({precision}, {scale})",
        "numeric": lambda precision, scale: f"DECIMAL({precision}, {scale})",
        "number": lambda precision, scale: f"DECIMAL({precision}, {scale})",
        "money": lambda precision, scale: f"DECIMAL({precision}, {scale})",
        "double": "DOUBLE",
        "float8": "DOUBLE",
        "double precision": "DOUBLE",
        "real": "REAL",
        "float4": "REAL",
        "float": "REAL",
        "bool": "BOOLEAN",
        "boolean": "BOOLEAN",
        "timestamp": "TIMESTAMP",
        "timestamptz": "TIMESTAMP WITH TIME ZONE",
        "datetime": "TIMESTAMP",
        "timestamp without time zone": "TIMESTAMP",
        "timestamp with time zone": "TIMESTAMP WITH TIME ZONE",
        "datetimeoffset": "TIMESTAMP WITH TIME ZONE",
        "timestamp_tz": "TIMESTAMP WITH TIME ZONE",
        "timestamp_ltz": "TIMESTAMP WITH TIME ZONE",
        "date": "DATE",
        "time": "TIME",
        "timetz": "TIME",
        "interval": "INTERVAL",
        "uuid": "UUID",
        "binary": "BLOB",
        "blob": "BLOB",
        "bytea": "BLOB",
        "varchar": "VARCHAR",
        "text": "VARCHAR",
        "char": "VARCHAR",
        "character": "VARCHAR",
        "character varying": "VARCHAR",
        "string": "VARCHAR",
        "bpchar": "VARCHAR",
        "nvarchar": "VARCHAR",
        "nchar": "VARCHAR",
        "ntext": "VARCHAR",
        "clob": "VARCHAR",
    },
)

FEDERATION_TIMEZONE_AWARE_DATA_TYPES: frozenset[str] = frozenset(
    {
        "timestamptz",
        "timetz",
        "datetimeoffset",
        "timestamp_tz",
        "timestamp_ltz",
    }
)

FEDERATION_EGRESS_STRIPPED_DETAIL_KEYS: frozenset[str] = frozenset({"source_id", "succeeded", "sources_queried"})

FEDERATION_REJECTION_BUCKETS: frozenset[str] = frozenset(
    {
        "MALFORMED_MEMBER_ANSWER",
        "JOIN_FAN_OUT",
    }
)

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

ORACLE_ENV_HOST: tuple[str, ...] = ("ORACLE_HOST", "ORACLE_SERVER")

ORACLE_ENV_PORT: tuple[str, ...] = ("ORACLE_PORT",)

ORACLE_ENV_USER: tuple[str, ...] = ("ORACLE_USER", "ORACLE_USERNAME")

ORACLE_ENV_PASSWORD: tuple[str, ...] = ("ORACLE_PASSWORD", "ORACLE_PWD")

ORACLE_ENV_SERVICE_NAME: tuple[str, ...] = ("ORACLE_SERVICE_NAME", "ORACLE_SERVICE")

ORACLE_ENV_SID: tuple[str, ...] = ("ORACLE_SID",)

ORACLE_ENV_SCHEMA: tuple[str, ...] = ("ORACLE_SCHEMA", "ORACLE_DEFAULT_SCHEMA")

ORACLE_ENV_AUTH_MODE: tuple[str, ...] = ("ORACLE_AUTH_MODE",)

ORACLE_ENV_WALLET_LOCATION: tuple[str, ...] = ("ORACLE_WALLET_LOCATION", "ORACLE_WALLET")

ORACLE_ENV_CONFIG_DIR: tuple[str, ...] = ("ORACLE_CONFIG_DIR",)

ORACLE_ENV_TOKEN: tuple[str, ...] = ("ORACLE_TOKEN", "ORACLE_ACCESS_TOKEN")

ORACLE_ENV_THICK_MODE: tuple[str, ...] = ("ORACLE_THICK_MODE",)

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

CSV_ENV_SOURCE_SELECTIONS: tuple[str, ...] = ("CSV_SOURCE_SELECTIONS",)

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

AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY: str = "AZURE_OPENAI_DEPLOYMENT_HEAVY"

TASK_MODEL_TO_DEPLOYMENT_FIELD: dict[str, str] = {
    "gpt-4.1-mini": "deployment_light",
    "gpt-4.1-nano": "deployment_light",
    "gpt-5-nano": "deployment_light",
    "gpt-5-mini": "deployment_light",
    "gpt-5.4-nano": "deployment_light",
    "gpt-5.4-mini": "deployment_heavy",
}

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
        "ORACLE_PASSWORD",
        "ORACLE_TOKEN",
        "ORACLE_ACCESS_TOKEN",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_OAUTH_TOKEN",
        "BIGQUERY_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
    },
)

TOML_SECTION_TO_ENGINE: dict[str, str] = {
    "postgresql": "postgresql",
    "databricks": "databricks",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "duckdb": "duckdb",
    "csv": "csv",
    "excel": "csv",
    "sqlite": "sqlite",
    "sqlserver": "sqlserver",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift",
}

TOML_ENGINE_FIELD_MAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "postgresql": (
        ("host", "POSTGRES_HOST"),
        ("port", "POSTGRES_PORT"),
        ("database", "POSTGRES_DB"),
        ("schema", "POSTGRES_SCHEMA"),
        ("user", "POSTGRES_USER"),
        ("password", "POSTGRES_PASSWORD"),
    ),
    "databricks": (
        ("host", "DATABRICKS_HOST"),
        ("http_path", "DATABRICKS_HTTP_PATH"),
        ("access_token", "DATABRICKS_ACCESS_TOKEN"),
        ("catalog", "DATABRICKS_CATALOG"),
        ("schema", "DATABRICKS_SCHEMA"),
    ),
    "mysql": (
        ("host", "MYSQL_HOST"),
        ("port", "MYSQL_PORT"),
        ("user", "MYSQL_USER"),
        ("password", "MYSQL_PASSWORD"),
        ("database", "MYSQL_DATABASE"),
    ),
    "mariadb": (
        ("host", "MARIADB_HOST"),
        ("port", "MARIADB_PORT"),
        ("user", "MARIADB_USER"),
        ("password", "MARIADB_PASSWORD"),
        ("database", "MARIADB_DATABASE"),
    ),
    "duckdb": (
        ("path", "DUCKDB_PATH"),
        ("database", "DUCKDB_DATABASE"),
        ("schema", "DUCKDB_SCHEMA"),
    ),
    "csv": (("directory", "CSV_DIRECTORY"),),
    "excel": (("directory", "CSV_DIRECTORY"),),
    "sqlite": (
        ("path", "SQLITE_PATH"),
        ("database", "SQLITE_DATABASE"),
    ),
    "sqlserver": (
        ("host", "SQLSERVER_HOST"),
        ("port", "SQLSERVER_PORT"),
        ("user", "SQLSERVER_USER"),
        ("password", "SQLSERVER_PASSWORD"),
        ("database", "SQLSERVER_DATABASE"),
        ("schema", "SQLSERVER_SCHEMA"),
        ("driver", "SQLSERVER_DRIVER"),
        ("auth_mode", "SQLSERVER_AUTH_MODE"),
        ("tenant_id", "SQLSERVER_TENANT_ID"),
        ("client_id", "SQLSERVER_CLIENT_ID"),
        ("client_secret", "SQLSERVER_CLIENT_SECRET"),
    ),
    "oracle": (
        ("host", "ORACLE_HOST"),
        ("port", "ORACLE_PORT"),
        ("user", "ORACLE_USER"),
        ("password", "ORACLE_PASSWORD"),
        ("service_name", "ORACLE_SERVICE_NAME"),
        ("sid", "ORACLE_SID"),
        ("schema", "ORACLE_SCHEMA"),
        ("auth_mode", "ORACLE_AUTH_MODE"),
        ("wallet_location", "ORACLE_WALLET_LOCATION"),
        ("config_dir", "ORACLE_CONFIG_DIR"),
        ("token", "ORACLE_TOKEN"),
        ("thick_mode", "ORACLE_THICK_MODE"),
    ),
    "snowflake": (
        ("account", "SNOWFLAKE_ACCOUNT"),
        ("user", "SNOWFLAKE_USER"),
        ("password", "SNOWFLAKE_PASSWORD"),
        ("database", "SNOWFLAKE_DATABASE"),
        ("schema", "SNOWFLAKE_SCHEMA"),
        ("warehouse", "SNOWFLAKE_WAREHOUSE"),
        ("role", "SNOWFLAKE_ROLE"),
        ("private_key_path", "SNOWFLAKE_PRIVATE_KEY_PATH"),
        ("private_key_passphrase", "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        ("authenticator", "SNOWFLAKE_AUTHENTICATOR"),
        ("oauth_token", "SNOWFLAKE_OAUTH_TOKEN"),
    ),
    "bigquery": (
        ("project", "BIGQUERY_PROJECT"),
        ("dataset", "BIGQUERY_DATASET"),
        ("credentials_path", "BIGQUERY_CREDENTIALS_PATH"),
        ("location", "BIGQUERY_LOCATION"),
    ),
    "redshift": (
        ("host", "REDSHIFT_HOST"),
        ("port", "REDSHIFT_PORT"),
        ("user", "REDSHIFT_USER"),
        ("password", "REDSHIFT_PASSWORD"),
        ("database", "REDSHIFT_DATABASE"),
        ("schema", "REDSHIFT_SCHEMA"),
        ("use_iam", "REDSHIFT_USE_IAM"),
        ("cluster_identifier", "REDSHIFT_CLUSTER_IDENTIFIER"),
        ("workgroup", "REDSHIFT_WORKGROUP"),
        ("region", "REDSHIFT_REGION"),
    ),
}

LLM_PRICE_TABLE_AS_OF: str = "2026-07-26"

LLM_PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-5.4-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.0625, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.0125, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
}

SCHEMA_GRAPH_ID_PREFIX: str = "sg_"

SCHEMA_GRAPH_ID_DETERMINISTIC_SEED_V1: str = "aetherdialect-sg-v1|"

REFUSAL_NOT_AVAILABLE_IN_CONTEXT: str = "refusal_not_available_in_context"

PROMPT_SCALAR_VALUE_TYPES: frozenset[str] = frozenset({"boolean", "date", "integer", "number", "string"})

NAMED_SCHEMA_CONTEXT_PREFIX: str = "schema_context."

MIGRATION_CHECKPOINT_SCHEMA_BASENAME: str = "schema_graph.json.gz"

SCHEMA_CONTEXT_CACHE_NAME: str = "schema_context.json"

SCHEMA_CONTEXT_CACHED_DDL: str = "_cached_schema_context.sql"

SCHEMA_CONTEXT_CACHED_NOTES: str = "_cached_schema_context_notes.txt"

SCHEMA_CONTEXT_NAMED_SPEC_GLOB: str = "schema_context.*.json"

SQL_STRING_LITERAL_COMMENT_MARKERS: tuple[str, ...] = ("--", "/*")

TEMPLATE_INTENT_KEY_INDEX_KEY: str = "intent_key_index"

WINDOW_REGISTRY_RANK_KIND_HINTS: frozenset[str] = frozenset(
    {"row_number", "rank", "dense_rank", "ntile", "percent_rank", "cume_dist"},
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

AETHERSPACE_SCOPE_MARKERS: tuple[str, ...] = (
    "aetherspace scope",
    "outside the active space",
)

EXECUTION_SCOPE_MARKERS: tuple[str, ...] = (
    "intent out of execution scope",
    "out of execution scope",
)

JOIN_PATH_TIE_OVERFLOW_MARKER: str = "__join_path_tie_overflow_count__"

REFUSAL_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
        DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
        DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
        DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
        DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
        DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
        DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
        DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
        DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
        DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
        DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
        DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
        DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
        DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
        DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
        DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
        DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
        DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
        DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
        DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
        DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
        DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
        DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
        DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    }
)

REFUSAL_CONDITION_CODES: dict[str, str] = {
    "permission_denial": DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    "scope_violation": DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    "invalid_question": DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    "parse_failure": DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    "declined_schema": DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    "tie_cap_exhaustion": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    "widened_clause_refusal": DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    "probe_placement": DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    "unsupported_column_type": DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    "null_in_negated_list": DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
    "opaque_expr": DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
    "ambiguous_date_literal": DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
    "union_column_missing": DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
    "join_path_unavailable": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    "aggregate_fan_out": DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    "hop_ceiling": DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    "cte_cap": DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    "capability_gap": DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    "not_available_in_context": DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    "subday_date_window": DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
}

OUTCOME_REFUSAL_CODES: dict[str, str] = {
    "permission_denied": DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    "restricted": DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    "conversational_deny": DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    "invalid_question": DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    "insufficient_knowledge": DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
    "parse_failed": DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    "schema_invalid_declined": DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    "not_available_in_context": DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
}

REPHRASE_HINT_REFUSAL_CODES: dict[str, str] = {
    "intent_parse_failed": DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    "schema_invalid_declined": DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    "join_path_unavailable": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    "restricted_question": DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    "conversational_deny": DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    "vague_question": DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    "federation_ineligible": DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
}

SCOPE_SENSITIVITY_REFUSAL_CODES: frozenset[str] = frozenset(
    {
        DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
        DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
        DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    }
)

REFUSAL_TERMINAL_OUTCOMES: frozenset[str] = frozenset(OUTCOME_REFUSAL_CODES.keys()) | frozenset({"validation_failed"})

EXPLAIN_VALIDATION_MESSAGES: dict[str, str] = {
    "explain_cost": "EXPLAIN cost gate exceeded configured limits.",
    "explain_schema": "SQL references objects that are not available in this context.",
    "explain_semantic": "SQL could not be validated against the schema.",
    "explain_transient": "SQL validation timed out.",
    "explain_failed": "SQL validation failed.",
}

REMEDIATION_SCOPE_MECHANISM_MARKERS: tuple[str, ...] = (
    "allow_columns",
    "deny_columns",
    "aetherspace",
)

REFUSAL_UNSUPPORTED_COLUMN_TYPE_ISSUE_IDS: frozenset[str] = frozenset(
    {
        "unsupported_column_type",
    }
)

REFUSAL_NULL_IN_NEGATED_LIST_ISSUE_IDS: frozenset[str] = frozenset(
    {
        "null_in_negated_list",
    }
)

DETERMINISTIC_PROBE_EDGE_KINDS: frozenset[str] = frozenset({"catalog_fk", "virtual_fk_bridge", "virtual_pk_bridge"})

AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV: str = "AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL"

DIAGNOSTIC_FORCE_DEPTH: int = 0

WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE: float = 0.25

BQ_DEFAULT_PARTITION_LOOKBACK_DAYS: int = 90

SEED_NORMALIZATION_BATCH_SIZE: int = 20

MIGRATION_DATA_OVERLAP_MIN: float = 0.15

MIGRATION_TABLE_RENAME_COLUMN_FRACTION: float = 0.60

WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY: int = 16

JSON_COMPACT_SEPARATORS: tuple[str, str] = (",", ":")

JSON_CONTAINMENT_ENGINES: frozenset[str] = frozenset({"postgresql", "mysql", "mariadb", "duckdb", "databricks"})

UPLOAD_COLUMN_TRANSFORM_IDS: tuple[str, ...] = (
    "parse_temporal",
    "strip_numeric_affix",
    "band_bounds",
    "band_value_map",
    "keep_canonical_columns",
    "derive_by_pattern",
    "drop_empty_columns",
    "null_tokens",
    "unpivot_columns",
)

REVIEW_GATED_UPLOAD_COLUMN_TRANSFORMS: frozenset[str] = frozenset(
    {
        "keep_canonical_columns",
        "drop_empty_columns",
        "unpivot_columns",
    }
)

META_DEFAULT_SOURCE_ID: str = "default"

MIGRATION_MAP_ACTION_REMAP: str = "remap"

MIGRATION_MAP_ACTION_DESTRUCTIVE: str = "destructive"

MIGRATION_MAP_ACTION_ABORT: str = "abort"

ARTIFACT_LAST_ACTION_REMAP_USER_MAP: str = "remap_user_map"

ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP: str = "destructive_user_map"

SEED_NORMALIZATION_JSON: str = "seed_question_normalization.json"

NORMALIZED_SEEDS_TXT: str = "seed_questions_normalized.txt"

AETHERSPACES_SEGMENT: str = "aetherspaces"

AETHERSPACE_UID_PREFIX: str = "S"

MASTER_AETHERSPACE_NAME: str = "master"

MASTER_AETHERSPACE_UID: str = "master"

CREDENTIAL_DEFAULT_AETHERSPACE_NAME: str = "_credential_default"

CREDENTIAL_DEFAULT_SNAPSHOT_FLAG: str = "credential_default"

CREDENTIAL_DEFAULT_FINGERPRINT_KEY: str = "credential_visibility_fingerprint"

CANONICAL_FEEDBACK_DIALECT: str = "duckdb"

TEMPLATE_STORE_SEGMENT: str = "intent_templates"

TEMPLATE_STORE_SPACES_SEGMENT: str = "spaces"

TEMPLATE_STORE_FEEDBACK_SEGMENT: str = "feedback"

FEEDBACK_SHARD_INDEX_KEY: str = "feedback_shard_index"

TEMPLATE_STORE_PARTITION_LRU_MAX: int = 32

TEMPLATE_STORE_LEGACY_SINGLE_FILE: str = "intent_templates.json.gz"

TEMPLATE_STORE_ORPHANED_SEGMENT: str = "orphaned"

MIGRATION_CHECKPOINT_PREFIX: str = ".migration_checkpoint_"

SEED_WARMUP_CACHE_ZIP: str = "seed_warmup_cache.zip"

LEGACY_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "schema_graph.json.gz",
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    "qsim_skeletons.json.gz",
    SEED_WARMUP_CACHE_ZIP,
)

SIMULATION_CACHE_EXACT_FILENAMES: tuple[str, ...] = (
    "qsim_skeletons.json.gz",
    "qsim_summary.json",
    SEED_WARMUP_CACHE_ZIP,
)

WARMUP_ANCHOR_LATTICE_SUBDIR: str = "anchor_lattice"

SIMULATION_CACHE_GLOB_PATTERNS: tuple[str, ...] = (
    "qsim_questions_v*.txt",
    "seed_warmup_report_v*.json",
    "seed_warmup_v*.zip",
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
    "qsim/summary_*.json",
    "qsim/index.jsonl",
    f"{WARMUP_ANCHOR_LATTICE_SUBDIR}/*",
)

LEGACY_ARTIFACT_GLOBS: tuple[str, ...] = (
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
)

SOFT_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        "explain_seq_scan_indexed",
        "explain_zero_estimate",
    }
)

RESULT_READER_KINDS: tuple[str, ...] = (
    "sqlalchemy",
    "spark",
    "connector",
    "bq_client",
    "bq_storage",
    "snowflake_arrow",
)

ARROW_RESULT_READER_KINDS: frozenset[str] = frozenset({"snowflake_arrow", "bq_storage"})

HIDDEN_SENSITIVITIES: frozenset[str] = frozenset({"hidden", "restricted"})

DISALLOWED_EXTRACT_UNITS: frozenset[str] = frozenset({"epoch"})

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

UPLOAD_INGEST_ENGINE_NAMES: frozenset[str] = frozenset({"duckdb", "csv"})

WEEK_START_DAY: str = "monday"

WEEK_NUMBERING: str = "iso"

YEAR_LITERAL_COMPARISON_OPS: frozenset[str] = frozenset({"=", ">", "<", ">=", "<="})

STRUCTURAL_DATA_TYPE_CANONICAL: Mapping[str, str] = MappingProxyType(
    {
        "int": "integer",
        "int2": "smallint",
        "int4": "integer",
        "int8": "bigint",
        "character varying": "varchar",
        "character": "char",
        "bool": "boolean",
        "float4": "real",
        "float8": "double",
        "double precision": "double",
        "timestamp without time zone": "timestamp",
        "timestamp with time zone": "timestamptz",
    }
)

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

STRING_TYPE_TOKENS: frozenset[str] = frozenset(
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

COMPATIBLE_TYPE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
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
        ("timestamptz", "timestamptz"),
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
)

GROUPBY_ADD: str = "GROUPBY_ADD"

ORDERBY_ADD: str = "ORDERBY_ADD"

GROUPBY_REMOVE: str = "GROUPBY_REMOVE"

DIMENSION_SWAP: str = "DIMENSION_SWAP"

TABLE_REMOVE: str = "TABLE_REMOVE"

BRIDGE_INTERMEDIATE_ADD: str = "BRIDGE_INTERMEDIATE_ADD"

INCLUDE_GOLD: str = "INCLUDE_GOLD"

TEMP_EXTRACT_GROUPBY: str = "TEMP_EXTRACT_GROUPBY"

NUM_ROUND_SELECT: str = "NUM_ROUND_SELECT"

DISTINCT_ADD: str = "DISTINCT_ADD"

SELECT_EXPR_PAIR_MULTIPLY: str = "SELECT_EXPR_PAIR_MULTIPLY"

SELECT_CASE_LABEL_ADD: str = "SELECT_CASE_LABEL_ADD"

ORDERBY_REMOVE: str = "ORDERBY_REMOVE"

SELECT_COL_TRIM: str = "SELECT_COL_TRIM"

DISTINCT_REMOVE: str = "DISTINCT_REMOVE"

SPLICE_SUBTREE: str = "SPLICE_SUBTREE"

EMI_MUTATE: str = "EMI_MUTATE"

CASE_CATEGORICAL_ADD: str = "CASE_CATEGORICAL_ADD"

SELECT_COALESCE_ADD: str = "SELECT_COALESCE_ADD"

EXPANSION_OPERATOR_IDS: frozenset[str] = frozenset(
    {
        WHERE_ADD,
        WHERE_EXPR_ADD,
        AGG_CHANGE,
        GROUPBY_ADD,
        ORDERBY_ADD,
        HAVING_VALUE_ADD,
        HAVING_EXPR_ADD,
        WHERE_REMOVE,
        GROUPBY_REMOVE,
        HAVING_REMOVE,
        JOIN_DIMENSION_ADD,
        JOIN_FACT_ADD,
        DIMENSION_SWAP,
        TABLE_REMOVE,
        BRIDGE_INTERMEDIATE_ADD,
        INCLUDE_GOLD,
        TEMP_EXTRACT_GROUPBY,
        TEMP_DATE_TRUNC_GROUPBY,
        TEMP_DATE_WINDOW_WHERE,
        TEMP_DATE_DIFF_WHERE,
        NUM_ROUND_SELECT,
        NUM_ABS_WHERE,
        DISTINCT_ADD,
        LIMIT_ADD,
        WHERE_OR_GROUP,
        SELECT_EXPR_PAIR_MULTIPLY,
        WINDOW_RANK_ADD,
        WINDOW_SUM_PARTITION_ADD,
        SELECT_CASE_LABEL_ADD,
        WINDOW_LAG_ADD,
        WINDOW_LEAD_ADD,
        WHERE_ILIKE_ADD,
        WHERE_ARRAY_CONTAINS_ADD,
        ORDERBY_REMOVE,
        LIMIT_REMOVE,
        SELECT_COL_TRIM,
        WINDOW_STRIP,
        DISTINCT_REMOVE,
        SPLICE_SUBTREE,
        EMI_MUTATE,
        CTE_WRAP_GROUPED,
        CTE_SCALAR_THRESHOLD,
        CASE_CATEGORICAL_ADD,
        WHERE_IN_LIST_ADD,
        WHERE_NULL_ADD,
        WHERE_NOT_NULL_ADD,
        HAVING_MATCH_SELECT_AGG,
        COUNT_DISTINCT_ADD,
        WINDOW_DENSE_RANK_ADD,
        WINDOW_RANK_FUNC_ADD,
        WINDOW_AVG_PARTITION_ADD,
        ORDERBY_WINDOW_COL_ADD,
        WHERE_LIKE_ADD,
        SELECT_COALESCE_ADD,
        SELECT_STRING_SCALAR_ADD,
        TEMP_EXTRACT_WHERE,
        CTE_UNNEST_ADD,
        SELF_JOIN_CTE_ADD,
        MULTI_CTE_CHAIN_ADD,
        SPLICE_HAVING_SUBTREE,
        SPLICE_WINDOW_SUBTREE,
    }
)

NULL_CHECK_OPS: frozenset[str] = frozenset({"is null", "is not null"})

TIMESTAMP_COLUMN_DATA_TYPES: frozenset[str] = frozenset(
    {
        "datetime",
        "datetime2",
        "datetimeoffset",
        "smalldatetime",
        "timestamp",
        "timestamptz",
        "timestamp_ntz",
        "timestamp_ltz",
        "timestamp_tz",
        "timestamp without time zone",
        "timestamp with time zone",
    }
)

STRING_OPS: frozenset[str] = frozenset({"like", "not like", "ilike", "not ilike", "contains"})

UNKNOWN_DATEPART_TO_EXTRACT_UNIT: Mapping[str, str] = MappingProxyType(
    {
        "year": "year",
        "month": "month",
        "day": "day",
        "quarter": "quarter",
        "dayofweek": "dow",
        "dow": "dow",
        "weekday": "dow",
    }
)

ARRAY_REWRITABLE_OPS: frozenset[str] = frozenset({"=", "!=", "like", "not like", "ilike", "not ilike"})

PK_STYLE_FK_STEMS: frozenset[str] = frozenset({"_id", "_key", "_uuid", "_pk"})

PK_NAME_SUFFIXES_FOR_LONGEST: tuple[str, ...] = ("_id", "_key", "_uuid", "_pk")

UF_EXCLUDE_SEMANTIC_INFERENCE_ONLY: frozenset[str] = frozenset({"semantic"})

INFERRED_COLLAPSE_TAGS: frozenset[str] = frozenset(
    {
        "suffix",
        "self",
        "composite",
        "semantic",
        "semantic_promoted",
    }
)

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
    "agg_in_where": "where_aggregation",
    "having_without_group": "having_validity",
    "explain_cartesian_join": "wrong_join",
    "explain_zero_estimate": "semantic_contradiction",
    "explain_seq_scan_indexed": "other",
    "explain_sort_spill": "wrong_sort_or_limit",
    "explain_temporary_table": "execution_explain_failed",
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
    "explain_zero_estimate": "_repair_where_overlap",
    "param_unbound": "_repair_param_binding",
}

CASE_ADD_OPS: frozenset[str] = frozenset(
    {
        SELECT_CASE_LABEL_ADD,
        CASE_CATEGORICAL_ADD,
    }
)

CSV_SUFFIXES: frozenset[str] = frozenset({".csv", ".xlsx"})

BOOL_LITERALS: frozenset[str] = frozenset({"1", "0", "true", "false", "t", "f", "yes", "no"})

INFERENCE_TAG_VALUES: frozenset[str] = frozenset(
    {
        "cross_source",
        "suffix",
        "self",
        "composite",
        "semantic",
        "semantic_promoted",
        "notes_structural",
        "user_override_structural",
        "user_override_semantic",
        "view_lineage",
    }
)

PK_INFERENCE_TAG_VALUES: frozenset[str] = frozenset({"ddl", "identity", "profile", "user_override"})

LITERAL_BEARING_CATEGORIES: frozenset[str] = frozenset({"missing_numeric_where", "missing_temporal_column"})

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
        "wrong_where_logic",
    }
)

KEPT_ISSUE_SEVERITIES: frozenset[str] = frozenset({"error", "warning"})

INSTRUCTIONAL_TABLE_PLACEHOLDER: str = "table"

INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER: str = "other_table"

INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER: str = "table.column"

INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER: str = "other_table.other_column"

INSTRUCTIONAL_TABLE_COLUMN_PLACEHOLDER: str = INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER

INSTRUCTIONAL_OTHER_TABLE_COLUMN_PLACEHOLDER: str = INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER

INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER: str = "link_table"

INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER: str = "junction_table"

INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER: str = "bridge_table"

INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER: str = "other_column"

INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER: str = "integer_column"

INSTRUCTIONAL_SHAPE_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER,
        INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER,
        INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER,
    }
)

UPLOAD_CURRENCY_AFFIX_TOKENS: frozenset[str] = frozenset(token for token in UPLOAD_SCALAR_AFFIX_TOKENS if token != "%")

ARITHMETIC_ROLES: frozenset[str] = frozenset({"numeric_measure", "numeric_categorical"})

STRUCTURAL_IDENTITY_VALUES: frozenset[int] = frozenset({0, 1})

UNSAFE_PARAM_LITERAL: str = "unsafe_param_literal"

IN_OPS: frozenset[str] = frozenset({"in", "not in"})

IN_STRING_SEPARATORS: re.Pattern[str] = re.compile(r"['\"]?\s*,\s*['\"]?")

DEFAULT_RANDOM_SEED: int = 2202

RANGE_OPS: frozenset[str] = frozenset({">", "<", ">=", "<="})

SHAPE_QUESTION_INDEX_KEY: str = "shape_question_index"

TEMPLATE_UNION_FAMILY_INDEX_KEY: str = "union_family_index"

TEMPLATE_QUESTION_TOKEN_INDEX_KEY: str = "question_token_index"

REDSHIFT_PROFILING_SAMPLE_PREDICATE: str = (
    "MOD(ABS(FNV_HASH(CAST({{col}} AS VARCHAR) || '{seed}')), 1000000) / 1000000.0 < {ratio}"
)

DUCKDB_PROFILING_SAMPLE_PREDICATE: str = "USING SAMPLE {pct:.4f} PERCENT (bernoulli, {seed})"

SQLITE_PROFILING_SAMPLE_PREDICATE: str = (
    "CAST((abs(hash({{col}} || '{seed}')) % 1000000) AS REAL) / 1000000.0 < {ratio}"
)

DUCKDB_EXPLAIN_CARTESIAN_TOKENS: tuple[str, ...] = ("CROSS_PRODUCT", "NESTED_LOOP_JOIN")

SQLITE_EXPLAIN_FULL_SCAN_TOKENS: tuple[str, ...] = ("SCAN TABLE", "SCAN ")

STRUCTURAL_CODE_TO_DIAG: Mapping[str, str] = MappingProxyType(
    {
        "ast_parse_failed": "ast_parse_failed",
        "multiple_statements": "multiple_statements",
        "no_root": "no_root",
        "not_select": "not_select",
        "subquery_not_allowed": "subquery_not_allowed",
        "using_not_allowed": "using_not_allowed",
        "cross_join_not_allowed": "cross_join_not_allowed",
        "self_join_not_allowed": "self_join_not_allowed",
        "exists_not_allowed": "exists_not_allowed",
        "lateral_not_allowed": "lateral_not_allowed",
        "forbidden_structure": "forbidden_structure",
        "cte_recursive": "forbidden_structure",
        "cte_malformed": "forbidden_structure",
        "cte_contains_subquery": "subquery_not_allowed",
        "cte_contains_exists": "exists_not_allowed",
        "cte_contains_set_op": "forbidden_structure",
    }
)

EXPANSION_SUBTREE_POOL_MAX: int = 128

SEED_FAILURE_CODE_REALISM_DROPPED: str = "realism_dropped"

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
        "ast_validate_bad_identifier",
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

PG_INNER_CONDITION_KEYS: tuple[str, ...] = ("Index Cond", "Recheck Cond", "Filter")

DBR_CARTESIAN_TOKENS: tuple[str, ...] = ("CartesianProduct", "BroadcastNestedLoopJoin")

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
    "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id JOIN sys.tables t ON i.object_id = t.object_id JOIN sys.schemas s ON t.schema_id = s.schema_id WHERE s.name = :s AND i.is_unique = 1 AND i.is_primary_key = 0 ORDER BY t.name, i.name, ic.key_ordinal"
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
    "((CASE qs.statement_end_offset WHEN -1 THEN DATALENGTH(st.text) ELSE qs.statement_end_offset END - qs.statement_start_offset)/2)+1) AS sql_text FROM sys.dm_exec_query_stats qs CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st ORDER BY qs.last_execution_time DESC"
)

SQLSERVER_SHOWPLAN_ROW_CACHE_MAX: int = 256

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

ORACLE_QUERY_LOG_AVAILABILITY_SQL: str = "SELECT 1 FROM V$SQL WHERE ROWNUM = 1"

ORACLE_QUERY_LOG_FETCH_SQL: str = (
    "SELECT sql_text FROM ("
    "  SELECT DISTINCT sql_fulltext AS sql_text, last_active_time "
    "  FROM V$SQL "
    "  WHERE last_active_time >= SYSTIMESTAMP - NUMTODSINTERVAL(:lookback_days, 'DAY') "
    "    AND sql_fulltext IS NOT NULL "
    "  ORDER BY last_active_time DESC"
    ") WHERE ROWNUM <= :max_queries"
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

SNOWFLAKE_QUERY_LOG_AVAILABILITY_SQL: str = (
    "SELECT 1 FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(RESULT_LIMIT => 1)) LIMIT 1"
)

SNOWFLAKE_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT QUERY_TEXT AS sql_text "
    "FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY("
    "END_TIME_RANGE_START => DATEADD(day, -:lookback_days, CURRENT_TIMESTAMP()), RESULT_LIMIT => :max_queries)) WHERE EXECUTION_STATUS = 'SUCCESS' ORDER BY START_TIME DESC"
)

BIGQUERY_QUERY_LOG_AVAILABILITY_SQL: str = "SELECT 1 FROM `{project}`.INFORMATION_SCHEMA.JOBS LIMIT 1"

BIGQUERY_QUERY_LOG_FETCH_SQL: str = (
    "SELECT DISTINCT query AS sql_text "
    "FROM `{project}`.INFORMATION_SCHEMA.JOBS "
    "WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL :lookback_days DAY) "
    "AND state = 'DONE' "
    "ORDER BY creation_time DESC LIMIT :max_queries"
)

DIAGNOSTIC_FUZZY_CUTOFF: float = 0.6

DISTINCT_ON_RANK_COLUMN: str = "__don_rn"

FIXED_WIDTH_TEXT_BASE_TYPES: frozenset[str] = frozenset({"char", "nchar", "bpchar", "character"})

UNSIGNED_INTEGER_TYPE_MAX: dict[str, int] = {
    "tinyint": 255,
    "smallint": 65535,
    "mediumint": 16777215,
    "int": 4294967295,
    "integer": 4294967295,
    "bigint": 18446744073709551615,
}

PERMISSION_DENIED_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        "access_policy",
        "denied_reference",
        "deny_bare_select",
        "sensitive_group_by",
    }
)

SCOPE_SENSITIVITY_FAILURE_KINDS: frozenset[str] = (
    frozenset(
        {
            "unknown_table",
            "unknown_column",
            "denied_reference",
            "deny_bare_select",
            "sensitive_group_by",
        }
    )
    | PERMISSION_DENIED_FAILURE_KINDS
)

EGRESS_STRIPPED_DETAIL_KEYS: frozenset[str] = frozenset({"table", "column", "reason", "name", "member_id", "source_id"})

PERMISSION_DENIED_CATEGORY_ORACLE_KINDS: frozenset[str] = frozenset(
    {
        "order_by_validity",
        "where_validity",
        "having_semantic",
    }
)

COLLATION_ENGINES: frozenset[str] = frozenset(
    {"postgresql", "redshift", "mysql", "mariadb", "sqlserver", "snowflake", "oracle"}
)

CASE_INSENSITIVE_COLLATION_ENGINES: frozenset[str] = frozenset({"mysql", "mariadb", "sqlserver"})

UNSIGNED_SEMANTICS_ENGINES: frozenset[str] = frozenset({"mysql", "mariadb"})

TIMESTAMPTZ_SEMANTICS_ENGINES: frozenset[str] = frozenset({"postgresql", "redshift", "snowflake", "duckdb", "bigquery"})

ROUNDING_MODE_HALF_EVEN_ENGINES: frozenset[str] = frozenset({"sqlite"})

DEFAULT_NULL_ORDERING_ASC: Literal["last"] = "last"

DEFAULT_NULL_ORDERING_DESC: Literal["first"] = "first"


DATABASE_ERROR_CLASSIFICATION_TRANSIENT: str = "transient"

DATABASE_ERROR_CLASSIFICATION_BY_MESSAGE_PATTERN: tuple[tuple[str, str], ...] = (
    ("connection reset", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("connection refused", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("server closed the connection", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("server closed", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("could not connect", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("deadlock detected", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("deadlock", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("lock wait timeout", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("statement timeout", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("statement_timeout", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("query cancelled", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("query canceled", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("too many connections", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("rate limit", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("name or service not known", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("could not translate host name", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("temporary failure in name resolution", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("temporarily unavailable", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("broken pipe", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("timed out", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("timeout", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("eof", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("network", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("unreachable", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("warehouse", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("cold-start", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("cold start", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("503", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("502", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
    ("429", DATABASE_ERROR_CLASSIFICATION_TRANSIENT),
)

DATABASE_ERROR_CLASSIFICATION_PERMANENT: str = "permanent"

DATABASE_ERROR_CLASSIFICATION_UNKNOWN: str = "unknown"

DATABASE_ERROR_CLASSIFICATION_BY_EXCEPTION_NAME: Mapping[str, str] = MappingProxyType(
    {
        "InterfaceError": DATABASE_ERROR_CLASSIFICATION_TRANSIENT,
    }
)

DATABASE_ERROR_CLASSIFICATION_TRANSIENT_ERRNOS: frozenset[int] = frozenset(
    {10060, 10061, 11001, 11002, 111, 113, 115, 116}
)
