"""Static package data: allow-lists, UI strings, and short error templates."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from sqlglot import exp

ENGINE_STORAGE_PLACEHOLDER_DIR: str = ".aetherdialect/__placeholder__"

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
    }
)

SESSION_PROMPT_YESNO: str = "Is this correct? (y/n): "
SESSION_PROMPT_REASON: str = "Please provide a reason: "
SESSION_USER_FEEDBACK_BODY: str = (
    "What was wrong?\n"
    "Tip: a single sentence is enough — for example 'wrong table', "
    "'missing date filter', or 'should aggregate by month'."
)
SESSION_INTENT_FEEDBACK_BODY: str = (
    "What should change about this interpretation?\n"
    "Tip: a single sentence is enough — for example 'wrong table', "
    "'missing date filter', or 'should aggregate by month'."
)

MIGRATION_HEADER_BY_TIER: dict[str, str] = {
    "additive": "Schema expanded with new tables or columns. Existing learning is kept.",
    "soft_refresh": "Refreshing cached metadata. Existing learning is kept.",
    "remap": "Schema renames detected. Mapping existing learning to the new names.",
    "destructive": "Learning reset: cache rebuilt from scratch (schema changed in ways that cannot be remapped).",
}

SAVED_LINE: str = "Saved."

FEEDBACK_NOTED_LINE: str = "Feedback noted. Try rephrasing your question for a better match."

SESSION_PERSISTENCE_FORMAT_VERSION: str = "0.2.1"
SUSPEND_STATE_FORMAT_VERSION: str = "0.2.1"

QUERY_RESULTS_HEADER: str = "Query Results"

FAILURE_TRACE_ROTATE_BYTES: int = 8388608

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
    "join_path_unavailable": (
        "These tables cannot be joined with the relationships currently in this schema.\n\n"
        "Tips: declare a foreign-key or semantic link between the tables named in the error, "
        "or narrow the question to tables that already connect.\n"
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
        "This question cannot be answered with the information currently available.\n\n"
        "Try rephrasing to ask about tables and columns you can see in the schema."
    ),
    "vague_question": (
        "I could not pin this question to specific tables or columns.\n\n"
        "Try naming the entity (a table or business object), the metric you want, and any filter (date range, "
        "status, region) so I have something concrete to map.\n"
    ),
    "federation_ineligible": (
        "Please rephrase your question.\n\n"
        "Tips: this federated question cannot be decomposed across members; "
        "try simpler per-source questions, declared cross-source joins, or single-member scope.\n"
    ),
    "federation_partial_failure": (
        "A federated query could not complete because one member failed after others succeeded. "
        "No partial answer was returned.\n\n"
        "Tips: retry the question, or ask on each member engine individually.\n"
    ),
    "federation_cap_exceeded": (
        "A federated query could not complete because a coordinator or member resource limit was exceeded.\n\n"
        "Tips: narrow the question, add filters, or reduce the result scope.\n"
    ),
    "federation_member_execution_failed": (
        "A federated query could not complete because one member failed.\n\n"
        "Tips: retry the question, or ask on each member engine individually.\n"
    ),
    "federation_member_probe_failed": (
        "Federation initialization could not connect to a member database.\n\n"
        "Tips: verify member connectivity and credentials, then retry federation setup.\n"
    ),
    "federation_turn_cancelled": (
        "The federated query was cancelled before it could finish.\n\n"
        "Tips: retry the question if you still need an answer.\n"
    ),
}

REMEDIATION_RESTRICTED_QUESTION: str = (
    "Review allow_columns and deny_columns scope for this aetherspace so the referenced "
    "tables or columns are visible to the end user."
)

USER_REJECTED_RESULT_BUCKET_TIPS: dict[str, str] = {
    "MISSING_FILTER": "Tips: name the filter or dimension you care about (time range, status, category).",
    "WRONG_GROUPING": "Tips: say whether you want totals per entity, per period, or overall.",
    "WRONG_AGGREGATION": "Tips: specify sum, average, count, or another metric clearly.",
    "WRONG_TIME_RANGE": "Tips: give an explicit date range or relative window.",
    "WRONG_TABLES_OR_JOINS": "Tips: name the tables or relationships that should connect your answer.",
    "WRONG_SORT_OR_LIMIT": "Tips: say how results should be ordered or how many rows you need.",
    "OTHER": "Tips: be more specific about columns, filters, grouping, or time range.",
    "MALFORMED_MEMBER_ANSWER": (
        "Tips: the member result did not match the requested projection; retry or narrow the question."
    ),
    "JOIN_FAN_OUT": (
        "Tips: the cross-source join multiplied rows; check that join keys are unique on the declared side."
    ),
}

JOIN_PRIOR_FEEDBACK_HEADING: str = "Previously rejected joins for this question (avoid these table sets / FK paths):"
JOIN_PRIOR_FEEDBACK_PATH_LABEL: str = "FK path:"

DETERMINISTIC_PROBE_EDGE_KINDS: frozenset[str] = frozenset({"catalog_fk", "virtual_fk_bridge", "virtual_pk_bridge"})

TABLE_SCOPE_REPAIR_REASON_TEXT: dict[str, str] = {
    "planner_align": "aligned with planner scope (join bridge)",
    "expression_reference": "referenced in query expressions",
    "unreferenced_table": "not referenced in query expressions",
    "join_bridge": "required by the chosen join path",
}

NORMALIZATION_ALLOWED_INTRODUCED_TOKENS: frozenset[str] = frozenset(
    {"list", "count", "sum", "average", "max", "min", "total", "of"},
)

INSTRUCTIONAL_TABLE_PLACEHOLDER: str = "table"
INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER: str = "other_table"
INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER: str = "table.column"
INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER: str = "other_table.other_column"
INSTRUCTIONAL_TABLE_COLUMN_PLACEHOLDER: str = INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER
INSTRUCTIONAL_OTHER_TABLE_COLUMN_PLACEHOLDER: str = INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER
INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER: str = "link_table"
INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER: str = "junction_table"
INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER: str = "bridge_table"
INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER: str = "date_column"
INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER: str = "other_date_column"
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

INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR: str = (
    "Instructional placeholder tokens appear in expression strings. Replace each with exact table.column "
    "names from schema_info. Do not leave angle-bracket markup, table_N or column_N instructional tokens, "
    f"or synthetic shape tokens from the prompt ({INSTRUCTIONAL_TABLE_PLACEHOLDER}, "
    f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}, {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, "
    f"{INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER})."
)

NATURAL_LANGUAGE_REFUSAL_PARSE_ERROR: str = (
    "natural_language contains refusal or permission prose while select_cols remain populated"
)

ARTIFACT_DIRECTORY_SEGMENT: str = "aetherdialect"

ENGINE_STORAGE_SLUG_MAX_CHARS: int = 180

AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV: str = "AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL"

SUPPORTED_ENGINES: frozenset[str] = frozenset()

CLASS_DELEGATED_METHODS = frozenset(
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
        "resolve_source_files",
        "set_source_selections",
        "apply_connection_credentials",
    }
)


EngineDriverRequirement = tuple[str | tuple[str, ...], str | tuple[str, ...], str]

ENGINE_DRIVER_REQUIREMENTS: dict[str, EngineDriverRequirement] = {
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
}


ARTIFACT_DIR_MODE: int = 0o700
ARTIFACT_FILE_MODE: int = 0o600
ARTIFACT_LOCK_TIMEOUT_SECONDS: float = 30.0
ARTIFACT_LOCK_POLL_INTERVAL_SECONDS: float = 0.05
SCHEMA_CLASSIFY_ERROR_DETAIL_CAP: int = 50
DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK: str = "STALE_ARTIFACT_LOCK"
DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL: str = "ARTIFACTS_DIR_NOT_LOCAL"
DIAGNOSTIC_FORCE_DEPTH: int = 0

NORMALIZATION_JACCARD_FLOOR: float = 0.4

TRUST_FLOOR: int = 1
TRUST_CEILING: int = 2
TRUST_AUTO_ACCEPT_THRESHOLD: int = 1

WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE: float = 0.25

WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS: float = 30.0
WRITE_QUEUE_MAX_BYTES_PER_DRAIN: int = 4194304
DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT: str = "WRITE_QUEUE_CORRUPT"
DIAGNOSTIC_CODE_WRITE_QUEUE_FULL: str = "WRITE_QUEUE_FULL"

SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS: int = 2000

MAX_NON_AGG_COL_DIFF = 2

BQ_DEFAULT_PARTITION_LOOKBACK_DAYS: int = 90

SEED_NORMALIZATION_BATCH_SIZE: int = 20

MIGRATION_DATA_OVERLAP_MIN: float = 0.15
MIGRATION_TABLE_RENAME_COLUMN_FRACTION: float = 0.60

WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY: int = 16

WARMUP_ROUND_TRIP_LIMIT: int = 100

EMPTY_JOIN_CANDIDATES: dict[str, Any] = {"candidates": []}

SCHEMA_GRAPH_ID_PREFIX: str = "sg_"
SCHEMA_GRAPH_ID_DETERMINISTIC_SEED_V1: str = "aetherdialect-sg-v1|"

PERMISSION_DENIED_USER_MESSAGE: str = (
    "You do not have access to one or more tables required by this answer. Please contact your administrator."
)

REFUSAL_NOT_AVAILABLE_IN_CONTEXT: str = "refusal_not_available_in_context"

REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE: str = (
    "This question refers to information that is not available in this context."
)

TABLE_PREVIEW_DEFAULT_LIMIT: int = 20
TABLE_PREVIEW_MAX_LIMIT: int = 200

JSON_COMPACT_SEPARATORS: tuple[str, str] = (",", ":")

SCHEMA_FIELD_DESCRIPTION: str = "description"
SCHEMA_FIELD_ROLE: str = "role"
SCHEMA_FIELD_TYPE: str = "type"
SCHEMA_FIELD_TRUTH_VALUE: str = "truth_value"
SCHEMA_FIELD_KEYS: str = "keys"
SCHEMA_FIELD_ENUM: str = "enum"
SCHEMA_FIELD_SAMPLES: str = "samples"
SCHEMA_FIELD_DERIVED: str = "derived"
SCHEMA_FIELD_RAW_TYPE: str = "raw_type"

UNKNOWN_VALUE_TYPE: str = "unknown"

JSON_COLUMN_TYPE_TOKENS: frozenset[str] = frozenset({"json", "jsonb"})

PROMPT_SCALAR_VALUE_TYPES: frozenset[str] = frozenset({"boolean", "date", "integer", "number", "string"})

JSON_CONTAINMENT_ENGINES: frozenset[str] = frozenset({"postgresql", "mysql", "mariadb", "duckdb", "databricks"})

INTERPRET_FIELDS: frozenset[str] = frozenset({SCHEMA_FIELD_DESCRIPTION, SCHEMA_FIELD_ENUM})
GROUND_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_DESCRIPTION,
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_SAMPLES,
        SCHEMA_FIELD_DERIVED,
    }
)
COMPOSE_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_KEYS,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_DERIVED,
    }
)
FULL_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_DESCRIPTION,
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_KEYS,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_DERIVED,
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

META_SCHEMA_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_kind", "headline", "counts", "tables", "relationships", "notes"],
    "additionalProperties": False,
    "properties": {
        "response_kind": {"type": "string", "const": "schema_catalog"},
        "headline": {"type": "string", "minLength": 1},
        "counts": {
            "type": "object",
            "required": ["tables", "columns", "members", "columns_in_table", "tables_in_member"],
            "additionalProperties": False,
            "properties": {
                "tables": {"type": ["integer", "null"]},
                "columns": {"type": ["integer", "null"]},
                "members": {"type": ["integer", "null"]},
                "columns_in_table": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": ["table", "columns"],
                            "additionalProperties": False,
                            "properties": {
                                "table": {"type": "string", "minLength": 1},
                                "columns": {"type": "integer"},
                            },
                        },
                    ]
                },
                "tables_in_member": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": ["source_id", "tables"],
                            "additionalProperties": False,
                            "properties": {
                                "source_id": {"type": "string", "minLength": 1},
                                "tables": {"type": "integer"},
                            },
                        },
                    ]
                },
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "source_id", "description", "columns"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "source_id": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "data_type", "role", "description"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "data_type": {"type": "string"},
                                "role": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["left", "right", "kind"],
                "additionalProperties": False,
                "properties": {
                    "left": {"type": "string", "minLength": 1},
                    "right": {"type": "string", "minLength": 1},
                    "kind": {"type": "string", "enum": ["fk", "semantic"]},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

META_KNOWLEDGE_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_kind", "message"],
    "additionalProperties": False,
    "properties": {
        "response_kind": {"type": "string", "const": "business_knowledge"},
        "message": {"type": "string", "minLength": 1},
    },
}

INTERPRET_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Identify the semantic entities, measures, conditions, grouping, ordering, ranking, row cap, conditional labeling, and time reasoning needed to answer the question.",
    "Reformulate unsupported full-SQL constructs into supported analysis shapes in plain language without naming IR or SQL operators.",
    "Infer whether the question needs row-level output, grouped output, a scalar answer, staged intermediate computation, a windowed comparison, or a conditional bucketed result.",
    "Recognize existence, absence, set difference, one-row-per-partition, and outer-join preservation needs and describe them in plain language without SQL operators.",
    "Use only the domain schema descriptions and enum heads to ground business concepts; capture any missing or ambiguous binding as internal planning uncertainty rather than refusing.",
    "Record grounding traceability for tables, enum values, and filter, having, or group_by constraints only; do not enumerate select output columns in grounding.",
)

GROUND_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Bind the interpret pathway to real schema identifiers and natural-language clause descriptions without emitting runtime IR.",
    "Populate select, filter, group_by, having, order_by, limit, window, and case prose fields; copy every literal into the matching prose field.",
    f"List semantic base tables in tables; omit {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables when its columns appear in prose.",
    f"Keep a table in join scope only via qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens in select, filter, group_by, having, order_by, or registry prose — not from the tables list alone.",
    f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
    f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
    "Use cte_steps when staged computation is needed; each step tables list may name base schema tables and prior cte_steps names this step reads from.",
    "Describe semi_join or anti_join probe CTE steps when existence, absence, or set difference is needed; mention distinct_on and preserve_tables in prose when the approach requires them.",
    "Never author join paths; the engine discovers foreign-key paths after structural encoding.",
)

COMPOSE_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Return columns or computed expressions; optionally return only distinct rows.",
    "Aggregate with count, sum, avg, min, max, string_agg, stddev, variance, or median; sum, avg, stddev, variance, and median require numeric columns; string_agg concatenates text with a separator param and optional within-aggregate ordering.",
    "Group by one or more columns (grouped grain), or compute a single aggregate over all rows (scalar grain); a query with no aggregation is row-level.",
    "Filter rows with =, !=, <, <=, >, >=, like, not like, ilike, not ilike, in, not in, between, is null, is not null, and array contains.",
    "Encode WHERE and HAVING boolean logic as nested PredicateGroup trees (op and/or with predicate leaves and child groups).",
    "Restrict aggregated results with a HAVING comparison using =, !=, <, <=, >, >=, in, not in, or between.",
    "Order by any column or expression ascending or descending with optional nulls first or nulls last, and cap rows with a limit.",
    "Compute arithmetic with +, -, *, / and wrap arithmetic in an aggregate.",
    "Apply scalar functions upper, lower, trim, ltrim, rtrim, length, abs, round, floor, ceil, date_trunc, date_part, extract (never epoch), coalesce, concat, year, month, day.",
    "Concatenate text with concat, never the || operator.",
    "Rank or number rows per group and compute running or offset values with window functions row_number, rank, dense_rank, ntile, percent_rank, cume_dist, sum, avg, lag, lead, first_value, last_value, nth_value, using partitioning, ordering, numeric_argument where required, and row or range frames.",
    "Produce conditional labels or buckets with CASE.",
    "Filter on a relative time window (last N days, weeks, months, quarters, or years) or on the difference between two date columns.",
    "Compare or shift a date by an integer number of days, and reference the current date or current timestamp.",
    "Break a computation into intermediate steps (WITH steps) for staged aggregation, self-comparison, per-entity ranking, or reuse; a later step reads an earlier step's named outputs.",
    "When the question asks which entities have a matching row elsewhere, declare a CTE with emission semi_join projecting the join keys.",
    "When the question asks which entities lack a matching row or differ from another set, declare a CTE with emission anti_join projecting the compared tuple.",
    "Return one row per partition with distinct_on and order_by_cols for within-partition ranking.",
    "Preserve anchor tables through joins with preserve_tables even when their columns are not selected.",
    "Not expressible: UNION, INTERSECT, recursive CTEs, and LATERAL joins.",
    "Join paths are discovered by the engine from foreign keys; never author joins or name junction tables except when the many-to-many set itself is requested; list only the base tables whose columns are used.",
)

FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    *COMPOSE_SUPPORTED_CAPABILITIES[:-2],
    "Do not emit SQL UNION; each listed table is already the complete relation for its entity.",
    "Not expressible: INTERSECT, recursive CTEs, and LATERAL joins.",
    COMPOSE_SUPPORTED_CAPABILITIES[-1],
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
ASK_PHASE_I: str = "I:plan.decompose"
ASK_PHASE_J: str = "J:sql.build_joins"
ASK_PHASE_K: str = "K:sql.validate_scope"
ASK_PHASE_L: str = "L:sql.execute"
ASK_PHASE_M: str = "M:plan.combine"
ASK_PHASE_N: str = "N:feedback"

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

SCHEMA_CLASSIFY_SYSTEM: str = (
    "Classify every table's role and every column listed in the input for each table.\n\n"
    "INPUT SCOPE:\n"
    "Each table object lists only the columns you must classify under columns. "
    "Return JSON with exactly those table keys. For each table, include exactly those column keys under columns — "
    "no additional columns and no omissions.\n\n"
    "TABLE ROLES:\n"
    "- dimension: reference/lookup table referenced by others, descriptive attributes\n"
    "- fact: transactional/event table with FKs to dimensions, contains measures\n"
    "- bridge: junction table for many-to-many, mostly FKs, few own columns\n"
    "- unknown: cannot confidently classify\n"
    "Use FK topology: tables referenced by many others are dimension; tables with many outbound FKs are fact; tables with only 2+ FKs and minimal columns are bridge.\n\n"
    "COLUMN ROLE DECISION PRIORITY (evaluate in order, first match wins):\n"
    "1. is_primary_key or is_foreign_key → identifier\n"
    "2. value_type is date → temporal (unless name clearly marks audit-only system timestamp like last_update → audit)\n"
    "3. name suggests binary state (is_*, has_*, active) and distinct_count = 2 → boolean\n"
    "4. value_type is integer or number and name suggests quantity/amount/price/count/size/distance → numeric_measure\n"
    "4b. value_type is integer and name suggests duration/period/lead_time/lag/tenure/offset in time units → temporal\n"
    "5. value_type is integer or number and name suggests code/rating/level/rank/status/tier/year → numeric_categorical\n"
    "6. value_type is integer or number with no clear name signal → numeric_measure (default for numeric value types)\n"
    "7. value_type is string and very high distinct_ratio → free_text\n"
    "8. value_type is string → categorical (default for string value types)\n\n"
    "AMBIGUOUS TWO-VALUE COLUMNS:\n"
    "When a column has exactly two sampled categorical values (e.g. positive/negative-style pairs), do not assume boolean unless name, value_type, FK topology, and profile_hints clearly support a flag.\n\n"
    "PROFILE HINTS (supporting evidence only — never override name/value_type signals):\n"
    "Each column may include a profile_hints object with distinct_count, distinct_ratio, and null_ratio. Use these to confirm or disambiguate when name and value_type are ambiguous.\n"
    "Do NOT use profile_hints as the primary reason to choose a role.\n\n"
    "CROSS-TABLE CONSISTENCY:\n"
    "- Columns with the same name and value_type across tables MUST receive the same role.\n"
    "- Deduce roles from names, value_type, FK topology, and profile_hints using the priority above.\n\n"
    "COLUMN DESCRIPTIONS:\n"
    "For each column, provide a short semantic description (max 8 words) describing what the column represents in business terms. Every column object MUST include a non-empty description.\n"
    "Role-based guidance for column descriptions:\n"
    "- identifier columns: describe what entity the ID refers to.\n"
    "- numeric_measure columns: state the unit or what is measured.\n"
    "- categorical columns: mention common category values or groupings.\n"
    "- temporal columns with value_type date: state what event the date/time marks.\n"
    "- temporal columns with value_type integer: state that the column holds a day-count or period length (not a calendar date).\n"
    "- audit columns: state that the column records when a row was last changed by the system.\n"
    "- boolean columns: describe the yes/no condition.\n"
    "- FK columns: MUST state what business data the target table provides when joined. Name the key descriptive columns on the target table (e.g. 'links to target_table for name, title, description').\n\n"
    "TABLE DESCRIPTIONS:\n"
    "For each table provide a one-line business purpose that includes: (a) what entity or event the table represents, (b) which related tables it connects to via foreign keys, and (c) the notable descriptive or measure columns it provides that users commonly ask about. Every table MUST include a non-empty description.\n\n"
    "SENSITIVITY (per column, optional):\n"
    'Include "sensitivity" in each column object: always null in this pass.\n'
    'A later second-pass refine step may set "restricted" or "hidden" only when domain notes explicitly require it.\n\n'
    "Reason internally, output only JSON:\n"
    '{"table1": {"table_role": "...", "description": "...", '
    '"columns": {"col1": {"role": "...", "description": "...", "sensitivity": null}, ...}}, ...}'
)

SCHEMA_NOTES_REFINE_SYSTEM: str = (
    "You refine base_classification using domain_notes.\n\n"
    "The base_classification was produced from profiling statistics and FK topology alone.\n"
    "Apply domain_notes to tighten descriptions, adjust table_role and column roles where notes explicitly require, "
    'and set column sensitivity to "restricted" or "hidden" only when domain_notes explicitly mark sensitive data for that column or category.\n'
    "Preserve substantive keywords and meaning from base table and column descriptions.\n"
    "Override table_role, column role, column description, table description, or sensitivity only when domain_notes are explicit; "
    "when notes are silent, keep the base values.\n"
    "Do not remove tables or columns from base_classification. Do not add new tables or columns.\n"
    "Keep exactly the same table keys and column keys as base_classification.\n"
    "Emit the full merged JSON with the same shape as base_classification.\n"
    "Reason internally, output only JSON:\n"
    '{"table1": {"table_role": "...", "description": "...", '
    '"columns": {"col1": {"role": "...", "description": "...", "sensitivity": null}, ...}}, ...}'
)

BUSINESS_KNOWLEDGE_NOTES_EXTRACT_SYSTEM: str = (
    "You extract business-knowledge entries from operator notes.\n"
    "Return a JSON array of objects with keys key, kind, and text only.\n"
    "kind must be one of: glossary, policy, metric, synonym, caveat.\n"
    "Include only definitions, policies, metrics, synonyms, and caveats that are not descriptions of a specific "
    "table or column already listed in the user payload schema_names.\n"
    "Do not invent facts absent from the notes. Do not include empty keys or empty text.\n"
    "Reason internally, output only the JSON array."
)

SCHEMA_CONSISTENCY_REFINE_SYSTEM: str = (
    "You receive base_classification JSON describing every table and column. The user message contains only "
    "base_classification under that key.\n\n"
    "Preserve the base output unless you detect a genuine cross-table inconsistency — for example the same "
    "column name and value_type assigned different roles in different tables. When you fix such an "
    "inconsistency, align the conflicting entries to the role that best matches the shared name, value_type, and "
    "FK topology.\n\n"
    "Do not invent new descriptions: keep each table description and column description from the base unless a "
    "detected inconsistency forces a minimal coordinated rewrite.\n"
    "Do not change sensitivity values from the base.\n"
    "Do not change column roles when the base assignment is already internally consistent.\n"
    "Do not remove tables or columns from base_classification. Do not add new tables or columns.\n\n"
    "Emit JSON identical in shape to base_classification.\n"
    "Reason internally, output only JSON:\n"
    '{"table1": {"table_role": "...", "description": "...", '
    '"columns": {"col1": {"role": "...", "description": "...", "sensitivity": null}, ...}}, ...}'
)

DESCRIPTION_REFINER_SYSTEM: str = (
    "You refine human-written database descriptions so a downstream text-to-SQL LLM can use them effectively. "
    "When previous_text is non-empty, mirror its prose style, length, and structural pattern (sentence shape, "
    "role mentions, qualifier ordering). Preserve every keyword and identifier the human wrote in text "
    "(column names, table names, units, values, conditions, references). Tighten phrasing, remove fluff, make role "
    "and business meaning explicit, and keep wording in plain prose. Do not invent facts the human did not state. "
    "Output ONLY valid JSON."
)

BUSINESS_KNOWLEDGE_REFINER_SYSTEM: str = (
    "You refine business-knowledge entries for a text-to-SQL system. "
    "Input is JSON with 'entries' (objects with key, kind, text) and 'schema_names' (qualified table and "
    "table.column names). Return ONLY JSON of the form "
    '{"entries": [{"key": "...", "kind": "...", "text": "..."}]} '
    "with the same keys as the input (or fewer when merging duplicate keys). "
    "Do not invent keys that were not in the input. Keep kind in "
    "{glossary, policy, metric, synonym, caveat}. "
    "Tighten prose; preserve human keywords; do not invent facts. "
    "Do not write definitions that merely rename a schema table or column—those belong in schema descriptions."
)

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

QSIM_FILL_SYSTEM: str = (
    "You are a SQL intent generator."
    " Given structural skeleton constraints, generate a valid query intent filling in specific columns and filters from the available schema."
    " Rules:"
    " 1. STRICTLY follow skeleton constraints for tables, aggregation presence, filter count, groupby count, having, orderby."
    " 2. Use ONLY columns from the provided schema in the specified tables."
    " 3. For filters, choose columns with meaningful filter potential (categorical, boolean/flag, or temporal columns)."
    " 4. For aggregation, use COUNT/SUM/AVG/MIN/MAX wrapping table.column in select_cols; aggregate numeric columns only (not IDs or foreign keys); COUNT may use any column or *."
    " 5. For groupby, choose categorical or temporal columns that make semantic sense."
    " 6. Ensure all column references use table.column format from the specified tables."
    " 7. Return ONLY valid JSON matching the specified format."
    " 8. For expr_comparison (filter expr-vs-expr), both expressions must be from different tables with compatible types."
    " 9. DISTINCT is only valid for non-aggregated queries."
    " 10. For orderby, include ASC or DESC direction suffix."
    " 11. DO NOT return columns or tables not in the provided schema."
    " 12. For having, expression must be an aggregation matching a select_cols aggregation."
)

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
DIAGNOSTIC_CODE_LARGE_RESULT_WARNING: str = "LARGE_RESULT_WARNING"
DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT: str = "SENSITIVITY_GATE_HIT"
DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY: str = "INTERPRET_GROUND_RETRY"
DIAGNOSTIC_CODE_COMPOSE_REPAIR: str = "COMPOSE_REPAIR"
DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART: str = "FALLBACK_FRESH_RESTART"
DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED: str = "CONFIG_FILE_VALUE_APPLIED"
DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED: str = "CONFIGURATION_KEY_IGNORED"
DIAGNOSTIC_CODE_ENGINE_INFO: str = "ENGINE_INFO"
DIAGNOSTIC_CODE_LLM_TURN_COST: str = "LLM_TURN_COST"
DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP: str = "SCHEMA_OVERRIDE_SKIP"
DIAGNOSTIC_CODE_OVERRIDE_NEEDS_RECONFIRMATION: str = "OVERRIDE_NEEDS_RECONFIRMATION"
DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT: str = "PK_INFERENCE_PROMPT"
DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION: str = "ZERO_ROW_WHERE_SUGGESTION"
DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED: str = "ZERO_ROW_WHERE_AUTO_FIXED"
DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING: str = "DATA_QUALITY_BLOCKING"
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

DATA_QUALITY_ISSUE_EMPTY_FILE: str = "empty_file"
DATA_QUALITY_ISSUE_DUPLICATE_HEADER: str = "duplicate_header"
DATA_QUALITY_ISSUE_BLANK_HEADER: str = "blank_header"
DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE: str = "header_not_row_one"
DATA_QUALITY_ISSUE_MULTIPLE_TABLES: str = "multiple_tables"
DATA_QUALITY_ISSUE_RAGGED_ROW: str = "ragged_row"
DATA_QUALITY_ISSUE_REPEATED_HEADER: str = "repeated_header"
DATA_QUALITY_ISSUE_SECTION_HEADING: str = "section_heading"
DATA_QUALITY_ISSUE_OVERFULL_ROW: str = "overfull_row"
DATA_QUALITY_ISSUE_BLANK_ROW: str = "blank_row"
DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW: str = "footer_note_row"
DATA_QUALITY_ISSUE_FORMULA_CELL: str = "formula_cell"
DATA_QUALITY_ISSUE_EMPTY_COLUMN: str = "empty_column"
DATA_QUALITY_ISSUE_SINGLE_COLUMN: str = "single_column"
DATA_QUALITY_ISSUE_MERGED_METADATA: str = "merged_metadata"
DATA_QUALITY_ISSUE_TOTAL_ROW: str = "total_row"
DATA_QUALITY_ISSUE_MERGED_CELLS: str = "merged_cells"
DATA_QUALITY_ISSUE_EMBEDDED_OBJECT: str = "embedded_object"
DATA_QUALITY_ISSUE_MIXED_TYPES: str = "mixed_types"
DATA_QUALITY_ISSUE_NUMBER_AS_TEXT: str = "number_as_text"
DATA_QUALITY_ISSUE_NULL_TOKEN: str = "null_token"
DATA_QUALITY_ISSUE_EXCEL_ERROR: str = "excel_error"
DATA_QUALITY_ISSUE_DUPLICATE_RELATION: str = "duplicate_relation"
DATA_QUALITY_ISSUE_EMPTY_SHEET: str = "empty_sheet"
DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED: str = "workbook_encrypted"
DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT: str = "workbook_corrupt"
DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE: str = "unsupported_type"
DATA_QUALITY_ISSUE_MERGEABLE_REGIONS: str = "mergeable_regions"
DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH: str = "merge_header_mismatch"
DATA_QUALITY_ISSUE_APPENDABLE_REGIONS: str = "appendable_regions"
DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH: str = "append_header_mismatch"
DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE: str = "invalid_merge_range"

DATA_QUALITY_SEVERITY_ADVISORY: str = "advisory"
DATA_QUALITY_SEVERITY_REVIEW: str = "review"
DATA_QUALITY_SEVERITY_BLOCKING: str = "blocking"
DATA_QUALITY_SEVERITY_FATAL: str = "fatal"

DATA_QUALITY_ISSUE_SEVERITY: dict[str, str] = {
    DATA_QUALITY_ISSUE_APPENDABLE_REGIONS: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_BLANK_HEADER: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_BLANK_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_DUPLICATE_HEADER: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_DUPLICATE_RELATION: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EMBEDDED_OBJECT: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_EMPTY_COLUMN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_EMPTY_FILE: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EMPTY_SHEET: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EXCEL_ERROR: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_FORMULA_CELL: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_MERGEABLE_REGIONS: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_MERGED_CELLS: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_MERGED_METADATA: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MIXED_TYPES: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MULTIPLE_TABLES: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_NULL_TOKEN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_NUMBER_AS_TEXT: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_OVERFULL_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_RAGGED_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_REPEATED_HEADER: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_SECTION_HEADING: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_SINGLE_COLUMN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_TOTAL_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED: DATA_QUALITY_SEVERITY_FATAL,
}

DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW: str = "candidate_header_row"
DATA_QUALITY_DETAIL_CANDIDATE_SHEET: str = "candidate_sheet"
DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE: str = "candidate_table_range"

DATA_QUALITY_NULL_TOKENS: tuple[str, ...] = (
    "",
    "na",
    "n/a",
    "null",
    "none",
    "-",
    "tbd",
    "#n/a",
)

DATA_QUALITY_EXCEL_ERROR_TOKENS: tuple[str, ...] = (
    "#ref!",
    "#div/0!",
    "#value!",
    "#name?",
    "#null!",
    "#num!",
)

DATA_QUALITY_SQL_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "all",
        "and",
        "any",
        "as",
        "between",
        "case",
        "column",
        "create",
        "date",
        "delete",
        "desc",
        "distinct",
        "drop",
        "else",
        "end",
        "except",
        "exists",
        "false",
        "from",
        "full",
        "group",
        "having",
        "in",
        "inner",
        "insert",
        "into",
        "is",
        "join",
        "left",
        "like",
        "limit",
        "not",
        "null",
        "on",
        "or",
        "order",
        "outer",
        "right",
        "select",
        "set",
        "table",
        "then",
        "true",
        "union",
        "update",
        "using",
        "values",
        "when",
        "where",
        "with",
    }
)

DATA_QUALITY_FOOTER_LABEL_PREFIXES: tuple[str, ...] = (
    "note",
    "notes",
    "source",
    "disclaimer",
    "footer",
    "comment",
    "see ",
)

DATA_QUALITY_TOTAL_PREFIXES: tuple[str, ...] = (
    "total",
    "totals",
    "subtotal",
    "grand total",
    "sum",
    "average",
    "mean",
)

DATA_QUALITY_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u20ac\u201d", "\u2014"),
    ("\u00e2\u20ac\u201c", "\u2013"),
    ("\u00e2\u20ac\u2122", "\u2019"),
    ("\u00e2\u20ac\u0153", "\u201c"),
    ("\u00e2\u20ac\u009d", "\u201d"),
    ("\u00e2\u20ac\u00a6", "\u2026"),
)

DATA_QUALITY_ZERO_WIDTH_CHARS: tuple[str, ...] = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060")

CSV_IDENTIFIER_NAMING_SYSTEM: str = (
    "You propose concise snake_case SQL identifiers for tabular upload labels. Output ONLY valid JSON "
    "matching identifier_naming_schema in the user payload. Use lowercase letters, digits, and "
    "underscores only. Start with a letter. Keep names short but readable. Do not invent business "
    "meaning beyond the supplied label text."
)

CSV_IDENTIFIER_NAMING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["identifier"],
    "additionalProperties": False,
    "properties": {"identifier": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}},
}


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
UPLOAD_SAMPLE_MAX_ROWS: int = 5
UPLOAD_INTERPRET_MAX_ROWS: int = 25
UPLOAD_BAND_VALUE_MAP_MAX_DISTINCT: int = 25
REVIEW_GATED_UPLOAD_COLUMN_TRANSFORMS: frozenset[str] = frozenset(
    {
        "keep_canonical_columns",
        "drop_empty_columns",
        "unpivot_columns",
    }
)

UPLOAD_COLUMN_TRANSFORMS_SYSTEM: str = (
    "You propose column transforms for one tabular upload relation. Output ONLY valid JSON matching "
    "column_transforms_schema in the user payload. Use transform_id values from upload_transform_ids "
    f"in the user payload ({', '.join(UPLOAD_COLUMN_TRANSFORM_IDS)}). Each proposal must name the "
    "target column label when required and supply params fields defined in the schema. Mark "
    "requires_review true for shape-changing transforms. Proposals are verified deterministically on "
    "the full column before apply; invalid proposals are rejected without changing data."
)

UPLOAD_COLUMN_TRANSFORMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["column_transforms"],
    "additionalProperties": False,
    "properties": {
        "column_transforms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["transform_id"],
                "additionalProperties": False,
                "properties": {
                    "transform_id": {"type": "string", "enum": list(UPLOAD_COLUMN_TRANSFORM_IDS)},
                    "column": {"type": "string"},
                    "requires_review": {"type": "boolean"},
                    "params": {"type": "object"},
                },
            },
        },
    },
}

UPLOAD_SUMMARY_SYSTEM: str = (
    "You summarize structured upload inspection findings for an operator. Output ONLY valid JSON with "
    "one summary field. Use issue codes, locations, severities, and suggested_selections from the "
    "user payload. Do not invent row values or business meaning beyond the supplied findings."
)

UPLOAD_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary"],
    "additionalProperties": False,
    "properties": {"summary": {"type": "string"}},
}

UPLOAD_INTERPRET_SYSTEM: str = (
    "You propose layout interpretation for one ambiguous tabular upload. Output ONLY valid JSON "
    "matching upload_interpret_schema in the user payload. Suggest header_row, table_range, "
    "append_regions, or merge_regions when structural scoring is inconclusive. Use only fields "
    "present in the schema. Proposals are verified against the grid before use."
)

UPLOAD_INTERPRET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "header_row": {"type": "integer"},
        "table_range": {"type": "string"},
        "append_regions": {"type": "array", "items": {"type": "string"}},
        "merge_regions": {"type": "array", "items": {"type": "string"}},
    },
}

UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS: frozenset[str] = frozenset(
    {
        "CSV_IDENTIFIER_NAMING_SYSTEM",
        "UPLOAD_COLUMN_TRANSFORMS_SYSTEM",
        "UPLOAD_INTERPRET_SYSTEM",
        "UPLOAD_SUMMARY_SYSTEM",
    }
)

CSV_SCHEMA_LITERAL_ORIGINAL_NAME_NOTE: str = (
    "For file-sourced tables, name is the normalized SQL identifier and original_name is the label "
    "from the uploaded file."
)

AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD: str = "write_queue_feedback_record"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT: str = "write_queue_template_reject"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT: str = "write_queue_template_accept"
AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL: str = "write_queue_override_proposal"
AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER: str = "federation_semijoin_key_transfer"
AUDIT_EVENT_ASK_SUSPEND: str = "ask_suspend"
AUDIT_EVENT_ASK_CANCELLED: str = "ask_cancelled"

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

ARTIFACT_FORMAT_VERSION: str = "0.2.1"
KNOWLEDGE_EXPORT_FORMAT_VERSION: str = "0.2.1"
META_ANSWER_FORMAT_VERSION: str = "0.2.1"
META_ANSWERS_FILENAME: str = "meta_answers.json"
META_DEFAULT_SOURCE_ID: str = "default"
META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE: str = "No business knowledge entries are configured for this engine or space."
NAMED_SCHEMA_CONTEXT_ARTIFACT_VERSION: str = "0.2.1"
NAMED_SCHEMA_CONTEXT_PREFIX: str = "schema_context."
MIN_COMPATIBLE_PACKAGE_VERSION: str = "0.2.1"
ARTIFACT_MANIFEST_FILENAME: str = "artifact_manifest.json"
RENAME_HISTORY_FILENAME: str = "rename_history.json.gz"
ARTIFACT_LOCK_FILENAME: str = ".aetherdialect_engine.lock"

SCHEMA_OVERRIDES_DEFAULT_FILENAME: str = "schema_overrides.json"
MIGRATION_MAP_FILENAME: str = "schema_migration_map.json"
MIGRATION_CHECKPOINT_SCHEMA_BASENAME: str = "schema_graph.json.gz"
WRITE_QUEUE_FILENAME: str = "write_queue.jsonl"

SCHEMA_INSTRUCTION_LIKE_LINE_PATTERNS: tuple[str, ...] = (
    r"(?i)^\s*ignore\s+(all\s+)?(prior|previous)\b",
    r"(?i)^\s*disregard\s+(all\s+)?(prior|previous)\b",
    r"(?i)^\s*system\s*:",
    r"(?i)^\s*assistant\s*:",
    r"(?i)^\s*you\s+are\b",
    r"(?i)^\s*always\s+filter\b",
    r"(?i)^\s*never\s+reveal\b",
    r"(?i)^\s*do\s+not\s+follow\b",
)
SCHEMA_INSTRUCTION_SCRUB_REPLACEMENT: str = "[scrubbed]"

MIGRATION_MAP_ACTION_REMAP: str = "remap"
MIGRATION_MAP_ACTION_DESTRUCTIVE: str = "destructive"
MIGRATION_MAP_ACTION_ABORT: str = "abort"

ARTIFACT_LAST_ACTION_REMAP_USER_MAP: str = "remap_user_map"
ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP: str = "destructive_user_map"

SCHEMA_OVERRIDES_APPLIED_SUFFIX: str = ".applied.json"
SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"
SCHEMA_OVERRIDES_SIDECAR_FILENAME: str = "applied_overrides.json"
SCHEMA_OVERRIDES_VERSION: str = "0.2.1"
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

SUSPEND_ID_TO_SESSION_KIND: dict[str, str] = {
    PIPELINE_SUSPEND_ID_DIRECT_REUSE: SESSION_KIND_AWAITING_REUSE_CONFIRM,
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
SCHEMA_CONTEXT_CACHE_VERSION: str = "0.2.1"
SCHEMA_JOIN_PATH_ENUMERATION_VERSION: str = "0.2.1"

AETHERSPACE_ARTIFACT_VERSION: str = "0.2.1"
AETHERSPACES_SEGMENT: str = "aetherspaces"
MASTER_AETHERSPACE_NAME: str = "master"
CANONICAL_FEEDBACK_DIALECT: str = "duckdb"

TEMPLATE_STORE_SEGMENT: str = "intent_templates"
TEMPLATE_STORE_SPACES_SEGMENT: str = "spaces"
SCHEMA_CONTEXT_NAMED_SPEC_GLOB: str = "schema_context.*.json"
TEMPLATE_STORE_HEADER_FILENAME: str = "header.json.gz"
TEMPLATE_STORE_PARTITION_PREFIX: str = "partition_"
TEMPLATE_STORE_FEEDBACK_SEGMENT: str = "feedback"
TEMPLATE_STORE_FEEDBACK_PARTITION_PREFIX: str = "partition_"
FEEDBACK_SHARD_INDEX_KEY: str = "feedback_shard_index"
TEMPLATE_STORE_PARTITION_COUNT: int = 256
TEMPLATE_STORE_PARTITION_LRU_MAX: int = 32
TEMPLATE_VALUE_HISTORY_MAX_ROWS: int = 64
TEMPLATE_STORE_LEGACY_SINGLE_FILE: str = "intent_templates.json.gz"
TEMPLATE_STORE_ORPHANED_SEGMENT: str = "orphaned"
MIGRATION_CHECKPOINT_PREFIX: str = ".migration_checkpoint_"
ORPHAN_RETENTION_SECONDS: int = 7 * 24 * 3600
DIAGNOSTIC_CODE_TEMPLATE_STORE_ORPHANED: str = "TEMPLATE_STORE_ORPHANED"
DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED: str = "TEMPLATE_REMAP_DIVERGED"
DIAGNOSTIC_CODE_MIGRATION_CHECKPOINT_ORPHANED: str = "MIGRATION_CHECKPOINT_ORPHANED"
DIAGNOSTIC_CODE_ARTIFACT_GROWTH: str = "ARTIFACT_GROWTH"
DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR: str = "ARTIFACT_LIMIT_NEAR"

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
    "qsim/summary_*.json",
    "qsim/index.jsonl",
    f"{WARMUP_ANCHOR_LATTICE_SUBDIR}/*",
)

SOFT_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        "explain_seq_scan_indexed",
        "explain_zero_estimate",
    }
)

DETERMINISTIC_REPAIR_TRACE_HEADINGS: dict[str, str] = {
    "dedup_extract_year_vs_column_literal": "intent_after_deterministic_repair.dedup_extract_year_vs_column_literal",
    "repair_intent_placeholder_tokens": "intent_after_deterministic_repair.repair_intent_placeholder_tokens",
    "normalize_count_star": "intent_after_deterministic_repair.normalize_count_star",
    "promote_temporal_keyword_rhs": "intent_after_deterministic_repair.promote_temporal_keyword_rhs",
    "dedup_value_vs_right_expr": "intent_after_deterministic_repair.dedup_value_vs_right_expr",
    "qualify_count_star_mulgroups": "intent_after_deterministic_repair.qualify_count_star_mulgroups",
    "lift_distinct_select_from_raw_sql": "intent_after_deterministic_repair.lift_distinct_select_from_raw_sql",
    "canonicalize_registry_ids": "intent_after_deterministic_repair.canonicalize_registry_ids",
    "encode_inline_self_join_as_cte": "intent_after_deterministic_repair.encode_inline_self_join_as_cte",
    "reorder_cte_steps_by_dag": "intent_after_deterministic_repair.reorder_cte_steps_by_dag",
    "normalize_cte_names": "intent_after_deterministic_repair.normalize_cte_names",
    "normalize_cte_output_aliases": "intent_after_deterministic_repair.normalize_cte_output_aliases",
    "rewrite_main_query_refs_to_final_cte_columns": (
        "intent_after_deterministic_repair.rewrite_main_query_refs_to_final_cte_columns"
    ),
    "ensure_cte_output_columns_exposure": "intent_after_deterministic_repair.ensure_cte_output_columns_exposure",
    "qualify_cte_output_columns": "intent_after_deterministic_repair.qualify_cte_output_columns",
    "derive_tables_from_intent": "intent_after_deterministic_repair.derive_tables_from_intent",
    "expand_shared_pk_tables_for_refs": "intent_after_deterministic_repair.expand_shared_pk_tables_for_refs",
    "sanitize_table_names": "intent_after_deterministic_repair.sanitize_table_names",
    "replace_unknown_scalar_funcs": "intent_after_deterministic_repair.replace_unknown_scalar_funcs",
    "enforce_grain_consistency": "intent_after_deterministic_repair.enforce_grain_consistency",
    "repair_window_partition_group_by_alignment": (
        "intent_after_deterministic_repair.repair_window_partition_group_by_alignment"
    ),
    "strip_redundant_identifier_group_by": "intent_after_deterministic_repair.strip_redundant_identifier_group_by",
    "strip_spurious_group_by": "intent_after_deterministic_repair.strip_spurious_group_by",
    "decompose_between_params": "intent_after_deterministic_repair.decompose_between_params",
    "auto_repair_where_having": "intent_after_deterministic_repair.auto_repair_where_having",
    "coerce_predicate_group_mode": "intent_after_deterministic_repair.coerce_predicate_group_mode",
    "normalize_where_havings": "intent_after_deterministic_repair.normalize_where_havings",
    "repair_null_equality_where": "intent_after_deterministic_repair.repair_null_equality_where",
    "strip_join_conditions": "intent_after_deterministic_repair.strip_join_conditions",
    "cte_grain_consistency": "intent_after_deterministic_repair.cte_grain_consistency",
    "sort_select_and_order_by": "intent_after_deterministic_repair.sort_select_and_order_by",
    "lift_distinct_modifier_in_multiply": "intent_after_deterministic_repair.lift_distinct_modifier_in_multiply",
    "simplify_exprs": "intent_after_deterministic_repair.simplify_exprs",
    "normalize_in_raw_values": "intent_after_deterministic_repair.normalize_in_raw_values",
    "promote_date_subtraction_to_date_diff": "intent_after_deterministic_repair.promote_date_subtraction_to_date_diff",
    "repair_misclassified_date_diff": "intent_after_deterministic_repair.repair_misclassified_date_diff",
    "normalize_date_diff_raw_values": "intent_after_deterministic_repair.normalize_date_diff_raw_values",
    "canonicalize_temporal_unit_args": "intent_after_deterministic_repair.canonicalize_temporal_unit_args",
    "strip_impossible_having": "intent_after_deterministic_repair.strip_impossible_having",
    "repair_fk_where_type_mismatch": "intent_after_deterministic_repair.repair_fk_where_type_mismatch",
    "resolve_where_value_case": "intent_after_deterministic_repair.resolve_where_value_case",
    "normalize_in_where_types": "intent_after_deterministic_repair.normalize_in_where_types",
    "normalize_boolean_where_values": "intent_after_deterministic_repair.normalize_boolean_where_values",
    "normalize_null_where_values": "intent_after_deterministic_repair.normalize_null_where_values",
    "expand_fk_select_to_descriptive": "intent_after_deterministic_repair.expand_fk_select_to_descriptive",
    "dedup_contradictory_where": "intent_after_deterministic_repair.dedup_contradictory_where",
    "repair_cumulative_phrasing_window_intent": (
        "intent_after_deterministic_repair.repair_cumulative_phrasing_window_intent"
    ),
    "repair_case_when_intent": "intent_after_deterministic_repair.repair_case_when_intent",
    "drop_invalid_case_registry_entries": "intent_after_deterministic_repair.drop_invalid_case_registry_entries",
    "repair_array_where_intent": "intent_after_deterministic_repair.repair_array_where_intent",
    "enforce_sensitivity_policy_intent": "intent_after_deterministic_repair.enforce_sensitivity_policy_intent",
    "tail_enforce_grain_consistency": "intent_after_deterministic_repair.tail_enforce_grain_consistency",
    "tail_normalize_where_havings": "intent_after_deterministic_repair.tail_normalize_where_havings",
}

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

RESULT_READER_KINDS: tuple[str, ...] = (
    "sqlalchemy",
    "spark",
    "connector",
    "bq_client",
    "bq_storage",
    "snowflake_arrow",
)

ARROW_RESULT_READER_KINDS: frozenset[str] = frozenset({"snowflake_arrow", "bq_storage"})

_INTENT_DATE_UNIT_AMOUNT_VALUE: dict[str, Any] = {
    "type": "object",
    "required": ["unit", "amount"],
    "properties": {
        "unit": {"enum": ["day", "week", "month", "quarter", "year"]},
        "amount": {"type": "integer", "minimum": 1},
    },
}

_INTENT_WHERE_ITEM_ALLOF: list[dict[str, Any]] = [
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
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "expr": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                            "direction": {"type": "string", "enum": ["asc", "desc", "ASC", "DESC"]},
                            "nulls": {"type": "string", "enum": ["first", "last"]},
                        },
                    },
                ]
            },
        },
        "where": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["and", "or"]},
                        "predicates": {"type": "array"},
                        "groups": {"type": "array"},
                    },
                },
                {"type": "array"},
                {"type": "null"},
            ]
        },
        "having": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["and", "or"]},
                        "predicates": {"type": "array"},
                        "groups": {"type": "array"},
                    },
                },
                {"type": "array"},
                {"type": "null"},
            ]
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
                    "where": {"oneOf": [{"type": "object"}, {"type": "array"}, {"type": "null"}]},
                    "having": {"oneOf": [{"type": "object"}, {"type": "array"}, {"type": "null"}]},
                    "output_columns": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                    },
                    "window_registry": {"type": "array"},
                    "case_registry": {"type": "array"},
                    "distinct_on": {"type": "array", "items": {"type": "string"}},
                    "preserve_tables": {"type": "array", "items": {"type": "string"}},
                    "emission": {
                        "type": "string",
                        "enum": ["semi_join", "anti_join"],
                    },
                },
            },
        },
        "window_registry": {"type": "array"},
        "case_registry": {"type": "array"},
        "limit": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "natural_language": {"type": "string"},
        "distinct_on": {"type": "array", "items": {"type": "string"}},
        "preserve_tables": {"type": "array", "items": {"type": "string"}},
        "grain": {
            "type": "string",
            "enum": ["row_level", "grouped", "scalar"],
        },
    },
}


LOGICAL_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tables", "select"],
    "additionalProperties": False,
    "properties": {
        "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "select": {"type": "string", "minLength": 1},
        "where": {"type": "string"},
        "group_by": {"type": "string"},
        "having": {"type": "string"},
        "order_by": {"type": "string"},
        "limit": {"oneOf": [{"type": "string"}, {"type": "null"}]},
        "window": {"type": "string"},
        "case": {"type": "string"},
        "cte_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "tables", "select"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "select": {"type": "string", "minLength": 1},
                    "where": {"type": "string"},
                    "group_by": {"type": "string"},
                    "having": {"type": "string"},
                    "order_by": {"type": "string"},
                    "limit": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "window": {"type": "string"},
                    "case": {"type": "string"},
                },
            },
        },
    },
}

PLANNER_PROSE_FIELDS: tuple[str, ...] = (
    "select",
    "where",
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
    "string_agg",
    "stddev",
    "variance",
    "median",
)

VALID_AGGREGATION_FUNCTIONS = frozenset(_AGGREGATION_FUNCTION_NAMES_ORDERED)

WINDOW_RANKING_FUNCTIONS = frozenset({"row_number", "rank", "dense_rank", "ntile", "percent_rank", "cume_dist"})

WINDOW_AGG_FUNCTIONS = frozenset({"sum", "avg"})

WINDOW_OFFSET_FUNCTIONS = frozenset({"lag", "lead"})

WINDOW_VALUE_FUNCTIONS = frozenset({"first_value", "last_value", "nth_value"})

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

FILE_ENGINE_NAMES: frozenset[str] = frozenset({"csv"})

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

UPLOAD_INGEST_ENGINE_NAMES: frozenset[str] = frozenset({"duckdb", "csv"})
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
UPLOAD_SCALAR_BAND_PATTERN_STRINGS: tuple[str, ...] = (
    r"(?i)\bto\b",
    r"(?i)\bor\s+more\b",
    r"(?i)\bless\s+than\b",
    r"(?i)\d+\s*[-–—]\s*\d+\s*%",
)
UPLOAD_SCALAR_BAND_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in UPLOAD_SCALAR_BAND_PATTERN_STRINGS
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$")
BARE_SCALAR_NUMBER_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")

UPLOAD_CURRENCY_AFFIX_TOKENS: frozenset[str] = frozenset(token for token in UPLOAD_SCALAR_AFFIX_TOKENS if token != "%")

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
    "binary",
    "unknown",
    "date_window",
    "date_diff",
}

WEEK_START_DAY: str = "monday"
WEEK_NUMBERING: str = "iso"

VALID_RELATIVE_DATE_UNITS = frozenset(
    {"day", "week", "month", "quarter", "half_year", "year", "hour", "minute", "second"}
)

SUBDAY_RELATIVE_DATE_UNITS: frozenset[str] = frozenset({"hour", "minute", "second"})

MYSQL_DATE_WINDOW_TRUNC_FORMAT: dict[str, str] = {
    "month": "%Y-%m-01",
    "year": "%Y-01-01",
}

MYSQL_DATE_WINDOW_SUBDAY_TRUNC_FORMAT: dict[str, str] = {
    "hour": "%Y-%m-%d %H:00:00",
    "minute": "%Y-%m-%d %H:%i:00",
    "second": "%Y-%m-%d %H:%i:%s",
}

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

VALID_WHERE_VALUE_TYPES = {
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
    "binary": "binary",
    "null": "null",
    "date_window": "date_window",
    "date_diff": "date_diff",
    "unknown": "unknown",
}

_BOOLEAN_WHERE_OPS = {"=", "!=", "in", "not in", "is null", "is not null"}

_CATEGORICAL_WHERE_OPS = {
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

_NUMERIC_CATEGORICAL_WHERE_OPS = {
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

_NUMERIC_WHERE_OPS = frozenset(
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

CTE_NUMERIC_WHERE_OPS = list(_NUMERIC_WHERE_OPS)

ROLE_ALLOWED_AGGREGATIONS = {
    "IDENTIFIER": {"count"},
    "CATEGORICAL": {"count", "min", "max", "string_agg"},
    "NUMERIC_CATEGORICAL": {"count", "min", "max", "string_agg"},
    "NUMERIC_MEASURE": {"count", "sum", "avg", "min", "max", "stddev", "variance", "median"},
    "TEMPORAL": {"count", "min", "max", "string_agg"},
    "BOOLEAN": {"count"},
    "FREE_TEXT": {"count", "string_agg"},
    "AUDIT": set(),
}

VALID_WHERE_OPS: frozenset[str] = frozenset(
    _BOOLEAN_WHERE_OPS | _CATEGORICAL_WHERE_OPS | _NUMERIC_CATEGORICAL_WHERE_OPS | frozenset({"contains"})
)

NUMERIC_ONLY_AGGREGATIONS = {"sum", "avg", "stddev", "variance", "median"}

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

AGGREGATION_ALLOWED_COLUMN_TYPES = {
    "count": ["integer", "string", "date", "number", "boolean"],
    "sum": ["integer", "number"],
    "avg": ["integer", "number"],
    "min": ["integer", "number", "string", "date"],
    "max": ["integer", "number", "string", "date"],
    "string_agg": ["string", "date", "integer", "number"],
    "stddev": ["integer", "number"],
    "variance": ["integer", "number"],
    "median": ["integer", "number"],
}

EXCLUDED_WHERE_PATTERNS = [
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

NUMERIC_TYPE_ARGUMENTS_RE = re.compile(r"\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", re.IGNORECASE)

SQL_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")
SQL_FIXED_POINT_LITERAL_RE = re.compile(r"^[+-]?\d+\.\d+$")
SQL_EXPONENT_LITERAL_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")

EXACT_NUMERIC_BASE_TYPES = frozenset(
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

INEXACT_NUMERIC_BASE_TYPES = frozenset(
    {
        "float",
        "real",
        "double",
        "double precision",
        "float4",
        "float8",
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

WINDOW_NUMERIC_ARG_FUNCTIONS = frozenset({"ntile", "nth_value"})

WINDOW_FUNCTIONS_WITHOUT_COLUMN_ARG = frozenset(
    {"row_number", "rank", "dense_rank", "ntile", "percent_rank", "cume_dist"}
)

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


WHERE_ADD = "WHERE_ADD"
WHERE_EXPR_ADD = "WHERE_EXPR_ADD"
AGG_CHANGE = "AGG_CHANGE"
GROUPBY_ADD = "GROUPBY_ADD"
ORDERBY_ADD = "ORDERBY_ADD"
HAVING_VALUE_ADD = "HAVING_VALUE_ADD"
HAVING_EXPR_ADD = "HAVING_EXPR_ADD"
WHERE_REMOVE = "WHERE_REMOVE"
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
TEMP_DATE_WINDOW_WHERE = "TEMP_DATE_WINDOW_WHERE"
TEMP_DATE_DIFF_WHERE = "TEMP_DATE_DIFF_WHERE"
NUM_ROUND_SELECT = "NUM_ROUND_SELECT"
NUM_ABS_WHERE = "NUM_ABS_WHERE"
DISTINCT_ADD = "DISTINCT_ADD"
LIMIT_ADD = "LIMIT_ADD"
WHERE_OR_GROUP = "WHERE_OR_GROUP"
SELECT_EXPR_PAIR_MULTIPLY = "SELECT_EXPR_PAIR_MULTIPLY"
WINDOW_RANK_ADD = "WINDOW_RANK_ADD"
WINDOW_SUM_PARTITION_ADD = "WINDOW_SUM_PARTITION_ADD"
SELECT_CASE_LABEL_ADD = "SELECT_CASE_LABEL_ADD"
WINDOW_LAG_ADD = "WINDOW_LAG_ADD"
WINDOW_LEAD_ADD = "WINDOW_LEAD_ADD"
WHERE_ILIKE_ADD = "WHERE_ILIKE_ADD"
WHERE_ARRAY_CONTAINS_ADD = "WHERE_ARRAY_CONTAINS_ADD"
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
WHERE_IN_LIST_ADD = "WHERE_IN_LIST_ADD"
WHERE_NULL_ADD = "WHERE_NULL_ADD"
WHERE_NOT_NULL_ADD = "WHERE_NOT_NULL_ADD"
HAVING_MATCH_SELECT_AGG = "HAVING_MATCH_SELECT_AGG"
COUNT_DISTINCT_ADD = "COUNT_DISTINCT_ADD"
WINDOW_DENSE_RANK_ADD = "WINDOW_DENSE_RANK_ADD"
WINDOW_RANK_FUNC_ADD = "WINDOW_RANK_FUNC_ADD"
WINDOW_AVG_PARTITION_ADD = "WINDOW_AVG_PARTITION_ADD"
ORDERBY_WINDOW_COL_ADD = "ORDERBY_WINDOW_COL_ADD"
WHERE_LIKE_ADD = "WHERE_LIKE_ADD"
SELECT_COALESCE_ADD = "SELECT_COALESCE_ADD"
SELECT_STRING_SCALAR_ADD = "SELECT_STRING_SCALAR_ADD"
TEMP_EXTRACT_WHERE = "TEMP_EXTRACT_WHERE"
CTE_UNNEST_ADD = "CTE_UNNEST_ADD"
SELF_JOIN_CTE_ADD = "SELF_JOIN_CTE_ADD"
MULTI_CTE_CHAIN_ADD = "MULTI_CTE_CHAIN_ADD"
SPLICE_HAVING_SUBTREE = "SPLICE_HAVING_SUBTREE"
SPLICE_WINDOW_SUBTREE = "SPLICE_WINDOW_SUBTREE"

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

LLM_JSON_ONLY_FOOTER: str = "Respond ONLY with valid JSON, no explanation."

LLM_PRESERVE_ANALYTICAL_CONTENT: str = (
    "Preserve all entities, filters, metrics, grouping, ordering, and limits implied by the source. "
    "Do not add or remove requirements."
)

QUESTION_NORMALIZE_VOCABULARY_HEADING: str = (
    "Vocabulary preferences (apply only when context fits; preserve logical negation and constraints):"
)
QUESTION_NORMALIZE_VOCABULARY_GUIDANCE: str = (
    "Verb phrasing: prefer concise analytic wording over vague conversational fillers whenever the question intent is unchanged. "
    "Aggregation: align stated aggregation language with the grain implied by the question. "
    "Temporal: when any time scope appears, state it explicitly enough for deterministic routing. "
    "Negation: preserve every explicit negator on the predicate it scopes; do not drop or soften negation."
)

QUESTION_VALIDATION_SYSTEM: str = (
    "You decide if user input is a database query request or not, and which answer path it needs.\n\n"
    "Treat the input as a VALID database query request whenever a reasonable relational database could store rows "
    "that answer it, OR when it asks about this database's schema structure (tables, columns, types, keys, "
    "relationships, inventory or counts of tables/columns/members, federation membership), OR when it asks for "
    "definitions, policies, metrics, synonyms, or caveats answerable from business knowledge.\n"
    "That includes list, show, get, find, count, sum, average, min, max, filter, sort, group, compare, rank, "
    "top-N, trend, or per-entity questions, including bounded counts such as listing two named entities.\n"
    "When the utterance is ambiguous but still plausibly data-seeking, choose VALID with query_type analytical.\n\n"
    "Mark as INVALID only when it is clearly one of the following:\n"
    '- Chitchat or meta conversation (e.g. "hello", "thanks", "who are you")\n'
    '- A request for SQL tutoring, query help, or how-to without asking for actual rows (e.g. "how do I write a join")\n'
    '- General world knowledge or opinion with no plausible tabular backing (e.g. "does the Earth orbit the Sun")\n\n'
    "The label restricted applies only when the user asks for a DML, DDL, or administrative database operation. "
    "DML covers any data mutation (delete, update, insert, merge, truncate, copy). "
    "DDL covers schema mutation (create, drop, alter, rename). "
    "Administrative covers privilege management, indexing, vacuuming, configuration, and any other non-analytical operation. "
    "Analytical questions never receive restricted, including questions that describe their solution using analytical "
    "primitives such as CTEs, subqueries, joins, aggregations, window functions, distinct, recursion, or set operations. "
    "Use of the literal words CTE, subquery, with, join, group, order, window, partition, recursive, or similar terms "
    "in the question never alone implies restricted.\n\n"
    "When valid_database_question is yes, set query_type to exactly one of:\n"
    "- analytical: the user wants data rows or aggregates from tables\n"
    "- schema_catalog: the user asks about this schema's structure, inventory, counts of tables/columns/members, "
    "or federation membership inventory\n"
    "- business_knowledge: the user asks for glossary, policy, metric, synonym, or caveat definitions from "
    "business knowledge\n"
    "When valid_database_question is no and the request is not restricted, set query_type to unspecified.\n\n"
    "Respond with JSON containing exactly three fields:\n"
    '- "valid_database_question": "yes" or "no"\n'
    '- "query_type": "analytical", "schema_catalog", "business_knowledge", "restricted", or "unspecified"\n'
    '- "corrected": the input with spelling typos fixed only. Do NOT remove, reorder, or rephrase any words.\n\n'
    f"{LLM_JSON_ONLY_FOOTER}"
)

META_SCHEMA_CATALOG_SYSTEM: str = (
    "You answer questions about the active database schema using only the JSON `schema` object in the user message.\n"
    "Never invent table names, column names, source ids, or relationships that are absent from `schema`.\n"
    "Copy inventory counts only from `schema.inventory` and `schema.members` into the `counts` fields; "
    "do not recount by scanning table lists or prose.\n"
    "Leave a `counts` field null when the question does not ask for that metric.\n"
    "Leave `tables` and `relationships` as empty arrays for pure count or inventory questions that need no detail list.\n"
    "Return JSON matching META_SCHEMA_ANSWER_SCHEMA exactly "
    "(response_kind, headline, counts, tables, relationships, notes).\n"
    f"{LLM_JSON_ONLY_FOOTER}"
)

META_BUSINESS_KNOWLEDGE_SYSTEM: str = (
    "You answer questions using only the JSON `business_knowledge` list in the user message.\n"
    "Each entry has key, kind, and text. Answer only from that list; never invent terms.\n"
    "If the list is empty, set `message` to exactly the configured empty-knowledge reply "
    f"({META_EMPTY_BUSINESS_KNOWLEDGE_MESSAGE!r}).\n"
    "Return JSON matching META_KNOWLEDGE_ANSWER_SCHEMA exactly "
    "(response_kind business_knowledge, message as natural-language prose).\n"
    f"{LLM_JSON_ONLY_FOOTER}"
)

QUESTION_CANONICALIZE_SYSTEM: str = (
    "You rewrite a typo-corrected database query into a canonical short query so that semantically identical "
    "questions hash to the same string.\n\n"
    "When the user message is JSON, field ``question`` carries the rewrite target; optional ``normalization_preferences`` "
    "is advisory context only.\n\n"
    "Apply these rules IN ORDER:\n"
    "0. Before any other rewrite, normalize quantifier and aggregation openers: map phrases such as "
    '"how many", "number of", and bare "count" asking for cardinality to the two-token prefix "count of"; '
    'map "total of", "totals for", and bare "sum" used as an aggregation opener to "sum of"; '
    'map "average of", "mean of", and bare "avg" used as an aggregation opener to "avg of"; '
    'map "maximum of", "largest", "highest", "max" used as an aggregation opener to "max of"; '
    'map "minimum of", "smallest", "lowest", "min" used as an aggregation opener to "min of". '
    "Preserve trailing nouns and filters after those prefixes.\n"
    '1. Replace any verb phrase whose only purpose is to ask for non-aggregated rows with the single token "list"; '
    "do not replace the aggregation prefixes introduced in rule 0, and do not replace aggregation verbs "
    "such as count, sum, average, max, min, or total when they already head a normalized aggregation phrase.\n"
    "2. Drop polite or filler clauses that do not carry analytical meaning.\n"
    "3. Replace plural common nouns with their singular base form. Do NOT singularize verbs.\n"
    "4. Preserve every number, date, quoted literal, comparison word, adjective, named entity, "
    "and any preposition immediately before a number/date/literal.\n"
    "5. Preserve original word order.\n"
    "6. If no rule applies, return the input unchanged.\n"
    "7. Never add a word that did not appear in the input (including any inflected form already present).\n\n"
    "Examples of rule 0 (JSON only illustrates the normalized field):\n"
    '{"question":"how many films are in the action category","normalized":"count of film in the action category"}\n'
    '{"question":"total payments last month by store","normalized":"sum of payment last month by store"}\n\n'
    "Respond with JSON containing exactly one field:\n"
    '- "normalized": the rewritten canonical short query.\n\n'
    f"{LLM_JSON_ONLY_FOOTER}"
)

WARMUP_PARAPHRASES_BY_STYLE_SYSTEM: str = (
    "Generate natural-language analyst questions grouped by stylistic palette slots. "
    f"{LLM_PRESERVE_ANALYTICAL_CONTENT} "
    "Use schema descriptions only for terminology consistency. "
    "Do not answer the question. Do not output SQL, identifiers, numbered steps, or JOIN recipes. "
    "For each style slot you may return zero paraphrases when that style does not fit; never force a poor match. "
    "Output ONLY valid JSON with field paraphrases_by_style mapping each style name to an array of strings."
)

WARMUP_FREEFORM_QUESTIONS_SYSTEM: str = (
    "You write natural-language analyst questions that faithfully describe what a SQL query computes. "
    "Use schema table descriptions only for business terminology. "
    "Do not output SQL, identifiers, numbered steps, or JOIN recipes. "
    "Return 1–3 concise questions in a JSON object with field questions as an array of strings."
)

SEED_QUESTION_CLARIFY_SYSTEM: str = (
    "You rephrase database analyst questions for clarity only. Do not answer them. "
    f"{LLM_PRESERVE_ANALYTICAL_CONTENT} "
    "Do not use SQL or qualified identifiers unless the source already does. "
    'Output only valid JSON: {"lines":[{"index":<int>,"clarified":"<string>"}]} with exactly one object '
    "per input index, indices matching the batch, no extra keys, no markdown."
)

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

VALID_FK_ADD_KEYS: frozenset[str] = frozenset(
    {"from", "to", "kind", "authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

VALID_FK_REMOVE_KEYS: frozenset[str] = frozenset({"from", "to"})

VALID_PK_ADD_KEYS: frozenset[str] = frozenset(
    {"table", "column", "authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

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

SANDBOX_QUESTION_TIERS: tuple[str, ...] = (
    "questions",
    "validation_failures",
    "views_questions",
)
_RENTAL_SHOP_BUNDLE_MEMBERS: tuple[str, ...] = (
    "rental_shop_seed.sql",
    "rental_shop.sql",
    "rental_shop_views.sql",
    "rental_shop_notes.txt",
    "fixtures/rental_shop_mock.json",
)
SANDBOX_FIXTURE_ALIASES: dict[str, str] = {
    "notes.txt": "rental_shop_notes.txt",
    "catalog_notes.txt": "rental_shop_notes.txt",
    "schema.sql": "rental_shop.sql",
}
RENTAL_SHOP_VIEW_NAMES: tuple[str, ...] = ("active_customer_v", "store_revenue_v", "film_catalog_v")
SANDBOX_DOCTOR_REQUIRED_MEMBERS: tuple[str, ...] = (
    *_RENTAL_SHOP_BUNDLE_MEMBERS,
    "questions.txt",
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
SANDBOX_BASELINE_CACHE_FILES: tuple[str, ...] = (
    "schema_graph.json.gz",
    "artifact_manifest.json",
    "schema_context.json",
)
SANDBOX_LEGACY_FAITHFULNESS_SPECS: dict[str, dict[str, object]] = {
    "which games support english?": {
        "sql_contains": ("game_supported_language",),
        "contains_join": True,
    },
    "which city has the most customers?": {
        "required_tables": ("city", "customer"),
        "contains_join": True,
    },
    "how many customers are in each country?": {
        "required_tables": ("country", "customer"),
        "contains_join": True,
    },
    "film title and replacement cost minus rental rate as profit margin": {
        "sql_excludes": ("interval",),
    },
    "what is the best pizza topping?": {"status": "invalid_question"},
    "what's the weather today?": {"status": "invalid_question"},
}
SANDBOX_DOCTOR_OPTIONAL_BASELINE_DIRS: tuple[str, ...] = (
    "artifacts_baseline/owner_views",
    "artifacts_baseline/consumer_views",
)
SANDBOX_DOCTOR_OPTIONAL_BASELINE_MEMBERS: tuple[str, ...] = (
    "artifacts_baseline/owner_views/schema_graph.json.gz",
    "artifacts_baseline/consumer_views/schema_graph.json.gz",
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
SANDBOX_MIN_FIXTURE_COUNT = 100
SANDBOX_MIN_INTENT_FIXTURE_COUNT = 50
SANDBOX_UNEXERCISED_PRODUCTION_STAGES: tuple[str, ...] = (
    "live_reflection_and_profiling",
    "probe_mismatch_partial_rebuild",
    "cold_build_descriptions_and_classification",
    "member_cold_reflect_profile_and_member_drift_migration_pending",
    "warmup_and_question_simulation",
    "model_turns_outside_recorded_fixtures",
)
SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS: tuple[str, ...] = (
    "Show rentals with malformed compose output for repair exercise.",
)
SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS: tuple[dict[str, object], ...] = (
    {
        "question": "Show rentals with malformed compose output for repair exercise.",
        "malformed_output": '{"tables":["rental"],"select":"count rentals","filter":}',
        "repair_output": '{"tables":["rental"],"select":"count rental rows","filter":"","group_by":"","having":"","order_by":"","limit":null,"window":"","case":""}',
        "repair_stage": "compose",
    },
)
SANDBOX_SCHEMA_LITERALS_FILENAME = "schema_literals.json"
SANDBOX_INTERPRET_DOMAIN_FILENAME = "schema_interpret_domain.json"
SANDBOX_DEFAULT_DATASET_NAME = "main"
SANDBOX_BUNDLED_MEMBER_SEEDS: tuple[tuple[str, str], ...] = (
    ("storefront", "federation_storefront_seed.sql"),
    ("catalog", "federation_catalog_seed.sql"),
    ("logistics", "federation_logistics_seed.sql"),
    ("crm", "federation_crm_seed.sql"),
)
SANDBOX_BUNDLED_DATASET_NAMES: frozenset[str] = frozenset(
    {SANDBOX_DEFAULT_DATASET_NAME, *(name for name, _ in SANDBOX_BUNDLED_MEMBER_SEEDS)}
)
SANDBOX_CONNECTION_HOST_ATTR = "_aether_sandbox_host"

MOCK_FIXTURE_STUB_SCHEMA_LITERALS: dict[str, str] = {"owner": "{}", "consumer": "{}"}

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

CONSUMER_EXAMPLE_NARROW_ALLOW_OBJECTS: frozenset[str] = frozenset(
    {
        "customer",
        "payment",
        "rental",
        "address",
        "city",
        "country",
    }
)

SANDBOX_CATALOG_SPACE_TABLES: frozenset[str] = frozenset(
    {
        "item",
        "film",
        "category",
        "item_category",
    }
)

WINDOW_ADD_OPS = frozenset(
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

CASE_ADD_OPS = frozenset(
    {
        SELECT_CASE_LABEL_ADD,
        CASE_CATEGORICAL_ADD,
    }
)

CTE_ADD_OPS = frozenset(
    {
        CTE_WRAP_GROUPED,
        CTE_SCALAR_THRESHOLD,
        CTE_UNNEST_ADD,
        SELF_JOIN_CTE_ADD,
        MULTI_CTE_CHAIN_ADD,
    }
)

HAVING_ADD_OPS = frozenset(
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
    "gpt-4.1-mini": "deployment_light",
    "gpt-4.1-nano": "deployment_light",
    "gpt-5-nano": "deployment_light",
    "gpt-5-mini": "deployment_light",
    "gpt-5.4-nano": "deployment_light",
    "gpt-5.4-mini": "deployment_heavy",
}

TASK_PROFILES: dict[str, dict[str, Any]] = {
    "intent": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "intent_format": {
        "temperature": 0,
    },
    "intent_schema_repair": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "feedback": {
        "temperature": 0,
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
    "synth": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "synth_variety": {
        "reasoning": {"effort": "minimal", "summary": "concise"},
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
        "cross_source",
        "suffix",
        "self",
        "composite",
        "semantic",
        "semantic_promoted",
        "user_override_structural",
        "user_override_semantic",
    }
)

PK_INFERENCE_TAG_VALUES: frozenset[str] = frozenset({"ddl", "identity", "profile", "user_override"})

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
        "existence_where_encoded_as_subquery": "compose",
        "self_reference_encoded_as_inline_self_join": "compose",
        "correlated_lookup_encoded_as_lateral": "compose",
    }
)

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

CASE_WHEN_QUALIFIED_COLUMN_REF_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$",
)

DESCRIPTION_OWNER_VALUES: frozenset[str] = frozenset(
    {"catalog", "profile", "notes", "llm_refinement", "space_notes", "user_override"}
)

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
    f"Concretely, suppose the intent connects `{INSTRUCTIONAL_TABLE_PLACEHOLDER}` to `{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` and the descriptions indicate two semantic "
    f"paths: `{INSTRUCTIONAL_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` for one semantic and "
    f"`{INSTRUCTIONAL_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` for another. "
    f"If the question requires the second semantic, add at least one column from `{INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}` or one column from "
    f"`{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}` (preferably to `select_cols`, ideally a primary key or the most descriptive column from each). "
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
        f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}) or other supported "
        "date functions for time differences."
    ),
    "grain_validity": "Use one of the allowed grain values: 'scalar', 'grouped', or 'row_level'.",
    "grain_consistency": "Ensure grain matches the query structure. 'grouped' requires group_by_cols and aggregation in select_cols. 'row_level' means no GROUP BY and no aggregation. 'scalar' means a single aggregated value with no GROUP BY.",
    "schema_validation": "One or more tables or columns do not exist in the schema. Check allowed_tables and use only exact column names from each table.",
    "unknown_table": "The table does not exist in the schema. Remove it from tables and rewrite any references to use only tables that appear in allowed_tables.",
    "unknown_column": "The column does not exist in its table. Check the schema for available columns with a similar name or meaning and replace the reference.",
    "semantic_contradiction": "The intent contains contradictory operations. Keep only the aggregation or pattern that matches the question.",
    "expression_type": "Arithmetic expressions require numeric columns. Ensure all operands in arithmetic and all comparison sides have compatible types.",
    "where_aggregation": "Conditions with aggregation functions (COUNT, SUM, AVG, MIN, MAX) belong in having, not where. Move the condition.",
    "having_aggregation": "HAVING conditions must have an aggregation function in left_expr. Conditions without aggregation belong in where.",
    "where_semantic": "Fix the filter comparison: remove self-comparisons and ensure type compatibility between left and right expressions.",
    "having_semantic": "Fix the HAVING comparison: remove self-comparisons and ensure type compatibility between aggregated expressions.",
    "nested_aggregation": "Nested aggregation is not allowed. Use a CTE: compute the inner aggregation in a CTE step, then aggregate the CTE output in the main query.",
    "mixed_aggregation": "An expression cannot mix aggregated and bare column terms. Either wrap all terms in an aggregation function or add bare columns to group_by_cols.",
    "group_by_membership": "Every non-aggregated column in select_cols must appear in group_by_cols when grain is 'grouped'. Add the missing column to group_by_cols or wrap it in an aggregation.",
    "order_by_aggregation": "ORDER BY cannot contain aggregation when grain is 'row_level'. Change grain to 'grouped' or remove the aggregation from order_by_cols.",
    "aggregation_hint": "The question contains a quantity-comparison phrase that typically requires COUNT or SUM aggregation with GROUP BY and HAVING. Add aggregation in select_cols, group_by_cols on the entity, having with the threshold, and set grain to 'grouped'.",
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
        "so the start/end boundary matches the schema's daily granularity. "
        'For absolute calendar bounds use {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} '
        "in ISO 8601 form only."
    ),
    "date_diff": (
        "For date-difference filters comparing two date columns "
        f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} compared to a duration), use value_type 'date_diff' with "
        "left_expr as the date subtraction expression (later minus earlier per the question), "
        "op as the comparison operator, and value as "
        '{"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. Prefer unit 'day' whenever the question phrases the duration in "
        "days or weeks. Do NOT use date_diff for relative date-window "
        "filters; use date_window instead."
    ),
    "date_integer_days": (
        "For date-shift arithmetic comparing a date column shifted by an integer day count to another date column, express "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} or "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} directly in left_expr/right_expr using "
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
    "wrong_join": (
        "The referenced tables cannot be joined through the schema. Do not resolve join errors by "
        "removing a table from tables — that answers a different question. If the tables are "
        "genuinely related, add foreign_keys_add or a semantic neighbour override. Otherwise keep "
        "both tables and let the turn refuse."
    ),
}

PLANNER_NL_CONVENTIONS_BODY: dict[str, Any] = {
    "mandatory": [
        "Copy every literal that constrains rows or ordering into the matching prose field; the encoder never sees the original question.",
        f"List only semantic base tables in top-level tables; omit {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables when its columns appear in prose.",
        "Reference window and case output names from select when you define them in window or case using as <registry_name>.",
        "Never emit SQL set operators, EXISTS, NOT EXISTS, LATERAL, param_key, raw_value, wNN, cNN, where_group, or IR vocabulary.",
        "Never use as <name> inside select, filter, having, group_by, order_by, or limit; select output aliases are assigned later by the pipeline.",
        "Leave group_by, order_by, and limit empty unless the question explicitly asks for grouping, ordering, or a row cap; do not invent presentation-layer sorting or grouping for context.",
        "Project select prose to primary keys, primary human-readable label columns such as names or titles, and every column referenced by stated filters, ordering, grouping, having, or limits; never enumerate every physical column unless the question explicitly asks for all columns, every column, or complete row dumps.",
        f"After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} in the appropriate prose field.",
        f"The tables list alone never keeps a table in join scope; qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens must appear in select, filter, group_by, having, order_by, or registry prose.",
        f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
        f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
        f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
        "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
        "Existence or membership conditions belong in filter prose with binding columns named.",
        "Copy relative time unit and amount into filter prose; copy two-date duration comparisons into filter prose naming both columns.",
        "For cte_steps, tables lists base schema tables and names of prior cte_steps this step reads from; do not use a separate dependency field.",
    ],
    "recommended": [
        f"Qualify columns as {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} when multiple listed tables share a column name.",
        f"Describe aggregates with phrases such as sum of {INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER}.",
        f"Describe anti-existence with plain language or {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}.pk is null style wording.",
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
    f"Emit a qualified column reference in the IR slot matching each clause for every {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} named in that clause's prose at that scope.",
    'Phrases such as last N days, past N weeks, or within the last N months map to value_type date_window with {"unit", "amount"}.',
    "Elapsed time between two date columns maps to value_type date_diff with left_expr as a subtraction expression.",
    "CURRENT_DATE and CURRENT_TIMESTAMP map to keyword right_expr leaves, not string raw_value literals.",
    "Integer columns with schema role temporal and type integer are day-count durations; compare them to elapsed-day expressions, not as calendar dates.",
    f"Date shifted by an integer duration column uses {INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} in left_expr or right_expr, not date_diff.",
)

ENCODER_IR_ASSEMBLY_RULES: tuple[str, ...] = (
    "where is a nested predicate tree (op, predicates, groups) parsed from planner filter prose.",
    "Emit raw_value for every WhereParam and HavingParam literal; never emit param_key or param_values.",
    f"Every column_ref is {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} using only structural_schema_for_chosen_tables identifiers.",
    f"Downstream repair keeps only tables with column references in each scope's IR; emit refs only for {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens named in the matching clause prose.",
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
    f"Qualify every column reference as {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} using names from the schema text and allowed_tables; "
    "never emit bare column names except the bare wNN and cNN registry tokens on select_cols that point at window_registry and case_registry entries. "
    "Qualify columns inside every window_registry.window_spec.partition_by, order_by, and argument, and inside every case_registry case_when branch (condition sides, result, else_result).",
    "Use only tables and columns from the provided schema text and allowed_tables; do not invent identifiers.",
    "Join path discovery, foreign-key traversal, and bridge or junction tables are handled only by the downstream engine after this JSON is parsed; never refuse or shrink the intent because tables look disconnected in the structural payload.",
    "Do not judge whether the question is answerable from schema connectivity; translate the planner prose into IR using only the listed planner tables and their columns.",
    "Grain must match structure: grouped requires group_by_cols and aggregation in select_cols; "
    "row_level means no GROUP BY and no aggregation in select_cols; "
    "scalar is a single aggregated result with no GROUP BY.",
    "Row-level predicates belong in where; predicates on aggregates belong in having. "
    "Never put join predicates in where or having.",
    (
        f"A where predicate may compare {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} to "
        f"{INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER} when the question asks for a comparison "
        "between two values. This is a comparison, not a relationship: it states how two values rank "
        "against each other, never how the two tables connect. Join paths remain discovered downstream "
        "from foreign keys."
    ),
    "SUM and AVG apply only to numeric measure columns; use COUNT for non-measure columns.",
    "Columns whose schema type is unknown may appear in select_cols only; do not filter or aggregate them.",
    "Nested aggregation is forbidden; compute inner aggregates in a CTE step, then aggregate in the main query.",
    "Do not use EXTRACT(EPOCH FROM ...) for time differences; subtract date columns directly or use supported date functions.",
    f"CTE output_columns are snake_case alias tokens matching ^[a-z_][a-z0-9_]*$; never qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, never function call text, never AS clauses; align positionally with select_cols. Reference CTE outputs only via cteN.<output_columns_token> in window_registry, where, having, order_by_cols, and select_cols.",
    'Relative date-window filters use value_type date_window with value {"unit", "amount"}; '
    'column-to-column date spans use value_type date_diff with value {"unit", "amount"}. Use singular unit names. '
    'Absolute calendar bounds in date_window use ISO 8601 start/end strings such as "2020-01-15".',
    "Integer columns with schema role temporal represent day-count durations; compare them to elapsed day expressions "
    "(date subtraction or keyword minus date), not as calendar dates.",
    "BETWEEN uses op between with value [lower, upper]. NULL checks use op is null or is not null without a value field.",
    (
        "Negated comparisons (!=, not in, not like, not between) include rows where the filtered column value "
        "is unknown when that column is nullable; use is null or is not null when the question is only about "
        "missing values."
    ),
    (
        "Encode WHERE and HAVING boolean logic as PredicateGroup trees: op and/or with predicate leaves "
        "and nested groups (nesting depth capped at 3). For (predicate_A OR predicate_B) emit op or with two "
        "predicate leaves. For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit op or with "
        "two op and groups, each holding two predicate leaves."
    ),
    "window_registry defines registry_id and window_spec; select_cols reference entries with bare wNN tokens. "
    "Never put a window_spec key on a select_cols entry.",
    "case_registry defines registry_id and case_when with non-empty branches; select_cols reference entries with bare cNN tokens. "
    "When the question asks for conditional labels or buckets over columns, populate case_registry rather than dropping the derived column.",
    "SELECT DISTINCT prefixes the column expr with the bare token DISTINCT and a space "
    f"('DISTINCT {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}'). Use COUNT(DISTINCT {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}) for distinct counts; "
    f"for COUNT(DISTINCT CONCAT({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER})) emit Shape A: COUNT MulGroup with distinct=true whose single multiply child is a CONCAT MulGroup (scalar_func='concat'); "
    "do not wrap DISTINCT around arbitrary expressions except as COUNT(DISTINCT ...). "
    "Do not embed COUNT(*) inside arithmetic subexpressions—use COUNT(*) only as a top-level aggregate where appropriate.",
    f"Arithmetic combines expressions with +, -, *, /; aggregations may wrap arithmetic (SUM({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} * {INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER})). "
    f"Subtract date columns directly ({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}) for day differences. "
    f"Add or subtract integer day counts from date columns ({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + 7) for due-date comparisons.",
    "String concatenation uses CONCAT(expr1, ' ', expr2, ...) in expr strings; do not use the SQL || operator (pipe-pipe).",
    f"Apply scalar functions such as ROUND after aggregates when needed (ROUND(SUM({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}), 2)).",
    "Use exact identifiers from the provided schema text; never leave synthetic shape tokens from this prompt "
    f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}, column, "
    f"{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}), generic instructional tokens (table_N, column_N), or angle-bracket markup in expressions.",
)

INTENT_PARSE_RULES_APPEND: tuple[str, ...] = (
    "output_format lists every required top-level key; use [] for empty arrays and null for unused scalars.",
    "natural_language is required: describe exactly what the emitted IR returns (tables, columns, filters, aggregates) using real entity names; never include refusal, unavailability, permission, or denial language.",
    (
        "Encode WHERE and HAVING boolean logic as PredicateGroup trees: op and/or with predicate leaves "
        "and nested groups. For (predicate_A OR predicate_B) emit a where or having PredicateGroup with op or and two "
        "predicate leaves. For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit op or with "
        "two op and groups, each holding two predicate leaves. "
        "Legacy flat where_param and where_group keys are accepted on import only; new intents must use where and having PredicateGroup trees. "
        "Use qualified column references from the schema for left_expr; bind literals via value placeholders as elsewhere in these rules. "
        "Each where_param element must include op."
    ),
)

PLANNER_CARDINALITY_RELATIONSHIP_RULE: str = (
    f"When {INSTRUCTIONAL_TABLE_PLACEHOLDER} links to {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} through both a direct FK and a junction table, use the direct FK when the "
    f"question concerns one {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} value per {INSTRUCTIONAL_TABLE_PLACEHOLDER} row; use the junction when it concerns the set of "
    f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} values per {INSTRUCTIONAL_TABLE_PLACEHOLDER} row."
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
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} syntax, SQL, IR tokens, join paths, or set operators. "
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
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} where needed. Copy every literal into the matching prose field. "
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
    f"qualified column reference in IR for every {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} named in the matching clause prose at that scope. "
    "For cte_steps, set cte_name from name and tables from tables; tables may contain base schema tables and prior "
    "planner step names matching earlier cte_steps names. "
    "Emit only operators in operator_reference, value types in value_type_reference, and constructs in "
    "supported_capabilities. NEVER emit param_key, param_values, or harvested-literal mappings; emit raw_value for "
    "Where and Having literals; leave select and aggregate aliases empty strings."
)

LOGICAL_DECOMPOSITION_GUIDANCE: tuple[str, ...] = (
    "Describe each prose field thoroughly enough that a structural converter can build the IR without re-reading the question.",
    "Only tables may name real schema tables; every other field is natural language.",
    "Use cte_steps when the question needs a reusable intermediate; each step lists name, tables, and the same prose fields as the top level; tables may name base schema tables and prior step names.",
    "Put window definitions in the window prose field and case definitions in the case prose field; use as <registry_name> only inside those two fields.",
    "Never describe joins as explicit paths; the engine discovers FK paths.",
    f"After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} in the appropriate prose field.",
    f"Omitting {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables is correct when its columns appear in prose; the tables list alone never keeps a table in join scope.",
    f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
    f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
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
    "Do not encode row filters as CASE branches; use where for row membership.",
    "Never put AVG, SUM, COUNT, MIN, MAX calls or OVER (...) frames inside where_param right_expr as raw_sql. "
    "Compare to aggregates using an extra cte_steps row, a scalar subquery shape allowed by the IR, or window_registry references.",
    "string_agg uses agg_func='string_agg', agg_sep_param_key for the delimiter, and optional agg_order_by for within-aggregate ordering.",
    "stddev, variance, and median use agg_func on a single numeric column; median is refused when the engine capability flag is off.",
)

INTENT_ANSWER_STYLE_GUIDANCE: tuple[str, ...] = LOGICAL_DECOMPOSITION_GUIDANCE + FORMAT_STRUCTURAL_GUIDANCE

SQL_AGG_FUNC_CALL_RE = re.compile(
    r"\b(?:count|sum|avg|min|max)\s*\(",
    re.IGNORECASE,
)

INTENT_PLACEHOLDER_ANGLE_RE = re.compile(
    rf"<(table_\d+|table\d+|column_\d+|col\d+|{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER}|{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}|value_from_question|measure_\d+|count_rows)>",
    re.IGNORECASE,
)

EXPR_TABLE_COLUMN_REF_RE = re.compile(r"\w+\.\w+")

REGISTRY_WINDOW_ID_RE = re.compile(r"^w\d{2}$")

REGISTRY_CASE_ID_RE = re.compile(r"^c\d{2}$")

AGG_PREFIXES = frozenset(
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

NUMERIC_RESULT_AGGS = frozenset({"count", "sum", "avg", "stddev", "variance", "median"})

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
PRE_QUOTED_IN_LIST_INLINE_RE: re.Pattern[str] = re.compile(r"^'(?:[^']|'')*'(?:,'(?:[^']|'')*')+$")

UNSAFE_PARAM_LITERAL: str = "unsafe_param_literal"
SQL_STRING_LITERAL_STATEMENT_TERMINATOR: str = ";"
SQL_STRING_LITERAL_COMMENT_MARKERS: tuple[str, ...] = ("--", "/*")
SQL_STRING_LITERAL_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")
MYSQL_NO_BACKSLASH_ESCAPES_SQL_MODE_TOKEN: str = "NO_BACKSLASH_ESCAPES"

IN_OPS = frozenset({"in", "not in"})

IN_STRING_SEPARATORS = re.compile(r"['\"]?\s*,\s*['\"]?")

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

QUESTION_NORMALIZATION_VERSION: str = "0.2.1"

QUESTION_NORMALIZATION_VERSION_KEY: str = "question_normalization_version"

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

MYSQL_PROFILING_SAMPLE_PREDICATE = "RAND({seed}) < {ratio}"

REDSHIFT_PROFILING_SAMPLE_PREDICATE = (
    "MOD(ABS(FNV_HASH(CAST({{col}} AS VARCHAR) || '{seed}')), 1000000) / 1000000.0 < {ratio}"
)

DUCKDB_PROFILING_SAMPLE_PREDICATE: str = "USING SAMPLE {pct:.4f} PERCENT (bernoulli, {seed})"

SQLITE_PROFILING_SAMPLE_PREDICATE: str = (
    "CAST((abs(hash({{col}} || '{seed}')) % 1000000) AS REAL) / 1000000.0 < {ratio}"
)

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

REALISM_CATEGORY_LIST: str = ", ".join(sorted(REALISM_DROP_REASON_CATEGORIES))

QUESTION_FROM_SQL_SYSTEM: str = (
    "You are given a SQL query and a schema description. "
    "Your job is to decide whether the query represents a realistic, "
    "meaningful business question and, if so, produce natural-language paraphrases "
    "that a human analyst would ask to obtain this query's result.\n\n"
    "Rules:\n"
    "- If the query is unrealistic, nonsensical, or produces meaningless "
    "results, set is_realistic to false and explain why in drop_reason.\n"
    "- If realistic, set questions to an array of up to three distinct, "
    "conversational paraphrases a non-technical user would ask. "
    "Do NOT use SQL jargon or raw column names — use natural business language.\n"
    "- Do not phrase the output as numbered steps, subqueries, JOIN recipes, or procedural SQL instructions; "
    "each entry must read as one coherent analyst question.\n"
    "- You may also set question (string) to the first paraphrase for compatibility; "
    "when questions is non-empty, question should match questions[0].\n"
    "- Output ONLY valid JSON with fields: "
    '"questions" (array of strings), "question" (string, optional legacy), '
    '"is_realistic" (boolean), "drop_reason" (string or null), and optionally '
    f'"drop_reason_category" (string) when is_realistic is false. '
    f"If present, drop_reason_category must be one of: {REALISM_CATEGORY_LIST}.\n"
)

STRUCTURAL_CODE_TO_DIAG: dict[str, str] = {
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

PG_SIMPLE_AGG_NAMES: frozenset[str] = frozenset(
    {"count", "sum", "avg", "min", "max", "stddev", "variance", "median", "string_agg"}
)

EXPANSION_SUBTREE_POOL_MAX: int = 128

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

WHERE_VALUE_TYPE_DATE_WINDOW: frozenset[str] = frozenset({"temporal", "date_window"})

WHERE_VALUE_TYPE_DATE_DIFF: frozenset[str] = frozenset({"date_diff"})

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

MYSQL_CONNECTION_CHARSET: str = "utf8mb4"

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
    "42501",
)

SQLGLOT_DIALECT_BY_ENGINE: dict[str, str] = {}

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

OVERRIDE_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {"authored_against_structural_hash", "authored_at", "needs_reconfirmation"}
)

VALID_COLUMN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"description", "sensitivity", "role", "boolean_truth_value", "usable", *OVERRIDE_PROVENANCE_KEYS},
)

VALID_TABLE_OVERRIDE_KEYS: frozenset[str] = frozenset({"description", "role", "columns", *OVERRIDE_PROVENANCE_KEYS})

VALID_TOP_LEVEL_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "version",
        "tables",
        "foreign_keys_add",
        "foreign_keys_remove",
        "primary_keys_add",
        "primary_keys_remove",
        "business_knowledge",
        "_readonly",
    }
)

MAX_REPAIR_ATTEMPTS_PER_CODE: int = 1

DIAGNOSTIC_FUZZY_CUTOFF: float = 0.6

MAX_PREDICATE_NESTING_DEPTH: int = 3

DISTINCT_ON_CTE_NAME_PREFIX: str = "don_"
DISTINCT_ON_RANK_COLUMN: str = "__don_rn"

ANTI_JOIN_PRESENCE_COLUMN_SUFFIX: str = "__present"

JOIN_ORPHAN_RATE_DIAGNOSTIC_FLOOR: float = 0.05
DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH: str = "JOIN_ORPHAN_RATE_HIGH"
DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY: str = "JOIN_NULLABLE_KEY"
JOIN_PATH_TIE_REFUSAL_CEILING: int = 64
JOIN_PATH_TIE_OVERFLOW_MARKER: str = "__join_path_tie_overflow_count__"
DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED: str = "JOIN_PATH_TIE_CEILING_EXCEEDED"
DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP: str = "JOIN_CANDIDATE_CAP"
DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE: str = "SEMANTIC_PROFILE_WHERE_EDGE"
DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED: str = "REDUNDANT_JOIN_WHERE_DROPPED"
DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_ELIMINATED: str = "REDUNDANT_KEY_JOIN_ELIMINATED"
DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_CAP_REACHED: str = "REDUNDANT_KEY_JOIN_CAP_REACHED"

ELIMINATE_REDUNDANT_KEY_JOINS_MAX_ITERATIONS: int = 8

DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR: str = "COMPARISON_JOIN_DETOUR"

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

DATE_COLUMN_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "_date",
        "_at",
        "_time",
        "_timestamp",
        "timestamp",
        "datetime",
        "create_date",
        "start_date",
        "end_date",
        "ordered_date",
        "received_date",
    }
)

TEMPLATE_STORE_FORMAT_VERSION: str = "0.2.1"

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

BUSINESS_KNOWLEDGE_DEFAULT_KIND: str = "glossary"
BUSINESS_KNOWLEDGE_COLUMN_REF_RE: re.Pattern[str] = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)

PLAN_PREVIEW_INTENT_PARSE_FAILED: str = "intent parse did not produce a runnable plan"

INTERPRET_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "interpret_plan_schema",
    "supported_capabilities",
    "schema_domain",
    "business_context",
    "question",
    "prior_question_feedback",
    "prior_user_corrections",
)

GROUND_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "logical_intent_json_schema",
    "nl_conventions",
    "logical_schema_rules",
    "supported_capabilities",
    "schema_literal_json",
    "interpret_plan",
    "business_context",
    "question",
    "logical_decomposition_guidance",
    "prior_question_feedback",
    "prior_user_corrections",
    "prior_grounding_failures",
)

COMPOSE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "output_format",
    "structural_json_keys",
    "critical_rules",
    "parse_rules_append",
    "format_structural_guidance",
    "supported_capabilities",
    "nl_phrase_mappings",
    "nl_to_ir_guidance",
    "ir_assembly_rules",
    "operator_reference",
    "value_type_reference",
    "structural_pattern_rules",
    "compose_priority_rules",
    "logical_to_ir_field_map",
    "cte_tables_encoding",
    "structural_schema_for_chosen_tables",
    "logical_intent",
)

SEMANTIC_REPAIR_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "critical_rules",
    "field_specifications",
    "structural_json_keys",
    "output_format",
    "errors_to_fix",
    "suggestions",
    "schema_info",
    "business_context",
    "current_intent",
    "question",
    "prior_question_feedback",
)

FORMAT_REPAIR_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "structural_json_keys",
    "instructions",
    "output_format",
    "question",
    "invalid_response",
    "parse_error",
)

PARSE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "structural_json_keys",
    "expression_format",
    "rules",
    "logical_decomposition_guidance",
    "format_structural_guidance",
    "output_format",
    "operator_reference",
    "value_type_reference",
    "naming_conventions",
    "schema_summary",
    "business_context",
    "allowed_tables",
    "question",
    "prior_question_feedback",
)

FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "extraction_rules",
    "output_format",
    "matched_question",
    "matched_values",
    "param_keys",
    "param_slots",
    "question",
)

DISPLAY_ALIAS_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "columns",
    "question",
)

JOIN_CHOICE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "output_format",
    "scopes",
    "deterministic_sql",
    "question",
)

FEDERATION_COMPOSITION_PHASE_A: str = "A:roster"
FEDERATION_COMPOSITION_PHASE_B: str = "B:merge"
FEDERATION_COMPOSITION_PHASE_C: str = "C:collapse"
FEDERATION_COMPOSITION_PHASE_D: str = "D:unify"
FEDERATION_COMPOSITION_PHASE_E: str = "E:edges"
FEDERATION_COMPOSITION_PHASE_F: str = "F:reconcile"
FEDERATION_COMPOSITION_PHASE_G: str = "G:identity"
FEDERATION_COMPOSITION_PHASE_H: str = "H:persist"

FEDERATION_MANIFEST_FILENAME: str = "federation_manifest.json"
FEDERATION_MAPPINGS_FILENAME: str = "federation_mappings.json"
FEDERATION_MAPPINGS_APPLIED_FILENAME: str = "applied_federation_mappings.json"
FEDERATION_COMPOSITE_SCHEMA_FILENAME: str = "composite_schema_graph.json.gz"
FEDERATION_MIGRATION_MAP_FILENAME: str = "federation_migration_map.json"
FEDERATION_PLAN_TEMPLATE_FILENAME: str = "federation_plan_templates.json"
FEDERATION_JOIN_FEEDBACK_SEGMENT: str = "feedback"
FEDERATION_JOIN_FEEDBACK_PREFIX: str = "join_"
FEDERATION_MAPPING_SUGGESTIONS_CACHE_FILENAME: str = "federation_mapping_suggestions_cache.json"
FEDERATION_STORAGE_PREFIX: str = "fed_"
FEDERATION_SOURCE_STORAGE_PREFIX: str = "fedsrc_"
APPLIED_MAP_ARCHIVE_RETENTION_COUNT: int = 3
APPLIED_MAP_ARCHIVE_TIMESTAMP_RE = re.compile(r"\.applied\.\d{8}T\d{6}Z\.json$")
FEDERATION_TEMPLATES_SEGMENT: str = "federation_templates"

FEDERATION_ARTIFACT_FORMAT_VERSION: str = "0.2.1"
FEDERATION_MEMBER_MANIFEST_FILENAME: str = "federation_member_manifest.json"
FEDERATION_METHOD_SEMANTICS: dict[str, str] = {
    "add_engine": "composite",
    "apply_federation_declaration": "composite",
    "apply_migration_map": "composite",
    "apply_overrides": "both",
    "apply_schema_overrides": "both",
    "asession": "composite",
    "clear_all_learning": "both",
    "clear_persisted_overrides": "member",
    "clear_simulation_caches": "both",
    "clear_template_store": "both",
    "close": "composite",
    "aetherspace": "composite",
    "apply_aetherspace": "composite",
    "delete_aetherspace": "composite",
    "export_aetherspace": "composite",
    "export_context": "composite",
    "list_contexts": "composite",
    "export_federation_declaration": "composite",
    "export_overrides": "both",
    "export_schema_overrides": "both",
    "get_qsim_summary": "composite",
    "get_questions_only": "composite",
    "get_seed_warmup_summary": "composite",
    "list_aetherspaces": "composite",
    "prepared_federated_outcome": "composite",
    "preview_table": "composite",
    "preview_plan": "composite",
    "mapping_suggestions": "composite",
    "remove_engine": "composite",
    "run_interactive": "composite",
    "run_qsim": "composite",
    "run_seed_warmup": "unsupported",
    "run_seed_warmup_from_history": "unsupported",
    "run_seed_warmup_from_query_log": "unsupported",
    "session": "composite",
    "show_config": "composite",
    "list_templates": "composite",
    "fetch_template": "composite",
    "execute_template": "composite",
    "export_knowledge": "composite",
    "export_space_knowledge": "composite",
    "export_metadata": "composite",
}
FEDERATION_MAPPINGS_VERSION: str = "0.2.1"
FEDERATION_MAPPINGS_MIN_VERSION: str = "0.2.1"
FEDERATION_PLAN_TEMPLATE_FORMAT_VERSION: str = "0.2.1"
FEDERATION_PLAN_TEMPLATE_FILE_CAP: int = 256
FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP: int = 64
FEDERATION_MAX_JOIN_PATH_TIE_CAP: int = 256
FEDERATION_MAX_JOIN_CANDIDATE_CAP: int = 1024
FEDERATION_ENUM_PROMPT_CAP: int = 10
SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS: int = 256
SCHEMA_DESCRIPTION_PROMPT_COUNT_CAP: int = 128
FEDERATION_AVERAGE_SCALE_HEADROOM: int = 6
FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION: int = 38
FEDERATION_COORDINATOR_DECIMAL_FALLBACK_SCALE: int = 9
FEDERATION_COORDINATOR_DECIMAL_FALLBACK: str = (
    f"DECIMAL({FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION}, {FEDERATION_COORDINATOR_DECIMAL_FALLBACK_SCALE})"
)
SCHEMA_ENRICHED_LINES_MAX_CHARS: int = 16384

FEDERATION_MAPPING_NAME_SUBSTRING_SCORE: float = 0.85
FEDERATION_MAPPING_VALUE_OVERLAP_FLOOR: float = 0.1
FEDERATION_MAPPING_NAME_SCORE_FLOOR: float = 0.8
FEDERATION_MAPPING_SCORE_NAME_WEIGHT: float = 0.5
FEDERATION_MAPPING_SCORE_OVERLAP_WEIGHT: float = 0.5

FEDERATION_COMPOSITE_RECONCILIATION_NOTE: str = (
    "When two descriptions refer to the same business concept, choose one canonical wording and role."
)

PROMPT_NEUTRALITY_AUDIT_CONSTANTS: frozenset[str] = frozenset(
    {
        "INTENT_INTERPRET_SYSTEM",
        "INTENT_GROUND_SYSTEM",
        "INTENT_COMPOSE_SYSTEM",
        "INTENT_CRITICAL_RULES",
        "INTENT_PARSE_RULES_APPEND",
        "INTENT_FORMAT_REPAIR_JSON_RULES",
        "ENCODER_IR_ASSEMBLY_RULES",
        "ENCODER_NL_TO_IR_GUIDANCE",
        "REPAIR_INSTRUCTIONS",
        "LOGICAL_DECOMPOSITION_GUIDANCE",
        "FORMAT_STRUCTURAL_GUIDANCE",
        "COMPOSE_SUPPORTED_CAPABILITIES",
        "FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES",
        "INTERPRET_SUPPORTED_CAPABILITIES",
        "GROUND_SUPPORTED_CAPABILITIES",
        "FEDERATION_COMPOSITE_RECONCILIATION_NOTE",
        "INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES",
        "INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE",
        "SEED_QUESTION_CLARIFY_SYSTEM",
        "PLANNER_CARDINALITY_RELATIONSHIP_RULE",
        "PLANNER_SHARED_PK_TABLE_SCOPE_RULE",
        "PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE",
    }
)

FEDERATION_QUALIFIED_COLUMN_REF_RE = re.compile(r"^([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)$")
FEDERATION_QUALIFIED_THREE_PART_REF_RE = re.compile(
    r"^([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)$",
)
FEDERATION_STORAGE_SLUG_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")
FEDERATION_CONNECTION_SLUG_NON_WORD_RE = re.compile(r"[^\w]+")

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
FEDERATION_MANIFEST_JOIN_KEYS: frozenset[str] = frozenset({"left", "right", "kind", "logical_key"})
FEDERATION_CROSS_SOURCE_JOIN_KINDS: frozenset[str] = frozenset({"inner", "left"})
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
FEDERATION_DECLARATION_VERSION: str = "0.2.1"
FEDERATION_DECLARATION_FILENAME: str = "federation_declaration.json"
FEDERATION_WARMUP_UNSUPPORTED_MESSAGE: str = "warmup is not supported on AetherFederation"
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

DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED: str = "ENUM_PROMPT_TRUNCATED"
DIAGNOSTIC_CODE_DESCRIPTION_PROMPT_TRUNCATED: str = "DESCRIPTION_PROMPT_TRUNCATED"
DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED: str = "DESCRIPTION_ENRICHMENT_FAILED"
DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP: str = "DESCRIPTION_ENRICHMENT_NOOP"
DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT: str = "SCHEMA_FK_CATALOG_ABSENT"
DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED: str = "SCHEMA_ROLE_TYPE_COERCED"
DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE: str = "SCHEMA_UNKNOWN_TYPE_UNUSABLE"

LLM_PRICE_TABLE_AS_OF: str = "2026-07-26"
LLM_PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-5.4-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.0625, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.0125, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
}

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

FEDERATION_TIMEZONE_AWARE_DATA_TYPES: frozenset[str] = frozenset(
    {
        "timestamptz",
        "timetz",
        "datetimeoffset",
        "timestamp_tz",
        "timestamp_ltz",
    }
)

MYSQL_TIMESTAMP_ENGINES: frozenset[str] = frozenset({"mysql", "mariadb"})

FIXED_WIDTH_TEXT_BASE_TYPES: frozenset[str] = frozenset({"char", "nchar", "bpchar", "character"})

UNSIGNED_INTEGER_TYPE_MAX: dict[str, int] = {
    "tinyint": 255,
    "smallint": 65535,
    "mediumint": 16777215,
    "int": 4294967295,
    "integer": 4294967295,
    "bigint": 18446744073709551615,
}

MAX_FLOAT_SAFE_INTEGER: int = 9007199254740992

ISO_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$")

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
DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE: str = "REFUSAL_PARSE_FAILURE"
DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA: str = "REFUSAL_DECLINED_SCHEMA"
DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP: str = "REFUSAL_JOIN_PATH_TIE_CAP"
DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET: str = "REFUSAL_CLAUSE_WIDENED_ROWSET"
DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT: str = "REFUSAL_PROBE_CTE_PLACEMENT"

REFUSAL_CATALOGUE: dict[str, dict[str, str]] = {
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED: {
        "user_text": PERMISSION_DENIED_USER_MESSAGE,
        "reformulation_hint": (
            "Ask your administrator for access to the required tables, or rephrase using only data you can query."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION: {
        "user_text": (
            "This question cannot be answered with the information currently available.\n\n"
            "Try rephrasing to ask about tables and columns you can see in the schema."
        ),
        "reformulation_hint": ("Try rephrasing to ask about tables and columns you can see in the schema."),
    },
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION: {
        "user_text": (
            "I could not pin this question to specific tables or columns.\n\n"
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, "
            "status, or region."
        ),
        "reformulation_hint": (
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, "
            "status, or region."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE: {
        "user_text": (
            "I could not understand the structure of this question.\n\n"
            "Try naming specific tables or columns, keeping filters simple and references clear."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\n"
            "Tips: mention specific tables or columns, keep filters simple, and avoid ambiguous references.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA: {
        "user_text": (
            "The proposed table and column mapping was declined.\n\n"
            "Try rephrasing using tables and columns that exist in this database, or ask about a related concept."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n"
            "Tips: use tables and columns that exist in this database, or ask about a related concept."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE: {
        "user_text": (
            "These tables could not be connected: {tables}. "
            "Declare a foreign-key or semantic relationship between them, or ask using fewer tables."
        ),
        "reformulation_hint": (
            "These tables cannot be joined with the relationships currently in this schema.\n\n"
            "Tips: declare a foreign-key or semantic link between the tables named in the error, "
            "or narrow the question to tables that already connect.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP: {
        "user_text": (
            "Too many equally short join paths between {source_table} and {target_table} "
            "({path_count} paths; limit {ceiling}). Narrow the tables in your question or declare "
            "which relationship to use."
        ),
        "reformulation_hint": (
            "Too many equally short join paths were found between the named tables.\n\n"
            "Tips: narrow the question to fewer tables or declare which relationship should connect them.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT: {
        "user_text": (
            "This aggregate would duplicate parent rows because of how the tables connect.\n\n"
            "Try grouping at the parent grain first, or use a join path that does not multiply rows."
        ),
        "reformulation_hint": (
            "Tips: group at the parent grain first, or choose a join path that does not multiply rows.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING: {
        "user_text": (
            "This comparison would require joining across too many tables.\n\n"
            "Try comparing values on tables that are closer together in the schema."
        ),
        "reformulation_hint": (
            "Tips: compare values on tables that are closer together, or simplify the comparison.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP: {
        "user_text": (
            "This question needs too many intermediate query steps.\n\n"
            "Try splitting the question into smaller parts or simplifying the logic."
        ),
        "reformulation_hint": ("Tips: split the question into smaller parts or simplify the intermediate logic.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP: {
        "user_text": (
            "This question shape cannot be answered with the databases currently available.\n\n"
            "Try a simpler question or ask on each source individually."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\nTips: try a simpler question or ask on each source individually.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET: {
        "user_text": (
            "A limit, sort, or distinct cannot be applied cleanly when a join multiplies rows.\n\n"
            "Group first or simplify joins before limiting or deduplicating results."
        ),
        "reformulation_hint": (
            "Please rephrase or retry.\n\n"
            "Tips: group first or simplify joins before limiting, sorting, or deduplicating results.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT: {
        "user_text": (
            "A filter step cannot be used in this join position.\n\n"
            "Restructure the question so filtering happens on the correct side of the join."
        ),
        "reformulation_hint": (
            "Please rephrase or retry.\n\n"
            "Tips: restructure the question so filtering happens on the correct side of the join.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST: {
        "user_text": (
            "This filter cannot be expressed: a NOT IN list cannot include null. "
            "Ask whether the column is null or not, or name only the non-null values to exclude."
        ),
        "reformulation_hint": ("Ask whether the column is null or not, or name only the non-null values to exclude.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR: {
        "user_text": (
            "This question uses an expression structure that cannot be compiled safely.\n\n"
            "Rephrase using explicit columns, filters, and aggregates supported by the schema."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\n"
            "Tips: use explicit table.column references and supported filters or aggregates.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN: {
        "user_text": (
            "This filter cannot be expressed: the column stores dates without time-of-day, "
            "so hour, minute, or second windows cannot be answered. Ask for a day-level window instead."
        ),
        "reformulation_hint": "Ask for a day-level date window instead of hours, minutes, or seconds.\n",
    },
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL: {
        "user_text": (
            "This filter cannot be expressed: the date bound is ambiguous. "
            "Use ISO 8601 form such as 2020-01-15 or 2020-01-15T14:30:00."
        ),
        "reformulation_hint": ("Use an unambiguous ISO 8601 date such as 2020-01-15 or 2020-01-15T14:30:00.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING: {
        "user_text": (
            "A column needed for this answer is missing from one or more databases in the group.\n\n"
            "Try asking over the sources that have the column, or declare a shared column mapping."
        ),
        "reformulation_hint": (
            "Try asking over the sources that have the column, or declare a shared column mapping.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE: {
        "user_text": (
            "This question cannot be answered: the {column} column has an unsupported data type "
            "and cannot be filtered or aggregated."
        ),
        "reformulation_hint": ("Try asking about a different column or a supported data type.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT: {
        "user_text": REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
        "reformulation_hint": ("Try rephrasing to ask about information that is available in this context.\n"),
    },
}

REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST]["user_text"]

REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN_MESSAGE: str = REFUSAL_CATALOGUE[
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN
]["user_text"]

REFUSAL_AMBIGUOUS_DATE_LITERAL_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL][
    "user_text"
]

REFUSAL_UNSUPPORTED_COLUMN_TYPE_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE][
    "user_text"
]

REFUSAL_DIAGNOSTIC_CODES: frozenset[str] = frozenset(REFUSAL_CATALOGUE.keys())

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
    "restricted": DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    "invalid_question": DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    "parse_failed": DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    "schema_invalid_declined": DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
}

REPHRASE_HINT_REFUSAL_CODES: dict[str, str] = {
    "intent_parse_failed": DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    "schema_invalid_declined": DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    "join_path_unavailable": DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    "restricted_question": DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    "vague_question": DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    "federation_ineligible": DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
}

PERMISSION_DENIED_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        "access_policy",
        "denied_reference",
        "deny_bare_select",
        "sensitive_group_by",
    }
)

PERMISSION_DENIED_CATEGORY_ORACLE_KINDS: frozenset[str] = frozenset(
    {
        "order_by_validity",
        "where_validity",
        "having_semantic",
    }
)

REFUSAL_UNSUPPORTED_COLUMN_TYPE_ISSUE_IDS: frozenset[str] = frozenset(
    {
        "unsupported_column_type",
    }
)

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

REFUSAL_NULL_IN_NEGATED_LIST_ISSUE_IDS: frozenset[str] = frozenset(
    {
        "null_in_negated_list",
    }
)

FEDERATION_REJECTION_BUCKETS: frozenset[str] = frozenset(
    {
        "MALFORMED_MEMBER_ANSWER",
        "JOIN_FAN_OUT",
    }
)

INELIGIBLE_ANSWERABLE_HINTS_BY_CODE: dict[str, str] = {
    "no_tables": "Try naming the entities and metrics you need from the schema.",
    "space_scope": "Try asking within the active space tables, or widen the space definition.",
    "projection_not_single_member": (
        "Try asking for columns that live on one member, or declare a logical column mapping."
    ),
    "union_column_missing": (
        "Try asking over the members that do have the column, or declare a logical column mapping spanning all members."
    ),
    "undeclared_join_path": ("Try naming entities that share a declared relationship in the federation manifest."),
    "cross_source_aggregate": "Try asking for the metric per entity or per category separately.",
    "cross_source_or_filter": "Try splitting the question into separate filters on each entity.",
    "cross_source_where_group_disjunction": "Try splitting the question into separate filters on each entity.",
    "cross_source_distinct": "Try asking for the list or count from one entity, then combine results manually.",
    "cross_source_order_by": "Try asking for the list or count from one entity, then combine results manually.",
    "cross_source_window": "Try ranking or windowing within one entity before combining, or ask without the window.",
    "cross_source_correlated_subquery": "Try expressing the lookup as a join or an intermediate step on one source.",
    "cross_source_scalar_subquery": "Try expressing the lookup as a join or an intermediate step on one source.",
    "cross_source_having": "Try asking for the metric per entity or per category separately.",
    "cross_source_distinct_on": "Try asking for one row per entity within each member before combining.",
    "cross_source_semijoin": "Try declaring a join key between the probe and driver tables in the federation manifest.",
    "cross_source_antijoin": "Try declaring a join key between the probe and driver tables in the federation manifest.",
    "cross_source_predicate_disjunction": "Try splitting the question into separate filters on each entity.",
    "semi_join_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "anti_join_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "distinct_on_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "preserve_tables_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "nested_predicate_groups": "Try simplifying boolean logic to AND-only filters on one member at a time.",
    "distinct_on_requires_order_by": "Try specifying how to rank rows within each partition (for example by date descending).",
    "unattributable_raw_sql": "Try rephrasing using explicit table.column references from the schema.",
    "member_capability": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "unknown": "Try asking for each part separately, or rephrasing without cross-source aggregation.",
}

STATISTICAL_AGG_EXCLUDED_ENGINES = frozenset({"sqlite", "csv"})
WINDOW_FRAMES_EXCLUDED_ENGINES = frozenset({"csv"})
ARRAY_CONTAINS_EXCLUDED_ENGINES = frozenset({"csv"})
COLLATION_ENGINES = frozenset({"postgresql", "redshift", "mysql", "mariadb", "sqlserver", "snowflake", "oracle"})
CASE_INSENSITIVE_COLLATION_ENGINES = frozenset({"mysql", "mariadb", "sqlserver"})
UNSIGNED_SEMANTICS_ENGINES = frozenset({"mysql", "mariadb"})
TIMESTAMPTZ_SEMANTICS_ENGINES = frozenset({"postgresql", "redshift", "snowflake", "duckdb", "bigquery"})
ROUNDING_MODE_HALF_EVEN_ENGINES = frozenset({"sqlite"})

DEFAULT_NULL_ORDERING_ASC: Literal["last"] = "last"
DEFAULT_NULL_ORDERING_DESC: Literal["first"] = "first"

REMOVED_BEHAVIOUR_ENVIRONMENT_KEYS: dict[str, str] = {
    "AETHERDIALECT_MAX_QUERY_COST_ROWS": "PolicyConfig.MAX_QUERY_COST_ROWS",
    "AETHERDIALECT_MAX_QUERY_COST_BYTES": "PolicyConfig.MAX_QUERY_COST_BYTES",
    "AETHERDIALECT_STATEMENT_TIMEOUT_MS": "EngineLimits.statement_timeout_ms",
    "AETHERDIALECT_LLM_TIMEOUT_MS": "PolicyConfig.LLM_TIMEOUT_MS",
    "AETHERDIALECT_PROFILE_TIMEOUT_MS": "EngineLimits.profile_timeout_ms",
    "AETHERDIALECT_EXPLAIN_TIMEOUT_MS": "PolicyConfig.EXPLAIN_TIMEOUT_MS",
    "AETHERDIALECT_LLM_BATCH_ENABLED": "PolicyConfig.LLM_BATCH_ENABLED",
    "AETHERDIALECT_TABULAR_LLM_ASSIST": "PolicyConfig.TABULAR_LLM_ASSIST",
}

UNUSABLE_NULL_RATIO_THRESHOLD: float = 0.99
SENTINEL_MODE_FREQUENCY_THRESHOLD: float = 0.99

DATABASE_ERROR_CLASSIFICATION_TRANSIENT: str = "transient"
DATABASE_ERROR_CLASSIFICATION_PERMANENT: str = "permanent"
DATABASE_ERROR_CLASSIFICATION_UNKNOWN: str = "unknown"

DATABASE_ERROR_CLASSIFICATION_BY_EXCEPTION_NAME: dict[str, str] = {
    "InterfaceError": DATABASE_ERROR_CLASSIFICATION_TRANSIENT,
}

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

DATABASE_ERROR_CLASSIFICATION_TRANSIENT_ERRNOS: frozenset[int] = frozenset(
    {10060, 10061, 11001, 11002, 111, 113, 115, 116}
)

REPHRASE_HINT_MESSAGES.update(
    {key: REFUSAL_CATALOGUE[code]["reformulation_hint"] for key, code in REPHRASE_HINT_REFUSAL_CODES.items()}
)
