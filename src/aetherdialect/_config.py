"""
Engine and policy settings, tunable thresholds, and shared validation constants.

`BOOLEAN_TRUTH_PATTERN_MAP` maps lowercased two-valued top-K sets to the canonical affirmative literal (lowercase) used when recording ``ColumnMetadata.boolean_truth_value``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from types import MappingProxyType
from typing import Any, ClassVar
from urllib.parse import quote

from platformdirs import user_data_dir

ARTIFACT_DIRECTORY_SEGMENT: str = "aetherdialect"
_ARTIFACT_USER_PARENT: str = user_data_dir(appname="aetherdialect", appauthor=False)
ENGINE_STORAGE_PLACEHOLDER_DIR: str = os.path.join(
    _ARTIFACT_USER_PARENT,
    ARTIFACT_DIRECTORY_SEGMENT,
    "__placeholder__",
)
ENGINE_STORAGE_SLUG_MAX_CHARS: int = 180

SUPPORTED_ENGINES: frozenset[str] = frozenset({"postgresql", "databricks"})

JSON_COMPACT_SEPARATORS: tuple[str, str] = (",", ":")

DIAGNOSTIC_CODE_REUSE_HIT: str = "REUSE_HIT"
DIAGNOSTIC_CODE_REUSE_MISS: str = "REUSE_MISS"
DIAGNOSTIC_CODE_LOW_CONFIDENCE: str = "LOW_CONFIDENCE"
DIAGNOSTIC_CODE_LARGE_RESULT_WARNING: str = "LARGE_RESULT_WARNING"
DIAGNOSTIC_CODE_PII_GATE_HIT: str = "PII_GATE_HIT"
DIAGNOSTIC_CODE_STAGE_A_RETRY: str = "STAGE_A_RETRY"
DIAGNOSTIC_CODE_STAGE_B_REPAIR: str = "STAGE_B_REPAIR"
DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART: str = "FALLBACK_FRESH_RESTART"
DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED: str = "CONFIG_FILE_VALUE_APPLIED"
DIAGNOSTIC_CODE_ENGINE_INFO: str = "ENGINE_INFO"
DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP: str = "SCHEMA_OVERRIDE_SKIP"

AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD: str = "write_queue_feedback_record"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_REJECT: str = "write_queue_template_reject"
AUDIT_EVENT_WRITE_QUEUE_TEMPLATE_ACCEPT: str = "write_queue_template_accept"
AUDIT_EVENT_WRITE_QUEUE_OVERRIDE_PROPOSAL: str = "write_queue_override_proposal"

CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC: str = (
    "deny_columns and allow_columns entries must be qualified as 'table.column' or '*.column'; "
    "bare column names are not permitted; got {spec!r}"
)

INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_BY_SENSITIVITY_VALUE: Mapping[str, str] = MappingProxyType(
    {
        "strict": "{location}: column {table}.{column} cannot appear in {surface} (strict classification).",
        "forbidden": "{location}: column {table}.{column} cannot appear in {surface} (forbidden classification).",
        "hygiene": "{location}: column {table}.{column} cannot appear in {surface} (hygiene classification).",
    }
)
INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_DEFAULT: str = (
    "{location}: column {table}.{column} cannot appear in {surface} (sensitivity policy)."
)

DEFAULT_RUNTIME_CONFIG_PATH: str = "./aetherdialect.toml"

MAX_STAGE_A_RETRIES: int = 2

PING_ATTEMPTS: int = 2

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
        "schema_invalid": {"type": "boolean"},
    },
}


def _build_logical_intent_schema() -> dict[str, Any]:
    """Return JSON Schema for planner :class:`LogicalIntent` LLM output."""

    prose: dict[str, Any] = {"type": "string"}
    limit_null: dict[str, Any] = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    cte_item: dict[str, Any] = {
        "type": "object",
        "required": ["name", "tables", "select"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "depends_on": {"type": "array", "items": {"type": "string"}},
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
        "properties": {
            "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "select": {"type": "string", "minLength": 1},
            "schema_invalid": {"type": "boolean"},
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
SQL_AGG_FUNC_CALL_RE = re.compile(
    r"\b(?:count|sum|avg|min|max)\s*\(",
    re.IGNORECASE,
)
WINDOW_RANKING_FUNCTIONS = frozenset({"row_number", "rank", "dense_rank"})
WINDOW_AGG_FUNCTIONS = frozenset({"sum", "avg"})
WINDOW_OFFSET_FUNCTIONS = frozenset({"lag", "lead"})
WINDOW_VALUE_FUNCTIONS = frozenset({"first_value", "last_value"})
VALID_WINDOW_FUNCTIONS = (
    WINDOW_RANKING_FUNCTIONS | WINDOW_AGG_FUNCTIONS | WINDOW_OFFSET_FUNCTIONS | WINDOW_VALUE_FUNCTIONS
)
VALID_SENSITIVITY_LEVELS = frozenset({"none", "hygiene", "strict", "forbidden", "pii", "restricted"})
HIDDEN_SENSITIVITIES = frozenset({"hygiene", "strict", "forbidden", "pii", "restricted"})
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
DATABRICKS_TABLE_QUALIFY_SKIP_IDENTIFIERS: frozenset[str] = frozenset(
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
INTENT_PLACEHOLDER_ANGLE_RE = re.compile(
    r"<(table_\d+|table\d+|column_\d+|col\d+|date_column_\d+|value_from_question|measure_\d+|count_rows)>",
    re.IGNORECASE,
)
VALID_RELATIVE_DATE_UNITS = frozenset(
    {"day", "week", "month", "quarter", "half_year", "year", "hour", "minute", "second"}
)
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
EXPR_TABLE_COLUMN_REF_RE = re.compile(r"\w+\.\w+")
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
BOOLEAN_FILTER_OPS = {"=", "!=", "in", "not in", "is null", "is not null"}
CATEGORICAL_FILTER_OPS = {
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
NUMERIC_CATEGORICAL_FILTER_OPS = {
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
NUMERIC_FILTER_OPS = frozenset(
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
CTE_NUMERIC_FILTER_OPS = list(NUMERIC_FILTER_OPS)
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
    BOOLEAN_FILTER_OPS | CATEGORICAL_FILTER_OPS | NUMERIC_CATEGORICAL_FILTER_OPS | frozenset({"contains"})
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

ARTIFACT_FORMAT_VERSION: int = 3
MIN_COMPATIBLE_PACKAGE_VERSION: str = "0.1.7"
ARTIFACT_MANIFEST_FILENAME: str = "artifact_manifest.json"
ARTIFACT_LOCK_FILENAME: str = ".aetherdialect_engine.lock"
ARTIFACT_LOCK_TIMEOUT_SECONDS: float = 30.0
ARTIFACT_LOCK_POLL_INTERVAL_SECONDS: float = 0.05

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

NORMALIZATION_JACCARD_FLOOR: float = 0.4

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

QUESTION_REUSE_AUTO_ACCEPT_THRESHOLD: int = 1

TRUST_FLOOR: int = 1

TRUST_CEILING: int = 2

TRUST_PROMOTE_INCREMENT: int = 1

TRUST_DEMOTE_DECREMENT: int = 0

TRUST_AUTO_ACCEPT_THRESHOLD: int = 1

SELF_JOIN_CTE_NAME_PREFIX: str = "sj_"

WARMUP_ROUND_TRIP_CARDINALITY_TOLERANCE: float = 0.25

SCHEMA_OVERRIDES_DEFAULT_FILENAME: str = "schema_overrides.json"

MIGRATION_MAP_FILENAME: str = "schema_migration_map.json"

WRITE_QUEUE_FILENAME: str = "write_queue.jsonl"

WRITE_QUEUE_DRAIN_TIMEOUT_SECONDS: float = 30.0

WRITE_QUEUE_MAX_BYTES_PER_DRAIN: int = 4194304

MIGRATION_MAP_ACTION_REMAP: str = "remap"

MIGRATION_MAP_ACTION_DESTRUCTIVE: str = "destructive"

MIGRATION_MAP_ACTION_ABORT: str = "abort"

ARTIFACT_LAST_ACTION_REMAP_USER_MAP: str = "remap_user_map"

ARTIFACT_LAST_ACTION_DESTRUCTIVE_USER_MAP: str = "destructive_user_map"

SCHEMA_OVERRIDES_APPLIED_SUFFIX: str = ".applied.json"
SCHEMA_OVERRIDES_APPLIED_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"

SCHEMA_OVERRIDES_SIDECAR_FILENAME: str = "applied_overrides.json"
SCHEMA_OVERRIDES_VERSION: int = 4
SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER: str = "ai"
SCHEMA_OVERRIDES_MAX_DESCRIPTION_CHARS: int = 2000

REGISTRY_WINDOW_ID_RE = re.compile(r"^w\d{2}$")
REGISTRY_CASE_ID_RE = re.compile(r"^c\d{2}$")

_DIAGNOSTIC_FORCE_DEPTH: int = 0


def diagnostic_debug_enabled() -> bool:
    """True when ``PolicyConfig.DEBUG`` or diagnostic capture (``telemetry_capture`` depth) is active."""

    return _DIAGNOSTIC_FORCE_DEPTH > 0 or PolicyConfig.DEBUG


def diagnostic_verbose_enabled() -> bool:
    """True when ``PolicyConfig.VERBOSE`` or diagnostic capture depth is active."""

    return _DIAGNOSTIC_FORCE_DEPTH > 0 or PolicyConfig.VERBOSE


def diagnostic_pipeline_trace_full_enabled() -> bool:
    """True when ``PolicyConfig.PIPELINE_TRACE_FULL`` or diagnostic capture depth is active."""

    return _DIAGNOSTIC_FORCE_DEPTH > 0 or PolicyConfig.PIPELINE_TRACE_FULL


def diagnostic_force_enter() -> None:
    """Increment nested diagnostic capture depth (used by ``telemetry_capture``)."""

    global _DIAGNOSTIC_FORCE_DEPTH
    _DIAGNOSTIC_FORCE_DEPTH += 1


def diagnostic_force_exit() -> None:
    """Decrement nested diagnostic capture depth."""

    global _DIAGNOSTIC_FORCE_DEPTH
    if _DIAGNOSTIC_FORCE_DEPTH > 0:
        _DIAGNOSTIC_FORCE_DEPTH -= 1


FAILURE_HINT_MAX_CHARS_PER_RECORD: int = 500
FAILURE_HINT_MAX_MESSAGES: int = 5
FAILURE_HINT_MAX_INJECT_CHARS: int = 1200
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
AGG_PREFIXES = frozenset({"COUNT(", "SUM(", "AVG(", "MIN(", "MAX("})
OP_FLIP: dict[str, str] = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
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
SCALAR_FUNC_DEFAULTS: dict[str, list] = {
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
STRUCTURAL_IDENTITY_VALUES = frozenset({0, 1})
STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE = re.compile(r":(s\d+)\b")
STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)*$")
IN_OPS = frozenset({"in", "not in"})
IN_STRING_SEPARATORS = re.compile(r"['\"]?\s*,\s*['\"]?")
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
AGG_KEYWORDS_RE = re.compile(r"\b(?:total|count|number\s+of|average|avg|sum|how\s+many)\b", re.IGNORECASE)
AGG_PATTERN = re.compile(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(.+?)\s*\)$", re.IGNORECASE)
TABLE_COL_PATTERN = re.compile(r"(\w+)\.(\w+)")
HAVING_COUNT_VALUES = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]
HAVING_SUM_AVG_VALUES = [10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]
HAVING_MIN_MAX_VALUES = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
DEFAULT_RANDOM_SEED = 42
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

RANGE_OPS = frozenset({">", "<", ">=", "<="})
IMPOSSIBLE_HAVING_RE = re.compile(
    r"^COUNT\b.*",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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

MAX_NON_AGG_COL_DIFF = 2

WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER: str = "rows"
WINDOW_DEFAULT_FRAME_START_WITH_ORDER: str = "UNBOUNDED PRECEDING"
WINDOW_DEFAULT_FRAME_END_WITH_ORDER: str = "CURRENT ROW"
WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER: str = "rows"
WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER: str = "UNBOUNDED PRECEDING"
WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER: str = "UNBOUNDED FOLLOWING"

PG_LAST_WINDOW_FRAME_OPTIONS_INLINE_DEFAULT: int = 1058
PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_UNBOUNDED_PAIR: int = 309
PG_LAST_WINDOW_FRAME_OPTIONS_RANGE_UNBOUNDED_CURRENT: int = 1075
PG_LAST_WINDOW_FRAME_OPTIONS_ROWS_OFFSET_CURRENT: int = 3093

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

RUNTIME_PARAPHRASE_COUNT: int = 4


def normalize_value_type(value_type: str) -> str:
    """
    Map a raw value-type string onto a canonical pipeline type.

    Args:

        value_type: LLM or schema value type.

    Returns:

        Normalised name from `VALUE_TYPE_NORMALIZATION` / `VALID_VALUE_TYPES`, else `'string'`.
    """
    if not value_type:
        return "string"
    vt_lower = value_type.lower().strip()
    if vt_lower in VALUE_TYPE_NORMALIZATION:
        return VALUE_TYPE_NORMALIZATION[vt_lower]
    if vt_lower in VALID_VALUE_TYPES:
        return vt_lower
    return "string"


def normalize_column_type(col_type: str) -> str:
    """
    Lowercase a SQL type and remove `(n)` / `(n,m)` parameter lists.

    Args:

        col_type: Raw SQL type (e.g. `VARCHAR(255)`).

    Returns:

        Base type name for lookup tables.
    """
    normalized = col_type.lower().strip()
    normalized = re.sub(r"\(\d+(?:,\s*\d+)?\)", "", normalized)
    normalized = normalized.strip()
    return normalized


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
        """
        Parse enum from code string or enum value.

        Legacy persisted ``"2"`` maps to :attr:`FUZZY_REUSE_FULL_PARAMS`. Legacy ``"4"`` maps to :attr:`UNION_TEMPLATE_AND_RUNTIME_WIDEN` when disambiguation metadata is absent.
        """
        if isinstance(value, GenerationPath):
            return value
        s = str(value).strip()
        if s == "2":
            return cls.FUZZY_REUSE_FULL_PARAMS
        if s == "4":
            return cls.UNION_TEMPLATE_AND_RUNTIME_WIDEN
        return cls(s)


def is_structural_param_key(key: str) -> bool:
    """Return True when *key* is a structural bind name (``s`` followed by digits)."""

    return len(key) >= 2 and key[0] == "s" and key[1:].isdigit()


def cost_cap_active(v: float | int | None) -> bool:
    """True when *v* is a positive finite bound; ``None``, ``0``, and negatives disable the cap."""

    if v is None:
        return False
    try:
        return float(v) > 0.0
    except (TypeError, ValueError):
        return False


def _read_optional_positive_float_env(name: str, *, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
    except ValueError:
        return default
    if v <= 0.0:
        return None
    return v


def _read_optional_positive_int_env(name: str, *, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return default
    if v <= 0:
        return None
    return v


class PolicyConfig:
    """
    ClassVar thresholds, penalties, stopwords, and SQL rejection patterns.

    Logging and tracing: set ClassVars ``DEBUG``, ``VERBOSE``, ``PIPELINE_TRACE_FULL``, or ``LIVE_DEEP_TRACE``. ``telemetry_capture(..., force_diagnostic_flags=True)`` bumps an internal depth counter so diagnostics emit into the capture buffer without mutating these ClassVars. Live tests opt in per session via ``live_tests/conftest.py`` so failures and optional full logs can be written to ``live_tests/results.txt``.

    Cache rebuild shortcuts: set ``REGENERATE_TEMPLATE_STORE``, ``REGENERATE_SCHEMA_GRAPH``, or ``REGENERATE_SKELETON_CACHE`` to skip loading the corresponding on-disk artifact when present.

    Semantic join hints (non-FK overlap): profiling keeps frequency ``top_k_values`` as today and loads a separate ascending distinct sample (``SEMANTIC_JOIN_ASC_DISTINCT_LIMIT``) for overlap; ``compute_semantic_profile_join_neighbors`` stores symmetric edges on ``ColumnMetadata.semantic_join_neighbors``. ``SEMANTIC_JOIN_MIN_OVERLAP_RATIO`` is the minimum ``|intersection| / min(|A|,|B|)`` on those two samples before an edge is recorded.
    """

    SCHEMA_CACHE_HASH_DEBUG_CLIP_CHARS: ClassVar[int] = 800

    JOIN_SHORTEST_PATH_TIE_CAP: ClassVar[int] = 4

    JOIN_CANDIDATE_CROSS_PRODUCT_CAP: ClassVar[int] = 16

    DEBUG: ClassVar[bool] = False
    VERBOSE: ClassVar[bool] = False
    PIPELINE_TRACE_FULL: ClassVar[bool] = False
    LIVE_DEEP_TRACE: ClassVar[bool] = False

    REGENERATE_TEMPLATE_STORE: ClassVar[bool] = False
    REGENERATE_SCHEMA_GRAPH: ClassVar[bool] = False
    REGENERATE_SKELETON_CACHE: ClassVar[bool] = False

    MAX_STAGE_B_REPAIRS: ClassVar[int] = 2
    MAX_FRESH_RESTARTS: ClassVar[int] = 1
    SEMANTIC_RESTART_REASONS: ClassVar[frozenset[str]] = frozenset({"semantic_oscillation", "semantic_max_rounds"})

    SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES: ClassVar[int] = 0

    SCHEMA_DESCRIPTION_TOP_VALUE_DISTINCT_RATIO_MAX: ClassVar[float] = 0.05

    MAX_USER_REFINEMENTS: ClassVar[int] = 1

    CATEGORICAL_MAX_RATIO = 0.05
    CATEGORICAL_MAX_CARDINALITY: ClassVar[int] = 50
    FREE_TEXT_CATEGORICAL_MAX_CARDINALITY = 200
    IDENTIFIER_MIN_UNIQUENESS = 0.98
    CATEGORICAL_SAMPLE_SIZE = 20

    UNUSABLE_NULL_RATIO_THRESHOLD = 0.99
    SENTINEL_MODE_FREQUENCY_THRESHOLD = 0.99
    INFERRED_PK_MIN_ROW_COUNT = 50
    FK_INFER_OVERLAP_MIN_RATIO = 0.10
    FK_INFER_OVERLAP_MIN_SAMPLE = 5

    SEMANTIC_JOIN_ASC_DISTINCT_LIMIT = 100
    SEMANTIC_JOIN_MIN_OVERLAP_RATIO = 0.15
    SEMANTIC_JOIN_MIN_DISTINCT = 4
    SEMANTIC_JOIN_MIN_INTERSECTION = 3

    FINAL_SQL_AUTO_ACCEPT_THRESHOLD = 0.95
    RESULT_ROW_COUNT_SOFT_WARNING: ClassVar[int] = 5_000
    FUZZY_MATCH_MAX_DISTANCE = 2
    QUESTION_TOKEN_INDEX_NEIGHBOR_CAP = 2048

    PENALTY_CAP = 0.30

    TRUST_PROMOTE_MAX_REJECT_RATIO = 0.25
    TRUST_PROMOTE_PER_QUESTION_ACCEPTS = 2
    PER_QUESTION_REJECT_OUT_THRESHOLD = 2

    PEN_BY_THREE_SOURCE_UNIT = 0.05

    MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION: ClassVar[int] = 8

    MAX_SUMMARY_BULLETS: ClassVar[int] = 6

    MAX_QUERY_COST_ROWS: ClassVar[float | None] = _read_optional_positive_float_env(
        "AETHERDIALECT_MAX_QUERY_COST_ROWS",
        default=50_000_000.0,
    )
    MAX_QUERY_COST_BYTES: ClassVar[float | None] = _read_optional_positive_float_env(
        "AETHERDIALECT_MAX_QUERY_COST_BYTES",
        default=50_000_000_000.0,
    )
    STATEMENT_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_STATEMENT_TIMEOUT_MS",
        default=30_000,
    )
    LLM_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_LLM_TIMEOUT_MS",
        default=60_000,
    )
    PROFILE_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_PROFILE_TIMEOUT_MS",
        default=120_000,
    )
    EXPLAIN_TIMEOUT_MS: ClassVar[int | None] = _read_optional_positive_int_env(
        "AETHERDIALECT_EXPLAIN_TIMEOUT_MS",
        default=None,
    )
    STOPWORDS = STOPWORDS_GRAMMATICAL_PARTICLES

    FORBIDDEN_SQL = [
        r"\bupdate\b",
        r"\bdelete\b",
        r"\binsert\b",
        r"\bmerge\b",
        r"\balter\b",
        r"\bdrop\b",
        r"\btruncate\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\bcreate\b",
        r"\bcomment\b",
        r"\brename\b",
        r"\bcall\b",
        r"\bexecute\b",
        r"\bdo\b",
        r"\bcopy\b",
        r";\s*\S",
        r"\bUNION\b",
        r"\bINTERSECT\b",
        r"\bEXCEPT\b",
        r"\bLATERAL\b",
        r"\bOFFSET\b",
        r"\bFETCH\s+FIRST\b",
        r"\bDISTINCT\s+ON\b",
        r"\bARRAY\s*\[",
        r"\bARRAY_AGG\b",
        r"::json\b",
        r"\bjson_",
        r"\bjsonb_",
        r"\bEXISTS\s*\(",
    ]


def effective_explain_timeout_ms() -> int | None:
    """
    Statement timeout for ``EXPLAIN`` paths only.

    Prefers :data:`PolicyConfig.EXPLAIN_TIMEOUT_MS` when set and positive; otherwise uses
    :data:`PolicyConfig.STATEMENT_TIMEOUT_MS`. Returns ``None`` when neither bound is active
    (``None`` / ``0`` / negative disable via :func:`cost_cap_active`).
    """

    if cost_cap_active(PolicyConfig.EXPLAIN_TIMEOUT_MS):
        return int(PolicyConfig.EXPLAIN_TIMEOUT_MS)  # type: ignore[arg-type]
    if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS):
        return int(PolicyConfig.STATEMENT_TIMEOUT_MS)
    return None


def effective_llm_timeout_ms() -> int:
    """
    Resolved HTTP timeout for OpenAI-compatible clients and :func:`aetherdialect._core_utils.llm_chat`.

    Uses :data:`PolicyConfig.LLM_TIMEOUT_MS` when positive; otherwise ``60_000`` ms.
    """

    tm = PolicyConfig.LLM_TIMEOUT_MS
    if cost_cap_active(tm):
        return int(tm)
    return 60_000


class PostgresRuntimeConfig:
    """PostgreSQL connection defaults and optional `SQL_FILE_PATH` (ClassVars)."""

    HOST: ClassVar[str] = "localhost"
    PORT: ClassVar[int] = 5432
    USER: ClassVar[str] = "postgres"
    PASSWORD: ClassVar[str | None] = None
    DATABASE: ClassVar[str | None] = None
    SCHEMA: ClassVar[str] = "public"

    SQL_FILE_PATH: ClassVar[str | None] = None

    @classmethod
    def db_url(cls) -> str:
        """
        Build a SQLAlchemy PostgreSQL URL from ClassVars, preferring ``psycopg`` when installed.

        Returns:

            SQLAlchemy connection URL.

        Raises:

            ValueError: If `PASSWORD` or `DATABASE` is unset.
        """
        if not cls.PASSWORD:
            raise ValueError("PostgreSQL password required")
        if not cls.DATABASE:
            raise ValueError("PostgreSQL database required")
        user_q = quote(str(cls.USER), safe="")
        pwd_q = quote(str(cls.PASSWORD), safe="")
        db_q = quote(str(cls.DATABASE), safe="")
        driver = "postgresql+psycopg2"
        if find_spec("psycopg") is not None:
            driver = "postgresql+psycopg"
        return f"{driver}://{user_q}:{pwd_q}@{cls.HOST}:{cls.PORT}/{db_q}"


class DatabricksRuntimeConfig:
    """Unity Catalog `CATALOG`/`SCHEMA` and optional ODBC connector settings (ClassVars)."""

    CATALOG: ClassVar[str | None] = None
    SCHEMA: ClassVar[str | None] = None

    SQL_FILE_PATH: ClassVar[str | None] = None

    SERVER_HOSTNAME: ClassVar[str | None] = None
    HTTP_PATH: ClassVar[str | None] = None
    ACCESS_TOKEN: ClassVar[str | None] = None

    @classmethod
    def has_native_connection(cls) -> bool:
        """
        True when hostname, HTTP path, and access token are all non- empty.

        Returns:

            Whether `databricks-sql-connector` can be used.
        """
        return bool(cls.SERVER_HOSTNAME and cls.HTTP_PATH and cls.ACCESS_TOKEN)

    @classmethod
    def validate(cls) -> None:
        """
        Require `CATALOG` and `SCHEMA`.

        Returns:

            None.

        Raises:

            ValueError: If either identifier is missing.
        """
        if not cls.CATALOG:
            raise ValueError("Databricks catalog required")
        if not cls.SCHEMA:
            raise ValueError("Databricks schema required")

    @classmethod
    def sqlalchemy_url(cls) -> str | None:
        """
        Build a SQLAlchemy URL for the Databricks SQL connector when PAT credentials exist.

        Returns:

            URL string, or ``None`` when native warehouse credentials are not configured.
        """
        if not cls.has_native_connection():
            return None

        token = quote(cls.ACCESS_TOKEN or "", safe="")
        host = cls.SERVER_HOSTNAME or ""
        http_path = quote(cls.HTTP_PATH or "", safe="")
        catalog = quote(cls.CATALOG or "", safe="")
        schema = quote(cls.SCHEMA or "", safe="")
        return f"databricks://token:{token}@{host}?http_path={http_path}&catalog={catalog}&schema={schema}"


SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PGPASSWORD",
        "POSTGRES_PASSWORD",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "DATABRICKS_TOKEN",
        "DATABRICKS_ACCESS_TOKEN",
    },
)


class EngineConfig:
    """
    Internal process-wide defaults for backend selection (`TYPE`/`RUNTIME`), LLM credentials/models, and JSON artifact paths.

    This class is not part of the public API and is not exported from the ``aetherdialect`` package root. The only supported user-facing configuration paths are the documented environment variables (for example ``AZURE_OPENAI_DEPLOYMENT_LIGHT`` and sibling slot variables) and the ``SchemaContext`` object passed to public entry points.
    """

    TYPE: ClassVar[str] = "postgresql"

    RUNTIME: ClassVar[type] = PostgresRuntimeConfig

    API_TOKEN: ClassVar[str | None] = os.environ.get("OPENAI_API_KEY")
    AZURE_API_TOKEN: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_API_KEY")
    LLM_PROVIDER: ClassVar[str] = "openai"
    OPENAI_MODEL: ClassVar[str] = "gpt-4o-mini"
    OPENAI_MODEL_INTENT: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_MODEL_JOIN: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_MODEL_SCHEMA_BASE: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_DDL: ClassVar[str] = "gpt-4.1-mini"
    OPENAI_MODEL_SCHEMA: ClassVar[str] = "gpt-5.4-mini"
    OPENAI_BASE_URL: ClassVar[str | None] = "https://api.openai.com/v1"
    AZURE_OPENAI_BASE_URL: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_BASE_URL")
    AZURE_OPENAI_ENDPOINT: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: ClassVar[str | None] = os.environ.get("AZURE_OPENAI_API_VERSION")

    SCHEMA_JSON_PATH: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    TEMPLATE_STORE_DIR: ClassVar[str] = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "intent_templates")

    @classmethod
    def azure_base_url(cls) -> str | None:
        """Return Azure OpenAI base URL in v1 form when configured."""
        if cls.AZURE_OPENAI_BASE_URL:
            return cls.AZURE_OPENAI_BASE_URL.rstrip("/")
        if cls.AZURE_OPENAI_ENDPOINT:
            return f"{cls.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1"
        return None


def llm_credentials_configured() -> bool:
    """
    Return True when at least one provider has its full credential set on ``EngineConfig``.

    OpenAI requires a non-empty :attr:`EngineConfig.API_TOKEN`. Azure OpenAI requires a non-empty :attr:`EngineConfig.AZURE_API_TOKEN`, :attr:`EngineConfig.AZURE_OPENAI_ENDPOINT` (or :attr:`EngineConfig.AZURE_OPENAI_BASE_URL`), and :attr:`EngineConfig.AZURE_OPENAI_API_VERSION`. Returns ``True`` when either provider is fully configured.
    """

    def _non_empty_str(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    openai_ok = _non_empty_str(EngineConfig.API_TOKEN)
    azure_ok = (
        _non_empty_str(EngineConfig.AZURE_API_TOKEN)
        and _non_empty_str(EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL)
        and _non_empty_str(EngineConfig.AZURE_OPENAI_API_VERSION)
    )
    return openai_ok or azure_ok


class QSimConfig:
    """QSim generation limits, ratios, sampling, output paths, and skeleton reference."""

    INTENT_TYPES = 20
    QUESTIONS_COUNT = 100
    MAX_TABLES_PER_INTENT = 3
    MAX_FILTERS_PER_INTENT = 4
    MAX_FILTER_COLUMNS = 2
    MAX_GROUP_BY_COLUMNS = 2

    MIN_AVG_VARIANTS_PER_INTENT = 1
    MAX_AVG_VARIANTS_PER_INTENT = 10

    MAX_NO_VARIANCE_RATIO = 0.25
    SINGLE_TABLE_RATIO = 0.40
    TWO_TABLE_RATIO = 0.40
    THREE_TABLE_RATIO = 0.20

    MAX_CONSECUTIVE_DUPLICATES = 5
    MAX_CONSECUTIVE_FAILURES = 5

    MIN_FILTER_RATIO = 0.70
    MIN_HAVING_RATIO = 0.15
    MIN_THREE_TABLE_RATIO = 0.10

    PROFILING_SAMPLE_THRESHOLD = 100_000
    PROFILING_SAMPLE_SIZE = 10_000

    RANDOM_SEED = DEFAULT_RANDOM_SEED

    SELECT_COL_GEOMETRIC_P: float = 0.6

    COMPLEXITY_TARGET_PROPORTIONS: dict[str, float] = {
        "simple": 0.20,
        "moderate": 0.40,
        "complex": 0.30,
        "highly_complex": 0.10,
    }

    EXCLUDED_FILTER_PATTERNS = EXCLUDED_FILTER_PATTERNS

    SKELETONS_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "qsim_skeletons.json.gz")

    MAX_ROLE_CLASSIFICATION_RETRIES = 2


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


SEED_NORMALIZATION_BATCH_SIZE: int = 20

INTERACTIVE_STAGE_DIRECT_REUSE = "direct_reuse_confirm"
INTERACTIVE_STAGE_INTENT_CONFIRM = "intent_confirm"
INTERACTIVE_STAGE_SQL_FEEDBACK = "sql_result_confirm"

PIPELINE_SUSPEND_ID_DIRECT_REUSE = "awaiting_direct_reuse_confirmation"
PIPELINE_SUSPEND_ID_INTENT_CONFIRM = "awaiting_intent_confirmation"
PIPELINE_SUSPEND_ID_INTENT_FEEDBACK = "awaiting_intent_rejection_feedback"
PIPELINE_SUSPEND_ID_SQL = "awaiting_sql_result_confirmation"
PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT = "awaiting_user_feedback_reject_reason"
PIPELINE_BUG_SQL_VALIDATION = "pipeline_bug_sql_validation"


POSTGRES_ENV_DATABASE: tuple[str, ...] = ("PGDATABASE", "POSTGRES_DB")
POSTGRES_ENV_USER: tuple[str, ...] = ("PGUSER", "POSTGRES_USER")
POSTGRES_ENV_PASSWORD: tuple[str, ...] = ("PGPASSWORD", "POSTGRES_PASSWORD")
POSTGRES_ENV_HOST: tuple[str, ...] = ("PGHOST", "PGHOSTADDR", "POSTGRES_HOST")
POSTGRES_ENV_PORT: tuple[str, ...] = ("PGPORT", "POSTGRES_PORT")
POSTGRES_ENV_SCHEMA: tuple[str, ...] = ("PGSCHEMA", "POSTGRES_SCHEMA")


DATABRICKS_ENV_SERVER_HOSTNAME: tuple[str, ...] = (
    "DATABRICKS_HOST",
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

OPENAI_ENV_REQUIRED: tuple[str, ...] = ("OPENAI_API_KEY",)

AZURE_OPENAI_ENV_REQUIRED: tuple[str, ...] = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)

AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT: str = "AZURE_OPENAI_DEPLOYMENT_LIGHT"
AZURE_OPENAI_ENV_DEPLOYMENT_MEDIUM: str = "AZURE_OPENAI_DEPLOYMENT_MEDIUM"
AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY: str = "AZURE_OPENAI_DEPLOYMENT_HEAVY"

PROFILING_TOP_K: int = 50


@dataclass(frozen=True, slots=True)
class LlmExecutionConfig:
    """
    Merged Azure OpenAI credentials plus execution cost and timeout limits for the engine runtime.

    Public operators configure three deployment slots named ``LIGHT``, ``MEDIUM``, and ``HEAVY`` that provision Azure deployments sized for the ``gpt-4o-mini``, ``gpt-4.1-mini``, and ``gpt-5.4-mini`` model classes respectively.

    Internal routing from logical model identifiers to these slots is not part of the public stability contract.
    """

    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    deployment_light: str
    deployment_medium: str
    deployment_heavy: str
    max_query_cost_rows: int
    max_query_cost_bytes: int
    statement_timeout_ms: int
    llm_timeout_ms: int
    profile_timeout_ms: int
    explain_timeout_ms: int | None


def load_runtime_config(
    *,
    merged_env: Mapping[str, str],
) -> LlmExecutionConfig:
    """
    Merge built-in defaults with a caller-supplied environment snapshot into one frozen LLM execution config.

    The *merged_env* mapping must be the effective environment produced during engine initialisation so runtime credentials do not drift from the configured process snapshot.

    Resolution order is defaults first, then the environment layer keyed by the canonical Azure OpenAI and execution-limit variable names.

    Args:

        merged_env: Mapping of effective environment strings used for the environment merge layer.

    Returns:

        The frozen :class:`LlmExecutionConfig`.

    Raises:

        ValueError: When numeric fields are negative after merge.
    """

    def _env_text(name: str) -> str:
        return str(merged_env.get(name, "") or "").strip()

    defaults: dict[str, Any] = {
        "azure_endpoint": "",
        "azure_api_key": "",
        "azure_api_version": "",
        "deployment_light": "",
        "deployment_medium": "",
        "deployment_heavy": "",
        "max_query_cost_rows": 50_000_000,
        "max_query_cost_bytes": 50_000_000_000,
        "statement_timeout_ms": 30_000,
        "llm_timeout_ms": 60_000,
        "profile_timeout_ms": 120_000,
        "explain_timeout_ms": None,
    }
    env_map: dict[str, str] = {
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_api_key": "AZURE_OPENAI_API_KEY",
        "azure_api_version": "AZURE_OPENAI_API_VERSION",
        "deployment_light": AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT,
        "deployment_medium": AZURE_OPENAI_ENV_DEPLOYMENT_MEDIUM,
        "deployment_heavy": AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY,
        "max_query_cost_rows": "AETHERDIALECT_MAX_QUERY_COST_ROWS",
        "max_query_cost_bytes": "AETHERDIALECT_MAX_QUERY_COST_BYTES",
        "statement_timeout_ms": "AETHERDIALECT_STATEMENT_TIMEOUT_MS",
        "llm_timeout_ms": "AETHERDIALECT_LLM_TIMEOUT_MS",
        "profile_timeout_ms": "AETHERDIALECT_PROFILE_TIMEOUT_MS",
        "explain_timeout_ms": "AETHERDIALECT_EXPLAIN_TIMEOUT_MS",
    }
    merged: dict[str, Any] = dict(defaults)
    for canon, env_name in env_map.items():
        raw = _env_text(env_name)
        if not raw:
            continue
        if canon in {
            "max_query_cost_rows",
            "max_query_cost_bytes",
            "statement_timeout_ms",
            "llm_timeout_ms",
            "profile_timeout_ms",
        }:
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            if iv < 0:
                raise ValueError(f"Invalid non-negative integer for {env_name}")
            merged[canon] = iv
        elif canon == "explain_timeout_ms":
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            merged[canon] = None if iv <= 0 else iv
        else:
            merged[canon] = raw
    for name in (
        "max_query_cost_rows",
        "max_query_cost_bytes",
        "statement_timeout_ms",
        "llm_timeout_ms",
        "profile_timeout_ms",
    ):
        v = merged.get(name)
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"Invalid runtime config for {name}")
    exm = merged.get("explain_timeout_ms")
    if exm is not None and (not isinstance(exm, int) or exm < 0):
        raise ValueError("Invalid runtime config for explain_timeout_ms")
    cfg = LlmExecutionConfig(
        azure_endpoint=str(merged.get("azure_endpoint") or ""),
        azure_api_key=str(merged.get("azure_api_key") or ""),
        azure_api_version=str(merged.get("azure_api_version") or ""),
        deployment_light=str(merged.get("deployment_light") or ""),
        deployment_medium=str(merged.get("deployment_medium") or ""),
        deployment_heavy=str(merged.get("deployment_heavy") or ""),
        max_query_cost_rows=int(merged["max_query_cost_rows"]),
        max_query_cost_bytes=int(merged["max_query_cost_bytes"]),
        statement_timeout_ms=int(merged["statement_timeout_ms"]),
        llm_timeout_ms=int(merged["llm_timeout_ms"]),
        profile_timeout_ms=int(merged["profile_timeout_ms"]),
        explain_timeout_ms=merged.get("explain_timeout_ms"),
    )
    return cfg


MIGRATION_DATA_OVERLAP_MIN: float = 0.15
MIGRATION_TABLE_RENAME_COLUMN_FRACTION: float = 0.60

SESSION_KIND_IDLE: str = "idle"
SESSION_KIND_AWAITING_INTENT_CONFIRM: str = "awaiting_intent_confirm"
SESSION_KIND_AWAITING_INTENT_FEEDBACK: str = "awaiting_intent_feedback"
SESSION_KIND_AWAITING_SQL_CONFIRM: str = "awaiting_sql_confirm"
SESSION_KIND_AWAITING_SQL_FEEDBACK: str = "awaiting_sql_feedback"
SESSION_KIND_AWAITING_MIGRATION_CONFIRM: str = "awaiting_migration_confirm"
SESSION_KIND_RESULT: str = "result"
SESSION_KIND_ERROR: str = "error"
SESSION_KIND_DONE: str = "done"

YES_NO_SESSION_KINDS: frozenset[str] = frozenset(
    {
        SESSION_KIND_AWAITING_INTENT_CONFIRM,
        SESSION_KIND_AWAITING_SQL_CONFIRM,
        SESSION_KIND_AWAITING_MIGRATION_CONFIRM,
    },
)

SUSPEND_ID_TO_SESSION_KIND: dict[str, str] = {
    PIPELINE_SUSPEND_ID_DIRECT_REUSE: SESSION_KIND_AWAITING_SQL_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM: SESSION_KIND_AWAITING_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_INTENT_FEEDBACK: SESSION_KIND_AWAITING_INTENT_FEEDBACK,
    PIPELINE_SUSPEND_ID_SQL: SESSION_KIND_AWAITING_SQL_CONFIRM,
    PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT: SESSION_KIND_AWAITING_SQL_FEEDBACK,
}

SEED_NORMALIZATION_JSON = "seed_question_normalization.json"
NORMALIZED_SEEDS_TXT = "seed_questions_normalized.txt"
QSIM_QUESTIONS_PATTERN = "qsim_questions_v{version}.txt"

JOIN_CHOICE_SCOPE_MAIN: str = "main"

SCHEMA_CONTEXT_CACHE_NAME: str = "schema_context.json"
SCHEMA_CONTEXT_CACHED_DDL: str = "_cached_schema_context.sql"
SCHEMA_CONTEXT_CACHED_NOTES: str = "_cached_schema_context_notes.txt"
SCHEMA_CONTEXT_CACHE_VERSION: int = 2

REGISTRY_REF_TOKEN_RE = re.compile(r"^[wc]\d{2}$")
CASE_RESULT_BARE_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CASE_RESULT_REGISTRY_TOKEN_RE = re.compile(r"^[wc]\d{2}$")

ROLE_VALUE_TYPE_COMPAT: dict[str, frozenset[str]] = {
    "boolean": frozenset({"boolean", "integer", "string"}),
    "numeric_measure": frozenset({"integer", "number"}),
    "numeric_categorical": frozenset({"integer", "number"}),
    "temporal": frozenset({"date"}),
    "free_text": frozenset({"string"}),
    "categorical": frozenset({"string", "integer", "number", "boolean"}),
    "identifier": frozenset({"string", "integer", "number"}),
}

REALISM_DROP_REASON_CATEGORIES: frozenset[str] = frozenset(
    {
        "nonsensical_sql",
        "tautology",
        "overly_narrow_filter",
        "pii_smell",
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
WARMUP_OPERATOR_FEATURE_TUPLE_4BIT_CARDINALITY: int = 16


class SeedWarmupConfig:
    """Seed warmup expansion depth, artifact paths, sampling caps, and date/limit presets."""

    MAX_SEED_QUESTIONS: int = 500

    MAX_FILTERS = 3
    MAX_TABLES = 3
    MAX_GROUPBY = 2

    MAX_EXPR_COMPARISONS = 2
    MAX_HAVING_CONDITIONS = 2
    MAX_EXPANSION_DEPTH = 2

    SEED_WARMUP_BUNDLE_PATTERN = "seed_warmup_v{version}.zip"
    SEED_WARMUP_REPORT_PATTERN = "seed_warmup_report_v{version}.json"
    WARMUP_PREFLIGHT_REPORT_PATTERN = "warmup_preflight_report_v{version}.json"
    SEED_WARMUP_CACHE_ZIP = "seed_warmup_cache.zip"
    WARMUP_CACHE_MANIFEST = "cache_manifest.json"
    WARMUP_CACHE_WORK_PREFIX = "work_units/"
    WARMUP_CACHE_GOLD_INTENTS_JSON = "gold/gold_intents.json"

    WARMUP_TARGET_CAP: int = 2000
    WARMUP_KEEP_ALL_BELOW: int = 2000
    WARMUP_MIN_GOLD_FRACTION: float = 0.15
    WARMUP_MAX_FILLBACK_ROUNDS: int = 2
    WARMUP_STRATUM_MIN: int = 2
    WARMUP_QUESTIONS_MAX: int = 3
    WARMUP_SAMPLING_POLICY_VERSION: str = "2"
    MAX_WARMUP_EXECUTE_UNITS: int = 500_000
    SEED_WARMUP_CODE_VERSION: str = "1"

    WARMUP_ANCHOR_LATTICE_SUBDIR: str = "anchor_lattice"
    WARMUP_ANCHOR_LATTICE_CODE_VERSION: str = "1"

    WARMUP_QUESTION_STYLES: tuple[str, ...] = (
        "formal",
        "colloquial",
        "imperative",
        "interrogative",
        "descriptive",
        "concise",
    )

    WARMUP_QUESTION_STYLE_GUIDANCE: dict[str, str] = {
        "formal": "Polished professional analyst tone; complete sentences; no slang.",
        "colloquial": "Casual everyday wording as a colleague would speak.",
        "imperative": "Lead with a verb; compact command-style request.",
        "interrogative": "Clear question form using wh-words or how as appropriate.",
        "descriptive": "Neutral narrative statement of the insight or figures requested.",
        "concise": "Minimal words; one short sentence or tight fragment only.",
    }

    COMPLEXITY_TARGET_PROPORTIONS: dict[str, float] = {
        "simple": 0.20,
        "moderate": 0.40,
        "complex": 0.30,
        "highly_complex": 0.10,
    }

    RULE_NLG_ANCHOR_COUNT: int = 12

    WARMUP_LLM_DIVERSITY_SUBSAMPLE_DIVISOR: int = 4

    WARMUP_MMR_LAMBDA: float = 0.7

    WARMUP_DIAGNOSTIC_REPAIR_MAX_ROUNDS: int = 1

    RANDOM_SEED = DEFAULT_RANDOM_SEED

    EXTRACT_EXPANSION_UNITS: list[str] = ["year", "month", "day", "quarter", "dow"]
    DATE_TRUNC_EXPANSION_UNITS: list[str] = ["month", "quarter", "year"]
    LIMIT_EXPANSION_VALUES: list[int] = [10, 50, 100]

    DATE_WINDOW_EXPANSION_PRESETS: list[dict[str, int | str]] = [
        {"unit": "day", "amount": 7},
        {"unit": "day", "amount": 30},
        {"unit": "day", "amount": 90},
        {"unit": "month", "amount": 1},
        {"unit": "month", "amount": 3},
        {"unit": "month", "amount": 6},
        {"unit": "month", "amount": 12},
        {"unit": "year", "amount": 1},
    ]

    DATE_DIFF_EXPANSION_PRESETS: list[dict[str, int | str]] = [
        {"unit": "day", "amount": 7},
        {"unit": "day", "amount": 30},
        {"unit": "day", "amount": 90},
    ]


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
        "preflight_skipped",
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

WARMUP_ROUND_TRIP_LIMIT: int = 100

SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM: str = "<num>"
SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR: str = "<str>"
SQL_TO_INTENT_LITERAL_PLACEHOLDER_DATE: str = "<date>"
SQL_TO_INTENT_PARAM_KEY_PREFIX: str = "sql_hist_lit_"

SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY: str = "sql_hist_limit_offset"

WARMUP_PARAPHRASE_COUNT_FROM_SQL: int = 5

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


EMPTY_JOIN_CANDIDATES: dict[str, Any] = {"candidates": []}


def seed_warmup_failure_code_from_validate_sql_error(
    message: str | None,
    *,
    failure_category: str | None = None,
) -> str:
    """Map ``validate_sql`` outcome to a seed-warmup validation failure code."""

    if failure_category:
        exec_bucket = {
            "execution_explain_failed": "explain_failed",
            "execution_timeout": "explain_transient",
            "execution_cost_exceeded": "explain_failed",
            "execution_schema_error": "explain_schema",
            "execution_semantic_error": "explain_semantic",
            "execution_other_error": "explain_failed",
        }
        hit = exec_bucket.get(failure_category)
        if hit is not None:
            return hit
        if failure_category == "schema" and (message or "").strip() == "not_select":
            return "ast_validate_unsupported_construct"
        if failure_category == "other" and (message or "").strip() == "forbidden_sql":
            return "ast_validate_other"
        if failure_category == "unbound_placeholder":
            return "ast_validate_unbound_placeholder"

    if not message:
        return "ast_validate_other"
    m = message.strip()
    for tag in (
        "explain_schema",
        "explain_semantic",
        "explain_transient",
        "explain_failed",
    ):
        if m.startswith(f"[{tag}]"):
            return tag
    if m == "not_select":
        return "ast_validate_unsupported_construct"
    if m == "forbidden_sql":
        return "ast_validate_other"
    if m == "unbound_placeholder":
        return "ast_validate_unbound_placeholder"
    low = m.lower()
    if "sql structure error:" in low:
        tail = m.split("SQL structure error:", 1)[-1].strip().lower()
        if "cte" in tail or "with " in tail:
            return "ast_validate_cte_error"
        if "from" in tail and ("missing" in tail or "no from" in tail):
            return "ast_validate_missing_from_clause"
        if "column" in tail and ("not exist" in tail or "undefined" in tail or "bad" in tail):
            return "ast_validate_bad_identifier"
        if "syntax" in tail or "parse" in tail:
            return "ast_validate_pglast_syntax"
        return "ast_validate_other"
    if "syntax" in low or "parse" in low:
        return "ast_validate_pglast_syntax"
    return "ast_validate_other"


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
SQL_BIND_TOKEN_RE: re.Pattern[str] = re.compile(r":(p\d+|s\d+)\b")
UNBOUND_PYFORMAT_PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)s|(?<!\w)%s(?!\w)")

PG_JOIN_NODE_TYPES: frozenset[str] = frozenset({"Nested Loop", "Hash Join", "Merge Join"})
PG_JOIN_CONDITION_KEYS: tuple[str, ...] = ("Join Filter", "Hash Cond", "Merge Cond")
PG_INNER_CONDITION_KEYS: tuple[str, ...] = ("Index Cond", "Recheck Cond", "Filter")
DBR_CARTESIAN_TOKENS: tuple[str, ...] = ("CartesianProduct", "BroadcastNestedLoopJoin")
EXPLAIN_PERMISSION_DENIED_PATTERNS: tuple[str, ...] = (
    "permission denied",
    "insufficient privilege",
    "access denied",
    "not authorized",
    "does not have permission",
    "does not have access",
    "operation not permitted",
)
SQLGLOT_DIALECT_BY_ENGINE: dict[str, str] = {
    "postgresql": "postgres",
    "databricks": "spark",
}
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
VALID_COLUMN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"description", "sensitivity", "pii", "role", "boolean_truth_value"},
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
        "hygiene",
        "strict",
        "forbidden",
        "pii",
        "restricted",
    ],
    "foreign_key_kind": ["structural", "semantic"],
}

TEMPLATE_STORE_SEGMENT: str = "intent_templates"
TEMPLATE_STORE_HEADER_FILENAME: str = "header.json.gz"
TEMPLATE_STORE_PARTITION_PREFIX: str = "partition_"
TEMPLATE_STORE_PARTITION_COUNT: int = 256
TEMPLATE_STORE_PARTITION_LRU_MAX: int = 32
TEMPLATE_STORE_LEGACY_SINGLE_FILE: str = "intent_templates.json.gz"

LEGACY_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "schema_graph.json.gz",
    TEMPLATE_STORE_LEGACY_SINGLE_FILE,
    "qsim_skeletons.json.gz",
    SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP,
)
LEGACY_ARTIFACT_GLOBS: tuple[str, ...] = (
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
)

SIMULATION_CACHE_EXACT_FILENAMES: tuple[str, ...] = (
    "qsim_skeletons.json.gz",
    "qsim_summary.json",
    SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP,
)

SIMULATION_CACHE_GLOB_PATTERNS: tuple[str, ...] = (
    "qsim_questions_v*.txt",
    "seed_warmup_report_v*.json",
    "seed_warmup_v*.zip",
    "warmup_preflight_report_v*.json",
    "warmup_preflight_drops_v*.jsonl",
    "warmup_preflight_drops_detail_v*.jsonl",
    "qsim_*.json.gz",
    "qsim_summary_*.json.gz",
    "qsim_skeletons_*.json.gz",
    f"{SeedWarmupConfig.WARMUP_ANCHOR_LATTICE_SUBDIR}/*",
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

SOFT_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        "explain_seq_scan_indexed",
        "explain_zero_estimate",
    }
)

MAX_REPAIR_ATTEMPTS_PER_CODE: int = 1
DIAGNOSTIC_FUZZY_CUTOFF: float = 0.6

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
