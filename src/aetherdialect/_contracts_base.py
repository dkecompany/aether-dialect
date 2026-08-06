"""Shared dataclasses and enums for schema graphs, validation, templates, QSim skeletons, and type helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any, ClassVar, Literal, Protocol, cast

import pandas

from ._constants import (
    BUSINESS_KNOWLEDGE_COLUMN_REF_RE,
    BUSINESS_KNOWLEDGE_DEFAULT_KIND,
    COLUMN_TYPE_TO_VALUE_TYPE,
    CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC,
    DATE_TYPE_TOKENS,
    DEFAULT_NULL_ORDERING_ASC,
    DEFAULT_NULL_ORDERING_DESC,
    DESCRIPTION_OWNER_VALUES,
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    FEDERATION_TIMEZONE_AWARE_DATA_TYPES,
    FIXED_WIDTH_TEXT_BASE_TYPES,
    INFERENCE_TAG_VALUES,
    MAX_FLOAT_SAFE_INTEGER,
    MAX_PREDICATE_NESTING_DEPTH,
    MYSQL_TIMESTAMP_ENGINES,
    NUMERIC_TYPE_ARGUMENTS_RE,
    NUMERIC_TYPE_TOKENS,
    OP_FLIP,
    PK_INFERENCE_TAG_VALUES,
    RAW_SQL_AGG_OR_WINDOW_RE,
    REFUSAL_CATALOGUE,
    REGISTRY_REF_TOKEN_RE,
    ROLE_OWNER_VALUES,
    STRING_TYPE_TOKENS,
    STRUCTURAL_DATA_TYPE_CANONICAL,
    UNSIGNED_INTEGER_TYPE_MAX,
)


class SchemaInclude(StrEnum):
    """Catalog reflection include mode for tables versus views."""

    TABLES = "tables"
    VIEWS = "views"

    @classmethod
    def coerce(cls, raw: Any) -> SchemaInclude:
        """Normalize a stored include string to a supported literal."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        if s.endswith(".TABLES"):
            return cls.TABLES
        if s.endswith(".VIEWS"):
            return cls.VIEWS
        return cls.TABLES


class SchemaRole(StrEnum):
    """Artifact ownership role for construction and write gates."""

    OWNER = "owner"
    CONSUMER = "consumer"

    @classmethod
    def coerce(cls, raw: Any) -> SchemaRole:
        """Normalize a stored or in-process role value to a ``SchemaRole`` member."""
        if isinstance(raw, cls):
            return raw
        s = str(getattr(raw, "value", raw) or "").strip().lower()
        for member in cls:
            if member.value == s:
                return member
        if s.endswith(".owner"):
            return cls.OWNER
        if s.endswith(".consumer"):
            return cls.CONSUMER
        raise ValueError(f"invalid schema role: {raw!r}")


class ApprovalState(StrEnum):
    """Registry approval gate for silent reuse and ``execute_template``."""

    APPROVED = "approved"
    PENDING = "pending"


class BusinessKnowledgeKind(StrEnum):
    """Closed vocabulary for business-knowledge entry kinds."""

    GLOSSARY = "glossary"
    POLICY = "policy"
    METRIC = "metric"
    SYNONYM = "synonym"
    CAVEAT = "caveat"


class QuestionRoute(StrEnum):
    """Validation gate route for an ask turn."""

    ANALYTICAL = "analytical"
    SCHEMA_CATALOG = "schema_catalog"
    BUSINESS_KNOWLEDGE = "business_knowledge"
    RESTRICTED = "restricted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class QuestionValidationResult:
    """Result of the ask-path question validation LLM gate."""

    accepted: bool
    route: QuestionRoute
    corrected: str


class ResultReaderKind(StrEnum):
    """Row-fetch backend identifier used by dialect execution paths."""

    SQLALCHEMY = "sqlalchemy"
    SPARK = "spark"
    CONNECTOR = "connector"
    BQ_CLIENT = "bq_client"
    BQ_STORAGE = "bq_storage"
    SNOWFLAKE_ARROW = "snowflake_arrow"


class FederationMethodScope(StrEnum):
    """Which federation surface a public method may target."""

    COMPOSITE = "composite"
    MEMBER = "member"
    BOTH = "both"
    UNSUPPORTED = "unsupported"


class FederationTopologyChange(StrEnum):
    """Roster delta between recorded federation members and the manifest."""

    NONE = "none"
    ADD = "add"
    REMOVE = "remove"
    MIXED = "mixed"


class TableKind(StrEnum):
    """Reflected relation kind from information_schema / engine catalogs."""

    TABLE = "table"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"


class ArrayStorageKind(StrEnum):
    """How array-valued columns are physically stored for containment ops."""

    NATIVE_ARRAY = "native_array"
    JSON_TEXT_ARRAY = "json_text_array"
    UNKNOWN = "unknown"


class OverlapComparison(StrEnum):
    """Value-overlap comparison rule when pairing profiled columns."""

    EXACT = "exact"
    CASE_FOLDED = "case_folded"


class FeedbackMode(StrEnum):
    """Whether interactive feedback is collected live or deferred in tests."""

    LIVE = "live"
    DEFERRED_TEST = "deferred_test"


class SandboxPreset(StrEnum):
    """Closed-world sandbox construction preset."""

    OWNER_WRITER = "owner_writer"
    CONSUMER_READER = "consumer_reader"
    FEDERATION = "federation"

    @classmethod
    def coerce(cls, raw: Any) -> SandboxPreset:
        """Normalize a sandbox preset label to a supported member."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"unsupported sandbox preset: {raw!r}")


class SandboxBuildSection(StrEnum):
    """Named sandbox corpus build section."""

    VALIDATION_FAILURES = "validation_failures"
    FEEDBACK_SAMPLES = "feedback_samples"

    @classmethod
    def coerce(cls, raw: Any) -> SandboxBuildSection:
        """Normalize a sandbox build-section label to a supported member."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"unsupported sandbox build section: {raw!r}")


class SandboxLlmMode(StrEnum):
    """Whether the sandbox LLM path is mocked or networked."""

    MOCK = "mock"
    NETWORK = "network"

    @classmethod
    def coerce(cls, raw: Any) -> SandboxLlmMode:
        """Normalize a sandbox LLM mode label to a supported member."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"unsupported sandbox llm mode: {raw!r}")


class WindowOperatorKind(StrEnum):
    """Window-operator family for operator feature vectors."""

    NONE = "none"
    RANK = "rank"
    AGGREGATE = "aggregate"
    NAVIGATE = "navigate"


class OrderByNullPlacement(StrEnum):
    """NULL ordering placement for ORDER BY."""

    FIRST = "first"
    LAST = "last"

    @classmethod
    def default_for_direction(cls, direction: str) -> OrderByNullPlacement:
        """Return DuckDB/PostgreSQL default null placement for an ORDER BY direction."""
        dir_up = (direction or "ASC").strip().upper()
        return cls(DEFAULT_NULL_ORDERING_DESC) if dir_up == "DESC" else cls(DEFAULT_NULL_ORDERING_ASC)

    @classmethod
    def coerce(cls, raw: Any) -> OrderByNullPlacement | None:
        """Normalize optional ORDER BY null-placement to ``first``, ``last``, or ``None``."""
        if raw in (cls.FIRST, cls.LAST, "first", "last"):
            return cls(str(raw))
        return None


class CteEmissionKind(StrEnum):
    """How a runtime CTE is emitted into the carrier SELECT."""

    JOIN_TABLE = "join_table"
    SCALAR_SUBQUERY = "scalar_subquery"
    SEMI_JOIN = "semi_join"
    ANTI_JOIN = "anti_join"

    @classmethod
    def coerce(cls, raw: Any) -> CteEmissionKind:
        """Normalize a stored emission string to a supported literal."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.JOIN_TABLE


class WindowFrameKind(StrEnum):
    """Window frame mode for PARTITION/ORDER windows."""

    ROWS = "rows"
    RANGE = "range"
    NONE = "none"

    @classmethod
    def coerce(cls, raw: Any) -> WindowFrameKind:
        """Normalize a window frame kind string to a supported member."""
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.NONE


@dataclass(frozen=True, slots=True)
class JoinEdge:
    """One JOIN to attach to a carrier SELECT. ``table`` is the bare physical table name being joined in. ``alias`` is the AS-alias used when the same physical table appears multiple times (self-join); ``None`` for a single-instance join. ``kind`` is ``"INNER"`` or ``"LEFT"``. Each ``on_terms`` tuple is ``(left_token, left_col, right_token, right_col)`` where the tokens are the table name or alias to qualify the column with in the ``ON`` clause."""

    table: str
    alias: str | None
    kind: Literal["INNER", "LEFT"]
    on_terms: tuple[tuple[str, str, str, str], ...] = field(default_factory=tuple)


class ComplexityTier(StrEnum):
    """Workload complexity band for QSim sampling and seed-warmup stratification."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class WarmupStyle(StrEnum):
    """Canonical warmup NL style keys aligned with SeedWarmupConfig.WARMUP_QUESTION_STYLES."""

    FORMAL = "formal"
    COLLOQUIAL = "colloquial"
    IMPERATIVE = "imperative"
    INTERROGATIVE = "interrogative"
    DESCRIPTIVE = "descriptive"
    CONCISE = "concise"
    KEYWORD = "keyword"
    BUSINESS_JARGON = "business_jargon"
    BEGINNER = "beginner"
    VERBOSE = "verbose"


class SensitivityClassification(StrEnum):
    """Single-column sensitivity tier for projection, filtering, and LLM visibility."""

    NONE = "none"
    RESTRICTED = "restricted"
    HIDDEN = "hidden"

    @classmethod
    def coerce(cls, raw: Any) -> SensitivityClassification | None:
        """Parse :class:`SensitivityClassification` tokens from overrides JSON or other string inputs."""
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        s = str(raw).strip().lower()
        if not s:
            return None
        for m in cls:
            if m.value == s:
                return m
        return None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SensitivityClassification:
        """Resolve a persisted ``sensitivity`` key into a :class:`SensitivityClassification`."""
        return cls.coerce(d.get("sensitivity")) or cls.NONE

    @classmethod
    def apply_to(cls, col: Any, value: SensitivityClassification | str | None) -> None:
        """Single writer for column sensitivity; clears profile samples when not NONE."""
        if value is None or (isinstance(value, str) and not str(value).strip()):
            resolved = cls.NONE
        elif isinstance(value, cls):
            resolved = value
        else:
            resolved = cls.coerce(str(value).strip().lower()) or cls.NONE
        col.sensitivity = resolved
        if resolved != cls.NONE:
            col.frequent_values = []
            col.value_overlap_sample = []
            col.min_val = None
            col.max_val = None


class NoveltyBand(StrEnum):
    """Expansion-depth novelty band for anchor-lattice sharing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkloadFamily(StrEnum):
    """High-level analyst workload shapes for sampling and lattice keys."""

    STATUS_REPORT = "status_report"
    BREAKDOWN = "breakdown"
    LEADERBOARD = "leaderboard"
    TREND = "trend"
    CHANGE_OVER_TIME = "change_over_time"
    SHARE_OF_TOTAL = "share_of_total"
    SEGMENT_COMPARISON = "segment_comparison"
    THRESHOLD_EXCEPTION = "threshold_exception"
    EXTRACT = "extract"
    LIFECYCLE_COHORT = "lifecycle_cohort"
    EXPLORATION_FOLLOWUP = "exploration_followup"


@dataclass(frozen=True)
class OperatorFeatureVector:
    """Boolean and bucketed operator footprint shared by QSim enumeration and seed-warmup coverage."""

    has_aggregate: bool
    has_grouping: bool
    has_having: bool
    window_kind: WindowOperatorKind
    has_self_join_via_cte: bool
    has_scalar_cte: bool
    has_unnest: bool
    has_case_when: bool
    has_date_window: bool
    has_date_diff: bool
    cte_depth_bucket: int
    join_breadth_bucket: int
    workload_family: WorkloadFamily


@dataclass(frozen=True)
class _AdvancedFeatureSpec:
    """Named advanced SQL intent capability surfaced to tier-conditioned QSim prompts."""

    feature_id: str
    summary: str
    example_fragment: str


@dataclass(frozen=True)
class _ComplexityTierSpec:
    """Declared bounds for one complexity band."""

    tier: ComplexityTier
    min_tables: int
    max_tables: int
    max_cte_steps: int
    allows_window: bool
    allows_multi_cte: bool
    summary: str
    example_sketch: str = ""


@dataclass(frozen=True)
class DatabaseFeatureCapability:
    """Feasibility snapshot derived from a live schema graph for tier and feature gating."""

    table_count: int
    fk_edge_count: int
    has_numeric_measures: bool
    has_date_columns: bool
    has_array_columns: bool
    has_categorical_columns: bool
    max_tables_on_any_join_path: int
    max_fk_chain_depth: int
    has_self_referential_fk: bool
    tables_supporting_self_join: frozenset[str]
    has_window_capable_table_sets: bool
    aggregatable_columns_by_table: dict[str, frozenset[str]]
    date_columns_by_table: dict[str, frozenset[str]]
    array_columns_by_table: dict[str, frozenset[str]]
    supports_semi_join: bool = True
    supports_anti_join: bool = True
    supports_predicate_nesting: bool = True
    supports_preserve_tables: bool = False
    supports_ordered_string_agg: bool = True
    supports_median: bool = True
    supports_stddev: bool = True
    supports_variance: bool = True
    supports_window_frames: bool = True
    supports_array_contains: bool = True
    supports_collation: bool = True
    supports_unsigned_semantics: bool = True
    supports_timestamptz_semantics: bool = True


@dataclass(frozen=True)
class _SurfaceTemplateSpec:
    """Declarative NL surface pattern for deterministic warmup anchoring."""

    construct_kind: str
    surface_forms: tuple[str, ...]


QSIM_SUPPORTED_ADVANCED_FEATURES: tuple[_AdvancedFeatureSpec, ...] = (
    _AdvancedFeatureSpec(
        feature_id="multi_cte_chain",
        summary="Stacked CTE definitions where one CTE references another.",
        example_fragment="WITH daily AS (...), rolled AS (SELECT ... FROM daily) SELECT ...",
    ),
    _AdvancedFeatureSpec(
        feature_id="scalar_cte_bridge",
        summary="Scalar-valued CTE row merged via cross join for threshold constants.",
        example_fragment="WITH params AS (SELECT 100.0 AS min_amt) SELECT ... FROM table CROSS JOIN params",
    ),
    _AdvancedFeatureSpec(
        feature_id="self_join_via_cte",
        summary="Second reference to a base table mediated through a named CTE.",
        example_fragment="WITH a1 AS (SELECT * FROM table) SELECT ... FROM table JOIN a1 ON ...",
    ),
    _AdvancedFeatureSpec(
        feature_id="window_partition_order",
        summary="ROW_NUMBER/RANK/SUM over PARTITION BY with ORDER BY.",
        example_fragment="ROW_NUMBER() OVER (PARTITION BY table.column ORDER BY table.other_column DESC)",
    ),
    _AdvancedFeatureSpec(
        feature_id="case_when_select",
        summary="CASE expressions in the projected SELECT list.",
        example_fragment="CASE WHEN table.column = 'x' THEN table.other_column ELSE 0 END",
    ),
    _AdvancedFeatureSpec(
        feature_id="date_window_filter",
        summary="Rolling calendar predicates using relative windows.",
        example_fragment="table.date_column >= CURRENT_DATE - INTERVAL '30 days'",
    ),
    _AdvancedFeatureSpec(
        feature_id="date_diff_shapes",
        summary="Difference-between-dates filters.",
        example_fragment="DATE_PART('day', table.end_date_column - table.start_date_column) > 3",
    ),
    _AdvancedFeatureSpec(
        feature_id="unnest_array_column",
        summary="EXPLODE/UNNEST typed array columns inside a subordinate SELECT.",
        example_fragment="FROM tags CROSS JOIN UNNEST(tag_ids) AS u(tag)",
    ),
    _AdvancedFeatureSpec(
        feature_id="distinct_select",
        summary="SELECT DISTINCT non-aggregated projections.",
        example_fragment="SELECT DISTINCT table.column FROM table",
    ),
    _AdvancedFeatureSpec(
        feature_id="ilike_predicate",
        summary="ILIKE / NOT ILIKE text predicates.",
        example_fragment="table.text_column ILIKE '%pattern%'",
    ),
    _AdvancedFeatureSpec(
        feature_id="having_aggregate_compare",
        summary="HAVING clauses comparing aggregated measures to literals.",
        example_fragment="HAVING SUM(table.column) > 5000",
    ),
)

QSIM_COMPLEXITY_TIER_SPECS: tuple[_ComplexityTierSpec, ...] = (
    _ComplexityTierSpec(
        tier=ComplexityTier.SIMPLE,
        min_tables=1,
        max_tables=1,
        max_cte_steps=0,
        allows_window=False,
        allows_multi_cte=False,
        summary="Single-table scans with optional equality or range filters; projections stay non-aggregated.",
        example_sketch="Show recent rows for a single entity by its id.",
    ),
    _ComplexityTierSpec(
        tier=ComplexityTier.MODERATE,
        min_tables=1,
        max_tables=2,
        max_cte_steps=0,
        allows_window=False,
        allows_multi_cte=False,
        summary="One or two joined tables with simple aggregates, light grouping, ORDER BY, or LIMIT.",
        example_sketch="Top 50 entities by a count over a recent period.",
    ),
    _ComplexityTierSpec(
        tier=ComplexityTier.COMPLEX,
        min_tables=2,
        max_tables=3,
        max_cte_steps=1,
        allows_window=True,
        allows_multi_cte=False,
        summary="Multi-table joins with grouped aggregates, HAVING, DISTINCT, or cross-column comparisons.",
        example_sketch="Average of a measure by group, keeping only groups with more than 100 related rows.",
    ),
    _ComplexityTierSpec(
        tier=ComplexityTier.HIGHLY_COMPLEX,
        min_tables=3,
        max_tables=3,
        max_cte_steps=3,
        allows_window=True,
        allows_multi_cte=True,
        summary="Dense shapes combining multiple predicates, aggregates at aligned grains, and ordered analytic heads.",
        example_sketch="Rank entities within each group by a contribution measure using layered filters.",
    ),
)


class FailureCategory(StrEnum):
    """Primary taxonomy for validation issues, rejections, avoid-example summaries, and intent failure log rows."""

    ACCESS_POLICY = "access_policy"
    DENIED_REFERENCE = "denied_reference"
    DENY_BARE_SELECT = "deny_bare_select"
    SENSITIVE_GROUP_BY = "sensitive_group_by"
    AGG_KEYWORD_MISSING = "agg_keyword_missing"
    AGGREGATION = "aggregation"
    AGGREGATION_HINT = "aggregation_hint"
    AGGREGATION_SEMANTICS = "aggregation_semantics"
    AGGREGATION_VALIDITY = "aggregation_validity"
    COLUMN_AMBIGUOUS = "column_ambiguous"
    COUNT_THRESHOLD_MISSING_HAVING = "count_threshold_missing_having"
    CTE_AGGREGATION = "cte_aggregation"
    CTE_CARDINALITY = "cte_cardinality"
    CTE_COLUMN_REFERENCE = "cte_column_reference"
    CTE_GRAIN_COMPATIBILITY = "cte_grain_compatibility"
    CTE_GRAIN_CONSISTENCY = "cte_grain_consistency"
    CTE_MISSING_JOIN_KEY = "cte_missing_join_key"
    CTE_STRUCTURE = "cte_structure"
    CTE_TABLE_REFERENCE = "cte_table_reference"
    CTE_TYPE_CONSISTENCY = "cte_type_consistency"
    CTE_USAGE = "cte_usage"
    DATE_DIFF = "date_diff"
    EXPRESSION_TYPE = "expression_type"
    EXTRACT_EPOCH = "extract_epoch"
    EXECUTION_EXPLAIN_FAILED = "execution_explain_failed"
    EXECUTION_OTHER_ERROR = "execution_other_error"
    EXECUTION_SCHEMA_ERROR = "execution_schema_error"
    EXECUTION_SEMANTIC_ERROR = "execution_semantic_error"
    EXECUTION_COST_EXCEEDED = "execution_cost_exceeded"
    EXECUTION_TIMEOUT = "execution_timeout"
    INTENT_ERROR = "intent_error"
    PERMISSION_ERROR = "permission_error"
    TRANSPORT_AUTH = "transport_auth"
    WHERE_AGGREGATION = "where_aggregation"
    WHERE_SEMANTIC = "where_semantic"
    WHERE_STRUCTURE = "where_structure"
    WHERE_VALIDITY = "where_validity"
    FOR_EACH_GROUPING = "for_each_grouping"
    GRAIN_CONSISTENCY = "grain_consistency"
    GRAIN_VALIDITY = "grain_validity"
    GROUP_BY_MEMBERSHIP = "group_by_membership"
    GROUP_BY_VALIDITY = "group_by_validity"
    HAVING_AGGREGATION = "having_aggregation"
    HAVING_SEMANTIC = "having_semantic"
    HAVING_VALIDITY = "having_validity"
    INTENT_PARSE_FAILED = "intent_parse_failed"
    INTENT_SCHEMA_INVALID_ABORT = "intent_schema_invalid_abort"
    INTENT_EMPTY_WINDOW = "intent_empty_window"
    INTENT_USER_DECLINED = "intent_user_declined"
    INTERPRETATION_MISMATCH = "interpretation_mismatch"
    MISSING_COLUMN = "missing_column"
    MISSING_DISTINCT = "missing_distinct"
    MISSING_NUMERIC_WHERE = "missing_numeric_where"
    MISSING_SCOPING_TABLE = "missing_scoping_table"
    MISSING_TEMPORAL_COLUMN = "missing_temporal_column"
    MIXED_AGGREGATION = "mixed_aggregation"
    NESTED_AGGREGATION = "nested_aggregation"
    ORDER_BY_AGGREGATION = "order_by_aggregation"
    ORDER_BY_VALIDITY = "order_by_validity"
    OPERATOR = "operator"
    OTHER = "other"
    PREDICATE_SIDEDNESS = "predicate_sidedness"
    REGISTRY = "registry"
    RESULT_OKAY_INTENT_WRONG = "result_okay_intent_wrong"
    SCHEMA = "schema"
    SCHEMA_VALIDATION = "schema_validation"
    SCALAR_SEMANTIC = "scalar_semantic"
    SCALAR_SEMANTICS = "scalar_semantics"
    SCALAR_VALIDITY = "scalar_validity"
    SELECT_VALIDITY = "select_validity"
    SENSITIVITY_ALL_SELECT_DROPPED = "sensitivity_all_select_dropped"
    SENSITIVITY_ALL_GROUP_BY_DROPPED = "sensitivity_all_group_by_dropped"
    STRUCTURAL = "structural"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    THRESHOLD_MISSING_HAVING = "threshold_missing_having"
    TYPE_ALIGNMENT = "type_alignment"
    TYPE_MISMATCH = "type_mismatch"
    UNBOUND_PLACEHOLDER = "unbound_placeholder"
    UNKNOWN_COLUMN = "unknown_column"
    UNKNOWN_TABLE = "unknown_table"
    WRONG_AGGREGATION = "wrong_aggregation"
    WRONG_COLUMN_SELECTION = "wrong_column_selection"
    WRONG_WHERE_LOGIC = "wrong_where_logic"
    WRONG_GRAIN = "wrong_grain"
    WRONG_HAVING = "wrong_having"
    WRONG_JOIN = "wrong_join"
    WRONG_SORT_OR_LIMIT = "wrong_sort_or_limit"
    WRONG_TABLES = "wrong_tables"
    WRONG_TIME_WINDOW = "wrong_time_window"
    FEDERATION_TURN_CANCELLED = "federation_turn_cancelled"
    TURN_CANCELLED = "cancelled"
    META_ERROR = "meta_error"

    @classmethod
    def parse(cls, raw: FailureCategory | str | None) -> FailureCategory | None:
        """Map a free-form category string to ``FailureCategory``."""
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        v = str(raw).strip()
        if not v:
            return None
        try:
            return cls(v)
        except ValueError:
            return cls.OTHER


_FAILURE_CATEGORY_MEMBER_ORDER: tuple[FailureCategory, ...] = tuple(FailureCategory)


class DiagnosticSeverity(StrEnum):
    """Closed severity set for session diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @classmethod
    def coerce(cls, level: DiagnosticSeverity | str) -> DiagnosticSeverity:
        """Normalize a severity value to a catalogue member."""
        if isinstance(level, cls):
            return level
        normalized = str(level).strip().lower()
        if normalized in {"warn", "_error"}:
            raise ValueError(f"invalid diagnostic severity: {level!r}")
        if normalized == "advisory":
            return cls.INFO
        if normalized == "review":
            return cls.WARNING
        if normalized in {"blocking", "fatal"}:
            return cls.ERROR
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"invalid diagnostic severity: {level!r}") from exc


class SqlDiagnosticCode(StrEnum):
    """Structured codes emitted by AST validation and EXPLAIN-plan diagnostics across registered dialects."""

    AST_PARSE_FAILED = "ast_parse_failed"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    SUBQUERY_NOT_ALLOWED = "subquery_not_allowed"
    LATERAL_NOT_ALLOWED = "lateral_not_allowed"
    CROSS_JOIN_NOT_ALLOWED = "cross_join_not_allowed"
    USING_NOT_ALLOWED = "using_not_allowed"
    SELF_JOIN_NOT_ALLOWED = "self_join_not_allowed"
    EXISTS_NOT_ALLOWED = "exists_not_allowed"
    FORBIDDEN_STRUCTURE = "forbidden_structure"
    NO_ROOT = "no_root"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    UNKNOWN_CTE = "unknown_cte"
    CTE_UNREFERENCED = "cte_unreferenced"
    PARAM_UNBOUND = "param_unbound"
    PARAM_UNDECLARED = "param_undeclared"
    NON_GROUPED_SELECT_COL = "non_grouped_select_col"
    AGG_IN_WHERE = "agg_in_where"
    HAVING_WITHOUT_GROUP = "having_without_group"
    EXPLAIN_CARTESIAN_JOIN = "explain_cartesian_join"
    EXPLAIN_ZERO_ESTIMATE = "explain_zero_estimate"
    EXPLAIN_SEQ_SCAN_INDEXED = "explain_seq_scan_indexed"
    EXPLAIN_SORT_SPILL = "explain_sort_spill"
    EXPLAIN_TEMPORARY_TABLE = "explain_temporary_table"
    EXPLAIN_OTHER = "explain_other"
    EXPLAIN_COST_EXCEEDED = "explain_cost_exceeded"


class DiagnosticCode(StrEnum):
    """Unified catalogue of session and SQL diagnostic codes."""

    DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL = "ARTIFACTS_DIR_NOT_LOCAL"
    DIAGNOSTIC_CODE_ARTIFACT_GROWTH = "ARTIFACT_GROWTH"
    DIAGNOSTIC_CODE_ARTIFACT_LIMIT_NEAR = "ARTIFACT_LIMIT_NEAR"
    DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED = "CANCEL_NOT_SUPPORTED"
    DIAGNOSTIC_CODE_COLUMN_CHARSET_MISMATCH = "COLUMN_CHARSET_MISMATCH"
    DIAGNOSTIC_CODE_COLUMN_PROFILE_FAILED = "COLUMN_PROFILE_FAILED"
    DIAGNOSTIC_CODE_COMPARISON_JOIN_DETOUR = "COMPARISON_JOIN_DETOUR"
    DIAGNOSTIC_CODE_COMPOSE_REPAIR = "COMPOSE_REPAIR"
    DIAGNOSTIC_CODE_COMPOSITE_DESCRIPTIVE_PROFILE_FAILED = "COMPOSITE_DESCRIPTIVE_PROFILE_FAILED"
    DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED = "CONFIGURATION_KEY_IGNORED"
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED = "CONFIG_FILE_VALUE_APPLIED"
    DIAGNOSTIC_CODE_COORDINATOR_LIMITS = "COORDINATOR_LIMITS"
    DIAGNOSTIC_CODE_DATA_QUALITY_ADVISORY = "DATA_QUALITY_ADVISORY"
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED = "DATA_QUALITY_AUTO_CORRECTED"
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ = "DATA_QUALITY_AUTO_READ"
    DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING = "DATA_QUALITY_BLOCKING"
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED = "DESCRIPTION_ENRICHMENT_FAILED"
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP = "DESCRIPTION_ENRICHMENT_NOOP"
    DIAGNOSTIC_CODE_DESCRIPTION_PROMPT_TRUNCATED = "DESCRIPTION_PROMPT_TRUNCATED"
    DIAGNOSTIC_CODE_ENGINE_INFO = "ENGINE_INFO"
    DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED = "ENUM_PROMPT_TRUNCATED"
    DIAGNOSTIC_CODE_FALLBACK_FRESH_RESTART = "FALLBACK_FRESH_RESTART"
    DIAGNOSTIC_CODE_FEDERATION_CAP_EXCEEDED = "FEDERATION_CAP_EXCEEDED"
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK = "FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK"
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_DECIMAL_FALLBACK = "FEDERATION_COORDINATOR_DECIMAL_FALLBACK"
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_EXECUTED = "FEDERATION_COORDINATOR_EXECUTED"
    DIAGNOSTIC_CODE_FEDERATION_INELIGIBLE = "FEDERATION_INELIGIBLE"
    DIAGNOSTIC_CODE_FEDERATION_JOIN_CANDIDATE_CAP = "FEDERATION_JOIN_CANDIDATE_CAP"
    DIAGNOSTIC_CODE_FEDERATION_JOIN_FAN_OUT = "FEDERATION_JOIN_FAN_OUT"
    DIAGNOSTIC_CODE_FEDERATION_MALFORMED_MEMBER_ANSWER = "FEDERATION_MALFORMED_MEMBER_ANSWER"
    DIAGNOSTIC_CODE_FEDERATION_MAPPING_DRIFT = "FEDERATION_MAPPING_DRIFT"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_EXECUTED = "FEDERATION_MEMBER_EXECUTED"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED = "FEDERATION_MEMBER_FAILED"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED = "FEDERATION_MEMBER_PROBE_FAILED"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_GENERATED = "FEDERATION_MEMBER_GENERATED"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_REMOVED = "FEDERATION_MEMBER_REMOVED"
    DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH = "FEDERATION_MEMBER_TIMEZONE_MISMATCH"
    DIAGNOSTIC_CODE_FEDERATION_PARTIAL_FAILURE = "FEDERATION_PARTIAL_FAILURE"
    DIAGNOSTIC_CODE_FEDERATION_PLAN_REPLAY = "FEDERATION_PLAN_REPLAY"
    DIAGNOSTIC_CODE_FEDERATION_POOL_UNDERSIZED = "FEDERATION_POOL_UNDERSIZED"
    DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS = "FEDERATION_REDUCTION_NULL_KEYS"
    DIAGNOSTIC_CODE_FEDERATION_SEMIJOIN_SKIPPED = "FEDERATION_SEMIJOIN_SKIPPED"
    DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED = "FEDERATION_SOURCES_QUERIED"
    DIAGNOSTIC_CODE_FEDERATION_TIMESTAMP_NORMALISED = "FEDERATION_TIMESTAMP_NORMALISED"
    DIAGNOSTIC_CODE_FEDERATION_TIME_ANCHOR = "FEDERATION_TIME_ANCHOR"
    DIAGNOSTIC_CODE_FEDERATION_TURN_CANCELLED = "FEDERATION_TURN_CANCELLED"
    DIAGNOSTIC_CODE_INTERPRET_GROUND_RETRY = "INTERPRET_GROUND_RETRY"
    DIAGNOSTIC_CODE_JOIN_CANDIDATE_CAP = "JOIN_CANDIDATE_CAP"
    DIAGNOSTIC_CODE_JOIN_NULLABLE_KEY = "JOIN_NULLABLE_KEY"
    DIAGNOSTIC_CODE_JOIN_ORPHAN_RATE_HIGH = "JOIN_ORPHAN_RATE_HIGH"
    DIAGNOSTIC_CODE_JOIN_PATH_TIE_CEILING_EXCEEDED = "JOIN_PATH_TIE_CEILING_EXCEEDED"
    DIAGNOSTIC_CODE_LARGE_RESULT_WARNING = "LARGE_RESULT_WARNING"
    DIAGNOSTIC_CODE_LLM_TURN_COST = "LLM_TURN_COST"
    DIAGNOSTIC_CODE_MATERIALIZED_VIEW_ANSWER = "MATERIALIZED_VIEW_ANSWER"
    DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED = "MEMBER_LIMIT_NARROWED"
    DIAGNOSTIC_CODE_MIGRATION_CHECKPOINT_ORPHANED = "MIGRATION_CHECKPOINT_ORPHANED"
    DIAGNOSTIC_CODE_OVERRIDE_NEEDS_RECONFIRMATION = "OVERRIDE_NEEDS_RECONFIRMATION"
    DIAGNOSTIC_CODE_PK_INFERENCE_PROMPT = "PK_INFERENCE_PROMPT"
    DIAGNOSTIC_CODE_PROFILE_TABLE_CLONE_FAILED = "PROFILE_TABLE_CLONE_FAILED"
    DIAGNOSTIC_CODE_REDUNDANT_JOIN_WHERE_DROPPED = "REDUNDANT_JOIN_WHERE_DROPPED"
    DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_CAP_REACHED = "REDUNDANT_KEY_JOIN_CAP_REACHED"
    DIAGNOSTIC_CODE_REDUNDANT_KEY_JOIN_ELIMINATED = "REDUNDANT_KEY_JOIN_ELIMINATED"
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT = "REFUSAL_AGGREGATE_FAN_OUT"
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL = "REFUSAL_AMBIGUOUS_DATE_LITERAL"
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP = "REFUSAL_CAPABILITY_GAP"
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP = "REFUSAL_CTE_CAP"
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING = "REFUSAL_HOP_CEILING"
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE = "REFUSAL_JOIN_PATH_UNAVAILABLE"
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST = "REFUSAL_NULL_IN_NEGATED_LIST"
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR = "REFUSAL_OPAQUE_EXPR"
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN = "REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN"
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING = "REFUSAL_UNION_COLUMN_MISSING"
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE = "REFUSAL_UNSUPPORTED_COLUMN_TYPE"
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT = "REFUSAL_NOT_AVAILABLE_IN_CONTEXT"
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED = "REFUSAL_PERMISSION_DENIED"
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION = "REFUSAL_SCOPE_VIOLATION"
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION = "REFUSAL_INVALID_QUESTION"
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE = "REFUSAL_PARSE_FAILURE"
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA = "REFUSAL_DECLINED_SCHEMA"
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP = "REFUSAL_JOIN_PATH_TIE_CAP"
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET = "REFUSAL_CLAUSE_WIDENED_ROWSET"
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT = "REFUSAL_PROBE_CTE_PLACEMENT"
    DIAGNOSTIC_CODE_REUSE_HIT = "REUSE_HIT"
    DIAGNOSTIC_CODE_REUSE_MISS = "REUSE_MISS"
    DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED = "ROUNDING_MODE_MIXED"
    DIAGNOSTIC_CODE_SCHEMA_FK_CATALOG_ABSENT = "SCHEMA_FK_CATALOG_ABSENT"
    DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED = "SCHEMA_ROLE_TYPE_COERCED"
    DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE = "SCHEMA_UNKNOWN_TYPE_UNUSABLE"
    DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED = "UPLOAD_UNIT_AFFIX_STRIPPED"
    DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED = "UPLOAD_TRANSFORM_REJECTED"
    DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_APPLIED = "UPLOAD_TRANSFORM_APPLIED"
    DIAGNOSTIC_CODE_SCHEMA_OVERRIDE_SKIP = "SCHEMA_OVERRIDE_SKIP"
    DIAGNOSTIC_CODE_SEMANTIC_PROFILE_WHERE_EDGE = "SEMANTIC_PROFILE_WHERE_EDGE"
    DIAGNOSTIC_CODE_SENSITIVITY_GATE_HIT = "SENSITIVITY_GATE_HIT"
    DIAGNOSTIC_CODE_SQL_PARSE_FAILED = "SQL_PARSE_FAILED"
    DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK = "STALE_ARTIFACT_LOCK"
    DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED = "TEMPLATE_REMAP_DIVERGED"
    DIAGNOSTIC_CODE_TEMPLATE_STORE_ORPHANED = "TEMPLATE_STORE_ORPHANED"
    DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT = "WRITE_QUEUE_CORRUPT"
    DIAGNOSTIC_CODE_WRITE_QUEUE_FULL = "WRITE_QUEUE_FULL"
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_AUTO_FIXED = "ZERO_ROW_WHERE_AUTO_FIXED"
    DIAGNOSTIC_CODE_ZERO_ROW_WHERE_SUGGESTION = "ZERO_ROW_WHERE_SUGGESTION"
    AST_PARSE_FAILED = "ast_parse_failed"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    SUBQUERY_NOT_ALLOWED = "subquery_not_allowed"
    LATERAL_NOT_ALLOWED = "lateral_not_allowed"
    CROSS_JOIN_NOT_ALLOWED = "cross_join_not_allowed"
    USING_NOT_ALLOWED = "using_not_allowed"
    SELF_JOIN_NOT_ALLOWED = "self_join_not_allowed"
    EXISTS_NOT_ALLOWED = "exists_not_allowed"
    FORBIDDEN_STRUCTURE = "forbidden_structure"
    NO_ROOT = "no_root"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    UNKNOWN_CTE = "unknown_cte"
    CTE_UNREFERENCED = "cte_unreferenced"
    PARAM_UNBOUND = "param_unbound"
    PARAM_UNDECLARED = "param_undeclared"
    NON_GROUPED_SELECT_COL = "non_grouped_select_col"
    AGG_IN_WHERE = "agg_in_where"
    HAVING_WITHOUT_GROUP = "having_without_group"
    EXPLAIN_CARTESIAN_JOIN = "explain_cartesian_join"
    EXPLAIN_ZERO_ESTIMATE = "explain_zero_estimate"
    EXPLAIN_SEQ_SCAN_INDEXED = "explain_seq_scan_indexed"
    EXPLAIN_SORT_SPILL = "explain_sort_spill"
    EXPLAIN_TEMPORARY_TABLE = "explain_temporary_table"
    EXPLAIN_OTHER = "explain_other"
    EXPLAIN_COST_EXCEEDED = "explain_cost_exceeded"


class UploadColumnTransformId(StrEnum):
    """Closed vocabulary for upload column transform proposals."""

    PARSE_TEMPORAL = "parse_temporal"
    STRIP_NUMERIC_AFFIX = "strip_numeric_affix"
    BAND_BOUNDS = "band_bounds"
    BAND_VALUE_MAP = "band_value_map"
    KEEP_CANONICAL_COLUMNS = "keep_canonical_columns"
    DERIVE_BY_PATTERN = "derive_by_pattern"
    DROP_EMPTY_COLUMNS = "drop_empty_columns"
    NULL_TOKENS = "null_tokens"
    UNPIVOT_COLUMNS = "unpivot_columns"


@dataclass(frozen=True)
class SqlDiagnostic:
    """Single structured finding from AST validation or EXPLAIN-plan analysis."""

    code: SqlDiagnosticCode
    message: str
    node_kind: str | None = None
    offending_identifier: str | None = None
    details: Mapping[str, str] = field(default_factory=dict)


class AetherError(Exception):
    """Root exception for all library failures. Catch this type for a single handler over every ``aetherdialect`` error."""


class ConfigError(AetherError, ValueError):
    """Raised when environment variables or static configuration are missing or contradictory."""

    data_quality_report: DataQualityReport | None = None


class SchemaNaming:
    """Normalize schema identifiers and scope column specs."""

    @staticmethod
    def norm_schema_identifier(name: str, *, what: str) -> str:
        """Lowercase and strip *name*; raise when empty after strip."""
        s = str(name).strip().lower()
        if not s:
            raise ValueError(f"{what} must be non-empty")
        return s

    @staticmethod
    def normalize_scope_column_spec(spec: str, *, field: str) -> str:
        """Normalize a ``table.column`` or ``source.table.column`` scope column spec."""
        raw = str(spec).strip()
        parts = raw.split(".")
        if len(parts) == 2:
            tbl_raw, col_raw = parts
        elif len(parts) == 3:
            _source_raw, tbl_raw, col_raw = parts
        else:
            raise ConfigError(CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC.format(spec=spec))
        tbl = SchemaNaming.norm_schema_identifier(tbl_raw, what=f"{field} table")
        col = SchemaNaming.norm_schema_identifier(col_raw, what=f"{field} column")
        if tbl == "*":
            return f"*.{col}"
        return f"{tbl}.{col}"


class ColumnTypeSemantics:
    """SQL column type normalization and semantic classification."""

    @staticmethod
    def normalize_column_type(col_type: str) -> str:
        """Lowercase a SQL type and remove `(n)` / `(n,m)` parameter lists."""
        normalized, _precision, _scale = ColumnTypeSemantics.split_column_type(col_type)
        return normalized

    @staticmethod
    def split_column_type(col_type: str) -> tuple[str, int | None, int | None]:
        """Lowercase a SQL type, strip parameter lists for lookup, and return parsed precision and scale."""
        normalized = col_type.lower().strip()
        match = NUMERIC_TYPE_ARGUMENTS_RE.search(normalized)
        precision = int(match.group(1)) if match else None
        scale = int(match.group(2)) if match and match.group(2) is not None else None
        normalized = re.sub(r"\(\d+(?:,\s*\d+)?\)", "", normalized)
        normalized = normalized.strip()
        return normalized, precision, scale

    @staticmethod
    def structural_data_type_key(data_type: str) -> str:
        """Return a canonical catalog-type token for structural diff and hashing."""
        normalized = ColumnTypeSemantics.normalize_column_type(data_type)
        if not normalized:
            return normalized
        return STRUCTURAL_DATA_TYPE_CANONICAL.get(normalized, normalized)

    @staticmethod
    def column_is_unsigned_from_data_type(
        data_type: str,
        *,
        reflected_unsigned: bool | None = None,
    ) -> bool:
        """Return whether a SQL column type carries unsigned integer semantics."""
        if reflected_unsigned is not None:
            return bool(reflected_unsigned)
        return "unsigned" in str(data_type or "").lower()

    @staticmethod
    def column_is_fixed_width_text_from_data_type(data_type: str) -> bool:
        """Return whether a SQL column type is fixed-width character text (CHAR/NCHAR)."""
        if not str(data_type or "").strip():
            return False
        base = ColumnTypeSemantics.normalize_column_type(data_type)
        return base in FIXED_WIDTH_TEXT_BASE_TYPES

    @staticmethod
    def column_timezone_aware_from_data_type(
        data_type: str,
        *,
        engine: str | None = None,
    ) -> bool:
        """Return whether a SQL temporal type preserves timezone offsets."""
        raw = str(data_type or "").strip().lower()
        if not raw:
            return False
        base = raw.split("(", 1)[0].strip()
        if base in FEDERATION_TIMEZONE_AWARE_DATA_TYPES:
            return True
        if "with time zone" in raw:
            return True
        if base == "timestamp":
            engine_norm = str(engine or "").strip().lower()
            return engine_norm in MYSQL_TIMESTAMP_ENGINES
        return False

    @staticmethod
    def column_unsigned_near_type_max(meta: Any) -> bool:
        """Return whether profiled maxima exceed float-safe range or approach an unsigned ceiling."""
        if not getattr(meta, "is_unsigned", False) or getattr(meta, "max_val", None) in (None, ""):
            return False
        try:
            profiled_max = int(str(meta.max_val).strip())
        except (TypeError, ValueError):
            return False
        if profiled_max > MAX_FLOAT_SAFE_INTEGER:
            return True
        base = ColumnTypeSemantics.normalize_column_type(meta.data_type)
        type_max = UNSIGNED_INTEGER_TYPE_MAX.get(base)
        if type_max is None:
            return False
        margin = max(1, type_max // 100)
        return profiled_max >= type_max - margin

    @staticmethod
    def is_numeric_type(data_type: str) -> bool:
        """Return whether a SQL data type string looks numeric."""
        dt = data_type.lower()
        return any(t in dt for t in NUMERIC_TYPE_TOKENS)

    @staticmethod
    def is_string_type(data_type: str) -> bool:
        """Return whether a SQL data type string looks string-like."""
        dt = data_type.lower()
        return any(t in dt for t in STRING_TYPE_TOKENS)

    @staticmethod
    def is_date_type(data_type: str) -> bool:
        """Return whether a SQL data type string looks date- or time- like."""
        dt = data_type.lower()
        return any(t in dt for t in DATE_TYPE_TOKENS)

    @staticmethod
    def data_type_to_value_type(data_type: str) -> str:
        """Map a SQL data type string to a prompt/value-type token."""
        normalized = ColumnTypeSemantics.normalize_column_type(data_type)
        vt = COLUMN_TYPE_TO_VALUE_TYPE.get(normalized)
        if vt:
            return vt
        if ColumnTypeSemantics.is_numeric_type(data_type):
            return "number"
        if ColumnTypeSemantics.is_date_type(data_type):
            return "date"
        if ColumnTypeSemantics.is_string_type(data_type):
            return "string"
        return ArrayStorageKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class LlmExecutionConfig:
    """Merged Azure OpenAI credentials plus execution cost and timeout limits for the engine runtime. Public operators configure two deployment slots named ``LIGHT`` and ``HEAVY`` that provision Azure deployments sized for the ``gpt-5-mini`` and ``gpt-5.4-mini`` model classes respectively. Internal routing from logical model identifiers to these slots is not part of the public stability contract."""

    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    deployment_light: str
    deployment_heavy: str
    max_query_cost_rows: int
    max_query_cost_bytes: int
    statement_timeout_ms: int
    llm_timeout_ms: int
    profile_timeout_ms: int
    explain_timeout_ms: int | None


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    """Bound engine type and runtime config for one engine instance or federated source."""

    engine_type: str
    runtime_config: Any


class InteractiveChoicePort(Protocol):
    """Bridges yes/no prompts to a session queue or stdin."""

    _pending_federation_plan_template: Any

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        ...

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Return a normalised choice or raise ``PipelineSuspended`` when the queue is empty."""
        ...


class QueryLogSource(Protocol):
    """Read-only fetcher of historical SQL statements from an engine query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when the source can run against *conn*."""
        ...

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Return distinct SQL texts newest-first within policy caps."""
        ...


class ResultBackendPort(Protocol):
    """Typing port for dialect row-fetch backends. Concrete ABC lives in ``_dialect_sqlglot_helper``."""

    kind: ResultReaderKind

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute *sql* and return result rows as tuples."""
        ...

    def fetch_arrow_table(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> Any:
        """Execute *sql* and return a PyArrow table when the driver supports it."""
        ...

    def cancel_statement(self) -> None:
        """Cancel an in-flight statement when the driver supports it."""
        ...

    def fetch_first_column_text(self, sql: str, params: dict[str, Any] | None = None) -> str:
        """Execute *sql* and join the first column of each row into newline-separated text."""
        ...


class RephraseHint(Enum):
    """User-facing rephrase hint categories printed when the pipeline cannot continue."""

    INTENT_PARSE_FAILED = "intent_parse_failed"
    SCHEMA_INVALID_DECLINED = "schema_invalid_declined"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    JOIN_PATH_UNAVAILABLE = "join_path_unavailable"
    USER_REJECTED_INTENT = "user_rejected_intent"
    USER_REJECTED_RESULT = "user_rejected_result"
    RESTRICTED_QUESTION = "restricted_question"
    VAGUE_QUESTION = "vague_question"
    FEDERATION_INELIGIBLE = "federation_ineligible"
    FEDERATION_PARTIAL_FAILURE = "federation_partial_failure"
    FEDERATION_TURN_CANCELLED = "federation_turn_cancelled"


@dataclass(frozen=True)
class RefusalCatalogueEntry:
    """User-facing refusal text and reformulation hint for one catalogue code."""

    user_text: str
    reformulation_hint: str


class RefusalCondition(StrEnum):
    """Enumerated refusal conditions mapped to stable diagnostic codes."""

    PERMISSION_DENIAL = "permission_denial"
    SCOPE_VIOLATION = "scope_violation"
    INVALID_QUESTION = "invalid_question"
    PARSE_FAILURE = "parse_failure"
    DECLINED_SCHEMA = "declined_schema"
    TIE_CAP_EXHAUSTION = "tie_cap_exhaustion"
    WIDENED_CLAUSE_REFUSAL = "widened_clause_refusal"
    PROBE_PLACEMENT = "probe_placement"
    UNSUPPORTED_COLUMN_TYPE = "unsupported_column_type"
    NULL_IN_NEGATED_LIST = "null_in_negated_list"
    AMBIGUOUS_DATE_LITERAL = "ambiguous_date_literal"
    UNION_COLUMN_MISSING = "union_column_missing"
    JOIN_PATH_UNAVAILABLE = "join_path_unavailable"
    AGGREGATE_FAN_OUT = "aggregate_fan_out"
    HOP_CEILING = "hop_ceiling"
    CTE_CAP = "cte_cap"
    CAPABILITY_GAP = "capability_gap"
    NOT_AVAILABLE_IN_CONTEXT = "not_available_in_context"
    SUBDAY_DATE_WINDOW = "subday_date_window"


_mock_fixture_recorded_corpus_count: Any = None


class MockFixtureMissingError(AetherError, RuntimeError):
    """Raised when the mock provider has no fixture for the requested LLM call."""

    @classmethod
    def register_recorded_corpus_count(cls, fn: Any) -> None:
        """Register sandbox recorded-corpus count provider for error messages."""
        global _mock_fixture_recorded_corpus_count
        _mock_fixture_recorded_corpus_count = fn

    def __init__(self, *, task: str, system: str, user: str) -> None:
        self.task = task
        self.system = system
        self.user = user
        recorded_count = (
            _mock_fixture_recorded_corpus_count() if callable(_mock_fixture_recorded_corpus_count) else None
        )
        if recorded_count is not None:
            super().__init__(
                "Sandbox is in recorded-corpus mode "
                f"({recorded_count} questions available). "
                "Ask a recorded question or call Sandbox.sandbox_questions(). "
                f"No mock fixture for task={task!r}.",
            )
            return
        import json

        skeleton = json.dumps(
            {
                "task": task,
                "system": system[:200] + ("..." if len(system) > 200 else ""),
                "user": user[:500] + ("..." if len(user) > 500 else ""),
                "output_text": "<paste model JSON/text here>",
            },
            ensure_ascii=False,
            indent=2,
        )
        super().__init__(
            f"No mock fixture for task={task!r}. Add an entry to the fixture corpus:\n{skeleton}",
        )


class OwnerOnlyOperationError(ConfigError):
    """Raised when a consumer-role instance attempts a schema-identity mutation."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"Operation {operation!r} requires role=SchemaRole.OWNER; this instance is a consumer.")


class MigrationPendingError(AetherError, ValueError):
    """Init terminated because schema_migration_map.json is required, malformed, missing after export, or conflicts with validation."""


class DatabaseConnectionError(AetherError, OSError):
    """Raised when the database driver rejects a connection attempt."""


class RetryableError(AetherError):
    """Marker base class for transient failures that may succeed on retry. Concrete subclasses combine this marker with :class:`DatabaseConnectionError`, :class:`RuntimeError`, etc. Integrators may use ``isinstance(exc, RetryableError)`` without inspecting messages."""


class DatabasePingFailed(DatabaseConnectionError, RetryableError):
    """Raised when a trivial ``SELECT 1`` ping fails after retries (network blips, overload)."""


class LlmTransientFailure(RuntimeError, RetryableError):
    """LLM request failed after retries due to rate limits, timeouts, or connection resets."""


class StatementTimeoutError(RuntimeError, RetryableError):
    """Raised when the database aborts work due to ``statement_timeout`` or warehouse timeouts."""


class DatabaseErrorClassification(StrEnum):
    """Driver failure permanence classification attached to :class:`DatabaseExecutionError`."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class DatabaseExecutionError(AetherError):
    """Raised when a database driver fails during statement execution after connect succeeds."""

    def __init__(
        self,
        message: str,
        *,
        driver_class: str | None = None,
        classification: str | DatabaseErrorClassification | None = None,
        retryable: bool = False,
        driver_detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.driver_class = driver_class
        self.classification = (
            classification.value if isinstance(classification, DatabaseErrorClassification) else classification
        )
        self.retryable = bool(retryable)
        self.driver_detail = dict(driver_detail) if driver_detail is not None else None
        super().__init__(message)


class RetryableDatabaseExecutionError(DatabaseExecutionError, RetryableError):
    """Transient database execution failure that may succeed on retry."""

    def __init__(
        self,
        message: str,
        *,
        driver_class: str | None = None,
        classification: str | DatabaseErrorClassification | None = None,
        driver_detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            driver_class=driver_class,
            classification=classification,
            retryable=True,
            driver_detail=driver_detail,
        )


class SchemaAccessError(AetherError, ValueError):
    """Raised when credentials cannot read the requested scope or the graph is unusable. Covers catalog reflection, construction invariants, and scope configuration at init. Runtime ``EXPLAIN`` / ``execute`` permission denial is raised as :class:`AccessError`, which is a subclass so integrators may catch this type for every access denial surface."""


class SessionActiveError(AetherError, RuntimeError):
    """Raised when ``PipelineSession.ask`` is called while a turn is already in progress."""


class SuspendedSessionExpiredError(SessionActiveError):
    """Raised when a suspended turn exceeds ``EngineLimits.suspended_session_ttl_seconds``."""


class SessionTurnCancelledError(AetherError):
    """Raised when a programmatic session turn is cancelled cooperatively."""


class SchemaInvariantError(AetherError, RuntimeError):
    """Raised by :func:`assert_schema_invariants` when the canonical containers fall out of sync. Indicates a programmer error elsewhere in the build pipeline: e.g. an FK referencing a missing column, a PK column missing from its table, an unwired column-table back reference, or a stale canonical-bearer index. Always indicates the offending source-of-truth has been violated and never represents a recoverable runtime condition."""


@dataclass(frozen=True, slots=True)
class ResolvedQualifiedRef:
    """One resolved ``source.table.column`` or ``table.column`` reference."""

    source_id: str
    table: str
    column: str
    qualified: str


@dataclass(frozen=True, slots=True)
class FederationTopologyReport:
    """Outcome of reconciling recorded federation members against the manifest."""

    change: FederationTopologyChange
    added_source_ids: tuple[str, ...] = ()
    removed_source_ids: tuple[str, ...] = ()
    plan_templates_invalidated: bool = False


@dataclass(frozen=True, slots=True)
class PersistedFederationInspection:
    """Declaration and roster loaded from a persisted ``fed_<id>`` artifact tree."""

    federation_id: str
    federation_dir: str
    manifest: FederationManifest
    mappings: FederationMappings
    roster: tuple[tuple[str, str, str, str], ...]


class FederationConfigError(ConfigError):
    """Raised when a federation manifest or mapping sidecar is invalid."""


class FederationDeclarationError(FederationConfigError):
    """Raised when federation declarations fail validation at build or registration time."""


class FederationMappingsAppliedSidecarError(FederationConfigError):
    """Raised when the applied federation mappings sidecar disagrees with the mappings file."""


class FederationIneligibleError(ConfigError):
    """Raised when a validated intent cannot be decomposed or executed as a federated plan."""


class FederationRuntimeError(ConfigError):
    """Raised when federated execution fails after planning succeeded."""


class FederationMemberExecutionError(FederationRuntimeError):
    """Raised when one federation member's query fails during execution. Carries ``source_id`` (the failing member) and ``phase`` (execution stage, typically ``"member"``). Catch this specifically to attribute the failure and decide retry versus surface; the turn may be retryable when the cause is a :class:`RetryableError`. Prefer this over catching the parent and inspecting attributes to distinguish member failure from a resource-cap breach."""

    def __init__(self, message: str, *, source_id: str, phase: str) -> None:
        self.source_id = str(source_id or "")
        self.phase = str(phase or "member")
        super().__init__(message)


class FederationPartialFailureError(FederationRuntimeError):
    """Terminal outcome when one member fails after others succeeded."""

    def __new__(
        cls,
        message: str,
        *,
        source_id: str,
        phase: str,
        succeeded: tuple[tuple[str, int, str], ...] = (),
        retryable: bool = False,
    ) -> FederationPartialFailureError:
        if retryable:
            return FederationRuntimeError.__new__(RetryableFederationPartialFailureError)
        return FederationRuntimeError.__new__(cls)

    def __init__(
        self,
        message: str,
        *,
        source_id: str,
        phase: str,
        succeeded: tuple[tuple[str, int, str], ...] = (),
        retryable: bool = False,
    ) -> None:
        self.source_id = str(source_id or "")
        self.phase = str(phase or "member")
        self.succeeded = tuple(succeeded)
        self.retryable = bool(retryable)
        super().__init__(message)


class RetryableFederationPartialFailureError(FederationPartialFailureError, RetryableError):
    """Partial federation failure that may succeed on retry."""


class FederationTurnCancelledError(FederationRuntimeError):
    """Terminal outcome when a federated turn is cancelled cooperatively."""

    def __init__(
        self,
        message: str,
        *,
        source_id: str,
        phase: str,
        succeeded: tuple[tuple[str, int, str], ...] = (),
    ) -> None:
        self.source_id = str(source_id or "")
        self.phase = str(phase or "member")
        self.succeeded = tuple(succeeded)
        super().__init__(message)


class FederationCapExceededError(FederationRuntimeError):
    """Raised when a federated row, byte, or timeout cap is exceeded. Carries ``limit_key`` (which cap fired, e.g. ``"row_cap"``) and optional ``source_id`` (empty string for coordinator-wide caps). Catch this specifically to prompt the caller to narrow or re-scope the question; retrying the same turn will not help. Prefer this over catching the parent and inspecting attributes to distinguish a cap breach from a per-member execution failure."""

    def __init__(self, message: str, *, limit_key: str, source_id: str = "") -> None:
        self.limit_key = str(limit_key or "")
        self.source_id = str(source_id or "")
        super().__init__(message)


class ResultCapExceededError(AetherError, RuntimeError):
    """Raised when a single-engine row or byte cap is exceeded during result fetch."""

    def __init__(self, message: str, *, limit_key: str, source_id: str = "") -> None:
        self.limit_key = str(limit_key or "")
        self.source_id = str(source_id or "")
        super().__init__(message)


class FederationMemberProbeError(FederationRuntimeError):
    """Raised when a federation member database probe fails during initialization."""

    def __init__(self, message: str, *, source_id: str) -> None:
        self.source_id = str(source_id or "")
        super().__init__(message)


class FederationMalformedMemberAnswerError(FederationMemberExecutionError):
    """Raised when a member result's projection does not match the prepared sub-intent."""

    rejection_bucket: str = "MALFORMED_MEMBER_ANSWER"

    def __init__(self, message: str, *, source_id: str, phase: str = "member") -> None:
        super().__init__(message, source_id=source_id, phase=phase)


class FederationJoinFanOutError(FederationRuntimeError):
    """Raised when a coordinator join multiplies rows beyond the declared key grain."""

    rejection_bucket: str = "JOIN_FAN_OUT"

    def __init__(self, message: str, *, source_id: str, phase: str = "coordinator") -> None:
        self.source_id = str(source_id or "")
        self.phase = str(phase or "coordinator")
        super().__init__(message)


class FederationInvariantError(ConfigError):
    """Raised when federation composition or replay invariants are violated."""


class FederationMemberUnprofilableError(FederationDeclarationError):
    """Raised when a federation member schema graph was not successfully profiled."""

    def __init__(self, message: str, *, source_id: str) -> None:
        self.source_id = str(source_id or "")
        super().__init__(message)


class MigrationTier(StrEnum):
    """Classified migration severity between a stored artifact fingerprint and the live graph."""

    NO_CHANGE = "no_change"
    PERMISSION_FILTERED = "permission_filtered"
    SOFT_REFRESH = "soft_refresh"
    ADDITIVE = "additive"
    REMAP = "remap"
    DESTRUCTIVE = "destructive"


class ColumnVisibilityBlockReason(StrEnum):
    """Machine-stable reason a column is blocked from LLM exposure or reference validation."""

    DENIED = "denied"
    NOT_IN_ALLOW_COLUMNS = "not_in_allow_columns"
    SENSITIVE_HIDDEN = "sensitive_hidden"
    SENSITIVE_RESTRICTED = "sensitive_restricted"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """Outcome of :meth:`~aetherdialect.AetherEngine.refresh` or :meth:`~aetherdialect.AetherFederation.refresh`."""

    migration_tier: MigrationTier
    schema_changed: bool
    objects_added: tuple[str, ...]
    objects_removed: tuple[str, ...]
    templates_invalidated: int
    orphans_removed: int
    bytes_reclaimed: int
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """Summary of actions taken when reconciling templates against a new schema fingerprint."""

    tier: MigrationTier
    renamed_tables: tuple[tuple[str, str], ...] = ()
    renamed_columns: tuple[tuple[str, str, str], ...] = ()
    refreshed_descriptions: tuple[str, ...] = ()
    destroyed_templates: int = 0
    remapped_templates: int = 0
    surgically_invalidated: int = 0
    dropped_tables: tuple[str, ...] = ()
    added_tables: tuple[str, ...] = ()
    added_columns: tuple[tuple[str, str], ...] = ()
    value_type_changed_columns: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SchemaMigrationMapEntry:
    """One renamed entity or column reference inside the schema migration map."""

    entry_type: str
    table: str = ""
    from_name: str = ""
    to_name: str = ""


@dataclass(frozen=True, slots=True)
class SchemaMigrationMap:
    """User-authored migration mapping consumed once at init time. When ``refresh_existing_descriptions_on_addition`` is true and the live schema diff includes newly added tables, the engine runs a full-graph LLM classifier pass after subset profiling and merges refreshed table/column descriptions and roles only for tables that were otherwise unchanged by the diff."""

    version: int
    action: str
    table_renames: tuple[SchemaMigrationMapEntry, ...]
    column_renames: tuple[SchemaMigrationMapEntry, ...]
    dropped_tables: tuple[str, ...]
    dropped_columns: tuple[SchemaMigrationMapEntry, ...]
    added_tables: tuple[str, ...]
    added_columns: tuple[SchemaMigrationMapEntry, ...]
    fk_remaps: tuple[SchemaMigrationMapEntry, ...] = ()
    pk_remaps: tuple[SchemaMigrationMapEntry, ...] = ()
    refresh_existing_descriptions_on_addition: bool = False


@dataclass(frozen=True, slots=True)
class OverrideSkip:
    """One JSON entry that was rejected during ``AetherEngine.apply_schema_overrides``."""

    path: str
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True)
class SidecarReconcileReport:
    """Rows removed from the persisted overrides sidecar during graph reconciliation."""

    pruned_paths: tuple[str, ...]
    wrote_disk: bool


@dataclass(frozen=True, slots=True)
class OverrideReport:
    """Summary of edits produced by ``AetherEngine.apply_schema_overrides``."""

    table_edits: int = 0
    column_edits: int = 0
    fks_added: int = 0
    fks_endorsed: int = 0
    fks_removed: int = 0
    pks_added: int = 0
    pks_endorsed: int = 0
    pks_blocked: int = 0
    changed_pk_blocks: bool = False
    changed_fk_blocks: bool = False
    coerced_columns: int = 0
    collapsed_inferences: int = 0
    descriptions_refined: int = 0
    business_knowledge_refined: int = 0
    business_knowledge_entries: tuple[BusinessKnowledgeEntry, ...] | None = None
    skipped: tuple[OverrideSkip, ...] = ()


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Structured status line emitted alongside ``notify`` / ``debug`` for programmatic consumers."""

    stage: str
    level: DiagnosticSeverity | str
    code: str
    message: str
    details: tuple[tuple[str, str], ...] = ()
    duration_ms: int | None = None
    source_id: str | None = None
    phase: str | None = None
    remediation: str | None = None
    subject: str | None = None
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", DiagnosticSeverity.coerce(self.level))


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Structured outcome of CSV/Excel upload validation before engine construction."""

    ok: bool
    issues: tuple[Diagnostic, ...]
    narrative: str
    suggested_selections: dict[str, dict[str, Any]] = field(default_factory=dict)
    confirmed_selections: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def requires_review(self) -> bool:
        """Return True when any issue needs caller confirmation before construction."""
        for issue in self.issues:
            for key, value in issue.details:
                if key == "review" and value == "yes":
                    return True
        return False

    def to_dict(self) -> dict[str, str]:
        """Return a heading-to-detail map for UI consumers."""
        out: dict[str, str] = {}
        for issue in self.issues:
            heading = str(issue.message).strip() or str(issue.code)
            detail_parts: list[str] = []
            for key, value in issue.details:
                detail_parts.append(f"{key}: {value}")
            detail = "; ".join(detail_parts) if detail_parts else heading
            out[heading] = detail
        return out

    def to_json_dict(self) -> dict[str, Any]:
        """Return a machine-readable report with structured issue rows."""
        issues_out: list[dict[str, Any]] = []
        for issue in self.issues:
            details_map = {key: value for key, value in issue.details}
            issue_code = details_map.pop("issue_code", issue.code)
            location = details_map.pop("location", "")
            severity = details_map.pop("severity", "")
            blocking_raw = details_map.pop("blocking", "no")
            details_map.pop("review", None)
            issue_level = issue.level.value if isinstance(issue.level, DiagnosticSeverity) else issue.level
            level = severity or issue_level
            issues_out.append(
                {
                    "code": issue_code,
                    "level": level,
                    "blocking": blocking_raw == "yes",
                    "location": location,
                    "message": issue.message,
                    "details": details_map,
                }
            )
        return {
            "ok": self.ok,
            "narrative": self.narrative,
            "issues": issues_out,
            "suggested_selections": dict(self.suggested_selections),
            "confirmed_selections": dict(self.confirmed_selections),
            "requires_review": self.requires_review,
        }


@dataclass(frozen=True, slots=True)
class IntentInterpretation:
    """Compact Interpret-stage traceability attached to session steps."""

    approach: str
    grounding: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class IntentSummary:
    """Compact projection of a resolved :class:`RuntimeIntent` for UI and telemetry."""

    tables: tuple[str, ...]
    select_cols: tuple[str, ...]
    filters: tuple[str, ...]
    group_by: tuple[str, ...]
    order_by: tuple[str, ...]
    limit: int | None
    natural_language: str


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    """One bind slot on an accepted template for programmatic callers."""

    handle: str
    current_value: ParamValue | None
    display_name: str
    column_expr: str = ""
    upper_handle: str = ""
    unit_handle: str = ""


@dataclass(frozen=True, slots=True)
class MigrationPreview:
    """Human-readable outcome of comparing a skeleton against the live prepared schema."""

    tier: Literal["compatible", "remap", "destructive"]
    affected_tables: tuple[str, ...]
    affected_columns: tuple[tuple[str, str], ...]
    skeleton_path: str


@dataclass(frozen=True, slots=True)
class SessionNotice:
    """Structured bookkeeping notice on a session step, separate from user-facing ``message``."""

    code: str
    level: Literal["info", "warning", "error"]
    message: str


@dataclass(frozen=True, slots=True)
class SessionStep:
    """Single observable point in a programmatic interactive turn. Carries whether the turn has finished, a short instruction string, a stage discriminant, optional SQL, tabular data, a free-form body, and an error string when the engine fails. done: True when the pipeline finished successfully or ended in a terminal error; False when the caller must respond via ``PipelineSession.step``. prompt: The short line the interactive layer should show immediately before collecting input (for example yes or no, or a free-text rejection reason prompt). kind: Stable stage identifier matching the active suspend kind or a terminal sentinel; used to branch programmatic UIs without parsing ``prompt``. sql: The formatted SQL under discussion when the step pertains to execution or confirmation; otherwise None. data: Row-level query preview or full result as a ``pandas.DataFrame``; None for scalar outcomes, previews trimmed to five rows at suspend boundaries, and the full frame on the terminal acceptance step when rows exist. message: Multi-line contextual body: consolidated intent confirmation, migration DDL, rejection guidance, or a rendered scalar value; empty or None when nothing extra should print beyond ``prompt`` and ``data``. error: Terminal failure explanation when ``done`` is True and processing stopped; otherwise None. intent_summary: Structured intent headline when the step reflects a parsed intent or later pipeline stages; otherwise None. diagnostics: Structured diagnostics captured during this step (from ``notify`` / ``debug`` when a collector is active). status: On terminal steps, a coarse outcome name: failure categories use the same string values as :class:`FailureCategory`; cooperative cancellation uses ``cancelled``; None on success or non-terminal steps. reply_shape: When ``done`` is False, whether the caller should collect a yes or no token or free text; None on terminal steps. semantic_warnings: Normalised warning strings for intent confirmation, often empty on non-intent suspend steps."""

    done: bool
    prompt: str | None
    kind: str
    sql: str | dict[str, str] | None = None
    data: pandas.DataFrame | None = None
    message: str | None = None
    error: str | None = None
    intent_summary: IntentSummary | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    status: str | None = None
    reply_shape: Literal["yes_no", "free_text"] | None = None
    semantic_warnings: tuple[str, ...] = ()
    interpretation: IntentInterpretation | None = None
    parameters: tuple[ParameterBinding, ...] = ()
    federated_bundle: Any | None = None
    federation_source_id: str | None = None
    federation_phase: str | None = None
    federation_limit_key: str | None = None
    federation_succeeded: tuple[tuple[str, int, str], ...] = ()
    retryable: bool = False
    notices: tuple[SessionNotice, ...] = ()
    data_truncated: bool = False
    llm_usage: LlmTurnUsageSummary | None = None
    refusal_code: str | None = None
    refusal_diagnostic_code: str | None = None
    template_id: str | None = None
    meta_payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WriteQueueEvent:
    """Structured event a reader-mode session records for a writer to apply later. kind: Discriminator selecting which writer-side handler applies (template accept or reject, paraphrase emission, override proposal materialisation, or question feedback). schema_graph_id: Stable schema-graph identity stamped at enqueue time; the writer matches and drops events when this value no longer matches the live snapshot. schema_hash: Advisory effective structural hash at event creation for audit and debug only. produced_at: ISO-8601 timestamp string when the reader enqueued the event. payload: Ordered key-value pairs serialising handler-specific fields; a tuple of pairs keeps the event hashable and avoids dict key-order ambiguity across processes."""

    kind: Literal[
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    ]
    schema_graph_id: str
    schema_hash: str
    produced_at: str
    payload: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PhaseProgressEvent:
    """Coarse phase transition during engine construction or an ask turn."""

    phase: str
    timestamp_iso: str
    source: str | None = None
    stage: int | str | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Lifecycle audit record for integrator sinks."""

    event_type: str
    timestamp_iso: str
    question: str | None
    schema_hash: str | None
    provider: Literal["openai", "azure", "mock"]
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LlmUsageRecord:
    """One LLM response worth of token usage attributed to a build, question, or run scope."""

    scope: Literal["build", "question", "run"]
    block_id: int
    task: str
    logical_model: str
    api_model: str
    provider: Literal["openai", "azure", "mock"]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cache_write_tokens: int | None
    attempt: int
    elapsed_ms: int
    phase: str = ""
    source_id: str = ""


@dataclass(frozen=True, slots=True)
class LlmTurnUsageSummary:
    """Aggregated LLM token usage for one interactive ask turn."""

    request_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """Frozen redacted configuration text for integrators."""

    text: str

    def format_human(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class SchemaStatsSnapshot:
    """Frozen schema statistics mapping."""

    stats: dict[str, Any]

    def format_human(self) -> str:
        lines = [f"{k}: {v}" for k, v in sorted(self.stats.items())]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SeedWarmupSummarySnapshot:
    """Newest seed-warmup summary text if present."""

    text: str

    def format_human(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class QSimSummarySnapshot:
    """QSim summary lines for a version range."""

    lines: tuple[str, ...]

    def format_human(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True, slots=True)
class StoredTemplateSummary:
    """Caller-visible summary row for one accepted template."""

    id: str
    approval_state: str = "approved"


@dataclass(frozen=True, slots=True)
class StoredTemplateDetail:
    """Full caller-visible detail for one accepted template."""

    summary: StoredTemplateSummary
    parameters: tuple[ParameterBinding, ...]
    approval_state: str = "approved"


@dataclass(frozen=True, slots=True)
class TemplateExecutionResult:
    """Result of executing a stored template with caller-supplied bind values."""

    rows: tuple[tuple[Any, ...], ...]
    sql: str
    display_sql: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TablePreviewResult:
    """Bounded table sample returned through scope and sensitivity gates."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Frozen scope object that narrows what a question may touch on a single :class:`~aetherdialect.AetherEngine` (one database connection). Parallel to :class:`FederationContext` (multi-member federation) and :class:`SpaceContext` (named subset at question time). Unlike those types, this context has no ``name`` field and is the only one that carries an optional ``sql_file``. Allow/deny lists, include mode, and content hashes of ``notes_file`` / ``sql_file`` feed the engine scope fingerprint. Args: allow_objects: Relation names permitted in scope. Empty means unrestricted (catalog reflection still respects ``include``). include: Catalog kinds to reflect when ``allow_objects`` is empty: ``tables`` or ``views`` (default ``tables``). ``both`` is rejected; reflect tables and views in separate passes when both are needed. deny_objects: Relation names excluded from scope. Empty means none denied. Must not overlap ``allow_objects``. deny_columns: Qualified ``table.column`` or ``*.column`` specs to exclude. Empty means none denied. Three-part ``source.table.column`` forms are accepted and normalised to ``table.column``. Must not name the same table as an ``allow_objects`` or ``deny_objects`` entry. allow_columns: Qualified ``table.column`` or ``*.column`` specs that further restrict visible columns. Empty means all columns of tables in scope (subject to denies). Three-part forms normalise like ``deny_columns``. notes_ file: Optional path to domain notes whose content hash enters the scope fingerprint. ``None`` (default) means no notes file. sql_ file: Optional path to DDL whose content hash enters the scope fingerprint and dialect probe. ``None`` (default) means no DDL file. :class:`FederationContext` deliberately omits this field. Raises: ConfigError: ``include`` is not one of ``tables`` / ``views`` / ``both``; ``notes_file`` or ``sql_file`` is present but blank; a column spec is not ``table.column`` / ``*.column`` / ``source.table.column``; ``allow_objects`` overlaps ``deny_objects``; or a table-specific ``deny_columns`` entry conflicts with ``allow_objects`` or ``deny_objects``. ValueError: An allow/deny identifier is empty after strip/lower."""

    allow_objects: frozenset[str] = frozenset()
    include: SchemaInclude = SchemaInclude.TABLES
    deny_objects: frozenset[str] = frozenset()
    deny_columns: frozenset[str] = frozenset()
    allow_columns: frozenset[str] = frozenset()
    notes_file: str | None = None
    notes: str | None = None
    sql_file: str | None = None

    def __post_init__(self) -> None:
        if self.include not in ("tables", "views"):
            raise ConfigError(f"include must be 'tables' or 'views', not {self.include!r}")
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        if self.notes is not None:
            notes_stripped = str(self.notes).strip()
            if not notes_stripped:
                raise ConfigError("notes must be omitted or non-empty text")
            object.__setattr__(self, "notes", notes_stripped)
        if self.notes is not None and self.notes_file is not None:
            raise ConfigError("set at most one of notes and notes_file")
        if self.sql_file is not None and not str(self.sql_file).strip():
            raise ConfigError("sql_file must be omitted or a non-empty path")
        allow_norm = frozenset(
            SchemaNaming.norm_schema_identifier(t, what="allow_objects entry") for t in self.allow_objects
        )
        deny_obj_norm = frozenset(
            SchemaNaming.norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects
        )
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"allow_objects and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_specs: list[str] = []
        for spec in self.deny_columns:
            normalized_specs.append(SchemaNaming.normalize_scope_column_spec(spec, field="deny_columns"))
        col_set = frozenset(normalized_specs)
        allow_col_specs: list[str] = []
        for spec in self.allow_columns:
            allow_col_specs.append(SchemaNaming.normalize_scope_column_spec(spec, field="allow_columns"))
        allow_col_set = frozenset(allow_col_specs)
        for t in allow_norm:
            for spec in col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"allow_objects entry {t!r} is denied by deny_columns entry {spec!r}")
        for t in deny_obj_norm:
            for spec in col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"deny_objects entry {t!r} conflicts with deny_columns entry {spec!r}")
        object.__setattr__(self, "allow_objects", allow_norm)
        object.__setattr__(self, "deny_objects", deny_obj_norm)
        object.__setattr__(self, "deny_columns", col_set)
        object.__setattr__(self, "allow_columns", allow_col_set)

    def qualified_denies(self) -> frozenset[tuple[str, str]]:
        """Return the subset of ``deny_columns`` parsed as ``(table, column)`` pairs."""
        result: set[tuple[str, str]] = set()
        for spec in self.deny_columns:
            if "." in spec:
                tbl, col = spec.split(".", 1)
                if tbl != "*":
                    result.add((tbl, col))
        return frozenset(result)

    def glob_column_denies(self) -> frozenset[str]:
        """Return column suffixes from ``*.column`` deny specs."""
        result: set[str] = set()
        for spec in self.deny_columns:
            tbl, col = spec.split(".", 1)
            if tbl == "*":
                result.add(col)
        return frozenset(result)

    def bare_denies(self) -> frozenset[str]:
        """Return bare deny tokens; always empty because only qualified ``table.column`` specs are accepted."""
        return frozenset()

    def qualified_allows(self) -> frozenset[tuple[str, str]]:
        """Return the subset of ``allow_columns`` parsed as ``(table, column)`` pairs."""
        result: set[tuple[str, str]] = set()
        for spec in self.allow_columns:
            tbl, col = spec.split(".", 1)
            if tbl != "*":
                result.add((tbl, col))
        return frozenset(result)

    def glob_column_allows(self) -> frozenset[str]:
        """Return column suffixes from ``*.column`` allow specs."""
        result: set[str] = set()
        for spec in self.allow_columns:
            tbl, col = spec.split(".", 1)
            if tbl == "*":
                result.add(col)
        return frozenset(result)

    def bare_allows(self) -> frozenset[str]:
        """Return bare allow tokens; always empty because only qualified ``table.column`` specs are accepted."""
        return frozenset()


@dataclass(frozen=True, slots=True)
class FederationContext:
    """Frozen scope object that narrows what a question may touch on an :class:`~aetherdialect.AetherFederation` (several member engines). Parallel to :class:`EngineContext` (single connection) and :class:`SpaceContext` (named subset at question time). Unlike :class:`EngineContext`, this type has no ``sql_file`` field; unlike :class:`SpaceContext`, it is not a named subset. ``notes_file`` content hash feeds the composite scope identity. Mappings-aware validation (see :func:`~aetherdialect._federation.validate_federation_context_against_mappings`) requires that denies of collapsed members use the **logical** table name, and rejects partial denies of ``union`` / ``replica`` member sets. Args: allow_objects: Relation names permitted in composite scope. Empty means unrestricted. Prefer logical names for collapsed mapped tables. include: Catalog kinds considered when building member graphs under this scope: ``tables`` or ``views`` (default ``tables``). ``both`` is rejected. deny_objects: Relation names excluded from composite scope. Empty means none denied. Must not overlap ``allow_objects``. For collapsed mapped tables, name the logical table, not a physical member table. deny_columns: Qualified ``table.column`` or ``*.column`` specs to exclude. Empty means none denied. Three-part ``source.table.column`` forms are accepted and normalised to ``table.column``. Must not name the same table as an ``allow_objects`` or ``deny_objects`` entry. allow_columns: Qualified ``table.column`` or ``*.column`` specs that further restrict visible columns. Empty means all columns of tables in scope (subject to denies). Three-part forms normalise like ``deny_columns``. notes_file: Optional path to domain notes whose content hash enters the composite scope fingerprint. ``None`` (default) means no notes file. Raises: ConfigError: ``include`` is not one of ``tables`` / ``views`` / ``both``; ``notes_file`` is present but blank; a column spec is not ``table.column`` / ``*.column`` / ``source.table.column``; ``allow_objects`` overlaps ``deny_objects``; or a table-specific ``deny_columns`` entry conflicts with ``allow_objects`` or ``deny_objects``. Also raised by mappings-aware validation when a scope entry names a collapsed physical member table, or partially denies a ``union`` / ``replica`` logical table. ValueError: An allow/deny identifier is empty after strip/lower."""

    allow_objects: frozenset[str] = frozenset()
    include: SchemaInclude = SchemaInclude.TABLES
    deny_objects: frozenset[str] = frozenset()
    deny_columns: frozenset[str] = frozenset()
    allow_columns: frozenset[str] = frozenset()
    notes_file: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.include not in ("tables", "views"):
            raise ConfigError(f"include must be 'tables' or 'views', not {self.include!r}")
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        if self.notes is not None:
            notes_stripped = str(self.notes).strip()
            if not notes_stripped:
                raise ConfigError("notes must be omitted or non-empty text")
            object.__setattr__(self, "notes", notes_stripped)
        if self.notes is not None and self.notes_file is not None:
            raise ConfigError("set at most one of notes and notes_file")
        allow_norm = frozenset(
            SchemaNaming.norm_schema_identifier(t, what="allow_objects entry") for t in self.allow_objects
        )
        deny_obj_norm = frozenset(
            SchemaNaming.norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects
        )
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"allow_objects and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_specs: list[str] = []
        for spec in self.deny_columns:
            normalized_specs.append(SchemaNaming.normalize_scope_column_spec(spec, field="deny_columns"))
        col_set = frozenset(normalized_specs)
        allow_col_specs: list[str] = []
        for spec in self.allow_columns:
            allow_col_specs.append(SchemaNaming.normalize_scope_column_spec(spec, field="allow_columns"))
        allow_col_set = frozenset(allow_col_specs)
        for t in allow_norm:
            for spec in col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"allow_objects entry {t!r} is denied by deny_columns entry {spec!r}")
        for t in deny_obj_norm:
            for spec in col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"deny_objects entry {t!r} conflicts with deny_columns entry {spec!r}")
        object.__setattr__(self, "allow_objects", allow_norm)
        object.__setattr__(self, "deny_objects", deny_obj_norm)
        object.__setattr__(self, "deny_columns", col_set)
        object.__setattr__(self, "allow_columns", allow_col_set)

    def qualified_denies(self) -> frozenset[tuple[str, str]]:
        """Return the subset of ``deny_columns`` parsed as ``(table, column)`` pairs."""
        result: set[tuple[str, str]] = set()
        for spec in self.deny_columns:
            if "." in spec:
                tbl, col = spec.split(".", 1)
                if tbl != "*":
                    result.add((tbl, col))
        return frozenset(result)

    def glob_column_denies(self) -> frozenset[str]:
        """Return column suffixes from ``*.column`` deny specs."""
        result: set[str] = set()
        for spec in self.deny_columns:
            tbl, col = spec.split(".", 1)
            if tbl == "*":
                result.add(col)
        return frozenset(result)

    def bare_denies(self) -> frozenset[str]:
        """Return bare deny tokens; always empty because only qualified ``table.column`` specs are accepted."""
        return frozenset()

    def qualified_allows(self) -> frozenset[tuple[str, str]]:
        """Return the subset of ``allow_columns`` parsed as ``(table, column)`` pairs."""
        result: set[tuple[str, str]] = set()
        for spec in self.allow_columns:
            tbl, col = spec.split(".", 1)
            if tbl != "*":
                result.add((tbl, col))
        return frozenset(result)

    def glob_column_allows(self) -> frozenset[str]:
        """Return column suffixes from ``*.column`` allow specs."""
        result: set[str] = set()
        for spec in self.allow_columns:
            tbl, col = spec.split(".", 1)
            if tbl == "*":
                result.add(col)
        return frozenset(result)

    def bare_allows(self) -> frozenset[str]:
        """Return bare allow tokens; always empty because only qualified ``table.column`` specs are accepted."""
        return frozenset()


@dataclass(frozen=True, slots=True)
class BusinessKnowledgeEntry:
    """Single prompt-time business knowledge item not tied to schema column descriptions."""

    key: str
    text: str
    kind: str = "glossary"

    @staticmethod
    def normalize(entry: BusinessKnowledgeEntry) -> BusinessKnowledgeEntry:
        """Strip whitespace, default blank kind to glossary, and refuse unknown kinds."""
        key = str(entry.key).strip()
        text = str(entry.text).strip()
        kind_raw = str(entry.kind or BUSINESS_KNOWLEDGE_DEFAULT_KIND).strip() or BUSINESS_KNOWLEDGE_DEFAULT_KIND
        allowed = {member.value for member in BusinessKnowledgeKind}
        if kind_raw not in allowed:
            raise ConfigError(f"unknown business knowledge kind: {kind_raw!r}")
        kind = kind_raw
        if not key:
            raise ConfigError("business knowledge entry key must be non-empty")
        if not text:
            raise ConfigError(f"business knowledge entry {key!r} must have non-empty text")
        if kind != entry.kind or key != entry.key or text != entry.text:
            return BusinessKnowledgeEntry(key=key, text=text, kind=kind)
        return entry

    @staticmethod
    def hidden_column_references(text: str, schema_graph: Any) -> list[str]:
        """Return qualified hidden column names referenced in *text*."""
        found: list[str] = []
        seen: set[str] = set()
        for match in BUSINESS_KNOWLEDGE_COLUMN_REF_RE.finditer(text):
            table_name, column_name = match.group(1), match.group(2)
            qualified = f"{table_name}.{column_name}"
            if qualified in seen:
                continue
            seen.add(qualified)
            table = schema_graph.tables.get(table_name)
            if table is None:
                continue
            column = table.columns.get(column_name)
            if column is None:
                continue
            if column.sensitivity == SensitivityClassification.HIDDEN:
                found.append(qualified)
        return found


@dataclass
class BusinessKnowledgeState:
    """Versioned snapshot of active business knowledge entries and digest."""

    version: int = 0
    entries: tuple[BusinessKnowledgeEntry, ...] = ()
    digest: str = ""

    @staticmethod
    def digest_for(entries: Sequence[BusinessKnowledgeEntry]) -> str:
        """Stable SHA-256 digest over normalized business knowledge entries."""
        payload = [{"key": entry.key, "kind": entry.kind, "text": entry.text} for entry in entries]
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def empty_digest() -> str:
        """Return the digest for an empty knowledge set."""
        return BusinessKnowledgeState.digest_for(())

    @staticmethod
    def validate_entries(
        entries: Sequence[BusinessKnowledgeEntry],
        schema_graph: Any,
    ) -> tuple[BusinessKnowledgeEntry, ...]:
        """Normalize entries and refuse hidden-column references."""
        normalized: list[BusinessKnowledgeEntry] = []
        seen_keys: set[str] = set()
        for raw in entries:
            if not isinstance(raw, BusinessKnowledgeEntry):
                raise TypeError("business knowledge entries must be BusinessKnowledgeEntry instances")
            entry = BusinessKnowledgeEntry.normalize(raw)
            if entry.key in seen_keys:
                raise ConfigError(f"duplicate business knowledge key: {entry.key!r}")
            seen_keys.add(entry.key)
            hidden_refs = BusinessKnowledgeEntry.hidden_column_references(entry.text, schema_graph)
            if hidden_refs:
                joined = ", ".join(sorted(hidden_refs))
                raise ConfigError(f"business knowledge entry {entry.key!r} references hidden column(s): {joined}")
            normalized.append(entry)
        return tuple(normalized)


class BusinessKnowledgeHolder:
    """Mutable versioned store for engine- or federation-level business knowledge."""

    def __init__(self) -> None:
        self._state = BusinessKnowledgeState(digest=BusinessKnowledgeState.empty_digest())

    def set(self, entries: Sequence[BusinessKnowledgeEntry], schema_graph: Any) -> int:
        normalized = BusinessKnowledgeState.validate_entries(entries, schema_graph)
        digest = BusinessKnowledgeState.digest_for(normalized)
        self._state = BusinessKnowledgeState(
            version=self._state.version + 1,
            entries=normalized,
            digest=digest,
        )
        return self._state.version

    def entries(self) -> tuple[BusinessKnowledgeEntry, ...]:
        return self._state.entries

    def digest(self) -> str:
        return self._state.digest

    def version(self) -> int:
        return self._state.version

    def scope_kwargs(self) -> dict[str, Any]:
        """Keyword args for business-knowledge scope binding."""
        return {"entries": self._state.entries, "digest": self._state.digest}


@dataclass(frozen=True, slots=True)
class OpenResourceInventory:
    """Test-only snapshot of library-owned resources still open."""

    locks: int
    temp_directories: int
    live_connections: int


@dataclass(frozen=True, slots=True)
class RenameMigrationAssessment:
    """Inferred rename migration with a confidence score in ``[0, 1]``."""

    plan: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]
    confidence: float


@dataclass(frozen=True)
class MemberEffectiveGrants:
    """Tables and columns a federation member role may read at composition time."""

    tables: frozenset[str]
    columns: frozenset[tuple[str, str]] | None = None


class FederationMemberEngine(Protocol):
    """Minimal member-engine surface used when deriving federation source bindings."""

    dialect: str
    _connection: object
    _context_name: object
    _schema_role: object
    _runtime_config: object | None


@dataclass(frozen=True, slots=True)
class SpaceContext:
    """
    Frozen scope object that narrows what a question may touch inside a named :class:`~aetherdialect.AetherSpace` subset. Parallel to :class:`EngineContext` (single-connection construction scope) and :class:`FederationContext` (multi-member construction scope). Unlike those types, this context is applied at question time only: it does not rebuild or re-hash the catalog, has no ``include`` / ``sql_file``, and uses ``tables`` / ``columns`` as its allow lists (same role as ``allow_objects`` / ``allow_columns`` on the other two). Optional ``notes_file`` is parallel to the other two contexts, but its content hash is recorded in the aetherspace snapshot so notes changes are detectable; it does not enter a catalog-rebuild fingerprint. Allow lists and deny lists interact as a narrow-then-exclude pair: ``tables`` and ``columns`` define the visible subset (empty allow = unrestricted at that level), then ``deny_objects`` and ``deny_columns`` remove names from what remains. A non-empty ``tables`` list requires every ``columns`` entry to name a table in that list; ``tables`` must not overlap ``deny_objects``; and a table-specific ``deny_columns`` entry must not target a table already in ``deny_objects``.

    Args:

        tables: Allowed table names (counterpart of ``allow_objects``). Empty
        means unrestricted over the parent graph's tables.
        columns: Allowed qualified ``table.column`` or ``*.column`` specs
        (counterpart of ``allow_columns``). Empty means all columns of tables
        in scope. Three-part ``source.table.column`` forms normalise to
        ``table.column``. When ``tables`` is non-empty, each entry's table
        must appear in ``tables``.
        deny_objects: Table names excluded after allow filtering. Empty means
        none denied. Must not overlap ``tables``.
        deny_columns: Qualified ``table.column`` or ``*.column`` specs excluded
        after allow filtering. Empty means none denied. Must not conflict
        with ``deny_objects`` for the same table.
        notes_file: Optional path to domain notes whose content is baked into
        the aetherspace snapshot (text plus content hash). ``None``
        (default) means no notes file. Unlike :class:`EngineContext` /
        :class:`FederationContext`, this hash does not feed a catalog
        rebuild fingerprint.

    Raises:

        ConfigError: ``notes_file`` is present but blank; a column spec is not
        ``table.column`` / ``*.column`` / ``source.table.column``; ``tables``
        overlaps ``deny_objects``; a ``columns`` entry references a table
        absent from a non-empty ``tables`` list; or a table-specific
        ``deny_columns`` entry conflicts with ``deny_objects``.
        ValueError: A table or column identifier is empty after strip/lower.
    """

    tables: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()
    deny_objects: frozenset[str] = frozenset()
    deny_columns: frozenset[str] = frozenset()
    notes_file: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        if self.notes is not None:
            notes_stripped = str(self.notes).strip()
            if not notes_stripped:
                raise ConfigError("notes must be omitted or non-empty text")
            object.__setattr__(self, "notes", notes_stripped)
        if self.notes is not None and self.notes_file is not None:
            raise ConfigError("set at most one of notes and notes_file")
        allow_norm = frozenset(SchemaNaming.norm_schema_identifier(t, what="tables entry") for t in self.tables)
        deny_obj_norm = frozenset(
            SchemaNaming.norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects
        )
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"SpaceContext tables and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_cols: list[str] = []
        for spec in self.columns:
            normalized_cols.append(SchemaNaming.normalize_scope_column_spec(spec, field="columns"))
        col_set = frozenset(normalized_cols)
        deny_col_specs: list[str] = []
        for spec in self.deny_columns:
            deny_col_specs.append(SchemaNaming.normalize_scope_column_spec(spec, field="deny_columns"))
        deny_col_set = frozenset(deny_col_specs)
        if allow_norm:
            for qc in col_set:
                tbl_part = qc.rsplit(".", 1)[0]
                if tbl_part not in allow_norm:
                    raise ConfigError(f"columns entry {qc!r} references table {tbl_part!r} not listed in tables")
        for t in deny_obj_norm:
            for spec in deny_col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"deny_objects entry {t!r} conflicts with deny_columns entry {spec!r}")
        object.__setattr__(self, "tables", allow_norm)
        object.__setattr__(self, "columns", col_set)
        object.__setattr__(self, "deny_objects", deny_obj_norm)
        object.__setattr__(self, "deny_columns", deny_col_set)


@dataclass(frozen=True, slots=True)
class FederationSourceLimits:
    """Per-source row cap and timeout overrides from a federation manifest."""

    row_cap: int | None = None
    timeout_ms: int | None = None
    semijoin_enabled: bool = True
    max_query_cost_rows: float | None = None
    max_query_cost_bytes: float | None = None
    profile_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class FederationSourceBinding:
    """One member-source binding in a federation manifest. Identifies a federated member by ``source_id`` (the registration key used in ``members`` maps, plan ``source_ids``, and composite ``table_namespace``). When deriving a binding from a live engine via :func:`~aetherdialect._federation.binding_from_member_engine`, a non-empty engine connection handle must equal that registration key. Args: source_id: Stable member identity / registration key. engine: Dialect/engine type label for the member (for example ``postgresql``, ``duckdb``). connection: Named connection handle within the engine config. Empty string (default) means unset; when set on a live engine it must match ``source_id``. context: Named engine-context label the member is bound under (default ``master``). role: Schema identity role for the member: ``owner`` or ``consumer`` (default ``consumer``). limits: Optional per-source row-cap / timeout / semijoin overrides. ``None`` (default) means use coordinator defaults. Raises: FederationConfigError: Not raised by this dataclass constructor itself; raised by :func:`~aetherdialect._federation.binding_from_member_engine` when the registration key disagrees with a non-empty engine connection handle (or the member role is not ``owner`` / ``consumer``)."""

    source_id: str
    engine: str
    connection: str = ""
    context: str = "master"
    role: SchemaRole = SchemaRole.CONSUMER
    limits: FederationSourceLimits | None = None
    session_timezone: str | None = None


@dataclass(frozen=True, slots=True)
class FederationCoordinatorConfig:
    """In-process DuckDB coordinator bounds and default per-source execution limits. The federation coordinator always materializes and combines member frames in DuckDB; engine selection is not configurable. Args: row_cap: Maximum total coordinator input/result rows. default_source_row_cap: Default per-member row cap when a source omits limits. default_source_timeout_ms: Default per-member timeout when a source omits limits. coordinator_timeout_ms: Wall-clock timeout for coordinator glue SQL execution. plan_timeout_ms: Wall- clock budget for an entire federated plan execution. semijoin_key_cap: Maximum distinct keys pushed as a semijoin filter. spill_row_threshold: Row count above which member frames spill to parquet. max_parallel_members: Maximum concurrent member query executions. total_input_byte_cap: Maximum total in-memory bytes across coordinator inputs."""

    row_cap: int = 500_000
    default_source_row_cap: int = 500_000
    default_source_timeout_ms: int = 30_000
    coordinator_timeout_ms: int = 30_000
    plan_timeout_ms: int = 300_000
    semijoin_key_cap: int = 50_000
    semijoin_key_distinct_floor: int = 2
    spill_row_threshold: int = 50_000
    max_parallel_members: int = 4
    total_input_byte_cap: int = 2_000_000_000


@dataclass(frozen=True, slots=True)
class FederationCrossSourceJoin:
    """Declared cross-source join edge (``left``/``right`` are ``table.column`` qualified)."""

    left: str
    right: str
    kind: str
    logical_key: str


@dataclass(frozen=True, slots=True)
class FederationTableAlias:
    """Explicit composite table name for one member physical table."""

    alias: str
    source: str
    table: str


@dataclass(frozen=True, slots=True)
class FederationManifest:
    """Authoritative federation deployment description."""

    federation_id: str
    sources: tuple[FederationSourceBinding, ...]
    table_namespace: dict[str, str]
    cross_source_joins: tuple[FederationCrossSourceJoin, ...]
    coordinator: FederationCoordinatorConfig
    aliases: tuple[FederationTableAlias, ...] = ()


@dataclass(frozen=True, slots=True)
class LogicalColumnMapping:
    """Operator-declared equivalence for one logical attribute across sources."""

    logical: str
    members: tuple[str, ...]
    role: str
    unify_in_graph: bool


@dataclass(frozen=True, slots=True)
class LogicalTableMember:
    """One physical table backing a logical federated table."""

    source: str
    table: str
    columns: dict[str, str]


@dataclass(frozen=True, slots=True)
class LogicalTableMapping:
    """Operator-declared equivalence for one logical table across sources."""

    logical: str
    members: tuple[LogicalTableMember, ...]
    semantics: Literal["union", "replica"]
    authoritative_source: str = ""


@dataclass(frozen=True, slots=True)
class FederationMappings:
    """Cross-source mapping sidecar replayed on composite rebuild."""

    version: str
    logical_columns: tuple[LogicalColumnMapping, ...] = ()
    logical_tables: tuple[LogicalTableMapping, ...] = ()


@dataclass(frozen=True, slots=True)
class FederationPlanTemplate:
    """Stored federation decomposition fingerprint keyed on the composite graph."""

    plan_id: str
    composite_schema_graph_id: str
    intent_key: str
    step_fingerprints: tuple[tuple[str, str], ...]
    combine_hash: str
    question: str = ""
    accepted_questions: tuple[str, ...] = ()
    format_version: str = "0.2.1"
    member_template_ids: tuple[tuple[str, str], ...] = ()
    residual_hash: str = ""
    join_feedback: tuple[str, ...] = ()
    manifest_hash: str = ""
    member_tuple_hash: str = ""


@dataclass(frozen=True, slots=True)
class FederationQualifiedRename:
    """One qualified ``table.column`` rename inside a federation migration map."""

    from_ref: str
    to_ref: str


@dataclass(frozen=True, slots=True)
class FederationMigrationMap:
    """Operator-authored federation migration consumed once at composite init."""

    version: int
    action: str
    qualified_column_renames: tuple[FederationQualifiedRename, ...] = ()
    namespace_renames: tuple[tuple[str, str], ...] = ()
    dropped_cross_source_joins: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class FederationMappingSuggestion:
    """Advisory cross-source mapping candidate; never applied without operator action."""

    logical: str
    members: tuple[str, ...]
    kind: str
    score: float
    role: str = "join_key"


@dataclass(frozen=True, slots=True)
class PlanPreviewResult:
    """Read-only projection of what a turn would run before execution."""

    question: str
    tables: tuple[str, ...]
    join_path: tuple[str, ...]
    member_source_ids: tuple[str, ...] = ()
    federates: bool = False
    ineligible_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRuntime:
    """Per-source bound dialect and artifact scope for federated execution."""

    source_id: str
    engine: str
    connection: str
    artifacts_dir: str
    dialect: Any
    sqlglot_dialect: str = ""
    native_connection: Any = None
    sqlalchemy_engine: Any = None


@dataclass(frozen=True, slots=True)
class AetherSpace:
    """
    Read-only descriptor for a named aetherspace scope.

    Args:

        name: Normalised space name.
        _scope: Internal ``tables`` / ``columns`` tuples for the space subset.

    notes: Merged notes text baked from :attr:`SpaceContext.notes_file` at
        define time, or ``None`` when no notes file was supplied.
    """

    name: str
    _scope: dict[str, tuple[str, ...]]
    notes: str | None = None

    def list_scope(self) -> dict[str, tuple[str, ...]]:
        """Return ``tables`` and ``columns`` tuples describing this space."""
        return dict(self._scope)


@dataclass(frozen=True, slots=True)
class CteIntent:
    """Planner-only natural-language description of one reusable intermediate aligned with a runtime CTE step."""

    name: str
    tables: tuple[str, ...] = ()
    select: str = ""
    where: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str | None = None
    window: str = ""
    case: str = ""


@dataclass(frozen=True, slots=True)
class LogicalIntent:
    """Planner-only natural-language plan consumed by the encoder; not persisted and not structural IR."""

    tables: tuple[str, ...]
    select: str
    where: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str | None = None
    window: str = ""
    case: str = ""
    cte_steps: tuple[CteIntent, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Engine, artifact root, frozen schema scope, and merged LLM plus execution limits for runtime introspection."""

    engine: str
    artifacts_dir: str
    engine_context: EngineContext | FederationContext
    llm_execution: LlmExecutionConfig
    execution_context: EngineContext | FederationContext | None = None


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Active LLM provider label after environment configuration."""

    provider: Literal["openai", "azure", "mock"]


@dataclass
class AetherEngineInitResult:
    """Mutable template bundle and graph produced by engine initialisation."""

    runtime_config: RuntimeConfig
    llm_config: LLMConfig
    schema_graph: Any
    dialect: Any
    artifacts_dir: str
    store: Any
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema_terms: set[str]
    schema_stats: dict[str, Any]
    schema_role: SchemaRole = SchemaRole.OWNER
    consumer_visible_objects: frozenset[str] | None = None
    context_name: str = "master"
    execution_context: EngineContext | FederationContext | None = None
    data_quality_report: DataQualityReport | None = None
    federation_manifest: FederationManifest | None = None
    federation_mappings: FederationMappings | None = None
    federation_member_graphs: dict[str, Any] | None = None
    federation_storage_dir: str | None = None
    federation_source_runtimes: dict[str, SourceRuntime] | None = None
    federation_mapping_suggestions: tuple[FederationMappingSuggestion, ...] = ()
    federation_dialects_by_source: dict[str, Any] | None = None
    engine_identity: EngineIdentity | None = None


@dataclass
class AetherFederationInitResult(AetherEngineInitResult):
    """Init bundle for :class:`~aetherdialect.AetherFederation`."""

    members: dict[str, Any] | None = None


class ColumnRole(Enum):
    """Column role for profiling and question simulation."""

    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    NUMERIC_CATEGORICAL = "numeric_categorical"
    NUMERIC_MEASURE = "numeric_measure"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    FREE_TEXT = "free_text"
    AUDIT = "audit"


class TableRole(Enum):
    """Table role for join constraint validation."""

    DIMENSION = "dimension"
    FACT = "fact"
    BRIDGE = "bridge"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkloadFamilySpec:
    """Declarative workload shape metadata for schema realization, sampling, and coverage keys."""

    family: WorkloadFamily
    preferred_complexity: ComplexityTier | None
    ranking_policy: str
    comparison_mode: str
    cardinality_regime: str
    required_table_roles: tuple[str, ...]


WORKLOAD_FAMILY_SPECS: dict[WorkloadFamily, WorkloadFamilySpec] = {
    WorkloadFamily.STATUS_REPORT: WorkloadFamilySpec(
        WorkloadFamily.STATUS_REPORT, ComplexityTier.MODERATE, "rank_none", "none", "small", ("fact",)
    ),
    WorkloadFamily.BREAKDOWN: WorkloadFamilySpec(
        WorkloadFamily.BREAKDOWN, ComplexityTier.MODERATE, "none", "categorical_slice", "medium", ("fact", "dimension")
    ),
    WorkloadFamily.LEADERBOARD: WorkloadFamilySpec(
        WorkloadFamily.LEADERBOARD, ComplexityTier.MODERATE, "top_k", "ordered_metric", "small", ("fact",)
    ),
    WorkloadFamily.TREND: WorkloadFamilySpec(
        WorkloadFamily.TREND, ComplexityTier.COMPLEX, "time_series", "temporal_sequence", "medium", ("fact",)
    ),
    WorkloadFamily.CHANGE_OVER_TIME: WorkloadFamilySpec(
        WorkloadFamily.CHANGE_OVER_TIME,
        ComplexityTier.COMPLEX,
        "period_over_period",
        "temporal_delta",
        "medium",
        ("fact",),
    ),
    WorkloadFamily.SHARE_OF_TOTAL: WorkloadFamilySpec(
        WorkloadFamily.SHARE_OF_TOTAL, ComplexityTier.COMPLEX, "ratio", "part_whole", "small", ("fact", "dimension")
    ),
    WorkloadFamily.SEGMENT_COMPARISON: WorkloadFamilySpec(
        WorkloadFamily.SEGMENT_COMPARISON,
        ComplexityTier.COMPLEX,
        "none",
        "cohort_contrast",
        "medium",
        ("fact", "dimension"),
    ),
    WorkloadFamily.THRESHOLD_EXCEPTION: WorkloadFamilySpec(
        WorkloadFamily.THRESHOLD_EXCEPTION,
        ComplexityTier.MODERATE,
        "exception_filter",
        "predicate_cutoff",
        "small",
        ("fact",),
    ),
    WorkloadFamily.EXTRACT: WorkloadFamilySpec(
        WorkloadFamily.EXTRACT, ComplexityTier.SIMPLE, "none", "none", "many", ("fact",)
    ),
    WorkloadFamily.LIFECYCLE_COHORT: WorkloadFamilySpec(
        WorkloadFamily.LIFECYCLE_COHORT,
        ComplexityTier.HIGHLY_COMPLEX,
        "cohort_retention",
        "lifecycle",
        "medium",
        ("fact", "dimension"),
    ),
    WorkloadFamily.EXPLORATION_FOLLOWUP: WorkloadFamilySpec(
        WorkloadFamily.EXPLORATION_FOLLOWUP, ComplexityTier.SIMPLE, "none", "ad_hoc", "many", ("fact",)
    ),
}


class InferenceTag(StrEnum):
    """Provenance tag for an :class:`FKEdge`. A catalog-declared edge is represented by ``None`` rather than a member of this enum so that presence-of-tag and identity-of- inferred-layer are reflected by a single attribute. Inherits ``str`` so members compare equal to their wire value and round-trip through JSON without custom encoding."""

    SUFFIX = "suffix"
    SELF = "self"
    COMPOSITE = "composite"
    SEMANTIC = "semantic"
    SEMANTIC_PROMOTED = "semantic_promoted"
    USER_STRUCTURAL = "user_override_structural"
    USER_SEMANTIC = "user_override_semantic"
    CROSS_SOURCE = "cross_source"

    @classmethod
    def coerce(cls, raw: object) -> InferenceTag | None:
        """Normalise raw cache or override input into :class:`InferenceTag` (``None`` for catalog)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in INFERENCE_TAG_VALUES:
            return cls(raw)
        raise ValueError(f"unknown FK inference_tag: {raw!r}")


class PkInferenceTag(StrEnum):
    """Provenance tag for an inferred or user-supplied primary key. Engine-reflected catalog keys use ``None`` (locked). SQL-file- declared keys use ``DDL`` (overridable). Inferred and user-supplied keys use the remaining members."""

    DDL = "ddl"
    IDENTITY = "identity"
    PROFILE = "profile"
    USER_OVERRIDE = "user_override"

    @classmethod
    def coerce(cls, raw: object) -> PkInferenceTag | None:
        """Normalise raw cache or override input into :class:`PkInferenceTag` (``None`` for catalog)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in PK_INFERENCE_TAG_VALUES:
            return cls(raw)
        raise ValueError(f"unknown pk_inference_tag: {raw!r}")


class RoleOwner(StrEnum):
    """Provenance for the writer that last set :attr:`ColumnMetadata.role`. The members are ordered by ascending precedence: a writer with strictly greater precedence may overwrite a role assigned by a lower-precedence owner, while equal-or-lower-precedence writers must skip the column. PK/FK coercion is treated as the highest authority because it is required for join correctness; user overrides win over LLM inference, which in turn wins over profile heuristics, which in turn wins over the default catalog fallback."""

    CATALOG = "catalog"
    PROFILE = "profile"
    LLM = "llm"
    BOOLEAN_COERCION = "boolean_coercion"
    USER_OVERRIDE = "user_override"
    PK_FK_COERCION = "pk_fk_coercion"

    @classmethod
    def coerce(cls, raw: object) -> RoleOwner | None:
        """Normalise raw cache or override input into :class:`RoleOwner` (``None`` when unset)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in ROLE_OWNER_VALUES:
            return cls(raw)
        raise ValueError(f"unknown role_owner: {raw!r}")

    @classmethod
    def can_overwrite(cls, current: RoleOwner | None, candidate: RoleOwner) -> bool:
        """Return whether a writer with provenance *candidate* may overwrite a role currently owned by *current*."""
        if current is None:
            return True
        return _ROLE_OWNER_PRECEDENCE[candidate] > _ROLE_OWNER_PRECEDENCE[current]


_ROLE_OWNER_PRECEDENCE: dict[RoleOwner, int] = {
    RoleOwner.CATALOG: 0,
    RoleOwner.PROFILE: 1,
    RoleOwner.LLM: 2,
    RoleOwner.BOOLEAN_COERCION: 3,
    RoleOwner.USER_OVERRIDE: 4,
    RoleOwner.PK_FK_COERCION: 5,
}


class DescriptionOwner(StrEnum):
    """Provenance for the writer that last set a description on a table or column. Members are ordered by ascending precedence; :meth:`set_on` enforces a strict-greater-precedence rule so a later writer can only overwrite an existing description when its provenance outranks the incumbent owner."""

    CATALOG = "catalog"
    PROFILE = "profile"
    NOTES = "notes"
    LLM_REFINEMENT = "llm_refinement"
    SPACE_NOTES = "space_notes"
    USER_OVERRIDE = "user_override"

    @classmethod
    def coerce(cls, raw: object) -> DescriptionOwner | None:
        """Normalise raw cache or override input into :class:`DescriptionOwner` (``None`` when unset)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in DESCRIPTION_OWNER_VALUES:
            return cls(raw)
        raise ValueError(f"unknown description_owner: {raw!r}")

    @classmethod
    def _rank(cls, owner: DescriptionOwner | None) -> int:
        if owner is None:
            return _DESCRIPTION_OWNER_PRECEDENCE[cls.CATALOG]
        if not isinstance(owner, cls):
            owner = cls(owner)
        return _DESCRIPTION_OWNER_PRECEDENCE[owner]

    @classmethod
    def resolve(
        cls,
        *candidates: tuple[str | None, DescriptionOwner | None],
    ) -> tuple[str, DescriptionOwner | None]:
        """Resolve simultaneous description candidates using owner precedence."""
        nonempty: list[tuple[str, DescriptionOwner | None]] = []
        for text, owner in candidates:
            cleaned = str(text or "").strip()
            if not cleaned:
                continue
            coerced = owner
            if coerced is not None and not isinstance(coerced, cls):
                coerced = cls(coerced)
            nonempty.append((cleaned, coerced))
        if not nonempty:
            return "", None
        max_rank = max(cls._rank(owner) for _, owner in nonempty)
        tier = [(text, owner) for text, owner in nonempty if cls._rank(owner) == max_rank]
        distinct_texts = sorted({text for text, _ in tier})
        if len(distinct_texts) == 1:
            winner_owner = next(owner for text, owner in tier if text == distinct_texts[0])
            return distinct_texts[0], winner_owner
        return "", None

    @classmethod
    def set_on(cls, target: Any, text: str | None, owner: DescriptionOwner) -> bool:
        """Single writer for ``description`` on tables and columns."""
        if text is None:
            return False
        current_owner = getattr(target, "description_owner", None)
        if current_owner is not None:
            if not isinstance(current_owner, cls):
                current_owner = cls(current_owner)
            if _DESCRIPTION_OWNER_PRECEDENCE[owner] < _DESCRIPTION_OWNER_PRECEDENCE[current_owner]:
                return False
        cur_desc = (getattr(target, "description", None) or "").strip()
        new_desc = str(text).strip()
        if cur_desc == new_desc and current_owner == owner:
            return False
        target.description = new_desc
        target.description_owner = owner
        return True


_DESCRIPTION_OWNER_PRECEDENCE: dict[DescriptionOwner, int] = {
    DescriptionOwner.CATALOG: 0,
    DescriptionOwner.PROFILE: 1,
    DescriptionOwner.NOTES: 2,
    DescriptionOwner.LLM_REFINEMENT: 3,
    DescriptionOwner.SPACE_NOTES: 4,
    DescriptionOwner.USER_OVERRIDE: 5,
}


class AccessError(SchemaAccessError, RuntimeError):
    """Raised when execute/explain/preview is refused by library scope or the warehouse."""

    def __init__(
        self,
        operation: Literal["explain", "execute", "preview_table"],
        message: str,
        *,
        relation: str | None = None,
        reason: Literal["scope", "warehouse"] = "warehouse",
    ) -> None:
        """Attach *operation*, human *message*, optional *relation*, and access *reason*."""
        self.operation = operation
        self.relation = relation
        self.reason = reason
        super().__init__(message)


class PipelineSuspended(AetherError):
    """Raised when a programmatic interactive turn must wait for the next ``submit_*`` call."""

    def __init__(self, state_id: str, message_for_caller: str, payload: Any | None = None) -> None:
        self.state_id = state_id
        self.message_for_caller = message_for_caller
        self.payload = payload
        super().__init__(message_for_caller)


class NoJoinPathError(AetherError):
    """Raised when multi-table scope has no foreign-key or semantic join path. This is a terminal, deterministic pipeline failure: no LLM call can invent a plausible join when neither the physical foreign-key graph nor the semantic edge set connects the requested tables."""

    def __init__(self, scope_label: str, tables: list[str]) -> None:
        self.scope_label = scope_label
        self.tables = list(tables)
        message = (
            f"No join path available in {scope_label} for tables: {', '.join(self.tables) if self.tables else '<none>'}"
        )
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation naming the disconnected tables."""
        tables = ", ".join(self.tables) if self.tables else "the requested tables"
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE]["user_text"].format(tables=tables)


class JoinPathTieCapExceededError(AetherError):
    """Raised when shortest-path enumeration for one table pair exceeds the refusal ceiling."""

    def __init__(self, source_table: str, target_table: str, path_count: int, ceiling: int) -> None:
        self.source_table = source_table
        self.target_table = target_table
        self.path_count = path_count
        self.ceiling = ceiling
        super().__init__(
            f"join path tie ceiling exceeded for {source_table!r} -> {target_table!r}: "
            f"{path_count} equal-length paths (limit {ceiling})"
        )

    @property
    def user_message(self) -> str:
        """User-facing explanation when too many equally short join paths exist."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP]["user_text"].format(
            source_table=self.source_table,
            target_table=self.target_table,
            path_count=str(self.path_count),
            ceiling=str(self.ceiling),
        )


class AggregateJoinFanOutError(AetherError):
    """Raised when a resolved join path would duplicate rows aggregated at parent grain."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ClauseWidenedRowsetError(AetherError):
    """Raised when LIMIT or DISTINCT ON would run on a join-widened row set."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation when clause modifiers conflict with join shape."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET]["user_text"]


class NullInNegatedListError(AetherError):
    """Raised when a NOT IN list literal contains null."""

    def __init__(self, column: str, message: str) -> None:
        self.column = column
        self.message_for_caller = message
        super().__init__(message)


class SubdayDateWindowOnDateColumnError(AetherError):
    """Raised when a sub-day date window is requested on a date-only column."""

    def __init__(self, column: str, message: str) -> None:
        self.column = column
        self.message_for_caller = message
        super().__init__(message)


class AmbiguousDateLiteralError(AetherError):
    """Raised when an absolute date bound is not valid ISO 8601."""

    def __init__(self, literal: str, message: str) -> None:
        self.literal = literal
        self.message_for_caller = message
        super().__init__(message)


class ProbeCtePlacementError(AetherError):
    """Raised when a semi-join or anti-join probe CTE is used as a join anchor or left operand."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation when a filter step is placed incorrectly in the join."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT]["user_text"]


class ComparisonJoinScopeExceededError(AetherError):
    """Raised when a cross-table comparison forces a join path beyond the allowed scope."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class JoinColumnCountMismatchError(AetherError):
    """Raised when a join signature pairs unequal numbers of left and right columns."""

    def __init__(self, segment: str, left_count: int, right_count: int) -> None:
        self.segment = segment
        self.left_count = left_count
        self.right_count = right_count
        super().__init__(
            f"join path segment {segment!r} pairs {left_count} left column(s) with {right_count} right column(s)"
        )


class JoinInjectionAlignmentError(AetherError):
    """Raised when ``join_sigs_ordered`` does not align one-to-one with dialect join carriers on deterministic SQL."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RegistryRenderError(AetherError, ValueError):
    """Raised when rendering an expression references a missing window or case registry id."""


class JoinInjectionFailedError(AetherError):
    """Raised when deterministic SQL cannot be rewritten with structured JOIN/WHERE edges via the dialect AST adapter."""

    def __init__(
        self, message: str, *, det_sql: str, join_sigs_ordered: list[list[str]], edge_kinds_ordered: list[list[str]]
    ) -> None:
        self.det_sql = det_sql
        self.join_sigs_ordered = join_sigs_ordered
        self.edge_kinds_ordered = edge_kinds_ordered
        super().__init__(message)


class LlmJsonExhausted(AetherError):
    """Raised by ``llm_json`` when every retry attempt fails to produce valid JSON. Callers decide whether exhaustion is recoverable (e.g., retry loops, deterministic fallbacks) or terminal."""

    def __init__(self, task: str, attempts: int) -> None:
        self.task = task
        self.attempts = attempts
        super().__init__(f"llm_json exhausted after {attempts} attempt(s) for task={task!r}")


@dataclass(frozen=True, slots=True)
class LlmBatchRequest:
    """One JSON-mode completion submitted through the OpenAI Batch API."""

    custom_id: str
    system: str
    user: str
    task: str = "default"


ScalarArg = str | int | float
ParamValue = str | int | float | bool | list[str | int | float]
RawValue = str | int | float | bool | list[str | int | float] | dict[str, str | int] | None

PROBE_CTE_EMISSION_KINDS: frozenset[str] = frozenset({CteEmissionKind.SEMI_JOIN.value, CteEmissionKind.ANTI_JOIN.value})


@dataclass
class ExprValue:
    """Parameterized literal value for expression arithmetic with param_key for template reuse."""

    value: Decimal | int | float | str | bool | None = 0
    param_key: str = ""

    @staticmethod
    def from_dict(d: Any) -> ExprValue:
        """
        Create ExprValue from dictionary.

        Args:

            d: Dictionary with 'value' and 'param_key' keys, or a bare numeric value.

        Returns:

            Populated ExprValue instance.
        """
        if isinstance(d, int | float | Decimal):
            return ExprValue(value=d)
        if isinstance(d, dict):
            return ExprValue(value=d.get("value", 0), param_key=d.get("param_key", ""))
        return ExprValue()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with 'value' and 'param_key' keys.
        """
        return {"value": self.value, "param_key": self.param_key}

    @property
    def signature_key(self) -> str:
        """Structural signature for template matching (ignores concrete. value). Returns: Always the string `val` (parameterisation uses `param_key` elsewhere)."""
        return "val"

    @staticmethod
    def sort_key(expr_value: ExprValue) -> tuple[int, str]:
        """Return a stable comparable key for sorting ``ExprValue`` lists."""
        value = expr_value.value
        if value is None:
            return (0, "")
        return (1, str(value))


@dataclass
class MulGroup:
    """Single multiplicative term: scalar_func(agg_func(inner_scalar_func(coefficient * multiply[0] * ... / divide[0] / ...))) with scalar_func_args and inner_scalar_func_args. `multiply` and `divide` carry nested `NormalizedExpr` sub-trees (column refs are leaf NormalizedExpr with `column_ref` set; CAST/COALESCE/EXTRACT/INTERVAL/ keyword nodes use the structural fields on `NormalizedExpr`). When `scalar_func` is ``concat``, `multiply` is an ordered list of CONCAT arguments rendered comma-separated inside ``CONCAT(...)``; `divide` and non- unit coefficients must remain empty. Otherwise `multiply` is a multiplicative chain rendered with ``*``."""

    coefficient: float = 1.0
    multiply: list[NormalizedExpr] = field(default_factory=list)
    divide: list[NormalizedExpr] = field(default_factory=list)
    agg_func: str | None = None
    scalar_func: str | None = None
    inner_scalar_func: str | None = None
    scalar_func_args: list[ScalarArg] = field(default_factory=list)
    inner_scalar_func_args: list[ScalarArg] = field(default_factory=list)
    coeff_param_key: str = ""
    sarg_param_keys: list[str] = field(default_factory=list)
    isarg_param_keys: list[str] = field(default_factory=list)
    distinct: bool = False
    agg_sep_param_key: str = ""
    agg_order_by: list[OrderByCol] = field(default_factory=list)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "agg_sep_param_key": "Param key for the string_agg delimiter literal.",
        "agg_order_by": "Optional within-aggregate ORDER BY list for string_agg.",
    }

    def __post_init__(self) -> None:
        """Coerce string entries to leaf NormalizedExpr, sort multiply/divide for multiplicative chains, preserve order for CONCAT args, and normalise function name casing/order."""
        multiply_terms = [NormalizedExpr.coerce_mul_term(t) for t in self.multiply]
        divide_terms = [NormalizedExpr.coerce_mul_term(t) for t in self.divide]
        if (self.scalar_func or "").lower() == "concat":
            self.multiply = multiply_terms
            self.divide = divide_terms
        else:
            self.multiply = sorted(multiply_terms, key=lambda e: e.signature_key)
            self.divide = sorted(divide_terms, key=lambda e: e.signature_key)
        if self.agg_func:
            self.agg_func = self.agg_func.lower()
        if self.scalar_func:
            self.scalar_func = self.scalar_func.lower()
        if self.inner_scalar_func:
            self.inner_scalar_func = self.inner_scalar_func.lower()
        if self.scalar_func and self.inner_scalar_func:
            if self.scalar_func == "extract":
                pass
            elif self.inner_scalar_func == "extract":
                self.scalar_func, self.inner_scalar_func = (self.inner_scalar_func, self.scalar_func)
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
                self.sarg_param_keys, self.isarg_param_keys = (self.isarg_param_keys, self.sarg_param_keys)
            elif self.scalar_func > self.inner_scalar_func:
                self.scalar_func, self.inner_scalar_func = (self.inner_scalar_func, self.scalar_func)
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
                self.sarg_param_keys, self.isarg_param_keys = (self.isarg_param_keys, self.sarg_param_keys)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> MulGroup:
        """Create MulGroup from dictionary; multiply/divide entries may be dicts (new nested form) or strings (legacy column ref) — both are accepted on read, always serialized as dicts on write."""
        return MulGroup(
            coefficient=d.get("coefficient", 1.0),
            multiply=[NormalizedExpr.coerce_mul_term(t) for t in d.get("multiply", [])],
            divide=[NormalizedExpr.coerce_mul_term(t) for t in d.get("divide", [])],
            agg_func=d.get("agg_func"),
            scalar_func=d.get("scalar_func"),
            inner_scalar_func=d.get("inner_scalar_func"),
            scalar_func_args=d.get("scalar_func_args", []),
            inner_scalar_func_args=d.get("inner_scalar_func_args", []),
            coeff_param_key=d.get("coeff_param_key", ""),
            sarg_param_keys=d.get("sarg_param_keys", []),
            isarg_param_keys=d.get("isarg_param_keys", []),
            distinct=bool(d.get("distinct", False)),
            agg_sep_param_key=str(d.get("agg_sep_param_key", "") or ""),
            agg_order_by=[
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else OrderByCol(expr=NormalizedExpr.parse_string_for_json(o))
                for o in d.get("agg_order_by", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all `MulGroup` fields; multiply/divide are nested dicts."""
        out = {
            "coefficient": self.coefficient,
            "multiply": [m.to_dict() for m in self.multiply],
            "divide": [d.to_dict() for d in self.divide],
            "agg_func": self.agg_func,
            "scalar_func": self.scalar_func,
            "inner_scalar_func": self.inner_scalar_func,
            "scalar_func_args": self.scalar_func_args,
            "inner_scalar_func_args": self.inner_scalar_func_args,
            "coeff_param_key": self.coeff_param_key,
            "sarg_param_keys": self.sarg_param_keys,
            "isarg_param_keys": self.isarg_param_keys,
        }
        if self.distinct:
            out["distinct"] = True
        if self.agg_sep_param_key:
            out["agg_sep_param_key"] = self.agg_sep_param_key
        if self.agg_order_by:
            out["agg_order_by"] = [o.to_dict() for o in self.agg_order_by]
        return out

    @property
    def signature_key(self) -> str:
        """Pipe-separated structural key (recurses through nested multiply/divide)."""
        parts = ["coeff"]
        if self.distinct:
            parts.append("distinct")
        if self.agg_func:
            parts.append(f"agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"*{m.signature_key}" for m in self.multiply)
        parts.extend(f"/{d.signature_key}" for d in self.divide)
        return "|".join(parts)

    @property
    def structural_key(self) -> str:
        """Like `signature_key` but omits the coefficient marker."""
        parts: list[str] = []
        if self.distinct:
            parts.append("distinct")
        if self.agg_func:
            parts.append(f"agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"*{m.signature_key}" for m in self.multiply)
        parts.extend(f"/{d.signature_key}" for d in self.divide)
        return "|".join(parts)


@dataclass
class NormalizedExpr:
    """Canonical sum-of-products expression: scalar_func(agg_func(inner_scalar_func(sum of add_groups minus sub_groups plus add_values minus sub_values))) with scalar_func_args and inner_scalar_func_args. Structural leaf forms (mutually exclusive with add_groups/sub_groups when set): - column_ref: bare or qualified column reference (`"t.c"`). - star: True for the SQL `*` token. - cast_type: when set, this expression is `CAST(<inner> AS cast_type)` where `<inner>` is the single child reachable via add_groups[0].multiply[0]. - interval: `(magnitude, unit)` for SQL `INTERVAL '<n>' <unit>`. - keyword: bare SQL keyword like ``current_date``."""

    add_groups: list[MulGroup] = field(default_factory=list)
    sub_groups: list[MulGroup] = field(default_factory=list)
    add_values: list[ExprValue] = field(default_factory=list)
    sub_values: list[ExprValue] = field(default_factory=list)
    agg_func: str | None = None
    scalar_func: str | None = None
    inner_scalar_func: str | None = None
    scalar_func_args: list[ScalarArg] = field(default_factory=list)
    inner_scalar_func_args: list[ScalarArg] = field(default_factory=list)
    sarg_param_keys: list[str] = field(default_factory=list)
    isarg_param_keys: list[str] = field(default_factory=list)
    is_numeric: bool = True
    column_ref: str | None = None
    star: bool = False
    cast_type: str | None = None
    interval: tuple[float, str] | None = None
    keyword: str | None = None
    raw_sql: str | None = None
    string_literal: str = ""
    _parse_expr_string_fn: ClassVar[Any] = None
    _render_expr_sql_fn: ClassVar[Any] = None

    def __post_init__(self) -> None:
        """Sort child groups/values and normalise outer function name. casing/order. Returns: None."""
        self.add_groups = sorted(self.add_groups, key=lambda g: g.signature_key)
        self.sub_groups = sorted(self.sub_groups, key=lambda g: g.signature_key)
        self.add_values = sorted(self.add_values, key=ExprValue.sort_key)
        self.sub_values = sorted(self.sub_values, key=ExprValue.sort_key)
        if self.agg_func:
            self.agg_func = self.agg_func.lower()
        if self.scalar_func:
            self.scalar_func = self.scalar_func.lower()
        if self.inner_scalar_func:
            self.inner_scalar_func = self.inner_scalar_func.lower()
        if self.scalar_func and self.inner_scalar_func:
            if self.scalar_func == "extract":
                pass
            elif self.inner_scalar_func == "extract":
                self.scalar_func, self.inner_scalar_func = (self.inner_scalar_func, self.scalar_func)
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
                self.sarg_param_keys, self.isarg_param_keys = (self.isarg_param_keys, self.sarg_param_keys)
            elif self.scalar_func > self.inner_scalar_func:
                self.scalar_func, self.inner_scalar_func = (self.inner_scalar_func, self.scalar_func)
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
                self.sarg_param_keys, self.isarg_param_keys = (self.isarg_param_keys, self.sarg_param_keys)
        if self.column_ref is not None:
            self.column_ref = str(self.column_ref).strip() or None
        if self.keyword is not None:
            self.keyword = str(self.keyword).strip().lower() or None
        if self.cast_type is not None:
            self.cast_type = str(self.cast_type).strip() or None
        if self.interval is not None:
            mag, unit = self.interval
            self.interval = (float(mag), str(unit).strip())
        if self.string_literal is not None:
            self.string_literal = str(self.string_literal).strip()
        if self.string_literal:
            self.column_ref = None
            self.raw_sql = None
            self.star = False
            self.keyword = None
            self.cast_type = None
            self.interval = None
            self.add_groups = []
            self.sub_groups = []
            self.add_values = []
            self.sub_values = []
            self.agg_func = None
            self.scalar_func = None
            self.inner_scalar_func = None
            self.scalar_func_args = []
            self.inner_scalar_func_args = []
            self.sarg_param_keys = []
            self.isarg_param_keys = []

    @classmethod
    def register_parse_expr_string(cls, fn: Any) -> None:
        """Register the SQL expression string parser used for JSON round-trips."""
        cls._parse_expr_string_fn = fn

    @classmethod
    def register_render_expr_sql(cls, fn: Any) -> None:
        """Register the SQL expression renderer used for prompt shorthand."""
        cls._render_expr_sql_fn = fn

    @classmethod
    def parse_string_for_json(cls, s: str) -> NormalizedExpr:
        """Parse a JSON string field that contains a SQL expression into a ``NormalizedExpr``."""
        t = (s or "").strip()
        if not t:
            return cls()
        fn = cls._parse_expr_string_fn
        if fn is not None:
            return cast(NormalizedExpr, fn(t))
        return cls.from_column(t)

    @staticmethod
    def from_dict(d: Any) -> NormalizedExpr:
        """Create NormalizedExpr from a dictionary, a column-reference string, or ``None``."""
        if d is None:
            return NormalizedExpr()
        if isinstance(d, str):
            return NormalizedExpr.from_column(d.strip())
        if isinstance(d, dict):
            s_lit = d.get("string_literal")
            if isinstance(s_lit, str) and s_lit.strip():
                return NormalizedExpr(string_literal=s_lit.strip())
            lit_plain = d.get("literal")
            if isinstance(lit_plain, str) and lit_plain.strip():
                return NormalizedExpr(string_literal=lit_plain.strip())
        column_ref_raw = d.get("column_ref")
        if column_ref_raw is None:
            legacy_ref = d.get("registry_ref")
            if isinstance(legacy_ref, str) and legacy_ref.strip():
                column_ref_raw = legacy_ref.strip()
        iv_raw = d.get("interval")
        iv: tuple[float, str] | None = None
        if isinstance(iv_raw, list | tuple) and len(iv_raw) == 2:
            iv = (float(iv_raw[0]), str(iv_raw[1]))
        return NormalizedExpr(
            add_groups=[MulGroup.from_dict(g) for g in d.get("add_groups", [])],
            sub_groups=[MulGroup.from_dict(g) for g in d.get("sub_groups", [])],
            add_values=[ExprValue.from_dict(v) for v in d.get("add_values", [])],
            sub_values=[ExprValue.from_dict(v) for v in d.get("sub_values", [])],
            agg_func=d.get("agg_func"),
            scalar_func=d.get("scalar_func"),
            inner_scalar_func=d.get("inner_scalar_func"),
            scalar_func_args=d.get("scalar_func_args", []),
            inner_scalar_func_args=d.get("inner_scalar_func_args", []),
            sarg_param_keys=d.get("sarg_param_keys", []),
            isarg_param_keys=d.get("isarg_param_keys", []),
            is_numeric=d.get("is_numeric", True),
            column_ref=column_ref_raw,
            star=bool(d.get("star", False)),
            cast_type=d.get("cast_type"),
            interval=iv,
            keyword=d.get("keyword"),
            raw_sql=d.get("raw_sql"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        out: dict[str, Any] = {
            "add_groups": [g.to_dict() for g in self.add_groups],
            "sub_groups": [g.to_dict() for g in self.sub_groups],
            "add_values": [v.to_dict() for v in self.add_values],
            "sub_values": [v.to_dict() for v in self.sub_values],
            "agg_func": self.agg_func,
            "scalar_func": self.scalar_func,
            "inner_scalar_func": self.inner_scalar_func,
            "scalar_func_args": self.scalar_func_args,
            "inner_scalar_func_args": self.inner_scalar_func_args,
            "sarg_param_keys": self.sarg_param_keys,
            "isarg_param_keys": self.isarg_param_keys,
            "is_numeric": self.is_numeric,
        }
        if self.column_ref:
            out["column_ref"] = self.column_ref
        if self.star:
            out["star"] = True
        if self.cast_type:
            out["cast_type"] = self.cast_type
        if self.interval is not None:
            out["interval"] = [self.interval[0], self.interval[1]]
        if self.keyword:
            out["keyword"] = self.keyword
        if self.raw_sql:
            out["raw_sql"] = self.raw_sql
        if self.string_literal:
            out["string_literal"] = self.string_literal
        return out

    @staticmethod
    def from_column(col: str) -> NormalizedExpr:
        """Build a leaf NormalizedExpr that references a single column (or `*`)."""
        s = col.strip()
        if s == "*":
            return NormalizedExpr(star=True)
        return NormalizedExpr(column_ref=s)

    @staticmethod
    def from_agg(agg_func: str, col: str) -> NormalizedExpr:
        """Build a NormalizedExpr for `agg_func(column)` with the column as a leaf child."""
        leaf = NormalizedExpr.from_column(col)
        return NormalizedExpr(add_groups=[MulGroup(multiply=[leaf], agg_func=agg_func.lower())])

    @property
    def signature_key(self) -> str:
        """Pipe-separated key over outer funcs, structural leaf info, and signed groups/values."""
        parts: list[str] = []
        if self.column_ref:
            parts.append(f"col={self.column_ref}")
        if self.star:
            parts.append("star")
        if self.keyword:
            parts.append(f"kw={self.keyword}")
        if self.cast_type:
            parts.append(f"cast={self.cast_type}")
        if self.interval is not None:
            parts.append(f"iv={self.interval[0]}:{self.interval[1]}")
        if self.raw_sql:
            parts.append(f"raw={self.raw_sql}")
        if self.string_literal:
            parts.append(f"strlit={self.string_literal!r}")
        if self.agg_func:
            parts.append(f"expr_agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"expr_scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"expr_sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"expr_inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"expr_iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"+{g.signature_key}" for g in self.add_groups)
        parts.extend(f"-{g.signature_key}" for g in self.sub_groups)
        parts.extend(f"+{v.signature_key}" for v in self.add_values)
        parts.extend(f"-{v.signature_key}" for v in self.sub_values)
        return "|".join(parts)

    @property
    def has_column_reference(self) -> bool:
        """Return True when this expression references any column, aggregate, scalar, or registry entry."""
        if self.string_literal:
            return False
        if self.raw_sql:
            return True
        if self.column_ref or self.star or self.keyword or self.cast_type or self.interval is not None:
            if self.column_ref:
                return True
            if self.cast_type:
                return True
            if self.star or self.keyword:
                return True
            if self.interval is not None:
                return True
        if self.add_groups or self.sub_groups:
            return True
        if self.agg_func or self.scalar_func or self.inner_scalar_func:
            return True
        return False

    @property
    def is_literal_only(self) -> bool:
        """Return True when this expression is composed solely of numeric literals."""
        return not self.has_column_reference

    @property
    def has_aggregation(self) -> bool:
        """Whether any subterm uses SQL aggregation (outer or per-`MulGroup`)."""
        if self.agg_func:
            return True
        raw_sql = self.raw_sql
        if raw_sql and RAW_SQL_AGG_OR_WINDOW_RE.search(raw_sql):
            return True
        for group in self.add_groups + self.sub_groups:
            if group.agg_func:
                return True
            for term in group.multiply + group.divide:
                if term.has_aggregation:
                    return True
        return False

    @property
    def primary_column(self) -> str:
        """Innermost column name reached by drilling into the first multiplicative term. Strips DISTINCT, walks through cast/scalar wrappers, returns "" when no column."""
        if self.string_literal:
            return ""
        if self.column_ref:
            return self.column_ref
        if self.star:
            return "*"
        if self.keyword:
            return self.keyword
        if self.interval is not None:
            return "interval"
        if self.raw_sql:
            return ""
        if not self.add_groups or not self.add_groups[0].multiply:
            return ""
        first = self.add_groups[0].multiply[0]
        return first.primary_column

    @property
    def primary_term(self) -> str:
        """First multiply operand of the first `add_groups` entry rendered as a token string. Returns the leaf `column_ref` for a column reference, ``"*"`` for star, the upper-cased `keyword` for a keyword leaf, or empty when no add_groups exist or the leaf is a complex sub-tree (cast/coalesce/case/interval)."""
        if self.column_ref:
            return self.column_ref
        if self.star:
            return "*"
        if self.keyword:
            return self.keyword.upper()
        if not self.add_groups or not self.add_groups[0].multiply:
            return ""
        first = self.add_groups[0].multiply[0]
        if first.column_ref:
            return first.column_ref
        if first.star:
            return "*"
        if first.keyword:
            return first.keyword.upper()
        fn = NormalizedExpr._render_expr_sql_fn
        if fn is not None:
            try:
                return cast(str, fn(first))
            except Exception:
                return ""
        return ""

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": "SQL expression text using qualified columns from the schema.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Shorthand ``expr`` string for LLM-facing JSON."""
        return {"expr": self.prompt_sql()}

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Canonical ``expr`` field example for prompts."""
        return {"expr": "table.column"}

    @classmethod
    def from_stored_json(cls, raw: Any) -> NormalizedExpr:
        """Coerce JSON or template `expr` payloads into a `NormalizedExpr`."""
        if isinstance(raw, str):
            return cls.from_column(raw)
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        if isinstance(raw, cls):
            return raw
        return cls()

    @staticmethod
    def coerce_mul_term(raw: Any) -> NormalizedExpr:
        """Coerce a multiply/divide list element to a `NormalizedExpr` leaf."""
        if isinstance(raw, NormalizedExpr):
            return raw
        if isinstance(raw, dict):
            return NormalizedExpr.from_dict(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if s == "*":
                return NormalizedExpr(star=True)
            if s == "":
                return NormalizedExpr()
            if s.upper().startswith("DISTINCT "):
                s = s[9:].strip()
            while s.startswith("(") and s.endswith(")"):
                inner = s[1:-1].strip()
                if not inner:
                    break
                s = inner
            if "(" in s or " " in s:
                fn = NormalizedExpr._parse_expr_string_fn
                if fn is None:
                    return NormalizedExpr(raw_sql=s)
                try:
                    parsed = cast(NormalizedExpr, fn(s))
                    if (
                        parsed.add_groups
                        and len(parsed.add_groups) == 1
                        and not parsed.sub_groups
                        and not parsed.add_values
                        and not parsed.sub_values
                    ):
                        g = parsed.add_groups[0]
                        if (
                            g.coefficient == 1.0
                            and not g.divide
                            and len(g.multiply) == 1
                            and not g.agg_func
                            and not g.scalar_func
                        ):
                            return g.multiply[0]
                    return parsed
                except Exception as exc:
                    raise ConfigError(f"expression parse failed: {exc}") from exc
            return NormalizedExpr(column_ref=s)
        return NormalizedExpr()

    def registry_ref(self) -> str | None:
        """Return the canonical registry id when this is a bare ``column_ref`` matching ``^[wc]\\d{2}$``."""
        if self.string_literal:
            return None
        col = (self.column_ref or "").strip()
        if not col or not REGISTRY_REF_TOKEN_RE.match(col):
            return None
        if self.add_groups or self.sub_groups or self.add_values or self.sub_values:
            return None
        if self.agg_func or self.scalar_func or self.inner_scalar_func:
            return None
        if self.star or self.cast_type or self.interval is not None:
            return None
        if self.keyword or self.raw_sql:
            return None
        return col

    def prompt_sql(self) -> str:
        """Render as the shorthand SQL string shown in LLM prompts."""
        ref = self.registry_ref()
        if ref:
            return ref
        if self.string_literal:
            return self.string_literal
        fn = NormalizedExpr._render_expr_sql_fn
        if fn is not None:
            try:
                return cast(str, fn(self))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        col = self.primary_column
        return col if col else ""


@dataclass(frozen=True)
class PredicateGroup:
    """Boolean composition tree for WHERE/HAVING predicates."""

    op: Literal["and", "or"] = "and"
    predicates: tuple[WhereParam | HavingParam, ...] = ()
    groups: tuple[PredicateGroup, ...] = ()
    _LEGACY_WHERE_GROUP_KEYS: ClassVar[tuple[str, ...]] = ("where_group",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", PredicateGroup._normalize_op(self.op))

    def is_empty(self) -> bool:
        return not self.predicates and not self.groups

    def depth(self) -> int:
        if not self.groups:
            return 1
        return 1 + max(child.depth() for child in self.groups)

    def leaves(self) -> list[WhereParam | HavingParam]:
        out: list[WhereParam | HavingParam] = []
        out.extend(self.predicates)
        for child in self.groups:
            out.extend(child.leaves())
        return out

    def leaf_count(self) -> int:
        return len(self.leaves())

    @staticmethod
    def _canonicalize_sides(predicate: Any) -> None:
        """Enforce column-bearing side on the left and flip the operator when a swap is required."""
        left = predicate.left_expr
        right = predicate.right_expr
        if right is None:
            return
        left_has_col = left.has_column_reference
        right_has_col = right.has_column_reference
        if left_has_col and not right_has_col:
            return
        if right_has_col and not left_has_col:
            predicate.left_expr, predicate.right_expr = right, left
            predicate.op = OP_FLIP.get(predicate.op, predicate.op)

    @staticmethod
    def _normalize_op(raw: Any) -> Literal["and", "or"]:
        text = str(raw or "and").strip().lower()
        return "or" if text == "or" else "and"

    @classmethod
    def _legacy_where_group_raw(cls, raw: Mapping[str, Any]) -> Any:
        for key in cls._LEGACY_WHERE_GROUP_KEYS:
            if key in raw:
                return raw[key]
        return None

    @staticmethod
    def _where_group_int_from_stored(raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, list | tuple):
            if not raw:
                return None
            raw = raw[0]
        if isinstance(raw, bool):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _legacy_bool_op_from_stored(raw: Any) -> str:
        text = str(raw or "AND").strip().upper()
        return "OR" if text == "OR" else "AND"

    @staticmethod
    def _clamp_negative_where_group(value: int | None) -> int | None:
        if value is not None and value < 0:
            return None
        return value

    @classmethod
    def _legacy_coerce_where_group_list(
        cls,
        items: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
    ) -> list[tuple[WhereParam | HavingParam, str, int | None]]:
        if not items:
            return []
        normalized: list[tuple[WhereParam | HavingParam, str, int | None]] = []
        for pred, bool_op, where_group in items:
            normalized.append(
                (pred, cls._legacy_bool_op_from_stored(bool_op), cls._clamp_negative_where_group(where_group))
            )
        any_grouped = any(fg is not None for _, _, fg in normalized)
        if not any_grouped:
            if len(normalized) <= 1:
                return normalized
            first_b = normalized[0][1]
            rest_all_or = all(normalized[j][1] == "OR" for j in range(1, len(normalized)))
            if first_b == "AND" and rest_all_or:
                return [(pred, "AND", gid) for gid, (pred, _, _) in enumerate(normalized, start=1)]
            return normalized
        max_gid = max((fg for _, _, fg in normalized if fg is not None), default=-1)
        next_gid = max_gid + 1
        out: list[tuple[WhereParam | HavingParam, str, int | None]] = []
        for pred, _, fg in normalized:
            if fg is None:
                out.append((pred, "AND", next_gid))
                next_gid += 1
            else:
                out.append((pred, "AND", fg))
        return out

    @classmethod
    def _combine(
        cls,
        left: PredicateGroup,
        right_pred: WhereParam | HavingParam,
        connector: Literal["and", "or"],
        right_group: PredicateGroup,
    ) -> PredicateGroup:
        if connector == "and" and left.op == "and" and not left.groups:
            if not right_group.groups and len(right_group.predicates) == 1:
                return PredicateGroup(op="and", predicates=left.predicates + (right_pred,))
        if connector == left.op and not left.groups and not right_group.groups:
            if connector == "and":
                return PredicateGroup(op="and", predicates=left.predicates + right_group.predicates)
            return PredicateGroup(op="or", groups=(left, right_group))
        return PredicateGroup(op=connector, groups=(left, right_group))

    @classmethod
    def _from_legacy_rows(
        cls,
        rows: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
    ) -> PredicateGroup | None:
        rows = cls._legacy_coerce_where_group_list(rows)
        if not rows:
            return None
        if any(fg is not None for _, _, fg in rows):
            ordered_ids: list[int] = []
            buckets: dict[int, list[WhereParam | HavingParam]] = {}
            for pred, _, fg in rows:
                gid = fg if fg is not None else 0
                if gid not in buckets:
                    ordered_ids.append(gid)
                    buckets[gid] = []
                buckets[gid].append(pred)
            and_groups = tuple(
                PredicateGroup(op="and", predicates=tuple(buckets[gid])) for gid in ordered_ids if buckets[gid]
            )
            if len(and_groups) == 1:
                return and_groups[0]
            return PredicateGroup(op="or", groups=and_groups)
        if len(rows) == 1:
            return PredicateGroup(op="and", predicates=(rows[0][0],))
        acc = PredicateGroup(op="and", predicates=(rows[0][0],))
        for idx in range(1, len(rows)):
            connector = cls._normalize_op(rows[idx - 1][1])
            right = PredicateGroup(op="and", predicates=(rows[idx][0],))
            acc = cls._combine(acc, rows[idx][0], connector, right)
        return acc

    @staticmethod
    def from_dict(d: dict[str, Any] | None, *, having: bool = False) -> PredicateGroup | None:
        if not d or not isinstance(d, dict):
            return None
        pred_cls = HavingParam if having else WhereParam
        preds = tuple(pred_cls.from_dict(item) if isinstance(item, dict) else item for item in d.get("predicates", []))
        groups = tuple(
            child for raw in d.get("groups", []) if (child := PredicateGroup.from_dict(raw, having=having)) is not None
        )
        group = PredicateGroup(op=PredicateGroup._normalize_op(d.get("op", "and")), predicates=preds, groups=groups)
        return None if group.is_empty() else group

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op}
        if self.predicates:
            payload["predicates"] = [pred.to_dict() if hasattr(pred, "to_dict") else pred for pred in self.predicates]
        if self.groups:
            payload["groups"] = [group.to_dict() for group in self.groups]
        return payload

    def to_prompt_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op}
        if self.predicates:
            payload["predicates"] = [pred.to_prompt_dict() for pred in self.predicates]
        if self.groups:
            payload["groups"] = [group.to_prompt_dict() for group in self.groups]
        return payload

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        return {
            "op": "and",
            "predicates": [WhereParam.prompt_example_dict()],
            "groups": [],
        }

    @classmethod
    def from_legacy_flat_where_list(cls, filters: list[WhereParam]) -> PredicateGroup | None:
        rows = [(fp, "AND", None) for fp in filters]
        return cls._from_legacy_rows(rows)

    @classmethod
    def from_legacy_having(cls, having: list[HavingParam]) -> PredicateGroup | None:
        rows = [(hp, "AND", None) for hp in having]
        return cls._from_legacy_rows(rows)

    @classmethod
    def from_legacy_flat_where_dicts(cls, fp_raw: list[Any]) -> PredicateGroup | None:
        rows: list[tuple[WhereParam, str, int | None]] = []
        for raw in fp_raw:
            if not isinstance(raw, dict):
                continue
            fp = WhereParam.from_dict(raw)
            rows.append(
                (
                    fp,
                    cls._legacy_bool_op_from_stored(raw.get("bool_op", "AND")),
                    cls._where_group_int_from_stored(cls._legacy_where_group_raw(raw)),
                )
            )
        return cls._from_legacy_rows(rows)

    @classmethod
    def from_legacy_having_dicts(cls, hp_raw: list[Any]) -> PredicateGroup | None:
        rows: list[tuple[HavingParam, str, int | None]] = []
        for raw in hp_raw:
            if not isinstance(raw, dict):
                continue
            hp = HavingParam.from_dict(raw)
            rows.append(
                (
                    hp,
                    cls._legacy_bool_op_from_stored(raw.get("bool_op", "AND")),
                    cls._where_group_int_from_stored(cls._legacy_where_group_raw(raw)),
                )
            )
        return cls._from_legacy_rows(rows)

    @classmethod
    def from_stored(cls, raw: Any, *, legacy_key: str, having: bool = False) -> PredicateGroup | None:
        if isinstance(raw, PredicateGroup):
            return None if raw.is_empty() else raw
        if isinstance(raw, dict):
            return cls.from_dict(raw, having=having)
        if isinstance(raw, list):
            if having:
                return cls.from_legacy_having_dicts(raw)
            return cls.from_legacy_flat_where_dicts(raw)
        return None

    @classmethod
    def _cnf_or_clauses(cls, group: PredicateGroup) -> list[PredicateGroup]:
        clauses: list[PredicateGroup] = []
        for pred in group.predicates:
            clauses.append(PredicateGroup(op="or", predicates=(pred,)))
        for child in group.groups:
            if child.op == "or":
                clauses.append(child)
            elif child.op == "and":
                clauses.extend(cls._cnf_or_clauses(child))
        return clauses

    @classmethod
    def _cnf_normalize_and_disjunct(cls, group: PredicateGroup) -> PredicateGroup:
        and_preds: list[WhereParam | HavingParam] = list(group.predicates)
        nested: list[PredicateGroup] = []
        for child in group.groups:
            normalized = cls.normalize_cnf(child)
            if normalized is None or normalized.is_empty():
                continue
            if normalized.op == "or":
                nested.append(normalized)
            else:
                and_preds.extend(normalized.predicates)
                nested.extend(normalized.groups)
        return PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))

    @staticmethod
    def _normalization_preference(group: PredicateGroup) -> Literal["dnf", "cnf", "auto"]:
        if group.op == "and" and any(child.op == "or" for child in group.groups):
            return "cnf"
        if group.op == "or" and any(child.op == "and" for child in group.groups):
            return "dnf"
        return "auto"

    @classmethod
    def normalize_dnf(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Normalize a predicate tree toward OR-of-AND (DNF) form when possible."""
        if group is None or group.is_empty():
            return None
        if group.op == "or":
            groups = tuple(cls.normalize_dnf(child) or child for child in group.groups)
            preds = group.predicates
            if preds and not groups:
                return PredicateGroup(op="or", predicates=preds)
            if not preds and groups:
                flat_groups = [g for g in groups if g is not None and not g.is_empty()]
                if len(flat_groups) == 1:
                    return flat_groups[0]
                return PredicateGroup(op="or", groups=tuple(flat_groups))
            and_groups = list(groups)
            if preds:
                and_groups.insert(0, PredicateGroup(op="and", predicates=preds))
            return PredicateGroup(op="or", groups=tuple(and_groups))
        if group.op == "and":
            and_preds: list[WhereParam | HavingParam] = list(group.predicates)
            nested: list[PredicateGroup] = []
            for child in group.groups:
                normalized = cls.normalize_dnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                if normalized.op == "or":
                    nested.append(normalized)
                else:
                    and_preds.extend(normalized.predicates)
                    nested.extend(normalized.groups)
            return PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))
        raise ValueError(f"unsupported predicate op: {group.op!r}")

    @classmethod
    def normalize_cnf(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Normalize a predicate tree toward AND-of-OR (CNF) form when possible."""
        if group is None or group.is_empty():
            return None
        if group.op == "and":
            or_clauses: list[PredicateGroup] = []
            for pred in group.predicates:
                or_clauses.append(PredicateGroup(op="or", predicates=(pred,)))
            for child in group.groups:
                normalized = cls.normalize_cnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                if normalized.op == "or":
                    or_clauses.append(normalized)
                else:
                    or_clauses.extend(cls._cnf_or_clauses(normalized))
            if not or_clauses:
                return None
            if len(or_clauses) == 1:
                return or_clauses[0]
            return PredicateGroup(op="and", groups=tuple(or_clauses))
        if group.op == "or":
            or_preds: list[WhereParam | HavingParam] = list(group.predicates)
            nested_or: list[PredicateGroup] = []
            for child in group.groups:
                if child.is_empty():
                    continue
                if child.op == "and":
                    nested_or.append(cls._cnf_normalize_and_disjunct(child))
                    continue
                normalized = cls.normalize_cnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                or_preds.extend(normalized.predicates)
                nested_or.extend(normalized.groups)
            result = PredicateGroup(op="or", predicates=tuple(or_preds), groups=tuple(nested_or))
            return None if result.is_empty() else result
        raise ValueError(f"unsupported predicate op: {group.op!r}")

    @classmethod
    def coerce(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        if group is None or group.is_empty():
            return None
        dnf = cls.normalize_dnf(group)
        cnf = cls.normalize_cnf(group)
        candidates: list[tuple[Literal["dnf", "cnf"], PredicateGroup]] = []
        if dnf is not None and not dnf.is_empty():
            candidates.append(("dnf", dnf))
        if cnf is not None and not cnf.is_empty():
            candidates.append(("cnf", cnf))
        if not candidates:
            return None
        within_limit = [
            (form, candidate) for form, candidate in candidates if candidate.depth() <= MAX_PREDICATE_NESTING_DEPTH
        ]
        if not within_limit:
            raise ValueError(f"predicate nesting exceeds MAX_PREDICATE_NESTING_DEPTH={MAX_PREDICATE_NESTING_DEPTH}")
        preference = cls._normalization_preference(group)
        preferred = [candidate for form, candidate in within_limit if form == preference]
        if preferred:
            return min(preferred, key=lambda candidate: candidate.depth())
        return min((candidate for _, candidate in within_limit), key=lambda candidate: candidate.depth())

    @classmethod
    def map(cls, group: PredicateGroup | None, fn: Any) -> PredicateGroup | None:
        if group is None or group.is_empty():
            return None
        preds = tuple(fn(pred) for pred in group.predicates)
        subgroups = tuple(mapped for mapped in (cls.map(child, fn) for child in group.groups) if mapped is not None)
        mapped_group = PredicateGroup(op=group.op, predicates=preds, groups=subgroups)
        return None if mapped_group.is_empty() else mapped_group

    @classmethod
    def reapply_leaves(
        cls, group: PredicateGroup | None, leaves: Sequence[WhereParam | HavingParam]
    ) -> PredicateGroup | None:
        """Replace predicate leaves in *group* with *leaves* in traversal order."""
        if group is None or group.is_empty():
            return None
        leaf_iter = iter(leaves)
        return cls.map(group, lambda _pred: next(leaf_iter))

    @classmethod
    def rebuild_from_leaves(
        cls, original: PredicateGroup | None, leaves: Sequence[WhereParam | HavingParam]
    ) -> PredicateGroup | None:
        """Preserve *original* tree shape when leaf counts match; otherwise flatten to AND."""
        if not leaves:
            return None
        if original is not None and len(leaves) == len(original.leaves()):
            return cls.reapply_leaves(original, leaves)
        return cls.from_list(list(leaves))

    @classmethod
    def partition(
        cls, group: PredicateGroup | None, keep_fn: Any
    ) -> tuple[PredicateGroup | None, PredicateGroup | None]:
        if group is None or group.is_empty():
            return None, None
        kept_preds: list[WhereParam | HavingParam] = []
        dropped_preds: list[WhereParam | HavingParam] = []
        for pred in group.predicates:
            (kept_preds if keep_fn(pred) else dropped_preds).append(pred)
        kept_groups: list[PredicateGroup] = []
        dropped_groups: list[PredicateGroup] = []
        for child in group.groups:
            kept_child, dropped_child = cls.partition(child, keep_fn)
            if kept_child is not None:
                kept_groups.append(kept_child)
            if dropped_child is not None:
                dropped_groups.append(dropped_child)
        kept = PredicateGroup(op=group.op, predicates=tuple(kept_preds), groups=tuple(kept_groups))
        dropped = PredicateGroup(op=group.op, predicates=tuple(dropped_preds), groups=tuple(dropped_groups))
        return (None if kept.is_empty() else kept, None if dropped.is_empty() else dropped)

    @staticmethod
    def merge(op: Literal["and", "or"], groups: Sequence[PredicateGroup | None]) -> PredicateGroup | None:
        nonempty = [group for group in groups if group is not None and not group.is_empty()]
        if not nonempty:
            return None
        if len(nonempty) == 1:
            return nonempty[0]
        preds: list[WhereParam | HavingParam] = []
        nested: list[PredicateGroup] = []
        for group in nonempty:
            if not group.groups and group.op == op and group.predicates:
                preds.extend(group.predicates)
            else:
                nested.append(group)
        return PredicateGroup(op=op, predicates=tuple(preds), groups=tuple(nested))

    @classmethod
    def parse_where_field(cls, d: Mapping[str, Any]) -> PredicateGroup | None:
        if "where" in d:
            raw = d.get("where")
            if raw is None:
                return None
            return cls.from_dict(raw) if isinstance(raw, dict) else None
        if "where_param" in d:
            return cls.from_legacy_flat_where_dicts(d.get("where_param", []))
        return None

    @classmethod
    def parse_having_field(cls, d: Mapping[str, Any]) -> PredicateGroup | None:
        if "having" in d:
            raw = d.get("having")
            if raw is None:
                return None
            return cls.from_dict(raw, having=True) if isinstance(raw, dict) else None
        if "having_param" in d:
            return cls.from_legacy_having_dicts(d.get("having_param", []))
        return None

    @staticmethod
    def predicate_leaves(group: PredicateGroup | None) -> list[WhereParam | HavingParam]:
        return group.leaves() if group else []

    @staticmethod
    def where_leaves(group: PredicateGroup | None) -> list[WhereParam]:
        """Return flat WHERE predicate leaves with a narrowed type."""
        if group is None:
            return []
        return [leaf for leaf in group.leaves() if isinstance(leaf, WhereParam)]

    @staticmethod
    def having_leaves(group: PredicateGroup | None) -> list[HavingParam]:
        """Return flat HAVING predicate leaves with a narrowed type."""
        if group is None:
            return []
        return [leaf for leaf in group.leaves() if isinstance(leaf, HavingParam)]

    @staticmethod
    def group_and(*preds: WhereParam | HavingParam) -> PredicateGroup | None:
        items = tuple(pred for pred in preds if pred is not None)
        if not items:
            return None
        return PredicateGroup(op="and", predicates=items)

    @staticmethod
    def from_list(items: Sequence[WhereParam | HavingParam] | None) -> PredicateGroup | None:
        if not items:
            return None
        return PredicateGroup(op="and", predicates=tuple(items))

    @classmethod
    def coerce_having_group_list(cls, having: list[HavingParam]) -> list[HavingParam]:
        """Deprecated: convert a legacy flat HAVING list into a predicate group and back."""
        group = cls.from_legacy_having(having)
        return cls.having_leaves(group)


@dataclass
class OrderByCol:
    """Order by column with expression and sort direction."""

    expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    direction: str = "ASC"
    nulls: OrderByNullPlacement | None = None

    def __post_init__(self) -> None:
        """
        Strip and upper-case `direction` (e.g. `ASC` / `DESC`).

        Returns:

            None.
        """
        self.direction = self.direction.strip().upper()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OrderByCol:
        """
        Create OrderByCol from dictionary.

        Args:

            d: Dictionary with 'expr' and 'direction' keys.

        Returns:

            Populated OrderByCol instance.
        """
        expr_raw = d.get("expr", {})
        if isinstance(expr_raw, str):
            expr = NormalizedExpr.parse_string_for_json(expr_raw)
        elif isinstance(expr_raw, dict):
            expr = NormalizedExpr.from_dict(expr_raw)
        elif isinstance(expr_raw, NormalizedExpr):
            expr = expr_raw
        else:
            expr = NormalizedExpr()
        return OrderByCol(
            expr=expr,
            direction=d.get("direction", "ASC"),
            nulls=OrderByNullPlacement.coerce(d.get("nulls")),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with the serialized expr and direction string.
        """
        out: dict[str, Any] = {"expr": self.expr.to_dict(), "direction": self.direction}
        if self.nulls is not None:
            out["nulls"] = self.nulls
        return out

    @property
    def is_aggregated(self) -> bool:
        """
        Whether the order key expression carries an aggregate.

        Returns:

            Same as `self.expr.has_aggregation`.
        """
        return self.expr.has_aggregation

    @property
    def signature_key(self) -> str:
        """
        Expr signature plus sort direction.

        Returns:

            `expr_key|DIRECTION` string.
        """
        return "|".join([self.expr.signature_key, self.direction])

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": "ORDER BY key as a SQL expression string.",
        "direction": "Sort direction asc or desc in lowercase in prompts.",
        "nulls": "Optional null placement first or last when the question requires explicit null ordering.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """ORDER BY entry shorthand."""
        out: dict[str, Any] = {
            "expr": self.expr.prompt_sql(),
            "direction": self.direction.lower(),
        }
        if self.nulls is not None:
            out["nulls"] = self.nulls
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example order_by_cols row."""
        return {"expr": "table.column", "direction": "asc", "nulls": "last"}


@dataclass
class WhereParam:
    """WHERE filter condition with left expression, operator, and optional right expression for expr-vs-expr comparisons."""

    left_expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    op: str = "="
    right_expr: NormalizedExpr | None = None
    value_type: str = "string"
    param_key: str | None = ""
    param_key_hi: str = ""
    param_key_unit: str = ""
    raw_value: RawValue = None

    def __post_init__(self) -> None:
        """
        Normalise operators/types, canonicalise expr-vs-expr sides, merge literals to the value side.

        Returns:

            None.
        """
        self.op = self.op.strip().lower()
        self.value_type = self.value_type.strip().lower()
        if self.right_expr is not None:
            PredicateGroup._canonicalize_sides(self)
            for ev in self.left_expr.add_values:
                self.right_expr.sub_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            for ev in self.left_expr.sub_values:
                self.right_expr.add_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            self.left_expr.add_values = []
            self.left_expr.sub_values = []
        elif (
            self.raw_value is not None
            and isinstance(self.raw_value, int | float)
            and not isinstance(self.raw_value, bool)
        ):
            offset = sum(float(cast(Any, ev.value) or 0) for ev in self.left_expr.add_values) - sum(
                float(cast(Any, ev.value) or 0) for ev in self.left_expr.sub_values
            )
            self.raw_value = self.raw_value - offset
            self.left_expr.add_values = []
            self.left_expr.sub_values = []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> WhereParam:
        """
        Create WhereParam from dictionary.

        Args:

            d: Dictionary with 'left_expr', 'op', optional 'right_expr', 'value_type', and 'param_key'.

        Returns:

            Populated WhereParam instance.
        """
        left_raw = d.get("left_expr", {})
        right_raw = d.get("right_expr")
        return WhereParam(
            left_expr=NormalizedExpr.from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(NormalizedExpr.from_stored_json(right_raw) if right_raw else None),
            value_type=d.get("value_type", "string"),
            param_key=d.get("param_key", ""),
            param_key_hi=d.get("param_key_hi", ""),
            param_key_unit=d.get("param_key_unit", ""),
            raw_value=d.get("value") or d.get("raw_value"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with filter fields; raw_value is intentionally excluded.
        """
        d: dict[str, Any] = {
            "left_expr": self.left_expr.to_dict(),
            "op": self.op,
            "right_expr": self.right_expr.to_dict() if self.right_expr else None,
            "value_type": self.value_type,
            "param_key": self.param_key,
        }
        if self.param_key_hi:
            d["param_key_hi"] = self.param_key_hi
        if self.param_key_unit:
            d["param_key_unit"] = self.param_key_unit
        return d

    @property
    def signature_key(self) -> str:
        """
        Structural key for WHERE-style template matching.

        Returns:

            Left expr, op, value type, and optional right expr signature joined by `|`.
        """
        parts = [self.left_expr.signature_key, self.op, self.value_type]
        if self.right_expr:
            parts.append(f"r:{self.right_expr.signature_key}")
        return "|".join(parts)

    def resolved_value(self, param_values: Mapping[str, Any] | None) -> Any:
        """Resolve the filter literal from inline storage or bound. parameters. After post-processing, ``raw_value`` may be cleared while ``param_key`` still identifies the bound slot in the owning body ``param_values`` map. Args: param_values: Bound parameter map; treated as empty when ``None``. Returns: ``raw_value`` when set; otherwise ``param_values[param_key]`` when ``param_key`` is non-empty; otherwise ``None``."""
        if self.raw_value is not None:
            return self.raw_value
        store = param_values or {}
        pk = (self.param_key or "").strip()
        pku = (self.param_key_unit or "").strip()
        vt = (self.value_type or "").lower()
        if vt == "date_diff" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                u = unit if isinstance(unit, str) and unit else "day"
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                return {"unit": u, "amount": a}
        if vt == "date_window" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                u = unit if isinstance(unit, str) and unit else "day"
                return {"unit": u, "amount": a}
        if not pk:
            return None
        return store.get(pk)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "left_expr": "SQL expression for the predicate left side using qualified columns.",
        "op": "Comparison or membership operator (lowercase).",
        "right_expr": (
            "Optional SQL expression for expr-vs-expr predicates; may reference a different table from "
            "left_expr to express a value comparison rather than a join relationship."
        ),
        "value_type": "Semantic type for expr-vs-value predicates.",
        "value": "Inline literal or structured date_window or date_diff payload.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """LLM shorthand dict with SQL strings for expression sides."""
        out: dict[str, Any] = {
            "left_expr": self.left_expr.prompt_sql(),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = self.right_expr.prompt_sql()
        elif self.raw_value is not None:
            out["value"] = self.raw_value
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example WHERE predicate shape for prompts."""
        return {
            "left_expr": "table.other_column",
            "op": "=",
            "value_type": "string",
            "value": "<literal>",
        }


@dataclass
class HavingParam:
    """Having condition with left expression, operator, and optional right expression for expr-vs-expr comparisons."""

    left_expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    op: str = "="
    right_expr: NormalizedExpr | None = None
    value_type: str = "number"
    param_key: str | None = ""
    param_key_unit: str = ""
    raw_value: RawValue = None

    def __post_init__(self) -> None:
        """
        Normalise operators/types, canonicalise expr-vs-expr sides, merge literals to the value side.

        Returns:

            None.
        """
        self.op = self.op.strip().lower()
        self.value_type = self.value_type.strip().lower()
        if self.right_expr is not None:
            PredicateGroup._canonicalize_sides(self)
            for ev in self.left_expr.add_values:
                self.right_expr.sub_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            for ev in self.left_expr.sub_values:
                self.right_expr.add_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            self.left_expr.add_values = []
            self.left_expr.sub_values = []
        elif (
            self.raw_value is not None
            and isinstance(self.raw_value, int | float)
            and not isinstance(self.raw_value, bool)
        ):
            offset = sum(float(cast(Any, ev.value) or 0) for ev in self.left_expr.add_values) - sum(
                float(cast(Any, ev.value) or 0) for ev in self.left_expr.sub_values
            )
            self.raw_value = self.raw_value - offset
            self.left_expr.add_values = []
            self.left_expr.sub_values = []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HavingParam:
        """
        Create HavingParam from dictionary.

        Args:

            d: Dictionary with 'left_expr', 'op', optional 'right_expr', 'value_type', and 'param_key'.

        Returns:

            Populated HavingParam instance.
        """
        left_raw = d.get("left_expr", {})
        right_raw = d.get("right_expr")
        return HavingParam(
            left_expr=NormalizedExpr.from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(NormalizedExpr.from_stored_json(right_raw) if right_raw else None),
            value_type=d.get("value_type", "number"),
            param_key=d.get("param_key", ""),
            param_key_unit=d.get("param_key_unit", ""),
            raw_value=d.get("value") or d.get("raw_value"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize HAVING parameters; omits `raw_value` (like. `WhereParam.to_dict`). Returns: Dict of exprs, op, types."""
        d: dict[str, Any] = {
            "left_expr": self.left_expr.to_dict(),
            "op": self.op,
            "right_expr": self.right_expr.to_dict() if self.right_expr else None,
            "value_type": self.value_type,
            "param_key": self.param_key,
        }
        if self.param_key_unit:
            d["param_key_unit"] = self.param_key_unit
        return d

    @property
    def signature_key(self) -> str:
        """
        Structural key for HAVING-style template matching.

        Returns:

            Same pipe-joined pattern as `WhereParam.signature_key`.
        """
        parts = [self.left_expr.signature_key, self.op, self.value_type]
        if self.right_expr:
            parts.append(f"r:{self.right_expr.signature_key}")
        return "|".join(parts)

    def resolved_value(self, param_values: Mapping[str, Any] | None) -> Any:
        """Resolve the HAVING literal from inline storage or bound. parameters. After post-processing, ``raw_value`` may be cleared while ``param_key`` still identifies the bound slot in the owning body ``param_values`` map. Args: param_values: Bound parameter map; treated as empty when ``None``. Returns: ``raw_value`` when set; otherwise ``param_values[param_key]`` when ``param_key`` is non-empty; otherwise ``None``."""
        if self.raw_value is not None:
            return self.raw_value
        store = param_values or {}
        pk = (self.param_key or "").strip()
        pku = (self.param_key_unit or "").strip()
        vt = (self.value_type or "").lower()
        if vt == "date_diff" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                u = unit if isinstance(unit, str) and unit else "day"
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                return {"unit": u, "amount": a}
        if vt == "date_window" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                u = unit if isinstance(unit, str) and unit else "day"
                return {"unit": u, "amount": a}
        if not pk:
            return None
        return store.get(pk)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "left_expr": "Aggregate or grouped SQL expression on the left side.",
        "op": "Comparison operator for aggregate predicates.",
        "right_expr": (
            "Optional SQL expression for agg-vs-agg predicates; may reference a different table from "
            "left_expr to express an aggregate comparison rather than a join relationship."
        ),
        "value_type": "Semantic type for agg-vs-value predicates.",
        "value": "Numeric or structured literal compared to the left aggregate.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """LLM shorthand dict with SQL strings for HAVING sides."""
        out: dict[str, Any] = {
            "left_expr": self.left_expr.prompt_sql(),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = self.right_expr.prompt_sql()
        elif self.raw_value is not None:
            out["value"] = self.raw_value
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example HAVING predicate shape for prompts."""
        return {
            "left_expr": "COUNT(table.column)",
            "op": ">",
            "value_type": "integer",
            "value": 1,
        }


PredicateLeaf = WhereParam | HavingParam


class JoinCandidateCapExceededError(AetherError):
    """Raised when join path cross-product enumeration exceeds the refusal cap."""

    def __init__(
        self,
        enumerated: int,
        cap: int,
        *,
        tables: list[str] | None = None,
        root: str | None = None,
    ) -> None:
        self.enumerated = enumerated
        self.cap = cap
        self.tables = list(tables) if tables is not None else None
        self.root = root
        tables_text = ",".join(self.tables) if self.tables else "?"
        root_text = f" root={self.root!r}" if self.root else ""
        super().__init__(
            f"join candidate cross-product cap exceeded: {enumerated} paths (limit {cap}) tables={tables_text}{root_text}"
        )


class JoinProbeEdgeKindMismatchError(AetherError):
    """Raised when join path signature and edge-kind lists are not aligned."""

    def __init__(self, signature_len: int, kinds_len: int) -> None:
        self.signature_len = signature_len
        self.kinds_len = kinds_len
        super().__init__(f"join path edge_kinds length mismatch: {kinds_len} kinds for {signature_len} segments")


class JoinPathKeyTypeError(AetherError):
    """Raised when a resolved join path pairs incompatible column types."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ArtifactLockTimeoutError(RuntimeError, RetryableError):
    """Raised when an artifact directory lock cannot be acquired before the timeout expires."""

    def __init__(
        self,
        artifacts_dir: str,
        holder_pid: int | None,
        *,
        timeout: float,
        lock_path: str | None = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.holder_pid = holder_pid
        self.timeout = timeout
        self.lock_path = lock_path
        pid_text = f" (held by pid {holder_pid})" if holder_pid is not None else ""
        super().__init__(
            f"Timed out waiting for artifact lock on {artifacts_dir!r} after {timeout:.1f}s{pid_text}",
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Typed view of ``artifact_manifest.json`` fields used by migration checks."""

    artifact_format_version: str = "0"
    created_with_package_version: str = ""
    min_compatible_package_version: str = ""
    last_action: str = ""
    last_action_at: str = ""
    structural_hash: str = ""
    profiling_hash: str = ""
    scope_hash: str = ""
    effective_structural_hash: str = ""
    schema_graph_id: str = ""
    notes_hash: str = ""
    semantic_edges_hash: str = ""
    last_migration_tier: str = ""
    last_migration_at: str = ""
