"""Shared dataclasses and enums for schema graphs, validation, templates, QSim skeletons, and type helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from itertools import product as iter_product
from typing import Any, ClassVar, Literal, Protocol, cast

from ._constants import (
    COLUMN_TYPE_TO_VALUE_TYPE,
    DATE_TYPE_TOKENS,
    DEFAULT_NULL_ORDERING_ASC,
    DEFAULT_NULL_ORDERING_DESC,
    DOMAIN_KNOWLEDGE_DEFAULT_KIND,
    FEDERATION_TIMEZONE_AWARE_DATA_TYPES,
    FIXED_WIDTH_TEXT_BASE_TYPES,
    MAX_FLOAT_SAFE_INTEGER,
    MAX_PREDICATE_DISTRIBUTE_LEAVES,
    MAX_PREDICATE_NESTING_DEPTH,
    MYSQL_TIMESTAMP_ENGINES,
    NUMERIC_TYPE_ARGUMENTS_RE,
    NUMERIC_TYPE_TOKENS,
    OP_FLIP,
    RAW_SQL_AGG_OR_WINDOW_RE,
    REGISTRY_REF_TOKEN_RE,
    STRING_TYPE_TOKENS,
    STRUCTURAL_DATA_TYPE_CANONICAL,
    STRUCTURAL_KNOWLEDGE_LEGACY_KINDS,
    STRUCTURAL_KNOWLEDGE_PAYLOAD_KEYS,
    UNSIGNED_INTEGER_TYPE_MAX,
)
from ._constants_runtime import CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC

_mock_fixture_recorded_corpus_count: Any = None


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


class ReflectMode(StrEnum):
    """Effective catalog reflection strategy for include / allow / deny scope."""

    ALLOW_LIST = "allow_list"
    BOTH_THEN_DENY = "both_then_deny"
    SINGLE_KIND = "single_kind"


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


class DomainKnowledgeKind(StrEnum):
    """Closed vocabulary for domain-knowledge entry kinds."""

    GLOSSARY = "glossary"
    POLICY = "policy"
    METRIC = "metric"
    SYNONYM = "synonym"
    CAVEAT = "caveat"


class StructuralKnowledgeKind(StrEnum):
    """Closed vocabulary for structural-knowledge fact kinds (not schema attach keys). Structural knowledge anchors to a relation, field, or small set of them; domain knowledge is unanchorable. Whether a record is structural or domain is decided per record at extraction by whether it anchors, not by kind."""

    RELATION = "relation"
    FIELD = "field"
    JOIN = "join"
    GRAIN = "grain"
    CARDINALITY = "cardinality"
    LIFECYCLE = "lifecycle"
    DECLARED_VALUE_SET = "declared_value_set"
    SENTINEL_SEMANTICS = "sentinel_semantics"
    UNIT_OF_MEASURE = "unit_of_measure"
    RELATION_SHAPE = "relation_shape"
    TERM_BINDING = "term_binding"
    PERIOD_CONVENTION = "period_convention"
    CONCEPT_ABSENCE = "concept_absence"


class KnowledgeMergeAuthority(StrEnum):
    """Who wins when two records share an identity but disagree."""

    MASTER_AUTHORITATIVE = "master_authoritative"
    PEER_EQUAL = "peer_equal"


class KnowledgeMergeDisposition(StrEnum):
    """Exactly one disposition per collision; no silent skip."""

    IDENTICAL = "identical"
    RECONCILABLE = "reconcilable"
    INCOMPATIBLE = "incompatible"


class ClaimVerificationOutcome(StrEnum):
    """Exactly one outcome per claim; no default branch."""

    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


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
    """Whether the sandbox LLM path uses offline fixtures or a network provider."""

    SANDBOX = "sandbox"
    MOCK = "mock"
    NETWORK = "network"

    @classmethod
    def coerce(cls, raw: Any) -> SandboxLlmMode:
        """Normalize a sandbox LLM mode label to a supported member."""
        if isinstance(raw, cls):
            if raw is cls.MOCK:
                return cls.SANDBOX
            return raw
        s = str(raw or "").strip()
        if s == "mock":
            return cls.SANDBOX
        for member in cls:
            if member.value == s:
                if member is cls.MOCK:
                    return cls.SANDBOX
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
    DOMAIN_JARGON = "domain_jargon"
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
    DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED = "CONFIG_FILE_VALUE_APPLIED"
    DIAGNOSTIC_CODE_COORDINATOR_LIMITS = "COORDINATOR_LIMITS"
    DIAGNOSTIC_CODE_DATA_QUALITY_ADVISORY = "DATA_QUALITY_ADVISORY"
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED = "DATA_QUALITY_AUTO_CORRECTED"
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ = "DATA_QUALITY_AUTO_READ"
    DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING = "DATA_QUALITY_BLOCKING"
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED = "DESCRIPTION_ENRICHMENT_FAILED"
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP = "DESCRIPTION_ENRICHMENT_NOOP"
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
    DIAGNOSTIC_CODE_STRUCTURE_NEEDS_RECONFIRMATION = "STRUCTURE_NEEDS_RECONFIRMATION"
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
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY = "REFUSAL_CONVERSATIONAL_DENY"
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE = "REFUSAL_INSUFFICIENT_KNOWLEDGE"
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION = "REFUSAL_INVALID_QUESTION"
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED = "REFUSAL_OPERATION_NOT_SUPPORTED"
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE = "REFUSAL_PARSE_FAILURE"
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION = "REFUSAL_UNMAPPABLE_QUESTION"
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
    DIAGNOSTIC_CODE_STRUCTURE_EDIT_SKIP = "STRUCTURE_EDIT_SKIP"
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
class EngineIdentity:
    """Bound engine type and runtime config for one engine instance or federated source."""

    engine_type: str
    runtime_config: Any


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
                "Ask a recorded question from docs/SANDBOX_DATA_REFERENCE.md. "
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
    """Init terminated because a migration map is required, malformed, or conflicts with validation."""

    skeleton_document: dict[str, Any] | None

    def __init__(self, message: str, *, skeleton_document: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.skeleton_document = skeleton_document


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
class AetherspaceDeleteResult:
    """Outcome of deleting one persisted aetherspace."""

    deleted: bool
    merge_counts: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.deleted


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """Outcome of :meth:`~aetherdialect.AetherEngine.refresh` or :meth:`~aetherdialect.AetherFederation.refresh`."""

    migration_tier: MigrationTier
    schema_changed: bool
    tables_added: tuple[str, ...]
    tables_removed: tuple[str, ...]
    columns_added: tuple[tuple[str, str], ...]
    columns_removed: tuple[tuple[str, str], ...]
    templates_invalidated: int
    orphans_removed: int
    bytes_reclaimed: int
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class SensitivityRatchetReport:
    """Summary of artifact scrubbing after a sensitivity increase."""

    domain_knowledge_dropped: int = 0
    structural_dropped: int = 0
    space_snapshots_updated: int = 0
    templates_dropped: int = 0
    feedback_rows_dropped: int = 0
    domain_knowledge_entries: tuple[DomainKnowledgeEntry, ...] | None = None


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
    """One JSON entry that was rejected during ``AetherEngine.apply_structure``."""

    path: str
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True)
class SidecarReconcileReport:
    """Rows removed from the persisted overrides sidecar during graph reconciliation."""

    pruned_paths: tuple[str, ...]
    wrote_disk: bool


@dataclass(frozen=True, slots=True)
class StructureReport:
    """Summary of edits produced by ``AetherEngine.apply_structure``."""

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
    domain_knowledge_refined: int = 0
    domain_knowledge_entries: tuple[DomainKnowledgeEntry, ...] | None = None
    sensitivity_increased_columns: frozenset[str] = frozenset()
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
class EngineContext:
    """Frozen scope object that narrows what a question may touch on a single :class:`~aetherdialect.AetherEngine` (one database connection). Parallel to :class:`FederationContext` (multi-member federation) and :class:`SpaceContext` (named subset at question time). Unlike those types, this context has no ``name`` field and is the only one that carries an optional ``sql_file``. Allow/deny lists, include mode, and content hashes of ``notes_file`` / ``sql_file`` feed the engine scope fingerprint. Args: allow_objects: Relation names permitted in scope. When non-empty, reflection resolves each name against the warehouse catalog without filtering by ``include`` (allow-list mode). Empty means unrestricted at the object level. include: Catalog kinds to reflect when both ``allow_objects`` and ``deny_objects`` are empty: ``tables`` or ``views`` (default ``tables``). Ignored when ``allow_objects`` is non-empty. When ``allow_objects`` is empty and ``deny_objects`` is non-empty, both kinds are reflected then denied names are removed. ``both`` is rejected. deny_objects: Relation names excluded from scope after reflection. When non-empty and ``allow_objects`` is empty, both tables and views are reflected first. Unknown deny names raise :class:`SchemaAccessError`. Must not overlap ``allow_objects``. deny_columns: Qualified ``table.column`` or ``*.column`` specs to exclude. Empty means none denied. Three-part ``source.table.column`` forms are accepted and normalised to ``table.column``. Must not name the same table as an ``allow_objects`` or ``deny_objects`` entry. allow_columns: Qualified ``table.column`` or ``*.column`` specs that further restrict visible columns. Empty means all columns of tables in scope (subject to denies). Three-part forms normalise like ``deny_columns``. notes_ file: Optional path to domain notes whose content hash enters the scope fingerprint. ``None`` (default) means no notes file. sql_ file: Optional path to DDL whose content hash enters the scope fingerprint and dialect probe. ``None`` (default) means no DDL file. :class:`FederationContext` deliberately omits this field. Raises: ConfigError: ``include`` is not one of ``tables`` / ``views`` / ``both``; ``notes_file`` or ``sql_file`` is present but blank; a column spec is not ``table.column`` / ``*.column`` / ``source.table.column``; ``allow_objects`` overlaps ``deny_objects``; or a table-specific ``deny_columns`` entry conflicts with ``allow_objects`` or ``deny_objects``. ValueError: An allow/deny identifier is empty after strip/lower."""

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
    """Frozen scope object that narrows what a question may touch on an :class:`~aetherdialect.AetherFederation` (several member engines). Parallel to :class:`EngineContext` (single connection) and :class:`SpaceContext` (named subset at question time). Unlike :class:`EngineContext`, this type has no ``sql_file`` field; unlike :class:`SpaceContext`, it is not a named subset. ``notes_file`` content hash feeds the composite scope identity. Mappings-aware validation (see :func:`~aetherdialect._federation_compose.validate_federation_context_against_mappings`) requires that denies of collapsed members use the **logical** table name, and rejects partial denies of ``union`` / ``replica`` member sets. Args: allow_objects: Relation names permitted in composite scope. Empty means unrestricted. Prefer logical names for collapsed mapped tables. include: Catalog kinds considered when building member graphs under this scope: ``tables`` or ``views`` (default ``tables``). ``both`` is rejected. deny_objects: Relation names excluded from composite scope. Empty means none denied. Must not overlap ``allow_objects``. For collapsed mapped tables, name the logical table, not a physical member table. deny_columns: Qualified ``table.column`` or ``*.column`` specs to exclude. Empty means none denied. Three-part ``source.table.column`` forms are accepted and normalised to ``table.column``. Must not name the same table as an ``allow_objects`` or ``deny_objects`` entry. allow_columns: Qualified ``table.column`` or ``*.column`` specs that further restrict visible columns. Empty means all columns of tables in scope (subject to denies). Three-part forms normalise like ``deny_columns``. notes_file: Optional path to domain notes whose content hash enters the composite scope fingerprint. ``None`` (default) means no notes file. Raises: ConfigError: ``include`` is not one of ``tables`` / ``views`` / ``both``; ``notes_file`` is present but blank; a column spec is not ``table.column`` / ``*.column`` / ``source.table.column``; ``allow_objects`` overlaps ``deny_objects``; or a table-specific ``deny_columns`` entry conflicts with ``allow_objects`` or ``deny_objects``. Also raised by mappings-aware validation when a scope entry names a collapsed physical member table, or partially denies a ``union`` / ``replica`` logical table. ValueError: An allow/deny identifier is empty after strip/lower."""

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
class KnowledgeScope:
    """Frozen in-scope entity set (table names plus qualified ``table.column`` names). Derivation, not removal: any caller that can only see part of a schema builds one of these and every downstream filter becomes a subset test against it — no prompt ever names what is outside it."""

    tables: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()

    def entity_set(self) -> frozenset[str]:
        """Return every in-scope identifier: table names plus qualified ``table.column`` names."""
        return self.tables | self.columns

    @staticmethod
    def _normalized(entities: Iterable[str]) -> frozenset[str]:
        return frozenset(str(e).strip().lower() for e in entities if str(e).strip())

    def contains(self, entity: str) -> bool:
        """Case-insensitive membership test against this scope's entity set."""
        want = str(entity).strip().lower()
        if not want:
            return False
        return want in KnowledgeScope._normalized(self.entity_set())

    def covers(self, entities: Iterable[str]) -> bool:
        """True when every member of *entities* is in this scope (subset test); vacuously true for an empty *entities*."""
        wanted = KnowledgeScope._normalized(entities)
        if not wanted:
            return True
        return wanted <= KnowledgeScope._normalized(self.entity_set())

    @staticmethod
    def from_schema_graph(schema_graph: Any) -> KnowledgeScope:
        """Build the unrestricted scope from every table/column on *schema_graph*."""
        return KnowledgeScope.from_visible_tables(schema_graph, None)

    @staticmethod
    def from_visible_tables(
        schema_graph: Any,
        visible_tables: Iterable[str] | None,
        *,
        exclude_sensitive: bool = True,
    ) -> KnowledgeScope:
        """Build a scope over *visible_tables* (``None`` means every table on *schema_graph*). Columns classified HIDDEN or RESTRICTED are left out of the resulting column set by default (the sensitivity-cleared subset); pass ``exclude_sensitive=False`` for a raw structural scope."""
        want = None if visible_tables is None else KnowledgeScope._normalized(visible_tables)
        tables: set[str] = set()
        columns: set[str] = set()
        for table_name, table in (getattr(schema_graph, "tables", None) or {}).items():
            if want is not None and str(table_name).strip().lower() not in want:
                continue
            tables.add(str(table_name))
            for column_name, column in (getattr(table, "columns", None) or {}).items():
                if exclude_sensitive and getattr(column, "sensitivity", None) in (
                    SensitivityClassification.HIDDEN,
                    SensitivityClassification.RESTRICTED,
                ):
                    continue
                columns.add(f"{table_name}.{column_name}")
        return KnowledgeScope(tables=frozenset(tables), columns=frozenset(columns))

    @staticmethod
    def from_engine_context(schema_graph: Any, visible_objects: frozenset[str] | None) -> KnowledgeScope:
        """Build a scope from a credential grant's ``visible_objects`` (bare table names; ``None`` means unrestricted)."""
        if visible_objects is None:
            return KnowledgeScope.from_schema_graph(schema_graph)
        table_names = frozenset(n for n in visible_objects if isinstance(n, str) and n.strip() and "." not in n)
        return KnowledgeScope.from_visible_tables(schema_graph, table_names)

    @staticmethod
    def from_space_context(schema_graph: Any, space_context: Any) -> KnowledgeScope:
        """Build a scope from a :class:`SpaceContext` allow-list (empty ``tables`` means unrestricted)."""
        tables = getattr(space_context, "tables", None)
        return KnowledgeScope.from_visible_tables(schema_graph, tables if tables else None)

    @staticmethod
    def from_space_snapshot(snapshot: Mapping[str, Any]) -> KnowledgeScope:
        """Build a scope from a persisted AetherSpace snapshot's ``tables``/``columns`` lists."""
        tables = frozenset(str(t) for t in (snapshot.get("tables") or ()) if str(t).strip())
        columns = frozenset(str(c) for c in (snapshot.get("columns") or ()) if "." in str(c))
        return KnowledgeScope(tables=tables, columns=columns)

    @staticmethod
    def union(scopes: Iterable[KnowledgeScope]) -> KnowledgeScope:
        """Combine multiple scopes into one (federation composite from member slices)."""
        tables: set[str] = set()
        columns: set[str] = set()
        for scope in scopes:
            tables |= set(scope.tables)
            columns |= set(scope.columns)
        return KnowledgeScope(tables=frozenset(tables), columns=frozenset(columns))


@dataclass(frozen=True, slots=True)
class DomainKnowledgeEntry:
    """Single prompt-time domain knowledge item not tied to schema column descriptions."""

    key: str
    text: str
    kind: str = "glossary"
    referenced_entities: frozenset[str] = frozenset()

    @staticmethod
    def normalize(entry: DomainKnowledgeEntry) -> DomainKnowledgeEntry:
        """Strip whitespace, default blank kind to glossary, refuse unknown kinds, and normalize referenced_entities."""
        key = str(entry.key).strip()
        text = str(entry.text).strip()
        kind_raw = str(entry.kind or DOMAIN_KNOWLEDGE_DEFAULT_KIND).strip() or DOMAIN_KNOWLEDGE_DEFAULT_KIND
        allowed = {member.value for member in DomainKnowledgeKind}
        if kind_raw not in allowed:
            raise ConfigError(f"unknown domain knowledge kind: {kind_raw!r}")
        kind = kind_raw
        if not key:
            raise ConfigError("domain knowledge entry key must be non-empty")
        if not text:
            raise ConfigError(f"domain knowledge entry {key!r} must have non-empty text")
        referenced = frozenset(str(e).strip() for e in entry.referenced_entities if str(e).strip())
        if kind != entry.kind or key != entry.key or text != entry.text or referenced != entry.referenced_entities:
            return DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced)
        return entry

    def in_scope(self, scope: KnowledgeScope) -> bool:
        """True when this entry's declared ``referenced_entities`` are all within *scope*. An empty reference set is a verified claim that the entry names no schema entity and is therefore safe in every scope."""
        return scope.covers(self.referenced_entities)

    @staticmethod
    def undeclared_schema_identifier_references(
        text: str,
        referenced_entities: frozenset[str],
        schema_graph: Any,
    ) -> list[str]:
        """Return schema identifiers appearing in *text* but absent from *referenced_entities*."""
        scope = KnowledgeScope.from_schema_graph(schema_graph)
        declared = KnowledgeScope._normalized(referenced_entities)
        expanded_declared = set(declared)
        for entity in declared:
            if "." in entity:
                expanded_declared.add(entity.split(".", 1)[0])
        forbidden = frozenset(
            entity for entity in scope.entity_set() if str(entity).strip().lower() not in expanded_declared
        )
        cleaned = str(text or "").strip()
        if not cleaned or not forbidden:
            return []
        hits: list[str] = []
        for token in sorted(forbidden, key=len, reverse=True):
            if not token:
                continue
            if re.search(rf"\b{re.escape(token)}\b", cleaned, flags=re.IGNORECASE):
                hits.append(token)
        return hits

    @staticmethod
    def hidden_qualified_names(schema_graph: Any) -> tuple[str, ...]:
        """Return sorted ``table.column`` names marked HIDDEN on *schema_graph*."""
        return tuple(
            qualified
            for qualified in DomainKnowledgeEntry.sensitive_qualified_names(schema_graph)
            if DomainKnowledgeEntry._column_sensitivity_at(schema_graph, qualified) == SensitivityClassification.HIDDEN
        )

    @staticmethod
    def sensitive_qualified_names(schema_graph: Any) -> tuple[str, ...]:
        """Return sorted ``table.column`` names marked HIDDEN or RESTRICTED on *schema_graph*."""
        names: list[str] = []
        tables = getattr(schema_graph, "tables", None) or {}
        for table_name, table in tables.items():
            columns = getattr(table, "columns", None) or {}
            for column_name, column in columns.items():
                sens = getattr(column, "sensitivity", None)
                if sens in (SensitivityClassification.HIDDEN, SensitivityClassification.RESTRICTED):
                    names.append(f"{table_name}.{column_name}")
        return tuple(sorted(names))

    @staticmethod
    def _column_sensitivity_at(schema_graph: Any, qualified: str) -> SensitivityClassification | None:
        if "." not in qualified:
            return None
        table_name, column_name = qualified.split(".", 1)
        tables = getattr(schema_graph, "tables", None) or {}
        table = tables.get(table_name)
        if table is None:
            return None
        columns = getattr(table, "columns", None) or {}
        column = columns.get(column_name)
        if column is None:
            return None
        return getattr(column, "sensitivity", None)

    @staticmethod
    def _unambiguous_sensitive_bare_names(schema_graph: Any) -> frozenset[str]:
        counts: dict[str, int] = {}
        for qualified in DomainKnowledgeEntry.sensitive_qualified_names(schema_graph):
            bare = qualified.split(".", 1)[1]
            counts[bare] = counts.get(bare, 0) + 1
        return frozenset(name for name, count in counts.items() if count == 1)

    @staticmethod
    def _qualified_token_appears(text: str, qualified: str) -> bool:
        """True when *qualified* appears in *text* with non-identifier boundaries."""
        hay = text.lower()
        needle = qualified.lower()
        start = 0
        while True:
            idx = hay.find(needle, start)
            if idx < 0:
                return False
            before_ok = idx == 0 or not (hay[idx - 1].isalnum() or hay[idx - 1] == "_")
            end = idx + len(needle)
            after_ok = end >= len(hay) or not (hay[end].isalnum() or hay[end] == "_")
            if before_ok and after_ok:
                return True
            start = idx + 1

    @staticmethod
    def sensitive_column_references(text: str, schema_graph: Any) -> list[str]:
        """Return schema-known sensitive ``table.column`` tokens referenced in *text*. Matches qualified ``table.column`` tokens and unambiguous bare column names (exactly one sensitive column on the schema shares that bare name)."""
        if not str(text or "").strip():
            return []
        found: set[str] = set()
        for qualified in DomainKnowledgeEntry.sensitive_qualified_names(schema_graph):
            if DomainKnowledgeEntry._qualified_token_appears(text, qualified):
                found.add(qualified)
        for bare in DomainKnowledgeEntry._unambiguous_sensitive_bare_names(schema_graph):
            if DomainKnowledgeEntry._qualified_token_appears(text, bare):
                for qualified in DomainKnowledgeEntry.sensitive_qualified_names(schema_graph):
                    if qualified.endswith(f".{bare}"):
                        found.add(qualified)
                        break
        return sorted(found)

    @staticmethod
    def hidden_column_references(text: str, schema_graph: Any) -> list[str]:
        """Return schema-known hidden ``table.column`` tokens that appear in *text*."""
        return [
            qualified
            for qualified in DomainKnowledgeEntry.sensitive_column_references(text, schema_graph)
            if DomainKnowledgeEntry._column_sensitivity_at(schema_graph, qualified) == SensitivityClassification.HIDDEN
        ]


@dataclass(frozen=True, slots=True)
class StructuralKnowledgeFact:
    """Semi-structured structural fact for description enrichment (not a schema attach key)."""

    kind: str
    text: str
    referenced_entities: frozenset[str] = frozenset()
    payload: dict[str, Any] | None = None

    @staticmethod
    def _normalize_payload(kind: str, raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Validate and normalize a kind-dispatched payload; raises on unknown kind or shape mismatch."""
        if kind in STRUCTURAL_KNOWLEDGE_LEGACY_KINDS:
            if raw is not None and raw:
                raise ConfigError(f"structural knowledge kind {kind!r} does not accept payload")
            return None
        expected_keys = STRUCTURAL_KNOWLEDGE_PAYLOAD_KEYS.get(kind)
        if expected_keys is None:
            raise ConfigError(f"unknown structural knowledge kind: {kind!r}")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"structural knowledge kind {kind!r} requires a payload object")
        actual_keys = frozenset(raw.keys())
        if actual_keys - expected_keys:
            raise ConfigError(
                f"structural knowledge kind {kind!r} payload keys must be a subset of {sorted(expected_keys)}; "
                f"got unexpected {sorted(actual_keys - expected_keys)}"
            )
        if kind == StructuralKnowledgeKind.DECLARED_VALUE_SET.value:
            values = raw.get("values")
            if not isinstance(values, list) or not values:
                raise ConfigError("declared_value_set payload values must be a non-empty list")
            normalized_values = [str(v).strip() for v in values]
            if not all(normalized_values):
                raise ConfigError("declared_value_set payload values must be non-empty strings")
            return {"values": normalized_values}
        if kind == StructuralKnowledgeKind.SENTINEL_SEMANTICS.value:
            sentinel_value = str(raw.get("sentinel_value") or "").strip()
            meaning = str(raw.get("meaning") or "").strip()
            if not sentinel_value or not meaning:
                raise ConfigError("sentinel_semantics payload sentinel_value and meaning must be non-empty")
            return {"sentinel_value": sentinel_value, "meaning": meaning}
        if kind == StructuralKnowledgeKind.UNIT_OF_MEASURE.value:
            unit = str(raw.get("unit") or "").strip()
            summable = raw.get("summable")
            if not unit:
                raise ConfigError("unit_of_measure payload unit must be non-empty")
            if not isinstance(summable, bool):
                raise ConfigError("unit_of_measure payload summable must be a boolean")
            return {"unit": unit, "summable": summable}
        if kind == StructuralKnowledgeKind.RELATION_SHAPE.value:
            shape = str(raw.get("shape") or "").strip()
            if not shape:
                raise ConfigError("relation_shape payload shape must be non-empty")
            return {"shape": shape}
        if kind == StructuralKnowledgeKind.TERM_BINDING.value:
            term = str(raw.get("term") or "").strip()
            binds_to = str(raw.get("binds_to") or "").strip()
            if not term or not binds_to:
                raise ConfigError("term_binding payload term and binds_to must be non-empty")
            return {"term": term, "binds_to": binds_to}
        if kind == StructuralKnowledgeKind.PERIOD_CONVENTION.value:
            boundary = str(raw.get("boundary") or "").strip()
            if not boundary:
                raise ConfigError("period_convention payload boundary must be non-empty")
            return {"boundary": boundary}
        if kind == StructuralKnowledgeKind.CONCEPT_ABSENCE.value:
            term = str(raw.get("term") or "").strip()
            if not term:
                raise ConfigError("concept_absence payload term must be non-empty")
            return {"term": term}
        if kind == StructuralKnowledgeKind.JOIN.value:
            if raw is None or not raw:
                return None
            negative = raw.get("negative")
            if negative is True:
                return {"negative": True}
            from_ref = str(raw.get("from") or "").strip()
            to_ref = str(raw.get("to") or "").strip()
            path_raw = raw.get("path_signature")
            path_signature: list[str] = []
            if isinstance(path_raw, list):
                path_signature = [str(s).strip() for s in path_raw if str(s).strip()]
            out: dict[str, Any] = {}
            if from_ref:
                out["from"] = from_ref
            if to_ref:
                out["to"] = to_ref
            if path_signature:
                out["path_signature"] = path_signature
            if not out:
                raise ConfigError("join payload requires negative, from/to endpoints, or path_signature")
            return out
        raise ConfigError(f"unknown structural knowledge kind: {kind!r}")

    @staticmethod
    def normalize(fact: StructuralKnowledgeFact) -> StructuralKnowledgeFact:
        """Strip whitespace, refuse unknown kinds or empty text, and validate referenced_entities and payload."""
        kind_raw = str(fact.kind or "").strip().lower()
        text = str(fact.text or "").strip()
        allowed = {member.value for member in StructuralKnowledgeKind}
        if kind_raw not in allowed:
            raise ConfigError(f"unknown structural knowledge kind: {kind_raw!r}")
        if not text:
            raise ConfigError("structural knowledge fact text must be non-empty")
        referenced = frozenset(str(e).strip() for e in fact.referenced_entities if str(e).strip())
        if not referenced:
            raise ConfigError("structural knowledge fact referenced_entities must be non-empty")
        if kind_raw == StructuralKnowledgeKind.JOIN.value and fact.payload is None:
            payload = None
        else:
            payload = StructuralKnowledgeFact._normalize_payload(kind_raw, fact.payload)
        normalized = StructuralKnowledgeFact(
            kind=kind_raw,
            text=text,
            referenced_entities=referenced,
            payload=payload,
        )
        if (
            kind_raw != fact.kind
            or text != fact.text
            or referenced != fact.referenced_entities
            or payload != fact.payload
        ):
            return normalized
        return fact

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk / LLM payload shape."""
        out: dict[str, Any] = {"kind": self.kind, "text": self.text}
        if self.referenced_entities:
            out["referenced_entities"] = sorted(self.referenced_entities)
        if self.payload is not None:
            out["payload"] = dict(self.payload)
        return out


@dataclass(frozen=True, slots=True)
class ClaimVerificationResult:
    """Per-claim profiling verdict."""

    fact: StructuralKnowledgeFact
    outcome: ClaimVerificationOutcome
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class NotesCoverageEntry:
    span: str
    disposition: str
    record_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"span": self.span, "disposition": self.disposition}
        if self.record_index is not None:
            out["record_index"] = self.record_index
        return out


@dataclass(frozen=True, slots=True)
class NotesExtractionLedger:
    entries: tuple[NotesCoverageEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}


@dataclass(frozen=True, slots=True)
class NotesExtractionResult:
    domain_knowledge: tuple[DomainKnowledgeEntry, ...]
    structural_knowledge: tuple[StructuralKnowledgeFact, ...]
    ledger: NotesExtractionLedger
    record_stream: tuple[tuple[str, DomainKnowledgeEntry | StructuralKnowledgeFact], ...] = ()


class FkAdmissionClassification(StrEnum):
    """Effect of admitting a proposed FK edge on the join graph."""

    ADDS_REACHABILITY = "adds-reachability"
    REDUCES_TIES = "reduces-ties"
    NEUTRAL = "neutral"
    INCREASES_TIES = "increases-ties"


@dataclass(frozen=True, slots=True)
class FkAdmissionReportEntry:
    """One admitted or rejected FK proposal with graph-effect classification."""

    from_ref: str
    to_ref: str
    classification: FkAdmissionClassification | None
    admitted: bool
    demoted_semantic: bool
    blocked_negative: bool
    reason: str
    override_kind: str = "structural"


@dataclass(frozen=True, slots=True)
class KnowledgeExtractionProposal:
    """Persisted versioned extraction proposal — source of truth for derived knowledge."""

    domain_knowledge: tuple[DomainKnowledgeEntry, ...]
    structural_knowledge: tuple[StructuralKnowledgeFact, ...]
    foreign_keys_add: tuple[dict[str, str], ...]
    coverage: dict[str, Any]
    notes_sha256: str | None = None
    scope_fingerprint: str | None = None
    extraction_diff: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeMergeStats:
    """Audit counts for every merge collision disposition."""

    identical: int = 0
    reconcilable: int = 0
    incompatible: int = 0
    provenance: Mapping[str, frozenset[str]] = field(default_factory=dict)


@dataclass
class DomainKnowledgeState:
    """Snapshot of active domain knowledge entries and digest."""

    entries: tuple[DomainKnowledgeEntry, ...] = ()
    digest: str = ""

    @staticmethod
    def digest_for(entries: Sequence[DomainKnowledgeEntry]) -> str:
        """Stable SHA-256 digest over normalized domain knowledge entries."""
        payload = [{"key": entry.key, "kind": entry.kind, "text": entry.text} for entry in entries]
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def empty_digest() -> str:
        """Return the digest for an empty knowledge set."""
        return DomainKnowledgeState.digest_for(())

    @staticmethod
    def validate_entries(
        entries: Sequence[DomainKnowledgeEntry],
        schema_graph: Any,
    ) -> tuple[DomainKnowledgeEntry, ...]:
        """Normalize entries and refuse sensitive-column references."""
        normalized: list[DomainKnowledgeEntry] = []
        seen_keys: set[str] = set()
        for raw in entries:
            if not isinstance(raw, DomainKnowledgeEntry):
                raise TypeError("domain knowledge entries must be DomainKnowledgeEntry instances")
            entry = DomainKnowledgeEntry.normalize(raw)
            if entry.key in seen_keys:
                raise ConfigError(f"duplicate domain knowledge key: {entry.key!r}")
            seen_keys.add(entry.key)
            sensitive_refs = DomainKnowledgeEntry.sensitive_column_references(entry.text, schema_graph)
            if sensitive_refs:
                joined = ", ".join(sorted(sensitive_refs))
                raise ConfigError(f"domain knowledge entry {entry.key!r} references sensitive column(s): {joined}")
            undeclared = DomainKnowledgeEntry.undeclared_schema_identifier_references(
                entry.text, entry.referenced_entities, schema_graph
            )
            if undeclared:
                raise ConfigError(
                    f"domain knowledge entry {entry.key!r} text names schema identifier(s) "
                    f"not declared in referenced_entities: {undeclared[0]!r}"
                )
            tables = getattr(schema_graph, "tables", None) or {}
            for ref in entry.referenced_entities:
                raw_ref = str(ref).strip()
                if not raw_ref:
                    continue
                if "." in raw_ref:
                    tname, cname = raw_ref.split(".", 1)
                    tbl = tables.get(tname)
                    if tbl is None:
                        raise ConfigError(
                            f"domain knowledge entry {entry.key!r} referenced_entities "
                            f"names unknown object: {raw_ref!r}"
                        )
                    col = tbl.columns.get(cname) if hasattr(tbl, "columns") else None
                    if col is None:
                        raise ConfigError(
                            f"domain knowledge entry {entry.key!r} referenced_entities "
                            f"names unknown object: {raw_ref!r}"
                        )
                    if getattr(col, "is_denied", False):
                        raise ConfigError(
                            f"domain knowledge entry {entry.key!r} referenced_entities names denied column: {raw_ref!r}"
                        )
                    sens = getattr(col, "sensitivity", None)
                    if sens in (SensitivityClassification.HIDDEN, SensitivityClassification.RESTRICTED):
                        raise ConfigError(
                            f"domain knowledge entry {entry.key!r} referenced_entities "
                            f"names hidden or restricted column: {raw_ref!r}"
                        )
                elif raw_ref not in tables:
                    raise ConfigError(
                        f"domain knowledge entry {entry.key!r} referenced_entities names unknown object: {raw_ref!r}"
                    )
            normalized.append(entry)
        return tuple(normalized)


class DomainKnowledgeHolder:
    """Mutable store for engine- or federation-level domain knowledge."""

    def __init__(self) -> None:
        self._state = DomainKnowledgeState(digest=DomainKnowledgeState.empty_digest())

    def set(self, entries: Sequence[DomainKnowledgeEntry], schema_graph: Any) -> None:
        normalized = DomainKnowledgeState.validate_entries(entries, schema_graph)
        digest = DomainKnowledgeState.digest_for(normalized)
        self._state = DomainKnowledgeState(entries=normalized, digest=digest)

    def entries(self) -> tuple[DomainKnowledgeEntry, ...]:
        return self._state.entries

    def digest(self) -> str:
        return self._state.digest

    def scope_kwargs(self) -> dict[str, Any]:
        """Keyword args for domain-knowledge scope binding."""
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

    @property
    def dialect(self) -> str: ...

    _connection: object | None
    _named_connection: object | None
    _context_name: object
    _schema_role: object
    _runtime_config: object | None
    _session_timezone: object | None
    _connection_mapping: Mapping[str, Any] | None


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
class AetherSpace:
    """
    Read-only descriptor for an aetherspace scope.

    Args:

        uid: Stable opaque identity (``master`` or ``S####``).
        name: Display label (may duplicate across spaces).
        tables: Table names in this space subset.
        columns: Qualified ``table.column`` names in this space subset.

    notes: Merged notes text baked from :attr:`SpaceContext.notes_file` at
        define time, or ``None`` when no notes file was supplied.
    """

    uid: str
    name: str
    tables: tuple[str, ...]
    columns: tuple[str, ...]
    notes: str | None = None


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
        """Create MulGroup from dictionary; multiply/divide entries may be dicts (nested form) or bare column-ref strings — both are accepted on read, always serialized as dicts on write."""
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
            stored_ref = d.get("registry_ref")
            if isinstance(stored_ref, str) and stored_ref.strip():
                column_ref_raw = stored_ref.strip()
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
    def from_column(col: object) -> NormalizedExpr:
        """Build a leaf NormalizedExpr that references a single column (or `*`)."""
        if not isinstance(col, str):
            return NormalizedExpr()
        s = col.strip()
        if not s:
            return NormalizedExpr()
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
            return cls.parse_string_for_json(raw)
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
        raise ConfigError(f"multiply/divide term must be an object or string, got {type(raw).__name__}")

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", PredicateGroup.normalize_op(self.op))

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
    def normalize_op(raw: Any) -> Literal["and", "or"]:
        text = str(raw or "and").strip().lower()
        return "or" if text == "or" else "and"

    @staticmethod
    def where_group_int_from_stored(raw: Any) -> int | None:
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
    def bool_op_from_stored(raw: Any) -> str:
        text = str(raw or "AND").strip().upper()
        return "OR" if text == "OR" else "AND"

    @staticmethod
    def _clamp_negative_where_group(value: int | None) -> int | None:
        if value is not None and value < 0:
            return None
        return value

    @classmethod
    def _coerce_where_group_list(
        cls,
        items: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
    ) -> list[tuple[WhereParam | HavingParam, str, int | None]]:
        if not items:
            return []
        normalized: list[tuple[WhereParam | HavingParam, str, int | None]] = []
        for pred, bool_op, where_group in items:
            normalized.append((pred, cls.bool_op_from_stored(bool_op), cls._clamp_negative_where_group(where_group)))
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
    def from_bool_op_rows(
        cls,
        rows: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
    ) -> PredicateGroup | None:
        rows = cls._coerce_where_group_list(rows)
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
            connector = cls.normalize_op(rows[idx - 1][1])
            right = PredicateGroup(op="and", predicates=(rows[idx][0],))
            acc = cls._combine(acc, rows[idx][0], connector, right)
        return acc

    @staticmethod
    def _item_looks_like_nested_group(item: Mapping[str, Any]) -> bool:
        """Return True when *item* is a nested boolean group rather than a leaf predicate."""
        if "left_expr" in item or "left_col" in item or "left_agg" in item:
            return False
        op = str(item.get("op") or "").strip().lower()
        if op not in ("and", "or"):
            return False
        return "predicates" in item or "groups" in item

    @staticmethod
    def from_dict(d: dict[str, Any] | None, *, having: bool = False) -> PredicateGroup | None:
        if not d or not isinstance(d, dict):
            return None
        pred_cls = HavingParam if having else WhereParam
        if not PredicateGroup._item_looks_like_nested_group(d) and (
            "left_expr" in d or "left_col" in d or "left_agg" in d
        ):
            leaf = pred_cls.from_dict(d)
            group = PredicateGroup(op="and", predicates=(leaf,), groups=())
            return None if group.is_empty() else group
        preds: list[WhereParam | HavingParam] = []
        nested: list[PredicateGroup] = []
        for item in d.get("predicates", []):
            if isinstance(item, dict) and PredicateGroup._item_looks_like_nested_group(item):
                child = PredicateGroup.from_dict(item, having=having)
                if child is not None:
                    nested.append(child)
                continue
            preds.append(pred_cls.from_dict(item) if isinstance(item, dict) else item)
        nested.extend(
            child for raw in d.get("groups", []) if (child := PredicateGroup.from_dict(raw, having=having)) is not None
        )
        group = PredicateGroup(
            op=PredicateGroup.normalize_op(d.get("op", "and")),
            predicates=tuple(preds),
            groups=tuple(nested),
        )
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
    def from_stored(cls, raw: Any, *, having: bool = False) -> PredicateGroup | None:
        if isinstance(raw, PredicateGroup):
            return None if raw.is_empty() else raw
        if isinstance(raw, dict):
            return cls.from_dict(raw, having=having)
        if isinstance(raw, list):
            raise ConfigError(
                "predicate groups must be nested objects; flat list / where_param shapes are not accepted"
            )
        return None

    @staticmethod
    def contradiction() -> PredicateGroup:
        """Return an always-false AND group used when a branch is unsatisfiable."""
        return PredicateGroup(
            op="and",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr(raw_sql="0"),
                    op="=",
                    raw_value=1,
                    value_type="integer",
                ),
            ),
        )

    def is_contradiction(self) -> bool:
        """Return True when this group is the canonical always-false sentinel."""
        if self.op != "and" or self.groups or len(self.predicates) != 1:
            return False
        leaf = self.predicates[0]
        return (
            leaf.op == "="
            and leaf.raw_value == 1
            and getattr(leaf.left_expr, "raw_sql", None) == "0"
            and leaf.right_expr is None
        )

    @staticmethod
    def leaf_identity_key(pred: WhereParam | HavingParam) -> str:
        """Value-aware identity for dedup and absorption (distinct from template signature_key)."""
        return pred.identity_key

    @classmethod
    def structural_key(cls, group: PredicateGroup | None) -> str:
        """Commutative structural fingerprint for subgroup idempotence and absorption."""
        if group is None or group.is_empty():
            return ""
        if group.is_contradiction():
            return "false"
        pred_keys = sorted(cls.leaf_identity_key(pred) for pred in group.predicates)
        child_keys = sorted(cls.structural_key(child) for child in group.groups)
        return f"{group.op}[{','.join(pred_keys)};{','.join(child_keys)}]"

    @classmethod
    def flatten(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Associatively flatten nested groups that share the parent connective."""
        if group is None or group.is_empty():
            return None
        preds: list[WhereParam | HavingParam] = list(group.predicates)
        nested: list[PredicateGroup] = []
        for child in group.groups:
            flat_child = cls.flatten(child)
            if flat_child is None or flat_child.is_empty():
                continue
            if flat_child.op == group.op:
                preds.extend(flat_child.predicates)
                nested.extend(flat_child.groups)
            else:
                nested.append(flat_child)
        result = PredicateGroup(op=group.op, predicates=tuple(preds), groups=tuple(nested))
        return None if result.is_empty() else result

    @classmethod
    def _dedupe_members(cls, group: PredicateGroup) -> PredicateGroup | None:
        """Drop idempotent duplicate leaves and duplicate child groups under *group.op*."""
        seen_preds: set[str] = set()
        kept_preds: list[WhereParam | HavingParam] = []
        for pred in group.predicates:
            key = cls.leaf_identity_key(pred)
            if key in seen_preds:
                continue
            seen_preds.add(key)
            kept_preds.append(pred)
        seen_groups: set[str] = set()
        kept_groups: list[PredicateGroup] = []
        for child in group.groups:
            key = cls.structural_key(child)
            if not key or key in seen_groups:
                continue
            seen_groups.add(key)
            kept_groups.append(child)
        result = PredicateGroup(op=group.op, predicates=tuple(kept_preds), groups=tuple(kept_groups))
        return None if result.is_empty() else result

    @classmethod
    def absorb(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Apply absorption: ``A ∨ (A ∧ B) → A`` and ``A ∧ (A ∨ B) → A``."""
        if group is None or group.is_empty():
            return None
        flat = cls.flatten(group)
        if flat is None or flat.is_empty():
            return None
        members: list[PredicateGroup] = []
        for pred in flat.predicates:
            members.append(PredicateGroup(op="and" if flat.op == "or" else "or", predicates=(pred,)))
        members.extend(flat.groups)
        if flat.op == "or":
            absorbed = cls._absorb_or_members(members)
        else:
            absorbed = cls._absorb_and_members(members)
        if not absorbed:
            return None
        preds: list[WhereParam | HavingParam] = []
        nested: list[PredicateGroup] = []
        for member in absorbed:
            if flat.op == "or":
                if member.op == "and":
                    nested.append(member)
                    continue
                if not member.groups and member.predicates:
                    nested.append(PredicateGroup(op="and", predicates=member.predicates))
                    continue
                nested.append(member)
                continue
            if flat.op == "and":
                if member.op == "or":
                    nested.append(member)
                    continue
                if not member.groups and member.predicates:
                    preds.extend(member.predicates)
                    continue
                nested.append(member)
                continue
        result = PredicateGroup(op=flat.op, predicates=tuple(preds), groups=tuple(nested))
        deduped = cls._dedupe_members(result)
        return None if deduped is None or deduped.is_empty() else deduped

    @classmethod
    def _member_leaf_key_set(cls, member: PredicateGroup) -> frozenset[str]:
        return frozenset(cls.leaf_identity_key(pred) for pred in member.leaves())

    @classmethod
    def _absorb_or_members(cls, members: list[PredicateGroup]) -> list[PredicateGroup]:
        """Under OR, drop conjunctions absorbed by a weaker (subset) disjunct."""
        key_sets = [cls._member_leaf_key_set(member) for member in members]
        keep = [True] * len(members)
        for i, keys_i in enumerate(key_sets):
            if not keep[i]:
                continue
            for j, keys_j in enumerate(key_sets):
                if i == j or not keep[j]:
                    continue
                if keys_i and keys_i < keys_j:
                    keep[j] = False
                elif keys_i and keys_i == keys_j and i < j:
                    keep[j] = False
        return [member for member, flagged in zip(members, keep, strict=True) if flagged]

    @classmethod
    def _absorb_and_members(cls, members: list[PredicateGroup]) -> list[PredicateGroup]:
        """Under AND, drop disjunctions absorbed by a stronger (subset) conjunct."""
        key_sets = [cls._member_leaf_key_set(member) for member in members]
        keep = [True] * len(members)
        for i, keys_i in enumerate(key_sets):
            if not keep[i]:
                continue
            for j, keys_j in enumerate(key_sets):
                if i == j or not keep[j]:
                    continue
                if keys_i and keys_i < keys_j:
                    keep[j] = False
                elif keys_i and keys_i == keys_j and i < j:
                    keep[j] = False
        return [member for member, flagged in zip(members, keep, strict=True) if flagged]

    @classmethod
    def _and_factors_for_dnf(cls, group: PredicateGroup) -> list[list[PredicateGroup]] | None:
        """Return Cartesian factors for AND-over-OR distribution, or None when capped."""
        factors: list[list[PredicateGroup]] = []
        for pred in group.predicates:
            factors.append([PredicateGroup(op="and", predicates=(pred,))])
        for child in group.groups:
            normalized = cls.normalize_dnf(child)
            if normalized is None or normalized.is_empty():
                continue
            if normalized.op == "or":
                alts: list[PredicateGroup] = []
                for pred in normalized.predicates:
                    alts.append(PredicateGroup(op="and", predicates=(pred,)))
                for nested in normalized.groups:
                    alts.append(nested if nested.op == "and" else PredicateGroup(op="and", groups=(nested,)))
                if not alts:
                    continue
                factors.append(alts)
            else:
                factors.append([normalized])
        product = 1
        for factor in factors:
            product *= max(len(factor), 1)
            if product > MAX_PREDICATE_DISTRIBUTE_LEAVES:
                return None
        return factors or [[PredicateGroup(op="and")]]

    @classmethod
    def _distribute_and_over_or(cls, group: PredicateGroup) -> PredicateGroup | None:
        """Expand ``A ∧ (B ∨ C)`` into ``(A ∧ B) ∨ (A ∧ C)`` when under the leaf-product cap."""
        factors = cls._and_factors_for_dnf(group)
        if factors is None:
            and_preds = list(group.predicates)
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
            result = PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))
            return None if result.is_empty() else result
        terms: list[PredicateGroup] = []
        for combo in iter_product(*factors):
            merged: PredicateGroup | None = None
            for part in combo:
                merged = part if merged is None else cls.merge("and", (merged, part))
            if merged is not None and not merged.is_empty():
                terms.append(merged if merged.op == "and" else PredicateGroup(op="and", groups=(merged,)))
        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return PredicateGroup(op="or", groups=tuple(terms))

    @classmethod
    def _or_factors_for_cnf(cls, group: PredicateGroup) -> list[list[PredicateGroup]] | None:
        """Return Cartesian factors for OR-over-AND distribution, or None when capped."""
        factors: list[list[PredicateGroup]] = []
        for pred in group.predicates:
            factors.append([PredicateGroup(op="or", predicates=(pred,))])
        for child in group.groups:
            normalized = cls.normalize_cnf(child)
            if normalized is None or normalized.is_empty():
                continue
            if normalized.op == "and":
                alts: list[PredicateGroup] = []
                for pred in normalized.predicates:
                    alts.append(PredicateGroup(op="or", predicates=(pred,)))
                for nested in normalized.groups:
                    alts.append(nested if nested.op == "or" else PredicateGroup(op="or", groups=(nested,)))
                if not alts:
                    continue
                factors.append(alts)
            else:
                factors.append([normalized])
        product = 1
        for factor in factors:
            product *= max(len(factor), 1)
            if product > MAX_PREDICATE_DISTRIBUTE_LEAVES:
                return None
        return factors or [[PredicateGroup(op="or")]]

    @classmethod
    def _distribute_or_over_and(cls, group: PredicateGroup) -> PredicateGroup | None:
        """Expand ``A ∨ (B ∧ C)`` into ``(A ∨ B) ∧ (A ∨ C)`` when under the leaf-product cap."""
        factors = cls._or_factors_for_cnf(group)
        if factors is None:
            or_preds = list(group.predicates)
            nested_or: list[PredicateGroup] = []
            for child in group.groups:
                if child.is_empty():
                    continue
                normalized = cls.normalize_cnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                if normalized.op == "and":
                    nested_or.append(normalized)
                else:
                    or_preds.extend(normalized.predicates)
                    nested_or.extend(normalized.groups)
            result = PredicateGroup(op="or", predicates=tuple(or_preds), groups=tuple(nested_or))
            return None if result.is_empty() else result
        clauses: list[PredicateGroup] = []
        for combo in iter_product(*factors):
            preds: list[WhereParam | HavingParam] = []
            nested: list[PredicateGroup] = []
            for part in combo:
                preds.extend(part.predicates)
                nested.extend(part.groups)
            clauses.append(PredicateGroup(op="or", predicates=tuple(preds), groups=tuple(nested)))
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return PredicateGroup(op="and", groups=tuple(clauses))

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

    @staticmethod
    def _normalization_preference(group: PredicateGroup) -> Literal["dnf", "cnf", "auto"]:
        if group.op == "or" and any(child.op == "and" for child in group.groups):
            return "dnf"
        if group.op == "and" and any(child.op == "or" for child in group.groups):
            for child in group.groups:
                if child.op == "or" and (
                    any(nested.op == "and" for nested in child.groups) or (child.predicates and child.groups)
                ):
                    return "dnf"
            return "cnf"
        return "auto"

    @classmethod
    def coerce(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Canonicalize via DNF/CNF, preferring shallower forms; preference breaks depth ties."""
        if group is None or group.is_empty():
            return None
        if group.is_contradiction():
            return group
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
        min_depth = min(candidate.depth() for _, candidate in within_limit)
        at_min = [(form, candidate) for form, candidate in within_limit if candidate.depth() == min_depth]
        preferred = [candidate for form, candidate in at_min if form == preference]
        chosen = preferred[0] if preferred else at_min[0][1]
        simplified = cls.absorb(cls.flatten(chosen))
        if simplified is None or simplified.is_empty():
            return None
        collapsed = cls.collapse_trivial(cls._dedupe_members(simplified))
        return None if collapsed is None or collapsed.is_empty() else collapsed

    @classmethod
    def collapse_trivial(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Lift singleton OR/AND wrappers so CNF/DNF do not keep redundant one-leaf groups."""
        if group is None or group.is_empty():
            return None
        flat = cls.flatten(group)
        if flat is None or flat.is_empty():
            return None
        if flat.op == "and":
            preds: list[WhereParam | HavingParam] = list(flat.predicates)
            nested: list[PredicateGroup] = []
            for child in flat.groups:
                collapsed = cls.collapse_trivial(child)
                if collapsed is None or collapsed.is_empty():
                    continue
                if collapsed.op == "or" and not collapsed.groups and len(collapsed.predicates) == 1:
                    preds.append(collapsed.predicates[0])
                else:
                    nested.append(collapsed)
            result = PredicateGroup(op="and", predicates=tuple(preds), groups=tuple(nested))
            return None if result.is_empty() else result
        preds = list(flat.predicates)
        nested = []
        for child in flat.groups:
            collapsed = cls.collapse_trivial(child)
            if collapsed is None or collapsed.is_empty():
                continue
            if collapsed.op == "and" and not collapsed.groups and len(collapsed.predicates) == 1:
                preds.append(collapsed.predicates[0])
            else:
                nested.append(collapsed)
        result = PredicateGroup(op="or", predicates=tuple(preds), groups=tuple(nested))
        return None if result.is_empty() else result

    @classmethod
    def normalize_dnf(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Normalize a predicate tree toward OR-of-AND (DNF), distributing when within cap."""
        if group is None or group.is_empty():
            return None
        flat = cls.flatten(group)
        if flat is None or flat.is_empty():
            return None
        if flat.op == "or":
            terms: list[PredicateGroup] = []
            for pred in flat.predicates:
                terms.append(PredicateGroup(op="and", predicates=(pred,)))
            for child in flat.groups:
                normalized = cls.normalize_dnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                if normalized.op == "or":
                    for pred in normalized.predicates:
                        terms.append(PredicateGroup(op="and", predicates=(pred,)))
                    terms.extend(normalized.groups)
                else:
                    terms.append(normalized)
            if not terms:
                return None
            if len(terms) == 1:
                return terms[0]
            return PredicateGroup(op="or", groups=tuple(terms))
        if flat.op == "and":
            if any((cls.normalize_dnf(child) or child).op == "or" for child in flat.groups):
                return cls._distribute_and_over_or(flat)
            and_preds: list[WhereParam | HavingParam] = list(flat.predicates)
            nested: list[PredicateGroup] = []
            for child in flat.groups:
                normalized = cls.normalize_dnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                and_preds.extend(normalized.predicates)
                nested.extend(normalized.groups)
            result = PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))
            return None if result.is_empty() else result
        raise ValueError(f"unsupported predicate op: {flat.op!r}")

    @classmethod
    def normalize_cnf(cls, group: PredicateGroup | None) -> PredicateGroup | None:
        """Normalize a predicate tree toward AND-of-OR (CNF), distributing when within cap."""
        if group is None or group.is_empty():
            return None
        flat = cls.flatten(group)
        if flat is None or flat.is_empty():
            return None
        if flat.op == "and":
            or_clauses: list[PredicateGroup] = []
            for pred in flat.predicates:
                or_clauses.append(PredicateGroup(op="or", predicates=(pred,)))
            for child in flat.groups:
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
        if flat.op == "or":
            if any((cls.normalize_cnf(child) or child).op == "and" for child in flat.groups):
                return cls._distribute_or_over_and(flat)
            or_preds: list[WhereParam | HavingParam] = list(flat.predicates)
            nested_or: list[PredicateGroup] = []
            for child in flat.groups:
                if child.is_empty():
                    continue
                normalized = cls.normalize_cnf(child)
                if normalized is None or normalized.is_empty():
                    continue
                or_preds.extend(normalized.predicates)
                nested_or.extend(normalized.groups)
            result = PredicateGroup(op="or", predicates=tuple(or_preds), groups=tuple(nested_or))
            return None if result.is_empty() else result
        raise ValueError(f"unsupported predicate op: {flat.op!r}")

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
        """Preserve *original* tree shape when leaf counts match. When counts differ, keep the original top-level connective as a flat group rather than forcing AND, which would collapse OR trees during leaf-mutating repairs."""
        if not leaves:
            return None
        if original is None:
            return cls.from_list(list(leaves))
        if len(leaves) == len(original.leaves()):
            return cls.reapply_leaves(original, leaves)
        return PredicateGroup(op=original.op, predicates=tuple(leaves))

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
        if "where_param" in d:
            raise ConfigError("where_param is not accepted; use nested where predicate groups")
        if "where" not in d:
            return None
        raw = d.get("where")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        if isinstance(raw, list):
            raise ConfigError("where must be a nested predicate group object, not a flat list")
        return None

    @classmethod
    def parse_having_field(cls, d: Mapping[str, Any]) -> PredicateGroup | None:
        if "having_param" in d:
            raise ConfigError("having_param is not accepted; use nested having predicate groups")
        if "having" not in d:
            return None
        raw = d.get("having")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return cls.from_dict(raw, having=True)
        if isinstance(raw, list):
            raise ConfigError("having must be a nested predicate group object, not a flat list")
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
        self.direction = str(self.direction or "ASC").strip().upper() or "ASC"

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
        direction_raw = d.get("direction")
        direction = str(direction_raw).strip() if isinstance(direction_raw, str) and direction_raw.strip() else "ASC"
        return OrderByCol(
            expr=expr,
            direction=direction,
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
        self.op = str(self.op or "=").strip().lower() or "="
        self.value_type = str(self.value_type).strip().lower()
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
        op_raw = d.get("op")
        op = str(op_raw).strip() if isinstance(op_raw, str) and op_raw.strip() else "="
        vt_raw = d.get("value_type")
        value_type = str(vt_raw).strip() if isinstance(vt_raw, str) and vt_raw.strip() else "string"
        return WhereParam(
            left_expr=NormalizedExpr.from_stored_json(left_raw),
            op=op,
            right_expr=(NormalizedExpr.from_stored_json(right_raw) if right_raw else None),
            value_type=value_type,
            param_key=d.get("param_key") or "",
            param_key_hi=d.get("param_key_hi") or "",
            param_key_unit=d.get("param_key_unit") or "",
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

    @property
    def identity_key(self) -> str:
        """Value-aware identity for boolean dedup and absorption (includes bound literals)."""
        parts = [self.signature_key]
        if self.right_expr is not None:
            return "|".join(parts)
        if self.raw_value is not None:
            parts.append(f"v={self.raw_value!r}")
        elif self.param_key:
            parts.append(f"pk={self.param_key}")
        if self.param_key_hi:
            parts.append(f"pkhi={self.param_key_hi}")
        if self.param_key_unit:
            parts.append(f"pku={self.param_key_unit}")
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
            "Optional SQL expression for expr-vs-expr predicates; may reference a different table from left_expr to express a value comparison rather than a join relationship."
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
        self.op = str(self.op or "=").strip().lower() or "="
        self.value_type = str(self.value_type).strip().lower()
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
        op_raw = d.get("op")
        op = str(op_raw).strip() if isinstance(op_raw, str) and op_raw.strip() else "="
        vt_raw = d.get("value_type")
        value_type = str(vt_raw).strip() if isinstance(vt_raw, str) and vt_raw.strip() else "number"
        return HavingParam(
            left_expr=NormalizedExpr.from_stored_json(left_raw),
            op=op,
            right_expr=(NormalizedExpr.from_stored_json(right_raw) if right_raw else None),
            value_type=value_type,
            param_key=d.get("param_key") or "",
            param_key_unit=d.get("param_key_unit") or "",
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

    @property
    def identity_key(self) -> str:
        """Value-aware identity for boolean dedup and absorption (includes bound literals)."""
        parts = [self.signature_key]
        if self.right_expr is not None:
            return "|".join(parts)
        if self.raw_value is not None:
            parts.append(f"v={self.raw_value!r}")
        elif self.param_key:
            parts.append(f"pk={self.param_key}")
        if self.param_key_unit:
            parts.append(f"pku={self.param_key_unit}")
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
            "Optional SQL expression for agg-vs-agg predicates; may reference a different table from left_expr to express an aggregate comparison rather than a join relationship."
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
    source_probe: str = ""
    store_fingerprint: str = ""
    ingest_source_probe: str = ""
