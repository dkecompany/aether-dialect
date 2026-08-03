"""Shared dataclasses and enums for schema graphs, validation, templates, QSim skeletons, and type helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol, cast

import pandas

from ._constants import (
    COLUMN_TYPE_TO_VALUE_TYPE,
    CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC,
    DATE_TYPE_TOKENS,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    INFERENCE_TAG_VALUES,
    MAX_PREDICATE_NESTING_DEPTH,
    NUMERIC_TYPE_TOKENS,
    OP_FLIP,
    PK_INFERENCE_TAG_VALUES,
    RAW_SQL_AGG_OR_WINDOW_RE,
    REGISTRY_REF_TOKEN_RE,
    ROLE_OWNER_VALUES,
    STRING_TYPE_TOKENS,
    STRUCTURAL_DATA_TYPE_CANONICAL,
)


def normalize_column_type(col_type: str) -> str:
    """Lowercase a SQL type and remove `(n)` / `(n,m)` parameter lists."""
    normalized = col_type.lower().strip()
    normalized = re.sub(r"\(\d+(?:,\s*\d+)?\)", "", normalized)
    normalized = normalized.strip()
    return normalized


def structural_data_type_key(data_type: str) -> str:
    """Return a canonical catalog-type token for structural diff and hashing."""
    normalized = normalize_column_type(data_type)
    if not normalized:
        return normalized
    return STRUCTURAL_DATA_TYPE_CANONICAL.get(normalized, normalized)


SchemaInclude = Literal["tables", "views"]
SchemaRole = Literal["owner", "consumer"]


class ComplexityTier(str, Enum):
    """Workload complexity band for QSim sampling and seed-warmup stratification."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class WarmupStyle(str, Enum):
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


class SensitivityClassification(str, Enum):
    """Single-column sensitivity tier for projection, filtering, and LLM visibility."""

    NONE = "none"
    RESTRICTED = "restricted"
    HIDDEN = "hidden"


def coerce_sensitivity_classification(raw: Any) -> SensitivityClassification | None:
    """Parse :class:`SensitivityClassification` tokens from overrides JSON or other string inputs."""
    if raw is None:
        return None
    if isinstance(raw, SensitivityClassification):
        return raw
    s = str(raw).strip().lower()
    if not s:
        return None
    for m in SensitivityClassification:
        if m.value == s:
            return m
    return None


def column_sensitivity_from_dict(d: Mapping[str, Any]) -> SensitivityClassification:
    """Resolve a persisted ``sensitivity`` key into a :class:`SensitivityClassification`."""
    return coerce_sensitivity_classification(d.get("sensitivity")) or SensitivityClassification.NONE


class NoveltyBand(str, Enum):
    """Expansion-depth novelty band for anchor-lattice sharing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkloadFamily(str, Enum):
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


WindowOperatorKind = Literal["none", "rank", "aggregate", "navigate"]


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


def default_order_by_null_placement(direction: str) -> OrderByNullPlacement:
    """Return DuckDB/PostgreSQL default null placement for an ORDER BY direction."""
    dir_up = (direction or "ASC").strip().upper()
    return "first" if dir_up == "DESC" else "last"


def _tier_feasible_for_capability(tier_key: str, cap: DatabaseFeatureCapability) -> bool:
    """Return whether a complexity tier remains achievable on this. database snapshot. Args: tier_key: One of ``simple``, ``moderate``, ``complex``, ``highly_complex``. cap: Capability snapshot from :func:`compute_database_feature_capability`. Returns: False when structural prerequisites for that tier are absent."""
    if cap.table_count <= 0:
        return False
    if tier_key == ComplexityTier.SIMPLE.value:
        return True
    if tier_key == ComplexityTier.MODERATE.value:
        return cap.table_count >= 1
    if tier_key == ComplexityTier.COMPLEX.value:
        return (
            cap.max_tables_on_any_join_path >= 3
            or (cap.table_count >= 2 and cap.fk_edge_count >= 1)
            or (cap.has_numeric_measures and cap.table_count >= 1)
        )
    if tier_key == ComplexityTier.HIGHLY_COMPLEX.value:
        return (
            cap.max_fk_chain_depth >= 2
            or cap.has_self_referential_fk
            or cap.max_tables_on_any_join_path >= 4
            or (cap.max_tables_on_any_join_path >= 3 and cap.has_window_capable_table_sets)
        )
    return False


def rebalance_complexity_target_proportions(
    proportions: Mapping[str, float], cap: DatabaseFeatureCapability
) -> dict[str, float]:
    """Zero unreachable tier mass and renormalize remaining targets for. QSim and warmup budgets. Args: proportions: Named tier weights summing to approximately one. cap: Live capability snapshot for structural feasibility. Returns: Renormalized tier weights over feasible tiers only."""
    keys = [
        ComplexityTier.SIMPLE.value,
        ComplexityTier.MODERATE.value,
        ComplexityTier.COMPLEX.value,
        ComplexityTier.HIGHLY_COMPLEX.value,
    ]
    feas = {k: _tier_feasible_for_capability(k, cap) for k in keys}
    raw_mass = sum(max(0.0, float(proportions.get(k, 0.0))) for k in keys if feas[k])
    if raw_mass <= 0.0:
        active = [k for k in keys if feas[k]]
        if not active:
            return {k: 0.25 for k in keys}
        u = 1.0 / float(len(active))
        return {k: (u if k in active else 0.0) for k in keys}
    out: dict[str, float] = {}
    for k in keys:
        if feas[k]:
            out[k] = max(0.0, float(proportions.get(k, 0.0))) / raw_mass
        else:
            out[k] = 0.0
    s = sum(out.values())
    if s <= 0.0:
        active = [k for k in keys if feas[k]]
        u = 1.0 / float(len(active))
        return {k: (u if k in active else 0.0) for k in keys}
    return {k: (v / s) for k, v in out.items()}


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


class FailureCategory(str, Enum):
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


_FAILURE_CATEGORY_MEMBER_ORDER: tuple[FailureCategory, ...] = tuple(FailureCategory)


def parse_failure_category(raw: FailureCategory | str | None) -> FailureCategory | None:
    """
    Map a free-form category string to ``FailureCategory``.

    Args:

        raw: Category from validation, classifier, or legacy payloads.

    Returns:

        ``None`` when *raw* is empty; otherwise the matching enum member or ``OTHER``.
    """
    if raw is None:
        return None
    if isinstance(raw, FailureCategory):
        return raw
    v = str(raw).strip()
    if not v:
        return None
    try:
        return FailureCategory(v)
    except ValueError:
        return FailureCategory.OTHER


class SqlDiagnosticCode(str, Enum):
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
    EXPLAIN_TYPE_MISMATCH = "explain_type_mismatch"
    EXPLAIN_PERMISSION_DENIED = "explain_permission_denied"
    EXPLAIN_OTHER = "explain_other"
    EXPLAIN_COST_EXCEEDED = "explain_cost_exceeded"


@dataclass(frozen=True)
class SqlDiagnostic:
    """Single structured finding from AST validation or EXPLAIN-plan analysis."""

    code: SqlDiagnosticCode
    message: str
    node_kind: str | None = None
    offending_identifier: str | None = None
    details: Mapping[str, str] = field(default_factory=dict)


def norm_schema_identifier(name: str, *, what: str) -> str:
    """Lowercase and strip *name*; raise when empty after strip."""
    s = str(name).strip().lower()
    if not s:
        raise ValueError(f"{what} must be non-empty")
    return s


class ConfigError(ValueError):
    """Raised when environment variables or static configuration are missing or contradictory."""


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
    runtime_config: type


class InteractiveChoicePort(Protocol):
    """Bridges yes/no prompts to a session queue or stdin."""

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        ...

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Return a normalised choice or raise ``PipelineSuspended`` when the queue is empty."""
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


_mock_fixture_recorded_corpus_count: Any = None


def register_mock_fixture_recorded_corpus_count(fn: Any) -> None:
    """Register sandbox recorded-corpus count provider for :class:`MockFixtureMissingError` messages."""
    global _mock_fixture_recorded_corpus_count
    _mock_fixture_recorded_corpus_count = fn


class MockFixtureMissingError(RuntimeError):
    """Raised when the mock provider has no fixture for the requested LLM call."""

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
                "Ask a recorded question or call sandbox_questions(). "
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
        super().__init__(f"Operation {operation!r} requires role='owner'; this instance is a consumer.")


class MigrationPendingError(ValueError):
    """Init terminated because schema_migration_map.json is required, malformed, missing after export, or conflicts with validation."""


class ConnectionError(OSError):
    """Raised when the database driver rejects a connection attempt."""


class RetryableError(Exception):
    """Marker base class for transient failures that may succeed on retry. Concrete subclasses combine this marker with :class:`ConnectionError`, :class:`RuntimeError`, etc. Integrators may use ``isinstance(exc, RetryableError)`` without inspecting messages."""


class DatabasePingFailed(ConnectionError, RetryableError):
    """Raised when a trivial ``SELECT 1`` ping fails after retries (network blips, overload)."""


class LlmTransientFailure(RuntimeError, RetryableError):
    """LLM request failed after retries due to rate limits, timeouts, or connection resets."""


class StatementTimeoutError(RuntimeError, RetryableError):
    """Raised when the database aborts work due to ``statement_timeout`` or warehouse timeouts."""


class SchemaAccessError(ValueError):
    """Raised when credentials cannot read the requested scope or the graph is unusable. Covers catalog reflection, construction invariants, and scope configuration at init. Runtime ``EXPLAIN`` / ``execute`` permission denial is raised as :class:`AccessError`, which is a subclass so integrators may catch this type for every access denial surface."""


class SessionActiveError(RuntimeError):
    """Raised when ``PipelineSession.ask`` is called while a turn is already in progress."""


class SessionTurnCancelledError(Exception):
    """Raised when a programmatic session turn is cancelled cooperatively."""


class SchemaInvariantError(RuntimeError):
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

    change: Literal["none", "add", "remove", "mixed"]
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
            return FederationRuntimeError.__new__(_RetryableFederationPartialFailureError)
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


class _RetryableFederationPartialFailureError(FederationPartialFailureError, RetryableError):
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


def _federation_table_source_index(
    schema: Any, mappings: FederationMappings, manifest: FederationManifest | None = None
) -> dict[str, str]:
    index: dict[str, str] = {}
    logical_member_sources: dict[str, set[str]] = {}
    for lt in mappings.logical_tables:
        logical_member_sources[lt.logical] = {m.source for m in lt.members}
    tables = getattr(schema, "tables", None) or {}
    for name, table in tables.items():
        member_source_ids = getattr(table, "member_source_ids", None) or ()
        if member_source_ids:
            if len(member_source_ids) == 1:
                index[name] = member_source_ids[0]
            continue
        source_id = getattr(table, "source_id", "") or ""
        if source_id:
            index[name] = source_id
        elif name in logical_member_sources:
            sources = logical_member_sources[name]
            if len(sources) == 1:
                index[name] = next(iter(sources))
    if manifest is not None:
        for name, source_id in manifest.table_namespace.items():
            if name in tables and name not in index:
                index[name] = source_id
    return index


def resolve_federation_qualified_ref(
    ref: str,
    *,
    manifest: FederationManifest,
    schema: Any | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> ResolvedQualifiedRef:
    """Resolve ``table.column`` or ``source.table.column``, erroring on ambiguity."""
    text = str(ref or "").strip()
    if not text:
        raise ConfigError("empty federation qualified reference")
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        source_id, table, column = three.group(1), three.group(2), three.group(3)
        declared_sources = {s.source_id for s in manifest.sources}
        if not declared_sources:
            declared_sources = set(manifest.table_namespace.values())
            if schema is not None:
                declared_sources |= {
                    str(getattr(table_meta, "source_id", "") or "").strip()
                    for table_meta in getattr(schema, "tables", {}).values()
                    if str(getattr(table_meta, "source_id", "") or "").strip()
                }
        if declared_sources and source_id not in declared_sources:
            raise ConfigError(f"unknown federation source in reference: {text!r}")
        ns_table = manifest.table_namespace.get(table, "")
        if ns_table and ns_table != source_id:
            raise ConfigError(
                f"qualified reference {text!r} disagrees with table_namespace ({table!r} -> {ns_table!r})"
            )
        if schema is not None:
            table_meta = tables.get(table) if (tables := getattr(schema, "tables", None)) else None
            if table_meta is not None:
                composite_source = getattr(table_meta, "source_id", "") or ""
                if composite_source and composite_source != source_id:
                    raise ConfigError(f"qualified reference {text!r} disagrees with composite source_id")
        return ResolvedQualifiedRef(source_id=source_id, table=table, column=column, qualified=f"{table}.{column}")
    two = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if not two:
        raise ConfigError(f"expected table.column or source.table.column: {text!r}")
    table, column = two.group(1), two.group(2)
    index = dict(source_by_table or {})
    if not index and schema is not None:
        index = _federation_table_source_index(
            schema, FederationMappings(version=FEDERATION_MAPPINGS_VERSION), manifest
        )
    candidates: set[str] = set()
    ns = manifest.table_namespace.get(table, "")
    if ns:
        candidates.add(ns)
    mapped = index.get(table, "")
    if mapped:
        candidates.add(mapped)
    if schema is not None:
        composite_table = getattr(schema, "tables", {}).get(table)
        if composite_table is not None:
            composite_source = getattr(composite_table, "source_id", "") or ""
            if composite_source:
                candidates.add(composite_source)
            candidates.update(getattr(composite_table, "member_source_ids", None) or ())
    candidates.discard("")
    if len(candidates) > 1:
        raise ConfigError(f"ambiguous federation reference {text!r}: sources {sorted(candidates)}")
    if not candidates:
        if not manifest.sources and not manifest.table_namespace:
            return ResolvedQualifiedRef(source_id="", table=table, column=column, qualified=f"{table}.{column}")
        raise ConfigError(f"cannot attribute federation reference: {text!r}")
    source_id = next(iter(candidates))
    return ResolvedQualifiedRef(source_id=source_id, table=table, column=column, qualified=f"{table}.{column}")


class MigrationTier(str, Enum):
    """Classified migration severity between a stored artifact fingerprint and the live graph."""

    NO_CHANGE = "no_change"
    PERMISSION_FILTERED = "permission_filtered"
    SOFT_REFRESH = "soft_refresh"
    ADDITIVE = "additive"
    REMAP = "remap"
    DESTRUCTIVE = "destructive"


class ColumnVisibilityBlockReason(str, Enum):
    """Machine-stable reason a column is blocked from LLM exposure or reference validation."""

    DENIED = "denied"
    NOT_IN_ALLOW_COLUMNS = "not_in_allow_columns"
    SENSITIVE_HIDDEN = "sensitive_hidden"
    SENSITIVE_RESTRICTED = "sensitive_restricted"
    UNUSABLE = "unusable"


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
    skipped: tuple[OverrideSkip, ...] = ()


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Structured status line emitted alongside ``notify`` / ``debug`` for programmatic consumers."""

    stage: str
    level: str
    code: str
    message: str
    details: tuple[tuple[str, str], ...] = ()
    duration_ms: int | None = None
    source_id: str | None = None


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
            level = severity or issue.level
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
    """Single observable point in a programmatic interactive turn. Carries whether the turn has finished, a short instruction string, a stage discriminant, optional SQL, tabular data, a free-form body, and an error string when the engine fails. done: True when the pipeline finished successfully or ended in a terminal error; False when the caller must respond via ``PipelineSession.step``. prompt: The short line the interactive layer should show immediately before collecting input (for example yes or no, or a free-text rejection reason prompt). kind: Stable stage identifier matching the active suspend kind or a terminal sentinel; used to branch programmatic UIs without parsing ``prompt``. sql: The formatted SQL under discussion when the step pertains to execution or confirmation; otherwise None. data: Row-level query preview or full result as a ``pandas.DataFrame``; None for scalar outcomes, previews trimmed to five rows at suspend boundaries, and the full frame on the terminal acceptance step when rows exist. message: Multi-line contextual body: consolidated intent confirmation, migration DDL, rejection guidance, or a rendered scalar value; empty or None when nothing extra should print beyond ``prompt`` and ``data``. error: Terminal failure explanation when ``done`` is True and processing stopped; otherwise None. intent_summary: Structured intent headline when the step reflects a parsed intent or later pipeline stages; otherwise None. diagnostics: Structured diagnostics captured during this step (from ``notify`` / ``debug`` when a collector is active). status: On terminal error steps, a coarse failure category name (same string values as :class:`FailureCategory`); None on success or non-terminal steps. reply_shape: When ``done`` is False, whether the caller should collect a yes or no token or free text; None on terminal steps. semantic_warnings: Normalised warning strings for intent confirmation, often empty on non-intent suspend steps."""

    done: bool
    prompt: str | None
    kind: str
    sql: str | None = None
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
    stage: int | None = None


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
    q_norm: str
    sql_param: str
    display_sql: str
    space: str
    use_count: int
    created_at: str
    last_used_at: str


@dataclass(frozen=True, slots=True)
class StoredTemplateDetail:
    """Full caller-visible detail for one accepted template."""

    summary: StoredTemplateSummary
    sql_fp: str
    tables_used: tuple[str, ...]
    trust_level: int
    parameters: tuple[ParameterBinding, ...]


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
    include: SchemaInclude = "tables"
    deny_objects: frozenset[str] = frozenset()
    deny_columns: frozenset[str] = frozenset()
    allow_columns: frozenset[str] = frozenset()
    notes_file: str | None = None
    sql_file: str | None = None

    def __post_init__(self) -> None:
        if self.include not in ("tables", "views"):
            raise ConfigError(f"include must be 'tables' or 'views', not {self.include!r}")
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        if self.sql_file is not None and not str(self.sql_file).strip():
            raise ConfigError("sql_file must be omitted or a non-empty path")
        allow_norm = frozenset(norm_schema_identifier(t, what="allow_objects entry") for t in self.allow_objects)
        deny_obj_norm = frozenset(norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects)
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"allow_objects and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_specs: list[str] = []
        for spec in self.deny_columns:
            normalized_specs.append(_normalize_scope_column_spec(spec, field="deny_columns"))
        col_set = frozenset(normalized_specs)
        allow_col_specs: list[str] = []
        for spec in self.allow_columns:
            allow_col_specs.append(_normalize_scope_column_spec(spec, field="allow_columns"))
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
    include: SchemaInclude = "tables"
    deny_objects: frozenset[str] = frozenset()
    deny_columns: frozenset[str] = frozenset()
    allow_columns: frozenset[str] = frozenset()
    notes_file: str | None = None

    def __post_init__(self) -> None:
        if self.include not in ("tables", "views"):
            raise ConfigError(f"include must be 'tables' or 'views', not {self.include!r}")
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        allow_norm = frozenset(norm_schema_identifier(t, what="allow_objects entry") for t in self.allow_objects)
        deny_obj_norm = frozenset(norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects)
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"allow_objects and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_specs: list[str] = []
        for spec in self.deny_columns:
            normalized_specs.append(_normalize_scope_column_spec(spec, field="deny_columns"))
        col_set = frozenset(normalized_specs)
        allow_col_specs: list[str] = []
        for spec in self.allow_columns:
            allow_col_specs.append(_normalize_scope_column_spec(spec, field="allow_columns"))
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


def _normalize_scope_column_spec(spec: str, *, field: str) -> str:
    """Normalize a ``table.column`` or ``source.table.column`` scope column spec."""
    raw = str(spec).strip()
    parts = raw.split(".")
    if len(parts) == 2:
        tbl_raw, col_raw = parts
    elif len(parts) == 3:
        _source_raw, tbl_raw, col_raw = parts
    else:
        raise ConfigError(CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC.format(spec=spec))
    tbl = norm_schema_identifier(tbl_raw, what=f"{field} table")
    col = norm_schema_identifier(col_raw, what=f"{field} column")
    if tbl == "*":
        return f"*.{col}"
    return f"{tbl}.{col}"


def normalize_scope_column_spec(spec: str, *, field: str) -> str:
    """Normalize a ``table.column`` or ``source.table.column`` scope column spec."""
    return _normalize_scope_column_spec(spec, field=field)


@dataclass(frozen=True, slots=True)
class BusinessKnowledgeEntry:
    """Single prompt-time business knowledge item not tied to schema column descriptions."""

    key: str
    text: str
    kind: str = "glossary"


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

    def __post_init__(self) -> None:
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        allow_norm = frozenset(norm_schema_identifier(t, what="tables entry") for t in self.tables)
        deny_obj_norm = frozenset(norm_schema_identifier(t, what="deny_objects entry") for t in self.deny_objects)
        overlap_obj = allow_norm & deny_obj_norm
        if overlap_obj:
            raise ConfigError(f"SpaceContext tables and deny_objects overlap: {sorted(overlap_obj)!r}")
        normalized_cols: list[str] = []
        for spec in self.columns:
            normalized_cols.append(_normalize_scope_column_spec(spec, field="columns"))
        col_set = frozenset(normalized_cols)
        deny_col_specs: list[str] = []
        for spec in self.deny_columns:
            deny_col_specs.append(_normalize_scope_column_spec(spec, field="deny_columns"))
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
    role: SchemaRole = "consumer"
    limits: FederationSourceLimits | None = None


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

    version: int
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
    format_version: int = 2
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
    schema_role: SchemaRole = "owner"
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


def is_numeric_type(data_type: str) -> bool:
    """
    Return whether a SQL data type string looks numeric.

    Args:

        data_type: Raw SQL type string.

    Returns:

        True if a numeric token appears in the lowercased string.
    """
    dt = data_type.lower()
    return any(t in dt for t in NUMERIC_TYPE_TOKENS)


def is_string_type(data_type: str) -> bool:
    """
    Return whether a SQL data type string looks string-like.

    Args:

        data_type: Raw SQL data type string.

    Returns:

        True if any string/text token is found in the lowercased type string.
    """
    dt = data_type.lower()
    return any(t in dt for t in STRING_TYPE_TOKENS)


def is_date_type(data_type: str) -> bool:
    """
    Return whether a SQL data type string looks date- or time-like.

    Args:

        data_type: Raw SQL data type string.

    Returns:

        True if any temporal token appears in the lowercased string.
    """
    dt = data_type.lower()
    return any(t in dt for t in DATE_TYPE_TOKENS)


def data_type_to_value_type(data_type: str) -> str:
    normalized = normalize_column_type(data_type)
    vt = COLUMN_TYPE_TO_VALUE_TYPE.get(normalized)
    if vt:
        return vt
    if is_numeric_type(data_type):
        return "number"
    if is_date_type(data_type):
        return "date"
    if is_string_type(data_type):
        return "string"
    return "string"


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


class InferenceTag(str, Enum):
    """Provenance tag for an :class:`FKEdge`. A catalog-declared edge is represented by ``None`` rather than a member of this enum so that presence-of-tag and identity-of- inferred-layer are reflected by a single attribute. Inherits ``str`` so members compare equal to their wire value and round-trip through JSON without custom encoding."""

    SUFFIX = "suffix"
    SELF = "self"
    COMPOSITE = "composite"
    SEMANTIC = "semantic"
    SEMANTIC_PROMOTED = "semantic_promoted"
    USER_STRUCTURAL = "user_override_structural"
    USER_SEMANTIC = "user_override_semantic"
    CROSS_SOURCE = "cross_source"


def coerce_inference_tag(raw: object) -> InferenceTag | None:
    """Normalise raw cache or override input into :class:`InferenceTag` (``None`` for catalog). Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, InferenceTag):
        return raw
    if isinstance(raw, str) and raw in INFERENCE_TAG_VALUES:
        return InferenceTag(raw)
    raise ValueError(f"unknown FK inference_tag: {raw!r}")


class PkInferenceTag(str, Enum):
    """Provenance tag for an inferred or user-supplied primary key. Engine-reflected catalog keys use ``None`` (locked). SQL-file- declared keys use ``DDL`` (overridable). Inferred and user-supplied keys use the remaining members."""

    DDL = "ddl"
    PROFILE = "profile"
    USER_OVERRIDE = "user_override"


def coerce_pk_inference_tag(raw: object) -> PkInferenceTag | None:
    """Normalise raw cache or override input into :class:`PkInferenceTag` (``None`` for catalog). Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, PkInferenceTag):
        return raw
    if isinstance(raw, str) and raw in PK_INFERENCE_TAG_VALUES:
        return PkInferenceTag(raw)
    raise ValueError(f"unknown pk_inference_tag: {raw!r}")


class RoleOwner(str, Enum):
    """Provenance for the writer that last set :attr:`ColumnMetadata.role`. The members are ordered by ascending precedence: a writer with strictly greater precedence may overwrite a role assigned by a lower-precedence owner, while equal-or-lower-precedence writers must skip the column. PK/FK coercion is treated as the highest authority because it is required for join correctness; user overrides win over LLM inference, which in turn wins over profile heuristics, which in turn wins over the default catalog fallback."""

    CATALOG = "catalog"
    PROFILE = "profile"
    LLM = "llm"
    BOOLEAN_COERCION = "boolean_coercion"
    USER_OVERRIDE = "user_override"
    PK_FK_COERCION = "pk_fk_coercion"


_ROLE_OWNER_PRECEDENCE: dict[RoleOwner, int] = {
    RoleOwner.CATALOG: 0,
    RoleOwner.PROFILE: 1,
    RoleOwner.LLM: 2,
    RoleOwner.BOOLEAN_COERCION: 3,
    RoleOwner.USER_OVERRIDE: 4,
    RoleOwner.PK_FK_COERCION: 5,
}


def coerce_role_owner(raw: object) -> RoleOwner | None:
    """Normalise raw cache or override input into :class:`RoleOwner` (``None`` when unset). Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, RoleOwner):
        return raw
    if isinstance(raw, str) and raw in ROLE_OWNER_VALUES:
        return RoleOwner(raw)
    raise ValueError(f"unknown role_owner: {raw!r}")


def can_overwrite_role(current: RoleOwner | None, candidate: RoleOwner) -> bool:
    """Return whether a writer with provenance *candidate* may overwrite a role currently owned by *current*. A column whose role has never been claimed (``current is None``) accepts any writer. Otherwise the candidate must have strictly greater precedence than the incumbent owner; equal-precedence writes are rejected so the first writer of a given tier wins deterministically."""
    if current is None:
        return True
    return _ROLE_OWNER_PRECEDENCE[candidate] > _ROLE_OWNER_PRECEDENCE[current]


class DescriptionOwner(str, Enum):
    """Provenance for the writer that last set a description on a table or column. Members are ordered by ascending precedence; the dedicated helper :func:`set_description` enforces a strict-greater-precedence rule so a later writer can only overwrite an existing description when its provenance outranks the incumbent owner."""

    CATALOG = "catalog"
    PROFILE = "profile"
    NOTES = "notes"
    LLM_REFINEMENT = "llm_refinement"
    USER_OVERRIDE = "user_override"


_DESCRIPTION_OWNER_PRECEDENCE: dict[DescriptionOwner, int] = {
    DescriptionOwner.CATALOG: 0,
    DescriptionOwner.PROFILE: 1,
    DescriptionOwner.NOTES: 2,
    DescriptionOwner.LLM_REFINEMENT: 3,
    DescriptionOwner.USER_OVERRIDE: 4,
}


def _description_owner_rank(owner: DescriptionOwner | None) -> int:
    if owner is None:
        return _DESCRIPTION_OWNER_PRECEDENCE[DescriptionOwner.CATALOG]
    if not isinstance(owner, DescriptionOwner):
        owner = DescriptionOwner(owner)
    return _DESCRIPTION_OWNER_PRECEDENCE[owner]


def resolve_descriptions(
    *candidates: tuple[str | None, DescriptionOwner | None],
) -> tuple[str, DescriptionOwner | None]:
    """Resolve simultaneous description candidates using owner precedence. Only non-empty texts participate. Among them, the highest-precedence owner tier wins; when that tier contains exactly one distinct text, return it with its owner, otherwise return an empty description."""
    nonempty: list[tuple[str, DescriptionOwner | None]] = []
    for text, owner in candidates:
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        coerced = owner
        if coerced is not None and not isinstance(coerced, DescriptionOwner):
            coerced = DescriptionOwner(coerced)
        nonempty.append((cleaned, coerced))
    if not nonempty:
        return "", None
    max_rank = max(_description_owner_rank(owner) for _, owner in nonempty)
    tier = [(text, owner) for text, owner in nonempty if _description_owner_rank(owner) == max_rank]
    distinct_texts = sorted({text for text, _ in tier})
    if len(distinct_texts) == 1:
        winner_owner = next(owner for text, owner in tier if text == distinct_texts[0])
        return distinct_texts[0], winner_owner
    return "", None


def set_description(target: Any, text: str | None, owner: DescriptionOwner) -> bool:
    """
    Single writer for ``description`` on tables and columns.

    Args:

        target: Either a :class:`TableMetadata` or :class:`ColumnMetadata` instance.
        text: New description text (``None`` rejects the write; empty string clears to ``""``).
        owner: Provenance of the writer.

    Returns:
        ``True`` when the description was actually updated, ``False`` when the
        write was rejected because *owner* has lower precedence than the current owner
        (strictly higher precedence wins; equal precedence allows replacement).
    """
    if text is None:
        return False
    current_owner = getattr(target, "description_owner", None)
    if current_owner is not None:
        if not isinstance(current_owner, DescriptionOwner):
            current_owner = DescriptionOwner(current_owner)
        if _DESCRIPTION_OWNER_PRECEDENCE[owner] < _DESCRIPTION_OWNER_PRECEDENCE[current_owner]:
            return False
    cur_desc = (getattr(target, "description", None) or "").strip()
    new_desc = str(text).strip()
    if cur_desc == new_desc and current_owner == owner:
        return False
    target.description = new_desc
    target.description_owner = owner
    return True


def set_sensitivity(col: Any, value: SensitivityClassification | str | None) -> None:
    """Single writer for :attr:`ColumnMetadata.sensitivity`. Accepts :class:`SensitivityClassification` or ``None`` for :attr:`SensitivityClassification.NONE`. Clears concrete profile samples whenever the classification is not :attr:`SensitivityClassification.NONE`."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        resolved = SensitivityClassification.NONE
    elif isinstance(value, SensitivityClassification):
        resolved = value
    else:
        sv = str(value).strip().lower()
        resolved = coerce_sensitivity_classification(sv) or SensitivityClassification.NONE
    col.sensitivity = resolved
    if resolved != SensitivityClassification.NONE:
        col.frequent_values = []
        col.value_overlap_sample = []
        col.min_val = None
        col.max_val = None


class AccessError(SchemaAccessError, RuntimeError):
    """Raised when the database refuses ``EXPLAIN`` or ``execute`` because of insufficient privileges."""

    def __init__(self, operation: Literal["explain", "execute"], message: str, *, relation: str | None = None) -> None:
        """Attach *operation*, human *message*, and optional *relation* hint for UX."""
        self.operation = operation
        self.relation = relation
        super().__init__(message)


class PipelineSuspended(Exception):
    """Raised when a programmatic interactive turn must wait for the next ``submit_*`` call."""

    def __init__(self, state_id: str, message_for_caller: str, payload: Any | None = None) -> None:
        self.state_id = state_id
        self.message_for_caller = message_for_caller
        self.payload = payload
        super().__init__(message_for_caller)


class NoJoinPathError(Exception):
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
        if self.tables:
            table_text = ", ".join(self.tables)
        else:
            table_text = "the requested tables"
        return (
            f"These tables could not be connected: {table_text}. "
            "Declare a foreign-key or semantic relationship between them, or ask using fewer tables."
        )


class JoinPathTieCapExceededError(Exception):
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


class AggregateJoinFanOutError(Exception):
    """Raised when a resolved join path would duplicate rows aggregated at parent grain."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ClauseWidenedRowsetError(Exception):
    """Raised when LIMIT or DISTINCT ON would run on a join-widened row set."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ProbeCtePlacementError(Exception):
    """Raised when a semi-join or anti-join probe CTE is used as a join anchor or left operand."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ComparisonJoinScopeExceededError(Exception):
    """Raised when a cross-table comparison forces a join path beyond the allowed scope."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class JoinColumnCountMismatchError(Exception):
    """Raised when a join signature pairs unequal numbers of left and right columns."""

    def __init__(self, segment: str, left_count: int, right_count: int) -> None:
        self.segment = segment
        self.left_count = left_count
        self.right_count = right_count
        super().__init__(
            f"join path segment {segment!r} pairs {left_count} left column(s) with {right_count} right column(s)"
        )


class JoinInjectionAlignmentError(Exception):
    """Raised when ``join_sigs_ordered`` does not align one-to-one with dialect join carriers on deterministic SQL."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class JoinInjectionFailedError(Exception):
    """Raised when deterministic SQL cannot be rewritten with structured JOIN/WHERE edges via the dialect AST adapter."""

    def __init__(
        self, message: str, *, det_sql: str, join_sigs_ordered: list[list[str]], edge_kinds_ordered: list[list[str]]
    ) -> None:
        self.det_sql = det_sql
        self.join_sigs_ordered = join_sigs_ordered
        self.edge_kinds_ordered = edge_kinds_ordered
        super().__init__(message)


class LlmJsonExhausted(Exception):
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


_PARSE_EXPR_STRING_FN: Any | None = None
_RENDER_EXPR_SQL_FN: Any | None = None


def register_parse_expr_string(fn: Any) -> None:
    global _PARSE_EXPR_STRING_FN
    _PARSE_EXPR_STRING_FN = fn


def register_render_expr_sql(fn: Any) -> None:
    global _RENDER_EXPR_SQL_FN
    _RENDER_EXPR_SQL_FN = fn


def parse_expr_string_for_json(s: str) -> NormalizedExpr:
    """Parse a JSON string field that contains a SQL expression into a. ``NormalizedExpr``. Args: s: Non-empty expression text from the model. Returns: Parsed structure when the registered parser is available; otherwise a column ref leaf."""
    t = (s or "").strip()
    if not t:
        return NormalizedExpr()
    fn = _PARSE_EXPR_STRING_FN
    if fn is not None:
        return cast(NormalizedExpr, fn(t))
    return NormalizedExpr.from_column(t)


ScalarArg = str | int | float
ParamValue = str | int | float | bool | list[str | int | float]
RawValue = str | int | float | bool | list[str | int | float] | dict[str, str | int] | None

CteEmissionKind = Literal["join_table", "scalar_subquery", "semi_join", "anti_join"]
OrderByNullPlacement = Literal["first", "last"]
WindowFrameKind = Literal["rows", "range", "none"]

PROBE_CTE_EMISSION_KINDS: frozenset[str] = frozenset({"semi_join", "anti_join"})


def coerce_cte_emission(raw: Any) -> CteEmissionKind:
    """
    Normalize a stored emission string to a supported literal.

    Args:

        raw: Value from JSON or legacy payloads.

    Returns:

        ``join_table`` unless ``raw`` is a supported probe emission literal.
        ``scalar_subquery`` is engine-owned and coerced to ``join_table``.
    """
    if raw in ("semi_join", "anti_join"):
        return cast(CteEmissionKind, raw)
    return "join_table"


def coerce_order_by_nulls(raw: Any) -> OrderByNullPlacement | None:
    """Normalize optional ORDER BY null-placement to ``first``, ``last``, or ``None``."""
    if raw in ("first", "last"):
        return cast(OrderByNullPlacement, raw)
    return None


def normalized_expr_from_stored_json(raw: Any) -> NormalizedExpr:
    """
    Coerce JSON or template `expr` payloads into a `NormalizedExpr`.

    Args:

        raw: String, dict, or existing `NormalizedExpr`.

    Returns:

        Normalised expression; empty expr if unsupported.
    """
    if isinstance(raw, str):
        return NormalizedExpr.from_column(raw)
    if isinstance(raw, dict):
        return NormalizedExpr.from_dict(raw)
    if isinstance(raw, NormalizedExpr):
        return raw
    return NormalizedExpr()


@dataclass
class ExprValue:
    """Parameterized literal value for expression arithmetic with param_key for template reuse."""

    value: float = 0.0
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
        if isinstance(d, int | float):
            return ExprValue(value=float(d))
        if isinstance(d, dict):
            return ExprValue(value=d.get("value", 0.0), param_key=d.get("param_key", ""))
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


def _coerce_mul_term(raw: Any) -> NormalizedExpr:
    """Coerce a multiply/divide list element to a `NormalizedExpr` leaf. Accepts a `NormalizedExpr` instance, a dict (round-trip), or a bare string. Bare strings that look like function calls or compound expressions are routed through the sqlglot-backed `parse_expr_string` parser to recover structural fields; simple identifiers become `column_ref` leaves."""
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
            fn = _PARSE_EXPR_STRING_FN
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
            except Exception:
                return NormalizedExpr(raw_sql=s)
        return NormalizedExpr(column_ref=s)
    return NormalizedExpr()


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
        """Coerce string entries to leaf NormalizedExpr, sort multiply/divide, and normalise function name casing/order."""
        self.multiply = sorted((_coerce_mul_term(t) for t in self.multiply), key=lambda e: e.signature_key)
        self.divide = sorted((_coerce_mul_term(t) for t in self.divide), key=lambda e: e.signature_key)
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
            multiply=[_coerce_mul_term(t) for t in d.get("multiply", [])],
            divide=[_coerce_mul_term(t) for t in d.get("divide", [])],
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
                OrderByCol.from_dict(o) if isinstance(o, dict) else OrderByCol(expr=parse_expr_string_for_json(o))
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

    def __post_init__(self) -> None:
        """Sort child groups/values and normalise outer function name. casing/order. Returns: None."""
        self.add_groups = sorted(self.add_groups, key=lambda g: g.signature_key)
        self.sub_groups = sorted(self.sub_groups, key=lambda g: g.signature_key)
        self.add_values = sorted(self.add_values, key=lambda v: v.value)
        self.sub_values = sorted(self.sub_values, key=lambda v: v.value)
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
        fn = _RENDER_EXPR_SQL_FN
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
        return {"expr": expr_prompt_sql(self)}

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Canonical ``expr`` field example for prompts."""
        return {"expr": "table.column"}


def expr_registry_ref(expr: NormalizedExpr) -> str | None:
    """Return the canonical registry id when *expr* is a bare ``column_ref`` matching ``^[wc]\\d{2}$``. A registry reference is conventionally encoded as a single bare ``column_ref`` with no other expression complexity. The rest of the system treats this leaf shape as the canonical way to point at a window or case registry entry from select, group_by, order_by, filter, or having."""
    if expr.string_literal:
        return None
    col = (expr.column_ref or "").strip()
    if not col or not REGISTRY_REF_TOKEN_RE.match(col):
        return None
    if expr.add_groups or expr.sub_groups or expr.add_values or expr.sub_values:
        return None
    if expr.agg_func or expr.scalar_func or expr.inner_scalar_func:
        return None
    if expr.star or expr.cast_type or expr.interval is not None:
        return None
    if expr.keyword or expr.raw_sql:
        return None
    return col


def expr_prompt_sql(expr: NormalizedExpr) -> str:
    """Render *expr* as the shorthand SQL string shown in LLM prompts. Registry references ``wNN`` / ``cNN`` emit as bare tokens; other expressions use the registered renderer when available."""
    ref = expr_registry_ref(expr)
    if ref:
        return ref
    if expr.string_literal:
        return expr.string_literal
    fn = _RENDER_EXPR_SQL_FN
    if fn is not None:
        try:
            return cast(str, fn(expr))
        except Exception:
            pass
    col = expr.primary_column
    return col if col else ""


def _canonicalize_predicate_sides(predicate: Any) -> None:
    """Enforce column-bearing side on the left and flip the operator when a swap is required. When exactly one of ``left_expr`` / ``right_expr`` contains column / aggregate / scalar / registry references, that side is moved to ``left_expr`` and the operator is flipped. When both sides are column-bearing or both are literal-only, sides are left untouched."""
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


def _normalize_predicate_op(raw: Any) -> Literal["and", "or"]:
    text = str(raw or "and").strip().lower()
    return "or" if text == "or" else "and"


_LEGACY_WHERE_GROUP_KEYS: tuple[str, ...] = ("where_group",)


def _legacy_where_group_raw(raw: Mapping[str, Any]) -> Any:
    for key in _LEGACY_WHERE_GROUP_KEYS:
        if key in raw:
            return raw[key]
    return None


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


@dataclass(frozen=True)
class PredicateGroup:
    """Boolean composition tree for WHERE/HAVING predicates."""

    op: Literal["and", "or"] = "and"
    predicates: tuple[WhereParam | HavingParam, ...] = ()
    groups: tuple[PredicateGroup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _normalize_predicate_op(self.op))

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
    def from_dict(d: dict[str, Any] | None, *, having: bool = False) -> PredicateGroup | None:
        if not d or not isinstance(d, dict):
            return None
        pred_cls = HavingParam if having else WhereParam
        preds = tuple(pred_cls.from_dict(item) if isinstance(item, dict) else item for item in d.get("predicates", []))
        groups = tuple(
            child for raw in d.get("groups", []) if (child := PredicateGroup.from_dict(raw, having=having)) is not None
        )
        group = PredicateGroup(op=_normalize_predicate_op(d.get("op", "and")), predicates=preds, groups=groups)
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


def _legacy_bool_op_from_stored(raw: Any) -> str:
    text = str(raw or "AND").strip().upper()
    return "OR" if text == "OR" else "AND"


def _clamp_negative_where_group(value: int | None) -> int | None:
    if value is not None and value < 0:
        return None
    return value


def _legacy_coerce_where_group_list(
    items: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
) -> list[tuple[WhereParam | HavingParam, str, int | None]]:
    if not items:
        return []
    normalized: list[tuple[WhereParam | HavingParam, str, int | None]] = []
    for pred, bool_op, where_group in items:
        normalized.append((pred, _legacy_bool_op_from_stored(bool_op), _clamp_negative_where_group(where_group)))
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


def _predicate_group_from_legacy_rows(
    rows: Sequence[tuple[WhereParam | HavingParam, str, int | None]],
) -> PredicateGroup | None:
    rows = _legacy_coerce_where_group_list(rows)
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
        connector = _normalize_predicate_op(rows[idx - 1][1])
        right = PredicateGroup(op="and", predicates=(rows[idx][0],))
        acc = _combine_predicate_groups(acc, rows[idx][0], connector, right)
    return acc


def _combine_predicate_groups(
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


def predicate_group_from_legacy_flat_where_list(filters: list[WhereParam]) -> PredicateGroup | None:
    rows = [(fp, "AND", None) for fp in filters]
    return _predicate_group_from_legacy_rows(rows)


def predicate_group_from_legacy_having(having: list[HavingParam]) -> PredicateGroup | None:
    rows = [(hp, "AND", None) for hp in having]
    return _predicate_group_from_legacy_rows(rows)


def predicate_group_from_legacy_flat_where_dicts(fp_raw: list[Any]) -> PredicateGroup | None:
    rows: list[tuple[WhereParam, str, int | None]] = []
    for raw in fp_raw:
        if not isinstance(raw, dict):
            continue
        fp = WhereParam.from_dict(raw)
        rows.append(
            (
                fp,
                _legacy_bool_op_from_stored(raw.get("bool_op", "AND")),
                _where_group_int_from_stored(_legacy_where_group_raw(raw)),
            )
        )
    return _predicate_group_from_legacy_rows(rows)


def predicate_group_from_legacy_having_dicts(hp_raw: list[Any]) -> PredicateGroup | None:
    rows: list[tuple[HavingParam, str, int | None]] = []
    for raw in hp_raw:
        if not isinstance(raw, dict):
            continue
        hp = HavingParam.from_dict(raw)
        rows.append(
            (
                hp,
                _legacy_bool_op_from_stored(raw.get("bool_op", "AND")),
                _where_group_int_from_stored(_legacy_where_group_raw(raw)),
            )
        )
    return _predicate_group_from_legacy_rows(rows)


def predicate_group_from_stored(raw: Any, *, legacy_key: str, having: bool = False) -> PredicateGroup | None:
    if isinstance(raw, PredicateGroup):
        return None if raw.is_empty() else raw
    if isinstance(raw, dict):
        group = PredicateGroup.from_dict(raw, having=having)
        return group
    if isinstance(raw, list):
        if having:
            return predicate_group_from_legacy_having_dicts(raw)
        return predicate_group_from_legacy_flat_where_dicts(raw)
    return None


def coerce_predicate_group(group: PredicateGroup | None) -> PredicateGroup | None:
    if group is None or group.is_empty():
        return None
    dnf = normalize_predicate_group_dnf(group)
    cnf = normalize_predicate_group_cnf(group)
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
    preference = _predicate_group_normalization_preference(group)
    preferred = [candidate for form, candidate in within_limit if form == preference]
    if preferred:
        return min(preferred, key=lambda candidate: candidate.depth())
    return min((candidate for _, candidate in within_limit), key=lambda candidate: candidate.depth())


def map_predicate_group(group: PredicateGroup | None, fn: Any) -> PredicateGroup | None:
    if group is None or group.is_empty():
        return None
    preds = tuple(fn(pred) for pred in group.predicates)
    subgroups = tuple(
        mapped for mapped in (map_predicate_group(child, fn) for child in group.groups) if mapped is not None
    )
    mapped_group = PredicateGroup(op=group.op, predicates=preds, groups=subgroups)
    return None if mapped_group.is_empty() else mapped_group


def reapply_predicate_leaves(
    group: PredicateGroup | None, leaves: Sequence[WhereParam | HavingParam]
) -> PredicateGroup | None:
    """Replace predicate leaves in *group* with *leaves* in traversal order."""
    if group is None or group.is_empty():
        return None
    leaf_iter = iter(leaves)
    return map_predicate_group(group, lambda _pred: next(leaf_iter))


def rebuild_predicate_group_from_leaves(
    original: PredicateGroup | None, leaves: Sequence[WhereParam | HavingParam]
) -> PredicateGroup | None:
    """Preserve *original* tree shape when leaf counts match; otherwise flatten to AND."""
    if not leaves:
        return None
    if original is not None and len(leaves) == len(original.leaves()):
        return reapply_predicate_leaves(original, leaves)
    return predicate_group_from_list(list(leaves))


def partition_predicate_group(
    group: PredicateGroup | None, keep_fn: Any
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
        kept_child, dropped_child = partition_predicate_group(child, keep_fn)
        if kept_child is not None:
            kept_groups.append(kept_child)
        if dropped_child is not None:
            dropped_groups.append(dropped_child)

    kept = PredicateGroup(op=group.op, predicates=tuple(kept_preds), groups=tuple(kept_groups))
    dropped = PredicateGroup(op=group.op, predicates=tuple(dropped_preds), groups=tuple(dropped_groups))
    return (None if kept.is_empty() else kept, None if dropped.is_empty() else dropped)


def merge_predicate_groups(op: Literal["and", "or"], groups: Sequence[PredicateGroup | None]) -> PredicateGroup | None:
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


def parse_where_field(d: Mapping[str, Any]) -> PredicateGroup | None:
    if "where" in d:
        raw = d.get("where")
        if raw is None:
            return None
        return PredicateGroup.from_dict(raw) if isinstance(raw, dict) else None
    if "where_param" in d:
        return predicate_group_from_legacy_flat_where_dicts(d.get("where_param", []))
    return None


def parse_having_field(d: Mapping[str, Any]) -> PredicateGroup | None:
    if "having" in d:
        raw = d.get("having")
        if raw is None:
            return None
        return PredicateGroup.from_dict(raw, having=True) if isinstance(raw, dict) else None
    if "having_param" in d:
        return predicate_group_from_legacy_having_dicts(d.get("having_param", []))
    return None


def predicate_leaves(group: PredicateGroup | None) -> list[WhereParam | HavingParam]:
    return group.leaves() if group else []


def where_leaves(group: PredicateGroup | None) -> list[WhereParam]:
    """Return flat WHERE predicate leaves with a narrowed type."""
    if group is None:
        return []
    return [leaf for leaf in group.leaves() if isinstance(leaf, WhereParam)]


def having_leaves(group: PredicateGroup | None) -> list[HavingParam]:
    """Return flat HAVING predicate leaves with a narrowed type."""
    if group is None:
        return []
    return [leaf for leaf in group.leaves() if isinstance(leaf, HavingParam)]


def predicate_group_and(*preds: WhereParam | HavingParam) -> PredicateGroup | None:
    items = tuple(pred for pred in preds if pred is not None)
    if not items:
        return None
    return PredicateGroup(op="and", predicates=items)


def predicate_group_from_list(items: Sequence[WhereParam | HavingParam] | None) -> PredicateGroup | None:
    if not items:
        return None
    return PredicateGroup(op="and", predicates=tuple(items))


def coerce_having_group_list(having: list[HavingParam]) -> list[HavingParam]:
    """Deprecated: convert a legacy flat HAVING list into a predicate group and back."""
    group = predicate_group_from_legacy_having(having)
    return having_leaves(group)


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
            expr = parse_expr_string_for_json(expr_raw)
        elif isinstance(expr_raw, dict):
            expr = NormalizedExpr.from_dict(expr_raw)
        elif isinstance(expr_raw, NormalizedExpr):
            expr = expr_raw
        else:
            expr = NormalizedExpr()
        return OrderByCol(
            expr=expr,
            direction=d.get("direction", "ASC"),
            nulls=coerce_order_by_nulls(d.get("nulls")),
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
            "expr": expr_prompt_sql(self.expr),
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
            _canonicalize_predicate_sides(self)
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
            offset = sum(ev.value for ev in self.left_expr.add_values) - sum(
                ev.value for ev in self.left_expr.sub_values
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
            left_expr=normalized_expr_from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(normalized_expr_from_stored_json(right_raw) if right_raw else None),
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
            "left_expr": expr_prompt_sql(self.left_expr),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = expr_prompt_sql(self.right_expr)
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
            _canonicalize_predicate_sides(self)
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
            offset = sum(ev.value for ev in self.left_expr.add_values) - sum(
                ev.value for ev in self.left_expr.sub_values
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
            left_expr=normalized_expr_from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(normalized_expr_from_stored_json(right_raw) if right_raw else None),
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
            "left_expr": expr_prompt_sql(self.left_expr),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = expr_prompt_sql(self.right_expr)
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


def _cnf_or_clauses(group: PredicateGroup) -> list[PredicateGroup]:
    clauses: list[PredicateGroup] = []
    for pred in group.predicates:
        clauses.append(PredicateGroup(op="or", predicates=(pred,)))
    for child in group.groups:
        if child.op == "or":
            clauses.append(child)
        elif child.op == "and":
            clauses.extend(_cnf_or_clauses(child))
    return clauses


def _cnf_normalize_and_disjunct(group: PredicateGroup) -> PredicateGroup:
    """Normalize an AND disjunct nested under OR without CNF-expanding its predicates."""
    and_preds: list[WhereParam | HavingParam] = list(group.predicates)
    nested: list[PredicateGroup] = []
    for child in group.groups:
        normalized = normalize_predicate_group_cnf(child)
        if normalized is None or normalized.is_empty():
            continue
        if normalized.op == "or":
            nested.append(normalized)
        else:
            and_preds.extend(normalized.predicates)
            nested.extend(normalized.groups)
    return PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))


def _predicate_group_normalization_preference(group: PredicateGroup) -> Literal["dnf", "cnf", "auto"]:
    if group.op == "and" and any(child.op == "or" for child in group.groups):
        return "cnf"
    if group.op == "or" and any(child.op == "and" for child in group.groups):
        return "dnf"
    return "auto"


def normalize_predicate_group_dnf(group: PredicateGroup | None) -> PredicateGroup | None:
    """Normalize a predicate tree toward OR-of-AND (DNF) form when possible."""
    if group is None or group.is_empty():
        return None
    if group.op == "or":
        groups = tuple(normalize_predicate_group_dnf(child) or child for child in group.groups)
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
    elif group.op == "and":
        and_preds: list[WhereParam | HavingParam] = list(group.predicates)
        nested: list[PredicateGroup] = []
        for child in group.groups:
            normalized = normalize_predicate_group_dnf(child)
            if normalized is None or normalized.is_empty():
                continue
            if normalized.op == "or":
                nested.append(normalized)
            else:
                and_preds.extend(normalized.predicates)
                nested.extend(normalized.groups)
        return PredicateGroup(op="and", predicates=tuple(and_preds), groups=tuple(nested))


def normalize_predicate_group_cnf(group: PredicateGroup | None) -> PredicateGroup | None:
    """Normalize a predicate tree toward AND-of-OR (CNF) form when possible."""
    if group is None or group.is_empty():
        return None
    if group.op == "and":
        or_clauses: list[PredicateGroup] = []
        for pred in group.predicates:
            or_clauses.append(PredicateGroup(op="or", predicates=(pred,)))
        for child in group.groups:
            normalized = normalize_predicate_group_cnf(child)
            if normalized is None or normalized.is_empty():
                continue
            if normalized.op == "or":
                or_clauses.append(normalized)
            else:
                or_clauses.extend(_cnf_or_clauses(normalized))
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
                nested_or.append(_cnf_normalize_and_disjunct(child))
                continue
            normalized = normalize_predicate_group_cnf(child)
            if normalized is None or normalized.is_empty():
                continue
            or_preds.extend(normalized.predicates)
            nested_or.extend(normalized.groups)
        result = PredicateGroup(op="or", predicates=tuple(or_preds), groups=tuple(nested_or))
        return None if result.is_empty() else result
