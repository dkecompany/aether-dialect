"""Shared dataclasses and enums for schema graphs, validation, templates, QSim skeletons, and type helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Literal

import pandas

from ._config import (
    COLUMN_TYPE_TO_VALUE_TYPE,
    CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC,
    DATE_TYPE_TOKENS,
    DEFAULT_RANDOM_SEED,
    EXCLUDED_FILTER_PATTERNS,
    HIDDEN_SENSITIVITIES,
    NUMERIC_TYPE_TOKENS,
    ROLE_ALLOWED_AGGREGATIONS,
    STRING_TYPE_TOKENS,
    LlmExecutionConfig,
    PolicyConfig,
    normalize_column_type,
)

SchemaInclude = Literal["tables", "views", "both"]


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


class SensitivityClassification(str, Enum):
    """Single-column sensitivity tier for projection, filtering, and LLM visibility."""

    NONE = "none"
    HYGIENE = "hygiene"
    STRICT = "strict"
    FORBIDDEN = "forbidden"


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


def sensitivity_classification_from_legacy_fields(sensitivity: Any, pii: Any) -> SensitivityClassification:
    """
    Map legacy ``sensitivity`` string plus optional ``pii`` tier to :class:`SensitivityClassification`.

    Documents that stored ``\"pii\"`` without a ``pii`` tier behave as :attr:`SensitivityClassification.STRICT`.
    """

    s = str(sensitivity).strip().lower() if sensitivity is not None else ""
    if s == "restricted":
        return SensitivityClassification.FORBIDDEN
    if s == "pii":
        tier = coerce_sensitivity_classification(pii)
        if tier is None or tier == SensitivityClassification.NONE:
            return SensitivityClassification.STRICT
        if tier in (
            SensitivityClassification.HYGIENE,
            SensitivityClassification.STRICT,
            SensitivityClassification.FORBIDDEN,
        ):
            return tier
        return SensitivityClassification.STRICT
    if s in ("", "none"):
        return SensitivityClassification.NONE
    direct = coerce_sensitivity_classification(sensitivity)
    if direct is not None:
        return direct
    return SensitivityClassification.NONE


def column_sensitivity_from_dict(d: Mapping[str, Any]) -> SensitivityClassification:
    """Resolve persisted ``sensitivity`` / ``pii`` keys into a single :class:`SensitivityClassification`."""

    sens = d.get("sensitivity")
    pii = d.get("pii")
    s = str(sens).strip().lower() if sens is not None else ""
    if s in ("pii", "restricted") or pii is not None:
        return sensitivity_classification_from_legacy_fields(sens, pii)
    return coerce_sensitivity_classification(sens) or SensitivityClassification.NONE


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
class AdvancedFeatureSpec:
    """Named advanced SQL intent capability surfaced to tier-conditioned QSim prompts."""

    feature_id: str
    summary: str
    example_fragment: str


@dataclass(frozen=True)
class ComplexityTierSpec:
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


def _tier_feasible_for_capability(tier_key: str, cap: DatabaseFeatureCapability) -> bool:
    """
    Return whether a complexity tier remains achievable on this database snapshot.

    Args:

        tier_key: One of ``simple``, ``moderate``, ``complex``, ``highly_complex``.

        cap: Capability snapshot from :func:`compute_database_feature_capability`.

    Returns:

        False when structural prerequisites for that tier are absent.
    """

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
    proportions: Mapping[str, float],
    cap: DatabaseFeatureCapability,
) -> dict[str, float]:
    """
    Zero unreachable tier mass and renormalize remaining targets for QSim and warmup budgets.

    Args:

        proportions: Named tier weights summing to approximately one.

        cap: Live capability snapshot for structural feasibility.

    Returns:

        Renormalized tier weights over feasible tiers only.
    """

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
class SurfaceTemplateSpec:
    """Declarative NL surface pattern for deterministic warmup anchoring."""

    construct_kind: str
    surface_forms: tuple[str, ...]


QSIM_SUPPORTED_ADVANCED_FEATURES: tuple[AdvancedFeatureSpec, ...] = (
    AdvancedFeatureSpec(
        feature_id="multi_cte_chain",
        summary="Stacked CTE definitions where one CTE references another.",
        example_fragment="WITH daily AS (...), rolled AS (SELECT ... FROM daily) SELECT ...",
    ),
    AdvancedFeatureSpec(
        feature_id="scalar_cte_bridge",
        summary="Scalar-valued CTE row merged via cross join for threshold constants.",
        example_fragment="WITH params AS (SELECT 100.0 AS min_amt) SELECT ... FROM orders CROSS JOIN params",
    ),
    AdvancedFeatureSpec(
        feature_id="self_join_via_cte",
        summary="Second reference to a base table mediated through a named CTE.",
        example_fragment="WITH o1 AS (SELECT * FROM orders) SELECT ... FROM orders JOIN o1 ON ...",
    ),
    AdvancedFeatureSpec(
        feature_id="window_partition_order",
        summary="ROW_NUMBER/RANK/SUM over PARTITION BY with ORDER BY.",
        example_fragment="ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC)",
    ),
    AdvancedFeatureSpec(
        feature_id="case_when_select",
        summary="CASE expressions in the projected SELECT list.",
        example_fragment="CASE WHEN status = 'open' THEN amount ELSE 0 END",
    ),
    AdvancedFeatureSpec(
        feature_id="date_window_filter",
        summary="Rolling calendar predicates using relative windows.",
        example_fragment="order_date >= CURRENT_DATE - INTERVAL '30 days'",
    ),
    AdvancedFeatureSpec(
        feature_id="date_diff_shapes",
        summary="Difference-between-dates filters.",
        example_fragment="DATE_PART('day', shipped_at - ordered_at) > 3",
    ),
    AdvancedFeatureSpec(
        feature_id="unnest_array_column",
        summary="EXPLODE/UNNEST typed array columns inside a subordinate SELECT.",
        example_fragment="FROM tags CROSS JOIN UNNEST(tag_ids) AS u(tag)",
    ),
    AdvancedFeatureSpec(
        feature_id="distinct_select",
        summary="SELECT DISTINCT non-aggregated projections.",
        example_fragment="SELECT DISTINCT country FROM customers",
    ),
    AdvancedFeatureSpec(
        feature_id="ilike_predicate",
        summary="ILIKE / NOT ILIKE text predicates.",
        example_fragment="note ILIKE '%refund%'",
    ),
    AdvancedFeatureSpec(
        feature_id="having_aggregate_compare",
        summary="HAVING clauses comparing aggregated measures to literals.",
        example_fragment="HAVING SUM(amount) > 5000",
    ),
)

QSIM_COMPLEXITY_TIER_SPECS: tuple[ComplexityTierSpec, ...] = (
    ComplexityTierSpec(
        tier=ComplexityTier.SIMPLE,
        min_tables=1,
        max_tables=1,
        max_cte_steps=0,
        allows_window=False,
        allows_multi_cte=False,
        summary="Single-table scans with optional equality or range filters; projections stay non-aggregated.",
        example_sketch="Show recent invoices for account 42.",
    ),
    ComplexityTierSpec(
        tier=ComplexityTier.MODERATE,
        min_tables=1,
        max_tables=2,
        max_cte_steps=0,
        allows_window=False,
        allows_multi_cte=False,
        summary="One or two joined tables with simple aggregates, light grouping, ORDER BY, or LIMIT.",
        example_sketch="Top 50 customers by orders last month.",
    ),
    ComplexityTierSpec(
        tier=ComplexityTier.COMPLEX,
        min_tables=2,
        max_tables=3,
        max_cte_steps=1,
        allows_window=True,
        allows_multi_cte=False,
        summary="Multi-table joins with grouped aggregates, HAVING, DISTINCT, or cross-column comparisons.",
        example_sketch="Average fulfillment hours by warehouse having more than 100 shipments.",
    ),
    ComplexityTierSpec(
        tier=ComplexityTier.HIGHLY_COMPLEX,
        min_tables=3,
        max_tables=3,
        max_cte_steps=3,
        allows_window=True,
        allows_multi_cte=True,
        summary="Dense shapes combining multiple predicates, aggregates at aligned grains, and ordered analytic heads.",
        example_sketch="Rank districts within each territory by margin contribution using layered filters.",
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
    FILTER_AGGREGATION = "filter_aggregation"
    FILTER_SEMANTIC = "filter_semantic"
    FILTER_STRUCTURE = "filter_structure"
    FILTER_VALIDITY = "filter_validity"
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
    MISSING_NUMERIC_FILTER = "missing_numeric_filter"
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
    WRONG_FILTER_LOGIC = "wrong_filter_logic"
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
    """Structured codes emitted by AST validation and EXPLAIN-plan diagnostics for both PostgreSQL and Databricks dialects."""

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


def _norm_schema_identifier(name: str, *, what: str) -> str:
    """Lowercase and strip *name*; raise when empty after strip."""

    s = str(name).strip().lower()
    if not s:
        raise ValueError(f"{what} must be non-empty")
    return s


class ConfigError(ValueError):
    """Raised when environment variables or static configuration are missing or contradictory."""


class Text2SQLError(ValueError):
    """Base class for recoverable Text2SQL engine lifecycle failures."""


class MigrationPendingError(Text2SQLError):
    """Init terminated because schema_migration_map.json is required, malformed, missing after export, or conflicts with validation."""


class ConnectionError(OSError):
    """Raised when the database driver rejects a connection attempt."""


class RetryableError(Exception):
    """
    Marker base class for transient failures that may succeed on retry.

    Concrete subclasses combine this marker with :class:`ConnectionError`,
    :class:`RuntimeError`, etc. Integrators may use ``isinstance(exc, RetryableError)``
    without inspecting messages.
    """


class DatabasePingFailed(ConnectionError, RetryableError):
    """Raised when a trivial ``SELECT 1`` ping fails after retries (network blips, overload)."""


class LlmTransientFailure(RuntimeError, RetryableError):
    """LLM request failed after retries due to rate limits, timeouts, or connection resets."""


class StatementTimeoutError(RuntimeError, RetryableError):
    """Raised when the database aborts work due to ``statement_timeout`` or warehouse timeouts."""


class SchemaAccessError(ValueError):
    """Raised when credentials cannot read the requested scope or the graph is unusable."""


class SessionActiveError(RuntimeError):
    """Raised when ``PipelineSession.ask`` is called while a turn is already in progress."""


class SchemaInvariantError(RuntimeError):
    """
    Raised by :func:`assert_schema_invariants` when the canonical containers fall out of sync.

    Indicates a programmer error elsewhere in the build pipeline: e.g. an FK referencing a missing column, a PK column missing from its table, an unwired column-table back reference, or a stale canonical-bearer index. Always indicates the offending source-of-truth has been violated and never represents a recoverable runtime condition.
    """


class MigrationTier(str, Enum):
    """Classified migration severity between a stored artifact fingerprint and the live graph."""

    NO_CHANGE = "no_change"
    SOFT_REFRESH = "soft_refresh"
    REMAP = "remap"
    DESTRUCTIVE = "destructive"


class ColumnVisibilityBlockReason(str, Enum):
    """Machine-stable reason a column is blocked from LLM exposure or reference validation."""

    DENIED = "denied"
    NOT_IN_ALLOW_COLUMNS = "not_in_allow_columns"
    SENSITIVE_PII = "sensitive_pii"
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
    """User-authored migration mapping consumed once at init time.

    When ``refresh_existing_descriptions_on_addition`` is true and the live schema diff includes newly added
    tables, the engine runs a full-graph LLM classifier pass after subset profiling and merges refreshed
    table/column descriptions and roles only for tables that were otherwise unchanged by the diff.
    """

    version: int
    action: str
    table_renames: tuple[SchemaMigrationMapEntry, ...]
    column_renames: tuple[SchemaMigrationMapEntry, ...]
    dropped_tables: tuple[str, ...]
    dropped_columns: tuple[SchemaMigrationMapEntry, ...]
    added_tables: tuple[str, ...]
    added_columns: tuple[SchemaMigrationMapEntry, ...]
    refresh_existing_descriptions_on_addition: bool = False


@dataclass(frozen=True, slots=True)
class OverrideSkip:
    """One JSON entry that was rejected during ``Text2SQL.apply_schema_overrides``."""

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
    """Summary of edits produced by ``Text2SQL.apply_schema_overrides``."""

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
class MigrationPreview:
    """Human-readable outcome of comparing a skeleton against the live prepared schema."""

    tier: Literal["compatible", "remap", "destructive"]
    affected_tables: tuple[str, ...]
    affected_columns: tuple[tuple[str, str], ...]
    skeleton_path: str


@dataclass(frozen=True, slots=True)
class SessionStep:
    """Single observable point in a programmatic interactive turn.

    Carries whether the turn has finished, a short instruction string, a stage discriminant, optional SQL, tabular data, a free-form body, and an error string when the engine fails.

    done:

        True when the pipeline finished successfully or ended in a terminal error; False when the caller must respond via ``PipelineSession.step``.

    prompt:

        The short line the interactive layer should show immediately before collecting input (for example yes or no, or a free-text rejection reason prompt).

    kind:

        Stable stage identifier matching the active suspend kind or a terminal sentinel; used to branch programmatic UIs without parsing ``prompt``.

    sql:

        The formatted SQL under discussion when the step pertains to execution or confirmation; otherwise None.

    data:

        Row-level query preview or full result as a ``pandas.DataFrame``; None for scalar outcomes, previews trimmed to five rows at suspend boundaries, and the full frame on the terminal acceptance step when rows exist.

    message:

        Multi-line contextual body: consolidated intent confirmation, migration DDL, rejection guidance, or a rendered scalar value; empty or None when nothing extra should print beyond ``prompt`` and ``data``.

    error:

        Terminal failure explanation when ``done`` is True and processing stopped; otherwise None.

    intent_summary:

        Structured intent headline when the step reflects a parsed intent or later pipeline stages; otherwise None.

    diagnostics:

        Structured diagnostics captured during this step (from ``notify`` / ``debug`` when a collector is active).

    status:

        On terminal error steps, a coarse failure category name (same string values as :class:`FailureCategory`); None on success or non-terminal steps.

    reply_shape:

        When ``done`` is False, whether the caller should collect a yes or no token or free text; None on terminal steps.

    semantic_warnings:

        Normalised warning strings for intent confirmation, often empty on non-intent suspend steps.
    """

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


@dataclass(frozen=True, slots=True)
class WriteQueueEvent:
    """Structured event a reader-mode session records for a writer to apply later.

    kind: Discriminator selecting which writer-side handler applies (template accept or reject, paraphrase emission, override proposal materialisation, or question feedback).

    schema_hash: Structural hash of the schema graph at event creation; the writer drops events when this value no longer matches the live graph.

    produced_at: ISO-8601 timestamp string when the reader enqueued the event.

    payload: Ordered key-value pairs serialising handler-specific fields; a tuple of pairs keeps the event hashable and avoids dict key-order ambiguity across processes.
    """

    kind: Literal[
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    ]
    schema_hash: str
    produced_at: str
    payload: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Lifecycle audit record for integrator sinks."""

    event_type: str
    timestamp_iso: str
    question: str | None
    schema_hash: str | None
    provider: Literal["openai", "azure"]
    details: tuple[tuple[str, str], ...] = ()


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
class SchemaContext:
    """Frozen schema scope: optional explicit relation names, include mode, deny lists, and paths."""

    allow_objects: frozenset[str] = frozenset()
    include: SchemaInclude = "tables"
    deny_columns: frozenset[str] = frozenset()
    allow_columns: frozenset[str] = frozenset()
    notes_file: str | None = None
    sql_file: str | None = None

    def __post_init__(self) -> None:
        if self.include not in ("tables", "views", "both"):
            raise ConfigError(f"include must be 'tables', 'views', or 'both', not {self.include!r}")
        if self.notes_file is not None and not str(self.notes_file).strip():
            raise ConfigError("notes_file must be omitted or a non-empty path")
        if self.sql_file is not None and not str(self.sql_file).strip():
            raise ConfigError("sql_file must be omitted or a non-empty path")
        allow_norm = frozenset(_norm_schema_identifier(t, what="allow_objects entry") for t in self.allow_objects)
        normalized_specs: list[str] = []
        for spec in self.deny_columns:
            raw = str(spec).strip()
            dot_count = raw.count(".")
            if dot_count != 1:
                raise ConfigError(CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC.format(spec=spec))
            tbl_raw, col_raw = raw.split(".", 1)
            tbl = _norm_schema_identifier(tbl_raw, what="deny_columns table")
            col = _norm_schema_identifier(col_raw, what="deny_columns column")
            if tbl == "*":
                normalized_specs.append(f"*.{col}")
            else:
                normalized_specs.append(f"{tbl}.{col}")
        col_set = frozenset(normalized_specs)
        allow_col_specs: list[str] = []
        for spec in self.allow_columns:
            raw = str(spec).strip()
            dot_count = raw.count(".")
            if dot_count != 1:
                raise ConfigError(CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC.format(spec=spec))
            tbl_raw, col_raw = raw.split(".", 1)
            tbl = _norm_schema_identifier(tbl_raw, what="allow_columns table")
            col = _norm_schema_identifier(col_raw, what="allow_columns column")
            if tbl == "*":
                allow_col_specs.append(f"*.{col}")
            else:
                allow_col_specs.append(f"{tbl}.{col}")
        allow_col_set = frozenset(allow_col_specs)
        for t in allow_norm:
            for spec in col_set:
                dt, _, _rest = spec.partition(".")
                if dt != "*" and dt == t:
                    raise ConfigError(f"allow_objects entry {t!r} is denied by deny_columns entry {spec!r}")
        object.__setattr__(self, "allow_objects", allow_norm)
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
class CteIntent:
    """Planner-only natural-language description of one reusable intermediate aligned with a runtime CTE step."""

    name: str
    depends_on: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    select: str = ""
    filter: str = ""
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
    filter: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str | None = None
    window: str = ""
    case: str = ""
    cte_steps: tuple[CteIntent, ...] = ()
    schema_invalid: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Engine, artifact root, frozen schema scope, and merged LLM plus execution limits for runtime introspection."""

    engine: Literal["postgresql", "databricks"]
    artifacts_dir: str
    schema_context: SchemaContext
    llm_execution: LlmExecutionConfig


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Active LLM provider label after environment configuration."""

    provider: Literal["openai", "azure"]


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


def is_numeric_value_type(value_type: str | None) -> bool:
    """Return True when normalized intent ``value_type`` is numeric (integer or number)."""

    if not value_type:
        return False
    vt = value_type.strip().lower()
    return vt in ("integer", "number")


def is_integer_value_type(value_type: str | None) -> bool:
    """Return True when ``value_type`` is integer."""

    if not value_type:
        return False
    return value_type.strip().lower() == "integer"


def is_temporal_value_type(value_type: str | None) -> bool:
    """Return True when ``value_type`` denotes a date or time column."""

    if not value_type:
        return False
    vt = value_type.strip().lower()
    return vt in ("date", "timestamp", "time", "datetime")


def is_string_value_type(value_type: str | None) -> bool:
    """Return True when ``value_type`` is string."""

    if not value_type:
        return False
    return value_type.strip().lower() == "string"


def is_boolean_value_type(value_type: str | None) -> bool:
    """Return True when ``value_type`` is boolean."""

    if not value_type:
        return False
    return value_type.strip().lower() == "boolean"


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
        WorkloadFamily.STATUS_REPORT,
        ComplexityTier.MODERATE,
        "rank_none",
        "none",
        "small",
        ("fact",),
    ),
    WorkloadFamily.BREAKDOWN: WorkloadFamilySpec(
        WorkloadFamily.BREAKDOWN,
        ComplexityTier.MODERATE,
        "none",
        "categorical_slice",
        "medium",
        ("fact", "dimension"),
    ),
    WorkloadFamily.LEADERBOARD: WorkloadFamilySpec(
        WorkloadFamily.LEADERBOARD,
        ComplexityTier.MODERATE,
        "top_k",
        "ordered_metric",
        "small",
        ("fact",),
    ),
    WorkloadFamily.TREND: WorkloadFamilySpec(
        WorkloadFamily.TREND,
        ComplexityTier.COMPLEX,
        "time_series",
        "temporal_sequence",
        "medium",
        ("fact",),
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
        WorkloadFamily.SHARE_OF_TOTAL,
        ComplexityTier.COMPLEX,
        "ratio",
        "part_whole",
        "small",
        ("fact", "dimension"),
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
        WorkloadFamily.EXTRACT,
        ComplexityTier.SIMPLE,
        "none",
        "none",
        "many",
        ("fact",),
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
        WorkloadFamily.EXPLORATION_FOLLOWUP,
        ComplexityTier.SIMPLE,
        "none",
        "ad_hoc",
        "many",
        ("fact",),
    ),
}


class InferenceTag(str, Enum):
    """
    Provenance tag for an :class:`FKEdge`.

    A catalog-declared edge is represented by ``None`` rather than a member of this enum so that presence-of-tag and identity-of-inferred-layer are reflected by a single attribute. Inherits ``str`` so members compare equal to their wire value and round-trip through JSON without custom encoding.
    """

    SUFFIX = "suffix"
    SELF = "self"
    COMPOSITE = "composite"
    SEMANTIC = "semantic"
    SEMANTIC_PROMOTED = "semantic_promoted"
    USER_STRUCTURAL = "user_override_structural"
    USER_SEMANTIC = "user_override_semantic"


_INFERENCE_TAG_VALUES: frozenset[str] = frozenset(t.value for t in InferenceTag)


def coerce_inference_tag(raw: object) -> InferenceTag | None:
    """
    Normalise raw cache or override input into :class:`InferenceTag` (``None`` for catalog).

    Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value.
    """

    if raw is None or raw == "":
        return None
    if isinstance(raw, InferenceTag):
        return raw
    if isinstance(raw, str) and raw in _INFERENCE_TAG_VALUES:
        return InferenceTag(raw)
    raise ValueError(f"unknown FK inference_tag: {raw!r}")


class PkInferenceTag(str, Enum):
    """
    Provenance tag for an inferred or user-supplied primary key.

    Catalog primary keys are represented by ``None``; only inferred or user-injected keys carry a member of this enum.
    """

    PROFILE = "profile"
    USER_OVERRIDE = "user_override"


_PK_INFERENCE_TAG_VALUES: frozenset[str] = frozenset(t.value for t in PkInferenceTag)


def coerce_pk_inference_tag(raw: object) -> PkInferenceTag | None:
    """
    Normalise raw cache or override input into :class:`PkInferenceTag` (``None`` for catalog).

    Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value.
    """

    if raw is None or raw == "":
        return None
    if isinstance(raw, PkInferenceTag):
        return raw
    if isinstance(raw, str) and raw in _PK_INFERENCE_TAG_VALUES:
        return PkInferenceTag(raw)
    raise ValueError(f"unknown pk_inference_tag: {raw!r}")


class RoleOwner(str, Enum):
    """
    Provenance for the writer that last set :attr:`ColumnMetadata.role`.

    The members are ordered by ascending precedence: a writer with strictly greater precedence may overwrite a role assigned by a lower-precedence owner, while equal-or-lower-precedence writers must skip the column. PK/FK coercion is treated as the highest authority because it is required for join correctness; user overrides win over LLM inference, which in turn wins over profile heuristics, which in turn wins over the default catalog fallback.
    """

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

_ROLE_OWNER_VALUES: frozenset[str] = frozenset(o.value for o in RoleOwner)


def coerce_role_owner(raw: object) -> RoleOwner | None:
    """
    Normalise raw cache or override input into :class:`RoleOwner` (``None`` when unset).

    Raises ``ValueError`` when *raw* is a non-empty string that does not match any enum value.
    """

    if raw is None or raw == "":
        return None
    if isinstance(raw, RoleOwner):
        return raw
    if isinstance(raw, str) and raw in _ROLE_OWNER_VALUES:
        return RoleOwner(raw)
    raise ValueError(f"unknown role_owner: {raw!r}")


def can_overwrite_role(current: RoleOwner | None, candidate: RoleOwner) -> bool:
    """
    Return whether a writer with provenance *candidate* may overwrite a role currently owned by *current*.

    A column whose role has never been claimed (``current is None``) accepts any writer. Otherwise the candidate must have strictly greater precedence than the incumbent owner; equal-precedence writes are rejected so the first writer of a given tier wins deterministically.
    """

    if current is None:
        return True
    return _ROLE_OWNER_PRECEDENCE[candidate] > _ROLE_OWNER_PRECEDENCE[current]


class DescriptionOwner(str, Enum):
    """
    Provenance for the writer that last set a description on a table or column.

    Members are ordered by ascending precedence; the dedicated helper :func:`set_description` enforces a strict-greater-precedence rule so a later writer can only overwrite an existing description when its provenance outranks the incumbent owner.
    """

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

_DESCRIPTION_OWNER_VALUES: frozenset[str] = frozenset(o.value for o in DescriptionOwner)


def _coerce_description_owner(raw: object) -> DescriptionOwner | None:
    """Normalise raw cache or override input into :class:`DescriptionOwner` (``None`` when unset)."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, DescriptionOwner):
        return raw
    if isinstance(raw, str) and raw in _DESCRIPTION_OWNER_VALUES:
        return DescriptionOwner(raw)
    raise ValueError(f"unknown description_owner: {raw!r}")


def set_description(target: Any, text: str | None, owner: DescriptionOwner) -> bool:
    """
    Single writer for ``description`` on tables and columns.

    Args:
        target: Either a :class:`TableMetadata` or :class:`ColumnMetadata` instance.

        text: New description text (``None`` rejects the write; empty string clears to ``\"\"``).

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
    """
    Single writer for :attr:`ColumnMetadata.sensitivity`.

    Accepts :class:`SensitivityClassification`, legacy ``\"pii\"`` / ``\"restricted\"`` strings from classifiers, or ``None`` for :attr:`SensitivityClassification.NONE`. Clears concrete profile samples whenever the classification is not :attr:`SensitivityClassification.NONE`.
    """

    if value is None or (isinstance(value, str) and not str(value).strip()):
        resolved = SensitivityClassification.NONE
    elif isinstance(value, SensitivityClassification):
        resolved = value
    else:
        sv = str(value).strip().lower()
        if sv == "restricted":
            resolved = SensitivityClassification.FORBIDDEN
        elif sv == "pii":
            resolved = SensitivityClassification.STRICT
        else:
            resolved = coerce_sensitivity_classification(value) or SensitivityClassification.NONE
    col.sensitivity = resolved
    if resolved != SensitivityClassification.NONE:
        col.top_k_values = []
        col.min_val = None
        col.max_val = None


@dataclass
class FKEdge:
    """Foreign key relationship between two tables."""

    src_table: str
    src_cols: list[str]
    dst_table: str
    dst_cols: list[str]
    inference_tag: InferenceTag | None = None

    def __post_init__(self) -> None:
        """Coerce ``inference_tag`` from raw cache strings into :class:`InferenceTag`."""
        if not isinstance(self.inference_tag, InferenceTag):
            self.inference_tag = coerce_inference_tag(self.inference_tag)


@dataclass
class CatalogTableStructuralConstraints:
    """
    Catalog-sourced primary-key column names, foreign-key edges, and single-column unique names for one table.

    Each :class:`FKEdge` carries ``src_table`` equal to the referencing table so the bundle can be converted into ``tables_meta`` foreign-key dicts without losing the child table identity.
    """

    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[FKEdge] = field(default_factory=list)
    unique_columns: list[str] = field(default_factory=list)


@dataclass
class CatalogStructuralConstraintsIndex:
    """
    Per-table structural constraint bundles keyed by lowercased relation name within one catalog schema.

    When ``tables`` is empty the caller should treat catalog reflection as unavailable and continue with DDL-based parsing.
    """

    tables: dict[str, CatalogTableStructuralConstraints] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CatalogStructuralConstraintsIndex:
        """
        Construct an empty index for failed information_schema queries.

        Returns:

            Empty :class:`CatalogStructuralConstraintsIndex` instance.
        """

        return cls(tables={})


@dataclass
class ValueDomain:
    """Value domain for sampling concrete values during question generation."""

    values: list[str] = field(default_factory=list)
    min_val: str | None = None
    max_val: str | None = None
    data_type: str | None = None
    value_type: str = ""

    def __post_init__(self) -> None:
        """Derive ``value_type`` from ``data_type`` when unset."""

        if not self.value_type and self.data_type:
            self.value_type = data_type_to_value_type(self.data_type)


@dataclass
class ColumnMetadata:
    """
    Consolidated column metadata with profile, role, and value domain.

    Holds counts, overrides, filter/aggregation rules, and boolean hints used by validation and generation.
    """

    name: str
    data_type: str
    enum_type_name: str | None = None
    is_primary_key: InitVar[bool] = False
    is_foreign_key: InitVar[bool] = False
    fk_target: InitVar[tuple[str, str] | None] = None
    role: str | None = None
    value_type: str = ""
    row_count: int = 0
    distinct_count: int = 0
    distinct_from_sample: bool = False
    distinct_ratio: float = 0.0
    null_ratio: float = 0.0
    min_val: str | None = None
    max_val: str | None = None
    top_k_values: list[str] = field(default_factory=list)
    is_aggregatable_override: bool | None = None
    is_groupable_override: bool | None = None
    is_filterable_override: bool | None = None
    valid_filter_ops: list[str] = field(default_factory=list)
    valid_aggregations: list[str] = field(default_factory=list)
    valid_having_ops: list[str] = field(default_factory=list)
    description: str = ""
    description_owner: DescriptionOwner | None = None
    is_unique: bool = False
    sensitivity: SensitivityClassification = SensitivityClassification.NONE
    element_type: str | None = None
    is_nullable: bool = True
    semantic_distinct_values: list[str] = field(default_factory=list)
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)
    is_denied: InitVar[bool] = False
    mode_frequency_ratio: float = 0.0
    is_canonical_duplicate: InitVar[bool] = True
    pk_inference_tag: PkInferenceTag | None = None
    role_owner: RoleOwner | None = None
    boolean_truth_value: str | None = None

    def __post_init__(
        self,
        is_primary_key: bool,
        is_foreign_key: bool,
        fk_target: tuple[str, str] | None,
        is_denied: bool,
        is_canonical_duplicate: bool,
    ) -> None:
        """
        Set `value_type` from `data_type` when `value_type` is empty and capture PK/FK/deny seeds.

        ``is_primary_key``, ``is_foreign_key``, ``fk_target``, and ``is_denied`` are accepted as constructor arguments for ergonomic standalone construction (tests, ad-hoc fixtures), but they are merely seeds: the authoritative store of primary-key membership is ``TableMetadata.primary_key``, of foreign-key membership is ``TableMetadata.foreign_keys``, and of deny-list membership is ``SchemaGraph.deny_columns`` on the owning graph. When this column is attached to a :class:`TableMetadata` (and that table to a :class:`SchemaGraph`), the relevant ``__post_init__`` consolidates the seeds into the canonical containers and clears them so the :attr:`is_primary_key`, :attr:`is_foreign_key`, :attr:`fk_target`, and :attr:`is_denied` properties always read from a single owner. The seed and back-reference attributes are stored outside the dataclass field set so :func:`dataclasses.asdict` does not traverse a column-table-graph cycle.
        """
        if not self.value_type and self.data_type:
            self.value_type = data_type_to_value_type(self.data_type)
        object.__setattr__(self, "_seed_is_primary_key", bool(is_primary_key))
        object.__setattr__(self, "_seed_is_foreign_key", bool(is_foreign_key))
        object.__setattr__(
            self,
            "_seed_fk_target",
            (tuple(fk_target) if isinstance(fk_target, (list, tuple)) and len(fk_target) == 2 else None),
        )
        object.__setattr__(self, "_seed_is_denied", bool(is_denied))
        object.__setattr__(self, "_seed_is_canonical_duplicate", bool(is_canonical_duplicate))
        object.__setattr__(self, "_owner_table", None)
        if not isinstance(self.sensitivity, SensitivityClassification):
            set_sensitivity(self, self.sensitivity)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ColumnMetadata:
        """
        Create `ColumnMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `ColumnMetadata` fields.

        Returns:

            Populated `ColumnMetadata` instance.
        """
        fk_target = None
        if d.get("fk_target"):
            fk_target = tuple(d["fk_target"]) if isinstance(d["fk_target"], list) else d["fk_target"]
        sens = column_sensitivity_from_dict(d)
        return ColumnMetadata(
            name=d.get("name", ""),
            data_type=d.get("data_type", ""),
            is_primary_key=d.get("is_primary_key", False),
            is_foreign_key=d.get("is_foreign_key", False),
            fk_target=fk_target,
            role=d.get("role"),
            value_type=d.get("value_type", ""),
            enum_type_name=d.get("enum_type_name"),
            row_count=d.get("row_count", 0),
            distinct_count=d.get("distinct_count", 0),
            distinct_from_sample=d.get("distinct_from_sample", False),
            distinct_ratio=d.get("distinct_ratio", 0.0),
            null_ratio=d.get("null_ratio", 0.0),
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
            top_k_values=d.get("top_k_values", []),
            is_aggregatable_override=d.get("is_aggregatable_override"),
            is_groupable_override=d.get("is_groupable_override"),
            is_filterable_override=d.get("is_filterable_override"),
            valid_filter_ops=d.get("valid_filter_ops", []),
            valid_aggregations=d.get("valid_aggregations", []),
            valid_having_ops=d.get("valid_having_ops", []),
            description=d.get("description", ""),
            description_owner=_coerce_description_owner(d.get("description_owner")),
            is_unique=d.get("is_unique", False),
            sensitivity=sens,
            element_type=d.get("element_type"),
            is_nullable=d.get("is_nullable", True),
            semantic_distinct_values=d.get("semantic_distinct_values", []),
            semantic_join_neighbors=[
                (str(x[0]), str(x[1]))
                for x in (d.get("semantic_join_neighbors") or [])
                if isinstance(x, (list, tuple)) and len(x) == 2
            ],
            is_denied=d.get("is_denied", False),
            mode_frequency_ratio=d.get("mode_frequency_ratio", 0.0),
            is_canonical_duplicate=d.get("is_canonical_duplicate", True),
            pk_inference_tag=coerce_pk_inference_tag(d.get("pk_inference_tag")),
            role_owner=coerce_role_owner(d.get("role_owner")),
            boolean_truth_value=d.get("boolean_truth_value"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `ColumnMetadata` fields as primitives.
        """
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "fk_target": list(self.fk_target) if self.fk_target else None,
            "role": self.role,
            "value_type": self.value_type,
            "enum_type_name": self.enum_type_name,
            "row_count": self.row_count,
            "distinct_count": self.distinct_count,
            "distinct_from_sample": self.distinct_from_sample,
            "distinct_ratio": self.distinct_ratio,
            "null_ratio": self.null_ratio,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "top_k_values": self.top_k_values,
            "is_aggregatable_override": self.is_aggregatable_override,
            "is_groupable_override": self.is_groupable_override,
            "is_filterable_override": self.is_filterable_override,
            "valid_filter_ops": self.valid_filter_ops,
            "valid_aggregations": self.valid_aggregations,
            "valid_having_ops": self.valid_having_ops,
            "description": self.description,
            "description_owner": (self.description_owner.value if self.description_owner is not None else None),
            "is_unique": self.is_unique,
            "sensitivity": self.sensitivity.value,
            "is_selectable": self.is_selectable,
            "element_type": self.element_type,
            "is_nullable": self.is_nullable,
            "semantic_distinct_values": self.semantic_distinct_values,
            "semantic_join_neighbors": [list(p) for p in self.semantic_join_neighbors],
            "is_denied": self.is_denied,
            "mode_frequency_ratio": self.mode_frequency_ratio,
            "is_canonical_duplicate": self.is_canonical_duplicate,
            "pk_inference_tag": (self.pk_inference_tag.value if self.pk_inference_tag is not None else None),
            "role_owner": (self.role_owner.value if self.role_owner is not None else None),
            "boolean_truth_value": self.boolean_truth_value,
        }

    @property
    def is_selectable(self) -> bool:
        """
        Whether the column may be projected in a ``SELECT`` list.

        :attr:`SensitivityClassification.FORBIDDEN` columns are never selectable. Only
        :attr:`SensitivityClassification.HYGIENE` may be projected among non-public tiers.
        """

        if self.sensitivity == SensitivityClassification.FORBIDDEN:
            return False
        if self.sensitivity == SensitivityClassification.HYGIENE:
            return True
        if self.sensitivity == SensitivityClassification.NONE:
            return True
        return False

    @property
    def is_usable(self) -> bool:
        """
        Whether the column has enough variance and signal to be exposed to the LLM.

        Returns:

            True for primary or foreign key columns regardless of other signals (structural columns must remain visible for joins). Otherwise False for columns with at most one distinct value, columns whose null ratio meets or exceeds ``PolicyConfig.UNUSABLE_NULL_RATIO_THRESHOLD``, or columns where one value dominates the non-null distribution at or above ``PolicyConfig.SENTINEL_MODE_FREQUENCY_THRESHOLD`` (sentinel-dominated columns carry no useful filter or grouping signal). Otherwise True.
        """
        if self.is_primary_key or self.is_foreign_key:
            return True
        if self.distinct_count is not None and self.distinct_count <= 1:
            return False
        if self.null_ratio >= PolicyConfig.UNUSABLE_NULL_RATIO_THRESHOLD:
            return False
        if self.mode_frequency_ratio >= PolicyConfig.SENTINEL_MODE_FREQUENCY_THRESHOLD:
            return False
        return True

    @property
    def is_visible(self) -> bool:
        """
        Whether this column should appear in LLM-facing schema context.

        Combines structural usability with policy gates. Returns False when the column is denied via ``SchemaContext.deny_columns`` (``is_denied``). :attr:`SensitivityClassification.FORBIDDEN` columns are never visible. :attr:`SensitivityClassification.HYGIENE` requires :attr:`is_usable`; stricter tiers are withheld from prompts.
        """

        if self.is_denied:
            return False
        if self.sensitivity == SensitivityClassification.FORBIDDEN:
            return False
        if self.sensitivity == SensitivityClassification.HYGIENE:
            return self.is_usable
        if self.sensitivity != SensitivityClassification.NONE:
            return False
        return self.is_usable

    def visibility_block_reason(self) -> ColumnVisibilityBlockReason | None:
        """Return why this column is not LLM-visible, or ``None`` when it is visible."""

        owner = self._owner_table
        graph = getattr(owner, "_owner_graph", None) if owner is not None else None
        if owner is not None and graph is not None:
            deny_set = graph.deny_columns.get(owner.name)
            if deny_set and self.name in deny_set:
                return ColumnVisibilityBlockReason.DENIED
            disallowed = graph.disallowed_columns.get(owner.name)
            if disallowed and self.name in disallowed:
                return ColumnVisibilityBlockReason.NOT_IN_ALLOW_COLUMNS
        elif self._seed_is_denied:
            return ColumnVisibilityBlockReason.DENIED
        if self.sensitivity == SensitivityClassification.HYGIENE:
            if not self.is_usable:
                return ColumnVisibilityBlockReason.UNUSABLE
            return None
        if self.sensitivity in (SensitivityClassification.STRICT, SensitivityClassification.FORBIDDEN):
            return ColumnVisibilityBlockReason.SENSITIVE_PII
        if not self.is_usable:
            return ColumnVisibilityBlockReason.UNUSABLE
        return None

    @property
    def is_filterable(self) -> bool:
        """
        Whether the column may appear in `WHERE` predicates.

        Returns:

            False if the name matches an excluded pattern; else override, key, or role-based rules.
        """
        for pattern in EXCLUDED_FILTER_PATTERNS:
            if re.search(pattern, self.name, re.IGNORECASE):
                return False
        if self.is_filterable_override is not None:
            return self.is_filterable_override
        if self.is_primary_key:
            return True
        if self.is_foreign_key:
            return True
        if self.role in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.AUDIT.value,
        ):
            return True
        return False

    def get_valid_filter_ops(self) -> list[str]:
        """
        Valid filter operators for this column, always including null checks.

        Returns:

            Operator strings such as `=`, `!=`, `like`, `between`, plus `is null` / `is not null`.
        """
        null_ops = ["is null", "is not null"]
        if self.valid_filter_ops:
            return list(set(self.valid_filter_ops + null_ops))
        return null_ops

    def get_valid_aggregations(self) -> set[str]:
        """
        Valid aggregation function names for this column.

        Returns:

            Lowercased names from `valid_aggregations`, or an empty set if none are stored.
        """
        if self.valid_aggregations:
            return set(agg.lower() for agg in self.valid_aggregations)
        if self.role:
            rk = self.role.upper()
            if rk in ROLE_ALLOWED_AGGREGATIONS:
                return {a.lower() for a in ROLE_ALLOWED_AGGREGATIONS[rk]}
        return set()

    def get_valid_having_ops(self) -> list[str]:
        """
        Valid `HAVING` operators for this column.

        Returns:

            A copy of `valid_having_ops` if set, otherwise an empty list.
        """
        if self.valid_having_ops:
            return list(self.valid_having_ops)
        return []

    @property
    def is_groupable(self) -> bool:
        """
        Whether the column may appear in `GROUP BY`.

        Returns:

            True when override, foreign key, or role allows grouping.
        """
        if self.is_groupable_override is not None:
            return self.is_groupable_override
        if self.is_foreign_key:
            return True
        return self.role in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.IDENTIFIER.value,
        )

    @property
    def is_aggregatable(self) -> bool:
        """
        Whether measures like `SUM` / `AVG` apply to this column.

        Returns:

            True when override is set, or role is numeric measure.
        """
        if self.is_aggregatable_override is not None:
            return self.is_aggregatable_override
        return self.role == ColumnRole.NUMERIC_MEASURE.value


def _column_metadata_is_foreign_key(self: ColumnMetadata) -> bool:
    """
    Whether this column participates as the source of any foreign-key edge on its owning table.

    Derived strictly from ``TableMetadata.foreign_keys`` once an owner is wired; before wiring, falls back to the constructor seed value (used by standalone fixtures with no parent table).
    """
    owner = self._owner_table
    if owner is None:
        return self._seed_is_foreign_key
    for fk in owner.foreign_keys:
        if self.name in fk.src_cols:
            return True
    return False


def _column_metadata_fk_target(self: ColumnMetadata) -> tuple[str, str] | None:
    """
    Destination ``(table, column)`` of the first foreign-key edge whose source includes this column.

    Looked up from ``TableMetadata.foreign_keys`` when an owner is wired; before wiring, returns the constructor seed.
    """
    owner = self._owner_table
    if owner is None:
        return self._seed_fk_target
    for fk in owner.foreign_keys:
        for sc, dc in zip(fk.src_cols, fk.dst_cols, strict=False):
            if sc == self.name:
                return (fk.dst_table, dc)
    return None


ColumnMetadata.is_foreign_key = property(_column_metadata_is_foreign_key)
ColumnMetadata.fk_target = property(_column_metadata_fk_target)


def _column_metadata_is_primary_key(self: ColumnMetadata) -> bool:
    """
    Whether this column appears in its owning table's primary-key list.

    Derived strictly from ``TableMetadata.primary_key`` once an owner is wired; before wiring, falls back to the constructor seed value (used by standalone fixtures with no parent table).
    """
    owner = self._owner_table
    if owner is None:
        return self._seed_is_primary_key
    return self.name in owner.primary_key


ColumnMetadata.is_primary_key = property(_column_metadata_is_primary_key)


def _column_metadata_is_denied(self: ColumnMetadata) -> bool:
    """
    Whether this column is denied by scope policy on its owning :class:`SchemaGraph`.

    True when the column appears in ``SchemaGraph.deny_columns`` or ``SchemaGraph.disallowed_columns`` for its owning table (once wired), or when the standalone fixture seed marks it denied.
    """
    owner = self._owner_table
    if owner is None:
        return self._seed_is_denied
    graph = getattr(owner, "_owner_graph", None)
    if graph is None:
        return self._seed_is_denied
    deny_set = graph.deny_columns.get(owner.name)
    if deny_set and self.name in deny_set:
        return True
    disallowed = graph.disallowed_columns.get(owner.name)
    return bool(disallowed and self.name in disallowed)


ColumnMetadata.is_denied = property(_column_metadata_is_denied)


def _column_metadata_is_canonical_duplicate(self: ColumnMetadata) -> bool:
    """
    Whether this column is the canonical bearer for its name across the schema graph.

    A column whose name is unique across all tables is trivially canonical. When the same name appears in two or more tables, exactly one bearer is selected by ``recompute_canonical_bearers`` on the owning :class:`SchemaGraph` and recorded in ``SchemaGraph._canonical_bearers``; that bearer reads ``True`` and the others read ``False``. Before owner-graph wiring, falls back to the constructor seed value.
    """
    owner = self._owner_table
    if owner is None:
        return self._seed_is_canonical_duplicate
    graph = getattr(owner, "_owner_graph", None)
    if graph is None:
        return self._seed_is_canonical_duplicate
    bearers = getattr(graph, "_canonical_bearers", None)
    if not bearers:
        return True
    bearer = bearers.get(self.name.lower())
    if bearer is None:
        return True
    return bearer == (owner.name, self.name)


ColumnMetadata.is_canonical_duplicate = property(_column_metadata_is_canonical_duplicate)


@dataclass
class TableMetadata:
    """Table metadata with nested columns, foreign keys, partition columns, and role."""

    name: str
    columns: dict[str, ColumnMetadata]
    primary_key: list[str]
    foreign_keys: list[FKEdge]
    kind: Literal["table", "view"] = "table"
    partition_columns: list[str] = field(default_factory=list)
    role: str | None = None
    row_count: int = 0
    description: str = ""
    description_owner: DescriptionOwner | None = None
    role_owner: RoleOwner | None = None
    composite_descriptive_ratios: dict[tuple[str, str], float] = field(
        default_factory=dict,
    )
    _user_semantic_neighbors: list[tuple[str, str, str, str]] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """
        Wire each child :class:`ColumnMetadata` back to this table and consolidate any PK / FK seeds.

        Tests and ad-hoc fixtures may pass ``is_primary_key``, ``is_foreign_key`` / ``fk_target`` as constructor arguments to :class:`ColumnMetadata` without separately populating :attr:`primary_key` or appending an :class:`FKEdge` to :attr:`foreign_keys`. After wiring the per-column ``_owner_table`` back-reference, this method (a) appends any PK seed to :attr:`primary_key` when not already present, and (b) synthesises a single-column :class:`FKEdge` for every column whose seed declares an FK that is not already covered by an entry in :attr:`foreign_keys`. The seeds are then cleared so the :class:`ColumnMetadata` properties always read from this table's :attr:`primary_key` and :attr:`foreign_keys` as the single source of truth.
        """
        covered_fk: set[str] = set()
        for fk in self.foreign_keys:
            for sc in fk.src_cols:
                covered_fk.add(sc)
        if isinstance(self.primary_key, str):
            self.primary_key = [self.primary_key] if self.primary_key else []
        existing_pk = set(self.primary_key)
        seeded_denies: set[str] = set()
        for cname, col in self.columns.items():
            col._owner_table = self
            if col._seed_is_primary_key and cname not in existing_pk:
                self.primary_key.append(cname)
                existing_pk.add(cname)
            if col._seed_is_foreign_key and col._seed_fk_target is not None and cname not in covered_fk:
                dst_t, dst_c = col._seed_fk_target
                self.foreign_keys.append(
                    FKEdge(
                        src_table=self.name,
                        src_cols=[cname],
                        dst_table=dst_t,
                        dst_cols=[dst_c],
                        inference_tag=None,
                    )
                )
                covered_fk.add(cname)
            if col._seed_is_denied:
                seeded_denies.add(cname)
            col._seed_is_primary_key = False
            col._seed_is_foreign_key = False
            col._seed_fk_target = None
        object.__setattr__(self, "_owner_graph", None)
        object.__setattr__(self, "_pending_denies", seeded_denies)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TableMetadata:
        """
        Create `TableMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `TableMetadata` fields.

        Returns:

            Populated `TableMetadata` with nested `ColumnMetadata` and `FKEdge` objects.
        """
        cols_raw = d.get("columns", {})
        columns = {k: ColumnMetadata.from_dict(v) for k, v in cols_raw.items()} if isinstance(cols_raw, dict) else {}
        fk_raw = d.get("foreign_keys", [])
        foreign_keys = [FKEdge(**fk) if isinstance(fk, dict) else fk for fk in fk_raw]
        kind_raw = d.get("kind", "table")
        kind: Literal["table", "view"] = "table" if kind_raw == "table" else "view"
        return TableMetadata(
            name=d.get("name", ""),
            columns=columns,
            primary_key=d.get("primary_key", []),
            foreign_keys=foreign_keys,
            kind=kind,
            partition_columns=d.get("partition_columns", []),
            role=d.get("role"),
            row_count=d.get("row_count", 0),
            description=d.get("description", ""),
            description_owner=_coerce_description_owner(d.get("description_owner")),
            role_owner=coerce_role_owner(d.get("role_owner")),
            composite_descriptive_ratios={
                tuple(k.split("|", 1)): v for k, v in d.get("composite_descriptive_ratios", {}).items() if "|" in k
            },
            _user_semantic_neighbors=[
                tuple(item)
                for item in (d.get("_user_semantic_neighbors", []) or [])
                if isinstance(item, (list, tuple)) and len(item) == 4
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `TableMetadata` fields; nested columns and foreign keys are serialized recursively.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "primary_key": self.primary_key,
            "foreign_keys": [asdict(fk) for fk in self.foreign_keys],
            "partition_columns": self.partition_columns,
            "role": self.role,
            "row_count": self.row_count,
            "description": self.description,
            "description_owner": (self.description_owner.value if self.description_owner is not None else None),
            "role_owner": (self.role_owner.value if self.role_owner is not None else None),
            "composite_descriptive_ratios": {
                f"{c1}|{c2}": ratio for (c1, c2), ratio in self.composite_descriptive_ratios.items()
            },
            "_user_semantic_neighbors": [list(t) for t in self._user_semantic_neighbors],
        }

    @property
    def column_names(self) -> list[str]:
        """
        Ordered column names for this table.

        Returns:

            Keys of `columns` as a list.
        """
        return list(self.columns.keys())


_SchemaGraphStatsFn = Callable[["SchemaGraph"], dict[str, Any]]
_SchemaGraphCapabilityFn = Callable[["SchemaGraph"], DatabaseFeatureCapability]

_schema_graph_stats_fn: _SchemaGraphStatsFn | None = None
_schema_graph_capability_fn: _SchemaGraphCapabilityFn | None = None


def set_schema_helpers(
    stats_fn: _SchemaGraphStatsFn,
    capability_fn: _SchemaGraphCapabilityFn,
) -> None:
    """
    Wire :meth:`SchemaGraph.refresh_schema_stats` and :attr:`SchemaGraph.database_feature_capability`
    to the implementations in :mod:`aetherdialect._schema` (called once at import time from that module).
    """

    global _schema_graph_stats_fn, _schema_graph_capability_fn
    _schema_graph_stats_fn = stats_fn
    _schema_graph_capability_fn = capability_fn


@dataclass
class SchemaGraph:
    """Schema graph with nested tables, join paths, and metadata."""

    tables: dict[str, TableMetadata]
    join_paths_multi: dict[str, dict[str, list[list[dict[str, Any]]]]]
    structural_hash: str = ""
    profiling_hash: str = ""
    scope_hash: str = ""
    effective_structural_hash: str = ""
    notes_hash: str = ""
    semantic_edges_hash: str = ""
    ddl_probe_hash: str = ""
    include: SchemaInclude = "tables"
    created_at: str = ""
    enum_values: dict[str, list[str]] | None = None
    schema_stats: dict[str, Any] | None = None
    deny_columns: dict[str, set[str]] = field(default_factory=dict)
    disallowed_columns: dict[str, set[str]] = field(default_factory=dict)
    notes_sha256: str = ""
    scope_descriptor: dict[str, Any] | None = None
    schema_revision: int = 0
    _database_feature_capability_cache: DatabaseFeatureCapability | None = field(default=None, repr=False, compare=False)
    _stats_dirty: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Wire owner-graph back-references and consolidate per-column deny seeds into ``deny_columns``.

        After this runs, ``deny_columns`` is the single source of truth for ``ColumnMetadata.is_denied``. Existing ``deny_columns`` entries are preserved; per-column seeds (set on standalone-built columns) and per-table pending-deny sets (collected by :meth:`TableMetadata.__post_init__`) are folded in.
        """
        deny_columns: dict[str, set[str]] = {k: set(v) for k, v in (self.deny_columns or {}).items()}
        disallowed_columns: dict[str, set[str]] = {k: set(v) for k, v in (self.disallowed_columns or {}).items()}
        for tbl_name, tbl in self.tables.items():
            object.__setattr__(tbl, "_owner_graph", self)
            pending = getattr(tbl, "_pending_denies", set())
            if pending:
                deny_columns.setdefault(tbl_name, set()).update(pending)
                object.__setattr__(tbl, "_pending_denies", set())
            for col_name, col in tbl.columns.items():
                if getattr(col, "_seed_is_denied", False):
                    deny_columns.setdefault(tbl_name, set()).add(col_name)
                    col._seed_is_denied = False
        self.deny_columns = deny_columns
        self.disallowed_columns = disallowed_columns
        if not hasattr(self, "_canonical_bearers"):
            object.__setattr__(self, "_canonical_bearers", {})

    def mark_stats_dirty(self) -> None:
        """Flag :attr:`schema_stats` as stale so the next :meth:`ensure_schema_stats` call recomputes it."""

        self._stats_dirty = True

    def refresh_schema_stats(self) -> dict[str, Any]:
        """Unconditionally recompute :attr:`schema_stats` from the current graph and clear the dirty flag."""

        fn = _schema_graph_stats_fn
        if fn is None:
            raise RuntimeError("Schema helpers not wired (aetherdialect._schema did not load)")
        self.schema_stats = fn(self)
        self._stats_dirty = False
        return self.schema_stats

    def ensure_schema_stats(self) -> dict[str, Any]:
        """Recompute :attr:`schema_stats` only when the dirty flag is set or the cached payload is missing/empty; otherwise return the cached value."""

        if self._stats_dirty or not self.schema_stats:
            return self.refresh_schema_stats()
        return self.schema_stats

    @property
    def fk_edges(self) -> list[FKEdge]:
        """
        All foreign-key edges declared on tables in the graph.

        Returns:

            Flattened list of `FKEdge` from every `TableMetadata.foreign_keys`.
        """
        return [fk for table in self.tables.values() for fk in table.foreign_keys]

    @property
    def table_names(self) -> list[str]:
        """
        Table names present in the graph.

        Returns:

            Keys of `tables` as a list.
        """
        return list(self.tables.keys())

    @property
    def schema_hash(self) -> str:
        """Alias for ``effective_structural_hash`` for legacy call sites."""

        return self.effective_structural_hash

    @property
    def database_feature_capability(self) -> DatabaseFeatureCapability:
        """
        Cached structural feasibility snapshot for tier-conditioned generators.

        Returns:

            :class:`DatabaseFeatureCapability` computed once per graph instance.
        """

        cached = self._database_feature_capability_cache
        if cached is None:
            cap_fn = _schema_graph_capability_fn
            if cap_fn is None:
                raise RuntimeError("Schema helpers not wired (aetherdialect._schema did not load)")
            cached = cap_fn(self)
            object.__setattr__(self, "_database_feature_capability_cache", cached)
        return cached

    def get_column(self, table: str, column: str) -> ColumnMetadata | None:
        """
        Look up column metadata by table and column name.

        Args:

            table: Table name to look up.

            column: Column name within that table.

        Returns:

            `ColumnMetadata` if found, otherwise None.
        """
        if table in self.tables and column in self.tables[table].columns:
            return self.tables[table].columns[column]
        return None

    def _schema_literal_public_role(self, role: str | None) -> str | None:
        if role is None:
            return None
        r = str(role).strip()
        if not r or r in (ColumnRole.IDENTIFIER.value, ColumnRole.AUDIT.value):
            return None
        return r

    def _schema_literal_column_type_token(self, col: ColumnMetadata) -> str:
        vt = (col.value_type or "").strip()
        if vt:
            return vt
        if col.data_type:
            return data_type_to_value_type(col.data_type)
        return "unknown"

    def _schema_literal_column_object(
        self,
        col: ColumnMetadata,
        *,
        structural_only: bool,
    ) -> dict[str, Any]:
        col_body: dict[str, Any] = {"type": self._schema_literal_column_type_token(col)}
        if col.is_primary_key:
            col_body["pk"] = True
        if col.fk_target:
            col_body["fk"] = f"{col.fk_target[0]}.{col.fk_target[1]}"
        if col.is_unique and not col.is_primary_key:
            col_body["unique"] = True
        if not structural_only:
            desc = (col.description or "").strip()
            if desc:
                col_body["description"] = desc
            pub_role = self._schema_literal_public_role(col.role)
            if pub_role is not None:
                col_body["role"] = pub_role
            if col_body["type"].lower() == "boolean":
                tv = (col.boolean_truth_value or "").strip()
                if tv:
                    col_body["truth_value"] = tv
        return col_body

    def _schema_literal_payload(
        self,
        *,
        structural_only: bool,
        table_filter: frozenset[str] | None,
    ) -> dict[str, Any]:
        root: dict[str, Any] = {}
        for tname in sorted(self.tables):
            if table_filter is not None and tname not in table_filter:
                continue
            tm = self.tables[tname]
            col_map: dict[str, dict[str, Any]] = {}
            for col_name in sorted(tm.columns.keys()):
                col = tm.columns[col_name]
                if not col.is_visible:
                    continue
                col_map[col_name] = self._schema_literal_column_object(col, structural_only=structural_only)
            table_body: dict[str, Any] = {"columns": col_map}
            if not structural_only:
                td = (tm.description or "").strip()
                if td:
                    table_body["description"] = td
                tr = self._schema_literal_public_role(tm.role)
                if tr is not None:
                    table_body["role"] = tr
            root[tname] = table_body
        if not structural_only and self.enum_values:
            enum_block: dict[str, Any] = {}
            for ename in sorted(self.enum_values.keys()):
                values = self.enum_values[ename]
                if len(values) <= 10:
                    enum_block[ename] = list(values)
                else:
                    enum_block[ename] = list(values[:10]) + ["..."]
            root["enum_types"] = enum_block
        return root

    @property
    def schema_literal_json(self) -> str:
        """
        JSON string describing visible, scope-permitted tables and columns for LLM prompts.

        Columns whose ``is_visible`` is false are omitted. Optional ``enum_types`` summarizes enumerated domains when present. Boolean columns may include ``truth_value`` when configured.

        Returns:

            Compact JSON text; an empty graph yields ``"{}"``.
        """

        payload = self._schema_literal_payload(structural_only=False, table_filter=None)
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def structural_schema_literal_json(self, tables: Iterable[str] | None = None) -> str:
        """
        Structural schema JSON with descriptions stripped, optionally restricted to *tables*.

        Args:

            tables: When ``None``, every graph table is included; otherwise only listed names that exist.

        Returns:

            Compact JSON text; unknown table names in *tables* are ignored.
        """

        filt: frozenset[str] | None = frozenset(str(t) for t in tables) if tables is not None else None
        payload = self._schema_literal_payload(structural_only=True, table_filter=filt)
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SchemaGraph:
        """
        Create `SchemaGraph` from a dictionary.

        Args:

            d: Dictionary with keys matching `SchemaGraph` fields, typically loaded from JSON.

        Returns:

            Populated `SchemaGraph` with nested `TableMetadata` instances.
        """
        tables_raw = d.get("tables", {})
        tables = {k: TableMetadata.from_dict(v) for k, v in tables_raw.items()}
        deny_cols_raw = d.get("deny_columns", {})
        deny_columns: dict[str, set[str]] = {}
        if isinstance(deny_cols_raw, dict):
            for tbl, cols in deny_cols_raw.items():
                if isinstance(cols, list):
                    deny_columns[str(tbl)] = set(str(c) for c in cols)
        disallowed_raw = d.get("disallowed_columns", {})
        disallowed_columns: dict[str, set[str]] = {}
        if isinstance(disallowed_raw, dict):
            for tbl, cols in disallowed_raw.items():
                if isinstance(cols, list):
                    disallowed_columns[str(tbl)] = set(str(c) for c in cols)
        legacy_hash = str(d.get("schema_hash", "") or "")
        structural_hash = str(d.get("structural_hash", "") or legacy_hash)
        profiling_hash = str(d.get("profiling_hash", "") or legacy_hash)
        scope_hash = str(d.get("scope_hash", "") or legacy_hash)
        effective_structural_hash = str(d.get("effective_structural_hash", "") or legacy_hash)
        inc_raw = d.get("include")
        if inc_raw in ("tables", "views", "both"):
            include_val: SchemaInclude = inc_raw
        else:
            okind = d.get("object_kind", "table")
            include_val = "views" if okind == "view" else "tables"
        return SchemaGraph(
            tables=tables,
            join_paths_multi=d.get("join_paths_multi", {}),
            structural_hash=structural_hash,
            profiling_hash=profiling_hash,
            scope_hash=scope_hash,
            effective_structural_hash=effective_structural_hash,
            notes_hash=str(d.get("notes_hash", "") or ""),
            semantic_edges_hash=str(d.get("semantic_edges_hash", "") or ""),
            ddl_probe_hash=str(d.get("ddl_probe_hash", "") or ""),
            include=include_val,
            created_at=d.get("created_at", ""),
            enum_values=d.get("enum_values"),
            schema_stats=d.get("schema_stats"),
            deny_columns=deny_columns,
            disallowed_columns=disallowed_columns,
            notes_sha256=str(d.get("notes_sha256", "") or ""),
            scope_descriptor=(d.get("scope_descriptor") if isinstance(d.get("scope_descriptor"), dict) else None),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `SchemaGraph` fields; nested tables are serialized recursively.
        """
        return {
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "join_paths_multi": self.join_paths_multi,
            "structural_hash": self.structural_hash,
            "profiling_hash": self.profiling_hash,
            "scope_hash": self.scope_hash,
            "effective_structural_hash": self.effective_structural_hash,
            "notes_hash": self.notes_hash,
            "semantic_edges_hash": self.semantic_edges_hash,
            "ddl_probe_hash": self.ddl_probe_hash,
            "include": self.include,
            "created_at": self.created_at,
            "enum_values": self.enum_values,
            "schema_stats": self.schema_stats,
            "deny_columns": {k: sorted(v) for k, v in self.deny_columns.items()},
            "disallowed_columns": {k: sorted(v) for k, v in self.disallowed_columns.items()},
            "notes_sha256": self.notes_sha256,
            "scope_descriptor": self.scope_descriptor,
        }


@dataclass
class ExpansionMetadata:
    """Metadata for intent expansion operations."""

    operator: str
    parent_intent_id: str | None = None
    depth: int = 0
    expansion_path: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExpansionMetadata:
        """
        Create `ExpansionMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `ExpansionMetadata` fields.

        Returns:

            Populated `ExpansionMetadata` instance.
        """
        return ExpansionMetadata(
            operator=d.get("operator", ""),
            parent_intent_id=d.get("parent_intent_id"),
            depth=d.get("depth", 0),
            expansion_path=d.get("expansion_path", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize expansion metadata to a plain dict.

        Returns:

            `asdict` of all fields.
        """
        return asdict(self)


@dataclass
class CteOutputColumnMeta:
    """Metadata for a CTE output column, including source, role, and aggregation info."""

    source: str
    agg_func: str = ""
    role: str | None = None
    filterable: bool = True
    aggregatable: bool = True
    data_type: str = "unknown"
    value_type: str = ""
    groupable: bool = True
    valid_filter_ops: list[str] = field(default_factory=list)
    valid_aggregations: list[str] = field(default_factory=list)
    valid_having_ops: list[str] = field(default_factory=list)
    sensitivity: str | None = None
    lineage_phys_table: str | None = None
    lineage_phys_column: str | None = None
    lineage_inherits_pk: bool = False
    lineage_fk_to_table: str | None = None
    lineage_fk_to_column: str | None = None
    semantic_distinct_values: list[str] = field(default_factory=list)
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set `value_type` from `data_type` when `value_type` is empty."""
        if not self.value_type and self.data_type:
            self.value_type = data_type_to_value_type(self.data_type)

    @property
    def is_selectable(self) -> bool:
        """Whether the CTE output column may be projected; derived from ``sensitivity`` (hidden tags suppress selection)."""

        if self.sensitivity is None:
            return True
        return str(self.sensitivity).strip().lower() not in HIDDEN_SENSITIVITIES

    def get_valid_filter_ops(self) -> list[str]:
        """
        Filter operators allowed on this CTE output column.

        Returns:

            Stored ops plus null checks, or defaults when `filterable`, else null checks only.
        """
        null_ops = ["is null", "is not null"]
        if self.valid_filter_ops:
            return list(set(self.valid_filter_ops + null_ops))
        if self.filterable:
            return [
                "=",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "in",
                "not in",
                "is null",
                "is not null",
            ]
        return null_ops

    def get_valid_aggregations(self) -> set[str]:
        """
        Aggregation names allowed on this CTE output column.

        Returns:

            Lowercased `valid_aggregations`, or defaults by `aggregatable` flag.
        """
        if self.valid_aggregations:
            return set(agg.lower() for agg in self.valid_aggregations)
        if not self.role:
            return set()
        rk = self.role.upper()
        if rk in ROLE_ALLOWED_AGGREGATIONS:
            return {a.lower() for a in ROLE_ALLOWED_AGGREGATIONS[rk]}
        if self.aggregatable:
            return {"count", "sum", "avg", "min", "max"}
        return {"count"}

    def get_valid_having_ops(self) -> list[str]:
        """
        `HAVING` operators allowed on this CTE output column.

        Returns:

            Stored list, comparison ops when `aggregatable`, or an empty list.
        """
        if self.valid_having_ops:
            return list(self.valid_having_ops)
        if self.aggregatable:
            return ["=", "!=", "<", "<=", ">", ">="]
        return []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CteOutputColumnMeta:
        """
        Create `CteOutputColumnMeta` from a dictionary.

        Args:

            d: Dictionary with keys matching `CteOutputColumnMeta` fields.

        Returns:

            Populated `CteOutputColumnMeta` instance.
        """
        return CteOutputColumnMeta(
            source=d.get("source", "passthrough"),
            agg_func=d.get("agg_func", ""),
            role=d.get("role"),
            filterable=d.get("filterable", True),
            aggregatable=d.get("aggregatable", True),
            data_type=d.get("data_type", "unknown"),
            value_type=d.get("value_type", ""),
            groupable=d.get("groupable", True),
            valid_filter_ops=d.get("valid_filter_ops", []),
            valid_aggregations=d.get("valid_aggregations", []),
            valid_having_ops=d.get("valid_having_ops", []),
            sensitivity=d.get("sensitivity"),
            lineage_phys_table=d.get("lineage_phys_table"),
            lineage_phys_column=d.get("lineage_phys_column"),
            lineage_inherits_pk=d.get("lineage_inherits_pk", False),
            lineage_fk_to_table=d.get("lineage_fk_to_table"),
            lineage_fk_to_column=d.get("lineage_fk_to_column"),
            semantic_distinct_values=d.get("semantic_distinct_values", []),
            semantic_join_neighbors=[
                (str(x[0]), str(x[1]))
                for x in (d.get("semantic_join_neighbors") or [])
                if isinstance(x, (list, tuple)) and len(x) == 2
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize CTE column meta to a plain dict.

        Returns:

            Plain dict including JSON-friendly neighbor pairs.
        """
        d = asdict(self)
        d["is_selectable"] = self.is_selectable
        d["semantic_join_neighbors"] = [list(p) for p in self.semantic_join_neighbors]
        return d


@dataclass
class VirtualColumnSpec:
    """Join-discovery view of one CTE output column with lifted physical lineage."""

    lineage_phys_table: str | None
    lineage_phys_column: str | None
    inherits_pk: bool
    fk_to: tuple[str, str] | None
    semantic_distinct_values: list[str]
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class VirtualTableSpec:
    """In-memory join graph node for a CTE keyed by ``cte_name``."""

    cte_name: str
    columns: dict[str, VirtualColumnSpec]
    emission: str = "join_table"


@dataclass
class RetryFailureContext:
    """Structured failure context for LLM retry guidance."""

    failure_type: str
    required_tables: list[str]
    used_tables: set[str]
    missing_tables: set[str]
    attempt_number: int


@dataclass
class SQLShape:
    """Structural features of a SQL query for comparison."""

    num_joins: int
    has_group_by: bool
    has_agg: bool
    num_cte: int = 0
    num_filters: int = 0
    num_having: int = 0
    has_distinct: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SQLShape:
        """
        Create `SQLShape` from a dictionary.

        Args:

            d: Dictionary with keys matching `SQLShape` fields.

        Returns:

            Populated `SQLShape` instance.
        """
        return SQLShape(
            num_joins=d.get("num_joins", 0),
            has_group_by=d.get("has_group_by", False),
            has_agg=d.get("has_agg", False),
            num_cte=d.get("num_cte", 0),
            num_filters=d.get("num_filters", 0),
            num_having=d.get("num_having", 0),
            has_distinct=d.get("has_distinct", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize shape flags to a plain dict.

        Returns:

            `asdict` of all fields.
        """
        return asdict(self)


@dataclass
class IntentIssue:
    """Issue detected during intent validation or resolution."""

    issue_id: str
    category: FailureCategory
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    responsible_stage: Literal["logical", "format"] = "format"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> IntentIssue:
        """
        Create `IntentIssue` from a dictionary.

        Args:

            d: Dictionary with keys matching `IntentIssue` fields.

        Returns:

            Populated `IntentIssue` instance.
        """
        raw_cat = d.get("category", "")
        if isinstance(raw_cat, FailureCategory):
            category: FailureCategory = raw_cat
        else:
            category = parse_failure_category(str(raw_cat) if raw_cat is not None else None) or FailureCategory.OTHER
        rs = d.get("responsible_stage", "format")
        stage: Literal["logical", "format"] = "logical" if rs == "logical" else "format"
        return IntentIssue(
            issue_id=d.get("issue_id", ""),
            category=category,
            severity=d.get("severity", "error"),
            message=d.get("message", ""),
            context=d.get("context", {}),
            responsible_stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the issue to a plain dict.

        Returns:

            Primitive field mapping including `context`.
        """
        return {
            "issue_id": self.issue_id,
            "category": self.category.value,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
            "responsible_stage": self.responsible_stage,
        }


STAGE_ATTRIBUTION_TABLE: Mapping[str, Literal["logical", "format"]] = MappingProxyType(
    {
        "column_not_found_in_chosen_tables": "format",
        "chosen_table_lacks_required_column": "format",
        "filter_targets_missing_column": "format",
        "joinpath_does_not_exist": "logical",
        "grain_inconsistent_with_chosen_tables": "logical",
        "cte_chosen_tables_inconsistent": "logical",
        "window_partition_column_missing": "format",
        "encoder_added_or_removed_tables": "format",
        "json_schema_violation": "format",
        "missing_required_field": "format",
        "invalid_operator": "format",
        "invalid_value_type": "format",
        "unqualified_column_reference": "format",
        "cte_dependency_cycle": "format",
        "window_frame_syntax_invalid": "format",
        "existence_filter_encoded_as_subquery": "format",
        "self_reference_encoded_as_inline_self_join": "format",
        "correlated_lookup_encoded_as_lateral": "format",
    }
)

_LOGICAL_FAILURE_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.UNKNOWN_TABLE,
        FailureCategory.WRONG_TABLES,
        FailureCategory.WRONG_JOIN,
        FailureCategory.GRAIN_CONSISTENCY,
        FailureCategory.GRAIN_VALIDITY,
        FailureCategory.CTE_TABLE_REFERENCE,
        FailureCategory.CTE_GRAIN_CONSISTENCY,
        FailureCategory.CTE_GRAIN_COMPATIBILITY,
        FailureCategory.WRONG_COLUMN_SELECTION,
        FailureCategory.WRONG_FILTER_LOGIC,
    }
)


LITERAL_BEARING_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.MISSING_NUMERIC_FILTER,
        FailureCategory.MISSING_TEMPORAL_COLUMN,
    }
)


def make_intent_issue(
    *,
    issue_id: str,
    category: FailureCategory,
    severity: str,
    message: str,
    context: dict[str, Any] | None = None,
    responsible_stage: Literal["logical", "format"] | None = None,
) -> IntentIssue:
    """
    Construct an :class:`IntentIssue` with ``responsible_stage`` from :data:`STAGE_ATTRIBUTION_TABLE` when omitted.

    Args:

        issue_id: Stable identifier; substring keys in :data:`STAGE_ATTRIBUTION_TABLE` select the default stage when *responsible_stage* is omitted.

        category: Failure category for the issue.

        severity: ``error``, ``warning``, or other severity token retained by validation.

        message: Human-readable explanation.

        context: Optional structured context copied into the issue.

        responsible_stage: When ``None``, the stage is inferred from *issue_id* and *category*.

    Returns:

        A new :class:`IntentIssue` with ``responsible_stage`` set explicitly or inferred.
    """

    ctx = dict(context or {})
    if responsible_stage is not None:
        return IntentIssue(
            issue_id=issue_id,
            category=category,
            severity=severity,
            message=message,
            context=ctx,
            responsible_stage=responsible_stage,
        )
    iid = (issue_id or "").lower()
    for key, stage in STAGE_ATTRIBUTION_TABLE.items():
        if key in iid:
            return IntentIssue(
                issue_id=issue_id,
                category=category,
                severity=severity,
                message=message,
                context=ctx,
                responsible_stage=stage,
            )
    if category in _LOGICAL_FAILURE_CATEGORIES:
        inferred: Literal["logical", "format"] = "logical"
    else:
        inferred = "format"
    return IntentIssue(
        issue_id=issue_id,
        category=category,
        severity=severity,
        message=message,
        context=ctx,
        responsible_stage=inferred,
    )


_KEPT_ISSUE_SEVERITIES: frozenset[str] = frozenset({"error", "warning"})


@dataclass
class IntentValidationResult:
    """
    Result container for intent validation with issue tracking.

    Only ``error`` and ``warning`` severity issues are retained; any ``info`` (or otherwise non-actionable) severity issue is dropped at construction time so downstream consumers never have to filter them out.
    """

    issues: list[IntentIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Drop any issue whose severity is not ``error`` or ``warning``."""
        self.issues = [i for i in self.issues if i.severity in _KEPT_ISSUE_SEVERITIES]

    @property
    def is_valid(self) -> bool:
        """
        Whether validation found no error-severity issues.

        Returns:

            True if no `IntentIssue` has `severity == 'error'`.
        """
        return not any(i.severity == "error" for i in self.issues)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> IntentValidationResult:
        """
        Create `IntentValidationResult` from a dictionary.

        Args:

            d: Dictionary with an `issues` list of serialized `IntentIssue` dicts.

        Returns:

            Populated `IntentValidationResult` with deserialized `IntentIssue` objects.
        """
        issues_raw = d.get("issues", [])
        return IntentValidationResult(
            issues=[IntentIssue.from_dict(i) for i in issues_raw],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize validation result for JSON.

        Returns:

            Dict with an `issues` list of serialized `IntentIssue` dicts.
        """
        return {"issues": [i.to_dict() for i in self.issues]}


@dataclass
class TemplateStats:
    """Template acceptance and rejection statistics."""

    accept: int = 0
    reject: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TemplateStats:
        """
        Create `TemplateStats` from a dictionary.

        Args:

            d: Dictionary with `accept` and `reject` integer keys.

        Returns:

            Populated `TemplateStats` instance.
        """
        return TemplateStats(
            accept=int(d.get("accept", 0)),
            reject=int(d.get("reject", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize accept/reject counts.

        Returns:

            `asdict` of `accept` and `reject`.
        """
        return asdict(self)


@dataclass
class QSimSkeleton:
    """Structural skeleton for QSim intent before the LLM fills semantics."""

    tables: list[str]
    has_aggregation: bool
    num_filters: int
    num_groupby: int
    has_orderby: bool
    num_having: int
    has_distinct: bool = False
    has_expr_comparison: bool = False


@dataclass
class SkeletonPool:
    """Tiered skeleton pool with round-robin table-set selection."""

    tier_a_by_table_set: dict[str, list[QSimSkeleton]]
    tier_b_by_table_set: dict[str, list[QSimSkeleton]]
    tier_c_by_table_set: dict[str, list[QSimSkeleton]]
    table_set_keys: list[str]
    tier_a_indices: dict[str, int]
    tier_b_indices: dict[str, int]
    tier_c_indices: dict[str, int]
    current_table_idx: int = 0


@dataclass
class TemplateInfo:
    """User-facing template information with obfuscated internals."""

    id: str
    natural_language: str
    example_question: str
    trust_level: str
    source: str


@dataclass
class RejectedTemplateInfo:
    """User-facing rejected template with generic categories."""

    id: str
    natural_language: str
    example_question: str
    rejection_category: str
    rejection_count: int


@dataclass
class SeedWarmupSummary:
    """Aggregate statistics for a seed warmup preflight or full run."""

    version: int
    total: int
    success: int
    failed: int
    success_rate: float
    seed_questions_loaded: int = 0
    gold_intents_total: int = 0
    unique_prompts: int = 0
    gold_new: int = 0
    gold_skipped: int = 0
    gold_failed: int = 0
    gold_user_rejected: int = 0
    deduped_prompts_count: int = 0
    gold_prompts_count: int = 0
    templates_added: int = 0
    validation_drop: int = 0
    realism_drop: int = 0
    question_generation_failed: int = 0
    early_pipeline_failed: int = 0


@dataclass
class QSimSummary:
    """QSim (question generation) run metadata with version, counts, and seed."""

    version: int
    num_intents: int
    num_questions: int
    seed: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimSummary:
        """
        Create `QSimSummary` from a dictionary.

        Args:

            d: Dictionary with keys matching `QSimSummary` fields.

        Returns:

            Populated `QSimSummary` instance.
        """
        return QSimSummary(
            version=int(d.get("version", 0)),
            num_intents=d.get("num_intents", 0),
            num_questions=d.get("num_questions", 0),
            seed=d.get("seed", DEFAULT_RANDOM_SEED),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize QSim run metadata.

        Returns:

            `asdict` of version, counts, and seed.
        """
        return asdict(self)


@dataclass
class SchemaLimits:
    """Internal schema-based limits for adaptive parameter validation."""

    max_filters: int
    max_groupby: int
    max_tables: int


@dataclass
class SkeletonLimits:
    """Schema-derived limits for QSim skeleton enumeration."""

    max_filters: int
    max_groupby: int
    max_having: int


class AccessError(RuntimeError):
    """Raised when the database refuses ``EXPLAIN`` or ``execute`` because of insufficient privileges."""

    def __init__(
        self,
        operation: Literal["explain", "execute"],
        message: str,
        *,
        relation: str | None = None,
    ) -> None:
        """Attach *operation*, human *message*, and optional *relation* hint for UX."""

        self.operation = operation
        self.relation = relation
        super().__init__(message)


class PipelineSuspended(Exception):
    """Raised when a programmatic interactive turn must wait for the next ``submit_*`` call."""

    def __init__(
        self,
        state_id: str,
        message_for_caller: str,
        payload: Any | None = None,
    ) -> None:
        self.state_id = state_id
        self.message_for_caller = message_for_caller
        self.payload = payload
        super().__init__(message_for_caller)


class NoJoinPathError(Exception):
    """
    Raised when multi-table scope has no foreign-key or semantic join path.

    This is a terminal, deterministic pipeline failure: no LLM call can invent a plausible join when neither the physical foreign-key graph nor the semantic edge set connects the requested tables.
    """

    def __init__(
        self,
        scope_label: str,
        tables: list[str],
    ) -> None:
        self.scope_label = scope_label
        self.tables = list(tables)
        message = (
            f"No join path available in {scope_label} for tables: {', '.join(self.tables) if self.tables else '<none>'}"
        )
        super().__init__(message)


class JoinInjectionAlignmentError(Exception):
    """Raised when ``join_sigs_ordered`` does not align one-to-one with dialect join carriers on deterministic SQL."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class JoinInjectionFailedError(Exception):
    """Raised when deterministic SQL cannot be rewritten with structured JOIN/WHERE edges via the dialect AST adapter."""

    def __init__(
        self,
        message: str,
        *,
        det_sql: str,
        join_sigs_ordered: list[list[str]],
        edge_kinds_ordered: list[list[str]],
    ) -> None:
        self.det_sql = det_sql
        self.join_sigs_ordered = join_sigs_ordered
        self.edge_kinds_ordered = edge_kinds_ordered
        super().__init__(message)


class LlmJsonExhausted(Exception):
    """
    Raised by ``llm_json`` when every retry attempt fails to produce valid JSON.

    Callers decide whether exhaustion is recoverable (e.g., retry loops, deterministic fallbacks) or terminal.
    """

    def __init__(self, task: str, attempts: int) -> None:
        self.task = task
        self.attempts = attempts
        super().__init__(f"llm_json exhausted after {attempts} attempt(s) for task={task!r}")
