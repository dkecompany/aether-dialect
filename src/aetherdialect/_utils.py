"""Hashing, stable JSON, SQL cleanup, artifact manifest I/O, LLM clients, and display helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

if sys.platform == "win32":
    pass
else:
    pass


import aetherdialect._constants

from ._config import (
    EngineLimits,
    EngineRuntimeConfig,
    FederationLimits,
    PolicyConfig,
)
from ._constants import (
    AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV,
    AETHERSPACE_SCOPE_MARKERS,
    AUDIT_EVENT_SQL_EXECUTION,
    DATABASE_ERROR_CLASSIFICATION_BY_EXCEPTION_NAME,
    DATABASE_ERROR_CLASSIFICATION_BY_MESSAGE_PATTERN,
    DATABASE_ERROR_CLASSIFICATION_TRANSIENT,
    DATABASE_ERROR_CLASSIFICATION_TRANSIENT_ERRNOS,
    DATABASE_ERROR_CLASSIFICATION_UNKNOWN,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    DIAGNOSTIC_CODE_FEDERATION_POOL_UNDERSIZED,
    DIAGNOSTIC_CODE_LLM_TURN_COST,
    DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED,
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    DOMAIN_KNOWLEDGE_FILENAME,
    EGRESS_STRIPPED_DETAIL_KEYS,
    ENGINE_DRIVER_REQUIREMENTS,
    EXACT_NUMERIC_BASE_TYPES,
    EXECUTION_SCOPE_MARKERS,
    FEDERATION_EGRESS_STRIPPED_DETAIL_KEYS,
    INEXACT_NUMERIC_BASE_TYPES,
    ISO_DATE_ONLY_RE,
    ISO_DATETIME_RE,
    LIKE_ESCAPE_CHAR,
    LLM_PRICE_PER_MILLION,
    LLM_PRICE_TABLE_AS_OF,
    MASTER_AETHERSPACE_NAME,
    MASTER_AETHERSPACE_UID,
    NUMERIC_TYPE_ARGUMENTS_RE,
    OUTCOME_REFUSAL_CODES,
    PERMISSION_DENIED_CATEGORY_ORACLE_KINDS,
    PERMISSION_DENIED_FAILURE_KINDS,
    PRE_QUOTED_IN_LIST_INLINE_RE,
    REFUSAL_CAPABILITY_GAP_REASON_CODES,
    REFUSAL_CAPABILITY_GAP_REASON_PREFIXES,
    REFUSAL_CTE_CAP_ISSUE_IDS,
    REFUSAL_NULL_IN_NEGATED_LIST_ISSUE_IDS,
    REFUSAL_TIMING_FLOOR_MS,
    REFUSAL_UNSUPPORTED_COLUMN_TYPE_ISSUE_IDS,
    REMEDIATION_SCOPE_MECHANISM_MARKERS,
    REPHRASE_HINT_REFUSAL_CODES,
    SCOPE_SENSITIVITY_FAILURE_KINDS,
    SCOPE_SENSITIVITY_REFUSAL_CODES,
    SQL_BIND_TOKEN_RE,
    SQL_EXPONENT_LITERAL_RE,
    SQL_FIXED_POINT_LITERAL_RE,
    SQL_INTEGER_LITERAL_RE,
    SQL_STRING_LITERAL_COMMENT_MARKERS,
    SQL_STRING_LITERAL_STATEMENT_TERMINATOR,
    STRUCTURAL_IDENTITY_VALUES,
    STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE,
    STRUCTURAL_KNOWLEDGE_FILENAME,
    STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE,
    UNBOUND_PYFORMAT_PLACEHOLDER_RE,
    UNKNOWN_VALUE_TYPE,
    UNSAFE_PARAM_LITERAL,
    VALID_VALUE_TYPES,
    VALUE_TYPE_NORMALIZATION,
)
from ._constants_runtime import (
    PERMISSION_DENIED_USER_MESSAGE,
    QUERY_RESULTS_HEADER,
    REFUSAL_CATALOGUE,
    REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE,
    REMEDIATION_RESTRICTED_QUESTION,
    REPHRASE_HINT_MESSAGES,
    USER_ERROR_PREFIX,
    USER_INVALID_INPUT_LINE,
    USER_REJECTED_RESULT_BUCKET_TIPS,
    USER_TERMINATED_LINE,
)
from ._contracts_base import (
    ColumnTypeSemantics,
    ConfigError,
    DatabaseErrorClassification,
    DatabaseExecutionError,
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    DomainKnowledgeEntry,
    DomainKnowledgeKind,
    DomainKnowledgeState,
    EngineContext,
    EngineIdentity,
    FailureCategory,
    FederationContext,
    KnowledgeScope,
    OverlapComparison,
    ReflectMode,
    RetryableDatabaseExecutionError,
    SchemaInvariantError,
    SchemaNaming,
)
from ._contracts_core import (
    AccessError,
    AggregateJoinFanOutError,
    AmbiguousDateLiteralError,
    ClauseWidenedRowsetError,
    ComparisonJoinScopeExceededError,
    FederatedSqlBundle,
    FederatedStatementRecord,
    FederationExecutionContext,
    InteractiveChoicePort,
    JoinPathTieCapExceededError,
    LlmExecutionConfig,
    LlmTurnUsageSummary,
    LlmUsageRecord,
    NoJoinPathError,
    NullInNegatedListError,
    PhaseProgressEvent,
    ProbeCtePlacementError,
    RefusalCatalogueEntry,
    RephraseHint,
    RuntimeIntent,
    SessionError,
    SessionOutcome,
    SessionStep,
    StepResult,
    SubdayDateWindowOnDateColumnError,
)
from ._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    IntentIssue,
    SchemaGraph,
)


def normalize_column_type(col_type: str) -> str:
    """Lowercase a SQL type and remove `(n)` / `(n,m)` parameter lists."""
    return ColumnTypeSemantics.normalize_column_type(col_type)


def split_column_type(col_type: str) -> tuple[str, int | None, int | None]:
    """Lowercase a SQL type, strip parameter lists for lookup, and return parsed precision and scale."""
    return ColumnTypeSemantics.split_column_type(col_type)


def structural_data_type_key(data_type: str) -> str:
    """Return a canonical catalog-type token for structural diff and hashing."""
    return ColumnTypeSemantics.structural_data_type_key(data_type)


def column_is_unsigned_from_data_type(
    data_type: str,
    *,
    reflected_unsigned: bool | None = None,
) -> bool:
    """Return whether a SQL column type carries unsigned integer semantics."""
    return ColumnTypeSemantics.column_is_unsigned_from_data_type(
        data_type,
        reflected_unsigned=reflected_unsigned,
    )


def column_is_fixed_width_text_from_data_type(data_type: str) -> bool:
    """Return whether a SQL column type is fixed-width character text (CHAR/NCHAR)."""
    return ColumnTypeSemantics.column_is_fixed_width_text_from_data_type(data_type)


def column_timezone_aware_from_data_type(
    data_type: str,
    *,
    engine: str | None = None,
) -> bool:
    """Return whether a SQL temporal type preserves timezone offsets."""
    return ColumnTypeSemantics.column_timezone_aware_from_data_type(data_type, engine=engine)


def column_unsigned_near_type_max(meta: Any) -> bool:
    """Return whether profiled maxima exceed float-safe range or approach an unsigned ceiling."""
    return ColumnTypeSemantics.column_unsigned_near_type_max(meta)


def is_numeric_type(data_type: str) -> bool:
    """Return whether a SQL data type string looks numeric."""
    return ColumnTypeSemantics.is_numeric_type(data_type)


def is_string_type(data_type: str) -> bool:
    """Return whether a SQL data type string looks string-like."""
    return ColumnTypeSemantics.is_string_type(data_type)


def is_date_type(data_type: str) -> bool:
    """Return whether a SQL data type string looks date- or time- like."""
    return ColumnTypeSemantics.is_date_type(data_type)


def data_type_to_value_type(data_type: str) -> str:
    """Map a SQL data type string to a prompt/value-type token."""
    return ColumnTypeSemantics.data_type_to_value_type(data_type)


def norm_schema_identifier(name: str, *, what: str) -> str:
    """Lowercase and strip *name*; raise when empty after strip."""
    return SchemaNaming.norm_schema_identifier(name, what=what)


def normalize_scope_column_spec(spec: str, *, field: str) -> str:
    """Normalize a ``table.column`` or ``source.table.column`` scope column spec."""
    return SchemaNaming.normalize_scope_column_spec(spec, field=field)


_DIAGNOSTIC_COLLECTOR: ContextVar[list[Diagnostic] | None] = ContextVar(
    "aetherdialect_diagnostic_collector",
    default=None,
)

_PENDING_INTENT_PARSE_REFUSAL: ContextVar[tuple[str, str] | None] = ContextVar(
    "aetherdialect_pending_intent_parse_refusal",
    default=None,
)

_ORPHAN_DIAGNOSTICS_LOCK = threading.Lock()
_ORPHAN_DIAGNOSTICS: list[tuple[EngineIdentity, Diagnostic]] = []

_DIAGNOSTIC_PRINT_LISTENER: ContextVar[Callable[[str], None] | None] = ContextVar(
    "aetherdialect_diagnostic_print_listener",
    default=None,
)

_DIAGNOSTIC_SINK: ContextVar[Callable[[Diagnostic], None] | None] = ContextVar(
    "aetherdialect_diagnostic_sink",
    default=None,
)


def push_diagnostic_sink(
    sink: Callable[[Diagnostic], None] | None,
) -> Token[Callable[[Diagnostic], None] | None]:
    """Bind *sink* to receive every :class:`Diagnostic` emitted by :func:`notify` within the active scope."""
    return _DIAGNOSTIC_SINK.set(sink)


def pop_diagnostic_sink(token: Token[Callable[[Diagnostic], None] | None]) -> None:
    """Restore the prior diagnostic sink."""
    _DIAGNOSTIC_SINK.reset(token)


LLM_EXECUTION_CONTEXT: ContextVar[LlmExecutionConfig | None] = ContextVar(
    "aetherdialect_llm_execution",
    default=None,
)

_ACTIVE_ENGINE_IDENTITY: ContextVar[EngineIdentity | None] = ContextVar(
    "aetherdialect_active_engine_identity",
    default=None,
)

_PENDING_CONSTRUCTION_ENGINE_IDENTITY: ContextVar[EngineIdentity | None] = ContextVar(
    "aetherdialect_pending_construction_engine_identity",
    default=None,
)

_FEDERATION_EXECUTION_CONTEXT: ContextVar[FederationExecutionContext | None] = ContextVar(
    "aetherdialect_federation_execution_context",
    default=None,
)


def parse_numeric_type_arguments(data_type: str) -> tuple[int | None, int | None]:
    """Parse precision and optional scale from a SQL numeric type string."""
    match = NUMERIC_TYPE_ARGUMENTS_RE.search(data_type)
    if not match:
        return None, None
    precision = int(match.group(1))
    scale = int(match.group(2)) if match.group(2) is not None else None
    return precision, scale


def collation_name_is_case_insensitive(name: str) -> bool:
    """Return True when a reflected collation name compares strings case-insensitively."""
    token = (name or "").strip().casefold()
    if not token:
        return False
    if token in {"c", "posix"}:
        return False
    if token.endswith("_ci") or token.endswith("_ci_as") or token.endswith("_ci_ai"):
        return True
    if "_cs" in token:
        return False
    if "_ci" in token:
        return True
    return False


def is_exact_numeric_data_type(data_type: str) -> bool:
    """Return whether a SQL numeric type preserves exact values."""
    base = normalize_column_type(data_type)
    if base in INEXACT_NUMERIC_BASE_TYPES:
        return False
    if base in EXACT_NUMERIC_BASE_TYPES:
        return True
    lowered = data_type.lower()
    if any(token in lowered for token in ("float", "double", "real")):
        return False
    return False


def column_numeric_metadata_from_data_type(
    data_type: str,
    *,
    reflected_precision: int | None = None,
    reflected_scale: int | None = None,
) -> tuple[int | None, int | None, bool]:
    """Derive precision, scale and exactness for a column type string."""
    precision = reflected_precision
    scale = reflected_scale
    if precision is None and scale is None:
        precision, scale = parse_numeric_type_arguments(data_type)
    return precision, scale, is_exact_numeric_data_type(data_type)


def column_metadata_requires_exact_comparison(meta: ColumnMetadata) -> bool:
    """Return whether comparisons must avoid binary-float widening for *meta*."""
    if meta.is_exact_numeric:
        return True
    return column_unsigned_near_type_max(meta)


def column_metadata_timezone_awareness_mismatch(left_meta: ColumnMetadata, right_meta: ColumnMetadata) -> bool:
    """Return True when two temporal columns disagree on timezone awareness."""
    if (left_meta.value_type or "").strip().lower() != "date":
        return False
    if (right_meta.value_type or "").strip().lower() != "date":
        return False
    return left_meta.is_timezone_aware != right_meta.is_timezone_aware


def parse_sql_numeric_literal(text: str) -> Decimal | int | float:
    """Parse a SQL numeric literal without widening exact values to binary floats."""
    stripped = text.strip()
    if SQL_INTEGER_LITERAL_RE.match(stripped):
        return int(stripped)
    if SQL_FIXED_POINT_LITERAL_RE.match(stripped):
        return Decimal(stripped)
    if SQL_EXPONENT_LITERAL_RE.match(stripped):
        return float(stripped)
    return Decimal(stripped)


def render_sql_numeric_literal(text: str, parsed: Decimal | int | float) -> str:
    """Render a parsed numeric literal back to SQL text."""
    return text


def render_sql_numeric_value(value: Decimal | int | float) -> str:
    """Render a numeric value for SQL when the original literal text is unavailable."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    return str(value)


def active_engine_identity() -> EngineIdentity:
    """Return the active engine identity for this execution context."""
    active = _ACTIVE_ENGINE_IDENTITY.get()
    if active is not None:
        return active
    raise RuntimeError(
        "no active engine identity; bind one with push_engine_identity before calling active_engine_identity"
    )


def active_engine_runtime_config() -> EngineRuntimeConfig:
    """Return the active engine runtime config for this execution context."""
    identity = active_engine_identity()
    runtime = identity.runtime_config
    if isinstance(runtime, type):
        return EngineRuntimeConfig.process_default_for_class(runtime)
    if not isinstance(runtime, EngineRuntimeConfig):
        raise TypeError(
            "active engine identity carries an unsupported runtime config value; "
            f"expected EngineRuntimeConfig, got {type(runtime).__name__}"
        )
    return runtime


def resolve_runtime_config(
    config: EngineRuntimeConfig | type[EngineRuntimeConfig] | None = None,
) -> EngineRuntimeConfig:
    """Resolve a runtime config instance, preferring an explicit value then the active identity."""
    if config is not None and not isinstance(config, type):
        return config
    if config is not None and isinstance(config, type):
        return config()
    try:
        return active_engine_runtime_config()
    except RuntimeError as exc:
        raise RuntimeError(
            "no active engine identity; pass a runtime config instance instead of relying on implicit defaults"
        ) from exc


def bound_engine_runtime_config() -> EngineRuntimeConfig:
    """Return the active engine runtime config; never guess a process default."""
    try:
        return active_engine_runtime_config()
    except RuntimeError as exc:
        raise RuntimeError(
            "no active engine identity; pass a runtime config instance instead of relying on implicit defaults"
        ) from exc


def push_engine_identity(identity: EngineIdentity) -> Token[EngineIdentity | None]:
    """Bind *identity* for nested pipeline and SQL generation calls."""
    return _ACTIVE_ENGINE_IDENTITY.set(identity)


def pop_engine_identity(token: Token[EngineIdentity | None]) -> None:
    """Restore the prior engine identity after :func:`push_engine_identity`."""
    _ACTIVE_ENGINE_IDENTITY.reset(token)


def bind_construction_orphan_identity(identity: EngineIdentity) -> Token[EngineIdentity | None]:
    """Bind *identity* for construction-time diagnostics emitted without an active collector."""
    return _PENDING_CONSTRUCTION_ENGINE_IDENTITY.set(identity)


def release_construction_orphan_identity(token: Token[EngineIdentity | None]) -> None:
    """Clear a construction-time orphan identity binding."""
    _PENDING_CONSTRUCTION_ENGINE_IDENTITY.reset(token)


def require_driver(engine_name: str) -> None:
    """Import the driver for *engine_name* or raise :class:`ConfigError` with install guidance."""
    spec = ENGINE_DRIVER_REQUIREMENTS.get(str(engine_name).strip().lower())
    if spec is None:
        return
    import_names, _distribution, extra_name = spec
    if isinstance(import_names, str):
        import_names = (import_names,)
    last_exc: ImportError | None = None
    for import_name in import_names:
        try:
            __import__(import_name)
            return
        except ImportError as exc:
            last_exc = exc
    driver_label = " or ".join(import_names)
    if len(import_names) > 1:
        raise ConfigError(f"pip install aetherdialect[{extra_name}] (requires {driver_label})") from last_exc
    raise ConfigError(f"pip install aetherdialect[{extra_name}]") from last_exc


_ACTIVE_ENGINE_LIMITS: ContextVar[EngineLimits | None] = ContextVar("aetherdialect_active_engine_limits", default=None)
_ACTIVE_FEDERATION_LIMITS: ContextVar[FederationLimits | None] = ContextVar(
    "aetherdialect_active_federation_limits", default=None
)


def push_engine_limits(limits: EngineLimits) -> Token[EngineLimits | None]:
    """Bind *limits* for nested execution calls."""
    return _ACTIVE_ENGINE_LIMITS.set(limits)


def pop_engine_limits(token: Token[EngineLimits | None]) -> None:
    """Restore the prior engine limits binding."""
    _ACTIVE_ENGINE_LIMITS.reset(token)


def active_engine_limits() -> EngineLimits:
    """Return the active engine limits for this execution context."""
    limits = _ACTIVE_ENGINE_LIMITS.get()
    if limits is None:
        raise RuntimeError(
            "no active engine limits; bind one with push_engine_limits before calling active_engine_limits"
        )
    return limits


def resolved_engine_limits() -> EngineLimits:
    """Return active engine limits when bound, otherwise :class:`EngineLimits` defaults."""
    limits = _ACTIVE_ENGINE_LIMITS.get()
    return limits if limits is not None else EngineLimits()


def push_federation_limits(limits: FederationLimits) -> Token[FederationLimits | None]:
    """Bind federation limits for nested federation execution."""
    return _ACTIVE_FEDERATION_LIMITS.set(limits)


def pop_federation_limits(token: Token[FederationLimits | None]) -> None:
    """Restore the prior federation limits binding."""
    _ACTIVE_FEDERATION_LIMITS.reset(token)


def active_federation_limits() -> FederationLimits:
    """Return the active federation limits for this execution context."""
    limits = _ACTIVE_FEDERATION_LIMITS.get()
    if limits is None:
        raise RuntimeError(
            "no active federation limits; bind one with push_federation_limits before calling active_federation_limits"
        )
    return limits


def resolved_federation_limits() -> FederationLimits:
    """Return active federation limits when bound, otherwise :class:`FederationLimits` defaults."""
    limits = _ACTIVE_FEDERATION_LIMITS.get()
    return limits if limits is not None else FederationLimits()


def engine_limits_for_owner(owner: Any) -> EngineLimits:
    """Resolve :class:`EngineLimits` for an engine or federation owner object."""
    raw = getattr(owner, "limits", None)
    if isinstance(raw, EngineLimits):
        return raw
    if isinstance(raw, FederationLimits):
        defaults = raw.member_defaults
        return defaults if isinstance(defaults, EngineLimits) else EngineLimits()
    return EngineLimits()


def federation_limits_for_owner(owner: Any) -> FederationLimits | None:
    """Return :class:`FederationLimits` when *owner* is a federation, else ``None``."""
    raw = getattr(owner, "limits", None)
    return raw if isinstance(raw, FederationLimits) else None


@contextmanager
def owner_limits_scope(owner: Any) -> Iterator[None]:
    """Bind engine limits (and federation limits when applicable) for the duration of a turn."""
    engine_token = push_engine_limits(engine_limits_for_owner(owner))
    federation = federation_limits_for_owner(owner)
    federation_token = push_federation_limits(federation) if federation is not None else None
    try:
        yield
    finally:
        if federation_token is not None:
            pop_federation_limits(federation_token)
        pop_engine_limits(engine_token)


def apply_federation_member_defaults(
    members: Mapping[str, Any],
    federation_limits: FederationLimits,
) -> None:
    """Apply ``member_defaults`` to member engines constructed without explicit ``limits=``."""
    defaults = federation_limits.member_defaults
    if defaults is None:
        return
    for engine in members.values():
        if getattr(engine, "_limits_explicit", False):
            continue
        if hasattr(engine, "_limits"):
            engine._limits = defaults


def sqlalchemy_pool_kwargs_from_limits(
    limits: EngineLimits,
    *,
    single_connection_pool: bool = False,
) -> dict[str, Any]:
    """Return SQLAlchemy pool keyword arguments for *limits*."""
    kwargs: dict[str, Any] = {
        "pool_recycle": limits.pool_recycle_seconds,
        "pool_pre_ping": limits.pool_pre_ping,
    }
    if not single_connection_pool:
        kwargs["pool_size"] = limits.pool_size
        kwargs["max_overflow"] = limits.pool_max_overflow
        kwargs["pool_timeout"] = limits.pool_timeout_seconds
    return kwargs


def sqlalchemy_url_uses_single_connection_pool(url: str) -> bool:
    """Return True when *url* targets an embedded engine with a one- connection pool."""
    lowered = str(url or "").strip().lower()
    return lowered.startswith("duckdb") or lowered.startswith("sqlite")


def narrow_member_engine_limits(member_limits: EngineLimits, federation_limits: FederationLimits) -> EngineLimits:
    """Keep caller-supplied member limits unless federation row cap is stricter."""
    if federation_limits.member_row_cap is None:
        return member_limits
    member_cap = member_limits.max_result_rows
    if member_cap is not None and member_cap <= federation_limits.member_row_cap:
        return member_limits
    narrowed = replace(member_limits, max_result_rows=federation_limits.member_row_cap)
    notify(
        "federation member row cap narrowed to match federation limit",
        stage="federation",
        code=DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED,
        level="warning",
        details=(("field", "max_result_rows"), ("value", str(federation_limits.member_row_cap))),
    )
    return narrowed


def validate_federation_pool_capacity(members: Mapping[str, Any], federation_limits: FederationLimits) -> None:
    """Emit a diagnostic when aggregate pool capacity is below parallel member execution."""
    aggregate = 0
    for engine in members.values():
        limits = getattr(engine, "limits", EngineLimits())
        aggregate += int(limits.pool_size) + int(limits.pool_max_overflow)
    needed = int(federation_limits.max_parallel_members)
    if aggregate >= needed:
        return
    notify(
        (f"federation aggregate pool capacity {aggregate} is below max_parallel_members {needed}"),
        stage="federation",
        code=DIAGNOSTIC_CODE_FEDERATION_POOL_UNDERSIZED,
        level="warning",
        source_id="",
        details=(
            ("phase", "composition"),
            ("aggregate_pool_capacity", str(aggregate)),
            ("max_parallel_members", str(needed)),
        ),
    )


def push_federation_execution_context(
    ctx: FederationExecutionContext,
) -> Token[FederationExecutionContext | None]:
    """Bind *ctx* for nested federated member worker calls."""
    return _FEDERATION_EXECUTION_CONTEXT.set(ctx)


def pop_federation_execution_context(
    token: Token[FederationExecutionContext | None],
) -> None:
    """Restore the prior federation execution context."""
    _FEDERATION_EXECUTION_CONTEXT.reset(token)


def active_federation_execution_context() -> FederationExecutionContext | None:
    """Return the active federation execution context, if any."""
    return _FEDERATION_EXECUTION_CONTEXT.get()


def federation_turn_cancelled() -> bool:
    """Return True when the active federated turn has been cancelled."""
    ctx = _FEDERATION_EXECUTION_CONTEXT.get()
    return bool(ctx is not None and ctx.cancelled)


_SESSION_TURN_CANCEL: ContextVar[threading.Event | None] = ContextVar(
    "aetherdialect_session_turn_cancel",
    default=None,
)
_TURN_ID: ContextVar[str | None] = ContextVar(
    "aetherdialect_turn_id",
    default=None,
)
_TURN_START_MONO: ContextVar[float | None] = ContextVar(
    "aetherdialect_turn_start_mono",
    default=None,
)
_ASK_PHASE_LAST_EMIT_MONO: ContextVar[float | None] = ContextVar(
    "aetherdialect_ask_phase_last_emit_mono",
    default=None,
)


def push_session_turn_cancel(event: threading.Event) -> Token[threading.Event | None]:
    """Bind *event* for cooperative cancellation of the active session turn."""
    return _SESSION_TURN_CANCEL.set(event)


def pop_session_turn_cancel(token: Token[threading.Event | None]) -> None:
    """Restore the prior session turn cancellation event."""
    _SESSION_TURN_CANCEL.reset(token)


def session_turn_cancelled() -> bool:
    """Return True when the active session turn has been cancelled."""
    event = _SESSION_TURN_CANCEL.get()
    return bool(event is not None and event.is_set())


def mint_turn_id() -> str:
    """Return a new opaque correlation id for one interactive ask turn."""
    return str(uuid.uuid4())


def push_turn_id(turn_id: str) -> Token[str | None]:
    """Bind *turn_id* for audit, diagnostic, and phase correlation within the active turn."""
    return _TURN_ID.set(turn_id)


def pop_turn_id(token: Token[str | None]) -> None:
    """Restore the prior turn correlation id."""
    _TURN_ID.reset(token)


def active_turn_id() -> str | None:
    """Return the active ask-turn correlation id when one is bound."""
    return _TURN_ID.get()


def push_turn_timing() -> tuple[Token[float | None], Token[float | None]]:
    """Bind turn-start and ask-phase emit clocks for one interactive ask turn."""
    now = time.perf_counter()
    return _TURN_START_MONO.set(now), _ASK_PHASE_LAST_EMIT_MONO.set(now)


def pop_turn_timing(
    start_token: Token[float | None],
    emit_token: Token[float | None],
) -> None:
    """Restore the prior turn timing clocks."""
    _ASK_PHASE_LAST_EMIT_MONO.reset(emit_token)
    _TURN_START_MONO.reset(start_token)


def turn_elapsed_ms() -> int | None:
    """Return milliseconds since the active ask turn started, if any."""
    start = _TURN_START_MONO.get()
    if start is None:
        return None
    return max(0, int((time.perf_counter() - start) * 1000))


def apply_refusal_timing_floor(elapsed_ms: int | None) -> int:
    """Pad refusal-terminal ``elapsed_ms`` so fast paths are not timing- distinguishable."""
    raw = max(0, int(elapsed_ms or 0))
    return max(REFUSAL_TIMING_FLOOR_MS, raw)


def details_with_turn_id(details: tuple[tuple[str, str], ...] = ()) -> tuple[tuple[str, str], ...]:
    """Prepend ``turn_id`` to *details* when a turn correlation id is active."""
    turn_id = active_turn_id()
    if turn_id is None:
        return details
    if any(key == "turn_id" for key, _ in details):
        return details
    return (("turn_id", turn_id),) + details


_CONSTRUCTION_PHASE_CALLBACK: ContextVar[Callable[[PhaseProgressEvent], None] | None] = ContextVar(
    "aetherdialect_construction_phase_callback",
    default=None,
)
_ASK_PHASE_CALLBACK: ContextVar[Callable[[PhaseProgressEvent], None] | None] = ContextVar(
    "aetherdialect_ask_phase_callback",
    default=None,
)
_AUDIT_EMIT: ContextVar[Callable[..., None] | None] = ContextVar(
    "aetherdialect_audit_emit",
    default=None,
)
_PIPELINE_TRACE_SINK: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "aetherdialect_pipeline_trace_sink",
    default=None,
)


def push_construction_phase_callback(
    callback: Callable[[PhaseProgressEvent], None] | None,
) -> Token[Callable[[PhaseProgressEvent], None] | None]:
    """Bind *callback* for construction-phase progress reporting."""
    return _CONSTRUCTION_PHASE_CALLBACK.set(callback)


def pop_construction_phase_callback(token: Token[Callable[[PhaseProgressEvent], None] | None]) -> None:
    """Restore the prior construction-phase progress callback."""
    _CONSTRUCTION_PHASE_CALLBACK.reset(token)


def push_ask_phase_callback(
    callback: Callable[[PhaseProgressEvent], None] | None,
) -> Token[Callable[[PhaseProgressEvent], None] | None]:
    """Bind *callback* for ask-turn phase progress reporting."""
    return _ASK_PHASE_CALLBACK.set(callback)


def pop_ask_phase_callback(token: Token[Callable[[PhaseProgressEvent], None] | None]) -> None:
    """Restore the prior ask-turn phase progress callback."""
    _ASK_PHASE_CALLBACK.reset(token)


def push_audit_emit(
    callback: Callable[..., None] | None,
) -> Token[Callable[..., None] | None]:
    """Bind *callback* for lifecycle audit events within the active turn or scope."""
    return _AUDIT_EMIT.set(callback)


def pop_audit_emit(token: Token[Callable[..., None] | None]) -> None:
    """Restore the prior audit emit callback."""
    _AUDIT_EMIT.reset(token)


def active_audit_emit() -> Callable[..., None] | None:
    """Return the active audit emit callback, if any."""
    return _AUDIT_EMIT.get()


def emit_sql_execution_audit(
    *,
    statement_hash: str,
    row_count: int,
    elapsed_ms: int,
    schema_hash: str | None = None,
) -> None:
    """Emit a ``sql_execution`` audit event when an audit sink is bound."""
    fn = active_audit_emit()
    if not callable(fn):
        return

    fn(
        AUDIT_EVENT_SQL_EXECUTION,
        schema_hash=schema_hash,
        details=details_with_turn_id(
            (
                ("statement_hash", statement_hash),
                ("row_count", str(max(0, int(row_count)))),
                ("elapsed_ms", str(max(0, int(elapsed_ms)))),
            )
        ),
        turn_id=active_turn_id(),
    )


def emit_construction_phase(
    phase: str,
    *,
    source: str | None = None,
    stage: int | None = None,
) -> None:
    """Invoke the active construction-phase callback, if any."""
    callback = _CONSTRUCTION_PHASE_CALLBACK.get()
    if callback is None:
        return
    callback(
        PhaseProgressEvent(
            phase=phase,
            timestamp_iso=datetime.now(UTC).isoformat(),
            source=source,
            stage=stage,
        )
    )


def emit_ask_phase(
    phase: str,
    *,
    source: str | None = None,
    stage: int | str | None = None,
) -> None:
    """Invoke the active ask-phase callback, if any."""
    callback = _ASK_PHASE_CALLBACK.get()
    if callback is None:
        return
    now = time.perf_counter()
    last = _ASK_PHASE_LAST_EMIT_MONO.get()
    if last is None:
        last = _TURN_START_MONO.get()
    elapsed_ms = max(0, int((now - last) * 1000)) if last is not None else 0
    _ASK_PHASE_LAST_EMIT_MONO.set(now)
    callback(
        PhaseProgressEvent(
            phase=phase,
            timestamp_iso=datetime.now(UTC).isoformat(),
            source=source,
            stage=stage,
            turn_id=active_turn_id(),
            elapsed_ms=elapsed_ms,
        )
    )


def coerce_format_version(value: Any) -> str:
    """Normalize a stored format/package version to a canonical string. Accepts package-style strings (``"0.2.1"``), integers (``1`` → ``"1"``), and simple dotted numeric forms (``1.0`` / ``"1.0"`` → ``"1.0"``)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        raise TypeError("boolean is not a format version")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        text = f"{value}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    text = str(value).strip()
    return text


def format_versions_match(stored: Any, expected: str) -> bool:
    """Return True when *stored* equals *expected* after :func:`coerce_format_version`. Also treats ``1`` / ``"1"`` / ``"1.0"`` as equivalent when both sides are numeric-only dotted segments (shorter side padded with zeros)."""
    left = coerce_format_version(stored)
    right = coerce_format_version(expected)
    if left == right:
        return True

    def _parts(s: str) -> tuple[str, ...]:
        return tuple(p for p in s.split(".") if p != "")

    lp, rp = _parts(left), _parts(right)
    if not lp or not rp:
        return False
    if all(p.isdigit() for p in lp + rp):
        n = max(len(lp), len(rp))
        lp2 = lp + ("0",) * (n - len(lp))
        rp2 = rp + ("0",) * (n - len(rp))
        return lp2 == rp2
    return False


def is_structural_param_key(key: str) -> bool:
    """Return True when *key* is a structural bind name (``s`` followed by digits)."""
    return len(key) >= 2 and key[0] == "s" and key[1:].isdigit()


def effective_explain_timeout_ms() -> int | None:
    """Statement timeout for ``EXPLAIN`` paths only. Prefers :data:`PolicyConfig.EXPLAIN_TIMEOUT_MS` when set and positive; otherwise uses the active :class:`EngineLimits` statement timeout when bound, else :class:`EngineLimits` defaults. Returns ``None`` when no timeout is active."""
    explain_tm = PolicyConfig.EXPLAIN_TIMEOUT_MS
    if cost_cap_active(explain_tm) and explain_tm is not None:
        return int(explain_tm)
    statement_tm = resolved_engine_limits().statement_timeout_ms
    if cost_cap_active(statement_tm) and statement_tm is not None:
        return int(statement_tm)
    return None


def effective_statement_timeout_ms() -> int | None:
    """Resolved statement timeout for execute paths from active :class:`EngineLimits`."""
    statement_tm = resolved_engine_limits().statement_timeout_ms
    if cost_cap_active(statement_tm) and statement_tm is not None:
        return int(statement_tm)
    return None


def effective_profile_timeout_ms() -> int | None:
    """Resolved profile timeout from active :class:`EngineLimits`."""
    profile_tm = resolved_engine_limits().profile_timeout_ms
    if cost_cap_active(profile_tm) and profile_tm is not None:
        return int(profile_tm)
    return None


def effective_llm_timeout_ms() -> int:
    """Resolved HTTP timeout for OpenAI-compatible clients and :func:`aetherdialect._llm_provider.LLMProvider.chat`. Uses:data:`PolicyConfig.LLM_TIMEOUT_MS` when positive; otherwise ``60_000`` ms."""
    tm = PolicyConfig.LLM_TIMEOUT_MS
    if cost_cap_active(tm) and tm is not None:
        return int(tm)
    return 60_000


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


@contextmanager
def llm_execution_scope(cfg: LlmExecutionConfig) -> Iterator[None]:
    """Bind *cfg* as the active :class:`LlmExecutionConfig` for nested LLM calls."""
    tok = LLM_EXECUTION_CONTEXT.set(cfg)
    try:
        yield
    finally:
        LLM_EXECUTION_CONTEXT.reset(tok)


def diagnostic_segment() -> list[Diagnostic]:
    """Return a new collector list and bind it as the active diagnostic buffer (call :func:`reset_diagnostic_collector` when done)."""
    buf: list[Diagnostic] = []
    return buf


def set_diagnostic_collector(buf: list[Diagnostic] | None) -> Any:
    """Bind *buf* as the active diagnostic collector; returns a token for :func:`reset_diagnostic_collector`."""
    return _DIAGNOSTIC_COLLECTOR.set(buf)


def reset_diagnostic_collector(token: Any) -> None:
    """Restore the previous diagnostic collector."""
    _DIAGNOSTIC_COLLECTOR.reset(token)


def _orphan_identity_key(identity: EngineIdentity) -> tuple[str, int]:
    """Stable per-engine key: type plus runtime config object identity."""
    return (identity.engine_type, id(identity.runtime_config))


def take_and_clear_orphan_diagnostics(identity: EngineIdentity) -> tuple[Diagnostic, ...]:
    """Drain construction-time diagnostics keyed to *identity*."""
    target_key = _orphan_identity_key(identity)
    with _ORPHAN_DIAGNOSTICS_LOCK:
        matched: list[Diagnostic] = []
        remaining: list[tuple[EngineIdentity, Diagnostic]] = []
        for entry_identity, diag in _ORPHAN_DIAGNOSTICS:
            if _orphan_identity_key(entry_identity) == target_key:
                matched.append(diag)
            else:
                remaining.append((entry_identity, diag))
        _ORPHAN_DIAGNOSTICS[:] = remaining
        return tuple(matched)


@contextmanager
def diagnostic_print_listener(fn: Callable[[str], None] | None) -> Iterator[None]:
    """Bind *fn* to receive human-readable copies of notify lines (used internally by ``run_interactive`` and other CLI-style entry points)."""
    tok = _DIAGNOSTIC_PRINT_LISTENER.set(fn)
    try:
        yield
    finally:
        _DIAGNOSTIC_PRINT_LISTENER.reset(tok)


def require_exact_keys(
    mapping: Mapping[str, Any],
    *,
    allowed: frozenset[str] | set[str] | tuple[str, ...],
    required: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
    context: str,
) -> None:
    """
    Validate *mapping* up front against an exact key set before any

    mutation runs.

    Raises :class:`ConfigError` naming every unknown key and every
        missing required key in one message so a malformed nested argument
        or JSON document fails whole, never partially applied.
    """
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{context}: expected an object/mapping, got {type(mapping).__name__}")
    allowed_set = frozenset(allowed)
    required_set = frozenset(required)
    keys = frozenset(str(k) for k in mapping.keys())
    unknown = sorted(keys - allowed_set)
    missing = sorted(required_set - keys)
    problems: list[str] = []
    if unknown:
        problems.append(f"unsupported keys {unknown!r}")
    if missing:
        problems.append(f"missing required keys {missing!r}")
    if problems:
        raise ConfigError(f"{context}: {'; '.join(problems)}")


def drain_diagnostic_collector() -> tuple[Diagnostic, ...]:
    """Extract and clear diagnostics from the active collector (used when building :class:`SessionStep`)."""
    buf = _DIAGNOSTIC_COLLECTOR.get()
    if not buf:
        return ()
    out = tuple(buf)
    buf.clear()
    return out


def stash_intent_parse_refusal(code: str, message: str) -> None:
    """Remember a crafted intent-parse refusal for the next generic parse-failed turn outcome."""
    _PENDING_INTENT_PARSE_REFUSAL.set((code, message))
    emit_session_refusal_diagnostic(code, message)


def take_intent_parse_refusal() -> tuple[str, str] | None:
    """Return and clear a stashed intent-parse refusal, if any."""
    pending = _PENDING_INTENT_PARSE_REFUSAL.get()
    if pending is None:
        return None
    _PENDING_INTENT_PARSE_REFUSAL.set(None)
    return pending


@contextmanager
def phase_timer(
    stage: str,
    *,
    source_id: str | None = None,
    code: str = DIAGNOSTIC_CODE_ENGINE_INFO,
    level: str = "info",
    phase: str | None = None,
) -> Iterator[None]:
    """Emit a timed diagnostic for *stage* when the block completes."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        details: tuple[tuple[str, str], ...] = (("phase", phase),) if phase else ()
        notify(
            stage,
            stage=stage,
            code=code,
            level=level,
            duration_ms=duration_ms,
            source_id=source_id,
            details=details,
        )


_LLM_USAGE_ACCUMULATOR: ContextVar[list[LlmUsageRecord] | None] = ContextVar(
    "aetherdialect_llm_usage_accumulator",
    default=None,
)
_LLM_USAGE_SCOPE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "aetherdialect_llm_usage_scope_stack",
    default=(),
)
_TURN_LLM_SCOPE: ContextVar[str | None] = ContextVar(
    "aetherdialect_turn_llm_scope",
    default=None,
)
_LLM_USAGE_BLOCK_ID: ContextVar[int] = ContextVar(
    "aetherdialect_llm_usage_block_id",
    default=0,
)
_LLM_PRICE_OVERRIDE: dict[str, dict[str, float]] | None = None
_LLM_USAGE_PHASE: ContextVar[str] = ContextVar("aetherdialect_llm_usage_phase", default="")
_LLM_USAGE_SOURCE_ID: ContextVar[str] = ContextVar("aetherdialect_llm_usage_source_id", default="")


def set_llm_price_table_override(table: dict[str, dict[str, float]] | None) -> None:
    """Replace the shipped OpenAI price table (for tests or operator overrides)."""
    global _LLM_PRICE_OVERRIDE
    _LLM_PRICE_OVERRIDE = dict(table) if table is not None else None


def llm_price_table_as_of() -> str:
    """Return the as-of date stamped on the active OpenAI price table."""
    return LLM_PRICE_TABLE_AS_OF


def _active_llm_price_table() -> dict[str, dict[str, float]]:
    return _LLM_PRICE_OVERRIDE if _LLM_PRICE_OVERRIDE is not None else LLM_PRICE_PER_MILLION


def _current_llm_usage_scope() -> Literal["build", "question", "run"]:
    """Return the scope for the next recorded usage. An explicit nested :func:`llm_usage_build_scope` / :func:`llm_usage_run_scope` / :func:`llm_usage_question_scope` context (innermost entry on the scope stack) always wins over an interactive turn scope set by :func:`set_turn_llm_scope`, so engine enrichment opened inside an open question turn still records against its own scope rather than swallowing into ``question``."""
    stack = _LLM_USAGE_SCOPE_STACK.get()
    if stack:
        return cast(Literal["build", "question", "run"], stack[-1])
    turn_scope = _TURN_LLM_SCOPE.get()
    if turn_scope in ("build", "question", "run"):
        return cast(Literal["build", "question", "run"], turn_scope)
    return "run"


@contextmanager
def llm_usage_session_scope() -> Iterator[None]:
    """Open a session-scoped LLM usage accumulator when one is not already active. Ask turns advance a cursor via :meth:`PipelineSession._emit_turn_llm_usage` and leave records in this buffer so hosts (live tests / sandbox recording) can :func:`snapshot_llm_usage_records` / flush an append-only invoice after each ask. Interactive hosts that want a clean buffer after summarizing a turn may still call :func:`drain_llm_usage_records` explicitly."""
    existing = _LLM_USAGE_ACCUMULATOR.get()
    if existing is not None:
        yield
        return
    buf: list[LlmUsageRecord] = []
    _LLM_USAGE_ACCUMULATOR.set(buf)
    yield


@contextmanager
def llm_usage_build_scope() -> Iterator[None]:
    """Attribute subsequent LLM usage records to a build phase."""
    stack = _LLM_USAGE_SCOPE_STACK.get()
    block_tok = _LLM_USAGE_BLOCK_ID.set(_LLM_USAGE_BLOCK_ID.get() + 1)
    scope_tok = _LLM_USAGE_SCOPE_STACK.set((*stack, "build"))
    try:
        yield
    finally:
        _LLM_USAGE_SCOPE_STACK.reset(scope_tok)
        _LLM_USAGE_BLOCK_ID.reset(block_tok)


@contextmanager
def llm_usage_run_scope() -> Iterator[None]:
    """Attribute subsequent LLM usage records to a run phase."""
    stack = _LLM_USAGE_SCOPE_STACK.get()
    block_tok = _LLM_USAGE_BLOCK_ID.set(_LLM_USAGE_BLOCK_ID.get() + 1)
    scope_tok = _LLM_USAGE_SCOPE_STACK.set((*stack, "run"))
    try:
        yield
    finally:
        _LLM_USAGE_SCOPE_STACK.reset(scope_tok)
        _LLM_USAGE_BLOCK_ID.reset(block_tok)


@contextmanager
def llm_usage_attribution(*, phase: str, source_id: str = "") -> Iterator[None]:
    """Bind federation phase and member source for nested LLM usage records."""
    phase_tok = _LLM_USAGE_PHASE.set(str(phase or ""))
    source_tok = _LLM_USAGE_SOURCE_ID.set(str(source_id or ""))
    try:
        yield
    finally:
        _LLM_USAGE_PHASE.reset(phase_tok)
        _LLM_USAGE_SOURCE_ID.reset(source_tok)


@contextmanager
def llm_usage_question_scope() -> Iterator[None]:
    """Attribute subsequent LLM usage records to a question phase."""
    stack = _LLM_USAGE_SCOPE_STACK.get()
    block_tok = _LLM_USAGE_BLOCK_ID.set(_LLM_USAGE_BLOCK_ID.get() + 1)
    scope_tok = _LLM_USAGE_SCOPE_STACK.set((*stack, "question"))
    try:
        yield
    finally:
        _LLM_USAGE_SCOPE_STACK.reset(scope_tok)
        _LLM_USAGE_BLOCK_ID.reset(block_tok)


def set_turn_llm_scope(scope: Literal["build", "question", "run"] | None) -> Token[str | None]:
    """Bind an interactive turn scope that overrides harness scope until reset."""
    if scope == "question":
        _LLM_USAGE_BLOCK_ID.set(_LLM_USAGE_BLOCK_ID.get() + 1)
    return _TURN_LLM_SCOPE.set(scope)


def reset_turn_llm_scope(token: Token[str | None]) -> None:
    """Restore the prior interactive turn scope. Hosts that resume a ``PipelineSession`` across thread-pool requests (for example FastAPI sync endpoints) may reset a token created in a different ``Context``. Clear the scope in that case instead of raising."""
    try:
        _TURN_LLM_SCOPE.reset(token)
    except ValueError:
        _TURN_LLM_SCOPE.set(None)


def snapshot_llm_usage_records() -> tuple[LlmUsageRecord, ...]:
    """Return a snapshot of records accumulated so far without clearing them."""
    buf = _LLM_USAGE_ACCUMULATOR.get()
    if not buf:
        return ()
    return tuple(buf)


def drain_llm_usage_records() -> tuple[LlmUsageRecord, ...]:
    """Extract and clear all accumulated LLM usage records."""
    buf = _LLM_USAGE_ACCUMULATOR.get()
    if not buf:
        return ()
    out = tuple(buf)
    buf.clear()
    return out


def scrub_schema_prose_for_prompt(text: str) -> str:
    """Delegate Ground scrubbing to SchemaGraph."""
    return SchemaGraph.scrub_schema_prose_for_prompt(text)


def reset_llm_usage_accumulator() -> None:
    """Clear the active LLM usage accumulator without returning records."""
    _LLM_USAGE_ACCUMULATOR.set(None)


def record_llm_usage(
    *,
    task: str,
    logical_model: str,
    api_model: str,
    provider: Literal["openai", "azure", "sandbox"],
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int | None,
    attempt: int,
    elapsed_ms: int,
) -> None:
    """Append one usage record when an accumulator is active; no-op otherwise."""
    buf = _LLM_USAGE_ACCUMULATOR.get()
    if buf is None:
        return
    buf.append(
        LlmUsageRecord(
            scope=_current_llm_usage_scope(),
            block_id=_LLM_USAGE_BLOCK_ID.get(),
            task=task,
            logical_model=logical_model,
            api_model=api_model,
            provider=provider,
            input_tokens=max(0, int(input_tokens)),
            cached_input_tokens=max(0, int(cached_input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            cache_write_tokens=None if cache_write_tokens is None else max(0, int(cache_write_tokens)),
            attempt=max(1, int(attempt)),
            elapsed_ms=max(0, int(elapsed_ms)),
            phase=_LLM_USAGE_PHASE.get(),
            source_id=_LLM_USAGE_SOURCE_ID.get(),
            turn_id=active_turn_id(),
        )
    )


def summarize_llm_usage_by_phase_and_source(
    records: tuple[LlmUsageRecord, ...] | list[LlmUsageRecord],
) -> tuple[tuple[str, str, str, int, int, int], ...]:
    """Aggregate token usage by scope, phase, and federation source."""
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for record in records:
        key = (record.scope, record.phase or record.task, record.source_id or "")
        slot = buckets.setdefault(key, [0, 0, 0])
        slot[0] += 1
        slot[1] += record.input_tokens
        slot[2] += record.output_tokens
    return tuple(
        (scope, phase, source_id, counts[0], counts[1], counts[2])
        for (scope, phase, source_id), counts in sorted(buckets.items())
    )


def emit_llm_usage_summary_diagnostics(
    records: tuple[LlmUsageRecord, ...] | list[LlmUsageRecord],
) -> None:
    """Emit one notify line per scope/phase bucket in *records*."""
    for scope, phase, source_id, requests, input_tokens, output_tokens in summarize_llm_usage_by_phase_and_source(
        records
    ):
        details: list[tuple[str, str]] = [
            ("requests", str(requests)),
            ("input_tokens", str(input_tokens)),
            ("output_tokens", str(output_tokens)),
        ]
        if source_id:
            details.append(("source_id", source_id))
        notify(
            f"LLM {scope}/{phase}: {requests} request(s), {input_tokens} input, {output_tokens} output tokens",
            stage="llm_usage_summary",
            code=DIAGNOSTIC_CODE_LLM_TURN_COST,
            level="info",
            details=tuple(details),
            source_id=source_id or None,
        )


def llm_call_cost_usd(record: LlmUsageRecord) -> float | None:
    """Return the OpenAI list-price cost for *record*, or ``None`` when unknown or not priced."""
    if record.provider != "openai":
        return None
    rates = _active_llm_price_table().get(record.logical_model)
    if rates is None:
        return None
    billable_input = max(0, record.input_tokens - record.cached_input_tokens)
    return (
        billable_input * rates["input"]
        + record.cached_input_tokens * rates["cached_input"]
        + record.output_tokens * rates["output"]
    ) / 1_000_000.0


def llm_call_audit_details(record: LlmUsageRecord) -> tuple[tuple[str, str], ...]:
    """Build audit detail pairs for one ``llm_call`` event."""
    pairs: list[tuple[str, str]] = [
        ("scope", record.scope),
        ("task", record.task),
        ("logical_model", record.logical_model),
        ("api_model", record.api_model),
        ("input_tokens", str(record.input_tokens)),
        ("cached_input_tokens", str(record.cached_input_tokens)),
        ("output_tokens", str(record.output_tokens)),
        ("attempt", str(record.attempt)),
        ("elapsed_ms", str(record.elapsed_ms)),
    ]
    if record.cache_write_tokens is not None:
        pairs.append(("cache_write_tokens", str(record.cache_write_tokens)))
    cost = llm_call_cost_usd(record)
    if cost is not None:
        pairs.append(("cost_usd", f"{cost:.6f}"))
    elif record.provider == "openai":
        pairs.append(("unpriced", record.logical_model))
    return tuple(pairs)


def llm_turn_cost_diagnostic(
    records: tuple[LlmUsageRecord, ...] | list[LlmUsageRecord],
    *,
    provider: Literal["openai", "azure", "sandbox"],
) -> Diagnostic | None:
    """Build a single turn-total cost diagnostic from *records*."""
    if not records:
        return None
    request_count = len(records)
    input_tokens = sum(r.input_tokens for r in records)
    cached_tokens = sum(r.cached_input_tokens for r in records)
    output_tokens = sum(r.output_tokens for r in records)
    details: list[tuple[str, str]] = [
        ("requests", str(request_count)),
        ("input_tokens", str(input_tokens)),
        ("cached_input_tokens", str(cached_tokens)),
        ("output_tokens", str(output_tokens)),
    ]
    message = (
        f"LLM turn: {request_count} request(s), "
        f"{input_tokens} input ({cached_tokens} cached), {output_tokens} output tokens"
    )
    if provider == "openai":
        costs = [c for r in records if (c := llm_call_cost_usd(r)) is not None]
        unpriced = sorted({r.logical_model for r in records if llm_call_cost_usd(r) is None})
        if costs:
            total_cost = sum(costs)
            message += f", ${total_cost:.6f}"
            details.append(("cost_usd", f"{total_cost:.6f}"))
            details.append(("price_table_as_of", llm_price_table_as_of()))
        if unpriced:
            details.append(("unpriced_models", ",".join(unpriced)))
    return Diagnostic(
        stage="llm",
        level=DiagnosticSeverity.INFO,
        code=DIAGNOSTIC_CODE_LLM_TURN_COST,
        message=message,
        details=tuple(details),
        phase="llm",
    )


def summarize_llm_turn_usage(
    records: tuple[LlmUsageRecord, ...] | list[LlmUsageRecord],
    *,
    provider: Literal["openai", "azure", "sandbox"] = "openai",
) -> LlmTurnUsageSummary | None:
    """Aggregate per-call records into one turn-level usage summary."""
    if not records:
        return None
    cost_usd: float | None = None
    if provider == "openai":
        costs = [c for r in records if (c := llm_call_cost_usd(r)) is not None]
        if costs:
            cost_usd = sum(costs)
    return LlmTurnUsageSummary(
        request_count=len(records),
        input_tokens=sum(r.input_tokens for r in records),
        cached_input_tokens=sum(r.cached_input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        cost_usd=cost_usd,
    )


def llm_turn_audit_details(
    records: tuple[LlmUsageRecord, ...] | list[LlmUsageRecord],
    *,
    provider: Literal["openai", "azure", "sandbox"],
) -> tuple[tuple[str, str], ...]:
    """Build audit detail pairs for one ``llm_turn`` summary event."""
    if not records:
        return (("requests", "0"),)
    request_count = len(records)
    input_tokens = sum(r.input_tokens for r in records)
    cached_tokens = sum(r.cached_input_tokens for r in records)
    output_tokens = sum(r.output_tokens for r in records)
    pairs: list[tuple[str, str]] = [
        ("requests", str(request_count)),
        ("input_tokens", str(input_tokens)),
        ("cached_input_tokens", str(cached_tokens)),
        ("output_tokens", str(output_tokens)),
    ]
    if provider == "openai":
        costs = [c for r in records if (c := llm_call_cost_usd(r)) is not None]
        unpriced = sorted({r.logical_model for r in records if llm_call_cost_usd(r) is None})
        if costs:
            pairs.append(("cost_usd", f"{sum(costs):.6f}"))
            pairs.append(("price_table_as_of", llm_price_table_as_of()))
        if unpriced:
            pairs.append(("unpriced_models", ",".join(unpriced)))
    return tuple(pairs)


def notify(
    message: str,
    *,
    stage: str | None = None,
    code: DiagnosticCode | str = DiagnosticCode.DIAGNOSTIC_CODE_ENGINE_INFO,
    level: DiagnosticSeverity | str = DiagnosticSeverity.INFO,
    duration_ms: int | None = None,
    details: tuple[tuple[str, str], ...] = (),
    source_id: str | None = None,
    phase: str | None = None,
    remediation: str | None = None,
    subject: str | None = None,
) -> None:
    """Append a diagnostic to the active collector and optionally mirror the line to a print listener."""
    eff_stage = stage or "notify"
    if isinstance(code, DiagnosticCode):
        code_str = code.value
    else:
        raw_code = str(code)
        try:
            code_str = DiagnosticCode(raw_code).value
        except ValueError:
            code_str = raw_code
    eff_phase = phase if phase is not None else eff_stage
    diag = Diagnostic(
        stage=eff_stage,
        level=level,
        code=code_str,
        message=message,
        details=details_with_turn_id(details),
        duration_ms=duration_ms,
        source_id=source_id,
        phase=eff_phase,
        remediation=remediation,
        subject=subject,
    )
    buf = _DIAGNOSTIC_COLLECTOR.get()
    if buf is not None:
        buf.append(diag)
    else:
        orphan_identity: EngineIdentity | None
        try:
            orphan_identity = active_engine_identity()
        except RuntimeError:
            orphan_identity = _PENDING_CONSTRUCTION_ENGINE_IDENTITY.get()
        if orphan_identity is not None:
            with _ORPHAN_DIAGNOSTICS_LOCK:
                _ORPHAN_DIAGNOSTICS.append((orphan_identity, diag))
    sink = _DIAGNOSTIC_SINK.get()
    if sink is not None:
        sink(diag)
    if prev_suppress:
        return
    fn = _DIAGNOSTIC_PRINT_LISTENER.get()
    if fn is not None:
        fn(message)
    if diagnostic_debug_enabled():
        rec: dict[str, Any] = {
            "kind": "notify",
            "stage": eff_stage,
            "code": code_str,
            "level": DiagnosticSeverity.coerce(diag.level).value,
            "message": message,
        }
        if duration_ms is not None:
            rec["duration_ms"] = duration_ms
        print(json.dumps(rec, ensure_ascii=False), file=sys.stderr, flush=True)


def failure_kind_is_permission_denied(error_kind: str | None, error_text: str | None = None) -> bool:
    """Return True when *error_kind* should terminate as a permission- denied access event."""
    if error_kind and error_kind in PERMISSION_DENIED_FAILURE_KINDS:
        return True
    if not error_kind or error_kind not in PERMISSION_DENIED_CATEGORY_ORACLE_KINDS:
        return False
    return bool(error_text) and error_text == REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE


def refusal_catalogue_entry(code: str) -> RefusalCatalogueEntry | None:
    """Return the catalogue entry for *code* when present."""
    raw = REFUSAL_CATALOGUE.get(code)
    if raw is None:
        return None
    return RefusalCatalogueEntry(user_text=raw["user_text"], reformulation_hint=raw["reformulation_hint"])


def refusal_user_text_for_code(code: str, **kwargs: str) -> str:
    """Return catalogue user text for *code*, formatting placeholders from *kwargs*."""
    entry = REFUSAL_CATALOGUE.get(code)
    if entry is None:
        return ""
    text = entry["user_text"]
    if kwargs:
        return text.format(**kwargs)
    return text


def refusal_reformulation_hint_for_code(code: str) -> str:
    """Return the reformulation hint for a catalogue refusal code."""
    entry = REFUSAL_CATALOGUE.get(code)
    if entry is None:
        return ""
    hint = entry["reformulation_hint"]
    if hint:
        return hint
    return entry["user_text"]


def refusal_diagnostic_code_for_rephrase_hint_key(key: str) -> str | None:
    """Map a rephrase-hint key to its refusal diagnostic code when catalogued."""
    return REPHRASE_HINT_REFUSAL_CODES.get(key)


def refusal_reformulation_hint_for_rephrase_hint_key(key: str) -> str:
    """Return the reformulation hint for a rephrase-hint key via the refusal catalogue."""
    code = refusal_diagnostic_code_for_rephrase_hint_key(key)
    if code:
        return refusal_reformulation_hint_for_code(code)
    return REPHRASE_HINT_MESSAGES.get(key, "")


def refusal_diagnostic_code_for_outcome(outcome: str) -> str | None:
    """Map an interactive terminal outcome to its refusal diagnostic code."""
    return OUTCOME_REFUSAL_CODES.get(outcome)


def is_aetherspace_scope_failure(
    *,
    failure_kind: str | None = None,
    error_message: str | None = None,
    refusal_diagnostic_code: str | None = None,
) -> bool:
    """Return True when a refusal was produced by an AetherSpace scope gate."""
    blob = (error_message or "").lower()
    if any(marker in blob for marker in AETHERSPACE_SCOPE_MARKERS):
        return True
    if refusal_diagnostic_code == DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION and "aetherspace" in blob:
        return True
    if failure_kind == FailureCategory.DENIED_REFERENCE.value and "aetherspace" in blob:
        return True
    return False


def is_execution_scope_failure(
    *,
    failure_kind: str | None = None,
    error_message: str | None = None,
) -> bool:
    """Return True when a refusal was produced by credential/EngineContext scope."""
    blob = (error_message or "").lower()
    if any(marker in blob for marker in EXECUTION_SCOPE_MARKERS):
        return True
    if failure_kind == FailureCategory.DENIED_REFERENCE.value and "aetherspace" not in blob:
        return True
    return False


def session_outcome_from_turn(
    raw_outcome: str,
    *,
    failure_kind: str | None = None,
    error_message: str | None = None,
    refusal_diagnostic_code: str | None = None,
    federation_limit_key: str | None = None,
) -> SessionOutcome:
    """Map an internal turn outcome onto the closed :class:`SessionOutcome` enum."""
    fk = str(failure_kind) if failure_kind else None
    err = str(error_message or "")
    lowered = err.lower()

    if federation_limit_key == "timeout_ms":
        return SessionOutcome.EXECUTION_TIMEOUT
    if federation_limit_key in {"explain_cost", "cost_cap", "explain_cost_cap"}:
        return SessionOutcome.COST_EXCEEDED
    if federation_limit_key:
        return SessionOutcome.LIMIT_EXCEEDED

    if is_aetherspace_scope_failure(
        failure_kind=fk,
        error_message=err,
        refusal_diagnostic_code=refusal_diagnostic_code,
    ):
        return SessionOutcome.UNANSWERABLE

    direct: dict[str, SessionOutcome] = {
        "restricted": SessionOutcome.UNSUPPORTED_OPERATION,
        "insufficient_knowledge": SessionOutcome.INSUFFICIENT_KNOWLEDGE,
        "user_declined": SessionOutcome.DECLINED,
        "intent_rejected": SessionOutcome.DECLINED,
        "schema_invalid_declined": SessionOutcome.DECLINED,
        "federation_turn_cancelled": SessionOutcome.CANCELLED,
        "parse_failed": SessionOutcome.PARSE_FAILED,
        "validation_failed": SessionOutcome.VALIDATION_FAILED,
        "conversational_deny": SessionOutcome.NOT_A_QUESTION,
        "invalid_question": SessionOutcome.NOT_A_QUESTION,
        "federation_partial_failure": SessionOutcome.EXECUTION_FAILED,
        "migration_pending": SessionOutcome.MIGRATION_PENDING,
        "permission_denied": SessionOutcome.FORBIDDEN,
    }
    if raw_outcome in direct and raw_outcome not in {"permission_denied", "validation_failed", "parse_failed"}:
        return direct[raw_outcome]

    if fk == FailureCategory.DENIED_REFERENCE.value:
        return SessionOutcome.FORBIDDEN
    if fk == FailureCategory.ACCESS_POLICY.value or failure_kind_is_permission_denied(fk, err):
        return SessionOutcome.FORBIDDEN
    if raw_outcome == "permission_denied":
        return SessionOutcome.FORBIDDEN
    if raw_outcome == "not_available_in_context":
        return SessionOutcome.INSUFFICIENT_KNOWLEDGE
    if terminal_refusal_is_scope_or_sensitivity(
        outcome=raw_outcome,
        failure_kind=fk,
        refusal_diagnostic_code=refusal_diagnostic_code,
    ):
        return SessionOutcome.FORBIDDEN
    if raw_outcome in direct:
        return direct[raw_outcome]
    if ("cost" in lowered or "explain_cost" in lowered) and ("exceed" in lowered or "cap" in lowered):
        return SessionOutcome.COST_EXCEEDED
    if "timeout" in lowered or "statement_timeout" in lowered:
        return SessionOutcome.EXECUTION_TIMEOUT
    if any(marker in lowered for marker in EXECUTION_SCOPE_MARKERS):
        return SessionOutcome.FORBIDDEN
    if any(marker in lowered for marker in ("permission denied", "access_policy", "denied_reference")):
        return SessionOutcome.FORBIDDEN
    if any(
        marker in lowered
        for marker in (
            "intent_parse_failed",
            "intent_schema_invalid",
            "schema_invalid",
            "could not compose intent",
            "intent_error",
        )
    ):
        return SessionOutcome.PARSE_FAILED
    return SessionOutcome.INTERNAL_ERROR


def session_error_from_turn_snap(snap: Mapping[str, Any]) -> SessionError | None:
    """Build a :class:`SessionError` from a stored interactive turn outcome."""
    raw_outcome = str(snap.get("outcome") or "success")
    if raw_outcome == "success":
        return None
    failure_kind = snap.get("failure_kind")
    error_message = snap.get("error")
    if error_message is not None and not isinstance(error_message, str):
        error_message = str(error_message)
    refusal_diagnostic_code = snap.get("refusal_diagnostic_code")
    detail_code = (
        str(refusal_diagnostic_code) if refusal_diagnostic_code else refusal_diagnostic_code_for_outcome(raw_outcome)
    )
    limit_key = str(snap.get("federation_limit_key") or "") or None
    return SessionError(
        code=session_outcome_from_turn(
            raw_outcome,
            failure_kind=str(failure_kind) if failure_kind else None,
            error_message=error_message,
            refusal_diagnostic_code=detail_code,
            federation_limit_key=limit_key,
        ),
        detail_code=detail_code,
        source_id=str(snap.get("federation_source_id") or "") or None,
        phase=str(snap.get("federation_phase") or "") or None,
        limit_key=limit_key,
    )


def session_error_from_terminal_message(
    message: str,
    *,
    federation_fields: Mapping[str, Any] | None = None,
    exc: BaseException | None = None,
) -> SessionError:
    """Build a :class:`SessionError` for a pipeline terminal error step."""
    fields = dict(federation_fields or ())
    limit_key = str(fields.get("federation_limit_key") or fields.get("limit_key") or "") or None
    detail_code = refusal_diagnostic_code_for_exception(exc) if exc is not None else None
    return SessionError(
        code=session_outcome_from_turn(
            "validation_failed",
            error_message=message,
            refusal_diagnostic_code=detail_code,
            federation_limit_key=limit_key,
        ),
        detail_code=detail_code,
        source_id=str(fields.get("federation_source_id") or "") or None,
        phase=str(fields.get("federation_phase") or "") or None,
        limit_key=limit_key,
    )


def refusal_diagnostic_code_for_exception(exc: BaseException) -> str | None:
    """Map a raised refusal exception to its stable diagnostic code."""
    if isinstance(exc, NoJoinPathError):
        return DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    if isinstance(exc, JoinPathTieCapExceededError):
        return DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP
    if isinstance(exc, AggregateJoinFanOutError):
        return DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT
    if isinstance(exc, ComparisonJoinScopeExceededError):
        return DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING
    if isinstance(exc, ClauseWidenedRowsetError):
        return DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET
    if isinstance(exc, ProbeCtePlacementError):
        return DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT
    if isinstance(exc, AccessError):
        return DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED
    if isinstance(exc, NullInNegatedListError):
        return DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST
    if isinstance(exc, SubdayDateWindowOnDateColumnError):
        return DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN
    if isinstance(exc, AmbiguousDateLiteralError):
        return DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL
    return None


def refusal_diagnostic_code_for_intent_issue(issue: IntentIssue) -> str | None:
    """Map a terminal intent issue to its stable refusal diagnostic code."""
    issue_id = str(issue.issue_id or "")
    if issue_id in REFUSAL_CTE_CAP_ISSUE_IDS:
        return DIAGNOSTIC_CODE_REFUSAL_CTE_CAP
    if issue_id in REFUSAL_NULL_IN_NEGATED_LIST_ISSUE_IDS:
        return DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST
    if issue_id in REFUSAL_UNSUPPORTED_COLUMN_TYPE_ISSUE_IDS:
        return DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE
    if issue_id == "comparison_join_hop_ceiling":
        return DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING
    if issue_id.startswith("clause_widened_rowset_"):
        return DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET
    if issue_id.startswith("probe_cte_"):
        return DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT
    if issue.category in (
        FailureCategory.DENY_BARE_SELECT,
        FailureCategory.DENIED_REFERENCE,
        FailureCategory.SENSITIVE_GROUP_BY,
    ):
        return DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
    if issue_id.startswith("sensitive_order_by_") or issue_id.startswith("sensitive_group_by_"):
        return DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
    if (
        issue.category.value in PERMISSION_DENIED_CATEGORY_ORACLE_KINDS
        and (issue.message or "") == REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE
    ):
        return DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT
    return None


def refusal_diagnostic_code_for_federation_reason(reason: str | None) -> str | None:
    """Map a federation ineligible reason to a capability-gap refusal code when applicable."""
    if not reason:
        return None
    lowered = reason.lower()
    if lowered.startswith("union logical column"):
        return DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING
    for prefix in REFUSAL_CAPABILITY_GAP_REASON_PREFIXES:
        if lowered.startswith(prefix):
            return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    if "is not supported by all federation members" in lowered:
        return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    for code in REFUSAL_CAPABILITY_GAP_REASON_CODES:
        if code in lowered:
            return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    return None


def terminal_refusal_is_scope_or_sensitivity(
    *,
    outcome: str = "",
    failure_kind: str | None = None,
    refusal_diagnostic_code: str | None = None,
) -> bool:
    """Return True when a terminal refusal should collapse to the uniform scope denial."""
    if outcome in ("permission_denied", "not_available_in_context"):
        return True
    if refusal_diagnostic_code in SCOPE_SENSITIVITY_REFUSAL_CODES:
        return True
    if failure_kind:
        if failure_kind in SCOPE_SENSITIVITY_FAILURE_KINDS:
            return True
        if failure_kind_is_permission_denied(failure_kind):
            return True
    return False


def refusal_terminal_cleared_egress_fields() -> dict[str, None]:
    """Session-step fields cleared on every catalogue refusal terminal before egress."""
    return {"intent_summary": None, "interpretation": None}


def catalogue_refusal_user_message(
    *,
    outcome: str,
    refusal_diagnostic_code: str | None,
    collapse_scope: bool,
    tables: str | None = None,
) -> str:
    """Return catalogue user text for a refusal terminal."""
    if collapse_scope:
        return PERMISSION_DENIED_USER_MESSAGE
    code = refusal_diagnostic_code or refusal_diagnostic_code_for_outcome(outcome)
    if not code:
        return ""
    if code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE and tables is not None:
        return refusal_user_text_for_code(code, tables=tables)
    return refusal_user_text_for_code(code)


def join_refusal_tables_placeholder(
    error: str | None,
    diagnostics: tuple[Diagnostic, ...] = (),
) -> str:
    """Best-effort table list for join-path catalogue formatting."""
    for diagnostic in diagnostics:
        detail_map = dict(diagnostic.details)
        tables = detail_map.get("tables")
        if tables:
            return str(tables)
    if error:
        quoted = re.findall(r"'([^']+)'", error)
        if quoted:
            return ", ".join(quoted)
        lowered = error.lower()
        if "could not be connected:" in lowered:
            tail = error.split(":", 1)[-1].strip().rstrip(".")
            if tail:
                return tail
    return "the requested tables"


def sanitize_refusal_remediation_for_egress(remediation: str | None) -> str | None:
    """Neutralize remediation text that discloses scope mechanism details."""
    if not remediation:
        return None
    lowered = remediation.lower()
    if any(marker in lowered for marker in REMEDIATION_SCOPE_MECHANISM_MARKERS):
        return REMEDIATION_RESTRICTED_QUESTION
    return remediation


def sanitize_refusal_diagnostic_for_egress(diagnostic: Diagnostic) -> Diagnostic:
    """Strip identifier-bearing fields from a refusal diagnostic before session egress."""
    from dataclasses import replace

    stripped_details = tuple(
        (key, value) for key, value in diagnostic.details if key not in EGRESS_STRIPPED_DETAIL_KEYS
    )
    message = diagnostic.message
    if terminal_refusal_is_scope_or_sensitivity(refusal_diagnostic_code=diagnostic.code):
        collapsed = catalogue_refusal_user_message(
            outcome="permission_denied",
            refusal_diagnostic_code=diagnostic.code,
            collapse_scope=True,
        )
        if collapsed:
            message = collapsed
    return replace(
        diagnostic,
        subject=None,
        details=stripped_details,
        remediation=sanitize_refusal_remediation_for_egress(diagnostic.remediation),
        message=message,
    )


def sanitize_refusal_diagnostics_for_egress(diagnostics: tuple[Diagnostic, ...]) -> tuple[Diagnostic, ...]:
    """Sanitize every diagnostic row attached to a refusal terminal step."""
    return tuple(sanitize_refusal_diagnostic_for_egress(diagnostic) for diagnostic in diagnostics)


def _opaque_member_handle(index: int) -> str:
    return f"member_{index}"


def _opaque_member_id_map(source_ids: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    counter = 0
    for raw in source_ids:
        sid = str(raw or "").strip()
        if not sid or sid in mapping:
            continue
        mapping[sid] = _opaque_member_handle(counter)
        counter += 1
    return mapping


def sanitize_federation_diagnostic_for_egress(diagnostic: Diagnostic) -> Diagnostic:
    """Strip member identity from federation diagnostics before session egress."""
    stripped_details = tuple(
        (key, value)
        for key, value in diagnostic.details
        if key not in EGRESS_STRIPPED_DETAIL_KEYS and key not in FEDERATION_EGRESS_STRIPPED_DETAIL_KEYS
    )
    source_id = diagnostic.source_id
    if source_id and source_id != "composite":
        source_id = "composite"
    message = diagnostic.message
    if "sources:" in message.lower() or "queried sources" in message.lower():
        message = "Federated turn queried one or more members."
    return replace(
        diagnostic,
        message=message,
        details=stripped_details,
        source_id=source_id,
        subject=None,
    )


def sanitize_federation_diagnostics_for_egress(diagnostics: tuple[Diagnostic, ...]) -> tuple[Diagnostic, ...]:
    """Sanitize every diagnostic row that may disclose federation member identity."""
    return tuple(sanitize_federation_diagnostic_for_egress(diagnostic) for diagnostic in diagnostics)


def sanitize_audit_details_for_egress(details: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Redact member identity keys from audit detail tuples for consumer callers."""
    stripped_keys = FEDERATION_EGRESS_STRIPPED_DETAIL_KEYS | frozenset({"source_id"})
    return tuple((key, value) for key, value in details if key not in stripped_keys)


def sanitize_federated_sql_for_egress(
    sql: str | dict[str, str] | None,
    *,
    source_id_map: Mapping[str, str] | None = None,
) -> str | dict[str, str] | None:
    """Replace federation member SQL map keys with opaque handles."""
    if not isinstance(sql, dict):
        return sql
    ordered_source_ids = [str(key) for key in sql if str(key).strip()]
    mapping = source_id_map or _opaque_member_id_map(ordered_source_ids)
    out: dict[str, str] = {}
    for sid, statement in sql.items():
        key = str(sid).strip()
        if not key:
            continue
        opaque = mapping.get(key, "composite")
        out[opaque] = str(statement)
    return out or None


def sanitize_federated_bundle_for_egress(
    bundle: Any | None,
    *,
    source_id_map: Mapping[str, str] | None = None,
) -> Any | None:
    """Replace member source ids in a federated execution bundle with opaque handles."""
    if bundle is None:
        return None
    if not isinstance(bundle, FederatedSqlBundle):
        return bundle
    ordered_source_ids = [
        str(getattr(rec, "source_id", "") or "")
        for rec in bundle.statements
        if str(getattr(rec, "phase", "member") or "member") == "member"
    ]
    mapping = source_id_map or _opaque_member_id_map(ordered_source_ids)
    statements: list[FederatedStatementRecord] = []
    for rec in bundle.statements:
        sid = str(rec.source_id or "").strip()
        opaque = mapping.get(sid, "composite") if sid else "composite"
        statements.append(replace(rec, source_id=opaque))
    read_window = tuple(
        (mapping.get(str(left or "").strip(), "composite"), str(right or "")) for left, right in bundle.read_window
    )
    return replace(bundle, statements=tuple(statements), read_window=read_window)


def sanitize_session_step_federation_fields_for_egress(
    *,
    federation_source_id: str | None,
    federation_succeeded: Sequence[tuple[str, int, str]],
    source_id_map: Mapping[str, str] | None = None,
) -> tuple[str | None, tuple[tuple[str, int, str], ...]]:
    """Return opaque federation attribution fields for consumer session egress."""
    ordered_source_ids = [str(federation_source_id or "").strip()] if federation_source_id else []
    ordered_source_ids.extend(str(row[0] or "") for row in federation_succeeded if row)
    mapping = source_id_map or _opaque_member_id_map(ordered_source_ids)
    opaque_succeeded = tuple(
        (mapping.get(str(row[0] or "").strip(), "composite"), int(row[1]), str(row[2] or ""))
        for row in federation_succeeded
        if row
    )
    opaque_source = None
    if federation_source_id:
        opaque_source = mapping.get(str(federation_source_id).strip(), "composite")
    return opaque_source, opaque_succeeded


def sanitize_session_step_for_egress(step: SessionStep) -> SessionStep:
    """Redact federation member identity from a :class:`SessionStep` before consumer egress."""
    ordered_source_ids: list[str] = []
    if step.error is not None and step.error.source_id:
        ordered_source_ids.append(str(step.error.source_id))
    if isinstance(step.sql, dict):
        ordered_source_ids.extend(str(key) for key in step.sql if str(key).strip())
    mapping = _opaque_member_id_map(ordered_source_ids)
    diagnostics = sanitize_federation_diagnostics_for_egress(step.diagnostics)
    error = step.error
    if error is not None and error.source_id:
        opaque_source = mapping.get(str(error.source_id).strip(), "composite")
        error = replace(error, source_id=opaque_source)
    return replace(
        step,
        sql=sanitize_federated_sql_for_egress(step.sql, source_id_map=mapping),
        error=error,
        diagnostics=diagnostics,
    )


def normalize_stored_interactive_turn_refusal(
    *,
    outcome: str,
    error: str | None,
    failure_kind: str | None = None,
    refusal_diagnostic_code: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Normalize stored interactive turn outcomes so history never retains raw scope refusals."""
    if terminal_refusal_is_scope_or_sensitivity(
        outcome=outcome,
        failure_kind=failure_kind,
        refusal_diagnostic_code=refusal_diagnostic_code,
    ):
        return "permission_denied", None, DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED
    if outcome in OUTCOME_REFUSAL_CODES or (outcome == "parse_failed" and refusal_diagnostic_code):
        code = refusal_diagnostic_code or refusal_diagnostic_code_for_outcome(outcome)
        if code:
            tables = (
                join_refusal_tables_placeholder(error)
                if code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
                else None
            )
            return (
                outcome,
                catalogue_refusal_user_message(
                    outcome=outcome,
                    refusal_diagnostic_code=code,
                    collapse_scope=False,
                    tables=tables,
                ),
                code,
            )
    return outcome, error, refusal_diagnostic_code


def emit_session_refusal_diagnostic(
    code: str,
    message: str,
    *,
    stage: str = "validation",
    source_id: str | None = None,
    details: tuple[tuple[str, str], ...] = (),
    subject: str | None = None,
) -> None:
    """Emit a structured refusal diagnostic for attachment to the active session step."""
    if code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_CTE_CAP:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code == DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR:
        notify(
            message,
            code=DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )
    elif code:
        notify(
            message,
            code=code,
            stage=stage,
            level="error",
            source_id=source_id,
            details=details,
            subject=subject,
        )


def refusal_message_for_exception(exc: BaseException) -> str:
    """Return the user-facing refusal text for *exc* when available."""
    code = refusal_diagnostic_code_for_exception(exc)
    if code:
        user_message = getattr(exc, "user_message", None)
        if isinstance(user_message, str) and user_message:
            return user_message
        catalogue_text = refusal_user_text_for_code(code)
        if catalogue_text:
            return catalogue_text
    user_message = getattr(exc, "user_message", None)
    if isinstance(user_message, str) and user_message:
        return user_message
    caller_message = getattr(exc, "message_for_caller", None)
    if isinstance(caller_message, str) and caller_message:
        return caller_message
    return str(exc)


_progress_depth = 0


def progress(message: str) -> None:
    """Emit a progress diagnostic when :func:`progress_enabled` is active and mirror it to a print listener."""
    if _progress_depth <= 0:
        return
    notify(message, stage="progress", code=DIAGNOSTIC_CODE_ENGINE_INFO)


@contextmanager
def progress_enabled() -> Iterator[None]:
    """Enable :func:`progress` writes for the duration of the block (supports nesting)."""
    global _progress_depth
    _progress_depth += 1
    try:
        yield
    finally:
        _progress_depth -= 1


def _running_in_jupyter() -> bool:
    """Return True when the current process runs inside a Jupyter or IPython kernel front-end."""
    return "ipykernel" in sys.modules


def echo_yes_no_answer(raw: str) -> None:
    """Echo ``Yes`` or ``No`` for *raw* input answer; emit nothing for invalid tokens or in a TTY terminal."""
    if not _running_in_jupyter():
        return
    token = raw.strip().lower()
    if token in {"y", "yes"}:
        print("Yes", flush=True)
    elif token in {"n", "no"}:
        print("No", flush=True)


def echo_user_text(raw: str) -> None:
    """Echo *raw* on its own line for Jupyter front-ends (terminal already shows what the user typed)."""
    if not _running_in_jupyter() or not raw:
        return
    print(raw, flush=True)


def _result(message: str) -> None:
    """Emit a query-_result line through :func:`notify` (mirrored to the diagnostic print listener when bound)."""
    notify(message, stage="user_result", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def error(message: str) -> None:
    """Emit an error line through :func:`notify`, prefixed with ``Error: ``."""
    notify(
        f"{USER_ERROR_PREFIX}{message}",
        stage="user_error",
        code=DiagnosticCode.DIAGNOSTIC_CODE_ENGINE_INFO,
        level=DiagnosticSeverity.ERROR,
    )


def prompt(message: str) -> str:
    """Display ``message``, read one line from stdin, and return it stripped."""
    return input(message).strip()


def terminated() -> None:
    """Emit the canonical user-termination line through :func:`notify`."""
    notify(
        USER_TERMINATED_LINE,
        stage="user_terminated",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def invalid_input(detail: str | None = None) -> None:
    """Emit the canonical invalid-input line through :func:`notify`, or *detail* when provided."""
    if detail:
        notify(
            detail.strip(),
            stage="user_invalid_input",
            code=DiagnosticCode.DIAGNOSTIC_CODE_ENGINE_INFO,
            level=DiagnosticSeverity.WARNING,
        )
    else:
        notify(
            USER_INVALID_INPUT_LINE,
            stage="user_invalid_input",
            code=DiagnosticCode.DIAGNOSTIC_CODE_ENGINE_INFO,
            level=DiagnosticSeverity.WARNING,
        )


def note_interactive_turn(
    choice_port: InteractiveChoicePort | None,
    *,
    outcome: str,
    error: str | None = None,
    sql: str | dict[str, str] | None = None,
    rows: list[tuple[Any, ...]] | None = None,
    columns: tuple[str, ...] | None = None,
    rejection_bucket: str | None = None,
    intent: RuntimeIntent | None = None,
    matched_template: Any | None = None,
    template_history_index: int | None = None,
    federated_bundle: Any | None = None,
    federated_plan: Any | None = None,
    generation_path: str | None = None,
    federation_source_id: str | None = None,
    federation_phase: str | None = None,
    federation_succeeded: Sequence[tuple[str, int, str]] | None = None,
    failure_kind: str | None = None,
    retryable: bool | None = None,
    refusal_diagnostic_code: str | None = None,
) -> None:
    """Record turn outcome on *choice_port* when it implements ``note_turn_outcome``."""
    if outcome == "parse_failed":
        generic_errors = {
            "Intent parse failed.",
            REPHRASE_HINT_MESSAGES["intent_parse_failed"],
            refusal_user_text_for_code(DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE),
        }
        if error is None or error in generic_errors:
            pending = take_intent_parse_refusal()
            if pending:
                refusal_diagnostic_code = pending[0]
                error = pending[1]
    outcome, error, refusal_diagnostic_code = normalize_stored_interactive_turn_refusal(
        outcome=outcome,
        error=error,
        failure_kind=failure_kind,
        refusal_diagnostic_code=refusal_diagnostic_code,
    )
    fn = getattr(choice_port, "note_turn_outcome", None)
    if callable(fn):
        fn(
            outcome=outcome,
            error=error,
            sql=sql,
            rows=rows,
            columns=columns,
            rejection_bucket=rejection_bucket,
            intent=intent,
            matched_template=matched_template,
            template_history_index=template_history_index,
            federated_bundle=federated_bundle,
            federated_plan=federated_plan,
            generation_path=generation_path,
            federation_source_id=federation_source_id,
            federation_phase=federation_phase,
            federation_succeeded=federation_succeeded,
            failure_kind=failure_kind,
            retryable=retryable,
            refusal_diagnostic_code=refusal_diagnostic_code,
        )


prev_sink: list[str] | None = None
prev_suppress: bool = False


@contextmanager
def telemetry_capture(
    *,
    suppress_console: bool = False,
    force_diagnostic_flags: bool = False,
) -> Iterator[list[str]]:
    """Collect ``debug`` / ``pipeline_trace`` lines into a buffer."""
    global prev_sink, prev_suppress
    buf: list[str] = []
    saved_sink = prev_sink
    saved_suppress = prev_suppress
    if force_diagnostic_flags:
        diagnostic_force_enter()
    prev_sink = buf
    prev_suppress = suppress_console
    try:
        yield buf
    finally:
        prev_sink = saved_sink
        prev_suppress = saved_suppress
        if force_diagnostic_flags:
            diagnostic_force_exit()


def debug(msg: str) -> None:
    """Print ``[DEBUG]`` + *msg* when ``PolicyConfig.DEBUG`` or debug. diagnostics are on (see ``diagnostic_debug_enabled``)."""
    line = f"[DEBUG] {msg}"
    if prev_sink is not None:
        prev_sink.append(line)
    if prev_suppress:
        return
    if diagnostic_debug_enabled():
        print(line)
        print(
            json.dumps({"kind": "debug", "message": msg}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )


def _format_pipeline_trace_block(heading: str, body: str) -> str:
    """Format a pipeline-trace block without performing I/O."""
    return f"[PIPELINE_TRACE] {heading}\n{body}"


def pipeline_trace(heading: str, body: str | Callable[[], str]) -> None:
    """Emit a ``[PIPELINE_TRACE]`` block when debug or diagnostic capture is active. *heading* is a dotted trace vocabulary label for the pipeline stage being traced; it is **not** a diagnostic code and is excluded from the diagnostic catalogue."""
    structured_sink = _PIPELINE_TRACE_SINK.get()
    telemetry_on = prev_sink is not None
    console_on = not prev_suppress and diagnostic_debug_enabled()
    if structured_sink is None and not telemetry_on and not console_on:
        return
    resolved = body() if callable(body) else body
    if structured_sink is not None:
        structured_sink(heading, resolved)
    block = _format_pipeline_trace_block(heading, resolved)
    if prev_sink is not None:
        prev_sink.append(block)
    if prev_suppress:
        return
    if not diagnostic_debug_enabled():
        return
    print(block)


@contextmanager
def pipeline_trace_sink(callback: Callable[[str, str], None]) -> Iterator[None]:
    """Bind *callback* to receive structured ``(heading, body)`` pipeline trace events."""
    token = _PIPELINE_TRACE_SINK.set(callback)
    try:
        yield
    finally:
        _PIPELINE_TRACE_SINK.reset(token)


def sha256(s: str) -> str:
    """SHA-256 hex digest of UTF-8 *s*."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_bucket(key: str, n: int) -> int:
    """Return a deterministic bucket index in ``[0, n)`` for *key*."""
    if n <= 0:
        raise ValueError("stable_bucket requires a positive bucket count")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _strip_fences(s: str) -> str:
    """Strip leading/trailing ``` fences and surrounding whitespace."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def canonicalize_sql(sql: str) -> str:
    """Normalize SQL whitespace, formatting, and join equality operand. order."""
    s = _strip_fences(sql).strip()
    s = s.rstrip(";").strip()
    s = re.sub(r"^EXPLAIN\s+(?:ANALYZE\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(?<![><!=])=(?![>=])", " = ", s)
    s = re.sub(r"\s+", " ", s).strip()

    def normalize_equality(m: re.Match[str]) -> str:
        left, right = m.group(1).strip(), m.group(2).strip()
        if left > right:
            left, right = right, left
        return f"{left} = {right}"

    s = re.sub(r"([^\s()><!=]+)\s*=\s*([^\s()><!=]+)", normalize_equality, s)
    return s


def stable_json(o: Any) -> str:
    """``json.dumps`` with sorted keys and minimal separators."""
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def phys_table_key(tbl: str) -> str:
    return (tbl or "").strip().lower()


def description_neutrality_violations(text: str, forbidden_tokens: frozenset[str]) -> list[str]:
    """Return forbidden tokens present in *text* using identifier word boundaries."""
    cleaned = str(text or "").strip()
    if not cleaned or not forbidden_tokens:
        return []
    hits: list[str] = []
    for token in sorted(forbidden_tokens, key=len, reverse=True):
        if not token:
            continue
        if re.search(rf"\b{re.escape(token)}\b", cleaned, flags=re.IGNORECASE):
            hits.append(token)
    return hits


def out_of_scope_description_tokens(full_graph: SchemaGraph, scope: KnowledgeScope) -> frozenset[str]:
    """Identifier vocabulary present on *full_graph* but absent from *scope*."""
    in_scope_column_names = {qc.split(".", 1)[1].strip().lower() for qc in scope.columns if "." in qc}
    tokens: set[str] = set()
    for table_name, table in full_graph.tables.items():
        if not scope.contains(table_name):
            tokens.add(table_name)
            original = (table.original_name or "").strip()
            if original:
                tokens.add(original)
        for column_name, column in table.columns.items():
            if scope.contains(f"{table_name}.{column_name}"):
                continue
            if column_name.strip().lower() in in_scope_column_names:
                continue
            tokens.add(column_name)
            col_original = (column.original_name or "").strip()
            if col_original and col_original.strip().lower() not in in_scope_column_names:
                tokens.add(col_original)
    return frozenset(tokens)


def clear_descriptions_naming_out_of_scope_entities(graph: SchemaGraph, tokens: frozenset[str]) -> int:
    """Blank descriptions on *graph* that name a token from *tokens*."""
    if not tokens:
        return 0
    cleared = 0
    for table in graph.tables.values():
        if table.description and description_neutrality_violations(table.description, tokens):
            table.description = ""
            cleared += 1
        for column in table.columns.values():
            if column.description and description_neutrality_violations(column.description, tokens):
                column.description = ""
                cleared += 1
    return cleared


def column_overlap_comparison_mode(left: ColumnMetadata, right: ColumnMetadata) -> OverlapComparison:
    """Return the overlap comparison rule when pairing two profiled columns."""
    if left.is_case_insensitive_collation or right.is_case_insensitive_collation:
        return OverlapComparison.CASE_FOLDED
    return OverlapComparison.EXACT


def _normalize_overlap_sample_value(value: object, *, case_fold: bool, rtrim_pad: bool = False) -> str | None:
    if value is None:
        return None
    s = normalize_text_value(str(value))
    if rtrim_pad:
        s = s.rstrip()
    else:
        s = s.strip()
    return s.casefold() if case_fold else s


def normalized_value_overlap_sets(
    left: ColumnMetadata,
    right: ColumnMetadata,
    *,
    record_comparison: bool = True,
) -> tuple[set[str], set[str], OverlapComparison]:
    """Normalize overlap samples for a pair, optionally recording the comparison rule used."""
    mode = column_overlap_comparison_mode(left, right)
    fold = mode == "case_folded"
    rtrim_pad = left.is_fixed_width_text or right.is_fixed_width_text
    left_set = {
        normalized
        for v in left.value_overlap_sample or []
        if (normalized := _normalize_overlap_sample_value(v, case_fold=fold, rtrim_pad=rtrim_pad)) is not None
    }
    right_set = {
        normalized
        for v in right.value_overlap_sample or []
        if (normalized := _normalize_overlap_sample_value(v, case_fold=fold, rtrim_pad=rtrim_pad)) is not None
    }
    if record_comparison:
        left.overlap_comparison = mode
        right.overlap_comparison = mode
    return left_set, right_set, mode


def _col_value_overlap_frozen(col: ColumnMetadata) -> frozenset[str]:
    vals = col.value_overlap_sample or []
    cleaned = {normalize_text_value(str(v).strip()) for v in vals if v is not None}
    cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
    return frozenset(sorted(cleaned)[:cap])


def profiling_value_overlap(older: SchemaGraph, newer: SchemaGraph) -> float:
    """Aggregate Jaccard overlap of value-overlap samples on shared ``(table, column)`` keys."""
    inter = 0
    union = 0
    for t in older.tables:
        if t not in newer.tables:
            continue
        ot = older.tables[t]
        nt = newer.tables[t]
        for c in ot.columns:
            if c not in nt.columns:
                continue
            a = _col_value_overlap_frozen(ot.columns[c])
            b = _col_value_overlap_frozen(nt.columns[c])
            u = len(a | b)
            if u == 0:
                continue
            union += u
            inter += len(a & b)
    if union == 0:
        return 1.0
    return inter / union


def _knowledge_fingerprint_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def knowledge_scope_fingerprint(schema_graph: SchemaGraph, scope: KnowledgeScope | None = None) -> str:
    """Stable digest over schema state and in-scope entity set."""
    resolved_scope = scope if scope is not None else KnowledgeScope.from_schema_graph(schema_graph)
    eff = _knowledge_fingerprint_str(getattr(schema_graph, "effective_structural_hash", None))
    struct = _knowledge_fingerprint_str(getattr(schema_graph, "structural_hash", None))
    prof = _knowledge_fingerprint_str(getattr(schema_graph, "profiling_hash", None))
    payload = {
        "effective_structural_hash": eff or struct,
        "profiling_hash": prof,
        "tables": sorted(t for t in resolved_scope.tables if isinstance(t, str)),
        "columns": sorted(c for c in resolved_scope.columns if isinstance(c, str)),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def knowledge_artifact_stamp_matches(
    raw: dict[str, Any],
    *,
    notes_sha256: str,
    scope_fingerprint: str,
) -> bool:
    """True when persisted artifact keys match the live derivation inputs."""
    want_notes = str(notes_sha256 or "").strip()
    have_notes = str(raw.get("notes_sha256") or "").strip()
    want_scope = str(scope_fingerprint or "").strip()
    have_scope = str(raw.get("scope_fingerprint") or "").strip()
    if want_notes and have_notes != want_notes:
        return False
    if want_scope and have_scope != want_scope:
        return False
    return True


def _entity_resolves_on_schema(entity: str, schema_graph: SchemaGraph) -> bool:
    text = str(entity).strip()
    if not text:
        return False
    if "." in text:
        table_name, column_name = text.split(".", 1)
        table = schema_graph.tables.get(table_name)
        return table is not None and column_name in table.columns
    return text in schema_graph.tables


def filter_domain_knowledge_by_resolvable_references(
    entries: Sequence[DomainKnowledgeEntry],
    schema_graph: SchemaGraph,
) -> tuple[tuple[DomainKnowledgeEntry, ...], int]:
    """Drop entries whose reference set no longer resolves; return kept rows and drop count."""
    kept: list[DomainKnowledgeEntry] = []
    dropped = 0
    for entry in entries:
        refs = entry.referenced_entities
        if refs and not all(_entity_resolves_on_schema(ref, schema_graph) for ref in refs):
            dropped += 1
            continue
        kept.append(entry)
    return tuple(kept), dropped


def prompt_json(body: Mapping[str, Any], key_order: tuple[str, ...]) -> str:
    """Serialize an outbound LLM user payload with explicit top-level key order. Static prompt sections (schemas, rules, output shapes) should precede per-question tails so provider prefix caching can reuse bytes across a run. Nested dict key order is preserved as built. Hashes and mock-fixture lookup continue to use :func:`stable_json`."""
    ordered: dict[str, Any] = {}
    seen: set[str] = set()
    for key in key_order:
        if key in body:
            ordered[key] = body[key]
            seen.add(key)
    for key, value in body.items():
        if key not in seen:
            ordered[key] = value
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


_PROMPT_CACHE_SCHEMA_HASH: ContextVar[str | None] = ContextVar(
    "aetherdialect_prompt_cache_schema_hash",
    default=None,
)


def active_prompt_cache_schema_hash() -> str | None:
    """Return the schema hash bound for :func:`resolve_prompt_cache_key`, if any."""
    return _PROMPT_CACHE_SCHEMA_HASH.get()


@contextmanager
def prompt_cache_schema_scope(schema_hash: str | None) -> Iterator[None]:
    """Bind *schema_hash* for nested LLM calls that should share a ``prompt_cache_key``."""
    cleaned = str(schema_hash or "").strip() or None
    tok = _PROMPT_CACHE_SCHEMA_HASH.set(cleaned)
    try:
        yield
    finally:
        _PROMPT_CACHE_SCHEMA_HASH.reset(tok)


_ACTIVE_DOMAIN_KNOWLEDGE: ContextVar[tuple[DomainKnowledgeEntry, ...] | None] = ContextVar(
    "aetherdialect_active_domain_knowledge",
    default=None,
)
_ACTIVE_DOMAIN_KNOWLEDGE_DIGEST: ContextVar[str | None] = ContextVar(
    "aetherdialect_active_domain_knowledge_digest",
    default=None,
)


def empty_domain_knowledge_digest() -> str:
    """Return the digest for an empty knowledge set."""
    return DomainKnowledgeState.empty_digest()


def domain_knowledge_digest(entries: Sequence[DomainKnowledgeEntry]) -> str:
    """Stable SHA-256 digest over normalized domain knowledge entries."""
    return DomainKnowledgeState.digest_for(entries)


def _normalize_domain_knowledge_entry(entry: DomainKnowledgeEntry) -> DomainKnowledgeEntry:
    return DomainKnowledgeEntry.normalize(entry)


def hidden_column_references_in_text(text: str, schema_graph: SchemaGraph) -> list[str]:
    """Return qualified hidden column names referenced in *text*."""
    return DomainKnowledgeEntry.hidden_column_references(text, schema_graph)


def validate_domain_knowledge_entries(
    entries: Sequence[DomainKnowledgeEntry],
    schema_graph: SchemaGraph,
) -> tuple[DomainKnowledgeEntry, ...]:
    """Normalize entries and refuse hidden-column references."""
    return DomainKnowledgeState.validate_entries(entries, schema_graph)


def domain_context_payload(entries: Sequence[DomainKnowledgeEntry]) -> list[dict[str, str]] | None:
    """Serialize active domain knowledge for intent prompt injection."""
    if not entries:
        return None
    return [
        {"key": entry.key, "kind": entry.kind, "text": scrub_schema_prose_for_prompt(entry.text)}
        for entry in sorted(entries, key=lambda entry: entry.key)
    ]


def domain_knowledge_artifact_path(artifacts_dir: str | os.PathLike[str]) -> Path:
    """Return ``{artifacts_dir}/domain_knowledge.json`` for master/engine DK persistence."""
    return Path(artifacts_dir) / DOMAIN_KNOWLEDGE_FILENAME


def structural_knowledge_artifact_path(artifacts_dir: str | os.PathLike[str]) -> Path:
    """Return ``{artifacts_dir}/structural_knowledge.json`` for persisted structural facts."""
    return Path(artifacts_dir) / STRUCTURAL_KNOWLEDGE_FILENAME


def delete_domain_knowledge_artifact(artifacts_dir: str | os.PathLike[str]) -> bool:
    """Remove persisted domain knowledge when notes are cleared. Returns True when a file was removed."""
    path = domain_knowledge_artifact_path(artifacts_dir)
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        return False
    return False


def load_domain_knowledge_artifact(
    artifacts_dir: str | os.PathLike[str],
    schema_graph: SchemaGraph,
    *,
    require_notes_match: bool = True,
) -> tuple[DomainKnowledgeEntry, ...] | None:
    """Load master domain knowledge from the artifacts dir, or ``None`` when absent/unusable. When *require_notes_match* is True (default), the artifact ``notes_sha256`` and ``scope_fingerprint`` must match the live graph so notes or schema edits invalidate the artifact and force a fresh extract. Migration remap/prune passes ``require_notes_match=False``."""
    path = domain_knowledge_artifact_path(artifacts_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    require_exact_keys(
        raw,
        allowed=frozenset(
            {
                "domain_knowledge",
                "domain_knowledge_digest",
                "notes_sha256",
                "scope_fingerprint",
                "format_version",
            }
        ),
        required=frozenset({"domain_knowledge"}),
        context="domain knowledge artifact",
    )
    if require_notes_match:
        want_notes = str(getattr(schema_graph, "notes_sha256", "") or "").strip()
        scope_fp = knowledge_scope_fingerprint(schema_graph)
        if not knowledge_artifact_stamp_matches(
            raw,
            notes_sha256=want_notes,
            scope_fingerprint=scope_fp,
        ):
            return None
    items = raw.get("domain_knowledge")
    if not isinstance(items, list):
        return None
    allowed_kinds = {member.value for member in DomainKnowledgeKind}
    out: list[DomainKnowledgeEntry] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        text = str(item.get("text") or "").strip()
        kind = str(item.get("kind") or DomainKnowledgeKind.GLOSSARY.value).strip() or DomainKnowledgeKind.GLOSSARY.value
        if not key or not text or key in seen:
            continue
        if kind not in allowed_kinds:
            kind = DomainKnowledgeKind.GLOSSARY.value
        if "referenced_entities" not in item:
            continue
        raw_referenced = item.get("referenced_entities")
        if not isinstance(raw_referenced, list) or not all(isinstance(r, str) for r in raw_referenced):
            continue
        referenced_entities = frozenset(str(r).strip() for r in raw_referenced if str(r).strip())
        try:
            entry = DomainKnowledgeEntry.normalize(
                DomainKnowledgeEntry(key=key, text=text, kind=kind, referenced_entities=referenced_entities)
            )
        except ConfigError:
            continue
        seen.add(entry.key)
        out.append(entry)
    if not out:
        return ()
    kept, dropped = filter_domain_knowledge_by_resolvable_references(out, schema_graph)
    if dropped:
        debug(f"[load_domain_knowledge_artifact] dropped {dropped} unresolvable domain knowledge entries")
    if not kept:
        return ()
    try:
        return validate_domain_knowledge_entries(kept, schema_graph)
    except ConfigError:
        return None


def active_domain_knowledge() -> tuple[DomainKnowledgeEntry, ...]:
    """Return domain knowledge entries bound in the current scope."""
    active = _ACTIVE_DOMAIN_KNOWLEDGE.get()
    return active if active is not None else ()


def active_domain_knowledge_digest() -> str | None:
    """Return the digest bound in the current scope, if any."""
    digest = _ACTIVE_DOMAIN_KNOWLEDGE_DIGEST.get()
    cleaned = str(digest or "").strip()
    return cleaned or None


@contextmanager
def domain_knowledge_scope(
    entries: Sequence[DomainKnowledgeEntry],
    digest: str | None,
) -> Iterator[None]:
    """Bind domain knowledge for nested intent parsing and prompt- cache routing."""
    normalized = tuple(entries)
    cleaned_digest = str(digest or "").strip() or None
    tok_entries: Token[tuple[DomainKnowledgeEntry, ...] | None] = _ACTIVE_DOMAIN_KNOWLEDGE.set(normalized)
    tok_digest: Token[str | None] = _ACTIVE_DOMAIN_KNOWLEDGE_DIGEST.set(cleaned_digest)
    try:
        yield
    finally:
        _ACTIVE_DOMAIN_KNOWLEDGE.reset(tok_entries)
        _ACTIVE_DOMAIN_KNOWLEDGE_DIGEST.reset(tok_digest)


def _tables_descriptions_payload_for_cache(tables: Any) -> dict[str, Any]:
    """Build sorted table/column description dict for prompt-cache hashing."""
    out: dict[str, Any] = {}
    for tname in sorted(tables):
        tbl = tables[tname]
        cols_payload: dict[str, str] = {}
        for cname in sorted(tbl.columns):
            desc = (tbl.columns[cname].description or "").strip()
            if desc:
                cols_payload[cname] = desc
        table_payload: dict[str, Any] = {}
        table_desc = (tbl.description or "").strip()
        if table_desc:
            table_payload["description"] = table_desc
        if cols_payload:
            table_payload["columns"] = cols_payload
        if table_payload:
            out[tname] = table_payload
    return out


def schema_prompt_cache_id(schema_graph: Any) -> str | None:
    """Return a stable schema identifier for prompt-cache routing."""
    graph_id = str(getattr(schema_graph, "schema_graph_id", "") or "").strip()
    if not graph_id:
        structural = str(getattr(schema_graph, "effective_structural_hash", "") or "").strip()
        if not structural:
            return None
        graph_id = structural
    tables = getattr(schema_graph, "tables", None)
    segments: list[str] = [graph_id]
    desc_hash = str(getattr(schema_graph, "descriptions_hash", "") or "").strip()
    if not desc_hash and tables is not None:
        desc_hash = descriptions_hash_fp(_tables_descriptions_payload_for_cache(tables))
    if desc_hash:
        segments.append(desc_hash[:16])
    prof_hash = str(getattr(schema_graph, "profiling_hash", "") or "").strip()
    if prof_hash:
        segments.append(prof_hash[:16])
    if tables is not None:
        meta_hash = metadata_hash_fp(_tables_metadata_payload_for_cache(tables))
        if meta_hash:
            segments.append(meta_hash[:16])
    return ":".join(segments) if graph_id else None


def build_case_folded_index(names: Iterable[str], *, kind: str) -> dict[str, str]:
    """Map case-folded identifiers to canonical spellings, refusing case-only duplicates."""
    groups: dict[str, list[str]] = {}
    for name in names:
        folded = name.lower()
        groups.setdefault(folded, []).append(name)
    index: dict[str, str] = {}
    for folded, members in groups.items():
        distinct = sorted({member for member in members}, key=lambda value: (value.lower(), value))
        if len(distinct) > 1:
            raise SchemaInvariantError(
                f"case-only {kind} collision: {distinct[0]!r} and {distinct[1]!r} cannot be distinguished by the engine"
            )
        index[folded] = distinct[0]
    return index


def colmap_signature(column_map: dict[str, str]) -> str:
    """SHA-256 of stable JSON for sorted ``column_map`` items."""
    return sha256(stable_json(sorted(column_map.items())))


def intent_id(d: dict[str, Any]) -> str:
    """Short intent id: first 16 hex chars of ``stable_json(d)`` hash."""
    return sha256(stable_json(d))[:16]


def structural_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint DDL-stable table payloads (kinds, columns, keys, FK. edges)."""
    return sha256(stable_json({"tables": tables_payload}))


def profiling_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint profiling-only payloads (counts, roles, top values, semantics)."""
    return sha256(stable_json({"tables": tables_payload}))


def descriptions_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint table and column description text across a schema graph."""
    return sha256(stable_json({"tables": tables_payload}))


def _tables_metadata_payload_for_cache(tables: Any) -> dict[str, Any]:
    """Build sorted metadata dict for prompt-cache hashing."""
    out: dict[str, Any] = {}
    for tname in sorted(tables):
        tbl = tables[tname]
        cols_payload: dict[str, Any] = {}
        for cname in sorted(tbl.columns):
            col = tbl.columns[cname]
            cols_payload[cname] = {
                "description": col.description or "",
                "description_owner": (col.description_owner.value if col.description_owner is not None else None),
                "role": col.role,
                "role_owner": (col.role_owner.value if col.role_owner is not None else None),
                "sensitivity": col.sensitivity,
            }
        out[tname] = {
            "description": tbl.description or "",
            "description_owner": (tbl.description_owner.value if tbl.description_owner is not None else None),
            "role": tbl.role,
            "role_owner": tbl.role_owner.value if tbl.role_owner is not None else None,
            "columns": cols_payload,
        }
    return out


def metadata_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint descriptions, roles, and sensitivities across a schema graph."""
    return sha256(stable_json({"tables": tables_payload}))


def _schema_scope_file_content_sha256(path: str | None) -> str:
    """Return a SHA-256 hex digest of the UTF-8 file at *path*, or ``""`` when the path is missing or not a readable file."""
    if path is None or not str(path).strip():
        return ""
    expanded = os.path.expanduser(str(path).strip())
    if not os.path.isfile(expanded):
        return ""
    with open(expanded, encoding="utf-8") as fh:
        return sha256(fh.read())


def notes_content_from_context(schema_context: EngineContext | FederationContext | Any) -> str | None:
    """Return notes text from inline ``notes`` or ``notes_file`` contents; ``None`` when neither yields text."""
    notes_inline = getattr(schema_context, "notes", None)
    if notes_inline is not None and str(notes_inline).strip():
        return str(notes_inline)
    notes_path = getattr(schema_context, "notes_file", None)
    if notes_path is None or not str(notes_path).strip():
        return None
    expanded = os.path.expanduser(str(notes_path).strip())
    if not os.path.isfile(expanded):
        return None
    with open(expanded, encoding="utf-8") as fh:
        text = fh.read()
    return text if text.strip() else None


def scope_hash_fp(schema_context: EngineContext | FederationContext) -> str:
    """Fingerprint allow/deny scope and optional DDL file contents. Notes content is intentionally excluded: notes changes are tracked via ``notes_hash`` / soft-refresh paths, not as a scope identity change. ``FederationContext`` has no ``sql_file`` slot, so that field reads as an empty string for federation scopes while the payload key stays present."""
    deny_cols = sorted(schema_context.deny_columns)
    allow_cols = sorted(schema_context.allow_columns)
    sql_file = getattr(schema_context, "sql_file", "")
    if schema_context.allow_objects:
        reflect_mode = ReflectMode.ALLOW_LIST.value
    elif schema_context.deny_objects:
        reflect_mode = ReflectMode.BOTH_THEN_DENY.value
    else:
        reflect_mode = ReflectMode.SINGLE_KIND.value
    payload = {
        "allow_objects": sorted(schema_context.allow_objects),
        "deny_objects": sorted(schema_context.deny_objects),
        "deny_columns": deny_cols,
        "allow_columns": allow_cols,
        "include": schema_context.include,
        "reflect_mode": reflect_mode,
        "sql_file_content_sha256": _schema_scope_file_content_sha256(sql_file),
    }
    return sha256(stable_json(payload))


def visibility_table_set_fingerprint(visible_tables: frozenset[str] | set[str] | Sequence[str]) -> str:
    """Stable short fingerprint of a sorted credential-visible table set."""
    tables = sorted({str(t).strip() for t in visible_tables if str(t).strip()})
    return hashlib.sha256(",".join(tables).encode("utf-8")).hexdigest()[:16]


def _scope_ctx_restricts_visibility(scope_ctx: EngineContext | FederationContext | None) -> bool:
    if scope_ctx is None:
        return False
    if scope_ctx.allow_objects or scope_ctx.deny_objects or scope_ctx.deny_columns or scope_ctx.allow_columns:
        return True
    sql_file = getattr(scope_ctx, "sql_file", "")
    return bool(str(sql_file or "").strip())


def get_aggregatable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    """Return column keys that can be aggregated with SUM, AVG, MIN, or MAX."""
    result: list[str] = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        if role == ColumnRole.NUMERIC_MEASURE.value:
            result.append(col_key)

    return result


def get_groupable_columns(table: str, schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    """Return column keys usable in GROUP BY clauses."""
    result: list[str] = []
    table_ir = schema.tables.get(table)
    if not table_ir:
        return result

    for col_name, col_meta in table_ir.columns.items():
        col_key = f"{table}.{col_name}"
        role = column_roles.get(col_key, col_meta.role or "unknown")
        if role in (ColumnRole.CATEGORICAL.value, ColumnRole.TEMPORAL.value, ColumnRole.NUMERIC_CATEGORICAL.value):
            result.append(col_key)

    return result


def simulation_artifact_partition_fp(
    *,
    space_uid: str = "",
    scope_ctx: EngineContext | FederationContext | None = None,
    visible_objects: frozenset[str] | None = None,
    space_tables: set[str] | None = None,
) -> str:
    """Return a short partition fingerprint for warmup/QSim caches; empty means owner default scope."""
    parts: list[str] = []
    uid = str(space_uid or "").strip()
    if uid and uid.lower() not in (MASTER_AETHERSPACE_NAME, MASTER_AETHERSPACE_UID.lower()):
        parts.append(f"space:{uid}")
    if space_tables:
        parts.append("tables:" + hashlib.sha256(",".join(sorted(space_tables)).encode("utf-8")).hexdigest()[:16])
    if visible_objects is not None:
        parts.append("vis:" + visibility_table_set_fingerprint(visible_objects))
    elif _scope_ctx_restricts_visibility(scope_ctx):
        parts.append("scope:" + scope_hash_fp(scope_ctx or EngineContext()))
    if not parts:
        return ""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def split_warmup_lattice_basename(name: str) -> tuple[str, str, str]:
    """Return ``(schema_graph_id, partition_fp, code_version)`` parsed from a lattice filename."""
    if not name.startswith("lattice_") or not name.endswith(".json"):
        return "", "", ""
    body = name[len("lattice_") : -len(".json")]
    if "_v" not in body:
        return body, "", ""
    base, code_version = body.rsplit("_v", 1)
    if len(base) >= 18 and base[-18:-16] == "__" and all(ch in "0123456789abcdef" for ch in base[-16:]):
        return base[:-18], base[-16:], code_version
    return base, "", code_version


def warmup_lattice_filename(schema_graph_id: str, partition_fp: str, code_version: str) -> str:
    """Return the on-disk anchor-lattice filename for one schema graph and scope partition."""
    graph_id = str(schema_graph_id or "").strip()
    fp = str(partition_fp or "").strip()
    if fp:
        return f"lattice_{graph_id}__{fp}_v{code_version}.json"
    return f"lattice_{graph_id}_v{code_version}.json"


def qsim_skeletons_filename(partition_fp: str = "") -> str:
    """Return the on-disk QSim skeleton cache filename for one scope partition."""
    fp = str(partition_fp or "").strip()
    if fp:
        return f"qsim_skeletons__{fp}.json.gz"
    return "qsim_skeletons.json.gz"


def effective_structural_hash_fp(structural_hash: str, scope_hash: str) -> str:
    """Combine structural and scope fingerprints into the template-store key."""
    return sha256(structural_hash + "|" + scope_hash)


def schema_hash_fp(tables_dict: dict[str, Any]) -> str:
    """SHA-256 of ``{"tables": tables_dict}`` JSON. Used by cache diagnostics and tests that pass arbitrary table-shaped dicts."""
    return sha256(stable_json({"tables": tables_dict}))


def _question_normalization_keep_char(ch: str) -> str:
    """Return *ch* when allowed in normalised questions, otherwise a space."""
    if ch.isalnum():
        return ch
    if ch.isspace() or ch in "_:/-.,?":
        return ch
    return " "


def normalize_text_value(value: str) -> str:
    """Return NFKC-normalised *value* for library-side text comparison only."""
    return unicodedata.normalize("NFKC", value)


def normalize_question(q: str) -> str:
    """Lowercase and clean *q*; restore single-quoted spans to original case."""
    q = q.strip()
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", q)

    quoted_values = []

    def preserve_quoted(m: re.Match[str]) -> str:
        quoted_values.append(m.group(1))
        return f"__QUOTED_{len(quoted_values) - 1}__"

    q = re.sub(r"'([^']*)'", preserve_quoted, q)

    q = q.lower()
    q = "".join(_question_normalization_keep_char(c) for c in q)
    q = re.sub(r"\s+", " ", q).strip()

    for i, val in enumerate(quoted_values):
        q = q.replace(f"__quoted_{i}__", f"'{val}'")

    return q


def _extract_first_json_object(s: str) -> str | None:
    """First top-level ``{...}`` substring via brace depth (after fence. strip)."""
    s = _strip_fences(s)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def safe_json_loads(s: str) -> Any | None:
    """Try ``json.loads``; on failure, parse first ``{...}`` fragment."""
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    frag = _extract_first_json_object(s)
    if frag:
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None


def _reconfigure_console_streams_to_utf8() -> None:
    """Force ``sys.stdout`` / ``sys.stderr`` to UTF-8 with replacement on undefined glyphs."""
    for _stream in (sys.stdout, sys.stderr):
        if _stream is None:
            continue
        if getattr(_stream, "encoding", "").lower() == "utf-8":
            continue
        reconfigure = getattr(_stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError, TypeError):
                pass


def paths_equal(a: Path | str, b: Path | str) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(a))) == os.path.normcase(os.path.abspath(os.fspath(b)))


def classify_database_error(exc: BaseException) -> str:
    """Return ``transient``, ``permanent``, or ``unknown`` for a driver failure."""
    exc_name = type(exc).__name__
    by_name = DATABASE_ERROR_CLASSIFICATION_BY_EXCEPTION_NAME.get(exc_name)
    if by_name is not None:
        return by_name
    msg = str(exc).lower()
    for pattern, classification in DATABASE_ERROR_CLASSIFICATION_BY_MESSAGE_PATTERN:
        if pattern in msg:
            return classification
    if isinstance(exc, OSError):
        errn = getattr(exc, "errno", None)
        if errn in DATABASE_ERROR_CLASSIFICATION_TRANSIENT_ERRNOS:
            return DATABASE_ERROR_CLASSIFICATION_TRANSIENT
    return DATABASE_ERROR_CLASSIFICATION_UNKNOWN


def wrap_database_execution_error(exc: BaseException) -> DatabaseExecutionError:
    """Wrap a driver exception in :class:`DatabaseExecutionError` with classification metadata."""
    classification = classify_database_error(exc)
    retryable = classification == DATABASE_ERROR_CLASSIFICATION_TRANSIENT
    driver_class = f"{type(exc).__module__}.{type(exc).__qualname__}"
    driver_detail = {"exception_type": type(exc).__name__, "message": str(exc)}
    cls = RetryableDatabaseExecutionError if retryable else DatabaseExecutionError
    return cls(
        "Database execution failed.",
        driver_class=driver_class,
        classification=DatabaseErrorClassification(classification),
        driver_detail=driver_detail,
    )


def engine_connect_likely_transient(exc: BaseException) -> bool:
    """Heuristic for cold-start or transport failures suitable for :class:`DatabasePingFailed`."""
    return classify_database_error(exc) == DATABASE_ERROR_CLASSIFICATION_TRANSIENT


def _normalize_sql_operator_spaces(sql: str) -> str:
    """Merge split operators: ``> =``, ``< =``, ``! =`` → ``>=``, ``<=``, ``!=``."""
    if not sql or not sql.strip():
        return sql
    s = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return s


def normalize_sql(sql: str) -> str:
    """After ``canonicalize_sql``, append default ``ASC`` where ORDER. BY. lacks direction."""
    s = _normalize_sql_operator_spaces(canonicalize_sql(sql))
    if not s:
        return s

    s_upper = s.upper()
    order_by_pos = s_upper.find("ORDER BY")
    if order_by_pos != -1:
        before_order = s[: order_by_pos + 8]
        after_order = s[order_by_pos + 8 :]

        limit_pos = after_order.upper().find("LIMIT")
        if limit_pos != -1:
            order_clause = after_order[:limit_pos].strip()
            rest = after_order[limit_pos:]
        else:
            order_clause = after_order.strip()
            rest = ""

        normalized_items = []
        for item in order_clause.split(","):
            item = item.strip()
            if not item:
                continue
            item_upper = item.upper()
            if item_upper.endswith(" ASC") or item_upper.endswith(" DESC"):
                normalized_items.append(item)
            else:
                normalized_items.append(f"{item} ASC")

        if normalized_items:
            s = f"{before_order} {', '.join(normalized_items)}"
            if rest:
                s = f"{s} {rest}"

    return s


def parse_iso_date_literal(text: str) -> date | datetime:
    """Parse an ISO 8601 calendar date or date-time literal."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty date literal")
    if ISO_DATE_ONLY_RE.match(raw):
        return date.fromisoformat(raw)
    if ISO_DATETIME_RE.match(raw):
        normalized = raw.replace(" ", "T", 1)
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        return datetime.fromisoformat(normalized)
    raise ValueError(f"ambiguous date literal: {raw!r}")


def escape_like_wildcards(value: str, escape_char: str = LIKE_ESCAPE_CHAR) -> str:
    """Escape ``%``, ``_``, and the escape character for SQL LIKE/ILIKE literals."""
    if not value:
        return value
    escaped_escape = escape_char + escape_char
    return value.replace(escape_char, escaped_escape).replace("%", escape_char + "%").replace("_", escape_char + "_")


def normalize_array_contains_param_value(value: Any) -> Any:
    """Strip whitespace and redundant surrounding quotes from array. ``contains`` operands. Keeps bind values free of decorative quotes; SQL generation also normalizes stored array elements per dialect so membership stays stable across data encodings."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.strip("%")
    return s


def refuse_unsafe_sql_string_literal_content(value: str) -> None:
    """Raise ``ValueError(UNSAFE_PARAM_LITERAL)`` when *value* cannot be inlined safely."""
    if SQL_STRING_LITERAL_STATEMENT_TERMINATOR in value:
        raise ValueError(UNSAFE_PARAM_LITERAL)
    for marker in SQL_STRING_LITERAL_COMMENT_MARKERS:
        if marker in value:
            raise ValueError(UNSAFE_PARAM_LITERAL)


def escape_sql_string_literal_body_base(value: str) -> str:
    """Escape a UTF-8 string body for safe use inside single-quoted SQL literals."""
    refuse_unsafe_sql_string_literal_content(value)
    return value.replace("'", "''")


def _escape_sql_single_quoted_literal(value: str) -> str:
    """Escape a UTF-8 string for safe use inside single-quoted SQL literals."""
    return escape_sql_string_literal_body_base(value)


def _resolve_string_literal_formatter(
    *,
    engine: str | None = None,
    dialect: Any | None = None,
) -> Callable[[str], str] | None:
    del engine
    if dialect is not None:
        quote = getattr(dialect, "quote_string_literal", None)
        if callable(quote):
            return cast(Callable[[str], str] | None, quote)
    return None


def _format_list_for_sql_inline(val: list[Any], *, format_literal: Callable[[str], str] | None = None) -> str:
    formatted_items = []
    for item in val:
        if isinstance(item, str):
            if format_literal is not None:
                formatted_items.append(format_literal(item))
            else:
                formatted_items.append(f"'{_escape_sql_single_quoted_literal(item)}'")
        else:
            formatted_items.append(str(item))
    return ", ".join(formatted_items)


def inline_allowlisted_param_value(val: Any) -> str | None:
    """Return a formatted SQL fragment only for explicit inline allowlist shapes."""
    if isinstance(val, str):
        if not val.strip():
            raise ValueError("unbound_placeholder")
        if PRE_QUOTED_IN_LIST_INLINE_RE.match(val):
            return val
        if STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE.match(val):
            return val
    return None


def substitute_params_for_execution(
    sql_param: str,
    params: dict[str, Any],
    *,
    engine: str | None = None,
    dialect: Any | None = None,
) -> str:
    """Keep bind tokens for driver binding; inline only explicit allowlist shapes and IN-list expansions."""
    format_literal = _resolve_string_literal_formatter(engine=engine, dialect=dialect)
    _result = sql_param
    for key in sorted(params.keys(), key=lambda k: -len(k)):
        val = params[key]
        if not key:
            continue
        formatted: str | None
        if isinstance(val, list):
            formatted = _format_list_for_sql_inline(val, format_literal=format_literal)
        else:
            formatted = inline_allowlisted_param_value(val)
        if formatted is None:
            continue
        for prefix in (":", "$", "@"):
            _result = _result.replace(f"{prefix}{key}", formatted)
    required: set[str] = set()
    for match in SQL_BIND_TOKEN_RE.finditer(_result):
        required.add(match.group(1))
    for match in re.finditer(r"%\((\w+)\)s", _result):
        required.add(match.group(1))
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError(f"unbound_placeholder: {', '.join(missing)}")
    return _result


def substitute_params(
    sql_param: str,
    params: dict[str, Any],
    *,
    engine: str | None = None,
    dialect: Any | None = None,
) -> str:
    """Replace ``:key``, ``@key``, and ``$key`` placeholders with formatted parameter values."""
    format_literal = _resolve_string_literal_formatter(engine=engine, dialect=dialect)
    _result = sql_param
    for key in sorted(params.keys(), key=lambda k: -len(k)):
        val = params[key]
        if not key:
            continue
        if isinstance(val, list):
            formatted = _format_list_for_sql_inline(val, format_literal=format_literal)
        elif isinstance(val, bool):
            formatted = "TRUE" if val else "FALSE"
        elif isinstance(val, str):
            if not val.strip():
                raise ValueError("unbound_placeholder")
            allowlisted = inline_allowlisted_param_value(val)
            if allowlisted is not None:
                formatted = allowlisted
            elif format_literal is not None:
                formatted = format_literal(val)
            else:
                formatted = f"'{_escape_sql_single_quoted_literal(val)}'"
        else:
            formatted = str(val)
        for prefix in (":", "$", "@"):
            _result = _result.replace(f"{prefix}{key}", formatted)
    if SQL_BIND_TOKEN_RE.search(_result) or UNBOUND_PYFORMAT_PLACEHOLDER_RE.search(_result):
        raise ValueError("unbound_placeholder")
    return _result


def _format_scalar_for_structural_sql_inline(val: Any) -> str:
    """Format a single bind value for structural placeholder inlining."""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, str):
        allowlisted = inline_allowlisted_param_value(val)
        if allowlisted is not None:
            return allowlisted
        return f"'{_escape_sql_single_quoted_literal(val)}'"
    return str(val)


def reduce_structural_sql_placeholders(
    sql_param: str,
    params: dict[str, Any],
    structural_defaults: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Inline ``:sN`` placeholders when values match defaults or identity structural values. Returns the reduced SQL string and the parameter map with structural keys removed."""
    sd = structural_defaults or {}
    keys_in_sql = sorted(
        set(STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE.findall(sql_param)),
        key=lambda k: (-len(k), k),
    )
    reduced = sql_param
    removed: set[str] = set()
    for sk in keys_in_sql:
        if sk not in params:
            continue
        val = params[sk]
        if sk in sd:
            if sd[sk] != val:
                continue
        elif val not in STRUCTURAL_IDENTITY_VALUES:
            continue
        lit = _format_scalar_for_structural_sql_inline(val)
        reduced = re.sub(rf":{re.escape(sk)}\b", lit, reduced)
        removed.add(sk)
    remaining = {k: v for k, v in params.items() if k not in removed}
    return reduced, remaining


def _format_cell(v: object) -> str:
    """String for one _result cell: ``NULL``, decimals, or ``str(v)``."""
    if v is None:
        return "NULL"
    if isinstance(v, Decimal):
        return f"{v:f}" if v == v.to_integral_value() else f"{v}"
    if isinstance(v, str):
        return v
    return str(v)


def print_query_result(
    rows: list[tuple[Any, ...]],
    sql: str,
    *,
    headers: list[str] | None = None,
) -> None:
    """Emit SQL and scalar answer or up to five aligned rows through. :func:`notify`."""
    out_lines: list[str] = [f"\n{QUERY_RESULTS_HEADER}\n", f"SQL:\n  {sql}\n"]
    if len(rows) == 1 and len(rows[0]) == 1:
        val = rows[0][0]
        out_lines.append(f"Answer: {_format_cell(val)}\n")
    else:
        sample = rows[:5]
        formatted = [[_format_cell(v) for v in row] for row in sample]
        num_cols = max(len(r) for r in formatted) if formatted else 0
        col_headers = (headers or [])[:num_cols]
        while len(col_headers) < num_cols:
            col_headers.append(f"col{len(col_headers) + 1}")
        widths = [len(h) for h in col_headers]
        for row in formatted:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(col_headers))
        sep_line = "  ".join("-" * widths[i] for i in range(num_cols))
        out_lines.append(f"  {header_line}")
        out_lines.append(f"  {sep_line}")
        for row in formatted:
            line = "  ".join((row[i] if i < len(row) else "").ljust(widths[i]) for i in range(num_cols))
            out_lines.append(f"  {line}")
        if len(rows) > 5:
            out_lines.append(f"  ... ({len(rows)} total rows)")
    notify(
        "\n".join(out_lines),
        stage="query_result",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def dataframe_to_row_tuples(df: Any) -> list[tuple[Any, ...]]:
    """Convert a pandas ``DataFrame`` to plain row tuples for :func:`print_query_result`."""
    if df is None:
        return []
    return [tuple(row) for row in df.values]


def interactive_yes_no(
    stage: str,
    prompt: str,
    options: list[str],
    silent_no: bool = False,
    *,
    choice_port: InteractiveChoicePort | None = None,
) -> str | None:
    """Resolve a yes/no prompt via an optional session port or stdin."""
    if choice_port is not None:
        return choice_port.take_yes_no(stage, prompt, options, silent_no)
    return ask_user_choice(prompt, options, silent_no)


def ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
    """Interactive ``input()`` for yes/no style choices (y/n/yes/no)."""
    options_display = "/".join(options)
    notify(
        f"{prompt} ({options_display}): ",
        stage="interactive_choice",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )
    try:
        user_input = input().strip()
    except (EOFError, KeyboardInterrupt):
        terminated()
        return None

    if not user_input or not user_input.strip():
        invalid_input()
        return None

    normalized = user_input.lower()
    if normalized in ("y", "yes"):
        notify(
            "Yes",
            stage="interactive_choice",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return "y"
    elif normalized in ("n", "no"):
        notify(
            "No",
            stage="interactive_choice",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        if not silent_no:
            terminated()
        return "n"

    invalid_input()
    return None


def _normalize_feature_word_key(text: str) -> str:
    """Collapse a feature phrase to lowercase alphanumerics for delimiter-insensitive matching."""
    return re.sub(r"[\s_\-]+", "", text.lower())


def resolve_feature_name_from_question(literal: str, question: str) -> str | None:
    """Resolve an ``item_feature.feature_name`` filter literal against the user question. Words are matched after stripping spaces, underscores, and hyphens. When a matching span exists in the question, the returned token uses underscores between words (the canonical ``item_feature.feature_name`` storage form)."""
    lit = str(literal or "").strip().strip("%").strip()
    if not lit or not question:
        return None
    lit_key = _normalize_feature_word_key(lit)
    if not lit_key:
        return None
    for match in re.finditer(r"[A-Za-z0-9]+(?:[\s_\-]+[A-Za-z0-9]+)*", question):
        span = match.group(0)
        if _normalize_feature_word_key(span) != lit_key:
            continue
        words = re.split(r"[\s_\-]+", span.strip())
        words = [w for w in words if w]
        if not words:
            continue
        return "_".join(w.lower() for w in words)
    if _normalize_feature_word_key(question).find(lit_key) >= 0:
        words = re.split(r"[\s_\-]+", lit.strip())
        words = [w for w in words if w]
        if words:
            return "_".join(w.lower() for w in words)
    return None


def print_rephrase_hint(
    reason: RephraseHint,
    *,
    rejection_bucket: str | None = None,
) -> None:
    """Print a tiered, suggestive rephrase hint for *reason*. Uses a fixed catalogue of short non-technical messages; never exposes validation logs or repair-loop internals to the user."""
    if rejection_bucket and reason in (
        RephraseHint.USER_REJECTED_RESULT,
        RephraseHint.USER_REJECTED_INTENT,
    ):
        tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(
            rejection_bucket,
            USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"],
        )
        notify(
            f"\nPlease retry or rephrase your question.\n{tip}",
            stage="rephrase_hint",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return
    hint = refusal_reformulation_hint_for_rephrase_hint_key(reason.value)
    refusal_code_by_hint = {
        RephraseHint.RESTRICTED_QUESTION: DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
        RephraseHint.VAGUE_QUESTION: DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    }
    refusal_code = refusal_code_by_hint.get(reason)
    if refusal_code:
        emit_session_refusal_diagnostic(refusal_code, f"\n{hint}", stage="rephrase_hint")
        return
    notify(
        f"\n{hint}",
        stage="rephrase_hint",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def print_info(title: str, items: dict[str, Any] | None = None, footer: str | None = None) -> None:
    """Emit *title*, optional indented *items*, and optional *footer* through :func:`notify`."""
    lines: list[str] = [f"\n{title}"]
    if items:
        for key, val in items.items():
            if isinstance(val, list | tuple | set):
                val = ", ".join(str(v) for v in val)
            lines.append(f"  {key}: {val}")
    if footer:
        lines.append(f"\n{footer}")
    notify(
        "\n".join(lines),
        stage="print_info",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def normalize_op(op: str) -> str:
    """Lowercase/whitespace-trim *op*; map ``==``, ``gte``, etc. to SQL. ops."""
    op_lower = re.sub(r"\s+", " ", op.lower().strip())
    mapping = {
        "==": "=",
        "<>": "!=",
        "ne": "!=",
        "eq": "=",
        "gt": ">",
        "lt": "<",
        "ge": ">=",
        "le": "<=",
        "gte": ">=",
        "lte": "<=",
    }
    return mapping.get(op_lower, op_lower)


def format_failure_trace(step: StepResult | list[StepResult] | object) -> str:
    """Format a step result or list of results into a diagnostic failure-trace string."""
    if isinstance(step, list):
        parts = [format_failure_trace(s) for s in step]
        return "\n\n".join(p for p in parts if p)
    lines: list[str] = []
    question = getattr(step, "question", None)
    if question:
        lines.append(f"question: {question}")
    error = getattr(step, "error", None)
    if error:
        lines.extend(str(error).strip().splitlines())
    intent = getattr(step, "intent", None)
    if intent is None:
        summary = getattr(step, "intent_summary", None)
        if summary is not None:
            tables = getattr(summary, "tables", None)
            if tables:
                lines.append(f"tables: {list(tables)}")
            grain = getattr(summary, "grain", None)
            if grain:
                lines.append(f"grain: {grain}")
    elif intent:
        it = intent
        lines.append(f"tables: {it.tables}")
        lines.append(f"grain: {it.grain}")
        gb_terms = [getattr(g, "primary_term", str(g)) for g in (it.group_by_cols or [])]
        if gb_terms:
            lines.append(f"group_by: {gb_terms}")
        agg_cols = [
            getattr(sc.expr, "primary_term", str(sc.expr))
            for sc in (it.select_cols or [])
            if getattr(sc, "is_aggregated", False)
        ]
        if agg_cols:
            lines.append(f"agg_select: {agg_cols}")
    sql = getattr(step, "sql", None)
    if sql:
        lines.append("sql:")
        lines.extend(str(sql).splitlines())
    llm_calls = getattr(step, "llm_calls", None)
    if llm_calls is not None:
        lines.append(f"llm_calls: {llm_calls}")
    status = getattr(step, "status", None)
    kind = getattr(step, "kind", None)
    if status is not None or kind is not None:
        lines.append(f"status: {status!r} kind: {kind!r}")
    captured_logs = getattr(step, "captured_logs", None)
    if captured_logs:
        for ln in captured_logs:
            lines.append(str(ln))
    diagnostics = getattr(step, "diagnostics", None)
    if diagnostics:
        for diag in diagnostics:
            code = getattr(diag, "code", None)
            message = getattr(diag, "message", None)
            if code or message:
                lines.append(f"diagnostic: {code}: {message}")
    return "\n".join(lines)


def build_session_step_trace(
    *,
    scenario_id: str,
    question: str,
    step: object | None,
    error: str | None = None,
    captured_logs: list[str] | None = None,
) -> StepResult:
    status = "failed"
    if step is not None and getattr(step, "done", False) and not error:
        status = str(getattr(step, "status", None) or "ok")
    intent = getattr(step, "intent", None) if step is not None else None
    sql = getattr(step, "sql", None) if step is not None else None
    step_error = error
    if not step_error and step is not None:
        step_error = getattr(step, "error", None) or getattr(step, "message", None)
    diagnostics = getattr(step, "diagnostics", ()) if step is not None else ()
    return StepResult(
        scenario_id=scenario_id,
        question=question,
        status=status,
        intent=intent if isinstance(intent, RuntimeIntent) else None,
        sql=str(sql) if sql else None,
        error=str(step_error).strip() if step_error else None,
        captured_logs=list(captured_logs or []),
        diagnostics=tuple(diagnostics) if diagnostics else (),
        kind=str(getattr(step, "kind", "") or "") or None,
    )


def run_with_pipeline_capture(
    fn: Callable[[], tuple[object | None, str]],
    *,
    auto_responses: list[str] | None = None,
) -> tuple[object | None, str, list[str]]:
    responses = ["y"] if auto_responses is None else auto_responses
    with pipeline_capture(auto_responses=responses) as capture:
        step, err = fn()
    return step, err, list(capture.get("logs", []))


def _make_prompt_responders(
    responses: list[str],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build FIFO auto-responders for ``ask_user_choice`` and ``interactive_yes_no`` sharing one queue."""
    queue = list(responses)

    def ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        if queue:
            return queue.pop(0)
        return "y"

    def interactive_yes_no(
        stage: str,
        prompt: str,
        options: list[str],
        silent_no: bool = False,
        *,
        choice_port: Any = None,
    ) -> str | None:
        if choice_port is not None:
            if queue:
                return queue.pop(0)
            return "y"
        if queue:
            return queue.pop(0)
        return "y"

    return ask_user_choice, interactive_yes_no


def _make_input_responder(reject_reason: str = "incorrect results") -> Callable[[str], str]:
    """Build a replacement for ``builtins.input`` that supplies canned text."""
    call_count = {"n": 0}

    def _fake_input(prompt: str = "") -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return reject_reason
        return "n"

    return _fake_input


@contextmanager
def pipeline_capture(
    auto_responses: list[str],
    reject_reason: str = "incorrect results",
    csv_dir: str = "",
) -> Iterator[dict[str, Any]]:
    """Patch interactive I/O for programmatic pipeline runs."""
    import importlib

    capture: dict[str, Any] = {"logs": []}
    ask_uc, iyn = _make_prompt_responders(auto_responses)
    input_responder = _make_input_responder(reject_reason)

    def _import_mod(short: str) -> Any:
        return importlib.import_module(f"aetherdialect.{short}")

    def _capturing_debug(msg: str) -> None:
        capture["logs"].append(f"[DEBUG] {msg}")

    _debug_module_names = (
        "_utils",
        "_pipeline_execute",
        "_pipeline_generate",
        "_sql_gen",
        "_validation_sql",
        "_validation_shape",
        "_validation_rules",
        "_intent_expr",
        "_intent_loop",
        "_intent_normalize",
        "_intent_bind",
        "_dialect",
        "_expansion_ops",
        "_utils_intent",
        "_templates",
        "_schema_graph",
        "_schema_profile",
        "_qsim",
        "_main_execution",
        "_seed_warmup",
        "_llm_provider",
    )
    extra_patches: list[Any] = []
    for short in _debug_module_names:
        mod = _import_mod(short)
        if hasattr(mod, "debug"):
            extra_patches.append(patch.object(mod, "debug", _capturing_debug))

    def _capturing_pipeline_trace(heading: str, body: str | Callable[[], str]) -> None:
        resolved = body() if callable(body) else body
        capture["logs"].append(f"[PIPELINE_TRACE] {heading}\n{resolved}")

    _trace_module_names = (
        "_utils",
        "_pipeline_execute",
        "_pipeline_generate",
        "_intent_loop",
        "_sql_gen",
        "_validation_sql",
        "_intent_bind",
        "_dialect",
        "_intent_normalize",
        "_llm_provider",
    )
    for short in _trace_module_names:
        mod = _import_mod(short)
        if hasattr(mod, "pipeline_trace"):
            extra_patches.append(patch.object(mod, "pipeline_trace", _capturing_pipeline_trace))

    core_utils_mod = _import_mod("_utils")
    pipeline_execute_mod = _import_mod("_pipeline_execute")
    pipeline_generate_mod = _import_mod("_pipeline_generate")
    main_interactive_mod = _import_mod("_main_interactive")
    main_exec_mod = _import_mod("_main_execution")

    if csv_dir:
        _original_save = pipeline_execute_mod.save_result_csv
        _csv_results_path = os.path.join(csv_dir, "results.csv")

        def _redirected_save(df: Any, *, output_path: str | os.PathLike[str] | None = None) -> None:
            _original_save(df, output_path=output_path or _csv_results_path)

        extra_patches.append(patch.object(pipeline_execute_mod, "save_result_csv", _redirected_save))
        live_testing_mod = _import_mod("_live_testing")
        if hasattr(live_testing_mod, "save_result_csv"):
            extra_patches.append(patch.object(live_testing_mod, "save_result_csv", _redirected_save))

    with (
        patch.object(core_utils_mod, "ask_user_choice", ask_uc),
        patch.object(core_utils_mod, "interactive_yes_no", iyn),
        patch.object(pipeline_execute_mod, "interactive_yes_no", iyn),
        patch.object(main_interactive_mod, "interactive_yes_no", iyn),
        patch.object(main_exec_mod, "interactive_yes_no", iyn, create=True),
        patch("builtins.input", input_responder),
    ):
        if hasattr(pipeline_generate_mod, "interactive_yes_no"):
            extra_patches.append(patch.object(pipeline_generate_mod, "interactive_yes_no", iyn))
        for p in extra_patches:
            p.start()
        try:
            yield capture
        finally:
            for p in extra_patches:
                p.stop()


def bind_params_for_sql(sql: str, param_values: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return bind map for EXPLAIN only when *sql* still contains ``:pN`` / ``@pN`` / ``$pN`` placeholders."""
    if not param_values:
        return None
    return param_values if SQL_BIND_TOKEN_RE.search(sql) else None


def reconcile_execute_bind_params(
    sql: str,
    param_values: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Filter *param_values* to tokens present in *sql* and fail when any bind token lacks a value."""
    if not SQL_BIND_TOKEN_RE.search(sql) and not UNBOUND_PYFORMAT_PLACEHOLDER_RE.search(sql):
        return None
    bind_map = bind_params_for_sql(sql, param_values) or {}
    required: set[str] = set()
    for match in SQL_BIND_TOKEN_RE.finditer(sql):
        required.add(match.group(1))
    for match in re.finditer(r"%\((\w+)\)s", sql):
        required.add(match.group(1))
    missing = sorted(required - set(bind_map.keys()))
    if missing:
        raise ValueError(f"unbound_placeholder: {', '.join(missing)}")
    return bind_map if bind_map else None


def cost_cap_active(v: float | int | None) -> bool:
    """True when *v* is a positive finite bound; ``None``, ``0``, and negatives disable the cap."""
    if v is None:
        return False
    try:
        return float(v) > 0.0
    except (TypeError, ValueError):
        return False


def diagnostic_debug_enabled() -> bool:
    """True when ``PolicyConfig.DEBUG`` or diagnostic capture (``telemetry_capture`` depth) is active."""
    return aetherdialect._constants.DIAGNOSTIC_FORCE_DEPTH > 0 or PolicyConfig.DEBUG


def diagnostic_pipeline_trace_full_enabled() -> bool:
    """True when full pipeline trace logging is enabled."""
    return diagnostic_debug_enabled()


def diagnostic_force_enter() -> None:
    """Increment nested diagnostic capture depth (used by ``telemetry_capture``)."""
    aetherdialect._constants.DIAGNOSTIC_FORCE_DEPTH += 1


def diagnostic_force_exit() -> None:
    """Decrement nested diagnostic capture depth."""
    if aetherdialect._constants.DIAGNOSTIC_FORCE_DEPTH > 0:
        aetherdialect._constants.DIAGNOSTIC_FORCE_DEPTH -= 1


def permission_denied_detail_logging_enabled() -> bool:
    """Return whether permission-denied failures may log SQL and driver detail at DEBUG."""
    raw = os.environ.get(AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV, "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def normalize_value_type(value_type: str) -> str:
    """Map a raw value-type string onto a canonical pipeline type."""
    if not value_type:
        return UNKNOWN_VALUE_TYPE
    vt_lower = value_type.lower().strip()
    if not vt_lower:
        return UNKNOWN_VALUE_TYPE
    if vt_lower in VALUE_TYPE_NORMALIZATION:
        return VALUE_TYPE_NORMALIZATION[vt_lower]
    if vt_lower in VALID_VALUE_TYPES:
        return vt_lower
    return UNKNOWN_VALUE_TYPE


def column_has_unknown_value_type(col: ColumnMetadata) -> bool:
    """Return True when profiling mapped the column onto the unknown value-type bucket."""
    return (col.value_type or "").strip().lower() == UNKNOWN_VALUE_TYPE


def join_signature_tables(sig: list[str]) -> set[str]:
    """Extract physical table names referenced in a join path signature."""
    tables: set[str] = set()
    for item in sig:
        if "->" not in item:
            continue
        left, right = item.split("->", 1)
        tables.add(left.split(".")[0].strip())
        tables.add(right.split(".")[0].strip())
    return tables
