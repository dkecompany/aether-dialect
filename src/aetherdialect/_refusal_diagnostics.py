"""Stable diagnostic codes for terminal session-step refusals."""

from __future__ import annotations

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    REFUSAL_CAPABILITY_GAP_REASON_CODES,
    REFUSAL_CAPABILITY_GAP_REASON_PREFIXES,
    REFUSAL_CTE_CAP_ISSUE_IDS,
)
from aetherdialect._contracts_base import (
    AggregateJoinFanOutError,
    ComparisonJoinScopeExceededError,
    NoJoinPathError,
)
from aetherdialect._contracts_schema import IntentIssue
from aetherdialect._core_utils import notify


def refusal_diagnostic_code_for_exception(exc: BaseException) -> str | None:
    """Map a raised refusal exception to its stable diagnostic code."""
    if isinstance(exc, NoJoinPathError):
        return DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE
    if isinstance(exc, AggregateJoinFanOutError):
        return DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT
    if isinstance(exc, ComparisonJoinScopeExceededError):
        return DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING
    return None


def refusal_diagnostic_code_for_intent_issue(issue: IntentIssue) -> str | None:
    """Map a terminal intent issue to its stable refusal diagnostic code."""
    issue_id = str(issue.issue_id or "")
    if issue_id in REFUSAL_CTE_CAP_ISSUE_IDS:
        return DIAGNOSTIC_CODE_REFUSAL_CTE_CAP
    if issue_id == "comparison_join_hop_ceiling":
        return DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING
    return None


def refusal_diagnostic_code_for_federation_reason(reason: str | None) -> str | None:
    """Map a federation ineligible reason to a capability-gap refusal code when applicable."""
    if not reason:
        return None
    lowered = reason.lower()
    for prefix in REFUSAL_CAPABILITY_GAP_REASON_PREFIXES:
        if lowered.startswith(prefix):
            return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    if "is not supported by all federation members" in lowered:
        return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    for code in REFUSAL_CAPABILITY_GAP_REASON_CODES:
        if code in lowered:
            return DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    return None


def emit_session_refusal_diagnostic(
    code: str,
    message: str,
    *,
    stage: str = "validation",
    source_id: str | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> None:
    """Emit a structured refusal diagnostic for attachment to the active session step."""
    notify_kwargs = {
        "stage": stage,
        "level": "error",
        "source_id": source_id,
        "details": details,
    }
    if code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE:
        notify(message, code=DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE, **notify_kwargs)
    elif code == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT:
        notify(message, code=DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT, **notify_kwargs)
    elif code == DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING:
        notify(message, code=DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING, **notify_kwargs)
    elif code == DIAGNOSTIC_CODE_REFUSAL_CTE_CAP:
        notify(message, code=DIAGNOSTIC_CODE_REFUSAL_CTE_CAP, **notify_kwargs)
    elif code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP:
        notify(message, code=DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP, **notify_kwargs)
    elif code:
        notify(message, code=code, **notify_kwargs)


def refusal_message_for_exception(exc: BaseException) -> str:
    """Return the user-facing refusal text for *exc* when available."""
    user_message = getattr(exc, "user_message", None)
    if isinstance(user_message, str) and user_message:
        return user_message
    caller_message = getattr(exc, "message_for_caller", None)
    if isinstance(caller_message, str) and caller_message:
        return caller_message
    return str(exc)
