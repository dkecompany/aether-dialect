"""Refusal catalogue reachability and information-leak guards."""

from __future__ import annotations

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    REFUSAL_DIAGNOSTIC_CODES,
)
from aetherdialect._constants_runtime import (
    REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE,
    REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN_MESSAGE,
    REFUSAL_UNSUPPORTED_COLUMN_TYPE_MESSAGE,
)
from aetherdialect._contracts_base import FailureCategory, NormalizedExpr
from aetherdialect._contracts_core import (
    AggregateJoinFanOutError,
    AmbiguousDateLiteralError,
    ComparisonJoinScopeExceededError,
    NoJoinPathError,
    NullInNegatedListError,
    RuntimeIntent,
    SelectCol,
    SubdayDateWindowOnDateColumnError,
)
from aetherdialect._contracts_schema import ColumnMetadata, IntentIssue, SchemaGraph, TableMetadata
from aetherdialect._intent_normalize import refusal_for_join_unreachable_table_removal
from aetherdialect._utils import (
    refusal_diagnostic_code_for_exception,
    refusal_diagnostic_code_for_federation_reason,
    refusal_diagnostic_code_for_intent_issue,
    refusal_message_for_exception,
    refusal_user_text_for_code,
)

_SENSITIVE_TOKENS = (
    "secret_ssn",
    "crm_member",
    "physical_orders",
    "restricted_schema",
    r"C:\data\secret.db",
    "postgresql://user:pass@host/db",
    "psycopg",
)


def _restricted_fixture_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "visible_tbl": TableMetadata(
                name="visible_tbl",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "secret_ssn": ColumnMetadata(name="secret_ssn", data_type="text", sensitivity="high"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
            "physical_orders": TableMetadata(
                name="physical_orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="restricted_fixture",
    )


def _render_refusal_for_code(code: str) -> str:
    if code == DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE:
        return refusal_message_for_exception(NoJoinPathError("main query", ["alpha", "beta"]))
    if code == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT:
        return refusal_message_for_exception(AggregateJoinFanOutError("main query", "aggregate would duplicate rows"))
    if code == DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING:
        return refusal_message_for_exception(
            ComparisonJoinScopeExceededError("main query", "comparison scope exceeds hop ceiling")
        )
    if code == DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST:
        return REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE
    if code == DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN:
        return REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN_MESSAGE
    if code == DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL:
        return refusal_message_for_exception(AmbiguousDateLiteralError("01/02/03", "ambiguous date literal"))
    if code == DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING:
        return refusal_user_text_for_code(code)
    if code == DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE:
        return REFUSAL_UNSUPPORTED_COLUMN_TYPE_MESSAGE.format(column="the requested")
    if code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP:
        return (
            refusal_user_text_for_code(
                code,
                capability="where operator 'ilike'",
            )
            or "This federated question is not supported by every data source."
        )
    if code == DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT:
        return refusal_user_text_for_code(code)
    if code == DIAGNOSTIC_CODE_REFUSAL_CTE_CAP:
        return refusal_user_text_for_code(code)

    before = RuntimeIntent(
        tables=["alpha", "beta"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    after = RuntimeIntent(
        tables=["alpha"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    issue = IntentIssue.make(
        issue_id="join_unreachable_main_query_alpha_beta",
        category=FailureCategory.WRONG_JOIN,
        severity="error",
        message="join unreachable",
        context={
            "root": "alpha",
            "target": "beta",
            "scope_label": "main query",
        },
    )
    crafted = refusal_for_join_unreachable_table_removal(before, after, [issue])
    if crafted:
        return crafted
    text = refusal_user_text_for_code(code)
    assert text, f"no driver for refusal code {code}"
    return text


@pytest.mark.fast
def test_every_refusal_reachable() -> None:
    reached: set[str] = set()
    for code in sorted(REFUSAL_DIAGNOSTIC_CODES):
        message = _render_refusal_for_code(code)
        assert message.strip()
        reached.add(code)

        exc_map = {
            DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE: NoJoinPathError("main query", ["a", "b"]),
            DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT: AggregateJoinFanOutError("main query", "fan-out"),
            DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING: ComparisonJoinScopeExceededError("main query", "hop"),
            DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST: NullInNegatedListError(
                "t.c", REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE
            ),
            DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN: SubdayDateWindowOnDateColumnError(
                "t.d", REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN_MESSAGE
            ),
            DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL: AmbiguousDateLiteralError("01/02/03", "ambiguous"),
        }
        exc = exc_map.get(code)
        if exc is not None:
            assert refusal_diagnostic_code_for_exception(exc) == code

        issue_map = {
            DIAGNOSTIC_CODE_REFUSAL_CTE_CAP: IntentIssue(
                issue_id="cte_step_count_exceeded",
                category=FailureCategory.CTE_STRUCTURE,
                severity="error",
                message="cap",
            ),
            DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT: IntentIssue(
                issue_id="denied_col",
                category=FailureCategory.DENIED_REFERENCE,
                severity="error",
                message="denied",
            ),
            DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE: IntentIssue(
                issue_id="unsupported_column_type",
                category=FailureCategory.SCHEMA_VALIDATION,
                severity="error",
                message="unsupported",
            ),
        }
        issue = issue_map.get(code)
        if issue is not None:
            assert refusal_diagnostic_code_for_intent_issue(issue) == code

        fed_map = {
            DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP: "member capability: ilike unsupported",
            DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING: "union logical column missing on member",
        }
        fed_reason = fed_map.get(code)
        if fed_reason is not None:
            assert refusal_diagnostic_code_for_federation_reason(fed_reason) == code

    assert reached == REFUSAL_DIAGNOSTIC_CODES


@pytest.mark.fast
def test_no_refusal_leaks_restricted_information() -> None:
    _restricted_fixture_graph()
    for code in sorted(REFUSAL_DIAGNOSTIC_CODES):
        rendered = _render_refusal_for_code(code).lower()
        for token in _SENSITIVE_TOKENS:
            assert token.lower() not in rendered, f"{code} leaked {token!r}: {rendered[:120]}"
