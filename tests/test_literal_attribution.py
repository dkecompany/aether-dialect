"""Unit tests for planner-prose literal attribution helpers."""

from __future__ import annotations

from aetherdialect._contracts_base import (
    CteIntent,
    FailureCategory,
    LogicalIntent,
)
from aetherdialect._contracts_schema import (
    IntentIssue,
)
from aetherdialect._intent_resolve import (
    attribute_post_compose_issue,
    literal_in_logical_prose,
)


def _issue_missing_numeric(value: str) -> IntentIssue:
    return IntentIssue(
        issue_id=f"missing_numeric_{value}_main",
        category=FailureCategory.MISSING_NUMERIC_WHERE,
        severity="warning",
        message=f"Coverage text mentions number '{value}'",
        context={"value": value, "location": "main"},
        responsible_stage="format",
    )


class TestLiteralInLogicalProse:
    def test_pg13_substring_in_filter(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="title",
            where="films whose rating equals 'PG-13'",
        )
        assert literal_in_logical_prose(li, "PG-13") is True

    def test_case_insensitive_pg13(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="title",
            where="rating equals pg-13",
        )
        assert literal_in_logical_prose(li, "PG-13") is True

    def test_token_in_second_cte_filter(self) -> None:
        li = LogicalIntent(
            tables=("a",),
            select="x",
            cte_steps=(
                CteIntent(name="c0", tables=("a",), select="pk"),
                CteIntent(
                    name="c1",
                    tables=("a", "c0"),
                    select="y",
                    where="amount is greater than 5",
                ),
            ),
        )
        assert literal_in_logical_prose(li, "5") is True

    def test_empty_token_false(self) -> None:
        li = LogicalIntent(tables=("t",), select="x")
        assert literal_in_logical_prose(li, "") is False


class TestAttributePostStageBIssue:
    def test_pg13_in_prose_routes_format(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="title",
            where="films whose rating equals 'PG-13'",
        )
        iss = _issue_missing_numeric("PG-13")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "compose"

    def test_vague_prose_routes_logical(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="title",
            where="films with the right rating",
        )
        iss = _issue_missing_numeric("PG-13")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "ground"

    def test_payments_above_5_format(self) -> None:
        li = LogicalIntent(
            tables=("payment",),
            select="amount",
            where="payments above 5",
        )
        iss = _issue_missing_numeric("5")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "compose"

    def test_wrong_digit_logical(self) -> None:
        li = LogicalIntent(
            tables=("payment",),
            select="amount",
            where="payments above 5",
        )
        iss = _issue_missing_numeric("13")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "ground"

    def test_missing_literal_mismatch_prose_logical(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="film_id",
            where="rating = 'R'",
        )
        iss = _issue_missing_numeric("PG-13")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "ground"

    def test_missing_literal_match_cte_prose_format(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="film_id",
            cte_steps=(
                CteIntent(
                    name="c1",
                    tables=("film",),
                    select="film_id",
                    where="length > 5",
                ),
            ),
        )
        iss = _issue_missing_numeric("5")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "compose"

    def test_missing_literal_mismatch_cte_prose_logical(self) -> None:
        li = LogicalIntent(
            tables=("film",),
            select="film_id",
            cte_steps=(
                CteIntent(
                    name="c1",
                    tables=("film",),
                    select="film_id",
                    where="length > 10",
                ),
            ),
        )
        iss = _issue_missing_numeric("13")
        out = attribute_post_compose_issue(iss, li)
        assert out.responsible_stage == "ground"

    def test_attribute_post_compose_issue_skip_non_literal_category(self) -> None:
        li = LogicalIntent(tables=("t",), select="x")
        iss = IntentIssue(
            issue_id="x",
            category=FailureCategory.UNKNOWN_TABLE,
            severity="error",
            message="m",
            responsible_stage="ground",
        )
        assert attribute_post_compose_issue(iss, li) is iss
