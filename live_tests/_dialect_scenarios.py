"""Dialect-specific live scenarios for bridge-tag and date/time SQL rendering. These exercises target per-engine syntax (``JSON_CONTAINS``, ``DATEDIFF``, ``DATE_TRUNC``, ``CHARINDEX``/``OPENJSON``, etc.) rather than full pipeline logic. Each engine module imports the subset relevant to its dialect."""

from __future__ import annotations

from aetherdialect._live_testing import Expected, Scenario
from live_tests.mydb_scenarios import FILM_SCOPED


def dialect_array_scenarios() -> list[Scenario]:
    """Membership filters on ``item_feature.feature_name`` bridge rows."""
    _film_feature = [
        *FILM_SCOPED,
        ["film", "item", "item_feature"],
        ["film", "item_feature"],
        ["item", "item_feature"],
    ]

    return [
        Scenario(
            id="DIAL-AR-001",
            question="list titles of films that have trailers in item_feature",
            expected=Expected(tables_one_of=_film_feature, min_rows=0),
            category="dialect_array",
        ),
        Scenario(
            id="DIAL-AR-002",
            question="how many films have behind_the_scenes in item_feature",
            expected=Expected(tables_one_of=_film_feature, min_rows=1),
            category="dialect_array",
        ),
        Scenario(
            id="DIAL-AR-003",
            question=(
                "list film titles that have all four item_feature values trailers, commentaries, "
                "deleted_scenes, and behind_the_scenes"
            ),
            expected=Expected(tables_one_of=_film_feature, min_rows=0),
            category="dialect_array",
        ),
    ]


def dialect_date_window_scenarios() -> list[Scenario]:
    """Relative and absolute date-window filters (``DATE_TRUNC`` / ``DATE_SUB`` / ``DATEADD``)."""
    return [
        Scenario(
            id="DIAL-DW-001",
            question="show rentals from the last 90 days",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-002",
            question="list rentals from the last 6 months",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-003",
            question="show rentals from the last year",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-004",
            question="list rentals between January 1 2023 and December 31 2023",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
    ]


def dialect_date_diff_scenarios() -> list[Scenario]:
    """Column-to-column date difference (``DATEDIFF`` / ``TIMESTAMPDIFF`` / ``DATE_DIFF``)."""
    return [
        Scenario(
            id="DIAL-DD-001",
            question="list rentals where the return was more than 7 days after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-002",
            question="list rentals where the return was more than 2 weeks after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-003",
            question="list rentals where the return was more than 1 month after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-004",
            question="list rentals where the return was more than 1 year after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
    ]


def dialect_case_insensitive_scenarios() -> list[Scenario]:
    """Case-insensitive string matching (``LOWER`` wrap; no ``ILIKE`` on MySQL/SQL Server)."""
    return [
        Scenario(
            id="DIAL-CI-001",
            question="list film titles containing the word Harbor",
            expected=Expected(tables=["item"], min_rows=1),
            category="dialect_case_insensitive",
        ),
    ]


def dialect_pagination_scenarios() -> list[Scenario]:
    """LIMIT/TOP/OFFSET pagination rendering."""
    return [
        Scenario(
            id="DIAL-PG-001",
            question="show the first 10 film titles alphabetically",
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1, max_rows=10),
            category="dialect_pagination",
        ),
    ]


def dialect_identifier_qualification_scenarios() -> list[Scenario]:
    """Schema-qualified or multi-part table references in FROM/JOIN."""
    return [
        Scenario(
            id="DIAL-IQ-001",
            question="list customer first name and last name for customers in store 1",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_identifier_qualification",
        ),
    ]


def dialect_cast_coercion_scenarios() -> list[Scenario]:
    """CAST/typed literal coercion in filters or projections."""
    return [
        Scenario(
            id="DIAL-CAST-001",
            question="list film titles where length is greater than 120",
            expected=Expected(tables=["film"], min_rows=0),
            category="dialect_cast",
        ),
    ]


def dialect_case_when_scenarios() -> list[Scenario]:
    """Minimal CASE WHEN rendering."""
    return [
        Scenario(
            id="DIAL-CASE-001",
            question="show film titles and whether length is over 120 minutes as long or short",
            expected=Expected(tables=["film"], min_rows=1),
            category="dialect_case_when",
        ),
    ]


def dialect_scalar_func_scenarios() -> list[Scenario]:
    """Scalar function rendering (upper/trim/coalesce)."""
    return [
        Scenario(
            id="DIAL-SCALAR-001",
            question="list upper case film titles",
            expected=Expected(tables=["item"], min_rows=1),
            category="dialect_scalar_func",
        ),
    ]


def dialect_boolean_scenarios() -> list[Scenario]:
    """Boolean column filters."""
    return [
        Scenario(
            id="DIAL-BOOL-001",
            question="list active customers",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_boolean",
        ),
    ]


def dialect_window_agg_scenarios() -> list[Scenario]:
    """Aggregate window functions (SUM/AVG OVER) for Databricks normalize path."""
    return [
        Scenario(
            id="DIAL-WIN-001",
            question="for each customer show their total payment amount and the running sum of payments ordered by payment date",
            expected=Expected(
                tables_one_of=[["payment", "customer"], ["customer", "payment"]],
                min_rows=1,
            ),
            category="dialect_window_agg",
        ),
    ]


def dialect_string_concat_scenarios() -> list[Scenario]:
    """CONCAT execution smoke (not a divergence test)."""
    return [
        Scenario(
            id="DIAL-CONCAT-001",
            question="show customer first and last names concatenated with a space",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_string_concat",
        ),
    ]


def dialect_explain_smoke_scenarios() -> list[Scenario]:
    """EXPLAIN hook smoke (parse/execute path only)."""
    return [
        Scenario(
            id="DIAL-EXPLAIN-001",
            question="how many films are in the database",
            expected=Expected(tables=["film"], min_rows=1, max_rows=1),
            category="dialect_explain",
        ),
    ]


def dialect_sqlite_safe_scenarios() -> list[Scenario]:
    """Dialect scenarios scoped to SQLite-supported syntax (JSON1 arrays, julianday date diff, date('now') windows)."""
    return (
        dialect_array_scenarios()
        + dialect_date_window_scenarios()
        + dialect_date_diff_scenarios()
        + dialect_case_insensitive_scenarios()
        + dialect_pagination_scenarios()
        + dialect_cast_coercion_scenarios()
        + dialect_case_when_scenarios()
        + dialect_scalar_func_scenarios()
        + dialect_boolean_scenarios()
        + dialect_string_concat_scenarios()
    )


def dialect_mysql_scenarios() -> list[Scenario]:
    """MySQL dialect-syntax scenarios targeting JSON, date windows, TIMESTAMPDIFF, and EXPLAIN."""
    return (
        dialect_array_scenarios()
        + dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:2]
        + dialect_pagination_scenarios()[:1]
        + dialect_explain_smoke_scenarios()[:1]
    )


def dialect_snowflake_scenarios() -> list[Scenario]:
    """Snowflake dialect-syntax scenarios targeting ARRAY_CONTAINS, DATEADD windows, and qualification."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:1]
        + dialect_identifier_qualification_scenarios()[:2]
    )


def dialect_bigquery_scenarios() -> list[Scenario]:
    """BigQuery dialect-syntax scenarios targeting arrays, date windows, and TABLESAMPLE profiling paths."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_date_window_scenarios()[:2]
        + dialect_case_insensitive_scenarios()[:1]
        + dialect_pagination_scenarios()[:1]
    )


def dialect_sqlserver_scenarios() -> list[Scenario]:
    """SQL Server dialect-syntax scenarios targeting OPENJSON arrays, OFFSET/FETCH, and SHOWPLAN smoke."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_pagination_scenarios()
        + dialect_date_diff_scenarios()[:1]
        + dialect_explain_smoke_scenarios()[:1]
    )


def dialect_redshift_scenarios() -> list[Scenario]:
    """Redshift dialect-syntax scenarios targeting ILIKE, SUPER/json arrays, and sortkey-aware filters."""
    return (
        dialect_array_scenarios()
        + dialect_case_insensitive_scenarios()
        + dialect_date_window_scenarios()[:1]
        + dialect_boolean_scenarios()[:1]
    )


def dialect_duckdb_scenarios() -> list[Scenario]:
    """DuckDB dialect-syntax scenarios targeting list_contains, ILIKE, date_diff, and USING SAMPLE paths."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_case_insensitive_scenarios()[:1]
        + dialect_date_diff_scenarios()[:2]
        + dialect_date_window_scenarios()[:1]
    )


def dialect_sqlite_scenarios() -> list[Scenario]:
    """SQLite dialect-syntax scenarios targeting json_each, julianday, date windows, and LIMIT pagination."""
    return dialect_sqlite_safe_scenarios()[:8]


def dialect_databricks_scenarios() -> list[Scenario]:
    """Databricks dialect-syntax scenarios targeting DATETRUNC windows, qualification, and scalar CTE joins."""
    return (
        dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:1]
        + dialect_identifier_qualification_scenarios()[:2]
        + dialect_scalar_func_scenarios()[:1]
    )
