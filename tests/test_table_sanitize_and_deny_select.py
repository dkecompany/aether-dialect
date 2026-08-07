"""Tests for symmetric table sanitization and deny-column bare-select checks."""

from aetherdialect._contracts_base import (
    FailureCategory,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._intent_repair import sanitize_table_names
from aetherdialect._validation_semantic import (
    validate_denied_references,
    validate_deny_bare_select,
    validate_logical_intent_numeric_coverage,
)


def _minimal_schema(*names: str) -> SchemaGraph:
    tables = {n: TableMetadata(name=n, columns={}, primary_key=[], foreign_keys=[], row_count=10) for n in names}
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        effective_structural_hash="x",
    )


def test_sanitize_table_names_corrects_cte_tables() -> None:
    schema = _minimal_schema("orders")
    cte = RuntimeCteStep(
        cte_name="c1",
        tables=["FROM orders"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
    )
    intent = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        cte_steps=[cte],
    )
    out = sanitize_table_names(intent, schema)
    assert out.cte_steps[0].tables == ["orders"]


def _schema_with_denied_secret() -> SchemaGraph:
    t = TableMetadata(
        name="t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "secret": ColumnMetadata(name="secret", data_type="text"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"t": t},
        join_paths_multi={},
        effective_structural_hash="x",
        deny_columns={"t": {"secret"}},
    )


def test_validate_deny_bare_select_flags_non_terminal_cte_bare_denied_col() -> None:
    """Probe/intermediate CTE bodies must be scanned for bare denied selects."""
    schema = _schema_with_denied_secret()
    inner = RuntimeCteStep(
        cte_name="inner_x",
        tables=["t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.secret"))],
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        cte_steps=[inner],
    )
    issues = validate_deny_bare_select(intent, schema)
    assert len(issues) == 1
    assert issues[0].category == FailureCategory.DENY_BARE_SELECT


def test_validate_denied_references_flags_non_terminal_cte_denied_col() -> None:
    """Probe/intermediate CTE bodies must be scanned for denied references."""
    schema = _schema_with_denied_secret()
    inner = RuntimeCteStep(
        cte_name="inner_x",
        tables=["t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.secret"),
                    op="=",
                    value_type="text",
                    raw_value="x",
                ),
            ]
        ),
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        cte_steps=[inner],
    )
    issues = validate_denied_references(intent, schema)
    assert len(issues) == 1
    assert issues[0].category == FailureCategory.DENIED_REFERENCE


def test_validate_deny_bare_select_flags_terminal_cte_bare_denied_col() -> None:
    t = TableMetadata(
        name="t",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "secret": ColumnMetadata(name="secret", data_type="text"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    schema = SchemaGraph(
        tables={"t": t},
        join_paths_multi={},
        effective_structural_hash="x",
        deny_columns={"t": {"secret"}},
    )
    cte = RuntimeCteStep(
        cte_name="x",
        tables=["t"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.secret"))],
    )
    intent = RuntimeIntent(
        tables=["t", "x"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        cte_steps=[cte],
    )
    issues = validate_deny_bare_select(intent, schema)
    assert len(issues) == 1
    assert issues[0].category == FailureCategory.DENY_BARE_SELECT


def test_tier2_numeric_coverage_short_circuits_on_empty_nl() -> None:
    """Numeric coverage from planner prose is skipped when no logical intent is supplied."""
    assert (
        validate_logical_intent_numeric_coverage(
            None,
            [],
            None,
            "CTE 'c1'",
            param_values={},
        )
        == []
    )


def test_runtime_and_cte_step_share_query_body_field_parity() -> None:
    """Runtime intent and CTE step expose the same structural body fields."""
    ri = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    assert hasattr(ri, "limit_param_key") and hasattr(ri, "param_values")
    cte = RuntimeCteStep(cte_name="x", tables=["t"])
    assert hasattr(cte, "param_values") and hasattr(cte, "limit_param_key")
