"""Clause-level validation when joins widen the row set."""

from __future__ import annotations

from aetherdialect._contracts_base import NormalizedExpr, OrderByCol
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import SchemaGraph, WindowRegistryStep, WindowSpec
from aetherdialect._validation_execute import validate_window_join_fan_out
from aetherdialect._validation_schema import validate_clause_widened_rowset
from tests.test_join_fan_out import _join_signature, _parent_child_schema


def _parent_child_intent(**overrides) -> RuntimeIntent:
    schema = _parent_child_schema()
    base = {
        "tables": ["parent", "child"],
        "grain": "row_level",
        "select_cols": [SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        "group_by_cols": [],
        "order_by_cols": [],
        "where": None,
        "chosen_join_path_signature": _join_signature(schema),
    }
    base.update(overrides)
    return RuntimeIntent(**base)


def test_distinct_on_order_must_begin_with_partition_expressions(simple_schema: SchemaGraph) -> None:
    from aetherdialect._validation_schema import validate_distinct_on_schema

    issues = validate_distinct_on_schema(
        [NormalizedExpr.from_column("customers.id")],
        [OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC")],
        simple_schema,
        {"customers"},
        context="main query",
    )
    assert any(i.issue_id.startswith("distinct_on_order_prefix_") for i in issues)


def test_distinct_on_matching_order_prefix_is_allowed(simple_schema: SchemaGraph) -> None:
    from aetherdialect._validation_schema import validate_distinct_on_schema

    issues = validate_distinct_on_schema(
        [NormalizedExpr.from_column("customers.id")],
        [
            OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC"),
            OrderByCol(expr=NormalizedExpr.from_column("customers.balance"), direction="DESC"),
        ],
        simple_schema,
        {"customers"},
        context="main query",
    )
    assert not any(i.issue_id.startswith("distinct_on_order_prefix_") for i in issues)


def test_window_over_multiplied_parent_refuses_with_fan_out_message() -> None:
    schema = _parent_child_schema()
    sig = _join_signature(schema)
    window = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("parent.id")],
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("parent.amount"), direction="DESC")],
        ),
    )
    intent = RuntimeIntent(
        tables=["parent", "child"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[window],
        chosen_join_path_signature=sig,
    )
    issues = validate_window_join_fan_out(intent, schema, "main query", from_anchor="parent")
    assert any(i.severity == "error" and "semi_join probe" in i.message for i in issues)


def test_limit_on_multiplied_anchor_refuses() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(limit=10)
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert any(i.severity == "error" and i.issue_id.startswith("clause_widened_rowset_limit_") for i in issues)


def test_order_by_on_multiplied_anchor_diagnostic() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("parent.amount"), direction="DESC")]
    )
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert any(i.severity == "warning" and i.issue_id.startswith("clause_widened_rowset_order_by_") for i in issues)


def test_select_distinct_on_multiplied_anchor_diagnostic() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(distinct_select_index=0)
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert any(
        i.severity == "warning" and i.issue_id.startswith("clause_widened_rowset_select_distinct_") for i in issues
    )


def test_distinct_on_partition_on_multiplied_side_refuses() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(
        distinct_on=[NormalizedExpr.from_column("parent.id")],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("parent.id"), direction="ASC")],
    )
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert any(i.severity == "error" and i.issue_id.startswith("clause_widened_rowset_distinct_on_") for i in issues)


def test_count_star_on_multiplied_anchor_diagnostic() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "*"))])
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert any(i.severity == "warning" and i.issue_id.startswith("clause_widened_rowset_count_star_") for i in issues)


def test_grouped_parent_query_exempt_from_clause_widened_checks() -> None:
    schema = _parent_child_schema()
    intent = _parent_child_intent(
        limit=10,
        group_by_cols=[NormalizedExpr.from_column("parent.id")],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
    )
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="parent")
    assert not any(i.issue_id.startswith("clause_widened_rowset_") for i in issues)


def test_many_to_one_path_has_no_clause_widened_issues() -> None:
    schema = _parent_child_schema()
    intent = RuntimeIntent(
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=10,
        chosen_join_path_signature=_join_signature(schema),
    )
    issues = validate_clause_widened_rowset(intent, schema, "main query", from_anchor="child")
    assert not any(i.issue_id.startswith("clause_widened_rowset_") for i in issues)
