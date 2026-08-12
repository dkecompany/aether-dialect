"""Fast proving tests for compose null-safety, grain repair, and CTE emission strip."""

from __future__ import annotations

import json

import pytest

from aetherdialect._contracts_base import NormalizedExpr, OrderByCol, WhereParam
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import SchemaGraph, WindowRegistryStep, WindowSpec
from aetherdialect._dialect import DialectRegistry
from aetherdialect._intent_bind import enforce_cte_grain_consistency, enforce_grain_consistency
from aetherdialect._intent_expr import parse_intent_response
from aetherdialect._intent_loop import apply_deterministic_repairs
from aetherdialect._sql_gen import build_deterministic_sql, render_expr_sql
from aetherdialect._validation_sql import validate_cte_emission_reclassification


def _empty_schema() -> SchemaGraph:
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})


@pytest.mark.fast
def test_order_by_col_from_dict_null_direction() -> None:
    col = OrderByCol.from_dict({"expr": "t.c", "direction": None})
    assert col.direction == "ASC"


@pytest.mark.fast
def test_where_param_from_dict_null_op_and_value_type() -> None:
    pred = WhereParam.from_dict({"left_expr": "t.c", "op": None, "value_type": None})
    assert pred.op == "="
    assert pred.value_type == "string"


@pytest.mark.fast
def test_window_spec_from_dict_null_function() -> None:
    spec = WindowSpec.from_dict({"function": None, "order_by": [{"expr": "t.c", "direction": None}]})
    assert spec.function == ""
    assert spec.order_by[0].direction == "ASC"


@pytest.mark.fast
def test_parse_intent_response_null_natural_language_and_window() -> None:
    raw = {
        "tables": ["t"],
        "select_cols": ["t.c"],
        "natural_language": None,
        "order_by_cols": [{"expr": "t.c", "direction": None}],
        "window_registry": [
            {
                "registry_id": "w01",
                "window_spec": {
                    "function": None,
                    "argument": None,
                    "frame_kind": None,
                    "order_by": [{"expr": "t.c", "direction": "desc"}],
                    "partition_by": [],
                },
            }
        ],
        "extra_llm_key": "drop_me",
    }
    detail: list[str] = []
    intent = parse_intent_response(json.dumps(raw), "q", parse_detail_out=detail)
    assert intent is not None, detail
    assert intent.natural_language == "q"
    assert intent.window_registry
    assert intent.window_registry[0].window_spec.function == "row_number"


@pytest.mark.fast
def test_enforce_grain_consistency_sets_scalar_for_agg_without_group_by() -> None:
    intent = RuntimeIntent(
        tables=["payment"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "payment.payment_id"))],
    )
    out = enforce_grain_consistency(intent, _empty_schema())
    assert out.grain == "scalar"


@pytest.mark.fast
def test_enforce_cte_grain_consistency_downgrades_false_scalar() -> None:
    cte = RuntimeCteStep(
        cte_name="cte1",
        tables=["t"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.c"))],
        output_columns=["c"],
    )
    out = enforce_cte_grain_consistency(cte)
    assert out.grain == "row_level"


@pytest.mark.fast
def test_declared_scalar_subquery_survives_and_classifies() -> None:
    cte = RuntimeCteStep(
        cte_name="cte1",
        tables=["item"],
        grain="scalar",
        emission="scalar_subquery",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("max", "item.replacement_cost"))],
        output_columns=["max_replacement_cost"],
    )
    intent = RuntimeIntent(
        tables=["item", "cte1"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("item.title")),
            SelectCol(expr=NormalizedExpr.from_column("item.replacement_cost")),
        ],
        cte_steps=[cte],
    )
    repaired = apply_deterministic_repairs(intent, _empty_schema())
    assert repaired.cte_steps
    declared = repaired.cte_steps[0].emission
    label = declared.value if hasattr(declared, "value") else str(declared)
    assert label == "scalar_subquery"
    issues = validate_cte_emission_reclassification(repaired, _empty_schema())
    assert not any(i.severity == "error" for i in issues)
    from aetherdialect._sql_gen import classify_cte_emission

    classified = classify_cte_emission(repaired.cte_steps[0], repaired, _empty_schema())
    assert (classified.value if hasattr(classified, "value") else str(classified)) == "scalar_subquery"


@pytest.mark.fast
def test_window_registry_fallback_resolves_cte_local_id() -> None:
    win = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(function="rank", order_by=[OrderByCol(expr=NormalizedExpr.from_column("cte1.n"))]),
    )
    cte1 = RuntimeCteStep(
        cte_name="cte1",
        tables=["actor"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("actor.actor_id")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "actor.actor_id")),
        ],
        group_by_cols=[NormalizedExpr.from_column("actor.actor_id")],
        output_columns=["actor_id", "n"],
    )
    cte2 = RuntimeCteStep(
        cte_name="cte2",
        tables=["cte1"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("cte1.actor_id")),
            SelectCol(expr=NormalizedExpr.from_column("cte1.n")),
            SelectCol(expr=NormalizedExpr.from_column("w01")),
        ],
        output_columns=["actor_id", "n", "film_rank"],
        window_registry=[win],
    )
    intent = RuntimeIntent(
        tables=["cte2"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte2.actor_id"))],
        cte_steps=[cte1, cte2],
    )
    dialect = DialectRegistry.get("duckdb")
    sql = build_deterministic_sql(intent, dialect=dialect)
    assert "OVER" in sql.upper()
    with WindowRegistryStep.render_fallback_scope([win], []):
        rendered = render_expr_sql(NormalizedExpr.from_column("w01"), dialect)
    assert "OVER" in rendered.upper()
