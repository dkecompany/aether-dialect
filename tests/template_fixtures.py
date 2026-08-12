"""Shared template fixtures for unit tests."""

from __future__ import annotations

from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import ConcreteIntent, Template, ValueHistory
from aetherdialect._contracts_schema import SQLShape, TemplateStats


def _minimal_template(*, question: str = "count of item in category x") -> Template:
    intent_sig = ConcreteIntent(
        intent_id="i1",
        tables=["t1"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t1.category"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    return Template(
        id="T0001",
        intent_signature=intent_sig,
        intent_key="k1",
        tables_used=["t1"],
        sql_param="SELECT 1 FROM t1 WHERE category = :p1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "x"}],
            questions=[question],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
        param_display_names={"p1": "category"},
    )
