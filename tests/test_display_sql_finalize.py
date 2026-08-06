"""Display SQL finalize on stored-template execution uses full substitution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import (
    ConcreteIntent,
    GenerationPath,
    SelectCol,
    SqlGenerationOutcome,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape
from aetherdialect._pipeline import execute_stored_template_by_ref


def _template() -> Template:
    intent_sig = ConcreteIntent(
        intent_id="t1",
        tables=["items"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("items.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("items.status"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                )
            ]
        ),
    )
    return Template(
        id="T0001",
        effective_structural_hash="eff_items",
        intent_signature=intent_sig,
        intent_key="ik_items",
        tables_used=["items"],
        sql_param="SELECT id FROM items WHERE status = :p1",
        sql_fp="fp_items",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(
            param_values=[{"p1": "active"}],
            questions=["list active items"],
            natural_language=["list active items"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _dialect() -> MagicMock:
    dialect = MagicMock()
    dialect.execute.return_value = [(1,)]
    dialect.sqlglot_dialect = "duckdb"
    return dialect


@pytest.mark.fast
def test_template_execution_display_inlines() -> None:
    """Stored-template display SQL inlines scalar binds for UI rendering."""
    tmpl = _template()
    dialect = _dialect()
    store: dict = {"templates": {tmpl.id: tmpl}}
    exec_sql = "SELECT id FROM items WHERE status = :p1"
    outcome = SqlGenerationOutcome(exec_sql, True, GenerationPath.EXACT_QUESTION_REUSE, tmpl)
    with patch("aetherdialect._pipeline.execute_reuse_with_params", return_value=outcome):
        result = execute_stored_template_by_ref(
            tmpl.id,
            {"p1": "active"},
            question="list active items",
            dialect=dialect,
            store=store,
            templates={tmpl.id: tmpl},
            rejected={},
            schema=MagicMock(),
        )
    assert result.sql == exec_sql
    assert "status = 'active'" in result.display_sql
    assert ":p1" not in result.display_sql
    assert "$p1" not in result.display_sql
