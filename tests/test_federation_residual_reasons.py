"""Tests for coordinator residual rendering and retired ineligible reason strings."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_core import OrderByCol, ResidualSpec, SelectCol
from aetherdialect._federation import render_federation_residual_sql
from aetherdialect._intent_process import NormalizedExpr

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "aetherdialect"
_RETIRED_REASONS = (
    "cross-source ORDER BY is not supported",
    "cross-source DISTINCT is not supported",
)


@pytest.mark.fast
def test_residual_order_by_renders_null_placement() -> None:
    residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("left_t.id"), direction="ASC", nulls="last"),),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual)
    assert "NULLS LAST" in sql.upper()


@pytest.mark.fast
def test_retired_cross_source_clause_reasons_have_no_emission_site() -> None:
    hits: list[str] = []
    for path in _SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for reason in _RETIRED_REASONS:
            if reason in text:
                hits.append(f"{path.relative_to(_REPO_ROOT)}: {reason}")
    assert not hits
