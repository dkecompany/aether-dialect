"""Tests that CONCAT MulGroup preserves multiply argument order."""

from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._sql_gen import _render_group_sql


def test_concat_preserves_multiply_order() -> None:
    """CONCAT MulGroup keeps multiply list order when signature keys would sort differently."""
    a = NormalizedExpr.from_column("a.col")
    b = NormalizedExpr.from_column("b.col")
    assert a.signature_key < b.signature_key

    g = MulGroup(scalar_func="concat", multiply=[b, a])
    assert [m.column_ref for m in g.multiply] == ["b.col", "a.col"]

    sql = _render_group_sql(g)
    b_pos = sql.index("b.col")
    a_pos = sql.index("a.col")
    assert b_pos < a_pos
