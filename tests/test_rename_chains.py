"""Rename history chain composition for multi-hop schema identity rotation."""

from __future__ import annotations

import pytest

from aetherdialect._templates import BrokenRenameChainError
from aetherdialect._templates_ops import TemplateOps


def _entry(
    from_id: str,
    to_id: str,
    *,
    tables: tuple[tuple[str, str], ...] = (),
    columns: tuple[tuple[str, str, str], ...] = (),
) -> dict:
    return {
        "from_schema_graph_id": from_id,
        "to_schema_graph_id": to_id,
        "renamed_tables": [list(pair) for pair in tables],
        "renamed_columns": [list(pair) for pair in columns],
        "applied_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.mark.fast
def test_two_hop_rename_composed() -> None:
    history = [
        _entry("sg_a", "sg_b", tables=(("orders", "sales_orders"),), columns=(("orders", "amt", "amount"),)),
        _entry("sg_b", "sg_c", tables=(("sales_orders", "so"),), columns=(("sales_orders", "amount", "total"),)),
    ]
    tables, columns = TemplateOps.compose_rename_chain(history, "sg_a", "sg_c")
    assert dict(tables) == {"orders": "so"}
    assert list(columns) == [("orders", "amt", "total")]


@pytest.mark.fast
def test_broken_chain_refused() -> None:
    history = [
        _entry("sg_a", "sg_b", tables=(("orders", "sales_orders"),)),
        _entry("sg_x", "sg_c", tables=(("sales_orders", "so"),)),
    ]
    with pytest.raises(BrokenRenameChainError, match="sg_a"):
        TemplateOps.compose_rename_chain(history, "sg_a", "sg_c")
