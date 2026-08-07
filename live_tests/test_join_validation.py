"""Seeded-intent join-validation live tests. Each case seeds a ``RuntimeIntent`` that omits a required bridge table and asserts ``generate_join_candidates`` proposes a candidate covering the bridge, so the join-path resolver actually runs instead of being short-circuited by a successful NL parse."""

from __future__ import annotations

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._pipeline import generate_join_candidates


def _candidate_tables(candidates: list[dict]) -> set[str]:
    """Collect source and destination table names from all candidate ``join_path_signature`` edges."""
    tables: set[str] = set()
    for cand in candidates:
        for edge_sig in cand.get("join_path_signature") or []:
            if not isinstance(edge_sig, str) or "->" not in edge_sig:
                continue
            src_side, dst_side = edge_sig.split("->", 1)
            src_table = src_side.split(".", 1)[0]
            dst_table = dst_side.split(".", 1)[0]
            if src_table:
                tables.add(src_table)
            if dst_table:
                tables.add(dst_table)
    return tables


def test_seeded_join_candidates_bridge_actor_film(schema) -> None:
    """``actor`` ↔ ``film`` requires the ``film_actor`` bridge to appear among candidates."""
    intent = RuntimeIntent(
        tables=["actor", "film"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("actor.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("film.title")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    join_candidates, _cmap, _cte_hints = generate_join_candidates(intent, schema)
    candidates = join_candidates.get("candidates") or []
    assert candidates, "join candidates must exist for actor/film"
    assert "film_actor" in _candidate_tables(candidates)


def test_seeded_join_candidates_bridge_item_category(schema) -> None:
    """``film`` ↔ ``category`` requires the ``item_category`` bridge table."""
    intent = RuntimeIntent(
        tables=["film", "category"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("film.title")),
            SelectCol(expr=NormalizedExpr.from_column("category.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    join_candidates, _cmap, _cte_hints = generate_join_candidates(intent, schema)
    candidates = join_candidates.get("candidates") or []
    assert candidates, "join candidates must exist for film/category"
    assert "item_category" in _candidate_tables(candidates)


def test_seeded_join_candidates_bridge_customer_country(schema) -> None:
    """``customer`` ↔ ``country`` goes through ``address`` and ``city``."""
    intent = RuntimeIntent(
        tables=["customer", "country"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("country.country")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    join_candidates, _cmap, _cte_hints = generate_join_candidates(intent, schema)
    candidates = join_candidates.get("candidates") or []
    assert candidates, "join candidates must exist for customer/country"
    resolved = _candidate_tables(candidates)
    assert "address" in resolved
    assert "city" in resolved
