"""Live database tests: ``generate_and_validate_sql`` from a fixed ``RuntimeIntent`` (no NL intent parse)."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._live_testing import deterministic_generate_validate_execute
from aetherdialect._templates import empty_template_store


def _llm_forbidden(*_args, **_kwargs) -> None:
    raise AssertionError("LLM should not run in this deterministic live_no_llm slice")


def test_deterministic_single_table_film_title(schema, t2s) -> None:
    intent = RuntimeIntent(
        tables=["item"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("item.title"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    store = empty_template_store(schema.effective_structural_hash)
    with patch("aetherdialect._pipeline.get_join_choice_from_llm", side_effect=_llm_forbidden):
        gen_out, rows = deterministic_generate_validate_execute(
            q_norm="deterministic film titles",
            intent=intent,
            schema=schema,
            dialect=t2s.dialect,
            store=store,
        )
    assert gen_out.success is True
    assert rows is not None
    assert len(rows) >= 1
    assert all(isinstance(r[0], str) for r in rows[: min(5, len(rows))])


def test_deterministic_film_language_join(schema, t2s) -> None:
    intent = RuntimeIntent(
        tables=["item", "language"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("item.title")),
            SelectCol(expr=NormalizedExpr.from_column("language.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    store = empty_template_store(schema.effective_structural_hash)
    with patch("aetherdialect._pipeline.get_join_choice_from_llm", side_effect=_llm_forbidden):
        gen_out, rows = deterministic_generate_validate_execute(
            q_norm="deterministic film and language",
            intent=intent,
            schema=schema,
            dialect=t2s.dialect,
            store=store,
        )
    assert gen_out.success is True
    assert rows is not None
    sql_u = (gen_out.sql or "").upper()
    assert "JOIN" in sql_u
    assert "ITEM" in sql_u and "LANGUAGE" in sql_u
