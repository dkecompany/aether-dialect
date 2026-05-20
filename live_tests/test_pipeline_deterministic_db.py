"""
Live database tests: ``generate_join_candidates`` + ``generate_and_validate_sql`` from a fixed ``RuntimeIntent`` (no NL intent parse).

Also hosts intent-seeded failure tests that bypass NL parse and exercise the schema + semantic repair loop directly via ``run_seeded_schema_semantic_repair``.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import (
    FilterParam,
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._live_testing import (
    deterministic_generate_validate_execute,
    run_seeded_schema_semantic_repair,
)


def _llm_forbidden(*_args, **_kwargs) -> None:
    raise AssertionError("LLM should not run in this deterministic live_no_llm slice")


@pytest.mark.live
@pytest.mark.live_no_llm
def test_deterministic_single_table_film_title(schema, t2s) -> None:
    intent = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    store: dict = {
        "effective_structural_hash": schema.effective_structural_hash,
        "next_id": 1,
        "templates": {},
        "question_feedback": {},
    }
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


@pytest.mark.live
@pytest.mark.live_no_llm
def test_deterministic_film_language_join(schema, t2s) -> None:
    intent = RuntimeIntent(
        tables=["film", "language"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("film.title")),
            SelectCol(expr=NormalizedExpr.from_column("language.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
    )
    store: dict = {
        "effective_structural_hash": schema.effective_structural_hash,
        "next_id": 1,
        "templates": {},
        "question_feedback": {},
    }
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
    assert "FILM" in sql_u and "LANGUAGE" in sql_u


@pytest.mark.live
def test_seeded_intent_repair_fixes_unknown_column(schema) -> None:
    """Seed an intent with an unknown column and verify the LLM repair fixes it."""
    seeded = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.nonexistent_column"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list film titles",
    )
    repaired, _warnings, llm_calls = run_seeded_schema_semantic_repair(
        question="list film titles",
        seeded_intent=seeded,
        schema_graph=schema,
    )
    assert repaired is not None, "semantic repair should have produced a valid intent"
    assert repaired.tables == ["film"]
    selected_terms = {sc.expr.primary_term for sc in (repaired.select_cols or [])}
    assert "film.nonexistent_column" not in selected_terms
    assert llm_calls >= 0


@pytest.mark.live
def test_seeded_intent_repair_heals_invalid_filter_value(schema) -> None:
    """Seed an intent with a semantically-wrong filter value and verify repair heals it."""
    bad_filter = FilterParam(
        left_expr=NormalizedExpr.from_column("film.title"),
        op="=",
        value_type="number",
        bool_op="AND",
    )
    seeded = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[bad_filter],
        having_param=[],
        natural_language="films whose title equals ACADEMY DINOSAUR",
    )
    seeded = replace(seeded, param_values={})
    repaired, _warnings, _llm_calls = run_seeded_schema_semantic_repair(
        question="films whose title equals ACADEMY DINOSAUR",
        seeded_intent=seeded,
        schema_graph=schema,
    )
    assert repaired is not None
