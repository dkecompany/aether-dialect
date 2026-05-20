"""
Seeded live tests that pin each ``GenerationPath`` routing branch in isolation.

Every test owns its own empty template store (``isolated_runner``) so path assertions cannot be contaminated by templates persisted on disk or seeded by earlier tests. The deterministic paths (``1``, ``2.1``, ``2.2``, ``5``) run end-to-end against the live database; the union / intent-direct paths (``3``, ``4.1``, ``4.2``, ``4.3``) still require the intent-parse LLM but their template fixtures are fully seeded so the only LLM work that matters is the parse itself.
"""

from __future__ import annotations

import pytest

from aetherdialect._config import GenerationPath
from aetherdialect._contracts_core import NormalizedExpr, RuntimeIntent, SelectCol
from aetherdialect._live_testing import Expected, Scenario, run_and_assert

from ._seed_helpers import (
    deterministic_join_choice_patch,
    isolated_runner,
    seed_template,
    seeded_runner,
)


def _customer_names_intent_with_limit(limit_value: int) -> RuntimeIntent:
    """Build a row-level customer-names intent with a literal ``LIMIT :s1`` binding."""

    intent = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("customer.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list customer first and last names",
        limit=limit_value,
        limit_param_key="s1",
        param_values={"s1": limit_value},
        sql_param="SELECT customer.first_name, customer.last_name FROM customer LIMIT :s1",
    )
    return intent


def _scenario_for(question: str, path: GenerationPath, *, scenario_id: str) -> Scenario:
    """Build a minimal ``Scenario`` that asserts the final generation path code."""

    return Scenario(
        id=scenario_id,
        question=question,
        expected=Expected(generation_path=path.code),
        category="generation_path",
    )


@pytest.mark.live
def test_gp1_exact_qnorm_reuse(schema, schema_terms, t2s) -> None:
    """Seed one template; re-asking the exact normalised question returns path ``1``."""

    with seeded_runner(schema, schema_terms, t2s, label="gp1", kits=("cold_templates",)) as runner:
        run_and_assert(
            runner,
            _scenario_for(
                "list customer first and last names",
                GenerationPath.EXACT_QUESTION_REUSE,
                scenario_id="GP1-EXACT",
            ),
            header="[GP1-EXACT]",
        )


@pytest.mark.live
def test_gp2_1_fuzzy_literal_structural(schema, schema_terms, t2s) -> None:
    """Fuzzy variant with identical structural defaults routes to path ``2.1``."""

    with (
        seeded_runner(schema, schema_terms, t2s, label="gp2_1", kits=("cold_templates",)) as runner,
        deterministic_join_choice_patch(),
    ):
        run_and_assert(
            runner,
            _scenario_for(
                "list customer first and last name",
                GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL,
                scenario_id="GP2_1-FUZZY-LITERAL",
            ),
            header="[GP2_1-FUZZY-LITERAL]",
        )


@pytest.mark.live
def test_gp2_2_fuzzy_full_params(schema, schema_terms, t2s) -> None:
    """Fuzzy variant where the history row's ``s1`` differs from defaults routes to ``2.2``."""

    seed_intent = _customer_names_intent_with_limit(5)
    with (
        isolated_runner(schema, schema_terms, t2s, label="gp2_2") as runner,
        deterministic_join_choice_patch(),
    ):
        seed_template(
            runner,
            q_norm="top 5 customer first and last names",
            intent=seed_intent,
            sql="SELECT first_name, last_name FROM customer LIMIT 5",
            trust_level=1,
            structural_override={"s1": 10},
        )
        run_and_assert(
            runner,
            _scenario_for(
                "top 5 customer first and last name",
                GenerationPath.FUZZY_REUSE_FULL_PARAMS,
                scenario_id="GP2_2-FUZZY-FULL",
            ),
            header="[GP2_2-FUZZY-FULL]",
        )


@pytest.mark.live
def test_gp3_intent_direct_match(schema, schema_terms, t2s) -> None:
    """Seed a trusted template; a differently-worded question parsing to the same intent returns ``3``."""

    with (
        seeded_runner(schema, schema_terms, t2s, label="gp3", kits=("baseline_templates",)) as runner,
        deterministic_join_choice_patch(),
    ):
        scenario = Scenario(
            id="GP3-INTENT-DIRECT",
            question="for each customer row output customer_id together with the first_name column",
            expected=Expected(
                generation_path_in=(
                    GenerationPath.INTENT_DIRECT_MATCH.code,
                    GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN.code,
                    GenerationPath.UNION_TEMPLATE_WIDEN.code,
                    GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE.code,
                ),
            ),
            category="generation_path",
        )
        run_and_assert(runner, scenario, header="[GP3-INTENT-DIRECT]")


@pytest.mark.live
def test_gp4_1_union_template_widen(schema, schema_terms, t2s) -> None:
    """Seed a single-column template; asking for a superset of columns routes to ``4.1``."""

    with (
        seeded_runner(schema, schema_terms, t2s, label="gp4_1", kits=("baseline_templates",)) as runner,
        deterministic_join_choice_patch(),
    ):
        scenario = Scenario(
            id="GP4_1-UNION-WIDEN",
            question="list each customer's first name and email address",
            expected=Expected(
                generation_path_in=(
                    GenerationPath.UNION_TEMPLATE_WIDEN.code,
                    GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN.code,
                ),
            ),
            category="generation_path",
        )
        run_and_assert(runner, scenario, header="[GP4_1-UNION-WIDEN]")


@pytest.mark.live
def test_gp4_2_union_template_and_runtime_widen(schema, schema_terms, t2s) -> None:
    """Seed a two-column template; asking for a partial-overlap column set routes to ``4.2``."""

    with (
        seeded_runner(schema, schema_terms, t2s, label="gp4_2", kits=("baseline_templates",)) as runner,
        deterministic_join_choice_patch(),
    ):
        scenario = Scenario(
            id="GP4_2-BOTH-WIDEN",
            question="list customer first names and emails",
            expected=Expected(
                generation_path_in=(
                    GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN.code,
                    GenerationPath.UNION_TEMPLATE_WIDEN.code,
                ),
            ),
            category="generation_path",
        )
        run_and_assert(runner, scenario, header="[GP4_2-BOTH-WIDEN]")


@pytest.mark.live
def test_gp4_3_runtime_subset_of_template(schema, schema_terms, t2s) -> None:
    """Seed a wider template; asking for a strict subset of its columns routes to ``4.3``."""

    with (
        seeded_runner(schema, schema_terms, t2s, label="gp4_3", kits=("baseline_templates",)) as runner,
        deterministic_join_choice_patch(),
    ):
        scenario = Scenario(
            id="GP4_3-RUNTIME-SUBSET",
            question="list customer first names only",
            expected=Expected(
                generation_path_in=(
                    GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE.code,
                    GenerationPath.INTENT_DIRECT_MATCH.code,
                    GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN.code,
                ),
            ),
            category="generation_path",
        )
        run_and_assert(runner, scenario, header="[GP4_3-RUNTIME-SUBSET]")


@pytest.mark.live
def test_gp5_fresh_generation(schema, schema_terms, t2s) -> None:
    """Empty store forces a fresh generation for every new question (path ``5``)."""

    with isolated_runner(schema, schema_terms, t2s, label="gp5") as runner:
        scenario = Scenario(
            id="GP5-FRESH",
            question="list film title rental rate and replacement cost for films limit fifty rows",
            expected=Expected(
                generation_path=GenerationPath.FRESH.code,
                tables=["film"],
                min_rows=1,
            ),
            category="generation_path",
        )
        run_and_assert(runner, scenario, header="[GP5-FRESH]")
