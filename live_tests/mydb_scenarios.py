"""
DVD Rental database scenario definitions for live pipeline testing.

Defines scenarios across multiple categories against the PostgreSQL ``dvdrental_new`` schema (15 tables: film with ``special_features`` as ``TEXT[]``, language, category, film_category, actor, film_actor, inventory, rental, payment, customer, staff, store, address, city, country). Each function returns a list of ``Scenario`` or ``SequenceScenario`` objects grouped by test category. ``generation_path_sequences`` holds the multi-step ``GenerationPath`` live sequences used by ``test_generation_paths_live``.
"""

from __future__ import annotations

from aetherdialect._config import GenerationPath
from aetherdialect._live_testing import Expected, Scenario, SequenceScenario


def single_table_scenarios() -> list[Scenario]:
    """Basic single-table queries with no joins."""
    return [
        Scenario(
            id="ST-001",
            question="list all film titles",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=2000,
                contains_join=False,
                grain="row_level",
                min_confidence=0.5,
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-002",
            question="show me all customer first names and last names",
            expected=Expected(tables=["customer"], min_rows=1, contains_join=False),
            category="single_table",
        ),
        Scenario(
            id="ST-003",
            question="list the distinct film ratings in the catalog",
            expected=Expected(tables=["film"], min_rows=1, max_rows=10),
            category="single_table",
        ),
        Scenario(
            id="ST-004",
            question="list all categories",
            expected=Expected(tables=["category"], min_rows=1, max_rows=20, contains_join=False),
            category="single_table",
        ),
        Scenario(
            id="ST-005",
            question="how many films are there",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                min_confidence=0.4,
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-006",
            question="show all actor first names and last names",
            expected=Expected(tables=["actor"], min_rows=1, contains_join=False),
            category="single_table",
        ),
        Scenario(
            id="ST-007",
            question="list the distinct languages in the catalog",
            expected=Expected(tables=["language"], min_rows=1, max_rows=10, contains_join=False),
            category="single_table",
        ),
        Scenario(
            id="ST-008",
            question="list all store ids",
            expected=Expected(tables=["store"], min_rows=1, max_rows=5, contains_join=False),
            category="single_table",
        ),
    ]


def multi_table_scenarios() -> list[Scenario]:
    """Multi-table join queries."""
    return [
        Scenario(
            id="MT-001",
            question="list all films and their language",
            expected=Expected(tables=["film", "language"], contains_join=True, min_rows=1),
            category="multi_table",
        ),
        Scenario(
            id="MT-002",
            question="show all films with their categories",
            expected=Expected(
                tables=["film", "category"],
                contains_join=True,
                min_rows=1,
                sql_contains=["film_category", "category"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-003",
            question="list all actors and the films they appeared in",
            expected=Expected(
                tables=["actor", "film"],
                contains_join=True,
                min_rows=1,
                sql_contains=["film_actor"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-004",
            question="show customer names with their city",
            expected=Expected(
                tables=["customer", "city"],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-005",
            question="list customer names and their country",
            expected=Expected(
                tables=["customer", "country"],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-006",
            question="show all rentals with customer first name and film title",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-007",
            question="list all payments with the customer name and staff name",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-008",
            question="show inventory count per store for each film",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-009",
            question="list films that are in the action category",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["action"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-010",
            question="show films in English",
            expected=Expected(
                tables=["film", "language"],
                contains_join=True,
                min_rows=1,
                sql_contains=["English", "language"],
            ),
            category="multi_table",
        ),
    ]


def aggregation_scenarios() -> list[Scenario]:
    """Aggregation and GROUP BY queries."""
    return [
        Scenario(
            id="AG-001",
            question="how many films are in each category",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-002",
            question="total payment amount by customer",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                max_rows=1000,
                min_confidence=0.25,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-003",
            question="average rental duration per film rating",
            expected=Expected(
                tables_one_of=[["film"], ["film", "rental"]],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-004",
            question="how many customers are in each city",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-005",
            question="what is the total number of rentals",
            expected=Expected(
                tables=["rental"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-006",
            question="maximum replacement cost of films by rating",
            expected=Expected(
                tables=["film"],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["MAX"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-007",
            question="minimum payment amount per customer",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["MIN"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-008",
            question="count of films per language",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-009",
            question="total revenue per store",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
    ]


def filtering_scenarios() -> list[Scenario]:
    """Filter and parameterization queries."""
    return [
        Scenario(
            id="FI-001",
            question="list all films with rating PG-13",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["PG-13"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-002",
            question="show customers from the city of London",
            expected=Expected(
                contains_join=True,
                sql_contains=["London"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-003",
            question="list films with a rental rate greater than 2.99",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["2.99"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-004",
            question="how many films have a length greater than 120 minutes",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["120"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-005",
            question="show all payments greater than 5 dollars",
            expected=Expected(
                tables_one_of=[
                    ["payment"],
                    ["payment", "staff"],
                    ["customer", "payment"],
                    ["customer", "payment", "staff"],
                ],
                min_rows=1,
                sql_contains=["5"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-006",
            question="list films released in 2006",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["2006"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-007",
            question="show the top 10 customers by total payment amount",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
                max_rows=10,
                sql_contains=["LIMIT"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-008",
            question="list all R rated films with replacement cost above 20",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["R", "20"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-009",
            question="show rentals from July 2005",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["2005"],
            ),
            category="filtering",
        ),
    ]


def cte_scenarios() -> list[Scenario]:
    """CTE handling and validation."""
    return [
        Scenario(
            id="CT-001",
            question="show the top 5 customers by total payment and their city",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=5,
                sql_contains=["customer", "city"],
            ),
            category="cte",
        ),
        Scenario(
            id="CT-003",
            question="list customers who have rented more than 30 films",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-004",
            question="what is the average payment per rental for each customer",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-005",
            question="show categories where the average film length is above 120 minutes",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-006",
            question="list actors who appeared in more than 30 films along with the count",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-007",
            question="show the total revenue by category",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="cte",
        ),
    ]


def template_reuse_scenarios() -> list[Scenario]:
    """
    Template reuse and trust building.

    These should be run after the single-table or multi-table tests have populated the template store with at least one accepted template.
    """
    return [
        Scenario(
            id="TR-001",
            question="list all film titles",
            expected=Expected(
                min_rows=1,
            ),
            category="template_reuse",
        ),
        Scenario(
            id="TR-002",
            question="list all film titles and their rating",
            expected=Expected(
                tables=["film"],
                min_rows=1,
            ),
            category="template_reuse",
        ),
        Scenario(
            id="TR-003",
            question="how many films are in each category",
            expected=Expected(
                min_rows=1,
            ),
            category="template_reuse",
        ),
        Scenario(
            id="TR-004",
            question="list all films with rating R",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["R"],
            ),
            category="template_reuse",
        ),
        Scenario(
            id="TR-005",
            question="show the top 5 customers by total payment amount",
            expected=Expected(
                min_rows=1,
                max_rows=5,
            ),
            category="template_reuse",
        ),
    ]


def schema_edge_scenarios() -> list[Scenario]:
    """Schema edge cases: bridge tables, long join chains, ambiguous columns."""
    return [
        Scenario(
            id="SE-001",
            question="list all actors in the comedy category",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-002",
            question="show the country for each customer",
            expected=Expected(
                tables=["customer", "country"],
                contains_join=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-003",
            question="list all films with their actors and categories",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-004",
            question="show the store address and city for each store",
            expected=Expected(
                tables=["store", "address", "city"],
                contains_join=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-005",
            question="list staff members with their store city",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-006",
            question="show films that have never been rented",
            expected=Expected(
                min_rows=0,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-007",
            question="list the total number of films per actor per category",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-009",
            question="list films that are available in exactly 2 stores",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["COUNT"],
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-010",
            question="show the district for each customer",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["district"],
            ),
            category="schema_edge",
        ),
        Scenario(
            id="SE-011",
            question="show total payments per customer",
            expected=Expected(
                tables=["customer", "payment"],
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="schema_edge",
        ),
    ]


def negative_scenarios() -> list[Scenario]:
    """Negative and forbidden SQL pattern tests."""
    return [
        Scenario(
            id="NG-001",
            question="delete all customers",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-002",
            question="update the film table set rating to R",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-003",
            question="drop table film",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-004",
            question="",
            expected=Expected(status="invalid_question"),
            auto_responses=[],
            category="negative",
        ),
        Scenario(
            id="NG-005",
            question="insert into film values (1, 'test', 'test')",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-006",
            question="what is the meaning of life",
            expected=Expected(status="invalid_question"),
            category="negative",
        ),
        Scenario(
            id="NG-007",
            question="tell me a joke about databases",
            expected=Expected(status="invalid_question"),
            category="negative",
        ),
        Scenario(
            id="NG-008",
            question="truncate the payment table",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-009",
            question="alter table film add column test varchar(100)",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-010",
            question="create index idx_test on film(title)",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-011",
            question="grant all privileges on film to public",
            expected=Expected(status="restricted"),
            category="negative",
        ),
        Scenario(
            id="NG-012",
            question="how is the weather today",
            expected=Expected(status="invalid_question"),
            category="negative",
        ),
    ]


def repair_loop_scenarios() -> list[Scenario]:
    """
    Repair loop and retry behaviour tests.

    These scenarios target complex queries that may trigger the SQL repair loop, testing that the pipeline can self-correct and still produce valid output.
    """
    return [
        Scenario(
            id="RL-001",
            question="show the top 3 categories by total rental count with the average payment per rental",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                max_rows=3,
            ),
            category="repair_loop",
        ),
        Scenario(
            id="RL-002",
            question="show the top 5 customers by total payment amount",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=5,
            ),
            category="repair_loop",
        ),
        Scenario(
            id="RL-003",
            question="list the 5 least rented films with their category and total revenue",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=5,
            ),
            category="repair_loop",
        ),
        Scenario(
            id="RL-004",
            question="show categories where total revenue exceeds 4000 with the film count",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="repair_loop",
        ),
    ]


def confidence_scenarios() -> list[Scenario]:
    """Confidence scoring tests with expected minimum thresholds."""
    return [
        Scenario(
            id="CF-004",
            question="how many customers are in each country",
            expected=Expected(
                min_confidence=0.25,
                contains_join=True,
                min_rows=1,
            ),
            category="confidence",
        ),
        Scenario(
            id="CF-005",
            question="what is the average film length",
            expected=Expected(
                min_confidence=0.4,
                min_rows=1,
                max_rows=1,
            ),
            category="confidence",
        ),
        Scenario(
            id="CF-006",
            question="list all customer email addresses",
            expected=Expected(
                min_confidence=0.4,
                min_rows=1,
            ),
            category="confidence",
        ),
    ]


def stateful_scenarios() -> list[SequenceScenario]:
    """Stateful sequence scenarios testing template store evolution."""
    return [
        SequenceScenario(
            id="SQ-001",
            category="stateful",
            steps=[
                Scenario(
                    id="SQ-001-A",
                    question="list all film titles and ratings",
                    expected=Expected(tables=["film"], min_rows=1),
                    category="stateful",
                ),
                Scenario(
                    id="SQ-001-B",
                    question="list all film titles and ratings",
                    expected=Expected(
                        reuse_type=(
                            "direct_reuse",
                            "intent_direct_reuse",
                            "intent_reuse",
                        ),
                        min_rows=1,
                    ),
                    category="stateful",
                ),
            ],
        ),
        SequenceScenario(
            id="SQ-002",
            category="stateful",
            steps=[
                Scenario(
                    id="SQ-002-A",
                    question="total payment amount by customer",
                    expected=Expected(
                        contains_group_by=True,
                        min_rows=1,
                    ),
                    category="stateful",
                ),
                Scenario(
                    id="SQ-002-B",
                    question="total payment amount by staff",
                    expected=Expected(
                        contains_group_by=True,
                        min_rows=1,
                    ),
                    category="stateful",
                ),
            ],
        ),
        SequenceScenario(
            id="SQ-003",
            category="stateful",
            steps=[
                Scenario(
                    id="SQ-003-A",
                    question="how many films are in the action category",
                    expected=Expected(min_rows=1, max_rows=1),
                    category="stateful",
                    feedback="y",
                ),
                Scenario(
                    id="SQ-003-B",
                    question="how many films are in the comedy category",
                    expected=Expected(min_rows=1, max_rows=1, sql_contains=["comedy"]),
                    category="stateful",
                    feedback="y",
                ),
                Scenario(
                    id="SQ-003-C",
                    question="how many films are in the horror category",
                    expected=Expected(min_rows=1, max_rows=1, sql_contains=["horror"]),
                    category="stateful",
                    feedback="y",
                ),
            ],
        ),
        SequenceScenario(
            id="SQ-004",
            category="stateful",
            steps=[
                Scenario(
                    id="SQ-004-A",
                    question="show the top 10 customers by total payment amount",
                    expected=Expected(min_rows=1, max_rows=10),
                    category="stateful",
                    feedback="y",
                ),
                Scenario(
                    id="SQ-004-B",
                    question="show the top 5 customers by total payment amount",
                    expected=Expected(min_rows=1, max_rows=5),
                    category="stateful",
                    feedback="y",
                ),
            ],
        ),
    ]


def trust_cycle_scenarios() -> list[SequenceScenario]:
    """Full trust lifecycle: build trust, erode it, reach hard block, and override."""
    return [
        SequenceScenario(
            id="TC-001",
            category="trust_cycle",
            steps=[
                Scenario(
                    id="TC-001-A",
                    question="show all customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="y",
                ),
                Scenario(
                    id="TC-001-B",
                    question="list all customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="incorrect results",
                ),
                Scenario(
                    id="TC-001-C",
                    question="display all customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="y",
                ),
                Scenario(
                    id="TC-001-D",
                    question="show every customer name",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="y",
                ),
                Scenario(
                    id="TC-001-E",
                    question="list every customer name",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="incorrect results",
                ),
                Scenario(
                    id="TC-001-F",
                    question="customer names please",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="incorrect results",
                ),
                Scenario(
                    id="TC-001-G",
                    question="all customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="incorrect results",
                ),
                Scenario(
                    id="TC-001-H",
                    question="show customer name list",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="wrong columns selected",
                ),
                Scenario(
                    id="TC-001-I",
                    question="list customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="n",
                    reject_reason="wrong columns selected",
                ),
                Scenario(
                    id="TC-001-J",
                    question="show all customer names",
                    expected=Expected(
                        tables=["customer"],
                        min_rows=1,
                    ),
                    category="trust_cycle",
                    feedback="y",
                    auto_responses=["y", "y"],
                ),
            ],
        ),
    ]


def rejection_feedback_scenarios() -> list[Scenario]:
    """Rejection feedback loop scenarios — user says no and provides a reason."""
    return [
        Scenario(
            id="RJ-001",
            question="how many films are there",
            expected=Expected(tables=["film"]),
            category="rejection",
            feedback="n",
            reject_reason="wrong columns selected",
        ),
        Scenario(
            id="RJ-002",
            question="list all customer names",
            expected=Expected(tables=["customer"]),
            category="rejection",
            feedback="n",
            reject_reason="too many rows",
        ),
        Scenario(
            id="RJ-003",
            question="total payment amount by customer",
            expected=Expected(contains_group_by=True),
            category="rejection",
            feedback="n",
            reject_reason="wrong intent",
        ),
    ]


def performance_scenarios() -> list[Scenario]:
    """
    Performance and cost-awareness scenarios.

    These test that the pipeline completes within a reasonable time and doesn't produce excessively large result sets.
    """
    return [
        Scenario(
            id="PF-002",
            question="how many rentals are there",
            expected=Expected(min_rows=1, max_rows=1),
            category="performance",
        ),
        Scenario(
            id="PF-004",
            question="average film length by rating",
            expected=Expected(min_rows=1, max_rows=10),
            category="performance",
        ),
        Scenario(
            id="PF-005",
            question="list all actors and the number of films they appeared in",
            expected=Expected(min_rows=1, max_rows=500),
            category="performance",
        ),
    ]


def having_scenarios() -> list[Scenario]:
    """HAVING clause queries that filter grouped results."""
    return [
        Scenario(
            id="HV-001",
            question="which categories have more than 50 films",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-002",
            question="show customers who have spent more than 100 total",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-004",
            question="list ratings that have an average rental rate above 3",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-005",
            question="show categories where average rental rate is above 3 or total number of films is more than 60",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                sql_contains=["HAVING", "OR"],
                min_rows=1,
            ),
            category="having",
        ),
    ]


def sql_vs_intent_scenarios() -> list[Scenario]:
    """SQL-vs-intent structural consistency checks."""
    return [
        Scenario(
            id="SI-001",
            question="list all film titles",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                contains_join=False,
                grain="row_level",
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-002",
            question="how many films are in each category",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-003",
            question="list films ordered by length descending",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-004",
            question="show the top 10 longest films",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY", "LIMIT"],
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-005",
            question="show ratings with more than 200 films",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-006",
            question="show film title and replacement cost minus rental rate as profit margin",
            expected=Expected(
                tables=["film"],
                min_rows=1,
            ),
            category="sql_vs_intent",
        ),
    ]


def subquery_scenarios() -> list[Scenario]:
    """Subquery patterns including NOT IN, EXISTS, and correlated subqueries."""
    return [
        Scenario(
            id="SB-001",
            question="list films that have never been rented",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-002",
            question="show customers who have never made a payment",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-003",
            question="list actors who have appeared in more films than average",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-004",
            question="which films have a rental rate higher than the average rental rate",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-005",
            question="show categories with fewer films than the average films per category",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-006",
            question="list customers who have rented films from both store 1 and store 2",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
    ]


def cte_join_scenarios() -> list[Scenario]:
    """CTE queries that also require joins across multiple tables."""
    return [
        Scenario(
            id="CJ-002",
            question="highest grossing film per category",
            expected=Expected(
                min_rows=1,
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-003",
            question="for each store show the top 3 customers by number of rentals",
            expected=Expected(
                min_rows=1,
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-004",
            question=(
                "first in a CTE count rentals per film_id, then join that CTE to film "
                "and list each film title with its rental count"
            ),
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["WITH", "rental", "film"],
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-005",
            question=(
                "use one CTE to get distinct inventory ids that were rented, "
                "then a second CTE joining that to inventory and film to list film titles, "
                "then select from the second CTE"
            ),
            expected=Expected(
                min_rows=1,
                sql_contains=["WITH"],
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-006",
            question=(
                "first CTE: from rental table group by customer_id and count rows as rental_cnt; "
                "second CTE: from payment table group by customer_id and sum amount as pay_sum; "
                "final select joining those two CTEs on customer_id showing customer_id, rental_cnt, and pay_sum"
            ),
            expected=Expected(
                min_rows=1,
                sql_contains=["WITH", "rental", "payment"],
            ),
            category="cte_join",
        ),
    ]


def join_validation_scenarios() -> list[Scenario]:
    """Join path and join-candidate validation scenarios."""
    return [
        Scenario(
            id="JV-001",
            question="show each customer and their store id",
            expected=Expected(
                min_rows=1,
                sql_contains_one_of=[["customer.store_id"], ["store.store_id"]],
            ),
            category="join_validation",
        ),
        Scenario(
            id="JV-004",
            question="show number of rentals for each film",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["inventory", "rental", "COUNT"],
            ),
            category="join_validation",
        ),
        Scenario(
            id="JV-005",
            question="show the manager name for each store",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=5,
                sql_contains=["store", "staff"],
            ),
            category="join_validation",
        ),
        Scenario(
            id="JV-006",
            question="show number of rentals per customer per store",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="join_validation",
        ),
    ]


def distinct_scenarios() -> list[Scenario]:
    """DISTINCT keyword queries."""
    return [
        Scenario(
            id="DT-001",
            question="list distinct cities where customers live",
            expected=Expected(
                min_rows=1,
            ),
            category="distinct",
        ),
        Scenario(
            id="DT-002",
            question="show all unique film ratings",
            expected=Expected(
                sql_contains=["DISTINCT"],
                min_rows=1,
            ),
            category="distinct",
        ),
        Scenario(
            id="DT-003",
            question="list distinct last names of actors",
            expected=Expected(
                sql_contains=["DISTINCT"],
                min_rows=1,
            ),
            category="distinct",
        ),
    ]


def multi_agg_scenarios() -> list[Scenario]:
    """Queries requiring multiple aggregate functions in a single SELECT."""
    return [
        Scenario(
            id="MA-001",
            question="show total payment and rental count per customer",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="multi_agg",
        ),
        Scenario(
            id="MA-002",
            question="average and maximum film length per rating",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="multi_agg",
        ),
        Scenario(
            id="MA-003",
            question="for each category show the count of films and average rental rate",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="multi_agg",
        ),
        Scenario(
            id="MA-004",
            question="minimum maximum and average replacement cost per rating",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="multi_agg",
        ),
    ]


def order_by_scenarios() -> list[Scenario]:
    """ORDER BY and sorting queries."""
    return [
        Scenario(
            id="OB-001",
            question="list all films ordered by title",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-002",
            question="show the 5 longest films",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=5,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-003",
            question="list the 10 most expensive films by replacement cost",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-004",
            question="show customers ordered by last name alphabetically",
            expected=Expected(
                tables=["customer"],
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-005",
            question="list the 3 most recently created customers",
            expected=Expected(
                tables=["customer"],
                min_rows=1,
                max_rows=3,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-006",
            question="show categories by film count descending",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["ORDER BY"],
                min_rows=1,
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-007",
            question="list the 5 cheapest films by rental rate",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=5,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
    ]


def like_pattern_scenarios() -> list[Scenario]:
    """LIKE and pattern matching queries."""
    return [
        Scenario(
            id="LK-001",
            question="list films with titles starting with 'A'",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["LIKE"],
            ),
            category="like_pattern",
        ),
        Scenario(
            id="LK-002",
            question="show customers whose last name starts with 'S'",
            expected=Expected(
                tables=["customer"],
                min_rows=1,
                sql_contains=["LIKE"],
            ),
            category="like_pattern",
        ),
        Scenario(
            id="LK-003",
            question="find actors whose first name contains 'an'",
            expected=Expected(
                tables=["actor"],
                min_rows=1,
                sql_contains=["LIKE"],
            ),
            category="like_pattern",
        ),
        Scenario(
            id="LK-004",
            question="show films with the word 'dog' in the description",
            expected=Expected(
                tables=["film"],
                min_rows=0,
                sql_contains=["LIKE"],
            ),
            category="like_pattern",
        ),
        Scenario(
            id="LK-005",
            question="list categories whose name starts with 'C'",
            expected=Expected(
                tables=["category"],
                min_rows=1,
                sql_contains=["LIKE"],
            ),
            category="like_pattern",
        ),
    ]


def null_filter_scenarios() -> list[Scenario]:
    """IS NULL and IS NOT NULL filter queries."""
    return [
        Scenario(
            id="NL-001",
            question="show all rentals that have not been returned",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                    ["inventory", "rental"],
                    ["customer", "inventory", "rental"],
                    ["customer", "inventory", "rental", "staff"],
                    ["inventory", "rental", "staff"],
                ],
                min_rows=0,
                sql_contains=["NULL"],
            ),
            category="null_filter",
        ),
        Scenario(
            id="NL-002",
            question="list customers with no email address",
            expected=Expected(
                tables=["customer"],
                min_rows=0,
                sql_contains=["NULL"],
            ),
            category="null_filter",
        ),
        Scenario(
            id="NL-003",
            question="show rentals that have been returned",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                    ["inventory", "rental"],
                    ["customer", "inventory", "rental"],
                    ["customer", "inventory", "rental", "staff"],
                    ["inventory", "rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["NOT NULL"],
            ),
            category="null_filter",
        ),
        Scenario(
            id="NL-004",
            question="list addresses with no postal code",
            expected=Expected(
                tables_one_of=[["address"], ["address", "city"]],
                min_rows=0,
                sql_contains=["NULL"],
            ),
            category="null_filter",
        ),
    ]


def count_distinct_scenarios() -> list[Scenario]:
    """COUNT DISTINCT and unique counting queries."""
    return [
        Scenario(
            id="CD-001",
            question="how many distinct customers have rented a film",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="count_distinct",
        ),
        Scenario(
            id="CD-002",
            question="count of unique films rented per store",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="count_distinct",
        ),
        Scenario(
            id="CD-003",
            question="how many different categories of films are there",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="count_distinct",
        ),
        Scenario(
            id="CD-004",
            question="number of distinct cities where customers live",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="count_distinct",
        ),
    ]


def compound_filter_scenarios() -> list[Scenario]:
    """Multi-condition WHERE clauses (implicit AND, same-column IN)."""
    return [
        Scenario(
            id="CF2-001",
            question="list PG-13 films with rental rate above 3 and length over 100",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["PG-13"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-003",
            question="list films rated R or NC-17 with replacement cost below 15",
            expected=Expected(
                tables=["film"],
                min_rows=1,
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-005",
            question="list films in English with rental duration greater than 5 days",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-006",
            question="show active customers who live in a city that starts with 'A'",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["LIKE"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-007",
            question="show films with rental rate above 3 or length under 60",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-008",
            question="films rated PG-13 with length over 120 or rated R with length over 150",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-009",
            question=(
                "list films where (rating is PG-13 and length under 90) or (rating is G and rental_rate above 2)"
            ),
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
    ]


def date_range_scenarios() -> list[Scenario]:
    """Date range and temporal filter queries."""
    return [
        Scenario(
            id="DR-005",
            question="show rentals on July 15 2005",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                ],
                min_rows=0,
                sql_contains=["2005"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-001",
            question="show all rentals between July 1 2005 and July 31 2005",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["2005"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-002",
            question="total payments collected in August 2005",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["2005"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-003",
            question="count of rentals per month",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-004",
            question="how many payments were made after February 2007",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["2007"],
            ),
            category="date_range",
        ),
    ]


def agg_filter_join_scenarios() -> list[Scenario]:
    """Compound queries combining aggregation, joins, and filters."""
    return [
        Scenario(
            id="AJ-001",
            question="total revenue from PG-13 films",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["SUM", "PG-13"],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-002",
            question="average payment amount per customer for customers from store 1",
            expected=Expected(
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-003",
            question="count of rentals per category for the action category",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["action"],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-004",
            question="average film length by category for categories with more than 60 films",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-005",
            question="total number of rentals for R rated films",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["R"],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-006",
            question="total revenue per country",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-007",
            question="list films with no rating",
            expected=Expected(
                tables=["film"],
                min_rows=0,
                sql_contains_one_of=[["IS NULL", "is null"], ["rating"]],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-008",
            question="list payments with amount greater than 5",
            expected=Expected(
                min_rows=1,
                sql_contains_one_of=[["5"], ["5.0"]],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-009",
            question="list films with rating PG-13",
            expected=Expected(
                min_rows=1,
                sql_contains=["PG-13"],
            ),
            category="agg_filter_join",
        ),
        Scenario(
            id="AJ-010",
            question="for each language show the top 3 longest films by length",
            expected=Expected(
                contains_cte=True,
                min_rows=1,
                sql_contains_one_of=[["ROW_NUMBER", "RANK", "DENSE_RANK"], ["PARTITION"]],
            ),
            category="agg_filter_join",
        ),
    ]


def boolean_filter_scenarios() -> list[Scenario]:
    """Boolean and active/inactive flag filtering queries."""
    return [
        Scenario(
            id="BF-001",
            question="list all active customers",
            expected=Expected(tables=["customer"], min_rows=1, sql_contains=["active"]),
            category="boolean_filter",
        ),
        Scenario(
            id="BF-002",
            question="show inactive customers",
            expected=Expected(tables=["customer"], min_rows=0, sql_contains=["active"]),
            category="boolean_filter",
        ),
        Scenario(
            id="BF-003",
            question="count active customers by store",
            expected=Expected(
                contains_group_by=True,
                min_rows=2,
                sql_contains=["active"],
            ),
            category="boolean_filter",
        ),
    ]


def scalar_func_scenarios() -> list[Scenario]:
    """Scalar function usage in SELECT, WHERE, ORDER BY."""
    return [
        Scenario(
            id="SF-001",
            question="show uppercase film titles",
            expected=Expected(tables=["film"], min_rows=1),
            category="scalar_func",
        ),
        Scenario(
            id="SF-002",
            question="list film titles and their length in hours",
            expected=Expected(tables=["film"], min_rows=1),
            category="scalar_func",
        ),
        Scenario(
            id="SF-003",
            question="show the year from the last rental date of each customer",
            expected=Expected(min_rows=1, contains_group_by=True),
            category="scalar_func",
        ),
    ]


def expr_select_scenarios() -> list[Scenario]:
    """Expression-based SELECT columns (arithmetic, concatenation)."""
    return [
        Scenario(
            id="ES-001",
            question="show film title and replacement cost minus rental rate as profit margin",
            expected=Expected(tables=["film"], min_rows=1),
            category="expr_select",
        ),
        Scenario(
            id="ES-002",
            question="list the total amount per customer and the average payment per rental",
            expected=Expected(contains_group_by=True, min_rows=1),
            category="expr_select",
        ),
    ]


def in_list_scenarios() -> list[Scenario]:
    """IN / NOT IN list filtering queries."""
    return [
        Scenario(
            id="IL-001",
            question="list films rated PG or PG-13",
            expected=Expected(tables=["film"], min_rows=1, sql_contains=["IN"]),
            category="in_list",
        ),
        Scenario(
            id="IL-002",
            question="show customers who are not in store 1 or store 2",
            expected=Expected(tables=["customer"], min_rows=0),
            category="in_list",
        ),
        Scenario(
            id="IL-003",
            question="films in action or comedy categories",
            expected=Expected(contains_join=True, min_rows=1),
            category="in_list",
        ),
    ]


def bridge_join_scenarios() -> list[Scenario]:
    """Bridge table (many-to-many) join queries."""
    return [
        Scenario(
            id="BJ-001",
            question="how many films is each actor in",
            expected=Expected(
                tables=["actor", "film_actor"],
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
            ),
            category="bridge_join",
        ),
        Scenario(
            id="BJ-002",
            question="show actors who appeared in at least 30 films",
            expected=Expected(
                tables=["actor", "film_actor"],
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["HAVING"],
            ),
            category="bridge_join",
        ),
        Scenario(
            id="BJ-003",
            question="show actors who appeared in more than 30 films",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["HAVING"],
            ),
            category="bridge_join",
        ),
    ]


def date_arithmetic_scenarios() -> list[Scenario]:
    """Date arithmetic and interval-based queries."""
    return [
        Scenario(
            id="DA-001",
            question="show rentals from the last 90 days",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                    ["inventory", "rental"],
                    ["customer", "inventory", "rental"],
                    ["customer", "inventory", "rental", "staff"],
                    ["inventory", "rental", "staff"],
                ],
                min_rows=0,
            ),
            category="date_arithmetic",
        ),
        Scenario(
            id="DA-002",
            question="average number of days between rental and return per customer",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="date_arithmetic",
        ),
        Scenario(
            id="DA-003",
            question="list rentals where the return was more than 7 days after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                    ["inventory", "rental"],
                    ["customer", "inventory", "rental"],
                    ["customer", "inventory", "rental", "staff"],
                    ["inventory", "rental", "staff"],
                ],
                min_rows=0,
            ),
            category="date_arithmetic",
        ),
    ]


def validation_failure_scenarios() -> list[Scenario]:
    """Scenarios where validation may fail; accept ok or validation_failed."""
    return [
        Scenario(
            id="VF-001",
            question="show customer first name and total of all payments",
            expected=Expected(
                status_in=("ok", "validation_failed"),
                tables=["customer", "payment"],
            ),
            category="validation_failure",
        ),
    ]


def intent_rejected_scenarios() -> list[Scenario]:
    """Scenarios where user declines intent confirmation."""
    return [
        Scenario(
            id="IR-001",
            question="For each store, list the sum of payment amounts in the last 30 days and the store address",
            expected=Expected(status="intent_rejected"),
            category="intent_rejected",
            auto_responses=["n"],
        ),
    ]


def column_names_scenarios() -> list[Scenario]:
    """Scenarios asserting expected output column headers."""
    return [
        Scenario(
            id="CN-001",
            question="list all film titles",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                column_names_one_of=[["film_id", "title"], ["title"]],
            ),
            category="column_names",
        ),
        Scenario(
            id="CN-002",
            question="show customer first name and last name",
            expected=Expected(
                tables=["customer"],
                min_rows=1,
                column_names_one_of=[
                    ["customer_id", "first_name", "last_name"],
                    ["first_name", "last_name"],
                ],
            ),
            category="column_names",
        ),
    ]


def row_value_check_scenarios() -> list[Scenario]:
    """Scenarios with custom row-value assertions."""
    return [
        Scenario(
            id="RV-001",
            question="how many films are there",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                row_value_check=lambda rows: len(rows) == 1 and isinstance(rows[0][0], (int, float)),
            ),
            category="row_value_check",
        ),
        Scenario(
            id="RV-002",
            question="how many films are in the action category",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                row_value_check=lambda rows: (
                    len(rows) == 1 and isinstance(rows[0][0], (int, float)) and rows[0][0] >= 0
                ),
            ),
            category="row_value_check",
        ),
    ]


def sql_excludes_scenarios() -> list[Scenario]:
    """Scenarios asserting forbidden SQL patterns."""
    return [
        Scenario(
            id="EX-001",
            question="list all film titles",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_excludes=["DELETE", "UPDATE", "INSERT", "DROP"],
            ),
            category="sql_excludes",
        ),
        Scenario(
            id="EX-002",
            question="list all categories",
            expected=Expected(
                tables=["category"],
                min_rows=1,
                sql_excludes=["JOIN"],
            ),
            category="sql_excludes",
        ),
    ]


def window_function_scenarios() -> list[Scenario]:
    """Live scenarios that should produce window functions (OVER clause)."""
    return [
        Scenario(
            id="WF-001",
            question="rank films by length descending using row number",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-002",
            question="for each film rating, list film titles with a row number ordered by length descending",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OVER (", "PARTITION BY"],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-003",
            question="for each film show title, length, and the average length of films with the same rating",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains_one_of=[
                    ["OVER (", "PARTITION BY"],
                    ["GROUP BY", "AVG("],
                ],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-004",
            question="rank films by rental rate highest first using rank with ties allowed",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-005",
            question="list each rental with a row number for that customer ordered by rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["OVER (", "PARTITION BY"],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-006",
            question="list payment id, amount, and running total of amount ordered by payment date",
            expected=Expected(
                tables=["payment"],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-007",
            question="dense rank actors by actor id when ordered by actor id",
            expected=Expected(
                tables=["actor"],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-008",
            question="for each customer show customer id and previous payment amount ordered by payment date",
            expected=Expected(
                tables_one_of=[
                    ["payment"],
                    ["customer", "payment"],
                ],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-009",
            question="list rentals with rental id and next rental date for the same inventory item ordered by rental date",
            expected=Expected(
                tables=["rental"],
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
    ]


def case_when_scenarios() -> list[Scenario]:
    """Live scenarios that should produce CASE expressions in SELECT."""
    return [
        Scenario(
            id="CW-001",
            question="list film titles with a label column that is premium when rental_rate is greater than 3 otherwise standard",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["CASE"],
            ),
            category="case_when",
        ),
        Scenario(
            id="CW-002",
            question=(
                "list each film title with a length bucket: "
                "when length is greater than 150 show long, "
                "when length is at least 100 and at most 150 show medium, "
                "otherwise show short"
            ),
            expected=Expected(
                tables=["film"],
                min_rows=1,
                sql_contains=["CASE", "WHEN"],
            ),
            category="case_when",
        ),
    ]


def restrictions_scenarios() -> list[Scenario]:
    """Restricted SQL forms such as window functions, UNION, and correlated subqueries."""
    return [
        Scenario(
            id="RS-001",
            question="for each film show its rank ordered by length descending",
            expected=Expected(
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="restrictions",
        ),
        Scenario(
            id="RS-002",
            question="show films with above average length",
            expected=Expected(
                min_rows=1,
            ),
            category="restrictions",
        ),
        Scenario(
            id="RS-003",
            question="combine action and comedy films into a single list",
            expected=Expected(
                min_rows=1,
                sql_excludes=["UNION"],
            ),
            category="restrictions",
        ),
        Scenario(
            id="RS-004",
            question="show customers who have more rentals than the average rentals per customer",
            expected=Expected(
                min_rows=1,
                sql_excludes=["EXISTS"],
            ),
            category="restrictions",
        ),
    ]


def ast_explain_scenarios() -> list[Scenario]:
    """Scenarios that exercise AST and EXPLAIN-based validation."""
    return [
        Scenario(
            id="AE-001",
            question="list all film titles",
            expected=Expected(
                tables=["film"],
                min_rows=1,
            ),
            category="ast_explain",
        ),
        Scenario(
            id="AE-002",
            question="list films and their language",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="ast_explain",
        ),
        Scenario(
            id="AE-003",
            question="first compute total payments per customer, then show the top 10 customers by total payments",
            expected=Expected(
                min_rows=1,
                max_rows=10,
            ),
            category="ast_explain",
        ),
    ]


def semantic_warnings_scenarios() -> list[Scenario]:
    """Scenarios where semantic warnings may be emitted during intent parsing. min_semantic_warnings exercises the assertion path."""
    return [
        Scenario(
            id="SW-001",
            question=(
                "list each category name with how many distinct films it has where that count is greater than ten"
            ),
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                min_rows=1,
            ),
            category="semantic_warnings",
        ),
    ]


def template_reuse_sequence_scenarios() -> list[SequenceScenario]:
    """Sequence scenarios ensuring template reuse by running single-table before template-reuse scenarios."""
    return [
        SequenceScenario(
            id="TR-SEQ-001",
            category="template_reuse_sequence",
            steps=[
                Scenario(
                    id="TR-SEQ-001-A",
                    question="list all film titles",
                    expected=Expected(
                        tables=["film"],
                        min_rows=1,
                        contains_join=False,
                    ),
                    category="template_reuse_sequence",
                ),
                Scenario(
                    id="TR-SEQ-001-B",
                    question="list all film titles",
                    expected=Expected(
                        reuse_type=("direct_reuse", "intent_direct_reuse"),
                        generation_path=GenerationPath.EXACT_QUESTION_REUSE,
                        min_rows=1,
                    ),
                    category="template_reuse_sequence",
                ),
            ],
        ),
    ]


def multi_cte_chain_scenarios() -> list[Scenario]:
    """Multi-step CTE chain queries."""
    return [
        Scenario(
            id="MC-001",
            question="first find total payments per customer, then show customers with above average total payments",
            expected=Expected(
                min_rows=1,
                grain_in=("row_level", "grouped"),
            ),
            category="multi_cte_chain",
        ),
        Scenario(
            id="MC-002",
            question="first get rental count per film, then list categories with above average rental count",
            expected=Expected(
                status_in=("ok", "validation_failed"),
            ),
            category="multi_cte_chain",
        ),
        Scenario(
            id="MC-003",
            question="first count rentals per store, then rank stores by total rentals",
            expected=Expected(
                min_rows=1,
            ),
            category="multi_cte_chain",
        ),
        Scenario(
            id="MC-004",
            question=(
                "first CTE: payment totals per customer_id; second CTE: join that to customer "
                "and keep customers in store 1; final select from the second CTE with customer name"
            ),
            expected=Expected(
                min_rows=1,
                sql_contains=["WITH"],
            ),
            category="multi_cte_chain",
        ),
    ]


def array_filter_scenarios() -> list[Scenario]:
    """
    Scenarios for ``film.special_features`` as ``TEXT[]`` / ``ARRAY<STRING>``.

    The pipeline should emit dialect-specific membership SQL (PostgreSQL ``unnest`` / ``btrim`` / ``EXISTS``, Databricks ``TRANSFORM`` / ``ARRAY_CONTAINS`` with ``TRIM``) when the intent uses ``contains`` on this array column. Stored feature strings match Sakila spelling (``Trailers``, ``Behind the Scenes``, ``Deleted Scenes``, ``Commentaries``). ``flatten_param_values`` strips decorative quotes from ``contains`` bind values so minor LLM quoting differences still match the database.
    """
    return [
        Scenario(
            id="AR-001",
            question="list titles of films that include Trailers in special features",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=2000,
                grain="row_level",
                sql_contains=["special_features"],
            ),
            category="array_filter",
        ),
        Scenario(
            id="AR-002",
            question="how many films have Behind the Scenes in special features",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["special_features"],
            ),
            category="array_filter",
        ),
        Scenario(
            id="AR-003",
            question="show film titles where special features include Deleted Scenes",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=2000,
                grain="row_level",
                sql_contains=["special_features"],
            ),
            category="array_filter",
        ),
    ]


def partition_scenarios() -> list[Scenario]:
    """
    Scenarios that exercise partition filter injection on Databricks.

    When schema has partition columns (e.g. ``rental.rental_pt`` on Delta), partition predicates are injected for partition pruning. Works as normal queries when tables have no partition columns.
    """
    return [
        Scenario(
            id="PT-001",
            question="how many rentals on 2005-07-15",
            expected=Expected(
                tables=["rental"],
                min_rows=0,
                max_rows=1000,
                grain="scalar",
                min_confidence=0.45,
            ),
            category="partition",
        ),
        Scenario(
            id="PT-002",
            question="list films with rating PG-13",
            expected=Expected(
                tables=["film"],
                min_rows=1,
                max_rows=500,
                grain="row_level",
                min_confidence=0.4,
            ),
            category="partition",
        ),
    ]


CATEGORY_LOADERS: dict[str, callable] = {
    "single_table": single_table_scenarios,
    "multi_table": multi_table_scenarios,
    "aggregation": aggregation_scenarios,
    "filtering": filtering_scenarios,
    "cte": cte_scenarios,
    "template_reuse": template_reuse_scenarios,
    "schema_edge": schema_edge_scenarios,
    "negative": negative_scenarios,
    "repair_loop": repair_loop_scenarios,
    "confidence": confidence_scenarios,
    "rejection": rejection_feedback_scenarios,
    "performance": performance_scenarios,
    "having": having_scenarios,
    "subquery": subquery_scenarios,
    "cte_join": cte_join_scenarios,
    "distinct": distinct_scenarios,
    "multi_agg": multi_agg_scenarios,
    "order_by": order_by_scenarios,
    "like_pattern": like_pattern_scenarios,
    "null_filter": null_filter_scenarios,
    "count_distinct": count_distinct_scenarios,
    "compound_filter": compound_filter_scenarios,
    "date_range": date_range_scenarios,
    "agg_filter_join": agg_filter_join_scenarios,
    "boolean_filter": boolean_filter_scenarios,
    "scalar_func": scalar_func_scenarios,
    "expr_select": expr_select_scenarios,
    "in_list": in_list_scenarios,
    "bridge_join": bridge_join_scenarios,
    "date_arithmetic": date_arithmetic_scenarios,
    "multi_cte_chain": multi_cte_chain_scenarios,
    "validation_failure": validation_failure_scenarios,
    "intent_rejected": intent_rejected_scenarios,
    "column_names": column_names_scenarios,
    "row_value_check": row_value_check_scenarios,
    "sql_excludes": sql_excludes_scenarios,
    "semantic_warnings": semantic_warnings_scenarios,
    "template_reuse_sequence": template_reuse_sequence_scenarios,
    "trust_cycle": trust_cycle_scenarios,
    "sql_vs_intent": sql_vs_intent_scenarios,
    "join_validation": join_validation_scenarios,
    "restrictions": restrictions_scenarios,
    "window_function": window_function_scenarios,
    "case_when": case_when_scenarios,
    "ast_explain": ast_explain_scenarios,
    "partition": partition_scenarios,
    "array_filter": array_filter_scenarios,
}


def all_scenarios() -> list[Scenario]:
    """
    Return every non-sequence scenario from all categories.

    Same set as the union of PostgreSQL ``live_tests/test_*.py`` modules that load from ``CATEGORY_LOADERS``. ``SequenceScenario`` entries are skipped. ``test_databricks`` uses this for parity with that union.
    """
    result: list[Scenario] = []
    for loader in CATEGORY_LOADERS.values():
        items = loader()
        result.extend(s for s in items if isinstance(s, Scenario))
    return result


def bundled_dvdrental_live_scenarios() -> list[Scenario]:
    """Alias for ``all_scenarios`` (explicit name for Databricks bundling)."""

    return all_scenarios()


def scenarios_by_category(category: str) -> list[Scenario]:
    """
    Return all scenarios for a given category string.

    Args: category: One of the keys in ``CATEGORY_LOADERS``.

    Returns: List of ``Scenario`` objects, or empty list for unknown categories.
    """
    loader = CATEGORY_LOADERS.get(category)
    if loader is None:
        return []
    return loader()
