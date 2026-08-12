"""Live pipeline scenario definitions for the ``rental_shop`` PostgreSQL schema (34 tables). Scenarios exercise catalog subtypes (``film``, ``book``, ``game``), bridge tables (``item_feature``, ``game_supported_language``, ``item_category``, ``film_actor``), rental operations (``rental``, ``reservation``, ``payment``, ``inventory``, ``inventory_status_history``, ``damage_report``), geography (``address``, ``city``, ``country``), procurement (``purchase_order``, ``purchase_line``, ``supplier``, ``warehouse``, ``stock_transfer``, ``delivery``, ``courier``), and reference entities (``author``, ``publisher``, ``language``, ``category``, ``promotion``, ``promotion_redemption``). Each function returns ``Scenario`` or ``SequenceScenario`` lists grouped by test category. ``generation_path_sequences`` holds multi-step ``GenerationPath`` live sequences for ``test_generation_paths_live``."""

from __future__ import annotations

from collections.abc import Callable

from aetherdialect._contracts_core import Expected, Scenario, SequenceScenario
from live_tests.mydb_profile import (
    RENTAL_SHOP_ACTOR_COUNT,
    RENTAL_SHOP_CATEGORY_COUNT,
    RENTAL_SHOP_CUSTOMER_COUNT,
    RENTAL_SHOP_DELIVERY_STATUS_COUNT,
    RENTAL_SHOP_FILM_CATEGORY_COUNT,
    RENTAL_SHOP_FILM_COUNT,
    RENTAL_SHOP_FILM_RATING_COUNT,
    RENTAL_SHOP_GAME_PLATFORM_COUNT,
    RENTAL_SHOP_LANGUAGE_COUNT,
    RENTAL_SHOP_PROMOTION_COUNT,
    RENTAL_SHOP_RENTAL_COUNT,
    RENTAL_SHOP_STORE_COUNT,
    RENTAL_SHOP_WAREHOUSE_COUNT,
)


def _rows_count_equals(n: int) -> Callable[[list[tuple]], bool]:
    return lambda rows: len(rows) == n


def _scalar_int_equals(n: int) -> Callable[[list[tuple]], bool]:
    return lambda rows: len(rows) == 1 and rows[0] and int(rows[0][0]) == n


def _grouped_rows_at_most(n: int) -> Callable[[list[tuple]], bool]:
    return lambda rows: 1 <= len(rows) <= n


# Post-reconcile table assertions for rental_shop item_id subtype pattern.
FILM_SCOPED: list[list[str]] = [["film"], ["item"], ["film", "item"]]
BOOK_SCOPED: list[list[str]] = [["book"], ["item"], ["book", "item"]]
GAME_SCOPED: list[list[str]] = [["game"], ["item"], ["game", "item"]]


def film_with(*extra: str) -> list[list[str]]:
    """Acceptable post-reconcile table sets for a film-scoped query plus extra tables."""
    return [sorted([*base, *extra]) for base in FILM_SCOPED]


def book_with(*extra: str) -> list[list[str]]:
    """Acceptable post-reconcile table sets for a book-scoped query plus extra tables."""
    return [sorted([*base, *extra]) for base in BOOK_SCOPED]


def game_with(*extra: str) -> list[list[str]]:
    """Acceptable post-reconcile table sets for a game-scoped query plus extra tables."""
    return [sorted([*base, *extra]) for base in GAME_SCOPED]


ACTOR_FILM_SCOPED: list[list[str]] = [
    ["actor", "film"],
    ["actor", "item"],
    ["actor", "film", "item"],
    ["actor", "film_actor", "item"],
    ["actor", "film_actor", "film"],
]

BOOK_AUTHOR_PUBLISHER: list[list[str]] = [
    ["author", "book", "publisher"],
    ["author", "item", "publisher"],
    ["author", "book", "item", "publisher"],
    ["book", "publisher"],
]

PURCHASE_COST_BY_SUPPLIER: list[list[str]] = [
    ["purchase_line", "supplier"],
    ["purchase_line", "purchase_order", "supplier"],
]

OPEN_PO_SUPPLIER: list[list[str]] = [
    ["purchase_order", "supplier"],
    ["purchase_line", "purchase_order", "supplier"],
]

PURCHASE_ORDER_SCOPED: list[list[str]] = [
    ["purchase_order"],
    ["purchase_line", "purchase_order"],
    ["purchase_order", "supplier"],
    ["purchase_line", "purchase_order", "supplier"],
]

ACTOR_FILM_ACTOR: list[list[str]] = [
    ["actor", "film"],
    ["actor", "item"],
    ["actor", "film", "item"],
    ["actor", "film_actor"],
    ["actor", "film", "film_actor"],
    ["actor", "item", "film_actor"],
]

AG003_FILM_RENTAL: list[list[str]] = [
    *film_with(),
    *film_with("rental"),
]

STORE_ADDRESS_CITY: list[list[str]] = [
    ["store", "address", "city"],
    ["address", "city"],
    ["store", "address"],
    ["city", "store"],
]


def single_table_scenarios() -> list[Scenario]:
    """Basic single-table queries with no joins."""
    base = [
        Scenario(
            id="ST-001",
            question="list all film titles",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=RENTAL_SHOP_FILM_COUNT,
                max_rows=RENTAL_SHOP_FILM_COUNT,
                contains_join=False,
                grain="row_level",
                row_value_check=_rows_count_equals(RENTAL_SHOP_FILM_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-002",
            question="show me all customer first names and last names",
            expected=Expected(
                tables=["customer"],
                min_rows=RENTAL_SHOP_CUSTOMER_COUNT,
                max_rows=RENTAL_SHOP_CUSTOMER_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_CUSTOMER_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-003",
            question="list the distinct film ratings in the catalog",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=RENTAL_SHOP_FILM_RATING_COUNT,
                max_rows=RENTAL_SHOP_FILM_RATING_COUNT,
                row_value_check=_rows_count_equals(RENTAL_SHOP_FILM_RATING_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-004",
            question="list all categories",
            expected=Expected(
                tables=["category"],
                min_rows=RENTAL_SHOP_CATEGORY_COUNT,
                max_rows=RENTAL_SHOP_CATEGORY_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_CATEGORY_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-005",
            question="how many films are there",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                max_rows=1,
                grain="scalar",
                row_value_check=_scalar_int_equals(RENTAL_SHOP_FILM_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-006",
            question="show all actor first names and last names",
            expected=Expected(
                tables=["actor"],
                min_rows=RENTAL_SHOP_ACTOR_COUNT,
                max_rows=RENTAL_SHOP_ACTOR_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_ACTOR_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-007",
            question="list the distinct languages in the catalog",
            expected=Expected(
                tables=["language"],
                min_rows=RENTAL_SHOP_LANGUAGE_COUNT,
                max_rows=RENTAL_SHOP_LANGUAGE_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_LANGUAGE_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-008",
            question="list all store ids",
            expected=Expected(
                tables=["store"],
                min_rows=RENTAL_SHOP_STORE_COUNT,
                max_rows=RENTAL_SHOP_STORE_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_STORE_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-009",
            question="list all promotion names",
            expected=Expected(
                tables=["promotion"],
                min_rows=RENTAL_SHOP_PROMOTION_COUNT,
                max_rows=RENTAL_SHOP_PROMOTION_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_PROMOTION_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-010",
            question="list the distinct delivery statuses",
            expected=Expected(
                tables=["delivery"],
                min_rows=RENTAL_SHOP_DELIVERY_STATUS_COUNT,
                max_rows=RENTAL_SHOP_DELIVERY_STATUS_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_DELIVERY_STATUS_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-011",
            question="list distinct warehouse names",
            expected=Expected(
                tables=["warehouse"],
                min_rows=RENTAL_SHOP_WAREHOUSE_COUNT,
                max_rows=RENTAL_SHOP_WAREHOUSE_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_WAREHOUSE_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-012",
            question="list distinct game platforms",
            expected=Expected(
                tables_one_of=GAME_SCOPED,
                min_rows=RENTAL_SHOP_GAME_PLATFORM_COUNT,
                max_rows=RENTAL_SHOP_GAME_PLATFORM_COUNT,
                contains_join=False,
                row_value_check=_rows_count_equals(RENTAL_SHOP_GAME_PLATFORM_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-013",
            question="which languages are available",
            expected=Expected(
                tables=["language"],
                min_rows=1,
                contains_join=False,
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-014",
            question="how many actors are in the database",
            expected=Expected(
                tables=["actor"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                contains_join=False,
                row_value_check=_scalar_int_equals(RENTAL_SHOP_ACTOR_COUNT),
            ),
            category="single_table",
        ),
        Scenario(
            id="ST-015",
            question="list all films in the catalog",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=RENTAL_SHOP_FILM_COUNT,
                max_rows=RENTAL_SHOP_FILM_COUNT,
                contains_join=False,
                grain="row_level",
                row_value_check=_rows_count_equals(RENTAL_SHOP_FILM_COUNT),
            ),
            category="single_table",
        ),
    ]
    return base


def multi_table_scenarios() -> list[Scenario]:
    """Multi-table join queries."""
    base = [
        Scenario(
            id="MT-001",
            question="list all films and their language",
            expected=Expected(tables_one_of=film_with("language"), contains_join=True, min_rows=1),
            category="multi_table",
        ),
        Scenario(
            id="MT-002",
            question="show all films with their categories",
            expected=Expected(
                tables_one_of=film_with("category") + [["category", "item", "item_category"]],
                contains_join=True,
                min_rows=1,
                sql_contains=["item_category", "category"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-003",
            question="list all actors and the films they appeared in",
            expected=Expected(
                tables_one_of=ACTOR_FILM_SCOPED,
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
                tables_one_of=[
                    ["city", "customer"],
                    ["address", "city", "customer"],
                ],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-005",
            question="list customer names and their country",
            expected=Expected(
                tables_one_of=[
                    ["country", "customer"],
                    ["address", "city", "country", "customer"],
                ],
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
                tables_one_of=film_with("language"),
                contains_join=True,
                min_rows=1,
                sql_contains=["English", "language"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-011",
            question="show customer first and last names with their delivery status",
            expected=Expected(
                tables_one_of=[
                    ["customer", "delivery"],
                    ["customer", "delivery", "rental"],
                ],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-012",
            question="list book titles with author name and publisher name",
            expected=Expected(
                tables_one_of=BOOK_AUTHOR_PUBLISHER,
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-013",
            question="show open purchase orders with supplier name and store id",
            expected=Expected(
                tables_one_of=OPEN_PO_SUPPLIER,
                contains_join=True,
                min_rows=1,
                sql_contains=["open"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-014",
            question="list deliveries with courier name and tracking number",
            expected=Expected(
                tables=["delivery", "courier"],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-015",
            question="show promotion redemptions with promo name and customer name",
            expected=Expected(
                tables_one_of=[
                    ["promotion_redemption", "promotion", "customer"],
                    ["promotion_redemption", "promotion", "customer", "rental"],
                ],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-016",
            question="list stock transfers with warehouse name and item title",
            expected=Expected(
                tables=["stock_transfer", "warehouse", "item"],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-017",
            question="show purchase lines with item title and unit cost",
            expected=Expected(
                tables=["purchase_line", "item"],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-018",
            question="which films include trailers",
            expected=Expected(
                tables_one_of=[
                    *FILM_SCOPED,
                    ["film", "item", "item_feature"],
                    ["film", "item_feature"],
                    ["item", "item_feature"],
                ],
                min_rows=1,
                max_rows=2000,
                grain="row_level",
                sql_contains=["item_feature"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-019",
            question="which films include deleted scenes",
            expected=Expected(
                tables_one_of=[
                    *FILM_SCOPED,
                    ["film", "item", "item_feature"],
                    ["film", "item_feature"],
                    ["item", "item_feature"],
                ],
                min_rows=1,
                max_rows=2000,
                grain="row_level",
                sql_contains=["item_feature"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-020",
            question="which games support English",
            expected=Expected(
                tables_one_of=[
                    *GAME_SCOPED,
                    ["game", "language"],
                    ["game", "game_supported_language", "language"],
                    ["game", "item", "game_supported_language", "language"],
                    ["item", "game", "game_supported_language", "language"],
                ],
                min_rows=1,
                grain="row_level",
                sql_contains=["game_supported_language"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-021",
            question="list all store locations by city",
            expected=Expected(
                tables_one_of=STORE_ADDRESS_CITY,
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-022",
            question="which films are in the Horror category",
            expected=Expected(
                contains_join=True,
                min_rows=0,
                sql_contains=["item_category"],
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-023",
            question="show active staff at each store",
            expected=Expected(
                tables_one_of=[
                    ["staff", "store"],
                    ["staff"],
                ],
                contains_join=True,
                min_rows=1,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-024",
            question="which customers from each country have made rentals",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-025",
            question="list stock transfers between warehouses this year",
            expected=Expected(
                tables_one_of=[
                    ["stock_transfer", "warehouse"],
                    ["stock_transfer"],
                    ["item", "stock_transfer", "warehouse"],
                    ["item", "stock_transfer", "store", "warehouse"],
                ],
                contains_join=True,
                min_rows=0,
            ),
            category="multi_table",
        ),
        Scenario(
            id="MT-026",
            question="how many rentals are linked to film titles",
            expected=Expected(
                tables_one_of=[
                    *film_with("rental", "inventory"),
                    ["rental", "inventory", "item"],
                    ["rental", "inventory", "item", "film"],
                ],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                contains_join=True,
            ),
            category="multi_table",
        ),
    ]
    return base


def aggregation_scenarios() -> list[Scenario]:
    """Aggregation and GROUP BY queries."""
    base = [
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
                max_rows=RENTAL_SHOP_CUSTOMER_COUNT,
                sql_contains=["SUM"],
                row_value_check=_grouped_rows_at_most(RENTAL_SHOP_CUSTOMER_COUNT),
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-003",
            question="average rental duration per film rating",
            expected=Expected(
                tables_one_of=AG003_FILM_RENTAL,
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
                row_value_check=_scalar_int_equals(RENTAL_SHOP_RENTAL_COUNT),
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-006",
            question="maximum replacement cost of films by rating",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                contains_group_by=True,
                grain="grouped",
                min_rows=RENTAL_SHOP_FILM_RATING_COUNT,
                max_rows=RENTAL_SHOP_FILM_RATING_COUNT,
                sql_contains=["MAX"],
                row_value_check=_rows_count_equals(RENTAL_SHOP_FILM_RATING_COUNT),
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
                min_rows=RENTAL_SHOP_LANGUAGE_COUNT,
                max_rows=RENTAL_SHOP_LANGUAGE_COUNT,
                row_value_check=_rows_count_equals(RENTAL_SHOP_LANGUAGE_COUNT),
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
        Scenario(
            id="AG-010",
            question="total discount amount by promotion type",
            expected=Expected(
                tables=["promotion_redemption", "promotion"],
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-011",
            question="count of purchase orders per supplier name",
            expected=Expected(
                tables_one_of=OPEN_PO_SUPPLIER,
                contains_join=True,
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-012",
            question="total purchase cost by supplier",
            expected=Expected(
                tables_one_of=PURCHASE_COST_BY_SUPPLIER,
                contains_group_by=True,
                contains_join=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-013",
            question="total discount amount by promotion name",
            expected=Expected(
                tables=["promotion_redemption", "promotion"],
                contains_join=True,
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-014",
            question="how many deliveries are in each status",
            expected=Expected(
                tables=["delivery"],
                contains_group_by=True,
                grain="grouped",
                min_rows=RENTAL_SHOP_DELIVERY_STATUS_COUNT,
                max_rows=RENTAL_SHOP_DELIVERY_STATUS_COUNT,
                row_value_check=_rows_count_equals(RENTAL_SHOP_DELIVERY_STATUS_COUNT),
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-015",
            question="how many open reservations are there",
            expected=Expected(tables=["reservation"], min_rows=0, max_rows=1, grain="scalar"),
            category="aggregation",
        ),
        Scenario(
            id="AG-016",
            question="how many open damage reports exist",
            expected=Expected(tables=["damage_report"], min_rows=0, max_rows=1, grain="scalar"),
            category="aggregation",
        ),
        Scenario(
            id="AG-017",
            question="which warehouse holds the most stock transfers",
            expected=Expected(
                tables_one_of=[
                    ["warehouse", "stock_transfer"],
                    ["stock_transfer", "warehouse"],
                ],
                min_rows=0,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-018",
            question="which suppliers have the most purchase lines",
            expected=Expected(
                tables_one_of=PURCHASE_COST_BY_SUPPLIER,
                min_rows=0,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-019",
            question="show delivery status counts by courier",
            expected=Expected(
                tables=["delivery", "courier"],
                min_rows=0,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-020",
            question="which author has the most books",
            expected=Expected(
                tables_one_of=[
                    ["author", "book"],
                    ["author", "item", "book"],
                    *BOOK_AUTHOR_PUBLISHER[:2],
                ],
                min_rows=0,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-021",
            question="how many items fall under each category name",
            expected=Expected(
                tables_one_of=[
                    ["category", "item"],
                    ["category", "item_category"],
                    ["category", "item", "item_category"],
                ],
                min_rows=0,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-022",
            question="how many games support French",
            expected=Expected(
                tables_one_of=[
                    *GAME_SCOPED,
                    ["game", "language"],
                    ["game", "game_supported_language", "language"],
                    ["game", "item", "game_supported_language", "language"],
                    ["item", "game", "game_supported_language", "language"],
                    ["game_supported_language", "language"],
                ],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["game_supported_language"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-023",
            question="how many catalog items by item type",
            expected=Expected(
                tables=["item"],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-024",
            question="how many games are in the catalog",
            expected=Expected(
                tables_one_of=GAME_SCOPED,
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-025",
            question="what is the total delivery fee by courier",
            expected=Expected(
                tables=["delivery", "courier"],
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-026",
            question="what is the average delivery fee for delivered shipments",
            expected=Expected(
                tables=["delivery"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-027",
            question="what is the average page count by publisher",
            expected=Expected(
                tables_one_of=BOOK_AUTHOR_PUBLISHER,
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-028",
            question="how many rentals were for books versus films",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-029",
            question="which staff member processed the most payments",
            expected=Expected(
                tables_one_of=[
                    ["payment", "staff"],
                    ["customer", "payment", "staff"],
                ],
                min_rows=0,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-030",
            question="for each country how many distinct cities have at least one customer",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-031",
            question="how many open damage reports exist per store",
            expected=Expected(
                tables_one_of=[
                    ["damage_report"],
                    ["damage_report", "store"],
                    ["damage_report", "inventory", "store"],
                ],
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-032",
            question="how many books do we have",
            expected=Expected(
                tables_one_of=BOOK_SCOPED,
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-033",
            question="what is the count of pending reservations by store",
            expected=Expected(
                tables_one_of=[
                    ["reservation"],
                    ["reservation", "store"],
                ],
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-034",
            question="what is the average rental rate by item type",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-035",
            question="what is the total revenue generated from rentals",
            expected=Expected(
                tables_one_of=[
                    ["payment"],
                    ["payment", "rental"],
                    ["customer", "payment", "rental"],
                ],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-036",
            question="how many rentals does each customer make on average",
            expected=Expected(
                grain_in=("grouped", "scalar"),
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-037",
            question="what is the total revenue generated by each staff member",
            expected=Expected(
                tables_one_of=[
                    ["payment", "staff"],
                    ["customer", "payment", "staff"],
                    ["payment", "rental", "staff"],
                ],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-038",
            question="what is the highest payment amount ever recorded",
            expected=Expected(
                tables=["payment"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["MAX"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-039",
            question="what is the average number of inventory copies per store",
            expected=Expected(
                tables_one_of=[
                    ["inventory", "store"],
                    ["inventory"],
                    ["store", "inventory"],
                ],
                grain_in=("grouped", "scalar"),
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-040",
            question="how many purchase orders are still open",
            expected=Expected(
                tables_one_of=PURCHASE_ORDER_SCOPED,
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-041",
            question="how many reservations expired without being fulfilled",
            expected=Expected(
                tables=["reservation"],
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-042",
            question="which publisher has the fewest books in the catalog",
            expected=Expected(
                tables_one_of=BOOK_AUTHOR_PUBLISHER,
                min_rows=0,
                max_rows=1,
                contains_group_by=True,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-043",
            question="which promotion had the most redemptions",
            expected=Expected(
                tables=["promotion", "promotion_redemption"],
                contains_join=True,
                contains_group_by=True,
                min_rows=0,
                max_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-044",
            question="what is the average rental duration",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    *film_with("rental"),
                ],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-045",
            question="how many films are in the horror category",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["horror"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-046",
            question="which actors appear in the most films",
            expected=Expected(
                tables_one_of=ACTOR_FILM_ACTOR,
                contains_group_by=True,
                min_rows=0,
                max_rows=10,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-047",
            question="what is the average payment amount",
            expected=Expected(
                tables=["payment"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-048",
            question="how many inventory items does each store have",
            expected=Expected(
                tables_one_of=[
                    ["inventory", "store"],
                    ["inventory"],
                    ["store", "inventory"],
                ],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-049",
            question="what is the count of pending versus fulfilled reservations",
            expected=Expected(
                tables=["reservation"],
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-050",
            question="books grouped by publisher name",
            expected=Expected(
                tables_one_of=BOOK_AUTHOR_PUBLISHER,
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-051",
            question="which promotion type has the highest average discount percent",
            expected=Expected(
                tables=["promotion"],
                contains_group_by=True,
                min_rows=0,
                max_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-052",
            question="what is the average rental duration by item type",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["AVG"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-053",
            question="what is the total payment amount",
            expected=Expected(
                tables=["payment"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-054",
            question="what is the average payment amount grouped by film category",
            expected=Expected(
                contains_group_by=True,
                grain="grouped",
                min_rows=1,
                sql_contains=["SUM"],
            ),
            category="aggregation",
        ),
        Scenario(
            id="AG-055",
            question="show payroll deductions grouped by staff member",
            expected=Expected(
                tables_one_of=[
                    ["staff"],
                    ["staff", "store"],
                ],
                contains_group_by=True,
                grain="grouped",
                min_rows=0,
            ),
            category="aggregation",
        ),
    ]
    return base


def filtering_scenarios() -> list[Scenario]:
    """Filter and parameterization queries."""
    base = [
        Scenario(
            id="FI-001",
            question="list all films with rating PG-13",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["2.99"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-004",
            question="how many films have a length greater than 120 minutes",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["R", "20"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-009",
            question="show rentals from July 2023",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["2023"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-010",
            question="list book titles with page count over 400",
            expected=Expected(
                tables_one_of=BOOK_SCOPED,
                min_rows=1,
                sql_contains=["page_count"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-011",
            question="list games with esrb rating T",
            expected=Expected(
                tables_one_of=GAME_SCOPED,
                min_rows=1,
                sql_contains=["esrb_rating"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-012",
            question="list all rentals made by customers with first name John",
            expected=Expected(
                contains_join=True,
                min_rows=0,
                sql_contains=["John"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-013",
            question="list items in the Sci-Fi category",
            expected=Expected(
                contains_join=True,
                min_rows=0,
                sql_contains=["Sci-Fi"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-014",
            question="items with rental rates between 2 and 4",
            expected=Expected(
                tables_one_of=[
                    *FILM_SCOPED,
                    ["item"],
                    ["book", "item"],
                    ["game", "item"],
                ],
                min_rows=0,
                sql_contains_one_of=[["BETWEEN"], ["2", "4"]],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-015",
            question="items where replacement cost is greater than 20",
            expected=Expected(
                tables_one_of=[
                    ["item"],
                    *FILM_SCOPED,
                    *BOOK_SCOPED,
                    *GAME_SCOPED,
                ],
                min_rows=1,
                sql_contains=["20"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-016",
            question="which staff members work at store 1",
            expected=Expected(
                tables_one_of=[
                    ["staff", "store"],
                    ["staff"],
                ],
                min_rows=0,
                sql_contains=["1"],
            ),
            category="filtering",
        ),
        Scenario(
            id="FI-017",
            question="how many rentals are currently overdue",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["inventory", "item", "rental"],
                ],
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="filtering",
        ),
    ]
    return base


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
            id="CT-002",
            question="list customers who have rented more than 29 films",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-003",
            question="what is the average payment per rental for each customer",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-004",
            question="show categories where the average film length is above 120 minutes",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-005",
            question="list actors who appeared in more than 30 films along with the count",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="cte",
        ),
        Scenario(
            id="CT-006",
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
                tables_one_of=[
                    ["country", "customer"],
                    ["address", "city", "country", "customer"],
                ],
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
                tables_one_of=STORE_ADDRESS_CITY,
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
            id="SE-008",
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
            id="SE-009",
            question="show the district for each customer",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["district"],
            ),
            category="schema_edge",
        ),
    ]


def negative_scenarios() -> list[Scenario]:
    """Negative and forbidden SQL pattern tests."""
    base = [
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
        Scenario(
            id="NG-013",
            question="what color should I paint my kitchen",
            expected=Expected(status="invalid_question"),
            category="negative",
        ),
        Scenario(
            id="NG-014",
            question="what is the best pizza topping",
            expected=Expected(status="invalid_question"),
            category="negative",
        ),
    ]
    return base


def repair_loop_scenarios() -> list[Scenario]:
    """Repair loop and retry behaviour tests. These scenarios target complex queries that may trigger the SQL repair loop, testing that the pipeline can self-correct and still produce valid output."""
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
            question="list the 5 least rented films with their category and total revenue",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=5,
            ),
            category="repair_loop",
        ),
        Scenario(
            id="RL-003",
            question="show categories where total revenue exceeds 4000 with the film count",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="repair_loop",
        ),
    ]


def confidence_scenarios() -> list[Scenario]:
    """Scenarios that check row and grain expectations."""
    return [
        Scenario(
            id="CF-001",
            question="how many customers are in each country",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                grain="grouped",
            ),
            category="confidence",
        ),
        Scenario(
            id="CF-002",
            question="what is the average film length",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="confidence",
        ),
        Scenario(
            id="CF-003",
            question="list all customer email addresses",
            expected=Expected(
                min_rows=1,
                tables=["customer"],
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
                    expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1),
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


def rejection_feedback_scenarios() -> list[Scenario]:
    """Rejection feedback loop scenarios — user says no and provides a reason."""
    return [
        Scenario(
            id="RJ-001",
            question="list all customer names",
            expected=Expected(tables=["customer"]),
            category="rejection",
            feedback="n",
            reject_reason="too many rows",
        ),
    ]


def performance_scenarios() -> list[Scenario]:
    """Performance and cost-awareness scenarios. These test that the pipeline completes within a reasonable time and doesn't produce excessively large result sets."""
    return []


def having_scenarios() -> list[Scenario]:
    """HAVING clause queries that filter grouped results."""
    base = [
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
            id="HV-003",
            question="list ratings that have an average rental rate above 3",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-004",
            question="show categories where average rental rate is above 3 or total number of films is more than 60",
            expected=Expected(
                contains_group_by=True,
                contains_join=True,
                sql_contains=["HAVING", "OR"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-005",
            question="promotion types where average discount percent is above 15",
            expected=Expected(
                tables=["promotion"],
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=0,
            ),
            category="having",
        ),
        Scenario(
            id="HV-006",
            question="promotions with more than 5 redemptions",
            expected=Expected(
                tables=["promotion", "promotion_redemption"],
                contains_join=True,
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-007",
            question="warehouses with more than 10 stock transfers",
            expected=Expected(
                tables=["warehouse", "stock_transfer"],
                contains_group_by=True,
                contains_join=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="having",
        ),
        Scenario(
            id="HV-008",
            question="list publishers with more than five books",
            expected=Expected(
                tables_one_of=[
                    ["publisher", "book"],
                    ["publisher", "item", "book"],
                    *BOOK_AUTHOR_PUBLISHER[:2],
                ],
                min_rows=0,
            ),
            category="having",
        ),
        Scenario(
            id="HV-009",
            question="which customers have rented more than 10 items",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=0,
            ),
            category="having",
        ),
        Scenario(
            id="HV-010",
            question="which categories have total revenue exceeding 500",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=0,
            ),
            category="having",
        ),
        Scenario(
            id="HV-011",
            question="which actors have appeared in more than 20 films",
            expected=Expected(
                tables_one_of=ACTOR_FILM_ACTOR,
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=0,
            ),
            category="having",
        ),
        Scenario(
            id="HV-012",
            question="list active customers who have made more than 5 rentals",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                sql_contains=["HAVING", "active"],
                min_rows=0,
            ),
            category="having",
        ),
    ]
    return base


def sql_vs_intent_scenarios() -> list[Scenario]:
    """SQL-vs-intent structural consistency checks."""
    return [
        Scenario(
            id="SI-001",
            question="list films ordered by length descending",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-002",
            question="show the top 10 longest films",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY", "LIMIT"],
            ),
            category="sql_vs_intent",
        ),
        Scenario(
            id="SI-003",
            question="show ratings with more than 200 films",
            expected=Expected(
                contains_group_by=True,
                sql_contains=["HAVING"],
                min_rows=1,
            ),
            category="sql_vs_intent",
        ),
    ]


def subquery_scenarios() -> list[Scenario]:
    """Subquery patterns including NOT IN, EXISTS, and correlated subqueries."""
    base = [
        Scenario(
            id="SB-001",
            question="show customers who have never made a payment",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-002",
            question="list actors who have appeared in more films than average",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-003",
            question="which films have a rental rate higher than the average rental rate",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-004",
            question="show categories with fewer films than the average films per category",
            expected=Expected(
                min_rows=1,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-005",
            question="list customers who have rented films from both store 1 and store 2",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
        Scenario(
            id="SB-006",
            question="list customers who have never rented an item",
            expected=Expected(
                min_rows=0,
            ),
            category="subquery",
        ),
    ]
    return base


def cte_join_scenarios() -> list[Scenario]:
    """CTE queries that also require joins across multiple tables."""
    return [
        Scenario(
            id="CJ-001",
            question="highest grossing film per category",
            expected=Expected(
                min_rows=1,
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-002",
            question="for each store show the top 3 customers by number of rentals",
            expected=Expected(
                min_rows=1,
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-003",
            question=(
                "first in a CTE count rentals per item_id, then join that CTE to film and item "
                "and list each film title with its rental count"
            ),
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["WITH", "rental", "item"],
            ),
            category="cte_join",
        ),
        Scenario(
            id="CJ-004",
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
            id="CJ-005",
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
            id="JV-002",
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
            id="JV-003",
            question="show the manager name for each store",
            expected=Expected(
                contains_join=True,
                min_rows=1,
                max_rows=12,
                sql_contains=["store", "staff"],
            ),
            category="join_validation",
        ),
        Scenario(
            id="JV-004",
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
    base = [
        Scenario(
            id="OB-001",
            question="list all films ordered by title",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-002",
            question="show the 5 longest films",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                max_rows=5,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-008",
            question="list all film titles alphabetically",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-009",
            question="which film has the shortest rental duration",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=0,
                max_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-010",
            question="list films ordered by rental duration descending",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-011",
            question="who are our top 5 customers by total payment",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
                max_rows=5,
                sql_contains=["LIMIT"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-012",
            question="which customers have rented the most items",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-013",
            question="what are the most frequently rented items",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-014",
            question="which customer has the highest single payment amount",
            expected=Expected(
                tables_one_of=[
                    ["customer", "payment"],
                    ["payment"],
                ],
                min_rows=0,
                max_rows=1,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-015",
            question="list the 10 most expensive items in the catalog by replacement cost",
            expected=Expected(
                tables_one_of=[
                    ["item"],
                    *FILM_SCOPED,
                    *BOOK_SCOPED,
                    *GAME_SCOPED,
                ],
                min_rows=1,
                max_rows=10,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
        Scenario(
            id="OB-016",
            question="which actors have the most film credits",
            expected=Expected(
                tables_one_of=ACTOR_FILM_ACTOR,
                contains_group_by=True,
                min_rows=0,
                max_rows=10,
                sql_contains=["ORDER BY"],
            ),
            category="order_by",
        ),
    ]
    return base


def like_pattern_scenarios() -> list[Scenario]:
    """LIKE and pattern matching queries."""
    return [
        Scenario(
            id="LK-001",
            question="list films with titles starting with 'A'",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=0,
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
                tables_one_of=FILM_SCOPED,
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
        Scenario(
            id="LK-006",
            question="list customers whose last name contains son case insensitive",
            expected=Expected(
                tables=["customer"],
                min_rows=0,
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
        Scenario(
            id="NL-005",
            question="which staff members have no email address on file",
            expected=Expected(
                tables=["staff"],
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
                row_value_check=_scalar_int_equals(RENTAL_SHOP_FILM_CATEGORY_COUNT),
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
        Scenario(
            id="CD-005",
            question="how many distinct customer full names are there",
            expected=Expected(
                tables=["customer"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["COUNT", "DISTINCT", "CONCAT"],
            ),
            category="count_distinct",
        ),
        Scenario(
            id="CD-006",
            question="how many distinct items were rented in total",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["inventory", "rental"],
                    ["item", "rental"],
                ],
                min_rows=1,
                max_rows=1,
                grain="scalar",
            ),
            category="count_distinct",
        ),
        Scenario(
            id="CD-007",
            question="how many distinct customers rented horror films",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["horror"],
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
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["PG-13"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-002",
            question="list films rated R or NC-17 with replacement cost below 15",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-003",
            question="list films in English with rental duration greater than 5 days",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-004",
            question="show active customers who live in a city that starts with 'A'",
            expected=Expected(
                contains_join=True,
                min_rows=0,
                sql_contains=["LIKE"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-005",
            question="show films with rental rate above 3 or length under 60",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-006",
            question="films rated PG-13 with length over 120 or rated R with length over 150",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
        Scenario(
            id="CF2-007",
            question=(
                "list films where (rating is PG-13 and length under 90) or (rating is G and rental_rate above 2)"
            ),
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["OR"],
            ),
            category="compound_filter",
        ),
    ]


def date_range_scenarios() -> list[Scenario]:
    """Date range and temporal filter queries."""
    base = [
        Scenario(
            id="DR-001",
            question="show rentals on July 15 2023",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                ],
                min_rows=0,
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-002",
            question="show all rentals between July 1 2023 and July 31 2023",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                ],
                min_rows=1,
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-003",
            question="total payments collected in August 2023",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-004",
            question="count of rentals per month",
            expected=Expected(
                contains_group_by=True,
                min_rows=1,
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-005",
            question="how many payments were made after February 2023",
            expected=Expected(
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-006",
            question="list inventory status changes in the last 90 days",
            expected=Expected(tables=["inventory_status_history"], min_rows=0),
            category="date_range",
        ),
        Scenario(
            id="DR-007",
            question="how many promotions ended in 2024",
            expected=Expected(
                tables=["promotion"],
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["2024"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-008",
            question="list purchase orders received in 2023",
            expected=Expected(
                tables_one_of=PURCHASE_ORDER_SCOPED,
                min_rows=0,
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-009",
            question="show deliveries dispatched in 2023",
            expected=Expected(
                tables=["delivery"],
                min_rows=1,
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-010",
            question="how many rentals happened in the last 30 days",
            expected=Expected(
                tables=["rental"],
                min_rows=0,
                max_rows=1,
                grain="scalar",
            ),
            category="date_range",
        ),
        Scenario(
            id="DR-011",
            question="how many stock transfers occurred in 2023",
            expected=Expected(
                tables=["stock_transfer"],
                min_rows=0,
                max_rows=1,
                grain="scalar",
                sql_contains=["2023"],
            ),
            category="date_range",
        ),
    ]
    return base


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
                tables_one_of=FILM_SCOPED,
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
            question="for each language show the top 3 longest films by length",
            expected=Expected(
                contains_cte=True,
                min_rows=1,
                sql_contains_one_of=[
                    ["ROW_NUMBER", "RANK", "DENSE_RANK"],
                    ["PARTITION"],
                ],
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
        Scenario(
            id="BF-004",
            question="list active promotions only",
            expected=Expected(
                tables=["promotion"],
                min_rows=1,
                sql_contains=["active"],
            ),
            category="boolean_filter",
        ),
        Scenario(
            id="BF-005",
            question="show inactive couriers",
            expected=Expected(
                tables=["courier"],
                min_rows=0,
                sql_contains=["active"],
            ),
            category="boolean_filter",
        ),
        Scenario(
            id="BF-006",
            question="list active suppliers only",
            expected=Expected(
                tables=["supplier"],
                min_rows=1,
                sql_contains=["active"],
            ),
            category="boolean_filter",
        ),
        Scenario(
            id="BF-007",
            question="show failed deliveries",
            expected=Expected(
                tables_one_of=[
                    ["delivery"],
                    ["address", "courier", "delivery"],
                ],
                min_rows=0,
                sql_contains=["failed"],
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
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1),
            category="scalar_func",
        ),
        Scenario(
            id="SF-002",
            question="list film titles and their length in hours",
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1),
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
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1),
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
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1, sql_contains=["IN"]),
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
                tables_one_of=ACTOR_FILM_ACTOR,
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
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
        Scenario(
            id="DA-004",
            question="list rentals that were returned after the due date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["item", "rental"],
                    ["inventory", "item", "rental"],
                    ["customer", "rental"],
                    ["rental", "staff"],
                    ["customer", "rental", "staff"],
                    ["customer", "item", "rental"],
                ],
                min_rows=0,
            ),
            category="date_arithmetic",
        ),
    ]


def validation_failure_scenarios() -> list[Scenario]:
    """Scenarios where validation may fail; accept ok or validation_failed."""
    base = [
        Scenario(
            id="VF-001",
            question="show customer first name and total of all payments",
            expected=Expected(
                status_in=("ok", "validation_failed"),
                tables=["customer", "payment"],
            ),
            category="validation_failure",
        ),
        Scenario(
            id="VF-002",
            question="which films have never been rented",
            expected=Expected(
                status_in=("ok", "validation_failed"),
                min_rows=0,
            ),
            category="validation_failure",
        ),
        Scenario(
            id="VF-003",
            question="show payroll deductions by employee SSN",
            expected=Expected(
                status_in=("ok", "validation_failed"),
                min_rows=0,
            ),
            category="validation_failure",
        ),
    ]
    return base


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
    base = [
        Scenario(
            id="WF-001",
            question="rank films by length descending using row number",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-002",
            question="for each film rating, list film titles with a row number ordered by length descending",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["OVER (", "PARTITION BY"],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-003",
            question="for each film show title, length, and the average length of films with the same rating",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
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
                    ["customer", "payment", "rental"],
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
        Scenario(
            id="WF-010",
            question="rank films by rental count within each category",
            expected=Expected(
                contains_join=True,
                contains_group_by=True,
                min_rows=1,
                sql_contains=["OVER ("],
            ),
            category="window_function",
        ),
        Scenario(
            id="WF-011",
            question=("for each store rank customers by total rental count and return only the top customer per store"),
            expected=Expected(
                contains_join=True,
                min_rows=1,
                sql_contains=["OVER (", "PARTITION BY"],
            ),
            category="window_function",
        ),
    ]
    return base


def case_when_scenarios() -> list[Scenario]:
    """Live scenarios that should produce CASE expressions in SELECT."""
    return [
        Scenario(
            id="CW-001",
            question="list film titles with a label column that is premium when rental_rate is greater than 3 otherwise standard",
            expected=Expected(
                tables_one_of=FILM_SCOPED,
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
                tables_one_of=FILM_SCOPED,
                min_rows=1,
                sql_contains=["CASE", "WHEN"],
            ),
            category="case_when",
        ),
        Scenario(
            id="CW-003",
            question=("for each item title show premium or standard when replacement cost exceeds 25 dollars"),
            expected=Expected(
                contains_join=False,
                min_rows=1,
                sql_contains=["CASE"],
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
            question="list films and their language",
            expected=Expected(
                contains_join=True,
                min_rows=1,
            ),
            category="ast_explain",
        ),
        Scenario(
            id="AE-002",
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


def sensitivity_enforcement_scenarios() -> list[Scenario]:
    """Restricted procurement cost and forbidden staff credential queries."""
    return [
        Scenario(
            id="SS-001",
            question="what is the average unit cost for purchase lines with quantity greater than 5",
            expected=Expected(
                tables=["purchase_line"],
                contains_group_by=False,
                min_rows=1,
                max_rows=1,
                grain="scalar",
                sql_contains=["unit_cost"],
            ),
            category="sensitivity_enforcement",
        ),
        Scenario(
            id="SS-002",
            question="list staff password values",
            expected=Expected(
                status="restricted",
                min_rows=0,
            ),
            category="sensitivity_enforcement",
        ),
    ]


def partition_scenarios() -> list[Scenario]:
    """Scenarios that exercise partition filter injection when schema carries pruning metadata. Delta ``rental_pt`` on Databricks is the primary fixture, but the same scenario runs on any engine whose reflected schema includes partition, sortkey, distkey, or clustering columns and whose intent supplies matching filters."""
    return [
        Scenario(
            id="PT-001",
            question="how many rentals on 2023-07-15",
            expected=Expected(
                tables=["rental"],
                min_rows=0,
                max_rows=1000,
                grain="scalar",
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
    "sql_vs_intent": sql_vs_intent_scenarios,
    "join_validation": join_validation_scenarios,
    "restrictions": restrictions_scenarios,
    "window_function": window_function_scenarios,
    "case_when": case_when_scenarios,
    "ast_explain": ast_explain_scenarios,
    "partition": partition_scenarios,
}


def all_scenarios() -> list[Scenario]:
    """Return every non-sequence scenario from all categories. Same set as the union of PostgreSQL ``live_tests/test_*.py`` modules that load from ``CATEGORY_LOADERS``. ``SequenceScenario`` entries are skipped. ``test_databricks`` uses this for parity with that union."""
    result: list[Scenario] = []
    for loader in CATEGORY_LOADERS.values():
        items = loader()
        result.extend(s for s in items if isinstance(s, Scenario))
    return result


CORE_ISOLATED_LIVE_SCENARIO_IDS: tuple[str, ...] = (
    "ST-001",
    "ST-002",
    "ST-003",
    "ST-004",
    "ST-005",
    "ST-006",
    "ST-007",
    "ST-008",
    "ST-009",
    "ST-010",
    "ST-011",
    "ST-012",
    "ST-014",
    "ST-015",
    "AG-002",
    "AG-005",
    "AG-006",
    "AG-008",
    "AG-014",
    "CD-003",
)


def scenario_has_exact_row_oracle(expected: Expected) -> bool:
    """Return whether *expected* asserts row cardinality beyond bare ``min_rows=1``."""
    if expected.row_value_check is not None:
        return True
    if expected.max_rows is not None:
        return True
    return False


def core_isolated_live_scenarios() -> list[Scenario]:
    """Return the high-traffic rental_shop scenarios run on isolated runners."""
    by_id = {scenario.id: scenario for scenario in all_scenarios()}
    missing = [scenario_id for scenario_id in CORE_ISOLATED_LIVE_SCENARIO_IDS if scenario_id not in by_id]
    if missing:
        raise KeyError(f"unknown core isolated scenario ids: {missing}")
    return [by_id[scenario_id] for scenario_id in CORE_ISOLATED_LIVE_SCENARIO_IDS]


def bundled_rental_shop_live_scenarios() -> list[Scenario]:
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


# --- dialect scenarios ---


def dialect_array_scenarios() -> list[Scenario]:
    """Membership filters on ``item_feature.feature_name`` bridge rows."""
    _film_feature = [
        *FILM_SCOPED,
        ["film", "item", "item_feature"],
        ["film", "item_feature"],
        ["item", "item_feature"],
    ]

    return [
        Scenario(
            id="DIAL-AR-001",
            question="list titles of films that have trailers in item_feature",
            expected=Expected(tables_one_of=_film_feature, min_rows=0),
            category="dialect_array",
        ),
        Scenario(
            id="DIAL-AR-002",
            question="how many films have behind_the_scenes in item_feature",
            expected=Expected(tables_one_of=_film_feature, min_rows=1),
            category="dialect_array",
        ),
        Scenario(
            id="DIAL-AR-003",
            question=(
                "list film titles that have all four item_feature values trailers, commentaries, "
                "deleted_scenes, and behind_the_scenes"
            ),
            expected=Expected(tables_one_of=_film_feature, min_rows=0),
            category="dialect_array",
        ),
    ]


def dialect_date_window_scenarios() -> list[Scenario]:
    """Relative and absolute date-window filters (``DATE_TRUNC`` / ``DATE_SUB`` / ``DATEADD``)."""
    return [
        Scenario(
            id="DIAL-DW-001",
            question="show rentals from the last 90 days",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-002",
            question="list rentals from the last 6 months",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-003",
            question="show rentals from the last year",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
        Scenario(
            id="DIAL-DW-004",
            question="list rentals between January 1 2023 and December 31 2023",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                    ["customer", "rental", "staff"],
                    ["rental", "staff"],
                ],
                min_rows=0,
            ),
            category="dialect_date_window",
        ),
    ]


def dialect_date_diff_scenarios() -> list[Scenario]:
    """Column-to-column date difference (``DATEDIFF`` / ``TIMESTAMPDIFF`` / ``DATE_DIFF``)."""
    return [
        Scenario(
            id="DIAL-DD-001",
            question="list rentals where the return was more than 7 days after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-002",
            question="list rentals where the return was more than 2 weeks after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-003",
            question="list rentals where the return was more than 1 month after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
        Scenario(
            id="DIAL-DD-004",
            question="list rentals where the return was more than 1 year after the rental date",
            expected=Expected(
                tables_one_of=[
                    ["rental"],
                    ["customer", "rental"],
                    ["inventory", "rental"],
                ],
                min_rows=0,
            ),
            category="dialect_date_diff",
        ),
    ]


def dialect_case_insensitive_scenarios() -> list[Scenario]:
    """Case-insensitive string matching (``LOWER`` wrap; no ``ILIKE`` on MySQL/SQL Server)."""
    return [
        Scenario(
            id="DIAL-CI-001",
            question="list film titles containing the word Harbor",
            expected=Expected(tables=["item"], min_rows=1),
            category="dialect_case_insensitive",
        ),
    ]


def dialect_pagination_scenarios() -> list[Scenario]:
    """LIMIT/TOP/OFFSET pagination rendering."""
    return [
        Scenario(
            id="DIAL-PG-001",
            question="show the first 10 film titles alphabetically",
            expected=Expected(tables_one_of=FILM_SCOPED, min_rows=1, max_rows=10),
            category="dialect_pagination",
        ),
    ]


def dialect_identifier_qualification_scenarios() -> list[Scenario]:
    """Schema-qualified or multi-part table references in FROM/JOIN."""
    return [
        Scenario(
            id="DIAL-IQ-001",
            question="list customer first name and last name for customers in store 1",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_identifier_qualification",
        ),
    ]


def dialect_cast_coercion_scenarios() -> list[Scenario]:
    """CAST/typed literal coercion in filters or projections."""
    return [
        Scenario(
            id="DIAL-CAST-001",
            question="list film titles where length is greater than 120",
            expected=Expected(tables=["film"], min_rows=0),
            category="dialect_cast",
        ),
    ]


def dialect_case_when_scenarios() -> list[Scenario]:
    """Minimal CASE WHEN rendering."""
    return [
        Scenario(
            id="DIAL-CASE-001",
            question="show film titles and whether length is over 120 minutes as long or short",
            expected=Expected(tables=["film"], min_rows=1),
            category="dialect_case_when",
        ),
    ]


def dialect_scalar_func_scenarios() -> list[Scenario]:
    """Scalar function rendering (upper/trim/coalesce)."""
    return [
        Scenario(
            id="DIAL-SCALAR-001",
            question="list upper case film titles",
            expected=Expected(tables=["item"], min_rows=1),
            category="dialect_scalar_func",
        ),
    ]


def dialect_boolean_scenarios() -> list[Scenario]:
    """Boolean column filters."""
    return [
        Scenario(
            id="DIAL-BOOL-001",
            question="list active customers",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_boolean",
        ),
    ]


def dialect_window_agg_scenarios() -> list[Scenario]:
    """Aggregate window functions (SUM/AVG OVER) for Databricks normalize path."""
    return [
        Scenario(
            id="DIAL-WIN-001",
            question="for each customer show their total payment amount and the running sum of payments ordered by payment date",
            expected=Expected(
                tables_one_of=[["payment", "customer"], ["customer", "payment"]],
                min_rows=1,
            ),
            category="dialect_window_agg",
        ),
    ]


def dialect_string_concat_scenarios() -> list[Scenario]:
    """CONCAT execution smoke (not a divergence test)."""
    return [
        Scenario(
            id="DIAL-CONCAT-001",
            question="show customer first and last names concatenated with a space",
            expected=Expected(tables=["customer"], min_rows=1),
            category="dialect_string_concat",
        ),
    ]


def dialect_explain_smoke_scenarios() -> list[Scenario]:
    """EXPLAIN hook smoke (parse/execute path only)."""
    return [
        Scenario(
            id="DIAL-EXPLAIN-001",
            question="how many films are in the database",
            expected=Expected(tables=["film"], min_rows=1, max_rows=1),
            category="dialect_explain",
        ),
    ]


def dialect_sqlite_safe_scenarios() -> list[Scenario]:
    """Dialect scenarios scoped to SQLite-supported syntax (JSON1 arrays, julianday date diff, date('now') windows)."""
    return (
        dialect_array_scenarios()
        + dialect_date_window_scenarios()
        + dialect_date_diff_scenarios()
        + dialect_case_insensitive_scenarios()
        + dialect_pagination_scenarios()
        + dialect_cast_coercion_scenarios()
        + dialect_case_when_scenarios()
        + dialect_scalar_func_scenarios()
        + dialect_boolean_scenarios()
        + dialect_string_concat_scenarios()
    )


def dialect_mysql_scenarios() -> list[Scenario]:
    """MySQL dialect-syntax scenarios targeting JSON, date windows, TIMESTAMPDIFF, and EXPLAIN."""
    return (
        dialect_array_scenarios()
        + dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:2]
        + dialect_pagination_scenarios()[:1]
        + dialect_explain_smoke_scenarios()[:1]
    )


def dialect_snowflake_scenarios() -> list[Scenario]:
    """Snowflake dialect-syntax scenarios targeting ARRAY_CONTAINS, DATEADD windows, and qualification."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:1]
        + dialect_identifier_qualification_scenarios()[:2]
    )


def dialect_bigquery_scenarios() -> list[Scenario]:
    """BigQuery dialect-syntax scenarios targeting arrays, date windows, and TABLESAMPLE profiling paths."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_date_window_scenarios()[:2]
        + dialect_case_insensitive_scenarios()[:1]
        + dialect_pagination_scenarios()[:1]
    )


def dialect_sqlserver_scenarios() -> list[Scenario]:
    """SQL Server dialect-syntax scenarios targeting OPENJSON arrays, OFFSET/FETCH, and SHOWPLAN smoke."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_pagination_scenarios()
        + dialect_date_diff_scenarios()[:1]
        + dialect_explain_smoke_scenarios()[:1]
    )


def dialect_oracle_scenarios() -> list[Scenario]:
    """Oracle dialect-syntax scenarios targeting JSON_TABLE arrays, OFFSET/FETCH, and EXPLAIN PLAN smoke."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_pagination_scenarios()
        + dialect_date_diff_scenarios()[:1]
        + dialect_explain_smoke_scenarios()[:1]
    )


def dialect_redshift_scenarios() -> list[Scenario]:
    """Redshift dialect-syntax scenarios targeting ILIKE, SUPER/json arrays, and sortkey-aware filters."""
    return (
        dialect_array_scenarios()
        + dialect_case_insensitive_scenarios()
        + dialect_date_window_scenarios()[:1]
        + dialect_boolean_scenarios()[:1]
    )


def dialect_duckdb_scenarios() -> list[Scenario]:
    """DuckDB dialect-syntax scenarios targeting list_contains, ILIKE, date_diff, and USING SAMPLE paths."""
    return (
        dialect_array_scenarios()[:2]
        + dialect_case_insensitive_scenarios()[:1]
        + dialect_date_diff_scenarios()[:2]
        + dialect_date_window_scenarios()[:1]
    )


def dialect_sqlite_scenarios() -> list[Scenario]:
    """SQLite dialect-syntax scenarios targeting json_each, julianday, date windows, and LIMIT pagination."""
    return dialect_sqlite_safe_scenarios()[:8]


def dialect_databricks_scenarios() -> list[Scenario]:
    """Databricks dialect-syntax scenarios targeting DATETRUNC windows, qualification, and scalar CTE joins."""
    return (
        dialect_date_window_scenarios()[:2]
        + dialect_date_diff_scenarios()[:1]
        + dialect_identifier_qualification_scenarios()[:2]
        + dialect_scalar_func_scenarios()[:1]
    )
