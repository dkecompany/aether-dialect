"""
AdventureWorks manufacturing scenario definitions.

Tables available:
  product (504 rows) — product_id, name, product_number, make_flag,
    finished_goods_flag, color, safety_stock_level, reorder_point,
    standard_cost, list_price, size, weight, days_to_manufacture,
    product_line, class, style, product_subcategory_id, product_model_id,
    sell_start_date, sell_end_date
  transaction_history (65 535 rows) — transaction_id, product_id,
    reference_order_id, transaction_date, transaction_type, quantity,
    actual_cost
  work_order (65 535 rows) — work_order_id, product_id, order_qty,
    stocked_qty, scrapped_qty, start_date, end_date, due_date,
    scrap_reason_id
  work_order_routing (59 832 rows) — work_order_id, product_id,
    operation_sequence, location_id, scheduled_start_date,
    actual_resource_hrs, planned_cost, actual_cost
  purchase_order_detail (8 845 rows) — purchase_order_id,
    purchase_order_detail_id, due_date, order_qty, product_id, unit_price,
    line_total, received_qty, rejected_qty, stocked_qty
"""

from __future__ import annotations

from aetherdialect._live_testing import Expected, Scenario


def product_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="AW-P-001",
            question="list all product names and their list prices",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                max_rows=600,
                contains_join=False,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-002",
            question="which products are finished goods?",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                contains_join=False,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-003",
            question="show me the top 10 most expensive products by list price",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                max_rows=10,
                contains_join=False,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-004",
            question="how many products are there per color?",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-005",
            question="what is the average standard cost per product line?",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-006",
            question="which products have a safety stock level below 100?",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                contains_join=False,
            ),
            category="product",
        ),
        Scenario(
            id="AW-P-007",
            question="show products that take more than 3 days to manufacture",
            expected=Expected(
                tables=["product"],
                min_rows=1,
                contains_join=False,
            ),
            category="product",
        ),
    ]


def transaction_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="AW-T-001",
            question="what is the total quantity sold per product?",
            expected=Expected(
                tables=["transaction_history"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="transaction",
        ),
        Scenario(
            id="AW-T-002",
            question="show me total actual cost per transaction type",
            expected=Expected(
                tables=["transaction_history"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="transaction",
        ),
        Scenario(
            id="AW-T-003",
            question="how many transactions happened each month?",
            expected=Expected(
                tables=["transaction_history"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="transaction",
        ),
        Scenario(
            id="AW-T-004",
            question="which products have the highest total actual cost in transaction history?",
            expected=Expected(
                tables=["transaction_history"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="transaction",
        ),
    ]


def work_order_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="AW-W-001",
            question="how many work orders are there per product?",
            expected=Expected(
                tables=["work_order"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="work_order",
        ),
        Scenario(
            id="AW-W-002",
            question="what is the total scrapped quantity across all work orders?",
            expected=Expected(
                tables=["work_order"],
                min_rows=1,
            ),
            category="work_order",
        ),
        Scenario(
            id="AW-W-003",
            question="show work orders where scrapped quantity is greater than 0",
            expected=Expected(
                tables=["work_order"],
                min_rows=1,
                contains_join=False,
            ),
            category="work_order",
        ),
        Scenario(
            id="AW-W-004",
            question="what is the average order quantity per product in work orders?",
            expected=Expected(
                tables=["work_order"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="work_order",
        ),
    ]


def join_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="AW-J-001",
            question="show product names with their total transaction quantity",
            expected=Expected(
                tables_one_of=[
                    ["product", "transaction_history"],
                    ["transaction_history", "product"],
                ],
                min_rows=1,
                contains_join=True,
                contains_group_by=True,
            ),
            category="join",
        ),
        Scenario(
            id="AW-J-002",
            question="list product names alongside their work order count",
            expected=Expected(
                tables_one_of=[
                    ["product", "work_order"],
                    ["work_order", "product"],
                ],
                min_rows=1,
                contains_join=True,
                contains_group_by=True,
            ),
            category="join",
        ),
        Scenario(
            id="AW-J-003",
            question="show product names and their total purchase order line total",
            expected=Expected(
                tables_one_of=[
                    ["product", "purchase_order_detail"],
                    ["purchase_order_detail", "product"],
                ],
                min_rows=1,
                contains_join=True,
                contains_group_by=True,
            ),
            category="join",
        ),
        Scenario(
            id="AW-J-004",
            question="which products have both work orders and purchase orders?",
            expected=Expected(
                min_rows=1,
                contains_join=True,
            ),
            category="join",
        ),
    ]


def routing_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="AW-R-001",
            question="what is the average actual resource hours per location?",
            expected=Expected(
                tables=["work_order_routing"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="routing",
        ),
        Scenario(
            id="AW-R-002",
            question="show total planned cost vs actual cost per location in work order routing",
            expected=Expected(
                tables=["work_order_routing"],
                min_rows=1,
                contains_group_by=True,
            ),
            category="routing",
        ),
        Scenario(
            id="AW-R-003",
            question="which work order routings had actual cost exceeding planned cost?",
            expected=Expected(
                tables=["work_order_routing"],
                min_rows=1,
            ),
            category="routing",
        ),
    ]
