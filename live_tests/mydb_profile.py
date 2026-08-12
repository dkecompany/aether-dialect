"""Per-database live-test profile: paths, role allow-lists, federation oracles, and DB-shaped helpers. Swap this module (plus ``mydb_scenarios.py`` and ``env.env``) when pointing live tests at another database."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from aetherdialect._constants import DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
from aetherdialect._contracts_schema import ColumnRole, SchemaGraph

# --- rental partition metadata ---


def apply_synthetic_rental_partition_metadata(sg: SchemaGraph) -> None:
    """Tag local DuckDB/SQLite rental_shop graphs with synthetic partition columns for pruning tests."""
    rental = sg.tables.get("rental")
    if rental is None:
        return
    if "rental_date" in rental.columns and not rental.partition_columns:
        rental.partition_columns = ["rental_date"]


# --- schema graph role allow-list ---

RENTAL_SHOP_COLUMN_ROLE_ALLOWLIST: dict[str, frozenset[str]] = {
    "actor.actor_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "actor.first_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "actor.last_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "actor.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "address.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "address.address": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "address.district": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "address.city_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "address.postal_code": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "address.phone": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "address.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "author.author_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "author.first_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "author.last_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "author.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "book.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "book.author_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "book.publisher_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "book.isbn": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "book.page_count": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "book.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "category.category_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "category.name": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "category.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "city.city_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "city.city": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "city.country_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "city.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "country.country_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "country.country": frozenset(
        {
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.FREE_TEXT.value,
        }
    ),
    "country.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "courier.courier_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "courier.courier_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "courier.phone": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "courier.country_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "courier.is_active": frozenset(
        {ColumnRole.BOOLEAN.value, ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}
    ),
    "courier.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "customer.customer_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "customer.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "customer.first_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "customer.last_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "customer.email": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "customer.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "customer.activebool": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "customer.create_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "customer.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "damage_report.damage_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "damage_report.rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "damage_report.inventory_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "damage_report.reported_by_staff_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "damage_report.severity": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "damage_report.repair_cost": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "damage_report.reported_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "damage_report.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "delivery.delivery_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "delivery.rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "delivery.courier_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "delivery.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "delivery.dispatched_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "delivery.delivered_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "delivery.status": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "delivery.delivery_fee": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "delivery.tracking_number": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "delivery.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "film.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "film.original_language_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "film.length": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "film.rating": frozenset(
        {ColumnRole.BOOLEAN.value, ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}
    ),
    "film.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "film_actor.actor_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "film_actor.film_item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "film_actor.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "game.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "game.platform": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "game.developer": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "game.esrb_rating": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "game.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "game_supported_language.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "game_supported_language.language_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "game_supported_language.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "inventory.inventory_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "inventory_status_history.status_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory_status_history.inventory_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory_status_history.status": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "inventory_status_history.changed_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "inventory_status_history.staff_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "inventory_status_history.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "item.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "item.item_type": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "item.title": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "item.description": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "item.release_year": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "item.language_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "item.rental_duration": frozenset(
        {
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "item.rental_rate": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "item.replacement_cost": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "item.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "item_category.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "item_category.category_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "item_category.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "item_feature.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "item_feature.feature_name": frozenset({ColumnRole.IDENTIFIER.value}),
    "item_feature.feature_type": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "item_feature.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "language.language_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "language.name": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "language.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "payment.payment_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "payment.rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "payment.amount": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "payment.payment_date": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "promotion.promotion_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "promotion.promo_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "promotion.promo_type": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "promotion.discount_pct": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "promotion.start_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "promotion.end_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "promotion.is_active": frozenset(
        {ColumnRole.BOOLEAN.value, ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}
    ),
    "promotion.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "promotion_redemption.redemption_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "promotion_redemption.promotion_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "promotion_redemption.rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "promotion_redemption.discount_amount": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "promotion_redemption.redeemed_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "promotion_redemption.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "publisher.publisher_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "publisher.publisher_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "publisher.country_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "publisher.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "purchase_line.line_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_line.po_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_line.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_line.quantity": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "purchase_line.unit_cost": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "purchase_line.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "purchase_order.po_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_order.supplier_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_order.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "purchase_order.ordered_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "purchase_order.received_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "purchase_order.status": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "purchase_order.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "rental.rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "rental.rental_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "rental.inventory_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "rental.customer_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "rental.return_date": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "rental.staff_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "rental.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "reservation.reservation_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "reservation.customer_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "reservation.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "reservation.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "reservation.reserved_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "reservation.expires_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "reservation.fulfilled_rental_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "reservation.status": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}),
    "reservation.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "staff.staff_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "staff.first_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "staff.last_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "staff.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "staff.email": frozenset({ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}),
    "staff.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "staff.active": frozenset(
        {ColumnRole.BOOLEAN.value, ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}
    ),
    "staff.username": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "staff.password": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "staff.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "stock_transfer.transfer_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "stock_transfer.item_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "stock_transfer.from_warehouse_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "stock_transfer.to_store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "stock_transfer.quantity": frozenset({ColumnRole.NUMERIC_MEASURE.value}),
    "stock_transfer.transferred_at": frozenset(
        {
            ColumnRole.AUDIT.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.CATEGORICAL.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.IDENTIFIER.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
        }
    ),
    "stock_transfer.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "store.store_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "store.manager_staff_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "store.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "store.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "supplier.supplier_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "supplier.supplier_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "supplier.country_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "supplier.is_active": frozenset(
        {ColumnRole.BOOLEAN.value, ColumnRole.CATEGORICAL.value, ColumnRole.NUMERIC_CATEGORICAL.value}
    ),
    "supplier.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
    "warehouse.warehouse_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "warehouse.warehouse_name": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "warehouse.address_id": frozenset({ColumnRole.IDENTIFIER.value}),
    "warehouse.capacity": frozenset(
        {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}
    ),
    "warehouse.last_update": frozenset({ColumnRole.AUDIT.value, ColumnRole.TEMPORAL.value}),
}

# --- federation oracles ---

# Sandbox subset corpus (partition seeds on storefront + catalog)
SANDBOX_SUBSET_JOIN_RENTAL_LINKED_TITLES = 192
SANDBOX_SUBSET_PAYMENT_TOTAL = Decimal("6947.73")
SANDBOX_SUBSET_HORROR_DISTINCT_CUSTOMERS = 22
SANDBOX_SUBSET_HORROR_PARTIAL_SUM_WRONG = 27

# Full live partition load across storefront, catalog, logistics and crm
LIVE_FULL_JOIN_RENTAL_LINKED_TITLES = 18758
LIVE_FULL_PAYMENT_TOTAL = Decimal("58596.41")
LIVE_FULL_HORROR_DISTINCT_CUSTOMERS = 555

# Single-member questions through the federated engine on the sandbox subset
ACTOR_COUNT = 6
CUSTOMER_COUNT = 73
RENTAL_COUNT = 2262
INVENTORY_COUNT = 542
FILM_CATALOG_ROW_COUNT = 96
CATEGORY_COUNT = 28

# Cross-source logistics delivery rows with a matching storefront rental (orphans excluded)
DELIVERY_RENTAL_LINKED_COUNT = 1

# Full rental_shop PostgreSQL corpus (scripts/source_rental_shop.py targets)
RENTAL_SHOP_FILM_COUNT = 1000
RENTAL_SHOP_ACTOR_COUNT = 200
RENTAL_SHOP_CUSTOMER_COUNT = 599
RENTAL_SHOP_RENTAL_COUNT = 17516
RENTAL_SHOP_STORE_COUNT = 12
RENTAL_SHOP_CATEGORY_COUNT = 28
RENTAL_SHOP_FILM_CATEGORY_COUNT = 16
RENTAL_SHOP_LANGUAGE_COUNT = 6
RENTAL_SHOP_WAREHOUSE_COUNT = 6
RENTAL_SHOP_DELIVERY_STATUS_COUNT = 4
RENTAL_SHOP_PROMOTION_COUNT = 25
RENTAL_SHOP_FILM_RATING_COUNT = 5
RENTAL_SHOP_GAME_PLATFORM_COUNT = 5

# --- federation member assertions ---


def _scalar_value(step: Any) -> float | int | Decimal | None:
    data = getattr(step, "data", None)
    if data is not None and len(data) == 1 and len(data.columns) == 1:
        return data.iloc[0, 0]
    message = str(getattr(step, "message", "") or "").strip()
    if message:
        try:
            return Decimal(message)
        except Exception:
            return None
    return None


def assert_staff_email_from_storefront_authoritative(step: Any) -> None:
    assert step.done, getattr(step, "error", step)
    assert not step.error, getattr(step, "message", "")
    assert step.sql
    sql_lower = str(step.sql).lower()
    assert "staff" in sql_lower
    assert "email" in sql_lower
    assert "crm" not in sql_lower
    data = getattr(step, "data", None)
    assert data is not None and len(data) > 0, "expected staff email rows"
    email_col = next((col for col in data.columns if "email" in str(col).lower()), None)
    assert email_col is not None
    values = [str(value) for value in data[email_col].tolist() if value is not None and str(value).strip()]
    assert values, "expected non-empty email values"
    assert any("@" in value for value in values)


def assert_median_payment_refused_for_mysql_family(step: Any) -> None:
    assert step.done, "expected terminal step"
    assert step.error, "median must refuse on mysql-family federation members"
    code = getattr(step, "refusal_diagnostic_code", None)
    err = getattr(step, "error", None)
    if code is None and err is not None:
        code = getattr(err, "detail_code", None)
    assert code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP, code
    parts = [str(getattr(step, "message", "") or "")]
    if isinstance(err, str):
        parts.append(err)
    elif err is not None:
        parts.append(str(getattr(err, "code", "") or ""))
    for diagnostic in getattr(step, "diagnostics", ()) or ():
        parts.append(str(getattr(diagnostic, "message", "") or ""))
    combined = " ".join(part for part in parts if part).lower()
    assert "median" in combined


def assert_delivery_rental_cross_source_linked_count(step: Any) -> None:
    assert step.done, getattr(step, "error", step)
    assert not step.error, getattr(step, "message", "")
    assert step.sql
    sql_lower = str(step.sql).lower()
    assert "delivery" in sql_lower
    assert "rental" in sql_lower
    succeeded = getattr(step, "federation_succeeded", ()) or ()
    if succeeded:
        sources = {str(entry[0]).lower() for entry in succeeded}
        assert {"logistics", "storefront"}.issubset(sources), sources
    value = _scalar_value(step)
    assert value is not None
    assert int(value) == DELIVERY_RENTAL_LINKED_COUNT


# --- profile paths / defaults ---

PROFILE_REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_OVERRIDES_PATH = PROFILE_REPO_ROOT / "scripts" / "data" / "rental_shop_overrides.json"
PROFILE_NOTES_DEFAULT = PROFILE_REPO_ROOT / "scripts" / "data" / "rental_shop_notes.txt"
PROFILE_SQL_DEFAULT = PROFILE_REPO_ROOT / "scripts" / "data" / "rental_shop.sql"
PROFILE_VIEWS_SQL = PROFILE_REPO_ROOT / "scripts" / "data" / "rental_shop_views.sql"
PROFILE_FEDERATION_DECLARATION = Path(__file__).resolve().parent / "fixtures" / "federation_live_declaration.json"
PROFILE_DATABASE_NAME_DEFAULT = "rental_shop"
PROFILE_CONSUMER_ALLOW_OBJECTS_BY_MEMBER: dict[str, frozenset[str]] = {
    "storefront": frozenset(
        {
            "address",
            "city",
            "country",
            "customer",
            "payment",
            "rental",
            "reservation",
            "staff",
            "store",
        }
    ),
    "catalog": frozenset(
        {
            "actor",
            "category",
            "city",
            "country",
            "film",
            "film_actor",
            "inventory",
            "item",
            "item_category",
            "language",
            "payment",
        }
    ),
    "logistics": frozenset(
        {
            "courier",
            "delivery",
            "purchase_order",
            "receipts",
            "supplier",
            "warehouse",
        }
    ),
    "crm": frozenset(
        {
            "customer",
            "promotion",
            "promotion_redemption",
            "staff",
        }
    ),
}
# Live pguser2 grants align with the catalog member scope for RBAC matrix tests.
PROFILE_CONSUMER_ALLOW_OBJECTS = PROFILE_CONSUMER_ALLOW_OBJECTS_BY_MEMBER["catalog"]
