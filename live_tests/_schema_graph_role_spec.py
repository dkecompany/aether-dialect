"""Per-column role allow-lists for rental_shop cross-engine schema-graph tests."""

from __future__ import annotations

from aetherdialect._contracts_base import ColumnRole

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
