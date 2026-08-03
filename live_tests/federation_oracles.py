"""Numeric oracles for live federation cross-engine slots."""

from __future__ import annotations

from decimal import Decimal

# Sandbox subset corpus (partition seeds on storefront + catalog before full rebuild)
SANDBOX_SUBSET_JOIN_RENTAL_LINKED_TITLES = 192
SANDBOX_SUBSET_PAYMENT_TOTAL = Decimal("6947.73")
SANDBOX_SUBSET_HORROR_DISTINCT_CUSTOMERS = 22
SANDBOX_SUBSET_HORROR_PARTIAL_SUM_WRONG = 27

# Full live partition load across storefront, catalog, logistics and crm
LIVE_FULL_JOIN_RENTAL_LINKED_TITLES = 18758
LIVE_FULL_PAYMENT_TOTAL = Decimal("58596.41")
LIVE_FULL_HORROR_DISTINCT_CUSTOMERS = 555

# Backward-compatible aliases for sandbox-subset callers
JOIN_RENTAL_LINKED_TITLES = SANDBOX_SUBSET_JOIN_RENTAL_LINKED_TITLES
PAYMENT_TOTAL = SANDBOX_SUBSET_PAYMENT_TOTAL
HORROR_DISTINCT_CUSTOMERS = SANDBOX_SUBSET_HORROR_DISTINCT_CUSTOMERS
HORROR_PARTIAL_SUM_WRONG = SANDBOX_SUBSET_HORROR_PARTIAL_SUM_WRONG

# Single-member questions through the federated engine on the sandbox subset
ACTOR_COUNT = 6
CUSTOMER_COUNT = 73
RENTAL_COUNT = 2262
INVENTORY_COUNT = 542
FILM_CATALOG_ROW_COUNT = 96
CATEGORY_COUNT = 28
