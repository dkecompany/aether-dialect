# Sandbox data reference

Closed-world inventory of every table, column, view, federation member, and bundled question shipped in the **rental_shop** sandbox bundle. Use this document when authoring `EngineContext(allow_objects=...)`, federation declarations, or AetherSpace scopes. For session walkthroughs and API patterns, see [Sandbox guide](SANDBOX.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md) → this document.

## Sections

| Section | Contents |
| --- | --- |
| [Overview](#overview) | Table and view counts, date range, closed-world rule |
| [Single-engine schema](#single-engine-schema) | All 34 tables with columns, keys, and foreign keys |
| [Views](#views) | Bundled analytical views and `include="views"` |
| [Bundled notes and named spaces](#bundled-notes-and-named-spaces) | Optional notes fixtures and `SpaceContext` subsets |
| [Structure and sensitivity fixtures](#structure-and-sensitivity-fixtures) | Hidden vs denied columns, bundled structure demos |
| [Consumer scopes](#consumer-scopes) | Owner role and narrowed consumer `allow_objects` |
| [Federation topology](#federation-topology) | Members, replicas, joins, and logical tables |
| [Question corpus](#question-corpus) | Bundled question tiers |
| [Authoring checklist](#authoring-checklist) | Closed-world naming rules |

---

## Overview

| Property | Value |
| --- | --- |
| Base tables | **34** (`rental_shop.sql`) |
| Analytical views | **3** (`rental_shop_views.sql`; loaded when `include="views"`) |
| DuckDB schema | `main` (default in-memory connection) |
| Synthetic activity dates | Predominantly **2022–2025** (some seed rows span earlier years) |
| Closed-world rule | Every table, column, member name, and bundled question in this document is namable in the sandbox |

The single-engine owner seed loads all 34 tables from `rental_shop.sql` into one in-memory DuckDB database. Views are defined separately and are not part of the base seed DDL.

---

## Single-engine schema

Tables are grouped by domain area. Column types match `rental_shop.sql`. Primary keys (PK) and foreign keys (FK) list the referenced table.

### Geography and language

#### `country`

| Column | Type | Key |
| --- | --- | --- |
| `country_id` | INTEGER | PK |
| `country` | VARCHAR(50) | UNIQUE |
| `last_update` | TIMESTAMP | |

#### `city`

| Column | Type | Key |
| --- | --- | --- |
| `city_id` | INTEGER | PK |
| `city` | VARCHAR(50) | |
| `country_id` | INTEGER | FK → `country.country_id` |
| `last_update` | TIMESTAMP | |

#### `address`

| Column | Type | Key |
| --- | --- | --- |
| `address_id` | INTEGER | PK |
| `address` | VARCHAR(50) | |
| `district` | VARCHAR(20) | |
| `city_id` | INTEGER | FK → `city.city_id` |
| `postal_code` | VARCHAR(10) | |
| `phone` | VARCHAR(20) | |
| `last_update` | TIMESTAMP | |

#### `language`

| Column | Type | Key |
| --- | --- | --- |
| `language_id` | INTEGER | PK |
| `name` | CHAR(20) | UNIQUE |
| `last_update` | TIMESTAMP | |

### Catalog and media

#### `actor`

| Column | Type | Key |
| --- | --- | --- |
| `actor_id` | INTEGER | PK |
| `first_name` | VARCHAR(45) | |
| `last_name` | VARCHAR(45) | |
| `last_update` | TIMESTAMP | |

#### `category`

| Column | Type | Key |
| --- | --- | --- |
| `category_id` | INTEGER | PK |
| `name` | VARCHAR(25) | UNIQUE |
| `last_update` | TIMESTAMP | |

#### `author`

| Column | Type | Key |
| --- | --- | --- |
| `author_id` | INTEGER | PK |
| `first_name` | VARCHAR(45) | |
| `last_name` | VARCHAR(45) | |
| `last_update` | TIMESTAMP | |

#### `publisher`

| Column | Type | Key |
| --- | --- | --- |
| `publisher_id` | INTEGER | PK |
| `publisher_name` | VARCHAR(120) | |
| `country_id` | INTEGER | FK → `country.country_id` |
| `last_update` | TIMESTAMP | |

#### `item`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK |
| `item_type` | VARCHAR(10) | CHECK: `film`, `book`, `game` |
| `title` | VARCHAR(255) | |
| `description` | TEXT | |
| `release_year` | INTEGER | |
| `language_id` | INTEGER | FK → `language.language_id` |
| `rental_duration` | SMALLINT | default 3 |
| `rental_rate` | NUMERIC(4,2) | default 4.99 |
| `replacement_cost` | NUMERIC(5,2) | default 19.99 |
| `last_update` | TIMESTAMP | |

#### `film`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `item.item_id` |
| `original_language_id` | INTEGER | FK → `language.language_id` |
| `length` | SMALLINT | |
| `rating` | VARCHAR(10) | default `G` |
| `last_update` | TIMESTAMP | |

#### `book`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `item.item_id` |
| `author_id` | INTEGER | FK → `author.author_id` |
| `publisher_id` | INTEGER | FK → `publisher.publisher_id` |
| `isbn` | VARCHAR(20) | UNIQUE |
| `page_count` | SMALLINT | |
| `last_update` | TIMESTAMP | |

#### `game`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `item.item_id` |
| `platform` | VARCHAR(30) | |
| `developer` | VARCHAR(80) | |
| `esrb_rating` | VARCHAR(10) | |
| `last_update` | TIMESTAMP | |

#### `item_category`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `item.item_id` |
| `category_id` | INTEGER | PK, FK → `category.category_id` |
| `last_update` | TIMESTAMP | |

#### `film_actor`

| Column | Type | Key |
| --- | --- | --- |
| `actor_id` | INTEGER | PK, FK → `actor.actor_id` |
| `film_item_id` | INTEGER | PK, FK → `film.item_id` |
| `last_update` | TIMESTAMP | |

#### `item_feature`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `item.item_id` |
| `feature_name` | VARCHAR(80) | PK |
| `feature_type` | VARCHAR(30) | |
| `last_update` | TIMESTAMP | |

#### `game_supported_language`

| Column | Type | Key |
| --- | --- | --- |
| `item_id` | INTEGER | PK, FK → `game.item_id` |
| `language_id` | INTEGER | PK, FK → `language.language_id` |
| `last_update` | TIMESTAMP | |

### Store operations

#### `store`

| Column | Type | Key |
| --- | --- | --- |
| `store_id` | INTEGER | PK |
| `manager_staff_id` | INTEGER | FK → `staff.staff_id` |
| `address_id` | INTEGER | FK → `address.address_id` |
| `last_update` | TIMESTAMP | |

#### `staff`

| Column | Type | Key |
| --- | --- | --- |
| `staff_id` | INTEGER | PK |
| `first_name` | VARCHAR(45) | |
| `last_name` | VARCHAR(45) | |
| `address_id` | INTEGER | FK → `address.address_id` |
| `email` | VARCHAR(50) | |
| `store_id` | INTEGER | FK → `store.store_id` |
| `active` | BOOLEAN | default true |
| `username` | VARCHAR(16) | UNIQUE |
| `password` | VARCHAR(40) | sensitivity exercise column |
| `ssn` | VARCHAR(11) | sensitivity exercise column |
| `last_update` | TIMESTAMP | |

#### `inventory`

| Column | Type | Key |
| --- | --- | --- |
| `inventory_id` | INTEGER | PK |
| `item_id` | INTEGER | FK → `item.item_id` |
| `store_id` | INTEGER | FK → `store.store_id` |
| `last_update` | TIMESTAMP | |

#### `customer`

| Column | Type | Key |
| --- | --- | --- |
| `customer_id` | INTEGER | PK |
| `store_id` | INTEGER | FK → `store.store_id` |
| `first_name` | VARCHAR(45) | |
| `last_name` | VARCHAR(45) | |
| `email` | VARCHAR(50) | |
| `address_id` | INTEGER | FK → `address.address_id` |
| `activebool` | BOOLEAN | default true |
| `create_date` | DATE | default CURRENT_DATE |
| `last_update` | TIMESTAMP | |

#### `rental`

| Column | Type | Key |
| --- | --- | --- |
| `rental_id` | INTEGER | PK |
| `rental_date` | TIMESTAMP | partition-pruning demo column |
| `inventory_id` | INTEGER | FK → `inventory.inventory_id` |
| `customer_id` | INTEGER | FK → `customer.customer_id` |
| `return_date` | TIMESTAMP | NULL = still checked out |
| `staff_id` | INTEGER | FK → `staff.staff_id` |
| `last_update` | TIMESTAMP | |

Overdue rule (from DDL comment): `return_date IS NULL AND rental_date + item.rental_duration < CURRENT_DATE`.

#### `payment`

| Column | Type | Key |
| --- | --- | --- |
| `payment_id` | INTEGER | PK |
| `rental_id` | INTEGER | FK → `rental.rental_id` |
| `amount` | NUMERIC(5,2) | |
| `payment_date` | TIMESTAMP | |

#### `reservation`

| Column | Type | Key |
| --- | --- | --- |
| `reservation_id` | INTEGER | PK |
| `customer_id` | INTEGER | FK → `customer.customer_id` |
| `item_id` | INTEGER | FK → `item.item_id` |
| `store_id` | INTEGER | FK → `store.store_id` |
| `reserved_at` | TIMESTAMP | |
| `expires_at` | TIMESTAMP | |
| `fulfilled_rental_id` | INTEGER | FK → `rental.rental_id` |
| `status` | VARCHAR(20) | CHECK: `pending`, `fulfilled`, `expired`, `cancelled` |
| `last_update` | TIMESTAMP | |

### Logistics and procurement

#### `courier`

| Column | Type | Key |
| --- | --- | --- |
| `courier_id` | INTEGER | PK |
| `courier_name` | VARCHAR(80) | |
| `phone` | VARCHAR(20) | |
| `country_id` | INTEGER | FK → `country.country_id` |
| `is_active` | BOOLEAN | default true |
| `last_update` | TIMESTAMP | |

#### `supplier`

| Column | Type | Key |
| --- | --- | --- |
| `supplier_id` | INTEGER | PK |
| `supplier_name` | VARCHAR(120) | UNIQUE |
| `country_id` | INTEGER | FK → `country.country_id` |
| `is_active` | BOOLEAN | default true |
| `last_update` | TIMESTAMP | |

#### `warehouse`

| Column | Type | Key |
| --- | --- | --- |
| `warehouse_id` | INTEGER | PK |
| `warehouse_name` | VARCHAR(80) | UNIQUE |
| `address_id` | INTEGER | FK → `address.address_id` |
| `capacity` | INTEGER | |
| `last_update` | TIMESTAMP | |

#### `purchase_order`

| Column | Type | Key |
| --- | --- | --- |
| `po_id` | INTEGER | PK |
| `supplier_id` | INTEGER | FK → `supplier.supplier_id` |
| `store_id` | INTEGER | FK → `store.store_id` |
| `ordered_date` | DATE | |
| `received_date` | DATE | |
| `status` | VARCHAR(20) | CHECK: `open`, `received`, `cancelled` |
| `last_update` | TIMESTAMP | |

#### `purchase_line`

| Column | Type | Key |
| --- | --- | --- |
| `line_id` | INTEGER | PK |
| `po_id` | INTEGER | FK → `purchase_order.po_id` |
| `item_id` | INTEGER | FK → `item.item_id` |
| `quantity` | SMALLINT | |
| `unit_cost` | NUMERIC(8,2) | |
| `last_update` | TIMESTAMP | |

Unique on (`po_id`, `item_id`).

#### `stock_transfer`

| Column | Type | Key |
| --- | --- | --- |
| `transfer_id` | INTEGER | PK |
| `item_id` | INTEGER | FK → `item.item_id` |
| `from_warehouse_id` | INTEGER | FK → `warehouse.warehouse_id` |
| `to_store_id` | INTEGER | FK → `store.store_id` |
| `quantity` | SMALLINT | |
| `transferred_at` | TIMESTAMP | |
| `last_update` | TIMESTAMP | |

#### `delivery`

| Column | Type | Key |
| --- | --- | --- |
| `delivery_id` | INTEGER | PK |
| `rental_id` | INTEGER | FK → `rental.rental_id` |
| `courier_id` | INTEGER | FK → `courier.courier_id` |
| `address_id` | INTEGER | FK → `address.address_id` |
| `dispatched_at` | TIMESTAMP | |
| `delivered_at` | TIMESTAMP | |
| `status` | VARCHAR(20) | CHECK: `dispatched`, `in_transit`, `delivered`, `failed`, `returned` |
| `delivery_fee` | NUMERIC(6,2) | |
| `tracking_number` | VARCHAR(30) | UNIQUE |
| `last_update` | TIMESTAMP | |

#### `inventory_status_history`

| Column | Type | Key |
| --- | --- | --- |
| `status_id` | INTEGER | PK |
| `inventory_id` | INTEGER | FK → `inventory.inventory_id` |
| `status` | VARCHAR(20) | CHECK: `available`, `rented`, `damaged`, `in_repair`, `lost`, `retired` |
| `changed_at` | TIMESTAMP | |
| `staff_id` | INTEGER | FK → `staff.staff_id` |
| `last_update` | TIMESTAMP | |

#### `damage_report`

| Column | Type | Key |
| --- | --- | --- |
| `damage_id` | INTEGER | PK |
| `rental_id` | INTEGER | FK → `rental.rental_id` |
| `inventory_id` | INTEGER | FK → `inventory.inventory_id` |
| `reported_by_staff_id` | INTEGER | FK → `staff.staff_id` |
| `severity` | VARCHAR(20) | CHECK: `minor`, `moderate`, `severe` |
| `repair_cost` | NUMERIC(8,2) | |
| `reported_at` | TIMESTAMP | |
| `last_update` | TIMESTAMP | |

### Promotions

#### `promotion`

| Column | Type | Key |
| --- | --- | --- |
| `promotion_id` | INTEGER | PK |
| `promo_name` | VARCHAR(120) | |
| `promo_type` | VARCHAR(30) | |
| `discount_pct` | NUMERIC(5,2) | |
| `start_date` | DATE | |
| `end_date` | DATE | |
| `is_active` | BOOLEAN | default true |
| `last_update` | TIMESTAMP | |

#### `promotion_redemption`

| Column | Type | Key |
| --- | --- | --- |
| `redemption_id` | INTEGER | PK |
| `promotion_id` | INTEGER | FK → `promotion.promotion_id` |
| `rental_id` | INTEGER | FK → `rental.rental_id` |
| `discount_amount` | NUMERIC(6,2) | |
| `redeemed_at` | TIMESTAMP | |
| `last_update` | TIMESTAMP | |

Unique on (`promotion_id`, `rental_id`).

---

## Views

Three analytical views ship in `rental_shop_views.sql`. They are separate from `rental_shop.sql`. Load them with `Sandbox().engine(include="views")` or `EngineContext(include="views")`. The `include` selector is per kind: `"tables"` reflects base tables only (default); `"views"` reflects views only.

| View | Columns | Base tables |
| --- | --- | --- |
| `active_customer_v` | `customer_id`, `store_id`, `first_name`, `last_name`, `email`, `create_date` | `customer` (filter `activebool = true`) |
| `store_revenue_v` | `store_id`, `total_revenue`, `payment_count` | `payment` → `rental` → `customer` (aggregated by store) |
| `film_catalog_v` | `item_id`, `title`, `release_year`, `rating`, `length`, `rental_rate` | `film` → `item` (filter `item_type = 'film'`) |

---

## Bundled notes and named spaces

Hosts may define named AetherSpaces with any `SpaceContext` table subset validated against the schema graph. **Notes are optional** — `SpaceContext(tables={...})` without `notes_file` is valid. The corpus also ships four member-aligned spaces (`storefront`, `catalog`, `logistics`, `crm`) whose table scopes match `federation_partition.json`. The same spaces are used for single-engine and federation demos.

```python
from aetherdialect import Sandbox, SpaceContext

with Sandbox() as sandbox:
    engine = sandbox.engine()
    space = engine.aetherspace(
        "catalog",
        space_context=SpaceContext(
            tables=frozenset({"item", "film", "category", "item_category"}),
        ),
    )
    with engine.session(space=space.uid) as session:
        session.accept_until_done("How many films are in the Horror category?")
```

Owner note files ship with the corpus. Full bundled text:

### `rental_shop_notes.txt` (default space)

```text
Film, book, and game share one catalog spine on item. Title, rental_duration, and rental_rate live there. item_type is film, book, or game and the type tables hang off item_id. People say title and they mean item.title. Film has rating and length. Games have esrb, platform, and developer. Books have author, publisher, and isbn. language_id on the item is the usual language join. Film can also carry original_language_id. Categories are names people care about more than ids, and a title can sit in more than one. Trailers, commentaries, and deleted_scenes are item_feature rows with audio or video, not columns on film. game_supported_language is the dubbing bridge. I forget it until someone asks.

Inventory is a physical copy at a store, not the abstract title. inventory_status_history.status is the copy state with available, rented, damaged, in_repair, lost, and retired. Use the latest change when someone asks if it is available. rental_date is checkout. return_date null means still out. Overdue is still out and past rental_date plus item.rental_duration. payment.amount is money on a rental. There is no FX table. A rental can have more than one payment row. There is no separate late-fee table. Staff credentials like ssn and passwords are hidden and off limits. customer.email is PII-ish. Phone lives on address. Store location is store to address to city to country.

Reservations are holds with pending, fulfilled, expired, or cancelled status, not an active rental. Promotions use promo_type values like clearance, new_member, loyalty_reward, weekday_special, and bundle. promotion_redemption ties a promo to a rental with discount_amount. There is no staff_id on the redemption. damage_report severity is minor, moderate, or severe. Delivery is optional home drop-off through courier. Most rentals never touch it. Procurement is purchase_order and purchase_line to a supplier. stock_transfer moves quantity from warehouse to store. Supplier and courier are vendors, not customers.

ARR in finance chat means annualized rental revenue from payment amounts, even though the rows are daily. NRR means revenue from returning customers, not first-time walk-ins. FY for ops chatter is April through March. That is whiteboard folklore, not a column. SLA in the warehouse corner means pick-to-ship hours. People loosely say under 24h is on-time though nothing encodes that. Deep catalog is merchandising slang for titles with release_year older than about ten years. VIP is informal for heavy promo redeemers, not a column. Churny is also slang. There is no churn table. It usually means no recent rental and no recent redemption. When people say busy Saturday they mean rental_date volume by store, not a staff schedule.
```

### `federation_storefront_notes.txt` (storefront member space)

```text
Customers, staff, stores, rentals, payments, and reservations sit with the address to city to country chain for where people and stores live. ARR for finance chat is annualized rental revenue rolled from payment.amount even though the ledger is daily rows. NRR is the same idea but only returning customers, not first-time walk-ins. A rental with return_date null means the copy is still out. Overdue is still out and past rental_date plus the agreed rental_duration when that duration is known. payment.amount is money. Never invent FX rates. A rental can show more than one payment row. There is no separate late-fee table. customer.email is sensitive. Phone is on address. FY for ops chatter is April through March, not calendar year. That is whiteboard folklore, not a column. Reservations are holds with pending, fulfilled, expired, or cancelled status, not an active rental. Staff credentials like ssn and passwords are off limits. If someone asks where store 2 is they mean store to address to city to country. Busy Saturday means rental_date volume by store, not a staff roster fiction.
```

### `federation_catalog_notes.txt` (catalog member space)

```text
Titles and media metadata live on item as the shared spine for film, book, and game. The type tables hang off item_id. People say title meaning item.title. Film has rating and length. Games use esrb, platform, and developer. Books lean on author, publisher, and isbn. item.language_id is the main language join. film.original_language_id is the original-language hook. ARR means licensed title count growth year over year, not cash revenue. Category names matter more than ids. An item can sit in multiple categories. Inventory rows are physical copies at a store, not the abstract title. Actor hooks through film_actor. Trailers and commentaries are item_feature lines, not film columns. Deep catalogue is merchandising slang for release_year older than about ten years. If someone asks how many films they usually mean film joined to item, not inventory copy counts. City and country may be present but titles do not hang geography off them. Use language for language questions.
```

### `federation_logistics_notes.txt` (logistics member space)

```text
Warehouses and the messy middle. stock_transfer moves item quantity from warehouse to store at transferred_at. purchase_order plus purchase_line is buying from a supplier for a store. received_date plus status received is when goods landed. inventory_status_history is the copy-state truth with available, rented, damaged, in_repair, lost, and retired. That is not a vibe and not a shortened repair label. damage_report is after a bad return with severity minor, moderate, or severe. Delivery and courier are optional home drop-off tied to a rental. Most rentals never touch them. unit_cost and delivery_fee are money. receipts is a payment-shaped ledger with rcpt_id, rent_id, amt, and dt. It is money against a rental, not a goods-received document. SLA means warehouse pick-to-ship hours. People loosely say under 24h is on-time even though nothing codes that rule. Dead stock is slang for copies that sat available forever with no rental buzz. There is no dead_stock table. FY is April through March when ops argues about transfer volume. Suppliers are vendors, not customers.
```

### `federation_crm_notes.txt` (crm member space)

```text
Customers sit with the promo machine. promo_type values in use are clearance, new_member, loyalty_reward, weekday_special, and bundle, with start and end dates and optional discount_pct. promotion_redemption is a promo used on a rental_id with discount_amount. There is no staff column on the redemption. Customer comes through the rental. Different promos can land on the same rental, but the same promotion_id cannot redeem twice on one rental_id. ARR often means active rewarded renters, customers with at least one redemption in the period, not cash revenue from payments. Churny customers are ones with no recent redemption and no rental buzz. There is no churn table. VIP is informal for heavy redeemers, not a column. Promo windows matter more than the promotion_id number when explaining a campaign. customer.email is sensitive. Staff may show up as people who work the floor, not as who keyed the code. If the exact promo_type list is needed, read the data. Do not invent member_special.
```

Use with `SpaceContext(tables=SANDBOX_MEMBER_SPACE_TABLES[<member>], notes_file="federation_<member>_notes.txt")` as shown in [Sandbox guide - Named AetherSpaces](SANDBOX.md#named-aetherspaces).

---

## Structure and sensitivity fixtures

### Hidden versus denied

| Mechanism | Where set | Effect |
| --- | --- | --- |
| `sensitivity: hidden` in structure document | `rental_shop_overrides.json` (corpus build input), `schema_structure_demo.json` (bundled runtime demo) | Column **remains in the schema graph** but is blocked from prompts and validation |
| `deny_columns` on `EngineContext` / `Sandbox().engine(...)` | constructor parameter | Column is **removed from the schema graph** before profiling — the engine does not know it exists |

Two structure fixture files ship with distinct roles.

### `rental_shop_overrides.json` (corpus baseline)

Applied during bundle build. Effects:

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | `sensitivity: hidden` |
| `foreign_keys_add` | Adds join edges not present in raw DDL: `film`/`book`/`game` → `item_category`; `game` → `game_supported_language`; `customer` → `address` → `city` → `country`; `rental` → `inventory` → `item` |

### `schema_structure_demo.json` (runtime demo)

Copy the JSON below and call `apply_structure(document)` on an owner engine — the same production workflow:

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | `sensitivity: hidden` |

Full bundled JSON:

```json
{
  "table_count": 1,
  "tables": [
    {
      "name": "staff",
      "columns": [
        {"name": "ssn", "data_type": "VARCHAR", "sensitivity": "hidden"},
        {"name": "password", "data_type": "VARCHAR", "sensitivity": "hidden"}
      ]
    }
  ],
  "relationships": [],
  "foreign_keys_add": [],
  "foreign_keys_remove": [],
  "primary_keys_add": [],
  "primary_keys_remove": []
}
```

---

## Consumer scopes

`role="consumer"` uses the **same question corpus** as the owner role — see [Question corpus](#question-corpus). Narrow visibility with a plain `EngineContext(allow_objects=...)`; any subset of the owner seed tables is valid, and there is no `create_consumer_*` helper. Questions that reference tables outside the allow list fail at ask or execution with permission or schema refusal. Consumer `allow_objects` is not an AetherSpace permission axis; the full security model is in [Security - Execution boundary](SECURITY.md#2-execution-boundary-and-credentials) and [Integrator guide - Multi-user deployment](INTEGRATOR_GUIDE.md#multi-user-deployment).

### Owner (default)

No `allow_objects` restriction — all **34** base tables from [Single-engine schema](#single-engine-schema) are visible.

### Consumer (`role="consumer"`)

When you call `sandbox.engine(role="consumer")` or pass a consumer `EngineContext` to `Sandbox().engine(...)`, open a **reader** session (`mode="reader"`). Reader sessions keep learning session-local and do **not** enqueue durable write-queue events. Table visibility follows your `allow_objects` set. If you omit `allow_objects`, the consumer sees the full **34**-table owner seed.

### Four documented member scopes

The four bundled federation member table sets ([Federation topology - Members](#members)) double as ready-made consumer `allow_objects` scopes on the **single engine** — no separate helper, just pass the same table set:

```python
from aetherdialect import EngineContext, Sandbox

storefront_scope = EngineContext(
    allow_objects=frozenset(
        {"address", "city", "country", "customer", "payment", "rental", "reservation", "staff", "store"}
    )
)
catalog_scope = EngineContext(
    allow_objects=frozenset(
        {
            "actor",
            "author",
            "book",
            "category",
            "city",
            "country",
            "film",
            "film_actor",
            "game",
            "game_supported_language",
            "inventory",
            "item",
            "item_category",
            "item_feature",
            "language",
            "payment",
            "publisher",
        }
    )
)
logistics_scope = EngineContext(
    allow_objects=frozenset(
        {
            "courier",
            "damage_report",
            "delivery",
            "inventory_status_history",
            "purchase_line",
            "purchase_order",
            "receipts",
            "stock_transfer",
            "supplier",
            "warehouse",
        }
    )
)
crm_scope = EngineContext(
    allow_objects=frozenset({"customer", "promotion", "promotion_redemption", "staff"})
)

with Sandbox() as sandbox:
    engine = sandbox.engine(storefront_scope, role="consumer")
```

Explicit allow frozensets (same values as `SANDBOX_MEMBER_SPACE_TABLES`):

| Member | `allow_objects` |
| --- | --- |
| `storefront` | `frozenset({"address", "city", "country", "customer", "payment", "rental", "reservation", "staff", "store"})` |
| `catalog` | `frozenset({"actor", "author", "book", "category", "city", "country", "film", "film_actor", "game", "game_supported_language", "inventory", "item", "item_category", "item_feature", "language", "payment", "publisher"})` |
| `logistics` | `frozenset({"courier", "damage_report", "delivery", "inventory_status_history", "purchase_line", "purchase_order", "receipts", "stock_transfer", "supplier", "warehouse"})` |
| `crm` | `frozenset({"customer", "promotion", "promotion_redemption", "staff"})` |

Questions referencing tables outside the chosen set (for example asking a `catalog`-scoped consumer about `staff`) fail with permission or schema errors — see [Sandbox guide - Owner vs consumer roles](SANDBOX.md#owner-vs-consumer-roles).

---

## Federation topology

Federation ID: **`sandbox_rental_shop`** (offline). Declaration: `federation_declaration.json`. Table partition map: `federation_partition.json`.

### Members

| Member | Offline engine | Tables |
| --- | --- | --- |
| `storefront` | DuckDB in-memory | `address`, `city`, `country`, `customer`, `payment`, `rental`, `reservation`, `staff`, `store` |
| `catalog` | DuckDB in-memory | `actor`, `author`, `book`, `category`, `city`, `country`, `film`, `film_actor`, `game`, `game_supported_language`, `inventory`, `item`, `item_category`, `item_feature`, `language`, `payment`, `publisher` |
| `logistics` | DuckDB in-memory | `courier`, `damage_report`, `delivery`, `inventory_status_history`, `purchase_line`, `purchase_order`, `receipts`, `stock_transfer`, `supplier`, `warehouse` |
| `crm` | DuckDB in-memory | `customer`, `promotion`, `promotion_redemption`, `staff` |

Offline construction: `Sandbox().federation("sandbox_rental_shop")` loads **all four** members on separate in-memory DuckDB connections with per-member artifact trees. Passing a partial `members=` map (for example only `storefront` and `catalog`) raises `ConfigError` — the sandbox federation is fixed to the full quartet.

### Payment union split

`payment` is present on `storefront` and `catalog`, and logistics contributes historical `receipts` into the same logical table. The declaration models it as a `union` — queries aggregate across those partitions without treating them as a single replica.

### Replica tables

| Logical table | Semantics | Authoritative member | Replica member |
| --- | --- | --- | --- |
| `city` | `replica` | `storefront` | `catalog` (local geography copy for publisher FKs; may drift) |
| `country` | `replica` | `storefront` | `catalog` (local geography copy for publisher FKs; may drift) |
| `customer` | `replica` | `storefront` | `crm` |
| `staff` | `replica` | `storefront` | `crm` (column subset only) |

### CRM `staff` column subset

The `crm` replica of `staff` exposes only:

`staff_id`, `first_name`, `last_name`, `store_id`

Credentials (`password`, `ssn`, `email`, `username`, `address_id`, `active`) never enter the CRM seed.

### Cross-source joins

| Left | Right | Kind | Logical key |
| --- | --- | --- | --- |
| `rental.inventory_id` | `inventory.inventory_id` | inner | `inventory_id` |
| `purchase_line.item_id` | `item.item_id` | inner | `item_id` |
| `delivery.rental_id` | `rental.rental_id` | inner | `rental_id` |
| `promotion_redemption.rental_id` | `rental.rental_id` | inner | `rental_id` |

### Logical columns

| Logical | Members | Role |
| --- | --- | --- |
| `inventory_id` | `rental.inventory_id`, `inventory.inventory_id` | `join_key` |
| `item_id` | `purchase_line.item_id`, `item.item_id` | `join_key` |
| `rental_id` | `delivery.rental_id`, `promotion_redemption.rental_id`, `rental.rental_id` | `join_key` |

### Logical tables

| Logical | Semantics | Members |
| --- | --- | --- |
| `payment` | `union` | `storefront.payment`, `catalog.payment`, `logistics.receipts` |
| `city` | `replica` (authoritative: `storefront`) | `storefront.city`, `catalog.city` |
| `country` | `replica` (authoritative: `storefront`) | `storefront.country`, `catalog.country` |
| `customer` | `replica` (authoritative: `storefront`) | `storefront.customer`, `crm.customer` |
| `staff` | `replica` (authoritative: `storefront`) | `storefront.staff`, `crm.staff` (column subset) |

Member roster fields (`sources`, `table_namespace`) are derived at compose time — do not author them in the declaration.

---

## Question corpus

Bundled question strings ship inside the installed package and are listed in the sections below. Copy questions from the lists below into `session.ask(...)` / `session.accept_until_done(...)` — there is no public method that returns them. Tiers and counts:

| Tier | Count | Use |
| --- | --- | --- |
| `questions` | 50 | General questions on the owner/consumer engine |
| `validation_failures` | 4 | Expected to end in terminal validation errors |
| `feedback_samples` | 1 | Anchor question for the rejection/feedback exercise ([Sandbox guide - Rejections and feedback](SANDBOX.md#rejections-and-feedback)) |
| `views_questions` | 3 | Questions for `include="views"` (not part of the `questions` tier) |

### `questions` tier (50)

- How many items are in the catalog by item type?
- How many books do we have?
- How many games are in the catalog?
- Which films include trailers?
- Which games support English?
- How many open reservations are there?
- Show active staff at each store.
- How many rentals happened in 2025?
- Who are our top 5 customers by total payment?
- Rank films by rental count within each category.
- What is the total delivery fee by courier?
- What is the count of pending reservations by store?
- What's the weather today?
- What is the best pizza topping?
- List all store locations by city.
- What is the average rental duration?
- Which films have the highest replacement cost?
- How many actors are in the database?
- What are the film ratings available?
- Which staff members work at store 1?
- List customers who have never rented an item.
- What is the total revenue by store?
- How many films are in the Horror category?
- Which languages are available?
- Which city has the most customers?
- How many customers are in each country?
- Which actors appear in the most films?
- Which actors have the most film credits?
- Which films are in the Horror category?
- What is the average payment amount?
- How many rentals are currently overdue?
- List films released in 2006.
- Which author has the most books?
- How many inventory items does each store have?
- List publishers with more than five books.
- Show purchase orders still open by supplier.
- How many damage reports are open?
- List inventory status changes in the last 90 days.
- Which warehouse holds the most stock?
- List promotion redemptions by promotion name.
- Show delivery status counts by courier.
- List stock transfers between warehouses this year.
- Which suppliers have the most purchase lines?
- What is the average page count by publisher?
- How many rentals are linked to film titles?
- What is the total payment amount?
- How many distinct customers rented horror films?
- Show payroll deductions grouped by staff member.
- What is the average payment amount grouped by film category?
- List all films in the catalog.

### Member-space question subsets

These are subsets of the `questions` tier that stay in-scope for the matching member-aligned AetherSpace (`session(space=...)`). Every question below remains valid on single-engine default space and on the full four-member federation.

#### Storefront member

- How many open reservations are there?
- Show active staff at each store.
- How many rentals happened in 2025?
- Who are our top 5 customers by total payment?
- What is the count of pending reservations by store?
- List all store locations by city.
- Which staff members work at store 1?
- List customers who have never rented an item.
- What is the total revenue by store?
- Which city has the most customers?
- How many customers are in each country?
- What is the average payment amount?
- How many rentals are currently overdue?

#### Catalog member

- How many items are in the catalog by item type?
- How many books do we have?
- How many games are in the catalog?
- Which films include trailers?
- Which games support English?
- What is the average rental duration?
- Which films have the highest replacement cost?
- How many actors are in the database?
- What are the film ratings available?
- How many films are in the Horror category?
- Which languages are available?
- Which actors appear in the most films?
- Which actors have the most film credits?
- Which films are in the Horror category?
- List films released in 2006.
- Which author has the most books?
- How many inventory items does each store have?
- List publishers with more than five books.
- What is the average page count by publisher?
- List all films in the catalog.

#### Logistics member

- What is the total delivery fee by courier?
- Show purchase orders still open by supplier.
- How many damage reports are open?
- List inventory status changes in the last 90 days.
- Which warehouse holds the most stock?
- Show delivery status counts by courier.
- List stock transfers between warehouses this year.
- Which suppliers have the most purchase lines?

#### CRM member

- List promotion redemptions by promotion name.

### `validation_failures` tier (4)

- Show payroll deductions by employee SSN.
- Show me all staff salaries.
- How many rentals happened on 2025-01-01?
- How many rentals were made in total?

### `feedback_samples` tier (1)

- The intent is wrong, count distinct films only.

### `views_questions` tier (3)

- How many active customers do we have?
- What is the total revenue by store?
- Which films are in the catalog view?

---

## Authoring checklist

The sandbox is closed-world. When authoring scopes or declarations, names must come from this reference:

| What you author | Source in this document |
| --- | --- |
| `EngineContext.allow_objects` table names | [Single-engine schema](#single-engine-schema) (all 34 tables) or [Consumer scopes](#consumer-scopes) |
| Federation member names | [Federation topology - Members](#members) (`storefront`, `catalog`, `logistics`, `crm`) |
| `cross_source_joins`, `logical_tables`, `logical_columns` | [Federation topology](#federation-topology) |
| AetherSpace table subsets | Any subset of the 34 base tables |
| Bundled question wording | [Question corpus](#question-corpus) tiers |
| View names | [Views](#views) (`active_customer_v`, `store_revenue_v`, `film_catalog_v`) |
