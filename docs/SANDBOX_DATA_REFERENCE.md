# Sandbox data reference

Closed-world inventory of every table, column, view, federation member, and practice question shipped in the **rental_shop** sandbox bundle. Use this document when authoring `EngineContext(allow_objects=...)`, federation declarations, or AetherSpace scopes. For session recipes and API patterns, see [Sandbox guide](SANDBOX.md).

**Reading order:** [README](../README.md) → [Sandbox guide](SANDBOX.md) → this document.

## Sections

| Section | Contents |
| --- | --- |
| [Overview](#overview) | Table and view counts, date range, closed-world rule |
| [Single-engine schema](#single-engine-schema) | All 34 tables with columns, keys, and foreign keys |
| [Views](#views) | Bundled analytical views and `include="views"` |
| [Bundled notes](#bundled-notes) | Owner and catalog notes emphasis |
| [Overrides and sensitivity fixtures](#overrides-and-sensitivity-fixtures) | Hidden vs denied columns |
| [Consumer scopes](#consumer-scopes) | Owner, standard consumer, and restricted allow lists |
| [Federation topology](#federation-topology) | Members, replicas, joins, and logical tables |
| [Offline versus live](#offline-versus-live) | In-memory vs warehouse namespaces |
| [Question corpus](#question-corpus) | Practice question tiers |

---

## Overview

| Property | Value |
| --- | --- |
| Base tables | **34** (`rental_shop.sql`) |
| Analytical views | **3** (`rental_shop_views.sql`; loaded when `include="views"`) |
| DuckDB schema | `main` (default in-memory connection) |
| Synthetic activity dates | Predominantly **2022–2025** (some seed rows span earlier years) |
| Closed-world rule | Every table, column, member name, and practice question in this document is namable in the sandbox |

The single-engine owner seed loads all 34 tables from `rental_shop.sql` into one in-memory DuckDB database. Views are defined separately and are not part of the base seed DDL.

---

## Single-engine schema

Tables are grouped by business area. Column types match `rental_shop.sql`. Primary keys (PK) and foreign keys (FK) list the referenced table.

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

Three analytical views ship in `rental_shop_views.sql`. They are **not** in `rental_shop.sql`. Load them with `AetherEngine.offline_sandbox(include="views")` or `EngineContext(include="views")`. The `include` selector is per kind: `"tables"` reflects base tables only (default); `"views"` reflects views only.

| View | Columns | Base tables |
| --- | --- | --- |
| `active_customer_v` | `customer_id`, `store_id`, `first_name`, `last_name`, `email`, `create_date` | `customer` (filter `activebool = true`) |
| `store_revenue_v` | `store_id`, `total_revenue`, `payment_count` | `payment` → `rental` → `customer` (aggregated by store) |
| `film_catalog_v` | `item_id`, `title`, `release_year`, `rating`, `length`, `rental_rate` | `film` → `item` (filter `item_type = 'film'`) |

---

## Bundled notes

Two owner note files ship with the corpus.

### `rental_shop_notes.txt` (default master space)

| Topic | Tables / columns emphasised |
| --- | --- |
| Unified catalog | `item`, `film`, `book`, `game`, `item_feature` |
| Rental lifecycle | `rental.rental_date`, `rental.return_date`, `item.rental_duration` |
| Inventory status | `inventory`, `inventory_status_history` |
| Payments | `payment` linked to `rental` |
| Staff privacy | `staff.password`, `staff.ssn` — must never appear in analysis |
| Logistics | `delivery`, `courier`, `damage_report` |
| Procurement | `purchase_order`, `purchase_line`, `stock_transfer`, `warehouse` |
| Promotions | `promotion`, `promotion_redemption` |

### `sandbox_space_catalog_notes.txt` (second-space demo)

| Topic | Tables / columns emphasised |
| --- | --- |
| Media catalog | `item`, `film`, `category`, `item_category` |
| Classification | category names and media titles over numeric IDs |
| Scope boundary | media attributes only — not transactions, revenue, or logistics |

Use with `SpaceContext(tables={...}, notes_file="sandbox_space_catalog_notes.txt")` as shown in [Sandbox guide - Named AetherSpaces](SANDBOX.md#named-aetherspaces).

---

## Overrides and sensitivity fixtures

Two override files ship with distinct roles.

### `rental_shop_overrides.json` (corpus baseline)

Applied during bundle build. Effects:

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | `sensitivity: hidden` |
| `foreign_keys_add` | Adds join edges not present in raw DDL: `film`/`book`/`game` → `item_category`; `game` → `game_supported_language`; `customer` → `address` → `city` → `country`; `rental` → `inventory` → `item` |

### `sandbox_overrides_demo.json` (runtime demo)

Applied via `apply_bundled_schema_overrides()`:

| Target | Effect |
| --- | --- |
| `staff.ssn`, `staff.password` | `sensitivity: hidden` |
| `film` | analyst description override |

### Hidden versus denied

| Mechanism | Where set | Effect |
| --- | --- | --- |
| `sensitivity: hidden` in overrides | `rental_shop_overrides.json`, `sandbox_overrides_demo.json` | Column **remains in the schema graph** but is blocked from prompts and validation |
| `deny_columns` on `EngineContext` / `offline_sandbox(...)` | constructor parameter | Column is **removed from the schema graph** before profiling — the engine does not know it exists |

---

## Consumer scopes

Exact table allow lists used by the sandbox consumer role.

### Owner (default)

No `allow_objects` restriction — all **34** base tables from section 2 are visible.

### Standard consumer (default when `role="consumer"`)

When you call `sandbox.engine(role="consumer")` without narrowing `EngineContext.allow_objects`, execution scope is set to this list:

`actor`, `address`, `author`, `book`, `category`, `city`, `country`, `courier`, `customer`, `damage_report`, `delivery`, `film`, `film_actor`, `game`, `game_supported_language`, `inventory`, `inventory_status_history`, `item`, `item_category`, `item_feature`, `language`, `payment`, `promotion`, `promotion_redemption`, `publisher`, `purchase_line`, `purchase_order`, `rental`, `reservation`, `staff`, `stock_transfer`, `store`, `supplier`, `warehouse`

**This list is identical to the full owner seed (34 tables).** The consumer preset today changes role, session mode (`reader`), and learning behaviour (write-queue enqueue) but **not** table visibility. Pass an explicit `EngineContext(allow_objects=...)` to narrow scope.

### Restricted consumer (permission-denied exercise)

When `role="consumer"` **and** `allow_objects` is exactly this six-table set, the sandbox activates the restricted consumer path:

`customer`, `payment`, `rental`, `address`, `city`, `country`

Questions referencing tables outside this set (for example `staff`) should fail with permission or schema errors. See [Sandbox guide - Owner vs consumer roles](SANDBOX.md#owner-vs-consumer-roles).

---

## Federation topology

Federation ID: **`sandbox_rental_shop`** (offline). Declaration: `federation_declaration.json`. Table partition map: `federation_partition.json`.

### Members

| Member | Offline engine | Tables |
| --- | --- | --- |
| `storefront` | DuckDB in-memory | `address`, `city`, `country`, `customer`, `payment`, `rental`, `reservation`, `staff`, `store` |
| `catalog` | DuckDB in-memory | `actor`, `author`, `book`, `category`, `film`, `film_actor`, `game`, `game_supported_language`, `inventory`, `item`, `item_category`, `item_feature`, `language`, `payment`, `publisher` |
| `logistics` | DuckDB in-memory | `courier`, `damage_report`, `delivery`, `inventory_status_history`, `purchase_line`, `purchase_order`, `stock_transfer`, `supplier`, `warehouse` |
| `crm` | DuckDB in-memory | `customer`, `promotion`, `promotion_redemption`, `staff` |

Offline construction: `Sandbox().federation("sandbox_rental_shop")` loads four separate in-memory DuckDB connections with per-member artifact trees.

### Payment union split

`payment` is the only table present on **two** members (`storefront` and `catalog`). The declaration models it as a `union` logical table — queries aggregate across both partitions without double-counting rows that exist on only one member.

### Replica tables

| Logical table | Semantics | Authoritative member | Replica member |
| --- | --- | --- | --- |
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
| `payment` | `union` | `storefront.payment`, `catalog.payment` |
| `customer` | `replica` (authoritative: `storefront`) | `storefront.customer`, `crm.customer` |
| `staff` | `replica` (authoritative: `storefront`) | `storefront.staff`, `crm.staff` (column subset) |

Member roster fields (`sources`, `table_namespace`) are derived at compose time — do not author them in the declaration.

---

## Offline versus live

The same four-member topology runs offline (bundled DuckDB seeds) and in live integration tests (real database engines). Logical table and join declarations are shared; only the backing engines and namespace names differ.

| Member | Offline (`sandbox_rental_shop`) | Live (`live_rental_shop`) |
| --- | --- | --- |
| `storefront` | DuckDB `:memory:` | PostgreSQL schema `rental_shop_fed_storefront` |
| `catalog` | DuckDB `:memory:` | MySQL database `rental_shop_fed_catalog` |
| `logistics` | DuckDB `:memory:` | PostgreSQL schema `rental_shop_fed_logistics` |
| `crm` | DuckDB `:memory:` | MariaDB database `rental_shop_fed_crm` |

| Property | Offline | Live |
| --- | --- | --- |
| Federation ID | `sandbox_rental_shop` | `live_rental_shop` |
| Declaration file | `federation_declaration.json` | `live_tests/fixtures/federation_live_declaration.json` |
| Member source names | `storefront`, `catalog`, `logistics`, `crm` | same |
| Logical tables / joins | identical shape | identical shape |
| Partition table lists | `federation_partition.json` | same map, loaded into real engines |

Offline partition seed files (`federation_*_seed.sql`) may ship as payment-only placeholders until a full corpus rebuild; `federation_partition.json` always lists the authoritative table roster per member.

---

## Question corpus

Practice strings live in `sandbox_questions.txt` inside the bundled `data.zip`. Tiers and counts:

| Tier | Count | Returned by |
| --- | --- | --- |
| `questions` | 50 | `AetherEngine.sandbox_questions()` |
| `validation_failures` | 4 | `AetherEngine.sandbox_validation_failure_demo()` |
| `feedback_samples` | 1 | `AetherEngine.sandbox_feedback_demo()` |
| `views_questions` | 3 | **Not** returned by `sandbox_questions()` — use when practising with `include="views"` |

### `questions` tier (50)

How many items are in the catalog by item type? · How many books do we have? · How many games are in the catalog? · Which films include trailers? · Which games support English? · How many open reservations are there? · Show active staff at each store. · How many rentals happened in 2025? · Who are our top 5 customers by total payment? · Rank films by rental count within each category. · What is the total delivery fee by courier? · What is the count of pending reservations by store? · What's the weather today? · What is the best pizza topping? · List all store locations by city. · What is the average rental duration? · Which films have the highest replacement cost? · How many actors are in the database? · What are the film ratings available? · Which staff members work at store 1? · List customers who have never rented an item. · What is the total revenue by store? · How many films are in the Horror category? · Which languages are available? · Which city has the most customers? · How many customers are in each country? · Which actors appear in the most films? · Which actors have the most film credits? · Which films are in the Horror category? · What is the average payment amount? · How many rentals are currently overdue? · List films released in 2006. · Which author has the most books? · How many inventory items does each store have? · List publishers with more than five books. · Show purchase orders still open by supplier. · How many damage reports are open? · List inventory status changes in the last 90 days. · Which warehouse holds the most stock? · List promotion redemptions by promotion name. · Show delivery status counts by courier. · List stock transfers between warehouses this year. · Which suppliers have the most purchase lines? · What is the average page count by publisher? · How many rentals are linked to film titles? · What is the total payment amount? · How many distinct customers rented horror films? · Show payroll deductions grouped by staff member. · What is the average payment amount grouped by film category? · List all films in the catalog.

### `validation_failures` tier (4)

Show payroll deductions by employee SSN. · Show me all staff salaries. · How many rentals happened on 2025-01-01? · How many rentals were made in total?

### `feedback_samples` tier (1)

The intent is wrong, count distinct films only.

### `views_questions` tier (3)

How many active customers do we have? · What is the total revenue by store? · Which films are in the catalog view?

---

## Authoring checklist

The sandbox is closed-world. When authoring scopes or declarations, names must come from this reference:

| What you author | Source in this document |
| --- | --- |
| `EngineContext.allow_objects` table names | [Single-engine schema](#single-engine-schema) (all 34 tables) or [Consumer scopes](#consumer-scopes) |
| Federation member names | [Federation topology - Members](#members) (`storefront`, `catalog`, `logistics`, `crm`) |
| `cross_source_joins`, `logical_tables`, `logical_columns` | [Federation topology](#federation-topology) |
| AetherSpace table subsets | Any subset of the 34 base tables |
| Practice question wording | [Question corpus](#question-corpus) tiers |
| View names | [Views](#views) (`active_customer_v`, `store_revenue_v`, `film_catalog_v`) |

---

**See also:** [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Security](SECURITY.md) | [README](../README.md)
