-- Federation logistics partition schema (CREATE-only; rows load from federation_logistics_data).

CREATE TABLE "courier" (

    courier_id INTEGER NOT NULL,
    courier_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    country_id INTEGER NOT NULL,
    is_active INTEGER,
    last_update TIMESTAMP

);

CREATE TABLE "damage_report" (

    damage_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    reported_by_staff_id INTEGER NOT NULL,
    severity TEXT NOT NULL,
    repair_cost REAL NOT NULL,
    reported_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "delivery" (

    delivery_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    courier_id INTEGER NOT NULL,
    address_id INTEGER NOT NULL,
    dispatched_at TIMESTAMP NOT NULL,
    delivered_at TIMESTAMP,
    status TEXT NOT NULL,
    delivery_fee REAL NOT NULL,
    tracking_number TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "inventory_status_history" (

    status_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    changed_at TIMESTAMP NOT NULL,
    staff_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "stock_transfer" (

    transfer_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    from_warehouse_id INTEGER NOT NULL,
    to_store_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    transferred_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "supplier" (

    supplier_id INTEGER NOT NULL,
    supplier_name TEXT NOT NULL,
    country_id INTEGER NOT NULL,
    is_active INTEGER,
    last_update TIMESTAMP

);

CREATE TABLE "warehouse" (

    warehouse_id INTEGER NOT NULL,
    warehouse_name TEXT NOT NULL,
    address_id INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE purchase_order (ord_id INTEGER NOT NULL, sup_id INTEGER NOT NULL, store_id INTEGER NOT NULL, ord_dt TEXT NOT NULL, recv_dt TEXT, status VARCHAR(20) NOT NULL, last_update TIMESTAMP NOT NULL);

CREATE TABLE purchase_line (line_id INTEGER NOT NULL, ord_id INTEGER NOT NULL, item_id INTEGER NOT NULL, quantity SMALLINT NOT NULL, unit_cost NUMERIC(8,2) NOT NULL, last_update TIMESTAMP NOT NULL);

CREATE TABLE receipts (rcpt_id INTEGER NOT NULL, rent_id INTEGER NOT NULL, amt REAL NOT NULL, dt TIMESTAMP NOT NULL);
