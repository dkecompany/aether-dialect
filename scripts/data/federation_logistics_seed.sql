-- Federation logistics partition seed (full partition; legacy receipts and orphan deliveries).

CREATE TABLE warehouse (
    warehouse_id INTEGER NOT NULL,
    warehouse_name VARCHAR(80) NOT NULL,
    address_id INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO warehouse (warehouse_id, warehouse_name, address_id, capacity, last_update) VALUES (1, 'Central Hub', 1, 5000, '2024-01-01 00:00:00');

CREATE TABLE supplier (
    supplier_id INTEGER NOT NULL,
    supplier_name VARCHAR(120) NOT NULL,
    country_id INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO supplier (supplier_id, supplier_name, country_id, is_active, last_update) VALUES (1, 'Northwind Supply', 1, 1, '2024-01-01 00:00:00');

CREATE TABLE courier (
    courier_id INTEGER NOT NULL,
    courier_name VARCHAR(80) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    country_id INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO courier (courier_id, courier_name, phone, country_id, is_active, last_update) VALUES (1, 'Swift Parcel', '555-0100', 1, 1, '2024-01-01 00:00:00');

CREATE TABLE purchase_order (
    ord_id INTEGER NOT NULL,
    sup_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    ord_dt TEXT NOT NULL,
    recv_dt TEXT,
    status VARCHAR(20) NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO purchase_order (ord_id, sup_id, store_id, ord_dt, recv_dt, status, last_update) VALUES (1, 1, 1, '2024-02-01', '2024-02-10', 'received', '2024-02-10 12:00:00');

CREATE TABLE purchase_line (
    line_id INTEGER NOT NULL,
    ord_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    quantity SMALLINT NOT NULL,
    unit_cost NUMERIC(8,2) NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO purchase_line (line_id, ord_id, item_id, quantity, unit_cost, last_update) VALUES (1, 1, 1000, 10, 12.50, '2024-02-10 12:00:00');

CREATE TABLE stock_transfer (
    transfer_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    from_warehouse_id INTEGER NOT NULL,
    to_store_id INTEGER NOT NULL,
    quantity SMALLINT NOT NULL,
    transferred_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO stock_transfer (transfer_id, item_id, from_warehouse_id, to_store_id, quantity, transferred_at, last_update) VALUES (1, 1000, 1, 1, 5, '2024-03-01 09:00:00', '2024-03-01 09:00:00');

CREATE TABLE delivery (
    delivery_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    courier_id INTEGER NOT NULL,
    shipped_at TIMESTAMP NOT NULL,
    delivered_at TIMESTAMP,
    tracking_number VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO delivery (delivery_id, rental_id, courier_id, shipped_at, delivered_at, tracking_number, status, last_update) VALUES (1, 1, 1, '2024-03-22 10:00:00', '2024-03-24 14:00:00', 'TRK-001', 'delivered', '2024-03-24 14:00:00');
INSERT INTO delivery (delivery_id, rental_id, courier_id, shipped_at, delivered_at, tracking_number, status, last_update) VALUES (9001, 9999001, 1, '2024-06-01 10:00:00', NULL, 'ORPHAN-9999001', 'shipped', '2024-06-01 10:00:00');
INSERT INTO delivery (delivery_id, rental_id, courier_id, shipped_at, delivered_at, tracking_number, status, last_update) VALUES (9002, 9999002, 1, '2024-06-01 10:00:00', NULL, 'ORPHAN-9999002', 'shipped', '2024-06-01 10:00:00');
INSERT INTO delivery (delivery_id, rental_id, courier_id, shipped_at, delivered_at, tracking_number, status, last_update) VALUES (9003, 9999003, 1, '2024-06-01 10:00:00', NULL, 'ORPHAN-9999003', 'shipped', '2024-06-01 10:00:00');

CREATE TABLE damage_report (
    report_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    reported_at TIMESTAMP NOT NULL,
    damage_type VARCHAR(40) NOT NULL,
    repair_cost NUMERIC(8,2),
    notes TEXT,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO damage_report (report_id, rental_id, reported_at, damage_type, repair_cost, notes, last_update) VALUES (1, 1, '2024-03-25 11:00:00', 'cosmetic', 5.00, 'minor wear', '2024-03-25 11:00:00');

CREATE TABLE inventory_status_history (
    history_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,
    changed_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO inventory_status_history (history_id, inventory_id, status, changed_at, last_update) VALUES (1, 1, 'available', '2024-01-01 00:00:00', '2024-01-01 00:00:00');

CREATE TABLE receipts (
    rcpt_id INTEGER NOT NULL,
    rent_id INTEGER NOT NULL,
    amt REAL NOT NULL,
    dt TEXT NOT NULL
);;
INSERT INTO receipts (rcpt_id, rent_id, amt, dt) VALUES (3, 3, 3.24, '2022-02-16 03:19:17');
INSERT INTO receipts (rcpt_id, rent_id, amt, dt) VALUES (6, 6, 2.50, '2020-05-02 22:36:52');
