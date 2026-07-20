CREATE TABLE country (
    country_id INTEGER NOT NULL,
    country VARCHAR(50) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE country ADD CONSTRAINT country_pkey PRIMARY KEY (country_id);
ALTER TABLE country ADD CONSTRAINT country_country_key UNIQUE (country);

CREATE TABLE city (
    city_id INTEGER NOT NULL,
    city VARCHAR(50) NOT NULL,
    country_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE city ADD CONSTRAINT city_pkey PRIMARY KEY (city_id);
ALTER TABLE city ADD CONSTRAINT city_country_id_fkey FOREIGN KEY (country_id) REFERENCES country (country_id);

CREATE TABLE address (
    address_id INTEGER NOT NULL,
    address VARCHAR(50) NOT NULL,
    district VARCHAR(20) NOT NULL,
    city_id INTEGER NOT NULL,
    postal_code VARCHAR(10),
    phone VARCHAR(20) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE address ADD CONSTRAINT address_pkey PRIMARY KEY (address_id);
ALTER TABLE address ADD CONSTRAINT address_city_id_fkey FOREIGN KEY (city_id) REFERENCES city (city_id);

CREATE TABLE language (
    language_id INTEGER NOT NULL,
    name CHAR(20) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE language ADD CONSTRAINT language_pkey PRIMARY KEY (language_id);
ALTER TABLE language ADD CONSTRAINT language_name_key UNIQUE (name);

CREATE TABLE actor (
    actor_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE actor ADD CONSTRAINT actor_pkey PRIMARY KEY (actor_id);

CREATE TABLE category (
    category_id INTEGER NOT NULL,
    name VARCHAR(25) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE category ADD CONSTRAINT category_pkey PRIMARY KEY (category_id);
ALTER TABLE category ADD CONSTRAINT category_name_key UNIQUE (name);

CREATE TABLE author (
    author_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE author ADD CONSTRAINT author_pkey PRIMARY KEY (author_id);

CREATE TABLE publisher (
    publisher_id INTEGER NOT NULL,
    publisher_name VARCHAR(120) NOT NULL,
    country_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE publisher ADD CONSTRAINT publisher_pkey PRIMARY KEY (publisher_id);
ALTER TABLE publisher ADD CONSTRAINT publisher_country_id_fkey FOREIGN KEY (country_id) REFERENCES country (country_id);

CREATE TABLE item (
    item_id INTEGER NOT NULL,
    item_type VARCHAR(10) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    release_year INTEGER,
    language_id INTEGER NOT NULL,
    rental_duration SMALLINT DEFAULT 3 NOT NULL,
    rental_rate NUMERIC(4,2) DEFAULT 4.99 NOT NULL,
    replacement_cost NUMERIC(5,2) DEFAULT 19.99 NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE item ADD CONSTRAINT item_pkey PRIMARY KEY (item_id);
ALTER TABLE item ADD CONSTRAINT item_language_id_fkey FOREIGN KEY (language_id) REFERENCES language (language_id);
ALTER TABLE item ADD CONSTRAINT item_item_type_check CHECK (item_type IN ('film', 'book', 'game'));

CREATE TABLE film (
    item_id INTEGER NOT NULL,
    original_language_id INTEGER,
    length SMALLINT,
    rating VARCHAR(10) DEFAULT 'G',
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE film ADD CONSTRAINT film_pkey PRIMARY KEY (item_id);
ALTER TABLE film ADD CONSTRAINT film_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE film ADD CONSTRAINT film_original_language_id_fkey FOREIGN KEY (original_language_id) REFERENCES language (language_id);

CREATE TABLE book (
    item_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    publisher_id INTEGER NOT NULL,
    isbn VARCHAR(20) NOT NULL,
    page_count SMALLINT NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE book ADD CONSTRAINT book_pkey PRIMARY KEY (item_id);
ALTER TABLE book ADD CONSTRAINT book_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE book ADD CONSTRAINT book_author_id_fkey FOREIGN KEY (author_id) REFERENCES author (author_id);
ALTER TABLE book ADD CONSTRAINT book_publisher_id_fkey FOREIGN KEY (publisher_id) REFERENCES publisher (publisher_id);
ALTER TABLE book ADD CONSTRAINT book_isbn_key UNIQUE (isbn);

CREATE TABLE game (
    item_id INTEGER NOT NULL,
    platform VARCHAR(30) NOT NULL,
    developer VARCHAR(80) NOT NULL,
    esrb_rating VARCHAR(10) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE game ADD CONSTRAINT game_pkey PRIMARY KEY (item_id);
ALTER TABLE game ADD CONSTRAINT game_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);

CREATE TABLE item_category (
    item_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE item_category ADD CONSTRAINT item_category_pkey PRIMARY KEY (item_id, category_id);
ALTER TABLE item_category ADD CONSTRAINT item_category_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE item_category ADD CONSTRAINT item_category_category_id_fkey FOREIGN KEY (category_id) REFERENCES category (category_id);

CREATE TABLE film_actor (
    actor_id INTEGER NOT NULL,
    film_item_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE film_actor ADD CONSTRAINT film_actor_pkey PRIMARY KEY (actor_id, film_item_id);
ALTER TABLE film_actor ADD CONSTRAINT film_actor_actor_id_fkey FOREIGN KEY (actor_id) REFERENCES actor (actor_id);
ALTER TABLE film_actor ADD CONSTRAINT film_actor_film_item_id_fkey FOREIGN KEY (film_item_id) REFERENCES film (item_id);

CREATE TABLE item_feature (
    item_id INTEGER NOT NULL,
    feature_name VARCHAR(80) NOT NULL,
    feature_type VARCHAR(30) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE item_feature ADD CONSTRAINT item_feature_pkey PRIMARY KEY (item_id, feature_name);
ALTER TABLE item_feature ADD CONSTRAINT item_feature_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);

CREATE TABLE game_supported_language (
    item_id INTEGER NOT NULL,
    language_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE game_supported_language ADD CONSTRAINT game_supported_language_pkey PRIMARY KEY (item_id, language_id);
ALTER TABLE game_supported_language ADD CONSTRAINT game_supported_language_item_id_fkey FOREIGN KEY (item_id) REFERENCES game (item_id);
ALTER TABLE game_supported_language ADD CONSTRAINT game_supported_language_language_id_fkey FOREIGN KEY (language_id) REFERENCES language (language_id);

CREATE TABLE store (
    store_id INTEGER NOT NULL,
    manager_staff_id INTEGER,
    address_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE store ADD CONSTRAINT store_pkey PRIMARY KEY (store_id);
ALTER TABLE store ADD CONSTRAINT store_address_id_fkey FOREIGN KEY (address_id) REFERENCES address (address_id);

CREATE TABLE staff (
    staff_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    address_id INTEGER NOT NULL,
    email VARCHAR(50),
    store_id INTEGER NOT NULL,
    active BOOLEAN DEFAULT true NOT NULL,
    username VARCHAR(16) NOT NULL,
    password VARCHAR(40),
    ssn VARCHAR(11),
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE staff ADD CONSTRAINT staff_pkey PRIMARY KEY (staff_id);
ALTER TABLE staff ADD CONSTRAINT staff_address_id_fkey FOREIGN KEY (address_id) REFERENCES address (address_id);
ALTER TABLE staff ADD CONSTRAINT staff_store_id_fkey FOREIGN KEY (store_id) REFERENCES store (store_id);
ALTER TABLE staff ADD CONSTRAINT staff_username_key UNIQUE (username);

ALTER TABLE store ADD CONSTRAINT store_manager_staff_id_fkey FOREIGN KEY (manager_staff_id) REFERENCES staff (staff_id);

CREATE TABLE inventory (
    inventory_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE inventory ADD CONSTRAINT inventory_pkey PRIMARY KEY (inventory_id);
ALTER TABLE inventory ADD CONSTRAINT inventory_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE inventory ADD CONSTRAINT inventory_store_id_fkey FOREIGN KEY (store_id) REFERENCES store (store_id);

CREATE TABLE customer (
    customer_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    email VARCHAR(50),
    address_id INTEGER NOT NULL,
    activebool BOOLEAN DEFAULT true NOT NULL,
    create_date DATE DEFAULT CURRENT_DATE NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE customer ADD CONSTRAINT customer_pkey PRIMARY KEY (customer_id);
ALTER TABLE customer ADD CONSTRAINT customer_store_id_fkey FOREIGN KEY (store_id) REFERENCES store (store_id);
ALTER TABLE customer ADD CONSTRAINT customer_address_id_fkey FOREIGN KEY (address_id) REFERENCES address (address_id);

-- Rental lifecycle: rental_date = checkout; return_date = actual return (NULL = still out).
-- Overdue: return_date IS NULL AND rental_date + item.rental_duration < CURRENT_DATE.
CREATE TABLE rental (
    rental_id INTEGER NOT NULL,
    rental_date TIMESTAMP NOT NULL,
    inventory_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    return_date TIMESTAMP,
    staff_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE rental ADD CONSTRAINT rental_pkey PRIMARY KEY (rental_id);
ALTER TABLE rental ADD CONSTRAINT rental_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES inventory (inventory_id);
ALTER TABLE rental ADD CONSTRAINT rental_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES customer (customer_id);
ALTER TABLE rental ADD CONSTRAINT rental_staff_id_fkey FOREIGN KEY (staff_id) REFERENCES staff (staff_id);
ALTER TABLE rental ADD CONSTRAINT rental_return_date_check CHECK (return_date IS NULL OR return_date >= rental_date);

CREATE TABLE payment (
    payment_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    amount NUMERIC(5,2) NOT NULL,
    payment_date TIMESTAMP NOT NULL
);

ALTER TABLE payment ADD CONSTRAINT payment_pkey PRIMARY KEY (payment_id);
ALTER TABLE payment ADD CONSTRAINT payment_rental_id_fkey FOREIGN KEY (rental_id) REFERENCES rental (rental_id);
ALTER TABLE payment ADD CONSTRAINT payment_amount_check CHECK (amount >= 0);

CREATE TABLE promotion (
    promotion_id INTEGER NOT NULL,
    promo_name VARCHAR(120) NOT NULL,
    promo_type VARCHAR(30) NOT NULL,
    discount_pct NUMERIC(5,2),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE promotion ADD CONSTRAINT promotion_pkey PRIMARY KEY (promotion_id);
ALTER TABLE promotion ADD CONSTRAINT promotion_discount_pct_check CHECK (discount_pct IS NULL OR (discount_pct >= 0 AND discount_pct <= 100));
ALTER TABLE promotion ADD CONSTRAINT promotion_date_check CHECK (end_date >= start_date);

CREATE TABLE promotion_redemption (
    redemption_id INTEGER NOT NULL,
    promotion_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    discount_amount NUMERIC(6,2) NOT NULL,
    redeemed_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE promotion_redemption ADD CONSTRAINT promotion_redemption_pkey PRIMARY KEY (redemption_id);
ALTER TABLE promotion_redemption ADD CONSTRAINT promotion_redemption_promotion_id_fkey FOREIGN KEY (promotion_id) REFERENCES promotion (promotion_id);
ALTER TABLE promotion_redemption ADD CONSTRAINT promotion_redemption_rental_id_fkey FOREIGN KEY (rental_id) REFERENCES rental (rental_id);
ALTER TABLE promotion_redemption ADD CONSTRAINT promotion_redemption_promotion_rental_key UNIQUE (promotion_id, rental_id);
ALTER TABLE promotion_redemption ADD CONSTRAINT promotion_redemption_discount_amount_check CHECK (discount_amount >= 0);

CREATE TABLE reservation (
    reservation_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    reserved_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    fulfilled_rental_id INTEGER,
    status VARCHAR(20) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE reservation ADD CONSTRAINT reservation_pkey PRIMARY KEY (reservation_id);
ALTER TABLE reservation ADD CONSTRAINT reservation_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES customer (customer_id);
ALTER TABLE reservation ADD CONSTRAINT reservation_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE reservation ADD CONSTRAINT reservation_store_id_fkey FOREIGN KEY (store_id) REFERENCES store (store_id);
ALTER TABLE reservation ADD CONSTRAINT reservation_fulfilled_rental_id_fkey FOREIGN KEY (fulfilled_rental_id) REFERENCES rental (rental_id);
ALTER TABLE reservation ADD CONSTRAINT reservation_status_check CHECK (status IN ('pending', 'fulfilled', 'expired', 'cancelled'));
ALTER TABLE reservation ADD CONSTRAINT reservation_expires_at_check CHECK (expires_at > reserved_at);

CREATE TABLE courier (
    courier_id INTEGER NOT NULL,
    courier_name VARCHAR(80) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    country_id INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE courier ADD CONSTRAINT courier_pkey PRIMARY KEY (courier_id);
ALTER TABLE courier ADD CONSTRAINT courier_country_id_fkey FOREIGN KEY (country_id) REFERENCES country (country_id);

CREATE TABLE supplier (
    supplier_id INTEGER NOT NULL,
    supplier_name VARCHAR(120) NOT NULL,
    country_id INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE supplier ADD CONSTRAINT supplier_pkey PRIMARY KEY (supplier_id);
ALTER TABLE supplier ADD CONSTRAINT supplier_country_id_fkey FOREIGN KEY (country_id) REFERENCES country (country_id);
ALTER TABLE supplier ADD CONSTRAINT supplier_supplier_name_key UNIQUE (supplier_name);

CREATE TABLE warehouse (
    warehouse_id INTEGER NOT NULL,
    warehouse_name VARCHAR(80) NOT NULL,
    address_id INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE warehouse ADD CONSTRAINT warehouse_pkey PRIMARY KEY (warehouse_id);
ALTER TABLE warehouse ADD CONSTRAINT warehouse_address_id_fkey FOREIGN KEY (address_id) REFERENCES address (address_id);
ALTER TABLE warehouse ADD CONSTRAINT warehouse_warehouse_name_key UNIQUE (warehouse_name);
ALTER TABLE warehouse ADD CONSTRAINT warehouse_capacity_check CHECK (capacity > 0);

CREATE TABLE purchase_order (
    po_id INTEGER NOT NULL,
    supplier_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    ordered_date DATE NOT NULL,
    received_date DATE,
    status VARCHAR(20) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE purchase_order ADD CONSTRAINT purchase_order_pkey PRIMARY KEY (po_id);
ALTER TABLE purchase_order ADD CONSTRAINT purchase_order_supplier_id_fkey FOREIGN KEY (supplier_id) REFERENCES supplier (supplier_id);
ALTER TABLE purchase_order ADD CONSTRAINT purchase_order_store_id_fkey FOREIGN KEY (store_id) REFERENCES store (store_id);
ALTER TABLE purchase_order ADD CONSTRAINT purchase_order_status_check CHECK (status IN ('open', 'received', 'cancelled'));
ALTER TABLE purchase_order ADD CONSTRAINT purchase_order_received_date_check CHECK (received_date IS NULL OR received_date >= ordered_date);

CREATE TABLE purchase_line (
    line_id INTEGER NOT NULL,
    po_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    quantity SMALLINT NOT NULL,
    unit_cost NUMERIC(8,2) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_pkey PRIMARY KEY (line_id);
ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_po_id_fkey FOREIGN KEY (po_id) REFERENCES purchase_order (po_id);
ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_po_item_key UNIQUE (po_id, item_id);
ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_quantity_check CHECK (quantity > 0);
ALTER TABLE purchase_line ADD CONSTRAINT purchase_line_unit_cost_check CHECK (unit_cost >= 0);

CREATE TABLE stock_transfer (
    transfer_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    from_warehouse_id INTEGER NOT NULL,
    to_store_id INTEGER NOT NULL,
    quantity SMALLINT NOT NULL,
    transferred_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE stock_transfer ADD CONSTRAINT stock_transfer_pkey PRIMARY KEY (transfer_id);
ALTER TABLE stock_transfer ADD CONSTRAINT stock_transfer_item_id_fkey FOREIGN KEY (item_id) REFERENCES item (item_id);
ALTER TABLE stock_transfer ADD CONSTRAINT stock_transfer_from_warehouse_id_fkey FOREIGN KEY (from_warehouse_id) REFERENCES warehouse (warehouse_id);
ALTER TABLE stock_transfer ADD CONSTRAINT stock_transfer_to_store_id_fkey FOREIGN KEY (to_store_id) REFERENCES store (store_id);
ALTER TABLE stock_transfer ADD CONSTRAINT stock_transfer_quantity_check CHECK (quantity > 0);

CREATE TABLE delivery (
    delivery_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    courier_id INTEGER NOT NULL,
    address_id INTEGER NOT NULL,
    dispatched_at TIMESTAMP NOT NULL,
    delivered_at TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    delivery_fee NUMERIC(6,2) NOT NULL,
    tracking_number VARCHAR(30) NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE delivery ADD CONSTRAINT delivery_pkey PRIMARY KEY (delivery_id);
ALTER TABLE delivery ADD CONSTRAINT delivery_rental_id_fkey FOREIGN KEY (rental_id) REFERENCES rental (rental_id);
ALTER TABLE delivery ADD CONSTRAINT delivery_courier_id_fkey FOREIGN KEY (courier_id) REFERENCES courier (courier_id);
ALTER TABLE delivery ADD CONSTRAINT delivery_address_id_fkey FOREIGN KEY (address_id) REFERENCES address (address_id);
ALTER TABLE delivery ADD CONSTRAINT delivery_tracking_number_key UNIQUE (tracking_number);
ALTER TABLE delivery ADD CONSTRAINT delivery_status_check CHECK (status IN ('dispatched', 'in_transit', 'delivered', 'failed', 'returned'));
ALTER TABLE delivery ADD CONSTRAINT delivery_delivered_at_check CHECK (delivered_at IS NULL OR delivered_at >= dispatched_at);
ALTER TABLE delivery ADD CONSTRAINT delivery_fee_check CHECK (delivery_fee >= 0);

CREATE TABLE inventory_status_history (
    status_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,
    changed_at TIMESTAMP NOT NULL,
    staff_id INTEGER NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE inventory_status_history ADD CONSTRAINT inventory_status_history_pkey PRIMARY KEY (status_id);
ALTER TABLE inventory_status_history ADD CONSTRAINT inventory_status_history_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES inventory (inventory_id);
ALTER TABLE inventory_status_history ADD CONSTRAINT inventory_status_history_staff_id_fkey FOREIGN KEY (staff_id) REFERENCES staff (staff_id);
ALTER TABLE inventory_status_history ADD CONSTRAINT inventory_status_history_status_check CHECK (status IN ('available', 'rented', 'damaged', 'in_repair', 'lost', 'retired'));

CREATE TABLE damage_report (
    damage_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    reported_by_staff_id INTEGER NOT NULL,
    severity VARCHAR(20) NOT NULL,
    repair_cost NUMERIC(8,2) NOT NULL,
    reported_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE damage_report ADD CONSTRAINT damage_report_pkey PRIMARY KEY (damage_id);
ALTER TABLE damage_report ADD CONSTRAINT damage_report_rental_id_fkey FOREIGN KEY (rental_id) REFERENCES rental (rental_id);
ALTER TABLE damage_report ADD CONSTRAINT damage_report_inventory_id_fkey FOREIGN KEY (inventory_id) REFERENCES inventory (inventory_id);
ALTER TABLE damage_report ADD CONSTRAINT damage_report_reported_by_staff_id_fkey FOREIGN KEY (reported_by_staff_id) REFERENCES staff (staff_id);
ALTER TABLE damage_report ADD CONSTRAINT damage_report_severity_check CHECK (severity IN ('minor', 'moderate', 'severe'));
ALTER TABLE damage_report ADD CONSTRAINT damage_report_repair_cost_check CHECK (repair_cost >= 0);
