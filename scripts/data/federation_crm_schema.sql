-- Federation crm partition schema (CREATE-only; rows load from federation_crm_data).

CREATE TABLE "promotion" (

    promotion_id INTEGER NOT NULL,
    promo_name TEXT NOT NULL,
    promo_type TEXT NOT NULL,
    discount_pct REAL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    is_active INTEGER,
    last_update TIMESTAMP

);

CREATE TABLE customer (customer_id INTEGER NOT NULL, store_id INTEGER NOT NULL, first_name VARCHAR(45) NOT NULL, last_name VARCHAR(45) NOT NULL, email_addr VARCHAR(50), address_id INTEGER NOT NULL, loyalty_tier VARCHAR(20), create_date TEXT NOT NULL, last_update TIMESTAMP);

CREATE TABLE "promotion_redemption" (

    redemption_id INTEGER NOT NULL,
    promotion_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    discount_amount REAL NOT NULL,
    redeemed_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE staff (staff_id INTEGER, first_name TEXT, last_name TEXT, store_id INTEGER);
