-- Federation CRM partition seed (full partition; diverged customer replica).

CREATE TABLE promotion (
    promotion_id INTEGER NOT NULL,
    promo_name VARCHAR(120) NOT NULL,
    promo_type VARCHAR(30) NOT NULL,
    discount_pct NUMERIC(5,2),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_active BOOLEAN NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO promotion (promotion_id, promo_name, promo_type, discount_pct, start_date, end_date, is_active, last_update) VALUES (1, 'Spring Subscription Promo', 'percent', 15.00, '2024-03-01', '2024-06-30', 1, '2024-03-01 00:00:00');

CREATE TABLE promotion_redemption (
    redemption_id INTEGER NOT NULL,
    promotion_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    discount_amount NUMERIC(6,2) NOT NULL,
    redeemed_at TIMESTAMP NOT NULL,
    last_update TIMESTAMP NOT NULL
);;
INSERT INTO promotion_redemption (redemption_id, promotion_id, rental_id, discount_amount, redeemed_at, last_update) VALUES (1, 1, 1, 1.50, '2024-03-22 08:05:00', '2024-03-22 08:05:00');

CREATE TABLE customer (
    customer_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    email_addr VARCHAR(50),
    address_id INTEGER NOT NULL,
    loyalty_tier VARCHAR(20),
    create_date DATE NOT NULL,
    last_update TIMESTAMP
);;
INSERT INTO customer (customer_id, store_id, first_name, last_name, email_addr, address_id, loyalty_tier, create_date, last_update) VALUES (1, 1, 'Mary (crm)', 'Smith', 'mary.smith@example.com', 1001, 'gold', '2020-01-15', '2024-01-01 00:00:00');
INSERT INTO customer (customer_id, store_id, first_name, last_name, email_addr, address_id, loyalty_tier, create_date, last_update) VALUES (2, 1, 'Patricia', 'Johnson', 'patricia.j@example.com', 2, 'silver', '2020-02-20', '2024-01-01 00:00:00');
INSERT INTO customer (customer_id, store_id, first_name, last_name, email_addr, address_id, loyalty_tier, create_date, last_update) VALUES (5, 2, 'Linda (crm)', 'Brown', NULL, 1005, 'bronze', '2020-05-10', '2024-01-01 00:00:00');

CREATE TABLE staff (
    staff_id INTEGER NOT NULL,
    first_name VARCHAR(45) NOT NULL,
    last_name VARCHAR(45) NOT NULL,
    store_id INTEGER NOT NULL
);;
INSERT INTO staff (staff_id, first_name, last_name, store_id) VALUES (1, 'Mike', 'Hillyer', 1);
INSERT INTO staff (staff_id, first_name, last_name, store_id) VALUES (2, 'Jon', 'Stephens', 2);
