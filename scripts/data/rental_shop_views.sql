-- Local-engine views only (not shipped in rental_shop.sql or sandbox seed).
-- Created by scripts/load_rental_shop_engines.py after CSV load.

CREATE VIEW active_customer_v AS
SELECT
    customer_id,
    store_id,
    first_name,
    last_name,
    email,
    create_date
FROM customer
WHERE activebool = true;

CREATE VIEW store_revenue_v AS
SELECT
    c.store_id,
    SUM(p.amount) AS total_revenue,
    COUNT(DISTINCT p.payment_id) AS payment_count
FROM payment p
JOIN rental r ON p.rental_id = r.rental_id
JOIN customer c ON r.customer_id = c.customer_id
GROUP BY c.store_id;

CREATE VIEW film_catalog_v AS
SELECT
    i.item_id,
    i.title,
    i.release_year,
    f.rating,
    f.length,
    i.rental_rate
FROM film f
JOIN item i ON f.item_id = i.item_id
WHERE i.item_type = 'film';
