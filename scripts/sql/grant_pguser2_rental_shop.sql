-- Idempotent PostgreSQL grants for the pguser2 consumer role (RBAC live tests).
--
-- Prerequisites:
--   1. rental_shop data is loaded (scripts/load_rental_shop_engines.py --engine postgresql).
--   2. Role pguser2 must already exist, for example:
--        CREATE ROLE pguser2 LOGIN PASSWORD 'your-password';
--
-- Run as a superuser or the rental_shop owner while connected to the target database:
--   psql -d rental_shop -f scripts/sql/grant_pguser2_rental_shop.sql
--
-- Grants SELECT on consumer tables only (includes item for join paths). All other tables
-- (e.g. staff) remain inaccessible to pguser2.

\set ON_ERROR_STOP on

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pguser2') THEN
        RAISE EXCEPTION
            'role pguser2 does not exist; create it first, e.g. CREATE ROLE pguser2 LOGIN PASSWORD ''...'';';
    END IF;
END
$$;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM pguser2;
REVOKE ALL ON SCHEMA public FROM pguser2;

DO $$
DECLARE
    db_name text := current_database();
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO pguser2', db_name);
END
$$;

GRANT USAGE ON SCHEMA public TO pguser2;

GRANT SELECT ON TABLE
    public.actor,
    public.address,
    public.category,
    public.city,
    public.country,
    public.customer,
    public.film,
    public.film_actor,
    public.item,
    public.item_category,
    public.inventory,
    public.language,
    public.payment,
    public.rental,
    public.store
TO pguser2;
