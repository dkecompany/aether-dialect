-- Federation storefront partition schema (CREATE-only; rows load from federation_storefront_data).

CREATE TABLE "address" (

    address_id INTEGER NOT NULL,
    address TEXT NOT NULL,
    district TEXT NOT NULL,
    city_id INTEGER NOT NULL,
    postal_code TEXT,
    phone TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "city" (

    city_id INTEGER NOT NULL,
    city TEXT NOT NULL,
    country_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "country" (

    country_id INTEGER NOT NULL,
    country TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "customer" (

    customer_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT,
    address_id INTEGER NOT NULL,
    activebool INTEGER,
    create_date TEXT,
    last_update TIMESTAMP

);

CREATE TABLE "payment" (

    payment_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    payment_date TIMESTAMP NOT NULL

);

CREATE TABLE rental (rental_id INTEGER NOT NULL, rental_date TIMESTAMP NOT NULL, inventory_id INTEGER NOT NULL, customer_id INTEGER NOT NULL, return_date TIMESTAMP, staff_id INTEGER NOT NULL, last_update TIMESTAMP NOT NULL);

CREATE TABLE "reservation" (

    reservation_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    reserved_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    fulfilled_rental_id INTEGER,
    status TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "staff" (

    staff_id INTEGER NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    address_id INTEGER NOT NULL,
    email TEXT,
    store_id INTEGER NOT NULL,
    active INTEGER,
    username TEXT NOT NULL,
    password TEXT,
    ssn TEXT,
    last_update TIMESTAMP

);

CREATE TABLE "store" (

    store_id INTEGER NOT NULL,
    manager_staff_id INTEGER,
    address_id INTEGER NOT NULL,
    last_update TIMESTAMP

);
