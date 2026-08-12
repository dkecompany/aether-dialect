-- Federation catalog partition schema (CREATE-only; rows load from federation_catalog_data).

CREATE TABLE "actor" (

    actor_id INTEGER NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "author" (

    author_id INTEGER NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "book" (

    item_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    publisher_id INTEGER NOT NULL,
    isbn TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "category" (

    category_id INTEGER NOT NULL,
    name TEXT NOT NULL,
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

CREATE TABLE "film" (

    item_id INTEGER NOT NULL,
    original_language_id INTEGER,
    length INTEGER,
    rating TEXT,
    last_update TIMESTAMP

);

CREATE TABLE "film_actor" (

    actor_id INTEGER NOT NULL,
    film_item_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "game" (

    item_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    developer TEXT NOT NULL,
    esrb_rating TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "game_supported_language" (

    item_id INTEGER NOT NULL,
    language_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "inventory" (

    inventory_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "item" (

    item_id INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    release_year INTEGER,
    language_id INTEGER NOT NULL,
    rental_duration INTEGER,
    rental_rate REAL,
    replacement_cost REAL,
    last_update TIMESTAMP

);

CREATE TABLE "item_category" (

    item_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "item_feature" (

    item_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "language" (

    language_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    last_update TIMESTAMP

);

CREATE TABLE "payment" (

    payment_id INTEGER NOT NULL,
    rental_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    payment_date TIMESTAMP NOT NULL

);

CREATE TABLE "publisher" (

    publisher_id INTEGER NOT NULL,
    publisher_name TEXT NOT NULL,
    country_id INTEGER NOT NULL,
    last_update TIMESTAMP

);
