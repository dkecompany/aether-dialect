"""Cross-engine schema-graph equivalence tests for rental_shop. Builds a normalized structural projection per configured engine (with SQL-file DDL merge so BigQuery receives PK/FK) and compares table sets, column value types, nullability, primary keys, and foreign-key topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aetherdialect._utils import data_type_to_value_type

from .live_support import build_engine_t2s, engine_schema, skip_unless_configured
from .mydb_profile import RENTAL_SHOP_COLUMN_ROLE_ALLOWLIST

LIVE_ENGINE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("postgresql", "PGDATABASE", "rental_shop"),
    ("sqlite", "SQLITE_DATABASE", "rental_shop"),
    ("duckdb", "DUCKDB_DATABASE", "rental_shop"),
    ("mysql", "MYSQL_DATABASE", "rental_shop"),
    ("mariadb", "MARIADB_DATABASE", "rental_shop"),
    ("sqlserver", "SQLSERVER_DATABASE", "rental_shop"),
    ("oracle", "ORACLE_SERVICE_NAME", "FREEPDB1"),
    ("redshift", "REDSHIFT_DATABASE", "rental_shop"),
    ("databricks", "DATABRICKS_SCHEMA", "rental_shop"),
    ("snowflake", "SNOWFLAKE_DATABASE", "rental_shop"),
    ("bigquery", "BIGQUERY_DATASET", "rental_shop"),
)


@dataclass(frozen=True)
class ColumnProjection:
    value_type: str
    is_nullable: bool


@dataclass(frozen=True)
class TableProjection:
    columns: dict[str, ColumnProjection]
    primary_key: frozenset[str]
    foreign_keys: frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]]


def _column_value_type(col: Any) -> str:
    vt = str(getattr(col, "value_type", "") or "").strip().lower()
    if vt:
        return vt
    return data_type_to_value_type(str(getattr(col, "data_type", "") or ""))


def normalize_schema_projection(schema: Any) -> dict[str, TableProjection]:
    """Build a case-insensitive structural projection ignoring raw data types."""
    out: dict[str, TableProjection] = {}
    for tname, tbl in schema.tables.items():
        tkey = str(tname).lower()
        cols: dict[str, ColumnProjection] = {}
        for cname, col in tbl.columns.items():
            ckey = str(cname).lower()
            cols[ckey] = ColumnProjection(
                value_type=_column_value_type(col),
                is_nullable=bool(col.is_nullable),
            )
        fk_set: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
        for edge in tbl.foreign_keys:
            fk_set.add(
                (
                    tkey,
                    tuple(sorted(str(c).lower() for c in edge.src_cols)),
                    str(edge.dst_table).lower(),
                    tuple(sorted(str(c).lower() for c in edge.dst_cols)),
                ),
            )
        out[tkey] = TableProjection(
            columns=cols,
            primary_key=frozenset(str(c).lower() for c in (tbl.primary_key or [])),
            foreign_keys=frozenset(fk_set),
        )
    return out


def _collect_available_engines() -> list[tuple[str, str]]:
    available: list[tuple[str, str]] = []
    for engine_name, schema_env, default_schema in LIVE_ENGINE_SPECS:
        if skip_unless_configured(engine_name) is not None:
            continue
        schema = engine_schema(schema_env, default_schema)
        available.append((engine_name, schema))
    return available


def _assert_role_allowlist(schema: Any, engine_name: str) -> None:
    for tname, tbl in schema.tables.items():
        for cname, col in tbl.columns.items():
            role = str(getattr(col, "role", "") or "").strip().lower()
            if not role:
                continue
            key = f"{tname}.{cname}"
            allowed = RENTAL_SHOP_COLUMN_ROLE_ALLOWLIST.get(key)
            assert allowed is not None, f"{engine_name}: missing role allow-list entry for {key}"
            assert role in allowed, f"{engine_name}: role {role!r} for {key} not in allow-list {sorted(allowed)}"


def _compare_projections(
    reference_engine: str,
    reference: dict[str, TableProjection],
    engine_name: str,
    candidate: dict[str, TableProjection],
) -> None:
    ref_tables = set(reference)
    cand_tables = set(candidate)
    assert ref_tables == cand_tables, (
        f"{engine_name} table set differs from {reference_engine}: "
        f"missing={sorted(ref_tables - cand_tables)} extra={sorted(cand_tables - ref_tables)}"
    )
    for tkey in sorted(ref_tables):
        ref_tbl = reference[tkey]
        cand_tbl = candidate[tkey]
        ref_cols = set(ref_tbl.columns)
        cand_cols = set(cand_tbl.columns)
        assert ref_cols == cand_cols, (
            f"{engine_name} column set for {tkey} differs from {reference_engine}: "
            f"missing={sorted(ref_cols - cand_cols)} extra={sorted(cand_cols - ref_cols)}"
        )
        for ckey in sorted(ref_cols):
            ref_col = ref_tbl.columns[ckey]
            cand_col = cand_tbl.columns[ckey]
            assert ref_col.value_type == cand_col.value_type, (
                f"{engine_name} value_type for {tkey}.{ckey} is {cand_col.value_type!r}, "
                f"expected {ref_col.value_type!r} from {reference_engine}"
            )
            assert ref_col.is_nullable == cand_col.is_nullable, (
                f"{engine_name} is_nullable for {tkey}.{ckey} is {cand_col.is_nullable}, "
                f"expected {ref_col.is_nullable} from {reference_engine}"
            )
        assert ref_tbl.primary_key == cand_tbl.primary_key, (
            f"{engine_name} PK set for {tkey} is {sorted(cand_tbl.primary_key)}, "
            f"expected {sorted(ref_tbl.primary_key)} from {reference_engine}"
        )
        assert ref_tbl.foreign_keys == cand_tbl.foreign_keys, (
            f"{engine_name} FK set for {tkey} differs from {reference_engine}: "
            f"ref-only={sorted(ref_tbl.foreign_keys - cand_tbl.foreign_keys)} "
            f"cand-only={sorted(cand_tbl.foreign_keys - ref_tbl.foreign_keys)}"
        )


@pytest.fixture(scope="module")
def available_engine_projections(t2s) -> dict[str, dict[str, TableProjection]]:
    engines = _collect_available_engines()
    if len(engines) < 1:
        pytest.skip("no live engines configured for schema-graph equivalence test")
    projections: dict[str, dict[str, TableProjection]] = {}
    for engine_name, schema in engines:
        if engine_name == "postgresql":
            projections[engine_name] = normalize_schema_projection(t2s._schema_graph)
            _assert_role_allowlist(t2s._schema_graph, engine_name)
            continue
        instance = build_engine_t2s(engine_name, schema)
        projections[engine_name] = normalize_schema_projection(instance._schema_graph)
        _assert_role_allowlist(instance._schema_graph, engine_name)
    return projections


def test_schema_graph_equivalent_across_engines(
    available_engine_projections: dict[str, dict[str, TableProjection]],
) -> None:
    """Every configured engine yields the same normalized schema projection."""
    if len(available_engine_projections) < 2:
        pytest.skip("need at least two configured engines for cross-engine comparison")
    ref_engine, ref_projection = next(iter(available_engine_projections.items()))
    for engine_name, projection in available_engine_projections.items():
        if engine_name == ref_engine:
            continue
        _compare_projections(ref_engine, ref_projection, engine_name, projection)
