"""Unit tests for the CSV/Excel in-memory DuckDB engine."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from aetherdialect._config import ConfigError, CsvRuntimeConfig, EngineConfig
from aetherdialect._dialect import get_dialect

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


@pytest.fixture(autouse=True)
def _reset_csv_runtime_config() -> None:
    orig_directory = CsvRuntimeConfig.DIRECTORY
    orig_files = CsvRuntimeConfig.FILES
    orig_connection = CsvRuntimeConfig.NATIVE_CONNECTION
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "csv"
    EngineConfig.RUNTIME = CsvRuntimeConfig
    try:
        CsvRuntimeConfig.DIRECTORY = None
        CsvRuntimeConfig.FILES = ()
        CsvRuntimeConfig.clear_attached_connection()
        yield
    finally:
        CsvRuntimeConfig.DIRECTORY = orig_directory
        CsvRuntimeConfig.FILES = orig_files
        CsvRuntimeConfig.NATIVE_CONNECTION = orig_connection
        EngineConfig.TYPE = _ORIG_ENGINE_TYPE
        EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


@pytest.fixture
def csv_fixture_dir(tmp_path: Path) -> Path:
    fixture = tmp_path / "csv_fixture"
    fixture.mkdir()
    (fixture / "customers.csv").write_text(
        "id,name\n1,Alice\n2,Bob\n",
        encoding="utf-8",
    )
    (fixture / "orders.csv").write_text(
        "id,customer_id,amount\n10,1,25.5\n11,2,40.0\n",
        encoding="utf-8",
    )
    return fixture


def _configure_files(*paths: Path) -> None:
    CsvRuntimeConfig.DIRECTORY = None
    CsvRuntimeConfig.FILES = tuple(str(path) for path in paths)


def _make_dialect() -> object:
    pytest.importorskip("duckdb")
    return get_dialect("csv", CsvRuntimeConfig)


def test_csv_query_join_and_aggregate(csv_fixture_dir: Path) -> None:
    _configure_files(csv_fixture_dir / "customers.csv", csv_fixture_dir / "orders.csv")
    dialect = _make_dialect()

    graph = dialect.reflect_schema_graph(include="tables")
    assert set(graph.tables) == {"customers", "orders"}

    join_rows = dialect.execute(
        "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id ORDER BY c.name"
    )
    assert join_rows == [("Alice", 25.5), ("Bob", 40.0)]

    agg_rows = dialect.execute(
        "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY customer_id"
    )
    assert agg_rows == [(1, 25.5), (2, 40.0)]


def test_csv_reflect_schema_from_headers(csv_fixture_dir: Path) -> None:
    _configure_files(csv_fixture_dir / "customers.csv")
    dialect = _make_dialect()

    graph = dialect.reflect_schema_graph(include="tables")
    customer = graph.tables["customers"]
    assert list(customer.columns) == ["id", "name"]
    assert customer.columns["id"].data_type.upper() == "INTEGER"
    assert customer.columns["name"].data_type.upper() == "VARCHAR"


def test_csv_add_and_delete_file_migration(csv_fixture_dir: Path) -> None:
    customers = csv_fixture_dir / "customers.csv"
    orders = csv_fixture_dir / "orders.csv"
    _configure_files(customers, orders)
    dialect = _make_dialect()
    assert set(dialect.reflect_schema_graph(include="tables").tables) == {"customers", "orders"}

    products = csv_fixture_dir / "products.csv"
    products.write_text("id,label\n1,Widget\n", encoding="utf-8")
    _configure_files(customers, orders, products)
    dialect = _make_dialect()
    assert set(dialect.reflect_schema_graph(include="tables").tables) == {
        "customers",
        "orders",
        "products",
    }

    _configure_files(customers, orders)
    dialect = _make_dialect()
    assert set(dialect.reflect_schema_graph(include="tables").tables) == {"customers", "orders"}


def test_csv_compute_ddl_probe_tracks_file_changes(csv_fixture_dir: Path) -> None:
    path = csv_fixture_dir / "customers.csv"
    _configure_files(path)
    dialect = _make_dialect()
    first = dialect.compute_ddl_probe(None)

    path.write_text("id,name\n1,Alice\n2,Bob\n3,Carol\n", encoding="utf-8")
    dialect = _make_dialect()
    second = dialect.compute_ddl_probe(None)

    assert first
    assert second
    assert first != second


def test_csv_locked_types_override_inference(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    fixture = tmp_path / "codes"
    fixture.mkdir()
    code_file = fixture / "codes.csv"
    code_file.write_text("code\n001\n002\n", encoding="utf-8")

    cache_path = tmp_path / "schema.json.gz"
    payload = {
        "tables": {
            "codes": {
                "name": "codes",
                "columns": {
                    "code": {
                        "name": "code",
                        "data_type": "VARCHAR",
                        "is_primary_key": False,
                        "is_foreign_key": False,
                        "fk_target": None,
                        "is_unique": False,
                        "is_generated": False,
                        "is_identity": False,
                        "is_nullable": True,
                        "enum_type_name": None,
                        "value_overlap_sample": [],
                    }
                },
                "foreign_keys": [],
                "description": "",
            }
        }
    }
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    EngineConfig.SCHEMA_JSON_PATH = str(cache_path)
    _configure_files(code_file)
    dialect = _make_dialect()

    graph = dialect.reflect_schema_graph(include="tables")
    assert graph.tables["codes"].columns["code"].data_type.upper() == "VARCHAR"

    rows = dialect.execute("SELECT code FROM codes ORDER BY code")
    assert rows == [("001",), ("002",)]


def test_csv_config_validation_mutually_exclusive(csv_fixture_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not both"):
        CsvRuntimeConfig.apply_environment(
            {
                "CSV_DIRECTORY": str(csv_fixture_dir),
                "CSV_FILES": str(csv_fixture_dir / "customers.csv"),
            }
        )


def test_csv_config_validation_duplicate_relation_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "items.csv"
    xlsx_path = tmp_path / "items.xlsx"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["id"])
    worksheet.append([2])
    workbook.save(xlsx_path)
    CsvRuntimeConfig.apply_environment({"CSV_FILES": f"{csv_path},{xlsx_path}"})
    with pytest.raises(ConfigError, match="duplicate relation"):
        CsvRuntimeConfig.resolve_source_files()
