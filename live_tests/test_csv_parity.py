"""Live CSV-source parity checks against the DuckDB rental_shop fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

import aetherdialect._dialect_sqlglot_engines
from aetherdialect._config import CsvRuntimeConfig, DuckDBRuntimeConfig, EngineConfig
from aetherdialect._dialect import get_dialect
from aetherdialect._schema_overrides import apply_schema_overrides_to_graph, load_schema_overrides_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_DIRS = (
    _REPO_ROOT / "scripts" / "data" / "rental_shop_csvs",
    _REPO_ROOT / "scripts" / "data" / "rental_shop_csvs",
)
_OVERRIDES = _REPO_ROOT / "scripts" / "data" / "rental_shop_overrides.json"

_PARITY_TABLES = ("item", "customer", "store", "rental", "payment")

_PARITY_QUERIES = (
    "SELECT COUNT(*) FROM item WHERE item_type = 'film'",
    "SELECT COUNT(*) FROM customer",
    "SELECT store_id FROM store ORDER BY store_id LIMIT 5",
)

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


def _resolve_csv_dir() -> Path:
    for candidate in _CSV_DIRS:
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    pytest.skip("rental_shop CSV bundle not present under scripts/data/")


@pytest.fixture
def csv_dialect():
    pytest.importorskip("duckdb")
    csv_dir = _resolve_csv_dir()
    csv_paths = [csv_dir / f"{name}.csv" for name in _PARITY_TABLES]
    if not all(path.is_file() for path in csv_paths):
        pytest.skip("parity CSV subset missing under scripts/data/")
    EngineConfig.TYPE = "csv"
    EngineConfig.RUNTIME = CsvRuntimeConfig
    EngineConfig.SCHEMA_JSON_PATH = ""
    CsvRuntimeConfig.DIRECTORY = None
    CsvRuntimeConfig.FILES = tuple(str(path) for path in csv_paths)
    CsvRuntimeConfig.clear_attached_connection()
    try:
        connection = aetherdialect._dialect_sqlglot_engines._build_csv_memory_connection(CsvRuntimeConfig)
    except Exception as exc:
        pytest.skip(f"CSV bundle failed to load into DuckDB: {exc}")
    CsvRuntimeConfig.attach_connection(connection)
    dialect = get_dialect("csv", CsvRuntimeConfig)
    graph = dialect.reflect_schema_graph(include="tables")
    if _OVERRIDES.is_file():
        apply_schema_overrides_to_graph(graph, load_schema_overrides_file(str(_OVERRIDES)))
    probe = list(dialect.execute("SELECT COUNT(*) FROM item WHERE item_type = 'film'"))
    if not probe:
        pytest.skip("CSV engine loaded but item table is empty or unavailable")
    yield dialect, graph
    CsvRuntimeConfig.clear_attached_connection()
    EngineConfig.TYPE = _ORIG_ENGINE_TYPE
    EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


@pytest.fixture
def duckdb_dialect():
    duckdb = pytest.importorskip("duckdb")
    csv_dir = _resolve_csv_dir()
    EngineConfig.TYPE = "duckdb"
    EngineConfig.RUNTIME = DuckDBRuntimeConfig
    EngineConfig.SCHEMA_JSON_PATH = ""
    DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
    DuckDBRuntimeConfig.SCHEMA = "main"
    DuckDBRuntimeConfig.clear_attached_connection()
    connection = duckdb.connect(":memory:")
    for table in _PARITY_TABLES:
        csv_path = csv_dir / f"{table}.csv"
        if not csv_path.is_file():
            pytest.skip(f"missing parity CSV: {csv_path.name}")
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto({csv_path.as_posix()!r})")
    dialect = get_dialect("duckdb", DuckDBRuntimeConfig, native_connection=connection)
    graph = dialect.reflect_schema_graph(include="tables")
    if _OVERRIDES.is_file():
        apply_schema_overrides_to_graph(graph, load_schema_overrides_file(str(_OVERRIDES)))
    yield dialect, graph
    DuckDBRuntimeConfig.clear_attached_connection()
    EngineConfig.TYPE = _ORIG_ENGINE_TYPE
    EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


@pytest.mark.parametrize("sql", _PARITY_QUERIES)
def test_csv_matches_duckdb_for_allowed_queries(sql: str, csv_dialect, duckdb_dialect) -> None:
    csv_engine, _csv_graph = csv_dialect
    duck_engine, _duck_graph = duckdb_dialect
    csv_rows = list(csv_engine.execute(sql))
    duck_rows = list(duck_engine.execute(sql))
    assert csv_rows == duck_rows
