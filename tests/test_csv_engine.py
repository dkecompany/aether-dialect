"""Unit tests for the CSV/Excel in-memory DuckDB engine."""

from __future__ import annotations

import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import CsvRuntimeConfig, EngineConfig
from aetherdialect._contracts_base import ConfigError, EngineContext
from aetherdialect._dialect import DialectRegistry
from aetherdialect._llm_provider import LLMProvider

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


@pytest.fixture(autouse=True)
def _reset_csv_runtime_config() -> None:
    orig_directory = CsvRuntimeConfig.DIRECTORY
    orig_files = CsvRuntimeConfig.FILES
    orig_selections = dict(CsvRuntimeConfig.SOURCE_SELECTIONS)
    orig_connection = CsvRuntimeConfig.NATIVE_CONNECTION
    orig_api_token = EngineConfig.API_TOKEN
    orig_llm_provider = EngineConfig.LLM_PROVIDER
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "csv"
    EngineConfig.RUNTIME = CsvRuntimeConfig
    try:
        CsvRuntimeConfig.DIRECTORY = None
        CsvRuntimeConfig.FILES = ()
        CsvRuntimeConfig.set_source_selections({})
        CsvRuntimeConfig.clear_attached_connection()
        yield
    finally:
        CsvRuntimeConfig.DIRECTORY = orig_directory
        CsvRuntimeConfig.FILES = orig_files
        CsvRuntimeConfig.set_source_selections(orig_selections)
        CsvRuntimeConfig.NATIVE_CONNECTION = orig_connection
        EngineConfig.TYPE = _ORIG_ENGINE_TYPE
        EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME
        EngineConfig.API_TOKEN = orig_api_token
        EngineConfig.LLM_PROVIDER = orig_llm_provider
        LLMProvider.clear_llm_clients()


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
    return DialectRegistry.get("csv", CsvRuntimeConfig)


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


@pytest.mark.fast
def test_csv_reflect_preserves_original_name_on_spaced_header(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    path.write_text("Customer Name,qty\nAlice,1\n", encoding="utf-8")
    _configure_files(path)
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=RuntimeError("offline")):
        dialect = _make_dialect()
    graph = dialect.reflect_schema_graph(include="tables")
    table = graph.tables["items"]
    column = table.columns["customer_name"]
    assert column.name == "customer_name"
    assert column.original_name == "Customer Name"


@pytest.mark.fast
def test_csv_source_selection_header_row(tmp_path: Path) -> None:
    path = tmp_path / "shifted.csv"
    path.write_text("Title\nid,name\n1,Alice\n", encoding="utf-8")
    CsvRuntimeConfig.set_source_selections({path.name: {"header_row": 2}})
    _configure_files(path)
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=RuntimeError("offline")):
        dialect = _make_dialect()
    graph = dialect.reflect_schema_graph(include="tables")
    assert "shifted" in graph.tables
    assert "id" in graph.tables["shifted"].columns


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


@pytest.mark.fast
def test_csv_engine_accepts_xlsx(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "items.xlsx"
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["id"])
    worksheet.append([1])
    workbook.save(xlsx_path)
    EngineConfig.TYPE = "csv"
    CsvRuntimeConfig.apply_environment({"CSV_FILES": str(xlsx_path)})
    paths = CsvRuntimeConfig.resolve_source_files()
    assert paths == (xlsx_path.resolve(),)


def _mock_upload_llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
    if task == "upload_summary":
        return {"summary": "Upload inspection completed."}
    if task == "upload_interpret":
        return {}
    raise AssertionError(f"unexpected llm_json task={task!r}")


def _write_csv_engine_config(tmp_path: Path, *paths: Path) -> Path:
    files_value = ",".join(path.as_posix() for path in paths)
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        f"""
[engine]
selected = "csv"

[csv]
files = "{files_value}"

[openai]
api_key = "test-key"
""",
        encoding="utf-8",
    )
    return config_path


def _patch_csv_schema_llm() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_mock_upload_llm_json))
    stack.enter_context(patch("aetherdialect._schema_overrides._profile_subset"))
    stack.enter_context(patch("aetherdialect._schema_overrides.apply_column_roles_llm"))
    return stack


@pytest.mark.fast
def test_construction_raises_when_review_needed_without_selections(tmp_path: Path) -> None:
    path = tmp_path / "shifted.csv"
    path.write_text("Title\nid,name\n1,Alice\n", encoding="utf-8")
    config_path = _write_csv_engine_config(tmp_path, path)
    CsvRuntimeConfig.apply_environment({"CSV_FILES": str(path)})
    with _patch_csv_schema_llm():
        with pytest.raises(ConfigError) as exc_info:
            AetherEngine(EngineContext(), artifacts_dir=str(tmp_path / "artifacts"), config_file=str(config_path))
    report = getattr(exc_info.value, "data_quality_report", None)
    assert report is not None
    assert report.requires_review is True


@pytest.mark.fast
def test_construction_succeeds_with_confirmed_selections(tmp_path: Path) -> None:
    path = tmp_path / "shifted.csv"
    path.write_text("Title\nid,name\n1,Alice\n", encoding="utf-8")
    config_path = _write_csv_engine_config(tmp_path, path)
    CsvRuntimeConfig.apply_environment({"CSV_FILES": str(path)})
    with _patch_csv_schema_llm():
        engine = AetherEngine(
            EngineContext(),
            artifacts_dir=str(tmp_path / "artifacts"),
            config_file=str(config_path),
            source_selections={path.name: {"header_row": 2}},
        )
    assert "shifted" in engine._schema_graph.tables
    assert "id" in engine._schema_graph.tables["shifted"].columns


@pytest.mark.fast
def test_engine_data_quality_report_populated_after_csv_construction(tmp_path: Path) -> None:
    path = tmp_path / "customers.csv"
    path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    config_path = _write_csv_engine_config(tmp_path, path)
    CsvRuntimeConfig.apply_environment({"CSV_FILES": str(path)})
    with _patch_csv_schema_llm():
        engine = AetherEngine(EngineContext(), artifacts_dir=str(tmp_path / "artifacts"), config_file=str(config_path))
    report = engine.data_quality_report
    assert report is not None
    assert report.ok is True
    assert isinstance(report.narrative, str)
    assert isinstance(report.issues, tuple)


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
    EngineConfig.TYPE = "csv"
    CsvRuntimeConfig.apply_environment({"CSV_FILES": f"{csv_path},{xlsx_path}"})
    with pytest.raises(ConfigError, match="duplicate relation"):
        CsvRuntimeConfig.resolve_source_files()
