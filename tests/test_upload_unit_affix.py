"""Scalar unit-affix stripping on CSV/Excel upload ingest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import duckdb
from aetherdialect._constants import DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED
from aetherdialect._data_quality import prepare_relations_for_paths
from aetherdialect._dialect_sqlglot_engines import CsvDialect


def _mock_llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
    if task == "upload_summary":
        return {"summary": "Upload inspection completed."}
    if task == "upload_interpret":
        return {}
    raise AssertionError(f"unexpected llm_json task={task!r}")


@pytest.fixture(autouse=True)
def _patch_upload_llm() -> None:
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_mock_llm_json):
        yield


def _schema_for_csv(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    relations = prepare_relations_for_paths((path,))
    assert len(relations) == 1
    relation = relations[0]
    connection = duckdb.connect(":memory:")
    CsvDialect.load_prepared_relation_into_native_connection(connection, relation)
    columns, types = CsvDialect._column_types_for_source(path)
    tables_meta = CsvDialect._tables_meta_from_relations(relations)
    comments = tables_meta[relation.relation_name].get("column_comments", {})
    return columns, types, comments


def _affix_diagnostics(path: Path) -> list[str]:
    codes: list[str] = []

    def _capture(_message: str, **kwargs: object) -> None:
        code = kwargs.get("code")
        if isinstance(code, str):
            codes.append(code)

    with patch("aetherdialect._data_quality.notify", side_effect=_capture):
        prepare_relations_for_paths((path,))
    return codes


@pytest.mark.fast
def test_uniform_currency_becomes_number(tmp_path: Path) -> None:
    path = tmp_path / "revenue.csv"
    path.write_text("amount\nUSD 10\nUSD 25\nUSD 30\n", encoding="utf-8")
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("INTEGER",)
    assert tuple(row["amount"] for row in relation.rows) == ("10", "25", "30")

    _columns, types, comments = _schema_for_csv(path)
    assert types == ["INTEGER"]
    assert "USD" in comments.get("amount", "")

    diagnostics = _affix_diagnostics(path)
    assert DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED in diagnostics


@pytest.mark.fast
def test_bare_number_mixed_with_affix_still_numeric(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text("qty\n10\nUSD 20\n30\n", encoding="utf-8")
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("INTEGER",)
    assert tuple(row["qty"] for row in relation.rows) == ("10", "20", "30")


@pytest.mark.fast
def test_second_currency_token_keeps_string(tmp_path: Path) -> None:
    path = tmp_path / "mixed_currency.csv"
    path.write_text("amount\nUSD 10\nAUD 10\n", encoding="utf-8")
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("VARCHAR",)
    assert tuple(row["amount"] for row in relation.rows) == ("USD 10", "AUD 10")


@pytest.mark.fast
def test_revenue_band_not_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bands.csv"
    path.write_text("revenue_band\n$10 to $20\n$30 to $40\n", encoding="utf-8")
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("VARCHAR",)
    assert relation.rows[0]["revenue_band"] == "$10 to $20"


@pytest.mark.fast
def test_percent_range_not_stripped(tmp_path: Path) -> None:
    path = tmp_path / "pct_range.csv"
    path.write_text("margin\n45% - 55%\n40% - 50%\n", encoding="utf-8")
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("VARCHAR",)
    assert relation.rows[0]["margin"] == "45% - 55%"
