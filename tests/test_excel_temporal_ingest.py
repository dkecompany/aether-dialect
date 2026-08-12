"""Excel date/datetime cells must survive xlsx ingest with correct DuckDB types."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._constants_runtime import DATA_QUALITY_ISSUE_MIXED_TYPES
from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole
from aetherdialect._data_quality import (
    prepare_relations_for_paths,
    validate_upload_sources,
)
from aetherdialect._dialect_sqlglot_engines import CsvDialect
from aetherdialect._schema_profile import _infer_column_role
from aetherdialect._utils import data_type_to_value_type


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


def _save_xlsx(path: Path, rows: list[list[object]], *, formats: dict[tuple[int, int], str] | None = None) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, value in enumerate(row, start=1):
            cell = worksheet.cell(row=row_idx, column=col_idx, value=value)
            if formats and (row_idx, col_idx) in formats:
                cell.number_format = formats[(row_idx, col_idx)]
    workbook.save(path)


@pytest.mark.fast
def test_openpyxl_datetime_column_becomes_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "events.xlsx"
    _save_xlsx(
        path,
        [
            ["event_at"],
            [datetime(2024, 6, 15, 14, 30, 45)],
            [datetime(2024, 7, 1, 9, 0, 0)],
        ],
    )
    relations = prepare_relations_for_paths((path,))
    assert len(relations) == 1
    assert relations[0].column_types == ("TIMESTAMP",)


@pytest.mark.fast
def test_date_only_column_becomes_date(tmp_path: Path) -> None:
    path = tmp_path / "orders.xlsx"
    _save_xlsx(
        path,
        [
            ["order_date"],
            [date(2024, 1, 15)],
            [date(2024, 2, 20)],
        ],
    )
    relations = prepare_relations_for_paths((path,))
    assert len(relations) == 1
    assert relations[0].column_types == ("DATE",)


@pytest.mark.fast
def test_date_formatted_serial_becomes_date(tmp_path: Path) -> None:
    path = tmp_path / "serial_dates.xlsx"
    serial = 45323.0
    _save_xlsx(
        path,
        [
            ["ship_date"],
            [serial],
            [serial + 1],
        ],
        formats={(2, 1): "yyyy-mm-dd", (3, 1): "yyyy-mm-dd"},
    )
    relations = prepare_relations_for_paths((path,))
    assert len(relations) == 1
    assert relations[0].column_types == ("DATE",)


@pytest.mark.fast
def test_mixed_date_and_text_stays_varchar_with_review(tmp_path: Path) -> None:
    path = tmp_path / "mixed.xlsx"
    _save_xlsx(
        path,
        [
            ["id", "due_on"],
            [1, date(2024, 3, 1)],
            [2, "pending"],
        ],
    )
    relations = prepare_relations_for_paths((path,))
    assert relations[0].column_types == ("INTEGER", "VARCHAR")

    report = validate_upload_sources((path,), log_sink=lambda _msg: None)
    assert report.requires_review is True
    mixed_issues = [
        issue
        for issue in report.issues
        if any(key == "issue_code" and value == DATA_QUALITY_ISSUE_MIXED_TYPES for key, value in issue.details)
    ]
    assert mixed_issues
    assert any(any(key == "review" and value == "yes" for key, value in issue.details) for issue in mixed_issues)


@pytest.mark.fast
def test_schema_value_type_date_allows_temporal_role(tmp_path: Path) -> None:
    path = tmp_path / "shipments.xlsx"
    _save_xlsx(
        path,
        [
            ["shipped_on"],
            [date(2024, 5, 10)],
            [date(2024, 5, 11)],
        ],
    )
    relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.column_types == ("DATE",)

    import duckdb

    connection = duckdb.connect(":memory:")
    CsvDialect._load_prepared_relation_into_connection(connection, relation)
    columns, types = CsvDialect._column_types_for_source(path)
    assert types == ["DATE"]

    col = ColumnMetadata(name="shipped_on", data_type=types[0])
    assert data_type_to_value_type(col.data_type) == "date"
    assert _infer_column_role(col) == ColumnRole.TEMPORAL
