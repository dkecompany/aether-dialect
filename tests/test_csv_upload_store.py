"""CSV upload store path and probe helpers."""

from __future__ import annotations

from pathlib import Path

from aetherdialect._constants import UPLOAD_STORE_FILENAME
from aetherdialect._dialect_sqlglot_engines import CsvDialect


def test_csv_upload_store_path() -> None:
    path = CsvDialect._csv_upload_store_path("/tmp/artifacts")
    assert path.name == UPLOAD_STORE_FILENAME
    assert path.parent == Path("/tmp/artifacts")


def test_csv_combined_store_probe() -> None:
    assert CsvDialect._csv_combined_store_probe("a", "") == "a"
    assert CsvDialect._csv_combined_store_probe("a", "b") == "a\nb"


def test_csv_source_probe_payload(tmp_path: Path) -> None:
    f = tmp_path / "t.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    payload = CsvDialect._csv_source_probe_payload([f])
    assert f.name in payload or str(f.resolve()) in payload or "|" in payload
    assert payload.count("|") >= 2
