"""Data-quality severity contract and inspect_tabular_upload tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import aetherdialect._constants
from aetherdialect import inspect_tabular_upload
from aetherdialect._contracts_base import ConfigError, DataQualityReport
from aetherdialect._constants import (
    DATA_QUALITY_ISSUE_DUPLICATE_HEADER,
    DATA_QUALITY_ISSUE_EMPTY_FILE,
    DATA_QUALITY_ISSUE_MIXED_TYPES,
    DATA_QUALITY_ISSUE_MULTIPLE_TABLES,
    DATA_QUALITY_ISSUE_RAGGED_ROW,
    DATA_QUALITY_ISSUE_SEVERITY,
    DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT,
    DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_SEVERITY_REVIEW,
)
from aetherdialect._data_quality import (
    detect_grid_issues,
    load_source_grids,
    normalize_grid,
    validate_upload_sources,
    _issue_code,
    _issue_severity,
)


def _all_data_quality_issue_constants() -> set[str]:
    return {
        value
        for name, value in vars(aetherdialect._constants).items()
        if name.startswith("DATA_QUALITY_ISSUE_") and isinstance(value, str)
    }


def _mock_llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
    if task == "upload_summary":
        return {"summary": "Upload inspection completed."}
    if task == "upload_interpret":
        return {}
    raise AssertionError(f"unexpected llm_json task={task!r}")


@pytest.fixture(autouse=True)
def _patch_upload_llm() -> None:
    with patch("aetherdialect._data_quality.llm_json", side_effect=_mock_llm_json):
        yield


# --- Severity contract ---


@pytest.mark.fast
def test_every_data_quality_issue_constant_in_severity_table() -> None:
    issue_codes = _all_data_quality_issue_constants()
    assert issue_codes, "expected at least one DATA_QUALITY_ISSUE_* constant"
    missing = issue_codes - set(DATA_QUALITY_ISSUE_SEVERITY)
    assert not missing, f"missing severity mapping for: {sorted(missing)}"
    extra = set(DATA_QUALITY_ISSUE_SEVERITY) - issue_codes
    assert not extra, f"severity table has unknown issue codes: {sorted(extra)}"


@pytest.mark.fast
def test_sample_issues_have_expected_severity() -> None:
    samples = {
        DATA_QUALITY_ISSUE_RAGGED_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
        DATA_QUALITY_ISSUE_DUPLICATE_HEADER: DATA_QUALITY_SEVERITY_ADVISORY,
        DATA_QUALITY_ISSUE_MIXED_TYPES: DATA_QUALITY_SEVERITY_ADVISORY,
        DATA_QUALITY_ISSUE_MULTIPLE_TABLES: DATA_QUALITY_SEVERITY_REVIEW,
        DATA_QUALITY_ISSUE_EMPTY_FILE: DATA_QUALITY_SEVERITY_BLOCKING,
        DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT: DATA_QUALITY_SEVERITY_FATAL,
    }
    for code, expected in samples.items():
        assert DATA_QUALITY_ISSUE_SEVERITY[code] == expected


@pytest.mark.fast
def test_duplicate_header_is_advisory_not_blocking(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text("id,id\n1,2\n", encoding="utf-8")
    grid = normalize_grid(load_source_grids(path)[0])
    issues = detect_grid_issues(grid)
    dup_issues = [
        issue
        for issue in issues
        if _issue_code(issue) == DATA_QUALITY_ISSUE_DUPLICATE_HEADER
    ]
    assert dup_issues, "expected duplicate_header issue"
    issue = dup_issues[0]
    assert _issue_severity(issue) == DATA_QUALITY_SEVERITY_ADVISORY
    report = validate_upload_sources((path,), log_sink=lambda _msg: None)
    assert report.ok is True


@pytest.mark.fast
def test_severity_exposed_in_report_json_level(tmp_path: Path) -> None:
    path = tmp_path / "shifted.csv"
    path.write_text("Survey export\nid,name\n1,Alice\n", encoding="utf-8")
    report = validate_upload_sources((path,), log_sink=lambda _msg: None)
    payload = report.to_json_dict()
    severities = {issue["level"] for issue in payload["issues"] if issue["code"] != "DATA_QUALITY_AUTO_READ"}
    assert severities <= {
        DATA_QUALITY_SEVERITY_ADVISORY,
        DATA_QUALITY_SEVERITY_REVIEW,
        DATA_QUALITY_SEVERITY_BLOCKING,
        DATA_QUALITY_SEVERITY_FATAL,
    }


# --- inspect_tabular_upload ---


@pytest.mark.fast
def test_inspect_tabular_upload_returns_report_without_engine(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    with patch("aetherdialect.aetherdialect.AetherEngine") as engine_cls:
        report = inspect_tabular_upload(path)
        engine_cls.assert_not_called()
    assert isinstance(report, DataQualityReport)
    assert report.ok is True
    assert hasattr(report, "requires_review")
    assert hasattr(report, "confirmed_selections")
    assert hasattr(report, "suggested_selections")
    assert hasattr(report, "narrative")
    assert hasattr(report, "issues")


@pytest.mark.fast
def test_inspect_blocking_empty_file_returns_report_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    report = inspect_tabular_upload(path)
    assert isinstance(report, DataQualityReport)
    assert report.ok is False
    assert any(_issue_code(issue) == DATA_QUALITY_ISSUE_EMPTY_FILE for issue in report.issues)
    assert any(_issue_severity(issue) == DATA_QUALITY_SEVERITY_BLOCKING for issue in report.issues)


@pytest.mark.fast
def test_inspect_fatal_corrupt_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(b"not a real xlsx file")
    with pytest.raises(ConfigError):
        inspect_tabular_upload(path)


@pytest.mark.fast
def test_inspect_does_not_call_duckdb_or_engine_construction(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    path.write_text("id,name\n1,Alice\n", encoding="utf-8")
    with (
        patch("duckdb.connect", side_effect=AssertionError("duckdb.connect must not run during inspect")),
        patch(
            "aetherdialect._main_execution.initialize_aether_engine",
            side_effect=AssertionError("engine construction must not run during inspect"),
        ),
        patch(
            "aetherdialect._data_quality.prepare_relations_for_paths",
            side_effect=AssertionError("relation materialisation must not run during inspect"),
        ),
    ):
        report = inspect_tabular_upload(path)
    assert report.ok is True


# --- Documentation contract ---

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_UPLOAD_DOC_PATHS = (
    _DOCS / "INTEGRATOR_GUIDE.md",
    _DOCS / "SUPPORT_MATRIX.md",
    _DOCS / "USER_GUIDE.md",
)
_SEVERITY_NAMES = ("Advisory", "Review", "Blocking", "Fatal")


@pytest.mark.fast
def test_upload_severity_names_in_integrator_guide_user_guide_support_matrix() -> None:
    for path in _UPLOAD_DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        missing = [name for name in _SEVERITY_NAMES if name not in text]
        assert not missing, f"{path.name} missing severity names: {missing}"
        assert "inspect_tabular_upload" in text, f"{path.name} must describe inspect_tabular_upload"
        assert "source_selections" in text, f"{path.name} must describe source_selections"


@pytest.mark.fast
def test_security_discloses_tabular_llm_assist() -> None:
    text = (_DOCS / "SECURITY.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "tabular_llm_assist" in lowered
    assert "cell" in lowered and ("sample" in lowered or "sampling" in lowered)
    assert "upload" in lowered
    assert "disable" in lowered or "turn off" in lowered or "turned off" in lowered
