"""Path comparison and case-insensitive extension checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pre_fix_failure: dict[str, str | None] = {
    "paths_equal": None,
    "json_suffix": None,
}


def _paths_equal(a: Path | str, b: Path | str) -> bool:
    try:
        from aetherdialect._core_utils import paths_equal
    except ImportError:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    return paths_equal(a, b)


def test_case_differing_paths_compare_equal_on_windows(tmp_path: Path) -> None:
    base = tmp_path / "Artifacts"
    base.mkdir()
    left = base / "fed_alpha"
    left.mkdir()
    right = base / "FED_ALPHA"

    if sys.platform == "win32":
        if not _paths_equal(left, right):
            pre_fix_failure["paths_equal"] = "case-differing absolute paths were not equal on Windows"
        assert _paths_equal(left, right), pre_fix_failure["paths_equal"]
    else:
        assert _paths_equal(left, left.resolve())
        assert _paths_equal(str(left), str(left.resolve()))

    editor = tmp_path / "mappings.JSON"
    editor.write_text("{}", encoding="utf-8")
    from aetherdialect._federation import archive_federation_mappings_file

    try:
        archive = archive_federation_mappings_file(str(editor))
    except Exception as exc:
        pre_fix_failure["json_suffix"] = f"uppercase .JSON extension rejected: {exc!r}"
        pytest.fail(pre_fix_failure["json_suffix"])
    assert archive.endswith("applied_federation_mappings.json")
