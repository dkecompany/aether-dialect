"""CSV connection slugs normalize path separators before hashing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._config import CsvRuntimeConfig


class _PosixShapedPath:
    """Path stand-in whose ``str()`` uses forward slashes."""

    def __init__(self, posix: str) -> None:
        self._posix = posix

    def as_posix(self) -> str:
        return self._posix

    def __str__(self) -> str:
        return self._posix


class _WindowsShapedPath:
    """Path stand-in whose ``str()`` uses backslashes."""

    def __init__(self, posix: str, windows: str) -> None:
        self._posix = posix
        self._windows = windows

    def as_posix(self) -> str:
        return self._posix

    def __str__(self) -> str:
        return self._windows


@pytest.mark.fast
def test_slug_identical_for_posix_and_windows_shaped_paths() -> None:
    """Separator style in ``str(path)`` must not change the slug digest."""
    logical = "C:/data/reports/sales.csv"
    posix_paths = (_PosixShapedPath(logical),)
    windows_paths = (_WindowsShapedPath(logical, "C:\\data\\reports\\sales.csv"),)

    with patch.object(CsvRuntimeConfig, "resolve_source_files", return_value=posix_paths):
        posix_slug = CsvRuntimeConfig().connection_slug_fields()["source"]
    with patch.object(CsvRuntimeConfig, "resolve_source_files", return_value=windows_paths):
        windows_slug = CsvRuntimeConfig().connection_slug_fields()["source"]

    assert posix_slug == windows_slug
