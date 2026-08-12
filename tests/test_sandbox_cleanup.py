"""Sandbox bundle extraction and temporary-directory cleanup."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest

from aetherdialect._sandbox import Sandbox, _DataBundleAccess
from tests._sandbox_csv_bundle import write_main_csv_ddl_bundle

_extract_data_bundle = Sandbox._extract_data_bundle


def _write_minimal_bundle(root: Path) -> None:
    write_main_csv_ddl_bundle(root)


def _write_zip_bundle(zip_path: Path, bundle_root: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in bundle_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_root).as_posix())


def _sandbox_temp_dirs() -> set[Path]:
    temp_root = Path(tempfile.gettempdir())
    return {
        path.resolve()
        for path in temp_root.iterdir()
        if path.is_dir() and path.name.startswith("aetherdialect_sandbox_")
    }


@pytest.mark.fast
def test_extract_directory_removed_when_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir()
    _write_minimal_bundle(bundle_root)
    zip_path = tmp_path / "data.zip"
    _write_zip_bundle(zip_path, bundle_root)
    monkeypatch.setattr(Sandbox, "data_zip_path", lambda **kwargs: zip_path)

    bundle_access = Sandbox._extract_data_bundle()
    assert isinstance(bundle_access, _DataBundleAccess)
    assert bundle_access.owns_cleanup is True
    extract_path = bundle_access.path
    assert extract_path.is_dir()

    try:
        assert (extract_path / "fixtures" / "rental_shop_mock.json").is_file()
    finally:
        if bundle_access.owns_cleanup:
            shutil.rmtree(bundle_access.path, ignore_errors=True)

    assert not extract_path.exists()


@pytest.mark.not_fast
def test_no_temp_directory_survives_sandbox_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir()
    _write_minimal_bundle(bundle_root)
    zip_path = tmp_path / "data.zip"
    _write_zip_bundle(zip_path, bundle_root)
    monkeypatch.setattr(Sandbox, "data_zip_path", lambda **kwargs: zip_path)

    before = _sandbox_temp_dirs()
    extract_path: Path | None = None
    artifacts_path: Path | None = None

    with Sandbox(auto_seed=False) as sandbox:
        assert sandbox._bundle_access.owns_cleanup is True
        assert sandbox._extract_dir
        extract_path = Path(sandbox._extract_dir)
        artifacts_path = Path(sandbox.artifacts_dir)
        assert extract_path.is_dir()
        assert artifacts_path.is_dir()

    assert extract_path is not None
    assert artifacts_path is not None
    assert not extract_path.exists()
    assert not artifacts_path.exists()
    assert _sandbox_temp_dirs() <= before
