"""Package-data declaration checks (single source of truth for CI/pre- commit via pytest)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "aetherdialect"
PYPROJECT = ROOT / "pyproject.toml"
CORPUS_REL = "sandbox/data.zip"


@dataclass(frozen=True)
class _CheckOutcome:
    """Result of validating declared package-data paths."""

    ok: bool
    deferred: bool
    message: str


def _declared_package_data_paths(pyproject: Path = PYPROJECT) -> list[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["aetherdialect"])


def _check_package_data(*, package_dir: Path, pyproject: Path) -> _CheckOutcome:
    declared = _declared_package_data_paths(pyproject)
    missing = [rel for rel in declared if not (package_dir / rel).is_file()]
    if not missing:
        return _CheckOutcome(ok=True, deferred=False, message="all declared package-data paths exist")
    non_corpus = [rel for rel in missing if rel != CORPUS_REL]
    if not non_corpus and CORPUS_REL in missing:
        return _CheckOutcome(
            ok=True,
            deferred=True,
            message=f"needs_corpus: declared corpus path missing: {CORPUS_REL}",
        )
    return _CheckOutcome(
        ok=False,
        deferred=False,
        message=f"declared package-data paths missing: {', '.join(missing)}",
    )


@pytest.mark.fast
def test_checker_defers_explicitly_when_only_corpus_missing(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "aetherdialect"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "py.typed").write_text("", encoding="utf-8")
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        '[tool.setuptools.package-data]\n"aetherdialect" = ["py.typed", "sandbox/data.zip"]\n',
        encoding="utf-8",
    )

    outcome = _check_package_data(package_dir=fake_pkg, pyproject=fake_pyproject)

    assert outcome.deferred is True
    assert "needs_corpus:" in outcome.message
    assert CORPUS_REL in outcome.message


@pytest.mark.needs_corpus
def test_declared_package_data_paths_exist() -> None:
    missing = [rel for rel in _declared_package_data_paths() if not (PACKAGE_ROOT / rel).is_file()]
    assert missing == [], f"declared package-data paths missing on disk: {missing}"


@pytest.mark.fast
def test_py_typed_present_and_declared() -> None:
    declared = _declared_package_data_paths()
    assert "py.typed" in declared
    py_typed = PACKAGE_ROOT / "py.typed"
    assert py_typed.is_file(), "src/aetherdialect/py.typed must exist"
    assert py_typed.read_text(encoding="utf-8") == ""


@pytest.mark.fast
def test_checker_fails_on_non_corpus_missing(tmp_path: Path) -> None:
    fake_pkg = tmp_path / "aetherdialect"
    fake_pkg.mkdir(parents=True)
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        '[tool.setuptools.package-data]\n"aetherdialect" = ["py.typed", "sandbox/data.zip"]\n',
        encoding="utf-8",
    )

    outcome = _check_package_data(package_dir=fake_pkg, pyproject=fake_pyproject)

    assert outcome.ok is False
    assert outcome.deferred is False
    assert "py.typed" in outcome.message


@pytest.mark.fast
def test_repo_package_data_ok_or_corpus_deferred() -> None:
    outcome = _check_package_data(package_dir=PACKAGE_ROOT, pyproject=PYPROJECT)
    assert outcome.ok or outcome.deferred, outcome.message
