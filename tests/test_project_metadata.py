"""Project metadata and README install instructions must agree with pyproject.toml."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _readme_extras() -> set[str]:
    text = README.read_text(encoding="utf-8")
    return set(re.findall(r"aetherdialect\[([a-z0-9_,]+)\]", text, flags=re.IGNORECASE))


def _readme_python_floor() -> str | None:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"Requires Python\s+([\d.]+)\s+or newer", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _pyproject_python_floor() -> str:
    requires = _load_pyproject()["project"]["requires-python"]
    match = re.search(r">=(\d+\.\d+)", requires)
    assert match is not None, f"unparseable requires-python: {requires!r}"
    return match.group(1)


def _readme_extra_parts() -> set[str]:
    parts: set[str] = set()
    for extra in _readme_extras():
        for part in extra.split(","):
            part = part.strip()
            if part:
                parts.add(part)
    return parts


@pytest.mark.fast
def test_readme_extras_exist() -> None:
    extras = set(_load_pyproject()["project"].get("optional-dependencies", {}))
    for part in _readme_extra_parts():
        assert part in extras, f"README documents extra [{part}] not declared in pyproject.toml"
    assert "duckdb" not in _readme_extra_parts(), "README must not document a phantom [duckdb] extra"


@pytest.mark.fast
def test_readme_and_pyproject_agree() -> None:
    pyproject = _load_pyproject()
    extras = set(pyproject["project"].get("optional-dependencies", {}))
    readme_extras = _readme_extras()
    for extra in readme_extras:
        for part in extra.split(","):
            part = part.strip()
            assert part in extras, f"README documents extra [{part}] not declared in pyproject.toml"

    readme_floor = _readme_python_floor()
    assert readme_floor is not None, "README must state a Python version floor"
    assert readme_floor == _pyproject_python_floor()


@pytest.mark.fast
def test_requires_python_ceiling_matches_ci() -> None:
    requires = _load_pyproject()["project"]["requires-python"]
    assert "<3.13" in requires, "requires-python must cap below 3.13 to match CI"


@pytest.mark.fast
def test_dev_extra_includes_csv_and_sandbox_packages() -> None:
    dev = _load_pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(dep.startswith("openpyxl") for dep in dev), "dev extra must include csv (openpyxl)"
    assert any(dep.startswith("duckdb-engine") for dep in dev), "dev extra must include sandbox (duckdb-engine)"


@pytest.mark.fast
def test_ci_packaging_matrix() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "twine check dist/*" in ci
    assert "windows-latest" in ci
    assert "3.13" not in ci


@pytest.mark.fast
def test_required_metadata_fields_present() -> None:
    project = _load_pyproject()["project"]
    classifiers = project.get("classifiers", [])
    assert classifiers, "project.classifiers must be populated"

    required_prefixes = (
        "Development Status ::",
        "Intended Audience ::",
        "Programming Language :: Python ::",
        "Operating System ::",
        "Topic ::",
    )
    for prefix in required_prefixes:
        assert any(c.startswith(prefix) for c in classifiers), f"missing classifier prefix {prefix!r}"

    assert project.get("keywords"), "project.keywords must be populated"
    urls = project.get("urls", {})
    for key in ("Homepage", "Repository", "Issues", "Documentation"):
        assert urls.get(key), f"project.urls.{key} must be set"

    assert project.get("license"), "project.license SPDX expression must be set"
    license_files = project.get("license-files") or project.get("license_files")
    assert license_files, "license-files must reference the repository license"
    assert any((ROOT / name).is_file() for name in license_files)


@pytest.mark.fast
def test_no_empty_extras() -> None:
    extras = _load_pyproject()["project"].get("optional-dependencies", {})
    empty = [name for name, deps in extras.items() if not deps]
    assert empty == [], f"optional extras must not be empty: {empty}"
