"""Artifact format version constants track the library version axis."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION, FEDERATION_ARTIFACT_FORMAT_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONSTANTS_PATH = _REPO_ROOT / "src" / "aetherdialect" / "_constants.py"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_VERSION_ASSIGN_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?:_VERSION|_FORMAT_VERSION)$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _package_version() -> Version:
    text = _PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None, "pyproject.toml missing [project] version"
    return Version(match.group(1))


def _format_version_constants() -> dict[str, str]:
    tree = ast.parse(_CONSTANTS_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if not _VERSION_ASSIGN_RE.match(name):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        value = node.value.value.strip()
        if not _SEMVER_RE.match(value):
            continue
        out[name] = value
    return out


@pytest.mark.fast
def test_engine_artifact_format_version_tracks_library() -> None:
    assert ARTIFACT_FORMAT_VERSION == "0.2.3"


@pytest.mark.fast
def test_federation_artifact_format_version_unchanged() -> None:
    assert FEDERATION_ARTIFACT_FORMAT_VERSION == "0.2.3"


@pytest.mark.fast
@pytest.mark.hygiene
def test_format_version_constants_not_ahead_of_package() -> None:
    """Every X.Y.Z format/artifact version constant must be ≤ package version."""
    package = _package_version()
    constants = _format_version_constants()
    assert constants, "expected library-aligned version constants in _constants.py"
    ahead: list[str] = []
    for name, raw in sorted(constants.items()):
        try:
            ver = Version(raw)
        except InvalidVersion:
            ahead.append(f"{name}={raw!r} (unparseable)")
            continue
        if ver > package:
            ahead.append(f"{name}={raw} > package {package}")
    assert not ahead, "format versions must not exceed library version:\n" + "\n".join(ahead)
