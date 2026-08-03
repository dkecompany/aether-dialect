"""Artifact format version constants gate stale persisted trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION, FEDERATION_ARTIFACT_FORMAT_VERSION

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_engine_artifact_format_version_is_six() -> None:
    assert ARTIFACT_FORMAT_VERSION == 6


@pytest.mark.fast
def test_federation_artifact_format_version_is_nine() -> None:
    assert FEDERATION_ARTIFACT_FORMAT_VERSION == 9


@pytest.mark.fast
def test_package_version_matches_changelog_release() -> None:
    import tomllib

    changelog = (_REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_version = pyproject["project"]["version"]
    assert f"## {pkg_version}" in changelog
