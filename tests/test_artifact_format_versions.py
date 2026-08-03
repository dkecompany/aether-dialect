"""Artifact format version constants gate stale persisted trees."""

from __future__ import annotations

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION, FEDERATION_ARTIFACT_FORMAT_VERSION


@pytest.mark.fast
def test_engine_artifact_format_version_is_six() -> None:
    assert ARTIFACT_FORMAT_VERSION == 6


@pytest.mark.fast
def test_federation_artifact_format_version_is_nine() -> None:
    assert FEDERATION_ARTIFACT_FORMAT_VERSION == 9
