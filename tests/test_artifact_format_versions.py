"""Artifact format version constants gate stale persisted trees."""

from __future__ import annotations

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION, FEDERATION_ARTIFACT_FORMAT_VERSION


@pytest.mark.fast
def test_engine_artifact_format_version_is_fourteen() -> None:
    assert ARTIFACT_FORMAT_VERSION == "0.2.1"


@pytest.mark.fast
def test_federation_artifact_format_version_is_eleven() -> None:
    assert FEDERATION_ARTIFACT_FORMAT_VERSION == "0.2.1"
