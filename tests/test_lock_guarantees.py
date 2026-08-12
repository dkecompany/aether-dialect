"""Artifact lock advisory guarantees and filesystem locality checks."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._utils_artifacts import artifact_lock, warn_if_artifacts_dir_not_local


@pytest.mark.fast
def test_network_path_reports_diagnostic(tmp_path) -> None:
    adir = str(tmp_path / "artifacts")
    buf: list = []
    token = set_diagnostic_collector(buf)
    try:
        with patch("aetherdialect._utils_artifacts._artifacts_dir_is_local_filesystem", return_value=False):
            warn_if_artifacts_dir_not_local(adir)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    codes = [d.code for d in diags]
    assert DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL in codes


@pytest.mark.fast
def test_artifact_lock_docstring_states_advisory_guarantee() -> None:
    doc = artifact_lock.__doc__ or ""
    lowered = doc.lower()
    assert "advisory" in lowered
    assert "cooperating" in lowered
    assert "local" in lowered
