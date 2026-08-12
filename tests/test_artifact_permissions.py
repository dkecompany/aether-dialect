"""Artifact files and directories get owner-only permissions after atomic writes."""

from __future__ import annotations

import os
import stat
import sys

import pytest

from aetherdialect._constants import ARTIFACT_FILE_MODE
from aetherdialect._federation_execute import _write_federation_json_atomic
from aetherdialect._utils_artifacts import (
    write_gzip_json_atomic,
    write_json_atomic,
    write_text_atomic,
)


@pytest.mark.fast
def test_new_artifact_file_mode_owner_only(tmp_path, monkeypatch) -> None:
    """Atomic artifact writers chmod owner-only (0o600) after replace."""
    chmod_calls: list[tuple[str, int]] = []
    real_chmod = os.chmod

    def tracking_chmod(path: str | os.PathLike[str], mode: int) -> None:
        chmod_calls.append((os.path.abspath(str(path)), mode))
        return real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", tracking_chmod)

    artifact_dir = tmp_path / "artifacts"
    json_path = artifact_dir / "manifest.json"
    text_path = artifact_dir / "notes.txt"
    gzip_path = artifact_dir / "graph.json.gz"
    fed_path = artifact_dir / "fed.json"

    write_json_atomic(json_path, {"k": "v"})
    write_text_atomic(text_path, "hello")
    write_gzip_json_atomic(str(gzip_path), {"g": 1}, sort_keys=True)
    _write_federation_json_atomic(str(fed_path), {"f": 1})

    written_paths = [
        os.path.abspath(str(json_path)),
        os.path.abspath(str(text_path)),
        os.path.abspath(str(gzip_path)),
        os.path.abspath(str(fed_path)),
    ]
    chmod_targets = {path for path, mode in chmod_calls if mode == ARTIFACT_FILE_MODE}
    assert chmod_targets == set(written_paths)

    if sys.platform == "win32":
        return

    for path in written_paths:
        mode_bits = stat.S_IMODE(os.stat(path).st_mode)
        assert mode_bits == ARTIFACT_FILE_MODE, f"{path} mode {oct(mode_bits)} != {oct(ARTIFACT_FILE_MODE)}"
