"""Artifact lock file lifecycle."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aetherdialect._constants import ARTIFACT_LOCK_FILENAME
from aetherdialect._utils_artifacts import artifact_lock


@pytest.mark.fast
def test_lock_file_removed_after_release(tmp_path: Path) -> None:
    adir = str(tmp_path / "artifacts")
    lock_path = os.path.join(adir, ARTIFACT_LOCK_FILENAME)
    with artifact_lock(adir):
        assert os.path.isfile(lock_path)
    assert not os.path.isfile(lock_path)


@pytest.mark.fast
def test_concurrent_holders_do_not_remove_each_others_locks(tmp_path: Path) -> None:
    adir = str(tmp_path / "artifacts")
    lock_path = os.path.join(adir, ARTIFACT_LOCK_FILENAME)
    with artifact_lock(adir):
        assert os.path.isfile(lock_path)
        with artifact_lock(adir):
            assert os.path.isfile(lock_path)
        assert os.path.isfile(lock_path)
    assert not os.path.isfile(lock_path)
