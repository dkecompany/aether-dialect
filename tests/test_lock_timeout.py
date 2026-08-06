"""Artifact lock timeout errors name the holder and support retry."""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any
from unittest.mock import patch

import pytest

from aetherdialect._constants import ARTIFACT_LOCK_FILENAME, DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK
from aetherdialect._contracts_base import ArtifactLockTimeoutError, RetryableError
from aetherdialect._core_utils import (
    artifact_lock,
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


@pytest.mark.fast
def test_timeout_names_holder_and_is_retryable(tmp_path) -> None:
    adir = str(tmp_path / "artifacts")
    holder_acquired = threading.Event()
    release_holder = threading.Event()
    holder_pid: list[int] = []

    def holder() -> None:
        with artifact_lock(adir, timeout=10.0):
            holder_pid.append(os.getpid())
            holder_acquired.set()
            release_holder.wait(timeout=5.0)

    th = threading.Thread(target=holder)
    th.start()
    try:
        assert holder_acquired.wait(timeout=5.0)
        with pytest.raises(ArtifactLockTimeoutError) as exc_info:
            with artifact_lock(adir, timeout=0.5):
                pass
        exc = exc_info.value
        assert isinstance(exc, RetryableError)
        assert os.path.normcase(exc.artifacts_dir) == os.path.normcase(adir)
        assert exc.holder_pid == holder_pid[0]
    finally:
        release_holder.set()
        th.join(timeout=5.0)


@pytest.mark.fast
def test_stale_lock_from_dead_process_recovered(tmp_path, monkeypatch) -> None:
    adir = str(tmp_path / "artifacts")
    os.makedirs(adir, exist_ok=True)
    lock_path = os.path.join(adir, ARTIFACT_LOCK_FILENAME)
    dead_pid = 9_999_999
    payload = json.dumps({"pid": dead_pid, "monotonic": 0.0}).encode("utf-8")
    with open(f"{lock_path}.holder", "wb") as fh:
        fh.write(payload)
    with open(lock_path, "wb") as fh:
        if sys.platform == "win32":
            fh.write(b"\x00")
        else:
            fh.write(payload)

    monkeypatch.setattr(
        "aetherdialect._core_utils._process_exists",
        lambda pid: pid != dead_pid,
    )

    if sys.platform == "win32":
        import msvcrt

        original_locking = msvcrt.locking
        blocked_once = {"done": False}

        def fake_locking(fd: Any, mode: int, nbytes: int) -> None:
            if not blocked_once["done"] and mode == msvcrt.LK_NBLCK:
                blocked_once["done"] = True
                raise OSError
            return original_locking(fd, mode, nbytes)

        lock_patch = patch.object(msvcrt, "locking", fake_locking)
    else:
        import fcntl

        original_flock = fcntl.flock
        blocked_once = {"done": False}

        def fake_flock(fd: Any, op: int) -> None:
            if not blocked_once["done"] and (op & fcntl.LOCK_NB):
                blocked_once["done"] = True
                raise BlockingIOError
            return original_flock(fd, op)

        lock_patch = patch.object(fcntl, "flock", fake_flock)

    buf: list = []
    token = set_diagnostic_collector(buf)
    try:
        with lock_patch:
            with artifact_lock(adir, timeout=2.0):
                pass
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert not os.path.isfile(lock_path)
    assert not os.path.isfile(f"{lock_path}.holder")
    stale_codes = [d.code for d in diags if d.code == DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK]
    assert stale_codes, f"expected STALE_ARTIFACT_LOCK diagnostic, got {diags!r}"
