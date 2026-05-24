"""Tests for the cross-platform reentrant ``artifact_lock``."""

from __future__ import annotations

import os
import threading
import time

import pytest

from aetherdialect._core_utils import artifact_lock, write_gzip_json_atomic


def test_artifact_lock_reentrant_same_thread(tmp_path):
    """Nested ``with artifact_lock(d):`` in one thread does not deadlock."""

    d = str(tmp_path)
    with artifact_lock(d):
        with artifact_lock(d):
            with artifact_lock(d):
                write_gzip_json_atomic(os.path.join(d, "x.json.gz"), {"v": 1}, sort_keys=True)


def test_artifact_lock_serializes_threads(tmp_path):
    """Two threads contending the same dir lock execute their critical sections serially."""

    d = str(tmp_path)
    in_section = 0
    max_concurrent = 0
    lock_state = threading.Lock()

    def worker():
        nonlocal in_section, max_concurrent
        with artifact_lock(d, timeout=10.0):
            with lock_state:
                in_section += 1
                if in_section > max_concurrent:
                    max_concurrent = in_section
            time.sleep(0.05)
            with lock_state:
                in_section -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max_concurrent == 1


def test_artifact_lock_timeout_raises(tmp_path):
    """If another holder never releases, ``artifact_lock`` raises ``TimeoutError``."""

    d = str(tmp_path)
    holder_acquired = threading.Event()
    release_holder = threading.Event()

    def holder():
        with artifact_lock(d, timeout=10.0):
            holder_acquired.set()
            release_holder.wait(timeout=5.0)

    th = threading.Thread(target=holder)
    th.start()
    try:
        assert holder_acquired.wait(timeout=5.0)
        with pytest.raises(TimeoutError):
            with artifact_lock(d, timeout=0.5):
                pass
    finally:
        release_holder.set()
        th.join(timeout=5.0)
