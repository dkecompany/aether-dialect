"""Restricted-environment configuration errors."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from aetherdialect._config import FederationLimits
from aetherdialect._contracts_base import ConfigError
from aetherdialect._federation_execute import _resolve_coordinator_temp_directory, compute_federation_storage_dir

pre_fix_failure: dict[str, str | None] = {
    "default_root": None,
    "spill_directory": None,
}


def _deny_write_access(monkeypatch: pytest.MonkeyPatch, blocked_root: Path) -> None:
    blocked = os.path.abspath(str(blocked_root))
    real_access = os.access

    def fake_access(path: str | os.PathLike[str], mode: int) -> bool:
        normalized = os.path.abspath(os.path.expanduser(str(path)))
        if mode == os.W_OK and (normalized == blocked or normalized.startswith(blocked + os.sep)):
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", fake_access)


@pytest.mark.fast
def test_unwritable_default_root_names_the_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = tmp_path / "blocked_user_data"
    blocked.mkdir()
    _deny_write_access(monkeypatch, blocked)
    monkeypatch.setattr(
        "aetherdialect._federation_execute.user_data_dir",
        lambda appname, appauthor: str(blocked),
    )

    with pytest.raises(ConfigError) as excinfo:
        compute_federation_storage_dir(None, "fed_test")

    message = str(excinfo.value)
    blocked_abs = os.path.abspath(str(blocked))
    if "explicit artifacts" not in message.lower() or blocked_abs.replace("\\", "") not in message.replace("\\", ""):
        pre_fix_failure["default_root"] = message
    assert blocked_abs.replace("\\", "") in message.replace("\\", ""), pre_fix_failure["default_root"]
    assert "explicit artifacts" in message.lower(), pre_fix_failure["default_root"]


@pytest.mark.fast
def test_unwritable_spill_directory_names_the_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = tmp_path / "blocked_temp"
    blocked.mkdir()
    _deny_write_access(monkeypatch, blocked)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(blocked))

    with pytest.raises(ConfigError) as excinfo:
        _resolve_coordinator_temp_directory(FederationLimits())

    message = str(excinfo.value)
    blocked_abs = os.path.abspath(str(blocked))
    if blocked_abs.replace("\\", "") not in message.replace("\\", "") or "writable" not in message.lower():
        pre_fix_failure["spill_directory"] = message
    assert blocked_abs.replace("\\", "") in message.replace("\\", ""), pre_fix_failure["spill_directory"]
    assert "writable" in message.lower(), pre_fix_failure["spill_directory"]
