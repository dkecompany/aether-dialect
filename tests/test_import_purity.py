"""Import-time purity: no environment reads or stream reconfiguration at import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_POISON_PREFIX = "__AETHERDIALECT_POISON__"
_POISON_ENV: dict[str, str] = {
    "OPENAI_API_KEY": f"{_POISON_PREFIX}openai",
    "AZURE_OPENAI_API_KEY": f"{_POISON_PREFIX}azure_token",
    "AZURE_OPENAI_BASE_URL": f"{_POISON_PREFIX}azure_base",
    "AZURE_OPENAI_ENDPOINT": f"{_POISON_PREFIX}azure_endpoint",
    "AZURE_OPENAI_API_VERSION": f"{_POISON_PREFIX}azure_version",
    "AETHERDIALECT_LLM_BATCH_ENABLED": "true",
    "AETHERDIALECT_TABULAR_LLM_ASSIST": "false",
    "POSTGRES_PASSWORD": f"{_POISON_PREFIX}pg",
    "POSTGRES_USER": f"{_POISON_PREFIX}pg_user",
    "POSTGRES_DB": f"{_POISON_PREFIX}pg_db",
}


def _run_isolated_script(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.fast
def test_import_reads_no_environment_variables() -> None:
    poison_repr = repr(_POISON_ENV)
    script = f"""
import importlib
import sys
from collections.abc import Mapping
from unittest.mock import patch
import os

class _RecordingEnviron(Mapping[str, str]):
    def __init__(self, base):
        self._base = dict(base)
        self.read_keys = []

    def __getitem__(self, key):
        self.read_keys.append(key)
        return self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def get(self, key, default=None):
        if key in self._base:
            self.read_keys.append(key)
            return self._base[key]
        return default

importlib.import_module("aetherdialect")
for name in list(sys.modules):
    if name == "aetherdialect" or name.startswith("aetherdialect."):
        del sys.modules[name]
importlib.invalidate_caches()

recording = _RecordingEnviron({poison_repr})
with patch("os.environ", recording):
    importlib.import_module("aetherdialect")
    from aetherdialect._config import EngineConfig, PolicyConfig

    if recording.read_keys:
        keys = sorted(set(recording.read_keys))
        print(f"import must not read os.environ; keys accessed: {{keys}}")
        sys.exit(1)
    if EngineConfig.API_TOKEN is not None:
        print("EngineConfig.API_TOKEN must stay None at import")
        sys.exit(1)
    if EngineConfig.AZURE_API_TOKEN is not None:
        print("EngineConfig.AZURE_API_TOKEN must stay None at import")
        sys.exit(1)
    if EngineConfig.AZURE_OPENAI_BASE_URL is not None:
        print("EngineConfig.AZURE_OPENAI_BASE_URL must stay None at import")
        sys.exit(1)
    if EngineConfig.AZURE_OPENAI_ENDPOINT is not None:
        print("EngineConfig.AZURE_OPENAI_ENDPOINT must stay None at import")
        sys.exit(1)
    if EngineConfig.AZURE_OPENAI_API_VERSION is not None:
        print("EngineConfig.AZURE_OPENAI_API_VERSION must stay None at import")
        sys.exit(1)
    if PolicyConfig.LLM_BATCH_ENABLED is not False:
        print("PolicyConfig.LLM_BATCH_ENABLED must stay False at import")
        sys.exit(1)
    if PolicyConfig.TABULAR_LLM_ASSIST is not True:
        print("PolicyConfig.TABULAR_LLM_ASSIST must stay True at import")
        sys.exit(1)
"""
    _run_isolated_script(script)


@pytest.mark.fast
def test_configuration_applied_at_construction() -> None:
    from aetherdialect._config import PolicyConfig

    poison_env = dict(_POISON_ENV)
    PolicyConfig.apply_environment(poison_env)
    assert PolicyConfig.LLM_BATCH_ENABLED is True
    assert PolicyConfig.TABULAR_LLM_ASSIST is False
    PolicyConfig.LLM_BATCH_ENABLED = False
    PolicyConfig.TABULAR_LLM_ASSIST = True


@pytest.mark.fast
def test_import_does_not_reconfigure_streams() -> None:
    script = """
import importlib
import sys

stdout_before = getattr(sys.stdout, "encoding", None)
stderr_before = getattr(sys.stderr, "encoding", None)

for name in list(sys.modules):
    if name == "aetherdialect" or name.startswith("aetherdialect."):
        del sys.modules[name]
importlib.invalidate_caches()
importlib.import_module("aetherdialect")

stdout_after = getattr(sys.stdout, "encoding", None)
stderr_after = getattr(sys.stderr, "encoding", None)
if stdout_after != stdout_before:
    print(f"stdout encoding changed: {stdout_before!r} -> {stdout_after!r}")
    sys.exit(1)
if stderr_after != stderr_before:
    print(f"stderr encoding changed: {stderr_before!r} -> {stderr_after!r}")
    sys.exit(1)
"""
    _run_isolated_script(script)
