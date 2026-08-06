"""sync_to_public.py mirrors the private repo into the public clone."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = ROOT / "sync_to_public.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_to_public", SYNC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.fast
def test_default_writes_without_apply_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sync_module()
    dest = tmp_path / "public_clone"
    dest.mkdir()
    (dest / ".git").mkdir()

    rel_paths = ["README.md", ".gitignore", "pyproject.toml"]
    monkeypatch.setattr(module, "_run_git_ls_files", lambda repo_root: rel_paths)

    class _ProcResult:
        returncode = 0
        stdout = ""

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[:3] == ["git", "ls-files", "--"]:
            result = _ProcResult()
            result.stdout = "\n".join(rel_paths) + "\n"
            return result
        if cmd[:2] == ["git", "status"]:
            return _ProcResult()
        raise AssertionError(f"unexpected subprocess: {cmd}")

    monkeypatch.setattr(module.subprocess, "run", fake_subprocess_run)

    rc = module.main(["--dest", str(dest)])

    assert rc == 0
    copied = dest / "README.md"
    assert copied.is_file(), "default sync must copy files without --apply"
    assert copied.read_text(encoding="utf-8") == (ROOT / "README.md").read_text(encoding="utf-8")
