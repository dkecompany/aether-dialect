#!/usr/bin/env python3
"""Run actionlint on workflow files (bootstrap binary when not on PATH)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ACTIONLINT_VERSION = "1.7.7"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _cache_dir() -> Path:
    if platform.system().lower() == "windows":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(base) / "aetherdialect-tools"
    return Path.home() / ".cache" / "aetherdialect-tools"


def _platform_asset() -> tuple[str, str, str]:
    """Return (asset_suffix, archive_kind, binary_name)."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        suffix = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
        return f"windows_{suffix}", "zip", "actionlint.exe"
    if system == "darwin":
        suffix = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
        return f"darwin_{suffix}", "tar.gz", "actionlint"
    if machine in {"arm64", "aarch64"}:
        return "linux_arm64", "tar.gz", "actionlint"
    return "linux_amd64", "tar.gz", "actionlint"


def ensure_actionlint() -> Path:
    on_path = shutil.which("actionlint")
    if on_path:
        return Path(on_path)

    asset, archive_kind, binary_name = _platform_asset()
    cache_dir = _cache_dir()
    exe = cache_dir / binary_name
    if exe.is_file():
        return exe

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"actionlint.{archive_kind}"
    url = (
        f"https://github.com/rhysd/actionlint/releases/download/v{ACTIONLINT_VERSION}/"
        f"actionlint_{ACTIONLINT_VERSION}_{asset}.{archive_kind}"
    )
    print(f"Downloading actionlint {ACTIONLINT_VERSION} ({asset})...", file=sys.stderr)
    urllib.request.urlretrieve(url, archive_path)
    if archive_kind == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract(binary_name, cache_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extract(binary_name, cache_dir)
    archive_path.unlink(missing_ok=True)
    if platform.system().lower() != "windows":
        exe.chmod(0o755)
    return exe


def main() -> int:
    exe = ensure_actionlint()
    if not WORKFLOWS.is_dir():
        return 0
    workflows = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    if not workflows:
        return 0
    return subprocess.call([str(exe), *[str(path) for path in workflows]])


if __name__ == "__main__":
    raise SystemExit(main())
