"""Load rental_shop CSVs into MySQL (legacy dvdrental_new entry name)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_ENGINE = "mysql"


def main() -> int:
    flag = _ENGINE.replace("_", "-")
    return subprocess.call(
        [sys.executable, str(_SCRIPTS / "load_rental_shop_engines.py"), f"--{flag}", *sys.argv[1:]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
