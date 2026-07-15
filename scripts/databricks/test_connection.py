"""Verify connectivity and rental_shop load completeness for Databricks."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _engine_entry import run_rental_shop

ENGINE = "databricks"

if __name__ == "__main__":
    run_rental_shop(ENGINE, "ping")
    run_rental_shop(ENGINE, "verify")
