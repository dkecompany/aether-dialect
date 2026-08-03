"""Verify connectivity and rental_shop load completeness for MySQL."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

ENGINE = "mysql"

if __name__ == "__main__":
    import argparse

    from load_rental_shop_engines import DEFAULT_ENV_FILE, _cmd_ping, _cmd_verify

    args = argparse.Namespace(engine=ENGINE, env_file=DEFAULT_ENV_FILE, schema=None)
    _cmd_ping(args)
    _cmd_verify(args)
