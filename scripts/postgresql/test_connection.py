"""Verify connectivity and rental_shop load completeness for PostgreSQL."""

from __future__ import annotations

import argparse

ENGINE = "postgresql"


def main() -> None:
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1]
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from load_rental_shop_engines import _cmd_ping, _cmd_verify

    args = argparse.Namespace(engine=ENGINE, env_file=None, schema=None)
    _cmd_ping(args)
    _cmd_verify(args)


if __name__ == "__main__":
    main()
