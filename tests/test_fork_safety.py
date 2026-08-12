"""Fork inheritance safety for live database handles."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from aetherdialect._dialect import Dialect
from aetherdialect._utils_artifacts import (
    assert_connection_usable_after_fork,
    register_dialect_live_handles,
    register_live_connection,
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="platform has no fork")
@pytest.mark.fast
def test_child_using_inherited_handle_raises() -> None:
    connection = object()
    dialect = MagicMock()
    dialect.supports_statement_cancellation = True
    register_live_connection(connection)
    register_dialect_live_handles(dialect)

    pid = os.fork()
    if pid == 0:
        try:
            assert_connection_usable_after_fork(connection)
            Dialect.cancel_in_flight_statement(dialect)
            os._exit(2)
        except RuntimeError:
            os._exit(0)
        except Exception:
            os._exit(3)
    else:
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
