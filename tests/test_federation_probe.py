"""Federation member connectivity probes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._federation import probe_federation_member_connections


@pytest.mark.fast
def test_probe_succeeds_for_normally_constructed_member() -> None:
    conn = MagicMock()
    sa_engine = MagicMock()
    sa_engine.connect.return_value.__enter__.return_value = conn

    class _MemberStub:
        dialect = "postgresql"

        def __init__(self) -> None:
            self._runtime_config = MagicMock(engine="postgresql")
            self._execution_engine = None
            self._sa = sa_engine

        @property
        def live_connection_handle(self) -> MagicMock:
            return self._sa

    member = _MemberStub()
    with patch(
        "aetherdialect._federation._probe_member_session_timezone",
        return_value=None,
    ):
        probe_federation_member_connections({"a": member})
    sa_engine.connect.assert_called_once()
