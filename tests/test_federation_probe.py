"""Federation member connectivity probes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._federation_execute import probe_federation_member_connections


@pytest.mark.fast
def test_probe_succeeds_for_normally_constructed_member() -> None:
    conn = MagicMock()
    sa_engine = MagicMock()
    sa_engine.connect.return_value.__enter__.return_value = conn

    member = MagicMock()
    member.dialect = "postgresql"
    member._runtime_config = MagicMock(engine="postgresql")
    member._execution_engine = sa_engine

    with patch(
        "aetherdialect._federation_execute._probe_member_session_timezone",
        return_value=None,
    ):
        probe_federation_member_connections({"a": member})
    sa_engine.connect.assert_called_once()
