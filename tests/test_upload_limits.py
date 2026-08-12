"""Upload byte ceiling enforcement tests."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._contracts_base import ConfigError
from aetherdialect._data_quality import _resolve_tabular_upload_path, inspect_tabular_upload
from aetherdialect._utils import pop_engine_limits, push_engine_limits


@pytest.mark.fast
def test_oversize_upload_refused() -> None:
    limits = EngineLimits(max_upload_bytes=100)
    token = push_engine_limits(limits)
    try:
        oversized = b"x" * 101
        with pytest.raises(ConfigError, match=r"upload size 101 bytes exceeds max_upload_bytes \(100\)"):
            _resolve_tabular_upload_path(oversized, filename="big.csv")

        with pytest.raises(ConfigError, match=r"upload size 101 bytes exceeds max_upload_bytes \(100\)"):
            inspect_tabular_upload(oversized, filename="big.csv")
    finally:
        pop_engine_limits(token)
