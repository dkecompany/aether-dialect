"""AetherFederation warmup entrypoints are consistently unsupported."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import FEDERATION_WARMUP_UNSUPPORTED_MESSAGE
from aetherdialect._contracts_base import ConfigError


def _federation_stub() -> AetherFederation:
    fed = AetherFederation.__new__(AetherFederation)
    fed._closed = False
    fed._pipeline_writer_lock = MagicMock()
    fed._pipeline_writer_lock.__enter__ = MagicMock(return_value=None)
    fed._pipeline_writer_lock.__exit__ = MagicMock(return_value=False)
    return fed


@pytest.mark.fast
def test_all_warmup_entrypoints_consistent() -> None:
    fed = _federation_stub()
    entrypoints = (
        lambda: fed.run_seed_warmup("seed.txt"),
        lambda: fed.run_seed_warmup_from_history("history.sql"),
        lambda: fed.run_seed_warmup_from_query_log(),
    )
    with (
        patch.object(AetherFederation, "_require_production_api"),
        patch.object(AetherFederation, "_require_open"),
        patch.object(AetherFederation, "_ensure_llm") as ensure_llm,
        patch("aetherdialect.aetherdialect.seed_warmup_run_once") as run_once,
    ):
        messages: list[str] = []
        for call in entrypoints:
            with pytest.raises(ConfigError) as exc_info:
                call()
            messages.append(str(exc_info.value))
    assert messages == [FEDERATION_WARMUP_UNSUPPORTED_MESSAGE] * 3
    assert all(m == "warmup is not supported on AetherFederation" for m in messages)
    run_once.assert_not_called()
    ensure_llm.assert_not_called()
