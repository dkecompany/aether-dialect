"""Deterministic-repair pipeline-trace headings come from a fixed vocabulary."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect import _constants_runtime
from aetherdialect._contracts_base import ConfigError
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._intent_loop import _trace_intent_after_deterministic_step


@pytest.mark.fast
def test_headings_come_from_the_fixed_set() -> None:
    heading_map = getattr(_constants_runtime, "DETERMINISTIC_REPAIR_TRACE_HEADINGS", None)
    assert isinstance(heading_map, dict) and heading_map
    sample_step = next(iter(heading_map))
    expected = heading_map[sample_step]
    captured: list[str] = []

    def _capture(heading: str, body: object) -> None:
        captured.append(heading)
        _ = body

    empty = RuntimeIntent()
    with (
        patch("aetherdialect._intent_loop.pipeline_trace", _capture),
        patch("aetherdialect._intent_loop.diagnostic_debug_enabled", return_value=True),
        patch("aetherdialect._intent_loop.diagnostic_pipeline_trace_full_enabled", return_value=True),
    ):
        _trace_intent_after_deterministic_step(sample_step, empty, empty, ["tables"])
    assert captured == [expected]
    with (
        patch("aetherdialect._intent_loop.pipeline_trace", _capture),
        patch("aetherdialect._intent_loop.diagnostic_debug_enabled", return_value=True),
        patch("aetherdialect._intent_loop.diagnostic_pipeline_trace_full_enabled", return_value=True),
        pytest.raises(ConfigError),
    ):
        _trace_intent_after_deterministic_step("__unknown_repair_step__", empty, empty, ["tables"])
