"""Format-repair exhaustion must fail closed on unacceptable intents."""

from __future__ import annotations

import json
from unittest.mock import patch

from aetherdialect._intent_process import _format_repair_loop


def test_placeholder_intent_after_exhaustion_is_none() -> None:
    """Placeholder leakage after all repair rounds yields ``None``, not a leaky intent."""
    bad = json.dumps(
        {
            "tables": ["film"],
            "grain": "row_level",
            "select_cols": ["table_1.film_id"],
        }
    )
    with patch("aetherdialect._intent_process.LLMProvider.chat", return_value=bad) as chat:
        intent, calls = _format_repair_loop("sys", bad, "q", max_retries=2)
    assert calls == 2
    assert chat.call_count == 2
    assert intent is None
