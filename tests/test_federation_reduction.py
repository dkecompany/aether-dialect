"""Federation equality-reduction diagnostics when null join keys are dropped."""

from __future__ import annotations

import pandas as pd

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS
from aetherdialect._core_utils import reset_diagnostic_collector, set_diagnostic_collector
from aetherdialect._federation import distinct_semijoin_keys


def test_null_keys_reported_when_dropped() -> None:
    frame = pd.DataFrame({"id": [1, None, 2, None, 3]})
    buf: list = []
    tok = set_diagnostic_collector(buf)
    try:
        keys = distinct_semijoin_keys(frame, "id", cap=10)
        assert keys == [1, 2, 3]
        reduction_diags = [d for d in buf if d.code == DIAGNOSTIC_CODE_FEDERATION_REDUCTION_NULL_KEYS]
        assert len(reduction_diags) == 1
        assert dict(reduction_diags[0].details).get("dropped_count") == "2"
        assert dict(reduction_diags[0].details).get("column") == "id"
    finally:
        reset_diagnostic_collector(tok)
