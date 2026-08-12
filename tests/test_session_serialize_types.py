"""Session step serialization encodes non-JSON-native cell types."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from aetherdialect._contracts_core import SessionStep
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_decimal_datetime_bytes_json_round_trip() -> None:
    amount = Decimal("19.9900")
    when = datetime(2024, 3, 15, 10, 30, 0)
    ts = pd.Timestamp("2024-06-01T12:00:00")
    blob = b"hello"
    data = pd.DataFrame(
        [{"amount": amount, "when": when, "ts": ts, "blob": blob, "nullable": None}],
    )
    step = SessionStep(done=True, prompt=None, kind="ok", data=data)

    payload = MainExecutionOps.serialize_session_step(step)
    json.dumps(payload)

    record = payload["data"]["records"][0]
    assert record["amount"] == format(amount, "f")
    assert record["when"] == when.isoformat()
    assert record["ts"] == ts.isoformat()
    assert record["blob"] == base64.b64encode(blob).decode("ascii")
    assert record["nullable"] is None

    row = step.data.iloc[0]
    assert isinstance(row["amount"], Decimal)
    assert row["amount"] == amount
    assert isinstance(row["when"], datetime)
    assert row["when"] == when
    assert isinstance(row["ts"], pd.Timestamp)
    assert row["ts"] == ts
    assert isinstance(row["blob"], bytes)
    assert row["blob"] == blob
    assert row["nullable"] is None
