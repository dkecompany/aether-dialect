"""SessionStep serialization includes parameters, sql dict, meta, and template_id."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ParameterBinding, SessionStep
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_roundtrip_parameters_sql_dict_meta_template_id() -> None:
    step = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        sql={"crm": "SELECT 1", "ops": "SELECT 2"},
        parameters=(ParameterBinding(handle="p1", current_value="2024-01-01", display_name="start date"),),
        template_id="T0001",
        meta_payload={"response_kind": "schema_catalog", "counts": {"tables": 2}},
    )
    payload = MainExecutionOps.serialize_session_step(step)
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored.sql == {"crm": "SELECT 1", "ops": "SELECT 2"}
    assert restored.template_id == "T0001"
    assert restored.meta_payload == {"response_kind": "schema_catalog", "counts": {"tables": 2}}
    assert len(restored.parameters) == 1
    assert restored.parameters[0].handle == "p1"
    assert restored.parameters[0].current_value == "2024-01-01"
