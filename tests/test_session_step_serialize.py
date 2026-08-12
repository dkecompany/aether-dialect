"""SessionStep serialization includes parameters, sql dict, and template_id."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_core import ParameterBinding, SessionStep
from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_roundtrip_parameters_sql_dict_template_id() -> None:
    step = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        sql={"crm": "SELECT 1", "ops": "SELECT 2"},
        parameters=(ParameterBinding(handle="p1", current_value="2024-01-01", display_name="start date"),),
        template_id="T0001",
        answer="schema catalog with 2 tables",
    )
    payload = MainExecutionOps.serialize_session_step(step)
    restored = MainExecutionOps.deserialize_session_step(payload)
    assert restored.sql == {"crm": "SELECT 1", "ops": "SELECT 2"}
    assert restored.template_id == "T0001"
    assert restored.answer == "schema catalog with 2 tables"
    assert len(restored.parameters) == 1
    assert restored.parameters[0].handle == "p1"
    assert restored.parameters[0].current_value == "2024-01-01"
