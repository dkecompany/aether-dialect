"""restore_serialized_state must preserve reader mode and consumer gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import OwnerOnlyOperationError
from aetherdialect._main_session import PipelineSession


@pytest.mark.fast
def test_reader_restore_stays_reader() -> None:
    owner = MagicMock()
    owner._schema_role = "consumer"
    owner._sandbox_closed = False
    owner._active_space_name = "analytics"

    restored_reader = PipelineSession.__new__(PipelineSession)
    restored_reader._session_mode = "reader"
    restored_reader._space_name = "analytics"
    restored_reader._data_row_cap = 25

    def _session(**kwargs):
        mode = kwargs.get("mode", "reader")
        if owner._schema_role == "consumer" and mode == "writer":
            raise OwnerOnlyOperationError("PipelineSession(mode='writer')")
        restored_reader._session_mode = mode
        restored_reader._space_name = str(kwargs.get("space", "master"))
        restored_reader._data_row_cap = kwargs.get("data_row_cap")
        return restored_reader

    owner.session = MagicMock(side_effect=_session)

    payload = {
        "format_version": 1,
        "state_id": "execute",
        "message": "confirm",
        "choice_queue": [],
        "turn_question": "how many",
        "mode": "reader",
        "space_name": "analytics",
        "data_row_cap": 25,
    }

    with patch(
        "aetherdialect._main_session.MainSessionSerdeOps.deserialize_suspended_state",
        return_value={
            "state_id": "execute",
            "message": "confirm",
            "choice_queue": [],
            "turn_question": "how many",
            "mode": "reader",
            "space_name": "analytics",
            "data_row_cap": 25,
            "resume_choice_stage_id": None,
            "suspend_payload": None,
        },
    ):
        sess = PipelineSession.restore_serialized_state(owner, payload)

    owner.session.assert_called_once_with(mode="reader", space="analytics", data_row_cap=25)
    assert sess._session_mode == "reader"
    assert sess.space_name == "analytics"
    assert sess._data_row_cap == 25

    with pytest.raises(OwnerOnlyOperationError):
        _session(mode="writer", space="analytics")
