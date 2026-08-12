"""Write-queue size limits, corruption recovery, and drain behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._constants import (
    DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT,
    WRITE_QUEUE_FILENAME,
    WRITE_QUEUE_MAX_BYTES_PER_DRAIN,
)
from aetherdialect._contracts_base import ConfigError
from aetherdialect._contracts_core import WriteQueueEvent
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import (
    drain_diagnostic_collector,
    pop_engine_limits,
    push_engine_limits,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._utils_artifacts import emit_write_queue_event


@pytest.mark.fast
def test_oversized_record_refused_at_write(tmp_path) -> None:
    limits = EngineLimits(write_queue_max_record_bytes=64)
    token = push_engine_limits(limits)
    try:
        ev = WriteQueueEvent(
            kind="feedback_record",
            schema_graph_id="sg_test000000000001__abcd1234",
            schema_hash="h",
            produced_at="2020-01-01T00:00:00+00:00",
            payload=(("q_norm", "x" * 200),),
        )
        with pytest.raises(ConfigError, match="write_queue_max_record_bytes"):
            emit_write_queue_event(str(tmp_path), ev, space_name="master")
        assert not (tmp_path / WRITE_QUEUE_FILENAME).is_file()
    finally:
        pop_engine_limits(token)


@pytest.mark.fast
def test_unparseable_queue_moved_aside_and_reported(tmp_path) -> None:
    adir = str(tmp_path)
    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    queue_path.write_bytes(b"x" * (WRITE_QUEUE_MAX_BYTES_PER_DRAIN + 1))

    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.schema_graph_id = "h1"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._dialect = None

    buf: list = []
    token = set_diagnostic_collector(buf)
    try:
        applied = MainExecutionOps.drain_write_queue(owner, adir)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert applied == 0
    assert not queue_path.is_file()
    corrupt_files = list(tmp_path.glob("write_queue.corrupt.*.jsonl"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].stat().st_size == WRITE_QUEUE_MAX_BYTES_PER_DRAIN + 1
    assert DIAGNOSTIC_CODE_WRITE_QUEUE_CORRUPT in [d.code for d in diags]


@pytest.mark.fast
def test_emit_write_queue_event_requires_space_name(tmp_path) -> None:
    ev = WriteQueueEvent(
        kind="feedback_record",
        schema_graph_id="sg_test000000000001__abcd1234",
        schema_hash="h",
        produced_at="2020-01-01T00:00:00+00:00",
        payload=(("q_norm", "q"),),
    )
    with pytest.raises(TypeError):
        emit_write_queue_event(str(tmp_path), ev)


@pytest.mark.fast
def test_drain_skips_events_missing_space_name(tmp_path) -> None:
    graph_id = "sg_drain000000000001__abcd1234"
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.schema_graph_id = graph_id
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._dialect = None
    line = (
        '{"kind":"override_proposal","schema_graph_id":"'
        f'{graph_id}","schema_hash":"h",'
        '"produced_at":"2020-01-01T00:00:00+00:00","payload":[["document_json","{}"]]}'
    )
    (tmp_path / WRITE_QUEUE_FILENAME).write_text(line + "\n", encoding="utf-8")
    assert MainExecutionOps.drain_write_queue(owner, str(tmp_path)) == 0
    assert (
        not (tmp_path / WRITE_QUEUE_FILENAME).exists()
        or (tmp_path / WRITE_QUEUE_FILENAME).read_text(encoding="utf-8").strip() == ""
    )
