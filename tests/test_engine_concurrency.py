"""Engine writer-lock contract on mutating facade methods."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import TemplateExecutionResult
from aetherdialect.aetherdialect import (
    MUTATING_ENGINE_METHODS,
    MUTATING_FEDERATION_METHODS,
    AetherEngine,
    AetherFederation,
    guarded_by_writer_lock,
)
from tests.test_aetherdialect import _make_aether_stub
from tests.test_reuse_saved_question import _minimal_template


@pytest.mark.fast
def test_every_mutating_method_takes_the_lock() -> None:
    for name in MUTATING_ENGINE_METHODS:
        method = getattr(AetherEngine, name)
        assert guarded_by_writer_lock(method), f"AetherEngine.{name} missing writer lock guard"
    for name in MUTATING_FEDERATION_METHODS:
        method = getattr(AetherFederation, name)
        assert guarded_by_writer_lock(method), f"AetherFederation.{name} missing writer lock guard"


@pytest.mark.fast
def test_concurrent_template_execution_is_safe() -> None:
    tmpl = _minimal_template()
    engine = _make_aether_stub(_templates={tmpl.id: tmpl}, _context_name="master")
    active = 0
    peak = 0
    counter_lock = threading.Lock()
    expected = TemplateExecutionResult(
        rows=((1,),),
        sql="SELECT 1",
        display_sql="SELECT 1",
        columns=("count",),
    )

    def slow_execute(*_args: object, **_kwargs: object) -> TemplateExecutionResult:
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with counter_lock:
            active -= 1
        return expected

    def run_once() -> None:
        engine.execute_template("T0001", {"p1": "y"})

    with patch("aetherdialect.aetherdialect.execute_stored_template_by_ref", side_effect=slow_execute):
        threads = [threading.Thread(target=run_once) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

    assert peak == 1, f"execute_template must serialize writers; peak concurrency was {peak}"
