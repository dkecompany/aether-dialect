"""Prepare federated SQL plan public signature checks."""

from __future__ import annotations

import inspect

import pytest

from aetherdialect._pipeline_execute import prepare_federated_sql_plan


@pytest.mark.fast
def test_prepare_federated_sql_plan_has_no_plan_cache_hit_parameter() -> None:
    params = inspect.signature(prepare_federated_sql_plan).parameters
    assert "plan_cache_hit" not in params
