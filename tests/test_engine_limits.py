"""Per-engine and federation limit dataclasses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, ConfigError
from aetherdialect._config import EngineLimits, FederationLimits
from aetherdialect._constants import DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED
from aetherdialect._core_utils import (
    active_engine_limits,
    active_federation_limits,
    narrow_member_engine_limits,
    pop_engine_limits,
    pop_federation_limits,
    push_engine_limits,
    push_federation_limits,
)


@pytest.mark.fast
def test_defaults_match_documented_table() -> None:
    limits = EngineLimits()
    assert limits.pool_size == 1
    assert limits.pool_max_overflow == 4
    assert limits.pool_recycle_seconds == 1800
    assert limits.pool_pre_ping is True
    assert limits.pool_timeout_seconds == 30
    assert limits.statement_timeout_ms == 30_000
    assert limits.profile_timeout_ms == 120_000
    assert limits.max_result_rows == 100_000
    assert limits.max_result_bytes == 268_435_456
    assert limits.result_fetch_batch_rows == 10_000
    assert limits.prompt_payload_max_bytes == 262_144
    assert limits.write_queue_max_record_bytes == 1_048_576
    assert limits.template_value_history_depth == 64
    assert limits.feedback_rows_per_question == 8
    assert limits.template_partition_cache_size == 32
    assert limits.artifact_lock_timeout_seconds == 30
    assert limits.applied_map_archive_count == 3

    fed = FederationLimits()
    assert fed.max_members == 8
    assert fed.max_parallel_members == 4
    assert fed.member_row_cap == 100_000
    assert fed.member_bytes_cap == 268_435_456
    assert fed.member_probe_timeout_seconds == 10
    assert fed.transfer_max_bytes == 536_870_912
    assert fed.reduction_key_max_count == 10_000
    assert fed.plan_step_count_max == 32
    assert fed.coordinator_memory_limit_bytes == 2_147_483_648
    assert fed.coordinator_threads == 4


@pytest.mark.fast
def test_none_means_unlimited_for_every_optional_field() -> None:
    engine_limits = EngineLimits()
    for field_name in EngineLimits.unlimited_optional_fields():
        assert getattr(engine_limits, field_name) is None
    fed_limits = FederationLimits()
    for field_name in FederationLimits.unlimited_optional_fields():
        assert getattr(fed_limits, field_name) is None


@pytest.mark.fast
def test_two_engines_have_independent_limits() -> None:
    with patch.object(AetherEngine, "_initialize_engine_bundle") as init_mock:
        bundle = MagicMock()
        bundle.dialect = MagicMock()
        bundle.data_quality_report = None
        init_mock.return_value = bundle
        left = AetherEngine(MagicMock(), artifacts_dir="x", limits=EngineLimits(pool_size=2))
        right = AetherEngine(MagicMock(), artifacts_dir="y", limits=EngineLimits(pool_size=3))
    assert left.limits.pool_size == 2
    assert right.limits.pool_size == 3


@pytest.mark.fast
def test_invalid_limits_refused() -> None:
    with pytest.raises(ConfigError, match="pool_size"):
        EngineLimits(pool_size=0)
    with pytest.raises(ConfigError, match="pool_max_overflow"):
        EngineLimits(pool_max_overflow=-1)
    with pytest.raises(ConfigError, match="max_result_rows"):
        EngineLimits(max_result_rows=-1)
    with pytest.raises(ConfigError, match="result_fetch_batch_rows"):
        EngineLimits(max_result_rows=100, result_fetch_batch_rows=200)
    with pytest.raises(ConfigError, match="max_parallel_members"):
        FederationLimits(max_members=2, max_parallel_members=4)


@pytest.mark.fast
def test_member_defaults_do_not_override_caller_supplied_member() -> None:
    member_limits = EngineLimits(max_result_rows=50_000)
    member = MagicMock()
    member.limits = member_limits
    fed_limits = FederationLimits(member_defaults=EngineLimits(max_result_rows=10_000))
    assert narrow_member_engine_limits(member.limits, fed_limits) is member_limits


@pytest.mark.fast
def test_stricter_member_limit_wins_and_reports() -> None:
    member_limits = EngineLimits(max_result_rows=200_000)
    fed_limits = FederationLimits(member_row_cap=100_000)
    with patch("aetherdialect._core_utils.notify") as notify_mock:
        narrowed = narrow_member_engine_limits(member_limits, fed_limits)
    assert narrowed.max_result_rows == 100_000
    notify_mock.assert_called()
    assert DIAGNOSTIC_CODE_MEMBER_LIMIT_NARROWED in str(notify_mock.call_args)


@pytest.mark.fast
def test_active_resolvers_require_context() -> None:
    with pytest.raises(RuntimeError, match="no active engine limits"):
        active_engine_limits()
    with pytest.raises(RuntimeError, match="no active federation limits"):
        active_federation_limits()
    token = push_engine_limits(EngineLimits(pool_size=2))
    try:
        assert active_engine_limits().pool_size == 2
    finally:
        pop_engine_limits(token)
    fed_token = push_federation_limits(FederationLimits(max_members=4))
    try:
        assert active_federation_limits().max_members == 4
    finally:
        pop_federation_limits(fed_token)
