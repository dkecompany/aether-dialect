"""Bind-parameter substitution edge cases."""

from __future__ import annotations

import pytest

from aetherdialect._utils import reconcile_execute_bind_params, substitute_params


def test_substitute_params_rejects_empty_string_scalar() -> None:
    with pytest.raises(ValueError, match="unbound_placeholder"):
        substitute_params("SELECT :p1", {"p1": ""})


def test_substitute_params_handles_at_prefix_like_colon() -> None:
    sql = substitute_params("SELECT @p1", {"p1": "trailers"})
    assert sql == "SELECT 'trailers'"


def test_reconcile_execute_bind_params_collects_at_tokens() -> None:
    bound = reconcile_execute_bind_params("SELECT @p1 WHERE x = :p2", {"p1": "a", "p2": 1})
    assert bound == {"p1": "a", "p2": 1}
