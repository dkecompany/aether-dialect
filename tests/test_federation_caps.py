"""Member row-cap precedence for federated and single-engine enforcement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._config import FederationLimits
from aetherdialect._federation import resolve_member_row_cap


@pytest.mark.fast
@pytest.mark.parametrize(
    ("member_cap", "coordinator_cap", "limits_cap", "expected"),
    [
        (10, 100, 1000, 10),
        (None, 50, 1000, 50),
        (None, None, 75, 75),
    ],
    ids=["member", "coordinator", "limits"],
)
def test_member_cap_precedence(
    member_cap: int | None,
    coordinator_cap: int | None,
    limits_cap: int | None,
    expected: int | None,
) -> None:
    source = SimpleNamespace(source_id="s1", limits=SimpleNamespace(row_cap=member_cap))
    manifest = SimpleNamespace(
        sources={"s1": source},
        coordinator=SimpleNamespace(row_cap=coordinator_cap, default_source_row_cap=coordinator_cap),
    )
    limits = FederationLimits(member_row_cap=limits_cap)
    assert resolve_member_row_cap(manifest, "s1", limits) == expected
