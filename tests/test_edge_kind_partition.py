"""Join-path edge kinds route to JOIN or WHERE buckets with no silent default."""

from __future__ import annotations

import pytest

from aetherdialect._constants import JOIN_EDGE_KIND_RANK, JOIN_PATH_EDGE_KIND_WHERE_BUCKET, JOIN_PATH_EDGE_KINDS
from aetherdialect._sql_gen import _partition_path_join_vs_where


def test_declared_edge_kind_registry_has_fourteen_members() -> None:
    assert len(JOIN_PATH_EDGE_KINDS) == 14
    assert JOIN_PATH_EDGE_KINDS == frozenset(JOIN_EDGE_KIND_RANK)


@pytest.mark.parametrize("kind", sorted(JOIN_PATH_EDGE_KINDS))
def test_declared_edge_kind_routes_to_exactly_one_bucket(kind: str) -> None:
    sig = ["left.col->right.col"]
    join_b, where_b = _partition_path_join_vs_where(sig, [kind])
    if kind in JOIN_PATH_EDGE_KIND_WHERE_BUCKET:
        assert join_b == []
        assert len(where_b) == 1
    else:
        assert len(join_b) == 1
        assert where_b == []
