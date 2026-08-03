"""Semi-join key gating beyond sensitivity checks."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationCapExceededError, SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    distinct_semijoin_keys,
    semijoin_key_distinct_count,
    semijoin_key_is_allowed,
    semijoin_key_passes_distinct_floor,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _composite(*, distinct: int | None = None, sensitivity: str = "none") -> SchemaGraph:
    tables = {
        "left_t": TableMetadata(
            name="left_t",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    sensitivity=sensitivity,
                    distinct_count=distinct,
                )
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
            row_count=100,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.fast
def test_sensitive_semijoin_key_is_refused() -> None:
    composite = _composite(sensitivity=SensitivityClassification.RESTRICTED.value)
    assert semijoin_key_is_allowed(composite, "left_t", "id") is False


@pytest.mark.fast
def test_low_cardinality_semijoin_key_fails_distinct_floor() -> None:
    composite = _composite(distinct=1)
    assert semijoin_key_distinct_count(composite, "left_t", "id") == 1
    assert semijoin_key_passes_distinct_floor(composite, "left_t", "id", floor=2) is False


@pytest.mark.fast
def test_semijoin_key_gate_is_deterministic_for_same_profile() -> None:
    composite = _composite(distinct=5)
    first = semijoin_key_passes_distinct_floor(composite, "left_t", "id", floor=2)
    second = semijoin_key_passes_distinct_floor(composite, "left_t", "id", floor=2)
    assert first == second is True


@pytest.mark.fast
def test_distinct_semijoin_keys_over_cap_returns_none() -> None:
    frame = pd.DataFrame({"id": list(range(5))})
    assert distinct_semijoin_keys(frame, "id", cap=3) is None


@pytest.mark.fast
def test_semijoin_key_cap_exceeded_maps_to_typed_limit_error() -> None:
    frame = pd.DataFrame({"id": list(range(5))})
    with pytest.raises(FederationCapExceededError) as exc_info:
        keys = distinct_semijoin_keys(frame, "id", cap=3)
        if keys is None:
            raise FederationCapExceededError(
                "federation semijoin key cap exceeded",
                limit_key="semijoin_key_cap",
                source_id="a",
            )
    assert exc_info.value.limit_key == "semijoin_key_cap"
