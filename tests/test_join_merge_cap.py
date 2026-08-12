"""Join merge cross-product refuses before materialising the full product."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

from aetherdialect._constants import JOIN_PATH_TIE_REFUSAL_CEILING
from aetherdialect._contracts_core import JoinCandidateCapExceededError
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import _candidate_join_paths_for_tables


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _cross_product_ambiguity_schema(variants_per_leg: int) -> SchemaGraph:
    tables: dict[str, TableMetadata] = {
        "root": TableMetadata(name="root", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        "t2": TableMetadata(name="t2", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        "t3": TableMetadata(name="t3", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
    }
    for index in range(variants_per_leg):
        for target in ("t2", "t3"):
            bridge = f"link_{target}_{index}"
            tables[bridge] = TableMetadata(
                name=bridge,
                columns={"id": _col("id"), "root_id": _col("root_id"), "target_id": _col("target_id")},
                primary_key=["id"],
                foreign_keys=[
                    FKEdge(src_table=bridge, src_cols=["root_id"], dst_table="root", dst_cols=["id"]),
                    FKEdge(src_table=bridge, src_cols=["target_id"], dst_table=target, dst_cols=["id"]),
                ],
            )
    join_paths_multi = recompute_join_paths_multi(tables)
    return SchemaGraph(tables=tables, join_paths_multi=join_paths_multi, effective_structural_hash="cross_product")


@pytest.mark.fast
def test_oversize_product_refuses_without_building_all() -> None:
    schema = _cross_product_ambiguity_schema(5)
    cap = 4
    product_calls: list[tuple[object, ...]] = []
    real_product = itertools.product

    def _counting_product(*args: object, **kwargs: object):
        product_calls.append(args)
        return real_product(*args, **kwargs)

    with patch("aetherdialect._sql_gen.itertools.product", side_effect=_counting_product):
        with pytest.raises(JoinCandidateCapExceededError) as exc_info:
            _candidate_join_paths_for_tables(
                schema,
                ["root", "t2", "t3"],
                cross_product_cap=cap,
                tie_cap=JOIN_PATH_TIE_REFUSAL_CEILING,
            )
    assert exc_info.value.enumerated == 25
    assert exc_info.value.cap == cap
    assert product_calls == []
