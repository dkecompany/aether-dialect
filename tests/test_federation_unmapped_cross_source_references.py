"""Cross-source join values absent from the parent member must be reported, not ignored."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    detect_unmapped_cross_source_references,
    parse_federation_declaration,
)
from aetherdialect._schema_graph import recompute_join_paths_multi

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _sandbox_corpus():
    return importlib.import_module("sandbox_corpus")


def _read_seed(name: str) -> str:
    return (_DATA / name).read_text(encoding="utf-8")


def _column(name: str, *, sample: tuple[str, ...]) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        sensitivity="none",
        value_overlap_sample=list(sample),
        row_count=max(len(sample), 1),
    )


def _table(name: str, *, source_id: str, columns: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=columns,
        primary_key=[next(iter(columns))],
        foreign_keys=[],
        source_id=source_id,
    )


@pytest.mark.fast
def test_sandbox_logistics_seed_delivery_rentals_include_orphans_not_in_storefront() -> None:
    sc = _sandbox_corpus()
    logistics = _read_seed("federation_logistics_seed.sql")
    storefront = _read_seed("federation_storefront_seed.sql")
    delivery_rentals = {int(value) for value in sc.parse_seed_insert_column_values(logistics, "delivery", "rental_id")}
    storefront_rentals = {int(value) for value in sc.parse_seed_insert_column_values(storefront, "rental", "rental_id")}
    orphans = delivery_rentals - storefront_rentals
    assert sc.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS == (9999001, 9999002, 9999003)
    assert orphans.issuperset(set(sc.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS))


@pytest.mark.fast
def test_detect_unmapped_cross_source_references_reports_orphan_delivery_rentals() -> None:
    sc = _sandbox_corpus()
    logistics = _read_seed("federation_logistics_seed.sql")
    storefront = _read_seed("federation_storefront_seed.sql")
    delivery_sample = sc.parse_seed_insert_column_values(logistics, "delivery", "rental_id")
    rental_sample = sc.parse_seed_insert_column_values(storefront, "rental", "rental_id")
    members = {
        "logistics": SchemaGraph(
            tables={
                "delivery": _table(
                    "delivery",
                    source_id="logistics",
                    columns={"rental_id": _column("rental_id", sample=delivery_sample)},
                )
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
        "storefront": SchemaGraph(
            tables={
                "rental": _table(
                    "rental",
                    source_id="storefront",
                    columns={"rental_id": _column("rental_id", sample=rental_sample)},
                )
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
    }
    manifest, mappings = parse_federation_declaration(
        json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8")),
    )
    messages = detect_unmapped_cross_source_references(members, manifest, mappings=mappings)
    joined = " ".join(messages)
    for rental_id in sc.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS:
        assert str(rental_id) in joined
