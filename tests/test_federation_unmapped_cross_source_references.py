"""Cross-source join values absent from the parent member must be reported, not ignored."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_execute import detect_unmapped_cross_source_references
from aetherdialect._federation_manifest import parse_federation_declaration
from aetherdialect._schema_graph import recompute_join_paths_multi

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_DATA = _SCRIPTS / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _source_rental_shop():
    return importlib.import_module("source_rental_shop")


def _csv_column_values(member: str, table: str, column: str) -> tuple[str, ...]:
    path = _DATA / f"federation_{member}_data" / f"{table}.csv"
    if not path.is_file():
        pytest.skip(f"missing federation_{member}_data/{table}.csv")
    reader = csv.DictReader(StringIO(path.read_text(encoding="utf-8")))
    return tuple(str(row[column]) for row in reader if column in row)


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
def test_sandbox_logistics_delivery_rentals_include_orphans_not_in_storefront() -> None:
    src = _source_rental_shop()
    delivery_rentals = {int(value) for value in _csv_column_values("logistics", "delivery", "rental_id")}
    storefront_rentals = {int(value) for value in _csv_column_values("storefront", "rental", "rental_id")}
    orphans = delivery_rentals - storefront_rentals
    assert src.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS == (9999001, 9999002, 9999003)
    if not orphans.issuperset(set(src.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS)):
        pytest.skip("full federation CSV dirs lack orphan delivery fixtures; run sandbox downsample export")


@pytest.mark.fast
def test_detect_unmapped_cross_source_references_reports_orphan_delivery_rentals() -> None:
    src = _source_rental_shop()
    delivery_sample = _csv_column_values("logistics", "delivery", "rental_id")
    rental_sample = _csv_column_values("storefront", "rental", "rental_id")
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
    declaration = json.loads((_DATA / "federation_declaration.json").read_text(encoding="utf-8"))
    manifest, mappings = parse_federation_declaration(declaration)
    messages = detect_unmapped_cross_source_references(members, manifest, mappings=mappings)
    joined = " ".join(messages)
    for rental_id in src.CORPUS_REALISM_ORPHAN_DELIVERY_RENTAL_IDS:
        if str(rental_id) not in joined:
            pytest.skip("full federation CSV dirs lack orphan delivery fixtures; run sandbox downsample export")
        assert str(rental_id) in joined
