"""Schema classify role/type degrade and honest failure preamble."""

from __future__ import annotations

from typing import Any

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import SCHEMA_CLASSIFY_ERROR_DETAIL_CAP
from aetherdialect._contracts_base import ColumnRole, TableRole
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_catalog import _coerce_llm_assigned_role, apply_column_roles_llm


def _usable_col(**kwargs: Any) -> ColumnMetadata:
    defaults: dict[str, Any] = {
        "distinct_count": 50,
        "distinct_ratio": 0.5,
        "null_ratio": 0.0,
    }
    defaults.update(kwargs)
    return ColumnMetadata(**defaults)


def _graph() -> SchemaGraph:
    col = _usable_col(name="label", data_type="varchar", value_type="string")
    table = TableMetadata(name="t", columns={"label": col}, primary_key=[], foreign_keys=[])
    return SchemaGraph(
        tables={"t": table},
        join_paths_multi={},
        effective_structural_hash="eff_hash",
        profiling_hash="profile_hash",
    )


@pytest.mark.fast
def test_string_column_numeric_role_degrades() -> None:
    graph = _graph()
    col = graph.tables["t"].columns["label"]
    out = _coerce_llm_assigned_role(col, "numeric_measure")
    assert out != "numeric_measure"
    assert out in {ColumnRole.CATEGORICAL.value, ColumnRole.FREE_TEXT.value, ColumnRole.IDENTIFIER.value}


@pytest.mark.fast
def test_no_llm_retry_burn_on_physical_type(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph()
    calls = {"n": 0}

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
    ):
        calls["n"] += 1
        payload = {
            "t": {
                "table_role": TableRole.FACT.value,
                "description": "table",
                "columns": {
                    "label": {
                        "role": ColumnRole.NUMERIC_MEASURE.value,
                        "description": "a label",
                        "sensitivity": None,
                    }
                },
            }
        }
        if cache_payload_out is not None:
            cache_payload_out.append(payload)
        return {
            "t": (
                TableRole.FACT.value,
                "table",
                {"label": (ColumnRole.NUMERIC_MEASURE.value, "a label", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_catalog.llm_classify_schema", fake_classify)
    apply_column_roles_llm(graph)
    assert calls["n"] == 1
    assert graph.tables["t"].columns["label"].role != ColumnRole.NUMERIC_MEASURE.value


@pytest.mark.fast
def test_error_preamble_names_role_type(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SCHEMA_CLASSIFY_ERROR_DETAIL_CAP == 50
    graph = _graph()

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
    ):
        return {
            "t": (
                TableRole.FACT.value,
                "table",
                {"label": (ColumnRole.CATEGORICAL.value, "", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_catalog.llm_classify_schema", fake_classify)
    monkeypatch.setattr(PolicyConfig, "MAX_ROLE_CLASSIFICATION_RETRIES", 0)
    with pytest.raises(RuntimeError, match="missing descriptions"):
        apply_column_roles_llm(graph)
