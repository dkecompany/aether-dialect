"""Role/value_type coercion during schema LLM classification."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED
from aetherdialect._constants_runtime import (
    COMPOSE_FIELDS,
    GROUND_FIELDS,
    SCHEMA_CLASSIFY_SYSTEM,
    SCHEMA_FIELD_RAW_TYPE,
)
from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata, TableRole
from aetherdialect._schema_profile import (
    _build_column_profile_for_llm,
    _schema_classification_cache_path,
    apply_column_roles_llm,
    llm_classify_schema,
    schema_classification_content_hash,
)


def _usable_col(**kwargs: Any) -> ColumnMetadata:
    defaults: dict[str, Any] = {
        "distinct_count": 50,
        "distinct_ratio": 0.5,
        "null_ratio": 0.0,
    }
    defaults.update(kwargs)
    return ColumnMetadata(**defaults)


def _fact_table(name: str, columns: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=columns, primary_key=[], foreign_keys=[])


def _graph(table: TableMetadata) -> SchemaGraph:
    return SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="eff_hash",
        profiling_hash="profile_hash",
        tables={table.name: table},
    )


def _classify_payload(
    table_name: str,
    table_role: str,
    table_description: str,
    columns: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    return {
        table_name: {
            "table_role": table_role,
            "description": table_description,
            "columns": {
                col_name: {"role": role, "description": desc, "sensitivity": None}
                for col_name, (role, desc) in columns.items()
            },
        }
    }


@pytest.mark.fast
def test_string_revenue_coerces_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    revenue = _usable_col(
        name="revenue",
        data_type="varchar",
        value_type="string",
    )
    table = _fact_table("sales", {"revenue": revenue})
    sg = _graph(table)
    notified: list[tuple[str, str]] = []

    def capture_notify(message: str, *, code: str = "", **_kwargs: Any) -> None:
        notified.append((message, code))

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
        structural_knowledge=None,
    ):
        payload = _classify_payload(
            "sales",
            TableRole.FACT.value,
            "sales facts",
            {"revenue": (ColumnRole.NUMERIC_MEASURE.value, "total revenue")},
        )
        if cache_payload_out is not None:
            cache_payload_out.append(payload)
        return {
            "sales": (
                TableRole.FACT.value,
                "sales facts",
                {"revenue": (ColumnRole.NUMERIC_MEASURE.value, "total revenue", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_profile.notify", capture_notify)
    monkeypatch.setattr("aetherdialect._schema_profile.llm_classify_schema", fake_classify)
    apply_column_roles_llm(sg)
    assert revenue.role != ColumnRole.NUMERIC_MEASURE.value
    assert revenue.description == "total revenue"
    assert any(code == DIAGNOSTIC_CODE_SCHEMA_ROLE_TYPE_COERCED for _msg, code in notified)


@pytest.mark.fast
def test_string_promo_type_never_stays_numeric_categorical(monkeypatch: pytest.MonkeyPatch) -> None:
    promo_type = _usable_col(
        name="promo_type",
        data_type="text",
        value_type="string",
        distinct_count=4,
        distinct_ratio=0.04,
    )
    table = _fact_table("promotions", {"promo_type": promo_type})
    sg = _graph(table)

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
        structural_knowledge=None,
    ):
        return {
            "promotions": (
                TableRole.DIMENSION.value,
                "promotion lookup",
                {"promo_type": (ColumnRole.NUMERIC_CATEGORICAL.value, "promotion tier code", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_profile.llm_classify_schema", fake_classify)
    apply_column_roles_llm(sg)
    assert promo_type.role == ColumnRole.CATEGORICAL.value


@pytest.mark.fast
def test_type_mismatch_does_not_retry_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    revenue = _usable_col(name="revenue", data_type="varchar", value_type="string")
    table = _fact_table("sales", {"revenue": revenue})
    sg = _graph(table)
    calls = {"n": 0}

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
        structural_knowledge=None,
    ):
        calls["n"] += 1
        return {
            "sales": (
                TableRole.FACT.value,
                "sales facts",
                {"revenue": (ColumnRole.NUMERIC_MEASURE.value, "total revenue", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_profile.llm_classify_schema", fake_classify)
    apply_column_roles_llm(sg)
    assert calls["n"] == 1


@pytest.mark.fast
def test_empty_description_retries_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    col = ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
    table = TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])
    sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={"t": table})
    calls = {"n": 0}

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
        structural_knowledge=None,
    ):
        calls["n"] += 1
        return {
            "t": (
                TableRole.DIMENSION.value,
                "",
                {"id": (ColumnRole.IDENTIFIER.value, "", None)},
            )
        }

    monkeypatch.setattr("aetherdialect._schema_profile.llm_classify_schema", fake_classify)
    with pytest.raises(RuntimeError, match="Schema LLM classification failed"):
        apply_column_roles_llm(sg)
    assert calls["n"] == PolicyConfig.MAX_ROLE_CLASSIFICATION_RETRIES + 1


@pytest.mark.fast
def test_cache_not_written_until_validation_passes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revenue = _usable_col(name="revenue", data_type="varchar", value_type="string")
    table = _fact_table("sales", {"revenue": revenue})
    sg = _graph(table)
    cache_path = tmp_path / "schema_graph.json.gz"
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(cache_path))
    scope = {"sales": frozenset({"revenue"})}
    content_hash = schema_classification_content_hash(sg, None, scope, structural_knowledge=())
    expected_cache = _schema_classification_cache_path(content_hash)
    assert expected_cache is not None
    mode = {"fail": True}

    def fake_classify(
        schema: SchemaGraph,
        notes_content: str | None = None,
        *,
        column_scope: dict[str, frozenset[str]] | None = None,
        cache_payload_out: list[dict[str, Any]] | None = None,
        structural_knowledge=None,
    ):
        payload = _classify_payload(
            "sales",
            TableRole.FACT.value,
            "sales facts" if not mode["fail"] else "",
            {"revenue": (ColumnRole.CATEGORICAL.value, "" if mode["fail"] else "promotion category")},
        )
        if cache_payload_out is not None:
            cache_payload_out.append(payload)
        return {
            "sales": (
                TableRole.FACT.value,
                "sales facts" if not mode["fail"] else "",
                {
                    "revenue": (
                        ColumnRole.CATEGORICAL.value,
                        "" if mode["fail"] else "promotion category",
                        None,
                    )
                },
            )
        }

    monkeypatch.setattr("aetherdialect._schema_profile.llm_classify_schema", fake_classify)
    with pytest.raises(RuntimeError, match="Schema LLM classification failed"):
        apply_column_roles_llm(sg)
    assert not expected_cache.is_file()

    mode["fail"] = False
    apply_column_roles_llm(sg)
    assert expected_cache.is_file()


@pytest.mark.fast
def test_classify_payload_has_value_type_not_data_type(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    col = _usable_col(name="amount", data_type="numeric", value_type="number", is_primary_key=True)
    table = TableMetadata(name="t", columns={"amount": col}, primary_key=["amount"], foreign_keys=[])
    sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={"t": table})
    captured_users: list[str] = []

    def fake_llm(
        system: str,
        user: str,
        max_retries: int = 3,
        timeout: Any = None,
        task: str = "default",
    ) -> str:
        captured_users.append(user)
        return json.dumps(
            _classify_payload(
                "t",
                TableRole.FACT.value,
                "amounts",
                {"amount": (ColumnRole.NUMERIC_MEASURE.value, "order amount")},
            )
        )

    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(tmp_path / "schema_graph.json.gz"))
    monkeypatch.setattr("aetherdialect._schema_profile.LLMProvider.chat", fake_llm)
    llm_classify_schema(sg, None)
    assert captured_users
    user_payload = json.loads(captured_users[0])
    column_profiles = user_payload["tables"][0]["columns"]
    assert column_profiles
    profile = column_profiles[0]
    assert "value_type" in profile
    assert "data_type" not in profile
    assert profile["value_type"] == "number"


@pytest.mark.fast
def test_schema_literal_never_emits_raw_type() -> None:
    col = ColumnMetadata(name="shape", data_type="geometry", value_type="unknown", description="Map outline")
    graph = SchemaGraph(
        tables={
            "sites": TableMetadata(
                name="sites",
                columns={
                    "site_id": ColumnMetadata(
                        name="site_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "shape": col,
                },
                primary_key=["site_id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="test",
    )
    for fields in (GROUND_FIELDS, COMPOSE_FIELDS):
        payload = json.loads(graph.schema_payload_json(fields, owner_master_scope=True))
        shape_body = payload["sites"]["columns"]["shape"]
        assert shape_body["type"] == "unknown"
        assert SCHEMA_FIELD_RAW_TYPE not in shape_body


@pytest.mark.fast
def test_classify_prompt_has_no_data_type_does_not_restrict() -> None:
    lowered = SCHEMA_CLASSIFY_SYSTEM.lower()
    assert "data type does not restrict" not in lowered
    assert "data_type" not in SCHEMA_CLASSIFY_SYSTEM
    assert "value_type" in SCHEMA_CLASSIFY_SYSTEM


@pytest.mark.fast
def test_build_column_profile_for_llm_uses_value_type_only() -> None:
    col = ColumnMetadata(name="revenue", data_type="varchar", value_type="string", is_primary_key=False)
    profile = _build_column_profile_for_llm(col)
    assert profile["value_type"] == "string"
    assert "data_type" not in profile
