"""LLM-proposed upload column transforms with deterministic verify-and- apply."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED
from aetherdialect._contracts_base import UploadColumnTransformId
from aetherdialect._contracts_schema import CsvSourceSelection
from aetherdialect._data_quality import prepare_relations_for_paths


def _mock_llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
    if task == "upload_summary":
        return {"summary": "Upload inspection completed."}
    if task == "upload_interpret":
        return {}
    if task == "upload_column_transforms":
        return {"column_transforms": []}
    raise AssertionError(f"unexpected llm_json task={task!r}")


@pytest.fixture(autouse=True)
def _patch_upload_llm() -> None:
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_mock_llm_json):
        yield


@pytest.fixture(autouse=True)
def _tabular_llm_assist_on() -> None:
    original = PolicyConfig.TABULAR_LLM_ASSIST
    PolicyConfig.TABULAR_LLM_ASSIST = True
    yield
    PolicyConfig.TABULAR_LLM_ASSIST = original


def _transform_payload(
    transform_id: str,
    *,
    column: str = "",
    params: dict[str, object] | None = None,
    requires_review: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "transform_id": transform_id,
        "requires_review": requires_review,
    }
    if column:
        body["column"] = column
    if params:
        body["params"] = params
    return body


def _prepare_with_transforms(
    path: Path,
    proposals: list[dict[str, object]],
    *,
    accepted: list[dict[str, object]] | None = None,
) -> list:
    transforms = accepted if accepted is not None else proposals

    def _llm(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
        if task == "upload_summary":
            return {"summary": "Upload inspection completed."}
        if task == "upload_interpret":
            return {}
        if task == "upload_column_transforms":
            return {"column_transforms": proposals}
        raise AssertionError(f"unexpected llm_json task={task!r}")

    selection = CsvSourceSelection(header_row=1, column_transforms=tuple(transforms))
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_llm):
        return prepare_relations_for_paths((path,), source_selections={path.name: selection})


@pytest.mark.fast
def test_strip_affix_bare_mixed_ok(tmp_path: Path) -> None:
    path = tmp_path / "qty.csv"
    path.write_text("qty\n10\nUSD 20\n30\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.STRIP_NUMERIC_AFFIX.value,
        column="qty",
        params={"affix_token": "USD"},
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert relation.column_types == ("INTEGER",)
    assert tuple(row["qty"] for row in relation.rows) == ("10", "20", "30")


@pytest.mark.fast
def test_strip_affix_second_unit_rejected(tmp_path: Path) -> None:
    path = tmp_path / "amount.csv"
    path.write_text("amount\nUSD 10\nAUD 10\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.STRIP_NUMERIC_AFFIX.value,
        column="amount",
        params={"affix_token": "USD"},
    )
    rejected: list[str] = []

    def _capture(_message: str, **kwargs: object) -> None:
        code = kwargs.get("code")
        if isinstance(code, str):
            rejected.append(code)

    with patch("aetherdialect._data_quality.notify", side_effect=_capture):
        relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert relation.column_types == ("VARCHAR",)
    assert DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED in rejected


@pytest.mark.fast
def test_strip_affix_uses_llm_exact_token_from_sample(tmp_path: Path) -> None:
    path = tmp_path / "custom.csv"
    path.write_text("amount,id\nXYZ10,1\nXYZ25,2\n10,3\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.STRIP_NUMERIC_AFFIX.value,
        column="amount",
        params={"affix_token": "XYZ"},
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert relation.column_types[0] == "INTEGER"
    assert tuple(row["amount"] for row in relation.rows) == ("10", "25", "10")


@pytest.mark.fast
def test_band_value_map_typo_collapses_to_canonical(tmp_path: Path) -> None:
    path = tmp_path / "tier.csv"
    path.write_text("tier\nHgh\nMed\nLow\nHgh\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.BAND_VALUE_MAP.value,
        column="tier",
        params={"value_map": {"Hgh": "High", "Med": "Medium", "Low": "Low"}},
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert tuple(row["tier"] for row in relation.rows) == ("High", "Medium", "Low", "High")


@pytest.mark.fast
def test_band_value_map_skipped_above_25_distinct(tmp_path: Path) -> None:
    rows = ["code"] + [f"v{i}" for i in range(26)]
    path = tmp_path / "wide.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.BAND_VALUE_MAP.value,
        column="code",
        params={"value_map": {"v0": "canonical"}},
    )
    captured_user: list[str] = []

    def _llm(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
        if task == "upload_column_transforms":
            captured_user.append(user)
            return {"column_transforms": [proposal]}
        if task == "upload_summary":
            return {"summary": "Upload inspection completed."}
        if task == "upload_interpret":
            return {}
        raise AssertionError(f"unexpected llm_json task={task!r}")

    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_llm):
        relations = prepare_relations_for_paths((path,))
    relation = relations[0]
    assert relation.rows[0]["code"] == "v0"
    if captured_user:
        payload = json.loads(captured_user[0])
        distincts = payload.get("column_distinct_values", {})
        assert "code" not in distincts or len(distincts.get("code", [])) <= 25


@pytest.mark.fast
def test_sample_prompt_sends_at_most_five_rows(tmp_path: Path) -> None:
    rows = ["id,value"] + [f"{i},{i}" for i in range(1, 21)]
    path = tmp_path / "many.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    captured_user: list[str] = []

    def _llm(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
        if task == "upload_column_transforms":
            captured_user.append(user)
            return {"column_transforms": []}
        if task == "upload_summary":
            return {"summary": "Upload inspection completed."}
        if task == "upload_interpret":
            return {}
        raise AssertionError(f"unexpected llm_json task={task!r}")

    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_llm):
        prepare_relations_for_paths((path,))
    assert captured_user
    payload = json.loads(captured_user[0])
    sample_rows = payload.get("sample_rows", [])
    assert len(sample_rows) <= 5


@pytest.mark.fast
def test_keep_canonical_drops_1to1_aliases(tmp_path: Path) -> None:
    path = tmp_path / "customers.csv"
    path.write_text("customer_id,cust_id,name\n1,1,alice\n2,2,bob\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.KEEP_CANONICAL_COLUMNS.value,
        params={"canonical_column": "customer_id", "alias_columns": ["cust_id"]},
        requires_review=True,
    )
    relations = _prepare_with_transforms(path, [proposal], accepted=[proposal])
    relation = relations[0]
    assert relation.columns == ("customer_id", "name")
    assert relation.rows[0]["customer_id"] == "1"


@pytest.mark.fast
def test_keep_canonical_rejected_when_not_functional(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("customer_id,cust_id,name\n1,1,alice\n1,2,bob\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.KEEP_CANONICAL_COLUMNS.value,
        params={"canonical_column": "customer_id", "alias_columns": ["cust_id"]},
        requires_review=True,
    )
    rejected: list[str] = []

    def _capture(_message: str, **kwargs: object) -> None:
        code = kwargs.get("code")
        if isinstance(code, str):
            rejected.append(code)

    with patch("aetherdialect._data_quality.notify", side_effect=_capture):
        relations = _prepare_with_transforms(path, [proposal], accepted=[proposal])
    relation = relations[0]
    assert "cust_id" in relation.columns
    assert DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED in rejected


@pytest.mark.fast
def test_derive_by_pattern_verified(tmp_path: Path) -> None:
    path = tmp_path / "sku.csv"
    path.write_text("sku,sku_num\nSKU-001,\nSKU-002,\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.DERIVE_BY_PATTERN.value,
        params={
            "source_column": "sku",
            "target_column": "sku_num",
            "pattern": r"^SKU-(\d+)$",
        },
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert tuple(row["sku_num"] for row in relation.rows) == ("001", "002")


@pytest.mark.fast
def test_null_tokens_from_proposal(tmp_path: Path) -> None:
    path = tmp_path / "status.csv"
    path.write_text("status,row_id\nN/A,1\nactive,2\n-,3\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.NULL_TOKENS.value,
        column="status",
        params={"tokens": ["N/A", "-"]},
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert relation.rows[0]["status"] == ""
    assert relation.rows[1]["status"] == "active"
    assert relation.rows[2]["status"] == ""


@pytest.mark.fast
def test_drop_empty_columns(tmp_path: Path) -> None:
    path = tmp_path / "sparse.csv"
    path.write_text("id,empty_col,name\n1,,alice\n2,,bob\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.DROP_EMPTY_COLUMNS.value,
        params={"columns": ["empty_col"]},
        requires_review=True,
    )
    relations = _prepare_with_transforms(path, [proposal], accepted=[proposal])
    relation = relations[0]
    assert "empty_col" not in relation.columns
    assert relation.columns == ("id", "name")


@pytest.mark.fast
def test_parse_temporal_text_dates_when_verified(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("ordered_on\n01/15/2024\n02/20/2024\n", encoding="utf-8")
    proposal = _transform_payload(
        UploadColumnTransformId.PARSE_TEMPORAL.value,
        column="ordered_on",
        params={"date_format": "%m/%d/%Y"},
    )
    relations = _prepare_with_transforms(path, [proposal])
    relation = relations[0]
    assert relation.column_types == ("DATE",)
    assert relation.rows[0]["ordered_on"] == "2024-01-15"
    assert relation.rows[1]["ordered_on"] == "2024-02-20"


@pytest.mark.fast
def test_tabular_llm_assist_off_skips_proposals(tmp_path: Path) -> None:
    path = tmp_path / "off.csv"
    path.write_text("status,row_id\nN/A,1\nactive,2\n", encoding="utf-8")
    called_tasks: list[str] = []

    def _llm(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, object]:
        called_tasks.append(task)
        if task == "upload_summary":
            return {"summary": "Upload inspection completed."}
        if task == "upload_interpret":
            return {}
        raise AssertionError(f"unexpected llm_json task={task!r}")

    PolicyConfig.TABULAR_LLM_ASSIST = False
    with patch("aetherdialect._data_quality.LLMProvider.json", side_effect=_llm):
        relations = prepare_relations_for_paths((path,))
    assert "upload_column_transforms" not in called_tasks
    assert relations[0].rows[0]["status"] == "N/A"
