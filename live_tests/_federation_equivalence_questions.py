"""Deterministic question generation for federation vs single-engine equivalence."""

from __future__ import annotations

from dataclasses import dataclass

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._contracts_base import ColumnRole
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._core_utils import is_date_type, is_numeric_type
from aetherdialect._qsim import get_aggregatable_columns, get_groupable_columns


@dataclass(frozen=True)
class FederationEquivalenceQuestion:
    """One schema-derived natural-language question for equivalence checking."""

    question_id: str
    question: str
    category: str


_AGG_FUNCS: tuple[tuple[str, str], ...] = (
    ("sum", "total"),
    ("avg", "average"),
    ("min", "minimum"),
    ("max", "maximum"),
    ("count", "count"),
)


def _column_roles(schema: SchemaGraph) -> dict[str, str]:
    roles: dict[str, str] = {}
    for table_name, table in sorted(schema.tables.items()):
        for col_name, col in sorted(table.columns.items()):
            key = f"{table_name}.{col_name}"
            role = (col.role or "").strip()
            if role:
                roles[key] = role
            elif is_numeric_type(str(col.data_type or "")):
                roles[key] = ColumnRole.NUMERIC_MEASURE.value
            elif is_date_type(str(col.data_type or "")):
                roles[key] = ColumnRole.TEMPORAL.value
            else:
                roles[key] = ColumnRole.CATEGORICAL.value
    return roles


def _joinable_table_pairs(schema: SchemaGraph) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    join_paths = schema.join_paths_multi or {}
    tables = sorted(join_paths.keys())
    for left_index, left in enumerate(tables):
        left_row = join_paths.get(left, {})
        for right in tables[left_index + 1 :]:
            right_paths = left_row.get(right) or join_paths.get(right, {}).get(left) or []
            if right_paths:
                pairs.append((left, right))
    return pairs


def _date_columns(schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    found: list[str] = []
    for table_name, table in sorted(schema.tables.items()):
        for col_name, col in sorted(table.columns.items()):
            key = f"{table_name}.{col_name}"
            role = column_roles.get(key, "")
            if role == ColumnRole.TEMPORAL.value or is_date_type(str(col.data_type or "")):
                found.append(key)
    return found


def _humanize_table(table_name: str) -> str:
    return table_name.replace("_", " ")


def _humanize_column(column_key: str) -> str:
    return column_key.split(".", 1)[-1].replace("_", " ")


def generate_federation_equivalence_questions(schema: SchemaGraph) -> list[FederationEquivalenceQuestion]:
    """Build a deterministically ordered corpus question set from schema shapes."""
    column_roles = _column_roles(schema)
    questions: list[FederationEquivalenceQuestion] = []

    for left, right in _joinable_table_pairs(schema):
        left_label = _humanize_table(left)
        right_label = _humanize_table(right)
        questions.append(
            FederationEquivalenceQuestion(
                question_id=f"join_count:{left}:{right}",
                question=f"how many {left_label} rows are linked to {right_label}",
                category="join_pair",
            )
        )
        questions.append(
            FederationEquivalenceQuestion(
                question_id=f"join_distinct_left:{left}:{right}",
                question=f"how many distinct {left_label} rows are linked to {right_label}",
                category="join_pair",
            )
        )

    for table_name in sorted(schema.tables.keys()):
        table_label = _humanize_table(table_name)
        aggregatable = get_aggregatable_columns(table_name, schema, column_roles)
        groupable = get_groupable_columns(table_name, schema, column_roles)
        for column_key in aggregatable:
            column_label = _humanize_column(column_key)
            for agg_func, phrase in _AGG_FUNCS:
                if agg_func == "count":
                    question_text = f"what is the {phrase} of {table_label} rows"
                else:
                    question_text = f"what is the {phrase} {column_label} for {table_label}"
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"aggregate:{agg_func}:{column_key}",
                        question=question_text,
                        category="aggregate",
                    )
                )
        for group_key in groupable:
            group_label = _humanize_column(group_key)
            for measure_key in aggregatable:
                measure_label = _humanize_column(measure_key)
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"grouping:sum:{measure_key}:by:{group_key}",
                        question=f"what is the total {measure_label} grouped by {group_label} for {table_label}",
                        category="grouping",
                    )
                )
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"grouping:count:{group_key}",
                        question=f"how many {table_label} rows are there grouped by {group_label}",
                        category="grouping",
                    )
                )

    for date_key in _date_columns(schema, column_roles):
        table_name, column_name = date_key.split(".", 1)
        table_label = _humanize_table(table_name)
        column_label = _humanize_column(date_key)
        for preset in SeedWarmupConfig.DATE_WINDOW_EXPANSION_PRESETS:
            unit = str(preset["unit"])
            amount = int(preset["amount"])
            unit_label = unit if amount == 1 else f"{unit}s"
            questions.append(
                FederationEquivalenceQuestion(
                    question_id=f"date_window:{date_key}:{unit}:{amount}",
                    question=(f"how many {table_label} rows have {column_label} in the last {amount} {unit_label}"),
                    category="date_window",
                )
            )

    questions.sort(key=lambda row: (row.category, row.question_id, row.question))
    return questions
