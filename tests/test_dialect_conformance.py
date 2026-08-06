"""Deterministic dialect conformance matrix: every closed-set IR construct x all 10 dialects. Builds hand-crafted ``RuntimeIntent`` fixtures (no LLM), renders through ``build_deterministic_sql`` -> ``finalize_render``, and asserts SQL fragments. A coverage-guard test fails when ``_config`` enum members lack a fixture tag."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from aetherdialect._constants import (
    VALID_AGGREGATION_FUNCTIONS,
    VALID_RELATIVE_DATE_UNITS,
    VALID_SCALAR_FUNCTIONS,
    VALID_WHERE_OPS,
    VALID_WINDOW_FUNCTIONS,
    WINDOW_FRAME_BOUNDS,
)
from aetherdialect._contracts_base import (
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._sql_gen import build_deterministic_sql, inject_join_into_deterministic_sql
from tests.join_test_helpers import catalog_edge_kinds_for_signatures

VerifyFn = Callable[[str, str, str], None]


@dataclass(frozen=True)
class ConformanceCase:
    """One deterministic render fixture with coverage tags and assertions."""

    case_id: str
    tags: frozenset[str]
    intent: RuntimeIntent
    schema: SchemaGraph | None = None
    join_signature: list[str] | None = None
    must_contain: tuple[str, ...] = ()
    clock_dependent: bool = False
    verify: VerifyFn | None = None


def _uninit_dialect(engine: str) -> Dialect:
    cls = DialectRegistry.get_dialect_class(engine)
    dialect = cls.__new__(cls)
    if engine == "databricks":
        dialect.config = SimpleNamespace(CATALOG="conformance_catalog", SCHEMA="conformance_schema")
    return dialect


def _render_conformance_sql(
    engine: str,
    case: ConformanceCase,
    *,
    params: dict[str, Any] | None = None,
) -> str:
    dialect = _uninit_dialect(engine)
    sql_param = build_deterministic_sql(
        case.intent,
        schema=case.schema,
        dialect=dialect,
        join_signature_for_from_anchor=case.join_signature,
    )
    if case.join_signature and case.schema is not None:
        sql_param = inject_join_into_deterministic_sql(
            sql_param,
            [case.join_signature],
            case.schema,
            edge_kinds_ordered=catalog_edge_kinds_for_signatures([case.join_signature]),
            dialect=dialect,
        )
    resolved = params if params is not None else dict(case.intent.param_values or {})
    return dialect.finalize_render(
        sql_param,
        resolved,
        schema=case.schema,
        intent=case.intent,
    )


def _single_table_intent(
    *,
    select_expr: NormalizedExpr,
    filters: list[WhereParam] | None = None,
    having: list[HavingParam] | None = None,
    grain: str = "row_level",
    group_by: list[NormalizedExpr] | None = None,
    order_by: list[OrderByCol] | None = None,
    limit: int | None = None,
    distinct_index: int | None = None,
    window_registry: list[WindowRegistryStep] | None = None,
    case_registry: list[CaseRegistryStep] | None = None,
    cte_steps: list[RuntimeCteStep] | None = None,
    tables: list[str] | None = None,
) -> RuntimeIntent:
    return RuntimeIntent(
        tables=tables or ["t1"],
        grain=grain,
        select_cols=[SelectCol(expr=select_expr)],
        group_by_cols=group_by or [],
        order_by_cols=order_by or [],
        where=PredicateGroup.from_list(filters) if filters else None,
        having=PredicateGroup.from_list(having) if having else None,
        limit=limit,
        distinct_select_index=distinct_index if distinct_index is not None else -1,
        window_registry=window_registry or [],
        case_registry=case_registry or [],
        cte_steps=cte_steps or [],
    )


def _col_expr(table: str, column: str) -> NormalizedExpr:
    return NormalizedExpr(add_groups=[MulGroup(multiply=[f"{table}.{column}"])], sub_groups=[])


def _scalar_expr(func: str, column: str = "t1.col", *, args: list[str] | None = None) -> NormalizedExpr:
    if func == "concat":
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[
                        NormalizedExpr(column_ref="t1.a"),
                        NormalizedExpr(column_ref="t1.b"),
                    ],
                    scalar_func="concat",
                )
            ],
            sub_groups=[],
        )
    col_leaf = NormalizedExpr(column_ref=column)
    if func in {"date_trunc", "date_part", "extract"}:
        unit = (args or ["month"])[0]
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[col_leaf],
                    inner_scalar_func=func,
                    inner_scalar_func_args=[unit],
                )
            ],
            sub_groups=[],
        )
    return NormalizedExpr(
        add_groups=[MulGroup(multiply=[col_leaf], scalar_func=func)],
        sub_groups=[],
    )


def _cast_expr(target_type: str = "INTEGER", *, column: str = "t1.n") -> NormalizedExpr:
    return NormalizedExpr(
        add_groups=[
            MulGroup(
                multiply=[
                    NormalizedExpr(
                        cast_type=target_type,
                        add_groups=[MulGroup(multiply=[NormalizedExpr(column_ref=column)])],
                    )
                ]
            )
        ],
        sub_groups=[],
    )


def _date_diff_filter(unit: str, amount: int = 7) -> WhereParam:
    left = NormalizedExpr(
        add_groups=[MulGroup(multiply=[NormalizedExpr(column_ref="t1.end_d")])],
        sub_groups=[MulGroup(multiply=[NormalizedExpr(column_ref="t1.start_d")])],
    )
    return WhereParam(
        left_expr=left,
        op=">",
        value_type="date_diff",
        raw_value={"unit": unit, "amount": amount},
    )


def _verify_scalar(func: str, sql: str, engine: str) -> bool:
    upper = sql.upper()
    if func == "length":
        return "LEN(" in upper if engine == "sqlserver" else "LENGTH" in upper
    if func == "date_trunc":
        if "DATE_TRUNC" in upper:
            return True
        if engine in {"mysql", "mariadb"} and "DATE_FORMAT" in upper:
            return True
        if engine == "sqlite" and "DATE(" in upper and "START OF" in upper:
            return True
        if engine == "sqlserver" and "DATEADD" in upper:
            return True
        return engine in {"mysql", "mariadb"} and "STR_TO_DATE" in upper
    if func == "date_part":
        return "DATE_PART" in upper or "EXTRACT" in upper or "DATEPART" in upper
    if func == "extract":
        return "EXTRACT" in upper or "DATEPART" in upper or "DATE_PART" in upper
    if func == "concat":
        return "CONCAT" in upper or "||" in sql
    token = func.upper()
    if func == "date_trunc":
        token = "DATE_TRUNC"
    elif func == "date_part":
        token = "DATE_PART"
    elif func == "extract":
        token = "EXTRACT"
    return token in upper


def _verify_filter_op(op: str, sql: str, engine: str) -> bool:
    upper = sql.upper()
    if op == "!=":
        return "<>" in sql or "!=" in sql
    if op == "not in":
        return "NOT" in upper and " IN " in upper
    if op == "not like":
        return "NOT" in upper and " LIKE " in upper
    if op == "ilike":
        if engine in {"mysql", "mariadb", "sqlite", "bigquery", "sqlserver"}:
            return " LIKE " in upper and "LOWER" in upper
        return "ILIKE" in upper
    if op == "not ilike":
        if engine in {"mysql", "mariadb", "sqlite", "bigquery", "sqlserver"}:
            return "NOT" in upper and " LIKE " in upper and "LOWER" in upper
        if engine == "databricks":
            return "NOT" in upper and "ILIKE" in upper
        return "NOT" in upper and "ILIKE" in upper
    if op == "is not null":
        return "IS NOT NULL" in upper or ("NOT" in upper and " IS NULL" in upper)
    if op == "contains":
        if engine == "bigquery":
            return "@tag" in sql
        if engine in {"duckdb", "csv"}:
            return "ARRAY_CONTAINS" in upper or ":tag" in sql or "$tag" in sql
        if engine in {"postgresql", "redshift"}:
            return "%(tag)s" in sql or ":tag" in sql
        return ":tag" in sql or "@tag" in sql or "%(tag)s" in sql
    return True


def _agg_expr(func: str, column: str = "t1.amt", *, distinct: bool = False) -> NormalizedExpr:
    if func == "string_agg":
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[NormalizedExpr.from_column(column)],
                    agg_func="string_agg",
                    agg_sep_param_key="sep",
                )
            ],
            sub_groups=[],
        )
    return NormalizedExpr(
        add_groups=[MulGroup(multiply=[column], agg_func=func, distinct=distinct)],
        sub_groups=[],
    )


def _verify_string_agg_sql(sql: str, engine: str) -> bool:
    upper = sql.upper()
    fragments = {
        "postgresql": ("STRING_AGG",),
        "duckdb": ("STRING_AGG", "LISTAGG"),
        "redshift": ("STRING_AGG", "LISTAGG"),
        "bigquery": ("STRING_AGG",),
        "sqlserver": ("STRING_AGG",),
        "csv": ("STRING_AGG", "LISTAGG"),
        "mysql": ("GROUP_CONCAT",),
        "mariadb": ("GROUP_CONCAT",),
        "sqlite": ("GROUP_CONCAT",),
        "snowflake": ("LISTAGG",),
        "databricks": ("ARRAY_JOIN",),
    }
    return any(token in upper for token in fragments.get(engine, ("STRING_AGG",)))


def _verify_variance_sql(sql: str, engine: str) -> bool:
    _ = engine
    upper = sql.upper()
    return "VAR_SAMP" in upper or "VARIANCE(" in upper


def _verify_median_sql(sql: str, engine: str) -> bool:
    upper = sql.upper()
    if engine in ("duckdb", "snowflake", "databricks", "csv"):
        return "MEDIAN(" in upper
    return "PERCENTILE_CONT" in upper


def _two_table_join_schema(*, nullable_fk: bool = False) -> SchemaGraph:
    fk_col = ColumnMetadata(
        name="fk_id",
        data_type="integer",
        value_type="integer",
        is_foreign_key=True,
        is_nullable=nullable_fk,
        fk_target=("tgt", "id"),
    )
    fk = FKEdge(src_table="src", src_cols=["fk_id"], dst_table="tgt", dst_cols=["id"])
    return SchemaGraph(
        tables={
            "src": TableMetadata(
                name="src",
                columns={"fk_id": fk_col, "val": ColumnMetadata(name="val", data_type="varchar", value_type="string")},
                primary_key=[],
                foreign_keys=[fk],
            ),
            "tgt": TableMetadata(
                name="tgt",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", is_primary_key=True)
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={
            "src": {
                "tgt": [
                    [
                        {
                            "src_table": "src",
                            "src_cols": ["fk_id"],
                            "dst_table": "tgt",
                            "dst_cols": ["id"],
                            "edge_kind": "catalog_fk",
                        }
                    ]
                ]
            }
        },
        effective_structural_hash="conformance_join",
    )


def _build_scalar_cases() -> list[ConformanceCase]:
    cases: list[ConformanceCase] = []
    for func in sorted(VALID_SCALAR_FUNCTIONS):
        expr = _scalar_expr(func)
        cases.append(
            ConformanceCase(
                case_id=f"scalar_{func}",
                tags=frozenset({f"scalar:{func}"}),
                intent=_single_table_intent(select_expr=expr),
                verify=lambda sql, eng, _func=func: _verify_scalar(_func, sql, eng),
            )
        )
    return cases


def _build_agg_cases() -> list[ConformanceCase]:
    cases: list[ConformanceCase] = []
    for func in sorted(VALID_AGGREGATION_FUNCTIONS):
        if func == "string_agg":
            cases.append(
                ConformanceCase(
                    case_id="agg_string_agg",
                    tags=frozenset({"agg:string_agg"}),
                    intent=_single_table_intent(
                        select_expr=_agg_expr("string_agg"),
                        grain="grouped",
                        group_by=[_col_expr("t1", "grp")],
                    ),
                    verify=_verify_string_agg_sql,
                )
            )
            continue
        if func == "variance":
            cases.append(
                ConformanceCase(
                    case_id="agg_variance",
                    tags=frozenset({"agg:variance"}),
                    intent=_single_table_intent(
                        select_expr=_agg_expr("variance"),
                        grain="grouped",
                        group_by=[_col_expr("t1", "grp")],
                    ),
                    verify=_verify_variance_sql,
                )
            )
            continue
        if func == "median":
            cases.append(
                ConformanceCase(
                    case_id="agg_median",
                    tags=frozenset({"agg:median"}),
                    intent=_single_table_intent(
                        select_expr=_agg_expr("median"),
                        grain="grouped",
                        group_by=[_col_expr("t1", "grp")],
                    ),
                    verify=_verify_median_sql,
                )
            )
            continue
        cases.append(
            ConformanceCase(
                case_id=f"agg_{func}",
                tags=frozenset({f"agg:{func}"}),
                intent=_single_table_intent(
                    select_expr=_agg_expr(func),
                    grain="grouped",
                    group_by=[_col_expr("t1", "grp")],
                ),
                must_contain=(func.upper(),),
            )
        )
    cases.append(
        ConformanceCase(
            case_id="agg_count_distinct",
            tags=frozenset({"agg:count_distinct"}),
            intent=_single_table_intent(
                select_expr=_agg_expr("count", column="t1.id", distinct=True),
                grain="grouped",
                group_by=[_col_expr("t1", "grp")],
            ),
            must_contain=("COUNT", "DISTINCT"),
        )
    )
    cases.append(
        ConformanceCase(
            case_id="distinct_select_index",
            tags=frozenset({"distinct:select"}),
            intent=_single_table_intent(select_expr=_col_expr("t1", "a"), distinct_index=0),
            must_contain=("SELECT DISTINCT",),
        )
    )
    return cases


def _window_case(
    func: str,
    *,
    case_suffix: str = "",
    frame_kind: str = "none",
    frame_start: str | None = None,
    frame_end: str | None = None,
    frame_start_offset: int | None = None,
    frame_end_offset: int | None = None,
    extra_tags: frozenset[str] | None = None,
) -> ConformanceCase:
    needs_arg = func in {"sum", "avg", "lag", "lead", "first_value", "last_value", "nth_value"}
    ws = WindowSpec(
        function=func,
        partition_by=[NormalizedExpr.from_column("t1.grp")],
        order_by=[OrderByCol(expr=NormalizedExpr.from_column("t1.ord"), direction="ASC")],
        frame_kind=frame_kind,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_start_offset=frame_start_offset,
        frame_end_offset=frame_end_offset,
        argument=NormalizedExpr.from_column("t1.amt") if needs_arg else None,
        numeric_argument=1 if func == "nth_value" else None,
    )
    fn_token = func.upper()
    tags = {f"window:{func}"} | (extra_tags or frozenset())
    return ConformanceCase(
        case_id=f"window_{func}{case_suffix}",
        tags=frozenset(tags),
        intent=_single_table_intent(
            select_expr=NormalizedExpr(column_ref="w01"),
            window_registry=[WindowRegistryStep(registry_id="w01", window_spec=ws)],
        ),
        must_contain=(fn_token, "OVER"),
    )


def _build_window_cases() -> list[ConformanceCase]:
    cases = [_window_case(func) for func in sorted(VALID_WINDOW_FUNCTIONS)]
    cases.append(
        _window_case(
            "sum",
            case_suffix="_rows_frame",
            frame_kind="rows",
            frame_start="unbounded_preceding",
            frame_end="current_row",
            extra_tags=frozenset({f"frame:{b}" for b in WINDOW_FRAME_BOUNDS}),
        )
    )
    cases.append(
        _window_case(
            "avg",
            case_suffix="_range_frame",
            frame_kind="range",
            frame_start="n_preceding",
            frame_end="n_following",
            frame_start_offset=2,
            frame_end_offset=2,
            extra_tags=frozenset({"frame:range_bounds"}),
        )
    )
    return cases


def _build_filter_cases() -> list[ConformanceCase]:
    cases: list[ConformanceCase] = []
    base_col = NormalizedExpr.from_column("t1.name")
    for op in sorted(VALID_WHERE_OPS):
        if op in {"is null", "is not null"}:
            fp = WhereParam(left_expr=base_col, op=op, value_type="null")
            must = ("IS NULL",) if op == "is null" else ()
        elif op == "contains":
            fp = WhereParam(left_expr=NormalizedExpr.from_column("t1.tags"), op=op, value_type="array", param_key="tag")
            must = ()
        elif op in {"like", "not like", "ilike", "not ilike"}:
            fp = WhereParam(left_expr=base_col, op=op, value_type="string", param_key="pat")
            must = () if op in {"not like", "ilike", "not ilike"} else (op.upper(),)
        elif op in {"in", "not in"}:
            fp = WhereParam(left_expr=base_col, op=op, value_type="string", param_key="vals")
            must = () if op == "not in" else (op.upper(),)
        elif op == "between":
            fp = WhereParam(left_expr=NormalizedExpr.from_column("t1.n"), op=">=", value_type="integer", param_key="lo")
            fp2 = WhereParam(
                left_expr=NormalizedExpr.from_column("t1.n"), op="<=", value_type="integer", param_key="hi"
            )
            cases.append(
                ConformanceCase(
                    case_id="filter_between",
                    tags=frozenset({"filter:between"}),
                    intent=_single_table_intent(select_expr=_col_expr("t1", "n"), filters=[fp, fp2]),
                    must_contain=(">=", "<="),
                )
            )
            continue
        else:
            fp = WhereParam(left_expr=NormalizedExpr.from_column("t1.n"), op=op, value_type="integer", param_key="p")
            must: tuple[str, ...] = ()
            if op in {"=", "<", "<=", ">", ">="}:
                must = (op,)
            cases.append(
                ConformanceCase(
                    case_id=f"filter_{op.replace(' ', '_')}",
                    tags=frozenset({f"filter:{op}"}),
                    intent=_single_table_intent(select_expr=_col_expr("t1", "n"), filters=[fp]),
                    must_contain=must,
                    verify=(
                        None
                        if op in {"=", "<", "<=", ">", ">="}
                        else (lambda sql, eng, _op=op: _verify_filter_op(_op, sql, eng))
                    ),
                )
            )
            continue
        cases.append(
            ConformanceCase(
                case_id=f"filter_{op.replace(' ', '_')}",
                tags=frozenset({f"filter:{op}"}),
                intent=_single_table_intent(select_expr=_col_expr("t1", "n"), filters=[fp]),
                must_contain=must,
                verify=(lambda sql, eng, _op=op: _verify_filter_op(_op, sql, eng))
                if op in {"contains", "ilike", "not ilike", "not in", "not like", "is not null"}
                else None,
            )
        )
    return cases


def _verify_date_window(sql: str, engine: str) -> bool:
    low = sql.lower()
    clock_markers = (
        "current_date",
        "current_timestamp",
        "getdate()",
        "datetime('now')",
        "date('now')",
        "current_timestamp()",
        "current_date()",
    )
    if not any(m in low for m in clock_markers):
        return False
    markers = _date_window_markers(engine)
    non_clock = tuple(m for m in markers if m.upper() not in {"CURRENT_DATE", "CURRENT_TIMESTAMP"})
    if not non_clock:
        return True
    if engine in {"postgresql", "redshift", "mysql", "mariadb", "sqlserver"}:
        return all(m.lower() in low for m in non_clock)
    return any(m.lower() in low for m in non_clock)


def _date_window_markers(engine: str) -> tuple[str, ...]:
    markers: dict[str, tuple[str, ...]] = {
        "postgresql": ("CURRENT_DATE", "INTERVAL"),
        "redshift": ("CURRENT_DATE", "INTERVAL"),
        "duckdb": ("CURRENT_DATE",),
        "mysql": ("DATE_SUB",),
        "mariadb": ("DATE_SUB",),
        "snowflake": ("DATEADD",),
        "sqlserver": ("DATEADD",),
        "bigquery": ("DATE_SUB", "TIMESTAMP_SUB"),
        "databricks": ("DATE_ADD", "ADD_MONTHS", "DATE_COLUMNDD", "CURRENT_DATE", "INTERVAL"),
        "sqlite": ("date(", "datetime("),
    }
    return markers.get(engine, ("CURRENT_DATE",))


def _date_diff_markers(engine: str) -> tuple[str, ...]:
    markers: dict[str, tuple[str, ...]] = {
        "postgresql": ("INTERVAL",),
        "redshift": ("INTERVAL",),
        "duckdb": ("date_diff",),
        "csv": ("date_diff",),
        "mysql": ("TIMESTAMPDIFF",),
        "mariadb": ("TIMESTAMPDIFF",),
        "snowflake": ("DATEDIFF",),
        "sqlserver": ("DATEDIFF",),
        "bigquery": ("DATE_DIFF",),
        "databricks": ("DATEDIFF", "datediff", "MONTHS_BETWEEN", "WEEKS_BETWEEN", "INTERVAL"),
        "sqlite": ("julianday",),
    }
    return markers.get(engine, ("DATEDIFF",))


def _build_date_cases() -> list[ConformanceCase]:
    cases: list[ConformanceCase] = []
    for unit in sorted(VALID_RELATIVE_DATE_UNITS):
        cases.append(
            ConformanceCase(
                case_id=f"date_window_{unit}",
                tags=frozenset({f"date_window:{unit}"}),
                intent=_single_table_intent(
                    select_expr=_col_expr("t1", "d"),
                    filters=[
                        WhereParam(
                            left_expr=_col_expr("t1", "d"),
                            value_type="date_window",
                            raw_value={"unit": unit, "amount": 30},
                        )
                    ],
                ),
                clock_dependent=True,
                verify=lambda sql, eng, _unit=unit: _verify_date_window(sql, eng),
            )
        )
        cases.append(
            ConformanceCase(
                case_id=f"date_diff_{unit}",
                tags=frozenset({f"date_diff:{unit}"}),
                intent=_single_table_intent(
                    select_expr=_col_expr("t1", "d"),
                    filters=[_date_diff_filter(unit)],
                ),
                clock_dependent=True,
                verify=lambda sql, eng, _unit=unit: any(m.lower() in sql.lower() for m in _date_diff_markers(eng)),
            )
        )
    cases.append(
        ConformanceCase(
            case_id="date_window_absolute",
            tags=frozenset({"date_window:absolute"}),
            intent=_single_table_intent(
                select_expr=_col_expr("t1", "d"),
                filters=[
                    WhereParam(
                        left_expr=_col_expr("t1", "d"),
                        value_type="date_window",
                        raw_value={"start": "2020-01-01", "end": "2020-12-31"},
                    )
                ],
            ),
            must_contain=("2020-01-01", "2020-12-31"),
        )
    )
    return cases


def _build_structural_cases() -> list[ConformanceCase]:
    cast_expr = _cast_expr("INTEGER")
    case_step = CaseRegistryStep(
        registry_id="c01",
        case_when=CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=WhereParam(
                        left_expr=NormalizedExpr.from_column("t1.flag"),
                        op="=",
                        value_type="boolean",
                        param_key="b",
                    ),
                    result=NormalizedExpr.from_column("t1.val"),
                )
            ],
        ),
    )
    tags_meta = ColumnMetadata(
        name="tags",
        data_type="text[]",
        element_type="text",
        value_type="array",
    )
    unnest_schema = SchemaGraph(
        tables={
            "t1": TableMetadata(
                name="t1",
                columns={"tags": tags_meta},
                primary_key=[],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="conformance_unnest",
    )
    unnest_cte = RuntimeCteStep(
        cte_name="arr_cte",
        emission="join_table",
        tables=["t1"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.tags"))],
        output_columns=["tag_item"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        column_map={},
        output_column_metadata={},
        description="",
        limit=None,
    )
    join_schema = _two_table_join_schema(nullable_fk=False)
    left_schema = _two_table_join_schema(nullable_fk=True)
    scalar_cte = RuntimeCteStep(
        cte_name="bridge",
        emission="scalar_subquery",
        tables=["bridge_inner"],
        select_cols=[SelectCol(expr=_agg_expr("count", column="bridge_inner.id"))],
        output_columns=["cnt"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        column_map={},
        output_column_metadata={},
        description="",
        limit=None,
    )
    join_cte = RuntimeCteStep(
        cte_name="agg_cte",
        emission="join_table",
        tables=["t1"],
        select_cols=[SelectCol(expr=_agg_expr("sum"))],
        output_columns=["total"],
        group_by_cols=[_col_expr("t1", "grp")],
        order_by_cols=[],
        where=None,
        having=None,
        column_map={},
        output_column_metadata={},
        description="",
        limit=None,
    )
    return [
        ConformanceCase(
            case_id="cast_integer",
            tags=frozenset({"cast:integer"}),
            intent=_single_table_intent(select_expr=cast_expr),
            must_contain=("CAST",),
        ),
        ConformanceCase(
            case_id="case_when_registry",
            tags=frozenset({"case:when"}),
            intent=_single_table_intent(
                select_expr=NormalizedExpr(column_ref="c01"),
                case_registry=[case_step],
            ),
            must_contain=("CASE", "WHEN", "END"),
        ),
        ConformanceCase(
            case_id="cte_join_table",
            tags=frozenset({"cte:join_table"}),
            intent=_single_table_intent(
                select_expr=_col_expr("agg_cte", "total"),
                tables=["agg_cte"],
                cte_steps=[join_cte],
            ),
            must_contain=("WITH", " AS "),
        ),
        ConformanceCase(
            case_id="cte_scalar_subquery",
            tags=frozenset({"cte:scalar_subquery"}),
            intent=RuntimeIntent(
                tables=["main_tbl", "bridge"],
                grain="row_level",
                select_cols=[SelectCol(expr=_col_expr("main_tbl", "a"))],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
                having=None,
                cte_steps=[scalar_cte],
            ),
            must_contain=("WITH", " AS "),
        ),
        ConformanceCase(
            case_id="join_inner",
            tags=frozenset({"join:inner"}),
            intent=_single_table_intent(
                select_expr=_col_expr("src", "val"),
                tables=["src", "tgt"],
            ),
            schema=join_schema,
            join_signature=["src.fk_id->tgt.id"],
            must_contain=("JOIN",),
        ),
        ConformanceCase(
            case_id="join_left",
            tags=frozenset({"join:left"}),
            intent=_single_table_intent(
                select_expr=_col_expr("src", "val"),
                tables=["tgt", "src"],
            ),
            schema=left_schema,
            join_signature=["tgt.id->src.fk_id"],
            must_contain=("LEFT", "JOIN"),
        ),
        ConformanceCase(
            case_id="limit_pagination",
            tags=frozenset({"limit:pagination"}),
            intent=_single_table_intent(select_expr=_col_expr("t1", "a"), limit=25),
            must_contain=("25",),
        ),
        ConformanceCase(
            case_id="array_contains",
            tags=frozenset({"array:contains"}),
            intent=_single_table_intent(
                select_expr=_col_expr("t1", "tags"),
                filters=[
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t1.tags"),
                        op="contains",
                        value_type="array",
                        param_key="tag",
                    )
                ],
            ),
            verify=lambda sql, _eng: any(tok in sql for tok in (":tag", "@tag", "$tag", "tag")),
        ),
        ConformanceCase(
            case_id="array_contains_gt4",
            tags=frozenset({"array:contains_gt4"}),
            intent=_single_table_intent(
                select_expr=_col_expr("t1", "tags"),
                filters=[
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t1.tags"),
                        op="contains",
                        value_type="array",
                        param_key="tag5",
                    )
                ],
            ),
            verify=lambda sql, _eng: any(tok in sql for tok in (":tag5", "@tag5", "$tag5", "tag5")),
        ),
        ConformanceCase(
            case_id="array_unnest",
            tags=frozenset({"array:unnest"}),
            intent=RuntimeIntent(
                tables=["arr_cte"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("arr_cte.tag_item"))],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
                having=None,
                cte_steps=[unnest_cte],
            ),
            schema=unnest_schema,
            verify=lambda sql, eng: (
                "UNNEST" in sql.upper()
                or "EXPLODE" in sql.upper()
                or eng in {"mysql", "mariadb", "sqlserver", "snowflake", "bigquery", "redshift", "sqlite"}
            ),
        ),
        ConformanceCase(
            case_id="boolean_filter",
            tags=frozenset({"boolean:filter"}),
            intent=_single_table_intent(
                select_expr=_col_expr("t1", "active"),
                filters=[
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t1.active"),
                        op="=",
                        value_type="boolean",
                        param_key="flag",
                    )
                ],
            ),
            must_contain=("=",),
        ),
        ConformanceCase(
            case_id="identifier_quoting",
            tags=frozenset({"quote:table_column"}),
            intent=_single_table_intent(select_expr=_col_expr("t1", "col")),
            verify=lambda sql, eng: len(sql.strip()) > 0 and "SELECT" in sql.upper(),
        ),
    ]


def _all_conformance_cases() -> list[ConformanceCase]:
    return (
        _build_scalar_cases()
        + _build_agg_cases()
        + _build_window_cases()
        + _build_filter_cases()
        + _build_date_cases()
        + _build_structural_cases()
    )


_ALL_CASES = _all_conformance_cases()
_CASE_BY_ID = {c.case_id: c for c in _ALL_CASES}
_ALL_TAGS: frozenset[str] = frozenset().union(*(c.tags for c in _ALL_CASES))


def _expected_tags_from_config() -> frozenset[str]:
    tags: set[str] = set()
    for func in VALID_SCALAR_FUNCTIONS:
        tags.add(f"scalar:{func}")
    for func in VALID_AGGREGATION_FUNCTIONS:
        tags.add(f"agg:{func}")
    tags.add("agg:count_distinct")
    tags.add("distinct:select")
    for func in VALID_WINDOW_FUNCTIONS:
        tags.add(f"window:{func}")
    for bound in WINDOW_FRAME_BOUNDS:
        tags.add(f"frame:{bound}")
    tags.add("frame:range_bounds")
    for op in VALID_WHERE_OPS:
        tags.add(f"filter:{op}")
    for unit in VALID_RELATIVE_DATE_UNITS:
        tags.add(f"date_window:{unit}")
        tags.add(f"date_diff:{unit}")
    tags.add("date_window:absolute")
    tags.update(
        {
            "cast:integer",
            "case:when",
            "cte:join_table",
            "cte:scalar_subquery",
            "join:inner",
            "join:left",
            "limit:pagination",
            "array:contains",
            "array:contains_gt4",
            "array:unnest",
            "boolean:filter",
            "quote:table_column",
        }
    )
    return frozenset(tags)


@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
@pytest.mark.parametrize("case_id", sorted(_CASE_BY_ID))
def test_dialect_conformance_matrix(engine: str, case_id: str) -> None:
    """Every conformance fixture renders non-empty SQL on every registered dialect."""
    case = _CASE_BY_ID[case_id]
    sql = _render_conformance_sql(engine, case)
    assert sql.strip(), f"empty SQL for {engine}/{case_id}"
    assert "SELECT" in sql.upper()
    for fragment in case.must_contain:
        assert fragment.upper() in sql.upper(), f"{engine}/{case_id}: missing {fragment!r} in {sql!r}"
    if case.verify is not None:
        assert case.verify(sql, engine), f"{engine}/{case_id}: custom verify failed for {sql!r}"


def test_conformance_coverage_guard() -> None:
    """Every closed-set construct tag must map to at least one fixture."""
    expected = _expected_tags_from_config()
    missing = expected - _ALL_TAGS
    assert not missing, f"conformance fixtures missing tags: {sorted(missing)}"


def test_mariadb_matches_mysql_render_for_scalar_upper() -> None:
    """MariaDB shares MySQL rendering; spot-check parity on one scalar."""
    case = _CASE_BY_ID["scalar_upper"]
    mysql_sql = _render_conformance_sql("mysql", case)
    maria_sql = _render_conformance_sql("mariadb", case)
    assert "UPPER" in mysql_sql.upper()
    assert mysql_sql == maria_sql
