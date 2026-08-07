"""Sqlglot-only SQL to RuntimeIntent extraction for non-Postgres engines."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, cast

import sqlglot
from sqlglot import exp

from ._constants import (
    ALLOWED_JOIN_KINDS,
    DEFAULT_WHERE_OP_MAP,
    DISTINCT_ON_CTE_NAME_PREFIX,
    DISTINCT_ON_RANK_COLUMN,
    SELF_JOIN_CTE_NAME_PREFIX,
    SIMPLE_AGG_NAMES,
    SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM,
    SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR,
    SQL_TO_INTENT_PARAM_KEY_PREFIX,
    SQLGLOT_AGG_FUNC_KEY_ALIASES,
    WINDOW_DEFAULT_FRAME_END_WITH_ORDER,
    WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER,
    WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER,
    WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER,
    WINDOW_DEFAULT_FRAME_START_WITH_ORDER,
    WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER,
    WINDOW_IMPORT_FUNC_ALIASES,
)
from ._contracts_base import (
    ConfigError,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    OrderByNullPlacement,
    PredicateGroup,
    WhereParam,
    WindowFrameKind,
)
from ._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    SchemaGraph,
    WindowRegistryStep,
    WindowSpec,
)
from ._dialect import Dialect
from ._dialect_sqlglot_helper import SqlglotEngineDialect


def _new_sqlglot_extra() -> Any:
    """Return per-select conversion scratch space for qualifier swaps and registry emission."""

    @dataclass
    class _SqlglotExtraLocal:
        qual_swap: dict[str, str] = field(default_factory=dict)
        case_registry: list[CaseRegistryStep] = field(default_factory=list)
        window_registry: list[WindowRegistryStep] = field(default_factory=list)
        self_join_steps: list[RuntimeCteStep] = field(default_factory=list)
        case_counter: int = 0
        window_counter: int = 0

        def next_case_id(self) -> str:
            self.case_counter += 1
            return f"c{self.case_counter:02d}"

        def next_window_id(self) -> str:
            self.window_counter += 1
            return f"w{self.window_counter:02d}"

    return _SqlglotExtraLocal()


def _from_clause_root(sel: exp.Select) -> exp.Expression | None:
    from_ = sel.args.get("from_")
    if from_ is None:
        return None
    if isinstance(from_, exp.From):
        return cast(exp.Expression | None, from_.this)
    return None


def _walk_from_branches(expr: exp.Expression | None) -> Iterator[exp.Expression]:
    if expr is None:
        return
    if isinstance(expr, exp.Join):
        yield from _walk_from_branches(expr.this)
        yield from _walk_from_branches(expr.expression)
        return
    yield expr


def _tables_from_from_root(root: exp.Expression | None) -> list[str]:
    names: list[str] = []
    for node in _walk_from_branches(root):
        if isinstance(node, exp.Table) and node.name:
            names.append(node.name.strip().lower())
    return names


def _select_join_nodes(sel: exp.Select) -> list[exp.Join]:
    joins = sel.args.get("joins")
    if not joins:
        return []
    return [j for j in joins if isinstance(j, exp.Join)]


def _iter_from_leaves(sel: exp.Select) -> Iterator[exp.Expression]:
    root = _from_clause_root(sel)
    yield from _walk_from_branches(root)
    for join in _select_join_nodes(sel):
        rhs = _join_rhs_unwrapped(join)
        if rhs is not None:
            yield rhs


def _join_is_allowed(join: exp.Join) -> bool:
    kind = join.args.get("kind")
    kind_u = str(kind).upper() if kind is not None else None
    if kind_u not in ALLOWED_JOIN_KINDS:
        return False
    return not join.args.get("using")


def _unwrap_alias(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Alias):
        node = node.this
    return node


def _join_rhs_unwrapped(join: exp.Join) -> exp.Expression | None:
    raw = join.args.get("expression") or join.args.get("this")
    return _unwrap_alias(raw) if raw is not None else None


def _table_alias_name(node: exp.Expression) -> tuple[str, str] | None:
    """Return ``(alias, relation_target)`` for a FROM leaf."""
    wrapped = node
    alias: str | None = None
    if isinstance(wrapped, exp.Alias):
        alias = str(wrapped.alias_or_name or "").strip()
        wrapped = wrapped.this
    if isinstance(wrapped, exp.Table):
        rel = str(wrapped.name or "").strip()
        if not rel:
            return None
        if alias is None and wrapped.alias:
            alias = str(wrapped.alias).strip()
        if alias is None:
            alias = rel
        return alias, rel
    return None


def _normalize_window_spec_ansi_defaults(ws: WindowSpec) -> WindowSpec:
    if ws.frame_kind != "none":
        return ws
    if ws.order_by:
        return replace(
            ws,
            frame_kind=cast(WindowFrameKind, WINDOW_DEFAULT_FRAME_KIND_WITH_ORDER),
            frame_start=WINDOW_DEFAULT_FRAME_START_WITH_ORDER,
            frame_end=WINDOW_DEFAULT_FRAME_END_WITH_ORDER,
        )
    return replace(
        ws,
        frame_kind=cast(WindowFrameKind, WINDOW_DEFAULT_FRAME_KIND_WITHOUT_ORDER),
        frame_start=WINDOW_DEFAULT_FRAME_START_WITHOUT_ORDER,
        frame_end=WINDOW_DEFAULT_FRAME_END_WITHOUT_ORDER,
    )


def _dialect_preparse_sql(dialect: Any, sql: str) -> str:
    fn = getattr(dialect, "preparse_sql_for_import", None)
    if callable(fn):
        return str(fn(sql))
    return sql


def _column_identifier_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Column):
        return str(node.name or "").strip().lower()
    return ""


def _is_rank_equals_one_predicate(pred: exp.Expression) -> bool:
    if not isinstance(pred, exp.EQ):
        return False
    if _column_identifier_name(pred.this) != DISTINCT_ON_RANK_COLUMN.lower():
        return False
    right = pred.expression
    return isinstance(right, exp.Literal) and right.is_int and int(right.this) == 1


def _rewrite_distinct_on_cte_wrapper_sql(sql: str, read_dialect: str) -> str:
    """Peel deterministic ``don_*`` CTE wrappers back to DISTINCT ON for import."""
    try:
        tree = sqlglot.parse_one(sql, dialect=read_dialect)
    except Exception:
        return sql
    if not isinstance(tree, exp.Select):
        return sql
    with_clause = tree.args.get("with_")
    if with_clause is None:
        return sql
    ctes = list(with_clause.expressions or ())
    if len(ctes) != 1:
        return sql
    cte = ctes[0]
    if not isinstance(cte, exp.CTE):
        return sql
    cte_name = str(cte.alias_or_name or "").strip().lower()
    if not cte_name.startswith(DISTINCT_ON_CTE_NAME_PREFIX):
        return sql
    inner = cte.this
    if not isinstance(inner, exp.Select):
        return sql

    partition_exprs: list[exp.Expression] = []
    window_order: exp.Order | None = None
    kept_exprs: list[exp.Expression] = []
    for proj in inner.expressions or ():
        rank_alias = proj.alias_or_name.strip().lower() if isinstance(proj, exp.Alias) else ""
        if rank_alias == DISTINCT_ON_RANK_COLUMN.lower():
            win = proj.this if isinstance(proj, exp.Alias) else None
            if not isinstance(win, exp.Window):
                return sql
            inner_fn = win.this
            fn_key = WINDOW_IMPORT_FUNC_ALIASES.get(_func_name(inner_fn), _func_name(inner_fn))
            if fn_key != "row_number":
                return sql
            partition_exprs = list(win.args.get("partition_by") or ())
            window_order = win.args.get("order")
            if not partition_exprs:
                return sql
            continue
        kept_exprs.append(proj)
    if not partition_exprs:
        return sql

    where_node = tree.args.get("where")
    if where_node is not None:
        pred = where_node.this if isinstance(where_node, exp.Where) else where_node
        if not _is_rank_equals_one_predicate(pred):
            return sql

    outer_order = tree.args.get("order")
    new_sel = inner.copy()
    new_sel.set("expressions", kept_exprs)
    new_sel.set("distinct", exp.Distinct(on=exp.Tuple(expressions=partition_exprs)))
    if outer_order is not None:
        new_sel.set("order", outer_order)
    elif window_order is not None:
        new_sel.set("order", window_order)
    new_sel.set("with_", None)
    try:
        return new_sel.sql(dialect=read_dialect)
    except Exception:
        return sql


def _dialect_map_where_op(dialect: Any, op_raw: str | None) -> str | None:
    fn = getattr(dialect, "map_import_where_op", None)
    if callable(fn):
        mapped = fn(op_raw)
        if mapped is not None or op_raw is None:
            return cast(str | None, mapped)
    if op_raw is None:
        return None
    key = str(op_raw).strip().lower()
    return DEFAULT_WHERE_OP_MAP.get(key)


def _dialect_map_scalar_func(dialect: Any, fn_name: str) -> str:
    fn = getattr(dialect, "map_import_scalar_func", None)
    if callable(fn):
        mapped = fn(fn_name)
        if mapped:
            return str(mapped).strip().lower()
    return fn_name.strip().lower()


def _dialect_import_unnest_policy(dialect: Any, node: exp.Expression) -> bool:
    if _func_name(node).strip().lower() != "unnest":
        return False
    policy_fn = getattr(dialect, "import_unnest_policy", None)
    if not callable(policy_fn):
        return False
    policy = policy_fn()
    return policy in ("unsupported", "from_only")


def _make_lit_key_factory(existing: Callable[[], str] | None = None) -> tuple[dict[str, Any], Callable[[], str]]:
    store: dict[str, Any] = {}
    counter = [0]

    def next_lit_key() -> str:
        if existing is not None:
            return existing()
        counter[0] += 1
        return f"{SQL_TO_INTENT_PARAM_KEY_PREFIX}{counter[0]}"

    return store, next_lit_key


def _const_int_only(node: exp.Expression) -> int | None:
    node = _unwrap_alias(node)
    if not isinstance(node, exp.Literal) or not node.is_int:
        return None
    try:
        return int(str(node.this))
    except (TypeError, ValueError):
        return None


def _const_payload(node: exp.Expression) -> tuple[Any, str, str] | None:
    node = _unwrap_alias(node)
    if isinstance(node, exp.Boolean):
        return bool(node.this), "boolean", SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR
    if isinstance(node, exp.Literal):
        if node.is_int:
            return int(str(node.this)), "integer", SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM
        if node.is_number:
            return float(str(node.this)), "number", SQL_TO_INTENT_LITERAL_PLACEHOLDER_NUM
        return str(node.this), "string", SQL_TO_INTENT_LITERAL_PLACEHOLDER_STR
    if isinstance(node, exp.Cast) and node.this is not None:
        return _const_payload(node.this)
    return None


def _mask_literal(
    param_store: dict[str, Any], next_lit_key: Callable[[], str], payload: tuple[Any, str, str]
) -> NormalizedExpr:
    raw_v, _vt, ph = payload
    key = next_lit_key()
    param_store[key] = raw_v
    return NormalizedExpr(string_literal=ph)


def _where_literal_payload(node: exp.Expression) -> tuple[Any, str] | None:
    pay = _const_payload(node)
    return (pay[0], pay[1]) if pay else None


def _qual_column_name(
    col: exp.Column, alias_map: dict[str, str], single_alias: str | None, extra: Any | None = None
) -> str | None:
    pre = str(col.table).strip() if col.table else None
    col_name = str(col.name or "").strip()
    if not col_name:
        return None
    if pre is not None and extra and pre in extra.qual_swap:
        pre = extra.qual_swap[pre]
    if pre is None:
        if single_alias:
            return f"{single_alias}.{col_name}"
        return None
    if pre in alias_map:
        return f"{pre}.{col_name}"
    return None


def _func_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this or "").strip().lower()
    if isinstance(node, exp.Func):
        sql_name_fn = getattr(node, "sql_name", None)
        name = node.key or (sql_name_fn() if callable(sql_name_fn) else "")
        return str(name or "").strip().lower()
    mapped = SqlglotEngineDialect.AGG_NODE_TO_NAME.get(type(node))
    if mapped:
        return mapped
    return str(getattr(node, "key", "") or "").strip().lower()


def _collect_product_factors(expr: NormalizedExpr) -> list[NormalizedExpr]:
    if (
        expr.add_groups
        and len(expr.add_groups) == 1
        and not expr.sub_groups
        and not expr.add_values
        and not expr.sub_values
        and not expr.string_literal
        and not expr.scalar_func
    ):
        g = expr.add_groups[0]
        if (
            not g.divide
            and not g.agg_func
            and not g.scalar_func
            and float(g.coefficient) == 1.0
            and not g.coeff_param_key
            and len(g.multiply) > 1
        ):
            return list(g.multiply)
    return [expr]


def _merge_product_exprs(left: NormalizedExpr, right: NormalizedExpr) -> NormalizedExpr:
    return NormalizedExpr(
        add_groups=[MulGroup(multiply=_collect_product_factors(left) + _collect_product_factors(right))]
    )


def _merge_ratio_exprs(left: NormalizedExpr, right: NormalizedExpr) -> NormalizedExpr:
    return NormalizedExpr(
        add_groups=[MulGroup(multiply=_collect_product_factors(left), divide=_collect_product_factors(right))]
    )


def _wrap_mul_term(expr: NormalizedExpr) -> MulGroup:
    return MulGroup(multiply=[expr])


def _expr_sql_token(expr: NormalizedExpr) -> str:
    if expr.column_ref:
        return str(expr.column_ref)
    if expr.string_literal:
        return str(expr.string_literal)
    if expr.keyword:
        return str(expr.keyword)
    return ""


def _interval_from_literal(raw: str) -> tuple[float, str] | None:
    t = (raw or "").strip()
    if not t:
        return None
    parts = t.split()
    if len(parts) >= 2:
        try:
            return (float(parts[0]), " ".join(parts[1:]).lower())
        except (TypeError, ValueError):
            return None
    if len(parts) == 1:
        try:
            return (float(parts[0]), "day")
        except (TypeError, ValueError):
            return None
    return None


def _over_node(node: exp.Expression) -> exp.Expression | None:
    over = node.args.get("over")
    if over is not None:
        return cast(exp.Expression | None, over)
    inner = getattr(node, "over", None)
    return cast(exp.Expression | None, inner)


def _window_partition_exprs(
    nodes: list[exp.Expression] | None,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None,
) -> list[NormalizedExpr] | None:
    out: list[NormalizedExpr] = []
    for n in nodes or ():
        ex = _expr_full(
            n,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if ex is None:
            return None
        out.append(ex)
    return out


def _window_sort_clause(
    order_node: exp.Expression | None,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    select_cols: list[SelectCol],
    extra: Any | None,
) -> list[OrderByCol] | None:
    if order_node is None:
        return []
    items: list[exp.Expression]
    if isinstance(order_node, exp.Order):
        items = list(order_node.expressions or [])
    else:
        items = [order_node]
    out: list[OrderByCol] = []
    for item in items:
        direction = "ASC"
        nulls: OrderByNullPlacement | None = None
        node = item
        if isinstance(item, exp.Ordered):
            node = item.this
            desc = item.args.get("desc")
            direction = "DESC" if desc else "ASC"
            item_sql = item.sql(dialect=None).upper()
            if "NULLS FIRST" in item_sql:
                nulls = OrderByNullPlacement.FIRST
            elif "NULLS LAST" in item_sql:
                nulls = OrderByNullPlacement.LAST
        ex: NormalizedExpr | None
        if isinstance(node, exp.Literal) and node.is_int:
            ord_i = _const_int_only(node)
            if ord_i is None or ord_i < 1 or ord_i > len(select_cols):
                return None
            ex = select_cols[ord_i - 1].expr
        else:
            ex = _expr_full(
                node,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
        if ex is None:
            return None
        out.append(OrderByCol(expr=ex, direction=direction, nulls=nulls))
    return out


def _window_frame_from_spec(
    spec: exp.Expression | None,
) -> tuple[WindowFrameKind, str | None, str | None, int | None, int | None] | None:
    if spec is None:
        return (WindowFrameKind.NONE, None, None, None, None)
    if isinstance(spec, exp.WindowSpec):
        kind_raw = str(spec.args.get("kind") or spec.args.get("type") or "").upper()
        frame_kind: WindowFrameKind = WindowFrameKind.NONE
        if kind_raw in ("ROWS", "ROW"):
            frame_kind = WindowFrameKind.ROWS
        elif kind_raw in ("RANGE"):
            frame_kind = WindowFrameKind.RANGE
        start = spec.args.get("start")
        end = spec.args.get("end")
        frame_start: str | None = None
        frame_end: str | None = None
        frame_start_offset: int | None = None
        if start is not None:
            frame_start = start.sql(dialect=None).upper()
            off = _const_int_only(start) if isinstance(start, exp.Expression) else None
            if off is not None:
                frame_start_offset = off
        if end is not None:
            frame_end = end.sql(dialect=None).upper()
        if frame_kind == "none" and frame_start is None and frame_end is None:
            return (WindowFrameKind.NONE, None, None, None, None)
        return (frame_kind, frame_start, frame_end, frame_start_offset, None)
    return (WindowFrameKind.NONE, None, None, None, None)


def _window_def_to_spec(
    over: exp.Expression,
    fn_node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    select_cols: list[SelectCol],
    extra: Any | None,
) -> WindowSpec | None:
    partition_by: list[exp.Expression] = []
    order_node: exp.Expression | None = None
    spec_node: exp.Expression | None = None
    over_cls = getattr(exp, "Over", None)
    if isinstance(over, exp.Window):
        partition_by = list(over.args.get("partition_by") or [])
        order_node = over.args.get("order")
        spec_node = over.args.get("spec")
    elif over_cls is not None and isinstance(over, over_cls):
        partition_by = list(over.args.get("partition_by") or [])
        order_node = over.args.get("order")
        spec_node = over.args.get("spec")
    else:
        return None
    fn_name = _dialect_map_scalar_func(dialect, _func_name(fn_node))
    fn_name = WINDOW_IMPORT_FUNC_ALIASES.get(fn_name, fn_name)
    if not fn_name:
        return None
    part = _window_partition_exprs(partition_by, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
    if part is None:
        return None
    order = _window_sort_clause(
        order_node, dialect, alias_map, single_alias, param_store, next_lit_key, select_cols, extra
    )
    if order is None:
        return None
    arg_expr: NormalizedExpr | None = None
    numeric_argument: int | None = None
    if fn_name == "ntile":
        numeric_argument = _const_int_only(fn_node.this) if fn_node.this is not None else None
        if numeric_argument is None:
            return None
    elif fn_name == "nth_value":
        arg_source = fn_node.this if hasattr(fn_node, "this") else None
        if arg_source is not None and not isinstance(arg_source, (exp.Star, exp.Distinct)):
            arg_expr = _expr_full(
                arg_source,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if arg_expr is None:
                return None
        offset_node = fn_node.args.get("offset") if hasattr(fn_node, "args") else None
        numeric_argument = _const_int_only(offset_node) if offset_node is not None else None
        if numeric_argument is None:
            return None
    else:
        arg_source = fn_node.this if hasattr(fn_node, "this") else None
        if arg_source is not None and not isinstance(arg_source, (exp.Star, exp.Distinct)):
            arg_expr = _expr_full(
                arg_source,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if arg_expr is None:
                return None
    frame = _window_frame_from_spec(spec_node)
    if frame is None:
        return None
    frame_kind, frame_start, frame_end, frame_start_offset, frame_end_offset = frame
    ws = WindowSpec(
        function=fn_name,
        partition_by=part,
        order_by=order,
        argument=arg_expr,
        numeric_argument=numeric_argument,
        frame_kind=frame_kind,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_start_offset=frame_start_offset,
        frame_end_offset=frame_end_offset,
    )
    return _normalize_window_spec_ansi_defaults(ws)


def _func_with_over_to_registry_step(
    fn_node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    select_cols: list[SelectCol],
    extra: Any,
) -> NormalizedExpr | None:
    over = _over_node(fn_node)
    if over is None:
        return None
    ws = _window_def_to_spec(
        over, fn_node, dialect, alias_map, single_alias, param_store, next_lit_key, select_cols, extra
    )
    if ws is None:
        return None
    rid = extra.next_window_id()
    extra.window_registry.append(WindowRegistryStep(registry_id=rid, window_spec=ws))
    return NormalizedExpr.from_column(rid)


def _case_to_registry_step(
    node: exp.Case,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any,
) -> NormalizedExpr | None:
    branches: list[CaseWhenBranch] = []
    for cw in node.args.get("ifs") or ():
        if not isinstance(cw, exp.If):
            return None
        cond = _single_predicate_to_where(cw.this, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
        if cond is None:
            return None
        cond = replace(cond)
        res = _expr_full(
            cw.args.get("true") or cw.expression,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if res is None:
            return None
        branches.append(CaseWhenBranch(condition=cond, result=res))
    else_result: NormalizedExpr | None = None
    default = node.args.get("default")
    if default is not None:
        else_result = _expr_full(
            default,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if else_result is None:
            return None
    rid = extra.next_case_id()
    extra.case_registry.append(
        CaseRegistryStep(registry_id=rid, case_when=CaseWhenExpr(branches=branches, else_result=else_result))
    )
    return NormalizedExpr.from_column(rid)


def _coalesce_to_expr(
    node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None,
) -> NormalizedExpr | None:
    args: list[exp.Expression] = []
    if isinstance(node, exp.Coalesce):
        if node.this is not None:
            args.append(node.this)
        args.extend(list(node.expressions or []))
    elif isinstance(node, exp.Anonymous) and str(node.this or "").lower() == "coalesce":
        args = list(node.expressions or [])
    else:
        return None
    if len(args) < 2:
        return None
    first = _expr_full(
        args[0],
        dialect,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        extra,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if first is None:
        return None
    trail_args: list[Any] = []
    trail_keys: list[str] = []
    for a in args[1:]:
        lit = _where_literal_payload(a)
        if lit is not None:
            pk = next_lit_key()
            param_store[pk] = lit[0]
            trail_keys.append(pk)
            if isinstance(lit[0], (int, float)):
                trail_args.append(float(lit[0]))
            else:
                trail_args.append(str(lit[0]))
            continue
        ex = _expr_full(
            a,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if ex is None or not ex.column_ref:
            return None
        trail_keys.append("")
        trail_args.append(str(ex.column_ref))
    return NormalizedExpr(
        add_groups=[MulGroup(multiply=[first])],
        scalar_func="coalesce",
        scalar_func_args=trail_args,
        sarg_param_keys=trail_keys,
    )


def _aggregate_to_expr(
    node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None,
) -> NormalizedExpr | None:
    if isinstance(node, exp.WithinGroup) and isinstance(node.this, exp.PercentileCont):
        lit = _const_payload(node.this.this)
        if lit is None or lit[0] != 0.5:
            return None
        order_node = node.args.get("expression")
        if order_node is None:
            return None
        order_items = _window_sort_clause(
            order_node, dialect, alias_map, single_alias, param_store, next_lit_key, [], extra
        )
        if order_items is None or len(order_items) != 1:
            return None
        median_operand = order_items[0].expr
        return NormalizedExpr(add_groups=[MulGroup(multiply=[median_operand], agg_func="median")])

    if isinstance(node, exp.GroupConcat):
        col_node = node.this
        order_cols: list[OrderByCol] = []
        if isinstance(col_node, exp.Order):
            parsed_order = _window_sort_clause(
                col_node, dialect, alias_map, single_alias, param_store, next_lit_key, [], extra
            )
            if parsed_order is None:
                return None
            order_cols = parsed_order
            col_node = col_node.this
        concat_operand = _expr_full(
            col_node,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if concat_operand is None:
            return None
        sep_node = node.args.get("separator")
        sep_lit = _const_payload(sep_node) if sep_node is not None else None
        if sep_lit is None or sep_lit[1] != "string":
            return None
        pk = next_lit_key()
        param_store[pk] = sep_lit[0]
        return NormalizedExpr(
            add_groups=[
                MulGroup(
                    multiply=[concat_operand],
                    agg_func="string_agg",
                    agg_sep_param_key=pk,
                    agg_order_by=order_cols,
                )
            ]
        )

    fn_name = _dialect_map_scalar_func(dialect, _func_name(node))
    fn_name = SQLGLOT_AGG_FUNC_KEY_ALIASES.get(fn_name, fn_name)
    if fn_name not in SIMPLE_AGG_NAMES:
        return None
    distinct_flag = False
    inner_node: exp.Expression | None = None
    if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
        return NormalizedExpr.from_agg(fn_name, "*")
    if isinstance(node, exp.Distinct):
        distinct_flag = True
        inner_node = node.this
    elif hasattr(node, "this"):
        inner_node = node.this
    if inner_node is None:
        return NormalizedExpr.from_agg(fn_name, "*")
    agg_operand = _expr_full(
        inner_node,
        dialect,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        extra,
        allow_aggregate=False,
        allow_window=False,
        select_cols=None,
    )
    if agg_operand is None:
        return None
    return NormalizedExpr(add_groups=[MulGroup(multiply=[agg_operand], agg_func=fn_name, distinct=distinct_flag)])


def _expr_leaf(
    node: exp.Expression,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> NormalizedExpr | None:
    node = _unwrap_alias(node)
    if isinstance(node, exp.Column):
        qn = _qual_column_name(node, alias_map, single_alias, extra)
        if qn is None:
            return None
        return NormalizedExpr.from_column(qn)
    if isinstance(node, exp.Star):
        return NormalizedExpr(star=True)
    pay = _const_payload(node)
    if pay is not None:
        return _mask_literal(param_store, next_lit_key, pay)
    if isinstance(node, exp.CurrentDate):
        return NormalizedExpr(keyword="current_date")
    if isinstance(node, exp.CurrentTimestamp):
        return NormalizedExpr(keyword="current_timestamp")
    if isinstance(node, exp.CurrentTime):
        return NormalizedExpr(keyword="current_time")
    return None


def _expr_full(
    node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
    *,
    allow_aggregate: bool = True,
    allow_window: bool = True,
    select_cols: list[SelectCol] | None = None,
) -> NormalizedExpr | None:
    node = _unwrap_alias(node)
    leaf = _expr_leaf(node, alias_map, single_alias, param_store, next_lit_key, extra)
    if leaf is not None:
        return leaf
    if isinstance(node, exp.Case):
        if extra is None:
            return None
        return _case_to_registry_step(node, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
    coalesce = _coalesce_to_expr(node, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
    if coalesce is not None:
        return coalesce
    if _over_node(node) is not None:
        if not (allow_window and extra is not None and select_cols is not None):
            return None
        return _func_with_over_to_registry_step(
            node, dialect, alias_map, single_alias, param_store, next_lit_key, select_cols, extra
        )
    if allow_aggregate:
        agg = _aggregate_to_expr(node, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
        if agg is not None:
            return agg
    if isinstance(node, exp.Cast):
        tn = node.args.get("to")
        type_name = tn.sql(dialect=None).strip() if tn is not None else ""
        if type_name and "interval" in type_name.lower():
            if node.this is None:
                return None
            sp = _const_payload(node.this)
            if sp is None or sp[1] != "string":
                return None
            iv = _interval_from_literal(str(sp[0]))
            if iv is None:
                return None
            return NormalizedExpr(interval=iv)
        inner = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if inner is None or not type_name:
            return None
        return NormalizedExpr(add_groups=[MulGroup(multiply=[inner])], cast_type=type_name)
    if isinstance(node, (exp.Anonymous, exp.Func)) and not isinstance(
        node, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Neg)
    ):
        fn_name = _dialect_map_scalar_func(dialect, _func_name(node))
        if _dialect_import_unnest_policy(dialect, node):
            return None
        if fn_name == "extract" and isinstance(node, exp.Extract):
            unit_node = node.this
            source_node = node.expression
            unit = getattr(unit_node, "name", None) or (unit_node.sql(dialect=None) if unit_node else "")
            source = (
                _expr_full(
                    source_node,
                    dialect,
                    alias_map,
                    single_alias,
                    param_store,
                    next_lit_key,
                    extra,
                    allow_aggregate=False,
                    allow_window=False,
                    select_cols=select_cols,
                )
                if source_node is not None
                else NormalizedExpr(star=True)
            )
            if source is None:
                return None
            return NormalizedExpr(
                add_groups=[MulGroup(multiply=[source], scalar_func="extract", scalar_func_args=[str(unit).lower()])]
            )
        if fn_name in {"upper", "lower", "trim"}:
            args = list(getattr(node, "expressions", None) or [])
            if node.this is not None and node.this not in args:
                args = [node.this] + args
            if len(args) != 1:
                return None
            inner = _expr_full(
                args[0],
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if inner is None:
                return None
            return NormalizedExpr(add_groups=[MulGroup(multiply=[inner], scalar_func=fn_name)])
        if fn_name == "concat":
            args = list(getattr(node, "expressions", None) or [])
            if node.this is not None and node.this not in args:
                args = [node.this] + args
            if len(args) < 2:
                return None
            parts: list[NormalizedExpr] = []
            for arg in args:
                part = _expr_full(
                    arg,
                    dialect,
                    alias_map,
                    single_alias,
                    param_store,
                    next_lit_key,
                    extra,
                    allow_aggregate=False,
                    allow_window=False,
                    select_cols=select_cols,
                )
                if part is None:
                    return None
                parts.append(part)
            return NormalizedExpr(add_groups=[MulGroup(multiply=parts, scalar_func="concat")])
        return None
    if isinstance(node, exp.Nullif):
        l_e = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=select_cols,
        )
        r_e = _expr_full(
            node.expression,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=select_cols,
        )
        if l_e is None or r_e is None:
            return None
        return NormalizedExpr(
            add_groups=[MulGroup(multiply=[l_e], scalar_func="nullif", scalar_func_args=[_expr_sql_token(r_e)])]
        )
    if isinstance(node, exp.Neg):
        inner = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if inner is None:
            return None
        return NormalizedExpr(sub_groups=[_wrap_mul_term(inner)])
    if isinstance(node, (exp.Add, exp.Sub)):
        left = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        right = _expr_full(
            node.expression,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if left is None or right is None:
            return None
        lg = _wrap_mul_term(left)
        rg = _wrap_mul_term(right)
        if isinstance(node, exp.Add):
            return NormalizedExpr(add_groups=[lg, rg])
        return NormalizedExpr(add_groups=[lg], sub_groups=[rg])
    if isinstance(node, exp.Mul):
        left = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        right = _expr_full(
            node.expression,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if left is None or right is None:
            return None
        return _merge_product_exprs(left, right)
    if isinstance(node, exp.Div):
        left = _expr_full(
            node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        right = _expr_full(
            node.expression,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=allow_aggregate,
            allow_window=allow_window,
            select_cols=select_cols,
        )
        if left is None or right is None:
            return None
        return _merge_ratio_exprs(left, right)
    return None


def _projection_to_expr(
    node: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None,
    select_cols: list[SelectCol],
) -> NormalizedExpr | None:
    if isinstance(node, exp.Alias):
        node = node.this
    return _expr_full(
        node,
        dialect,
        alias_map,
        single_alias,
        param_store,
        next_lit_key,
        extra,
        allow_aggregate=True,
        allow_window=True,
        select_cols=select_cols,
    )


def _collect_in_literal_values(
    rexpr: exp.Expression | list[exp.Expression] | tuple[exp.Expression, ...],
) -> tuple[list[Any], str] | None:
    seq: list[exp.Expression]
    if isinstance(rexpr, exp.Tuple):
        seq = list(rexpr.expressions or [])
    elif isinstance(rexpr, (list, tuple)):
        seq = list(rexpr)
    else:
        seq = [rexpr]
    out: list[Any] = []
    vt: str | None = None
    for item in seq:
        lit = _where_literal_payload(item)
        if lit is None:
            return None
        val, t = lit
        if vt is None:
            vt = t
        elif vt != t:
            return None
        out.append(val)
    if not out or vt is None:
        return None
    return out, vt


def _comparison_op(node: exp.Expression) -> str | None:
    mapping: dict[type[exp.Expression], str] = {
        exp.EQ: "=",
        exp.NEQ: "<>",
        exp.LT: "<",
        exp.GT: ">",
        exp.LTE: "<=",
        exp.GTE: ">=",
        exp.Like: "like",
        exp.ILike: "ilike",
    }
    for cls, op in mapping.items():
        if isinstance(node, cls):
            return op
    nlike_cls = getattr(exp, "NLike", None)
    if nlike_cls is not None and isinstance(node, nlike_cls):
        inner = node.this
        if isinstance(inner, exp.Like):
            return "not like"
        if isinstance(inner, exp.ILike):
            return "not ilike"
    return None


def _single_predicate_to_where(
    p: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> WhereParam | None:
    p = _unwrap_alias(p)
    if isinstance(p, exp.Exists):
        return None
    if isinstance(p, exp.Not) and isinstance(p.this, exp.In):
        in_node = p.this
        left_e = _expr_full(
            in_node.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        collected = _collect_in_literal_values(in_node.expression)
        if collected is None:
            return None
        vals, vt = collected
        pk = next_lit_key()
        param_store[pk] = vals
        return WhereParam(left_expr=left_e, op="not in", value_type=vt, param_key=pk)
    if isinstance(p, exp.Is):
        left_e = _expr_full(
            p.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        rhs = p.expression
        if isinstance(rhs, exp.Not) and isinstance(rhs.this, exp.Null):
            return WhereParam(left_expr=left_e, op="is not null", value_type="null")
        if isinstance(rhs, exp.Null):
            return WhereParam(left_expr=left_e, op="is null", value_type="null")
        return None
    if isinstance(p, exp.Between):
        left_e = _expr_full(
            p.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        lo_lit = _where_literal_payload(p.args.get("low") or p.expression)
        hi_node = p.args.get("high")
        hi_lit = _where_literal_payload(hi_node) if hi_node is not None else None
        if lo_lit is None or hi_lit is None or lo_lit[1] != hi_lit[1]:
            return None
        pk_lo = next_lit_key()
        pk_hi = next_lit_key()
        param_store[pk_lo] = lo_lit[0]
        param_store[pk_hi] = hi_lit[0]
        return WhereParam(left_expr=left_e, op="between", value_type=lo_lit[1], param_key=pk_lo, param_key_hi=pk_hi)
    if isinstance(p, exp.In):
        left_e = _expr_full(
            p.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        in_exprs = p.expressions if p.expressions is not None else p.expression
        collected = _collect_in_literal_values(in_exprs)
        if collected is None:
            return None
        vals, vt = collected
        pk = next_lit_key()
        param_store[pk] = vals
        return WhereParam(left_expr=left_e, op="in", value_type=vt, param_key=pk)
    mapped = _comparison_op(p)
    if mapped is not None:
        mapped = _dialect_map_where_op(dialect, mapped) or mapped
        left_e = _expr_full(
            p.this,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if left_e is None:
            return None
        rexpr = p.expression
        if isinstance(rexpr, exp.Column):
            right_q = _qual_column_name(rexpr, alias_map, single_alias, extra)
            if right_q is None:
                return None
            return WhereParam(
                left_expr=left_e, op=mapped, right_expr=NormalizedExpr.from_column(right_q), value_type="column"
            )
        lit = _where_literal_payload(rexpr)
        if lit is not None:
            pk = next_lit_key()
            param_store[pk] = lit[0]
            return WhereParam(left_expr=left_e, op=mapped, value_type=lit[1], param_key=pk)
        right_e = _expr_full(
            rexpr,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if right_e is None:
            return None
        return WhereParam(left_expr=left_e, op=mapped, right_expr=right_e, value_type="column")
    return None


def _flatten_bool_and(node: exp.Expression) -> list[exp.Expression]:
    node = _unwrap_alias(node)
    if isinstance(node, exp.And):
        return _flatten_bool_and(node.this) + _flatten_bool_and(node.expression)
    return [node]


def _bool_nesting_too_deep(node: exp.Expression, depth: int) -> bool:
    if depth > 2:
        return True
    node = _unwrap_alias(node)
    if isinstance(node, exp.Not):
        return True
    if isinstance(node, (exp.And, exp.Or)):
        for arg in (node.this, node.expression):
            if arg is not None and _bool_nesting_too_deep(arg, depth + 1):
                return True
    return False


def _walk_bool_to_predicate_group(
    where: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> PredicateGroup | None:
    where = _unwrap_alias(where)
    if _bool_nesting_too_deep(where, 0):
        return None
    if isinstance(where, exp.Not):
        return None
    if isinstance(where, exp.Or):
        arms = _flatten_or_arms(where)
        or_groups: list[PredicateGroup] = []
        for arg in arms:
            if isinstance(arg, exp.And):
                sub = _flatten_bool_and(arg)
                and_preds: list[WhereParam] = []
                for subp in sub:
                    if isinstance(subp, exp.Exists):
                        return None
                    fp = _single_predicate_to_where(
                        subp, dialect, alias_map, single_alias, param_store, next_lit_key, extra
                    )
                    if fp is None:
                        return None
                    and_preds.append(fp)
                if and_preds:
                    or_groups.append(PredicateGroup(op="and", predicates=tuple(and_preds)))
                continue
            if isinstance(arg, exp.Exists):
                return None
            fp = _single_predicate_to_where(arg, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
            if fp is None:
                return None
            or_groups.append(PredicateGroup(op="and", predicates=(fp,)))
        if not or_groups:
            return None
        if len(or_groups) == 1:
            return or_groups[0]
        return PredicateGroup(op="or", groups=tuple(or_groups))
    if isinstance(where, exp.And):
        parts = _flatten_bool_and(where)
    else:
        parts = [where]
    preds: list[WhereParam] = []
    nested: list[PredicateGroup] = []
    for p in parts:
        if isinstance(p, exp.Exists):
            continue
        if isinstance(p, (exp.And, exp.Or)):
            child = _walk_bool_to_predicate_group(p, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
            if child is None:
                return None
            nested.append(child)
            continue
        fp = _single_predicate_to_where(p, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
        if fp is None:
            return None
        preds.append(fp)
    if not preds and not nested:
        return None
    if not nested:
        return PredicateGroup(op="and", predicates=tuple(preds))
    return PredicateGroup(op="and", predicates=tuple(preds), groups=tuple(nested))


def _walk_bool_to_where_groups(
    where: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> list[WhereParam] | None:
    group = _walk_bool_to_predicate_group(where, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
    return [cast(WhereParam, leaf) for leaf in group.leaves()] if group is not None else None


def _flatten_or_arms(node: exp.Expression) -> list[exp.Expression]:
    node = _unwrap_alias(node)
    if isinstance(node, exp.Or):
        return _flatten_or_arms(node.this) + _flatten_or_arms(node.expression)
    return [node]


def _where_to_where_params(
    where: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> PredicateGroup | None:
    return _walk_bool_to_predicate_group(where, dialect, alias_map, single_alias, param_store, next_lit_key, extra)


def _having_to_params(
    having: exp.Expression,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> list[HavingParam] | None:
    having = _unwrap_alias(having)
    if isinstance(having, exp.Or):
        return None
    parts = _flatten_bool_and(having) if isinstance(having, exp.And) else [having]
    out: list[HavingParam] = []
    for p in parts:
        if isinstance(p, exp.Exists):
            return None
        if isinstance(p, exp.Is):
            left_e = _expr_full(
                p.this,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=None,
            )
            if left_e is None:
                return None
            rhs = p.expression
            if isinstance(rhs, exp.Not) and isinstance(rhs.this, exp.Null):
                op_n = "is not null"
            elif isinstance(rhs, exp.Null):
                op_n = "is null"
            else:
                return None
            out.append(HavingParam(left_expr=left_e, op=op_n, value_type="null"))
            continue
        mapped = _comparison_op(p)
        if mapped is None:
            return None
        mapped = _dialect_map_where_op(dialect, mapped) or mapped
        left_node = p.this
        right_node = p.expression
        if left_node is None or right_node is None:
            return None
        left_n = _aggregate_to_expr(left_node, dialect, alias_map, single_alias, param_store, next_lit_key, extra)
        if left_n is None:
            left_n = _expr_full(
                left_node,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=True,
                allow_window=False,
                select_cols=None,
            )
        if left_n is None:
            return None
        if isinstance(right_node, exp.Column):
            right_q = _qual_column_name(right_node, alias_map, single_alias, extra)
            if right_q is None:
                return None
            out.append(
                HavingParam(
                    left_expr=left_n, op=mapped, right_expr=NormalizedExpr.from_column(right_q), value_type="column"
                )
            )
            continue
        lit = _where_literal_payload(right_node)
        if lit is not None:
            pk = next_lit_key()
            param_store[pk] = lit[0]
            out.append(HavingParam(left_expr=left_n, op=mapped, value_type=lit[1], param_key=pk))
            continue
        right_e = _expr_full(
            right_node,
            dialect,
            alias_map,
            single_alias,
            param_store,
            next_lit_key,
            extra,
            allow_aggregate=False,
            allow_window=False,
            select_cols=None,
        )
        if right_e is None:
            return None
        out.append(HavingParam(left_expr=left_n, op=mapped, right_expr=right_e, value_type="column"))
    return out


def _group_clause(
    group_node: exp.Group,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    select_cols: list[SelectCol],
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> list[NormalizedExpr] | None:
    out: list[NormalizedExpr] = []
    for n in group_node.expressions or ():
        if isinstance(n, exp.Column):
            qn = _qual_column_name(n, alias_map, single_alias, extra)
            if qn is None:
                return None
            out.append(NormalizedExpr.from_column(qn))
        elif isinstance(n, exp.Literal) and n.is_int:
            idx = _const_int_only(n)
            if idx is None or idx < 1 or idx > len(select_cols):
                return None
            out.append(select_cols[idx - 1].expr)
        else:
            ex = _expr_full(
                n,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if ex is None:
                return None
            out.append(ex)
    return out


def _sort_clause(
    order_node: exp.Order,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    select_cols: list[SelectCol],
    extra: Any | None = None,
) -> list[OrderByCol] | None:
    return _window_sort_clause(
        order_node, dialect, alias_map, single_alias, param_store, next_lit_key, select_cols, extra
    )


def _distinct_fields(
    sel: exp.Select,
    dialect: Any,
    alias_map: dict[str, str],
    single_alias: str | None,
    select_cols: list[SelectCol],
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any | None = None,
) -> tuple[int, list[NormalizedExpr]]:
    """Return ``(distinct_select_index, distinct_on)`` from the SELECT distinct clause."""
    distinct = sel.args.get("distinct")
    if not distinct:
        return -1, []
    if distinct is True:
        return 0, []
    if isinstance(distinct, exp.Distinct):
        on_exprs = distinct.args.get("on") or distinct.expressions or ()
        if not on_exprs:
            return 0, []
        distinct_on: list[NormalizedExpr] = []
        for dexpr in on_exprs:
            ex = _expr_full(
                dexpr,
                dialect,
                alias_map,
                single_alias,
                param_store,
                next_lit_key,
                extra,
                allow_aggregate=False,
                allow_window=False,
                select_cols=select_cols,
            )
            if ex is None and isinstance(dexpr, exp.Column):
                qn = _qual_column_name(dexpr, alias_map, single_alias, extra)
                if qn:
                    ex = NormalizedExpr.from_column(qn)
            if ex is not None:
                distinct_on.append(ex)
        if distinct_on:
            return -1, distinct_on
        return 0, []
    return 0, []


def _distinct_select_index(
    sel: exp.Select,
    alias_map: dict[str, str],
    single_alias: str | None,
    select_cols: list[SelectCol],
    extra: Any | None = None,
) -> int:
    """Derive ``distinct_select_index`` from plain ``DISTINCT`` (not ``DISTINCT ON``)."""
    idx, distinct_on = _distinct_fields(sel, None, alias_map, single_alias, select_cols, {}, lambda: "", extra)
    if distinct_on:
        return -1
    return idx


def _infer_output_columns(sel: exp.Select) -> list[str]:
    names: list[str] = []
    for i, proj in enumerate(sel.expressions or ()):
        if isinstance(proj, exp.Alias) and proj.alias:
            names.append(str(proj.alias).strip())
        else:
            names.append(f"col_{i}")
    return names


def _collect_range_bindings(
    sel: exp.Select, schema_tables: set[str], allowed_cte: frozenset[str]
) -> list[tuple[str, str]] | None:
    for join in _select_join_nodes(sel):
        if not _join_is_allowed(join):
            return None
    bindings: list[tuple[str, str]] = []
    for node in _iter_from_leaves(sel):
        pair = _table_alias_name(node)
        if pair is None:
            if isinstance(_unwrap_alias(node), exp.Subquery):
                return None
            return None
        alias, rel = pair
        rel_l = rel.lower()
        if rel_l not in schema_tables and rel_l not in allowed_cte:
            return None
        bindings.append((alias, rel_l if rel_l in schema_tables else rel))
    return bindings


def _count_alias_refs(sel: exp.Select, alias: str) -> int:
    n = 0
    for col in sel.find_all(exp.Column):
        if col.table and str(col.table).strip() == alias:
            n += 1
    return n


def _rewrite_table_to_cte(sel: exp.Select, lift_alias: str, physical: str, cte_name: str) -> bool:
    changed = False

    def rewrite_leaf(expr: exp.Expression) -> None:
        nonlocal changed
        pair = _table_alias_name(expr)
        if pair is None:
            return
        alias, rel = pair
        if alias == lift_alias and rel == physical:
            new_table = exp.Table(
                this=exp.to_identifier(cte_name), alias=exp.TableAlias(this=exp.to_identifier(cte_name))
            )
            expr.replace(new_table)
            changed = True

    def walk(expr: exp.Expression | None) -> None:
        if expr is None:
            return
        if isinstance(expr, exp.Join):
            walk(expr.this)
            walk(expr.expression)
            return
        rewrite_leaf(expr)

    root = _from_clause_root(sel)
    walk(root)
    for join in _select_join_nodes(sel):
        rhs = _join_rhs_unwrapped(join)
        if rhs is not None:
            rewrite_leaf(rhs)
    return changed


def _try_lift_self_join(sel: exp.Select, schema_tables: set[str], allowed_cte: frozenset[str], extra: Any) -> bool:
    if _from_clause_root(sel) is None and not _select_join_nodes(sel):
        return True
    bindings = _collect_range_bindings(sel, schema_tables, allowed_cte)
    if bindings is None:
        return False
    physical_groups: dict[str, list[str]] = {}
    for alias, tgt in bindings:
        if tgt in schema_tables:
            physical_groups.setdefault(tgt, []).append(alias)
    dup_targets = [t for t, als in physical_groups.items() if len(als) > 1]
    if not dup_targets:
        return True
    if len(dup_targets) != 1:
        return False
    phys = dup_targets[0]
    aliases = physical_groups[phys]
    if len(aliases) != 2:
        return False
    a0, a1 = aliases[0], aliases[1]
    n0, n1 = _count_alias_refs(sel, a0), _count_alias_refs(sel, a1)
    lift = a0 if n0 < n1 else a1
    cte_name = f"{SELF_JOIN_CTE_NAME_PREFIX}{lift}"
    if cte_name in allowed_cte or cte_name in schema_tables:
        return False
    if not _rewrite_table_to_cte(sel, lift, phys, cte_name):
        return False
    extra.qual_swap[lift] = cte_name
    extra.self_join_steps.append(
        RuntimeCteStep(
            cte_name=cte_name,
            tables=[phys],
            select_cols=[SelectCol(expr=NormalizedExpr(star=True))],
            output_columns=["col_0"],
            grain="row_level",
            distinct_select_index=-1,
        )
    )
    return True


def _try_lift_from_subqueries(
    sel: exp.Select,
    schema: SchemaGraph,
    dialect: Any,
    allowed_cte: frozenset[str],
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    extra: Any,
) -> bool:
    if _from_clause_root(sel) is None and not _select_join_nodes(sel):
        return True
    pending: list[tuple[exp.Subquery, str, str]] = []
    for branch in list(_iter_from_leaves(sel)):
        outer = branch
        node = _unwrap_alias(branch)
        if isinstance(node, exp.Subquery):
            alias = ""
            if isinstance(outer, exp.Alias):
                alias = str(outer.alias or outer.alias_or_name or "").strip()
            elif isinstance(outer, exp.Table) and outer.alias:
                alias = str(outer.alias_or_name or "").strip()
            if not alias:
                alias = f"sq_{len(pending)}"
            cte_name = alias
            if cte_name in allowed_cte:
                return False
            inner = node.this
            if not isinstance(inner, exp.Select):
                return False
            pending.append((node, alias, cte_name))
    for subq, alias, cte_name in pending:
        inner = subq.this
        if not isinstance(inner, exp.Select):
            return False
        chunk = _materialize_cte_body(inner, schema, dialect, cte_name, param_store, next_lit_key, allowed_cte)
        if chunk is None:
            return False
        extra.self_join_steps.extend(chunk)
        new_table = exp.Table(this=exp.to_identifier(cte_name), alias=exp.TableAlias(this=exp.to_identifier(alias)))
        subq.replace(new_table)
    return True


def _runtime_cte_step_from_body(sel: exp.Select, body: RuntimeIntent, cte_name: str) -> RuntimeCteStep:
    return RuntimeCteStep(
        cte_name=cte_name,
        tables=list(body.tables or []),
        select_cols=list(body.select_cols or []),
        group_by_cols=list(body.group_by_cols or []),
        order_by_cols=list(body.order_by_cols or []),
        where=body.where,
        having=body.having,
        param_values={},
        output_columns=_infer_output_columns(sel),
        grain=body.grain or "row_level",
        limit=body.limit,
        limit_param_key=body.limit_param_key or "",
        distinct_select_index=int(body.distinct_select_index),
        window_registry=list(body.window_registry or []),
        case_registry=list(body.case_registry or []),
    )


def _materialize_cte_body(
    sel: exp.Select,
    schema: SchemaGraph,
    dialect: Any,
    cte_name: str,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    allowed_cte: frozenset[str],
) -> list[RuntimeCteStep] | None:
    nested_steps: list[RuntimeCteStep] = []
    allowed_here: set[str] = set(allowed_cte)
    with_clause = sel.args.get("with_")
    if with_clause is not None:
        if with_clause.args.get("recursive"):
            return None
        prior: set[str] = set()
        for cte in with_clause.expressions or ():
            if not isinstance(cte, exp.CTE):
                return None
            nm = str(cte.alias_or_name or "").strip()
            inner_allowed = frozenset(prior | allowed_here)
            inner_sel = cte.this
            if not isinstance(inner_sel, exp.Select):
                return None
            sub = _materialize_cte_body(inner_sel, schema, dialect, nm, param_store, next_lit_key, inner_allowed)
            if sub is None:
                return None
            nested_steps.extend(sub)
            prior.add(nm)
        allowed_here |= prior
        sel = sel.copy()
        sel.set("with_", None)
    body_pair = _runtime_intent_body_from_select(
        sel, schema, dialect, param_store, next_lit_key, frozenset(allowed_here), _new_sqlglot_extra()
    )
    body, sj_local = body_pair
    if body is None:
        return None
    step = _runtime_cte_step_from_body(sel, body, cte_name)
    return nested_steps + sj_local + [step]


def _runtime_intent_body_from_select(
    sel: exp.Select,
    schema: SchemaGraph,
    dialect: Any,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    allowed_cte: frozenset[str],
    extra: Any | None = None,
) -> tuple[RuntimeIntent | None, list[RuntimeCteStep]]:
    if sel.find(exp.Union):
        return None, []
    sx = extra if extra is not None else _new_sqlglot_extra()
    schema_tables = set(schema.tables.keys())
    if not _try_lift_from_subqueries(sel, schema, dialect, allowed_cte, param_store, next_lit_key, sx):
        return None, []
    if not _try_lift_self_join(sel, schema_tables, allowed_cte, sx):
        return None, []
    extra_allowed = frozenset(set(allowed_cte) | {s.cte_name for s in sx.self_join_steps})
    bindings = _collect_range_bindings(sel, schema_tables, extra_allowed)
    if bindings is None:
        return None, []
    alias_map = {alias: target for alias, target in bindings}
    scope_aliases = sorted(alias_map.keys())
    single_alias: str | None = scope_aliases[0] if len(scope_aliases) == 1 else None
    physical_tables = sorted({alias_map[a] for a in alias_map if alias_map[a] in schema_tables})
    for rel in physical_tables:
        if rel not in schema_tables:
            return None, []
    select_cols: list[SelectCol] = []
    for proj in sel.expressions or ():
        ex = _projection_to_expr(proj, dialect, alias_map, single_alias, param_store, next_lit_key, sx, select_cols)
        if ex is None:
            return None, []
        select_cols.append(SelectCol(expr=ex))
    distinct_idx, distinct_on = _distinct_fields(
        sel, dialect, alias_map, single_alias, select_cols, param_store, next_lit_key, sx
    )
    where_group: PredicateGroup | None = None
    where_node = sel.args.get("where")
    if where_node is not None:
        pred = where_node.this if isinstance(where_node, exp.Where) else where_node
        where_group = _where_to_where_params(pred, dialect, alias_map, single_alias, param_store, next_lit_key, sx)
        if where_group is None:
            return None, []
    group_by_cols: list[NormalizedExpr] = []
    group_node = sel.args.get("group")
    if group_node is not None and isinstance(group_node, exp.Group):
        gb = _group_clause(group_node, dialect, alias_map, single_alias, select_cols, param_store, next_lit_key, sx)
        if gb is None:
            return None, []
        group_by_cols = gb
    order_by_cols: list[OrderByCol] = []
    order_node = sel.args.get("order")
    if order_node is not None and isinstance(order_node, exp.Order):
        ob = _sort_clause(order_node, dialect, alias_map, single_alias, param_store, next_lit_key, select_cols, sx)
        if ob is None:
            return None, []
        order_by_cols = ob
    having_param: list[HavingParam] = []
    having_node = sel.args.get("having")
    if having_node is not None:
        hpred = having_node.this if isinstance(having_node, exp.Having) else having_node
        hp = _having_to_params(hpred, dialect, alias_map, single_alias, param_store, next_lit_key, sx)
        if hp is None:
            return None, []
        having_param = hp
    limit_val: int | None = None
    limit_node = sel.args.get("limit")
    if limit_node is not None and isinstance(limit_node, exp.Limit):
        off_node = limit_node.args.get("offset")
        if off_node is not None:
            off_i = _const_int_only(off_node)
            if off_i is None:
                return None, []
            param_store[SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY] = off_i
        lim_expr = limit_node.expression
        if lim_expr is not None:
            lim_i = _const_int_only(lim_expr)
            if lim_i is None:
                return None, []
            limit_val = lim_i
    intent = RuntimeIntent(
        tables=physical_tables,
        grain="row_level",
        select_cols=select_cols,
        group_by_cols=group_by_cols,
        order_by_cols=order_by_cols,
        where=where_group,
        having=PredicateGroup.from_list(having_param),
        param_values={},
        cte_steps=[],
        natural_language="",
        limit=limit_val,
        distinct_select_index=distinct_idx,
        distinct_on=distinct_on,
        window_registry=list(sx.window_registry),
        case_registry=list(sx.case_registry),
    )
    return intent, list(sx.self_join_steps)


def _convert_select_stmt(
    sel: exp.Select,
    schema: SchemaGraph,
    dialect: Any,
    param_store: dict[str, Any],
    next_lit_key: Callable[[], str],
    *,
    outer_allowed_cte: frozenset[str] | None = None,
) -> tuple[RuntimeIntent, list[RuntimeCteStep]] | None:
    allowed_cte = set(outer_allowed_cte or ())
    cte_steps: list[RuntimeCteStep] = []
    with_clause = sel.args.get("with_")
    if with_clause is not None:
        if with_clause.args.get("recursive"):
            return None
        prior: set[str] = set()
        for cte in with_clause.expressions or ():
            if not isinstance(cte, exp.CTE):
                return None
            name = str(cte.alias_or_name or "").strip()
            inner = cte.this
            if not isinstance(inner, exp.Select):
                return None
            inner_allowed = frozenset(prior | allowed_cte)
            chunk = _materialize_cte_body(inner, schema, dialect, name, param_store, next_lit_key, inner_allowed)
            if chunk is None:
                return None
            cte_steps.extend(chunk)
            prior.add(name)
        allowed_cte = allowed_cte | prior
        sel = sel.copy()
        sel.set("with_", None)
    body_pair = _runtime_intent_body_from_select(
        sel, schema, dialect, param_store, next_lit_key, frozenset(allowed_cte), _new_sqlglot_extra()
    )
    body, sj_body = body_pair
    if body is None:
        return None
    return body, sj_body + cte_steps


def runtime_from_sqlglot_tree(
    tree: exp.Expression,
    schema: SchemaGraph,
    dialect: Any,
    *,
    param_store: dict[str, Any] | None = None,
    next_lit_key: Callable[[], str] | None = None,
) -> RuntimeIntent | None:
    """Materialise a :class:`RuntimeIntent` from a sqlglot SELECT tree."""
    if not isinstance(tree, exp.Select):
        inner = tree.find(exp.Select) if tree is not None else None
        if inner is None:
            return None
        tree = inner
    store = param_store if param_store is not None else {}
    key_fn = next_lit_key
    if key_fn is None:
        _, key_fn = _make_lit_key_factory()
    out = _convert_select_stmt(tree, schema, dialect, store, key_fn)
    if out is None:
        return None
    intent, cte_steps = out
    tables_agg: set[str] = set(intent.tables or [])
    for st in cte_steps:
        tables_agg.update(st.tables or [])
    return replace(intent, tables=sorted(tables_agg), param_values=dict(store), cte_steps=cte_steps)


def convert_sql_via_sqlglot(sql: str, schema: SchemaGraph, dialect: Dialect | Any) -> RuntimeIntent:
    """Parse SQL with the dialect's sqlglot reader and extract a :class:`RuntimeIntent`."""
    read_dialect = str(getattr(dialect, "sqlglot_dialect", "") or "postgres")
    prepped = _rewrite_distinct_on_cte_wrapper_sql(_dialect_preparse_sql(dialect, sql), read_dialect)
    try:
        tree = sqlglot.parse_one(prepped, dialect=read_dialect)
    except Exception as exc:
        raise ConfigError(f"sqlglot parse failed: {exc}") from exc
    if not isinstance(tree, exp.Select):
        inner = tree.find(exp.Select) if tree is not None else None
        if inner is None:
            raise ValueError("expected SELECT")
        tree = inner
    rt = runtime_from_sqlglot_tree(tree, schema, dialect)
    if rt is None:
        raise ValueError("unsupported SELECT shape for sqlglot conversion")
    return rt
