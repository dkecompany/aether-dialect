"""Shared sqlglot dialect helpers: parse/validate mixins, EXPLAIN scanners, partition pruning, information_schema helpers, and result backends."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, cast

import sqlglot
from sqlalchemy import create_engine, text

from ._config import (
    EngineRuntimeConfig,
    PolicyConfig,
)
from ._constants import (
    DBR_CARTESIAN_TOKENS,
    DBR_ZERO_ROW_RE,
    DUCKDB_EXPLAIN_CARTESIAN_TOKENS,
    DUCKDB_EXPLAIN_ESTIMATED_CARDINALITY_RE,
    INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL,
    INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL,
    INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL,
    INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL,
    INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL,
    NAMED_PLACEHOLDER_RE,
    PG_INNER_CONDITION_KEYS,
    PG_JOIN_CONDITION_KEYS,
    PG_JOIN_NODE_TYPES,
    QUALIFY_SKIP_IDENTIFIERS,
    REDSHIFT_SVV_FOREIGN_KEYS_SQL,
    SQL_BIND_TOKEN_RE,
    SQLITE_EXPLAIN_FULL_SCAN_TOKENS,
    ResultReaderKind,
)
from ._contracts_base import (
    AccessError,
    DatabasePingFailed,
    EngineContext,
    FilterParam,
    NormalizedExpr,
    SqlDiagnostic,
    SqlDiagnosticCode,
    StatementTimeoutError,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import (
    CatalogStructuralConstraintsIndex,
    CatalogTableStructuralConstraints,
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
)
from ._core_utils import (
    cost_cap_active,
    debug,
    diagnostic_debug_enabled,
    effective_explain_timeout_ms,
    engine_connect_likely_transient,
    normalize_array_contains_param_value,
    pipeline_trace,
    reconcile_execute_bind_params,
    sha256,
    stable_json,
)
from ._dialect import (
    Dialect,
    JoinEdge,
    check_schema_references_shared,
    explain_cost_gate_violation,
    is_permission_denied_error,
    normalize_named_placeholders,
    register_profile_schema_native_dispatch,
)
from ._schema_catalog import (
    array_element_type_from_data_type,
    looks_like_json_array_values,
    profile_schema,
    profile_schema_spark,
    profile_schema_sql_connector,
)

_STRUCTURAL_CODE_TO_DIAG: dict[str, SqlDiagnosticCode] = {
    "ast_parse_failed": SqlDiagnosticCode.AST_PARSE_FAILED,
    "multiple_statements": SqlDiagnosticCode.MULTIPLE_STATEMENTS,
    "no_root": SqlDiagnosticCode.NO_ROOT,
    "not_select": SqlDiagnosticCode.NOT_SELECT,
    "subquery_not_allowed": SqlDiagnosticCode.SUBQUERY_NOT_ALLOWED,
    "using_not_allowed": SqlDiagnosticCode.USING_NOT_ALLOWED,
    "cross_join_not_allowed": SqlDiagnosticCode.CROSS_JOIN_NOT_ALLOWED,
    "self_join_not_allowed": SqlDiagnosticCode.SELF_JOIN_NOT_ALLOWED,
    "exists_not_allowed": SqlDiagnosticCode.EXISTS_NOT_ALLOWED,
    "lateral_not_allowed": SqlDiagnosticCode.LATERAL_NOT_ALLOWED,
    "forbidden_structure": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_recursive": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_malformed": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
    "cte_contains_subquery": SqlDiagnosticCode.SUBQUERY_NOT_ALLOWED,
    "cte_contains_exists": SqlDiagnosticCode.EXISTS_NOT_ALLOWED,
    "cte_contains_set_op": SqlDiagnosticCode.FORBIDDEN_STRUCTURE,
}

ArrayStorageKind = Literal["native_array", "json_text_array", "unknown"]


def _is_string_like_data_type(data_type: str) -> bool:
    """Return True when *data_type* names a scalar text column."""
    sl = (data_type or "").strip().lower()
    return any(tok in sl for tok in ("varchar", "nvarchar", "text", "char", "string", "clob"))


def array_storage_kind(meta: ColumnMetadata | None) -> ArrayStorageKind:
    """Classify how an array-like column is stored for ``contains`` rendering."""
    if meta is None:
        return "unknown"
    is_native, _ = array_element_type_from_data_type(meta.data_type or "")
    if is_native:
        return "native_array"
    dt = (meta.data_type or "").strip().lower()
    if dt == "json" and meta.element_type:
        return "json_text_array"
    if meta.element_type and looks_like_json_array_values(meta.frequent_values or []):
        return "json_text_array"
    if meta.element_type and _is_string_like_data_type(meta.data_type or ""):
        return "json_text_array"
    if meta.element_type:
        return "json_text_array"
    return "unknown"


def quoted_json_element_token_predicate(
    *,
    column_sql: str,
    param_key: str,
    position_fn: Literal["INSTR", "STRPOS", "CHARINDEX"],
    value_cast: str,
    needle_style: Literal["concat", "sqlite_pipe"] = "concat",
) -> str:
    """Build a scalar quoted-token substring predicate for JSON-text array columns."""
    norm_param = f"LOWER(TRIM(BOTH '%' FROM CAST(:{param_key} AS {value_cast})))"
    if needle_style == "sqlite_pipe":
        needle = f"'\"' || {norm_param} || '\"'"
    else:
        needle = f"CONCAT('\"', {norm_param}, '\"')"
    haystack = f"LOWER(CAST({column_sql} AS {value_cast}))"
    if position_fn == "INSTR":
        return f"INSTR({haystack}, {needle}) > 0"
    if position_fn == "STRPOS":
        return f"STRPOS({haystack}, {needle}) > 0"
    if position_fn == "CHARINDEX":
        return f"CHARINDEX({needle}, {haystack}) > 0"
    raise ValueError(f"unsupported position_fn={position_fn!r}")


def _from_clause_root(sel: sqlglot.exp.Select) -> sqlglot.exp.Expression | None:
    from_ = sel.args.get("from_")
    if from_ is None:
        return None
    if isinstance(from_, sqlglot.exp.From):
        return cast(sqlglot.exp.Expression | None, from_.this)
    return None


def _walk_from_branches(expr: sqlglot.exp.Expression | None) -> Iterator[Any]:
    if expr is None:
        return
    if isinstance(expr, sqlglot.exp.Join):
        yield from _walk_from_branches(expr.this)
        yield from _walk_from_branches(expr.expression)
        return
    yield expr


def _tables_from_from_root(root: sqlglot.exp.Expression | None) -> list[str]:
    names: list[str] = []
    for node in _walk_from_branches(root):
        if isinstance(node, sqlglot.exp.Table) and node.name:
            names.append(node.name.strip().lower())
    return names


def _join_rhs_unwrapped(join: sqlglot.exp.Join) -> sqlglot.exp.Expression | None:
    raw = join.args.get("expression") or join.args.get("this")
    node = raw
    while isinstance(node, sqlglot.exp.Alias):
        node = node.this
    return node


def _validate_select_structural_inner(
    select: sqlglot.exp.Select,
    scalar_cte_names: frozenset[str] | None = None,
) -> tuple[bool, str]:
    if list(select.find_all(sqlglot.exp.Exists)):
        return False, "exists_not_allowed"
    if list(select.find_all(sqlglot.exp.Lateral)):
        return False, "lateral_not_allowed"
    for join in select.find_all(sqlglot.exp.Join):
        if join.args.get("using"):
            return False, "using_not_allowed"
        kind = join.args.get("kind")
        if kind is not None and str(kind).upper() == "CROSS":
            allowed = False
            right = _join_rhs_unwrapped(join)
            if scalar_cte_names and isinstance(right, sqlglot.exp.Table):
                rn = (right.name or "").strip().lower()
                if rn and rn in scalar_cte_names:
                    allowed = True
            if not allowed:
                return False, "cross_join_not_allowed"
    if list(select.find_all(sqlglot.exp.Subquery)):
        return False, "subquery_not_allowed"
    if list(select.find_all(sqlglot.exp.Union)):
        return False, "forbidden_structure"
    names = _tables_from_from_root(_from_clause_root(select))
    if len(names) >= 2 and len(names) != len(set(names)):
        return False, "self_join_not_allowed"
    return True, ""


def _validate_with_ctes(with_clause: sqlglot.exp.With) -> tuple[bool, str]:
    if with_clause.args.get("recursive"):
        return False, "cte_recursive"
    for cte in with_clause.expressions:
        body = cte.this
        if isinstance(body, sqlglot.exp.Union):
            return False, "cte_contains_set_op"
        if isinstance(body, sqlglot.exp.Select):
            nested = body.args.get("with_")
            if nested is not None:
                okn, errn = _validate_with_ctes(nested)
                if not okn:
                    return False, errn
            inner = body.copy()
            inner.set("with_", None)
            ok, err = _validate_select_structural_inner(inner)
            if not ok:
                return False, err
    return True, ""


def ast_structural_valid_sqlglot(
    sql: str,
    *,
    sqlglot_dialect: str,
    scalar_cte_names: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Return structural validity for a single SELECT via sqlglot."""
    try:
        tree = sqlglot.parse_one(sql, dialect=sqlglot_dialect)
    except Exception:
        return False, "ast_parse_failed"
    if isinstance(tree, sqlglot.exp.Union):
        return False, "multiple_statements"
    if not isinstance(tree, sqlglot.exp.Select):
        return False, "not_select"
    select = tree
    wc = select.args.get("with_")
    if wc is not None:
        okw, errw = _validate_with_ctes(wc)
        if not okw:
            return False, errw
    main = select.copy()
    main.set("with_", None)
    return _validate_select_structural_inner(main, scalar_cte_names)


class _SqlglotParseHost(Protocol):
    sqlglot_dialect: ClassVar[str]

    def quote_table_column(self, table: str, column: str) -> str: ...


class _SqlalchemyExecutionHost(Protocol):
    config: EngineRuntimeConfig
    engine: Any | None

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None: ...

    def finalize_render(
        self,
        sql: str,
        params: dict[str, Any],
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> str: ...

    def _disable_explain_on_permission_denied(self, error_message: str) -> bool: ...

    def profile_schema_dispatch(self, sg: SchemaGraph) -> None: ...

    def qualified_table_ref(
        self,
        table: str,
        *,
        kind: Literal["table", "view"] = "table",
    ) -> str: ...

    def quote_identifier(self, ident: str) -> str: ...


class SqlglotParseMixin:
    """Sqlglot parse, join attachment, projection, and semantic validation helpers."""

    sqlglot_dialect: ClassVar[str] = ""

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """Validate SQL structurally and semantically via sqlglot."""
        ok, code = ast_structural_valid_sqlglot(
            sql,
            sqlglot_dialect=self.sqlglot_dialect,
            scalar_cte_names=scalar_cte_names,
        )
        if not ok:
            mapped = _STRUCTURAL_CODE_TO_DIAG.get(code, SqlDiagnosticCode.FORBIDDEN_STRUCTURE)
            return [SqlDiagnostic(code=mapped, message=code, node_kind=None)]
        diags: list[SqlDiagnostic] = []
        try:
            tree = sqlglot.parse_one(sql, dialect=self.sqlglot_dialect)
        except Exception:
            return [SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="parse failed")]
        if not isinstance(tree, sqlglot.exp.Select):
            return diags
        cte_names = self._collect_cte_names(tree)
        alias_to_table = self._collect_table_aliases(tree)
        if schema is not None:
            refs = self._collect_column_refs(tree)
            diags += check_schema_references_shared(refs, alias_to_table, cte_names, schema)
        diags += self._check_grouping(tree)
        diags += self._check_cte_closure(tree, cte_names)
        if declared_params is not None:
            diags += self._check_param_coverage(sql, declared_params)
        return diags

    def _collect_cte_names(self, tree: sqlglot.exp.Select) -> set[str]:
        names: set[str] = set()
        wc = tree.args.get("with_")
        if wc is None:
            return names
        for cte in wc.expressions or ():
            alias = cte.alias_or_name
            if isinstance(alias, str) and alias:
                names.add(alias.lower())
        return names

    def _collect_table_aliases(self, tree: sqlglot.exp.Select) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in tree.find_all(sqlglot.exp.Table):
            real = (t.name or "").lower()
            if not real:
                continue
            out[real] = real
            alias_node = t.args.get("alias")
            if alias_node is not None:
                a = alias_node.name if hasattr(alias_node, "name") else None
                if isinstance(a, str) and a:
                    out[a.lower()] = real
        return out

    def _collect_column_refs(self, tree: sqlglot.exp.Select) -> list[tuple[str | None, str]]:
        out: list[tuple[str | None, str]] = []
        for c in tree.find_all(sqlglot.exp.Column):
            col = c.name or ""
            if not col or col == "*":
                continue
            tbl = c.table or None
            out.append((tbl or None, col))
        return out

    def _check_grouping(self, tree: sqlglot.exp.Select) -> list[SqlDiagnostic]:
        diags: list[SqlDiagnostic] = []
        where = tree.args.get("where")
        if where is not None:
            for agg in where.find_all(sqlglot.exp.AggFunc):
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.AGG_IN_WHERE,
                        message=f"aggregate {type(agg).__name__.lower()!r} in WHERE",
                        node_kind="AggFunc",
                        offending_identifier=type(agg).__name__.lower(),
                    )
                )
                break
        having = tree.args.get("having")
        group = tree.args.get("group")
        if having is not None and group is None:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.HAVING_WITHOUT_GROUP,
                    message="HAVING without GROUP BY",
                    node_kind="Select",
                )
            )
        return diags

    def _check_cte_closure(self, tree: sqlglot.exp.Select, cte_names: set[str]) -> list[SqlDiagnostic]:
        if not cte_names:
            return []
        referenced: set[str] = set()
        for t in tree.find_all(sqlglot.exp.Table):
            name = (t.name or "").lower()
            if name in cte_names:
                referenced.add(name)
        for col in tree.find_all(sqlglot.exp.Column):
            tbl = col.table
            name = tbl if isinstance(tbl, str) else getattr(tbl, "name", "") or ""
            if name and name.lower() in cte_names:
                referenced.add(name.lower())
        unreferenced = sorted(cte_names - referenced)
        return [
            SqlDiagnostic(
                code=SqlDiagnosticCode.CTE_UNREFERENCED,
                message=f"CTE {n!r} is defined but never referenced",
                node_kind="CTE",
                offending_identifier=n,
            )
            for n in unreferenced
        ]

    def _check_param_coverage(self, sql: str, declared: set[str]) -> list[SqlDiagnostic]:
        diags: list[SqlDiagnostic] = []
        seen: set[str] = set()
        for match in NAMED_PLACEHOLDER_RE.finditer(sql):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in declared:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.PARAM_UNBOUND,
                        message=f"unbound placeholder :{name}",
                        node_kind="Parameter",
                        offending_identifier=name,
                    )
                )
        for match in SQL_BIND_TOKEN_RE.finditer(sql):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in declared:
                prefix = sql[match.start()]
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.PARAM_UNBOUND,
                        message=f"unbound placeholder {prefix}{name}",
                        node_kind="Parameter",
                        offending_identifier=name,
                    )
                )
        for match in re.finditer(r"@(p\d+|s\d+)\b", sql):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in declared:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.PARAM_UNBOUND,
                        message=f"unbound placeholder @{name}",
                        node_kind="Parameter",
                        offending_identifier=name,
                    )
                )
        return diags

    def parse_select(self, sql: str) -> sqlglot.exp.Select | None:
        """Parse *sql* via sqlglot; return ``None`` for non-``SELECT`` roots or parse failure."""
        try:
            tree = sqlglot.parse_one(sql, dialect=self.sqlglot_dialect)
        except Exception:
            return None
        if not isinstance(tree, sqlglot.exp.Select):
            return None
        return tree

    def ordered_join_carrier_froms(self, parsed: sqlglot.exp.Select) -> list[sqlglot.exp.Select] | None:
        """Return inner-CTE ``Select`` nodes followed by the outer ``Select``."""
        out: list[sqlglot.exp.Select] = []
        wc = parsed.args.get("with_")
        if wc:
            for cte in wc.expressions:
                inner = cte.this
                if isinstance(inner, sqlglot.exp.Select) and isinstance(inner.args.get("from_"), sqlglot.exp.From):
                    out.append(inner)
        if isinstance(parsed.args.get("from_"), sqlglot.exp.From):
            out.append(parsed)
        return out

    def from_anchor_of(self, carrier: sqlglot.exp.Select) -> str | None:
        """Return the bare table name of *carrier*'s leftmost ``FROM`` leaf."""
        from_node = carrier.args.get("from_") if isinstance(carrier, sqlglot.exp.Select) else None
        if not isinstance(from_node, sqlglot.exp.From):
            return None
        target = from_node.this
        while isinstance(target, sqlglot.exp.Join):
            inner = target.this
            if inner is None:
                return None
            target = inner
        if not isinstance(target, sqlglot.exp.Table):
            return None
        name = target.name or ""
        if not name:
            return None
        return name.lower()

    def attach_joins(
        self,
        parsed: sqlglot.exp.Select,
        from_handle: sqlglot.exp.Select,
        edges: list[JoinEdge],
    ) -> bool:
        """Build sqlglot ``Join`` nodes from *edges* and append them to *from_handle*."""
        if not edges:
            return False
        new_joins: list[sqlglot.exp.Join] = []
        for edge in edges:
            on_expr = self._build_on_expr(edge.on_terms)
            if on_expr is None:
                return False
            table_node = sqlglot.exp.Table(this=sqlglot.exp.to_identifier(edge.table))
            if edge.alias:
                table_node.set(
                    "alias",
                    sqlglot.exp.TableAlias(this=sqlglot.exp.to_identifier(edge.alias)),
                )
            join_kwargs: dict[str, Any] = {
                "this": table_node,
                "on": on_expr,
                "kind": "INNER",
            }
            if edge.kind == "LEFT":
                join_kwargs["side"] = "LEFT"
                join_kwargs["kind"] = None
            new_joins.append(sqlglot.exp.Join(**{k: v for k, v in join_kwargs.items() if v is not None}))
        existing = list(from_handle.args.get("joins") or [])
        from_handle.set("joins", existing + new_joins)
        return True

    def _build_on_expr(
        self,
        on_terms: tuple[tuple[str, str, str, str], ...],
    ) -> sqlglot.exp.Expression | None:
        if not on_terms:
            return None
        eqs: list[sqlglot.exp.Expression] = []
        host = cast(_SqlglotParseHost, self)
        for left_token, left_col, right_token, right_col in on_terms:
            lhs_sql = host.quote_table_column(left_token, left_col)
            rhs_sql = host.quote_table_column(right_token, right_col)
            try:
                pred_tree = sqlglot.parse_one(
                    f"SELECT 1 FROM t WHERE {lhs_sql} = {rhs_sql}",
                    dialect=self.sqlglot_dialect,
                )
            except Exception:
                return None
            where_node = pred_tree.args.get("where")
            if where_node is None:
                return None
            eqs.append(where_node.this)
        node: sqlglot.exp.Expression = eqs[0]
        for nxt in eqs[1:]:
            node = sqlglot.exp.And(this=node, expression=nxt)
        pipeline_trace(
            "pipeline.join_resolve.dialect_quote_join_clause",
            lambda: stable_json({"conjuncts": len(on_terms)}),
        )
        return node

    def attach_extra_from_and_where(
        self,
        parsed: sqlglot.exp.Select,
        from_handle: sqlglot.exp.Select,
        extra_from_tables: list[str],
        where_edges: list[JoinEdge],
    ) -> bool:
        """Append comma-FROM tables and AND equality predicates into the carrier ``WHERE``."""
        if not extra_from_tables and not where_edges:
            return True
        if extra_from_tables:
            existing_joins = list(from_handle.args.get("joins") or [])
            for tbl in extra_from_tables:
                existing_joins.append(
                    sqlglot.exp.Join(
                        this=sqlglot.exp.Table(this=sqlglot.exp.to_identifier(tbl)),
                        kind="CROSS",
                    ),
                )
            from_handle.set("joins", existing_joins)
        if not where_edges:
            return True
        new_eqs: list[sqlglot.exp.Expression] = []
        for edge in where_edges:
            pred = self._build_on_expr(edge.on_terms)
            if pred is None:
                return False
            new_eqs.append(pred)
        if not new_eqs:
            return True
        new_pred: sqlglot.exp.Expression = new_eqs[0]
        for nxt in new_eqs[1:]:
            new_pred = sqlglot.exp.And(this=new_pred, expression=nxt)
        existing_where = from_handle.args.get("where")
        if existing_where is None:
            from_handle.set("where", sqlglot.exp.Where(this=new_pred))
        else:
            existing_pred = existing_where.this
            merged_pred: sqlglot.exp.Expression = sqlglot.exp.And(this=existing_pred, expression=new_pred)
            existing_where.set("this", merged_pred)
        return True

    def replace_projection(
        self,
        parsed: sqlglot.exp.Select,
        items: list[tuple[str, str | None]],
    ) -> bool:
        """Replace the outer ``Select`` projection list by parsing each expression via sqlglot."""
        new_exprs: list[sqlglot.exp.Expression] = []
        for expr_sql, alias in items:
            try:
                tree = sqlglot.parse_one(f"SELECT {expr_sql}", dialect=self.sqlglot_dialect)
            except Exception:
                return False
            if not isinstance(tree, sqlglot.exp.Select):
                return False
            tlist = tree.args.get("expressions") or []
            if len(tlist) != 1:
                return False
            value_node = tlist[0]
            if isinstance(value_node, sqlglot.exp.Alias) and alias:
                value_node = value_node.this
            if alias:
                value_node = sqlglot.exp.alias_(value_node, alias)
            new_exprs.append(value_node)
        parsed.set("expressions", new_exprs)
        return True

    def emit_sql(self, parsed: sqlglot.exp.Select) -> str:
        """Render *parsed* via sqlglot preserving ``:pN`` / ``:sN`` placeholders."""
        return normalize_named_placeholders(parsed.sql(dialect=self.sqlglot_dialect))


class SqlalchemyExecutionMixin:
    """SQLAlchemy engine bootstrap, EXPLAIN, execute, profiling, and DDL probe helpers."""

    config: EngineRuntimeConfig
    engine: Any | None
    _backend: ResultBackend | None

    def __init__(self, config: EngineRuntimeConfig, sqlalchemy_engine: Any | None = None) -> None:
        """Attach runtime configuration and open a SQLAlchemy engine when not supplied. Embedded DuckDB and SQLite dialects pass a pre-built ``StaticPool`` engine that wraps the same native connection used for execution."""
        Dialect.__init__(cast(Dialect, self), config)
        self.engine = sqlalchemy_engine
        self._backend = None
        if self.engine is not None:
            self._ensure_result_backend()
            return
        url = config.db_url() if hasattr(config, "db_url") else ""
        connect_args = config.connect_args() if hasattr(config, "connect_args") else {}
        try:
            self.engine = create_engine(url, connect_args=connect_args or {}, future=True)
        except Exception as exc:
            if engine_connect_likely_transient(exc):
                raise DatabasePingFailed(f"SQLAlchemy engine creation failed: {exc}") from exc
            raise
        self._ensure_result_backend()

    def _ensure_result_backend(self) -> None:
        """Attach a SQLAlchemy result backend when ``self.engine`` is available."""
        if self._backend is not None:
            return
        if getattr(self, "engine", None) is not None:
            host = cast(_SqlalchemyExecutionHost, self)
            self._backend = SqlAlchemyResultBackend(
                self.engine,
                dialect_name=self.__class__.__name__,
                timeout_sql_provider=host.profile_statement_timeout_sql,
            )

    @property
    def result_backend(self) -> ResultBackend | None:
        """Return the active row-fetch backend for this dialect instance."""
        self._ensure_result_backend()
        return self._backend

    def parse_explain_plan(
        self,
        rows: list[Any],
        *,
        schema: SchemaGraph | None = None,
    ) -> tuple[float | None, float | None, list[SqlDiagnostic], str]:
        """Parse engine-specific EXPLAIN output rows into estimates and. soft diagnostics."""
        _ = rows, schema
        return None, None, [], ""

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run ``EXPLAIN`` via SQLAlchemy and return structured findings."""
        host = cast(_SqlalchemyExecutionHost, self)
        finalized = host.finalize_render(
            sql,
            params or {},
            schema=schema,
            intent=intent,
        )
        explain_sql = self.explain_statement_sql(finalized)
        try:
            backend = self.result_backend
            if backend is None:
                return True, [], ""
            tm = effective_explain_timeout_ms()
            explain_params = params if params and SQL_BIND_TOKEN_RE.search(explain_sql) else None
            rows = backend.fetch_rows(explain_sql, explain_params, timeout_ms=tm)
            est_rows, est_bytes, soft_diags, plan_text = self.parse_explain_plan(
                list(rows),
                schema=schema,
            )
            if est_rows is not None or est_bytes is not None:
                failed, why = explain_cost_gate_violation(est_rows, est_bytes)
                if failed:
                    return (
                        False,
                        soft_diags
                        + [
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                                message=why,
                            )
                        ],
                        why,
                    )
            return True, soft_diags, plan_text
        except Exception as e:
            err = str(e)
            if host._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (
                False,
                [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)],
                err,
            )

    def explain_statement_sql(self, finalized_sql: str) -> str:
        """Return the engine-specific EXPLAIN wrapper for *finalized_sql*."""
        return f"EXPLAIN {finalized_sql}"

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via the active result backend and return row tuples."""
        backend = self.result_backend
        if backend is None:
            raise RuntimeError(f"{self.__class__.__name__} has no result backend")
        tm = PolicyConfig.STATEMENT_TIMEOUT_MS if cost_cap_active(PolicyConfig.STATEMENT_TIMEOUT_MS) else None
        return backend.fetch_rows(sql, params, timeout_ms=tm)

    def profile_schema(self, sg: SchemaGraph) -> None:
        """Run column profiling via :meth:`Dialect.profile_schema_dispatch`."""
        cast(_SqlalchemyExecutionHost, self).profile_schema_dispatch(sg)

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return SHA-256 over information_schema column and UNIQUE rows or empty string on failure."""
        _ = schema_context
        try:
            host = cast(_SqlalchemyExecutionHost, self)
            schema_name = str(getattr(host.config, "SCHEMA", None) or getattr(host.config, "DATABASE", None) or "")
            if not schema_name or host.engine is None:
                return ""
            with host.engine.connect() as conn:
                rows = conn.execute(
                    text(INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL),
                    {"s": schema_name},
                ).fetchall()
                uniq_rows = conn.execute(
                    text(INFORMATION_SCHEMA_UNIQUE_COLUMNS_DDL_PROBE_SQL),
                    {"s": schema_name},
                ).fetchall()
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            payload_uniq = "\n".join("|".join("" if c is None else str(c) for c in r) for r in uniq_rows)
            return sha256(payload_cols + "\n##UNIQUE##\n" + payload_uniq)
        except Exception as exc:
            debug(f"[{self.__class__.__name__}.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def refresh_full_table_distinct_for_pk_inference(
        self,
        table_name: str,
        col_name: str,
        *,
        table_kind: Literal["table", "view"] = "table",
    ) -> tuple[int, int, float] | None:
        """Run full-table statistics for PK inference after sampled. profiling."""
        try:
            _ = table_kind
            host = cast(_SqlalchemyExecutionHost, self)
            backend = self.result_backend
            if backend is None:
                return None
            q_table = host.qualified_table_ref(table_name, kind=table_kind)
            q_col = host.quote_identifier(col_name)
            sql = (
                f"SELECT COUNT(*) AS cnt, COUNT(DISTINCT {q_col}) AS dist, "
                f"COUNT(*) - COUNT({q_col}) AS nulls FROM {q_table}"
            )
            rows = backend.fetch_rows(sql)
            if not rows:
                return None
            row = rows[0]
            cnt = int(row[0] or 0)
            dist = int(row[1] or 0)
            nulls = int(row[2] or 0)
            nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
            return (dist, cnt, nr)
        except Exception as exc:
            debug(
                f"[{self.__class__.__name__}.refresh_full_table_distinct_for_pk_inference] failed: {exc!r}",
            )
            return None

    def refresh_composite_distinct_for_pk_inference(
        self,
        table_name: str,
        col_names: list[str],
        *,
        table_kind: Literal["table", "view"] = "table",
    ) -> tuple[int, int, float] | None:
        """Run full-table composite distinct statistics for multi-column PK inference."""
        if not col_names:
            return None
        try:
            host = cast(_SqlalchemyExecutionHost, self)
            backend = self.result_backend
            if backend is None:
                return None
            q_table = host.qualified_table_ref(table_name, kind=table_kind)
            if len(col_names) == 1:
                q_col = host.quote_identifier(col_names[0])
                distinct_expr = f"COUNT(DISTINCT {q_col})"
                null_expr = f"COUNT(*) - COUNT({q_col})"
            else:
                parts = ", ".join(host.quote_identifier(c) for c in col_names)
                distinct_expr = f"COUNT(DISTINCT ({parts}))"
                null_checks = " OR ".join(f"{host.quote_identifier(c)} IS NULL" for c in col_names)
                null_expr = f"SUM(CASE WHEN {null_checks} THEN 1 ELSE 0 END)"
            sql = f"SELECT COUNT(*) AS cnt, {distinct_expr} AS dist, {null_expr} AS nulls FROM {q_table}"
            rows = backend.fetch_rows(sql)
            if not rows:
                return None
            row = rows[0]
            cnt = int(row[0] or 0)
            dist = int(row[1] or 0)
            nulls = int(row[2] or 0)
            nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
            return (dist, cnt, nr)
        except Exception as exc:
            debug(
                f"[{self.__class__.__name__}.refresh_composite_distinct_for_pk_inference] failed: {exc!r}",
            )
            return None


class SqlglotEngineDialect(SqlglotParseMixin, SqlalchemyExecutionMixin, Dialect):
    """Base dialect using sqlglot for parse/emit and SQLAlchemy for execution."""

    def qualified_table_ref(
        self,
        table: str,
        kind: Literal["table", "view"] = "table",
    ) -> str:
        """Return a schema-qualified table reference when a schema name is configured."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Return PK, FK, and UNIQUE metadata from ``information_schema`` when available."""
        schema_name = self.schema_name()
        connection = getattr(self, "_native_connection", None)
        return structural_constraints_index_for_schema(
            self,
            schema_name,
            engine=getattr(self, "engine", None),
            connection=connection,
        )

    def pre_execute_rewrite(self, sql: str) -> str:
        """Apply engine-specific SQL rewrites before parameter substitution."""
        return super().pre_execute_rewrite(sql)

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Normalize finalized SQL after substitution for engine- specific emission quirks."""
        return super().post_render_normalize(sql, stage=stage)

    def profile_schema_dispatch(self, sg: SchemaGraph) -> None:
        """Profile tables using the active native backend chain with SQLAlchemy fallback."""
        super().profile_schema_dispatch(sg)

    def apply_execute_cost_limits(self, target: Any) -> None:
        """Apply engine-specific execute-time cost caps to a driver handle or job config."""
        super().apply_execute_cost_limits(target)

    def finalize_render(
        self,
        sql_param: str,
        params: dict[str, Any],
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
        execution_sql_override: str | None = None,
        structural_defaults: dict[str, Any] | None = None,
    ) -> str:
        """Produce executable SQL through the shared sqlglot render pipeline."""
        return super().finalize_render(
            sql_param,
            params,
            schema=schema,
            intent=intent,
            execution_sql_override=execution_sql_override,
            structural_defaults=structural_defaults,
        )


def can_explain_for_backends(
    dialect: Dialect,
    *,
    sqlalchemy_engine: Any | None = None,
    native_connection: Any | None = None,
    spark_session: Any | None = None,
) -> bool:
    """Return True when any configured execution backend can run ``EXPLAIN``."""
    if dialect._explain_disabled:
        return False
    if sqlalchemy_engine is not None:
        return True
    if native_connection is not None:
        return True
    if spark_session is not None:
        return True
    return False


def profile_schema_native_dispatch(dialect: Dialect, sg: SchemaGraph) -> None:
    """Profile *sg* using native backends when available, otherwise SQLAlchemy."""
    if dialect.name == "databricks":
        connection = getattr(dialect, "connection", None)
        spark = getattr(dialect, "spark", None)
        engine = getattr(dialect, "engine", None)
        backend = getattr(dialect, "_backend", None)
        catalog = dialect.catalog_name() or ""
        schema_name = dialect.schema_name()
        if backend is not None:
            kind = getattr(backend, "kind", "")
            if kind == "connector" and connection is not None:
                profile_schema_sql_connector(connection, catalog, schema_name, sg, dialect=dialect)
                return
            if kind == "spark":
                if spark is None:
                    from pyspark.sql import SparkSession

                    spark = SparkSession.builder.getOrCreate()
                profile_schema_spark(spark, catalog, schema_name, sg, dialect=dialect)
                return
        if engine is not None:
            profile_schema(engine, sg, dialect=dialect)
            return
        raise NotImplementedError(f"{type(dialect).__name__} has no Databricks profiling backend")
    if dialect.name == "snowflake":
        backend = getattr(dialect, "_backend", None)
        snowpark = getattr(dialect, "_snowpark_session", None)
        engine = getattr(dialect, "engine", None)
        if backend is not None and getattr(backend, "kind", "") == "snowflake_arrow":
            if engine is not None:
                profile_schema(engine, sg, dialect=dialect)
                return
            if snowpark is not None:
                profile_schema(snowpark, sg, dialect=dialect)
                return
        if engine is not None:
            profile_schema(engine, sg, dialect=dialect)
            return
        raise NotImplementedError(f"{type(dialect).__name__} has no Snowflake profiling backend")
    if dialect.name == "bigquery":
        client = getattr(dialect, "_bq_client", None)
        engine = getattr(dialect, "engine", None)
        if client is not None and engine is not None:
            profile_schema(engine, sg, dialect=dialect)
            return
        if client is not None:
            profile_schema(client, sg, dialect=dialect)
            return
        if engine is not None:
            profile_schema(engine, sg, dialect=dialect)
            return
        raise NotImplementedError(f"{type(dialect).__name__} has no BigQuery profiling backend")
    engine = getattr(dialect, "engine", None)
    if engine is not None:
        profile_schema(engine, sg, dialect=dialect)
        return
    raise NotImplementedError(f"{type(dialect).__name__} has no profiling backend")


def normalize_datetrunc_sql(sql: str, *, sqlglot_dialect: str) -> str:
    """Rewrite parsed ``Anonymous`` ``DATETRUNC`` nodes to ``DATE_TRUNC`` with canonical argument order."""
    try:
        tree = sqlglot.parse_one(sql, dialect=sqlglot_dialect)
    except Exception:
        debug(f"[normalize_datetrunc_sql] sqlglot parse failed; preserving input SQL (len={len(sql)})")
        return sql
    exp = sqlglot.expressions
    for anon in list(tree.find_all(exp.Anonymous)):
        if str(anon.this).upper() != "DATETRUNC":
            continue
        parts = anon.expressions
        if len(parts) != 2:
            continue
        try:
            e0, e1 = parts[0], parts[1]
            lit0 = isinstance(e0, exp.Literal)
            lit1 = isinstance(e1, exp.Literal)
            if lit0 and not lit1:
                unit_sql = e0.sql(dialect=sqlglot_dialect)
                expr_sql = e1.sql(dialect=sqlglot_dialect)
            elif lit1 and not lit0:
                unit_sql = e1.sql(dialect=sqlglot_dialect)
                expr_sql = e0.sql(dialect=sqlglot_dialect)
            else:
                expr_sql = e0.sql(dialect=sqlglot_dialect)
                unit_sql = e1.sql(dialect=sqlglot_dialect)
            if sqlglot_dialect in ("postgres", "redshift", "duckdb"):
                unit_norm = unit_sql.strip("'\"")
                frag = f"SELECT DATE_TRUNC('{unit_norm.lower()}', {expr_sql})"
            else:
                frag = f"SELECT DATE_TRUNC({unit_sql}, {expr_sql})"
            wrapped = sqlglot.parse_one(frag, dialect=sqlglot_dialect)
            dtn = list(wrapped.find_all(exp.DateTrunc))
            if dtn:
                anon.replace(dtn[0])
                continue
            dtn = list(wrapped.find_all(cast(Any, exp.TimestampTrunc)))
            if dtn:
                anon.replace(dtn[0])
        except Exception:
            continue
    try:
        out = tree.sql(dialect=sqlglot_dialect)
        if sql.strip() and not out.strip():
            debug(f"[normalize_datetrunc_sql] sqlglot emission empty; preserving input SQL (len={len(sql)})")
            return sql
        return normalize_named_placeholders(out)
    except Exception:
        debug(f"[normalize_datetrunc_sql] sqlglot serialize failed; preserving input SQL (len={len(sql)})")
        return sql


def _information_schema_literal_sql(sql_template: str, schema_name: str) -> str:
    """Substitute ``:s`` in an information_schema probe with a quoted schema literal."""
    esc = schema_name.replace("'", "''")
    return sql_template.replace(":s", f"'{esc}'")


def _information_schema_rows_from_connection(
    connection: Any,
    schema_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch normalized information_schema constraint and column rows via a DB-API connection."""
    cursor = connection.cursor()
    try:
        tc_rows = information_schema_connector_fetchall_dict_rows(
            cursor,
            _information_schema_literal_sql(INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL, schema_name),
        )
        kcu_rows = information_schema_connector_fetchall_dict_rows(
            cursor,
            _information_schema_literal_sql(INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL, schema_name),
        )
        rc_rows = information_schema_connector_fetchall_dict_rows(
            cursor,
            _information_schema_literal_sql(INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL, schema_name),
        )
        cols_rows = information_schema_connector_fetchall_dict_rows(
            cursor,
            _information_schema_literal_sql(INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL, schema_name),
        )
        return tc_rows, kcu_rows, rc_rows, cols_rows
    finally:
        cursor.close()


def _information_schema_rows_from_engine(
    engine: Any,
    schema_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch normalized information_schema constraint and column rows via SQLAlchemy."""
    with engine.connect() as conn:
        tc_raw = conn.execute(text(INFORMATION_SCHEMA_TABLE_CONSTRAINTS_SQL), {"s": schema_name}).fetchall()
        kcu_raw = conn.execute(text(INFORMATION_SCHEMA_KEY_COLUMN_USAGE_SQL), {"s": schema_name}).fetchall()
        rc_raw = conn.execute(text(INFORMATION_SCHEMA_REFERENTIAL_CONSTRAINTS_SQL), {"s": schema_name}).fetchall()
        cols_raw = conn.execute(text(INFORMATION_SCHEMA_COLUMNS_DDL_PROBE_SQL), {"s": schema_name}).fetchall()
    tc_cols = ["constraint_schema", "constraint_name", "table_schema", "table_name", "constraint_type"]
    kcu_cols = [
        "constraint_schema",
        "constraint_name",
        "table_schema",
        "table_name",
        "column_name",
        "ordinal_position",
    ]
    rc_cols = ["constraint_schema", "constraint_name", "unique_constraint_schema", "unique_constraint_name"]
    col_cols = ["table_schema", "table_name", "column_name", "ordinal_position", "data_type", "is_nullable"]

    def rows_to_dicts(raw_rows: list[Any], col_names: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in raw_rows:
            out.append(information_schema_normalize_row(dict(zip(col_names, row, strict=True))))
        return out

    return (
        rows_to_dicts(list(tc_raw), tc_cols),
        rows_to_dicts(list(kcu_raw), kcu_cols),
        rows_to_dicts(list(rc_raw), rc_cols),
        rows_to_dicts(list(cols_raw), col_cols),
    )


def structural_constraints_index_for_schema(
    dialect: Dialect,
    schema_name: str,
    *,
    engine: Any | None = None,
    connection: Any | None = None,
) -> CatalogStructuralConstraintsIndex:
    """Load PK, FK, and single-column UNIQUE metadata from ``information_schema`` for *schema_name*."""
    if not schema_name:
        return CatalogStructuralConstraintsIndex.empty()
    try:
        if connection is not None:
            tc_rows, kcu_rows, rc_rows, cols_rows = _information_schema_rows_from_connection(connection, schema_name)
        elif engine is not None:
            tc_rows, kcu_rows, rc_rows, cols_rows = _information_schema_rows_from_engine(engine, schema_name)
        else:
            return CatalogStructuralConstraintsIndex.empty()
        idx = structural_constraints_index_from_information_schema_rows(tc_rows, kcu_rows, rc_rows)
        idx.column_nullability = column_nullability_from_information_schema_rows(cols_rows)
        if dialect.name == "redshift" and engine is not None:
            with engine.connect() as conn:
                svv_raw = conn.execute(text(REDSHIFT_SVV_FOREIGN_KEYS_SQL), {"s": schema_name}).fetchall()
            for row in svv_raw:
                child_table = str(row[2] or "")
                child_col = str(row[3] or "")
                parent_table = str(row[5] or "")
                parent_col = str(row[6] or "")
                if not child_table or not child_col or not parent_table or not parent_col:
                    continue
                ctk = child_table.lower()
                bundle = idx.tables.setdefault(ctk, CatalogTableStructuralConstraints())
                edge = FKEdge(
                    src_table=child_table,
                    src_cols=[child_col],
                    dst_table=parent_table,
                    dst_cols=[parent_col],
                )
                if edge not in bundle.foreign_keys:
                    bundle.foreign_keys.append(edge)
        return idx
    except Exception as exc:
        debug(f"[structural_constraints_index_for_schema] failed for {dialect.name}: {exc!r}")
        return CatalogStructuralConstraintsIndex.empty()


def sqlite_structural_constraints_index(engine: Any) -> CatalogStructuralConstraintsIndex:
    """Build structural constraints from ``PRAGMA foreign_key_list`` and ``PRAGMA table_info``."""
    if engine is None:
        return CatalogStructuralConstraintsIndex.empty()
    tables_out: dict[str, CatalogTableStructuralConstraints] = {}
    nullability: dict[str, dict[str, bool]] = {}
    try:
        with engine.connect() as conn:
            table_rows = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"),
            ).fetchall()
            for (table_name,) in table_rows:
                tkey = str(table_name).lower()
                tables_out.setdefault(tkey, CatalogTableStructuralConstraints())
                info_rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
                nullability[tkey] = {}
                for info in info_rows:
                    cname = str(info[1])
                    nullability[tkey][cname] = int(info[3] or 0) == 0
                fk_rows = conn.execute(text(f'PRAGMA foreign_key_list("{table_name}")')).fetchall()
                for fk in fk_rows:
                    child_col = str(fk[3])
                    parent_table = str(fk[2])
                    parent_col = str(fk[4])
                    edge = FKEdge(
                        src_table=str(table_name),
                        src_cols=[child_col],
                        dst_table=parent_table,
                        dst_cols=[parent_col],
                    )
                    tables_out[tkey].foreign_keys.append(edge)
        return CatalogStructuralConstraintsIndex(tables=tables_out, column_nullability=nullability)
    except Exception as exc:
        debug(f"[sqlite_structural_constraints_index] failed: {exc!r}")
        return CatalogStructuralConstraintsIndex.empty()


def qualify_tables_ast(
    sql: str,
    *,
    sqlglot_dialect: str,
    catalog: str | None,
    schema: str,
    cte_names: set[str],
    backtick: bool,
) -> str:
    """Qualify bare table references with ``schema`` (and optional catalog) using sqlglot AST."""
    if not sql or not sql.strip():
        return sql
    if not schema:
        return sql
    cte_names_lower = {n.lower() for n in cte_names if n}
    skip_lower = {s.lower() for s in QUALIFY_SKIP_IDENTIFIERS}
    try:
        parsed = sqlglot.parse_one(sql, read=sqlglot_dialect)
    except Exception:
        debug(f"[qualify_tables_ast] sqlglot parse failed; preserving input SQL (len={len(sql)})")
        return sql
    for cte in parsed.find_all(sqlglot.exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names_lower.add(alias.lower())
    for table in parsed.find_all(sqlglot.exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        if name in cte_names_lower:
            continue
        if name in skip_lower:
            continue
        if table.args.get("db") or table.args.get("catalog"):
            continue
        table.set("db", sqlglot.exp.to_identifier(schema, quoted=backtick))
        if catalog:
            table.set("catalog", sqlglot.exp.to_identifier(catalog, quoted=backtick))
    try:
        out = parsed.sql(dialect=sqlglot_dialect, identify=backtick)
        if sql.strip() and not out.strip():
            debug(f"[qualify_tables_ast] sqlglot emission empty; preserving input SQL (len={len(sql)})")
            return sql
        return out
    except Exception:
        debug(f"[qualify_tables_ast] sqlglot serialize failed; preserving input SQL (len={len(sql)})")
        return sql


def explain_diag_cartesian_join(message: str, *, node_kind: str | None = None) -> SqlDiagnostic:
    """Build a cartesian-join soft diagnostic from an engine-specific. message."""
    return SqlDiagnostic(
        code=SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN,
        message=message,
        node_kind=node_kind,
    )


def explain_diag_zero_estimate(message: str, *, node_kind: str | None = None) -> SqlDiagnostic:
    """Build a zero-row estimate soft diagnostic."""
    return SqlDiagnostic(
        code=SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE,
        message=message,
        node_kind=node_kind,
    )


def explain_diag_seq_scan_indexed(
    message: str,
    *,
    relation_name: str,
    node_kind: str | None = "Seq Scan",
) -> SqlDiagnostic:
    """Build a sequential-scan-on-indexed-column soft diagnostic."""
    return SqlDiagnostic(
        code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED,
        message=message,
        node_kind=node_kind,
        offending_identifier=relation_name,
    )


def pg_relation_indexed_columns(schema: SchemaGraph | None, relation_name: str) -> set[str]:
    """Return indexed column names for *relation_name* from a schema. graph."""
    if schema is None:
        return set()
    table = schema.tables.get(relation_name) or schema.tables.get(relation_name.lower())
    if table is None:
        for key, meta in schema.tables.items():
            if key.lower() == relation_name.lower():
                table = meta
                break
    if table is None:
        return set()
    indexed: set[str] = set()
    for col_name, col in table.columns.items():
        if col.is_primary_key or col.is_unique:
            indexed.add(col_name.lower())
    for fk in table.foreign_keys:
        for sc in fk.src_cols:
            indexed.add(sc.lower())
    return indexed


def mysql_relation_indexed_columns(schema: SchemaGraph | None, relation_name: str) -> set[str]:
    """Return indexed column names for a MySQL relation from schema. graph metadata."""
    indexed = pg_relation_indexed_columns(schema, relation_name)
    if schema is None:
        return indexed
    table = schema.tables.get(relation_name) or schema.tables.get(relation_name.lower())
    if table is None:
        for key, meta in schema.tables.items():
            if key.lower() == relation_name.lower():
                table = meta
                break
    if table is None:
        return indexed
    for col_name in getattr(table, "indexed_columns", []) or []:
        indexed.add(str(col_name).lower())
    return indexed


def pg_walk_explain_plan(node: dict[str, Any], schema: SchemaGraph | None) -> list[SqlDiagnostic]:
    """Recursively walk a PostgreSQL ``EXPLAIN (FORMAT JSON)`` plan. node. and emit soft diagnostics."""
    diags: list[SqlDiagnostic] = []
    node_type = str(node.get("Node Type", ""))
    if node_type in PG_JOIN_NODE_TYPES:
        has_join_cond = any(k in node for k in PG_JOIN_CONDITION_KEYS)
        inner_plans = node.get("Plans", []) or []
        inner_has_cond = any(any(k in p for k in PG_INNER_CONDITION_KEYS) for p in inner_plans)
        if not has_join_cond and not inner_has_cond:
            diags.append(
                explain_diag_cartesian_join(
                    f"{node_type} without join condition",
                    node_kind=node_type,
                )
            )
    plan_rows = node.get("Plan Rows")
    if isinstance(plan_rows, (int, float)) and plan_rows == 0:
        diags.append(
            explain_diag_zero_estimate(
                "planner estimates zero rows",
                node_kind=node_type or None,
            )
        )
    if node_type == "Seq Scan":
        relation_name = str(node.get("Relation Name", ""))
        filter_text = str(node.get("Filter", ""))
        if filter_text and relation_name:
            indexed = pg_relation_indexed_columns(schema, relation_name)
            if indexed and any(col in filter_text for col in indexed):
                diags.append(
                    explain_diag_seq_scan_indexed(
                        f"sequential scan on {relation_name} filters indexed column",
                        relation_name=relation_name,
                    )
                )
    for child in node.get("Plans", []) or []:
        diags.extend(pg_walk_explain_plan(child, schema))
    return diags


def pg_diagnostics_from_explain_json(raw: Any, schema: SchemaGraph | None) -> list[SqlDiagnostic]:
    """Parse a PostgreSQL ``EXPLAIN (FORMAT JSON)`` row payload into. soft diagnostics."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        payload = raw
    if not isinstance(payload, list) or not payload:
        return []
    head = payload[0]
    if not isinstance(head, dict):
        return []
    plan = head.get("Plan")
    if not isinstance(plan, dict):
        return []
    return pg_walk_explain_plan(plan, schema)


def pg_root_plan_estimates(plan: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return coarse ``(plan_rows, estimated_bytes)`` from a PostgreSQL. JSON plan root."""
    rows_v = plan.get("Plan Rows")
    width_v = plan.get("Plan Width")
    pr = float(rows_v) if isinstance(rows_v, (int, float)) else None
    pw = float(width_v) if isinstance(width_v, (int, float)) else None
    if pr is None:
        return None, None
    est_bytes = pr * pw if pw is not None else None
    return pr, est_bytes


def databricks_diagnostics_from_explain_text(text_payload: str) -> list[SqlDiagnostic]:
    """Scan a Spark/Databricks ``EXPLAIN`` text payload for soft plan- shape findings."""
    if not text_payload:
        return []
    diags: list[SqlDiagnostic] = []
    if any(tok in text_payload for tok in DBR_CARTESIAN_TOKENS):
        diags.append(
            explain_diag_cartesian_join("Spark plan contains an unconditioned join"),
        )
    if DBR_ZERO_ROW_RE.search(text_payload):
        diags.append(explain_diag_zero_estimate("Spark plan estimates zero rows"))
    return diags


def databricks_plan_stats_from_explain_text(
    text_payload: str,
) -> tuple[float | None, float | None]:
    """Extract coarse row and byte estimates from Spark/Databricks. ``EXPLAIN COST`` text."""
    if not text_payload:
        return None, None
    row_est: float | None = None
    for pat in (
        r"(?i)Statistics\s*\([^)]*rowCount\s*=\s*(\d+)",
        r"(?i)rowCount[=:\s]+(\d+)",
        r"(?i)numRows[=:\s]+(\d+)",
    ):
        match = re.search(pat, text_payload)
        if match:
            try:
                row_est = float(match.group(1))
                break
            except (TypeError, ValueError):
                continue
    byte_est: float | None = None
    match_sz = re.search(
        r"(?i)sizeInBytes\s*=\s*([\d.]+)\s*([KMGT]?iB|[KMGT]?B|bytes?)",
        text_payload,
    )
    if match_sz:
        try:
            val = float(match_sz.group(1))
            unit = (match_sz.group(2) or "b").lower().replace("bytes", "b").replace("byte", "b")
            mult = {
                "b": 1.0,
                "ib": 1.0,
                "kib": 1024.0,
                "mib": 1024.0**2,
                "gib": 1024.0**3,
                "tib": 1024.0**4,
                "kb": 1000.0,
                "mb": 1_000_000.0,
                "gb": 1e9,
                "tb": 1e12,
            }
            byte_est = val * mult.get(unit, 1.0)
        except (TypeError, ValueError, IndexError):
            byte_est = None
    return row_est, byte_est


def mysql_diagnostics_from_explain_json(
    raw: Any,
    schema: SchemaGraph | None = None,
) -> list[SqlDiagnostic]:
    """Parse MySQL ``EXPLAIN FORMAT=JSON`` output for soft diagnostics."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        payload = raw
    if not isinstance(payload, dict):
        return []
    query_block = payload.get("query_block") or payload
    if not isinstance(query_block, dict):
        return []
    diags: list[SqlDiagnostic] = []
    table = query_block.get("table") or {}
    if isinstance(table, dict):
        if table.get("access_type") == "ALL":
            table_name = str(table.get("table_name") or "")
            if table_name:
                diags.append(
                    explain_diag_seq_scan_indexed(
                        f"full table scan on {table_name}",
                        relation_name=table_name,
                        node_kind="ALL",
                    )
                )
                attached = str(table.get("attached_condition") or "")
                if attached and schema is not None:
                    indexed = mysql_relation_indexed_columns(schema, table_name)
                    if indexed and any(col in attached.lower() for col in indexed):
                        diags.append(
                            explain_diag_seq_scan_indexed(
                                f"full table scan on {table_name} filters indexed column",
                                relation_name=table_name,
                                node_kind="ALL",
                            )
                        )
        if table.get("using_filesort"):
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.EXPLAIN_OTHER,
                    message="MySQL plan uses filesort",
                    node_kind="filesort",
                )
            )
        if table.get("using_temporary_table"):
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.EXPLAIN_OTHER,
                    message="MySQL plan uses temporary table",
                    node_kind="temporary_table",
                )
            )
    nested_loop = query_block.get("nested_loop")
    if isinstance(nested_loop, list):
        for join in nested_loop:
            if not isinstance(join, dict):
                continue
            join_table = join.get("table") or {}
            if isinstance(join_table, dict) and join_table.get("access_type") == "ALL":
                table_name = str(join_table.get("table_name") or "")
                if table_name:
                    diags.append(
                        explain_diag_seq_scan_indexed(
                            f"full table scan on {table_name}",
                            relation_name=table_name,
                            node_kind="ALL",
                        )
                    )
    cost_info = query_block.get("cost_info") or {}
    if isinstance(cost_info, dict):
        rows_examined = cost_info.get("query_cost")
        if rows_examined == 0:
            diags.append(explain_diag_zero_estimate("MySQL plan estimates zero cost"))
    return diags


def mysql_root_plan_estimates(raw: Any) -> tuple[float | None, float | None]:
    """Return coarse row estimate from MySQL ``EXPLAIN FORMAT=JSON`` output."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return None, None
    else:
        payload = raw
    if not isinstance(payload, dict):
        return None, None
    query_block = payload.get("query_block") or payload
    if not isinstance(query_block, dict):
        return None, None
    table = query_block.get("table") or {}
    if isinstance(table, dict):
        rows = table.get("rows_examined_per_scan") or table.get("rows")
        if isinstance(rows, (int, float)):
            return float(rows), None
    return None, None


def redshift_diagnostics_from_explain_text(text_payload: str) -> list[SqlDiagnostic]:
    """Scan Redshift EXPLAIN text for soft plan-shape findings."""
    if not text_payload:
        return []
    diags: list[SqlDiagnostic] = []
    if "XN Nested Loop" in text_payload and "Join Filter:" not in text_payload:
        diags.append(explain_diag_cartesian_join("Redshift nested loop without join filter"))
    for dist_token in ("DS_BCAST_INNER", "DS_DIST_BOTH", "DS_DIST_INNER"):
        if dist_token in text_payload:
            diags.append(
                explain_diag_cartesian_join(
                    f"Redshift plan contains {dist_token} distribution risk",
                    node_kind=dist_token,
                )
            )
    if re.search(r"(?i)\brows=0\b", text_payload):
        diags.append(explain_diag_zero_estimate("Redshift plan estimates zero rows"))
    return diags


def duckdb_diagnostics_from_explain_text(plan_text: str) -> list[SqlDiagnostic]:
    """Return soft diagnostics parsed from a DuckDB EXPLAIN plan string."""
    diags: list[SqlDiagnostic] = []
    upper = (plan_text or "").upper()
    if any(token in upper for token in DUCKDB_EXPLAIN_CARTESIAN_TOKENS):
        diags.append(explain_diag_cartesian_join("DuckDB plan contains a cross product / nested-loop join"))
    return diags


def duckdb_root_plan_estimates(plan_text: str) -> tuple[float | None, float | None]:
    """Return (estimated_rows, estimated_bytes) from a DuckDB EXPLAIN plan string."""
    if not plan_text:
        return None, None
    match = re.search(DUCKDB_EXPLAIN_ESTIMATED_CARDINALITY_RE, plan_text)
    if not match:
        return None, None
    try:
        return float(match.group(1)), None
    except (TypeError, ValueError):
        return None, None


def sqlite_diagnostics_from_query_plan(plan_text: str) -> list[SqlDiagnostic]:
    """Return soft diagnostics parsed from a SQLite EXPLAIN QUERY PLAN string."""
    diags: list[SqlDiagnostic] = []
    upper = (plan_text or "").upper()
    if any(token in upper for token in SQLITE_EXPLAIN_FULL_SCAN_TOKENS):
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED,
                message="SQLite query plan performs a full table scan",
                node_kind=None,
            )
        )
    return diags


def bigquery_diagnostics_from_dry_run(
    total_bytes_processed: float,
    *,
    referenced_tables: list[str] | None = None,
    partition_filter_present: bool = True,
    require_partition_filter_tables: list[str] | None = None,
) -> list[SqlDiagnostic]:
    """Build soft diagnostics from a BigQuery dry-run job."""
    diags: list[SqlDiagnostic] = []
    cap = PolicyConfig.MAX_QUERY_COST_BYTES
    if cost_cap_active(cap) and cap is not None and total_bytes_processed > float(cap) * 0.5:
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED,
                message=f"BigQuery dry-run would scan {int(total_bytes_processed)} bytes",
            )
        )
    missing = list(require_partition_filter_tables or [])
    if missing and not partition_filter_present:
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED,
                message=f"BigQuery query may scan partitioned table(s) without filter: {', '.join(missing)}",
            )
        )
    _ = referenced_tables
    return diags


def redshift_root_plan_estimates(
    text_payload: str,
) -> tuple[float | None, float | None]:
    """Extract coarse row estimate from Redshift EXPLAIN text."""
    if not text_payload:
        return None, None
    row_est: float | None = None
    match_rows = re.search(r"(?i)\brows=(\d+(?:\.\d+)?)", text_payload)
    if match_rows:
        try:
            row_est = float(match_rows.group(1))
        except (TypeError, ValueError):
            row_est = None
    byte_est: float | None = None
    match_width = re.search(r"(?i)\bwidth=(\d+(?:\.\d+)?)", text_payload)
    if match_width and row_est is not None:
        try:
            byte_est = row_est * float(match_width.group(1))
        except (TypeError, ValueError):
            byte_est = None
    return row_est, byte_est


def sqlserver_diagnostics_from_showplan_xml(text_payload: str) -> list[SqlDiagnostic]:
    """Parse SQL Server ``SHOWPLAN_XML`` text for soft diagnostics."""
    if not text_payload or not text_payload.strip():
        return []
    diags: list[SqlDiagnostic] = []
    if "MissingIndex" in text_payload:
        diags.append(
            SqlDiagnostic(
                code=SqlDiagnosticCode.EXPLAIN_OTHER,
                message="SQL Server plan reports missing index recommendation",
                node_kind="MissingIndex",
            )
        )
    if re.search(r'PhysicalOp="Table Scan"', text_payload, re.IGNORECASE):
        diags.append(
            explain_diag_seq_scan_indexed(
                "SQL Server table scan in SHOWPLAN_XML",
                relation_name="",
                node_kind="Table Scan",
            )
        )
    if re.search(r'PhysicalOp="Clustered Index Scan"', text_payload, re.IGNORECASE):
        if "Seek Predicates" not in text_payload and "Predicate" not in text_payload:
            diags.append(
                explain_diag_seq_scan_indexed(
                    "SQL Server clustered index scan without seek predicate",
                    relation_name="",
                    node_kind="Clustered Index Scan",
                )
            )
    return diags


def sqlserver_diagnostics_from_showplan_rows(rows: list[tuple[Any, ...]]) -> list[SqlDiagnostic]:
    """Parse SQL Server ``SHOWPLAN_ALL`` rows for soft diagnostics."""
    if not rows:
        return []
    diags: list[SqlDiagnostic] = []
    estimate_rows: float | None = None
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        joined = " | ".join(cells)
        if "Nested Loops" in joined and "Seek Predicates" not in joined and "Predicates" not in joined:
            diags.append(explain_diag_cartesian_join("SQL Server nested loop without predicate"))
        for cell in cells:
            if cell.startswith("EstimateRows"):
                try:
                    estimate_rows = float(cell.split("=", 1)[1].strip())
                except (TypeError, ValueError, IndexError):
                    continue
        if estimate_rows == 0.0:
            diags.append(explain_diag_zero_estimate("SQL Server plan estimates zero rows"))
    return diags


def sqlserver_root_plan_estimates(rows: list[tuple[Any, ...]]) -> tuple[float | None, float | None]:
    """Extract coarse row estimate from SQL Server ``SHOWPLAN_ALL`` rows."""
    for row in rows:
        for cell in row:
            text = str(cell or "")
            if text.startswith("EstimateRows"):
                try:
                    return float(text.split("=", 1)[1].strip()), None
                except (TypeError, ValueError, IndexError):
                    continue
    return None, None


def snowflake_diagnostics_from_explain_json(raw: Any) -> list[SqlDiagnostic]:
    """Parse Snowflake ``EXPLAIN USING JSON`` output for soft. diagnostics."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        payload = raw
    if not isinstance(payload, list) or not payload:
        return []
    diags: list[SqlDiagnostic] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        op = str(item.get("operation") or "")
        if "CartesianJoin" in op:
            diags.append(explain_diag_cartesian_join("Snowflake CartesianJoin operation"))
        partitions = item.get("partitionsAssigned")
        if partitions == 0:
            diags.append(explain_diag_zero_estimate("Snowflake plan assigns zero partitions"))
    return diags


def snowflake_root_plan_estimates(raw: Any) -> tuple[float | None, float | None]:
    """Return coarse row and byte estimates from Snowflake ``EXPLAIN. USING JSON`` output."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return None, None
    else:
        payload = raw
    if not isinstance(payload, list) or not payload:
        return None, None
    row_est: float | None = None
    byte_est: float | None = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows = item.get("rows")
        if isinstance(rows, (int, float)):
            row_est = float(rows)
        bytes_val = item.get("bytesAssigned") or item.get("statistics", {}).get("bytesAssigned")
        if isinstance(bytes_val, (int, float)):
            byte_est = float(bytes_val)
    return row_est, byte_est


@dataclass(frozen=True)
class PartitionSqlAdapter:
    """Dialect-specific SQL formatting hooks for partition predicate. injection."""

    quote_table_column: Callable[[str, str], str]
    format_literal: Callable[[Any], str]
    sqlglot_dialect: str


def partition_get_column_ref(expr: NormalizedExpr) -> tuple[str | None, str | None]:
    """Extract table and column names from a normalized expression. primary term."""
    term = (expr.primary_term or "").strip()
    if not term:
        return None, None
    if "." in term:
        parts = term.rsplit(".", 1)
        return parts[0].strip() or None, parts[1].strip() or None
    return None, term


def partition_format_predicate(
    adapter: PartitionSqlAdapter,
    table: str,
    col: str,
    fp: FilterParam,
    params: dict[str, Any],
) -> str | None:
    """Format a single partition predicate using *adapter* quoting and. literal rules."""
    qual = adapter.quote_table_column(table, col)
    val = fp.param_key and params.get(fp.param_key)
    if val is None and fp.raw_value is not None:
        val = fp.raw_value
    if fp.op == "=":
        if val is None:
            return None
        return f"{qual} = {adapter.format_literal(val)}"
    if fp.op in (">=", "<=", ">", "<"):
        if val is None:
            return None
        return f"{qual} {fp.op} {adapter.format_literal(val)}"
    if fp.op == "in":
        if val is None:
            return None
        if isinstance(val, list):
            parts = [adapter.format_literal(v) for v in val]
            return f"{qual} IN ({', '.join(parts)})"
        return f"{qual} IN ({adapter.format_literal(val)})"
    return None


def partition_format_grouped_predicate(
    adapter: PartitionSqlAdapter,
    table: str,
    col: str,
    fps: list[FilterParam],
    params: dict[str, Any],
) -> str | None:
    """Format grouped partition predicates as ``IN`` or bounded range. SQL."""
    qual = adapter.quote_table_column(table, col)
    ops = {fp.op for fp in fps}
    if ops <= {"="} and len(fps) > 1:
        parts = []
        for fp in fps:
            val = fp.param_key and params.get(fp.param_key) or fp.raw_value
            if val is not None:
                parts.append(adapter.format_literal(val))
        if parts:
            return f"{qual} IN ({', '.join(parts)})"
        return None
    if ops <= {">=", "<="} and len(fps) == 2:
        ge = next((f for f in fps if f.op == ">="), None)
        le = next((f for f in fps if f.op == "<="), None)
        if ge and le:
            v1 = ge.param_key and params.get(ge.param_key) or ge.raw_value
            v2 = le.param_key and params.get(le.param_key) or le.raw_value
            if v1 is not None and v2 is not None:
                return f"({qual} >= {adapter.format_literal(v1)} AND {qual} <= {adapter.format_literal(v2)})"
        return None
    if len(fps) == 1:
        return partition_format_predicate(adapter, table, col, fps[0], params)
    return None


def partition_contains_filter_param_keys(intent: RuntimeIntent) -> set[str]:
    """Collect ``param_key`` values from ``contains`` filters in main. and CTE intents."""
    keys: set[str] = set()
    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            if fp.op == "contains" and fp.param_key:
                keys.add(fp.param_key)
    for fp in intent.filters_param or []:
        if fp.op == "contains" and fp.param_key:
            keys.add(fp.param_key)
    return keys


def partition_flatten_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """Merge CTE and main params and normalize values used by. ``contains`` filters."""
    merged: dict[str, Any] = {}
    for cte in intent.cte_steps or []:
        merged.update(cte.param_values or {})
    merged.update(intent.param_values or {})
    contains_keys = partition_contains_filter_param_keys(intent)
    if not contains_keys:
        return merged
    out = dict(merged)
    for key in contains_keys:
        if key in out:
            out[key] = normalize_array_contains_param_value(out[key])
    return out


def partition_build_predicates(
    adapter: PartitionSqlAdapter,
    schema: SchemaGraph,
    intent: RuntimeIntent,
    params: dict[str, Any],
    *,
    column_selector: Callable[[Any], list[str]] | None = None,
) -> list[str]:
    """Build partition predicates from intent filters and schema. metadata."""
    tables = intent.tables or []
    filters = intent.filters_param or []
    if not tables or not filters:
        return []

    def default_selector(table_meta: Any) -> list[str]:
        return list(getattr(table_meta, "partition_columns", []) or [])

    selector = column_selector or default_selector
    grouped: dict[tuple[str, str], list[FilterParam]] = {}

    for table_name in tables:
        table_meta = schema.tables.get(table_name)
        if not table_meta:
            continue
        part_cols = selector(table_meta)
        if not part_cols:
            continue
        part_cols_lower = {c.lower(): c for c in part_cols}
        for fp in filters:
            table_part, col_part = partition_get_column_ref(fp.left_expr)
            if not col_part:
                continue
            col_lower = col_part.lower() if col_part else ""
            if col_lower not in part_cols_lower:
                continue
            actual_col = part_cols_lower[col_lower]
            table_for_pred = table_part or (tables[0] if tables else "")
            if table_part and table_part.lower() not in {t.lower() for t in tables}:
                continue
            key = (table_for_pred.lower(), actual_col.lower())
            grouped.setdefault(key, []).append(fp)

    result: list[str] = []
    for (table_key, col_key), fps in grouped.items():
        table_name = next((t for t in tables if t.lower() == table_key), tables[0] if tables else "")
        table_meta = schema.tables.get(table_name)
        part_cols = selector(table_meta) if table_meta else []
        col_name = next((c for c in part_cols if c.lower() == col_key), col_key)
        pred = partition_format_grouped_predicate(adapter, table_name, col_name, fps, params)
        if pred:
            result.append(pred)
    return result


def partition_predicate_already_in_sql(sql: str, predicates: list[str]) -> bool:
    """Return True when every partition predicate already appears in. the. SQL text."""
    sql_norm = sql.replace(" ", "").replace("\n", " ").lower()
    for pred in predicates:
        pred_norm = pred.replace(" ", "").lower()
        if pred_norm not in sql_norm:
            return False
    return True


def partition_append_where_via_ast(
    adapter: PartitionSqlAdapter,
    sql: str,
    predicate: str,
) -> str | None:
    """Append *predicate* to the WHERE clause using a sqlglot AST round- trip."""
    try:
        tree = sqlglot.parse_one(sql, read=adapter.sqlglot_dialect)
    except Exception:
        return None
    if not isinstance(tree, sqlglot.exp.Select):
        return None
    try:
        updated = tree.where(predicate, append=True, dialect=adapter.sqlglot_dialect)
    except Exception:
        return None
    return updated.sql(dialect=adapter.sqlglot_dialect)


def partition_append_to_where(adapter: PartitionSqlAdapter, sql: str, predicate: str) -> str:
    """Append *predicate* to the SQL WHERE clause via sqlglot; raise on. failure."""
    out = partition_append_where_via_ast(adapter, sql, predicate)
    if out is None:
        raise ValueError("AST refused to append WHERE predicate; SQL is unparseable")
    return out


def inject_partition_predicates(
    adapter: PartitionSqlAdapter,
    sql: str,
    schema: SchemaGraph,
    intent: RuntimeIntent,
    *,
    column_selector: Callable[[Any], list[str]] | None = None,
) -> str:
    """Append missing partition predicates for pruning when absent from. *sql*."""
    params = partition_flatten_param_values(intent)
    predicates = partition_build_predicates(
        adapter,
        schema,
        intent,
        params,
        column_selector=column_selector,
    )
    if not predicates:
        return sql
    if partition_predicate_already_in_sql(sql, predicates):
        return sql
    combined = " AND ".join(predicates)
    return partition_append_to_where(adapter, sql, combined)


def table_requires_pruning_filter(table_meta: Any) -> bool:
    """Return True when catalog metadata marks a table as requiring a pruning filter."""
    return bool(getattr(table_meta, "require_partition_filter", False))


def append_required_partition_filter_guard(
    sql: str,
    *,
    schema: SchemaGraph,
    intent: RuntimeIntent,
    sqlglot_dialect: str,
    column_selector: Callable[[Any], list[str]],
    default_predicate_sql: Callable[[str, str], str],
    intent_equality_for_column: Callable[[RuntimeIntent, str, str], str | None] | None = None,
) -> str:
    """Append mandatory partition-filter guard predicates when tables require them."""
    tables = intent.tables or []
    predicates: list[str] = []
    equality_fn = intent_equality_for_column or (lambda _i, _t, _c: None)
    for table_name in tables:
        meta = schema.tables.get(table_name)
        if meta is None or not table_requires_pruning_filter(meta):
            continue
        part_cols = column_selector(meta)
        if not part_cols:
            continue
        part_col = part_cols[0]
        date_pred = equality_fn(intent, table_name, part_col)
        if date_pred:
            predicates.append(date_pred)
        else:
            predicates.append(default_predicate_sql(table_name, part_col))
    if not predicates:
        return sql
    combined = " AND ".join(predicates)
    try:
        tree = sqlglot.parse_one(sql, dialect=sqlglot_dialect)
    except Exception:
        return sql
    if not isinstance(tree, sqlglot.exp.Select):
        return sql
    try:
        updated = tree.where(combined, append=True, dialect=sqlglot_dialect)
    except Exception:
        return sql
    return updated.sql(dialect=sqlglot_dialect)


def information_schema_normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return *row* with lowercased string keys for stable Unity. Catalog. driver column naming."""
    return {str(k).lower(): v for k, v in row.items()}


def information_schema_connector_fetchall_dict_rows(cursor: Any, sql: str) -> list[dict[str, Any]]:
    """Execute *sql* on *cursor* and return lower-keyed row dicts."""
    cursor.execute(sql)
    if not cursor.description:
        return []
    col_names = [d[0] for d in cursor.description]
    return [
        information_schema_normalize_row(dict(zip(col_names, row, strict=True))) for row in (cursor.fetchall() or [])
    ]


def information_schema_spark_collect_normalized_dicts(spark: Any, sql: str) -> list[dict[str, Any]]:
    """Execute *sql* on *spark* and return lower-keyed row dicts."""
    rows: list[dict[str, Any]] = []
    for r in spark.sql(sql).collect():
        d = r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)
        rows.append(information_schema_normalize_row(d))
    return rows


def information_schema_key_column_lists(
    kcu_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    """Group ``key_column_usage`` rows into ordered column-name lists. keyed by constraint identity."""
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for r in kcu_rows:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        pos = int(r.get("ordinal_position") or 0)
        cname = str(r.get("column_name") or "")
        buckets.setdefault((cs, cn), []).append((pos, cname))
    out: dict[tuple[str, str], list[str]] = {}
    for key, pairs in buckets.items():
        pairs.sort(key=lambda x: x[0])
        out[key] = [p[1] for p in pairs if p[1]]
    return out


def information_schema_trailing_relation_name(ref: str) -> str:
    """Return the trailing SQL identifier segment from a possibly. qualified ``catalog.schema.table`` reference."""
    s = str(ref or "").strip()
    if not s:
        return ""
    parts = re.split(r"\s*\.\s*", s)
    tokens: list[str] = []
    for part in parts:
        t = part.strip().strip("`").strip('"').strip()
        if t:
            tokens.append(t)
    return tokens[-1] if tokens else ""


def structural_constraints_index_from_information_schema_rows(
    tc_rows: list[dict[str, Any]],
    kcu_rows: list[dict[str, Any]],
    rc_rows: list[dict[str, Any]],
) -> CatalogStructuralConstraintsIndex:
    """Join normalized Unity ``information_schema`` constraint rows. into. a constraints index."""
    tc_norm = [information_schema_normalize_row(dict(r)) for r in tc_rows]
    kcu_norm = [information_schema_normalize_row(dict(r)) for r in kcu_rows]
    rc_norm = [information_schema_normalize_row(dict(r)) for r in rc_rows]
    kcu_cols = information_schema_key_column_lists(kcu_norm)
    tc_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in tc_norm:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        tc_by_key[(cs, cn)] = r
    rc_by_fk: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rc_norm:
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        rc_by_fk[(cs, cn)] = r
    tables_out: dict[str, CatalogTableStructuralConstraints] = {}

    def bundle_for(tk: str) -> CatalogTableStructuralConstraints:
        if tk not in tables_out:
            tables_out[tk] = CatalogTableStructuralConstraints()
        return tables_out[tk]

    def append_pk(tk: str, cols: list[str]) -> None:
        b = bundle_for(tk)
        seen = set(b.primary_keys)
        for c in cols:
            if c and c not in seen:
                seen.add(c)
                b.primary_keys.append(c)

    for r in tc_norm:
        ctype = str(r.get("constraint_type") or "").strip().upper()
        cs = str(r.get("constraint_schema") or "")
        cn = str(r.get("constraint_name") or "")
        tname = str(r.get("table_name") or "")
        tk = tname.lower()
        cols = kcu_cols.get((cs, cn), [])
        if ctype == "PRIMARY KEY":
            append_pk(tk, cols)
        elif ctype == "UNIQUE" and len(cols) == 1:
            b = bundle_for(tk)
            uq = cols[0]
            if uq and uq not in b.unique_columns:
                b.unique_columns.append(uq)
        elif ctype == "FOREIGN KEY":
            rc = rc_by_fk.get((cs, cn))
            if not rc:
                continue
            ucs = str(rc.get("unique_constraint_schema") or "")
            ucn = str(rc.get("unique_constraint_name") or "")
            parent_tc = tc_by_key.get((ucs, ucn))
            if not parent_tc:
                continue
            parent_table = str(parent_tc.get("table_name") or "")
            parent_cols = kcu_cols.get((ucs, ucn), [])
            child_table = str(r.get("table_name") or "")
            child_cols = kcu_cols.get((cs, cn), [])
            if not child_cols or not parent_cols or len(child_cols) != len(parent_cols):
                continue
            dst_simple = information_schema_trailing_relation_name(parent_table)
            ctk = child_table.lower()
            edge = FKEdge(
                src_table=child_table,
                src_cols=list(child_cols),
                dst_table=dst_simple,
                dst_cols=list(parent_cols),
            )
            bundle_for(ctk).foreign_keys.append(edge)

    return CatalogStructuralConstraintsIndex(tables=tables_out)


def column_nullability_from_information_schema_rows(
    col_rows: list[dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    """Build per-table column nullability maps from ``information_schema.columns`` rows."""
    out: dict[str, dict[str, bool]] = {}
    for r in col_rows:
        norm = information_schema_normalize_row(dict(r))
        tname = str(norm.get("table_name") or "").lower()
        cname = str(norm.get("column_name") or "")
        if not tname or not cname:
            continue
        nullable = str(norm.get("is_nullable") or "").upper() == "YES"
        out.setdefault(tname, {})[cname] = nullable
    return out


class ResultBackend(ABC):
    """Abstract row-fetch backend for dialect execution paths. Concrete backends wrap SQLAlchemy engines, native drivers, Spark sessions, or accelerator clients."""

    kind: ResultReaderKind = "sqlalchemy"

    @abstractmethod
    def fetch_rows(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_ms: int | None = None,
    ) -> list[tuple[Any, ...]]:
        """Execute *sql* and return result rows as tuples."""

    def fetch_first_column_text(self, sql: str, params: dict[str, Any] | None = None) -> str:
        """Execute *sql* and join the first column of each row into. newline-separated text."""
        rows = self.fetch_rows(sql, params)
        return chr(10).join(str(r[0]) for r in rows if r and r[0] is not None)


class SqlAlchemyResultBackend(ResultBackend):
    """SQLAlchemy engine backend used by sqlglot engines and Postgres. Applies optional statement timeouts and maps permission/timeout errors to contract exceptions."""

    kind = "sqlalchemy"

    def __init__(
        self,
        engine: Any,
        *,
        dialect_name: str = "",
        timeout_sql_provider: Any | None = None,
    ) -> None:
        """Wrap a SQLAlchemy engine for row fetch operations."""
        self._engine = engine
        self._dialect_name = dialect_name
        self._timeout_sql_provider = timeout_sql_provider

    def fetch_rows(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_ms: int | None = None,
    ) -> list[tuple[Any, ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._dialect_name}.execute] sql=" + chr(10) + f"{sql}")
        try:
            exec_params = reconcile_execute_bind_params(sql, params) or {}
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                ms = int(timeout_ms)
                timeout_sql = self._timeout_sql_provider(ms) if self._timeout_sql_provider is not None else None
                with self._engine.begin() as conn:
                    if timeout_sql:
                        conn.execute(text(timeout_sql))
                    rows = conn.execute(text(sql), exec_params).fetchall()
            else:
                with self._engine.connect() as conn:
                    rows = conn.execute(text(sql), exec_params).fetchall()
            return [tuple(r) for r in rows]
        except KeyError as e:
            raise ValueError(f"unbound_placeholder: {e.args[0]}") from e
        except Exception as e:
            err = str(e)
            if is_permission_denied_error(err):
                raise AccessError("execute", err) from e
            el = err.lower()
            if "timeout" in el and ("statement" in el or "cancel" in el or "deadline" in el):
                raise StatementTimeoutError(err) from e
            raise


class SqlServerResultBackend(SqlAlchemyResultBackend):
    """SQL Server backend applying statement timeouts via the ODBC driver command timeout."""

    def fetch_rows(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_ms: int | None = None,
    ) -> list[tuple[Any, ...]]:
        if diagnostic_debug_enabled():
            debug(f"[{self._dialect_name}.execute] sql=" + chr(10) + f"{sql}")
        try:
            if timeout_ms is not None and cost_cap_active(timeout_ms):
                secs = max(1, int(timeout_ms) // 1000)
                with self._engine.connect() as conn:
                    raw = conn.connection
                    if raw is not None and hasattr(raw, "timeout"):
                        raw.timeout = secs
                    rows = conn.execute(text(sql), params or {}).fetchall()
                return [tuple(r) for r in rows]
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params or {}).fetchall()
            return [tuple(r) for r in rows]
        except Exception as e:
            err = str(e)
            if is_permission_denied_error(err):
                raise AccessError("execute", err) from e
            el = err.lower()
            if "timeout" in el and ("statement" in el or "cancel" in el or "deadline" in el):
                raise StatementTimeoutError(err) from e
            raise


register_profile_schema_native_dispatch(profile_schema_native_dispatch)
