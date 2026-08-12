"""PostgreSQL dialect implementation using pglast and SQLAlchemy."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal, cast

from sqlalchemy import create_engine, text

from ._config import EngineConfig, EngineLimits, EngineRuntimeConfig, PostgresRuntimeConfig
from ._constants import (
    DOLLAR_PLACEHOLDER_RE,
    NAMED_PLACEHOLDER_RE,
    PG_AGG_FUNCNAMES,
    SQL_STRING_LITERAL_CONTROL_CHAR_RE,
    STRUCTURAL_CODE_TO_DIAG,
    UNSAFE_PARAM_LITERAL,
)
from ._contracts_base import (
    EngineContext,
    JoinEdge,
    SchemaInclude,
    SqlDiagnostic,
    SqlDiagnosticCode,
    TableKind,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import ColumnMetadata, SchemaGraph
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._dialect_sqlglot_helper import (
    PartitionSqlAdapter,
    SqlAlchemyResultBackend,
    SqlglotEngineDialect,
    SqlglotParseMixin,
)
from ._schema_profile import profile_schema
from ._schema_reflect import load_or_create_schema_postgresql
from ._utils import (
    canonicalize_sql,
    debug,
    effective_explain_timeout_ms,
    effective_statement_timeout_ms,
    refuse_unsafe_sql_string_literal_content,
    require_driver,
    sha256,
    sqlalchemy_pool_kwargs_from_limits,
)

_PgNodeType: Any
try:
    from pglast.ast import Node as _PgNodeImported

    _PgNodeType = _PgNodeImported
except ImportError:
    _PgNodeType = ()


@dataclass(frozen=True)
class PostgresQueryLogSource:
    @staticmethod
    def _stable_sql_text_for_history(sql_text: str) -> str:
        s = re.sub(r"\b\d+\.\d+\b", "<num>", sql_text)
        s = re.sub(r"\b\d+\b", "<num>", s)
        s = re.sub(r"'(?:[^']|'')*'", "<str>", s)
        return s

    def is_available(self, conn: Any) -> bool:
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements' LIMIT 1")
            row = cur.fetchone()
            try:
                cur.close()
            except (OSError, AttributeError, TypeError):
                pass
            return row is not None
        except Exception:
            return False

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        del lookback_days
        try:
            cur = conn.cursor()
        except Exception:
            return []
        parts = [
            "SELECT query",
            "FROM pg_stat_statements",
            "WHERE calls >= %s",
        ]
        bind: list[Any] = [int(min_runs)]
        if user_filter:
            parts.append("AND userid = (SELECT oid FROM pg_roles WHERE rolname = %s LIMIT 1)")
            bind.append(str(user_filter))
        parts.append("ORDER BY calls DESC NULLS LAST")
        parts.append("LIMIT %s")
        bind.append(int(max_queries))
        stmt = " ".join(parts)
        try:
            cur.execute(stmt, tuple(bind))
            rows = cur.fetchall() or []
        except Exception:
            try:
                cur.close()
            except (OSError, AttributeError, TypeError):
                pass
            return []
        try:
            cur.close()
        except (OSError, AttributeError, TypeError):
            pass
        out: list[str] = []
        for row in rows:
            if not row:
                continue
            raw_q = row[0]
            if raw_q is None:
                continue
            out.append(self._stable_sql_text_for_history(str(raw_q)))
        return out


class _PgLastRuntime:
    """Lazy-loaded pglast bundle for PostgreSQL-only code paths."""

    __slots__ = ("parse_sql", "ast", "join_type", "a_expr_kind", "bool_expr_type", "raw_stream_cls")

    def __init__(self) -> None:
        import pglast
        from pglast.enums import A_Expr_Kind, BoolExprType, JoinType
        from pglast.stream import RawStream

        self.parse_sql = pglast.parse_sql
        self.ast = pglast.ast
        self.join_type = JoinType
        self.a_expr_kind = A_Expr_Kind
        self.bool_expr_type = BoolExprType
        self.raw_stream_cls = RawStream


_pgl_runtime: _PgLastRuntime | None = None


class _PgParsedSelect:
    """Container for a pglast-parsed ``SELECT`` plus its named- placeholder round-trip map."""

    __slots__ = ("root", "name_to_index", "index_to_name")

    def __init__(self, root: Any, name_to_index: dict[str, int], index_to_name: dict[int, str]) -> None:
        self.root = root
        self.name_to_index = name_to_index
        self.index_to_name = index_to_name


class PostgresDialect(Dialect):
    """PostgreSQL implementation using pglast and SQLAlchemy."""

    name: str = "postgresql"
    sqlglot_dialect: ClassVar[str] = "postgres"
    registry_canonical_rank: ClassVar[int] = 6
    registry_qualified_table_ref: ClassVar[bool] = True

    @staticmethod
    def pg_walk_nodes(root: Any) -> Iterator[Any]:
        stack: list[Any] = [root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            yield node
            if isinstance(node, _PgNodeType):
                for attr in node:
                    if attr == "ancestors":
                        continue
                    val = getattr(node, attr, None)
                    if val is None:
                        continue
                    if isinstance(val, list | tuple):
                        for item in val:
                            if isinstance(item, _PgNodeType):
                                stack.append(item)
                    elif isinstance(val, _PgNodeType):
                        stack.append(val)
                continue
            try:
                attrs = vars(node)
            except TypeError:
                continue
            for value in attrs.values():
                if isinstance(value, list | tuple):
                    for item in value:
                        if hasattr(item, "__class__") and not isinstance(item, str | int | float | bool | bytes):
                            stack.append(item)
                elif hasattr(value, "__class__") and not isinstance(value, str | int | float | bool | bytes):
                    stack.append(value)

    @staticmethod
    def pg_node_kind(node: Any) -> str:
        return getattr(node, "__class__", type("x", (), {})).__name__

    @staticmethod
    def pg_columnref_to_pair(node: Any) -> tuple[str | None, str] | None:
        fields = getattr(node, "fields", None) or ()
        parts: list[str] = []
        for fld in fields:
            kind = PostgresDialect.pg_node_kind(fld)
            if kind == "String":
                sval = getattr(fld, "sval", None) or getattr(fld, "str", None)
                if isinstance(sval, str):
                    parts.append(sval)
                    continue
                return None
            if kind == "A_Star":
                return None
            return None
        if not parts:
            return None
        if len(parts) == 1:
            return None, parts[0]
        return parts[-2], parts[-1]

    @staticmethod
    def pg_funcname(node: Any) -> str:
        fn = getattr(node, "funcname", None) or ()
        parts: list[str] = []
        for f in fn:
            sval = getattr(f, "sval", None) or getattr(f, "str", None)
            if isinstance(sval, str):
                parts.append(sval)
        return ".".join(parts).lower() if parts else ""

    @staticmethod
    def _pg_plan_rows_from_explain_payload(payload: Any) -> float | None:
        if payload is None:
            return None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return None
        if isinstance(payload, list) and payload:
            root = payload[0]
            if isinstance(root, dict):
                plan = root.get("Plan")
                if isinstance(plan, dict):
                    pr = plan.get("Plan Rows")
                    if isinstance(pr, (int, float)):
                        return float(pr)
        return None

    @staticmethod
    def require_pglast() -> _PgLastRuntime:
        global _pgl_runtime
        if _pgl_runtime is None:
            try:
                _pgl_runtime = _PgLastRuntime()
            except ImportError as exc:
                raise ImportError("PostgresDialect requires the 'pglast' package (optional postgresql extra).") from exc
        return _pgl_runtime

    @staticmethod
    def pg_encode_named_placeholders(sql: str) -> tuple[str, dict[str, int], dict[int, str]]:
        """Replace ``:name`` placeholders with ``$N`` so pglast can parse the SQL."""
        name_to_index: dict[str, int] = {}
        index_to_name: dict[int, str] = {}

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in name_to_index:
                idx = len(name_to_index) + 1
                name_to_index[name] = idx
                index_to_name[idx] = name
            return f"${name_to_index[name]}"

        encoded = NAMED_PLACEHOLDER_RE.sub(repl, sql)
        return encoded, name_to_index, index_to_name

    @staticmethod
    def _pg_decode_dollar_placeholders(sql: str, index_to_name: dict[int, str]) -> str:
        """Restore original ``:name`` placeholders from pglast-emitted ``$N`` markers."""
        if not index_to_name:
            return sql

        def repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            name = index_to_name.get(idx)
            return f":{name}" if name is not None else match.group(0)

        return DOLLAR_PLACEHOLDER_RE.sub(repl, sql)

    @staticmethod
    def append_pglast_select_targets(root: Any, expr_sqls: Sequence[str]) -> bool:
        """Append SELECT-list target entries onto a pglast ``SelectStmt`` root."""
        p = PostgresDialect.require_pglast()
        targets = list(getattr(root, "targetList", None) or ())
        for expr_sql in expr_sqls:
            encoded, _, _ = PostgresDialect.pg_encode_named_placeholders(expr_sql)
            try:
                probe = p.parse_sql(f"SELECT {encoded}")
            except Exception:
                return False
            if not probe:
                return False
            probe_select = getattr(probe[0], "stmt", None)
            if probe_select is None or type(probe_select).__name__ != "SelectStmt":
                return False
            tlist = getattr(probe_select, "targetList", None) or ()
            if len(tlist) != 1:
                return False
            first_target = cast(Any, tlist)[0]
            targets.append(first_target)
        try:
            root.targetList = tuple(targets)
        except Exception:
            return False
        return True

    @property
    def supports_ilike(self) -> bool:
        """Return True because PostgreSQL exposes ``ILIKE``."""
        return True

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return PostgreSQL's portable binary collation for reproducible ordering."""
        return "C"

    @property
    def supports_unnest_select_item(self) -> bool:
        """Return True because PostgreSQL ``UNNEST`` is a set-returning function valid in SELECT."""
        return True

    @property
    def integer_division_truncates(self) -> bool:
        """Return True because PostgreSQL integer division truncates."""
        return True

    def escape_string_literal(self, value: str) -> str:
        """Escape a PostgreSQL string literal body, refusing embedded control characters."""
        s = str(value)
        refuse_unsafe_sql_string_literal_content(s)
        if SQL_STRING_LITERAL_CONTROL_CHAR_RE.search(s):
            raise ValueError(UNSAFE_PARAM_LITERAL)
        if "\\" in s:
            return s.replace("\\", "\\\\").replace("'", "''")
        return s.replace("'", "''")

    def quote_string_literal(self, text: str) -> str:
        """Render a PostgreSQL string literal, using E-strings when backslashes are present."""
        s = str(text)
        body = self.escape_string_literal(s)
        if "\\" in s:
            return f"E'{body}'"
        return f"'{body}'"

    @property
    def parse_backend(self) -> Literal["pglast", "sqlglot"]:
        """Return ``pglast`` as the SQL-to-intent parse backend."""
        return "pglast"

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return PostgreSQL current timestamp or date for inclusive window upper bounds."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return PostgreSQL ``SET LOCAL statement_timeout`` for profiling sessions."""
        return f"SET LOCAL statement_timeout = {int(timeout_ms)}"

    def session_timezone_sql(self) -> str | None:
        """Return PostgreSQL session time-zone lookup SQL."""
        return "SELECT current_setting('TIMEZONE')"

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append declarative partition predicates when schema and intent are available."""
        if schema is None or intent is None:
            return sql
        adapter = PartitionSqlAdapter(
            quote_table_column=self.quote_table_column,
            format_literal=self.quote_string_literal,
            sqlglot_dialect="postgres",
        )
        return adapter.inject_partition_predicates(sql, schema, intent)

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Rewrite ``DATETRUNC(expr, unit)`` to PostgreSQL ``DATE_TRUNC(unit, expr)``."""
        if stage != "post_substitute":
            return sql
        return SqlglotParseMixin.normalize_datetrunc_sql(sql, sqlglot_dialect=self.sqlglot_dialect)

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: Any | None = None
    ) -> float | None:
        """Return PostgreSQL planner row estimate from ``EXPLAIN (FORMAT JSON)``."""
        eng = getattr(self, "engine", None)
        if eng is None:
            return None
        try:
            finalized = self.finalize_render(sql_text, {}, schema=schema, intent=intent)
            explain_sql = f"EXPLAIN (FORMAT JSON, COSTS true) {finalized}"
            with eng.connect() as conn:
                rows = conn.execute(text(explain_sql), {}).fetchall()
            payload = rows[0][0] if rows else None
            return self._pg_plan_rows_from_explain_payload(payload)
        except Exception:
            return None

    def query_log_source(self) -> Any | None:
        """Return the PostgreSQL ``pg_stat_statements`` query-log source."""
        return PostgresQueryLogSource()

    def __init__(
        self,
        config: EngineRuntimeConfig,
        sqlalchemy_engine: Any | None = None,
        *,
        limits: EngineLimits | None = None,
    ):
        """Create a SQLAlchemy engine from `PostgresRuntimeConfig`. When ``SCHEMA`` is set, session ``search_path`` is pinned so unqualified identifiers resolve into that schema for reflection, profiling, and execution."""
        require_driver("postgresql")
        try:
            self.require_pglast()
        except ImportError as e:
            raise ImportError("PostgresDialect requires the 'pglast' package (optional postgresql extra).") from e
        super().__init__(config)
        pg_config = cast(PostgresRuntimeConfig, config)
        pool_kwargs = sqlalchemy_pool_kwargs_from_limits(limits or EngineLimits())
        if sqlalchemy_engine is not None:
            self.engine = sqlalchemy_engine
        else:
            connect_args: dict[str, Any] = {}
            schema = str(pg_config.SCHEMA or "").strip()
            if schema and all(ch.isalnum() or ch == "_" for ch in schema):
                connect_args["options"] = f"-csearch_path={schema},public"
            self.engine = create_engine(
                pg_config.db_url(),
                future=True,
                connect_args=connect_args,
                **pool_kwargs,
            )
        self._result_backend = SqlAlchemyResultBackend(
            self.engine,
            dialect_name="postgresql",
            timeout_sql_provider=self.profile_statement_timeout_sql,
        )

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a schema-qualified table reference when ``SCHEMA`` is configured."""
        _ = kind
        schema = self.schema_name()
        if schema:
            return f"{self.quote_identifier(schema)}.{self.quote_identifier(table)}"
        return self.quote_identifier(table)

    @property
    def result_backend(self) -> SqlAlchemyResultBackend:
        return self._result_backend

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True because PostgreSQL exposes ``pg_cancel_backend``."""
        return True

    @property
    def logical_engine_name(self) -> str:
        return "PostgreSQL"

    def _strip_schema(self, ident: str) -> str:
        """Strip schema prefix from an identifier and return a lowercase table name."""
        s = (ident or "").strip().lower()
        if "." in s:
            s = s.split(".")[-1]
        return s

    def _collect_from_items(
        self, fr: Any, scalar_cte_names: frozenset[str] | None = None
    ) -> tuple[bool, dict[str, str], bool, bool, bool, bool, bool]:
        """Collect FROM-clause aliases and flags for unsupported join shapes."""
        alias_to_table: dict[str, str] = {}
        has_subquery = False
        has_using = False
        has_cross_join = False
        has_self_join = False
        seen_tables: set[str] = set()
        ok = True

        def add_alias(relname: str, alias: Any) -> None:
            nonlocal alias_to_table, has_self_join, seen_tables
            t = self._strip_schema(relname)
            if t in seen_tables:
                has_self_join = True
            seen_tables.add(t)
            if alias is None:
                alias_to_table[t] = t
                return
            an = getattr(alias, "aliasname", None)
            if isinstance(an, str) and an:
                alias_to_table[self._strip_schema(an)] = t
            alias_to_table[t] = t

        def walk(item: Any) -> bool:
            nonlocal has_subquery, has_using, has_cross_join, ok
            if item is None:
                ok = False
                return False
            tag = getattr(item, "__class__", type("x", (), {})).__name__
            if tag == "RangeVar":
                add_alias(getattr(item, "relname", "") or "", getattr(item, "alias", None))
                return True
            if tag == "JoinExpr":
                if getattr(item, "usingClause", None) is not None or getattr(item, "isNatural", False):
                    has_using = True
                join_type = getattr(item, "jointype", None)
                if join_type is not None and str(join_type) == "JoinType.JOIN_INNER":
                    quals = getattr(item, "quals", None)
                    if quals is None:
                        allow = False
                        if scalar_cte_names:
                            rarg = getattr(item, "rarg", None)
                            rtag = getattr(rarg, "__class__", type("x", (), {})).__name__ if rarg is not None else ""
                            if rtag == "RangeVar" and rarg is not None:
                                reln = (getattr(rarg, "relname", "") or "").lower()
                                if reln and reln in scalar_cte_names:
                                    allow = True
                        if not allow:
                            has_cross_join = True
                if not walk(getattr(item, "larg", None)):
                    return False
                if not walk(getattr(item, "rarg", None)):
                    return False
                return True
            if tag in {
                "RangeSubselect",
                "RangeFunction",
                "RangeTableFunc",
                "RangeTableSample",
            }:
                has_subquery = True
                ok = False
                return False
            ok = False
            return False

        if fr is None:
            return False, {}, False, False, False, False, False
        for it in fr if isinstance(fr, list | tuple) else [fr]:
            if not walk(it):
                ok = False
                break
            return (ok, alias_to_table, has_subquery, has_using, has_cross_join, has_self_join, True)
        return (ok, alias_to_table, has_subquery, has_using, has_cross_join, has_self_join, True)

    def _validate_cte_bodies(self, with_clause: Any) -> tuple[bool, str]:
        """Validate CTE bodies against structural restrictions. Forbids recursive CTEs, subqueries, EXISTS sublinks, and set operations inside any CTE body. Window functions and ``CASE`` expressions are allowed."""
        if with_clause is None:
            return True, ""

        if getattr(with_clause, "recursive", False):
            return False, "cte_recursive"

        ctes = getattr(with_clause, "ctes", [])
        if not ctes:
            return True, ""

        for cte in ctes:
            cte_query = getattr(cte, "ctequery", None)
            if cte_query is None:
                return False, "cte_malformed"

            def walk_cte(n: Any) -> str | None:
                tag = getattr(n, "__class__", type("x", (), {})).__name__
                if tag in {"RangeSubselect", "SubLink"}:
                    if tag == "SubLink":
                        sublink_type = getattr(n, "subLinkType", None)
                        if sublink_type is not None and sublink_type == 0:
                            return "cte_contains_exists"
                    return "cte_contains_subquery"
                if tag == "SetOperationStmt":
                    return "cte_contains_set_op"
                try:
                    attrs = vars(n)
                except TypeError:
                    return None
                for attr in attrs.values():
                    if isinstance(attr, list):
                        for x in attr:
                            if hasattr(x, "__class__"):
                                err = walk_cte(x)
                                if err:
                                    return err
                    elif hasattr(attr, "__class__"):
                        err = walk_cte(attr)
                        if err:
                            return err
                return None

            err = walk_cte(cte_query)
            if err:
                return False, err

        return True, ""

    def _ast_structural_valid(self, sql: str, scalar_cte_names: frozenset[str] | None = None) -> tuple[bool, str]:
        """Validate SQL structure using the pglast AST. Checks that the SQL is a single SELECT statement free of subqueries in ``FROM``, CROSS JOINs, self-joins, USING clauses, EXISTS sublinks, LATERAL, and set operations. Window functions and ``CASE`` expressions are allowed. Also validates any CTE bodies with the same rules."""
        try:
            p = self.require_pglast()
            encoded, _, _ = self.pg_encode_named_placeholders(canonicalize_sql(sql))
            stmts = p.parse_sql(encoded)
        except Exception:
            return False, "ast_parse_failed"

        if not stmts or len(stmts) != 1:
            return False, "multiple_statements"

        root = getattr(stmts[0], "stmt", None)
        if root is None:
            return False, "no_root"

        if getattr(root, "__class__", type("x", (), {})).__name__ != "SelectStmt":
            return False, "not_select"

        with_clause = getattr(root, "withClause", None)
        has_cte = with_clause is not None

        if has_cte:
            ok, err = self._validate_cte_bodies(with_clause)
            if not ok:
                return False, err

        fr = getattr(root, "fromClause", None)
        if fr is not None:
            _, _, has_subq, has_using, has_cross, has_self, _ = self._collect_from_items(fr, scalar_cte_names)
            if has_subq:
                return False, "subquery_not_allowed"
            if has_using:
                return False, "using_not_allowed"
            if has_cross:
                return False, "cross_join_not_allowed"
            if has_self:
                return False, "self_join_not_allowed"

        has_exists = False
        has_lateral = False

        def walk(n: Any) -> bool:
            nonlocal has_exists, has_lateral
            tag = getattr(n, "__class__", type("x", (), {})).__name__
            if tag in {
                "RangeSubselect",
                "SubLink",
                "SetOperationStmt",
            }:
                if tag == "SubLink":
                    sublink_type = getattr(n, "subLinkType", None)
                    if sublink_type is not None and sublink_type == 0:
                        has_exists = True
                return False
            if tag == "RangeFunction":
                is_lateral = getattr(n, "lateral", False)
                if is_lateral:
                    has_lateral = True
                    return False

            try:
                attrs = vars(n)
            except TypeError:
                return True

            for attr in attrs.values():
                if isinstance(attr, list):
                    for x in attr:
                        if hasattr(x, "__class__") and not walk(x):
                            return False
                elif hasattr(attr, "__class__"):
                    if not walk(attr):
                        return False
            return True

        if not walk(root):
            if has_exists:
                return False, "exists_not_allowed"
            if has_lateral:
                return False, "lateral_not_allowed"
            return False, "forbidden_structure"

        return True, ""

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """Validate SQL via pglast structurally and (when *schema* is given) semantically."""
        ok, code = self._ast_structural_valid(sql, scalar_cte_names=scalar_cte_names)
        if not ok:
            mapped = SqlDiagnosticCode(STRUCTURAL_CODE_TO_DIAG.get(code, SqlDiagnosticCode.FORBIDDEN_STRUCTURE.value))
            return [SqlDiagnostic(code=mapped, message=code, node_kind=None)]
        diags: list[SqlDiagnostic] = []
        try:
            p = self.require_pglast()
            encoded, _, _ = self.pg_encode_named_placeholders(canonicalize_sql(sql))
            stmts = p.parse_sql(encoded)
        except Exception:
            return [SqlDiagnostic(code=SqlDiagnosticCode.AST_PARSE_FAILED, message="parse failed")]
        if not stmts:
            return diags
        root = getattr(stmts[0], "stmt", None)
        if root is None:
            return diags
        cte_names = self._pg_collect_cte_names(root)
        alias_to_table = self._pg_collect_table_aliases(root)
        if schema is not None:
            refs = self._pg_collect_column_refs(root)
            diags += Dialect.check_schema_references_shared(refs, alias_to_table, cte_names, schema)
        diags += self._pg_check_grouping(root)
        diags += self._pg_check_cte_closure(root, cte_names)
        if declared_params is not None:
            diags += self._pg_check_param_coverage(sql, declared_params)
        return diags

    def _pg_collect_cte_names(self, root: Any) -> set[str]:
        """Return the set of lowercased CTE names defined on *root*'s ``WITH`` clause."""
        names: set[str] = set()
        with_clause = getattr(root, "withClause", None)
        if with_clause is None:
            return names
        for cte in getattr(with_clause, "ctes", None) or ():
            ctename = getattr(cte, "ctename", None)
            if isinstance(ctename, str) and ctename:
                names.add(ctename.lower())
        return names

    def _pg_collect_table_aliases(self, root: Any) -> dict[str, str]:
        """Return ``{alias_or_table_lc: real_table_lc}`` for every ``RangeVar`` reachable from *root*."""
        out: dict[str, str] = {}
        for node in self.pg_walk_nodes(root):
            if self.pg_node_kind(node) != "RangeVar":
                continue
            relname = getattr(node, "relname", None) or ""
            if not relname:
                continue
            real = relname.lower()
            out[real] = real
            alias = getattr(node, "alias", None)
            if alias is not None:
                aliasname = getattr(alias, "aliasname", None)
                if isinstance(aliasname, str) and aliasname:
                    out[aliasname.lower()] = real
        return out

    def _pg_collect_column_refs(self, root: Any) -> list[tuple[str | None, str]]:
        """Return ``(prefix, column)`` pairs for every ``ColumnRef`` reachable from *root*."""
        out: list[tuple[str | None, str]] = []
        for node in self.pg_walk_nodes(root):
            if self.pg_node_kind(node) != "ColumnRef":
                continue
            pair = self.pg_columnref_to_pair(node)
            if pair is not None:
                out.append(pair)
        return out

    def _pg_check_grouping(self, root: Any) -> list[SqlDiagnostic]:
        """Emit grain diagnostics for *root*: aggregates in WHERE and HAVING-without-GROUP-BY. Skips the non-grouped-select-col check; renderer grain enforcement is the authoritative path."""
        diags: list[SqlDiagnostic] = []
        where = getattr(root, "whereClause", None)
        if where is not None:
            for node in self.pg_walk_nodes(where):
                if self.pg_node_kind(node) == "FuncCall":
                    name = self.pg_funcname(node)
                    if name in PG_AGG_FUNCNAMES:
                        diags.append(
                            SqlDiagnostic(
                                code=SqlDiagnosticCode.AGG_IN_WHERE,
                                message=f"aggregate {name!r} in WHERE",
                                node_kind="FuncCall",
                                offending_identifier=name,
                            )
                        )
                        break
        having = getattr(root, "havingClause", None)
        group = getattr(root, "groupClause", None) or ()
        if having is not None and not group:
            diags.append(
                SqlDiagnostic(
                    code=SqlDiagnosticCode.HAVING_WITHOUT_GROUP,
                    message="HAVING without GROUP BY",
                    node_kind="SelectStmt",
                )
            )
        return diags

    def _pg_check_cte_closure(self, root: Any, cte_names: set[str]) -> list[SqlDiagnostic]:
        """Flag CTE names that are defined but never referenced by a ``RangeVar`` outside their own definition."""
        if not cte_names:
            return []
        referenced: set[str] = set()
        with_clause = getattr(root, "withClause", None)
        defining_queries: set[int] = set()
        if with_clause is not None:
            for cte in getattr(with_clause, "ctes", None) or ():
                inner = getattr(cte, "ctequery", None)
                if inner is not None:
                    defining_queries.add(id(inner))
        for node in self.pg_walk_nodes(root):
            if self.pg_node_kind(node) != "RangeVar":
                continue
            relname = getattr(node, "relname", None)
            if isinstance(relname, str) and relname.lower() in cte_names:
                referenced.add(relname.lower())
        unreferenced = sorted(cte_names - referenced)
        return [
            SqlDiagnostic(
                code=SqlDiagnosticCode.CTE_UNREFERENCED,
                message=f"CTE {n!r} is defined but never referenced",
                node_kind="CommonTableExpr",
                offending_identifier=n,
            )
            for n in unreferenced
        ]

    def _pg_check_param_coverage(self, sql: str, declared: set[str]) -> list[SqlDiagnostic]:
        """Emit a diagnostic for each ``:name`` placeholder in *sql* not present in *declared*."""
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
                        node_kind="ParamRef",
                        offending_identifier=name,
                    )
                )
        return diags

    def parse_select(self, sql: str) -> _PgParsedSelect | None:
        """Parse *sql* with pglast after encoding ``:name`` placeholders as ``$N``. Single-line ``--`` comments are stripped before whitespace canonicalization so they do not absorb subsequent clauses when newlines collapse. Returns ``None`` for non-``SELECT`` roots, multi-statement input, or parse failure."""
        decommented = re.sub(r"--[^\n]*", "", sql)
        encoded, name_to_index, index_to_name = self.pg_encode_named_placeholders(canonicalize_sql(decommented))
        try:
            p = self.require_pglast()
            stmts = p.parse_sql(encoded)
        except Exception:
            return None
        if not stmts or len(stmts) != 1:
            return None
        root = getattr(stmts[0], "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return None
        return _PgParsedSelect(stmts[0], name_to_index, index_to_name)

    def ordered_join_carrier_froms(self, parsed: _PgParsedSelect) -> list[Any] | None:
        """Return the inner-CTE ``SelectStmt`` nodes (left-to-right) followed by the outer ``SelectStmt``. Each handle is a ``SelectStmt`` whose ``fromClause`` is rewritten by :meth:`attach_joins`."""
        root = getattr(parsed.root, "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return None
        carriers: list[Any] = []
        with_clause = getattr(root, "withClause", None)
        if with_clause is not None:
            ctes = getattr(with_clause, "ctes", None) or ()
            for cte in ctes:
                inner = getattr(cte, "ctequery", None)
                if inner is not None and type(inner).__name__ == "SelectStmt":
                    if getattr(inner, "fromClause", None) is not None:
                        carriers.append(inner)
        if getattr(root, "fromClause", None) is not None:
            carriers.append(root)
        return carriers

    def from_anchor_of(self, carrier: Any) -> str | None:
        """Read the bare table name of *carrier*'s leftmost ``FROM`` leaf. When the first ``FROM`` element is a ``JoinExpr`` tree (for example after ``CROSS JOIN`` attachment), this walks ``larg`` until a ``RangeVar`` is reached. Returns ``None`` for subqueries or empty ``FROM``."""
        from_clause: Sequence[Any] = getattr(carrier, "fromClause", None) or ()
        if len(from_clause) != 1:
            return None
        first = from_clause[0]
        while type(first).__name__ == "JoinExpr":
            first = getattr(first, "larg", None)
            if first is None:
                return None
        if type(first).__name__ != "RangeVar":
            return None
        relname = getattr(first, "relname", None)
        if not relname:
            return None
        return str(relname).lower()

    def attach_joins(self, parsed: _PgParsedSelect, from_handle: Any, edges: list[JoinEdge]) -> bool:
        """Build a left-deep tree of pglast ``JoinExpr`` nodes from *edges* and replace *from_handle*'s ``fromClause`` with the resulting single-element list."""
        if not edges:
            return False
        from_clause: Sequence[Any] = getattr(from_handle, "fromClause", None) or ()
        if len(from_clause) != 1:
            return False
        p = self.require_pglast()
        current: Any = from_clause[0]
        for edge in edges:
            quals = self._pg_build_on_quals(edge.on_terms)
            if quals is None:
                return False
            rarg = p.ast.RangeVar(relname=edge.table, inh=True, relpersistence="p")
            if edge.alias:
                rarg.alias = p.ast.Alias(aliasname=edge.alias)
            jt = p.join_type.JOIN_INNER if edge.kind == "INNER" else p.join_type.JOIN_LEFT
            current = p.ast.JoinExpr(jointype=jt, isNatural=False, larg=current, rarg=rarg, quals=quals)
        try:
            from_handle.fromClause = (current,)
        except Exception:
            return False
        return True

    @staticmethod
    def _pg_build_on_quals(on_terms: tuple[tuple[str, str, str, str], ...]) -> Any | None:
        """Return a single ``A_Expr`` or an ``AND``-joined ``BoolExpr`` over equality conjuncts."""
        if not on_terms:
            return None
        p = PostgresDialect.require_pglast()
        eqs: list[Any] = []
        for left_token, left_col, right_token, right_col in on_terms:
            lhs = p.ast.ColumnRef(fields=(p.ast.String(sval=left_token), p.ast.String(sval=left_col)))
            rhs = p.ast.ColumnRef(fields=(p.ast.String(sval=right_token), p.ast.String(sval=right_col)))
            eqs.append(p.ast.A_Expr(kind=p.a_expr_kind.AEXPR_OP, name=(p.ast.String(sval="="),), lexpr=lhs, rexpr=rhs))
        if len(eqs) == 1:
            return eqs[0]
        return p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=tuple(eqs))

    def attach_extra_from_and_where(
        self, parsed: _PgParsedSelect, from_handle: Any, extra_from_tables: list[str], where_edges: list[JoinEdge]
    ) -> bool:
        """Append RangeVar entries to ``fromClause`` and AND equality predicates into ``whereClause``."""
        if not extra_from_tables and not where_edges:
            return True
        p = self.require_pglast()
        existing_from = list(getattr(from_handle, "fromClause", None) or ())
        for tbl in extra_from_tables:
            existing_from.append(p.ast.RangeVar(relname=tbl, inh=True, relpersistence="p"))
        try:
            from_handle.fromClause = tuple(existing_from)
        except Exception:
            return False
        if not where_edges:
            return True
        new_eqs: list[Any] = []
        for edge in where_edges:
            for left_token, left_col, right_token, right_col in edge.on_terms:
                lhs = p.ast.ColumnRef(fields=(p.ast.String(sval=left_token), p.ast.String(sval=left_col)))
                rhs = p.ast.ColumnRef(fields=(p.ast.String(sval=right_token), p.ast.String(sval=right_col)))
                new_eqs.append(
                    p.ast.A_Expr(kind=p.a_expr_kind.AEXPR_OP, name=(p.ast.String(sval="="),), lexpr=lhs, rexpr=rhs)
                )
        if not new_eqs:
            return True
        if len(new_eqs) == 1:
            new_pred: Any = new_eqs[0]
        else:
            new_pred = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=tuple(new_eqs))
        existing_where = getattr(from_handle, "whereClause", None)
        if existing_where is None:
            merged: Any = new_pred
        elif (
            type(existing_where).__name__ == "BoolExpr"
            and getattr(existing_where, "boolop", None) == p.bool_expr_type.AND_EXPR
        ):
            merged_args = tuple(getattr(existing_where, "args", ()) or ()) + tuple(new_eqs)
            merged = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=merged_args)
        else:
            merged = p.ast.BoolExpr(
                boolop=p.bool_expr_type.AND_EXPR,
                args=((existing_where, new_pred) if len(new_eqs) == 1 else (existing_where, *new_eqs)),
            )
        try:
            from_handle.whereClause = merged
        except Exception:
            return False
        return True

    def attach_where_sql_fragments(self, from_handle: Any, fragments: list[str]) -> bool:
        """AND-inject raw SQL predicate fragments into the carrier ``WHERE`` clause."""
        if not fragments:
            return True
        p = self.require_pglast()
        new_eqs: list[Any] = []
        for frag in fragments:
            try:
                parsed = p.parse_sql(f"SELECT 1 WHERE {frag}")
            except Exception:
                return False
            stmt = parsed[0].stmt if parsed else None
            if stmt is None or type(stmt).__name__ != "SelectStmt":
                return False
            where_clause = getattr(stmt, "whereClause", None)
            if where_clause is None:
                return False
            new_eqs.append(where_clause)
        if not new_eqs:
            return True
        if len(new_eqs) == 1:
            new_pred: Any = new_eqs[0]
        else:
            new_pred = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=tuple(new_eqs))
        existing_where = getattr(from_handle, "whereClause", None)
        if existing_where is None:
            merged: Any = new_pred
        elif (
            type(existing_where).__name__ == "BoolExpr"
            and getattr(existing_where, "boolop", None) == p.bool_expr_type.AND_EXPR
        ):
            merged_args = tuple(getattr(existing_where, "args", ()) or ()) + tuple(new_eqs)
            merged = p.ast.BoolExpr(boolop=p.bool_expr_type.AND_EXPR, args=merged_args)
        else:
            merged = p.ast.BoolExpr(
                boolop=p.bool_expr_type.AND_EXPR,
                args=((existing_where, new_pred) if len(new_eqs) == 1 else (existing_where, *new_eqs)),
            )
        try:
            from_handle.whereClause = merged
        except Exception:
            return False
        return True

    def replace_projection(self, parsed: _PgParsedSelect, items: list[tuple[str, str | None]]) -> bool:
        """Replace the outer ``SelectStmt``'s ``targetList`` with ``ResTarget`` nodes parsed from *items*."""
        root = getattr(parsed.root, "stmt", None)
        if root is None or type(root).__name__ != "SelectStmt":
            return False
        p = self.require_pglast()
        new_targets: list[Any] = []
        for expr_sql, alias in items:
            encoded, _, _ = self.pg_encode_named_placeholders(expr_sql)
            try:
                probe = p.parse_sql(f"SELECT {encoded}")
            except Exception:
                return False
            if not probe:
                return False
            probe_select = getattr(probe[0], "stmt", None)
            if probe_select is None or type(probe_select).__name__ != "SelectStmt":
                return False
            tlist: Sequence[Any] = getattr(probe_select, "targetList", None) or ()
            if len(tlist) != 1:
                return False
            value_node = tlist[0].val
            new_targets.append(p.ast.ResTarget(name=alias or None, val=value_node))
        try:
            root.targetList = tuple(new_targets)
        except Exception:
            return False
        return True

    def emit_sql(self, parsed: _PgParsedSelect) -> str:
        """Render *parsed* via pglast ``RawStream`` and decode ``$N`` back to ``:name``."""
        p = self.require_pglast()
        raw_stream_ctor: Any = p.raw_stream_cls
        stream_factory: Any = raw_stream_ctor()
        rendered = cast(str, stream_factory(parsed.root))
        return self._pg_decode_dollar_placeholders(rendered, parsed.index_to_name)

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run PostgreSQL ``EXPLAIN (FORMAT JSON, COSTS true)`` and return ``(ok, diagnostics, raw_message)``. ``ok`` is False only on hard validation failures (parse errors, unknown identifiers, timeouts). Permission-denied disables EXPLAIN for the remainder of this dialect instance and is reported as ``ok=True`` with no diagnostics so the caller can proceed without treating missing privileges as invalid SQL. Soft plan-shape findings (suspected cartesian joins, zero-row estimates, sequential scans on indexed columns) are emitted as :class:`SqlDiagnostic` entries with codes from ``SOFT_DIAGNOSTIC_CODES`` in ``_config`` so callers may apply confidence penalties without rejecting the SQL."""
        finalized = self.finalize_render(sql, params or {}, schema=schema, intent=intent)
        explain_sql = f"EXPLAIN (FORMAT JSON, COSTS true) {finalized}"
        try:
            tm = effective_explain_timeout_ms()
            if tm is not None:
                ms = int(tm)
                with self.engine.begin() as conn:
                    conn.execute(text(f"SET LOCAL statement_timeout = {ms}"))
                    rows = conn.execute(text(explain_sql), params or {}).fetchall()
            else:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(explain_sql), params or {}).fetchall()
            payload: Any = None
            if rows:
                first_row = rows[0]
                payload = first_row[0] if len(first_row) > 0 else None
            pay = payload
            if isinstance(pay, str):
                try:
                    pay = json.loads(pay)
                except (ValueError, TypeError):
                    pay = None
            est_rows: float | None = None
            est_bytes: float | None = None
            if isinstance(pay, list) and pay and isinstance(pay[0], dict):
                rp = pay[0].get("Plan")
                if isinstance(rp, dict):
                    est_rows, est_bytes = SqlglotEngineDialect.pg_root_plan_estimates(rp)
            failed, why = Dialect.explain_cost_gate_violation(est_rows, est_bytes, dialect=self)
            if failed:
                return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_COST_EXCEEDED, message=why)], why)
            soft_diags = SqlglotEngineDialect.pg_diagnostics_from_explain_json(payload, schema)
            return True, soft_diags, ""
        except Exception as e:
            err = str(e)
            if self._disable_explain_on_permission_denied(err):
                return True, [], ""
            return (False, [SqlDiagnostic(code=SqlDiagnosticCode.EXPLAIN_OTHER, message=err)], err)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL via SQLAlchemy and return row tuples."""
        tm = effective_statement_timeout_ms()
        return self._result_backend.fetch_rows(sql, params, timeout_ms=tm)

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render PostgreSQL interval date-difference comparison."""
        scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
        sql = f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"
        return Dialect.emit_via_ast(sql, "postgres")

    def render_containment(self, column_sql: str, value_param: str, value_type: str) -> str | None:
        """Render PostgreSQL JSON-array containment with the ``@>`` operator."""
        _ = value_type
        return f"({column_sql})::jsonb @> jsonb_build_array(CAST({value_param} AS TEXT))"

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: ColumnMetadata | None = None,
        value_type: str = "string",
    ) -> str:
        """Render PostgreSQL array membership as a single ``ANY``-comparison predicate. Avoids ``EXISTS`` / subquery / ``ARRAY[`` constructs so the fragment passes ``_enforce_select_only`` and ``_ast_structural_valid``. Lowercases both sides and trims surrounding whitespace and quote characters from the bound value for case-insensitive, quote-tolerant matching against ``text[]`` columns."""
        kind = SqlglotParseMixin.array_storage_kind(column_meta) if column_meta is not None else "native_array"
        if kind == "json_text_array":
            return SqlglotParseMixin.emit_json_containment_predicate(
                self,
                column_sql=column_sql,
                param_key=param_key,
                value_type=value_type,
                sqlglot_dialect="postgres",
            )
        delimiter = "CHR(31)"
        lowered_elements = f"string_to_array(LOWER(array_to_string({column_sql}, {delimiter})), {delimiter})"
        norm_param = f"LOWER(BTRIM(CAST(:{param_key} AS TEXT), ' ' || CHR(34) || CHR(39)))"
        sql = f"{norm_param} = ANY({lowered_elements})"
        return Dialect.emit_via_ast(sql, "postgres")

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render PostgreSQL ``UNNEST`` for SELECT list."""
        sql = f"UNNEST({column_sql}) AS {alias}"
        return Dialect.emit_via_ast(sql, "postgres")

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render PostgreSQL date window boundaries."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            sql = f"{column} {op} DATE_TRUNC('{unit}', {clock})"
        else:
            scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
            sql = f"{column} {op} {clock} - INTERVAL '{scaled} {plural_unit}'"
        return Dialect.emit_via_ast(sql, "postgres")

    def filter_selectable_relation_names(self, schema_name: str, names: Sequence[str]) -> list[str]:
        """Drop relations the current login cannot ``SELECT`` (catalog may still list them)."""
        ordered = [str(n) for n in names if str(n).strip()]
        if not ordered:
            return []
        sql = text(
            "SELECT c.relname FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = ANY(:names) "
            "AND has_table_privilege(c.oid, 'SELECT') "
            "ORDER BY c.relname"
        )
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"schema": schema_name, "names": ordered}).fetchall()
        except Exception as exc:
            debug(f"[dialect.PostgresDialect.filter_selectable_relation_names] skipped: {exc!r}")
            return ordered
        allowed = {str(r[0]) for r in rows}
        return [n for n in ordered if n in allowed]

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Reflect PostgreSQL metadata or parse ``EngineContext.sql_file`` DDL."""
        return load_or_create_schema_postgresql(
            self.engine,
            include=include,
            allow_objects=allow_objects,
            deny_objects=deny_objects,
            schema_json_path=EngineConfig.SCHEMA_JSON_PATH,
            sql_file=sql_file,
            schema_name=str(cast(PostgresRuntimeConfig, self.config).SCHEMA or "public"),
            relation_name_filter=self.filter_selectable_relation_names,
        )

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return SHA-256 over ``information_schema.columns`` rows for the configured PostgreSQL schema. Always returns ``""`` instead of raising on connection / permission / query errors so the caller falls back to the existing fingerprint validation path."""
        _ = schema_context
        try:
            schema_name = str(cast(PostgresRuntimeConfig, self.config).SCHEMA or "public")
            cols_sql = (
                "SELECT table_schema, table_name, column_name, ordinal_position, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_schema = :s "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            unique_sql = (
                "SELECT kcu.table_schema, kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_schema = kcu.constraint_schema "
                " AND tc.constraint_name = kcu.constraint_name "
                "WHERE tc.table_schema = :s AND tc.constraint_type = 'UNIQUE' "
                "ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name"
            )
            with self.engine.connect() as conn:
                rows = conn.execute(text(cols_sql), {"s": schema_name}).fetchall()
                uniq_rows = conn.execute(text(unique_sql), {"s": schema_name}).fetchall()
            payload_cols = "\n".join("|".join("" if c is None else str(c) for c in r) for r in rows)
            payload_uniq = "\n".join("|".join("" if c is None else str(c) for c in r) for r in uniq_rows)
            return sha256(payload_cols + "\n##UNIQUE##\n" + payload_uniq)
        except Exception as exc:
            debug(f"[dialect.PostgresDialect.compute_ddl_probe] failed, returning empty: {exc!r}")
            return ""

    def profile_schema(self, sg: SchemaGraph) -> None:
        """Run SQLAlchemy-backed column profiling for PostgreSQL."""
        profile_schema(self.engine, sg, dialect=self)

    def refresh_full_table_distinct_for_pk_inference(
        self, table_name: str, col_name: str, *, table_kind: TableKind = TableKind.TABLE
    ) -> tuple[int, int, float] | None:
        """Run full-table statistics for PK inference after sampled profiling."""
        try:
            _ = table_kind
            safe_tbl = str(table_name).replace('"', '""')
            safe_col = str(col_name).replace('"', '""')
            sql = text(
                f'SELECT COUNT(*) AS cnt, COUNT(DISTINCT "{safe_col}") AS dist, '
                f'COUNT(*) - COUNT("{safe_col}") AS nulls FROM "{safe_tbl}"'
            )
            with self.engine.connect() as conn:
                row = conn.execute(sql).fetchone()
            if not row:
                return None
            cnt = int(row[0] or 0)
            dist = int(row[1] or 0)
            nulls = int(row[2] or 0)
            nr = float(nulls) / float(cnt) if cnt > 0 else 0.0
            return (dist, cnt, nr)
        except Exception as exc:
            debug(f"[dialect.PostgresDialect.refresh_full_table_distinct_for_pk_inference] failed: {exc!r}")
            return None

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Return a ``TABLESAMPLE BERNOULLI`` suffix for PostgreSQL statistics."""
        if table_kind == "view":
            return ""
        if not use_sample:
            return ""
        pct = 100 * sample_size / row_count if row_count else 0.0
        return f"TABLESAMPLE BERNOULLI ({pct:.2f}) REPEATABLE ({random_seed})"

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """PostgreSQL samples the base table directly with ``TABLESAMPLE``."""
        return table_kind == "view"


DialectRegistry.register_dialect("postgresql", PostgresDialect, PostgresRuntimeConfig)
