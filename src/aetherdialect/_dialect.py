"""Database dialect abstraction: AST/EXPLAIN validation, CTE extraction, and execution helpers. ``sqlglot`` is a required core dependency. Engine-specific dialect implementations live in companion modules registered at package import time. ``register_profile_schema_native_dispatch`` is invoked from ``_dialect_sqlglot_helper`` at module load so native profiling can call back without ``_dialect`` importing that helper module."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, ClassVar, Literal, Protocol, cast

import sqlglot

import aetherdialect._constants

from ._config import EngineConfig, EngineRuntimeConfig, PolicyConfig
from ._constants import (
    AGGREGATE_FUNCTION_NAMES,
    CASE_INSENSITIVE_COLLATION_ENGINES,
    COLLATION_ENGINES,
    DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED,
    EMBEDDED_ENGINE_NAMES,
    ENGINE_MODULE_PATHS,
    EXPLAIN_PERMISSION_DENIED_PATTERNS,
    JSON_CONTAINMENT_ENGINES,
    PG_NAMED_PLACEHOLDER_RE,
    QUALIFY_SKIP_IDENTIFIERS,
    ROUNDING_MODE_HALF_EVEN_ENGINES,
    SQLGLOT_AGG_FUNC_KEY_ALIASES,
    SQLGLOT_DIALECT_BY_ENGINE,
    SUBDAY_RELATIVE_DATE_UNITS,
    TIMESTAMPTZ_SEMANTICS_ENGINES,
    TOML_ENGINE_FIELD_MAPS,
    UNSIGNED_SEMANTICS_ENGINES,
)
from ._contracts_base import (
    DiagnosticSeverity,
    EngineContext,
    EngineIdentity,
    JoinEdge,
    QueryLogSource,
    ResultReaderKind,
    SchemaInclude,
    SqlDiagnostic,
    SqlDiagnosticCode,
    TableKind,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import CatalogStructuralConstraintsIndex, SchemaGraph
from ._core_utils import (
    active_engine_identity,
    assert_dialect_usable_after_fork,
    canonicalize_sql,
    collation_name_is_case_insensitive,
    cost_cap_active,
    debug,
    escape_sql_string_literal_body_base,
    notify,
    pipeline_trace,
    reduce_structural_sql_placeholders,
    sha256,
    stable_json,
    substitute_params,
    substitute_params_for_execution,
)


class Dialect:
    """Base interface for dialect-specific SQL validation and introspection."""

    name: str = "base"
    sqlglot_dialect: ClassVar[str] = ""
    registry_canonical_rank: ClassVar[int] = 1_000
    registry_native_backend: ClassVar[bool] = False
    registry_embedded: ClassVar[bool] = False
    registry_structural_index: ClassVar[bool] = False
    registry_qualified_table_ref: ClassVar[bool] = False
    registry_statistical_agg_excluded: ClassVar[bool] = False
    registry_window_frames_excluded: ClassVar[bool] = False
    registry_array_contains_excluded: ClassVar[bool] = False
    registry_toml_section: ClassVar[str | None] = None
    _PROFILE_SCHEMA_NATIVE_DISPATCH: ClassVar[Callable[..., None] | None] = None

    @staticmethod
    def trace_finalize_render_stage(stage: str, sql_in: str, sql_out: str) -> None:
        """Log one ``finalize_render`` sub-step for debugging and. ``PIPELINE_TRACE`` capture."""
        debug(f"[dialect.finalize_render.{stage}] in_sql_len={len(sql_in)} out_sql_len={len(sql_out)}")
        pipeline_trace(f"dialect.finalize_render.{stage}", lambda: stable_json({"in": sql_in, "out": sql_out}))

    @staticmethod
    def explain_cost_gate_violation(
        est_rows: float | None,
        est_bytes: float | None,
        *,
        dialect: Any | None = None,
        max_query_cost_rows: float | None = None,
        max_query_cost_bytes: float | None = None,
    ) -> tuple[bool, str]:
        """Return ``(True, message)`` when planner estimates exceed configured caps."""
        if max_query_cost_rows is None and dialect is not None:
            override = getattr(dialect, "max_query_cost_rows", None)
            if override is not None:
                max_query_cost_rows = float(override)
        if max_query_cost_bytes is None and dialect is not None:
            override = getattr(dialect, "max_query_cost_bytes", None)
            if override is not None:
                max_query_cost_bytes = float(override)
        caps_r = PolicyConfig.MAX_QUERY_COST_ROWS if max_query_cost_rows is None else max_query_cost_rows
        caps_b = PolicyConfig.MAX_QUERY_COST_BYTES if max_query_cost_bytes is None else max_query_cost_bytes
        over_r = caps_r is not None and cost_cap_active(caps_r) and est_rows is not None and est_rows > caps_r
        over_b = caps_b is not None and cost_cap_active(caps_b) and est_bytes is not None and est_bytes > caps_b
        if not (over_r or over_b):
            return False, ""
        msg = (
            f"EXPLAIN cost gate exceeded: estimated_rows={est_rows} estimated_bytes={est_bytes} "
            f"(limits rows<={caps_r} bytes<={caps_b})"
        )
        return True, msg

    @staticmethod
    def finalize_executable_sql(
        sql_param: str,
        params: dict[str, Any],
        structural_defaults: dict[str, Any] | None = None,
        *,
        sqlglot_dialect: str,
        for_display: bool = False,
        engine: str | None = None,
        dialect: Dialect | None = None,
    ) -> str:
        """Reduce structural placeholders, substitute parameters, then AST- simplify the literal SQL."""
        bind_engine = engine
        if bind_engine is None and dialect is not None:
            bind_engine = getattr(dialect, "name", None)
        if bind_engine is None:
            try:
                bind_engine = active_engine_identity().engine_type
            except Exception:
                bind_engine = None
        reduced, remaining = reduce_structural_sql_placeholders(sql_param, dict(params), structural_defaults)
        if for_display:
            substituted = substitute_params(reduced, remaining, engine=bind_engine, dialect=dialect)
        else:
            substituted = substitute_params_for_execution(reduced, remaining, engine=bind_engine, dialect=dialect)
        return Dialect._sql_simplify_executable(substituted, sqlglot_dialect=sqlglot_dialect)

    @staticmethod
    def is_permission_denied_error(message: str) -> bool:
        """Return True when *message* indicates the database refused EXPLAIN due to credentials."""
        lower = (message or "").lower()
        return any(pat in lower for pat in EXPLAIN_PERMISSION_DENIED_PATTERNS)

    @staticmethod
    def active_sqlglot_dialect() -> str:
        """Return the sqlglot dialect token for the active engine identity."""
        return DialectRegistry.sqlglot_dialect_for_engine(active_engine_identity().engine_type)

    @staticmethod
    def _inspect_parse(sql: str, *, sqlglot_dialect: str) -> sqlglot.exp.Expression | None:
        """Parse *sql* with the given sqlglot *sqlglot_dialect*; returns ``None`` on parser failure."""
        if not sql or not isinstance(sql, str):
            return None
        if not sqlglot_dialect:
            raise ValueError("_inspect_parse requires a non-empty sqlglot_dialect")
        try:
            return sqlglot.parse_one(sql, read=sqlglot_dialect)
        except Exception:
            return None

    @staticmethod
    def normalize_named_placeholders(sql: str) -> str:
        """Convert dialect-specific named placeholders back to ``:name`` form. sqlglot's Postgres generator emits ``%(name)s`` when serialising :class:`sqlglot.expressions.Placeholder` nodes, but the rest of the pipeline (``substitute_params``, SQLAlchemy ``text(...)`` binds) expects the canonical ``:name`` form. This helper rewrites ``%(name)s`` → ``:name`` so the placeholder template stays dialect- agnostic regardless of the round-trip dialect used during AST simplification or parameter abstraction."""
        return PG_NAMED_PLACEHOLDER_RE.sub(lambda m: f":{m.group(1)}", sql)

    @staticmethod
    def relative_window_uses_timestamp(unit: str) -> bool:
        """Return True when a relative date-window unit is finer than one day."""
        return (unit or "").strip().lower() in SUBDAY_RELATIVE_DATE_UNITS

    @staticmethod
    def format_interval_unit(unit: str, amount: int) -> tuple[int, str]:
        """Return ``(amount, unit)`` rewritten to a SQL-compatible ANSI. interval unit. SQL ``INTERVAL`` literals do not understand ``quarter`` or ``half_year``; both PostgreSQL and Spark expect base units such as ``month``. This helper converts those composite units to ``month`` (``quarter`` -> 3, ``half_year`` -> 6) and pluralises the unit when *amount* is not 1 so the rendered fragment reads naturally (``2 days``, ``1 month``, etc.)."""
        canonical = (unit or "").strip().lower()
        if canonical == "quarter":
            scaled = amount * 3
            base = "month"
        elif canonical == "half_year":
            scaled = amount * 6
            base = "month"
        else:
            scaled = amount
            base = canonical or "day"
        plural = f"{base}s" if scaled != 1 else base
        return scaled, plural

    @staticmethod
    def emit_via_ast(sql: str, dialect_name: str) -> str:
        """Round-trip a SQL fragment through the sqlglot AST and re- emit. via. the dialect generator. Used by dialect render helpers so the final fragment passes through a parser/generator pair (rather than being built only by f-string concatenation). The post-processor restores ``:name`` placeholders that the Postgres generator otherwise rewrites to ``%(name)s``."""
        try:
            tree = sqlglot.parse_one(sql, dialect=dialect_name)
        except Exception:
            return sql
        return Dialect.normalize_named_placeholders(tree.sql(dialect=dialect_name))

    @staticmethod
    def sqlglot_quote_identifier(ident: str, sqlglot_dialect: str = "duckdb", *, quoted: bool = True) -> str:
        """Quote a single SQL identifier via the sqlglot dialect generator."""
        s = str(ident).strip()
        if not s:
            return s
        return sqlglot.exp.to_identifier(s, quoted=quoted).sql(dialect=sqlglot_dialect)

    @staticmethod
    def sqlglot_quote_table_column(
        table: str, column: str, sqlglot_dialect: str = "duckdb", *, quoted: bool = True
    ) -> str:
        """Return dialect-safe ``table.column`` via the sqlglot identifier generator."""
        return (
            f"{Dialect.sqlglot_quote_identifier(table, sqlglot_dialect, quoted=quoted)}"
            f".{Dialect.sqlglot_quote_identifier(column, sqlglot_dialect, quoted=quoted)}"
        )

    @staticmethod
    def _outer_select(parsed: sqlglot.exp.Expression) -> sqlglot.exp.Select | None:
        """Return the outer ``Select`` from a parsed expression, ignoring CTE inner selects."""
        if isinstance(parsed, sqlglot.exp.Select):
            return parsed
        inner = parsed.find(sqlglot.exp.Select)
        return inner if isinstance(inner, sqlglot.exp.Select) else None

    @staticmethod
    def sql_outer_select_aliases(sql: str, *, sqlglot_dialect: str) -> list[str]:
        """Return the column display names of the outermost ``SELECT`` projection list."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return []
        select = Dialect._outer_select(parsed)
        if select is None:
            return []
        headers: list[str] = []
        for proj in select.expressions or []:
            alias = proj.alias_or_name if hasattr(proj, "alias_or_name") else ""
            if alias:
                headers.append(alias)
                continue
            if isinstance(proj, sqlglot.exp.Column):
                headers.append(proj.name)
                continue
            headers.append(proj.sql().replace(" ", "_"))
        return headers

    @staticmethod
    def sql_outer_has_join_or_comma_from(sql: str, *, sqlglot_dialect: str) -> bool:
        """Return True when the outer ``SELECT`` uses an explicit JOIN or a comma-separated multi-relation FROM."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return False
        select = Dialect._outer_select(parsed)
        if select is None:
            return False
        if select.args.get("joins"):
            return True
        from_clause = select.args.get("from")
        if from_clause is None:
            return False
        tables: list[sqlglot.exp.Expression] = []
        if isinstance(from_clause, sqlglot.exp.From):
            first = from_clause.this
            if first is not None:
                tables.append(first)
            extras = from_clause.args.get("expressions") or []
            tables.extend(extras)
        return sum(1 for t in tables if isinstance(t, (sqlglot.exp.Table, sqlglot.exp.Subquery))) >= 2

    @staticmethod
    def sql_count_outer_joins(sql: str, *, sqlglot_dialect: str) -> int:
        """Return the number of explicit JOIN clauses across all SELECT scopes."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return 0
        return sum(len(node.args.get("joins") or []) for node in parsed.find_all(sqlglot.exp.Select))

    @staticmethod
    def sql_has_group_by(sql: str, *, sqlglot_dialect: str) -> bool:
        """Return True when any ``SELECT`` in *sql* has a ``GROUP BY`` clause."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return False
        return any(node.args.get("group") for node in parsed.find_all(sqlglot.exp.Select))

    @staticmethod
    def sql_has_distinct(sql: str, *, sqlglot_dialect: str) -> bool:
        """Return True when any ``SELECT`` in *sql* uses ``SELECT DISTINCT``."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return False
        return any(node.args.get("distinct") for node in parsed.find_all(sqlglot.exp.Select))

    @staticmethod
    def sql_has_aggregate(sql: str, *, sqlglot_dialect: str) -> bool:
        """Return True when *sql* contains an aggregate function call. Classification matches IR parsing: sqlglot ``Func.key`` / Anonymous name is canonicalized via ``SQLGLOT_AGG_FUNC_KEY_ALIASES`` and checked against ``AGGREGATE_FUNCTION_NAMES``."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return False
        for node in parsed.walk():
            if isinstance(node, sqlglot.exp.Anonymous):
                func_name = (node.name or "").lower()
            elif isinstance(node, sqlglot.exp.Func):
                func_name = (node.key or "").lower()
            else:
                continue
            canonical = SQLGLOT_AGG_FUNC_KEY_ALIASES.get(func_name, func_name)
            if canonical in AGGREGATE_FUNCTION_NAMES:
                return True
        return False

    @staticmethod
    def _sql_cte_names(sql: str, *, sqlglot_dialect: str) -> set[str]:
        """Return lowercase names of all CTE definitions in *sql*."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return set()
        names: set[str] = set()
        with_clause = parsed.args.get("with") if isinstance(parsed, sqlglot.exp.Select) else None
        if with_clause is None:
            with_clause = parsed.find(sqlglot.exp.With)
        if with_clause is None:
            return names
        for cte in with_clause.expressions or []:
            if isinstance(cte, sqlglot.exp.CTE):
                alias_name = cte.alias_or_name
                if alias_name:
                    names.add(alias_name.lower())
        return names

    @staticmethod
    def sql_tables_referenced(sql: str, *, sqlglot_dialect: str) -> set[str]:
        """Return lowercase physical-table names referenced in *sql*, excluding CTE definitions."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return set()
        cte_names = Dialect._sql_cte_names(sql, sqlglot_dialect=sqlglot_dialect)
        tables: set[str] = set()
        for tbl in parsed.find_all(sqlglot.exp.Table):
            name = (tbl.name or "").lower()
            if name and name not in cte_names:
                tables.add(name)
        return tables

    @staticmethod
    def _simplify_arithmetic_identities_in_tree(tree: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
        """In-place simplify ``1*x``, ``x*1``, ``x+0``, ``x-0``, drop ``LIMIT NULL`` and rewrite ``NOT (X IS NULL)`` to ``X IS NOT NULL``."""
        for node in list(tree.walk()):
            if isinstance(node, sqlglot.exp.Mul):
                left = node.left
                right = node.right
                if isinstance(left, sqlglot.exp.Literal) and not left.is_string and left.this in ("1", "1.0"):
                    node.replace(right.copy())
                    continue
                if isinstance(right, sqlglot.exp.Literal) and not right.is_string and right.this in ("1", "1.0"):
                    node.replace(left.copy())
                    continue
            if isinstance(node, (sqlglot.exp.Add, sqlglot.exp.Sub)):
                right = node.right
                if isinstance(right, sqlglot.exp.Literal) and not right.is_string and right.this in ("0", "0.0"):
                    node.replace(node.left.copy())
                    continue
            if isinstance(node, sqlglot.exp.Not):
                inner = node.this
                if (
                    isinstance(inner, sqlglot.exp.Is)
                    and isinstance(inner.args.get("expression"), sqlglot.exp.Null)
                    and inner.this is not None
                ):
                    replacement = sqlglot.exp.Is(
                        this=inner.this.copy(), expression=sqlglot.exp.Not(this=sqlglot.exp.Null())
                    )
                    node.replace(replacement)
                    continue
        for select in tree.find_all(sqlglot.exp.Select):
            limit_node = select.args.get("limit")
            if limit_node is None:
                continue
            expr = limit_node.expression if hasattr(limit_node, "expression") else None
            if isinstance(expr, sqlglot.exp.Null) or (
                isinstance(expr, sqlglot.exp.Column) and (expr.name or "").lower() == "none"
            ):
                select.set("limit", None)
        return tree

    @staticmethod
    def _sql_simplify_executable(sql: str, *, sqlglot_dialect: str) -> str:
        """Drop trivial arithmetic identities (``1*x``, ``x*1``, ``x+0``, ``x-0``) and ``LIMIT NULL/None`` via AST."""
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            debug(f"[_sql_simplify_executable] parse returned None; preserving input SQL (len={len(sql)})")
            return sql
        simplified = Dialect._simplify_arithmetic_identities_in_tree(parsed)
        try:
            out = Dialect.normalize_named_placeholders(simplified.sql(dialect=sqlglot_dialect))
            if sql.strip() and not out.strip():
                debug(f"[_sql_simplify_executable] sqlglot emission empty; preserving input SQL (len={len(sql)})")
                return sql
            return out
        except Exception:
            debug(f"[_sql_simplify_executable] sqlglot round-trip refused; preserving input SQL (len={len(sql)})")
            return sql

    @staticmethod
    def parameter_abstract(sql: str, *, sqlglot_dialect: str) -> tuple[str, dict[str, Any]]:
        """Replace literal nodes with ``:p1``, ``:p2``, … via sqlglot AST. traversal. Numeric literals are recorded as their parsed value (``int`` or ``float``); string literals are recorded with surrounding single quotes preserved. Returns ``(sql, {})`` unchanged when the SQL is unparseable."""
        if not isinstance(sql, str) or not sql:
            return sql, {}
        parsed = Dialect._inspect_parse(sql, sqlglot_dialect=sqlglot_dialect)
        if parsed is None:
            return sql, {}
        params: dict[str, Any] = {}
        counter = 0
        for literal in list(parsed.find_all(sqlglot.exp.Literal)):
            counter += 1
            key = f"p{counter}"
            if literal.is_string:
                params[key] = f"'{literal.name}'"
            else:
                raw_value = str(literal.this)
                params[key] = raw_value
            literal.replace(sqlglot.exp.Placeholder(this=key))
        try:
            rendered = Dialect.normalize_named_placeholders(parsed.sql(dialect=sqlglot_dialect))
        except Exception:
            return sql, {}
        return " ".join(rendered.split()).strip(), params

    @staticmethod
    def compute_sql_fp(sql: str, *, sqlglot_dialect: str) -> str:
        """Return the canonical-abstracted-lowercased SHA-256 fingerprint for identity keys."""
        if not sql:
            return sha256("")
        canon = canonicalize_sql(sql)
        abstracted, _ = Dialect.parameter_abstract(canon, sqlglot_dialect=sqlglot_dialect)
        return sha256(abstracted.lower())

    @staticmethod
    def check_schema_references_shared(
        refs: list[tuple[str | None, str]], alias_to_table: dict[str, str], cte_names: set[str], schema: SchemaGraph
    ) -> list[SqlDiagnostic]:
        """Validate ``(table_or_alias, column)`` pairs against *schema*. Resolves each prefix through *alias_to_table*. References whose resolved table is a CTE name in *cte_names* are skipped (CTE projection columns are not in the schema graph). Unqualified references are checked for ambiguity across all FROM-side tables; qualified references are checked for table existence and column membership using lowercase normalisation."""
        diags: list[SqlDiagnostic] = []
        seen: set[tuple[str | None, str]] = set()
        from_tables: list[str] = []
        for alias_key, real in alias_to_table.items():
            real_low = (real or "").lower()
            if real_low and real_low not in cte_names and real_low in schema.tables:
                if real_low not in from_tables:
                    from_tables.append(real_low)
            _ = alias_key
        for prefix, column in refs:
            column_low = (column or "").lower()
            if not column_low or column_low == "*":
                continue
            key = (prefix.lower() if prefix else None, column_low)
            if key in seen:
                continue
            seen.add(key)
            if prefix is None:
                owners = [t for t in from_tables if column_low in schema.tables[t].columns]
                if len(owners) == 0:
                    diags.append(
                        SqlDiagnostic(
                            code=SqlDiagnosticCode.UNKNOWN_COLUMN,
                            message=f"unknown column {column_low!r}",
                            node_kind="ColumnRef",
                            offending_identifier=column_low,
                        )
                    )
                elif len(owners) > 1:
                    diags.append(
                        SqlDiagnostic(
                            code=SqlDiagnosticCode.AMBIGUOUS_COLUMN,
                            message=f"ambiguous column {column_low!r} in {owners}",
                            node_kind="ColumnRef",
                            offending_identifier=column_low,
                            details={"owners": ",".join(owners)},
                        )
                    )
                continue
            prefix_low = prefix.lower()
            resolved = (alias_to_table.get(prefix_low) or prefix_low).lower()
            if resolved in cte_names:
                continue
            if resolved not in schema.tables:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.UNKNOWN_TABLE,
                        message=f"unknown table {resolved!r}",
                        node_kind="Table",
                        offending_identifier=resolved,
                    )
                )
                continue
            if column_low not in schema.tables[resolved].columns:
                diags.append(
                    SqlDiagnostic(
                        code=SqlDiagnosticCode.UNKNOWN_COLUMN,
                        message=f"unknown column {resolved}.{column_low}",
                        node_kind="ColumnRef",
                        offending_identifier=f"{resolved}.{column_low}",
                    )
                )
        return diags

    @staticmethod
    def _reflect_include_for_schema_build(ctx: EngineContext) -> SchemaInclude:
        """Mirror :func:`aetherdialect._schema_graph.effective_reflect_include` so partial and full rebuilds agree."""
        return ctx.include

    @staticmethod
    def register_profile_schema_native_dispatch(fn: Callable[..., None]) -> None:
        """Store the native schema-profiling callback registered by ``_dialect_sqlglot_helper`` at module load time."""
        Dialect._PROFILE_SCHEMA_NATIVE_DISPATCH = fn

    @staticmethod
    def _sqlglot_identifier_name(node: Any) -> str:
        """Return the bare identifier string from a sqlglot identifier node."""
        if node is None:
            return ""
        name = getattr(node, "name", None)
        if name:
            return str(name).strip().strip('"').strip("`")
        inner = getattr(node, "this", None)
        if inner is not None and inner is not node:
            return Dialect._sqlglot_identifier_name(inner)
        return str(node).strip().strip('"').strip("`")

    @staticmethod
    def _qualify_tables_ast(
        sql: str, *, sqlglot_dialect: str, catalog: str | None, schema: str, cte_names: set[str], backtick: bool
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
            debug(f"[_qualify_tables_ast] sqlglot parse failed; preserving input SQL (len={len(sql)})")
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
            existing_catalog = table.args.get("catalog")
            existing_db = table.args.get("db")
            if existing_catalog is not None:
                continue
            if existing_db is not None:
                existing_schema = Dialect._sqlglot_identifier_name(existing_db)
                if existing_schema and existing_schema.lower() == schema.lower():
                    continue
                table.set("db", sqlglot.exp.to_identifier(schema, quoted=backtick))
                continue
            table.set("db", sqlglot.exp.to_identifier(schema, quoted=backtick))
            if catalog:
                table.set("catalog", sqlglot.exp.to_identifier(catalog, quoted=backtick))
        try:
            out = parsed.sql(dialect=sqlglot_dialect, identify=backtick)
            if sql.strip() and not out.strip():
                debug(f"[_qualify_tables_ast] sqlglot emission empty; preserving input SQL (len={len(sql)})")
                return sql
            return Dialect.normalize_named_placeholders(out)
        except Exception:
            debug(f"[_qualify_tables_ast] sqlglot serialize failed; preserving input SQL (len={len(sql)})")
            return sql

    @staticmethod
    def cancel_in_flight_statement(dialect: Any) -> None:
        """Cancel an in-flight statement or report when the engine cannot."""
        assert_dialect_usable_after_fork(dialect)
        if not getattr(dialect, "supports_statement_cancellation", True):
            engine_name = str(getattr(dialect, "logical_engine_name", None) or getattr(dialect, "name", "database"))
            notify(
                f"Statement cancellation is not supported for {engine_name}; in-flight work may continue.",
                stage="execution",
                code=DIAGNOSTIC_CODE_CANCEL_NOT_SUPPORTED,
                level=DiagnosticSeverity.WARNING,
                details=(("engine", engine_name),),
            )
            return
        cancel = getattr(dialect, "cancel_statement", None)
        if callable(cancel):
            cancel()

    def __init__(self, config: EngineRuntimeConfig) -> None:
        """Attach runtime configuration used by dialect operations."""
        self.config = config
        self._explain_disabled: bool = False

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        """Backwards-shaped wrapper over :meth:`ast_validate_full`. Returns ``(True, "")`` when no diagnostics are emitted, otherwise ``(False, first_code)`` where ``first_code`` is the string value of the first diagnostic's code."""
        diags = self.ast_validate_full(sql)
        if not diags:
            return True, ""
        return False, str(diags[0].code.value)

    def ast_validate_full(
        self,
        sql: str,
        *,
        schema: SchemaGraph | None = None,
        declared_params: set[str] | None = None,
        scalar_cte_names: frozenset[str] | None = None,
    ) -> list[SqlDiagnostic]:
        """Validate SQL structurally and (when *schema* is provided) semantically without a live connection."""
        raise NotImplementedError

    def parse_select(self, sql: str) -> Any | None:
        """Parse *sql* with the dialect's native AST library and return. an opaque handle. The handle is consumed only by :meth:`ordered_join_carrier_froms`, :meth:`attach_joins`, and :meth:`emit_sql` on the same dialect instance. Returns ``None`` when the SQL cannot be parsed or is not a single SELECT."""
        raise NotImplementedError

    def ordered_join_carrier_froms(self, parsed: Any) -> list[Any] | None:
        """Return per-FROM handles in JOIN-placeholder injection order. Order is each CTE inner SELECT's FROM left-to-right followed by the outer SELECT's FROM. Returns ``None`` for unsupported shapes (e.g. top-level ``UNION``)."""
        raise NotImplementedError

    def attach_joins(self, parsed: Any, from_handle: Any, edges: list[JoinEdge]) -> bool:
        """Attach the given structured *edges* as JOIN nodes onto. *from_handle*. Implementations construct dialect-native JOIN AST nodes directly from *edges* and graft them into *from_handle* without re-parsing any SQL fragment."""
        raise NotImplementedError

    def attach_extra_from_and_where(
        self, parsed: Any, from_handle: Any, extra_from_tables: list[str], where_edges: list[JoinEdge]
    ) -> bool:
        """AND-inject *where_edges*' equality predicates into. *from_handle*'s ``WHERE`` and append any *extra_from_tables* to its ``FROM`` clause. Used to render semantic-profile edges (``edge_kind`` ``semantic_profile`` / ``semantic_profile_virtual``) as comma-FROM + ``WHERE`` equality predicates rather than ``JOIN ... ON``. ``where_edges[i].on_terms`` is a tuple of ``(left_token, left_col, right_token, right_col)`` equality conjuncts that get AND-ed into the existing ``WHERE``. Returns ``True`` on success (including the no-op case when both lists are empty), ``False`` when grafting fails."""
        raise NotImplementedError

    def attach_where_sql_fragments(self, from_handle: Any, fragments: list[str]) -> bool:
        """AND-inject raw SQL predicate fragments into *from_handle*'s ``WHERE`` clause."""
        _ = from_handle
        _ = fragments
        return not fragments

    def from_anchor_of(self, carrier: Any) -> str | None:
        """Return the bare anchor table name of *carrier*'s ``FROM`` clause. Used by :mod:`aetherdialect._sql_gen` to orient join signatures around the carrier's ``FROM`` table without resorting to text regex over the rendered SQL prefix."""
        raise NotImplementedError

    def replace_projection(self, parsed: Any, items: list[tuple[str, str | None]]) -> bool:
        """Replace the outer ``SELECT`` projection list with *items*. Each ``(expr_sql, alias)`` pair is parsed as a single SELECT- list expression in the dialect's native parser and grafted as a ``ResTarget``/``sqlglot.exp.Alias`` node so the surrounding SQL is reconstructed without text splicing. Returns ``True`` on success and ``False`` when any expression or the host statement cannot be parsed."""
        raise NotImplementedError

    def emit_sql(self, parsed: Any) -> str:
        """Re-emit SQL from *parsed* preserving ``:pN`` / ``:sN`` placeholders verbatim."""
        raise NotImplementedError

    def explain_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, str]:
        """Backwards-shaped wrapper over :meth:`explain_diagnose`. Returns ``(ok, raw_message)`` discarding structured diagnostics."""
        ok, _diags, raw = self.explain_diagnose(sql, params, schema=schema, intent=intent)
        return ok, raw

    def explain_diagnose(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        schema: SchemaGraph | None = None,
        intent: RuntimeIntent | None = None,
    ) -> tuple[bool, list[SqlDiagnostic], str]:
        """Run ``EXPLAIN`` against the live engine and return. structured. findings."""
        raise NotImplementedError

    def can_explain(self) -> bool:
        """Return True when ``explain_sql`` can run against a live or embedded engine. ``EXPLAIN`` is always attempted when a backend exists; it is permanently disabled for this dialect instance once a permission-denied error has been observed (see :meth:`_disable_explain_on_permission_denied`)."""
        if self._explain_disabled:
            return False
        return getattr(self, "engine", None) is not None

    @property
    def result_reader_kind(
        self,
    ) -> ResultReaderKind:
        """Return the execution backend kind used to fetch query rows."""
        return ResultReaderKind.SQLALCHEMY

    @property
    def dialect_label(self) -> str:
        """Return a stable engine label for query-log and display routing."""
        return str(self.name)

    @property
    def supports_ilike(self) -> bool:
        """Return True when the dialect accepts ``ilike`` / ``not ilike`` filter operators."""
        return False

    @property
    def supports_case_insensitive_wrap(self) -> bool:
        """Return True when case-insensitive comparison can render without native ``ILIKE``."""
        return True

    @property
    def supports_unnest_select_item(self) -> bool:
        """Return True when an array unnest/explode may appear directly as a SELECT-list item. Only set-returning-function dialects (PostgreSQL ``UNNEST``, Spark ``EXPLODE``) accept this. Engines whose unnest is table- valued (Snowflake ``FLATTEN``, MySQL ``JSON_TABLE``, SQL Server ``OPENJSON``, BigQuery ``UNNEST`` in ``FROM``, Redshift) require a ``FROM``-clause lateral join instead, so they inherit ``False`` and the projection falls back to selecting the array column as-is rather than emitting invalid SQL."""
        return False

    @property
    def parse_backend(self) -> Literal["pglast", "sqlglot"]:
        """Return the primary SQL-to-intent parse backend for this dialect."""
        return "sqlglot"

    @property
    def sql_file_parse_dialect(self) -> str:
        """Return the sqlglot dialect token used when parsing ``EngineContext.sql_file`` DDL."""
        return self.sqlglot_dialect

    def preparse_sql_for_import(self, sql: str) -> str:
        """Normalize SQL text before sqlglot import parsing."""
        return sql

    def map_import_where_op(self, op_raw: str | None) -> str | None:
        """Map a dialect-specific filter operator token to a normalized IR op, or ``None`` to use defaults."""
        _ = op_raw
        return None

    def map_import_scalar_func(self, fn_name: str) -> str:
        """Normalize a scalar function name for import extraction."""
        return fn_name

    def import_unnest_policy(self) -> Literal["select_item", "from_only", "unsupported"]:
        """Indicate where UNNEST/array explode may appear during SQL import."""
        if self.supports_unnest_select_item:
            return "select_item"
        return "from_only"

    def postprocess_imported_intent(self, intent: RuntimeIntent, schema: SchemaGraph) -> RuntimeIntent:
        """Apply dialect-specific last-mile stamps after shared import normalization."""
        _ = schema
        return intent

    def extra_where_ops(self) -> frozenset[str]:
        """Return dialect-specific WHERE operators advertised to intent- parse prompts."""
        extra = set(getattr(type(self), "EXTRA_WHERE_OPS", frozenset()))
        if self.supports_ilike:
            extra.update({"ilike", "not ilike"})
        else:
            extra.discard("ilike")
            extra.discard("not ilike")
        return frozenset(extra)

    def date_window_upper_bound_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the SQL expression for an inclusive date-window upper. bound."""
        return self.date_window_clock_sql(unit, anchor=anchor)

    def date_window_clock_sql(self, unit: str, *, anchor: datetime | None = None) -> str:
        """Return the clock expression used by relative date-window rendering."""
        if anchor is not None:
            return self.render_date_window_anchor_literal(anchor, unit)
        if Dialect.relative_window_uses_timestamp(unit):
            return "CURRENT_TIMESTAMP"
        return "CURRENT_DATE"

    def render_date_window_anchor_literal(self, anchor: datetime, unit: str) -> str:
        """Render a bound anchor instant as a dialect-specific date/timestamp literal."""
        aware = anchor if anchor.tzinfo is not None else anchor.replace(tzinfo=UTC)
        if Dialect.relative_window_uses_timestamp(unit):
            text = aware.isoformat(sep=" ", timespec="microseconds")
            return f"TIMESTAMP '{text}'"
        return f"DATE '{aware.date().isoformat()}'"

    def render_date_literal(self, value: date | datetime) -> str:
        """Render an absolute calendar date or timestamp literal."""
        if isinstance(value, datetime):
            aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            text = aware.isoformat(sep=" ", timespec="microseconds")
            return f"TIMESTAMP '{text}'"
        return f"DATE '{value.isoformat()}'"

    def normalize_window_agg_sql_frag(self, frag: str) -> str:
        """Normalize a window or aggregate SQL fragment for dialect- specific qualification rules."""
        return frag

    def profile_statement_timeout_sql(self, timeout_ms: int) -> str | None:
        """Return a session statement-timeout SQL statement for. profiling, or ``None`` when unsupported."""
        _ = timeout_ms
        return None

    def session_timezone_sql(self) -> str | None:
        """Return SQL that yields the active session time zone, or ``None`` when unsupported."""
        return None

    def inject_pruning_predicates(
        self, sql: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> str:
        """Append engine-specific pruning predicates when the WHERE. clause omits required keys."""
        _ = schema, intent
        return sql

    def explain_row_estimate(
        self, sql_text: str, *, schema: SchemaGraph | None = None, intent: RuntimeIntent | None = None
    ) -> float | None:
        """Return planner row-count estimate for *sql_text*, or ``None`` when unavailable."""
        _ = sql_text, schema, intent
        return None

    def query_log_source(self) -> QueryLogSource | None:
        """Return a dialect-specific query-log source, or ``None`` when. unsupported."""
        return None

    def quote_table_column(self, table: str, column: str) -> str:
        """Return a dialect-safe ``table.column`` reference for SQL. emission."""
        dialect = getattr(type(self), "sqlglot_dialect", "") or ""
        if not dialect:
            raise NotImplementedError
        return Dialect.sqlglot_quote_table_column(table, column, dialect)

    def _disable_explain_on_permission_denied(self, error_message: str) -> bool:
        """Flip ``_explain_disabled`` when *error_message* indicates a credentials issue. Returns True when the error was classified as permission denied (and EXPLAIN has been disabled for this dialect instance), otherwise False."""
        if Dialect.is_permission_denied_error(error_message):
            if not self._explain_disabled:
                debug(
                    f"[dialect.explain_sql] permission denied ({error_message!r}); "
                    f"disabling EXPLAIN for this dialect instance"
                )
            self._explain_disabled = True
            return True
        return False

    def catalog_name(self) -> str | None:
        """Return the active catalog or project name from runtime config."""
        config = getattr(self, "config", None)
        if config is None:
            return None
        for attr in ("CATALOG", "PROJECT", "DATABASE"):
            value = getattr(config, attr, None)
            if value:
                return str(value)
        return None

    def schema_name(self) -> str:
        """Return the active schema or database name from runtime config."""
        config = getattr(self, "config", None)
        if config is None:
            return ""
        for attr in ("SCHEMA", "DATABASE", "DATASET"):
            value = getattr(config, attr, None)
            if value:
                return str(value)
        return ""

    def qualified_table_ref(self, table: str, kind: TableKind = TableKind.TABLE) -> str:
        """Return a dialect-safe fully qualified table reference for profiling and execution."""
        _ = kind
        return self.quote_identifier(table)

    def structural_constraints_index(self) -> CatalogStructuralConstraintsIndex:
        """Return PK, FK, and single-column UNIQUE metadata for the active catalog scope."""
        return CatalogStructuralConstraintsIndex.empty()

    def table_kinds_map(self) -> dict[str, str]:
        """Return lowercased relation name to ``information_schema.tables.table_type`` strings."""
        return {}

    def pre_execute_rewrite(self, sql: str) -> str:
        """Apply engine-specific SQL rewrites before parameter substitution."""
        return sql

    def post_render_normalize(self, sql: str, *, stage: str) -> str:
        """Normalize finalized SQL after substitution for engine- specific emission quirks."""
        _ = stage
        return sql

    def apply_execute_cost_limits(self, target: Any) -> None:
        """Apply engine-specific execute-time cost caps to a driver handle or job config."""
        _ = target

    def _qualify_uses_backtick_identifiers(self) -> bool:
        """Return True when :func:`_qualify_tables_ast` should emit backtick-quoted identifiers."""
        return False

    def _qualify_tables_for_execution(self, sql: str) -> str:
        """Qualify bare table references with the active schema (and optional catalog)."""
        sch = self.schema_name()
        if not sch or not sql or not sql.strip():
            return sql
        return Dialect._qualify_tables_ast(
            sql,
            sqlglot_dialect=self.sqlglot_dialect,
            catalog=self.catalog_name(),
            schema=sch,
            cte_names=set(),
            backtick=self._qualify_uses_backtick_identifiers(),
        )

    def explain_validation_sql(self, sql: str, param_values: dict[str, Any] | None = None) -> str:
        """Return SQL suitable for AST and EXPLAIN validation before execution rewrites."""
        _ = param_values
        return sql

    def render_date_window_inclusive_upper(
        self, left_rendered: str, unit: str, *, anchor: datetime | None = None
    ) -> str:
        """Render the inclusive upper bound for a relative date_window filter."""
        return f"{left_rendered} <= {self.date_window_upper_bound_sql(unit, anchor=anchor)}"

    def prepare_for_execution(self, sql: str) -> str:
        """Return SQL in the form required for execution."""
        return self.pre_execute_rewrite(sql)

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
        """Produce executable SQL through the shared render pipeline."""
        sql_in_raw = execution_sql_override or sql_param
        substituted = Dialect.finalize_executable_sql(
            sql_in_raw, params, structural_defaults, sqlglot_dialect=self.sqlglot_dialect
        )
        Dialect.trace_finalize_render_stage("finalize_executable_sql", sql_in_raw, substituted)
        rewritten = self.pre_execute_rewrite(substituted)
        Dialect.trace_finalize_render_stage("pre_execute_rewrite", substituted, rewritten)
        non_empty_in = (execution_sql_override or sql_param or "").strip()
        if non_empty_in and not rewritten.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=pre_execute_rewrite"
            )
        normalized = self.post_render_normalize(rewritten, stage="post_substitute")
        Dialect.trace_finalize_render_stage("post_render_normalize", rewritten, normalized)
        if non_empty_in and not normalized.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=post_render_normalize"
            )
        qualified = self._qualify_tables_for_execution(normalized)
        Dialect.trace_finalize_render_stage("qualify_tables_for_execution", normalized, qualified)
        if non_empty_in and not qualified.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=qualify_tables_for_execution"
            )
        pruned = self.inject_pruning_predicates(qualified, schema=schema, intent=intent)
        Dialect.trace_finalize_render_stage("inject_pruning_predicates", qualified, pruned)
        if non_empty_in and not pruned.strip():
            raise RuntimeError(
                "dialect.finalize_render produced empty SQL from non-empty input; "
                "last_non_empty_stage=inject_pruning_predicates"
            )
        return pruned

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute SQL and return rows as tuples."""
        raise NotImplementedError

    @property
    def supports_statement_cancellation(self) -> bool:
        """Return True when the active driver exposes an in-flight cancel hook."""
        return True

    @property
    def logical_engine_name(self) -> str:
        """Return a user-facing engine label for diagnostics."""
        label = str(getattr(self, "name", "") or "").strip()
        return label.replace("_", " ").title() or self.__class__.__name__

    def cancel_statement(self) -> None:
        """Cancel an in-flight statement when the active driver supports it."""
        backend = getattr(self, "result_backend", None)
        if backend is None:
            return
        cancel = getattr(backend, "cancel_statement", None)
        if callable(cancel):
            cancel()

    def quote_identifier(self, ident: str) -> str:
        """Quote a single SQL identifier via the sqlglot dialect generator."""
        dialect = getattr(type(self), "sqlglot_dialect", "") or ""
        if dialect:
            return Dialect.sqlglot_quote_identifier(ident, dialect)
        return Dialect.sqlglot_quote_identifier(ident, "postgres")

    def quote_schema_qualified(self, name: str) -> str:
        """Quote a dotted identifier path as one quoted fragment per. segment."""
        parts = [p for p in str(name).strip().split(".") if p]
        if not parts:
            return self.quote_identifier(name)
        return ".".join(self.quote_identifier(p) for p in parts)

    def escape_string_literal(self, value: str) -> str:
        """Escape *value* for embedding inside a single-quoted SQL string literal body."""
        return escape_sql_string_literal_body_base(str(value))

    def quote_string_literal(self, text: str) -> str:
        """Render a string value as a single-quoted SQL string literal."""
        return f"'{self.escape_string_literal(text)}'"

    def render_date_diff(
        self, left_expr: str, op: str, unit: str, amount: int, *, minuend_sql: str = "", subtrahend_sql: str = ""
    ) -> str:
        """Render a date-difference comparison predicate."""
        scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
        return f"({left_expr}) {op} INTERVAL '{scaled} {plural_unit}'"

    def render_date_integer_days(self, base_sql: str, sign: str, offset_sql: str) -> str:
        """Render date-column plus or minus an integer day count."""
        if sign == "+":
            return f"({base_sql} + ({offset_sql}) * INTERVAL '1 day')"
        return f"({base_sql} - ({offset_sql}) * INTERVAL '1 day')"

    @property
    def supports_ordered_string_agg(self) -> bool:
        """Return True when ``string_agg`` may carry an ``ORDER BY`` clause inside the aggregate."""
        return True

    def render_string_agg(self, expr_sql: str, sep_sql: str, order_by_sql: str) -> str:
        """Render a per-group string concatenation aggregate."""
        if order_by_sql:
            return f"STRING_AGG({expr_sql}, {sep_sql} ORDER BY {order_by_sql})"
        return f"STRING_AGG({expr_sql}, {sep_sql})"

    @property
    def supports_median(self) -> bool:
        """Return True when a native or percentile median aggregate is available."""
        return True

    @property
    def supports_semi_join(self) -> bool:
        """Return True when the engine can emit semi-join (EXISTS-style) predicates."""
        return True

    @property
    def supports_anti_join(self) -> bool:
        """Return True when the engine can emit anti-join (NOT EXISTS- style) predicates."""
        return True

    @property
    def supports_predicate_nesting(self) -> bool:
        """Return True when nested boolean predicate groups are supported."""
        return True

    @property
    def supports_stddev(self) -> bool:
        """Return True when the engine can render a sample standard- deviation aggregate."""
        return not type(self).registry_statistical_agg_excluded

    @property
    def supports_variance(self) -> bool:
        """Return True when the engine can render a sample variance aggregate."""
        return self.supports_stddev

    @property
    def supports_window_frames(self) -> bool:
        """Return True when the engine can render explicit window ROWS/RANGE frames."""
        return not type(self).registry_window_frames_excluded

    @property
    def supports_array_contains(self) -> bool:
        """Return True when the engine can render array ``contains`` predicates."""
        return not type(self).registry_array_contains_excluded

    @property
    def supports_collation(self) -> bool:
        """Return True when the engine exposes explicit ``COLLATE`` semantics."""
        return self.name in COLLATION_ENGINES

    @property
    def default_deterministic_collation(self) -> str | None:
        """Return the collation name to use when ordering must be reproducible."""
        return None

    def quote_collation_name(self, name: str) -> str:
        """Quote a collation identifier for SQL emission."""
        if self.name in ("postgresql", "redshift"):
            return f'"{name}"'
        return name

    def pin_deterministic_order_expr(self, rendered_expr: str) -> str:
        """Apply deterministic ``COLLATE`` to an ORDER BY expression when configured."""
        collation = self.default_deterministic_collation
        if not collation:
            return rendered_expr
        return f"{rendered_expr} COLLATE {self.quote_collation_name(collation)}"

    @property
    def is_case_insensitive_collation(self) -> bool:
        """Return True when the connection default collation compares strings case-insensitively."""
        return self.name in CASE_INSENSITIVE_COLLATION_ENGINES

    def column_is_case_insensitive_collation(self, col: Any) -> bool:
        """Return True when *col* compares strings case-insensitively under its collation or connection default."""
        collation = getattr(col, "collation", None)
        if collation:
            return collation_name_is_case_insensitive(str(collation))
        return self.is_case_insensitive_collation

    @property
    def supports_unsigned_semantics(self) -> bool:
        """Return True when the engine exposes unsigned integer semantics."""
        return self.name in UNSIGNED_SEMANTICS_ENGINES

    @property
    def supports_timestamptz_semantics(self) -> bool:
        """Return True when the engine distinguishes timezone-aware timestamps."""
        return self.name in TIMESTAMPTZ_SEMANTICS_ENGINES

    @property
    def integer_division_truncates(self) -> bool:
        """Return True when dividing two integers truncates unless the numerator is cast."""
        return False

    @property
    def rounding_mode(self) -> str:
        """Return the dialect's documented ``ROUND`` tie-breaking mode (``half_up`` or ``half_even``)."""
        if self.name in ROUNDING_MODE_HALF_EVEN_ENGINES:
            return "half_even"
        return "half_up"

    def render_date_trunc(self, unit: str, expr_sql: str) -> str:
        """Render calendar truncation for *unit* on *expr_sql* using native units where available."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            return (
                f"CASE WHEN EXTRACT(MONTH FROM {expr_sql}) <= 6 "
                f"THEN DATE_TRUNC('year', {expr_sql}) "
                f"ELSE DATE_TRUNC('year', {expr_sql}) + INTERVAL '6 months' END"
            )
        if unit_norm == "quarter":
            return f"DATE_TRUNC('quarter', {expr_sql})"
        return f"DATE_TRUNC('{unit_norm}', {expr_sql})"

    def render_extract(self, unit: str, expr_sql: str) -> str:
        """Render calendar extraction for *unit* from *expr_sql* using native units where available."""
        unit_norm = (unit or "").strip().lower()
        if unit_norm == "half_year":
            return f"CASE WHEN EXTRACT(MONTH FROM {expr_sql}) <= 6 THEN 1 ELSE 2 END"
        if unit_norm == "quarter":
            return f"EXTRACT(QUARTER FROM {expr_sql})"
        return f"EXTRACT({unit_norm.upper()} FROM {expr_sql})"

    def render_median(self, expr_sql: str) -> str:
        """Render a median aggregate using the dialect's portable spelling."""
        return f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {expr_sql})"

    def render_containment(self, column_sql: str, value_param: str, value_type: str) -> str | None:
        """Return SQL for JSON-array containment, or ``None`` when the dialect has no native operator."""
        _ = (column_sql, value_param, value_type)
        return None

    @property
    def supports_json_containment(self) -> bool:
        """Return True when the dialect can render native JSON-array containment."""
        return self.name in JSON_CONTAINMENT_ENGINES

    def render_array_contains(
        self,
        column_sql: str,
        param_key: str,
        *,
        column_meta: Any | None = None,
        value_type: str = "string",
    ) -> str:
        """Render array membership (contains) for WHERE/HAVING."""
        _ = (column_meta, value_type)
        return f":{param_key} = ANY({column_sql})"

    def render_array_unnest(self, column_sql: str, alias: str) -> str:
        """Render UNNEST or equivalent for a SELECT list item."""
        return f"UNNEST({column_sql}) AS {alias}"

    def render_date_window(
        self, column: str, op: str, unit: str, amount: int, *, anchor: datetime | None = None
    ) -> str:
        """Render a relative date-window boundary expression."""
        clock = self.date_window_clock_sql(unit, anchor=anchor)
        if amount == 0:
            return f"{column} {op} DATE_TRUNC('{unit}', {clock})"
        scaled, plural_unit = Dialect.format_interval_unit(unit, amount)
        return f"{column} {op} {clock} - INTERVAL '{scaled} {plural_unit}'"

    def render_case_insensitive_wrap(self, expr: str) -> str:
        """Wrap an expression for case-insensitive string comparison."""
        return f"LOWER({expr})"

    def render_fixed_width_text_wrap(self, expr: str) -> str:
        """Wrap an expression to strip trailing padding from fixed-width text."""
        return f"RTRIM({expr})"

    def finalize_like_predicate_sql(self, sql: str) -> str:
        """Return SQL for a LIKE/ILIKE predicate, optionally normalized for the dialect parser."""
        return sql

    def render_order_by_col(self, rendered_expr: str, direction: str, nulls: str | None) -> str:
        """Render one ORDER BY key with optional explicit null placement."""
        dir_up = (direction or "ASC").strip().upper() or "ASC"
        base = f"{rendered_expr} {dir_up}"
        if nulls not in ("first", "last"):
            return base
        if self.name in ("mysql", "mariadb", "sqlserver"):
            is_null = f"({rendered_expr} IS NULL)"
            lead_dir = "DESC" if nulls == "first" else "ASC"
            return f"{is_null} {lead_dir}, {base}"
        placement = "NULLS FIRST" if nulls == "first" else "NULLS LAST"
        return f"{base} {placement}"

    def reflect_schema_graph(
        self,
        *,
        include: SchemaInclude = SchemaInclude.TABLES,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from catalog or DDL fallback."""
        raise NotImplementedError

    def compute_ddl_probe(self, schema_context: EngineContext) -> str:
        """Return a cheap deterministic fingerprint of the live DDL the. cache should be valid against. Concrete dialects should run a single ``information_schema.columns`` query (or equivalent) scoped to the active schema/catalog and return a SHA-256 hex digest over the sorted ``(table, column, ordinal_position, data_type, is_nullable)`` rows. This probe is consulted by :func:`aetherdialect._schema.build_schema_graph` to short- circuit cache loads without re-reflecting or re-profiling the schema. The base implementation returns an empty string, which disables the fast path and forces the existing fingerprint-based cache validation. Returning ``""`` is also the contract for "probe not available at collection time" (e.g., transient DB error): callers must never propagate exceptions from this method."""
        _ = schema_context
        return ""

    def compute_row_count_probe(self, sg: SchemaGraph) -> str:
        """Return a cheap fingerprint of live table row counts for profiling-cache drift checks, or ``""`` when unavailable."""
        _ = sg
        return ""

    def reflect_only(self, schema_context: EngineContext) -> SchemaGraph:
        """Reflect a structural-only ``SchemaGraph`` honouring ``schema_context.include``. Used by the partial-rebuild diff path: only structural shape (tables, columns, FKs) is needed in order to compute a :class:`SchemaDiff`; profiling is run later, on the affected subset only. The default implementation delegates to :meth:`reflect_schema_graph` with the effective include kind. Dialects may override to skip work that is unnecessary for the diff (e.g., enum value enrichment)."""
        include = Dialect._reflect_include_for_schema_build(schema_context)
        allow_obj = schema_context.allow_objects if schema_context.allow_objects else None
        deny_obj = schema_context.deny_objects if schema_context.deny_objects else None
        return self.reflect_schema_graph(
            include=include, allow_objects=allow_obj, deny_objects=deny_obj, sql_file=schema_context.sql_file
        )

    def profile_schema_dispatch(self, sg: SchemaGraph) -> None:
        """Profile tables using the active native backend chain with SQLAlchemy fallback."""
        if Dialect._PROFILE_SCHEMA_NATIVE_DISPATCH is None:
            raise NotImplementedError(f"{type(self).__name__} has no profiling backend")
        Dialect._PROFILE_SCHEMA_NATIVE_DISPATCH(self, sg)

    def profile_schema(self, sg: SchemaGraph) -> None:
        """Populate column statistics and physical metadata on *sg* in place."""
        self.profile_schema_dispatch(sg)

    def refresh_full_table_distinct_for_pk_inference(
        self, table_name: str, col_name: str, *, table_kind: TableKind = TableKind.TABLE
    ) -> tuple[int, int, float] | None:
        """Run full-table statistics for PK inference after sampled profiling."""
        return None

    def refresh_composite_distinct_for_pk_inference(
        self, table_name: str, col_names: list[str], *, table_kind: TableKind = TableKind.TABLE
    ) -> tuple[int, int, float] | None:
        """Run full-table composite distinct statistics for multi-column PK inference."""
        return None

    def profiling_text_cast_sql(self, expr: str) -> str:
        """Return a dialect-correct text cast expression for profiling. overlap queries."""
        return f"CAST({expr} AS TEXT)"

    def profiling_ordered_limit_sample_suffix(self, sample_size: int) -> str:
        """Return a deterministic ordered row cap for engines without seeded sampling."""
        return f"ORDER BY 1 LIMIT {sample_size}"

    def profiling_stats_sample_suffix(
        self,
        *,
        use_sample: bool,
        row_count: int,
        sample_size: int,
        random_seed: int,
        table_kind: TableKind = TableKind.TABLE,
    ) -> str:
        """Build a deterministic ``FROM``-clause sampling suffix for row-count statistics queries. The base dialect uses an ordered row cap (``ORDER BY 1 LIMIT``) so subquery sampling does not depend on physical heap order. Callers that embed the suffix in ``(SELECT col FROM tbl …)`` should rewrite ``ORDER BY 1`` to ``ORDER BY col`` when building the subquery."""
        _ = row_count, random_seed, table_kind
        if not use_sample:
            return ""
        return self.profiling_ordered_limit_sample_suffix(sample_size)

    def profiling_stats_use_subquery_when_sampling(self, table_kind: TableKind = TableKind.TABLE) -> bool:
        """Return True when distinct/null stats must scan a sampled. subquery."""
        return True


class _BaseDialectCtor(Protocol):
    def __call__(self, config: Any) -> Dialect: ...


class _SqlalchemyDialectCtor(Protocol):
    def __call__(self, config: Any, *, sqlalchemy_engine: Any | None = ...) -> Dialect: ...


class _EmbeddedDialectCtor(Protocol):
    def __call__(
        self, config: Any, *, sqlalchemy_engine: Any | None = ..., native_connection: Any | None = ...
    ) -> Dialect: ...


class DialectRegistry:
    """Engine dialect registration, capability lookup, and construction."""

    _DIALECT_REGISTRY: ClassVar[dict[str, type[Dialect]]] = {}
    _RUNTIME_REGISTRY: ClassVar[dict[str, type]] = {}

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return known engine names from static paths and the runtime registry."""
        return sorted(set(ENGINE_MODULE_PATHS) | set(cls._DIALECT_REGISTRY))

    @classmethod
    def _ensure_engine_registered(cls, engine_type: str) -> None:
        name = (engine_type or "").strip().lower()
        if not name or name in cls._DIALECT_REGISTRY:
            return
        module_path = ENGINE_MODULE_PATHS.get(name)
        if module_path is None:
            return
        importlib.import_module(module_path)

    @classmethod
    def get_registered_engines(cls) -> list[str]:
        """Return sorted registered engine names (alias for :meth:`list_engines`)."""
        return cls.list_engines()

    @classmethod
    def derive_dialect_registry_surfaces(cls) -> dict[str, Any]:
        """Derive engine registry surfaces from registered dialect class metadata."""
        registered = dict(cls._DIALECT_REGISTRY)
        canonical_order = tuple(
            sorted(registered.keys(), key=lambda engine: (registered[engine].registry_canonical_rank, engine))
        )
        return {
            "canonical_engine_order": canonical_order,
            "native_backend_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_native_backend
            ),
            "embedded_engine_names": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_embedded
            ),
            "structural_index_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_structural_index
            ),
            "qualified_table_ref_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_qualified_table_ref
            ),
            "statistical_agg_excluded_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_statistical_agg_excluded
            ),
            "window_frames_excluded_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_window_frames_excluded
            ),
            "array_contains_excluded_engines": frozenset(
                engine for engine, reg_cls in registered.items() if reg_cls.registry_array_contains_excluded
            ),
            "toml_field_map_engines": frozenset(
                section
                for engine, reg_cls in registered.items()
                if (section := (reg_cls.registry_toml_section or engine)) in TOML_ENGINE_FIELD_MAPS
            ),
        }

    @classmethod
    def _sync_dialect_registry_constants(cls) -> None:
        """Publish derived registry surfaces into :mod:`aetherdialect._constants`."""
        derived = cls.derive_dialect_registry_surfaces()
        const = cast(Any, aetherdialect._constants)
        const.CANONICAL_ENGINE_ORDER = derived["canonical_engine_order"]
        const.NATIVE_BACKEND_ENGINES = derived["native_backend_engines"]
        const.EMBEDDED_ENGINE_NAMES = derived["embedded_engine_names"]
        const.STRUCTURAL_INDEX_ENGINES = derived["structural_index_engines"]
        const.QUALIFIED_TABLE_REF_ENGINES = derived["qualified_table_ref_engines"]
        const.STATISTICAL_AGG_EXCLUDED_ENGINES = derived["statistical_agg_excluded_engines"]
        const.WINDOW_FRAMES_EXCLUDED_ENGINES = derived["window_frames_excluded_engines"]
        const.ARRAY_CONTAINS_EXCLUDED_ENGINES = derived["array_contains_excluded_engines"]

    @classmethod
    def set_supported_engines(cls, engines: frozenset[str]) -> None:
        """Update the runtime registry of dialect names after ``register_dialect`` calls."""
        cast(Any, aetherdialect._constants).SUPPORTED_ENGINES = engines

    @classmethod
    def register_dialect(cls, name: str, dialect_cls: type[Dialect], runtime_config_cls: type | None = None) -> None:
        """Register a dialect implementation under an engine name."""
        cls._DIALECT_REGISTRY[name] = dialect_cls
        if runtime_config_cls is not None:
            cls._RUNTIME_REGISTRY[name] = runtime_config_cls
        if dialect_cls.sqlglot_dialect:
            cast(Any, aetherdialect._constants).SQLGLOT_DIALECT_BY_ENGINE[name] = dialect_cls.sqlglot_dialect
        cls.set_supported_engines(frozenset(cls._DIALECT_REGISTRY))
        cls._sync_dialect_registry_constants()

    @classmethod
    def get_dialect_class(cls, engine_type: str) -> type[Dialect]:
        """Return the registered dialect class for *engine_type* without. constructing an instance. Raises: ValueError: When *engine_type* is not registered."""
        cls._ensure_engine_registered(engine_type)
        if engine_type not in cls._DIALECT_REGISTRY:
            raise ValueError(f"Unsupported dialect: {engine_type}")
        return cls._DIALECT_REGISTRY[engine_type]

    @classmethod
    def get_runtime_config_class(cls, engine_type: str) -> type:
        """Return the registered runtime config class for *engine_type*."""
        cls._ensure_engine_registered(engine_type)
        if engine_type not in cls._RUNTIME_REGISTRY:
            raise ValueError(f"No runtime config registered for engine: {engine_type}")
        return cls._RUNTIME_REGISTRY[engine_type]

    @classmethod
    def sqlglot_dialect_for_engine(cls, engine_type: str) -> str:
        """Return the sqlglot dialect token for *engine_type*."""
        et = (engine_type or "").strip().lower()
        if et in cls._DIALECT_REGISTRY:
            token = cls._DIALECT_REGISTRY[et].sqlglot_dialect
            if token:
                return token
        fallback_token = SQLGLOT_DIALECT_BY_ENGINE.get(et)
        if fallback_token is None:
            raise ValueError(
                f"No sqlglot dialect mapping for engine type {et!r}; expected one of "
                f"{sorted(cls._DIALECT_REGISTRY) or sorted(SQLGLOT_DIALECT_BY_ENGINE)}"
            )
        return fallback_token

    @classmethod
    def extra_where_ops_for_engine(cls, engine_type: str | None = None) -> frozenset[str]:
        """Return dialect-specific WHERE operators without constructing a dialect instance."""
        et = (engine_type or EngineConfig.TYPE).strip().lower()
        dialect_cls = cls.get_dialect_class(et)
        stub = object.__new__(dialect_cls)
        return stub.extra_where_ops()

    @classmethod
    def member_supports_ilike_semantics(cls, engine_type: str) -> bool:
        """Return True when a member can express case-insensitive string filters."""
        dialect_cls = cls.get_dialect_class(engine_type.strip().lower())
        stub = object.__new__(dialect_cls)
        if stub.supports_ilike:
            return True
        return bool(stub.supports_case_insensitive_wrap)

    @staticmethod
    def dialect_supports_ilike_semantics(dialect: Dialect) -> bool:
        """Return True when *dialect* can express case-insensitive string filters."""
        if dialect.supports_ilike:
            return True
        return bool(dialect.supports_case_insensitive_wrap)

    @classmethod
    def dialect_stub_for_engine(cls, engine_type: str) -> Dialect | None:
        """Return an uninitialized dialect instance for capability introspection."""
        engine = (engine_type or "").strip().lower()
        if not engine:
            return None
        try:
            dialect_cls = cls.get_dialect_class(engine)
        except (ValueError, KeyError):
            return None
        return object.__new__(dialect_cls)

    @classmethod
    def _engine_supports_attr(cls, engine_type: str, attr: str, *, default: bool = True) -> bool:
        stub = cls.dialect_stub_for_engine(engine_type)
        if stub is None:
            return default
        return bool(getattr(stub, attr))

    @classmethod
    def engine_supports_ordered_string_agg(cls, engine_type: str) -> bool:
        """Return True when the engine can render ``string_agg`` with an in- aggregate ``ORDER BY``."""
        return cls._engine_supports_attr(engine_type, "supports_ordered_string_agg")

    @classmethod
    def engine_supports_median(cls, engine_type: str) -> bool:
        """Return True when the engine exposes a median aggregate."""
        return cls._engine_supports_attr(engine_type, "supports_median")

    @classmethod
    def engine_supports_stddev(cls, engine_type: str) -> bool:
        """Return True when the engine can render a sample standard- deviation aggregate."""
        return cls._engine_supports_attr(engine_type, "supports_stddev")

    @classmethod
    def engine_supports_variance(cls, engine_type: str) -> bool:
        """Return True when the engine can render a sample variance aggregate."""
        return cls._engine_supports_attr(engine_type, "supports_variance")

    @classmethod
    def engine_supports_window_frames(cls, engine_type: str) -> bool:
        """Return True when the engine can render explicit window ROWS/RANGE frames."""
        return cls._engine_supports_attr(engine_type, "supports_window_frames")

    @classmethod
    def engine_supports_array_contains(cls, engine_type: str) -> bool:
        """Return True when the engine can render array ``contains`` predicates."""
        return cls._engine_supports_attr(engine_type, "supports_array_contains")

    @classmethod
    def engine_supports_json_containment(cls, engine_type: str) -> bool:
        """Return True when the engine can render native JSON-array containment."""
        return cls._engine_supports_attr(engine_type, "supports_json_containment", default=False)

    @classmethod
    def engine_supports_collation(cls, engine_type: str) -> bool:
        """Return True when the engine exposes explicit ``COLLATE`` semantics."""
        return cls._engine_supports_attr(engine_type, "supports_collation", default=False)

    @classmethod
    def engine_default_deterministic_collation(cls, engine_type: str) -> str | None:
        """Return the collation name to use when ordering must be reproducible on *engine_type*."""
        stub = cls.dialect_stub_for_engine(engine_type)
        if stub is None:
            return None
        return stub.default_deterministic_collation

    @classmethod
    def engine_is_case_insensitive_collation(cls, engine_type: str) -> bool:
        """Return True when the engine's default collation compares strings case-insensitively."""
        return cls._engine_supports_attr(engine_type, "is_case_insensitive_collation", default=False)

    @classmethod
    def engine_supports_unsigned_semantics(cls, engine_type: str) -> bool:
        """Return True when the engine exposes unsigned integer semantics."""
        return cls._engine_supports_attr(engine_type, "supports_unsigned_semantics", default=False)

    @classmethod
    def engine_supports_timestamptz_semantics(cls, engine_type: str) -> bool:
        """Return True when the engine distinguishes timezone-aware timestamps."""
        return cls._engine_supports_attr(engine_type, "supports_timestamptz_semantics", default=False)

    @classmethod
    def engine_integer_division_truncates(cls, engine_type: str) -> bool:
        """Return True when integer division truncates unless the numerator is cast."""
        return cls._engine_supports_attr(engine_type, "integer_division_truncates", default=False)

    @classmethod
    def engine_rounding_mode(cls, engine_type: str) -> str:
        """Return the documented rounding tie-breaking mode for *engine_type*."""
        stub = cls.dialect_stub_for_engine(engine_type)
        if stub is None:
            return "half_up"
        return str(stub.rounding_mode)

    @classmethod
    def engine_supports_semi_join(cls, engine_type: str) -> bool:
        """Return True when the engine can emit semi-join predicates."""
        return cls._engine_supports_attr(engine_type, "supports_semi_join")

    @classmethod
    def engine_supports_anti_join(cls, engine_type: str) -> bool:
        """Return True when the engine can emit anti-join predicates."""
        return cls._engine_supports_attr(engine_type, "supports_anti_join")

    @classmethod
    def engine_supports_predicate_nesting(cls, engine_type: str) -> bool:
        """Return True when nested boolean predicate groups are supported."""
        return cls._engine_supports_attr(engine_type, "supports_predicate_nesting")

    @classmethod
    def resolve_dialect(cls, name_or_api: str | Dialect) -> Dialect:
        """Return a dialect instance for helpers that accept either an engine name or a live dialect."""
        if isinstance(name_or_api, Dialect):
            return name_or_api
        if isinstance(name_or_api, str):
            return cls.get_dialect(name_or_api)
        raise TypeError(f"Expected str or Dialect, got {type(name_or_api).__name__}")

    @classmethod
    def get_dialect(
        cls,
        engine_type: str | None = None,
        config: Any | None = None,
        sqlalchemy_engine: Any | None = None,
        *,
        native_connection: Any | None = None,
    ) -> Dialect:
        """Construct the dialect implementation for an engine type."""
        if engine_type is None:
            identity = active_engine_identity()
            engine_type = identity.engine_type
        if config is None:
            resolved_identity: EngineIdentity | None
            try:
                resolved_identity = active_engine_identity()
            except RuntimeError:
                resolved_identity = None
            if (
                resolved_identity is not None
                and str(resolved_identity.engine_type).strip().lower() == str(engine_type).strip().lower()
            ):
                config = resolved_identity.runtime_config
            else:
                config = cls.get_runtime_config_class(engine_type)
        if isinstance(config, type):
            config = EngineRuntimeConfig.process_default_for_class(config)
        cls._ensure_engine_registered(str(engine_type))
        if engine_type not in cls._DIALECT_REGISTRY:
            raise ValueError(f"Unsupported dialect: {engine_type}")
        ctor = cls._DIALECT_REGISTRY[engine_type]
        if engine_type in EMBEDDED_ENGINE_NAMES:
            return cast(_EmbeddedDialectCtor, ctor)(
                config, sqlalchemy_engine=sqlalchemy_engine, native_connection=native_connection
            )
        if sqlalchemy_engine is not None:
            return cast(_SqlalchemyDialectCtor, ctor)(config, sqlalchemy_engine=sqlalchemy_engine)
        return cast(_BaseDialectCtor, ctor)(config)

    get = get_dialect
    get_class = get_dialect_class
    register = register_dialect
    derive_surfaces = derive_dialect_registry_surfaces
