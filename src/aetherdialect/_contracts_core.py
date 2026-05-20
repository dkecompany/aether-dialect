"""Intent/template models: normalized expressions, runtime/concrete intents, filters, CTEs, and conversions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any, ClassVar, Literal, NamedTuple, Protocol

from ._config import (
    AGG_PATTERN,
    FILTER_VALUE_TYPE_DATE_DIFF,
    FILTER_VALUE_TYPE_DATE_WINDOW,
    OP_FLIP,
    REGISTRY_REF_TOKEN_RE,
    GenerationPath,
    SeedWarmupConfig,
    WINDOW_REGISTRY_AGG_KIND_HINTS,
    WINDOW_REGISTRY_NAV_KIND_HINTS,
    WINDOW_REGISTRY_RANK_KIND_HINTS,
)
from ._contracts_base import (
    ComplexityTier,
    CteOutputColumnMeta,
    ExpansionMetadata,
    NoveltyBand,
    OperatorFeatureVector,
    QSimSkeleton,
    SQLShape,
    TemplateStats,
    WarmupStyle,
    WindowOperatorKind,
    WorkloadFamily,
)

_PARSE_EXPR_STRING_FN: Any = None
_RENDER_EXPR_SQL_FN: Any = None


def register_parse_expr_string(fn: Any) -> None:
    global _PARSE_EXPR_STRING_FN
    _PARSE_EXPR_STRING_FN = fn


def register_render_expr_sql(fn: Any) -> None:
    global _RENDER_EXPR_SQL_FN
    _RENDER_EXPR_SQL_FN = fn


def _parse_expr_string_for_json(s: str) -> NormalizedExpr:
    """
    Parse a JSON string field that contains a SQL expression into a ``NormalizedExpr``.

    Args:

        s: Non-empty expression text from the model.

    Returns:

        Parsed structure when the registered parser is available; otherwise a column ref leaf.
    """
    t = (s or "").strip()
    if not t:
        return NormalizedExpr()
    fn = _PARSE_EXPR_STRING_FN
    if fn is not None:
        return fn(t)
    return NormalizedExpr.from_column(t)


ScalarArg = str | int | float
ParamValue = str | int | float | bool | list[str | int | float]
RawValue = str | int | float | bool | list[str | int | float] | dict[str, str | int] | None

CteEmissionKind = Literal["join_table", "scalar_subquery"]
WindowFrameKind = Literal["rows", "range", "none"]


def _coerce_cte_emission(raw: Any) -> CteEmissionKind:
    """
    Normalize a stored emission string to a supported literal.

    Args:

        raw: Value from JSON or legacy payloads.

    Returns:

        ``join_table`` unless ``raw`` is exactly ``scalar_subquery``.
    """
    return "scalar_subquery" if raw == "scalar_subquery" else "join_table"


def _normalized_expr_from_stored_json(raw: Any) -> NormalizedExpr:
    """
    Coerce JSON or template `expr` payloads into a `NormalizedExpr`.

    Args:

        raw: String, dict, or existing `NormalizedExpr`.

    Returns:

        Normalised expression; empty expr if unsupported.
    """
    if isinstance(raw, str):
        return NormalizedExpr.from_column(raw)
    if isinstance(raw, dict):
        return NormalizedExpr.from_dict(raw)
    if isinstance(raw, NormalizedExpr):
        return raw
    return NormalizedExpr()


@dataclass
class ExprValue:
    """Parameterized literal value for expression arithmetic with param_key for template reuse."""

    value: float = 0.0
    param_key: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExprValue:
        """
        Create ExprValue from dictionary.

        Args:

            d: Dictionary with 'value' and 'param_key' keys, or a bare numeric value.

        Returns:

            Populated ExprValue instance.
        """
        if isinstance(d, int | float):
            return ExprValue(value=float(d))
        return ExprValue(value=d.get("value", 0.0), param_key=d.get("param_key", ""))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with 'value' and 'param_key' keys.
        """
        return {"value": self.value, "param_key": self.param_key}

    @property
    def signature_key(self) -> str:
        """
        Structural signature for template matching (ignores concrete value).

        Returns:

            Always the string `val` (parameterisation uses `param_key` elsewhere).
        """
        return "val"


def _coerce_mul_term(raw: Any) -> NormalizedExpr:
    """
    Coerce a multiply/divide list element to a `NormalizedExpr` leaf.

    Accepts a `NormalizedExpr` instance, a dict (round-trip), or a bare string. Bare strings that look like function calls or compound expressions are routed through the sqlglot-backed `parse_expr_string` parser to recover structural fields; simple identifiers become `column_ref` leaves.
    """
    if isinstance(raw, NormalizedExpr):
        return raw
    if isinstance(raw, dict):
        return NormalizedExpr.from_dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s == "*":
            return NormalizedExpr(star=True)
        if s == "":
            return NormalizedExpr()
        if s.upper().startswith("DISTINCT "):
            s = s[9:].strip()
        while s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            if not inner:
                break
            s = inner
        if "(" in s or " " in s:
            try:
                parsed = _PARSE_EXPR_STRING_FN(s)
                if (
                    parsed.add_groups
                    and len(parsed.add_groups) == 1
                    and not parsed.sub_groups
                    and not parsed.add_values
                    and not parsed.sub_values
                ):
                    g = parsed.add_groups[0]
                    if (
                        g.coefficient == 1.0
                        and not g.divide
                        and len(g.multiply) == 1
                        and not g.agg_func
                        and not g.scalar_func
                    ):
                        return g.multiply[0]
                return parsed
            except Exception:
                return NormalizedExpr(raw_sql=s)
        return NormalizedExpr(column_ref=s)
    return NormalizedExpr()


@dataclass
class MulGroup:
    """
    Single multiplicative term: scalar_func(agg_func(inner_scalar_func(coefficient * multiply[0] * ...

    / divide[0] / ...))) with scalar_func_args and inner_scalar_func_args.

    `multiply` and `divide` carry nested `NormalizedExpr` sub-trees (column refs are leaf NormalizedExpr with `column_ref` set; CAST/COALESCE/EXTRACT/INTERVAL/ keyword nodes use the structural fields on `NormalizedExpr`).

    When `scalar_func` is ``concat``, `multiply` is an ordered list of CONCAT arguments rendered comma-separated inside ``CONCAT(...)``; `divide` and non-unit coefficients must remain empty. Otherwise `multiply` is a multiplicative chain rendered with ``*``.
    """

    coefficient: float = 1.0
    multiply: list[NormalizedExpr] = field(default_factory=list)
    divide: list[NormalizedExpr] = field(default_factory=list)
    agg_func: str | None = None
    scalar_func: str | None = None
    inner_scalar_func: str | None = None
    scalar_func_args: list[ScalarArg] = field(default_factory=list)
    inner_scalar_func_args: list[ScalarArg] = field(default_factory=list)
    coeff_param_key: str = ""
    sarg_param_keys: list[str] = field(default_factory=list)
    isarg_param_keys: list[str] = field(default_factory=list)
    distinct: bool = False

    def __post_init__(self) -> None:
        """
        Coerce string entries to leaf NormalizedExpr, sort multiply/divide, and
        normalise function name casing/order.
        """
        self.multiply = sorted(
            (_coerce_mul_term(t) for t in self.multiply),
            key=lambda e: e.signature_key,
        )
        self.divide = sorted(
            (_coerce_mul_term(t) for t in self.divide),
            key=lambda e: e.signature_key,
        )
        if self.agg_func:
            self.agg_func = self.agg_func.lower()
        if self.scalar_func:
            self.scalar_func = self.scalar_func.lower()
        if self.inner_scalar_func:
            self.inner_scalar_func = self.inner_scalar_func.lower()
        if self.scalar_func and self.inner_scalar_func:
            if self.scalar_func == "extract":
                pass
            elif self.inner_scalar_func == "extract":
                self.scalar_func, self.inner_scalar_func = (
                    self.inner_scalar_func,
                    self.scalar_func,
                )
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
            elif self.scalar_func > self.inner_scalar_func:
                self.scalar_func, self.inner_scalar_func = (
                    self.inner_scalar_func,
                    self.scalar_func,
                )
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> MulGroup:
        """
        Create MulGroup from dictionary; multiply/divide entries may be dicts (new
        nested form) or strings (legacy column ref) — both are accepted on read,
        always serialized as dicts on write.
        """
        return MulGroup(
            coefficient=d.get("coefficient", 1.0),
            multiply=[_coerce_mul_term(t) for t in d.get("multiply", [])],
            divide=[_coerce_mul_term(t) for t in d.get("divide", [])],
            agg_func=d.get("agg_func"),
            scalar_func=d.get("scalar_func"),
            inner_scalar_func=d.get("inner_scalar_func"),
            scalar_func_args=d.get("scalar_func_args", []),
            inner_scalar_func_args=d.get("inner_scalar_func_args", []),
            coeff_param_key=d.get("coeff_param_key", ""),
            sarg_param_keys=d.get("sarg_param_keys", []),
            isarg_param_keys=d.get("isarg_param_keys", []),
            distinct=bool(d.get("distinct", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all `MulGroup` fields; multiply/divide are nested dicts."""
        out = {
            "coefficient": self.coefficient,
            "multiply": [m.to_dict() for m in self.multiply],
            "divide": [d.to_dict() for d in self.divide],
            "agg_func": self.agg_func,
            "scalar_func": self.scalar_func,
            "inner_scalar_func": self.inner_scalar_func,
            "scalar_func_args": self.scalar_func_args,
            "inner_scalar_func_args": self.inner_scalar_func_args,
            "coeff_param_key": self.coeff_param_key,
            "sarg_param_keys": self.sarg_param_keys,
            "isarg_param_keys": self.isarg_param_keys,
        }
        if self.distinct:
            out["distinct"] = True
        return out

    @property
    def signature_key(self) -> str:
        """Pipe-separated structural key (recurses through nested multiply/divide)."""
        parts = ["coeff"]
        if self.distinct:
            parts.append("distinct")
        if self.agg_func:
            parts.append(f"agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"*{m.signature_key}" for m in self.multiply)
        parts.extend(f"/{d.signature_key}" for d in self.divide)
        return "|".join(parts)

    @property
    def structural_key(self) -> str:
        """Like `signature_key` but omits the coefficient marker."""
        parts: list[str] = []
        if self.distinct:
            parts.append("distinct")
        if self.agg_func:
            parts.append(f"agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"*{m.signature_key}" for m in self.multiply)
        parts.extend(f"/{d.signature_key}" for d in self.divide)
        return "|".join(parts)


_RAW_SQL_AGG_OR_WINDOW_RE = re.compile(
    r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(|OVER\s*\(",
    re.IGNORECASE,
)


@dataclass
class NormalizedExpr:
    """
    Canonical sum-of-products expression: scalar_func(agg_func(inner_scalar_func(sum of add_groups minus sub_groups plus add_values minus sub_values))) with scalar_func_args and inner_scalar_func_args.

    Structural leaf forms (mutually exclusive with add_groups/sub_groups when set): - column_ref: bare or qualified column reference (`"t.c"`). - star: True for the SQL `*` token. - cast_type: when set, this expression is `CAST(<inner> AS cast_type)` where `<inner>` is the single child reachable via add_groups[0].multiply[0]. - interval: `(magnitude, unit)` for SQL `INTERVAL '<n>' <unit>`. - keyword: bare SQL keyword like ``current_date``.
    """

    add_groups: list[MulGroup] = field(default_factory=list)
    sub_groups: list[MulGroup] = field(default_factory=list)
    add_values: list[ExprValue] = field(default_factory=list)
    sub_values: list[ExprValue] = field(default_factory=list)
    agg_func: str | None = None
    scalar_func: str | None = None
    inner_scalar_func: str | None = None
    scalar_func_args: list[ScalarArg] = field(default_factory=list)
    inner_scalar_func_args: list[ScalarArg] = field(default_factory=list)
    sarg_param_keys: list[str] = field(default_factory=list)
    isarg_param_keys: list[str] = field(default_factory=list)
    is_numeric: bool = True
    column_ref: str | None = None
    star: bool = False
    cast_type: str | None = None
    interval: tuple[float, str] | None = None
    keyword: str | None = None
    raw_sql: str | None = None
    string_literal: str = ""

    def __post_init__(self) -> None:
        """
        Sort child groups/values and normalise outer function name casing/order.

        Returns:

            None.
        """
        self.add_groups = sorted(self.add_groups, key=lambda g: g.signature_key)
        self.sub_groups = sorted(self.sub_groups, key=lambda g: g.signature_key)
        self.add_values = sorted(self.add_values, key=lambda v: v.value)
        self.sub_values = sorted(self.sub_values, key=lambda v: v.value)
        if self.agg_func:
            self.agg_func = self.agg_func.lower()
        if self.scalar_func:
            self.scalar_func = self.scalar_func.lower()
        if self.inner_scalar_func:
            self.inner_scalar_func = self.inner_scalar_func.lower()
        if self.scalar_func and self.inner_scalar_func:
            if self.scalar_func == "extract":
                pass
            elif self.inner_scalar_func == "extract":
                self.scalar_func, self.inner_scalar_func = (
                    self.inner_scalar_func,
                    self.scalar_func,
                )
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
            elif self.scalar_func > self.inner_scalar_func:
                self.scalar_func, self.inner_scalar_func = (
                    self.inner_scalar_func,
                    self.scalar_func,
                )
                self.scalar_func_args, self.inner_scalar_func_args = (
                    self.inner_scalar_func_args,
                    self.scalar_func_args,
                )
        if self.column_ref is not None:
            self.column_ref = str(self.column_ref).strip() or None
        if self.keyword is not None:
            self.keyword = str(self.keyword).strip().lower() or None
        if self.cast_type is not None:
            self.cast_type = str(self.cast_type).strip() or None
        if self.interval is not None:
            mag, unit = self.interval
            self.interval = (float(mag), str(unit).strip())
        if self.string_literal is not None:
            self.string_literal = str(self.string_literal).strip()
        if self.string_literal:
            self.column_ref = None
            self.raw_sql = None
            self.star = False
            self.keyword = None
            self.cast_type = None
            self.interval = None
            self.add_groups = []
            self.sub_groups = []
            self.add_values = []
            self.sub_values = []
            self.agg_func = None
            self.scalar_func = None
            self.inner_scalar_func = None
            self.scalar_func_args = []
            self.inner_scalar_func_args = []
            self.sarg_param_keys = []
            self.isarg_param_keys = []

    @staticmethod
    def from_dict(d: Any) -> NormalizedExpr:
        """Create NormalizedExpr from a dictionary, a column-reference string, or ``None``."""
        if d is None:
            return NormalizedExpr()
        if isinstance(d, str):
            return NormalizedExpr.from_column(d.strip())
        if isinstance(d, dict):
            s_lit = d.get("string_literal")
            if isinstance(s_lit, str) and s_lit.strip():
                return NormalizedExpr(string_literal=s_lit.strip())
            lit_plain = d.get("literal")
            if isinstance(lit_plain, str) and lit_plain.strip():
                return NormalizedExpr(string_literal=lit_plain.strip())
        column_ref_raw = d.get("column_ref")
        if column_ref_raw is None:
            legacy_ref = d.get("registry_ref")
            if isinstance(legacy_ref, str) and legacy_ref.strip():
                column_ref_raw = legacy_ref.strip()
        iv_raw = d.get("interval")
        iv: tuple[float, str] | None = None
        if isinstance(iv_raw, list | tuple) and len(iv_raw) == 2:
            iv = (float(iv_raw[0]), str(iv_raw[1]))
        return NormalizedExpr(
            add_groups=[MulGroup.from_dict(g) for g in d.get("add_groups", [])],
            sub_groups=[MulGroup.from_dict(g) for g in d.get("sub_groups", [])],
            add_values=[ExprValue.from_dict(v) for v in d.get("add_values", [])],
            sub_values=[ExprValue.from_dict(v) for v in d.get("sub_values", [])],
            agg_func=d.get("agg_func"),
            scalar_func=d.get("scalar_func"),
            inner_scalar_func=d.get("inner_scalar_func"),
            scalar_func_args=d.get("scalar_func_args", []),
            inner_scalar_func_args=d.get("inner_scalar_func_args", []),
            sarg_param_keys=d.get("sarg_param_keys", []),
            isarg_param_keys=d.get("isarg_param_keys", []),
            is_numeric=d.get("is_numeric", True),
            column_ref=column_ref_raw,
            star=bool(d.get("star", False)),
            cast_type=d.get("cast_type"),
            interval=iv,
            keyword=d.get("keyword"),
            raw_sql=d.get("raw_sql"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        out: dict[str, Any] = {
            "add_groups": [g.to_dict() for g in self.add_groups],
            "sub_groups": [g.to_dict() for g in self.sub_groups],
            "add_values": [v.to_dict() for v in self.add_values],
            "sub_values": [v.to_dict() for v in self.sub_values],
            "agg_func": self.agg_func,
            "scalar_func": self.scalar_func,
            "inner_scalar_func": self.inner_scalar_func,
            "scalar_func_args": self.scalar_func_args,
            "inner_scalar_func_args": self.inner_scalar_func_args,
            "sarg_param_keys": self.sarg_param_keys,
            "isarg_param_keys": self.isarg_param_keys,
            "is_numeric": self.is_numeric,
        }
        if self.column_ref:
            out["column_ref"] = self.column_ref
        if self.star:
            out["star"] = True
        if self.cast_type:
            out["cast_type"] = self.cast_type
        if self.interval is not None:
            out["interval"] = [self.interval[0], self.interval[1]]
        if self.keyword:
            out["keyword"] = self.keyword
        if self.raw_sql:
            out["raw_sql"] = self.raw_sql
        if self.string_literal:
            out["string_literal"] = self.string_literal
        return out

    @staticmethod
    def from_column(col: str) -> NormalizedExpr:
        """Build a leaf NormalizedExpr that references a single column (or `*`)."""
        s = col.strip()
        if s == "*":
            return NormalizedExpr(star=True)
        return NormalizedExpr(column_ref=s)

    @staticmethod
    def from_agg(agg_func: str, col: str) -> NormalizedExpr:
        """Build a NormalizedExpr for `agg_func(column)` with the column as a leaf child."""
        leaf = NormalizedExpr.from_column(col)
        return NormalizedExpr(add_groups=[MulGroup(multiply=[leaf], agg_func=agg_func.lower())])

    @property
    def signature_key(self) -> str:
        """Pipe-separated key over outer funcs, structural leaf info, and signed groups/values."""
        parts: list[str] = []
        if self.column_ref:
            parts.append(f"col={self.column_ref}")
        if self.star:
            parts.append("star")
        if self.keyword:
            parts.append(f"kw={self.keyword}")
        if self.cast_type:
            parts.append(f"cast={self.cast_type}")
        if self.interval is not None:
            parts.append(f"iv={self.interval[0]}:{self.interval[1]}")
        if self.raw_sql:
            parts.append(f"raw={self.raw_sql}")
        if self.string_literal:
            parts.append(f"strlit={self.string_literal!r}")
        if self.agg_func:
            parts.append(f"expr_agg={self.agg_func}")
        if self.scalar_func:
            parts.append(f"expr_scalar={self.scalar_func}")
        if self.scalar_func_args:
            parts.append(f"expr_sargs={len(self.scalar_func_args)}")
        if self.inner_scalar_func:
            parts.append(f"expr_inner={self.inner_scalar_func}")
        if self.inner_scalar_func_args:
            parts.append(f"expr_iargs={len(self.inner_scalar_func_args)}")
        parts.extend(f"+{g.signature_key}" for g in self.add_groups)
        parts.extend(f"-{g.signature_key}" for g in self.sub_groups)
        parts.extend(f"+{v.signature_key}" for v in self.add_values)
        parts.extend(f"-{v.signature_key}" for v in self.sub_values)
        return "|".join(parts)

    @property
    def has_column_reference(self) -> bool:
        """Return True when this expression references any column, aggregate, scalar, or registry entry."""
        if self.string_literal:
            return False
        if self.raw_sql:
            return True
        if self.column_ref or self.star or self.keyword or self.cast_type or self.interval is not None:
            if self.column_ref:
                return True
            if self.cast_type:
                return True
            if self.star or self.keyword:
                return True
            if self.interval is not None:
                return True
        if self.add_groups or self.sub_groups:
            return True
        if self.agg_func or self.scalar_func or self.inner_scalar_func:
            return True
        return False

    @property
    def is_literal_only(self) -> bool:
        """Return True when this expression is composed solely of numeric literals."""
        return not self.has_column_reference

    @property
    def has_aggregation(self) -> bool:
        """Whether any subterm uses SQL aggregation (outer or per-`MulGroup`)."""
        if self.agg_func:
            return True
        raw_sql = self.raw_sql
        if raw_sql and _RAW_SQL_AGG_OR_WINDOW_RE.search(raw_sql):
            return True
        for group in self.add_groups + self.sub_groups:
            if group.agg_func:
                return True
            for term in group.multiply + group.divide:
                if term.has_aggregation:
                    return True
        return False

    @property
    def primary_column(self) -> str:
        """
        Innermost column name reached by drilling into the first multiplicative term.
        Strips DISTINCT, walks through cast/scalar wrappers, returns "" when no column.
        """
        if self.string_literal:
            return ""
        if self.column_ref:
            return self.column_ref
        if self.star:
            return "*"
        if self.keyword:
            return self.keyword
        if self.interval is not None:
            return "interval"
        if self.raw_sql:
            return ""
        if not self.add_groups or not self.add_groups[0].multiply:
            return ""
        first = self.add_groups[0].multiply[0]
        return first.primary_column

    @property
    def primary_term(self) -> str:
        """
        First multiply operand of the first `add_groups` entry rendered as a token string.

        Returns the leaf `column_ref` for a column reference, ``"*"`` for star, the upper-cased `keyword` for a keyword leaf, or empty when no add_groups exist or the leaf is a complex sub-tree (cast/coalesce/case/interval).
        """
        if self.column_ref:
            return self.column_ref
        if self.star:
            return "*"
        if self.keyword:
            return self.keyword.upper()
        if not self.add_groups or not self.add_groups[0].multiply:
            return ""
        first = self.add_groups[0].multiply[0]
        if first.column_ref:
            return first.column_ref
        if first.star:
            return "*"
        if first.keyword:
            return first.keyword.upper()
        try:
            return _RENDER_EXPR_SQL_FN(first)
        except Exception:
            return ""

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": "SQL expression text using qualified columns from the schema.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """
        Shorthand ``expr`` string for LLM-facing JSON.
        """

        return {"expr": _expr_prompt_sql(self)}

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Canonical ``expr`` field example for prompts."""

        return {"expr": "tbl_a.col_a"}


def expr_registry_ref(expr: NormalizedExpr) -> str | None:
    """
    Return the canonical registry id when *expr* is a bare ``column_ref`` matching ``^[wc]\\d{2}$``.

    A registry reference is conventionally encoded as a single bare ``column_ref`` with no other expression complexity. The rest of the system treats this leaf shape as the canonical way to point at a window or case registry entry from select, group_by, order_by, filter, or having.
    """
    if expr.string_literal:
        return None
    col = (expr.column_ref or "").strip()
    if not col or not REGISTRY_REF_TOKEN_RE.match(col):
        return None
    if expr.add_groups or expr.sub_groups or expr.add_values or expr.sub_values:
        return None
    if expr.agg_func or expr.scalar_func or expr.inner_scalar_func:
        return None
    if expr.star or expr.cast_type or expr.interval is not None:
        return None
    if expr.keyword or expr.raw_sql:
        return None
    return col


def _expr_prompt_sql(expr: NormalizedExpr) -> str:
    """
    Render *expr* as the shorthand SQL string shown in LLM prompts.

    Registry references ``wNN`` / ``cNN`` emit as bare tokens; other expressions use the
    registered renderer when available.
    """

    ref = expr_registry_ref(expr)
    if ref:
        return ref
    if expr.string_literal:
        return expr.string_literal
    fn = _RENDER_EXPR_SQL_FN
    if fn is not None:
        try:
            return fn(expr)
        except Exception:
            pass
    col = expr.primary_column
    return col if col else ""


def _canonicalize_predicate_sides(predicate: Any) -> None:
    """
    Enforce column-bearing side on the left and flip the operator when a swap is required.

    When exactly one of ``left_expr`` / ``right_expr`` contains column / aggregate / scalar / registry references, that side is moved to ``left_expr`` and the operator is flipped. When both sides are column-bearing or both are literal-only, sides are left untouched.
    """

    left = predicate.left_expr
    right = predicate.right_expr
    if right is None:
        return
    left_has_col = left.has_column_reference
    right_has_col = right.has_column_reference
    if left_has_col and not right_has_col:
        return
    if right_has_col and not left_has_col:
        predicate.left_expr, predicate.right_expr = right, left
        predicate.op = OP_FLIP.get(predicate.op, predicate.op)


def _filter_group_int_from_stored(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, list | tuple):
        if not raw:
            return None
        raw = raw[0]
    if isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class FilterParam:
    """Filter condition with left expression, operator, and optional right expression for expr-vs-expr comparisons."""

    left_expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    op: str = "="
    right_expr: NormalizedExpr | None = None
    value_type: str = "string"
    param_key: str = ""
    param_key_hi: str = ""
    param_key_unit: str = ""
    raw_value: RawValue = None
    bool_op: str = "AND"
    filter_group: int | None = None

    def __post_init__(self) -> None:
        """
        Normalise operators/types, canonicalise expr-vs-expr sides, merge literals to the value side.

        Returns:

            None.
        """
        self.op = self.op.strip().lower()
        self.value_type = self.value_type.strip().lower()
        self.bool_op = self.bool_op.strip().upper() if self.bool_op else "AND"
        if self.bool_op not in ("AND", "OR"):
            self.bool_op = "AND"
        if self.right_expr is not None:
            _canonicalize_predicate_sides(self)
            for ev in self.left_expr.add_values:
                self.right_expr.sub_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            for ev in self.left_expr.sub_values:
                self.right_expr.add_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            self.left_expr.add_values = []
            self.left_expr.sub_values = []
        elif (
            self.raw_value is not None
            and isinstance(self.raw_value, int | float)
            and not isinstance(self.raw_value, bool)
        ):
            offset = sum(ev.value for ev in self.left_expr.add_values) - sum(
                ev.value for ev in self.left_expr.sub_values
            )
            self.raw_value = self.raw_value - offset
            self.left_expr.add_values = []
            self.left_expr.sub_values = []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FilterParam:
        """
        Create FilterParam from dictionary.

        Args:

            d: Dictionary with 'left_expr', 'op', optional 'right_expr', 'value_type', and 'param_key'.

        Returns:

            Populated FilterParam instance.
        """
        left_raw = d.get("left_expr", {})
        right_raw = d.get("right_expr")
        fg_raw = d.get("filter_group")
        return FilterParam(
            left_expr=_normalized_expr_from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(_normalized_expr_from_stored_json(right_raw) if right_raw else None),
            value_type=d.get("value_type", "string"),
            param_key=d.get("param_key", ""),
            param_key_hi=d.get("param_key_hi", ""),
            param_key_unit=d.get("param_key_unit", ""),
            raw_value=d.get("value") or d.get("raw_value"),
            bool_op=d.get("bool_op", "AND"),
            filter_group=_filter_group_int_from_stored(fg_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with filter fields; raw_value is intentionally excluded.
        """
        d: dict[str, Any] = {
            "left_expr": self.left_expr.to_dict(),
            "op": self.op,
            "right_expr": self.right_expr.to_dict() if self.right_expr else None,
            "value_type": self.value_type,
            "param_key": self.param_key,
        }
        if self.param_key_hi:
            d["param_key_hi"] = self.param_key_hi
        if self.param_key_unit:
            d["param_key_unit"] = self.param_key_unit
        if self.bool_op != "AND":
            d["bool_op"] = self.bool_op
        if self.filter_group is not None:
            d["filter_group"] = self.filter_group
        return d

    @property
    def signature_key(self) -> str:
        """
        Structural key for WHERE-style template matching.

        Returns:

            Left expr, op, value type, and optional right expr signature joined by `|`.
        """
        parts = [self.left_expr.signature_key, self.op, self.value_type]
        if self.right_expr:
            parts.append(f"r:{self.right_expr.signature_key}")
        return "|".join(parts)

    def resolved_value(self, param_values: Mapping[str, Any] | None) -> Any:
        """
        Resolve the filter literal from inline storage or bound parameters.

        After post-processing, ``raw_value`` may be cleared while ``param_key`` still identifies the bound slot in the owning body ``param_values`` map.

        Args:

            param_values: Bound parameter map; treated as empty when ``None``.

        Returns:

            ``raw_value`` when set; otherwise ``param_values[param_key]`` when ``param_key`` is non-empty; otherwise ``None``.
        """

        if self.raw_value is not None:
            return self.raw_value
        store = param_values or {}
        pk = (self.param_key or "").strip()
        pku = (self.param_key_unit or "").strip()
        vt = (self.value_type or "").lower()
        if vt == "date_diff" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                u = unit if isinstance(unit, str) and unit else "day"
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                return {"unit": u, "amount": a}
        if vt == "date_window" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                u = unit if isinstance(unit, str) and unit else "day"
                return {"unit": u, "amount": a}
        if not pk:
            return None
        return store.get(pk)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "left_expr": "SQL expression for the predicate left side using qualified columns.",
        "op": "Comparison or membership operator (lowercase).",
        "right_expr": "Optional SQL expression for expr-vs-expr predicates.",
        "value_type": "Semantic type for expr-vs-value predicates.",
        "value": "Inline literal or structured date_window or date_diff payload.",
        "bool_op": "AND or OR connecting to the next filter entry.",
        "filter_group": "Optional integer for OR-of-AND grouping.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """LLM shorthand dict with SQL strings for expression sides."""

        out: dict[str, Any] = {
            "left_expr": _expr_prompt_sql(self.left_expr),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = _expr_prompt_sql(self.right_expr)
        elif self.raw_value is not None:
            out["value"] = self.raw_value
        if self.bool_op != "AND":
            out["bool_op"] = self.bool_op
        if self.filter_group is not None:
            out["filter_group"] = self.filter_group
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example WHERE predicate shape for prompts."""

        return {
            "left_expr": "tbl_a.col_b",
            "op": "=",
            "value_type": "string",
            "value": "<literal>",
        }


@dataclass
class HavingParam:
    """Having condition with left expression, operator, and optional right expression for expr-vs-expr comparisons."""

    left_expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    op: str = "="
    right_expr: NormalizedExpr | None = None
    value_type: str = "number"
    param_key: str = ""
    param_key_unit: str = ""
    raw_value: RawValue = None
    bool_op: str = "AND"
    filter_group: int | None = None

    def __post_init__(self) -> None:
        """
        Normalise operators/types, canonicalise expr-vs-expr sides, merge literals to the value side.

        Returns:

            None.
        """
        self.op = self.op.strip().lower()
        self.value_type = self.value_type.strip().lower()
        self.bool_op = self.bool_op.strip().upper() if self.bool_op else "AND"
        if self.bool_op not in ("AND", "OR"):
            self.bool_op = "AND"
        if self.right_expr is not None:
            _canonicalize_predicate_sides(self)
            for ev in self.left_expr.add_values:
                self.right_expr.sub_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            for ev in self.left_expr.sub_values:
                self.right_expr.add_values.append(ExprValue(value=ev.value, param_key=ev.param_key))
            self.left_expr.add_values = []
            self.left_expr.sub_values = []
        elif (
            self.raw_value is not None
            and isinstance(self.raw_value, int | float)
            and not isinstance(self.raw_value, bool)
        ):
            offset = sum(ev.value for ev in self.left_expr.add_values) - sum(
                ev.value for ev in self.left_expr.sub_values
            )
            self.raw_value = self.raw_value - offset
            self.left_expr.add_values = []
            self.left_expr.sub_values = []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HavingParam:
        """
        Create HavingParam from dictionary.

        Args:

            d: Dictionary with 'left_expr', 'op', optional 'right_expr', 'value_type', and 'param_key'.

        Returns:

            Populated HavingParam instance.
        """
        left_raw = d.get("left_expr", {})
        right_raw = d.get("right_expr")
        fg_raw = d.get("filter_group")
        return HavingParam(
            left_expr=_normalized_expr_from_stored_json(left_raw),
            op=d.get("op", "="),
            right_expr=(_normalized_expr_from_stored_json(right_raw) if right_raw else None),
            value_type=d.get("value_type", "number"),
            param_key=d.get("param_key", ""),
            param_key_unit=d.get("param_key_unit", ""),
            raw_value=d.get("value") or d.get("raw_value"),
            bool_op=d.get("bool_op", "AND"),
            filter_group=_filter_group_int_from_stored(fg_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize HAVING parameters; omits `raw_value` (like `FilterParam.to_dict`).

        Returns:

            Dict of exprs, op, types, and optional `bool_op` / `filter_group`.
        """
        d: dict[str, Any] = {
            "left_expr": self.left_expr.to_dict(),
            "op": self.op,
            "right_expr": self.right_expr.to_dict() if self.right_expr else None,
            "value_type": self.value_type,
            "param_key": self.param_key,
        }
        if self.param_key_unit:
            d["param_key_unit"] = self.param_key_unit
        if self.bool_op != "AND":
            d["bool_op"] = self.bool_op
        if self.filter_group is not None:
            d["filter_group"] = self.filter_group
        return d

    @property
    def signature_key(self) -> str:
        """
        Structural key for HAVING-style template matching.

        Returns:

            Same pipe-joined pattern as `FilterParam.signature_key`.
        """
        parts = [self.left_expr.signature_key, self.op, self.value_type]
        if self.right_expr:
            parts.append(f"r:{self.right_expr.signature_key}")
        return "|".join(parts)

    def resolved_value(self, param_values: Mapping[str, Any] | None) -> Any:
        """
        Resolve the HAVING literal from inline storage or bound parameters.

        After post-processing, ``raw_value`` may be cleared while ``param_key`` still identifies the bound slot in the owning body ``param_values`` map.

        Args:

            param_values: Bound parameter map; treated as empty when ``None``.

        Returns:

            ``raw_value`` when set; otherwise ``param_values[param_key]`` when ``param_key`` is non-empty; otherwise ``None``.
        """

        if self.raw_value is not None:
            return self.raw_value
        store = param_values or {}
        pk = (self.param_key or "").strip()
        pku = (self.param_key_unit or "").strip()
        vt = (self.value_type or "").lower()
        if vt == "date_diff" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                u = unit if isinstance(unit, str) and unit else "day"
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                return {"unit": u, "amount": a}
        if vt == "date_window" and pk and pku:
            amt = store.get(pk)
            unit = store.get(pku)
            if amt is not None or unit is not None:
                a = int(amt) if amt is not None and not isinstance(amt, bool) else 0
                u = unit if isinstance(unit, str) and unit else "day"
                return {"unit": u, "amount": a}
        if not pk:
            return None
        return store.get(pk)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "left_expr": "Aggregate or grouped SQL expression on the left side.",
        "op": "Comparison operator for aggregate predicates.",
        "right_expr": "Optional SQL expression for agg-vs-agg predicates.",
        "value_type": "Semantic type for agg-vs-value predicates.",
        "value": "Numeric or structured literal compared to the left aggregate.",
        "bool_op": "AND or OR connecting to the next HAVING entry.",
        "filter_group": "Optional integer grouping OR-of-AND conjunct blocks.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """LLM shorthand dict with SQL strings for HAVING sides."""

        out: dict[str, Any] = {
            "left_expr": _expr_prompt_sql(self.left_expr),
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expr is not None:
            out["right_expr"] = _expr_prompt_sql(self.right_expr)
        elif self.raw_value is not None:
            out["value"] = self.raw_value
        if self.bool_op != "AND":
            out["bool_op"] = self.bool_op
        if self.filter_group is not None:
            out["filter_group"] = self.filter_group
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example HAVING predicate shape for prompts."""

        return {
            "left_expr": "COUNT(tbl_a.col_a)",
            "op": ">",
            "value_type": "integer",
            "value": 1,
        }


@dataclass
class WindowSpec:
    """Window function specification for a SELECT column (dialect- agnostic)."""

    function: str
    partition_by: list[NormalizedExpr] = field(default_factory=list)
    order_by: list[OrderByCol] = field(default_factory=list)
    argument: NormalizedExpr | None = None
    frame_kind: WindowFrameKind = "none"
    frame_start: str | None = None
    frame_end: str | None = None
    frame_start_offset: int | None = None
    frame_end_offset: int | None = None

    def __post_init__(self) -> None:
        """
        Strip and lower-case `function`.

        Returns:

            None.
        """
        self.function = self.function.strip().lower()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> WindowSpec:
        """
        Parse a window spec from JSON-compatible dicts and strings.

        Args:

            d: Mapping with `function`, optional `partition_by`, `order_by`, `argument`.

        Returns:

            Populated `WindowSpec` with nested `NormalizedExpr` / `OrderByCol` objects.
        """
        part_raw = d.get("partition_by", [])
        partition_by: list[NormalizedExpr] = []
        for p in part_raw:
            if isinstance(p, dict):
                partition_by.append(NormalizedExpr.from_dict(p))
            elif isinstance(p, str):
                partition_by.append(_parse_expr_string_for_json(p))
        ob_raw = d.get("order_by", [])
        order_by: list[OrderByCol] = []
        for o in ob_raw:
            if isinstance(o, dict):
                order_by.append(OrderByCol.from_dict(o))
            elif isinstance(o, str):
                order_by.append(OrderByCol(expr=_parse_expr_string_for_json(o)))
        arg_raw = d.get("argument")
        argument = None
        if isinstance(arg_raw, dict):
            argument = NormalizedExpr.from_dict(arg_raw)
        elif isinstance(arg_raw, str) and arg_raw:
            argument = _parse_expr_string_for_json(arg_raw)
        fk_raw = (d.get("frame_kind") or "none").strip().lower()
        frame_kind: WindowFrameKind = "rows" if fk_raw == "rows" else ("range" if fk_raw == "range" else "none")
        fs = d.get("frame_start")
        fe = d.get("frame_end")
        fso = d.get("frame_start_offset")
        feo = d.get("frame_end_offset")

        def _off(x: Any) -> int | None:
            if isinstance(x, bool) or x is None:
                return None
            if isinstance(x, int):
                return x
            if isinstance(x, float) and x == int(x):
                return int(x)
            try:
                return int(x)
            except (TypeError, ValueError):
                return None

        return WindowSpec(
            function=d.get("function", ""),
            partition_by=partition_by,
            order_by=order_by,
            argument=argument,
            frame_kind=frame_kind,
            frame_start=str(fs).strip() if isinstance(fs, str) and fs.strip() else None,
            frame_end=str(fe).strip() if isinstance(fe, str) and fe.strip() else None,
            frame_start_offset=_off(fso),
            frame_end_offset=_off(feo),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize window function name, keys, and optional argument.

        Returns:

            JSON-friendly dict; `argument` omitted when unset.
        """
        out: dict[str, Any] = {
            "function": self.function,
            "partition_by": [p.to_dict() for p in self.partition_by],
            "order_by": [o.to_dict() for o in self.order_by],
            "frame_kind": self.frame_kind,
        }
        if self.argument is not None:
            out["argument"] = self.argument.to_dict()
        if self.frame_start is not None:
            out["frame_start"] = self.frame_start
        if self.frame_end is not None:
            out["frame_end"] = self.frame_end
        if self.frame_start_offset is not None:
            out["frame_start_offset"] = self.frame_start_offset
        if self.frame_end_offset is not None:
            out["frame_end_offset"] = self.frame_end_offset
        return out

    @property
    def signature_key(self) -> str:
        """
        Stable key over window function, partition, order, and argument exprs.

        Returns:

            Pipe-separated string prefixed with `win=`.
        """
        parts = [f"win={self.function}"]
        parts.extend(f"p:{e.signature_key}" for e in self.partition_by)
        parts.extend(f"o:{o.signature_key}" for o in self.order_by)
        if self.argument:
            parts.append(f"a:{self.argument.signature_key}")
        if self.frame_kind != "none":
            parts.append(f"fk={self.frame_kind}")
            if self.frame_start:
                parts.append(f"fs={self.frame_start}")
            if self.frame_end:
                parts.append(f"fe={self.frame_end}")
        return "|".join(parts)

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "function": "Lowercase window function name such as row_number, rank, sum, or lag.",
        "partition_by": "List of SQL expressions for PARTITION BY.",
        "order_by": "Ordered sort keys with direction inside the OVER clause.",
        "argument": "Inner SQL expression for windowed aggregates and offsets.",
        "frame_kind": "rows, range, or none when no explicit frame.",
        "frame_start": (
            "Frame start bound when frame_kind is rows or range (e.g. UNBOUNDED PRECEDING, CURRENT ROW, N PRECEDING)."
        ),
        "frame_end": (
            "Frame end bound when frame_kind is rows or range (e.g. CURRENT ROW, UNBOUNDED FOLLOWING, N FOLLOWING)."
        ),
        "frame_start_offset": "Integer row/range offset when the bound uses N PRECEDING or N FOLLOWING.",
        "frame_end_offset": "Integer row/range offset for the end bound when using N PRECEDING or N FOLLOWING.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Shorthand window definition for LLM JSON."""

        out: dict[str, Any] = {"function": self.function}
        if self.partition_by:
            out["partition_by"] = [_expr_prompt_sql(e) for e in self.partition_by]
        if self.order_by:
            out["order_by"] = [o.to_prompt_dict() for o in self.order_by]
        if self.argument is not None and self.argument.signature_key:
            out["argument"] = _expr_prompt_sql(self.argument)
        if self.frame_kind != "none":
            out["frame_kind"] = self.frame_kind
        if self.frame_start is not None:
            out["frame_start"] = self.frame_start
        if self.frame_end is not None:
            out["frame_end"] = self.frame_end
        if self.frame_start_offset is not None:
            out["frame_start_offset"] = self.frame_start_offset
        if self.frame_end_offset is not None:
            out["frame_end_offset"] = self.frame_end_offset
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example window_spec block for prompts."""

        return {
            "function": "row_number",
            "partition_by": [],
            "order_by": [{"expr": "tbl_a.col_a", "direction": "desc"}],
        }


_CASE_WHEN_QUALIFIED_COLUMN_REF_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$",
)


def _case_when_string_result_expr(value: str) -> NormalizedExpr:
    """
    Interpret a JSON string CASE THEN/ELSE token as either a column ref or a string literal.

    A single ``table.column`` token (two SQL identifiers separated by one dot) uses the
    column-reference path; every other non-empty stripped string becomes a string literal
    for SQL rendering.

    Args:

        value: Raw string from intent JSON.

    Returns:

        ``NormalizedExpr`` with ``column_ref`` or ``string_literal`` set, or empty when
        ``value`` is blank after stripping.
    """

    t = (value or "").strip()
    if not t:
        return NormalizedExpr()
    if _CASE_WHEN_QUALIFIED_COLUMN_REF_RE.fullmatch(t):
        return NormalizedExpr.from_column(t)
    return NormalizedExpr(string_literal=t)


@dataclass
class CaseWhenBranch:
    """Single WHEN branch for a CASE expression in SELECT."""

    condition: FilterParam = field(default_factory=FilterParam)
    result: NormalizedExpr = field(default_factory=NormalizedExpr)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseWhenBranch:
        """
        Parse one `WHEN ... THEN ...` branch from a dict.

        Args:

            d: Mapping with `condition` and `result` (dict or string for result).

        Returns:

            `CaseWhenBranch` with `FilterParam` and `NormalizedExpr`.
        """
        cond = d.get("condition", {})
        lit = d.get("literal_string")
        lit_top = d.get("literal")
        if isinstance(lit, str) and lit.strip():
            res_expr = NormalizedExpr(string_literal=lit.strip())
        elif isinstance(lit_top, str) and lit_top.strip():
            res_expr = NormalizedExpr(string_literal=lit_top.strip())
        else:
            res = d.get("result", {})
            if isinstance(res, dict):
                res_expr = NormalizedExpr.from_dict(res)
            elif isinstance(res, str) and res.strip():
                res_expr = _case_when_string_result_expr(str(res))
            else:
                res_expr = NormalizedExpr()
        return CaseWhenBranch(
            condition=replace(
                (FilterParam.from_dict(cond) if isinstance(cond, dict) else FilterParam()),
                bool_op="AND",
                filter_group=None,
            ),
            result=res_expr,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize condition and result expressions.

        Returns:

            Dict with `condition` and `result` sub-dicts.
        """
        out: dict[str, Any] = {"condition": self.condition.to_dict()}
        if self.result.string_literal:
            out["literal_string"] = self.result.string_literal
        else:
            out["result"] = self.result.to_dict()
        return out

    @property
    def signature_key(self) -> str:
        """
        Branch fingerprint for template deduplication.

        Returns:

            `condition_key=>result_key` string.
        """
        return f"{self.condition.signature_key}=>{self.result.signature_key}"

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "condition": "Row-level predicate object describing the WHEN clause.",
        "result": "SQL expression evaluated when the condition matches.",
        "literal_string": "Alternative to result: raw string literal for THEN (quoted in SQL).",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """One WHEN branch in shorthand LLM form."""

        out: dict[str, Any] = {"condition": self.condition.to_prompt_dict()}
        if self.result.string_literal:
            out["literal_string"] = self.result.string_literal
        else:
            out["result"] = _expr_prompt_sql(self.result)
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example CASE branch for prompts."""

        return {
            "condition": FilterParam.prompt_example_dict(),
            "result": "tbl_a.col_a",
        }


@dataclass
class CaseWhenExpr:
    """CASE expression for SELECT only."""

    branches: list[CaseWhenBranch] = field(default_factory=list)
    else_result: NormalizedExpr | None = None
    condition_scope: str = "filter"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseWhenExpr:
        """
        Parse a full `CASE` from JSON (branches plus optional `else_result`).

        Args:

            d: Mapping with `branches` list, optional `else_result`, and optional
            `condition_scope` (`"filter"` or `"having"`; defaults to `"filter"`).

        Returns:

            `CaseWhenExpr` with ordered branches and optional else expression.
        """
        br_raw = d.get("branches", [])
        branches = [CaseWhenBranch.from_dict(b) if isinstance(b, dict) else CaseWhenBranch() for b in br_raw]
        else_result = None
        else_lit = d.get("else_literal_string")
        if isinstance(else_lit, str) and else_lit.strip():
            else_result = NormalizedExpr(string_literal=else_lit.strip())
        else:
            er = d.get("else_result")
            if isinstance(er, dict):
                else_result = NormalizedExpr.from_dict(er)
            elif isinstance(er, str) and er:
                else_result = _case_when_string_result_expr(er)
        scope_raw = str(d.get("condition_scope", "filter")).strip().lower()
        scope = scope_raw if scope_raw in ("filter", "having") else "filter"
        return CaseWhenExpr(branches=branches, else_result=else_result, condition_scope=scope)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize all branches and optional else clause.

        Returns:

            Dict with `branches` list; `else_result` included when set; `condition_scope`
            included only when it differs from the default `"filter"`.
        """
        out: dict[str, Any] = {"branches": [b.to_dict() for b in self.branches]}
        if self.else_result is not None:
            if self.else_result.string_literal:
                out["else_literal_string"] = self.else_result.string_literal
            else:
                out["else_result"] = self.else_result.to_dict()
        if self.condition_scope and self.condition_scope != "filter":
            out["condition_scope"] = self.condition_scope
        return out

    @property
    def signature_key(self) -> str:
        """
        Full `CASE` structural fingerprint.

        Returns:

            `case|<scope>|` plus branch keys and optional `else:` suffix.
        """
        parts = [b.signature_key for b in self.branches]
        if self.else_result:
            parts.append(f"else:{self.else_result.signature_key}")
        return f"case|{self.condition_scope}|" + "|".join(parts)

    @property
    def has_aggregated_condition(self) -> bool:
        """Return True when any branch condition references a SQL aggregate."""
        for br in self.branches:
            cond = br.condition
            if cond is None:
                continue
            if cond.left_expr.has_aggregation:
                return True
            if cond.right_expr is not None and cond.right_expr.has_aggregation:
                return True
        return False

    @property
    def has_aggregated_output(self) -> bool:
        """Return True when any branch result or ELSE clause references a SQL aggregate."""
        for br in self.branches:
            if br.result.has_aggregation:
                return True
        if self.else_result is not None and self.else_result.has_aggregation:
            return True
        return False

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "branches": "Ordered WHEN branches each with condition and result strings.",
        "else_result": "Optional ELSE SQL expression.",
        "else_literal_string": "Optional ELSE raw string literal (quoted in SQL).",
        "condition_scope": "filter or having when branch predicates match that scope.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """CASE expression shorthand for LLM JSON."""

        out: dict[str, Any] = {
            "branches": [b.to_prompt_dict() for b in self.branches],
        }
        if self.else_result is not None:
            if self.else_result.string_literal:
                out["else_literal_string"] = self.else_result.string_literal
            else:
                er = _expr_prompt_sql(self.else_result)
                if er:
                    out["else_result"] = er
        if self.condition_scope != "filter":
            out["condition_scope"] = self.condition_scope
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Minimal CASE example for prompts."""

        return {
            "branches": [
                {
                    "condition": FilterParam.prompt_example_dict(),
                    "result": "tbl_a.col_a",
                }
            ],
            "else_result": None,
        }


_REGISTRY_WIN: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_WIN", default=())
_REGISTRY_CASE: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_CASE", default=())


@contextmanager
def registry_render_scope(
    window_registry: list[WindowRegistryStep] | None,
    case_registry: list[CaseRegistryStep] | None,
) -> Iterator[None]:
    """
    Bind window/case registry lists for the current thread while rendering or analysing expressions.

    Select columns that use ``registry_ref`` resolve through these lists when computing ``is_aggregated`` or emitting SQL.
    """

    t_w = _REGISTRY_WIN.set(tuple(window_registry or ()))
    t_c = _REGISTRY_CASE.set(tuple(case_registry or ()))
    try:
        yield
    finally:
        _REGISTRY_WIN.reset(t_w)
        _REGISTRY_CASE.reset(t_c)


def current_window_registry_steps() -> tuple[WindowRegistryStep, ...]:
    """Return the window registry list bound by :func:`registry_render_scope`."""

    return _REGISTRY_WIN.get()


def current_case_registry_steps() -> tuple[CaseRegistryStep, ...]:
    """Return the case registry list bound by :func:`registry_render_scope`."""

    return _REGISTRY_CASE.get()


@dataclass
class WindowRegistryStep:
    """Named window definition referenced by ``registry_ref`` on select expressions."""

    registry_id: str
    window_spec: WindowSpec = field(default_factory=lambda: WindowSpec(function="row_number"))

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "registry_id": "Window registry token such as w01 referenced from expressions.",
        "window_spec": (
            "Nested object whose keys are exactly those listed for WindowSpec in structural_json_keys "
            "(function, partition_by, order_by, optional argument, optional frame_kind, frame_start, "
            "frame_end, frame_start_offset, frame_end_offset)."
        ),
    }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> WindowRegistryStep:
        """
        Parse a window registry entry from JSON.

        Args:

            d: Mapping with ``registry_id`` and ``window_spec``. Legacy ``base_expr`` is merged
            into ``window_spec.argument`` when ``argument`` is unset.

        Returns:

            Populated ``WindowRegistryStep``.
        """

        ws_raw = d.get("window_spec") or {}
        ws: WindowSpec
        if isinstance(ws_raw, dict) and ws_raw.get("function"):
            ws = WindowSpec.from_dict(ws_raw)
        else:
            ws = WindowSpec(function="row_number")
        base_payload = d.get("base_expr")
        if base_payload not in (None, {}, []):
            migrated = _normalized_expr_from_stored_json(base_payload)
            if migrated.signature_key and ws.argument is None:
                ws = replace(ws, argument=migrated)
        return WindowRegistryStep(
            registry_id=str(d.get("registry_id", "")).strip(),
            window_spec=ws,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this registry step for JSON storage.

        Returns:

            Plain dict with ``registry_id`` and ``window_spec``.
        """

        return {
            "registry_id": self.registry_id,
            "window_spec": self.window_spec.to_dict(),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Shorthand registry entry for LLM repair and parse examples."""

        return {
            "registry_id": self.registry_id,
            "window_spec": self.window_spec.to_prompt_dict(),
        }

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example window_registry row for prompts."""

        return {"registry_id": "w01", "window_spec": WindowSpec.prompt_example_dict()}

    @classmethod
    def prompt_example_dict_framed(cls) -> dict[str, Any]:
        """Second registry row illustrating PARTITION BY, ORDER BY, argument, and frame bounds."""

        return {
            "registry_id": "w02",
            "window_spec": {
                "function": "sum",
                "partition_by": ["tbl_a.col_a"],
                "order_by": [{"expr": "tbl_a.col_b", "direction": "asc"}],
                "argument": "tbl_a.col_c",
                "frame_kind": "rows",
                "frame_start": "UNBOUNDED PRECEDING",
                "frame_end": "CURRENT ROW",
            },
        }

    @property
    def signature_key(self) -> str:
        """
        Structural fingerprint for template hashing and union checks.

        Returns:

            Stable string over id and window spec.
        """

        return "|".join([self.registry_id, self.window_spec.signature_key])


@dataclass
class CaseRegistryStep:
    """Named CASE expression referenced by ``registry_ref`` on select expressions."""

    registry_id: str
    label: str = ""
    case_when: CaseWhenExpr = field(default_factory=CaseWhenExpr)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseRegistryStep:
        """
        Parse a case registry entry from JSON.

        Args:

            d: Mapping with ``registry_id`` and ``case_when``.

        Returns:

            Populated ``CaseRegistryStep``.
        """

        cw_raw = d.get("case_when")
        if isinstance(cw_raw, list):
            cw = CaseWhenExpr(
                branches=[(CaseWhenBranch.from_dict(b) if isinstance(b, dict) else CaseWhenBranch()) for b in cw_raw],
            )
        elif isinstance(cw_raw, dict):
            cw = CaseWhenExpr.from_dict(cw_raw)
        else:
            cw = CaseWhenExpr()
        return CaseRegistryStep(
            registry_id=str(d.get("registry_id", "")).strip(),
            label=str(d.get("label", "")),
            case_when=cw,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this registry step for JSON storage.

        Returns:

            Plain dict with ``registry_id``, ``label``, and ``case_when``.
        """

        return {
            "registry_id": self.registry_id,
            "label": self.label,
            "case_when": self.case_when.to_dict(),
        }

    @property
    def signature_key(self) -> str:
        """
        Structural fingerprint for template hashing and union checks.

        Returns:

            Stable string over id, label, and case-when shape.
        """

        return "|".join([self.registry_id, self.label, self.case_when.signature_key])

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "registry_id": "Case registry token such as c01 referenced from expressions.",
        "label": "Optional human-readable label for diagnostics.",
        "case_when": (
            "CASE body object named exactly case_when (not alternate wrapper keys). "
            "Its keys are exactly those listed for CaseWhenExpr in structural_json_keys "
            "(branches, optional else_result, optional condition_scope)."
        ),
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Shorthand case registry row for LLM JSON."""

        out: dict[str, Any] = {
            "registry_id": self.registry_id,
            "case_when": self.case_when.to_prompt_dict(),
        }
        if self.label:
            out["label"] = self.label
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example case_registry row for prompts."""

        return {
            "registry_id": "c01",
            "case_when": CaseWhenExpr.prompt_example_dict(),
        }


class EffectiveSelectParts(NamedTuple):
    """Effective expression and optional window or CASE payloads after registry resolution."""

    expr: NormalizedExpr
    window_spec: WindowSpec | None
    case_when: CaseWhenExpr | None


def effective_select_parts(
    sc: SelectCol,
    window_registry: Sequence[WindowRegistryStep] | None = None,
    case_registry: Sequence[CaseRegistryStep] | None = None,
) -> EffectiveSelectParts:
    """
    Resolve ``expr`` registry references against optional or context-bound registries.

    Args:

        sc: Select column.

        window_registry: Optional explicit window registry list.

        case_registry: Optional explicit case registry list.

    Returns:

        Effective base expression plus window specification and CASE body from registries when ``expr`` is a bare ``wNN`` / ``cNN`` token.
    """

    wr_seq = tuple(window_registry) if window_registry is not None else _REGISTRY_WIN.get()
    cr_seq = tuple(case_registry) if case_registry is not None else _REGISTRY_CASE.get()
    win_by_id = {s.registry_id: s for s in wr_seq}
    case_by_id = {s.registry_id: s for s in cr_seq}
    rid = expr_registry_ref(sc.expr) or ""
    if rid.startswith("w"):
        step = win_by_id.get(rid)
        if step is None:
            return EffectiveSelectParts(sc.expr, None, None)
        return EffectiveSelectParts(NormalizedExpr(), step.window_spec, None)
    if rid.startswith("c"):
        step = case_by_id.get(rid)
        if step is None:
            return EffectiveSelectParts(sc.expr, None, None)
        return EffectiveSelectParts(NormalizedExpr(), None, step.case_when)
    return EffectiveSelectParts(sc.expr, None, None)


@dataclass
class SelectCol:
    """Select column with a single normalized SQL expression (including bare ``wNN`` / ``cNN`` registry tokens)."""

    expr: NormalizedExpr = field(default_factory=NormalizedExpr)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SelectCol:
        """
        Create `SelectCol` from dictionary.

        Args:

            d: Mapping with ``expr`` (dict, str, or ``NormalizedExpr``).

        Returns:

            Populated column with parsed ``expr`` only; CASE and windows must live in registries.
        """
        expr_raw = d.get("expr", {})
        if isinstance(expr_raw, str):
            expr = NormalizedExpr.from_column(expr_raw)
        elif isinstance(expr_raw, dict):
            expr = NormalizedExpr.from_dict(expr_raw)
        elif isinstance(expr_raw, NormalizedExpr):
            expr = expr_raw
        else:
            expr = NormalizedExpr()
        return SelectCol(expr=expr)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize ``expr`` to a plain dictionary.

        Returns:

            Dict containing ``expr``.
        """
        return {"expr": self.expr.to_dict()}

    @property
    def is_aggregated(self) -> bool:
        """
        Whether the column uses SQL aggregation (expr or certain window funcs).

        Returns:

            True from `expr.has_aggregation` or window `sum` / `avg`.
        """
        parts = effective_select_parts(self, None, None)
        if expr_registry_ref(self.expr) is not None:
            if parts.window_spec is not None:
                if parts.window_spec.function in {"sum", "avg"}:
                    return True
                return parts.expr.has_aggregation
            if parts.case_when is not None:
                return (
                    parts.case_when.has_aggregated_output
                    or parts.case_when.has_aggregated_condition
                    or parts.expr.has_aggregation
                )
            return True
        if parts.expr.has_aggregation:
            return True
        return False

    @property
    def signature_key(self) -> str:
        """
        Structural key combining base expr and resolved registry payloads.

        Union merge, ``display_alias_map``, and template dedupe align runtime and concrete
        columns by this string rather than by raw SQL text.

        Returns:

            Expr signature for bare registry refs; otherwise primary expr signature.
        """
        if expr_registry_ref(self.expr) is not None:
            return self.expr.signature_key
        return self.expr.signature_key

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": (
            "SQL expression string, DISTINCT-qualified column, aggregation, or bare wNN/cNN registry token "
            "(definitions live only in window_registry and case_registry)."
        ),
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """SELECT list column shorthand for LLM JSON."""

        return {"expr": _expr_prompt_sql(self.expr)}

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example select_cols entry."""

        return {"expr": "tbl_a.col_a"}


@dataclass
class OrderByCol:
    """Order by column with expression and sort direction."""

    expr: NormalizedExpr = field(default_factory=NormalizedExpr)
    direction: str = "ASC"

    def __post_init__(self) -> None:
        """
        Strip and upper-case `direction` (e.g. `ASC` / `DESC`).

        Returns:

            None.
        """
        self.direction = self.direction.strip().upper()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OrderByCol:
        """
        Create OrderByCol from dictionary.

        Args:

            d: Dictionary with 'expr' and 'direction' keys.

        Returns:

            Populated OrderByCol instance.
        """
        expr_raw = d.get("expr", {})
        if isinstance(expr_raw, str):
            expr = _parse_expr_string_for_json(expr_raw)
        elif isinstance(expr_raw, dict):
            expr = NormalizedExpr.from_dict(expr_raw)
        elif isinstance(expr_raw, NormalizedExpr):
            expr = expr_raw
        else:
            expr = NormalizedExpr()
        return OrderByCol(
            expr=expr,
            direction=d.get("direction", "ASC"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with the serialized expr and direction string.
        """
        return {"expr": self.expr.to_dict(), "direction": self.direction}

    @property
    def is_aggregated(self) -> bool:
        """
        Whether the order key expression carries an aggregate.

        Returns:

            Same as `self.expr.has_aggregation`.
        """
        return self.expr.has_aggregation

    @property
    def signature_key(self) -> str:
        """
        Expr signature plus sort direction.

        Returns:

            `expr_key|DIRECTION` string.
        """
        return "|".join([self.expr.signature_key, self.direction])

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": "ORDER BY key as a SQL expression string.",
        "direction": "Sort direction asc or desc in lowercase in prompts.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """ORDER BY entry shorthand."""

        return {
            "expr": _expr_prompt_sql(self.expr),
            "direction": self.direction.lower(),
        }

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example order_by_cols row."""

        return {"expr": "tbl_a.col_a", "direction": "asc"}


class QueryBody(Protocol):
    """
    Structural query fragment shared by the main intent body and each CTE step.

    Used for symmetric repairs and validation entry points that apply identically per scope.
    """

    tables: list[str]
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    filters_param: list[FilterParam]
    having_param: list[HavingParam]
    limit: int | None
    limit_param_key: str
    grain: str
    param_values: dict[str, ParamValue]
    window_registry: list[WindowRegistryStep]
    case_registry: list[CaseRegistryStep]
    chosen_join_candidate_id: str
    chosen_join_path_signature: list[str]
    column_map: dict[str, str]


class ConcreteQueryBody(Protocol):
    """Template-stored query body fields mirroring ``QueryBody`` for structural round-trips."""

    tables: list[str]
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    filters_param: list[FilterParam]
    having_param: list[HavingParam]
    limit: int | None
    limit_param_key: str
    grain: str
    param_values: dict[str, ParamValue]
    window_registry: list[WindowRegistryStep]
    case_registry: list[CaseRegistryStep]
    chosen_join_candidate_id: str
    chosen_join_path_signature: list[str]
    column_map: dict[str, str]


@dataclass
class ConcreteCteStep:
    """CTE step structural signature for template storage."""

    cte_name: str
    tables: list[str] = field(default_factory=list)
    select_cols: list[SelectCol] = field(default_factory=list)
    group_by_cols: list[NormalizedExpr] = field(default_factory=list)
    order_by_cols: list[OrderByCol] = field(default_factory=list)
    filters_param: list[FilterParam] = field(default_factory=list)
    having_param: list[HavingParam] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    grain: str = "row_level"
    limit: int | None = None
    limit_param_key: str = ""
    emission: CteEmissionKind = "join_table"
    column_map: dict[str, str] = field(default_factory=dict)
    output_column_metadata: dict[str, CteOutputColumnMeta] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ConcreteCteStep:
        """
        Create ConcreteCteStep from dictionary.

        Args:

            d: Dictionary with keys matching ConcreteCteStep fields.

        Returns:

            Populated ConcreteCteStep with nested expression objects.
        """
        sc_raw = d.get("select_cols", [])
        gbc_raw = d.get("group_by_cols", [])
        obc_raw = d.get("order_by_cols", [])
        fp_raw = d.get("filters_param", [])
        hp_raw = d.get("having_param", [])
        ocm_raw = d.get("output_column_metadata", {})
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        select_cols = [
            (
                SelectCol.from_dict(s)
                if isinstance(s, dict)
                else (SelectCol(expr=NormalizedExpr.from_column(s)) if isinstance(s, str) else s)
            )
            for s in sc_raw
        ]
        group_by_cols = [
            (
                NormalizedExpr.from_dict(g)
                if isinstance(g, dict)
                else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
            )
            for g in gbc_raw
        ]
        order_by_cols = [
            (
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
            )
            for o in obc_raw
        ]
        return ConcreteCteStep(
            cte_name=d.get("cte_name", ""),
            tables=d.get("tables", []),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            filters_param=[FilterParam.from_dict(f) if isinstance(f, dict) else f for f in fp_raw],
            having_param=[HavingParam.from_dict(h) if isinstance(h, dict) else h for h in hp_raw],
            output_columns=d.get("output_columns", []),
            grain=d.get("grain", "row_level"),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            emission=_coerce_cte_emission(d.get("emission")),
            column_map=d.get("column_map", {}),
            output_column_metadata={
                k: CteOutputColumnMeta.from_dict(v) if isinstance(v, dict) else v for k, v in ocm_raw.items()
            },
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=d.get("chosen_join_path_signature", []),
            param_values=d.get("param_values", {}),
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all ConcreteCteStep fields, with nested expressions serialized.
        """
        return {
            "cte_name": self.cte_name,
            "tables": self.tables,
            "select_cols": [s.to_dict() for s in self.select_cols],
            "group_by_cols": [g.to_dict() for g in self.group_by_cols],
            "order_by_cols": [o.to_dict() for o in self.order_by_cols],
            "filters_param": [f.to_dict() for f in self.filters_param],
            "having_param": [h.to_dict() for h in self.having_param],
            "output_columns": self.output_columns,
            "grain": self.grain,
            "limit": self.limit,
            "limit_param_key": self.limit_param_key,
            "emission": self.emission,
            "column_map": self.column_map,
            "output_column_metadata": {k: v.to_dict() for k, v in self.output_column_metadata.items()},
            "chosen_join_candidate_id": self.chosen_join_candidate_id,
            "chosen_join_path_signature": self.chosen_join_path_signature,
            "param_values": self.param_values,
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
        }


@dataclass
class RuntimeCteStep:
    """
    CTE step specification for WITH clause queries with runtime values.

    At JSON parse time, ``output_columns`` must match ``select_cols`` in length; each name must satisfy ``^[a-z_][a-z0-9_]*$``. Post-processing may rewrite aliases via ``derive_cte_output_columns`` and related repairs so stored ``output_columns`` are canonical, not necessarily identical to the first LLM strings.
    """

    cte_name: str
    description: str = ""
    tables: list[str] = field(default_factory=list)
    select_cols: list[SelectCol] = field(default_factory=list)
    group_by_cols: list[NormalizedExpr] = field(default_factory=list)
    order_by_cols: list[OrderByCol] = field(default_factory=list)
    filters_param: list[FilterParam] = field(default_factory=list)
    having_param: list[HavingParam] = field(default_factory=list)
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)
    grain: str = "row_level"
    limit: int | None = None
    limit_param_key: str = ""
    emission: CteEmissionKind = "join_table"
    column_map: dict[str, str] = field(default_factory=dict)
    output_column_metadata: dict[str, CteOutputColumnMeta] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RuntimeCteStep:
        """
        Create RuntimeCteStep from dictionary.

        Args:

            d: Dictionary with keys matching RuntimeCteStep fields.

        Returns:

            Populated RuntimeCteStep with nested expression objects and runtime param_values.
        """
        sc_raw = d.get("select_cols", [])
        gbc_raw = d.get("group_by_cols", [])
        obc_raw = d.get("order_by_cols", [])
        fp_raw = d.get("filters_param", [])
        hp_raw = d.get("having_param", [])
        ocm_raw = d.get("output_column_metadata", {})
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        select_cols = [
            (
                SelectCol.from_dict(s)
                if isinstance(s, dict)
                else (SelectCol(expr=NormalizedExpr.from_column(s)) if isinstance(s, str) else s)
            )
            for s in sc_raw
        ]
        group_by_cols = [
            (
                NormalizedExpr.from_dict(g)
                if isinstance(g, dict)
                else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
            )
            for g in gbc_raw
        ]
        order_by_cols = [
            (
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
            )
            for o in obc_raw
        ]
        return RuntimeCteStep(
            cte_name=d.get("cte_name", ""),
            description=d.get("description", ""),
            tables=d.get("tables", []),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            filters_param=[FilterParam.from_dict(f) if isinstance(f, dict) else f for f in fp_raw],
            having_param=[HavingParam.from_dict(h) if isinstance(h, dict) else h for h in hp_raw],
            param_values=d.get("param_values", {}),
            output_columns=d.get("output_columns", []),
            grain=d.get("grain", "row_level"),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            emission=_coerce_cte_emission(d.get("emission")),
            column_map=d.get("column_map", {}),
            output_column_metadata={
                k: CteOutputColumnMeta.from_dict(v) if isinstance(v, dict) else v for k, v in ocm_raw.items()
            },
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=d.get("chosen_join_path_signature", []),
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all RuntimeCteStep fields including param_values and limit_param_key.
        """
        return {
            "cte_name": self.cte_name,
            "description": self.description,
            "tables": self.tables,
            "select_cols": [s.to_dict() for s in self.select_cols],
            "group_by_cols": [g.to_dict() for g in self.group_by_cols],
            "order_by_cols": [o.to_dict() for o in self.order_by_cols],
            "filters_param": [f.to_dict() for f in self.filters_param],
            "having_param": [h.to_dict() for h in self.having_param],
            "param_values": self.param_values,
            "output_columns": self.output_columns,
            "grain": self.grain,
            "limit": self.limit,
            "limit_param_key": self.limit_param_key,
            "emission": self.emission,
            "column_map": self.column_map,
            "output_column_metadata": {k: v.to_dict() for k, v in self.output_column_metadata.items()},
            "chosen_join_candidate_id": self.chosen_join_candidate_id,
            "chosen_join_path_signature": self.chosen_join_path_signature,
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
        }

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "cte_name": "WITH clause name referenced downstream.",
        "description": "Short explanation of what the CTE computes.",
        "tables": "Tables feeding this CTE body.",
        "select_cols": "SELECT list entries aligned with output_columns length.",
        "group_by_cols": "GROUP BY expressions as SQL strings.",
        "order_by_cols": "ORDER BY entries with expr and direction.",
        "filters_param": "Row-level predicates inside the CTE.",
        "having_param": "Aggregate predicates inside the CTE.",
        "limit": "Optional integer row cap.",
        "output_columns": "Snake_case aliases for projected columns.",
        "window_registry": "Window definitions scoped to this CTE.",
        "case_registry": "CASE registry definitions scoped to this CTE.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """CTE body shorthand without internal-only bookkeeping fields."""

        return {
            "cte_name": self.cte_name,
            "description": self.description,
            "tables": list(self.tables),
            "select_cols": [sc.to_prompt_dict() for sc in self.select_cols],
            "group_by_cols": [_expr_prompt_sql(g) for g in self.group_by_cols],
            "order_by_cols": [o.to_prompt_dict() for o in self.order_by_cols],
            "filters_param": [fp.to_prompt_dict() for fp in self.filters_param],
            "having_param": [hp.to_prompt_dict() for hp in self.having_param],
            "limit": self.limit,
            "output_columns": list(self.output_columns),
            "window_registry": [w.to_prompt_dict() for w in self.window_registry],
            "case_registry": [c.to_prompt_dict() for c in self.case_registry],
        }

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Canonical example CTE step for prompts."""

        return {
            "cte_name": "cte1",
            "description": "Intermediate result.",
            "tables": ["tbl_a"],
            "select_cols": [SelectCol.prompt_example_dict()],
            "group_by_cols": [],
            "order_by_cols": [],
            "filters_param": [],
            "having_param": [],
            "limit": None,
            "output_columns": ["out_a"],
            "window_registry": [],
            "case_registry": [],
        }

    @property
    def expected_rows(self) -> str:
        """
        Coarse row-cardinality hint from `grain` and `limit`.

        Returns:

            `one` for scalar grain, `few` when a limit is set, else `many`.
        """
        if self.grain == "scalar":
            return "one"
        return "few" if self.limit else "many"


def _runtime_cte_to_concrete(runtime: RuntimeCteStep) -> ConcreteCteStep:
    """
    Convert RuntimeCteStep to ConcreteCteStep for template storage.

    Args:

        runtime: The runtime CTE step containing values and expressions.

    Returns:

        ConcreteCteStep with ``param_values`` stripped; structural fields including
        ``limit_param_key`` and ``emission`` are preserved.
    """
    return ConcreteCteStep(
        cte_name=runtime.cte_name,
        tables=runtime.tables,
        select_cols=runtime.select_cols,
        group_by_cols=runtime.group_by_cols,
        order_by_cols=runtime.order_by_cols,
        filters_param=runtime.filters_param,
        having_param=runtime.having_param,
        output_columns=runtime.output_columns,
        grain=runtime.grain,
        limit=runtime.limit,
        limit_param_key=runtime.limit_param_key,
        emission=runtime.emission,
        column_map=runtime.column_map,
        output_column_metadata=runtime.output_column_metadata,
        chosen_join_candidate_id=runtime.chosen_join_candidate_id,
        chosen_join_path_signature=runtime.chosen_join_path_signature,
        param_values={},
        window_registry=list(runtime.window_registry or []),
        case_registry=list(runtime.case_registry or []),
        distinct_select_index=runtime.distinct_select_index,
    )


def concrete_cte_to_runtime(concrete: ConcreteCteStep) -> RuntimeCteStep:
    """
    Convert ConcreteCteStep to RuntimeCteStep for pipeline execution.

    Args:

        concrete: The stored concrete CTE step from a template.

    Returns:

        RuntimeCteStep with blank description and empty param_values ready for execution.
    """
    return RuntimeCteStep(
        cte_name=concrete.cte_name,
        description="",
        tables=concrete.tables,
        select_cols=concrete.select_cols,
        group_by_cols=concrete.group_by_cols,
        order_by_cols=concrete.order_by_cols,
        filters_param=concrete.filters_param,
        having_param=concrete.having_param,
        param_values={},
        output_columns=concrete.output_columns,
        grain=concrete.grain,
        limit=concrete.limit,
        limit_param_key=concrete.limit_param_key,
        emission=concrete.emission,
        column_map=concrete.column_map,
        output_column_metadata=concrete.output_column_metadata,
        chosen_join_candidate_id=concrete.chosen_join_candidate_id,
        chosen_join_path_signature=concrete.chosen_join_path_signature,
        window_registry=list(concrete.window_registry or []),
        case_registry=list(concrete.case_registry or []),
        distinct_select_index=concrete.distinct_select_index,
    )


@dataclass
class RuntimeIntent:
    """
    Runtime intent container for pipeline execution with structural fields and values.

    The main query body does not carry ``output_columns``; the dialect layer determines the terminal result column list. Each ``RuntimeCteStep`` carries ``output_columns`` for its WITH definition surface.
    """

    tables: list[str]
    grain: str
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    filters_param: list[FilterParam]
    having_param: list[HavingParam] = field(default_factory=list)
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    cte_steps: list[RuntimeCteStep] = field(default_factory=list)
    natural_language: str = ""
    limit: int | None = None
    limit_param_key: str = ""
    column_map: dict[str, str] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1
    extra_tables: set[str] = field(default_factory=set)
    sql_param: str = ""
    sql_shape: SQLShape | None = None
    schema_invalid: bool = False

    @property
    def expected_rows(self) -> str:
        """
        Coarse row-cardinality hint from `grain` and top-level `limit`.

        Returns:

            `one` for scalar grain, `few` when limited, else `many`.
        """
        if self.grain == "scalar":
            return "one"
        return "few" if self.limit else "many"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RuntimeIntent:
        """
        Create RuntimeIntent from dictionary.

        Args:

            d: Dictionary with keys matching RuntimeIntent fields, typically from JSON storage.

        Returns:

            Populated RuntimeIntent with all nested objects deserialized.
        """
        sc_raw = d.get("select_cols", [])
        gbc_raw = d.get("group_by_cols", [])
        obc_raw = d.get("order_by_cols", [])
        fp_raw = d.get("filters_param", [])
        hp_raw = d.get("having_param", [])
        cte_raw = d.get("cte_steps", [])
        join_sig_raw = d.get("chosen_join_path_signature", [])
        if isinstance(join_sig_raw, str):
            join_sig_raw = [join_sig_raw] if join_sig_raw else []
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        select_cols = [
            (
                SelectCol.from_dict(s)
                if isinstance(s, dict)
                else (SelectCol(expr=NormalizedExpr.from_column(s)) if isinstance(s, str) else s)
            )
            for s in sc_raw
        ]
        group_by_cols = [
            (
                NormalizedExpr.from_dict(g)
                if isinstance(g, dict)
                else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
            )
            for g in gbc_raw
        ]
        order_by_cols = [
            (
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
            )
            for o in obc_raw
        ]
        return RuntimeIntent(
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            filters_param=[FilterParam.from_dict(fp) if isinstance(fp, dict) else fp for fp in fp_raw],
            having_param=[HavingParam.from_dict(hp) if isinstance(hp, dict) else hp for hp in hp_raw],
            param_values=d.get("param_values", {}),
            cte_steps=[RuntimeCteStep.from_dict(cte) if isinstance(cte, dict) else cte for cte in cte_raw],
            natural_language=d.get("natural_language", ""),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            column_map=d.get("column_map", {}),
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=join_sig_raw,
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
            extra_tables=set(d.get("extra_tables", [])),
            sql_param=d.get("sql_param", ""),
            sql_shape=(SQLShape.from_dict(d["sql_shape"]) if d.get("sql_shape") else None),
            schema_invalid=d.get("schema_invalid", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all RuntimeIntent fields, with nested objects serialized recursively.
        """
        return {
            "tables": self.tables,
            "grain": self.grain,
            "expected_rows": self.expected_rows,
            "select_cols": [s.to_dict() for s in self.select_cols],
            "group_by_cols": [g.to_dict() for g in self.group_by_cols],
            "order_by_cols": [o.to_dict() for o in self.order_by_cols],
            "filters_param": [fp.to_dict() for fp in self.filters_param],
            "having_param": [hp.to_dict() for hp in self.having_param],
            "param_values": self.param_values,
            "cte_steps": [cte.to_dict() for cte in self.cte_steps],
            "natural_language": self.natural_language,
            "limit": self.limit,
            "limit_param_key": self.limit_param_key,
            "column_map": self.column_map,
            "chosen_join_candidate_id": self.chosen_join_candidate_id,
            "chosen_join_path_signature": self.chosen_join_path_signature,
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
            "extra_tables": sorted(self.extra_tables),
            "sql_param": self.sql_param,
            "sql_shape": self.sql_shape.to_dict() if self.sql_shape else None,
            "schema_invalid": self.schema_invalid,
        }

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "schema_invalid": (
            "Structural formatter hint only; must be false in formatter output. "
            "The orchestrator overwrites this field using planner schema_invalid after parsing."
        ),
        "tables": "Tables and CTE names whose columns appear in the query body.",
        "select_cols": "Non-empty SELECT list entries as expr strings or registry tokens.",
        "group_by_cols": "GROUP BY expressions as SQL strings when grouping applies.",
        "order_by_cols": "ORDER BY entries with expr and direction.",
        "filters_param": "Row-level predicates for the main query.",
        "having_param": "Aggregate predicates for the main query.",
        "limit": "Optional integer row cap.",
        "natural_language": "Short description of what the query returns.",
        "cte_steps": "WITH clause steps using the same body shape as the main intent.",
        "window_registry": "Window definitions scoped to the main query.",
        "case_registry": "CASE registry definitions scoped to the main query.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Main intent shorthand without execution-only fields."""

        return {
            "schema_invalid": self.schema_invalid,
            "tables": list(self.tables),
            "select_cols": [sc.to_prompt_dict() for sc in self.select_cols],
            "group_by_cols": [_expr_prompt_sql(g) for g in self.group_by_cols],
            "order_by_cols": [o.to_prompt_dict() for o in self.order_by_cols],
            "filters_param": [fp.to_prompt_dict() for fp in self.filters_param],
            "having_param": [hp.to_prompt_dict() for hp in self.having_param],
            "limit": self.limit,
            "natural_language": self.natural_language,
            "cte_steps": [cte.to_prompt_dict() for cte in self.cte_steps],
            "window_registry": [w.to_prompt_dict() for w in self.window_registry],
            "case_registry": [c.to_prompt_dict() for c in self.case_registry],
        }

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Illustrative intent JSON matching LLM parse output shape."""

        return {
            "schema_invalid": False,
            "tables": ["tbl_a", "tbl_b"],
            "select_cols": [
                SelectCol.prompt_example_dict(),
                {"expr": "COUNT(tbl_a.col_b)"},
                {"expr": "w01"},
                {"expr": "c01"},
            ],
            "group_by_cols": ["tbl_a.col_a"],
            "order_by_cols": [OrderByCol.prompt_example_dict()],
            "filters_param": [FilterParam.prompt_example_dict()],
            "having_param": [],
            "limit": None,
            "natural_language": "one-line description of the result",
            "cte_steps": [],
            "window_registry": [
                WindowRegistryStep.prompt_example_dict(),
                WindowRegistryStep.prompt_example_dict_framed(),
            ],
            "case_registry": [CaseRegistryStep.prompt_example_dict()],
        }

    @property
    def has_aggregation(self) -> bool:
        """
        Whether any `select_cols` entry is aggregated.

        Returns:

            True if any `SelectCol.is_aggregated` is true.
        """
        with registry_render_scope(self.window_registry, self.case_registry):
            return any(s.is_aggregated for s in self.select_cols)


def intent_prompt_structural_index() -> dict[str, list[str]]:
    """
    Allowed JSON object keys per structural type for LLM intent JSON (positive closure).

    Nested objects must use only the keys listed for their type; unknown sibling keys should not be emitted.

    ``sql_expression`` applies wherever an expression is carried as a SQL string (fields named ``expr``,
    ``left_expr``, ``right_expr``, ``argument``, partition/order entries, ``group_by_cols`` strings, etc.).
    """

    return {
        "RuntimeIntent": sorted(RuntimeIntent.PROMPT_FIELD_SPEC.keys()),
        "RuntimeCteStep": sorted(RuntimeCteStep.PROMPT_FIELD_SPEC.keys()),
        "SelectCol": sorted(SelectCol.PROMPT_FIELD_SPEC.keys()),
        "OrderByCol": sorted(OrderByCol.PROMPT_FIELD_SPEC.keys()),
        "FilterParam": sorted(FilterParam.PROMPT_FIELD_SPEC.keys()),
        "HavingParam": sorted(HavingParam.PROMPT_FIELD_SPEC.keys()),
        "WindowRegistryStep": sorted(WindowRegistryStep.PROMPT_FIELD_SPEC.keys()),
        "WindowSpec": sorted(WindowSpec.PROMPT_FIELD_SPEC.keys()),
        "CaseRegistryStep": sorted(CaseRegistryStep.PROMPT_FIELD_SPEC.keys()),
        "CaseWhenExpr": sorted(CaseWhenExpr.PROMPT_FIELD_SPEC.keys()),
        "CaseWhenBranch": sorted(CaseWhenBranch.PROMPT_FIELD_SPEC.keys()),
        "sql_expression": sorted(NormalizedExpr.PROMPT_FIELD_SPEC.keys()),
    }


@dataclass
class ConcreteIntent:
    """Structural intent for template storage without values or natural language."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    filters_param: list[FilterParam]
    having_param: list[HavingParam] = field(default_factory=list)
    cte_steps: list[ConcreteCteStep] = field(default_factory=list)
    limit: int | None = None
    limit_param_key: str = ""
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    column_map: dict[str, str] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ConcreteIntent:
        """
        Create ConcreteIntent from dictionary.

        Args:

            d: Dictionary with keys matching ConcreteIntent fields.

        Returns:

            Populated ConcreteIntent with nested expression objects.
        """
        sc_raw = d.get("select_cols", [])
        gbc_raw = d.get("group_by_cols", [])
        obc_raw = d.get("order_by_cols", [])
        fp_raw = d.get("filters_param", [])
        hp_raw = d.get("having_param", [])
        cte_raw = d.get("cte_steps", [])
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        select_cols = [
            (
                SelectCol.from_dict(s)
                if isinstance(s, dict)
                else (SelectCol(expr=NormalizedExpr.from_column(s)) if isinstance(s, str) else s)
            )
            for s in sc_raw
        ]
        group_by_cols = [
            (
                NormalizedExpr.from_dict(g)
                if isinstance(g, dict)
                else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
            )
            for g in gbc_raw
        ]
        order_by_cols = [
            (
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
            )
            for o in obc_raw
        ]
        return ConcreteIntent(
            intent_id=d.get("intent_id", ""),
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            filters_param=[FilterParam.from_dict(fp) if isinstance(fp, dict) else fp for fp in fp_raw],
            having_param=[HavingParam.from_dict(hp) if isinstance(hp, dict) else hp for hp in hp_raw],
            cte_steps=[ConcreteCteStep.from_dict(cte) if isinstance(cte, dict) else cte for cte in cte_raw],
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            param_values=d.get("param_values", {}),
            column_map=d.get("column_map", {}),
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=d.get("chosen_join_path_signature", []),
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all ConcreteIntent fields, with nested expressions serialized.
        """
        return {
            "intent_id": self.intent_id,
            "tables": self.tables,
            "grain": self.grain,
            "select_cols": [s.to_dict() for s in self.select_cols],
            "group_by_cols": [g.to_dict() for g in self.group_by_cols],
            "order_by_cols": [o.to_dict() for o in self.order_by_cols],
            "filters_param": [fp.to_dict() for fp in self.filters_param],
            "having_param": [hp.to_dict() for hp in self.having_param],
            "cte_steps": [cte.to_dict() for cte in self.cte_steps],
            "limit": self.limit,
            "limit_param_key": self.limit_param_key,
            "param_values": self.param_values,
            "column_map": self.column_map,
            "chosen_join_candidate_id": self.chosen_join_candidate_id,
            "chosen_join_path_signature": self.chosen_join_path_signature,
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
        }


@dataclass
class SeedWarmupIntent:
    """Unified intent for seed warmup with inline values and expansion metadata."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    filters_param: list[FilterParam]
    having_param: list[HavingParam]
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    cte_steps: list[RuntimeCteStep] = field(default_factory=list)
    question: str = ""
    natural_language: str = ""
    expansion_metadata: ExpansionMetadata | None = None
    limit: int | None = None
    distinct_select_index: int = -1
    seed_prompt_original: str = ""
    seed_prompt_normalized: str = ""
    seed_index: int | None = None
    source: str = "gold"
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SeedWarmupIntent:
        """
        Create SeedWarmupIntent from dictionary.

        Args:

            d: Dictionary with keys matching SeedWarmupIntent fields.

        Returns:

            Populated SeedWarmupIntent with nested expression and metadata objects.
        """
        sc_raw = d.get("select_cols", [])
        gbc_raw = d.get("group_by_cols", [])
        obc_raw = d.get("order_by_cols", [])
        fp_raw = d.get("filters_param", d.get("filters", []))
        hp_raw = d.get("having_param", d.get("having", []))
        cte_raw = d.get("cte_steps", [])
        em_raw = d.get("expansion_metadata")
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        select_cols = [
            (
                SelectCol.from_dict(s)
                if isinstance(s, dict)
                else (SelectCol(expr=NormalizedExpr.from_column(s)) if isinstance(s, str) else s)
            )
            for s in sc_raw
        ]
        group_by_cols = [
            (
                NormalizedExpr.from_dict(g)
                if isinstance(g, dict)
                else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
            )
            for g in gbc_raw
        ]
        order_by_cols = [
            (
                OrderByCol.from_dict(o)
                if isinstance(o, dict)
                else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
            )
            for o in obc_raw
        ]
        return SeedWarmupIntent(
            intent_id=d.get("intent_id", ""),
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            filters_param=[FilterParam.from_dict(fp) if isinstance(fp, dict) else fp for fp in fp_raw],
            having_param=[HavingParam.from_dict(hp) if isinstance(hp, dict) else hp for hp in hp_raw],
            param_values=d.get("param_values", {}),
            cte_steps=[RuntimeCteStep.from_dict(cte) if isinstance(cte, dict) else cte for cte in cte_raw],
            question=d.get("question", ""),
            natural_language=str(d.get("natural_language", "") or ""),
            expansion_metadata=ExpansionMetadata.from_dict(em_raw) if em_raw else None,
            limit=d.get("limit"),
            distinct_select_index=int(d.get("distinct_select_index", -1)),
            seed_prompt_original=str(d.get("seed_prompt_original", "") or ""),
            seed_prompt_normalized=str(d.get("seed_prompt_normalized", "") or ""),
            seed_index=d.get("seed_index"),
            source=str(d.get("source", "gold") or "gold"),
            window_registry=[WindowRegistryStep.from_dict(w) for w in wr_raw if isinstance(w, dict)],
            case_registry=[CaseRegistryStep.from_dict(c) for c in cr_raw if isinstance(c, dict)],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize seed warmup intent including optional `expansion_metadata`.

        Returns:

            Plain dict of all fields; `expansion_metadata` only if present.
        """
        result = {
            "intent_id": self.intent_id,
            "tables": self.tables,
            "grain": self.grain,
            "select_cols": [s.to_dict() for s in self.select_cols],
            "group_by_cols": [g.to_dict() for g in self.group_by_cols],
            "order_by_cols": [o.to_dict() for o in self.order_by_cols],
            "filters_param": [fp.to_dict() for fp in self.filters_param],
            "having_param": [hp.to_dict() for hp in self.having_param],
            "param_values": self.param_values,
            "cte_steps": [cte.to_dict() for cte in self.cte_steps],
            "question": self.question,
            "natural_language": self.natural_language,
            "limit": self.limit,
            "distinct_select_index": self.distinct_select_index,
            "seed_prompt_original": self.seed_prompt_original,
            "seed_prompt_normalized": self.seed_prompt_normalized,
            "seed_index": self.seed_index,
            "source": self.source,
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
        }
        if self.expansion_metadata:
            result["expansion_metadata"] = self.expansion_metadata.to_dict()
        return result

    def to_runtime_intent(self) -> RuntimeIntent:
        """
        Convert to RuntimeIntent for pipeline execution.

        Returns:

            RuntimeIntent built from this intent with an inferred column_map.
        """
        column_map = {}
        for sc in self.select_cols:
            col = sc.expr.primary_column
            if "." in col:
                table, bare = col.split(".", 1)
                if table in self.tables:
                    column_map[bare] = table
        for fp in self.filters_param:
            col = fp.left_expr.primary_column
            if "." in col:
                table, bare = col.split(".", 1)
                if table in self.tables:
                    column_map[bare] = table

        return RuntimeIntent(
            tables=self.tables,
            grain=self.grain,
            select_cols=self.select_cols,
            group_by_cols=self.group_by_cols,
            order_by_cols=self.order_by_cols,
            filters_param=self.filters_param,
            having_param=self.having_param,
            param_values=self.param_values,
            cte_steps=self.cte_steps,
            natural_language=self.natural_language or "",
            limit=self.limit,
            column_map=column_map,
            window_registry=list(self.window_registry),
            case_registry=list(self.case_registry),
            distinct_select_index=self.distinct_select_index,
        )


def _intent_tables_have_duplicate_tables(tables: list[str]) -> bool:
    """Return True when the same qualified table identifier appears more than once."""

    names = [t for t in tables if t]
    return len(names) != len(set(names))


def _seed_intent_has_self_join_via_tables(intent: SeedWarmupIntent) -> bool:
    """Detect duplicate table references on the main intent or any CTE branch."""

    if _intent_tables_have_duplicate_tables(intent.tables or []):
        return True
    for step in intent.cte_steps or []:
        if _intent_tables_have_duplicate_tables(step.tables or []):
            return True
    return False


def _seed_intent_has_scalar_cte(intent: SeedWarmupIntent) -> bool:
    """Return True when any CTE step is emitted as a scalar subquery."""

    for step in intent.cte_steps or []:
        if step.emission == "scalar_subquery":
            return True
    return False


def _seed_intent_filter_date_flags(intent: SeedWarmupIntent) -> tuple[bool, bool]:
    """Return whether filters carry date-window versus date-difference semantics."""

    has_dw = False
    has_dd = False
    for fp in intent.filters_param or []:
        vt = (fp.value_type or "").strip().lower()
        if vt in FILTER_VALUE_TYPE_DATE_WINDOW:
            has_dw = True
        if vt in FILTER_VALUE_TYPE_DATE_DIFF:
            has_dd = True
    return has_dw, has_dd


def _seed_intent_json_contains_unnest(intent: SeedWarmupIntent) -> bool:
    """Heuristic UNNEST detection across the serialized intent footprint."""

    blob = json.dumps(intent.to_dict(), sort_keys=True, default=str).lower()
    return "unnest" in blob


def _window_operator_kind_from_registry(
    steps: list[WindowRegistryStep],
) -> WindowOperatorKind:
    """Map window registry rows to a coarse rank versus aggregate versus navigate class."""

    if not steps:
        return "none"
    saw_rank = False
    saw_agg = False
    saw_nav = False
    for st in steps:
        fn = (st.window_spec.function or "").strip().lower()
        if fn in WINDOW_REGISTRY_NAV_KIND_HINTS:
            saw_nav = True
        elif fn in WINDOW_REGISTRY_RANK_KIND_HINTS:
            saw_rank = True
        elif fn in WINDOW_REGISTRY_AGG_KIND_HINTS:
            saw_agg = True
        elif fn:
            saw_agg = True
    if saw_nav:
        return "navigate"
    if saw_rank:
        return "rank"
    if saw_agg:
        return "aggregate"
    return "none"


def infer_workload_family_for_seed_intent(intent: SeedWarmupIntent) -> WorkloadFamily:
    """Map observable structural cues to a canonical workload family for sampling keys."""

    group_n = len(intent.group_by_cols or [])
    sel_cols = intent.select_cols or []
    has_agg = any(sc.is_aggregated for sc in sel_cols)
    if intent.limit is not None and (intent.order_by_cols or []):
        return WorkloadFamily.LEADERBOARD
    if group_n > 0 and has_agg:
        return WorkloadFamily.BREAKDOWN
    for fp in intent.filters_param or []:
        vt = (fp.value_type or "").strip().lower()
        if vt in FILTER_VALUE_TYPE_DATE_WINDOW:
            return WorkloadFamily.CHANGE_OVER_TIME
        if vt in FILTER_VALUE_TYPE_DATE_DIFF:
            return WorkloadFamily.TREND
    if intent.having_param:
        return WorkloadFamily.THRESHOLD_EXCEPTION
    if intent.distinct_select_index >= 0:
        return WorkloadFamily.STATUS_REPORT
    return WorkloadFamily.EXTRACT


def operator_feature_vector_for_seed_intent(
    intent: SeedWarmupIntent,
) -> OperatorFeatureVector:
    """
    Summarize observable structural operators for diversity metrics and lattice keys.

    Args:

        intent: Warmup intent after normalization and optional expansion.

    Returns:

        Frozen footprint aligned with seed-warmup covering-array dimensions.
    """

    sel_cols = intent.select_cols or []
    has_agg = any(sc.is_aggregated for sc in sel_cols)
    has_gb = len(intent.group_by_cols or []) > 0
    has_hav = len(intent.having_param or []) > 0
    win_kind = _window_operator_kind_from_registry(list(intent.window_registry or []))
    self_join = _seed_intent_has_self_join_via_tables(intent)
    scalar_cte = _seed_intent_has_scalar_cte(intent)
    unnest = _seed_intent_json_contains_unnest(intent)
    case_when = len(intent.case_registry or []) > 0
    date_win, date_diff = _seed_intent_filter_date_flags(intent)
    cte_n = len(intent.cte_steps or [])
    if cte_n <= 0:
        cte_b = 0
    elif cte_n == 1:
        cte_b = 1
    else:
        cte_b = 2
    tc = len(intent.tables or [])
    if tc <= 1:
        jb = 0
    elif tc == 2:
        jb = 1
    else:
        jb = 2
    fam = infer_workload_family_for_seed_intent(intent)
    return OperatorFeatureVector(
        has_aggregate=has_agg,
        has_grouping=has_gb,
        has_having=has_hav,
        window_kind=win_kind,
        has_self_join_via_cte=self_join,
        has_scalar_cte=scalar_cte,
        has_unnest=unnest,
        has_case_when=case_when,
        has_date_window=date_win,
        has_date_diff=date_diff,
        cte_depth_bucket=cte_b,
        join_breadth_bucket=jb,
        workload_family=fam,
    )


def warmup_coverage_atoms_for_seed_intent(intent: SeedWarmupIntent) -> frozenset[str]:
    """
    Unary and pairwise tags for submodular warmup coverage and expansion scoring.

    Args:

        intent: Warmup intent row after expansion.

    Returns:

        Frozen set of hashed coverage atoms including selected second-order pairs.
    """

    v = operator_feature_vector_for_seed_intent(intent)
    tier = classify_seed_warmup_intent_complexity(intent).value
    cell = {
        f"tier:{tier}",
        f"fam:{v.workload_family.value}",
        f"win:{v.window_kind}",
        f"agg:{int(v.has_aggregate)}",
        f"gb:{int(v.has_grouping)}",
        f"hav:{int(v.has_having)}",
        f"cteb:{v.cte_depth_bucket}",
        f"jbb:{v.join_breadth_bucket}",
        f"scte:{int(v.has_scalar_cte)}",
        f"unnest:{int(v.has_unnest)}",
        f"casew:{int(v.has_case_when)}",
    }
    atoms: set[str] = set(cell)
    ordered = sorted(cell)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            atoms.add(f"pair:{ordered[i]}|{ordered[j]}")
    return frozenset(atoms)


def classify_seed_warmup_intent_complexity(intent: SeedWarmupIntent) -> ComplexityTier:
    """
    Assign a discrete complexity tier from observable structural features.

    Uses table count, CTE depth, window and CASE registries, aggregates, GROUP BY, and HAVING.

    Args:

        intent: Warmup intent after expansion and substitution.

    Returns:

        One of :class:`ComplexityTier`.
    """

    tables_n = len(intent.tables or [])
    cte_n = len(intent.cte_steps or [])
    win_n = len(intent.window_registry or [])
    case_n = len(intent.case_registry or [])
    group_n = len(intent.group_by_cols or [])
    hav_n = len(intent.having_param or [])
    sel_cols = intent.select_cols or []
    has_agg = any(sc.is_aggregated for sc in sel_cols)
    has_ord = len(intent.order_by_cols or []) > 0
    lim_set = intent.limit is not None

    if (
        cte_n >= 2
        or (cte_n >= 1 and tables_n >= 3)
        or (win_n >= 1 and group_n >= 1)
        or (cte_n >= 1 and win_n >= 1)
        or (case_n >= 1 and cte_n >= 1 and tables_n >= 2)
    ):
        return ComplexityTier.HIGHLY_COMPLEX

    if tables_n >= 3 or cte_n >= 1 or hav_n >= 1 or (has_agg and group_n >= 1) or win_n >= 1 or case_n >= 2:
        return ComplexityTier.COMPLEX

    if tables_n >= 2 or has_agg or group_n >= 1 or has_ord or lim_set:
        return ComplexityTier.MODERATE

    return ComplexityTier.SIMPLE


@dataclass(frozen=True)
class AnchorLatticeKey:
    """Stable coordinates for NL anchor reuse across synthetic warmup rows."""

    family: WorkloadFamily
    tier: ComplexityTier
    style: WarmupStyle
    novelty_band: NoveltyBand


@dataclass
class AnchorLatticeCell:
    """One lattice cell holding representative intent id and canonical NL anchors."""

    key: AnchorLatticeKey
    representative_intent_id: str
    anchors: tuple[str, ...]


@dataclass
class AnchorLattice:
    """Partition of warmup intents into lattice cells for deduplicated NL generation."""

    cells: dict[AnchorLatticeKey, AnchorLatticeCell]


def anchor_lattice_key_for_seed_intent(intent: SeedWarmupIntent) -> AnchorLatticeKey:
    """
    Build a stable lattice cell key for NL reuse across synthetic warmup rows.

    Args:

        intent: Warmup intent row after expansion.

    Returns:

        Frozen key tuple used for shared anchor retrieval per schema fingerprint.
    """

    tier = classify_seed_warmup_intent_complexity(intent)
    fam = WorkloadFamily.EXTRACT
    tid = intent.seed_index if intent.seed_index is not None else 0
    em = intent.expansion_metadata
    depth = int(em.depth) if em and em.depth is not None else 0
    if depth <= 0:
        nov = NoveltyBand.LOW
    elif depth == 1:
        nov = NoveltyBand.MEDIUM
    else:
        nov = NoveltyBand.HIGH
    digest = hashlib.sha256(
        (
            f"warmup_lattice_style:{SeedWarmupConfig.WARMUP_SAMPLING_POLICY_VERSION}:{tid}:{tier.value}:{nov.value}"
        ).encode(),
    ).hexdigest()
    slot = int(digest[0:2], 16) % len(SeedWarmupConfig.WARMUP_QUESTION_STYLES)
    style_s = SeedWarmupConfig.WARMUP_QUESTION_STYLES[slot]
    ws = WarmupStyle(style_s)
    return AnchorLatticeKey(family=fam, tier=tier, style=ws, novelty_band=nov)


def anchor_lattice_signature(key: AnchorLatticeKey, schema_fp: str) -> str:
    """
    Stable digest string for JSON persistence keyed by lattice cell and schema hash.

    Args:

        key: Lattice coordinates.

        schema_fp: Effective structural schema fingerprint.

    Returns:

        Hex SHA-256 digest for cache files.
    """

    payload = "|".join(
        [
            key.family.value,
            key.tier.value,
            key.style.value,
            key.novelty_band.value,
            schema_fp,
        ],
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class QSimFilter:
    """Lightweight filter for QSim intent with column reference and operator."""

    column: str
    op: str
    value_type: str
    right_column: str = ""

    @property
    def is_expr_comparison(self) -> bool:
        """
        Whether the filter compares two expressions via `right_column`.

        Returns:

            True when `right_column` is non-empty.
        """
        return bool(self.right_column)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimFilter:
        """
        Create QSimFilter from dictionary.

        Args:

            d: Dictionary with 'column', 'op', 'value_type', and optional 'right_column' keys.

        Returns:

            Populated QSimFilter instance.
        """
        return QSimFilter(
            column=d.get("column", ""),
            op=d.get("op", "="),
            value_type=d.get("value_type", "categorical"),
            right_column=d.get("right_column", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with filter fields; right_column only included when set.
        """
        result = {"column": self.column, "op": self.op, "value_type": self.value_type}
        if self.right_column:
            result["right_column"] = self.right_column
        return result


@dataclass
class QSimHaving:
    """Lightweight having condition for QSim intent with aggregate expression."""

    expression: str
    op: str
    value_type: str
    right_expression: str = ""

    @property
    def is_expression_comparison(self) -> bool:
        """
        Whether HAVING compares two expressions via `right_expression`.

        Returns:

            True when `right_expression` is non-empty.
        """
        return bool(self.right_expression)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimHaving:
        """
        Create QSimHaving from dictionary.

        Args:

            d: Dictionary with 'expression', 'op', 'value_type', and optional 'right_expression' keys.

        Returns:

            Populated QSimHaving instance.
        """
        return QSimHaving(
            expression=d.get("expression", ""),
            op=d.get("op", ">"),
            value_type=d.get("value_type", "number"),
            right_expression=d.get("right_expression", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with having fields; right_expression only included when set.
        """
        result = {
            "expression": self.expression,
            "op": self.op,
            "value_type": self.value_type,
        }
        if self.right_expression:
            result["right_expression"] = self.right_expression
        return result


def classify_qsim_skeleton_complexity(skeleton: QSimSkeleton) -> ComplexityTier:
    """
    Map a QSim structural skeleton to the discrete tier used for quota sampling.

    Args:

        skeleton: Enumerated shape prior to LLM fill.

    Returns:

        Tier bucket aligned with :func:`classify_qsim_intent_complexity` outcomes.
    """

    tables_n = len(skeleton.tables or [])
    hav_n = int(skeleton.num_having)
    group_n = int(skeleton.num_groupby)
    agg = skeleton.has_aggregation
    if tables_n >= 3 or hav_n >= 1 or (agg and group_n >= 1) or skeleton.has_expr_comparison:
        return ComplexityTier.COMPLEX
    if tables_n >= 2 or agg or group_n >= 1 or skeleton.has_orderby or skeleton.has_distinct:
        return ComplexityTier.MODERATE
    return ComplexityTier.SIMPLE


def classify_qsim_intent_complexity(intent: QSimIntent) -> ComplexityTier:
    """
    Classify a filled QSim intent using observable SQL-shape cues without CTE or window registries.

    Args:

        intent: Post-fill intent with string select columns.

    Returns:

        One of :class:`ComplexityTier`.
    """

    tables_n = len(intent.tables or [])
    hav_n = len(intent.having_param or [])
    group_n = len(intent.group_by_cols or [])
    sel_cols = intent.select_cols or []
    has_agg = any(AGG_PATTERN.match(str(sc)) for sc in sel_cols)
    has_ord = len(intent.order_by_cols or []) > 0
    lim_set = intent.limit is not None
    if tables_n >= 3 or hav_n >= 1 or (has_agg and group_n >= 1):
        return ComplexityTier.COMPLEX
    if tables_n >= 2 or has_agg or group_n >= 1 or has_ord or lim_set or intent.distinct:
        return ComplexityTier.MODERATE
    return ComplexityTier.SIMPLE


def qsim_intent_matches_target_tier(classified: ComplexityTier, target: ComplexityTier) -> bool:
    """
    Return whether a filled intent meets the structural floor implied by the sampled target tier.

    Args:

        classified: Tier from :func:`classify_qsim_intent_complexity`.

        target: Tier slot chosen for this QSim draw.

    Returns:

        True when the fill satisfies the lower bound for the target band.
    """

    rank_map = {
        ComplexityTier.SIMPLE: 0,
        ComplexityTier.MODERATE: 1,
        ComplexityTier.COMPLEX: 2,
        ComplexityTier.HIGHLY_COMPLEX: 3,
    }
    c = rank_map[classified]
    need = min(rank_map[target], rank_map[ComplexityTier.COMPLEX])
    return c >= need


@dataclass
class QSimIntent:
    """Unified intent for QSim question generation with optional values."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[str]
    group_by_cols: list[str]
    order_by_cols: list[str]
    filters_param: list[QSimFilter]
    having_param: list[QSimHaving]
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    question: str = ""
    variant_idx: int = 0
    limit: int | None = None
    distinct: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimIntent:
        """
        Create QSimIntent from dictionary.

        Args:

            d: Dictionary with keys matching QSimIntent fields.

        Returns:

            Populated QSimIntent instance.
        """
        fp_raw = d.get("filters_param", d.get("filters", []))
        hp_raw = d.get("having_param", d.get("having", []))
        return QSimIntent(
            intent_id=d.get("intent_id", ""),
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=d.get("select_cols", []),
            group_by_cols=d.get("group_by_cols", []),
            order_by_cols=d.get("order_by_cols", []),
            filters_param=[QSimFilter.from_dict(fp) if isinstance(fp, dict) else fp for fp in fp_raw],
            having_param=[QSimHaving.from_dict(hp) if isinstance(hp, dict) else hp for hp in hp_raw],
            param_values=d.get("param_values", {}),
            question=d.get("question", ""),
            variant_idx=d.get("variant_idx", 0),
            limit=d.get("limit"),
            distinct=d.get("distinct", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all QSimIntent fields.
        """
        return {
            "intent_id": self.intent_id,
            "tables": self.tables,
            "grain": self.grain,
            "select_cols": self.select_cols,
            "group_by_cols": self.group_by_cols,
            "order_by_cols": self.order_by_cols,
            "filters_param": [fp.to_dict() for fp in self.filters_param],
            "having_param": [hp.to_dict() for hp in self.having_param],
            "param_values": self.param_values,
            "question": self.question,
            "variant_idx": self.variant_idx,
            "limit": self.limit,
            "distinct": self.distinct,
        }


def runtime_intent_to_concrete(runtime: RuntimeIntent, intent_id: str) -> ConcreteIntent:
    """
    Extract `ConcreteIntent` from `RuntimeIntent` for template storage.

    Args:

        runtime: Live intent with `param_values` and optional NL fields.

        intent_id: Identifier to attach to the stored structural intent.

    Returns:

        Structural copy with runtime-only fields stripped and CTE steps concretised.
    """
    return ConcreteIntent(
        intent_id=intent_id,
        tables=runtime.tables,
        grain=runtime.grain,
        select_cols=runtime.select_cols,
        group_by_cols=runtime.group_by_cols,
        order_by_cols=runtime.order_by_cols,
        filters_param=runtime.filters_param,
        having_param=runtime.having_param,
        cte_steps=[_runtime_cte_to_concrete(cte) for cte in runtime.cte_steps],
        limit=runtime.limit,
        limit_param_key=runtime.limit_param_key,
        param_values={},
        column_map=runtime.column_map,
        chosen_join_candidate_id=runtime.chosen_join_candidate_id,
        chosen_join_path_signature=runtime.chosen_join_path_signature,
        window_registry=list(runtime.window_registry or []),
        case_registry=list(runtime.case_registry or []),
        distinct_select_index=runtime.distinct_select_index,
    )


def concrete_intent_to_runtime_skeleton(concrete: ConcreteIntent) -> RuntimeIntent:
    """Build a ``RuntimeIntent`` mirroring *concrete* with empty ``param_values``."""

    return RuntimeIntent(
        tables=list(concrete.tables or []),
        grain=concrete.grain or "row_level",
        select_cols=list(concrete.select_cols or []),
        group_by_cols=list(concrete.group_by_cols or []),
        order_by_cols=list(concrete.order_by_cols or []),
        filters_param=list(concrete.filters_param or []),
        having_param=list(concrete.having_param or []),
        param_values={},
        cte_steps=[concrete_cte_to_runtime(c) for c in (concrete.cte_steps or [])],
        natural_language="",
        limit=concrete.limit,
        limit_param_key=concrete.limit_param_key,
        column_map=dict(concrete.column_map or {}),
        chosen_join_candidate_id=concrete.chosen_join_candidate_id or "",
        chosen_join_path_signature=list(concrete.chosen_join_path_signature or []),
        window_registry=list(concrete.window_registry or []),
        case_registry=list(concrete.case_registry or []),
        distinct_select_index=concrete.distinct_select_index,
    )


@dataclass
class ValueHistory:
    """Value history for Template tracking historical query values as a flat dict."""

    param_values: list[dict[str, ParamValue]]
    questions: list[str]
    natural_language: list[str]
    accept_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Align ``accept_counts`` with stored question rows."""

        while len(self.accept_counts) < len(self.questions):
            self.accept_counts.append(1)
        if len(self.accept_counts) > len(self.questions):
            self.accept_counts = self.accept_counts[: len(self.questions)]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ValueHistory:
        """
        Create ValueHistory from dictionary.

        Args:

            d: Dictionary with ``param_values``, ``questions``, ``natural_language``, and optional ``accept_counts`` keys.

        Returns:

            Populated ValueHistory instance.
        """
        questions = list(d.get("questions", []) or [])
        ac_raw = d.get("accept_counts")
        if isinstance(ac_raw, list) and len(ac_raw) == len(questions):
            accept_counts = [int(x) for x in ac_raw]
        else:
            accept_counts = [1] * len(questions)
        return ValueHistory(
            param_values=d.get("param_values", []),
            questions=questions,
            natural_language=d.get("natural_language", []),
            accept_counts=accept_counts,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with parallel history lists including ``accept_counts``.
        """
        return {
            "param_values": self.param_values,
            "questions": self.questions,
            "natural_language": self.natural_language,
            "accept_counts": list(self.accept_counts),
        }

    def append_question_variant(
        self,
        question: str,
        *,
        accept_count: int,
        param_values: dict[str, ParamValue] | None = None,
        natural_language: str = "",
        dedup: bool = True,
    ) -> None:
        """
        Append a paraphrased question row with its accept counter and optional execution snapshot.

        Args:

            question: Stored question text for matching (canonical paraphrase).

            accept_count: Per-row accept aggregate used by reuse auto-accept gating.

            param_values: Optional flat parameter dict; defaults to the latest row when omitted.

            natural_language: Optional intent description aligned with this row.

            dedup: When True, skip appending when an identical ``question`` row already exists.

        Returns:

            None.
        """
        pv = param_values if param_values is not None else (dict(self.param_values[-1]) if self.param_values else {})
        if dedup:
            if question in self.questions:
                return
        self.param_values.append(pv)
        self.questions.append(question)
        self.natural_language.append(natural_language or "")
        self.accept_counts.append(accept_count)

    def add(
        self,
        param_values: dict[str, ParamValue],
        question: str,
        natural_language: str,
        *,
        accept_increment: int = 1,
    ) -> None:
        """
        Merge into an existing identical ``question`` row or append a new aligned row.

        Args:

            param_values: Flat dict of parameter keys to resolved values.

            question: Natural-language question for this execution.

            natural_language: Intent description (stored as empty string if omitted).

            accept_increment: Amount to add to ``accept_counts`` when merging or initializing a row.

        Returns:

            None.
        """
        for idx, existing in enumerate(self.questions):
            if existing == question:
                self.accept_counts[idx] += accept_increment
                self.param_values[idx] = param_values
                self.natural_language[idx] = natural_language or ""
                return
        self.param_values.append(param_values)
        self.questions.append(question)
        self.natural_language.append(natural_language or "")
        self.accept_counts.append(accept_increment)

    def __len__(self) -> int:
        """
        Number of stored history rows.

        Returns:

            Length of ``questions`` (aligned with other lists).
        """
        return len(self.questions)


class RejectionBucket(str, Enum):
    """High-level category for user rejection feedback (agnostic labels)."""

    MISSING_FILTER = "MISSING_FILTER"
    WRONG_GROUPING = "WRONG_GROUPING"
    WRONG_AGGREGATION = "WRONG_AGGREGATION"
    WRONG_TIME_RANGE = "WRONG_TIME_RANGE"
    WRONG_TABLES_OR_JOINS = "WRONG_TABLES_OR_JOINS"
    WRONG_SORT_OR_LIMIT = "WRONG_SORT_OR_LIMIT"
    OTHER = "OTHER"


class FeedbackKind(str, Enum):
    """Discriminator for stored question feedback entries."""

    VALIDATION_FAILURE = "validation_failure"
    INTENT_REJECTED = "intent_rejected"


def _feedback_kind_from_raw(raw: Any) -> FeedbackKind:
    """Coerce a stored string to ``FeedbackKind``, defaulting to validation failure."""

    s = str(raw or "").strip()
    for m in FeedbackKind:
        if m.value == s:
            return m
    return FeedbackKind.VALIDATION_FAILURE


def _rejection_bucket_from_raw(raw: Any) -> RejectionBucket:
    """Coerce a stored string to ``RejectionBucket``, defaulting to OTHER."""

    s = str(raw or "").strip()
    for m in RejectionBucket:
        if m.value == s:
            return m
    return RejectionBucket.OTHER


@dataclass(frozen=True)
class QuestionFeedbackEntry:
    """One LLM-summarized failure or rejection event for a normalised question and intent-structure hash."""

    summary: str
    buckets: tuple[RejectionBucket, ...]
    kind: FeedbackKind
    effective_structural_hash: str
    intent_structural_hash: str
    intent_payload: str
    created_at: str
    updated_at: str
    is_post_restart: bool = False
    source: Literal["engine"] = "engine"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""

        return {
            "summary": self.summary,
            "buckets": [b.value for b in self.buckets],
            "kind": self.kind.value,
            "effective_structural_hash": self.effective_structural_hash,
            "intent_structural_hash": self.intent_structural_hash,
            "intent_payload": self.intent_payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_post_restart": self.is_post_restart,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QuestionFeedbackEntry:
        """Deserialize from ``to_dict`` output or legacy single-bucket rows."""

        raw_b = d.get("buckets")
        bkt_tuple: tuple[RejectionBucket, ...]
        if isinstance(raw_b, list) and raw_b:
            bkt_tuple = tuple(_rejection_bucket_from_raw(x) for x in raw_b)
        else:
            bkt_tuple = (_rejection_bucket_from_raw(d.get("bucket")),)
        created = str(d.get("created_at", "") or "")
        updated = str(d.get("updated_at", "") or "") or created
        src: Literal["engine"] = "engine"
        return QuestionFeedbackEntry(
            summary=str(d.get("summary", "") or ""),
            buckets=bkt_tuple,
            kind=_feedback_kind_from_raw(d.get("kind")),
            effective_structural_hash=str(d.get("effective_structural_hash", "") or ""),
            intent_structural_hash=str(d.get("intent_structural_hash", "") or ""),
            intent_payload=str(d.get("intent_payload", "") or ""),
            created_at=created,
            updated_at=updated,
            is_post_restart=bool(d.get("is_post_restart", False)),
            source=src,
        )

    def to_prompt_row(self) -> dict[str, str]:
        """Serialize one entry for embedding under ``prior_question_feedback`` in prompts."""

        return {
            "kind": self.kind.value,
            "summary": self.summary,
            "buckets": ",".join(b.value for b in self.buckets),
            "effective_structural_hash": self.effective_structural_hash,
            "intent_structural_hash": self.intent_structural_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_post_restart": str(bool(self.is_post_restart)),
            "source": self.source,
        }


@dataclass
class FeedbackCounts:
    """Per-(template, question_norm) accept/reject counts with provenance."""

    accepts: int = 0
    rejects: int = 0
    last_path: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FeedbackCounts:
        """Reconstruct ``FeedbackCounts`` from a serialized mapping."""
        return FeedbackCounts(
            accepts=int(d.get("accepts", 0) or 0),
            rejects=int(d.get("rejects", 0) or 0),
            last_path=int(d.get("last_path", 0) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "accepts": int(self.accepts),
            "rejects": int(self.rejects),
            "last_path": int(self.last_path),
        }


@dataclass
class Template:
    """Validated and accepted query template."""

    id: str
    effective_structural_hash: str
    intent_signature: ConcreteIntent
    intent_key: str
    tables_used: list[str]
    sql_param: str
    sql_fp: str
    shape: SQLShape
    colmap_sig: str
    value_history: ValueHistory
    stats: TemplateStats
    source: str = "human"
    trust_level: int = 1
    structural_defaults: dict[str, str | int | float] = field(default_factory=dict)
    display_alias_map: dict[str, str] = field(default_factory=dict)
    feedback_by_question: dict[str, FeedbackCounts] = field(default_factory=dict)

    @property
    def chosen_join_candidate_id(self) -> str:
        """
        Join candidate id from the embedded concrete intent.

        Returns:

            `intent_signature.chosen_join_candidate_id`.
        """
        return self.intent_signature.chosen_join_candidate_id

    @property
    def chosen_join_path_signature(self) -> list[str]:
        """
        Join path signature from the embedded concrete intent.

        Returns:

            `intent_signature.chosen_join_path_signature`.
        """
        return self.intent_signature.chosen_join_path_signature

    @property
    def schema_hash(self) -> str:
        """Alias for ``effective_structural_hash``."""

        return self.effective_structural_hash

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Template:
        """
        Create Template from dictionary with nested dataclass reconstruction.

        Args:

            d: Dictionary with all Template fields, including nested intent_signature, value_history, stats, and shape dicts.

        Returns:

            Fully populated Template instance.
        """
        intent_sig = d.get("intent_signature", {})
        if isinstance(intent_sig, dict):
            intent_sig = ConcreteIntent.from_dict(intent_sig)
        vh_data = d.get("value_history", {})
        value_history = ValueHistory.from_dict(vh_data) if isinstance(vh_data, dict) else vh_data
        stats_data = d.get("stats", {})
        stats = TemplateStats.from_dict(stats_data) if isinstance(stats_data, dict) else stats_data
        shape_data = d.get("shape", {})
        shape = SQLShape.from_dict(shape_data) if isinstance(shape_data, dict) else shape_data
        fbq_raw = d.get("feedback_by_question", {}) or {}
        feedback_by_question: dict[str, FeedbackCounts] = {}
        if isinstance(fbq_raw, dict):
            for q, counts in fbq_raw.items():
                if isinstance(counts, dict):
                    feedback_by_question[str(q)] = FeedbackCounts.from_dict(counts)
                elif isinstance(counts, FeedbackCounts):
                    feedback_by_question[str(q)] = counts
        return Template(
            id=d.get("id", ""),
            effective_structural_hash=d.get("effective_structural_hash", d.get("schema_hash", "")),
            intent_signature=intent_sig,
            intent_key=d.get("intent_key", ""),
            tables_used=d.get("tables_used", []),
            sql_param=d.get("sql_param", ""),
            sql_fp=d.get("sql_fp", ""),
            shape=shape,
            colmap_sig=d.get("colmap_sig", ""),
            value_history=value_history,
            stats=stats,
            source=d.get("source", "human"),
            trust_level=d.get("trust_level", 1),
            structural_defaults=d.get("structural_defaults", {}),
            display_alias_map=dict(d.get("display_alias_map") or {}),
            feedback_by_question=feedback_by_question,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary with nested dataclass conversion.

        Returns:

            Dictionary with all Template fields, with intent_signature, value_history, stats, and shape serialized recursively.
        """
        intent_sig = (
            self.intent_signature.to_dict() if hasattr(self.intent_signature, "to_dict") else self.intent_signature
        )
        vh_dict = self.value_history.to_dict() if hasattr(self.value_history, "to_dict") else self.value_history
        stats_dict = self.stats.to_dict() if hasattr(self.stats, "to_dict") else self.stats
        return {
            "id": self.id,
            "effective_structural_hash": self.effective_structural_hash,
            "intent_signature": intent_sig,
            "intent_key": self.intent_key,
            "tables_used": self.tables_used,
            "sql_param": self.sql_param,
            "sql_fp": self.sql_fp,
            "shape": self.shape.to_dict(),
            "colmap_sig": self.colmap_sig,
            "value_history": vh_dict,
            "stats": stats_dict,
            "source": self.source,
            "trust_level": self.trust_level,
            "structural_defaults": self.structural_defaults,
            "display_alias_map": self.display_alias_map,
            "feedback_by_question": {q: c.to_dict() for q, c in self.feedback_by_question.items()},
        }


@dataclass
class StructuralCompareResult:
    """Unified structural comparison between a runtime intent and a template."""

    non_agg_symmetric_diff: int
    union_eligible: bool
    union_cols: list[Any] = field(default_factory=list)
    cols_changed: bool = False
    union_sql_path: GenerationPath | None = None
    similarity_score: float | None = None


@dataclass
class SeedWarmupResult:
    """Result of a single seed warmup run."""

    intent: RuntimeIntent
    question: str
    questions: list[str] = field(default_factory=list)
    sql: str | None = None
    rows: list | None = None
    success: bool = False
    error: str | None = None
    validation_issues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    llm_response: str | None = None
    sql_generation_attempts: int = 0
    repair_loop_count: int = 0
    drop_reason_category: str | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    sqlstate: str | None = None
    message_key: str | None = None
    preflight_execute_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary.

        Returns:

            Dictionary with all SeedWarmupResult fields including the serialized intent.
        """
        return {
            "intent": self.intent.to_dict() if self.intent else None,
            "question": self.question,
            "questions": list(self.questions),
            "sql": self.sql,
            "rows": self.rows,
            "success": self.success,
            "error": self.error,
            "validation_issues": self.validation_issues,
            "confidence": self.confidence,
            "llm_response": self.llm_response,
            "sql_generation_attempts": self.sql_generation_attempts,
            "repair_loop_count": self.repair_loop_count,
            "drop_reason_category": self.drop_reason_category,
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "sqlstate": self.sqlstate,
            "message_key": self.message_key,
            "preflight_execute_ok": self.preflight_execute_ok,
        }


@dataclass
class TemplateMatch:
    """Result of template matching against intent."""

    intent: RuntimeIntent | None = None
    best_template: Template | None = None
    similarity_score: float = 0.0
    reuse_type: str = "none"
    semantic_warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    reuse_candidate_normalized: str | None = None
    reuse_history_index: int | None = None


@dataclass(frozen=True, slots=True)
class SqlGenerationOutcome:
    """Result of SQL generation and validation."""

    sql: str
    success: bool
    generation_path: GenerationPath
    matched_template: Template | None
    structural_match_templates: tuple[Template, ...] = ()
    sql_validation_error: str | None = None
    join_matches_template: bool | None = None
    error_kind: str | None = None
    explain_soft_diagnostics: int = 0


@dataclass(frozen=True, slots=True)
class DirectReuseSuspendContext:
    """Captured state to finish direct SQL reuse after a deferred confirmation prompt."""

    q_norm: str
    ref_tmpl: Template
    dialect: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema: Any
    intent: RuntimeIntent
    sql: str
    rows: tuple[tuple[Any, ...], ...]
    display_sql: str
    headers: tuple[str, ...] | None
    is_exact: bool
    reuse_path: GenerationPath
    sd_reuse: dict[str, Any] | None
    form_storage: QuestionFormStorage | None = None


@dataclass(frozen=True, slots=True)
class QuestionFormStorage:
    """Typo-corrected question text plus optional LLM canonical form for parallel ``ValueHistory`` rows."""

    corrected: str
    normalized_optional: str | None = None
    normalized_negative_memory_dropped: bool = False
    accept_via_normalized_lookup_only: bool = False


class RefinementRetry(Exception):
    """Signals that user rejection was recorded and the caller should re-parse intent within the same ask turn."""

    __slots__ = ()


@dataclass
class RefinementContext:
    """Turn-local state for bounded silent intent refinement after user rejection."""

    corrected_question: str
    form_storage: QuestionFormStorage | None = None
    accumulated_reasons: list[str] = field(default_factory=list)
    refinement_rounds_executed: int = 0
    pending_retry: bool = False
    last_intent_key: str | None = None
    block_further_refinement: bool = False
    skip_refinement_increment_once: bool = False
    conversation_rejection_hints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InteractiveTailSnapshot:
    """Frozen bundle to resume an interactive turn after intent confirmation or a hard-block prompt."""

    q_norm: str
    intent: RuntimeIntent
    schema: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema_terms: set[str]
    dialect: Any
    semantic_warnings: tuple[dict[str, Any], ...]
    has_union_match: bool
    cols_changed: bool
    matched_template: Template | None
    union_select_cols: tuple[Any, ...] | None
    structural_match_templates: tuple[Template, ...]
    ikey: str
    intent_sim: float
    union_sql_path: GenerationPath | None = None
    union_candidate_template_ids: tuple[str, ...] = ()
    form_storage: QuestionFormStorage | None = None


@dataclass(frozen=True, slots=True)
class SqlFeedbackSuspendContext:
    """Resume payload for final SQL accept/reject after execution."""

    tail: InteractiveTailSnapshot
    execution_intent: RuntimeIntent
    sql: str
    rows: tuple[tuple[Any, ...], ...]
    conf: float
    tmpl_sd: dict[str, Any] | None
    gen_out: SqlGenerationOutcome
    matched_rejected_template: Any
    force_feedback: bool


@dataclass(frozen=True, slots=True)
class UserFeedbackRejectSuspendContext:
    """Resume payload when SQL feedback rejection needs a free-text reason (programmatic ``step``)."""

    intent: RuntimeIntent
    sql: str
    schema: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    q_norm: str
    generation_path: GenerationPath | str
    matched_template: Template | None
    matched_rejected_template: Any
    dialect: Any
    structural_match_templates: tuple[Template, ...] | list[Template] | None = None
    rejection_classify_retry_used: bool = False
