"""Intent/template models: normalized expressions, runtime/concrete intents, filters, CTEs, and conversions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, NamedTuple

from ._config import (
    SeedWarmupConfig,
)
from ._constants import (
    FILTER_VALUE_TYPE_DATE_DIFF,
    FILTER_VALUE_TYPE_DATE_WINDOW,
    WINDOW_REGISTRY_AGG_KIND_HINTS,
    WINDOW_REGISTRY_NAV_KIND_HINTS,
    WINDOW_REGISTRY_RANK_KIND_HINTS,
    GenerationPath,
)
from ._contracts_base import (
    QSIM_SUPPORTED_ADVANCED_FEATURES,
    ComplexityTier,
    CteEmissionKind,
    DatabaseFeatureCapability,
    FilterParam,
    HavingParam,
    NormalizedExpr,
    NoveltyBand,
    OperatorFeatureVector,
    OrderByCol,
    ParamValue,
    WarmupStyle,
    WindowOperatorKind,
    WorkloadFamily,
    coerce_cte_emission,
    expr_prompt_sql,
    expr_registry_ref,
)
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    CteOutputColumnMeta,
    ExpansionMetadata,
    SQLShape,
    TemplateStats,
    WindowRegistryStep,
    WindowSpec,
    current_case_registry_steps,
    current_window_registry_steps,
    registry_render_scope,
)


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
    Resolve ``expr`` registry references against optional or context- bound registries.

    Args:

        sc: Select column.
        window_registry: Optional explicit window registry list.
        case_registry: Optional explicit case registry list.

    Returns:

        Effective base expression plus window specification and CASE body from registries when ``expr`` is a bare ``wNN`` / ``cNN`` token.
    """
    wr_seq = tuple(window_registry) if window_registry is not None else current_window_registry_steps()
    cr_seq = tuple(case_registry) if case_registry is not None else current_case_registry_steps()
    win_by_id = {s.registry_id: s for s in wr_seq}
    case_by_id = {s.registry_id: s for s in cr_seq}
    rid = expr_registry_ref(sc.expr) or ""
    if rid.startswith("w"):
        win_step = win_by_id.get(rid)
        if win_step is None:
            return EffectiveSelectParts(sc.expr, None, None)
        return EffectiveSelectParts(NormalizedExpr(), win_step.window_spec, None)
    if rid.startswith("c"):
        case_step = case_by_id.get(rid)
        if case_step is None:
            return EffectiveSelectParts(sc.expr, None, None)
        return EffectiveSelectParts(NormalizedExpr(), None, case_step.case_when)
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
    def output_alias(self) -> str | None:
        """Optional display alias when present in legacy or LLM payloads (not persisted on ``expr``)."""
        return None

    @property
    def is_aggregated(self) -> bool:
        """Whether the column uses SQL aggregation (expr or certain. window funcs). Returns: True from `expr.has_aggregation` or window `sum` / `avg`."""
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
        """Structural key combining base expr and resolved registry. payloads. Union merge, ``display_alias_map``, and template dedupe align runtime and concrete columns by this string rather than by raw SQL text. Returns: Expr signature for bare registry refs; otherwise primary expr signature."""
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
        return {"expr": expr_prompt_sql(self.expr)}

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example select_cols entry."""
        return {"expr": "table.column"}


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
            emission=coerce_cte_emission(d.get("emission")),
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
    """CTE step specification for WITH clause queries with runtime values. At JSON parse time, ``output_columns`` must match ``select_cols`` in length; each name must satisfy ``^[a-z_][a-z0-9_]*$``. Post- processing may rewrite aliases via ``derive_cte_output_columns`` and related repairs so stored ``output_columns`` are canonical, not necessarily identical to the first LLM strings."""

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
            emission=coerce_cte_emission(d.get("emission")),
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
            "group_by_cols": [expr_prompt_sql(g) for g in self.group_by_cols],
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
            "tables": ["table"],
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
    """Runtime intent container for pipeline execution with structural fields and values. The main query body does not carry ``output_columns``; the dialect layer determines the terminal result column list. Each ``RuntimeCteStep`` carries ``output_columns`` for its WITH definition surface."""

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
    planner_cte_names: list[str] = field(default_factory=list)

    @property
    def expected_rows(self) -> str:
        """Coarse row-cardinality hint from `grain` and top-level. `limit`. Returns: `one` for scalar grain, `few` when limited, else `many`."""
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
        "tables": "Tables and CTE names whose columns appear in the query body.",
        "select_cols": "Non-empty SELECT list entries as expr strings or registry tokens.",
        "group_by_cols": "GROUP BY expressions as SQL strings when grouping applies.",
        "order_by_cols": "ORDER BY entries with expr and direction.",
        "filters_param": "Row-level predicates for the main query.",
        "having_param": "Aggregate predicates for the main query.",
        "limit": "Optional integer row cap.",
        "natural_language": "Short description of exactly what the IR returns; no refusal or permission language.",
        "cte_steps": "WITH clause steps using the same body shape as the main intent.",
        "window_registry": "Window definitions scoped to the main query.",
        "case_registry": "CASE registry definitions scoped to the main query.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Main intent shorthand without execution-only fields."""
        return {
            "tables": list(self.tables),
            "select_cols": [sc.to_prompt_dict() for sc in self.select_cols],
            "group_by_cols": [expr_prompt_sql(g) for g in self.group_by_cols],
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
            "tables": ["table", "other_table"],
            "select_cols": [
                SelectCol.prompt_example_dict(),
                {"expr": "COUNT(table.other_column)"},
                {"expr": "COUNT(DISTINCT CONCAT(table.column, other_table.other_column))"},
                {"expr": "w01"},
                {"expr": "c01"},
            ],
            "group_by_cols": ["table.column"],
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
    """Allowed JSON object keys per structural type for LLM intent JSON (positive closure). Nested objects must use only the keys listed for their type; unknown sibling keys should not be emitted. ``sql_expression`` applies wherever an expression is carried as a SQL string (fields named ``expr``, ``left_expr``, ``right_expr``, ``argument``, partition/order entries, ``group_by_cols`` strings, etc.)."""
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
        """Serialize seed warmup intent including optional. `expansion_metadata`. Returns: Plain dict of all fields; `expansion_metadata` only if present."""
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


@dataclass(frozen=True)
class PipelineFeatureSpec:
    """Named SQL capability tag shared by warmup coverage reporting and QSim gating."""

    feature_id: str
    summary: str
    qsim_advanced: bool = False
    expansion_only: bool = False


_EXPANSION_ONLY_PIPELINE_FEATURES: tuple[PipelineFeatureSpec, ...] = (
    PipelineFeatureSpec("in_list", "IN-list membership filters on categorical columns.", expansion_only=True),
    PipelineFeatureSpec("null_filter", "IS NULL / IS NOT NULL filters.", expansion_only=True),
    PipelineFeatureSpec("like_filter", "LIKE pattern filters on categorical strings.", expansion_only=True),
    PipelineFeatureSpec("coalesce_select", "COALESCE-wrapped nullable measures in SELECT.", expansion_only=True),
    PipelineFeatureSpec(
        "string_scalar_select", "String scalar functions in SELECT (upper, trim).", expansion_only=True
    ),
    PipelineFeatureSpec("extract_filter", "EXTRACT-based temporal filters.", expansion_only=True),
    PipelineFeatureSpec("count_distinct", "COUNT(DISTINCT identifier) projections.", expansion_only=True),
    PipelineFeatureSpec("cte_wrap", "Grouped body wrapped in a named CTE.", expansion_only=True),
    PipelineFeatureSpec("numeric_case_label", "Numeric CASE label columns (0/1 flags).", expansion_only=True),
    PipelineFeatureSpec(
        "categorical_case_label", "Categorical CASE label columns with string outputs.", expansion_only=True
    ),
)


def _pipeline_specs_from_qsim_advanced() -> tuple[PipelineFeatureSpec, ...]:
    return tuple(
        PipelineFeatureSpec(
            spec.feature_id,
            spec.summary,
            qsim_advanced=True,
        )
        for spec in QSIM_SUPPORTED_ADVANCED_FEATURES
    )


PIPELINE_FEATURE_SPECS: tuple[PipelineFeatureSpec, ...] = (
    _pipeline_specs_from_qsim_advanced() + _EXPANSION_ONLY_PIPELINE_FEATURES
)
PIPELINE_FEATURE_TAGS: frozenset[str] = frozenset(spec.feature_id for spec in PIPELINE_FEATURE_SPECS)
_PIPELINE_FEATURE_LABELS: dict[str, str] = {spec.feature_id: spec.summary for spec in PIPELINE_FEATURE_SPECS}


def feature_tag_label(tag: str) -> str:
    """Return the human summary for a pipeline feature tag, or the tag itself."""
    return _PIPELINE_FEATURE_LABELS.get(tag, tag)


def _pipeline_feature_feasible_on_capability(feature_id: str, cap: DatabaseFeatureCapability) -> bool:
    """Return whether a pipeline feature tag can plausibly exist on this schema snapshot."""
    if feature_id in ("date_window_filter", "date_diff_shapes", "date_window", "date_diff", "extract_filter"):
        return cap.has_date_columns
    if feature_id in ("unnest_array_column", "unnest"):
        return cap.has_array_columns
    if feature_id in ("multi_cte_chain", "scalar_cte_bridge", "self_join_via_cte", "cte_wrap"):
        return cap.fk_edge_count >= 1 and cap.max_fk_chain_depth >= 1
    if feature_id in ("window_partition_order", "rank_window"):
        return cap.has_window_capable_table_sets
    if feature_id in ("case_when_select", "numeric_case_label", "categorical_case_label"):
        return cap.has_categorical_columns
    if feature_id in ("having_aggregate_compare",):
        return cap.has_numeric_measures
    if feature_id in ("ilike_predicate", "like_filter"):
        return cap.has_categorical_columns
    if feature_id in ("distinct_select", "count_distinct", "in_list", "null_filter"):
        return cap.table_count > 0
    if feature_id in ("coalesce_select", "string_scalar_select"):
        return cap.has_numeric_measures or cap.has_categorical_columns
    return True


def feasible_features_for_capability(cap: DatabaseFeatureCapability) -> frozenset[str]:
    """Return pipeline feature tags achievable on the given schema capability snapshot."""
    return frozenset(tag for tag in PIPELINE_FEATURE_TAGS if _pipeline_feature_feasible_on_capability(tag, cap))


def _runtime_intent_from_detect_input(intent: RuntimeIntent | SeedWarmupIntent) -> RuntimeIntent:
    if isinstance(intent, SeedWarmupIntent):
        return intent.to_runtime_intent()
    return intent


def _select_expr_has_scalar(select_cols: list[SelectCol], names: frozenset[str]) -> bool:
    for sc in select_cols or []:
        expr = sc.expr
        sf = (expr.scalar_func or "").strip().lower()
        if sf in names:
            return True
        isf = (expr.inner_scalar_func or "").strip().lower()
        if isf in names:
            return True
    return False


def _select_has_count_distinct(select_cols: list[SelectCol]) -> bool:
    for sc in select_cols or []:
        expr = sc.expr
        if (expr.agg_func or "").strip().lower() != "count":
            continue
        for group in expr.add_groups or []:
            if group.distinct:
                return True
    return False


def _runtime_filter_date_flags(intent: RuntimeIntent) -> tuple[bool, bool]:
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


def _runtime_json_contains_unnest(intent: RuntimeIntent) -> bool:
    """Heuristic UNNEST detection across the serialized intent footprint."""
    blob = json.dumps(
        {
            "select_cols": [sc.to_dict() if hasattr(sc, "to_dict") else str(sc) for sc in (intent.select_cols or [])],
            "cte_steps": [
                step.to_dict() if hasattr(step, "to_dict") else str(step) for step in (intent.cte_steps or [])
            ],
        },
        sort_keys=True,
        default=str,
    ).lower()
    return "unnest" in blob


def _runtime_has_self_join(intent: RuntimeIntent) -> bool:
    """Detect duplicate table references on the main intent or any CTE branch."""
    if _intent_tables_have_duplicate_tables(intent.tables or []):
        return True
    for step in intent.cte_steps or []:
        if _intent_tables_have_duplicate_tables(step.tables or []):
            return True
    return False


def detect_intent_features(intent: RuntimeIntent | SeedWarmupIntent) -> frozenset[str]:
    """Detect structural pipeline feature tags present on an intent."""
    rt = _runtime_intent_from_detect_input(intent)
    tags: set[str] = set()
    cte_n = len(rt.cte_steps or [])
    if cte_n >= 2:
        tags.add("multi_cte_chain")
    elif cte_n == 1:
        step = rt.cte_steps[0]
        if step.emission == "scalar_subquery":
            tags.add("scalar_cte_bridge")
        else:
            tags.add("cte_wrap")
    if _runtime_has_self_join(rt):
        tags.add("self_join_via_cte")
    win_kind = _window_operator_kind_from_registry(list(rt.window_registry or []))
    if win_kind != "none":
        tags.add("window_partition_order")
        if win_kind == "rank":
            tags.add("rank_window")
    if rt.case_registry:
        tags.add("case_when_select")
        if any(
            (
                branch.result
                and branch.result.add_values
                and any(isinstance(v.value, str) for v in branch.result.add_values)
            )
            for cr in rt.case_registry
            for branch in cr.case_when.branches
        ):
            tags.add("categorical_case_label")
        else:
            tags.add("numeric_case_label")
    date_win, date_diff = _runtime_filter_date_flags(rt)
    if date_win:
        tags.add("date_window_filter")
        tags.add("date_window")
    if date_diff:
        tags.add("date_diff_shapes")
        tags.add("date_diff")
    if rt.distinct_select_index >= 0:
        tags.add("distinct_select")
    if rt.having_param:
        tags.add("having_aggregate_compare")
    if _runtime_json_contains_unnest(rt):
        tags.add("unnest_array_column")
        tags.add("unnest")
    for fp in rt.filters_param or []:
        op = (fp.op or "").strip().lower()
        vt = (fp.value_type or "").strip().lower()
        if op in ("in", "not in") or vt == "string_list":
            tags.add("in_list")
        if op in ("is null", "is not null"):
            tags.add("null_filter")
        if op == "ilike":
            tags.add("ilike_predicate")
        if op == "like":
            tags.add("like_filter")
        if vt == "contains":
            tags.add("array_contains")
        left_sf = ""
        if fp.left_expr is not None:
            left_sf = (fp.left_expr.scalar_func or fp.left_expr.inner_scalar_func or "").strip().lower()
        if left_sf == "extract":
            tags.add("extract_filter")
    if _select_has_count_distinct(list(rt.select_cols or [])):
        tags.add("count_distinct")
    if _select_expr_has_scalar(list(rt.select_cols or []), frozenset({"coalesce"})):
        tags.add("coalesce_select")
    if _select_expr_has_scalar(list(rt.select_cols or []), frozenset({"upper", "lower", "trim", "ltrim", "rtrim"})):
        tags.add("string_scalar_select")
    return frozenset(tags & PIPELINE_FEATURE_TAGS)


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
    """Summarize observable structural operators for diversity metrics. and lattice keys. Args: intent: Warmup intent after normalization and optional expansion. Returns: Frozen footprint aligned with seed- warmup covering-array dimensions."""
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
    Unary and pairwise tags for submodular warmup coverage and.

    expansion scoring. Args: intent: Warmup intent row after expansion.

    Returns:

        Frozen set of hashed coverage atoms including selected second-order
        pairs.
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
    feat_tags = {f"feat:{feat}" for feat in sorted(detect_intent_features(intent))}
    atoms: set[str] = set(cell) | feat_tags
    ordered = sorted(cell | feat_tags)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            atoms.add(f"pair:{ordered[i]}|{ordered[j]}")
    return frozenset(atoms)


def classify_seed_warmup_intent_complexity(intent: SeedWarmupIntent) -> ComplexityTier:
    """Assign a discrete complexity tier from observable structural. features. Uses table count, CTE depth, window and CASE registries, aggregates, GROUP BY, and HAVING. Args: intent: Warmup intent after expansion and substitution. Returns: One of :class:`ComplexityTier`."""
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
    Build a stable lattice cell key for NL reuse across synthetic.

    warmup rows. Args: intent: Warmup intent row after expansion.

    Returns:

        Frozen key tuple used for shared anchor retrieval per schema
        fingerprint.
    """
    tier = classify_seed_warmup_intent_complexity(intent)
    fam = infer_workload_family_for_seed_intent(intent)
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
    """Stable digest string for JSON persistence keyed by lattice cell. and schema hash. Args: key: Lattice coordinates. schema_fp: Effective structural schema fingerprint. Returns: Hex SHA-256 digest for cache files."""
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


def runtime_intent_to_concrete(runtime: RuntimeIntent, intent_id: str) -> ConcreteIntent:
    """Extract `ConcreteIntent` from `RuntimeIntent` for template. storage. Args: runtime: Live intent with `param_values` and optional NL fields. intent_id: Identifier to attach to the stored structural intent. Returns: Structural copy with runtime-only fields stripped and CTE steps concretised."""
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
        Append a paraphrased question row with its accept counter.

        and. optional execution snapshot. Args: question: Stored

        question text for matching (canonical paraphrase). accept_count:

        Per-row accept aggregate used by reuse auto-accept gating.

        param_values: Optional flat parameter dict; defaults to the

        latest row when omitted. natural_language: Optional intent

        description aligned with this row. dedup: When True, skip

        appending when an identical ``question`` row already exists.

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
        """Merge into an existing identical ``question`` row or append. a. new aligned row. Args: param_values: Flat dict of parameter keys to resolved values. question: Natural-language question for this execution. natural_language: Intent description (stored as empty string if omitted). accept_increment: Amount to add to ``accept_counts`` when merging or initializing a row. Returns: None."""
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
    intent_signature: ConcreteIntent
    intent_key: str
    tables_used: list[str]
    sql_param: str
    sql_fp: str
    shape: SQLShape
    colmap_sig: str
    value_history: ValueHistory
    stats: TemplateStats
    effective_structural_hash: str = ""
    schema_graph_id: str = ""
    source: str = "human"
    trust_level: int = 1
    structural_defaults: dict[str, str | int | float] = field(default_factory=dict)
    display_alias_map: dict[str, str] = field(default_factory=dict)
    param_display_names: dict[str, str] = field(default_factory=dict)
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
        """Create Template from dictionary with nested dataclass. reconstruction. Args: d: Dictionary with all Template fields, including nested intent_signature, value_history, stats, and shape dicts. Returns: Fully populated Template instance."""
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
            schema_graph_id=str(d.get("schema_graph_id", "") or ""),
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
            param_display_names=dict(d.get("param_display_names") or {}),
            feedback_by_question=feedback_by_question,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary with nested dataclass. conversion. Returns: Dictionary with all Template fields, with intent_signature, value_history, stats, and shape serialized recursively."""
        intent_sig = (
            self.intent_signature.to_dict() if hasattr(self.intent_signature, "to_dict") else self.intent_signature
        )
        vh_dict = self.value_history.to_dict() if hasattr(self.value_history, "to_dict") else self.value_history
        stats_dict = self.stats.to_dict() if hasattr(self.stats, "to_dict") else self.stats
        return {
            "id": self.id,
            "schema_graph_id": self.schema_graph_id,
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
            "param_display_names": self.param_display_names,
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
    rows: list[Any] | None = None
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
    execute_ok: bool = False

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
            "execute_ok": self.execute_ok,
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
class InterpretPlan:
    """Interpret-stage natural-language plan with optional grounding traceability."""

    approach: str
    tables: tuple[str, ...]
    grounding: tuple[tuple[str, str], ...] = ()
    schema_invalid: bool = False
    missing: str = ""


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
    interpretation: InterpretPlan | None = None


@dataclass(frozen=True, slots=True)
class SqlExecuteSuspendContext:
    """Resume payload for the separated execute step before SQL feedback."""

    tail: InteractiveTailSnapshot
    execution_intent: RuntimeIntent
    sql: str
    gen_out: SqlGenerationOutcome
    matched_rejected_template: Any
    force_feedback: bool
    tmpl_sd: dict[str, Any] | None
    rows: tuple[tuple[Any, ...], ...] = ()
    conf: float | None = None


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
