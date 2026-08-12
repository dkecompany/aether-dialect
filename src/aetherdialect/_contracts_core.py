"""Intent/template models: normalized expressions, runtime/concrete intents, filters, CTEs, and conversions."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, ClassVar, Literal, NamedTuple, Protocol, cast

import pandas

from ._config import PolicyConfig, SeedWarmupConfig
from ._constants import (
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    WHERE_VALUE_TYPE_DATE_DIFF,
    WHERE_VALUE_TYPE_DATE_WINDOW,
)
from ._constants_runtime import REFUSAL_CATALOGUE
from ._contracts_base import (
    QSIM_SUPPORTED_ADVANCED_FEATURES,
    AetherError,
    ApprovalState,
    ComplexityTier,
    CteEmissionKind,
    DatabaseFeatureCapability,
    DataQualityReport,
    Diagnostic,
    EngineContext,
    EngineIdentity,
    FederationContext,
    HavingParam,
    NormalizedExpr,
    NoveltyBand,
    OperatorFeatureVector,
    OrderByCol,
    ParamValue,
    PredicateGroup,
    SchemaAccessError,
    SchemaRole,
    WarmupStyle,
    WhereParam,
    WorkloadFamily,
)
from ._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    CteOutputColumnMeta,
    ExpansionMetadata,
    FederationManifest,
    FederationMappings,
    FederationMappingSuggestion,
    SourceRuntime,
    SQLShape,
    TemplateStats,
    WindowRegistryStep,
    WindowSpec,
)


class GenerationPath(StrEnum):
    """Canonical SQL generation path codes (1, 2.1, 2.2, 3, 4.1, 4.2, 4.3, 5, 6)."""

    EXACT_QUESTION_REUSE = "1"
    FUZZY_REUSE_LITERAL_STRUCTURAL = "2.1"
    FUZZY_REUSE_FULL_PARAMS = "2.2"
    INTENT_DIRECT_MATCH = "3"
    UNION_TEMPLATE_WIDEN = "4.1"
    UNION_TEMPLATE_AND_RUNTIME_WIDEN = "4.2"
    RUNTIME_SUBSET_TEMPLATE_WIDE = "4.3"
    FRESH = "5"
    FEDERATION_PLAN = "6"

    @property
    def code(self) -> str:
        """Return the path code string."""
        return str(self.value)

    @property
    def label(self) -> str:
        """Return a stable readable path label."""
        if self is GenerationPath.EXACT_QUESTION_REUSE:
            return "exact_question_reuse"
        if self is GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL:
            return "fuzzy_reuse_literal_structural"
        if self is GenerationPath.FUZZY_REUSE_FULL_PARAMS:
            return "fuzzy_reuse_full_params"
        if self is GenerationPath.INTENT_DIRECT_MATCH:
            return "intent_direct_match"
        if self is GenerationPath.UNION_TEMPLATE_WIDEN:
            return "union_template_widen"
        if self is GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN:
            return "union_template_and_runtime_widen"
        if self is GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE:
            return "runtime_subset_template_wide"
        if self is GenerationPath.FEDERATION_PLAN:
            return "federation_plan"
        return "fresh"

    @classmethod
    def parse(
        cls,
        value: str | GenerationPath,
    ) -> GenerationPath:
        """Parse enum from code string or enum value. Persisted ``"2"`` maps to :attr:`FUZZY_REUSE_FULL_PARAMS`. Persisted ``"4"`` maps to :attr:`UNION_TEMPLATE_AND_RUNTIME_WIDEN` when disambiguation metadata is absent."""
        if isinstance(value, GenerationPath):
            return value
        s = str(value).strip()
        if s == "2":
            return cls.FUZZY_REUSE_FULL_PARAMS
        if s == "4":
            return cls.UNION_TEMPLATE_AND_RUNTIME_WIDEN
        return cls(s)


class EffectiveSelectParts(NamedTuple):
    """Effective expression and optional window or CASE payloads after registry resolution."""

    expr: NormalizedExpr
    window_spec: WindowSpec | None
    case_when: CaseWhenExpr | None


@dataclass
class SelectCol:
    """Select column with a single normalized SQL expression (including bare ``wNN`` / ``cNN`` registry tokens)."""

    expr: NormalizedExpr = field(default_factory=NormalizedExpr)

    def effective_parts(
        self,
        window_registry: Sequence[WindowRegistryStep] | None = None,
        case_registry: Sequence[CaseRegistryStep] | None = None,
    ) -> EffectiveSelectParts:
        """Resolve ``expr`` registry references against optional window/case registries."""
        wr_seq = tuple(window_registry or ())
        cr_seq = tuple(case_registry or ())
        win_by_id = {s.registry_id: s for s in wr_seq}
        case_by_id = {s.registry_id: s for s in cr_seq}
        rid = self.expr.registry_ref() or ""
        if rid.startswith("w"):
            win_step = win_by_id.get(rid)
            if win_step is None:
                return EffectiveSelectParts(self.expr, None, None)
            return EffectiveSelectParts(NormalizedExpr(), win_step.window_spec, None)
        if rid.startswith("c"):
            case_step = case_by_id.get(rid)
            if case_step is None:
                return EffectiveSelectParts(self.expr, None, None)
            return EffectiveSelectParts(NormalizedExpr(), None, case_step.case_when)
        return EffectiveSelectParts(self.expr, None, None)

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
            expr = NormalizedExpr.parse_string_for_json(expr_raw)
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
        """Optional display alias when present in stored or LLM payloads (not persisted on ``expr``)."""
        return None

    def is_aggregated_with(
        self,
        window_registry: Sequence[WindowRegistryStep] | None = None,
        case_registry: Sequence[CaseRegistryStep] | None = None,
    ) -> bool:
        """Whether the column uses SQL aggregation given optional registries."""
        parts = self.effective_parts(window_registry, case_registry)
        if self.expr.registry_ref() is not None:
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
    def is_aggregated(self) -> bool:
        """Whether the column uses SQL aggregation (expr or certain window funcs)."""
        return self.is_aggregated_with()

    @property
    def signature_key(self) -> str:
        """Structural key combining base expr and resolved registry payloads."""
        return self.expr.signature_key

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "expr": (
            "SQL expression string, DISTINCT-qualified column, aggregation, or bare wNN/cNN registry token (definitions live only in window_registry and case_registry)."
        ),
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """SELECT list column shorthand for LLM JSON."""
        return {"expr": self.expr.prompt_sql()}

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
    where: PredicateGroup | None = None
    having: PredicateGroup | None = None
    output_columns: list[str] = field(default_factory=list)
    grain: str = "row_level"
    limit: int | None = None
    limit_param_key: str = ""
    emission: CteEmissionKind = CteEmissionKind.JOIN_TABLE
    column_map: dict[str, str] = field(default_factory=dict)
    output_column_metadata: dict[str, CteOutputColumnMeta] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1
    preserve_tables: list[str] = field(default_factory=list)
    distinct_on: list[NormalizedExpr] = field(default_factory=list)

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
        where = PredicateGroup.parse_where_field(d)
        having = PredicateGroup.parse_having_field(d)
        ocm_raw = d.get("output_column_metadata", {})
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        do_raw = d.get("distinct_on", [])
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
        distinct_on = [
            (
                NormalizedExpr.from_dict(x)
                if isinstance(x, dict)
                else (NormalizedExpr.from_column(x) if isinstance(x, str) else x)
            )
            for x in do_raw
        ]
        return ConcreteCteStep(
            cte_name=d.get("cte_name", ""),
            tables=d.get("tables", []),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
            output_columns=d.get("output_columns", []),
            grain=d.get("grain", "row_level"),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            emission=CteEmissionKind.coerce(d.get("emission")),
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
            preserve_tables=list(d.get("preserve_tables", [])),
            distinct_on=distinct_on,
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
            "where": self.where.to_dict() if self.where else None,
            "having": self.having.to_dict() if self.having else None,
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
            "preserve_tables": list(self.preserve_tables),
            "distinct_on": [d.to_dict() for d in self.distinct_on],
        }

    def to_runtime(self) -> RuntimeCteStep:
        """Convert this stored concrete CTE step into a runtime step."""
        return RuntimeCteStep(
            cte_name=self.cte_name,
            description="",
            tables=self.tables,
            select_cols=self.select_cols,
            group_by_cols=self.group_by_cols,
            order_by_cols=self.order_by_cols,
            where=self.where,
            having=self.having,
            param_values={},
            output_columns=self.output_columns,
            grain=self.grain,
            limit=self.limit,
            limit_param_key=self.limit_param_key,
            emission=self.emission,
            column_map=self.column_map,
            output_column_metadata=self.output_column_metadata,
            chosen_join_candidate_id=self.chosen_join_candidate_id,
            chosen_join_path_signature=self.chosen_join_path_signature,
            window_registry=list(self.window_registry or []),
            case_registry=list(self.case_registry or []),
            distinct_select_index=self.distinct_select_index,
            preserve_tables=list(self.preserve_tables or []),
            distinct_on=list(self.distinct_on or []),
        )


@dataclass
class RuntimeCteStep:
    """CTE step specification for WITH clause queries with runtime values. At JSON parse time, ``output_columns`` must match ``select_cols`` in length; each name must satisfy ``^[a-z_][a-z0-9_]*$``. Post- processing may rewrite aliases via ``derive_cte_output_columns`` and related repairs so stored ``output_columns`` are canonical, not necessarily identical to the first LLM strings."""

    cte_name: str
    description: str = ""
    tables: list[str] = field(default_factory=list)
    select_cols: list[SelectCol] = field(default_factory=list)
    group_by_cols: list[NormalizedExpr] = field(default_factory=list)
    order_by_cols: list[OrderByCol] = field(default_factory=list)
    where: PredicateGroup | None = None
    having: PredicateGroup | None = None
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)
    grain: str = "row_level"
    limit: int | None = None
    limit_param_key: str = ""
    emission: CteEmissionKind = CteEmissionKind.JOIN_TABLE
    column_map: dict[str, str] = field(default_factory=dict)
    output_column_metadata: dict[str, CteOutputColumnMeta] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    resolved_join_tables: list[str] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1
    preserve_tables: list[str] = field(default_factory=list)
    distinct_on: list[NormalizedExpr] = field(default_factory=list)
    comparison_only_tables: list[str] = field(default_factory=list)

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
        where = PredicateGroup.parse_where_field(d)
        having = PredicateGroup.parse_having_field(d)
        ocm_raw = d.get("output_column_metadata", {})
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        do_raw = d.get("distinct_on", [])
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
        distinct_on = [
            (
                NormalizedExpr.from_dict(x)
                if isinstance(x, dict)
                else (NormalizedExpr.from_column(x) if isinstance(x, str) else x)
            )
            for x in do_raw
        ]
        return RuntimeCteStep(
            cte_name=d.get("cte_name", ""),
            description=d.get("description", ""),
            tables=d.get("tables", []),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
            param_values=d.get("param_values", {}),
            output_columns=d.get("output_columns", []),
            grain=d.get("grain", "row_level"),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            emission=CteEmissionKind.coerce(d.get("emission")),
            column_map=d.get("column_map", {}),
            output_column_metadata={
                k: CteOutputColumnMeta.from_dict(v) if isinstance(v, dict) else v for k, v in ocm_raw.items()
            },
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=d.get("chosen_join_path_signature", []),
            resolved_join_tables=list(d.get("resolved_join_tables", [])),
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
            preserve_tables=list(d.get("preserve_tables", [])),
            distinct_on=distinct_on,
            comparison_only_tables=list(d.get("comparison_only_tables", [])),
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
            "where": self.where.to_dict() if self.where else None,
            "having": self.having.to_dict() if self.having else None,
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
            "resolved_join_tables": list(self.resolved_join_tables),
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
            "preserve_tables": list(self.preserve_tables),
            "distinct_on": [d.to_dict() for d in self.distinct_on],
            "comparison_only_tables": list(self.comparison_only_tables),
        }

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "cte_name": "WITH clause name referenced downstream.",
        "description": "Short explanation of what the CTE computes.",
        "tables": "Tables feeding this CTE body.",
        "select_cols": "SELECT list entries aligned with output_columns length.",
        "group_by_cols": "GROUP BY expressions as SQL strings.",
        "order_by_cols": "ORDER BY entries with expr and direction.",
        "where": "Row-level predicate tree inside the CTE.",
        "having": "Aggregate predicate tree inside the CTE.",
        "limit": "Optional integer row cap.",
        "output_columns": "Snake_case aliases for projected columns.",
        "window_registry": "Window definitions scoped to this CTE.",
        "case_registry": "CASE registry definitions scoped to this CTE.",
        "distinct_on": "Optional DISTINCT ON partition expressions for per-partition row selection.",
        "preserve_tables": "Tables whose rows must survive even when no join partner matches.",
        "emission": (
            "Optional semi_join or anti_join when the question requires existence, absence, or set difference; optional join_table or scalar_subquery as shape hints the engine reclassifies; omit when unsure. "
            "On probe CTEs do not add presence-marker columns or filters, choose join kind, mark distinct, set distinct_on or preserve_tables, or place the probe first in tables."
        ),
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """CTE body shorthand without internal-only bookkeeping fields."""
        emission_value = self.emission.value if isinstance(self.emission, CteEmissionKind) else self.emission
        return {
            "cte_name": self.cte_name,
            "description": self.description,
            "tables": list(self.tables),
            "select_cols": [sc.to_prompt_dict() for sc in self.select_cols],
            "group_by_cols": [g.prompt_sql() for g in self.group_by_cols],
            "order_by_cols": [o.to_prompt_dict() for o in self.order_by_cols],
            "where": self.where.to_prompt_dict() if self.where else None,
            "having": self.having.to_prompt_dict() if self.having else None,
            "limit": self.limit,
            "output_columns": list(self.output_columns),
            "window_registry": [w.to_prompt_dict() for w in self.window_registry],
            "case_registry": [c.to_prompt_dict() for c in self.case_registry],
            "distinct_on": [d.prompt_sql() for d in self.distinct_on],
            "preserve_tables": list(self.preserve_tables or []),
            "emission": emission_value,
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
            "where": None,
            "having": None,
            "limit": None,
            "output_columns": ["out_a"],
            "window_registry": [],
            "case_registry": [],
            "emission": "semi_join",
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

    def to_concrete(self) -> ConcreteCteStep:
        """Convert this runtime CTE step into a concrete template step."""
        return ConcreteCteStep(
            cte_name=self.cte_name,
            tables=self.tables,
            select_cols=self.select_cols,
            group_by_cols=self.group_by_cols,
            order_by_cols=self.order_by_cols,
            where=self.where,
            having=self.having,
            output_columns=self.output_columns,
            grain=self.grain,
            limit=self.limit,
            limit_param_key=self.limit_param_key,
            emission=self.emission,
            column_map=self.column_map,
            output_column_metadata=self.output_column_metadata,
            chosen_join_candidate_id=self.chosen_join_candidate_id,
            chosen_join_path_signature=self.chosen_join_path_signature,
            param_values={},
            window_registry=list(self.window_registry or []),
            case_registry=list(self.case_registry or []),
            distinct_select_index=self.distinct_select_index,
            preserve_tables=list(self.preserve_tables or []),
            distinct_on=list(self.distinct_on or []),
        )


@dataclass(frozen=True)
class TableScopeRepair:
    """Recorded engine adjustment to a scope's table list with an explanatory reason."""

    scope_label: str
    tables: tuple[str, ...]
    action: Literal["add", "remove"]
    reason: Literal["interpret_align", "expression_reference", "unreferenced_table", "join_bridge"]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TableScopeRepair:
        action = str(d.get("action", "add"))
        reason = str(d.get("reason", "interpret_align"))
        return TableScopeRepair(
            scope_label=str(d.get("scope_label", "main query")),
            tables=tuple(str(t) for t in (d.get("tables") or ())),
            action="remove" if action == "remove" else "add",
            reason=cast(
                Literal["interpret_align", "expression_reference", "unreferenced_table", "join_bridge"],
                reason
                if reason in ("interpret_align", "expression_reference", "unreferenced_table", "join_bridge")
                else "interpret_align",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_label": self.scope_label,
            "tables": list(self.tables),
            "action": self.action,
            "reason": self.reason,
        }


@dataclass
class RuntimeIntent:
    """Runtime intent container for pipeline execution with structural fields and values. The main query body does not carry ``output_columns``; the dialect layer determines the terminal result column list. Each ``RuntimeCteStep`` carries ``output_columns`` for its WITH definition surface."""

    tables: list[str] = field(default_factory=list)
    grain: str = ""
    select_cols: list[SelectCol] = field(default_factory=list)
    group_by_cols: list[NormalizedExpr] = field(default_factory=list)
    order_by_cols: list[OrderByCol] = field(default_factory=list)
    where: PredicateGroup | None = None
    having: PredicateGroup | None = None
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    cte_steps: list[RuntimeCteStep] = field(default_factory=list)
    natural_language: str = ""
    limit: int | None = None
    limit_param_key: str = ""
    column_map: dict[str, str] = field(default_factory=dict)
    chosen_join_candidate_id: str = ""
    chosen_join_path_signature: list[str] = field(default_factory=list)
    resolved_join_tables: list[str] = field(default_factory=list)
    window_registry: list[WindowRegistryStep] = field(default_factory=list)
    case_registry: list[CaseRegistryStep] = field(default_factory=list)
    distinct_select_index: int = -1
    sql_param: str = ""
    sql_shape: SQLShape | None = None
    schema_invalid: bool = False
    interpret_cte_names: list[str] = field(default_factory=list)
    preserve_tables: list[str] = field(default_factory=list)
    distinct_on: list[NormalizedExpr] = field(default_factory=list)
    comparison_only_tables: list[str] = field(default_factory=list)
    table_scope_repairs: list[TableScopeRepair] = field(default_factory=list)

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
        where = PredicateGroup.parse_where_field(d)
        having = PredicateGroup.parse_having_field(d)
        cte_raw = d.get("cte_steps", [])
        join_sig_raw = d.get("chosen_join_path_signature", [])
        if isinstance(join_sig_raw, str):
            join_sig_raw = [join_sig_raw] if join_sig_raw else []
        wr_raw = d.get("window_registry", [])
        cr_raw = d.get("case_registry", [])
        do_raw = d.get("distinct_on", [])
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
        distinct_on = [
            (
                NormalizedExpr.from_dict(x)
                if isinstance(x, dict)
                else (NormalizedExpr.from_column(x) if isinstance(x, str) else x)
            )
            for x in do_raw
        ]
        return RuntimeIntent(
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
            param_values=d.get("param_values", {}),
            cte_steps=[RuntimeCteStep.from_dict(cte) if isinstance(cte, dict) else cte for cte in cte_raw],
            natural_language=d.get("natural_language", ""),
            limit=d.get("limit"),
            limit_param_key=d.get("limit_param_key", ""),
            column_map=d.get("column_map", {}),
            chosen_join_candidate_id=d.get("chosen_join_candidate_id", ""),
            chosen_join_path_signature=join_sig_raw,
            resolved_join_tables=list(d.get("resolved_join_tables", [])),
            window_registry=[WindowRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in wr_raw],
            case_registry=[CaseRegistryStep.from_dict(x) if isinstance(x, dict) else x for x in cr_raw],
            distinct_select_index=int(d.get("distinct_select_index", -1)),
            sql_param=d.get("sql_param", ""),
            sql_shape=(SQLShape.from_dict(d["sql_shape"]) if d.get("sql_shape") else None),
            schema_invalid=d.get("schema_invalid", False),
            interpret_cte_names=list(d.get("interpret_cte_names", [])),
            preserve_tables=list(d.get("preserve_tables", [])),
            distinct_on=distinct_on,
            comparison_only_tables=list(d.get("comparison_only_tables", [])),
            table_scope_repairs=[
                TableScopeRepair.from_dict(x) if isinstance(x, dict) else x
                for x in (d.get("table_scope_repairs") or [])
            ],
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
            "where": self.where.to_dict() if self.where else None,
            "having": self.having.to_dict() if self.having else None,
            "param_values": self.param_values,
            "cte_steps": [cte.to_dict() for cte in self.cte_steps],
            "natural_language": self.natural_language,
            "limit": self.limit,
            "limit_param_key": self.limit_param_key,
            "column_map": self.column_map,
            "chosen_join_candidate_id": self.chosen_join_candidate_id,
            "chosen_join_path_signature": self.chosen_join_path_signature,
            "resolved_join_tables": list(self.resolved_join_tables),
            "window_registry": [w.to_dict() for w in self.window_registry],
            "case_registry": [c.to_dict() for c in self.case_registry],
            "distinct_select_index": self.distinct_select_index,
            "sql_param": self.sql_param,
            "sql_shape": self.sql_shape.to_dict() if self.sql_shape else None,
            "schema_invalid": self.schema_invalid,
            "interpret_cte_names": list(self.interpret_cte_names),
            "preserve_tables": list(self.preserve_tables),
            "distinct_on": [d.to_dict() for d in self.distinct_on],
            "comparison_only_tables": list(self.comparison_only_tables),
            "table_scope_repairs": [r.to_dict() for r in self.table_scope_repairs],
        }

    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "tables": "Tables and CTE names whose columns appear in the query body.",
        "select_cols": "Non-empty SELECT list entries as expr strings or registry tokens.",
        "group_by_cols": "GROUP BY expressions as SQL strings when grouping applies.",
        "order_by_cols": "ORDER BY entries with expr and direction.",
        "where": "Row-level predicate tree for the main query.",
        "having": "Aggregate predicate tree for the main query.",
        "limit": "Optional integer row cap.",
        "natural_language": (
            "One conversational sentence restating what answer the user gets; no table/column names, join paths, or SQL jargon."
        ),
        "cte_steps": "WITH clause steps using the same body shape as the main intent.",
        "window_registry": "Window definitions scoped to the main query.",
        "case_registry": "CASE registry definitions scoped to the main query.",
        "distinct_on": "Optional DISTINCT ON partition expressions for per-partition row selection.",
        "preserve_tables": "Tables whose rows must survive even when no join partner matches.",
        "grain": "Query grain: row_level, grouped, or scalar.",
    }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Main intent shorthand without execution-only fields."""
        return {
            "tables": list(self.tables),
            "select_cols": [sc.to_prompt_dict() for sc in self.select_cols],
            "group_by_cols": [g.prompt_sql() for g in self.group_by_cols],
            "order_by_cols": [o.to_prompt_dict() for o in self.order_by_cols],
            "where": self.where.to_prompt_dict() if self.where else None,
            "having": self.having.to_prompt_dict() if self.having else None,
            "limit": self.limit,
            "natural_language": self.natural_language,
            "cte_steps": [cte.to_prompt_dict() for cte in self.cte_steps],
            "window_registry": [w.to_prompt_dict() for w in self.window_registry],
            "case_registry": [c.to_prompt_dict() for c in self.case_registry],
            "distinct_on": [d.prompt_sql() for d in self.distinct_on],
            "preserve_tables": list(self.preserve_tables or []),
            "grain": self.grain or "row_level",
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
            "where": PredicateGroup.prompt_example_dict(),
            "having": None,
            "limit": None,
            "natural_language": "one-line description of the result",
            "window_registry": [
                WindowRegistryStep.prompt_example_dict(),
                WindowRegistryStep.prompt_example_dict_framed(),
            ],
            "case_registry": [CaseRegistryStep.prompt_example_dict()],
            "distinct_on": ["table.column"],
            "preserve_tables": ["other_table"],
            "cte_steps": [
                {
                    **RuntimeCteStep.prompt_example_dict(),
                    "cte_name": "existence_probe",
                    "description": "Entities with a matching row elsewhere.",
                    "output_columns": ["entity_key"],
                    "select_cols": [{"expr": "other_table.entity_key"}],
                    "emission": "semi_join",
                },
                {
                    **RuntimeCteStep.prompt_example_dict(),
                    "cte_name": "absence_probe",
                    "description": "Entities missing from the compared set.",
                    "output_columns": ["entity_key", "compared_value"],
                    "select_cols": [
                        {"expr": "table.entity_key"},
                        {"expr": "table.compared_value"},
                    ],
                    "emission": "anti_join",
                },
            ],
        }

    @classmethod
    def prompt_structural_index(cls) -> dict[str, list[str]]:
        """Allowed JSON object keys per structural type for LLM intent JSON (positive closure)."""
        return {
            "RuntimeIntent": sorted(cls.PROMPT_FIELD_SPEC.keys()),
            "RuntimeCteStep": sorted(RuntimeCteStep.PROMPT_FIELD_SPEC.keys()),
            "SelectCol": sorted(SelectCol.PROMPT_FIELD_SPEC.keys()),
            "OrderByCol": sorted(OrderByCol.PROMPT_FIELD_SPEC.keys()),
            "FilterParam": sorted(WhereParam.PROMPT_FIELD_SPEC.keys()),
            "HavingParam": sorted(HavingParam.PROMPT_FIELD_SPEC.keys()),
            "WindowRegistryStep": sorted(WindowRegistryStep.PROMPT_FIELD_SPEC.keys()),
            "WindowSpec": sorted(WindowSpec.PROMPT_FIELD_SPEC.keys()),
            "CaseRegistryStep": sorted(CaseRegistryStep.PROMPT_FIELD_SPEC.keys()),
            "CaseWhenExpr": sorted(CaseWhenExpr.PROMPT_FIELD_SPEC.keys()),
            "CaseWhenBranch": sorted(CaseWhenBranch.PROMPT_FIELD_SPEC.keys()),
            "sql_expression": sorted(NormalizedExpr.PROMPT_FIELD_SPEC.keys()),
        }

    @property
    def has_aggregation(self) -> bool:
        """Whether any `select_cols` entry is aggregated."""
        return any(s.is_aggregated_with(self.window_registry, self.case_registry) for s in self.select_cols)

    def to_concrete(self, intent_id: str) -> ConcreteIntent:
        """Extract structural ``ConcreteIntent`` for template storage."""
        return ConcreteIntent(
            intent_id=intent_id,
            tables=self.tables,
            grain=self.grain,
            select_cols=self.select_cols,
            group_by_cols=self.group_by_cols,
            order_by_cols=self.order_by_cols,
            where=self.where,
            having=self.having,
            cte_steps=[cte.to_concrete() for cte in self.cte_steps],
            limit=self.limit,
            limit_param_key=self.limit_param_key,
            param_values={},
            column_map=self.column_map,
            chosen_join_candidate_id=self.chosen_join_candidate_id,
            chosen_join_path_signature=self.chosen_join_path_signature,
            window_registry=list(self.window_registry or []),
            case_registry=list(self.case_registry or []),
            distinct_select_index=self.distinct_select_index,
            preserve_tables=list(self.preserve_tables or []),
            distinct_on=list(self.distinct_on or []),
        )

    @staticmethod
    def _tables_have_duplicates(tables: list[str]) -> bool:
        names = [t for t in tables if t]
        return len(names) != len(set(names))

    def _has_self_join(self) -> bool:
        if self._tables_have_duplicates(self.tables or []):
            return True
        for step in self.cte_steps or []:
            if self._tables_have_duplicates(step.tables or []):
                return True
        return False

    def _where_date_flags(self) -> tuple[bool, bool]:
        has_dw = False
        has_dd = False
        for fp in PredicateGroup.where_leaves(self.where):
            vt = (fp.value_type or "").strip().lower()
            if vt in WHERE_VALUE_TYPE_DATE_WINDOW:
                has_dw = True
            if vt in WHERE_VALUE_TYPE_DATE_DIFF:
                has_dd = True
        return has_dw, has_dd

    def _json_contains_unnest(self) -> bool:
        blob = json.dumps(
            {
                "select_cols": [sc.to_dict() for sc in (self.select_cols or [])],
                "cte_steps": [step.to_dict() for step in (self.cte_steps or [])],
            },
            sort_keys=True,
            default=str,
        ).lower()
        return "unnest" in blob

    @staticmethod
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

    @staticmethod
    def _select_has_count_distinct(select_cols: list[SelectCol]) -> bool:
        for sc in select_cols or []:
            expr = sc.expr
            if (expr.agg_func or "").strip().lower() != "count":
                continue
            for group in expr.add_groups or []:
                if group.distinct:
                    return True
        return False

    def detect_features(self) -> frozenset[str]:
        """Detect structural pipeline feature tags present on this intent."""
        tags: set[str] = set()
        cte_n = len(self.cte_steps or [])
        if cte_n >= 2:
            tags.add("multi_cte_chain")
        elif cte_n == 1:
            step = self.cte_steps[0]
            if step.emission == "scalar_subquery":
                tags.add("scalar_cte_bridge")
            else:
                tags.add("cte_wrap")
        if self._has_self_join():
            tags.add("self_join_via_cte")
        win_kind = WindowRegistryStep.operator_kind(list(self.window_registry or []))
        if win_kind != "none":
            tags.add("window_partition_order")
            if win_kind == "rank":
                tags.add("rank_window")
        if self.case_registry:
            tags.add("case_when_select")
            if any(
                (
                    branch.result
                    and branch.result.add_values
                    and any(isinstance(v.value, str) for v in branch.result.add_values)
                )
                for cr in self.case_registry
                for branch in cr.case_when.branches
            ):
                tags.add("categorical_case_label")
            else:
                tags.add("numeric_case_label")
        date_win, date_diff = self._where_date_flags()
        if date_win:
            tags.add("date_window_filter")
            tags.add("date_window")
        if date_diff:
            tags.add("date_diff_shapes")
            tags.add("date_diff")
        if self.distinct_select_index >= 0:
            tags.add("distinct_select")
        if self.having:
            tags.add("having_aggregate_compare")
        if self._json_contains_unnest():
            tags.add("unnest_array_column")
            tags.add("unnest")
        for fp in PredicateGroup.predicate_leaves(self.where):
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
        if self._select_has_count_distinct(list(self.select_cols or [])):
            tags.add("count_distinct")
        if self._select_expr_has_scalar(list(self.select_cols or []), frozenset({"coalesce"})):
            tags.add("coalesce_select")
        if self._select_expr_has_scalar(
            list(self.select_cols or []), frozenset({"upper", "lower", "trim", "ltrim", "rtrim"})
        ):
            tags.add("string_scalar_select")
        return frozenset(tags & PIPELINE_FEATURE_TAGS)


@dataclass
class ConcreteIntent:
    """Structural intent for template storage without values or natural language."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    where: PredicateGroup | None
    having: PredicateGroup | None = None
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
    preserve_tables: list[str] = field(default_factory=list)
    distinct_on: list[NormalizedExpr] = field(default_factory=list)

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
        where = PredicateGroup.parse_where_field(d)
        having = PredicateGroup.parse_having_field(d)
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
        do_raw = d.get("distinct_on", [])
        distinct_on = [
            (
                NormalizedExpr.from_dict(x)
                if isinstance(x, dict)
                else (NormalizedExpr.from_column(x) if isinstance(x, str) else x)
            )
            for x in do_raw
        ]
        return ConcreteIntent(
            intent_id=d.get("intent_id", ""),
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=select_cols,
            group_by_cols=group_by_cols,
            order_by_cols=order_by_cols,
            where=where,
            having=having,
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
            preserve_tables=list(d.get("preserve_tables", [])),
            distinct_on=distinct_on,
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
            "where": self.where.to_dict() if self.where else None,
            "having": self.having.to_dict() if self.having else None,
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
            "preserve_tables": list(self.preserve_tables),
            "distinct_on": [d.to_dict() for d in self.distinct_on],
        }

    def to_runtime_skeleton(self) -> RuntimeIntent:
        """Build a ``RuntimeIntent`` mirroring this concrete intent with empty ``param_values``."""
        return RuntimeIntent(
            tables=list(self.tables or []),
            grain=self.grain or "row_level",
            select_cols=list(self.select_cols or []),
            group_by_cols=list(self.group_by_cols or []),
            order_by_cols=list(self.order_by_cols or []),
            where=self.where,
            having=self.having,
            param_values={},
            cte_steps=[c.to_runtime() for c in (self.cte_steps or [])],
            natural_language="",
            limit=self.limit,
            limit_param_key=self.limit_param_key,
            column_map=dict(self.column_map or {}),
            chosen_join_candidate_id=self.chosen_join_candidate_id or "",
            chosen_join_path_signature=list(self.chosen_join_path_signature or []),
            window_registry=list(self.window_registry or []),
            case_registry=list(self.case_registry or []),
            distinct_select_index=self.distinct_select_index,
            preserve_tables=list(self.preserve_tables or []),
            distinct_on=list(self.distinct_on or []),
        )


@dataclass(frozen=True)
class PipelineFeatureSpec:
    """Named SQL capability tag shared by warmup coverage reporting and QSim gating."""

    feature_id: str
    summary: str
    qsim_advanced: bool = False
    expansion_only: bool = False

    @classmethod
    def from_qsim_advanced(cls) -> tuple[PipelineFeatureSpec, ...]:
        return tuple(
            cls(spec.feature_id, spec.summary, qsim_advanced=True) for spec in QSIM_SUPPORTED_ADVANCED_FEATURES
        )

    @staticmethod
    def label_for(tag: str) -> str:
        """Return the human summary for a pipeline feature tag, or the tag itself."""
        return _PIPELINE_FEATURE_LABELS.get(tag, tag)

    @staticmethod
    def feasible_on_capability(feature_id: str, cap: DatabaseFeatureCapability) -> bool:
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
        if feature_id in ("having_aggregate_compare"):
            return cap.has_numeric_measures
        if feature_id in ("ilike_predicate", "like_filter"):
            return cap.has_categorical_columns
        if feature_id in ("distinct_select", "count_distinct", "in_list", "null_filter"):
            return cap.table_count > 0
        if feature_id in ("coalesce_select", "string_scalar_select"):
            return cap.has_numeric_measures or cap.has_categorical_columns
        return True

    @classmethod
    def feasible_features_for_capability(cls, cap: DatabaseFeatureCapability) -> frozenset[str]:
        """Return pipeline feature tags achievable on the given schema capability snapshot."""
        return frozenset(tag for tag in PIPELINE_FEATURE_TAGS if cls.feasible_on_capability(tag, cap))


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

PIPELINE_FEATURE_SPECS: tuple[PipelineFeatureSpec, ...] = (
    PipelineFeatureSpec.from_qsim_advanced() + _EXPANSION_ONLY_PIPELINE_FEATURES
)
PIPELINE_FEATURE_TAGS: frozenset[str] = frozenset(spec.feature_id for spec in PIPELINE_FEATURE_SPECS)
_PIPELINE_FEATURE_LABELS: dict[str, str] = {spec.feature_id: spec.summary for spec in PIPELINE_FEATURE_SPECS}


@dataclass(frozen=True)
class AnchorLatticeKey:
    """Stable coordinates for NL anchor reuse across synthetic warmup rows."""

    family: WorkloadFamily
    tier: ComplexityTier
    style: WarmupStyle
    novelty_band: NoveltyBand

    def signature(self, schema_fp: str) -> str:
        """Stable digest string for JSON persistence keyed by lattice cell and schema hash."""
        payload = "|".join(
            [
                self.family.value,
                self.tier.value,
                self.style.value,
                self.novelty_band.value,
                schema_fp,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


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
        self, param_values: dict[str, ParamValue], question: str, natural_language: str, *, accept_increment: int = 1
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


class RejectionBucket(StrEnum):
    """High-level category for user rejection feedback (agnostic labels)."""

    MISSING_FILTER = "MISSING_FILTER"
    WRONG_GROUPING = "WRONG_GROUPING"
    WRONG_AGGREGATION = "WRONG_AGGREGATION"
    WRONG_TIME_RANGE = "WRONG_TIME_RANGE"
    WRONG_TABLES_OR_JOINS = "WRONG_TABLES_OR_JOINS"
    WRONG_SORT_OR_LIMIT = "WRONG_SORT_OR_LIMIT"
    OTHER = "OTHER"
    MALFORMED_MEMBER_ANSWER = "MALFORMED_MEMBER_ANSWER"
    JOIN_FAN_OUT = "JOIN_FAN_OUT"

    @classmethod
    def from_raw(cls, raw: Any) -> RejectionBucket:
        """Coerce a stored string to ``RejectionBucket``, defaulting to OTHER."""
        s = str(raw or "").strip()
        for m in cls:
            if m.value == s:
                return m
        return cls.OTHER


class FeedbackKind(StrEnum):
    """Discriminator for stored question feedback entries."""

    VALIDATION_FAILURE = "validation_failure"
    INTENT_REJECTED = "intent_rejected"

    @classmethod
    def from_raw(cls, raw: Any) -> FeedbackKind:
        """Coerce a stored string to ``FeedbackKind``, defaulting to validation failure."""
        s = str(raw or "").strip()
        for m in cls:
            if m.value == s:
                return m
        return cls.VALIDATION_FAILURE


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
    member_source_id: str = ""
    rejected_join_path_signature: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        out: dict[str, Any] = {
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
        if self.member_source_id:
            out["member_source_id"] = self.member_source_id
        if self.rejected_join_path_signature:
            out["rejected_join_path_signature"] = list(self.rejected_join_path_signature)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QuestionFeedbackEntry:
        """Deserialize from ``to_dict`` output or single-bucket rows."""
        raw_b = d.get("buckets")
        bkt_tuple: tuple[RejectionBucket, ...]
        if isinstance(raw_b, list) and raw_b:
            bkt_tuple = tuple(RejectionBucket.from_raw(x) for x in raw_b)
        else:
            bkt_tuple = (RejectionBucket.from_raw(d.get("bucket")),)
        created = str(d.get("created_at", "") or "")
        updated = str(d.get("updated_at", "") or "") or created
        src: Literal["engine"] = "engine"
        raw_sig = d.get("rejected_join_path_signature")
        if isinstance(raw_sig, list):
            rej_sig = tuple(str(x).strip() for x in raw_sig if str(x).strip())
        else:
            rej_sig = ()
        return QuestionFeedbackEntry(
            summary=str(d.get("summary", "") or ""),
            buckets=bkt_tuple,
            kind=FeedbackKind.from_raw(d.get("kind")),
            effective_structural_hash=str(d.get("effective_structural_hash", "") or ""),
            intent_structural_hash=str(d.get("intent_structural_hash", "") or ""),
            intent_payload=str(d.get("intent_payload", "") or ""),
            created_at=created,
            updated_at=updated,
            is_post_restart=bool(d.get("is_post_restart", False)),
            source=src,
            member_source_id=str(d.get("member_source_id", "") or ""),
            rejected_join_path_signature=rej_sig,
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
    member_source_id: str = ""
    member_engine: str = ""
    federation_plan_id: str = ""
    federation_plan_only: bool = False
    schema_column_types: dict[str, str] = field(default_factory=dict)
    footprint_tables: tuple[str, ...] = ()
    footprint_columns: tuple[str, ...] = ()
    approval_state: ApprovalState = ApprovalState.APPROVED

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

    @staticmethod
    def _parse_approval_state(raw: Any) -> ApprovalState:
        """Normalize a stored approval_state value to :class:`ApprovalState`."""
        if isinstance(raw, ApprovalState):
            return raw
        text = str(raw or "").strip().lower()
        if text == ApprovalState.PENDING.value:
            return ApprovalState.PENDING
        return ApprovalState.APPROVED

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
            effective_structural_hash=str(d.get("effective_structural_hash", "") or ""),
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
            member_source_id=str(d.get("member_source_id", "") or ""),
            member_engine=str(d.get("member_engine", "") or ""),
            federation_plan_id=str(d.get("federation_plan_id", "") or ""),
            federation_plan_only=bool(d.get("federation_plan_only", False)),
            schema_column_types={
                str(k): str(v)
                for k, v in (d.get("schema_column_types") or {}).items()
                if str(k).strip() and str(v).strip()
            },
            footprint_tables=tuple(str(x).strip() for x in (d.get("footprint_tables") or ()) if str(x).strip()),
            footprint_columns=tuple(str(x).strip() for x in (d.get("footprint_columns") or ()) if str(x).strip()),
            approval_state=Template._parse_approval_state(d.get("approval_state")),
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
            "member_source_id": self.member_source_id,
            "member_engine": self.member_engine,
            "federation_plan_id": self.federation_plan_id,
            "federation_plan_only": self.federation_plan_only,
            "schema_column_types": dict(self.schema_column_types),
            "footprint_tables": list(self.footprint_tables),
            "footprint_columns": list(self.footprint_columns),
            "approval_state": self.approval_state.value
            if isinstance(self.approval_state, ApprovalState)
            else str(self.approval_state or ApprovalState.APPROVED.value),
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
    refusal_diagnostic_code: str | None = None
    explain_soft_diagnostics: int = 0
    explain_soft_findings: tuple[Any, ...] = ()
    federated_steps: tuple[Any, ...] = ()
    federation_plan_id: str = ""
    federation_dir: str = ""


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


class RefinementRetry(AetherError):
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
class TurnPolicySnapshot:
    """Per-turn policy knobs frozen at suspend so resume uses the same gates."""

    max_compose_repairs: int
    max_interpret_ground_retries: int
    trust_auto_accept_threshold: int


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
    preview_rows: tuple[tuple[Any, ...], ...] = ()
    sql_parameters: tuple[tuple[str, Any], ...] = ()
    suspended_at: datetime | None = None
    federated_prepare: FederatedPrepareOutcome | None = None
    federation_plan_id: str = ""
    federation_exec_context: tuple[tuple[str, Any], ...] = ()
    turn_policy: TurnPolicySnapshot | None = None


@dataclass(frozen=True, slots=True)
class SqlFeedbackSuspendContext:
    """Resume payload for final SQL accept/reject after execution."""

    tail: InteractiveTailSnapshot
    execution_intent: RuntimeIntent
    sql: str
    preview_rows: tuple[tuple[Any, ...], ...]
    sql_parameters: tuple[tuple[str, Any], ...]
    suspended_at: datetime | None
    tmpl_sd: dict[str, Any] | None
    gen_out: SqlGenerationOutcome
    matched_rejected_template: Any
    force_feedback: bool
    federated_prepare: FederatedPrepareOutcome | None = None
    federated_bundle: FederatedSqlBundle | None = None


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


@dataclass(frozen=True, slots=True)
class SourceStep:
    """One per-source sub-plan produced by federation decomposition."""

    source_id: str
    sub_intent: RuntimeIntent
    projected_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JoinSpec:
    """Declared cross-source join between two materialized source frames."""

    left_source: str
    right_source: str
    left_key: str
    right_key: str
    logical_key: str
    kind: str


@dataclass(frozen=True, slots=True)
class UnionSpec:
    """Combine frames from multiple sources into one logical table."""

    logical_table: str
    member_source_ids: tuple[str, ...]
    semantics: Literal["union", "replica"]


@dataclass(frozen=True, slots=True)
class FederationReducingEdge:
    """One upstream key-reduction edge feeding a member stage."""

    driving_source_id: str
    target_source_id: str
    driving_key: str
    target_key: str
    edge_kind: Literal["semijoin", "filter_keys"] = "semijoin"


@dataclass(frozen=True, slots=True)
class ResidualSpec:
    """Coordinator-level clauses that span more than one source."""

    select_cols: tuple[SelectCol, ...] = ()
    group_by_cols: tuple[NormalizedExpr, ...] = ()
    order_by_cols: tuple[OrderByCol, ...] = ()
    where: PredicateGroup | None = None
    having: PredicateGroup | None = None
    distinct_on: tuple[NormalizedExpr, ...] = ()
    distinct_select_index: int = -1
    limit: int | None = None
    limit_param_key: str = ""
    window_registry: tuple[Any, ...] = ()
    case_registry: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class FederatedStage:
    """One stage in a federated execution graph (member fragment, spanning CTE, or coordinator)."""

    stage_id: str
    kind: Literal["member", "coordinator", "cte"]
    source_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    reducing_edges: tuple[FederationReducingEdge, ...] = ()
    spanning_cte_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FederationExecutionWave:
    """One schedulable wave in a federated execution graph."""

    stage: FederatedStage
    member_steps: tuple[SourceStep, ...] = ()


@dataclass(slots=True)
class FederationExecutionContext:
    """Cancellation and attribution state for one federated execution turn."""

    plan_id: str = ""
    temporal_bind: AnchoredTemporalBind | None = None
    audit_emit: Any | None = field(default=None, repr=False, compare=False)
    plan_started_monotonic: float | None = None
    plan_deadline_monotonic: float | None = None
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        """Return True when the federated turn has been cancelled."""
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next federation stage or batch boundary. Does not interrupt an already-started member database statement; workers finish or fail on their own, and the coordinator stops before starting the next stage."""
        self._cancel_event.set()


@dataclass(frozen=True, slots=True)
class FederatedPlan:
    """Deterministic federation decomposition for one validated intent."""

    steps: tuple[SourceStep, ...]
    combine: tuple[JoinSpec, ...] | None = None
    union_specs: tuple[UnionSpec, ...] = ()
    residual: ResidualSpec | None = None
    ineligible_reason: str | None = None
    stages: tuple[FederatedStage, ...] = ()
    grain: str = "row_level"
    scope_sources: frozenset[str] = frozenset()
    lifted_probe_ctes: tuple[RuntimeCteStep, ...] = ()


@dataclass(frozen=True, slots=True)
class FederatedPreparedStep:
    """One generated per-source SQL bundle awaiting coordinator execution."""

    source_id: str
    sub_intent: RuntimeIntent
    sql: str
    structural_defaults: dict[str, Any] | None = None
    matched_template: Any | None = None


@dataclass(frozen=True, slots=True)
class FederatedStatementRecord:
    """One executed SQL statement in a federated turn (member source or coordinator)."""

    source_id: str
    engine: str
    statement: str
    row_count: int = 0
    read_instant: str = ""
    row_cap: int | None = None
    timeout_ms: int | None = None
    duration_ms: int | None = None
    failure: str | None = None
    phase: Literal["member", "coordinator"] = "member"
    combine_kind: str = ""


@dataclass(frozen=True, slots=True)
class FederatedSqlBundle:
    """Structured federated execution artifact: per-source statements plus coordinator glue."""

    statements: tuple[FederatedStatementRecord, ...]
    display_sql: str = ""
    column_names: tuple[str, ...] = ()
    read_window: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class FederatedExecutionOutcome:
    """Coordinator rows and the structured statement bundle for one federated execution."""

    rows: tuple[tuple[Any, ...], ...]
    bundle: FederatedSqlBundle


@dataclass(frozen=True, slots=True)
class FederationMemberResolvedLimits:
    """Resolved per-member execution limits after manifest and policy fallbacks."""

    source_id: str
    row_cap: int
    timeout_ms: int
    max_query_cost_rows: float | None
    max_query_cost_bytes: float | None
    profile_timeout_ms: int | None


@dataclass(frozen=True, slots=True)
class FederatedPrepareOutcome:
    """Per-source SQL generation for a federated plan (execution deferred)."""

    success: bool
    plan: FederatedPlan
    display_sql: str
    steps: tuple[FederatedPreparedStep, ...] = ()
    per_source_sql: tuple[tuple[str, str], ...] = ()
    glue_sql: str = ""
    bundle: FederatedSqlBundle | None = None
    sql_validation_error: str | None = None
    error_kind: str | None = None
    source_id: str = ""
    phase: Literal["prepare", "member", "coordinator"] = "prepare"
    composite_schema_graph_id: str = ""
    combine_hash: str = ""
    step_fingerprints: tuple[tuple[str, str], ...] = ()
    member_schema_graph_ids: tuple[tuple[str, str], ...] = ()
    member_resolved_limits: tuple[FederationMemberResolvedLimits, ...] = ()


@dataclass(frozen=True, slots=True)
class FederatedSqlOutcome:
    """Result of federated per-source SQL generation, execution, and coordinator join."""

    success: bool
    sql: str
    rows: tuple[tuple[Any, ...], ...]
    per_source_sql: tuple[tuple[str, str], ...] = ()
    bundle: FederatedSqlBundle | None = None
    prepared: FederatedPrepareOutcome | None = None
    sql_validation_error: str | None = None
    error_kind: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionReuseMatch:
    """Trusted template history row selected by fuzzy token match against a candidate question."""

    template_id: str
    history_index: int
    stored_normalized_text: str
    candidate_normalized: str
    token_edit_sum: int

    @property
    def is_exact_string_reuse(self) -> bool:
        """True when normalized candidate text equals the stored normalized history string (generation path 1)."""
        return self.candidate_normalized == self.stored_normalized_text


class ScopeClass(StrEnum):
    """Join candidate mixture for per-scope disambiguation policy."""

    single_table = "single_table"
    single_fk = "single_fk"
    multi_fk = "multi_fk"
    single_fk_with_semantic = "single_fk_with_semantic"
    multi_fk_with_semantic = "multi_fk_with_semantic"
    semantic_only = "semantic_only"
    empty = "empty"


class UnionSelectColumnDelta(StrEnum):
    """Select-list delta between runtime intent and template concrete intent (non-aggregated keys)."""

    EQUAL = "equal"
    TEMPLATE_ONLY_EXTRA = "template_only_extra"
    INTENT_ONLY_EXTRA = "intent_only_extra"
    BOTH_EXTRA = "both_extra"


@dataclass
class StepResult:
    """Mutable capture of pipeline outputs, metrics, and logs for one scenario run."""

    scenario_id: str
    question: str
    status: str = "unknown"
    intent: RuntimeIntent | None = None
    sql: str | None = None
    rows: list[tuple[Any, ...]] | None = None
    confidence: float | None = None
    reuse_type: str | None = None
    template_id: str | None = None
    validation_failed: bool = False
    feedback: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    captured_logs: list[str] = field(default_factory=list)
    semantic_warnings: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    reject_reason_actual: str | None = None
    classified_category: str | None = None
    classified_reason: str | None = None
    generation_path: str | None = None
    pending_feedback: Any | None = None
    diagnostics: tuple[Any, ...] = ()
    kind: str | None = None


@dataclass
class RestartBudget:
    """Mutable counter bounding the number of fresh full-parse restarts per top-level invocation."""

    fresh_restarts_left: int

    @classmethod
    def default(cls) -> RestartBudget:
        """Construct a budget initialised from :data:`PolicyConfig.MAX_FRESH_RESTARTS`."""
        return cls(fresh_restarts_left=PolicyConfig.MAX_FRESH_RESTARTS)


@dataclass(frozen=True, slots=True)
class TrustedTemplateHit:
    """Trusted template row selected by fuzzy question match against ``value_history`` (paths 2.x)."""

    template: Template
    reuse_hit: QuestionReuseMatch


@dataclass(frozen=True, slots=True)
class FederationTableSet:
    """Tables referenced by an intent with per-table source attribution."""

    tables: frozenset[str]
    source_by_table: Mapping[str, str]
    sources: frozenset[str]


@dataclass(frozen=True)
class AnchoredTemporalBind:
    """Single turn-start anchor for relative temporal predicates in federated member SQL."""

    anchor_iso: str


@dataclass(frozen=True, slots=True)
class CoordinatorMemberFrame:
    """Per-member coordinator input delivered as Arrow or pandas."""

    kind: Literal["arrow", "pandas"]
    table: Any
    column_names: tuple[str, ...]

    def row_count(self) -> int:
        if self.kind == "arrow":
            return int(self.table.num_rows)
        return len(self.table)

    def memory_bytes(self) -> int:
        if self.kind == "arrow":
            return int(self.table.nbytes)
        usage = self.table.memory_usage(deep=True)
        return int(usage.sum())


@dataclass
class PendingFeedback:
    """Deferred feedback payload for post-assertion commit."""

    choice: str
    intent: RuntimeIntent
    sql: str
    schema: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    q_norm: str
    generation_path: GenerationPath
    matched_template: Any
    matched_rejected_template: Any | None
    dialect: Any
    canned_reject_reason: str = ""
    structural_match_templates: tuple[Any, ...] = ()
    join_matches_template: bool | None = None


@dataclass
class Expected:
    """Optional checks for one run; `None` or defaults skip the corresponding assertion. When ``reuse_type`` is checked, values align with pipeline routing: ``direct_reuse`` (question match, ``GenerationPath`` 1–2), ``intent_direct_reuse`` (union, same columns, path 3), ``intent_reuse`` (union, columns changed, path 4)."""

    tables: list[str] | None = None
    tables_one_of: list[list[str]] | None = None
    grain_in: tuple[str, ...] | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    reuse_type: str | tuple[str, ...] | None = None
    contains_join: bool | None = None
    contains_group_by: bool | None = None
    contains_cte: bool | None = None
    sql_contains: list[str] | None = None
    sql_contains_one_of: list[list[str]] | None = None
    sql_excludes: list[str] | None = None
    grain: str | tuple[str, ...] | None = None
    should_fail_validation: bool = False
    column_names_one_of: list[list[str]] | None = None
    row_value_check: Callable[[list[tuple[Any, ...]]], bool] | None = None
    min_semantic_warnings: int | None = None
    status: str | None = None
    status_in: tuple[str, ...] | None = None
    generation_path: str | None = None
    generation_path_in: tuple[str, ...] | None = None
    max_llm_calls: int | None = None


@dataclass
class Scenario:
    """One NL question, expectations, canned prompts, and metadata for a live run."""

    id: str
    question: str
    expected: Expected
    category: str = ""
    auto_responses: list[str] | None = None
    feedback: str = "y"
    reject_reason: str = "incorrect results"
    sequence_id: str | None = None


@dataclass
class SequenceScenario:
    """Ordered scenarios sharing template state for stateful live tests."""

    id: str
    steps: list[Scenario]
    category: str = ""


@dataclass
class SoftFailure:
    """One recorded mismatch from a soft assertion."""

    field: str
    expected: Any
    actual: Any
    message: str


class SoftAssert:
    """Collect soft assertion failures; call `report()` to raise one combined error."""

    def __init__(self) -> None:
        """Initialize an empty failure list."""
        self.failures: list[SoftFailure] = []

    def check(self, condition: bool, field_name: str, expected: Any, actual: Any, message: str = "") -> None:
        """Append a `SoftFailure` when `condition` is false."""
        if not condition:
            msg = message or f"{field_name}: expected {expected!r}, got {actual!r}"
            self.failures.append(SoftFailure(field=field_name, expected=expected, actual=actual, message=msg))

    @property
    def passed(self) -> bool:
        """True if no failures were recorded."""
        return len(self.failures) == 0

    def report(self, header: str = "") -> None:
        """Raise `AssertionError` with all failures, or return if. `passed`."""
        if self.passed:
            return
        lines = [header] if header else []
        for f in self.failures:
            lines.append(f"  [{f.field}] {f.message}")
        raise AssertionError("\n".join(lines))


@dataclass
class LiveTestRunner:
    """Live-pipeline scenario runner state. ``run`` / ``run_deferred`` are registered by ``aetherdialect._live_testing``."""

    schema: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema_terms: set[str]
    csv_dir: str = ""
    dialect: Any | None = None
    run_impl: ClassVar[Any] = None
    run_deferred_impl: ClassVar[Any] = None

    def run(self, scenario: Scenario, retries: int = 0) -> Any:
        """Execute a single scenario against the live pipeline."""
        impl = type(self).run_impl
        if impl is None:
            raise RuntimeError("LiveTestRunner.run is not registered")
        return impl(self, scenario, retries=retries)

    def run_deferred(self, scenario: Scenario, retries: int = 0) -> Any:
        """Execute one scenario while deferring feedback persistence."""
        impl = type(self).run_deferred_impl
        if impl is None:
            raise RuntimeError("LiveTestRunner.run_deferred is not registered")
        return impl(self, scenario, retries=retries)

    def clone(self) -> LiveTestRunner:
        """Return an isolated runner with deep-copied mutable state."""
        return LiveTestRunner(
            schema=self.schema,
            store=deepcopy(self.store),
            templates=deepcopy(self.templates),
            rejected=deepcopy(self.rejected),
            schema_terms=set(self.schema_terms),
            csv_dir=self.csv_dir,
            dialect=self.dialect,
        )

    def adopt_state_from(self, other: LiveTestRunner) -> None:
        """Replace mutable state with another runner's state."""
        self.store = other.store
        self.templates = other.templates
        self.rejected = other.rejected
        self.schema_terms = other.schema_terms


class AccessError(SchemaAccessError, RuntimeError):
    """Raised when execute/explain is refused by library scope or the warehouse."""

    def __init__(
        self,
        operation: Literal["explain", "execute"],
        message: str,
        *,
        relation: str | None = None,
        reason: Literal["scope", "warehouse"] = "warehouse",
    ) -> None:
        """Attach *operation*, human *message*, optional *relation*, and access *reason*."""
        self.operation = operation
        self.relation = relation
        self.reason = reason
        super().__init__(message)


class PipelineSuspended(AetherError):
    """Raised when a programmatic interactive turn must wait for the next ``submit_*`` call."""

    def __init__(self, state_id: str, message_for_caller: str, payload: Any | None = None) -> None:
        self.state_id = state_id
        self.message_for_caller = message_for_caller
        self.payload = payload
        super().__init__(message_for_caller)


class NoJoinPathError(AetherError):
    """Raised when multi-table scope has no foreign-key or semantic join path. This is a terminal, deterministic pipeline failure: no LLM call can invent a plausible join when neither the physical foreign-key graph nor the semantic edge set connects the requested tables."""

    def __init__(self, scope_label: str, tables: list[str]) -> None:
        self.scope_label = scope_label
        self.tables = list(tables)
        message = (
            f"No join path available in {scope_label} for tables: {', '.join(self.tables) if self.tables else '<none>'}"
        )
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation naming the disconnected tables."""
        tables = ", ".join(self.tables) if self.tables else "the requested tables"
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE]["user_text"].format(tables=tables)


class JoinPathTieCapExceededError(AetherError):
    """Raised when shortest-path enumeration for one table pair exceeds the refusal ceiling."""

    def __init__(self, source_table: str, target_table: str, path_count: int, ceiling: int) -> None:
        self.source_table = source_table
        self.target_table = target_table
        self.path_count = path_count
        self.ceiling = ceiling
        super().__init__(
            f"join path tie ceiling exceeded for {source_table!r} -> {target_table!r}: "
            f"{path_count} equal-length paths (limit {ceiling})"
        )

    @property
    def user_message(self) -> str:
        """User-facing explanation when too many equally short join paths exist."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP]["user_text"].format(
            source_table=self.source_table,
            target_table=self.target_table,
            path_count=str(self.path_count),
            ceiling=str(self.ceiling),
        )


class AggregateJoinFanOutError(AetherError):
    """Raised when a resolved join path would duplicate rows aggregated at parent grain."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class ClauseWidenedRowsetError(AetherError):
    """Raised when LIMIT or DISTINCT ON would run on a join-widened row set."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation when clause modifiers conflict with join shape."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET]["user_text"]


class NullInNegatedListError(AetherError):
    """Raised when a NOT IN list literal contains null."""

    def __init__(self, column: str, message: str) -> None:
        self.column = column
        self.message_for_caller = message
        super().__init__(message)


class SubdayDateWindowOnDateColumnError(AetherError):
    """Raised when a sub-day date window is requested on a date-only column."""

    def __init__(self, column: str, message: str) -> None:
        self.column = column
        self.message_for_caller = message
        super().__init__(message)


class AmbiguousDateLiteralError(AetherError):
    """Raised when an absolute date bound is not valid ISO 8601."""

    def __init__(self, literal: str, message: str) -> None:
        self.literal = literal
        self.message_for_caller = message
        super().__init__(message)


class ProbeCtePlacementError(AetherError):
    """Raised when a semi-join or anti-join probe CTE is used as a join anchor or left operand."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)

    @property
    def user_message(self) -> str:
        """User-facing explanation when a filter step is placed incorrectly in the join."""
        return REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT]["user_text"]


class ComparisonJoinScopeExceededError(AetherError):
    """Raised when a cross-table comparison forces a join path beyond the allowed scope."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class JoinColumnCountMismatchError(AetherError):
    """Raised when a join signature pairs unequal numbers of left and right columns."""

    def __init__(self, segment: str, left_count: int, right_count: int) -> None:
        self.segment = segment
        self.left_count = left_count
        self.right_count = right_count
        super().__init__(
            f"join path segment {segment!r} pairs {left_count} left column(s) with {right_count} right column(s)"
        )


class JoinInjectionAlignmentError(AetherError):
    """Raised when ``join_sigs_ordered`` does not align one-to-one with dialect join carriers on deterministic SQL."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RegistryRenderError(AetherError, ValueError):
    """Raised when rendering an expression references a missing window or case registry id."""


class JoinInjectionFailedError(AetherError):
    """Raised when deterministic SQL cannot be rewritten with structured JOIN/WHERE edges via the dialect AST adapter."""

    def __init__(
        self, message: str, *, det_sql: str, join_sigs_ordered: list[list[str]], edge_kinds_ordered: list[list[str]]
    ) -> None:
        self.det_sql = det_sql
        self.join_sigs_ordered = join_sigs_ordered
        self.edge_kinds_ordered = edge_kinds_ordered
        super().__init__(message)


class LlmJsonExhausted(AetherError):
    """Raised by ``llm_json`` when every retry attempt fails to produce valid JSON. Callers decide whether exhaustion is recoverable (e.g., retry loops, deterministic fallbacks) or terminal."""

    def __init__(self, task: str, attempts: int) -> None:
        self.task = task
        self.attempts = attempts
        super().__init__(f"llm_json exhausted after {attempts} attempt(s) for task={task!r}")


@dataclass(frozen=True, slots=True)
class LlmBatchRequest:
    """One JSON-mode completion submitted through the OpenAI Batch API."""

    custom_id: str
    system: str
    user: str
    task: str = "default"


class JoinCandidateCapExceededError(AetherError):
    """Raised when join path cross-product enumeration exceeds the refusal cap."""

    def __init__(
        self,
        enumerated: int,
        cap: int,
        *,
        tables: list[str] | None = None,
        root: str | None = None,
    ) -> None:
        self.enumerated = enumerated
        self.cap = cap
        self.tables = list(tables) if tables is not None else None
        self.root = root
        tables_text = ",".join(self.tables) if self.tables else "?"
        root_text = f" root={self.root!r}" if self.root else ""
        super().__init__(
            f"join candidate cross-product cap exceeded: {enumerated} paths (limit {cap}) tables={tables_text}{root_text}"
        )


class JoinProbeEdgeKindMismatchError(AetherError):
    """Raised when join path signature and edge-kind lists are not aligned."""

    def __init__(self, signature_len: int, kinds_len: int) -> None:
        self.signature_len = signature_len
        self.kinds_len = kinds_len
        super().__init__(f"join path edge_kinds length mismatch: {kinds_len} kinds for {signature_len} segments")


class JoinPathKeyTypeError(AetherError):
    """Raised when a resolved join path pairs incompatible column types."""

    def __init__(self, scope_label: str, message: str) -> None:
        self.scope_label = scope_label
        self.message_for_caller = message
        super().__init__(message)


class QuestionRoute(StrEnum):
    """Validation gate route for an ask turn."""

    ANALYTICAL = "analytical"
    SCHEMA_CATALOG = "schema_catalog"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    SCHEMA_AND_KNOWLEDGE = "schema_and_knowledge"
    RESTRICTED = "restricted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class QuestionValidationResult:
    """Result of the ask-path question validation LLM gate."""

    accepted: bool
    route: QuestionRoute
    corrected: str
    invalid_kind: str | None = None


class ResultReaderKind(StrEnum):
    """Row-fetch backend identifier used by dialect execution paths."""

    SQLALCHEMY = "sqlalchemy"
    SPARK = "spark"
    CONNECTOR = "connector"
    BQ_CLIENT = "bq_client"
    BQ_STORAGE = "bq_storage"
    SNOWFLAKE_ARROW = "snowflake_arrow"


class FeedbackMode(StrEnum):
    """Whether interactive feedback is collected live or deferred in tests."""

    LIVE = "live"
    DEFERRED_TEST = "deferred_test"


@dataclass(frozen=True, slots=True)
class LlmExecutionConfig:
    """Merged Azure OpenAI credentials plus execution cost and timeout limits for the engine runtime. Public operators configure two deployment slots named ``LIGHT`` and ``HEAVY`` that provision Azure deployments sized for the ``gpt-5-mini`` and ``gpt-5.4-mini`` model classes respectively. Internal routing from logical model identifiers to these slots is not part of the public stability contract."""

    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    deployment_light: str
    deployment_heavy: str
    max_query_cost_rows: int
    max_query_cost_bytes: int
    statement_timeout_ms: int
    llm_timeout_ms: int
    profile_timeout_ms: int
    explain_timeout_ms: int | None


class InteractiveChoicePort(Protocol):
    """Routes yes/no prompts to a session queue or stdin."""

    _pending_federation_plan_template: Any

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        ...

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Return a normalised choice or raise ``PipelineSuspended`` when the queue is empty."""
        ...


class PipelineSessionMarker:
    """Marker base for PipelineSession isinstance checks without importing session."""


class QueryLogSource(Protocol):
    """Read-only fetcher of historical SQL statements from an engine query log."""

    def is_available(self, conn: Any) -> bool:
        """Return True when the source can run against *conn*."""
        ...

    def fetch(
        self, conn: Any, *, lookback_days: int, max_queries: int, min_runs: int, user_filter: str | None
    ) -> list[str]:
        """Return distinct SQL texts newest-first within policy caps."""
        ...


class ResultBackendPort(Protocol):
    """Typing port for dialect row-fetch backends. Concrete ABC lives in ``_dialect_sqlglot_helper``."""

    kind: ResultReaderKind

    def fetch_rows(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute *sql* and return result rows as tuples."""
        ...

    def fetch_arrow_table(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout_ms: int | None = None
    ) -> Any:
        """Execute *sql* and return a PyArrow table when the driver supports it."""
        ...

    def cancel_statement(self) -> None:
        """Cancel an in-flight statement when the driver supports it."""
        ...

    def fetch_first_column_text(self, sql: str, params: dict[str, Any] | None = None) -> str:
        """Execute *sql* and join the first column of each row into newline-separated text."""
        ...


class RephraseHint(Enum):
    """User-facing rephrase hint categories printed when the pipeline cannot continue."""

    INTENT_PARSE_FAILED = "intent_parse_failed"
    SCHEMA_INVALID_DECLINED = "schema_invalid_declined"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    JOIN_PATH_UNAVAILABLE = "join_path_unavailable"
    USER_REJECTED_INTENT = "user_rejected_intent"
    USER_REJECTED_RESULT = "user_rejected_result"
    RESTRICTED_QUESTION = "restricted_question"
    CONVERSATIONAL_DENY = "conversational_deny"
    VAGUE_QUESTION = "vague_question"
    FEDERATION_INELIGIBLE = "federation_ineligible"
    FEDERATION_PARTIAL_FAILURE = "federation_partial_failure"
    FEDERATION_TURN_CANCELLED = "federation_turn_cancelled"


@dataclass(frozen=True)
class RefusalCatalogueEntry:
    """User-facing refusal text and reformulation hint for one catalogue code."""

    user_text: str
    reformulation_hint: str


class RefusalCondition(StrEnum):
    """Enumerated refusal conditions mapped to stable diagnostic codes."""

    PERMISSION_DENIAL = "permission_denial"
    SCOPE_VIOLATION = "scope_violation"
    INVALID_QUESTION = "invalid_question"
    PARSE_FAILURE = "parse_failure"
    DECLINED_SCHEMA = "declined_schema"
    TIE_CAP_EXHAUSTION = "tie_cap_exhaustion"
    WIDENED_CLAUSE_REFUSAL = "widened_clause_refusal"
    PROBE_PLACEMENT = "probe_placement"
    UNSUPPORTED_COLUMN_TYPE = "unsupported_column_type"
    NULL_IN_NEGATED_LIST = "null_in_negated_list"
    AMBIGUOUS_DATE_LITERAL = "ambiguous_date_literal"
    UNION_COLUMN_MISSING = "union_column_missing"
    JOIN_PATH_UNAVAILABLE = "join_path_unavailable"
    AGGREGATE_FAN_OUT = "aggregate_fan_out"
    HOP_CEILING = "hop_ceiling"
    CTE_CAP = "cte_cap"
    CAPABILITY_GAP = "capability_gap"
    NOT_AVAILABLE_IN_CONTEXT = "not_available_in_context"
    SUBDAY_DATE_WINDOW = "subday_date_window"


_mock_fixture_recorded_corpus_count: Any = None


@dataclass(frozen=True, slots=True)
class IntentInterpretation:
    """Compact Interpret-stage traceability attached to session steps."""

    approach: str
    grounding: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class IntentSummary:
    """Compact projection of a resolved :class:`RuntimeIntent` for UI and telemetry."""

    tables: tuple[str, ...]
    select_cols: tuple[str, ...]
    filters: tuple[str, ...]
    group_by: tuple[str, ...]
    order_by: tuple[str, ...]
    limit: int | None
    natural_language: str


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    """One bind slot on an accepted template for programmatic callers. handle: the bind token identifying this slot; current_value: the bound value in effect; display_name: a human-readable label for the slot; column_expr: the column expression the slot binds against, when known."""

    handle: str
    current_value: ParamValue | None
    display_name: str
    column_expr: str = ""


@dataclass(frozen=True, slots=True)
class MigrationPreview:
    """Human-readable outcome of comparing a skeleton against the live prepared schema."""

    tier: Literal["compatible", "remap", "destructive"]
    affected_tables: tuple[str, ...]
    affected_columns: tuple[tuple[str, str], ...]
    skeleton_document: dict[str, Any]


class SessionOutcome(StrEnum):
    """Closed set of terminal failure codes carried on ``SessionError.code``."""

    FORBIDDEN = "forbidden"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    UNANSWERABLE = "unanswerable"
    INSUFFICIENT_KNOWLEDGE = "insufficient_knowledge"
    NOT_A_QUESTION = "not_a_question"
    PARSE_FAILED = "parse_failed"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_TIMEOUT = "execution_timeout"
    COST_EXCEEDED = "cost_exceeded"
    LIMIT_EXCEEDED = "limit_exceeded"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    MIGRATION_PENDING = "migration_pending"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class SessionError:
    """Structured terminal failure for a session turn. code: the closed :class:`SessionOutcome` member that ended the turn. detail_code: the one :class:`DiagnosticCode` / :class:`SqlDiagnosticCode` catalogue entry responsible, when a specific one applies. source_id: federation member identity when the failure is attributable to one source; None on a non-federated turn. phase: federation stage (for example ``member`` or ``coordinator``) when the failure occurred during federated execution; None otherwise. limit_key: the configured limit name when the failure was a limit or cost breach; None otherwise."""

    code: SessionOutcome
    detail_code: str | None = None
    source_id: str | None = None
    phase: str | None = None
    limit_key: str | None = None


@dataclass(frozen=True, slots=True)
class SessionStep:
    """Single observable point in a programmatic interactive turn. Carries whether the turn has finished, a short instruction string, a stage discriminant, and exactly one of three terminal shapes: ``answer`` for a metadata question, ``sql`` with ``data`` for an analytical question, or ``error`` for a failure. done: True when the pipeline finished successfully or ended in a terminal error; False when the caller must respond via ``PipelineSession.step``. prompt: The short line the interactive layer should show immediately before collecting input (for example yes or no, or a free-text rejection reason prompt). kind: Stable stage identifier matching the active suspend kind or a terminal sentinel; used to branch programmatic UIs without parsing ``prompt``. reply_shape: When ``done`` is False, whether the caller should collect a yes or no token or free text; None on terminal steps. sql: The formatted SQL under discussion when the step pertains to execution or confirmation; otherwise None. data: The rows available at this step as a ``pandas.DataFrame``, with ``data_truncated`` stating whether more rows exist beyond what is carried; None for scalar or answer-only outcomes. intent_summary: Structured intent headline when the step reflects a parsed intent or later pipeline stages; otherwise None. semantic_warnings: Normalised warning strings for intent confirmation, often empty on non-intent suspend steps. answer: The rendered metadata answer for a terminal step that is not SQL-bearing; None on every other step. diagnostics: Structured diagnostics captured during this step (from ``notify`` / ``debug`` when a collector is active). error: The structured terminal failure when the turn ended without an answer or a result; None otherwise."""

    done: bool
    prompt: str | None
    kind: str
    reply_shape: Literal["yes_no", "free_text"] | None = None
    sql: str | dict[str, str] | None = None
    data: pandas.DataFrame | None = None
    data_truncated: bool = False
    parameters: tuple[ParameterBinding, ...] = ()
    intent_summary: IntentSummary | None = None
    semantic_warnings: tuple[str, ...] = ()
    answer: str | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    error: SessionError | None = None
    template_id: str | None = None
    turn_id: str | None = None
    elapsed_ms: int | None = None
    llm_usage: LlmTurnUsageSummary | None = None


@dataclass(frozen=True, slots=True)
class WriteQueueEvent:
    """Structured event a reader-mode session records for a writer to apply later. kind: Discriminator selecting which writer-side handler applies (template accept or reject, paraphrase emission, override proposal materialisation, or question feedback). schema_graph_id: Stable schema-graph identity stamped at enqueue time; the writer matches and drops events when this value no longer matches the live snapshot. schema_hash: Advisory effective structural hash at event creation for audit and debug only. produced_at: ISO-8601 timestamp string when the reader enqueued the event. payload: Ordered key-value pairs serialising handler-specific fields; a tuple of pairs keeps the event hashable and avoids dict key-order ambiguity across processes."""

    kind: Literal[
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    ]
    schema_graph_id: str
    schema_hash: str
    produced_at: str
    payload: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PhaseProgressEvent:
    """Coarse phase transition during engine construction or an ask turn. elapsed_ms: For ask-turn phases, milliseconds since the previous phase emit in the same turn, or since turn start for the first emit; None for construction phases."""

    phase: str
    timestamp_iso: str
    source: str | None = None
    stage: int | str | None = None
    turn_id: str | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Lifecycle audit record for integrator sinks."""

    event_type: str
    timestamp_iso: str
    question: str | None
    schema_hash: str | None
    provider: Literal["openai", "azure", "sandbox"]
    details: tuple[tuple[str, str], ...] = ()
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class LlmUsageRecord:
    """One LLM response worth of token usage attributed to a build, question, or run scope."""

    scope: Literal["build", "question", "run"]
    block_id: int
    task: str
    logical_model: str
    api_model: str
    provider: Literal["openai", "azure", "sandbox"]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cache_write_tokens: int | None
    attempt: int
    elapsed_ms: int
    phase: str = ""
    source_id: str = ""
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class LlmTurnUsageSummary:
    """Aggregated LLM token usage for one interactive ask turn."""

    request_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """Frozen redacted configuration text for integrators."""

    text: str

    def format_human(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class SchemaStatsSnapshot:
    """Frozen schema statistics mapping."""

    stats: dict[str, Any]

    def format_human(self) -> str:
        lines = [f"{k}: {v}" for k, v in sorted(self.stats.items())]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SeedWarmupSummarySnapshot:
    """Newest seed-warmup summary text if present."""

    text: str

    def format_human(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class QSimSummarySnapshot:
    """QSim summary lines for a version range."""

    lines: tuple[str, ...]

    def format_human(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True, slots=True)
class StoredTemplateSummary:
    """Caller-visible summary row for one accepted template."""

    id: str
    approval_state: str = "approved"


@dataclass(frozen=True, slots=True)
class StoredTemplateDetail:
    """Full caller-visible detail for one accepted template."""

    summary: StoredTemplateSummary
    parameters: tuple[ParameterBinding, ...]
    approval_state: str = "approved"


@dataclass(frozen=True, slots=True)
class TemplateExecutionResult:
    """Result of executing a stored template with caller-supplied bind values."""

    rows: tuple[tuple[Any, ...], ...]
    sql: str
    display_sql: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Engine, artifact root, frozen schema scope, and merged LLM plus execution limits for runtime introspection."""

    engine: str
    artifacts_dir: str
    engine_context: EngineContext | FederationContext
    llm_execution: LlmExecutionConfig
    execution_context: EngineContext | FederationContext | None = None


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Active LLM provider label after environment configuration."""

    provider: Literal["openai", "azure", "sandbox"]


@dataclass
class AetherEngineInitResult:
    """Mutable template bundle and graph produced by engine initialisation."""

    runtime_config: RuntimeConfig
    llm_config: LLMConfig
    schema_graph: Any
    dialect: Any
    artifacts_dir: str
    store: Any
    templates: dict[str, Any]
    rejected: dict[str, Any]
    schema_terms: set[str]
    schema_stats: dict[str, Any]
    schema_role: SchemaRole = SchemaRole.OWNER
    consumer_visible_objects: frozenset[str] | None = None
    context_name: str = "master"
    execution_context: EngineContext | FederationContext | None = None
    data_quality_report: DataQualityReport | None = None
    federation_manifest: FederationManifest | None = None
    federation_mappings: FederationMappings | None = None
    federation_member_graphs: dict[str, Any] | None = None
    federation_storage_dir: str | None = None
    federation_source_runtimes: dict[str, SourceRuntime] | None = None
    federation_mapping_suggestions: tuple[FederationMappingSuggestion, ...] = ()
    federation_dialects_by_source: dict[str, Any] | None = None
    engine_identity: EngineIdentity | None = None


@dataclass
class AetherFederationInitResult(AetherEngineInitResult):
    """Init bundle for :class:`~aetherdialect.AetherFederation`."""

    members: dict[str, Any] | None = None


@dataclass
class SeedWarmupIntent:
    """Unified intent for seed warmup with inline values and expansion metadata."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[SelectCol]
    group_by_cols: list[NormalizedExpr]
    order_by_cols: list[OrderByCol]
    where: PredicateGroup | None
    having: PredicateGroup | None
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
        where = PredicateGroup.parse_where_field(d)
        having = PredicateGroup.parse_having_field(d)
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
            where=where,
            having=having,
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
            "where": self.where.to_dict() if self.where else None,
            "having": self.having.to_dict() if self.having else None,
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
        for fp in PredicateGroup.predicate_leaves(self.where):
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
            where=self.where,
            having=self.having,
            param_values=self.param_values,
            cte_steps=self.cte_steps,
            natural_language=self.natural_language or "",
            limit=self.limit,
            column_map=column_map,
            window_registry=list(self.window_registry),
            case_registry=list(self.case_registry),
            distinct_select_index=self.distinct_select_index,
        )

    @staticmethod
    def _tables_have_duplicates(tables: list[str]) -> bool:
        names = [t for t in tables if t]
        return len(names) != len(set(names))

    def _has_self_join_via_tables(self) -> bool:
        if self._tables_have_duplicates(self.tables or []):
            return True
        for step in self.cte_steps or []:
            if self._tables_have_duplicates(step.tables or []):
                return True
        return False

    def _has_scalar_cte(self) -> bool:
        return any(step.emission == "scalar_subquery" for step in (self.cte_steps or []))

    def _where_date_flags(self) -> tuple[bool, bool]:
        has_dw = False
        has_dd = False
        for fp in PredicateGroup.where_leaves(self.where):
            vt = (fp.value_type or "").strip().lower()
            if vt in WHERE_VALUE_TYPE_DATE_WINDOW:
                has_dw = True
            if vt in WHERE_VALUE_TYPE_DATE_DIFF:
                has_dd = True
        return has_dw, has_dd

    def _json_contains_unnest(self) -> bool:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).lower()
        return "unnest" in blob

    def workload_family(self) -> WorkloadFamily:
        """Map observable structural cues to a canonical workload family."""
        group_n = len(self.group_by_cols or [])
        sel_cols = self.select_cols or []
        has_agg = any(sc.is_aggregated for sc in sel_cols)
        if self.limit is not None and (self.order_by_cols or []):
            return WorkloadFamily.LEADERBOARD
        if group_n > 0 and has_agg:
            return WorkloadFamily.BREAKDOWN
        for fp in PredicateGroup.where_leaves(self.where):
            vt = (fp.value_type or "").strip().lower()
            if vt in WHERE_VALUE_TYPE_DATE_WINDOW:
                return WorkloadFamily.CHANGE_OVER_TIME
            if vt in WHERE_VALUE_TYPE_DATE_DIFF:
                return WorkloadFamily.TREND
        if PredicateGroup.having_leaves(self.having):
            return WorkloadFamily.THRESHOLD_EXCEPTION
        if self.distinct_select_index >= 0:
            return WorkloadFamily.STATUS_REPORT
        return WorkloadFamily.EXTRACT

    def operator_feature_vector(self) -> OperatorFeatureVector:
        """Summarize observable structural operators for diversity metrics."""
        sel_cols = self.select_cols or []
        has_agg = any(sc.is_aggregated for sc in sel_cols)
        has_gb = len(self.group_by_cols or []) > 0
        has_hav = len(PredicateGroup.having_leaves(self.having)) > 0
        win_kind = WindowRegistryStep.operator_kind(list(self.window_registry or []))
        self_join = self._has_self_join_via_tables()
        scalar_cte = self._has_scalar_cte()
        unnest = self._json_contains_unnest()
        case_when = len(self.case_registry or []) > 0
        date_win, date_diff = self._where_date_flags()
        cte_n = len(self.cte_steps or [])
        if cte_n <= 0:
            cte_b = 0
        elif cte_n == 1:
            cte_b = 1
        else:
            cte_b = 2
        tc = len(self.tables or [])
        if tc <= 1:
            jb = 0
        elif tc == 2:
            jb = 1
        else:
            jb = 2
        fam = self.workload_family()
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

    def complexity_tier(self) -> ComplexityTier:
        """Assign a discrete complexity tier from observable structural features."""
        tables_n = len(self.tables or [])
        cte_n = len(self.cte_steps or [])
        win_n = len(self.window_registry or [])
        case_n = len(self.case_registry or [])
        group_n = len(self.group_by_cols or [])
        hav_n = len(PredicateGroup.having_leaves(self.having))
        sel_cols = self.select_cols or []
        has_agg = any(sc.is_aggregated for sc in sel_cols)
        has_ord = len(self.order_by_cols or []) > 0
        lim_set = self.limit is not None

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

    def coverage_atoms(self) -> frozenset[str]:
        """Unary and pairwise tags for submodular warmup coverage scoring."""
        v = self.operator_feature_vector()
        tier = self.complexity_tier().value
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
        feat_tags = {f"feat:{feat}" for feat in sorted(self.to_runtime_intent().detect_features())}
        atoms: set[str] = set(cell) | feat_tags
        ordered = sorted(cell | feat_tags)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                atoms.add(f"pair:{ordered[i]}|{ordered[j]}")
        return frozenset(atoms)

    def anchor_lattice_key(self) -> AnchorLatticeKey:
        """Build a stable lattice cell key for NL reuse across synthetic warmup rows."""
        tier = self.complexity_tier()
        fam = self.workload_family()
        tid = self.seed_index if self.seed_index is not None else 0
        em = self.expansion_metadata
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
            ).encode()
        ).hexdigest()
        slot = int(digest[0:2], 16) % len(SeedWarmupConfig.WARMUP_QUESTION_STYLES)
        style_s = SeedWarmupConfig.WARMUP_QUESTION_STYLES[slot]
        ws = WarmupStyle(style_s)
        return AnchorLatticeKey(family=fam, tier=tier, style=ws, novelty_band=nov)


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
