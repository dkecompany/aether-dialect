"""Schema graph metadata: columns, tables, expansion specs, and SQL shape flags."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import InitVar, asdict, dataclass, field, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from ._constants import (
    AGG_PATTERN,
    CASE_WHEN_QUALIFIED_COLUMN_REF_RE,
    DEFAULT_RANDOM_SEED,
    DESCRIPTION_OWNER_VALUES,
    EXACT_NUMERIC_BASE_TYPES,
    FEDERATION_ENUM_PROMPT_CAP,
    HIDDEN_SENSITIVITIES,
    INEXACT_NUMERIC_BASE_TYPES,
    INFERENCE_TAG_VALUES,
    KEPT_ISSUE_SEVERITIES,
    LOGICAL_FAILURE_CATEGORIES,
    NUMERIC_TYPE_ARGUMENTS_RE,
    PK_INFERENCE_TAG_VALUES,
    PROMPT_SCALAR_VALUE_TYPES,
    ROLE_ALLOWED_AGGREGATIONS,
    ROLE_OWNER_VALUES,
    SENTINEL_MODE_FREQUENCY_THRESHOLD,
    UNKNOWN_VALUE_TYPE,
    UNUSABLE_NULL_RATIO_THRESHOLD,
    WINDOW_REGISTRY_AGG_KIND_HINTS,
    WINDOW_REGISTRY_NAV_KIND_HINTS,
    WINDOW_REGISTRY_RANK_KIND_HINTS,
)
from ._constants_runtime import (
    COMPOSE_FIELDS,
    CSV_SCHEMA_LITERAL_ORIGINAL_NAME_NOTE,
    FULL_FIELDS,
    GROUND_FIELDS,
    INTERPRET_FIELDS,
    SCHEMA_FIELD_DERIVED,
    SCHEMA_FIELD_DESCRIPTION,
    SCHEMA_FIELD_ENUM,
    SCHEMA_FIELD_KEYS,
    SCHEMA_FIELD_ROLE,
    SCHEMA_FIELD_SAMPLES,
    SCHEMA_FIELD_TRUTH_VALUE,
    SCHEMA_FIELD_TYPE,
    SCHEMA_INSTRUCTION_LIKE_LINE_PATTERNS,
    SCHEMA_INSTRUCTION_SCRUB_REPLACEMENT,
    STAGE_ATTRIBUTION_TABLE,
)
from ._contracts_base import (
    ArrayStorageKind,
    ColumnTypeSemantics,
    ColumnVisibilityBlockReason,
    ComplexityTier,
    ConfigError,
    DatabaseFeatureCapability,
    DataQualityReport,
    FailureCategory,
    NormalizedExpr,
    OrderByCol,
    OverrideSkip,
    ParamValue,
    SchemaInclude,
    SchemaInvariantError,
    SchemaRole,
    SensitivityClassification,
    StructuralKnowledgeFact,
    TableKind,
    WhereParam,
    WindowFrameKind,
    WindowOperatorKind,
    WorkloadFamily,
)


@dataclass
class _ColumnMetadataCore:
    """Consolidated column metadata with profile, role, and value domain. Holds counts, overrides, filter/aggregation rules, and boolean hints used by validation and generation."""

    name: str
    data_type: str
    original_name: str = ""
    enum_type_name: str | None = None
    is_primary_key: InitVar[bool] = False
    is_foreign_key: InitVar[bool] = False
    fk_target: InitVar[tuple[str, str] | None] = None
    role: str | None = None
    value_type: str = ""
    row_count: int = 0
    distinct_count: int = 0
    distinct_from_sample: bool = False
    distinct_ratio: float | None = 0.0
    null_ratio: float | None = 0.0
    min_val: str | None = None
    max_val: str | None = None
    frequent_values: list[str] = field(default_factory=list)
    value_overlap_sample: list[str] = field(default_factory=list)
    character_set: str | None = None
    collation: str | None = None
    is_case_insensitive_collation: bool = False
    overlap_comparison: str = "exact"
    is_aggregatable_override: bool | None = None
    is_groupable_override: bool | None = None
    is_filterable_override: bool | None = None
    valid_where_ops: list[str] = field(default_factory=list)
    valid_aggregations: list[str] = field(default_factory=list)
    valid_having_ops: list[str] = field(default_factory=list)
    description: str = ""
    description_owner: DescriptionOwner | None = None
    base_description: str = ""
    is_unique: bool = False
    is_generated: bool = False
    is_identity: bool = False
    sensitivity: SensitivityClassification = SensitivityClassification.NONE
    element_type: str | None = None
    is_nullable: bool = True
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)
    is_denied: InitVar[bool] = False
    mode_frequency_ratio: float = 0.0
    profile_failed: bool = False
    profile_skipped_reason: str | None = None
    is_canonical_duplicate: InitVar[bool] = True
    pk_inference_tag: PkInferenceTag | None = None
    role_owner: RoleOwner | None = None
    boolean_truth_value: str | None = None
    usable_override: bool | None = None
    numeric_precision: int | None = None
    numeric_scale: int | None = None
    is_exact_numeric: bool = False
    is_unsigned: bool = False
    is_timezone_aware: bool = False
    is_fixed_width_text: bool = False

    def prompt_value_type(self) -> str:
        """Return the normalized value-type token used for prompt sample guards."""
        vt = (self.value_type or "").strip().lower()
        if vt:
            return vt
        if self.data_type:
            return ColumnTypeSemantics.data_type_to_value_type(self.data_type).strip().lower()
        return ArrayStorageKind.UNKNOWN

    def assert_prompt_value_scalar(
        self,
        *,
        table_name: str | None,
        contribution: str,
    ) -> None:
        """Raise when non-scalar column data would be serialized into a prompt."""
        vt = self.prompt_value_type()
        if vt in PROMPT_SCALAR_VALUE_TYPES:
            return
        loc = f"{table_name}.{self.name}" if table_name else self.name
        raise SchemaInvariantError(
            f"column {loc!r} has value_type {vt!r} but contributed {contribution} to a prompt; "
            f"only {sorted(PROMPT_SCALAR_VALUE_TYPES)} may reach prompts"
        )

    def __post_init__(
        self,
        is_primary_key: bool,
        is_foreign_key: bool,
        fk_target: tuple[str, str] | None,
        is_denied: bool,
        is_canonical_duplicate: bool,
    ) -> None:
        """Set `value_type` from `data_type` when `value_type` is empty and capture PK/FK/deny seeds. ``is_primary_key``, ``is_foreign_key``, ``fk_target``, and ``is_denied`` are accepted as constructor arguments for ergonomic standalone construction (tests, ad-hoc fixtures), but they are merely seeds: the authoritative store of primary-key membership is ``TableMetadata.primary_key``, of foreign-key membership is ``TableMetadata.foreign_keys``, and of deny-list membership is ``SchemaGraph.deny_columns`` on the owning graph. When this column is attached to a :class:`TableMetadata` (and that table to a :class:`SchemaGraph`), the relevant ``__post_init__`` consolidates the seeds into the canonical containers and clears them so the :attr:`is_primary_key`, :attr:`is_foreign_key`, :attr:`fk_target`, and :attr:`is_denied` properties always read from a single owner. The seed and back- reference attributes are stored outside the dataclass field set so :func:`dataclasses.asdict` does not traverse a column-table- graph cycle."""
        if not self.value_type and self.data_type:
            self.value_type = ColumnTypeSemantics.data_type_to_value_type(self.data_type)
        if (
            self.data_type
            and self.numeric_precision is None
            and self.numeric_scale is None
            and not self.is_exact_numeric
        ):
            match = NUMERIC_TYPE_ARGUMENTS_RE.search(self.data_type)
            if match:
                self.numeric_precision = int(match.group(1))
                self.numeric_scale = int(match.group(2)) if match.group(2) is not None else None
            base = ColumnTypeSemantics.normalize_column_type(self.data_type)
            if base in INEXACT_NUMERIC_BASE_TYPES:
                self.is_exact_numeric = False
            elif base in EXACT_NUMERIC_BASE_TYPES:
                self.is_exact_numeric = True
            elif self.value_type == "integer":
                self.is_exact_numeric = True
            elif self.value_type == "number":
                self.is_exact_numeric = base in {"decimal", "numeric", "money"}
        seed_unsigned = self.is_unsigned
        seed_timezone_aware = self.is_timezone_aware
        seed_fixed_width_text = self.is_fixed_width_text
        if self.data_type:
            detected_unsigned = ColumnTypeSemantics.column_is_unsigned_from_data_type(self.data_type)
            object.__setattr__(self, "is_unsigned", seed_unsigned or detected_unsigned)
            detected_timezone_aware = ColumnTypeSemantics.column_timezone_aware_from_data_type(self.data_type)
            object.__setattr__(self, "is_timezone_aware", seed_timezone_aware or detected_timezone_aware)
            detected_fixed_width_text = ColumnTypeSemantics.column_is_fixed_width_text_from_data_type(self.data_type)
            object.__setattr__(
                self,
                "is_fixed_width_text",
                seed_fixed_width_text or detected_fixed_width_text,
            )
        if not self.is_exact_numeric and self.is_unsigned and ColumnTypeSemantics.column_unsigned_near_type_max(self):
            object.__setattr__(self, "is_exact_numeric", True)
        object.__setattr__(self, "_seed_is_primary_key", bool(is_primary_key))
        object.__setattr__(self, "_seed_is_foreign_key", bool(is_foreign_key))
        object.__setattr__(
            self,
            "_seed_fk_target",
            (tuple(fk_target) if isinstance(fk_target, (list, tuple)) and len(fk_target) == 2 else None),
        )
        object.__setattr__(self, "_seed_is_denied", bool(is_denied))
        object.__setattr__(self, "_seed_is_canonical_duplicate", bool(is_canonical_duplicate))
        object.__setattr__(self, "_owner_table", None)
        SensitivityClassification.apply_to(self, self.sensitivity)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ColumnMetadata:
        """
        Create `ColumnMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `ColumnMetadata` fields.

        Returns:

            Populated `ColumnMetadata` instance.
        """
        fk_target = None
        if d.get("fk_target"):
            fk_target = tuple(d["fk_target"]) if isinstance(d["fk_target"], list) else d["fk_target"]
        sens = SensitivityClassification.from_dict(d)
        frequent_values = list(d.get("frequent_values") or [])
        value_overlap_sample = list(d.get("value_overlap_sample") or [])
        return ColumnMetadata(
            name=d.get("name", ""),
            data_type=d.get("data_type", ""),
            original_name=str(d.get("original_name", "") or ""),
            is_primary_key=d.get("is_primary_key", False),
            is_foreign_key=d.get("is_foreign_key", False),
            fk_target=fk_target,
            role=d.get("role"),
            value_type=d.get("value_type", ""),
            enum_type_name=d.get("enum_type_name"),
            row_count=d.get("row_count", 0),
            distinct_count=d.get("distinct_count", 0),
            distinct_from_sample=d.get("distinct_from_sample", False),
            distinct_ratio=d.get("distinct_ratio", 0.0),
            null_ratio=d.get("null_ratio", 0.0),
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
            frequent_values=frequent_values,
            is_aggregatable_override=d.get("is_aggregatable_override"),
            is_groupable_override=d.get("is_groupable_override"),
            is_filterable_override=d.get("is_filterable_override"),
            valid_where_ops=d.get("valid_where_ops", []),
            valid_aggregations=d.get("valid_aggregations", []),
            valid_having_ops=d.get("valid_having_ops", []),
            description=d.get("description", ""),
            description_owner=DescriptionOwner.coerce(d.get("description_owner")),
            base_description=str(d.get("base_description", "") or ""),
            is_unique=d.get("is_unique", False),
            is_generated=bool(d.get("is_generated", False)),
            is_identity=bool(d.get("is_identity", False)),
            sensitivity=sens,
            element_type=d.get("element_type"),
            is_nullable=d.get("is_nullable", True),
            value_overlap_sample=value_overlap_sample,
            character_set=d.get("character_set"),
            collation=d.get("collation"),
            is_case_insensitive_collation=bool(d.get("is_case_insensitive_collation", False)),
            overlap_comparison=str(d.get("overlap_comparison", "exact") or "exact"),
            semantic_join_neighbors=[
                (str(x[0]), str(x[1]))
                for x in (d.get("semantic_join_neighbors") or [])
                if isinstance(x, (list, tuple)) and len(x) == 2
            ],
            is_denied=d.get("is_denied", False),
            mode_frequency_ratio=d.get("mode_frequency_ratio", 0.0),
            profile_failed=bool(d.get("profile_failed", False)),
            profile_skipped_reason=d.get("profile_skipped_reason"),
            is_canonical_duplicate=d.get("is_canonical_duplicate", True),
            pk_inference_tag=PkInferenceTag.coerce(d.get("pk_inference_tag")),
            role_owner=RoleOwner.coerce(d.get("role_owner")),
            boolean_truth_value=d.get("boolean_truth_value"),
            usable_override=d.get("usable_override"),
            numeric_precision=d.get("numeric_precision"),
            numeric_scale=d.get("numeric_scale"),
            is_exact_numeric=bool(d.get("is_exact_numeric", False)),
            is_unsigned=bool(d.get("is_unsigned", False)),
            is_timezone_aware=bool(d.get("is_timezone_aware", False)),
            is_fixed_width_text=bool(d.get("is_fixed_width_text", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `ColumnMetadata` fields as primitives.
        """
        col = cast("ColumnMetadata", self)
        return {
            "name": self.name,
            "original_name": self.original_name,
            "data_type": self.data_type,
            "is_primary_key": col.is_primary_key,
            "is_foreign_key": col.is_foreign_key,
            "fk_target": list(col.fk_target) if col.fk_target else None,
            "role": self.role,
            "value_type": self.value_type,
            "enum_type_name": self.enum_type_name,
            "row_count": self.row_count,
            "distinct_count": self.distinct_count,
            "distinct_from_sample": self.distinct_from_sample,
            "distinct_ratio": self.distinct_ratio,
            "null_ratio": self.null_ratio,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "frequent_values": self.frequent_values,
            "is_aggregatable_override": self.is_aggregatable_override,
            "is_groupable_override": self.is_groupable_override,
            "is_filterable_override": self.is_filterable_override,
            "valid_where_ops": self.valid_where_ops,
            "valid_aggregations": self.valid_aggregations,
            "valid_having_ops": self.valid_having_ops,
            "description": self.description,
            "description_owner": (self.description_owner.value if self.description_owner is not None else None),
            "base_description": self.base_description,
            "is_unique": self.is_unique,
            "is_generated": self.is_generated,
            "is_identity": self.is_identity,
            "sensitivity": self.sensitivity.value,
            "is_selectable": col.is_selectable,
            "element_type": self.element_type,
            "is_nullable": self.is_nullable,
            "value_overlap_sample": self.value_overlap_sample,
            "character_set": self.character_set,
            "collation": self.collation,
            "is_case_insensitive_collation": self.is_case_insensitive_collation,
            "overlap_comparison": self.overlap_comparison,
            "semantic_join_neighbors": [list(p) for p in self.semantic_join_neighbors],
            "is_denied": col.is_denied,
            "mode_frequency_ratio": self.mode_frequency_ratio,
            "profile_failed": self.profile_failed,
            "profile_skipped_reason": self.profile_skipped_reason,
            "is_canonical_duplicate": col.is_canonical_duplicate,
            "pk_inference_tag": (self.pk_inference_tag.value if self.pk_inference_tag is not None else None),
            "role_owner": (self.role_owner.value if self.role_owner is not None else None),
            "boolean_truth_value": self.boolean_truth_value,
            "numeric_precision": self.numeric_precision,
            "numeric_scale": self.numeric_scale,
            "is_exact_numeric": self.is_exact_numeric,
            "is_unsigned": self.is_unsigned,
            "is_timezone_aware": self.is_timezone_aware,
            "is_fixed_width_text": self.is_fixed_width_text,
        }

    def visibility_block_reason(self) -> ColumnVisibilityBlockReason | None:
        """Return why this column is not LLM-visible, or ``None`` when it is visible."""
        col = cast("ColumnMetadata", self)
        owner = col._owner_table
        graph = getattr(owner, "_owner_graph", None) if owner is not None else None
        if owner is not None and graph is not None:
            deny_set = graph.deny_columns.get(owner.name)
            if deny_set and self.name in deny_set:
                return ColumnVisibilityBlockReason.DENIED
            disallowed = graph.disallowed_columns.get(owner.name)
            if disallowed and self.name in disallowed:
                return ColumnVisibilityBlockReason.NOT_IN_ALLOW_COLUMNS
        elif col._seed_is_denied:
            return ColumnVisibilityBlockReason.DENIED
        if self.sensitivity == SensitivityClassification.HIDDEN:
            return ColumnVisibilityBlockReason.SENSITIVE_HIDDEN
        if self.sensitivity == SensitivityClassification.RESTRICTED:
            if not col.is_usable:
                return ColumnVisibilityBlockReason.UNUSABLE
            return None
        if not col.is_usable:
            return ColumnVisibilityBlockReason.UNUSABLE
        return None

    def get_valid_where_ops(self) -> list[str]:
        """Valid filter operators for this column, always including. null. checks. Returns: Operator strings such as `=`, `!=`, `like`, `between`, plus `is null` / `is not null`."""
        null_ops = ["is null", "is not null"]
        if self.valid_where_ops:
            return sorted(set(self.valid_where_ops + null_ops))
        return null_ops

    def get_valid_aggregations(self) -> set[str]:
        """
        Valid aggregation function names for this column.

        Returns:

            Lowercased names from `valid_aggregations`, or an empty set if none are stored.
        """
        if self.valid_aggregations:
            return set(agg.lower() for agg in self.valid_aggregations)
        if self.role:
            rk = self.role.upper()
            if rk in ROLE_ALLOWED_AGGREGATIONS:
                return {a.lower() for a in ROLE_ALLOWED_AGGREGATIONS[rk]}
        return set()

    def get_valid_having_ops(self) -> list[str]:
        """
        Valid `HAVING` operators for this column.

        Returns:

            A copy of `valid_having_ops` if set, otherwise an empty list.
        """
        if self.valid_having_ops:
            return list(self.valid_having_ops)
        return []


class ColumnMetadata(_ColumnMetadataCore):
    _seed_is_primary_key: bool
    _seed_is_foreign_key: bool
    _seed_fk_target: tuple[str, str] | None
    _seed_is_denied: bool
    _seed_is_canonical_duplicate: bool
    _owner_table: TableMetadata | None

    @property
    def is_selectable(self) -> bool:
        """Whether the column may be projected in a ``SELECT`` list."""
        if self.sensitivity in (SensitivityClassification.RESTRICTED, SensitivityClassification.HIDDEN):
            return False
        return True

    @property
    def is_usable(self) -> bool:
        """Whether the column has enough variance and signal to be exposed to the LLM."""
        if (self.value_type or "").strip().lower() == UNKNOWN_VALUE_TYPE:
            return False
        if self.is_primary_key or self.is_foreign_key:
            return True
        if self.usable_override is True:
            return True
        if self.distinct_count is not None and self.distinct_count <= 1:
            return False
        if self.null_ratio is not None and self.null_ratio >= UNUSABLE_NULL_RATIO_THRESHOLD:
            return False
        if self.mode_frequency_ratio >= SENTINEL_MODE_FREQUENCY_THRESHOLD:
            return False
        return True

    @property
    def is_visible(self) -> bool:
        """Whether this column should appear in LLM-facing schema context."""
        if self.is_denied:
            return False
        if self.sensitivity == SensitivityClassification.HIDDEN:
            return False
        return self.is_usable

    @property
    def is_filterable(self) -> bool:
        """
        Whether the column may appear in `WHERE` predicates.

        Returns:

            False when denied, classified restricted/hidden, or value type is unknown;
            else override, key, or role-based rules.
        """
        if self.is_denied:
            return False
        if self.sensitivity in (SensitivityClassification.RESTRICTED, SensitivityClassification.HIDDEN):
            return False
        if (self.value_type or "").strip().lower() == UNKNOWN_VALUE_TYPE:
            return False
        if self.is_filterable_override is not None:
            return self.is_filterable_override
        if self.is_primary_key:
            return True
        if self.is_foreign_key:
            return True
        if self.role in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.NUMERIC_MEASURE.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.FREE_TEXT.value,
            ColumnRole.AUDIT.value,
        ):
            return True
        return False

    @property
    def is_groupable(self) -> bool:
        """
        Whether the column may appear in `GROUP BY`.

        Returns:

            True when override, foreign key, or role allows grouping.
        """
        if (self.value_type or "").strip().lower() == UNKNOWN_VALUE_TYPE:
            return False
        if self.is_groupable_override is not None:
            return self.is_groupable_override
        if self.is_foreign_key:
            return True
        return self.role in (
            ColumnRole.CATEGORICAL.value,
            ColumnRole.NUMERIC_CATEGORICAL.value,
            ColumnRole.BOOLEAN.value,
            ColumnRole.TEMPORAL.value,
            ColumnRole.IDENTIFIER.value,
        )

    @property
    def is_aggregatable(self) -> bool:
        """
        Whether measures like `SUM` / `AVG` apply to this column.

        Returns:

            True when override is set, or role is numeric measure.
        """
        if (self.value_type or "").strip().lower() == UNKNOWN_VALUE_TYPE:
            return False
        if self.is_aggregatable_override is not None:
            return self.is_aggregatable_override
        return self.role == ColumnRole.NUMERIC_MEASURE.value

    @property
    def is_foreign_key(self) -> bool:
        """Whether this column participates as the source of any foreign-key edge on its owning table. Derived strictly from ``TableMetadata.foreign_keys`` once an owner is wired; before wiring, falls back to the constructor seed value (used by standalone fixtures with no parent table)."""
        owner = self._owner_table
        if owner is None:
            return self._seed_is_foreign_key
        for fk in owner.foreign_keys:
            if self.name in fk.src_cols:
                return True
        return False

    @property
    def fk_target(self) -> tuple[str, str] | None:
        """Destination ``(table, column)`` of the first foreign-key edge whose source includes this column. Looked up from ``TableMetadata.foreign_keys`` when an owner is wired; before wiring, returns the constructor seed."""
        owner = self._owner_table
        if owner is None:
            return self._seed_fk_target
        for fk in owner.foreign_keys:
            for sc, dc in zip(fk.src_cols, fk.dst_cols, strict=False):
                if sc == self.name:
                    return (fk.dst_table, dc)
        return None

    @property
    def is_primary_key(self) -> bool:
        """Whether this column appears in its owning table's primary-key list. Derived strictly from ``TableMetadata.primary_key`` once an owner is wired; before wiring, falls back to the constructor seed value (used by standalone fixtures with no parent table)."""
        owner = self._owner_table
        if owner is None:
            return self._seed_is_primary_key
        return self.name in owner.primary_key

    @property
    def is_denied(self) -> bool:
        """Whether this column is denied by scope policy on its owning :class:`SchemaGraph`. True when the column appears in ``SchemaGraph.deny_columns`` or ``SchemaGraph.disallowed_columns`` for its owning table (once wired), or when the standalone fixture seed marks it denied."""
        owner = self._owner_table
        if owner is None:
            return self._seed_is_denied
        graph = getattr(owner, "_owner_graph", None)
        if graph is None:
            return self._seed_is_denied
        deny_set = graph.deny_columns.get(owner.name)
        if deny_set and self.name in deny_set:
            return True
        disallowed = graph.disallowed_columns.get(owner.name)
        return bool(disallowed and self.name in disallowed)

    @property
    def is_canonical_duplicate(self) -> bool:
        """Whether this column is the canonical bearer for its name across the schema graph. A column whose name is unique across all tables is trivially canonical. When the same name appears in two or more tables, exactly one bearer is selected by ``recompute_canonical_bearers`` on the owning :class:`SchemaGraph` and recorded in ``SchemaGraph._canonical_bearers``; that bearer reads ``True`` and the others read ``False``. Before owner-graph wiring, falls back to the constructor seed value."""
        owner = self._owner_table
        if owner is None:
            return self._seed_is_canonical_duplicate
        graph = getattr(owner, "_owner_graph", None)
        if graph is None:
            return self._seed_is_canonical_duplicate
        bearers = getattr(graph, "_canonical_bearers", None)
        if not bearers:
            return True
        bearer = bearers.get(self.name.lower())
        if bearer is None:
            return True
        return bool(bearer == (owner.name, self.name))


@dataclass
class TableMetadata:
    """Table metadata with nested columns, foreign keys, partition columns, and role."""

    name: str
    columns: dict[str, ColumnMetadata]
    primary_key: list[str]
    foreign_keys: list[FKEdge]
    original_name: str = ""
    source_id: str = ""
    member_source_ids: list[str] = field(default_factory=list)
    column_member_sources: dict[str, list[str]] = field(default_factory=dict)
    kind: TableKind = TableKind.TABLE
    last_refreshed_at: str | None = None
    partition_columns: list[str] = field(default_factory=list)
    partition_type: str | None = None
    require_partition_filter: bool = False
    clustering_fields: list[str] = field(default_factory=list)
    clustering_key: str | None = None
    distkey: str | None = None
    sortkey: list[str] = field(default_factory=list)
    diststyle: str | None = None
    indexed_columns: list[str] = field(default_factory=list)
    size_mb: float | None = None
    encoded: bool | None = None
    quote_decision: str | None = None
    view_definition: str = ""
    role: str | None = None
    row_count: int = 0
    profile_failed: bool = False
    description: str = ""
    description_owner: DescriptionOwner | None = None
    base_description: str = ""
    role_owner: RoleOwner | None = None
    composite_descriptive_ratios: dict[tuple[str, str], float] = field(default_factory=dict)
    _user_semantic_neighbors: list[tuple[str, str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Wire each child :class:`ColumnMetadata` back to this table and consolidate any PK / FK seeds. Tests and ad-hoc fixtures may pass ``is_primary_key``, ``is_foreign_key`` / ``fk_target`` as constructor arguments to :class:`ColumnMetadata` without separately populating :attr:`primary_key` or appending an :class:`FKEdge` to :attr:`foreign_keys`. After wiring the per-column ``_owner_table`` back-reference, this method (a) appends any PK seed to :attr:`primary_key` when not already present, and (b) synthesises a single-column :class:`FKEdge` for every column whose seed declares an FK that is not already covered by an entry in :attr:`foreign_keys`. The seeds are then cleared so the :class:`ColumnMetadata` properties always read from this table's :attr:`primary_key` and :attr:`foreign_keys` as the single source of truth."""
        covered_fk: set[str] = set()
        for fk in self.foreign_keys:
            for sc in fk.src_cols:
                covered_fk.add(sc)
        pk_raw: Any = self.primary_key
        if isinstance(pk_raw, str):
            self.primary_key = [pk_raw] if pk_raw else []
        existing_pk = set(self.primary_key)
        seeded_denies: set[str] = set()
        for cname, col in self.columns.items():
            col._owner_table = self
            if col._seed_is_primary_key and cname not in existing_pk:
                self.primary_key.append(cname)
                existing_pk.add(cname)
            if col._seed_is_foreign_key and col._seed_fk_target is not None and cname not in covered_fk:
                dst_t, dst_c = col._seed_fk_target
                self.foreign_keys.append(
                    FKEdge(src_table=self.name, src_cols=[cname], dst_table=dst_t, dst_cols=[dst_c], inference_tag=None)
                )
                covered_fk.add(cname)
            if col._seed_is_denied:
                seeded_denies.add(cname)
            col._seed_is_primary_key = False
            col._seed_is_foreign_key = False
            col._seed_fk_target = None
        object.__setattr__(self, "_owner_graph", None)
        object.__setattr__(self, "_pending_denies", seeded_denies)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TableMetadata:
        """
        Create `TableMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `TableMetadata` fields.

        Returns:

            Populated `TableMetadata` with nested `ColumnMetadata` and `FKEdge` objects.
        """
        cols_raw = d.get("columns", {})
        columns = {k: ColumnMetadata.from_dict(v) for k, v in cols_raw.items()} if isinstance(cols_raw, dict) else {}
        fk_raw = d.get("foreign_keys", [])
        foreign_keys = [FKEdge(**fk) if isinstance(fk, dict) else fk for fk in fk_raw]
        kind_raw = d.get("kind", "table")
        if kind_raw == "materialized_view":
            kind: TableKind = TableKind.MATERIALIZED_VIEW
        elif kind_raw == "view":
            kind = TableKind.VIEW
        else:
            kind = TableKind.TABLE
        last_refreshed_raw = d.get("last_refreshed_at")
        last_refreshed_at = str(last_refreshed_raw).strip() if last_refreshed_raw else None
        return TableMetadata(
            name=d.get("name", ""),
            original_name=str(d.get("original_name", "") or ""),
            source_id=str(d.get("source_id", "") or ""),
            member_source_ids=list(d.get("member_source_ids", []) or []),
            column_member_sources={str(k): list(v) for k, v in (d.get("column_member_sources", {}) or {}).items()},
            columns=columns,
            primary_key=d.get("primary_key", []),
            foreign_keys=foreign_keys,
            kind=kind,
            last_refreshed_at=last_refreshed_at,
            partition_columns=d.get("partition_columns", []),
            partition_type=d.get("partition_type"),
            require_partition_filter=bool(d.get("require_partition_filter", False)),
            clustering_fields=list(d.get("clustering_fields", []) or []),
            clustering_key=d.get("clustering_key"),
            distkey=d.get("distkey"),
            sortkey=list(d.get("sortkey", []) or []),
            diststyle=d.get("diststyle"),
            indexed_columns=list(d.get("indexed_columns", []) or []),
            size_mb=d.get("size_mb"),
            encoded=d.get("encoded"),
            quote_decision=d.get("quote_decision"),
            view_definition=str(d.get("view_definition", "") or ""),
            role=d.get("role"),
            row_count=d.get("row_count", 0),
            profile_failed=bool(d.get("profile_failed", False)),
            description=d.get("description", ""),
            description_owner=DescriptionOwner.coerce(d.get("description_owner")),
            base_description=str(d.get("base_description", "") or ""),
            role_owner=RoleOwner.coerce(d.get("role_owner")),
            composite_descriptive_ratios={
                tuple(k.split("|", 1)): v for k, v in d.get("composite_descriptive_ratios", {}).items() if "|" in k
            },
            _user_semantic_neighbors=[
                tuple(item)
                for item in (d.get("_user_semantic_neighbors", []) or [])
                if isinstance(item, (list, tuple)) and len(item) == 4
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `TableMetadata` fields; nested columns and foreign keys are serialized recursively.
        """
        return {
            "name": self.name,
            "original_name": self.original_name,
            "source_id": self.source_id,
            "member_source_ids": list(self.member_source_ids),
            "column_member_sources": {k: list(v) for k, v in self.column_member_sources.items()},
            "kind": self.kind,
            "last_refreshed_at": self.last_refreshed_at,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "primary_key": self.primary_key,
            "foreign_keys": [asdict(fk) for fk in self.foreign_keys],
            "partition_columns": self.partition_columns,
            "partition_type": self.partition_type,
            "require_partition_filter": self.require_partition_filter,
            "clustering_fields": self.clustering_fields,
            "clustering_key": self.clustering_key,
            "distkey": self.distkey,
            "sortkey": self.sortkey,
            "diststyle": self.diststyle,
            "indexed_columns": self.indexed_columns,
            "size_mb": self.size_mb,
            "encoded": self.encoded,
            "quote_decision": self.quote_decision,
            "view_definition": self.view_definition,
            "role": self.role,
            "row_count": self.row_count,
            "profile_failed": self.profile_failed,
            "description": self.description,
            "description_owner": (self.description_owner.value if self.description_owner is not None else None),
            "base_description": self.base_description,
            "role_owner": (self.role_owner.value if self.role_owner is not None else None),
            "composite_descriptive_ratios": {
                f"{c1}|{c2}": ratio for (c1, c2), ratio in self.composite_descriptive_ratios.items()
            },
            "_user_semantic_neighbors": [list(t) for t in self._user_semantic_neighbors],
        }

    @property
    def column_names(self) -> list[str]:
        """
        Ordered column names for this table.

        Returns:

            Keys of `columns` as a list.
        """
        return list(self.columns.keys())


_SchemaGraphStatsFn = Callable[["SchemaGraph"], dict[str, Any]]
_SchemaGraphCapabilityFn = Callable[["SchemaGraph"], DatabaseFeatureCapability]


@dataclass
class SchemaGraph:
    """Schema graph with nested tables, join paths, and metadata."""

    tables: dict[str, TableMetadata]
    join_paths_multi: dict[str, dict[str, list[list[dict[str, Any]]]]]
    structural_hash: str = ""
    profiling_hash: str = ""
    descriptions_hash: str = ""
    scope_hash: str = ""
    effective_structural_hash: str = ""
    schema_graph_id: str = ""
    notes_hash: str = ""
    semantic_edges_hash: str = ""
    ddl_probe_hash: str = ""
    include: SchemaInclude = SchemaInclude.TABLES
    created_at: str = ""
    enum_values: dict[str, list[str]] | None = None
    schema_stats: dict[str, Any] | None = None
    deny_columns: dict[str, set[str]] = field(default_factory=dict)
    disallowed_columns: dict[str, set[str]] = field(default_factory=dict)
    notes_sha256: str = ""
    structural_knowledge: tuple[StructuralKnowledgeFact, ...] = ()
    scope_descriptor: dict[str, Any] | None = None
    federation_membership: dict[str, str] | None = None
    schema_revision: int = 0
    _database_feature_capability_cache: DatabaseFeatureCapability | None = field(
        default=None, repr=False, compare=False
    )
    _stats_dirty: bool = field(default=True, repr=False, compare=False)
    _last_structure_skipped: tuple[OverrideSkip, ...] = field(default=(), repr=False, compare=False)
    _override_internal_blocks: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _stats_fn: _SchemaGraphStatsFn | None = field(default=None, repr=False, compare=False)
    _capability_fn: _SchemaGraphCapabilityFn | None = field(default=None, repr=False, compare=False)
    _default_stats_fn: ClassVar[_SchemaGraphStatsFn | None] = None
    _default_capability_fn: ClassVar[_SchemaGraphCapabilityFn | None] = None

    def __post_init__(self) -> None:
        """Wire owner-graph back-references and consolidate per-column deny seeds into ``deny_columns``. After this runs, ``deny_columns`` is the single source of truth for ``ColumnMetadata.is_denied``. Existing ``deny_columns`` entries are preserved; per-column seeds (set on standalone-built columns) and per-table pending-deny sets (collected by :meth:`TableMetadata.__post_init__`) are folded in."""
        if self._stats_fn is None and type(self)._default_stats_fn is not None:
            self._stats_fn = type(self)._default_stats_fn
        if self._capability_fn is None and type(self)._default_capability_fn is not None:
            self._capability_fn = type(self)._default_capability_fn
        deny_columns: dict[str, set[str]] = {k: set(v) for k, v in (self.deny_columns or {}).items()}
        disallowed_columns: dict[str, set[str]] = {k: set(v) for k, v in (self.disallowed_columns or {}).items()}
        for tbl_name, tbl in self.tables.items():
            object.__setattr__(tbl, "_owner_graph", self)
            pending: set[str] = getattr(tbl, "_pending_denies", set())
            if pending:
                deny_columns.setdefault(tbl_name, set()).update(pending)
                object.__setattr__(tbl, "_pending_denies", set())
            for col_name, col in tbl.columns.items():
                if getattr(col, "_seed_is_denied", False):
                    deny_columns.setdefault(tbl_name, set()).add(col_name)
                    col._seed_is_denied = False
        self.deny_columns = deny_columns
        self.disallowed_columns = disallowed_columns
        if not hasattr(self, "_canonical_bearers"):
            object.__setattr__(self, "_canonical_bearers", {})

    def refresh_schema_stats(self) -> dict[str, Any]:
        """Unconditionally recompute :attr:`schema_stats` from the current graph and clear the dirty flag."""
        fn = self._stats_fn
        if fn is None:
            raise RuntimeError("Schema helpers not wired (aetherdialect._schema_reflect did not load)")
        self.schema_stats = fn(self)
        self._stats_dirty = False
        return self.schema_stats

    def ensure_schema_stats(self) -> dict[str, Any]:
        """Recompute :attr:`schema_stats` only when the dirty flag is set or the cached payload is missing/empty; otherwise return the cached value."""
        if self._stats_dirty or not self.schema_stats:
            return self.refresh_schema_stats()
        return self.schema_stats

    @property
    def fk_edges(self) -> list[FKEdge]:
        """
        All foreign-key edges declared on tables in the graph.

        Returns:

            Flattened list of `FKEdge` from every `TableMetadata.foreign_keys`.
        """
        return [fk for table in self.tables.values() for fk in table.foreign_keys]

    @property
    def table_names(self) -> list[str]:
        """
        Table names present in the graph.

        Returns:

            Keys of `tables` as a list.
        """
        return list(self.tables.keys())

    @property
    def database_feature_capability(self) -> DatabaseFeatureCapability:
        """Cached structural feasibility snapshot for tier-conditioned. generators. Returns: :class:`DatabaseFeatureCapability` computed once per graph instance."""
        cached = self._database_feature_capability_cache
        if cached is None:
            cap_fn = self._capability_fn
            if cap_fn is None:
                raise RuntimeError("Schema helpers not wired (aetherdialect._schema_reflect did not load)")
            cached = cap_fn(self)
            object.__setattr__(self, "_database_feature_capability_cache", cached)
        return cached

    def set_helpers(self, stats_fn: _SchemaGraphStatsFn, capability_fn: _SchemaGraphCapabilityFn) -> None:
        """Wire refresh/capability helpers for this graph instance."""
        self._stats_fn = stats_fn
        self._capability_fn = capability_fn
        self._database_feature_capability_cache = None

    @classmethod
    def configure_default_helpers(cls, stats_fn: _SchemaGraphStatsFn, capability_fn: _SchemaGraphCapabilityFn) -> None:
        """Wire process-default helpers applied to new graphs that have none set."""
        cls._default_stats_fn = stats_fn
        cls._default_capability_fn = capability_fn

    def get_column(self, table: str, column: str) -> ColumnMetadata | None:
        """
        Look up column metadata by table and column name.

        Args:

            table: Table name to look up.
            column: Column name within that table.

        Returns:

            `ColumnMetadata` if found, otherwise None.
        """
        if table in self.tables and column in self.tables[table].columns:
            return self.tables[table].columns[column]
        return None

    def _schema_literal_spans_multiple_sources(self) -> bool:
        """Return True when this graph represents a multi-member federation composite."""
        source_ids = {str(tm.source_id or "").strip() for tm in self.tables.values()}
        source_ids.discard("")
        return len(source_ids) >= 2

    def _scrub_federation_prompt_token(self, text: str, forbidden_tokens: frozenset[str]) -> str:
        cleaned = str(text or "").strip()
        if not cleaned or not forbidden_tokens:
            return cleaned
        for token in sorted(forbidden_tokens, key=len, reverse=True):
            if token:
                cleaned = re.sub(re.escape(token), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[_\s]+", " ", cleaned).strip(" _-")
        return cleaned

    def _schema_literal_public_role(self, role: str | None) -> str | None:
        if role is None:
            return None
        r = str(role).strip()
        if not r or r in (ColumnRole.IDENTIFIER.value, ColumnRole.AUDIT.value):
            return None
        return r

    def _schema_literal_column_type_token(self, col: ColumnMetadata) -> str:
        vt = (col.value_type or "").strip()
        if vt:
            return vt
        if col.data_type:
            return ColumnTypeSemantics.data_type_to_value_type(col.data_type)
        return ArrayStorageKind.UNKNOWN

    @staticmethod
    def scrub_schema_prose_for_prompt(text: str) -> str:
        """Strip or neutralise instruction-like lines from schema prose before Ground payloads."""
        raw = str(text or "")
        if not raw.strip():
            return raw
        patterns = tuple(re.compile(p) for p in SCHEMA_INSTRUCTION_LIKE_LINE_PATTERNS)
        kept: list[str] = []
        removed_line = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and any(pat.search(stripped) for pat in patterns):
                removed_line = True
                continue
            kept.append(line)
        if removed_line:
            out = "\n".join(kept).strip()
            out = re.sub(r"\n{3,}", "\n\n", out).strip()
            return out if out else SCHEMA_INSTRUCTION_SCRUB_REPLACEMENT
        matched = False
        cleaned = raw
        for pat in patterns:
            if pat.search(cleaned):
                matched = True
                cleaned = pat.sub("", cleaned)
        if not matched:
            return raw
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r.,;:-—")
        return cleaned if cleaned else SCHEMA_INSTRUCTION_SCRUB_REPLACEMENT

    def _schema_literal_column_object(
        self,
        col: ColumnMetadata,
        *,
        fields: frozenset[str],
        table_name: str | None = None,
        description_overlay: dict[str, Any] | None = None,
        omit_original_name: bool = False,
        scrub_instruction_like: bool = False,
    ) -> dict[str, Any]:
        col_body: dict[str, Any] = {}
        if SCHEMA_FIELD_TYPE in fields:
            col_body["type"] = self._schema_literal_column_type_token(col)
        if not omit_original_name and col.original_name and col.original_name.strip() and col.original_name != col.name:
            col_body["original_name"] = col.original_name.strip()
        if SCHEMA_FIELD_KEYS in fields:
            if col.is_primary_key:
                col_body["pk"] = True
            if col.fk_target:
                col_body["fk"] = f"{col.fk_target[0]}.{col.fk_target[1]}"
            if col.is_unique and not col.is_primary_key:
                col_body["unique"] = True
        if SCHEMA_FIELD_DERIVED in fields and col.is_generated:
            col_body["derived"] = True
        if SCHEMA_FIELD_DESCRIPTION in fields:
            overlay_desc = ""
            overlay_owner: DescriptionOwner | None = None
            if description_overlay and table_name:
                meta_entry = description_overlay.get("column_meta", {}).get(f"{table_name}.{col.name}")
                if isinstance(meta_entry, dict):
                    overlay_desc = str(meta_entry.get("description", "")).strip()
                    overlay_owner = DescriptionOwner.coerce(meta_entry.get("description_owner"))
                    if overlay_owner is None and overlay_desc:
                        overlay_owner = DescriptionOwner.SPACE_NOTES
            desc, _ = DescriptionOwner.resolve(
                (overlay_desc, overlay_owner),
                (col.description, col.description_owner),
            )
            if desc:
                if scrub_instruction_like:
                    desc = SchemaGraph.scrub_schema_prose_for_prompt(desc)
                col_body["description"] = desc
        if SCHEMA_FIELD_ROLE in fields:
            pub_role = self._schema_literal_public_role(col.role)
            if pub_role is not None:
                col_body["role"] = pub_role
        if SCHEMA_FIELD_TRUTH_VALUE in fields and col_body.get("type", "").lower() == "boolean":
            tv = (col.boolean_truth_value or "").strip()
            if tv:
                col_body["truth_value"] = tv
        elif SCHEMA_FIELD_TRUTH_VALUE in fields and SCHEMA_FIELD_TYPE not in fields:
            vt = self._schema_literal_column_type_token(col)
            if vt.lower() == "boolean":
                tv = (col.boolean_truth_value or "").strip()
                if tv:
                    col_body["truth_value"] = tv
        if SCHEMA_FIELD_SAMPLES in fields:
            if col.prompt_value_type() in PROMPT_SCALAR_VALUE_TYPES:
                samples: dict[str, Any] = {}
                if col.distinct_count and col.sensitivity != SensitivityClassification.RESTRICTED:
                    samples["distinct_count"] = col.distinct_count
                if col.min_val is not None:
                    samples["min_val"] = col.min_val
                if col.max_val is not None:
                    samples["max_val"] = col.max_val
                if col.frequent_values:
                    samples["frequent_values"] = [
                        (
                            SchemaGraph.scrub_schema_prose_for_prompt(str(value))
                            if scrub_instruction_like
                            else str(value)
                        )
                        for value in col.frequent_values
                    ]
                if samples:
                    col.assert_prompt_value_scalar(
                        table_name=table_name,
                        contribution="prompt sample",
                    )
                    col_body[SCHEMA_FIELD_SAMPLES] = samples
        if not col_body and SCHEMA_FIELD_TYPE in fields:
            col_body["type"] = self._schema_literal_column_type_token(col)
        return col_body

    def _schema_literal_payload(
        self,
        *,
        fields: frozenset[str],
        table_filter: frozenset[str] | None,
        column_filter: frozenset[str] | None = None,
        owner_master_scope: bool = False,
        description_overlay: dict[str, Any] | None = None,
        scrub_instruction_like: bool = False,
    ) -> dict[str, Any]:
        root: dict[str, Any] = {}
        omit_original_name = True
        col_allow_by_table: dict[str, set[str]] | None = None
        if column_filter:
            col_allow_by_table = {}
            for qc in column_filter:
                if "." not in qc:
                    continue
                tbl, bare_col = qc.rsplit(".", 1)
                col_allow_by_table.setdefault(tbl, set()).add(bare_col)
        for tname in sorted(self.tables):
            if table_filter is not None and tname not in table_filter:
                continue
            tm = self.tables[tname]
            col_map: dict[str, dict[str, Any]] = {}
            allowed_col_names = col_allow_by_table.get(tname) if col_allow_by_table is not None else None
            for col_name in sorted(tm.columns.keys()):
                if allowed_col_names is not None and col_name not in allowed_col_names:
                    continue
                col = tm.columns[col_name]
                if not owner_master_scope and not col.is_visible:
                    continue
                col_obj = self._schema_literal_column_object(
                    col,
                    fields=fields,
                    table_name=tname,
                    description_overlay=description_overlay,
                    omit_original_name=omit_original_name,
                    scrub_instruction_like=scrub_instruction_like,
                )
                if col_obj:
                    col_map[col_name] = col_obj
            table_body: dict[str, Any] = {"columns": col_map}
            if SCHEMA_FIELD_DESCRIPTION in fields:
                overlay_desc = ""
                overlay_owner: DescriptionOwner | None = None
                if description_overlay:
                    overlay_desc = str(description_overlay.get("table_descriptions", {}).get(tname, "")).strip()
                    if overlay_desc:
                        owners = description_overlay.get("_table_description_owners") or {}
                        raw_owner = owners.get(tname) if isinstance(owners, dict) else None
                        overlay_owner = DescriptionOwner.coerce(raw_owner) if raw_owner else None
                        if overlay_owner is None:
                            overlay_owner = DescriptionOwner.SPACE_NOTES
                td, _ = DescriptionOwner.resolve(
                    (overlay_desc, overlay_owner),
                    (tm.description, tm.description_owner),
                )
                if td:
                    if scrub_instruction_like:
                        td = SchemaGraph.scrub_schema_prose_for_prompt(td)
                    table_body["description"] = td
            if SCHEMA_FIELD_ROLE in fields:
                tr = self._schema_literal_public_role(tm.role)
                if tr is not None:
                    table_body["role"] = tr
            if not omit_original_name and tm.original_name and tm.original_name.strip() and tm.original_name != tm.name:
                table_body["original_name"] = tm.original_name.strip()
            root[tname] = table_body
        if SCHEMA_FIELD_ENUM in fields and self.enum_values:
            enum_block: dict[str, Any] = {}
            truncated_enums: list[str] = []
            forbidden_tokens = getattr(self, "_federation_description_forbidden_tokens", None) or frozenset()
            for ename in sorted(self.enum_values.keys()):
                values = self.enum_values[ename]
                if values:
                    prompt_enum_name = ename.split("::", 1)[-1]
                    for tname, tm in self.tables.items():
                        for col in tm.columns.values():
                            enum_type_name = (col.enum_type_name or "").strip()
                            if not enum_type_name:
                                continue
                            if enum_type_name == ename or enum_type_name.split("::", 1)[-1] == prompt_enum_name:
                                col.assert_prompt_value_scalar(
                                    table_name=tname,
                                    contribution="enum label",
                                )
                prompt_name = ename.split("::", 1)[-1]
                if forbidden_tokens:
                    prompt_name = self._scrub_federation_prompt_token(prompt_name, forbidden_tokens) or prompt_name
                capped_values = list(values[:FEDERATION_ENUM_PROMPT_CAP])
                if len(values) > FEDERATION_ENUM_PROMPT_CAP:
                    capped_values.append("...")
                    truncated_enums.append(prompt_name)
                prompt_values: list[str] = []
                for value in capped_values:
                    if str(value) == "...":
                        prompt_values.append("...")
                        continue
                    scrubbed = (
                        self._scrub_federation_prompt_token(str(value), forbidden_tokens)
                        if forbidden_tokens
                        else str(value)
                    )
                    if forbidden_tokens and not scrubbed:
                        raise ConfigError("composite enum label must not name a source or member")
                    prompt_values.append(
                        SchemaGraph.scrub_schema_prose_for_prompt(scrubbed) if scrub_instruction_like else scrubbed
                    )
                enum_block[prompt_name] = prompt_values
            if truncated_enums:
                object.__setattr__(self, "_last_enum_truncations", tuple(truncated_enums))
            root["enum_types"] = enum_block
        if not omit_original_name and any(
            (tm.original_name and tm.original_name != tm.name)
            or any(col.original_name and col.original_name != col.name for col in tm.columns.values())
            for tm in self.tables.values()
        ):
            root["upload_label_note"] = CSV_SCHEMA_LITERAL_ORIGINAL_NAME_NOTE
        return root

    def _resolve_payload_where(
        self,
        *,
        visible_objects: frozenset[str] | None = None,
        allowed_columns: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        deny_columns: frozenset[str] | None = None,
    ) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        """Derive table and column filters from interactive scope kwargs."""
        graph_tables = frozenset(self.tables.keys())
        if visible_objects is not None:
            vis_tables = frozenset(t for t in visible_objects if t in self.tables)
        else:
            vis_tables = frozenset()
        if deny_objects:
            deny_set = set(deny_objects)
            base_tables = vis_tables if vis_tables else graph_tables
            vis_tables = frozenset(t for t in base_tables if t not in deny_set)
        col_restrict = frozenset(allowed_columns) if allowed_columns else frozenset()
        if deny_columns:
            deny_col_set = set(deny_columns)
            if col_restrict:
                col_restrict = frozenset(qc for qc in col_restrict if qc not in deny_col_set)
            else:
                all_cols = frozenset(
                    f"{tname}.{cname}" for tname in sorted(self.tables) for cname in sorted(self.tables[tname].columns)
                )
                col_restrict = frozenset(qc for qc in all_cols if qc not in deny_col_set)
        if col_restrict:
            scope_tables = vis_tables if vis_tables else graph_tables
            return scope_tables, col_restrict
        if vis_tables:
            return vis_tables, None
        if deny_objects or deny_columns:
            return vis_tables, None
        return None, None

    def schema_payload_json(
        self,
        fields: frozenset[str],
        *,
        table_filter: frozenset[str] | None = None,
        column_filter: frozenset[str] | None = None,
        owner_master_scope: bool = False,
        description_overlay: dict[str, Any] | None = None,
        scrub_instruction_like: bool = False,
    ) -> str:
        """Serialize a field-filtered schema payload for LLM prompts."""
        payload = self._schema_literal_payload(
            fields=fields,
            table_filter=table_filter,
            column_filter=column_filter,
            owner_master_scope=owner_master_scope,
            description_overlay=description_overlay,
            scrub_instruction_like=scrub_instruction_like,
        )
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def schema_payload_interpret(
        self,
        *,
        visible_objects: frozenset[str] | None = None,
        allowed_columns: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        deny_columns: frozenset[str] | None = None,
        owner_master_scope: bool = False,
        description_overlay: dict[str, Any] | None = None,
    ) -> str:
        """Domain-only schema payload (descriptions and enum heads) for the Interpret stage."""
        table_filter, column_filter = self._resolve_payload_where(
            visible_objects=visible_objects,
            allowed_columns=allowed_columns,
            deny_objects=deny_objects,
            deny_columns=deny_columns,
        )
        return self.schema_payload_json(
            INTERPRET_FIELDS,
            table_filter=table_filter,
            column_filter=column_filter,
            owner_master_scope=owner_master_scope,
            description_overlay=description_overlay,
        )

    def schema_payload_ground(
        self,
        *,
        visible_objects: frozenset[str] | None = None,
        allowed_columns: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        deny_columns: frozenset[str] | None = None,
        owner_master_scope: bool = False,
        description_overlay: dict[str, Any] | None = None,
    ) -> str:
        """Ground-stage schema payload with descriptions, roles, and value types."""
        table_filter, column_filter = self._resolve_payload_where(
            visible_objects=visible_objects,
            allowed_columns=allowed_columns,
            deny_objects=deny_objects,
            deny_columns=deny_columns,
        )
        return self.schema_payload_json(
            GROUND_FIELDS,
            table_filter=table_filter,
            column_filter=column_filter,
            owner_master_scope=owner_master_scope,
            description_overlay=description_overlay,
            scrub_instruction_like=True,
        )

    def schema_payload_compose(self, tables: Iterable[str], *, owner_master_scope: bool = False) -> str:
        """Compose-stage structural schema payload scoped to chosen tables."""
        filt = frozenset(str(t) for t in tables if t in self.tables)
        return self.schema_payload_json(
            COMPOSE_FIELDS, table_filter=filt if filt else None, owner_master_scope=owner_master_scope
        )

    @property
    def schema_literal_json(self) -> str:
        """JSON string describing visible, scope-permitted tables and columns for LLM prompts."""
        return self.schema_payload_json(FULL_FIELDS)

    def schema_literal_json_for_tables(self, allowed_tables: frozenset[str]) -> str:
        """Schema literal JSON restricted to *allowed_tables* (consumer ``visible_objects`` whitelist)."""
        return self.schema_payload_json(FULL_FIELDS, table_filter=allowed_tables)

    def schema_literal_json_for_scope(self, allowed_tables: frozenset[str], allowed_columns: frozenset[str]) -> str:
        """Schema literal JSON restricted to *allowed_tables* and optionally *allowed_columns*."""
        table_filter = allowed_tables if allowed_tables else None
        column_filter = allowed_columns if allowed_columns else None
        return self.schema_payload_json(FULL_FIELDS, table_filter=table_filter, column_filter=column_filter)

    @staticmethod
    def resolve_intent_visible_objects(
        *,
        visible_objects: frozenset[str] | None,
        execution_visible_objects: frozenset[str] | None,
    ) -> frozenset[str] | None:
        """Intersect space and credential/execution whitelists for post- LLM gates. Used at SQL generation and execution time. LLM prompt payloads must not apply ``execution_visible_objects``; those stay owner- graph complete aside from optional aetherspace ``visible_objects`` filters."""
        if visible_objects is None:
            return execution_visible_objects
        if execution_visible_objects is None:
            return frozenset(visible_objects)
        return frozenset(visible_objects) & frozenset(execution_visible_objects)

    def validate_tables_exist(self, tables: tuple[str, ...]) -> list[IntentIssue]:
        """Emit logical-stage issues for interpret table tokens absent from this schema graph."""
        known = frozenset(self.tables.keys())
        out: list[IntentIssue] = []
        for name in tables:
            if name in known:
                continue
            out.append(
                IntentIssue.make(
                    issue_id=f"unknown_table_{name}",
                    category=FailureCategory.UNKNOWN_TABLE,
                    severity="error",
                    message=f"Unknown table {name!r} in interpret tables list.",
                    context={"table": name},
                    responsible_stage="ground",
                )
            )
        return out

    def validate_cte_tables_and_dag(self, logical: Any) -> list[IntentIssue]:
        """Validate CTE table tokens against schema tables and prior step names; reject cycles."""
        issues: list[IntentIssue] = []
        steps = list(getattr(logical, "cte_steps", None) or ())
        if not steps:
            return issues
        known_schema = frozenset(self.tables.keys())
        prior_names: set[str] = set()
        seen_step_names: set[str] = set()
        step_names_ordered: list[str] = []
        for idx_, step in enumerate(steps):
            name = (getattr(step, "name", None) or "").strip()
            if not name:
                issues.append(
                    IntentIssue.make(
                        issue_id=f"cte_empty_name_{idx_}",
                        category=FailureCategory.CTE_TABLE_REFERENCE,
                        severity="error",
                        message="CTE step is missing a non-empty name.",
                        context={"index": idx_},
                        responsible_stage="ground",
                    )
                )
                continue
            if name in seen_step_names:
                issues.append(
                    IntentIssue.make(
                        issue_id=f"cte_duplicate_name_{name}",
                        category=FailureCategory.CTE_TABLE_REFERENCE,
                        severity="error",
                        message=f"Duplicate CTE step name {name!r}.",
                        context={"cte": name, "index": idx_},
                        responsible_stage="ground",
                    )
                )
            seen_step_names.add(name)
            step_names_ordered.append(name)
            for token in list(getattr(step, "tables", None) or ()):
                tok = str(token or "").strip()
                if not tok:
                    continue
                if tok in known_schema or tok in prior_names:
                    continue
                issues.append(
                    IntentIssue.make(
                        issue_id=f"cte_unknown_table_{name}_{tok}",
                        category=FailureCategory.CTE_TABLE_REFERENCE,
                        severity="error",
                        message=f"CTE {name!r} references unknown table token {tok!r}.",
                        context={"cte": name, "table": tok},
                        responsible_stage="ground",
                    )
                )
            prior_names.add(name)
        name_set = set(step_names_ordered)
        deps: dict[str, set[str]] = {n: set() for n in step_names_ordered}
        for step in steps:
            name = (getattr(step, "name", None) or "").strip()
            if not name:
                continue
            for token in list(getattr(step, "tables", None) or ()):
                tok = str(token or "").strip()
                if tok in name_set and tok != name:
                    deps[name].add(tok)
        visiting: set[str] = set()
        visited: set[str] = set()

        def _dfs(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for nxt in deps.get(node, ()):
                if _dfs(nxt):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        for n in step_names_ordered:
            if _dfs(n):
                issues.append(
                    IntentIssue.make(
                        issue_id="cte_cycle",
                        category=FailureCategory.CTE_TABLE_REFERENCE,
                        severity="error",
                        message="CTE steps form a dependency cycle.",
                        context={"steps": step_names_ordered},
                        responsible_stage="ground",
                    )
                )
                break
        return issues

    def structural_schema_literal_json(self, tables: Iterable[str] | None = None) -> str:
        """
        Structural schema JSON with descriptions stripped,

        optionally.

        restricted to *tables*. Args: tables: When ``None``, every graph

        table is included; otherwise only listed names that exist.

        Returns:

            Compact JSON text; unknown table names in *tables* are ignored.
        """
        filt: frozenset[str] | None = frozenset(str(t) for t in tables) if tables is not None else None
        return self.schema_payload_json(COMPOSE_FIELDS, table_filter=filt)

    @staticmethod
    def _structural_knowledge_from_cache(raw: Any) -> tuple[StructuralKnowledgeFact, ...]:
        """Parse cached ``structural_knowledge`` list into facts. Rejects unknown shapes."""
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ConfigError("structural_knowledge cache must be a list")
        out: list[StructuralKnowledgeFact] = []
        seen: set[str] = set()
        allowed_fact_keys = frozenset({"kind", "text", "referenced_entities", "payload"})
        for item in raw:
            if not isinstance(item, dict):
                raise ConfigError("structural_knowledge cache items must be objects")
            keys = set(item.keys())
            if keys == {"target", "text"} or keys == {"text", "target"}:
                raise ConfigError("structural_knowledge target/text facts are not supported; re-derive from notes")
            extra = keys - allowed_fact_keys
            if extra:
                raise ConfigError(f"structural_knowledge fact has unexpected keys: {sorted(extra)}")
            if "kind" not in item or "text" not in item:
                raise ConfigError("structural_knowledge fact requires kind and text")
            kind = str(item.get("kind") or "").strip().lower()
            if kind == "residual":
                raise ConfigError("structural knowledge kind residual is not supported; re-derive from notes")
            text_val = str(item.get("text") or "").strip()
            if "referenced_entities" not in item:
                raise ConfigError("structural_knowledge fact requires referenced_entities")
            raw_referenced = item.get("referenced_entities")
            if not isinstance(raw_referenced, list) or not all(isinstance(r, str) for r in raw_referenced):
                raise ConfigError("structural_knowledge referenced_entities must be a list of strings")
            referenced_entities = frozenset(r.strip() for r in raw_referenced if r.strip())
            payload_raw = item.get("payload")
            if payload_raw is None:
                payload: dict[str, Any] | None = None if "payload" not in item else {}
            elif isinstance(payload_raw, dict):
                payload = payload_raw
            else:
                raise ConfigError("structural_knowledge payload must be an object when present")
            fact = StructuralKnowledgeFact.normalize(
                StructuralKnowledgeFact(
                    kind=kind,
                    text=text_val,
                    referenced_entities=referenced_entities,
                    payload=payload,
                )
            )
            dedupe = f"{fact.kind}::{fact.text.lower()}"
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(fact)
        return tuple(out)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SchemaGraph:
        """
        Create `SchemaGraph` from a dictionary.

        Args:

            d: Dictionary with keys matching `SchemaGraph` fields, typically loaded from JSON.

        Returns:

            Populated `SchemaGraph` with nested `TableMetadata` instances.
        """
        tables_raw = d.get("tables", {})
        tables = {k: TableMetadata.from_dict(v) for k, v in tables_raw.items()}
        deny_cols_raw = d.get("deny_columns", {})
        deny_columns: dict[str, set[str]] = {}
        if isinstance(deny_cols_raw, dict):
            for tbl, cols in deny_cols_raw.items():
                if isinstance(cols, list):
                    deny_columns[str(tbl)] = set(str(c) for c in cols)
        disallowed_raw = d.get("disallowed_columns", {})
        disallowed_columns: dict[str, set[str]] = {}
        if isinstance(disallowed_raw, dict):
            for tbl, cols in disallowed_raw.items():
                if isinstance(cols, list):
                    disallowed_columns[str(tbl)] = set(str(c) for c in cols)
        structural_hash = str(d.get("structural_hash", "") or "")
        profiling_hash = str(d.get("profiling_hash", "") or "")
        descriptions_hash = str(d.get("descriptions_hash", "") or "")
        scope_hash = str(d.get("scope_hash", "") or "")
        effective_structural_hash = str(d.get("effective_structural_hash", "") or "")
        schema_graph_id = str(d.get("schema_graph_id", "") or "")
        inc_raw = d.get("include")
        if inc_raw is not None:
            include_val = SchemaInclude.coerce(inc_raw)
        else:
            okind = d.get("object_kind", "table")
            include_val = SchemaInclude.VIEWS if okind == "view" else SchemaInclude.TABLES
        if "schema_revision" in d:
            schema_revision = int(d.get("schema_revision") or 0)
        else:
            schema_revision = 1 if tables else 0
        return SchemaGraph(
            tables=tables,
            join_paths_multi=d.get("join_paths_multi", {}),
            structural_hash=structural_hash,
            profiling_hash=profiling_hash,
            descriptions_hash=descriptions_hash,
            scope_hash=scope_hash,
            effective_structural_hash=effective_structural_hash,
            schema_graph_id=schema_graph_id,
            notes_hash=str(d.get("notes_hash", "") or ""),
            semantic_edges_hash=str(d.get("semantic_edges_hash", "") or ""),
            ddl_probe_hash=str(d.get("ddl_probe_hash", "") or ""),
            include=include_val,
            created_at=d.get("created_at", ""),
            enum_values=d.get("enum_values"),
            schema_stats=d.get("schema_stats"),
            deny_columns=deny_columns,
            disallowed_columns=disallowed_columns,
            notes_sha256=str(d.get("notes_sha256", "") or ""),
            structural_knowledge=SchemaGraph._structural_knowledge_from_cache(d.get("structural_knowledge")),
            scope_descriptor=(d.get("scope_descriptor") if isinstance(d.get("scope_descriptor"), dict) else None),
            federation_membership=(
                dict(d["federation_membership"]) if isinstance(d.get("federation_membership"), dict) else None
            ),
            schema_revision=schema_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON storage.

        Returns:

            Dictionary with all `SchemaGraph` fields; nested tables are serialized recursively.
        """
        return {
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "join_paths_multi": self.join_paths_multi,
            "structural_hash": self.structural_hash,
            "profiling_hash": self.profiling_hash,
            "descriptions_hash": self.descriptions_hash,
            "scope_hash": self.scope_hash,
            "effective_structural_hash": self.effective_structural_hash,
            "schema_graph_id": self.schema_graph_id,
            "notes_hash": self.notes_hash,
            "semantic_edges_hash": self.semantic_edges_hash,
            "ddl_probe_hash": self.ddl_probe_hash,
            "include": SchemaInclude.coerce(self.include).value,
            "created_at": self.created_at,
            "enum_values": self.enum_values,
            "schema_stats": self.schema_stats,
            "deny_columns": {k: sorted(v) for k, v in self.deny_columns.items()},
            "disallowed_columns": {k: sorted(v) for k, v in self.disallowed_columns.items()},
            "notes_sha256": self.notes_sha256,
            "structural_knowledge": [f.to_dict() for f in self.structural_knowledge],
            "scope_descriptor": self.scope_descriptor,
            "federation_membership": self.federation_membership,
            "schema_revision": self.schema_revision,
        }


@dataclass
class VirtualColumnSpec:
    """Join-discovery view of one CTE output column with lifted physical lineage."""

    lineage_phys_table: str | None
    lineage_phys_column: str | None
    inherits_pk: bool
    fk_to: tuple[str, str] | None
    semantic_distinct_values: list[str]
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class VirtualTableSpec:
    """In-memory join graph node for a CTE keyed by ``cte_name``."""

    cte_name: str
    columns: dict[str, VirtualColumnSpec]
    emission: str = "join_table"


@dataclass
class RetryFailureContext:
    """Structured failure context for LLM retry guidance."""

    failure_type: str
    required_tables: list[str]
    used_tables: set[str]
    missing_tables: set[str]
    attempt_number: int


@dataclass
class SQLShape:
    """Structural features of a SQL query for comparison."""

    num_joins: int
    has_group_by: bool
    has_agg: bool
    num_cte: int = 0
    num_where: int = 0
    num_having: int = 0
    has_distinct: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SQLShape:
        """
        Create `SQLShape` from a dictionary.

        Args:

            d: Dictionary with keys matching `SQLShape` fields.

        Returns:

            Populated `SQLShape` instance.
        """
        return SQLShape(
            num_joins=d.get("num_joins", 0),
            has_group_by=d.get("has_group_by", False),
            has_agg=d.get("has_agg", False),
            num_cte=d.get("num_cte", 0),
            num_where=d.get("num_where", 0),
            num_having=d.get("num_having", 0),
            has_distinct=d.get("has_distinct", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize shape flags to a plain dict.

        Returns:

            `asdict` of all fields.
        """
        return asdict(self)


@dataclass
class FKEdge:
    """Foreign key relationship between two tables."""

    src_table: str
    src_cols: list[str]
    dst_table: str
    dst_cols: list[str]
    inference_tag: InferenceTag | None = None
    join_kind: str | None = None

    def __post_init__(self) -> None:
        """Coerce ``inference_tag`` from raw cache strings into :class:`InferenceTag`."""
        if not isinstance(self.inference_tag, InferenceTag):
            self.inference_tag = InferenceTag.coerce(self.inference_tag)


@dataclass
class CatalogTableStructuralConstraints:
    """Catalog-sourced primary-key column names, foreign-key edges, and single-column unique names for one table. Each:class:`FKEdge` carries ``src_table`` equal to the referencing table so the bundle can be converted into ``tables_meta`` foreign- key dicts without losing the child table identity."""

    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[FKEdge] = field(default_factory=list)
    unique_columns: list[str] = field(default_factory=list)


@dataclass
class CatalogStructuralConstraintsIndex:
    """Per-table structural constraint bundles keyed by lowercased relation name within one catalog schema. When ``tables`` is empty the caller should treat catalog reflection as unavailable and continue with DDL-based parsing."""

    tables: dict[str, CatalogTableStructuralConstraints] = field(default_factory=dict)
    column_nullability: dict[str, dict[str, bool]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CatalogStructuralConstraintsIndex:
        """Construct an empty index for failed information_schema. queries. Returns: Empty :class:`CatalogStructuralConstraintsIndex` instance."""
        return cls(tables={})


@dataclass
class ValueDomain:
    """Value domain for sampling concrete values during question generation."""

    values: list[str] = field(default_factory=list)
    min_val: str | None = None
    max_val: str | None = None
    data_type: str | None = None
    value_type: str = ""

    def __post_init__(self) -> None:
        """Derive ``value_type`` from ``data_type`` when unset."""
        if not self.value_type and self.data_type:
            self.value_type = ColumnTypeSemantics.data_type_to_value_type(self.data_type)


@dataclass
class IntentIssue:
    """Issue detected during intent validation or resolution."""

    issue_id: str
    category: FailureCategory
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    responsible_stage: Literal["ground", "compose", "interpret"] = "compose"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> IntentIssue:
        """
        Create `IntentIssue` from a dictionary.

        Args:

            d: Dictionary with keys matching `IntentIssue` fields.

        Returns:

            Populated `IntentIssue` instance.
        """
        raw_cat = d.get("category", "")
        if isinstance(raw_cat, FailureCategory):
            category: FailureCategory = raw_cat
        else:
            category = FailureCategory.parse(str(raw_cat) if raw_cat is not None else None) or FailureCategory.OTHER
        rs = d.get("responsible_stage", "compose")
        if rs == "ground":
            stage: Literal["ground", "compose", "interpret"] = "ground"
        elif rs == "interpret":
            stage = "interpret"
        else:
            stage = "compose"
        return IntentIssue(
            issue_id=d.get("issue_id", ""),
            category=category,
            severity=d.get("severity", "error"),
            message=d.get("message", ""),
            context=d.get("context", {}),
            responsible_stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the issue to a plain dict.

        Returns:

            Primitive field mapping including `context`.
        """
        return {
            "issue_id": self.issue_id,
            "category": self.category.value,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
            "responsible_stage": self.responsible_stage,
        }

    @staticmethod
    def make(
        *,
        issue_id: str,
        category: FailureCategory,
        severity: str,
        message: str,
        context: dict[str, Any] | None = None,
        responsible_stage: Literal["ground", "compose", "interpret"] | None = None,
    ) -> IntentIssue:
        """Construct an :class:`IntentIssue` with ``responsible_stage`` from attribution table when omitted."""
        ctx = dict(context or {})
        if responsible_stage is not None:
            return IntentIssue(
                issue_id=issue_id,
                category=category,
                severity=severity,
                message=message,
                context=ctx,
                responsible_stage=responsible_stage,
            )
        iid = (issue_id or "").lower()
        for key, stage in STAGE_ATTRIBUTION_TABLE.items():
            if key in iid:
                return IntentIssue(
                    issue_id=issue_id,
                    category=category,
                    severity=severity,
                    message=message,
                    context=ctx,
                    responsible_stage=stage,
                )
        if category.value in LOGICAL_FAILURE_CATEGORIES:
            inferred: Literal["ground", "compose"] = "ground"
        else:
            inferred = "compose"
        return IntentIssue(
            issue_id=issue_id,
            category=category,
            severity=severity,
            message=message,
            context=ctx,
            responsible_stage=inferred,
        )


@dataclass
class IntentValidationResult:
    """Result container for intent validation with issue tracking. Only ``error`` and ``warning`` severity issues are retained; any ``info`` (or otherwise non-actionable) severity issue is dropped at construction time so downstream consumers never have to filter them out."""

    issues: list[IntentIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Drop any issue whose severity is not ``error`` or ``warning``."""
        self.issues = [i for i in self.issues if i.severity in KEPT_ISSUE_SEVERITIES]

    @property
    def is_valid(self) -> bool:
        """
        Whether validation found no error-severity issues.

        Returns:

            True if no `IntentIssue` has `severity == 'error'`.
        """
        return not any(i.severity == "error" for i in self.issues)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> IntentValidationResult:
        """
        Create `IntentValidationResult` from a dictionary.

        Args:

            d: Dictionary with an `issues` list of serialized `IntentIssue` dicts.

        Returns:

            Populated `IntentValidationResult` with deserialized `IntentIssue` objects.
        """
        issues_raw = d.get("issues", [])
        return IntentValidationResult(issues=[IntentIssue.from_dict(i) for i in issues_raw])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize validation result for JSON.

        Returns:

            Dict with an `issues` list of serialized `IntentIssue` dicts.
        """
        return {"issues": [i.to_dict() for i in self.issues]}


@dataclass
class TemplateStats:
    """Template acceptance and rejection statistics."""

    accept: int = 0
    reject: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TemplateStats:
        """
        Create `TemplateStats` from a dictionary.

        Args:

            d: Dictionary with `accept` and `reject` integer keys.

        Returns:

            Populated `TemplateStats` instance.
        """
        return TemplateStats(accept=int(d.get("accept", 0)), reject=int(d.get("reject", 0)))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize accept/reject counts.

        Returns:

            `asdict` of `accept` and `reject`.
        """
        return asdict(self)


@dataclass
class QSimSkeleton:
    """Structural skeleton for QSim intent before the LLM fills semantics."""

    tables: list[str]
    has_aggregation: bool
    num_where: int
    num_groupby: int
    has_orderby: bool
    num_having: int
    has_distinct: bool = False
    has_expr_comparison: bool = False
    advanced_slot: str | None = None

    def complexity_tier(self) -> ComplexityTier:
        """Map this structural skeleton to the discrete tier used for quota sampling."""
        tables_n = len(self.tables or [])
        hav_n = int(self.num_having)
        group_n = int(self.num_groupby)
        agg = self.has_aggregation
        if tables_n >= 3 or hav_n >= 1 or (agg and group_n >= 1) or self.has_expr_comparison:
            return ComplexityTier.COMPLEX
        if tables_n >= 2 or agg or group_n >= 1 or self.has_orderby or self.has_distinct:
            return ComplexityTier.MODERATE
        return ComplexityTier.SIMPLE


@dataclass
class SkeletonPool:
    """Tiered skeleton pool with round-robin table-set selection."""

    tier_a_by_table_set: dict[str, list[QSimSkeleton]]
    tier_b_by_table_set: dict[str, list[QSimSkeleton]]
    tier_c_by_table_set: dict[str, list[QSimSkeleton]]
    table_set_keys: list[str]
    tier_a_indices: dict[str, int]
    tier_b_indices: dict[str, int]
    tier_c_indices: dict[str, int]
    current_table_idx: int = 0


@dataclass
class SeedWarmupSummary:
    """Aggregate statistics for a seed warmup run."""

    version: int
    total: int
    success: int
    failed: int
    success_rate: float
    seed_questions_loaded: int = 0
    gold_intents_total: int = 0
    unique_prompts: int = 0
    gold_new: int = 0
    gold_skipped: int = 0
    gold_failed: int = 0
    gold_user_rejected: int = 0
    deduped_prompts_count: int = 0
    gold_prompts_count: int = 0
    templates_added: int = 0
    validation_drop: int = 0
    realism_drop: int = 0
    question_generation_failed: int = 0
    early_pipeline_failed: int = 0


@dataclass
class QSimSummary:
    """QSim (question generation) run metadata with version, counts, and seed."""

    version: int
    num_intents: int
    num_questions: int
    seed: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimSummary:
        """
        Create `QSimSummary` from a dictionary.

        Args:

            d: Dictionary with keys matching `QSimSummary` fields.

        Returns:

            Populated `QSimSummary` instance.
        """
        return QSimSummary(
            version=int(d.get("version", 0)),
            num_intents=d.get("num_intents", 0),
            num_questions=d.get("num_questions", 0),
            seed=d.get("seed", DEFAULT_RANDOM_SEED),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize QSim run metadata.

        Returns:

            `asdict` of version, counts, and seed.
        """
        return asdict(self)


@dataclass
class SchemaLimits:
    """Internal schema-based limits for adaptive parameter validation."""

    max_where_predicates: int
    max_groupby: int
    max_tables: int


@dataclass
class SkeletonLimits:
    """Schema-derived limits for QSim skeleton enumeration."""

    max_where_predicates: int
    max_groupby: int
    max_having: int


@dataclass
class QSimWhereParam:
    """Lightweight filter for QSim intent with column reference and operator."""

    column: str
    op: str
    value_type: str
    right_column: str = ""

    @property
    def is_expr_comparison(self) -> bool:
        """Whether the filter compares two expressions via. `right_column`. Returns: True when `right_column` is non-empty."""
        return bool(self.right_column)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimWhereParam:
        """
        Create QSimWhereParam from dictionary.

        Args:

            d: Dictionary with 'column', 'op', 'value_type', and optional 'right_column' keys.

        Returns:

            Populated QSimWhereParam instance.
        """
        return QSimWhereParam(
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
        """Whether HAVING compares two expressions via. `right_expression`. Returns: True when `right_expression` is non-empty."""
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


@dataclass
class QSimIntent:
    """Unified intent for QSim question generation with optional values."""

    intent_id: str
    tables: list[str]
    grain: str
    select_cols: list[str]
    group_by_cols: list[str]
    order_by_cols: list[str]
    where: list[QSimWhereParam]
    having_param: list[QSimHaving]
    param_values: dict[str, ParamValue] = field(default_factory=dict)
    question: str = ""
    variant_idx: int = 0
    limit: int | None = None
    distinct: bool = False

    def complexity_tier(self) -> ComplexityTier:
        """Classify using observable SQL-shape cues without CTE or window registries."""
        tables_n = len(self.tables or [])
        hav_n = len(self.having_param or [])
        group_n = len(self.group_by_cols or [])
        sel_cols = self.select_cols or []
        has_agg = any(AGG_PATTERN.match(str(sc)) for sc in sel_cols)
        has_ord = len(self.order_by_cols or []) > 0
        lim_set = self.limit is not None
        if tables_n >= 3 or hav_n >= 1 or (has_agg and group_n >= 1):
            return ComplexityTier.COMPLEX
        if tables_n >= 2 or has_agg or group_n >= 1 or has_ord or lim_set or self.distinct:
            return ComplexityTier.MODERATE
        return ComplexityTier.SIMPLE

    @staticmethod
    def matches_target_tier(classified: ComplexityTier, target: ComplexityTier) -> bool:
        """Return whether a filled intent meets the structural floor for the target tier."""
        rank_map = {
            ComplexityTier.SIMPLE: 0,
            ComplexityTier.MODERATE: 1,
            ComplexityTier.COMPLEX: 2,
            ComplexityTier.HIGHLY_COMPLEX: 3,
        }
        c = rank_map[classified]
        need = min(rank_map[target], rank_map[ComplexityTier.COMPLEX])
        return c >= need

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QSimIntent:
        """
        Create QSimIntent from dictionary.

        Args:

            d: Dictionary with keys matching QSimIntent fields.

        Returns:

            Populated QSimIntent instance.
        """
        fp_raw = d.get("where", [])
        hp_raw = d.get("having", [])
        return QSimIntent(
            intent_id=d.get("intent_id", ""),
            tables=d.get("tables", []),
            grain=d.get("grain", "row_level"),
            select_cols=d.get("select_cols", []),
            group_by_cols=d.get("group_by_cols", []),
            order_by_cols=d.get("order_by_cols", []),
            where=[QSimWhereParam.from_dict(fp) if isinstance(fp, dict) else fp for fp in fp_raw],
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
            "where": [fp.to_dict() for fp in self.where],
            "having_param": [hp.to_dict() for hp in self.having_param],
            "param_values": self.param_values,
            "question": self.question,
            "variant_idx": self.variant_idx,
            "limit": self.limit,
            "distinct": self.distinct,
        }


@dataclass
class WindowSpec:
    """Window function specification for a SELECT column (dialect- agnostic)."""

    function: str
    partition_by: list[NormalizedExpr] = field(default_factory=list)
    order_by: list[OrderByCol] = field(default_factory=list)
    argument: NormalizedExpr | None = None
    numeric_argument: int | None = None
    frame_kind: WindowFrameKind = WindowFrameKind.NONE
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
        self.function = str(self.function or "").strip().lower()

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
        for p in part_raw or []:
            if isinstance(p, dict):
                partition_by.append(NormalizedExpr.from_dict(p))
            elif isinstance(p, str):
                partition_by.append(NormalizedExpr.parse_string_for_json(p))
        ob_raw = d.get("order_by", [])
        order_by: list[OrderByCol] = []
        for o in ob_raw or []:
            if isinstance(o, dict):
                order_by.append(OrderByCol.from_dict(o))
            elif isinstance(o, str):
                order_by.append(OrderByCol(expr=NormalizedExpr.parse_string_for_json(o)))
        arg_raw = d.get("argument")
        argument = None
        if isinstance(arg_raw, dict):
            argument = NormalizedExpr.from_dict(arg_raw)
        elif isinstance(arg_raw, str) and arg_raw:
            argument = NormalizedExpr.parse_string_for_json(arg_raw)
        fk_raw = str(d.get("frame_kind") or "none").strip().lower()
        frame_kind: WindowFrameKind = (
            WindowFrameKind.ROWS
            if fk_raw == "rows"
            else (WindowFrameKind.RANGE if fk_raw == "range" else WindowFrameKind.NONE)
        )
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

        fn_raw = d.get("function")
        function = str(fn_raw).strip() if isinstance(fn_raw, str) else ""
        return WindowSpec(
            function=function,
            partition_by=partition_by,
            order_by=order_by,
            argument=argument,
            numeric_argument=_off(d.get("numeric_argument")),
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
        if self.numeric_argument is not None:
            out["numeric_argument"] = self.numeric_argument
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
        """Stable key over window function, partition, order, and. argument exprs. Returns: Pipe-separated string prefixed with `win=`."""
        parts = [f"win={self.function}"]
        parts.extend(f"p:{e.signature_key}" for e in self.partition_by)
        parts.extend(f"o:{o.signature_key}" for o in self.order_by)
        if self.argument:
            parts.append(f"a:{self.argument.signature_key}")
        if self.numeric_argument is not None:
            parts.append(f"n:{self.numeric_argument}")
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
        "numeric_argument": "Positive integer bucket count or offset for ntile and nth_value.",
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
            out["partition_by"] = [e.prompt_sql() for e in self.partition_by]
        if self.order_by:
            out["order_by"] = [o.to_prompt_dict() for o in self.order_by]
        if self.argument is not None and self.argument.signature_key:
            out["argument"] = self.argument.prompt_sql()
        if self.numeric_argument is not None:
            out["numeric_argument"] = self.numeric_argument
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
            "function": "ntile",
            "numeric_argument": 4,
            "partition_by": [],
            "order_by": [{"expr": "table.column", "direction": "desc"}],
        }


@dataclass
class CaseWhenBranch:
    """Single WHEN branch for a CASE expression in SELECT."""

    condition: WhereParam = field(default_factory=WhereParam)
    result: NormalizedExpr = field(default_factory=NormalizedExpr)

    @staticmethod
    def string_result_expr(value: str) -> NormalizedExpr:
        """Interpret a JSON string CASE THEN/ELSE token as column ref or string literal."""
        t = (value or "").strip()
        if not t:
            return NormalizedExpr()
        if CASE_WHEN_QUALIFIED_COLUMN_REF_RE.fullmatch(t):
            return NormalizedExpr.from_column(t)
        return NormalizedExpr(string_literal=t)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseWhenBranch:
        """Parse one `WHEN ... THEN ...` branch from a dict. Args: d: Mapping with `condition` and `result` (dict or string for result). Returns: `CaseWhenBranch` with `WhereParam` and `NormalizedExpr`."""
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
                res_expr = CaseWhenBranch.string_result_expr(str(res))
            else:
                res_expr = NormalizedExpr()
        return CaseWhenBranch(
            condition=replace(WhereParam.from_dict(cond) if isinstance(cond, dict) else WhereParam()), result=res_expr
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
            out["result"] = self.result.prompt_sql()
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Example CASE branch for prompts."""
        return {
            "condition": WhereParam.prompt_example_dict(),
            "result": "table.column",
        }


@dataclass
class CaseWhenExpr:
    """CASE expression for SELECT only."""

    branches: list[CaseWhenBranch] = field(default_factory=list)
    else_result: NormalizedExpr | None = None
    condition_scope: str = "where"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseWhenExpr:
        """Parse a full `CASE` from JSON (branches plus optional. `else_result`). Args: d: Mapping with `branches` list, optional `else_result`, and optional `condition_scope` (`"where"` or `"having"`; defaults to `"where"`). Returns: `CaseWhenExpr` with ordered branches and optional else expression."""
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
                else_result = CaseWhenBranch.string_result_expr(er)
        scope_raw = str(d.get("condition_scope", "where")).strip().lower()
        if scope_raw == "filter":
            scope_raw = "where"
        scope = scope_raw if scope_raw in ("where", "having") else "where"
        return CaseWhenExpr(branches=branches, else_result=else_result, condition_scope=scope)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize all branches and optional else clause.

        Returns:

            Dict with `branches` list; `else_result` included when set; `condition_scope`
            included only when it differs from the default `"where"`.
        """
        out: dict[str, Any] = {"branches": [b.to_dict() for b in self.branches]}
        if self.else_result is not None:
            if self.else_result.string_literal:
                out["else_literal_string"] = self.else_result.string_literal
            else:
                out["else_result"] = self.else_result.to_dict()
        if self.condition_scope and self.condition_scope != "where":
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
        "condition_scope": "where or having when branch predicates match that scope.",
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
                er = self.else_result.prompt_sql()
                if er:
                    out["else_result"] = er
        if self.condition_scope != "where":
            out["condition_scope"] = self.condition_scope
        return out

    @classmethod
    def prompt_example_dict(cls) -> dict[str, Any]:
        """Minimal CASE example for prompts."""
        return {
            "branches": [
                {
                    "condition": WhereParam.prompt_example_dict(),
                    "result": "table.column",
                }
            ],
            "else_result": None,
        }


_REGISTRY_WIN: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_WIN", default=())
_REGISTRY_CASE: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_CASE", default=())
_REGISTRY_WIN_FALLBACK: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_WIN_FALLBACK", default=())
_REGISTRY_CASE_FALLBACK: ContextVar[tuple[Any, ...]] = ContextVar("_REGISTRY_CASE_FALLBACK", default=())


@dataclass
class WindowRegistryStep:
    """Named window definition referenced by ``registry_ref`` on select expressions."""

    registry_id: str
    window_spec: WindowSpec = field(default_factory=lambda: WindowSpec(function="row_number"))
    PROMPT_FIELD_SPEC: ClassVar[dict[str, str]] = {
        "registry_id": "Window registry token such as w01 referenced from expressions.",
        "window_spec": (
            "Nested object whose keys are exactly those listed for WindowSpec in structural_json_keys (function, partition_by, order_by, optional argument, optional frame_kind, frame_start, frame_end, frame_start_offset, frame_end_offset)."
        ),
    }

    @staticmethod
    @contextmanager
    def render_scope(
        window_registry: list[WindowRegistryStep] | None, case_registry: list[CaseRegistryStep] | None
    ) -> Iterator[None]:
        """Bind window/case registry lists for the current thread while rendering expressions."""
        t_w = _REGISTRY_WIN.set(tuple(window_registry or ()))
        t_c = _REGISTRY_CASE.set(tuple(case_registry or ()))
        try:
            yield
        finally:
            _REGISTRY_WIN.reset(t_w)
            _REGISTRY_CASE.reset(t_c)

    @staticmethod
    @contextmanager
    def render_fallback_scope(
        window_registry: list[WindowRegistryStep] | None, case_registry: list[CaseRegistryStep] | None
    ) -> Iterator[None]:
        """Bind intent-wide registry fallbacks consulted when the active scope misses an id."""
        t_w = _REGISTRY_WIN_FALLBACK.set(tuple(window_registry or ()))
        t_c = _REGISTRY_CASE_FALLBACK.set(tuple(case_registry or ()))
        try:
            yield
        finally:
            _REGISTRY_WIN_FALLBACK.reset(t_w)
            _REGISTRY_CASE_FALLBACK.reset(t_c)

    @staticmethod
    def current_steps() -> tuple[WindowRegistryStep, ...]:
        """Return the window registry list bound by :meth:`render_scope`."""
        return _REGISTRY_WIN.get()

    @staticmethod
    def lookup(registry_id: str) -> WindowRegistryStep | None:
        """Resolve a window registry id from the active scope, then the intent-wide fallback."""
        rid = str(registry_id or "").strip()
        if not rid:
            return None
        for step in _REGISTRY_WIN.get():
            if getattr(step, "registry_id", None) == rid:
                return cast(WindowRegistryStep, step)
        for step in _REGISTRY_WIN_FALLBACK.get():
            if getattr(step, "registry_id", None) == rid:
                return cast(WindowRegistryStep, step)
        return None

    @staticmethod
    def operator_kind(steps: list[WindowRegistryStep]) -> WindowOperatorKind:
        """Map window registry rows to a coarse rank versus aggregate versus navigate class."""
        if not steps:
            return WindowOperatorKind.NONE
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
            return WindowOperatorKind.NAVIGATE
        if saw_rank:
            return WindowOperatorKind.RANK
        if saw_agg:
            return WindowOperatorKind.AGGREGATE
        return WindowOperatorKind.NONE

    @staticmethod
    def from_dict(d: dict[str, Any]) -> WindowRegistryStep:
        """
        Parse a window registry entry from JSON.

        Args:

            d: Mapping with ``registry_id`` and ``window_spec``. When ``base_expr`` is present and ``window_spec.argument`` is unset, ``base_expr`` is merged into ``argument``.

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
            migrated = NormalizedExpr.from_stored_json(base_payload)
            if migrated.signature_key and ws.argument is None:
                ws = replace(ws, argument=migrated)
        return WindowRegistryStep(registry_id=str(d.get("registry_id", "")).strip(), window_spec=ws)

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
                "partition_by": ["table.column"],
                "order_by": [{"expr": "table.other_column", "direction": "asc"}],
                "argument": "table.argument_column",
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
    def current_steps() -> tuple[CaseRegistryStep, ...]:
        """Return the case registry list bound by :meth:`WindowRegistryStep.render_scope`."""
        return _REGISTRY_CASE.get()

    @staticmethod
    def lookup(registry_id: str) -> CaseRegistryStep | None:
        """Resolve a case registry id from the active scope, then the intent-wide fallback."""
        rid = str(registry_id or "").strip()
        if not rid:
            return None
        for step in _REGISTRY_CASE.get():
            if getattr(step, "registry_id", None) == rid:
                return cast(CaseRegistryStep, step)
        for step in _REGISTRY_CASE_FALLBACK.get():
            if getattr(step, "registry_id", None) == rid:
                return cast(CaseRegistryStep, step)
        return None

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
                branches=[(CaseWhenBranch.from_dict(b) if isinstance(b, dict) else CaseWhenBranch()) for b in cw_raw]
            )
        elif isinstance(cw_raw, dict):
            cw = CaseWhenExpr.from_dict(cw_raw)
        else:
            cw = CaseWhenExpr()
        return CaseRegistryStep(
            registry_id=str(d.get("registry_id", "")).strip(), label=str(d.get("label", "")), case_when=cw
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
            "Its keys are exactly those listed for CaseWhenExpr in structural_json_keys (branches, optional else_result, optional condition_scope)."
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


@dataclass(frozen=True)
class TableDiff:
    """Per-table delta between a cached and a freshly-reflected ``SchemaGraph``. Entries are sorted tuples to keep equality + hashing deterministic in tests. ``retyped_columns`` records catalog type changes where the normalized ``value_type`` changes (profile must be refreshed). ``redeclared_columns`` holds pure ``data_type`` widenings (for example ``varchar(50)`` to ``text``) where ``value_type`` is unchanged; those updates merge metadata without clearing profiling samples. ``value_type_changed_columns`` mirrors the ``(column, old_vt, new_vt)`` entries implied by ``retyped_columns``. ``renamed_columns`` is populated by :func:`resolve_column_renames` after profile overlap matching; columns appearing here are removed from ``added_columns`` / ``dropped_columns``."""

    added_columns: tuple[str, ...] = ()
    dropped_columns: tuple[str, ...] = ()
    redeclared_columns: tuple[tuple[str, str, str], ...] = ()
    retyped_columns: tuple[tuple[str, str, str], ...] = ()
    value_type_changed_columns: tuple[tuple[str, str, str], ...] = ()
    renamed_columns: tuple[tuple[str, str], ...] = ()
    nullability_changed_columns: tuple[str, ...] = ()
    uniqueness_changed_columns: tuple[str, ...] = ()
    indexes_changed: bool = False
    view_definition_changed: bool = False
    fk_changed: bool = False
    pk_changed: bool = False

    @property
    def is_empty(self) -> bool:
        return (
            not self.added_columns
            and not self.dropped_columns
            and not self.redeclared_columns
            and not self.retyped_columns
            and not self.renamed_columns
            and not self.nullability_changed_columns
            and not self.uniqueness_changed_columns
            and not self.indexes_changed
            and not self.view_definition_changed
            and not self.fk_changed
            and not self.pk_changed
        )

    @property
    def needs_profile(self) -> bool:
        """True when applying this diff requires re-profiling the table. Pure-rename tables keep cached profiles. Adds and value-type retypes always need profiling; pure ``redeclared_columns`` (same ``value_type``) do not. Tables whose catalog PK or FK edge sets changed are pulled into :meth:`SchemaDiff.changed_table_names` so subset reprofiling refreshes statistics on those relations even when no columns were added or retyped."""
        return bool(self.added_columns or self.retyped_columns)


@dataclass
class SchemaDiff:
    """Whole-graph delta consumed by :func:`apply_diff` and downstream invalidation."""

    added_tables: tuple[str, ...] = ()
    dropped_tables: tuple[str, ...] = ()
    table_renames: tuple[tuple[str, str], ...] = ()
    per_table: dict[str, TableDiff] = field(default_factory=dict)
    cross_table_column_moves: tuple[tuple[str, str, str, str], ...] = ()
    dropped_user_fks: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    dropped_catalog_fks: list[tuple[str, str, str, str]] = field(default_factory=list)
    ported_user_fks: list[tuple[str, str, str, str, str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.added_tables
            and not self.dropped_tables
            and not self.table_renames
            and not self.per_table
            and not self.cross_table_column_moves
        )

    def implies_rename_remapping(self) -> bool:
        """True when template rename migration should treat this diff as a REMAP-tier rename."""
        if self.table_renames:
            return True
        return any(td.renamed_columns for td in self.per_table.values())

    def changed_table_names(self) -> set[str]:
        """Tables in the *new* graph that need subset profiling (adds, retypes, catalog PK/FK shape changes)."""
        out: set[str] = set(self.added_tables)
        for _old, new in self.table_renames:
            out.add(new)
        for tname, td in self.per_table.items():
            if td.needs_profile or td.pk_changed or td.fk_changed:
                out.add(tname)
        return out


@dataclass(frozen=True, slots=True)
class SheetGrid:
    """Raw tabular grid from one CSV file or one Excel worksheet."""

    source_path: Path
    sheet_name: str
    cells: tuple[tuple[str, ...], ...]
    merged_ranges: tuple[str, ...] = ()
    has_charts: bool = False
    has_images: bool = False
    excel_tables: tuple[str, ...] = ()
    excel_table_ranges: tuple[str, ...] = ()
    csv_single_column: bool = False
    header_row_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class CsvSourceSelection:
    """Per-upload interpretation choices for CSV/Excel sources."""

    sheet: str = ""
    header_row: int | None = None
    skip_rows: int = 0
    table_range: str = ""
    merge_regions: tuple[str, ...] = ()
    append_regions: tuple[str, ...] = ()
    column_transforms: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ScalarAffixColumnPlan:
    """Accepted scalar affix strip plan for one upload column."""

    duckdb_type: str
    unit_label: str
    has_percent: bool


@dataclass(frozen=True, slots=True)
class PreparedRelation:
    """One loadable table derived from a validated grid."""

    relation_name: str
    source_path: Path
    sheet_name: str
    original_table_label: str
    columns: tuple[str, ...]
    original_column_labels: tuple[str, ...]
    column_types: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    column_unit_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UploadIngestResult:
    """Outcome of materialising validated uploads into an existing embedded member."""

    relation_names: tuple[str, ...]
    report: DataQualityReport
    schema_diff: SchemaDiff | None = None


@dataclass(frozen=True, slots=True)
class FederationSourceLimits:
    """Per-source row cap and timeout overrides from a federation manifest."""

    row_cap: int | None = None
    timeout_ms: int | None = None
    semijoin_enabled: bool = True
    max_query_cost_rows: float | None = None
    max_query_cost_bytes: float | None = None
    profile_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class FederationSourceBinding:
    """One member-source binding in a federation manifest. Identifies a federated member by ``source_id`` (the registration key used in ``members`` maps, plan ``source_ids``, and composite ``table_namespace``). When deriving a binding from a live engine via :func:`~aetherdialect._federation_manifest.binding_from_member_engine`, a non-empty engine connection handle must equal that registration key. Args: source_id: Stable member identity / registration key. engine: Dialect/engine type label for the member (for example ``postgresql``, ``duckdb``). connection: Named connection handle within the engine config. Empty string (default) means unset; when set on a live engine it must match ``source_id``. context: Named engine-context label the member is bound under (default ``master``). role: Schema identity role for the member: ``owner`` or ``consumer`` (default ``consumer``). limits: Optional per-source row-cap / timeout / semijoin overrides. ``None`` (default) means use coordinator defaults. Raises: FederationConfigError: Not raised by this dataclass constructor itself; raised by :func:`~aetherdialect._federation_manifest.binding_from_member_engine` when the registration key disagrees with a non-empty engine connection handle (or the member role is not ``owner`` / ``consumer``)."""

    source_id: str
    engine: str
    connection: str = ""
    context: str = "master"
    role: SchemaRole = SchemaRole.CONSUMER
    limits: FederationSourceLimits | None = None
    session_timezone: str | None = None


@dataclass(frozen=True, slots=True)
class FederationCoordinatorConfig:
    """In-process DuckDB coordinator bounds and default per-source execution limits. The federation coordinator always materializes and combines member frames in DuckDB; engine selection is not configurable. Args: row_cap: Maximum total coordinator input/result rows. default_source_row_cap: Default per-member row cap when a source omits limits. default_source_timeout_ms: Default per-member timeout when a source omits limits. coordinator_timeout_ms: Wall-clock timeout for coordinator glue SQL execution. plan_timeout_ms: Wall- clock budget for an entire federated plan execution. semijoin_key_cap: Maximum distinct keys pushed as a semijoin filter. spill_row_threshold: Row count above which member frames spill to parquet. max_parallel_members: Maximum concurrent member query executions. total_input_byte_cap: Maximum total in-memory bytes across coordinator inputs."""

    row_cap: int = 500_000
    default_source_row_cap: int = 500_000
    default_source_timeout_ms: int = 30_000
    coordinator_timeout_ms: int = 30_000
    plan_timeout_ms: int = 300_000
    semijoin_key_cap: int = 50_000
    semijoin_key_distinct_floor: int = 2
    spill_row_threshold: int = 50_000
    max_parallel_members: int = 4
    total_input_byte_cap: int = 2_000_000_000


@dataclass(frozen=True, slots=True)
class FederationCrossSourceJoin:
    """Declared cross-source join edge (``left``/``right`` are ``table.column`` qualified)."""

    left: str
    right: str
    kind: str
    logical_key: str


@dataclass(frozen=True, slots=True)
class FederationTableAlias:
    """Explicit composite table name for one member physical table."""

    alias: str
    source: str
    table: str


@dataclass(frozen=True, slots=True)
class FederationManifest:
    """Authoritative federation deployment description."""

    federation_id: str
    sources: tuple[FederationSourceBinding, ...]
    table_namespace: dict[str, str]
    cross_source_joins: tuple[FederationCrossSourceJoin, ...]
    coordinator: FederationCoordinatorConfig
    aliases: tuple[FederationTableAlias, ...] = ()


@dataclass(frozen=True, slots=True)
class LogicalColumnMapping:
    """Operator-declared equivalence for one logical attribute across sources."""

    logical: str
    members: tuple[str, ...]
    role: str
    unify_in_graph: bool


@dataclass(frozen=True, slots=True)
class LogicalTableMember:
    """One physical table backing a logical federated table."""

    source: str
    table: str
    columns: dict[str, str]


@dataclass(frozen=True, slots=True)
class LogicalTableMapping:
    """Operator-declared equivalence for one logical table across sources."""

    logical: str
    members: tuple[LogicalTableMember, ...]
    semantics: Literal["union", "replica"]
    authoritative_source: str = ""


@dataclass(frozen=True, slots=True)
class FederationMappings:
    """Cross-source mapping sidecar replayed on composite rebuild."""

    version: str
    logical_columns: tuple[LogicalColumnMapping, ...] = ()
    logical_tables: tuple[LogicalTableMapping, ...] = ()


@dataclass(frozen=True, slots=True)
class FederationPlanTemplate:
    """Stored federation decomposition fingerprint keyed on the composite graph."""

    plan_id: str
    composite_schema_graph_id: str
    intent_key: str
    step_fingerprints: tuple[tuple[str, str], ...]
    combine_hash: str
    question: str = ""
    accepted_questions: tuple[str, ...] = ()
    format_version: str = "0.2.3"
    member_template_ids: tuple[tuple[str, str], ...] = ()
    residual_hash: str = ""
    join_feedback: tuple[str, ...] = ()
    manifest_hash: str = ""
    member_tuple_hash: str = ""


@dataclass(frozen=True, slots=True)
class FederationQualifiedRename:
    """One qualified ``table.column`` rename inside a federation migration map."""

    from_ref: str
    to_ref: str


@dataclass(frozen=True, slots=True)
class FederationMigrationMap:
    """Operator-authored federation migration consumed once at composite init."""

    version: int
    action: str
    qualified_column_renames: tuple[FederationQualifiedRename, ...] = ()
    namespace_renames: tuple[tuple[str, str], ...] = ()
    dropped_cross_source_joins: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class FederationMappingSuggestion:
    """Advisory cross-source mapping candidate; never applied without operator action."""

    logical: str
    members: tuple[str, ...]
    kind: str
    score: float
    role: str = "join_key"


@dataclass(frozen=True, slots=True)
class SourceRuntime:
    """Per-source bound dialect and artifact scope for federated execution."""

    source_id: str
    engine: str
    connection: str
    artifacts_dir: str
    dialect: Any
    sqlglot_dialect: str = ""
    native_connection: Any = None
    sqlalchemy_engine: Any = None


@dataclass(frozen=True, slots=True)
class CteIntent:
    """Planner-only natural-language description of one reusable intermediate aligned with a runtime CTE step."""

    name: str
    tables: tuple[str, ...] = ()
    select: str = ""
    where: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str | None = None
    window: str = ""
    case: str = ""


@dataclass(frozen=True, slots=True)
class LogicalIntent:
    """Interpret-stage natural-language plan consumed by ground and compose; not persisted and not structural IR."""

    tables: tuple[str, ...]
    select: str
    where: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str | None = None
    window: str = ""
    case: str = ""
    cte_steps: tuple[CteIntent, ...] = ()


class ColumnRole(Enum):
    """Column role for profiling and question simulation."""

    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    NUMERIC_CATEGORICAL = "numeric_categorical"
    NUMERIC_MEASURE = "numeric_measure"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    FREE_TEXT = "free_text"
    AUDIT = "audit"


class TableRole(Enum):
    """Table role for join constraint validation."""

    DIMENSION = "dimension"
    FACT = "fact"
    BRIDGE = "bridge"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkloadFamilySpec:
    """Declarative workload shape metadata for schema realization, sampling, and coverage keys."""

    family: WorkloadFamily
    preferred_complexity: ComplexityTier | None
    ranking_policy: str
    comparison_mode: str
    cardinality_regime: str
    required_table_roles: tuple[str, ...]


WORKLOAD_FAMILY_SPECS: dict[WorkloadFamily, WorkloadFamilySpec] = {
    WorkloadFamily.STATUS_REPORT: WorkloadFamilySpec(
        WorkloadFamily.STATUS_REPORT, ComplexityTier.MODERATE, "rank_none", "none", "small", ("fact",)
    ),
    WorkloadFamily.BREAKDOWN: WorkloadFamilySpec(
        WorkloadFamily.BREAKDOWN, ComplexityTier.MODERATE, "none", "categorical_slice", "medium", ("fact", "dimension")
    ),
    WorkloadFamily.LEADERBOARD: WorkloadFamilySpec(
        WorkloadFamily.LEADERBOARD, ComplexityTier.MODERATE, "top_k", "ordered_metric", "small", ("fact",)
    ),
    WorkloadFamily.TREND: WorkloadFamilySpec(
        WorkloadFamily.TREND, ComplexityTier.COMPLEX, "time_series", "temporal_sequence", "medium", ("fact",)
    ),
    WorkloadFamily.CHANGE_OVER_TIME: WorkloadFamilySpec(
        WorkloadFamily.CHANGE_OVER_TIME,
        ComplexityTier.COMPLEX,
        "period_over_period",
        "temporal_delta",
        "medium",
        ("fact",),
    ),
    WorkloadFamily.SHARE_OF_TOTAL: WorkloadFamilySpec(
        WorkloadFamily.SHARE_OF_TOTAL, ComplexityTier.COMPLEX, "ratio", "part_whole", "small", ("fact", "dimension")
    ),
    WorkloadFamily.SEGMENT_COMPARISON: WorkloadFamilySpec(
        WorkloadFamily.SEGMENT_COMPARISON,
        ComplexityTier.COMPLEX,
        "none",
        "cohort_contrast",
        "medium",
        ("fact", "dimension"),
    ),
    WorkloadFamily.THRESHOLD_EXCEPTION: WorkloadFamilySpec(
        WorkloadFamily.THRESHOLD_EXCEPTION,
        ComplexityTier.MODERATE,
        "exception_filter",
        "predicate_cutoff",
        "small",
        ("fact",),
    ),
    WorkloadFamily.EXTRACT: WorkloadFamilySpec(
        WorkloadFamily.EXTRACT, ComplexityTier.SIMPLE, "none", "none", "many", ("fact",)
    ),
    WorkloadFamily.LIFECYCLE_COHORT: WorkloadFamilySpec(
        WorkloadFamily.LIFECYCLE_COHORT,
        ComplexityTier.HIGHLY_COMPLEX,
        "cohort_retention",
        "lifecycle",
        "medium",
        ("fact", "dimension"),
    ),
    WorkloadFamily.EXPLORATION_FOLLOWUP: WorkloadFamilySpec(
        WorkloadFamily.EXPLORATION_FOLLOWUP, ComplexityTier.SIMPLE, "none", "ad_hoc", "many", ("fact",)
    ),
}


class InferenceTag(StrEnum):
    """Provenance tag for an :class:`FKEdge`. A catalog-declared edge is represented by ``None`` rather than a member of this enum so that presence-of-tag and identity-of- inferred-layer are reflected by a single attribute. Inherits ``str`` so members compare equal to their wire value and round-trip through JSON without custom encoding."""

    SUFFIX = "suffix"
    SELF = "self"
    COMPOSITE = "composite"
    SEMANTIC = "semantic"
    SEMANTIC_PROMOTED = "semantic_promoted"
    NOTES_STRUCTURAL = "notes_structural"
    USER_STRUCTURAL = "user_override_structural"
    USER_SEMANTIC = "user_override_semantic"
    CROSS_SOURCE = "cross_source"
    VIEW_LINEAGE = "view_lineage"

    @classmethod
    def coerce(cls, raw: object) -> InferenceTag | None:
        """Normalise raw cache or override input into :class:`InferenceTag` (``None`` for catalog)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in INFERENCE_TAG_VALUES:
            return cls(raw)
        raise ValueError(f"unknown FK inference_tag: {raw!r}")


class PkInferenceTag(StrEnum):
    """Provenance tag for an inferred or user-supplied primary key. Engine-reflected catalog keys use ``None`` (locked). SQL-file- declared keys use ``DDL`` (overridable). Inferred and user-supplied keys use the remaining members."""

    DDL = "ddl"
    IDENTITY = "identity"
    PROFILE = "profile"
    USER_OVERRIDE = "user_override"

    @classmethod
    def coerce(cls, raw: object) -> PkInferenceTag | None:
        """Normalise raw cache or override input into :class:`PkInferenceTag` (``None`` for catalog)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in PK_INFERENCE_TAG_VALUES:
            return cls(raw)
        raise ValueError(f"unknown pk_inference_tag: {raw!r}")


class RoleOwner(StrEnum):
    """Provenance for the writer that last set :attr:`ColumnMetadata.role`. The members are ordered by ascending precedence: a writer with strictly greater precedence may overwrite a role assigned by a lower-precedence owner, while equal-or-lower-precedence writers must skip the column. PK/FK coercion is treated as the highest authority because it is required for join correctness; user overrides win over LLM inference, which in turn wins over profile heuristics, which in turn wins over the default catalog fallback."""

    CATALOG = "catalog"
    PROFILE = "profile"
    LLM = "llm"
    BOOLEAN_COERCION = "boolean_coercion"
    USER_OVERRIDE = "user_override"
    PK_FK_COERCION = "pk_fk_coercion"

    @classmethod
    def coerce(cls, raw: object) -> RoleOwner | None:
        """Normalise raw cache or override input into :class:`RoleOwner` (``None`` when unset)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in ROLE_OWNER_VALUES:
            return cls(raw)
        raise ValueError(f"unknown role_owner: {raw!r}")

    @classmethod
    def can_overwrite(cls, current: RoleOwner | None, candidate: RoleOwner) -> bool:
        """Return whether a writer with provenance *candidate* may overwrite a role currently owned by *current*."""
        if current is None:
            return True
        return _ROLE_OWNER_PRECEDENCE[candidate] > _ROLE_OWNER_PRECEDENCE[current]


_ROLE_OWNER_PRECEDENCE: dict[RoleOwner, int] = {
    RoleOwner.CATALOG: 0,
    RoleOwner.PROFILE: 1,
    RoleOwner.LLM: 2,
    RoleOwner.BOOLEAN_COERCION: 3,
    RoleOwner.USER_OVERRIDE: 4,
    RoleOwner.PK_FK_COERCION: 5,
}


class DescriptionOwner(StrEnum):
    """Provenance for the writer that last set a description on a table or column. Members are ordered by ascending precedence; :meth:`set_on` enforces a strict-greater-precedence rule so a later writer can only overwrite an existing description when its provenance outranks the incumbent owner."""

    CATALOG = "catalog"
    PROFILE = "profile"
    NOTES = "notes"
    LLM_REFINEMENT = "llm_refinement"
    SPACE_NOTES = "space_notes"
    USER_OVERRIDE = "user_override"

    @classmethod
    def coerce(cls, raw: object) -> DescriptionOwner | None:
        """Normalise raw cache or override input into :class:`DescriptionOwner` (``None`` when unset)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) and raw in DESCRIPTION_OWNER_VALUES:
            return cls(raw)
        raise ValueError(f"unknown description_owner: {raw!r}")

    @classmethod
    def _rank(cls, owner: DescriptionOwner | None) -> int:
        if owner is None:
            return _DESCRIPTION_OWNER_PRECEDENCE[cls.CATALOG]
        if not isinstance(owner, cls):
            owner = cls(owner)
        return _DESCRIPTION_OWNER_PRECEDENCE[owner]

    @classmethod
    def resolve(
        cls,
        *candidates: tuple[str | None, DescriptionOwner | None],
    ) -> tuple[str, DescriptionOwner | None]:
        """Resolve simultaneous description candidates using owner precedence."""
        nonempty: list[tuple[str, DescriptionOwner | None]] = []
        for text, owner in candidates:
            cleaned = str(text or "").strip()
            if not cleaned:
                continue
            coerced = owner
            if coerced is not None and not isinstance(coerced, cls):
                coerced = cls(coerced)
            nonempty.append((cleaned, coerced))
        if not nonempty:
            return "", None
        max_rank = max(cls._rank(owner) for _, owner in nonempty)
        tier = [(text, owner) for text, owner in nonempty if cls._rank(owner) == max_rank]
        distinct_texts = sorted({text for text, _ in tier})
        if len(distinct_texts) == 1:
            winner_owner = next(owner for text, owner in tier if text == distinct_texts[0])
            return distinct_texts[0], winner_owner
        return "", None

    @classmethod
    def set_on(cls, target: Any, text: str | None, owner: DescriptionOwner) -> bool:
        """Single writer for ``description`` on tables and columns."""
        if text is None:
            return False
        current_owner = getattr(target, "description_owner", None)
        if current_owner is not None:
            if not isinstance(current_owner, cls):
                current_owner = cls(current_owner)
            if _DESCRIPTION_OWNER_PRECEDENCE[owner] < _DESCRIPTION_OWNER_PRECEDENCE[current_owner]:
                return False
        cur_desc = (getattr(target, "description", None) or "").strip()
        new_desc = str(text).strip()
        if cur_desc == new_desc and current_owner == owner:
            return False
        target.description = new_desc
        target.description_owner = owner
        return True


_DESCRIPTION_OWNER_PRECEDENCE: dict[DescriptionOwner, int] = {
    DescriptionOwner.CATALOG: 0,
    DescriptionOwner.PROFILE: 1,
    DescriptionOwner.NOTES: 2,
    DescriptionOwner.LLM_REFINEMENT: 3,
    DescriptionOwner.SPACE_NOTES: 4,
    DescriptionOwner.USER_OVERRIDE: 5,
}


@dataclass
class ExpansionMetadata:
    """Metadata for intent expansion operations."""

    operator: str
    parent_intent_id: str | None = None
    depth: int = 0
    expansion_path: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExpansionMetadata:
        """
        Create `ExpansionMetadata` from a dictionary.

        Args:

            d: Dictionary with keys matching `ExpansionMetadata` fields.

        Returns:

            Populated `ExpansionMetadata` instance.
        """
        return ExpansionMetadata(
            operator=d.get("operator", ""),
            parent_intent_id=d.get("parent_intent_id"),
            depth=d.get("depth", 0),
            expansion_path=d.get("expansion_path", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize expansion metadata to a plain dict.

        Returns:

            `asdict` of all fields.
        """
        return asdict(self)


@dataclass
class CteOutputColumnMeta:
    """Metadata for a CTE output column, including source, role, and aggregation info."""

    source: str
    agg_func: str = ""
    role: str | None = None
    filterable: bool = True
    aggregatable: bool = True
    data_type: str = "unknown"
    value_type: str = ""
    groupable: bool = True
    valid_where_ops: list[str] = field(default_factory=list)
    valid_aggregations: list[str] = field(default_factory=list)
    valid_having_ops: list[str] = field(default_factory=list)
    sensitivity: str | None = None
    lineage_phys_table: str | None = None
    lineage_phys_column: str | None = None
    lineage_inherits_pk: bool = False
    lineage_fk_to_table: str | None = None
    lineage_fk_to_column: str | None = None
    semantic_distinct_values: list[str] = field(default_factory=list)
    semantic_join_neighbors: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set `value_type` from `data_type` when `value_type` is empty."""
        if not self.value_type and self.data_type:
            self.value_type = ColumnTypeSemantics.data_type_to_value_type(self.data_type)

    @property
    def is_selectable(self) -> bool:
        """Whether the CTE output column may be projected; derived from ``sensitivity`` (hidden tags suppress selection)."""
        if self.sensitivity is None:
            return True
        return str(self.sensitivity).strip().lower() not in HIDDEN_SENSITIVITIES

    def get_valid_where_ops(self) -> list[str]:
        """
        Filter operators allowed on this CTE output column.

        Returns:

            Stored ops plus null checks, or defaults when `filterable`, else null checks only.
        """
        null_ops = ["is null", "is not null"]
        if self.valid_where_ops:
            return sorted(set(self.valid_where_ops + null_ops))
        if self.filterable:
            return [
                "=",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "in",
                "not in",
                "is null",
                "is not null",
            ]
        return null_ops

    def get_valid_aggregations(self) -> set[str]:
        """
        Aggregation names allowed on this CTE output column.

        Returns:

            Lowercased `valid_aggregations`, or defaults by `aggregatable` flag.
        """
        if self.valid_aggregations:
            return set(agg.lower() for agg in self.valid_aggregations)
        if not self.role:
            return set()
        rk = self.role.upper()
        if rk in ROLE_ALLOWED_AGGREGATIONS:
            return {a.lower() for a in ROLE_ALLOWED_AGGREGATIONS[rk]}
        if self.aggregatable:
            return {"count", "sum", "avg", "min", "max", "stddev", "variance", "median", "string_agg"}
        return {"count"}

    def get_valid_having_ops(self) -> list[str]:
        """
        `HAVING` operators allowed on this CTE output column.

        Returns:

            Stored list, comparison ops when `aggregatable`, or an empty list.
        """
        if self.valid_having_ops:
            return list(self.valid_having_ops)
        if self.aggregatable:
            return ["=", "!=", "<", "<=", ">", ">="]
        return []

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CteOutputColumnMeta:
        """
        Create `CteOutputColumnMeta` from a dictionary.

        Args:

            d: Dictionary with keys matching `CteOutputColumnMeta` fields.

        Returns:

            Populated `CteOutputColumnMeta` instance.
        """
        return CteOutputColumnMeta(
            source=d.get("source", "passthrough"),
            agg_func=d.get("agg_func", ""),
            role=d.get("role"),
            filterable=d.get("filterable", True),
            aggregatable=d.get("aggregatable", True),
            data_type=d.get("data_type", "unknown"),
            value_type=d.get("value_type", ""),
            groupable=d.get("groupable", True),
            valid_where_ops=d.get("valid_where_ops", []),
            valid_aggregations=d.get("valid_aggregations", []),
            valid_having_ops=d.get("valid_having_ops", []),
            sensitivity=d.get("sensitivity"),
            lineage_phys_table=d.get("lineage_phys_table"),
            lineage_phys_column=d.get("lineage_phys_column"),
            lineage_inherits_pk=d.get("lineage_inherits_pk", False),
            lineage_fk_to_table=d.get("lineage_fk_to_table"),
            lineage_fk_to_column=d.get("lineage_fk_to_column"),
            semantic_distinct_values=d.get("semantic_distinct_values", []),
            semantic_join_neighbors=[
                (str(x[0]), str(x[1]))
                for x in (d.get("semantic_join_neighbors") or [])
                if isinstance(x, (list, tuple)) and len(x) == 2
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize CTE column meta to a plain dict.

        Returns:

            Plain dict including JSON-friendly neighbor pairs.
        """
        d = asdict(self)
        d["is_selectable"] = self.is_selectable
        d["semantic_join_neighbors"] = [list(p) for p in self.semantic_join_neighbors]
        return d


@dataclass(frozen=True, slots=True)
class PersistedFederationInspection:
    """Declaration and roster loaded from a persisted ``fed_<id>`` artifact tree."""

    federation_id: str
    federation_dir: str
    manifest: FederationManifest
    mappings: FederationMappings
    roster: tuple[tuple[str, str, str, str], ...]
