"""CSV/Excel upload parsing, validation, and identifier resolution."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import os
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any

from ._config import EngineLimits, PolicyConfig
from ._constants import (
    BARE_SCALAR_NUMBER_RE,
    CSV_IDENTIFIER_NAMING_SCHEMA,
    CSV_IDENTIFIER_NAMING_SYSTEM,
    DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW,
    DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE,
    DATA_QUALITY_EXCEL_ERROR_TOKENS,
    DATA_QUALITY_FOOTER_LABEL_PREFIXES,
    DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH,
    DATA_QUALITY_ISSUE_APPENDABLE_REGIONS,
    DATA_QUALITY_ISSUE_BLANK_HEADER,
    DATA_QUALITY_ISSUE_BLANK_ROW,
    DATA_QUALITY_ISSUE_DUPLICATE_HEADER,
    DATA_QUALITY_ISSUE_DUPLICATE_RELATION,
    DATA_QUALITY_ISSUE_EMBEDDED_OBJECT,
    DATA_QUALITY_ISSUE_EMPTY_COLUMN,
    DATA_QUALITY_ISSUE_EMPTY_FILE,
    DATA_QUALITY_ISSUE_EMPTY_SHEET,
    DATA_QUALITY_ISSUE_EXCEL_ERROR,
    DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW,
    DATA_QUALITY_ISSUE_FORMULA_CELL,
    DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE,
    DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE,
    DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH,
    DATA_QUALITY_ISSUE_MERGEABLE_REGIONS,
    DATA_QUALITY_ISSUE_MERGED_CELLS,
    DATA_QUALITY_ISSUE_MERGED_METADATA,
    DATA_QUALITY_ISSUE_MIXED_TYPES,
    DATA_QUALITY_ISSUE_MULTIPLE_TABLES,
    DATA_QUALITY_ISSUE_NULL_TOKEN,
    DATA_QUALITY_ISSUE_NUMBER_AS_TEXT,
    DATA_QUALITY_ISSUE_OVERFULL_ROW,
    DATA_QUALITY_ISSUE_RAGGED_ROW,
    DATA_QUALITY_ISSUE_REPEATED_HEADER,
    DATA_QUALITY_ISSUE_SECTION_HEADING,
    DATA_QUALITY_ISSUE_SEVERITY,
    DATA_QUALITY_ISSUE_SINGLE_COLUMN,
    DATA_QUALITY_ISSUE_TOTAL_ROW,
    DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE,
    DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT,
    DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED,
    DATA_QUALITY_MOJIBAKE_REPLACEMENTS,
    DATA_QUALITY_NULL_TOKENS,
    DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_SQL_RESERVED_WORDS,
    DATA_QUALITY_TOTAL_PREFIXES,
    DATA_QUALITY_ZERO_WIDTH_CHARS,
    DIAGNOSTIC_CODE_DATA_QUALITY_ADVISORY,
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
    DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ,
    DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING,
    DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_APPLIED,
    DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED,
    DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED,
    ISO_DATE_RE,
    ISO_TIMESTAMP_RE,
    REVIEW_GATED_UPLOAD_COLUMN_TRANSFORMS,
    UPLOAD_BAND_VALUE_MAP_MAX_DISTINCT,
    UPLOAD_COLUMN_TRANSFORM_IDS,
    UPLOAD_COLUMN_TRANSFORMS_SCHEMA,
    UPLOAD_COLUMN_TRANSFORMS_SYSTEM,
    UPLOAD_CURRENCY_AFFIX_TOKENS,
    UPLOAD_INTERPRET_MAX_ROWS,
    UPLOAD_INTERPRET_SCHEMA,
    UPLOAD_INTERPRET_SYSTEM,
    UPLOAD_SAMPLE_MAX_ROWS,
    UPLOAD_SCALAR_AFFIX_TOKENS_SORTED,
    UPLOAD_SCALAR_BAND_PATTERNS,
    UPLOAD_SUMMARY_SCHEMA,
    UPLOAD_SUMMARY_SYSTEM,
)
from ._contracts_base import (
    ConfigError,
    DataQualityReport,
    Diagnostic,
    DiagnosticSeverity,
    UploadColumnTransformId,
)
from ._contracts_schema import (
    CsvSourceSelection,
    PreparedRelation,
    ScalarAffixColumnPlan,
    SchemaGraph,
    SheetGrid,
)
from ._core_utils import active_engine_limits, debug, notify, require_driver, stable_json
from ._llm_provider import LLMProvider

_XLSX_DECLARED_COLUMN_TYPES: dict[tuple[str, str], tuple[str, ...]] = {}


def parse_source_selections(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, CsvSourceSelection]:
    """Convert config JSON objects into :class:`CsvSourceSelection` records."""
    out: dict[str, CsvSourceSelection] = {}
    for name, body in raw.items():
        header_raw = body.get("header_row")
        header_row = int(header_raw) if header_raw is not None and str(header_raw).strip() else None
        merge_raw = body.get("merge_regions", ())
        merge_regions: tuple[str, ...]
        if isinstance(merge_raw, str):
            merge_regions = (merge_raw.strip(),) if merge_raw.strip() else ()
        elif isinstance(merge_raw, Sequence) and not isinstance(merge_raw, (str, bytes)):
            merge_regions = tuple(str(item).strip() for item in merge_raw if str(item).strip())
        else:
            merge_regions = ()
        append_raw = body.get("append_regions", ())
        append_regions: tuple[str, ...]
        if isinstance(append_raw, str):
            append_regions = (append_raw.strip(),) if append_raw.strip() else ()
        elif isinstance(append_raw, Sequence) and not isinstance(append_raw, (str, bytes)):
            append_regions = tuple(str(item).strip() for item in append_raw if str(item).strip())
        else:
            append_regions = ()
        transforms_raw = body.get("column_transforms", ())
        column_transforms: list[Mapping[str, Any]] = []
        if isinstance(transforms_raw, Sequence) and not isinstance(transforms_raw, (str, bytes)):
            for item in transforms_raw:
                if isinstance(item, Mapping):
                    column_transforms.append(dict(item))
        out[str(name)] = CsvSourceSelection(
            sheet=str(body.get("sheet", "") or ""),
            header_row=header_row,
            skip_rows=int(body.get("skip_rows", 0) or 0),
            table_range=str(body.get("table_range", "") or ""),
            merge_regions=merge_regions,
            append_regions=append_regions,
            column_transforms=tuple(column_transforms),
        )
    return out


def load_source_grids(path: Path, *, selection: CsvSourceSelection | None = None) -> list[SheetGrid]:
    """Load every sheet from a CSV or Excel source into raw grids."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        grids = [_load_csv_grid(path)]
    elif suffix == ".xlsx":
        grids = _load_xlsx_grids(path)
    else:
        raise ValueError(f"unsupported upload type: {path}")
    if selection and selection.sheet:
        grids = [grid for grid in grids if grid.sheet_name == selection.sheet]
    return [apply_source_selection(grid, selection) for grid in grids]


def apply_source_selection(grid: SheetGrid, selection: CsvSourceSelection | None) -> SheetGrid:
    """Crop *grid* using a caller-confirmed sheet/header/range selection."""
    if selection is None:
        return grid
    cells = list(grid.cells)
    range_start_row = 1
    if selection.table_range:
        match = re.match(r"^([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$", selection.table_range.strip())
        if match:
            range_start_row = int(match.group(2))
        cells = _crop_table_range(cells, selection.table_range)
    header_row = selection.header_row
    header_row_confirmed = header_row is not None
    if header_row is not None and header_row > range_start_row:
        offset = header_row - range_start_row
        if 0 < offset < len(cells):
            cells = cells[offset:]
    elif selection.skip_rows > 0 and selection.skip_rows < len(cells):
        cells = cells[selection.skip_rows :]
    return SheetGrid(
        source_path=grid.source_path,
        sheet_name=grid.sheet_name,
        cells=tuple(cells),
        merged_ranges=grid.merged_ranges,
        has_charts=grid.has_charts,
        has_images=grid.has_images,
        excel_tables=grid.excel_tables,
        excel_table_ranges=grid.excel_table_ranges,
        csv_single_column=grid.csv_single_column,
        header_row_confirmed=header_row_confirmed,
    )


def normalize_grid(grid: SheetGrid, *, normalize_cell_newlines: bool = True) -> SheetGrid:
    """Apply auto-read normalization to every cell in *grid*."""
    rows: list[tuple[str, ...]] = []
    for row in grid.cells:
        normalized_row = tuple(
            _normalize_cell_text(cell, normalize_cell_newlines=normalize_cell_newlines) for cell in row
        )
        rows.append(normalized_row)
    return SheetGrid(
        source_path=grid.source_path,
        sheet_name=grid.sheet_name,
        cells=tuple(rows),
        merged_ranges=grid.merged_ranges,
        has_charts=grid.has_charts,
        has_images=grid.has_images,
        excel_tables=grid.excel_tables,
        excel_table_ranges=grid.excel_table_ranges,
        csv_single_column=grid.csv_single_column,
        header_row_confirmed=grid.header_row_confirmed,
    )


def detect_grid_issues(grid: SheetGrid) -> list[Diagnostic]:
    """Return syntax-only data-quality findings for one grid."""
    issues: list[Diagnostic] = []
    location_prefix = _grid_location_prefix(grid)
    if not grid.cells:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_EMPTY_FILE,
                level="error",
                message="The file or sheet is empty.",
                location=location_prefix,
                blocking=True,
            )
        )
        return issues
    if grid.has_charts or grid.has_images:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_EMBEDDED_OBJECT,
                level="error",
                message="Embedded charts or images were detected.",
                location=location_prefix,
                blocking=True,
            )
        )
    if grid.csv_single_column:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_SINGLE_COLUMN,
                level="warning",
                message="The CSV appears to contain only one column.",
                location=location_prefix,
                blocking=False,
            )
        )
    non_empty_rows = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if not non_empty_rows:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_EMPTY_SHEET,
                level="error",
                message="The sheet contains only blank rows.",
                location=location_prefix,
                blocking=True,
            )
        )
        return issues
    width = max(len(grid.cells[r]) for r in non_empty_rows)
    first_populated = non_empty_rows[0]
    inferred_header = infer_header_row(grid)
    header_idx = first_populated
    if inferred_header is not None:
        header_idx = inferred_header - 1
    canonical_header = _pad_row(grid.cells[first_populated], width)
    repeated_rows = [
        row_idx
        for row_idx in non_empty_rows
        if row_idx != first_populated and _rows_equal_cells(_pad_row(grid.cells[row_idx], width), canonical_header)
    ]
    if repeated_rows:
        row_loc = f"{location_prefix}!A{repeated_rows[0] + 1}:{_col_letter(width - 1)}{repeated_rows[0] + 1}"
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_REPEATED_HEADER,
                level="error",
                message=(
                    "Remove the repeated header rows inside the data — this sheet looks like a "
                    "stacked report, not one plain table."
                ),
                location=row_loc,
                blocking=True,
            )
        )
    issues.extend(_merged_cell_issues(grid, header_idx + 1, location_prefix))
    if not grid.header_row_confirmed and inferred_header is not None and inferred_header - 1 != first_populated:
        header_loc = f"{location_prefix}!A{inferred_header}:{_col_letter(width - 1)}{inferred_header}"
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE,
                level="warning",
                message=(
                    f"Confirm the header row (looks like row {inferred_header}, {header_loc}). "
                    "Remove the rows above it or set CSV_SOURCE_SELECTIONS."
                ),
                location=f"{location_prefix}!A{first_populated + 1}:{_col_letter(width - 1)}{first_populated + 1}",
                blocking=False,
                extra_details=(
                    (DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW, str(inferred_header)),
                    (
                        DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE,
                        f"A{inferred_header}:{_col_letter(width - 1)}{len(grid.cells)}",
                    ),
                ),
            )
        )
    elif not grid.header_row_confirmed and first_populated > 0:
        header_row_num = first_populated + 1
        header_loc = f"{location_prefix}!A{header_row_num}:{_col_letter(width - 1)}{header_row_num}"
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE,
                level="warning",
                message=(
                    f"Confirm the header row (looks like row {header_row_num}, {header_loc}). "
                    "Remove the rows above it or set CSV_SOURCE_SELECTIONS."
                ),
                location=f"{location_prefix}!A{first_populated + 1}:{_col_letter(width - 1)}{first_populated + 1}",
                blocking=False,
                extra_details=(
                    (DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW, str(first_populated + 1)),
                    (
                        DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE,
                        f"A{first_populated + 1}:{_col_letter(width - 1)}{len(grid.cells)}",
                    ),
                ),
            )
        )
    padded = [_pad_row(grid.cells[r], width) for r in range(header_idx, len(grid.cells))]
    raw_slice = grid.cells[header_idx:]
    row_pairs = [
        (raw_slice[idx], padded[idx]) for idx in range(len(padded)) if any(cell.strip() for cell in padded[idx])
    ]
    if not row_pairs:
        return issues
    header = row_pairs[0][1]
    header_raw = row_pairs[0][0]
    named_end = _named_header_end(header)
    for col_idx, label in enumerate(header[: named_end + 1]):
        if not label.strip():
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_BLANK_HEADER,
                    level="error",
                    message=f"Add a header name in column {_col_letter(col_idx)} (cell {_col_letter(col_idx)}{header_idx + 1}).",
                    location=f"{location_prefix}!{_col_letter(col_idx)}{header_idx + 1}",
                    blocking=True,
                )
            )
    seen: set[str] = set()
    for col_idx, label in enumerate(header[: named_end + 1]):
        key = label.strip().lower()
        if not key:
            continue
        if key in seen:
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_DUPLICATE_HEADER,
                    level="error",
                    message=f"Rename column {label!r} at {_col_letter(col_idx)}{header_idx + 1} — this name is already used.",
                    location=f"{location_prefix}!{_col_letter(col_idx)}{header_idx + 1}",
                    blocking=True,
                )
            )
        seen.add(key)
    table_regions = _detect_side_by_side_regions(padded[0])
    if len(table_regions) > 1:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_MULTIPLE_TABLES,
                level="error",
                message="Multiple side-by-side tables were detected in one sheet.",
                location=location_prefix,
                blocking=False,
                review=True,
            )
        )
    data_row_pairs = row_pairs[1:]
    footer_indices = _footer_note_indices(data_row_pairs, header)
    for row_offset, (raw_row, row) in enumerate(data_row_pairs, start=header_idx + 2):
        pair_idx = row_offset - header_idx - 2
        if not any(cell.strip() for cell in row):
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_BLANK_ROW,
                    level="error",
                    message=f"Remove the blank row at A{row_offset}:{_col_letter(width - 1)}{row_offset}.",
                    location=f"{location_prefix}!A{row_offset}:{_col_letter(width - 1)}{row_offset}",
                    blocking=True,
                )
            )
            continue
        if pair_idx in footer_indices:
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW,
                    level="warning",
                    message="A trailing note or footer row appears below the data.",
                    location=f"{location_prefix}!A{row_offset}",
                    blocking=False,
                )
            )
            continue
        if _is_structurally_ragged_row(raw_row, header_raw):
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_RAGGED_ROW,
                    level="error",
                    message=f"Add the missing values in row {row_offset} ({location_prefix}!A{row_offset}).",
                    location=f"{location_prefix}!A{row_offset}:{_col_letter(width - 1)}{row_offset}",
                    blocking=True,
                )
            )
        if _is_structurally_overfull_row(raw_row, header_raw):
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_OVERFULL_ROW,
                    level="error",
                    message=f"Remove the extra values in row {row_offset} ({location_prefix}!A{row_offset}).",
                    location=f"{location_prefix}!A{row_offset}:{_col_letter(width - 1)}{row_offset}",
                    blocking=True,
                )
            )
        if _looks_total_row(row):
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_TOTAL_ROW,
                    level="error",
                    message=f"Remove the total or subtotal row at A{row_offset}:{_col_letter(width - 1)}{row_offset}.",
                    location=f"{location_prefix}!A{row_offset}:{_col_letter(width - 1)}{row_offset}",
                    blocking=True,
                )
            )
    section_rows = 0
    for pair_idx, (_raw, row) in enumerate(data_row_pairs):
        if pair_idx in footer_indices or not any(cell.strip() for cell in row):
            continue
        if _looks_section_heading_row(row):
            section_rows += 1
    if section_rows >= 2:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_SECTION_HEADING,
                level="error",
                message=(
                    "Remove the section title rows inside the table — this sheet looks like a "
                    "formatted report, not one plain table."
                ),
                location=location_prefix,
                blocking=True,
            )
        )
    col_count = _named_header_end(header) + 1
    last_row_num = header_idx + len(row_pairs)
    for col_idx in range(col_count):
        samples = [
            row[col_idx]
            for pair_idx, (_raw, row) in enumerate(data_row_pairs)
            if pair_idx not in footer_indices and col_idx < len(row) and row[col_idx].strip()
        ]
        issues.extend(
            _column_value_issues(
                grid,
                header_idx,
                col_idx,
                header[col_idx],
                samples,
                last_row_num=last_row_num,
            )
        )
    return _collapse_issues(issues)


def apply_structural_fixes(grid: SheetGrid) -> tuple[SheetGrid, list[Diagnostic]]:
    """Auto-correct common upload layouts before validation."""
    corrected: list[Diagnostic] = []
    working = grid
    if not working.cells:
        return working, corrected

    trimmed, trim_issues = _trim_blank_border_grid(working)
    if trimmed is not working:
        corrected.extend(trim_issues)
        working = trimmed
    if not working.cells:
        return working, corrected

    transposed = _transpose_key_value_grid(working)
    if transposed is not working:
        corrected.append(
            _make_issue(
                code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
                level="info",
                message="Transposed the key-value layout into a header row.",
                location=_grid_location_prefix(working),
                blocking=False,
                diagnostic_code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
            )
        )
        working = transposed

    if not working.header_row_confirmed:
        skipped, skip_issues = _skip_rows_above_inferred_header(working)
        if skipped is not working:
            corrected.extend(skip_issues)
            working = skipped

    padded = _pad_grid_to_width(working)
    if padded is not working:
        corrected.append(
            _make_issue(
                code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
                level="info",
                message="Padded short rows to the header width.",
                location=_grid_location_prefix(working),
                blocking=False,
                diagnostic_code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
            )
        )
        working = padded

    return working, corrected


def grid_to_relation(
    grid: SheetGrid,
    *,
    relation_name: str,
    column_names: Sequence[str],
    original_column_labels: Sequence[str],
) -> PreparedRelation:
    """Project a validated grid into column names, inferred types, and row dicts."""
    if not grid.cells:
        raise ValueError("cannot build relation from empty grid")
    data_rows = grid.cells[1:]
    width = len(column_names)
    types: list[str] = []
    samples_by_col: list[list[str]] = [[] for _ in range(width)]
    for row in data_rows:
        padded = _pad_row(row, width)
        for col_idx in range(width):
            value = padded[col_idx] if col_idx < len(padded) else ""
            if value.strip():
                samples_by_col[col_idx].append(value)
    affix_plans: list[ScalarAffixColumnPlan | None] = []
    for col_idx in range(width):
        cache_key = (str(grid.source_path.resolve()), grid.sheet_name)
        declared = _XLSX_DECLARED_COLUMN_TYPES.get(cache_key)
        if declared is not None and col_idx < len(declared):
            affix_plans.append(None)
        else:
            affix_plans.append(_plan_scalar_affix_column(samples_by_col[col_idx]))
    for col_idx in range(width):
        affix_plan = affix_plans[col_idx]
        cache_key = (str(grid.source_path.resolve()), grid.sheet_name)
        declared = _XLSX_DECLARED_COLUMN_TYPES.get(cache_key)
        if declared is not None and col_idx < len(declared):
            types.append(declared[col_idx])
        elif affix_plan is not None:
            types.append(affix_plan.duckdb_type)
        else:
            types.append(infer_duckdb_column_type(samples_by_col[col_idx]))
    unit_labels: list[str] = []
    for col_idx in range(width):
        affix_plan = affix_plans[col_idx]
        if affix_plan is not None:
            unit_labels.append(affix_plan.unit_label)
            notify(
                f"Stripped unit affix {affix_plan.unit_label!r} from column {column_names[col_idx]!r}.",
                stage="data_quality",
                code=DIAGNOSTIC_CODE_UPLOAD_UNIT_AFFIX_STRIPPED,
                level="info",
                details=(
                    ("column", column_names[col_idx]),
                    ("unit", affix_plan.unit_label),
                    ("location", _grid_location_prefix(grid)),
                ),
            )
        else:
            unit_labels.append("")
    rows: list[dict[str, str]] = []
    for row in data_rows:
        padded = _pad_row(row, width)
        row_map: dict[str, str] = {}
        for col_idx in range(width):
            raw = padded[col_idx] if col_idx < len(padded) else ""
            affix_plan = affix_plans[col_idx]
            if affix_plan is not None and raw.strip():
                row_map[column_names[col_idx]] = _strip_scalar_affix_cell(raw, affix_plan)
            else:
                row_map[column_names[col_idx]] = raw
        if any(str(v).strip() for v in row_map.values()):
            rows.append(row_map)
    original_table_label = _table_original_label(grid)
    return PreparedRelation(
        relation_name=relation_name,
        source_path=grid.source_path,
        sheet_name=grid.sheet_name,
        original_table_label=original_table_label,
        columns=tuple(column_names),
        original_column_labels=tuple(original_column_labels),
        column_types=tuple(types),
        column_unit_labels=tuple(unit_labels),
        rows=tuple(rows),
    )


def validate_upload_sources(
    paths: Sequence[Path],
    *,
    log_sink: Callable[[str], None] | None = None,
    apply_auto_correct: bool = True,
    normalize_cell_newlines: bool = True,
    source_selections: Mapping[str, CsvSourceSelection] | None = None,
) -> DataQualityReport:
    """Validate CSV/Excel uploads and optionally apply auto-correct reshaping."""
    sink = log_sink if log_sink is not None else notify
    all_issues: list[Diagnostic] = []
    relation_names: dict[str, str] = {}
    selections = source_selections or {}
    for path in paths:
        selection = selections.get(path.name)
        try:
            grids = load_source_grids(path, selection=selection)
        except ValueError as exc:
            message = str(exc).lower()
            if "encrypt" in message or "password" in message:
                code = DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED
            elif "unsupported" in message:
                code = DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE
            else:
                code = DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT
            all_issues.append(
                _make_issue(
                    code=code,
                    level="error",
                    message=str(exc),
                    location=path.name,
                    blocking=True,
                )
            )
            continue
        multi_sheet = _workbook_has_multiple_sheets(path)
        for grid in grids:
            normalized = normalize_grid(grid, normalize_cell_newlines=normalize_cell_newlines)
            all_issues.append(
                _make_issue(
                    code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ,
                    level="info",
                    message="Auto-read applied encoding and whitespace normalization.",
                    location=_grid_location_prefix(normalized),
                    blocking=False,
                    diagnostic_code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_READ,
                )
            )
            working = normalized
            if apply_auto_correct:
                working, corrected = apply_structural_fixes(working)
                all_issues.extend(corrected)
            region_iter, region_issues = _regions_for_grid(working, selection)
            all_issues.extend(region_issues)
            if not (selection and (selection.merge_regions or selection.append_regions)):
                advisory = _mergeable_regions_advisory(working)
                if advisory is not None:
                    all_issues.append(advisory)
                append_advisory = _appendable_regions_advisory(working)
                if append_advisory is not None:
                    all_issues.append(append_advisory)
            for region_grid, region_index, region_count in region_iter:
                all_issues.extend(detect_grid_issues(region_grid))
                relation = relation_name_for_grid(
                    region_grid,
                    multi_sheet_workbook=multi_sheet,
                    region_index=region_index,
                    region_count=region_count,
                )
                prior = relation_names.get(relation.lower())
                if prior is not None:
                    all_issues.append(
                        _make_issue(
                            code=DATA_QUALITY_ISSUE_DUPLICATE_RELATION,
                            level="error",
                            message=f"Rename table {relation!r} — this name is already used.",
                            location=_grid_location_prefix(region_grid),
                            blocking=True,
                        )
                    )
                relation_names[relation.lower()] = relation
    blocking = [issue for issue in all_issues if _issue_is_blocking(issue)]
    ok = not blocking and not any(_issue_requires_review(issue) for issue in all_issues)
    suggested = _suggested_selections_from_issues(paths, all_issues)
    if PolicyConfig.TABULAR_LLM_ASSIST:
        suggested = _enrich_suggested_selections_with_column_transforms(
            paths,
            selections,
            suggested,
        )
    narrative = _build_narrative(paths, all_issues, ok=ok)
    if PolicyConfig.TABULAR_LLM_ASSIST and ok:
        narrative = _llm_upload_summary_narrative(
            paths,
            all_issues,
            suggested,
            ok=ok,
            fallback=narrative,
        )
    for issue in all_issues:
        notify(
            issue.message,
            stage="data_quality",
            code=issue.code,
            level=issue.level,
            details=issue.details,
        )
    sink(narrative)
    return DataQualityReport(
        ok=ok,
        issues=tuple(all_issues),
        narrative=narrative,
        suggested_selections=suggested,
    )


def relation_name_for_grid(
    grid: SheetGrid,
    *,
    multi_sheet_workbook: bool,
    region_index: int = 1,
    region_count: int = 1,
) -> str:
    """Return the default relation name for one grid."""
    stem = grid.source_path.stem
    if multi_sheet_workbook and grid.sheet_name:
        base = f"{stem}__{_deterministic_identifier(grid.sheet_name)}"
    else:
        base = stem
    if region_count > 1:
        return f"{base}__{region_index}"
    return base


def pinned_names_from_schema_graph(
    schema_graph: SchemaGraph | None,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Extract upload-label to pinned-name maps from a cached schema graph."""
    table_pins: dict[str, str] = {}
    column_pins: dict[str, dict[str, str]] = {}
    if schema_graph is None:
        return table_pins, column_pins
    for table in schema_graph.tables.values():
        table_label = (table.original_name or "").strip() or table.name
        table_pins[table_label] = table.name
        column_map: dict[str, str] = {}
        for column in table.columns.values():
            column_label = (column.original_name or "").strip() or column.name
            column_map[column_label] = column.name
        column_pins[table_label] = column_map
    return table_pins, column_pins


def resolve_identifier_name(
    label: str,
    *,
    kind: str,
    pinned_names: Mapping[str, str],
    reserved: set[str],
) -> str:
    """Resolve a pinned or newly assigned SQL identifier for one upload label."""
    original_key = label.strip()
    if not original_key:
        return _dedupe_identifier("column", reserved)
    cached = pinned_names.get(original_key)
    if cached:
        return cached
    if _label_needs_llm_identifier(original_key, reserved=reserved):
        proposed = _llm_identifier(original_key, kind=kind)
        if not proposed:
            proposed = _deterministic_identifier(original_key)
    else:
        proposed = _deterministic_identifier(original_key)
    return _dedupe_identifier(proposed, reserved)


def prepare_relations_for_paths(
    paths: Sequence[Path],
    *,
    pinned_table_names: Mapping[str, str] | None = None,
    pinned_column_names: Mapping[str, Mapping[str, str]] | None = None,
    apply_auto_correct: bool = True,
    normalize_cell_newlines: bool = True,
    source_selections: Mapping[str, CsvSourceSelection] | None = None,
) -> list[PreparedRelation]:
    """Parse uploads into loadable relations with pinned identifier names."""
    table_pins = dict(pinned_table_names or {})
    column_pins = {table: dict(cols) for table, cols in (pinned_column_names or {}).items()}
    relations: list[PreparedRelation] = []
    reserved_tables: set[str] = set()
    selections = source_selections or {}
    for path in paths:
        selection = selections.get(path.name)
        try:
            loaded_grids = load_source_grids(path, selection=selection)
        except ValueError:
            continue
        multi_sheet = _workbook_has_multiple_sheets(path)
        for grid in loaded_grids:
            normalized = normalize_grid(grid, normalize_cell_newlines=normalize_cell_newlines)
            working = normalized
            if apply_auto_correct:
                working, _ = apply_structural_fixes(working)
            region_iter, region_issues = _regions_for_grid(working, selection)
            if any(_issue_is_blocking(issue) for issue in region_issues):
                continue
            for region_grid, region_index, region_count in region_iter:
                issues = detect_grid_issues(region_grid)
                if any(_issue_is_blocking(issue) for issue in issues):
                    continue
                if not region_grid.cells:
                    continue
                original_table_label = _table_original_label(
                    region_grid,
                    region_index=region_index,
                    region_count=region_count,
                )
                pinned_table = table_pins.get(original_table_label)
                if pinned_table:
                    relation_name = _dedupe_identifier(pinned_table, reserved_tables)
                elif region_count > 1:
                    relation_name = _dedupe_identifier(
                        relation_name_for_grid(
                            region_grid,
                            multi_sheet_workbook=multi_sheet,
                            region_index=region_index,
                            region_count=region_count,
                        ),
                        reserved_tables,
                    )
                else:
                    relation_name = resolve_identifier_name(
                        original_table_label,
                        kind="table",
                        pinned_names=table_pins,
                        reserved=reserved_tables,
                    )
                reserved_tables.add(relation_name)
                header_labels = list(region_grid.cells[0])
                col_reserved: set[str] = set()
                column_names: list[str] = []
                table_column_pins = column_pins.get(original_table_label, {})
                for label in header_labels:
                    col_name = resolve_identifier_name(
                        label,
                        kind="column",
                        pinned_names=table_column_pins,
                        reserved=col_reserved,
                    )
                    col_reserved.add(col_name)
                    column_names.append(col_name)
                relations.append(
                    _finalize_prepared_relation(
                        grid_to_relation(
                            region_grid,
                            relation_name=relation_name,
                            column_names=column_names,
                            original_column_labels=header_labels,
                        ),
                        region_grid=region_grid,
                        selection=selection,
                    )
                )
    return relations


def _load_csv_grid(path: Path) -> SheetGrid:
    text = _read_text_with_encoding(path)
    rows, single_column = _parse_csv_rows(text)
    return SheetGrid(source_path=path, sheet_name="", cells=tuple(rows), csv_single_column=single_column)


def infer_header_row(grid: SheetGrid) -> int | None:
    """Return the most likely 1-based header row for *grid*."""
    if grid.header_row_confirmed and grid.cells:
        return 1
    excel_header = _infer_header_from_excel_tables(grid)
    if excel_header is not None:
        return excel_header
    if not grid.cells:
        return None
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if not non_empty:
        return None
    width = max(len(grid.cells[r]) for r in non_empty)
    best_row = non_empty[0] + 1
    best_score = -1.0
    for row_idx in non_empty[:20]:
        row = _pad_row(grid.cells[row_idx], width)
        score = _header_row_score(row, grid.cells, row_idx, width)
        rounded = round(score, 1)
        best_rounded = round(best_score, 1)
        if rounded > best_rounded or (rounded == best_rounded and row_idx + 1 < best_row):
            best_score = score
            best_row = row_idx + 1
    return best_row


def _sniff_csv_format(text: str) -> tuple[str, str]:
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter, dialect.quotechar or '"'
    except csv.Error:
        return ",", '"'


def _parse_csv_rows(text: str) -> tuple[list[tuple[str, ...]], bool]:
    delimiter, quotechar = _sniff_csv_format(text)

    def _read_rows(delimiter_value: str) -> list[tuple[str, ...]]:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter_value, quotechar=quotechar)
        return [tuple(cell for cell in row) for row in reader]

    rows = _read_rows(delimiter)
    if not rows:
        return [], False
    if max(len(row) for row in rows) > 1:
        return rows, False
    for candidate in (";", "\t", "|"):
        if candidate == delimiter:
            continue
        alt_rows = _read_rows(candidate)
        if alt_rows and max(len(row) for row in alt_rows) > 1:
            return alt_rows, False
    return rows, True


def _crop_table_range(cells: list[tuple[str, ...]], table_range: str) -> list[tuple[str, ...]]:
    match = re.match(r"^([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$", table_range.strip())
    if not match:
        return cells
    col_start = _col_index(match.group(1))
    row_start = int(match.group(2)) - 1
    col_end = _col_index(match.group(3))
    row_end = int(match.group(4)) - 1
    cropped: list[tuple[str, ...]] = []
    for row_idx in range(row_start, min(row_end + 1, len(cells))):
        row = cells[row_idx]
        cropped.append(tuple(row[col_start : col_end + 1] if col_end < len(row) else row[col_start:]))
    return cropped


def _col_index(letters: str) -> int:
    result = 0
    for ch in letters.upper():
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1


def _header_row_score(
    header: Sequence[str],
    cells: Sequence[tuple[str, ...]],
    row_idx: int,
    width: int,
) -> float:
    populated = sum(1 for cell in header if cell.strip())
    if populated == 0:
        return 0.0
    text_like = sum(1 for cell in header if cell.strip() and not _looks_numeric_header_cell(cell))
    score = text_like / max(populated, 1) * 0.5
    below = cells[row_idx + 1 : row_idx + 6]
    if below:
        consistent = 0
        for col_idx in range(width):
            kinds = {_value_kind(_pad_row(row, width)[col_idx]) for row in below if any(c.strip() for c in row)}
            kinds.discard("empty")
            if len(kinds) <= 1:
                consistent += 1
        score += 0.5 * (consistent / max(width, 1))
    if row_idx == 0:
        score += 0.05
    labels = [cell.strip().lower() for cell in header if cell.strip()]
    if labels:
        dupes = len(labels) - len(set(labels))
        score -= (dupes / len(labels)) * 0.25
        avg_len = sum(len(cell.strip()) for cell in header if cell.strip()) / len(labels)
        if avg_len > 45:
            score -= 0.25
    return min(max(score, 0.0), 1.0)


def _looks_numeric_header_cell(value: str) -> bool:
    if _plan_scalar_affix_column([value]) is not None:
        return True
    return _value_kind(value) in {"int", "float"}


def _is_date_only_excel_format(number_format: str) -> bool:
    fmt = (number_format or "").lower()
    if not fmt or fmt == "general":
        return False
    if not any(ch in fmt for ch in "ymd"):
        return False
    time_markers = ("h", "s", ":", "am/pm", "a/p", "[h]")
    return not any(marker in fmt for marker in time_markers)


def _format_datetime_value(value: datetime, *, date_only: bool) -> str:
    if date_only:
        return value.date().isoformat()
    return value.isoformat(timespec="seconds")


def excel_cell_to_text(cell: object) -> str:
    value = getattr(cell, "value", cell)
    if value is None:
        return ""
    if isinstance(value, datetime):
        number_format = str(getattr(cell, "number_format", "") or "")
        date_only = _is_date_only_excel_format(number_format) and not (
            value.hour or value.minute or value.second or value.microsecond
        )
        return _format_datetime_value(value, date_only=date_only)
    if isinstance(value, date):
        return value.isoformat()
    number_format = str(getattr(cell, "number_format", "") or "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers = importlib.import_module("openpyxl.styles.numbers")
        if numbers.is_date_format(number_format):
            from_excel = importlib.import_module("openpyxl.utils.datetime").from_excel
            parsed = from_excel(value)
            if isinstance(parsed, datetime):
                return _format_datetime_value(
                    parsed,
                    date_only=_is_date_only_excel_format(number_format),
                )
            if isinstance(parsed, date):
                return parsed.isoformat()
    return str(value)


def _temporal_sample_kind(value: str) -> str | None:
    text = value.strip()
    if ISO_DATE_RE.match(text):
        return "date"
    if ISO_TIMESTAMP_RE.match(text):
        return "timestamp"
    return None


def _xlsx_declared_column_types(path: Path, sheet_name: str, rows: Sequence[Sequence[str]]) -> tuple[str, ...]:
    if not rows:
        return ()
    width = max(len(row) for row in rows)
    samples_by_col: list[list[str]] = [[] for _ in range(width)]
    for row in rows[1:]:
        padded = _pad_row(tuple(row), width)
        for col_idx in range(width):
            value = padded[col_idx] if col_idx < len(padded) else ""
            if value.strip():
                samples_by_col[col_idx].append(value)
    return tuple(infer_duckdb_column_type(samples) for samples in samples_by_col)


def _load_xlsx_grids(path: Path) -> list[SheetGrid]:
    require_driver("csv")
    openpyxl = importlib.import_module("openpyxl")
    try:
        workbook = openpyxl.load_workbook(path, read_only=False, data_only=True)
    except Exception as exc:
        message = str(exc).lower()
        if "encrypt" in message or "password" in message:
            raise ValueError("workbook is encrypted or password-protected") from exc
        raise ValueError(f"workbook could not be opened: {exc}") from exc
    grids: list[SheetGrid] = []
    try:
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            if getattr(worksheet, "sheet_state", "visible") != "visible":
                continue
            rows: list[tuple[str, ...]] = []
            for row_cells in worksheet.iter_rows():
                if row_cells is None:
                    continue
                rows.append(tuple(excel_cell_to_text(cell) for cell in row_cells))
            if rows:
                _XLSX_DECLARED_COLUMN_TYPES[(str(path.resolve()), str(sheet_name))] = _xlsx_declared_column_types(
                    path, str(sheet_name), rows
                )
            merged = tuple(str(item) for item in getattr(worksheet.merged_cells, "ranges", ()))
            tables = tuple(str(item) for item in getattr(worksheet, "tables", {}).keys())
            table_ranges = tuple(str(table.ref) for table in getattr(worksheet, "tables", {}).values())
            has_charts = bool(getattr(worksheet, "_charts", ()))
            has_images = bool(getattr(worksheet, "_images", ()))
            grids.append(
                SheetGrid(
                    source_path=path,
                    sheet_name=str(sheet_name),
                    cells=tuple(rows),
                    merged_ranges=merged,
                    has_charts=has_charts,
                    has_images=has_images,
                    excel_tables=tables,
                    excel_table_ranges=table_ranges,
                )
            )
    finally:
        workbook.close()
    return grids


def _read_text_with_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    for bad, good in DATA_QUALITY_MOJIBAKE_REPLACEMENTS:
        text = text.replace(bad, good)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_cell_text(value: str, *, normalize_cell_newlines: bool) -> str:
    text = str(value)
    for ch in DATA_QUALITY_ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    text = text.strip()
    if normalize_cell_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _pad_row(row: Sequence[str], width: int) -> tuple[str, ...]:
    padded = list(row) + [""] * max(0, width - len(row))
    return tuple(padded[:width])


def _grid_location_prefix(grid: SheetGrid) -> str:
    if grid.sheet_name:
        return f"{grid.source_path.name}!{grid.sheet_name}"
    return grid.source_path.name


def _table_original_label(grid: SheetGrid, *, region_index: int = 1, region_count: int = 1) -> str:
    if grid.sheet_name:
        base = f"{grid.source_path.name}!{grid.sheet_name}"
    else:
        base = grid.source_path.stem
    if region_count > 1:
        return f"{base}!region{region_index}"
    return base


def _col_letter(index: int) -> str:
    result = ""
    num = index + 1
    while num:
        num, rem = divmod(num - 1, 26)
        result = chr(65 + rem) + result
    return result


def _split_blank_separated_regions(grid: SheetGrid) -> list[SheetGrid]:
    """Split *grid* into sub-grids at fully blank rows."""
    if not grid.cells:
        return [grid]
    row_groups: list[list[tuple[str, ...]]] = []
    current: list[tuple[str, ...]] = []
    for row in grid.cells:
        if not any(cell.strip() for cell in row):
            if current:
                row_groups.append(current)
                current = []
            continue
        current.append(row)
    if current:
        row_groups.append(current)
    if len(row_groups) <= 1:
        return [grid]
    regions: list[SheetGrid] = []
    for rows in row_groups:
        regions.append(
            SheetGrid(
                source_path=grid.source_path,
                sheet_name=grid.sheet_name,
                cells=tuple(rows),
                merged_ranges=grid.merged_ranges,
                has_charts=grid.has_charts,
                has_images=grid.has_images,
                excel_tables=grid.excel_tables,
                excel_table_ranges=grid.excel_table_ranges,
                csv_single_column=grid.csv_single_column,
                header_row_confirmed=grid.header_row_confirmed,
            )
        )
    return regions


def _clone_grid_with_cells(grid: SheetGrid, cells: Sequence[tuple[str, ...]]) -> SheetGrid:
    return SheetGrid(
        source_path=grid.source_path,
        sheet_name=grid.sheet_name,
        cells=tuple(cells),
        merged_ranges=grid.merged_ranges,
        has_charts=grid.has_charts,
        has_images=grid.has_images,
        excel_tables=grid.excel_tables,
        excel_table_ranges=grid.excel_table_ranges,
        csv_single_column=grid.csv_single_column,
        header_row_confirmed=grid.header_row_confirmed,
    )


def _trim_blank_border_grid(grid: SheetGrid) -> tuple[SheetGrid, list[Diagnostic]]:
    """Trim blank leading and trailing rows/columns from *grid*."""
    if not grid.cells:
        return grid, []
    non_empty_rows = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if not non_empty_rows:
        return grid, []
    first_row = non_empty_rows[0]
    last_row = non_empty_rows[-1]
    width = max(len(grid.cells[r]) for r in non_empty_rows)
    trimmed_rows: list[tuple[str, ...]] = []
    for row_idx in range(first_row, last_row + 1):
        trimmed_rows.append(grid.cells[row_idx])
    non_empty_cols = [
        col_idx for col_idx in range(width) if any(col_idx < len(row) and row[col_idx].strip() for row in trimmed_rows)
    ]
    if not non_empty_cols:
        return grid, []
    first_col = non_empty_cols[0]
    last_col = non_empty_cols[-1]
    cropped = tuple(
        tuple(row[col_idx] for col_idx in range(first_col, min(len(row), last_col + 1))) for row in trimmed_rows
    )
    corrected: list[Diagnostic] = []
    if first_row > 0 or first_col > 0 or last_row < len(grid.cells) - 1 or last_col < width - 1:
        corrected.append(
            _make_issue(
                code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
                level="info",
                message="Trimmed blank leading or trailing rows and columns.",
                location=_grid_location_prefix(grid),
                blocking=False,
                diagnostic_code=DIAGNOSTIC_CODE_DATA_QUALITY_AUTO_CORRECTED,
            )
        )
        return _clone_grid_with_cells(grid, cropped), corrected
    return grid, corrected


def _is_key_value_column_header(row: Sequence[str]) -> bool:
    if len(row) < 2:
        return False
    left = row[0].strip().lower()
    right = row[1].strip().lower()
    return left in {"field", "key", "attribute", "property", "name"} and right in {"value", "val", "data"}


def _looks_key_value_grid(grid: SheetGrid) -> bool:
    if not grid.cells:
        return False
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if len(non_empty) < 2:
        return False
    width = max(len(grid.cells[r]) for r in non_empty)
    if width != 2:
        return False
    rows = [_pad_row(grid.cells[row_idx], width) for row_idx in non_empty]
    start = 1 if _is_key_value_column_header(rows[0]) else 0
    if len(rows) - start < 2:
        return False
    keys = [row[0].strip() for row in rows[start:]]
    if not all(keys) or len({key.lower() for key in keys}) != len(keys):
        return False
    if all(_value_kind(key) in {"int", "float"} for key in keys):
        return False
    if start == 0:
        below = rows[1:]
        numeric_keys = sum(1 for row in below if _value_kind(row[0]) in {"int", "float"})
        if numeric_keys >= max(1, len(below) // 2):
            return False
        if _header_row_score(rows[0], grid.cells, non_empty[0], width) >= 0.5:
            return False
    return True


def _transpose_key_value_grid(grid: SheetGrid) -> SheetGrid:
    if not _looks_key_value_grid(grid):
        return grid
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    width = max(len(grid.cells[r]) for r in non_empty)
    rows = [_pad_row(grid.cells[row_idx], width) for row_idx in non_empty]
    start = 1 if _is_key_value_column_header(rows[0]) else 0
    keys = [row[0].strip() for row in rows[start:]]
    values = [row[1].strip() for row in rows[start:]]
    return _clone_grid_with_cells(grid, (tuple(keys), tuple(values)))


def _has_repeated_header_row(grid: SheetGrid) -> bool:
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if len(non_empty) < 2:
        return False
    width = max(len(grid.cells[r]) for r in non_empty)
    canonical = _pad_row(grid.cells[non_empty[0]], width)
    return any(_rows_equal_cells(_pad_row(grid.cells[idx], width), canonical) for idx in non_empty[1:])


def _skip_rows_above_inferred_header(grid: SheetGrid) -> tuple[SheetGrid, list[Diagnostic]]:
    if len(_blank_separated_region_spans(grid)) > 1 or _has_repeated_header_row(grid):
        return grid, []
    inferred = infer_header_row(grid)
    if inferred is None or inferred <= 1:
        return grid, []
    offset = inferred - 1
    if offset >= len(grid.cells):
        return grid, []
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if not non_empty:
        return grid, []
    first_populated = non_empty[0]
    width = max(len(grid.cells[r]) for r in non_empty)
    location_prefix = _grid_location_prefix(grid)
    header_loc = f"{location_prefix}!A{inferred}:{_col_letter(width - 1)}{inferred}"
    issue = _make_issue(
        code=DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE,
        level="warning",
        message=(
            f"Confirm the header row (looks like row {inferred}, {header_loc}). "
            "Remove the rows above it or set CSV_SOURCE_SELECTIONS."
        ),
        location=f"{location_prefix}!A{first_populated + 1}:{_col_letter(width - 1)}{first_populated + 1}",
        blocking=False,
        extra_details=(
            (DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW, str(inferred)),
            (
                DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE,
                f"A{inferred}:{_col_letter(width - 1)}{len(grid.cells)}",
            ),
        ),
    )
    return _clone_grid_with_cells(grid, grid.cells[offset:]), [issue]


def _pad_grid_to_width(grid: SheetGrid) -> SheetGrid:
    if not grid.cells:
        return grid
    width = max(len(row) for row in grid.cells)
    padded_cells = tuple(_pad_row(row, width) for row in grid.cells)
    if padded_cells == grid.cells:
        return grid
    return _clone_grid_with_cells(grid, padded_cells)


def _split_grid_at_row_indices(grid: SheetGrid, start_indices: Sequence[int]) -> list[SheetGrid]:
    chunks: list[SheetGrid] = []
    ordered = sorted(set(start_indices))
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else len(grid.cells)
        rows = grid.cells[start:end]
        if rows:
            chunks.append(_clone_grid_with_cells(grid, rows))
    return chunks if chunks else [grid]


def _split_report_layout_regions(grid: SheetGrid) -> list[SheetGrid]:
    """Split one grid at repeated headers or section-heading delimiters."""
    if not grid.cells:
        return [grid]
    non_empty = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if len(non_empty) < 2:
        return [grid]
    width = max(len(grid.cells[r]) for r in non_empty)
    header_idx = non_empty[0]
    canonical_header = _pad_row(grid.cells[header_idx], width)
    repeated_splits = [
        idx for idx in non_empty[1:] if _rows_equal_cells(_pad_row(grid.cells[idx], width), canonical_header)
    ]
    if repeated_splits:
        return _split_grid_at_row_indices(grid, [header_idx, *repeated_splits])

    section_splits = [idx for idx in non_empty[1:] if _looks_section_heading_row(_pad_row(grid.cells[idx], width))]
    if not section_splits:
        return [grid]

    chunk_rows: list[tuple[tuple[str, ...], ...]] = []
    data_start = header_idx + 1
    for sec_idx in section_splits:
        if sec_idx > data_start:
            chunk_rows.append((grid.cells[header_idx],) + grid.cells[data_start:sec_idx])
        data_start = sec_idx + 1
    trailing = tuple(
        row
        for row in grid.cells[data_start:]
        if any(cell.strip() for cell in row) and not _looks_section_heading_row(_pad_row(row, width))
    )
    if trailing:
        chunk_rows.append((grid.cells[header_idx],) + trailing)
    if not chunk_rows:
        return [grid]
    return [_clone_grid_with_cells(grid, rows) for rows in chunk_rows]


def _indexed_regions(grid: SheetGrid) -> list[tuple[SheetGrid, int, int]]:
    """Return region grids with 1-based index and total count."""
    regions: list[SheetGrid] = []
    for blank_region in _split_blank_separated_regions(grid):
        regions.extend(_split_report_layout_regions(blank_region))
    count = len(regions)
    return [(region, idx, count) for idx, region in enumerate(regions, start=1)]


def _regions_for_grid(
    grid: SheetGrid,
    selection: CsvSourceSelection | None,
) -> tuple[list[tuple[SheetGrid, int, int]], list[Diagnostic]]:
    """Return region grids for validation/loading, honoring merge/append regions when set."""
    if selection and selection.append_regions:
        appended, append_issues = _build_merged_region_grid(
            grid,
            selection.append_regions,
            header_mismatch_code=DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH,
        )
        if appended is None:
            return [], append_issues
        return [(appended, 1, 1)], append_issues
    if selection and selection.merge_regions:
        merged, merge_issues = _build_merged_region_grid(
            grid,
            selection.merge_regions,
            header_mismatch_code=DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH,
        )
        if merged is None:
            return [], merge_issues
        return [(merged, 1, 1)], merge_issues
    return _indexed_regions(grid), []


def _blank_separated_region_spans(grid: SheetGrid) -> list[tuple[int, int]]:
    """Return inclusive 0-based row spans for blank-separated regions."""
    if not grid.cells:
        return []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for row_idx, row in enumerate(grid.cells):
        if not any(cell.strip() for cell in row):
            if start is not None:
                spans.append((start, row_idx - 1))
                start = None
            continue
        if start is None:
            start = row_idx
    if start is not None:
        spans.append((start, len(grid.cells) - 1))
    return spans


def _span_a1_range(grid: SheetGrid, start_row: int, end_row: int) -> str:
    """Format an absolute A1 range for one row span inside *grid*."""
    non_empty_rows = [idx for idx, row in enumerate(grid.cells) if any(cell.strip() for cell in row)]
    if not non_empty_rows:
        return f"A{start_row + 1}"
    width = max(len(grid.cells[r]) for r in non_empty_rows)
    last_col = _col_letter(max(width - 1, 0))
    return f"A{start_row + 1}:{last_col}{end_row + 1}"


def _normalized_header_cells(grid: SheetGrid) -> tuple[str, ...] | None:
    """Return the normalized header row for *grid*, or None when empty."""
    if not grid.cells:
        return None
    return tuple(cell.strip().lower() for cell in grid.cells[0])


def _mergeable_regions_advisory(grid: SheetGrid) -> Diagnostic | None:
    """Emit one advisory when blank-separated regions share identical headers."""
    spans = _blank_separated_region_spans(grid)
    if len(spans) < 2:
        return None
    headers: list[tuple[str, ...]] = []
    ranges: list[str] = []
    for start_row, end_row in spans:
        region_rows = grid.cells[start_row : end_row + 1]
        if not region_rows:
            continue
        header = tuple(cell.strip().lower() for cell in region_rows[0])
        if not any(header):
            return None
        headers.append(header)
        ranges.append(_span_a1_range(grid, start_row, end_row))
    if len(headers) < 2 or len({header for header in headers}) != 1:
        return None
    range_text = " and ".join(ranges)
    return _make_issue(
        code=DATA_QUALITY_ISSUE_MERGEABLE_REGIONS,
        level="info",
        message=(
            f"This sheet has {len(ranges)} blocks with the same headers ({range_text}). "
            "They were loaded as separate tables. To combine them, set merge_regions in "
            "CSV_SOURCE_SELECTIONS, or remove the blank rows between them and re-upload."
        ),
        location=_grid_location_prefix(grid),
        blocking=False,
    )


def _appendable_regions_advisory(grid: SheetGrid) -> Diagnostic | None:
    """Emit one advisory when blank-separated regions share identical headers for append."""
    spans = _blank_separated_region_spans(grid)
    if len(spans) < 2:
        return None
    headers: list[tuple[str, ...]] = []
    ranges: list[str] = []
    for start_row, end_row in spans:
        region_rows = grid.cells[start_row : end_row + 1]
        if not region_rows:
            continue
        header = tuple(cell.strip().lower() for cell in region_rows[0])
        if not any(header):
            return None
        headers.append(header)
        ranges.append(_span_a1_range(grid, start_row, end_row))
    if len(headers) < 2 or len({header for header in headers}) != 1:
        return None
    range_text = " and ".join(ranges)
    return _make_issue(
        code=DATA_QUALITY_ISSUE_APPENDABLE_REGIONS,
        level="info",
        message=(
            f"This sheet has {len(ranges)} blocks with the same headers ({range_text}). "
            "They were loaded as separate tables. To append them into one table, set append_regions "
            "in CSV_SOURCE_SELECTIONS, or remove the blank rows between them and re-upload."
        ),
        location=_grid_location_prefix(grid),
        blocking=False,
    )


def _build_merged_region_grid(
    grid: SheetGrid,
    merge_ranges: Sequence[str],
    *,
    header_mismatch_code: str = DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH,
) -> tuple[SheetGrid | None, list[Diagnostic]]:
    """Union *merge_ranges* into one grid, keeping the first header row."""
    location_prefix = _grid_location_prefix(grid)
    issues: list[Diagnostic] = []
    if not merge_ranges:
        return None, issues
    all_rows: list[tuple[str, ...]] = []
    canonical_header: tuple[str, ...] | None = None
    for range_ref in merge_ranges:
        cropped = _crop_table_range(list(grid.cells), range_ref)
        if not cropped:
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE,
                    level="error",
                    message=f"Remove or fix the invalid merge range {range_ref!r}.",
                    location=location_prefix,
                    blocking=True,
                )
            )
            return None, issues
        header = tuple(cell.strip().lower() for cell in cropped[0])
        if canonical_header is None:
            canonical_header = header
            all_rows.extend(cropped)
            continue
        if header != canonical_header:
            issues.append(
                _make_issue(
                    code=header_mismatch_code,
                    level="error",
                    message=(
                        f"Remove or fix the mismatched headers between {merge_ranges[0]!r} and "
                        f"{range_ref!r} — the column names do not match."
                    ),
                    location=location_prefix,
                    blocking=True,
                )
            )
            return None, issues
        all_rows.extend(cropped[1:])
    if not all_rows:
        return None, issues
    merged = SheetGrid(
        source_path=grid.source_path,
        sheet_name=grid.sheet_name,
        cells=tuple(all_rows),
        merged_ranges=grid.merged_ranges,
        has_charts=grid.has_charts,
        has_images=grid.has_images,
        excel_tables=grid.excel_tables,
        excel_table_ranges=grid.excel_table_ranges,
        csv_single_column=grid.csv_single_column,
        header_row_confirmed=grid.header_row_confirmed,
    )
    return merged, issues


def _issue_detail(issue: Diagnostic, key: str, default: str = "") -> str:
    for detail_key, value in issue.details:
        if detail_key == key:
            return value
    return default


def _issue_code(issue: Diagnostic) -> str:
    return _issue_detail(issue, "issue_code", issue.code)


def _issue_severity(issue: Diagnostic) -> str:
    stored = _issue_detail(issue, "severity", "")
    if stored:
        return stored
    code = _issue_detail(issue, "issue_code", "")
    if code in DATA_QUALITY_ISSUE_SEVERITY:
        return DATA_QUALITY_ISSUE_SEVERITY[code]
    return issue.level.value if isinstance(issue.level, DiagnosticSeverity) else str(issue.level)


def _column_key_from_location(location: str) -> str | None:
    match = re.search(r"!([A-Za-z]+)\d", location)
    if match is None:
        return None
    return match.group(1).upper()


def _collapse_issues(issues: list[Diagnostic]) -> list[Diagnostic]:
    """Collapse redundant sibling issues before returning them."""
    column_issue_precedence: dict[str, int] = {
        DATA_QUALITY_ISSUE_EXCEL_ERROR: 0,
        DATA_QUALITY_ISSUE_MIXED_TYPES: 1,
        DATA_QUALITY_ISSUE_NUMBER_AS_TEXT: 2,
        DATA_QUALITY_ISSUE_NULL_TOKEN: 3,
        DATA_QUALITY_ISSUE_FORMULA_CELL: 4,
        DATA_QUALITY_ISSUE_EMPTY_COLUMN: 5,
    }
    row_noise_codes = frozenset(
        {
            DATA_QUALITY_ISSUE_BLANK_ROW,
            DATA_QUALITY_ISSUE_RAGGED_ROW,
            DATA_QUALITY_ISSUE_OVERFULL_ROW,
            DATA_QUALITY_ISSUE_TOTAL_ROW,
        }
    )
    report_layout_codes = frozenset(
        {
            DATA_QUALITY_ISSUE_REPEATED_HEADER,
            DATA_QUALITY_ISSUE_SECTION_HEADING,
        }
    )
    if not issues:
        return issues
    report_layout_prefixes: set[str] = set()
    for issue in issues:
        if _issue_code(issue) in report_layout_codes:
            location = _issue_detail(issue, "location")
            prefix = location.split("!")[0] if location else ""
            report_layout_prefixes.add(prefix)

    filtered: list[Diagnostic] = []
    for issue in issues:
        code = _issue_code(issue)
        location = _issue_detail(issue, "location")
        prefix = location.split("!")[0] if location else ""
        if code in row_noise_codes and prefix in report_layout_prefixes:
            continue
        filtered.append(issue)

    column_groups: dict[str, Diagnostic] = {}
    passthrough: list[Diagnostic] = []
    for issue in filtered:
        code = _issue_code(issue)
        if code not in column_issue_precedence:
            passthrough.append(issue)
            continue
        column_key = _column_key_from_location(_issue_detail(issue, "location"))
        if column_key is None:
            passthrough.append(issue)
            continue
        group_key = f"{_issue_detail(issue, 'location').split('!')[0]}!{column_key}"
        current = column_groups.get(group_key)
        if current is None:
            column_groups[group_key] = issue
        elif column_issue_precedence.get(code, 99) < column_issue_precedence.get(_issue_code(current), 99):
            column_groups[group_key] = issue

    collapsed = passthrough + list(column_groups.values())
    seen_messages: set[str] = set()
    deduped: list[Diagnostic] = []
    for issue in collapsed:
        if issue.message in seen_messages:
            continue
        seen_messages.add(issue.message)
        deduped.append(issue)
    return deduped


def _column_range_location(
    location_prefix: str,
    col_idx: int,
    *,
    header_row_num: int,
    last_row_num: int,
) -> str:
    col = _col_letter(col_idx)
    if last_row_num <= header_row_num:
        return f"{location_prefix}!{col}{header_row_num}"
    return f"{location_prefix}!{col}{header_row_num}:{col}{last_row_num}"


def _label_needs_llm_identifier(label: str, *, reserved: set[str]) -> bool:
    text = label.strip()
    if not text:
        return False
    deterministic = _deterministic_identifier(text)
    if deterministic in reserved:
        return True
    if deterministic in DATA_QUALITY_SQL_RESERVED_WORDS:
        return True
    if text[0].isdigit():
        return True
    if any(ord(ch) > 127 for ch in text):
        return True
    if any(not ch.isalnum() and ch not in "_ " for ch in text):
        return True
    if _valid_identifier(text) and text == deterministic:
        return False
    return False


def _detect_side_by_side_regions(header: Sequence[str]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for idx, cell in enumerate(header):
        if cell.strip():
            if start is None:
                start = idx
        elif start is not None:
            regions.append((start, idx - 1))
            start = None
    if start is not None:
        regions.append((start, len(header) - 1))
    return [region for region in regions if any(header[i].strip() for i in range(region[0], region[1] + 1))]


def _column_value_issues(
    grid: SheetGrid,
    header_idx: int,
    col_idx: int,
    header_label: str,
    samples: Sequence[str],
    *,
    last_row_num: int,
) -> list[Diagnostic]:
    issues: list[Diagnostic] = []
    location_prefix = _grid_location_prefix(grid)
    header_row_num = header_idx + 1
    col_loc = _column_range_location(
        location_prefix,
        col_idx,
        header_row_num=header_row_num,
        last_row_num=last_row_num,
    )
    col_ref = f"column {header_label!r} ({_col_letter(col_idx)}{header_row_num}:{_col_letter(col_idx)}{last_row_num})"
    for sample in samples:
        lowered = sample.strip().lower()
        if lowered in DATA_QUALITY_EXCEL_ERROR_TOKENS:
            issues.append(
                _make_issue(
                    code=DATA_QUALITY_ISSUE_EXCEL_ERROR,
                    level="error",
                    message=f"Remove the Excel error value {sample!r} in {col_ref}.",
                    location=col_loc,
                    blocking=True,
                )
            )
            break
    null_tokens = {token for token in DATA_QUALITY_NULL_TOKENS if token}
    if null_tokens and any(value.strip().lower() in null_tokens for value in samples):
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_NULL_TOKEN,
                level="error",
                message=f"Replace the ambiguous blank tokens in {col_ref} with real empty cells or one null value.",
                location=col_loc,
                blocking=True,
            )
        )
    kind_flags = _storage_kinds(samples)
    temporal_and_text = "temporal" in kind_flags and "text" in kind_flags
    if temporal_and_text:
        issues.append(
            _make_issue(
                code="mixed_temporal_text",
                level="review",
                message=(f"Column {col_ref} mixes dates with text values; keeping the column as text."),
                location=col_loc,
                blocking=False,
                review=True,
                extra_details=(("issue_code", DATA_QUALITY_ISSUE_MIXED_TYPES),),
            )
        )
    elif len(kind_flags) > 1:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_MIXED_TYPES,
                level="error",
                message=f"Make every value the same type in {col_ref}, or remove the stray values.",
                location=col_loc,
                blocking=True,
            )
        )
    if any(_looks_number_as_text(value) for value in samples) and _plan_scalar_affix_column(samples) is None:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_NUMBER_AS_TEXT,
                level="error",
                message=f"Convert the numbers stored as text in {col_ref} to real numbers.",
                location=col_loc,
                blocking=True,
            )
        )
    if header_label.strip() and not samples:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_EMPTY_COLUMN,
                level="warning",
                message=f"Add data to {col_ref} or remove the empty column.",
                location=col_loc,
                blocking=False,
            )
        )
    if any(value.strip().startswith("=") for value in samples):
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_FORMULA_CELL,
                level="error",
                message=f"Replace the formula cells in {col_ref} with their calculated values.",
                location=col_loc,
                blocking=True,
            )
        )
    return issues


def _named_header_end(header: Sequence[str]) -> int:
    last = -1
    for idx, cell in enumerate(header):
        if cell.strip():
            last = idx
    return last


def _is_structurally_ragged_row(row: Sequence[str], header: Sequence[str]) -> bool:
    """Return True when the parsed row has fewer physical columns than the header."""
    if not any(cell.strip() for cell in row):
        return False
    return len(row) < len(header)


def _is_structurally_overfull_row(row: Sequence[str], header: Sequence[str]) -> bool:
    """Return True when the parsed row has more physical columns than the header."""
    if not any(cell.strip() for cell in row):
        return False
    return len(row) > len(header)


def _parse_a1_range(ref: str) -> tuple[int, int, int, int] | None:
    match = re.match(r"^([A-Za-z]+)(\d+)(?::([A-Za-z]+)(\d+))?$", ref.strip())
    if not match:
        return None
    col_start = _col_index(match.group(1))
    row_start = int(match.group(2)) - 1
    if match.group(3):
        col_end = _col_index(match.group(3))
        row_end = int(match.group(4)) - 1
    else:
        col_end = col_start
        row_end = row_start
    return row_start, row_end, col_start, col_end


def _infer_header_from_excel_tables(grid: SheetGrid) -> int | None:
    if not grid.excel_table_ranges:
        return None
    bounds = _parse_a1_range(grid.excel_table_ranges[0])
    if bounds is None:
        return None
    return bounds[0] + 1


def _merged_cell_issues(grid: SheetGrid, header_row: int, location_prefix: str) -> list[Diagnostic]:
    if not grid.merged_ranges:
        return []
    header_idx = header_row - 1
    body_merges: list[str] = []
    metadata_merges: list[str] = []
    for merged in grid.merged_ranges:
        bounds = _parse_a1_range(merged)
        if bounds is None:
            body_merges.append(merged)
            continue
        row_end = bounds[1]
        if row_end < header_idx:
            metadata_merges.append(merged)
        else:
            body_merges.append(merged)
    issues: list[Diagnostic] = []
    if body_merges:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_MERGED_CELLS,
                level="error",
                message="Merged cells were detected inside the table body.",
                location=f"{location_prefix}!{body_merges[0]}",
                blocking=True,
            )
        )
    if metadata_merges:
        issues.append(
            _make_issue(
                code=DATA_QUALITY_ISSUE_MERGED_METADATA,
                level="warning",
                message="Merged cells were detected above the likely header row.",
                location=f"{location_prefix}!{metadata_merges[0]}",
                blocking=False,
            )
        )
    return issues


def _footer_note_indices(
    data_row_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]],
    header: Sequence[str],
) -> set[int]:
    header_populated = sum(1 for cell in header if cell.strip())
    indices: set[int] = set()
    for idx in range(len(data_row_pairs) - 1, -1, -1):
        _raw, row = data_row_pairs[idx]
        populated = sum(1 for cell in row if cell.strip())
        if populated == 0:
            continue
        if _looks_footer_note_row(row, header_populated):
            indices.add(idx)
        elif populated >= min(3, header_populated) or populated >= header_populated:
            break
    return indices


def _looks_footer_note_row(row: Sequence[str], header_populated: int) -> bool:
    populated = sum(1 for cell in row if cell.strip())
    if populated == 0:
        return False
    first = _first_nonempty_cell(row).lower()
    if any(first == prefix or first.startswith(prefix) for prefix in DATA_QUALITY_FOOTER_LABEL_PREFIXES):
        return True
    return populated <= 2 and header_populated >= 5 and populated < header_populated


def _looks_total_row(row: Sequence[str]) -> bool:
    first = _first_nonempty_cell(row)
    if not first:
        return False
    lowered = first.lower()
    return any(lowered == prefix or lowered.startswith(f"{prefix} ") for prefix in DATA_QUALITY_TOTAL_PREFIXES)


def _rows_equal_cells(left: Sequence[str], right: Sequence[str]) -> bool:
    left_text = [cell.strip().lower() for cell in left]
    right_text = [cell.strip().lower() for cell in right]
    return left_text == right_text and any(left_text)


def _looks_section_heading_row(row: Sequence[str]) -> bool:
    populated = [cell.strip() for cell in row if cell.strip()]
    if len(populated) != 1:
        return False
    label = populated[0]
    if _looks_numeric_header_cell(label):
        return False
    return len(label) > 3


def _first_nonempty_cell(row: Sequence[str]) -> str:
    for cell in row:
        if cell.strip():
            return cell.strip()
    return ""


def _storage_kinds(samples: Sequence[str]) -> set[str]:
    if _plan_scalar_affix_column(samples) is not None:
        return {"number"}
    kinds: set[str] = set()
    for value in samples:
        if not value.strip():
            continue
        kind = _value_kind(value)
        if kind in {"int", "float"}:
            kinds.add("number")
        elif kind in {"date", "timestamp"}:
            kinds.add("temporal")
        elif kind != "empty":
            kinds.add(kind)
    return kinds


def _value_kind(value: str) -> str:
    text = value.strip()
    if not text:
        return "empty"
    try:
        if "." not in text and "e" not in text.lower():
            int(text)
            return "int"
    except ValueError:
        pass
    try:
        float(text)
        return "float"
    except ValueError:
        pass
    if text.lower() in ("true", "false", "yes", "no"):
        return "bool"
    temporal_kind = _temporal_sample_kind(text)
    if temporal_kind == "date":
        return "date"
    if temporal_kind == "timestamp":
        return "timestamp"
    return "text"


def _looks_number_as_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if _plan_scalar_affix_column([text]) is not None:
        return False
    if any(ch.isalpha() for ch in text):
        return False
    if re.match(r"^\d+(?:\.\d+)?%\s*[-\u2013]\s*\d+(?:\.\d+)?%$", text):
        return False
    if "$" in text or "%" in text:
        return True
    if "," in text:
        return bool(re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", text))
    return False


def _scalar_value_has_band_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in UPLOAD_SCALAR_BAND_PATTERNS)


def _is_bare_scalar_number(text: str) -> bool:
    return bool(BARE_SCALAR_NUMBER_RE.match(text.strip()))


def _find_scalar_affix_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        matched: str | None = None
        for token in UPLOAD_SCALAR_AFFIX_TOKENS_SORTED:
            if text.startswith(token, index):
                matched = token
                break
        if matched is None:
            index += 1
            continue
        spans.append((index, index + len(matched), matched))
        index += len(matched)
    return spans


def _strip_scalar_affix_spans(text: str, spans: Sequence[tuple[int, int, str]]) -> str:
    if not spans:
        return text.strip()
    parts: list[str] = []
    cursor = 0
    for start, end, _token in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts).strip()


def _parse_scalar_affix_cell(text: str) -> tuple[str, frozenset[str], bool] | None:
    raw = text.strip()
    if not raw or _scalar_value_has_band_marker(raw):
        return None
    if _is_bare_scalar_number(raw):
        return raw.replace(",", ""), frozenset(), False
    spans = _find_scalar_affix_spans(raw)
    if not spans:
        return None
    stripped = _strip_scalar_affix_spans(raw, spans)
    if any(char.isalpha() for char in stripped):
        return None
    if not _is_bare_scalar_number(stripped):
        return None
    tokens = frozenset(token for _start, _end, token in spans)
    has_percent = "%" in tokens
    numeric = stripped.replace(",", "")
    if has_percent:
        return numeric, tokens, True
    return numeric, tokens, False


def _infer_numeric_duckdb_type(values: Sequence[str]) -> str:
    non_empty = [str(value).strip() for value in values if str(value).strip()]
    if not non_empty:
        return "VARCHAR"
    try:
        if all("." not in value and "e" not in value.lower() for value in non_empty) and all(
            int(value) for value in non_empty
        ):
            return "INTEGER"
    except ValueError:
        pass
    try:
        if all(float(value) for value in non_empty):
            return "DOUBLE"
    except ValueError:
        pass
    return "VARCHAR"


def _plan_scalar_affix_column(samples: Sequence[str]) -> ScalarAffixColumnPlan | None:
    non_empty = [str(value).strip() for value in samples if str(value).strip()]
    if not non_empty:
        return None
    currency_tokens: set[str] = set()
    has_percent = False
    stripped_values: list[str] = []
    for value in non_empty:
        parsed = _parse_scalar_affix_cell(value)
        if parsed is None:
            return None
        numeric, tokens, cell_has_percent = parsed
        currency_tokens.update(token for token in tokens if token in UPLOAD_CURRENCY_AFFIX_TOKENS)
        has_percent = has_percent or cell_has_percent or "%" in tokens
        if has_percent and not cell_has_percent and "%" not in tokens and _is_bare_scalar_number(value):
            try:
                fraction = float(numeric)
            except ValueError:
                return None
            if not 0 <= fraction <= 1:
                return None
        stripped_values.append(numeric)
    if len(currency_tokens) > 1:
        return None
    duckdb_type = _infer_numeric_duckdb_type(stripped_values)
    if duckdb_type == "VARCHAR":
        return None
    if currency_tokens:
        unit_label = sorted(currency_tokens, key=len, reverse=True)[0]
    elif has_percent:
        unit_label = "%"
    else:
        return None
    return ScalarAffixColumnPlan(
        duckdb_type=duckdb_type,
        unit_label=unit_label,
        has_percent=has_percent,
    )


def _strip_scalar_affix_cell(value: str, plan: ScalarAffixColumnPlan) -> str:
    parsed = _parse_scalar_affix_cell(value)
    if parsed is None:
        return value
    numeric, _tokens, _cell_has_percent = parsed
    if plan.has_percent and "%" not in value and _is_bare_scalar_number(value):
        try:
            fraction = float(numeric)
        except ValueError:
            return numeric
        if 0 <= fraction <= 1:
            return numeric
    return numeric


def infer_duckdb_column_type(samples: Sequence[str]) -> str:
    non_empty = [str(v).strip() for v in samples if str(v).strip()]
    if not non_empty:
        return "VARCHAR"
    affix_plan = _plan_scalar_affix_column(non_empty)
    if affix_plan is not None:
        return affix_plan.duckdb_type
    temporal_kinds = {_temporal_sample_kind(value) for value in non_empty}
    temporal_kinds.discard(None)
    if temporal_kinds:
        if any(_temporal_sample_kind(value) is None for value in non_empty):
            return "VARCHAR"
        if temporal_kinds == {"date"}:
            return "DATE"
        return "TIMESTAMP"
    if all(v.lower() in ("1", "0", "true", "false", "t", "f", "yes", "no") for v in non_empty):
        return "BOOLEAN"
    try:
        if all("." not in v and "e" not in v.lower() for v in non_empty) and all(int(v) for v in non_empty):
            return "INTEGER"
    except ValueError:
        pass
    try:
        if all(float(v) for v in non_empty):
            return "DOUBLE"
    except ValueError:
        pass
    return "VARCHAR"


def _make_issue(
    *,
    code: str,
    level: str,
    message: str,
    location: str,
    blocking: bool = False,
    diagnostic_code: str | None = None,
    guidance: str = "",
    extra_details: tuple[tuple[str, str], ...] = (),
    review: bool = False,
) -> Diagnostic:
    severity = DATA_QUALITY_ISSUE_SEVERITY.get(code)
    if severity is not None:
        blocking = severity == DATA_QUALITY_SEVERITY_BLOCKING
        review = severity == DATA_QUALITY_SEVERITY_REVIEW
        level = severity
    resolved_code = diagnostic_code or (
        DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING
        if blocking or severity == DATA_QUALITY_SEVERITY_FATAL
        else DIAGNOSTIC_CODE_DATA_QUALITY_ADVISORY
    )
    details: list[tuple[str, str]] = [("location", location), ("issue_code", code)]
    details.extend(extra_details)
    if guidance:
        details.append(("guidance", guidance))
    if severity is not None:
        details.append(("severity", severity))
    details.append(("blocking", "yes" if blocking else "no"))
    details.append(("review", "yes" if review else "no"))
    return Diagnostic(
        stage="data_quality",
        level=level,
        code=resolved_code,
        message=message,
        details=tuple(details),
        phase="data_quality",
    )


def _issue_requires_review(issue: Diagnostic) -> bool:
    if _issue_severity(issue) == DATA_QUALITY_SEVERITY_REVIEW:
        return True
    for key, value in issue.details:
        if key == "review":
            return value == "yes"
    return False


def _issue_is_blocking(issue: Diagnostic) -> bool:
    if _issue_severity(issue) == DATA_QUALITY_SEVERITY_BLOCKING:
        return True
    for key, value in issue.details:
        if key == "blocking":
            return value == "yes"
    return issue.code == DIAGNOSTIC_CODE_DATA_QUALITY_BLOCKING


def _build_narrative(paths: Sequence[Path], issues: Sequence[Diagnostic], *, ok: bool) -> str:
    blocking = sum(1 for issue in issues if _issue_is_blocking(issue))
    if ok:
        return f"Validated {len(paths)} upload source(s); no blocking data-quality issues found."
    return (
        f"Validated {len(paths)} upload source(s); found {blocking} blocking data-quality issue(s) "
        "that need attention before loading."
    )


def _upload_file_content_seed(path: Path) -> int:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return int(digest[:8], 16)


def _relation_column_name_for_label(relation: PreparedRelation, label: str) -> str | None:
    labels = relation.original_column_labels
    names = relation.columns
    for idx, header_label in enumerate(labels):
        if header_label == label:
            return names[idx]
    if label in names:
        return label
    return None


def _relation_label_for_column_name(relation: PreparedRelation, column_name: str) -> str:
    names = relation.columns
    labels = relation.original_column_labels
    for idx, name in enumerate(names):
        if name == column_name:
            return labels[idx]
    return column_name


def _relation_column_values(relation: PreparedRelation, column_name: str) -> list[str]:
    return [str(row.get(column_name, "")) for row in relation.rows]


def _column_distinct_values(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    observed: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in observed:
            continue
        observed.add(text)
        seen.append(text)
    return tuple(seen)


def _build_upload_sample_payload(
    grid: SheetGrid,
    path: Path,
    *,
    max_rows: int,
    include_distinct_values: bool,
) -> dict[str, Any]:
    header_labels = list(grid.cells[0]) if grid.cells else []
    data_rows = list(grid.cells[1:])
    seed = _upload_file_content_seed(path)
    rng = random.Random(seed)
    indices = list(range(len(data_rows)))
    rng.shuffle(indices)
    chosen = indices[:max_rows]
    sample_rows: list[dict[str, str]] = []
    for row_idx in chosen:
        row = _pad_row(data_rows[row_idx], len(header_labels))
        sample_rows.append({header_labels[col_idx]: row[col_idx] for col_idx in range(len(header_labels))})
    payload: dict[str, Any] = {
        "header_labels": header_labels,
        "sample_rows": sample_rows,
        "location": _grid_location_prefix(grid),
    }
    if include_distinct_values:
        column_distinct_values: dict[str, list[str]] = {}
        for col_idx, label in enumerate(header_labels):
            values = []
            for row in data_rows:
                padded = _pad_row(row, len(header_labels))
                value = padded[col_idx] if col_idx < len(padded) else ""
                if str(value).strip():
                    values.append(str(value))
            distinct = _column_distinct_values(values)
            if distinct and len(distinct) <= UPLOAD_BAND_VALUE_MAP_MAX_DISTINCT:
                column_distinct_values[label] = list(distinct)
        payload["column_distinct_values"] = column_distinct_values
    return payload


def _build_upload_column_transform_payload(grid: SheetGrid, path: Path) -> dict[str, Any]:
    payload = _build_upload_sample_payload(
        grid,
        path,
        max_rows=UPLOAD_SAMPLE_MAX_ROWS,
        include_distinct_values=True,
    )
    payload["upload_transform_ids"] = list(UPLOAD_COLUMN_TRANSFORM_IDS)
    payload["column_transforms_schema"] = UPLOAD_COLUMN_TRANSFORMS_SCHEMA
    return payload


def _build_upload_interpret_payload(grid: SheetGrid, path: Path) -> dict[str, Any]:
    payload = _build_upload_sample_payload(
        grid,
        path,
        max_rows=UPLOAD_INTERPRET_MAX_ROWS,
        include_distinct_values=False,
    )
    payload["upload_interpret_schema"] = UPLOAD_INTERPRET_SCHEMA
    return payload


def _llm_upload_interpret(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = stable_json(dict(payload))
    try:
        response = LLMProvider.json(UPLOAD_INTERPRET_SYSTEM, body, retries=0, task="upload_interpret")
    except Exception as exc:
        debug(f"[data_quality._llm_upload_interpret] llm unavailable: {exc!r}")
        return {}
    out: dict[str, Any] = {}
    header_row = response.get("header_row")
    if header_row is not None:
        out["header_row"] = int(header_row)
    table_range = str(response.get("table_range", "")).strip()
    if table_range:
        out["table_range"] = table_range
    append_raw = response.get("append_regions")
    if isinstance(append_raw, Sequence) and not isinstance(append_raw, (str, bytes)):
        append_regions = [str(item).strip() for item in append_raw if str(item).strip()]
        if append_regions:
            out["append_regions"] = append_regions
    merge_raw = response.get("merge_regions")
    if isinstance(merge_raw, Sequence) and not isinstance(merge_raw, (str, bytes)):
        merge_regions = [str(item).strip() for item in merge_raw if str(item).strip()]
        if merge_regions:
            out["merge_regions"] = merge_regions
    return out


def _normalize_column_transform(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    transform_id = str(raw.get("transform_id", "")).strip()
    if transform_id not in UPLOAD_COLUMN_TRANSFORM_IDS:
        return None
    params_raw = raw.get("params", {})
    params = dict(params_raw) if isinstance(params_raw, Mapping) else {}
    requires_review = bool(raw.get("requires_review", False))
    if transform_id in REVIEW_GATED_UPLOAD_COLUMN_TRANSFORMS:
        requires_review = True
    normalized: dict[str, Any] = {
        "transform_id": transform_id,
        "requires_review": requires_review,
        "params": params,
    }
    column = str(raw.get("column", "")).strip()
    if column:
        normalized["column"] = column
    return normalized


def _llm_upload_column_transforms(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    body = stable_json(dict(payload))
    try:
        response = LLMProvider.json(
            UPLOAD_COLUMN_TRANSFORMS_SYSTEM,
            body,
            retries=0,
            task="upload_column_transforms",
        )
    except Exception as exc:
        debug(f"[data_quality._llm_upload_column_transforms] llm unavailable: {exc!r}")
        return []
    raw_transforms = response.get("column_transforms", [])
    if not isinstance(raw_transforms, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_transforms:
        if not isinstance(item, Mapping):
            continue
        normalized = _normalize_column_transform(item)
        if normalized is not None:
            out.append(normalized)
    return out


def _llm_upload_summary_narrative(
    paths: Sequence[Path],
    issues: Sequence[Diagnostic],
    suggested: Mapping[str, Mapping[str, Any]],
    *,
    ok: bool,
    fallback: str,
) -> str:
    issue_rows: list[dict[str, str]] = []
    for issue in issues:
        issue_rows.append(
            {
                "code": issue.code,
                "message": issue.message,
                "level": issue.level,
            }
        )
    payload = stable_json(
        {
            "paths": [path.name for path in paths],
            "ok": ok,
            "issues": issue_rows,
            "suggested_selections": dict(suggested),
            "upload_summary_schema": UPLOAD_SUMMARY_SCHEMA,
        }
    )
    try:
        response = LLMProvider.json(UPLOAD_SUMMARY_SYSTEM, payload, retries=0, task="upload_summary")
    except Exception as exc:
        debug(f"[data_quality._llm_upload_summary_narrative] llm unavailable: {exc!r}")
        return fallback
    summary = str(response.get("summary", "")).strip()
    return summary or fallback


def _enrich_suggested_selections_with_column_transforms(
    paths: Sequence[Path],
    selections: Mapping[str, CsvSourceSelection],
    suggested: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out = dict(suggested)
    for path in paths:
        selection = selections.get(path.name)
        if selection and selection.column_transforms:
            continue
        try:
            grids = load_source_grids(path, selection=selection)
        except ValueError:
            continue
        for grid in grids:
            normalized = normalize_grid(grid, normalize_cell_newlines=True)
            working = normalized
            working, _ = apply_structural_fixes(working)
            region_iter, region_issues = _regions_for_grid(working, selection)
            if any(_issue_is_blocking(issue) for issue in region_issues):
                continue
            for region_grid, _region_index, _region_count in region_iter:
                issues = detect_grid_issues(region_grid)
                if any(_issue_is_blocking(issue) for issue in issues):
                    continue
                entry = dict(out.get(path.name, {}))
                if PolicyConfig.TABULAR_LLM_ASSIST and selection is None:
                    interpret_payload = _build_upload_interpret_payload(region_grid, path)
                    interpret_hints = _llm_upload_interpret(interpret_payload)
                    for key, value in interpret_hints.items():
                        entry.setdefault(key, value)
                payload = _build_upload_column_transform_payload(region_grid, path)
                proposals = _llm_upload_column_transforms(payload)
                review_transforms = [item for item in proposals if item.get("requires_review")]
                if review_transforms:
                    entry["column_transforms"] = review_transforms
                if entry:
                    out[path.name] = entry
    return out


def _notify_transform_rejected(transform_id: str, relation: PreparedRelation, reason: str) -> None:
    notify(
        f"Rejected upload column transform {transform_id!r}: {reason}",
        stage="data_quality",
        code=DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_REJECTED,
        level="info",
        details=(
            ("transform_id", transform_id),
            ("relation", relation.relation_name),
            ("reason", reason),
        ),
    )


def _notify_transform_applied(transform_id: str, relation: PreparedRelation) -> None:
    notify(
        f"Applied upload column transform {transform_id!r} on relation {relation.relation_name!r}.",
        stage="data_quality",
        code=DIAGNOSTIC_CODE_UPLOAD_TRANSFORM_APPLIED,
        level="info",
        details=(
            ("transform_id", transform_id),
            ("relation", relation.relation_name),
        ),
    )


def _verify_proposed_affix_strip(values: Sequence[str], affix_token: str) -> bool:
    if not affix_token:
        return False
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if _is_bare_scalar_number(text):
            continue
        if text.startswith(affix_token):
            remainder = text[len(affix_token) :].strip().replace(",", "")
            if not _is_bare_scalar_number(remainder):
                return False
            continue
        return False
    return True


def _apply_proposed_affix_strip(value: str, affix_token: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if _is_bare_scalar_number(text):
        return text.replace(",", "")
    if text.startswith(affix_token):
        remainder = text[len(affix_token) :].strip().replace(",", "")
        return remainder
    return value


def _verify_band_value_map(values: Sequence[str], value_map: Mapping[str, str]) -> bool:
    if not value_map:
        return False
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in value_map:
            return False
    return True


def _verify_keep_canonical_columns(
    relation: PreparedRelation, canonical_label: str, alias_labels: Sequence[str]
) -> bool:
    canonical_name = _relation_column_name_for_label(relation, canonical_label)
    if canonical_name is None:
        return False
    alias_names: list[str] = []
    for label in alias_labels:
        name = _relation_column_name_for_label(relation, label)
        if name is None:
            return False
        alias_names.append(name)
    canon_to_alias: dict[str, str] = {}
    alias_to_canon: dict[str, str] = {}
    for row in relation.rows:
        canonical_value = str(row.get(canonical_name, "")).strip()
        for alias_name in alias_names:
            alias_value = str(row.get(alias_name, "")).strip()
            if not canonical_value and not alias_value:
                continue
            if canonical_value in canon_to_alias and canon_to_alias[canonical_value] != alias_value:
                return False
            canon_to_alias[canonical_value] = alias_value
            if alias_value in alias_to_canon and alias_to_canon[alias_value] != canonical_value:
                return False
            alias_to_canon[alias_value] = canonical_value
    return True


def _verify_derive_by_pattern(relation: PreparedRelation, params: Mapping[str, Any]) -> bool:
    source_label = str(params.get("source_column", "")).strip()
    target_label = str(params.get("target_column", "")).strip()
    pattern_text = str(params.get("pattern", "")).strip()
    if not source_label or not target_label or not pattern_text:
        return False
    source_name = _relation_column_name_for_label(relation, source_label)
    target_name = _relation_column_name_for_label(relation, target_label)
    if source_name is None or target_name is None:
        return False
    try:
        pattern = re.compile(pattern_text)
    except re.error:
        return False
    for row in relation.rows:
        source_value = str(row.get(source_name, "")).strip()
        if not source_value:
            continue
        match = pattern.search(source_value)
        if match is None:
            return False
        if match.lastindex:
            derived = match.group(1)
        else:
            derived = match.group(0)
        if not str(derived).strip():
            return False
    return True


def _verify_null_tokens(values: Sequence[str], tokens: Sequence[str]) -> bool:
    token_set = {str(token).strip().lower() for token in tokens if str(token).strip()}
    if not token_set:
        return False
    return True


def _verify_drop_empty_columns(relation: PreparedRelation, labels: Sequence[str]) -> bool:
    for label in labels:
        column_name = _relation_column_name_for_label(relation, label)
        if column_name is None:
            return False
        if any(str(row.get(column_name, "")).strip() for row in relation.rows):
            return False
    return bool(labels)


def _verify_parse_temporal(values: Sequence[str], date_format: str) -> bool:
    if not date_format:
        return False
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            datetime.strptime(text, date_format)
        except ValueError:
            return False
    return True


def _verify_column_transform(relation: PreparedRelation, transform: Mapping[str, Any]) -> bool:
    transform_id = str(transform.get("transform_id", ""))
    params = transform.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    column_label = str(transform.get("column", "")).strip()
    if transform_id == UploadColumnTransformId.STRIP_NUMERIC_AFFIX.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return False
        affix_token = str(params.get("affix_token", ""))
        return _verify_proposed_affix_strip(_relation_column_values(relation, column_name), affix_token)
    if transform_id == UploadColumnTransformId.BAND_VALUE_MAP.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return False
        value_map_raw = params.get("value_map", {})
        if not isinstance(value_map_raw, Mapping):
            return False
        value_map = {str(key): str(value) for key, value in value_map_raw.items()}
        distinct_count = len(_column_distinct_values(_relation_column_values(relation, column_name)))
        if distinct_count > UPLOAD_BAND_VALUE_MAP_MAX_DISTINCT:
            return False
        return _verify_band_value_map(_relation_column_values(relation, column_name), value_map)
    if transform_id == UploadColumnTransformId.KEEP_CANONICAL_COLUMNS.value:
        canonical_label = str(params.get("canonical_column", "")).strip()
        alias_raw = params.get("alias_columns", ())
        alias_labels = (
            [str(item).strip() for item in alias_raw]
            if isinstance(alias_raw, Sequence) and not isinstance(alias_raw, (str, bytes))
            else []
        )
        return _verify_keep_canonical_columns(relation, canonical_label, alias_labels)
    if transform_id == UploadColumnTransformId.DERIVE_BY_PATTERN.value:
        return _verify_derive_by_pattern(relation, params)
    if transform_id == UploadColumnTransformId.NULL_TOKENS.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return False
        tokens_raw = params.get("tokens", ())
        tokens = (
            [str(item) for item in tokens_raw]
            if isinstance(tokens_raw, Sequence) and not isinstance(tokens_raw, (str, bytes))
            else []
        )
        return _verify_null_tokens(_relation_column_values(relation, column_name), tokens)
    if transform_id == UploadColumnTransformId.DROP_EMPTY_COLUMNS.value:
        labels_raw = params.get("columns", ())
        labels = (
            [str(item).strip() for item in labels_raw]
            if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes))
            else []
        )
        return _verify_drop_empty_columns(relation, labels)
    if transform_id == UploadColumnTransformId.PARSE_TEMPORAL.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return False
        date_format = str(params.get("date_format", "")).strip()
        return _verify_parse_temporal(_relation_column_values(relation, column_name), date_format)
    return False


def _apply_column_transform(relation: PreparedRelation, transform: Mapping[str, Any]) -> PreparedRelation:
    transform_id = str(transform.get("transform_id", ""))
    params = transform.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    column_label = str(transform.get("column", "")).strip()
    rows = [dict(row) for row in relation.rows]
    columns = list(relation.columns)
    labels = list(relation.original_column_labels)
    types = list(relation.column_types)
    unit_labels = list(relation.column_unit_labels)
    name_to_index = {name: idx for idx, name in enumerate(columns)}

    if transform_id == UploadColumnTransformId.STRIP_NUMERIC_AFFIX.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return relation
        affix_token = str(params.get("affix_token", ""))
        for row in rows:
            row[column_name] = _apply_proposed_affix_strip(row.get(column_name, ""), affix_token)
        col_idx = name_to_index[column_name]
        samples = [str(row[column_name]) for row in rows if str(row[column_name]).strip()]
        types[col_idx] = _infer_numeric_duckdb_type(samples)
        unit_labels[col_idx] = affix_token
    elif transform_id == UploadColumnTransformId.BAND_VALUE_MAP.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return relation
        value_map_raw = params.get("value_map", {})
        value_map = (
            {str(key): str(value) for key, value in value_map_raw.items()} if isinstance(value_map_raw, Mapping) else {}
        )
        for row in rows:
            raw = str(row.get(column_name, "")).strip()
            row[column_name] = value_map.get(raw, raw)
    elif transform_id == UploadColumnTransformId.KEEP_CANONICAL_COLUMNS.value:
        _canonical_label = str(params.get("canonical_column", "")).strip()
        alias_raw = params.get("alias_columns", ())
        alias_labels = (
            [str(item).strip() for item in alias_raw]
            if isinstance(alias_raw, Sequence) and not isinstance(alias_raw, (str, bytes))
            else []
        )
        drop_names = []
        for alias_label in alias_labels:
            alias_name = _relation_column_name_for_label(relation, alias_label)
            if alias_name is not None:
                drop_names.append(alias_name)
        keep_indices = [idx for idx, name in enumerate(columns) if name not in drop_names]
        columns = [columns[idx] for idx in keep_indices]
        labels = [labels[idx] for idx in keep_indices]
        types = [types[idx] for idx in keep_indices]
        unit_labels = [unit_labels[idx] for idx in keep_indices]
        for row in rows:
            for drop_name in drop_names:
                row.pop(drop_name, None)
    elif transform_id == UploadColumnTransformId.DERIVE_BY_PATTERN.value:
        source_label = str(params.get("source_column", "")).strip()
        target_label = str(params.get("target_column", "")).strip()
        pattern_text = str(params.get("pattern", "")).strip()
        source_name = _relation_column_name_for_label(relation, source_label)
        target_name = _relation_column_name_for_label(relation, target_label)
        if source_name is None or target_name is None:
            return relation
        pattern = re.compile(pattern_text)
        for row in rows:
            source_value = str(row.get(source_name, "")).strip()
            if not source_value:
                row[target_name] = ""
                continue
            match = pattern.search(source_value)
            if match is None:
                continue
            row[target_name] = match.group(1) if match.lastindex else match.group(0)
    elif transform_id == UploadColumnTransformId.NULL_TOKENS.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return relation
        tokens_raw = params.get("tokens", ())
        tokens = (
            {str(item).strip().lower() for item in tokens_raw if str(item).strip()}
            if isinstance(tokens_raw, Sequence) and not isinstance(tokens_raw, (str, bytes))
            else set()
        )
        for row in rows:
            raw = str(row.get(column_name, "")).strip()
            if raw.lower() in tokens:
                row[column_name] = ""
    elif transform_id == UploadColumnTransformId.DROP_EMPTY_COLUMNS.value:
        labels_raw = params.get("columns", ())
        drop_labels = (
            [str(item).strip() for item in labels_raw]
            if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes))
            else []
        )
        drop_names = []
        for label in drop_labels:
            name = _relation_column_name_for_label(relation, label)
            if name is not None:
                drop_names.append(name)
        keep_indices = [idx for idx, name in enumerate(columns) if name not in drop_names]
        columns = [columns[idx] for idx in keep_indices]
        labels = [labels[idx] for idx in keep_indices]
        types = [types[idx] for idx in keep_indices]
        unit_labels = [unit_labels[idx] for idx in keep_indices]
        for row in rows:
            for drop_name in drop_names:
                row.pop(drop_name, None)
    elif transform_id == UploadColumnTransformId.PARSE_TEMPORAL.value:
        column_name = _relation_column_name_for_label(relation, column_label)
        if column_name is None:
            return relation
        date_format = str(params.get("date_format", "")).strip()
        col_idx = name_to_index[column_name]
        for row in rows:
            raw = str(row.get(column_name, "")).strip()
            if not raw:
                continue
            parsed = datetime.strptime(raw, date_format)
            if date_format.endswith("%H:%M:%S") or "%H" in date_format:
                row[column_name] = parsed.strftime("%Y-%m-%dT%H:%M:%S")
                types[col_idx] = "TIMESTAMP"
            else:
                row[column_name] = parsed.date().isoformat()
                types[col_idx] = "DATE"

    return PreparedRelation(
        relation_name=relation.relation_name,
        source_path=relation.source_path,
        sheet_name=relation.sheet_name,
        original_table_label=relation.original_table_label,
        columns=tuple(columns),
        original_column_labels=tuple(labels),
        column_types=tuple(types),
        column_unit_labels=tuple(unit_labels),
        rows=tuple(rows),
    )


def _apply_column_transforms_on_relation(
    relation: PreparedRelation,
    transforms: Sequence[Mapping[str, Any]],
) -> PreparedRelation:
    current = relation
    for transform in transforms:
        transform_id = str(transform.get("transform_id", ""))
        if not _verify_column_transform(current, transform):
            _notify_transform_rejected(transform_id, current, "full-column verification failed")
            continue
        current = _apply_column_transform(current, transform)
        _notify_transform_applied(transform_id, current)
    return current


def _pending_column_transforms_for_relation(
    region_grid: SheetGrid,
    path: Path,
    selection: CsvSourceSelection | None,
) -> list[dict[str, Any]]:
    if selection and selection.column_transforms:
        out: list[dict[str, Any]] = []
        for item in selection.column_transforms:
            normalized = _normalize_column_transform(item)
            if normalized is not None:
                out.append(normalized)
        return out
    if not PolicyConfig.TABULAR_LLM_ASSIST:
        return []
    payload = _build_upload_column_transform_payload(region_grid, path)
    proposals = _llm_upload_column_transforms(payload)
    return [item for item in proposals if not item.get("requires_review")]


def _finalize_prepared_relation(
    relation: PreparedRelation,
    *,
    region_grid: SheetGrid,
    selection: CsvSourceSelection | None,
) -> PreparedRelation:
    transforms = _pending_column_transforms_for_relation(region_grid, relation.source_path, selection)
    if not transforms:
        return relation
    return _apply_column_transforms_on_relation(relation, transforms)


def _infer_upload_suffix(data: bytes, filename: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in {".csv", ".xlsx"}:
            return suffix
    if data.startswith(b"PK\x03\x04"):
        return ".xlsx"
    return ".csv"


def _upload_engine_limits() -> EngineLimits:
    try:
        return active_engine_limits()
    except RuntimeError:
        return EngineLimits()


def _enforce_upload_byte_limit(size: int) -> None:
    max_bytes = _upload_engine_limits().max_upload_bytes
    if size > max_bytes:
        raise ConfigError(f"upload size {size} bytes exceeds max_upload_bytes ({max_bytes})")


def _read_upload_source_bytes(source: IO[bytes] | IO[str]) -> bytes:
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except (OSError, io.UnsupportedOperation):
            pass
    payload = source.read()
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = bytes(payload)
    _enforce_upload_byte_limit(len(data))
    return data


def _upload_display_name(data: bytes, filename: str | None) -> str:
    suffix = _infer_upload_suffix(data, filename)
    if filename:
        display_name = Path(filename).name
        if not display_name:
            display_name = f"upload{suffix}"
    else:
        display_name = f"upload{suffix}"
    if not display_name.lower().endswith(suffix):
        display_name = f"{Path(display_name).stem}{suffix}"
    return display_name


def _materialize_upload_bytes(data: bytes, filename: str | None) -> tuple[Path, Path]:
    _enforce_upload_byte_limit(len(data))
    display_name = _upload_display_name(data, filename)
    temp_dir = Path(tempfile.mkdtemp(prefix=".aetherdialect_upload_"))
    temp_path = temp_dir / display_name
    temp_path.write_bytes(data)
    return temp_path, temp_dir


def _resolve_tabular_upload_path(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview | IO[bytes] | IO[str],
    *,
    filename: str | None = None,
) -> tuple[Path, Path | None]:
    """Return ``(resolved_path, temp_dir_to_cleanup)`` for one upload source."""
    if isinstance(source, (str, os.PathLike)):
        return Path(os.fspath(source)), None
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _materialize_upload_bytes(bytes(source), filename)
    if hasattr(source, "read"):
        upload_name = filename
        if upload_name is None:
            obj_name = getattr(source, "name", None)
            if isinstance(obj_name, str) and obj_name and obj_name not in {"<stdin>", "<stdout>", "<stderr>"}:
                upload_name = Path(obj_name).name
        return _materialize_upload_bytes(_read_upload_source_bytes(source), upload_name)
    raise TypeError(
        "inspect_tabular_upload expects a path, bytes, or a readable file-like object",
    )


def inspect_tabular_upload(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview | IO[bytes] | IO[str],
    *,
    filename: str | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> DataQualityReport:
    """Inspect one CSV or Excel upload without constructing an engine."""
    resolved, temp_dir = _resolve_tabular_upload_path(source, filename=filename)
    try:
        report = validate_upload_sources((resolved,), log_sink=log_sink or (lambda _msg: None))
        for issue in report.issues:
            if _issue_severity(issue) == DATA_QUALITY_SEVERITY_FATAL:
                raise ConfigError(issue.message)
        return report
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _suggested_selections_from_issues(
    paths: Sequence[Path],
    issues: Sequence[Diagnostic],
) -> dict[str, dict[str, Any]]:
    """Build per-filename interpretation hints from review and advisory issues."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        filename = path.name
        selection: dict[str, Any] = {}
        for issue in issues:
            location = next((value for key, value in issue.details if key == "location"), "")
            if not str(location).startswith(filename):
                continue
            for key, value in issue.details:
                if key == DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW and value:
                    selection["header_row"] = int(value)
                elif key == DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE and value:
                    selection["table_range"] = value
        if selection:
            out[filename] = selection
    return out


def _deterministic_identifier(label: str) -> str:
    text = label.strip().lower()
    out: list[str] = []
    prev_underscore = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    candidate = "".join(out).strip("_")
    if not candidate:
        candidate = "item"
    if candidate[0].isdigit():
        candidate = f"c_{candidate}"
    if len(candidate) > 120:
        candidate = candidate[:120].rstrip("_")
    return candidate


def _dedupe_identifier(base: str, reserved: set[str]) -> str:
    candidate = base or "item"
    if candidate not in reserved:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in reserved:
        suffix += 1
    return f"{candidate}_{suffix}"


def _llm_identifier(label: str, *, kind: str) -> str:
    payload = stable_json(
        {
            "task": f"Propose one SQL identifier for this {kind} label.",
            "label": label,
            "identifier_naming_schema": CSV_IDENTIFIER_NAMING_SCHEMA,
        }
    )
    try:
        response = LLMProvider.json(CSV_IDENTIFIER_NAMING_SYSTEM, payload, retries=0, task="default")
    except Exception as exc:
        debug(f"[data_quality._llm_identifier] llm unavailable: {exc!r}")
        return ""
    ident = str(response.get("identifier", "")).strip()
    if _valid_identifier(ident):
        return ident
    return ""


def _workbook_has_multiple_sheets(path: Path) -> bool:
    if path.suffix.lower() != ".xlsx":
        return False
    require_driver("csv")
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.load_workbook(path, read_only=True)
    try:
        return len(workbook.sheetnames) > 1
    finally:
        workbook.close()


def _valid_identifier(text: str) -> bool:
    if not text or not ("a" <= text[0] <= "z"):
        return False
    for ch in text[1:]:
        if not (ch.isdigit() or ch == "_"):
            return False
    return True
