"""Live-test LLM cost invoice writer (not shipped with the package)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import LlmUsageRecord
from aetherdialect._core_utils import (
    llm_call_cost_usd,
    llm_price_table_as_of,
    snapshot_llm_usage_records,
)

_INVOICE_PATH = Path(__file__).parent / "invoice.txt"


def invoice_path() -> Path:
    """Return the live-test invoice output path."""
    return _INVOICE_PATH


def clear_invoice_file() -> None:
    """Truncate the invoice file at session start."""
    _INVOICE_PATH.write_text("", encoding="utf-8")


def _format_call_line(record: LlmUsageRecord) -> str:
    cost = llm_call_cost_usd(record)
    cost_part = f" cost=${cost:.6f}" if cost is not None else ""
    if record.provider == "openai" and cost is None:
        cost_part = f" unpriced={record.logical_model}"
    return (
        f"  {record.logical_model} task={record.task} "
        f"in={record.input_tokens} cached={record.cached_input_tokens} "
        f"out={record.output_tokens}{cost_part}"
    )


def _block_total_line(records: Sequence[LlmUsageRecord]) -> str:
    in_tok = sum(r.input_tokens for r in records)
    cached = sum(r.cached_input_tokens for r in records)
    out_tok = sum(r.output_tokens for r in records)
    costs = [c for r in records if (c := llm_call_cost_usd(r)) is not None]
    cost_part = f" cost=${sum(costs):.6f}" if costs else ""
    return f"  total in={in_tok} cached={cached} out={out_tok}{cost_part}"


def write_invoice_file(records: Sequence[LlmUsageRecord] | None = None) -> None:
    """Write per-call lines, per-block totals, and a run summary to ``invoice.txt``."""
    rows = list(records if records is not None else snapshot_llm_usage_records())
    lines: list[str] = []
    provider = EngineConfig.LLM_PROVIDER
    if provider == "openai":
        lines.append(f"price_table_as_of={llm_price_table_as_of()}")
    lines.append(f"provider={provider}")
    lines.append("")

    blocks: list[tuple[str, int, list[LlmUsageRecord]]] = []
    for record in rows:
        if blocks and blocks[-1][0] == record.scope and blocks[-1][1] == record.block_id:
            blocks[-1][2].append(record)
        else:
            blocks.append((record.scope, record.block_id, [record]))

    build_blocks = [b for b in blocks if b[0] == "build"]
    question_blocks = [b for b in blocks if b[0] == "question"]
    run_blocks = [b for b in blocks if b[0] == "run"]

    def _write_block(title: str, block: Sequence[LlmUsageRecord]) -> None:
        if not block:
            return
        lines.append(f"[{title}]")
        for record in block:
            lines.append(_format_call_line(record))
        lines.append(_block_total_line(block))
        lines.append("")

    for idx, (_scope, _block_id, block) in enumerate(build_blocks, start=1):
        _write_block(f"build_{idx}", block)
    for idx, (_scope, _block_id, block) in enumerate(question_blocks, start=1):
        _write_block(f"question_{idx}", block)
    for idx, (_scope, _block_id, block) in enumerate(run_blocks, start=1):
        _write_block(f"run_{idx}", block)

    build_records = [r for _s, _b, block in build_blocks for r in block]
    question_records = [r for _s, _b, block in question_blocks for r in block]
    run_records = [r for _s, _b, block in run_blocks for r in block]

    build_cost = sum(c for r in build_records if (c := llm_call_cost_usd(r)) is not None)
    question_cost = sum(c for r in question_records if (c := llm_call_cost_usd(r)) is not None)
    run_cost = sum(c for r in run_records if (c := llm_call_cost_usd(r)) is not None)
    total_cost = build_cost + question_cost + run_cost

    unpriced = sorted({r.logical_model for r in rows if r.provider == "openai" and llm_call_cost_usd(r) is None})

    lines.append("[run_total]")
    lines.append(f"  build_cost=${build_cost:.6f}")
    lines.append(f"  question_cost=${question_cost:.6f} questions={len(question_blocks)}")
    if run_records:
        lines.append(f"  other_cost=${run_cost:.6f}")
    if provider == "openai":
        lines.append(f"  total_cost=${total_cost:.6f}")
    if unpriced:
        lines.append(f"  unpriced_models={','.join(unpriced)}")
    lines.append("  note=reported totals are a floor; failed retries and batch calls carry no usage")

    _INVOICE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
