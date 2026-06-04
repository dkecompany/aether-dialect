# User guide

**Reading order:** this guide → [Integrator guide](INTEGRATOR_GUIDE.md) → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

This guide is for analysts and operators: install the package, configure credentials, understand scope and overrides, ask questions, handle migrations and warmup, and avoid common mistakes. Programmatic embedding (sessions, threading, the write queue) lives in the [Integrator guide](INTEGRATOR_GUIDE.md). Types and every configuration key are in the [API reference](API_REFERENCE.md).

## Install

```bash
pip install aetherdialect
pip install "aetherdialect[postgresql]"
pip install "aetherdialect[databricks]"
pip install "aetherdialect[postgresql,databricks]"
```

Requires Python 3.10 or newer. The PyPI package name and import name are both **`aetherdialect`**.

## Configure

Point your application at a single TOML file via `Text2SQL(..., config_file=...)`. The file carries LLM credentials, engine selection (`postgresql` or `databricks`), database connection fields, and execution limits. When you pass `config_file`, every mapped field that appears in the file is authoritative for that environment key (empty assignments clear inherited shell values); fields you omit still read from `os.environ`. When you omit `config_file`, settings come only from `os.environ`. The library never mutates `os.environ` during reads. OpenAI model routing uses internal defaults; Azure OpenAI still uses the `[azure_openai.deployments]` names in TOML (see the [API reference](API_REFERENCE.md)). The full flattening rules and key names are listed there.

Minimal illustration (placeholders only):

```toml
[engine]
selected = "postgresql"

[postgresql]
host = "localhost"
port = "5432"
database = "mydb"
user = "readonly"
password = "REPLACE_ME"

[openai]
api_key = "REPLACE_ME"
```

When PostgreSQL and Databricks are both fully configured with usable drivers, set `AETHERDIALECT_ENGINE` or TOML `[engine] selected` to `postgresql` or `databricks`. When both OpenAI and Azure OpenAI credentials are present, set `AETHERDIALECT_LLM_PROVIDER` or TOML `[llm] provider` to `openai` or `azure`.

## First run

Constructing the `Text2SQL` engine for the first time against a database can take several seconds. It reflects the catalog, profiles every column in scope, classifies roles, and writes a snapshot under the engine storage directory. Later constructions reuse that snapshot when the live fingerprint still matches; details are in [How it works — Engine storage](HOW_IT_WORKS.md).

## SchemaContext

`SchemaContext` is the frozen scope object you pass into `Text2SQL` (unless you rely on a persisted context loaded from disk). Fields that matter day to day:

- **`include`** — `"tables"`, `"views"`, or `"both"`; controls which relation kinds enter the graph.
- **`allow_objects`** — optional allow-list of relation names; when empty, everything in scope from the catalog is eligible subject to denies.
- **`deny_columns`** — only `table.column` or `*.column` entries; bare column names are rejected. Denied columns are removed from the in-memory graph before profiling and classification, so they never appear as `ColumnMetadata` rows, never receive sensitivity roles, and are absent from LLM-facing schema serializations while the deny remains in effect. The frozen `SchemaContext` and cached scope payload still record the deny list for operators and replay.
- **`allow_columns`** — same qualification rules as `deny_columns`; when non-empty, only listed columns (plus structural PK/FK columns required for joins) survive reflection.
- **`notes_file`** — optional path to a plain-text or Markdown file read as analyst domain notes (see below).
- **`sql_file`** — optional path to DDL or annotated SQL that seeds or constrains how the graph is interpreted.
- **Statistical omission** — columns that fail statistical usability gates (for example extreme null rate, single-value, or sentinel-dominated distributions as implemented in profiling) are omitted entirely from LLM-facing schema literals and cannot be targeted by name in natural-language querying through the normal planner path.

Sensitivity tiers and repair behaviour are in [Sensitivity classification](#sensitivity-classification) below.

## Sensitivity classification

Each column carries a `SensitivityClassification` set by the classifier or analyst overrides (`schema_overrides.json`). Legacy `pii` / `restricted` strings in override JSON normalize into these tiers on load.

| Tier            | LLM visibility                    | Projection and grouping                                                                         | Filters                                                                |
| --------------- | --------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **`none`**      | Full when statistically usable    | Bare `SELECT`, `GROUP BY`, `ORDER BY` allowed                                                   | Allowed                                                                |
| **`hygiene`**   | May appear in prompts when usable | Aggregates and `GROUP BY` allowed; not bare `SELECT`                                            | Allowed                                                                |
| **`strict`**    | Withheld from prompts             | No bare `SELECT`, `GROUP BY`, or `ORDER BY`; aggregates may wrap the column where policy allows | Equality-shaped filters and aggregate-wrapped references where allowed |
| **`forbidden`** | Withheld                          | No bare projection, `GROUP BY`, or `ORDER BY`                                                   | Predicates referencing the column are rejected                         |

**Denied columns** (`deny_columns` on `SchemaContext`) differ from hidden sensitivity: denied columns are removed from the graph entirely and cannot be targeted by name.

**Repair during a turn**

- The engine **strips** non-viable `SELECT` or `GROUP BY` entries when a valid query remains (diagnostics are emitted).
- The engine **terminates** the turn when `WHERE` or `HAVING` still references a forbidden column, or when stripping would remove every projection or every required grouping key for a grouped grain.

Full policy detail is in [Security — Sensitivity](SECURITY.md).

## Asking a question

A turn moves through silent validation gates whenever possible; you only see prompts when the engine needs a human decision.

When the engine is not confident about how to translate your question, it prints a natural-language readback of the SQL it plans to generate and asks for a yes/no. The readback paraphrases the SQL, not the intent JSON. When the engine is confident, this prompt is skipped and you only confirm the final result. The readback re-appears for the same question whenever the system has any prior rejection recorded for that exact question wording (regardless of the rejection reason).

You will also confirm the **final result** (SQL plus preview rows) before the engine treats the answer as accepted learning.

As you accept results, the engine builds confidence on that question. Once you have accepted the same question twice and rejections on this template remain under one-in-four, future turns return the stored SQL with no LLM call and no prompt.

Rejection with a short reason is summarised into bounded categories so later turns on the same question steer away from the same mistake; join-choice hints apply when the failure was about tables or joins.

## Notes file

Set `SchemaContext.notes_file` to a path whose contents are plain text or Markdown (the engine treats the file as text). Useful content:

- One or two sentences per important table describing what it represents.
- Glossary terms, business definitions, and abbreviations.
- Disambiguation rules (“when the analyst says X they mean column Y”).
- Explicit sensitivity statements (for example that a particular email column is treated as personal data).

The engine uses notes to refine descriptions, guide role classification, and drive sensitivity tagging. The classifier sets non-`none` classifications only when the notes are explicit; it does not infer sensitivity from column names alone.

If a cached run already recorded a notes path and you omit `notes_file` on a later run, the cached path is reused. Editing the notes file changes the notes fingerprint without necessarily invalidating templates; see [How it works](HOW_IT_WORKS.md) for how hashes interact.

## Schema overrides

Overrides live in one JSON file beside your working directory. Your integration layer exports a starter file, you edit descriptions, roles, sensitivity, and optional structural hints (added or removed foreign keys, primary-key endorsements), then applies the file through the public API. JSON `null` on `description` or `role` means “no change” for that key: the apply step skips it and prunes the key from the resolved document before the sidecar is written. Invalid role tokens or roles incompatible with the column’s `value_type` are skipped with a diagnostic notify and are also pruned from the persisted document. The editable surface, validation rules, and apply semantics are documented in the [API reference — Schema overrides](API_REFERENCE.md). After a successful apply, the resolved layer is persisted and replayed on every subsequent graph build until you clear it with the documented reset call.

## Migration

Migration is never silent. When the catalog changes structurally, the engine writes `schema_migration_map.json` and stops. After you save your edit, the engine resumes; on success the file is renamed to `schema_migration_map.applied.json` so it cannot be replayed.

Open the skeleton the engine writes. A representative shape:

```jsonc
{
  "version": 1,
  "action": "remap",
  "table_renames": [{ "from": "customer", "to": "customers" }],
  "column_renames": [
    { "table": "customers", "from": "cust_id", "to": "customer_id" },
  ],
  "dropped_tables": [],
  "dropped_columns": [],
  "added_tables": [],
  "added_columns": [],
}
```

`action` must be one of:

- **`remap`** — rewrite identifiers in stored templates and value history while preserving learning where possible.
- **`destructive`** — clear persisted templates and related caches on disk while leaving the database unchanged; overrides survive because the sidecar replays.
- **`abort`** — stop initialisation without applying changes so you can investigate drift first.

When new tables or columns are added, the engine always profiles them and classifies roles. Setting `refresh_existing_descriptions_on_addition` to `true` additionally asks the model to revise the descriptions of unchanged tables so they reflect new cross-table relationships introduced by the addition. The field defaults to `false`; add it beside `action` when you need that behaviour.

If validation fails, `MigrationPendingError` returns with the offending entries; the map file stays in place for correction.

## Seed warmup

Three full warmup paths exist for a fresh engine storage directory so common questions hit the cache sooner. All are invoked through `Text2SQL` methods documented in the [API reference](API_REFERENCE.md):

1. **Seed-question warmup** — a newline-delimited file of natural-language questions; each line is parsed, validated, executed when safe, and folded into templates with paraphrases.
2. **SQL-history warmup** — a newline-delimited file of historical `SELECT` statements; each line is reverse-engineered into the internal intent shape, then the same validation pipeline runs.
3. **Query-log warmup** — the engine reads supported system tables (`pg_stat_statements` on PostgreSQL, `system.query.history` on Databricks) using the same credentials as normal operation.

**Dry-run seed warmup** (`dry_run_warmup`) runs the seed-question file through the same validation and execution checks as `run_seed_warmup`, but skips question-side LLM calls and does not persist new templates; use it for CI or operator smoke checks before a full warmup.

See the [support matrix](SUPPORT_MATRIX.md) for the per-construct table of what we do not generate and how the engine reformulates it.

## Resetting learning

The public API exposes scoped resets: clearing only persisted overrides, only the template store, only simulation caches, or all learning at once. Each method’s side effects are spelled out in the [API reference](API_REFERENCE.md). Templates live under a partitioned `intent_templates/` directory (`header.json.gz` plus `partition_<NN>.json.gz`); `clear_template_store` removes that tree (and any legacy monolithic file) without you deleting the whole engine storage directory by hand.

## Common pitfalls

- **The engine answers against the wrong table.** Improve table and column descriptions, tighten your notes file vocabulary, and ensure the question names the business entity the way your notes describe it. Foreign keys are secondary until the wording matches the model you want.
- **Two related tables are not being joined.** Add or fix the foreign key in the database, or use `foreign_keys_add` / `foreign_keys_remove` in the override JSON so the graph matches reality once and every future question benefits.
- **A column the engine should use never appears.** Check usability signals (constant-, null-, or sentinel-dominated columns may be hidden), deny lists, and sensitivity tags before assuming a bug.
- **A simple question takes too long on first run.** Profiling a large warehouse is intentionally thorough; subsequent runs reuse the cached snapshot. If the snapshot keeps rebuilding, inspect the last migration tier: a destructive tier clears templates and warmup products each time it runs.
- **`MigrationPendingError` keeps returning.** `action` must be `remap`, `destructive`, or `abort`. Invalid JSON or unknown enum values fail validation and leave the file in place for editing.

---

**See also:** [README](../README.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [How it works](HOW_IT_WORKS.md) · [Support matrix](SUPPORT_MATRIX.md) · [Security](SECURITY.md) · [API reference](API_REFERENCE.md)
