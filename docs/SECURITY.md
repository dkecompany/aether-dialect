# Security model

Nothing in the engine generates raw SQL from free-form text. The LLM is asked only to fill bounded slots in a structured intent JSON. That intent is parsed, validated against the schema graph, and only then materialised into SQL by a deterministic generator. Four guards (intent JSON shape, dialect AST validation, schema/catalog alignment, dialect EXPLAIN) sit between the LLM and the database. The forbidden-SQL regex list and the parsed-AST structural validator both run before execution.

This document describes what the engine sees, what it sends to the LLM provider, what it persists on disk, and the threat model behind the design choices. It complements your warehouse's security posture, not replaces it.

**Next:** [How it works](HOW_IT_WORKS.md) · [User guide](USER_GUIDE.md) · [API reference](API_REFERENCE.md) · [Support matrix](SUPPORT_MATRIX.md)

Configuration is built from a merged in-process mapping: without **`config_file`**, a string copy of `os.environ` only. With **`config_file`**, each mapped field present in the TOML is authoritative (non-empty values replace the environment copy; empty values remove that key); fields omitted from the file still read from `os.environ`. Full rules are in the [API reference](API_REFERENCE.md). The library does not load a `.env` file implicitly and does not mutate `os.environ` while reading settings.

## 1. Threat model

The engine is built for the case where:

- A trusted operator configures the engine with database credentials and an LLM provider.
- A semi-trusted user asks natural-language questions; the user is permitted to read the data the credentials authorise but should not be able to bypass the analytical SQL surface.
- The LLM provider sees the prompts the engine sends. Whatever the operator considers acceptable to send to the configured provider must include the schema metadata listed in Section 2 below.
- The local filesystem under the **engine storage directory** is operator-controlled. That directory is always `join(artifacts_parent, "aetherdialect", connection_slug)` where `artifacts_parent` is the absolute expanded `artifacts_dir` argument to `Text2SQL` when set, or the per-user data directory when omitted (see [API reference](API_REFERENCE.md)). The engine treats that directory as trusted storage.

The engine's defences are layered to address three concrete risks:

| Risk                                                     | Mitigation                                                                                                                            |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| User crafts a question that runs arbitrary SQL           | `SELECT`-only enforcement, `FORBIDDEN_SQL` regex list, dialect AST validation, structured intent IR, `EXPLAIN` gate before execution. |
| LLM output escapes the analytical subset                 | Same gates plus validators that reject constructs the IR cannot represent safely.                                                     |
| Sensitive column data leaks through prompts or artifacts | `pii` and `restricted` sensitivity tags, deny lists, `is_visible` gate, top-k caps; no full raw row dumps in artifacts.               |

The engine is **not** designed for the case where the database credentials themselves are untrusted. Database-level security (least-privilege roles, network isolation, audit logging) is the operator's responsibility.

### Persisted versus sent-to-LLM (summary)

| Data class                     | Persisted on disk under the engine storage directory                                                                                                     | Sent to the LLM provider                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Table names                    | yes (`schema_graph.json.gz`, templates)                                                                                                                  | yes (schema literals and prompts)                                      |
| Visible column names           | yes                                                                                                                                                      | yes (post `is_visible` / scope filters)                                |
| Column descriptions / roles    | yes                                                                                                                                                      | yes (merged into prompt-safe schema text)                              |
| Top-k profiling values         | yes, bounded per column                                                                                                                                  | mostly no; rare enum-detection paths may include small heads           |
| Raw database rows              | no                                                                                                                                                       | no                                                                     |
| Query result rows              | no (session-only frames)                                                                                                                                 | no                                                                     |
| Failed SQL                     | no long-term verbatim dump; feedback stores **summarised** text                                                                                          | summarised failure payloads only in the feedback summariser path       |
| Reuse SQL templates            | yes (`intent_templates/` directory: `header.json.gz` plus lazy `partition_*.json.gz` shards; legacy `intent_templates.json.gz` removed on upgrade paths) | parameterised template text for bounded extraction when that path runs |
| Accepted templates (text form) | yes                                                                                                                                                      | yes when prompts include prior accepted shapes                         |
| Runtime config TOML            | yes (operator-owned path)                                                                                                                                | no                                                                     |

Call-site detail is in section 2 below.

## 2. LLM context inventory

What the engine sends to the LLM, by call site. Each entry below is a complete description of the prompt content for that call. The default is "metadata only" — no row data flows through the LLM unless the user puts it into the question or the notes file.

The engine uses **OpenAI-shaped models** with Azure or OpenAI transports. Integrators supply deployments named in configuration. Internal routing from engine tasks to those deployments is not documented here.

### 2.1 Question validation (`validate_question`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** classify input as DB question vs chitchat; allowed `query_type`; typo-only correction guidance; JSON output only.
- **User content:** the raw user question string.
- **Schema sent:** none.
- **Sample data sent:** none.
- **Raw rows sent:** none.

### 2.2 Intent parser (`full_intent_parse` and structural formatter / repairs)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** the deterministic intent-parser or formatter prompt; field specifications; rules; output-format example.
- **User content:** JSON envelope with the question, the prompt-safe schema literal (visible columns: names, types, PK / FK markers, roles, descriptions, filterability flags, **enum head truncated to 10 values**), the allowed table set, and optional prior-question feedback (failure category bucket, summary text, hashes — **not** the raw failed intent payload).
- **Schema sent:** the full prompt-safe schema literal for the visible scope when the call includes schema.
- **Sample data sent:** **enum head only** (≤10 string values for enum-typed columns); no `top_k_values` for non-enum columns.
- **Raw rows sent:** none.

### 2.3 Question normalisation (`normalize_question_via_llm`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** canonicalisation rules (preserve literals, preserve digit order, no expansion, neutral guidance).
- **User content:** a JSON object with the typo-corrected question plus a fixed `normalization_preferences` block (heading plus neutral vocabulary guidance text; no example tokens).
- **Schema sent:** none.
- **Sample data sent:** none.
- **Raw rows sent:** none.
- **Output guard:** `_enforce_normalization_guard` (Jaccard floor on stopword-filtered tokens, word-count cap, digit preservation, capital-token preservation, allowlist for introduced tokens).

### 2.4 Join choice (`get_join_choice_from_llm`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** join selector; JSON output schema; optional block listing prior rejected joins for this question.
- **User content:** the typo-corrected question, the deterministic SQL skeleton, and the candidate join shapes (table sets, signature ids, qualified column endpoints).
- **Schema sent:** indirectly via the join signatures and SQL skeleton; the full schema literal is not re-sent.
- **Sample data sent:** none.
- **Raw rows sent:** none.

### 2.5 Schema role classification (`_llm_classify_schema`, base pass)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** rubric for table roles, column roles, hints, profile-hint usage rules.
- **User content:** per-table envelopes containing `table`, declared FKs, and column profiles. Column profiles include name, data type, PK / FK flags, and **distinct ratio / null ratio**; when `PolicyConfig.SCHEMA_DESCRIPTION_TOP_VALUE_SAMPLES` is greater than zero, qualifying string-like columns with low distinct_ratio may also include a capped `top_values` list sampled from profiling `top_k_values` (columns with sensitivity `pii` or `restricted`, denied columns, or high distinct_ratio never receive `top_values`).
- **Schema sent:** all visible tables and columns.
- **Sample data sent:** **distinct counts and null ratios** always; **optional** small `top_values` lists only when the policy ClassVars above request it and every qualifier passes.
- **Raw rows sent:** none.

### 2.6 Schema role classification (`_llm_classify_schema`, always-on consistency refine pass)

- **Model:** same family as the base pass.
- **System prompt:** merge `domain_notes` into the base classification when notes are configured; otherwise perform the always-on consistency refine pass; tighten descriptions; preserve schema shape; **set `pii` only when notes are explicit** when notes drive the merge.
- **User content:** when notes are present, the base classification JSON plus the **full notes file content** as a string field; when notes are absent, the refine pass still runs on the base classification payload alone (there is no branch that skips this pass while graph build continues).
- **Schema sent:** as in the base pass.
- **Sample data sent:** none in the notes payload itself when notes are absent; when notes are present, whatever the notes author wrote is sent verbatim.
- **Raw rows sent:** none, unless the notes author included row excerpts.

**When a notes file is configured and sent, that payload is the single largest free-form text the engine may send to the LLM for this stage.** Treat every line as provider-visible.

### 2.7 Description refinement (`_refine_descriptions_via_llm`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** polish descriptions; preserve human keywords and identifiers.
- **User content:** a small batch of `{path, kind, text, previous_text}` rows — only entries whose description text actually changed are batched.
- **Sample data sent:** none.

### 2.8 SQL DDL parser (`_parse_sql_file_via_llm` and deterministic DDL path)

- **Model:** OpenAI-compatible deployment configured for this task (when the LLM path runs).
- **System prompt:** deterministic SQL DDL parser; JSON output schema.
- **User content:** the **entire DDL file** content (the `sql_file` argument).
- **Sample data sent:** any literals embedded in the DDL (defaults, comments).

**NOT NULL** and **UNIQUE** are parsed in both deterministic and LLM-backed DDL paths so inferred nullability and constraints stay aligned regardless of which parser produced the row.

If your DDL file contains sensitive identifiers or comments, keep the same security posture you would for a production schema dump.

### 2.9 Question feedback summariser (`summarize_failure_for_memory`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** compress the failure to a small bucket plus a short summary; **do not include raw SQL in the summary**.
- **User content:** the question, the failure kind, the structured intent JSON, the user's free-text reason, optionally the failed SQL, optionally validator error rows.
- **Sample data sent:** any literal values embedded in the intent (filter literals); the SQL when supplied.

### 2.10 Question generation (seed warmup, NL synthesis)

- **Model:** OpenAI-compatible deployments (generation and judge may use different cost bands).
- **System prompt:** generate a realistic NL question from a structured intent or SQL; preserve fidelity.
- **User content:** the intent or SQL, plus the description-enriched schema lines for the involved tables (table and column descriptions, types, roles — explicitly **no top-k value lists** in `schema_context_enriched_lines_for_tables`).
- **Sample data sent:** none.

### 2.11 Paraphrase generation (`generate_paraphrases_of_seed_question`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** rephrase the seed question; preserve entities, filters, and metrics.
- **User content:** the seed question, the description-enriched schema lines for the involved tables, and the style slots.
- **Sample data sent:** none.

### 2.12 QSim intent fill (`_llm_fill_intent`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** skeleton-constrained intent JSON; column whitelist rules; DISTINCT and expression-comparison rules.
- **User content:** `build_schema_context` output (table descriptions plus visible columns: type, PK / FK, filter flag — **no `top_k_values`**), filterable / aggregatable / groupable column lists, skeleton constraints, retry hints.
- **Sample data sent:** none.

### 2.13 Fuzzy-reuse parameter extraction (`extract_fuzzy_reuse_params`)

- **Model:** OpenAI-compatible deployment configured for this task.
- **System prompt:** extract bound parameters for direct reuse.
- **User content:** the parameterised SQL string, the historical matched question, the historical matched parameter values, the current question, the parameter keys, extraction rules.
- **Sample data sent:** the previous parameter values for the matched template.

This is the only call site that intentionally sends prior bound values back to the LLM. The values are user-provided literals that have already passed through the engine's gates; they are not raw row content from the database.

## 3. Disk artifact inventory

Every file the engine writes under the **engine storage directory** (`<artifacts_parent>/aetherdialect/<connection_slug>/`). Nothing here contains raw row data.

| File                                                                        | Contents                                                                                                                                                                                                                      | Sensitivity                                                                                                                                   |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `artifact_manifest.json`                                                    | Six fingerprints plus `last_action`.                                                                                                                                                                                          | Low. Hashes only.                                                                                                                             |
| `schema_graph.json.gz`                                                      | Frozen `SchemaGraph` snapshot: tables, columns, types, declared and inferred PK / FK, roles, descriptions, sensitivity, profiling stats, top-k values for visible columns.                                                    | Medium. Contains `top_k_values` for visible columns (cap `PROFILING_TOP_K = 50`). Treat like a schema dump plus a tiny representative sample. |
| `intent_templates/`                                                         | Partitioned template store: `header.json.gz` (indexes, feedback, partition map) plus `partition_<NN>.json.gz` shards (`NN` in `00`–`ff`). Legacy monolithic `intent_templates.json.gz` is deleted when present during clears. | Medium-high. Captures user-supplied question text and bound literals. Treat as user-input log.                                                |
| `applied_overrides.json`                                                    | Resolved user layer: descriptions, role assignments, sensitivity tags, added or removed PK / FK edges, internal block lists.                                                                                                  | Low-medium. User-authored text plus structural metadata.                                                                                      |
| `schema_context.json`                                                       | Last-known `SchemaContext`: include mode, allow / deny lists, notes file path, sql file path.                                                                                                                                 | Low. Configuration only.                                                                                                                      |
| `write_queue.jsonl`                                                         | Append-only `WriteQueueEvent` records when the reader defers learning mutations.                                                                                                                                              | Medium. May contain question text and template identifiers.                                                                                   |
| `qsim_skeletons.json.gz`, `qsim_summary.json`, `qsim_v*_questions.txt`      | Synthetic skeleton enumerations and generated questions.                                                                                                                                                                      | Low. No raw data.                                                                                                                             |
| `seed_warmup_cache.zip`, `seed_warmup_v*.zip`, `seed_warmup_report_v*.json` | Seed warmup outputs.                                                                                                                                                                                                          | Medium. Captures questions and intent payloads from your seed file.                                                                           |
| `anchor_lattice/*`                                                          | Internal warmup support files.                                                                                                                                                                                                | Low.                                                                                                                                          |

To back up your learning, copy the whole engine storage directory (or the parent you passed as `artifacts_dir`, which contains the `aetherdialect` segment). To reset, remove that directory or point `Text2SQL` at a fresh parent path.

## 4. Sensitivity tags and PII handling

The authoritative behaviour matrix for `SensitivityClassification`, projection, grouping, ordering, predicates, and LLM visibility lives in the [User guide — SchemaContext](USER_GUIDE.md) under the same product headings (no duplicate tables here). This section records the security posture only.

- Column sensitivity is a single enum on `ColumnMetadata` (`none`, `hygiene`, `strict`, `forbidden`). Legacy `pii` / `restricted` strings in persisted JSON normalize at load time; there is no long-lived parallel string tier.
- **Strip versus terminate:** the intent repair pass may drop individual `SELECT` or `GROUP BY` entries that reference hidden-sensitivity columns and emit diagnostics, but semantic validation **terminates** the turn when `WHERE` or `HAVING` still references a column that is not selectable under policy, or when stripping would remove every projection or every grouping key required for a grouped grain.
- **Analyst overrides** on the JSON surface remain `string` or JSON `null` for `description` and `role` only; other keys follow the machine schema in the API reference.

## 5. Deny lists

`SchemaContext.deny_columns` and `allow_columns` accept only qualified `table.column` or `*.column` tokens; bare names are rejected at construction.

Denied columns are **removed from the reflected graph** before profiling and LLM classification: they do not exist as `ColumnMetadata` rows while the deny is active, so they are not profiled, not sensitivity-tagged, and cannot be joined on. The deny specification remains on the frozen `SchemaContext` and in cached scope metadata for operators and replay. `allow_columns` narrows the reflected column set while still retaining PK/FK endpoints needed for join integrity.

## 6. SQL surface (what the engine refuses to generate)

See the [support matrix](SUPPORT_MATRIX.md) for the per-construct table of what we refuse to generate, plus the reformulations the intent parser is instructed to use. The forbidden-SQL regex list, the dialect AST structural validator, and the schema-alignment validator together guarantee that none of those constructs reach execution.

## 7. EXPLAIN-only execution gate

Every generated SQL statement is parsed by the dialect AST validator and then run through `EXPLAIN` (when the dialect supports it) before execution. EXPLAIN failures are classified into structured diagnostics; soft codes (`EXPLAIN_SEQ_SCAN_INDEXED`, `EXPLAIN_ZERO_ESTIMATE`) act as plan-quality hints and do not fail the turn, while hard codes do.

EXPLAIN is a defence-in-depth layer, not the primary one. The structural validator and the regex list catch most security-relevant patterns before EXPLAIN runs.

## 8. Network surface

The engine reaches the LLM and the database only through the merged configuration mapping. There is no opt-out; both endpoints are required for any operation that executes the full pipeline.

- **The database**, using credentials from the merged configuration environment. The driver is selected by engine: `psycopg2-binary` for PostgreSQL, `databricks-sql-connector` (preferred) or PySpark (fallback) for Databricks.
- **The LLM provider**, using credentials from the same merged mapping.

No outbound traffic goes anywhere else for core operation.

## 9. Operational guidance

- Use a least-privilege database role (`SELECT` plus whatever your engine needs for `EXPLAIN`).
- Keep `notes_file` content reviewed; the engine sends it verbatim to the LLM during the schema classification refine pass when notes are configured.
- Keep `sql_file` content reviewed; the engine sends the entire file to the LLM when the LLM DDL path runs.
- Audit the on-disk artifacts when handing them between environments. The `top_k_values` cap (`PROFILING_TOP_K = 50`) is enforced at profile time; if you raise a column's sensitivity above `none` after profiling, the cached top-k for that column is dropped on the next rebuild.
- Treat the resolved engine storage directory (and any `artifacts_dir` parent you pass in) as you would treat a credential store: filesystem-private, backed up, and not shared across trust boundaries.
- Never embed credentials in `notes_file`, `sql_file`, or schema descriptions. The engine has no way to redact them before the LLM sees them.

## 10. Reporting a vulnerability

The repository's security contact is the maintainer listed on the GitHub repository page. Open an issue marked private when the GitHub repo allows it, or email the maintainer directly. Do not file public issues for vulnerabilities that affect deployed installations.
