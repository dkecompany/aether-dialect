# Integrator guide

How an engineer embeds `aetherdialect` in a service, job runner, notebook, or automation: the six core objects, the `SessionStep` contract, sync and async loops, reader/writer and owner/consumer deployment, federation and file-engine embedding, LLM wiring, and observability. Operator-facing semantics (notes, structure documents, migration, warmup): [User guide](USER_GUIDE.md). Exact signatures, TOML keys, JSON shapes, and exception catalogue: [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → this document → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Core objects](#core-objects) | Six constructible types and role/mode knobs |
| [Engine storage](#engine-storage) | Artifacts directory and connection slug |
| [Configuration boundary](#configuration-boundary) | Connection identity vs behavioural limits |
| [The session contract: suspend and terminal steps](#the-session-contract-suspend-and-terminal-steps) | `SessionStep`, `done`, `kind`, `reply_shape` |
| [Minimal embedding (sync)](#minimal-embedding-sync) | Smallest correct `ask` / `step` loop |
| [Async embedding](#async-embedding) | Same loop with `AsyncPipelineSession` |
| [Reader and writer split](#reader-and-writer-split) | Multi-process reader vs writer sessions |
| [Multi-user deployment](#multi-user-deployment) | Owner/consumer on shared artifacts |
| [Embedding a federation](#embedding-a-federation) | Composite `AetherFederation` embedding |
| [Embedding file uploads (CSV/Excel)](#embedding-file-uploads-csvexcel) | File-engine construction and validation |
| [LLM provider wiring](#llm-provider-wiring) | Azure two-slot mapping |
| [Observability](#observability) | Audit, diagnostics, phase callbacks |
| [Guarantees](#guarantees) | Determinism, replay limits, federation numerics, cost caps |

---

## Core objects

Meet the six public types in composition order: each scopes the next. Operator semantics for allow/deny lists, notes, and spaces live in the [User guide](USER_GUIDE.md); this section is what an integrator constructs and wires.

### AetherEngine

Root facade for one database connection. Construction reflects the catalog (or loads cache), builds or loads the default schema graph, and stamps artifact fingerprints. One instance binds exactly one database. Operator overview: [User guide — AetherEngine](USER_GUIDE.md#aetherengine).

### EngineContext

Frozen scope passed into `AetherEngine(...)`. Controls which relations enter the graph, default notes/DDL paths, and execution-time allow/deny lists (`allow_objects`, `deny_objects`, `allow_columns`, `deny_columns`). The implicit first scope is always the **default context**. Saved scope presets are registered with `engine.engine_context(name, context)` and later selected by passing the **preset name string** to `AetherEngine(...)`. Operator detail: [User guide — EngineContext](USER_GUIDE.md#enginecontext).

### AetherFederation

Composite facade over member `AetherEngine` instances. Each member keeps its own artifact tree; the federation tree lives beside them. Construct with `members` as a sequence of engines (each member's `source_id` is its connection name) plus `declaration=` (path or dict). Use `fed.session(...)` for the same `PipelineSession` / `AsyncPipelineSession` contract as a single engine. Worked declaration example: [Sandbox — Federation declaration format](SANDBOX.md#federation-declaration-format). Schema: [API reference — Federation documents](API_REFERENCE.md#federation-documents). Operator overview: [User guide — AetherFederation](USER_GUIDE.md#aetherfederation).

### FederationContext

Optional composite scope on `AetherFederation(..., context=...)`. Same allow/deny shape as `EngineContext`, plus optional `notes` or `notes_file` on the default federation context (set at most one). Qualified `source.table.column` tokens resolve member ownership; bare names that match more than one member raise. Federation `allow_objects` and `deny_objects` must stay within each member engine's effective object visibility; violations raise at federation init. Operator detail: [User guide — FederationContext](USER_GUIDE.md#federationcontext).

### AetherSpace

Named knowledge partition over the default graph on an engine or federation. Define with `engine.aetherspace(name, space_context=...)` or the federation equivalent — **owner-only**. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit. Prefer `session(space=uid)` after create. Template learning is partitioned by space **uid**. Operator overview: [User guide — AetherSpace](USER_GUIDE.md#aetherspace).

### SpaceContext

Frozen allow/deny lists for a named space: `tables`, `columns`, `deny_objects`, `deny_columns`, and optional `notes` or `notes_file` (set at most one). Applied at question time for knowledge and template partitioning. All `SpaceContext` instances are created on the default engine or federation context. Operator detail: [User guide — SpaceContext](USER_GUIDE.md#spacecontext).

### Engine role and session mode

| Knob | Values | Effect |
| --- | --- | --- |
| `AetherEngine(..., role=...)` / `AetherFederation(..., role=...)` | `owner` (default), `consumer` | **Engine role:** owner builds/mutates artifacts; consumer pins the owner snapshot. |
| `engine.session(mode=...)` / `fed.session(mode=...)` | `writer` (default), `reader` | **Session mode:** writer persists learning; reader keeps learning session-local. |

---

## Engine storage

The objects above persist under a shared root so owner, consumer, reader, and writer processes can point at the same learning:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

- **`<artifacts_parent>`** — expanded `artifacts_dir` argument, or the platform user-data directory when omitted.
- **`<connection_slug>`** — deterministic from **database location keys** ([API reference — Artifact storage slug](API_REFERENCE.md#artifact-storage-slug)). Credentials never appear in the slug.

Each database connection gets its own directory and its own default context snapshot. Named engine contexts and AetherSpaces persist as JSON sidecars under the same tree. Federation composites use `fed_<federation_id>/` beside member `conn_<slug>/` trees as parallel siblings under the same artifacts root. Layout detail: [How it works — Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-fingerprints).

The artifacts parent directory must reside on a **local filesystem** so advisory artifact locks can serialize cooperating processes reliably; network-mounted paths may emit an `ARTIFACTS_DIR_NOT_LOCAL` diagnostic at construction.

### Artifacts are library-owned

The artifacts directory is **library-owned**: never edit, move, or partially restore files under `<artifacts_parent>/aetherdialect/` by hand. User-facing changes go through export, edit in the working directory, then apply:

| Change | Export | Apply |
| --- | --- | --- |
| Structure (roles, sensitivity, keys) | `export_structure(space=...)` | `apply_structure(document)` |
| Space knowledge / descriptions | `export_knowledge(space=...)` | `apply_knowledge(space, document)` |
| Federation declaration | `export_federation()` | `apply_federation(document)` |
| Migration decisions | `preview_migration_map()` | `apply_migration_map(document)` |

On a federation, member structural edits use `export_structure` / `apply_structure` on each member engine. Cross-source identifier drift uses `apply_migration_map(document)` on the federation after editing the federation migration map document.

---

## Configuration boundary

The TOML config file and environment variables carry **connection identity** only: engine selection, hosts, ports, databases, schemas, warehouses, catalogs, roles, users, passwords, tokens, key paths, LLM provider and API keys, named connection blocks, artifacts roots, and file-engine source paths.

Alternatively, pass a **connection mapping** to `AetherEngine(..., connection={...})` with the same identity keys. Mapping values apply to that instance only and are **never** written to `os.environ`. Pair with `token_provider=` when secrets must rotate without reconstructing the engine (`refresh()`).

When `config_file` is set, TOML values are authoritative for every flattened key the file claims — environment variables do not override them. When `config_file` is omitted, the process environment supplies connection identity.

**Behaviour** — pool sizing, timeouts, caps, retention counts, and non-security policy flags — is set only through `EngineLimits` and `FederationLimits` constructor arguments, or explicitly via `EngineLimits.from_config_file(path)` and `FederationLimits.from_config_file(path)` reading `[limits]` and `[federation_limits]` from the same TOML document. Limits are never flattened into the environment.


| Side | Examples |
| --- | --- |
| Connection identity (env / TOML) | `[postgresql].host`, `PGHOST`, `[engine].selected`, `OPENAI_API_KEY`, `[llm].provider`, `CSV_DIRECTORY` |
| Behaviour (`EngineLimits` / `FederationLimits`) | `pool_size`, `statement_timeout_ms`, `max_result_rows`, `max_members`, `member_row_cap` |

```python
from aetherdialect import AetherEngine, EngineContext, EngineLimits

limits = EngineLimits.from_config_file("./aetherdialect.toml")
engine = AetherEngine(EngineContext(), artifacts_dir="./run", config_file="./aetherdialect.toml", limits=limits)
```

---

## The session contract: suspend and terminal steps

Every `session.ask(...)` or `session.step(...)` returns a **`SessionStep`**. Your integration should branch on **`step.kind`** and **`step.done`** — not on parsing `step.prompt` text.

### Terminal vs suspended

| State | `step.done` | What to do |
| --- | --- | --- |
| **Suspended** | `False` | Pipeline waits for user input. Render `step.prompt` (and optional SQL preview), collect a reply, call `session.step(reply)`. |
| **Terminal** | `True` | Turn finished. Read `answer`, or `sql`/`data`, or `error`. No further `step` call for this turn unless you `ask` a new question. |

Use **`step.reply_shape`** (`"yes_no"` or `"free_text"`) to choose UI controls while suspended.

### Public `kind` values

Inside the pipeline, suspend points carry an internal `state_id`. Before returning `SessionStep`, the engine maps each internal id to a stable public **`kind`** string. Integrators should use **`kind` only**.

| Public `kind` | Typical suspend | `reply_shape` | Meaning |
| --- | --- | --- | --- |
| `awaiting_intent_confirm` | intent confirmation | `yes_no` | Confirm interpreted plan. |
| `awaiting_intent_feedback` | intent rejection | `free_text` | Supply reason intent is wrong. |
| `awaiting_reuse_confirm` | direct template reuse | `yes_no` | Confirm a matched template reuse hit. |
| `awaiting_sql_confirm` | post-execution SQL preview | `yes_no` | Confirm generated SQL after execution. |
| `execute` | execute gate | `yes_no` | Confirm running stored SQL. |
| `awaiting_sql_feedback` | SQL rejection | `free_text` | Supply reason SQL is wrong. |
| `result` | — | — | Terminal analytical success (`step.done == True`). |
| `meta` | — | — | Terminal metadata answer (`step.done == True`); read `step.answer`. |
| `error` | — | — | Terminal failure (`step.done == True`, `step.error` set). |
| `idle` | — | — | Session reset / no active turn. |

### Template replay vs fresh generation

Generation paths for exact and fuzzy question reuse pin the stored template SQL and its recorded join signature. They rebind literal and structural parameters against the template's stored slot schema; missing or wrongly typed extractions abort reuse and fall through to fresh generation.

**Equality with a cold generation path is not promised.** Treat replay as "execute this trusted template with new binds," not as a deterministic duplicate of today's full pipeline output.

After a successful analytical turn, store `step.template_id` and re-run later with `engine.execute_template(step.template_id, {b.handle: b.current_value for b in step.parameters})`. `AetherFederation` exposes the same `list_templates` / `fetch_template` / `execute_template` trio.

### SessionStep fields (embedding checklist)

| Field | Use |
| --- | --- |
| `done` | Loop until `True`. Metadata turns finish in one step (`done=True`) with no confirm loop. |
| `kind` | Branch UI logic ([table above](#public-kind-values)). |
| `prompt` | Short line to show before collecting input (interactive layer). |
| `reply_shape` | `"yes_no"` vs `"free_text"` while suspended. |
| `answer` | Terminal metadata answer text. |
| `sql` | Dialect parameterized SQL: `str` on single-engine; `dict[str, str]` on multi-member federation; `None` when absent. |
| `data` | Result rows at this step; `data_truncated` states whether more exist. |
| `parameters` | Bind slots with `handle`, `current_value`, `display_name`, `column_expr`. |
| `template_id` | Stable template id for `execute_template`. |
| `intent_summary` | Structured interpretation headline on intent-related steps. |
| `semantic_warnings` | Model-authored caveats on intent-confirm steps. |
| `error` | [`SessionError`](API_REFERENCE.md#sessionerror) with closed [`SessionOutcome`](API_REFERENCE.md#sessionoutcome) on terminal failure. |
| `diagnostics` | Turn-level tracing rows. |
| `llm_usage` | Per-turn token and cost summary when available. |

Full field list: [API reference — SessionStep](API_REFERENCE.md#sessionstep). Outcome mapping: [Troubleshooting — SessionOutcome](TROUBLESHOOTING.md#sessionoutcome).

### Ask routes (caller still only calls `ask`)

Validation classifies the question internally. Integrators never pass a route flag — **always branch on the returned step**:

```python
step = session.ask("...")
if step.answer is not None:
    handle_meta(step.answer)
elif step.sql is not None:
    handle_sql(step.sql, step.data, step.template_id)
elif step.error is not None:
    handle_error(step.error)  # map step.error.code to product UX
else:
    handle_suspend(step)
```

Map `step.error.code` to product sentences; optional built-in refusal prose: [Troubleshooting — REFUSAL_CATALOGUE](TROUBLESHOOTING.md#refusal-catalogue).

### Knowledge export without asking

Use `export_structure(space=...)` for the read-only visible catalog. Use `export_knowledge(space=...)` / `apply_knowledge(space, document)` for space domain knowledge and description overlays. Prefer `ask()` when the user asked in natural language; prefer export when the caller only needs structured inventory.

### Permission denial

When the database or execution scope blocks a turn:

| Signal | How to detect | Action |
| --- | --- | --- |
| Synchronous API | `except AccessError as exc` during validation or direct execute helpers | Map to your product's access-denied UX; read `exc.operation`. |
| Session turn | `step.error` with `step.error.code == SessionOutcome.FORBIDDEN` | Map `error.code` to UX; use `error.detail_code` for analytics. |

`SchemaAccessError` remains the init-time access surface; runtime execute/EXPLAIN denial is `AccessError`. AetherSpace scope refusals map to `unanswerable`, not `forbidden` ([Troubleshooting — SessionOutcome](TROUBLESHOOTING.md#sessionoutcome)).

---

## Minimal embedding (sync)

```python
from aetherdialect import EngineContext, AetherEngine

engine = AetherEngine(
    EngineContext(),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
)

with engine.session() as session:
    step = session.ask("Top 5 customers by total payment?")
    while not step.done:
        if step.reply_shape == "yes_no":
            reply = "y"
        else:
            reply = "count distinct customers only"
        step = session.step(reply)

    if step.error is None:
        print(step.sql)
        print(step.data)
```

Helper methods:

- `session.ask_until_done(question, on_confirm="y")` — auto-answers **yes/no** suspends only; raises on free-text suspends.
- `session.accept_until_done(question, ...)` — auto-answers both yes/no and free-text suspends until the turn ends.

---

## Async embedding

`AsyncPipelineSession` (from `engine.asession()`) runs the same contract on worker threads:

```python
async with engine.asession() as session:
    step = await session.ask("...")
    while not step.done:
        reply = await get_user_input(step)
        step = await session.step(reply)
```

Method table: [API reference — AsyncPipelineSession](API_REFERENCE.md#asyncpipelinesession).

---

## Reader and writer split

More than one process can share one artifacts directory when you split write responsibility:

- **Writer** (`mode="writer"`, default): persists learning. Writer-mode turns drain `write_queue.jsonl` at turn start under the artifacts lock. One active writer per artifacts directory is recommended.
- **Reader** (`mode="reader"`): keeps learning session-local. Readers do not enqueue durable write-queue events.

Mechanism detail: [How it works — Concurrent sessions and durability](HOW_IT_WORKS.md#8-concurrent-sessions-and-durability). Offline demo: [Sandbox guide — Reader/writer sessions](SANDBOX.md#readerwriter-sessions).

---

## Multi-user deployment

Owner/consumer builds directly on the reader/writer split and the shared artifacts directory.

Pattern for **any supported database engine**:

1. **Owner writer** — one process with `role="owner"`, `mode="writer"`, full `EngineContext`, and durable `artifacts_dir` on shared storage. Builds the default graph and persists learning.
2. **Consumer readers** — processes with `role="consumer"`, restricted database credentials, and `mode="reader"`. On open the library loads the owner `schema_graph.json.gz`, runs a cheap SQL privilege probe, and builds a credential **subset** working graph.
3. **Credential-default AetherSpace** — the library auto-ensures a notes-free system space for the caller's selectable grant fingerprint and defaults `session()` to that space for consumers (`default_space_uid`). Product catalogs must omit it: `list_aetherspaces()` excludes it unless `include_system=True`.
4. **Align scope** — `EngineContext.allow_objects` / `allow_columns` (or `FederationContext` on a composite) intersect credential visibility on security gates. Security RBAC is context ∩ credentials ∩ non-HIDDEN. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit.
5. **User AetherSpaces** — further narrow model focus per team or product surface; create scope must stay ⊆ effective visibility (owner-only create).

Readers share the owner's artifacts directory but never mutate it directly.

```python
owner = AetherEngine(
    EngineContext(allow_objects=frozenset({"orders", "customers"})),
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="owner",
)

consumer = AetherEngine(
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="consumer",
)
with consumer.session(mode="reader") as session:
    session.accept_until_done("How many orders last month?")
```

Try the same pattern in the sandbox: [Sandbox guide — Owner vs consumer roles](SANDBOX.md#owner-vs-consumer-roles).

### Restricted environments

When `artifacts_dir` is omitted, the library uses the platform user-data directory as the artifacts parent. If that default directory is not writable, construction raises `ConfigError` naming the attempted path. Set `artifacts_dir` and, when needed, `FederationLimits.coordinator_temp_dir` to writable local paths before embedding in locked-down service accounts, containers, or CI sandboxes.

---

## Embedding a federation

Once a single-engine embedding loop works, the composite case is the same session contract over several member engines.

Register members as a sequence of `AetherEngine` instances. Each member keeps its own artifact tree under `conn_<connection>/`; the federation tree lives at `fed_<federation_id>/`.

```python
from aetherdialect import AetherEngine, AetherFederation

orders_db = AetherEngine(..., connection="orders", artifacts_dir="/artifacts")
catalog_db = AetherEngine(..., connection="catalog", artifacts_dir="/artifacts")
fed = AetherFederation(
    "commerce",
    members=[orders_db, catalog_db],
    declaration="/path/to/federation_declaration.json",
    artifacts_dir="/artifacts",
)
with fed.session() as session:
    step = session.ask("...")
    while not step.done:
        step = session.step(reply_for(step))
```

Cross-source joins and logical mappings are declared in the federation declaration dict passed to `declaration=` at construction. Compose decomposes multi-source intents into per-source sub-intents and combines frames in a DuckDB coordinator. Multi-member turns set `SessionStep.sql` to a `dict` of member SQL.

### Cancelling a federated turn

Call `session.cancel()` on the **same** `PipelineSession` (or `await session.cancel()` on `AsyncPipelineSession`) that owns the in-flight turn. It returns `True` only when that session has an active turn. Cancellation is cooperative: the coordinator observes it **between member stages or batches** and also cancels any in-flight database statement on a member.

### End-to-end walkthrough

1. **Named connections** — in `aetherdialect.toml`, give each database its own connection block. Set `connection="name"` per `AetherEngine(...)`.
2. **Member engines** — construct one `AetherEngine` per connection with a shared `artifacts_dir`. Wait for first construction on every member before composing.
3. **Author declaration** — build a federation declaration dict (`federation_id`, `cross_source_joins`, `coordinator` caps, optional `logical_tables` / `logical_columns`). Resolve table-name collisions with `aliases` or logical mappings before construction.
4. **Compose** — `AetherFederation(name, members=[...], declaration=..., artifacts_dir=...)`. The federation `name` must equal `federation_id` in the declaration.
5. **First question** — same `ask` / `step` loop as a single engine. On multi-member turns `isinstance(step.sql, dict)` is true.
6. **Re-entry** — `export_federation()` / `apply_federation(document)`; member drift uses `preview_migration_map()` / `apply_migration_map(document)` on each member engine.
7. **Add or remove a member** — register a new `AetherEngine`, update the declaration, and reconstruct when `artifact_format_version` mismatches.

### Federation errors

| Exception | When raised |
| --- | --- |
| `FederationDeclarationError` | Manifest or mappings JSON is structurally invalid or member references do not resolve. |
| `FederationIneligibleError` | A validated intent cannot be decomposed into a federated plan. |
| `FederationRuntimeError` | Coordinator or member execution fails after planning. |
| `FederationPartialFailureError` | One member failed after others succeeded. |
| `FederationInvariantError` | Prepared plan pins no longer match the live composite graph. |
| `FederationMemberExecutionError` | One member's query failed during execution. |
| `FederationCapExceededError` | A federated row, byte, or timeout cap was exceeded. |

Full inheritance chain: [API reference — Exceptions](API_REFERENCE.md#exceptions).

---

## Embedding file uploads (CSV/Excel)

When `AETHERDIALECT_ENGINE` is `csv`, uploads are validated before DuckDB load. **Inspect first, then construct.**

Severity definitions (**Advisory**, **Review**, **Blocking**, **Fatal**): [User guide — CSV and Excel uploads](USER_GUIDE.md#csv-and-excel-uploads).

```python
from aetherdialect import AetherEngine, EngineContext, inspect_tabular_upload

path = "/uploads/items.xlsx"
report = inspect_tabular_upload(path)
if report.requires_review:
    selections = {"items.xlsx": report.suggested_selections.get("items.xlsx", {})}
else:
    selections = {}

engine = AetherEngine(
    EngineContext(),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
    source_selections=selections,
)
```

`inspect_tabular_upload` reads and analyses the grid only — it does not open DuckDB or construct an engine. Model-assisted upload interpretation: [Security — Upload inspection](SECURITY.md#57-upload-inspection-csv-file-engine).

---

## LLM provider wiring

When `AETHERDIALECT_LLM_PROVIDER` is `azure`, provision exactly two deployments:

| Slot | Env var | Deploy this model |
| --- | --- | --- |
| `light` | `AZURE_OPENAI_DEPLOYMENT_LIGHT` | `gpt-5-mini` |
| `heavy` | `AZURE_OPENAI_DEPLOYMENT_HEAVY` | `gpt-5.4-mini` |

Full key tables: [API reference — Configuration](API_REFERENCE.md#configuration).

---

## Observability

| Channel | Access | Granularity |
| --- | --- | --- |
| **`audit_sink`** | Constructor callback | Coarse lifecycle (`init`, `ask_begin`, `ask_done`, `close`, and related admin events). |
| **`phase_callback`** | Constructor callback | Coarse `PhaseProgressEvent` transitions during construction and ask turns. |
| **`diagnostic_sink`** | Constructor callback | `Diagnostic` rows on the diagnostic channel. |
| **`SessionStep.diagnostics`** | Every `ask` / `step` return | Turn-level codes (`REUSE_HIT`, structured refusals, ...). |

Diagnostic catalogue: [Troubleshooting — Diagnostic codes](TROUBLESHOOTING.md#diagnostic-codes).

---

## Guarantees

What the library promises integrators, and what it deliberately does not. For refused question shapes and per-engine capability limits, see the [Support matrix](SUPPORT_MATRIX.md).

### Deterministic ask rebuild

When the artifacts tree is unchanged and the resolved intent is the same, the **ask rebuild** path produces byte-identical SQL. This applies to fresh generation after intent confirmation — not to template replay.

### Template replay boundaries

Template reuse on exact and fuzzy question paths pins the stored template SQL and its recorded **join signature**. Replay is **not** promised to match a cold regenerate after schema enrichment or graph changes.

### Federation numeric exactness

Exact numeric types remain exact through **federation egress**. Approximate numeric types may be widened to float at federation boundaries.

### EXPLAIN cost caps

Query cost caps are enforced only where the [Support matrix](SUPPORT_MATRIX.md) shows an active EXPLAIN cost gate for your engine. When the warehouse returns no row estimate, the gate is **fail-open**.

### Host configuration vs library-owned artifacts

| You configure | Library-owned (do not hand-edit) |
| --- | --- |
| `EngineLimits` / `FederationLimits` | Files under `<artifacts_parent>/aetherdialect/` |
| Engine and federation contexts, AetherSpaces | Template stores, fingerprint sidecars |
| `audit_sink`, phase callbacks, diagnostic_sink | Internal graph snapshots and learning partitions |

Export/apply dict pairs ([Artifacts are library-owned](#artifacts-are-library-owned)) are the supported edit surface.

### Unsupported constructs

Question shapes the engine refuses are listed in [SUPPORT_MATRIX](SUPPORT_MATRIX.md).

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
