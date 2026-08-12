# Getting started

Hands-on onboarding: install the PyPI extra for your engine, wire credentials, and run your first question. Each `AetherEngine` binds exactly one database. Operator day-to-day semantics live in the [User guide](USER_GUIDE.md); embedding patterns in the [Integrator guide](INTEGRATOR_GUIDE.md). Exact signatures and TOML key tables live in the [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Try the sandbox](#try-the-sandbox) | One sandbox question |
| [Connect your warehouse](#connect-your-warehouse) | TOML, notes, construction, first question |
| [Federation](#federation) | Multiple databases |
| [When things fail](#when-things-fail) | Common startup errors |

---

Two onboarding paths:

1. **Try the sandbox** — bundled rental shop in memory, no warehouse credentials.
2. **Connect your warehouse** — the same session API against any supported engine.

Install the PyPI extra and TOML block for the engine you use.

## Try the sandbox

```bash
pip install aetherdialect
```

```python
from aetherdialect import Sandbox

with Sandbox() as sandbox:
    engine = sandbox.engine()
    with engine.session() as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
    print(step.data)
```

Each `Sandbox()` handle is self-contained. When the `with` block ends, temp extract directories and owned artifacts are deleted. Full sandbox walkthrough: [Sandbox guide](SANDBOX.md).

---

## Connect your warehouse

### What you will build

1. `aetherdialect.toml` — LLM credentials plus one database block
2. Optional `schema_notes.txt` — domain vocabulary (see [User guide — Notes file](USER_GUIDE.md#notes-file))
3. `ask.py` — constructs `AetherEngine` and runs one question
4. `./my_run/aetherdialect/<connection_slug>/` — versioned engine storage

### Pick your engine

Install the matching extra, set `[engine] selected` in TOML (or `AETHERDIALECT_ENGINE`), and fill one database section. Full key lists: [API reference — Configuration](API_REFERENCE.md#configuration).

There is no separate `excel` engine. CSV and Excel (`.xlsx`) uploads both use the registered `csv` engine.

| Engine | `pip install` | TOML `selected` | TOML section | Upload suffixes |
| --- | --- | --- | --- | --- |
| SQLite | `pip install aetherdialect` (stdlib driver; no extra) | `sqlite` | `[sqlite]` | - |
| DuckDB | `aetherdialect[duckdb]` | `duckdb` | `[duckdb]` | - |
| CSV / Excel | `aetherdialect[csv]` | `csv` | `[csv]` | `.csv` and `.xlsx` |
| MySQL | `aetherdialect[mysql]` | `mysql` | `[mysql]` | - |
| MariaDB | `aetherdialect[mariadb]` | `mariadb` | `[mariadb]` | - |
| SQL Server | `aetherdialect[sqlserver]` | `sqlserver` | `[sqlserver]` | - |
| Oracle | `aetherdialect[oracle]` | `oracle` | `[oracle]` | - |
| PostgreSQL | `aetherdialect[postgresql]` | `postgresql` | `[postgresql]` | - |
| Redshift | `aetherdialect[redshift]` | `redshift` | `[redshift]` | - |
| Databricks | `aetherdialect[databricks]` | `databricks` | `[databricks]` | - |
| Snowflake | `aetherdialect[snowflake]` | `snowflake` | `[snowflake]` | - |
| BigQuery | `aetherdialect[bigquery]` | `bigquery` | `[bigquery]` | - |

You also need an OpenAI API key (or Azure OpenAI — see [API reference](API_REFERENCE.md#configuration)).

### Step 1 — Test the database connection

Confirm connectivity before constructing `AetherEngine`, using the same credentials you will put in TOML:

```python
from sqlalchemy import create_engine, text

engine = create_engine("YOUR_SQLALCHEMY_URL_HERE")
with engine.connect() as conn:
    row = conn.execute(text("SELECT 1 AS ok")).one()
    print("connected:", row.ok)
```

If `SELECT 1` fails, fix credentials or networking before involving the text-to-SQL pipeline.

### Step 2 — Create `aetherdialect.toml`

Every production setup needs `[engine] selected`, one database block, and LLM credentials.

**PostgreSQL** (same pattern for MySQL, MariaDB, SQL Server, Redshift with their sections):

```toml
[engine]
selected = "postgresql"

[postgresql]
host = "localhost"
port = "5432"
database = "your_database"
user = "readonly"
password = "REPLACE_ME"
schema = "public"

[openai]
api_key = "REPLACE_ME"
```

Behavioural limits (timeouts, result caps, pools) are set with `limits=EngineLimits(...)` on the constructor, or `EngineLimits.from_config_file("./aetherdialect.toml")` reading a `[limits]` table.

When `config_file` is set, TOML values are authoritative for every flattened key the file claims — shell environment variables do not override them. When `config_file` is omitted, the process environment supplies connection identity. Merge order: [API reference — Merge order](API_REFERENCE.md#merge-order).

### Step 3 — Domain notes (optional, recommended)

Plain text beside your script, passed as `EngineContext.notes_file` (or inline `EngineContext.notes` — set at most one). See [User guide — Notes file](USER_GUIDE.md#notes-file).

### Step 4 — Wire `EngineContext` and construct the engine

```python
from aetherdialect import EngineContext, AetherEngine

ctx = EngineContext(
    include="tables",
    notes_file="./schema_notes.txt",
)

engine = AetherEngine(
    ctx,
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
)
```

On large catalogs, narrow scope with `allow_objects`.

### Step 5 — First construction

The first construction reflects the catalog and builds a versioned snapshot. See [User guide — First run](USER_GUIDE.md#first-run).

### Step 6 — Choose your API

| | `run_interactive()` | `session()` |
| --- | --- | --- |
| Output | stdout prompts | `SessionStep` fields |
| Loop | one question per call | `ask` / `step` until `done` |
| AetherSpace | optional `space=<uid>` | same `space=` kwarg |
| Best for | terminal demos | services, notebooks, MCP |

Embedded products should use `session()` and branch on `step.kind`. Full contract: [Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

```python
with engine.session() as session:
    step = session.ask("How many rentals are currently overdue?")
    while not step.done:
        reply = input(step.prompt or "y/n: ").strip()
        step = session.step(reply)
    print(step.sql or step.error)
```

Omit `space` for the default space (`engine.default_space_uid`). Consumer deployments: [Integrator guide — Multi-user deployment](INTEGRATOR_GUIDE.md#multi-user-deployment).

### Step 7 — Run your first question

```bash
python ask.py
```

Accept/reject behavior and template reuse: [User guide — Asking a question](USER_GUIDE.md#asking-a-question).

---

## Federation

Build one `AetherEngine` per connection, author a federation declaration dict (suggested persistence name: `federation_declaration.json`), then construct `AetherFederation(name, members=[...], declaration=...)`. The session API is unchanged; federated turns decompose per member and combine in an in-process DuckDB coordinator. Worked example: [Sandbox — Federation walkthrough](SANDBOX.md#federation-walkthrough). Schema: [API reference — Federation documents](API_REFERENCE.md#federation-documents). Embedding guide: [Integrator guide — Embedding a federation](INTEGRATOR_GUIDE.md#embedding-a-federation).

---

## When things fail

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `ConfigError: ... api_key ...` | Missing LLM key | Set `[openai] api_key` or env |
| `ConfigError: ... engine ...` | Multiple engines configured | Set `[engine] selected` |
| `DatabaseConnectionError` / `DatabasePingFailed` | Bad DB credentials or network | Re-run Step 1 |
| Silence after `Profiling [1/N]` | Profiling in progress | Wait; narrow `allow_objects` |
| `MigrationPendingError` | Catalog drift | Apply migration map — [User guide](USER_GUIDE.md#migration) |

Full procedure: [User guide — Common pitfalls](USER_GUIDE.md#common-pitfalls).

---

**See also:** [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
