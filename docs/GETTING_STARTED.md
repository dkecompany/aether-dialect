# Getting started

Hands-on onboarding: install the PyPI extra for your engine, wire credentials, and run your first question. Each `AetherEngine` binds exactly one database. Operator day-to-day semantics live in the [User guide](USER_GUIDE.md); embedding patterns in the [Integrator guide](INTEGRATOR_GUIDE.md). Exact signatures and TOML key tables live in the [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) -> this document -> [User guide](USER_GUIDE.md) -> [Integrator guide](INTEGRATOR_GUIDE.md) -> [Sandbox guide](SANDBOX.md) -> [API reference](API_REFERENCE.md) -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Offline practice](#offline-practice) | One sandbox question |
| [Connect your warehouse](#connect-your-warehouse) | TOML, notes, construction, first question |
| [When things fail](#when-things-fail) | Common startup errors |

---

Two onboarding paths:

1. **Offline practice** - bundled rental shop in memory, no warehouse credentials.
2. **Connect your warehouse** - the same session API against any supported engine.

Install the PyPI extra and TOML block for the engine you use.

**Multiple databases?** Build one `AetherEngine` per connection, author `federation_declaration.json`, then construct `AetherFederation(name, members=..., declaration_file=...)`. The session API is unchanged; federated turns decompose per member and combine in an in-process DuckDB coordinator. Declaration format: [Sandbox - Federation declaration format](SANDBOX.md#federation-declaration-format). Start with [Integrator guide - Embedding a federation](INTEGRATOR_GUIDE.md#embedding-a-federation) after your first single-engine question works.

## Offline practice

```bash
pip install aetherdialect
```

```python
from aetherdialect import AetherEngine

with AetherEngine.offline_sandbox() as sb:
    with sb.session() as session:
        step = session.accept_until_done("How many films are there?")
    print(step.sql)
    print(step.data)
```

Each `offline_sandbox()` call wipes any prior temp artifacts, unpacks the bundled seed, and rebuilds from scratch. When the `with` block ends (or you call `sb.close()`), temp extract directories and owned artifacts are deleted. Full offline exercises: [Sandbox guide](SANDBOX.md).

---

## Connect your warehouse

### What you will build

1. `aetherdialect.toml` - LLM credentials plus one database block
2. Optional `schema_notes.txt` - domain vocabulary (see [User guide - Notes file](USER_GUIDE.md#notes-file))
3. `ask.py` - constructs `AetherEngine` and runs one question
4. `./my_run/aetherdialect/<connection_slug>/` - versioned engine storage

### Pick your engine

Install the matching extra, set `[engine] selected` in TOML (or `AETHERDIALECT_ENGINE`), and fill one database section. Full key lists: [API reference - Configuration](API_REFERENCE.md#configuration) and [Database connection reference](API_REFERENCE.md#database-connection-reference).

There is no separate `excel` engine. CSV and Excel (`.xlsx`) uploads both use the registered `csv` engine.

| Engine | `pip install` | TOML `selected` | TOML section | Upload suffixes |
| --- | --- | --- | --- | --- |
| SQLite | `aetherdialect[sqlite]` | `sqlite` | `[sqlite]` | - |
| DuckDB | `aetherdialect` | `duckdb` | `[duckdb]` | - |
| CSV / Excel | `aetherdialect[csv]` | `csv` | `[csv]` | `.csv` and `.xlsx` |
| MySQL | `aetherdialect[mysql]` | `mysql` | `[mysql]` | - |
| MariaDB | `aetherdialect[mariadb]` | `mariadb` | `[mariadb]` | - |
| SQL Server | `aetherdialect[sqlserver]` | `sqlserver` | `[sqlserver]` | - |
| PostgreSQL | `aetherdialect[postgresql]` | `postgresql` | `[postgresql]` | - |
| Redshift | `aetherdialect[redshift]` | `redshift` | `[redshift]` | - |
| Databricks | `aetherdialect[databricks]` | `databricks` | `[databricks]` | - |
| Snowflake | `aetherdialect[snowflake]` | `snowflake` | `[snowflake]` | - |
| BigQuery | `aetherdialect[bigquery]` | `bigquery` | `[bigquery]` | - |

You also need an OpenAI API key (or Azure OpenAI - see [API reference](API_REFERENCE.md#configuration)).

### Step 1 - Test the database connection

Confirm connectivity before constructing `AetherEngine`, using the same credentials you will put in TOML:

```python
from sqlalchemy import create_engine, text

engine = create_engine("YOUR_SQLALCHEMY_URL_HERE")
with engine.connect() as conn:
    row = conn.execute(text("SELECT 1 AS ok")).one()
    print("connected:", row.ok)
```

Examples:

```python
create_engine("sqlite:////absolute/path/to/your.db")
create_engine("duckdb:////absolute/path/to/your.duckdb")
create_engine("postgresql+psycopg2://user:password@localhost:5432/your_database")
```

If `SELECT 1` fails, fix credentials or networking before involving the text-to-SQL pipeline.

### Step 2 - Create `aetherdialect.toml`

Every production setup needs `[engine] selected`, one database block, and LLM credentials.

**SQLite:**

```toml
[engine]
selected = "sqlite"

[sqlite]
path = "/absolute/path/to/your.db"

[openai]
api_key = "REPLACE_ME"
```

**DuckDB:**

```toml
[engine]
selected = "duckdb"

[duckdb]
path = "/absolute/path/to/your.duckdb"

[openai]
api_key = "REPLACE_ME"
```

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

[execution]
max_query_cost_rows = "10000"
statement_timeout_ms = "30000"
```

Snowflake, Databricks, and BigQuery use different auth shapes. Copy the matching section from the [API reference flattening table](API_REFERENCE.md#config_file-toml-flattening).

Environment variables can override mapped keys when the same name is set in the shell.

### Step 3 - Domain notes (optional, recommended)

Plain text beside your script, passed as `EngineContext.notes_file`. Use one or two sentences per important table, join hints, and explicit sensitivity statements. The engine uses these to tag roles and tiers. See [User guide - Notes file](USER_GUIDE.md#notes-file) for the format.

### Step 4 - Wire `EngineContext` and construct the engine

`EngineContext` is frozen scope: which relations enter the graph, optional notes and DDL paths, and allow/deny lists. Set `include="views"` when your warehouse exposes analytical views (default `"tables"`). To reflect both base tables and views, run separate engine constructions or scope passes - `include="both"` is rejected. Semantics: [User guide - EngineContext](USER_GUIDE.md#enginecontext).

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

On large catalogs, narrow scope:

```python
ctx = EngineContext(
    include="tables",
    notes_file="./schema_notes.txt",
    allow_objects=frozenset({"orders", "customers", "products"}),
)
```

### Step 5 - First construction

The first construction reflects the catalog and builds a versioned snapshot. Timing scales with table count. See [User guide - First run](USER_GUIDE.md#first-run) for progress detail and drift handling.

### Step 6 - Choose your API

A `SessionStep` is one observable point in a programmatic turn: it carries `kind`, `done`, `prompt`, and optional `sql` / `data` / `error`. Full contract: [Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

| | `run_interactive()` | `session()` |
| --- | --- | --- |
| Output | stdout prompts | `SessionStep` fields |
| Loop | one question per call | `ask` / `step` until `done` |
| AetherSpace | optional `space="name"` (default `"master"`) | same `space=` kwarg |
| Best for | terminal demos | services, notebooks, MCP |
| Suspend handling | built-in `input()` | you render `step.prompt`, call `step()` |

Embedded products should use `session()` and branch on `step.kind`.

**Terminal demo:**

```python
from aetherdialect import EngineContext, AetherEngine

engine = AetherEngine(
    EngineContext(notes_file="./schema_notes.txt"),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
)
engine.run_interactive()
```

**What you see on stdout:** construction prints profiling progress first. During a turn, `run_interactive()` emits diagnostic messages and shows a prompt before each `input()`. The sequence is the same on every engine. Branch embedded UIs on `SessionStep.kind`, not on parsing prompt text ([Integrator guide](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps)).

**Programmatic loop** (detail in [Integrator guide - Minimal embedding](INTEGRATOR_GUIDE.md#minimal-embedding-sync)):

```python
with engine.session() as session:
    step = session.ask("How many rentals are currently overdue?")
    while not step.done:
        reply = input(step.prompt or "y/n: ").strip()
        step = session.step(reply)
    print(step.sql or step.error)
```

Sessions default to the **`master`** [AetherSpace](USER_GUIDE.md#aetherspace) unless you pass `space="name"`.

### Step 7 - Run your first question

```bash
python ask.py
```

Accept/reject behavior and template reuse: [User guide - Asking a question](USER_GUIDE.md#asking-a-question).

---

## When things fail

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `ConfigError: ... api_key ...` | Missing LLM key | Set `[openai] api_key` or env |
| `ConfigError: ... engine ...` | Multiple engines configured | Set `[engine] selected` |
| `ConnectionError` / `DatabasePingFailed` | Bad DB credentials or network | Re-run Step 1 |
| Silence after `Profiling [1/N]` | Profiling in progress | Wait; narrow `allow_objects` |
| `MigrationPendingError` | Catalog drift | Edit `schema_migration_map.json` - [User guide](USER_GUIDE.md#migration) |

Full procedure: [User guide - Common pitfalls](USER_GUIDE.md#common-pitfalls).

---

**See also:** [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
