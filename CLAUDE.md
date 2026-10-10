# Claude Code Project Instructions

## Project Overview

This is **langgraph-agent**, a local-first 4-Agent AI system for software development experiments.

- **Architect** — the leading authority. Sets architectural direction before planning, then holds the approval gate: the run ends on its `approved` verdict, not the Builder's say-so. No tools.
- **Planner** — interprets goals, creates structured plans, routes to next agent.
- **Researcher** — gathers context through the GraphRAG search tool (`search_knowledge_graph`).
- **Builder** — implements plans using filesystem, git, terminal, and test tools.

Inference defaults to local: every seat runs a model the Ollama daemon on this
machine serves from its own weights, so a fresh checkout needs no API key and
no ollama.com credentials at all. Ollama Cloud tags and Anthropic remain
available per seat for anyone who wants them. The embedding model also
runs locally through the same daemon, serving `qwen3-embedding:latest` -- the
daemon owns every model's placement, so nothing in this project touches torch
or a card itself. The embedding belongs to GraphRAG, not to a seat.

Tech stack: Python 3.12+, LangGraph, PostgreSQL + pgvector + NetworkX, tools served in-process under MCP-style names.

## Quick Reference

```bash
# Install everything on Arch / Omarchy (packages, venv, Ollama models, SearxNG,
# PostgreSQL + pgvector in Docker), then prove the embedder, the corpus store, git/gh and
# every seat actually work
./install.sh

# Install dependencies
pip install -e ".[dev]"

# Run all checks
ruff check src/ tests/ serve.py scripts/ spectral_graph/ example_usage.py ollama_client.py
mypy src/langgraph_agent/ serve.py ollama_client.py spectral_graph/
python -m pytest tests/ -v

# Start the web console
./launch_console.sh

# Or the same console in a container (Arch image, host networking)
docker compose up --build

# Find out which model actually works in which seat
python scripts/diagnose_seats.py --list
python scripts/diagnose_seats.py --phase probe

# The console as its user, in a real browser (needs the browser extra:
# pip install -e ".[dev,browser]"); read-only unless --allow
python scripts/browser_agent.py doctor
python scripts/browser_agent.py check --spawn --stub-seats --no-rebuild

# One shareable report on the whole machine: GPUs, daemon, models, database, console
python scripts/diagnose_machine.py --quick

# Run the example
python example_usage.py
```

## Project Structure

```
├── src/langgraph_agent/
│   ├── __init__.py            # create_agent_graph, initial_state, AgentState, ResearchStatus, Verdict
│   ├── state.py               # AgentState schema, initial_state() + ResearchStatus / Verdict enums
│   ├── config.py              # Seats and model tags, LLM setup, the daemon's circuit + StubLLM
│   ├── nodes.py               # Architect, Planner, Researcher, Builder nodes + prompt loading
│   ├── graph.py               # StateGraph wiring + conditional edges
│   ├── control.py             # RUN_CONTROL (the emergency stop), GPU_ARBITER, activity meters
│   ├── graphrag_server.py     # GraphRAG knowledge base: entity graph + chunked vector store
│   ├── corpus_store.py        # Where the corpus lives: PostgreSQL + pgvector, one schema per corpus
│   ├── embedding_calibration.json # The floor's off-domain questions (JSON: never indexed)
│   ├── mcp_client.py          # The agents' tool belts, served in-process
│   ├── dwell.py               # git_dwell: working tree to merged PR, through CI; the console's finish
│   ├── lexical.py             # BM25 + rank fusion: the lexical half of search
│   ├── web_research.py        # Online research: keyless search, our own selection gate
│   ├── html_text.py           # HTML → text by link density. No library, no dependency
│   ├── corpus_health.py       # Is the corpus still the archive? missing / extra / oversized
│   ├── corpus_spectral.py     # connectivity / topics / bottleneck / duplicate_entities (mixin)
│   ├── projects.py            # Generated projects: where a run writes, and opting one into the corpus
│   └── self_healing/          # Retries, circuit breakers and the healing journal (see Self-healing)
│       ├── logger.py          # SelfHealingLogger: severity-levelled healing log + event journal
│       └── decorators.py      # call_with_retry, Circuit, circuit_states, reset_circuit
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   ├── conftest.py            # Forces StubLLM; switches the web phase off; closes every circuit
│   ├── test_git_dwell.py      # The git pipeline against a real remote and a stub GitHub
│   ├── test_claims.py         # Documentation claims, made executable
│   ├── test_corpus_absent.py  # The two doors: reading never creates a corpus
│   ├── test_corpus_roots.py   # The corpus is research and deliberate embeds, never the checkout
│   ├── test_graph.py          # Pytest suite
│   ├── test_diagnose_seats.py # Guards the seat diagnostic's verdicts
│   ├── test_thinking.py       # Per-seat capability: the thinking switch, and tool support
│   ├── test_console_stop.py   # Emergency stop, deferred exit, snapshot
│   ├── test_chunking.py       # Document chunking, chunk ids, search collapse
│   ├── test_corpus_admin.py   # Corpus clear / export / reindex guards
│   ├── test_corpus_store.py   # The store against PostgreSQL: exact search, transactions, the lock
│   ├── store_doubles.py       # The store's own methods over a test's fake collection
│   ├── test_embedding_device.py # The one embedder, and the light that says when it is working
│   ├── test_mcp_tools.py      # Builder tool belt
│   ├── test_imports.py        # Pins the package's public surface, and initial_state
│   ├── test_lexical.py        # BM25, rank fusion, the relevance floor
│   ├── test_self_healing.py   # self_healing: journal, retry policy, circuits
│   ├── test_healing_integration.py # Self-healing at every seam the application uses it
│   ├── test_uploads.py        # Uploading a document into the corpus
│   ├── test_web_research.py   # Fan-out, the selection gate, storage, the empty answers
│   ├── test_html_text.py      # The from-scratch HTML reader
│   ├── test_web_entities.py   # Fetched pages stay out of the entity graph
│   ├── test_run_research_phase.py # Where the research phase sits in a run
│   ├── test_corpus_staleness.py   # Corpus vs disk, and not crying wolf
│   ├── test_startup_index.py  # The corpus is rebuilt when the console comes up
│   ├── test_graph_queries.py  # Undirected traversal of the knowledge graph
│   ├── test_gpu_arbiter.py    # The cards: one model at a time, every layer on the GPU
│   ├── test_research_length.py # How much retrieved evidence reaches the Builder
│   ├── test_rpc_params.py     # RPC parameters: typed, bounded, refused by name
│   ├── test_projects.py       # Generated projects: the held-out walk, the write scope, embedding
│   ├── test_spectral_graph.py # The spectral_graph package, against closed-form spectra
│   ├── test_dwell_tool.py     # The command-line dwell tool's confinement to this folder
│   ├── test_ollama_keepalive.py # The keep-alive drop-in, against stand-in sudo and systemctl
│   ├── test_browser_agent.py  # The browser agent's guard, passes and tools; live ones under BROWSER_TESTS=1
│   ├── test_diagnose_machine.py # The machine diagnostic: parsers, redaction, a degraded machine
│   └── test_claude_tools.py   # Claude Code setup, against stand-in claude, npx and git
├── scripts/
│   ├── verify_and_test.py     # Manual verification: dependencies, seats, a search, the suite
│   ├── cloud_smoke.py         # One run with every seat on Anthropic
│   ├── spectral_benchmark.py  # Graph-architecture sweep behind the A-numbers
│   ├── diagnose_seats.py      # Role probes + team runs per seating
│   ├── dwell.py               # The dwell pipeline from the command line, confined to this folder
│   ├── network_check.sh       # Which download hosts the network lets through; the installers' first step
│   ├── ollama_keepalive.sh    # systemd keeps the daemon running: enabled at boot, restarted on any exit
│   ├── browser_agent.py       # The console in a real browser: scripted passes, one-shot tools, an RPC guard
│   ├── diagnose_machine.py    # One redacted, shareable report on the machine the console runs on
│   └── claude_tools.sh        # Claude Code, its claude.ai sign-in and the Playwright MCP server, per machine
├── frontend/
│   ├── index.html             # Web console SPA
│   └── README.md
├── .github/workflows/
│   └── ci.yml                 # ruff, mypy, pytest, on every push
├── spectral_graph/            # the spectral code: at the root, outside the installed package
│   ├── laplacian.py           # Laplacian matrix constructions
│   ├── spectrum.py            # Laplacian eigenvalues, dense or shift-inverted
│   ├── fiedler.py             # the Fiedler vector
│   ├── clustering.py          # spectral clustering, conductance, the Cheeger bounds
│   └── embedding.py           # spectral embedding via Laplacian eigenvectors
├── experimental/              # an agent run's own notes; never in the corpus
├── docker/
│   ├── entrypoint.sh          # the container's start: git identity, GitHub login, and what it can reach
│   ├── install.sh             # Docker-only install: no host Python; the image runs baked-in code
│   ├── up.sh                  # starts it, passing your gh login and git identity at every start
│   └── compose.yml            # the Docker-only stack: ~/Projects mounted, no code mounted
├── dockerfile                 # the console as an Arch image; the daemon stays on the host
├── docker-compose.yml         # host networking: the daemon, the database and SearxNG are on it
├── .dockerignore              # the venv, the caches, and every per-machine artifact
├── install.sh                 # Arch / Omarchy: everything, from nothing to a running console
├── cuda-embed-ollama.sh       # NVIDIA cards below compute 7.5: Ollama's CUDA 12 build, model 100% on the GPU
├── launch_console.sh          # builds .venv on first launch, starts serve.py, waits on /api/status, opens a browser
├── serve.py                   # Python HTTP server + API backend + the self-healing monitor
├── example_usage.py           # Demo script
├── ollama_client.py           # one prompt to the local daemon, bounded by the project's own timeout
├── README.md                  # User-facing documentation
└── .env.example               # Environment variables template
```

## Conventions

- **Python formatting/linting:** `ruff` configured in `ruff.toml`.
- **Type checking:** `mypy` configured in `mypy.ini`.
- **Tests:** `pytest` in `tests/`.
- **Default seats** (`DEFAULT_SEATS` in `config.py`, derived from
  `_DEFAULT_AGENT_MODELS` for `DEFAULT_PROVIDER`; every model tag is a named
  constant there, spelled once):

  | Seat | Provider | Model |
  |---|---|---|
  | Architect | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Planner | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Researcher | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Builder | ollama | `qwen3.8:latest` |

  Three seats run local weights the daemon holds itself; the Builder is the
  exception, since its work *is* tool calls and neither dolphin tag reports
  `tools` -- `qwen3.8:latest` does. Anthropic and Ollama Cloud tags remain
  available per seat; no seat uses any of them by default, so a fresh
  checkout runs without an API key or ollama.com credentials of its own.
- **Never send `temperature` to a modern Anthropic model.** Sampling parameters
  were removed on the Opus 5 / Sonnet 5 / 4.6+ families and are rejected with a
  400 that reads like an auth failure. `_accepts_temperature()` gates this.
- **Tool specialization (critical):** Architect and Planner get no tools; the
  Researcher's node calls GraphRAG read-only tools only, and its seat is handed
  what they returned rather than the tools; the Builder gets filesystem, git,
  terminal and test tools only. Keep GraphRAG out of `BUILDER_TOOLS`.
- **A run starts from `initial_state(goal, ...)`** (`state.py`); never write the
  state literal out by hand.

## State Schema

Every node reads/writes `AgentState`:

```python
{
    "goal": str,
    "architecture": str,
    "verdict": "plan" | "approved" | "revise" | "need_research",
    "messages": list[str],
    "plan": str,
    "research": str,
    "builder_report": str,
    "next_agent": "Researcher" | "Builder" | "END",
  "research_status": "ready_for_builder" | "need_replan" |
  "no_relevant_knowledge", "blockers": str, "files_changed": list[str],
  "failed_verification": list[str], "unverified": list[str],
  "builder_cut_off": "" | "turn_cap" | "deadline" | "no_tools", "lint_failed": list[str],
  "dwell": dict,  # status: merged | pending | checks_failed | open | committed | local | failed
  "expect_failures": bool, "discuss_only": bool, "output_dir": str,
  "step_count": int,
}
```

`stopped`, `stop_reason`, `run_id` and `elapsed_s` are added to the payload
`run_goal` returns, not to `AgentState`: that schema describes what the agents
write, not how the run ended.

The Architect writes `architecture` and `verdict`; the verdict is what routes
the loop and what ends it. `step_count` is incremented by the Architect gate,
not by the Builder, and by a Researcher that sends the Planner back
(`need_replan`): that loop never reaches the gate or the Builder, and would
otherwise run uncounted. Either way `MAX_STEPS` ends the run.

## Common Tasks

### Run the test suite
```bash
python -m pytest tests/ -v
```

Adding a Builder/Researcher tool, changing a seat's model, or adding a fifth
agent: see the `extend-agent` skill.

### The console API
One `POST /rpc` taking `{method, params}` and returning `{result, elapsed_ms}`
or `{error: {message}, elapsed_ms}`; methods live in `RPC_METHODS` in
`serve.py`. `GET /api/status` is the one other route: `launch_console.sh`, the
container's health check and the console driver poll it.
Parameters are typed, bounded and refused by name (`_int_param`, `_float_param`,
`_bool_param`, `_str_param`), parsed before anything is touched.
`run_goal` blocks its own thread for the whole run -- hence
`ThreadingHTTPServer` -- and a second one is refused while it is in flight.
It listens on loopback (`CONSOLE_HOST` overrides) and `_foreign_request`
refuses a write whose Origin is another site, and on loopback any request whose
Host is not loopback (DNS rebinding): the console runs goals, and a goal reaches
a Builder that runs programs. No CORS header is sent; the page is same-origin.

## Important Notes

- Do not let every agent call every tool; the specialization is the whole point.
- Empty state fields must render as `(empty)` in the state injection block.
- **The Builder's account of its own work is not evidence.** `files_changed` is
  appended only when a write tool reports success, never from prose.
  `state["files_changed"]` accumulates across passes while the local list in
  `builder_node` is this pass alone, and a path no longer on disk is retracted
  from the record and named in the report. "Described but not written" compares
  both spellings through `_report_path_key`.
- **Writes cannot leave the project.** `_resolve_write_path` resolves the whole
  path, leaf included, so a symlinked parent or leaf that leads out is refused;
  on a run given a project, `_outside_output_dir` confines writes to
  `projects/<name>`. Reads are not confined, deliberately: `terminal_execute`
  runs any program, so the tool belt is not a sandbox (see `_filesystem_read`).
- **What a pass writes is run and linted.** `_verify_written_files` executes
  every `RUNNABLE_SUFFIXES` file (a module inside a package is imported, not
  executed) and `_lint_written_files` runs `ruff check` with the project's own
  config first. Files that failed earlier are re-checked until clean. A file
  nobody ran (`unverified`), a lint failure, and a pass cut off before it
  finished all block approval; `expect_failures` suppresses only the
  runtime-failure block, never the others.
- Verification runs headless (`HEADLESS_VERIFY_ENV`) and with `stdin` closed.
- **`terminal_execute` has no shell.** The command is `shlex.split` and run
  directly, so metacharacters are inert data; `SHELL_OPERATORS` and
  `_GLUED_REDIRECT` refuse operators that survive as whole argv tokens, `cwd`
  replaces `cd` (validated by `_resolve_cwd`), and a requested `timeout` is
  clamped by `TERMINAL_TIMEOUT_MAX_SECONDS`. A timeout reports the tail of what
  the file printed.
- **`git_dwell` runs its stages in fixed order** (`dwell.py`) -- survey, branch,
  stage, commit, update, push, pr, checks, merge, cleanup -- whatever order the
  caller lists them in, and it will not commit onto the default branch (or a
  detached HEAD); a repository's first commit, with no base to branch from,
  founds its branch instead. A branch whose pull request already merged is
  finished: the work moves to a fresh branch from `origin/<default>`. `update`
  merges the default branch in before the push, so CI tests what will land; a
  push refused because the branch moved there merges it in and pushes once
  more. Never a rebase, never a force-push. Nothing staged is a success, and
  the rest still runs when the branch has commits to ship; `paths` commits
  only what it names (a pathspec on the commit, not just on the add). No
  `origin` skips every remote stage, as a success: the commit stays local.
- **`git_dwell` never merges past a check it did not read.** `checks` waits
  for the PR's CI on the pushed head, bounded: the node passes what is left of
  the Builder's deadline (`_checks_budget`), since a tool call is never
  abandoned. Red stops at `checks` with the failing jobs and `failed_log`;
  still running is *pending* -- success, PR left open, merge skipped. `merge`
  reads the checks itself when `checks` did not run, waits on people (changes
  requested, a required review, a draft) as pending, stops on a conflict, hands
  a protected branch to GitHub's auto-merge, and squash-merges with
  `--match-head-commit`, so a push after the checks were read is refused.
  `cleanup` runs only once GitHub says merged: the branch deleted there (a
  refusal is a warning) and here (`-D`, safe only because it merged), and the
  default branch here fast-forwarded to what landed. On a run given a project
  the git tools act in `projects/<name>`, and `git_dwell` refuses unless that is
  a repository of its own: git climbs to the checkout otherwise, where
  `projects/` is ignored and only the operator's own work could be staged.
- **A dwell that ships is proven first.** `_dwell_gate` runs the pass's own
  proof (`_prove`) before a call with push, pr, checks or merge reaches git; a
  failure stops it at `prove` with nothing committed. The pass's last dwell is
  `state["dwell"]` (`_dwell_record`): `checks_failed` blocks approval until a
  dwell sees them pass, and a run that ends `pending` hands the PR to the
  console, which finishes it (`_follow_pull_requests`, below).
- **`expect_failures`, `research_web` and `discuss_only` are per-run flags set
  by the caller, never by an agent.** A discussion run binds no tools at all
  and forces online research off.
- **Work run under `_with_deadline` must not write to state** -- the abandoned
  worker cannot be cancelled and may finish after the node returned. It is
  told instead (`control.abandoned`): a streamed seat call stops at its next
  token and lets go of `GPU_ARBITER`, since the socket timeout bounds only the
  gap between tokens. The
  Builder's deadline never abandons a tool call. A timed-out Architect can
  never rule `approved`, and a timed-out or off-format Planner must leave a
  non-empty `plan` (`_PLANNER_TIMED_OUT`, `_PLANNER_NO_STEPS`), since the gate
  counts a step only while a plan exists.
- **A silent Researcher is not research**: `_said_nothing` routes to the
  Builder and names the seat's model. Every exit from `researcher_node` but the
  emergency stop's (which ends the run anyway) sets `research_status`, and
  `_route_from_planner` forces the opening cycle through the Researcher.
- **Retrieval decides whether a seat is consulted at all.** `_gather_research`
  returns the retrieved chunks without invoking the Researcher's model whenever
  the best hit (`best_score`, the statistic the floor is measured on) clears
  `relevance_floor()`, marking any passage under it; below it, the seat judges
  the passages that came back (`_retrieval_for_the_seat`). The floor is
  measured per corpus into the corpus's own row in the database -- questions
  drawn from the corpus's own documents (a fetched page's goal, else a
  heading) against the off-domain ones in `embedding_calibration.json` --
  `None` until it has been, measured again once the corpus has changed
  (`corpus_signature`), and never borrowed between embedding models. The
  Planner's map asks for the floor before it searches. Search is hybrid:
  BM25 re-ranks the dense window
  (`lexical.py`), ranks fused rather than scores, so every result keeps the
  cosine the floor is read off.
- **Documents are embedded in chunks** (`CHUNK_MAX_TOKENS` 254 against a
  256-token window); `search` collapses chunks back onto documents, and a
  document's previous chunks are replaced by its new ones, with its graph
  node and edges, in one transaction (`replace_document`, `save_document_graph`).
- **The corpus lives in PostgreSQL** (`corpus_store.py`): one schema per
  corpus directory, named from its resolved path (`corpus_schema`), holding
  `chunks` (a `vector(EMBEDDING_DIMENSIONS)` column), `graph_nodes`,
  `graph_edges` and a one-row `corpus` table naming the embedding model and
  carrying the floor. A schema of another model reads as absent and is rebuilt
  empty by the creating door. **Search is exact cosine** -- pgvector's ANN
  indexes cap at 4,000 dimensions and the model answers in 4,096 -- so add no
  ANN index without quantizing, and re-measuring the floor. `kb.collection` is
  the store: its read API is the collection one the fakes in tests implement,
  and `tests/store_doubles.py` supplies the rest over them.
  `DATABASE_URL` names the server (`DEFAULT_DATABASE_URL` otherwise); the
  suite uses its own database, `langgraph_agent_test`, and `REQUIRE_POSTGRES=1`
  makes an unreachable server fail rather than skip the `postgres` tests.
- **No corpus exists until someone indexes one, and reading is not indexing.**
  `get_knowledge_base()` creates and is reserved for indexing;
  `open_knowledge_base()` returns `None` and is what every read goes through.
  The corpus is rebuilt when the console starts and again before every run
  (`REBUILD_CORPUS=0` switches off both), `_claim_the_rebuild` holds
  a session advisory lock keyed by the corpus's schema (`rebuild_claim`) so two
  consoles cannot interleave rebuilds, and a
  reindex rebuilds rather than accumulates while keeping the vectors of
  unchanged documents.
- **Changing the corpus is refused while a run or a rebuild is in flight**
  (`clear_corpus`, `upload_document`); `export_corpus` is not, since reading
  takes nothing away. `clear()` empties chunks, graph and floor record in one
  transaction, and only then the in-memory graph. `clear_corpus` first deletes
  what a rebuild would walk (`remove_corpus_sources`): fetched pages and uploads
  go, projects are opted out but never deleted. Without that, the next start
  embedded the same files again.
- **An uploaded document is a file first**: `store_uploaded_document` writes
  under `uploads/` and only then embeds, so `uploads/` must stay one of
  `CORPUS_ROOTS` or the next rebuild deletes the upload silently.
- Fetched pages under `research/web/` and `ENTITY_FREE_SUFFIXES` files are
  retrievable but mint no entities, decided from the path so a reindex cannot
  reverse it. `connectivity()` reports them apart from real isolates.
- **`ENTITY_STOPWORDS` is a hand-audited list, not a rule.** Re-run the
  position-free-capital count when the vocabulary moves -- in either direction,
  since deleting files moves it too -- and the pinned top-twenty in
  `test_claims.py` is what tells you it has.
- `CORPUS_SKIP_DIRS` are pruned by whole directory name under a corpus root, so
  an opted-in project's own `src/` and `tests/` are indexed and `rebuild/` is
  not `build/`.
- **The corpus is researched archive data, never the checkout.** The walk reads
  `CORPUS_ROOTS` alone -- `research/web/`, `uploads/`, and `projects/<name>`
  once opted in -- so a fresh install has no corpus, and anything else in the
  store is pruned by the next rebuild. An operator who wants a checkout file
  searchable uploads it. The entity census in `test_claims.py` walks the
  checkout explicitly (`roots=("",)`): it audits the extractor, not the corpus.
- **There is exactly one embedding model** (`EMBEDDING_MODEL_NAME`), and a
  corpus belongs to it: vectors from two models share no space. The chunker cuts
  with that model's own tokenizer.
- **Placement is the daemon's, and only one model is on the cards.**
  `OLLAMA_EMBED_OPTIONS` and `OLLAMA_SEAT_GPU_OPTIONS` force every layer onto
  the GPU, `GPU_ARBITER` (`control.py`) serializes the embedder against the
  seats and evicts the resident model, and a seat whose forced load does not
  fit is rebuilt unforced and retried once -- the embedder never is, since a
  split changes its vectors. Its `num_batch` is the chunker's window, not the
  load's: the last card, here also the display's, holds a compute buffer sized
  by it, and a chunk in one pass embeds bit-identically at any batch that holds
  it (`test_every_chunk_fits_one_pass_of_the_load`). The arbiter's wait is
  unbounded: a waiter never runs beside the holder. Across processes the daemon
  enforces the same through `OLLAMA_MAX_LOADED_MODELS=1` /
  `OLLAMA_NUM_PARALLEL=1`, a drop-in `install.sh` writes.
- **The emergency stop is cooperative.** `RUN_CONTROL` is a process-global
  checked at node tops, in the Builder's turn loop, before each verified file,
  between online-research pages and embedding batches, and between supersteps
  -- never inside a tool batch. Every exit path writes
  `runs/last_run.json`, and `shutdown` defers the exit to the run's own
  `finally` under `_run_lock`.
- **Three nested timeouts, none redundant**: `LLM_TIMEOUT_SECONDS` bounds one
  provider call at the socket (spelled differently per provider),
  `NODE_DEADLINE_SECONDS` / `BUILDER_DEADLINE_SECONDS` bound a node turn, and
  `RUN_BUDGET_SECONDS` bounds the run but is checked only between supersteps.
- The seat lights come from `ACTIVITY` and `run_progress.turns`, never
  `run_progress["node"]`, which names the seat that just *finished*; the
  embedder's light comes from `EMBEDDER_ACTIVITY`.
- A seat with no credentials silently becomes `StubLLM`. Key presence is not
  liveness: `get_agent_status()` reports the real outcome of each call
  (`_seat_failures`), and `stubbed` and `live` failures are worded apart.
- A seat's thinking box says what its next call does: a switchable model is
  always sent the flag, off included, since omitting it means *on*.
- Nothing is written under `knowledge/` any more; what an older console left
  there is gitignored, and `__pycache__` is never committed.
- **The container joins this machine's network** (`network_mode: host`): the
  daemon and SearxNG are loopback-only on the host. **Its database is compose's
  own `postgres` service** (pgvector image, `pgdata` volume, health-checked,
  `depends_on` it), on Docker's bridge and published on 127.0.0.1:5433 only --
  never on the host network, where a trust-auth server listens on every
  interface -- and compose sets `DATABASE_URL` to it over `.env`'s. The daemon
  stays on the host, only code and `.env` arrive from the checkout, read-only (every directory the
  app writes is a named volume, the corpus is in its own server,
  and the port is 8081, so a console on the host shares nothing with it), and `docker stop`
  asks for the same exit the console's X does.

- **The browser agent is the console's user, and read-only unless told.**
  `scripts/browser_agent.py` routes every `/rpc` call its pages make through a
  guard: a mutating method needs its `--allow` key *and* a loopback console, and
  `clear_corpus`, `set_seat`, `set_thinking`, `embed_project` and
  `dismiss_pull_request` are never sent; a `stop_run` must name its run, and
  Stop is pressed only on a run the agent started. The guard is the page's
  route, so shared and service workers are off and `eval` is for reading
  only: code it runs has the page's powers. `--spawn` runs a console of its own
  from a temporary directory (own corpus schema, `RUNS_DIR`,
  `FOLLOW_PULL_REQUESTS=0`); `--stub-seats` sets every key to an empty string,
  which `load_dotenv` cannot refill, and refuses to go on unless all four seats
  report a stub. Chromium's sandbox is off only as root, and Playwright is
  imported lazily, so CI tests the agent with none. `diagnose_machine.py` is
  read-only unless `--with-runs` and never starts a console; everything it
  writes passes through its `redact`. No `.mcp.json` is committed: the
  Playwright MCP server is registered per machine (`scripts/claude_tools.sh`),
  and `.claude/settings.json` denies its `browser_run_code_unsafe`.

## Self-healing

`self_healing` wraps calls rather than living inside them: `call_with_retry`
retries what its policy calls transient, a named `Circuit` stops calling a
service that keeps failing until one trial call after its cooldown succeeds
(half-open, it admits that one and refuses the rest; it keeps the books under
its lock and never holds it across a call, so it never serializes them),
and every action lands in the healing journal (`get_healing_logger()`), which
the console reads and a run's snapshot carries (each run is one healing
session). An open spell is one story however long it lasts: its first refusal
and first failed trial are journalled, the rest counted into the line that
closes it. Where it is used:

- **The database is one circuit, `POSTGRES`** (`corpus_store.py`), opened only
  by a server that cannot be reached (`database_unreachable`: no SQLSTATE, or a
  connection one) -- a deadlock or serialization failure is retried and never
  counted, and a broken connection takes the idle pool with it. A store call
  is one transaction, retried briefly while unreachable -- safe, since a
  transaction the connection dropped under was rolled back -- and a rebuild
  that meets the open circuit ends as `unavailable` naming it; the monitor
  checks the database (`postgres` in the health map) and redoes that rebuild
  once it answers. The corpus then reads `unavailable`, never `absent`.
- **The Ollama daemon is one circuit, `OLLAMA_DAEMON`** (`config.py`): seats,
  the embedder and every status read go through it (`daemon_request`), and only
  an unreachable daemon trips it (`daemon_unreachable`) -- an HTTP error is the
  daemon answering. `retry_unreachable` retries a daemon call briefly, since an
  unreachable daemon ran nothing; the emergency stop ends the wait. Underneath
  it systemd keeps the daemon running: both installers write a keep-alive
  drop-in (`scripts/ollama_keepalive.sh`) -- enabled at boot, `Restart=always`,
  no start limit. Every Ollama seat words an unreachable daemon the same way
  (`DAEMON_UNREACHABLE`), as OFFLINE, until it answers.
- **The embedder** retries a failed model load (5xx) on its own longer
  schedule, and a schedule that still ends in a 5xx opens `EMBEDDER_LOAD`
  (`embedder-load`): every embed is refused at once until a single-attempt
  trial after its cooldown loads. A rebuild that meets an open circuit ends as
  `unavailable`, naming it, instead of failing document by document.
- **Cloud seats** get one circuit per provider (`PROVIDER_CIRCUITS`), opened by
  outages (`provider_unavailable`), never by a 4xx. Their SDKs retry requests
  themselves, so nothing retries on top.
- **The GPU fallback** -- a forced load that did not fit, rebuilt unforced -- is
  journalled as a recovery action.
- **Online research**: one circuit per search backend; a DuckDuckGo bot check
  trips it at once. Searches and page fetches retry only failures a second try
  can fix (refused, dropped, 502-504), never a timeout.
- **`git_dwell`'s push and fetch** are retried when they never reached the
  remote; nothing else a Builder tool does is ever retried, since every other
  tool has effects.
- **A pull request a run left pending** is followed by the monitor
  (`_follow_pull_requests`, kept in `runs/pull_requests.json`): every
  `PULL_REQUEST_FOLLOW_SECONDS` it runs `finish_pull_request` -- checks, merge,
  cleanup, no waiting -- on the branch, number and head the run recorded, and
  merges only that head. Never while a run is in flight, and a run is refused
  while one is being finished. A red check is journalled and shown, never
  fixed; repeated trouble or `PULL_REQUEST_FOLLOW_HOURS` stops it. The header
  shows each as a chip. `FOLLOW_PULL_REQUESTS=0` switches the follow off and
  `RUNS_DIR` moves `runs/`, for a second console that only looks (the browser
  agent's `--spawn`, the installer's console check): it must never merge the
  real console's pull requests.
- **The monitor** (`_self_healing_monitor` in `serve.py`) checks the daemon,
  SearxNG and the corpus every `HEALTH_CHECK_SECONDS`, logs each change of
  health, re-probes open circuits, and rebuilds a corpus whose last rebuild
  stopped because the embedder could not be reached. The console shows an open
  circuit in the header (click it to let the next call through now) and the
  journal in the telemetry feed and the State tab. A corpus whose rebuild met
  `embedder-load` is rebuilt the same way, once that circuit's cooldown is over.

Retrying never extends a deadline's guarantee: work under `_with_deadline`
still never writes to state, and a retry inside an abandoned seat call ends on
its own within seconds.

## Documentation claims are tests

`tests/test_claims.py` exists because prose goes stale silently: **a claim
worth writing down is a claim worth failing a build over.** Most of its guards
*recompute* the claim from the repository -- every cited path resolves, the
Project Structure tree matches disk in both directions, the seat table matches
`DEFAULT_SEATS`, the documented Python floor matches `pyproject.toml`, a bare
`pytest` collects `tests/` alone, and CI runs the same commands the Quick
Reference does.

Two are different. A figure quoted in prose is checked against its constant
through a small registry -- better still, name the constant and let the reader
look it up. And the top-twenty entity audit **cannot** be a rule, so it is
*pinned*: when it fails, a newcomer has reached the top twenty and the audit is
asking to be redone, not reporting a bug. Its census counts only the files
every checkout has, so `uploads/`, `projects/` and fetched pages are out.

## Troubleshooting

- **GraphRAG returns no results** -- Check whether there is a corpus at all; the
  header reads `no corpus` when none has been built. One is built when the
  console starts and again before the Architect opens, from the directory the
  server was started in, so the usual causes are a server started somewhere
  with nothing to index, or `REBUILD_CORPUS=0`.
- **No LLM output / canned text** -- A seat pointed at Anthropic needs
  `ANTHROPIC_API_KEY` in `.env`; without one it runs `StubLLM` and the console
  shows a `NO KEY` chip. Ollama seats need the daemon running and signed in
  (`ollama signin`) for `:cloud` tags.
- **A 400 from Anthropic that looks like an auth error** -- Something is passing
  `temperature` to an Opus 5 / Sonnet 5 / 4.6+ model.
- **The embedder is on the CPU, or embedding is slow** -- `ollama ps` names the
  split. `journalctl -u ollama` reading `skipping CUDA device` means Arch's
  CUDA 13 build on a card it cannot drive: run `./cuda-embed-ollama.sh`.
- **A local seat runs mostly on the CPU, or one card looks untouched** --
  the forced load failed and the seat rebuilt itself unforced; the journal names
  the allocation that did not fit. A model larger than the cards is *meant* to
  read that way. Two models resident at once means something reached the daemon
  around `GPU_ARBITER`.
- **The header reads `stale: N not indexed` right after startup** -- Read the
  `[Corpus]` line in `/tmp/ambiguity-console.log`: it names each file that
  failed and why, usually a model load short of GPU memory.
- **The header shows `ollama-daemon down`** -- the daemon stopped answering and
  its circuit opened, so calls fail at once instead of each waiting it out.
  It closes by itself once the daemon answers a trial call; clicking the chip
  lets the next call through now. A rebuild it interrupted is redone by the
  monitor once the daemon is back. systemd restarts a daemon that exited;
  `scripts/ollama_keepalive.sh check` says whether the installer set that up.
- **The header shows `embedder-load down`** -- the embedding model's forced
  load failed a whole retry schedule. `journalctl -u ollama` names the card and
  the allocation that failed, and `nvidia-smi` what else holds that card -- on
  the display card, the desktop and any browser. Click the chip to retry now.
- **Online research finds nothing, or reports DuckDuckGo's bot check** -- Check
  `SEARXNG_URL` is in `.env` and the instance answers; an HTTP 403 means `json`
  is missing from its `search.formats`.
- **`docker` says permission denied, or nothing answers on port 5432** -- The
  `docker` group applies only after a reboot, and Omarchy enables only the
  socket, so `docker.service` must be enabled for the database to come back up.
- **The header shows `postgres down`, or the corpus reads `unavailable`** -- the
  database stopped answering: `docker ps | grep postgres18`, then
  `docker logs postgres18`. A server without pgvector is reported by name in the
  health map; `./install.sh` moves `postgres18` to
  `pgvector/pgvector:pg18-trixie` on the same data volume.
- **The corpus tests are skipped** -- no server answers at `DATABASE_URL`;
  the suite's database is created on that server on first use.

## Where the reasoning went

This file used to carry the rationale behind every rule above -- what was
measured, what was tried and rejected, and which incident each guard came from.
It was cut on 2026-09-20 because a memory file is loaded into every session and
it had reached 150,060 characters. **The full text is the commit before that
one:** `git show 2074d08:CLAUDE.md`, and `git log -p CLAUDE.md` for how each
rule arrived. Before changing a rule above, read its paragraph there -- most of
them record an alternative that was built, measured and rejected.
