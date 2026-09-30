# Claude Code Project Instructions

## Project Overview

This is **langgraph-agent**, a local-first 4-Agent AI system for software development experiments.

- **Architect** — the leading authority. Sets architectural direction before planning, then holds the approval gate: the run ends on its `approved` verdict, not the Builder's say-so. No tools.
- **Planner** — interprets goals, creates structured plans, routes to next agent.
- **Researcher** — gathers context via the GraphRAG MCP tool (`search_knowledge_graph`).
- **Builder** — implements plans using filesystem, git, terminal, and test MCP tools.

Inference defaults to local: every seat runs a model the Ollama daemon on this
machine serves from its own weights, so a fresh checkout needs no API key and
no ollama.com credentials at all. Ollama Cloud tags, Anthropic and OpenAI
remain available per seat for anyone who wants them. The embedding model also
runs locally through the same daemon, serving `qwen3-embedding:latest` -- the
daemon owns every model's placement, so nothing in this project touches torch
or a card itself. The embedding belongs to GraphRAG, not to a seat.

Tech stack: Python 3.12+, LangGraph, Chroma + NetworkX, MCP (local stdio-compatible tool binding).

## Quick Reference

```bash
# Install everything on Arch / Omarchy (packages, venv, Ollama models, SearxNG,
# PostgreSQL in Docker), then prove the embedder, the database, git/gh and
# every seat actually work
./install.sh

# Install dependencies
pip install -e ".[dev]"

# Run all checks
ruff check src/ tests/ serve.py scripts/ example_usage.py test_cloud.py ollama_client.py
mypy src/langgraph_agent/ serve.py ollama_client.py
python -m pytest tests/ -v

# Start the web console
./launch_console.sh

# Or the same console in a container (Arch image, host networking)
docker compose up --build

# Find out which model actually works in which seat
python scripts/diagnose_seats.py --list
python scripts/diagnose_seats.py --phase probe

# Run the example
python example_usage.py
```

## Project Structure

```
├── src/langgraph_agent/
│   ├── __init__.py            # create_agent_graph, AgentState, ResearchStatus, Verdict
│   ├── state.py               # AgentState schema + ResearchStatus / Verdict enums
│   ├── config.py              # Seats, LLM setup (Anthropic + Ollama Cloud) + StubLLM
│   ├── nodes.py               # Architect, Planner, Researcher, Builder nodes + prompt loading
│   ├── graph.py               # StateGraph wiring + conditional edges
│   ├── control.py             # RUN_CONTROL: the emergency stop signal
│   ├── graphrag_server.py     # GraphRAG MCP server (knowledge graph + vector store)
│   ├── embedding_calibration.json # Floor questions for a new embedding model (JSON: never indexed)
│   ├── mcp_client.py          # MCP client / local tool bindings
│   ├── lexical.py             # BM25 + rank fusion: the lexical half of search
│   ├── web_research.py        # Online research: keyless search, our own selection gate
│   ├── html_text.py           # HTML → text by link density. No library, no dependency
│   ├── corpus_health.py       # Is the corpus still the project? missing / extra / oversized
│   ├── corpus_spectral.py     # connectivity / topics / bottleneck / duplicate_entities (mixin)
│   ├── projects.py            # Generated projects: where a run writes, and opting one into the corpus
│   ├── exceptions.py          # Public error surface (re-exports _internal/)
│   ├── self_healing/          # Opt-in retry / circuit-breaker decorators (see note below)
│   │   ├── logger.py          # SelfHealingLogger: structured, severity-leveled healing log
│   │   └── decorators.py      # retry_with_backoff, circuit_breaker, self_healing_wrapper
│   └── _internal/
│       └── exceptions.py      # LangGraphAgentError and its five subclasses
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   ├── conftest.py            # Forces StubLLM; switches the web phase off
│   ├── test_git_dwell.py      # The ordered git pipeline, and its two refusals
│   ├── test_claims.py         # Documentation claims, made executable
│   ├── test_corpus_absent.py  # The two doors: reading never creates a corpus
│   ├── test_graph.py          # Pytest suite
│   ├── test_diagnose_seats.py # Guards the seat diagnostic's verdicts
│   ├── test_thinking.py       # Per-seat capability: the thinking switch, and tool support
│   ├── test_console_stop.py   # Emergency stop, deferred exit, snapshot
│   ├── test_chunking.py       # Document chunking, chunk ids, search collapse
│   ├── test_corpus_admin.py   # Corpus clear / export / reindex guards
│   ├── test_embedding_device.py # The one embedder, and the light that says when it is working
│   ├── test_mcp_tools.py      # Builder tool belt
│   ├── test_imports.py        # Pins the package's public surface
│   ├── test_lexical.py        # BM25, rank fusion, the relevance floor
│   ├── test_self_healing.py   # self_healing: logger, retry, circuit breaker
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
│   └── test_spectral_graph.py # The spectral_graph package
├── scripts/
│   ├── verify_and_test.py     # Manual verification runner
│   ├── auto_verify.py         # Silent verification
│   ├── quick_test.sh          # Bash quick check
│   ├── full_setup.py          # Automated setup + verification
│   ├── spectral_benchmark.py  # Graph-architecture sweep behind the A-numbers
│   └── diagnose_seats.py      # Role probes + team runs per seating
├── frontend/
│   ├── index.html             # Web console SPA
│   └── README.md
├── .github/workflows/
│   └── ci.yml                 # ruff, mypy, pytest, root scripts, on every push
├── spectral_graph/            # the A1-A5 spectral code: at the root, outside the installed package
│   ├── laplacian.py           # Laplacian matrix constructions
│   ├── spectrum.py            # eigenvalues, eigenpairs, algebraic connectivity
│   ├── fiedler.py             # the Fiedler vector and spectral bipartitioning
│   ├── clustering.py          # spectral clustering, conductance, the Cheeger bounds
│   ├── embedding.py           # spectral embedding via Laplacian eigenvectors
│   ├── operations.py          # spectral graph arithmetic
│   └── stability.py           # numerical stability utilities
├── experimental/              # an agent run's own notes; out of the corpus (PROJECT_INDEX_EXCLUDES)
├── docker/
│   └── entrypoint.sh          # the container's start: git identity, and what it can reach
├── Dockerfile                 # the console as an Arch image; the daemon stays on the host
├── Dockerfile.kali            # the same console on Kali rolling; gh from GitHub's own repo
├── docker-compose.yml         # host networking: the daemon, the database and SearxNG are on it
├── .dockerignore              # the venv, the caches, and every per-machine artifact
├── install.sh                 # Arch / Omarchy: everything, from nothing to a running console
├── cuda-embed-ollama.sh       # NVIDIA cards below compute 7.5: Ollama's CUDA 12 build, model 100% on the GPU
├── launch_console.sh          # starts serve.py and waits on /api/status before opening a browser
├── serve.py                   # Python HTTP server + API backend
├── example_usage.py           # Demo script
├── test_cloud.py              # Cloud LLM end-to-end test
├── ollama_client.py           # one prompt to the local daemon, bounded by the project's own timeout
├── test_spectral_graph.py     # imports every spectral_graph module; run from the root by CI
├── verify_spectrum.py         # spectral_graph.spectrum, run from the root by CI
├── verify_fiedler.py          # spectral_graph.fiedler, run from the root by CI
├── verify_clustering.py       # spectral_graph.clustering, run from the root by CI
├── verify_embedding.py        # spectral_graph.embedding, run from the root by CI
├── README.md                  # User-facing documentation
└── .env.example               # Environment variables template
```

## Conventions

- **Python formatting/linting:** `ruff` configured in `ruff.toml`.
- **Type checking:** `mypy` configured in `mypy.ini`.
- **Tests:** `pytest` in `tests/`.
- **Default seats** (`DEFAULT_SEATS` in `config.py`):

  | Seat | Provider | Model |
  |---|---|---|
  | Architect | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Planner | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Researcher | ollama | `hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M` |
  | Builder | ollama | `qwen3.8:latest` |

  Three seats run local weights the daemon holds itself; the Builder is the
  exception, since its work *is* tool calls and neither dolphin tag reports
  `tools` -- `qwen3.8:latest` does. Anthropic, OpenAI and Ollama Cloud tags
  remain available per seat; no seat uses any of them by default, so a fresh
  checkout runs without an API key or ollama.com credentials of its own.
- **Never send `temperature` to a modern Anthropic model.** Sampling parameters
  were removed on the Opus 5 / Sonnet 5 / 4.6+ families and are rejected with a
  400 that reads like an auth failure. `_accepts_temperature()` gates this.
- **Tool specialization (critical):** Architect and Planner get no tools; the
  Researcher's node calls GraphRAG read-only tools only, and its seat is handed
  what they returned rather than the tools; the Builder gets filesystem, git,
  terminal and test tools only. Keep GraphRAG out of `BUILDER_TOOLS`.
- **`self_healing` is a standalone utility, not wired into any node.** Reach
  for its `retry_with_backoff` / `circuit_breaker` decorators at a *new*
  integration point, never retrofitted onto the seat or MCP call paths, which
  have a resilience design of their own (the deadlines below, and the rule that
  work under a deadline never writes to state).

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
  "builder_cut_off": "" | "turn_cap" | "deadline", "lint_failed": list[str],
  "expect_failures": bool, "discuss_only": bool, "output_dir": str,
  "step_count": int,
}
```

`stopped`, `stop_reason`, `run_id` and `elapsed_s` are added to the payload
`run_goal` returns, not to `AgentState`: that schema describes what the agents
write, not how the run ended.

The Architect writes `architecture` and `verdict`; the verdict is what routes
the loop and what ends it. `step_count` is incremented by the Architect gate,
not by the Builder — every cycle passes the gate, but a Planner/Researcher loop
never reaches the Builder and would otherwise run uncounted.

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
`serve.py`, and the `/api/*` routes are compatibility wrappers over the same
functions (`launch_console.sh` polls `/api/status`, so it must keep working).
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
- **`git_dwell` runs its stages in fixed order** -- survey, branch, stage,
  commit, push, pr, merge -- whatever order the caller lists them in, and it
  will not commit onto the default branch. The default pipeline ends at
  `merge` (`--squash --delete-branch`); naming `stages` without it stops at
  `pr`. Nothing staged is a success, and `paths` commits only what it names.
- **`expect_failures`, `research_web` and `discuss_only` are per-run flags set
  by the caller, never by an agent.** A discussion run binds no tools at all
  and forces online research off.
- **Work run under `_with_deadline` must not write to state** -- the abandoned
  worker cannot be cancelled and may finish after the node returned. The
  Builder's deadline never abandons a tool call. A timed-out Architect can
  never rule `approved`, and a timed-out or off-format Planner must leave a
  non-empty `plan` (`_PLANNER_TIMED_OUT`, `_PLANNER_NO_STEPS`), since the gate
  counts a step only while a plan exists.
- **A silent Researcher is not research**: `_said_nothing` routes to the
  Builder and names the seat's model. Every exit from `researcher_node` sets
  `research_status`, and `_route_from_planner` forces the opening cycle through
  the Researcher.
- **Retrieval decides whether a seat is consulted at all.** `_gather_research`
  returns the retrieved chunks without invoking the Researcher's model whenever
  the top hit clears `relevance_floor()`, marking any passage under it; below
  it, the seat judges the passages that came back (`_retrieval_for_the_seat`).
  The floor is measured per corpus into
  `floor_calibration.json`, `None` until it has been, and never borrowed
  between embedding models. Search is hybrid: BM25 re-ranks the dense window
  (`lexical.py`), ranks fused rather than scores, so every result keeps the
  cosine the floor is read off.
- **Documents are embedded in chunks** (`CHUNK_MAX_TOKENS` 254 against a
  256-token window); `search` collapses chunks back onto documents, and a
  document's previous chunks are deleted before its new ones land.
- **No corpus exists until someone indexes one, and reading is not indexing.**
  `get_knowledge_base()` creates and is reserved for indexing;
  `open_knowledge_base()` returns `None` and is what every read goes through.
  The corpus is rebuilt when the console starts and again before every run
  (`INDEX_PROJECT_BEFORE_RUN=0` switches off both), `_claim_the_rebuild` holds
  an `flock` beside the store so two consoles cannot interleave rebuilds, and a
  reindex rebuilds rather than accumulates while keeping the vectors of
  unchanged documents.
- **Changing the corpus is refused while a run or a rebuild is in flight**
  (`clear_corpus`, `upload_document`); `export_corpus` is not, since reading
  takes nothing away. `clear()` empties Chroma first, then the graph, then the
  floor record, and must reach disk.
- **An uploaded document is a file first**: `store_uploaded_document` writes
  under `uploads/` and only then embeds, so `uploads/` must stay *out* of
  `PROJECT_INDEX_EXCLUDES` or the next rebuild deletes the upload silently.
- Fetched pages under `research/web/` and `ENTITY_FREE_SUFFIXES` files are
  retrievable but mint no entities, decided from the path so a reindex cannot
  reverse it. `connectivity()` reports them apart from real isolates.
- **`ENTITY_STOPWORDS` is a hand-audited list, not a rule.** Re-run the
  position-free-capital count when the vocabulary moves -- in either direction,
  since deleting files moves it too -- and the pinned top-twenty in
  `test_claims.py` is what tells you it has.
- `PROJECT_INDEX_EXCLUDES` entries are matched as plain substrings, not globs.
- **The corpus never indexes this program's own source.** `src/`, `tests/`,
  `prompts/`, `frontend/` and `spectral_graph/` are in `PROJECT_INDEX_EXCLUDES`,
  so GraphRAG holds project knowledge -- docs, config, whatever a run is
  actually about -- rather than the four agents' own code and prompts. An
  operator who wants the source searchable opts it in the same way as any
  other document: upload it, or embed a generated project explicitly.
- **There is exactly one embedding model** (`EMBEDDING_MODEL_NAME`), a corpus
  belongs to it, and `set_embedding_model` refuses: vectors from two models
  share no space. The chunker cuts with that model's own tokenizer.
- **Placement is the daemon's, and only one model is on the cards.**
  `OLLAMA_EMBED_OPTIONS` and `OLLAMA_SEAT_GPU_OPTIONS` force every layer onto
  the GPU, `GPU_ARBITER` (`control.py`) serializes the embedder against the
  seats and evicts the resident model, and a seat whose forced load does not
  fit is rebuilt unforced and retried once.
- **The emergency stop is cooperative.** `RUN_CONTROL` is a process-global
  checked at node tops, in the Builder's turn loop, before each verified file
  and between supersteps -- never inside a tool batch. Every exit path writes
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
- Knowledge base files under `knowledge/` are runtime artifacts, and
  `__pycache__` is never committed.
- **The container joins this machine's network** (`network_mode: host`): the
  daemon, PostgreSQL and SearxNG are all loopback-only on the host. The daemon
  stays on the host, the checkout arrives as a bind mount, and `docker stop`
  asks for the same exit the console's X does.

## Documentation claims are tests

`tests/test_claims.py` exists because prose goes stale silently: **a claim
worth writing down is a claim worth failing a build over.** Most of its guards
*recompute* the claim from the repository -- every cited path resolves, the
Project Structure tree matches disk in both directions, the seat table matches
`DEFAULT_SEATS`, the documented Python floor matches `pyproject.toml`, no
walked file exceeds `MAX_INDEXABLE_BYTES` (and none passes
`INDEXABLE_WARN_RATIO` of it), and CI runs the same commands the Quick
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
  with nothing to index, or `INDEX_PROJECT_BEFORE_RUN=0`.
- **No LLM output / canned text** -- A seat pointed at Anthropic or OpenAI needs
  that provider's key in `.env`; without one it runs `StubLLM` and the console
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
- **Online research finds nothing, or reports DuckDuckGo's bot check** -- Check
  `SEARXNG_URL` is in `.env` and the instance answers; an HTTP 403 means `json`
  is missing from its `search.formats`.
- **`docker` says permission denied, or nothing answers on port 5432** -- The
  `docker` group applies only after a reboot, and Omarchy enables only the
  socket, so `docker.service` must be enabled for the database to come back up.
- **Tests are slow** -- The first run opens Chroma; later runs reuse the cached
  singleton.

## Where the reasoning went

This file used to carry the rationale behind every rule above -- what was
measured, what was tried and rejected, and which incident each guard came from.
It was cut on 2026-09-20 because a memory file is loaded into every session and
it had reached 150,060 characters. **The full text is the commit before that
one:** `git show 2074d08:CLAUDE.md`, and `git log -p CLAUDE.md` for how each
rule arrived. Before changing a rule above, read its paragraph there -- most of
them record an alternative that was built, measured and rejected.
