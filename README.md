# 4-Agent AI System

**Architect · Planner · Researcher · Builder**

A multi-agent system for software development experiments, powered by LangGraph and GraphRAG.

Inference is **local by default**. Three of the four seats run a model the
local Ollama daemon serves from its own weights; the Builder runs a different
local model because its work *is* tool calls, and the two local dolphin tags
don't report `tools`. Nothing needs an API key or ollama.com credentials out
of the box. Ollama Cloud tags and Anthropic are still available per seat if
you want them.

## 🎨 Web Console

```bash
# Quick launch
./launch_console.sh

# Or manually
python serve.py
# Open: http://localhost:8080
```

Five tabs: **Engineer** (give the Architect a goal, watch the stages),
**Graph** (the knowledge graph as a force-directed map), **Retrieval**
(semantic search plus a telemetry log of RPCs and healing events), **Corpus**
(the indexed documents, and buttons to upload, export or clear them),
and **State** (self-healing status, then the raw `AgentState`).

**The console heals what it can by itself.** A dead Ollama daemon or search
backend opens a *circuit*: calls to it fail at once instead of each waiting it
out, the header shows a red `… down` chip, and the first call after a short
cooldown tests whether it is back (click the chip to test now). A seat call or
an embed that meets a daemon mid-restart is retried briefly; a corpus rebuild
the daemon interrupted is redone once it answers again. The State tab lists
each service's health, every circuit, and the healing journal; a run's
snapshot carries its own healing events.

*Attach* — in the Engineer tab, beside Run — puts a document of your own into
the corpus, and so does *Upload documents* on the Corpus tab; they are the same
thing reached from two places, and you can drop files straight onto the
transcript instead. The Researcher retrieves what you add from the next run
onward, and the reply says where each file went and how many passages it became.

An upload is **written to `uploads/` and indexed from there**, rather than
embedded directly into the store. That is what makes it survive: a reindex
rebuilds the corpus from the files on disk and prunes everything else, so a
document that lived only in the index would vanish at the next rebuild without
anything saying so. Text only — the types a reindex reads, which the upload
buttons list on hover. There is no PDF
extractor here, and one is refused rather than embedded as whatever its bytes
decode to, which would look like a real source in the corpus afterwards. The
same size limit a reindex applies is applied on the way in, for the same
reason. Uploads are gitignored; that hides them from git, not from the corpus.

*Export* downloads the whole corpus as one JSON file — the knowledge graph plus
every chunk with its text and metadata. Embeddings are left out: they are most
of the bytes and the least portable part, and the embedder is local, so a
reindex regenerates them.

*Clear corpus* empties the store in place. The files under `knowledge/` stay,
holding an empty index — the same shape a reindex leaves behind. It arms on the
first click and disarms itself after a few seconds.

**The corpus is an archive, not the project.** It holds exactly three things:
pages the online research phase fetched (`research/web/`), documents you
uploaded (`uploads/`), and generated projects you opted in (`projects/<name>`).
The checkout itself — README, CLAUDE.md, install.sh, config, source — is never
walked or embedded, so a fresh install has no corpus at all until you research
or upload something.

**Every run brings that archive's index up to date, and there is nothing to
press.** There is no Reindex button. A run, and the console coming up, re-read
whatever has changed in those three places and prune whatever has left them. A
document whose text has not changed keeps the vectors it already has, so a
rebuild with nothing to do costs well under a second and never loads the
embedding model.
Set `REBUILD_CORPUS=0` for a machine that wants its corpus frozen.
Indexing being the only act that creates the store is why the header keeps
three states apart: *absent* (nobody has indexed here),
*empty* (a corpus that exists and holds nothing — what *Clear corpus* leaves),
and the counts, once there is something to count. *Export* and *Clear* are
disabled while it is absent; creating a store in order to empty it would leave
behind the thing you were asking to be rid of. The embedding model —
`qwen3-embedding:latest`, served by the local Ollama daemon — is loaded by the
daemon on the first index or search, not at startup; the daemon owns which
card it runs on, and the Crew panel's embedder card reports how much of it the
daemon kept on the CPU.

*Clear corpus* and an upload are both refused while a run is in flight, and
the refusal says which run. (The rebuild is the exception that proves it: it
happens *before* the run starts, which is the only ordering that obeys the same
rule rather than needing an exemption from it.) The Researcher searches this corpus, and
changing it underneath a run does not fail its search — an emptied corpus
answers "nothing found", and one midway through a rebuild answers from the part
of itself that exists so far. The run would plan around an absence that was
manufactured out from under it, without anything raising. An upload cannot do
that — it only adds — but it does mutate the graph a search may be reading, and
a run is best answered by the corpus it started against.

The left rail is the crew: one card per seat, each with its model, where the
prompt goes (`REMOTE` / `LOCAL`), and a status chip when the seat cannot
actually run — `NO KEY`, `FAILING`, `OFFLINE` or `NOT PULLED`.

Each card also has a **thinking** checkbox: ticked, the seat's model reasons
before it answers; unticked, it answers straight away — faster and cheaper,
usually weaker on hard steps. It starts unticked, and the flag is sent either
way, so an unticked seat is told not to think rather than left to its model's
own default. The box shows what the next call will
actually do, and it is grayed out when the model offers no switch: it cannot
think (the dolphin tags), always thinks (Claude Fable), or the Ollama daemon did not
say — hover for which. Like a model change, it lasts until the server restarts.

### Stopping a run

**Stop** sits next to Run and halts the run in flight. It is cooperative: work
already started — a file write, a staged commit, a test run, a model call —
finishes first, and nothing further is begun, so a stopped run never leaves a
half-written file behind. Expect it to land within one tool call rather than
instantly.

Nothing is thrown away. The run returns its state as it stood, the console shows
what was written, what nobody got to run, and what blocked, and the server keeps
that snapshot in `runs/last_run.json` — so a reload, or a run that died on a
failing seat, still has something to show. Reloading during a run reattaches to
it, Stop included.

The **×** in the top right shuts down the console and the server, the same way
Ctrl+C does. With a run in flight it asks first, then stops the run and waits
for it to write its snapshot before the process goes — so exiting mid-run is as
recoverable as stopping one.

The **expect failures** checkbox is unchanged by this. It excuses a file the run
*meant* to fail, which is still executed and still reported; a file the stop
prevented anyone from running is unproven rather than expected, so it goes on
blocking approval either way.

See [`frontend/README.md`](frontend/README.md) for full documentation.

## Architecture

```
START → Architect → Planner → (Researcher | Builder) → Architect → END
             ^                                             |
             +--------- revise / need_research -------------+
```

The Architect runs twice per cycle: once to set direction before anything is
planned, and again as the approval gate. The Builder does not decide the work is
finished — it reports, and the authority that set the constraints rules on it.

**The opening cycle always reaches the Researcher**, whichever agent the Planner
named. A run rebuilds the corpus before the Architect opens and may embed
fetched web pages into it, and nothing else forces anything to read it: the
Researcher ran only when a plan happened to route there, so a confidently
written plan meant a run that built a corpus and consulted none of it. On a goal
the corpus answers, that hop costs a search and no model call at all — retrieval
returns straight from GraphRAG whenever the top hit clears the relevance floor.
Later cycles route as the Planner asks.

### The Four Agents

| Agent | Responsibility | Default seat | Tools |
|---|---|---|---|
| **Architect** | Sets direction and constraints; rules `approved` / `revise` / `need_research` | Dolphin 2.9.1 9B (ollama, local) | None (reasoning only) |
| **Planner** | Turns goals into structured plans, routes next | Dolphin 2.9.1 9B (ollama, local) | None (reasoning only) |
| **Researcher** | Gathers deep, relationship-aware knowledge | Dolphin 2.9.1 9B (ollama, local) | GraphRAG search (read-only) |
| **Builder** | Implements the plan (writes code, edits files) | `qwen3.8:latest` (ollama, local) | Filesystem, Git, Terminal, Tests |

Every seat is reassignable live from its dropdown in the console, which lists
exactly what `ollama ls` reports (embedders excluded), so a tag you pull appears
on the next poll; selections last for the life of the process. `qwen3-embedding:latest`
is the embedder's, not a seat's: it cannot chat. Only `qwen3.8:latest`
reports `tools`, which is why it -- not either dolphin -- holds the Builder.

### Building blocks

| Part | Role |
|---|---|
| **LangGraph** | Orchestration: control flow, shared state, loops |
| **GraphRAG** | Retrieval: hybrid search over a chunked corpus, beside an entity graph |
| **Tool belts** | Each seat's tools, served in-process under MCP-style names |
| **Self-healing** | Retries, circuit breakers and a healing journal around every external call |

## Installation

On Arch / Omarchy, one command does all of it: system packages, `.env`, a
`.venv`, the Ollama daemon and its sign-in, the seat and embedding models (the
embedding model included, by `ollama pull`), the embedding model's tokenizer, a
SearxNG for online research, a PostgreSQL in Docker, the checks, and an
"Ambiguity Console" entry in the app launcher. Then it proves the result rather
than assuming it: the embedder embeds, the database answers a query, git and gh
can finish the Builder's pipeline, the console starts, and every seat answers a
test prompt. It is safe to re-run.

It builds no corpus — the corpus holds only what you research or upload.
The GPU driver is the machine's own setup (Omarchy installs it), but
where the embedding model runs is not left to chance: the app loads it with
every layer on the GPU, and on NVIDIA cards below compute capability 7.5 —
the Maxwell, Pascal and Volta cards Omarchy drives with its `nvidia-580xx`
driver, which CUDA 13 no longer supports — the installer runs
`cuda-embed-ollama.sh`. That script replaces Arch's CUDA 13 Ollama build with
Ollama's own CUDA 12 build of the same version and proves the model sits 100%
on the GPU; run it with `--check` to see where it sits now.

The database is the one Omarchy's own installer runs: `postgres:18` as the
`postgres18` container, published on 127.0.0.1:5432 only, with no password. The
installer enables `docker.service` so it survives a reboot, adds you to the
`docker` group (root-equivalent, and applied after a reboot;
`--no-docker-group` keeps Docker behind sudo), and writes `DATABASE_URL` into
`.env`. Nothing in the app reads that URL — the corpus stays in Chroma — but
the console exports `.env` to everything it runs, so a script the Builder
writes can use it. `--no-postgres` skips the database entirely.

```bash
./install.sh            # --help lists --minimal, --no-system, --no-searxng, ...
```

Everything it installs is free to use. Elsewhere, or by hand:

```bash
pip install -e ".[dev]"
```

### Running in a container

An Arch image of the console, wired to the machine it runs on:

```bash
docker compose up --build        # http://localhost:8081
```

**Host networking is the configuration**, and it is what makes the database
work without a line of its own. The console talks to three things on this
machine, and all three are loopback-only on purpose: the Ollama daemon on
127.0.0.1:11434, the `postgres18` container on 127.0.0.1:5432 -- published
there and nowhere else, which is why it can afford trust authentication -- and
the SearxNG on 127.0.0.1:8888. A bridged container reaches none of them, so
`DATABASE_URL` and `OLLAMA_BASE_URL` mean the same thing inside the container
as outside it, and nothing has to be republished to the world.

What stays on the host: the **Ollama daemon**. It owns the embedding model's
placement on the GPU and holds the ollama.com credentials the `:cloud` tags are
proxied with, so the image only ever speaks HTTP to it. The entrypoint says
whether it answers, because with no daemon behind them every default seat fails
its first call and the corpus cannot be embedded.

What is shared with the host Python install: **code and `.env`, read-only,
and nothing else.** `src/`, `serve.py`, `frontend/`, `prompts/` and
`spectral_graph/` are mounted from the checkout so an edit shows on restart, but
the container cannot write to them. Everything the app writes -- `knowledge/`
(the corpus and the relevance floor measured for it), `runs/`, `uploads/`,
`research/web/`, `projects/` and the rest -- is a named Docker volume, so the
container and a console started on the host (`./launch_console.sh`, port 8080)
never share a corpus, a run snapshot or a generated project, and can run at
once. The container is on **8081** (`AMBIGUITY_PORT` moves it). Volumes survive
a rebuild; `docker compose down -v` forgets them. To hand the container a
document, upload it through its console. It runs as uid 1000
(`AMBIGUITY_UID`/`AMBIGUITY_GID` for any other account).

`git_dwell`'s `push`, `pr` and `merge` stages need your own credentials:
uncomment the `~/.gitconfig` and `~/.config/gh` mounts in `docker-compose.yml`.
Without them commits carry a fallback identity and `gh` has no account.

`docker compose stop` is the console's exit button rather than a kill -- a run
in flight is stopped and the exit deferred until it has written its snapshot,
so it stays as recoverable as a stop from the browser. A stop never cuts a seat
call short, so that can take as long as the call in flight:
`stop_grace_period` outlasts the longest node deadline
(`BUILDER_DEADLINE_SECONDS`), and `docker compose kill` is the kill.

If Docker is still behind sudo (the `docker` group applies after a reboot), the
same files work rootless with Podman:

```bash
podman build --format docker -f dockerfile -t ambiguity-console .
podman run --rm --network host --userns=keep-id --stop-timeout 900 -v "$PWD":/app ambiguity-console
```

`--stop-timeout` is the same allowance: without it `podman stop` kills the
console after ten seconds, before a run in flight has written its snapshot.

**Bridged instead**, if the host network is not acceptable: attach both
containers to a network of their own, name the database by container, and tell
the daemon to listen past loopback.

```bash
docker network create ambiguity-net
docker network connect ambiguity-net postgres18
# then, in docker-compose.yml: drop network_mode, join ambiguity-net, and set
#   DATABASE_URL=postgresql://postgres@postgres18:5432/postgres
#   OLLAMA_BASE_URL=http://host.docker.internal:11434
#   CONSOLE_HOST=0.0.0.0          # the container's own interfaces
# with ports: ["127.0.0.1:8080:8080"] so the host reaches the console on its
# loopback only, extra_hosts: ["host.docker.internal:host-gateway"], and the
# daemon started with OLLAMA_HOST=0.0.0.0 so it answers off loopback.
```

The console listens on loopback unless `CONSOLE_HOST` says otherwise, because
it runs goals and the Builder runs programs. Inside a bridged container
loopback is the container's own, hence `CONSOLE_HOST=0.0.0.0` there -- with the
published port bound to the host's `127.0.0.1`, so nothing else on the network
reaches it.

That is three changes to solve one problem, and the first of them is what the
database's missing password rests on -- which is why it is not the default.

### Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

No API key is required: every default seat runs a model the local Ollama daemon
serves from its own weights. Sign the daemon in (`ollama signin`) only if you
move a seat onto a `:cloud` tag.

A key is only needed if you move a seat onto Anthropic:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

A seat pointed at a provider whose key is missing falls back to a canned stub.
It does so **visibly** — the seat card shows a `NO KEY` chip and the console
banners it — rather than pretending to run the model.

A key that exists but does not work (no credits, rate limited, model not on the
account) is a different failure: the seat shows `FAILING` with the provider's own
message and runs abort rather than quietly producing stub text. That state is
recorded from the actual outcome of a call, so it appears after the first run.

The local dolphin/qwen3.8 tags need only the daemon running -- no sign-in,
no credentials, since the daemon serves them from its own weights:

```bash
ollama pull hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M   # Architect, Planner, Researcher
ollama pull qwen3.8:latest                                           # Builder
```

Point a seat at an Ollama Cloud tag instead (`ARCHITECT_MODEL=nemotron-3-ultra:cloud`,
say) and it needs the daemon signed in, since `:cloud` tags proxy to
ollama.com on credentials the daemon holds:

```bash
ollama signin
ollama pull nemotron-3-ultra:cloud
```

The embedding model (`qwen3-embedding:latest`) is served by the Ollama daemon like the seats' models and pulled the same way — `install.sh` does it, or `ollama pull qwen3-embedding:latest`. Only its tokenizer is fetched from Hugging Face, so the chunker can cut passages in-process; nothing here runs or needs torch.

## Usage

### Basic Example

```python
from langgraph_agent import create_agent_graph, initial_state
from langgraph_agent.graph import RECURSION_LIMIT

graph = create_agent_graph()
result = graph.invoke(
    initial_state("Create a hello.txt file containing 'Hello World'"),
    {"recursion_limit": RECURSION_LIMIT},
)
print(result["plan"])
print(result["builder_report"])
```

`initial_state(goal, expect_failures=..., discuss_only=..., output_dir=...)`
takes the three per-run flags the console offers; `output_dir="projects/<name>"`
confines the Builder's writes to a generated project.

### On a cloud provider

Every seat defaults to a local model, so a key alone moves nothing. Point a seat
at a provider in `.env` -- `ARCHITECT_PROVIDER=anthropic`, optionally
`ARCHITECT_MODEL=...` -- or from its dropdown in the console.
`python scripts/cloud_smoke.py` runs one goal with every seat moved onto
Anthropic, when `ANTHROPIC_API_KEY` is set.

## Running Tests

```bash
# With the stub LLM: fast, offline, no daemon or key needed
python -m pytest tests/ -v
```

## Diagnosing the seats

`scripts/diagnose_seats.py` answers "which model actually works in which seat",
in two phases that are deliberately kept apart because one is far cheaper than
the other.

```bash
# What it knows how to run — costs nothing
python scripts/diagnose_seats.py --list
python scripts/diagnose_seats.py --dry-run

# Phase 1 only: one call per model per role
python scripts/diagnose_seats.py --phase probe

# Phase 2 only: whole runs, one sandbox each
python scripts/diagnose_seats.py --phase teams --configs baseline,heavy-gate --verbose

# Include the paid Anthropic controls
python scripts/diagnose_seats.py --anthropic --exercise all
```

**Phase 1 — role probes.** One bounded call per (model, role) pair, through the
real role prompt and the real parser the node uses. It answers whether a model
can hold a seat at all: did it answer, did the answer parse, and — for the
Builder — can it call a tool. `empty` is the status that matters, because a
seat that returns nothing loops the run rather than failing it.

The Architect is asked twice: to rule on finished work and on blocked work.
It is the seat that ends the run, so its failures do not show up in its own
output — a gate that approves regardless ends runs that produced nothing
(`rubber`), and one that never approves cycles to the step ceiling
(`cautious`). One verdict cannot be told apart from a fixed one. Both fixtures
must be fair: grade a gate against a report that does not deserve approval and
the models with the best judgment are the ones marked down.

**Phase 2 — team runs.** Each configuration runs the same short exercise through
the same graph the console drives, instrumented per node. It answers what the
probes cannot: whether four seats that each work alone make progress *together*.
Watch the `cycles` column — a run that passes the gate repeatedly while
producing nothing is the hand-off loop worth acting on.

Team runs let the Builder write files, so each one runs in its own sandbox
directory (a `chdir`), never in the project. Reports land in
`reports/diagnostics/<timestamp>/` as both `report.md` and `results.json`.

> Start one run from the console first if you care about the results: these
> exercises call the seats directly rather than through `rpc_run_goal`, so
> nothing builds the corpus for them. Against an empty corpus every Researcher
> falls back to its model, which reads exactly like a bad Researcher seat — the
> script warns when it finds one, and records the corpus size in the report.

Nothing here is a benchmark. One short exercise per configuration is a data
point against non-deterministic models, not a ranking.

## Development Tools

The same checks CI runs:

```bash
ruff check src/ tests/ serve.py scripts/ spectral_graph/ example_usage.py ollama_client.py
mypy src/langgraph_agent/ serve.py ollama_client.py spectral_graph/
python -m pytest tests/ -v
```

## Example

```bash
# Two goals on the default local seats, written into projects/example/
python example_usage.py
```

## Shared State Schema

Per the 4-Agent System specification:

| Field              | Written By  | Description                                    |
|--------------------|-------------|------------------------------------------------|
| `goal`             | User        | The user's original goal                       |
| `messages`         | All         | Conversation history / log                     |
| `architecture`     | Architect   | Direction and constraints, injected downstream  |
| `verdict`          | Architect   | `plan` \| `approved` \| `revise` \| `need_research` |
| `plan`             | Planner     | Structured plan with steps                     |
| `research`         | Researcher  | Findings from GraphRAG queries                 |
| `builder_report`   | Builder     | Implementation report                          |
| `next_agent`       | Planner     | Which agent runs next                          |
| `research_status`  | Researcher  | `ready_for_builder` \| `need_replan` \| `no_relevant_knowledge` |
| `blockers`         | Builder     | What's blocking progress                       |
| `files_changed`    | Builder     | Files a write tool reported writing            |
| `failed_verification` | Builder  | Written files that failed to run, or were never run |
| `unverified`       | Builder     | The part of `failed_verification` nobody ran   |
| `builder_cut_off`  | Builder     | `turn_cap` \| `deadline` when a pass ended early |
| `lint_failed`      | Builder     | Written Python files that still fail `ruff check` |
| `expect_failures`  | Caller      | A file that runs and fails stops blocking approval |
| `discuss_only`     | Caller      | No tools at all; the run proposes and changes nothing |
| `output_dir`       | Caller      | `projects/<name>`, the one directory the Builder may write |
| `step_count`       | Architect   | Cycles through the gate (`MAX_STEPS` is the ceiling) |

`initial_state()` builds the starting state; see `AgentState` in `state.py`.

## Tool belts

`mcp_client.py` serves every tool in-process under an MCP-style name, each
taking a dict of arguments and returning a JSON-serialisable dict:

| Tool | Agent | Purpose |
|------|-------|---------|
| `search_knowledge_graph` | Researcher | Search the knowledge graph + vector store |
| `query_knowledge_graph` | Researcher | Query entity/relationship neighborhoods |
| `filesystem_read` | Builder | Read a file |
| `filesystem_write` | Builder | Write a file |
| `git_status` | Builder | `git status --porcelain` |
| `git_diff` | Builder | `git diff` |
| `git_dwell` | Builder | The whole git flow in order: survey, branch, stage, commit, push, open a PR, merge it. Never commits onto the default branch. Runs every stage by default; name `stages` without `merge` to stop at the PR |
| `terminal_execute` | Builder | Run one program, no shell (killed after `TERMINAL_TIMEOUT_SECONDS`, default 60; pass `timeout` to raise) |
| `run_tests` | Builder | Run `pytest`, in the project root or in a `cwd` it is given (a generated project's own suite) |

The Researcher and Builder nodes call these tools through `MCPClient`, preserving
the documented specialization:

- Architect → no tools
- Planner → no tools
- Researcher → GraphRAG read-only tools only
- Builder → filesystem / git / terminal / test tools only

None of them is retried except a `git push` that never reached the remote: every
other tool has effects, and running one twice is not a recovery.

## Project Structure

```
├── src/langgraph_agent/   # the package: graph, nodes, seats, GraphRAG, tools, self_healing
├── prompts/               # one system prompt per seat
├── frontend/              # the web console
├── spectral_graph/        # spectral graph theory behind the corpus diagnostics
├── scripts/               # the seat diagnostic, the cloud smoke run, benchmarks
├── tests/                 # the suite; runs offline on the stub LLM
├── serve.py               # the console's server and its self-healing monitor
├── install.sh             # Arch / Omarchy: everything, from nothing to a running console
└── example_usage.py       # two goals through the loop
```

CLAUDE.md carries the full tree, file by file.

## Hardware Requirements

**Local by default.** The embedding model and every seat run through the
local Ollama daemon. The Builder's `qwen3.8:latest` is 17 GB against a 6 GB
card and always takes the CPU fallback (see `AGENT_LLM_OPTIONS` in
`config.py`); the three dolphin-seated roles are 5.3-5.7 GB and load 100% on
the GPU. A GPU is not required -- everything still runs on CPU, just slower --
but is what keeps the Architect/Planner/Researcher seats fast. Point any seat
at Anthropic or an Ollama Cloud tag instead if you'd rather not run
weights locally at all.

## Next Steps

1. **Human-in-the-loop** — Add approval gates before the Builder executes
2. **Persistence** — Add LangGraph checkpointing for long-running agents
