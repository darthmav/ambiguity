---
name: console
description: Build, run, and drive the Ambiguity 4-agent console. Use when asked to start the console or web UI, run the server, reindex the knowledge graph, take a screenshot of the console, run the tests, or interact with the running app.
---

# Ambiguity 4-agent console

A LangGraph + GraphRAG console: a `ThreadingHTTPServer` on :8080 that serves
the SPA in `frontend/` and answers `POST /rpc` with `{method, params}`. Drive
it with **`.claude/skills/console/driver.py`** — it speaks the same RPC surface
the SPA does, and screenshots the UI with headless chromium (no Playwright, no
xvfb, no browser extension).

All paths below are relative to the project root.

## Prerequisites

Python ≥ 3.12 (CI runs 3.12 and 3.14) and `chromium` on PATH for screenshots:

```bash
python3 --version
which chromium
```

**Live agent runs also need the local Ollama daemon**, because every default
seat is an Ollama *Cloud* tag that the local daemon proxies:

```bash
systemctl is-enabled ollama.service   # enabled
systemctl is-active  ollama.service   # active
```

## Build

There is no venv in a fresh checkout. On Arch / Omarchy `./install.sh` builds
it. Nothing in the project touches torch or a card itself -- the embedding
model is `qwen3-embedding:latest`, served by the local Ollama daemon, which
owns its GPU placement. The only Hugging Face download is that model's
tokenizer, which the in-process chunker cuts passages with. By hand:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Verify:

```bash
.claude/skills/console/driver.py doctor
```

```
root        <checkout>
interpreter <checkout>/.venv/bin/python
venv        present
import      langgraph_agent OK
chromium    /usr/bin/chromium
server      up
```

## Run (agent path)

```bash
.claude/skills/console/driver.py up                 # start, wait for readiness
.claude/skills/console/driver.py smoke              # end-to-end check, non-zero on failure
.claude/skills/console/driver.py shot /tmp/c.png    # headless screenshot
.claude/skills/console/driver.py down               # clean stop
```

`up` takes ~8s. A passing `smoke` looks like this:

```
  PASS  status responds  embedding=qwen3-embedding:latest
  PASS  corpus indexed  corpus=indexed
  PASS  graph has nodes  nodes=358 edges=449
  PASS  graph not stale
  PASS  documents listed  n=20
  PASS  semantic search returns hits  top=README.md
  PASS  four seats configured  n=4
  note  0/4 seats live -- run_goal will fail until a tag is pulled
  PASS  unknown method returns error envelope, not 500

SMOKE OK
```

(`n=20`, not the 75 an older run of this skill saw: `PROJECT_INDEX_EXCLUDES`
now keeps `src/`, `tests/`, `prompts/`, `frontend/` and `spectral_graph/` out
of the corpus, so only project-level docs/config get indexed. `top=README.md`
follows from the same change -- there is no `src/langgraph_agent/graph.py` in
the corpus to match anymore.)

Any of the 20 RPC methods directly:

```bash
.claude/skills/console/driver.py rpc rag_stats
.claude/skills/console/driver.py rpc search_documents '{"query":"planner agent","k":3}'
.claude/skills/console/driver.py seats
```

Methods: `rag_stats bottleneck topics duplicate_entities list_documents
query_graph search_documents reindex upload_document export_corpus clear_corpus
list_seats set_seat llm_options status run_goal run_progress stop_run last_run
shutdown`.

### Building the knowledge graph

The corpus is absent in a fresh checkout — the Graph tab draws "No graph yet"
and `rag_stats` reports zeros. **Nothing builds it but a run**, which rebuilds
it before the Architect opens; there is no script, no install step and no
button. Any run does it, so this is only a shortcut:

```bash
.claude/skills/console/driver.py reindex
```

It starts the server if it is down and drives one discussion-only goal. Takes a
while the first time (the local Ollama daemon loads `qwen3-embedding:latest`
and may pull it first); after that a rebuild that changes nothing costs ~0.1s,
because a
document whose text has not changed keeps the vectors it has. Expect:

```
[Corpus] The corpus on this machine was empty, so the project was indexed
         before the run: 20 document(s), 351 passage(s) in 295.3s.
corpus: indexed -- 20 documents, 351 chunks, 358 nodes
```

**This needs two things live before it works, and a fresh daemon has
neither:**

1. **The embedder must be pulled.** `qwen3-embedding:latest` is not on a
   fresh Ollama daemon despite being *the* embedding model every index and
   search goes through. `ollama pull qwen3-embedding:latest` first (~4.7GB;
   `ollama list | grep embed` to check).
2. **The Architect's seat must actually be live.** `reindex` drives a real
   `run_goal`, so the Architect has to answer. The shipped default is a
   local dolphin tag (see DEFAULT_SEATS in `config.py`) -- if it is not
   pulled, `run_goal` fails before any indexing happens. Check
   `driver.py rpc list_seats` for a `"live": true` entry and
   `driver.py rpc set_seat '{"role":"architect","provider":"ollama","model":"<tag>"}'`
   onto whatever is. A seat pointed at an Ollama Cloud tag instead can also
   fail with `status code: 410` (the tag was retired upstream) or
   `status code: 402` (the account has no credits for it) -- both are
   provider-side, not this project's; `driver.py rpc llm_options` lists
   what else is pulled locally to try.

## Run (human path)

```bash
./launch_console.sh
```

Prints seat status, starts the server, opens a browser at
http://localhost:8080, tails the log, stops on Ctrl+C. It launches
`python serve.py` off PATH, so **it only works if the venv is active** —
otherwise it dies with `ModuleNotFoundError: No module named 'langgraph'`.

Five tabs: Engineer, Graph (default), Retrieval, Corpus, State.

## Test

```bash
.venv/bin/ruff check src/ tests/ serve.py scripts/ example_usage.py test_cloud.py
.venv/bin/mypy src/langgraph_agent/ serve.py
.venv/bin/python -m pytest tests/ -q
```

Observed: ruff clean, `mypy` — no issues in 21 source files, pytest
**789 passed, 1 warning in ~36s** (the warning is chromadb calling the
deprecated `asyncio.iscoroutinefunction` on 3.14; upstream, ignore it).

Tests use a stub LLM — no seats, no daemon, no keys needed.

## Gotchas

- **A container can be "up" and healthy while running stale code.** If
  `:8080` is being served by the project's own `docker compose` setup
  (`docker ps` shows `ambiguity-console-1`), the checkout is bind-mounted so
  the *files* on disk are current, but the *process* only re-imports them on
  its own restart -- an hours-old container keeps running whatever
  `graphrag_server.py` looked like when it last started. `driver.py doctor`'s
  `server up` cannot tell the difference; it only checks that something
  answers `/api/status`. Confirmed this session: `rag_stats` kept reporting
  pre-fix behavior until `docker restart ambiguity-console-1` (not
  `driver.py restart` -- see next bullet) picked up a merged code change.
- **`driver.py up`/`down`/`restart` assume they own the process**, via a
  `subprocess.Popen` + HTTP liveness poll (`is_up()`). Against a container:
  `up` sees the container already answering and no-ops (`"already up at
  ..."`) rather than starting anything of its own; `down`/`restart` call the
  app's own `shutdown` RPC, which the container's `serve.py` answers the same
  as a bare process -- so it *will* stop the container's main process (and
  the container with it), not just "your copy" of the server. When the
  console is containerized, restart it with `docker restart
  ambiguity-console-1`, not `driver.py restart`.
- **An Ollama Cloud tag can be dead on arrival even if you point a seat at
  one.** Ollama retires tags outright sometimes -- `run_goal`/`reindex` then
  fail immediately with `status code: 410`, before any indexing happens, and
  `ollama pull` on that tag fails too ("file does not exist"). This is
  independent of the "all four seats ship dead / NOT PULLED" gotcha below: a
  *pulled*, listed tag can still be a dead tag. Check `rpc list_seats` for
  `"live": true` before assuming a failed run is a pull/credentials problem.
- **Indexing from outside the server used to leave it blind, and that is why
  nothing does it any more.** A script wrote the graph to disk while a live
  `serve.py` kept the NetworkX graph it had built at startup and never re-read
  it. The symptom was a split: `search_documents` returned real hits (chunks
  come from chroma) while `rag_stats` reported `total_nodes: 0, total_edges: 0,
  total_documents: 0` and `staleness.stale: true`, and the Graph tab stayed
  empty *forever*. The rebuild now happens inside the running server, on the
  object the console reads, so the split cannot occur — and a restart is no
  longer part of building a corpus.

- **Never `pkill -f serve.py`.** The pattern matches the shell running the
  pkill, so it kills its own caller — the command dies with exit 144 and the
  server survives. Same trap with `pgrep -f reindex.py`, which reports a
  finished reindex as still RUNNING. Use `driver.py down` (the app's own
  `shutdown` RPC) and read the log for completion.

- **A failed RPC method returns HTTP 200** with an `error` member, by design —
  the console renders errors into its telemetry log. A 200 does not mean the
  call worked; look at the body.

- **All four seats ship dead.** `DEFAULT_SEATS` is four local Ollama tags --
  a dolphin tag (no tool support, but none of the three need it) for the
  Architect, Planner and Researcher, `qwen3.8:latest` (the local tag that
  does report tools) for the Builder -- and the default embedder is the
  Ollama tag `qwen3-embedding:latest`. A fresh daemon has none of them
  pulled, so every seat badges `NOT PULLED` and `run_goal` fails.
  Everything read-only — Graph,
  Retrieval, Corpus, State, all of `smoke` — works fine without them. (The
  older `console.md` claimed the default backend was Anthropic and told you to
  set `ANTHROPIC_API_KEY`; both were wrong. `.env` holds no API keys at all,
  only `BUILDER_DEADLINE_SECONDS`, `VERIFY_RESERVE_SECONDS`,
  `MAX_BUILDER_TOOL_TURNS`, `RUN_BUDGET_SECONDS`.)

- **The default tab is Graph, not Engineer** — `aria-selected` is on
  `button.tab[data-p="graph"]`. Screenshots land on the graph.

- **Screenshot byte size is the fastest liveness check.** The empty console is
  ~70KB; a drawn graph is ~900KB. `driver.py shot` warns below 120KB.

- **The graph is a force layout that needs time to settle.** `driver.py shot`
  passes `--virtual-time-budget=15000`; a smaller budget captures a tangle
  mid-simulation.

- Sending JSON to `/rpc` by hand from bash is easy to mangle — a bad quote gets
  you `{"error": {"message": "bad JSON"}}` from the server, which looks like a
  server problem and is not. Use `driver.py rpc`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langgraph'` | venv missing or not used. Re-run Build; `driver.py` picks up `.venv` automatically. |
| Graph tab empty, `rag_stats` nodes=0, but search returns hits | Reindexed under a live server. `driver.py restart` (no reindex needed). |
| Shell command exits 144, server still running | `pkill -f serve.py` matched its own caller. Use `driver.py down`. |
| `{"error": {"message": "bad JSON"}}` | Shell quoting mangled the payload, not a server fault. Use `driver.py rpc`. |
| Seats badge `NOT PULLED`, runs fail | Expected on a fresh box. Pull the tag, or set a seat to a provider you have via `rpc set_seat`. |
| `run failed: ... status code: 410` | An Ollama Cloud tag the seat points at was retired upstream. `rpc list_seats` for a `"live": true` tag, `rpc set_seat` onto it. |
| `run failed: ... status code: 402 -- not included in your free usage` | Account has no credits for that tag. Try another from `rpc llm_options`. |
| `driver.py rpc search_documents` errors `model "qwen3-embedding:latest" not found` | Embedder isn't pulled. `ollama pull qwen3-embedding:latest` (~4.7GB). |
| A merged code change doesn't show up in `rag_stats`/behavior | `:8080` may be a `docker compose` container running stale in-memory code, not a process `driver.py` manages. `docker ps` for `ambiguity-console-1`; if present, `docker restart ambiguity-console-1`, not `driver.py restart`. |
| `driver.py shot` says "no chromium on PATH" | Install chromium, or screenshot from a browser against http://localhost:8080. |
