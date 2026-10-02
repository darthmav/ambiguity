---
name: console
description: Build, run, and drive the Ambiguity 4-agent console. Use when asked to start the console or web UI, run the server, take a screenshot of the console, check its self-healing state, run the tests, or interact with the running app.
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

**Live agent runs also need the local Ollama daemon**, which serves every
default seat from local weights, and the embedder:

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

`up` takes ~8s. A passing `smoke` on a fresh machine looks like this:

```
  PASS  status responds  embedding=qwen3-embedding:latest
  PASS  corpus state reported  corpus=absent
  note  the archive is empty -- upload a document or research online
  PASS  four seats configured  n=4
  note  0/4 seats live -- run_goal will fail until a tag is pulled
  PASS  self-healing reports  circuits=3
  PASS  unknown method returns error envelope, not 500

SMOKE OK
```

With something in the corpus it also checks the graph, staleness, the
document list and a search. Any RPC method directly (the full list, with
parameters, is the table in `frontend/README.md`):

```bash
.claude/skills/console/driver.py rpc rag_stats
.claude/skills/console/driver.py rpc search_documents '{"query":"planner agent","top_k":3}'
.claude/skills/console/driver.py rpc healing
.claude/skills/console/driver.py seats
```

### The corpus

The corpus is the research archive -- pages online research kept
(`research/web/`), uploads (`uploads/`) and generated projects opted in
(`projects/<name>`) -- never the checkout, so a fresh machine has none. The
console rebuilds it from the archive when it starts, again before every run,
and once more when the self-healing monitor sees the embedder come back after
interrupting a rebuild. To give it something, upload a document:

```bash
.claude/skills/console/driver.py rpc upload_document '{"name":"notes.md","content":"..."}'
```

Embedding needs `qwen3-embedding:latest` pulled (`ollama pull
qwen3-embedding:latest`, ~4.7GB); a live run also needs the seats' tags.

### Self-healing

`rpc healing` returns every circuit (`ollama-daemon`, `anthropic-api`,
`web-search:<backend>`), each service's last health check, and
the healing journal; `rpc reset_circuit '{"name":"ollama-daemon"}'` closes one
by hand. A stopped daemon shows as an open `ollama-daemon` circuit and a red
header chip; it closes by itself once the daemon answers a trial call.

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

The checks CI runs:

```bash
.venv/bin/ruff check src/ tests/ serve.py scripts/ spectral_graph/ example_usage.py ollama_client.py
.venv/bin/mypy src/langgraph_agent/ serve.py ollama_client.py spectral_graph/
.venv/bin/python -m pytest tests/ -q
```

Expect one warning on Python 3.14: chromadb calling the deprecated
`asyncio.iscoroutinefunction` -- upstream, ignore it. Tests use a stub LLM --
no seats, no daemon, no keys needed.

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
  one.** Ollama retires tags outright sometimes -- `run_goal` then fails
  immediately with `status code: 410`, and
  `ollama pull` on that tag fails too ("file does not exist"). This is
  independent of the "all four seats ship dead / NOT PULLED" gotcha below: a
  *pulled*, listed tag can still be a dead tag. Check `rpc list_seats` for
  `"live": true` before assuming a failed run is a pull/credentials problem.
- **Only the server indexes.** Nothing outside a running `serve.py` writes the
  corpus, so the graph it serves is always the one on disk.

- **Never `pkill -f serve.py`.** The pattern matches the shell running the
  pkill, so it kills its own caller — the command dies with exit 144 and the
  server survives. Use `driver.py down` (the app's own `shutdown` RPC).

- **A failed RPC method returns HTTP 200** with an `error` member, by design —
  the console renders errors into its telemetry log. A 200 does not mean the
  call worked; look at the body.

- **All four seats ship dead.** `DEFAULT_SEATS` is four local Ollama tags --
  a dolphin tag (no tool support, but none of the three need it) for the
  Architect, Planner and Researcher, `qwen3.8:latest` (the local tag that
  does report tools) for the Builder -- and the default embedder is the
  Ollama tag `qwen3-embedding:latest`. A fresh daemon has none of them
  pulled, so every seat badges `NOT PULLED` and `run_goal` fails.
  Everything read-only — Graph, Retrieval, Corpus, State, all of `smoke` —
  works fine without them. `.env` holds no API keys by default.

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
| Graph tab empty, `rag_stats` corpus=absent | The archive is empty. Upload a document, or run a goal with research online. |
| Header shows `ollama-daemon down` | The daemon stopped answering. Start it; the circuit closes on the next trial call, or `rpc reset_circuit` now. |
| Shell command exits 144, server still running | `pkill -f serve.py` matched its own caller. Use `driver.py down`. |
| `{"error": {"message": "bad JSON"}}` | Shell quoting mangled the payload, not a server fault. Use `driver.py rpc`. |
| Seats badge `NOT PULLED`, runs fail | Expected on a fresh box. Pull the tag, or set a seat to a provider you have via `rpc set_seat`. |
| `run failed: ... status code: 410` | An Ollama Cloud tag the seat points at was retired upstream. `rpc list_seats` for a `"live": true` tag, `rpc set_seat` onto it. |
| `run failed: ... status code: 402 -- not included in your free usage` | Account has no credits for that tag. Try another from `rpc llm_options`. |
| `driver.py rpc search_documents` errors `model "qwen3-embedding:latest" not found` | Embedder isn't pulled. `ollama pull qwen3-embedding:latest` (~4.7GB). |
| A merged code change doesn't show up in `rag_stats`/behavior | `:8080` may be a `docker compose` container running stale in-memory code, not a process `driver.py` manages. `docker ps` for `ambiguity-console-1`; if present, `docker restart ambiguity-console-1`, not `driver.py restart`. |
| `driver.py shot` says "no chromium on PATH" | Install chromium, or screenshot from a browser against http://localhost:8080. |
