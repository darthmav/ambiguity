# 4-Agent Console — Frontend

The web interface for the 4-Agent AI System, modelled on the original Ambiguity
console: five tabs, a crew rail, and a force-directed view of the knowledge
graph.

## Quick Start

```bash
python serve.py
# Open: http://localhost:8080
```

Opening `frontend/index.html` directly will not work: every panel is fed by the
API, so the page needs the server behind it.

## Layout

```
┌──────────────┬──────────────────────────────────────────────────────────┐
│ ● AMBIGUITY  │ ENGINEER GRAPH RETRIEVAL CORPUS STATE   docs · chunks ·  │
│   CONSOLE    │                                         nodes · edges    │
├──────────────┴──────────────────────────────────────────────────────────┤
│ ⚠ degraded-seat banner (hidden when every seat can run)                  │
├──────────────┬──────────────────────────────────────────────────────────┤
│ CREW    4    │                                                          │
│ ┌──────────┐ │                                                          │
│ │● ARCHITECT│ │                 active panel                            │
│ │[opus-5  ▾]│ │                                                          │
│ │anthropic ☑│ │                                                          │
│ │ NO KEY    │ │                                                          │
│ └──────────┘ │                                                          │
│ …4 cards…    │                                                          │
│ Embedding    │                                                          │
└──────────────┴──────────────────────────────────────────────────────────┘
```

The brand dot pulses green while the backend answers and turns red when it stops.

## Tabs

**Engineer** — give the Architect a goal. There is no per-node event stream, so
the feed says the loop is running with an elapsed timer, then renders one stage
card per agent from the run's own message log. It shows the path the run
actually took, not a fixed sequence.

*Stop* halts the run at its next safe boundary. It is cooperative — the file
write or model call already in flight finishes, and nothing further starts — so
it lands within a tool call rather than instantly, and never leaves a file half
written. The run then renders as `STOPPED` rather than as a verdict, with what
was written, what nobody ran, and what blocked. The server keeps that snapshot,
so reloading the page gets it back; reloading *during* a run reattaches to it
instead, Stop included.

*expect failures* is unrelated to Stop and stays what it was: it excuses a file
the run meant to fail, not one nobody executed.

*Attach* puts a document of your own into the corpus the Researcher searches,
and so does dropping files onto the transcript. The answer comes back in the
transcript — where each file went, how many passages it became, and anything
the corpus refused with the reason. The Corpus tab's *Upload documents* is the
same thing from the other end; both call `upload_document`.

The *×* in the top right ends the session: it shuts down `serve.py` itself, not
just the page. With a run going it asks first, then stops the run and lets it
save its state before the server exits.

**Graph** — the knowledge graph. *Sweep all* draws the whole corpus from one
`graph_overview` call; *Trace* centres on one node, and the depth spinner
controls how far a trace walks. Documents are green, entities blue. Entities
are labelled on hover; documents are labelled while there are few enough to
read — all of them on a small trace, the best-connected on a sweep — and every
one once you zoom in. Scroll to zoom about the pointer, drag the background to
pan, and double-click it or press *Reset view* to reset. Hovering a node keeps
it and its neighbours and steps everything else back. Nodes are draggable.

The sweep keeps only entities shared by four or more documents — below that,
per-document noise buries the structure — and leaves out documents that share no
entity with the rest (fetched pages and config files among them), saying how
many. A trace keeps everything, since a node asked for by name should not have
neighbours hidden.

**Retrieval** — semantic search with score bars, plus a rolling telemetry log of
every non-quiet RPC (time, method, milliseconds; red on error).

**Corpus** — the indexed documents, and buttons to upload, reindex, export or
clear them. Clicking a document jumps to the Graph tab and traces it.

*Upload documents* writes each file under `uploads/` and indexes it from there.
That is the point rather than an implementation detail: a reindex rebuilds the
corpus from the files on disk and prunes everything else, so a document
embedded only into the index would disappear at the next rebuild with nothing
reporting it. Text only — the types a reindex reads, listed on hover; a PDF is
refused rather
than embedded as whatever its bytes decode to. Re-uploading a name replaces
that document rather than adding a second copy of it.

**State** — the raw `AgentState` from the last run.

## Crew rail

One card per seat: role-coloured dot and border, a dropdown of the seat models
the console offers (`qwen3.8:latest`, two local dolphin tags, `kimi-k3:cloud`;
other tags the daemon carries are left out, and `set_seat` refuses them),
the provider with a
*thinking* checkbox beside it, and chips for placement (`REMOTE` / `LOCAL`)
and `NO KEY`.

The checkbox shows what the seat's next call will do, because a switchable
model is always sent the flag — off included. It is grayed out when the model
offers no switch, with the reason on hover: the model cannot think, always
thinks, or nobody could say (a daemon that did not answer, or an OpenAI model
whose reasoning is set by effort rather than on and off). The last case shows
neither ticked nor clear. Support is asked of the Ollama daemon per tag, so a
newly pulled model needs no code change.

The status chip is the important one, and it distinguishes two different
failures rather than blaming them both on a missing key:

| Chip | Meaning | What a run does |
|---|---|---|
| `NO KEY` | no credentials at all | seat becomes a stub; the run completes with canned text |
| `FAILING` | last call failed (no credits, key rejected, rate limited) | the run fails outright |
| `OFFLINE` | Ollama daemon unreachable | the run fails outright |
| `NOT PULLED` | the tag is not on the daemon | the run fails outright |

`FAILING` is recorded from the actual outcome of the last call, not from a probe
— a key can be present and correct and the seat still unusable. Hover the chip
for the provider's own message. A seat clears itself the next time a call
succeeds. Reassigning a seat takes effect immediately and lasts for the life of
the server process.

## API

The console drives a single endpoint:

```bash
curl -s localhost:8080/rpc -X POST -H 'content-type: application/json' \
  -d '{"method":"rag_stats","params":{}}'
```

Responses are `{"result": …, "elapsed_ms": N}` or
`{"error": {"message": …}, "elapsed_ms": N}` — a failed method is not a failed
request, so both come back 200.

| Method | Params | Returns |
|---|---|---|
| `status` | — | seats, embedding model, corpus state, degraded seats |
| `rag_stats` | — | documents, chunks, nodes, edges |
| `list_documents` | — | every document node |
| `query_graph` | `node_id`, `max_depth`, `min_degree` | `center_node`, `related_nodes`, `edges` |
| `search_documents` | `query`, `top_k` | ranked results |
| `reindex` | — | indexed / skipped / errors plus fresh stats |
| `upload_document` | `name`, `content` | where it was stored, its passage count, fresh stats |
| `list_seats` | — | the four seats and whether each can run |
| `set_seat` | `agent`, `provider`, `model` | the updated seat |
| `set_thinking` | `agent`, `thinking` (a JSON boolean) | the updated seat; refused for a model with no switch |
| `llm_options` | — | the seat models the console offers (`AGENT_LLM_OPTIONS`), nothing else |
| `run_goal` | `goal` | the final `AgentState` |

`/api/status`, `/api/llm-options`, `/api/run`, `/api/search` and `/api/set-llm`
remain as compatibility wrappers over the same functions; `launch_console.sh`
polls `/api/status` as its readiness check.

## Customization

Colors are CSS variables at the top of `index.html`:

```css
:root {
  --architect: #f2b544;   /* seat colors */
  --planner:   #4ea3ff;
  --researcher:#76d13a;
  --builder:   #ff6a8a;

  --bg:   #0a0b0d;        /* layered surfaces */
  --bg-2: #101216;
  --bg-3: #15181e;
}
```

Port: `PORT=3000 python serve.py`.

## Troubleshooting

**Graph tab is empty** — nothing has been indexed here yet. Start a run: the
corpus is built before the Architect opens. Check `rag_stats` reports non-zero
nodes afterwards.

**A seat shows NO KEY** — it is pointed at Anthropic or OpenAI and that
provider's key is unset. The default seats are all Ollama; if those are the
ones failing, the daemon is down or not signed in (`ollama signin`).

**Server won't start** — port in use; `PORT=3001 python serve.py`.

## Architecture

```
frontend/
  index.html          # Single-page app (HTML + CSS + JS), zero build
serve.py              # Python HTTP server + /rpc backend
```

No npm, no bundler, no framework, no external requests — one file of plain
HTML, CSS and vanilla JS, with the layout, spring layout and theme carried over
from the original Ambiguity console.
