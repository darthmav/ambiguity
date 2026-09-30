---
name: extend-agent
description: How to add a new Builder tool, add a new Researcher tool, change which model holds a seat, or add a fifth agent to the langgraph-agent crew. Use when asked to add/extend a tool, reassign a seat's model, or add a new agent role.
---

# Extending the agent crew

### Add a new Builder tool
1. Add the method to `_discover_tools()` in `src/langgraph_agent/mcp_client.py`,
   named under `filesystem_`, `git_`, `terminal_` or `test_`.
2. Add its JSON schema to `BUILDER_TOOLS` in `src/langgraph_agent/nodes.py` --
   a tool the MCP client exposes and `BUILDER_TOOLS` omits is refused by name.
3. Add a test, and update the README's MCP Integration section.

### Add a new Researcher tool
Read-only GraphRAG tools only: extend `src/langgraph_agent/graphrag_server.py`
and expose it through `mcp_client.py`.

### Change a seat's model
Update `DEFAULT_SEATS` and `_DEFAULT_AGENT_MODELS` in
`src/langgraph_agent/config.py`, the seat table in CLAUDE.md, and `.env.example`; add
the tag to `AGENT_LLM_OPTIONS`, which is the whole offer -- `set_seat` refuses
anything not in it and `test_rpc_params.py` pins the list. Check the tag
reports `tools` before seating it as the Builder: that seat's work *is* tool
calls, and `get_agent_status` puts a **NO TOOLS** chip on its card alone.

### Add a fifth agent
`AGENTS` in `config.py` is the seat list everything iterates. Adding one means:
a node in `nodes.py`, a prompt in `prompts/`, wiring plus a router in
`graph.py`, entries in `AGENTS` / `DEFAULT_SEATS` / `_DEFAULT_AGENT_MODELS`, a
`StubLLM` branch, and a `ROLE_COLOR` entry in `frontend/index.html`.
