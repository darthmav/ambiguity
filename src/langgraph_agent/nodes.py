"""Agent nodes: Architect, Planner, Researcher, Builder.

Implements the 4-Agent System with strict prompts and tool binding:
- Architect: reasoning only, no tools; sets direction and holds the approval gate
- Planner: reasoning only, no tools
- Researcher: GraphRAG MCP tools only
- Builder: filesystem, git, terminal tools only
"""

import asyncio
import json
import keyword
import os
import re
import shlex
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph_agent.config import get_agent_llm
from langgraph_agent.control import RUN_CONTROL
from langgraph_agent.mcp_client import (
    TERMINAL_TIMEOUT_MAX_SECONDS,
    TERMINAL_TIMEOUT_SECONDS,
    mcp_client,
)
from langgraph_agent.state import AgentState, ResearchStatus, Verdict


def _call_mcp_tool_sync(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool from a synchronous LangGraph node.

    The MCP client is async, so this helper bridges sync and async contexts.

    The branch is decided by asking whether a loop is running, never by
    catching `RuntimeError` from `asyncio.run`. That is what this used to do,
    and a tool raising `RuntimeError` of its own -- the embedder's "Ollama
    could not embed ...", say -- landed in the fallback, whose
    `get_event_loop()` has no loop to return on a worker thread (or on any
    thread, from Python 3.14) and raised "There is no current event loop"
    in its place. The caller was told about a loop nobody asked for, and the
    tool's own error was gone. A loop that *is* running on this thread cannot
    be re-entered either, so that case runs the call on a thread of its own.
    """

    async def _call() -> Any:
        async with mcp_client() as client:
            return await client.call_tool(tool_name, arguments)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call())

    box: list[Any] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            box.append(asyncio.run(_call()))
        except BaseException as exc:  # re-raised on the caller's thread below
            error.append(exc)

    worker = threading.Thread(target=_run, name="mcp-tool-call")
    worker.start()
    worker.join()
    if error:
        raise error[0]
    return box[0]


_T = TypeVar("_T")

# How long one node may spend gathering its answer. This is the second half of
# the timeout story and is not made redundant by `LLM_TIMEOUT_SECONDS`: that one
# bounds the socket, so it catches a connection that goes quiet but not a model
# that streams tokens slowly and indefinitely, and not a node that makes several
# calls each of which finishes just inside its own limit.
NODE_DEADLINE_SECONDS = float(os.getenv("NODE_DEADLINE_SECONDS", "150"))

# How many retrieved passages reach the Builder, and how much of each.
#
# Both were quietly throwing away work that had already been paid for. The
# search asked for five results and the formatter forwarded three, so two
# passages were retrieved, ranked, re-ranked and dropped unread -- and they are
# the *diverse* two, since `SEARCH_ESCALATION` widens the window precisely to
# stop one file monopolising the hits. One number now feeds both the request
# and the slice, so they cannot drift apart again.
#
# The 300-character cut was worse, because of what it kept. A chunk is the unit
# retrieval judges: it is selected *because* it matched, and the matching
# sentence can sit anywhere inside it. Measured on this corpus, chunks run to a
# mean of 882 characters and a p99 of 1,373, so 300 kept **34%** of a typical
# passage -- always the opening 34%, never the part that matched. On a
# document from `research/web` the opening is the provenance header, and the
# effect was total: the top-scoring source in the run of 2026-09-09 reached the
# Builder as its title, its URL, its retrieval timestamp and the goal it was
# fetched for, truncated mid-word, with not one character of the article
# attached. The Builder was handed a citation and no evidence.
#
# 1,500 clears the p99, so in practice a passage arrives whole and the cap is a
# guard against a pathological chunk rather than a routine trim. Five whole
# passages is roughly 4,400 characters against the 900 that used to arrive.
RESEARCH_RESULTS = int(os.getenv("RESEARCH_RESULTS", "5"))
RESEARCH_SNIPPET_CHARS = int(os.getenv("RESEARCH_SNIPPET_CHARS", "1500"))


class _Deadline:
    """A monotonic countdown shared across the several calls one node makes.

    The Researcher runs a single retrieval and can be bounded by wrapping it.
    A node that makes many calls in sequence -- the Builder's turn loop, then
    its verification pass -- needs the budget to travel with it, or each call
    gets the full allowance and the node as a whole is bounded by nothing.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def _with_deadline(work: Callable[[], _T], seconds: float, fallback: _T) -> _T:
    """Run `work`, giving up on it after `seconds` and returning `fallback`.

    The abandoned call cannot actually be cancelled -- Python cannot interrupt a
    thread blocked on a socket -- so the worker is left to unwind on its own
    when the client timeout fires. Two consequences shape this code.

    First, `work` must not write to state: a late finisher would otherwise land
    its result in a state the graph had already moved past. Callers pass a
    function that only reads and apply what it returns themselves.

    Second, the worker is a bare daemon thread rather than a
    `ThreadPoolExecutor`. Pool threads are non-daemon and the module's atexit
    hook joins them, so one abandoned worker would hold up interpreter shutdown
    for as long as it stayed blocked -- turning a bounded node into an
    unkillable server.
    """
    box: list[Any] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            box.append(work())
        except BaseException as exc:  # re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=_run, daemon=True, name="node-deadline")
    thread.start()
    thread.join(timeout=seconds)

    if thread.is_alive():
        return fallback
    if error:
        # A seat that failed outright is not a timeout; let it raise so
        # `_SeatLLM` records the reason and the console can show it.
        raise error[0]
    return cast("_T", box[0])


def _load_prompt(name: str, fallback: str) -> str:
    """Load a system prompt from the prompts/ directory.

    Falls back to the inline string so the module works even if the prompt
    files are not present (e.g., during distribution).
    """
    try:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / f"{name}.txt"
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return fallback


# System prompts from the 4-Agent System documentation.
# Loaded from prompts/ when available, with inline fallbacks.
_ARCHITECT_PROMPT_INLINE = """You are the Architect. You are the leading authority
on this project: you set the architectural direction before any work starts, and
you decide when work is finished.

You run twice per cycle. Before the Planner sees the goal, you decide how the
work should be shaped and rule `plan`. After the Builder reports, you judge the
result against the goal and your own constraints and rule `approved` (the goal
is met -- this ends the run), `revise` (the approach needs replanning), or
`need_research` (blocked on knowledge nobody has gathered).

You must not write code, call tools, or gather knowledge yourself.

There is no waiting: every verdict routes onward immediately, and the run never
pauses for approval. Never write a constraint that holds work back until
something is approved -- nothing arrives to release it. Write constraints the
Builder can satisfy in one pass; if the work needs stages, constrain this pass
to the first and rule `revise` when it comes back. When you rule `revise` on
work that looks like the last cycle's, change the constraints.

Output exactly this format and nothing else:

## Architecture
<how this work should be shaped>

## Constraints
- <a constraint the Planner and Builder must respect>

## Verdict
plan | approved | revise | need_research"""


_PLANNER_PROMPT_INLINE = """You are the Planner agent in a four-agent software system.
The other agents are Architect, Researcher and Builder. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

Your only job:
1. Understand the user's goal.
2. Break it into clear, ordered steps.
3. Choose the next agent: Researcher or Builder.
4. Write the plan in the exact format below.

You must:
- Be concise and specific.
- Name concrete artifacts (files, endpoints, behaviors) when the user named them.
- Choose Researcher when the work depends on existing code, docs, architecture, or relationships across files.
- Choose Builder when the task is fully specified, research is already in state, or retrieval would not help.
- If this is a loop (you can see prior research or a builder report), update the plan; do not ignore what already happened.
- If blockers are present, address them in the new plan.

You must not:
- Write or propose code.
- Call tools.
- Retrieve knowledge yourself.
- Invent files or APIs that were not in the user goal or in prior state.
- Talk to the user in a conversational way. Output only the plan format.

Output exactly this format and nothing else:

## Goal
[one sentence]

## Steps
1. ...
2. ...
3. ...

## Next Agent
Researcher
OR
Builder

## Notes
[assumptions, risks, why this next agent, what to skip]
"""

_RESEARCHER_PROMPT_INLINE = """You are the Researcher agent in a four-agent software system.
The other agents are Architect, Planner and Builder. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

Your only job:
Gather high-quality, relevant knowledge so the Builder can implement the plan.
You have no tools. The knowledge base (GraphRAG) has already been searched for
the plan before you are asked, and what that search returned is given to you
below the plan, with the reason none of it was accepted automatically. You are
asked only when retrieval did not clearly answer the plan, so judging those
passages is the work.

You must:
- Read each retrieved passage and decide whether it bears on the plan's steps.
- Summarize what the relevant passages say. Cite the path of every passage you rely on.
- Give the Builder concrete recommendations: what to change, where, what to reuse, what not to break.
- If the passages have little or nothing relevant, say so explicitly and answer no_relevant_knowledge. Do not invent a codebase.
- If you need a different question or a replan, say so in Recommendations.

You must not:
- Write or modify code.
- Claim to have searched, read or retrieved anything beyond the passages you were given.
- Make the final implementation decision as if you were the Builder.

Output exactly this format and nothing else:

## Key Findings
- ...

## Relevant Context
[summary of the passages that bear on the plan, with their paths]

## Recommendations for Builder
[what to implement, which files/entities, constraints]

## Status
ready_for_builder
OR
need_replan
OR
no_relevant_knowledge
"""

_BUILDER_PROMPT_INLINE = """You are the Builder agent in a four-agent software system.
The other agents are Architect, Planner and Researcher. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

Your only job:
Implement the plan using the research provided. Use tools to make real changes.

You must:
- Follow the plan and the research. Do not freelance a new design if research already specified one.
- Use filesystem / git / test tools to actually change files. Do not only describe code.
- Write clean, working code consistent with the existing project.
- After changes, report what you did.
- If you cannot finish, set blockers clearly so the graph can loop to Researcher or Planner.
- Prefer the smallest change that satisfies the plan.

You must not:
- Call GraphRAG or knowledge-search tools.
- Ignore the plan or the research.
- Claim you changed a file if you did not call a write tool.
- Expand scope beyond the plan.

Output exactly this format after you finish using tools:

## Changes Made
- ...

## Files Modified
- path/to/file
- ...

## Next Steps / Blockers
[none | what is blocked and what information is needed]
"""

ARCHITECT_PROMPT = _load_prompt("architect", _ARCHITECT_PROMPT_INLINE)
PLANNER_PROMPT = _load_prompt("planner", _PLANNER_PROMPT_INLINE)
RESEARCHER_PROMPT = _load_prompt("researcher", _RESEARCHER_PROMPT_INLINE)
BUILDER_PROMPT = _load_prompt("builder", _BUILDER_PROMPT_INLINE)


def _get_state_injection(state: AgentState) -> str:
    """Create the state injection block sent every turn.

    As specified in the documentation:
    Empty fields should explicitly say `(empty)`.
    """

    def _fmt(key: str, default: str = "(empty)") -> str:
        value = state.get(key)
        if value is None or value == "" or value == []:
            return default
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    # Every seat has to know, and the Architect above all. It is the gate: it
    # rules on whether the work is done, and on a discussion run there is no
    # work to find -- `files_changed` is empty by construction and the Builder
    # reports what it *would* do. Without this line the gate judged a proposal
    # by build standards and could never be satisfied, so the run went round
    # Researcher -> Builder -> Architect until the step ceiling: measured at 8
    # cycles, every one of them reporting "proposal ready, nothing was
    # changed". The mode is in the shared injection rather than the Architect's
    # prompt alone so the Planner stops planning edits and the Builder is told
    # twice, which costs one line.
    mode = (
        "\nMode: DISCUSSION ONLY -- no tools, nothing on this machine will be "
        "changed. The product of this run is the proposal itself: rule on "
        "whether it answers the goal, not on whether files exist."
        if state.get("discuss_only")
        else ""
    )
    # Told to every seat rather than the Builder alone, so the Planner names
    # paths the Builder is allowed to write.
    output_dir = str(state.get("output_dir") or "")
    if output_dir and not state.get("discuss_only"):
        mode += (
            f"\nOutput directory: {output_dir}/ -- every file this run writes "
            "goes under it. It is a separate project, not part of this checkout."
        )

    return f"""## Current state{mode}
Goal: {_fmt("goal")}
Architecture: {_fmt("architecture")}
Verdict: {_fmt("verdict")}
Plan: {_fmt("plan")}
Research: {_fmt("research")}
Builder report: {_fmt("builder_report")}
Blockers: {_fmt("blockers")}
Files changed: {_fmt("files_changed")}
Failed verification: {_fmt("failed_verification")}
Never executed: {_fmt("unverified")}
Builder cut off: {_fmt("builder_cut_off")}
Lint failures: {_fmt("lint_failed")}
Step: {state.get("step_count", 0)}
"""


# What may stand between a one-word section's heading and its word. The value
# is read, not merely found: a model writes it on the heading's own line
# (`## Verdict: revise`) as readily as below it, and decorates a lone word the
# way it decorates any emphasis -- `**revise**`, `` `revise` ``, `- revise`.
# Each of those read as no answer at all, and no answer has a fallback: the
# gate's is `approved`, so an Architect ruling `**revise**` on the Builder's
# report ended the run as approved, and a Researcher's `**need_replan**` read
# as ready_for_builder.
_SECTION_VALUE_LEAD = r"[ \t]*:?\s*(?:[-*+>][ \t]+)?[*_`\"']*"


def _parse_planner_output(content: str) -> dict[str, Any]:
    """Parse Planner output into structured format.

    Expected format:
    ## Goal
    ...

    ## Steps
    1. ...

    ## Next Agent
    Researcher | Builder

    ## Notes
    ...
    """
    result = {"plan": "", "next_agent": "Builder", "notes": ""}

    # Extract goal
    goal_match = re.search(r"## Goal\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if goal_match:
        result["goal"] = goal_match.group(1).strip()

    # Extract steps
    steps_match = re.search(r"## Steps\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if steps_match:
        result["plan"] = steps_match.group(1).strip()

    # Extract next agent. Capitalised on the way in, so state holds the one
    # spelling AgentState documents whatever case the seat wrote it in.
    agent_match = re.search(
        rf"## Next Agent{_SECTION_VALUE_LEAD}(Researcher|Builder)\b", content, re.IGNORECASE
    )
    if agent_match:
        result["next_agent"] = agent_match.group(1).strip().capitalize()

    # Extract notes
    notes_match = re.search(r"## Notes\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if notes_match:
        result["notes"] = notes_match.group(1).strip()

    return result


def _parse_researcher_output(content: str) -> dict[str, Any]:
    """Parse Researcher output into structured format.

    Expected format:
    ## Key Findings
    ...

    ## Relevant Context
    ...

    ## Recommendations for Builder
    ...

    ## Status
    ready_for_builder | need_replan | no_relevant_knowledge
    """
    result = {
        "key_findings": "",
        "relevant_context": "",
        "recommendations": "",
        "status": ResearchStatus.READY_FOR_BUILDER.value,
    }

    # Extract sections
    findings_match = re.search(
        r"## Key Findings\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if findings_match:
        result["key_findings"] = findings_match.group(1).strip()

    context_match = re.search(
        r"## Relevant Context\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if context_match:
        result["relevant_context"] = context_match.group(1).strip()

    recs_match = re.search(
        r"## Recommendations for Builder\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if recs_match:
        result["recommendations"] = recs_match.group(1).strip()

    status_match = re.search(
        rf"## Status{_SECTION_VALUE_LEAD}(ready_for_builder|need_replan|no_relevant_knowledge)\b",
        content,
        re.IGNORECASE,
    )
    if status_match:
        result["status"] = status_match.group(1).lower()

    return result


# A `## Files Modified` line is prose, and the paths in `files_changed` are raw
# `filesystem_write` arguments. The "Described but not written" check compares
# the two, so every way the model can spell a path it really did write turns
# into a false accusation -- and it is the worst one the report makes: the
# mirror of a Builder claiming a file it never wrote, pointing the other way,
# read by an Architect that rules on the report. The model decorates the line
# in all the ordinary ways -- `- \`test_spectral_graph.py\``, `- **file.py**`,
# `- ./file.py`, an absolute path, `* file.py`, `1. file.py` -- and above all
# it annotates: `- test_spectral_graph.py (new file)`, which put a written,
# executed, passing file under "Described but not written".
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MARKDOWN_WRAP = re.compile(r"^(?:\*\*|__|`|\*|_)+|(?:\*\*|__|`|\*|_)+$")
# A whitelist rather than "strip any trailing parenthetical": this project has
# real filenames that carry parentheses (examples/filter_band_pass_(40_60_hz).png),
# and mangling one of those would reintroduce the same false accusation.
_PATH_ANNOTATION = re.compile(
    r"\s+\((?:new|newly|created|create|added|modified|updated|edited|changed|"
    r"rewritten|rewrote|overwritten|existing|unchanged|deleted|removed)\b[^()]*\)$",
    re.IGNORECASE,
)
# Lines that are an answer of "nothing", not a path. Without these a Builder
# that honestly reported writing no files was accused of not writing "None".
_NOT_A_PATH = {"", ".", "-", "none", "n/a", "na", "(none)", "nothing", "no files"}


def _report_path_key(text: str) -> str:
    """Normalize one `## Files Modified` entry for comparison against tool records.

    Returns "" for a line that is not naming a file at all. Normalization is
    deliberately one-directional in its risk: over-stripping could only ever
    hide a real accusation, while under-stripping invents one, and an invented
    one is what reaches the Architect as evidence.
    """
    path = _LIST_MARKER.sub("", text).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip()
    path = _PATH_ANNOTATION.sub("", path).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip().rstrip(":,;").strip()

    if not path or path.lower() in _NOT_A_PATH:
        return ""
    # Prose, not a path: a sentence in place of a bullet list. Five words is
    # well clear of any real filename, and missing one costs a warning we did
    # not print rather than one we made up.
    if len(path.split()) >= 5:
        return ""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(Path.cwd())
        except ValueError:
            # Outside the project; keep it absolute and let it compare as-is.
            pass
    return os.path.normpath(str(candidate))


def _parse_builder_output(content: str) -> dict[str, Any]:
    """Parse Builder output into structured format.

    Expected format:
    ## Changes Made
    ...

    ## Files Modified
    - path/to/file
    ...

    ## Next Steps / Blockers
    ...
    """
    result = {"changes_made": "", "files_modified": [], "next_steps_blockers": ""}

    changes_match = re.search(
        r"## Changes Made\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if changes_match:
        result["changes_made"] = changes_match.group(1).strip()

    files_match = re.search(
        r"## Files Modified\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if files_match:
        # Extract file paths from bullet list, normalized: the raw line carries
        # list markers, markdown and "(new file)"-style annotations that no
        # recorded tool path will ever match.
        seen: set[str] = set()
        paths: list[str] = []
        for line in files_match.group(1).strip().split("\n"):
            key = _report_path_key(line)
            if key and key not in seen:
                seen.add(key)
                paths.append(key)
        result["files_modified"] = paths

    # "## Blockers" is the heading the prompt asks for. "## Next Steps /
    # Blockers" was the old one, and a heading that invited next steps got them.
    blockers_match = re.search(
        r"## (?:Next Steps / )?Blockers\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if blockers_match:
        result["next_steps_blockers"] = blockers_match.group(1).strip()

    return result


def _parse_architect_output(content: str, reviewing: bool = False) -> dict[str, Any]:
    """Parse Architect output into architecture text and a verdict.

    Args:
        content: Raw model output.
        reviewing: Whether this was the gate pass. It picks the fallback when
            no verdict parses: before the Builder has reported the only sound
            ruling is `plan`, and after it the sound ruling is `approved`, so a
            malformed response ends the run instead of looping on it forever.
    """
    result: dict[str, Any] = {
        "architecture": "",
        "verdict": Verdict.APPROVED.value if reviewing else Verdict.PLAN.value,
    }

    architecture = ""
    arch_match = re.search(
        r"## Architecture\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if arch_match:
        architecture = arch_match.group(1).strip()

    # Constraints ride along in the same field: they are the half the Planner
    # and Builder actually have to obey, and state injection renders one value.
    constraints_match = re.search(
        r"## Constraints\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if constraints_match and constraints_match.group(1).strip():
        constraints = constraints_match.group(1).strip()
        architecture = f"{architecture}\n\nConstraints:\n{constraints}".strip()

    result["architecture"] = architecture

    verdicts = "|".join(verdict.value for verdict in Verdict)
    verdict_match = re.search(
        rf"## Verdict{_SECTION_VALUE_LEAD}({verdicts})\b", content, re.IGNORECASE
    )
    if verdict_match:
        result["verdict"] = verdict_match.group(1).strip().lower()

    return result


def _rule_on_state(state: AgentState, reviewing: bool) -> dict[str, str]:
    """Ask the Architect for direction or a ruling; return the parsed output.

    Reads state; never writes it, so it is safe to run under `_with_deadline`.
    """
    state_injection = _get_state_injection(state)
    goal = state.get("goal", "")
    task = (
        "The Builder has reported. Rule on the work."
        if reviewing
        else "Set the architectural direction for this goal."
    )

    messages = [
        SystemMessage(content=ARCHITECT_PROMPT),
        HumanMessage(content=f"{state_injection}\n\nUser goal: {goal}\n\n{task}"),
    ]

    llm = get_agent_llm("architect")
    response = llm.invoke(messages)
    return _parse_architect_output(_as_text(response.content), reviewing=reviewing)


def architect_node(state: AgentState) -> AgentState:
    """Architect: set direction, then rule on whether the work is done.

    No tools. Runs twice per cycle -- as the entry authority before the Planner,
    and as the approval gate after the Builder reports. A populated builder
    report is what tells the seat which pass it is on. Whatever it answers, the
    opening pass rules `plan` -- see the comment where that is enforced.

    Bounded by `NODE_DEADLINE_SECONDS`. The fallback verdict is never
    `approved`, and that is the whole point of choosing one here: this gate is
    what ends the run, so a seat that stalled must not be able to end one
    successfully. `revise` on the gate pass and `plan` on the opening pass both
    route back to the Planner, and the step counter below carries a repeatedly
    stalling Architect to MAX_STEPS instead of letting it spin.
    """
    reviewing = bool(state.get("builder_report"))

    if RUN_CONTROL.stopped():
        # A stopped gate can no more end the run successfully than a timed-out
        # one can, and for the same reason: this node is what ends it. Nothing
        # was ruled here, so nothing is approved. `step_count` is deliberately
        # not advanced -- this pass did no work -- and the architecture already
        # in state is kept, since the recovered run is read with it.
        if reviewing:
            state["verdict"] = Verdict.REVISE.value
        state["messages"].append(
            "[Architect] Stopped by the emergency stop before this seat ran; "
            "no verdict was reached."
        )
        return state

    parsed = _with_deadline(
        lambda: _rule_on_state(state, reviewing), NODE_DEADLINE_SECONDS, None
    )
    timed_out = parsed is None
    if parsed is None:
        parsed = {
            "architecture": "",
            "verdict": (
                Verdict.REVISE.value if state.get("plan") else Verdict.PLAN.value
            ),
        }

    # The opening pass has exactly one ruling, and the prompt says so: `plan`.
    # Nothing has been planned or built, so there is nothing to approve -- an
    # opening `approved` ended the run before any seat but this one had worked
    # -- and `need_research` routed around the Planner, sending the Researcher
    # an empty `plan` to search on. That second path does not end: the gate
    # counts a step only while a plan exists (below), and a Planner that never
    # runs never writes one, so the Researcher -> Builder -> Architect loop went
    # uncounted until LangGraph's recursion limit killed the run by exception.
    # A missing plan marks the opening pass, for the reason the step count
    # gives. What the seat actually said is kept in the feed line.
    opening = not state.get("plan")
    answered = parsed["verdict"]
    if opening and answered != Verdict.PLAN.value:
        parsed["verdict"] = Verdict.PLAN.value

    # Keep the opening architecture if the gate pass did not restate it --
    # losing it mid-run would strip the constraints out of every later prompt.
    if parsed["architecture"]:
        state["architecture"] = parsed["architecture"]
    state["verdict"] = parsed["verdict"]

    # Work without evidence cannot be approved, whatever the Architect
    # concluded. This is the one place the gate's ruling is overridden, and it
    # is deliberate: the Architect reads the failure in `blockers` and had
    # approved past it. The verdict is rewritten rather than the routing
    # patched, so the state says what actually happened. Four things block:
    #
    # - a file that ran and failed, unless the run asked for failing files;
    # - a file nobody executed (`unverified`), whatever that opt-out says. It is
    #   for a file meant to fail, not for a gap in the evidence, and clearing
    #   the whole list under it let unrun files through;
    # - a Builder pass cut off before it finished (`builder_cut_off`). Its
    #   blocker said so, the Architect approved anyway, and nothing stopped it.
    # - a file that fails lint (`lint_failed`), whatever the opt-out says: CI
    #   lints a fixture meant to fail at runtime all the same.
    #
    # Note the cost: a goal that legitimately calls for a failing file needs
    # `expect_failures`, and work too big for one pass is not approved until a
    # pass finishes it. MAX_STEPS still ends a run that never gets there.
    unverified = list(state.get("unverified") or [])
    failed_files = [
        path for path in (state.get("failed_verification") or []) if path not in unverified
    ]
    if state.get("expect_failures"):
        # The caller asked for a failing file, so a failure is the product.
        # The list stays in state and the report still shows it; it just does
        # not overrule the gate.
        failed_files = []
    raw_cut_off = str(state.get("builder_cut_off") or "")
    reasons: list[str] = []
    if failed_files:
        reasons.append(f"{len(failed_files)} file(s) do not run")
    if unverified:
        reasons.append(f"{len(unverified)} file(s) never executed")
    lint_failed = list(state.get("lint_failed") or [])
    if lint_failed:
        reasons.append(f"{len(lint_failed)} file(s) fail lint")
    if raw_cut_off:
        # A reason nobody has wording for is still a pass that did not finish.
        reasons.append(
            _CUT_OFF_REASONS.get(raw_cut_off, f"the Builder did not finish ({raw_cut_off})")
        )
    overridden = bool(reasons) and state["verdict"] == Verdict.APPROVED.value
    if overridden:
        state["verdict"] = Verdict.REVISE.value

    # The gate is the one point every cycle passes through, so it is where the
    # loop counter belongs. The Builder used to own it, which let a
    # Planner/Researcher loop run without ever counting a step.
    # Every pass but the opening one closes a cycle, so that is what counts a
    # step. Keying this off `reviewing` instead undercounted: a Builder that
    # reports nothing leaves `reviewing` False, the gate reads the cycle as a
    # fresh opening pass and sends the work round again, and that shape repeats
    # uncounted until LangGraph hits its recursion limit and kills the run --
    # discarding every message the run had produced. A missing plan is what
    # actually marks the entry pass; a missing report does not.
    if state.get("plan"):
        state["step_count"] = state.get("step_count", 0) + 1

    if timed_out:
        # Said plainly, because the run reads this line to know what happened:
        # a verdict nobody actually reached is not the same as a ruling.
        note = (
            f" (no response within {int(NODE_DEADLINE_SECONDS)}s -- "
            "not a ruling, and never an approval)"
        )
    elif opening and answered != state["verdict"]:
        note = (
            f" (the seat answered `{answered}`, but the opening pass rules "
            "`plan`: nothing has been planned or built yet)"
        )
    elif overridden:
        note = f" (approval blocked: {'; '.join(reasons)})"
    elif reasons:
        note = f" ({'; '.join(reasons)})"
    else:
        note = ""
    state["messages"].append(f"[Architect] Verdict: {state['verdict']}{note}")

    return state


# Whether the Planner is shown the project files the corpus ranks closest to
# the goal before it plans. Switched off by `tests/conftest.py` for every test,
# for the reason it switches off indexing: a planning test must not pass or
# fail by what the checkout it ran in happens to have indexed.
PLANNER_PROJECT_MAP = os.getenv("PLANNER_PROJECT_MAP", "1").strip().lower() not in {
    "0", "false", "no",
}

# How many corpus hits the Planner is shown. Enough to name the files a goal
# touches, few enough that a small local seat still reads the goal itself.
PLANNER_MAP_RESULTS = 6

# The longest excerpt of each hit the Planner sees. The map is for deciding
# where work goes, not for reading the code -- that is the Researcher's pass.
PLANNER_MAP_EXCERPT_CHARS = 160


def _project_map(goal: str) -> str:
    """The project files the corpus ranks closest to the goal, for the Planner.

    The Planner used to plan from the goal and the Architect's direction alone,
    so it could name no file the goal did not, and its plan tended to restate
    the goal as steps. That costs twice: the Researcher searches on `plan`, so a
    plan naming nothing is a search for nothing in particular.

    Measured on 2026-09-12 with the local 9B Planner seat, three goals planned
    each way: without the map no plan named a single project file, and one
    listed the Planner's own instructions as its steps ("Break the goal into
    clear, ordered steps"). With it, the goal about seat timeouts was planned
    against `config.py`, `mcp_client.py` and `nodes.py`. The other two were
    about the console's UI, which that corpus had not yet indexed -- the map can
    only name what retrieval can find.

    Read through the same tool the Researcher uses, and only hits over the
    embedding model's relevance floor are shown (`relevance_floor`, measured
    on the built corpus rather than hard-coded) -- a map of files the corpus
    does not consider related would be invented structure, which is also why a
    model with no floor yet gets no map at all. The Planner still calls no
    tool: this is context handed to it, like the state injection. Returns ""
    when there is nothing worth showing, and never raises.
    """
    if not goal.strip():
        return ""
    try:
        response = _call_mcp_tool_sync(
            "search_knowledge_graph", {"query": goal, "top_k": PLANNER_MAP_RESULTS}
        )
        from langgraph_agent.graphrag_server import relevance_floor

        floor = relevance_floor()
    except Exception:
        return ""
    if floor is None:
        return ""

    results = response.get("results", []) if isinstance(response, dict) else []
    lines = []
    for result in results:
        if not isinstance(result, dict):
            continue
        score = float(result.get("score") or 0.0)
        if score <= floor:
            continue
        path = str((result.get("metadata") or {}).get("path") or result.get("id") or "")
        excerpt = " ".join(str(result.get("content") or "").split())
        lines.append(f"- {path} ({score:.2f}): {excerpt[:PLANNER_MAP_EXCERPT_CHARS]}")
    if not lines:
        return ""
    return (
        "Project files the corpus ranks closest to this goal (these paths exist):\n"
        + "\n".join(lines)
    )


def _make_plan(state: AgentState) -> dict[str, Any]:
    """Ask the Planner for a plan; return the parsed output.

    Reads state; never writes it, so it is safe to run under `_with_deadline`.
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)
    goal = state.get("goal", "")

    project_map = _project_map(goal) if PLANNER_PROJECT_MAP else ""
    map_block = f"\n\n{project_map}" if project_map else ""

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"{state_injection}{map_block}\n\nUser goal: {goal}"),
    ]

    llm = get_agent_llm("planner")
    response = llm.invoke(messages)

    # Parse the structured output
    return _parse_planner_output(_as_text(response.content))


# What a timed-out Planner leaves in `plan`. It must not be empty, and that is
# load-bearing rather than cosmetic: the Architect increments `step_count` only
# while a plan exists, so an empty one would send the run round the
# Planner/Builder loop uncounted until LangGraph's recursion limit killed it by
# exception -- discarding every message the run had produced. That is the exact
# failure the counter was moved to the gate to prevent, and an empty fallback
# here would reintroduce it through the back door.
_PLANNER_TIMED_OUT = (
    "1. The Planner did not respond within {seconds}s, so this goal was never "
    "broken into steps.\n"
    "2. Do not guess at the plan. Report the goal as unplanned, and set that "
    "as a blocker so the Architect sees why nothing was implemented.\n"
)


# What a Planner that *answered* off-format leaves in `plan`, and it is
# load-bearing for exactly the reason `_PLANNER_TIMED_OUT` is. `plan` was
# whatever `## Steps` matched, so a reply that skipped the heading -- prose, a
# bare list, a JSON object -- set it to the empty string while the feed still
# said "Plan created", and the gate counts a step only while a plan exists -- so
# the cycle ran uncounted. Measured with this guard removed: a whole run through
# such a seat ends at `step_count` 0, which is the state that lets Planner ->
# Builder -> Architect repeat until LangGraph's recursion limit kills the run by
# exception, discarding every message it produced, whenever the gate does not
# approve. That is the same failure the timeout fallback exists to prevent,
# reached through the one door that was unguarded: the seat replying rather than
# stalling. Worded apart from the timeout, and from a plan that really is
# unplannable, because all three arrive with no steps and only this one means
# the model cannot hold the seat -- the same distinction `_RESEARCH_EMPTY`
# draws for the Researcher.
_PLANNER_NO_STEPS = (
    "1. The Planner's seat replied without a readable `## Steps` section, so "
    "this goal was never broken into steps.\n"
    "2. Do not guess at the plan. Report the goal as unplanned and set that as "
    "a blocker, saying the Planner's reply could not be read as a plan -- this "
    "is a seat that cannot hold its format, not a goal that resists "
    "planning.\n"
)


# The stable openings of the two placeholders above, without the parts that
# vary -- `_PLANNER_TIMED_OUT` carries the deadline, and neither is worth
# reconstructing at a call site.
_PLACEHOLDER_PLAN_OPENINGS = (
    "1. The Planner did not respond within",
    "1. The Planner's seat replied without a readable",
)


def plan_is_placeholder(plan: str) -> bool:
    """Is this `plan` a note about the Planner failing, rather than a plan?

    Both placeholders exist so `plan` is never empty -- see
    `_PLANNER_NO_STEPS` for why that emptiness was load-bearing -- and both are
    written on a path that also forces `next_agent` to Builder, deliberately.
    So anything that wants to *override* that routing has to be able to tell
    the two apart, and `_route_from_planner` is that caller: a plan nobody
    wrote is not a query worth searching on, and the seat already stalled once.
    """
    return plan.lstrip().startswith(_PLACEHOLDER_PLAN_OPENINGS)


def planner_node(state: AgentState) -> AgentState:
    """Planner: Interpret goal, create structured plan, choose next agent.

    As specified:
    - No tools
    - Output strict format
    - Routes to Researcher when knowledge needed, Builder when task is clear

    Bounded by `NODE_DEADLINE_SECONDS`; see `_PLANNER_TIMED_OUT` for why the
    fallback plan is a real string rather than an empty one.
    """
    if RUN_CONTROL.stopped():
        # Any plan already in state is kept -- see `_PLANNER_TIMED_OUT` for why
        # an empty one here is load-bearing rather than cosmetic. `next_agent`
        # is left alone so the recovered state still records the last real
        # routing decision rather than one nobody made.
        state["messages"].append(
            "[Planner] Stopped by the emergency stop before this seat ran."
        )
        return state

    parsed = _with_deadline(
        lambda: _make_plan(state), NODE_DEADLINE_SECONDS, None
    )

    if parsed is None:
        # An existing plan beats the placeholder: on a revise cycle state
        # already holds a real one, and re-running it is better information
        # than a note saying the seat stalled.
        state["plan"] = state.get("plan") or _PLANNER_TIMED_OUT.format(
            seconds=int(NODE_DEADLINE_SECONDS)
        )
        # Builder rather than Researcher: it is the shorter path back to the
        # Architect, which is the only node that can end the run, and a second
        # slow seat in between is the last thing a stalling run needs.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Planner] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder"
        )
        return state

    plan = parsed.get("plan", "")
    if not plan.strip():
        # The seat answered and the answer carried no steps -- see
        # `_PLANNER_NO_STEPS` for why an empty `plan` cannot be written here.
        # An existing plan beats the placeholder, as on the timeout path: a
        # revise cycle already holds a real one, and re-running it is better
        # information than a note about the seat.
        state["plan"] = state.get("plan") or _PLANNER_NO_STEPS
        # Builder rather than whatever the reply routed to, which is a second
        # reason this cannot be left alone: `_gather_research` searches on
        # `plan`, so a Researcher hop with no plan is a search for the empty
        # string, and retrieval that thin is precisely what falls through to
        # the Researcher's own model. The Builder is also the shorter path back
        # to the Architect, the only node that can end the run.
        state["next_agent"] = "Builder"
        # Named in the feed the way the silent Researcher is, because changing
        # the seat's model is the only thing that fixes it.
        state["messages"].append(
            "[Planner] Seat returned no plan steps; routing to Builder "
            "(check the Planner's model)"
        )
        return state

    # Update state
    state["plan"] = plan
    state["next_agent"] = parsed.get("next_agent", "Builder")
    state["messages"].append(f"[Planner] Plan created. Next agent: {state['next_agent']}")

    return state


def _as_text(content: Any) -> str:
    """Flatten a message's content to the text the model answered with.

    Providers differ: some return a plain string, some a list of content
    blocks. `len()` and the section regexes both read a list as truthy
    non-empty, so an answer that carried no text at all still looked like
    findings.

    Every node reads its seat's reply through here, because a Claude model
    that thinks answers with a list -- a `thinking` block, then the text --
    and a regex handed that list raises, while `str()` of it is a Python repr
    whose sections never match. Only `text` blocks are kept: the reasoning is
    how the model got to its answer, not part of it, and a plan or verdict
    parsed out of it would be one the model never gave.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            (block.get("text", "") if block.get("type", "text") == "text" else "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _said_nothing(content: str, parsed: dict[str, Any]) -> bool:
    """True when the Researcher's answer carries no findings.

    Two shapes count as nothing: an empty (or whitespace) response, and one
    that filled in the headings and the status line but left every section
    blank. Both reach the Builder as an empty `research`.
    """
    if not content.strip():
        return True
    return not any(
        parsed.get(section, "").strip()
        for section in ("key_findings", "relevant_context", "recommendations")
    )


# Internal marker, never a `research_status` in state: `researcher_node` maps
# it to `no_relevant_knowledge` and emits its own message. It exists for the
# same reason the deadline's does -- a seat that answered with nothing and a
# corpus with nothing to say both reach the Builder empty-handed, and the feed
# is where an operator finds out which one happened.
_SEAT_EMPTY = "seat_empty"


# What the Researcher hands the Builder when its seat answered with nothing.
# Worded apart from the timeout and from a genuinely empty corpus: all three
# arrive with no findings, and only this one means the seat is not working.
# Saying so in the text is what stops the Builder's report from implying the
# corpus was searched and found wanting.
_RESEARCH_EMPTY = (
    "## Key Findings\nNone -- the Researcher's seat returned no findings.\n\n"
    "## Relevant Context\nThe model answered with nothing usable, so no "
    "retrieval was summarized. This says nothing about the knowledge base: "
    "the corpus may hold relevant material that was never reported. Check "
    "the seat's model on the console before reading this as an empty "
    "corpus.\n\n"
    "## Recommendations for Builder\nWork from the plan alone, and say in the "
    "report that it was built without research because the Researcher seat "
    "returned nothing.\n\n"
    "## Status\nno_relevant_knowledge"
)


def _research_snippet(content: str) -> str:
    """A retrieved passage as the Builder should see it: whole, or visibly cut.

    Says when it truncated. Everything else in this project that shortens text
    on its way somewhere announces it -- `_fit_to_index_limit` writes a note
    onto the page, `_timeout_detail` keeps the tail and says so -- because a
    silent trim is indistinguishable from a source that simply had nothing more
    to say, and the Builder has no way to ask.
    """
    text = (content or "").strip()
    if len(text) <= RESEARCH_SNIPPET_CHARS:
        return text
    return text[:RESEARCH_SNIPPET_CHARS].rstrip() + f"\n   [... passage truncated at {RESEARCH_SNIPPET_CHARS} characters]"


# How much of what retrieval returned the Researcher's seat is shown when it is
# asked to judge it -- a search whose best hit fell under the relevance floor.
# Smaller than what reaches the Builder, deliberately: the seat is a local model
# with a few thousand tokens of window, the prompt and the state already fill
# part of it, and the question it answers is whether these passages bear on the
# plan at all, which their opening lines settle.
RESEARCHER_SEAT_PASSAGES = 3
RESEARCHER_SEAT_EXCERPT_CHARS = 600


def _passage_source(result: dict[str, Any]) -> str:
    """Where a retrieved passage came from, to the line when the search could tell."""
    source = str(result.get("id") or "unknown source")
    if result.get("line"):
        source += f":{result['line']}"
    return source


def _retrieval_for_the_seat(
    results: list[dict[str, Any]], floor: float | None, why_none: str
) -> str:
    """What retrieval found, put in front of the Researcher's seat to judge.

    The seat is asked only when the search did not clearly answer the plan, and
    it used to be asked with nothing: its prompt told it to call GraphRAG, it
    was offered no tool, and the passages the search *had* returned were
    dropped before it saw them -- so a local model with no evidence was left to
    invent a codebase or narrate a tool call it could not make. It is now
    shown what came back, told why none of it was accepted automatically, and
    judges it; `why_none` says why there is nothing, when there is nothing.
    """
    if not results:
        return (
            f"Retrieval: nothing to judge -- {why_none}. Do not present anything "
            "as retrieved. If the plan depends on knowing existing code or docs, "
            "answer `no_relevant_knowledge`."
        )

    top = float(results[0].get("score") or 0.0)
    if floor is None:
        verdict = (
            "This corpus has no measured relevance floor yet, so none of them "
            "could be accepted automatically"
        )
    else:
        verdict = (
            f"The best scored {top:.2f}, under the relevance floor of "
            f"{floor:.2f}, so none was accepted automatically"
        )
    lines = [
        f"Retrieval: the knowledge base was searched for this plan and returned "
        f"the passages below. {verdict}. Judge each yourself: cite a path only "
        "where its passage supports the point, and answer "
        "`no_relevant_knowledge` if none bears on the plan.",
    ]
    for i, result in enumerate(results[:RESEARCHER_SEAT_PASSAGES], 1):
        text = str(result.get("content") or "").strip()
        if len(text) > RESEARCHER_SEAT_EXCERPT_CHARS:
            text = text[:RESEARCHER_SEAT_EXCERPT_CHARS].rstrip() + " [...]"
        score = float(result.get("score") or 0.0)
        lines.append(f"\n{i}. {_passage_source(result)} (score {score:.2f})\n{text}")
    return "\n".join(lines)


def _gather_research(state: AgentState) -> tuple[str, str]:
    """Retrieve for the Researcher and return `(findings, status)`.

    Reads state; never writes it. Split out of `researcher_node` so it can run
    under `_with_deadline`, which may abandon it still running -- see that
    function for why a worker that writes state is a bug.
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)
    plan = state.get("plan", "")

    # Call GraphRAG through the MCP tool boundary
    graphrag_results: dict[str, Any] | None = None
    # What the Researcher's seat is shown when the search did not answer: the
    # passages it returned, the floor they were held to, and -- when there are
    # none -- why.
    results: list[dict[str, Any]] = []
    floor: float | None = None
    why_none = "the search returned nothing for this plan"
    if not plan.strip():
        why_none = "the plan is empty, so there was nothing to search on"

    try:
        search_response: dict[str, Any] = _call_mcp_tool_sync(
            "search_knowledge_graph", {"query": plan, "top_k": RESEARCH_RESULTS}
        )
        results = list(search_response.get("results") or [])
        if search_response.get("source") == "no_corpus":
            why_none = "no corpus has been built on this machine"

        # Imported here rather than at module scope for the reason
        # `mcp_client` does the same: `graphrag_server` pulls in chromadb, and
        # the tool call above has already paid for that by the time we rule on
        # what it returned. The floor is a property of the embedding model, so
        # it is read from where the model is named -- measured against the
        # built corpus by `calibrate_relevance_floor`, and `None` until that
        # has run. A model with no floor cannot tell an answer from noise, so
        # nothing it retrieves counts as answered.
        from langgraph_agent.graphrag_server import relevance_floor

        floor = relevance_floor()

        # Check if we got real results
        if floor is not None and results and results[0].get("score", 0) > floor:
            graphrag_results = {"results": results, "source": "local_graphrag"}

            # Try to get graph info too via the MCP query tool. The top hit's
            # id is its document's node in the graph, so it is asked for by
            # that id. A name derived from it used to be asked for instead --
            # the file's stem, or for a file at the root the plan's first
            # word ("1.") -- and a stem can resolve to an entity of the same
            # name rather than the document (`CLAUDE` to `Claude`).
            try:
                first_doc = results[0].get("id", "")
                if first_doc:
                    graph_response = _call_mcp_tool_sync(
                        "query_knowledge_graph", {"entity": first_doc, "hops": 2}
                    )
                    graphrag_results["graph"] = graph_response
            except Exception:
                pass  # Graph query is optional
    except Exception as e:
        # GraphRAG MCP tool unavailable or failed; will use LLM fallback
        graphrag_results = {"error": str(e)}
        why_none = f"the knowledge-base search failed ({e})"

    # Check if we got real results from GraphRAG
    has_real_results = (
        graphrag_results
        and graphrag_results.get("source") == "local_graphrag"
        and graphrag_results.get("results")
        and len(graphrag_results["results"]) > 0
    )

    if has_real_results:
        assert graphrag_results is not None and floor is not None
        # Format GraphRAG results into Researcher output format
        shown = graphrag_results.get("results", [])[:RESEARCH_RESULTS]
        research_findings = "## Key Findings\n"

        above = 0
        for i, result in enumerate(shown, 1):
            content = _research_snippet(result.get("content", ""))
            score = float(result.get("score") or 0.0)
            # Where it came from, to the line when the search could tell. The
            # passages used to arrive with no source, so the Builder was handed
            # code it could not open the file of.
            research_findings += f"\n{i}. {_passage_source(result)}\n{content}"
            if result.get("related_entities"):
                research_findings += (
                    f"\n   Related: {', '.join(str(e) for e in result['related_entities'][:3])}"
                )
            # Only the top hit has to clear the floor for retrieval to count as
            # an answer, and the rest ride along -- they are the diverse hits
            # SEARCH_ESCALATION widens the window for. So each says which side
            # of the floor it is on, rather than all of them being announced as
            # relevant: the Builder weighs one under the floor as weaker.
            if score > floor:
                above += 1
                research_findings += f"\n   Score: {score:.2f}\n"
            else:
                research_findings += f"\n   Score: {score:.2f} (under the relevance floor)\n"

        research_findings += "\n## Relevant Context\n"
        research_findings += (
            f"Retrieved {len(shown)} passage(s) from the knowledge base; {above} "
            f"scored over its relevance floor of {floor:.2f}, and any under it is "
            "marked above.\n"
        )

        graph_response = graphrag_results.get("graph", {})
        if graph_response and graph_response.get("subgraph_nodes", 0) > 0:
            research_findings += (
                f"\nKnowledge graph has {graph_response['subgraph_nodes']} nodes "
                f"and {graph_response['subgraph_edges']} relationships.\n"
            )

        research_findings += "\n## Recommendations for Builder\n"
        research_findings += "Use the retrieved documentation as reference for implementation.\n"
        research_findings += "\n## Status\nready_for_builder"

        research_status = "ready_for_builder"

    else:
        # The seat judges what the search returned, when it did not clearly
        # answer the plan -- see `_retrieval_for_the_seat`.
        retrieval = _retrieval_for_the_seat(results, floor, why_none)
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(
                content=f"{state_injection}\n\nPlan to research:\n{plan}\n\n{retrieval}"
            ),
        ]

        llm = get_agent_llm("researcher")
        response = llm.invoke(messages)
        research_findings = _as_text(response.content)

        # Parse status from LLM response
        parsed = _parse_researcher_output(research_findings)
        research_status = parsed.get("status", "ready_for_builder")

        # A seat that answered with nothing has not done research, whatever the
        # parsed status says -- and the status defaults to `ready_for_builder`,
        # so silence was being announced as success. That is the same rule the
        # Builder is held to: the seat's account of its own work is not
        # evidence. The cost of missing it is not one bad cycle but a loop --
        # empty `research` reaches the Builder, whose report says the store is
        # empty, so the gate rules `need_research` and sends it back to the
        # same silent seat, burning a step at the ceiling every time.
        if _said_nothing(research_findings, parsed):
            research_findings = _RESEARCH_EMPTY
            research_status = _SEAT_EMPTY

    return research_findings, research_status


# What the Researcher hands the Builder when it runs out of time. The status is
# `no_relevant_knowledge` rather than `need_replan` because a deadline says
# nothing about the plan -- looping back to the Planner would re-run the same
# slow retrieval and burn the run's budget on it. The Builder is told plainly
# that it has no research, so its report cannot silently imply otherwise.
_RESEARCH_TIMED_OUT = (
    "## Key Findings\nNone -- retrieval did not finish.\n\n"
    "## Relevant Context\nThe Researcher was stopped at its "
    "{seconds}s deadline before it produced findings. Treat this as no "
    "research rather than as an empty corpus: the knowledge base may well "
    "hold relevant material that was not retrieved in time.\n\n"
    "## Recommendations for Builder\nWork from the plan alone, and say in "
    "the report that it was built without research.\n\n"
    "## Status\nno_relevant_knowledge"
)


def researcher_node(state: AgentState) -> AgentState:
    """Researcher: Query GraphRAG, summarize findings, recommend approach.

    As specified:
    - GraphRAG tools only
    - Output strict format with status
    - Status guides next steps (ready_for_builder | need_replan | no_relevant_knowledge)

    Calls the GraphRAG MCP tool (`search_knowledge_graph`) rather than importing
    the knowledge base directly, preserving the documented tool boundary.
    Falls back to the LLM if the knowledge base is empty or the MCP tool fails.

    Bounded by `NODE_DEADLINE_SECONDS`. Without it a stalled seat hung the whole
    run here: `RUN_BUDGET_SECONDS` is checked between graph supersteps, and a
    node that never returns never reaches one, so the run sat inside this
    function indefinitely while the console still showed the Planner as current.
    """
    if RUN_CONTROL.stopped():
        # Distinct from an empty corpus: nothing was retrieved because nothing
        # was attempted. Leaving `research` untouched keeps whatever an earlier
        # cycle found instead of overwriting it with a note.
        state["messages"].append(
            "[Researcher] Stopped by the emergency stop before this seat ran."
        )
        return state

    timed_out = (_RESEARCH_TIMED_OUT.format(seconds=int(NODE_DEADLINE_SECONDS)), "timed_out")
    research_findings, research_status = _with_deadline(
        lambda: _gather_research(state), NODE_DEADLINE_SECONDS, timed_out
    )

    deadline_hit = research_status == "timed_out"
    seat_empty = research_status == _SEAT_EMPTY
    if deadline_hit or seat_empty:
        research_status = ResearchStatus.NO_RELEVANT_KNOWLEDGE.value

    # Update state
    state["research"] = research_findings
    state["research_status"] = research_status

    # Route based on status
    if deadline_hit:
        # Worded apart from the ordinary `no_relevant_knowledge` message: a
        # corpus with nothing to say and a seat that stopped answering both
        # reach the Builder empty-handed, and only one of them is a problem.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Researcher] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder without research"
        )
    elif seat_empty:
        # Routed to the Builder, not back to the Planner: the plan is not what
        # failed, and re-planning would send the run at the same silent seat
        # again. Named in the feed so the operator can change the seat's model,
        # which is the only thing that actually fixes it.
        state["next_agent"] = "Builder"
        state["messages"].append(
            "[Researcher] Seat returned no findings; routing to Builder "
            "without research (check the Researcher's model)"
        )
    elif research_status == "need_replan":
        state["next_agent"] = "Planner"
        state["messages"].append("[Researcher] Needs replan")
    elif research_status == "no_relevant_knowledge":
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] No relevant knowledge, proceeding")
    else:
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] Research complete, routing to Builder")

    return state


# The tools the Builder is allowed to call. GraphRAG is deliberately absent:
# retrieval belongs to the Researcher, and a Builder that can search the corpus
# stops working from the plan it was handed.
BUILDER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filesystem_read",
            "description": (
                "Read a UTF-8 text file and return its contents. Read a file "
                "before rewriting it, so the rewrite keeps what is already there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the project root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_write",
            "description": (
                "Write the COMPLETE new contents of a file, creating parent "
                "directories as needed. This replaces the entire file, so to "
                "change an existing file call filesystem_read first and send "
                "the whole modified text back -- never send only the new lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {
                        "type": "string",
                        "description": "The entire contents the file should have afterwards.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working tree status as porcelain output.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the unstaged diff, optionally limited to one path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Optional path to diff."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_dwell",
            "description": (
                "Run the whole git flow in order: survey, branch, stage, "
                "commit, push, open a pull request, merge it. Prefer this over "
                "a series of terminal git commands -- it runs the stages in "
                "the only order that works, stops at the first failure and "
                "says which stage stopped it. It never commits onto the "
                "default branch: it creates a branch instead. The default runs "
                "every stage, so the change lands on the default branch as one "
                "squashed commit and the branch is deleted. To stop short of "
                "that -- to leave the pull request open for someone to read -- "
                "name the stages you want and leave 'merge' out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "Commit message; its first line becomes the pull "
                            "request title and the whole message its body. "
                            "Required for the commit stage."
                        ),
                    },
                    "stages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stages to run, from: survey, branch, stage, "
                            "commit, push, pr, merge. They always run in that "
                            "order. Defaults to all of them; pass a shorter "
                            "list to stop early, e.g. everything up to 'pr' to "
                            "leave the pull request unmerged."
                        ),
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Paths to stage. Defaults to every change in the "
                            "working tree; name paths to commit only your own."
                        ),
                    },
                    "branch": {
                        "type": "string",
                        "description": (
                            "Branch to create when HEAD is the default branch. "
                            "Derived from the message when omitted."
                        ),
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": (
                "Run one program in the project. There is no shell: the command is "
                "split into arguments and run directly, so quoting works and any "
                "character may appear inside an argument -- "
                "python -c \"import x; print(x.y)\" is fine, and so are ( ) ; ^ "
                "in a regex. "
                "Because there is no shell, shell *syntax* does nothing here. A "
                "pipe, a redirect (2>/dev/null included), && , || or ; is refused "
                "with an error saying so, rather than handed to the program as a "
                "meaningless argument. None of them is needed: the whole output "
                "comes back to you, with stdout and stderr as separate fields, so "
                "read it rather than piping it through head or grep, and run one "
                "program per call. To write a file use filesystem_write. "
                "Nothing is expanded either: * and ~ reach the program exactly as "
                "written, as do $(...), $VAR and backticks. find . -name \"*.py\" "
                "works, because find wants the literal; cat dir/* does not -- list "
                "the directory with ls and then name the file, and write absolute "
                "paths instead of ~. "
                "For the same reason there is no `cd` to run -- it is a shell builtin, "
                "not a program -- so pass `cwd` to choose the directory the command "
                "runs in. "
                f"The command is killed after {int(TERMINAL_TIMEOUT_SECONDS)} seconds "
                "unless you pass a longer `timeout`; a kill is reported as a timeout "
                "with whatever the command printed first, which is not the same thing "
                "as the command failing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to run the command in. Defaults to the project "
                            "root, which is also what a relative path here is relative "
                            "to. Use this instead of `cd`, which is not a program and "
                            "cannot be run."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Seconds to allow before the command is killed. Defaults "
                            f"to {int(TERMINAL_TIMEOUT_SECONDS)}, capped at "
                            f"{int(TERMINAL_TIMEOUT_MAX_SECONDS)}. Raise it for a "
                            "command you expect to be slow rather than reading the "
                            "kill as a failure."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the pytest suite, optionally limited to one path. Pass cwd "
                "to run a separate project's suite in its own directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional test path."},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to run pytest in. Defaults to the project "
                            "root, where the suite is tests/; given a cwd and no "
                            "path, pytest collects from that directory."
                        ),
                    },
                },
            },
        },
    },
]

BUILDER_TOOL_NAMES = {tool["function"]["name"] for tool in BUILDER_TOOLS}

# A `discuss_only` run gets **no** tools at all. This briefly allowed the three
# read-only ones -- `filesystem_read`, `git_status`, `git_diff` -- on the
# argument that reading changes nothing and grounds the discussion. The
# operator asked for zero, and zero is the stronger guarantee to state: "the
# Builder was offered no tools" needs no argument about which reads are
# harmless, and it cannot be weakened later by a tool added to the read-only
# set that turns out to do more than read.
#
# It is an empty set rather than a shorter list because the tools are then not
# bound at all -- `builder_node` takes the same path as a seat whose model
# cannot call tools, which already exists and is already tested. So there is no
# tool loop to reason about on a discussion run, and the guarantee is
# structural rather than a filter that has to be right.
DISCUSSION_TOOL_NAMES: frozenset[str] = frozenset()

DISCUSSION_TOOLS: list[dict[str, Any]] = []

# Appended to the Builder's prompt on a discussion run. Without it the seat
# spends its turns discovering the refusals one at a time: it is told to
# implement the plan, reaches for `filesystem_write`, is refused, and tries
# again -- `MAX_BUILDER_TOOL_TURNS` is 8, and a pass can exhaust it learning
# what it was never going to be allowed to do. The same reason `cwd` went into
# the tool schema rather than being left to an error message.
DISCUSSION_NOTE = (
    "\n\nThis run is DISCUSSION ONLY. You have no tools: you cannot read "
    "files, write files, run commands or run tests. Work from the plan and the "
    "research findings you were given. Do not report work as done and do not "
    "list files under '## Files Modified' -- describe what you would change, "
    "which files it would touch, and what you would need to verify it. That "
    "description is the product of this run. Answer '## Blockers' with none: a "
    "proposal is not blocked by needing review or approval."
)

# Appended to the Builder's prompt when the operator pointed the run at a
# generated project, for the reason `DISCUSSION_NOTE` exists: a seat that learns
# the boundary from refusals spends its turns doing it.
OUTPUT_DIR_NOTE = (
    "\n\nThis run builds a separate project in {output_dir}/. Write every "
    "file under that directory, spelling the full path from the project root "
    "(for example {output_dir}/main.py) -- filesystem_write refuses any other "
    "path. Pass cwd={output_dir} to terminal_execute and run_tests when you "
    "run what you wrote."
)


def _outside_output_dir(path: str, output_dir: str) -> str | None:
    """Why `filesystem_write` may not write `path` on this run, or None.

    Resolved rather than matched as text, the way `_resolve_write_path` is, so
    `projects/x/../../nodes.py` is caught and a symlink cannot lead out. Checked
    here rather than in the MCP client because the scope is a property of the
    run, and the client is shared by every run the process serves.
    """
    root = Path.cwd().resolve()
    scope = (root / output_dir).resolve()
    try:
        (root / path).resolve().relative_to(scope)
    except ValueError:
        return (
            f"Refusing to write {path!r}: this run writes only under "
            f"{output_dir}/. Spell the path from the project root, e.g. "
            f"{output_dir}/{Path(path).name or 'main.py'}."
        )
    return None

# How many times the Builder may think-and-call before the node gives up. Each
# turn is a cloud round trip. The default leans on the Architect gate getting
# another cycle anyway -- which holds only when a pass finishes a unit of work.
# On an open-ended goal it does not: the Builder is cut off mid-task every pass,
# reports nothing the gate can approve, and the run circles to its budget. So
# this is overridable, and a broad goal wants it raised.
MAX_BUILDER_TOOL_TURNS = int(os.getenv("MAX_BUILDER_TOOL_TURNS", "8"))

# Wall-clock ceiling for the whole Builder turn, larger than the Researcher's
# because this node legitimately makes many calls: up to MAX_BUILDER_TOOL_TURNS
# round trips, each of which may run tests. The turn cap bounds how many calls
# it makes and says nothing about how long they take -- the same gap that let a
# stalled Researcher hang a run.
BUILDER_DEADLINE_SECONDS = float(os.getenv("BUILDER_DEADLINE_SECONDS", "240"))

# Held back from the tool loop so the verification pass always gets to run.
# Verification is what stops the Builder claiming work it never proved, so
# letting the loop spend the entire budget would trade the guarantee for one
# more tool call. Whatever the loop leaves unused is added to this.
VERIFY_RESERVE_SECONDS = float(os.getenv("VERIFY_RESERVE_SECONDS", "60"))

# A tool result this long is summarised rather than pasted whole. Large reads
# are the reason: a whole file in the transcript crowds out the plan.
MAX_TOOL_RESULT_CHARS = 20000

# How much of a failure's reason its report line carries. One line in the
# tool's or the program's own words is enough to tell a refused pipe from a
# missing file; the whole of it is already in the transcript the Builder reads.
MAX_FAILURE_REASON_CHARS = 120


def _failure_reason(result: Any) -> str:
    """One line saying why a tool call failed, for its report line.

    `-> failed` alone cannot tell a refused pipe from a missing file from a
    broken machine, and the Architect rules on this report: on the run of
    2026-09-10 eleven of forty-seven lines read `failed` with nothing beside
    them, eight of them the tool correctly refusing shell syntax, and the
    operator reading them had no way to see that. The tool's own `error` comes
    first, being written to be read; then the last line on stderr, which is
    where a traceback puts its exception and `cat` its only line; then stdout,
    where pytest puts its summary; then the exit status.
    """
    if not isinstance(result, dict):
        return ""
    line = ""
    error = str(result.get("error") or "").strip()
    if error:
        line = error.splitlines()[0]
    else:
        for stream in ("stderr", "stdout"):
            lines = [ln.strip() for ln in str(result.get(stream) or "").splitlines() if ln.strip()]
            if lines:
                line = lines[-1]
                break
        if not line and result.get("returncode") is not None:
            line = f"exit {result['returncode']}"
    if len(line) > MAX_FAILURE_REASON_CHARS:
        line = line[: MAX_FAILURE_REASON_CHARS - 3].rstrip() + "..."
    return line


def _run_builder_tools(
    llm: Any,
    messages: list[Any],
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline,
    allowed: frozenset[str] | set[str] = BUILDER_TOOL_NAMES,
    output_dir: str = "",
) -> tuple[str, bool, bool, bool]:
    """Let the Builder call tools until it stops asking for them.

    Returns the Builder's closing message, whether it ran out of turns, whether
    it ran out of time, and whether the operator stopped it. `files_changed` is
    appended to only when a write tool reports success, so the list stays a
    record of what happened rather than what was claimed.

    The deadline is enforced in two places, and only one of them may abandon
    work. The model's own call is wrapped, because discarding a half-received
    response costs nothing but the turn. The tool calls underneath it are not:
    they write files, stage commits and run commands, and a worker abandoned
    mid-`filesystem_write` would keep writing into the project after this node
    returned. So each turn's tools always run to completion, and the budget is
    re-checked at the top of the next turn instead.

    The emergency stop obeys the same rule, and is checked in the same place --
    never inside the batch below. Skipping the remaining calls of a batch would
    leave `ToolMessage` replies missing for `tool_call_id`s the model has
    already been told about, which corrupts the message list rather than ending
    cleanly. It is kept apart from the deadline so the report can say which one
    happened: a run the operator stopped must not be described as one that
    exceeded its own budget.
    """
    for _ in range(MAX_BUILDER_TOOL_TURNS):
        if RUN_CONTROL.stopped():
            return "", False, False, True
        if deadline.expired():
            return "", False, True, False

        # None is the sentinel for "gave up"; a real response is never None.
        response = _with_deadline(
            lambda: llm.invoke(messages), deadline.remaining(), None
        )
        if response is None:
            return "", False, True, False

        calls = list(getattr(response, "tool_calls", None) or [])

        if not calls:
            return _as_text(response.content), False, False, False

        messages.append(response)

        for call in calls:
            name = str(call.get("name", ""))
            args = dict(call.get("args") or {})

            if name not in allowed:
                # Refused rather than run: the tool split is the whole point,
                # and a Researcher tool reaching the Builder is a real bug
                # worth surfacing in the report instead of silently serving.
                # Checked here as well as withheld from the offered list,
                # because the two answer different questions: the list is what
                # the model is told about, and this is what actually runs. On a
                # discussion run that difference is the whole guarantee, so it
                # does not rest on the model having read its tool schema.
                why = (
                    "cannot run on a discussion-only run"
                    if name in BUILDER_TOOL_NAMES
                    else "is not a Builder tool"
                )
                result: Any = {"success": False, "error": f"{name} {why}"}
            elif (
                name == "filesystem_write"
                and output_dir
                and (outside := _outside_output_dir(str(args.get("path", "")), output_dir))
            ):
                result = {"success": False, "error": outside}
            else:
                try:
                    result = _call_mcp_tool_sync(name, args)
                except Exception as exc:
                    result = {"success": False, "error": str(exc)}

            ok = bool(result.get("success")) if isinstance(result, dict) else False

            if name == "filesystem_write" and ok:
                path = str(args.get("path", ""))
                if path and path not in files_changed:
                    files_changed.append(path)

            # The Architect rules on this report, so a line has to say enough
            # to be ruled on. `find . -type f -> ok` names no directory, and
            # once the Builder started passing `cwd` the command alone stopped
            # locating anything: the same relative command means a different
            # thing in every directory it could have run in.
            # The same goes for a failure: the line names why, or eight refused
            # pipes read exactly like eight broken commands.
            target = args.get("path") or args.get("command") or ""
            where = f" [cwd={args['cwd']}]" if args.get("cwd") else ""
            outcome = "ok"
            if not ok:
                reason = _failure_reason(result)
                outcome = f"failed: {reason}" if reason else "failed"
            if name == "git_dwell":
                # This one call is a whole pipeline, and `ok` alone hides which
                # part of it ran. A push that failed after a commit succeeded
                # is a different situation from one that never committed, and
                # the Architect rules on this line. So the stages are named,
                # and the argument shown is the stage list rather than a path
                # -- `git_dwell()` would otherwise log an empty target for the
                # most consequential call the Builder can make.
                target = ",".join(str(x) for x in (args.get("stages") or [])) or "default"
                done = [
                    e["stage"] for e in (result.get("stages") or [])
                    if isinstance(e, dict) and e.get("ok")
                ]
                reached = f" [{' -> '.join(done)}]" if done else ""
                outcome = outcome if ok else f"{outcome} at {result.get('stopped_at', '?')}"
                tool_log.append(f"{name}({target}){reached} -> {outcome}")
            else:
                tool_log.append(f"{name}({target}){where} -> {outcome}")

            payload = json.dumps(result, default=str)
            if len(payload) > MAX_TOOL_RESULT_CHARS:
                # Flagged, not silently clipped: a truncated read that the model
                # then writes back would delete the tail of the file.
                payload = (
                    payload[:MAX_TOOL_RESULT_CHARS]
                    + '..."TRUNCATED": "Result cut short. Do not write this '
                    'content back to a file -- it is incomplete."'
                )

            messages.append(
                ToolMessage(content=payload, tool_call_id=str(call.get("id", "")))
            )

    return "", True, False, False


# Files the Builder writes that can be executed as a script. Anything else it
# produces -- markdown, config, data -- has nothing to run.
RUNNABLE_SUFFIXES = (".py",)

# Rules ruff may fix on the Builder's behalf: only those that cannot change what
# the code does -- trailing whitespace, a missing final newline, import order,
# and two spellings of the same object: `datetime.UTC` for `timezone.utc`
# (UP017) and `TimeoutError` for its aliases (UP041), each the identical object
# since 3.11. Those two arrived with ruff's target moving to 3.12, and without
# them a correct file using the older name would block approval over a
# spelling. Everything else ruff reports is the Builder's to fix, because a fix
# ruff marks "safe" can still delete an import that was kept for its side
# effect.
LINT_AUTOFIX_RULES = ("W291", "W292", "W293", "I001", "UP017", "UP041")

# Per-file ceiling for the lint pass. ruff answers in milliseconds; this bounds
# a wedged interpreter, not the linter.
LINT_TIMEOUT_SECONDS = 30

# Findings shown per file in the report. The Builder can run ruff itself for the
# rest; the report only has to say what kind of trouble a file is in.
MAX_LINT_FINDINGS_SHOWN = 8

# Verification runs a file to prove it does not raise, with nobody watching.
# Anything that opens a window waits for a human to close it, so a correct
# script ending in `plt.show()` -- the ordinary way to write a plotting
# example -- burned a full VERIFY_TIMEOUT_SECONDS and came back FAILED. The
# display variables are removed as well as MPLBACKEND being set, because a
# library that checks for a display itself never consults MPLBACKEND.
HEADLESS_VERIFY_ENV: dict[str, str | None] = {
    "MPLBACKEND": "Agg",
    "DISPLAY": None,
    "WAYLAND_DISPLAY": None,
}

# Per-file ceiling for the verification pass. A written script that hangs is a
# failed verification, not a reason to stall the whole run.
VERIFY_TIMEOUT_SECONDS = 60

# Introduces what a hung file printed before it was killed.
_TIMEOUT_OUTPUT_HEADING = "\nOutput before it was killed (tail):\n"

# Verification output kept in the report. Enough for the next cycle to see the
# traceback that matters, not so much that it buries the plan.
MAX_VERIFY_DETAIL_CHARS = 800

# What an import proves, said in the report so the Architect rules on what was
# actually established rather than on "clean" alone.
PACKAGE_MODULE_IMPORT_NOTE = (
    "imported as `{module}`: its syntax, its imports and its module-level code "
    "ran; its __main__ block, if it has one, did not"
)

# The least time worth starting a file in. Below this the remaining slice
# rounds down to a timeout nothing can finish inside, and the file comes back
# FAILED -- which is not merely useless but wrong: it accuses a working file of
# not running, and sets the blocker that says so. Better to admit it was never
# executed.
MIN_VERIFY_SLICE_SECONDS = 1.0

# Why a file was left unrun when the operator stopped the run. Worded apart
# from the deadline reason because the two are not the same event, and the
# report is the only place anyone finds out which one happened.
VERIFY_STOPPED_REASON = (
    "not executed: the run was stopped before this file was reached. It is "
    "unproven, not passing, and is re-checked on the next cycle."
)

# Why a file was left unrun when the Builder's budget ran out.
VERIFY_DEADLINE_SKIP_REASON = (
    "not executed: the Builder's deadline passed before this file was reached. "
    "It is unproven, not passing, and is re-checked on the next cycle."
)

# How each verification status reads in the report. "NOT RUN" is shouted like
# "FAILED" on purpose: an unexecuted file is not a passing one.
_VERIFY_LABELS = {
    "ok": "ran clean",
    "imported": "imported clean",
    "failed": "FAILED",
    "unverified": "NOT RUN",
}

# Why the Builder's last pass ended before it finished, as the gate words it.
# Keyed by `builder_cut_off`.
_CUT_OFF_REASONS = {
    "turn_cap": "the Builder ran out of tool turns before it finished",
    "deadline": "the Builder hit its deadline before it finished",
}


# A Builder often answers the Blockers section with "none" and then keeps
# writing -- "none - Note: this file raises by design". Only the leading token
# is the answer; the rest is commentary, and keeping the whole string made
# state claim something was blocked while literally saying "none".
#
# Matched only where the token stands as a complete clause: followed by the end
# of the string or a separator. "none of the tests pass" continues into a real
# sentence and stays a blocker -- swallowing that would be the silent success
# this module spends its time preventing. The match is deliberately narrow, so
# an unrecognised phrasing ("none needed") is kept as a blocker rather than
# dropped: a spurious blocker costs a cycle, a dropped one costs the guarantee.
_NO_BLOCKER = re.compile(
    r"^\s*(?:none|n/?a|nothing|no\s+blockers?)\s*(?:$|[-\u2014\u2013:;.,])",
    re.IGNORECASE,
)


def _clean_blockers(text: str) -> str:
    """The Blockers section as an actual blocker, or empty when it says none."""
    if not text or _NO_BLOCKER.match(text):
        return ""
    return text.strip()


def _import_target(path: str) -> tuple[str, str] | None:
    """How to verify a file inside a package: (import root, dotted module name).

    `python pkg/mod.py` puts *pkg* on sys.path rather than the directory above
    it, so a module that imports its own package absolutely -- `from pkg.other
    import x`, the normal way to write one -- dies with ModuleNotFoundError no
    matter how correct it is. Such a file used to be skipped, and a skip proves
    nothing: on 2026-09-12 a run whose only products were two package modules
    was approved as "complete and verified" with neither ever executed. So it
    is imported instead, the way its callers use it -- `import pkg.mod`, with
    the directory above the outermost package on the path.

    The walk up stops at the first directory without an `__init__.py`, the test
    Python itself uses to decide what a package is. An `__init__.py` is the
    package itself, so it is imported by the package's name.

    Returns None for a file that is not inside a package, and for one whose
    package path is not a valid module name (a directory called `my-pkg` cannot
    be imported by anyone): both are executed as scripts.
    """
    file = Path(path).resolve()
    directory = file.parent
    if not (directory / "__init__.py").exists():
        return None
    parts = [] if file.name == "__init__.py" else [file.stem]
    while (directory / "__init__.py").exists() and directory.parent != directory:
        parts.insert(0, directory.name)
        directory = directory.parent
    if not all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        return None
    return str(directory), ".".join(parts)


def _timeout_detail(result: dict[str, Any]) -> str:
    """Describe a timed-out verification with the output it produced.

    The timeout message alone says a file hung but not where, which is the
    difference between a script that blocked on its first line and one that
    did all its work and then waited at `plt.show()`. Without it the Builder
    guesses -- it read a bare timeout as a missing dependency once, installed
    a package that was already there, and spent a second full timeout on an
    identical retry. The tail is kept rather than the head: what a hung
    process printed last is how far it got.
    """
    message = str(result.get("error") or "timed out").strip()
    printed = str(result.get("stdout") or "").strip() or str(result.get("stderr") or "").strip()
    if not printed:
        return message[:MAX_VERIFY_DETAIL_CHARS]

    room = MAX_VERIFY_DETAIL_CHARS - len(message) - len(_TIMEOUT_OUTPUT_HEADING)
    if room <= 0:
        return message[:MAX_VERIFY_DETAIL_CHARS]

    tail = printed[-room:]
    if len(tail) < len(printed):
        tail = "..." + tail[3:]
    return f"{message}{_TIMEOUT_OUTPUT_HEADING}{tail}"


def _verify_written_files(
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline | None = None,
) -> list[tuple[str, str, str]]:
    """Execute the runnable files the Builder wrote and report what happened.

    Writing a file is not evidence that it works. The Builder previously
    reported "Implementation complete" for a module it had never executed, and
    the Architect approved it -- the file raised an AssertionError the first
    time anyone ran it. Running it here means a broken file comes back as a
    blocker the loop can act on, rather than as a success nobody checked.

    A file inside a package is imported rather than executed -- see
    `_import_target` -- and reads "imported", which clears exactly as "ok" does.
    Both run under this process's own interpreter rather than whatever `python`
    is first on PATH: a launcher without the virtualenv active found an
    interpreter with none of the project's dependencies, and every file that
    imported one would have been reported broken.

    Each file is bounded by VERIFY_TIMEOUT_SECONDS, but the number of files is
    not, so `deadline` bounds the pass as a whole. Files past it come back
    "unverified" rather than "ok": treating an unrun file as passing is the
    exact false clearance this pass exists to prevent, and the caller keeps
    them in `failed_verification` so the next cycle re-runs them. The emergency
    stop lands in the same place and with the same status, for the same reason.

    Returns one (path, status, detail) per runnable file, where status is
    "ok", "imported", "failed" or "unverified".
    """
    results: list[tuple[str, str, str]] = []

    for path in files_changed:
        if not path.endswith(RUNNABLE_SUFFIXES):
            continue

        if RUN_CONTROL.stopped():
            # `unverified`: nobody ran this file, which is
            # exactly the gap this pass exists to surface. It keeps blocking
            # approval even under `expect_failures`, because that opt-out is
            # for a file the run meant to fail -- still executed, still
            # reported -- not for one that was never executed at all.
            results.append((path, "unverified", VERIFY_STOPPED_REASON))
            tool_log.append(f"verify({path}) -> not run (stopped)")
            continue

        if deadline is not None and deadline.remaining() < MIN_VERIFY_SLICE_SECONDS:
            results.append((path, "unverified", VERIFY_DEADLINE_SKIP_REASON))
            tool_log.append(f"verify({path}) -> not run (deadline)")
            continue

        target = _import_target(path)
        env: dict[str, str | None] = dict(HEADLESS_VERIFY_ENV)
        # Quoted because there is no shell to split on: a path with a space in
        # it reached the interpreter as two arguments.
        python = shlex.quote(sys.executable)
        if target is None:
            command, passed = f"{python} {shlex.quote(path)}", "ok"
        else:
            root, module = target
            command, passed = f'{python} -c "import {module}"', "imported"
            inherited = os.environ.get("PYTHONPATH")
            env["PYTHONPATH"] = root + (os.pathsep + inherited if inherited else "")

        try:
            result = _call_mcp_tool_sync(
                "terminal_execute",
                {
                    "command": command,
                    # Never let one file overrun what is left for the rest, and
                    # never hand it a slice too small to run in -- see
                    # MIN_VERIFY_SLICE_SECONDS.
                    "timeout": (
                        VERIFY_TIMEOUT_SECONDS
                        if deadline is None
                        else max(1, int(min(VERIFY_TIMEOUT_SECONDS, deadline.remaining())))
                    ),
                    "env": env,
                },
            )
        except Exception as exc:
            results.append((path, "failed", str(exc)))
            tool_log.append(f"verify({path}) -> failed")
            continue

        ok = bool(result.get("success")) if isinstance(result, dict) else False
        detail = ""
        if isinstance(result, dict) and not ok:
            # stderr first: a traceback is what the next cycle needs to see.
            detail = str(
                result.get("stderr") or result.get("error") or result.get("stdout") or ""
            ).strip()[:MAX_VERIFY_DETAIL_CHARS]
            if result.get("timed_out"):
                detail = _timeout_detail(result)
        elif ok and target is not None:
            detail = PACKAGE_MODULE_IMPORT_NOTE.format(module=target[1])

        status = passed if ok else "failed"
        results.append((path, status, detail))
        tool_log.append(f"verify({path}) -> {status}")

    return results


def _lint_written_files(
    paths: list[str],
    tool_log: list[str],
    deadline: _Deadline | None = None,
) -> tuple[list[tuple[str, list[tuple[int, str, str]]]], list[str], str]:
    """Lint the Python the Builder wrote, the way CI will.

    Running a file proves it does not raise; it says nothing about whether CI
    accepts it. The module an agent run wrote on 2026-09-12 imported, ran and
    was approved -- carrying 103 ruff errors, so its first push would have
    failed. This runs `ruff check` over each Python file with the project's own
    configuration, applies only `LINT_AUTOFIX_RULES`, and reports what is left.

    Returns (failures, fixed, unavailable): each path with the findings left
    after the fixes, as (line, code, message); the paths ruff tidied; and why
    ruff could not run at all, or "" when it did. A linter that is missing or
    broken is not a defect in the file, so it is reported rather than failed,
    and the pass stops at the first sign of it -- every file would say the same.
    """
    failures: list[tuple[str, list[tuple[int, str, str]]]] = []
    fixed: list[str] = []
    python = shlex.quote(sys.executable)
    rules = ",".join(LINT_AUTOFIX_RULES)

    for path in paths:
        if not path.endswith(".py"):
            continue
        # The stop and the deadline leave lint unrun. The verification pass
        # that follows reports the same files as unproven, which already blocks.
        if RUN_CONTROL.stopped() or (
            deadline is not None and deadline.remaining() < MIN_VERIFY_SLICE_SECONDS
        ):
            tool_log.append(f"lint({path}) -> not run")
            continue
        try:
            before = Path(path).read_bytes()
        except OSError:
            continue  # gone from disk: the verification pass accounts for it

        try:
            result = _call_mcp_tool_sync(
                "terminal_execute",
                {
                    "command": (
                        f"{python} -m ruff check --no-cache --fix --fixable {rules} "
                        f"--output-format json {shlex.quote(path)}"
                    ),
                    "timeout": (
                        LINT_TIMEOUT_SECONDS
                        if deadline is None
                        else max(1, int(min(LINT_TIMEOUT_SECONDS, deadline.remaining())))
                    ),
                    "env": {"NO_COLOR": "1"},
                },
            )
        except Exception as exc:
            tool_log.append(f"lint({path}) -> not run")
            return failures, fixed, f"the lint pass could not start: {exc}"

        output = result if isinstance(result, dict) else {}
        try:
            findings = json.loads(str(output.get("stdout") or ""))
        except json.JSONDecodeError:
            findings = None
        if not isinstance(findings, list):
            stderr = str(output.get("stderr") or output.get("error") or "").strip()
            tool_log.append(f"lint({path}) -> not run")
            if "No module named ruff" in stderr:
                return failures, fixed, "ruff is not installed for this interpreter"
            if output.get("timed_out"):
                return failures, fixed, f"ruff timed out on {path}"
            last = stderr.splitlines()[-1] if stderr else "no readable answer"
            return failures, fixed, f"ruff could not check {path}: {last}"

        try:
            if Path(path).read_bytes() != before:
                fixed.append(path)
        except OSError:
            pass

        left = [
            (
                int((finding.get("location") or {}).get("row") or 0),
                str(finding.get("code") or finding.get("name") or "syntax"),
                str(finding.get("message") or "").strip(),
            )
            for finding in findings
            if isinstance(finding, dict)
        ]
        if left:
            failures.append((path, left))
        tool_log.append(f"lint({path}) -> {'failed' if left else 'clean'}")

    return failures, fixed, ""


def builder_node(state: AgentState) -> AgentState:
    """Builder: implement the plan by actually calling tools.

    As specified:
    - Filesystem, git, terminal and test tools only (no GraphRAG)
    - Actually make changes via tools
    - Report in strict format
    - Set blockers if stuck

    The model drives the tools. An earlier version regex-scraped the plan for
    `create <file>` plus a quoted string and wrote that, which meant it could
    only ever create whole new files from a plan phrased just so -- every other
    goal, including editing an existing file, reported "Implementation
    complete" having changed nothing.
    """
    state_injection = _get_state_injection(state)
    plan = state.get("plan", "")
    research = state.get("research", "")

    files_changed: list[str] = []
    tool_log: list[str] = []
    exhausted = False
    out_of_time = False
    stopped = False

    # The loop is held to the budget minus the verification reserve; whatever it
    # does not spend is handed on below, so a quick build still gets a long
    # verification pass and a slow one cannot starve it entirely.
    loop_deadline = _Deadline(
        max(0.0, BUILDER_DEADLINE_SECONDS - VERIFY_RESERVE_SECONDS)
    )

    # Set by the caller and never by an agent, like `expect_failures`. A seat
    # cannot vote itself the right to act.
    discuss_only = bool(state.get("discuss_only"))
    allowed = DISCUSSION_TOOL_NAMES if discuss_only else BUILDER_TOOL_NAMES
    # Also the caller's: where this run may write. Empty means this checkout.
    output_dir = "" if discuss_only else str(state.get("output_dir") or "")
    note = (
        DISCUSSION_NOTE
        if discuss_only
        else OUTPUT_DIR_NOTE.format(output_dir=output_dir) if output_dir else ""
    )

    messages: list[Any] = [
        SystemMessage(content=BUILDER_PROMPT + note),
        HumanMessage(
            content=f"{state_injection}\n\nPlan to implement:\n{plan}\n\n"
            f"Research findings:\n{research}"
        ),
    ]

    llm = get_agent_llm("builder")
    try:
        # Nothing is bound on a discussion run, which puts this node on the
        # no-tool path below -- the same one a seat whose model cannot call
        # tools already takes. There is then no tool loop at all, so "took no
        # action" is a property of the code that ran rather than of a filter
        # having been complete.
        tool_llm = None if discuss_only else llm.bind_tools(BUILDER_TOOLS)
    except AttributeError:
        # A seat whose model cannot call tools at all -- StubLLM, or a tag
        # without tool support. It still reports; it just cannot change a file.
        tool_llm = None

    if RUN_CONTROL.stopped():
        # Nothing new is started once the stop is in -- no model call, no
        # tools. The verification pass below still runs, and its own stop check
        # brings back every carried file as unproven rather than clear, which
        # is what keeps a stopped run from looking finished.
        content, stopped = "", True
    elif tool_llm is None:
        reply = _with_deadline(
            lambda: _as_text(llm.invoke(messages).content),
            loop_deadline.remaining(),
            None,
        )
        out_of_time = reply is None
        content = reply or ""
    else:
        content, exhausted, out_of_time, stopped = _run_builder_tools(
            tool_llm, messages, files_changed, tool_log, loop_deadline, allowed,
            output_dir,
        )

    # Every runnable file the Builder wrote is executed before it gets to claim
    # the work is done. This is in code rather than left to the prompt for the
    # same reason files_changed is: the Builder's own account of its work is
    # not evidence.
    #
    # Files that failed on an earlier pass are re-checked even when this pass
    # did not touch them. Verifying only what was just written let the Builder
    # clear a failure by doing nothing: the broken file stayed on disk, the
    # next pass wrote nothing, the failure list came back empty and the gate
    # approved. A file clears only by running clean.
    # A carried file that is gone from disk drops out instead of failing.
    # Deleting it is a real fix -- a file that does not exist cannot raise --
    # but `python <missing path>` exits non-zero forever, so re-running it
    # pinned failed_verification open and the gate rewrote every `approved` to
    # `revise` until the step ceiling ended the run. Only carried paths get
    # this; a path in files_changed was just written by a tool that reported
    # success, and its absence would be the write lying.
    carried = [
        path
        for path in (state.get("failed_verification") or [])
        if path not in files_changed and Path(path).exists()
    ]
    # The reserve plus whatever the tool loop left unspent.
    verify_deadline = _Deadline(VERIFY_RESERVE_SECONDS + loop_deadline.remaining())
    # Lint first, so what is executed below is the file after ruff's whitespace
    # and import fixes. A file that failed lint on an earlier pass is re-linted
    # though this pass left it alone, on the rule that keeps a failed file on
    # `failed_verification`: it clears by passing, never by omission.
    lint_carried = [
        path
        for path in (state.get("lint_failed") or [])
        if path not in files_changed and path not in carried and Path(path).exists()
    ]
    lint_failures, lint_fixed, lint_unavailable = _lint_written_files(
        files_changed + carried + lint_carried, tool_log, verify_deadline
    )
    lint_failed = [path for path, _ in lint_failures]
    verification = _verify_written_files(
        files_changed + carried, tool_log, verify_deadline
    )
    failed = [
        (path, detail) for path, status, detail in verification if status == "failed"
    ]
    unverified = [path for path, status, _ in verification if status == "unverified"]

    parsed = _parse_builder_output(content)
    # A loop cut off at the turn cap has no closing message to parse, and "No
    # report produced." read as a Builder that did nothing -- above a report
    # listing every tool call it made.
    builder_report = parsed.get("changes_made") or content or (
        f"No closing report: the Builder used all {MAX_BUILDER_TOOL_TURNS} tool "
        "turns without writing one."
        if exhausted
        else "No report produced."
    )

    if verification:
        builder_report += (
            "\n\nVerification (each runnable file was executed, and each package "
            "module imported, except as noted):\n"
        )
        # fall through to the per-file lines below
        builder_report += "\n".join(
            f"- {path}: {_VERIFY_LABELS[status]}" + (f"\n{detail}" if detail else "")
            for path, status, detail in verification
        )
    if unverified:
        why = (
            "the run was stopped"
            if stopped
            else f"the Builder ran out of its {int(BUILDER_DEADLINE_SECONDS)}s budget"
        )
        builder_report += (
            f"\n\n{len(unverified)} file(s) were not executed: {why}. They are "
            "unproven rather than working, and are re-checked next cycle."
        )

    if lint_failures or lint_fixed or lint_unavailable:
        builder_report += "\n\nLint (ruff, with this project's configuration):"
        if lint_unavailable:
            builder_report += f"\n- not run: {lint_unavailable}"
        for path, findings in lint_failures:
            builder_report += f"\n- {path}: {len(findings)} error(s)"
            builder_report += "".join(
                f"\n  line {line}: {code} {message}"
                for line, code, message in findings[:MAX_LINT_FINDINGS_SHOWN]
            )
            if len(findings) > MAX_LINT_FINDINGS_SHOWN:
                builder_report += (
                    f"\n  ...and {len(findings) - MAX_LINT_FINDINGS_SHOWN} more; "
                    f"`python -m ruff check {path}` lists them all"
                )
        if lint_fixed:
            builder_report += (
                "\n- whitespace, import order and alias spellings fixed automatically in: "
                + ", ".join(lint_fixed)
            )

    if tool_log:
        builder_report += "\n\nTool calls:\n" + "\n".join(f"- {c}" for c in tool_log)

    # `files_changed` above is what *this* pass wrote, and the verification
    # logic needs it to stay that way: `carried` leans on it to tell a path
    # this pass rewrote from one only an earlier pass touched. The state field
    # answers a different question -- what the whole run produced -- so it
    # accumulates. Overwriting it per pass meant a run that wrote a file on one
    # cycle and nothing on the next ended reporting it had changed nothing
    # while the file sat on disk, and the Architect ruled on that empty record:
    # a build with a file to its name was approved as having produced none.
    # That is the same false account as a Builder claiming a file it never
    # wrote, pointing the other way.
    previously_changed = list(state.get("files_changed") or [])
    all_files_changed = previously_changed + [
        path for path in files_changed if path not in previously_changed
    ]

    # files_changed is deliberately NOT taken from the model's prose. A file
    # counts as changed only when a write tool reported success for it; a model
    # that describes writing a module it never wrote would otherwise have the
    # console report "changed this machine" for work that never touched disk.
    # Checked against the run's whole record rather than this pass: a file an
    # earlier pass wrote did come from a successful write call, so naming it
    # again is not a claim about work that never happened.
    # Both sides go through the same normalizer before they are compared. The
    # tool side needs it as much as the prose side: `filesystem_write` records
    # whatever argument the model passed, so a pass that wrote `./foo.py` and a
    # report naming `foo.py` are the same file and must not read as a lie.
    written = {_report_path_key(path) for path in all_files_changed}
    claimed = [path for path in parsed.get("files_modified", []) if path not in written]
    if claimed:
        builder_report += (
            "\n\nDescribed but not written (no successful write call): "
            + ", ".join(claimed)
        )

    # A path the run wrote and something later removed is not a file on disk,
    # and `files_changed` is read as though every entry were one -- by the
    # console's "changed this machine" notice, and by the Architect through the
    # state injection block. The run of 2026-09-11 ended naming four paths of
    # which three did not exist: `gen_overview.py` and two under `/tmp`,
    # written on one pass and deleted with `rm` on a later one, with nothing
    # retracting them. That is the mirror of "described but not written" -- the
    # report's harshest claim, which this function guards in the other
    # direction a dozen lines above -- and it was unguarded.
    #
    # Dropped from the record rather than carried, because the record's readers
    # take it for what is on disk now: a run whose every product was deleted
    # has produced nothing, and reporting that plainly is the accurate account,
    # not the empty one CLAUDE.md warns about. Named in the report rather than
    # dropped silently, for the reason `_research_snippet` announces a cut --
    # an entry that simply vanishes is indistinguishable from one that was
    # never made, and the report is where the Architect finds out.
    #
    # This runs *after* `written`, deliberately. `written` is what answers the
    # "described but not written" accusation, and it has to keep seeing the
    # whole record: a file this pass wrote and then removed did come from a
    # successful write call, and naming it in the report is not a lie.
    removed_paths = [path for path in all_files_changed if not Path(path).exists()]
    if removed_paths:
        all_files_changed = [
            path for path in all_files_changed if path not in removed_paths
        ]
        builder_report += (
            "\n\nWritten earlier and no longer on disk: "
            + ", ".join(removed_paths)
            + ". Dropped from the run's file record, which names what is on "
            "disk now."
        )

    # A discussion run's Builder has no tools and nothing to be blocked on, so
    # what it writes under Blockers is part of its proposal. On the run of
    # 2026-09-12 that read "This proposal requires approval from the Architect
    # to proceed to implementation", which then sat in `blockers` and in the
    # recovery block as though the run were stuck. It stays in the report.
    blockers = _clean_blockers(parsed.get("next_steps_blockers", ""))
    if discuss_only and blockers:
        builder_report += f"\n\nWhat the proposal says it would need:\n{blockers}"
        blockers = ""

    # A failed verification outranks whatever the model concluded: it wrote a
    # file that does not run, and the Architect must see that as unfinished --
    # unless the run was asked for one, in which case it is the product, not a
    # defect. It is still executed and still reported either way.
    expected = bool(state.get("expect_failures"))
    if failed and not expected:
        blockers = "Files that do not run: " + "; ".join(
            f"{path} ({detail.splitlines()[-1] if detail else 'no output'})"
            for path, detail in failed
        )

    # An unproven file blocks whatever `expect_failures` says. That opt-out is
    # for a file the run meant to fail -- it is still executed and still
    # reported -- not for one that was never executed at all, which is a gap in
    # the evidence rather than an expected result.
    if unverified:
        lead = "Not executed before the stop" if stopped else "Not executed before the deadline"
        note = f"{lead}: " + ", ".join(unverified)
        blockers = f"{blockers}. {note}" if blockers else note

    # Whatever `expect_failures` says, like an unrun file: CI lints a fixture
    # that is meant to fail at runtime all the same.
    if lint_failures:
        note = "Files that fail lint: " + "; ".join(
            f"{path} ({', '.join(sorted({code for _, code, _ in findings}))})"
            for path, findings in lint_failures
        )
        blockers = f"{blockers}. {note}" if blockers else note

    if stopped and not blockers:
        blockers = (
            "Stopped by the emergency stop before the Builder finished. "
            "Anything it had already written is kept; anything it did not get "
            "to run is listed above as unproven."
        )

    if out_of_time and not blockers:
        blockers = (
            f"Builder stopped at its {int(BUILDER_DEADLINE_SECONDS)}s deadline "
            "without finishing. Anything it had already written is kept and "
            "verified; narrow the plan or raise BUILDER_DEADLINE_SECONDS."
        )

    if exhausted and not blockers:
        blockers = (
            f"Builder stopped after {MAX_BUILDER_TOOL_TURNS} tool turns without "
            "finishing. Narrow the plan or split it into smaller steps."
        )

    state["builder_report"] = builder_report
    state["files_changed"] = all_files_changed
    # Written every pass, including empty, so a file that gets fixed on a later
    # cycle stops blocking the gate. Unverified paths ride along so the next
    # cycle re-runs them: a file nobody executed must not clear by omission,
    # which is the same rule that keeps a failed file on the list.
    state["failed_verification"] = [path for path, _ in failed] + unverified
    # The gate reads these two apart from the list above. `unverified` blocks
    # approval whatever `expect_failures` says, and the gate could not honour
    # that while it had one list: it cleared the whole of failed_verification
    # under the opt-out, unrun files included. A pass cut off before it
    # finished blocks too -- its blocker always said so, and on 2026-09-12 a
    # run was approved over one.
    state["unverified"] = list(unverified)
    state["builder_cut_off"] = (
        "turn_cap" if exhausted else "deadline" if out_of_time else ""
    )
    state["lint_failed"] = lint_failed
    state["blockers"] = blockers
    # The feed line has to carry the verification result too. "Implementation
    # complete" beside a file that does not run is the same false claim this
    # pass exists to catch, and it is what the Architect reads in state.
    if failed or lint_failed:
        problems = []
        if failed:
            suffix = " (expected for this run)" if expected else ""
            problems.append(f"{len(failed)} do not run{suffix}")
        if lint_failed:
            problems.append(f"{len(lint_failed)} fail lint")
        summary = f"Wrote {len(files_changed)} file(s); " + "; ".join(problems)
    elif unverified:
        why = "stopped" if stopped else "deadline"
        summary = (
            f"Wrote {len(files_changed)} file(s); {len(unverified)} not run "
            f"({why})"
        )
    elif stopped:
        # Never "Implementation complete", for the same reason a deadline-cut
        # Builder never says it: the node was cut off, and the feed line is
        # what the Architect and the operator read.
        summary = f"Stopped by the operator. Files: {len(files_changed)}"
    elif out_of_time:
        # Never "Implementation complete": the node was cut off, and the feed
        # saying otherwise is the same false claim the verification pass exists
        # to catch.
        summary = (
            f"Stopped at the {int(BUILDER_DEADLINE_SECONDS)}s deadline. "
            f"Files: {len(files_changed)}"
        )
    elif exhausted:
        # Never "Implementation complete": the loop ran out of turns with the
        # model still asking for tools, so the Builder did not finish. The
        # blocker above says so, and this line used to contradict it -- on
        # 2026-09-12 a Builder stopped at the turn cap read "Implementation
        # complete. Files: 2" in the feed, and the Architect approved the run.
        summary = (
            f"Stopped after {MAX_BUILDER_TOOL_TURNS} tool turns without "
            f"finishing. Files: {len(files_changed)}"
        )
    elif discuss_only:
        # Never "Implementation complete" -- nothing was implemented, and the
        # Architect rules on this line. The same rule the stop and the deadline
        # already follow: a pass that could not act must not read as one that
        # acted.
        summary = "Discussion only: proposal ready, nothing was changed"
    else:
        summary = f"Implementation complete. Files: {len(files_changed)}"
    # Every count above is this pass, which is what just happened and so what
    # the feed should say. But a pass that wrote nothing, on a run that has
    # already produced files, would read as though the run had lost them -- so
    # the run total is named whenever it differs from the pass.
    if len(all_files_changed) > len(files_changed):
        summary += f" ({len(all_files_changed)} changed so far this run)"
    state["messages"].append(f"[Builder] {summary}")

    # step_count is incremented by the Architect gate, not here: every cycle
    # passes through the gate, but a Planner/Researcher loop never reaches the
    # Builder, and counting here let those loops run uncounted.

    return state
