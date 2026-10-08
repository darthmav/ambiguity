"""The four seats as LangGraph nodes: Architect, Planner, Researcher, Builder.

- Architect: no tools. Sets the direction, then holds the approval gate.
- Planner: no tools. Turns the goal into steps and picks the next seat.
- Researcher: the node runs the read-only GraphRAG tools and hands the seat
  what they returned.
- Builder: filesystem, git, terminal and test tools, and nothing else.

No node takes a seat's account of its own work on trust: verdicts, plans and
research are parsed from fixed sections, and what the Builder wrote is linted
and run before anyone rules on it.
"""

import functools
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
from langgraph_agent.control import RUN_CONTROL, abandonable
from langgraph_agent.dwell import (
    DWELL_CHECKS_WAIT_SECONDS,
    DWELL_DEFAULT_STAGES,
    SHIPPING_STAGES,
)
from langgraph_agent.mcp_client import (
    TERMINAL_TIMEOUT_MAX_SECONDS,
    TERMINAL_TIMEOUT_SECONDS,
    MCPClient,
)
from langgraph_agent.state import AgentState, ResearchStatus, Verdict


def _call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Run one tool. Every node goes through here, so a test fakes a tool in one place."""
    return MCPClient().call_tool(tool_name, arguments)


_T = TypeVar("_T")

# How long one node may spend on its answer. `LLM_TIMEOUT_SECONDS` bounds the
# socket: it catches a connection gone quiet, not a model streaming slowly
# without end, nor a node whose several calls each finish just inside it.
NODE_DEADLINE_SECONDS = float(os.getenv("NODE_DEADLINE_SECONDS", "180"))

# How many retrieved passages reach the Builder, and how much of each. One
# number feeds the search and the slice, so the two cannot drift apart. A chunk
# is chosen because it matched somewhere inside it, so the cap clears the p99
# chunk (about 1,400 characters): a passage arrives whole, and the cap only
# guards against a pathological one.
RESEARCH_RESULTS = int(os.getenv("RESEARCH_RESULTS", "5"))
RESEARCH_SNIPPET_CHARS = int(os.getenv("RESEARCH_SNIPPET_CHARS", "1500"))


class _Deadline:
    """A countdown shared by the calls one node makes, so the node is bounded as a whole."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def _with_deadline(work: Callable[[], _T], seconds: float, fallback: _T) -> _T:
    """Run `work`, giving up on it after `seconds` and returning `fallback`.

    Python cannot cancel a thread, so an abandoned worker is told instead
    (`control.abandoned`): a streamed seat call stops at its next token and
    lets go of the cards, and one blocked on a silent socket unwinds when the
    client timeout fires. Hence `work` must not write to state -- a late
    finisher would land in a state the graph had moved past -- and the worker
    is a daemon thread: a pool's threads are joined at exit, so one stuck
    worker would hold up shutdown.
    """
    box: list[Any] = []
    error: list[BaseException] = []
    given_up = threading.Event()

    def _run() -> None:
        try:
            with abandonable(given_up):
                box.append(work())
        except BaseException as exc:  # re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=_run, daemon=True, name="node-deadline")
    thread.start()
    thread.join(timeout=seconds)

    if thread.is_alive():
        given_up.set()
        return fallback
    if error:
        # A seat that failed outright is not a timeout; let it raise so
        # `_SeatLLM` records the reason and the console can show it.
        raise error[0]
    return cast("_T", box[0])


# The seats' system prompts, one file each under the project root's prompts/.
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@functools.cache
def seat_prompt(name: str) -> str:
    """A seat's system prompt, read the first time the seat needs it.

    Never at import: importing the package must not need the checkout's data
    files, and the image's build stage imports it before prompts/ is copied
    in. There is no fallback copy to drift from the file.
    """
    try:
        return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"The {name} prompt is missing ({exc}). The seats' prompts live in "
            f"{PROMPTS_DIR}; run from a checkout of the project."
        ) from exc


def _get_state_injection(state: AgentState) -> str:
    """The state block every seat is sent each turn; an empty field reads `(empty)`."""

    def _fmt(key: str, default: str = "(empty)") -> str:
        value = state.get(key)
        if value is None or value == "" or value == []:
            return default
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    # Every seat is told, the Architect above all: on a discussion run there is
    # no work to find, and a gate judging a proposal by build standards is
    # never satisfied.
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
Pull request: {_dwell_summary(state.get("dwell")) or "(empty)"}
Step: {state.get("step_count", 0)}
"""


def _dwell_summary(record: dict[str, Any] | None) -> str:
    """One line on where the run's git work stands, from its `dwell` record; "" for none."""
    if not record:
        return ""
    status = record.get("status")
    pr = f"#{record['number']}" if record.get("number") else "the pull request"
    url = f" ({record['url']})" if record.get("url") else ""
    branch = record.get("branch") or "its branch"
    if status == "merged":
        landed = f" as {str(record['merge_commit'])[:7]}" if record.get("merge_commit") else ""
        return f"{pr}{url} merged into {record.get('default_branch') or 'the default'}{landed}"
    if status == "pending":
        return f"{pr}{url} open, waiting: {record.get('pending')}"
    if status == "checks_failed":
        failing = ", ".join(record.get("checks_failed") or []) or "a check"
        return f"{pr}{url} open, and its checks fail: {failing}"
    if status == "open":
        return f"{pr}{url} open"
    if status == "local":
        return f"committed on {branch}; there is no remote, so nothing was pushed"
    if status == "committed":
        return f"committed on {branch}"
    reason = str(record.get("error") or "").splitlines()[0] if record.get("error") else ""
    return f"git_dwell stopped at {record.get('stopped_at') or '?'}" + (
        f": {reason}" if reason else ""
    )


# How much of a stopped dwell's error the record keeps: its first lines say
# which stage and why; the tool result already carried the rest.
MAX_DWELL_ERROR_CHARS = 500


def _dwell_record(result: dict[str, Any]) -> dict[str, Any]:
    """What the run's git work is and where it stands, from one `git_dwell` result.

    `status` is what the gate and the console read: "merged", "pending" (open,
    waiting on its checks or a review -- the console finishes it), "checks_failed"
    (blocks approval), "open", "committed", "local" (no remote), or "failed".
    """
    pr = result.get("pull_request") or {}
    if result.get("merged"):
        status = "merged"
    elif result.get("checks_failed"):
        status = "checks_failed"
    elif not result.get("success"):
        status = "failed"
    elif result.get("pending"):
        status = "pending"
    elif result.get("local_only"):
        status = "local"
    elif pr.get("number") or pr.get("url"):
        status = "open"
    else:
        status = "committed"
    return {
        "status": status,
        "cwd": str(result.get("cwd") or ""),
        "branch": str(result.get("branch") or ""),
        "default_branch": str(result.get("default_branch") or ""),
        "number": pr.get("number"),
        "url": str(pr.get("url") or ""),
        "head": str(result.get("head") or ""),
        "pending": str(result.get("pending") or ""),
        "checks_failed": [str(name) for name in (result.get("checks_failed") or [])],
        "merge_commit": str(result.get("merge_commit") or ""),
        "stopped_at": str(result.get("stopped_at") or ""),
        "error": str(result.get("error") or "")[:MAX_DWELL_ERROR_CHARS],
    }



# What may stand between a one-word section's heading and its word: a colon on
# the heading's own line (`## Verdict: revise`), a list marker, and the
# emphasis a model puts on a lone word (`**revise**`). Each used to read as no
# answer, which fell through to the section's fallback -- for the gate,
# `approved`.
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

    goal_match = re.search(r"## Goal\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if goal_match:
        result["goal"] = goal_match.group(1).strip()

    steps_match = re.search(r"## Steps\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if steps_match:
        result["plan"] = steps_match.group(1).strip()

    # Capitalised, so state holds the one spelling AgentState documents.
    agent_match = re.search(
        rf"## Next Agent{_SECTION_VALUE_LEAD}(Researcher|Builder)\b", content, re.IGNORECASE
    )
    if agent_match:
        result["next_agent"] = agent_match.group(1).strip().capitalize()

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


# A `## Files Modified` line is prose, and `files_changed` holds raw
# `filesystem_write` arguments. Every decoration a model puts on a path it did
# write -- a list marker, backticks, bold, `./`, an absolute path, a `(new
# file)` note -- would otherwise read as "Described but not written".
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MARKDOWN_WRAP = re.compile(r"^(?:\*\*|__|`|\*|_)+|(?:\*\*|__|`|\*|_)+$")
# A whitelist, not "any trailing parenthetical": real filenames carry
# parentheses too (`band_pass_(40_60_hz).png`).
_PATH_ANNOTATION = re.compile(
    r"\s+\((?:new|newly|created|create|added|modified|updated|edited|changed|"
    r"rewritten|rewrote|overwritten|existing|unchanged|deleted|removed)\b[^()]*\)$",
    re.IGNORECASE,
)
# Lines that answer "nothing" rather than name a path.
_NOT_A_PATH = {"", ".", "-", "none", "n/a", "na", "(none)", "nothing", "no files"}


def _report_path_key(text: str) -> str:
    """One `## Files Modified` entry, normalized to compare with the tool records.

    Returns "" for a line that names no file. Errs toward stripping: stripping
    too much can only hide an accusation, too little invents one, and an
    invented one reaches the Architect as evidence.
    """
    path = _LIST_MARKER.sub("", text).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip()
    path = _PATH_ANNOTATION.sub("", path).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip().rstrip(":,;").strip()

    if not path or path.lower() in _NOT_A_PATH:
        return ""
    # A sentence, not a filename.
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

    ## Blockers
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
        # Normalized: see `_report_path_key`.
        seen: set[str] = set()
        paths: list[str] = []
        for line in files_match.group(1).strip().split("\n"):
            key = _report_path_key(line)
            if key and key not in seen:
                seen.add(key)
                paths.append(key)
        result["files_modified"] = paths

    # "## Blockers" is what the prompt asks for; the older heading still
    # parses.
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
        SystemMessage(content=seat_prompt("architect")),
        HumanMessage(content=f"{state_injection}\n\nUser goal: {goal}\n\n{task}"),
    ]

    llm = get_agent_llm("architect")
    response = llm.invoke(messages)
    return _parse_architect_output(_as_text(response.content), reviewing=reviewing)


def architect_node(state: AgentState) -> AgentState:
    """Architect: set direction, then rule on whether the work is done.

    No tools. Runs as the entry authority before the Planner and as the
    approval gate after the Builder reports. Bounded by `NODE_DEADLINE_SECONDS`;
    a gate that timed out or was stopped never rules `approved`, because this
    gate is what ends the run.
    """
    reviewing = bool(state.get("builder_report"))

    if RUN_CONTROL.stopped():
        # Nothing was ruled, so nothing is approved. No step is counted -- this
        # pass did no work -- and the architecture in state is kept.
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

    # The opening pass has one ruling, `plan`: nothing exists yet to approve,
    # and `need_research` would send the Researcher an empty plan, round a loop
    # the step count never sees. What the seat said is kept in the feed line.
    opening = not state.get("plan")
    answered = parsed["verdict"]
    if opening and answered != Verdict.PLAN.value:
        parsed["verdict"] = Verdict.PLAN.value

    # Keep the opening architecture if the gate pass did not restate it --
    # losing it mid-run would strip the constraints out of every later prompt.
    if parsed["architecture"]:
        state["architecture"] = parsed["architecture"]
    state["verdict"] = parsed["verdict"]

    # Work without evidence is not approved, whatever the Architect concluded
    # -- the one place its ruling is overridden, rewritten in the verdict
    # itself so state says what happened. Blocking: a file that ran and failed
    # (unless the run expects failures), a file nobody ran, a lint failure, a
    # Builder pass cut off before it finished, and a pull request whose checks
    # fail.
    unverified = list(state.get("unverified") or [])
    failed_files = [
        path for path in (state.get("failed_verification") or []) if path not in unverified
    ]
    if state.get("expect_failures"):
        # The caller asked for failing files: still listed and reported, never
        # blocking.
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
    dwell = state.get("dwell") or {}
    if dwell.get("status") == "checks_failed":
        # CI is the same evidence the proof is, run where the change will land.
        failing = ", ".join(dwell.get("checks_failed") or []) or "a check"
        reasons.append(f"the pull request's checks fail ({failing})")
    overridden = bool(reasons) and state["verdict"] == Verdict.APPROVED.value
    if overridden:
        state["verdict"] = Verdict.REVISE.value

    # Every pass but the opening one closes a cycle, so it counts a step --
    # here, where every cycle passes. A missing plan, not a missing report,
    # marks the opening pass: a Builder that reports nothing must not restart
    # the count.
    if state.get("plan"):
        state["step_count"] = state.get("step_count", 0) + 1

    if timed_out:
        # A verdict nobody reached is not a ruling, and the feed says so.
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


# Whether the Planner is shown what the corpus ranks closest to the goal.
# `tests/conftest.py` switches it off, so no planning test depends on a local
# corpus.
PLANNER_CORPUS_MAP = os.getenv("PLANNER_CORPUS_MAP", "1").strip().lower() not in {
    "0", "false", "no",
}

# How many corpus hits the Planner is shown: enough to name what a goal
# touches, few enough that a small local seat still reads the goal.
PLANNER_MAP_RESULTS = 6

# The map says where work goes; reading the passages is the Researcher's job.
PLANNER_MAP_EXCERPT_CHARS = 160

CORPUS_MAP_HEADING = (
    "Archived documents the corpus ranks closest to this goal (these paths exist):"
)


def _corpus_map(goal: str) -> str:
    """What the corpus ranks closest to the goal, for the Planner; "" for nothing.

    A Planner shown nothing can name only what the goal names, and the
    Researcher searches on its plan. Only hits over `relevance_floor()` are
    shown, so a model with no floor yet gets no map. The Planner still calls
    no tool: this is context handed to it. Never raises.
    """
    if not goal.strip():
        return ""
    try:
        # The floor first: with none there is no map, and searching anyway
        # loaded the embedder -- evicting the seat about to plan -- for nothing.
        from langgraph_agent.graphrag_server import relevance_floor

        floor = relevance_floor()
        if floor is None:
            return ""
        response = _call_tool(
            "search_knowledge_graph", {"query": goal, "top_k": PLANNER_MAP_RESULTS}
        )
    except Exception:
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
    return CORPUS_MAP_HEADING + "\n" + "\n".join(lines)


def _make_plan(state: AgentState) -> dict[str, Any]:
    """Ask the Planner for a plan; return the parsed output.

    Reads state; never writes it, so it is safe to run under `_with_deadline`.
    """
    state_injection = _get_state_injection(state)
    goal = state.get("goal", "")

    corpus_map = _corpus_map(goal) if PLANNER_CORPUS_MAP else ""
    map_block = f"\n\n{corpus_map}" if corpus_map else ""

    messages = [
        SystemMessage(content=seat_prompt("planner")),
        HumanMessage(content=f"{state_injection}{map_block}\n\nUser goal: {goal}"),
    ]

    llm = get_agent_llm("planner")
    response = llm.invoke(messages)

    return _parse_planner_output(_as_text(response.content))


# What a timed-out Planner leaves in `plan`. Never empty: the gate counts a
# step only while a plan exists, so an empty one would loop uncounted until
# LangGraph's recursion limit killed the run.
_PLANNER_TIMED_OUT = (
    "1. The Planner did not respond within {seconds}s, so this goal was never "
    "broken into steps.\n"
    "2. Do not guess at the plan. Report the goal as unplanned, and set that "
    "as a blocker so the Architect sees why nothing was implemented.\n"
)


# What a Planner that answered off-format leaves in `plan`, non-empty for the
# same reason. Worded apart from the timeout, since only this one means the
# model cannot hold the seat.
_PLANNER_NO_STEPS = (
    "1. The Planner's seat replied without a readable `## Steps` section, so "
    "this goal was never broken into steps.\n"
    "2. Do not guess at the plan. Report the goal as unplanned and set that as "
    "a blocker, saying the Planner's reply could not be read as a plan -- this "
    "is a seat that cannot hold its format, not a goal that resists "
    "planning.\n"
)


# The fixed openings of both placeholders; `_PLANNER_TIMED_OUT` goes on to name
# the deadline.
_PLACEHOLDER_PLAN_OPENINGS = (
    "1. The Planner did not respond within",
    "1. The Planner's seat replied without a readable",
)


def plan_is_placeholder(plan: str) -> bool:
    """Is `plan` a note about the Planner failing, rather than a plan?

    Both placeholders route to the Builder. `_route_from_planner` overrides
    routing and has to tell them apart: a plan nobody wrote is not worth a
    search.
    """
    return plan.lstrip().startswith(_PLACEHOLDER_PLAN_OPENINGS)


def planner_node(state: AgentState) -> AgentState:
    """Planner: turn the goal into steps and choose the next seat.

    No tools. Bounded by `NODE_DEADLINE_SECONDS`, and never leaves `plan`
    empty: the gate counts a step only while a plan exists.
    """
    if RUN_CONTROL.stopped():
        # Any plan in state is kept, and `next_agent` still records the last
        # real routing decision.
        state["messages"].append(
            "[Planner] Stopped by the emergency stop before this seat ran."
        )
        return state

    parsed = _with_deadline(
        lambda: _make_plan(state), NODE_DEADLINE_SECONDS, None
    )

    if parsed is None:
        # On a revise cycle the existing plan beats the placeholder.
        state["plan"] = state.get("plan") or _PLANNER_TIMED_OUT.format(
            seconds=int(NODE_DEADLINE_SECONDS)
        )
        # The Builder is the shorter way back to the Architect, the only node
        # that ends a run.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Planner] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder"
        )
        return state

    plan = parsed.get("plan", "")
    if not plan.strip():
        # The seat answered without steps. As on the timeout path, an existing
        # plan beats the placeholder.
        state["plan"] = state.get("plan") or _PLANNER_NO_STEPS
        # To the Builder, wherever the reply routed: a search on a placeholder
        # finds nothing.
        state["next_agent"] = "Builder"
        # Named in the feed: changing the seat's model is the only fix.
        state["messages"].append(
            "[Planner] Seat returned no plan steps; routing to Builder "
            "(check the Planner's model)"
        )
        return state

    state["plan"] = plan
    state["next_agent"] = parsed.get("next_agent", "Builder")
    state["messages"].append(f"[Planner] Plan created. Next agent: {state['next_agent']}")

    return state


def _as_text(content: Any) -> str:
    """The text a model answered with, whatever shape its content came in.

    Some providers answer with a list of content blocks; a thinking model sends
    a `thinking` block before the text. Only `text` blocks are kept: a plan or
    verdict parsed out of the reasoning would be one the model never gave.
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
# it to `no_relevant_knowledge` with a feed line of its own, since a silent
# seat and a corpus with nothing to say look the same to the Builder.
_SEAT_EMPTY = "seat_empty"


# What the Builder gets when the seat answered with nothing -- worded apart
# from a timeout and an empty corpus, because only this one means the seat is
# broken.
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
    """A retrieved passage as the Builder sees it: whole, or visibly cut.

    A silent trim would read as a source that had nothing more to say.
    """
    text = (content or "").strip()
    if len(text) <= RESEARCH_SNIPPET_CHARS:
        return text
    return text[:RESEARCH_SNIPPET_CHARS].rstrip() + f"\n   [... passage truncated at {RESEARCH_SNIPPET_CHARS} characters]"


# How much retrieval the Researcher's seat is shown when asked to judge it:
# less than the Builder gets, since the seat is a small local model and whether
# a passage bears on the plan shows in its opening lines.
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

    Asked only when the search did not clearly answer the plan, the seat is
    shown what came back and why none of it was accepted, so it judges evidence
    instead of inventing some. `why_none` says why there is nothing, when there
    is nothing.
    """
    if not results:
        return (
            f"Retrieval: nothing to judge -- {why_none}. Do not present anything "
            "as retrieved. If the plan depends on knowing existing code or docs, "
            "answer `no_relevant_knowledge`."
        )

    from langgraph_agent.graphrag_server import best_score

    top = best_score(results)
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


def _retrieved_findings(results: list[dict[str, Any]], floor: float, graph: Any) -> str:
    """Retrieval that answered the plan, written up in the Researcher's format.

    Only the best hit had to clear the floor. The rest ride along -- the diverse
    hits `SEARCH_ESCALATION` widens the window for -- each marked with its side
    of the floor, so the Builder weighs a weaker passage as weaker.
    """
    shown = results[:RESEARCH_RESULTS]
    findings = "## Key Findings\n"
    above = 0
    for i, result in enumerate(shown, 1):
        snippet = _research_snippet(result.get("content", ""))
        findings += f"\n{i}. {_passage_source(result)}\n{snippet}"
        if result.get("related_entities"):
            related = ", ".join(str(e) for e in result["related_entities"][:3])
            findings += f"\n   Related: {related}"
        score = float(result.get("score") or 0.0)
        if score > floor:
            above += 1
            findings += f"\n   Score: {score:.2f}\n"
        else:
            findings += f"\n   Score: {score:.2f} (under the relevance floor)\n"

    findings += (
        "\n## Relevant Context\n"
        f"Retrieved {len(shown)} passage(s) from the knowledge base; {above} "
        f"scored over its relevance floor of {floor:.2f}, and any under it is "
        "marked above.\n"
    )
    if graph and graph.get("subgraph_nodes", 0) > 0:
        findings += (
            f"\nKnowledge graph has {graph['subgraph_nodes']} nodes "
            f"and {graph['subgraph_edges']} relationships.\n"
        )
    return findings + (
        "\n## Recommendations for Builder\n"
        "Use the retrieved documentation as reference for implementation.\n"
        "\n## Status\nready_for_builder"
    )


def _gather_research(state: AgentState) -> tuple[str, str]:
    """Retrieve for the Researcher and return `(findings, status)`.

    Retrieval answers by itself when its best hit clears `relevance_floor()`
    (`best_score`, the statistic the floor was measured on); otherwise the seat
    judges what came back. Reads state and never writes it, so it can run under
    `_with_deadline`.
    """
    plan = state.get("plan", "")
    results: list[dict[str, Any]] = []
    floor: float | None = None
    accepted = False
    graph: Any = {}
    why_none = (
        "the search returned nothing for this plan"
        if plan.strip()
        else "the plan is empty, so there was nothing to search on"
    )
    try:
        response = _call_tool(
            "search_knowledge_graph", {"query": plan, "top_k": RESEARCH_RESULTS}
        )
        results = list(response.get("results") or [])
        if response.get("source") == "no_corpus":
            why_none = "no corpus has been built on this machine"
        # Imported late, as mcp_client does: graphrag_server pulls in the database driver.
        from langgraph_agent.graphrag_server import best_score, relevance_floor

        floor = relevance_floor()
        accepted = floor is not None and bool(results) and best_score(results) > floor
        if accepted and results[0].get("id"):
            # A hit's id is its document's node in the graph.
            try:
                graph = _call_tool(
                    "query_knowledge_graph", {"entity": results[0]["id"], "hops": 2}
                )
            except Exception:
                pass  # the graph is an extra, never a reason to fail
    except Exception as exc:
        why_none = f"the knowledge-base search failed ({exc})"

    if accepted and floor is not None:
        return _retrieved_findings(results, floor, graph), ResearchStatus.READY_FOR_BUILDER.value

    messages = [
        SystemMessage(content=seat_prompt("researcher")),
        HumanMessage(
            content=f"{_get_state_injection(state)}\n\nPlan to research:\n{plan}\n\n"
            f"{_retrieval_for_the_seat(results, floor, why_none)}"
        ),
    ]
    findings = _as_text(get_agent_llm("researcher").invoke(messages).content)
    parsed = _parse_researcher_output(findings)
    # Silence is not research, whatever the defaulted status says: announced as
    # success, it sent an empty `research` on to the Builder and the gate back
    # to the same silent seat, a step at a time.
    if _said_nothing(findings, parsed):
        return _RESEARCH_EMPTY, _SEAT_EMPTY
    return findings, parsed["status"]


# What the Builder gets when the Researcher runs out of time.
# `no_relevant_knowledge`, not `need_replan`: a deadline says nothing about the
# plan, and replanning would rerun the same slow retrieval.
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
    """Researcher: search the knowledge base for the plan, and say what bears on it.

    The node runs the read-only GraphRAG tools itself; the seat is consulted
    only when retrieval did not answer on its own (`_gather_research`). Bounded
    by `NODE_DEADLINE_SECONDS`: `RUN_BUDGET_SECONDS` is checked only between
    supersteps, so a stalled seat would otherwise hang the run here.
    """
    if RUN_CONTROL.stopped():
        # Nothing was attempted, so `research` keeps what an earlier cycle
        # found.
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

    state["research"] = research_findings
    state["research_status"] = research_status

    if deadline_hit:
        # Worded apart from an empty corpus, which is not a problem; this is.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Researcher] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder without research"
        )
    elif seat_empty:
        # To the Builder, not the Planner: the plan did not fail, and
        # replanning would come back to the same silent seat. The feed names
        # the fix.
        state["next_agent"] = "Builder"
        state["messages"].append(
            "[Researcher] Seat returned no findings; routing to Builder "
            "without research (check the Researcher's model)"
        )
    elif research_status == "need_replan":
        state["next_agent"] = "Planner"
        state["messages"].append("[Researcher] Needs replan")
        # A replan is a cycle the Architect's gate never sees -- the Planner
        # and the Researcher can go round without reaching the Builder -- so
        # it is counted here, and `MAX_STEPS` bounds this loop too.
        state["step_count"] = state.get("step_count", 0) + 1
    elif research_status == "no_relevant_knowledge":
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] No relevant knowledge, proceeding")
    else:
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] Research complete, routing to Builder")

    return state


# The Builder's belt. GraphRAG is absent on purpose: retrieval is the
# Researcher's, and a Builder that searches stops working from the plan it was
# handed.
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
                "Carry your change all the way to the default branch: survey, "
                "branch, stage, commit, update (merge the default branch in), "
                "push, open a pull request, wait for its CI checks, merge it, and "
                "clean up the branch. Prefer this over a series of terminal git "
                "commands -- it runs the stages in the only order that works, "
                "stops at the first one that cannot go on and says which. It "
                "never commits onto the default branch, never force-pushes, and "
                "never merges past a failing check or one it did not read. "
                "Before anything is pushed, what you wrote this pass is linted "
                "and run; if that fails the call stops at 'prove' with nothing "
                "committed. A failing check stops it at 'checks' with the failing "
                "jobs and the tail of the log in failed_log: fix them and call "
                "git_dwell again, which commits the fix and pushes it. Checks "
                "still running when the time runs out are not a failure: the "
                "result says 'pending', the pull request stays open, and the "
                "console merges it once they pass. With no remote the commit "
                "stays in the repository. To leave the pull request open for "
                "someone to read, name the stages you want and leave 'merge' out."
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
                            "commit, update, push, pr, checks, merge, cleanup. "
                            "They always run in that order. Defaults to all of "
                            "them; pass a shorter list to stop early, e.g. "
                            "everything up to 'pr' to leave the pull request "
                            "unmerged, or ['checks', 'merge', 'cleanup'] to "
                            "finish one left pending."
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
                            "Branch to create when HEAD is the default branch, "
                            "or when this branch's pull request has already "
                            "merged. Derived from the message when omitted."
                        ),
                    },
                    "checks_timeout": {
                        "type": "number",
                        "description": (
                            "Seconds to wait for the pull request's checks. "
                            "Bounded by what is left of your deadline; 0 reads "
                            "them once without waiting."
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

# Appended to the Builder's prompt on a discussion run, so the seat writes a
# proposal rather than reporting work it had no tools to do.
DISCUSSION_NOTE = (
    "\n\nThis run is DISCUSSION ONLY. You have no tools: you cannot read "
    "files, write files, run commands or run tests. Work from the plan and the "
    "research findings you were given. Do not report work as done and do not "
    "list files under '## Files Modified' -- describe what you would change, "
    "which files it would touch, and what you would need to verify it. That "
    "description is the product of this run. Answer '## Blockers' with none: a "
    "proposal is not blocked by needing review or approval."
)

# Appended when the run builds a generated project, so the seat knows the
# boundary before a refusal teaches it.
OUTPUT_DIR_NOTE = (
    "\n\nThis run builds a separate project in {output_dir}/. Write every "
    "file under that directory, spelling the full path from the project root "
    "(for example {output_dir}/main.py) -- filesystem_write refuses any other "
    "path. Pass cwd={output_dir} to terminal_execute and run_tests when you "
    "run what you wrote. The git tools act in {output_dir}/ itself, and "
    "git_dwell needs it to be a repository of its own (git init it first)."
)


# The Builder's git tools, which on a run given a project act in that project.
_GIT_TOOL_NAMES = frozenset({"git_status", "git_diff", "git_dwell"})


def _outside_output_dir(path: str, output_dir: str) -> str | None:
    """Why `filesystem_write` may not write `path` on this run, or None.

    Resolved rather than matched as text, so `projects/x/../../nodes.py` and a
    symlink leading out are both caught. Checked here, not in the tool client,
    because the scope belongs to the run and the client serves every run.
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


# How many times the Builder may think-and-call in one pass. A broad goal wants
# it raised: a pass cut off at the cap reports nothing the gate can approve.
MAX_BUILDER_TOOL_TURNS = int(os.getenv("MAX_BUILDER_TOOL_TURNS", "8"))

# Wall-clock ceiling for one Builder pass. The turn cap bounds how many calls
# it makes, not how long they take.
BUILDER_DEADLINE_SECONDS = float(os.getenv("BUILDER_DEADLINE_SECONDS", "240"))

# Held back from the tool loop so the verification pass always runs; what the
# loop leaves unspent is added to it.
VERIFY_RESERVE_SECONDS = float(os.getenv("VERIFY_RESERVE_SECONDS", "60"))

# A tool result longer than this is cut, and says so: a whole large file in the
# transcript crowds out the plan.
MAX_TOOL_RESULT_CHARS = 20000

# How much of a failure's reason its report line carries: enough to tell a
# refused pipe from a missing file.
MAX_FAILURE_REASON_CHARS = 120


def _failure_reason(result: Any) -> str:
    """One line saying why a tool call failed, for its report line.

    The tool's own `error` first, being written to be read; then the last line
    on stderr, where a traceback puts its exception; then stdout, where pytest
    puts its summary; then the exit status.
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


# Held back from a `git_dwell` call's checks wait for the stages after it --
# the merge and the cleanup -- and for the push and pull request before the
# wait begins, all of which finish inside the Builder's deadline.
DWELL_TAIL_RESERVE_SECONDS = 45.0


def _checks_budget(requested: Any, deadline: _Deadline) -> float:
    """How long a `git_dwell` call may wait on CI: what was asked, within the deadline.

    A tool call is never abandoned, so the wait has to end inside the pass on
    its own. Checks still running past it are not a failure: the pull request is
    left pending, for the console to finish.
    """
    try:
        wanted = DWELL_CHECKS_WAIT_SECONDS if requested is None else float(requested)
    except (TypeError, ValueError):
        wanted = DWELL_CHECKS_WAIT_SECONDS
    if wanted != wanted:  # NaN
        wanted = DWELL_CHECKS_WAIT_SECONDS
    return max(0.0, min(wanted, deadline.remaining() - DWELL_TAIL_RESERVE_SECONDS))


def _dwell_gate(
    state: AgentState, files_changed: list[str], tool_log: list[str], deadline: _Deadline
) -> Callable[[dict[str, Any]], str]:
    """The proof a `git_dwell` call must pass before anything leaves the machine.

    The pass's own proof (`_prove`) runs after the tool loop, so a dwell called
    mid-pass would push what nobody had linted or run. This runs the same proof
    first, on what the pass has written so far and what an earlier pass left
    failing, and returns why the call may not go ahead, or "". Only a call
    that ships -- push, pr, checks or merge -- is held to it: a commit that
    stays local is not seen by anyone.
    """

    def gate(args: dict[str, Any]) -> str:
        stages = {str(x) for x in (args.get("stages") or DWELL_DEFAULT_STAGES)}
        if not stages & SHIPPING_STAGES:
            return ""
        scratch: list[str] = []
        verification, (lint_failures, _, _) = _prove(state, files_changed, scratch, deadline)
        if not verification and not lint_failures:
            return ""
        expected = bool(state.get("expect_failures"))
        problems = [
            f"{path} does not run ({detail.splitlines()[-1] if detail else 'no output'})"
            for path, status, detail in verification
            if status == "failed" and not expected
        ]
        problems += [
            f"{path} was not run before the deadline or the stop"
            for path, status, _ in verification
            if status == "unverified"
        ]
        problems += [
            f"{path} fails lint ({', '.join(sorted({code for _, code, _ in findings}))})"
            for path, findings in lint_failures
        ]
        if not problems:
            tool_log.append(f"prove(before git_dwell) -> clean ({len(verification)} run)")
            return ""
        tool_log.append(f"prove(before git_dwell) -> {len(problems)} problem(s)")
        return (
            "nothing was committed or pushed, because what this run wrote is not "
            "proven: " + "; ".join(problems) + ". Fix them and call git_dwell again, or "
            "name stages that stop at 'commit' to commit without pushing."
        )

    return gate


def _dwell_line(args: dict[str, Any], result: Any, outcome: str) -> str:
    """A `git_dwell` call's report line: the stages asked for, how far it got, how it ended.

    One call is a whole pipeline, so a push that failed after a commit must not
    read as one that never committed, and a pull request left waiting on its
    checks must not read as merged.
    """
    record = result if isinstance(result, dict) else {}
    target = ",".join(str(x) for x in (args.get("stages") or [])) or "default"
    done = [
        entry["stage"] for entry in (record.get("stages") or [])
        if isinstance(entry, dict) and entry.get("ok")
    ]
    reached = f" [{' -> '.join(done)}]" if done else ""
    if not record.get("success"):
        outcome = f"{outcome} at {record.get('stopped_at', '?')}"
    elif record.get("pending"):
        outcome = f"ok, pending: {record['pending']}"
    elif record.get("merged"):
        number = (record.get("pull_request") or {}).get("number")
        outcome = f"ok, merged #{number}" if number else "ok, merged"
    return f"git_dwell({target}){reached} -> {outcome}"


def _run_builder_tools(
    llm: Any,
    messages: list[Any],
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline,
    output_dir: str = "",
    gate: Callable[[dict[str, Any]], str] | None = None,
    dwells: list[dict[str, Any]] | None = None,
) -> tuple[str, bool, bool, bool]:
    """Let the Builder call tools until it stops asking for them.

    Returns its closing message, and whether it ran out of turns, ran out of
    time, or was stopped. `files_changed` grows only when a write reports
    success. A `git_dwell` call is held to `gate` (`_dwell_gate`), given a
    checks wait that ends inside `deadline`, and its result appended to
    `dwells`.

    Only the model's own call is abandoned at the deadline. A tool call never
    is -- a worker abandoned mid-write would keep writing after the node
    returned -- so a turn's tools always finish, and the deadline and the
    emergency stop are checked between turns. Never inside a batch: a skipped
    call would leave a `tool_call_id` the model was told about with no reply.
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
            if output_dir and name in _GIT_TOOL_NAMES:
                # The project's own repository, never the checkout around it:
                # `projects/` is ignored there, so `git add -A` could only stage
                # the operator's own work -- and the default pipeline merges.
                args["cwd"] = output_dir
            if name == "git_dwell":
                args["checks_timeout"] = _checks_budget(args.get("checks_timeout"), deadline)

            if name not in BUILDER_TOOL_NAMES:
                # Refused, not run: the client serves the Researcher's tools
                # too, and what the model was offered is not what it may call.
                result: Any = {"success": False, "error": f"{name} is not a Builder tool"}
            elif (
                name == "filesystem_write"
                and output_dir
                and (outside := _outside_output_dir(str(args.get("path", "")), output_dir))
            ):
                result = {"success": False, "error": outside}
            elif name == "git_dwell" and gate is not None and (refusal := gate(args)):
                result = {
                    "success": False, "error": f"prove: {refusal}",
                    "stopped_at": "prove", "stages": [],
                }
            else:
                try:
                    result = _call_tool(name, args)
                except Exception as exc:
                    result = {"success": False, "error": str(exc)}

            ok = bool(result.get("success")) if isinstance(result, dict) else False

            if name == "filesystem_write" and ok:
                path = str(args.get("path", ""))
                if path and path not in files_changed:
                    files_changed.append(path)

            # The Architect rules on these lines, so each says where it ran
            # and, when it failed, why.
            target = args.get("path") or args.get("command") or ""
            where = f" [cwd={args['cwd']}]" if args.get("cwd") else ""
            outcome = "ok"
            if not ok:
                reason = _failure_reason(result)
                outcome = f"failed: {reason}" if reason else "failed"
            if name == "git_dwell":
                tool_log.append(_dwell_line(args, result, outcome))
                if dwells is not None and isinstance(result, dict):
                    dwells.append({**result, "cwd": str(args.get("cwd") or "")})
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


# What the verification pass can run; markdown, config and data have nothing to
# run.
RUNNABLE_SUFFIXES = (".py",)

# Rules ruff may fix on the Builder's behalf: only those that cannot change
# behaviour -- trailing whitespace, a final newline, import order, and two
# spellings of one object (`datetime.UTC`, UP017; `TimeoutError`, UP041). The
# rest is the Builder's to fix: a "safe" fix can still drop an import kept for
# its side effect.
LINT_AUTOFIX_RULES = ("W291", "W292", "W293", "I001", "UP017", "UP041")

# Per-file ceiling for the lint pass. ruff answers in milliseconds; this bounds
# a wedged interpreter, not the linter.
LINT_TIMEOUT_SECONDS = 30

# Findings shown per file: enough to say what kind of trouble a file is in;
# ruff lists the rest.
MAX_LINT_FINDINGS_SHOWN = 8

# Verification runs with nobody watching, so nothing may wait on a window: a
# script ending in `plt.show()` would sit out the timeout and read as failed.
# The display variables go too, for libraries that look for a display
# themselves.
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

# The least time worth starting a file in: a smaller slice times out on a
# working file and accuses it of failing. Better to admit it was never run.
MIN_VERIFY_SLICE_SECONDS = 1.0

# Why a file was left unrun when the operator stopped the run, worded apart
# from the deadline's reason.
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
    "no_tools": "the Builder's model cannot call tools, so nothing was built",
}


# A Blockers section that answers "none" -- and often keeps writing ("none -
# Note: this raises by design"). Only a leading token standing as a complete
# clause counts: "none of the tests pass" is a real blocker, and an
# unrecognised phrasing stays one, since a spurious blocker costs a cycle and a
# dropped one costs the guarantee.
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

    `python pkg/mod.py` puts `pkg` itself on sys.path, so a module importing
    its own package absolutely fails however correct it is. It is imported
    instead, the way its callers use it, with the directory above the outermost
    package on the path; an `__init__.py` is imported by its package's name.

    Returns None for a file outside any package, or one whose package path is
    not a valid module name: both are run as scripts.
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
    """A timed-out verification, with the tail of what the file printed.

    The timeout alone says a file hung, not where: blocked on its first line,
    or done and waiting at `plt.show()`. The tail is kept because what a hung
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
    cwd: str = "",
) -> list[tuple[str, str, str]]:
    """Run the runnable files the Builder wrote, and report what happened.

    Writing a file is not evidence that it works. A module inside a package is
    imported rather than executed (`_import_target`), both under this process's
    own interpreter, which has the project's dependencies. `deadline` bounds the
    pass as a whole, and a file past it or past the emergency stop comes back
    "unverified" -- never "ok". `cwd` is where each file runs: a project run's
    directory, where the Builder was told to run what it wrote, so a file that
    opens its neighbours by relative path runs the same for the proof.

    Returns (path, status, detail) per runnable file, where status is "ok",
    "imported", "failed" or "unverified".
    """
    results: list[tuple[str, str, str]] = []

    for path in files_changed:
        if not path.endswith(RUNNABLE_SUFFIXES):
            continue

        if RUN_CONTROL.stopped():
            # Unrun is not passing: it blocks approval even under
            # `expect_failures`.
            results.append((path, "unverified", VERIFY_STOPPED_REASON))
            tool_log.append(f"verify({path}) -> not run (stopped)")
            continue

        if deadline is not None and deadline.remaining() < MIN_VERIFY_SLICE_SECONDS:
            results.append((path, "unverified", VERIFY_DEADLINE_SKIP_REASON))
            tool_log.append(f"verify({path}) -> not run (deadline)")
            continue

        target = _import_target(path)
        env: dict[str, str | None] = dict(HEADLESS_VERIFY_ENV)
        # Quoted: the command is split like a shell's, so a path with a space
        # would arrive as two arguments.
        python = shlex.quote(sys.executable)
        if target is None:
            # Absolute, so it names the same file from whichever `cwd` it runs in.
            script = str(Path(path).resolve()) if cwd else path
            command, passed = f"{python} {shlex.quote(script)}", "ok"
        else:
            root, module = target
            command, passed = f'{python} -c "import {module}"', "imported"
            inherited = os.environ.get("PYTHONPATH")
            env["PYTHONPATH"] = root + (os.pathsep + inherited if inherited else "")

        try:
            result = _call_tool(
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
                    **({"cwd": cwd} if cwd else {}),
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


# Each file left failing lint, with its findings as (line, code, message).
_LintFailures = list[tuple[str, list[tuple[int, str, str]]]]


def _lint_written_files(
    paths: list[str],
    tool_log: list[str],
    deadline: _Deadline | None = None,
) -> tuple[_LintFailures, list[str], str]:
    """Lint the Python the Builder wrote, the way CI will.

    Running a file proves it does not raise, not that CI accepts it. Each file
    gets `ruff check` with the project's own configuration and
    `LINT_AUTOFIX_RULES` applied, and what is left is reported.

    Returns (failures, fixed, unavailable): the findings left per file, the
    paths ruff tidied, and why ruff could not run at all ("" when it did). A
    missing linter is no defect in the file, so it is reported rather than
    failed, and the pass stops at the first sign of it.
    """
    failures: _LintFailures = []
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
            result = _call_tool(
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


def _seat_pass(
    state: AgentState,
    output_dir: str,
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline,
    dwells: list[dict[str, Any]] | None = None,
) -> tuple[str, bool, bool, bool, str]:
    """The Builder's seat at work.

    Returns (closing message, out of turns, out of time, stopped, why it could
    not act), and appends each `git_dwell` result to `dwells`. A discussion run
    binds no tools, so it takes the path of a model that cannot call any: there
    is no tool loop for it to act through. The last element is empty unless a
    real model was refused its tools on a run that needed them -- such a pass
    can only describe work, so it must not read as having done it. A stub
    answers in canned text by design and is exempt.
    """
    discuss_only = bool(state.get("discuss_only"))
    note = (
        DISCUSSION_NOTE
        if discuss_only
        else OUTPUT_DIR_NOTE.format(output_dir=output_dir) if output_dir else ""
    )
    messages: list[Any] = [
        SystemMessage(content=seat_prompt("builder") + note),
        HumanMessage(
            content=f"{_get_state_injection(state)}\n\nPlan to implement:\n"
            f"{state.get('plan', '')}\n\nResearch findings:\n{state.get('research', '')}"
        ),
    ]
    llm = get_agent_llm("builder")
    no_tools = ""
    try:
        tool_llm = None if discuss_only else llm.bind_tools(BUILDER_TOOLS)
    except AttributeError as exc:  # StubLLM, or a model without tool support
        tool_llm = None
        if not getattr(llm, "is_stub", False):
            no_tools = str(exc).removeprefix("bind_tools: ").removeprefix("bind_tools") or (
                "the model offers no tool binding"
            )

    if RUN_CONTROL.stopped():
        return "", False, False, True, no_tools
    if tool_llm is None:
        reply = _with_deadline(
            lambda: _as_text(llm.invoke(messages).content), deadline.remaining(), None
        )
        return reply or "", False, reply is None, False, no_tools
    gate = _dwell_gate(state, files_changed, tool_log, deadline)
    return (
        *_run_builder_tools(
            tool_llm, messages, files_changed, tool_log, deadline, output_dir,
            gate=gate, dwells=dwells,
        ),
        "",
    )


def _prove(
    state: AgentState, files_changed: list[str], tool_log: list[str], deadline: _Deadline
) -> tuple[list[tuple[str, str, str]], tuple[_LintFailures, list[str], str]]:
    """Lint, then run, what this pass wrote and what an earlier pass left failing.

    A failing file is re-checked until it passes, so it cannot clear by being
    left alone; one an earlier pass wrote and something since deleted drops out,
    since deleting it is a real fix. Lint goes first, so what runs is the file
    after ruff's fixes.
    """
    carried = [
        path
        for path in (state.get("failed_verification") or [])
        if path not in files_changed and Path(path).exists()
    ]
    lint_carried = [
        path
        for path in (state.get("lint_failed") or [])
        if path not in files_changed and path not in carried and Path(path).exists()
    ]
    lint = _lint_written_files(files_changed + carried + lint_carried, tool_log, deadline)
    # A discussion run writes nothing, and has no project to run in.
    cwd = "" if state.get("discuss_only") else str(state.get("output_dir") or "")
    return _verify_written_files(files_changed + carried, tool_log, deadline, cwd), lint


def _proof_report(
    verification: list[tuple[str, str, str]],
    stopped: bool,
    lint: tuple[_LintFailures, list[str], str],
) -> str:
    """The report's account of what was run and linted, file by file."""
    report = ""
    if verification:
        report += (
            "\n\nVerification (each runnable file was executed, and each package "
            "module imported, except as noted):\n"
        )
        report += "\n".join(
            f"- {path}: {_VERIFY_LABELS[status]}" + (f"\n{detail}" if detail else "")
            for path, status, detail in verification
        )
    unverified = [path for path, status, _ in verification if status == "unverified"]
    if unverified:
        why = (
            "the run was stopped"
            if stopped
            else f"the Builder ran out of its {int(BUILDER_DEADLINE_SECONDS)}s budget"
        )
        report += (
            f"\n\n{len(unverified)} file(s) were not executed: {why}. They are "
            "unproven rather than working, and are re-checked next cycle."
        )

    failures, fixed, unavailable = lint
    if failures or fixed or unavailable:
        report += "\n\nLint (ruff, with this project's configuration):"
        if unavailable:
            report += f"\n- not run: {unavailable}"
        for path, findings in failures:
            report += f"\n- {path}: {len(findings)} error(s)"
            report += "".join(
                f"\n  line {line}: {code} {message}"
                for line, code, message in findings[:MAX_LINT_FINDINGS_SHOWN]
            )
            if len(findings) > MAX_LINT_FINDINGS_SHOWN:
                report += (
                    f"\n  ...and {len(findings) - MAX_LINT_FINDINGS_SHOWN} more; "
                    f"`python -m ruff check {path}` lists them all"
                )
        if fixed:
            report += (
                "\n- whitespace, import order and alias spellings fixed automatically in: "
                + ", ".join(fixed)
            )
    return report


def builder_node(state: AgentState) -> AgentState:
    """Builder: implement the plan through its tools, then prove what was written.

    The report, `blockers` and the feed line say what was proven, never what the
    seat claimed: a file counts as written only when a write call succeeded, and
    as working only when it ran.
    """
    discuss_only = bool(state.get("discuss_only"))
    # Set by the caller, never by a seat, like `expect_failures`.
    output_dir = "" if discuss_only else str(state.get("output_dir") or "")
    files_changed: list[str] = []  # this pass alone; the state field is the run's
    tool_log: list[str] = []
    dwells: list[dict[str, Any]] = []

    # The tool loop gets the budget less the verification reserve, and whatever
    # it leaves unspent is added to the reserve: a slow build cannot starve the
    # proof.
    loop_deadline = _Deadline(max(0.0, BUILDER_DEADLINE_SECONDS - VERIFY_RESERVE_SECONDS))
    content, exhausted, out_of_time, stopped, no_tools = _seat_pass(
        state, output_dir, files_changed, tool_log, loop_deadline, dwells
    )
    verification, lint = _prove(
        state,
        files_changed,
        tool_log,
        _Deadline(VERIFY_RESERVE_SECONDS + loop_deadline.remaining()),
    )
    lint_failures = lint[0]
    lint_failed = [path for path, _ in lint_failures]
    failed = [(path, detail) for path, status, detail in verification if status == "failed"]
    unverified = [path for path, status, _ in verification if status == "unverified"]
    # A stop that lands during the proof leaves files unrun just as one during
    # the build does, and must not be reported as the deadline.
    stopped = stopped or any(detail == VERIFY_STOPPED_REASON for _, _, detail in verification)

    parsed = _parse_builder_output(content)
    # A loop cut off at the turn cap has no closing message to parse.
    builder_report = parsed.get("changes_made") or content or (
        f"No closing report: the Builder used all {MAX_BUILDER_TOOL_TURNS} tool "
        "turns without writing one."
        if exhausted
        else "No report produced."
    )
    builder_report += _proof_report(verification, stopped, lint)
    # The pass's last git_dwell says where the change stands; a pass that ran
    # none leaves the record of an earlier one, so failing checks keep blocking
    # until a dwell sees them pass.
    dwell = _dwell_record(dwells[-1]) if dwells else dict(state.get("dwell") or {})
    if dwells:
        builder_report += f"\n\nPull request: {_dwell_summary(dwell)}"
        failed_log = str(dwells[-1].get("failed_log") or "")
        if failed_log:
            builder_report += f"\nThe failing job's log ends:\n{failed_log[-MAX_VERIFY_DETAIL_CHARS:]}"
    if tool_log:
        builder_report += "\n\nTool calls:\n" + "\n".join(f"- {c}" for c in tool_log)

    # The state field is what the whole run produced, so it accumulates: a file
    # written on one pass is still the run's after a pass that wrote nothing.
    previously_changed = list(state.get("files_changed") or [])
    all_files_changed = previously_changed + [
        path for path in files_changed if path not in previously_changed
    ]

    # Written means a write call reported success, never that the report names
    # the file. Checked against the run's whole record, both spellings through
    # one normalizer, so `./foo.py` and `foo.py` are the same file.
    written = {_report_path_key(path) for path in all_files_changed}
    claimed = [path for path in parsed.get("files_modified", []) if path not in written]
    if claimed:
        builder_report += (
            "\n\nDescribed but not written (no successful write call): "
            + ", ".join(claimed)
        )

    # The other direction: the record is read as what is on disk now, so a path
    # since removed is retracted -- after `written`, which must still see the
    # whole record -- and named, so it does not silently vanish.
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

    # A discussion run has nothing to be blocked on: its Blockers section is
    # part of the proposal, and stays in the report.
    blockers = _clean_blockers(parsed.get("next_steps_blockers", ""))
    if discuss_only and blockers:
        builder_report += f"\n\nWhat the proposal says it would need:\n{blockers}"
        blockers = ""

    # The proof outranks the seat's conclusion. `expect_failures` excuses only a
    # file that ran and failed; an unrun file and a lint failure block whatever
    # it says.
    expected = bool(state.get("expect_failures"))
    if failed and not expected:
        blockers = "Files that do not run: " + "; ".join(
            f"{path} ({detail.splitlines()[-1] if detail else 'no output'})"
            for path, detail in failed
        )
    if unverified:
        lead = "Not executed before the stop" if stopped else "Not executed before the deadline"
        note = f"{lead}: " + ", ".join(unverified)
        blockers = f"{blockers}. {note}" if blockers else note
    if lint_failures:
        note = "Files that fail lint: " + "; ".join(
            f"{path} ({', '.join(sorted({code for _, code, _ in findings}))})"
            for path, findings in lint_failures
        )
        blockers = f"{blockers}. {note}" if blockers else note
    if dwell.get("status") == "checks_failed":
        note = "The pull request's checks fail: " + (
            ", ".join(dwell.get("checks_failed") or []) or "a check"
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
    if no_tools and not blockers:
        blockers = (
            f"The Builder's model cannot call tools ({no_tools}), so this pass could "
            "not write or run anything. Seat the Builder on a model that reports "
            "`tools` (`ollama show <tag>` lists its capabilities)."
        )

    state["builder_report"] = builder_report
    state["files_changed"] = all_files_changed
    # Rewritten every pass, so a file fixed later stops blocking; unverified
    # paths ride along so the next cycle re-runs them.
    state["failed_verification"] = [path for path, _ in failed] + unverified
    # Apart from the list above, because the gate excuses failures under
    # `expect_failures` and must not excuse unrun files with them.
    state["unverified"] = list(unverified)
    state["builder_cut_off"] = (
        "turn_cap" if exhausted
        else "deadline" if out_of_time
        else "no_tools" if no_tools
        else ""
    )
    state["lint_failed"] = lint_failed
    state["blockers"] = blockers
    state["dwell"] = dwell

    # The Architect rules on this line, so it never says "Implementation
    # complete" for a pass that wrote a broken file, was cut off or stopped, or
    # could not act at all.
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
        summary = f"Stopped by the operator. Files: {len(files_changed)}"
    elif out_of_time:
        summary = (
            f"Stopped at the {int(BUILDER_DEADLINE_SECONDS)}s deadline. "
            f"Files: {len(files_changed)}"
        )
    elif exhausted:
        summary = (
            f"Stopped after {MAX_BUILDER_TOOL_TURNS} tool turns without "
            f"finishing. Files: {len(files_changed)}"
        )
    elif no_tools:
        summary = (
            f"Could not act: the Builder's model cannot call tools. "
            f"Files: {len(files_changed)}"
        )
    elif discuss_only:
        summary = "Discussion only: proposal ready, nothing was changed"
    else:
        summary = f"Implementation complete. Files: {len(files_changed)}"
    # The counts are this pass; the run's total is named when it differs, so a
    # pass that wrote nothing does not read as the run losing its files.
    if len(all_files_changed) > len(files_changed):
        summary += f" ({len(all_files_changed)} changed so far this run)"
    if dwells:
        summary += f"; pull request: {_dwell_summary(dwell)}"
    state["messages"].append(f"[Builder] {summary}")

    return state
