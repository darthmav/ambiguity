#!/usr/bin/env python3
"""Find out which model actually works in which seat.

Two phases, deliberately separate, because they answer different questions and
one is far cheaper than the other.

**Phase 1 -- role probes.** One bounded call per (model, role) pair, through
the *real* role prompt and the *real* parser the node uses. It answers "can
this model hold this seat at all": did it answer, did the answer parse, and --
for the Builder -- can it call a tool. The Architect is asked twice, once
about finished work and once about blocked work, because the gate is the seat
that ends the run and a single verdict cannot be told from a fixed one. A model that fails its probe cannot be
rescued by a good team around it, and finding that out costs one call instead
of a whole run. This is the phase that would have caught the silent Researcher
in seconds.

**Phase 2 -- team runs.** Each configuration runs the same short exercise
through the same graph the console drives, instrumented per node. It answers
the question the probes cannot: whether four seats that each work alone make
progress *together*, or spend the run handing the same empty state back and
forth.

Both phases are read-mostly by design, with two exceptions worth knowing:

- Team runs let the Builder write files, so each one runs in its own sandbox
  directory under the scratchpad, never in the project. The Builder's tools
  resolve paths against the process cwd, so the sandbox is a `chdir`.
- The GraphRAG singleton resolves `./knowledge` on first use, so it is warmed
  from the project root *before* any sandbox exists. Every config then searches
  the real corpus, and no config gets an accidentally empty one.

Usage:

    python scripts/diagnose_seats.py --list
    python scripts/diagnose_seats.py --phase probe
    python scripts/diagnose_seats.py --phase teams --configs baseline,heavy-gate
    python scripts/diagnose_seats.py --anthropic --exercise all

Nothing here is a benchmark. One short exercise per configuration is a
data point, not a ranking, and the report says so where it matters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langgraph_agent.config import (
    AGENTS,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    DOLPHIN3_CYBER_8B,
    DOLPHIN_9B,
    NEMOTRON_3_ULTRA,
    QWEN3_8,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def quiet_logs() -> None:
    """Stop the libraries from writing over the report.

    Two separate sources, one fix each. `httpx` logs every provider call at
    INFO, which is one line per seat per turn and buries the node timings this
    script exists to show. `TQDM_DISABLE` covers anything that would draw a
    progress bar over the report instead of logging one.
    """
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("httpx", "httpcore", "psycopg", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("TQDM_DISABLE", "1")


# --------------------------------------------------------------------------
# Candidates and configurations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A model that could hold a seat."""

    key: str
    provider: str
    model: str
    note: str
    paid: bool = False


# Every model the diagnostic knows how to seat. `paid` ones are skipped unless
# --anthropic is passed: this script is meant to be run repeatedly while
# dialling a configuration in, and a default that bills on every run is a
# default nobody runs twice.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("dolphin-2.9.1", "ollama", DOLPHIN_9B,
              "Local, completion-only; holds Architect, Planner and "
              "Researcher by default"),
    Candidate("dolphin3-cyber", "ollama", DOLPHIN3_CYBER_8B,
              "Local, completion-only; a second choice for the three "
              "tool-free seats"),
    Candidate("qwen", "ollama", QWEN3_8,
              "Local, the only tag here with tools; holds Builder by default"),
    Candidate("nemotron", "ollama", NEMOTRON_3_ULTRA,
              "Ollama Cloud; the cloud model the console's pull list carries"),
    Candidate("opus", "anthropic", CLAUDE_OPUS, "Paid control", paid=True),
    Candidate("sonnet", "anthropic", CLAUDE_SONNET, "Paid control", paid=True),
    Candidate("haiku", "anthropic", "claude-haiku-4-5", "Paid control", paid=True),
)

BY_KEY: dict[str, Candidate] = {c.key: c for c in CANDIDATES}


@dataclass(frozen=True)
class TeamConfig:
    """One seating of the four agents."""

    name: str
    seats: dict[str, str]  # role -> candidate key
    rationale: str

    @property
    def paid(self) -> bool:
        return any(BY_KEY[k].paid for k in self.seats.values())


# Seatings, each varying one thing you could act on. They are not arbitrary
# permutations: a permutation sweep of a few models over four seats is hundreds
# of runs and tells you less, because nothing distinguishes a result from
# noise. Each of these has a stated reason to exist.
TEAM_CONFIGS: tuple[TeamConfig, ...] = (
    TeamConfig(
        "baseline",
        {"architect": "dolphin-2.9.1", "planner": "dolphin-2.9.1",
         "researcher": "dolphin-2.9.1", "builder": "qwen"},
        "The shipped defaults: dolphin-2.9.1 on the Architect, Planner and "
        "Researcher, qwen on the Builder. Everything else is measured against "
        "this. Keep it equal to DEFAULT_SEATS -- a control that has drifted "
        "from what the project ships is measuring nothing anyone runs.",
    ),
    TeamConfig(
        "cloud-planning",
        {"architect": "dolphin-2.9.1", "planner": "nemotron",
         "researcher": "nemotron", "builder": "qwen"},
        "Baseline with the cloud model planning and researching. Asks whether "
        "the two seats that read and reason are where a larger model pays.",
    ),
    TeamConfig(
        "cloud-builder",
        {"architect": "dolphin-2.9.1", "planner": "dolphin-2.9.1",
         "researcher": "dolphin-2.9.1", "builder": "nemotron"},
        "Baseline with the cloud model on the Builder. The Builder is the only "
        "seat that calls tools, so it is where a stronger model should pay.",
    ),
    TeamConfig(
        "heavy-gate",
        {"architect": "nemotron", "planner": "dolphin-2.9.1",
         "researcher": "dolphin-2.9.1", "builder": "qwen"},
        "Baseline with a big reasoner on the gate. The Architect ends the run, "
        "so a weak gate shows up as loops rather than as bad text.",
    ),
    TeamConfig(
        "mixed",
        {"architect": "nemotron", "planner": "dolphin-2.9.1",
         "researcher": "dolphin-2.9.1", "builder": "nemotron"},
        "Baseline with nemotron on the seats that rule and act, dolphin on the "
        "two that plan and read: does the cloud model belong on the gate and "
        "the Builder together?",
    ),
    TeamConfig(
        "nemotron-solo",
        {"architect": "nemotron", "planner": "nemotron",
         "researcher": "nemotron", "builder": "nemotron"},
        "A uniform seating on the cloud model. Read against baseline, it asks "
        "whether mixing local and cloud earns its keep.",
    ),
    TeamConfig(
        "anthropic-control",
        {"architect": "opus", "planner": "opus",
         "researcher": "sonnet", "builder": "sonnet"},
        "Paid control. If this one also loops, the framework is the "
        "bottleneck, not the models.",
    ),
    TeamConfig(
        "spend-on-the-gate",
        {"architect": "opus", "planner": "dolphin-2.9.1",
         "researcher": "dolphin-2.9.1", "builder": "qwen"},
        "Buys only the seat that decides when to stop. The cheapest way to "
        "find out whether the gate is what is failing.",
    ),
)

CONFIGS_BY_NAME: dict[str, TeamConfig] = {c.name: c for c in TEAM_CONFIGS}


# --------------------------------------------------------------------------
# Exercises
# --------------------------------------------------------------------------


def _retrieval_floor() -> str:
    """The measured relevance floor, read from the corpus rather than restated.

    Read at all because a restated one drifted: this help text said 0.40
    while the value was 0.37, and that number is not decoration -- it is the
    gate deciding whether the Researcher's model is consulted at all, in the
    one place an operator goes to find out which seat to trust.

    It is no longer a constant: the run that finishes a corpus measures one
    against that corpus and stores it with the corpus, so a machine with no
    measured corpus answers "unmeasured" -- which is the truth a stale number
    never told.

    Imported inside the function, not at module scope, because `--list` and
    `--help` must not pay for the corpus imports. If the package will not import, the
    function's *name* is the honest answer; a stale number is not.
    """
    try:
        from langgraph_agent.graphrag_server import relevance_floor
    except Exception:  # pragma: no cover - the script still has to print
        return "relevance_floor"
    floor = relevance_floor()
    return f"{floor:g}" if floor is not None else "unmeasured"


@dataclass(frozen=True)
class Exercise:
    """A short goal, plus what counts as having done it."""

    name: str
    goal: str
    expect_files: bool
    what_it_tests: str


EXERCISES: dict[str, Exercise] = {
    "build": Exercise(
        "build",
        "Create a file `slugify_tool.py` at the top level of the working "
        "directory. It must define `slugify(text: str) -> str`, which "
        "lowercases the text, replaces every run of non-alphanumeric "
        "characters with a single hyphen, and strips leading and trailing "
        "hyphens. At the bottom, under `if __name__ == \"__main__\":`, print "
        "the result of `slugify(\"  Hello, World!  \")`. Keep it to one file "
        "and do not create anything else.",
        expect_files=True,
        what_it_tests="All four seats end to end, including whether the "
                      "Builder can call a tool and produce a file that runs.",
    ),
    "research": Exercise(
        "research",
        "Explain how this project's Architect approval gate decides that a "
        "run is finished, and what overrules its `approved` verdict. Report "
        "the answer only -- create no files and change nothing.",
        expect_files=False,
        what_it_tests="Whether the gate can end a run with no files to point "
                      "at, and retrieval when the corpus holds the answer. "
                      "The corpus is the research archive, never this "
                      "checkout, so it holds the gate's documentation only "
                      "once someone has uploaded it (CLAUDE.md, say); until "
                      "then this question reaches the Researcher's model as "
                      "`offcorpus` does. Once it holds it, a hit over the "
                      "measured relevance floor is formatted straight into "
                      "the findings without the seat being called, so two "
                      "Researchers score alike here (this corpus's floor: "
                      "{floor}). Use `offcorpus` for the model.",
    ),
    "offcorpus": Exercise(
        "offcorpus",
        "Explain how PostgreSQL's MVCC vacuum decides which tuples are safe to "
        "freeze, and what the wraparound risk is if it falls behind. Report "
        "the answer only -- create no files and change nothing.",
        expect_files=False,
        what_it_tests="The Researcher's *model*, which the `research` exercise "
                      "cannot reach. `_gather_research` formats retrieval "
                      "straight into the output whenever the best hit clears "
                      "the measured relevance floor and only calls the seat "
                      "below it, so a question this corpus cannot answer is "
                      "the only team exercise where the Researcher's model is "
                      "the variable. It exists because the old hard-coded "
                      "floor of 0.3 once sat inside the off-corpus population "
                      "of the embedding model then in use, which meant the "
                      "exercise was silently testing retrieval, not the seat, "
                      "for as long as that number stood.",
    ),
    "plan": Exercise(
        "plan",
        "Produce a plan only, and create no files: how would you add a "
        "fifth agent named Reviewer to this system, between the Builder and "
        "the Architect?",
        expect_files=False,
        what_it_tests="Architect and Planner alone -- the cheapest signal on "
                      "whether the gate and the plan are coherent.",
    ),
}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class ProbeResult:
    model_key: str
    provider: str
    model: str
    role: str
    status: str          # ok | empty | malformed | error | stubbed
    seconds: float
    chars: int
    detail: str
    parsed: dict[str, Any] = field(default_factory=dict)
    excerpt: str = ""
    failure_reason: str = ""


@dataclass
class NodeVisit:
    node: str
    seconds: float
    message: str


@dataclass
class TeamResult:
    config: str
    exercise: str
    seats: dict[str, str]
    outcome: str = "not-run"
    seconds: float = 0.0
    steps: int = 0
    verdict: str = ""
    visits: list[NodeVisit] = field(default_factory=list)
    role_seconds: dict[str, float] = field(default_factory=dict)
    role_calls: dict[str, int] = field(default_factory=dict)
    field_chars: dict[str, int] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    failed_verification: list[str] = field(default_factory=list)
    blockers: str = ""
    messages: list[str] = field(default_factory=list)
    seat_failures: dict[str, str] = field(default_factory=dict)
    gate_passes: int = 0
    empty_handoffs: int = 0
    error: str = ""
    sandbox: str = ""
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def cyan(t: str) -> str:
    return _c(t, "36")


_STATUS_COLOR = {
    "ok": green, "approved": green,
    "empty": red, "error": red, "failed": red, "ceiling": red,
    "malformed": yellow, "stubbed": yellow, "budget": yellow,
    "cautious": yellow, "rubber": red, "inverted": red,
    "skipped": dim,
}


def paint(status: str) -> str:
    return _STATUS_COLOR.get(status, str)(status)


def rule(title: str = "") -> None:
    width = shutil.get_terminal_size((100, 24)).columns
    if not title:
        print(dim("-" * width))
        return
    bar = "-" * max(4, width - len(title) - 3)
    print(f"{bold(title)} {dim(bar)}")


def wrap(text: str, indent: str = "    ", width: int | None = None) -> str:
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 100)
    return textwrap.fill(
        " ".join(text.split()), width=width,
        initial_indent=indent, subsequent_indent=indent,
    )


def excerpt(text: Any, limit: int = 240) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


# --------------------------------------------------------------------------
# Phase 1 -- role probes
# --------------------------------------------------------------------------


# A fixed synthetic state, so every probe is asked exactly the same thing and
# differences between models are the only variable. It is deliberately mid-run:
# an empty state lets a weak model get away with generic text, while a state
# with a plan and a report in it demands the model actually respond to what is
# in front of it.
def probe_state(role: str) -> dict[str, Any]:
    from langgraph_agent.state import initial_state

    state: dict[str, Any] = {
        **initial_state(
            "Add a retry with backoff to the HTTP client used by the "
            "indexer, and cover it with a test."
        ),
        "architecture": "Keep the retry inside the client, not at the call "
                        "sites. Bound total attempts; no unbounded loops.",
        "step_count": 1,
    }
    if role in ("researcher", "builder"):
        state["plan"] = (
            "1. Add a `_retry` helper to the HTTP client.\n"
            "2. Wrap the request call in it, with a bounded attempt count.\n"
            "3. Add a test that forces two failures and asserts one success."
        )
    if role == "builder":
        state["research"] = (
            "## Key Findings\nThe client is a thin wrapper over httpx.\n\n"
            "## Status\nready_for_builder"
        )
    return state


# The Architect probe runs the gate pass, not the opening one: the gate is the
# harder ruling and the one that ends the run, so it is where a weak seat
# actually costs something.
def gate_probe_state() -> dict[str, Any]:
    state = probe_state("architect")
    state["plan"] = "1. Add a `_retry` helper.\n2. Wrap the request call."
    # This report must actually satisfy the goal, every clause of it. The
    # first version claimed only "a cap of 3 attempts" while the goal asked
    # for retry *with backoff*, and both careful models correctly ruled
    # `revise` -- the probe then recorded them as gates that never approve.
    # Grading judgment against a fixture that does not deserve approval
    # punishes exactly the models worth seating. So: backoff named with real
    # delays, the architecture's constraints met, and a test that asserts the
    # delays rather than merely passing.
    state["builder_report"] = (
        "- Added `_retry(fn, attempts=3, backoff=0.5)` to client.py. It sleeps "
        "backoff * 2 ** (n - 1) between attempts and stops after 3, so the "
        "loop is bounded.\n"
        "- Wrapped the client's own request call in it; no call site changed.\n"
        "- Added test_retry.py: forces two failures, asserts one success, and "
        "asserts the recorded delays were 0.5s and 1.0s. It passes.\n"
        "Tool calls:\n- filesystem_write(client.py) -> ok\n"
        "- filesystem_write(test_retry.py) -> ok\n"
        "- test_run(test_retry.py) -> ok, 1 passed"
    )
    state["files_changed"] = ["client.py", "test_retry.py"]
    return state


# The gate's other half. One fixture cannot tell a good gate from a model
# stuck on one answer: rule `approved` on a finished report and you look
# decisive, but only until you rule it on an unfinished one too. So this is
# the same pass with the work plainly not done -- nothing written, an
# explicit blocker, no plan step completed.
#
# Deliberately no `failed_verification`: that list makes `architect_node`
# rewrite `approved` to `revise` on its own, so a fixture using it would
# measure the framework's safety net rather than the seat's judgment. Here
# nothing but the model stands between the run and a premature `approved`.
def blocked_gate_probe_state() -> dict[str, Any]:
    state = probe_state("architect")
    state["plan"] = "1. Add a `_retry` helper.\n2. Wrap the request call."
    state["builder_report"] = (
        "- Could not start: the HTTP client module could not be located.\n"
        "- No files were written and no tests were run.\n"
        "Tool calls:\n- filesystem_read(client.py) -> error: no such file"
    )
    state["files_changed"] = []
    state["blockers"] = (
        "Cannot find the HTTP client module named in the plan. Nothing was "
        "implemented."
    )
    return state


@contextmanager
def _forced_llm_research(nodes: Any) -> Iterator[None]:
    """Make the Researcher probe take the model path.

    `_gather_research` prefers GraphRAG and only calls the model when
    retrieval comes back thin -- so against a healthy corpus the probe would
    grade the corpus, not the seat. Stubbing retrieval to nothing is what
    makes this a probe of the model.
    """
    original = nodes._call_tool
    nodes._call_tool = lambda *a, **k: {"results": []}
    try:
        yield
    finally:
        nodes._call_tool = original


def run_probe(candidate: Candidate, role: str, mods: dict[str, Any]) -> ProbeResult:
    """One call, through the real prompt and the real parser."""
    nodes, config = mods["nodes"], mods["config"]

    config.set_agent_llm(role, candidate.provider, candidate.model)
    config._seat_failures.pop(role, None)

    result = ProbeResult(
        model_key=candidate.key, provider=candidate.provider,
        model=candidate.model, role=role, status="error",
        seconds=0.0, chars=0, detail="",
    )

    # A seat with no usable credentials silently becomes StubLLM, which answers
    # every probe with canned text. Reporting that as `ok` would be the same
    # class of lie the Researcher guard exists to stop.
    status = config.get_agent_status(role)
    if status.get("stubbed"):
        result.status = "stubbed"
        result.detail = status.get("reason") or "No credentials; running StubLLM"
        return result

    started = time.monotonic()
    try:
        if role == "architect":
            # Two passes, because the gate is the only seat whose failure is
            # invisible in its own output. A verdict that parses tells you
            # nothing about whether it was the right one, and this seat is
            # what ends the run: rule `approved` too readily and a run stops
            # having produced nothing, rule it never and the run cycles to the
            # step ceiling. Only asking twice separates judgment from a model
            # that has one answer.
            done = nodes._rule_on_state(gate_probe_state(), reviewing=True)
            blocked = nodes._rule_on_state(
                blocked_gate_probe_state(), reviewing=True)

            text = done.get("architecture", "")
            verdict = str(done.get("verdict", ""))
            blocked_verdict = str(blocked.get("verdict", ""))
            result.parsed = {"verdict": verdict,
                             "verdict_when_blocked": blocked_verdict}
            result.chars = len(str(text))
            result.excerpt = excerpt(text)

            valid = {v.value for v in mods["Verdict"]}
            approved = mods["Verdict"].APPROVED.value
            if not str(text).strip() and not verdict:
                result.status, result.detail = "empty", "No architecture, no verdict"
            elif verdict not in valid:
                result.status = "malformed"
                result.detail = f"Verdict {verdict!r} is not one of {sorted(valid)}"
            elif blocked_verdict not in valid:
                result.status = "malformed"
                result.detail = (f"Verdict {blocked_verdict!r} on the blocked "
                                 f"report is not one of {sorted(valid)}")
            elif verdict == approved and blocked_verdict != approved:
                result.status = "ok"
                result.detail = (f"Approved finished work, ruled "
                                 f"{blocked_verdict!r} on blocked work")
            elif verdict == approved and blocked_verdict == approved:
                # The dangerous one, and the reason for the second fixture: a
                # gate that approves regardless ends runs that produced
                # nothing, and it looks decisive doing it.
                result.status = "rubber"
                result.detail = ("Approved a report that wrote nothing and "
                                 "declared a blocker -- this gate would end a "
                                 "run that did no work")
            elif verdict != approved and blocked_verdict != approved:
                result.status = "cautious"
                result.detail = (f"Ruled {verdict!r} on finished work too -- "
                                 f"never approves, so runs cycle to the step "
                                 f"ceiling")
            else:
                result.status = "inverted"
                result.detail = (f"Ruled {verdict!r} on finished work but "
                                 f"{blocked_verdict!r} on blocked work")

        elif role == "planner":
            parsed = nodes._make_plan(probe_state("planner"))
            plan = str(parsed.get("plan", ""))
            nxt = parsed.get("next_agent", "")
            result.parsed = {"next_agent": nxt}
            result.chars, result.excerpt = len(plan), excerpt(plan)
            if not plan.strip():
                result.status, result.detail = "empty", "No plan"
            elif nxt not in ("Researcher", "Builder"):
                result.status = "malformed"
                result.detail = f"Routed to {nxt!r}, which is not a seat"
            else:
                steps = sum(1 for ln in plan.splitlines() if ln.strip())
                result.status = "ok"
                result.detail = f"{steps} plan lines, routed to {nxt}"

        elif role == "researcher":
            with _forced_llm_research(nodes):
                findings, rstatus = nodes._gather_research(probe_state("researcher"))
            findings = str(findings)
            result.chars, result.excerpt = len(findings), excerpt(findings)
            result.parsed = {"research_status": rstatus}
            sections = nodes._parse_researcher_output(findings)
            filled = [
                s for s in ("key_findings", "relevant_context", "recommendations")
                if str(sections.get(s, "")).strip()
            ]
            # The status is the authoritative signal, and it has to be read
            # before the text is. When the seat says nothing, `_gather_research`
            # does not return nothing -- it returns `_RESEARCH_EMPTY`, a
            # fully-formed three-section fallback explaining that the seat said
            # nothing. Grading the text alone therefore scored the framework's
            # own apology as the seat's answer: `_said_nothing` was False, all
            # three sections were filled, and a silent Researcher came back
            # `ok`. That is the precise failure this probe exists to catch,
            # reported as its opposite.
            if rstatus == nodes._SEAT_EMPTY:
                result.status = "empty"
                result.detail = ("Answered with nothing usable -- this seat "
                                 "would loop the run (text shown is the "
                                 "framework's fallback, not the model's)")
            # Kept as a second net: if the node ever hands back the seat's own
            # empty answer instead of the fallback, this still catches it.
            elif nodes._said_nothing(findings, sections):
                result.status = "empty"
                result.detail = ("Answered with nothing usable -- this seat "
                                 "would loop the run")
            elif len(filled) < 3:
                result.status = "malformed"
                result.detail = f"Only {len(filled)}/3 sections filled: {filled}"
            else:
                result.status = "ok"
                result.detail = f"3/3 sections, status {rstatus!r}"

        elif role == "builder":
            llm = config.get_agent_llm("builder")
            try:
                tool_llm = llm.bind_tools(nodes.BUILDER_TOOLS)
            except AttributeError:
                result.status = "malformed"
                result.detail = "Model cannot bind tools; it could never write a file"
                result.seconds = time.monotonic() - started
                return result

            from langchain_core.messages import HumanMessage, SystemMessage

            response = tool_llm.invoke([
                SystemMessage(content=nodes.seat_prompt("builder")),
                HumanMessage(content=(
                    "Write a file `retry_helper.py` containing a `retry(fn, "
                    "attempts=3)` function. Call the filesystem_write tool to "
                    "create it. Do not describe the file; write it."
                )),
            ])
            calls = list(getattr(response, "tool_calls", []) or [])
            text = nodes._as_text(getattr(response, "content", ""))
            result.chars, result.excerpt = len(text), excerpt(text)
            names = [c.get("name", "?") for c in calls]
            result.parsed = {"tool_calls": names}
            offered = {
                t["function"]["name"] for t in nodes.BUILDER_TOOLS
            }
            unknown = [n for n in names if n not in offered]
            if not calls:
                # The distinction that matters: a Builder that only narrates
                # never appends to `files_changed`, so it reports work it did
                # not do and the gate has nothing to rule on.
                result.status = "empty"
                result.detail = "Described the file instead of calling a tool"
            elif unknown:
                result.status = "malformed"
                result.detail = f"Called tools that are not offered: {unknown}"
            else:
                result.status = "ok"
                result.detail = f"Called {', '.join(names)}"

    except Exception as exc:
        result.status = "error"
        result.detail = f"{type(exc).__name__}: {excerpt(exc, 160)}"
    finally:
        result.seconds = time.monotonic() - started
        result.failure_reason = config._seat_failures.get(role, "")

    return result


def phase_probes(
    candidates: list[Candidate], roles: list[str], mods: dict[str, Any]
) -> list[ProbeResult]:
    print()
    rule("PHASE 1  role probes")
    print(wrap(
        "One call per model per role, through the real prompt and the real "
        "parser. `ok` means the seat answered and the answer parsed. `empty` "
        "is the failure that loops a run: the seat returns nothing and the "
        "framework has to notice.", indent="  "))
    print()

    results: list[ProbeResult] = []
    total = len(candidates) * len(roles)
    n = 0
    for cand in candidates:
        print(f"  {bold(cand.key)} {dim(cand.model)}")
        for role in roles:
            n += 1
            print(f"    {n:>2}/{total} {role:<11}", end=" ", flush=True)
            res = run_probe(cand, role, mods)
            results.append(res)
            line = f"{paint(res.status):<9} {res.seconds:6.1f}s {res.chars:>6}ch"
            print(f"{line}  {dim(res.detail)}")
            if res.failure_reason:
                print(f"{'':>19}{red('seat failure')}: {res.failure_reason}")
            if res.excerpt and res.status != "ok":
                print(dim(wrap(f"got: {res.excerpt}", indent=" " * 19)))
        print()
    return results


def probe_matrix(results: list[ProbeResult], roles: list[str]) -> None:
    """The matrix is the point of phase 1: read it down a column to pick a seat."""
    rule("probe matrix")
    keys: list[str] = []
    for r in results:
        if r.model_key not in keys:
            keys.append(r.model_key)
    index = {(r.model_key, r.role): r for r in results}

    head = f"  {'model':<12}" + "".join(f"{role[:11]:<14}" for role in roles)
    print(bold(head))
    for key in keys:
        cells = []
        for role in roles:
            r = index.get((key, role))
            if r is None:
                cells.append(f"{dim('-'):<14}")
                continue
            mark = {"ok": "ok", "empty": "EMPTY", "malformed": "malf",
                    "error": "ERR", "stubbed": "stub", "cautious": "cautn",
                    "rubber": "RUBBR", "inverted": "INVRT"}.get(
                        r.status, r.status)
            cells.append(f"{paint(mark)} {r.seconds:>5.1f}s".ljust(
                14 + (len(paint(mark)) - len(mark))))
        print(f"  {key:<12}" + "".join(cells))
    print()


# --------------------------------------------------------------------------
# Phase 2 -- team runs
# --------------------------------------------------------------------------


@contextmanager
def sandbox(tag: str, root: Path) -> Iterator[Path]:
    """Run a config where the Builder's writes cannot reach the project.

    The Builder resolves tool paths against the process cwd, so containment is
    a chdir. The directory is a git repo because the Builder has git tools and
    a seat that gets an error from every one of them is being graded on the
    sandbox rather than on itself.
    """
    path = root / tag
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=False,
                   capture_output=True)
    (path / "README.md").write_text(
        "Scratch working tree for a diagnostic run.\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=False,
                   capture_output=True)
    subprocess.run(["git", "-c", "user.email=d@x", "-c", "user.name=diag",
                    "commit", "-qm", "base"], cwd=path, check=False,
                   capture_output=True)

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def instrumented_graph(mods: dict[str, Any], sink: list[NodeVisit]) -> Any:
    """Compile the real graph with each node timed.

    `graph.py` imports the node functions by name, so the wrappers are
    installed on the graph module rather than on `nodes` -- patching the
    latter would leave the compiled graph holding the originals and every
    timing would read zero.
    """
    graph_mod = mods["graph_mod"]
    originals = {}
    for role in AGENTS:
        attr = f"{role}_node"
        originals[attr] = getattr(graph_mod, attr)

    def wrap_node(attr: str, fn: Any, label: str) -> Any:
        def wrapped(state: dict[str, Any]) -> dict[str, Any]:
            started = time.monotonic()
            try:
                return fn(state)
            finally:
                elapsed = time.monotonic() - started
                msgs = state.get("messages") or []
                sink.append(NodeVisit(label, elapsed,
                                      str(msgs[-1]) if msgs else ""))
        return wrapped

    try:
        for role in AGENTS:
            attr = f"{role}_node"
            setattr(graph_mod, attr,
                    wrap_node(attr, originals[attr], role))
        return graph_mod.create_agent_graph()
    finally:
        # Restored immediately: the graph is compiled with the wrappers already
        # captured, so leaving them installed would only stack another layer on
        # the next config.
        for attr, fn in originals.items():
            setattr(graph_mod, attr, fn)


def score_run(res: TeamResult, exercise: Exercise, budget: float) -> None:
    """A transparent score, so a ranking can be argued with.

    Every part is shown in the report. The weights say what this project
    values: finishing on the gate's own verdict, doing it in few cycles, and
    not handing empty state between seats.
    """
    parts: dict[str, float] = {}
    parts["finished"] = 40.0 if res.outcome == "approved" else 0.0
    parts["produced"] = 0.0
    if exercise.expect_files:
        parts["produced"] = 20.0 if res.files_changed else 0.0
    else:
        parts["produced"] = 20.0 if res.field_chars.get("builder_report", 0) > 200 else 0.0
    parts["research"] = 10.0 if res.field_chars.get("research", 0) > 200 else 0.0
    # Cycles and empty handoffs are the thing being hunted, so they subtract
    # rather than merely failing to add.
    parts["cycles"] = -6.0 * max(0, res.gate_passes - 1)
    parts["empty_handoffs"] = -8.0 * res.empty_handoffs
    parts["unverified"] = -10.0 if res.failed_verification else 0.0
    # Scored against the budget the run was actually given, not a fixed
    # divisor. `20 - seconds/15` reached zero at 300s, which used to be the
    # entire budget: every run that used its time scored the same nothing, so
    # the term stopped separating a brisk run from one that only just fit.
    # Raising the budget would have retired it silently rather than visibly.
    parts["speed"] = (
        round(20.0 * max(0.0, 1.0 - res.seconds / budget), 1) if budget > 0 else 0.0
    )
    res.score_parts = {k: round(v, 1) for k, v in parts.items()}
    res.score = round(sum(parts.values()), 1)


def run_team(
    cfg: TeamConfig, exercise: Exercise, budget: float,
    mods: dict[str, Any], sandbox_root: Path, verbose: bool,
) -> TeamResult:
    config, control = mods["config"], mods["control"]

    seats = {role: BY_KEY[key].model for role, key in cfg.seats.items()}
    res = TeamResult(config=cfg.name, exercise=exercise.name, seats=dict(seats))

    config._seat_failures.clear()
    for role, key in cfg.seats.items():
        cand = BY_KEY[key]
        config.set_agent_llm(role, cand.provider, cand.model)

    visits: list[NodeVisit] = []
    graph = instrumented_graph(mods, visits)

    from langgraph_agent.state import initial_state

    state: dict[str, Any] = dict(initial_state(exercise.goal))

    run_id = uuid.uuid4().hex
    control.RUN_CONTROL.arm(run_id)
    started = time.monotonic()
    last = state
    seen = 0

    try:
        with sandbox(f"{cfg.name}-{exercise.name}", sandbox_root) as box:
            res.sandbox = str(box)
            try:
                for event in graph.stream(
                    state, {"recursion_limit": mods["RECURSION_LIMIT"]}
                ):
                    for _node, node_state in event.items():
                        if not isinstance(node_state, dict):
                            continue
                        last = node_state
                        if verbose:
                            for visit in visits[seen:]:
                                msg = excerpt(visit.message, 110)
                                print(f"      {cyan(visit.node):<20} "
                                      f"{visit.seconds:6.1f}s  {dim(msg)}")
                            seen = len(visits)

                    # Between supersteps, exactly where serve.py checks: a node
                    # in flight never reaches a boundary, so this bounds the
                    # run and never the node.
                    if time.monotonic() - started > budget:
                        res.outcome = "budget"
                        control.RUN_CONTROL.stop(
                            run_id, f"Diagnostic budget of {int(budget)}s reached.")
                        break
            except Exception as exc:
                res.outcome = "error"
                res.error = f"{type(exc).__name__}: {excerpt(exc, 300)}"
                if verbose:
                    print(red(wrap(traceback.format_exc(limit=3), indent="      ")))
    finally:
        control.RUN_CONTROL.disarm()
        res.seconds = time.monotonic() - started

    if verbose:
        for visit in visits[seen:]:
            print(f"      {cyan(visit.node):<20} {visit.seconds:6.1f}s  "
                  f"{dim(excerpt(visit.message, 110))}")

    res.visits = visits
    res.steps = int(last.get("step_count", 0))
    res.verdict = str(last.get("verdict", ""))
    res.files_changed = list(last.get("files_changed", []))
    res.failed_verification = list(last.get("failed_verification", []))
    res.blockers = str(last.get("blockers", ""))
    res.messages = [str(m) for m in last.get("messages", [])]
    res.seat_failures = dict(config._seat_failures)

    for role in AGENTS:
        role_visits = [v for v in visits if v.node == role]
        res.role_calls[role] = len(role_visits)
        res.role_seconds[role] = round(sum(v.seconds for v in role_visits), 1)

    for fieldname in ("architecture", "plan", "research", "builder_report"):
        res.field_chars[fieldname] = len(str(last.get(fieldname, "")))

    # A gate pass that did not end the run is a cycle: the Architect looked at
    # the work and sent it back. One is normal. Several, with nothing being
    # produced in between, is the hand-off loop this whole script exists for.
    res.gate_passes = max(0, res.role_calls.get("architect", 0) - 1)
    res.empty_handoffs = sum(
        1 for m in res.messages
        if "returned no findings" in m or "without research" in m
    )

    if res.outcome == "not-run":
        if res.verdict == "approved":
            res.outcome = "approved"
        elif res.steps >= mods["MAX_STEPS"]:
            res.outcome = "ceiling"
        else:
            res.outcome = "unfinished"

    score_run(res, exercise, budget)
    return res


def phase_teams(
    configs: list[TeamConfig], exercises: list[Exercise], budget: float,
    mods: dict[str, Any], sandbox_root: Path, verbose: bool,
) -> list[TeamResult]:
    print()
    rule("PHASE 2  team runs")
    print(wrap(
        "Each configuration runs the same exercise through the same graph the "
        "console drives, in its own sandbox. Watch the cycles column: a run "
        "that passes the gate repeatedly is the back-and-forth, and a run that "
        "does it while producing nothing is the failure worth acting on.",
        indent="  "))
    print()

    results: list[TeamResult] = []
    for cfg in configs:
        for ex in exercises:
            seat_line = "  ".join(f"{r[:4]}={cfg.seats[r]}" for r in AGENTS)
            rule(f"{cfg.name} / {ex.name}")
            print(f"    {dim(seat_line)}")
            print(wrap(cfg.rationale, indent="    "))
            print()
            res = run_team(cfg, ex, budget, mods, sandbox_root, verbose)
            results.append(res)
            print()
            print(f"    outcome  {paint(res.outcome)}   "
                  f"steps {res.steps}   cycles {res.gate_passes}   "
                  f"{res.seconds:.1f}s   score {bold(str(res.score))}")
            print(f"    produced  files={res.files_changed or '[]'}  "
                  f"research={res.field_chars.get('research', 0)}ch  "
                  f"report={res.field_chars.get('builder_report', 0)}ch")
            if res.empty_handoffs:
                print(f"    {red('empty handoffs')}: {res.empty_handoffs} "
                      f"(a seat handed on nothing)")
            if res.failed_verification:
                print(f"    {red('unverified')}: {res.failed_verification}")
            if res.seat_failures:
                for role, reason in res.seat_failures.items():
                    print(f"    {red('seat failure')} {role}: {reason}")
            if res.error:
                print(f"    {red('error')}: {res.error}")
            print()
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def team_table(results: list[TeamResult]) -> None:
    rule("team results")
    head = (f"  {'config':<20}{'exercise':<10}{'outcome':<12}{'steps':>6}"
            f"{'cycles':>8}{'files':>7}{'secs':>8}{'score':>8}")
    print(bold(head))
    for r in sorted(results, key=lambda x: -x.score):
        print(f"  {r.config:<20}{r.exercise:<10}"
              f"{paint(r.outcome):<12}{r.steps:>6}{r.gate_passes:>8}"
              f"{len(r.files_changed):>7}{r.seconds:>8.1f}{r.score:>8.1f}")
    print()


def recommend(
    probes: list[ProbeResult], teams: list[TeamResult]
) -> tuple[dict[str, str], list[str]]:
    """Pick a seating from the evidence, and say what the evidence does not cover."""
    notes: list[str] = []
    best: dict[str, str] = {}

    # A role nobody probed is not a role every model failed. Reporting the two
    # the same way had `--phase teams` print "No model passed the architect
    # probe" for a phase that was never run -- an invented failure, which is
    # the one thing a diagnostic must not produce.
    probed_roles = {p.role for p in probes}
    if not probes:
        notes.append("Phase 1 did not run, so there is no per-seat "
                     "recommendation here -- these are team results only.")

    for role in AGENTS:
        if role not in probed_roles:
            continue
        ok = [p for p in probes if p.role == role and p.status == "ok"]
        if not ok:
            notes.append(f"No model passed the {role} probe -- nothing to recommend "
                         f"for that seat.")
            continue
        # Fastest passing seat. Latency is the only quality signal a single
        # probe honestly supports: it says the seat answers in the shape the
        # parser wants, not that its answer is good.
        ok.sort(key=lambda p: p.seconds)
        best[role] = ok[0].model_key
        if len(ok) > 1:
            notes.append(
                f"{role}: {len(ok)} models passed; picked {ok[0].model_key} on "
                f"latency ({ok[0].seconds:.1f}s vs {ok[1].seconds:.1f}s for "
                f"{ok[1].model_key}). Probe latency is not answer quality.")

    if teams:
        ranked = sorted(teams, key=lambda t: -t.score)
        top = ranked[0]
        notes.append(f"Best measured seating was {top.config} "
                     f"(score {top.score}, {top.outcome}).")
        if len({t.config for t in teams}) > 1 and len(ranked) > 1:
            notes.append(f"Runner-up {ranked[1].config} scored {ranked[1].score}; "
                         f"one exercise per config is a data point, not a ranking.")
    return best, notes


def write_report(
    out_dir: Path, probes: list[ProbeResult], teams: list[TeamResult],
    roles: list[str], meta: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta,
        "probes": [asdict(p) for p in probes],
        "teams": [asdict(t) for t in teams],
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best, notes = recommend(probes, teams)

    lines: list[str] = [
        "# Seat diagnostics", "",
        f"- Run at: {meta['started_at']}",
        f"- Exercises: {', '.join(meta['exercises'])}",
        f"- Budget per run: {meta['budget']}s",
        f"- Node deadline {meta['node_deadline']}s / "
        f"Builder deadline {meta['builder_deadline']}s / "
        f"tool turns {meta['tool_turns']}", "",
    ]

    corpus = meta.get("corpus_documents")
    if corpus is not None:
        lines.insert(5, f"- Corpus: {corpus} documents")
    if corpus == 0:
        lines += [
            "> **The corpus was empty for these runs.** Every Researcher fell "
            "back to its model, which reads exactly like a bad Researcher "
            "seat. These exercises call the seats directly rather than through "
            "`rpc_run_goal`, so nothing built the corpus for them: start one "
            "run from the console and re-run this before drawing any "
            "conclusion about retrieval or about the `research` exercise.",
            "",
        ]

    if probes:
        lines += ["## Role probes", "",
                  "One call per model per role, through the real prompt and "
                  "parser. `empty` is the failure that loops a run.", "",
                  "| model | " + " | ".join(roles) + " |",
                  "|---|" + "---|" * len(roles)]
        keys: list[str] = []
        for p in probes:
            if p.model_key not in keys:
                keys.append(p.model_key)
        index = {(p.model_key, p.role): p for p in probes}
        for key in keys:
            cells = []
            for role in roles:
                p = index.get((key, role))
                cells.append("-" if p is None
                             else f"{p.status} ({p.seconds:.1f}s)")
            lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
        lines.append("")

        failures = [p for p in probes if p.status != "ok"]
        if failures:
            lines += ["### Probe failures", ""]
            for p in failures:
                lines.append(f"- **{p.model_key} / {p.role}** — `{p.status}`: "
                             f"{p.detail}")
                if p.failure_reason:
                    lines.append(f"  - seat failure: {p.failure_reason}")
            lines.append("")

    if teams:
        lines += ["## Team runs", "",
                  "| config | exercise | outcome | steps | cycles | files | "
                  "secs | score |", "|---|---|---|---|---|---|---|---|"]
        for t in sorted(teams, key=lambda x: -x.score):
            lines.append(
                f"| `{t.config}` | {t.exercise} | {t.outcome} | {t.steps} | "
                f"{t.gate_passes} | {len(t.files_changed)} | {t.seconds:.1f} | "
                f"{t.score} |")
        lines.append("")
        lines += ["### Per-run detail", ""]
        for t in sorted(teams, key=lambda x: -x.score):
            lines += [
                f"#### `{t.config}` / {t.exercise} — {t.outcome} "
                f"(score {t.score})", "",
                "Seats: " + ", ".join(f"{r}=`{t.seats[r]}`" for r in AGENTS), "",
                f"- Verdict: `{t.verdict or '(none)'}`, steps {t.steps}, "
                f"gate cycles {t.gate_passes}, {t.seconds:.1f}s",
                "- Field sizes: " + ", ".join(
                    f"{k}={v}ch" for k, v in t.field_chars.items()),
                "- Time per seat: " + ", ".join(
                    f"{k}={v}s×{t.role_calls.get(k, 0)}"
                    for k, v in t.role_seconds.items()),
                f"- Files: {t.files_changed or 'none'}",
            ]
            if t.failed_verification:
                lines.append(f"- Unverified: {t.failed_verification}")
            if t.empty_handoffs:
                lines.append(f"- Empty handoffs: {t.empty_handoffs}")
            if t.seat_failures:
                for role, reason in t.seat_failures.items():
                    lines.append(f"- Seat failure `{role}`: {reason}")
            if t.error:
                lines.append(f"- Error: `{t.error}`")
            lines += ["- Score parts: " + ", ".join(
                f"{k} {v:+}" for k, v in t.score_parts.items()), "",
                "<details><summary>Feed</summary>", ""]
            lines += [f"    {m}" for m in t.messages]
            lines += ["", "</details>", ""]

    lines += ["## Reading this", ""]
    if best:
        lines.append("Probe-passing seating, fastest first: " + ", ".join(
            f"{role}=`{key}`" for role, key in best.items()))
        lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines += [
        "",
        "Caveats that matter more than the numbers:",
        "",
        "- A probe says a seat answers in the shape the parser wants. It does "
        "not say the answer is correct, and latency is not quality.",
        "- One exercise per configuration is a single sample against "
        "non-deterministic models. Re-run before trusting a small gap.",
        "- Team runs happen in a sandbox with the real corpus but an empty "
        "working tree, so a Builder that would have edited existing project "
        "files is being asked an easier question here.",
        "",
    ]

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose which model works in which seat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s --list
              %(prog)s --phase probe
              %(prog)s --phase teams --configs baseline,heavy-gate --verbose
              %(prog)s --anthropic --exercise all
        """),
    )
    p.add_argument("--phase", choices=("probe", "teams", "all"), default="all")
    p.add_argument("--models", default="",
                   help="Comma-separated candidate keys for phase 1 "
                        "(default: every free candidate)")
    p.add_argument("--roles", default=",".join(AGENTS),
                   help="Comma-separated roles to probe")
    p.add_argument("--configs", default="",
                   help="Comma-separated team config names "
                        "(default: every free config)")
    p.add_argument("--exercise", default="build",
                   help="build | research | plan | all")
    p.add_argument("--anthropic", action="store_true",
                   help="Include the paid Anthropic candidates and configs")
    p.add_argument("--budget", type=float, default=900.0,
                   help="Wall-clock seconds per team run (default: 900)")
    p.add_argument("--node-deadline", type=float, default=150.0)
    p.add_argument("--builder-deadline", type=float, default=240.0)
    p.add_argument("--llm-timeout", type=float, default=None,
                   help="Seconds one seat call may wait at the socket "
                        "(default: LLM_TIMEOUT_SECONDS, the console's own)")
    p.add_argument("--tool-turns", type=int, default=8,
                   help="Builder tool turns, i.e. MAX_BUILDER_TOOL_TURNS "
                        "(default: 8, the shipped value)")
    p.add_argument("--warm-deadline", type=float, default=120.0,
                   help="Seconds to wait for the corpus to load (default: 120)")
    p.add_argument("--out", default="",
                   help="Report directory (default: reports/diagnostics/<ts>)")
    p.add_argument("--verbose", action="store_true",
                   help="Print every node visit as it happens")
    p.add_argument("--list", action="store_true",
                   help="Show candidates, configs and exercises, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run without spending a call")
    return p.parse_args(argv)


def show_catalogue() -> None:
    rule("candidates")
    # The model column is sized to the longest tag rather than fixed: a local
    # `hf.co/...` tag is over twice the width of a `:cloud` one and ran straight
    # into the note, which read as a note about the wrong model.
    width = max(len(c.model) for c in CANDIDATES) + 2
    for c in CANDIDATES:
        tag = red(" paid") if c.paid else ""
        print(f"  {c.key:<16}{c.provider:<11}{c.model:<{width}}{dim(c.note)}{tag}")
    print()
    rule("team configurations")
    for cfg in TEAM_CONFIGS:
        tag = red(" paid") if cfg.paid else ""
        print(f"  {bold(cfg.name)}{tag}")
        print(f"    {dim('  '.join(f'{r}={cfg.seats[r]}' for r in AGENTS))}")
        print(wrap(cfg.rationale, indent="    "))
    print()
    rule("exercises")
    for ex in EXERCISES.values():
        print(f"  {bold(ex.name)}  {dim('writes files' if ex.expect_files else 'no files')}")
        print(wrap(ex.what_it_tests.replace("{floor}", _retrieval_floor()),
                   indent="    "))
    print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # A sweep is something you watch, and `--verbose` exists to print each node
    # visit as it happens -- but redirected stdout is block-buffered, so piping
    # the run to a log left everything after the first flushed line sitting in
    # an 8KB buffer. A hung warm then looked identical to a slow one: the last
    # thing on screen was the half-line printed before it. `serve.py` does the
    # same thing for the same reason.
    # (typed as TextIO, which does not declare reconfigure; it is a
    # TextIOWrapper at runtime whenever stdout is a real stream.)
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    if args.list:
        show_catalogue()
        return 0

    # Deadlines are module constants read from the environment at import time,
    # so they are set before `nodes` is imported below. `config` is imported at
    # the top of this file, so its socket timeout is set on the module instead
    # (`--llm-timeout`), which every seat call reads when it is built.
    #
    # These started tighter than the shipped 150s/240s, on the theory that a
    # sweep should be watchable and a wedged seat should not hold it for four
    # minutes. Measurement beat that theory. At a 60s node deadline the qwen
    # Planner was cut twice in one sweep while working normally -- it returned
    # a plan in 8.6s and 27.2s on other passes -- and a cut seat is recorded as
    # a seat failure. An impatient diagnostic does not report a slow seat as
    # slow; it reports a working seat as broken, which is the same class of
    # false verdict as scoring a silent Researcher `ok`.
    #
    # So every deadline here now equals the shipped default -- node 150s,
    # Builder 240s, and the 60s verification reserve that goes with it. That
    # is the point rather than a coincidence: a seat is being measured for a
    # console that will run it under those numbers, so measuring it under
    # tighter ones answers a question nobody asked. The budget is what keeps a
    # sweep bounded, and it is the knob to turn.
    #
    # `--tool-turns` belongs to the same set. Running the Builder at 5 turns
    # where the console gives it 8 makes it hit the cap and set the "stopped
    # after N tool turns without finishing" blocker on work it would have
    # finished, which is a failure the diagnostic invented -- the same shape
    # as an under-sized verification reserve.
    os.environ["NODE_DEADLINE_SECONDS"] = str(args.node_deadline)
    os.environ["BUILDER_DEADLINE_SECONDS"] = str(args.builder_deadline)
    # Held back from the Builder's tool loop so the verification pass gets to
    # run at all. It was cut to 25s to match a shortened Builder deadline; now
    # that the deadline is back to the shipped 240s this goes back with it. A
    # reserve too small for the pass leaves files `unverified`, and unverified
    # blocks approval -- so an under-sized reserve invents blockers rather than
    # merely saving time.
    os.environ["VERIFY_RESERVE_SECONDS"] = "60"

    quiet_logs()

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from langgraph_agent import config, control, nodes
    from langgraph_agent import graph as graph_mod
    from langgraph_agent.state import Verdict

    nodes.MAX_BUILDER_TOOL_TURNS = args.tool_turns
    if args.llm_timeout is not None:
        config.LLM_TIMEOUT_SECONDS = args.llm_timeout

    mods: dict[str, Any] = {
        "config": config, "control": control, "nodes": nodes,
        "graph_mod": graph_mod, "Verdict": Verdict,
        "RECURSION_LIMIT": graph_mod.RECURSION_LIMIT,
        "MAX_STEPS": graph_mod.MAX_STEPS,
    }

    # Selections
    candidates = [c for c in CANDIDATES if args.anthropic or not c.paid]
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in BY_KEY]
        if unknown:
            print(red(f"Unknown model keys: {unknown}. Try --list."))
            return 2
        candidates = [BY_KEY[m] for m in wanted]

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    bad_roles = [r for r in roles if r not in AGENTS]
    if bad_roles:
        print(red(f"Unknown roles: {bad_roles}"))
        return 2

    configs = [c for c in TEAM_CONFIGS if args.anthropic or not c.paid]
    if args.configs:
        wanted_cfg = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown_cfg = [c for c in wanted_cfg if c not in CONFIGS_BY_NAME]
        if unknown_cfg:
            print(red(f"Unknown configs: {unknown_cfg}. Try --list."))
            return 2
        configs = [CONFIGS_BY_NAME[c] for c in wanted_cfg]

    if args.exercise == "all":
        exercises = list(EXERCISES.values())
    else:
        names = [e.strip() for e in args.exercise.split(",") if e.strip()]
        unknown_ex = [e for e in names if e not in EXERCISES]
        if unknown_ex:
            print(red(f"Unknown exercises: {unknown_ex}"))
            return 2
        exercises = [EXERCISES[e] for e in names]

    do_probe = args.phase in ("probe", "all")
    do_teams = args.phase in ("teams", "all")

    # The architect probe makes two calls, not one: the gate is asked to rule
    # on finished work and on blocked work, because one answer cannot be told
    # from a fixed answer. See run_probe.
    calls_per_model = len(roles) + (1 if "architect" in roles else 0)
    probe_calls = len(candidates) * calls_per_model if do_probe else 0
    team_runs = len(configs) * len(exercises) if do_teams else 0

    rule("plan")
    print(f"  phase          {args.phase}")
    print(f"  probes         {probe_calls} calls "
          f"({len(candidates)} models x {len(roles)} roles"
          f"{'; the gate is asked twice' if 'architect' in roles else ''})")
    print(f"  team runs      {team_runs} "
          f"({len(configs)} configs x {len(exercises)} exercises), "
          f"budget {args.budget:.0f}s each")
    print(f"  worst case     ~{(team_runs * args.budget) / 60:.0f} min of runs "
          f"plus probes")
    print(f"  deadlines      node {args.node_deadline:.0f}s / "
          f"builder {args.builder_deadline:.0f}s / turns {args.tool_turns} / "
          f"socket {config.LLM_TIMEOUT_SECONDS:.0f}s")
    print(f"  paid seats     {'included' if args.anthropic else 'excluded'}")
    if do_teams:
        print(f"  configs        {', '.join(c.name for c in configs)}")
    print()

    if args.dry_run:
        print(dim("  --dry-run: nothing was called."))
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else (
        PROJECT_ROOT / "reports" / "diagnostics" / stamp)
    sandbox_root = Path(
        os.getenv("DIAG_SANDBOX_ROOT", "")
        or Path(os.getenv("TMPDIR", "/tmp")) / f"seat-diag-{stamp}"
    )

    # Warmed here, at the project root, on purpose: the singleton resolves
    # `./knowledge` on first use, and team runs execute inside a sandbox cwd.
    # Without this the first config would bind the corpus to its own empty
    # sandbox and every Researcher after it would search nothing -- which
    # looks exactly like a bad Researcher model.
    corpus_documents: int | None = None
    if do_teams:
        print(dim("  warming the corpus (loads the embedding model)..."), end=" ",
              flush=True)
        warm_started = time.monotonic()

        def _warm() -> int:
            # `open_knowledge_base`, so a sweep that only meant to measure
            # seats does not leave a corpus behind on a machine that had none.
            # No corpus counts as zero documents, which is what the warning
            # below is already about.
            from langgraph_agent.graphrag_server import open_knowledge_base
            kb = open_knowledge_base()
            return int(kb.collection.count()) if kb is not None else 0

        try:
            # Bounded for the same reason every node is. Opening the store can
            # mean the daemon loading the embedding model, which nothing here
            # times on its own: one hung sweep sat here for ten minutes
            # having printed half a line, with nothing to say what it was
            # waiting for. A warm that never returns must not cost the whole
            # sweep. If it does time out the singleton may still be coming up
            # behind us, and the node deadlines are the backstop for that --
            # the run is told the corpus is unknown rather than told it is
            # empty, because those are different things.
            corpus_documents = nodes._with_deadline(_warm, args.warm_deadline, None)
            if corpus_documents is None:
                print(yellow(f"timed out after {args.warm_deadline:.0f}s"))
                print(wrap(yellow(
                    "The corpus never came up, so retrieval is unknown rather "
                    "than known-empty. Team runs continue and the node "
                    "deadlines still bound them, but do not read a thin "
                    "Researcher in this sweep as a bad seat. Try "
                    "HF_HUB_OFFLINE=1 if the embedding model is already "
                    "cached."), indent="  "))
            else:
                print(dim(f"{corpus_documents} documents, "
                          f"{time.monotonic() - warm_started:.1f}s"))
        except Exception as exc:
            print(yellow(f"failed: {excerpt(exc, 120)}"))
            print(wrap(yellow(
                "Team runs will continue, but the Researcher will fall back to "
                "the model for every query. Start one run from the console "
                "first if you meant to test retrieval."), indent="  "))

        # An empty corpus does not raise: retrieval simply returns nothing, the
        # Researcher falls through to the model, and the run reads exactly like
        # one with a bad Researcher in it. That is the failure this script is
        # most often pointed at, so it must never be the thing the script
        # itself is silently doing to every configuration.
        if corpus_documents == 0:
            print()
            print(wrap(yellow(
                "The corpus is empty, so every Researcher will find nothing and "
                "fall back to its model. That looks identical to a bad "
                "Researcher seat, and the `research` exercise is measuring "
                "nothing. These exercises drive the seats directly, so nothing "
                "here indexes: start one run from the console first if you "
                "meant to test retrieval."), indent="  "))
            print()

    probes: list[ProbeResult] = []
    teams: list[TeamResult] = []
    interrupted = False

    try:
        if do_probe:
            probes = phase_probes(candidates, roles, mods)
            probe_matrix(probes, roles)
        if do_teams:
            teams = phase_teams(configs, exercises, args.budget, mods,
                                sandbox_root, args.verbose)
            team_table(teams)
    except KeyboardInterrupt:
        interrupted = True
        print()
        print(yellow("  Interrupted. Reporting what finished."))
        control.RUN_CONTROL.stop(reason="Interrupted at the terminal.")
        control.RUN_CONTROL.disarm()

    meta = {
        "started_at": stamp,
        "exercises": [e.name for e in exercises] if do_teams else [],
        "budget": args.budget,
        "node_deadline": args.node_deadline,
        "builder_deadline": args.builder_deadline,
        "llm_timeout": config.LLM_TIMEOUT_SECONDS,
        "tool_turns": args.tool_turns,
        "anthropic_included": args.anthropic,
        "interrupted": interrupted,
        "sandbox_root": str(sandbox_root),
        "corpus_documents": corpus_documents,
    }
    json_path, md_path = write_report(out_dir, probes, teams, roles, meta)

    rule("what to do with this")
    best, notes = recommend(probes, teams)
    if best:
        print("  Probe-passing seats, fastest first:")
        for role, key in best.items():
            print(f"    {role:<12}{bold(key):<20}{dim(BY_KEY[key].model)}")
        print()
    for note in notes:
        print(wrap(f"- {note}", indent="  "))
    print()
    print(f"  report   {md_path}")
    print(f"  raw      {json_path}")
    if do_teams:
        print(f"  sandboxes{'':>1}{sandbox_root}  "
              f"{dim('(the Builder wrote here, not in the project)')}")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print()
        raise SystemExit(130) from None
