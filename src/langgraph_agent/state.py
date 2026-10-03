"""State schema for the agent graph.

The 4-Agent System:
- Architect writes: architecture, verdict
- Planner writes: plan, next_agent
- Researcher writes: research, research_status
- Builder writes: builder_report, files_changed, blockers, failed_verification,
  unverified, builder_cut_off, lint_failed
"""

from enum import Enum
from typing import TypedDict


class ResearchStatus(str, Enum):
    """Status from Researcher to guide next steps."""

    READY_FOR_BUILDER = "ready_for_builder"
    NEED_REPLAN = "need_replan"
    NO_RELEVANT_KNOWLEDGE = "no_relevant_knowledge"


class Verdict(str, Enum):
    """The Architect's ruling, which decides where the loop goes next.

    The Architect runs twice per cycle -- once to set direction, once as the
    approval gate -- so PLAN is the opening ruling and the other three are
    what it can say about work the Builder has already reported.
    """

    PLAN = "plan"
    APPROVED = "approved"
    REVISE = "revise"
    NEED_RESEARCH = "need_research"


class AgentState(TypedDict):
    """Shared state passed between all nodes.

    Attributes:
        goal: The user's original goal/objective
        messages: Conversation history / log
        architecture: Architectural direction and constraints from the Architect
        verdict: The Architect's ruling (Verdict enum value)
        plan: Structured plan from Planner
        research: Findings from Researcher
        builder_report: Implementation report from Builder
        next_agent: Which agent runs next ("Researcher" | "Builder" | "END")
        research_status: Status from Researcher (ready_for_builder | need_replan | no_relevant_knowledge)
        blockers: What's blocking progress (set by Builder when stuck)
        discuss_only: Set by the caller, never by an agent. The seats reason
            about the goal and the Builder is offered no tools at all -- no
            read, no write, no terminal, no tests -- so a run cannot change the
            project. The online research phase is off for the same reason: it
            writes pages to disk and embeds them.
        output_dir: Set by the caller, never by an agent. Empty for a run
            working on this checkout; otherwise `projects/<name>`, the one
            directory the Builder's `filesystem_write` may write into. A
            project there stays out of the corpus until the operator embeds it.
        files_changed: List of file paths modified by Builder
        failed_verification: Files the Builder wrote that did not pass
            verification -- they ran and failed, or nobody executed them. Set
            by the Builder every pass and carried into the next, so a file
            clears only by passing, never by omission.
        unverified: The part of failed_verification nobody executed, because
            the deadline or the emergency stop came first. It blocks approval
            whatever expect_failures says: that opt-out is for a file meant to
            fail, not for a gap in the evidence.
        builder_cut_off: Why the Builder's last pass ended before it finished
            -- "turn_cap" or "deadline" -- or could not start: "no_tools", a
            model refused its tools on a run that needed them. Empty when it
            finished. While it is set the Architect cannot approve.
        lint_failed: Python files the Builder wrote or carried that still fail
            `ruff check` after the fixes it is allowed to make. Re-linted every
            pass until clean, and blocking approval whatever expect_failures
            says.
        expect_failures: Per-run opt-out, set by the caller and never by an
            agent. A file that runs and fails is still executed, still reported
            and still listed in failed_verification -- it just stops blocking
            approval and sets no blocker. For goals whose product is a failing
            file: a deliberate fixture, an expected-to-fail test.
        step_count: Number of steps taken (for loop limit)
    """

    goal: str
    messages: list[str]
    architecture: str
    verdict: str  # Verdict enum value
    plan: str
    research: str
    builder_report: str
    next_agent: str  # "Researcher" | "Builder" | "END"
    research_status: str  # ResearchStatus enum value
    blockers: str
    files_changed: list[str]
    failed_verification: list[str]
    unverified: list[str]
    builder_cut_off: str  # "" | "turn_cap" | "deadline" | "no_tools"
    lint_failed: list[str]
    discuss_only: bool
    output_dir: str  # "" | "projects/<name>"
    expect_failures: bool
    step_count: int


def initial_state(
    goal: str,
    *,
    expect_failures: bool = False,
    discuss_only: bool = False,
    output_dir: str = "",
) -> AgentState:
    """The state every run starts from: the goal, the caller's three per-run
    flags, and every field an agent writes empty.

    `research_status` starts empty because it is what marks the Researcher as
    having run, so the opening cycle retrieves before the Builder acts.
    """
    return AgentState(
        goal=goal,
        messages=[],
        architecture="",
        verdict="",
        plan="",
        research="",
        builder_report="",
        next_agent="Researcher",
        research_status="",
        blockers="",
        files_changed=[],
        failed_verification=[],
        unverified=[],
        builder_cut_off="",
        lint_failed=[],
        discuss_only=discuss_only,
        output_dir=output_dir,
        expect_failures=expect_failures,
        step_count=0,
    )
