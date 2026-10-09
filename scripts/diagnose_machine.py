#!/usr/bin/env python3
"""One command that diagnoses this machine and writes a report you can share.

    .venv/bin/python scripts/diagnose_machine.py              # read-only, all of it
    .venv/bin/python scripts/diagnose_machine.py --quick      # no embedder, no seat probes
    .venv/bin/python scripts/diagnose_machine.py --with-runs  # plus one run on the live console

The console fails in ways that look alike from inside it: a seat that is slow
because the daemon split it onto the CPU reads the same as a seat that is slow
because it is big, and a corpus that will not build reads the same whether the
card, the daemon or the database is the cause. The answer is usually on the
machine around the console -- the driver, the daemon's drop-ins, the database
container, a key exported in a shell -- so this walks the machine section by
section, the way CLAUDE.md's troubleshooting would have you walk it by hand,
and writes down what it found and what to do about it.

It changes nothing by default. Every section reads: versions, the daemon's
answers, the database inside a read-only transaction, the console's own read
RPCs, the browser agent's read-only passes. Three sections put load on the
cards -- one embed, one probe per seat, one short generation per local model --
and they are skipped, with the reason, while the console is running a goal,
rebuilding its corpus or finishing a pull request, since they would compete
with it. It never starts the console: one that is down is reported, with how
to start it. `--with-runs` is the one switch that does change something; its
help says what.

What it writes, under `reports/diagnostics/<stamp>/` unless `--out` says
otherwise: report.md (a verdict per section, the problems with their fixes,
then the details), results.json, logs/ (raw outputs), and the seat
diagnostic's and the browser agent's own reports beside them; and next to the
directory, <stamp>-share.tar.gz, which leaves out traces, recordings, downloads
and any .env. Everything written goes through `redact`: keys, tokens, URL
passwords, email addresses and this user's home and runtime paths.

Exit status: 0 when no section found a problem, 1 when one did or could not
finish, 2 when nothing could be diagnosed at all.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "ambiguity-diagnostics/1"

# What a report must never carry, each with the kind the report counts it as.
# Ordered: a provider's own key shape is named before the generic `NAME=value`
# rule could swallow it.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "<anthropic key>"),
    ("api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "<api key>"),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<github token>"),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<github token>"),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "<huggingface token>"),
    ("bearer", re.compile(r"(?i)\b(bearer)[ \t]+(?!<)[A-Za-z0-9._~+/=\-]{8,}"),
     r"\1 <redacted>"),
    ("authorization", re.compile(r"(?i)\b(authorization:[ \t]*(?:basic|token))[ \t]+(?!<)\S+"),
     r"\1 <redacted>"),
)

# `ANTHROPIC_API_KEY=...`, `"api_key": "..."`, `PGPASSWORD: ...` and
# `x-api-key: ...`. Horizontal whitespace only, so an empty value never reaches
# into the next line. A bare value stops at a quote, a backslash or an `&`: in
# JSON-escaped text or a query string, what follows is not part of it, and
# eating a closing quote leaves a string that no longer parses.
_SECRET_NAME = r"[A-Za-z0-9_\-]*(?:key|token|secret|password|passwd)"
_ASSIGNED_SECRET = re.compile(
    rf"(?i)(?<![A-Za-z0-9_\-])({_SECRET_NAME})(\\?[\"']?[ \t]*[=:][ \t]*)"
    r"(\\\"[^\"\n]*?\\\"|\"[^\"\n]*\"|'[^'\n]*'|[^\s,;&}\"'\\]+)"
)
_SECRET_KEY = re.compile(rf"(?i){_SECRET_NAME}")
# Values that say there is no secret, which a report is better off keeping.
_NO_SECRET = frozenset({"none", "null", "nil", "unset", "n/a", "true", "false"})
_CREDENTIAL = re.compile(r"[A-Za-z0-9._~+/=\-]{8,}")


def _names_a_setting(name: str, quoted: bool) -> bool:
    """Whether `name: value` reads as a setting rather than a sentence.

    `ANTHROPIC_API_KEY:`, `x-api-key:`, `"token":`, `PGPASSWORD:` and
    `password:` are settings; `no key: canned stub output` and `monkey: banana`
    are prose, and a seat label garbled into `no key: <redacted>` would both
    hide what the report is for and claim a secret that was never there.
    """
    return (quoted or "_" in name or "-" in name or name.isupper()
            or name.lower() in ("password", "passwd", "secret"))


def _credential_shaped(value: str) -> bool:
    """A word that could be a key: long, token characters only, letters and digits."""
    return (_CREDENTIAL.fullmatch(value) is not None and any(c.isdigit() for c in value)
            and any(c.isalpha() for c in value))


def _says_no_secret(value: str) -> bool:
    return not value or value.startswith("<") or value.strip("()").lower() in _NO_SECRET
_URL_PASSWORD = re.compile(r"\b([a-z][a-z0-9+.\-]*://[^/\s:@]+):([^/\s@]+)@", re.I)
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+\-]+)@([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})\b")
_RUN_USER = re.compile(r"/run/user/\d+")
# Another account's home, in a process list or a journal line: the name is the
# person, the rest of the path is not.
_OTHER_HOME = re.compile(r"/home/(?!<)[^/\s:'\"]+")


def redact(text: str, counts: Counter[str] | None = None) -> str:
    """`text` with secrets, addresses and this user's paths taken out.

    Every replacement is counted by kind into `counts` when one is given, so a
    report can say what it removed without saying what it was.
    """
    tally: Counter[str] = counts if counts is not None else Counter()

    for kind, pattern, replacement in _SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        tally[kind] += n

    def _assigned(m: re.Match[str]) -> str:
        name, sep, raw = m.group(1), m.group(2), m.group(3)
        quote = '\\"' if raw.startswith('\\"') else raw[:1] if raw[:1] in "\"'" else ""
        value = raw[len(quote):len(raw) - len(quote)]
        if _says_no_secret(value):
            return m.group(0)
        # `NAME=value` is always a setting; after a colon, a bare lowercase
        # name is a setting only when what follows looks like a credential.
        if "=" not in sep and not (_names_a_setting(name, quoted=sep[:1] in "\"'\\")
                                   or _credential_shaped(value)):
            return m.group(0)
        tally["assigned-secret"] += 1
        return f"{name}{sep}{quote}<redacted>{quote}"

    text = _ASSIGNED_SECRET.sub(_assigned, text)

    text, n = _URL_PASSWORD.subn(r"\1:<redacted>@", text)
    tally["url-password"] += n

    def _email(m: re.Match[str]) -> str:
        # `git@github.com:owner/repo` is a remote, not anyone's address.
        if m.group(1) == "git":
            return m.group(0)
        tally["email"] += 1
        return "<email>"

    text = _EMAIL.sub(_email, text)

    home = os.path.expanduser("~")
    if len(home) > 1:
        text, n = re.subn(re.escape(home) + r"(?![A-Za-z0-9_\-])", "~", text)
        tally["home"] += n

    text, n = _OTHER_HOME.subn("/home/<user>", text)
    tally["home"] += n

    text, n = _RUN_USER.subn("/run/user/<uid>", text)
    tally["runtime-dir"] += n

    for kind in [k for k, v in tally.items() if not v]:
        del tally[kind]
    return text


def redact_data(value: Any, counts: Counter[str] | None = None) -> Any:
    """`value` with every string in it -- keys too -- passed through `redact`.

    Done on the structure rather than on its JSON, so a replacement can never
    land inside an escape and leave a results.json that does not parse. A
    string under a key that names a secret (`{"api_key": "..."}`) is the secret
    itself, by the same rule `redact` reads `name: value` with.
    """
    if isinstance(value, str):
        return redact(value, counts)
    if isinstance(value, dict):
        return {redact(str(k), counts): _redact_entry(str(k), v, counts)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(v, counts) for v in value]
    return value


def _redact_entry(key: str, value: Any, counts: Counter[str] | None) -> Any:
    if (isinstance(value, str) and _SECRET_KEY.fullmatch(key) and not _says_no_secret(value)
            and (_names_a_setting(key, quoted=False) or _credential_shaped(value))):
        if counts is not None:
            counts["assigned-secret"] += 1
        return "<redacted>"
    return redact_data(value, counts)


def redact_json_text(text: str, counts: Counter[str] | None = None, *,
                     lines: bool = False) -> str:
    """JSON (or JSON lines) text redacted as data and written back as JSON.

    Text that does not parse is redacted as text: it was never valid JSON for
    a reader to lose.
    """
    if lines:
        return "".join(redact_json_text(line, counts) if line.strip() else line
                        for line in text.splitlines(keepends=True))
    try:
        value = json.loads(text)
    except ValueError:
        return redact(text, counts)
    cleaned = redact_data(value, counts)
    if cleaned == value:
        return text
    # Laid out as it came: a JSON line stays one line.
    ending = "\n" if text.endswith("\n") else ""
    return json.dumps(cleaned, indent=2 if "\n" in text.strip() else None) + ending


# --------------------------------------------------------------------------
# What the machine is asked through: commands and HTTP, both injectable
# --------------------------------------------------------------------------


@dataclass
class Ran:
    """One command's outcome. `code` is None when it never ran to an exit."""

    code: int | None
    out: str = ""
    err: str = ""
    missing: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.out.strip(), self.err.strip()) if part)


def first_line(ran: Ran, limit: int = 160) -> str:
    """The first non-empty line a command printed, for a version or a state."""
    for line in ran.text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def run_command(
    cmd: list[str], timeout: float = 10.0, *, env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> Ran:
    """Run `cmd` without a shell, stdin closed, bounded by `timeout`.

    A missing program is an answer (`missing`), not an exception: most of what
    this asks about is optional on some machine.
    """
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
            env=env, cwd=cwd, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return Ran(None, err=f"{cmd[0]} is not installed", missing=True)
    except subprocess.TimeoutExpired as exc:
        def _text(raw: Any) -> str:
            return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")

        return Ran(None, _text(exc.stdout), _text(exc.stderr) + f"\ntimed out after {timeout:g}s",
                   timed_out=True)
    except OSError as exc:
        return Ran(None, err=f"{cmd[0]}: {exc}")
    return Ran(done.returncode, done.stdout or "", done.stderr or "")


@dataclass
class Answer:
    """One HTTP exchange. `status` is None when nothing answered."""

    status: int | None
    body: str = ""
    error: str = ""

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except ValueError:
            return None


def _loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


# The services asked here are this machine's own, so an HTTP proxy in the
# environment is gone around rather than asked to reach 127.0.0.1.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_request(method: str, url: str, body: Any = None, timeout: float = 5.0) -> Answer:
    """One request; a JSON `body` is sent as JSON. Never raises."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    opener = _DIRECT if _loopback(url) else urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            return Answer(response.status, response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        return Answer(exc.code, text, f"HTTP {exc.code}")
    except Exception as exc:
        return Answer(None, error=f"{type(exc).__name__}: {exc}")


def read_text(path: Path, limit: int = 4_000_000) -> str | None:
    """A file's text, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return None


def tail_text(path: Path, max_bytes: int = 1_000_000) -> str | None:
    """The last `max_bytes` of a file: a console left running for a week has a long log."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return None


# --------------------------------------------------------------------------
# Sections, problems and their fixes
# --------------------------------------------------------------------------


@dataclass
class Problem:
    what: str
    evidence: str = ""
    fix: str = ""
    kind: str = ""


@dataclass
class Section:
    status: str  # ok | problem | skipped | error
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    name: str = ""
    duration_s: float = 0.0


# What to do about each kind of problem, in the words CLAUDE.md's
# troubleshooting already uses, or naming the installer step that does it, so
# the report and the docs give one answer. Every path named exists: a test
# holds this table to that.
FIXES: dict[str, str] = {
    "console-down": (
        "start it with `./launch_console.sh` (its log is /tmp/ambiguity-console.log); this "
        "diagnostic never starts the console itself"),
    "venv-missing": "run `./install.sh`, or `./launch_console.sh`, which builds .venv on first launch",
    "low-disk": (
        "free space where the models live: each local seat model is several GB, and the "
        "embedding model more"),
    "userns": (
        "the browser's sandbox needs unprivileged user namespaces "
        "(kernel.unprivileged_userns_clone=1, user.max_user_namespaces above 0); "
        "`scripts/browser_agent.py doctor` names what it found"),
    "git-missing": "`sudo pacman -S git`, or re-run `./install.sh`",
    "gh-missing": (
        "`sudo pacman -S github-cli`, or re-run `./install.sh`: the pr and merge stages of "
        "git_dwell need gh"),
    "no-chromium": (
        "`sudo pacman -S chromium`, or re-run `./install.sh`; `scripts/browser_agent.py doctor` "
        "says which browser it would use"),
    "node-missing": (
        "`sudo pacman -S nodejs npm`, or re-run `./install.sh`: the Playwright MCP server runs "
        "through npx"),
    "playwright-missing": "`.venv/bin/pip install -e \".[dev,browser]\"`; re-running `./install.sh` does it",
    "docker-service": (
        "`sudo systemctl enable --now docker.service`: Omarchy enables only the socket, so the "
        "database does not come back up after a reboot otherwise"),
    "docker-group": "the docker group applies only after a reboot, or a new login",
    "claude-missing": "`scripts/claude_tools.sh install` installs Claude Code with its official installer",
    "claude-signin": (
        "`scripts/claude_tools.sh install` runs `claude auth login --claudeai` once; Claude in "
        "Chrome and Remote Control need a claude.ai sign-in"),
    "claude-key-env": (
        "remove the variable where it is set (named above): any of them overrides the claude.ai "
        "sign-in, and Claude in Chrome then stays off"),
    "mcp-unregistered": (
        "`scripts/claude_tools.sh install` registers the Playwright MCP server for this "
        "checkout; `scripts/claude_tools.sh check` says what is missing"),
    "mcp-failed": "`scripts/browser_agent.py mcp-check` starts the server on its own and names what fails",
    "gh-signin": "`gh auth login --web`; re-running `./install.sh` asks once",
    "ollama-signin": (
        "`ollama signin` (a free account is enough), then re-run `./install.sh` to pull the "
        "ollama.com tags"),
    "network": (
        "allow the entries `scripts/network_check.sh` printed (its output is in the logs): "
        "nothing on this machine can let a host through, the network's allowlist can"),
    "no-gpu": (
        "every model would run on the CPU: the NVIDIA driver is the machine's own setup "
        "(Omarchy installs it), after which nvidia-smi lists the cards"),
    "nvidia-smi-missing": (
        "the card is there but the driver's tools are not: install the NVIDIA driver packages "
        "for this card (Omarchy's own setup does), then reboot"),
    "nvidia-smi-failed": "nvidia-smi could not talk to the driver, usually one updated without a reboot: reboot",
    "cuda-old-card": (
        "run `./cuda-embed-ollama.sh`: Arch's CUDA 13 build skips cards below compute 7.5, and "
        "Ollama's own CUDA 12 build drives them"),
    "ollama-missing": "`sudo pacman -S ollama`, or re-run `./install.sh`",
    "daemon-down": (
        "`sudo systemctl restart ollama`; once `scripts/ollama_keepalive.sh install` has run, "
        "systemd restarts a daemon that exits (`scripts/ollama_keepalive.sh check` says whether "
        "it has)"),
    "keepalive": (
        "`scripts/ollama_keepalive.sh install`: the daemon then starts at boot and is restarted "
        "whenever it exits"),
    "one-model": (
        "re-run `./install.sh`: its ollama step writes the one-model drop-in "
        "(OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_NUM_PARALLEL=1) and restarts the daemon"),
    "model-not-pulled": (
        "`ollama pull` each tag named, or re-run `./install.sh`, which pulls every seat's tag "
        "and the embedding model"),
    "builder-no-tools": (
        "seat the Builder on a tag whose capabilities include tools (qwen3.8:latest is the "
        "default): BUILDER_MODEL in .env, or the console's seat menu"),
    "two-resident": (
        "one model at a time is the rule: something reached the daemon around the console's "
        "arbiter, or the one-model drop-in is missing (see the ollama section)"),
    "gpu-oom": (
        "`journalctl -u ollama` names the card and the allocation that failed, and nvidia-smi "
        "what else holds that card: on the display card, the desktop and any browser"),
    "embedder-failed": (
        "is it pulled? `ollama pull qwen3-embedding:latest`; `journalctl -u ollama` names a "
        "load that failed"),
    "embedder-dimensions": (
        "the tag no longer names the model the corpus was built for: re-pull it "
        "(`ollama pull qwen3-embedding:latest`)"),
    "embedder-cpu": (
        "`ollama ps` names the split; `journalctl -u ollama` reading 'skipping CUDA device' means "
        "Arch's CUDA 13 build on a card it cannot drive: run `./cuda-embed-ollama.sh`"),
    "embedder-split": (
        "every layer is forced onto the cards, so a split means something else holds them: "
        "nvidia-smi names it (the desktop and any browser on the display card)"),
    "seat-probe": (
        "`python scripts/diagnose_seats.py --phase probe` names each failure; move the seat "
        "with its _MODEL variable in .env or the console's seat menu "
        "(`python scripts/diagnose_seats.py --list` shows the candidates)"),
    "seat-split": (
        "the forced load did not fit and the seat ran partly on the CPU: `journalctl -u ollama` "
        "names the allocation, and nvidia-smi what else holds the card"),
    "seat-no-key": "a seat on Anthropic needs ANTHROPIC_API_KEY in .env; without one it answers with canned text",
    "seat-down": (
        "the console's own reason is quoted above; `python scripts/diagnose_seats.py --phase "
        "probe` tries the seat directly"),
    "postgres-down": (
        "`docker ps | grep postgres18`, then `docker logs postgres18`; "
        "`sudo systemctl enable --now docker.service` brings it back at boot"),
    "no-pgvector": "re-run `./install.sh`: it moves postgres18 to pgvector's image on the same data volume",
    "corpus-other-model": (
        "nothing to do by hand: it reads as absent until the next rebuild replaces it, which "
        "the console does when it starts and before every run"),
    "corpus-unavailable": (
        "the database stopped answering: `docker ps | grep postgres18`, then "
        "`docker logs postgres18`"),
    "corpus-stale": (
        "read the [Corpus] line in /tmp/ambiguity-console.log (excerpted under console-after): "
        "it names each file that failed and why, usually a model load short of GPU memory"),
    "circuit-open": (
        "a circuit closes by itself once its service answers a trial call; clicking its chip "
        "in the console header lets the next call through now"),
    "searxng-json": (
        "add json under search.formats in its settings.yml, then "
        "`systemctl --user restart ambiguity-searxng`"),
    "searxng-down": (
        "`systemctl --user restart ambiguity-searxng`, and `podman logs ambiguity-searxng` says "
        "why it stopped; re-running `./install.sh` rewrites it"),
    "pr-checks-failed": (
        "a red check is shown, never fixed: start a run to fix it, or finish the pull request "
        "on github.com"),
    "console-errors": "the journal lines quoted above, and the console log excerpt under console-after",
    "browser-cannot-run": "`scripts/browser_agent.py doctor` says why no browser launches here",
    "browser-pass": "the browser agent's own report beside this one names the pass, with its screenshots",
    "suite-failed": "the failures are in the suite's log; `python -m pytest tests/ -q` reproduces them",
}


def finding(kind: str, what: str, evidence: str = "") -> Problem:
    """A problem of `kind`, carrying that kind's fix."""
    return Problem(what=what, evidence=evidence.strip()[:600], fix=FIXES.get(kind, ""), kind=kind)


def concluded(summary: str, data: dict[str, Any], problems: list[Problem]) -> Section:
    return Section("problem" if problems else "ok", summary, data, problems)


def skipped(reason: str, data: dict[str, Any] | None = None) -> Section:
    return Section("skipped", reason, data or {})


# --------------------------------------------------------------------------
# Pure readers of what the machine printed (tested on their own)
# --------------------------------------------------------------------------

GPU_QUERY = (
    "index,name,driver_version,memory.total,memory.used,compute_cap,display_active,"
    "pstate,temperature.gpu,utilization.gpu"
)
GPU_FIELDS = (
    "index", "name", "driver", "memory_total_mib", "memory_used_mib", "compute_cap",
    "display_active", "pstate", "temperature_c", "utilization_pct",
)
_GPU_NUMBERS = frozenset({"index", "memory_total_mib", "memory_used_mib", "temperature_c",
                          "utilization_pct"})


def _gpu_value(key: str, raw: str) -> Any:
    raw = raw.strip()
    # nvidia-smi says `[N/A]` or `[Not Supported]` for a field a card cannot report.
    if not raw or raw.startswith("[") or raw.upper() == "N/A":
        return None
    if key in _GPU_NUMBERS:
        try:
            return int(float(raw))
        except ValueError:
            return None
    if key == "display_active":
        return raw.lower() in ("enabled", "yes", "1")
    return raw


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    """One dict per card from `--query-gpu=GPU_QUERY --format=csv,noheader,nounits`.

    Tolerant on purpose: "No devices were found" is no cards, a field a card
    cannot report is None, and a name with a comma in it is put back together.
    """
    cards: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or not row[0].strip().isdigit():
            continue
        extra = len(row) - len(GPU_FIELDS)
        if extra < 0:
            continue
        if extra:
            row = [row[0], ", ".join(row[1:2 + extra]), *row[2 + extra:]]
        cards.append({key: _gpu_value(key, raw) for key, raw in zip(GPU_FIELDS, row, strict=True)})
    return cards


def compute_below(cap: Any, floor: tuple[int, int] = (7, 5)) -> bool:
    """Whether a compute capability such as "6.1" is below `floor`; unknown is not."""
    match = re.fullmatch(r"(\d+)\.(\d+)", str(cap or "").strip())
    return match is not None and (int(match[1]), int(match[2])) < floor


def parse_compute_apps(text: str) -> list[dict[str, Any]]:
    """`--query-compute-apps=pid,process_name,used_memory` rows."""
    apps: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) >= 3 and row[0].strip().isdigit():
            used = _gpu_value("memory_used_mib", row[-1])
            apps.append({"pid": int(row[0]), "process": ", ".join(row[1:-1]).strip(),
                         "memory_used_mib": used})
    return apps


_LSPCI_DEVICE = re.compile(
    r"^(\S+)\s+(VGA compatible controller|3D controller|Display controller)\s*(?:\[[0-9a-f]{4}\])?:\s*(.*)$"
)


def parse_lspci(text: str) -> list[dict[str, Any]]:
    """Display devices from `lspci -nnk`, each with the driver bound to it."""
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line[:1].strip():
            current = None
            match = _LSPCI_DEVICE.match(line)
            if match:
                current = {"slot": match[1], "class": match[2], "device": match[3].strip(),
                           "driver": None, "nvidia": "[10de:" in match[3]}
                devices.append(current)
        elif current is not None:
            bound = re.match(r"\s*Kernel driver in use:\s*(\S+)", line)
            if bound:
                current["driver"] = bound[1]
    return devices


def placement(ps: Any) -> list[dict[str, Any]]:
    """What `/api/ps` says is loaded, and how much of each sits on the CPU."""
    rows: list[dict[str, Any]] = []
    for entry in (ps or {}).get("models") or []:
        size = int(entry.get("size") or 0)
        vram = int(entry.get("size_vram") or 0)
        rows.append({
            "model": str(entry.get("name") or entry.get("model") or ""),
            "size_gib": round(size / 2**30, 2),
            "vram_gib": round(vram / 2**30, 2),
            # None when the daemon gave no size: an unknown split is not a clean one.
            "cpu_share": round(max(0.0, 1.0 - vram / size), 3) if size > 0 else None,
            "expires_at": entry.get("expires_at"),
        })
    return rows


def one_model_settings(text: str) -> dict[str, str]:
    """The variables in `systemctl show ollama.service -p Environment`."""
    text = text.strip()
    if text.startswith("Environment="):
        text = text[len("Environment="):]
    try:
        words = shlex.split(text)
    except ValueError:
        words = text.split()
    return dict(word.split("=", 1) for word in words if "=" in word)


ONE_MODEL_VARIABLES = ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL")


def one_model_gaps(settings: dict[str, str]) -> list[str]:
    """Each one-model variable that is not 1, worded; empty when the drop-in holds."""
    return [f"{name} is {settings[name]!r}, not 1" if name in settings else f"{name} is not set"
            for name in ONE_MODEL_VARIABLES if settings.get(name) != "1"]


_JOURNAL_KINDS = (
    ("skipping", re.compile(r"skipping CUDA device", re.I)),
    ("oom", re.compile(r"out of memory|cudaMalloc", re.I)),
    ("offloaded", re.compile(r"offloaded", re.I)),
)


def scan_journal(text: str) -> dict[str, list[str]]:
    """The daemon's journal lines that say where a model went, or why not."""
    found: dict[str, list[str]] = {kind: [] for kind, _ in _JOURNAL_KINDS}
    for line in text.splitlines():
        for kind, pattern in _JOURNAL_KINDS:
            if pattern.search(line):
                found[kind].append(line.strip())
    return found


def same_tag(a: str, b: str) -> bool:
    """`qwen3.8` and `qwen3.8:latest` are one model; the daemon reports the second."""
    def full(tag: str) -> str:
        return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"

    return full(a) == full(b)


def is_cloud_tag(tag: str) -> bool:
    return tag.endswith((":cloud", "-cloud"))


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")
    return values


def parse_meminfo(text: str) -> dict[str, float]:
    """`/proc/meminfo` fields, in GiB."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if match:
            values[match[1]] = round(int(match[2]) / 2**20, 1)
    return values


def env_names(text: str) -> list[str]:
    """The names a `.env` sets -- never its values."""
    names = []
    for line in text.splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if match:
            names.append(match[1])
    return sorted(set(names))


def rc_exports(text: str, names: tuple[str, ...]) -> list[str]:
    """Which of `names` a shell rc file sets (bash/zsh `export`, fish `set -x`)."""
    found = []
    for name in names:
        pattern = re.compile(
            rf"^\s*(?:export\s+|declare\s+-x\s+)?{name}\s*=|^\s*set\s+(?:-\w+\s+)*{name}\b",
            re.M,
        )
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            if not text[line_start:match.start()].strip().startswith("#"):
                found.append(name)
                break
    return found


def parse_listeners(text: str, ports: tuple[int, ...]) -> dict[str, list[dict[str, Any]]]:
    """Who listens on each of `ports`, from `ss -ltnpH`."""
    held: dict[str, list[dict[str, Any]]] = {str(port): [] for port in ports}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]
        port = local.rsplit(":", 1)[-1]
        if port in held:
            programs = sorted(set(re.findall(r'\("([^"]+)",pid=', line)))
            held[port].append({"address": local, "programs": programs})
    return held


def network_groups(script_text: str) -> set[str]:
    """The groups `scripts/network_check.sh` knows, read from its own table."""
    table = re.search(r"^TABLE='\n(.*?)^'", script_text, re.S | re.M)
    if not table:
        return set()
    return {row.split("|", 1)[0] for row in table.group(1).splitlines() if "|" in row}


def pytest_summary(text: str) -> str:
    """pytest's last summary line ("3 failed, 120 passed in 9.1s"), or ""."""
    for line in reversed(text.splitlines()):
        if re.search(r"\b(passed|failed|error|no tests ran)\b", line):
            return line.strip(" =")
    return ""


_GPU_FIT_MARKS = ("out of memory", "cudamalloc", "unable to allocate", "insufficient memory",
                  "failed to allocate")


# --------------------------------------------------------------------------
# The context every section is handed
# --------------------------------------------------------------------------

DEFAULT_BASE = "http://localhost:8080"
DEFAULT_CONSOLE_LOG = "/tmp/ambiguity-console.log"
DEFAULT_OLLAMA = "http://localhost:11434"

# The variables that take Claude Code off a claude.ai sign-in. `.env` is
# sourced with `set -a` by the installers, so they reach children unasked.
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
RC_FILES = (".bashrc", ".zshrc", ".profile", ".config/fish/config.fish")
NATIVE_HOST_MANIFEST = "com.anthropic.claude_code_browser_extension.json"
BROWSER_PROFILES = ("chromium", "google-chrome", "BraveSoftware/Brave-Browser")
CONSOLE_PORTS = (8080, 8081, 5432, 5433, 8888, 11434)
NETWORK_REQUIRED = ("pypi", "ollama", "hf", "tokenizer", "github")
NETWORK_OPTIONAL = ("arch", "dockerhub", "research", "cloud", "npm", "claude", "playwright")
PACKAGES = (
    "python", "git", "github-cli", "curl", "ollama", "ollama-cuda", "nvidia-utils",
    "base-devel", "xdg-utils", "podman", "crun", "docker", "postgresql-libs", "chromium",
    "nodejs", "npm", "ffmpeg", "nvtop", "pciutils", "lsof", "noto-fonts", "noto-fonts-emoji",
    "wl-clipboard",
)
MIN_FREE_GIB = 10.0
SEARXNG_CONTAINER = "ambiguity-searxng"
POSTGRES_CONTAINER = "postgres18"
CONSOLE_READS = ("status", "rag_stats", "healing", "list_seats", "run_progress",
                 "embedding_activity", "last_run")
# What scripts/claude_tools.sh allows `claude mcp get`, which starts the
# server to health-check it: a first npx start downloads the package.
MCP_GET_SECONDS = 120.0


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


class GpuSampler:
    """`nvidia-smi -lms 500` in the background: peak VRAM and load per section.

    A section's cost on the cards is gone by the time it returns, so it is
    sampled while it runs, and each sample is booked to whichever section is
    current.
    """

    def __init__(self, current: Callable[[], str]) -> None:
        self._current = current
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.peaks: dict[str, dict[str, dict[str, int]]] = {}

    def start(self) -> bool:
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return True

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            used = _gpu_value("memory_used_mib", parts[1])
            load = _gpu_value("utilization_pct", parts[2])
            card = self.peaks.setdefault(self._current() or "start", {}).setdefault(
                parts[0], {"memory_used_mib": 0, "utilization_pct": 0})
            if used is not None:
                card["memory_used_mib"] = max(card["memory_used_mib"], used)
            if load is not None:
                card["utilization_pct"] = max(card["utilization_pct"], load)

    def stop(self) -> dict[str, dict[str, dict[str, int]]]:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peaks


@dataclass
class Context:
    """Everything a section may ask the machine through, and what earlier ones found.

    `run`, `http` and `read` are the only ways out, so a test hands in fakes and
    gets a whole report from a machine that does not exist.
    """

    out: Path
    args: argparse.Namespace
    run: Callable[..., Ran] = run_command
    http: Callable[..., Answer] = http_request
    read: Callable[[Path], str | None] = read_text
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    home: Path = field(default_factory=Path.home)
    root: Path = ROOT
    euid: int = field(default_factory=os.geteuid)
    # Whether the GPU sampler may start: a test has no cards to sample.
    background: bool = True
    tty: bool = field(default_factory=lambda: _stdin_is_tty())
    ask: Callable[[str], str] = input
    argv: list[str] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    current: str = ""
    sampler: GpuSampler | None = None
    # Learned as the sections run, and read by the ones after.
    facts_cache: dict[str, Any] | None = None
    facts_error: str = ""
    gpus: list[dict[str, Any]] = field(default_factory=list)
    daemon_up: bool | None = None
    console_up: bool | None = None
    journal_seq: int | None = None
    journal_cache: dict[str, Any] | None = None

    @property
    def logs(self) -> Path:
        return self.out / "logs"

    @property
    def base(self) -> str:
        return str(self.args.base).rstrip("/")

    @property
    def python(self) -> str:
        """The venv's interpreter, where the project and its dependencies are."""
        venv = self.root / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def cmd(self, cmd: list[str], timeout: float = 10.0, **kwargs: Any) -> Ran:
        """`run`, with this process's starting environment unless told otherwise."""
        kwargs.setdefault("env", self.env)
        return self.run(cmd, timeout, **kwargs)

    def log(self, name: str, text: str) -> str:
        """Write `text`, redacted, to logs/<name>; return the path the report cites."""
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.logs / name).write_text(redact(text, self.counts), encoding="utf-8")
        return f"logs/{name}"

    def log_json(self, name: str, value: Any) -> str:
        """Write `value` to logs/<name> as JSON, redacted as data so the file still parses."""
        self.logs.mkdir(parents=True, exist_ok=True)
        # A round trip first, so what `default=str` turns into text is redacted too.
        plain = json.loads(json.dumps(value, default=str))
        (self.logs / name).write_text(json.dumps(redact_data(plain, self.counts), indent=2),
                                      encoding="utf-8")
        return f"logs/{name}"

    def rpc(self, method: str, params: dict[str, Any] | None = None,
            timeout: float = 30.0) -> tuple[Any, str]:
        """One console RPC: `(result, "")`, or `(None, why)`."""
        answer = self.http("POST", f"{self.base}/rpc", {"method": method, "params": params or {}},
                           timeout)
        if answer.status != 200:
            return None, answer.error or f"HTTP {answer.status}"
        payload = answer.json()
        if not isinstance(payload, dict):
            return None, "the console did not answer with JSON"
        if "error" in payload:
            return None, str((payload.get("error") or {}).get("message") or payload["error"])
        return payload.get("result"), ""

    def env_without_keys(self) -> dict[str, str]:
        return {k: v for k, v in self.env.items() if k not in KEY_VARIABLES}

    def ollama_url(self) -> str:
        facts = self.facts()
        return str(facts.get("ollama_url") or self.env.get("OLLAMA_BASE_URL")
                   or DEFAULT_OLLAMA).rstrip("/")

    def facts(self) -> dict[str, Any]:
        """The project's own answers: effective seats, the embedding model, the
        database, read through the venv so the console and this agree."""
        if self.facts_cache is None:
            found, error = self.snippet("facts", FACTS_SNIPPET, 60)
            self.facts_cache, self.facts_error = (found or {}), error
            if found:
                self.log_json("facts.json", found)
        return self.facts_cache

    def snippet(self, marker: str, code: str, timeout: float,
                *args: str) -> tuple[dict[str, Any] | None, str]:
        """Run `code` under the venv's interpreter; return the JSON it marked, or why not.

        In a child, not here: the project loads `.env` into the environment of
        whatever imports it, and a hung daemon call is then bounded by a timeout
        instead of holding this process.
        """
        ran = self.cmd([self.python, "-c", code, str(self.root), *args], timeout,
                       cwd=str(self.root))
        prefix = f"@@{marker} "
        for line in reversed(ran.out.splitlines()):
            if line.startswith(prefix):
                try:
                    found = json.loads(line[len(prefix):])
                except ValueError:
                    break
                if isinstance(found, dict):
                    return found, ""
        if ran.missing:
            return None, f"{self.python} is not there"
        if ran.timed_out:
            return None, f"timed out after {timeout:g}s"
        tail = ran.text.splitlines()[-3:]
        return None, " / ".join(tail) or f"exit {ran.code}"

    def ollama_journal(self) -> dict[str, Any]:
        """The daemon's last two hours of journal, scanned; read once, shared."""
        if self.journal_cache is None:
            # Not `-q`: that also hides the hint a user outside the journal's
            # groups is given, and an unreadable journal would read as a quiet one.
            ran = self.cmd(["journalctl", "-u", "ollama", "--since", "-2h", "--no-pager",
                            "-o", "short-iso"], 20)
            text = ran.text
            lowered = text.lower()
            readable = ran.ok and not any(
                hint in lowered for hint in ("insufficient permissions",
                                             "not seeing messages from other users",
                                             "no journal files were opened"))
            found = scan_journal(text) if readable else {}
            if readable and text.strip():
                lines = text.splitlines()
                kept = [line for kind in found.values() for line in kind][-200:]
                self.log("ollama-journal.txt", "\n".join(kept + ["", "-- last lines --",
                                                                 *lines[-100:]]))
            why = ("journalctl is not installed" if ran.missing
                   else "" if readable else first_line(ran) or f"exit {ran.code}")
            self.journal_cache = {"readable": readable, "why": why, **found}
        return self.journal_cache

    def busy_reasons(self) -> list[str]:
        """Why the console would compete with a section that loads models; empty when idle.

        Read fresh each time, from the console's own payloads: a run can start
        while the diagnostic is halfway through.
        """
        answer = self.http("GET", f"{self.base}/api/status", None, 5)
        status = answer.json() if answer.status == 200 else None
        if not isinstance(status, dict):
            return []
        reasons: list[str] = []
        progress, _ = self.rpc("run_progress", timeout=10)
        if status.get("run_in_flight") or (isinstance(progress, dict) and progress.get("running")):
            goal = (progress or {}).get("goal") if isinstance(progress, dict) else ""
            reasons.append("a run is in flight" + (f" ({goal[:60]})" if goal else ""))
        indexing = status.get("indexing") or {}
        if isinstance(indexing, dict) and indexing.get("running"):
            reasons.append("a corpus rebuild is in flight"
                           + (f" ({indexing.get('message')})" if indexing.get("message") else ""))
        # The console says when its monitor is finishing a pull request; one
        # too old to say reads as not finishing, and its follower holds no
        # cards anyway.
        follow = status.get("pull_request_follow")
        if isinstance(follow, dict) and follow.get("running"):
            number = follow.get("number")
            reasons.append(f"pull request {f'#{number} ' if number else ''}is being finished")
        return reasons


# --------------------------------------------------------------------------
# What the project itself is asked, in a child under the venv's interpreter
# --------------------------------------------------------------------------

FACTS_SNIPPET = r"""
import importlib.util, json, logging, os, sys
logging.disable(logging.CRITICAL)
root = sys.argv[1]
from langgraph_agent import config
from langgraph_agent.corpus_store import corpus_schema, database_url, redacted_url
from langgraph_agent.graphrag_server import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME, OLLAMA_EMBED_OPTIONS)
candidates = {}
try:
    spec = importlib.util.spec_from_file_location(
        "diagnose_seats", os.path.join(root, "scripts", "diagnose_seats.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_seats"] = module
    spec.loader.exec_module(module)
    candidates = {c.model: c.key for c in module.CANDIDATES if c.provider == "ollama"}
except Exception:
    pass
seats = []
for agent in config.AGENTS:
    info = config.get_agent_model_info(agent)
    ollama = info["provider"] == "ollama"
    seats.append({"role": agent, "provider": info["provider"], "model": info["model"],
                  "local": ollama and config.is_local_ollama_model(info["model"]),
                  "candidate": candidates.get(info["model"]) if ollama else None})
print("@@facts " + json.dumps({
    "seats": seats,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_dimensions": EMBEDDING_DIMENSIONS,
    "embed_options": OLLAMA_EMBED_OPTIONS,
    "seat_gpu_options": config.OLLAMA_SEAT_GPU_OPTIONS,
    "ollama_url": config.ollama_base_url(),
    "database_url": redacted_url(database_url()),
    "this_schema": corpus_schema(os.path.join(root, "knowledge")),
    "searxng_url": os.environ.get("SEARXNG_URL", ""),
}, default=str))
"""

# One cold and one warm vector through the class a run uses, as install.sh
# proves the embedder; a model this loaded is unloaded again, so the daemon is
# left holding what it held.
EMBEDDER_SNIPPET = r"""
import json, logging, time
logging.disable(logging.CRITICAL)
from langgraph_agent import config, self_healing
from langgraph_agent.graphrag_server import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME, OllamaEmbedder)
model = EMBEDDING_MODEL_NAME
out = {"model": model, "expected_dimensions": EMBEDDING_DIMENSIONS}
loaded_before = config.ollama_cpu_share(model) is not None
out["loaded_before"] = loaded_before
embedder = OllamaEmbedder(model)
try:
    started = time.monotonic()
    vector = embedder.encode("a diagnostic asks for one vector")
    out["cold_s"] = round(time.monotonic() - started, 2)
    started = time.monotonic()
    embedder.encode("and then for one more, with the model loaded")
    out["warm_s"] = round(time.monotonic() - started, 2)
    out["dimensions"] = len(vector)
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    out["unreachable"] = (config.daemon_unreachable(exc)
                          or isinstance(exc, self_healing.CircuitOpenError))
out["cpu_share"] = embedder.cpu_share
out["placement_note"] = embedder.placement_note
if not loaded_before:
    out["unloaded"] = config.unload_ollama_model(model)
print("@@embedder " + json.dumps(out, default=str))
"""

# The one-word prompt install.sh sends a seat, through its real call path, for
# the seats the seat diagnostic has no candidate for.
PROBE_SNIPPET = r"""
import json, logging, sys
logging.disable(logging.CRITICAL)
from langgraph_agent.config import get_agent_llm, get_agent_status
out = {}
for agent in [a for a in sys.argv[2].split(",") if a]:
    silent, error = None, ""
    if get_agent_status(agent)["live"]:
        try:
            reply = get_agent_llm(agent).invoke("Reply with the single word: ready")
            silent = not str(getattr(reply, "content", reply)).strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
    seat = get_agent_status(agent)
    out[agent] = {"live": seat["live"], "badge": seat["badge"], "reason": seat["reason"],
                  "model": seat["model"], "provider": seat["provider"],
                  "silent": silent, "error": error}
print("@@probe " + json.dumps(out, default=str))
"""

# Read-only: the session refuses writes, and nothing here would make one.
DATABASE_SNIPPET = r"""
import json, logging
logging.disable(logging.CRITICAL)
out = {}
try:
    import langgraph_agent.config  # loads .env, as the console does
    import psycopg
    from psycopg import sql
    from langgraph_agent.corpus_store import (
        SCHEMA_PREFIX, database_unreachable, database_url, redacted_url)
except Exception as exc:
    print("@@database " + json.dumps({"error": f"{type(exc).__name__}: {exc}", "import": True}))
    raise SystemExit(0)
url = database_url()
out["url"] = redacted_url(url)
try:
    with psycopg.connect(url, connect_timeout=3,
                         options="-c default_transaction_read_only=on") as conn:
        out["server"] = conn.execute("SHOW server_version").fetchone()[0]
        out["version"] = conn.execute("SELECT version()").fetchone()[0]
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        out["pgvector"] = row[0] if row else None
        row = conn.execute(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        out["pgvector_available"] = row[0] if row else None
        names = [r[0] for r in conn.execute(
            "SELECT nspname FROM pg_namespace WHERE starts_with(nspname, %s) ORDER BY 1",
            (SCHEMA_PREFIX,))]
        schemas = []
        for name in names:
            entry = {"schema": name}
            def table(t, name=name):
                return sql.SQL("{}.{}").format(sql.Identifier(name), sql.Identifier(t))
            if conn.execute("SELECT to_regclass(%s)", (f'"{name}".corpus',)).fetchone()[0]:
                row = conn.execute(sql.SQL(
                    "SELECT persist_dir, embedding_model, dimensions, floor_record FROM {}"
                ).format(table("corpus"))).fetchone()
                if row:
                    floor = row[3] if isinstance(row[3], dict) else {}
                    entry.update(persist_dir=row[0], model=row[1], dimensions=row[2],
                                 floor=floor.get("floor"), floor_model=floor.get("model"),
                                 floor_measured_at=floor.get("measured_at"),
                                 floor_too_small=floor.get("too_small"))
            if conn.execute("SELECT to_regclass(%s)", (f'"{name}".chunks',)).fetchone()[0]:
                row = conn.execute(sql.SQL(
                    "SELECT count(*), count(DISTINCT doc_id) FROM {}").format(table("chunks"))
                ).fetchone()
                entry.update(chunks=row[0], documents=row[1])
            schemas.append(entry)
        out["schemas"] = schemas
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    out["unreachable"] = database_unreachable(exc)
print("@@database " + json.dumps(out, default=str))
"""


# --------------------------------------------------------------------------
# The sections, in the order they run
# --------------------------------------------------------------------------


def _free_gib(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 2**30, 1)
    except OSError:
        return None


def _models_dir(ctx: Context) -> Path:
    """Where the daemon keeps its weights: `OLLAMA_MODELS`, else the usual places."""
    if ctx.env.get("OLLAMA_MODELS"):
        return Path(ctx.env["OLLAMA_MODELS"])
    for candidate in (ctx.home / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models"),
                      Path("/var/lib/ollama")):
        if candidate.exists():
            return candidate
    return ctx.home


def repo_state(ctx: Context) -> dict[str, Any]:
    head = ctx.cmd(["git", "-C", str(ctx.root), "rev-parse", "--short", "HEAD"], 10)
    branch = ctx.cmd(["git", "-C", str(ctx.root), "rev-parse", "--abbrev-ref", "HEAD"], 10)
    status = ctx.cmd(["git", "-C", str(ctx.root), "status", "--porcelain"], 20)
    return {
        "head": first_line(head) if head.ok else None,
        "branch": first_line(branch) if branch.ok else None,
        "dirty": len([line for line in status.out.splitlines() if line.strip()]) if status.ok
        else None,
    }


def section_machine(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []

    release = parse_os_release(ctx.read(Path("/etc/os-release")) or "")
    data["os"] = release.get("PRETTY_NAME") or release.get("NAME") or platform.system()
    omarchy = ctx.cmd(["omarchy-version"], 5)
    data["omarchy"] = first_line(omarchy) if omarchy.ok else None
    kernel = ctx.cmd(["uname", "-r"], 5)
    data["kernel"] = first_line(kernel) if kernel.ok else platform.release()
    cpuinfo = ctx.read(Path("/proc/cpuinfo")) or ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    data["cpu"] = model[1].strip() if model else platform.processor() or None
    data["cpus"] = os.cpu_count()
    memory = parse_meminfo(ctx.read(Path("/proc/meminfo")) or "")
    data["ram_gib"] = memory.get("MemTotal")
    data["ram_available_gib"] = memory.get("MemAvailable")

    models = _models_dir(ctx)
    data["models_dir"] = str(models)
    data["models_free_gib"] = _free_gib(models)
    data["repo_free_gib"] = _free_gib(ctx.root)
    if data["models_free_gib"] is not None and data["models_free_gib"] < MIN_FREE_GIB:
        problems.append(finding("low-disk", f"{data['models_free_gib']} GiB free where the models "
                                f"live ({models})"))

    data["session"] = {
        "type": ctx.env.get("XDG_SESSION_TYPE") or None,
        "desktop": ctx.env.get("XDG_CURRENT_DESKTOP") or None,
        "wayland": bool(ctx.env.get("WAYLAND_DISPLAY")),
        "x11": bool(ctx.env.get("DISPLAY")),
        "hyprland": bool(ctx.env.get("HYPRLAND_INSTANCE_SIGNATURE")),
    }

    venv = ctx.root / ".venv" / "bin" / "python"
    if venv.exists():
        version = ctx.cmd([str(venv), "--version"], 10)
        data["venv_python"] = first_line(version) or None
    else:
        data["venv_python"] = None
        problems.append(finding("venv-missing", f"there is no .venv in {ctx.root}"))
    data["python"] = platform.python_version()
    data["repo"] = repo_state(ctx)

    # What Chromium's sandbox needs from the kernel. Root never gets the sandbox
    # (the browser agent turns it off for root), so only a user is held to it.
    def sysctl(path: str) -> str | None:
        text = ctx.read(Path(path))
        return text.strip() if text is not None else None

    userns = {
        "kernel.unprivileged_userns_clone": sysctl("/proc/sys/kernel/unprivileged_userns_clone"),
        "user.max_user_namespaces": sysctl("/proc/sys/user/max_user_namespaces"),
        "kernel.apparmor_restrict_unprivileged_userns": sysctl(
            "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"),
    }
    data["userns"] = userns
    blocked = [name for name, value, bad in (
        ("kernel.unprivileged_userns_clone", userns["kernel.unprivileged_userns_clone"], "0"),
        ("user.max_user_namespaces", userns["user.max_user_namespaces"], "0"),
        ("kernel.apparmor_restrict_unprivileged_userns",
         userns["kernel.apparmor_restrict_unprivileged_userns"], "1"),
    ) if value == bad]
    if blocked and ctx.euid != 0:
        problems.append(finding("userns", "the kernel refuses the user namespaces the browser's "
                                "sandbox needs", ", ".join(blocked)))

    ram = f"{data['ram_gib']} GiB ram" if data["ram_gib"] else "ram unknown"
    free = (f"{data['models_free_gib']} GiB free for models" if data["models_free_gib"] is not None
            else "disk unknown")
    omarchy_note = f" (omarchy {data['omarchy']})" if data["omarchy"] else ""
    summary = (f"{data['os']}{omarchy_note}, kernel {data['kernel']}, {data['cpus']} cpus, "
               f"{ram}, {free}")
    return concluded(summary, data, problems)


TOOL_VERSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git", ("git", "--version")),
    ("gh", ("gh", "--version")),
    ("curl", ("curl", "--version")),
    ("docker", ("docker", "--version")),
    ("podman", ("podman", "--version")),
    ("node", ("node", "--version")),
    ("npm", ("npm", "--version")),
    ("npx", ("npx", "--version")),
    ("ffmpeg", ("ffmpeg", "-version")),
    ("chromium", ("chromium", "--version")),
    ("chromium-arch", ("/usr/lib/chromium/chromium", "--version")),
    ("google-chrome", ("google-chrome-stable", "--version")),
    ("claude", ("claude", "--version")),
    ("yay", ("yay", "--version")),
    ("nvtop", ("nvtop", "--version")),
)


def _playwright_browsers(ctx: Context) -> list[str]:
    """Browsers Playwright downloaded for itself, which the browser agent can also use."""
    where = Path(ctx.env.get("PLAYWRIGHT_BROWSERS_PATH") or ctx.home / ".cache" / "ms-playwright")
    try:
        return sorted(p.name for p in where.glob("chromium*") if p.is_dir())
    except OSError:
        return []


def section_tools(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []
    versions: dict[str, str | None] = {}
    for name, cmd in TOOL_VERSIONS:
        ran = ctx.cmd(list(cmd), 20)
        versions[name] = first_line(ran) if ran.ok else None
    for name, module in (("ruff", "ruff"), ("mypy", "mypy"), ("pytest", "pytest")):
        ran = ctx.cmd([ctx.python, "-m", module, "--version"], 20)
        versions[name] = first_line(ran) if ran.ok else None
    playwright = ctx.cmd([ctx.python, "-c", "import importlib.metadata as m; "
                          "print(m.version('playwright'))"], 20)
    versions["playwright (venv)"] = first_line(playwright) if playwright.ok else None
    data["versions"] = versions
    data["playwright_browsers"] = _playwright_browsers(ctx)
    ctx.log("tool-versions.txt", "\n".join(f"{k}: {v or '-'}" for k, v in versions.items()))

    for tool, kind in (("git", "git-missing"), ("gh", "gh-missing")):
        if versions[tool] is None:
            problems.append(finding(kind, f"{tool} is not installed (or does not run)"))
    if not any(versions[b] for b in ("chromium", "chromium-arch", "google-chrome")) \
            and not data["playwright_browsers"]:
        problems.append(finding("no-chromium", "no Chromium or Chrome to drive the console with"))
    if versions["npx"] is None:
        problems.append(finding("node-missing", "npx is not installed, so the Playwright MCP "
                                "server cannot start"))
    if (ctx.root / ".venv").exists() and versions["playwright (venv)"] is None:
        problems.append(finding("playwright-missing", "the venv has no playwright, so the browser "
                                "agent cannot run"))

    groups = ctx.cmd(["id", "-nG"], 5)
    data["docker_group"] = "docker" in groups.out.split() if groups.ok else None
    systemctl = ctx.cmd(["systemctl", "--version"], 5).ok
    data["systemd"] = systemctl
    # The database install.sh runs is a container on this machine; one
    # elsewhere does not care whether docker starts at boot here.
    database = str(ctx.facts().get("database_url") or "postgresql://127.0.0.1")
    local_database = _loopback(database.replace("postgresql://", "http://", 1).replace(
        "postgres://", "http://", 1))
    if versions["docker"] and systemctl:
        enabled = first_line(ctx.cmd(["systemctl", "is-enabled", "docker.service"], 10))
        data["docker_service"] = enabled or None
        if local_database and enabled not in ("enabled", "static", "alias"):
            problems.append(finding("docker-service", f"docker.service is {enabled or 'unknown'}, "
                                    "so the database does not start at boot"))

    if versions["claude"]:
        which = ctx.cmd(["which", "-a", "claude"], 5)
        found = list(dict.fromkeys(which.out.split())) if which.ok else []
        data["claude_on_path"] = found
        if len(found) > 1 and "mise" in found[0]:
            data["claude_note"] = "a mise shim comes first on PATH and may shadow the installed claude"
        # From the checkout, as scripts/claude_tools.sh registers it: a local-scope
        # server is keyed by the project's path. And as long as that script
        # allows, since `mcp get` health-checks the server by starting npx.
        registered = ctx.cmd(["claude", "mcp", "get", "playwright"], MCP_GET_SECONDS,
                             env=ctx.env_without_keys(), cwd=str(ctx.root))
        text = registered.text
        if registered.timed_out:
            data["playwright_mcp"] = f"no answer within {MCP_GET_SECONDS:g}s"
            problems.append(finding("mcp-failed", "Claude Code did not say whether the "
                                    "Playwright MCP server is registered here",
                                    data["playwright_mcp"]))
        elif not registered.ok and re.search(r"no mcp server|not found", text, re.I):
            data["playwright_mcp"] = "not registered"
            problems.append(finding("mcp-unregistered", "the Playwright MCP server is not "
                                    "registered with Claude Code here", first_line(registered)))
        elif not registered.ok:
            data["playwright_mcp"] = f"could not be read: {first_line(registered)}"
            problems.append(finding("mcp-failed", "Claude Code could not say whether the "
                                    "Playwright MCP server is registered here",
                                    first_line(registered)))
        else:
            state = re.search(r"^\s*Status:\s*(.+)$", text, re.M)
            scope = re.search(r"^\s*Scope:\s*(.+)$", text, re.M)
            data["playwright_mcp"] = {"status": state[1].strip() if state else None,
                                      "scope": scope[1].strip() if scope else None}
            if state and re.search(r"fail|✗|error", state[1], re.I):
                problems.append(finding("mcp-failed", "the Playwright MCP server is registered "
                                        "but does not connect", state[1]))

    pacman = ctx.cmd(["pacman", "-Q", *PACKAGES], 20)
    if pacman.missing:
        data["packages"] = None
    else:
        installed = dict(line.split(None, 1) for line in pacman.out.splitlines() if " " in line)
        data["packages"] = {"installed": installed,
                            "missing": re.findall(r"package '([^']+)' was not found", pacman.err)}
        ctx.log("pacman.txt", pacman.text)
    freeze = ctx.cmd([ctx.python, "-m", "pip", "freeze"], 60)
    if freeze.ok:
        data["pip_freeze"] = ctx.log("pip-freeze.txt", freeze.out)

    present = sorted(k for k, v in versions.items() if v)
    absent = sorted(k for k, v in versions.items() if not v)
    summary = f"{len(present)} found; missing: {', '.join(absent) or 'none'}"
    return concluded(summary, data, problems)


def section_sign_ins(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []
    notes: list[str] = []

    # The three key variables are taken out of the child's environment: any of
    # them makes `auth status` describe the variable, not the sign-in that
    # Claude in Chrome needs.
    status = ctx.cmd(["claude", "auth", "status"], 30, env=ctx.env_without_keys())
    if status.missing:
        data["claude"] = None
        problems.append(finding("claude-missing", "Claude Code is not installed"))
    else:
        text = status.out
        parsed: Any = None
        if "{" in text:
            try:
                parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except ValueError:
                parsed = None
        parsed = parsed if isinstance(parsed, dict) else {}
        # Only these two: the rest can carry an address or an organisation.
        claude = {"logged_in": parsed.get("loggedIn"), "method": parsed.get("authMethod")}
        data["claude"] = claude
        if not claude["logged_in"]:
            problems.append(finding("claude-signin", "Claude Code is not signed in",
                                    first_line(status) if not parsed else ""))
        elif claude["method"] != "claude.ai":
            problems.append(finding("claude-signin", f"Claude Code is signed in through "
                                    f"{claude['method']}, and Claude in Chrome needs a claude.ai "
                                    "sign-in"))

    exported = [name for name in KEY_VARIABLES if ctx.env.get(name)]
    in_files: dict[str, list[str]] = {}
    for rc in RC_FILES:
        rc_text = ctx.read(ctx.home / rc)
        if rc_text:
            names = rc_exports(rc_text, KEY_VARIABLES)
            if names:
                in_files[f"~/{rc}"] = names
    data["key_variables"] = {"exported_here": exported, "in_rc_files": in_files}
    if exported or in_files:
        where = [f"exported in this shell ({', '.join(exported)})"] if exported else []
        where += [f"{', '.join(names)} in {rc}" for rc, names in in_files.items()]
        problems.append(finding("claude-key-env", "a key variable overrides Claude Code's own "
                                "sign-in", "; ".join(where)))

    gh = ctx.cmd(["gh", "auth", "status"], 20)
    data["gh_signed_in"] = None if gh.missing else gh.ok
    if not gh.missing and not gh.ok:
        problems.append(finding("gh-signin", "gh is not signed in to github.com"))

    manifests = [f"~/.config/{profile}/NativeMessagingHosts/{NATIVE_HOST_MANIFEST}"
                 for profile in BROWSER_PROFILES
                 if (ctx.home / ".config" / profile / "NativeMessagingHosts"
                     / NATIVE_HOST_MANIFEST).exists()]
    data["claude_in_chrome_host"] = manifests
    if not manifests:
        notes.append("no Claude in Chrome native host is installed: install the extension from "
                     "the Chrome Web Store, then run `claude --chrome` and `/chrome` once")

    facts = ctx.facts()
    cloud = [seat["model"] for seat in facts.get("seats") or []
             if seat.get("provider") == "ollama" and is_cloud_tag(str(seat.get("model")))]
    if cloud:
        me = ctx.http("POST", f"{ctx.ollama_url()}/api/me", {}, 5)
        data["ollama_com"] = {"status": me.status, "cloud_seats": cloud}
        if me.status not in (200, 404, None):
            problems.append(finding("ollama-signin", f"{', '.join(cloud)} runs on ollama.com and "
                                    "the daemon is not signed in"))
    if notes:
        data["notes"] = notes

    claude_word = ("not installed" if data["claude"] is None
                   else f"claude {'signed in' if data['claude']['logged_in'] else 'signed out'}"
                   + (f" ({data['claude']['method']})" if data["claude"]["method"] else ""))
    gh_word = {None: "gh not installed", True: "gh signed in", False: "gh signed out"}[
        data["gh_signed_in"]]
    summary = f"{claude_word}; {gh_word}; chrome host {'found' if manifests else 'absent'}"
    return concluded(summary, data, problems)


def section_network(ctx: Context) -> Section:
    script = ctx.root / "scripts" / "network_check.sh"
    known = network_groups(ctx.read(script) or "")
    if not known:
        return skipped("scripts/network_check.sh is not in this checkout")
    required = [g for g in NETWORK_REQUIRED if g in known]
    optional = [g for g in NETWORK_OPTIONAL if g in known]
    data: dict[str, Any] = {
        "required": required, "optional": optional,
        "not_in_this_check": sorted(set(NETWORK_REQUIRED + NETWORK_OPTIONAL) - known),
    }
    ran = ctx.cmd(["bash", str(script), *required, "--optional", *optional], 90)
    data["log"] = ctx.log("network_check.txt", ran.text)
    data["exit"] = ran.code
    failed = [line.strip() for line in ran.out.splitlines() if line.strip().startswith("✗")]
    missed = [line.strip() for line in ran.out.splitlines() if line.strip().startswith("- ")]
    data["failed"], data["optional_missed"] = failed, missed
    problems: list[Problem] = []
    if ran.code == 1 or failed:
        problems.append(finding("network", "a host the install needs does not answer",
                                "\n".join(failed)))
    elif not ran.ok:
        problems.append(finding("network", "the network check could not run",
                                first_line(ran) or f"exit {ran.code}"))
    summary = (f"{len(required)} required groups, {len(failed)} failing; "
               f"{len(optional)} optional, {len(missed)} not answering")
    return concluded(summary, data, problems)


def section_gpu(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []
    lspci = ctx.cmd(["lspci", "-nnk"], 10)
    devices = parse_lspci(lspci.out) if lspci.ok else []
    data["display_devices"] = devices if not lspci.missing else None
    if lspci.ok:
        ctx.log("lspci.txt", "\n".join(f"{d['slot']} {d['class']}: {d['device']} "
                                       f"(driver {d['driver']})" for d in devices))
    nvidia_listed = any(d["nvidia"] for d in devices)

    smi = ctx.cmd(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], 20)
    if smi.missing:
        if nvidia_listed:
            problems.append(finding("nvidia-smi-missing", "lspci shows an NVIDIA card, but there "
                                    "is no nvidia-smi", ", ".join(d["device"] for d in devices
                                                                  if d["nvidia"])))
            return concluded("an NVIDIA card without its driver's tools", data, problems)
        others = ", ".join(d["device"] for d in devices)
        problems.append(finding("no-gpu", "no NVIDIA card was found",
                                others or ("neither nvidia-smi nor lspci is installed"
                                           if lspci.missing else "lspci lists no display device")))
        return concluded("no NVIDIA card", data, problems)
    if not smi.ok:
        problems.append(finding("nvidia-smi-failed", "nvidia-smi is installed but failed",
                                smi.text[-400:]))
        return concluded("nvidia-smi failed", data, problems)

    cards = parse_gpu_csv(smi.out)
    ctx.gpus = cards
    data["cards"] = cards
    full = ctx.cmd(["nvidia-smi"], 20)
    ctx.log("nvidia-smi.txt", full.text + "\n\n" + smi.out)
    apps = ctx.cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], 20)
    data["compute_apps"] = parse_compute_apps(apps.out) if apps.ok else None

    old = [card for card in cards if compute_below(card.get("compute_cap"))]
    if old:
        journal = ctx.ollama_journal()
        named = ", ".join(f"{c['index']}: {c['name']} (compute {c['compute_cap']})" for c in old)
        if journal.get("skipping"):
            problems.append(finding("cuda-old-card", "the daemon skips a card too old for its "
                                    "CUDA build", f"{named}; journal: {journal['skipping'][-1]}"))
        else:
            data["note"] = (f"{named} is below compute 7.5; the journal "
                            + ("does not say the daemon skips it" if journal.get("readable")
                               else f"could not be read to confirm ({journal.get('why')})"))

    if not ctx.args.quick and ctx.background and cards:
        ctx.sampler = GpuSampler(lambda: ctx.current)
        data["sampler"] = "running" if ctx.sampler.start() else "could not start"

    total = sum(card.get("memory_total_mib") or 0 for card in cards)
    names = "; ".join(f"{c['index']}: {c['name']} ({c['memory_total_mib']} MiB, compute "
                      f"{c['compute_cap']}{', display' if c['display_active'] else ''})"
                      for c in cards)
    summary = f"{len(cards)} card(s), {total} MiB: {names}" if cards else "nvidia-smi lists no card"
    if not cards:
        problems.append(finding("no-gpu", "nvidia-smi answered but listed no card", smi.text[:300]))
    return concluded(summary, data, problems)


def section_ollama(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []
    url = ctx.ollama_url()
    data["url"] = url
    remote = bool(ctx.env.get("OLLAMA_BASE_URL")) and not _loopback(url)
    data["remote"] = remote

    version = ctx.cmd(["ollama", "--version"], 10)
    data["cli"] = first_line(version) if version.ok else None
    answer = ctx.http("GET", f"{url}/api/version", None, 5)
    ctx.daemon_up = answer.status == 200
    data["daemon"] = (answer.json() or {}).get("version") if ctx.daemon_up else None

    if version.missing and not remote:
        problems.append(finding("ollama-missing", "ollama is not installed"))

    # systemd's view: only of a daemon this machine runs.
    unit: dict[str, Any] | None = None
    if not remote and ctx.cmd(["systemctl", "--version"], 5).ok:
        enabled = first_line(ctx.cmd(["systemctl", "is-enabled", "ollama.service"], 10))
        active = first_line(ctx.cmd(["systemctl", "is-active", "ollama.service"], 10))
        unit = {"enabled": enabled or None, "active": active or None}
        if enabled and "not-found" not in enabled and "No such" not in enabled:
            keepalive = ctx.cmd(["bash", str(ctx.root / "scripts" / "ollama_keepalive.sh"),
                                 "check"], 20)
            unit["keepalive"] = keepalive.text
            if not keepalive.ok:
                problems.append(finding("keepalive", "systemd does not keep the daemon running",
                                        "\n".join(line for line in keepalive.out.splitlines()
                                                  if "✗" in line)))
            shown = ctx.cmd(["systemctl", "show", "ollama.service", "-p", "Environment"], 10)
            settings = one_model_settings(shown.out)
            unit["one_model"] = {name: settings.get(name) for name in ONE_MODEL_VARIABLES}
            gaps = one_model_gaps(settings)
            if gaps:
                problems.append(finding("one-model", "the daemon is not limited to one model at "
                                        "a time", "; ".join(gaps)))
    data["systemd"] = unit

    if not ctx.daemon_up:
        if not version.missing or remote:
            problems.append(finding("daemon-down", f"the Ollama daemon is not answering at {url}",
                                    answer.error or f"HTTP {answer.status}"))
    else:
        tags_answer = ctx.http("GET", f"{url}/api/tags", None, 10)
        tags = [str(m.get("name")) for m in (tags_answer.json() or {}).get("models") or []]
        data["pulled"] = sorted(tags)
        ctx.log_json("ollama-tags.json", tags_answer.json())
        facts = ctx.facts()
        wanted: dict[str, list[str]] = {}
        for seat in facts.get("seats") or []:
            if seat.get("provider") == "ollama" and seat.get("model"):
                wanted.setdefault(str(seat["model"]), []).append(str(seat["role"]))
        if facts.get("embedding_model"):
            wanted.setdefault(str(facts["embedding_model"]), []).append("embedder")
        if not facts:
            data["configured"] = f"unknown: the project could not be asked ({ctx.facts_error})"
        configured: dict[str, Any] = {}
        missing: list[str] = []
        for tag, roles in wanted.items():
            pulled = any(same_tag(tag, have) for have in tags)
            caps = None
            if pulled:
                show = ctx.http("POST", f"{url}/api/show", {"model": tag}, 10)
                listed = (show.json() or {}).get("capabilities")
                caps = [str(c) for c in listed] if isinstance(listed, list) else None
            configured[tag] = {"for": roles, "pulled": pulled, "capabilities": caps}
            if not pulled:
                missing.append(f"{tag} ({', '.join(roles)})")
            if "builder" in roles and caps is not None and "tools" not in caps:
                problems.append(finding("builder-no-tools", f"the Builder's model {tag} cannot "
                                        "call tools", f"capabilities: {', '.join(caps)}"))
        if wanted:
            data["configured"] = configured
        if missing:
            problems.append(finding("model-not-pulled", "a configured model is not on the daemon",
                                    "; ".join(missing)))
        ps = ctx.http("GET", f"{url}/api/ps", None, 10)
        loaded = placement(ps.json())
        data["loaded"] = loaded
        local = [row for row in loaded if not is_cloud_tag(row["model"])]
        if len(local) > 1:
            problems.append(finding("two-resident", "more than one model is on the cards at once",
                                    ", ".join(row["model"] for row in local)))

    journal = ctx.ollama_journal()
    data["journal"] = {"readable": journal.get("readable"), "why": journal.get("why") or None,
                       "skipping": journal.get("skipping", [])[-3:],
                       "oom": journal.get("oom", [])[-3:],
                       "offloaded": journal.get("offloaded", [])[-5:]}
    if journal.get("oom"):
        problems.append(finding("gpu-oom", "the daemon ran out of GPU memory in the last two hours",
                                journal["oom"][-1]))

    state = (f"daemon {data['daemon']} answering" if ctx.daemon_up
             else "daemon not answering")
    summary = f"{state} at {url}; {len(data.get('pulled') or [])} models pulled, " \
              f"{len(data.get('loaded') or [])} loaded"
    return concluded(summary, data, problems)


def _load_gate(ctx: Context) -> Section | None:
    """A skip for a section that loads models, when the console is busy or the
    daemon is known to be down; None when it may go ahead."""
    reasons = ctx.busy_reasons()
    if reasons:
        return skipped(f"skipped because {' and '.join(reasons)}: it would compete with the "
                       "console for the cards", {"busy": reasons})
    if ctx.daemon_up is False:
        return skipped("skipped: the Ollama daemon is not answering (see the ollama section)")
    return None


def section_embedder(ctx: Context) -> Section:
    gate = _load_gate(ctx)
    if gate is not None:
        return gate
    found, error = ctx.snippet("embedder", EMBEDDER_SNIPPET, 600)
    if found is None:
        return Section("error", f"the embedder could not be asked: {error}")
    ctx.log_json("embedder.json", found)
    problems: list[Problem] = []
    model = found.get("model")
    if found.get("error"):
        kind = "daemon-down" if found.get("unreachable") else "embedder-failed"
        problems.append(finding(kind, f"{model} could not embed", str(found["error"])))
        return concluded(f"{model} could not embed", found, problems)
    if found.get("dimensions") != found.get("expected_dimensions"):
        problems.append(finding("embedder-dimensions", f"{model} answered with "
                                f"{found.get('dimensions')} dimensions, the corpus is built for "
                                f"{found.get('expected_dimensions')}"))
    share = found.get("cpu_share")
    if isinstance(share, (int, float)) and share >= 0.99:
        problems.append(finding("embedder-cpu", f"{model} runs wholly on the CPU"))
    elif isinstance(share, (int, float)) and share > 0:
        problems.append(finding("embedder-split", f"{model} is split onto the CPU",
                                str(found.get("placement_note") or f"{share:.0%} on the CPU")))
    where = ("placement unknown" if share is None else "100% on the GPU" if share == 0
             else f"{share:.0%} on the CPU")
    summary = (f"{model}: {found.get('dimensions')} dimensions, cold {found.get('cold_s')}s"
               f"{' (was loaded)' if found.get('loaded_before') else ''}, warm "
               f"{found.get('warm_s')}s, {where}")
    return concluded(summary, found, problems)


def _generate_timing(ctx: Context, url: str, model: str,
                     options: dict[str, Any]) -> dict[str, Any]:
    """One short generation: load time, tokens per second, and where it landed."""
    def resident() -> dict[str, Any] | None:
        rows = placement(ctx.http("GET", f"{url}/api/ps", None, 10).json())
        return next((row for row in rows if same_tag(row["model"], model)), None)

    loaded_before = resident() is not None
    body = {"model": model, "prompt": "Name three colours, one word each.", "stream": False,
            "options": {**options, "num_predict": 64}}
    result: dict[str, Any] = {"model": model, "loaded_before": loaded_before, "forced": True}
    answer = ctx.http("POST", f"{url}/api/generate", body, 600)
    if answer.status != 200 and any(mark in answer.body.lower() for mark in _GPU_FIT_MARKS):
        # What a seat does when its forced load does not fit: once more, unforced.
        result["forced"] = False
        result["forced_error"] = answer.body[:300]
        answer = ctx.http("POST", f"{url}/api/generate",
                          {**body, "options": {"num_predict": 64}}, 600)
    reply = answer.json() if answer.status == 200 else None
    if not isinstance(reply, dict):
        result["error"] = answer.error or answer.body[:300] or f"HTTP {answer.status}"
        return result

    def per_second(count: Any, nanos: Any) -> float | None:
        return round(count / (nanos / 1e9), 1) if count and nanos else None

    result.update(
        load_s=round((reply.get("load_duration") or 0) / 1e9, 2),
        total_s=round((reply.get("total_duration") or 0) / 1e9, 2),
        prompt_tokens_per_s=per_second(reply.get("prompt_eval_count"),
                                       reply.get("prompt_eval_duration")),
        eval_tokens_per_s=per_second(reply.get("eval_count"), reply.get("eval_duration")),
    )
    landed = resident()
    if landed:
        result.update(size_gib=landed["size_gib"], vram_gib=landed["vram_gib"],
                      cpu_share=landed["cpu_share"])
    if not loaded_before:
        ctx.http("POST", f"{url}/api/generate", {"model": model, "keep_alive": 0}, 60)
        result["unloaded"] = True
    return result


def section_seats(ctx: Context) -> Section:
    gate = _load_gate(ctx)
    if gate is not None:
        return gate
    facts = ctx.facts()
    seats = facts.get("seats") or []
    if not seats:
        return Section("error", f"the seats could not be read from the project: {ctx.facts_error}")
    data: dict[str, Any] = {"seats": seats}
    problems: list[Problem] = []

    keys = sorted({str(s["candidate"]) for s in seats if s.get("candidate")})
    probes: list[dict[str, Any]] = []
    if keys:
        seats_dir = ctx.out / "seats"
        ran = ctx.cmd([ctx.python, str(ctx.root / "scripts" / "diagnose_seats.py"), "--phase",
                       "probe", "--models", ",".join(keys), "--out", str(seats_dir)], 2400,
                      cwd=str(ctx.root))
        data["diagnose_seats"] = {"exit": ran.code, "log": ctx.log("diagnose_seats.txt", ran.text)}
        report = read_text(seats_dir / "results.json")
        try:
            probes = list(json.loads(report or "{}").get("probes") or [])
        except ValueError:
            probes = []
        if not probes:
            problems.append(finding("seat-probe", "the seat diagnostic reported no probes",
                                    "\n".join(ran.text.splitlines()[-5:])))
    by_pair = {(p.get("model_key"), p.get("role")): p for p in probes}
    data["probes"] = [{k: p.get(k) for k in ("model_key", "role", "status", "seconds", "detail")}
                      for p in probes]
    for seat in seats:
        probe = by_pair.get((seat.get("candidate"), seat.get("role")))
        if probe is not None and probe.get("status") != "ok":
            problems.append(finding("seat-probe", f"the {seat['role']} seat ({seat['model']}) "
                                    f"probes {probe.get('status')}",
                                    str(probe.get("failure_reason") or probe.get("detail") or "")))

    uncovered = [str(s["role"]) for s in seats if not s.get("candidate")]
    if uncovered:
        found, error = ctx.snippet("probe", PROBE_SNIPPET, 600, ",".join(uncovered))
        data["one_word_probes"] = found or {"error": error}
        for role, seat in (found or {}).items():
            if not seat.get("live"):
                kind = {"NO KEY": "seat-no-key", "OFFLINE": "daemon-down",
                        "NOT PULLED": "model-not-pulled"}.get(str(seat.get("badge")), "seat-down")
                problems.append(finding(kind, f"the {role} seat ({seat.get('model')}) cannot run",
                                        f"{seat.get('badge')}: {seat.get('reason')}"))
            elif seat.get("silent"):
                problems.append(finding("seat-down", f"the {role} seat ({seat.get('model')}) "
                                        "answered a one-word prompt with nothing"))
            elif seat.get("error"):
                problems.append(finding("seat-down", f"the {role} seat ({seat.get('model')}) "
                                        "failed its one-word prompt", str(seat["error"])))

    url = ctx.ollama_url()
    options = facts.get("seat_gpu_options") or {"num_gpu": 999}
    vram_gib = sum(card.get("memory_total_mib") or 0 for card in ctx.gpus) / 1024
    timings = []
    for model in dict.fromkeys(str(s["model"]) for s in seats if s.get("local")):
        timing = _generate_timing(ctx, url, model, options)
        timings.append(timing)
        share = timing.get("cpu_share")
        # A model larger than the cards is meant to split; one that would fit is not.
        fits = vram_gib and (timing.get("size_gib") or 0) <= vram_gib
        if isinstance(share, (int, float)) and share > 0 and fits:
            problems.append(finding("seat-split", f"{model} ran {share:.0%} on the CPU although "
                                    f"it would fit the cards ({timing.get('size_gib')} GiB of "
                                    f"{vram_gib:.1f})"))
    data["generate"] = timings
    ctx.log_json("generate-timing.json", timings)

    rates = ", ".join(f"{t['model'].rsplit('/', 1)[-1]} {t.get('eval_tokens_per_s')} tok/s"
                      for t in timings if t.get("eval_tokens_per_s"))
    ok_probes = sum(1 for p in probes if p.get("status") == "ok")
    summary = (f"{ok_probes}/{len(probes)} probes ok" if probes else "no candidate probes") + (
        f"; {rates}" if rates else "")
    return concluded(summary, data, problems)


def section_database(ctx: Context) -> Section:
    problems: list[Problem] = []
    found, error = ctx.snippet("database", DATABASE_SNIPPET, 45)
    data: dict[str, Any] = dict(found or {"error": error})
    facts = ctx.facts()
    ours = facts.get("this_schema")
    for entry in data.get("schemas") or []:
        entry["this_checkout"] = entry.get("schema") == ours
        if entry["this_checkout"] and entry.get("model") and facts.get("embedding_model") \
                and entry["model"] != facts["embedding_model"]:
            problems.append(finding("corpus-other-model", f"this checkout's corpus was built by "
                                    f"{entry['model']}, not {facts['embedding_model']}"))
    if found is None:
        problems.append(finding("venv-missing" if "is not there" in error else "postgres-down",
                                "the database could not be asked", error))
    elif found.get("import"):
        problems.append(finding("venv-missing", "the venv cannot import the database client",
                                str(found.get("error"))))
    elif found.get("error"):
        problems.append(finding("postgres-down", f"the database at {found.get('url')} does not "
                                "answer" if found.get("unreachable") else "the database refused",
                                str(found["error"])))
    elif not found.get("pgvector"):
        problems.append(finding("no-pgvector", "the database has no pgvector extension",
                                f"available: {found.get('pgvector_available') or 'no'}"))

    container = ctx.cmd(["docker", "ps", "-a", "--filter", f"name={POSTGRES_CONTAINER}",
                         "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"], 15)
    if container.missing:
        data["container"] = None
    elif not container.ok and "permission denied" in container.text.lower():
        data["container"] = "docker refused: permission denied"
        # Only worth fixing when the database is down: docker behind sudo is
        # Omarchy's default, and install.sh --no-docker-group keeps it.
        if problems:
            problems.append(finding("docker-group", "docker cannot be asked without sudo, so "
                                    "the database container could not be looked at",
                                    first_line(container)))
    else:
        data["container"] = container.out.strip() or "none"
        if container.out.strip():
            logs = ctx.cmd(["docker", "logs", "--tail", "50", POSTGRES_CONTAINER], 15)
            data["container_log"] = ctx.log("postgres18-docker.txt", logs.text)
    ctx.log_json("database.json", data)

    schemas = data.get("schemas") or []
    if found and not found.get("error") and not found.get("import"):
        mine = next((s for s in schemas if s.get("this_checkout")), None)
        corpus = (f"this checkout's corpus {mine.get('chunks')} chunks, floor {mine.get('floor')}"
                  if mine else "no corpus for this checkout")
        summary = (f"postgres {found.get('server')}, pgvector {found.get('pgvector') or 'missing'}, "
                   f"{len(schemas)} corpus schema(s); {corpus}")
    else:
        summary = "the database could not be asked"
    return concluded(summary, data, problems)


def section_searxng(ctx: Context) -> Section:
    facts = ctx.facts()
    url = (ctx.env.get("SEARXNG_URL") or str(facts.get("searxng_url") or "")).strip().rstrip("/")
    if not url:
        return skipped("SEARXNG_URL is not set: online research asks DuckDuckGo directly")
    data: dict[str, Any] = {"url": url}
    problems: list[Problem] = []
    answer = ctx.http("GET", f"{url}/search?q=ambiguity&format=json", None, 15)
    data["status"] = answer.status
    if answer.status == 200:
        payload = answer.json()
        data["results"] = len(payload.get("results") or []) if isinstance(payload, dict) else None
    elif answer.status == 403:
        problems.append(finding("searxng-json", "SearxNG refuses JSON (HTTP 403): json is missing "
                                "from its search.formats"))
    else:
        problems.append(finding("searxng-down", f"SearxNG does not answer at {url}",
                                answer.error or f"HTTP {answer.status}"))
    unit = ctx.cmd(["systemctl", "--user", "is-active", f"{SEARXNG_CONTAINER}.service"], 10)
    data["unit"] = None if unit.missing else first_line(unit) or None
    container = ctx.cmd(["podman", "ps", "-a", "--filter", f"name={SEARXNG_CONTAINER}",
                         "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"], 15)
    data["container"] = None if container.missing else (container.out.strip() or "none")
    if container.ok and container.out.strip():
        logs = ctx.cmd(["podman", "logs", "--tail", "50", SEARXNG_CONTAINER], 15)
        data["container_log"] = ctx.log("searxng-podman.txt", logs.text)
    results = data.get("results")
    summary = (f"answers JSON at {url} ({results} results)" if answer.status == 200
               else f"{url}: {answer.error or f'HTTP {answer.status}'}")
    return concluded(summary, data, problems)


def _console_problems(reads: dict[str, Any]) -> list[Problem]:
    """What the console's own read RPCs say is wrong, in the console's words."""
    problems: list[Problem] = []
    healing = reads.get("healing") or {}
    for circuit in healing.get("circuits") or []:
        if circuit.get("state") in ("open", "half-open"):
            problems.append(finding("circuit-open", f"the {circuit.get('name')} circuit is "
                                    f"{circuit.get('state')}",
                                    f"{circuit.get('failures')} failures, retry in "
                                    f"{circuit.get('retry_in_s')}s"))
    health_kinds = {"ollama-daemon": "daemon-down", "postgres": "postgres-down",
                    "searxng": "searxng-down", "corpus": "corpus-unavailable"}
    for name, result in (healing.get("health") or {}).items():
        if isinstance(result, dict) and result.get("status") not in (None, "healthy"):
            problems.append(finding(health_kinds.get(name, "console-errors"),
                                    f"the console reports {name} {result.get('status')}",
                                    str(result.get("details") or "")))
    stats = reads.get("rag_stats") or {}
    if stats.get("corpus") == "unavailable":
        problems.append(finding("corpus-unavailable", "the corpus is unavailable",
                                str(stats.get("note") or "")))
    staleness = stats.get("staleness") or {}
    if isinstance(staleness, dict) and staleness.get("stale"):
        counts = {k: len(v) if isinstance(v, list) else v for k, v in staleness.items()
                  if k in ("missing", "extra", "oversized")}
        problems.append(finding("corpus-stale", "the corpus no longer matches the archive",
                                json.dumps(counts)))
    # Seats that are down for one reason -- four seats on one daemon that is
    # down -- are one problem, not four.
    down: dict[tuple[str, str], list[str]] = {}
    for seat in (reads.get("list_seats") or {}).get("seats") or []:
        if not seat.get("live"):
            down.setdefault((str(seat.get("badge")), str(seat.get("reason") or "")),
                            []).append(str(seat.get("role")))
        if seat.get("tools_note"):
            problems.append(finding("builder-no-tools", str(seat["tools_note"])))
    for (badge, reason), roles in down.items():
        kind = {"NO KEY": "seat-no-key", "OFFLINE": "daemon-down",
                "NOT PULLED": "model-not-pulled"}.get(badge, "seat-down")
        seats = f"the {roles[0]} seat" if len(roles) == 1 else f"the {', '.join(roles)} seats"
        problems.append(finding(kind, f"the console shows {seats} {badge}", reason))
    status = reads.get("status") or {}
    device = status.get("embedding_device") or {}
    share = device.get("cpu_share") if isinstance(device, dict) else None
    if isinstance(share, (int, float)) and share > 0:
        problems.append(finding("embedder-cpu" if share >= 0.99 else "embedder-split",
                                f"the console's embedder runs {share:.0%} on the CPU",
                                str(device.get("note") or "")))
    for pr in status.get("pull_requests") or []:
        if isinstance(pr, dict) and (pr.get("checks_failed") or pr.get("status") == "checks_failed"):
            problems.append(finding("pr-checks-failed", f"pull request #{pr.get('number')} has "
                                    "failing checks", str(pr.get("detail") or "")))
    return problems


def _last_run_summary(snapshot: Any) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    keep = ("run_id", "finished_at", "elapsed_s", "verdict", "stopped", "stop_reason",
            "over_budget", "builder_cut_off", "research_status", "step_count", "discuss_only",
            "error")
    summary = {key: snapshot.get(key) for key in keep if key in snapshot}
    dwell = snapshot.get("dwell")
    if isinstance(dwell, dict) and dwell:
        summary["dwell"] = dwell.get("status")
    summary["files_changed"] = len(snapshot.get("files_changed") or [])
    return summary


def section_console_before(ctx: Context) -> Section:
    data: dict[str, Any] = {"base": ctx.base}
    problems: list[Problem] = []

    listeners = ctx.cmd(["ss", "-ltnpH"], 10)
    data["listeners"] = parse_listeners(listeners.out, CONSOLE_PORTS) if listeners.ok else None
    if listeners.ok:
        ctx.log("ss.txt", listeners.out)
    env_file = ctx.read(ctx.root / ".env")
    data["env_names"] = env_names(env_file) if env_file is not None else None

    answer = ctx.http("GET", f"{ctx.base}/api/status", None, 5)
    if answer.status != 200 or not isinstance(answer.json(), dict):
        ctx.console_up = False
        port = str(urlsplit(ctx.base).port or "")
        holders = (data["listeners"] or {}).get(port) or []
        held = f"; port {port} is held by {holders}" if holders else ""
        problems.append(finding("console-down", f"the console is not answering at {ctx.base}",
                                (answer.error or f"HTTP {answer.status}") + held))
        return concluded(f"not answering at {ctx.base}", data, problems)

    ctx.console_up = True
    reads: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for method in CONSOLE_READS:
        result, error = ctx.rpc(method)
        if error:
            errors[method] = error
        reads[method] = result
        if method != "last_run":
            ctx.log_json(f"console-{method}.json", result)
    snapshot = (reads.get("last_run") or {}).get("snapshot") if reads.get("last_run") else None
    if isinstance(snapshot, dict):
        ctx.log_json("console-last_run.json", {**snapshot,
                                               "messages": (snapshot.get("messages") or [])[-50:]})
    if errors:
        data["rpc_errors"] = errors
    events = (reads.get("healing") or {}).get("events") or []
    if ctx.journal_seq is None:
        ctx.journal_seq = max((int(e.get("seq") or 0) for e in events), default=0)

    status = reads.get("status") or {}
    stats = reads.get("rag_stats") or {}
    data["corpus"] = {key: stats.get(key) for key in ("corpus", "total_documents", "total_chunks",
                                                      "total_nodes", "archive")}
    data["embedding"] = {"model": status.get("embedding"), "device": status.get("embedding_device")}
    data["indexing"] = status.get("indexing")
    data["run_in_flight"] = status.get("run_in_flight")
    data["pull_requests"] = status.get("pull_requests")
    data["seats"] = [{k: seat.get(k) for k in ("role", "provider", "model", "live", "badge",
                                               "reason", "tools")}
                     for seat in (reads.get("list_seats") or {}).get("seats") or []]
    data["circuits"] = (reads.get("healing") or {}).get("circuits")
    data["health"] = (reads.get("healing") or {}).get("health")
    data["last_run"] = _last_run_summary(snapshot)
    problems += _console_problems(reads)

    live = sum(1 for seat in data["seats"] if seat.get("live"))
    summary = (f"answering at {ctx.base}; corpus {data['corpus'].get('corpus')}, "
               f"{live}/{len(data['seats'])} seats live"
               + ("; a run is in flight" if status.get("run_in_flight") else ""))
    return concluded(summary, data, problems)


def _redact_tree(ctx: Context, directory: Path) -> None:
    """Pass every text file a helper wrote under `directory` through `redact`, in place.

    JSON is redacted as data and written back as JSON: a replacement made in
    its text can land inside an escape, and a results.json that no longer
    parses loses every pass it carried.
    """
    if not directory.is_dir():
        return
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".json", ".jsonl", ".txt", ".log", ".csv"):
            text = read_text(path, limit=50_000_000)
            if text is not None:
                if path.suffix in (".json", ".jsonl"):
                    cleaned = redact_json_text(text, ctx.counts, lines=path.suffix == ".jsonl")
                else:
                    cleaned = redact(text, ctx.counts)
                if cleaned != text:
                    path.write_text(cleaned, encoding="utf-8")


def _browser_check(ctx: Context, extra: list[str], out_name: str,
                   timeout: float) -> tuple[Ran, dict[str, Any]]:
    out_dir = ctx.out / out_name
    ran = ctx.cmd([ctx.python, str(ctx.root / "scripts" / "browser_agent.py"), "check",
                   "--base", ctx.base, "--out", str(out_dir), *extra], timeout, cwd=str(ctx.root))
    ctx.log(f"{out_name}.txt", ran.text)
    _redact_tree(ctx, out_dir)
    try:
        results = json.loads(read_text(out_dir / "results.json") or "{}")
    except ValueError:
        results = {}
    return ran, results if isinstance(results, dict) else {}


def _passes_problems(ran: Ran, results: dict[str, Any], label: str) -> list[Problem]:
    if ran.code == 2 or ran.missing or ran.timed_out:
        return [finding("browser-cannot-run", f"the browser agent could not run its {label}",
                        "\n".join(ran.text.splitlines()[-5:]))]
    problems = [finding("browser-pass", f"the browser pass {p.get('name')} failed",
                        str(p.get("reason") or ""))
                for p in results.get("passes") or []
                if isinstance(p, dict) and p.get("status") == "fail"]
    if ran.code == 1 and not problems:
        problems.append(finding("browser-pass", f"the browser agent's {label} failed",
                                "\n".join(ran.text.splitlines()[-5:])))
    return problems


def section_browser(ctx: Context) -> Section:
    if ctx.args.no_browser:
        return skipped("skipped: --no-browser")
    if not (ctx.root / "scripts" / "browser_agent.py").exists():
        return skipped("scripts/browser_agent.py is not in this checkout")
    if ctx.console_up is not True:
        answer = ctx.http("GET", f"{ctx.base}/api/status", None, 5)
        ctx.console_up = answer.status == 200
    if not ctx.console_up:
        return skipped(f"skipped: the console is not answering at {ctx.base}")

    data: dict[str, Any] = {}
    # The default set searches through the embedder, which the console's
    # arbiter puts behind a run's seat or a rebuild and which then evicts the
    # model they hold: while either is in flight the passes are the quick set.
    busy = ctx.busy_reasons()
    quick = bool(ctx.args.quick or busy)
    if busy and not ctx.args.quick:
        data["heavy_passes"] = f"skipped because {' and '.join(busy)}"
    ran, results = _browser_check(ctx, ["--quick"] if quick else [], "browser",
                                  300 if quick else 900)
    passes = [p for p in results.get("passes") or [] if isinstance(p, dict)]
    data["exit"] = ran.code
    data["browser"] = results.get("browser")
    data["passes"] = [{k: p.get(k) for k in ("name", "status", "reason", "duration_s")}
                      for p in passes]
    rpc = results.get("rpc") or {}
    data["rpc_error_envelopes"] = len(rpc.get("error_envelopes") or [])
    data["page_errors"] = len(results.get("page_errors") or [])
    problems = _passes_problems(ran, results, "read-only passes")

    if ctx.args.with_runs:
        reasons = ctx.busy_reasons()
        if reasons:
            data["runs"] = f"skipped because {' and '.join(reasons)}"
        else:
            run_ran, run_results = _browser_check(
                ctx, ["--passes", "run", "--allow", "run"], "browser-run", 2400)
            data["runs"] = {"exit": run_ran.code, "passes": [
                {k: p.get(k) for k in ("name", "status", "reason", "duration_s")}
                for p in run_results.get("passes") or [] if isinstance(p, dict)]}
            problems += _passes_problems(run_ran, run_results, "run pass")

    tally = Counter(str(p.get("status")) for p in passes)
    summary = (", ".join(f"{n} {s}" for s, n in sorted(tally.items())) or "no passes reported") + (
        f"; {data['rpc_error_envelopes']} rpc error envelopes" if data["rpc_error_envelopes"]
        else "")
    return concluded(summary, data, problems)


def section_console_after(ctx: Context) -> Section:
    data: dict[str, Any] = {}
    problems: list[Problem] = []
    path = Path(ctx.env.get("CONSOLE_LOG") or DEFAULT_CONSOLE_LOG)
    text = tail_text(path)
    if text is not None:
        kept = [line for line in text.splitlines()
                if re.search(r"\[Corpus\]|Traceback|error|failed|✗", line, re.I)][-120:]
        data["log"] = {"path": str(path), "lines": len(kept),
                       "excerpt": ctx.log("console-log.txt", "\n".join(kept)),
                       "last": kept[-5:]}
    else:
        data["log"] = {"path": str(path), "lines": None}

    if ctx.console_up is not True:
        answer = ctx.http("GET", f"{ctx.base}/api/status", None, 5)
        ctx.console_up = answer.status == 200
    if not ctx.console_up:
        return skipped(f"skipped: the console is not answering at {ctx.base}"
                       + ("; its log is excerpted in the logs" if text is not None else ""), data)

    healing, error = ctx.rpc("healing", {"since": ctx.journal_seq or 0})
    if error:
        return Section("error", f"the healing journal could not be read: {error}", data)
    events = (healing or {}).get("events") or []
    ctx.log_json("healing-delta.json", events)
    by_level = Counter(str(e.get("level")) for e in events)
    data["journal"] = {"since": ctx.journal_seq, "events": len(events), "by_level": dict(by_level),
                       "warnings": [e.get("message") for e in events
                                    if e.get("level") in ("WARNING", "ERROR", "CRITICAL")][-20:]}
    data["circuits_open"] = [c.get("name") for c in (healing or {}).get("circuits") or []
                             if c.get("state") != "closed"]
    failures = [e for e in events if e.get("level") in ("ERROR", "CRITICAL")]
    if failures:
        problems.append(finding("console-errors", f"the console journalled {len(failures)} "
                                "error(s) while this ran",
                                "\n".join(str(e.get("message")) for e in failures[-3:])))
    summary = (f"{len(events)} journal events since the first read ("
               + (", ".join(f"{n} {lvl.lower()}" for lvl, n in sorted(by_level.items())) or "none")
               + ")")
    return concluded(summary, data, problems)


def section_suite(ctx: Context) -> Section:
    ran = ctx.cmd([ctx.python, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], 3600,
                  cwd=str(ctx.root))
    summary = pytest_summary(ran.text)
    data = {"exit": ran.code, "log": ctx.log("pytest.txt", ran.text), "summary": summary}
    problems = [] if ran.ok else [finding("suite-failed", "the test suite does not pass",
                                          summary or first_line(ran))]
    return concluded(summary or f"exit {ran.code}", data, problems)


SECTIONS = ("machine", "tools", "sign-ins", "network", "gpu", "ollama", "embedder", "seats",
            "database", "searxng", "console-before", "browser", "console-after", "suite")
# Left out of --quick: each loads a model, and together they are most of the time.
HEAVY = frozenset({"embedder", "seats"})


def section_function(name: str) -> Callable[[Context], Section]:
    """Looked up when called, so a test can stand a section in for another."""
    function: Callable[[Context], Section] = globals()["section_" + name.replace("-", "_")]
    return function


def select_sections(args: argparse.Namespace) -> list[str]:
    """Which sections this call runs, in their order; ValueError naming an unknown one."""
    if args.sections:
        wanted = [name.strip() for name in args.sections.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in SECTIONS]
        if unknown:
            raise ValueError(f"no section {', '.join(unknown)}; the sections are "
                             + ", ".join(SECTIONS))
        return [name for name in SECTIONS if name in wanted]
    names = [name for name in SECTIONS if name != "suite" or args.with_tests]
    if args.quick:
        names = [name for name in names if name not in HEAVY]
    return names


def guarded(ctx: Context, name: str, function: Callable[[Context], Section]) -> Section:
    """Run one section; whatever it raises becomes an `error` section, never a crash."""
    ctx.current = name
    started = time.monotonic()
    try:
        section = function(ctx)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        where = ctx.log(f"{name}-error.txt", traceback.format_exc())
        section = Section("error", f"{type(exc).__name__}: {exc}"[:300], {"traceback": where})
    section.name = name
    section.duration_s = round(time.monotonic() - started, 1)
    return section


# --------------------------------------------------------------------------
# The report, the results and the bundle
# --------------------------------------------------------------------------

_MARKS = {"ok": "✓", "problem": "✗", "skipped": "-", "error": "!"}


def verdict_of(sections: list[Section], interrupted: bool = False) -> str:
    if interrupted or any(s.status == "error" for s in sections):
        return "incomplete"
    return "problems" if any(s.problems for s in sections) else "ok"


def exit_code(sections: list[Section]) -> int:
    if sections and all(s.status == "error" for s in sections):
        return 2
    return 1 if any(s.status in ("problem", "error") or s.problems for s in sections) else 0


def _peak(peaks: dict[str, Any], name: str) -> str:
    cards = peaks.get(name) or {}
    return ", ".join(f"{index}: {card['memory_used_mib']} MiB / {card['utilization_pct']}%"
                     for index, card in sorted(cards.items())) or "-"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_report(results: dict[str, Any]) -> str:
    """report.md from results.json's own content, so the two never disagree."""
    sections = results["sections"]
    peaks = (sections.get("gpu") or {}).get("data", {}).get("peaks") or {}
    problems = [(name, p) for name, s in sections.items() for p in s["problems"]]
    repo = results.get("repo") or {}
    lines = [
        "# machine diagnostic",
        "",
        f"- verdict: **{results['verdict']}** ({len(problems)} problem(s) in "
        f"{sum(1 for s in sections.values() if s['problems'])} section(s))",
        f"- started {results['started']}, took {results['duration_s']}s",
        f"- checkout: {repo.get('head') or '?'} on {repo.get('branch') or '?'}, "
        f"{repo.get('dirty') if repo.get('dirty') is not None else '?'} file(s) changed",
        f"- command: `{' '.join(results['argv'])}`",
        "",
        "| section | status | summary | seconds | peak vram / load |",
        "|---|---|---|---|---|",
    ]
    for name, section in sections.items():
        lines.append(f"| {name} | {_MARKS.get(section['status'], '')} {section['status']} | "
                     f"{_cell(section['summary'])} | {section['duration_s']} | "
                     f"{_cell(_peak(peaks, name))} |")
    lines.append("")
    if problems:
        lines += ["## problems and fixes", ""]
        for number, (name, problem) in enumerate(problems, 1):
            lines.append(f"{number}. **{name}**: {problem['what']}")
            if problem["evidence"]:
                evidence = problem["evidence"].replace("\n", " / ")
                lines.append(f"   - evidence: `{evidence.replace('`', chr(39))}`")
            if problem["fix"]:
                lines.append(f"   - fix: {problem['fix']}")
        lines.append("")
    redactions = results.get("redactions") or {}
    if redactions.get("count"):
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(redactions["kinds"].items()))
        lines += [f"redacted before writing: {kinds}.", ""]
    lines += ["## details", ""]
    for name, section in sections.items():
        lines += [f"<details><summary>{name}: {section['status']}, {_cell(section['summary'])}"
                  "</summary>", "", "```json",
                  json.dumps(section["data"], indent=2, default=str), "```", "", "</details>", ""]
    return "\n".join(lines)


def _never_shared(relative: Path) -> bool:
    """Files the share bundle leaves out whatever else asks for them."""
    name = relative.name.lower()
    return (name == ".env" or name.endswith(".env") or name.startswith(".env.")
            or name in ("trace.zip",) or relative.suffix.lower() in (".gif", ".zip", ".webm",
                                                                      ".mp4", ".har")
            or "downloads" in (part.lower() for part in relative.parts))


def bundle_members(out: Path) -> list[Path]:
    """What the share bundle carries, relative to `out`: the report, the results,
    the logs, the seat reports, and the browser reports with their screenshots.
    Never a trace, a recording, a download or an .env."""
    members: list[Path] = []
    for name in ("report.md", "results.json"):
        if (out / name).is_file():
            members.append(Path(name))
    patterns = ["logs/**/*", "seats/*.md", "seats/*.json", "browser*/report.md",
                "browser*/results.json", "browser*/shots/**/*"]
    for pattern in patterns:
        for path in sorted(out.glob(pattern)):
            if path.is_file():
                relative = path.relative_to(out)
                if not _never_shared(relative) and relative not in members:
                    members.append(relative)
    return members


def write_bundle(out: Path, members: list[Path]) -> Path:
    """<out>-share.tar.gz, with owners stripped: the account name is not the machine's state."""
    bundle = out.parent / f"{out.name}-share.tar.gz"

    def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(bundle, "w:gz") as tar:
        for relative in members:
            tar.add(out / relative, arcname=f"{out.name}/{relative}", filter=anonymous)
    return bundle


def _size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / 2**20:.1f} MB"


def share_as_gist(ctx: Context, members: list[Path], say: Callable[[str], None]) -> int:
    """The bundle's text files as a secret gist, after a typed yes; the exit status."""
    files: list[Path] = []
    names: set[str] = set()
    for relative in members:
        if relative.suffix in (".md", ".json", ".txt") and relative.name not in names:
            names.add(relative.name)
            files.append(relative)
    say("--gist would upload these files from the bundle:")
    for relative in files:
        say(f"    {relative}")
    say("A secret gist is unlisted, not private: anyone with its URL can read it.")
    if not ctx.tty:
        say("--gist needs a terminal to type yes in; nothing was uploaded.")
        return 1
    if ctx.ask("Type yes to create the gist: ").strip() != "yes":
        say("Nothing was uploaded.")
        return 1
    ran = ctx.cmd(["gh", "gist", "create", "--desc",
                   f"ambiguity machine diagnostic {ctx.out.name}",
                   *[str(ctx.out / f) for f in files]], 120)
    say(first_line(ran) if ran.ok else f"gh gist create failed: {first_line(ran)}")
    return 0 if ran.ok else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="diagnose_machine",
        description="Diagnose this machine for the console and write a report you can share.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "sections, in order: " + ", ".join(SECTIONS) + "\n"
            "--quick leaves out embedder and seats (and the gpu sampler); suite runs only with\n"
            "--with-tests. Read-only unless --with-runs. Exit status: 0 no problems, 1 problems\n"
            "found, 2 nothing could be diagnosed."
        ),
    )
    parser.add_argument("--base", default=os.environ.get("CONSOLE_URL") or DEFAULT_BASE,
                        help="the console to ask (default: CONSOLE_URL, else %(default)s)")
    parser.add_argument("--out", default="", help="report directory "
                        "(default: reports/diagnostics/<YYYYmmdd-HHMMSS>)")
    parser.add_argument("--quick", action="store_true",
                        help="no embedder, no seat probes, no gpu sampler; the browser's quick passes")
    parser.add_argument("--with-runs", action="store_true",
                        help="also send one discussion-only goal through the browser to the live "
                        "console. This is not read-only: real seats answer it, the corpus is "
                        "rebuilt first as before every run, and the console's last run "
                        "(runs/last_run.json) is overwritten")
    parser.add_argument("--with-tests", action="store_true",
                        help="also run the test suite (python -m pytest tests/)")
    parser.add_argument("--sections", default="", help="run only these, comma-separated")
    parser.add_argument("--no-browser", action="store_true", help="skip the browser agent")
    parser.add_argument("--save-log", action="append", default=[], metavar="FILE",
                        help="copy this log into the report's logs/, redacted (repeatable)")
    parser.add_argument("--gist", action="store_true",
                        help="upload the report as a secret gist after a typed yes")
    parser.add_argument("--json", action="store_true",
                        help="print results.json at the end; progress goes to stderr")
    return parser.parse_args(argv)


def diagnose(ctx: Context, names: list[str]) -> int:
    """Run `names` in order, write the report and the bundle; the exit status."""
    say_to = sys.stderr if ctx.args.json else sys.stdout

    def say(text: str) -> None:
        print(text, file=say_to, flush=True)

    started = time.time()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    say(f"diagnosing this machine into {ctx.out}")

    saved: list[str] = []
    for raw in ctx.args.save_log:
        source = Path(raw)
        text = read_text(source, limit=50_000_000)
        if text is None:
            say(f"  could not read {redact(raw)}; not saved")
            continue
        saved.append(ctx.log(f"saved-{source.name}", text))

    # The journal's high-water mark before anything here touches the console,
    # so console-after's delta covers the whole diagnostic.
    if ctx.journal_seq is None:
        healing, error = ctx.rpc("healing", timeout=10)
        if not error and isinstance(healing, dict):
            ctx.journal_seq = max((int(e.get("seq") or 0) for e in healing.get("events") or []),
                                  default=0)

    sections: list[Section] = []
    interrupted = False
    try:
        for name in names:
            if name in ("embedder", "seats", "browser", "network", "suite"):
                say(f"  … {name}")
            section = guarded(ctx, name, section_function(name))
            sections.append(section)
            say(f"  {_MARKS.get(section.status, '?')} {name:<15}{redact(section.summary)}")
    except KeyboardInterrupt:
        interrupted = True
        done = {s.name for s in sections}
        sections += [Section("skipped", "interrupted at the terminal", name=name)
                     for name in names if name not in done]
        say("  interrupted: writing what finished")
    finally:
        ctx.current = ""
        if ctx.sampler is not None:
            peaks = ctx.sampler.stop()
            for section in sections:
                if section.name == "gpu":
                    section.data["peaks"] = peaks

    repo = repo_state(ctx)
    results: dict[str, Any] = {
        "schema": SCHEMA,
        "started": started_at,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "argv": ["diagnose_machine.py", *ctx.argv],
        "repo": {"head": repo["head"], "branch": repo["branch"], "dirty": repo["dirty"]},
        "verdict": verdict_of(sections, interrupted),
        "sections": {s.name: {"status": s.status, "summary": s.summary, "data": s.data,
                              "problems": [asdict(p) for p in s.problems],
                              "duration_s": s.duration_s}
                     for s in sections},
        "saved_logs": saved,
    }
    # The files helpers wrote under the report directory are redacted before
    # anything is read back into the report, or bundled.
    _redact_tree(ctx, ctx.out)
    results = redact_data(results, ctx.counts)

    bundle = ctx.out.parent / f"{ctx.out.name}-share.tar.gz"

    def write(with_bytes: int | None) -> None:
        results["redactions"] = {"count": sum(ctx.counts.values()), "kinds": dict(ctx.counts)}
        results["bundle"] = {"path": redact(str(bundle)), "bytes": with_bytes,
                             "files": [str(m) for m in members]}
        (ctx.out / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                              encoding="utf-8")
        (ctx.out / "report.md").write_text(redact(render_report(results)), encoding="utf-8")

    members: list[Path] = []
    write(None)
    members = bundle_members(ctx.out)
    write(None)
    bundle = write_bundle(ctx.out, members)
    size = bundle.stat().st_size
    # The copy inside the bundle cannot know its own size; the one on disk does.
    write(size)

    # An interrupted run is incomplete, whatever the sections that finished say.
    code = max(exit_code(sections), 1) if interrupted else exit_code(sections)
    problems = sum(len(s.problems) for s in sections)
    errors = sum(1 for s in sections if s.status == "error")
    say("")
    say(f"verdict: {results['verdict']} ({problems} problem(s), {errors} section error(s))")
    say(f"report  {ctx.out / 'report.md'}")
    say(f"share   {bundle} ({_size(size)})")
    if ctx.args.json:
        print(json.dumps(results, indent=2, default=str))

    # A desktop without notifications is not a problem: the outcome is ignored.
    ctx.cmd(["notify-send", "--app-name=ambiguity", "machine diagnostic finished",
             f"{results['verdict']}: {problems} problem(s)"], 5)
    if ctx.args.gist and share_as_gist(ctx, members, say) != 0 and code == 0:
        code = 1
    return code


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    """The command line. `overrides` are `Context` fields: how the tests fake a machine."""
    args = parse_args(argv)
    try:
        names = select_sections(args)
    except ValueError as exc:
        print(f"diagnose_machine: {exc}", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "reports" / "diagnostics" / stamp
    try:
        (out / "logs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"diagnose_machine: cannot write {out}: {exc}", file=sys.stderr)
        return 2
    overrides.setdefault("argv", list(argv) if argv is not None else sys.argv[1:])
    ctx = Context(out=out.resolve(), args=args, **overrides)
    return diagnose(ctx, names)


if __name__ == "__main__":
    sys.exit(main())
