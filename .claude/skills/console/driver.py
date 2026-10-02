#!/usr/bin/env python3
"""Drive the Ambiguity 4-agent console without a human at the keyboard.

Stdlib only, on purpose: `up` and `doctor` have to be able to run and give a
useful answer on a machine where `pip install -e .` has not happened yet, so
this file cannot import anything the project depends on.

Everything here talks to the same surface the SPA talks to -- POST /rpc with
{method, params} -- so a check that passes here and a click in the browser
cannot quietly come to mean different things.

Usage (from the project root):

    .claude/skills/console/driver.py up
    .claude/skills/console/driver.py rpc rag_stats
    .claude/skills/console/driver.py rpc search_documents '{"query":"planner","top_k":3}'
    .claude/skills/console/driver.py shot /tmp/console.png
    .claude/skills/console/driver.py smoke
    .claude/skills/console/driver.py down
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PORT = int(os.getenv("PORT", "8080"))
BASE = f"http://localhost:{PORT}"
SERVER_LOG = Path(os.getenv("CONSOLE_LOG", "/tmp/ambiguity-console.log"))

# The venv this repo installs into. CI uses the runner's bare interpreter, but
# on a developer box the deps live here and `python serve.py` off PATH would
# start an interpreter that cannot import langgraph.
VENV_PY = ROOT / ".venv" / "bin" / "python"


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _env() -> dict[str, str]:
    # `src/` layout: serve.py imports langgraph_agent, which lives under src/.
    # An editable install puts it on the path too, but setting this keeps the
    # driver working against a plain checkout that was never pip-installed.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


# --------------------------------------------------------------------------
# RPC
# --------------------------------------------------------------------------


def rpc(method: str, params: dict[str, Any] | None = None, timeout: int = 120) -> dict:
    """One RPC call. Raises on transport failure, returns the envelope as-is.

    Note the server answers a failed *method* with HTTP 200 and an `error`
    member -- see handle_rpc in serve.py. Callers that care must look at the
    body; a 200 here does not mean the call worked.
    """
    body = json.dumps({"method": method, "params": params or {}}).encode()
    req = urllib.request.Request(
        f"{BASE}/rpc", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def is_up(timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/api/status", timeout=timeout).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def cmd_up(_args: list[str]) -> int:
    if is_up():
        print(f"already up at {BASE}")
        return 0
    if not VENV_PY.exists():
        print(f"!! no venv at {VENV_PY} -- see SKILL.md Build", file=sys.stderr)
    log = SERVER_LOG.open("wb")
    subprocess.Popen(
        [_python(), "serve.py"],
        cwd=ROOT,
        env=_env(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(90):
        if is_up():
            print(f"up at {BASE}  (log: {SERVER_LOG})")
            return 0
        time.sleep(1)
    print(f"!! server never answered {BASE}/api/status", file=sys.stderr)
    print(SERVER_LOG.read_text()[-2000:], file=sys.stderr)
    return 1


def cmd_down(_args: list[str]) -> int:
    """Stop via the app's own `shutdown` RPC.

    Deliberately not `pkill -f serve.py`: that pattern also matches the shell
    running the pkill, so it kills its own caller and returns 144. The server
    grew a shutdown method for the console's exit button; reuse it.
    """
    if not is_up():
        print("not running")
        return 0
    try:
        rpc("shutdown", timeout=10)
    except (urllib.error.URLError, OSError):
        pass  # it closed the socket on the way out; that is the success case
    for _ in range(20):
        if not is_up():
            print("stopped")
            return 0
        time.sleep(1)
    print("!! still answering after shutdown", file=sys.stderr)
    return 1


def cmd_restart(args: list[str]) -> int:
    return cmd_down(args) or cmd_up(args)


# --------------------------------------------------------------------------
# screenshot
# --------------------------------------------------------------------------


def cmd_shot(args: list[str]) -> int:
    """Headless screenshot of the console.

    Plain `chromium --headless --screenshot`. No Playwright, no xvfb, no
    browser extension -- the SPA renders identically headless, and this works
    over SSH on a box with no display.
    """
    out = Path(args[0] if args else "/tmp/ambiguity-console.png").resolve()
    browser = next(
        (b for b in ("chromium", "chromium-browser", "google-chrome") if shutil.which(b)),
        None,
    )
    if browser is None:
        print("!! no chromium on PATH", file=sys.stderr)
        return 1
    if not is_up():
        print("!! server is not up -- run `driver.py up` first", file=sys.stderr)
        return 1
    subprocess.run(
        [
            browser, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--window-size=1600,1000",
            # The graph is a force layout that settles over a second or two; a
            # smaller budget screenshots a tangle mid-simulation.
            "--virtual-time-budget=15000",
            f"--screenshot={out}", BASE,
        ],
        check=True,
        capture_output=True,
    )
    size = out.stat().st_size if out.exists() else 0
    print(f"{out}  ({size} bytes)")
    # An empty console is ~70KB; a drawn graph is ~900KB. Worth saying out loud,
    # because "screenshot written" and "screenshot shows anything" differ here.
    if size < 120_000:
        print("   (small -- probably the empty state; corpus indexed?)")
    return 0


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def cmd_seats(_args: list[str]) -> int:
    seats = rpc("list_seats")["result"]["seats"]
    dead = 0
    # Sized to the longest model rather than fixed: a pulled Hugging Face tag
    # runs past sixty characters and ran straight into the provider column.
    width = max((len(s["model"]) for s in seats), default=22) + 2
    for s in seats:
        mark = "live" if s["live"] else f"DEAD: {s['reason']}"
        print(f"  {s['role']:11}{s['model']:{width}}{s['provider']:10}{mark}")
        dead += not s["live"]
    if dead:
        print(f"\n{dead}/{len(seats)} seats dead -- run_goal will fail. See SKILL.md Gotchas.")
    return 0


def cmd_rpc(args: list[str]) -> int:
    if not args:
        print("usage: driver.py rpc <method> [json-params]", file=sys.stderr)
        return 2
    params = json.loads(args[1]) if len(args) > 1 else {}
    print(json.dumps(rpc(args[0], params), indent=2)[:4000])
    return 0


def cmd_doctor(_args: list[str]) -> int:
    print(f"root        {ROOT}")
    print(f"interpreter {_python()}")
    print(f"venv        {'present' if VENV_PY.exists() else 'MISSING -- see SKILL.md Build'}")
    try:
        subprocess.run(
            [_python(), "-c", "import langgraph_agent"],
            cwd=ROOT, env=_env(), check=True, capture_output=True,
        )
        print("import      langgraph_agent OK")
    except subprocess.CalledProcessError as exc:
        print(f"import      FAILED: {exc.stderr.decode().strip().splitlines()[-1]}")
    print(f"chromium    {shutil.which('chromium') or 'MISSING (shot unavailable)'}")
    print(f"server      {'up' if is_up() else 'down'}")
    return 0


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------


def cmd_smoke(_args: list[str]) -> int:
    """End-to-end check against a running server. Non-zero on any failure."""
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    if not is_up():
        print("!! server not up -- run `driver.py up` first", file=sys.stderr)
        return 1

    st = rpc("status")["result"]
    check("status responds", bool(st.get("embedding")), f"embedding={st.get('embedding')}")

    stats = rpc("rag_stats")["result"]
    corpus = stats.get("corpus")
    check("corpus state reported", corpus in ("absent", "empty", "indexed"), f"corpus={corpus}")
    if corpus == "indexed":
        check("graph has nodes", stats.get("total_nodes", 0) > 0,
              f"nodes={stats.get('total_nodes')} edges={stats.get('total_edges')}")
        stale = (stats.get("staleness") or {}).get("stale")
        check("corpus matches the archive", stale is False)
        docs = rpc("list_documents")["result"]["documents"]
        check("documents listed", len(docs) > 0, f"n={len(docs)}")
        hits = rpc("search_documents", {"query": "planner agent", "top_k": 3})["result"]["results"]
        check("semantic search returns hits", len(hits) > 0,
              f"top={hits[0]['id']}" if hits else "")
    else:
        # Not a failure: the corpus is the research archive, which a fresh
        # machine has none of until something is researched or uploaded.
        print("  note  the archive is empty -- upload a document or research online")

    seats = rpc("list_seats")["result"]["seats"]
    check("four seats configured", len(seats) == 4, f"n={len(seats)}")
    live = sum(s["live"] for s in seats)
    # Not a failure: the shipped seats are local models a fresh box may not
    # have pulled yet (install.sh pulls them), and every read-only tab works
    # without them.
    print(f"  note  {live}/{len(seats)} seats live"
          f"{'' if live else ' -- run_goal will fail until a tag is pulled'}")

    healing = rpc("healing")["result"]
    check("self-healing reports", isinstance(healing.get("circuits"), list),
          f"circuits={len(healing.get('circuits', []))}")
    for circuit in healing.get("circuits", []):
        if circuit["state"] != "closed":
            print(f"  note  circuit {circuit['name']} is {circuit['state']}")
    for name, result in (healing.get("health") or {}).items():
        if result.get("status") != "healthy":
            print(f"  note  {name}: {result.get('status')} -- {result.get('details')}")

    bad = rpc("no_such_method")
    check("unknown method returns error envelope, not 500", "error" in bad)

    print(f"\n{'SMOKE FAILED: ' + ', '.join(failures) if failures else 'SMOKE OK'}")
    return 1 if failures else 0


COMMANDS = {
    "up": cmd_up, "down": cmd_down, "restart": cmd_restart,
    "shot": cmd_shot, "seats": cmd_seats,
    "rpc": cmd_rpc, "smoke": cmd_smoke, "doctor": cmd_doctor,
}


def main() -> int:
    # Line-buffered, so progress lines appear as they happen under a pipe.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands: " + "  ".join(COMMANDS))
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
