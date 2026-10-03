#!/bin/bash
# Launch the 4-Agent Console with frontend and auto-open browser

set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  Ambiguity 4-Agent Console"
echo "============================================"
echo ""

# Started from the app launcher there is no terminal to read, so a launch that
# cannot go on also says so on the desktop when it can.
fail() {
    echo "✗ $1" >&2
    if command -v notify-send >/dev/null 2>&1; then
        notify-send -u critical "Ambiguity Console" "$1" 2>/dev/null || true
    fi
    exit 1
}

# The project's packages live in .venv. install.sh builds it, but a checkout
# that never ran it -- or whose pyproject.toml has changed since -- is set up
# here, before anything imports the project: `python` on PATH is otherwise
# whatever comes first (on Omarchy, mise's interpreter, with none of them), and
# serve.py died on its first import. Activated, the venv is also first on PATH
# for the Builder's own terminal commands.
VENV=.venv
INSTALL_LOG=/tmp/ambiguity-install.log
# What the venv was last installed from; install.sh writes the same stamp, so
# the first launch after it installs nothing.
DEPS_STAMP="$VENV/.ambiguity-deps"
PY_FLOOR="$(sed -n 's/^requires-python *= *">=\([0-9][0-9.]*\)".*/\1/p' pyproject.toml)"
PY_FLOOR="${PY_FLOOR:-3.12}"
floor_ok() {
    "$1" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '$PY_FLOOR'.split('.'))))" \
        2>/dev/null
}
deps_stamp() { sha256sum pyproject.toml | cut -d' ' -f1; }
# Every runtime dependency pyproject.toml declares, plus the Builder's own
# tools (ruff, pytest), read from the declaration rather than a list kept here.
deps_present() {
    "$VENV/bin/python" - 2>/dev/null <<'PY'
import re
import sys
import tomllib
from importlib import metadata

project = tomllib.load(open("pyproject.toml", "rb"))["project"]
wanted = ["langgraph-agent", *project["dependencies"],
          *project.get("optional-dependencies", {}).get("tools", [])]
for requirement in wanted:
    try:
        metadata.version(re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0])
    except metadata.PackageNotFoundError:
        sys.exit(1)
PY
}

# A venv whose interpreter was removed underneath it -- a mise upgrade, an Arch
# Python minor bump -- still has a bin/python symlink, pointing at nothing.
if ! { [ -x "$VENV/bin/python" ] && floor_ok "$VENV/bin/python"; }; then
    if [ -e "$VENV" ]; then
        echo "  .venv is broken or older than Python $PY_FLOOR; rebuilding it"
        rm -rf "$VENV"
    fi
    # The first interpreter on PATH that is new enough and can make a venv: an
    # interpreter without ensurepip (some distribution and tool-managed builds)
    # is passed over rather than ending the launch.
    BASE_PY=""
    for candidate in python3 python python3.14 python3.13 python3.12 /usr/bin/python3; do
        candidate="$(command -v "$candidate" 2>/dev/null)" || continue
        floor_ok "$candidate" || continue
        echo "  first launch: creating .venv with $("$candidate" --version 2>&1) ($candidate)"
        if "$candidate" -m venv "$VENV" >>"$INSTALL_LOG" 2>&1; then
            BASE_PY="$candidate"
            break
        fi
        echo "  $candidate could not create a venv; trying the next interpreter"
        rm -rf "$VENV"
    done
    [ -n "$BASE_PY" ] || fail "No Python >= $PY_FLOOR on PATH could build .venv (sudo pacman -S python, or ./install.sh); see $INSTALL_LOG"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip >>"$INSTALL_LOG" 2>&1 || true
fi
if [ "$(cat "$DEPS_STAMP" 2>/dev/null || true)" != "$(deps_stamp)" ] || ! deps_present; then
    echo "  installing the project's dependencies into .venv (pip install -e \".[tools]\");"
    echo "  a first install takes a few minutes -- the log is $INSTALL_LOG"
    if "$VENV/bin/python" -m pip install --quiet -e ".[tools]" >>"$INSTALL_LOG" 2>&1 \
            && deps_present; then
        deps_stamp >"$DEPS_STAMP"
        echo "✓ Dependencies installed"
    else
        tail -n 15 "$INSTALL_LOG" >&2 || true
        fail "could not install the project's dependencies into .venv; see $INSTALL_LOG"
    fi
fi
# shellcheck source=/dev/null
. "$VENV/bin/activate"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    # Owner-only, as install.sh makes it: an API key goes here if a seat is
    # moved to a paid provider, and a copy inherits the template's mode.
    if cp .env.example .env 2>/dev/null; then chmod 600 .env; fi
fi

# Load environment safely
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    . ./.env
    set +a
    echo "✓ Loaded .env"
fi

# After .env, so a PORT set there -- or in the shell -- moves the address this
# script polls and opens along with the one serve.py binds. It was fixed at
# 8080 before .env was read: serve.py took the new port while this kept
# waiting on 8080, and opened whatever else was answering there.
PORT="${PORT:-8080}"
export PORT
URL="http://localhost:${PORT}"

# Report the seats from config.py rather than re-deriving them from env vars.
# Guessing here is how this line ended up claiming a Claude model for seats
# that were running something else -- or nothing at all.
if [ -d "src" ]; then
    PYTHONPATH=src python3 - <<'PY' 2>/dev/null || true
import logging

# The healing journal's lines -- a daemon that is down opens its circuit
# while the seats are read -- belong to the console's feed, not this summary,
# which says the same thing once per seat.
logging.disable(logging.CRITICAL)

from langgraph_agent.config import AGENTS, get_agent_status  # noqa: E402

seats = {agent: get_agent_status(agent) for agent in AGENTS}
# Sized to the longest tag, as install.sh's summary is: a local tag runs to
# 55 characters, and a fixed column ran it into the provider.
width = max(len(seat["model"]) for seat in seats.values()) + 2
for agent, seat in seats.items():
    flag = "" if seat["live"] else f"  !! {seat['reason']}"
    print(f"  {agent:11}{seat['model']:<{width}}{seat['provider']}{flag}")
PY
fi

# The console's own answer, not merely an answer: any server on this port
# replies to the request, and one that was not the console was taken for it,
# opened in the browser, and serve.py never started.
is_server_ready() {
    if command -v curl &> /dev/null; then
        curl -sf --max-time 2 "${URL}/api/status" 2>/dev/null | grep -q '"indexes_on_run"'
    else
        python3 -c "import sys, urllib.request; body = urllib.request.urlopen('${URL}/api/status', timeout=1).read(); sys.exit(0 if b'\"indexes_on_run\"' in body else 1)" > /dev/null 2>&1
    fi
}

open_browser() {
    local url="$1"
    if command -v xdg-open &> /dev/null; then
        xdg-open "$url" &
    elif command -v open &> /dev/null; then
        open "$url" &
    elif command -v python3 &> /dev/null; then
        python3 -c "import webbrowser; webbrowser.open('$url')" &
    elif command -v python &> /dev/null; then
        python -c "import webbrowser; webbrowser.open('$url')" &
    else
        echo "Please open your browser manually: $url"
        return 1
    fi
}

# Already up (the app launcher clicked twice): open it rather than fail on a
# port that is in use.
if is_server_ready; then
    echo "✓ Console already running at ${URL}"
    open_browser "${URL}" || true
    exit 0
fi

echo ""
echo "Starting frontend server on ${URL}..."

# The values serve.py reads as off.
case "${REBUILD_CORPUS:-1}" in
    0|false|no)
        echo "  (REBUILD_CORPUS is off: nothing rebuilds the corpus, at"
        echo "   startup or before a run.)"
        ;;
    *)
        echo "  (The server brings the corpus up to date with the archive --"
        echo "   research/web/, uploads/ and opted-in projects/ -- once it is up;"
        echo "   the header shows the rebuild, and every run checks it again"
        echo "   before the Architect opens.)"
        ;;
esac

# The corpus lives in PostgreSQL. Not fatal -- the console starts without it
# and reports the corpus unavailable -- but said here, where someone is
# looking, rather than only in the header.
DB_URL="${DATABASE_URL:-postgresql://postgres@127.0.0.1:5432/postgres}"
# Shown without its password, as the console shows it (`redacted_url`).
DB_SHOWN="$(printf '%s' "${DB_URL%%\?*}" | sed -E 's#(://[^:/@]+:)[^@]*@#\1***@#')"
if command -v psql >/dev/null 2>&1; then
    if PGCONNECT_TIMEOUT=3 psql "$DB_URL" -w -X -q -t -A -c 'select 1' >/dev/null 2>&1; then
        echo "✓ Corpus database answers"
    else
        echo "⚠️  The corpus database does not answer at ${DB_SHOWN}:"
        echo "   docker start postgres18 (or ./install.sh); the console starts anyway"
        echo "   and reports the corpus as unavailable until it does."
    fi
fi

python serve.py > /tmp/ambiguity-console.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (generous timeout for slower CPUs)
echo -n "  Waiting for server"
for _ in $(seq 1 60); do
    if is_server_ready; then
        echo ""
        echo "✓ Server ready"
        break
    fi
    echo -n "."
    sleep 0.5
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "✗ Server failed to start. Log:"
    cat /tmp/ambiguity-console.log
    exit 1
fi

echo ""
echo "Opening browser..."
open_browser "${URL}" || true

echo ""
echo "Press Ctrl+C to stop, or use the console's exit button (top right)"
echo "============================================"
echo ""

# Tail server log so output is visible, then stop with the server
trap 'kill "$SERVER_PID" 2>/dev/null || true; exit 0' INT TERM
wait "$SERVER_PID"
