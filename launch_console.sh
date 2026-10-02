#!/bin/bash
# Launch the 4-Agent Console with frontend and auto-open browser

set -e

cd "$(dirname "$0")"

# The project's packages live in .venv (install.sh builds it). Without this,
# `python` is whatever is first on PATH -- on Omarchy that is mise's
# interpreter, which has none of them, so serve.py died on its first import.
# It also puts the venv first on PATH for the Builder's own terminal commands.
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    . .venv/bin/activate
fi

echo "============================================"
echo "  Ambiguity 4-Agent Console"
echo "============================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env 2>/dev/null || true
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
from langgraph_agent.config import AGENTS, get_agent_status

for agent in AGENTS:
    seat = get_agent_status(agent)
    flag = "" if seat["live"] else f"  !! {seat['reason']}"
    print(f"  {agent:11}{seat['model']:22}{seat['provider']}{flag}")
PY
fi

is_server_ready() {
    if command -v curl &> /dev/null; then
        curl -s "${URL}/api/status" > /dev/null 2>&1
    else
        python3 -c "import urllib.request; urllib.request.urlopen('${URL}/api/status', timeout=1)" > /dev/null 2>&1
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
if command -v psql >/dev/null 2>&1; then
    if PGCONNECT_TIMEOUT=3 psql "$DB_URL" -w -X -q -t -A -c 'select 1' >/dev/null 2>&1; then
        echo "✓ Corpus database answers"
    else
        echo "⚠️  The corpus database does not answer at ${DB_URL%%\?*}:"
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
