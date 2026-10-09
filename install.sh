#!/usr/bin/env bash
# One command from a fresh Arch / Omarchy machine to a console that runs.
#
# Safe to re-run. Every step looks before it acts, so a second run on a
# finished machine installs nothing, asks for no password, and ends by telling
# you whether each seat can actually run -- which is the only definition of
# "installed" that matters here.
#
# Everything it installs is free to use. System packages come from the Arch
# repos, Python packages from PyPI, the embedding model from the local Ollama
# daemon (its tokenizer from Hugging Face -- the chunker cuts passages with it
# in-process), the SearxNG and pgvector PostgreSQL images from Docker Hub, and inference
# from models that daemon runs from its own weights. No default seat needs an
# account: an Ollama Cloud tag -- which a free ollama.com account can run -- is
# pulled as an extra seat choice only when the daemon is signed in, and a
# sign-in is asked for only when a seat is set up to use one. Nothing here asks
# for an API key: Anthropic stays optional, and unconfigured.
#
# The driver is the machine's own setup, not this script's -- Omarchy installs
# it. Nothing in this project touches torch or a card itself: the daemon owns
# the embedding model's placement, and OLLAMA_EMBED_OPTIONS in
# graphrag_server.py loads it with every layer on the cards. The one GPU
# decision made here is which Ollama build that daemon runs: a card too old for
# Arch's CUDA 13 build gets Ollama's own CUDA 12 build instead, through
# cuda-embed-ollama.sh (step 4). What this script then does is one real encode
# through the class a run uses, and says what the daemon reported back.
#
# Usage:
#   ./install.sh                everything below
#   ./install.sh --minimal      skip the optional developer and container tools
#   ./install.sh --no-system    skip pacman entirely (no sudo); Python side only
#   ./install.sh --no-cuda12    keep Arch's Ollama build on NVIDIA cards its
#                               CUDA 13 cannot drive (they embed on the CPU)
#   ./install.sh --no-searxng   do not run a SearxNG for online research
#   ./install.sh --no-postgres  do not run a PostgreSQL in Docker; the corpus
#                               lives in PostgreSQL, so DATABASE_URL in .env
#                               must then name a server with pgvector
#   ./install.sh --no-docker-group
#                               keep Docker behind sudo, Omarchy's default,
#                               rather than adding you to the docker group
#   ./install.sh --no-checks    skip ruff / mypy / pytest
#   ./install.sh --no-probe     do not send each seat a one-word test prompt
#   ./install.sh --no-desktop   do not add the console to the app launcher
#   ./install.sh --yes          never prompt (pacman --noconfirm, no sign-ins)
#
#   SEARXNG_PORT=8899 ./install.sh
#                               another loopback port for SearxNG when 8888 is
#                               taken (it is Jupyter's default); .env works too

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

MINIMAL=0 SYSTEM=1 CUDA12=1 SEARXNG=1 POSTGRES=1 DOCKER_GROUP=1 CHECKS=1 PROBE=1 DESKTOP=1 ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --minimal)    MINIMAL=1 ;;
        --no-system)  SYSTEM=0 ;;
        --no-cuda12)  CUDA12=0 ;;
        --no-searxng) SEARXNG=0 ;;
        --no-postgres) POSTGRES=0 ;;
        --no-docker-group) DOCKER_GROUP=0 ;;
        --no-checks)  CHECKS=0 ;;
        --no-probe)   PROBE=0 ;;
        --no-desktop) DESKTOP=0 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        -h|--help)    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# Problems are collected rather than fatal from the Ollama step on: a machine
# that is not signed in yet still gets its venv and checks, and the summary
# says exactly what is left instead of the script dying halfway.
PROBLEMS=()
problem() { PROBLEMS+=("$1"); echo "  ✗ $1"; }
# Notes are not problems: everything works, but slower than it could, and the
# summary should say so rather than leave it in the scrollback.
NOTES=()
step()    { echo; echo "► $1"; }
ok()      { echo "  ✓ $1"; }
interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; }

echo "============================================"
echo "  Ambiguity 4-Agent Console -- install"
echo "============================================"

if [ "$(id -u)" -eq 0 ]; then
    # The venv, the corpus and the launcher all belong to the user who runs
    # the console. Built as root, every one of them is owned by root and the
    # first ordinary run fails on a permission error far from its cause.
    echo "Run this as your own user, not root; it calls sudo where it must." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 0. Network
# ---------------------------------------------------------------------------

# Before anything downloads. Behind an egress allowlist each blocked host
# otherwise fails its own step, worded by whichever tool met it; asked up
# front, every host this install needs is named at once, with what needs it.
step "Network (the hosts this install downloads from)"
net_groups=(pypi ollama hf tokenizer github)
[ "$SYSTEM" -eq 1 ] && net_groups+=(arch)
{ [ "$POSTGRES" -eq 1 ] || [ "$SEARXNG" -eq 1 ]; } && net_groups+=(dockerhub)
NETWORK_OK=1
if ! scripts/network_check.sh "${net_groups[@]}" --optional research cloud; then
    NETWORK_OK=0
    problem "the network did not let through every host this install needs; allow the entries above"
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

# Required: the console cannot start, or cannot do its job, without these.
REQUIRED=(
    python       # the interpreter, if nothing on PATH is new enough
    git          # the Builder's git_ tools
    github-cli   # gh: git_dwell's pr and merge stages end its default pipeline
    curl         # launch_console.sh's readiness check, and the daemon probe
    ollama       # the daemon every default seat runs through
    base-devel   # a compiler, for any wheel with no binary for this Python
    xdg-utils    # xdg-open: how the console opens in your browser
)

# Online research: SearxNG runs in a rootless podman container, so it needs no
# daemon, no group membership and no sudo after this step. crun is named
# because podman accepts either OCI runtime and pacman would otherwise stop to
# ask which.
SEARXNG_PACKAGES=(podman crun)

# PostgreSQL runs in Docker, as Omarchy's own development databases do, and
# holds the corpus: chunks with their pgvector embeddings, the entity graph and
# the relevance floor. postgresql-libs is the client on the host -- psql --
# which is how the step below proves the URL it writes into .env really logs in.
POSTGRES_PACKAGES=(docker postgresql-libs)

# The console's own image (README.md > Running in a container), optional like
# USEFUL and apart from the database, so --no-postgres does not take it away:
# `docker compose up --build` needs the engine and docker-compose, and the
# dockerfile's cache mounts and heredocs need BuildKit, which docker-buildx
# provides (Omarchy ships both plugins; plain Arch does not, and neither pulls
# in the engine). Podman, with crun for the reason above, builds and runs the
# same image rootless while the docker group has not applied yet.
# docker.service and the docker group are still set up by the PostgreSQL step.
CONTAINER_PACKAGES=(docker docker-compose docker-buildx podman crun)

# Useful: nothing breaks without them, but working on this repo is worse.
USEFUL=(
    ripgrep      # rg: searching a codebase this size
    fd           # finding files by name
    jq           # reading /rpc replies: curl ... | jq
    shellcheck   # linting this script and launch_console.sh
    btop         # watching the embedder and the test suite use the machine
)

if [ "$SYSTEM" -eq 1 ]; then
    step "System packages (pacman)"
    if ! command -v pacman >/dev/null; then
        echo "  pacman not found: this installer targets Arch / Omarchy." >&2
        echo "  Install the equivalents of: ${REQUIRED[*]}" >&2
        echo "  then re-run with --no-system." >&2
        exit 1
    fi

    wanted=("${REQUIRED[@]}")
    [ "$SEARXNG" -eq 1 ] && wanted+=("${SEARXNG_PACKAGES[@]}")
    [ "$POSTGRES" -eq 1 ] && wanted+=("${POSTGRES_PACKAGES[@]}")
    [ "$MINIMAL" -eq 0 ] && wanted+=("${USEFUL[@]}" "${CONTAINER_PACKAGES[@]}")

    # The console needs a browser, and Omarchy ships Chromium. Only a machine
    # with none at all gets one, so an existing choice is never second-guessed.
    browser_found=0
    for b in chromium firefox brave google-chrome-stable librewolf zen-browser vivaldi; do
        command -v "$b" >/dev/null && { browser_found=1; break; }
    done
    [ "$browser_found" -eq 0 ] && wanted+=(chromium)

    # docker, podman and crun each sit in two groups; each is named once, so
    # the counts below are packages rather than mentions.
    mapfile -t wanted < <(printf '%s\n' "${wanted[@]}" | awk '!seen[$0]++')

    # `pacman -T` prints what is not satisfied, provides included, so a rerun
    # on a finished machine reaches no sudo prompt at all.
    mapfile -t missing < <(pacman -T "${wanted[@]}" || true)
    if [ "${#missing[@]}" -eq 0 ]; then
        ok "all ${#wanted[@]} packages already installed"
    else
        echo "  installing: ${missing[*]}"
        pacman_flags=(-S --needed)
        [ "$ASSUME_YES" -eq 1 ] && pacman_flags+=(--noconfirm)
        # No -y: `pacman -Sy <pkg>` is a partial upgrade, which Arch does not
        # support. If the mirrors have moved on, upgrade the system properly
        # (omarchy-update, or sudo pacman -Syu) and re-run.
        if ! sudo pacman "${pacman_flags[@]}" "${missing[@]}"; then
            if [ "$NETWORK_OK" -eq 0 ]; then
                echo "  pacman failed, and the Network step above found hosts that" >&2
                echo "  did not answer: start there, then re-run ./install.sh." >&2
                exit 1
            fi
            echo "  pacman failed. If it could not find a package or file, the" >&2
            echo "  package database is stale: run omarchy-update (or" >&2
            echo "  sudo pacman -Syu) and then re-run ./install.sh." >&2
            exit 1
        fi
        ok "installed ${#missing[@]} package(s)"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

# Before the Python environment, so .env can say which Python anything needs.
step "Configuration (.env)"
if [ -f .env ]; then
    ok ".env exists; left as it is"
else
    cp .env.example .env
    # Owner-only: an API key goes here if a seat is moved to a paid provider,
    # and a copy inherits the template's world-readable mode.
    chmod 600 .env
    ok "created .env from .env.example (no key needed for the default seats)"
fi
set -a
# shellcheck source=/dev/null
. ./.env
set +a

# The loopback port for the SearxNG this script runs. Read after .env, so either
# the environment or .env can move it off 8888, which is also Jupyter's default.
SEARXNG_PORT="${SEARXNG_PORT:-8888}"
case "$SEARXNG_PORT" in
    ''|*[!0-9]*) echo "SEARXNG_PORT must be a port number, not '$SEARXNG_PORT'" >&2; exit 2 ;;
esac
SEARXNG_LOCAL="http://127.0.0.1:$SEARXNG_PORT"

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------

step "Python environment (.venv)"

# The floor is read from pyproject.toml rather than restated here, so the day
# it moves this script moves with it.
FLOOR="$(sed -n 's/^requires-python *= *">=\([0-9][0-9.]*\)".*/\1/p' pyproject.toml)"
FLOOR="${FLOOR:-3.12}"
floor_ok() {
    "$1" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '$FLOOR'.split('.'))))" \
        2>/dev/null
}

# A venv whose interpreter was removed underneath it -- a mise upgrade, an
# Arch Python minor bump -- still has a bin/python symlink, pointing at
# nothing. Test it by running it, and rebuild rather than patch.
if [ -x "$VENV/bin/python" ] && floor_ok "$VENV/bin/python"; then
    ok "reusing .venv ($("$VENV/bin/python" --version))"
else
    [ -e "$VENV" ] && { echo "  .venv is broken or too old; rebuilding it"; rm -rf "$VENV"; }
    # The first interpreter on PATH wins, so an Omarchy machine gets mise's
    # Python if that is what you chose; /usr/bin/python3 is the fallback. One
    # that is new enough but cannot make a venv -- no ensurepip, as some
    # distribution and tool-managed builds ship -- is passed over, the way
    # launch_console.sh passes over it.
    BASE_PY=""
    for candidate in python3 python python3.14 python3.13 python3.12 /usr/bin/python3; do
        candidate="$(command -v "$candidate" 2>/dev/null)" || continue
        floor_ok "$candidate" || continue
        if "$candidate" -m venv "$VENV" >/tmp/ambiguity-venv.log 2>&1; then
            BASE_PY="$candidate"
            break
        fi
        echo "  $candidate could not create a venv (see /tmp/ambiguity-venv.log); trying the next interpreter"
        rm -rf "$VENV"
    done
    if [ -z "$BASE_PY" ]; then
        echo "  No Python >= $FLOOR found that can make a venv. Install it" >&2
        echo "  (sudo pacman -S python) or drop --no-system, and re-run." >&2
        exit 1
    fi
    ok "created .venv with $("$VENV/bin/python" --version) from $BASE_PY"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip

echo "  installing the project and its dev tools (pip install -e \".[dev]\")"
"$PY" -m pip install --quiet -e ".[dev]"
ok "langgraph-agent installed in editable mode"
# What the venv was installed from. launch_console.sh reinstalls when
# pyproject.toml no longer matches it, so a launch right after this one installs
# nothing, and a pull that changes the dependencies is picked up at the next.
sha256sum pyproject.toml | cut -d' ' -f1 >"$VENV/.ambiguity-deps"

# pip installs what pyproject.toml asks for and never removes what it stopped
# asking for, so a venv older than these changes still carries the packages
# behind them: pybreaker (the circuit breakers are the project's own now),
# chromadb (the corpus moved to PostgreSQL), and the OpenAI stack (the provider
# was removed). Each goes only if nothing left in the venv requires it.
mapfile -t dropped < <("$PY" - <<'PY'
import re
from importlib import metadata

DROPPED = {"pybreaker", "chromadb", "langchain-openai", "openai", "tiktoken"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed = {canonical(d.metadata["Name"]): d for d in metadata.distributions() if d.metadata["Name"]}
candidates = {name for name in DROPPED if name in installed}
needed = set()
for name, dist in installed.items():
    if name in candidates:
        continue
    for spec in dist.requires or []:
        if "extra ==" in spec:
            continue
        needed.add(canonical(re.split(r"[\s<>=!~;\[(]", spec, maxsplit=1)[0]))
for name in sorted(candidates - needed):
    print(name)
PY
)
if [ "${#dropped[@]}" -gt 0 ]; then
    if "$PY" -m pip uninstall --quiet -y "${dropped[@]}"; then
        ok "removed what the project no longer uses: ${dropped[*]}"
    else
        NOTES+=("could not remove packages the project no longer uses (${dropped[*]}); harmless, but: $PY -m pip uninstall ${dropped[*]}")
    fi
fi

# ---------------------------------------------------------------------------
# 4. Ollama: daemon, sign-in, seat models
# ---------------------------------------------------------------------------

step "Ollama"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_HOST="$OLLAMA_URL"

daemon_up() { curl -s -m 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

if ! command -v ollama >/dev/null; then
    problem "ollama is not installed (drop --no-system, or: sudo pacman -S ollama)"
else
    if ! daemon_up && [ -z "${OLLAMA_BASE_URL:-}" ] && systemctl cat ollama.service >/dev/null 2>&1; then
        if [ "$SYSTEM" -eq 1 ]; then
            echo "  starting ollama.service (and enabling it at boot)"
            sudo systemctl enable --now ollama.service || true
            for _ in $(seq 1 20); do daemon_up && break; sleep 0.5; done
        else
            # --no-system promises no sudo, so the command is named instead.
            echo "  ollama.service is not running, and --no-system does not start it: sudo systemctl enable --now ollama.service"
        fi
    fi

    # One model resident, one request at a time, enforced by the daemon itself.
    # GPU_ARBITER serializes this console's own seats and embedder, but only
    # inside one process; a second console, a script or diagnose_seats.py
    # reaches the daemon around it. With these the daemon queues the next
    # request until the current one finishes, and unloads before it loads.
    one_model=/etc/systemd/system/ollama.service.d/one-model.conf
    one_model_body=$'[Service]\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\nEnvironment="OLLAMA_NUM_PARALLEL=1"\n'
    if [ -z "${OLLAMA_BASE_URL:-}" ] && systemctl cat ollama.service >/dev/null 2>&1; then
        if [ "$(cat "$one_model" 2>/dev/null)"$'\n' = "$one_model_body" ]; then
            ok "the daemon holds one model at a time ($one_model)"
        elif [ "$SYSTEM" -eq 1 ]; then
            echo "  limiting the daemon to one model at a time ($one_model)"
            if sudo mkdir -p "${one_model%/*}" \
                && printf '%s' "$one_model_body" | sudo tee "$one_model" >/dev/null \
                && sudo systemctl daemon-reload \
                && sudo systemctl restart ollama.service; then
                for _ in $(seq 1 20); do daemon_up && break; sleep 0.5; done
                ok "the daemon holds one model at a time"
            else
                problem "could not limit the daemon to one model; see $one_model"
            fi
        else
            echo "  --no-system does not configure the daemon; for one model at a time write $one_model with OLLAMA_MAX_LOADED_MODELS=1 and OLLAMA_NUM_PARALLEL=1"
        fi
    fi

    # Every seat and the embedder are this one daemon, so it is kept running:
    # enabled at boot and restarted whenever it exits (scripts/ollama_keepalive.sh).
    if [ -z "${OLLAMA_BASE_URL:-}" ] && systemctl cat ollama.service >/dev/null 2>&1; then
        if scripts/ollama_keepalive.sh check >/dev/null 2>&1; then
            ok "the daemon starts at boot and is restarted whenever it exits"
        elif [ "$SYSTEM" -eq 1 ]; then
            echo "  starting the daemon at boot and restarting it whenever it exits"
            scripts/ollama_keepalive.sh install \
                || problem "the daemon is not kept running; scripts/ollama_keepalive.sh check says what is missing"
        else
            echo "  --no-system does not configure the daemon; to keep it running: scripts/ollama_keepalive.sh install"
        fi
    fi

    if ! daemon_up; then
        problem "Ollama daemon unreachable at $OLLAMA_URL"
    else
        ok "daemon answering at $OLLAMA_URL ($(ollama --version 2>/dev/null | awk '{print $NF}'))"

        # What there is to pull, and what each tag is for -- read from
        # config.py and graphrag_server plus any .env override, rather than a
        # list kept here. A `seat` tag is what a seat runs now: missing, it
        # shows NOT PULLED and fails its run. The `embed` tag is the one
        # embedding model: missing, every run's corpus phase fails. An
        # `option` is a tag AGENT_LLM_OPTIONS suggests moving a seat onto,
        # pulled up front so a live switch never waits on a download; the
        # console's dropdowns -- and `set_seat` -- offer whatever `ollama ls`
        # reports, so pulling one here is also what puts it in them. A seat or
        # the embedder that cannot be pulled is a problem; an option is a note.
        #
        # Captured rather than streamed into mapfile: a process substitution
        # hides its exit status, so a broken import would pull nothing and
        # say so nowhere.
        wanted_models=()
        if model_list="$("$PY" - <<'PY'
from langgraph_agent.config import AGENTS, AGENT_LLM_OPTIONS, get_agent_model_info
from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME

roles: dict[str, str] = {}
for agent in AGENTS:
    info = get_agent_model_info(agent)
    if info["provider"] == "ollama":
        roles.setdefault(info["model"], "seat")
roles.setdefault(EMBEDDING_MODEL_NAME, "embed")
for option in AGENT_LLM_OPTIONS:
    if option["provider"] == "ollama":
        roles.setdefault(option["model"], "option")
print("\n".join(f"{role}\t{model}" for model, role in roles.items()))
PY
        )"; then
            mapfile -t wanted_models <<<"$model_list"
        else
            problem "could not read the seat and embedding models from config.py, so nothing was pulled"
        fi

        is_cloud_tag() { case "$1" in *:cloud|*-cloud) return 0 ;; *) return 1 ;; esac; }
        cloud_seats=()
        for entry in "${wanted_models[@]}"; do
            if [ "${entry%%$'\t'*}" = seat ] && is_cloud_tag "${entry#*$'\t'}"; then
                cloud_seats+=("${entry#*$'\t'}")
            fi
        done

        # ollama.com is needed only for `:cloud` tags. Every default seat runs
        # on weights this daemon holds, so a machine with no account is a
        # complete install, and is asked to sign in only when a seat it is set
        # up to use is a cloud tag. /api/me answers 200 only when the daemon is
        # signed in; one too old to have the route answers 404, and then the
        # pulls below are the test instead.
        me_status() { curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "$OLLAMA_URL/api/me"; }
        signed="$(me_status || true)"
        if [ "$signed" != "200" ] && [ "$signed" != "404" ] && [ "${#cloud_seats[@]}" -gt 0 ] && interactive; then
            echo "  ${cloud_seats[*]} runs on ollama.com, and the daemon is not signed in (a free account is enough)"
            ollama signin || true
            signed="$(me_status || true)"
        fi
        cloud_ready=1
        case "$signed" in
            200) ok "signed in to ollama.com" ;;
            404) echo "  (this daemon cannot report its sign-in; the pulls will tell)" ;;
            *)
                cloud_ready=0
                if [ "${#cloud_seats[@]}" -gt 0 ]; then
                    problem "${cloud_seats[*]} runs on ollama.com and the daemon is not signed in: run 'ollama signin', then re-run ./install.sh"
                else
                    echo "  not signed in to ollama.com: no seat needs it, so Ollama Cloud choices are skipped"
                fi
                ;;
        esac

        pulled="$(ollama list 2>/dev/null | awk 'NR > 1 {print $1}')"
        skipped_cloud=()
        for entry in "${wanted_models[@]}"; do
            [ -z "$entry" ] && continue
            role="${entry%%$'\t'*}"
            model="${entry#*$'\t'}"
            if grep -qxF "$model" <<<"$pulled"; then
                ok "$model already on the daemon"
            elif is_cloud_tag "$model" && [ "$cloud_ready" -eq 0 ]; then
                # A cloud seat is already a problem above; a cloud option is
                # only skipped, and named once below.
                if [ "$role" = option ]; then
                    skipped_cloud+=("$model")
                fi
            elif ollama pull "$model"; then
                ok "pulled $model"
            elif [ "$role" = option ]; then
                NOTES+=("could not pull $model, an optional seat choice: the console offers it once 'ollama pull $model' succeeds")
            else
                problem "could not pull $model (signed in? tag spelled right?)"
            fi
        done
        if [ "${#skipped_cloud[@]}" -gt 0 ]; then
            NOTES+=("Ollama Cloud choices not pulled (${skipped_cloud[*]}): run 'ollama signin' and re-run ./install.sh to offer them")
        fi

        # Older NVIDIA cards need Ollama's CUDA 12 runner. Arch's ollama-cuda
        # is built with CUDA 13, which dropped compute capability below 7.5 --
        # Maxwell, Pascal and Volta, the cards Omarchy drives with its
        # nvidia-580xx packages -- so on those the daemon embeds on the CPU:
        # hours for a first corpus. cuda-embed-ollama.sh installs Ollama's own
        # build of the same version, whose cuda_v12 runner drives them, points
        # the packaged service at it, and proves the model lands wholly on the
        # cards. Run here, after the pulls, because that proof embeds with the
        # model; and with --no-cleanup, because removing packages is that
        # script's to do when someone runs it, not an install's. A daemon
        # elsewhere (OLLAMA_BASE_URL) is not this machine's to change.
        old_cards="$(timeout 10 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | awk -F. '/^[0-9]+\.[0-9]+$/ && $1 * 10 + $2 < 75' || true)"
        if [ -n "$old_cards" ] && [ -z "${OLLAMA_BASE_URL:-}" ]; then
            # --no-system promises no sudo, so there it can only look.
            if [ "$SYSTEM" -eq 0 ]; then
                gpu_fix=(./cuda-embed-ollama.sh --check)
            else
                gpu_fix=(./cuda-embed-ollama.sh --no-cleanup)
                [ "$ASSUME_YES" -eq 1 ] && gpu_fix+=(--yes)
            fi
            if [ "$CUDA12" -eq 0 ]; then
                echo "  keeping Arch's Ollama build (--no-cuda12), which cannot drive this machine's NVIDIA cards"
            elif [ ! -x ./cuda-embed-ollama.sh ]; then
                problem "cuda-embed-ollama.sh is missing, and these NVIDIA cards need the CUDA 12 build it installs"
            else
                echo "  NVIDIA cards below compute 7.5: ${gpu_fix[*]}"
                if AMBIGUITY_ROOT="$ROOT" AMBIGUITY_VENV="$VENV" "${gpu_fix[@]}"; then
                    ok "Ollama's CUDA 12 build holds the embedding model 100% on the GPU, 0% on the CPU"
                elif [ "$SYSTEM" -eq 0 ]; then
                    problem "these NVIDIA cards need Ollama's CUDA 12 build, and --no-system does not install it: ./cuda-embed-ollama.sh"
                else
                    problem "cuda-embed-ollama.sh could not put the embedding model on the cards; see above"
                fi
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. Embedding tokenizer and corpus
# ---------------------------------------------------------------------------

step "Embedding tokenizer"
# The embedding model itself is pulled with the seats above -- the daemon runs
# it. What this fetches is its tokenizer, the files the in-process chunker
# cuts passages with, so the first search is not also a download and a machine
# that goes offline later still has them. The name comes from graphrag_server,
# where the corpus and the relevance floor live too.
if "$PY" - 2>/tmp/ambiguity-embedder.log <<'PY'
from langgraph_agent.graphrag_server import EMBEDDING_TOKENIZER_NAME
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER_NAME)
print(f"  ✓ {EMBEDDING_TOKENIZER_NAME} tokenizer cached")
PY
then :; else problem "could not fetch the embedding tokenizer; see /tmp/ambiguity-embedder.log"; fi

# The corpus is not built here, and there is no step that builds one. It holds
# researched archive data only -- pages online research fetched, documents the
# operator uploaded, projects the operator opted in -- never this checkout, so
# a fresh install has nothing to index. The tokenizer above is cached so the
# first embed is an embed and not also a download.

# ---------------------------------------------------------------------------
# 6. Git, for the Builder's git_dwell
# ---------------------------------------------------------------------------

step "Git (the Builder's git_dwell)"
# git_dwell's default pipeline commits, pushes, and opens and merges a pull
# request with gh -- in this checkout, or on a run given a project (the
# console's default), in that project's own repository under projects/, which
# the Builder makes with `git init`. None of that is exercised until a Builder
# reaches it mid-run, where a missing identity or a signed-out gh stops the
# pipeline at that stage, and the Builder spends its turns on a repository
# that was never broken. So both are asked now.
if ! command -v git >/dev/null; then
    problem "git is not installed (drop --no-system, or: sudo pacman -S git)"
else
    # Asked inside this checkout, so a repo-local identity counts as well...
    git_name="$(git config user.name || true)"
    git_email="$(git config user.email || true)"
    # ...and outside it, which is all a project's own repository sees: one set
    # only in this checkout leaves every project run unable to commit.
    shared_name="$(git config --global user.name || git config --system user.name || true)"
    shared_email="$(git config --global user.email || git config --system user.email || true)"
    identity_fix="git config --global user.name 'Your Name' && git config --global user.email you@example.com"
    if [ -n "$shared_name" ] && [ -n "$shared_email" ]; then
        if [ "$git_name <$git_email>" = "$shared_name <$shared_email>" ]; then
            ok "commits are authored as $git_name <$git_email>, here and in a project's own repository"
        else
            ok "commits are authored as $git_name <$git_email> here, and as $shared_name <$shared_email> in a project's own repository"
        fi
    elif [ -n "$git_name" ] && [ -n "$git_email" ]; then
        problem "git's identity ($git_name <$git_email>) is set only in this checkout, so a run given a project cannot commit in its own repository: $identity_fix"
    else
        problem "git has no identity, so the Builder cannot commit: $identity_fix"
    fi

    # A project's own repository starts with no remote, so there git_dwell
    # commits and skips its remote stages until someone adds one; what is
    # asked below is this checkout's.
    origin="$(git remote get-url origin 2>/dev/null || true)"
    if [ -z "$origin" ]; then
        # A checkout with no remote is legitimate; the pipeline just ends early.
        echo "  no origin remote: git_dwell will commit on a branch and skip push, pr, checks and merge"
    elif [[ "$origin" != *github.com[:/]* ]]; then
        echo "  origin is not on github.com: git_dwell's pr, checks and merge stages use gh and will stop there"
    elif ! command -v gh >/dev/null; then
        problem "gh is not installed, so git_dwell stops at its pr stage (drop --no-system, or: sudo pacman -S github-cli)"
    else
        gh_signed_in() { gh auth status --hostname github.com >/dev/null 2>&1; }
        if ! gh_signed_in && interactive; then
            echo "  gh is not signed in to github.com; git_dwell opens pull requests, reads their checks and merges them with it"
            if gh auth login --hostname github.com --git-protocol https; then
                # Makes gh git's credential helper for github.com, which is
                # what lets git_dwell's plain `git push` authenticate. Only
                # after a sign-in this script started: an existing setup's
                # credentials are not this script's to rewrite.
                gh auth setup-git --hostname github.com || true
            fi
        fi
        if gh_signed_in; then
            ok "gh signed in to github.com (git_dwell can open, check and merge pull requests)"
        else
            problem "gh is not signed in: run 'gh auth login', then re-run ./install.sh"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. SearxNG, for online research
# ---------------------------------------------------------------------------

if [ "$SEARXNG" -eq 1 ]; then
    step "SearxNG (online research)"
    # Without it, online research goes through DuckDuckGo's keyless endpoint,
    # which answers a handful of quick requests from one address with its bot
    # check and then nothing for about half an hour -- and a goal fans out into
    # several searches. A SearxNG of your own asks many engines and blocks
    # nobody. It runs as a rootless podman container under your user's
    # systemd, so it starts with your session, binds loopback only, and needs
    # no root after the packages above.
    #
    # Two files, both yours afterwards. settings.yml is written once and never
    # overwritten; the quadlet unit is rewritten only when it differs.
    # FORCE_OWNERSHIP=false and a read-only file mount matter together: the
    # image's entrypoint otherwise chowns its config volume to its own user,
    # which under rootless podman leaves the host file owned by a subordinate
    # uid you cannot edit.
    searxng_conf="${XDG_CONFIG_HOME:-$HOME/.config}/ambiguity/searxng"
    searxng_unit="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd/ambiguity-searxng.container"
    searxng_image="docker.io/searxng/searxng:latest"
    searxng_up() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$1/healthz" || true)" = "200" ]; }

    # localhost and 127.0.0.1 are one instance, and the .env template once said
    # localhost: that line must not read as someone else's SearxNG.
    searxng_named="${SEARXNG_URL:-}"
    searxng_named="${searxng_named%/}"
    searxng_named="${searxng_named/localhost:/127.0.0.1:}"
    if [ -n "$searxng_named" ] && [ "$searxng_named" != "$SEARXNG_LOCAL" ]; then
        echo "  SEARXNG_URL names $SEARXNG_URL; using that instance instead of running one"
    elif ! command -v podman >/dev/null; then
        problem "podman is not installed, so SearxNG cannot run (drop --no-system, or: sudo pacman -S podman crun)"
    elif ! grep -q "^$(id -un):" /etc/subuid 2>/dev/null; then
        problem "rootless podman needs subordinate ids for $(id -un): sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(id -un)"
    elif ! systemctl --user show-environment >/dev/null 2>&1; then
        problem "no systemd user session to run SearxNG under (log in to a desktop or console session, then re-run)"
    else
        changed=0
        mkdir -p "$searxng_conf"
        if [ ! -f "$searxng_conf/settings.yml" ]; then
            secret="$("$PY" -c 'import secrets; print(secrets.token_hex(32))')"
            cat >"$searxng_conf/settings.yml" <<EOF
# Written by install.sh for the Ambiguity console's online research, and never
# overwritten by it: edit freely, then systemctl --user restart ambiguity-searxng.
# Only what differs from SearxNG's own defaults is here.
use_default_settings: true

general:
  instance_name: "Ambiguity research"

server:
  secret_key: "$secret"
  # One local caller, not a public instance. The limiter would rate-limit the
  # searches a single goal fans out into, and needs a valkey server besides.
  limiter: false
  public_instance: false
  image_proxy: false

search:
  # web_research asks for JSON. SearxNG allows only html by default and answers
  # any other format with 403.
  formats:
    - html
    - json
EOF
            # Owner-only, for the secret key. The image runs as root inside the
            # container, which rootless podman maps to you, so it still reads it.
            chmod 600 "$searxng_conf/settings.yml"
            changed=1
            ok "wrote $searxng_conf/settings.yml"
        elif ! grep -qE '^[[:space:]]*-[[:space:]]*json[[:space:]]*$' "$searxng_conf/settings.yml"; then
            problem "$searxng_conf/settings.yml does not list json under search.formats, and the app asks SearxNG for JSON"
        else
            ok "settings.yml present; left as it is"
        fi

        # No network-online.target: a user unit cannot wait on a system target,
        # and quadlet adds its own wait (podman-user-wait-network-online).
        unit_text="$(cat <<EOF
# Written by install.sh for the Ambiguity console's online research.
[Unit]
Description=SearxNG for the Ambiguity console's online research

[Container]
Image=$searxng_image
ContainerName=ambiguity-searxng
PublishPort=127.0.0.1:$SEARXNG_PORT:8080
Volume=$searxng_conf/settings.yml:/etc/searxng/settings.yml:ro
Volume=ambiguity-searxng-data:/var/cache/searxng
Environment=FORCE_OWNERSHIP=false
Environment=SEARXNG_BASE_URL=$SEARXNG_LOCAL/

[Service]
Restart=on-failure
TimeoutStartSec=300
# Rootless podman's pasta copies the host's routes when the container starts.
# Started at login before Wi-Fi is up, it gets none, and every engine then fails
# on DNS until a restart. Wait (bounded, never fatal) for a default route.
ExecStartPre=/usr/bin/sh -c 'for _ in \$(seq 1 90); do ip route show default | grep -q . && exit 0; sleep 1; done; exit 0'

[Install]
WantedBy=default.target
EOF
        )"
        if [ "$(cat "$searxng_unit" 2>/dev/null || true)" != "$unit_text" ]; then
            mkdir -p "$(dirname "$searxng_unit")"
            printf '%s\n' "$unit_text" >"$searxng_unit"
            changed=1
            ok "wrote $searxng_unit"
        fi

        # Pulled here rather than by the unit's first start, which systemd
        # would time out on a slow connection.
        if podman image exists "$searxng_image"; then
            ok "$searxng_image present"
        elif podman pull --quiet "$searxng_image" >/dev/null; then
            ok "pulled $searxng_image"
        else
            problem "could not pull $searxng_image"
        fi

        systemctl --user daemon-reload || true
        if [ "$changed" -eq 1 ]; then
            systemctl --user restart ambiguity-searxng.service || true
        else
            systemctl --user start ambiguity-searxng.service || true
        fi
        for _ in $(seq 1 60); do searxng_up "$SEARXNG_LOCAL" && break; sleep 1; done

        # Answering is not reaching the internet. A container whose pasta came up
        # without a route answers /healthz and finds nothing, so the container is
        # asked to open a connection out, and restarted (which re-reads the host's
        # routes) until it can. Three tries, spaced, and never fatal: a machine
        # that is genuinely offline is reported by the search test below.
        searxng_egress() {
            podman exec ambiguity-searxng python3 -c \
                'import socket; socket.create_connection(("1.1.1.1", 443), timeout=5)' >/dev/null 2>&1
        }
        if searxng_up "$SEARXNG_LOCAL"; then
            for attempt in 1 2 3; do
                searxng_egress && break
                echo "  SearxNG's container has no route out (attempt $attempt); restarting it"
                for _ in $(seq 1 30); do ip route show default 2>/dev/null | grep -q . && break; sleep 1; done
                systemctl --user restart ambiguity-searxng.service || true
                for _ in $(seq 1 60); do searxng_up "$SEARXNG_LOCAL" && break; sleep 1; done
                sleep 2
            done
            searxng_egress && ok "SearxNG's container can reach the internet" \
                || NOTES+=("SearxNG's container has no route out; is this machine online? then: systemctl --user restart ambiguity-searxng")
        fi

        if searxng_up "$SEARXNG_LOCAL"; then
            ok "SearxNG answering at $SEARXNG_LOCAL (starts with your session)"
            # The app uses SearxNG only when SEARXNG_URL says so, and only once
            # it answers is it safe to say so: a URL to nothing would fail every
            # search that DuckDuckGo would at least have tried.
            if [ -z "${SEARXNG_URL:-}" ]; then
                printf '\n# Added by install.sh: the SearxNG it runs for online research.\nSEARXNG_URL=%s\n' \
                    "$SEARXNG_LOCAL" >>.env
                export SEARXNG_URL="$SEARXNG_LOCAL"
                ok "SEARXNG_URL=$SEARXNG_LOCAL added to .env"
            fi
        elif ! systemctl --user is-active --quiet ambiguity-searxng.service \
                && [ -n "$(ss -ltnH "sport = :$SEARXNG_PORT" 2>/dev/null)" ]; then
            problem "port $SEARXNG_PORT is taken by another program, so SearxNG cannot bind it: set SEARXNG_PORT in .env to a free port and re-run"
        else
            problem "SearxNG did not answer at $SEARXNG_LOCAL; see: journalctl --user -u ambiguity-searxng"
        fi
    fi

    # Whichever instance SEARXNG_URL names, it is asked one question through
    # the request the app itself sends, so "answering" means answering the app:
    # the JSON format enabled and the reply in the shape it parses.
    if [ -n "${SEARXNG_URL:-}" ]; then
        searx_status=0
        "$PY" - <<'PY' || searx_status=$?
import sys
import time

import httpx

from langgraph_agent import web_research

try:
    with httpx.Client() as client:
        # A fresh instance's engines warm up and rate-limit in bursts, so an
        # empty answer is asked again, spaced, before it is called empty.
        hits = []
        for attempt in range(6):
            hits = web_research._search_searxng("python packaging", 5, client)
            if hits:
                break
            time.sleep(5)
except httpx.HTTPStatusError as exc:
    code = exc.response.status_code
    hint = " (403: json is not in the instance's search.formats)" if code == 403 else ""
    print(f"  ✗ {web_research.SEARXNG_URL} refused the app's search: HTTP {code}{hint}")
    sys.exit(1)
except (httpx.HTTPError, ValueError) as exc:
    print(f"  ✗ {web_research.SEARXNG_URL} could not answer the app's search: {type(exc).__name__}: {exc}")
    sys.exit(1)
if not hits:
    print(f"  ! {web_research.SEARXNG_URL} answered with no results: its engines may be refusing it for now")
    sys.exit(3)
print(f"  ✓ a search through the app returned {len(hits)} result(s)")
PY
        case "$searx_status" in
            0) ;;
            3) NOTES+=("SearxNG answered a test search with no results; see SearxNG above") ;;
            *) problem "online research cannot use SEARXNG_URL; see SearxNG above" ;;
        esac
    fi
fi

# ---------------------------------------------------------------------------
# 8. PostgreSQL, in Docker
# ---------------------------------------------------------------------------

if [ "$POSTGRES" -eq 1 ]; then
    step "PostgreSQL + pgvector (Docker)"
    # The development database Omarchy's own installer runs
    # (omarchy-install-docker-dbs PostgreSQL), with its flags exactly, so this
    # and that menu entry make one container rather than two fighting over the
    # port: postgres18, published on loopback only, restarting with the daemon,
    # and trust authentication -- a local connection logs in without a
    # password, which is the development setting and the reason it is never
    # published past 127.0.0.1.
    #
    # One difference: the image. The corpus lives here -- every chunk with its
    # embedding, the entity graph, the relevance floor, one schema per corpus
    # (src/langgraph_agent/corpus_store.py) -- and embeddings need pgvector,
    # which postgres:18 does not ship. pgvector/pgvector:pg18-trixie is
    # postgres:18 with the extension added, on the same Debian 13: the bookworm
    # build has an older glibc, whose collations differ from the ones a
    # postgres:18 data directory was created with. A postgres18 already running
    # postgres:18 -- Omarchy's menu makes exactly that -- is recreated from it
    # below, on the same data volume, so nothing in it is lost.
    #
    # The console also exports .env to everything it runs, so a script the
    # Builder writes finds the database there.
    postgres_container=postgres18
    postgres_image=pgvector/pgvector:pg18-trixie
    postgres_local="postgresql://postgres@127.0.0.1:5432/postgres"
    me="$(id -un)"
    # A URL's password, if it carries one, stays out of the terminal.
    redacted() { sed -E 's#(://[^:/@]*):[^@/]*@#\1:***@#' <<<"$1"; }
    # Answering is logging in and running a query through the URL itself, not
    # an open port: a server that wants a password accepts connections too.
    pg_answers() { PGCONNECT_TIMEOUT=3 psql "$1" -w -X -q -t -A -c 'select 1' >/dev/null 2>&1; }
    pg_version() { PGCONNECT_TIMEOUT=3 psql "$1" -w -X -q -t -A -c 'show server_version' 2>/dev/null | cut -d' ' -f1; }
    pg_has_vector() {
        [ "$(PGCONNECT_TIMEOUT=3 psql "$1" -w -X -q -t -A \
            -c "select count(*) from pg_available_extensions where name = 'vector'" 2>/dev/null)" = 1 ]
    }
    pg_vector_version() { PGCONNECT_TIMEOUT=3 psql "$1" -w -X -q -t -A -c "select extversion from pg_extension where extname = 'vector'" 2>/dev/null; }
    # Waits out a first start, which initialises its data directory before it
    # listens.
    pg_wait() { for _ in $(seq 1 60); do pg_answers "$1" && return 0; sleep 1; done; return 1; }

    if ! command -v docker >/dev/null; then
        problem "docker is not installed, so PostgreSQL cannot run (drop --no-system, or: sudo pacman -S docker postgresql-libs)"
    elif ! command -v psql >/dev/null; then
        problem "psql is not installed, so the database cannot be checked (drop --no-system, or: sudo pacman -S postgresql-libs)"
    else
        # The docker group. The daemon runs as root, so membership is
        # passwordless root for anything running as you, which is why Omarchy
        # leaves the account out and puts Docker behind sudo. This install opts
        # in -- the toggle Omarchy offers as Setup > Security > Sudoless Docker --
        # so the database can be looked after without a password (docker logs
        # postgres18, docker exec -it postgres18 psql); --no-docker-group keeps
        # Omarchy's default. Membership is read from the account, as
        # omarchy-sudo-docker --configured reads it: a group granted now is
        # real, but only a new login -- in practice a reboot -- carries it.
        if [[ " $(id -nG "$me" 2>/dev/null) " == *" docker "* ]]; then
            if [ -e /var/run/docker.sock ] && [ ! -w /var/run/docker.sock ]; then
                ok "$me is in the docker group, from the next login (a reboot, on Omarchy)"
            else
                ok "$me is in the docker group"
            fi
        elif [ "$DOCKER_GROUP" -eq 0 ]; then
            echo "  not adding $me to the docker group (--no-docker-group): docker stays behind sudo"
        elif [ "$SYSTEM" -eq 0 ]; then
            problem "$me is not in the docker group, and --no-system does not add you: sudo usermod -aG docker $me"
        else
            echo "  adding $me to the docker group: passwordless root for your account (--no-docker-group skips this)"
            if sudo usermod -aG docker "$me"; then
                # What Omarchy's own toggle records, so its update asks for the reboot.
                if command -v omarchy-state >/dev/null; then omarchy-state set reboot-required || true; fi
                ok "$me added to the docker group"
                NOTES+=("you were added to the docker group, which applies after a reboot; until then docker still needs sudo")
            else
                problem "could not add $me to the docker group: sudo usermod -aG docker $me"
            fi
        fi

        # Omarchy enables docker.socket and not the service, so the daemon
        # starts when something first talks to it -- and --restart
        # unless-stopped restarts a container when the daemon starts. After a
        # reboot the database would be down until someone happened to run
        # docker, so the service is enabled, as ollama.service is above.
        if [ "$(systemctl is-enabled docker.service 2>/dev/null || true)" = "enabled" ]; then
            ok "docker.service starts at boot, so the database comes back after a reboot"
        elif [ "$SYSTEM" -eq 0 ]; then
            echo "  docker.service does not start at boot, and --no-system does not enable it: sudo systemctl enable --now docker.service"
            NOTES+=("PostgreSQL is down after a reboot until something runs docker: sudo systemctl enable docker.service")
        else
            echo "  enabling docker.service, so the database comes back after a reboot"
            if sudo systemctl enable --now docker.service; then
                ok "docker.service starts at boot"
            else
                problem "could not enable docker.service: sudo systemctl enable --now docker.service"
            fi
        fi

        # Whichever server DATABASE_URL names is the one asked, so a URL
        # pointing somewhere else is checked and left alone, as SEARXNG_URL is.
        # postgres:// and localhost are spellings of this script's own URL.
        database_url="${DATABASE_URL:-$postgres_local}"
        case "$database_url" in
            postgres://*) database_own="postgresql://${database_url#postgres://}" ;;
            *)            database_own="$database_url" ;;
        esac
        database_own="${database_own/@localhost:/@127.0.0.1:}"
        # The daemon, when it is needed: directly when the socket is writable,
        # through sudo when it is not -- a group granted above does not reach
        # this run.
        docker_cmd=()
        if [ -w /var/run/docker.sock ]; then
            docker_cmd=(docker)
        elif [ "$SYSTEM" -eq 1 ]; then
            docker_cmd=(sudo docker)
        fi
        answered=0
        if pg_answers "$database_url"; then
            answered=1
        elif [ "$database_own" != "$postgres_local" ]; then
            problem "DATABASE_URL names $(redacted "$database_url"), which does not answer: fix it in .env, or remove it and re-run to have this script run PostgreSQL"
        else
            started=0
            [ "${#docker_cmd[@]}" -gt 0 ] && echo "  nothing answers at 127.0.0.1:5432 yet; bringing up $postgres_container (${docker_cmd[*]})"
            if [ "${#docker_cmd[@]}" -eq 0 ]; then
                problem "PostgreSQL does not answer at 127.0.0.1:5432, and --no-system cannot reach Docker: sudo docker start $postgres_container, or re-run without --no-system"
            elif state="$("${docker_cmd[@]}" container inspect -f '{{.State.Status}}' "$postgres_container" 2>/dev/null)"; then
                # Started, never recreated: the data lives in this container's
                # volume, and a new container would come up empty beside it.
                if "${docker_cmd[@]}" start "$postgres_container" >/dev/null; then
                    started=1
                    ok "started the existing $postgres_container container (it was $state)"
                else
                    problem "could not start the $postgres_container container; see: docker logs $postgres_container"
                fi
            elif [ -n "$(ss -ltnH 'sport = :5432' 2>/dev/null)" ]; then
                problem "port 5432 is taken, and the server there does not let $postgres_local log in: point DATABASE_URL in .env at it, or free the port, and re-run"
            else
                # Pulled on its own first, so a slow download reads as one
                # rather than as a container that will not start.
                if ! "${docker_cmd[@]}" image inspect "$postgres_image" >/dev/null 2>&1; then
                    echo "  pulling $postgres_image"
                    "${docker_cmd[@]}" pull --quiet "$postgres_image" >/dev/null || true
                fi
                if "${docker_cmd[@]}" run -d --restart unless-stopped -p "127.0.0.1:5432:5432" \
                        --name="$postgres_container" -e POSTGRES_HOST_AUTH_METHOD=trust \
                        "$postgres_image" >/dev/null; then
                    started=1
                    ok "created the $postgres_container container from $postgres_image"
                else
                    problem "could not run $postgres_image as $postgres_container; see the docker error above"
                fi
            fi
            if [ "$started" -eq 1 ]; then
                if pg_wait "$postgres_local"; then
                    answered=1
                else
                    problem "PostgreSQL did not answer at $postgres_local; see: docker logs $postgres_container"
                fi
            fi
        fi

        # The corpus needs pgvector. A server that lacks it is either this
        # script's own container on the old image -- recreated from the
        # pgvector image on the same data -- or someone else's, which is left
        # alone and named.
        if [ "$answered" -eq 1 ] && ! pg_has_vector "$database_url"; then
            current_image=""
            if [ "$database_own" = "$postgres_local" ] && [ "${#docker_cmd[@]}" -gt 0 ]; then
                current_image="$("${docker_cmd[@]}" container inspect -f '{{.Config.Image}}' "$postgres_container" 2>/dev/null || true)"
            fi
            if [ "$database_own" != "$postgres_local" ]; then
                problem "$(redacted "$database_url") has no pgvector, and the corpus stores its embeddings with it: install the extension there, or point DATABASE_URL at a server that has it"
                answered=0
            elif [ -z "$current_image" ]; then
                problem "the PostgreSQL at 127.0.0.1:5432 has no pgvector and is not a $postgres_container container this script can recreate: run it from $postgres_image, or point DATABASE_URL at a server with pgvector"
                answered=0
            else
                # Where its data lives: the volume (or directory) mounted at
                # /var/lib/postgresql, which the new container mounts in turn.
                data_mount="$("${docker_cmd[@]}" container inspect -f \
                    '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{if eq .Type "volume"}}{{.Name}}{{else}}{{.Source}}{{end}}{{end}}{{end}}' \
                    "$postgres_container" 2>/dev/null || true)"
                if [ -z "$data_mount" ]; then
                    problem "$postgres_container ($current_image) has no data volume at /var/lib/postgresql, so recreating it would lose its data: move it to $postgres_image by hand"
                    answered=0
                else
                    echo "  $postgres_container runs $current_image, which has no pgvector;"
                    echo "  recreating it from $postgres_image on the same data ($data_mount)"
                    "${docker_cmd[@]}" pull --quiet "$postgres_image" >/dev/null || true
                    answered=0
                    if "${docker_cmd[@]}" image inspect "$postgres_image" >/dev/null 2>&1 \
                            && "${docker_cmd[@]}" rm -f "$postgres_container" >/dev/null \
                            && "${docker_cmd[@]}" run -d --restart unless-stopped -p "127.0.0.1:5432:5432" \
                                --name="$postgres_container" -e POSTGRES_HOST_AUTH_METHOD=trust \
                                -v "$data_mount:/var/lib/postgresql" "$postgres_image" >/dev/null; then
                        if pg_wait "$postgres_local" && pg_has_vector "$postgres_local"; then
                            answered=1
                            ok "recreated $postgres_container from $postgres_image; its data is where it was"
                        else
                            problem "$postgres_container did not come back with pgvector; see: docker logs $postgres_container"
                        fi
                    else
                        problem "could not recreate $postgres_container from $postgres_image (its data is still in $data_mount); see the docker error above"
                    fi
                fi
            fi
        fi

        if [ "$answered" -eq 1 ]; then
            ok "PostgreSQL $(pg_version "$database_url") answers at $(redacted "$database_url")"
            # Created here as well as by the console, so a role that may create
            # it does so once and the check below asks a database that has it.
            if PGCONNECT_TIMEOUT=3 psql "$database_url" -w -X -q -c 'create extension if not exists vector' >/dev/null 2>&1 \
                    && [ -n "$(pg_vector_version "$database_url")" ]; then
                ok "pgvector $(pg_vector_version "$database_url") is enabled"
            else
                problem "pgvector is available but could not be enabled in $(redacted "$database_url"): run 'create extension vector' there as a superuser"
            fi
            # Written only once a query has gone through it, for SEARXNG_URL's
            # reason: a URL to nothing is worse than no URL.
            if [ -z "${DATABASE_URL:-}" ]; then
                printf '\n# Added by install.sh: the PostgreSQL it runs in Docker.\nDATABASE_URL=%s\n' \
                    "$postgres_local" >>.env
                export DATABASE_URL="$postgres_local"
                ok "DATABASE_URL=$postgres_local added to .env"
            fi

            # An open port and an extension are presence, not a working store.
            # So one vector of the embedding model's width goes through the
            # class a run uses -- into a schema of its own, read back by exact
            # search, and dropped -- and no corpus is touched.
            "$PY" - <<'PY' || problem "the corpus store could not write and search a vector; see PostgreSQL above"
import sys
import tempfile

import numpy as np

from langgraph_agent.corpus_store import create_corpus_store
from langgraph_agent.graphrag_server import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME

with tempfile.TemporaryDirectory() as scratch:
    try:
        store = create_corpus_store(
            f"{scratch}/install-check",
            embedding_model=EMBEDDING_MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
        )
    except Exception as exc:
        print(f"  ✗ {type(exc).__name__}: {exc}")
        sys.exit(1)
    try:
        vector = np.ones(EMBEDDING_DIMENSIONS, dtype=np.float32)
        store.replace_document(
            "install-check.md", ["install-check.md#0000"], [vector], ["install check"],
            [{"doc_id": "install-check.md", "chunk_index": 0}],
        )
        hits = store.query([vector], n_results=1)
        if hits["ids"] != [["install-check.md#0000"]] or abs(hits["distances"][0][0]) > 1e-6:
            print(f"  ✗ the vector stored is not the vector found: {hits['ids']}")
            sys.exit(1)
    except Exception as exc:
        print(f"  ✗ {type(exc).__name__}: {exc}")
        sys.exit(1)
    finally:
        store.drop()
print(f"  ✓ the corpus store writes and finds a {EMBEDDING_DIMENSIONS}-dimension vector")
PY
        fi
    fi
elif [ -z "${DATABASE_URL:-}" ]; then
    NOTES+=("--no-postgres: the corpus lives in PostgreSQL, so set DATABASE_URL in .env to a server with pgvector before starting the console")
fi

# ---------------------------------------------------------------------------
# 9. Checks
# ---------------------------------------------------------------------------

if [ "$CHECKS" -eq 1 ]; then
    step "Checks (CLAUDE.md's Quick Reference)"
    # Read out of the Quick Reference instead of copied, the same promise
    # ci.yml keeps: "all checks pass" means one thing wherever it is said.
    mapfile -t checks < <(sed -n '/^# Run all checks$/,/^```$/p' CLAUDE.md \
                          | grep -E '^(ruff|mypy|python -m pytest)')
    if [ "${#checks[@]}" -eq 0 ]; then
        problem "no checks found under 'Run all checks' in CLAUDE.md"
    fi
    for check in "${checks[@]}"; do
        log="/tmp/ambiguity-check-${check%% *}.log"
        if PATH="$VENV/bin:$PATH" bash -c "$check" >"$log" 2>&1; then
            ok "$check"
        else
            problem "$check -- failed; see $log"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 10. App launcher
# ---------------------------------------------------------------------------

if [ "$DESKTOP" -eq 1 ]; then
    step "App launcher"
    # A freedesktop entry, which is what Omarchy's launcher (Super+Space)
    # lists. It runs launch_console.sh without a terminal: the console's own
    # exit button stops the server, and launching again while it is up just
    # opens the page.
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    cat >"$apps/ambiguity-console.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Ambiguity Console
Comment=Architect, Planner, Researcher and Builder, in the browser
Exec="$ROOT/launch_console.sh"
Path=$ROOT
Icon=utilities-terminal
Terminal=false
Categories=Development;
EOF
    command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true
    ok "\"Ambiguity Console\" added to the app launcher"
fi

# ---------------------------------------------------------------------------
# 11. Console: does it start?
# ---------------------------------------------------------------------------

step "Console"
# The promise at the top of this script is a console that runs, and nothing
# above starts one. So serve.py is started on a free port, asked for
# /api/status -- what launch_console.sh waits on -- and for the page itself,
# then stopped. REBUILD_CORPUS=0 is what leaves the machine as it was
# found: a console brings the corpus up to date as soon as it is serving, so
# without it this check started a first build -- the embedding model loaded
# onto the cards, a corpus schema created -- and then killed it seconds in. SIGTERM,
# not SIGINT: bash starts background jobs with SIGINT ignored, and the server
# would never see it.
console_log=/tmp/ambiguity-console-check.log
console_port="$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
PORT="$console_port" REBUILD_CORPUS=0 "$PY" serve.py >"$console_log" 2>&1 &
console_pid=$!
console_up=0
for _ in $(seq 1 60); do
    if curl -sf -m 2 "http://127.0.0.1:$console_port/api/status" >/dev/null 2>&1; then
        console_up=1
        break
    fi
    kill -0 "$console_pid" 2>/dev/null || break
    sleep 0.5
done
# The page is read whole, not piped into grep -q: grep exits at the first match,
# curl then fails writing the rest, and pipefail reads that as a failed fetch.
console_page=""
[ "$console_up" -eq 1 ] && console_page="$(curl -sf -m 5 "http://127.0.0.1:$console_port/" || true)"
if [ "$console_up" -eq 1 ] && grep -qi '<html' <<<"$console_page"; then
    ok "serve.py starts, answers /api/status and serves the console page"
elif [ "$console_up" -eq 1 ]; then
    problem "serve.py answers /api/status but did not serve the console page; see $console_log"
else
    problem "serve.py did not start; see $console_log"
fi
kill "$console_pid" 2>/dev/null || true
wait "$console_pid" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Summary: can the embedder and each seat actually run?
# ---------------------------------------------------------------------------

step "Embedder"
# Pulling a model proves it is on the daemon, not that it embeds, and a run
# would find out only in its corpus phase. So the active model is asked for one
# vector through the class a run uses -- no corpus is opened or built -- and
# where the daemon placed it is read back: a split reply is identical to a whole
# one, and the split is what turns a first build of minutes into hours
# (OLLAMA_EMBED_OPTIONS in graphrag_server.py has the measurements). A model
# this check loaded is unloaded again, so the daemon is left holding what it
# held and the next run's GPU placement is not decided by an install.
embed_status=0
"$PY" - <<'PY' || embed_status=$?
import logging
import subprocess
import sys

# The retries and the circuit opening are the healing journal's; the line
# printed below says the same once.
logging.disable(logging.CRITICAL)

from langgraph_agent.config import daemon_unreachable, ollama_cpu_share  # noqa: E402
from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME, OllamaEmbedder  # noqa: E402
from langgraph_agent.self_healing import CircuitOpenError  # noqa: E402

model = EMBEDDING_MODEL_NAME

loaded_before = ollama_cpu_share(model) is not None
embedder = OllamaEmbedder(model)
try:
    vector = embedder.encode("install check")
except Exception as exc:
    # An unreachable daemon raises its own error (or, once its circuit opens,
    # the refusal), never a RuntimeError: caught as one, it ended this step in
    # a traceback and the summary blamed the GPU placement.
    if isinstance(exc, CircuitOpenError) or daemon_unreachable(exc):
        print(f"  ✗ the Ollama daemon did not answer, so {model} could not be asked to embed")
        sys.exit(2)
    print(f"  ✗ {model} could not embed: {type(exc).__name__}: {exc}")
    sys.exit(3)
print(f"  ✓ {model} embeds ({len(vector)} dimensions)")

status = 0
share = embedder.cpu_share
if share is None:
    print(f"  (the daemon did not say where it placed {model})")
elif share >= 0.99:
    print(f"  ✗ {model} is wholly on the CPU: the daemon initialised no GPU for it, so a first "
          "corpus build takes hours. journalctl -u ollama says why; 'skipping CUDA device' "
          "is a card Arch's CUDA 13 build cannot drive, which cuda-embed-ollama.sh fixes.")
    status = 1
elif share > 0:
    # Every layer is forced onto the cards, so a split is not a slow machine
    # but a daemon ignoring the options -- and the model is to be 0% on the CPU.
    print(f"  ✗ {embedder.placement_note}")
    status = 1
else:
    print(f"  ✓ {model} is 100% on the GPU, 0% on the CPU")

if not loaded_before:
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass
sys.exit(status)
PY
case "$embed_status" in
    0) ;;
    2) problem "the embedder could not be checked: the Ollama daemon is not answering (see Ollama above)" ;;
    3) problem "the embedding model could not embed; see Embedder above (pulled? ollama pull $(sed -n 's/^EMBEDDING_MODEL_NAME = "\(.*\)"/\1/p' src/langgraph_agent/graphrag_server.py))" ;;
    *) problem "the embedder is not running 100% on the GPU as configured; see Embedder above" ;;
esac

step "Seats"
# get_agent_status is the one thing that knows whether a seat is live, stubbed
# or failing, so it is asked rather than inferred from what was installed. But
# a tag on the daemon and a signed-in account are presence, not liveness -- the
# account may have no cloud access, or be rate limited -- so each live seat is
# first sent one short prompt through its real call path, in parallel, and a
# failure lands in get_agent_status the way it would mid-run.
if ! PROBE="$PROBE" "$PY" - <<'PY'
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# A daemon that is down opens its circuit while the seats are read; each
# seat's line below says so once, which the healing journal's lines repeated.
logging.disable(logging.CRITICAL)

from langgraph_agent.config import AGENTS, get_agent_llm, get_agent_status  # noqa: E402

silent: set[str] = set()


def probe(agent: str) -> None:
    try:
        reply = get_agent_llm(agent).invoke("Reply with the single word: ready")
    except Exception:
        return  # the seat wrapper recorded why; get_agent_status reports it below
    if not str(getattr(reply, "content", reply)).strip():
        silent.add(agent)


if os.environ.get("PROBE") == "1":
    live = [agent for agent in AGENTS if get_agent_status(agent)["live"]]
    if live:
        print("  (each live seat was sent a one-word prompt)")
        with ThreadPoolExecutor(len(live)) as pool:
            list(pool.map(probe, live))

dead = 0
width = max(len(get_agent_status(agent)["model"]) for agent in AGENTS) + 2
for agent in AGENTS:
    seat = get_agent_status(agent)
    if not seat["live"]:
        mark, note = "✗", f"  -- {seat['badge']}: {seat['reason']}"
    elif agent in silent:
        mark, note = "✗", "  -- SILENT: answered a one-word prompt with nothing"
    else:
        mark, note = "✓", ""
    print(f"  {mark} {agent:11}{seat['model']:<{width}}{seat['provider']}{note}")
    dead += mark == "✗"
sys.exit(1 if dead else 0)
PY
then
    problem "not every seat can run; see the list above"
fi

print_notes() {
    [ "${#NOTES[@]}" -eq 0 ] && return 0
    echo "  Worth knowing:"
    for n in "${NOTES[@]}"; do echo "    - $n"; done
}

echo
echo "============================================"
if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    echo "  Ready. Start the console with:"
    echo "    ./launch_console.sh"
    [ "$DESKTOP" -eq 1 ] && echo "  or pick \"Ambiguity Console\" from the app launcher."
    print_notes
    echo "============================================"
    exit 0
fi
echo "  Installed, with ${#PROBLEMS[@]} thing(s) left to fix:"
for p in "${PROBLEMS[@]}"; do echo "    - $p"; done
print_notes
echo "  Fix them and re-run ./install.sh; finished steps are skipped."
echo "============================================"
exit 1
