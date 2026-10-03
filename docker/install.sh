#!/usr/bin/env bash
# The Docker-only install: from a fresh Arch / Omarchy machine to a console
# running in a container, with no Python environment on the host.
#
# ../install.sh sets up the console to run *on the host*: a venv, the checks,
# a launcher. This sets up only what the container cannot carry itself:
#   * Docker, with docker.service at boot (and the docker group);
#   * the Ollama daemon and the models the seats and the embedder use -- it
#     stays on the host, where the GPUs and the weights are;
#   * a GitHub login (gh), which up.sh hands the container at every start;
#   * docker/.env, the container's settings, naming your projects folder;
#   * SearxNG for online research, unless one already answers;
#   * the image, built once from this checkout; then it starts the console.
# Everything the console runs is inside the image, so a host with no Python
# runs it; the checkout is only what the image is built from.
#
# Safe to re-run: every step looks before it acts.
#
# Usage:
#   ./docker/install.sh                     everything below
#   ./docker/install.sh --projects DIR      the folder whose projects the console
#                                           works in (default ~/Projects)
#   ./docker/install.sh --no-searxng        no online research server
#   ./docker/install.sh --no-docker-group   keep Docker behind sudo
#   ./docker/install.sh --no-start          set up and build, do not start
#   ./docker/install.sh --yes               never prompt (no sign-ins)

set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE_FILE=docker/compose.yml
DENV=docker/.env

PROJECTS_DIR="" SEARXNG=1 DOCKER_GROUP=1 START=1 ASSUME_YES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --projects)   PROJECTS_DIR="${2:?--projects needs a directory}"; shift ;;
        --projects=*) PROJECTS_DIR="${1#--projects=}" ;;
        --no-searxng) SEARXNG=0 ;;
        --no-docker-group) DOCKER_GROUP=0 ;;
        --no-start)   START=0 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        -h|--help)    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

PROBLEMS=()
NOTES=()
problem() { PROBLEMS+=("$1"); echo "  ✗ $1"; }
step()    { echo; echo "► $1"; }
ok()      { echo "  ✓ $1"; }
interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; }

# One line of docker/.env set to a value: replaced where it is, appended where
# it is not, so a re-run never stacks duplicates.
env_set() {
    local key="$1" value="$2"
    if grep -q "^$key=" "$DENV"; then
        sed -i "s|^$key=.*|$key=$value|" "$DENV"
    else
        printf '%s=%s\n' "$key" "$value" >>"$DENV"
    fi
}
env_get() { sed -n "s/^[[:space:]]*$1=//p" "$DENV" 2>/dev/null | tail -n 1; }

echo "============================================"
echo "  Ambiguity 4-Agent Console -- Docker install"
echo "============================================"

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as your own user, not root; it calls sudo where it must." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

step "System packages (pacman)"
# No python, no compiler: the image carries its own. git and github-cli are
# for the host's GitHub login, which the container borrows.
REQUIRED=(docker docker-compose docker-buildx ollama git github-cli curl)
if ! command -v pacman >/dev/null; then
    echo "  pacman not found: this installer targets Arch / Omarchy." >&2
    echo "  Install the equivalents of: ${REQUIRED[*]}, then re-run." >&2
    exit 1
fi
mapfile -t missing < <(pacman -T "${REQUIRED[@]}" || true)
if [ "${#missing[@]}" -eq 0 ]; then
    ok "all ${#REQUIRED[@]} packages already installed"
else
    echo "  installing: ${missing[*]}"
    pacman_flags=(-S --needed)
    [ "$ASSUME_YES" -eq 1 ] && pacman_flags+=(--noconfirm)
    # No -y, for ../install.sh's reason: a partial upgrade is not supported.
    if ! sudo pacman "${pacman_flags[@]}" "${missing[@]}"; then
        echo "  pacman failed; if a package was not found, run omarchy-update" >&2
        echo "  (or sudo pacman -Syu) and re-run." >&2
        exit 1
    fi
    ok "installed ${#missing[@]} package(s)"
fi

# ---------------------------------------------------------------------------
# 2. Docker
# ---------------------------------------------------------------------------

step "Docker"
me="$(id -un)"
# Omarchy enables docker.socket, not the service; enabled, the database and
# SearxNG come back after a reboot without anything having to poke Docker.
if [ "$(systemctl is-enabled docker.service 2>/dev/null || true)" = "enabled" ]; then
    ok "docker.service starts at boot"
elif sudo systemctl enable --now docker.service; then
    ok "enabled docker.service"
else
    problem "could not enable docker.service: sudo systemctl enable --now docker.service"
fi

# The docker group is passwordless root for your account, which is why Omarchy
# leaves you out of it; --no-docker-group keeps that, and up.sh uses sudo.
if [[ " $(id -nG "$me" 2>/dev/null) " == *" docker "* ]]; then
    ok "$me is in the docker group"
elif [ "$DOCKER_GROUP" -eq 0 ]; then
    echo "  not adding $me to the docker group (--no-docker-group): docker stays behind sudo"
elif sudo usermod -aG docker "$me"; then
    if command -v omarchy-state >/dev/null; then omarchy-state set reboot-required || true; fi
    ok "$me added to the docker group"
    NOTES+=("the docker group applies after a reboot; until then up.sh runs docker through sudo")
else
    problem "could not add $me to the docker group: sudo usermod -aG docker $me"
fi

docker_cmd=(docker)
if [ -e /var/run/docker.sock ] && [ ! -w /var/run/docker.sock ]; then
    docker_cmd=(sudo docker)
fi
compose=("${docker_cmd[@]}" compose -f "$COMPOSE_FILE")
for _ in $(seq 1 20); do "${docker_cmd[@]}" info >/dev/null 2>&1 && break; sleep 0.5; done
if "${docker_cmd[@]}" info >/dev/null 2>&1; then
    ok "the Docker daemon answers (${docker_cmd[*]})"
else
    problem "the Docker daemon does not answer: sudo systemctl status docker"
fi

# ---------------------------------------------------------------------------
# 3. Settings (docker/.env)
# ---------------------------------------------------------------------------

step "Settings (docker/.env)"
if [ -f "$DENV" ]; then
    ok "$DENV exists; only the keys below are updated"
else
    cp .env.example "$DENV"
    chmod 600 "$DENV"
    printf '\n# Written by docker/install.sh.\n' >>"$DENV"
    ok "created $DENV from .env.example"
fi

if [ -z "$PROJECTS_DIR" ]; then
    PROJECTS_DIR="$(env_get AMBIGUITY_PROJECTS)"
    PROJECTS_DIR="${PROJECTS_DIR:-$HOME/Projects}"
fi
PROJECTS_DIR="$(realpath -m "${PROJECTS_DIR/#\~/$HOME}")"
mkdir -p "$PROJECTS_DIR"
env_set AMBIGUITY_PROJECTS "$PROJECTS_DIR"
ok "projects folder: $PROJECTS_DIR -> /app/projects in the container"
mapfile -t repos < <(find "$PROJECTS_DIR" -mindepth 2 -maxdepth 2 -name .git -printf '%h\n' 2>/dev/null | sort)
if [ "${#repos[@]}" -gt 0 ]; then
    echo "  repositories the console can work in:"
    for r in "${repos[@]}"; do echo "    - ${r##*/}"; done
fi
case "$(realpath .)" in
    "$PROJECTS_DIR"/*)
        NOTES+=("this checkout is inside $PROJECTS_DIR, so a run given project '$(basename "$(realpath .)")' can change the console's own source")
        ;;
esac

# ---------------------------------------------------------------------------
# 4. The image
# ---------------------------------------------------------------------------

step "Image (built from this checkout)"
# Built before the models are pulled: which models to pull is read from the
# code, and the image is the only Python here.
if "${compose[@]}" build console; then
    ok "ambiguity-console:latest built"
else
    problem "the image did not build; see the docker output above"
fi
image_python() {
    "${compose[@]}" run --rm --no-deps -T --entrypoint python console -
}

# ---------------------------------------------------------------------------
# 5. Ollama
# ---------------------------------------------------------------------------

step "Ollama"
OLLAMA_URL="$(env_get OLLAMA_BASE_URL)"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export OLLAMA_HOST="$OLLAMA_URL"
daemon_up() { curl -s -m 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

if ! daemon_up && systemctl cat ollama.service >/dev/null 2>&1; then
    echo "  starting ollama.service (and enabling it at boot)"
    sudo systemctl enable --now ollama.service || true
    for _ in $(seq 1 20); do daemon_up && break; sleep 0.5; done
fi

# One model resident, one request at a time -- ../install.sh's drop-in, word
# for word, so the two installers never rewrite each other's.
one_model=/etc/systemd/system/ollama.service.d/one-model.conf
one_model_body=$'[Service]\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\nEnvironment="OLLAMA_NUM_PARALLEL=1"\n'
if systemctl cat ollama.service >/dev/null 2>&1; then
    if [ "$(cat "$one_model" 2>/dev/null)"$'\n' = "$one_model_body" ]; then
        ok "the daemon holds one model at a time"
    elif sudo mkdir -p "${one_model%/*}" \
            && printf '%s' "$one_model_body" | sudo tee "$one_model" >/dev/null \
            && sudo systemctl daemon-reload \
            && sudo systemctl restart ollama.service; then
        for _ in $(seq 1 20); do daemon_up && break; sleep 0.5; done
        ok "limited the daemon to one model at a time ($one_model)"
    else
        problem "could not limit the daemon to one model; see $one_model"
    fi
fi

if ! daemon_up; then
    problem "the Ollama daemon does not answer at $OLLAMA_URL"
else
    ok "daemon answering at $OLLAMA_URL"
    # The seats' and the embedder's tags, read from the code in the image with
    # docker/.env's overrides applied -- the list ../install.sh pulls, minus
    # the optional seat choices, which the console offers once pulled.
    if model_list="$(image_python <<'PY'
from langgraph_agent.config import AGENTS, get_agent_model_info
from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME

models = {EMBEDDING_MODEL_NAME}
for agent in AGENTS:
    info = get_agent_model_info(agent)
    if info["provider"] == "ollama":
        models.add(info["model"])
print("\n".join(sorted(models)))
PY
    )"; then
        pulled="$(ollama list 2>/dev/null | awk 'NR > 1 {print $1}')"
        while read -r model; do
            [ -z "$model" ] && continue
            if grep -qxF "$model" <<<"$pulled"; then
                ok "$model already on the daemon"
            elif ollama pull "$model"; then
                ok "pulled $model"
            else
                problem "could not pull $model (a :cloud tag needs 'ollama signin')"
            fi
        done <<<"$model_list"
    else
        problem "could not read the seat and embedding models from the image, so nothing was pulled"
    fi

    # Cards below compute 7.5 need Ollama's CUDA 12 build to embed on the GPU;
    # the script that installs it proves the result through the host venv,
    # which this install does not make, so it is named rather than run.
    old_cards="$(timeout 10 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk -F. '/^[0-9]+\.[0-9]+$/ && $1 * 10 + $2 < 75' || true)"
    # cuda12.conf is the drop-in that script points the service at its build with.
    if [ -n "$old_cards" ] && [ ! -f /etc/systemd/system/ollama.service.d/cuda12.conf ]; then
        NOTES+=("NVIDIA cards below compute 7.5 embed on the CPU with Arch's Ollama build; ../install.sh --no-checks --no-desktop installs the CUDA 12 build (it needs the host venv)")
    fi
fi

# ---------------------------------------------------------------------------
# 6. GitHub
# ---------------------------------------------------------------------------

step "GitHub (git_dwell's push, pull request and merge)"
gh_signed_in() { gh auth status --hostname github.com >/dev/null 2>&1; }
if ! gh_signed_in && interactive; then
    echo "  gh is not signed in to github.com; the container borrows this login at every start"
    gh auth login --hostname github.com --git-protocol https || true
fi
if gh_signed_in; then
    ok "gh signed in to github.com; up.sh passes the login to the container"
else
    problem "gh is not signed in: run 'gh auth login', then ./docker/up.sh"
fi
if [ -n "$(git config --global user.name || true)" ] && [ -n "$(git config --global user.email || true)" ]; then
    ok "commits are authored as $(git config --global user.name) <$(git config --global user.email)>"
else
    NOTES+=("git has no global name/email, so the container commits as 'Ambiguity Builder': git config --global user.name ...; git config --global user.email ...")
fi

# ---------------------------------------------------------------------------
# 7. SearxNG
# ---------------------------------------------------------------------------

if [ "$SEARXNG" -eq 1 ]; then
    step "SearxNG (online research)"
    port="$(env_get SEARXNG_PORT)"
    port="${port:-8888}"
    local_url="http://127.0.0.1:$port"
    searxng_up() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$1/healthz" || true)" = "200" ]; }
    profiles="$(env_get COMPOSE_PROFILES)"
    if [[ ",$profiles," != *",searxng,"* ]] && searxng_up "$local_url"; then
        # The host install's podman SearxNG, most likely: shared, not doubled.
        ok "a SearxNG already answers at $local_url; the console uses it"
    else
        conf=docker/searxng/settings.yml
        if [ ! -f "$conf" ]; then
            mkdir -p "${conf%/*}"
            secret="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
            cat >"$conf" <<EOF
# Written by docker/install.sh, never overwritten: edit, then ./docker/up.sh.
use_default_settings: true
general:
  instance_name: "Ambiguity research"
server:
  secret_key: "$secret"
  limiter: false
  public_instance: false
  image_proxy: false
search:
  # web_research asks for JSON, which SearxNG refuses (403) unless listed.
  formats:
    - html
    - json
EOF
            # Readable by the image's own searxng user; the key only signs
            # this local instance's cookies.
            chmod 644 "$conf"
            ok "wrote $conf"
        fi
        env_set COMPOSE_PROFILES searxng
        ok "SearxNG will run in the stack on $local_url"
    fi
    env_set SEARXNG_URL "$local_url"
fi

# ---------------------------------------------------------------------------
# 8. Start
# ---------------------------------------------------------------------------

if [ "$START" -eq 1 ] && [ "${#PROBLEMS[@]}" -eq 0 ]; then
    step "Console"
    if ./docker/up.sh; then
        ok "the console is up"
        # What the container itself sees, which is the point of all of this.
        # shellcheck disable=SC2016  # expanded by the container's shell
        "${compose[@]}" exec -T console sh -c '
            n=$(find /app/projects -mindepth 1 -maxdepth 1 -type d | wc -l)
            echo "  ✓ the console sees $n project folder(s) in /app/projects"
            if gh auth status --hostname github.com >/dev/null 2>&1; then
                echo "  ✓ the console is signed in to GitHub"
            else
                echo "  ✗ the console is not signed in to GitHub"
            fi' || true
    else
        problem "the console did not start: ${compose[*]} logs console"
    fi
elif [ "$START" -eq 1 ]; then
    NOTES+=("not started, since something above needs fixing first; then ./docker/up.sh")
fi

echo
echo "============================================"
if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    echo "  Ready. Start it again any time (after a reboot, say) with:"
    echo "    ./docker/up.sh             (--build after a git pull, --down to stop)"
else
    echo "  ${#PROBLEMS[@]} thing(s) left to fix:"
    for p in "${PROBLEMS[@]}"; do echo "    - $p"; done
    echo "  Fix them and re-run ./docker/install.sh; finished steps are skipped."
fi
if [ "${#NOTES[@]}" -gt 0 ]; then
    echo "  Worth knowing:"
    for n in "${NOTES[@]}"; do echo "    - $n"; done
fi
echo "============================================"
[ "${#PROBLEMS[@]}" -eq 0 ]
