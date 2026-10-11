#!/usr/bin/env bash
# cuda-embed-ollama.sh -- put qwen3-embedding on THIS machine's NVIDIA cards,
# through an Ollama build whose CUDA can drive them, and clear out what the
# torch-era embedder left behind.
#
# Why this is machine-specific: the cards here are two GTX 1060 3GB, compute
# capability 6.1 (Pascal), on Omarchy's own nvidia-580xx driver -- the branch
# Omarchy installs for cards without GSP firmware (Maxwell, Pascal, Volta).
# CUDA 13 dropped exactly those cards, and Arch's ollama-cuda is built with
# CUDA 13, so its daemon logs "skipping CUDA device -- compute capability not in
# compiled architectures" and embeds on the CPU. Ollama's own release of the
# same version ships a cuda_v12 runner as well (CUDA 12.8, compiled for 5.0
# through 12.0), which found both cards as CUDA devices here on 2026-09-16.
#
# The target is the model 100% on the GPU, 0% on the CPU. The CUDA build alone
# does not get there: the daemon's own estimate offloaded 24 of 37 layers with
# 1.1 GB unused on each card (70% GPU, a batch of 8 in 23.4s). All 37 fit, and
# OLLAMA_EMBED_OPTIONS in graphrag_server.py forces them (num_gpu), which took
# the same batch to 6.5s. The probe below loads the model exactly that way.
#
# What it does, in order:
#   1. Driver: Omarchy's driver branch for these cards is installed, and CUDA's
#      kernel module (nvidia_uvm) is loaded.
#   2. Probe: which build ollama.service runs, and where it puts
#      qwen3-embedding. Wholly on the cards already: nothing to install.
#   3. Refuse if an agent run is in flight in the console: restarting the
#      daemon cuts off its corpus phase and its seats.
#   4. Install Ollama's own build of the packaged version under /opt/ollama,
#      checksum-verified, and point ollama.service at it with a drop-in. The
#      package keeps the unit, the ollama user and /var/lib/ollama, so the
#      models and the ollama.com sign-in carry over; deleting the drop-in
#      undoes it.
#   5. Verify: 100% on the GPU and 0% on the CPU, loaded the way the app loads
#      it (OLLAMA_EMBED_OPTIONS); anything less fails the script.
#   6. Remove what nothing needs any more: ollama-vulkan, ollama-cuda and the
#      CUDA 13 toolkit it pulled in, older builds under /opt/ollama, and torch,
#      triton, sentence-transformers and their nvidia-*/cuda-* wheels in the
#      venv -- the last only once the chunker's tokenizer is shown to load
#      without them.
#
# Usage:
#   ./cuda-embed-ollama.sh               diagnose and fix
#   ./cuda-embed-ollama.sh --check       diagnose only, change nothing
#   ./cuda-embed-ollama.sh --no-cleanup  fix, but remove nothing (install.sh runs this)
#   ./cuda-embed-ollama.sh --yes         no prompts (pacman --noconfirm)
#
# Environment overrides:
#   AMBIGUITY_ROOT   project checkout   (default: this script's directory)
#   AMBIGUITY_VENV   venv to clean      (default $AMBIGUITY_ROOT/.venv)
#   CONSOLE_URL      console checked for a run in flight (default http://localhost:8080)

set -euo pipefail

ROOT="${AMBIGUITY_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
VENV="${AMBIGUITY_VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"
CONSOLE_URL="${CONSOLE_URL:-http://localhost:8080}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

MODE=fix
CLEANUP=1
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --no-cleanup) CLEANUP=0 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "no venv at $VENV (run ./install.sh, or set AMBIGUITY_VENV)"
[ -z "${OLLAMA_BASE_URL:-}" ] || die "OLLAMA_BASE_URL points at $OLLAMA_BASE_URL: that daemon is not this machine's to change"

# --------------------------------------------------------------------------
# Probe: where does the daemon put the embedding model? Through the class a
# run uses, loaded with the options a run sends.
# exit 0 = wholly on the GPU, 1 = split or on the CPU, 2 = could not embed
# --------------------------------------------------------------------------
probe() {
    ( cd "$ROOT" && "$PY" - <<'EOF'
import sys
from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME, OllamaEmbedder

embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
try:
    vector = embedder.encode("gpu probe")
except RuntimeError as exc:
    print(f"  {exc}")
    sys.exit(2)
share = embedder.cpu_share
if share is None:
    print(f"  {EMBEDDING_MODEL_NAME} embeds ({len(vector)} dims), but the daemon did not say where")
    sys.exit(1)
gpu = round((1 - share) * 100)
print(f"  {EMBEDDING_MODEL_NAME}: {gpu}% on the GPU, {100 - gpu}% on the CPU")
sys.exit(0 if share == 0 else 1)
EOF
    )
}

# --------------------------------------------------------------------------
# The torch-era wheels. Unwanted means in the family and reached by nothing
# outside it, extras and markers honoured. nvidia-* wheels share install paths,
# so if any of them is still needed none is removed. And torch only goes once
# the chunker's tokenizer loads with torch hidden: transformers is still a
# dependency, and a reinstall after a wrong guess is 2 GB.
# --------------------------------------------------------------------------
venv_cleanup() {   # $1 = "check" to only report
    ( cd "$ROOT" && "$PY" - "$1" <<'EOF'
import os
import subprocess
import sys
from importlib.metadata import distributions

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

check_only = sys.argv[1] == "check"
FAMILY = {"torch", "torchvision", "torchaudio", "triton", "sentence-transformers"}


def in_family(name):
    return name in FAMILY or name.startswith(("nvidia-", "cuda-"))


dists = {}
for d in distributions():
    name = d.metadata["Name"]
    if name:
        dists.setdefault(canonicalize_name(name), d)

keep, seen = set(), set()
stack = [(n, frozenset()) for n in dists if not in_family(n)]
while stack:
    name, extras = stack.pop()
    if (name, extras) in seen or name not in dists:
        continue
    seen.add((name, extras))
    keep.add(name)
    for spec in dists[name].requires or []:
        req = Requirement(spec)
        envs = [{"extra": e} for e in extras] or [{"extra": ""}]
        if req.marker is None or any(req.marker.evaluate(env) for env in envs):
            stack.append((canonicalize_name(req.name), frozenset(req.extras)))

unwanted = sorted(n for n in dists if in_family(n) and n not in keep)
needed = sorted(n for n in dists if in_family(n) and n in keep)
if needed:
    print(f"  kept, because something else in the venv needs them: {' '.join(needed)}")
if any(n.startswith(("nvidia-", "cuda-")) for n in needed):
    unwanted = [n for n in unwanted if not n.startswith(("nvidia-", "cuda-"))]
if not unwanted:
    print("  no torch-era packages in the venv")
    sys.exit(0)

size = 0
for n in unwanted:
    for p in dists[n].files or []:
        f = dists[n].locate_file(p)
        if os.path.isfile(f):
            size += os.path.getsize(f)
print(f"  unwanted: {' '.join(unwanted)} (~{size / 2**30:.1f} GB)")
if check_only:
    sys.exit(0)

hidden = ["torch", "torchvision", "torchaudio", "triton", "sentence_transformers"]
# Through the property the chunker itself calls, so this tests the real load
# path (cache-first fallback included) and not a second spelling of it.
tokenizer_check = (
    "import sys\n"
    f"for m in {hidden!r}: sys.modules[m] = None\n"
    "from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME, OllamaEmbedder\n"
    "tok = OllamaEmbedder(EMBEDDING_MODEL_NAME).tokenizer\n"
    "assert tok('a passage to cut')['input_ids']\n"
)
check = subprocess.run([sys.executable, "-c", tokenizer_check], capture_output=True, text=True)
if check.returncode != 0:
    print("  NOT removed: the embedding tokenizer does not load without torch here:")
    print("  " + (check.stderr.strip().splitlines() or ["(no output)"])[-1])
    sys.exit(1)
print("  the embedding tokenizer loads with torch hidden")

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *unwanted], check=True,
               stdout=subprocess.DEVNULL)
print(f"  removed {len(unwanted)} package(s)")
EOF
    )
}

console_state() {   # prints: down | idle | running
    if ! curl -fsS -m 3 "$CONSOLE_URL/api/status" >/dev/null 2>&1; then
        echo down
        return
    fi
    # Captured, not piped into grep -q: under pipefail an early grep exit can
    # fail curl and read a running console as an idle one.
    local reply
    reply="$(curl -fsS -m 5 -X POST -H 'Content-Type: application/json' \
        -d '{"method":"run_progress","params":{}}' "$CONSOLE_URL/rpc" 2>/dev/null || true)"
    if [[ "$reply" =~ \"running\":\ *true ]]; then echo running; else echo idle; fi
}

# ==========================================================================

say "Driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- no NVIDIA driver, so there is nothing CUDA can use"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | sed 's/^/  /'
# Omarchy chooses the driver by GSP firmware, and its own helpers say which
# side of that line these cards sit on, so this script and Omarchy never
# disagree about a card. Plain Arch has no helpers and no opinion to check.
driver_pkgs=()
if command -v omarchy-hw-nvidia-without-gsp >/dev/null && omarchy-hw-nvidia-without-gsp; then
    driver_pkgs=(nvidia-580xx-dkms nvidia-580xx-utils)
elif command -v omarchy-hw-nvidia-gsp >/dev/null && omarchy-hw-nvidia-gsp; then
    driver_pkgs=(nvidia-open-dkms nvidia-utils)
fi
for pkg in "${driver_pkgs[@]}"; do
    installed="$(pacman -Q "$pkg" 2>/dev/null || true)"
    [ -n "$installed" ] || die "$pkg is Omarchy's driver for these cards and is not installed: sudo pacman -S ${driver_pkgs[*]}, then reboot"
    echo "  $installed  (Omarchy's driver for these cards)"
done
[ -e /dev/nvidia-uvm ] || die "nvidia_uvm is not loaded, and CUDA cannot start without it: sudo modprobe nvidia_uvm"
driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk -F. 'NR == 1 {print $1}')"
[ "${driver_major:-0}" -ge 525 ] || die "driver $driver_major is too old for CUDA 12 (it needs 525 or newer)"
echo "  nvidia_uvm loaded; driver $driver_major runs CUDA 12"

# Below 7.5 is what Arch's CUDA 13 build cannot drive.
old_cards="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader \
    | awk -F', *' '$2 ~ /^[0-9]+\.[0-9]+$/ { split($2, v, "."); if (v[1] * 10 + v[2] < 75) printf "%s%s (%s)", (n++ ? ", " : ""), $1, $2 }' \
    || true)"

say "Ollama"
command -v ollama >/dev/null || die "ollama is not installed: sudo pacman -S ollama"
# The same version as the packaged CLI, so client and daemon agree. After
# pacman upgrades ollama, a re-run fetches the matching build.
version="$(pacman -Q ollama 2>/dev/null | awk '{print $2}' | sed -E 's/^[0-9]+://; s/-[^-]*$//' || true)"
[[ "$version" =~ ^[0-9]+(\.[0-9]+)+$ ]] || die "cannot tell which Ollama version is packaged ('$version')"
[ "$(uname -m)" = x86_64 ] || die "Ollama's CUDA 12 build is only fetched for x86_64"
build_dir="/opt/ollama/$version"
dropin=/etc/systemd/system/ollama.service.d/cuda12.conf
cache="${XDG_CACHE_HOME:-$HOME/.cache}/ambiguity/ollama/v$version"
release="https://github.com/ollama/ollama/releases/download/v$version"
asset=ollama-linux-amd64.tar.zst
dropin_text="$(printf '%s\n' \
    "# Written by cuda-embed-ollama.sh: Ollama's own $version build, whose cuda_v12 runner" \
    "# drives the cards Arch's CUDA 13 build skips. Delete this file and restart" \
    "# ollama.service to go back to the packaged build." \
    "[Service]" \
    "ExecStart=" \
    "ExecStart=$build_dir/bin/ollama serve")"

# What the service is running, read off its main process rather than its unit:
# a daemon-reload changes the unit without touching the process that serves.
serving() {
    local pid
    pid="$(systemctl show ollama.service -p MainPID --value 2>/dev/null || true)"
    [ -n "$pid" ] && [ "$pid" != 0 ] && ps -o args= -p "$pid" 2>/dev/null || true
}
on_build() { [[ "$(serving)" == "$build_dir/bin/ollama "* ]]; }
have_build() { [ -d "$build_dir/lib/ollama/cuda_v12" ]; }
dropin_written() { [ "$(cat "$dropin" 2>/dev/null || true)" = "$dropin_text" ]; }
daemon_up() { curl -s -m 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

running_now="$(serving)"
echo "  packaged: ollama $version; ollama.service runs: ${running_now:-(nothing)}"
if [ -n "$old_cards" ]; then
    echo "  $old_cards: below compute 7.5, so they need the CUDA 12 build"
else
    echo "  every card is Turing or newer, which Arch's ollama-cuda drives: no CUDA 12 build needed"
fi

rc=0
if daemon_up; then probe || rc=$?; else echo "  the daemon is not answering at $OLLAMA_URL"; rc=2; fi
need_install=0
if [ -n "$old_cards" ] && ! { have_build && dropin_written && on_build && [ "$rc" -eq 0 ]; }; then
    need_install=1
fi

if [ "$MODE" = check ]; then
    say "Leftovers"
    venv_cleanup check
    if on_build; then
        for pkg in ollama-vulkan ollama-cuda; do
            if pacman -Q "$pkg" >/dev/null 2>&1; then echo "  unused beside the CUDA 12 build: $pkg"; fi
        done
    fi
    if [ "$need_install" -eq 1 ]; then
        echo -e "\nBROKEN: run without --check to install Ollama's CUDA 12 build."
        exit 1
    fi
    if [ "$rc" -ne 0 ]; then
        echo -e "\nNOT ON THE CARDS: see the probe above."
        exit 1
    fi
    echo -e "\nOK: the model is 100% on the GPU, 0% on the CPU."
    exit 0
fi

if [ "$need_install" -eq 1 ]; then
    state="$(console_state)"
    [ "$state" != running ] || die "an agent run is in flight in the console -- stop it or let it finish, then re-run this"

    if ! have_build; then
        say "Installing Ollama's own $version build (1.4 GB download, once)"
        mkdir -p "$cache"
        # The release's own checksum list. A tag's list never changes, so a
        # cached copy serves a re-run without the network.
        [ -s "$cache/sha256sum.txt" ] \
            || curl -fsSL -m 60 -o "$cache/sha256sum.txt" "$release/sha256sum.txt" \
            || rm -f "$cache/sha256sum.txt"
        sum="$(awk -v a="$asset" '$2 == a || $2 == "./" a {print $1}' "$cache/sha256sum.txt" 2>/dev/null || true)"
        [ -n "$sum" ] || die "no checksum for $asset in $release/sha256sum.txt"
        if [ "$(sha256sum "$cache/$asset" 2>/dev/null | cut -d' ' -f1)" = "$sum" ]; then
            echo "  using the verified download in $cache"
        else
            progress=(-sS)
            [ -t 1 ] && progress=(--progress-bar)
            curl -fL --retry 3 "${progress[@]}" -o "$cache/$asset.part" "$release/$asset" \
                || { rm -f "$cache/$asset.part"; die "could not download $release/$asset"; }
            if [ "$(sha256sum "$cache/$asset.part" | cut -d' ' -f1)" != "$sum" ]; then
                rm -f "$cache/$asset.part"
                die "the download does not match the release's checksum"
            fi
            mv "$cache/$asset.part" "$cache/$asset"
            echo "  downloaded and verified against the release's sha256sum.txt"
        fi
        sudo rm -rf "$build_dir.part"
        sudo mkdir -p "$build_dir.part"
        sudo tar --zstd --no-same-owner -xf "$cache/$asset" -C "$build_dir.part" \
            || { sudo rm -rf "$build_dir.part"; die "could not unpack the build into $build_dir"; }
        if [ ! -d "$build_dir.part/lib/ollama/cuda_v12" ]; then
            sudo rm -rf "$build_dir.part"
            die "Ollama $version's release carries no cuda_v12 runner, so nothing here drives $old_cards"
        fi
        sudo rm -rf "$build_dir"
        sudo mv "$build_dir.part" "$build_dir"
        echo "  unpacked into $build_dir"
    fi

    say "Pointing ollama.service at it"
    if ! dropin_written; then
        sudo mkdir -p "${dropin%/*}"
        printf '%s\n' "$dropin_text" | sudo tee "$dropin" >/dev/null
        echo "  wrote $dropin"
    fi
    restarted_at="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "  restarting ollama.service (every model unloads; a request in flight is cut off)"
    sudo systemctl daemon-reload || die "systemctl daemon-reload failed"
    sudo systemctl restart ollama.service || die "ollama.service did not restart; see: systemctl status ollama"
    for _ in $(seq 1 40); do daemon_up && break; sleep 0.5; done
    on_build || die "ollama.service is not running $build_dir/bin/ollama after the restart; see: systemctl status ollama"
    echo "  ollama.service runs: $(serving)"
    journalctl -u ollama.service --since "$restarted_at" --no-pager 2>/dev/null \
        | grep -oE 'library=[A-Za-z]+ compute=[0-9.]+ name=[A-Za-z0-9]+ description="[^"]*"' \
        | sed 's/^/  found: /' || true

    say "Verifying"
    rc=0
    probe || rc=$?
    if [ "$rc" -eq 2 ]; then
        die "the daemon could not embed with the CUDA 12 build; see the probe above"
    elif [ "$rc" -ne 0 ]; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
            | sed 's/^/  on the cards: /' || true
        die "the model is not wholly on the cards with the CUDA 12 build; see above for what else holds them"
    fi
fi

if [ "$CLEANUP" -eq 1 ]; then
    say "Removing what nothing needs any more"
    if on_build; then
        for old in /opt/ollama/*; do
            [ -d "$old" ] && [ ! -L "$old" ] && [ "$old" != "$build_dir" ] || continue
            sudo rm -rf -- "$old"
            echo "  removed the older build $old"
        done
        unused=()
        for pkg in ollama-vulkan ollama-cuda; do
            if pacman -Q "$pkg" >/dev/null 2>&1; then unused+=("$pkg"); fi
        done
        if [ "${#unused[@]}" -gt 0 ]; then
            # -s takes the CUDA 13 toolkit with ollama-cuda when nothing else
            # needs it; pacman lists everything before it removes anything.
            pacman_flags=(-Rns)
            [ "$ASSUME_YES" -eq 1 ] && pacman_flags+=(--noconfirm)
            sudo pacman "${pacman_flags[@]}" "${unused[@]}" || echo "  pacman removed nothing; still installed: ${unused[*]}"
        else
            echo "  no Arch Ollama GPU packages beside the CUDA 12 build"
        fi
    fi
    venv_cleanup fix || die "the venv cleanup stopped; nothing past the message above was removed"
    "$PY" -m pip check | sed 's/^/  /' || die "pip check found broken requirements after the cleanup"
fi

if [ -n "$old_cards" ]; then
    echo -e "\nDONE: Ollama's CUDA 12 build holds qwen3-embedding 100% on the GPU, 0% on the CPU, across $old_cards."
else
    echo -e "\nDONE: nothing on this machine needs the CUDA 12 build."
fi
