#!/usr/bin/env bash
# Claude Code on this machine, set up to drive the console in a real browser.
#
# The console needs none of this: it, its seats and scripts/browser_agent.py
# run with no account at all. What this adds is a Claude Code session here that
# can open the console in a browser itself -- through the Playwright MCP server,
# which needs no sign-in, or through Claude in Chrome, which needs a claude.ai
# one. Shared by install.sh and docker/install.sh.
#
# `install` sees to four things, each only when it is missing, so a re-run on a
# finished machine changes nothing:
#   1. the Claude Code CLI, from Anthropic's native installer
#      (curl -fsSL https://claude.ai/install.sh | bash), which needs no sudo and
#      puts claude in ~/.local/bin. At a terminal it asks first; --yes installs
#      it without asking. ~/.local/bin is put on PATH for this run only: no
#      shell rc file is edited, so a PATH that lacks it is named, not changed.
#   2. the sign-in. `claude auth status` is read with ANTHROPIC_API_KEY,
#      ANTHROPIC_AUTH_TOKEN and CLAUDE_CODE_OAUTH_TOKEN removed from its
#      environment (install.sh exports .env, which may hold the first), and only
#      a claude.ai sign-in counts: Claude in Chrome is off under a key or a
#      token. At a terminal it runs `claude auth login --claudeai` -- one
#      consent page in the browser -- with `git config user.email` filled in
#      unless that is a GitHub noreply address. --yes never signs in, and
#      neither does a run with no terminal: either leaves a note instead.
#   3. the Playwright MCP server, @playwright/mcp at a pinned version: fetched
#      into npx's cache, then registered with Claude Code for this checkout
#      only -- local scope, in Claude Code's own settings, never committed:
#        claude mcp add --scope local playwright -- npx -y @playwright/mcp@0.0.83
#          --executable-path CHROMIUM --isolated
#          --output-dir CHECKOUT/reports/diagnostics/playwright-mcp
#      and --no-sandbox as root, where Chromium cannot sandbox itself.
#      A server already registered under that name is never replaced: one with
#      other arguments is named, with the commands that would replace it. When
#      the project's venv is here, scripts/browser_agent.py mcp-check then starts
#      the server and drives a page with it.
#   4. Claude in Chrome: whether its native-messaging host is installed. The
#      extension is one click in the Chrome Web Store, which no script can make,
#      so a missing one is a note with the link.
#
# `check` prints one line for each of the four and exits 1 when any is missing.
#
# Usage:
#   scripts/claude_tools.sh install [--yes]
#   scripts/claude_tools.sh check
#
#   install exits 0 when what it set out to do is in place and 1 when something
#   it tried failed; a note -- what only you can do -- is not a failure. With
#   CLAUDE_TOOLS_NOTES naming a file, each note is also appended to it, one per
#   line, which is how the installers carry them into their summaries.

set -uo pipefail

SELF="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 2
# Physical, because Claude Code keys a local-scope server by the project's path
# and the MCP server is handed the output directory as given.
ROOT="$(pwd -P)"

INSTALLER_URL=https://claude.ai/install.sh
# Pinned: the server drives a Chromium whose version it does not choose, and an
# unpinned npx fetches whatever was published last.
MCP_PACKAGE=@playwright/mcp@0.0.83
MCP_OUTPUT_DIR="$ROOT/reports/diagnostics/playwright-mcp"
CHROME_STORE=https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn
CHROME_MANIFEST=com.anthropic.claude_code_browser_extension.json
# The config directories Claude Code writes that manifest under on Linux.
CHROME_DIRS=(chromium google-chrome BraveSoftware/Brave-Browser)
# Each of these turns Claude in Chrome off, and any of them may arrive in this
# environment from .env; every claude call below runs without them.
KEY_VARS=(ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN)

# Overridable only so the tests can stand in for them.
SYSTEM_CHROMIUM="${CLAUDE_TOOLS_SYSTEM_CHROMIUM:-/usr/lib/chromium/chromium}"
PY="${CLAUDE_TOOLS_PYTHON:-$ROOT/.venv/bin/python}"

FAILED=0
ok()    { echo "  ✓ $1"; }
cross() { echo "  ✗ $1"; }
fail()  { cross "$1"; FAILED=1; }
note() {
    echo "  ! $1"
    if [ -n "${CLAUDE_TOOLS_NOTES:-}" ]; then printf '%s\n' "$1" >>"$CLAUDE_TOOLS_NOTES"; fi
}

ASSUME_YES=0
# A terminal to answer on, unless --yes said never to ask. CLAUDE_TOOLS_INTERACTIVE
# (0 or 1) stands in for the terminal, so the tests can drive both answers.
interactive() {
    [ "$ASSUME_YES" -eq 0 ] || return 1
    case "${CLAUDE_TOOLS_INTERACTIVE:-}" in
        1) return 0 ;;
        0) return 1 ;;
        *) [ -t 0 ] ;;
    esac
}

unset_keys=()
for var in "${KEY_VARS[@]}"; do unset_keys+=(-u "$var"); done
claude_run() { env "${unset_keys[@]}" claude "$@"; }
# Bounded, for the calls nobody answers: `mcp get` health-checks the server.
# Their stdin is /dev/null, never the terminal: timeout puts claude in a process
# group of its own, and a claude that sets up a terminal from there is stopped
# (SIGTTOU) until the bound ends it, having printed nothing -- at a terminal,
# every sign-in read as unknown and check hung. -k kills one still stopped.
claude_bounded() {
    local secs="$1"
    shift
    timeout -k 10 "$secs" env "${unset_keys[@]}" claude "$@" </dev/null
}

# Where the native installer puts claude. Added for this run only, and last, so
# a claude already on PATH still wins.
LOCAL_BIN="$HOME/.local/bin"
case ":$PATH:" in
    *":$LOCAL_BIN:"*) LOCAL_BIN_ON_PATH=1 ;;
    *) LOCAL_BIN_ON_PATH=0; export PATH="$PATH:$LOCAL_BIN" ;;
esac

have_claude() { command -v claude >/dev/null 2>&1; }

claude_version() { claude_bounded 30 --version 2>/dev/null | head -n 1 | sed 's/ (Claude Code)$//'; }

# The authMethod `claude auth status` reports (JSON by default): none,
# claude.ai, api_key, oauth_token, ... -- empty when it could not be read.
auth_method() {
    claude_bounded 30 auth status 2>/dev/null \
        | sed -n 's/.*"authMethod"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1
}

# The browser the MCP server drives. The binary itself comes before Arch's
# /usr/bin/chromium launcher, which adds ~/.config/chromium-flags.conf (on
# Omarchy, an extension to load) to every start; BROWSER_AGENT_CHROMIUM, which
# the browser agent reads too, comes before both. Playwright's own build comes
# last: on ./install.sh --no-system, which installs no chromium and fetches that
# build for the browser agent instead, it is the only browser there is.
resolve_browser() {
    local name
    if [ -n "${BROWSER_AGENT_CHROMIUM:-}" ]; then
        [ -x "$BROWSER_AGENT_CHROMIUM" ] && echo "$BROWSER_AGENT_CHROMIUM"
        return
    fi
    if [ -x "$SYSTEM_CHROMIUM" ]; then echo "$SYSTEM_CHROMIUM"; return; fi
    for name in chromium chromium-browser google-chrome-stable; do
        command -v "$name" 2>/dev/null && return
    done
    playwright_chromium
}

# The newest chromium-<revision>/chrome-linux*/chrome under PLAYWRIGHT_BROWSERS_PATH,
# else under Playwright's cache: where scripts/browser_agent.py looks too.
playwright_chromium() {
    local cache build revision candidate best="" best_revision=-1
    for cache in ${PLAYWRIGHT_BROWSERS_PATH:+"$PLAYWRIGHT_BROWSERS_PATH"} "$HOME/.cache/ms-playwright"; do
        for build in "$cache"/chromium-*; do
            revision="${build##*/chromium-}"
            [[ "$revision" =~ ^[0-9]+$ ]] || continue
            for candidate in "$build"/chrome-linux*/chrome; do
                if [ -x "$candidate" ] && [ "$revision" -gt "$best_revision" ]; then
                    best="$candidate" best_revision="$revision"
                fi
            done
        done
        if [ -n "$best" ]; then echo "$best"; return; fi
    done
}

mcp_args_for() {  # browser
    MCP_ARGS=(-y "$MCP_PACKAGE" --executable-path "$1" --isolated --output-dir "$MCP_OUTPUT_DIR")
    # Chromium cannot sandbox itself as root (a container, a CI box), and the
    # server keeps the sandbox on unless told: as root, every page it opened
    # failed. One of the arguments, so a registration made as root reads as ours.
    if [ "$(id -u 2>/dev/null)" = 0 ]; then MCP_ARGS+=(--no-sandbox); fi
}

# What Claude Code has registered as `playwright` for this checkout: sets
# MCP_STATE to absent, ours (exactly this checkout's arguments), other or
# unknown, with MCP_SEEN (its command line) and MCP_REMOVE (the command Claude
# Code itself gives for removing it).
read_registration() {  # expected browser, may be empty
    local out command args
    MCP_SEEN="" MCP_REMOVE=""
    out="$(claude_bounded 120 mcp get playwright 2>&1)"
    if grep -q 'No MCP server named' <<<"$out"; then
        MCP_STATE=absent
        return
    fi
    command="$(sed -n 's/^  Command: //p' <<<"$out" | head -n 1)"
    args="$(sed -n 's/^  Args: //p' <<<"$out" | head -n 1)"
    MCP_REMOVE="$(sed -n 's/^To remove this server, run: //p' <<<"$out" | head -n 1)"
    if [ -z "$command" ]; then
        MCP_STATE=unknown
        MCP_SEEN="$(grep -m 1 . <<<"$out")"
        return
    fi
    MCP_SEEN="$command $args"
    MCP_STATE=other
    if [ -n "$1" ]; then
        mcp_args_for "$1"
        [ "$command" = npx ] && [ "$args" = "${MCP_ARGS[*]}" ] && MCP_STATE=ours
    fi
}

chrome_manifest() {
    local dir
    for dir in "${CHROME_DIRS[@]}"; do
        if [ -f "$HOME/.config/$dir/NativeMessagingHosts/$CHROME_MANIFEST" ]; then
            echo "$HOME/.config/$dir/NativeMessagingHosts/$CHROME_MANIFEST"
            return 0
        fi
    done
    return 1
}

chrome_note() {
    note "Claude in Chrome is not set up: add the extension from $CHROME_STORE (one click, which no script can make), then run claude --chrome and /chrome"
}

install_cli() {
    local answer="" path real seen=() distinct=()
    if ! have_claude; then
        if [ "$ASSUME_YES" -eq 1 ]; then
            answer=y
        elif interactive; then
            read -r -p "  Claude Code is not installed. Install it with Anthropic's native installer (no sudo)? [y/N] " answer \
                || answer=""
        fi
        case "$answer" in
            y|Y|yes|Yes|YES)
                echo "  installing Claude Code: curl -fsSL $INSTALLER_URL | bash"
                if ! curl -fsSL "$INSTALLER_URL" | bash; then
                    fail "the Claude Code installer failed; see above (scripts/network_check.sh claude says whether its hosts answer)"
                    return 1
                fi
                hash -r
                if ! have_claude; then
                    fail "the Claude Code installer finished, but no claude is on PATH or in $LOCAL_BIN"
                    return 1
                fi
                ;;
            *)
                if interactive; then
                    note "Claude Code was not installed, as asked; scripts/claude_tools.sh install adds it any time"
                else
                    note "Claude Code is not installed, and without a terminal nothing asked: scripts/claude_tools.sh install --yes"
                fi
                return 1
                ;;
        esac
    fi
    ok "Claude Code $(claude_version) ($(command -v claude))"

    if [ "$LOCAL_BIN_ON_PATH" -eq 0 ] && [ "$(command -v claude)" = "$LOCAL_BIN/claude" ]; then
        note "claude is in $LOCAL_BIN, which your shell's PATH does not include: add it in your shell's own config (nothing here edits one)"
    fi
    # A mise shim earlier on PATH runs mise's claude instead of the native one,
    # and the two update separately; resolved, so one file reached twice is one.
    while IFS= read -r path; do
        real="$(readlink -f "$path" 2>/dev/null || echo "$path")"
        [[ " ${seen[*]} " == *" $real "* ]] && continue
        seen+=("$real")
        distinct+=("$path")
    done < <(type -ap claude)
    if [ "${#distinct[@]}" -gt 1 ]; then
        note "more than one claude is on PATH (${distinct[*]}); the first runs, and a mise shim there shadows the native install"
    fi
    return 0
}

sign_in() {
    local method email login=()
    method="$(auth_method)"
    if [ "$method" = claude.ai ]; then
        ok "signed in with claude.ai"
        return 0
    fi
    if ! interactive; then
        note "Claude Code is not signed in with claude.ai (${method:-unknown}), which Claude in Chrome needs: claude auth login --claudeai"
        return 0
    fi
    echo "  Claude in Chrome needs a claude.ai sign-in (now: ${method:-unknown}); one consent page opens in your browser"
    login=(auth login --claudeai)
    email="$(git config user.email 2>/dev/null || true)"
    case "$email" in
        ''|*@users.noreply.github.com) ;;
        *) login+=(--email "$email") ;;
    esac
    claude_run "${login[@]}" || true
    method="$(auth_method)"
    if [ "$method" = claude.ai ]; then
        ok "signed in with claude.ai"
    else
        fail "still not signed in with claude.ai (${method:-unknown}): claude auth login --claudeai"
    fi
}

install_mcp() {
    local browser npx_path add_out check_out check_status fetch
    if ! npx_path="$(command -v npx)"; then
        note "npx is missing, so the Playwright MCP server cannot run: sudo pacman -S nodejs npm"
        return 0
    fi
    browser="$(resolve_browser)"
    if [ -z "$browser" ]; then
        if [ -n "${BROWSER_AGENT_CHROMIUM:-}" ]; then
            note "BROWSER_AGENT_CHROMIUM names $BROWSER_AGENT_CHROMIUM, which is not a program, so the Playwright MCP server was not set up"
        else
            # pacman is what --no-system rules out; Playwright's own build
            # needs no sudo, wherever the venv has Playwright to fetch it.
            fetch="sudo pacman -S chromium"
            if [ -x "$PY" ] && "$PY" -c "import playwright" >/dev/null 2>&1; then
                fetch=".venv/bin/python -m playwright install chromium (no sudo), or $fetch"
            fi
            note "no Chromium for the Playwright MCP server to drive: $fetch, or set BROWSER_AGENT_CHROMIUM; then scripts/claude_tools.sh install"
        fi
        return 0
    fi

    # Fetched now, so the first session that starts the server is not also a
    # download that its MCP start-up timeout may cut short.
    if timeout 300 npx -y "$MCP_PACKAGE" --help >/dev/null 2>&1; then
        ok "$MCP_PACKAGE is in npx's cache"
    else
        note "could not fetch $MCP_PACKAGE (scripts/network_check.sh npm says whether the registry answers); npx tries again when the server first starts"
    fi

    read_registration "$browser"
    mcp_args_for "$browser"
    case "$MCP_STATE" in
        ours)
            ok "the Playwright MCP server is registered for this checkout ($browser)"
            ;;
        absent)
            if add_out="$(claude_bounded 60 mcp add --scope local playwright -- npx "${MCP_ARGS[@]}" 2>&1)"; then
                ok "registered the Playwright MCP server for this checkout ($browser)"
            else
                fail "could not register the Playwright MCP server: $(grep -m 1 . <<<"$add_out")"
                return 1
            fi
            ;;
        other)
            note "a playwright MCP server is already registered with other arguments ($MCP_SEEN) and was left as it is; to use this checkout's: ${MCP_REMOVE:-claude mcp remove playwright} && claude mcp add --scope local playwright -- npx ${MCP_ARGS[*]}"
            ;;
        *)
            fail "could not read Claude Code's MCP servers: claude mcp get playwright said: ${MCP_SEEN:-nothing}"
            return 1
            ;;
    esac

    # A registration is a line in a settings file; starting the server and
    # driving a page with it is what says it works. That needs the project's
    # venv, which a Docker-only install does not make.
    if [ ! -x "$PY" ] || [ ! -f scripts/browser_agent.py ]; then
        echo "  (no project venv here, so the server was registered but not test-started)"
        return 0
    fi
    check_status=0
    check_out="$(timeout 600 "$PY" scripts/browser_agent.py mcp-check --npx "$npx_path" --chromium "$browser" 2>&1)" \
        || check_status=$?
    case "$check_status" in
        0) ok "the Playwright MCP server starts and drives $browser" ;;
        2) note "the Playwright MCP server could not be test-started here (npx or the browser is missing): .venv/bin/python scripts/browser_agent.py mcp-check" ;;
        *)
            tail -n 15 <<<"$check_out" | sed 's/^/    /'
            fail "the Playwright MCP server did not pass its check: .venv/bin/python scripts/browser_agent.py mcp-check"
            ;;
    esac
}

install_tools() {
    local manifest
    if install_cli; then
        sign_in
        install_mcp
    fi
    if manifest="$(chrome_manifest)"; then
        ok "Claude in Chrome's native host is installed ($manifest)"
    else
        chrome_note
    fi
    return "$FAILED"
}

check() {
    local bad=0 method browser manifest var exported=()
    # This runs in the user's own shell, so what is set here is what a claude
    # started from it inherits: any of these replaces the claude.ai sign-in,
    # and Claude in Chrome is off under it.
    for var in "${KEY_VARS[@]}"; do
        if [ -n "${!var:-}" ]; then exported+=("$var"); fi
    done
    if [ "${#exported[@]}" -gt 0 ]; then
        cross "${exported[*]} set in this shell overrides the claude.ai sign-in (Claude in Chrome is off under it): remove it from your shell's config"
        bad=1
    fi
    if have_claude; then
        ok "Claude Code $(claude_version) ($(command -v claude))"
        method="$(auth_method)"
        if [ "$method" = claude.ai ]; then
            ok "signed in with claude.ai"
        else
            cross "not signed in with claude.ai (${method:-unknown}): claude auth login --claudeai"; bad=1
        fi
        browser="$(resolve_browser)"
        read_registration "$browser"
        case "$MCP_STATE" in
            ours) ok "the Playwright MCP server is registered for this checkout ($browser)" ;;
            other) ok "a playwright MCP server is registered, with other arguments than this checkout's: $MCP_SEEN" ;;
            absent) cross "no Playwright MCP server is registered for this checkout: scripts/claude_tools.sh install"; bad=1 ;;
            *) cross "could not read Claude Code's MCP servers: ${MCP_SEEN:-nothing}"; bad=1 ;;
        esac
    else
        cross "Claude Code is not installed: scripts/claude_tools.sh install"
        cross "not signed in with claude.ai (no Claude Code)"
        cross "no Playwright MCP server is registered (no Claude Code)"
        bad=1
    fi
    if manifest="$(chrome_manifest)"; then
        ok "Claude in Chrome's native host is installed ($manifest)"
    else
        cross "Claude in Chrome's native host is not installed: the extension from $CHROME_STORE, then claude --chrome and /chrome"
        bad=1
    fi
    return "$bad"
}

action="${1:-}"
[ $# -gt 0 ] && shift
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        *) echo "claude_tools: unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

case "$action" in
    install) install_tools ;;
    check) check ;;
    -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$SELF" ;;
    *) echo "usage: $0 install [--yes] | check (try --help)" >&2; exit 2 ;;
esac
