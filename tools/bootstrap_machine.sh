#!/usr/bin/env bash
# bootstrap_machine.sh — first-time setup for a Harmony host. Idempotent:
# re-running is safe.
#
# Two roles (--role, default control):
#   control  Linux operator host — device I/O + EEG/LSL decoder + Qt control
#            panel + CPU perception. Runs the full setup below; env =
#            environment.yml (conda env 'lsl').
#   server   GPU perception host — perception stack only. env =
#            environment.server.yml (conda env 'harmony-server'); skips the
#            control-only steps (subject skeleton, robot/EEG checklist).
#            environment.server.yml is cross-platform; Windows servers can't
#            run this bash script, so create the env directly:
#            conda env create -f environment.server.yml
#
# control steps (perception was folded in-tree in WS3, so there is no longer a
# sibling harmony_vlm repo to clone):
#   1. Create config_local.py from config_local.example.py and prompt for the
#      must-set keys (WORKING_DIR, DATA_DIR).
#   2. Materialize a CurrentStudy/sub-PILOT007 BIDS-ish skeleton with one
#      example session (ses-S001) plus models/ and training_data/.
#   3. Create the conda env (skip if present).
#   4. Wire up the repo-local pre-commit hook (.githooks/pre-commit).
#   5. Optionally run tools/setup_linux_ufw.sh (asks for sudo).
#
# Then prints a checklist of irreducibly manual steps (eegoSports,
# LabRecorder, Pupil Labs Companion, dialout group, robot networking).
#
# Usage:
#   bash tools/bootstrap_machine.sh [flags]
#
# Flags (all optional — script is interactive otherwise):
#   --role=ROLE           control (default) or server
#   --data-dir=PATH       Skip the DATA_DIR prompt (control only)
#   --no-conda            Skip conda env creation
#   --no-firewall         Skip the ufw step
#   --non-interactive     Fail if any unfilled key has no flag/default
#   -h, --help            Print this header

set -euo pipefail

# --- helpers ---------------------------------------------------------------

c_cyan='\033[36m'; c_green='\033[32m'; c_yellow='\033[33m'; c_red='\033[31m'; c_reset='\033[0m'
section() { printf "\n${c_cyan}== %s ==${c_reset}\n" "$1"; }
ok()      { printf "${c_green}[ok]${c_reset} %s\n" "$1"; }
skip()    { printf "${c_yellow}[skip]${c_reset} %s\n" "$1"; }
warn()    { printf "${c_yellow}[warn]${c_reset} %s\n" "$1"; }
err()     { printf "${c_red}[err]${c_reset} %s\n" "$1" >&2; }

prompt_default() {
    # prompt_default <var_name> <prompt> <default>
    local __var="$1" __prompt="$2" __default="$3" __ans
    if [ "$NON_INTERACTIVE" = "true" ]; then
        printf -v "$__var" "%s" "$__default"
        return
    fi
    read -r -p "$__prompt [$__default]: " __ans
    printf -v "$__var" "%s" "${__ans:-$__default}"
}

# --- arg parsing -----------------------------------------------------------

ROLE="control"
DATA_DIR_ARG=""
DO_CONDA=true
DO_FIREWALL=true
NON_INTERACTIVE=false

for arg in "$@"; do
    case "$arg" in
        --role=*)          ROLE="${arg#*=}" ;;
        --data-dir=*)      DATA_DIR_ARG="${arg#*=}" ;;
        --no-conda)        DO_CONDA=false ;;
        --no-firewall)     DO_FIREWALL=false ;;
        --non-interactive) NON_INTERACTIVE=true ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) err "unknown flag: $arg"; exit 2 ;;
    esac
done

if [ "$ROLE" != "control" ] && [ "$ROLE" != "server" ]; then
    err "--role must be 'control' or 'server' (got: $ROLE)"; exit 2
fi

# --- sanity ----------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f config_local.example.py ] || [ ! -f environment.yml ]; then
    err "must run from a Harmony repo root (config_local.example.py / environment.yml not found)"
    exit 1
fi

if [ "$(uname -s)" != "Linux" ]; then
    warn "this script targets the Linux operator host; running on $(uname -s) is unsupported"
fi

# --- step 1: config_local.py -----------------------------------------------

section "config_local.py"

if [ -f config_local.py ]; then
    skip "config_local.py already exists — leaving it alone (edit by hand if you need to change values)"
elif [ "$ROLE" = "server" ]; then
    cp config_local.example.py config_local.py
    ok "created config_local.py from example"
    warn "server role: edit config_local.py for the perception keys — PERCEPTION_MODELS_DIR, GOOGLE_API_KEY, and the relay/UDP hosts (FRAME_RELAY_*, VLM_*)"
else
    cp config_local.example.py config_local.py
    ok "created config_local.py from example"

    DEFAULT_WORKING="$REPO_ROOT/"
    DEFAULT_DATA="$HOME/Documents/CurrentStudy"

    prompt_default WORKING_DIR  "WORKING_DIR (repo root)"             "$DEFAULT_WORKING"
    prompt_default DATA_DIR     "DATA_DIR (XDFs / training data)"     "${DATA_DIR_ARG:-$DEFAULT_DATA}"

    # Replace the two placeholder lines in config_local.py. Each example value
    # is unique so anchor on the full assignment.
    python3 - "$WORKING_DIR" "$DATA_DIR" <<'PY'
import sys, pathlib, re
working, data = sys.argv[1:]
p = pathlib.Path("config_local.py")
text = p.read_text()
def replace(text, key, val):
    pat = re.compile(rf'^{key}\s*=.*$', re.M)
    return pat.sub(f'{key} = {val!r}', text, count=1)
text = replace(text, "WORKING_DIR", working)
text = replace(text, "DATA_DIR", data)
p.write_text(text)
PY
    ok "wrote WORKING_DIR / DATA_DIR into config_local.py"
    warn "all other keys keep their example defaults — open config_local.py and edit hosts/ports if your machine isn't single-machine loopback"
fi

# Re-read DATA_DIR from config_local.py so subsequent steps use the real
# on-disk value, not whatever was prompted (covers the already-existed branch).
DATA_DIR="$(python3 -c 'import config_local; print(config_local.DATA_DIR)')"

# --- step 2: CurrentStudy/sub-PILOT007 skeleton ----------------------------

section "CurrentStudy/sub-PILOT007 skeleton"

if [ "$ROLE" = "server" ]; then
    skip "server role — no subject data layout needed"
elif [ -z "$DATA_DIR" ]; then
    warn "DATA_DIR is empty — skipping subject skeleton"
else
    SUB_ROOT="$DATA_DIR/sub-PILOT007"
    SES_DIR="$SUB_ROOT/ses-S001/eeg"
    if [ -d "$SUB_ROOT" ]; then
        skip "$SUB_ROOT already exists"
    else
        mkdir -p "$SES_DIR" "$SUB_ROOT/models" "$SUB_ROOT/training_data"
        cat > "$SES_DIR/README.txt" <<EOF
LabRecorder writes XDF files into this directory. Expected filename:
  sub-PILOT007_ses-S001_task-Default_run-001_eeg.xdf

Real sessions usually carry modifier keywords (OFFLINE/ONLINE, FES/NOFES,
e.g. ses-S001OFFLINE_NOFES, ses-S002ONLINE_FES). The experiment driver
creates those subdirs lazily on first run; this skeleton is just an
example of the layout.

Top-level subject dirs:
  models/         classification_probabilities_*.csv from online runs
  training_data/  XDFs copied here for offline training
EOF
        ok "created $SUB_ROOT (ses-S001 + models + training_data)"
    fi
fi

# --- step 3: conda env -----------------------------------------------------

if [ "$ROLE" = "server" ]; then
    ENV_NAME="harmony-server"; ENV_FILE="environment.server.yml"
else
    ENV_NAME="lsl"; ENV_FILE="environment.yml"
fi

section "conda env '$ENV_NAME'"

if [ "$DO_CONDA" = "false" ]; then
    skip "--no-conda passed"
elif [ ! -f "$ENV_FILE" ]; then
    warn "$ENV_FILE not found — skipping conda env creation"
elif ! command -v conda >/dev/null 2>&1; then
    warn "conda not on PATH — skipping; install Miniconda then re-run with no flags"
elif conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    skip "conda env '$ENV_NAME' already exists"
else
    # Both env files are conda-forge-only by design (avoids the Anaconda
    # defaults-channel commercial ToS). But recent conda (24+/26+) does a ToS
    # pre-flight on the *globally configured* channels before honoring the
    # file, so a default install (channels: [defaults]) aborts here even for a
    # conda-forge env. Warn with the fix rather than silently editing the
    # user's conda config or accepting Anaconda's ToS on their behalf.
    if conda config --show channels 2>/dev/null | grep -qE '^[[:space:]]*-[[:space:]]+defaults[[:space:]]*$'; then
        warn "conda is configured with the 'defaults' channel; conda env create may abort on its ToS gate."
        warn "  This repo is conda-forge-only. To fix (no Anaconda ToS needed):"
        warn "    conda config --system --remove channels defaults && conda config --system --add channels conda-forge && conda config --set channel_priority strict"
    fi
    conda env create -f "$ENV_FILE" -n "$ENV_NAME"
    ok "created conda env '$ENV_NAME'"
fi

# --- step 4: repo-local pre-commit hook ------------------------------------

section "repo pre-commit hook"

if [ ! -d .githooks ]; then
    warn ".githooks/ not found in repo — skipping"
else
    current="$(git config --get core.hooksPath || true)"
    if [ "$current" = ".githooks" ]; then
        skip "core.hooksPath already set to .githooks"
    else
        git config core.hooksPath .githooks
        ok "set git core.hooksPath to .githooks"
    fi
fi

# --- step 5: ufw -----------------------------------------------------------

section "ufw (frame_relay TCP 5591)"

if [ "$DO_FIREWALL" = "false" ]; then
    skip "--no-firewall passed"
elif [ ! -x tools/setup_linux_ufw.sh ] && [ ! -f tools/setup_linux_ufw.sh ]; then
    skip "tools/setup_linux_ufw.sh not found"
else
    if [ "$NON_INTERACTIVE" = "true" ]; then
        skip "non-interactive mode — run \`sudo bash tools/setup_linux_ufw.sh\` manually"
    else
        read -r -p "Run firewall setup now (needs sudo)? [y/N]: " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            sudo bash tools/setup_linux_ufw.sh
            ok "firewall configured"
        else
            skip "user declined — run \`sudo bash tools/setup_linux_ufw.sh\` later"
        fi
    fi
fi

# --- final checklist -------------------------------------------------------

section "Manual steps remaining (script can't do these)"

if [ "$ROLE" = "server" ]; then
cat <<'EOF'
[ ] NVIDIA driver + CUDA runtime installed; `nvidia-smi` lists the GPU
[ ] torch sees CUDA: python -c "import torch; print(torch.cuda.is_available())"
[ ] Perception weights present at PERCEPTION_MODELS_DIR (FastSAM-s.pt, depth_pro.pt)
[ ] GOOGLE_API_KEY set in config_local.py (Gemini intent reasoning)
[ ] Reachable from the control host: relay dial-in (FRAME_RELAY_DIAL_HOST)
    and UDP result push (VLM_* hosts/ports) — open the firewall accordingly
EOF
else
cat <<'EOF'
[ ] eegoSports (Linux build) installed and on PATH; LSL stream visible
[ ] LabRecorder installed
[ ] Pupil Labs Companion app paired with Neon and reachable on the LAN
    (set NEON_COMPANION_HOST in config_local.py, or leave empty for mDNS)
[ ] Add user to 'dialout' group for /dev/ttyACM0 access (Arduino):
        sudo usermod -aG dialout $USER  &&  log out / back in
[ ] sshpass installed (for control_panel.py robot SSH helpers):
        sudo apt install sshpass gnome-terminal
[ ] Robot reachable at the IP/port set in config.UDP_ROBOT
[ ] Edit any non-loopback hosts in config_local.py for cross-machine setups
    (FRAME_RELAY_*, VLM_*, GAZE_* — see config_local.example.py for guidance)
EOF
fi

ok "bootstrap complete"
