#!/usr/bin/env bash
# bootstrap_machine.sh — first-time setup for a Harmony control computer
# (Linux operator host). Idempotent: re-running is safe.
#
# Performs the deterministic parts of new-machine setup:
#   1. Create config_local.py from config_local.example.py and prompt for
#      the must-set keys (WORKING_DIR, DATA_DIR, VLM_REPO_DIR).
#   2. Materialize a CurrentStudy/sub-PILOT007 BIDS-ish skeleton with one
#      example session (ses-S001) plus models/ and training_data/.
#   3. Create the conda 'lsl' env from environment.yml (skip if present).
#   4. Clone the sibling harmony_vlm repo to VLM_REPO_DIR (skip if present).
#   5. Wire up the repo-local pre-commit hook (.githooks/pre-commit).
#   6. Optionally run tools/setup_linux_ufw.sh (asks for sudo).
#
# Then prints a checklist of irreducibly manual steps (eegoSports,
# LabRecorder, Pupil Labs Companion, dialout group, robot networking).
#
# Usage:
#   bash tools/bootstrap_machine.sh [flags]
#
# Flags (all optional — script is interactive otherwise):
#   --data-dir=PATH       Skip the DATA_DIR prompt
#   --vlm-repo-dir=PATH   Skip the VLM_REPO_DIR prompt
#   --no-conda            Skip conda env creation
#   --no-vlm              Skip cloning harmony_vlm
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

DATA_DIR_ARG=""
VLM_REPO_DIR_ARG=""
DO_CONDA=true
DO_VLM=true
DO_FIREWALL=true
NON_INTERACTIVE=false

for arg in "$@"; do
    case "$arg" in
        --data-dir=*)      DATA_DIR_ARG="${arg#*=}" ;;
        --vlm-repo-dir=*)  VLM_REPO_DIR_ARG="${arg#*=}" ;;
        --no-conda)        DO_CONDA=false ;;
        --no-vlm)          DO_VLM=false ;;
        --no-firewall)     DO_FIREWALL=false ;;
        --non-interactive) NON_INTERACTIVE=true ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) err "unknown flag: $arg"; exit 2 ;;
    esac
done

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
else
    cp config_local.example.py config_local.py
    ok "created config_local.py from example"

    DEFAULT_WORKING="$REPO_ROOT/"
    DEFAULT_DATA="$HOME/Documents/CurrentStudy"
    DEFAULT_VLM="$(dirname "$REPO_ROOT")/harmony_vlm"

    prompt_default WORKING_DIR  "WORKING_DIR (repo root)"             "$DEFAULT_WORKING"
    prompt_default DATA_DIR     "DATA_DIR (XDFs / training data)"     "${DATA_DIR_ARG:-$DEFAULT_DATA}"
    prompt_default VLM_REPO_DIR "VLM_REPO_DIR (sibling harmony_vlm)"  "${VLM_REPO_DIR_ARG:-$DEFAULT_VLM}"

    # Replace the three placeholder lines in config_local.py. Each
    # example value is unique so anchor on the full assignment.
    python3 - "$WORKING_DIR" "$DATA_DIR" "$VLM_REPO_DIR" <<'PY'
import sys, pathlib, re
working, data, vlm = sys.argv[1:]
p = pathlib.Path("config_local.py")
text = p.read_text()
def replace(text, key, val):
    pat = re.compile(rf'^{key}\s*=.*$', re.M)
    return pat.sub(f'{key} = {val!r}', text, count=1)
text = replace(text, "WORKING_DIR", working)
text = replace(text, "DATA_DIR", data)
text = replace(text, "VLM_REPO_DIR", vlm)
p.write_text(text)
PY
    ok "wrote WORKING_DIR / DATA_DIR / VLM_REPO_DIR into config_local.py"
    warn "all other keys keep their example defaults — open config_local.py and edit hosts/ports if your machine isn't single-machine loopback"
fi

# Re-read the values from config_local.py so subsequent steps use the
# real-on-disk values, not whatever was prompted (covers the
# already-existed branch).
DATA_DIR="$(python3 -c 'import config_local; print(config_local.DATA_DIR)')"
VLM_REPO_DIR="$(python3 -c 'import config_local; print(config_local.VLM_REPO_DIR)')"

# --- step 2: CurrentStudy/sub-PILOT007 skeleton ----------------------------

section "CurrentStudy/sub-PILOT007 skeleton"

if [ -z "$DATA_DIR" ]; then
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

section "conda env 'lsl'"

if [ "$DO_CONDA" = "false" ]; then
    skip "--no-conda passed"
elif ! command -v conda >/dev/null 2>&1; then
    warn "conda not on PATH — skipping; install Miniconda then re-run with no flags"
elif conda env list | awk '{print $1}' | grep -qx "lsl"; then
    skip "conda env 'lsl' already exists"
else
    conda env create -f environment.yml -n lsl
    ok "created conda env 'lsl'"
fi

# --- step 4: sibling harmony_vlm clone -------------------------------------

section "sibling harmony_vlm repo"

if [ "$DO_VLM" = "false" ]; then
    skip "--no-vlm passed"
elif [ -z "$VLM_REPO_DIR" ]; then
    warn "VLM_REPO_DIR is empty — skipping clone"
elif [ -d "$VLM_REPO_DIR/.git" ]; then
    skip "$VLM_REPO_DIR already a git repo"
else
    mkdir -p "$(dirname "$VLM_REPO_DIR")"
    git clone https://github.com/vivianchen98/harmony_vlm.git "$VLM_REPO_DIR"
    ok "cloned harmony_vlm to $VLM_REPO_DIR"
fi

# --- step 5: repo-local pre-commit hook ------------------------------------

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

# --- step 6: ufw -----------------------------------------------------------

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

ok "bootstrap complete"
