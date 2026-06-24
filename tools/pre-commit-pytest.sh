#!/bin/bash
# Git pre-commit hook: run the fast pytest suite (`pytest tests/ -m "not slow" -q`)
# before allowing the commit through. Designed to live at .git/hooks/pre-commit;
# kept under tools/ so it can be version-controlled and installed per clone via
# `tools/install-pre-commit-hook.sh` (see tests/README.md).
#
# Wired up by Harmony_Test_Suite_Plan.md §5.3. If pytest is unavailable, the
# hook is a no-op — it does not block commits on machines without the lsl env.
#
# Bypass: `git commit --no-verify`. NEVER bypass on main / master.

set -u

# Resolve a pytest runner for the `lsl` env. Order:
#   1. the env's pytest binary directly — fast path on the Linux primary host.
#   2. pytest already on PATH — env activated in the calling shell.
#   3. `conda run -n lsl pytest` — covers any host where the env exists but is
#      not activated. This is the native-Windows case: under Git Bash `conda`
#      is on PATH but the env's pytest lives at Scripts/pytest.exe and is not,
#      so without this branch the gate silently no-ops on Windows.
# If none resolve, the hook is a no-op (machines without the env).
PYTEST_CMD=()
linux_pytest="/home/arman-admin/opt/miniconda/envs/lsl/bin/pytest"
if [ -x "$linux_pytest" ]; then
  PYTEST_CMD=("$linux_pytest")
elif command -v pytest >/dev/null 2>&1; then
  PYTEST_CMD=("$(command -v pytest)")
else
  conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
  # Confirm the env actually has pytest before committing to it, so a missing
  # env stays a skip (don't block the commit) rather than a hard conda error.
  if [ -n "$conda_bin" ] && "$conda_bin" run -n lsl python -c "import pytest" >/dev/null 2>&1; then
    # --no-capture-output so pytest's progress/failures stream to the terminal.
    PYTEST_CMD=("$conda_bin" run -n lsl --no-capture-output pytest)
  fi
fi

if [ "${#PYTEST_CMD[@]}" -eq 0 ]; then
  echo "pre-commit-pytest: pytest not on PATH and no usable 'lsl' env — skipping." >&2
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${repo_root}" ]; then
  exit 0
fi

cd "$repo_root" || exit 0

if [ ! -d "tests" ]; then
  exit 0
fi

echo "pre-commit-pytest: running fast suite (-m 'not slow') ..." >&2
"${PYTEST_CMD[@]}" tests/ -m "not slow" -q --tb=line
rc=$?

if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "pre-commit-pytest: commit BLOCKED — fix the failing tests above and retry." >&2
  echo "  To bypass for an emergency (NEVER on main/master): git commit --no-verify" >&2
  exit "$rc"
fi

exit 0
