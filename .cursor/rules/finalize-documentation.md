---
description: >-
  When the user asks to commit, push, merge, or finalize work, update CHANGELOG.md
  and README.md for user-visible changes unless they say to skip documentation.
alwaysApply: true
---

# Finalize documentation (Cursor agent workflow)

This is the closest equivalent to a “pre-push doc hook”: **the agent** runs these steps when the user requests **commit**, **push**, **merge**, **finalize**, **ship**, **PR ready**, or similar—**not** a separate Cursor or Git product hook.

Unless the user explicitly says **skip docs**, **docs later**, or **commit only**:

1. **Review scope** — From `git status`, `git diff --cached`, and the task’s touched paths, list files that affect **operators, analysts, or new clones**: e.g. `config.py`, `control_panel.py`, `README.md`, `CHANGELOG.md`, `ExperimentDriver*.py`, `Utils/runtime_common.py`, `Utils/networking.py`, `Utils/EEGStreamState.py`, `Utils/experiment_utils.py`, `Generate_Riemannian_adaptive.py`, `generate_xgboost*.py`, `explore_dataset_library.py`, `environment.yml`, training or gaze entrypoints.

2. **`CHANGELOG.md`** — Under **`[Unreleased]`**, add short bullets for behavioral or operational impact (what changed for runs, training, or config). Create or repair `[Unreleased]` if missing. Do not log pure refactors with no outward effect.

3. **`README.md`** — Update **only if** setup steps, primary commands, architecture blubs, or configuration stories changed. Keep edits minimal and accurate.

4. **Order** — Prefer applying doc edits **in the same commit** (or immediately before `git push`) as the code changes, so the remote always has matching docs.

5. **Accuracy** — Do not document features that are not in the code. If unsure, say what was verified vs assumed.

Git’s `.githooks/pre-commit` only **prints hints**; this rule is what makes an **agent** actually edit the files.
