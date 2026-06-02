"""Config-sweep harness for the clinical ERD refined figures (Tier 3).

This is an analysis-only orchestrator that ties together the two existing
ERD tools without re-implementing either:

  * `Analyze_clinical_erd_refined.py` — the heavy MNE/TFR figure generator.
    It is *shelled out to* (never imported) so this orchestrator holds no
    MNE objects and its own RSS stays tiny.
  * `evaluate_erd_quality.py` — the pure-python rule-based scorer. Its
    `score_dir` is *imported* (no MNE, no TFR) and called directly.

Workflow:
  1. Build a small explicit config grid (spatial filter x window-end x a
     rejection arm) and predict each config's `variant_tag` exactly as the
     ERD script composes it (see `_variant_tag`).
  2. Generation phase (optional): run the ERD generator once per config,
     STRICTLY SEQUENTIAL and MEMORY-CAPPED via `systemd-run --user --scope`.
     The box has frozen from parallel TFR runs, so every generation blocks
     until it finishes before the next starts (subprocess.run, never
     Popen-parallel).
  3. Scoring phase: score each config's already-emitted npz side-cars with
     `score_dir(npz_dir, variant=tag)` and reduce to one cohort score.
  4. Ranking + top-N: rank configs by cohort score, write
     `sweep_<run_tag>/sweep_results.{json,csv}`, and symlink the top-N
     configs' per-subject + cohort PNGs into `sweep_<run_tag>/top_n/`
     (zero-cost — the figures already exist from the generation phase).

Memory cap (MANDATORY): each generation is wrapped as
  systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=0 --quiet \
      -- python Analyze_clinical_erd_refined.py <flags> > <log> 2>&1
This caps a single TFR pass and forbids swap; sequential execution keeps
the box from freezing.

CLI:
    python sweep_erd_quality.py \
        [--filters car,hjorth,csd] [--window-ends 5] [--reject-z 5] \
        [--with-noreject] [--subjects CSV] [--top-n 5] \
        [--run-tag latest] [--dry-run] [--skip-generate]

Tier 3 (analysis-only). Does not touch any realtime/hardware file and does
not modify either ERD tool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Repo root on sys.path so `evaluate_erd_quality` and the helpers package
# resolve regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate_erd_quality import score_dir  # noqa: E402
from exploration.clinical_analysis._helpers import (  # noqa: E402
    clin_pictures_root,
)

# The ERD generator and its log directory. The generator is invoked through
# the *current* interpreter (sys.executable) so the sweep runs in whatever
# conda env the user launched it from.
_ERD_SCRIPT = _REPO_ROOT / "Analyze_clinical_erd_refined.py"
_LOG_DIR = Path("/tmp/clinfig_logs")

# The ERD script's default analysis-window end (s). Mirrors
# CONFIG_A_DISPLAY_BASELINE["trial_win"][1] which the script defaults to 5;
# a window-end equal to this is "default" and is NOT tagged (see
# Analyze_clinical_erd_refined.py:717-719).
_DEFAULT_WINDOW_END = 5.0


# ----------------------------------------------------------------------
# variant_tag prediction (mirror of Analyze_clinical_erd_refined.py main)
# ----------------------------------------------------------------------

def _variant_tag(spatial_filter: str, reject_z: float, window_end: float,
                 subjects: list[str]) -> str:
    """Predict the ERD script's output `variant_tag` for a config.

    This replicates EXACTLY the tag composition in
    `Analyze_clinical_erd_refined.py:707-719` so the harness can name the
    npz / PNG files a given config will produce without running it:

      tag = "_" + spatial_filter                         (always; line 707)
      if subjects:  tag += "_subj-" + "-".join(...)      (lines 708-711)
      if reject_z <= 0:  tag += "_noreject"              (lines 715-716)
      if |window_end - 5| > 1e-9:  tag += f"_w{end:g}"   (lines 717-719)

    `subjects` is the list of full subject labels (e.g. "CLIN_SUBJ_004");
    the script sorts them and strips the "CLIN_SUBJ_" prefix, so the order
    passed here does not matter.
    """
    tag = f"_{spatial_filter}"
    if subjects:
        tag += "_subj-" + "-".join(
            s.replace("CLIN_SUBJ_", "") for s in sorted(subjects)
        )
    if reject_z <= 0:
        tag += "_noreject"
    if abs(float(window_end) - _DEFAULT_WINDOW_END) > 1e-9:
        tag += f"_w{float(window_end):g}"
    return tag


def _config_name(cfg: dict) -> str:
    """Short filesystem-safe name for a config (used in log filenames).

    Derived from the predicted variant_tag with the leading underscore
    stripped, so `sweep_<config>.log` reads e.g. `sweep_car.log`,
    `sweep_csd_noreject.log`.
    """
    return cfg["variant_tag"].lstrip("_") or "default"


# ----------------------------------------------------------------------
# Config grid
# ----------------------------------------------------------------------

def build_grid(filters: list[str], window_ends: list[float],
               reject_z: float, with_noreject: bool,
               subjects: list[str]) -> list[dict]:
    """Cartesian product filters x window_ends x rejection-arm.

    The rejection arm is "on" (the supplied reject_z) for every cell, plus
    an extra "off" (reject_z=0, which the ERD script reads as disabled) cell
    per (filter, window_end) when `with_noreject` is set. Each config dict
    carries its predicted `variant_tag` so downstream phases can find files.
    """
    grid: list[dict] = []
    for filt in filters:
        for win in window_ends:
            arms = [reject_z]
            if with_noreject:
                arms.append(0.0)  # <=0 disables rejection in the ERD script
            for rz in arms:
                cfg = {
                    "filter": filt,
                    "reject_z": float(rz),
                    "window_end": float(win),
                }
                cfg["variant_tag"] = _variant_tag(
                    filt, float(rz), float(win), subjects,
                )
                grid.append(cfg)
    return grid


# ----------------------------------------------------------------------
# Generation phase (capped, sequential)
# ----------------------------------------------------------------------

def _generation_cmd(cfg: dict, subjects: list[str]) -> list[str]:
    """Build the argv for one ERD generation, WITHOUT the systemd wrapper.

    Only non-default flags are passed so the produced variant_tag matches
    the prediction: --reject-z is included only for the off-arm (the on-arm
    uses the script's default), and --window-end only when non-default.
    """
    cmd = [
        sys.executable, str(_ERD_SCRIPT),
        "--spatial-filter", cfg["filter"],
    ]
    if subjects:
        cmd += ["--subjects", ",".join(subjects)]
    # reject_z: the ERD default is the "on" arm, so only pass --reject-z for
    # the off arm (<=0) to keep the on-arm tag clean. Passing the default
    # numeric value would still produce the same tag, but omitting it keeps
    # the command identical to a plain default run for auditability.
    if cfg["reject_z"] <= 0:
        cmd += ["--reject-z", f"{cfg['reject_z']:g}"]
    if abs(cfg["window_end"] - _DEFAULT_WINDOW_END) > 1e-9:
        cmd += ["--window-end", f"{cfg['window_end']:g}"]
    return cmd


def _capped_cmd(inner: list[str], log_path: Path) -> list[str]:
    """Wrap an argv in the mandatory memory-capped systemd-run scope.

    MemoryMax=11G caps a single TFR pass; MemorySwapMax=0 forbids swap so a
    runaway pass is OOM-killed instead of thrashing the box. The scope runs
    one command and exits, so sequential subprocess.run calls never overlap.
    stdout+stderr are redirected to log_path by the shell.
    """
    quoted = " ".join(_shquote(a) for a in inner)
    return [
        "systemd-run", "--user", "--scope",
        "-p", "MemoryMax=11G", "-p", "MemorySwapMax=0", "--quiet",
        "--", "bash", "-c", f"{quoted} > {_shquote(str(log_path))} 2>&1",
    ]


def _shquote(s: str) -> str:
    """Minimal POSIX shell quoting for embedding argv in a bash -c string."""
    if s and all(c.isalnum() or c in "-_./=," for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def run_generation(grid: list[dict], subjects: list[str],
                   dry_run: bool) -> dict[str, dict]:
    """Run (or, if dry_run, print) one capped ERD generation per config,
    strictly sequential. Returns {variant_tag: {exit_code, elapsed_s, log}}.

    A nonzero exit is RECORDED and the sweep CONTINUES to the next config
    (surface real failures in the summary rather than aborting the batch).
    The systemd wrapper itself can also fail (e.g. bus unavailable); that
    too is captured as the exit status rather than swallowed.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}
    for i, cfg in enumerate(grid, 1):
        name = _config_name(cfg)
        log_path = _LOG_DIR / f"sweep_{name}.log"
        inner = _generation_cmd(cfg, subjects)
        capped = _capped_cmd(inner, log_path)
        if dry_run:
            # Print exactly what would run, both the inner command and the
            # full capped wrapper, so the operator can eyeball the cap.
            print(f"[{i}/{len(grid)}] config={name} tag={cfg['variant_tag']}")
            print(f"    inner : {' '.join(_shquote(a) for a in inner)}")
            print(f"    capped: {' '.join(_shquote(a) for a in capped)}")
            status[cfg["variant_tag"]] = {
                "exit_code": None, "elapsed_s": None, "log": str(log_path),
            }
            continue
        print(f"[{i}/{len(grid)}] generating config={name} "
              f"tag={cfg['variant_tag']} -> {log_path}")
        t0 = time.time()
        proc = subprocess.run(capped)  # blocks until this config finishes
        elapsed = time.time() - t0
        print(f"    exit={proc.returncode} elapsed={elapsed:.1f}s")
        status[cfg["variant_tag"]] = {
            "exit_code": proc.returncode,
            "elapsed_s": round(elapsed, 1),
            "log": str(log_path),
        }
    return status


# ----------------------------------------------------------------------
# Scoring phase
# ----------------------------------------------------------------------

def _cohort_score(rows: list[dict]) -> tuple[float | None, int, dict]:
    """Reduce a config's scored rows to a cohort-level score.

    Mirrors the scorer's own cohort-summary convention
    (evaluate_erd_quality.py:685-697): the per-config cohort score is the
    MEDIAN of S over ELIGIBLE BILATERAL rows, here pooled across all of the
    config's subjects/sessions. Returns (cohort_S, n_eligible_bilat,
    gate_tally). cohort_S is None when no eligible bilat row exists.
    """
    bilat_S = [
        r["S"] for r in rows
        if r["cluster"] == "bilat" and r["eligible"] and r["S"] is not None
    ]
    gate_tally: dict[str, int] = {}
    for r in rows:
        for g in r["gates_failed"]:
            gate_tally[g] = gate_tally.get(g, 0) + 1
    if not bilat_S:
        return None, 0, gate_tally
    bilat_S = sorted(bilat_S)
    n = len(bilat_S)
    median = (bilat_S[n // 2] if n % 2
              else 0.5 * (bilat_S[n // 2 - 1] + bilat_S[n // 2]))
    return float(median), n, gate_tally


def run_scoring(grid: list[dict], npz_dir: Path,
                gen_status: dict[str, dict]) -> list[dict]:
    """Score each config's npz and assemble one result record per config."""
    results: list[dict] = []
    for cfg in grid:
        tag = cfg["variant_tag"]
        rows = score_dir(npz_dir, variant=tag)
        cohort_S, n_elig, gate_tally = _cohort_score(rows)
        gen = gen_status.get(tag, {})
        results.append({
            "filter": cfg["filter"],
            "reject_z": cfg["reject_z"],
            "window_end": cfg["window_end"],
            "variant_tag": tag,
            "cohort_S": cohort_S,
            "n_eligible_bilat": n_elig,
            "n_rows_scored": len(rows),
            "gate_tally": gate_tally,
            "no_eligible_bilat": cohort_S is None,
            "gen_exit_code": gen.get("exit_code"),
            "gen_elapsed_s": gen.get("elapsed_s"),
            "gen_log": gen.get("log"),
        })
    return results


# ----------------------------------------------------------------------
# Ranking, output, top-N linking
# ----------------------------------------------------------------------

def rank_results(results: list[dict]) -> list[dict]:
    """Return results sorted by cohort_S descending, None last.

    Sort key puts None-scored configs (no eligible bilat rows) at the end
    regardless of the others' magnitudes; ties break by variant_tag for a
    stable, reproducible order.
    """
    return sorted(
        results,
        key=lambda r: (
            r["cohort_S"] is None,
            -(r["cohort_S"] if r["cohort_S"] is not None else 0.0),
            r["variant_tag"],
        ),
    )


def _png_targets(erd_dir: Path, tag: str) -> list[Path]:
    """Existing per-subject + cohort PNGs for a variant tag.

    Per-subject figures match `*_6panel_mi_rest<tag>.png` and the cohort
    figure is `cohort_6panel_mi_rest<tag>.png`
    (Analyze_clinical_erd_refined.py:831,841). The glob naturally excludes
    longer-tagged neighbours only at the suffix, so anchor on the trailing
    tag + extension.
    """
    suffix = f"_6panel_mi_rest{tag}.png"
    return sorted(p for p in erd_dir.glob(f"*{suffix}")
                  if p.name.endswith(suffix))


def link_top_n(ranked: list[dict], top_n: int, erd_dir: Path,
               top_dir: Path) -> dict[str, list[str]]:
    """Symlink the top-N configs' PNGs into `top_dir`.

    Symlinks (not copies) keep this zero-cost: the figures already exist
    from the generation phase. Returns {variant_tag: [linked filenames]}.
    A config with no PNGs on disk (e.g. generation failed) links nothing
    and is recorded with an empty list — surfaced, not silently skipped.
    Re-running overwrites stale links (the run_tag dir is reused).
    """
    top_dir.mkdir(parents=True, exist_ok=True)
    linked: dict[str, list[str]] = {}
    for cfg in ranked[:top_n]:
        tag = cfg["variant_tag"]
        names: list[str] = []
        for src in _png_targets(erd_dir, tag):
            dst = top_dir / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            names.append(src.name)
        linked[tag] = names
    return linked


def _write_outputs(ranked: list[dict], linked: dict[str, list[str]],
                   out_dir: Path) -> tuple[Path, Path]:
    """Write sweep_results.json and sweep_results.csv. Returns both paths."""
    json_path = out_dir / "sweep_results.json"
    csv_path = out_dir / "sweep_results.csv"
    payload = {
        "ranked": ranked,
        "top_n_linked": linked,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # CSV: one row per config, gate tally flattened to a "G1:3;G2:1" cell.
    import csv as _csv
    fields = [
        "rank", "variant_tag", "filter", "reject_z", "window_end",
        "cohort_S", "n_eligible_bilat", "n_rows_scored", "gate_tally",
        "gen_exit_code", "gen_elapsed_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(ranked, 1):
            tally = ";".join(f"{g}:{r['gate_tally'][g]}"
                             for g in sorted(r["gate_tally"]))
            w.writerow({
                "rank": i,
                "variant_tag": r["variant_tag"],
                "filter": r["filter"],
                "reject_z": r["reject_z"],
                "window_end": r["window_end"],
                "cohort_S": "" if r["cohort_S"] is None else r["cohort_S"],
                "n_eligible_bilat": r["n_eligible_bilat"],
                "n_rows_scored": r["n_rows_scored"],
                "gate_tally": tally,
                "gen_exit_code": ("" if r["gen_exit_code"] is None
                                  else r["gen_exit_code"]),
                "gen_elapsed_s": ("" if r["gen_elapsed_s"] is None
                                  else r["gen_elapsed_s"]),
            })
    return json_path, csv_path


def _print_ranked(ranked: list[dict], linked: dict[str, list[str]]) -> None:
    """Print a scannable ranked table to stdout."""
    print("\n=== Sweep ranking (by cohort median S over eligible bilat) ===")
    header = (f"{'rank':>4}  {'variant_tag':<28}  {'cohort_S':>8}  "
              f"{'n_elig':>6}  {'gates':<20}  {'exit':>4}  links")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked, 1):
        s = "  --  " if r["cohort_S"] is None else f"{r['cohort_S']:.4f}"
        tally = ",".join(f"{g}:{r['gate_tally'][g]}"
                         for g in sorted(r["gate_tally"])) or "-"
        exit_c = "-" if r["gen_exit_code"] is None else str(r["gen_exit_code"])
        n_links = len(linked.get(r["variant_tag"], []))
        link_str = str(n_links) if r["variant_tag"] in linked else "-"
        print(f"{i:>4}  {r['variant_tag']:<28}  {s:>8}  "
              f"{r['n_eligible_bilat']:>6}  {tally:<20}  {exit_c:>4}  "
              f"{link_str}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_subjects(raw: str) -> list[str]:
    """Parse a --subjects CSV into a list of full subject labels.

    Bare numbers and short forms are normalised to CLIN_SUBJ_NNN so the
    predicted variant_tag matches the ERD script (which only strips the
    prefix it itself added). Empty input -> [] (full cohort).
    """
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("CLIN_SUBJ_"):
            out.append(tok)
        elif tok.isdigit():
            out.append(f"CLIN_SUBJ_{int(tok):03d}")
        else:
            out.append(tok)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filters", default="car,hjorth,csd",
        help="Comma-separated spatial filters to sweep (default car,hjorth,csd).",
    )
    parser.add_argument(
        "--window-ends", default="5",
        help="Comma-separated post-cue window-end values in s (default 5).",
    )
    parser.add_argument(
        "--reject-z", type=float, default=5.0,
        help="Robust-z rejection threshold for the rejection-ON arm "
             "(default 5). The ERD script's own default supplies this arm.",
    )
    parser.add_argument(
        "--with-noreject", action="store_true",
        help="Add a rejection-OFF arm (reject-z=0) per filter/window for the "
             "on/off comparison.",
    )
    parser.add_argument(
        "--subjects", default="",
        help="Comma-separated subject filter (e.g. CLIN_SUBJ_004 or 004). "
             "Empty = full cohort.",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top-ranked configs to symlink into top_n/ (default 5).",
    )
    parser.add_argument(
        "--run-tag", default="latest",
        help="Names the output dir sweep_<run_tag>/. Injectable (no hidden "
             "datetime.now) so runs are reproducible; default 'latest' is "
             "OVERWRITTEN on re-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the generation commands that WOULD run, then exit.",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip generation; score already-emitted npz only.",
    )
    args = parser.parse_args(argv)

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    window_ends = [float(w) for w in args.window_ends.split(",") if w.strip()]
    subjects = _parse_subjects(args.subjects)

    grid = build_grid(
        filters, window_ends, args.reject_z, args.with_noreject, subjects,
    )
    print(f"Config grid: {len(grid)} configs")
    for cfg in grid:
        print(f"  {cfg['filter']:<7} reject_z={cfg['reject_z']:g} "
              f"window_end={cfg['window_end']:g} -> tag '{cfg['variant_tag']}'")

    if args.dry_run:
        print("\n--dry-run: commands that WOULD run (generation skipped):")
        run_generation(grid, subjects, dry_run=True)
        return 0

    if args.skip_generate:
        print("\n--skip-generate: scoring already-emitted npz.")
        gen_status: dict[str, dict] = {}
    else:
        print("\nGeneration phase (capped, sequential):")
        gen_status = run_generation(grid, subjects, dry_run=False)

    erd_dir = clin_pictures_root() / "erd_refined"
    npz_dir = erd_dir / "per_trial"
    if not npz_dir.exists():
        raise FileNotFoundError(f"npz dir does not exist: {npz_dir}")

    results = run_scoring(grid, npz_dir, gen_status)
    ranked = rank_results(results)

    out_dir = erd_dir / f"sweep_{args.run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    top_dir = out_dir / "top_n"
    linked = link_top_n(ranked, args.top_n, erd_dir, top_dir)
    json_path, csv_path = _write_outputs(ranked, linked, out_dir)

    _print_ranked(ranked, linked)
    print(f"\nWrote:\n  {json_path}\n  {csv_path}\n  {top_dir}/ "
          f"({sum(len(v) for v in linked.values())} symlinks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
