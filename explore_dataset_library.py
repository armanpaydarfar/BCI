#!/usr/bin/env python3
"""
Offline dataset library exploration for expert decoder pool curation.

Stages:
  baseline+screen — screen session EEG files only (…/ses-*/eeg/*.xdf), score/rank candidates
                    and compute baseline reference stats from sub-<baseline>/training_data/*.xdf.
  deep            — optional session-held-out benchmark on explicit shortlist paths

Examples:
  python explore_dataset_library.py --out-dir ./exploration_run_001
  python explore_dataset_library.py --backends mdm,xgb_cov --max-files 50
  python explore_dataset_library.py --stage deep --deep-models mdm,xgb_cov --shortlist-file paths.txt

Outputs (baseline+screen): RUN_SUMMARY.txt, CSV (+ optional Parquet), PROPOSED_ADDITIONS.txt.

If you pass more than one backend in --backends, each model gets its own outputs (no combined ranking):
  ranked_proposals_<backend>.csv, PROPOSED_ADDITIONS_<backend>.txt, and RUN_SUMMARY sections per model.
  candidates.csv merges per-model composite/flags as composite_score__<backend> (for spreadsheets only).
  PROPOSED_ADDITIONS.txt is an index; ranked_proposals.csv mirrors the first backend in your list (compat).

Per-file errors and uncaught exceptions are recorded and the scan continues. Live console: `→` when a file
starts, optional `k-fold i/n` during decoders, `feat=`/`dec=` timing on the result line, checkpoints every 25 files.
The candidate scan includes only files under `.../ses-*/eeg/` and skips anything under `training_data/`.
Baseline reference remains only .xdf in sub-<baseline>/training_data/ (no subfolders), evaluated silently.
Session filters aggregate all logs/ONLINE_*|OFFLINE_* snapshots for that ses-* (one EEG per session vs many log runs;
EEG filename run* is not paired to a single log folder). Use --mi-arm and default ERRP exclusion as in
Utils/dataset_exploration/run_metadata.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ["MNE_USE_NUMBA"] = "false"

import mne
import numpy as np
import pandas as pd

import config
from Utils.dataset_exploration.baseline import config_snapshot, list_baseline_xdf_files, reference_stats
from Utils.dataset_exploration.discovery import (
    discover_xdf_files,
    duplicate_groups_by_size,
    index_xdf_path,
    path_matches_exclude,
)
from Utils.dataset_exploration.io_reports import write_df, write_json, write_proposed_additions, write_run_summary
from Utils.dataset_exploration.metrics_decoders import kfold_decoder_metrics
from Utils.dataset_exploration.metrics_file_level import compute_static_metrics, extract_features_for_file
from Utils.dataset_exploration.run_metadata import exploration_filter_reason
from Utils.dataset_exploration.scoring import (
    add_scores_and_flags,
    merge_multi_primary_scored,
    rank_proposals,
    report_primary_backends,
)


def _parse_backends(raw: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"mdm", "xgb_cov", "xgb_cov_erd"}
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported backend(s): {bad}. Allowed: {sorted(allowed)}")
    return items


def _parse_exclude_substrings(raw: str) -> tuple[str, ...]:
    parts = tuple(x.strip() for x in raw.split(",") if x.strip())
    return parts if parts else ("OBS", "old")


def _resolve_exploration_filters(args: argparse.Namespace) -> tuple[str | None, bool]:
    """
    Returns (mi_arm_required_lower_or_None, skip_errp).

    mi_arm None → no arm filtering. skip_errp True → drop ERRP-ish runs/paths.
    """
    if args.mi_arm == "config":
        want = getattr(config, "ARM_SIDE", "Right").strip().lower()
    elif args.mi_arm == "any":
        want = None
    else:
        want = str(args.mi_arm).strip().lower()
    skip_errp = not bool(getattr(args, "include_errp", False))
    return want, skip_errp


def _infer_lr_from_channel_name(channel_name: str) -> str | None:
    """
    Very lightweight left/right heuristic for 10-20-style EEG names.

    - '...z' / 'Z' channels are treated as midline and return None.
    - If a trailing/embedded digit exists: odd -> left, even -> right.
    """
    if channel_name is None:
        return None
    s = str(channel_name).strip().upper()
    if not s:
        return None
    if s.endswith("Z") or s == "Z":
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    d = int(m.group(1))
    return "left" if (d % 2 == 1) else "right"


def _compute_erd_laterality(
    feats: dict[str, object],
    *,
    mi_arm: str | None,
    bilateral_ratio_threshold: float = 0.85,
    rest_leak_threshold: float = 0.1,
) -> dict[str, object]:
    """
    Returns per-file ERD laterality diagnostics.

    ERD features are stored flattened as:
      erd.reshape(n_windows, n_channels, n_bands)
    where the channel/band ordering matches `Utils/xgb_feature_pipeline.py`.
    """
    erd = feats.get("erd")
    channel_names = feats.get("channel_names")
    n_ch = feats.get("n_channels")

    if erd is None or channel_names is None or n_ch is None:
        return {}

    erd_arr = np.asarray(erd)
    if erd_arr.ndim != 2 or erd_arr.shape[0] < 2:
        return {}

    ch_list = list(channel_names)  # type: ignore[arg-type]
    n_ch_i = len(ch_list)
    if n_ch_i <= 0:
        return {}

    if erd_arr.shape[1] % n_ch_i != 0:
        return {}
    n_bands = int(erd_arr.shape[1] // n_ch_i)
    if n_bands <= 0:
        return {}

    erd_tensor = erd_arr.reshape(erd_arr.shape[0], n_ch_i, n_bands)
    labels = np.asarray(feats.get("labels"))
    rest_label = feats.get("rest_label")
    mi_label = feats.get("mi_label")
    if labels.ndim != 1 or labels.shape[0] != erd_tensor.shape[0]:
        return {}
    rest_mask = labels == int(rest_label) if rest_label is not None else np.zeros(labels.shape[0], dtype=bool)
    mi_mask = labels == int(mi_label) if mi_label is not None else np.zeros(labels.shape[0], dtype=bool)
    # ROI-based laterality (requested):
    # contralateral seed set = {CP1, CP5, C3, FC1, FC5}; ipsilateral = mirrored even channels.
    roi_left = {"CP1", "CP5", "C3", "FC1", "FC5"}
    roi_right = {"CP2", "CP6", "C4", "FC2", "FC6"}

    # Keep robustness to whitespace / case in channel names.
    ch_norm = [str(ch).strip().upper() for ch in ch_list]
    left_idx = [i for i, ch in enumerate(ch_norm) if ch in roi_left]
    right_idx = [i for i, ch in enumerate(ch_norm) if ch in roi_right]

    eps = 1e-12
    # Signed ERD strength:
    # ERD is log(P_task/P_base), so desynchronization (drop in power) is negative.
    # We convert to positive "strength of ERD drop" via max(0, -ERD).
    def _strength(mask: np.ndarray, idxs: list[int]) -> float:
        if not idxs or not np.any(mask):
            return float("nan")
        return float(np.mean(np.maximum(0.0, -erd_tensor[mask][:, idxs, :])))

    # All-windows strength (kept for continuity / diagnostics)
    left_strength = _strength(np.ones(labels.shape[0], dtype=bool), left_idx)
    right_strength = _strength(np.ones(labels.shape[0], dtype=bool), right_idx)
    # MI-only strength (used for lateralization pattern)
    left_strength_mi = _strength(mi_mask, left_idx)
    right_strength_mi = _strength(mi_mask, right_idx)
    # REST-only ERD leak strength (penalty driver)
    left_strength_rest = _strength(rest_mask, left_idx)
    right_strength_rest = _strength(rest_mask, right_idx)

    denom_lr = max(left_strength_mi, right_strength_mi) + eps if not any(pd.isna([left_strength_mi, right_strength_mi])) else float("nan")
    bilateral_ratio = (
        float(min(left_strength_mi, right_strength_mi) / (denom_lr + eps))
        if np.isfinite(left_strength_mi) and np.isfinite(right_strength_mi) and denom_lr > 0
        else float("nan")
    )
    bilateral_flag = (
        bool(bilateral_ratio >= float(bilateral_ratio_threshold))
        if np.isfinite(bilateral_ratio)
        else False
    )

    out: dict[str, object] = {
        "erd_left_strength": left_strength,
        "erd_right_strength": right_strength,
        "erd_left_strength_mi": left_strength_mi,
        "erd_right_strength_mi": right_strength_mi,
        "erd_left_strength_rest": left_strength_rest,
        "erd_right_strength_rest": right_strength_rest,
        "erd_bilateral_ratio": bilateral_ratio,
        "erd_bilateral_flag": bilateral_flag,
        "erd_roi_left_n": int(len(left_idx)),
        "erd_roi_right_n": int(len(right_idx)),
    }

    # Concise pattern label for quick triage:
    # - weak: low ERD strength on both sides
    # - bilateral: both sides strong and similar
    # - contra / ipsi: side dominance relative to requested arm
    strength_total = (
        float(left_strength_mi + right_strength_mi)
        if np.isfinite(left_strength_mi) and np.isfinite(right_strength_mi)
        else float("nan")
    )
    out["erd_strength_total"] = strength_total
    rest_leak_strength = (
        float(max(left_strength_rest, right_strength_rest))
        if np.isfinite(left_strength_rest) and np.isfinite(right_strength_rest)
        else float("nan")
    )
    out["erd_rest_leak_strength"] = rest_leak_strength
    out["erd_rest_leak_flag"] = bool(np.isfinite(rest_leak_strength) and rest_leak_strength >= float(rest_leak_threshold))

    if mi_arm not in ("left", "right"):
        out["erd_contra_minus_ipsi"] = float("nan")
        out["erd_contra_dominant"] = False
        if np.isfinite(strength_total):
            if strength_total < 0.05:
                out["erd_pattern"] = "weak"
            elif bilateral_flag:
                out["erd_pattern"] = "bilateral"
            else:
                out["erd_pattern"] = "mixed"
        else:
            out["erd_pattern"] = "unknown"
        return out

    # For MI of 'right', contralateral motor cortex is left hemisphere (and vice-versa).
    contra_hemi = "left" if mi_arm == "right" else "right"
    ipsi_hemi = "right" if contra_hemi == "left" else "left"

    contra_strength = left_strength_mi if contra_hemi == "left" else right_strength_mi
    ipsi_strength = left_strength_mi if ipsi_hemi == "left" else right_strength_mi
    contra_minus_ipsi = (
        float(contra_strength - ipsi_strength)
        if np.isfinite(contra_strength) and np.isfinite(ipsi_strength)
        else float("nan")
    )
    out["erd_contra_minus_ipsi"] = contra_minus_ipsi
    out["erd_contra_dominant"] = bool(contra_minus_ipsi > 0) if np.isfinite(contra_minus_ipsi) else False
    if not np.isfinite(strength_total):
        out["erd_pattern"] = "unknown"
    elif strength_total < 0.05:
        out["erd_pattern"] = "weak"
    elif bilateral_flag:
        out["erd_pattern"] = "bilateral"
    elif np.isfinite(contra_minus_ipsi) and contra_minus_ipsi > 0:
        out["erd_pattern"] = "contra"
    elif np.isfinite(contra_minus_ipsi) and contra_minus_ipsi < 0:
        out["erd_pattern"] = "ipsi"
    else:
        out["erd_pattern"] = "mixed"
    return out


def evaluate_path(
    xdf_path: str,
    *,
    backends: list[str],
    n_splits: int,
    min_windows: int,
    target_ambig: float,
    score_threshold: float,
    laterality_arm: str | None = None,
    on_kfold_step: Callable[[int, int], None] | None = None,
    progress_after_features: bool = False,
) -> dict:
    row: dict = {"path": xdf_path, "filename": Path(xdf_path).name}
    t0 = time.perf_counter()

    include_beta = "xgb_cov" in backends or "xgb_cov_erd" in backends
    # Always compute ERD features for screening metrics (cheap vs covariance path).
    feats = extract_features_for_file(
        xdf_path,
        include_erd=True,
        include_beta_cov=include_beta,
    )
    t_after_extract = time.perf_counter()

    if "error" in feats:
        row["eval_error"] = feats["error"]
        row["eval_s_load_features_s"] = float(t_after_extract - t0)
        row["eval_s_kfold_decode_s"] = 0.0
        return row

    row["n_channels"] = feats.get("n_channels", np.nan)
    row["n_windows"] = feats.get("n_windows", np.nan)
    row["class_counts_json"] = json.dumps(feats.get("class_counts", {}))

    static = compute_static_metrics(feats)
    for k, v in static.items():
        row[k] = v

    # ERD laterality tracking (left vs right + contra/ipsi if mi_arm is known).
    later = _compute_erd_laterality(feats, mi_arm=laterality_arm)
    for k, v in later.items():
        row[k] = v

    n_win = int(feats.get("n_windows", 0) or 0)
    if n_win < min_windows:
        row["eval_error"] = f"below_min_windows({n_win}<{min_windows})"
        row["eval_s_load_features_s"] = float(t_after_extract - t0)
        row["eval_s_kfold_decode_s"] = 0.0
        return row

    if progress_after_features and on_kfold_step is not None:
        print(
            f"    · features ready ({n_win} windows, {int(row.get('n_channels', 0) or 0)} ch) → k-fold…",
            flush=True,
        )

    dec = kfold_decoder_metrics(
        feats,
        backends=backends,
        n_splits=n_splits,
        target_ambig=target_ambig,
        score_threshold=score_threshold,
        on_kfold_step=on_kfold_step,
    )
    t_end = time.perf_counter()
    row["eval_s_load_features_s"] = float(t_after_extract - t0)
    row["eval_s_kfold_decode_s"] = float(t_end - t_after_extract)

    if "eval_error" in dec:
        row["eval_error"] = dec["eval_error"]
    for k, v in dec.items():
        if k != "eval_error":
            row[k] = v

    return row


def _status_from_row(row: dict, *, primary: str) -> str:
    ev = row.get("eval_error", "")
    if ev is None or (isinstance(ev, float) and pd.isna(ev)) or str(ev).strip() == "":
        return "OK"
    if str(ev).startswith("below_min_windows"):
        return "LOW"
    return "ERR"


def _print_banner(
    *,
    data_dir: Path,
    baseline_subject: str,
    backends: list[str],
    report_primaries: list[str],
    args: argparse.Namespace,
    exclude_substrings: tuple[str, ...],
    mi_arm_display: str,
    include_errp: bool,
) -> None:
    print("=" * 78)
    print("Harmony — offline dataset library exploration")
    print("=" * 78)
    print(f"  data_dir           : {data_dir}")
    print(f"  baseline_subject   : {baseline_subject}  (pool: …/sub-{baseline_subject}/training_data/)")
    print(f"  backends           : {', '.join(backends)}")
    if len(report_primaries) > 1:
        print(f"  scoring / reports  : separate list per decoder → {', '.join(report_primaries)}")
    else:
        print(f"  scoring (primary)  : {report_primaries[0]}")
    print(f"  n_splits           : {args.n_splits}  min_windows : {args.min_windows}")
    print(f"  score_threshold    : {args.score_threshold}  target_ambig : {args.target_ambig}")
    print(f"  exclude_substrings : {', '.join(exclude_substrings)}  (case-insensitive)")
    print(f"  mi_arm filter      : {mi_arm_display}  (no log / missing ARM_SIDE → keep file)")
    print(f"  ERRP sessions      : {'included (--include-errp)' if include_errp else 'excluded (default)'}")
    if args.max_files > 0:
        print(f"  max_files (debug)  : {args.max_files}")
    print("=" * 78)


def _print_discovery_stats(
    paths: list[str],
    baseline_paths: set[str],
    data_dir: Path,
    baseline_subject: str,
) -> None:
    print("\n[Discovery]")
    print(f"  Total .xdf files (recursive under data_dir): {len(paths)}")
    in_pool = sum(1 for p in paths if p in baseline_paths)
    print(f"  Files matching baseline training_data list:   {in_pool}")
    mods: Counter = Counter()
    subjects: Counter = Counter()
    for p in paths:
        meta = index_xdf_path(p, data_dir=data_dir, baseline_subject=baseline_subject)
        mods[meta.modality] += 1
        if meta.subject:
            subjects[meta.subject] += 1
    print(f"  By modality: {dict(mods)}")
    print(f"  Distinct subjects (from path): {len(subjects)}")
    dup = duplicate_groups_by_size(paths)
    if dup:
        ndup = sum(len(g) - 1 for g in dup.values())
        print(f"  Hint: {len(dup)} file sizes shared by multiple paths (~{ndup} possible duplicates)")
    else:
        print("  No same-size path groups (cheap duplicate hint).")


def _timing_suffix(row: dict) -> str:
    ext = row.get("eval_s_load_features_s")
    dec = row.get("eval_s_kfold_decode_s")
    if ext is None or dec is None:
        return ""
    try:
        e, d = float(ext), float(dec)
    except (TypeError, ValueError):
        return ""
    return f"  feat={e:.1f}s dec={d:.1f}s"


def _print_eval_checkpoint(
    rows_so_far: list[dict],
    idx_done: int,
    n_total: int,
    t_run0: float,
    *,
    report_primaries: list[str],
) -> None:
    primary0 = report_primaries[0]
    stc = Counter(_status_from_row(r, primary=primary0) for r in rows_so_far)
    ok_rows = [r for r in rows_so_far if _status_from_row(r, primary=primary0) == "OK"]
    auc_bits: list[str] = []
    for pb in report_primaries:
        col = f"{pb}__auc"
        vals: list[float] = []
        for r in ok_rows:
            v = r.get(col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                vals.append(float(v))
        if vals:
            auc_bits.append(f"mean_{pb}_auc={sum(vals) / len(vals):.3f}")
    elapsed = time.perf_counter() - t_run0
    eta_s = (elapsed / float(idx_done)) * float(n_total - idx_done) if idx_done > 0 else 0.0
    bits = (
        f"OK={stc.get('OK', 0)} LOW={stc.get('LOW', 0)} ERR={stc.get('ERR', 0)}"
        + (f"  |  {' '.join(auc_bits)}" if auc_bits else "")
        + f"  |  elapsed {elapsed / 60.0:.1f}m  ETA ~{eta_s / 60.0:.1f}m"
    )
    print(f"  — checkpoint {idx_done}/{n_total}: {bits}", flush=True)


def _print_file_result(
    idx: int,
    n_total: int,
    row: dict,
    *,
    report_primaries: list[str],
    elapsed_s: float,
) -> None:
    fn = row.get("filename", Path(row.get("path", "")).name)
    primary0 = report_primaries[0]
    st = _status_from_row(row, primary=primary0)
    subj = row.get("subject") or "—"
    mod = row.get("modality") or "—"
    nw = row.get("n_windows", "—")
    ev = row.get("eval_error", "")
    ev_short = (str(ev)[:56] + "…") if ev and len(str(ev)) > 56 else (ev or "")
    t_suf = _timing_suffix(row)

    def _auc_s(pb: str) -> str:
        col = f"{pb}__auc"
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return f"{float(v):.3f}"
        return "—"

    auc_part = (
        " | ".join(f"{p}={_auc_s(p)}" for p in report_primaries)
        if len(report_primaries) > 1
        else f"{report_primaries[0]}_auc={_auc_s(report_primaries[0])}"
    )

    if st == "OK":
        nw_s = str(int(nw)) if nw is not None and not pd.isna(nw) else "—"
        print(
            f"  [{idx:>4}/{n_total}] {st:<3}  {mod:<8}  subj={subj:<14}  n_win={nw_s:>6}  "
            f"{auc_part}  ({elapsed_s:5.1f}s){t_suf}  {fn}",
            flush=True,
        )
        l = row.get("erd_left_strength")
        rr = row.get("erd_right_strength")
        contra = row.get("erd_contra_minus_ipsi")
        bi = row.get("erd_bilateral_flag")
        if l is not None and rr is not None and not (isinstance(l, float) and pd.isna(l)) and not (isinstance(rr, float) and pd.isna(rr)):
            l_f = float(l)
            rr_f = float(rr)
            if contra is not None and not (isinstance(contra, float) and pd.isna(contra)):
                contra_f = float(contra)
                contra_s = f"contra-ipsi={contra_f:.3g}  "
            else:
                contra_s = ""
            bi_s = f"bilat={int(bi)}" if isinstance(bi, (bool, np.bool_)) else "bilat=—"
            pat = str(row.get("erd_pattern", "unknown"))
            st = row.get("erd_strength_total")
            st_s = f"S={float(st):.3g}  " if st is not None and not (isinstance(st, float) and pd.isna(st)) else ""
            rl = row.get("erd_rest_leak_strength")
            rl_s = f"restERD={float(rl):.3g}  " if rl is not None and not (isinstance(rl, float) and pd.isna(rl)) else ""
            print(
                f"        ERD hemi: L={l_f:.3g} R={rr_f:.3g}  {st_s}{contra_s}{rl_s}{bi_s}  pattern={pat}",
                flush=True,
            )
        for be in ("xgb_cov", "xgb_cov_erd"):
            key = f"{be}__top_features_gain"
            raw = row.get(key)
            if raw is None or raw == "" or (isinstance(raw, float) and pd.isna(raw)):
                continue
            s = str(raw)
            if s.startswith("failed:"):
                print(f"        {be} top_gain: {s}", flush=True)
                continue
            max_len = 118
            tail = " …" if len(s) > max_len else ""
            print(f"        {be} top_gain: {s[:max_len]}{tail}", flush=True)
    elif st == "LOW":
        nw_s = str(int(nw)) if nw is not None and not (isinstance(nw, float) and pd.isna(nw)) else str(nw)
        print(
            f"  [{idx:>4}/{n_total}] {st:<3}  {mod:<8}  subj={subj:<14}  n_win={nw_s:>6}  "
            f"skip ({ev_short})  ({elapsed_s:5.1f}s){t_suf}  {fn}",
            flush=True,
        )
    else:
        print(
            f"  [{idx:>4}/{n_total}] {st:<3}  {mod:<8}  subj={subj:<14}  "
            f"({elapsed_s:5.1f}s){t_suf}  {ev_short}  {fn}",
            flush=True,
        )


def _aggregate_error_counts(df: pd.DataFrame) -> Counter:
    c: Counter = Counter()
    for ev in df["eval_error"] if "eval_error" in df.columns else []:
        if pd.isna(ev) or str(ev).strip() == "":
            c["ok"] += 1
        elif str(ev).startswith("below_min_windows"):
            c["below_min_windows"] += 1
        else:
            key = str(ev).split(":", 1)[0].split("(", 1)[0].strip()[:40]
            c[key] += 1
    return c


def _build_run_summary_text(
    *,
    df: pd.DataFrame,
    scored: pd.DataFrame,
    ranked_by_primary: dict[str, pd.DataFrame],
    report_primaries: list[str],
    df_base: pd.DataFrame,
    out_dir: Path,
    elapsed_s: float,
    args: argparse.Namespace,
    data_dir: Path,
    baseline_subject: str,
    backends: list[str],
    output_paths: dict[str, str | None],
) -> str:
    lines: list[str] = []
    lines.append("Harmony dataset library exploration — RUN SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Elapsed: {elapsed_s:.1f}s")
    lines.append(f"data_dir: {data_dir}")
    multi = len(report_primaries) > 1
    rep_mode = (
        f"per-model ({len(report_primaries)} decoders)"
        if multi
        else f"single ({report_primaries[0]})"
    )
    lines.append(
        f"baseline_subject: {baseline_subject}  backends: {backends}  report_mode: {rep_mode}"
    )
    lines.append("")

    lines.append("Files")
    lines.append("-" * 72)
    lines.append(f"  Scanned: {len(df)}")
    err_c = _aggregate_error_counts(df)
    lines.append(f"  Parse/eval OK (full metrics): {err_c.get('ok', 0)}")
    lines.append(f"  Below min_windows:            {err_c.get('below_min_windows', 0)}")
    lines.append(f"  Failed / skipped:             {len(df) - err_c.get('ok', 0) - err_c.get('below_min_windows', 0)}")
    other = {k: v for k, v in err_c.items() if k not in ("ok", "below_min_windows")}
    if other:
        lines.append("  Error categories:")
        for k, v in sorted(other.items(), key=lambda x: -x[1]):
            lines.append(f"    {v:>4}  {k}")
    lines.append(f"  Baseline pool rows (in table): {df['in_baseline_pool'].sum() if 'in_baseline_pool' in df.columns else '—'}")
    if "erd_pattern" in df.columns:
        pat = (
            df["erd_pattern"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        lines.append(f"  ERD pattern counts (all rows): {pat}")
    if "erd_rest_leak_strength" in df.columns:
        leak = pd.to_numeric(df["erd_rest_leak_strength"], errors="coerce")
        leak_n = int((leak >= float(args.erd_rest_leak_threshold)).sum())
        lines.append(
            f"  REST ERD leak (>= {args.erd_rest_leak_threshold:g}): {leak_n} rows  | hard penalty={args.erd_rest_leak_penalty:g}"
        )
    lines.append("")

    base_ok = df_base[df_base["eval_error"].isna() | (df_base["eval_error"] == "")] if "eval_error" in df_base.columns else df_base
    if len(base_ok):
        lines.append("Baseline subset (successful rows) — AUC snapshot by decoder")
        lines.append("-" * 72)
        for pb in report_primaries:
            auc_col = f"{pb}__auc"
            if auc_col in base_ok.columns:
                s = pd.to_numeric(base_ok[auc_col], errors="coerce").dropna()
                if len(s):
                    lines.append(
                        f"  {pb:<12} auc median={s.median():.4f}  "
                        f"IQR=[{s.quantile(0.25):.4f}, {s.quantile(0.75):.4f}]  n={len(s)}"
                    )
        if "fisher_tangent_mu" in base_ok.columns:
            f = pd.to_numeric(base_ok["fisher_tangent_mu"], errors="coerce").dropna()
            if len(f):
                lines.append(f"  fisher_tangent_mu median={f.median():.4f}")

        # Optional: ERD laterality diagnostics (left vs right + contra vs ipsi).
        if "erd_bilateral_ratio" in base_ok.columns and "erd_bilateral_flag" in base_ok.columns:
            br = pd.to_numeric(base_ok["erd_bilateral_ratio"], errors="coerce").dropna()
            bf = base_ok["erd_bilateral_flag"]
            bf_rate = float(pd.to_numeric(bf, errors="coerce").fillna(0.0).mean()) if len(br) else float("nan")
            lines.append(
                f"  ERD bilateral_ratio median={br.median():.3f}  bilateral_rate={bf_rate:.2f}  (baseline subset)"
            )
        if "erd_contra_minus_ipsi" in base_ok.columns:
            ci = pd.to_numeric(base_ok["erd_contra_minus_ipsi"], errors="coerce").dropna()
            if len(ci):
                lines.append(f"  ERD contra_minus_ipsi median={ci.median():.3f}")
        lines.append("")

    for pb in report_primaries:
        ranked = ranked_by_primary[pb]
        if "composite_score" in ranked.columns:
            cs = pd.to_numeric(ranked["composite_score"], errors="coerce")
            pos = ranked[cs > 0]
        else:
            pos = ranked
        auc_col = f"{pb}__auc"
        lines.append(f"Proposals — decoder primary: {pb}")
        lines.append("-" * 72)
        lines.append(f"  Ranked rows: {len(ranked)}  with composite_score > 0: {len(pos)}")
        show = ranked.head(15)
        if len(show):
            lines.append("  Top 15 (composite | auc | modality | subject | filename)")
            for _, r in show.iterrows():
                fn = str(r.get("filename", ""))[:48]
                lines.append(
                    f"    {_fmt4(r.get('composite_score'))}  {_fmt4(r.get(auc_col))}  "
                    f"{str(r.get('modality', '')):8}  {str(r.get('subject', ''))[:10]:10}  {fn}"
                )
        lines.append("")

        if "erd_contra_minus_ipsi" in ranked.columns and "erd_bilateral_flag" in ranked.columns:
            top15 = ranked.head(15)
            ci = pd.to_numeric(top15["erd_contra_minus_ipsi"], errors="coerce").dropna()
            br = pd.to_numeric(top15["erd_bilateral_ratio"], errors="coerce").dropna() if "erd_bilateral_ratio" in top15.columns else None
            bf = top15["erd_bilateral_flag"] if "erd_bilateral_flag" in top15.columns else None
            bf_rate = float(pd.to_numeric(bf, errors="coerce").fillna(0.0).mean()) if bf is not None else float("nan")
            if len(ci):
                lines.append(f"  ERD laterality (top15): contra_minus_ipsi median={ci.median():.3f}  bilateral_rate={bf_rate:.2f}")
            elif br is not None and len(br):
                lines.append(f"  ERD laterality (top15): bilateral_ratio median={br.median():.3f}  bilateral_rate={bf_rate:.2f}")

    if multi:
        lines.append(
            "Each model above has its own composite_score and proposal_rank — compare files only within "
            "the same decoder. (Per-file K-fold already evaluated every selected backend.)"
        )
        lines.append("")

    lines.append("Output files")
    lines.append("-" * 72)
    for k, v in sorted(output_paths.items()):
        lines.append(f"  {k}: {v or '(not written)'}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _fmt4(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "   —   "
    try:
        return f"{float(x):7.3f}"
    except (TypeError, ValueError):
        return str(x)[:7].ljust(7)


def run_deep_stage(shortlist: list[str], models: list[str]) -> dict:
    from Utils.transfer_benchmark_core import build_session_dataset, run_session_heldout_benchmark

    include_erd = "xgb_cov_erd" in models
    include_beta = "xgb_cov" in models or "xgb_cov_erd" in models
    sessions = build_session_dataset(
        shortlist,
        include_beta_cov=include_beta,
        include_erd=include_erd,
    )
    return run_session_heldout_benchmark(model_names=models, sessions=sessions, print_importance=False)


def main() -> int:
    mne.set_log_level("WARNING")

    parser = argparse.ArgumentParser(description="Harmony offline dataset library exploration")
    parser.add_argument("--data-dir", type=str, default=None, help="Override config.DATA_DIR")
    parser.add_argument("--baseline-subject", type=str, default=None, help="Override TRAINING_SUBJECT for baseline pool")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for reports")
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline+screen",
        choices=("baseline+screen", "deep"),
        help="baseline+screen: full scan; deep: session-held-out on --shortlist-file",
    )
    parser.add_argument("--backends", type=str, default="mdm,xgb_cov", help="Comma-separated: mdm,xgb_cov,xgb_cov_erd")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-windows", type=int, default=256)
    parser.add_argument("--target-ambig", type=float, default=0.3)
    parser.add_argument("--score-threshold", type=float, default=0.5, help="For historical dominance on P(correct class)")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only first N paths after sort (debug)")
    parser.add_argument("--skew-penalty-lambda", type=float, default=0.5)
    parser.add_argument(
        "--erd-rest-leak-threshold",
        type=float,
        default=0.1,
        help="REST-class ERD leak threshold for hard penalty (strength units).",
    )
    parser.add_argument(
        "--erd-rest-leak-penalty",
        type=float,
        default=1.5,
        help="Hard penalty subtracted from composite_score when REST ERD leak exceeds threshold.",
    )
    parser.add_argument("--shortlist-file", type=str, default=None, help="One .xdf path per line for deep stage")
    parser.add_argument("--deep-models", type=str, default="mdm,xgb_cov", help="Models for deep stage")
    parser.add_argument("--proposal-max", type=int, default=200)
    parser.add_argument(
        "--exclude-substrings",
        type=str,
        default="OBS,old",
        help="Comma-separated tokens; skip files whose path or filename contains any (case-insensitive). "
        "Used for recursive discovery and for filtering --shortlist-file in deep stage.",
    )
    parser.add_argument(
        "--mi-arm",
        type=str,
        default="config",
        choices=("config", "right", "left", "any"),
        help="Match logs/*/config_snapshot.json ARM_SIDE: 'config' uses config.ARM_SIDE; "
        "'any' disables; missing snapshot or missing key still keeps file.",
    )
    parser.add_argument(
        "--include-errp",
        action="store_true",
        help="Include ERRP-style data (SELECT_ERRP_CHANNELS in snapshot or ERRP path under training_data).",
    )
    parser.add_argument(
        "--no-kfold-progress",
        action="store_true",
        help="Disable per-fold lines and the ‘features ready → k-fold’ line (still prints → per file and timing split).",
    )
    args = parser.parse_args()
    exclude_tpl = _parse_exclude_substrings(args.exclude_substrings)
    want_mi_arm, skip_errp = _resolve_exploration_filters(args)

    data_dir = Path(args.data_dir or config.DATA_DIR).expanduser().resolve()
    baseline_subject = args.baseline_subject or getattr(config, "TRAINING_SUBJECT", "PILOT007")
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.stage == "deep":
        if not args.shortlist_file:
            print("deep stage requires --shortlist-file", file=sys.stderr)
            return 2
        paths = []
        with open(args.shortlist_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(str(Path(line).expanduser().resolve()))
        paths = [p for p in paths if not path_matches_exclude(p, exclude_tpl)]
        deep_kept: list[str] = []
        for p in paths:
            reason, _ = exploration_filter_reason(p, mi_arm=want_mi_arm, skip_errp=skip_errp)
            if reason:
                print(f"[deep] skip {Path(p).name}: {reason}", file=sys.stderr)
            else:
                deep_kept.append(p)
        paths = deep_kept
        if len(paths) < 2:
            print(
                "Need at least 2 XDF paths after exclude + mi-arm/ERRP filters.",
                file=sys.stderr,
            )
            return 2
        models = _parse_backends(args.deep_models)
        print("=" * 78)
        print("Harmony — explore_dataset_library.py  [stage: deep]")
        print("=" * 78)
        print(f"  out_dir       : {out_dir}")
        print(f"  exclude_substrings (shortlist filter): {', '.join(exclude_tpl)}")
        print(f"  mi_arm filter : {want_mi_arm or 'any'}  |  ERRP: {'included' if args.include_errp else 'excluded'}")
        print(f"  shortlist     : {len(paths)} paths")
        for i, p in enumerate(paths, 1):
            print(f"    {i:>2}. {p}")
        print(f"  deep_models   : {models}")
        print("=" * 78)
        print("\nRunning session-held-out benchmark (see transfer_benchmark_core prints)…\n")
        results = run_deep_stage(paths, models)

        def _agg_json_safe(agg: dict) -> dict:
            out = {}
            for kk, vv in agg.items():
                if hasattr(vv, "tolist"):
                    out[kk] = vv.tolist()
                else:
                    out[kk] = vv
            return out

        agg_json = {k: _agg_json_safe(v["aggregate"]) for k, v in results.items()}
        write_json(out_dir / "deep_session_heldout.json", agg_json)
        print("\n" + "=" * 78)
        print("DEEP STAGE — aggregate metrics (also in deep_session_heldout.json)")
        print("=" * 78)
        for name, a in agg_json.items():
            cm = a.get("cm", [])
            print(f"\n  {name}")
            print(f"    acc={a.get('acc')}  auc={a.get('auc')}  bal_acc={a.get('bal_acc')}  "
                  f"macro_f1={a.get('macro_f1')}  brier={a.get('brier')}")
            if cm:
                print(f"    confusion [REST,MI true × pred]: {cm}")
        print("\n" + "=" * 78)
        return 0

    backends = _parse_backends(args.backends)
    report_primaries = report_primary_backends(backends)
    if want_mi_arm is None:
        mi_arm_banner = "any (disabled)"
    elif args.mi_arm == "config":
        mi_arm_banner = f"match {want_mi_arm!r} (from config.ARM_SIDE)"
    else:
        mi_arm_banner = f"match {want_mi_arm!r}"
    _print_banner(
        data_dir=data_dir,
        baseline_subject=baseline_subject,
        backends=backends,
        report_primaries=report_primaries,
        args=args,
        exclude_substrings=exclude_tpl,
        mi_arm_display=mi_arm_banner,
        include_errp=bool(args.include_errp),
    )

    # Discovery
    all_paths_raw = discover_xdf_files(data_dir, exclude_substrings=exclude_tpl)
    baseline_paths = set(
        list_baseline_xdf_files(data_dir, baseline_subject, exclude_substrings=exclude_tpl)
    )
    baseline_paths_list = sorted(baseline_paths)

    # Per your request: only evaluate EEG recordings from .../ses-*/eeg/*.xdf
    # (skip anything under .../training_data/), deduped by real resolved file path.
    eeg_paths: list[str] = []
    seen_realpaths: set[str] = set()
    for p in all_paths_raw:
        rp = str(Path(p).resolve())
        if rp in seen_realpaths:
            continue
        rp_sl = rp.replace("\\", "/").lower()
        if "/training_data/" in rp_sl:
            continue
        if Path(rp).parent.name.lower() != "eeg":
            continue
        seen_realpaths.add(rp)
        eeg_paths.append(rp)
    eeg_paths.sort()
    all_paths = eeg_paths

    if args.max_files > 0:
        all_paths = all_paths[: args.max_files]

    if not all_paths:
        print(f"No .xdf files found under {data_dir}", file=sys.stderr)
        return 1

    _print_discovery_stats(all_paths, baseline_paths, data_dir, baseline_subject)
    if baseline_paths_list:
        print(
            f"Baseline reference (training_data/{baseline_subject}/* .xdf): {len(baseline_paths_list)} files will be evaluated silently for reference stats.",
            flush=True,
        )

    print("\n[Per-file evaluation]")
    print("  Legend: OK = full metrics  |  LOW = below min_windows  |  ERR = load/segment/decoder failure")
    print("  XGB lines: top_gain = in-sample full-fit gain importances (see CSV column *__top_features_gain)")
    print("  → line = file started; feat/dec = load+features vs k-fold+importance wall time; checkpoint every 25 files")
    print("-" * 78)

    t_run0 = time.perf_counter()
    rows: list[dict] = []
    show_kfold_progress = not bool(args.no_kfold_progress)
    n_paths_total = len(all_paths)
    checkpoint_every = 25

    for i, p in enumerate(all_paths):
        t0 = time.perf_counter()
        meta = index_xdf_path(p, data_dir=data_dir, baseline_subject=baseline_subject)
        gate_reason, gate_extras = exploration_filter_reason(
            p, mi_arm=want_mi_arm, skip_errp=skip_errp
        )
        if gate_reason:
            r = {
                "path": p,
                "filename": Path(p).name,
                "eval_error": gate_reason,
                **gate_extras,
            }
            r["subject"] = meta.subject
            r["session_folder"] = meta.session_folder
            r["modality"] = meta.modality
            r["is_symlink"] = meta.is_symlink
            r["size_bytes"] = meta.size_bytes
            r["in_baseline_pool"] = meta.in_baseline_pool or (p in baseline_paths)
            rows.append(r)
            _print_file_result(
                i + 1,
                len(all_paths),
                r,
                report_primaries=report_primaries,
                elapsed_s=time.perf_counter() - t0,
            )
            continue
        print(f"  → [{i + 1}/{n_paths_total}] {Path(p).name}", flush=True)
        try:
            r = evaluate_path(
                p,
                backends=backends,
                n_splits=args.n_splits,
                min_windows=args.min_windows,
                target_ambig=args.target_ambig,
                score_threshold=args.score_threshold,
                laterality_arm=want_mi_arm,
                on_kfold_step=(
                    (lambda k, nk: print(f"    · k-fold {k}/{nk}", flush=True))
                    if show_kfold_progress
                    else None
                ),
                progress_after_features=show_kfold_progress,
            )
        except Exception as e:
            print(
                f"  [!] uncaught exception on {Path(p).name}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            r = {
                "path": p,
                "filename": Path(p).name,
                "eval_error": f"uncaught_exception: {type(e).__name__}: {e}",
            }
        r.update(gate_extras)
        r["subject"] = meta.subject
        r["session_folder"] = meta.session_folder
        r["modality"] = meta.modality
        r["is_symlink"] = meta.is_symlink
        r["size_bytes"] = meta.size_bytes
        r["in_baseline_pool"] = meta.in_baseline_pool or (p in baseline_paths)
        rows.append(r)
        _print_file_result(
            i + 1,
            len(all_paths),
            r,
            report_primaries=report_primaries,
            elapsed_s=time.perf_counter() - t0,
        )
        if n_paths_total and (i + 1) % checkpoint_every == 0:
            _print_eval_checkpoint(
                rows,
                i + 1,
                n_paths_total,
                t_run0,
                report_primaries=report_primaries,
            )

    if n_paths_total and (n_paths_total % checkpoint_every != 0) and rows:
        _print_eval_checkpoint(
            rows,
            n_paths_total,
            n_paths_total,
            t_run0,
            report_primaries=report_primaries,
        )

    # Evaluate baseline reference files (training_data) silently so ref stats work
    # even though we do not include them in the EEG-only scan.
    if baseline_paths_list:
        t_base0 = time.perf_counter()
        for p in baseline_paths_list:
            meta = index_xdf_path(p, data_dir=data_dir, baseline_subject=baseline_subject)
            gate_reason, gate_extras = exploration_filter_reason(p, mi_arm=want_mi_arm, skip_errp=skip_errp)
            if gate_reason:
                r = {
                    "path": p,
                    "filename": Path(p).name,
                    "eval_error": gate_reason,
                    **gate_extras,
                }
                r["subject"] = meta.subject
                r["session_folder"] = meta.session_folder
                r["modality"] = meta.modality
                r["is_symlink"] = meta.is_symlink
                r["size_bytes"] = meta.size_bytes
                r["in_baseline_pool"] = meta.in_baseline_pool or (p in baseline_paths)
                rows.append(r)
                continue
            r = evaluate_path(
                p,
                backends=backends,
                n_splits=args.n_splits,
                min_windows=args.min_windows,
                target_ambig=args.target_ambig,
                score_threshold=args.score_threshold,
                laterality_arm=want_mi_arm,
                on_kfold_step=None,
                progress_after_features=False,
            )
            r.update(gate_extras)
            r["subject"] = meta.subject
            r["session_folder"] = meta.session_folder
            r["modality"] = meta.modality
            r["is_symlink"] = meta.is_symlink
            r["size_bytes"] = meta.size_bytes
            r["in_baseline_pool"] = meta.in_baseline_pool or (p in baseline_paths)
            rows.append(r)
        print(
            f"Baseline reference eval finished in {time.perf_counter() - t_base0:.1f}s.",
            flush=True,
        )

    elapsed = time.perf_counter() - t_run0
    df = pd.DataFrame(rows)

    # Reference from baseline rows only
    base_mask = df["in_baseline_pool"].astype(bool) if "in_baseline_pool" in df.columns else pd.Series([False] * len(df))
    df_base = df[base_mask].copy()
    if df_base.empty:
        print(
            "WARNING: No files flagged in_baseline_pool; using successfully parsed files as weak reference.",
            file=sys.stderr,
        )
        ok = df["eval_error"].isna() | (df["eval_error"] == "")
        df_base = df[ok].copy()

    ref = reference_stats(df_base)

    baseline_ch = (
        df_base["n_channels"].dropna().astype(int).unique().tolist() if "n_channels" in df_base.columns else []
    )

    thr_tag = str(args.score_threshold).replace(".", "p")
    scored_by_primary: dict[str, pd.DataFrame] = {}
    for pb in report_primaries:
        scored_by_primary[pb] = add_scores_and_flags(
            df,
            ref,
            primary_backend=pb,
            baseline_channel_counts=baseline_ch,
            pool_subject=baseline_subject,
            skew_penalty_lambda=args.skew_penalty_lambda,
            frac_thresh_col_suffix=thr_tag,
            erd_rest_leak_threshold=args.erd_rest_leak_threshold,
            erd_rest_leak_penalty=args.erd_rest_leak_penalty,
        )

    if len(report_primaries) == 1:
        scored = scored_by_primary[report_primaries[0]]
    else:
        scored = merge_multi_primary_scored(df, scored_by_primary)

    ranked_by_primary = {
        pb: rank_proposals(scored_by_primary[pb], exclude_in_pool=True, min_composite=None)
        for pb in report_primaries
    }
    ranked_default = ranked_by_primary[report_primaries[0]]

    out_paths: dict[str, str | None] = {}
    csv_b, pq_b = write_df(df_base, out_dir, basename="baseline_summary")
    out_paths["baseline_summary.csv"] = csv_b
    out_paths["baseline_summary.parquet"] = pq_b
    csv_c, pq_c = write_df(scored, out_dir, basename="candidates")
    out_paths["candidates.csv"] = csv_c
    out_paths["candidates.parquet"] = pq_c

    csv_r, pq_r = write_df(ranked_default, out_dir, basename="ranked_proposals")
    out_paths["ranked_proposals.csv"] = csv_r
    out_paths["ranked_proposals.parquet"] = pq_r

    multi_report = len(report_primaries) > 1
    run_meta_base = {
        "data_dir": str(data_dir),
        "baseline_subject": baseline_subject,
        "backends": ",".join(backends),
        "report_primaries": report_primaries,
        "score_threshold": args.score_threshold,
    }

    if multi_report:
        index_lines = [
            "# Proposal lists — one file per model (rankings are not merged across decoders).",
            f"# Models in this run: {', '.join(report_primaries)}",
            "",
        ]
        for pb in report_primaries:
            index_lines.append(f"#   PROPOSED_ADDITIONS_{pb}.txt   ← recommendations for {pb} only")
        index_lines.append("")
        index_lines.append(
            f"ranked_proposals.csv / .parquet = same ordering as the first model in your --backends list "
            f"({report_primaries[0]}). Use ranked_proposals_<model>.csv for each model’s own ranking."
        )
        (out_dir / "PROPOSED_ADDITIONS.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        out_paths["PROPOSED_ADDITIONS.txt"] = str(out_dir / "PROPOSED_ADDITIONS.txt")

        for pb in report_primaries:
            run_meta = {**run_meta_base, "primary_backend": pb}
            path_pb = out_dir / f"PROPOSED_ADDITIONS_{pb}.txt"
            write_proposed_additions(
                ranked_by_primary[pb],
                path_pb,
                max_lines=args.proposal_max,
                primary_backend=pb,
                run_meta=run_meta,
            )
            out_paths[f"PROPOSED_ADDITIONS_{pb}.txt"] = str(path_pb)

            csv_rp, pq_rp = write_df(ranked_by_primary[pb], out_dir, basename=f"ranked_proposals_{pb}")
            out_paths[f"ranked_proposals_{pb}.csv"] = csv_rp
            out_paths[f"ranked_proposals_{pb}.parquet"] = pq_rp
    else:
        run_meta = {**run_meta_base, "primary_backend": report_primaries[0]}
        write_proposed_additions(
            ranked_default,
            out_dir / "PROPOSED_ADDITIONS.txt",
            max_lines=args.proposal_max,
            primary_backend=report_primaries[0],
            run_meta=run_meta,
        )
        out_paths["PROPOSED_ADDITIONS.txt"] = str(out_dir / "PROPOSED_ADDITIONS.txt")

    snap = config_snapshot()
    snap["data_dir"] = str(data_dir)
    snap["baseline_subject"] = baseline_subject
    snap["backends"] = backends
    snap["report_primaries"] = report_primaries
    snap["n_files_scanned"] = len(all_paths)
    snap["score_threshold"] = args.score_threshold
    snap["explore_mi_arm_arg"] = args.mi_arm
    snap["explore_mi_arm_effective"] = want_mi_arm
    snap["explore_include_errp"] = bool(args.include_errp)
    snap["elapsed_seconds_eval_loop"] = round(elapsed, 2)
    write_json(out_dir / "run_config_snapshot.json", snap)
    out_paths["run_config_snapshot.json"] = str(out_dir / "run_config_snapshot.json")

    write_json(out_dir / "baseline_reference_stats.json", {k: dict(v) for k, v in ref.items()})
    out_paths["baseline_reference_stats.json"] = str(out_dir / "baseline_reference_stats.json")

    summary_text = _build_run_summary_text(
        df=df,
        scored=scored,
        ranked_by_primary=ranked_by_primary,
        report_primaries=report_primaries,
        df_base=df_base,
        out_dir=out_dir,
        elapsed_s=elapsed,
        args=args,
        data_dir=data_dir,
        baseline_subject=baseline_subject,
        backends=backends,
        output_paths=out_paths,
    )
    write_run_summary(out_dir / "RUN_SUMMARY.txt", summary_text)
    out_paths["RUN_SUMMARY.txt"] = str(out_dir / "RUN_SUMMARY.txt")

    print("\n" + summary_text)
    print(f"\nAll artifacts written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
