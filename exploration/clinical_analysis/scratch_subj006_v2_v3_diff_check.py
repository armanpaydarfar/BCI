"""Diff V2 vs V3 per-trial npz arrays for CLIN_SUBJ_006.

Reports, per session per (cluster, marker) key:
  * whether per_trial_pct arrays are byte-identical
  * shape (n_trials, n_time) on each side
  * channels list on each side
  * if differing: max|V3-V2| and the median trace scalar over (1, t_end)
    that drives D1

Read-only diagnostic; writes to stdout only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


PER_TRIAL = Path(
    r"C:\Users\arman\Pictures\clin_analysis_subj006_motor15_input\per_trial"
)
SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
KEYS = ("contra_mi", "contra_rest", "bilat_mi", "bilat_rest",
        "ipsi_mi", "ipsi_rest")


def _load(npz_path):
    z = np.load(npz_path)
    present = [k for k in str(z["keys"]).split(",") if k]
    out = {}
    for key in present:
        out[key] = {
            "ptp": np.asarray(z[f"{key}__ptp"], dtype=np.float64),
            "times": np.asarray(z[f"{key}__times"], dtype=np.float64),
            "channels": [c for c in str(z[f"{key}__channels"]).split(",") if c],
        }
    return out


def _scalar_window_median(ptp, times):
    """Median across trials of the per-trial trace, then median over
    [1, t_max] — this is the D1 substrate per evaluate_erd_quality:282-292
    (scalar_of_trace of med_mi over scalar_mask)."""
    med = np.median(ptp, axis=0)
    smask = times >= 1.0
    return float(np.median(med[smask])) if smask.any() else float("nan")


def main():
    for sess in SESSIONS:
        print(f"\n=== {SUBJECT} / {sess} ===")
        v2 = _load(PER_TRIAL / f"{SUBJECT}_{sess}_V2.npz")
        v3 = _load(PER_TRIAL / f"{SUBJECT}_{sess}_V3.npz")
        for key in KEYS:
            a = v2.get(key)
            b = v3.get(key)
            if a is None or b is None:
                print(f"  {key:14s}  one side missing  (V2={a is None}, "
                      f"V3={b is None})")
                continue
            same_shape = a["ptp"].shape == b["ptp"].shape
            same_chs = a["channels"] == b["channels"]
            if same_shape and np.array_equal(a["ptp"], b["ptp"]):
                eq = "byte-identical"
                diff = "—"
            else:
                if same_shape:
                    md = float(np.max(np.abs(b["ptp"] - a["ptp"])))
                    diff = f"max|Δ|={md:.3f}%"
                else:
                    diff = "shape mismatch"
                eq = "DIFFER"
            mv2 = _scalar_window_median(a["ptp"], a["times"])
            mv3 = _scalar_window_median(b["ptp"], b["times"])
            print(f"  {key:14s}  {eq:16s}  {diff:18s}  "
                  f"V2 shape={a['ptp'].shape} chs[{len(a['channels'])}]  "
                  f"V3 shape={b['ptp'].shape} chs[{len(b['channels'])}]")
            print(f"    median-scalar (drives D1):  V2={mv2:+.3f}%  "
                  f"V3={mv3:+.3f}%")


if __name__ == "__main__":
    main()
