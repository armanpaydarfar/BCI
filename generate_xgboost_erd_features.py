"""
XGBoost branch using ERD-style features only.

Backbone (preserved up to windowed EEG extraction):
EEG -> temporal filtering -> segmentation -> (optional CSD) -> ERD features

ERD feature formulation (online-compatible):
For each trial window (analysis segment) and for each channel:
  - compute bandpower P_win in ERD bands via FFT on the current window
  - compute bandpower P_base using a causal baseline window just before cue:
      [ts_start - baseline_duration, ts_start]
  - ERD feature = log(P_win / P_base) in log-power space
Features are tabular: (n_channels * n_bands) per window.
"""

import os
import pickle
import numpy as np

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import mne

import config
import Generate_Riemannian_adaptive as base
from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd
from Utils.xgb_train_eval import train_xgb_dual_thresholds


def main():
    mne.set_log_level("WARNING")

    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ]
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")

    apply_csd = bool(getattr(config, "SURFACE_LAPLACIAN_TOGGLE", False))
    compute_erd = True
    print(f"[erd-only] APPLY_CSD(surface laplacian)={apply_csd}")

    all_feats = []
    all_labels = []

    for xdf_path in xdf_files:
        print(f"\n📂 Processing file: {xdf_path}")
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)
        segments, labels, erd_feats, _ch_names = segment_and_extract_cov_erd(
            eeg_stream,
            marker_stream,
            compute_erd=compute_erd,
            apply_csd=apply_csd,
        )
        if erd_feats is None or len(erd_feats) == 0:
            raise RuntimeError("ERD feature extraction returned no samples.")
        all_feats.append(erd_feats)
        all_labels.append(labels)

    X = np.concatenate(all_feats)
    y = np.concatenate(all_labels)

    model_bundle = train_xgb_dual_thresholds(
        X=X,
        labels=y,
        feature_tag="erd_bands_logratio",
        n_splits=int(getattr(config, "N_SPLITS", 8)),
        target_ambig=float(getattr(config, "TARGET_AMBIG", 0.20)),
    )

    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)
    out_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_xgb_erd_features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"✅ Saved ERD-only covariance-free XGBoost model to: {out_path}")


if __name__ == "__main__":
    main()

