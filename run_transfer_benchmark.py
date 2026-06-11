"""
Entrypoint for threshold-free session-held-out transfer benchmark.

Example:
  python run_transfer_benchmark.py --models mdm,xgb_cov,xgb_cov_erd
"""

import os
import argparse

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import mne

from Utils.transfer_benchmark_core import (
    build_session_dataset,
    run_session_heldout_benchmark,
)


def _parse_models(raw: str):
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"mdm", "xgb_cov", "xgb_cov_erd"}
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported model(s): {bad}. Allowed: {sorted(allowed)}")
    return items


def main():
    parser = argparse.ArgumentParser(description="Threshold-free transfer benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="mdm,xgb_cov,xgb_cov_erd",
        help="Comma-separated models: mdm,xgb_cov,xgb_cov_erd",
    )
    parser.add_argument(
        "--no-importance",
        action="store_true",
        help="Disable feature-importance summaries for XGBoost models",
    )
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    models = _parse_models(args.models)

    include_erd = "xgb_cov_erd" in models
    include_beta_cov = "xgb_cov" in models or "xgb_cov_erd" in models
    sessions = build_session_dataset(include_beta_cov=include_beta_cov, include_erd=include_erd)

    run_session_heldout_benchmark(
        model_names=models,
        sessions=sessions,
        print_importance=(not args.no_importance),
    )


if __name__ == "__main__":
    main()

