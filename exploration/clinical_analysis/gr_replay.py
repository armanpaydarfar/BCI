"""Offline replay of Kumar 2024 GR + Zanini 2018 RA per CLIN session.

Analysis-only. Replicates the math at `Utils/runtime_common.py:307-336`
(`_adaptive_recenter_cov`, Kumar Eq. 8/9) and Zanini 2018
`Zanini_2018_TBME_TransferLearning_Riemannian_BCI.pdf` Eq. (9) batch
affine transformation without importing the Tier-2 runtime module. The
runtime module is READ-ONLY per CLAUDE.md.

Two recentering modes are exposed:

1. `GRState` + `gr_apply` — online Kumar 2024 GR.

   The runtime implementation (cited verbatim by line range):

       if counter == 0 or Prev_T is None:
           Prev_T = cov                                # line 328-329
       T_test = geodesic_riemann(Prev_T, cov, 1 / (counter + 1))   # 330
       T_invsqrtm = invsqrtm(Prev_T)                              # 331
       cov_rec = T_invsqrtm @ cov @ T_invsqrtm.T                  # 332
       if bool(update_recentering):                               # 333
           Prev_T = T_test                                        # 334
           counter += 1                                           # 335
       return cov_rec                                             # 336

   Key invariant: the whitener uses `Prev_T` (= T_{i-1}), NOT the freshly
   computed `T_test`. State update happens AFTER whitening (the
   alignment matrix lags the update by one trial). This is Kumar 2024
   Eq. (8)/(9) per `Kumar_2024_PNASNexus_TransferLearning.pdf` p. 12.

   In the runtime, `Prev_T`/`counter` are module globals reset only on
   process start (per `ExperimentDriver_Online.py:103-110`). With
   `config.SAVE_ADAPTIVE_T = False` (the default for CLIN_SUBJ_003..008)
   and no `adaptive_T.pkl` on disk, the runtime cold-starts GR at the
   beginning of every ONLINE_* driver process (= per run). The pass-2
   ablation reset per session; pass-2-fix resets per run (matching the
   runtime cadence).

2. `zanini_ra_apply` — batch Zanini 2018 RA per Eq. (9).

   For each session (or each run within a session), compute the
   Riemannian (Karcher) mean R of the reference-state covariances (rest
   trials, per Zanini 2018 §IV "In the MI dataset, reference EEG
   signals are directly available" — the rest-period covariances serve
   as R), then whiten every trial:

       C_i^RA = R^{-1/2} · C_i · R^{-1/2}

   This is a one-shot batch transform: every trial sees the same R,
   computed over the run's rest-only covariances. No state lag, no
   per-trial update. The headline difference vs Kumar GR is:
   batch-vs-online and rest-only-vs-all-trials.
"""

from __future__ import annotations

import numpy as np
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann


class GRState:
    """Per-(session|run) GR state (Prev_T, counter).

    Replicates the module-global pair at `Utils/runtime_common.py:53-54`
    but scoped to one instance per reset boundary. The pass-2-fix ablation
    instantiates one `GRState` per ONLINE_* run (matching runtime
    behaviour: `ExperimentDriver_Online.py:103-110` cold-starts the pair
    at every driver process invocation when `SAVE_ADAPTIVE_T = False`).
    """

    __slots__ = ("Prev_T", "counter")

    def __init__(self) -> None:
        self.Prev_T: np.ndarray | None = None
        self.counter: int = 0


def gr_apply(state: GRState, cov: np.ndarray) -> np.ndarray:
    """Apply Kumar 2024 GR recentering to one covariance matrix.

    Mirrors `Utils/runtime_common.py:328-335` for the mu-band branch.
    Mutates `state.Prev_T` and `state.counter` after computing the
    whitened output. See module docstring for the math.

    Args:
        state: per-reset-boundary GR state. First call initialises
            `state.Prev_T = cov` (matching runtime line 328-329).
        cov: shape (C, C) trial covariance matrix.

    Returns:
        Whitened covariance shape (C, C) = T_{i-1}^{-1/2} @ cov @
        T_{i-1}^{-1/2}.T. Returned BEFORE the state update so the
        whitener lags one trial (Kumar Eq. 8/9).
    """
    if state.counter == 0 or state.Prev_T is None:
        state.Prev_T = cov
    T_test = geodesic_riemann(state.Prev_T, cov, 1.0 / (state.counter + 1))
    T_invsqrtm = invsqrtm(state.Prev_T)
    cov_rec = T_invsqrtm @ cov @ T_invsqrtm.T
    state.Prev_T = T_test
    state.counter += 1
    return cov_rec


def zanini_ra_apply(
    covs: np.ndarray, labels: np.ndarray, *,
    reference: str = "rest",
    rest_label: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Zanini 2018 batch RA per Eq. (9).

    Compute the Karcher mean R of the reference-state covariances in
    `covs`, then whiten every covariance through R^{-1/2}:

        C_i^RA = R^{-1/2} @ C_i @ R^{-1/2}

    Args:
        covs: shape (N, C, C) trial covariances (chronological within a
            single reset boundary — session or run).
        labels: shape (N,) class labels (100 = REST, 200 = MI).
        reference: "rest" → R = Karcher mean of rest-only covariances
            (Zanini 2018 §IV recommendation for the MI dataset, where
            "reference EEG signals are directly available" between
            trials); "all" → R = Karcher mean over all trials.
        rest_label: integer code for the reference (rest) class.

    Returns:
        (covs_ra, R) — whitened covariances and the reference matrix.
        Raises ValueError if no rest covariances are present and
        `reference == "rest"`.
    """
    if reference == "rest":
        ref_covs = covs[labels == rest_label]
        if len(ref_covs) == 0:
            raise ValueError(
                "zanini_ra_apply: no rest covariances available "
                f"(label={rest_label}); set reference='all' to use the "
                "full-batch Karcher mean instead."
            )
    elif reference == "all":
        ref_covs = covs
    else:
        raise ValueError(f"Unknown reference mode: {reference!r}")

    # pyriemann.mean_riemann implements the Riemannian (Karcher) mean
    # via the same `arg min ∑ d²(P_i, P)` formulation as Zanini Eq. (4).
    R = mean_riemann(ref_covs)
    R_invsqrtm = invsqrtm(R)
    out = np.stack(
        [R_invsqrtm @ c @ R_invsqrtm for c in covs], axis=0,
    )
    return out, R
