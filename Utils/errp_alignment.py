"""
Utils/errp_alignment.py — Euclidean Alignment for cross-subject ErrP transfer.

He & Wu (2019) — "Transfer Learning for Brain-Computer Interfaces: A
Euclidean Space Data Alignment Approach". Each subject's covariance
distribution is recentered to the identity by left- and right-multiplying
each epoch's covariance with the inverse square root of the subject's
mean covariance. Equivalently — and this is the formulation used here —
each raw epoch X is replaced by `R^{-1/2} X` where R is the per-subject
mean covariance, since `cov(R^{-1/2} X) = R^{-1/2} cov(X) R^{-1/2}`.

Used by the cross-subject ErrP pipeline to remove inter-subject spatial
and amplitude variability before xDAWN / Riemannian classification, both
during LOSO training (per training and held-out subject) and at runtime
(30–60 s session-start bootstrap).

Numerical: matrix square root is computed in float64 via symmetric
eigendecomposition with a trace-relative shrinkage on the eigenvalues to
keep the operator well-conditioned even when channels are nearly linearly
dependent (e.g. after CAR rereferencing).
"""

from __future__ import annotations

import numpy as np


def _mean_covariance_euclidean(epochs: np.ndarray) -> np.ndarray:
    """
    Arithmetic (Euclidean) mean of per-epoch sample covariances.

    Each per-epoch covariance is `(X X^T) / (n_samples - 1)`. Returns the
    average across epochs as a (n_channels, n_channels) symmetric matrix
    in float64.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must have shape (n_epochs, n_channels, n_samples), "
            f"got {epochs.shape}"
        )
    n_epochs, n_ch, n_samp = epochs.shape
    if n_epochs < 1:
        raise ValueError("Need at least one epoch to fit an EA reference.")

    Xd = epochs.astype(np.float64, copy=False)
    # Vectorised over epochs: covs shape (n_epochs, n_ch, n_ch).
    covs = np.einsum("ect,edt->ecd", Xd, Xd) / float(n_samp - 1)
    return covs.mean(axis=0)


def _inv_sqrt_psd(C: np.ndarray, shrinkage: float = 1e-6) -> np.ndarray:
    """
    Inverse matrix square root of a symmetric (PSD-ish) matrix.

    Uses `eigh` and clips eigenvalues from below at `shrinkage * tr(C) / n`
    so that rank-deficient inputs (CAR collapses one eigenvalue to ~0) do
    not blow up the inverse. Returns a symmetric float64 matrix.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {C.shape}")
    n = C.shape[0]
    Cs = 0.5 * (C + C.T)  # enforce symmetry against floating-point drift
    lam_floor = shrinkage * float(np.trace(Cs)) / n
    if not np.isfinite(lam_floor) or lam_floor <= 0:
        lam_floor = 1e-12
    w, V = np.linalg.eigh(Cs)
    w = np.clip(w, lam_floor, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def fit_ea_reference(epochs: np.ndarray, shrinkage: float = 1e-6) -> np.ndarray:
    """
    Fit a Euclidean Alignment reference matrix from one subject's epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Unlabelled (or labelled, label is irrelevant) EEG epochs from one
        subject. He & Wu 2019 recommends 30–60 s of any task-irrelevant
        EEG; in our LOSO training we use all that subject's labelled
        epochs since alignment is unsupervised.
    shrinkage : float
        Trace-relative floor on the mean-covariance eigenvalues
        (default 1e-6). Required when CAR has reduced channel rank by 1.

    Returns
    -------
    ref_inv_sqrt : ndarray, shape (n_channels, n_channels), float64
        `R^{-1/2}` where R is the mean covariance across the subject's
        epochs. Apply via `apply_ea(...)` to map subsequent epochs into
        the identity-centred frame.
    """
    R = _mean_covariance_euclidean(epochs)
    return _inv_sqrt_psd(R, shrinkage=shrinkage)


def apply_ea(epochs: np.ndarray, ref_inv_sqrt: np.ndarray) -> np.ndarray:
    """
    Apply a fitted EA reference to a batch of epochs.

    Implements `X_aligned = R^{-1/2} @ X` per epoch, which has the property
    `cov(X_aligned) = R^{-1/2} cov(X) R^{-1/2}`. Subjects whose epochs
    were used to fit `ref_inv_sqrt` will have their mean covariance
    mapped to the identity; subjects who did not contribute to the fit
    are still left-multiplied by the same operator (this is the runtime
    case — bootstrap data from the new subject produces a *new*
    reference per session).

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
    ref_inv_sqrt : ndarray, shape (n_channels, n_channels)

    Returns
    -------
    aligned : ndarray, same shape and dtype as `epochs`.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must have shape (n_epochs, n_channels, n_samples), "
            f"got {epochs.shape}"
        )
    if ref_inv_sqrt.ndim != 2 or ref_inv_sqrt.shape[0] != ref_inv_sqrt.shape[1]:
        raise ValueError(
            f"ref_inv_sqrt must be square, got {ref_inv_sqrt.shape}"
        )
    if epochs.shape[1] != ref_inv_sqrt.shape[0]:
        raise ValueError(
            f"channel count mismatch: epochs has {epochs.shape[1]} ch, "
            f"ref_inv_sqrt expects {ref_inv_sqrt.shape[0]}"
        )
    # einsum keeps everything contiguous; cast back to original dtype.
    aligned = np.einsum("cd,edt->ect", ref_inv_sqrt.astype(np.float64), epochs.astype(np.float64))
    return aligned.astype(epochs.dtype, copy=False)
