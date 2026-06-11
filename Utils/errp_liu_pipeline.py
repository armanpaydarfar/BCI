"""
Utils/errp_liu_pipeline.py — Liu et al. (2025) CCA feature pipeline and diagonal LDA.

Implements the decoding pipeline described in Liu, Iwane et al. 2025
§Methods ("Decoding the presence/absence of ErrP in the BCI group"):

    bandpassed epoch [200, 800] ms
      → template-based CCA spatial filter (top 3 components)
      → per-component temporal features (decimated to 32 Hz, ~20 samples)
      → per-component Welch PSD at 4, 6, 8, 10 Hz
      → concatenate (3 × 24 = 72 features)
      → min-max normalise per feature
      → classifier (diagonal LDA here; XGB is built elsewhere)

Design decisions vs the paper:
- Single class-contrast template (error_mean − correct_mean). This is the
  most discriminative single projection for template CCA; stacking
  [T_err; T_corr] gives an equivalent subspace, but the difference is
  sparser and produces cleaner top-3 directions in practice.
- Integer decimation via `scipy.signal.resample_poly(up=1, down=16)`
  (512 Hz → 32 Hz). Kaiser-windowed anti-alias is the scipy default.
- Welch PSD uses the full epoch as one segment (`nperseg=n_samples`).
  At 600 ms this gives ~1.67 Hz bin spacing; the 4/6/8/10 Hz targets are
  pulled by nearest-bin index, which we record in `feature_spec` so
  reproduction is deterministic.
- Regularised inverse square roots via symmetric eigendecomposition with
  a trace-relative floor, so CAR rank-deficient inputs (one zero
  eigenvalue by construction) stay well-conditioned.

Reference: Liu, Iwane et al. (2025). doi:10.1101/2025.04.26.650792
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, welch


# =============================================================================
# Numerical helper — shared with errp_alignment
# =============================================================================

def _inv_sqrt_psd(C: np.ndarray, shrinkage: float = 1e-6) -> np.ndarray:
    """Inverse matrix square root of a symmetric (PSD-ish) matrix.

    Trace-relative eigenvalue floor keeps CAR rank-deficient inputs
    well-conditioned. Returns float64 regardless of input dtype.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {C.shape}")
    n = C.shape[0]
    Cs = 0.5 * (C.astype(np.float64, copy=False) + C.astype(np.float64, copy=False).T)
    lam_floor = shrinkage * float(np.trace(Cs)) / n
    if not np.isfinite(lam_floor) or lam_floor <= 0:
        lam_floor = 1e-12
    w, V = np.linalg.eigh(Cs)
    w = np.clip(w, lam_floor, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


# =============================================================================
# LiuCCAFeaturizer — template-based CCA + 72-dim feature vector
# =============================================================================

class LiuCCAFeaturizer:
    """Template-CCA spatial filter + temporal + spectral features.

    Parameters
    ----------
    n_components : int
        Number of CCA components retained (Liu paper uses 3).
    fs : int
        Input sampling rate in Hz (Liu data is 512 Hz).
    target_fs : int
        Decimated sampling rate for temporal features (Liu paper uses 32 Hz).
    psd_freqs : tuple of float
        PSD probe frequencies in Hz (Liu paper uses 4, 6, 8, 10 Hz).
    shrinkage : float
        Trace-relative eigenvalue floor for CCA whitening.

    Attributes set by ``fit``
    -------------------------
    cca_weights_ : ndarray, shape (n_channels, n_components)
        Left spatial filter `W_X` — apply as `W_X.T @ epoch`.
    template_ : ndarray, shape (n_channels, n_samples)
        Class-contrast template used to build the CCA target.
    feat_min_, feat_range_ : ndarray, shape (n_features,)
        Min-max normaliser parameters, fit on training-fold feature vectors.
    psd_bin_idx_ : ndarray, shape (len(psd_freqs),)
        Welch PSD bin indices nearest to each probe frequency; recorded
        so reproduction is bin-exact.
    """

    def __init__(
        self,
        n_components: int = 3,
        fs: int = 512,
        target_fs: int = 32,
        psd_freqs: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0),
        shrinkage: float = 1e-6,
    ) -> None:
        if fs % target_fs != 0:
            raise ValueError(
                f"fs ({fs}) must be an integer multiple of target_fs ({target_fs}) "
                f"for polyphase decimation."
            )
        self.n_components = int(n_components)
        self.fs = int(fs)
        self.target_fs = int(target_fs)
        self.psd_freqs = tuple(float(f) for f in psd_freqs)
        self.shrinkage = float(shrinkage)
        self._down = int(fs // target_fs)

        self.cca_weights_: np.ndarray | None = None
        self.template_: np.ndarray | None = None
        self.feat_min_: np.ndarray | None = None
        self.feat_range_: np.ndarray | None = None
        self.psd_bin_idx_: np.ndarray | None = None
        self.n_temporal_: int | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,  # accepted for sklearn-style signatures; unused
    ) -> "LiuCCAFeaturizer":
        """Fit template, CCA spatial filter, and feature normaliser."""
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if X.ndim != 3:
            raise ValueError(f"X must be (n_trials, n_channels, n_samples), got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if not (set(np.unique(y).tolist()) <= {0, 1}):
            raise ValueError(f"y must be binary {{0, 1}}, got labels {np.unique(y)}")

        T_err = X[y == 1].mean(axis=0)
        T_corr = X[y == 0].mean(axis=0)
        self.template_ = (T_err - T_corr).astype(np.float64)

        self.cca_weights_ = self._fit_cca_spatial_filter(X, self.template_)

        feats = self._extract_features(X, self.cca_weights_, record_bins=True)
        self.feat_min_ = feats.min(axis=0)
        feat_max = feats.max(axis=0)
        rng = feat_max - self.feat_min_
        self.feat_range_ = np.where(rng > 1e-12, rng, 1.0).astype(np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply CCA spatial filter, extract features, apply normaliser."""
        if self.cca_weights_ is None or self.feat_min_ is None:
            raise RuntimeError("LiuCCAFeaturizer must be fit before transform.")
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be (n_trials, n_channels, n_samples), got {X.shape}")
        feats = self._extract_features(X, self.cca_weights_, record_bins=False)
        feats = (feats - self.feat_min_) / self.feat_range_
        return feats.astype(np.float32, copy=False)

    def fit_transform(self, X, y, groups=None):
        self.fit(X, y, groups=groups)
        return self.transform(X)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _fit_cca_spatial_filter(self, X: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Template-based CCA via whitened cross-covariance SVD.

        Treats each time sample as an observation (n_samples_total = n_trials *
        n_samples) and each channel as a feature. Returns the top-``k`` left
        directions in the original (un-whitened) channel space, so the filter
        can be applied as ``W_X.T @ epoch``.

        The covariance matrices (n_channels × n_channels) are small; we
        compute them by reducing over trials and samples with ``einsum`` on
        the original float32 X, accumulating in float64. This avoids the
        ~2 GiB float64 flattened copies (``X_flat``, ``T_rep`` and their
        centered versions) that otherwise blow past the memory budget on
        the 32-subject pooled fold.
        """
        n_trials, n_ch, n_samp = X.shape
        N = n_trials * n_samp

        # Per-channel mean over all observations (trials × samples).
        mean_X = X.mean(axis=(0, 2), dtype=np.float64)                 # (ch,)
        T64    = np.asarray(template, dtype=np.float64)                # (ch, samples)
        T_mean = T64.mean(axis=1)                                      # (ch,)
        T_cent = T64 - T_mean[:, None]                                 # (ch, samples)

        # Cxx = (X_c @ X_c.T) / (N-1).  Identity:  X_c X_c.T = XX.T - N μμ.T
        XX   = np.einsum("tcs,tds->cd", X, X, dtype=np.float64)        # (ch, ch)
        Cxx  = (XX - N * np.outer(mean_X, mean_X)) / max(N - 1, 1)

        # T_rep centered across N columns is T_cent tiled n_trials times,
        # so T_c T_c.T = n_trials · T_cent T_cent.T.
        Cyy  = n_trials * (T_cent @ T_cent.T) / max(N - 1, 1)

        # Cxy = X_c @ T_c.T / (N-1).  Since Σ_s T_cent[c',s] = 0, the mean_X
        # term cancels and we only need the per-sample trial-sum of X.
        sum_X_s = X.sum(axis=0, dtype=np.float64)                      # (ch, samples)
        Cxy = (sum_X_s @ T_cent.T) / max(N - 1, 1)

        Cxx_is = _inv_sqrt_psd(Cxx, shrinkage=self.shrinkage)
        Cyy_is = _inv_sqrt_psd(Cyy, shrinkage=self.shrinkage)

        K = Cxx_is @ Cxy @ Cyy_is
        U, _, _ = np.linalg.svd(K, full_matrices=False)
        k = self.n_components
        W_X = Cxx_is @ U[:, :k]
        return W_X.astype(np.float64)

    def _extract_features(
        self,
        X: np.ndarray,
        W_X: np.ndarray,
        record_bins: bool,
    ) -> np.ndarray:
        """Project via CCA, decimate to target_fs, concat temporal + PSD-at-probes."""
        # einsum with explicit float64 accumulator gives us a (n_trials, k,
        # n_samples) float64 output without a full-size float64 copy of X —
        # projected is ~k/n_channels the size of X, so the peak footprint
        # here is dominated by the small output, not the input.
        projected = np.einsum("ck,tcs->tks", W_X, X, dtype=np.float64)

        temporal = resample_poly(projected, up=1, down=self._down, axis=-1)
        if self.n_temporal_ is None:
            self.n_temporal_ = int(temporal.shape[-1])
        elif temporal.shape[-1] != self.n_temporal_:
            raise RuntimeError(
                f"Decimated length changed ({temporal.shape[-1]} vs {self.n_temporal_}) — "
                f"input epoch length must be constant."
            )

        n_per = projected.shape[-1]
        freqs, psd = welch(
            projected,
            fs=self.fs,
            nperseg=n_per,
            noverlap=0,
            axis=-1,
            scaling="density",
        )
        if record_bins or self.psd_bin_idx_ is None:
            self.psd_bin_idx_ = np.array(
                [int(np.argmin(np.abs(freqs - f))) for f in self.psd_freqs],
                dtype=int,
            )
        psd_feats = psd[:, :, self.psd_bin_idx_]

        n_trials = X.shape[0]
        feats = np.concatenate(
            [temporal.reshape(n_trials, -1), psd_feats.reshape(n_trials, -1)],
            axis=1,
        )
        return feats.astype(np.float64)


# =============================================================================
# DiagonalLDA — closed-form posterior from the paper's Eq. 1
# =============================================================================

class DiagonalLDA:
    """Gaussian naive-Bayes / diagonal LDA classifier for binary ErrP decoding.

    log p(c | x) ∝ -½ Σ_i (x_i − μ_{c,i})² / σ_i²  +  log π_c

    Uses a pooled diagonal variance (single σ_i² per feature, shared across
    classes), matching Liu et al. Eq. 1. Returned probabilities are
    softmaxed over the two classes.
    """

    def __init__(self, var_floor: float = 1e-12) -> None:
        self.var_floor = float(var_floor)
        self.mu_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiagonalLDA":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if not (set(np.unique(y).tolist()) <= {0, 1}):
            raise ValueError(f"y must be binary {{0, 1}}, got labels {np.unique(y)}")

        d = X.shape[1]
        self.mu_ = np.zeros((2, d), dtype=np.float64)
        self.prior_ = np.zeros(2, dtype=np.float64)
        for c in (0, 1):
            mask = (y == c)
            n_c = int(mask.sum())
            if n_c == 0:
                raise ValueError(f"Class {c} has zero training samples.")
            self.mu_[c] = X[mask].mean(axis=0)
            self.prior_[c] = n_c / float(X.shape[0])

        # Pooled diagonal variance: residual² around each sample's own class mean.
        residuals = X - self.mu_[y]
        n = X.shape[0]
        self.var_ = (residuals ** 2).sum(axis=0) / max(n - 2, 1)
        self.var_ = np.maximum(self.var_, self.var_floor)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None:
            raise RuntimeError("DiagonalLDA must be fit before predict_proba.")
        X = np.asarray(X, dtype=np.float64)
        log_lik = np.empty((X.shape[0], 2), dtype=np.float64)
        for c in (0, 1):
            sq = ((X - self.mu_[c]) ** 2) / self.var_
            log_lik[:, c] = -0.5 * sq.sum(axis=1) + np.log(self.prior_[c])
        log_lik -= log_lik.max(axis=1, keepdims=True)
        exp_ll = np.exp(log_lik)
        return exp_ll / exp_ll.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
