"""
test_runtime_common_features.py

Guards the "silent decoder miscompute" bug class (Plan §6 #6). Example
commits:

  - 0931932 / 0076698  LedoitWolf was being fit on a covariance instead
                       of the raw window — _shrink_single_cov now
                       refuses to run with raw_window=None when
                       LEDOITWOLF is on.
  - 7816b24             Whitening missing on beta band — _build_xgb
                       features path; the recentering call must be
                       applied per-band.

This file locks in algebraic properties of the two single-cov helpers
that sit at the bottom of the decoder feature stack:

  - Utils/runtime_common.py:256-261  `_covariance_from_window`
  - Utils/runtime_common.py:264-286  `_shrink_single_cov`
  - Utils/runtime_common.py:289-300  `_tangent_with_fitted_ref_single`

Strategy: rather than freezing literal floats (which become opaque
maintenance burdens once the algorithm legitimately shifts), the
tests assert on algebraic invariants — symmetry, PSD-ness, trace
normalization, the LedoitWolf shrinkage equation
((1−λ)C + λ(tr C / n) I), and that the LedoitWolf path refuses to
run with raw_window=None (the post-0931932 contract).
"""

from __future__ import annotations

import numpy as np
import pytest

import config
import Utils.runtime_common as RC
from Utils.runtime_common import (
    _covariance_from_window,
    _shrink_single_cov,
    _tangent_with_fitted_ref_single,
)


# ─── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def wire_runtime_common(monkeypatch):
    """Drivers wire `runtime_common.config = config` and `runtime_common.model`
    at startup (file:35-37). Tests need to mimic that wiring."""
    monkeypatch.setattr(RC, "config", config)


@pytest.fixture
def fake_window():
    """Synthetic 15-channel, 1-second window at FS=512 with correlated
    structure so the covariance is well-defined."""
    rng = np.random.default_rng(2026)
    n_ch, n_t = 15, 512
    # Inject a low-rank correlated component so the cov isn't isotropic.
    mixing = rng.standard_normal((n_ch, 5))
    latent = rng.standard_normal((5, n_t))
    noise = rng.standard_normal((n_ch, n_t)) * 0.3
    return mixing @ latent + noise


# ─── _covariance_from_window ──────────────────────────────────────────────

class TestCovarianceFromWindow:
    def test_symmetric(self, fake_window):
        cov = _covariance_from_window(fake_window)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_trace_one(self, fake_window):
        cov = _covariance_from_window(fake_window)
        assert np.isclose(np.trace(cov), 1.0, atol=1e-10)

    def test_psd(self, fake_window):
        cov = _covariance_from_window(fake_window)
        eigs = np.linalg.eigvalsh(cov)
        # Numeric floor for tiny negative eigenvalues from float roundoff.
        assert eigs.min() > -1e-10

    def test_zero_window_falls_back_to_safe_trace(self):
        """The implementation clamps trace to 1e-12 when the raw covariance
        is zero (file:259-260) — division then produces zeros without
        blowing up."""
        zero_win = np.zeros((5, 512))
        cov = _covariance_from_window(zero_win)
        assert cov.shape == (5, 5)
        assert np.all(cov == 0)


# ─── _shrink_single_cov ───────────────────────────────────────────────────

class TestShrinkSingleCov:
    def test_classic_shrinkage_path_is_symmetric_and_psd(self, fake_window, monkeypatch):
        """With LEDOITWOLF=0 (the default), shrinkage uses the per-backend
        constant SHRINKAGE_PARAM_XGB / SHRINKAGE_PARAM_MDM."""
        monkeypatch.setattr(config, "LEDOITWOLF", 0)
        cov = _covariance_from_window(fake_window)
        shrunk = _shrink_single_cov(cov)
        assert shrunk.shape == cov.shape
        assert np.allclose(shrunk, shrunk.T, atol=1e-10)
        eigs = np.linalg.eigvalsh(shrunk)
        assert eigs.min() > 0, "Shrinkage should always produce a PD matrix"

    def test_ledoitwolf_path_requires_raw_window(self, fake_window, monkeypatch):
        """Post-0931932 contract: with LEDOITWOLF=1 the function MUST receive
        the raw window (n_channels x n_timepoints). Passing only the cov
        raises RuntimeError so the bad-LW-on-cov path can never recur
        (file:271-276)."""
        monkeypatch.setattr(config, "LEDOITWOLF", 1)
        cov = _covariance_from_window(fake_window)
        with pytest.raises(RuntimeError, match="raw EEG window"):
            _shrink_single_cov(cov, raw_window=None)

    def test_ledoitwolf_path_returns_convex_combination(self, fake_window, monkeypatch):
        """When raw_window is supplied, the result must satisfy the
        LedoitWolf shrinkage identity (1-λ)*cov + λ*(tr cov / n)*I for
        some λ ∈ [0, 1]. We don't assert on λ's value (LW chooses it from
        the data) but on the convex-combination form."""
        monkeypatch.setattr(config, "LEDOITWOLF", 1)
        cov = _covariance_from_window(fake_window)
        n = cov.shape[0]
        target = (np.trace(cov) / n) * np.eye(n)

        shrunk = _shrink_single_cov(cov, raw_window=fake_window)
        # Solve for λ from the diagonal: shrunk_ii = (1-λ)*cov_ii + λ*(tr/n)
        # → λ = (shrunk_ii - cov_ii) / (tr/n - cov_ii).
        # Pick a diagonal entry where cov_ii != tr/n to avoid 0/0.
        diag_idx = int(np.argmax(np.abs(np.diag(cov) - target[0, 0])))
        denom = target[diag_idx, diag_idx] - cov[diag_idx, diag_idx]
        lam = (shrunk[diag_idx, diag_idx] - cov[diag_idx, diag_idx]) / denom
        assert 0.0 <= lam <= 1.0, f"λ={lam} out of [0,1]"

        expected = (1 - lam) * cov + lam * target
        assert np.allclose(shrunk, expected, atol=1e-8)


# ─── _tangent_with_fitted_ref_single ──────────────────────────────────────

def _spd_matrix(n, *, seed=0):
    """Build a synthetic n×n SPD matrix to act as a tangent-space ref."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


class TestTangentWithFittedRefSingle:
    def test_mu_ref_used_for_mu_state(self, fake_window, monkeypatch):
        """state_name='mu' must look up `tangent_ref_mu` from the model
        bundle (file:293-294)."""
        cov = _covariance_from_window(fake_window)
        n = cov.shape[0]
        ref_mu = _spd_matrix(n, seed=1)
        bundle = {"model": object(), "scaler": object(), "label_to_bin": {0: 0, 1: 1},
                  "tangent_ref_mu": ref_mu}
        monkeypatch.setattr(RC, "model", bundle)
        out = _tangent_with_fitted_ref_single(cov, state_name="mu")
        # tangent_space returns the upper-triangular tangent vector for an
        # n×n SPD matrix → length n*(n+1)/2.
        assert out.shape == (n * (n + 1) // 2,)
        assert np.all(np.isfinite(out))

    def test_beta_state_uses_beta_ref(self, fake_window, monkeypatch):
        cov = _covariance_from_window(fake_window)
        n = cov.shape[0]
        ref_mu = _spd_matrix(n, seed=1)
        ref_beta = _spd_matrix(n, seed=2)
        bundle = {"model": object(), "scaler": object(), "label_to_bin": {0: 0, 1: 1},
                  "tangent_ref_mu": ref_mu, "tangent_ref_beta": ref_beta}
        monkeypatch.setattr(RC, "model", bundle)
        out_mu = _tangent_with_fitted_ref_single(cov, state_name="mu")
        out_beta = _tangent_with_fitted_ref_single(cov, state_name="beta")
        # Different reference matrices → different tangent vectors.
        assert not np.allclose(out_mu, out_beta)

    def test_missing_ref_raises(self, fake_window, monkeypatch):
        """If the bundle lacks the requested ref the call must hard-fail
        rather than silently fall back to identity (file:295-299, post-
        7816b24 spirit)."""
        cov = _covariance_from_window(fake_window)
        bundle = {"model": object(), "scaler": object(), "label_to_bin": {0: 0, 1: 1}}
        monkeypatch.setattr(RC, "model", bundle)
        with pytest.raises(RuntimeError, match="tangent_ref_mu"):
            _tangent_with_fitted_ref_single(cov, state_name="mu")

    def test_non_dict_model_raises(self, fake_window, monkeypatch):
        cov = _covariance_from_window(fake_window)
        monkeypatch.setattr(RC, "model", object())  # not a dict
        with pytest.raises(RuntimeError, match="XGBoost decoder bundle"):
            _tangent_with_fitted_ref_single(cov, state_name="mu")
