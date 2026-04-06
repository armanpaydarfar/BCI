"""
rbnnet_model.py
---------------
PyTorch implementation of RBNNet for EEG Motor Imagery classification.

Two architectures are provided:

  RBNNet (single-band) — faithful to Liu et al. NER 2023:
    EEG (1 band) → covariance → {BiMap-RBN-ReEig}×n_blocks → {BiMap-RBN}
                             → LogEig → Linear(n_features, 2)

  DualBandRBNNet — extension for independent mu + beta processing:
    C_mu   → RBNNetEncoder → LogEig → LayerNorm ] concat → Linear(2*n_features, 2)
    C_beta → RBNNetEncoder → LogEig → LayerNorm ]

    Per-stream LayerNorm addresses inter-band scale differences that arise from
    mu and beta covariances having different eigenvalue magnitudes. It is applied
    after LogEig (in Euclidean space), so it does not violate SPD manifold geometry.
    Single-band RBNNet omits LayerNorm, remaining faithful to the paper.

References:
  Liu, Kumar, Alawieh, Carnahan & Millan. NER 2023.
    "On Transfer Learning for Naive Brain Computer Interface Users."
  Huang & Van Gool. AAAI 2017. "A Riemannian Network for SPD Matrix Learning."
  Brooks et al. NeurIPS 2019. "Riemannian Batch Normalization for SPD Neural Networks."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: matrix functions via eigendecomposition
# ---------------------------------------------------------------------------

_JITTER = 1e-6

def _sym_eigh(A):
    """Eigendecompose a symmetric matrix; return (eigenvalues, eigenvectors).
    torch.linalg.eigh assumes symmetric input — more stable than torch.eig for SPD.
    A small jitter (1e-6 * I) is added before decomposition to prevent convergence
    failures on ill-conditioned or near-singular matrices during training.
    """
    jitter = _JITTER * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.eigh(A + jitter)


def _mat_pow(A, p):
    """A^p for SPD matrix via eigendecomposition."""
    vals, vecs = _sym_eigh(A)
    vals_p = vals.clamp(min=1e-10).pow(p)
    return vecs @ torch.diag_embed(vals_p) @ vecs.transpose(-2, -1)


def _mat_log(A):
    """Matrix logarithm for SPD matrix via eigendecomposition."""
    vals, vecs = _sym_eigh(A)
    vals_log = vals.clamp(min=1e-10).log()
    return vecs @ torch.diag_embed(vals_log) @ vecs.transpose(-2, -1)


def _mat_exp(A):
    """Matrix exponential for symmetric matrix via eigendecomposition."""
    vals, vecs = _sym_eigh(A)
    vals_exp = vals.exp()
    return vecs @ torch.diag_embed(vals_exp) @ vecs.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Utility: flatten upper triangle (LogEig output -> Euclidean vector)
# ---------------------------------------------------------------------------

def flatten_upper_triangle(X):
    """
    Flatten upper triangle (including diagonal) of a batch of symmetric matrices.
    Input:  (..., n, n)
    Output: (..., n*(n+1)//2)
    """
    n = X.shape[-1]
    idx = torch.triu_indices(n, n, offset=0, device=X.device)
    return X[..., idx[0], idx[1]]


# ---------------------------------------------------------------------------
# Epsilon threshold for ReEig (derived from calibration data)
# ---------------------------------------------------------------------------

def compute_epsilon_threshold(cov_matrices_np, variance_retained=0.995):
    """
    Determine ReEig rectification threshold epsilon from calibration covariances.
    Following Liu et al.: for each trial covariance find the eigenvalue at which
    cumulative variance >= variance_retained, then average across trials.

    Parameters
    ----------
    cov_matrices_np : np.ndarray (n_trials, n_ch, n_ch)
    variance_retained : float, default 0.995

    Returns
    -------
    epsilon : float
    """
    thresholds = []
    for cov in cov_matrices_np:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        total_var = eigvals.sum()
        if total_var <= 0:
            thresholds.append(1e-4)
            continue
        cumulative = np.cumsum(eigvals) / total_var
        idx = np.searchsorted(cumulative, variance_retained)
        idx = min(idx, len(eigvals) - 1)
        thresholds.append(float(eigvals[idx]))
    return max(float(np.mean(thresholds)), 1e-6)


# ---------------------------------------------------------------------------
# SPD manifold layers
# ---------------------------------------------------------------------------

class BiMapLayer(nn.Module):
    """Bilinear Mapping: C_out = W @ C_in @ W^T. W QR-initialized for full row-rank."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        W_init = torch.randn(d_out, d_in)
        W_init, _ = torch.linalg.qr(W_init.T)
        W_init = W_init.T
        self.W = nn.Parameter(W_init)

    def forward(self, X):
        # X: (batch, d_in, d_in) -> (batch, d_out, d_out)
        return self.W @ X @ self.W.T


class ReEigLayer(nn.Module):
    """Eigenvalue Rectification: C_out = U @ max(eps*I, Sigma) @ U^T."""
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X):
        vals, vecs = _sym_eigh(X)
        vals_rect = vals.clamp(min=self.epsilon)
        return vecs @ torch.diag_embed(vals_rect) @ vecs.transpose(-2, -1)


class LogEigLayer(nn.Module):
    """
    Eigenvalue Logarithm: maps SPD manifold to tangent space at I, then flattens.
    X = U @ log(Sigma) @ U^T  ->  flatten upper triangle
    Output size: n*(n+1)//2
    """
    def forward(self, X):
        return flatten_upper_triangle(_mat_log(X))


class RBNLayer(nn.Module):
    """
    Riemannian Batch Normalization. Normalizes Riemannian mean toward I (mean-only).
    Training: Karcher flow batch mean. Inference: EMA running mean.
    """
    def __init__(self, n, momentum=0.1, karcher_steps=10):
        super().__init__()
        self.n             = n
        self.momentum      = momentum
        self.karcher_steps = karcher_steps
        self.register_buffer("running_mean", torch.eye(n))

    def _karcher_mean(self, X):
        G = X.mean(dim=0)
        for _ in range(self.karcher_steps):
            # Compute G^{±0.5} from a single eigendecomposition instead of two.
            vals, vecs = _sym_eigh(G)
            vals_c = vals.clamp(min=1e-10)
            G_sqrt    = vecs @ torch.diag_embed(vals_c.pow( 0.5)) @ vecs.T
            G_invsqrt = vecs @ torch.diag_embed(vals_c.pow(-0.5)) @ vecs.T
            S         = _mat_log(G_invsqrt @ X @ G_invsqrt)
            G         = G_sqrt @ _mat_exp(S.mean(dim=0)) @ G_sqrt
        return G

    def forward(self, X):
        if self.training:
            G = self._karcher_mean(X)
            with torch.no_grad():
                self.running_mean.copy_(
                    _mat_pow(self.running_mean, 1.0 - self.momentum) @
                    _mat_pow(G, self.momentum)
                )
        else:
            G = self.running_mean
        G_invsqrt = _mat_pow(G, -0.5)
        return G_invsqrt @ X @ G_invsqrt


# ---------------------------------------------------------------------------
# Shared encoder trunk (manifold processing, no classifier head)
# ---------------------------------------------------------------------------

class RBNNetEncoder(nn.Module):
    """
    SPD manifold processing trunk: {BiMap-RBN-ReEig}×n_blocks -> {BiMap-RBN} -> LogEig.
    Output: flat Euclidean vector of n_ch*(n_ch+1)//2 elements.
    Used by both RBNNet and DualBandRBNNet.
    """
    def __init__(self, n_ch, epsilon=1e-4, n_blocks=2):
        super().__init__()
        self.n_ch      = n_ch
        self.epsilon   = epsilon
        self.n_blocks  = n_blocks
        self.n_features = n_ch * (n_ch + 1) // 2

        blocks = []
        for _ in range(n_blocks):
            blocks.append(BiMapLayer(n_ch, n_ch))
            blocks.append(RBNLayer(n_ch))
            blocks.append(ReEigLayer(epsilon))
        self.spd_blocks  = nn.ModuleList(blocks)
        self.final_bimap = BiMapLayer(n_ch, n_ch)
        self.final_rbn   = RBNLayer(n_ch)
        self.logeig      = LogEigLayer()

    def forward(self, X):
        """X: (batch, n_ch, n_ch) -> (batch, n_features)"""
        out = X
        for layer in self.spd_blocks:
            out = layer(out)
        out = self.final_bimap(out)
        out = self.final_rbn(out)
        return self.logeig(out)


# ---------------------------------------------------------------------------
# Single-band RBNNet  — faithful to Liu et al. NER 2023
# ---------------------------------------------------------------------------

class RBNNet(nn.Module):
    """
    Single-band RBNNet (paper-faithful). No LayerNorm.

    Use with mu-only (8-13 Hz) or wide-band (8-30 Hz) covariances.
    Matches the architecture in Liu et al. NER 2023 exactly.

    Parameters
    ----------
    n_ch      : SPD matrix dimension (number of EEG channels)
    n_classes : output classes (default 2: REST / MI)
    epsilon   : ReEig threshold (from compute_epsilon_threshold)
    n_blocks  : {BiMap-RBN-ReEig} repetitions (default 2, per paper)
    """
    def __init__(self, n_ch, n_classes=2, epsilon=1e-4, n_blocks=2):
        super().__init__()
        self.n_ch      = n_ch
        self.n_classes = n_classes
        self.epsilon   = epsilon
        self.n_blocks  = n_blocks

        self.encoder    = RBNNetEncoder(n_ch, epsilon, n_blocks)
        self.classifier = nn.Linear(self.encoder.n_features, n_classes)

    def forward(self, X):
        """X: (batch, n_ch, n_ch) -> logits (batch, n_classes)"""
        return self.classifier(self.encoder(X))

    def predict_proba(self, X):
        """Softmax probabilities (batch, n_classes) — [P(REST), P(MI)]"""
        return F.softmax(self.forward(X), dim=-1)


# ---------------------------------------------------------------------------
# Dual-band RBNNet — independent mu + beta streams with per-stream LayerNorm
# ---------------------------------------------------------------------------

class DualBandRBNNet(nn.Module):
    """
    Dual-band extension: independent mu and beta encoders, merged at the classifier.

    Each band has its own RBNNetEncoder with separate weights, separate epsilon,
    and separate RBN running means. After LogEig, each stream's flat vector is
    normalized by an independent LayerNorm (per-sample, safe at batch size 1)
    to address inter-band scale differences. Concatenated output feeds a shared
    Linear classifier.

    Architecture:
      C_mu   -> RBNNetEncoder(eps_mu)   -> LayerNorm(n_features) ]
                                                                   cat -> Linear(2*n_features, 2)
      C_beta -> RBNNetEncoder(eps_beta) -> LayerNorm(n_features) ]

    Parameters
    ----------
    n_ch         : SPD matrix dimension (same for both bands)
    epsilon_mu   : ReEig threshold for mu encoder
    epsilon_beta : ReEig threshold for beta encoder
    n_classes    : output classes (default 2)
    n_blocks     : {BiMap-RBN-ReEig} repetitions per encoder (default 2)
    """
    def __init__(self, n_ch, epsilon_mu=1e-4, epsilon_beta=1e-4,
                 n_classes=2, n_blocks=2):
        super().__init__()
        self.n_ch         = n_ch
        self.n_classes    = n_classes
        self.epsilon_mu   = epsilon_mu
        self.epsilon_beta = epsilon_beta
        self.n_blocks     = n_blocks

        self.mu_encoder   = RBNNetEncoder(n_ch, epsilon_mu,   n_blocks)
        self.beta_encoder = RBNNetEncoder(n_ch, epsilon_beta, n_blocks)

        n_features = self.mu_encoder.n_features  # same for both (same n_ch)

        # Per-stream LayerNorm: normalizes across feature dim per sample.
        # elementwise_affine=True adds learned scale+shift (default).
        self.mu_norm   = nn.LayerNorm(n_features)
        self.beta_norm = nn.LayerNorm(n_features)

        self.classifier = nn.Linear(2 * n_features, n_classes)

    def forward(self, C_mu, C_beta):
        """
        C_mu  : (batch, n_ch, n_ch) mu-band SPD matrices
        C_beta: (batch, n_ch, n_ch) beta-band SPD matrices
        Returns: logits (batch, n_classes)
        """
        feat_mu   = self.mu_norm(self.mu_encoder(C_mu))
        feat_beta = self.beta_norm(self.beta_encoder(C_beta))
        return self.classifier(torch.cat([feat_mu, feat_beta], dim=-1))

    def predict_proba(self, C_mu, C_beta):
        """Softmax probabilities (batch, n_classes) — [P(REST), P(MI)]"""
        return F.softmax(self.forward(C_mu, C_beta), dim=-1)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def build_rbnnet(n_ch, epsilon, n_classes=2, n_blocks=2):
    """Construct a single-band RBNNet (paper-faithful)."""
    return RBNNet(n_ch=n_ch, n_classes=n_classes, epsilon=epsilon, n_blocks=n_blocks)


def build_dual_band_rbnnet(n_ch, epsilon_mu, epsilon_beta, n_classes=2, n_blocks=2):
    """Construct a DualBandRBNNet."""
    return DualBandRBNNet(
        n_ch=n_ch, epsilon_mu=epsilon_mu, epsilon_beta=epsilon_beta,
        n_classes=n_classes, n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# Bundle serialization / deserialization
# ---------------------------------------------------------------------------

def _unwrap_compiled(model):
    """
    Return the underlying nn.Module from a torch.compile-wrapped OptimizedModule,
    or the model itself if it is not compiled.

    torch.compile wraps the model in torch._dynamo.eval_frame.OptimizedModule and
    stores the original module as _orig_mod in nn.Module._modules.  This means
    compiled_model.state_dict() produces keys prefixed with "_orig_mod." which are
    incompatible with the fresh (uncompiled) model used in load_rbnnet_bundle.
    Unwrapping before serialisation avoids this mismatch entirely.
    """
    orig = getattr(model, "_orig_mod", None)
    if orig is not None and isinstance(orig, torch.nn.Module):
        return orig
    return model


def save_rbnnet_bundle(model, label_to_bin, bin_to_label, tl_star, th_star,
                       roc_auc, channel_names, training_meta, path):
    """
    Serialize a trained RBNNet or DualBandRBNNet bundle to pickle.
    Compatible with runtime_common.py dispatch logic.

    If *model* is a torch.compile-wrapped OptimizedModule the underlying
    nn.Module is extracted before serialisation so that state_dict keys are
    not prefixed with "_orig_mod." (which would break load_rbnnet_bundle).
    """
    import pickle
    # Unwrap torch.compile wrapper so state_dict keys are clean.
    raw_model = _unwrap_compiled(model)
    use_beta = isinstance(raw_model, DualBandRBNNet)
    model_config = {
        "n_ch":      raw_model.n_ch,
        "n_blocks":  raw_model.n_blocks,
        "n_classes": raw_model.n_classes,
        "use_beta":  use_beta,
    }
    if use_beta:
        model_config["epsilon_mu"]   = raw_model.epsilon_mu
        model_config["epsilon_beta"] = raw_model.epsilon_beta
    else:
        model_config["epsilon"] = raw_model.epsilon

    bundle = {
        "type":             "rbnnet",
        "model_state_dict": raw_model.state_dict(),
        "model_config":     model_config,
        "label_to_bin":     label_to_bin,
        "bin_to_label":     bin_to_label,
        "tl_star":          float(tl_star),
        "th_star":          float(th_star),
        "roc_auc":          float(roc_auc),
        "channel_names":    list(channel_names),
        "training_meta":    training_meta,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[RBNNet] Bundle saved -> {path}")


def load_rbnnet_bundle(path):
    """
    Load a serialized bundle and reconstruct the model in eval mode.
    Adds a 'model' key with the ready-to-use network.
    """
    import pickle
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    cfg      = bundle["model_config"]
    use_beta = cfg.get("use_beta", False)

    if use_beta:
        model = DualBandRBNNet(
            n_ch=cfg["n_ch"],
            epsilon_mu=cfg["epsilon_mu"],
            epsilon_beta=cfg["epsilon_beta"],
            n_classes=cfg.get("n_classes", 2),
            n_blocks=cfg.get("n_blocks", 2),
        )
    else:
        model = RBNNet(
            n_ch=cfg["n_ch"],
            epsilon=cfg["epsilon"],
            n_classes=cfg.get("n_classes", 2),
            n_blocks=cfg.get("n_blocks", 2),
        )

    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    bundle["model"] = model
    return bundle
