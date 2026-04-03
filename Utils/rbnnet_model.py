"""
rbnnet_model.py
---------------
PyTorch implementation of RBNNet for EEG Motor Imagery classification.

RBNNet is a deep network on the SPD (Symmetric Positive Definite) manifold,
combining SPDNet (Huang & Van Gool, AAAI 2017) with Riemannian Batch
Normalization (Brooks et al., NeurIPS 2019).

Architecture (from Liu et al., NER 2023):
    {BiMap - RBN - ReEig} x2  →  {BiMap - RBN}  →  {LogEig}  →  FC → output

Reference: Liu, Kumar, Alawieh, Carnahan & Millán, NER 2023.
           "On Transfer Learning for Naive Brain Computer Interface Users."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: matrix functions via eigendecomposition
# ---------------------------------------------------------------------------

def _sym_eigh(A):
    """Eigendecompose a symmetric matrix; return (eigenvalues, eigenvectors).
    Uses torch.linalg.eigh which assumes symmetric/Hermitian input and is
    numerically more stable than torch.eig for SPD matrices.
    """
    return torch.linalg.eigh(A)


def _mat_pow(A, p):
    """Compute A^p for a symmetric positive definite matrix via eigendecomposition."""
    vals, vecs = _sym_eigh(A)
    vals_p = vals.clamp(min=1e-10).pow(p)
    return vecs @ torch.diag_embed(vals_p) @ vecs.transpose(-2, -1)


def _mat_log(A):
    """Compute matrix logarithm of a symmetric positive definite matrix."""
    vals, vecs = _sym_eigh(A)
    vals_log = vals.clamp(min=1e-10).log()
    return vecs @ torch.diag_embed(vals_log) @ vecs.transpose(-2, -1)


def _mat_exp(A):
    """Compute matrix exponential of a symmetric matrix via eigendecomposition."""
    vals, vecs = _sym_eigh(A)
    vals_exp = vals.exp()
    return vecs @ torch.diag_embed(vals_exp) @ vecs.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Utility: flatten / unflatten upper triangle (for LogEig output)
# ---------------------------------------------------------------------------

def flatten_upper_triangle(X):
    """
    Flatten the upper triangle (including diagonal) of a batch of symmetric matrices.
    Input:  (batch, n, n)
    Output: (batch, n*(n+1)//2)
    """
    n = X.shape[-1]
    idx = torch.triu_indices(n, n, offset=0, device=X.device)
    return X[..., idx[0], idx[1]]


# ---------------------------------------------------------------------------
# Epsilon threshold for ReEig
# ---------------------------------------------------------------------------

def compute_epsilon_threshold(cov_matrices_np, variance_retained=0.995):
    """
    Determine the ReEig rectification threshold epsilon from calibration data.
    Following the paper: evaluate eigenvalues of every trial covariance, and
    find the threshold at which `variance_retained` (default 99.5%) of the
    variance is retained across all subjects/trials.

    Parameters
    ----------
    cov_matrices_np : np.ndarray, shape (n_trials, n_ch, n_ch)
        Shrinkage-regularized covariance matrices from calibration data.
    variance_retained : float
        Fraction of variance to retain (default 0.995 from paper).

    Returns
    -------
    epsilon : float
        The rectification threshold to use in all ReEig layers.
    """
    thresholds = []
    for cov in cov_matrices_np:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # descending
        total_var = eigvals.sum()
        if total_var <= 0:
            thresholds.append(1e-4)
            continue
        cumulative = np.cumsum(eigvals) / total_var
        # Find the smallest eigenvalue that still keeps >= variance_retained
        idx = np.searchsorted(cumulative, variance_retained)
        idx = min(idx, len(eigvals) - 1)
        thresholds.append(float(eigvals[idx]))

    eps = float(np.mean(thresholds))
    return max(eps, 1e-6)  # safety floor


# ---------------------------------------------------------------------------
# SPD manifold layers
# ---------------------------------------------------------------------------

class BiMapLayer(nn.Module):
    """
    Bilinear Mapping on the SPD manifold.
    C_out = W @ C_in @ W^T

    W is initialized via QR decomposition of a random Gaussian matrix to
    start orthogonal (full row-rank), ensuring SPD output from SPD input.

    Parameters
    ----------
    d_in  : input SPD matrix dimension
    d_out : output SPD matrix dimension (d_out <= d_in to reduce; == d_in to preserve)
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Initialize W as a semi-orthogonal matrix
        W_init = torch.randn(d_out, d_in)
        W_init, _ = torch.linalg.qr(W_init.T)  # QR of (d_in, d_out)
        W_init = W_init.T                        # Back to (d_out, d_in)
        self.W = nn.Parameter(W_init)

    def forward(self, X):
        """
        Parameters
        ----------
        X : (batch, d_in, d_in)  SPD matrices

        Returns
        -------
        (batch, d_out, d_out)  SPD matrices
        """
        # C_out = W @ C_in @ W^T
        return self.W @ X @ self.W.T


class ReEigLayer(nn.Module):
    """
    Eigenvalue Rectification on the SPD manifold.
    C_out = U @ max(eps*I, Sigma) @ U^T

    Provides nonlinearity analogous to ReLU: clips small eigenvalues at eps,
    preventing degeneracy of the SPD structure.

    Parameters
    ----------
    epsilon : float  — rectification threshold (computed from calibration data)
    """
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X):
        """
        Parameters
        ----------
        X : (batch, n, n)  SPD matrices

        Returns
        -------
        (batch, n, n)  SPD matrices with rectified eigenvalues
        """
        vals, vecs = _sym_eigh(X)
        vals_rect = vals.clamp(min=self.epsilon)
        return vecs @ torch.diag_embed(vals_rect) @ vecs.transpose(-2, -1)


class LogEigLayer(nn.Module):
    """
    Eigenvalue Logarithm: maps the SPD manifold to its tangent space at I,
    then flattens to a Euclidean feature vector.

    X = U @ log(Sigma) @ U^T  (then flatten upper triangle)

    Output size: n*(n+1)//2 per sample.
    """
    def forward(self, X):
        """
        Parameters
        ----------
        X : (batch, n, n)  SPD matrices (all eigenvalues strictly positive
            after ReEig in the preceding layer)

        Returns
        -------
        (batch, n*(n+1)//2)  flattened tangent-space vectors
        """
        log_X = _mat_log(X)
        return flatten_upper_triangle(log_X)


class RBNLayer(nn.Module):
    """
    Riemannian Batch Normalization on the SPD manifold.

    Normalizes the Riemannian (geometric) mean of the batch to the identity
    matrix. Only mean normalization — no variance normalization — as per
    Brooks et al. (NeurIPS 2019) and the paper.

    During training: Riemannian mean is estimated from the current batch via
    Karcher flow (iterative gradient descent on the manifold).
    During inference: uses an exponential moving average of the running mean.

    Parameters
    ----------
    n        : SPD matrix dimension
    momentum : EMA momentum for running mean update (default 0.1)
    karcher_steps : iterations for Karcher flow convergence (default 10)
    """
    def __init__(self, n, momentum=0.1, karcher_steps=10):
        super().__init__()
        self.n = n
        self.momentum = momentum
        self.karcher_steps = karcher_steps
        # Running mean — initialized to identity (not a parameter, not saved with grad)
        self.register_buffer("running_mean", torch.eye(n))

    def _karcher_mean(self, X):
        """
        Estimate Riemannian mean of batch X via Karcher flow.
        X: (batch, n, n)
        Returns: (n, n)  Riemannian mean matrix G
        """
        # Initialize at the arithmetic mean (fast approximation)
        G = X.mean(dim=0)
        for _ in range(self.karcher_steps):
            G_invsqrt = _mat_pow(G, -0.5)   # G^{-1/2}
            G_sqrt = _mat_pow(G, 0.5)         # G^{1/2}
            # Log map each X_i at G
            S = _mat_log(G_invsqrt @ X @ G_invsqrt)   # (batch, n, n)
            S_mean = S.mean(dim=0)                      # (n, n)
            G = G_sqrt @ _mat_exp(S_mean) @ G_sqrt
        return G

    def forward(self, X):
        """
        Parameters
        ----------
        X : (batch, n, n)  SPD matrices

        Returns
        -------
        (batch, n, n)  normalized SPD matrices
        """
        if self.training:
            G = self._karcher_mean(X)
            # Update running mean via geodesic interpolation
            with torch.no_grad():
                self.running_mean.copy_(
                    _mat_pow(self.running_mean, 1.0 - self.momentum) @
                    _mat_pow(G, self.momentum)
                )
        else:
            G = self.running_mean

        # Normalize: C_norm = G^{-1/2} @ C @ G^{-1/2}
        G_invsqrt = _mat_pow(G, -0.5)
        return G_invsqrt @ X @ G_invsqrt


# ---------------------------------------------------------------------------
# Full RBNNet
# ---------------------------------------------------------------------------

class RBNNet(nn.Module):
    """
    Full RBNNet architecture for EEG Motor Imagery classification.

    Architecture (following Liu et al., NER 2023):
      {BiMap - RBN - ReEig} x n_blocks  →  {BiMap - RBN}  →  {LogEig}  →  FC

    All BiMap layers preserve the SPD dimension (d_in == d_out == n_ch), as
    the paper found dimension reduction below the channel count hurt performance.

    Parameters
    ----------
    n_ch      : number of EEG channels = SPD matrix dimension
    n_classes : number of output classes (default 2: REST / MI)
    epsilon   : ReEig threshold (computed from calibration data)
    n_blocks  : number of {BiMap-RBN-ReEig} blocks before the final {BiMap-RBN}
                (default 2, matching the paper's architecture)
    """
    def __init__(self, n_ch, n_classes=2, epsilon=1e-4, n_blocks=2):
        super().__init__()
        self.n_ch = n_ch
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.n_blocks = n_blocks

        # Build {BiMap - RBN - ReEig} x n_blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(BiMapLayer(n_ch, n_ch))
            blocks.append(RBNLayer(n_ch))
            blocks.append(ReEigLayer(epsilon))
        self.spd_blocks = nn.ModuleList(blocks)

        # Final {BiMap - RBN} (no ReEig before LogEig)
        self.final_bimap = BiMapLayer(n_ch, n_ch)
        self.final_rbn = RBNLayer(n_ch)

        # LogEig: SPD → flat Euclidean vector
        self.logeig = LogEigLayer()

        # Classification head
        n_features = n_ch * (n_ch + 1) // 2
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, X):
        """
        Parameters
        ----------
        X : (batch, n_ch, n_ch)  SPD covariance matrices

        Returns
        -------
        logits : (batch, n_classes)  — pass through softmax for probabilities
        """
        # {BiMap - RBN - ReEig} blocks
        out = X
        for layer in self.spd_blocks:
            out = layer(out)

        # Final {BiMap - RBN}
        out = self.final_bimap(out)
        out = self.final_rbn(out)

        # LogEig: manifold → Euclidean
        out = self.logeig(out)   # (batch, n_features)

        # Classification
        logits = self.classifier(out)
        return logits

    def predict_proba(self, X):
        """
        Convenience method returning softmax probabilities.

        Parameters
        ----------
        X : (batch, n_ch, n_ch)  SPD covariance matrices

        Returns
        -------
        probs : (batch, n_classes)  — [P(REST), P(MI)]
        """
        logits = self.forward(X)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def build_rbnnet(n_ch, epsilon, n_classes=2, n_blocks=2):
    """
    Construct an RBNNet instance from the given parameters.

    Parameters
    ----------
    n_ch     : SPD matrix dimension (number of EEG channels used)
    epsilon  : ReEig rectification threshold
    n_classes: number of output classes (2 for binary MI/REST)
    n_blocks : number of {BiMap-RBN-ReEig} blocks

    Returns
    -------
    RBNNet instance (not yet trained)
    """
    return RBNNet(n_ch=n_ch, n_classes=n_classes, epsilon=epsilon, n_blocks=n_blocks)


def save_rbnnet_bundle(model, label_to_bin, bin_to_label, tl_star, th_star,
                       roc_auc, channel_names, training_meta, path):
    """
    Serialize the trained RBNNet model bundle to a pickle file.

    The bundle format mirrors XGBoost and MDM bundles for compatibility
    with the existing runtime_common.py dispatch logic.
    """
    import pickle
    bundle = {
        "type": "rbnnet",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_ch": model.n_ch,
            "n_blocks": model.n_blocks,
            "epsilon": model.epsilon,
            "n_classes": model.n_classes,
        },
        "label_to_bin": label_to_bin,
        "bin_to_label": bin_to_label,
        "tl_star": float(tl_star),
        "th_star": float(th_star),
        "roc_auc": float(roc_auc),
        "channel_names": list(channel_names),
        "training_meta": training_meta,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[RBNNet] Model bundle saved to {path}")


def load_rbnnet_bundle(path):
    """
    Load a serialized RBNNet bundle and reconstruct the model.

    Returns
    -------
    bundle : dict  with "model" key holding a ready-to-use RBNNet in eval mode
    """
    import pickle
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    cfg = bundle["model_config"]
    model = RBNNet(
        n_ch=cfg["n_ch"],
        n_classes=cfg.get("n_classes", 2),
        epsilon=cfg["epsilon"],
        n_blocks=cfg.get("n_blocks", 2),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    bundle["model"] = model
    return bundle
