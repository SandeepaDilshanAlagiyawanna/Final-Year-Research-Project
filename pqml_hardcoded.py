"""
=============================================================================
Noise-Assisted Photonic Quantum Machine Learning  –  4-Mode Circuit
# Inspired by ENAQT: controlled noise improves transport / learning flow
=============================================================================
Author  : A.M.A.Sandeepa D. Alagiyawanna (215506J)
Dept    : Computational Mathematics, Faculty of IT, University of Moratuwa

Circuit: 4 optical modes (qumodes) × 2 variational layers

Gate sequence per layer (symplectic / quadrature-mean representation):
  1. Rotation gates     – R(φ_k)          on each mode k   [4 params]
  2. Squeezing gates    – S(r_k, φ_k)     on each mode k   [8 params]
  3. Beamsplitters      – BS(θ,φ)         brick-wall layout [6 params]
       Layer A pairs: (0,1), (2,3)
       Layer B pairs: (1,2)          ← improves inter-mode connectivity (brick-wall scheme)
  4. Noise channels     – Loss η + phase σ_φ (noise-assisted only)
  5. Displacement gates – D(r_k, i_k)     on each mode k   [8 params]

Params per layer: 4 + 8 + 6 + 8 = 26
Total θ params : 26 × 2 = 52
Total λ params : 4  (η + σ_φ per layer; jointly optimised, σ_φ is a bounded, learned parameter derived from raw variables via tanh scaling.)

Architecture matches Figure 1 (proposal):
  Classical Dataset → Classical Processing (normalisation)
  → Quantum Feature Encoding (4-mode amplitude+phase)
  → Variational Photonic QC (θ)
  → Parameterised Noise Model (jointly optimised λ = {η, σ_φ})
  → Quantum Measurement (homodyne x-quadrature, averaged over 4 modes)
  → Classical Optimizer (θ, λ) → [feedback] → Performance Comparison
=============================================================================
"""

import warnings

warnings.filterwarnings("ignore")
from tensorflow.keras.datasets import cifar10
from sklearn.datasets import fetch_openml
import numpy as np, time
import matplotlib
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_digits, make_moons, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from sklearn.datasets import fetch_openml

SEED = 42
np.random.seed(SEED)

import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Circuit configuration  ← 4 modes
# ─────────────────────────────────────────────────────────────────────────────
N_MODES = 4  # ← 4 optical modes (qumodes)
N_LAYERS = 2  # variational depth

# Beamsplitter pairs in brick-wall layout for 4 modes:
#   Row A: (0,1), (2,3)   → 2 BSs
#   Row B: (1,2)          → 1 BS
#   Total: 3 BSs × 2 params each = 6 params
BS_PAIRS = [(0, 1), (2, 3), (1, 2)]  # ← 3 BSs for 4 modes
N_BS_PARAMS = len(BS_PAIRS) * 2  # 6

# Params per layer
N_ROT = N_MODES  #  4  rotation angles
N_SQ = 2 * N_MODES  #  8  squeezing (r, φ) per mode
N_DISP = 2 * N_MODES  #  8  displacement (re, im) per mode
PPL = N_ROT + N_SQ + N_BS_PARAMS + N_DISP  # 4+8+6+8 = 26

# λ structure: [logit(η_L1), σ_φ_L1, logit(η_L2), σ_φ_L2, ...]
# per layer: one transmissivity (η) + one phase diffusion std (σ_φ)
N_LAMBDA = 2 * N_LAYERS  # η + σ_φ per layer

# ── Colours ───────────────────────────────────────────────────────────────────
CR = "#E57373"  # baseline
CG = "#4CAF50"  # noise-assisted

# =============================================================================
#  Photonic Gate Matrices (symplectic quadrature-mean form)
# =============================================================================


def rot_mat(phi):
    """2×2 rotation matrix for a single mode."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def squeeze_mat(r, phi):
    """2×2 squeezing matrix S(r, φ) for a single mode."""
    R = rot_mat(phi)
    return R @ np.diag([np.exp(-r), np.exp(r)]) @ R.T


def apply_beamsplitter(v, theta, phi, i, j):
    """
    Apply BS(theta, phi) between modes i and j of quadrature vector v.
    v has length 2*N_MODES; mode k occupies indices [2k, 2k+1].
    """
    idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    sub = v[idx]
    t, r = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    BS = np.array(
        [
            [t, -r * cp, 0, r * sp],
            [r * cp, t, -r * sp, 0],
            [0, r * sp, t, r * cp],
            [-r * sp, 0, -r * cp, t],
        ]
    )
    v = v.copy()
    v[np.array(idx)] = BS @ sub
    return v


# =============================================================================
#  Quantum Feature Encoding  (4-mode amplitude + phase encoding)
# =============================================================================


def encode(x):
    """
    Encodes classical feature vector x ∈ [0,1]^n into the quadrature
    means of 4 optical modes.

    For mode k:
        amplitude  a_k = sqrt(|x[k % len(x)]|)
        phase     φ_k = π · x[(k+1) % len(x)]
        x-quad  = a_k · cos(φ_k)
        p-quad  = a_k · sin(φ_k)

    For datasets with fewer than 4 features (e.g. Moons has 2),
    features are wrapped cyclically so all 4 modes are loaded.
    For datasets with 4 features (e.g. Iris), each feature maps 1-to-1.
    """
    v = np.zeros(2 * N_MODES)
    for k in range(N_MODES):
        i = k % len(x)
        a = np.sqrt(abs(x[i]) + 1e-8)
        ph = np.pi * x[(i + 1) % len(x)]
        v[2 * k] = a * np.cos(ph)
        v[2 * k + 1] = a * np.sin(ph)
    return v


# =============================================================================
#  Variational Layer  (4-mode)
# =============================================================================


def var_layer(v, p, eta=None, sigma_phi=None, stochastic_phase=0.0):
    """
    One variational layer for 4 modes.

    Gate order: Rotation → Squeezing → Beamsplitters (brick-wall)
                → [Noise] → Displacement

    p   : layer parameter vector of length PPL = 26
    eta : photon-loss transmissivity ∈ (0,1]; None = baseline (no loss)
    sigma_phi : learnable phase diffusion std; None = no learnable phase noise
    stochastic_phase : std of phase diffusion noise (test-time evaluation only)
        *** Noise is applied deterministically during training (via η and σ_φ), and stochastic phase noise is added during evaluation to simulate measurement uncertainty, for deterministic gradient flow ***
    """
    o = 0

    # ── 1. Rotation gates (one per mode) ──────────────────────────
    for k in range(N_MODES):
        v[2 * k : 2 * k + 2] = rot_mat(p[o + k]) @ v[2 * k : 2 * k + 2]
    o += N_ROT  # o = 4

    # ── 2. Squeezing gates (r, φ per mode) ────────────────────────
    for k in range(N_MODES):
        v[2 * k : 2 * k + 2] = (
            squeeze_mat(p[o + 2 * k], p[o + 2 * k + 1]) @ v[2 * k : 2 * k + 2]
        )
    o += N_SQ  # o = 12

    # ── 3. Beamsplitters – brick-wall: (0,1), (2,3), (1,2) ───────
    for pair_idx, (i, j) in enumerate(BS_PAIRS):
        theta = p[o + 2 * pair_idx]
        phi = p[o + 2 * pair_idx + 1]
        v = apply_beamsplitter(v, theta, phi, i, j)
    o += N_BS_PARAMS  # o = 18

    # ── 4. Parameterised Environmental Noise (λ block) ────────────
    if eta is not None:
        # Photon-loss channel: amplitude-damps all modes by sqrt(η)
        # (dominant noise in integrated photonic waveguides)
        v = np.sqrt(eta) * v

    # Learnable phase diffusion (jointly optimised with θ during training)
    if sigma_phi is not None and sigma_phi > 0:
        mask = np.zeros_like(v)
        mask[1::2] = 1.0  # p-quadratures only
        v = v + sigma_phi * np.tanh(v) * mask

    # Test-time stochastic phase (for realistic quantum measurement variability)
    if stochastic_phase > 0:
        noise_eval = np.random.normal(0, stochastic_phase, v.shape)
        mask = np.zeros_like(v)
        mask[1::2] = 1.0  # p-quadratures only
        v = v + noise_eval * mask

    # ── 5. Displacement gates (re, im per mode) ───────────────────
    for k in range(N_MODES):
        v[2 * k] += p[o + 2 * k] * np.sqrt(2)
        v[2 * k + 1] += p[o + 2 * k + 1] * np.sqrt(2)
    # o += N_DISP  (not needed after last gate)

    return v


# =============================================================================
#  Full Circuit Forward Pass
# =============================================================================


def forward(x, theta, lambda_params=None, stochastic_phase=0.0):
    """
    Full variational photonic circuit for one sample x.

    Measurement: average of x-quadrature expectation values over all 4
    modes (indices 0, 2, 4, 6) → more information than single-mode
    readout, analogous to multi-mode homodyne detection.

    lambda_params : length-(2*N_LAYERS) array [logit(η_L1), σ_φ_L1, logit(η_L2), σ_φ_L2, ...]
                    None = baseline (no noise).
    """
    v = encode(x)
    for L in range(N_LAYERS):
        p = theta[L * PPL : (L + 1) * PPL]
        eta = None
        sigma_phi = None
        if lambda_params is not None:
            eta = sigmoid(lambda_params[2 * L])
            raw_sigma = lambda_params[2 * L + 1]
            sigma_phi = 0.2 * np.tanh(raw_sigma)
            sigma_phi = np.abs(sigma_phi) # keep σ positive and reasonable
        v = var_layer(
            v, p, eta=eta, sigma_phi=sigma_phi, stochastic_phase=stochastic_phase
        )

    # Multi-mode homodyne: mean of x-quadratures across all 4 modes
    x_quads = v[0::2]  # indices 0, 2, 4, 6
    measurement = x_quads.mean()
    return sigmoid(measurement)


def batch_forward(X, theta, lam=None, stochastic_phase=0.0):
    return np.array([forward(x, theta, lam, stochastic_phase) for x in X])


# =============================================================================
#  Loss
# =============================================================================


def bce(yt, yp, eps=1e-7):
    yp = np.clip(yp, eps, 1 - eps)
    return -np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))


def loss_fn(params, X, y, noise_assisted):
    nt = N_LAYERS * PPL
    theta = params[:nt]
    lam = params[nt:] if noise_assisted else None
    yp = batch_forward(X, theta, lam, stochastic_phase=0.0)
    reg = 1e-4 * np.sum(theta**2)
    return bce(y, yp) + reg


# =============================================================================
#  Training Loop  (Classical Optimiser block)
# =============================================================================


def train(X_tr, y_tr, X_val, y_val, noise_assisted=False, epochs=50):
    """
    Hybrid classical-quantum training.

    θ  : N_LAYERS × PPL = 52 circuit parameters
    λ  : 2 parameters per layer (η, σ_φ), total = 4 
         jointly optimised with θ when noise_assisted=True.

    Optimiser: L-BFGS-B in mini-batch epochs (smooth, gradient-based).
    Noise enters deterministically during training (smooth loss landscape).
    """
    rng = np.random.default_rng(SEED)
    nt = N_LAYERS * PPL  # 52

    theta0 = rng.uniform(-np.pi, np.pi, nt)

    if noise_assisted:
        # Initialize λ = [logit(η), σ_φ, ...] per layer
        # logit(0.90) ≈ 2.197 → η ≈ 0.90 (mild loss)
        # σ_φ initialized ≈ 0.05 (small phase diffusion)
        lam0 = np.zeros(N_LAMBDA)
        for L in range(N_LAYERS):
            lam0[2 * L] = 2.197  # logit(η_L)
            lam0[2 * L + 1] = 0.05  # σ_φ_L
        p0 = np.concatenate([theta0, lam0])
    else:
        p0 = theta0.copy()

    history = {
        "tl": [],
        "vl": [],
        "va": [],
        "eta": [[] for _ in range(N_LAYERS)],
        "sigma_phi": [[] for _ in range(N_LAYERS)],
    }
    p = p0.copy()

    for ep in range(epochs):
        # mini-batch
        idx = rng.choice(len(X_tr), size=min(32, len(X_tr)), replace=False)
        Xb, yb = X_tr[idx], y_tr[idx]

        res = minimize(
            loss_fn,
            p,
            args=(Xb, yb, noise_assisted),
            method="L-BFGS-B",
            options={"maxiter": 12, "ftol": 1e-14, "gtol": 1e-8},
        )
        p = res.x

        tl = loss_fn(p, X_tr, y_tr, noise_assisted)
        vl = loss_fn(p, X_val, y_val, noise_assisted)
        lam_cur = p[nt:] if noise_assisted else None
        vp = batch_forward(X_val, p[:nt], lam_cur)
        va = accuracy_score(y_val, (vp >= 0.5).astype(int))

        history["tl"].append(float(tl))
        history["vl"].append(float(vl))
        history["va"].append(float(va))

        if noise_assisted:
            for L in range(N_LAYERS):
                history["eta"][L].append(float(sigmoid(p[nt + 2 * L])))

                raw_sigma = p[nt + 2 * L + 1]
                sigma_logged = 0.2 * np.abs(np.tanh(raw_sigma))

                history["sigma_phi"][L].append(float(sigma_logged))

        if ep % 10 == 0:
            eta_str = ""
            if noise_assisted:
                etas = [f"{sigmoid(p[nt+2*L]):.3f}" for L in range(N_LAYERS)]
                sigs = [
                    f"{0.2 * np.abs(np.tanh(p[nt+2*L+1])):.4f}"
                    for L in range(N_LAYERS)
                ]
                eta_str = f" | η={etas} | σ_φ={sigs}"
            print(f"   ep{ep:3d} | train_loss={tl:.4f} | val_acc={va:.3f}{eta_str}")

    theta_final = p[:nt]
    lam_final = p[nt:] if noise_assisted else None
    return theta_final, lam_final, history


# =============================================================================
#  Evaluation
# =============================================================================


def evaluate(X, y, theta, lam=None, n_shots=20):
    """
    For noise-assisted model, averages n_shots stochastic evaluations
    to approximate quantum shot statistics.
    """
    if lam is not None:
        all_p = np.stack(
            [
                batch_forward(X, theta, lam, stochastic_phase=0.02)
                for _ in range(n_shots)
            ],
            axis=0,
        )
        yp = all_p.mean(axis=0)
    else:
        yp = batch_forward(X, theta, lam=None)

    yc = (yp >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, yp)
    except:
        auc = float("nan")
    return {
        "acc": accuracy_score(y, yc),
        "f1": f1_score(y, yc, zero_division=0),
        "auc": auc,
        "cm": confusion_matrix(y, yc),
        "prob": yp,
        "pred": yc,
    }


# =============================================================================
#  Datasets
# =============================================================================

def cifar10_data(n_components=4, n_samples=5000):
    print("   Loading CIFAR-10 …")

    X, y = fetch_openml("CIFAR_10_small", version=1, return_X_y=True, parser="auto")

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    X = X.astype(np.float32)
    y = y.astype(int)

    # Binary: airplane vs automobile
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    y = (y == 1).astype(float)

    # Subsample safely
    n_available = len(X)

    if n_samples >= n_available:
        print(f"   ⚠ Using full dataset ({n_available} samples)")
        idx = np.random.permutation(n_available)
        X, y = X[idx, :], y[idx]
    else:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y,
            train_size=n_samples,
            stratify=y,
            random_state=SEED
        )

    # PCA → 4D
    print("   Applying PCA → 4D …")
    X = PCA(n_components=n_components, random_state=SEED).fit_transform(X)

    X = MinMaxScaler().fit_transform(X)

    return X, y, f"CIFAR-10 (0 vs 1, PCA→{n_components}D)"

def cifar10_full_data(n_components=4, n_samples=10000):
    print("   Loading CIFAR-10 (FULL dataset via Keras)…")
    (X, y), _ = cifar10.load_data()

    X = X.reshape(len(X), -1).astype(np.float32)  # flatten (3072 features)
    y = y.flatten()

    # Binary: airplane (0) vs automobile (1)
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    y = (y == 1).astype(float)

    print(f"   Available samples: {len(X)}")

    # Subsample safely
    n_available = len(X)

    if n_samples >= n_available:
        idx = np.random.permutation(n_available)
        X, y = X[idx], y[idx]
    else:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y,
            train_size=n_samples,
            stratify=y,
            random_state=SEED
        )

    # PCA → 4D
    print("   Applying PCA → 4D …")
    X = PCA(n_components=n_components, random_state=SEED).fit_transform(X)

    X = MinMaxScaler().fit_transform(X)

    return X, y, f"CIFAR-10 FULL (0 vs 1, PCA→{n_components}D)"

def fashion_mnist_data(n_components=4, use_case="performance"):
    """
    Fashion-MNIST: Classes 0 vs 6 (T-shirt vs Shirt), reduced via PCA.
    Much harder than MNIST digits due to overlapping clothing classes.

    n_components : int, PCA dimensionality
        2: For visualization and decision boundary plots
        4: For performance evaluation and final results (more informative)
    use_case : str, "visualization" or "performance"
    """
    print("   Loading Fashion-MNIST (may take ~30s on first run) …")
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, parser="auto")
    X = X.astype(np.float32)

    # Convert y to numeric if needed
    if hasattr(y, "values"):
        y = np.array(y.values, dtype=int)
    else:
        y = np.array(y, dtype=int)

    # Filter for classes: 0 (T-shirt/top) vs 6 (Shirt)
    m = (y == 0) | (y == 6)
    X = X[m] if hasattr(X, "iloc") else X[m]
    y = y[m]

    # Convert to numpy array if pandas
    if hasattr(X, "values"):
        X = X.values

    # Remap labels: 0→0, 6→1
    y = (y == 6).astype(float)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=SEED)
    X = pca.fit_transform(X)

    # Normalize to [0, 1]
    X = MinMaxScaler().fit_transform(X)

    if n_components == 2:
        desc = "Fashion-MNIST (T-shirt vs Shirt, PCA→2D)"
    elif n_components == 4:
        desc = "Fashion-MNIST (T-shirt vs Shirt, PCA→4D)"
    else:
        desc = f"Fashion-MNIST (T-shirt vs Shirt, PCA→{n_components}D)"

    return X, y, desc


def breast_cancer_data(n_components=4, use_case="performance"):
    """
    Breast Cancer Dataset: Real-world medical binary classification.
    30 features → reduced to n_components via PCA.

    n_components : int, PCA dimensionality
        2: For visualization and decision boundary plots
        4: For performance evaluation and final results (more informative)
    use_case : str, "visualization" or "performance"
    """
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, random_state=SEED)
    X = pca.fit_transform(X)

    # Normalize to [0, 1]
    X = MinMaxScaler().fit_transform(X)

    if n_components == 2:
        desc = "Breast Cancer (Medical, PCA→2D)"
    elif n_components == 4:
        desc = "Breast Cancer (Medical, PCA→4D)"
    else:
        desc = f"Breast Cancer (Medical, PCA→{n_components}D)"

    return X, y.astype(float), desc

def split_data(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    Xt, Xv, yt, yv = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
    )
    return Xt, Xv, Xte, yt, yv, yte


# =============================================================================
#  Decision Boundary Plot (2-feature datasets only)
# =============================================================================


def draw_boundary(X, y, theta, lam, ax, title):
    h = 0.05
    x0, x1 = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y0, y1 = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z = batch_forward(np.c_[xx.ravel(), yy.ravel()], theta, lam).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=40, cmap="RdYlGn", alpha=0.72)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1], c=CR, edgecolors="k", s=28, zorder=5, label="Cl-0"
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1], c=CG, edgecolors="k", s=28, zorder=5, label="Cl-1"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


# =============================================================================
#  Classical Baselines (Logistic Regression & SVM)
# =============================================================================


def train_classical_baselines(X_tr, y_tr, X_te, y_te):
    """
    Train classical ML baselines (LogReg, SVM) for publication-ready comparison.
    Returns dict with performance metrics.
    """
    results = {}

    # Logistic Regression
    print("   ▶ Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, y_prob)
    except:
        auc = float("nan")
    results["LogReg"] = {
        "acc": accuracy_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "auc": auc,
        "cm": confusion_matrix(y_te, y_pred),
        "prob": y_prob,
        "pred": y_pred,
    }
    print(
        f"      LogReg | Acc={results['LogReg']['acc']:.4f}  F1={results['LogReg']['f1']:.4f}"
    )

    # Support Vector Machine (RBF kernel)
    print("   ▶ SVM (RBF) …")
    svm = SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_te)
    y_prob = svm.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, y_prob)
    except:
        auc = float("nan")
    results["SVM"] = {
        "acc": accuracy_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "auc": auc,
        "cm": confusion_matrix(y_te, y_pred),
        "prob": y_prob,
        "pred": y_pred,
    }
    print(
        f"      SVM     | Acc={results['SVM']['acc']:.4f}  F1={results['SVM']['f1']:.4f}"
    )

    return results


# =============================================================================
#  Main Experiment
# =============================================================================

EP = 50

# Print config banner
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  Noise-Assisted PQML  –  A.M.A.Sandeepa D. Alagiyawanna          ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"\nCircuit : {N_MODES} modes × {N_LAYERS} layers")
print(f"BS pairs: {BS_PAIRS}  ({len(BS_PAIRS)} beamsplitters / layer)")
print(
    f"θ params: {N_LAYERS * PPL}  ({PPL} per layer: "
    f"rot={N_ROT} + sq={N_SQ} + BS={N_BS_PARAMS} + disp={N_DISP})"
)
print(f"λ params: {N_LAMBDA}  (logit-η + σ_φ per layer; jointly optimised)")
print(f"Noise   : Training (deterministic) → Eval (stochastic, σ_φ=0.02)")
print(f"Measure : mean quadrature over all {N_MODES} modes")
print(f"Epochs  : {EP}")
print(f"\nDatasets:")
print(
    f"  🔹 Fashion-MNIST (T-shirt vs Shirt) – Case 1: PCA→4D [PERFORMANCE EVALUATION]"
)
print(
    f"  🔹 Fashion-MNIST (T-shirt vs Shirt) – Case 2: PCA→2D [VISUALIZATION & BOUNDARIES]"
)
print(f"  🔹 Breast Cancer (Medical, Real-world) – Binary classification\n")

experiments = []

# Define all dataset loaders with configurations
dataset_loaders = [
    (
        "fashion_mnist_4d",
        lambda: fashion_mnist_data(n_components=4, use_case="performance"),
        "performance",
    ),
    (
        "fashion_mnist_2d",
        lambda: fashion_mnist_data(n_components=2, use_case="visualization"),
        "visualization",
    ),
    (
        "breast_cancer",
        lambda: breast_cancer_data(n_components=4, use_case="performance"),
        "performance",
    ),
    (
        "cifar10_4d",
        lambda: cifar10_data(n_components=4, n_samples=5000),
        "performance",
    ),
    (
        "cifar10_full_4d",
        lambda: cifar10_full_data(n_components=4, n_samples=5000),
        "performance",
    ),
]

for loader_name, loader_fn, use_case in dataset_loaders:
    X, y, name = loader_fn()
    Xt, Xv, Xte, yt, yv, yte = split_data(X, y)
    print(f"\n{'='*62}\nDataset : {name}")
    if use_case == "performance":
        print(f"  ► Use Case: PERFORMANCE EVALUATION (more features)")
    elif use_case == "visualization":
        print(f"  ► Use Case: VISUALIZATION & DECISION BOUNDARIES")
    print(f"Features: {X.shape[1]}  (cyclically mapped onto {N_MODES} modes)")
    print(f"Split   : train={len(Xt)} | val={len(Xv)} | test={len(Xte)}")
    print(f"{'='*62}")

    print("\n ▶ BASELINE (θ only – no noise channels) …")
    t0 = time.time()
    tb, _, hb = train(Xt, yt, Xv, yv, noise_assisted=False, epochs=EP)
    t_base = time.time() - t0
    rb = evaluate(Xte, yte, tb, None)

    print("\n ▶ NOISE-ASSISTED (joint optimisation of θ and λ=[η,σ_φ] per layer) …")
    t0 = time.time()
    tn, ln, hn = train(Xt, yt, Xv, yv, noise_assisted=True, epochs=EP)
    t_noise = time.time() - t0
    rn = evaluate(Xte, yte, tn, ln)

    print("\n ▶ CLASSICAL BASELINES …")
    t0 = time.time()
    rc = train_classical_baselines(Xt, yt, Xte, yte)
    t_classical = time.time() - t0

    print(f"\n  Results on test set:")
    print(
        f"  Baseline PQML      | Acc={rb['acc']:.4f}  F1={rb['f1']:.4f}  AUC={rb['auc']:.4f}"
    )
    print(
        f"  Noise-Assisted PQML| Acc={rn['acc']:.4f}  F1={rn['f1']:.4f}  AUC={rn['auc']:.4f}"
    )
    print(
        f"  Logistic Regression| Acc={rc['LogReg']['acc']:.4f}  F1={rc['LogReg']['f1']:.4f}  AUC={rc['LogReg']['auc']:.4f}"
    )
    print(
        f"  SVM (RBF)          | Acc={rc['SVM']['acc']:.4f}  F1={rc['SVM']['f1']:.4f}  AUC={rc['SVM']['auc']:.4f}"
    )

    experiments.append(
        {
            "name": name,
            "use_case": use_case,
            "hb": hb,
            "hn": hn,
            "rb": rb,
            "rn": rn,
            "rc": rc,
            "tb": (tb, None),
            "tn": (tn, ln),
            "Xte": Xte,
            "yte": yte,
            "t_base": t_base,
            "t_noise": t_noise,
            "t_classical": t_classical,
        }
    )

# =============================================================================
#  Master Figure (Updated for 3 datasets)
# =============================================================================

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor("#F8F9FA")
gs = gridspec.GridSpec(9, 6, figure=fig, hspace=0.60, wspace=0.40)

# ── Row 0: Title ──────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.text(
    0.5,
    0.72,
    "Noise-Assisted Photonic Quantum Machine Learning",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="#1A252F",
)
ax0.text(
    0.5,
    0.38,
    f"Fashion-MNIST 4D (Performance) | Fashion-MNIST 2D (Visualization) | Breast Cancer (Medical)\n"
    f"Baseline PQML (θ only)  vs.  Noise-Assisted PQML (joint θ, λ optimisation)  |  "
    f"{N_MODES} modes × {N_LAYERS} layers  |  {EP} epochs",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#333",
)
ax0.text(
    0.5,
    0.05,
    "A.M.A.S.D. Alagiyawanna – 215506J | "
    "Dept of Computational Mathematics, University of Moratuwa, 2026",
    ha="center",
    va="center",
    fontsize=9,
    color="#777",
)

# ── Rows 1-2: Loss & accuracy curves (3 datasets) ────────────────────────────
for ci, exp in enumerate(experiments):
    # Loss curves
    ax_l = fig.add_subplot(gs[1, ci * 2 : ci * 2 + 2])
    # Accuracy curves
    ax_a = fig.add_subplot(gs[2, ci * 2 : ci * 2 + 2])
    ep_r = range(1, EP + 1)

    ax_l.plot(ep_r, exp["hb"]["tl"], c=CR, lw=1.5, alpha=0.4, ls="--")
    ax_l.plot(ep_r, exp["hb"]["vl"], c=CR, lw=2, label="Baseline (val)")
    ax_l.plot(ep_r, exp["hn"]["tl"], c=CG, lw=1.5, alpha=0.4, ls="--")
    ax_l.plot(ep_r, exp["hn"]["vl"], c=CG, lw=2, label="Noise-Assisted (val)")
    ax_l.set_title(f"BCE Loss – {exp['name']}", fontsize=10)
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Loss")
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3)

    ax_a.plot(ep_r, exp["hb"]["va"], c=CR, lw=2, label="Baseline")
    ax_a.plot(ep_r, exp["hn"]["va"], c=CG, lw=2, label="Noise-Assisted")
    ax_a.set_title(f"Validation Accuracy – {exp['name']}", fontsize=10)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1.1)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

# ── Rows 3-4: Confusion matrices (6 total: 3 datasets x 2 models) ──────────────
for ci, exp in enumerate(experiments):
    # Baseline CM
    ax_cm1 = fig.add_subplot(gs[3 + ci // 2, (ci % 2) * 3 : (ci % 2) * 3 + 1])
    # Noise-assisted CM
    ax_cm2 = fig.add_subplot(gs[3 + ci // 2, (ci % 2) * 3 + 1 : (ci % 2) * 3 + 2])

    sns.heatmap(
        exp["rb"]["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm1,
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_cm1.set_title(f"Baseline\n{exp['name']}", fontsize=9)

    sns.heatmap(
        exp["rn"]["cm"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax_cm2,
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_cm2.set_title(f"Noise-Assisted\n{exp['name']}", fontsize=9)

# ── Row 5: Probability distribution (MNIST 2D only) ──────────────────────────
# Only plot if we have at least 2 experiments
if len(experiments) >= 2:
    ax_pb = fig.add_subplot(gs[5, 0:3])
    mnist_2d_exp = experiments[1]  # MNIST 2D is index 1
    for key, col, th, lm in [
        ("Baseline", CR, mnist_2d_exp["tb"][0], None),
        ("Noise-Assisted", CG, mnist_2d_exp["tn"][0], mnist_2d_exp["tn"][1]),
    ]:
        pr = batch_forward(mnist_2d_exp["Xte"], th, lm)
        ax_pb.hist(
            pr[mnist_2d_exp["yte"] == 0],
            bins=15,
            alpha=0.55,
            color=col,
            density=True,
            label=f"{key} Cl-0",
        )
        ax_pb.hist(
            pr[mnist_2d_exp["yte"] == 1],
            bins=15,
            alpha=0.35,
            color=col,
            density=True,
            edgecolor="k",
            lw=0.6,
            label=f"{key} Cl-1",
        )
    ax_pb.axvline(0.5, color="k", lw=1.5, ls=":", label="Decision boundary (0.5)")
    ax_pb.set_title(
        f"Prediction Probability Distribution – {mnist_2d_exp['name']}", fontsize=10
    )
    ax_pb.set_xlabel("P(class = 1)")
    ax_pb.set_ylabel("Density")
    ax_pb.legend(fontsize=7, ncol=2)
    ax_pb.grid(alpha=0.3)

# ── Row 5: Decision boundary (Fashion-MNIST 2D only) ─────────────────────────
# Fashion-MNIST 2D is the second experiment (index 1)
if len(experiments) >= 2:
    fashion_2d_exp = experiments[1]
    ax_d1 = fig.add_subplot(gs[5, 3:5])
    ax_d2 = fig.add_subplot(gs[5, 5:])
    draw_boundary(
        fashion_2d_exp["Xte"],
        fashion_2d_exp["yte"],
        fashion_2d_exp["tb"][0],
        None,
        ax_d1,
        f"Baseline – {fashion_2d_exp['name']}",
    )
    draw_boundary(
        fashion_2d_exp["Xte"],
        fashion_2d_exp["yte"],
        fashion_2d_exp["tn"][0],
        fashion_2d_exp["tn"][1],
        ax_d2,
        f"Noise-Assisted – {fashion_2d_exp['name']}",
    )

# ── Rows 6-7: Bar comparison (Accuracy & AUC) ─────────────────────────────────
dnames = [e["name"] for e in experiments]
x = np.arange(len(dnames))
w = 0.25

ax_bar_acc = fig.add_subplot(gs[6:8, 0:3])
ax_bar_auc = fig.add_subplot(gs[6:8, 3:6])

for ax, metric, ylabel in [
    (ax_bar_acc, "acc", "Test Accuracy"),
    (ax_bar_auc, "auc", "Test AUC-ROC"),
]:
    bv = [e["rb"][metric] for e in experiments]
    nv = [e["rn"][metric] for e in experiments]
    lv = [e["rc"]["LogReg"][metric] for e in experiments]
    sv = [e["rc"]["SVM"][metric] for e in experiments]

    b1 = ax.bar(
        x - 1.5 * w, bv, w, label="Baseline PQML", color=CR, alpha=0.85, edgecolor="k"
    )
    b2 = ax.bar(
        x - 0.5 * w,
        nv,
        w,
        label="Noise-Assisted PQML",
        color=CG,
        alpha=0.85,
        edgecolor="k",
    )
    b3 = ax.bar(
        x + 0.5 * w,
        lv,
        w,
        label="Logistic Regression",
        color="#90CAF9",
        alpha=0.85,
        edgecolor="k",
    )
    b4 = ax.bar(
        x + 1.5 * w,
        sv,
        w,
        label="SVM (RBF)",
        color="#FFB74D",
        alpha=0.85,
        edgecolor="k",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dnames, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel} Comparison – All Models", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)

out = "outputs/pqml_4mode_results.png"
fig.savefig(out, dpi=145, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nMaster figure → {out}")

# =============================================================================
#  Save Individual Images (Separate files, no captions)
# =============================================================================

print("\nSaving individual images …")

# 1. Loss curves (per dataset)
for ci, exp in enumerate(experiments):
    fig, ax = plt.subplots(figsize=(8, 6))
    ep_r = range(1, EP + 1)
    ax.plot(ep_r, exp["hb"]["tl"], c=CR, lw=1.5, alpha=0.4, ls="--")
    ax.plot(ep_r, exp["hb"]["vl"], c=CR, lw=2, label="Baseline (val)")
    ax.plot(ep_r, exp["hn"]["tl"], c=CG, lw=1.5, alpha=0.4, ls="--")
    ax.plot(ep_r, exp["hn"]["vl"], c=CG, lw=2, label="Noise-Assisted (val)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    out_name = f"outputs/loss_curves_dataset{ci+1}.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

# 2. Accuracy curves (per dataset)
for ci, exp in enumerate(experiments):
    fig, ax = plt.subplots(figsize=(8, 6))
    ep_r = range(1, EP + 1)
    ax.plot(ep_r, exp["hb"]["va"], c=CR, lw=2, label="Baseline")
    ax.plot(ep_r, exp["hn"]["va"], c=CG, lw=2, label="Noise-Assisted")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    out_name = f"outputs/accuracy_curves_dataset{ci+1}.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

# 3. Confusion matrices (baseline and noise-assisted, per dataset)
for ci, exp in enumerate(experiments):
    # Baseline CM
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        exp["rb"]["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    fig.patch.set_facecolor("white")
    out_name = f"outputs/confusion_matrix_baseline_dataset{ci+1}.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

    # Noise-assisted CM
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        exp["rn"]["cm"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax,
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    fig.patch.set_facecolor("white")
    out_name = f"outputs/confusion_matrix_noise_assisted_dataset{ci+1}.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

# 4. Probability distribution (Fashion-MNIST 2D only - index 1)
if len(experiments) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    fashion_2d_exp = experiments[1]
    for key, col, th, lm in [
        ("Baseline", CR, fashion_2d_exp["tb"][0], None),
        ("Noise-Assisted", CG, fashion_2d_exp["tn"][0], fashion_2d_exp["tn"][1]),
    ]:
        pr = batch_forward(fashion_2d_exp["Xte"], th, lm)
        ax.hist(
            pr[fashion_2d_exp["yte"] == 0],
            bins=15,
            alpha=0.55,
            color=col,
            density=True,
            label=f"{key} Cl-0",
        )
        ax.hist(
            pr[fashion_2d_exp["yte"] == 1],
            bins=15,
            alpha=0.35,
            color=col,
            density=True,
            edgecolor="k",
            lw=0.6,
            label=f"{key} Cl-1",
        )
    ax.axvline(0.5, color="k", lw=1.5, ls=":", label="Decision boundary (0.5)")
    ax.set_xlabel("P(class = 1)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    out_name = "outputs/probability_distribution.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

# 5. Decision boundaries (Fashion-MNIST 2D only - index 1)
if len(experiments) >= 2:
    fashion_2d_exp = experiments[1]
    # Baseline boundary
    fig, ax = plt.subplots(figsize=(8, 7))
    h = 0.05
    x0, x1 = (
        fashion_2d_exp["Xte"][:, 0].min() - 0.1,
        fashion_2d_exp["Xte"][:, 0].max() + 0.1,
    )
    y0, y1 = (
        fashion_2d_exp["Xte"][:, 1].min() - 0.1,
        fashion_2d_exp["Xte"][:, 1].max() + 0.1,
    )
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z = batch_forward(
        np.c_[xx.ravel(), yy.ravel()], fashion_2d_exp["tb"][0], None
    ).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=40, cmap="RdYlGn", alpha=0.72)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 0, 0],
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 0, 1],
        c=CR,
        edgecolors="k",
        s=28,
        zorder=5,
        label="Class 0",
    )
    ax.scatter(
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 1, 0],
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 1, 1],
        c=CG,
        edgecolors="k",
        s=28,
        zorder=5,
        label="Class 1",
    )
    ax.legend(fontsize=10)
    fig.patch.set_facecolor("white")
    out_name = "outputs/decision_boundary_baseline.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

    # Noise-assisted boundary
    fig, ax = plt.subplots(figsize=(8, 7))
    Z = batch_forward(
        np.c_[xx.ravel(), yy.ravel()], fashion_2d_exp["tn"][0], fashion_2d_exp["tn"][1]
    ).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=40, cmap="RdYlGn", alpha=0.72)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 0, 0],
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 0, 1],
        c=CR,
        edgecolors="k",
        s=28,
        zorder=5,
        label="Class 0",
    )
    ax.scatter(
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 1, 0],
        fashion_2d_exp["Xte"][fashion_2d_exp["yte"] == 1, 1],
        c=CG,
        edgecolors="k",
        s=28,
        zorder=5,
        label="Class 1",
    )
    ax.legend(fontsize=10)
    fig.patch.set_facecolor("white")
    out_name = "outputs/decision_boundary_noise_assisted.png"
    fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")

# 6. Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(12, 7))
dnames = [e["name"] for e in experiments]
x = np.arange(len(dnames))
w = 0.25

bv = [e["rb"]["acc"] for e in experiments]
nv = [e["rn"]["acc"] for e in experiments]
lv = [e["rc"]["LogReg"]["acc"] for e in experiments]
sv = [e["rc"]["SVM"]["acc"] for e in experiments]

ax.bar(x - 1.5 * w, bv, w, label="Baseline PQML", color=CR, alpha=0.85, edgecolor="k")
ax.bar(
    x - 0.5 * w, nv, w, label="Noise-Assisted PQML", color=CG, alpha=0.85, edgecolor="k"
)
ax.bar(
    x + 0.5 * w,
    lv,
    w,
    label="Logistic Regression",
    color="#90CAF9",
    alpha=0.85,
    edgecolor="k",
)
ax.bar(
    x + 1.5 * w, sv, w, label="SVM (RBF)", color="#FFB74D", alpha=0.85, edgecolor="k"
)

ax.set_xticks(x)
ax.set_xticklabels(dnames, fontsize=10)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.patch.set_facecolor("white")
out_name = "outputs/accuracy_comparison.png"
fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✓ {out_name}")

# 7. AUC comparison bar chart
fig, ax = plt.subplots(figsize=(12, 7))
bv = [e["rb"]["auc"] for e in experiments]
nv = [e["rn"]["auc"] for e in experiments]
lv = [e["rc"]["LogReg"]["auc"] for e in experiments]
sv = [e["rc"]["SVM"]["auc"] for e in experiments]

ax.bar(x - 1.5 * w, bv, w, label="Baseline PQML", color=CR, alpha=0.85, edgecolor="k")
ax.bar(
    x - 0.5 * w, nv, w, label="Noise-Assisted PQML", color=CG, alpha=0.85, edgecolor="k"
)
ax.bar(
    x + 0.5 * w,
    lv,
    w,
    label="Logistic Regression",
    color="#90CAF9",
    alpha=0.85,
    edgecolor="k",
)
ax.bar(
    x + 1.5 * w, sv, w, label="SVM (RBF)", color="#FFB74D", alpha=0.85, edgecolor="k"
)

ax.set_xticks(x)
ax.set_xticklabels(dnames, fontsize=10)
ax.set_ylabel("Test AUC-ROC", fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.patch.set_facecolor("white")
out_name = "outputs/auc_comparison.png"
fig.savefig(out_name, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✓ {out_name}")

# ── Console summary table ──────────────────────────────────────────────────────
print(f"\n{'═'*95}")
print(f"{'Dataset':<35}{'Model':<25}{'Acc':>7}{'F1':>7}{'AUC':>7}{'Time(s)':>10}")
print(f"{'═'*95}")
for e in experiments:
    use_case_str = f"[{e['use_case'].upper()}]" if e["use_case"] else ""
    dataset_label = f"{e['name']:<34}{use_case_str}"
    print(
        f"{dataset_label:<35}{'Baseline PQML':<25}"
        f"{e['rb']['acc']:>7.4f}{e['rb']['f1']:>7.4f}{e['rb']['auc']:>7.4f}{e['t_base']:>10.2f}"
    )
    print(
        f"{'':<35}{'Noise-Assisted PQML':<25}"
        f"{e['rn']['acc']:>7.4f}{e['rn']['f1']:>7.4f}{e['rn']['auc']:>7.4f}{e['t_noise']:>10.2f}"
    )
    print(
        f"{'':<35}{'Logistic Regression':<25}"
        f"{e['rc']['LogReg']['acc']:>7.4f}{e['rc']['LogReg']['f1']:>7.4f}{e['rc']['LogReg']['auc']:>7.4f}{0.001:>10.2f}"
    )
    print(
        f"{'':<35}{'SVM (RBF)':<25}"
        f"{e['rc']['SVM']['acc']:>7.4f}{e['rc']['SVM']['f1']:>7.4f}{e['rc']['SVM']['auc']:>7.4f}{0.001:>10.2f}"
    )
    print(f"{'─'*95}")
print(f"{'═'*95}")

print("\n✓ Complete.  Output: pqml_4mode_results.png")
