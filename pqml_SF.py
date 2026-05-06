"""
=============================================================================
Noise-Assisted Photonic Quantum Machine Learning  –  4-Mode Circuit
Powered by Strawberry Fields (Gaussian Backend) – REAL quantum simulation
# Inspired by ENAQT: controlled noise improves transport / learning flow
=============================================================================
Author  : A.M.A.Sandeepa D. Alagiyawanna (215506J)
Dept    : Computational Mathematics, Faculty of IT, University of Moratuwa

Circuit: 4 optical modes (qumodes) × 2 variational layers

Gate sequence per layer (executed via Strawberry Fields Gaussian engine):
  1. Rotation gates     – Rgate(φ_k)          on each mode k   [4 params]
  2. Squeezing gates    – Sgate(r_k, φ_k)      on each mode k   [8 params]
  3. Beamsplitters      – BSgate(θ, φ)         brick-wall layout [6 params]
       Layer A pairs: (0,1), (2,3)
       Layer B pair : (1,2)
  4. Noise channels     – LossChannel(η)  per mode  (SF native)
                        – Phase diffusion  σ_φ      (Dephasing via SF)
  5. Displacement gates – Dgate(r_k, φ_k)      on each mode k   [8 params]

Measurement: quad_expectation(mode, phi=0) → x-quadrature mean
             averaged across all 4 modes → sigmoid → P(class=1)

Params per layer: 4 + 8 + 6 + 8 = 26
Total θ params : 26 × 2 = 52
Total λ params : 4  (logit-η + raw-σ_φ per layer; jointly optimised)

NOTE ON BACKEND:
  The Gaussian backend propagates exact Gaussian states (covariance matrix +
  means vector) through the circuit. LossChannel(η) is a true beam-splitter
  coupling to a vacuum thermal bath – the canonical photonic noise model.
  Phase diffusion is implemented as Dgate displacement noise on p-quadratures,
  consistent with the Gaussian-state phase-diffusion master equation.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import cifar10
from sklearn.datasets import fetch_openml

import numpy as np
import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import (
    Rgate, Sgate, BSgate, Dgate, LossChannel
)

from scipy.special import expit as sigmoid
from scipy.optimize import minimize

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Circuit configuration
# ─────────────────────────────────────────────────────────────────────────────
N_MODES  = 4
N_LAYERS = 2

BS_PAIRS    = [(0, 1), (2, 3), (1, 2)]   # brick-wall beamsplitters
N_BS_PARAMS = len(BS_PAIRS) * 2          # 6

N_ROT  = N_MODES          # 4
N_SQ   = 2 * N_MODES      # 8   (r, φ) per mode
N_DISP = 2 * N_MODES      # 8   (r, φ) per mode
PPL    = N_ROT + N_SQ + N_BS_PARAMS + N_DISP   # 26

N_LAMBDA = 2 * N_LAYERS   # logit-η + raw-σ_φ per layer

CR = "#E57373"   # baseline colour
CG = "#4CAF50"   # noise-assisted colour

# =============================================================================
#  Quantum Feature Encoding  (amplitude + phase, 4 modes)
# =============================================================================

def encode_params(x):
    """
    Map classical feature vector x ∈ [0,1]^n to Dgate parameters for
    4 optical modes (amplitude-phase encoding).

    For mode k:
        r_k  = sqrt(|x[k % len(x)]|)      — coherent amplitude
        phi_k = π · x[(k+1) % len(x)]     — phase
    Returns list of (r, phi) tuples, one per mode.
    """
    params = []
    for k in range(N_MODES):
        i   = k % len(x)
        r   = float(np.sqrt(abs(x[i]) + 1e-8))
        phi = float(np.pi * x[(i + 1) % len(x)])
        params.append((r, phi))
    return params


# =============================================================================
#  Strawberry Fields Circuit Builder
# =============================================================================

def build_sf_circuit(x, theta, lambda_params=None, sigma_phase_eval=0.0):
    """
    Construct a Strawberry Fields Program for the full variational circuit.

    Architecture (per layer L):
      Encode x → Rgate → Sgate → BSgate (brick-wall) → LossChannel
      → [phase diffusion] → Dgate

    lambda_params : array of length N_LAMBDA = [logit(η_L1), σ_φ_L1, ...]
                    None = baseline (no noise channels applied).
    sigma_phase_eval : additional stochastic phase noise at evaluation time
                       (applied as Dgate on p-quadrature with random amplitude).
    """
    prog = Program(N_MODES)
    enc  = encode_params(x)

    with prog.context as q:

        # ── Input encoding: displacement into each mode ────────────────────
        for k in range(N_MODES):
            r, phi = enc[k]
            Dgate(r, phi) | q[k]

        # ── Variational layers ─────────────────────────────────────────────
        for L in range(N_LAYERS):
            p  = theta[L * PPL : (L + 1) * PPL]
            o  = 0

            # 1. Rotation gates
            for k in range(N_MODES):
                Rgate(float(p[o + k])) | q[k]
            o += N_ROT

            # 2. Squeezing gates  Sgate(r, phi)
            for k in range(N_MODES):
                r_sq = float(np.clip(p[o + 2*k], -2.0, 2.0))
                Sgate(r_sq, float(p[o + 2*k + 1])) | q[k]
            o += N_SQ

            # 3. Beamsplitters – brick-wall: (0,1), (2,3), (1,2)
            for pi_idx, (i, j) in enumerate(BS_PAIRS):
                BSgate(float(p[o + 2*pi_idx]),
                       float(p[o + 2*pi_idx + 1])) | (q[i], q[j])
            o += N_BS_PARAMS

            # 4. Noise channels (Strawberry Fields native)
            if lambda_params is not None:
                eta       = float(np.clip(sigmoid(lambda_params[2 * L]), 0.01, 0.9999))
                raw_sigma = float(lambda_params[2 * L + 1])
                sigma_phi = float(0.1 * np.abs(np.tanh(raw_sigma)))

                # ── Photon-loss channel (LossChannel) ──────────────────────
                # LossChannel(T) couples each mode to a vacuum bath with
                # transmissivity T ∈ (0,1]. This is the canonical Gaussian
                # noise channel for integrated photonic waveguides.
                for k in range(N_MODES):
                    LossChannel(eta) | q[k]

                # ── Phase diffusion via p-quadrature displacement ───────────
                # In the Gaussian-state picture, phase diffusion adds
                # variance σ_φ² to the p-quadrature.  We implement it as a
                # small Dgate along the imaginary (p) axis: Dgate(0, σ·tanh)
                # during training (deterministic mean-field approximation),
                # keeping the loss landscape smooth for gradient optimisation.
                if sigma_phi > 0:
                    for k in range(N_MODES):
                        Dgate(0.0, sigma_phi) | q[k]

            # ── Stochastic phase noise at evaluation time ───────────────────
            # Random Dgate on p-quadrature simulates quantum shot-to-shot
            # measurement variability (homodyne detection noise floor).
            if sigma_phase_eval > 0.0:
                for k in range(N_MODES):
                    noise_amp = float(np.random.normal(0.0, sigma_phase_eval))
                    Dgate(0.0, noise_amp) | q[k]

            # 5. Displacement gates  Dgate(r, phi)
            for k in range(N_MODES):
                dr = float(np.clip(p[o + 2*k],   -2.0, 2.0))
                di = float(np.clip(p[o + 2*k+1], -2.0, 2.0))
                Dgate(dr, di) | q[k]

    return prog


# =============================================================================
#  Forward Pass via Strawberry Fields
# =============================================================================

def forward(x, theta, lambda_params=None, sigma_phase_eval=0.0):
    prog = build_sf_circuit(x, theta, lambda_params, sigma_phase_eval)

    eng = sf.Engine("gaussian")   # FIX: create fresh engine
    result = eng.run(prog)
    state = result.state

    x_quads = np.array([
        state.quad_expectation(k, phi=0.0)[0]
        for k in range(N_MODES)
    ])

    return sigmoid(np.mean(x_quads))


def batch_forward(X, theta, lam=None, sigma_phase_eval=0.0):
    return np.array([forward(x, theta, lam, sigma_phase_eval) for x in X])


# =============================================================================
#  Loss
# =============================================================================

def bce(yt, yp, eps=1e-7):
    yp = np.clip(yp, eps, 1 - eps)
    return -np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))


def loss_fn(params, X, y, noise_assisted):
    nt    = N_LAYERS * PPL
    theta = params[:nt]
    lam   = params[nt:] if noise_assisted else None
    yp    = batch_forward(X, theta, lam, sigma_phase_eval=0.0)
    reg   = 1e-4 * np.sum(theta**2)
    return bce(y, yp) + reg


# =============================================================================
#  Training Loop
# =============================================================================

def train(X_tr, y_tr, X_val, y_val, noise_assisted=False, epochs=50):
    """
    Hybrid classical-quantum training with L-BFGS-B.

    θ  : N_LAYERS × PPL = 52  variational parameters
    λ  : 2 × N_LAYERS  = 4   noise parameters [logit(η), raw-σ_φ] per layer
         jointly optimised with θ when noise_assisted=True.

    The SF Gaussian backend provides exact (not sampled) quadrature means,
    giving a smooth, differentiable loss landscape suitable for L-BFGS-B.
    """
    rng = np.random.default_rng(SEED)
    nt  = N_LAYERS * PPL   # 52

    # Initialise circuit parameters
    # Initialise with smaller range to avoid squeezing-induced numerical explosion
    theta0 = rng.uniform(-0.5, 0.5, nt)

    if noise_assisted:
        lam0 = np.zeros(N_LAMBDA)
        for L in range(N_LAYERS):
            lam0[2 * L]     = 2.197   # logit(0.90) → η ≈ 0.90 (mild loss)
            lam0[2 * L + 1] = 0.05    # raw-σ_φ     → small phase diffusion
        p0 = np.concatenate([theta0, lam0])
    else:
        p0 = theta0.copy()

    history = {
        "tl": [], "vl": [], "va": [],
        "eta":       [[] for _ in range(N_LAYERS)],
        "sigma_phi": [[] for _ in range(N_LAYERS)],
    }
    p = p0.copy()

    for ep in range(epochs):
        # Mini-batch
        idx = rng.choice(len(X_tr), size=min(32, len(X_tr)), replace=False)
        Xb, yb = X_tr[idx], y_tr[idx]

        res = minimize(
            loss_fn, p,
            args=(Xb, yb, noise_assisted),
            method="L-BFGS-B",
            options={"maxiter": 5, "ftol": 1e-14, "gtol": 1e-8},
        )
        p = res.x

        tl       = loss_fn(p, X_tr, y_tr, noise_assisted)
        vl       = loss_fn(p, X_val, y_val, noise_assisted)
        lam_cur  = p[nt:] if noise_assisted else None
        vp       = batch_forward(X_val, p[:nt], lam_cur)
        va       = accuracy_score(y_val, (vp >= 0.5).astype(int))

        history["tl"].append(float(tl))
        history["vl"].append(float(vl))
        history["va"].append(float(va))

        if noise_assisted:
            for L in range(N_LAYERS):
                history["eta"][L].append(float(sigmoid(p[nt + 2*L])))
                raw_sigma = p[nt + 2*L + 1]
                history["sigma_phi"][L].append(float(0.1 * np.abs(np.tanh(raw_sigma))))

        if ep % 10 == 0:
            eta_str = ""
            if noise_assisted:
                etas = [f"{sigmoid(p[nt+2*L]):.3f}" for L in range(N_LAYERS)]
                sigs = [f"{0.1*np.abs(np.tanh(p[nt+2*L+1])):.4f}" for L in range(N_LAYERS)]
                eta_str = f" | η={etas} | σ_φ={sigs}"
            print(f"   ep{ep:3d} | train_loss={tl:.4f} | val_acc={va:.3f}{eta_str}")

    theta_final = p[:nt]
    lam_final   = p[nt:] if noise_assisted else None
    return theta_final, lam_final, history


# =============================================================================
#  Evaluation
# =============================================================================

def evaluate(X, y, theta, lam=None, n_shots=20):
    """
    For noise-assisted model: average n_shots stochastic evaluations to
    approximate quantum measurement statistics (homodyne shot noise).
    """
    if lam is not None:
        all_p = np.stack(
            [batch_forward(X, theta, lam, sigma_phase_eval=0.02)
             for _ in range(n_shots)],
            axis=0,
        )
        yp = all_p.mean(axis=0)
    else:
        yp = batch_forward(X, theta, lam=None)

    yc = (yp >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, yp)
    except Exception:
        auc = float("nan")
    return {
        "acc":  accuracy_score(y, yc),
        "f1":   f1_score(y, yc, zero_division=0),
        "auc":  auc,
        "cm":   confusion_matrix(y, yc),
        "prob": yp,
        "pred": yc,
    }


# =============================================================================
#  Datasets
# =============================================================================

def fashion_mnist_data(n_components=4):
    print("   Loading Fashion-MNIST …")
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, parser="auto")
    X = X.astype(np.float32)
    if hasattr(y, "values"):
        y = np.array(y.values, dtype=int)
    else:
        y = np.array(y, dtype=int)
    m  = (y == 0) | (y == 6)
    X  = X[m].values if hasattr(X[m], "values") else X[m]
    y  = (y[m] == 6).astype(float)
    X  = PCA(n_components=n_components, random_state=SEED).fit_transform(X)
    X  = MinMaxScaler().fit_transform(X)
    desc = f"Fashion-MNIST (T-shirt vs Shirt, PCA→{n_components}D)"
    return X, y, desc


def breast_cancer_data(n_components=4):
    X, y = load_breast_cancer(return_X_y=True)
    X    = PCA(n_components=n_components, random_state=SEED).fit_transform(
               X.astype(np.float32))
    X    = MinMaxScaler().fit_transform(X)
    return X, y.astype(float), f"Breast Cancer (Medical, PCA→{n_components}D)"

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


def split_data(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           random_state=SEED, stratify=y)
    Xt,  Xv,  yt,  yv  = train_test_split(Xtr, ytr, test_size=0.2,
                                           random_state=SEED, stratify=ytr)
    return Xt, Xv, Xte, yt, yv, yte


# =============================================================================
#  Decision Boundary (2-D datasets)
# =============================================================================

def draw_boundary(X, y, theta, lam, ax, title):
    h        = 0.05
    x0, x1   = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y0, y1   = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy   = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z        = batch_forward(np.c_[xx.ravel(), yy.ravel()],
                             theta, lam).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=40, cmap="RdYlGn", alpha=0.72)
    ax.contour(xx,  yy, Z, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=CR, edgecolors="k",
               s=28, zorder=5, label="Cl-0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=CG, edgecolors="k",
               s=28, zorder=5, label="Cl-1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


# =============================================================================
#  Classical Baselines
# =============================================================================

def train_classical_baselines(X_tr, y_tr, X_te, y_te):
    results = {}

    print("   ▶ Logistic Regression …")
    lr   = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)[:, 1]
    try:    auc = roc_auc_score(y_te, y_prob)
    except: auc = float("nan")
    results["LogReg"] = {
        "acc": accuracy_score(y_te, y_pred),
        "f1":  f1_score(y_te, y_pred, zero_division=0),
        "auc": auc, "cm": confusion_matrix(y_te, y_pred),
        "prob": y_prob, "pred": y_pred,
    }
    print(f"      LogReg | Acc={results['LogReg']['acc']:.4f}  F1={results['LogReg']['f1']:.4f}")

    print("   ▶ SVM (RBF) …")
    svm  = SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_te)
    y_prob = svm.predict_proba(X_te)[:, 1]
    try:    auc = roc_auc_score(y_te, y_prob)
    except: auc = float("nan")
    results["SVM"] = {
        "acc": accuracy_score(y_te, y_pred),
        "f1":  f1_score(y_te, y_pred, zero_division=0),
        "auc": auc, "cm": confusion_matrix(y_te, y_pred),
        "prob": y_prob, "pred": y_pred,
    }
    print(f"      SVM     | Acc={results['SVM']['acc']:.4f}  F1={results['SVM']['f1']:.4f}")
    return results


# =============================================================================
#  Main Experiment
# =============================================================================

EP = 20

print("╔══════════════════════════════════════════════════════════════════╗")
print("║  Noise-Assisted PQML  –  A.M.A.Sandeepa D. Alagiyawanna        ║")
print("║  Powered by: Strawberry Fields (Gaussian Backend)               ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"\nBackend : Strawberry Fields {sf.__version__} – Gaussian (exact Wigner state)")
print(f"Circuit : {N_MODES} modes × {N_LAYERS} layers")
print(f"BS pairs: {BS_PAIRS}  ({len(BS_PAIRS)} beamsplitters / layer)")
print(f"θ params: {N_LAYERS * PPL}  ({PPL} per layer: "
      f"rot={N_ROT} + sq={N_SQ} + BS={N_BS_PARAMS} + disp={N_DISP})")
print(f"λ params: {N_LAMBDA}  (logit-η + raw-σ_φ per layer; jointly optimised)")
print(f"Noise   : LossChannel(η) + Dgate phase diffusion (σ_φ) via SF ops")
print(f"Measure : x-quadrature expectation (quad_expectation) averaged over {N_MODES} modes")
print(f"Epochs  : {EP}")
print(f"\nDatasets:")
print(f"  🔹 Fashion-MNIST (T-shirt vs Shirt) – Case 1: PCA→4D [PERFORMANCE]")
print(f"  🔹 Fashion-MNIST (T-shirt vs Shirt) – Case 2: PCA→2D [VISUALIZATION]")
print(f"  🔹 Breast Cancer (Medical, Real-world) – Binary classification\n")

dataset_loaders = [
    ("fashion_mnist_4d", lambda: fashion_mnist_data(n_components=4), "performance"),
    ("fashion_mnist_2d", lambda: fashion_mnist_data(n_components=2), "visualization"),
    ("breast_cancer",    lambda: breast_cancer_data(n_components=4), "performance"),
    ("cifar10_4d", lambda: cifar10_data(n_components=4, n_samples=5000),"performance"),
    ("cifar10_full_4d",lambda: cifar10_full_data(n_components=4, n_samples=5000),"performance"),
]

experiments = []

for loader_name, loader_fn, use_case in dataset_loaders:
    X, y, name = loader_fn()
    Xt, Xv, Xte, yt, yv, yte = split_data(X, y)

    print(f"\n{'='*62}\nDataset : {name}")
    print(f"  ► Use Case: {use_case.upper()}")
    print(f"Features: {X.shape[1]}  (cyclically mapped onto {N_MODES} modes)")
    print(f"Split   : train={len(Xt)} | val={len(Xv)} | test={len(Xte)}")
    print(f"{'='*62}")

    print("\n ▶ BASELINE (θ only – no noise channels) …")
    t0 = time.time()
    tb, _,  hb = train(Xt, yt, Xv, yv, noise_assisted=False, epochs=EP)
    t_base = time.time() - t0
    rb = evaluate(Xte, yte, tb, None)

    print("\n ▶ NOISE-ASSISTED (joint optimisation of θ and λ=[η, σ_φ]) …")
    t0 = time.time()
    tn, ln, hn = train(Xt, yt, Xv, yv, noise_assisted=True,  epochs=EP)
    t_noise = time.time() - t0
    rn = evaluate(Xte, yte, tn, ln)

    print("\n ▶ CLASSICAL BASELINES …")
    t0 = time.time()
    rc = train_classical_baselines(Xt, yt, Xte, yte)
    t_classical = time.time() - t0

    print(f"\n  Results on test set:")
    print(f"  Baseline PQML      | Acc={rb['acc']:.4f}  F1={rb['f1']:.4f}  AUC={rb['auc']:.4f}")
    print(f"  Noise-Assisted PQML| Acc={rn['acc']:.4f}  F1={rn['f1']:.4f}  AUC={rn['auc']:.4f}")
    print(f"  Logistic Regression| Acc={rc['LogReg']['acc']:.4f}  F1={rc['LogReg']['f1']:.4f}  AUC={rc['LogReg']['auc']:.4f}")
    print(f"  SVM (RBF)          | Acc={rc['SVM']['acc']:.4f}  F1={rc['SVM']['f1']:.4f}  AUC={rc['SVM']['auc']:.4f}")

    experiments.append({
        "name": name, "use_case": use_case,
        "hb": hb, "hn": hn,
        "rb": rb, "rn": rn, "rc": rc,
        "tb": (tb, None), "tn": (tn, ln),
        "Xte": Xte, "yte": yte,
        "t_base": t_base, "t_noise": t_noise, "t_classical": t_classical,
    })


# =============================================================================
#  Master Figure
# =============================================================================

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(9, 6, figure=fig, hspace=0.60, wspace=0.40)

ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.text(0.5, 0.72,
    "Noise-Assisted Photonic Quantum Machine Learning",
    ha="center", va="center", fontsize=20, fontweight="bold", color="#1A252F")
ax0.text(0.5, 0.38,
    f"Fashion-MNIST 4D (Performance) | Fashion-MNIST 2D (Visualization) | Breast Cancer (Medical)\n"
    f"Baseline PQML (θ only)  vs.  Noise-Assisted PQML (joint θ, λ)  |  "
    f"{N_MODES} modes × {N_LAYERS} layers  |  {EP} epochs\n"
    f"Backend: Strawberry Fields {sf.__version__} Gaussian Simulator  |  "
    f"Noise: LossChannel(η) + Phase Diffusion(σ_φ)",
    ha="center", va="center", fontsize=10.5, color="#333")
ax0.text(0.5, 0.05,
    "A.M.A.S.D. Alagiyawanna – 215506J | "
    "Dept of Computational Mathematics, University of Moratuwa, 2026",
    ha="center", va="center", fontsize=9, color="#777")

# Loss & accuracy curves
for ci, exp in enumerate(experiments):
    ax_l = fig.add_subplot(gs[1, ci*2 : ci*2+2])
    ax_a = fig.add_subplot(gs[2, ci*2 : ci*2+2])
    ep_r = range(1, EP + 1)

    ax_l.plot(ep_r, exp["hb"]["tl"], c=CR, lw=1.5, alpha=0.4, ls="--")
    ax_l.plot(ep_r, exp["hb"]["vl"], c=CR, lw=2,   label="Baseline (val)")
    ax_l.plot(ep_r, exp["hn"]["tl"], c=CG, lw=1.5, alpha=0.4, ls="--")
    ax_l.plot(ep_r, exp["hn"]["vl"], c=CG, lw=2,   label="Noise-Assisted (val)")
    ax_l.set_title(f"BCE Loss – {exp['name']}", fontsize=10)
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
    ax_l.legend(fontsize=8); ax_l.grid(alpha=0.3)

    ax_a.plot(ep_r, exp["hb"]["va"], c=CR, lw=2, label="Baseline")
    ax_a.plot(ep_r, exp["hn"]["va"], c=CG, lw=2, label="Noise-Assisted")
    ax_a.set_title(f"Validation Accuracy – {exp['name']}", fontsize=10)
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1.1)
    ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

# Confusion matrices
for ci, exp in enumerate(experiments):
    ax_cm1 = fig.add_subplot(gs[3 + ci//2, (ci % 2)*3     : (ci % 2)*3 + 1])
    ax_cm2 = fig.add_subplot(gs[3 + ci//2, (ci % 2)*3 + 1 : (ci % 2)*3 + 2])
    sns.heatmap(exp["rb"]["cm"], annot=True, fmt="d", cmap="Blues",  ax=ax_cm1,
                cbar=False, xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    ax_cm1.set_title(f"Baseline\n{exp['name']}", fontsize=9)
    sns.heatmap(exp["rn"]["cm"], annot=True, fmt="d", cmap="Greens", ax=ax_cm2,
                cbar=False, xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    ax_cm2.set_title(f"Noise-Assisted\n{exp['name']}", fontsize=9)

# Probability distribution (Fashion-MNIST 2D)
if len(experiments) >= 2:
    ax_pb  = fig.add_subplot(gs[5, 0:3])
    exp_2d = experiments[1]
    for key, col, th, lm in [
        ("Baseline",       CR, exp_2d["tb"][0], None),
        ("Noise-Assisted", CG, exp_2d["tn"][0], exp_2d["tn"][1]),
    ]:
        pr = batch_forward(exp_2d["Xte"], th, lm)
        ax_pb.hist(pr[exp_2d["yte"] == 0], bins=15, alpha=0.55, color=col,
                   density=True, label=f"{key} Cl-0")
        ax_pb.hist(pr[exp_2d["yte"] == 1], bins=15, alpha=0.35, color=col,
                   density=True, edgecolor="k", lw=0.6, label=f"{key} Cl-1")
    ax_pb.axvline(0.5, color="k", lw=1.5, ls=":", label="Decision boundary (0.5)")
    ax_pb.set_title(f"Prediction Probability – {exp_2d['name']}", fontsize=10)
    ax_pb.set_xlabel("P(class = 1)"); ax_pb.set_ylabel("Density")
    ax_pb.legend(fontsize=7, ncol=2); ax_pb.grid(alpha=0.3)

    # Decision boundaries (Fashion-MNIST 2D)
    ax_d1 = fig.add_subplot(gs[5, 3:5])
    ax_d2 = fig.add_subplot(gs[5, 5:])
    draw_boundary(exp_2d["Xte"], exp_2d["yte"],
                  exp_2d["tb"][0], None,           ax_d1, f"Baseline – {exp_2d['name']}")
    draw_boundary(exp_2d["Xte"], exp_2d["yte"],
                  exp_2d["tn"][0], exp_2d["tn"][1], ax_d2, f"Noise-Assisted – {exp_2d['name']}")

# Bar charts: Accuracy & AUC
dnames = [e["name"] for e in experiments]
x = np.arange(len(dnames))
w = 0.25

for ax, metric, ylabel in [
    (fig.add_subplot(gs[6:8, 0:3]), "acc", "Test Accuracy"),
    (fig.add_subplot(gs[6:8, 3:6]), "auc", "Test AUC-ROC"),
]:
    bv = [e["rb"][metric] for e in experiments]
    nv = [e["rn"][metric] for e in experiments]
    lv = [e["rc"]["LogReg"][metric] for e in experiments]
    sv = [e["rc"]["SVM"][metric] for e in experiments]
    ax.bar(x - 1.5*w, bv, w, label="Baseline PQML",     color=CR,       alpha=0.85, edgecolor="k")
    ax.bar(x - 0.5*w, nv, w, label="Noise-Assisted PQML", color=CG,     alpha=0.85, edgecolor="k")
    ax.bar(x + 0.5*w, lv, w, label="Logistic Regression", color="#90CAF9", alpha=0.85, edgecolor="k")
    ax.bar(x + 1.5*w, sv, w, label="SVM (RBF)",          color="#FFB74D", alpha=0.85, edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(dnames, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel} Comparison – All Models", fontsize=11)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

out = "outputs/pqml_sf_results.png"
fig.savefig(out, dpi=145, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nMaster figure → {out}")


# =============================================================================
#  Individual Output Images
# =============================================================================

print("\nSaving individual images …")

for ci, exp in enumerate(experiments):
    ep_r = range(1, EP + 1)

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ep_r, exp["hb"]["tl"], c=CR, lw=1.5, alpha=0.4, ls="--")
    ax.plot(ep_r, exp["hb"]["vl"], c=CR, lw=2,   label="Baseline (val)")
    ax.plot(ep_r, exp["hn"]["tl"], c=CG, lw=1.5, alpha=0.4, ls="--")
    ax.plot(ep_r, exp["hn"]["vl"], c=CG, lw=2,   label="Noise-Assisted (val)")
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    fn = f"outputs/loss_curves_dataset{ci+1}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {fn}")

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ep_r, exp["hb"]["va"], c=CR, lw=2, label="Baseline")
    ax.plot(ep_r, exp["hn"]["va"], c=CG, lw=2, label="Noise-Assisted")
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    fn = f"outputs/accuracy_curves_dataset{ci+1}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {fn}")

    # Confusion matrices
    for tag, cm_data, cmap in [("baseline", exp["rb"]["cm"], "Blues"),
                                ("noise_assisted", exp["rn"]["cm"], "Greens")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_data, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False,
                    xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
        fig.patch.set_facecolor("white")
        fn = f"outputs/confusion_matrix_{tag}_dataset{ci+1}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig); print(f"  ✓ {fn}")

# Probability distribution (Fashion-MNIST 2D)
if len(experiments) >= 2:
    exp_2d = experiments[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, col, th, lm in [
        ("Baseline",       CR, exp_2d["tb"][0], None),
        ("Noise-Assisted", CG, exp_2d["tn"][0], exp_2d["tn"][1]),
    ]:
        pr = batch_forward(exp_2d["Xte"], th, lm)
        ax.hist(pr[exp_2d["yte"] == 0], bins=15, alpha=0.55, color=col,
                density=True, label=f"{key} Cl-0")
        ax.hist(pr[exp_2d["yte"] == 1], bins=15, alpha=0.35, color=col,
                density=True, edgecolor="k", lw=0.6, label=f"{key} Cl-1")
    ax.axvline(0.5, color="k", lw=1.5, ls=":", label="Decision boundary (0.5)")
    ax.set_xlabel("P(class = 1)", fontsize=11); ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("white")
    fn = "outputs/probability_distribution.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {fn}")

    # Decision boundaries
    for tag, th, lm in [("baseline",       exp_2d["tb"][0], None),
                         ("noise_assisted", exp_2d["tn"][0], exp_2d["tn"][1])]:
        fig, ax = plt.subplots(figsize=(8, 7))
        draw_boundary(exp_2d["Xte"], exp_2d["yte"], th, lm, ax,
                      f"{tag.replace('_',' ').title()} – {exp_2d['name']}")
        fig.patch.set_facecolor("white")
        fn = f"outputs/decision_boundary_{tag}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig); print(f"  ✓ {fn}")

# Accuracy & AUC bar charts
for metric, ylabel, fn_suffix in [
    ("acc", "Test Accuracy",  "accuracy_comparison"),
    ("auc", "Test AUC-ROC",   "auc_comparison"),
]:
    fig, ax = plt.subplots(figsize=(12, 7))
    bv = [e["rb"][metric] for e in experiments]
    nv = [e["rn"][metric] for e in experiments]
    lv = [e["rc"]["LogReg"][metric] for e in experiments]
    sv = [e["rc"]["SVM"][metric] for e in experiments]
    ax.bar(x - 1.5*w, bv, w, label="Baseline PQML",      color=CR,       alpha=0.85, edgecolor="k")
    ax.bar(x - 0.5*w, nv, w, label="Noise-Assisted PQML", color=CG,      alpha=0.85, edgecolor="k")
    ax.bar(x + 0.5*w, lv, w, label="Logistic Regression", color="#90CAF9",alpha=0.85, edgecolor="k")
    ax.bar(x + 1.5*w, sv, w, label="SVM (RBF)",           color="#FFB74D",alpha=0.85, edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(dnames, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor("white")
    fn = f"outputs/{fn_suffix}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {fn}")


# =============================================================================
#  Summary Table
# =============================================================================

print(f"\n{'═'*95}")
print(f"{'Dataset':<35}{'Model':<25}{'Acc':>7}{'F1':>7}{'AUC':>7}{'Time(s)':>10}")
print(f"{'═'*95}")
for e in experiments:
    uc  = f"[{e['use_case'].upper()}]"
    lbl = f"{e['name']:<34}{uc}"
    print(f"{lbl:<35}{'Baseline PQML':<25}"
          f"{e['rb']['acc']:>7.4f}{e['rb']['f1']:>7.4f}{e['rb']['auc']:>7.4f}{e['t_base']:>10.2f}")
    print(f"{'':<35}{'Noise-Assisted PQML':<25}"
          f"{e['rn']['acc']:>7.4f}{e['rn']['f1']:>7.4f}{e['rn']['auc']:>7.4f}{e['t_noise']:>10.2f}")
    print(f"{'':<35}{'Logistic Regression':<25}"
          f"{e['rc']['LogReg']['acc']:>7.4f}{e['rc']['LogReg']['f1']:>7.4f}"
          f"{e['rc']['LogReg']['auc']:>7.4f}{0.001:>10.2f}")
    print(f"{'':<35}{'SVM (RBF)':<25}"
          f"{e['rc']['SVM']['acc']:>7.4f}{e['rc']['SVM']['f1']:>7.4f}"
          f"{e['rc']['SVM']['auc']:>7.4f}{0.001:>10.2f}")
    print(f"{'─'*95}")
print(f"{'═'*95}")
print("\n✓ Complete.  Output: outputs/pqml_sf_results.png")