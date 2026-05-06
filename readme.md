# Noise-Assisted Photonic Quantum Machine Learning (NA-PQML)

## Theoretical Extensions of Machine Learning Using Noise-Assisted Quantum Algorithms in Photonic Architectures

---

## Author

**A.M.A.S.D. Alagiyawanna (215506J)**  
BSc Hons in Artificial Intelligence  
Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  
Sri Lanka

---

## Supervisor

**Senior Professor Asoka Karunananda**  
Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  
Sri Lanka

---

# Overview

This repository contains the implementation of the proposed:

> **Noise-Assisted Photonic Quantum Machine Learning (NA-PQML)** framework

developed as part of the undergraduate research thesis:

> *"Theoretical Extensions of Machine Learning Using Noise-Assisted Quantum Algorithms in Photonic Architectures"*

The project introduces a novel learning paradigm where environmental noise in photonic quantum systems is treated as a constructive resource instead of an unwanted disturbance.

The framework is inspired by:

- Environmental-Assisted Quantum Transport (ENAQT)
- Variational Quantum Circuits (VQC)
- Hybrid Quantum-Classical Optimization
- Noise-aware learning dynamics

---

# Research Motivation

Traditional Photonic Quantum Machine Learning (PQML) systems primarily focus on:

- Noise suppression
- Error mitigation
- Decoherence reduction

However, real-world photonic systems are inherently noisy due to:

- Photon loss
- Phase fluctuations
- Detector inefficiencies
- Thermal interference

This project proposes that controlled environmental noise can:

- Improve generalization
- Stabilize optimization
- Enhance robustness
- Improve convergence behavior

---

# Implementations Included

This repository contains two implementations of the proposed framework.

---

# 1. Strawberry Fields Photonic Quantum Implementation

**File:** `pqml_SF.py`

This implementation uses:

- Strawberry Fields Gaussian backend
- Real photonic quantum circuit simulation
- Continuous-variable (CV) quantum computing
- Native photonic noise channels

## Features

- 4 optical modes (qumodes)
- 2 variational layers
- Gaussian-state simulation
- LossChannel noise modeling
- Phase diffusion noise
- Hybrid optimization using L-BFGS-B

## Photonic Gates Used

- `Rgate`
- `Sgate`
- `BSgate`
- `Dgate`
- `LossChannel`

## Backend

- Strawberry Fields Gaussian Engine

---

# 2. Hardcoded Simulation Model

**File:** `pqml_hardcoded.py`

This implementation recreates the photonic circuit mathematically using:

- Symplectic transformations
- Quadrature mean evolution
- Analytical photonic gate operations

## Purpose

- Lightweight simulation
- Faster experimentation
- Theoretical validation
- Comparison against full photonic simulation

---

# Datasets Used

The framework was evaluated on:

| Dataset | Task |
|---|---|
| Fashion-MNIST | T-shirt vs Shirt Classification |
| Breast Cancer Dataset | Binary Medical Classification |
| CIFAR-10 | Airplane vs Automobile |

## Preprocessing

- PCA dimensionality reduction
- MinMax normalization

---

# Quantum Circuit Architecture

## Configuration

| Component | Value |
|---|---|
| Optical Modes | 4 |
| Variational Layers | 2 |
| Parameters per Layer | 26 |
| Total Circuit Parameters | 52 |
| Noise Parameters | 4 |

---

# Variational Layer Structure

Each layer contains:

1. Rotation Gates  
2. Squeezing Gates  
3. Beamsplitters  
4. Noise Channels  
5. Displacement Gates  

---

# Noise-Assisted Learning

The proposed framework jointly optimizes:

- Quantum circuit parameters (θ)
- Noise parameters (λ)

Noise parameters include:

- Photon loss transmissivity (η)
- Phase diffusion strength (σφ)

---

# Installation

## Python Version

Recommended:

```bash
Python 3.10
```

Compatible:

```bash
Python >=3.9,<3.12
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Main Dependencies

- numpy
- scipy
- scikit-learn
- tensorflow
- strawberryfields
- matplotlib
- seaborn
- pandas

---

# Running the Project

## Strawberry Fields Implementation

```bash
python pqml_SF.py
```

---

## Hardcoded Simulation

```bash
python pqml_hardcoded.py
```

---

# Output

The framework generates:

- Training curves
- Validation accuracy plots
- Confusion matrices
- Decision boundaries
- Comparative benchmark visualizations

Generated outputs are saved in:

```bash
outputs/
```

---

# Key Contributions

This research introduces:

- Noise-assisted photonic quantum learning
- Joint optimization of circuit and noise parameters
- Controlled quantum noise injection
- Adaptive PQML framework for noisy environments
- Dual implementation framework:
  - Full photonic simulation
  - Hardcoded mathematical simulation

---

# Research Keywords

- Quantum Machine Learning
- Photonic Quantum Computing
- Variational Quantum Circuits
- Noise-Assisted Learning
- Hybrid Quantum-Classical Optimization
- Continuous Variable Quantum Computing
- Strawberry Fields

---

# Thesis Information

## Title

*Theoretical Extensions of Machine Learning Using Noise-Assisted Quantum Algorithms in Photonic Architectures*

## Institution

Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  
Sri Lanka

## Degree Program

BSc Hons in Artificial Intelligence

---

# Citation

If you use this work, please cite:

```bibtex
@thesis{alagiyawanna2026napqml,
  author  = {A.M.A.S.D. Alagiyawanna},
  title   = {Theoretical Extensions of Machine Learning Using Noise-Assisted Quantum Algorithms in Photonic Architectures},
  school  = {University of Moratuwa},
  year    = {2026}
}
```

---

# Acknowledgements

This research was conducted under the supervision of:

**Senior Professor Asoka Karunananda**  
Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa

Special thanks to:

- University of Moratuwa
- Department of Computational Mathematics
- Strawberry Fields by Xanadu
- Open-source scientific computing community

---

# License

This project is intended for academic and research purposes.