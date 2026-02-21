# Quantum Circuit Design

[← Back to README](../../README.md) · [MDP & Rewards](mdp_and_rewards.md) · **Circuit Design** · [Training Pipeline →](training_pipeline.md)

---

All quantum components are defined in [`vqdqn_circuit.py`](../../q_rlstc/quantum/vqdqn_circuit.py).

## Circuit Architecture

```
[Angle Encoding] → [Variational Layer 1] → [Data Re-upload] → [Variational Layer 2] → [Measurement]
```

## 1. Angle Encoding

Each feature is mapped to a single-qubit rotation:

```
For each qubit i ∈ [0, n_qubits):
    Apply RY(2 · arctan(xᵢ)) to qubit i
```

```python
def angle_encode(features, scaling='arctan'):
    """Maps (-∞, ∞) → (-π, π) monotonically."""
    return 2 * np.arctan(features)
```

**Why `arctan` scaling?**
- Unbounded features → bounded angles (no wrap-around)
- Sign-preserving: positive features → positive angles
- Graceful saturation: extreme values don't cause numerical issues
- No normalisation required — each feature encoded independently

> [!WARNING]
> **Saturation at large magnitudes.** Because `arctan` asymptotes at ±π/2, features with large absolute values lose discriminative power: e.g. `arctan(10000) ≈ arctan(20000) ≈ π/2`. If OD features are on the order of thousands, a dataset-dependent scaling factor (e.g. dividing by the feature's observed range or standard deviation) should be applied **before** angle encoding. The code supports `scaling='linear'` and `scaling='sigmoid'` as alternatives if pre-normalised features are available.

**Why _not_ amplitude encoding for the policy?**
Amplitude encoding would require normalising the state vector (losing magnitude information), more complex prep circuits, and would break parameter-shift gradient computation.

## 2. Variational Layers

### Version A/B/D — Hardware-Efficient Ansatz (HEA)

Each variational layer applies:

#### Rotation Block

```
Per qubit i: RY(θ₂ᵢ) → RZ(θ₂ᵢ₊₁)
```

2 trainable parameters per qubit per layer (RY + RZ).

#### Entanglement Block — Linear CNOT Chain

```
Qubit 0: ─RY(θ₀)─RZ(θ₁)─●────────────────────────
                          │
Qubit 1: ─RY(θ₂)─RZ(θ₃)─X──●─────────────────────
                             │
Qubit 2: ─RY(θ₄)─RZ(θ₅)────X──●──────────────────
                                │
Qubit 3: ─RY(θ₆)─RZ(θ₇)───────X──●───────────────
                                   │
Qubit 4: ─RY(θ₈)─RZ(θ₉)──────────X───────────────
```

**Linear entanglement** (vs. ring or full) is chosen for:
- Lower circuit depth
- Fewer two-qubit gates = less noise
- Sufficient expressivity for 5–8 qubit systems
- Simpler transpilation to hardware

### Version C — Equivariant Quantum Circuit (EQC)

Version C uses an SO(2)-equivariant circuit designed to respect the rotational symmetry of GPS coordinate data:

```
Per qubit i: RZ(θᵢ)        — Phase rotation only (preserves equivariance)

Entanglement: Circular CNOT ring
    qubit 0 → qubit 1 → qubit 2 → ... → qubit N-1 → qubit 0
```

This results in ~24 parameters with a more constrained (but symmetry-appropriate) structure.

> [!NOTE]
> **Ansatz classification.** The HEA RY-RZ-CNOT structure is a two-local ansatz closest to **Circuit 2** in Sim et al. (2019), "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms" ([arXiv:1905.10876](https://arxiv.org/abs/1905.10876), Figure 2). The EQC in Version C is custom-designed for spatial data invariance.

## 3. Data Re-uploading

Between variational layers, the input state is re-encoded:

```
Encoding → Layer 1 → Re-encoding → Layer 2 → Measurement
```

**Why?** Standard VQCs suffer from limited input-dependent expressivity. Data re-uploading:
- Increases expressivity without increasing depth significantly
- Allows each layer to process input differently
- Empirically improves trainability

From Pérez-Salinas et al. (2020), "Data re-uploading for a universal quantum classifier."

## 4. Q-Value Extraction

### Version A — Standard Readout

```python
Q(EXTEND) = ⟨Z₀⟩ × scale₀ + bias₀    # ∈ [-1, 1] → scaled
Q(CUT)    = ⟨Z₁⟩ × scale₁ + bias₁
```

### Version B — Multi-Observable Readout

```python
Q(EXTEND) = w₀·⟨Z₀⟩ + w₁·⟨Z₂Z₃⟩     # Single-qubit + parity
Q(CUT)    = w₂·⟨Z₁⟩ + w₃·⟨Z₄Z₅⟩
```

Parity observables `⟨ZₐZᵦ⟩` capture two-qubit correlations, enabling richer output encoding.

### Version C — Softmax Distribution

```python
# Full distribution from all measurement probabilities
π(EXTEND|s) = softmax(f(counts))[0]
π(CUT|s)    = softmax(f(counts))[1]
π(DROP|s)   = softmax(f(counts))[2]
```

SAC agent samples actions from this distribution and uses entropy regularisation.

### Version D — Multi-Qubit Z Readout

```python
Q(EXTEND) = ⟨Z₀⟩ × scale₀ + bias₀
Q(CUT)    = ⟨Z₁⟩ × scale₁ + bias₁
Q(SKIP)   = ⟨Z₂⟩ × scale₂ + bias₂   # Optional, when SKIP enabled
```

### Expectation Computation

```python
def compute_expectation_from_counts(counts, shots, qubit_idx, n_qubits):
    """⟨Zᵢ⟩ from measurement bitstrings."""
    expectation = 0.0
    for bitstring, count in counts.items():
        bit = bitstring[-(qubit_idx + 1)]   # Little-endian convention
        sign = 1 if bit == '0' else -1
        expectation += sign * count
    return expectation / shots               # ∈ [-1, 1]
```

## 5. Gates Used

| Gate | Role | Count (per variational layer) | Count (encoding layer) |
|---|---|---|---|
| **RY** | Variational rotation / state encoding | n_qubits | n_qubits |
| **RZ** | Phase rotation (variational only) | n_qubits (HEA) / n_qubits (EQC) | 0 |
| **CNOT** | Entanglement | n_qubits − 1 (linear) / n_qubits (circular, C) | 0 |

## 6. Parameter Count

| Component | Version A (5q) | Version B (8q) | Version C (6q) | Version D (5q) |
|---|---|---|---|---|
| Encoding angles | 5 (fixed) | 8 (fixed) | 6 (fixed) | 5 (fixed) |
| Rotations per layer | 10 (5×2) | 16 (8×2) | ~12 | 10 (5×2) |
| Layers | 2 | 2 | 2 | 3 |
| **Total trainable** | **20** | **32** | **~24** | **30** |

Compare to classical RLSTC: ~450 parameters in a 5→64→2 MLP.

## 7. Circuit Depth Analysis

| Component | Depth |
|---|---|
| Encoding layer | 1 (parallel RY) |
| Variational layer 1 | ~4 (rotations + CNOT chain) |
| Re-encoding | 1 |
| Variational layer 2 | ~4 |
| (Version D: layer 3) | ~4 |
| Measurement | 1 |
| **Total (A/B)** | **~11** |
| **Total (C)** | **~9** (EQC is more compact) |
| **Total (D)** | **~15** (3 variational layers) |

This shallow depth keeps the circuit within the coherence time of:
- **IBM Eagle** (~100μs T₂): Estimated fidelity ~85% (A/B), ~80% (D)
- **IBM Heron** (~200μs T₂): Estimated fidelity ~95% (A/B), ~90% (D)

## The Encoding Dichotomy

Q-RLSTC uses _two_ encoding strategies for different purposes:

| Component | Encoding | Qubits | Purpose |
|---|---|---|---|
| VQ-DQN Policy | Angle | n_features | Fast, repeated; gradients available |
| Distance Estimation (optional) | Amplitude | log₂(n) | Required for swap test inner product |

This separation is intentional — a single encoding scheme would force compromises in both use cases. See [Distance & Clustering](distance_and_clustering.md) for the swap test details.

---

**Next:** [Training Pipeline →](training_pipeline.md)
