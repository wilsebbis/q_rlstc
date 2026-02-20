# Q-RLSTC Technical Design Deep Dive

> How exactly Q-RLSTC differs from classical RLSTC, how Versions A/B/C/D diverge,
> and how the VQ-DQN design compares to dina-quantum's approach.

---

## 1. Classical RLSTC → Q-RLSTC: What Changed

| **Component** | **Classical RLSTC** | **Q-RLSTC** |
|---|---|---|
| **Policy Network** | Classical MLP | Variational Quantum Circuit (VQ-DQN) |
| **Parameters** | Thousands (neural net weights) | 20–56 (rotation angles) |
| **Model Size** | ~100 KB+ | **80–224 bytes** |
| **Optimizer** | Backpropagation + Adam | **SPSA** (gradient-free, 2 function evals) |
| **Exploration** | ε-greedy | ε-greedy (A/B) or **entropy-regularised SAC** (C) |
| **State Encoding** | Raw feature vector | **Angle encoding**: RY(2·arctan(x_i)) |
| **Hypothesis Space** | Unconstrained function class | Constrained by Hilbert space geometry |
| **Hardware Target** | GPU/CPU | **NISQ quantum processors** (5–8 qubits) |

### Why Quantum?

The key insight is **not** that quantum is faster — it isn't for this problem size. The advantages are:

1. **Extreme parameter efficiency**: 20 parameters vs thousands means the model can be serialised in 80 bytes, enabling federated learning on mobile devices
2. **Implicit regularisation**: The bounded Hilbert space and trigonometric activation functions (from angle encoding) naturally prevent overfitting
3. **Entanglement as feature interaction**: CNOT gates create correlations between features that would require many classical layers to represent
4. **NISQ hardware compatibility**: The shallow circuits (depth 2–4) run within coherence times of current IBM/IonQ hardware

---

## 2. The VQ-DQN Circuit Architecture

### Angle Encoding Layer

Each feature `x_i` from the state vector is encoded as a rotation angle on qubit `i`:

```
θ_i = 2 · arctan(x_i)
```

This maps ℝ → (−π, π) smoothly, avoiding the saturation problems of linear or sigmoid encodings. Specifically:

| Encoding | Formula | Range | Issue |
|---|---|---|---|
| **Arctan** (Q-RLSTC) | 2·arctan(x) | (−π, π) | None — smooth, bounded |
| Linear | π·x | Assumes x ∈ [−1,1] | Saturates outside range |
| Sigmoid | π·(2σ(x)−1) | (−π, π) | Flat gradients at extremes |

The arctan encoding was chosen because GPS-derived features (segment lengths, variances, curvatures) have unbounded ranges, and arctan handles arbitrary magnitudes gracefully.

### Variational Ansatz Layers

Each variational layer applies:

```
For each qubit i:
    RY(θ_{layer,i,0})    — Y-axis rotation (amplitude mixing)
    RZ(θ_{layer,i,1})    — Z-axis rotation (phase)
```

followed by an entanglement layer:

```
CNOT chain: qubit_0 → qubit_1 → qubit_2 → ... → qubit_{N-1}
```

This is a **Hardware-Efficient Ansatz (HEA)** — designed to minimise circuit depth on near-term quantum hardware where long gate chains cause decoherence.

> [!NOTE]
> The RY-RZ pairing (2 rotations per qubit per layer) was chosen over the more common RX-RY-RZ (3 rotations) to keep parameter count low. With 5 qubits × 2 layers × 2 rotations = **20 parameters total**.

### Measurement & Q-Value Readout

**Version A (Standard Readout):**
```
Q(s, EXTEND) = ⟨Z₀⟩    (Pauli-Z expectation on qubit 0)
Q(s, CUT)    = ⟨Z₁⟩    (Pauli-Z expectation on qubit 1)
```

Each expectation value is computed from measurement counts:
```
⟨Z_i⟩ = (count_0 − count_1) / total_shots
```
where `count_0` = number of times qubit i measured |0⟩, `count_1` = number of times measured |1⟩.

**Version B (Multi-Observable Readout):**
```
Q(s, EXTEND) = w₀·⟨Z₀⟩ + w₁·⟨Z₁Z₂⟩ + w₂·⟨Z₂Z₃⟩ + bias₀
Q(s, CUT)    = w₃·⟨Z₁⟩ + w₄·⟨Z₃Z₄⟩ + w₅·⟨Z₅Z₆⟩ + bias₁
```

The parity terms ⟨Z_aZ_b⟩ capture **entanglement-derived correlations** — information that single-qubit measurements cannot access. This is what makes Version B "quantum enhanced": it reads out richer information from the same circuit execution.

---

## 3. Version A vs B vs C vs D

### Architecture Comparison

| Feature | **Version A** | **Version B** | **Version C** | **Version D** |
|---|---|---|---|---|
| **Label** | Classical Parity (5q) | Quantum Enhanced (8q) | Next-Gen Q-RNN (6q) | VLDB Aligned (5q) |
| **Qubits** | 5 | 8 | 6 (5 data + 1 shadow) | 5 |
| **State Dim** | 5D | 8D | 5D + memory signal | 5D (VLDB exact) |
| **Parameters** | 20 | 32 | ~24 (EQC is more efficient) | **30** (5q × 3 layers) |
| **Model Size** | 80 bytes | 128 bytes | ~96 bytes | **120 bytes** |
| **State Features** | OD, baseline, lengths | OD, baseline, angles, curvature, density | OD, baseline, lengths + shadow | **OD_cut, OD_extend, baseline, start_hilbert, end_hilbert** |
| **Ansatz** | HEA (RY-RZ + linear CNOT) | HEA (RY-RZ + linear CNOT) | **EQC** (RZ + circular CNOT) | HEA (RY-RZ + linear CNOT) |
| **Readout** | ⟨Z₀⟩, ⟨Z₁⟩ | Linear combo of ⟨ZZ⟩ parity | Soft π(a\|s) via softmax | ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ |
| **Agent** | ε-greedy DQN | ε-greedy DQN | **SAC** (entropy-regularised) | ε-greedy DQN |
| **Actions** | 2 [EXTEND, CUT] | 2 [EXTEND, CUT] | **3** [EXTEND, CUT, DROP] | **3** [EXTEND, CUT, **SKIP**] |
| **Memory** | None | None | **Shadow qubit** (Q-RNN) | None |
| **Optimizer** | SPSA | SPSA | **m-SPSA** (momentum) | SPSA |
| **Shots** | Fixed (512 train, 4096 eval) | Fixed | **Adaptive** (32 → 512) | Fixed |

### Version A: Classical Parity (5q)

The baseline configuration. Matches classical RLSTC's decision quality with 5 qubits:

- **5 state features**: segment length, local variance, centroid distance, trajectory progress, segment count
- **Why 5 qubits?** One qubit per state feature — the simplest possible mapping
- **Readout**: Each action's Q-value is read from a single qubit's Z-expectation
- **Purpose**: Prove that a quantum circuit can match classical MLP performance with ~1000× fewer parameters

### Version B: Quantum Enhanced (8q)

Adds 3 additional state features that leverage the quantum circuit's structure:

- **+3 features** (8D total): angle spread variance, curvature gradient, segment density
  - `angle_spread`: Variance of arctan-encoded angles — measures Bloch sphere coverage
  - `curvature_gradient`: Rate of change of trajectory curvature — second-order geometric signal
  - `segment_density`: Local density of already-placed boundaries
- **8 qubits** for 8 features, with multi-observable readout
- **Purpose**: Demonstrate quantum advantage through parity-based readout that captures entanglement correlations classical networks cannot efficiently access

### Version C: Next-Gen Q-RNN (6q)

The Q-RLSTC 2.0 flagship. Bundles all advanced features:

- **Shadow qubit** (qubit 0): carries quantum state across time steps, creating recurrent memory
- **EQC ansatz**: SO(2)-equivariant gates that respect rotational symmetry of GPS data
- **SAC agent**: replaces ε-greedy with entropy-maximising soft policy
- **DROP action**: actively filters GPS noise/anomalies
- **Adaptive shots**: 32-shot burst for confident decisions, full 512 only when uncertain
- **m-SPSA**: momentum-averaged gradients for smoother convergence through shot noise

### Version D: VLDB Aligned (5q)

Strict 1:1 mapping of the VLDB 2024 paper's MDP to a VQC function approximator:

**Baseline (paper-exact):**
- **State vector**: `s_t = (OD_s, OD_n, OD_b, L_b, L_f)` — Equation (19) of the paper
- **Actions**: Binary `{EXTEND, CUT}` — the paper's exact action space
- **Reward**: `r_t = OD(s_t) - OD(s_{t+1})` — maximising total reward = minimising terminal OD
- **OD_b**: TRACLUS expert baseline (paper ablation confirms it improves results)
- **DQN machinery**: replay buffer, target network sync, ε-greedy — all classical
- **Quantum substitution**: Replace the paper's 2-layer FFN (5→64→2 ≈ 514 params) with a VQC producing `Q(s, EXTEND)` and `Q(s, CUT)` via Z-expectations
- **30 parameters** (5q × 3 layers × 2 rotations) vs 514 classical — an empirical observation, not a proof of advantage

**Extension (paper Section 5.10):**
- **Q-SKIP**: opt-in ternary action `{EXTEND, CUT, SKIP(S)}` that fast-forwards S points. Requires defining `s_{t+1}` after skip and reward `OD(s_i) - OD(s_{i+S+1})`. Reduces circuit evaluations per trajectory.

**Research Variants (labeled, NOT baseline):**
- Hilbert curve spatial anchors replacing L_b/L_f
- Target network self-play replacing TRACLUS OD_b

---

## 4. SPSA Optimizer Design

### Why Not Backpropagation?

Quantum circuits don't have analytic gradients accessible through the measurement process. Options for gradient estimation:

| Method | Circuit Evaluations per Step | Accuracy | Practicality |
|---|---|---|---|
| **Parameter Shift Rule** | 2 × n_params (= 40 for 20 params) | Exact | Expensive on hardware |
| **SPSA** | **2** (regardless of n_params) | Estimated | ✅ Practical |
| **Finite Difference** | 2 × n_params | Estimated | Expensive |

SPSA wins because it needs only **2 circuit evaluations per step** — critical when each evaluation requires running a quantum circuit with 512 shots.

### SPSA Hyperparameters

```python
a_k = a / (A + k + 1)^α    # Learning rate schedule
c_k = c / (k + 1)^γ        # Perturbation schedule

# Defaults:
A = 20      # Stability constant (~10-20% of expected iterations)
a = 0.12    # Initial learning rate
c = 0.10    # Initial perturbation (larger than theory suggests for shot noise)
α = 0.602   # Theory optimal: 1.0, practice: ~0.6
γ = 0.101   # Theory optimal: 1/6, practice: ~0.1
```

### Momentum-SPSA (Version C)

```
g̃_k = β · g̃_{k-1} + (1 − β) · g_k
```

With β = 0.9, each gradient estimate is an exponential moving average of the last ~10 raw estimates. This smooths out the inherently noisy gradients from quantum measurement statistics.

---

## 5. Q-RLSTC VQ-DQN vs dina-quantum VQ-DQN

These two systems use fundamentally different architectures despite both being called "VQ-DQN":

| Design Choice | **Q-RLSTC** | **dina-quantum** |
|---|---|---|
| **Framework** | Pure Qiskit circuits, manual execution | PyTorch + `qiskit_machine_learning` + `TorchConnector` |
| **State Encoding** | `arctan` angle encoding → RY gates | `π/x` angle encoding → RX gates (via `AngleStateEncoder`) |
| **Feature Map** | None (direct encoding) | **ZZFeatureMap** (entangling feature map) |
| **Ansatz** | Custom HEA: RY-RZ + linear CNOT | `RealAmplitudes` (TwoLocal) **or BQN** (Bayesian QNN) |
| **Entanglement** | Linear CNOT chain | Circular CNOT ring (RealAmplitudes default) |
| **Trainable Gates** | RY, RZ (2 per qubit per layer) | RY only (1 per qubit per layer — RealAmplitudes) |
| **Optimizer** | **SPSA** (gradient-free) | **Adam / SPSA** (via TorchConnector backprop) |
| **QNN Interface** | Manual counts → expectations | `SamplerQNN` → `TorchConnector` (autograd) |
| **Output** | Pauli-Z expectations ⟨Z_i⟩ | Full measurement distribution → truncation to n_actions |
| **Ancilla Qubits** | None (Version A/B) or 1 shadow (C) | Optional BQN ancilla register |
| **Action Space** | 2–3 actions (segmentation) | Variable (database index selection) |
| **Domain** | Trajectory segmentation | Database index tuning (Q-DINA) |
| **Gradient Method** | 2-eval SPSA (hardware-efficient) | PyTorch autograd via TorchConnector (simulation-only) |

### Key Architectural Differences

**1. Encoding Philosophy:**
- Q-RLSTC: `θ = 2·arctan(x)` — smooth, handles unbounded GPS features
- dina-quantum: `θ = π/x` — maps (0, ∞) → (0, π], but **singular at x=0** (guarded with `torch.where`)

**2. Feature Map:**
- Q-RLSTC: **No separate feature map** — angle encoding directly serves as data embedding
- dina-quantum: Uses **ZZFeatureMap** (entangling feature map) to create feature interactions *before* the variational layer. This adds circuit depth but enables richer input representations

**3. Ansatz Architecture:**
- Q-RLSTC (HEA): RY-RZ rotation pairs + linear CNOT — minimal depth, NISQ-friendly
- dina-quantum (TwoLocal): `RealAmplitudes` with only RY rotations + circular CNOT — even shallower but less expressive
- dina-quantum (BQN): Bayesian Quantum Neural Network with ancilla-controlled U-blocks — much deeper, more expressive, but requires more qubits

**4. Measurement Strategy:**
- Q-RLSTC: Reads individual qubit expectations (⟨Z₀⟩, ⟨Z₁⟩) or parity correlations (⟨Z_aZ_b⟩)
- dina-quantum: Reads **full sampling distribution** (2^n bitstring probabilities), then truncates to action dimension — uses more information but requires more shots for convergence

**5. Gradient Pipeline:**
- Q-RLSTC: **Pure SPSA** — 2 circuit evaluations, no backprop chain, works on real hardware
- dina-quantum: `TorchConnector` wraps the quantum circuit in PyTorch autograd — enables Adam optimiser and standard deep learning training loops, but **only works in simulation** (requires statevector access for gradient computation)

---

## 6. Reward Function Design

### Reward Components (All Versions)

| Signal | Weight | Trigger | Purpose |
|---|---|---|---|
| Boundary sharpness | +0.5 × angle/π | On CUT | Encourage cuts at direction changes |
| Segment penalty | −0.1 | On CUT | Prevent over-segmentation |
| Variance delta | ±0.1 × Δvariance | Every step | Local clustering quality signal |
| Separability bonus | +0.3 × inter-centroid dist | On CUT (≥2 segs) | Encourage distinct clusters |
| Degeneracy penalty | −2.0 | Episode end, ≤1 segment | Prevent single-cluster collapse |
| Empty cluster penalty | −1.0 | Episode end, <2 effective segs | Prevent trivial segmentations |
| DROP anomaly bonus | +0.2 | On DROP (if actually anomalous) | Reward good noise filtering |
| DROP unnecessary penalty | −0.1 | On DROP (if not anomalous) | Discourage over-dropping |
| DROP micro-penalty | −0.05 | On DROP (always) | Baseline cost for data removal |
| SKIP reward (linear) | +0.05 × S | On SKIP (low variance segment) | Reward efficient fast-forward |
| SKIP penalty (non-linear) | −0.05 | On SKIP (high variance segment) | Discourage skipping over features |

### Anti-Gaming Constraints

- **Minimum segment length** = 3 points (CUT disallowed if segment too short)
- **Maximum segments** = 50 (episode terminates early)
- **CUT forced to EXTEND** if below minimum length

---

## 7. Measurement Parameters Summary

| Parameter | Version A | Version B | Version C | Version D |
|---|---|---|---|---|
| **Training shots** | 512 | 512 | **32 burst** → 512 fallback | 512 |
| **Eval shots** | 4096 | 4096 | 4096 | 4096 |
| **Confidence threshold** | N/A | N/A | 0.3 (margin for adaptive shots) | N/A |
| **ε start** | 1.0 | 1.0 | N/A (SAC uses entropy) | 1.0 |
| **ε min** | 0.1 | 0.1 | N/A | 0.1 |
| **ε decay** | 0.995 | 0.995 | N/A | 0.995 |
| **Entropy α** | N/A | N/A | 0.2 | N/A |
| **SPSA momentum β** | 0 (disabled) | 0 (disabled) | 0.9 | 0 (disabled) |
| **Discount γ** | 0.90 | 0.90 | 0.90 | 0.90 |
| **Batch size** | 32 | 32 | 32 | 32 |
| **Target update freq** | 10 episodes | 10 episodes | 10 episodes | 10 episodes |
| **Skip distance S** | N/A | N/A | N/A | **5** |
| **Variational layers** | 2 | 2 | 2 | **3** |
| **Total parameters** | 20 | 32 | ~24 | **30** |
