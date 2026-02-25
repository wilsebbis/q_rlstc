# Chapter 2: Background and Proposed Method

## 2.1 Trajectory Data and Segmentation

A trajectory is a finite sequence of timestamped spatial coordinates $T = \{(x_1, y_1, t_1), \ldots, (x_n, y_n, t_n)\}$. Sub-trajectory clustering partitions such sequences into segments and groups them into spatially coherent clusters. The segmentation–clustering coupling makes this a bilevel optimisation problem: optimal segmentation requires knowledge of cluster structure, and cluster structure emerges only after segmentation.

### Classical Approaches

**TRACLUS** decouples segmentation and clustering using the Minimum Description Length (MDL) principle — minimising the combined cost of compressed representation and model complexity. MDL is a natural regulariser but is static and domain-blind. **S2T-Clustering** segments via local density criteria; **SubCLUS** formulates segmentation as set-cover. All require manually tuned parameters that fail to adapt to dynamic environments.

### RLSTC (Lee et al., VLDB 2024)

RLSTC models segmentation as a sequential MDP: a DQN agent decides EXTEND or CUT at each trajectory point, guided by a reward derived from the Overall Distance (OD) metric — the mean Integrated Edit Distance (IED) between each segment and its nearest cluster centre. RLSTC consistently outperforms heuristic baselines.

---

## 2.2 Reinforcement Learning Foundations

### MDP Specification

| Component | Definition |
|---|---|
| **State** $s_t$ | 5-dimensional observation vector (§2.5) |
| **Action** $a_t$ | $\in \{\text{EXTEND}, \text{CUT}\}$ |
| **Reward** $r_t$ | Shaped reward based on clustering quality change (§2.7) |
| **Transition** | Deterministic: advance to next trajectory point |
| **Termination** | End of trajectory |

### Deep Q-Networks

A DQN approximates $Q^*(s, a)$ via $Q(s, a; \theta)$, trained with experience replay, a target network for bootstrap stability, and $\epsilon$-greedy exploration. This implementation uses **Double DQN** (online network selects actions, target network evaluates) to reduce overestimation bias.

### SPSA Optimisation

SPSA estimates gradients by simultaneously perturbing all parameters along a random direction vector. Each step requires exactly **2 circuit evaluations** ($\theta + c\Delta$ and $\theta - c\Delta$), regardless of parameter count — a constant $O(1)$ scaling critical for quantum circuits, where the Parameter-Shift Rule would require $2N$ evaluations.

| Parameter | Value | Source |
|---|---|---|
| $a$ (learning rate scale) | 0.12 | Spall 1998 |
| $c$ (perturbation scale) | 0.08 | Spall 1998 |
| $A$ (stability constant) | 20 | ~10% of expected iterations |
| $\alpha$ (LR decay) | 0.602 | Spall 1998 theory |
| $\gamma$ (perturbation decay) | 0.101 | Spall 1998 theory |
| Momentum | 0.9 | m-SPSA variant |
| Gradient clip | 1.0 | Prevents exploding updates |

---

## 2.3 Variational Quantum Computing

### Parameterised Quantum Circuits

A PQC maps classical data to quantum states, applies parameterised unitary transformations, and extracts classical outputs via measurement — a trainable function approximator analogous to a neural network.

### Angle Encoding

Each feature $x_i$ maps to a single-qubit rotation: $R_Y(2 \cdot \arctan(x_i))$, mapping $(-\infty, \infty) \rightarrow (-\pi, \pi)$ monotonically. This uses 1 qubit per feature with no exponential state preparation.

> **Saturation caveat.** $\arctan$ asymptotes at $\pm\pi/2$; features with large absolute values lose discriminative power. Dataset-dependent scaling should precede encoding if features span orders of magnitude.

### Hardware-Efficient Ansatz (HEA)

Alternating layers of single-qubit rotations ($R_Y$, $R_Z$) and linear CNOT entanglement. Linear connectivity (each qubit entangled with its neighbour) minimises circuit depth and two-qubit gate count.

### Data Re-uploading

Input features are re-encoded between variational layers (Pérez-Salinas et al., 2020), increasing expressivity without significant depth increase.

---

## 2.4 VQ-DQN Circuit Design

### Architecture

```
[Angle Encoding] → [HEA Layer 1] → [Re-upload] → [HEA Layer 2] → [Re-upload] → [HEA Layer 3] → [Measurement]
```

Each HEA layer: $R_Y(\theta) \rightarrow R_Z(\theta)$ per qubit (10 params) + linear CNOT chain (4 gates).

```
Qubit 0: ─RY─RZ─●──────────
                 │
Qubit 1: ─RY─RZ─X──●───────
                    │
Qubit 2: ─RY─RZ────X──●────
                       │
Qubit 3: ─RY─RZ───────X──●─
                          │
Qubit 4: ─RY─RZ──────────X─
```

### Parameter Count

| Component | Count |
|---|---|
| Variational rotations | 10 × 3 layers = **30** |
| Affine readout (scale + bias per action) | **4** |
| **Total trainable** | **34** |

### Q-Value Extraction

```
Q(EXTEND) = ⟨Z₀⟩ × scale₀ + bias₀     # ⟨Z₀⟩ ∈ [-1, 1]
Q(CUT)    = ⟨Z₁⟩ × scale₁ + bias₁
```

The bounded expectation values $[-1, +1]$ provide natural output regularisation: Q-values cannot diverge without the affine head amplifying them, providing control absent in unbounded MLP activations.

### Circuit Depth

Total depth ~15 (3 variational layers + 2 re-encodings + encoding + measurement). This depth keeps two-qubit gate count low and remains within typical coherence and gate-error constraints of current superconducting hardware.

---

## 2.5 State Representation (Version D — VLDB Aligned)

The primary thesis experiments use Version D, implementing a 1:1 mapping of the VLDB paper's state features:

| # | Feature | Description |
|---|---|---|
| 0 | $OD_s$ | Projected OD if we CUT here |
| 1 | $OD_n$ | Projected OD if we EXTEND |
| 2 | $OD_b$ | TRACLUS expert baseline cost |
| 3 | $L_b$ | Normalised backward segment length |
| 4 | $L_f$ | Normalised forward remaining length |

The TRACLUS baseline ($OD_b$) serves as a functional prior; its contribution is confirmed by ablation (A1).

---

## 2.6 Models Under Test

### Primary Controlled Comparison (SPSA-Matched Cohort)

All models share the same SPSA optimiser, isolating the function approximator as the **sole experimental variable**. Causal attribution claims are restricted to this cohort.

| Model | Architecture | Parameters |
|---|---|---|
| **VQ-DQN** | 5-qubit, 3-layer HEA | 34 |
| **MLP-34 (SPSA)** | 5→4→2 MLP | 34 (parameter-matched) |
| **Control A** | 5→2 linear | 12 (capacity lower bound) |
| **Control B** | 5→64→2 MLP | 514 (moderate capacity) |
| **Control C** | 5→32→32→2 MLP | 1,314 (SPSA ceiling) |

### Contextual Comparison (Adam Cohort)

These quantify the "SPSA handicap" — how much performance classical networks recover with standard backpropagation. **Not used for causal attribution** of quantum vs. classical function approximation.

| Model | Architecture | Parameters |
|---|---|---|
| **MLP-34 (Adam)** | 5→4→2 MLP | 34 |
| **Control D** | 5→2 linear | 12 |
| **Control E** | 5→64→2 MLP | 514 |
| **Control F** | 5→32→32→2 MLP | 1,314 (Adam ceiling) |

### Non-RL Baselines

| Baseline | Method | Purpose |
|---|---|---|
| **Uniform Window** | Cut every $W$ points | Budget-matched, no learning |
| **Random Policy** | Cut with probability $p$ (D1 sweep) | Lower bound — is learning needed? |

**Total: 9 RL models + 2 non-RL baselines = 11 experimental conditions.**

---

## 2.7 Reward Design and Stabilisation

The reward is constructed incrementally; each component addresses a specific failure mode identified during development:

| Component | Value | Addresses |
|---|---|---|
| OD improvement (base) | $\text{scale} \times (OD_{old} - OD_{new})$ | Primary learning signal |
| CUT penalty | $-0.12$ per cut | Over-segmentation attractor |
| EXTEND cost | $-0.01$ per extend | Idle extending |
| L_MIN = 3 | Hard action override | Micro-segments |
| Q-value clamping | ±10 | Value explosion (observed: Q → 78M) |
| TD target clamping | ±10 | Bootstrap instability |
| Complexity regulariser | $-0.03 \times \text{cut\_rate}$ at episode end | End-of-episode segmentation penalty |

Ablation E6 quantifies each term's contribution, distinguishing load-bearing components (necessary for convergence) from cosmetic ones.

---

## 2.8 Training Pipeline

All RL models share identical training hyperparameters — the only variable is the policy network architecture.

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Replay buffer | 5,000 |
| Discount $\gamma$ | 0.90 |
| Huber $\delta$ | 1.0 |
| $\epsilon$ start / min / decay | 1.0 / 0.1 / 0.99 per episode |
| Target update | Hard copy every 10 episodes |

We define the **small-data regime**: 30 trajectories (27 train / 3 validation), 2 epochs (54 training episodes per seed). This budget is intentionally constrained — the research question examines function approximation under data scarcity, the regime where parameter efficiency is most relevant.

---

## 2.9 Architecture Variants

Four VQC variants are defined for controlled study. **Version D is the primary thesis vehicle**; Versions A, B, and C are documented for completeness.

| Version | Qubits | State | Ansatz | Params | Agent | Purpose |
|---|---|---|---|---|---|---|
| **D** | 5 | 5D (VLDB exact) | HEA (3L) | 34 | ε-DQN | Strict VLDB reproduction — primary vehicle |
| A | 5 | 5D (matches RLSTC) | HEA (2L) | 20 | ε-DQN | Scientific control: isolate approximator |
| B | 8 | 8D (+angle, curvature, density) | HEA (2L) | 32 | ε-DQN | Richer features, more qubits |
| C | 6 | 5D + shadow memory | EQC (2L) | ~24 | SAC | Recurrent memory, entropy-regularised |