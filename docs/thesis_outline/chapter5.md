# Chapter 5: Conclusions and Future Work

## 5.1 Summary of Contributions

### C1. Structural Degeneracy in Trajectory Clustering Evaluation

We identified and characterised a structural degeneracy in ValCR, the standard RLSTC evaluation metric: per-segment averaging combined with IED's length dependence creates a fragmentation attractor where segmentation rate alone drives the metric, independent of clustering quality. The D1 diagnostic confirms this empirically — a random policy sweep reduces ValCR from 9.21 to ~1.42 without any learning. We proposed budget-constrained evaluation (reporting ValCR at matched CUT thresholds) as a general mitigation framework applicable to all RL-based trajectory segmentation methods. This contribution is independent of the quantum component and directly relevant to the trajectory mining community.

### C2. Hybrid Quantum-Classical RL Framework

Q-RLSTC replaces the classical DQN policy with a 34-parameter VQ-DQN (5-qubit, 3-layer HEA with affine readout) while retaining all other components as identical classical implementations. The modular controlled substitution design enables clean scientific attribution: observed performance differences arise from the function approximator, not confounded pipeline changes.

### C3. Regime-Specific VQC Competitiveness

Under matched SPSA training (same optimiser, same small-data budget, same pipeline), the 34-parameter VQ-DQN was evaluated against 4 SPSA-matched classical controls (12–1,314 parameters) and contextualised by 4 Adam-trained controls representing the classical ceiling. Multi-seed validation (5 seeds) with Mann-Whitney U, Cohen's d, and bootstrap 95% CI provides statistically grounded comparison. The 15–38× parameter compression is the core value proposition.

Causal attribution claims are restricted to the SPSA-matched cohort. Adam baselines contextualise whether SPSA fundamentally limits classical networks or whether the VQC's structure provides genuine advantages in the tested regime.

### C4. NISQ-Feasible Circuit Design (Simulation-Based)

The VQ-DQN circuit operates within NISQ constraints: 5 qubits, ~15 depth, angle encoding, no mid-circuit measurement, constant-cost SPSA. Robustness is characterised via shot sweeps (128–2,048) and hardware-inspired noise simulations (Eagle, Heron profiles). These experiments characterise expected degradation; they are not a hardware demonstration. The entanglement ablation (AB1) probes whether observed parameter efficiency correlates with quantum correlations or with generic architectural constraints of the bounded hypothesis class.

### C5. Diagnostic Identification of Metric Denominator Pathology

During multi-seed evaluation, we identified a critical "denominator bug" where calculating ValCR against a globally static baseline OD mathematically decoupled the metric from the active validation fold. This flaw artificially inflated cross-seed variance and produced quantized CR plateaus, masquerading as policy instability. By correcting the evaluation pipeline to compute a fold-specific dynamic baseline under a fixed "always-extend" policy, we restored the metric's validity as a true competitive ratio and eliminated the artificial variance. This autopsy demonstrates the necessity of rigorous metric validation in trajectory RL.

---

## 5.2 Limitations

1. **Training speed.** VQ-DQN training is ~100× slower than classical controls due to circuit simulation. The value proposition is parameter footprint (34 params = 136 bytes), not wall-clock time.
2. **No broad classical superiority.** Under Adam with sufficient data, classical networks achieve strong results. Competitiveness is specific to the SPSA + small-data regime.
3. **Simulation only.** No physical quantum hardware validation.
4. **Single dataset.** T-Drive taxi GPS. Generalisation to maritime, pedestrian, or multi-modal trajectory domains is untested.
5. **Attribution uncertainty.** The VQC's competitiveness may reflect favourable SPSA landscape properties of the ansatz (bounded outputs acting as implicit regularisation) rather than quantum-mechanical effects. Disentangling these requires further investigation.

---

## 5.3 Future Work

### 5.3.1 Constrained RL Formulation

The current approach mitigates ValCR degeneracy via reward shaping and post-hoc budgeted evaluation. A Lagrangian optimisation with explicit CUT budget in the training objective would internalise budgeted evaluation directly into the learning process. This is the highest-priority extension.

### 5.3.2 Real Quantum Hardware Validation

Executing the training loop on physical IBM Quantum hardware (Eagle, Heron) with readout error mitigation (TREX, M3) is the most critical validation step. The interaction between REM post-processing and SPSA gradient estimation under physical gate drift is a specific open question.

### 5.3.3 Hybrid Quantum Actor-Critic

Extending VQ-DQN to actor-critic (Q-SAC) may improve stability and sample efficiency. Version C provides a theoretical foundation with entropy-regularised policies and shadow-qubit recurrent memory. Empirical evaluation under the same controlled conditions as Version D is a natural next step.

### 5.3.4 Adaptive Shot Allocation

Dynamic measurement allocation based on Q-margin uncertainty could reduce evaluation overhead by an estimated 50–70% — fewer shots when policy preference is clear, more when uncertain.

### 5.3.5 Mechanism Analysis

Beyond the AB1 entanglement ablation, deeper analysis — partial entanglement topologies, entanglement entropy tracking during training, Fisher information comparisons — would clarify whether quantum correlations causally contribute to parameter efficiency or whether the efficiency arises from generic properties of the bounded, structured hypothesis class.

### 5.3.6 Multi-Dataset Validation

Extending to T-Drive, GeoLife, Porto taxi, and Foursquare check-ins would test whether regime-specific competitiveness generalises across spatial scales, sampling rates, and movement modalities.

### 5.3.7 Federated Edge Deployment

The compact parameter footprint (136 bytes) enables efficient federated learning. Edge devices would train on local GPS data and transmit byte-scale SPSA gradient updates rather than megabyte-scale classical parameters. Future work should quantify energy conservation and convergence under distributed training.