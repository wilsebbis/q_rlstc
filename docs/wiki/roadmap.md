# Research Directions — Future Work


---

> These are grounded extensions of the current Q-RLSTC research prototype. Each is supported by existing literature and feasible on current or near-term hardware.

---

## Near-Term Extensions

### 1. Hybrid Quantum Actor-Critic (Q-SAC)

**Current limitation:** Double DQN with ε-greedy exploration is rigid. In a noisy quantum landscape, discrete exploration can cause the agent to bounce out of promising regions.

**Extension:** Upgrade to a hybrid Quantum Soft Actor-Critic, where the VQ-DQN serves as the quantum actor outputting a continuous action probability, paired with a lightweight classical critic for value estimation. Entropy maximisation replaces ε-greedy heuristics and may prevent degenerate policy collapse.

**Feasibility:** Hybrid quantum-classical actor-critic methods have been demonstrated in prior work (Jerbi et al., 2021; Lockwood & Si, 2020). The VQ-DQN circuit requires no modification — only the training objective changes.

### 2. Adaptive Shot Allocation

**Current limitation:** Fixed shot counts (512–1024) for every single decision point waste QPU budget when the policy is already confident.

**Extension:** Dynamic measurement allocation based on Q-margin uncertainty:
- If Q-margin is wide (agent confident): use 32 shots and act immediately
- If Q-margin is narrow (ambiguous): trigger a deep read (512+ shots) to resolve noise

**Impact:** Could reduce total quantum evaluation overhead by 50–70% while preserving policy accuracy. This is a practical engineering improvement directly applicable to cloud QPU cost optimisation.

### 3. Symmetry-Aware Circuit Architectures (EQC)

**Current limitation:** The hardware-efficient ansatz treats spatial coordinates as abstract numbers. The circuit wastes training cycles re-learning rotational invariances that could be baked in.

**Extension:** Replace HEA with equivariant quantum circuits that commute with 2D spatial transformation groups (SO(2) rotations). This shrinks the hypothesis space and may accelerate convergence.

**Feasibility:** Equivariant quantum neural networks are an active research area (Nguyen et al., 2022; Larocca et al., 2022) with demonstrated benefits for geometric learning tasks.

### 4. Expanded Action Space: Active Denoising (DROP Action)

**Current limitation:** The binary {EXTEND, CUT} action space forces the agent to include noisy GPS points (urban canyon effects, signal bounces) in segments, skewing cluster centroids.

**Extension:** Expand to a 3-action MDP: {EXTEND, CUT, DROP}. The DROP action discards anomalous points while bridging valid coordinates, evolving Q-RLSTC into a simultaneous trajectory filter and segmenter.

**Anti-gaming:** Each DROP incurs a micro-penalty to prevent degenerate policies that discard everything.

### 5. Hybrid Quantum-Classical Recurrent Policies

**Current limitation:** The current architecture makes decisions based on a single-step observation. Trajectories are deeply sequential — a gradual slowdown over 10 points implies an upcoming stop, which a snapshot may miss.

**Extension:** Incorporate latent state propagation through hybrid quantum-classical recurrent architectures, where classical memory augments single-step quantum feature maps. This could provide temporal context without heavy LSTM layers that would destroy parameter efficiency.

**Note:** This does not require persistent quantum coherence across timesteps — the recurrence operates at the classical-quantum interface, with quantum circuits processing augmented observations at each step.

### 6. Distributed Training via Compact Parameter Footprint

**Current limitation:** Centralising trajectory data for training is a privacy and bandwidth concern.

**Extension:** The compact parameter footprint of variational quantum policies (34 parameters = 136 bytes at float32) may enable efficient federated learning. Local devices simulate the 5-qubit circuit, compute gradient updates from private data, and transmit only the parameter delta — raw GPS data stays on-device.

**Scope:** This is a parameter-efficiency argument, not a quantum-security claim. The value proposition is communication bandwidth, not cryptographic protection.

---

## Items Removed from Active Roadmap

The following items from earlier versions of this document have been removed or archived because they are either hardware-infeasible on current NISQ devices, insufficiently grounded in experimental evidence, or tangential to the core contribution:

- **Quantum coresets / variational state preparation** — requires qRAM (does not exist); exponential speedup claims without formal proof are inappropriate
- **QAOA-based trajectory alignment** — interesting theoretically but unrelated to the segmentation policy contribution; would dilute focus
- **Hamiltonian continuous-time policies** — speculative quantum control theory; no connection to current experiments
- **Entangled fleet synchronisation** — requires distributed entanglement (hardware-infeasible)
- **Differentiable quantum architecture search** — no experiments; too speculative for a results paper

---

← [Architecture](architecture.md) | [Experimental Design](experimental_design.md) →
