# Q-RLSTC 2.0 Roadmap — Advanced Iterations & Next-Generation Upgrades

While Parts 1 and 2 established Q-RLSTC as a mathematically sound, parameter-efficient proof-of-concept for NISQ hardware, the system currently operates under strict Markov assumptions, uses a basic value-based RL paradigm (DQN), and assumes relatively clean data.

To evolve Q-RLSTC from a controlled research environment into a robust, production-ready framework for real-world mobility data, six advanced architectural and algorithmic upgrades are planned.

---

## 1. Breaking the Markov Limit: Quantum Recurrent Memory (Q-RNN)

**Limitation:** The current classical feature extractor calculates a state representing an isolated snapshot in time. Trajectories are deeply sequential — a gradual slow-down over 10 consecutive points implies an upcoming stop, which a single snapshot might miss, leading the agent to falsely `CUT` during stop-and-go traffic.

**Iteration:** Introduce a **Shadow Qubit** or **Phase Encoding** to create a Quantum Recurrent Neural Network (Q-RNN) effect.

- **Shadow Qubit Mechanic:** Dedicate 1 of the 5–8 available qubits as a "memory register." Instead of measuring and resetting this qubit at every step, its quantum state is preserved, carried over, and entangled with the new spatial features encoded at the next step.
- **Result:** Quantum information from the past directly interferes with the present. This grants the agent a latent temporal memory — allowing it to track the sequential "rhythm" of a trajectory — without adding heavy classical RNN/LSTM layers that would destroy parameter efficiency.

---

## 2. Algorithmic Shift: Hybrid Quantum Soft Actor-Critic (Q-SAC)

**Limitation:** The current Double DQN relies on ε-greedy exploration — essentially flipping a coin to take a random action (0 or 1). In a noisy quantum landscape, this rigid, discrete exploration can cause the agent to bounce out of optimal local minima. Furthermore, DQNs are prone to value overestimation.

**Iteration:** Upgrade to a **Hybrid Quantum Soft Actor-Critic (Q-SAC)**.

- **Division of Labour:** A lightweight **Classical Critic** network absorbs the heavy lifting of estimating the continuous Value function V(s). Meanwhile, the VQ-DQN is repurposed as the **Quantum Actor**, outputting a continuous probability distribution π(a|s) (e.g., "85% confidence to CUT").
- **Entropy Maximisation:** Q-SAC adds an "entropy bonus" to the reward function. The agent is rewarded for keeping its options open (exploring smoothly) until it is absolutely sure. This entirely replaces the clunky ε-greedy heuristic and prevents the quantum agent from collapsing into degenerate policies.

---

## 3. Action Space Expansion: Active Denoising (The `DROP` Action)

**Limitation:** Real-world GPS data suffers from "urban canyon" effects — sudden, physically impossible jumps when signals bounce off skyscrapers. The current `[EXTEND, CUT]` binary forces the agent to include garbage points in a sub-trajectory, skewing K-Means centroids.

**Iteration:** Expand to a 3-Action MDP: **`[EXTEND, CUT, DROP]`**.

- **Mechanic:** If the agent detects a massive anomaly (e.g., instant 200 mph spike, impossible curvature gradient), it selects `DROP`. The point is entirely discarded, bridging the previous point to the next valid coordinate.
- **Anti-Gaming:** Each `DROP` incurs a micro-penalty. However, successfully bypassing a severe outlier prevents the massive variance penalty from integrating garbage data. Q-RLSTC evolves into a simultaneous **trajectory filter and segmenter**.

---

## 4. Hardware Efficiency: Adaptive Shot Allocation & m-SPSA

**Limitation:** Running 512–1024 measurement "shots" to average out state collapse for every single point is a massive bottleneck for QPU cloud APIs and slows SPSA optimisation.

**Iteration:** Implement **Adaptive Shot Scaling** and **Momentum-SPSA**.

- **Dynamic Shots:** Execute a rapid burst of 32 shots. If the policy margin is wide (e.g., P(CUT) > 0.9), the agent is confident and acts immediately. If ambiguous, trigger a deep read (512+ shots) to resolve quantum noise.
- **Momentum-SPSA (m-SPSA):** A classical momentum term (like Adam) is added to gradient-free SPSA. By tracking a moving average of past gradients, the system "blasts through" bad gradient estimations caused by shot noise.
- **Result:** Cuts total quantum compute time by up to 60% while turning early-training shot noise into a feature (organic exploration).

---

## 5. Geometric Intelligence: Equivariant Quantum Circuits (EQC)

**Limitation:** The current Hardware-Efficient Ansatz (HEA) treats spatial coordinates as abstract numbers. The circuit wastes thousands of training cycles "re-learning" that a sharp 90° left turn heading North deserves a `CUT`, just as a 90° left turn heading East does.

**Iteration:** Replace the HEA with an **Equivariant Quantum Neural Network (EQNN)**.

- **Mechanic:** Design the quantum entanglement topology (CNOT chains) and rotation gates to inherently commute with 2D spatial transformation groups (like SO(2) for rotations), achieving mathematical rotational invariance.
- **Result:** The circuit "bakes in" geometric rules. If the agent learns to cut a corner in one direction, its quantum state automatically knows how to execute the same cut in any direction. This drastically shrinks the hypothesis space, heavily accelerating convergence.

---

## 6. The Killer App: Privacy-Preserving Federated Edge Learning (Q-FL)

**Limitation:** Centralising millions of daily commuting routes to train a global RL agent is a severe privacy vulnerability and a massive bandwidth bottleneck.

**Iteration:** Leverage Q-RLSTC's microscopic parameter size for **Federated Edge Learning**.

- **The Math:** A classical deep RL model requires transmitting millions of parameters (megabytes). Q-RLSTC Version A has exactly **20 parameters**. At 32-bit floats, the entire "brain" is exactly **80 bytes** — smaller than a standard SMS.
- **Pipeline:**
  1. Smartphones / delivery vehicles run Layer 1 (Feature Extraction) locally.
  2. They simulate the 5-qubit VQ-DQN locally (a fraction of a millisecond on a modern CPU).
  3. They compute local SPSA gradient updates from private driving data.
  4. They transmit *only* the 80-byte gradient update to a central cloud server.
- **Benefit:** Raw GPS data never leaves the user's device, guaranteeing absolute privacy. The quantum parameter bottleneck acts as a cryptographic one-way function — it is mathematically intractable to reverse-engineer a user's physical trajectory from 20 aggregated quantum rotation angles.

---

← [Noise & Hardware](noise_and_hardware.md) | [Architecture](architecture.md) →
