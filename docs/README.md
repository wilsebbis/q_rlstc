# Q-RLSTC: Complete Technical Reference

A comprehensive technical document explaining the Q-RLSTC (Quantum-Enhanced Reinforcement Learning for Sub-Trajectory Clustering) implementation. This document enables understanding the system deeply enough to reimplement it from first principles.

---

## Document Structure

This technical reference is organized into seven parts across two files:

### [Part 1: System Architecture & Philosophy](./technical_reference_part1.md#part-1-system-architecture--philosophy)
- What Q-RLSTC is and why it exists
- The sub-trajectory clustering problem domain
- Why reinforcement learning fits this problem
- Why quantum computation is applied
- System overview and design philosophy

### [Part 2: Quantum Encoding Dichotomy](./technical_reference_part1.md#part-2-quantum-encoding-dichotomy)
- The two encoding strategies used in Q-RLSTC
- Angle encoding for VQ-DQN policy
- Amplitude encoding for swap test distance
- Why different encodings for different purposes

### [Part 3: VQ-DQN Circuit Design](./technical_reference_part1.md#part-3-vq-dqn-circuit-design)
- Circuit architecture overview
- Input layer: angle encoding
- Variational layers: Hardware-Efficient Ansatz
- Data re-uploading technique
- Parameter count analysis
- Q-value extraction via Z-expectations
- Circuit depth analysis

### [Part 4: Distance Estimation & Clustering](./technical_reference_part2.md#part-4-distance-estimation--clustering)
- The distance computation challenge
- Integrated Euclidean Distance (IED)
- Incremental distance computation
- Quantum distance via swap test
- Hybrid k-means clustering
- Cluster center computation

### [Part 5: RL Training & Optimization](./technical_reference_part2.md#part-5-rl-training--optimization)
- MDP structure (states, actions, rewards)
- SPSA vs parameter-shift gradients
- SPSA algorithm and hyperparameters
- Experience replay
- Double DQN for stability
- Target network updates
- Epsilon-greedy exploration
- Outer training loop

### [Part 6: Comparative Analysis](./technical_reference_part2.md#part-6-comparative-analysis)
- System comparison matrix
- RLSTCcode (classical baseline)
- TheFinalQRLSTC (direct predecessor)
- qDINA (different domain, similar quantum RL)
- qmeans (pure quantum distance)

### [Part 7: Engineering Tradeoffs & Expected Results](./technical_reference_part2.md#part-7-engineering-tradeoffs--expected-results)
- Circuit depth vs noise resilience
- Parameter count vs trainability
- Shots vs statistical precision
- Angle vs amplitude encoding
- Classical vs quantum training
- Expected results and benchmarks
- Known limitations
- Future directions

---

## Quick Reference: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Qubits | 5 | One per state feature |
| Variational layers | 2 | NISQ-friendly depth |
| Rotation sequence | RY-RZ-RY | Full single-qubit expressivity |
| Entanglement | Linear CNOT | Lower noise than ring/full |
| Encoding (VQ-DQN) | Angle | Fast, direct gradients |
| Encoding (distance) | Amplitude | Required for swap test |
| Optimizer | Adam or SPSA | Adam for precision, SPSA for scale |
| Shots (train) | 512 | Balance speed vs precision |
| Shots (eval) | 1024 | Higher precision for metrics |
| Parameters | ~30 | 93% fewer than classical |

---

## Quick Reference: Comparison to Related Systems

| System | Domain | Quantum Component | Key Difference |
|--------|--------|-------------------|----------------|
| **RLSTCcode** | Trajectories | None (classical) | MLP instead of VQC |
| **TheFinalQRLSTC** | Trajectories | VQ-DQN | Less modular, fewer noise models |
| **qDINA** | Databases | BQN/TwoLocal | SPSA-only, larger action space |
| **qmeans** | Clustering | Swap test | No RL, unsupervised only |

---

## How to Read This Document

**For implementation**:
1. Read Parts 1-3 to understand the quantum circuit design
2. Study Part 5 for the training algorithm
3. Reference Part 4 for distance computations

**For research**:
1. Read Part 6 for comparative context
2. Study Part 7 for tradeoff analysis
3. Focus on encoding dichotomy (Part 2) for quantum design decisions

**For evaluation**:
1. Skip to Part 7 for expected results and limitations
2. Review Part 6 for benchmark comparisons
3. Check tradeoffs section for hardware considerations

---

## Files in docs/

- `technical_reference_part1.md` — Parts 1-3 (Architecture, Encoding, Circuit)
- `technical_reference_part2.md` — Parts 4-7 (Distance, Training, Comparison, Tradeoffs)
- `README.md` — This index document
