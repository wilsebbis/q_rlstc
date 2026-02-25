# Related Work and Evolution of Scope

## Scope Narrowing Overview

```
┌─────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────────┐
│  SURVEY (Short Paper)│   │ CONSTRAINTS (Feasibility)│   │ THESIS (Controlled)     │
│                     │   │                          │   │                         │
│ 5 quantum levers:   │──▶│ NISQ realism:            │──▶│ Policy network ONLY     │
│ • Clustering init   │   │ • Init/distance need     │   │                         │
│ • Distance estim.   │   │   exponential encoding   │   │ VQC replaces MLP in DQN │
│ • Policy network    │   │ • Depth/qubit limits     │   │ Everything else stays   │
│ • Exploration       │   │                          │   │ classical               │
│ • Optimization      │   │ Attribution:             │   │                         │
│                     │   │ • Multi-component swap   │   │ Clean attribution:      │
│                     │   │   → entangled variables   │   │ • Same environment      │
│                     │   │ • Speed claims dominated │   │ • Same reward shaping   │
│                     │   │   by classical I/O       │   │ • Same SPSA optimizer   │
│                     │   │                          │   │ • Same evaluation       │
│                     │   │ Controlled comparison:   │   │                         │
│                     │   │ • Need single variable   │   │ RQ: Does VQC achieve    │
│                     │   │ • Need matched baselines │   │ lower ValCR under       │
│                     │   │ • Need multi-seed        │   │ matched SPSA training?  │
└─────────────────────┘   └──────────────────────────┘   └─────────────────────────┘
```

---

## 1. Original Hypothesis Space

The Q-RLSTC framework was initially proposed as a broad hybrid quantum-classical system for sub-trajectory clustering [Q-RLSTC Short Paper], surveying multiple candidate quantum subcomponents:

- **Quantum clustering initialization** — using quantum sampling (e.g., Grover-based) for initial cluster centre selection.
- **Quantum distance estimation** — encoding trajectory segments into quantum states for inner-product-based similarity.
- **Variational quantum policy network** — replacing the classical DQN with a parameterized quantum circuit.
- **Quantum exploration** — leveraging measurement stochasticity as a natural exploration mechanism.
- **Quantum-native optimization** — using gradient-free methods (SPSA) suited to NISQ circuit training.

This survey established that quantum components could plausibly augment each stage of the RLSTC pipeline.

## 2. Feasibility and Confound Analysis

In practice, three constraints narrow the viable research scope:

### 2.1 NISQ Realism

Quantum clustering initialization (Grover search) and quantum distance estimation (amplitude encoding) require circuit depths and qubit counts beyond current NISQ hardware for the trajectory dimensions involved. The short paper acknowledges these barriers: *data encoding limits, NISQ noise, hybrid overhead, and decoding probabilistic outputs* [Q-RLSTC Short Paper, §Discussion].

### 2.2 Scientific Attribution

Replacing multiple components simultaneously entangles experimental variables. Any observed difference could not be attributed to a specific quantum lever. Wall-clock speedup claims would be dominated by classical bottlenecks (environment stepping, distance matrix computation, cluster maintenance) rather than the quantum subcomponent.

### 2.3 Controlled Comparison Requirements

A defensible empirical study requires: (a) a single substituted component, (b) identical training conditions for quantum and classical variants, and (c) multiple classical baselines spanning different parameter scales and optimizer families.

## 3. Refined Thesis Question

> Our earlier work introduced Q-RLSTC as a broad hybrid framework, surveying several candidate quantum subcomponents of RLSTC — including quantum clustering initialization, quantum distance estimation, and variational quantum reinforcement learning. In this thesis, we refine that vision into a controlled, NISQ-feasible research question by isolating the policy network as the sole quantum component. This enables attribution: any observed differences arise from the function approximator under identical environment dynamics, reward shaping, and evaluation protocol, rather than from confounded changes in distance computation or clustering.

**Research Question:**

> Under a fixed small-data budget and gradient-free optimization (SPSA), does a shallow 5-qubit VQC policy achieve lower validation Competitive Ratio than parameter-matched and larger classical MLP policies when used as the sole substituted component in an RLSTC DQN agent?

This formulation:

- **Isolates the policy network** as the sole quantum component
- **Enables attribution**: differences arise from the function approximator's properties, not confounded system changes
- **Is NISQ-feasible**: 5-qubit, 3-layer HEA (34 params, depth ~15, angle encoding)
- **Directly addresses the short paper's barriers**: we pick the subcomponent where data encoding limits are manageable and evaluation is clean

## 4. Relationship to Prior Work

| Aspect | Short Paper (Survey) | This Thesis (Controlled Study) |
|---|---|---|
| Scope | 5 quantum levers surveyed | 1 lever isolated (policy network) |
| Evaluation | Conceptual feasibility | Empirical with matched baselines |
| Models | Proposed VQ-DQN architecture | Implemented + 8 classical controls |
| Optimizer | SPSA discussed theoretically | SPSA shared across all SPSA models; Adam controls included |
| Data regime | Not specified | 30 trajectories, 2 epochs (controlled) |
| Seeds | Not applicable | 5 seeds, significance tests |
| Claim | Quantum components may help RLSTC | VQC is regime-competitive under SPSA (not "advantage") |

## 5. What This Structure Demonstrates

This narrowing is standard scientific practice — not scope retreat:

1. The short paper **establishes the hypothesis space** (what could be tried)
2. The feasibility analysis **eliminates infeasible and confounded options** (what can't be cleanly tested)
3. The thesis **produces controlled evidence for the cleanest remaining lever** (what works, with evidence)

An examiner reading this sequence sees: broad vision → principled narrowing → rigorous test. This is the mark of scientific maturity.
