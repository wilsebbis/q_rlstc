# Debugging Guide

[← Back to README](../../README_2.md) · [Experimental Design](experimental_design.md) · **Debugging** · [API Reference →](api_reference.md)

---

## Common Failure Modes

### 1. Q-Values Stuck at Zero

**Symptoms:** Both `Q(EXTEND)` and `Q(CUT)` return ~0 for all states.

**Causes:**
- Variational parameters initialised symmetrically → no gradient signal
- Learning rate too low → SPSA steps too small to escape
- Shot count too low → expectations drowned in noise

**Fix:**
```python
# Randomise initial parameters
params = np.random.uniform(-np.pi, np.pi, size=n_params)

# Increase initial learning rate
spsa_config.a = 0.2  # Default 0.1

# Verify expectation values manually
circuit = builder.build(state, params)
counts = backend.run(circuit, shots=4096).result().get_counts()
print(compute_expectation_from_counts(counts, 4096, qubit=0, n_qubits=5))
```

### 2. Agent Always Extends (Never Cuts)

**Symptoms:** 0-1 segments per trajectory. F1 near zero.

**Causes:**
- Reward for EXTEND > reward for CUT (penalty too high)
- ε-greedy exploration exhausted before learning CUT value
- MIN_SEGMENT_LEN too large relative to trajectory

**Fix:**
- Reduce `SEGMENT_PENALTY` (e.g., 0.1 → 0.01)
- Slower epsilon decay: `epsilon_decay = 0.995`
- Check that boundary sharpness is being computed correctly

### 3. Agent Always Cuts (Over-Segmentation)

**Symptoms:** 50+ segments per trajectory. High OD.

**Causes:**
- Boundary sharpness reward dominates OD improvement
- No minimum segment length enforcement
- `MAX_SEGMENTS` not enforced

**Fix:**
- Verify `MIN_SEGMENT_LEN` constraint is triggering
- Increase `SEGMENT_PENALTY` or decrease `beta` (sharpness weight)
- Add logging: `if action == CUT: print(f"seg_len={seg_len}, sharpness={sharpness}")`

### 4. Training Loss Not Decreasing

**Symptoms:** TD loss oscillates without improvement.

**Causes:**
- SPSA perturbation too large → estimates are random
- Target network stale → Q-targets are wrong
- Replay buffer too small → overfitting recent experiences

**Fix:**
```python
# Smaller perturbation
spsa_config.c = 0.05

# More frequent target updates
rl_config.target_update_freq = 5

# Larger buffer
rl_config.memory_size = 20_000
```

### 5. Noise Crashes Training (Noisy Backend)

**Symptoms:** Works on ideal, diverges on Eagle/Heron.

**Causes:**
- Readout errors systematically bias Q-values
- Decoherence reduces circuit fidelity below usable threshold
- SPSA gradient estimates too noisy

**Fix:**
- Enable readout mitigation: `noise_config.use_mitigation = True`
- Increase shots: 512 → 2048
- Reduce circuit depth (fewer layers)
- Try a less aggressive noise model first (simple → Eagle)

## Diagnostic Functions

### Circuit Inspection

```python
from q_rlstc.quantum.vqdqn_circuit import VQDQNCircuitBuilder

builder = VQDQNCircuitBuilder(n_qubits=5, n_layers=2)
state = np.array([0.5, 0.3, -0.2, 0.1, 0.8])
params = np.random.uniform(-np.pi, np.pi, size=20)

circuit = builder.build(state, params)
print(circuit.draw())           # Text diagram
print(f"Depth: {circuit.depth()}")
print(f"Gates: {circuit.count_ops()}")
```

### Expectation Value Sanity Check

```python
# All-zero state should give known expectations
state = np.zeros(5)
params = np.zeros(20)
circuit = builder.build(state, params)
# With zero params and zero state: ⟨Z⟩ should be +1 for all qubits
```

### Replay Buffer Inspection

```python
buffer = replay_buffer
recent = buffer.buffer[-5:]
for exp in recent:
    print(f"s={exp.state}, a={exp.action}, r={exp.reward:.4f}, done={exp.done}")
```

## Extension Points

### Adding a New Feature (Version B)

1. Add computation in `StateFeatureExtractorB._compute`
2. Update `n_features` → increase qubit count in config
3. Update `VQDQNConfig.n_qubits` and `n_params`
4. Re-run circuit depth analysis

### Adding a New Noise Model

1. Define in `backends.py → get_noise_model()`
2. Add gate errors, thermal relaxation, readout errors
3. Test with `pytest tests/test_noise_models.py`

### Swapping the Optimizer

1. Implement `step(params, loss_fn)` interface
2. Replace `SPSAOptimizer` in `VQDQNAgent.__init__`
3. Ensure gradient estimation is compatible with shot-based evaluation

### Adding a New Distance Metric

1. Implement in `clustering/metrics.py`
2. Use in `MDPEnvironment._compute_reward_components()`
3. Update state features if the metric provides useful state signal

---

**Next:** [API Reference →](api_reference.md)
