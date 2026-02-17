# Noise & Hardware Simulation

[← Back to README](../../README.md) · [Comparison](comparison.md) · **Noise & Hardware** · [Experimental Design →](experimental_design.md)

---

Q-RLSTC includes a full noise simulation stack with no classical RLSTC equivalent. All components are in the [`quantum/`](../../q_rlstc/quantum/) package.

## Backend Factory

Defined in [`backends.py`](../../q_rlstc/quantum/backends.py):

```python
def get_backend(mode: str, noise_model_name: str = None):
    if mode == "ideal":
        return AerSimulator()
    elif mode == "noisy_sim":
        noise_model = get_noise_model(noise_model_name)
        return AerSimulator(noise_model=noise_model)
```

## Available Noise Profiles

| Name | 1Q Error | 2Q Error | Readout | T₁ | T₂ | Target |
|---|---|---|---|---|---|---|
| **Ideal** | 0 | 0 | 0 | ∞ | ∞ | Algorithmic debugging |
| **Simple** | 0.1% | 1.0% | 2.0% | — | — | Quick noise impact checks |
| **Eagle** | 0.05% | 0.8% | — | 300μs | 150μs | IBM Eagle 127-qubit emulation |
| **Heron** | 0.02% | 0.2% | — | 400μs | 200μs | IBM Heron next-gen emulation |

### Estimated Circuit Fidelity

For the 5-qubit VQ-DQN (depth ~11, 8 CNOTs):

| Backend | Estimated Fidelity |
|---|---|
| Ideal | 100% |
| Simple | ~90% |
| Eagle | ~85% |
| Heron | ~95% |

## Readout Error Mitigation

Defined in [`mitigation.py`](../../q_rlstc/quantum/mitigation.py):

```python
class ReadoutMitigator:
    def calibrate(self, backend, shots=8192):
        """Build calibration matrix by measuring all basis states."""
        # Run 2ⁿ calibration circuits
        # Build M[i,j] = P(measure i | prepared j)
        self.calibration_matrix = M

    def apply(self, counts):
        """Correct raw counts via matrix pseudo-inverse."""
        # Solve linear system → corrected probabilities
        # Clip negative values, re-normalise
        return corrected_counts
```

### Mitigation Pipeline

```
Raw counts → Calibration matrix (pseudo-inverse) → Clip negatives → Re-normalise → Corrected counts
```

Falls back to pass-through if no calibration has been performed.

## Configuration

```python
@dataclass
class NoiseConfig:
    use_noise: bool = False
    noise_model: str = "depolarizing"     # or "thermal", "ibm_fake"
    use_mitigation: bool = True
    calibration_shots: int = 8192
```

## Noise Impact on Training

| Effect | Mechanism | Mitigation |
|---|---|---|
| Q-value variance | Shot noise in expectations | More shots (512→1024) |
| Gradient noise | SPSA gradient estimate corrupted | Decaying perturbation `cₖ` |
| Readout bias | Systematic measurement errors | Calibration matrix |
| Decoherence | T₁/T₂ decay during circuit | Shallow depth (≤11 layers) |

---

**Next:** [Experimental Design →](experimental_design.md)
