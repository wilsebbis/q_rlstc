# Experimental Setup & Reproducibility Protocol

## 1. Hardware & Software Environment

| Component | Specification |
|---|---|
| Machine | Apple MacBook Pro (M-series) |
| Python | 3.12.x |
| NumPy | 2.4.x |
| Qiskit | 1.x (statevector simulator) |
| OS | macOS |
| Git hash | Recorded per run (see JSON output `env.git_hash`) |

> All experiments use `_collect_env_metadata()` in `run_thesis_experiments.py` to record the exact Python, NumPy, and Qiskit versions at runtime into the JSON output.

## 2. Dataset

| Property | Value |
|---|---|
| Source | T-drive taxi dataset (Beijing, 2008–2009) |
| Preprocessing | `Tdrive_norm_traj` (normalized) |
| Cluster centers | `tdrive_clustercenter` (pre-computed) |
| Training size | 27 trajectories (90% of 30) |
| Validation size | 3 trajectories (10% of 30, fixed partition) |
| State dimensionality | 5 (IED, split_OD, min_dist, segment_length, step_index) |
| Action space | Binary: EXTEND (0) / CUT (1) |

The 90/10 split is deterministic per seed via `TrajectoryScheduler(validation_pct=0.1, seed=seed)`.

## 3. Protocol Constants (Shared Across All Models)

All agents share identical hyperparameters — the only variable is the policy network architecture.

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 32 | Standard DQN |
| `memory_size` | 5,000 | Fits in RAM, sufficient for 30-trajectory regime |
| `gamma` (discount) | 0.90 | Sub-episode horizons are ~10–50 steps |
| `huber_delta` | 1.0 | Standard smooth L1 loss |
| `epsilon_start` | 1.0 | Full exploration initially |
| `epsilon_min` | 0.1 | Retain 10% exploration floor |
| `epsilon_decay` | 0.99 | Per-episode decay |
| `target_update_freq` | 10 | Episodes between target network sync |
| `L_MIN` | 3 | Minimum segment length before CUT allowed |
| `CUT_PENALTY` | 0.12 | Per-cut reward penalty |
| `EXTEND_COST` | 0.01 | Per-extend small cost to break ties |
| `COMPLEXITY_LAMBDA` | 0.03 | End-of-episode cut-rate regularizer |

## 4. SPSA Optimizer Configuration

Applied identically to VQ-DQN and all SPSA classical controls.

| Parameter | Value | Source |
|---|---|---|
| a (learning rate scale) | 0.12 | Spall 1998 defaults |
| c (perturbation scale) | 0.08 | Spall 1998 defaults |
| A (stability constant) | 20 | ~10% of expected iterations |
| alpha (LR decay rate) | 0.602 | Spall 1998 theory |
| gamma (pert decay rate) | 0.101 | Spall 1998 theory |
| momentum | 0.9 | m-SPSA variant |
| gradient clip | 1.0 | Prevents exploding updates |

## 5. Models Under Test

| Model | Kind | Params | Architecture |
|---|---|---|---|
| VQ-DQN (5q×3L) | quantum | 34 | 5-qubit, 3-layer HEA, angle encoding |
| MLP-34 (SPSA) | classical-SPSA | 34 | [4]-hidden MLP (param-matched) |
| MLP-34 (Adam) | classical-Adam | 34 | [4]-hidden MLP (param-matched) |
| Control A (linear) | classical-SPSA | 12 | Linear (no hidden) |
| Control B (h=64) | classical-SPSA | 450 | [64]-hidden MLP |
| Control C (h=32×32) | classical-SPSA | 1,314 | [32,32]-hidden MLP |
| Control D (Adam linear) | classical-Adam | 12 | Linear (no hidden) |
| Control E (Adam h=64) | classical-Adam | 450 | [64]-hidden MLP |
| Control F (Adam h=32×32) | classical-Adam | 1,314 | [32,32]-hidden MLP |

## 6. Evaluation Protocol

### Training Definition

We define **one epoch** as a full pass over the scheduled training trajectories, each generating one episode per trajectory under ε-greedy exploration. With 27 training trajectories and 2 epochs, each model processes exactly 54 training episodes per seed.

### Compute-Budget Parity

All SPSA-trained models receive identical:
- **Environment steps**: same trajectories, same step counts
- **SPSA iterations**: one update per `batch_size` replay samples
- **Forward evaluations**: SPSA = 2 evaluations per gradient estimate (θ+δ and θ-δ)

VQC vs MLP forward-pass cost differs, but this is irrelevant since we claim no speedup.

### Validation

- **Frequency**: After every training epoch
- **Policy**: Greedy (ε = 0) — `agent.act(obs, greedy=True)`
- **Metric**: ValCR = OD / basesim, SSE, CUT%, segment count
- **Multi-seed**: 5 seeds (42, 123, 7, 99, 2025), report mean ± std

### Best-Epoch Selection

- **Criterion**: Lowest ValCR across epochs (argmin over epoch index)
- **Tie-breaking**: Earlier epoch wins (first occurrence of minimum)
- **Scope**: Per-seed; aggregated across seeds via mean ± std of per-seed bests

### Significance Testing

- Mann-Whitney U test (nonparametric, VQ-DQN vs each control)
- Cohen's d effect size with interpretation labels
- Bootstrap 95% confidence interval on mean difference (10,000 resamples)

## 7. Determinism & Reproducibility Guarantees

### Seeded RNGs

Each experiment seeds the following at run start:
- `np.random.seed(seed)` — NumPy global RNG
- `random.seed(seed)` — Python stdlib RNG
- `ReplayBuffer(seed=seed)` — replay sampling
- `TrajectoryScheduler(seed=seed)` — train/val split and epoch ordering

### Qiskit Determinism

Statevector simulation (shots=0) is fully deterministic — no sampling. Shot-based simulations (E3) use Qiskit's internal RNG seeded per circuit execution.

### Run Identification

Each JSON output uniquely identifies a run via:
- `args.seed` / `args.seeds` — random seed(s)
- `args.amount` — dataset size
- `args.epochs` — training epochs
- `args.experiments` — experiment IDs
- `env.git_hash` — code version
- `env.timestamp` — wall-clock start time
- Per-model: `model`, `kind`, `noise`, `shots`, `params`

### What Constitutes a "Run"

A run is uniquely identified by the tuple: `(seed, dataset_amount, epochs, model_id, optimizer_kind, shots, noise_model, git_hash)`. Two runs with the same tuple produce identical results under statevector simulation.

## 8. Reproducing Results

```bash
# Clone and setup
git clone <repo>
cd q_rlstc
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Full multi-seed E1 experiment (~90 min)
python experiments/run_thesis_experiments.py \
    --experiments E1 \
    --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 \
    --output-dir results/thesis_multiseed

# Significance tests
python experiments/run_significance_test.py \
    results/thesis_multiseed/thesis_results_*.json

# Robustness sweeps (shots, noise)
python experiments/run_thesis_experiments.py \
    --experiments E2,E3 \
    --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 \
    --output-dir results/thesis_robustness

# Entanglement ablation
python experiments/run_thesis_experiments.py \
    --experiments AB1 \
    --amount 30 --epochs 2 \
    --seeds 42,123,7,99,2025 \
    --output-dir results/thesis_ablation
```

## 9. Metrics Glossary

| Metric | Formula | Interpretation |
|---|---|---|
| OD | mean(IED(segment, center)) | Average segment-to-center distance |
| CR (ValCR) | OD / basesim | Normalized quality (lower = better) |
| SSE | Σᵢ Σₛ∈Cᵢ IED(s, centerᵢ)² | Within-cluster compactness |
| CUT% | cuts / (cuts + extends) × 100 | Segmentation aggressiveness |
| Episodes-to-best | #episodes before best CR epoch | Sample efficiency (coarse) |
| Actions-to-best | #RL steps before best CR epoch | Sample efficiency (granular) |
| Q-margin | mean(Q_extend) - mean(Q_cut) | Policy preference direction |
