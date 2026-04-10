# Quantum vs Classical RLSTC Experimental Test Bench

This module constitutes a rigorous, repeatable pipeline for comparing a Quantum VQ-DQN proxy against the legacy Classical RLSTC framework (Liang et al., 2024).

## Configuration Modes

The bench enforces two distinct evaluation methodologies, ensuring that empirical testing does not confuse standard "repository-level defaults" with the stricter constraints listed in the original academic paper.

- **`paper_baseline`**: Executes under the rigorous specifications indicated in the text: `tau = 0.1`, Trajectory Preprocessing enabled, `k = 10`, using `dIED` distances evaluated against Overall Distance (OD).
- **`repo_baseline`**: Executes under the more permissive default conditions bundled with the original repository scripts: `tau = 0.2`, Trajectory Preprocessing implicitly disabled.

## Pipeline Usage

Execute trials using the unifying `run_bench.py` runner script:

```bash
# Run Classical under Paper conditions
python run_bench.py --mode paper_baseline --backend classical

# Run Quantum under Repo conditions
python run_bench.py --mode repo_baseline --backend quantum
```

## Plotting Results

Export publication-ready comparison bar charts (measuring `OD` and `Runtime`) natively:

```bash
python plotting.py --mode paper_baseline
```

Plots are exported directly to `experiments/bench/plots/`.

## Scaffolding

- **Phase 3 (Parameter Sweeps):** `run_bench.py` scaffolding includes `--sweep-k` and `--sweep-qubits` hooks which currently drop into a NotImplemented placeholder stack.
- **Phase 4 (Diagnostic Fallbacks):** `diagnostics.py` contains placeholders for trajectory time-gap splitting protocols should the classical output degenerate.
