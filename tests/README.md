# Tests for Q-RLSTC

## Running Tests

```bash
cd /Users/wilsebbis/Developer/q_rlstc
pip install -e ".[dev]"
pytest tests/ -v
```

## Test Files

| File | What it tests |
|---|---|
| `test_angle_encoding.py` | Angle encoding functions and normalisation |
| `test_hea_depth.py` | HEA circuit structure and layer counts |
| `test_kmeans_update.py` | Clustering centroid updates |
| `test_metrics.py` | OD, F1, silhouette metric computations |
| `test_replay_buffer.py` | Experience replay sampling and capacity |
| `test_spsa.py` | SPSA optimizer convergence on toy loss |
| `test_training_smoke.py` | Training loop end-to-end smoke tests |

## Diagnostics (not pytest)

These are standalone scripts in `experiments/`:

| Script | Purpose |
|---|---|
| `diagnose_valcr.py` | Tests ValCR metric for degeneracy via random policy sweep |
| `run_rigorous_benchmark.py` | Full E1/E2/E3 benchmark (see [docs](../docs/wiki/rigorous_benchmark.md)) |
