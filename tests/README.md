# Tests for Q-RLSTC

## Running Tests

```bash
cd /Users/wilsebbis/Developer/q_rlstc
pip install -e ".[dev]"
pytest tests/ -v
```

## Test Files

- `test_angle_encoding.py` - Angle encoding functions
- `test_hea_depth.py` - HEA circuit structure
- `test_swaptest_distance_basic.py` - Swap test distance
- `test_kmeans_update.py` - Clustering centroid updates
- `test_training_smoke.py` - Training loop smoke tests
