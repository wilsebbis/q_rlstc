#!/bin/bash
set -e

echo "Starting Tier 1 Main Controlled Comparison Matrix..."

mkdir -p results/tier1_main_matrix/tdrive
mkdir -p results/tier1_main_matrix/geolife

for SEED in 1 2 3 4 5; do
  echo ">>> Running T-Drive Core Comparison (Seed $SEED) <<<"
  uv run python experiments/run_cross_comparison.py \
      --run all \
      --traj-path q_rlstc/data/Tdrive_norm_traj \
      --centers-path q_rlstc/data/tdrive_clustercenter \
      --amount 500 \
      --seed $SEED \
      --output-dir results/tier1_main_matrix/tdrive/seed_$SEED \
      --adaptive-shots

  echo ">>> Running GeoLife Core Comparison (Seed $SEED) <<<"
  uv run python experiments/run_cross_comparison.py \
      --run all \
      --traj-path q_rlstc/data/geolife_norm_traj \
      --centers-path q_rlstc/data/geolife_clustercenter \
      --amount 500 \
      --seed $SEED \
      --output-dir results/tier1_main_matrix/geolife/seed_$SEED \
      --adaptive-shots
done

echo "Matrix complete."
