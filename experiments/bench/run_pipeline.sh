#!/bin/bash
# Stop execution on any error
set -e

echo "========================================"
echo "    Q-RLSTC Test Bench Executer         "
echo "========================================"

echo "[1/4] Ensuring dependencies are installed..."
# Install standard numeric, quantum, and plotting dependencies
python3 -m pip install numpy matplotlib || true
python3 -m pip install qiskit qiskit-aer || true
python3 -m pip install tensorflow==1.15.0 || \
    echo "Note: TensorFlow 1.15 requires Python 3.6. Proceeding without it (Classical outputs will fallback)..."

echo "[2/4] Running Classical Paper Baseline..."
PYTHONPATH=. python3 experiments/bench/run_bench.py --mode paper_baseline --backend classical

echo "[3/4] Running Quantum Paper Baseline..."
PYTHONPATH=. python3 experiments/bench/run_bench.py --mode paper_baseline --backend quantum

echo "[4/4] Generating Comparison Plots..."
PYTHONPATH=. python3 experiments/bench/plotting.py --mode paper_baseline

echo "========================================"
echo " Pipeline Execution Complete!           "
echo " Check experiments/bench/plots/         "
echo "========================================"
