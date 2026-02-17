#!/usr/bin/env python3
"""Compare Version A vs Version B results across dataset sizes.

Loads metrics from experiment output directories and produces
a comparison table of the 4 key metrics.

Usage:
    python experiments/compare_versions.py
    python experiments/compare_versions.py --size small
    python experiments/compare_versions.py --size medium --phase online
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np


def load_metrics(version: str, size: str, phase: str = "online") -> dict:
    """Load metrics.json for a specific experiment run."""
    path = Path("outputs") / f"version_{version.lower()}" / f"{size}_{phase}" / "metrics.json"
    
    if not path.exists():
        return None
    
    with open(path) as f:
        return json.load(f)


def format_comparison_table(metrics_a: dict, metrics_b: dict, size: str, phase: str):
    """Print formatted comparison table."""
    
    print(f"\n{'=' * 70}")
    print(f"  Version A vs B Comparison — {size}, {phase}")
    print(f"{'=' * 70}")
    
    rows = [
        ("Qubits", "n_qubits", "", int),
        ("Parameters", "n_params", "", int),
        ("Feature Dim", "feature_dim", "D", int),
        ("Circuit Depth", "circuit_depth", "", int),
        ("", None, None, None),  # separator
        ("ΔOD", "delta_od", "", lambda x: f"{x:.4f}"),
        ("Final OD", "final_od", "", lambda x: f"{x:.4f}"),
        ("Segmentation F1", "final_f1", "", lambda x: f"{x:.4f}"),
        ("Convergence Ep", "convergence_episode", "", int),
        ("Param Efficiency", "parameter_efficiency", "", lambda x: f"{x:.6f}"),
        ("", None, None, None),  # separator
        ("Avg Final Reward", "avg_final_reward", "", lambda x: f"{x:.4f}"),
        ("Runtime", "runtime_seconds", "s", lambda x: f"{x:.1f}"),
    ]
    
    # Header
    print(f"\n  {'Metric':<22} {'Version A':>14} {'Version B':>14} {'Δ (B-A)':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    
    for label, key, suffix, fmt in rows:
        if key is None:
            print()
            continue
        
        val_a = metrics_a.get(key, "N/A") if metrics_a else "N/A"
        val_b = metrics_b.get(key, "N/A") if metrics_b else "N/A"
        
        if val_a == "N/A" or val_b == "N/A":
            str_a = str(val_a)
            str_b = str(val_b)
            delta_str = "—"
        else:
            if callable(fmt):
                str_a = str(fmt(val_a))
                str_b = str(fmt(val_b))
            else:
                str_a = str(val_a)
                str_b = str(val_b)
            
            # Compute delta
            try:
                delta = float(val_b) - float(val_a)
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:.4f}"
            except (ValueError, TypeError):
                delta_str = "—"
        
        print(f"  {label:<22} {str_a:>14} {str_b:>14} {delta_str:>14}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare Version A vs B Results")
    parser.add_argument("--size", type=str, default=None,
                        help="Compare specific size (default: all available)")
    parser.add_argument("--phase", type=str, default="online",
                        choices=["online", "offline"])
    
    args = parser.parse_args()
    
    sizes = [args.size] if args.size else ["small", "medium", "large", "xlarge"]
    
    found_any = False
    for size in sizes:
        metrics_a = load_metrics("A", size, args.phase)
        metrics_b = load_metrics("B", size, args.phase)
        
        if metrics_a or metrics_b:
            found_any = True
            format_comparison_table(metrics_a, metrics_b, size, args.phase)
    
    if not found_any:
        print("\nNo experiment results found.")
        print("Run experiments first:")
        print("  python experiments/run_experiment.py --version A --size small --phase online")
        print("  python experiments/run_experiment.py --version B --size small --phase online")
        print("\nThen compare:")
        print("  python experiments/compare_versions.py --size small")


if __name__ == "__main__":
    main()
