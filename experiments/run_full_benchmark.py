#!/usr/bin/env python3
"""Full benchmark runner for Q-RLSTC with plots and metrics.

Runs Classical Parity (5q) and Quantum Enhanced (8q) on synthetic data,
optionally under NISQ noise models, then generates publication-quality
plots and a metrics summary.

Usage:
    # Quick smoke test (noiseless, both versions, auto backend)
    python experiments/run_full_benchmark.py

    # Only Classical Parity version
    python experiments/run_full_benchmark.py --version A

    # Only Quantum Enhanced version
    python experiments/run_full_benchmark.py --version B

    # Medium tier with noise comparison on MLX
    python experiments/run_full_benchmark.py --tier medium --noise --backend mlx

    # Compare all available compute backends
    python experiments/run_full_benchmark.py --compare-backends

    # Resume an interrupted run
    python experiments/run_full_benchmark.py --resume

    # Custom seed and output directory
    python experiments/run_full_benchmark.py --seed 123 --output-dir ./my_results
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from q_rlstc.accelerator import get_device_info, resolve_backend
from q_rlstc.config import VERSION_LABELS
from q_rlstc.visualization.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="Q-RLSTC Full Benchmark with Plots & Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Quick noiseless run, both versions
  %(prog)s --version A                  # Classical Parity only
  %(prog)s --version B                  # Quantum Enhanced only
  %(prog)s --tier medium --noise        # Full comparison with noise
  %(prog)s --backend mlx                # Force Apple MLX acceleration
  %(prog)s --compare-backends           # Compare CPU vs MLX vs CUDA
        """,
    )
    parser.add_argument(
        "--tier", type=str, default="small",
        choices=["small", "medium", "large"],
        help="Benchmark tier: small (~5 min), medium (~15 min), large (~45+ min)"
    )
    parser.add_argument(
        "--version", type=str, default="both",
        choices=["A", "B", "both"],
        help=(
            "Which version to run: "
            f"A = {VERSION_LABELS['A']}, "
            f"B = {VERSION_LABELS['B']}, "
            "both = run both (default)"
        ),
    )
    parser.add_argument(
        "--noise", action="store_true",
        help="Include Eagle and Heron noise model runs (default: noiseless only)"
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "cpu", "mlx", "cuda"],
        help="Compute backend for array ops (default: auto-detect)"
    )
    parser.add_argument(
        "--compare-backends", action="store_true",
        help="Run benchmark on every available backend and compare timing"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory"
    )

    args = parser.parse_args()

    # Show device info
    dev = get_device_info()
    print(f"\n🖥  Detected backends: {dev['backends']}")
    print(f"   Best: {dev['best'].upper()}")

    versions = ["A", "B"] if args.version == "both" else [args.version]
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.compare_backends:
        # Run on every available backend
        backends = ["cpu"]
        if dev["backends"].get("mlx"):
            backends.append("mlx")
        if dev["backends"].get("cuda"):
            backends.append("cuda")

        print(f"\n🔄  Backend comparison mode: {backends}")
        all_results = {}

        for be in backends:
            print(f"\n{'═' * 60}")
            print(f"  BACKEND: {be.upper()}")
            print(f"{'═' * 60}")

            runner = BenchmarkRunner(
                tier=args.tier,
                seed=args.seed,
                output_dir=output_dir,
                include_noise=args.noise,
                versions=versions,
                compute_backend=be,
            )
            results = runner.run(resume=args.resume)
            all_results[be] = results
            runner.print_summary(results)

        # Print cross-backend timing comparison
        print(f"\n{'═' * 60}")
        print("  CROSS-BACKEND TIMING COMPARISON")
        print(f"{'═' * 60}")
        for key in sorted(next(iter(all_results.values())).keys()):
            print(f"\n  {key}:")
            for be, res_dict in all_results.items():
                if key in res_dict:
                    t = res_dict[key].runtime_seconds
                    print(f"    {be.upper():<8}: {t:.2f}s")
    else:
        runner = BenchmarkRunner(
            tier=args.tier,
            seed=args.seed,
            output_dir=output_dir,
            include_noise=args.noise,
            versions=versions,
            compute_backend=args.backend,
        )

        results = runner.run(resume=args.resume)

        # Generate plots
        try:
            runner.generate_plots(results)
        except ImportError:
            print("\n⚠ matplotlib not installed — skipping plots.")
            print("  Install with: uv pip install matplotlib")

        runner.print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
