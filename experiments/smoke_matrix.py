#!/usr/bin/env python3
"""Smoke-Matrix Runner — minimal sanity check across all training modes.

Runs the smallest possible training configurations to verify:
  1. No hidden confounds between modes (shaping leaks, wrong protocol, etc.)
  2. All log fields are populated (ValCR, wValCR, medCR, Δ_rand, CUT%, λ)
  3. Parity mode header shows exact RLSTC-matched protocol keys
  4. JSON output is valid and contains expected fields

Configs (6 total):
  A) CONTROLLED_SPSA:  VQ-DQN  (statevector, 1 epoch, 2 seeds)
  B) CONTROLLED_SPSA:  SPSA-Classical  (1 epoch, 2 seeds)
  C) RLSTC_PARITY:     VQ-DQN  (statevector, 1 epoch, 2 seeds)
  D) RLSTC_PARITY:     Original-Classical  (1 epoch, 2 seeds)
  E) CONSTRAINED-ON:   VQ-DQN with USE_LAGRANGIAN=True (1 epoch, 2 seeds)
  F) CONSTRAINED-OFF:  VQ-DQN with USE_LAGRANGIAN=False (1 epoch, 2 seeds)

Runtime: ~3-8 minutes total (tiny data, 1 epoch, statevector).

Usage:
    python experiments/smoke_matrix.py
    python experiments/smoke_matrix.py --amount 50 --seeds 42,7
    python experiments/smoke_matrix.py --traj-path /path/to/data --centers-path /path/to/centers
"""

import argparse
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Project setup ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_DIR = _PROJECT_ROOT / "q_rlstc" / "data"

from run_thesis_experiments import (
    ModelSpec,
    TrainingMode,
    PROTOCOL,
    RLSTC_PARITY_PROTOCOL,
    build_agent,
    train_and_evaluate,
    TeePrinter,
    _collect_env_metadata,
)

# ═══════════════════════════════════════════════════════════════════
#  Smoke-matrix configurations
# ═══════════════════════════════════════════════════════════════════

def _smoke_specs() -> List[Tuple[str, ModelSpec, str]]:
    """Return (label, spec, description) tuples for the smoke matrix.

    Each entry is one cell in the matrix. Labels are short IDs for
    compact log output.
    """
    return [
        # ── A) Controlled SPSA: VQ-DQN ─────────────────────────
        (
            "CTRL-VQ",
            ModelSpec(
                name="VQ-DQN-5q-3L",
                kind="quantum",
                n_qubits=5,
                n_layers=3,
                shots=0,
                training_mode=TrainingMode.CONTROLLED_SPSA,
            ),
            "Controlled SPSA / VQ-DQN (statevector)",
        ),
        # ── B) Controlled SPSA: Classical ────────────────────────
        (
            "CTRL-CL",
            ModelSpec(
                name="SPSA-64-34p",
                kind="classical",
                hidden_sizes=[64],
                training_mode=TrainingMode.CONTROLLED_SPSA,
            ),
            "Controlled SPSA / Classical DQN [64]",
        ),
        # ── C) RLSTC Parity: VQ-DQN ────────────────────────────
        (
            "PAR-VQ",
            ModelSpec(
                name="VQ-DQN-Parity",
                kind="quantum",
                n_qubits=5,
                n_layers=3,
                shots=0,
                training_mode=TrainingMode.RLSTC_PARITY,
            ),
            "RLSTC Parity / VQ-DQN (statevector)",
        ),
        # ── D) RLSTC Parity: Original Classical ──────────────────
        (
            "PAR-ORIG",
            ModelSpec(
                name="Original-DQN-Parity",
                kind="original",
                training_mode=TrainingMode.RLSTC_PARITY,
            ),
            "RLSTC Parity / Original 5→64→2 DQN",
        ),
        # ── E) Constrained ON: VQ-DQN ───────────────────────────
        (
            "LAG-ON",
            ModelSpec(
                name="VQ-DQN-Lagrangian-ON",
                kind="quantum",
                n_qubits=5,
                n_layers=3,
                shots=0,
                training_mode=TrainingMode.CONTROLLED_SPSA,
            ),
            "Lagrangian ON (USE_LAGRANGIAN=True)",
        ),
        # ── F) Constrained OFF: VQ-DQN ──────────────────────────
        (
            "LAG-OFF",
            ModelSpec(
                name="VQ-DQN-Lagrangian-OFF",
                kind="quantum",
                n_qubits=5,
                n_layers=3,
                shots=0,
                training_mode=TrainingMode.CONTROLLED_SPSA,
            ),
            "Lagrangian OFF (USE_LAGRANGIAN=False — for A/B comparison)",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
#  Log field validators
# ═══════════════════════════════════════════════════════════════════

# Fields that MUST appear in every result JSON
REQUIRED_JSON_FIELDS = {
    "model_name", "seed", "best_val_cr", "best_greedy_cut_pct",
    "val_crs", "total_episodes",
}

# Fields specific to controlled (non-parity) mode
CONTROLLED_EXTRA_FIELDS = {
    "val_wvalcr",
}

# Fields that must exist when Lagrangian is ON
LAGRANGIAN_FIELDS = {
    "final_lagrangian_lambda",
}


def validate_result(label: str, result: Dict[str, Any], is_parity: bool, has_lagrangian: bool) -> List[str]:
    """Validate result JSON contains all expected fields.

    Returns list of validation errors (empty = all good).
    """
    errors = []

    for field in REQUIRED_JSON_FIELDS:
        if field not in result:
            errors.append(f"[{label}] Missing required field: {field}")

    if not is_parity:
        for field in CONTROLLED_EXTRA_FIELDS:
            if field not in result:
                errors.append(f"[{label}] Missing controlled field: {field}")

    if has_lagrangian:
        for field in LAGRANGIAN_FIELDS:
            if field not in result:
                errors.append(f"[{label}] Missing Lagrangian field: {field}")
            elif result.get(field) is None:
                errors.append(f"[{label}] Lagrangian field is None: {field}")

    # ValCR should be positive
    if "best_val_cr" in result and result["best_val_cr"] is not None:
        if result["best_val_cr"] <= 0:
            errors.append(f"[{label}] ValCR not positive: {result['best_val_cr']}")

    # Budget violation flag
    if "budget_violated" in result and result["budget_violated"]:
        errors.append(f"[{label}] ⚠ budget_violated=True (informational, not fatal)")

    return errors


def validate_log_output(label: str, log_text: str, is_parity: bool) -> List[str]:
    """Validate that printed log contains expected metric strings.

    Returns list of validation warnings.
    """
    warnings = []

    # Every run should print ValCR
    if "ValCR=" not in log_text and "valcr" not in log_text.lower():
        warnings.append(f"[{label}] Log missing ValCR output")

    # Controlled runs should print wCR
    if not is_parity and "wCR=" not in log_text:
        warnings.append(f"[{label}] Log missing wCR (weighted ValCR) output")

    # Parity runs should mention the protocol
    if is_parity and "parity" not in log_text.lower() and "RLSTC" not in log_text:
        warnings.append(f"[{label}] Parity run log doesn't mention parity/RLSTC")

    return warnings


# ═══════════════════════════════════════════════════════════════════
#  Parity protocol header
# ═══════════════════════════════════════════════════════════════════

def print_protocol_header(spec: ModelSpec):
    """Print the active protocol keys for verification."""
    is_parity = spec.training_mode == TrainingMode.RLSTC_PARITY
    proto = RLSTC_PARITY_PROTOCOL if is_parity else PROTOCOL
    mode_label = "RLSTC_PARITY" if is_parity else "CONTROLLED_SPSA"

    print(f"\n{'='*60}")
    print(f"  Protocol: {mode_label}")
    print(f"{'='*60}")

    key_params = [
        "gamma", "epsilon_decay", "EPSILON_DECAY_MODE",
        "EPSILON_DECAY_PER_STEP", "L_MIN", "CUT_PENALTY",
        "EXTEND_COST", "COMPLEXITY_LAMBDA", "MIN_CUT_BONUS",
        "USE_LAGRANGIAN", "REWARD_MODE", "EXPLORATION_MODE",
        "FORCED_CUT_PROB", "OPTIMISTIC_CUT_BIAS",
        "USE_STRATIFIED_REPLAY",
    ]
    for k in key_params:
        v = proto.get(k, "N/A")
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════════

def run_smoke_matrix(
    traj_path: str,
    centers_path: str,
    n_trajectories: int = 100,
    n_epochs: int = 1,
    seeds: List[int] = None,
    output_dir: str = "results/smoke",
) -> Dict[str, Any]:
    """Run the full smoke matrix and return aggregated results.

    Returns dict with per-config results and validation summary.
    """
    if seeds is None:
        seeds = [42, 7]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    specs = _smoke_specs()
    all_results = {}
    all_errors = []
    all_warnings = []
    timings = {}

    total = len(specs) * len(seeds)
    completed = 0

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  SMOKE MATRIX — Q-RLSTC Sanity Check".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    print(f"║  Configs: {len(specs)}  |  Seeds: {seeds}  |  "
          f"Epochs: {n_epochs}  |  Trajectories: {n_trajectories}".ljust(68) + " ║")
    print("╚" + "═"*68 + "╝\n")

    for label, spec, description in specs:
        is_parity = spec.training_mode == TrainingMode.RLSTC_PARITY
        has_lagrangian = (not is_parity) and (label != "LAG-OFF")

        print_protocol_header(spec)
        print(f"▶ [{label}] {description}")
        print(f"  Seeds: {seeds}")

        config_results = []

        for seed in seeds:
            completed += 1
            run_label = f"{label}/s{seed}"
            print(f"\n  ── [{completed}/{total}] {run_label} ──")

            # Build agent
            agent = build_agent(spec, seed)

            # For LAG-OFF: temporarily disable Lagrangian
            # (We monkey-patch PROTOCOL for this specific run)
            original_lag = None
            if label == "LAG-OFF":
                original_lag = PROTOCOL["USE_LAGRANGIAN"]
                PROTOCOL["USE_LAGRANGIAN"] = False

            # Capture stdout for log validation
            tee = TeePrinter()
            t0 = time.time()

            try:
                with tee:
                    result = train_and_evaluate(
                        agent=agent,
                        spec=spec,
                        traj_path=traj_path,
                        centers_path=centers_path,
                        n_trajectories=n_trajectories,
                        n_epochs=n_epochs,
                        seed=seed,
                    )
                elapsed = time.time() - t0
                timings[run_label] = elapsed

                # Validate JSON fields
                errs = validate_result(run_label, result, is_parity, has_lagrangian)
                all_errors.extend(errs)

                # Validate log output
                warns = validate_log_output(run_label, tee.getvalue(), is_parity)
                all_warnings.extend(warns)

                config_results.append({
                    "seed": seed,
                    "result": result,
                    "elapsed_s": elapsed,
                    "errors": errs,
                    "warnings": warns,
                })

                # Summary line
                vcr = result.get("best_val_cr", "?")
                cut = result.get("best_greedy_cut_pct", "?")
                lam = result.get("final_lagrangian_lambda", "—")
                wcr = result.get("val_wvalcr", "—")
                print(f"  ✓ ValCR={vcr} CUT%={cut} λ={lam} wCR={wcr} ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                all_errors.append(f"[{run_label}] CRASHED: {e}")
                config_results.append({
                    "seed": seed,
                    "result": None,
                    "elapsed_s": elapsed,
                    "errors": [f"CRASH: {e}"],
                    "warnings": [],
                })
                print(f"  ✗ CRASH after {elapsed:.1f}s: {e}")

            finally:
                # Restore protocol if we patched it
                if original_lag is not None:
                    PROTOCOL["USE_LAGRANGIAN"] = original_lag

        all_results[label] = {
            "description": description,
            "is_parity": is_parity,
            "has_lagrangian": has_lagrangian,
            "runs": config_results,
        }

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  SMOKE MATRIX SUMMARY".center(68) + "║")
    print("╠" + "═"*68 + "╣")

    n_errors = len([e for e in all_errors if "CRASH" in e or "Missing" in e])
    n_warns = len(all_warnings)

    if n_errors == 0:
        print("║" + "  ✅ ALL CONFIGS PASSED — no crashes, all fields present".ljust(68) + "║")
    else:
        print("║" + f"  ❌ {n_errors} ERROR(S) FOUND".ljust(68) + "║")
        for e in all_errors:
            if "CRASH" in e or "Missing" in e:
                print("║" + f"    • {e}".ljust(68) + "║")

    if n_warns > 0:
        print("║" + f"  ⚠ {n_warns} WARNING(S)".ljust(68) + "║")
        for w in all_warnings:
            print("║" + f"    • {w}".ljust(68) + "║")

    print("╠" + "═"*68 + "╣")
    total_time = sum(timings.values())
    print("║" + f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)".ljust(68) + "║")

    # Per-config timing
    for label, _, desc in specs:
        config_time = sum(v for k, v in timings.items() if k.startswith(label))
        print("║" + f"    {label}: {config_time:.1f}s — {desc[:45]}".ljust(68) + "║")

    print("╚" + "═"*68 + "╝")

    # ── Cross-config comparison table ──────────────────────────
    print("\n┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Config      │ ValCR    │ CUT%     │ λ        │ wCR      │")
    print("├─────────────┼──────────┼──────────┼──────────┼──────────┤")

    for label, _, _ in specs:
        runs = all_results[label]["runs"]
        for r in runs:
            if r["result"] is None:
                print(f"│ {label:11s} │ CRASHED  │          │          │          │")
                continue
            res = r["result"]
            vcr = f"{res.get('best_val_cr', 0):.4f}"
            cut = f"{res.get('best_greedy_cut_pct', 0):.1f}%"
            lam = res.get("final_lagrangian_lambda")
            lam_s = f"{lam:.4f}" if lam is not None else "—"
            wcr = res.get("val_wvalcr")
            wcr_s = f"{wcr:.4f}" if wcr is not None else "—"
            print(f"│ {label:11s} │ {vcr:8s} │ {cut:8s} │ {lam_s:8s} │ {wcr_s:8s} │")

    print("└─────────────┴──────────┴──────────┴──────────┴──────────┘")

    # ── Save JSON ─────────────────────────────────────────────
    report = {
        "metadata": _collect_env_metadata(),
        "config": {
            "n_trajectories": n_trajectories,
            "n_epochs": n_epochs,
            "seeds": seeds,
        },
        "results": {},
        "errors": all_errors,
        "warnings": all_warnings,
        "timings": timings,
    }

    # Serialize results (skip non-serializable objects)
    for label, data in all_results.items():
        report["results"][label] = {
            "description": data["description"],
            "is_parity": data["is_parity"],
            "has_lagrangian": data["has_lagrangian"],
            "runs": [
                {
                    "seed": r["seed"],
                    "elapsed_s": r["elapsed_s"],
                    "errors": r["errors"],
                    "warnings": r["warnings"],
                    "best_val_cr": r["result"].get("best_val_cr") if r["result"] else None,
                    "best_cut_pct": r["result"].get("best_greedy_cut_pct") if r["result"] else None,
                    "final_lambda": r["result"].get("final_lagrangian_lambda") if r["result"] else None,
                    "val_wvalcr": r["result"].get("val_wvalcr") if r["result"] else None,
                    "budget_violated": r["result"].get("budget_violated") if r["result"] else None,
                }
                for r in data["runs"]
            ],
        }

    json_path = out_path / "smoke_matrix.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Report saved to: {json_path}")

    return report


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Q-RLSTC Smoke Matrix — minimal sanity check across all modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--traj-path",
        default=str(_DATA_DIR / "Tdrive_norm_traj"),
        help="Path to trajectory data",
    )
    parser.add_argument(
        "--centers-path",
        default=str(_DATA_DIR / "tdrive_clustercenter"),
        help="Path to cluster centers",
    )
    parser.add_argument(
        "--amount", type=int, default=100,
        help="Number of trajectories (default: 100, keep small for smoke)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs (default: 1 — smoke test only)",
    )
    parser.add_argument(
        "--seeds", default="42,7",
        help="Comma-separated seeds (default: 42,7)",
    )
    parser.add_argument(
        "--output-dir", default="results/smoke",
        help="Output directory (default: results/smoke)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    report = run_smoke_matrix(
        traj_path=args.traj_path,
        centers_path=args.centers_path,
        n_trajectories=args.amount,
        n_epochs=args.epochs,
        seeds=seeds,
        output_dir=args.output_dir,
    )

    # Exit with error code if any crashes
    n_crashes = len([e for e in report["errors"] if "CRASH" in e])
    sys.exit(1 if n_crashes > 0 else 0)


if __name__ == "__main__":
    main()
