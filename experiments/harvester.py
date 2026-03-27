import sys
from pathlib import Path
import json
import csv
import glob

def harvest_random_sweeps(results_dir: Path):
    """Load tuples from all *random_sweep.json files."""
    tuples = []
    for p in results_dir.rglob("*random_sweep.json"):
        with open(p, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                tuples.extend(data)
    return tuples

def get_best_metrics_under_budget(history: list, budget: float, mtype: str) -> dict:
    """Find the best valid point under a budget."""
    # Find all points <= budget
    valid = [h for h in history if h.get("cut_pct", float('inf')) <= budget * 100 + 0.5] # +0.5 for rounding
    if not valid:
        return None
    # the metric is either "val_cr", "nvalcr", "wvalcr"
    best = min(valid, key=lambda x: x[mtype])
    return best

def harvest_cross_comparisons(results_dir: Path):
    """Extract standard agent tuples from cross-comparison json histories."""
    cut_probs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    tuples = []
    
    for p in results_dir.rglob("comparison_results.json"):
        # We need to infer seed and dataset from the path or contents.
        # usually path is like results/tier1_main_matrix/tdrive/seed_1/comparison_results.json
        parts = p.parts
        dataset = "unknown"
        seed = 42
        for part in parts:
            if part in ["tdrive", "geolife", "porto"]:
                dataset = part
            if part.startswith("seed_"):
                try:
                    seed = int(part.split("_")[1])
                except:
                    pass
                    
        with open(p, "r") as f:
            data = json.load(f)
            
        # Parse 'classical' and 'quantum' chunks
        for sys_key, chunk in data.items():
            if not isinstance(chunk, dict) or "history" not in chunk:
                continue
            history = chunk["history"]
            
            # Identify model specifics
            if chunk.get("system") == "quantum_qrlstc_v_d":
                model_name = "vqc_v_d"
                opt = "spsa"
            elif chunk.get("system") == "classical_adam":
                model_name = "classic_dqn"
                opt = "adam"
            elif chunk.get("system") == "classical_spsa":
                model_name = "classic_dqn"
                opt = "spsa"
            else:
                model_name = sys_key
                opt = "unknown"
                
            for prob in cut_probs:
                for target_m, json_m in [("raw_ValCR", "val_cr"), ("nValCR", "nvalcr"), ("wValCR", "wvalcr")]:
                    best_pt = get_best_metrics_under_budget(history, prob, json_m)
                    
                    if best_pt:
                        realized = best_pt.get("cut_pct", 0) / 100.0
                        nsegs = best_pt.get("n_segments", 0)
                        
                        collapse = "healthy"
                        if realized < 0.005: collapse = "never_cut"
                        elif realized > 0.4: collapse = "always_cut"
                        
                        tup = {
                            "metric_type": target_m,
                            "metric_value": best_pt[json_m],
                            "cut_budget": float(prob),
                            "realized_cut": float(realized),
                            "seed": seed,
                            "dataset": dataset,
                            "model": model_name,
                            "optimizer": opt,
                            "params": chunk.get("param_count", 0),
                            "collapse_label": collapse,
                            "train_time_sec": chunk.get("elapsed_time", 0.0),
                            "shots_used": 1024 if "quantum" in sys_key else 0, # Hack for now if not tracked
                            "n_segments": nsegs,
                            "avg_seg_length": 0.0, 
                            "med_seg_length": 0.0,
                            "sse": best_pt.get("sse", 0.0),
                            "od": best_pt.get("val_cr", 0.0) # approx
                        }
                        tuples.append(tup)
                        
    return tuples

def main():
    results_dir = Path("results")
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
        
    print(f"Harvesting results from {results_dir}...")
    
    random_tuples = harvest_random_sweeps(results_dir)
    print(f"Found {len(random_tuples)} random policy metric tuples.")
    
    cross_tuples = harvest_cross_comparisons(results_dir)
    print(f"Found {len(cross_tuples)} trained model metric tuples.")
    
    all_tuples = random_tuples + cross_tuples
    if not all_tuples:
        print("No metrics found! Exiting.")
        return
        
    master_json = results_dir / "master_results.json"
    with open(master_json, "w") as f:
        json.dump(all_tuples, f, indent=2)
        
    # Write CSV for easier pandas loading if wanted
    master_csv = results_dir / "master_results.csv"
    keys = list(all_tuples[0].keys())
    with open(master_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for t in all_tuples:
            writer.writerow(t)
            
    print(f"\nSuccessfully wrote {len(all_tuples)} aggregated metrics into:")
    print(f"- {master_json}")
    print(f"- {master_csv}")

if __name__ == "__main__":
    main()
