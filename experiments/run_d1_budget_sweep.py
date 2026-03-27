import argparse
import json
import time
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

from q_rlstc.data.rlstc_mdp import TrajRLclus
from q_rlstc.data.rlstc_cluster import (
    compute_overdist, 
    compute_overdist_per_point, 
    compute_overdist_length_weighted,
    compute_sse,
)

def run_random_sweep(args):
    """Run a random policy sweep across multiple target CUT probabilities."""
    
    cut_probs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    amount = args.amount
    sidx = int(amount * 0.9)
    eidx = amount
    
    for seed in args.seeds:
        np.random.seed(seed)
        random.seed(seed)
        
        # We need a fresh env for every seed to ensure consistent starting points 
        # (though env data is mostly static after loading)
        print(f"\n--- Loading TrajRLclus for Seed {seed} ---")
        env = TrajRLclus(args.traj_path, args.centers_path, args.centers_path)
        
        for prob in cut_probs:
            start_time = time.time()
            
            # Reset eval clusters
            for i in env.clusters_E.keys():
                env.clusters_E[i][0] = []
                env.clusters_E[i][1] = []
                env.clusters_E[i][3] = defaultdict(list)
                if len(env.clusters_E[i]) > 4:
                    env.clusters_E[i][4] = []
                else:
                    env.clusters_E[i].append([]) # initialize segment lengths
            
            val_n_extend = 0
            val_n_cut = 0
            
            for e in range(sidx, eidx):
                observation, steps = env.reset(e, "E")
                seg_len = 0
                for idx in range(1, steps):
                    # Random policy
                    act = 1 if np.random.rand() < prob else 0
                    
                    # Force extend if below L_MIN
                    seg_len += 1
                    if act == 1 and seg_len < 3:
                        act = 0
                        
                    if act == 0:
                        val_n_extend += 1
                    else:
                        val_n_cut += 1
                        seg_len = 0 # reset length counter on cut
                        
                    observation_, _ = env.step(e, act, idx, "E")
                    observation = observation_
                    
            # Compute Metrics
            val_od = compute_overdist(env.clusters_E)
            raw_valcr = val_od / env.basesim_E if env.basesim_E > 0 else float('inf')
            
            n_od = compute_overdist_per_point(env.clusters_E)
            nvalcr = n_od / env.basesim_E if env.basesim_E > 0 else float('inf')
            
            w_od = compute_overdist_length_weighted(env.clusters_E)
            wvalcr = w_od / env.basesim_E if env.basesim_E > 0 else float('inf')
            
            sse = compute_sse(env.clusters_E)
            
            val_total = val_n_extend + val_n_cut
            realized_cut = val_n_cut / val_total if val_total > 0 else 0.0
            
            # Segments stats
            all_segment_lengths = []
            for c_id in env.clusters_E:
                if len(env.clusters_E[c_id]) > 4:
                    all_segment_lengths.extend(env.clusters_E[c_id][4])
                    
            n_segments = len(all_segment_lengths)
            avg_seg_length = np.mean(all_segment_lengths) if n_segments > 0 else 0.0
            med_seg_length = np.median(all_segment_lengths) if n_segments > 0 else 0.0
            
            wall_time = time.time() - start_time
            
            # Collapse labeling
            if realized_cut < 0.005 or n_segments <= (eidx - sidx):
                collapse = "never_cut"
            elif realized_cut > 0.4 and med_seg_length <= 3.0:
                collapse = "always_cut"
            else:
                collapse = "healthy"
                
            print(f"Seed {seed} | Prob {prob:.2f} | CUT% {realized_cut*100:05.2f}% | "
                  f"rawValCR: {raw_valcr:.4f} | nValCR: {nvalcr:.4f} | wValCR: {wvalcr:.4f} | "
                  f"Segs: {n_segments} | AvgLen: {avg_seg_length:.1f}")
                  
            base_tuple = {
                "cut_budget": float(prob),
                "realized_cut": float(realized_cut),
                "seed": seed,
                "dataset": args.dataset_name,
                "model": "random",
                "optimizer": "none",
                "params": 0,
                "collapse_label": collapse,
                "train_time_sec": float(wall_time),
                "shots_used": 0,
                "n_segments": int(n_segments),
                "avg_seg_length": float(avg_seg_length),
                "med_seg_length": float(med_seg_length),
                "sse": float(sse),
                "od": float(val_od)
            }
            
            for m_type, m_val in [("raw_ValCR", raw_valcr), ("nValCR", nvalcr), ("wValCR", wvalcr)]:
                tup = base_tuple.copy()
                tup["metric_type"] = m_type
                tup["metric_value"] = float(m_val)
                results_list.append(tup)
                
    # Save the huge list of tuples to json
    out_file = out_dir / f"{args.dataset_name}_random_sweep.json"
    with open(out_file, "w") as f:
        json.dump(results_list, f, indent=2)
        
    print(f"\n[+] Saved random sweep results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", required=True)
    parser.add_argument("--centers-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--amount", type=int, default=500, help="Number of trajectories")
    
    args = parser.parse_args()
    run_random_sweep(args)
