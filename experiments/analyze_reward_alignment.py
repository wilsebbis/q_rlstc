#!/usr/bin/env python3
"""A4-ALIGNMENT: Reward-Metric Alignment Analyzer.

This script parses the generated JSON experiment results (e.g., E1) 
and plots the correlation between the RLSTC clustering metrics 
(ValCR, Overal Distance) and standard trajectory metrics 
(Discrete Fréchet distance, Dynamic Time Warping).

Usage:
    python analyze_reward_alignment.py --results-dir results/thesis
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/thesis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plot_dir = results_dir / "alignment_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(results_dir.glob("thesis_results_*.json"))
    if not json_files:
        print(f"No json files found in {results_dir}")
        return

    n_plots = 0
    for jfile in json_files:
        with open(jfile, "r") as f:
            data = json.load(f)

        for exp_name, exp_runs in data.items():
            if not isinstance(exp_runs, list):
                continue
                
            for run_idx, run in enumerate(exp_runs):
                model_name = run.get("model", f"run_{run_idx}")
                safe_name = model_name.replace(" ", "_").replace("/", "")
                
                dtws = run.get("val_dtws", [])
                frechets = run.get("val_frechets", [])
                valcrs = run.get("val_crs", [])
                mdls = run.get("val_mdls", [])
                mhds = run.get("val_mhds", [])
                
                # Cleanup infs
                dtws = [d if d != float('inf') else np.nan for d in dtws]
                frechets = [f if f != float('inf') else np.nan for f in frechets]
                mdls = [m if m != float('inf') else np.nan for m in mdls]
                mhds = [m if m != float('inf') else np.nan for m in mhds]
                
                if not dtws or not valcrs or np.isnan(dtws).all():
                    continue  # Metric tracking not present or failed
                
                epochs = list(range(1, len(valcrs) + 1))
                
                # Figure 1: Standard Metrics Evolution (DTW/Fréchet)
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.scatter(valcrs, dtws, c=epochs, cmap="plasma", s=50, edgecolors='k')
                ax1.set_xlabel("ValCR (RLSTC Primary Metric)")
                ax1.set_ylabel("Mean DTW (Standard Metric)")
                ax1.set_title(f"DTW vs ValCR\n{model_name}")
                
                # Annotate epochs
                for i, (x, y) in enumerate(zip(valcrs, dtws)):
                    if not np.isnan(y):
                        ax1.annotate(f"E{epochs[i]}", (x, y), fontsize=8, textcoords="offset points", xytext=(3, 3))
                
                # Plot 2: Frechet vs ValCR evolution
                ax2.scatter(valcrs, frechets, c=epochs, cmap="viridis", s=50, edgecolors='k')
                ax2.set_xlabel("ValCR (RLSTC Primary Metric)")
                ax2.set_ylabel("Mean Fréchet Distance")
                ax2.set_title(f"Fréchet vs ValCR\n{model_name}")
                
                for i, (x, y) in enumerate(zip(valcrs, frechets)):
                    if not np.isnan(y):
                        ax2.annotate(f"E{epochs[i]}", (x, y), fontsize=8, textcoords="offset points", xytext=(3, 3))
                
                fig1.tight_layout()
                fig1.savefig(str(plot_dir / f"alignment_scatter_standard_{exp_name}_{safe_name}.png"), dpi=150)
                plt.close(fig1)
                n_plots += 1
                
                # Figure 2: Advanced Robust Metrics Evolution (MDL/MHD)
                if mdls and mhds and not np.isnan(mdls).all():
                    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    ax3.scatter(valcrs, mdls, c=epochs, cmap="cool", s=50, edgecolors='k')
                    ax3.set_xlabel("ValCR (RLSTC Primary Metric)")
                    ax3.set_ylabel("MDL Cost")
                    ax3.set_title(f"MDL vs ValCR\n{model_name}")
                    
                    for i, (x, y) in enumerate(zip(valcrs, mdls)):
                        if not np.isnan(y):
                            ax3.annotate(f"E{epochs[i]}", (x, y), fontsize=8, textcoords="offset points", xytext=(3, 3))
                            
                    ax4.scatter(valcrs, mhds, c=epochs, cmap="Wistia", s=50, edgecolors='k')
                    ax4.set_xlabel("ValCR (RLSTC Primary Metric)")
                    ax4.set_ylabel("Mean MHD")
                    ax4.set_title(f"Modified Hausdorff Distance vs ValCR\n{model_name}")
                    
                    for i, (x, y) in enumerate(zip(valcrs, mhds)):
                        if not np.isnan(y):
                            ax4.annotate(f"E{epochs[i]}", (x, y), fontsize=8, textcoords="offset points", xytext=(3, 3))
                            
                    fig2.tight_layout()
                    fig2.savefig(str(plot_dir / f"alignment_scatter_advanced_{exp_name}_{safe_name}.png"), dpi=150)
                    plt.close(fig2)
                    n_plots += 1
                
                # Figure 3: Dual Y-axis trajectory overlay over time
                fig3, ax_y1 = plt.subplots(figsize=(8, 4))
                ax_y2 = ax_y1.twinx()
                
                l1 = ax_y1.plot(epochs, valcrs, 'b.-', label="ValCR")
                l2 = ax_y2.plot(epochs, dtws, 'r.-', label="DTW")
                if mdls and not np.isnan(mdls).all():
                    l3 = ax_y2.plot(epochs, mdls, 'g.--', label="MDL")
                else:
                    l3 = []
                
                ax_y1.set_xlabel("Epoch")
                ax_y1.set_ylabel("ValCR", color='blue')
                ax_y2.set_ylabel("DTW/MDL", color='red')
                
                lines = l1 + l2 + l3
                labels = [l.get_label() for l in lines]
                ax_y1.legend(lines, labels, loc='upper center')
                
                plt.title(f"Tracking Trajectory Training\n{model_name}")
                fig3.tight_layout()
                fig3.savefig(str(plot_dir / f"alignment_timeline_{exp_name}_{safe_name}.png"), dpi=150)
                plt.close(fig3)
                n_plots += 1
                
    print(f"Generated {n_plots} diagnostic plots in {plot_dir}")

if __name__ == "__main__":
    main()
