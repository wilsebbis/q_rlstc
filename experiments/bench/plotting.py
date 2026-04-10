import os
import json
import matplotlib.pyplot as plt
import argparse

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_results(results_dir):
    data = {"paper_baseline": {}, "repo_baseline": {}}
    if not os.path.exists(results_dir):
        return data
        
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                content = json.load(f)
                mode = content.get("mode")
                backend = content.get("backend")
                if mode and backend:
                    data[mode][backend] = content["results"]
    return data

def plot_bar_comparison(data, mode, metric, y_label, title, output_filename):
    if mode not in data or not data[mode]:
        print(f"No data available for mode: {mode}")
        return

    backends = []
    values = []
    for backend, results in data[mode].items():
        if results and results.get(metric) is not None:
            backends.append(backend.capitalize())
            values.append(results[metric])
            
    if not backends:
        print(f"No {metric} data available to plot for {mode}.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(backends, values, color=['#1f77b4', '#ff7f0e'], width=0.5)
    
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot: {output_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Plotting Utility")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory containing JSON results")
    parser.add_argument("--plots-dir", type=str, default="./plots", help="Directory to save output PNG plots")
    parser.add_argument("--mode", type=str, default="paper_baseline", choices=["paper_baseline", "repo_baseline"], help="Which baseline to plot comparatives for.")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    plots_dir = os.path.abspath(args.plots_dir)
    ensure_dir(plots_dir)

    all_data = load_results(results_dir)

    # 1. Baseline Comparison Plot (OD)
    plot_bar_comparison(
        all_data, 
        mode=args.mode, 
        metric="OD", 
        y_label="Overall Distance (OD)", 
        title=f"Clustering Performance: Classical vs Quantum\n({args.mode})", 
        output_filename=os.path.join(plots_dir, f"{args.mode}_OD_comparison.png")
    )

    # 2. Execution Time Comparison Plot
    plot_bar_comparison(
        all_data, 
        mode=args.mode, 
        metric="runtime", 
        y_label="Execution Time (Seconds)", 
        title=f"Runtime Overhead: Classical vs Quantum\n({args.mode})", 
        output_filename=os.path.join(plots_dir, f"{args.mode}_Runtime_comparison.png")
    )

if __name__ == "__main__":
    main()
