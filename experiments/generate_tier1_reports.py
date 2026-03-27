#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
import sys
import numpy as np
from pathlib import Path

def generate_reports(master_json_path: Path, output_dir: Path):
    if not master_json_path.exists():
        print(f"Error: {master_json_path} not found.")
        sys.exit(1)
        
    with open(master_json_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out random baseline context
    models = df[df['model'] != 'random']
    if models.empty:
        print("No trained model data available.")
        return
        
    print("\n" + "="*80)
    print("                 T I E R   1   E V A L U A T I O N   R E P O R T")
    print("="*80)
    
    # 1. Main Controlled Comparison Table
    print("\n--- 1. MAIN COMPARISON MATRIX (nValCR & Params) ---")
    # For a fair typical benchmark, slice at say ~15-20% cut budget
    fair_slice = models[(models['cut_budget'] >= 0.15) & (models['cut_budget'] <= 0.20)]
    if not fair_slice.empty:
        agg = fair_slice.groupby(['dataset', 'model', 'optimizer']).agg(
            nValCR_mean=('metric_value', lambda x: x[fair_slice.loc[x.index, 'metric_type'] == 'nValCR'].mean()),
            nValCR_std=('metric_value', lambda x: x[fair_slice.loc[x.index, 'metric_type'] == 'nValCR'].std()),
            params=('params', 'max')
        ).reset_index()
        print(agg.to_string(index=False))
        agg.to_csv(output_dir / "t1_main_comparison.csv", index=False)
    else:
        print("No data available at 15-20% cut budget slice.")
        
    # 2. Rank Instability Analysis
    print("\n--- 2. RANK INSTABILITY (raw ValCR vs nValCR) ---")
    raw_df = df[df['metric_type'] == 'raw_ValCR'].groupby(['dataset', 'model'])['metric_value'].mean().reset_index()
    nval_df = df[df['metric_type'] == 'nValCR'].groupby(['dataset', 'model'])['metric_value'].mean().reset_index()
    
    if not raw_df.empty and not nval_df.empty:
        for ds in df['dataset'].unique():
            print(f"\n  Dataset: {ds.upper()}")
            # Raw rankings (lower is better normally, but let's just show sorted)
            raw_s = raw_df[raw_df['dataset'] == ds].sort_values('metric_value')
            nval_s = nval_df[nval_df['dataset'] == ds].sort_values('metric_value')
            
            raw_ranks = {row['model']: i+1 for i, row in enumerate(raw_s.to_dict('records'))}
            nval_ranks = {row['model']: i+1 for i, row in enumerate(nval_s.to_dict('records'))}
            
            res = []
            for m in raw_ranks.keys():
                jump = raw_ranks[m] - nval_ranks[m]
                direction = "->" if jump == 0 else ("UP \u2191" if jump > 0 else "DN \u2193")
                res.append(f"{m:20s}: Raw Rank {raw_ranks[m]} | nValCR Rank {nval_ranks[m]} | {direction} (Shift: {abs(jump)})")
            
            for line in res: print("    " + line)
            
    # 3. Collapse & Stability Analysis
    print("\n--- 3. SEED STABILITY & COLLAPSE RATES ---")
    collapse_stats = models.groupby(['dataset', 'model', 'optimizer']).agg(
        Total_Runs=('collapse_label', 'count'),
        Healthy_Runs=('collapse_label', lambda x: (x == 'healthy').sum()),
        AlwaysCut_Runs=('collapse_label', lambda x: (x == 'always_cut').sum()),
        NeverCut_Runs=('collapse_label', lambda x: (x == 'never_cut').sum())
    ).reset_index()
    collapse_stats['Collapse_Rate_%'] = 100 * (1 - (collapse_stats['Healthy_Runs'] / collapse_stats['Total_Runs']))
    print(collapse_stats.to_string(index=False))
    collapse_stats.to_csv(output_dir / "t1_collapse_stats.csv", index=False)

    print("\nReports successfully written to", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=Path("results/tier1_main_matrix/master_results.json"))
    parser.add_argument("--out", type=Path, default=Path("results/tier1_reports"))
    args = parser.parse_args()
    generate_reports(args.json, args.out)
