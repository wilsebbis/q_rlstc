import os

base_dir = "RLSTCcode-main/subtrajcluster"

mappings = {
    "traj.py": "q_rlstc/data/rlstc_traj.py",
    "point.py": "q_rlstc/data/rlstc_point.py",
    "segment.py": "q_rlstc/data/rlstc_segment.py",
    "point_xy.py": "q_rlstc/data/rlstc_point_xy.py",
    "MDP.py": "q_rlstc/data/rlstc_mdp.py (augmented by CMDP constraints)",
    "MDPwoODb.py": "Deprecated. Fully subsumed by q_rlstc/data/rlstc_mdp.py",
    "cluster.py": "q_rlstc/data/rlstc_cluster.py",
    "initcenters.py": "Deprecated/Merged into q_rlstc/data/rlstc_cluster.py",
    "rl_splitmethod.py": "q_rlstc/rl/classical_agent.py and generic experiment runners",
    "rl_nn.py": "q_rlstc/rl/original_classical_agent.py and the new VQ-DQN/PyTorch networks",
    "rl_train.py": "experiments/run_cross_comparison.py (Unified training loop)",
    "rl_estimate.py": "experiments/run_cross_comparison.py (Unified evaluation loop)",
    "rl_estimatewoODb.py": "Deprecated.",
    "trajdistance.py": "q_rlstc/data/rlstc_trajdistance.py",
    "preprocessing.py": "q_rlstc/data/preprocessing.py",
    "iteration.py": "q_rlstc/baselines/ (Fixed Window, Heading Change)",
    "crosstrain.py": "experiments/run_cross_comparison.py",
    "crossvalidate.py": "experiments/run_cross_comparison.py"
}

for filename, new_location in mappings.items():
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename}, not found.")
        continue
        
    with open(filepath, "r") as f:
        content = f.read()
        
    # Check if already injected
    if "LEGACY ARCHITECTURE MAPPING:" in content:
        print(f"Skipping {filename}, already annotated.")
        continue
        
    docstring = f'''"""
=============================================================================
LEGACY ARCHITECTURE MAPPING:
This file is part of the original classical baseline (`RLSTCcode-main`).
In the modernized `q_rlstc` architecture, this component has been refactored,
deprecated, or replaced by:
-> {new_location}
=============================================================================
"""

'''
    
    # Insert right after any imports or just at the top if no imports, or after the first docstring
    # The safest is just to prepend it to the file.
    
    new_content = docstring + content
    
    with open(filepath, "w") as f:
        f.write(new_content)
        
    print(f"Annotated {filename}")

print("Done annotating legacy baseline.")
