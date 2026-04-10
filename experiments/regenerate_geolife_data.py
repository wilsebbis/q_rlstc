#!/usr/bin/env python3
"""Regenerate GeoLife datasets with correct state-point splitting to fix map visuals."""

import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_rlstc.data.preprocessing import preprocess_pipeline

def regenerate_geolife():
    traj_path = "q_rlstc/data/Geolife"
    out_norm_path = "q_rlstc/data/geolife_norm_traj"
    out_test_path = "q_rlstc/data/geolife_testdata"
    
    print("1/3 Running preprocessing pipeline with state-point splitting...")
    # NOTE: split_by_time_gap is now included in preprocess_pipeline
    simplified_trajs = preprocess_pipeline(traj_path, out_norm_path)
    
    print("2/3 Generating test dataset (1000 items)...")
    test_trajs = simplified_trajs[:1000]
    with open(out_test_path, 'wb') as f:
        pickle.dump(test_trajs, f, protocol=2)
        
    print("3/3 Regenerating cluster centers is skipped to maintain K=10 centroid stability.")
    print("Data regenerated successfully. You can now test visual output with recreate_geolife_fig16.py.")

if __name__ == '__main__':
    regenerate_geolife()
