import sys, os, pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_segment as rlstc_segment
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy

from q_rlstc.data.rlstc_mdp import TrajRLclus
from q_rlstc.data.rlstc_trajdistance import traj2trajIED

data_path = 'q_rlstc/data/geolife_testdata'
cluster_path = 'q_rlstc/data/geolife_clustercenter'

print("Loading Geolife environment...")
env = TrajRLclus(data_path, cluster_path, cluster_path)

trajectories = env.trajsdata[:1000]

# Update cluster generates the mathematical 'spokes'
env.update_cluster('E')

centroids = []
for c_id in sorted(env.clusters_E.keys()):
    # geometric mathematical spoke
    center_pts = env.clusters_E[c_id][2] 
    if not center_pts:
        continue
    
    # Find the single Original Full Trajectory that is closest to this spoke
    min_dist = float('inf')
    best_traj = None
    
    # Randomly downsample to 100 choices to avoid taking hours on IED distance
    import random
    choices = random.sample(trajectories, 100)
    
    for t in choices:
        dist = traj2trajIED(t.points, center_pts)
        if dist < min_dist:
            min_dist = dist
            best_traj = t
            
    if best_traj:
        # Save the points of the best natural trajectory
        centroids.append(best_traj.points)

print(f"Found {len(centroids)} natural global medoids.")
