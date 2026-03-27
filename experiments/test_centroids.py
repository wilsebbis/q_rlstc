import sys, os, pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_segment as rlstc_segment
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy

from q_rlstc.data.rlstc_mdp import TrajRLclus

data_path = 'q_rlstc/data/geolife_testdata'
cluster_path = 'q_rlstc/data/geolife_clustercenter'
env = TrajRLclus(data_path, cluster_path, cluster_path)

for c_id in sorted(env.clusters_E.keys()):
    pts = env.clusters_E[c_id][2]
    print(f"Cluster {c_id}: {len(pts)} points")
    if len(pts) > 0:
        print(f"  First point: x={pts[0].x}, y={pts[0].y}")
