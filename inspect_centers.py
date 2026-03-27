import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy
from q_rlstc.data.rlstc_mdp import TrajRLclus
env = TrajRLclus('q_rlstc/data/geolife_testdata', 'q_rlstc/data/geolife_clustercenter', 'q_rlstc/data/geolife_clustercenter')
for c_id in sorted(env.clusters_E.keys()):
    pts = env.clusters_E[c_id][2]
    print(f"Cluster {c_id}: {len(pts)} points")
