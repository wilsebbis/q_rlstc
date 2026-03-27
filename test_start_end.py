import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from q_rlstc.data.rlstc_mdp import TrajRLclus
env = TrajRLclus('q_rlstc/data/geolife_testdata', 'q_rlstc/data/geolife_clustercenter', 'q_rlstc/data/geolife_clustercenter')
for c_id in sorted(env.clusters_E.keys()):
    pts = env.clusters_E[c_id][2]
    if pts:
        p1 = pts[0]
        p2 = pts[-1]
        print(f"Cluster {c_id}: Start({p1.x:.3f}, {p1.y:.3f}) -> End({p2.x:.3f}, {p2.y:.3f})")
