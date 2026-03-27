import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_segment as rlstc_segment
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy

from q_rlstc.data.rlstc_mdp import TrajRLclus

env = TrajRLclus('q_rlstc/data/Tdrive_testdata', 'q_rlstc/data/tdrive_clustercenter', 'q_rlstc/data/tdrive_clustercenter')

print("Sub-trajectory point 0:")
print(f"  x={env.trajsdata[0].points[0].x}, y={env.trajsdata[0].points[0].y}")

print(f"T-Drive has {len(env.clusters_E.keys())} clusters.")

print("Centroid point 0:")
pts = env.clusters_E[0][2]
print(f"  x={pts[0].x}, y={pts[0].y}")
