import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from q_rlstc.data.rlstc_mdp import TrajRLclus

env = TrajRLclus('q_rlstc/data/geolife_testdata', 'q_rlstc/data/geolife_clustercenter', 'q_rlstc/data/geolife_clustercenter')

print("--- Geolife Clusters ---")
for c_id in sorted(env.clusters_E.keys()):
    entry = env.clusters_E[c_id]
    print(f"\nCluster {c_id}")
    for i, item in enumerate(entry):
        print(f"  idx {i}: type={type(item)}")
        if isinstance(item, list):
            print(f"    len={len(item)}")
            if len(item) > 0:
                print(f"    first element type={type(item[0])}")

print("\n--- Coordinate ranges ---")
trajectories = env.trajsdata[:10]
for idx, traj in enumerate(trajectories[:2]):
    xs = [p.x for p in traj.points[:10]]
    ys = [p.y for p in traj.points[:10]]
    print(f"traj {idx} sample x (lat?): {min(xs):.3f} to {max(xs):.3f}")
    print(f"traj {idx} sample y (lon?): {min(ys):.3f} to {max(ys):.3f}")

for c_id in sorted(env.clusters_E.keys())[:2]:
    centroid = env.clusters_E[c_id][2]
    xs = [p.x for p in centroid[:10]]
    ys = [p.y for p in centroid[:10]]
    print(f"centroid {c_id} [2] sample x: {min(xs):.3f} to {max(xs):.3f}")
    print(f"centroid {c_id} [2] sample y: {min(ys):.3f} to {max(ys):.3f}")

