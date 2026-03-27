import sys, os, math
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from q_rlstc.data.rlstc_mdp import TrajRLclus

env = TrajRLclus('q_rlstc/data/geolife_testdata', 'q_rlstc/data/geolife_clustercenter', 'q_rlstc/data/geolife_clustercenter')
trajectories = env.trajsdata[:1000]

# Geolife: p.x is Lat, p.y is Lon
# Center of the bounding box
lats = [p.x for t in trajectories for p in t.points]
lons = [p.y for t in trajectories for p in t.points]
center_lat = 39.93
center_lon = 116.40

buckets = {i: [] for i in range(8)} # 8 directional buckets

for t in trajectories:
    if not t.points: continue
    start = t.points[0]
    end = t.points[-1]
    
    # only consider trajectories that start near the core
    dist_to_center = math.hypot(start.x - center_lat, start.y - center_lon)
    if dist_to_center > 0.05:
        continue
        
    diff_lat = end.x - start.x
    diff_lon = end.y - start.y
    dist = math.hypot(diff_lat, diff_lon)
    
    # Must travel a reasonable distance
    if dist < 0.05: continue
        
    angle = math.degrees(math.atan2(diff_lat, diff_lon))
    if angle < 0: angle += 360
    
    bucket_idx = int(angle // 45)
    buckets[bucket_idx].append((dist, t))

representatives = []
for idx in range(8):
    if buckets[idx]:
        # get the one that travels the furthest in this direction
        best = max(buckets[idx], key=lambda x: x[0])[1]
        representatives.append(best)

print(f"Found {len(representatives)} global radial representatives.")
