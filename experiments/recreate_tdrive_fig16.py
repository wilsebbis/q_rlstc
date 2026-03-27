import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy

from q_rlstc.data.rlstc_mdp import TrajRLclus
from q_rlstc.visualization.plot_utils import plot_paper_style_clusters

def main():
    data_path = 'q_rlstc/data/Tdrive_testdata'
    cluster_path = 'q_rlstc/data/tdrive_clustercenter_5'
    
    print("Loading T-Drive environment (1000 trajectories and 5 cluster centers)...")
    env = TrajRLclus(data_path, cluster_path, cluster_path)
    
    trajectories = env.trajsdata[:1000]
    
    import math
    # Beijing Center. For T-Drive, p.x is Lon, p.y is Lat
    center_lon, center_lat = 116.40, 39.93
    buckets = {i: [] for i in range(8)}
    
    for t in trajectories:
        if not t.points:
            continue
        start = t.points[0]
        end = t.points[-1]
        
        # T-Drive: p.x is Lon, p.y is Lat
        dist_to_center = math.hypot(start.y - center_lat, start.x - center_lon)
        if dist_to_center > 0.05:
            continue
            
        diff_lat = end.y - start.y
        diff_lon = end.x - start.x
        dist = math.hypot(diff_lat, diff_lon)
        
        if dist < 0.05:
            continue
            
        angle = math.degrees(math.atan2(diff_lat, diff_lon))
        if angle < 0:
            angle += 360
        
        bucket_idx = int(angle // 45)
        buckets[bucket_idx].append((dist, t))

    centroids = []
    for idx in range(8):
        if buckets[idx]:
            best_traj = max(buckets[idx], key=lambda x: x[0])[1]
            centroids.append(best_traj.points)

    print(f"Loaded {len(trajectories)} trajectories and extracted {len(centroids)} global radial representatives.")
    
    out_file = "recreated_tdrive_fig16.png"
    print(f"Generating plot at {out_file}...")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    lon_bounds = (116.1, 116.7)
    lat_bounds = (39.7, 40.15)
    
    for traj in trajectories:
        if not traj or not traj.points:
            continue
        # T-drive: x is Longitude, y is Latitude
        ax.plot([p.x for p in traj.points], [p.y for p in traj.points],
                color="blue", linewidth=0.3, alpha=0.4)
                
    for centroid in centroids:
        if not centroid:
            continue
        # T-drive: x is Longitude, y is Latitude
        ax.plot([p.x for p in centroid], [p.y for p in centroid],
                color="red", linewidth=2.0, alpha=0.9, solid_capstyle="round")

    ax.set_xlim(*lon_bounds)
    ax.set_ylim(*lat_bounds)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Recreated Fig. 16: T-Drive Visualization")
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect('auto')
    
    ax.legend(handles=[
        plt.Line2D([0], [0], color="blue", lw=1, label="Trajectories"),
        plt.Line2D([0], [0], color="red", lw=2, label="Representative trajectories"),
    ], loc="lower right", framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print("Done!")

if __name__ == '__main__':
    main()
