import sys
import os
import pickle

# Ensure we can import q_rlstc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# The original pickled files from RLSTCcode expect a module named 'traj' or 'point'
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['point_xy'] = rlstc_point_xy

from q_rlstc.data.rlstc_mdp import TrajRLclus
from q_rlstc.visualization.plot_utils import plot_paper_style_clusters

import math
import urllib.request
from PIL import Image
import io

def _deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def _num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def add_osm_background(ax, zoom=10):
    """Download and stitch OpenStreetMap tiles for the current ax limits."""
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    
    x0, y0 = _deg2num(lat_max, lon_min, zoom)
    x1, y1 = _deg2num(lat_min, lon_max, zoom)
    
    bounds_lat_max, bounds_lon_min = _num2deg(x0, y0, zoom)
    bounds_lat_min, bounds_lon_max = _num2deg(x1 + 1, y1 + 1, zoom)
    
    headers = {'User-Agent': 'q_rlstc_viz/1.0'}
    
    total_w = (x1 - x0 + 1) * 256
    total_h = (y1 - y0 + 1) * 256
    stitched = Image.new('RGB', (total_w, total_h))
    
    print(f"  Downloading {x1 - x0 + 1}x{y1 - y0 + 1} OSM background tiles...")
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    img_data = response.read()
                tile = Image.open(io.BytesIO(img_data))
                stitched.paste(tile, ((x - x0) * 256, (y - y0) * 256))
            except Exception as e:
                print(f"  Failed tile {x}/{y}: {e}")
                
    ax.imshow(stitched, extent=(bounds_lon_min, bounds_lon_max, bounds_lat_min, bounds_lat_max), 
              alpha=0.6, zorder=-10)

def main():
    base_dir = os.path.dirname(__file__)
    centroid_path = os.path.join(base_dir, 'geolife_rlstc_clusters')
    geo_subtraj_path = os.path.join(base_dir, 'geo-subtrajs')
    
    import random
    print("Loading sub-trajectories...")
    with open(geo_subtraj_path, 'rb') as f:
        all_subtrajs = pickle.load(f)
    print(f"Loaded {len(all_subtrajs)} sub-trajectories")
    
    # Sample 1000 subtrajectories for background to avoid massive overdrawing/memory issues
    trajectories = random.sample(all_subtrajs, min(1000, len(all_subtrajs)))
    print(f"Sampled {len(trajectories)} sub-trajectories for the background plot.")
    
    # Read the structurally-fixed native clusters
    result = pickle.load(open(centroid_path, 'rb'))
    quantum_segments = []
    classical_segments = []
    
    # K-Means stores array of lists. For every cluster:
    for c in result[0][2].values():
        # Extracted authentic structural spanning centroids (Quantum)
        if hasattr(c[1], 'points'):
            seg = [(p.x, p.y) for p in c[1].points]
            if len(seg) > 1:
                quantum_segments.append(seg)
                
        # Extract the original paper's fallback method (Classical)
        if len(c) > 3 and len(c[3]) > 0:
            if hasattr(c[3][0], 'points'):
                seg_c = [(p.x, p.y) for p in c[3][0].points]
                if len(seg_c) > 1:
                    classical_segments.append(seg_c)
                    
    print(f"Loaded {len(quantum_segments)} Quantum/Medoid paths and {len(classical_segments)} Classical paths.")
    
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
    
    # Filter bounds to match the dense cluster area shown in Figure 16
    lon_bounds = (115.7, 117.4)
    lat_bounds = (39.4, 40.9)
    
    # ── Generate both the standard and classical versions ──
    variants = [
        {
            "out_file": "recreated_quantum_geolife_fig16.png",
            "title": "Recreated Fig. 16: Geolife (Quantum VQ-DQN)",
            "rep_color": "red",
            "rep_label": "Representative trajectories",
            "segments": quantum_segments
        },
        {
            "out_file": "recreated_classical_geolife_fig16.png",
            "title": "Recreated Fig. 16: Geolife (Classical DQN)",
            "rep_color": "red",
            "rep_label": "Representative trajectories",
            "segments": classical_segments
        },
    ]
    
    for v in variants:
        print(f"Generating plot at {v['out_file']}...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use LineCollection for ultra-fast plotting of background trajectories
        segments = []
        for traj in trajectories:
            if not traj or not traj.points:
                continue
            seg = [(p.x, p.y) for p in traj.points]
            segments.append(seg)
        
        lc = LineCollection(segments, colors="blue", linewidths=0.3, alpha=0.4)
        ax.add_collection(lc)

        # Plot representative line sub-trajectories with Red LineCollection
        rep_lc = LineCollection(v["segments"], colors=v["rep_color"], linewidths=1.5, alpha=0.9)
        ax.add_collection(rep_lc)

        ax.set_xlim(*lon_bounds)
        ax.set_ylim(*lat_bounds)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(v["title"])
        
        add_osm_background(ax, zoom=11)
        
        # Remove top/right spines and set aspect correctly
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('auto')
        
        ax.legend(handles=[
            plt.Line2D([0], [0], color="blue", lw=1, label="Sub-trajectories"),
            plt.Line2D([0], [0], color=v["rep_color"], lw=3,
                       label=v["rep_label"]),
        ], loc="lower right", framealpha=0.9)
        
        fig.tight_layout()
        fig.savefig(v["out_file"], dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {v['out_file']}")
    
    print("Done!")

if __name__ == '__main__':
    main()
