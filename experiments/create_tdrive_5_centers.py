import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import q_rlstc.data.rlstc_traj as rlstc_traj
import q_rlstc.data.rlstc_point as rlstc_point
import q_rlstc.data.rlstc_segment as rlstc_segment
import q_rlstc.data.rlstc_point_xy as rlstc_point_xy
sys.modules['traj'] = rlstc_traj
sys.modules['point'] = rlstc_point
sys.modules['segment'] = rlstc_segment
sys.modules['point_xy'] = rlstc_point_xy

# Import the logic from RLSTCcode-main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../RLSTCcode-main/subtrajcluster')))
from initcenters import getbaseclus, saveclus, initialize_centers

trajs = pickle.load(open('q_rlstc/data/Tdrive_testdata', 'rb'), encoding='bytes')
subtrajs = pickle.load(open('q_rlstc/data/ied_subtrajs_1000', 'rb'), encoding='bytes')

res = saveclus(5, subtrajs, trajs, 1000)
pickle.dump(res, open('q_rlstc/data/tdrive_clustercenter_5', 'wb'), protocol=2)
print("Created tdrive_clustercenter_5 with 5 clusters.")
