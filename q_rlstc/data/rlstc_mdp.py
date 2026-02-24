import pickle
import numpy as np
from collections import defaultdict
from .rlstc_point import Point
from .rlstc_segment import Segment
from .rlstc_traj import Traj
from .rlstc_trajdistance import traj2trajIED
from .rlstc_cluster import incremental_mindist, add2clusdict, update_centers, compute_overdist


# ── Custom unpickler for RLSTCcode pickle files ──────────────────────
# Pickle stores the original module path (e.g. "traj.Traj").  We need
# to redirect those to q_rlstc.data.rlstc_* so unpickling works.
class _RLSTCUnpickler(pickle.Unpickler):
    _MODULE_MAP = {
        "traj": "q_rlstc.data.rlstc_traj",
        "point": "q_rlstc.data.rlstc_point",
        "segment": "q_rlstc.data.rlstc_segment",
        "point_xy": "q_rlstc.data.rlstc_point_xy",
    }

    def find_class(self, module, name):
        module = self._MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def _load_pickle(path):
    """Load an RLSTCcode pickle with module-name redirection."""
    with open(path, "rb") as f:
        return _RLSTCUnpickler(f).load()


class TrajRLclus():
    def __init__(self, cand_train, base_centers_T, base_centers_E,
                 min_seg_len=3, ablate_odb=False):
        self.n_actions = 2
        self.ablate_odb = ablate_odb
        self.n_features = 4 if ablate_odb else 5
        self.min_seg_len = max(1, min_seg_len)  # hard floor
        centers_T = _load_pickle(base_centers_T)
        centers_data_T = centers_T[0][2]
        k = len(centers_data_T) 
        self.RW = 0.0
        self.clusters_T = defaultdict(list) 
        self.clusters_E = defaultdict(list)

        for i in range(k):
            self.clusters_T[i].append([])  # [0] distances
            self.clusters_T[i].append([])  # [1] sub-trajectories
        for i in range(k):
            self.clusters_E[i].append([])  # [0] distances
            self.clusters_E[i].append([])  # [1] sub-trajectories
        self.allsubtrajs_T = []
        self.allsubtrajs_E = []
        self.allsubindexes_E = []
        self._load(cand_train, base_centers_T, base_centers_E)
        
    def _load(self,cand_train, base_centers_T, base_centers_E):
        cand_train_data = _load_pickle(cand_train)
        self.trajsdata = cand_train_data 
        centers_T = _load_pickle(base_centers_T)
        self.basesim_T = centers_T[0][1]
        centers_data_T = centers_T[0][2] 
        centers_E = _load_pickle(base_centers_E)
        self.basesim_E = centers_E[0][1]
        centers_data_E = centers_E[0][2]
        for i in range(len(centers_data_T)):
            self.clusters_T[i].append(centers_data_T[i][1].points)  # [2] center points
            self.clusters_T[i].append(defaultdict(list))             # [3] time-point dict
            self.clusters_T[i].append([])                            # [4] segment lengths
        for i in range(len(centers_data_E)):
            self.clusters_E[i].append(centers_data_E[i][1].points)  # [2] center points
            self.clusters_E[i].append(defaultdict(list))             # [3] time-point dict
            self.clusters_E[i].append([])                            # [4] segment lengths
    
    def reset(self, episode, label = 'T'):
        self.split_point = 0
        self.length = self.trajsdata[episode].size
        self.k_dict = dict() 
        k = len(self.clusters_T)
        for i in range(k):
            self.k_dict[i] = dict()
            self.k_dict[i]['mid_dist'] = 1e10
            self.k_dict[i]['real_dist'] = 1e10
            self.k_dict[i]['lastp'] = Point(0, 0, 0)
            self.k_dict[i]['j'] = 0
        self.traj_num = 1
        self.minsim = 0
        self.k = 0
        if label == 'T':
            self.minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, 1, self.k_dict, self.clusters_T, episode)
        else:
            self.minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, 1, self.k_dict, self.clusters_E, episode)
        self.next_minsim = self.minsim
        center_points = self.clusters_T[self.k][2]
        if label == 'E':
            center_points = self.clusters_E[self.k][2]
        self.overall_sim = traj2trajIED(self.trajsdata[episode].points, center_points) 
        self.split_overdist = self.minsim
        
        if self.ablate_odb:
            observation = np.array([self.overall_sim, self.minsim,
                                    2 / self.length,
                                    (self.length - 1) / self.length]).reshape(1, -1)
        else:
            observation = np.array([self.overall_sim, self.minsim, self.overall_sim*10,
                                    2 / self.length,
                                    (self.length - 1) / self.length]).reshape(1, -1)
       
        self.split = []  
        self.subtrajindex = []  
        self.subtraj = []
        self._seg_start_idx = 0  # tracks segment start for L_MIN
        return observation, self.length

    def step(self, episode, action, index, label='T'):
        # ── L_MIN enforcement: silently override CUT → EXTEND ──────
        if action == 1:
            seg_len = index - self._seg_start_idx + 1  # +1: inclusive
            remaining = self.length - index             # tail after cut
            if seg_len < self.min_seg_len or remaining < self.min_seg_len:
                action = 0  # force EXTEND
        self._last_action = action  # expose for callers

        if action == 0:
            if index + 1 != self.length: 
                if label == 'T':
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode],self.split_point, index+1, self.k_dict, self.clusters_T, episode)
                else:
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, index + 1,
                                                              self.k_dict, self.clusters_E, episode)
                if self.split_point == 0:
                    self.split_overdist = self.next_minsim / self.traj_num
                else:
                    self.split_overdist = (self.overall_sim * self.traj_num + self.next_minsim) / (self.traj_num + 1)
            else: 
                subtraj_points = self.trajsdata[episode].points[self.split_point: index+1]
                size = len(subtraj_points)
                ts, te = subtraj_points[0].t, subtraj_points[-1].t
                subtraj = Traj(subtraj_points,size,ts,te,self.trajsdata[episode].traj_id)
                self.subtrajindex.append([self.split_point, index])
                self.split.append(index)
                self.traj_num += 1
                if label == 'T':
                    self.clusters_T[self.k][1].append(subtraj)
                    if self.next_minsim != 1e10:
                        self.clusters_T[self.k][0].append(self.next_minsim)
                        self.clusters_T[self.k][4].append(subtraj.size)
                        add2clusdict(subtraj.points, self.clusters_T, self.k)
                if label == 'E':
                    self.clusters_E[self.k][1].append(subtraj)
                    if self.next_minsim != 1e10:
                        self.clusters_E[self.k][0].append(self.next_minsim)
                        self.clusters_E[self.k][4].append(subtraj.size)
                        add2clusdict(subtraj.points, self.clusters_E, self.k)
                self.overall_sim = self.split_overdist
           
            if self.ablate_odb:
                observation = np.array([self.overall_sim, self.split_overdist,
                                        (index - self.split_point + 2) / self.length,
                                        (self.length - (index + 1)) / self.length]).reshape(1, -1)
            else:
                observation = np.array([self.overall_sim, self.split_overdist, self.overall_sim*10,
                                        (index - self.split_point + 2) / self.length,
                                        (self.length - (index + 1)) / self.length]).reshape(1, -1)
            reward = 0
            return observation, reward

        if action == 1:
            self.split.append(index)
            self.minsim = self.next_minsim
            subtraj_points = self.trajsdata[episode].points[self.split_point: index + 1]
            size = len(subtraj_points)
            ts = subtraj_points[0].t
            te = subtraj_points[-1].t
            subtraj = Traj(subtraj_points, size, ts, te, self.trajsdata[episode].traj_id)
            if label == 'T':
                self.clusters_T[self.k][1].append(subtraj)
                if self.next_minsim != 1e10:
                    self.clusters_T[self.k][0].append(self.next_minsim)
                    self.clusters_T[self.k][4].append(subtraj.size)
                    add2clusdict(subtraj.points, self.clusters_T,self.k)
            if label == 'E':
                self.clusters_E[self.k][1].append(subtraj)
                if self.next_minsim !=  1e10:
                    self.clusters_E[self.k][0].append(self.next_minsim)
                    self.clusters_E[self.k][4].append(subtraj.size)
                    add2clusdict(subtraj.points,self.clusters_E,self.k)
            last_overall_sim = self.overall_sim
            if self.split_point != 0:
                self.traj_num += 1
            self.overall_sim = self.split_overdist
            self.split_point = index
            self._seg_start_idx = index  # new segment starts here
            if index + 1 != self.length:
                if label == 'T':
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode],self.split_point,index+1,self.k_dict,self.clusters_T, episode)
                else:
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, index + 1,
                                                              self.k_dict, self.clusters_E, episode)
                self.split_overdist = (self.overall_sim * self.traj_num + self.next_minsim) / (self.traj_num + 1)
           
            if self.ablate_odb:
                observation = np.array([self.overall_sim, self.split_overdist,
                                        (index - self.split_point + 2) / self.length,
                                        (self.length - (index + 1)) / self.length]).reshape(1, -1)
            else:
                observation = np.array([self.overall_sim, self.split_overdist, self.overall_sim*10,
                                        (index - self.split_point + 2) / self.length,
                                        (self.length - (index + 1)) / self.length]).reshape(1, -1)
    
            reward = last_overall_sim - self.overall_sim
            return observation, reward
    
    def output(self, label ='T'):
        if label == 'T':
            return [self.overall_sim, self.RW, self.clusters_T]
        if label == 'E':
            return [self.overall_sim, self.RW, self.clusters_E]
    
    def update_cluster(self, label='T'): 
        if label == 'T': 
            self.basesim_T, self.clusters_T = update_centers(self.clusters_T,3, 0.095)
            for i in self.clusters_T.keys():
                self.clusters_T[i][0] = []
                self.clusters_T[i][1] = []
                self.clusters_T[i][3] = defaultdict(list)
        if label == 'E':  
            self.basesim_E, self.clusters_E = update_centers(self.clusters_E, 3, 0.095)
            for i in self.clusters_E.keys():
                self.clusters_E[i][0] = []
                self.clusters_E[i][1] = []
                self.clusters_E[i][3] = defaultdict(list)
        