import os
import json
import argparse

class BenchConfig:
    def __init__(self, mode, backend, overrides=None):
        self.mode = mode
        self.backend = backend
        self._load_defaults()
        if overrides:
            self._apply_overrides(overrides)
            
    def _load_defaults(self):
        config_path = os.path.join(os.path.dirname(__file__), "config", "default_bench.json")
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
            
        if self.mode not in all_configs:
            raise ValueError(f"Unknown mode '{self.mode}'. Available modes: {list(all_configs.keys())}")
            
        cfg = all_configs[self.mode]
        self.tau = cfg.get("tau", 0.1)
        self.k = cfg.get("k", 10)
        self.preprocessing = cfg.get("preprocessing", False)
        self.metric = cfg.get("metric", "OD")
        self.distance = cfg.get("distance", "dIED")
        self.testdata = cfg.get("testdata", "../data/Tdrive_testdata")
        self.base_cluster = cfg.get("base_cluster", "../data/tdrive_clustercenter")
        self.modeldir = cfg.get("modeldir", "../savemodels/kfoldmodels2")
        self.amount = cfg.get("amount", 1000)

    def _apply_overrides(self, overrides):
        for k, v in overrides.items():
            if v is not None:
                setattr(self, k, v)

    def print_state(self):
        print("="*50)
        print(" EXPERIMENTAL TEST BENCH EXECUTION")
        print("="*50)
        print(f" Mode:                 {self.mode}")
        print(f" Backend:              {self.backend}")
        print(f" Dataset:              {self.testdata}")
        print(f" Cluster Center Path:  {self.base_cluster}")
        print(f" Model Directory:      {self.modeldir}")
        print(f" Parameter - K:        {self.k}")
        print(f" Convergence (tau):    {self.tau}")
        print(f" Preprocessing active: {self.preprocessing}")
        print(f" Metric reported:      {self.metric}")
        print(f" Distance objective:   {self.distance}")
        print(f" Amount:               {self.amount}")
        print("="*50)

def parse_args():
    parser = argparse.ArgumentParser(description="Quantum vs Classical Experimental Baseline Bench")
    parser.add_argument("--mode", type=str, required=True, choices=["paper_baseline", "repo_baseline"],
                        help="Baseline mode configuration to mimic.")
    parser.add_argument("--backend", type=str, required=True, choices=["classical", "quantum"],
                        help="Select the learning/model backend.")
    
    # Phase 3 Scaffoldings parameter sweep hooks
    parser.add_argument("--sweep-k", type=str, default=None, help="Phase 3: Scaffold for sweeping K values (e.g., '10,20,30')")
    parser.add_argument("--sweep-qubits", type=str, default=None, help="Phase 3: Scaffold for sweeping qubits")
    
    return parser.parse_args()
