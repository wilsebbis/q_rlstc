#!/usr/bin/env python3
"""Unified Thesis Experiment Runner for Q-RLSTC.

Runs every experiment, diagnostic, and comparison relevant to the thesis.
Prints all results to terminal AND saves to report files (.md + .json).

Experiments:
  D1  ValCR vs CUT% sweep (metric-only, no reward/learning)
  D2  Q-margin profiling  (Q_extend – Q_cut per epoch, all val steps)
  D3  Replay buffer distribution audit (actual buffer histogram)
  D4  Policy basin test    (forced extend/cut/random under drift)
  E1  Core Quantum Utility (VQ-DQN vs Controls A/B/C, noiseless)
  E2  NISQ Viability       (VQ-DQN, Eagle/Heron noise)
  E3  Shot Sensitivity     (VQ-DQN, 128/512/2048 shots)
  E4  Drift Resilience     (VQ-DQN vs Control B, drift mode)
  E5  Low-Data             (VQ-DQN vs Control B, 10%/25%/50%)
  E6  Version Progression  (Quantum A vs B vs D)
  S1  Scalability          (timing at 250-2500 trajectories)
  X1  Cross-validation     (K-fold OD stability)
  A1  OD_b Ablation        (full vs 4D observations)

Usage::

    # Full thesis run (~2-4 hours)
    python experiments/run_thesis_experiments.py

    # Quick diagnostic subset (~10 min)
    python experiments/run_thesis_experiments.py --experiments D1,D2,E1

    # Control data size for faster iteration
    python experiments/run_thesis_experiments.py --amount 100 --epochs 2

    # Regenerate plots from previous JSON
    python experiments/run_thesis_experiments.py --plots-only results/thesis/thesis_results.json
"""

import argparse
import io
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Training Mode: CONTROLLED_SPSA (thesis default) vs RLSTC_PARITY
# ═══════════════════════════════════════════════════════════════════════

class TrainingMode(Enum):
    """Training regime selector.

    CONTROLLED_SPSA: Thesis default — SPSA optimizer, reward shaping,
        Lagrangian CUT budget, anti-gaming constraints.
    RLSTC_PARITY: Reproduces RLSTC paper's training loop as closely as
        possible — SGD, γ=0.99, raw OD-delta reward, soft target τ=0.001,
        per-step ε decay. Enables direct comparison under RLSTC's own
        conditions.
    """
    CONTROLLED_SPSA = "controlled_spsa"
    RLSTC_PARITY = "rlstc_parity"


RLSTC_PARITY_PROTOCOL = {
    # ── Matched to RLSTC paper (sub-traj.pdf) ──
    "batch_size": 32,
    "memory_size": 5000,
    "gamma": 0.99,                      # RLSTC paper
    "huber_delta": 1.0,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.99,              # per-step (RLSTC: ε ← 0.99ε each step)
    "EPSILON_DECAY_MODE": "per_step",
    "EPSILON_DECAY_PER_STEP": 0.99,
    "target_update_freq": 1,            # soft update every episode
    "SOFT_TARGET_TAU": 0.001,           # ω from RLSTC paper (θ̂ ← ωθ̂ + (1−ω)θ)
    "USE_SOFT_TARGET": True,
    # ── Shaping OFF (pure OD delta reward) ──
    "L_MIN": 1,                         # no min-segment constraint
    "CUT_PENALTY": 0.0,
    "EXTEND_COST": 0.0,
    "COMPLEXITY_LAMBDA": 0.0,
    "MIN_CUT_BONUS": 0.0,
    "MIN_CUT_BONUS_FINAL": 0.0,
    "USE_LAGRANGIAN": False,
    "FORCED_CUT_PROB": 0.0,
    "FORCED_CUT_EPOCHS": 0,
    "EXPLORATION_MODE": "epsilon_greedy",
    "OPTIMISTIC_CUT_BIAS": 0.0,
    "USE_STRATIFIED_REPLAY": False,
    "REWARD_MODE": "raw_od_delta",      # r_t = OD_t − OD_{t+1}
    "Q_CLIP_RANGE": 50.0,
    "COLLAPSE_CUT_THRESHOLD": 0.0,      # no collapse detection in parity
}


def _collect_env_metadata() -> Dict[str, str]:
    """Collect git hash, python version, package versions for reproducibility."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
    }
    try:
        meta["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        meta["git_hash"] = "unknown"
    try:
        import qiskit
        meta["qiskit"] = qiskit.__version__
    except ImportError:
        meta["qiskit"] = "not installed"
    return meta

# ── Project path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "q_rlstc" / "data"
sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Tee printer — captures everything to a buffer AND prints to stdout
# ═══════════════════════════════════════════════════════════════════════

class TeePrinter:
    """Context manager that duplicates stdout to a StringIO buffer."""

    def __init__(self):
        self.buffer = io.StringIO()
        self._stdout = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout

    def write(self, text):
        self._stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self._stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  Protocol constants — identical for ALL models
# ═══════════════════════════════════════════════════════════════════════

PROTOCOL = {
    "batch_size": 32,
    "memory_size": 5000,
    "gamma": 0.90,
    "huber_delta": 1.0,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.99,
    "target_update_freq": 10,
    "L_MIN": 3,
    "CUT_PENALTY": 0.12,
    "EXTEND_COST": 0.01,
    "COMPLEXITY_LAMBDA": 0.03,
    "MIN_CUT_BONUS": 0.30,       # bonus for first CUT in episode (starts HIGH, anneals)
    "MIN_CUT_BONUS_FINAL": 0.15, # final MIN_CUT_BONUS after curriculum annealing
    "COLLAPSE_CUT_THRESHOLD": 1.0,  # CUT% below this → collapsed
    # Fix 2: Adaptive Lagrangian cut budget (R6 advisor overhaul)
    "USE_LAGRANGIAN": True,        # enable adaptive CUT penalty
    "TARGET_CUT_PCT": 10.0,        # b_soft: target CUT rate (%) for dual λ update
    "B_HARD_CUT_PCT": 30.0,        # b_hard: evaluation threshold — CUT% above this is flagged
    "LAGRANGIAN_LR": 0.02,         # dual variable learning rate
    "LAMBDA_MAX": 2.0,             # absolute cap to prevent runaway
    "LAMBDA_DELTA_MAX": 0.05,      # R6(B): max Δλ per epoch (damp oscillation)
    "LAMBDA_CUT_EMA": 0.9,         # R6(B): EMA decay for smoothed cut rate
    "LAMBDA_FREEZE_EPOCHS": 1,     # R6(C): freeze λ for first N epochs
    # Fix 4: Faster epsilon for small data
    "EPSILON_DECAY_MODE": "per_step",   # "per_episode" or "per_step"
    "EPSILON_DECAY_PER_STEP": 0.9995,   # ε≈0.1 after ~4600 steps
    # Fix 4b: Forced-cut curriculum (R6(E): stronger)
    "FORCED_CUT_PROB": 0.25,       # R6(E): probability of forced CUT (was 0.15)
    "FORCED_CUT_EPOCHS": 2,        # R6(E): force CUT in first N epochs (was 1)
    # Fix 2b: Action-stratified replay
    "USE_STRATIFIED_REPLAY": True,  # enable action-stratified replay
    "MIN_CUT_QUOTA": 0.3,           # minimum CUT fraction per batch
    # Fix 3: Boltzmann exploration
    "EXPLORATION_MODE": "boltzmann",   # "epsilon_greedy" or "boltzmann"
    "BOLTZMANN_TEMP_START": 1.0,
    "BOLTZMANN_TEMP_MIN": 0.1,
    "BOLTZMANN_TEMP_DECAY": 0.99,
    # Fix 4c: Optimistic CUT bias
    "OPTIMISTIC_CUT_BIAS": 0.5,
    # Fix 5: Q-clip range
    "Q_CLIP_RANGE": 50.0,
}


METRIC_DEFINITIONS = """
## Metric Definitions
- **ValCR (Validation Competitive Ratio)**: OD_segmented / OD_baseline.
  Numerator: RMS distance from each point to its segment centroid (post-segmentation).
  Denominator (bs): RMS distance under the always-extend baseline on the SAME
  validation fold. Lower is better. <1.0 means segmentation improves clustering.
  NOTE: This is NOT the standard online-algorithms competitive ratio.
  ⚠ CR values are ONLY comparable when computed with IDENTICAL bs.
- **GreedyCUT%**: Fraction of validation timesteps where the greedy policy
  (ε=0, no exploration) selects CUT (action=1) AFTER L_MIN override.
  Numerator: count of effective CUT actions in validation.
  Denominator: total validation timesteps (CUT + EXTEND).
- **TrainCUT%**: Fraction of training actions that were CUT, INCLUDING
  exploration (ε-greedy) actions. Higher early due to random exploration.
- **BufCUT%**: CUT action fraction IN THE REPLAY BUFFER. May lag behind
  current policy because buffer contains old experience.
- **#segs (total)**: Total segments across ALL validation trajectories.
  Each trajectory produces at least 1 segment (the terminal segment).
  #segs = Σ_episodes (cuts_in_episode + 1).
- **segs/traj**: Average segments per validation trajectory = #segs / n_val_trajectories.
- **Silhouette Coefficient**: Standard cluster validity metric [-1, 1].
  Higher is better. Measures intra-cluster vs inter-cluster distance.
- **SSE**: Sum of squared errors from cluster centroids.
- **Q-margin**: Mean(Q_extend - Q_cut) across validation steps. Positive
  means the agent prefers extending; negative means it prefers cutting.
- **OD (Overall Distance)**: RMS distance from each point to its segment centroid.
- **bs (basesim)**: OD under the always-extend (no-segmentation) baseline for the
  same validation fold. Used as the denominator of ValCR.
  ⚠ If bs differs between two runs, their CR values are on different scales.
- **L_MIN override**: When the agent selects CUT but the current segment is
  shorter than L_MIN (={L_MIN}) points, OR the remaining trajectory is shorter
  than L_MIN, the environment SILENTLY overrides CUT → EXTEND. The logged
  action is the POST-override (effective) action.
""".format(L_MIN=3)


# ═══════════════════════════════════════════════════════════════════════
#  Model specs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    name: str
    kind: str           # "quantum", "classical" (SPSA), "adam" (backprop), or "original"
    hidden_sizes: list = field(default_factory=list)
    n_layers: int = 3
    shots: int = 0      # 0 = statevector
    noise_model: str = "ideal"
    n_qubits: int = 5
    run_type: str = "standard"
    data_fraction: float = 1.0
    entanglement: str = "linear"  # 'linear', 'circular', 'full', 'none'
    version: str = "D"            # quantum circuit version: 'D' (standard), 'E' (Quantum B)
    training_mode: TrainingMode = TrainingMode.CONTROLLED_SPSA


def _get_protocol(spec: ModelSpec) -> Dict[str, Any]:
    """Return the active protocol dict based on training mode."""
    if spec.training_mode == TrainingMode.RLSTC_PARITY:
        return RLSTC_PARITY_PROTOCOL
    return PROTOCOL


def build_agent(spec: ModelSpec, seed: int):
    """Construct agent from spec (same as run_rigorous_benchmark)."""
    proto = _get_protocol(spec)
    if spec.kind == "quantum":
        from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig
        cfg = AgentConfig(
            version=spec.version,
            n_qubits=spec.n_qubits,
            n_layers=spec.n_layers,
            gamma=proto["gamma"],
            epsilon_start=proto["epsilon_start"],
            epsilon_min=proto["epsilon_min"],
            epsilon_decay=proto["epsilon_decay"],
            shots=spec.shots,
            target_update_freq=proto["target_update_freq"],
            entanglement=spec.entanglement,
            exploration_mode=proto.get("EXPLORATION_MODE", "epsilon_greedy"),
            boltzmann_temp=proto.get("BOLTZMANN_TEMP_START", 1.0),
            boltzmann_temp_min=proto.get("BOLTZMANN_TEMP_MIN", 0.1),
            boltzmann_temp_decay=proto.get("BOLTZMANN_TEMP_DECAY", 0.99),
            q_clip_range=proto.get("Q_CLIP_RANGE", 50.0),
            optimistic_cut_bias=proto.get("OPTIMISTIC_CUT_BIAS", 0.0),
        )
        backend = None
        if spec.noise_model != "ideal":
            from q_rlstc.quantum.backends import get_backend
            backend = get_backend(mode="noisy_sim", noise_model_name=spec.noise_model)
        return VQDQNAgent(config=cfg, backend=backend, seed=seed)
    elif spec.kind == "adam":
        from q_rlstc.rl.adam_classical_agent import AdamClassicalDQN, AdamAgentConfig
        cfg = AdamAgentConfig(
            hidden_sizes=spec.hidden_sizes,
            gamma=proto["gamma"],
            epsilon_start=proto["epsilon_start"],
            epsilon_min=proto["epsilon_min"],
            epsilon_decay=proto["epsilon_decay"],
            use_double_dqn=True,
            target_update_freq=proto["target_update_freq"],
            exploration_mode=proto.get("EXPLORATION_MODE", "epsilon_greedy"),
            boltzmann_temp=proto.get("BOLTZMANN_TEMP_START", 1.0),
            boltzmann_temp_min=proto.get("BOLTZMANN_TEMP_MIN", 0.1),
            boltzmann_temp_decay=proto.get("BOLTZMANN_TEMP_DECAY", 0.99),
            q_clip_range=proto.get("Q_CLIP_RANGE", 50.0),
            optimistic_cut_bias=proto.get("OPTIMISTIC_CUT_BIAS", 0.0),
        )
        return AdamClassicalDQN(config=cfg, seed=seed)
    elif spec.kind == "original":
        from q_rlstc.rl.original_classical_agent import OriginalClassicalDQN, OriginalAgentConfig
        cfg = OriginalAgentConfig(
            hidden_size=64,
            gamma=proto["gamma"],
            epsilon_start=proto["epsilon_start"],
            epsilon_min=proto["epsilon_min"],
            epsilon_decay=proto["epsilon_decay"],
        )
        return OriginalClassicalDQN(config=cfg, seed=seed)
    else:
        from q_rlstc.rl.spsa_classical_agent import SPSAClassicalDQN, ClassicalAgentConfig
        cfg = ClassicalAgentConfig(
            hidden_sizes=spec.hidden_sizes,
            gamma=proto["gamma"],
            epsilon_start=proto["epsilon_start"],
            epsilon_min=proto["epsilon_min"],
            epsilon_decay=proto["epsilon_decay"],
            use_double_dqn=True,
            target_update_freq=proto["target_update_freq"],
            exploration_mode=proto.get("EXPLORATION_MODE", "epsilon_greedy"),
            boltzmann_temp=proto.get("BOLTZMANN_TEMP_START", 1.0),
            boltzmann_temp_min=proto.get("BOLTZMANN_TEMP_MIN", 0.1),
            boltzmann_temp_decay=proto.get("BOLTZMANN_TEMP_DECAY", 0.99),
            q_clip_range=proto.get("Q_CLIP_RANGE", 50.0),
            optimistic_cut_bias=proto.get("OPTIMISTIC_CUT_BIAS", 0.0),
        )
        return SPSAClassicalDQN(config=cfg, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
#  Unified training loop (with diagnostic hooks)
# ═══════════════════════════════════════════════════════════════════════

def compute_fold_basesim(env, sidx, eidx):
    """Compute baseline OD (always-extend) for a specific validation fold."""
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from collections import defaultdict
    for i in env.clusters_E.keys():
        env.clusters_E[i][0] = []
        env.clusters_E[i][1] = []
        env.clusters_E[i][3] = defaultdict(list)
    for e in range(sidx, eidx):
        obs, steps = env.reset(e, "E")
        for idx in range(1, steps):
            env.step(e, 0, idx, "E")
    try:
        fold_od = float(compute_overdist(env.clusters_E))
    except (ZeroDivisionError, ValueError):
        fold_od = 1.0
    for i in env.clusters_E.keys():
        env.clusters_E[i][0] = []
        env.clusters_E[i][1] = []
        env.clusters_E[i][3] = defaultdict(list)
    return fold_od


def train_and_evaluate(
    agent,
    spec: ModelSpec,
    traj_path: str,
    centers_path: str,
    n_trajectories: int,
    n_epochs: int,
    seed: int,
) -> Dict[str, Any]:
    """Run training + evaluation for one model with diagnostic instrumentation."""
    proto = _get_protocol(spec)
    is_parity = spec.training_mode == TrainingMode.RLSTC_PARITY

    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import (
        compute_overdist, compute_sse, compute_overdist_length_weighted
    )
    from q_rlstc.rl.replay_buffer import ReplayBuffer
    from q_rlstc.data.trajectory_scheduler import TrajectoryScheduler

    np.random.seed(seed)
    random.seed(seed)

    validation_pct = 0.1
    scheduler = TrajectoryScheduler(
        n_trajectories=n_trajectories,
        validation_pct=validation_pct,
        mode=spec.run_type,
        data_fraction=spec.data_fraction,
        seed=seed,
    )
    sidx, eidx = scheduler.validation_range()

    env = TrajRLclus(traj_path, centers_path, centers_path)
    fold_basesim = compute_fold_basesim(env, sidx, eidx)
    replay = ReplayBuffer(max_size=PROTOCOL["memory_size"], seed=seed)

    _ied_scale = max(env.basesim_T, 1e-8)

    def normalize_obs(obs):
        o = obs.copy().flatten()
        o[0] /= _ied_scale
        o[1] /= _ied_scale
        o[2] /= _ied_scale * 10
        return o.reshape(obs.shape)

    def scale_reward(r):
        return float(np.clip(r / _ied_scale, -1.0, 1.0))

    L_MIN = proto["L_MIN"]
    CUT_PENALTY = proto["CUT_PENALTY"]
    EXTEND_COST = proto["EXTEND_COST"]
    COMPLEXITY_LAMBDA = proto["COMPLEXITY_LAMBDA"]
    MIN_CUT_BONUS = proto["MIN_CUT_BONUS"]
    batch_size = proto["batch_size"]

    # RLSTC Parity: soft target update configuration
    use_soft_target = proto.get("USE_SOFT_TARGET", False)
    soft_target_tau = proto.get("SOFT_TARGET_TAU", 0.001)
    reward_mode = proto.get("REWARD_MODE", "shaped")  # "shaped" or "raw_od_delta"

    # Fix 2: Adaptive Lagrangian cut penalty (R6 advisor overhaul)
    use_lagrangian = proto.get("USE_LAGRANGIAN", False)
    lagrangian_lambda = CUT_PENALTY  # initialize from static value
    target_cut_pct = proto.get("TARGET_CUT_PCT", 10.0)
    lagrangian_lr = proto.get("LAGRANGIAN_LR", 0.02)
    lambda_max = proto.get("LAMBDA_MAX", 2.0)
    lambda_delta_max = proto.get("LAMBDA_DELTA_MAX", 0.05)  # R6(B): Δλ clamp
    lambda_cut_ema_decay = proto.get("LAMBDA_CUT_EMA", 0.9) # R6(B): EMA
    lambda_freeze_epochs = proto.get("LAMBDA_FREEZE_EPOCHS", 1)  # R6(C)
    lagrangian_history = []  # track λ over epochs
    cut_rate_ema = None  # R6(B): EMA-smoothed cut rate (init from first epoch)
    
    # R6(D): MIN_CUT_BONUS curriculum — start high, anneal to final
    min_cut_bonus_start = proto.get("MIN_CUT_BONUS", 0.30)
    min_cut_bonus_final = proto.get("MIN_CUT_BONUS_FINAL", 0.15)
    current_min_cut_bonus = min_cut_bonus_start  # will be annealed per epoch

    # Fix 2b: Action-stratified replay
    use_stratified = proto.get("USE_STRATIFIED_REPLAY", True)
    min_cut_quota = proto.get("MIN_CUT_QUOTA", 0.3)

    # Fix 4b: Forced-cut curriculum
    forced_cut_prob = proto.get("FORCED_CUT_PROB", 0.15)
    forced_cut_epochs = proto.get("FORCED_CUT_EPOCHS", 1)

    # Fix 4: Epsilon decay mode
    eps_mode = proto.get("EPSILON_DECAY_MODE", "per_episode")
    eps_per_step = proto.get("EPSILON_DECAY_PER_STEP", 0.9995)

    # Tracking
    all_rewards = []
    val_crs, val_cut_pcts, val_seg_counts = [], [], []
    val_ods = []            # Raw OD per epoch (numerator of CR)
    val_basesims = []       # basesim per epoch (denominator of CR)
    val_cr_medians = []     # Median per-trajectory CR per epoch
    val_sses = []           # SSE per epoch
    val_silhouettes = []    # Silhouette coefficient per epoch
    val_wvalcrs = []        # Length-weighted ValCR per epoch (fragmentation-robust)
    q_margins = []          # D2: mean(Q_extend - Q_cut) per epoch (all val steps)
    replay_cut_pcts = []    # D3: CUT ratio in training actions per epoch
    replay_buf_cut_pcts = []  # D5: CUT ratio in actual replay buffer
    # Fix 6: ΔQ distribution logging
    delta_q_train_stats = []  # per-epoch ΔQ stats from training
    delta_q_val_stats = []    # per-epoch ΔQ stats from validation
    # R6(F): per-epoch reward component tracking
    r_cut_means = []     # mean immediate reward when action=CUT
    r_extend_means = []  # mean immediate reward when action=EXTEND
    best_bundle = {"val_cr": float("inf"), "cut_pct": 0.0,
                   "n_segs": 0, "epoch": -1, "avg_reward": 0.0, "sse": 0.0,
                   "silhouette": 0.0}
    # Sample efficiency: total episodes and actions processed
    total_episodes = 0
    total_actions = 0
    # Timing breakdown
    time_env = 0.0    # environment step time
    time_agent = 0.0  # agent act + update time
    has_q = hasattr(agent, 'get_q_values')  # check once, used in train + eval

    start_time = time.time()

    print(f"\n{'─'*70}")
    # Determine optimizer name for legend
    opt_name = {"quantum": "SPSA", "classical": "SPSA",
                "adam": "Adam", "original": "SGD"}.get(spec.kind, spec.kind)
    mode_str = f" [{spec.training_mode.value}]" if is_parity else ""
    q_info = ""
    if spec.kind == "quantum":
        q_info = (f" | {spec.n_qubits}q×{spec.n_layers}L "
                  f"{spec.entanglement} | {spec.noise_model} | shots={spec.shots}")
    legend = (f"{spec.name} — {opt_name} — {agent.n_params}p"
              f" — {n_trajectories}traj/{n_epochs}ep/seed={seed}{q_info}{mode_str}")
    print(f"  {legend}")
    if is_parity:
        print(f"  ⚙ RLSTC PARITY MODE — γ={proto['gamma']}, "
              f"ε_decay={eps_per_step}/step, "
              f"soft_τ={soft_target_tau}, reward=raw_OD_delta")
    else:
        cut_pen_str = 'λ-adaptive' if use_lagrangian else f'{CUT_PENALTY:.4f}'
        print(f"  Reward weights: CUT_PEN={cut_pen_str}"
              f" EXTEND_COST={EXTEND_COST} MIN_CUT_BONUS={MIN_CUT_BONUS}"
              f" COMPLEXITY_λ={COMPLEXITY_LAMBDA}")
        if use_lagrangian:
            print(f"  Lagrangian: target={target_cut_pct:.0f}% lr={lagrangian_lr} λ_init={CUT_PENALTY}")
    print(f"  Epsilon: mode={eps_mode}"
          f" {'decay/step='+str(eps_per_step) if eps_mode == 'per_step' else 'decay/ep='+str(proto['epsilon_decay'])}")
    print(f"  bs (fold baseline OD) = {fold_basesim:.6f}")
    print(f"  Epochs: {n_epochs} × {scheduler.active_training_size} trajectories"
          f" | val={eidx-sidx} trajectories")
    print(f"{'─'*70}")

    for epoch in range(n_epochs):
        idxlist = scheduler.sample_epoch()
        epoch_rewards = []
        epoch_cuts_in_training = 0
        epoch_extends_in_training = 0
        epoch_forced_cuts = 0
        epoch_delta_q_train = []  # Fix 6: ΔQ per training step
        epoch_r_cuts = []   # R6(F): immediate rewards on CUT actions
        epoch_r_extends = []  # R6(F): immediate rewards on EXTEND actions
        epoch_start = time.time()
        _last_tick = epoch_start

        for ep_idx, episode in enumerate(idxlist):
            # Progress tick — print every 30s so long silences are visible
            now = time.time()
            if now - _last_tick >= 30:
                elapsed_total = now - start_time
                elapsed_epoch = now - epoch_start
                print(f"    ⏱ {elapsed_total:.0f}s total | "
                      f"epoch {epoch+1}/{n_epochs} | "
                      f"episode {ep_idx+1}/{len(idxlist)} | "
                      f"{elapsed_epoch:.0f}s this epoch",
                      flush=True)
                _last_tick = now
            observation, steps = env.reset(episode, "T")
            raw_split_od = observation.flatten()[1]
            observation = normalize_obs(observation)
            episode_reward = 0.0
            n_cuts, n_steps = 0, 0
            total_episodes += 1

            for index in range(1, steps):
                done = (index == steps - 1)
                t_act = time.time()

                # Fix 6: Log ΔQ during training
                if has_q:
                    _tq = agent.get_q_values(observation.flatten())
                    epoch_delta_q_train.append(float(_tq[1] - _tq[0]))  # Q_cut - Q_ext

                action = agent.act(observation)

                # Fix 4b: Forced-cut curriculum
                if (epoch < forced_cut_epochs
                    and n_cuts == 0
                    and n_steps >= L_MIN
                    and np.random.random() < forced_cut_prob):
                    action = 1  # force CUT
                    epoch_forced_cuts += 1

                time_agent += time.time() - t_act

                t_env = time.time()
                observation_, reward = env.step(episode, action, index, "T")
                time_env += time.time() - t_env
                actual_action = env._last_action
                raw_split_od_next = observation_.flatten()[1]
                total_actions += 1

                if actual_action == 0 and reward == 0:
                    reward = raw_split_od - raw_split_od_next

                reward = scale_reward(reward)

                if reward_mode == "raw_od_delta":
                    # RLSTC Parity: pure OD delta, no shaping
                    if actual_action == 0:
                        epoch_extends_in_training += 1
                        epoch_r_extends.append(float(reward))
                    if actual_action == 1:
                        n_cuts += 1
                        epoch_cuts_in_training += 1
                        epoch_r_cuts.append(float(reward))
                else:
                    # Use adaptive λ if Lagrangian enabled, else static penalty
                    active_cut_pen = lagrangian_lambda if use_lagrangian else CUT_PENALTY

                    if actual_action == 0:
                        reward -= EXTEND_COST
                        epoch_extends_in_training += 1
                        epoch_r_extends.append(float(reward))
                    if actual_action == 1:
                        # R6(D): Anti-collapse bonus with curriculum annealing
                        if n_cuts == 0:
                            reward += current_min_cut_bonus
                        reward -= active_cut_pen
                        n_cuts += 1
                        epoch_cuts_in_training += 1
                        epoch_r_cuts.append(float(reward))
                n_steps += 1

                raw_split_od = raw_split_od_next
                observation_ = normalize_obs(observation_)

                episode_reward += reward
                replay.add(observation.flatten(), actual_action, reward,
                           observation_.flatten(), done)

                if done:
                    break

                if replay.is_ready(batch_size):
                    # Fix 2b: Action-stratified replay
                    if use_stratified:
                        states, actions, rewards_b, next_states, dones = \
                            replay.sample_batch_stratified(batch_size, min_cut_quota)
                    else:
                        states, actions, rewards_b, next_states, dones = \
                            replay.sample_batch(batch_size)
                    t_upd = time.time()
                    agent.update(states, actions, rewards_b, next_states, dones)
                    time_agent += time.time() - t_upd

                # Fix 4: Per-step epsilon decay for small-data regimes
                if eps_mode == "per_step":
                    agent.epsilon = max(
                        agent.config.epsilon_min,
                        agent.epsilon * eps_per_step
                    )

                observation = observation_

            # Complexity regularizer
            if n_steps > 0:
                cut_rate = n_cuts / n_steps
                episode_reward -= COMPLEXITY_LAMBDA * cut_rate

            all_rewards.append(episode_reward)
            epoch_rewards.append(episode_reward)
            if eps_mode == "per_episode":
                agent.decay_epsilon()
            # RLSTC Parity: soft target update at end of each episode
            if use_soft_target and hasattr(agent, 'soft_update'):
                agent.soft_update(tau=soft_target_tau)

        # ── End-of-epoch validation (SINGLE PASS — fixed) ──────
        # Collects: cut/extend counts, per-episode segs, Q-margins
        env.allsubtrajs_E = []
        val_n_extend, val_n_cut = 0, 0
        val_segs = 0
        q_extend_vals, q_cut_vals = [], []

        for e in range(sidx, eidx):
            obs, s = env.reset(e, "E")
            obs = normalize_obs(obs)
            ep_cuts = 0
            for idx in range(1, s):
                # D2: Q-margin — log on EVERY val step
                if has_q:
                    q_vals = agent.get_q_values(obs.flatten())
                    q_extend_vals.append(float(q_vals[0]))
                    q_cut_vals.append(float(q_vals[1]))

                act = agent.act(obs, greedy=True)
                obs, _ = env.step(e, act, idx, "E")
                actual = env._last_action
                if actual == 0:
                    val_n_extend += 1
                else:
                    val_n_cut += 1
                    ep_cuts += 1
                obs = normalize_obs(obs)
            val_segs += ep_cuts + 1  # segments = cuts + 1 per episode

        try:
            val_od = compute_overdist(env.clusters_E)
            val_basesim = fold_basesim
            val_cr = float(val_od / max(val_basesim, 1e-8))  # ε-stabilised
        except (ZeroDivisionError, ValueError):
            val_od = float('inf')
            val_basesim = 0.0
            val_cr = float('inf')
        val_crs.append(val_cr)
        val_ods.append(float(val_od))
        val_basesims.append(val_basesim)

        # Per-trajectory CR for median (robust to denominator blowup)
        per_traj_crs = []
        for cid in env.clusters_E.keys():
            dists = env.clusters_E[cid][0]
            if dists:
                traj_od = sum(dists) / len(dists)
                traj_cr = traj_od / max(val_basesim, 1e-8)
                per_traj_crs.append(traj_cr)
        val_cr_median = float(np.median(per_traj_crs)) if per_traj_crs else val_cr
        val_cr_medians.append(val_cr_median)

        # wValCR — length-weighted OD (robust to fragmentation attractor)
        try:
            val_wod = float(compute_overdist_length_weighted(env.clusters_E))
            val_wvalcr = val_wod / max(val_basesim, 1e-8)
        except Exception:
            val_wvalcr = float('inf')
        val_wvalcrs.append(val_wvalcr)

        try:
            val_sse = float(compute_sse(env.clusters_E))
        except (ZeroDivisionError, ValueError):
            val_sse = float('inf')
        val_sses.append(val_sse)

        # Silhouette coefficient — compute from segment data
        try:
            from q_rlstc.clustering.metrics import silhouette_score as sil_score
            seg_data = []
            seg_labels = []
            for cid, cdata in env.clusters_E.items():
                subtrajs = cdata[3] if len(cdata) > 3 else {}
                for tid, points in (subtrajs.items() if isinstance(subtrajs, dict) else []):
                    for pt in points:
                        seg_data.append(pt)
                        seg_labels.append(cid)
            if len(seg_data) >= 2 and len(set(seg_labels)) >= 2:
                seg_data_arr = np.array(seg_data)
                seg_labels_arr = np.array(seg_labels)
                val_sil = float(sil_score(seg_data_arr, seg_labels_arr))
            else:
                val_sil = 0.0
        except Exception:
            val_sil = 0.0
        val_silhouettes.append(val_sil)

        val_total = val_n_extend + val_n_cut
        cut_pct = 100 * val_n_cut / val_total if val_total else 0
        val_cut_pcts.append(cut_pct)
        val_seg_counts.append(val_segs)

        # D2: Q-margin (mean across ALL val steps this epoch)
        if q_extend_vals:
            margin = float(np.mean(q_extend_vals) - np.mean(q_cut_vals))
            q_margins.append(margin)
            # Fix 6: ΔQ distribution from validation (Q_cut - Q_ext)
            dq_val = np.array(q_cut_vals) - np.array(q_extend_vals)
            delta_q_val_stats.append({
                "mean": float(np.mean(dq_val)),
                "median": float(np.median(dq_val)),
                "std": float(np.std(dq_val)),
                "min": float(np.min(dq_val)),
                "max": float(np.max(dq_val)),
                "pct_positive": float(100 * np.mean(dq_val > 0)),
            })
        else:
            q_margins.append(0.0)
            delta_q_val_stats.append({})

        # Fix 6: ΔQ distribution from training
        if epoch_delta_q_train:
            dq_t = np.array(epoch_delta_q_train)
            delta_q_train_stats.append({
                "mean": float(np.mean(dq_t)),
                "median": float(np.median(dq_t)),
                "std": float(np.std(dq_t)),
                "min": float(np.min(dq_t)),
                "max": float(np.max(dq_t)),
                "pct_positive": float(100 * np.mean(dq_t > 0)),
            })
        else:
            delta_q_train_stats.append({})

        # D3: Replay distribution — training actions this epoch
        total_training_actions = epoch_cuts_in_training + epoch_extends_in_training
        rp_cut_pct = (100 * epoch_cuts_in_training / total_training_actions
                      if total_training_actions else 0)
        replay_cut_pcts.append(rp_cut_pct)

        # D5: Actual replay buffer histogram (sample from buffer)
        buf_cut_pct = 0.0
        if hasattr(replay, 'buffer') and len(replay.buffer) > 0:
            buf_actions = [t[1] for t in replay.buffer]
            buf_cut_pct = 100 * sum(buf_actions) / len(buf_actions)
        elif hasattr(replay, 'actions') and len(replay.actions) > 0:
            buf_cut_pct = 100 * np.mean(replay.actions)
        replay_buf_cut_pcts.append(buf_cut_pct)

        # Best-epoch bundle
        improved = ""
        if val_cr < best_bundle["val_cr"]:
            improved = " ★"
            best_bundle = {
                "val_cr": val_cr, "cut_pct": cut_pct,
                "val_od": float(val_od), "val_basesim": val_basesim,
                "val_cr_median": val_cr_median,
                "val_wvalcr": val_wvalcr,
                "n_segs": val_segs, "epoch": epoch + 1,
                "avg_reward": float(np.mean(epoch_rewards)),
                "sse": val_sse,
                "silhouette": val_sil,
                "episodes_to_best": total_episodes,
                "actions_to_best": total_actions,
            }

        # Q diagnostic: use first real val observation (not arbitrary)
        q_diag = ""
        if has_q and q_extend_vals:
            q_diag = (f" | Q̄ext={np.mean(q_extend_vals):+.3f}"
                      f" Q̄cut={np.mean(q_cut_vals):+.3f}")

        # Compute segments per trajectory for clarity
        n_val_episodes = eidx - sidx
        segs_per_traj = val_segs / max(n_val_episodes, 1)

        # Fix 6: ΔQ inline stats
        dq_val_str = ""
        if delta_q_val_stats[-1]:
            ds = delta_q_val_stats[-1]
            dq_val_str = f" | ΔQval={ds['mean']:+.3f}±{ds['std']:.3f}({ds['pct_positive']:.0f}%>0)"
        dq_train_str = ""
        if delta_q_train_stats[-1]:
            ds = delta_q_train_stats[-1]
            dq_train_str = f" | ΔQtrn={ds['mean']:+.3f}±{ds['std']:.3f}"
        forced_str = f" | forced={epoch_forced_cuts}" if epoch_forced_cuts > 0 else ""

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: "
              f"ValCR={val_cr:.4f} (OD={val_od:.4f}/bs={val_basesim:.4f}) "
              f"medCR={val_cr_median:.4f} wCR={val_wvalcr:.4f} | "
              f"Sil={val_sil:+.3f} | "
              f"R̄={np.mean(epoch_rewards):+.3f} | "
              f"GreedyCUT={cut_pct:.0f}% | "
              f"#segs={val_segs}({segs_per_traj:.1f}/traj) | "
              f"ε={agent.epsilon:.3f} | "
              f"Qmargin={q_margins[-1]:+.4f} | "
              f"TrainCUT={rp_cut_pct:.0f}% | "
              f"BufCUT={buf_cut_pct:.0f}%{improved}{q_diag}"
              f"{dq_val_str}{dq_train_str}{forced_str}")

        # Reset eval clusters
        for i in env.clusters_E.keys():
            env.clusters_E[i][0] = []
            env.clusters_E[i][1] = []
            env.clusters_E[i][3] = defaultdict(list)

        env.update_cluster("T")
        scheduler.update()

        # R6(F): Log reward component breakdown
        r_cut_mean = float(np.mean(epoch_r_cuts)) if epoch_r_cuts else 0.0
        r_ext_mean = float(np.mean(epoch_r_extends)) if epoch_r_extends else 0.0
        r_cut_means.append(r_cut_mean)
        r_extend_means.append(r_ext_mean)
        print(f"    R6 reward components: r̄_cut={r_cut_mean:+.4f} r̄_ext={r_ext_mean:+.4f}"
              f" (Δ={r_cut_mean - r_ext_mean:+.4f}, "
              f"bonus={current_min_cut_bonus:.3f})")

        # R6(D): Anneal MIN_CUT_BONUS — linear schedule from start→final
        if n_epochs > 1:
            progress = min(1.0, (epoch + 1) / n_epochs)
            current_min_cut_bonus = (min_cut_bonus_start
                                      + progress * (min_cut_bonus_final - min_cut_bonus_start))

        # R6: Overhauled λ controller — advisor-identified root cause fix
        # Key principle: λ must track the signal being optimized (batch CUT rate),
        # NOT greedy evaluation CUT rate. Greedy is logged as diagnostic only.
        if use_lagrangian:
            epoch_total = epoch_cuts_in_training + epoch_extends_in_training
            batch_cut_rate = (100.0 * epoch_cuts_in_training / max(epoch_total, 1))
            greedy_rate = cut_pct  # diagnostic only — NOT used for λ

            # R6(B): EMA-smooth the batch CUT rate to reduce noise
            if cut_rate_ema is None:
                cut_rate_ema = batch_cut_rate  # initialize from first observation
            else:
                cut_rate_ema = (lambda_cut_ema_decay * cut_rate_ema
                                + (1 - lambda_cut_ema_decay) * batch_cut_rate)

            lambda_before = lagrangian_lambda

            # R6(C): Freeze λ for first N epochs — let policy learn basic cut behavior
            if epoch < lambda_freeze_epochs:
                delta_lambda = 0.0
                freeze_status = "FROZEN"
            else:
                # Compute raw Δλ from EMA-smoothed batch rate
                raw_delta = lagrangian_lr * (cut_rate_ema - target_cut_pct)
                # R6(B): Clamp Δλ to prevent wild swings (±0.05 default)
                delta_lambda = max(-lambda_delta_max, min(lambda_delta_max, raw_delta))
                freeze_status = "ACTIVE"

            lagrangian_lambda = max(0.0, min(lambda_max,
                                             lagrangian_lambda + delta_lambda))
            lagrangian_history.append(lagrangian_lambda)
            print(f"    λ [{freeze_status}]: {lambda_before:.4f} → {lagrangian_lambda:.4f} | "
                  f"batch={batch_cut_rate:.1f}% ema={cut_rate_ema:.1f}% "
                  f"greedy={greedy_rate:.1f}%(diag) "
                  f"target={target_cut_pct:.0f}% Δλ={delta_lambda:+.4f}"
                  f" (raw={raw_delta:+.4f} clamped)" if epoch >= lambda_freeze_epochs
                  else f"    λ [{freeze_status}]: {lambda_before:.4f} (held) | "
                       f"batch={batch_cut_rate:.1f}% greedy={greedy_rate:.1f}%(diag)")

    elapsed = time.time() - start_time

    # Collapse detection: if best-epoch CUT% below threshold → collapsed
    collapse_thresh = proto.get("COLLAPSE_CUT_THRESHOLD", 1.0)
    is_collapsed = best_bundle["cut_pct"] < collapse_thresh
    if is_collapsed and not is_parity:
        print(f"  ⚠ COLLAPSED: CUT%={best_bundle['cut_pct']:.1f}% < "
              f"{collapse_thresh}% — policy never learned to cut")

    # Budget violation flag (CMDP b_hard threshold)
    b_hard = proto.get("B_HARD_CUT_PCT", 30.0)
    budget_violated = best_bundle["cut_pct"] > b_hard if b_hard > 0 else False

    return {
        "model": spec.name,
        "kind": spec.kind,
        "noise": spec.noise_model,
        "params": agent.n_params,
        "run_type": spec.run_type,
        "data_fraction": spec.data_fraction,
        "collapsed": is_collapsed,
        "budget_violated": budget_violated,
        # Configuration metadata (advisor item #5)
        "config": {
            "n_qubits": spec.n_qubits,
            "n_layers": spec.n_layers,
            "shots": spec.shots,
            "entanglement": spec.entanglement,
            "noise_model": spec.noise_model,
            "version": getattr(agent, 'version', 'N/A'),
            "training_mode": spec.training_mode.value,
            "protocol": dict(proto),
            "env_metadata": _collect_env_metadata(),
        },
        # Best-epoch bundle
        "val_cr": best_bundle["val_cr"],
        "val_od": best_bundle.get("val_od", 0.0),
        "val_basesim": best_bundle.get("val_basesim", 0.0),
        "val_cr_median": best_bundle.get("val_cr_median", 0.0),
        "val_wvalcr": best_bundle.get("val_wvalcr", 0.0),
        "cut_pct": best_bundle["cut_pct"],  # GreedyCUT% (validation, post-L_MIN override)
        "n_segs": best_bundle["n_segs"],    # total segments across all val trajectories
        "n_val_episodes": eidx - sidx,       # number of validation trajectories
        "segs_per_traj": best_bundle["n_segs"] / max(eidx - sidx, 1),
        "best_epoch": best_bundle["epoch"],
        "sse": best_bundle.get("sse", 0.0),
        "silhouette": best_bundle.get("silhouette", 0.0),
        # Sample efficiency
        "episodes_to_best": best_bundle.get("episodes_to_best", total_episodes),
        "actions_to_best": best_bundle.get("actions_to_best", total_actions),
        "total_episodes": total_episodes,
        "total_actions": total_actions,
        # Final-epoch metrics
        "final_val_cr": val_crs[-1] if val_crs else float('inf'),
        "final_cut_pct": val_cut_pcts[-1] if val_cut_pcts else 0.0,
        "final_n_segs": val_seg_counts[-1] if val_seg_counts else 0,
        "final_sse": val_sses[-1] if val_sses else 0.0,
        "final_silhouette": val_silhouettes[-1] if val_silhouettes else 0.0,
        # Per-epoch series
        "val_crs": val_crs,
        "val_ods": val_ods,
        "val_basesims": val_basesims,
        "val_cr_medians": val_cr_medians,
        "val_wvalcrs": val_wvalcrs,
        "val_cut_pcts": val_cut_pcts,
        "val_seg_counts": val_seg_counts,
        "val_sses": val_sses,
        "val_silhouettes": val_silhouettes,
        "q_margins": q_margins,
        "replay_cut_pcts": replay_cut_pcts,
        "replay_buf_cut_pcts": replay_buf_cut_pcts,
        "all_rewards": [float(r) for r in all_rewards],
        # Lagrangian tracking
        "lagrangian_history": [float(l) for l in lagrangian_history],
        "final_lagrangian_lambda": float(lagrangian_lambda) if use_lagrangian else None,
        # Fix 6: ΔQ stats
        "delta_q_train_stats": delta_q_train_stats,
        "delta_q_val_stats": delta_q_val_stats,
        # Timing breakdown
        "wall_time": elapsed,
        "time_env": time_env,
        "time_agent": time_agent,
        "time_overhead": elapsed - time_env - time_agent,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

def get_e1_specs():
    """E1 — Core Quantum Utility (all models, noiseless)."""
    return [
        ModelSpec("VQ-DQN (5q×3L)",     "quantum", n_layers=3),
        # Parameter-matched baselines (34 params — same as VQ-DQN)
        ModelSpec("MLP-34 (SPSA)",       "classical", hidden_sizes=[4]),
        ModelSpec("MLP-34 (Adam)",        "adam", hidden_sizes=[4]),
        # SPSA-optimized controls (same optimizer as quantum)
        ModelSpec("Control A (linear)",  "classical", hidden_sizes=[]),
        ModelSpec("Control B (h=64)",    "classical", hidden_sizes=[64]),
        ModelSpec("Control C (h=32×32)", "classical", hidden_sizes=[32, 32]),
        # Adam-optimized controls (backprop — removes SPSA handicap objection)
        ModelSpec("Control D (Adam linear)",  "adam", hidden_sizes=[]),
        ModelSpec("Control E (Adam h=64)",    "adam", hidden_sizes=[64]),
        ModelSpec("Control F (Adam h=32×32)", "adam", hidden_sizes=[32, 32]),
        # Original RLSTC baseline (advisor item #10 — segmentation stats)
        ModelSpec("Original RLSTC (h=64)", "original"),
    ]

def get_e2_specs():
    """E2 — NISQ Viability (noise profiles)."""
    return [
        ModelSpec("VQ-DQN (Eagle)",  "quantum", noise_model="eagle",  shots=1024),
        ModelSpec("VQ-DQN (Heron)",  "quantum", noise_model="heron",  shots=1024),
    ]

def get_e3_specs():
    """E3 — Shot Sensitivity."""
    return [
        ModelSpec("VQ-DQN (128)",   "quantum", shots=128),
        ModelSpec("VQ-DQN (512)",   "quantum", shots=512),
        ModelSpec("VQ-DQN (2048)",  "quantum", shots=2048),
    ]

def get_e4_specs():
    """E4 — Drift Resilience."""
    return [
        ModelSpec("VQ-DQN (drift)",    "quantum", run_type="drift"),
        ModelSpec("Control B (drift)", "classical", hidden_sizes=[64], run_type="drift"),
    ]

def get_e5_specs():
    """E5 — Low-Data Generalization."""
    specs = []
    for frac in [0.10, 0.25, 0.50]:
        specs.append(ModelSpec(f"VQ-DQN ({frac:.0%})", "quantum",
                               run_type="low_data", data_fraction=frac))
        specs.append(ModelSpec(f"Control B ({frac:.0%})", "classical",
                               hidden_sizes=[64], run_type="low_data",
                               data_fraction=frac))
    return specs

def get_e6_specs():
    """E6 — Version Progression."""
    return [
        ModelSpec("VQ-DQN D (5q×3L)", "quantum", n_qubits=5, n_layers=3),
    ]


def get_e7_specs():
    """E7 — Configuration Sweep (advisor item #1: model capacity study).

    Varies qubits, layers, and ansatz type to study expressivity vs
    barren plateau tradeoff. All combinations use shots=0 (statevector).
    """
    specs = []
    for n_qubits in [4, 5, 6, 7, 8]:
        for n_layers in [2, 3, 4, 5]:
            for entanglement in ["linear", "circular", "full"]:
                name = f"VQ-DQN ({n_qubits}q×{n_layers}L {entanglement})"
                specs.append(ModelSpec(
                    name, "quantum",
                    n_qubits=n_qubits, n_layers=n_layers,
                    entanglement=entanglement,
                ))
    return specs


def get_e8_specs(n_traj: int):
    """E8 — Data Scaling (Fix 6: sample efficiency comparison).

    Compares VQ-DQN vs best classical at a specific trajectory count.
    Called multiple times with different n_traj values.
    """
    return [
        ModelSpec(f"VQ-DQN ({n_traj}t)",        "quantum",   n_layers=3),
        ModelSpec(f"Original RLSTC ({n_traj}t)", "original"),
        ModelSpec(f"Control B ({n_traj}t)",      "classical", hidden_sizes=[64]),
        ModelSpec(f"MLP-34 Adam ({n_traj}t)",    "adam",      hidden_sizes=[4]),
    ]


def get_ablation_entanglement_specs():
    """Ablation: entanglement matters? (no-CNOT vs linear CNOT)."""
    return [
        ModelSpec("VQ-DQN (no-CNOT)", "quantum", n_layers=3, entanglement="none"),
        ModelSpec("VQ-DQN (linear)",  "quantum", n_layers=3, entanglement="linear"),
    ]


def get_parity_specs():
    """PARITY — RLSTC Parity Comparison.

    Runs VQ-DQN, parameter-matched classical MLP, and Original RLSTC under
    RLSTC's own training conditions (γ=0.99, per-step ε, SGD, soft τ=0.001,
    raw OD-delta reward). Paired with the same models under thesis CONTROLLED_SPSA
    conditions for direct A/B comparison.
    """
    parity = TrainingMode.RLSTC_PARITY
    thesis = TrainingMode.CONTROLLED_SPSA
    return [
        # ── Under RLSTC parity conditions ──
        ModelSpec("VQ-DQN (parity)",     "quantum",   n_layers=3,
                  training_mode=parity),
        ModelSpec("MLP-34 SPSA (parity)","classical",  hidden_sizes=[4],
                  training_mode=parity),
        ModelSpec("RLSTC (parity)",      "original",
                  training_mode=parity),
        # ── Under thesis conditions (controlled comparison) ──
        ModelSpec("VQ-DQN (thesis)",     "quantum",   n_layers=3,
                  training_mode=thesis),
        ModelSpec("MLP-34 SPSA (thesis)","classical",  hidden_sizes=[4],
                  training_mode=thesis),
        ModelSpec("RLSTC (thesis)",      "original",
                  training_mode=thesis),
    ]


def get_e9_specs():
    """E9 — Quantum B: Best quantum with all quantum-specific optimizations.

    Version E: learnable input scaling + anti-barren-plateau init +
    circular entanglement + multi-observable readout + data re-uploading.
    Compared against both Quantum A and best classical controls.
    """
    return [
        # Quantum B — everything turned to 11
        ModelSpec("VQ-DQN-B (5q×3L)",  "quantum", n_qubits=5, n_layers=3, version="E"),
        # Quantum A — fair fight baseline (same model, no quantum tricks)
        ModelSpec("VQ-DQN-A (5q×3L)",  "quantum", n_qubits=5, n_layers=3, version="D"),
        # Classical controls
        ModelSpec("MLP-34 (SPSA)",      "classical", hidden_sizes=[4]),
        ModelSpec("MLP-34 (Adam)",       "adam", hidden_sizes=[4]),
        ModelSpec("Control B (h=64)",   "classical", hidden_sizes=[64]),
    ]


# ═══════════════════════════════════════════════════════════════════════
#  RA1: Reward Ablation — naive vs shaped reward
# ═══════════════════════════════════════════════════════════════════════

def run_ra1_reward_ablation(traj_path, centers_path, n_traj, n_epochs, seed=42):
    """RA1 — Reward ablation: demonstrate CR degeneracy under naive reward.

    Trains VQ-DQN under two reward regimes:
      (a) SHAPED: with L_MIN, CUT_PENALTY, EXTEND_COST, COMPLEXITY_LAMBDA
      (b) NAIVE:  raw OD improvement only — no anti-gaming constraints

    If the degeneracy argument holds, the naive variant should converge to
    near-always-cut (CUT% → 100%) with artificially low ValCR.
    """
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist, compute_sse
    from q_rlstc.rl.replay_buffer import ReplayBuffer
    from q_rlstc.data.trajectory_scheduler import TrajectoryScheduler

    conditions = [
        ("VQ-DQN (shaped)",  PROTOCOL["L_MIN"], PROTOCOL["CUT_PENALTY"],
         PROTOCOL["EXTEND_COST"], PROTOCOL["COMPLEXITY_LAMBDA"]),
        ("VQ-DQN (naive)",   1, 0.0, 0.0, 0.0),
    ]

    print(f"\n{'═'*80}")
    print("  RA1: REWARD ABLATION — Naive vs Shaped Reward")
    print(f"  Demonstrates CR degeneracy under naive reward (no constraints)")
    print(f"{'═'*80}\n")

    results = []
    for cond_name, l_min, cut_pen, ext_cost, comp_lam in conditions:
        np.random.seed(seed)
        random.seed(seed)

        spec = ModelSpec(cond_name, "quantum", n_layers=3)
        agent = build_agent(spec, seed)

        scheduler = TrajectoryScheduler(
            n_trajectories=n_traj, validation_pct=0.1,
            mode="standard", seed=seed)
        sidx, eidx = scheduler.validation_range()

        env = TrajRLclus(traj_path, centers_path, centers_path)
        fold_basesim = compute_fold_basesim(env, sidx, eidx)
        replay = ReplayBuffer(max_size=PROTOCOL["memory_size"], seed=seed)
        _ied_scale = max(env.basesim_T, 1e-8)

        def normalize_obs(obs):
            o = obs.copy().flatten()
            o[0] /= _ied_scale; o[1] /= _ied_scale; o[2] /= _ied_scale * 10
            return o.reshape(obs.shape)
        def scale_reward(r):
            return float(np.clip(r / _ied_scale, -1.0, 1.0))

        batch_size = PROTOCOL["batch_size"]
        best_cr = float("inf")
        start_time = time.time()

        print(f"\n  {cond_name}  (L_MIN={l_min}, CUT_PEN={cut_pen}, "
              f"EXT_COST={ext_cost}, COMP_LAM={comp_lam})")
        print(f"  {'-'*55}")

        epoch_data = []
        for epoch in range(n_epochs):
            idxlist = scheduler.sample_epoch()
            epoch_start = time.time()
            _last_tick = epoch_start

            for ep_idx, episode in enumerate(idxlist):
                now = time.time()
                if now - _last_tick >= 30:
                    print(f"    ⏱ {now - start_time:.0f}s total | "
                          f"epoch {epoch+1}/{n_epochs} | "
                          f"episode {ep_idx+1}/{len(idxlist)}", flush=True)
                    _last_tick = now

                obs, steps = env.reset(episode, "T")
                raw_split_od = obs.flatten()[1]
                obs = normalize_obs(obs)
                n_cuts, n_steps = 0, 0

                for index in range(1, steps):
                    done = (index == steps - 1)
                    action = agent.act(obs)
                    obs_, reward = env.step(episode, action, index, "T")
                    actual = env._last_action
                    raw_next = obs_.flatten()[1]

                    if actual == 0 and reward == 0:
                        reward = raw_split_od - raw_next
                    reward = scale_reward(reward)

                    # Apply shaping (or not)
                    if actual == 0:
                        reward -= ext_cost
                    if actual == 1:
                        reward -= cut_pen
                        n_cuts += 1
                    n_steps += 1

                    raw_split_od = raw_next
                    obs_ = normalize_obs(obs_)
                    replay.add(obs.flatten(), actual, reward, obs_.flatten(), done)
                    if done:
                        break
                    if replay.is_ready(batch_size):
                        s, a, r, ns, d = replay.sample_batch(batch_size)
                        agent.update(s, a, r, ns, d)
                    obs = obs_

                if n_steps > 0 and comp_lam > 0:
                    pass  # complexity reg applied via outer reward
                agent.decay_epsilon()

            # Validation
            val_n_cut, val_n_ext = 0, 0
            for e in range(sidx, eidx):
                ob, s = env.reset(e, "E")
                ob = normalize_obs(ob)
                for idx in range(1, s):
                    act = agent.act(ob, greedy=True)
                    ob, _ = env.step(e, act, idx, "E")
                    if env._last_action == 1:
                        val_n_cut += 1
                    else:
                        val_n_ext += 1
                    ob = normalize_obs(ob)

            try:
                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / max(fold_basesim, 1e-8))
            except (ZeroDivisionError, ValueError):
                val_cr = float('inf')

            val_total = val_n_cut + val_n_ext
            cut_pct = 100 * val_n_cut / val_total if val_total > 0 else 0

            epoch_data.append({"epoch": epoch+1, "val_cr": val_cr, "cut_pct": cut_pct})
            marker = " ★" if val_cr < best_cr else ""
            if val_cr < best_cr:
                best_cr = val_cr
            print(f"    Epoch {epoch+1}: ValCR={val_cr:.4f} CUT={cut_pct:.0f}%{marker}")

            for i in env.clusters_E.keys():
                env.clusters_E[i][0] = []; env.clusters_E[i][1] = []
                env.clusters_E[i][3] = defaultdict(list)
            env.update_cluster("T")
            scheduler.update()

        elapsed = time.time() - start_time
        r = {
            "condition": cond_name,
            "l_min": l_min, "cut_penalty": cut_pen,
            "extend_cost": ext_cost, "complexity_lambda": comp_lam,
            "best_val_cr": best_cr,
            "final_cut_pct": epoch_data[-1]["cut_pct"],
            "epoch_data": epoch_data,
            "wall_time": elapsed,
        }
        results.append(r)

    # Verdict
    print(f"\n{'─'*60}")
    naive = [r for r in results if "naive" in r["condition"].lower()]
    shaped = [r for r in results if "shaped" in r["condition"].lower()]
    if naive and shaped:
        n_cut = naive[0]["final_cut_pct"]
        s_cut = shaped[0]["final_cut_pct"]
        print(f"  NAIVE  → CUT={n_cut:.0f}%, ValCR={naive[0]['best_val_cr']:.4f}")
        print(f"  SHAPED → CUT={s_cut:.0f}%, ValCR={shaped[0]['best_val_cr']:.4f}")
        if n_cut > 80:
            print(f"  ✓ Degeneracy CONFIRMED: naive policy converges to CUT={n_cut:.0f}%")
        else:
            print(f"  ⚠ Naive CUT% = {n_cut:.0f}% — partially degenerate")
    print(f"{'─'*60}")

    return {"experiment": "RA1", "results": results}


# ═══════════════════════════════════════════════════════════════════════
#  D1: ValCR vs CUT% sweep
# ═══════════════════════════════════════════════════════════════════════

def run_d1_valcr_sweep(traj_path, centers_path, n_traj, seed=42):
    """D1 — ValCR vs CUT% with random policies."""
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import (
        compute_overdist, compute_overdist_per_point, compute_overdist_length_weighted
    )

    cut_probs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]

    print(f"\n{'═'*80}")
    print("  D1: ValCR vs CUT% — Random Policy Diagnostic (METRIC-ONLY)")
    print(f"  {n_traj} trajectories, no learning, no reward shaping")
    print(f"  NOTE: Tests metric degeneracy only — reward alignment is separate.")
    print(f"{'═'*80}\n")

    header = (f"{'CutProb':>8s}  {'ActualCUT%':>10s}  "
              f"{'ValCR':>8s}  {'nValCR':>8s}  {'wValCR':>8s}  "
              f"{'#Segs':>6s}  {'AvgLen':>7s}")
    print(header)
    print("─" * 80)

    results = []
    for p in cut_probs:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)

        env = TrajRLclus(traj_path, centers_path, centers_path)
        val_pct = 0.1
        sidx = int(n_traj * (1 - val_pct))
        fold_basesim = compute_fold_basesim(env, sidx, n_traj)

        # Train pass
        for episode in range(n_traj):
            obs, steps = env.reset(episode, "T")
            for idx in range(1, steps):
                action = 1 if rng.random() < p else 0
                obs, _ = env.step(episode, action, idx, "T")

        # Eval pass
        total_cuts, total_extends, total_segs = 0, 0, 0
        seg_lengths = []
        for e in range(sidx, n_traj):
            obs, steps = env.reset(e, "E")
            seg_len, n_cuts_ep = 1, 0
            for idx in range(1, steps):
                action = 1 if rng.random() < p else 0
                obs, _ = env.step(e, action, idx, "E")
                actual = env._last_action
                seg_len += 1
                if actual == 1:
                    total_cuts += 1
                    n_cuts_ep += 1
                    seg_lengths.append(seg_len)
                    seg_len = 1
                else:
                    total_extends += 1
            seg_lengths.append(seg_len)
            total_segs += n_cuts_ep + 1

        try:
            val_od = compute_overdist(env.clusters_E)
            val_cr = float(val_od / max(fold_basesim, 1e-8))
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')

        # Normalized variants
        try:
            val_pp = compute_overdist_per_point(env.clusters_E)
            n_val_cr = float(val_pp / max(fold_basesim, 1e-8))
        except (ZeroDivisionError, ValueError):
            n_val_cr = float('inf')

        try:
            val_lw = compute_overdist_length_weighted(env.clusters_E)
            w_val_cr = float(val_lw / max(fold_basesim, 1e-8))
        except (ZeroDivisionError, ValueError):
            w_val_cr = float('inf')

        total_actions = total_cuts + total_extends
        actual_cut_pct = 100 * total_cuts / total_actions if total_actions else 0

        r = {"cut_prob": p, "actual_cut_pct": actual_cut_pct, "val_cr": val_cr,
             "n_val_cr": n_val_cr, "w_val_cr": w_val_cr,
             "total_segs": total_segs,
             "avg_seg_len": float(np.mean(seg_lengths)) if seg_lengths else 0}
        results.append(r)
        print(f"{p:>8.0%}  {actual_cut_pct:>9.1f}%  "
              f"{val_cr:>8.4f}  {n_val_cr:>8.4f}  {w_val_cr:>8.4f}  "
              f"{total_segs:>6d}  "
              f"{r['avg_seg_len']:>7.1f}")

    print("─" * 80)

    # Verdict
    val_crs = [r['val_cr'] for r in results if r['val_cr'] < float('inf')]
    if len(val_crs) >= 2:
        is_monotonic = all(val_crs[i] >= val_crs[i+1]
                          for i in range(len(val_crs)-1))
        if is_monotonic:
            print("\n  ⚠️  VERDICT: ValCR monotonically decreases with CUT%.")
            print("     Metric is STRUCTURALLY DEGENERATE.")
        else:
            print("\n  ✓  ValCR is NOT monotonically decreasing with CUT%.")
            best = min(results, key=lambda r: r['val_cr'])
            print(f"     Best ValCR = {best['val_cr']:.4f} at CUT = {best['cut_prob']:.0%}")

    return {"experiment": "D1", "results": results}


# ═══════════════════════════════════════════════════════════════════════
#  D4: Policy basin test (forced policies under drift)
# ═══════════════════════════════════════════════════════════════════════

def run_d4_policy_basin(traj_path, centers_path, n_traj, seed=42):
    """D4 — Evaluate forced policies: always-extend, always-cut, random.

    Tests the segmentation landscape under drift mode so we can see
    exactly what ValCR values each policy basin produces.
    """
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from q_rlstc.data.trajectory_scheduler import TrajectoryScheduler

    policies = [
        ("always-extend", 0.00),
        ("rare-cut-5%",   0.05),
        ("light-cut-10%", 0.10),
        ("moderate-20%",  0.20),
        ("balanced-50%",  0.50),
        ("always-cut",    1.00),
    ]

    print(f"\n{'═'*80}")
    print("  D4: Policy Basin Test — Forced Policies (Drift Mode)")
    print(f"  {n_traj} trajectories, drift scheduler active")
    print(f"{'═'*80}\n")

    header = (f"{'Policy':<18s} {'CUT%':>6s} {'ValCR':>8s} "
              f"{'#Segs':>6s} {'AvgLen':>7s}")
    print(header)
    print("─" * 55)

    results = []
    for name, cut_p in policies:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)

        scheduler = TrajectoryScheduler(
            n_trajectories=n_traj, validation_pct=0.1,
            mode="drift", seed=seed)
        sidx, eidx = scheduler.validation_range()

        env = TrajRLclus(traj_path, centers_path, centers_path)
        fold_basesim = compute_fold_basesim(env, sidx, eidx)

        # Training pass with drift schedule
        for epoch in range(3):
            idxlist = scheduler.sample_epoch()
            for episode in idxlist:
                obs, steps = env.reset(episode, "T")
                for idx in range(1, steps):
                    action = 1 if rng.random() < cut_p else 0
                    obs, _ = env.step(episode, action, idx, "T")
            env.update_cluster("T")
            scheduler.update()

        # Eval pass
        total_cuts, total_extends, total_segs = 0, 0, 0
        seg_lengths = []
        for e in range(sidx, eidx):
            obs, steps = env.reset(e, "E")
            seg_len, ep_cuts = 1, 0
            for idx in range(1, steps):
                action = 1 if rng.random() < cut_p else 0
                obs, _ = env.step(e, action, idx, "E")
                actual = env._last_action
                seg_len += 1
                if actual == 1:
                    total_cuts += 1
                    ep_cuts += 1
                    seg_lengths.append(seg_len)
                    seg_len = 1
                else:
                    total_extends += 1
            seg_lengths.append(seg_len)
            total_segs += ep_cuts + 1

        try:
            val_od = compute_overdist(env.clusters_E)
            val_cr = float(val_od / max(fold_basesim, 1e-8))
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')


        total_actions = total_cuts + total_extends
        actual_cut_pct = 100 * total_cuts / total_actions if total_actions else 0
        avg_sl = float(np.mean(seg_lengths)) if seg_lengths else 0

        r = {"policy": name, "cut_prob": cut_p,
             "actual_cut_pct": actual_cut_pct, "val_cr": val_cr,
             "total_segs": total_segs, "avg_seg_len": avg_sl}
        results.append(r)
        print(f"{name:<18s} {actual_cut_pct:>5.1f}% {val_cr:>8.4f} "
              f"{total_segs:>6d} {avg_sl:>7.1f}")

    print("─" * 55)
    best = min(results, key=lambda r: r['val_cr'] if r['val_cr'] < float('inf') else 1e9)
    print(f"\n  Best under drift: {best['policy']} → ValCR={best['val_cr']:.4f}")

    return {"experiment": "D4", "results": results}


# ═══════════════════════════════════════════════════════════════════════
#  S1: Scalability timing
# ═══════════════════════════════════════════════════════════════════════

def run_s1_scalability(traj_path, centers_path, seed=42):
    """S1 — Scalability: greedy inference timing at various trajectory counts."""
    from q_rlstc.data.rlstc_mdp import TrajRLclus

    amounts = [250, 500, 1000]
    print(f"\n{'═'*60}")
    print("  S1: Scalability — Greedy Inference Timing")
    print(f"{'═'*60}\n")

    results = []
    for n in amounts:
        env = TrajRLclus(traj_path, centers_path, centers_path)
        np.random.seed(seed)

        t0 = time.time()
        for episode in range(min(n, 500)):
            obs, steps = env.reset(episode, "T")
            for idx in range(1, steps):
                obs, _ = env.step(episode, 0, idx, "T")
        elapsed = time.time() - t0
        r = {"n_trajectories": min(n, 500), "time_s": elapsed}
        results.append(r)
        print(f"  N={min(n,500):>5d}: {elapsed:.2f}s")

    return {"experiment": "S1", "results": results}


# ═══════════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════════

def print_summary_table(all_results: List[Dict], experiment: str):
    """Print a formatted summary table for an experiment's results."""
    print(f"\n{'═'*120}")
    print(f"  {experiment} — Summary")
    print(f"{'═'*120}\n")

    header = (f"{'Model':<30s} {'Params':>6s} {'ValCR':>8s} "
              f"{'OD':>8s} {'bs':>8s} "
              f"{'CUT%':>6s} {'#Segs':>6s} {'Sil':>6s} {'BestEp':>6s} "
              f"{'Time':>7s} {'Qmargin':>8s}")
    print(header)
    print("─" * 120)

    for r in all_results:
        qm = r.get("q_margins", [])
        qm_str = f"{qm[-1]:+.4f}" if qm else "N/A"
        od_val = r.get("val_od", 0.0)
        bs_val = r.get("val_basesim", 0.0)
        sil_val = r.get("silhouette", r.get("final_silhouette", 0.0))
        print(f"{r['model']:<30s} "
              f"{r['params']:>6d} "
              f"{r['val_cr']:>8.4f} "
              f"{od_val:>8.4f} "
              f"{bs_val:>8.4f} "
              f"{r['cut_pct']:>5.0f}% "
              f"{r['n_segs']:>6d} "
              f"{sil_val:>+6.3f} "
              f"{r.get('best_epoch', 0):>6d} "
              f"{r['wall_time']:>6.1f}s "
              f"{qm_str:>8s}")

    print("─" * 120)

    # Baseline integrity check: warn if bs varies across models
    bs_values = [r.get("val_basesim", 0.0) for r in all_results if r.get("val_basesim", 0.0) > 0]
    if bs_values and (max(bs_values) - min(bs_values)) > 0.001:
        print(f"  ⚠ BASELINE INCONSISTENCY: bs ranges from {min(bs_values):.4f} to "
              f"{max(bs_values):.4f} — CR values may not be directly comparable")


def run_multi_seed_experiment(
    spec_fn,
    traj_path: str,
    centers_path: str,
    n_traj: int,
    n_epochs: int,
    seeds: List[int],
    experiment_label: str,
) -> List[Dict]:
    """Run all specs from spec_fn across multiple seeds, aggregating mean±std.

    Returns a list of dicts (one per spec) with aggregated metrics.
    """
    specs = spec_fn()
    aggregated = []

    for spec in specs:
        per_seed_results = []
        for si, seed in enumerate(seeds):
            print(f"  [seed {si+1}/{len(seeds)}: {seed}] {spec.name}")
            agent = build_agent(spec, seed)
            r = train_and_evaluate(
                agent, spec, traj_path, centers_path,
                n_traj, n_epochs, seed,
            )
            per_seed_results.append(r)

        # ── Collapse-aware aggregation ─────────────────────────
        n_collapsed = sum(1 for r in per_seed_results if r.get("collapsed", False))
        healthy = [r for r in per_seed_results if not r.get("collapsed", False)]
        if not healthy:
            healthy = per_seed_results  # fallback: all collapsed

        crs = [r["val_cr"] for r in per_seed_results]
        cuts = [r["cut_pct"] for r in per_seed_results]
        segs = [r["n_segs"] for r in per_seed_results]
        times = [r["wall_time"] for r in per_seed_results]
        qms = [r["q_margins"][-1] for r in per_seed_results if r.get("q_margins")]
        sses = [r.get("sse", 0.0) for r in per_seed_results]
        eps_to_best = [r.get("episodes_to_best", 0) for r in per_seed_results]
        acts_to_best = [r.get("actions_to_best", 0) for r in per_seed_results]
        t_envs = [r.get("time_env", 0.0) for r in per_seed_results]
        t_agents = [r.get("time_agent", 0.0) for r in per_seed_results]

        # Healthy-seed aggregation for primary metrics
        healthy_crs = [r["val_cr"] for r in healthy]
        healthy_cuts = [r["cut_pct"] for r in healthy]

        agg = per_seed_results[0].copy()
        # Primary: healthy-only mean ± std
        agg["val_cr"] = float(np.mean(healthy_crs))
        agg["val_cr_std"] = float(np.std(healthy_crs))
        agg["cut_pct"] = float(np.mean(healthy_cuts))
        agg["cut_pct_std"] = float(np.std(healthy_cuts))
        agg["n_segs"] = int(np.mean(segs))
        agg["wall_time"] = float(np.sum(times))
        agg["n_seeds"] = len(seeds)
        agg["n_collapsed"] = n_collapsed
        agg["per_seed_crs"] = crs
        agg["per_seed_cuts"] = cuts
        agg["per_seed_collapsed"] = [r.get("collapsed", False) for r in per_seed_results]
        # Full-population (all seeds) for reference
        agg["val_cr_all"] = float(np.mean(crs))
        agg["val_cr_all_std"] = float(np.std(crs))
        # SSE
        agg["sse"] = float(np.mean(sses))
        agg["sse_std"] = float(np.std(sses))
        agg["per_seed_sses"] = sses
        # Sample efficiency
        agg["episodes_to_best"] = float(np.mean(eps_to_best))
        agg["actions_to_best"] = float(np.mean(acts_to_best))
        # Timing breakdown (means across seeds)
        agg["time_env"] = float(np.mean(t_envs))
        agg["time_agent"] = float(np.mean(t_agents))
        if qms:
            agg["q_margins"] = [float(np.mean(qms))]
            agg["q_margin_std"] = float(np.std(qms))

        if n_collapsed:
            print(f"  ⚠ {spec.name}: {n_collapsed}/{len(seeds)} seeds collapsed "
                  f"(healthy-only ValCR={agg['val_cr']:.4f}±{agg['val_cr_std']:.4f}, "
                  f"all-seeds={agg['val_cr_all']:.4f}±{agg['val_cr_all_std']:.4f})")
        aggregated.append(agg)

    return aggregated


def print_multi_seed_table(results: List[Dict], experiment: str):
    """Print summary table with mean±std for multi-seed results (lower ValCR = better)."""
    print(f"\n{'═'*110}")
    n_seeds = results[0].get('n_seeds', 1)
    print(f"  {experiment} — Multi-Seed Summary ({n_seeds} seeds) | ValCR: lower = better")
    print(f"{'═'*110}\n")

    header = (f"{'Model':<30s} {'Params':>6s} {'ValCR (healthy)':>18s} "
              f"{'CUT%':>12s} {'Coll':>5s} {'#Segs':>6s} {'Time':>8s} {'Qmargin':>14s}")
    print(header)
    print("─" * 110)

    for r in results:
        cr_str = f"{r['val_cr']:.4f}±{r.get('val_cr_std', 0):.4f}"
        cut_str = f"{r['cut_pct']:.0f}%±{r.get('cut_pct_std', 0):.0f}%"
        n_coll = r.get('n_collapsed', 0)
        coll_str = f"{n_coll}/{r.get('n_seeds', 1)}"
        qm = r.get("q_margins", [])
        qm_std = r.get("q_margin_std", 0)
        qm_str = f"{qm[-1]:+.3f}±{qm_std:.3f}" if qm else "N/A"
        print(f"{r['model']:<30s} "
              f"{r['params']:>6d} "
              f"{cr_str:>18s} "
              f"{cut_str:>12s} "
              f"{coll_str:>5s} "
              f"{r['n_segs']:>6d} "
              f"{r['wall_time']:>7.0f}s "
              f"{qm_str:>14s}")

    print("─" * 110)


def print_pareto_table(d1_results, agent_results: List[Dict]):
    """Print Pareto-constrained ValCR table.

    Shows best ValCR from random policy AND learned agents at CUT ≤ thresholds.
    """
    thresholds = [5, 10, 20, 30, 40, 50, 80]
    print(f"\n{'═'*80}")
    print("  Pareto: Lowest (best) ValCR at CUT ≤ threshold  [lower = better]")
    print(f"{'═'*80}\n")

    header = f"{'CUT ≤':<8s}"
    sources = ["Random (D1)"]
    for r in agent_results:
        sources.append(r["model"])
    for s in sources:
        header += f"  {s:>16s}"
    print(header)
    print("─" * 80)

    for thresh in thresholds:
        row = f"{'≤'+str(thresh)+'%':<8s}"
        # Random baseline (D1)
        if d1_results:
            d1r = d1_results["results"]
            valid = [r for r in d1r if r["cut_prob"] * 100 <= thresh]
            if valid:
                best = min(r["val_cr"] for r in valid)
                row += f"  {best:>16.4f}"
            else:
                row += f"  {'—':>16s}"
        else:
            row += f"  {'—':>16s}"
        # Learned agents
        for r in agent_results:
            if r["cut_pct"] <= thresh:
                row += f"  {r['val_cr']:>16.4f}"
            else:
                cut_str = f"({r['cut_pct']:.0f}%)"
                row += f"  {cut_str:>16s}"
        print(row)

    print("─" * 80)


def print_constrained_leaderboard(agent_results: List[Dict], d1_results=None):
    """Print CUT-constrained leaderboard — primary thesis result table.

    For each CUT% threshold, shows the best model. This is the
    degeneracy-immune evaluation the committee needs to see.
    """
    thresholds = [5, 10, 20]
    print(f"\n{'═'*100}")
    print("  CONSTRAINED LEADERBOARD: Best model at CUT% ≤ threshold")
    print("  (Immunizes against 'you just cut more' critique)")
    print(f"{'═'*100}\n")

    print("  Because CR decreases sharply with segmentation rate even for random")
    print("  policies (D1), all results are interpreted jointly with CUT%/#segs")
    print("  and relative to the random-cut envelope.\n")

    header = (f"{'CUT ≤':<8s} {'Best Model':<30s} {'ValCR':>8s} "
              f"{'CUT%':>6s} {'#Segs':>6s} {'OD':>8s} {'bs':>8s} "
              f"{'Sil':>6s} {'Regime':<10s}")
    print(header)
    print("─" * 100)

    for thresh in thresholds:
        eligible = [r for r in agent_results if r.get("cut_pct", 100) <= thresh]
        if not eligible:
            print(f"{'≤'+str(thresh)+'%':<8s} {'(no model qualifies)':<30s}")
            continue
        best = min(eligible, key=lambda r: r["val_cr"])
        regime = "SPSA" if best.get("kind") in ("quantum", "classical") else "SGD/Adam"
        sil = best.get("silhouette", best.get("final_silhouette", 0.0))
        print(f"{'≤'+str(thresh)+'%':<8s} {best['model']:<30s} "
              f"{best['val_cr']:>8.4f} "
              f"{best['cut_pct']:>5.0f}% "
              f"{best['n_segs']:>6d} "
              f"{best.get('val_od', 0.0):>8.4f} "
              f"{best.get('val_basesim', 0.0):>8.4f} "
              f"{sil:>+6.3f} "
              f"{regime:<10s}")

        # Also show random baseline at same constraint (if D1 available)
        if d1_results:
            d1r = d1_results.get("results", [])
            d1_valid = [r for r in d1r if r.get("cut_prob", 1.0) * 100 <= thresh]
            if d1_valid:
                d1_best = min(d1_valid, key=lambda r: r["val_cr"])
                print(f"{'':>8s} {'  └─ Random baseline':<30s} "
                      f"{d1_best['val_cr']:>8.4f} "
                      f"{d1_best['cut_prob']*100:>5.0f}% ")

    print("─" * 100)


# ═══════════════════════════════════════════════════════════════════════
#  Plot generation
# ═══════════════════════════════════════════════════════════════════════

def generate_plots(all_experiment_results: Dict, plot_dir: Path):
    """Generate all thesis plots from collected results."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        from q_rlstc.visualization.plot_utils import (
            plot_learning_curves,
            plot_metric_comparison,
            plot_elbow,
            plot_silhouette_analysis,
            plot_timing_per_k,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping plots.")
        return

    print(f"\n{'═'*60}")
    print("  Generating Plots")
    print(f"{'═'*60}\n")
    n_plots = 0

    # ── D1: ValCR vs CUT% ───────────────────────────────────────
    d1 = all_experiment_results.get("D1")
    if d1:
        fig, ax = plt.subplots(figsize=(8, 5))
        d1r = d1["results"]
        probs = [r["cut_prob"] * 100 for r in d1r]
        crs = [r["val_cr"] for r in d1r]
        ax.plot(probs, crs, "o-", linewidth=2, color="#3cb44b")
        for x, y in zip(probs, crs):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("CUT Probability (%)")
        ax.set_ylabel("ValCR (lower = better)")
        # Extract bs from D1 results for subtitle
        d1_bs = d1.get("fold_basesim", "")
        if not d1_bs and d1.get("results"):
            # Try to extract from first result
            d1_bs = d1["results"][0].get("fold_basesim", "")
        bs_str = f" (bs={d1_bs:.4f})" if isinstance(d1_bs, (int, float)) else ""
        ax.set_title(f"D1: ValCR vs CUT% (Random Policy){bs_str}")
        fig.tight_layout()
        fig.savefig(str(plot_dir / "d1_valcr_vs_cut.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ d1_valcr_vs_cut.png")

    # ── Pareto: ValCR-vs-CUT% overlay (agents on D1 baseline) ───
    # Shows whether learned agents beat random at matched CUT rates
    d1 = all_experiment_results.get("D1")
    agent_points = []  # collect (CUT%, ValCR, name, kind) from all experiments
    for exp_name, exp_data in all_experiment_results.items():
        if not isinstance(exp_data, list):
            continue
        for r in exp_data:
            if "val_cr" in r and "val_cut_pcts" in r and r["val_cut_pcts"]:
                agent_points.append({
                    "cut_pct": r["val_cut_pcts"][-1],  # final epoch CUT%
                    "val_cr": r["val_cr"],              # best ValCR
                    "name": r["model"],
                    "kind": r.get("kind", "classical"),
                    "exp": exp_name,
                })
    if d1 or agent_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        # D1 random baseline curve
        if d1:
            d1r = d1["results"]
            d1_cuts = [r["cut_prob"] * 100 for r in d1r]
            d1_crs = [r["val_cr"] for r in d1r]
            ax.plot(d1_cuts, d1_crs, "o--", linewidth=1.5, color="#aaaaaa",
                    markersize=5, label="Random policy (D1)", zorder=1)
            # Shade the Pareto-optimal random region
            best_cr = min(d1_crs)
            ax.axhline(y=best_cr, color="#cccccc", linestyle=":", alpha=0.7,
                       label=f"Random optimum ({best_cr:.3f})")
        # Learned agent points
        markers = {"quantum": "D", "classical": "s"}
        colors_q = {"quantum": "#e6194B", "classical": "#4363d8"}
        for pt in agent_points:
            m = markers.get(pt["kind"], "o")
            c = colors_q.get(pt["kind"], "#808080")
            ax.scatter(pt["cut_pct"], pt["val_cr"], marker=m, s=120,
                       c=c, edgecolors="black", linewidth=0.8, zorder=3,
                       label=f"{pt['name']} ({pt['exp']})")
            ax.annotate(f"{pt['val_cr']:.3f}",
                        (pt["cut_pct"], pt["val_cr"]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color=c)
        ax.set_xlabel("CUT% (segmentation rate)")
        ax.set_ylabel("ValCR (lower = better)")
        # Add CUT% constraint lines
        for thresh, ls in [(5, ':'), (10, '--'), (20, '-.')]:
            ax.axvline(x=thresh, color='#999999', linestyle=ls, alpha=0.4,
                       label=f'CUT≤{thresh}%')
        ax.set_title("Pareto Frontier: Learned Agents vs Random Baseline\n"
                     "(CR decreases with CUT% even for random — evaluate jointly)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(str(plot_dir / "pareto_valcr_vs_cut.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ pareto_valcr_vs_cut.png")

    # ── Diagnostic: per-seed CUT% vs ValCR & #segs vs OD ────────
    for exp_name, exp_data in all_experiment_results.items():
        if not isinstance(exp_data, list):
            continue
        for r in exp_data:
            if not r.get("val_cut_pcts") or not r.get("val_crs"):
                continue
            safe_name = r["model"].replace(" ", "_").replace("/", "")
            # CUT% vs ValCR per epoch
            fig, ax = plt.subplots(figsize=(6, 4))
            epochs_x = list(range(1, len(r["val_cut_pcts"]) + 1))
            ax.scatter(r["val_cut_pcts"], r["val_crs"], c=epochs_x,
                       cmap="viridis", s=60, edgecolors="black", linewidth=0.5)
            for i, (cx, cy) in enumerate(zip(r["val_cut_pcts"], r["val_crs"])):
                ax.annotate(f"E{i+1}", (cx, cy), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))
            ax.set_xlabel("CUT%")
            ax.set_ylabel("ValCR (lower = better)")
            collapsed_tag = " [COLLAPSED]" if r.get("collapsed") else ""
            ax.set_title(f"{r['model']}: CUT% vs ValCR{collapsed_tag}")
            fig.tight_layout()
            fig.savefig(str(plot_dir / f"diag_cut_vs_cr_{safe_name}.png"), dpi=120)
            plt.close(fig)
            n_plots += 1

            # #segs vs OD per epoch
            if r.get("val_ods") and r.get("val_seg_counts"):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(r["val_seg_counts"], r["val_ods"], c=epochs_x,
                           cmap="plasma", s=60, edgecolors="black", linewidth=0.5)
                for i, (sx, sy) in enumerate(zip(r["val_seg_counts"], r["val_ods"])):
                    ax.annotate(f"E{i+1}", (sx, sy), fontsize=7,
                                textcoords="offset points", xytext=(4, 4))
                ax.set_xlabel("#Segments")
                ax.set_ylabel("OD (lower = better)")
                ax.set_title(f"{r['model']}: #Segs vs OD{collapsed_tag}")
                fig.tight_layout()
                fig.savefig(str(plot_dir / f"diag_segs_vs_od_{safe_name}.png"), dpi=120)
                plt.close(fig)
                n_plots += 1
        print(f"  ✓ diagnostics for {exp_name}")

    # ── D2: Q-margin evolution ──────────────────────────────────
    for exp_name, exp_data in all_experiment_results.items():
        if not isinstance(exp_data, list):
            continue
        for r in exp_data:
            qm = r.get("q_margins", [])
            if len(qm) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(1, len(qm)+1), qm, "s-", linewidth=2)
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Q(extend) – Q(cut)")
                safe_name = r["model"].replace(" ", "_").replace("/", "")
                ax.set_title(f"D2: Q-Margin — {r['model']}")
                fig.tight_layout()
                path = plot_dir / f"d2_qmargin_{safe_name}.png"
                fig.savefig(str(path), dpi=150)
                plt.close(fig)
                n_plots += 1
                print(f"  ✓ d2_qmargin_{safe_name}.png")

    # ── E1+: ValCR comparison bar chart ─────────────────────────
    for exp_name, exp_data in all_experiment_results.items():
        if not isinstance(exp_data, list) or not exp_data:
            continue
        if not all("val_cr" in r for r in exp_data):
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["model"] for r in exp_data]
        crs = [r["val_cr"] for r in exp_data]
        colors = ["#e6194B" if r.get("kind") == "quantum" else "#4363d8"
                  for r in exp_data]
        bars = ax.bar(names, crs, color=colors, alpha=0.85)
        for bar, cr in zip(bars, crs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{cr:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("ValCR (lower = better)")
        # Add bs to subtitle from first result
        bs_val = exp_data[0].get("val_basesim", exp_data[0].get("config", {}).get("bs", ""))
        bs_str = f"\n(bs={bs_val:.4f}, L_MIN={PROTOCOL['L_MIN']}, CUT_PEN={PROTOCOL['CUT_PENALTY']})" if isinstance(bs_val, (int, float)) and bs_val > 0 else ""
        ax.set_title(f"{exp_name}: Validation CR Comparison{bs_str}", fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(str(plot_dir / f"{exp_name.lower()}_valcr_comparison.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ {exp_name.lower()}_valcr_comparison.png")

    # ── E4/E5: CUT% evolution per epoch ─────────────────────────
    for exp_name, exp_data in all_experiment_results.items():
        if not isinstance(exp_data, list):
            continue
        has_cuts = any(len(r.get("val_cut_pcts", [])) > 1 for r in exp_data)
        if not has_cuts:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in exp_data:
            pcts = r.get("val_cut_pcts", [])
            if pcts:
                ax.plot(range(1, len(pcts)+1), pcts, "o-", label=r["model"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CUT%")
        ax.set_title(f"{exp_name}: CUT% Evolution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(plot_dir / f"{exp_name.lower()}_cut_evolution.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ {exp_name.lower()}_cut_evolution.png")

    # ── S1: Scalability timing ──────────────────────────────────
    s1 = all_experiment_results.get("S1")
    if s1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = [r["n_trajectories"] for r in s1["results"]]
        ts = [r["time_s"] for r in s1["results"]]
        ax.plot(ns, ts, "o-", linewidth=2, color="#f58231")
        ax.set_xlabel("Number of Trajectories")
        ax.set_ylabel("Time (s)")
        ax.set_title("S1: Scalability — Greedy Inference")
        fig.tight_layout()
        fig.savefig(str(plot_dir / "s1_scalability.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ s1_scalability.png")

    print(f"\n  Generated {n_plots} plots → {plot_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════════════════

def generate_report(all_results: Dict, output_dir: Path, terminal_log: str):
    """Generate markdown + JSON report files."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON dump
    json_path = output_dir / f"thesis_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    # Markdown report
    md_path = output_dir / f"thesis_report_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Q-RLSTC Thesis Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(METRIC_DEFINITIONS)
        f.write("\n")
        f.write(f"## Protocol Constants\n\n")
        f.write("```json\n")
        f.write(json.dumps(PROTOCOL, indent=2))
        f.write("\n```\n\n")
        f.write(f"## Full Terminal Output\n\n")
        f.write("```\n")
        f.write(terminal_log)
        f.write("\n```\n\n")
        f.write(f"## Plots\n\n")
        plot_dir = output_dir / "plots"
        if plot_dir.exists():
            for pg in sorted(plot_dir.glob("*.png")):
                f.write(f"![{pg.stem}](plots/{pg.name})\n\n")
        f.write(f"\n## Raw Results (JSON)\n\n")
        f.write(f"See `{json_path.name}` for machine-readable data.\n")
    print(f"  Report → {md_path}")

    return json_path, md_path


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_REGISTRY = {
    "D1": "ValCR vs CUT% sweep (metric-only)",
    "D2": "Q-margin profiling (piggybacked on E1-E7)",
    "D3": "Replay training-action distribution (piggybacked on E1-E7)",
    "D4": "Policy basin test (forced policies under drift)",
    "D5": "Replay buffer histogram (piggybacked on E1-E7)",
    "E1": "Core Quantum Utility",
    "E2": "NISQ Viability",
    "E3": "Shot Sensitivity",
    "E4": "Drift Resilience",
    "E5": "Low-Data Generalization",
    "E6": "Version Progression",
    "E7": "Configuration Sweep (qubits × layers × ansatz)",
    "E8": "Data Scaling (sample efficiency: 30/50/100/300 traj)",
    "E9": "Quantum B (input scaling + anti-BP + circular entanglement)",
    "S1": "Scalability Timing",
    "AB1": "Entanglement Ablation (no-CNOT vs linear)",
    "RA1": "Reward Ablation (naive vs shaped)",
    "PARITY": "RLSTC Parity Comparison (thesis vs RLSTC conditions)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Thesis Experiment Runner for Q-RLSTC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--traj-path",
        default=str(_DATA_DIR / "Tdrive_norm_traj"),
    )
    parser.add_argument(
        "--centers-path",
        default=str(_DATA_DIR / "tdrive_clustercenter"),
    )
    parser.add_argument("--amount", type=int, default=500,
                        help="Number of trajectories (default: 500)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per model (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seeds for multi-seed runs (e.g., 42,123,7,99,2025). "
             "When set, E1-E6 run each model across all seeds and report mean±std.",
    )
    parser.add_argument("--output-dir", default="results/thesis",
                        help="Output directory (default: results/thesis)")
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment IDs (default: all). "
             f"Available: {','.join(EXPERIMENT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--plots-only",
        default=None,
        help="Path to existing JSON — regenerate plots only, skip experiments.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    def save_intermediate_state(results_dict: Dict[str, Any], current_tee: TeePrinter):
        """Save JSON, Markdown, and plots incrementally so data isn't lost on crash."""
        print(f"\n  [Auto-Save] Saving intermediate results to {output_dir}...")
        try:
            generate_plots(results_dict, plot_dir)
            generate_report(results_dict, output_dir, current_tee.getvalue())
        except Exception as e:
            print(f"  [Auto-Save Error] Could not save intermediate state: {e}")

    # ── Plots-only mode ──────────────────────────────────────────
    if args.plots_only:
        with open(args.plots_only) as f:
            data = json.load(f)
        generate_plots(data, plot_dir)
        return 0

    # ── Parse experiment selection ────────────────────────────────
    if args.experiments:
        selected = [e.strip().upper() for e in args.experiments.split(",")]
    else:
        selected = list(EXPERIMENT_REGISTRY.keys())

    tee = TeePrinter()
    env_meta = _collect_env_metadata()
    all_results: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "args": vars(args),
        "env": env_meta,
    }

    with tee:
        print(f"\n{'═'*70}")
        print(f"  Q-RLSTC THESIS EXPERIMENT RUNNER")
        print(f"  {env_meta['timestamp']}")
        print(f"  Git: {env_meta['git_hash'][:12]}")
        print(f"  Python: {sys.version.split()[0]}  |  NumPy: {np.__version__}")
        print(f"  Experiments: {', '.join(selected)}")
        print(f"  Data: {args.amount} trajectories, {args.epochs} epochs")
        print(f"  Seed: {args.seed}")
        seed_list = None
        if args.seeds:
            seed_list = [int(s.strip()) for s in args.seeds.split(",")]
            print(f"  Multi-seed: {seed_list}")
        print(f"  Output: {output_dir}")
        print(f"{'═'*70}")

        # ── D1: ValCR vs CUT% ────────────────────────────────────
        if "D1" in selected:
            d1_n = min(args.amount, 100)
            d1_result = run_d1_valcr_sweep(
                args.traj_path, args.centers_path, d1_n, args.seed)
            all_results["D1"] = d1_result
            save_intermediate_state(all_results, tee)

        # ── D4: Policy basin test ────────────────────────────────
        if "D4" in selected:
            d4_n = min(args.amount, 100)
            d4_result = run_d4_policy_basin(
                args.traj_path, args.centers_path, d4_n, args.seed)
            all_results["D4"] = d4_result
            save_intermediate_state(all_results, tee)

        # ── E1: Core Quantum Utility ─────────────────────────────
        if "E1" in selected:
            print(f"\n{'═'*70}")
            print(f"  E1: CORE QUANTUM UTILITY")
            print(f"{'═'*70}")
            if seed_list and len(seed_list) > 1:
                e1_results = run_multi_seed_experiment(
                    get_e1_specs, args.traj_path, args.centers_path,
                    args.amount, args.epochs, seed_list, "E1")
                all_results["E1"] = e1_results
                print_multi_seed_table(e1_results, "E1: Core Quantum Utility")
            else:
                e1_results = []
                for spec in get_e1_specs():
                    agent = build_agent(spec, args.seed)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        args.amount, args.epochs, args.seed)
                    e1_results.append(r)
                all_results["E1"] = e1_results
                print_summary_table(e1_results, "E1: Core Quantum Utility")
            save_intermediate_state(all_results, tee)

        # ── E2: NISQ Viability ───────────────────────────────────
        if "E2" in selected:
            print(f"\n{'═'*70}")
            print(f"  E2: NISQ VIABILITY")
            print(f"{'═'*70}")
            e2_results = []
            for spec in get_e2_specs():
                agent = build_agent(spec, args.seed)
                r = train_and_evaluate(
                    agent, spec, args.traj_path, args.centers_path,
                    args.amount, args.epochs, args.seed)
                e2_results.append(r)
            all_results["E2"] = e2_results
            print_summary_table(e2_results, "E2: NISQ Viability")
            save_intermediate_state(all_results, tee)

        # ── E3: Shot Sensitivity ─────────────────────────────────
        if "E3" in selected:
            print(f"\n{'═'*70}")
            print(f"  E3: SHOT SENSITIVITY")
            print(f"{'═'*70}")
            e3_results = []
            for spec in get_e3_specs():
                agent = build_agent(spec, args.seed)
                r = train_and_evaluate(
                    agent, spec, args.traj_path, args.centers_path,
                    args.amount, args.epochs, args.seed)
                e3_results.append(r)
            all_results["E3"] = e3_results
            print_summary_table(e3_results, "E3: Shot Sensitivity")
            save_intermediate_state(all_results, tee)

        # ── E4: Drift Resilience ─────────────────────────────────
        if "E4" in selected:
            print(f"\n{'═'*70}")
            print(f"  E4: DRIFT RESILIENCE")
            print(f"{'═'*70}")
            e4_results = []
            for spec in get_e4_specs():
                agent = build_agent(spec, args.seed)
                r = train_and_evaluate(
                    agent, spec, args.traj_path, args.centers_path,
                    args.amount, args.epochs, args.seed)
                e4_results.append(r)
            all_results["E4"] = e4_results
            print_summary_table(e4_results, "E4: Drift Resilience")
            save_intermediate_state(all_results, tee)

        # ── E5: Low-Data ─────────────────────────────────────────
        if "E5" in selected:
            print(f"\n{'═'*70}")
            print(f"  E5: LOW-DATA GENERALIZATION")
            print(f"{'═'*70}")
            e5_results = []
            for spec in get_e5_specs():
                agent = build_agent(spec, args.seed)
                r = train_and_evaluate(
                    agent, spec, args.traj_path, args.centers_path,
                    args.amount, args.epochs, args.seed)
                e5_results.append(r)
            all_results["E5"] = e5_results
            print_summary_table(e5_results, "E5: Low-Data Generalization")
            save_intermediate_state(all_results, tee)

        # ── E6: Version Progression ──────────────────────────────
        if "E6" in selected:
            print(f"\n{'═'*70}")
            print(f"  E6: VERSION PROGRESSION")
            print(f"{'═'*70}")
            e6_results = []
            for spec in get_e6_specs():
                agent = build_agent(spec, args.seed)
                r = train_and_evaluate(
                    agent, spec, args.traj_path, args.centers_path,
                    args.amount, args.epochs, args.seed)
                e6_results.append(r)
            all_results["E6"] = e6_results
            print_summary_table(e6_results, "E6: Version Progression")
            save_intermediate_state(all_results, tee)

        # ── S1: Scalability ──────────────────────────────────────
        if "S1" in selected:
            s1_result = run_s1_scalability(
                args.traj_path, args.centers_path, args.seed)
            all_results["S1"] = s1_result
            save_intermediate_state(all_results, tee)

        # ── AB1: Entanglement Ablation ───────────────────────────────
        if "AB1" in selected:
            print(f"\n{'═'*70}")
            print(f"  AB1: ENTANGLEMENT ABLATION (no-CNOT vs linear)")
            print(f"{'═'*70}")
            if seed_list and len(seed_list) > 1:
                ab1_results = run_multi_seed_experiment(
                    get_ablation_entanglement_specs, args.traj_path,
                    args.centers_path, args.amount, args.epochs, seed_list, "AB1")
                all_results["AB1"] = ab1_results
                print_multi_seed_table(ab1_results, "AB1: Entanglement Ablation")
            else:
                ab1_results = []
                for spec in get_ablation_entanglement_specs():
                    agent = build_agent(spec, args.seed)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        args.amount, args.epochs, args.seed)
                    ab1_results.append(r)
                all_results["AB1"] = ab1_results
                print_summary_table(ab1_results, "AB1: Entanglement Ablation")
            save_intermediate_state(all_results, tee)

        # ── E7: Configuration Sweep ──────────────────────────────
        if "E7" in selected:
            print(f"\n{'═'*70}")
            print(f"  E7: CONFIGURATION SWEEP (qubits × layers × ansatz)")
            print(f"{'═'*70}")
            e7_specs = get_e7_specs()
            print(f"  {len(e7_specs)} configurations to evaluate")
            if seed_list and len(seed_list) > 1:
                e7_results = run_multi_seed_experiment(
                    get_e7_specs, args.traj_path, args.centers_path,
                    args.amount, args.epochs, seed_list, "E7")
                all_results["E7"] = e7_results
                print_multi_seed_table(e7_results, "E7: Configuration Sweep")
            else:
                e7_results = []
                for spec in e7_specs:
                    agent = build_agent(spec, args.seed)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        args.amount, args.epochs, args.seed)
                    e7_results.append(r)
                all_results["E7"] = e7_results
                print_summary_table(e7_results, "E7: Configuration Sweep")
            save_intermediate_state(all_results, tee)

        # ── E8: Data Scaling ──────────────────────────────────────
        if "E8" in selected:
            print(f"\n{'═'*70}")
            print(f"  E8: DATA SCALING (sample efficiency)")
            print(f"{'═'*70}")
            e8_results = []
            data_sizes = [30, 50, 100, 300]
            for n_traj in data_sizes:
                actual_n = min(n_traj, args.amount)
                print(f"\n  --- {actual_n} trajectories ---")
                for spec in get_e8_specs(actual_n):
                    agent = build_agent(spec, args.seed)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        actual_n, args.epochs, args.seed)
                    e8_results.append(r)
            all_results["E8"] = e8_results
            print_summary_table(e8_results, "E8: Data Scaling")
            save_intermediate_state(all_results, tee)

        # ── E9: Quantum B ──────────────────────────────────────────
        if "E9" in selected:
            print(f"\n{'═'*70}")
            print(f"  E9: QUANTUM B — Best Quantum (input scaling + anti-BP + circular)")
            print(f"{'═'*70}")
            e9_results = []
            for spec in get_e9_specs():
                for s in seeds:
                    agent = build_agent(spec, s)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        args.amount, args.epochs, s)
                    e9_results.append(r)
            all_results["E9"] = e9_results
            print_summary_table(e9_results, "E9: Quantum B")
            save_intermediate_state(all_results, tee)

        # ── RA1: Reward Ablation ─────────────────────────────────
        if "RA1" in selected:
            ra1_result = run_ra1_reward_ablation(
                args.traj_path, args.centers_path,
                min(args.amount, 30), args.epochs, args.seed)
            all_results["RA1"] = ra1_result
            save_intermediate_state(all_results, tee)

        # ── PARITY: RLSTC Parity Comparison ──────────────────────
        if "PARITY" in selected:
            print(f"\n{'═'*70}")
            print(f"  PARITY: RLSTC PARITY COMPARISON")
            print(f"  (VQ-DQN / MLP-34 / Original under RLSTC vs Thesis conditions)")
            print(f"{'═'*70}")
            if seed_list and len(seed_list) > 1:
                parity_results = run_multi_seed_experiment(
                    get_parity_specs, args.traj_path, args.centers_path,
                    args.amount, args.epochs, seed_list, "PARITY")
                all_results["PARITY"] = parity_results
                print_multi_seed_table(parity_results, "PARITY: RLSTC Parity")
            else:
                parity_results = []
                for spec in get_parity_specs():
                    agent = build_agent(spec, args.seed)
                    r = train_and_evaluate(
                        agent, spec, args.traj_path, args.centers_path,
                        args.amount, args.epochs, args.seed)
                    parity_results.append(r)
                all_results["PARITY"] = parity_results
                print_summary_table(parity_results, "PARITY: RLSTC Parity")
            save_intermediate_state(all_results, tee)

        # ── Grand summary ────────────────────────────────────────
        print(f"\n{'═'*70}")
        print(f"  GRAND SUMMARY")
        print(f"{'═'*70}\n")

        all_model_results = []
        for key in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "AB1"]:
            data = all_results.get(key)
            if isinstance(data, list):
                all_model_results.extend(data)

        if all_model_results:
            # ── Regime-separated tables ──────────────────────────
            spsa_regime = [r for r in all_model_results
                           if r.get("kind") in ("quantum", "classical")]
            sgd_regime = [r for r in all_model_results
                          if r.get("kind") in ("adam", "original")]

            if sgd_regime:
                print_summary_table(sgd_regime, "Regime A: SGD/Adam (backprop)")
            if spsa_regime:
                print_summary_table(spsa_regime, "Regime B: SPSA (gradient-free)")

            # ── Constrained leaderboard ──────────────────────────
            d1_data = all_results.get("D1")
            print_constrained_leaderboard(all_model_results, d1_data)

            # Pareto table (D1 + learned agents)
            if d1_data or len(all_model_results) > 0:
                print_pareto_table(d1_data, all_model_results)

            # D2/D3/D5 diagnostic summary
            print(f"\n{'─'*70}")
            print(f"  D2/D3/D5: Diagnostic Summary")
            print(f"  (Q-margins, Training Action Dist, Replay Buffer Dist)")
            print(f"{'─'*70}\n")
            for r in all_model_results:
                qm = r.get("q_margins", [])
                rp = r.get("replay_cut_pcts", [])
                rb = r.get("replay_buf_cut_pcts", [])
                qm_str = " → ".join(f"{m:+.3f}" for m in qm) if qm else "N/A"
                rp_str = " → ".join(f"{p:.0f}%" for p in rp) if rp else "N/A"
                rb_str = " → ".join(f"{p:.0f}%" for p in rb) if rb else "N/A"
                print(f"  {r['model']:<30s}")
                print(f"    D2 Q-margin:    {qm_str}")
                print(f"    D3 TrainCUT%:   {rp_str}")
                print(f"    D5 BufferCUT%:  {rb_str}")
                print()

        # ── Generate plots ───────────────────────────────────────
        generate_plots(all_results, plot_dir)

    # ── Save reports (outside tee) ───────────────────────────────
    terminal_log = tee.getvalue()
    json_path, md_path = generate_report(all_results, output_dir, terminal_log)

    print(f"\n{'═'*70}")
    print(f"  ✓ ALL EXPERIMENTS COMPLETE")
    print(f"  JSON:   {json_path}")
    print(f"  Report: {md_path}")
    print(f"  Plots:  {plot_dir}")
    print(f"{'═'*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
