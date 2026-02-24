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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
}


# ═══════════════════════════════════════════════════════════════════════
#  Model specs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    name: str
    kind: str           # "quantum", "classical" (SPSA), or "adam" (backprop)
    hidden_sizes: list = field(default_factory=list)
    n_layers: int = 3
    shots: int = 0      # 0 = statevector
    noise_model: str = "ideal"
    n_qubits: int = 5
    run_type: str = "standard"
    data_fraction: float = 1.0


def build_agent(spec: ModelSpec, seed: int):
    """Construct agent from spec (same as run_rigorous_benchmark)."""
    if spec.kind == "quantum":
        from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig
        cfg = AgentConfig(
            version="D",
            n_qubits=spec.n_qubits,
            n_layers=spec.n_layers,
            gamma=PROTOCOL["gamma"],
            epsilon_start=PROTOCOL["epsilon_start"],
            epsilon_min=PROTOCOL["epsilon_min"],
            epsilon_decay=PROTOCOL["epsilon_decay"],
            shots=spec.shots,
            target_update_freq=PROTOCOL["target_update_freq"],
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
            gamma=PROTOCOL["gamma"],
            epsilon_start=PROTOCOL["epsilon_start"],
            epsilon_min=PROTOCOL["epsilon_min"],
            epsilon_decay=PROTOCOL["epsilon_decay"],
            use_double_dqn=True,
            target_update_freq=PROTOCOL["target_update_freq"],
        )
        return AdamClassicalDQN(config=cfg, seed=seed)
    elif spec.kind == "original":
        from q_rlstc.rl.original_classical_agent import OriginalClassicalDQN, OriginalAgentConfig
        cfg = OriginalAgentConfig(
            hidden_size=64,
            gamma=PROTOCOL["gamma"],
            epsilon_start=PROTOCOL["epsilon_start"],
            epsilon_min=PROTOCOL["epsilon_min"],
            epsilon_decay=PROTOCOL["epsilon_decay"],
        )
        return OriginalClassicalDQN(config=cfg, seed=seed)
    else:
        from q_rlstc.rl.spsa_classical_agent import SPSAClassicalDQN, ClassicalAgentConfig
        cfg = ClassicalAgentConfig(
            hidden_sizes=spec.hidden_sizes,
            gamma=PROTOCOL["gamma"],
            epsilon_start=PROTOCOL["epsilon_start"],
            epsilon_min=PROTOCOL["epsilon_min"],
            epsilon_decay=PROTOCOL["epsilon_decay"],
            use_double_dqn=True,
            target_update_freq=PROTOCOL["target_update_freq"],
        )
        return SPSAClassicalDQN(config=cfg, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
#  Unified training loop (with diagnostic hooks)
# ═══════════════════════════════════════════════════════════════════════

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

    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
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

    L_MIN = PROTOCOL["L_MIN"]
    CUT_PENALTY = PROTOCOL["CUT_PENALTY"]
    EXTEND_COST = PROTOCOL["EXTEND_COST"]
    COMPLEXITY_LAMBDA = PROTOCOL["COMPLEXITY_LAMBDA"]
    batch_size = PROTOCOL["batch_size"]

    # Tracking
    all_rewards = []
    val_crs, val_cut_pcts, val_seg_counts = [], [], []
    q_margins = []          # D2: mean(Q_extend - Q_cut) per epoch (all val steps)
    replay_cut_pcts = []    # D3: CUT ratio in training actions per epoch
    replay_buf_cut_pcts = []  # D5: CUT ratio in actual replay buffer
    best_bundle = {"val_cr": float("inf"), "cut_pct": 0.0,
                   "n_segs": 0, "epoch": -1, "avg_reward": 0.0}

    start_time = time.time()

    print(f"\n{'─'*55}")
    print(f"  {spec.name}  ({agent.n_params} params)")
    print(f"  Noise: {spec.noise_model} | Shots: {spec.shots}")
    print(f"  Mode: {spec.run_type} | Data: {spec.data_fraction:.0%}")
    print(f"  Epochs: {n_epochs} × {scheduler.active_training_size} trajectories")
    print(f"{'─'*55}")

    for epoch in range(n_epochs):
        idxlist = scheduler.sample_epoch()
        epoch_rewards = []
        epoch_cuts_in_training = 0
        epoch_extends_in_training = 0
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

            for index in range(1, steps):
                done = (index == steps - 1)
                action = agent.act(observation)

                observation_, reward = env.step(episode, action, index, "T")
                actual_action = env._last_action
                raw_split_od_next = observation_.flatten()[1]

                if actual_action == 0 and reward == 0:
                    reward = raw_split_od - raw_split_od_next

                reward = scale_reward(reward)

                if actual_action == 0:
                    reward -= EXTEND_COST
                    epoch_extends_in_training += 1
                if actual_action == 1:
                    reward -= CUT_PENALTY
                    n_cuts += 1
                    epoch_cuts_in_training += 1
                n_steps += 1

                raw_split_od = raw_split_od_next
                observation_ = normalize_obs(observation_)

                episode_reward += reward
                replay.add(observation.flatten(), actual_action, reward,
                           observation_.flatten(), done)

                if done:
                    break

                if replay.is_ready(batch_size):
                    states, actions, rewards_b, next_states, dones = \
                        replay.sample_batch(batch_size)
                    agent.update(states, actions, rewards_b, next_states, dones)

                observation = observation_

            # Complexity regularizer
            if n_steps > 0:
                cut_rate = n_cuts / n_steps
                episode_reward -= COMPLEXITY_LAMBDA * cut_rate

            all_rewards.append(episode_reward)
            epoch_rewards.append(episode_reward)
            agent.decay_epsilon()

        # ── End-of-epoch validation (SINGLE PASS — fixed) ──────
        # Collects: cut/extend counts, per-episode segs, Q-margins
        env.allsubtrajs_E = []
        val_n_extend, val_n_cut = 0, 0
        val_segs = 0
        q_extend_vals, q_cut_vals = [], []
        has_q = hasattr(agent, 'get_q_values')

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
            val_cr = float(val_od / env.basesim_E)
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')
        val_crs.append(val_cr)

        val_total = val_n_extend + val_n_cut
        cut_pct = 100 * val_n_cut / val_total if val_total else 0
        val_cut_pcts.append(cut_pct)
        val_seg_counts.append(val_segs)

        # D2: Q-margin (mean across ALL val steps this epoch)
        if q_extend_vals:
            margin = float(np.mean(q_extend_vals) - np.mean(q_cut_vals))
            q_margins.append(margin)
        else:
            q_margins.append(0.0)

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
                "n_segs": val_segs, "epoch": epoch + 1,
                "avg_reward": float(np.mean(epoch_rewards)),
            }

        # Q diagnostic: use first real val observation (not arbitrary)
        q_diag = ""
        if has_q and q_extend_vals:
            q_diag = (f" | Q̄ext={np.mean(q_extend_vals):+.3f}"
                      f" Q̄cut={np.mean(q_cut_vals):+.3f}")

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: "
              f"ValCR={val_cr:.4f} | "
              f"R̄={np.mean(epoch_rewards):+.3f} | "
              f"CUT={cut_pct:.0f}% | "
              f"#segs={val_segs} | "
              f"ε={agent.epsilon:.3f} | "
              f"Qmargin={q_margins[-1]:+.4f} | "
              f"ReplayCUT={rp_cut_pct:.0f}% | "
              f"BufCUT={buf_cut_pct:.0f}%{improved}{q_diag}")

        # Reset eval clusters
        for i in env.clusters_E.keys():
            env.clusters_E[i][0] = []
            env.clusters_E[i][1] = []
            env.clusters_E[i][3] = defaultdict(list)

        env.update_cluster("T")
        scheduler.update()

    elapsed = time.time() - start_time

    return {
        "model": spec.name,
        "kind": spec.kind,
        "noise": spec.noise_model,
        "params": agent.n_params,
        "run_type": spec.run_type,
        "data_fraction": spec.data_fraction,
        # Best-epoch bundle
        "val_cr": best_bundle["val_cr"],
        "cut_pct": best_bundle["cut_pct"],
        "n_segs": best_bundle["n_segs"],
        "best_epoch": best_bundle["epoch"],
        # Final-epoch metrics
        "final_val_cr": val_crs[-1] if val_crs else float('inf'),
        "final_cut_pct": val_cut_pcts[-1] if val_cut_pcts else 0.0,
        "final_n_segs": val_seg_counts[-1] if val_seg_counts else 0,
        # Per-epoch series
        "val_crs": val_crs,
        "val_cut_pcts": val_cut_pcts,
        "val_seg_counts": val_seg_counts,
        "q_margins": q_margins,
        "replay_cut_pcts": replay_cut_pcts,
        "replay_buf_cut_pcts": replay_buf_cut_pcts,
        "all_rewards": [float(r) for r in all_rewards],
        # Timing
        "wall_time": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

def get_e1_specs():
    """E1 — Core Quantum Utility (all models, noiseless)."""
    return [
        ModelSpec("VQ-DQN (5q×3L)",     "quantum", n_layers=3),
        # SPSA-optimized controls (same optimizer as quantum)
        ModelSpec("Control A (linear)",  "classical", hidden_sizes=[]),
        ModelSpec("Control B (h=64)",    "classical", hidden_sizes=[64]),
        ModelSpec("Control C (h=32×32)", "classical", hidden_sizes=[32, 32]),
        # Adam-optimized controls (backprop — removes SPSA handicap objection)
        ModelSpec("Control D (Adam linear)",  "adam", hidden_sizes=[]),
        ModelSpec("Control E (Adam h=64)",    "adam", hidden_sizes=[64]),
        ModelSpec("Control F (Adam h=32×32)", "adam", hidden_sizes=[32, 32]),
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
            val_cr = float(val_od / env.basesim_E)
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')

        # Normalized variants
        try:
            val_pp = compute_overdist_per_point(env.clusters_E)
            n_val_cr = float(val_pp / env.basesim_E)
        except (ZeroDivisionError, ValueError):
            n_val_cr = float('inf')

        try:
            val_lw = compute_overdist_length_weighted(env.clusters_E)
            w_val_cr = float(val_lw / env.basesim_E)
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
            val_cr = float(val_od / env.basesim_E)
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
    print(f"\n{'═'*90}")
    print(f"  {experiment} — Summary")
    print(f"{'═'*90}\n")

    header = (f"{'Model':<30s} {'Params':>6s} {'ValCR':>8s} "
              f"{'CUT%':>6s} {'#Segs':>6s} {'BestEp':>6s} "
              f"{'Time':>7s} {'Qmargin':>8s}")
    print(header)
    print("─" * 90)

    for r in all_results:
        qm = r.get("q_margins", [])
        qm_str = f"{qm[-1]:+.4f}" if qm else "N/A"
        print(f"{r['model']:<30s} "
              f"{r['params']:>6d} "
              f"{r['val_cr']:>8.4f} "
              f"{r['cut_pct']:>5.0f}% "
              f"{r['n_segs']:>6d} "
              f"{r.get('best_epoch', 0):>6d} "
              f"{r['wall_time']:>6.1f}s "
              f"{qm_str:>8s}")

    print("─" * 90)


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

        # Aggregate across seeds
        crs = [r["val_cr"] for r in per_seed_results]
        cuts = [r["cut_pct"] for r in per_seed_results]
        segs = [r["n_segs"] for r in per_seed_results]
        times = [r["wall_time"] for r in per_seed_results]
        qms = [r["q_margins"][-1] for r in per_seed_results if r.get("q_margins")]

        agg = per_seed_results[0].copy()
        agg["val_cr"] = float(np.mean(crs))
        agg["val_cr_std"] = float(np.std(crs))
        agg["cut_pct"] = float(np.mean(cuts))
        agg["cut_pct_std"] = float(np.std(cuts))
        agg["n_segs"] = int(np.mean(segs))
        agg["wall_time"] = float(np.sum(times))
        agg["n_seeds"] = len(seeds)
        agg["per_seed_crs"] = crs
        agg["per_seed_cuts"] = cuts
        if qms:
            agg["q_margins"] = [float(np.mean(qms))]
            agg["q_margin_std"] = float(np.std(qms))
        aggregated.append(agg)

    return aggregated


def print_multi_seed_table(results: List[Dict], experiment: str):
    """Print summary table with mean±std for multi-seed results."""
    print(f"\n{'═'*100}")
    print(f"  {experiment} — Multi-Seed Summary ({results[0].get('n_seeds', 1)} seeds)")
    print(f"{'═'*100}\n")

    header = (f"{'Model':<30s} {'Params':>6s} {'ValCR':>14s} "
              f"{'CUT%':>12s} {'#Segs':>6s} {'Time':>8s} {'Qmargin':>14s}")
    print(header)
    print("─" * 100)

    for r in results:
        cr_str = f"{r['val_cr']:.4f}±{r.get('val_cr_std', 0):.4f}"
        cut_str = f"{r['cut_pct']:.0f}%±{r.get('cut_pct_std', 0):.0f}%"
        qm = r.get("q_margins", [])
        qm_std = r.get("q_margin_std", 0)
        qm_str = f"{qm[-1]:+.3f}±{qm_std:.3f}" if qm else "N/A"
        print(f"{r['model']:<30s} "
              f"{r['params']:>6d} "
              f"{cr_str:>14s} "
              f"{cut_str:>12s} "
              f"{r['n_segs']:>6d} "
              f"{r['wall_time']:>7.0f}s "
              f"{qm_str:>14s}")

    print("─" * 100)


def print_pareto_table(d1_results, agent_results: List[Dict]):
    """Print Pareto-constrained ValCR table.

    Shows best ValCR from random policy AND learned agents at CUT ≤ thresholds.
    """
    thresholds = [5, 10, 20, 30, 40, 50, 80]
    print(f"\n{'═'*80}")
    print("  Pareto: Best ValCR at CUT ≤ threshold")
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
        ax.set_title("D1: ValCR vs CUT% (Random Policy)")
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
        ax.set_title("Pareto Frontier: Learned Agents vs Random Baseline")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(str(plot_dir / "pareto_valcr_vs_cut.png"), dpi=150)
        plt.close(fig)
        n_plots += 1
        print(f"  ✓ pareto_valcr_vs_cut.png")

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
        ax.set_title(f"{exp_name}: Validation CR Comparison")
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
    "D2": "Q-margin profiling (piggybacked on E1-E6)",
    "D3": "Replay training-action distribution (piggybacked on E1-E6)",
    "D4": "Policy basin test (forced policies under drift)",
    "D5": "Replay buffer histogram (piggybacked on E1-E6)",
    "E1": "Core Quantum Utility",
    "E2": "NISQ Viability",
    "E3": "Shot Sensitivity",
    "E4": "Drift Resilience",
    "E5": "Low-Data Generalization",
    "E6": "Version Progression",
    "S1": "Scalability Timing",
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

        # ── D4: Policy basin test ────────────────────────────────
        if "D4" in selected:
            d4_n = min(args.amount, 100)
            d4_result = run_d4_policy_basin(
                args.traj_path, args.centers_path, d4_n, args.seed)
            all_results["D4"] = d4_result

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

        # ── S1: Scalability ──────────────────────────────────────
        if "S1" in selected:
            s1_result = run_s1_scalability(
                args.traj_path, args.centers_path, args.seed)
            all_results["S1"] = s1_result

        # ── Grand summary ────────────────────────────────────────
        print(f"\n{'═'*70}")
        print(f"  GRAND SUMMARY")
        print(f"{'═'*70}\n")

        all_model_results = []
        for key in ["E1", "E2", "E3", "E4", "E5", "E6"]:
            data = all_results.get(key)
            if isinstance(data, list):
                all_model_results.extend(data)

        if all_model_results:
            print_summary_table(all_model_results, "All Models")

            # Pareto table (D1 + learned agents)
            d1_data = all_results.get("D1")
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
