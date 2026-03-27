#!/usr/bin/env python3
"""Cross-system comparison: RLSTCcode (classical MLP) vs Q-RLSTC (VQC).

Runs up to 4 agent types on the same T-Drive data with matched hyperparameters:
  - OriginalClassicalDQN: 5→64→2 MLP, SGD, ~458 params (faithful RLSTCcode)
  - AdamClassicalDQN:     5→64→2 MLP, Adam, ~514 params (modern optimizer)
  - SPSAClassicalDQN:     5→64→2 MLP, SPSA, ~514 params (same optimizer as VQ-DQN)
  - VQ-DQN:               5q × 3L VQC, SPSA, 34 params (quantum circuit)

Usage::

    python experiments/run_cross_comparison.py \\
        --traj-path  q_rlstc/data/Tdrive_norm_traj \\
        --centers-path q_rlstc/data/tdrive_clustercenter \\
        --amount 500 \\
        --output-dir results/cross_comparison \\
        --run all
"""

import argparse
import json
import os
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import random

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "q_rlstc" / "data"
sys.path.insert(0, str(_PROJECT_ROOT))

# Import the canonical OriginalClassicalDQN from the rl module
from q_rlstc.rl.original_classical_agent import OriginalClassicalDQN


# ═══════════════════════════════════════════════════════════════════════
# Legacy NumpyDQN kept for backward compatibility — delegates to OriginalClassicalDQN
# ═══════════════════════════════════════════════════════════════════════

class NumpyDQN:
    """Minimal DQN with a 5→64→2 MLP implemented in pure NumPy.

    Matches RLSTCcode's DeepQNetwork architecture exactly:
      - 1 hidden layer (64 units, ReLU)
      - Huber loss, SGD optimizer
      - ε-greedy exploration with decay
      - Target network with soft updates
      - Experience replay buffer (deque, maxlen=5000)
    """

    def __init__(self, state_size: int, action_size: int, seed: int = 1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995   # Per-EPISODE, not per-step
        self.learning_rate = 0.001
        self.memory = deque(maxlen=5000)
        self._last_td_error = 0.0    # For diagnostics

        rng = np.random.RandomState(seed)

        # Xavier initialisation (matches Keras Dense default)
        self.w1 = rng.randn(state_size, 64) * np.sqrt(2.0 / (state_size + 64))
        self.b1 = np.zeros(64)
        self.w2 = rng.randn(64, action_size) * np.sqrt(2.0 / (64 + action_size))
        self.b2 = np.zeros(action_size)

        # Target network (copy)
        self.tw1 = self.w1.copy()
        self.tb1 = self.b1.copy()
        self.tw2 = self.w2.copy()
        self.tb2 = self.b2.copy()

    # ── forward pass ──────────────────────────────────────────────

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def _forward(self, state, target=False):
        """Forward pass through the MLP. state: (batch, 5) or (1, 5)."""
        w1, b1, w2, b2 = (
            (self.tw1, self.tb1, self.tw2, self.tb2) if target
            else (self.w1, self.b1, self.w2, self.b2)
        )
        h = self._relu(state @ w1 + b1)
        return h @ w2 + b2  # (batch, action_size)

    # ── action selection ──────────────────────────────────────────

    def act(self, state):
        """ε-greedy action."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self._forward(state)
        return int(np.argmax(q[0]))

    def online_act(self, state):
        """Greedy action (no exploration)."""
        q = self._forward(state)
        return int(np.argmax(q[0]))

    # ── replay buffer ─────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ── training step ─────────────────────────────────────────────

    def replay(self, episode_unused, batch_size):
        """One SGD step with Double DQN and Huber loss."""
        minibatch = random.sample(self.memory, batch_size)

        states = np.vstack([t[0] for t in minibatch])       # (B, 5)
        actions = np.array([t[1] for t in minibatch])        # (B,)
        rewards = np.array([t[2] for t in minibatch])        # (B,)
        next_states = np.vstack([t[3] for t in minibatch])   # (B, 5)
        dones = np.array([t[4] for t in minibatch], float)   # (B,)

        # Current Q-values
        q_pred = self._forward(states)                       # (B, A)

        # Double DQN: online net selects action, target net evaluates
        q_online_next = self._forward(next_states, target=False)
        q_target_next = self._forward(next_states, target=True)
        targets = q_pred.copy()
        for i in range(batch_size):
            best_action = np.argmax(q_online_next[i])
            targets[i, actions[i]] = (
                rewards[i] + (1 - dones[i]) * self.gamma * q_target_next[i, best_action]
            )

        # Track TD error for diagnostics
        self._last_td_error = float(np.mean(np.abs(targets - q_pred)))

        # Backprop with Huber loss (clip_delta=1.0)
        self._sgd_step(states, targets, q_pred)

        # NOTE: ε decay is per-EPISODE in the training loop, NOT here

    def _sgd_step(self, states, targets, q_pred):
        """Manual SGD with Huber loss gradient + gradient clipping."""
        B = states.shape[0]

        # Huber error
        error = targets - q_pred  # (B, A)
        clip_delta = 1.0
        huber_grad = np.where(
            np.abs(error) <= clip_delta,
            -error,                          # quadratic region
            -clip_delta * np.sign(error),    # linear region
        ) / B

        # Backprop through linear output layer
        h = self._relu(states @ self.w1 + self.b1)  # (B, 64)
        dw2 = h.T @ huber_grad                       # (64, A)
        db2 = huber_grad.sum(axis=0)                  # (A,)

        # Backprop through ReLU + hidden layer
        dh = huber_grad @ self.w2.T                   # (B, 64)
        dh = dh * (h > 0).astype(float)               # ReLU mask
        dw1 = states.T @ dh                            # (5, 64)
        db1 = dh.sum(axis=0)                           # (64,)

        # Gradient clipping (max global norm = 10)
        grads = [dw1, db1, dw2, db2]
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if global_norm > 10.0:
            scale = 10.0 / global_norm
            dw1, db1, dw2, db2 = [g * scale for g in grads]

        # SGD update
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    # ── target network ────────────────────────────────────────────

    def update_target_model(self):
        self.tw1 = self.w1.copy()
        self.tb1 = self.b1.copy()
        self.tw2 = self.w2.copy()
        self.tb2 = self.b2.copy()

    def soft_update(self, tau):
        self.tw1 = tau * self.w1 + (1 - tau) * self.tw1
        self.tb1 = tau * self.b1 + (1 - tau) * self.tb1
        self.tw2 = tau * self.w2 + (1 - tau) * self.tw2
        self.tb2 = tau * self.b2 + (1 - tau) * self.tb2

    # ── persistence ───────────────────────────────────────────────

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, path):
        d = np.load(path)
        self.w1, self.b1 = d["w1"], d["b1"]
        self.w2, self.b2 = d["w2"], d["b2"]
        self.update_target_model()

    @property
    def param_count(self):
        return (self.w1.size + self.b1.size + self.w2.size + self.b2.size)


# ---------------------------------------------------------------------------
# Classical RLSTCcode runner
# ---------------------------------------------------------------------------

def run_classical_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 1,
) -> dict:
    """Run the classical RLSTCcode training pipeline.

    Uses the RLSTCcode MDP environment with a pure-NumPy DQN (no TensorFlow).

    Returns:
        Dict with training_cr, validation_cr, elapsed_time, param_count.
    """
    print("\n" + "=" * 60)
    print("  CLASSICAL EXPERIMENT (NumPy MLP 5→64→2)")
    print("=" * 60)

    # Import RLSTCcode modules (MDP + cluster are pure NumPy)
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist

    np.random.seed(seed)
    random.seed(seed)

    # Match RLSTCcode defaults
    validation_percent = 0.1
    sidx = int(amount * (1 - validation_percent))
    eidx = amount

    env = TrajRLclus(traj_path, centers_path, centers_path)
    RL = NumpyDQN(env.n_features, env.n_actions, seed=seed)

    from q_rlstc.data.observation_tracker import ObservationTracker
    from q_rlstc.rl.cmdp import CutBudgetConstraint, LagrangeMultiplier, constraint_cost
    from q_rlstc.rl.reward_shaping import base_geometric_reward, penalized_reward

    # ── CMDP Constraints & Observation Tracking ──────────
    obs_tracker = ObservationTracker(feature_dim=5, clip=3.0, warmup_steps=1000)
    budget_constraint = CutBudgetConstraint(beta=0.15) 
    lagrange = LagrangeMultiplier(init_lambda=0.12, lr_lambda=0.01, clamp_min=0.0, clamp_max=2.0)

    # ── Reward alignment constants ─────────────────────────────────
    L_MIN = 3           # Min steps before CUT is allowed
    EXTEND_COST = 0.01  # Per-step cost for EXTEND (prevents never-cut basin)

    print(f"  Parameters: {RL.param_count}")
    print(f"  Reward fixes: L_min={L_MIN}, λ_ext={EXTEND_COST}")

    # Training loop
    batch_size = 32
    n_rounds = 2
    results = {
        "system": "classical_rlstc",
        "param_count": RL.param_count,
        "training_cr": [],
        "validation_cr": [],
    }

    idxlist = list(range(amount))
    start_time = time.time()
    best_val_cr = float("inf")  # Lower is better

    for round_num in range(n_rounds):
        random.shuffle(idxlist)
        ep_rewards = []   # Per-episode reward accumulator
        ep_q_maxes = []   # Track max Q-values
        ep_seg_counts = []  # Segment count per episode
        round_cuts = 0
        round_actions = 0

        # ε reset at Round 2: re-enable exploration
        if round_num > 0:
            RL.epsilon = 0.5
            print(f"  ── Round {round_num+1}: ε reset to {RL.epsilon:.2f} ──")

        for ep_idx, episode in enumerate(idxlist):
            if ep_idx % 50 == 0:
                # Diagnostic summary
                diag = ""
                if ep_rewards:
                    avg_segs = np.mean(ep_seg_counts[-50:]) if ep_seg_counts else 0
                    diag = (f" | R̄={np.mean(ep_rewards[-50:]):+.4f}"
                            f" |TD|={RL._last_td_error:.4f}"
                            f" | segs={avg_segs:.1f}")
                print(f"  Round {round_num+1}/{n_rounds}, "
                      f"ep {ep_idx}/{amount}, "
                      f"ε={RL.epsilon:.3f}{diag}", flush=True)

            observation, steps = env.reset(episode, "T")
            raw_split_od = observation.flatten()[1]
            observation = obs_tracker.normalize(observation.flatten())
            episode_reward = 0.0
            seg_len = 0        # Steps in current segment
            n_cuts = 0         # CUT count this episode
            ep_actions = 0

            for index in range(1, steps):
                done = (index == steps - 1)
                
                # Reshape for the manual classical network
                action = RL.act(observation.reshape(1, -1))

                # Min segment length: force EXTEND if too short
                seg_len += 1
                if action == 1 and seg_len < L_MIN:
                    action = 0  # override to EXTEND

                observation_, raw_env_reward = env.step(episode, action, index, "T")

                raw_split_od_next = observation_.flatten()[1]

                base_geom = 0.0
                if action == 0 and raw_env_reward == 0:
                    base_geom = raw_split_od - raw_split_od_next
                else:
                    base_geom = raw_env_reward

                base_geom = float(np.clip(base_geom / max(env.basesim_T, 1e-8), -1.0, 1.0))

                # Apply CMDP penalty
                cost = constraint_cost(action)
                reward = penalized_reward(base_geom, cost, lagrange.value())

                # Per-action costs
                if action == 0:
                    reward -= EXTEND_COST  # Prevent never-cut basin
                if action == 1:
                    n_cuts += 1
                    seg_len = 0  # Reset after CUT
                    
                ep_actions += 1

                raw_split_od = raw_split_od_next
                observation_ = obs_tracker.normalize(observation_.flatten())

                episode_reward += reward
                RL.remember(observation, action, reward, observation_, done)
                if done:
                    break
                if len(RL.memory) > batch_size:
                    RL.replay(episode, batch_size)
                    RL.soft_update(0.05)
                observation = observation_

            ep_rewards.append(episode_reward)
            ep_seg_counts.append(n_cuts + 1)  # segments = cuts + 1
            round_cuts += n_cuts
            round_actions += ep_actions

            # Track Q-values for diagnostics
            test_state = observation.flatten() if observation.ndim > 1 else observation
            q_vals = RL._forward(test_state.reshape(1, -1))
            ep_q_maxes.append(float(np.max(q_vals)))

            # ε decay: per-EPISODE, 0.995 → reaches 0.05 at ~600 episodes
            if RL.epsilon > RL.epsilon_min:
                RL.epsilon *= RL.epsilon_decay

            # Periodic evaluation
            if ep_idx % 100 == 0 and ep_idx != 0:
                empirical_cut_rate = round_cuts / max(round_actions, 1)
                new_lambda = lagrange.update(empirical_cut_rate, budget_constraint)
                
                round_cuts = 0
                round_actions = 0
                
                # Validation CR — track action distribution
                env.allsubtrajs_E = []
                val_n_extend, val_n_cut = 0, 0
                for e in range(sidx, eidx):
                    obs, s = env.reset(e, "E")
                    obs = obs_tracker.normalize(obs.flatten(), update=False)
                    for idx in range(1, s):
                        act = RL.online_act(obs.reshape(1,-1))
                        if act == 0:
                            val_n_extend += 1
                        else:
                            val_n_cut += 1
                        obs, _ = env.step(e, act, idx, "E")
                        obs = obs_tracker.normalize(obs.flatten(), update=False)

                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / env.basesim_E)
                
                # New metrics tracking
                from q_rlstc.data.rlstc_cluster import compute_overdist_per_point, compute_overdist_length_weighted, compute_sse
                val_n_od = compute_overdist_per_point(env.clusters_E)
                val_nvalcr = float(val_n_od / env.basesim_E)
                val_w_od = compute_overdist_length_weighted(env.clusters_E)
                val_wvalcr = float(val_w_od / env.basesim_E)
                val_sse = float(compute_sse(env.clusters_E))
                
                # Segment tracking
                n_segments_val = sum(len(c[4]) for c in env.clusters_E.values() if len(c) > 4)

                train_od = compute_overdist(env.clusters_T)
                train_cr = float(train_od / env.basesim_T)

                results["training_cr"].append(train_cr)
                results["validation_cr"].append(val_cr)
                
                val_total = val_n_extend + val_n_cut
                cut_pct = 100 * val_n_cut / val_total if val_total else 0
                
                if "history" not in results:
                    results["history"] = []
                results["history"].append({
                    "ep": ep_idx,
                    "val_cr": val_cr,
                    "nvalcr": val_nvalcr,
                    "wvalcr": val_wvalcr,
                    "sse": val_sse,
                    "cut_pct": cut_pct,
                    "n_segments": n_segments_val
                })

                improved = " ★" if val_cr < best_val_cr else ""
                best_val_cr = min(best_val_cr, val_cr)
                val_total = val_n_extend + val_n_cut
                cut_pct = 100 * val_n_cut / val_total if val_total else 0
                emp_pct = 100 * empirical_cut_rate

                print(f"  Round {round_num+1}, ep {ep_idx}: "
                      f"Train CR={train_cr:.4f}, Val CR={val_cr:.4f}"
                      f" | Q̄max={np.mean(ep_q_maxes[-100:]):.4f}"
                      f" | R̄={np.mean(ep_rewards[-100:]):+.4f}"
                      f" | Val CUT={cut_pct:.0f}% (β={budget_constraint.beta*100:.0f}%, λ={new_lambda:.3f}){improved}")

                # Reset eval clusters
                for i in env.clusters_E.keys():
                    env.clusters_E[i][0] = []
                    env.clusters_E[i][1] = []
                    env.clusters_E[i][3] = defaultdict(list)

        env.update_cluster("T")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["final_training_cr"] = results["training_cr"][-1] if results["training_cr"] else None
    results["final_validation_cr"] = results["validation_cr"][-1] if results["validation_cr"] else None

    # Save model
    model_path = output_dir / "classical_model.npz"
    RL.save(str(model_path))
    results["model_path"] = str(model_path)

    print(f"\n  Classical done in {elapsed:.1f}s")
    print(f"  Final Train CR: {results['final_training_cr']}")
    print(f"  Final Val CR:   {results['final_validation_cr']}")

    return results


# ---------------------------------------------------------------------------
# Quantum Q-RLSTC runner (Version D, noiseless)
# ---------------------------------------------------------------------------

def run_quantum_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 42,
    adaptive_shots: bool = False,
) -> dict:
    """Run Q-RLSTC Version D on the same RLSTCcode MDP.

    Uses the VQ-DQN agent (5-qubit, 3-layer VQC, SPSA) with the
    *exact same* TrajRLclus environment and pickle data as the
    classical experiment—only the function approximator differs.

    Hyperparameters matched to classical:
      - gamma=0.99, epsilon decay 0.99→0.1
      - replay buffer 5000, batch_size=32
      - noiseless Aer statevector (exact Q-values)

    Returns:
        Dict with training_cr, validation_cr, elapsed_time, param_count.
    """
    print("\n" + "=" * 60)
    print("  QUANTUM EXPERIMENT (Q-RLSTC Version D — VQC 5q×3L)")
    print("=" * 60)

    # ── Agent setup (matches classical hyperparams) ───────────────
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig
    from q_rlstc.rl.replay_buffer import ReplayBuffer
    from q_rlstc.data.observation_tracker import ObservationTracker
    from q_rlstc.rl.cmdp import CutBudgetConstraint, LagrangeMultiplier, constraint_cost
    from q_rlstc.rl.reward_shaping import base_geometric_reward, penalized_reward

    np.random.seed(seed)
    random.seed(seed)

    # ── Agent setup (matches classical hyperparams) ───────────────
    agent_cfg = AgentConfig(
        version="D",
        n_qubits=5,
        n_layers=3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,   # Slower decay: reaches 0.1 at ep ~460
        shots=1024 if adaptive_shots else 0, # 0 = exact, 1024 = noisy base limit
        target_update_freq=10,
        adaptive_shots=adaptive_shots,
        confidence_delta=0.05,
    )
    agent = VQDQNAgent(config=agent_cfg, seed=seed)
    buffer = ReplayBuffer(max_size=5000, seed=seed)
    param_count = agent.n_params
    print(f"  Parameters: {param_count}")

    # ── RLSTCcode MDP (same pickle data as classical) ─────────────
    validation_percent = 0.1
    sidx = int(amount * (1 - validation_percent))
    eidx = amount

    env = TrajRLclus(traj_path, centers_path, centers_path)

    # ── CMDP Constraints & Observation Tracking ──────────
    # Normalize with moving z-score tracker instead of static max-scaling
    obs_tracker = ObservationTracker(feature_dim=5, clip=3.0, warmup_steps=1000)

    # CMDP logic instead of hardcoded CUT_PENALTY
    budget_constraint = CutBudgetConstraint(beta=0.15) # Target 15% budget
    lagrange = LagrangeMultiplier(init_lambda=0.12, lr_lambda=0.01, clamp_min=0.0, clamp_max=2.0)

    L_MIN = 3
    EXTEND_COST = 0.01

    # Quantum-specific: smaller batch + less frequent updates for speed
    batch_size = 4      # 4 instead of 32 → 8x fewer circuit evals in SPSA
    update_freq = 8     # Update every 8 steps instead of every step
    n_rounds = 2
    results = {
        "system": "quantum_qrlstc_v_d",
        "param_count": param_count,
        "training_cr": [],
        "validation_cr": [],
    }

    idxlist = list(range(amount))
    start_time = time.time()
    step_counter = 0

    for round_num in range(n_rounds):
        random.shuffle(idxlist)
        ep_rewards = []
        ep_seg_counts = []
        round_cuts = 0
        round_actions = 0

        # ε reset at Round 2: re-enable exploration
        if round_num > 0:
            agent.epsilon = 0.5
            print(f"  ── Round {round_num+1}: ε reset to {agent.epsilon:.2f} ──")

        for ep_idx, episode in enumerate(idxlist):
            if ep_idx % 50 == 0:
                diag = ""
                if ep_rewards:
                    avg_segs = np.mean(ep_seg_counts[-50:]) if ep_seg_counts else 0
                    diag = (f" | R̄={np.mean(ep_rewards[-50:]):+.4f}"
                            f" | segs={avg_segs:.1f}")
                print(f"  Round {round_num+1}/{n_rounds}, "
                      f"episode {ep_idx}/{amount}, "
                      f"ε={agent.epsilon:.3f}{diag}", flush=True)

            observation, steps = env.reset(episode, "T")
            raw_split_od = observation.flatten()[1]
            observation = obs_tracker.normalize(observation.flatten())
            episode_reward = 0.0
            seg_len = 0
            n_cuts = 0
            ep_actions = 0

            for index in range(1, steps):
                done = (index == steps - 1)
                
                # We expect the tracker to return 1D flat output so no flatten needed
                action = agent.act(observation)

                # Min segment length: force EXTEND if too short
                seg_len += 1
                if action == 1 and seg_len < L_MIN:
                    action = 0

                observation_, raw_env_reward = env.step(episode, action, index, "T")

                raw_split_od_next = observation_.flatten()[1]

                # Base geometric reward logic
                base_geom = 0.0
                if action == 0 and raw_env_reward == 0:
                    base_geom = raw_split_od - raw_split_od_next
                else:
                    base_geom = raw_env_reward
                
                # Z-normalize scale to keep neural math happy
                base_geom = float(np.clip(base_geom / max(env.basesim_T, 1e-8), -1.0, 1.0))
                
                # Apply CMDP penalty
                cost = constraint_cost(action)
                reward = penalized_reward(base_geom, cost, lagrange.value())

                if action == 0:
                    reward -= EXTEND_COST
                if action == 1:
                    n_cuts += 1
                    seg_len = 0
                    
                ep_actions += 1

                raw_split_od = raw_split_od_next
                observation_ = obs_tracker.normalize(observation_.flatten())
                episode_reward += reward

                buffer.add(observation, action, reward, observation_, done)

                if done:
                    break

                # SPSA update (every update_freq steps to save time)
                step_counter += 1
                if buffer.is_ready(batch_size) and step_counter % update_freq == 0:
                    states, actions, rewards_b, next_states, dones = \
                        buffer.sample_batch(batch_size)
                    # Batched update: targets computed internally
                    agent.update(states, actions, rewards_b, next_states, dones)

                observation = observation_

            ep_rewards.append(episode_reward)
            ep_seg_counts.append(n_cuts + 1)
            round_cuts += n_cuts
            round_actions += ep_actions

            agent.decay_epsilon()

            # Periodic evaluation
            if ep_idx % 100 == 0 and ep_idx != 0:
                # CMDP Slow Timescale Update 
                empirical_cut_rate = round_cuts / max(round_actions, 1)
                new_lambda = lagrange.update(empirical_cut_rate, budget_constraint)
                
                # Reset accumulators for next dual evaluation period
                round_cuts = 0
                round_actions = 0
                
                env.allsubtrajs_E = []
                val_n_extend, val_n_cut = 0, 0
                for e in range(sidx, eidx):
                    obs, s = env.reset(e, "E")
                    obs = obs_tracker.normalize(obs.flatten(), update=False)
                    for idx in range(1, s):
                        act = agent.act(obs, greedy=True)
                        if act == 0:
                            val_n_extend += 1
                        else:
                            val_n_cut += 1
                        obs, _ = env.step(e, act, idx, "E")
                        obs = obs_tracker.normalize(obs.flatten(), update=False)

                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / env.basesim_E)
                train_od = compute_overdist(env.clusters_T)
                train_cr = float(train_od / env.basesim_T)

                results["training_cr"].append(train_cr)
                results["validation_cr"].append(val_cr)

                val_total = val_n_extend + val_n_cut
                cut_pct = 100 * val_n_cut / val_total if val_total else 0
                emp_pct = 100 * empirical_cut_rate

                print(f"  Round {round_num+1}, ep {ep_idx}: "
                      f"Train CR={train_cr:.4f}, Val CR={val_cr:.4f}"
                      f" | Val CUT={cut_pct:.1f}% (β={budget_constraint.beta*100:.0f}%, λ={new_lambda:.3f})"
                      f" | R̄={np.mean(ep_rewards[-100:]):+.4f}")

                for i in env.clusters_E.keys():
                    env.clusters_E[i][0] = []
                    env.clusters_E[i][1] = []
                    env.clusters_E[i][3] = defaultdict(list)

        env.update_cluster("T")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["final_training_cr"] = (
        results["training_cr"][-1] if results["training_cr"] else None)
    results["final_validation_cr"] = (
        results["validation_cr"][-1] if results["validation_cr"] else None)

    # Save checkpoint
    ckpt_path = output_dir / "quantum_agent.npz"
    agent.save_checkpoint(str(ckpt_path))
    results["checkpoint_path"] = str(ckpt_path)

    print(f"\n  Quantum done in {elapsed:.1f}s")
    print(f"  Final Train CR: {results['final_training_cr']}")
    print(f"  Final Val CR:   {results['final_validation_cr']}")

    return results


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

def compare_results(classical: dict, quantum: dict, output_dir: Path):
    """Compare and report on both experiments."""
    print("\n" + "=" * 60)
    print("  CROSS-SYSTEM COMPARISON")
    print("=" * 60)

    report = {
        "classical": classical,
        "quantum": quantum,
    }

    # Summary table
    print(f"\n{'Metric':<30} {'Classical':>15} {'Quantum D':>15}")
    print("-" * 60)
    print(f"{'Parameters':<30} {classical['param_count']:>15} {quantum['param_count']:>15}")
    print(f"{'Parameter ratio':<30} {'1.0×':>15} "
          f"{classical['param_count']/max(quantum['param_count'],1):.1f}× fewer")
    print(f"{'Training time (s)':<30} {classical.get('elapsed_time',0):>15.1f} "
          f"{quantum.get('elapsed_time',0):>15.1f}")

    if classical.get("final_validation_cr") and quantum.get("final_validation_cr"):
        c_cr = classical["final_validation_cr"]
        q_cr = quantum["final_validation_cr"]
        diff_pct = (q_cr - c_cr) / c_cr * 100 if c_cr != 0 else float("inf")
        print(f"{'Final Validation CR':<30} {c_cr:>15.4f} {q_cr:>15.4f}")
        print(f"{'CR difference':<30} {'':>15} {diff_pct:>+14.2f}%")
    else:
        c_val = str(classical.get('final_validation_cr') or 'N/A')
        q_val = str(quantum.get('final_validation_cr') or 'N/A')
        print(f"{'Final Validation CR':<30} {c_val:>15} {q_val:>15}")

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_adam_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 1,
) -> dict:
    """Run AdamClassicalDQN (modern optimizer, same architecture)."""
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from q_rlstc.rl.adam_classical_agent import AdamClassicalDQN, AdamAgentConfig
    from q_rlstc.rl.replay_buffer import ReplayBuffer

    print("\n" + "=" * 60)
    print("  ADAM EXPERIMENT (Adam MLP 5→64→2)")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)

    env = TrajRLclus(traj_path, centers_path, centers_path)
    cfg = AdamAgentConfig(hidden_sizes=[64])
    agent = AdamClassicalDQN(config=cfg, seed=seed)
    buf = ReplayBuffer(max_size=5000, seed=seed)

    from q_rlstc.data.observation_tracker import ObservationTracker
    from q_rlstc.rl.cmdp import CutBudgetConstraint, LagrangeMultiplier, constraint_cost
    from q_rlstc.rl.reward_shaping import base_geometric_reward, penalized_reward

    obs_tracker = ObservationTracker(feature_dim=5, clip=3.0, warmup_steps=1000)
    budget_constraint = CutBudgetConstraint(beta=0.15) 
    lagrange = LagrangeMultiplier(init_lambda=0.12, lr_lambda=0.01, clamp_min=0.0, clamp_max=2.0)

    L_MIN, EXTEND_COST = 3, 0.01

    validation_pct = 0.1
    sidx = int(amount * (1 - validation_pct))
    n_rounds = 2
    batch_size = 32
    start_time = time.time()
    best_val_cr = float("inf")
    results = {"system": "adam_classical", "param_count": agent.n_params}

    for round_num in range(n_rounds):
        idxlist = list(range(amount))
        random.shuffle(idxlist)
        round_cuts = 0
        round_actions = 0
        for episode in idxlist:
            obs, steps = env.reset(episode, 'T')
            raw_split_od = obs.flatten()[1]
            obs = obs_tracker.normalize(obs.flatten())
            seg_len = 0
            n_cuts = 0
            ep_actions = 0
            for idx in range(1, steps):
                done = (idx == steps - 1)
                seg_len += 1
                action = agent.act(obs.reshape(1, -1))
                if action == 1 and seg_len < L_MIN:
                    action = 0
                obs_next, raw_r = env.step(episode, action, idx, 'T')
                raw_split_od_next = obs_next.flatten()[1]
                
                base_geom = 0.0
                if action == 0 and raw_r == 0:
                    base_geom = raw_split_od - raw_split_od_next
                else:
                    base_geom = raw_r

                base_geom = float(np.clip(base_geom / max(env.basesim_T, 1e-8), -1.0, 1.0))
                cost = constraint_cost(action)
                reward = penalized_reward(base_geom, cost, lagrange.value())
                
                if action == 1:
                    n_cuts += 1
                    seg_len = 0
                else:
                    reward -= EXTEND_COST
                    
                ep_actions += 1
                raw_split_od = raw_split_od_next
                obs_next = obs_tracker.normalize(obs_next.flatten())
                buf.add(obs, action, reward, obs_next, done)
                if done:
                    break
                if len(buf) >= batch_size:
                    batch = buf.sample(batch_size)
                    s = np.array([e.state for e in batch])
                    a = np.array([e.action for e in batch])
                    r = np.array([e.reward for e in batch])
                    ns = np.array([e.next_state for e in batch])
                    d = np.array([e.done for e in batch], dtype=float)
                    agent.update(s, a, r, ns, d)
                obs = obs_next
                
            round_cuts += n_cuts
            round_actions += ep_actions
            
        agent.decay_epsilon()
        
        empirical_cut_rate = round_cuts / max(round_actions, 1)
        new_lambda = lagrange.update(empirical_cut_rate, budget_constraint)
        round_cuts = 0
        round_actions = 0

        # Eval
        val_n_extend, val_n_cut = 0, 0
        for e in range(sidx, amount):
            o, steps = env.reset(e, 'E')
            o = obs_tracker.normalize(o.flatten(), update=False)
            for idx in range(1, steps):
                a = agent.act(o.reshape(1, -1), greedy=True)
                if a == 0:
                    val_n_extend += 1
                else:
                    val_n_cut += 1
                o, _ = env.step(e, a, idx, 'E')
                o = obs_tracker.normalize(o.flatten(), update=False)
        try:
            val_cr = float(compute_overdist(env.clusters_E) / env.basesim_E)
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')
            
        # New metrics tracking
        from q_rlstc.data.rlstc_cluster import compute_overdist_per_point, compute_overdist_length_weighted, compute_sse
        try:
            val_nvalcr = float(compute_overdist_per_point(env.clusters_E) / env.basesim_E)
            val_wvalcr = float(compute_overdist_length_weighted(env.clusters_E) / env.basesim_E)
            val_sse = float(compute_sse(env.clusters_E))
        except:
            val_nvalcr, val_wvalcr, val_sse = float('inf'), float('inf'), float('inf')
            
        n_segments_val = sum(len(c[4]) for c in env.clusters_E.values() if len(c) > 4)
            
        if val_cr < best_val_cr:
            best_val_cr = val_cr
            
        val_total = val_n_extend + val_n_cut
        cut_pct = 100 * val_n_cut / val_total if val_total else 0
        
        if "history" not in results:
            results["history"] = []
        results["history"].append({
            "ep": round_num,
            "val_cr": val_cr,
            "nvalcr": val_nvalcr,
            "wvalcr": val_wvalcr,
            "sse": val_sse,
            "cut_pct": cut_pct,
            "n_segments": n_segments_val
        })
        print(f"  Round {round_num}: val_cr={val_cr:.4f} | Val CUT={cut_pct:.0f}% (β={budget_constraint.beta*100:.0f}%, λ={new_lambda:.3f})")
        for i in env.clusters_E.keys():
            env.clusters_E[i][0], env.clusters_E[i][1] = [], []
            env.clusters_E[i][3] = defaultdict(list)
        env.update_cluster('T')

    results["final_validation_cr"] = best_val_cr
    results["elapsed_time"] = time.time() - start_time
    return results


def run_spsa_classical_experiment(
    traj_path: str,
    centers_path: str,
    amount: int,
    output_dir: Path,
    seed: int = 1,
) -> dict:
    """Run SPSAClassicalDQN (same optimizer as quantum)."""
    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from q_rlstc.rl.spsa_classical_agent import SPSAClassicalDQN, ClassicalAgentConfig
    from q_rlstc.rl.replay_buffer import ReplayBuffer

    print("\n" + "=" * 60)
    print("  SPSA CLASSICAL EXPERIMENT (SPSA MLP 5→64→2)")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)

    env = TrajRLclus(traj_path, centers_path, centers_path)
    cfg = ClassicalAgentConfig(hidden_sizes=[64])
    agent = SPSAClassicalDQN(config=cfg, seed=seed)
    buf = ReplayBuffer(max_size=5000, seed=seed)

    from q_rlstc.data.observation_tracker import ObservationTracker
    from q_rlstc.rl.cmdp import CutBudgetConstraint, LagrangeMultiplier, constraint_cost
    from q_rlstc.rl.reward_shaping import base_geometric_reward, penalized_reward

    obs_tracker = ObservationTracker(feature_dim=5, clip=3.0, warmup_steps=1000)
    budget_constraint = CutBudgetConstraint(beta=0.15) 
    lagrange = LagrangeMultiplier(init_lambda=0.12, lr_lambda=0.01, clamp_min=0.0, clamp_max=2.0)

    L_MIN, EXTEND_COST = 3, 0.01

    validation_pct = 0.1
    sidx = int(amount * (1 - validation_pct))
    n_rounds = 2
    batch_size = 32
    start_time = time.time()
    best_val_cr = float("inf")
    results = {"system": "spsa_classical", "param_count": agent.n_params}

    for round_num in range(n_rounds):
        idxlist = list(range(amount))
        random.shuffle(idxlist)
        round_cuts = 0
        round_actions = 0
        for episode in idxlist:
            obs, steps = env.reset(episode, 'T')
            raw_split_od = obs.flatten()[1]
            obs = obs_tracker.normalize(obs.flatten())
            seg_len = 0
            n_cuts = 0
            ep_actions = 0
            for idx in range(1, steps):
                done = (idx == steps - 1)
                seg_len += 1
                action = agent.act(obs.reshape(1, -1))
                if action == 1 and seg_len < L_MIN:
                    action = 0
                obs_next, raw_r = env.step(episode, action, idx, 'T')
                raw_split_od_next = obs_next.flatten()[1]
                
                base_geom = 0.0
                if action == 0 and raw_r == 0:
                    base_geom = raw_split_od - raw_split_od_next
                else:
                    base_geom = raw_r

                base_geom = float(np.clip(base_geom / max(env.basesim_T, 1e-8), -1.0, 1.0))
                cost = constraint_cost(action)
                reward = penalized_reward(base_geom, cost, lagrange.value())
                
                if action == 1:
                    n_cuts += 1
                    seg_len = 0
                else:
                    reward -= EXTEND_COST
                    
                ep_actions += 1
                raw_split_od = raw_split_od_next
                obs_next = obs_tracker.normalize(obs_next.flatten())
                buf.add(obs, action, reward, obs_next, done)
                if done:
                    break
                if len(buf) >= batch_size:
                    batch = buf.sample(batch_size)
                    s = np.array([e.state for e in batch])
                    a = np.array([e.action for e in batch])
                    r = np.array([e.reward for e in batch])
                    ns = np.array([e.next_state for e in batch])
                    d = np.array([e.done for e in batch], dtype=float)
                    agent.update(s, a, r, ns, d)
                obs = obs_next
                
            round_cuts += n_cuts
            round_actions += ep_actions
            
        agent.decay_epsilon()
        
        empirical_cut_rate = round_cuts / max(round_actions, 1)
        new_lambda = lagrange.update(empirical_cut_rate, budget_constraint)
        round_cuts = 0
        round_actions = 0

        val_n_extend, val_n_cut = 0, 0
        for e in range(sidx, amount):
            o, steps = env.reset(e, 'E')
            o = obs_tracker.normalize(o.flatten(), update=False)
            for idx in range(1, steps):
                a = agent.act(o.reshape(1, -1), greedy=True)
                if a == 0:
                    val_n_extend += 1
                else:
                    val_n_cut += 1
                o, _ = env.step(e, a, idx, 'E')
                o = obs_tracker.normalize(o.flatten(), update=False)
        try:
            val_cr = float(compute_overdist(env.clusters_E) / env.basesim_E)
        except (ZeroDivisionError, ValueError):
            val_cr = float('inf')
            
        # New metrics tracking
        from q_rlstc.data.rlstc_cluster import compute_overdist_per_point, compute_overdist_length_weighted, compute_sse
        try:
            val_nvalcr = float(compute_overdist_per_point(env.clusters_E) / env.basesim_E)
            val_wvalcr = float(compute_overdist_length_weighted(env.clusters_E) / env.basesim_E)
            val_sse = float(compute_sse(env.clusters_E))
        except:
            val_nvalcr, val_wvalcr, val_sse = float('inf'), float('inf'), float('inf')
            
        n_segments_val = sum(len(c[4]) for c in env.clusters_E.values() if len(c) > 4)

        if val_cr < best_val_cr:
            best_val_cr = val_cr
            
        val_total = val_n_extend + val_n_cut
        cut_pct = 100 * val_n_cut / val_total if val_total else 0
        
        if "history" not in results:
            results["history"] = []
        results["history"].append({
            "ep": round_num,
            "val_cr": val_cr,
            "nvalcr": val_nvalcr,
            "wvalcr": val_wvalcr,
            "sse": val_sse,
            "cut_pct": cut_pct,
            "n_segments": n_segments_val
        })
        val_total = val_n_extend + val_n_cut
        cut_pct = 100 * val_n_cut / val_total if val_total else 0
        print(f"  Round {round_num}: val_cr={val_cr:.4f} | Val CUT={cut_pct:.0f}% (β={budget_constraint.beta*100:.0f}%, λ={new_lambda:.3f})")
        for i in env.clusters_E.keys():
            env.clusters_E[i][0], env.clusters_E[i][1] = [], []
            env.clusters_E[i][3] = defaultdict(list)
        env.update_cluster('T')

    results["final_validation_cr"] = best_val_cr
    results["elapsed_time"] = time.time() - start_time
    return results


def compare_all_results(all_results: dict, output_dir: Path):
    """Compare and report on all agent experiments."""
    print("\n" + "=" * 80)
    print("  CROSS-SYSTEM COMPARISON — ALL AGENTS")
    print("=" * 80)

    print(f"\n{'Agent':<40s}  {'Params':>7s}  {'ValCR':>10s}  {'Time':>8s}")
    print("─" * 80)
    for key, data in all_results.items():
        name = data.get("system", key)
        params = data.get("param_count", "?")
        cr = data.get("final_validation_cr", "N/A")
        elapsed = data.get("elapsed_time", 0)
        cr_str = f"{cr:.4f}" if isinstance(cr, (int, float)) else str(cr)
        print(f"{name:<40s}  {params:>7}  {cr_str:>10s}  {elapsed:>7.1f}s")
    print("─" * 80)

    results_path = output_dir / "all_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-system comparison: RLSTCcode vs Q-RLSTC (all agents)"
    )
    parser.add_argument(
        "--traj-path",
        default=str(_DATA_DIR / "Tdrive_norm_traj"),
        help="Path to trajectory pickle (default: q_rlstc/data/Tdrive_norm_traj)",
    )
    parser.add_argument(
        "--centers-path",
        default=str(_DATA_DIR / "tdrive_clustercenter"),
        help="Path to cluster centers pickle (default: q_rlstc/data/tdrive_clustercenter)",
    )
    parser.add_argument("--amount", type=int, default=500)
    parser.add_argument("--output-dir", default="results/cross_comparison")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--run",
        choices=["all", "both", "classical", "quantum", "adam", "spsa_classical"],
        default="all",
        help="Which system(s) to run: all, both (classical+quantum), or individual",
    )
    parser.add_argument(
        "--adaptive-shots", action="store_true",
        help="Enable quantum hardware-aware Hoeffding allocation limits"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    run_set = args.run

    # Classical (OriginalClassicalDQN — faithful RLSTCcode)
    if run_set in ("all", "both", "classical"):
        all_results["classical"] = run_classical_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    # Adam (modern optimizer control)
    if run_set in ("all", "adam"):
        all_results["adam"] = run_adam_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    # SPSA Classical (same optimizer as quantum)
    if run_set in ("all", "spsa_classical"):
        all_results["spsa_classical"] = run_spsa_classical_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    # Quantum VQ-DQN
    if run_set in ("all", "both", "quantum"):
        all_results["quantum"] = run_quantum_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed,
            adaptive_shots=args.adaptive_shots
        )

    # Comparison
    if len(all_results) >= 2:
        compare_all_results(all_results, output_dir)
    elif len(all_results) == 1:
        key = list(all_results.keys())[0]
        path = output_dir / f"{key}_results.json"
        with open(path, "w") as f:
            json.dump(all_results[key], f, indent=2, default=str)
        print(f"\nResults saved to {path}")
    # Backward compat: also run old compare_results if both classical + quantum
    if "classical" in all_results and "quantum" in all_results:
        compare_results(all_results["classical"], all_results["quantum"], output_dir)


if __name__ == "__main__":
    main()
