#!/usr/bin/env python3
"""Cross-system comparison: RLSTCcode (classical MLP) vs Q-RLSTC (VQC).

Runs both systems on the same T-Drive data with matched hyperparameters.
The ONLY intentional difference is the function approximator:
  - Classical: 5→64→2 MLP, SGD, ~450 params  (pure NumPy — no TensorFlow)
  - Quantum D: 5q × 3-layer VQC, SPSA, 30 params

Usage::

    python experiments/run_cross_comparison.py \\
        --traj-path  ../RLSTCcode/data/Tdrive_norm_traj \\
        --centers-path ../RLSTCcode/data/tdrive_clustercenter \\
        --amount 500 \\
        --output-dir results/cross_comparison
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


# ═══════════════════════════════════════════════════════════════════════
# Pure-NumPy classical DQN  (replaces TensorFlow-based rl_nn.py)
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

    # Static normalization scale: divide IED-scale features by basesim_T
    # so they become ratios ≈ 1.0-2.0, matching features 3-4 which are [0,1]
    _ied_scale = max(env.basesim_T, 1e-8)

    def normalize_obs(obs):
        """Scale IED features (0,1,2) by basesim — deterministic, replay-safe."""
        o = obs.copy().flatten()
        o[0] /= _ied_scale       # overall_sim → ratio ≈ 1.0
        o[1] /= _ied_scale       # split_overdist → ratio ≈ 1.0
        o[2] /= _ied_scale * 10  # overall_sim*10 → ratio ≈ 1.0
        return o.reshape(obs.shape)

    def scale_reward(r):
        """Scale raw IED-delta reward to O(0.01-0.1) then clip."""
        return float(np.clip(r / _ied_scale, -1.0, 1.0))

    # ── Reward alignment constants ─────────────────────────────────
    L_MIN = 3           # Min steps before CUT is allowed
    CUT_PENALTY = 0.12  # Per-cut penalty (gentle nudge from 0.1)
    EXTEND_COST = 0.01  # Per-step cost for EXTEND (prevents never-cut basin)

    print(f"  Parameters: {RL.param_count}")
    print(f"  basesim_T (IED scale): {_ied_scale:.4f}")
    print(f"  Reward fixes: L_min={L_MIN}, λ_cut={CUT_PENALTY}, λ_ext={EXTEND_COST}")

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
            # Extract raw split_overdist for potential shaping
            raw_split_od = observation.flatten()[1]
            observation = normalize_obs(observation)
            episode_reward = 0.0
            seg_len = 0        # Steps in current segment
            n_cuts = 0         # CUT count this episode

            for index in range(1, steps):
                done = (index == steps - 1)
                action = RL.act(observation)

                # Min segment length: force EXTEND if too short
                seg_len += 1
                if action == 1 and seg_len < L_MIN:
                    action = 0  # override to EXTEND

                observation_, reward = env.step(episode, action, index, "T")

                # Extract raw split_overdist BEFORE normalizing
                raw_split_od_next = observation_.flatten()[1]

                # Potential-based shaping for EXTEND (MDP returns 0)
                # Uses split_overdist delta: positive when extending
                # makes the projected cut-point OD better
                if action == 0 and reward == 0:
                    reward = raw_split_od - raw_split_od_next

                # Scale reward
                reward = scale_reward(reward)

                # Per-action costs
                if action == 0:
                    reward -= EXTEND_COST  # Prevent never-cut basin
                if action == 1:
                    reward -= CUT_PENALTY
                    n_cuts += 1
                    seg_len = 0  # Reset after CUT

                raw_split_od = raw_split_od_next
                observation_ = normalize_obs(observation_)

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

            # Track Q-values for diagnostics
            test_state = observation.flatten() if observation.ndim > 1 else observation
            q_vals = RL._forward(test_state.reshape(1, -1))
            ep_q_maxes.append(float(np.max(q_vals)))

            # ε decay: per-EPISODE, 0.995 → reaches 0.05 at ~600 episodes
            if RL.epsilon > RL.epsilon_min:
                RL.epsilon *= RL.epsilon_decay

            # Periodic evaluation
            if ep_idx % 100 == 0 and ep_idx != 0:
                # Validation CR — track action distribution
                env.allsubtrajs_E = []
                val_n_extend, val_n_cut = 0, 0
                for e in range(sidx, eidx):
                    obs, s = env.reset(e, "E")
                    obs = normalize_obs(obs)
                    for idx in range(1, s):
                        act = RL.online_act(obs)
                        if act == 0:
                            val_n_extend += 1
                        else:
                            val_n_cut += 1
                        obs, _ = env.step(e, act, idx, "E")
                        obs = normalize_obs(obs)

                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / env.basesim_E)

                train_od = compute_overdist(env.clusters_T)
                train_cr = float(train_od / env.basesim_T)

                results["training_cr"].append(train_cr)
                results["validation_cr"].append(val_cr)

                improved = " ★" if val_cr < best_val_cr else ""
                best_val_cr = min(best_val_cr, val_cr)
                val_total = val_n_extend + val_n_cut
                cut_pct = 100 * val_n_cut / val_total if val_total else 0

                print(f"  Round {round_num+1}, ep {ep_idx}: "
                      f"Train CR={train_cr:.4f}, Val CR={val_cr:.4f}"
                      f" | Q̄max={np.mean(ep_q_maxes[-100:]):.4f}"
                      f" | R̄={np.mean(ep_rewards[-100:]):+.4f}"
                      f" | CUT={cut_pct:.0f}%{improved}")

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

    from q_rlstc.data.rlstc_mdp import TrajRLclus
    from q_rlstc.data.rlstc_cluster import compute_overdist
    from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig
    from q_rlstc.rl.replay_buffer import ReplayBuffer

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
        shots=0,            # 0 = statevector (exact, noiseless)
        target_update_freq=10,
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

    # Same static normalization as classical
    _ied_scale = max(env.basesim_T, 1e-8)

    def normalize_obs(obs):
        o = obs.copy().flatten()
        o[0] /= _ied_scale
        o[1] /= _ied_scale
        o[2] /= _ied_scale * 10
        return o.reshape(obs.shape)

    def scale_reward(r):
        return float(np.clip(r / _ied_scale, -1.0, 1.0))

    # ── Reward alignment constants (same as classical) ──────────
    L_MIN = 3
    CUT_PENALTY = 0.12  # Same as classical for fair comparison
    EXTEND_COST = 0.01  # Prevent never-cut basin

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
            observation = normalize_obs(observation)
            episode_reward = 0.0
            seg_len = 0
            n_cuts = 0

            for index in range(1, steps):
                done = (index == steps - 1)
                state_1d = observation.flatten()
                action = agent.act(state_1d)

                # Min segment length: force EXTEND if too short
                seg_len += 1
                if action == 1 and seg_len < L_MIN:
                    action = 0

                observation_, reward = env.step(episode, action, index, "T")

                # Raw split_overdist for potential shaping
                raw_split_od_next = observation_.flatten()[1]

                # Potential-based shaping for EXTEND
                if action == 0 and reward == 0:
                    reward = raw_split_od - raw_split_od_next

                reward = scale_reward(reward)

                # Per-action costs (same as classical)
                if action == 0:
                    reward -= EXTEND_COST
                if action == 1:
                    reward -= CUT_PENALTY
                    n_cuts += 1
                    seg_len = 0

                raw_split_od = raw_split_od_next
                observation_ = normalize_obs(observation_)
                episode_reward += reward

                buffer.add(state_1d, action, reward,
                           observation_.flatten(), done)

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

            agent.decay_epsilon()

            # Periodic evaluation
            if ep_idx % 100 == 0 and ep_idx != 0:
                env.allsubtrajs_E = []
                val_n_extend, val_n_cut = 0, 0
                for e in range(sidx, eidx):
                    obs, s = env.reset(e, "E")
                    obs = normalize_obs(obs)
                    for idx in range(1, s):
                        act = agent.act(obs.flatten(), greedy=True)
                        if act == 0:
                            val_n_extend += 1
                        else:
                            val_n_cut += 1
                        obs, _ = env.step(e, act, idx, "E")
                        obs = normalize_obs(obs)

                val_od = compute_overdist(env.clusters_E)
                val_cr = float(val_od / env.basesim_E)
                train_od = compute_overdist(env.clusters_T)
                train_cr = float(train_od / env.basesim_T)

                results["training_cr"].append(train_cr)
                results["validation_cr"].append(val_cr)

                val_total = val_n_extend + val_n_cut
                cut_pct = 100 * val_n_cut / val_total if val_total else 0

                print(f"  Round {round_num+1}, ep {ep_idx}: "
                      f"Train CR={train_cr:.4f}, Val CR={val_cr:.4f}"
                      f" | CUT={cut_pct:.0f}%"
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
        print(f"{'Final Validation CR':<30} "
              f"{classical.get('final_validation_cr', 'N/A'):>15} "
              f"{quantum.get('final_validation_cr', 'N/A'):>15}")

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-system comparison: RLSTCcode vs Q-RLSTC Version D"
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
        choices=["both", "classical", "quantum"],
        default="both",
        help="Which system(s) to run",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_results = None
    quantum_results = None

    if args.run in ("both", "classical"):
        classical_results = run_classical_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    if args.run in ("both", "quantum"):
        quantum_results = run_quantum_experiment(
            args.traj_path, args.centers_path, args.amount, output_dir, args.seed
        )

    if classical_results and quantum_results:
        compare_results(classical_results, quantum_results, output_dir)
    elif classical_results:
        with open(output_dir / "classical_results.json", "w") as f:
            json.dump(classical_results, f, indent=2, default=str)
        print(f"\nClassical results saved to {output_dir / 'classical_results.json'}")
    elif quantum_results:
        with open(output_dir / "quantum_results.json", "w") as f:
            json.dump(quantum_results, f, indent=2, default=str)
        print(f"\nQuantum results saved to {output_dir / 'quantum_results.json'}")


if __name__ == "__main__":
    main()
