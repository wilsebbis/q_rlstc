"""Soft-DQN / Entropy-Regularized Target Computation.

Implements the soft value function:

    V(s) = α · log( Σ_a exp(Q(s,a) / α) )

which is the log-sum-exp smoothing of the hard max.  When α → 0 this
recovers the standard max_a Q(s,a); when α > 0 it provides entropy
regularization that actively resists policy collapse (e.g. the
"never-cut" failure mode).

Usage:
    targets = soft_value(q_next, alpha=0.1)    # drop-in for max(Q)

The α parameter controls exploration greediness and should be tuned
or annealed during training (high α = more exploration, low α = greedy).

Reference:
    Haarnoja et al., "Soft Actor-Critic" (2018)
    Song et al., "Revisiting DQN with Soft Bellman" (2019)
"""

import numpy as np


def soft_value(
    q_values: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute entropy-regularized soft value from Q-values.

    V(s) = α · log( Σ_a exp(Q(s,a) / α) )

    Numerically stable via log-sum-exp trick:
        V(s) = M + α · log( Σ_a exp((Q(s,a) - M) / α) )
    where M = max_a Q(s,a).

    Args:
        q_values: Q-values of shape (batch, n_actions) or (n_actions,).
        alpha: Entropy temperature.  α → 0 recovers hard max.
            Must be > 0.

    Returns:
        Soft value of shape (batch,) or scalar.

    Raises:
        ValueError: If alpha <= 0.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    q = np.asarray(q_values, dtype=np.float64)
    scalar_input = q.ndim == 1

    if scalar_input:
        q = q[np.newaxis, :]  # (1, n_actions)

    # Log-sum-exp trick for numerical stability
    m = q.max(axis=1, keepdims=True)       # (batch, 1)
    shifted = (q - m) / alpha              # (batch, n_actions)
    lse = m.squeeze(-1) + alpha * np.log(np.sum(np.exp(shifted), axis=1))

    return float(lse[0]) if scalar_input else lse


def soft_policy(
    q_values: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Boltzmann policy derived from soft Q-values.

    π(a|s) = exp( (Q(s,a) - V(s)) / α )
           = softmax( Q(s,a) / α )

    Args:
        q_values: Q-values of shape (batch, n_actions) or (n_actions,).
        alpha: Entropy temperature.

    Returns:
        Action probabilities with same shape as q_values.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    q = np.asarray(q_values, dtype=np.float64)
    shifted = q / alpha
    shifted -= shifted.max(axis=-1, keepdims=True)  # numerical stability
    exp_q = np.exp(shifted)
    return exp_q / exp_q.sum(axis=-1, keepdims=True)
