"""SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Gradient-free optimization suitable for variational quantum circuits
where gradient evaluation is expensive.

Q-RLSTC 2.0: Adds Momentum-SPSA (m-SPSA) for faster convergence
by tracking a moving average of past gradients, enabling the system
to "blast through" bad gradient estimations from shot noise.

Q-RLSTC 2.1: Adds Common Random Numbers (CRN) option for reduced
variance — fixes measurement seed across (θ+cδ, θ-cδ) evaluations
so noise cancels in the gradient estimate.

Based on: J.C. Spall, "An Overview of the Simultaneous Perturbation Method
for Efficient Optimization" (1998)
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SPSAConfig:
    """Configuration for SPSA optimizer.
    
    Attributes:
        A: Stability constant for learning rate schedule.
        a: Initial learning rate scale.
        c: Initial perturbation magnitude.
        alpha: Learning rate decay exponent.
        gamma: Perturbation decay exponent.
        max_iter: Maximum iterations.
        seed: Random seed.
        use_momentum: Enable momentum-SPSA (m-SPSA).
        momentum: Momentum coefficient (β) for gradient averaging.
    """
    A: int = 20
    a: float = 0.12
    c: float = 0.08
    alpha: float = 0.602
    gamma: float = 0.101
    max_iter: int = 100
    seed: int = 42
    use_momentum: bool = True
    momentum: float = 0.9


class SPSAOptimizer:
    """SPSA optimizer for variational parameters.
    
    Uses simultaneous perturbation to estimate gradients with
    only 2 function evaluations per iteration.
    
    Q-RLSTC 2.0: Optional momentum-averaged gradients (m-SPSA).
    When enabled, tracks g̃_k = β·g̃_{k-1} + (1-β)·g_k and uses
    g̃_k for parameter updates. This smooths out noisy gradient
    estimates from quantum measurement shot noise.
    """
    
    def __init__(
        self,
        A: int = 20,
        a: float = 0.12,
        c: float = 0.08,
        alpha: float = 0.602,
        gamma: float = 0.101,
        max_grad_norm: float = 10.0,
        seed: int = 42,
        use_momentum: bool = False,
        momentum: float = 0.9,
        n_perturbations: int = 1,
        use_crn: bool = False,
        crn_base_seed: int = 0,
        param_scales: Optional[np.ndarray] = None,
    ):
        """Initialize SPSA optimizer.
        
        Args:
            A: Stability constant (typically 10-20% of max iterations).
            a: Initial learning rate scale.
            c: Initial perturbation magnitude.
            alpha: Learning rate decay exponent (theory: 1.0, practice: ~0.6).
            gamma: Perturbation decay exponent (theory: 1/6, practice: ~0.1).
            max_grad_norm: Maximum gradient norm for clipping (NISQ noise robustness).
            seed: Random seed for perturbations.
            use_momentum: Enable momentum-SPSA.
            momentum: Momentum coefficient β (0.9 typical).
            n_perturbations: Number of independent gradient estimates to average.
            use_crn: Common Random Numbers — fix measurement seed across
                (θ+cδ, θ-cδ) so shot noise cancels in the gradient.
            crn_base_seed: Base seed for CRN (only used when use_crn=True).
            param_scales: Per-parameter perturbation scaling (shape: n_params).
                If provided, δ is element-wise multiplied by param_scales.
                Use to apply larger perturbations to circuit angles vs. output
                weights. Default: uniform (all 1.0).
        """
        self.A = A
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        
        # Momentum-SPSA (m-SPSA)
        self.use_momentum = use_momentum
        self.beta = momentum
        self._momentum_buffer: Optional[np.ndarray] = None
        
        # Averaged SPSA (K-sample per step)
        self.n_perturbations = max(1, n_perturbations)
        
        # Common Random Numbers (CRN)
        self.use_crn = use_crn
        self.crn_base_seed = crn_base_seed
        
        # Per-parameter perturbation scaling
        self._param_scales = param_scales  # set externally or None
        self._current_crn_seed = None  # set during compute_gradient for CRN
    
    def _get_learning_rate(self, k: int) -> float:
        """Compute learning rate for iteration k.
        
        a_k = a / (A + k + 1)^alpha
        """
        return self.a / ((self.A + k + 1) ** self.alpha)
    
    def _get_perturbation_magnitude(self, k: int) -> float:
        """Compute perturbation magnitude for iteration k.
        
        c_k = c / (k + 1)^gamma
        """
        return self.c / ((k + 1) ** self.gamma)
    
    def _sample_perturbation(self, n_params: int) -> np.ndarray:
        """Sample Bernoulli ±1 perturbation vector."""
        return self.rng.choice([-1, 1], size=n_params).astype(np.float64)
    
    def compute_gradient(
        self,
        loss_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> np.ndarray:
        """Estimate gradient using SPSA, optionally with momentum.
        
        Uses two-sided finite difference with simultaneous perturbation.
        If momentum is enabled, returns the exponentially averaged gradient.
        
        Args:
            loss_fn: Function that takes params and returns loss.
            params: Current parameter vector.
        
        Returns:
            Estimated gradient vector (momentum-averaged if enabled).
        """
        n_params = len(params)
        c_k = self._get_perturbation_magnitude(self.iteration)
        
        # Per-parameter scaling if provided
        scales = self._param_scales
        if scales is not None:
            scales = np.asarray(scales)
            if len(scales) != n_params:
                scales = None  # fallback if mismatch
        
        # Averaged SPSA: K independent gradient estimates
        K = self.n_perturbations
        if K == 1:
            delta = self._sample_perturbation(n_params)
            if scales is not None:
                delta = delta * scales
            # CRN: set deterministic seed for +/- evaluations so shot noise cancels
            if self.use_crn:
                self._current_crn_seed = hash((self.crn_base_seed, self.iteration, 0, 1)) % (2**31)
            loss_plus = loss_fn(params + c_k * delta)
            if self.use_crn:
                self._current_crn_seed = hash((self.crn_base_seed, self.iteration, 0, -1)) % (2**31)
            loss_minus = loss_fn(params - c_k * delta)
            self._current_crn_seed = None
            raw_gradient = (loss_plus - loss_minus) / (2 * c_k * delta)
        else:
            grad_sum = np.zeros(n_params)
            for ki in range(K):
                delta = self._sample_perturbation(n_params)
                if scales is not None:
                    delta = delta * scales
                if self.use_crn:
                    self._current_crn_seed = hash((self.crn_base_seed, self.iteration, ki, 1)) % (2**31)
                loss_plus = loss_fn(params + c_k * delta)
                if self.use_crn:
                    self._current_crn_seed = hash((self.crn_base_seed, self.iteration, ki, -1)) % (2**31)
                loss_minus = loss_fn(params - c_k * delta)
                grad_sum += (loss_plus - loss_minus) / (2 * c_k * delta)
            self._current_crn_seed = None
            raw_gradient = grad_sum / K
        
        # Apply momentum averaging if enabled
        if self.use_momentum:
            if self._momentum_buffer is None:
                self._momentum_buffer = np.zeros(n_params)
            self._momentum_buffer = (
                self.beta * self._momentum_buffer +
                (1 - self.beta) * raw_gradient
            )
            return self._momentum_buffer.copy()
        
        return raw_gradient
    
    def step(
        self,
        loss_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Perform one SPSA optimization step.
        
        Args:
            loss_fn: Loss function.
            params: Current parameters.
        
        Returns:
            Tuple of (updated params, gradient norm).
        """
        gradient = self.compute_gradient(loss_fn, params)
        grad_norm = np.linalg.norm(gradient)
        
        # Gradient clipping for NISQ noise robustness
        if grad_norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / grad_norm)
            grad_norm = self.max_grad_norm
        
        a_k = self._get_learning_rate(self.iteration)
        
        # Update parameters
        new_params = params - a_k * gradient
        
        self.iteration += 1
        
        return new_params, grad_norm
    
    def optimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run SPSA optimization loop.
        
        Args:
            loss_fn: Loss function to minimize.
            initial_params: Starting parameters.
            max_iter: Maximum iterations.
            tolerance: Stop if gradient norm < tolerance.
            callback: Optional callback(iter, params, loss).
        
        Returns:
            Tuple of (optimized params, final loss).
        """
        params = np.asarray(initial_params).copy()
        
        for i in range(max_iter):
            params, grad_norm = self.step(loss_fn, params)
            
            if callback is not None:
                loss = loss_fn(params)
                callback(i, params, loss)
            
            if grad_norm < tolerance:
                break
        
        return params, loss_fn(params)
    
    def reset(self) -> None:
        """Reset iteration counter and momentum buffer."""
        self.iteration = 0
        self._momentum_buffer = None


def spsa_step(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    iteration: int,
    A: int = 20,
    a: float = 0.12,
    c: float = 0.08,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Perform a single SPSA step.
    
    Convenience function for one-off updates.
    
    Args:
        loss_fn: Loss function.
        params: Current parameters.
        iteration: Current iteration number.
        A, a, c, alpha, gamma: SPSA hyperparameters.
        seed: Random seed.
    
    Returns:
        Updated parameters.
    """
    rng = np.random.default_rng(seed)
    n_params = len(params)
    
    # Compute schedules
    a_k = a / ((A + iteration + 1) ** alpha)
    c_k = c / ((iteration + 1) ** gamma)
    
    # Perturbation
    delta = rng.choice([-1, 1], size=n_params).astype(np.float64)
    
    # Two-sided difference
    loss_plus = loss_fn(params + c_k * delta)
    loss_minus = loss_fn(params - c_k * delta)
    
    # Gradient estimate and update
    gradient = (loss_plus - loss_minus) / (2 * c_k * delta)
    
    return params - a_k * gradient
