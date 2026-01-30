"""SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Gradient-free optimization suitable for variational quantum circuits
where gradient evaluation is expensive.

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
    """
    A: int = 20
    a: float = 0.12
    c: float = 0.08
    alpha: float = 0.602
    gamma: float = 0.101
    max_iter: int = 100
    seed: int = 42


class SPSAOptimizer:
    """SPSA optimizer for variational parameters.
    
    Uses simultaneous perturbation to estimate gradients with
    only 2 function evaluations per iteration.
    """
    
    def __init__(
        self,
        A: int = 20,
        a: float = 0.12,
        c: float = 0.08,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: int = 42,
    ):
        """Initialize SPSA optimizer.
        
        Args:
            A: Stability constant (typically 10-20% of max iterations).
            a: Initial learning rate scale.
            c: Initial perturbation magnitude.
            alpha: Learning rate decay exponent (theory: 1.0, practice: ~0.6).
            gamma: Perturbation decay exponent (theory: 1/6, practice: ~0.1).
            seed: Random seed for perturbations.
        """
        self.A = A
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
    
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
        """Estimate gradient using SPSA.
        
        Uses two-sided finite difference with simultaneous perturbation.
        
        Args:
            loss_fn: Function that takes params and returns loss.
            params: Current parameter vector.
        
        Returns:
            Estimated gradient vector.
        """
        n_params = len(params)
        c_k = self._get_perturbation_magnitude(self.iteration)
        
        # Sample perturbation
        delta = self._sample_perturbation(n_params)
        
        # Evaluate function at perturbed points
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta
        
        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)
        
        # Estimate gradient
        gradient = (loss_plus - loss_minus) / (2 * c_k * delta)
        
        return gradient
    
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
        a_k = self._get_learning_rate(self.iteration)
        
        # Update parameters
        new_params = params - a_k * gradient
        
        self.iteration += 1
        
        return new_params, np.linalg.norm(gradient)
    
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
        """Reset iteration counter."""
        self.iteration = 0


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
