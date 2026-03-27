"""Constrained Markov Decision Process (CMDP) mechanisms.

Provides a Lagrangian multiplier-based constraint system to control
trajectory segmentation budgets dynamically, replacing brittle static penalties.
"""

from typing import Tuple

class CutBudgetConstraint:
    """Defines a maximum budget for CUT actions in a trajectory."""
    
    def __init__(self, beta: float):
        """
        Args:
            beta: Target or maximum fraction of points to cut (e.g., 0.15 for 15%).
        """
        self.beta = beta

class LagrangeMultiplier:
    """Manages a dual variable (lambda) for Lagrangian relaxation of a CMDP.
    
    The multiplier adjusts automatically via gradient ascent on the dual objective.
    When empirical cost > budget, lambda increases (harsher penalty).
    When empirical cost < budget, lambda decreases (lower penalty).
    """
    
    def __init__(
        self, 
        init_lambda: float = 0.0, 
        lr_lambda: float = 0.01, 
        clamp_min: float = 0.0, 
        clamp_max: float = 10.0
    ):
        """
        Args:
            init_lambda: Initial penalty scale.
            lr_lambda: Learning rate for slow-timescale dual updates.
            clamp_min: Minimum penalty (typically 0.0 to prevent rewarding cuts).
            clamp_max: Maximum penalty to prevent numerical explosion.
        """
        self.current_lambda = init_lambda
        self.lr_lambda = lr_lambda
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
    def update(self, empirical_cost: float, constraint: CutBudgetConstraint) -> float:
        """Update the multiplier based on a single episode or batch empirical cost.
        
        Args:
            empirical_cost: The realized cut fraction (J_c).
            constraint: The target budget (beta).
            
        Returns:
            The new lambda value.
        """
        # Gradient ascent step on dual: lambda = lambda + lr * (J_c - beta)
        violation = empirical_cost - constraint.beta
        self.current_lambda += self.lr_lambda * violation
        
        # Clamp bounds
        self.current_lambda = max(self.clamp_min, min(self.clamp_max, self.current_lambda))
        
        return self.current_lambda
        
    def value(self) -> float:
        """Get the current penalty scale."""
        return self.current_lambda


def constraint_cost(action: int) -> float:
    """Evaluate the empirical constraint cost of an action.
    
    In trajectory segmentation, action=1 is a CUT, which uses budget.
    action=0 is an EXTEND, which uses 0 budget.
    
    Args:
        action: Agent chosen action index.
        
    Returns:
        Cost of 1.0 for a cut, 0.0 otherwise.
    """
    return 1.0 if action == 1 else 0.0
