"""Reward shaping functions for CMDP training.

Separates raw geometric alignment rewards from behavioral constraint
penalties to allow dynamic Lagrange multiplier tuning.
"""

def base_geometric_reward(
    distance_before: float, 
    distance_after: float, 
    scale_factor: float = 1.0
) -> float:
    """Calculate the raw, unpenalized reward from geometric improvement.
    
    Args:
        distance_before: Global distance metric before the action.
        distance_after: Global distance metric after the action.
        scale_factor: Normalization factor so rewards aren't tiny floats.
        
    Returns:
        The geometric reward signal (positive implies improvement).
    """
    improvement = distance_before - distance_after
    return improvement * scale_factor

def penalized_reward(
    base_reward: float, 
    action_cost: float, 
    current_lambda: float
) -> float:
    """Apply the dynamic Lagrangian penalty to the geometric reward.
    
    Args:
        base_reward: The geometric reward (from base_geometric_reward).
        action_cost: The constraint cost of the chosen action.
        current_lambda: Current multiplier scale from a LagrangeMultiplier.
        
    Returns:
        The final constrained reward to be fed into the Q-learning update.
    """
    return base_reward - (current_lambda * action_cost)
