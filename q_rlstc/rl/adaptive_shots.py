import numpy as np

def hoeffding_bound(delta: float, n_shots: int, r_range: float = 2.0) -> float:
    """
    Computes the Hoeffding bound for a given confidence level and number of shots.
    
    Formula: epsilon = R * sqrt(ln(2/delta) / (2 * n_shots))
    where R is the range of the random variable. For Q-values estimated
    from probabilities scaled to [-1, 1], R is 2.0.
    
    Args:
        delta: Allowed failure probability (e.g., 0.05 for 95% confidence).
        n_shots: Number of measurement shots.
        r_range: Range of the observable.
        
    Returns:
        The margin of error epsilon.
    """
    return r_range * np.sqrt(np.log(2.0 / delta) / (2.0 * max(1, n_shots)))

def needs_more_shots(q_values: np.ndarray,
                     n_shots: int,
                     max_shots: int,
                     confidence_delta: float = 0.05,
                     r_range: float = 2.0) -> bool:
    """
    Determines if more shots are needed to confidently separate the top two Q-values.
    
    Args:
        q_values: The current estimated Q-values.
        n_shots: The number of shots used to estimate them.
        max_shots: The maximum allowed shots (hard ceiling).
        confidence_delta: The allowed failure probability for the gap.
        r_range: The range of the random variable.
        
    Returns:
        True if the gap between the top two Q-values is smaller than the 
        Hoeffding confidence bounds AND we haven't reached max_shots.
    """
    if n_shots >= max_shots:
        return False
        
    if len(q_values) < 2:
        return False
        
    # Sort descending to get top 2
    sorted_q = np.sort(q_values)[::-1]
    gap = sorted_q[0] - sorted_q[1]
    
    # We apply the bound to the difference of expectations. 
    # A conservative bound is 2 * epsilon.
    epsilon = hoeffding_bound(confidence_delta, n_shots, r_range)
    
    # If the empirical gap is smaller than or equal to 2 * epsilon, 
    # we aren't confident enough that the top action is strictly better.
    return bool(gap <= 2.0 * epsilon)
