import warnings

def split_trajectories_on_time_gap(trajectories, time_gap_threshold_seconds):
    """
    Phase 4 Placeholder: Preprocessing diagnostic hook.
    
    Given a list of Traj objects, this function should scan the points within 
    each trajectory. If the time difference between consecutive points exceeds
    `time_gap_threshold_seconds`, the trajectory should be split into two 
    separate sub-trajectories to avoid artificial straight-line teleportation artifacts.
    
    Parameters:
      trajectories: list of original Traj objects
      time_gap_threshold_seconds: int boundary
      
    Returns:
      A new list of processed Traj objects.
    """
    warnings.warn(
        "Trajectory time-gap splitting is currently just a scaffold. "
        "Original trajectories are being passed through unmodified.",
        UserWarning
    )
    
    # TODO: Implement the detection loop over traj.points
    # For point in traj.points:
    #    if (current_time - previous_time) > time_gap_threshold_seconds:
    #         split trajectory and append independently
    
    return trajectories

def generate_preprocessing_diagnostic_plot(original_trajs, processed_trajs, output_path):
    """
    Optional hook to plot 'before' and 'after' maps highlighting the removed segments.
    """
    raise NotImplementedError("Diagnostic plotting not yet implemented. Requires matplotlib dependency handling.")
