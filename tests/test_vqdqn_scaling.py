import numpy as np
from q_rlstc.quantum.vqdqn_circuit import _fast_vqc_probs

# 5 features
state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# 6 qubits, 2 layers, 12 params per layer = 24 total params
params = np.random.rand(24)

print("Running _fast_vqc_probs with 6 qubits and 5 features...")
# This should now succeed instead of throwing an IndexError
probs = _fast_vqc_probs(
    state=state, 
    params=params, 
    n_qubits=6, 
    n_layers=2, 
    use_data_reuploading=True, 
    entanglement='linear'
)
print("Success! Output probabilities shape:", probs.shape)
