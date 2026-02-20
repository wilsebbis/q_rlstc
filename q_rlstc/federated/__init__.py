"""Federated Edge Learning for Q-RLSTC (Q-FL).

Leverages Q-RLSTC's microscopic parameter size (20-56 params = 80-224 bytes)
for privacy-preserving federated learning. Raw GPS data never leaves the
user's device — only the gradient update is transmitted.

Package:
    - serialization: 80-byte model encode/decode
    - edge_client: local inference + SPSA gradient computation
    - parameter_server: gradient aggregation (FedAvg) across clients
"""
