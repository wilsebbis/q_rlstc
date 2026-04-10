import os
import sys

# Ensure the submodules can run appropriately
sys.path.append(os.path.dirname(__file__))

from config_parser import parse_args, BenchConfig
from evaluator import run_evaluation

def main():
    args = parse_args()
    
    # Check for scaffolded parameter sweeps (Phase 3 Hook)
    if args.sweep_k:
        print(f"Phase 3 Placeholder: K-Sweep requested for values {args.sweep_k}")
        print("Scaffolding activated. Full parallel parameter sweeping is not yet implemented.")
    if args.sweep_qubits:
        print(f"Phase 3 Placeholder: Qubit-Sweep requested for values {args.sweep_qubits}")
        print("Scaffolding activated. Ansatz scaling is not yet implemented.")

    # Initialize configuration engine
    config = BenchConfig(mode=args.mode, backend=args.backend)
    
    # Diagnostic logging
    config.print_state()
    
    # Run evaluation
    run_evaluation(config)

if __name__ == "__main__":
    main()
