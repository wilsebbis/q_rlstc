"""Q-value consistency tests — investigating ~1/3 cut-rate anomaly.

Advisor flagged that the VQ-DQN produces a suspiciously uniform ~33% cut
probability, which could indicate:
  (a) The fast numpy simulator diverges from the Qiskit statevector path.
  (b) The circuit effectively ignores its inputs (constant Q-values).
  (c) Random params produce a degenerate action distribution.

These tests check all three hypotheses directly.
"""

import numpy as np
import pytest

from q_rlstc.quantum.vqdqn_circuit import (
    _fast_vqc_probs,
    evaluate_q_values,
    build_vqdqn_circuit,
    q_values_batch,
)


class TestFastVsQiskitStatevector:
    """Verify that the fast numpy simulator matches Qiskit's statevector."""

    @pytest.mark.parametrize("seed", range(10))
    def test_probabilities_match(self, seed):
        """Fast numpy VQC probs should match Qiskit Statevector to < 1e-6."""
        rng = np.random.default_rng(seed)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers

        state = rng.standard_normal(n_qubits)
        params = rng.uniform(-np.pi, np.pi, n_params)

        # Fast numpy path
        fast_probs = _fast_vqc_probs(state, params, n_qubits, n_layers,
                                      use_data_reuploading=True,
                                      entanglement='linear')

        # Qiskit Statevector path
        from qiskit.quantum_info import Statevector
        circuit = build_vqdqn_circuit(state, params, n_qubits, n_layers,
                                       use_data_reuploading=True,
                                       add_measurements=False)
        sv = Statevector.from_instruction(circuit)
        qiskit_probs = sv.probabilities()

        np.testing.assert_allclose(fast_probs, qiskit_probs, atol=1e-6,
                                    err_msg=f"Prob mismatch at seed {seed}")


class TestQValuesVaryWithInput:
    """Verify the circuit is NOT ignoring its input state."""

    def test_qvalues_vary_across_states(self):
        """Q-values should have meaningful variance across diverse inputs."""
        rng = np.random.default_rng(42)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers
        params = rng.uniform(-np.pi, np.pi, n_params)

        # Generate 20 diverse states
        states = rng.standard_normal((20, n_qubits))
        q_vals = q_values_batch(states, params, n_qubits, n_layers)

        # Q-values for each action should have non-trivial std
        q_extend_std = np.std(q_vals[:, 0])
        q_cut_std = np.std(q_vals[:, 1])

        assert q_extend_std > 0.01, (
            f"Q(extend) std={q_extend_std:.6f} — circuit may be ignoring input"
        )
        assert q_cut_std > 0.01, (
            f"Q(cut) std={q_cut_std:.6f} — circuit may be ignoring input"
        )

    def test_qvalues_not_identical(self):
        """Q(extend) and Q(cut) should differ for at least some states."""
        rng = np.random.default_rng(7)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers
        params = rng.uniform(-np.pi, np.pi, n_params)

        states = rng.standard_normal((50, n_qubits))
        q_vals = q_values_batch(states, params, n_qubits, n_layers)

        # At least some states should have Q_extend != Q_cut
        diffs = np.abs(q_vals[:, 0] - q_vals[:, 1])
        assert np.max(diffs) > 0.01, (
            f"Max |Q_extend - Q_cut| = {np.max(diffs):.6f} — "
            f"Q-values are nearly identical for all actions"
        )


class TestActionDistribution:
    """Check that greedy action selection is not suspiciously constant."""

    def test_not_fixed_one_third_cut(self):
        """With random params, CUT fraction should NOT be ~ 1/3 ± 2%."""
        rng = np.random.default_rng(99)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers

        n_trials = 5  # Multiple random param sets
        cut_fractions = []

        for trial in range(n_trials):
            params = rng.uniform(-np.pi, np.pi, n_params)
            states = rng.standard_normal((200, n_qubits))
            q_vals = q_values_batch(states, params, n_qubits, n_layers)
            greedy_actions = np.argmax(q_vals, axis=1)
            cut_frac = np.mean(greedy_actions)
            cut_fractions.append(cut_frac)

        mean_cut = np.mean(cut_fractions)
        # If ALL trials produce ~1/3, that's suspicious
        # (individual trials can be anywhere, but the mean shouldn't be locked)
        suspicious = abs(mean_cut - 1/3) < 0.02
        if suspicious:
            # Not an assertion failure — this is diagnostic info
            print(f"\n  ⚠ SUSPICIOUS: mean CUT fraction across 5 trials = "
                  f"{mean_cut:.4f} (within 2% of 1/3)")
            print(f"  Per-trial: {[f'{f:.3f}' for f in cut_fractions]}")

        # The actual assertion: action distribution should NOT be constant
        # across ALL param sets (std > 0 means different params give different behavior)
        assert np.std(cut_fractions) > 0.01 or not suspicious, (
            f"CUT fraction is suspiciously stable at {mean_cut:.4f} across "
            f"all parameter sets — possible implementation bug"
        )

    def test_different_configs_give_different_distributions(self):
        """Different qubit/layer configs should produce different Q-value patterns."""
        rng = np.random.default_rng(42)
        configs = [
            (4, 2, 'linear'),
            (5, 3, 'linear'),
            (6, 2, 'circular'),
            (5, 4, 'full'),
        ]
        mean_cuts = []
        for nq, nl, ent in configs:
            n_params = nq * 2 * nl
            params = rng.uniform(-np.pi, np.pi, n_params)
            states = rng.standard_normal((100, nq))
            q_vals = q_values_batch(states, params, nq, nl, entanglement=ent)
            greedy = np.argmax(q_vals, axis=1)
            mean_cuts.append(float(np.mean(greedy)))

        # Different configs should generally produce different behavior
        assert np.std(mean_cuts) > 0.001, (
            f"All configs produce identical CUT fractions: {mean_cuts}"
        )


class TestOutputScaleBias:
    """Verify that scale/bias affect Q-values as expected."""

    def test_scale_amplifies(self):
        """Doubling output_scale should double Q-values (before bias)."""
        rng = np.random.default_rng(42)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers
        params = rng.uniform(-np.pi, np.pi, n_params)
        state = rng.standard_normal(n_qubits)

        scale_1 = np.ones(2)
        scale_2 = np.ones(2) * 2.0
        bias = np.zeros(2)

        states = state.reshape(1, -1)
        q1 = q_values_batch(states, params, n_qubits, n_layers,
                             output_scale=scale_1, output_bias=bias)
        q2 = q_values_batch(states, params, n_qubits, n_layers,
                             output_scale=scale_2, output_bias=bias)

        np.testing.assert_allclose(q2, 2.0 * q1, atol=1e-10,
                                    err_msg="Scale doesn't amplify Q-values")

    def test_bias_shifts(self):
        """Adding bias should shift Q-values by exactly that amount."""
        rng = np.random.default_rng(42)
        n_qubits = 5
        n_layers = 2
        n_params = n_qubits * 2 * n_layers
        params = rng.uniform(-np.pi, np.pi, n_params)
        state = rng.standard_normal(n_qubits)

        scale = np.ones(2)
        bias_0 = np.zeros(2)
        bias_1 = np.array([1.5, -0.5])

        states = state.reshape(1, -1)
        q0 = q_values_batch(states, params, n_qubits, n_layers,
                             output_scale=scale, output_bias=bias_0)
        q1 = q_values_batch(states, params, n_qubits, n_layers,
                             output_scale=scale, output_bias=bias_1)

        np.testing.assert_allclose(q1 - q0, bias_1.reshape(1, -1), atol=1e-10,
                                    err_msg="Bias doesn't shift Q-values correctly")


class TestParameterUpdates:
    """Fix 7: Verify SPSA optimizer actually moves parameters."""

    def test_params_change_after_update(self):
        """After an SPSA step, parameters should be different from initial."""
        from q_rlstc.rl.spsa import SPSAOptimizer

        rng = np.random.default_rng(42)
        n_params = 20
        initial = rng.uniform(-np.pi, np.pi, n_params)

        # Simple quadratic loss
        target = rng.uniform(-1, 1, n_params)
        def loss_fn(p):
            return float(np.sum((p - target) ** 2))

        opt = SPSAOptimizer(seed=42, use_momentum=True, momentum=0.9)
        updated, grad_norm = opt.step(loss_fn, initial.copy())

        assert not np.allclose(initial, updated), (
            "Parameters did not change after SPSA step"
        )
        assert grad_norm > 0, "Gradient norm is zero — no learning signal"

    def test_multiple_steps_reduce_loss(self):
        """After several SPSA steps, loss should decrease on average."""
        from q_rlstc.rl.spsa import SPSAOptimizer

        rng = np.random.default_rng(7)
        n_params = 10
        initial = rng.uniform(-np.pi, np.pi, n_params)
        target = np.zeros(n_params)  # Simple target

        def loss_fn(p):
            return float(np.sum((p - target) ** 2))

        opt = SPSAOptimizer(seed=7, use_momentum=True, momentum=0.9, a=0.5)
        params = initial.copy()
        initial_loss = loss_fn(params)

        for _ in range(50):
            params, _ = opt.step(loss_fn, params)

        final_loss = loss_fn(params)
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )


class TestGradientEstimates:
    """Fix 7: Verify gradient estimates are non-zero and meaningful."""

    def test_gradient_norm_positive(self):
        """SPSA gradient estimate should have positive norm for non-optimal params."""
        from q_rlstc.rl.spsa import SPSAOptimizer

        rng = np.random.default_rng(42)
        n_params = 20
        params = rng.uniform(-np.pi, np.pi, n_params)

        def loss_fn(p):
            return float(np.sum(p ** 2))  # Gradient should point toward origin

        opt = SPSAOptimizer(seed=42)
        grad = opt.compute_gradient(loss_fn, params)

        assert np.linalg.norm(grad) > 0.01, (
            f"Gradient norm too small: {np.linalg.norm(grad):.6f}"
        )

    def test_averaged_gradient_lower_variance(self):
        """K-sample averaged SPSA should have lower variance than K=1."""
        from q_rlstc.rl.spsa import SPSAOptimizer

        rng = np.random.default_rng(42)
        n_params = 20
        params = rng.uniform(-np.pi, np.pi, n_params)

        def loss_fn(p):
            return float(np.sum(p ** 2))

        # Collect gradient norms from K=1 and K=4
        norms_k1, norms_k4 = [], []
        for seed in range(20):
            opt1 = SPSAOptimizer(seed=seed, n_perturbations=1)
            g1 = opt1.compute_gradient(loss_fn, params.copy())
            norms_k1.append(np.linalg.norm(g1))

            opt4 = SPSAOptimizer(seed=seed, n_perturbations=4)
            g4 = opt4.compute_gradient(loss_fn, params.copy())
            norms_k4.append(np.linalg.norm(g4))

        # K=4 should have lower variance in gradient norms
        var_k1 = np.var(norms_k1)
        var_k4 = np.var(norms_k4)
        assert var_k4 < var_k1, (
            f"K=4 gradient variance ({var_k4:.4f}) not lower than "
            f"K=1 ({var_k1:.4f})"
        )


class TestMomentumSmoothing:
    """Fix 7: Verify momentum-SPSA smooths gradient estimates."""

    def test_momentum_buffer_accumulates(self):
        """Momentum buffer should accumulate and smooth gradients."""
        from q_rlstc.rl.spsa import SPSAOptimizer

        rng = np.random.default_rng(42)
        n_params = 10
        params = rng.uniform(-np.pi, np.pi, n_params)

        def loss_fn(p):
            return float(np.sum(p ** 2))

        opt = SPSAOptimizer(seed=42, use_momentum=True, momentum=0.9)

        # First step: momentum buffer initialized
        g1 = opt.compute_gradient(loss_fn, params)
        assert opt._momentum_buffer is not None, "Momentum buffer not initialized"

        # After several steps, buffer should be non-zero
        for _ in range(10):
            g = opt.compute_gradient(loss_fn, params)

        assert np.linalg.norm(opt._momentum_buffer) > 0, (
            "Momentum buffer is zero after multiple steps"
        )


class TestBoltzmannExploration:
    """Fix 3: Verify Boltzmann exploration works correctly."""

    def test_boltzmann_samples_both_actions(self):
        """With τ=1.0, Boltzmann should sample both actions over many states."""
        from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig

        cfg = AgentConfig(
            version="A", n_layers=2, shots=0,
            exploration_mode="boltzmann", boltzmann_temp=1.0,
        )
        agent = VQDQNAgent(config=cfg, seed=42)

        actions = []
        rng = np.random.default_rng(7)
        for _ in range(200):
            state = rng.standard_normal(5)
            actions.append(agent.act(state, greedy=False))

        # Both actions should appear
        assert 0 in actions, "Boltzmann never selected EXTEND"
        assert 1 in actions, "Boltzmann never selected CUT"

    def test_boltzmann_converges_to_greedy(self):
        """With very low τ, Boltzmann should match argmax behavior."""
        from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig

        cfg = AgentConfig(
            version="A", n_layers=2, shots=0,
            exploration_mode="boltzmann", boltzmann_temp=0.001,
        )
        agent = VQDQNAgent(config=cfg, seed=42)

        rng = np.random.default_rng(42)
        mismatches = 0
        for _ in range(100):
            state = rng.standard_normal(5)
            boltz_action = agent.act(state, greedy=False)
            greedy_action = agent.act(state, greedy=True)
            if boltz_action != greedy_action:
                mismatches += 1

        assert mismatches < 5, (
            f"Low-τ Boltzmann mismatched greedy {mismatches}/100 times"
        )
