"""Smoke test for training loop."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from q_rlstc.config import QRLSTCConfig
from q_rlstc.data.synthetic import generate_synthetic_trajectories
from q_rlstc.rl.train import Trainer, train_qrlstc
from q_rlstc.rl.vqdqn_agent import VQDQNAgent, AgentConfig


class TestTrainingSmokeTest:
    """Smoke tests to verify training doesn't crash."""
    
    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast testing."""
        config = QRLSTCConfig()
        config.training.n_epochs = 1
        config.training.n_trajectories = 2
        config.vqdqn.shots_train = 64
        config.vqdqn.n_layers = 1
        config.rl.batch_size = 4
        config.rl.memory_size = 100
        config.noise.use_noise = False
        return config
    
    @pytest.fixture
    def small_dataset(self):
        """Generate small test dataset."""
        return generate_synthetic_trajectories(
            n_trajectories=2,
            n_segments_range=(2, 2),
            seed=42,
        )
    
    def test_trainer_creation(self, small_dataset, minimal_config):
        """Trainer can be created without errors"""
        trainer = Trainer(small_dataset, minimal_config)
        
        assert trainer.agent is not None
        assert trainer.buffer is not None
    
    def test_agent_creation(self):
        """VQDQNAgent can be created"""
        config = AgentConfig(
            n_qubits=3,
            n_layers=1,
            shots=32,
        )
        agent = VQDQNAgent(config=config)
        
        assert agent.n_params == 3 * 2 * 1  # 6 params
        assert agent.epsilon == 1.0
    
    def test_agent_act(self):
        """Agent can select actions"""
        config = AgentConfig(n_qubits=3, n_layers=1, shots=32)
        agent = VQDQNAgent(config=config)
        
        state = np.array([0.1, 0.2, 0.3])
        action = agent.act(state)
        
        assert action in [0, 1]
    
    def test_training_1_episode(self, small_dataset, minimal_config):
        """Training loop runs for 1 epoch without crashing"""
        trainer = Trainer(small_dataset, minimal_config)
        result = trainer.train(n_epochs=1, verbose=False)
        
        assert result.n_epochs == 1
        assert result.runtime_seconds > 0
    
    def test_convenience_function(self, small_dataset, minimal_config):
        """train_qrlstc convenience function works"""
        result = train_qrlstc(
            small_dataset,
            n_epochs=1,
            config=minimal_config,
            verbose=False,
        )
        
        assert 'final_od' in dir(result)
        assert 'final_f1' in dir(result)


class TestAgentComponents:
    """Tests for individual agent components."""
    
    def test_epsilon_decay(self):
        """Epsilon decays correctly"""
        config = AgentConfig(
            n_qubits=3,
            n_layers=1,
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_min=0.1,
        )
        agent = VQDQNAgent(config=config)
        
        assert agent.epsilon == 1.0
        agent.decay_epsilon()
        assert agent.epsilon == 0.5
        agent.decay_epsilon()
        assert agent.epsilon == 0.25
    
    def test_target_network_update(self):
        """Target network updates correctly"""
        config = AgentConfig(n_qubits=3, n_layers=1)
        agent = VQDQNAgent(config=config)
        
        # Modify params
        agent.params = np.ones_like(agent.params)
        
        # Target should be different
        assert not np.allclose(agent.params, agent.target_params)
        
        # After update, should match
        agent.update_target_network()
        np.testing.assert_array_equal(agent.params, agent.target_params)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
