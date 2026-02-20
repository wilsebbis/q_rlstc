"""Tests for the replay buffer.

Verifies circular capacity, sampling, is_ready threshold, and
experience tuple integrity.
"""

import numpy as np
import pytest

from q_rlstc.rl.replay_buffer import ReplayBuffer, Experience


class TestReplayBufferBasics:
    """Core push/sample behaviour."""

    def setup_method(self):
        self.buf = ReplayBuffer(max_size=10)

    def test_push_and_len(self):
        """Buffer length should track pushes."""
        for i in range(5):
            self.buf.add(
                state=np.array([float(i)]),
                action=0,
                reward=1.0,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
        assert len(self.buf) == 5

    def test_circular_overflow(self):
        """Buffer should not exceed max_size."""
        for i in range(25):
            self.buf.add(
                state=np.array([float(i)]),
                action=i % 2,
                reward=float(i),
                next_state=np.array([float(i + 1)]),
                done=(i == 24),
            )
        assert len(self.buf) == 10

    def test_sample_batch_size(self):
        """Sampled batch should have requested size."""
        for i in range(20):
            self.buf.add(
                state=np.array([float(i)]),
                action=0,
                reward=1.0,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
        batch = self.buf.sample(4)
        assert len(batch) == 4

    def test_sample_returns_experience_tuples(self):
        """Each sampled item should be an Experience."""
        for i in range(10):
            self.buf.add(
                state=np.array([1.0, 2.0]),
                action=1,
                reward=0.5,
                next_state=np.array([3.0, 4.0]),
                done=False,
            )
        batch = self.buf.sample(3)
        for item in batch:
            assert isinstance(item, Experience)
            assert item.state is not None
            assert item.action in (0, 1)


class TestReplayBufferReady:
    """is_ready threshold."""

    def test_not_ready_when_empty(self):
        buf = ReplayBuffer(max_size=100)
        assert not buf.is_ready(min_size=32)

    def test_not_ready_when_too_few(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(10):
            buf.add(
                state=np.array([0.0]),
                action=0,
                reward=0.0,
                next_state=np.array([0.0]),
                done=False,
            )
        assert not buf.is_ready(min_size=32)

    def test_ready_when_enough(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(32):
            buf.add(
                state=np.array([0.0]),
                action=0,
                reward=0.0,
                next_state=np.array([0.0]),
                done=False,
            )
        assert buf.is_ready(min_size=32)
