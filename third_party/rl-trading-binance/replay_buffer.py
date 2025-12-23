# replay_buffer.py
import logging
import random
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_frames: int,
        epsilon: float,
        state_shape: Tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame_idx = 0

        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity <<= 1
        self.tree = np.zeros(2 * self.tree_capacity - 1, dtype=np.float64)

        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.idx = 0
        self.size = 0
        self.max_priority = 1.0

        logger.info(f"Initialized PER buffer with capacity={capacity}, alpha={alpha}")

    def _beta(self) -> float:
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        return beta

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        data_idx = self.idx
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = done

        tree_idx = np.array([data_idx + self.tree_capacity - 1])
        priority = np.array([self.max_priority**self.alpha])

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_vectorized(tree_idx, change)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame_idx += 1

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index in tree with cumulative priority s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        # If we are in a parent node, check if the cumulative priority s falls
        # within the range of the left child.
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            # Otherwise, it's in the right child's range.
            return self._retrieve(right, s - self.tree[left])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.size >= batch_size, "Not enough samples in buffer"

        total_p = self.tree[0]
        assert total_p > 0.0, "Total priority must be positive in PER buffer"

        segment = total_p / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        data_indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)

        beta = self._beta()
        min_prob = np.min(self.tree[self.tree_capacity - 1 : self.tree_capacity - 1 + self.size]) / total_p
        max_weight = (min_prob * self.size) ** (-beta)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            node_idx = self._retrieve(0, s)
            data_idx = node_idx - (self.tree_capacity - 1)
            if data_idx >= self.size:
                data_idx = self.size - 1
                node_idx = data_idx + (self.tree_capacity - 1)

            indices[i] = node_idx
            data_indices[i] = data_idx

        p_samples = self.tree[indices] / total_p
        weights = (p_samples * self.size) ** (-beta)
        weights /= max_weight

        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            indices,
            weights,
        )

    def _propagate_vectorized(self, indices: np.ndarray, changes: np.ndarray) -> None:
        """Propagates changes up the tree in a vectorized, iterative manner."""
        if len(indices) == 0:
            return

        current_indices = indices
        current_changes = changes

        while np.any(current_indices > 0):
            parents = (current_indices - 1) // 2

            unique_parents, inverse_indices = np.unique(parents, return_inverse=True)
            aggregated_changes = np.bincount(inverse_indices, weights=current_changes)

            np.add.at(self.tree, unique_parents, aggregated_changes)

            current_indices = unique_parents
            current_changes = aggregated_changes

            mask = current_indices > 0
            current_indices = current_indices[mask]
            current_changes = current_changes[mask]

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Vectorized update of priorities for selected tree nodes.
        Non-finite td_errors (NaN/inf) are skipped to prevent corrupting the tree.
        """
        indices = indices.astype(np.int64)

        finite_mask = np.isfinite(td_errors)
        if not np.all(finite_mask):
            if logger.isEnabledFor(logging.DEBUG):
                non_finite_count = np.sum(~finite_mask)
                logger.debug(f"Skipping {non_finite_count} non-finite td_error(s) in PER update.")
            indices = indices[finite_mask]
            td_errors = td_errors[finite_mask]

        if len(indices) == 0:
            return

        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        changes = new_priorities - self.tree[indices]
        self.tree[indices] = new_priorities

        self._propagate_vectorized(indices, changes)

        if new_priorities.size > 0:
            self.max_priority = max(self.max_priority, np.max(new_priorities))

    def __len__(self) -> int:
        return self.size