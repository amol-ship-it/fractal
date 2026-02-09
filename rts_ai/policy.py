"""
Neural Network Policy - GridNet architecture for spatial action selection.

Uses a CNN encoder to process the spatial observation, then a transposed-CNN
decoder to produce per-cell action logits. Matches the architecture used in
the MicroRTS PPO papers (GridNet).

Architecture:
  Encoder: Conv2d -> ReLU -> Conv2d -> ReLU (spatial features)
  Actor:   ConvTranspose2d -> ReLU -> ConvTranspose2d (per-cell action logits)
  Critic:  Flatten -> Linear -> ReLU -> Linear (state value)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from game.game_state import NUM_FEATURE_PLANES
from game.actions import get_action_space_dims, DIMS_PER_CELL


class GridNetPolicy:
    """
    GridNet policy network implemented in pure NumPy.

    This is a simplified but functional CNN-like policy that processes
    spatial observations and outputs per-cell action distributions.
    Uses NumPy for portability (no PyTorch/TF dependency required).

    For production training, this can be swapped with a PyTorch GridNet.
    """

    def __init__(self, map_height: int, map_width: int,
                 hidden_dim: int = 64, lr: float = 2.5e-4):
        self.map_height = map_height
        self.map_width = map_width
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_features = NUM_FEATURE_PLANES
        self.action_dims = get_action_space_dims()
        self.total_action_dim = sum(self.action_dims)

        # Initialize network weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        h, w = self.map_height, self.map_width
        nf = self.num_features
        hd = self.hidden_dim

        # Encoder: input (H, W, NF) -> hidden (H, W, HD)
        # Simplified as a spatial linear layer (1x1 conv equivalent)
        scale = np.sqrt(2.0 / nf)
        self.enc_w1 = np.random.randn(nf, hd).astype(np.float32) * scale
        self.enc_b1 = np.zeros(hd, dtype=np.float32)

        scale = np.sqrt(2.0 / hd)
        self.enc_w2 = np.random.randn(hd, hd).astype(np.float32) * scale
        self.enc_b2 = np.zeros(hd, dtype=np.float32)

        # Spatial mixing (3x3 convolution approximated as shift-and-add)
        self.spatial_w = np.random.randn(hd, hd).astype(np.float32) * scale * 0.1

        # Actor head: hidden -> action logits per cell
        scale = np.sqrt(2.0 / hd)
        self.actor_w = np.random.randn(hd, self.total_action_dim).astype(np.float32) * scale * 0.01
        self.actor_b = np.zeros(self.total_action_dim, dtype=np.float32)

        # Critic head: hidden -> value
        flat_dim = h * w * hd
        scale = np.sqrt(2.0 / flat_dim)
        # Reduce spatial dims first
        self.critic_reduce_w = np.random.randn(hd, 1).astype(np.float32) * np.sqrt(2.0 / hd)
        self.critic_w1 = np.random.randn(h * w, 128).astype(np.float32) * np.sqrt(2.0 / (h * w))
        self.critic_b1 = np.zeros(128, dtype=np.float32)
        self.critic_w2 = np.random.randn(128, 1).astype(np.float32) * np.sqrt(2.0 / 128) * 0.01
        self.critic_b2 = np.zeros(1, dtype=np.float32)

        # Collect all parameters for gradient updates
        self._params = [
            'enc_w1', 'enc_b1', 'enc_w2', 'enc_b2', 'spatial_w',
            'actor_w', 'actor_b',
            'critic_reduce_w', 'critic_w1', 'critic_b1', 'critic_w2', 'critic_b2',
        ]

    def forward(self, obs: np.ndarray, action_mask: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through the policy network.

        Args:
            obs: (batch, H, W, NF) observation tensor
            action_mask: (batch, H*W, total_action_dim) valid action mask

        Returns:
            action_logits: (batch, H*W, total_action_dim) masked logits
            values: (batch,) state values
            hidden: (batch, H, W, HD) hidden representations
        """
        batch = obs.shape[0]
        h, w = self.map_height, self.map_width

        # Encoder
        # Layer 1: spatial linear (1x1 conv)
        x = obs.reshape(batch * h * w, self.num_features)
        x = x @ self.enc_w1 + self.enc_b1
        x = np.maximum(x, 0)  # ReLU
        x = x.reshape(batch, h, w, self.hidden_dim)

        # Simple spatial mixing (shift features from neighbors)
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
        spatial = (padded[:, :-2, 1:-1, :] + padded[:, 2:, 1:-1, :] +
                   padded[:, 1:-1, :-2, :] + padded[:, 1:-1, 2:, :]) / 4.0
        spatial_mixed = spatial.reshape(batch * h * w, self.hidden_dim) @ self.spatial_w
        x = x + spatial_mixed.reshape(batch, h, w, self.hidden_dim)

        # Layer 2
        hidden = x.reshape(batch * h * w, self.hidden_dim)
        hidden = hidden @ self.enc_w2 + self.enc_b2
        hidden = np.maximum(hidden, 0)
        hidden = hidden.reshape(batch, h, w, self.hidden_dim)

        # Actor: per-cell action logits
        actor_in = hidden.reshape(batch * h * w, self.hidden_dim)
        logits = actor_in @ self.actor_w + self.actor_b
        logits = logits.reshape(batch, h * w, self.total_action_dim)

        # Apply action mask (set invalid actions to -1e8)
        logits = np.where(action_mask > 0.5, logits, -1e8)

        # Critic: state value
        critic_spatial = hidden.reshape(batch, h * w, self.hidden_dim)
        critic_reduced = (critic_spatial @ self.critic_reduce_w).squeeze(-1)  # (batch, H*W)
        critic_hidden = critic_reduced @ self.critic_w1 + self.critic_b1
        critic_hidden = np.maximum(critic_hidden, 0)
        values = (critic_hidden @ self.critic_w2 + self.critic_b2).squeeze(-1)

        return logits, values, hidden

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray,
                   deterministic: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample actions from the policy.

        Returns:
            actions: (batch, H*W, DIMS_PER_CELL) sampled actions
            log_probs: (batch,) log probabilities
            values: (batch,) state values
        """
        logits, values, _ = self.forward(obs, action_mask)
        batch = obs.shape[0]
        hw = self.map_height * self.map_width

        actions = np.zeros((batch, hw, DIMS_PER_CELL), dtype=np.int32)
        total_log_prob = np.zeros(batch, dtype=np.float32)

        offset = 0
        for dim_idx, dim_size in enumerate(self.action_dims):
            dim_logits = logits[:, :, offset:offset + dim_size]
            dim_mask = action_mask[:, :, offset:offset + dim_size]

            for b in range(batch):
                for cell in range(hw):
                    cell_logits = dim_logits[b, cell]
                    cell_mask = dim_mask[b, cell]

                    # Softmax over valid actions
                    valid_logits = np.where(cell_mask > 0.5, cell_logits, -1e8)
                    max_logit = np.max(valid_logits)

                    if max_logit < -1e7:
                        # No valid actions for this dimension
                        actions[b, cell, dim_idx] = 0
                        continue

                    exp_logits = np.exp(valid_logits - max_logit)
                    probs = exp_logits / (np.sum(exp_logits) + 1e-8)

                    if deterministic:
                        action = np.argmax(probs)
                    else:
                        action = np.random.choice(dim_size, p=probs)

                    actions[b, cell, dim_idx] = action
                    total_log_prob[b] += np.log(probs[action] + 1e-8)

            offset += dim_size

        return actions, total_log_prob, values

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters as a dict."""
        return {name: getattr(self, name).copy() for name in self._params}

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set parameters from a dict."""
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value.copy())

    def save(self, filepath: str):
        """Save model parameters to file."""
        params = self.get_params()
        np.savez(filepath, **params)

    def load(self, filepath: str):
        """Load model parameters from file."""
        data = np.load(filepath)
        params = {key: data[key] for key in data.files}
        self.set_params(params)
