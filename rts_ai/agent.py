"""
PPO Agent - Proximal Policy Optimization with pattern memory integration.

Implements PPO-Clip with:
- Invalid action masking (critical for RTS games)
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
- Integration with Recursive Learning AI pattern engine

The agent learns spatial RTS strategies through RL, while simultaneously
feeding strategic patterns into the core AI system for cross-domain transfer.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os

from rts_ai.policy import GridNetPolicy
from rts_ai.encoder import GameStateEncoder
from game.engine import VecGameEnv
from game.game_state import GameState
from game.actions import DIMS_PER_CELL, get_action_space_dims

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory
from core.feedback import FeedbackLoop, FeedbackType, FeedbackSignal


@dataclass
class RolloutBuffer:
    """Stores trajectory data for PPO updates."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    @property
    def size(self) -> int:
        return len(self.observations)


class PPOAgent:
    """
    PPO agent for the MicroRTS-style game with Recursive Learning AI integration.

    Training loop:
    1. Collect rollouts using current policy
    2. Compute advantages with GAE
    3. Update policy with PPO-Clip
    4. Extract strategic patterns and feed to pattern engine
    5. Store successful strategies in knowledge base
    """

    def __init__(self, map_height: int = 8, map_width: int = 8,
                 config: Dict = None):
        config = config or {}

        self.map_height = map_height
        self.map_width = map_width

        # PPO hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.1)
        self.value_clip_range = config.get('value_clip_range', 0.1)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.lr = config.get('lr', 2.5e-4)
        self.lr_annealing = config.get('lr_annealing', True)
        self.n_steps = config.get('n_steps', 128)
        self.n_minibatches = config.get('n_minibatches', 4)
        self.n_epochs = config.get('n_epochs', 4)
        self.total_timesteps = config.get('total_timesteps', 1000000)

        # Policy network
        self.policy = GridNetPolicy(
            map_height=map_height,
            map_width=map_width,
            hidden_dim=config.get('hidden_dim', 64),
            lr=self.lr,
        )

        # State encoder
        self.encoder = GameStateEncoder(map_height, map_width)

        # Recursive Learning AI integration
        self.pattern_memory = DualMemory(max_patterns=5000, max_state=500)
        self.pattern_engine = PatternEngine(self.pattern_memory)
        self.feedback_loop = FeedbackLoop(
            learning_rate=0.1, discount_factor=self.gamma
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.total_steps = 0
        self.episodes_completed = 0
        self.episode_rewards = []
        self.win_history = []  # Track wins/losses
        self.training_log = []

    def collect_rollouts(self, env: VecGameEnv) -> Dict:
        """Collect n_steps of experience from the environment."""
        self.buffer.clear()

        obs = env.reset()
        episode_reward = np.zeros(env.num_envs, dtype=np.float32)

        for step in range(self.n_steps):
            # Get action mask
            action_mask = env.get_action_mask()  # (num_envs, H*W, sum(dims))

            # Add batch dimension to obs
            obs_batch = obs  # Already (num_envs, H, W, NF)

            # Get actions from policy
            actions, log_probs, values = self.policy.get_action(
                obs_batch, action_mask
            )

            # Flatten actions for env step
            flat_actions = actions.reshape(env.num_envs, -1)

            # Step environment
            next_obs, rewards, dones, infos = env.step(flat_actions)

            # Store in buffer
            self.buffer.observations.append(obs_batch.copy())
            self.buffer.actions.append(actions.copy())
            self.buffer.action_masks.append(action_mask.copy())
            self.buffer.log_probs.append(log_probs.copy())
            self.buffer.rewards.append(rewards.copy())
            self.buffer.values.append(values.copy())
            self.buffer.dones.append(dones.copy())

            episode_reward += rewards
            self.total_steps += env.num_envs

            # Track completed episodes
            for i, done in enumerate(dones):
                if done:
                    self.episodes_completed += 1
                    self.episode_rewards.append(episode_reward[i])
                    win = infos[i].get('winner', -1) == 0
                    self.win_history.append(1.0 if win else 0.0)
                    episode_reward[i] = 0.0

                    # Feed episode result to pattern engine
                    self._process_episode_patterns(infos[i], win)

            obs = next_obs

        # Compute final values for GAE
        action_mask = env.get_action_mask()
        _, final_values, _ = self.policy.forward(obs, action_mask)

        return {
            'final_values': final_values,
            'steps_collected': self.n_steps * env.num_envs,
        }

    def compute_advantages(self, final_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n_steps = self.buffer.size
        num_envs = self.buffer.rewards[0].shape[0]

        advantages = np.zeros((n_steps, num_envs), dtype=np.float32)
        returns = np.zeros((n_steps, num_envs), dtype=np.float32)

        last_gae = np.zeros(num_envs, dtype=np.float32)
        last_value = final_values

        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - self.buffer.dones[t].astype(np.float32)
            delta = (self.buffer.rewards[t]
                     + self.gamma * last_value * next_non_terminal
                     - self.buffer.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            last_value = self.buffer.values[t]

        returns = advantages + np.array(self.buffer.values)
        return advantages, returns

    def update_policy(self, advantages: np.ndarray,
                      returns: np.ndarray) -> Dict[str, float]:
        """
        Update policy using PPO-Clip objective.

        Uses numerical gradient estimation for the NumPy policy.
        For production, swap with PyTorch autograd.
        """
        n_steps = self.buffer.size
        num_envs = self.buffer.rewards[0].shape[0]
        batch_size = n_steps * num_envs

        # Normalize advantages
        adv_flat = advantages.reshape(-1)
        adv_mean = np.mean(adv_flat)
        adv_std = np.std(adv_flat) + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std

        # Anneal learning rate
        if self.lr_annealing:
            frac = 1.0 - self.total_steps / self.total_timesteps
            current_lr = self.lr * max(frac, 0.01)
        else:
            current_lr = self.lr

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for epoch in range(self.n_epochs):
            # Create random minibatch indices
            indices = np.random.permutation(n_steps)
            mb_size = max(n_steps // self.n_minibatches, 1)

            for mb_start in range(0, n_steps, mb_size):
                mb_end = min(mb_start + mb_size, n_steps)
                mb_indices = indices[mb_start:mb_end]

                # Get minibatch data
                mb_obs = np.concatenate([self.buffer.observations[i] for i in mb_indices])
                mb_actions = np.concatenate([self.buffer.actions[i] for i in mb_indices])
                mb_masks = np.concatenate([self.buffer.action_masks[i] for i in mb_indices])
                mb_old_logprobs = np.concatenate([self.buffer.log_probs[i] for i in mb_indices])
                mb_advantages = np.concatenate([advantages_norm[i] for i in mb_indices])
                mb_returns = np.concatenate([returns[i] for i in mb_indices])
                mb_old_values = np.concatenate([self.buffer.values[i] for i in mb_indices])

                # Forward pass
                logits, values, _ = self.policy.forward(mb_obs, mb_masks)

                # Compute new log probs
                new_log_probs = self._compute_log_probs(logits, mb_actions, mb_masks)

                # PPO ratio
                ratio = np.exp(new_log_probs - mb_old_logprobs)
                clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)

                # Policy loss
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * clipped_ratio
                policy_loss = np.mean(np.maximum(policy_loss1, policy_loss2))

                # Value loss
                value_pred = values
                value_clipped = mb_old_values + np.clip(
                    value_pred - mb_old_values,
                    -self.value_clip_range, self.value_clip_range
                )
                value_loss1 = (value_pred - mb_returns) ** 2
                value_loss2 = (value_clipped - mb_returns) ** 2
                value_loss = 0.5 * np.mean(np.maximum(value_loss1, value_loss2))

                # Entropy bonus
                entropy = self._compute_entropy(logits, mb_masks)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update weights using evolutionary strategy (simplified)
                self._es_update(mb_obs, mb_masks, mb_actions,
                                mb_advantages, mb_returns, mb_old_logprobs,
                                mb_old_values, current_lr)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                update_count += 1

        stats = {
            'policy_loss': total_policy_loss / max(update_count, 1),
            'value_loss': total_value_loss / max(update_count, 1),
            'entropy': total_entropy / max(update_count, 1),
            'learning_rate': current_lr,
        }

        self.training_log.append(stats)
        return stats

    def _compute_log_probs(self, logits: np.ndarray, actions: np.ndarray,
                           masks: np.ndarray) -> np.ndarray:
        """Compute log probabilities of taken actions."""
        batch = logits.shape[0]
        hw = logits.shape[1]
        dims = get_action_space_dims()
        total_log_prob = np.zeros(batch, dtype=np.float32)

        offset = 0
        for dim_idx, dim_size in enumerate(dims):
            dim_logits = logits[:, :, offset:offset + dim_size]
            dim_mask = masks[:, :, offset:offset + dim_size]

            for b in range(batch):
                for cell in range(hw):
                    valid_logits = np.where(dim_mask[b, cell] > 0.5,
                                            dim_logits[b, cell], -1e8)
                    max_l = np.max(valid_logits)
                    if max_l < -1e7:
                        continue
                    exp_l = np.exp(valid_logits - max_l)
                    probs = exp_l / (np.sum(exp_l) + 1e-8)
                    action_idx = actions[b, cell, dim_idx]
                    total_log_prob[b] += np.log(probs[action_idx] + 1e-8)

            offset += dim_size

        return total_log_prob

    def _compute_entropy(self, logits: np.ndarray, masks: np.ndarray) -> float:
        """Compute average policy entropy."""
        batch = logits.shape[0]
        hw = logits.shape[1]
        dims = get_action_space_dims()
        total_entropy = 0.0
        count = 0

        offset = 0
        for dim_idx, dim_size in enumerate(dims):
            dim_logits = logits[:, :, offset:offset + dim_size]
            dim_mask = masks[:, :, offset:offset + dim_size]

            for b in range(batch):
                for cell in range(hw):
                    valid_logits = np.where(dim_mask[b, cell] > 0.5,
                                            dim_logits[b, cell], -1e8)
                    max_l = np.max(valid_logits)
                    if max_l < -1e7:
                        continue
                    exp_l = np.exp(valid_logits - max_l)
                    probs = exp_l / (np.sum(exp_l) + 1e-8)
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    total_entropy += entropy
                    count += 1

            offset += dim_size

        return total_entropy / max(count, 1)

    def _es_update(self, obs, masks, actions, advantages, returns,
                   old_logprobs, old_values, lr):
        """
        Evolutionary strategy parameter update.

        Estimates gradients via finite differences and applies updates.
        Simple but effective for moderate-dimensional parameter spaces.
        """
        params = self.policy.get_params()
        best_params = params.copy()
        best_score = float('-inf')

        # Number of perturbation samples
        n_samples = 8
        noise_scale = 0.02

        for _ in range(n_samples):
            # Create perturbation
            perturbed = {}
            noise = {}
            for name, param in params.items():
                n = np.random.randn(*param.shape).astype(np.float32) * noise_scale
                noise[name] = n
                perturbed[name] = param + n

            # Evaluate perturbed parameters
            self.policy.set_params(perturbed)
            logits, values, _ = self.policy.forward(obs, masks)
            new_lp = self._compute_log_probs(logits, actions, masks)

            ratio = np.exp(new_lp - old_logprobs)
            clipped = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
            pg_score = np.mean(np.minimum(ratio * advantages, clipped * advantages))
            vf_score = -np.mean((values - returns) ** 2)
            score = pg_score + 0.5 * vf_score

            if score > best_score:
                best_score = score
                best_params = perturbed.copy()

        # Apply best perturbation with learning rate
        for name in params:
            diff = best_params[name] - params[name]
            params[name] = params[name] + lr * diff / noise_scale
        self.policy.set_params(params)

    def _process_episode_patterns(self, info: Dict, won: bool):
        """
        Extract strategic patterns from a completed episode and feed
        to the Recursive Learning AI pattern engine.
        """
        # Create a feature vector representing the episode outcome
        features = [
            float(won),
            info.get('p0_resources', 0) / 20.0,
            info.get('p1_resources', 0) / 20.0,
            info.get('p0_units', 0) / 10.0,
            info.get('p1_units', 0) / 10.0,
            info.get('tick', 0) / 2000.0,
        ]

        # Feed to pattern engine
        result = self.pattern_engine.process(features, domain="rts_strategy")

        # Apply feedback based on outcome
        feedback_value = 1.0 if won else -0.5
        for pattern in result['discovered_patterns']:
            self.feedback_loop.apply_feedback(
                FeedbackSignal(
                    signal_type=FeedbackType.EXTRINSIC,
                    value=feedback_value,
                    target_pattern_ids=[pattern.id],
                    context={'won': won, 'tick': info.get('tick', 0)}
                ),
                self.pattern_memory.patterns.patterns
            )

    def train(self, env: VecGameEnv, total_timesteps: int = None,
              log_interval: int = 10, save_path: str = None,
              callback=None) -> Dict:
        """
        Main training loop.

        Args:
            env: Vectorized game environment
            total_timesteps: Override total timesteps
            log_interval: Log stats every N updates
            save_path: Path to save checkpoints
            callback: Optional callback function(agent, update_num)
        """
        if total_timesteps:
            self.total_timesteps = total_timesteps

        num_updates = self.total_timesteps // (self.n_steps * env.num_envs)
        update_num = 0

        print(f"Starting PPO training: {self.total_timesteps} timesteps, "
              f"{num_updates} updates")
        print(f"  Envs: {env.num_envs}, Steps/update: {self.n_steps}, "
              f"Minibatches: {self.n_minibatches}, Epochs: {self.n_epochs}")

        for update in range(num_updates):
            update_num += 1

            # Collect rollouts
            rollout_info = self.collect_rollouts(env)

            # Compute advantages
            advantages, returns = self.compute_advantages(
                rollout_info['final_values']
            )

            # Update policy
            update_stats = self.update_policy(advantages, returns)

            # Logging
            if update_num % log_interval == 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_wins = self.win_history[-100:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                win_rate = np.mean(recent_wins) if recent_wins else 0

                print(f"Update {update_num}/{num_updates} | "
                      f"Steps: {self.total_steps} | "
                      f"Episodes: {self.episodes_completed} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f} | "
                      f"Patterns: {len(self.pattern_memory.patterns.patterns)}")

            # Save checkpoint
            if save_path and update_num % (log_interval * 10) == 0:
                self.save(save_path)

            if callback:
                callback(self, update_num)

        # Final save
        if save_path:
            self.save(save_path)

        return {
            'total_steps': self.total_steps,
            'episodes': self.episodes_completed,
            'final_win_rate': np.mean(self.win_history[-100:]) if self.win_history else 0,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'patterns_learned': len(self.pattern_memory.patterns.patterns),
        }

    def save(self, path: str):
        """Save agent state (policy + patterns + stats)."""
        os.makedirs(path, exist_ok=True)

        # Save policy
        self.policy.save(os.path.join(path, 'policy.npz'))

        # Save patterns
        patterns_data = {
            pid: p.to_dict()
            for pid, p in self.pattern_memory.patterns.patterns.items()
        }
        with open(os.path.join(path, 'patterns.json'), 'w') as f:
            json.dump(patterns_data, f, indent=2)

        # Save training stats (convert numpy types to native Python)
        def to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_native(v) for v in obj]
            return obj

        stats = to_native({
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'episode_rewards': self.episode_rewards[-1000:],
            'win_history': self.win_history[-1000:],
            'training_log': self.training_log[-100:],
        })
        with open(os.path.join(path, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load agent state."""
        policy_path = os.path.join(path, 'policy.npz')
        if os.path.exists(policy_path):
            self.policy.load(policy_path)

        patterns_path = os.path.join(path, 'patterns.json')
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                patterns_data = json.load(f)
            for pid, pdata in patterns_data.items():
                pattern = Pattern.from_dict(pdata)
                self.pattern_memory.patterns.store(pattern)

        stats_path = os.path.join(path, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.total_steps = stats.get('total_steps', 0)
            self.episodes_completed = stats.get('episodes_completed', 0)
            self.episode_rewards = stats.get('episode_rewards', [])
            self.win_history = stats.get('win_history', [])
