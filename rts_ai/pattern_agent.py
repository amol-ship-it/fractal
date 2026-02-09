"""
Pattern Agent - Training loop for the pattern-based AI.

Runs episodes of the RTS game, using the PatternPolicy to make decisions
each tick, then feeds win/loss outcomes back to refine patterns.

No neural network. No gradient descent. Learning happens through the
four pillars: feedback, approximability, composability, exploration.

Usage:
    python -m rts_ai.train_pattern --episodes 500
"""

import numpy as np
import json
import os
import time
from typing import Dict, Optional

from rts_ai.pattern_policy import PatternPolicy, Strategy
from rts_ai.encoder import GameStateEncoder
from game.engine import GameEngine
from game.game_state import GameState
from game.ai_opponents import RushAI, EconomyAI, DefensiveAI, RandomAI


AI_OPPONENTS = {
    'rush': RushAI,
    'economy': EconomyAI,
    'defensive': DefensiveAI,
    'random': RandomAI,
}


class PatternAgent:
    """
    Training coordinator for the pattern-based AI.

    Runs full game episodes, feeds outcomes back to the PatternPolicy,
    and tracks learning progress.
    """

    def __init__(self, map_height: int = 8, map_width: int = 8):
        self.map_height = map_height
        self.map_width = map_width
        self.policy = PatternPolicy(map_height, map_width)

        # Training stats
        self.episodes_completed = 0
        self.win_history = []
        self.episode_ticks = []

    def play_episode(self, opponent_name: str = 'rush',
                     max_ticks: int = 2000) -> Dict:
        """
        Play one full game episode using the pattern policy.

        Returns:
            Dict with game outcome info.
        """
        engine = GameEngine(map_size=self.map_height, max_ticks=max_ticks)
        state = engine.reset()
        opponent = AI_OPPONENTS[opponent_name]()

        self.policy.begin_episode()

        while not state.done:
            # Pattern AI decides as player 0
            p0_action = self.policy.get_action(state, player=0)
            # Opponent decides as player 1
            p1_action = opponent.get_action(state, player=1)
            # Step the game
            state, info = engine.step(p0_action, p1_action)

        # Determine outcome
        won = state.winner == 0
        self.episodes_completed += 1
        self.win_history.append(1.0 if won else 0.0)
        self.episode_ticks.append(state.tick)

        # Feed outcome back to policy (this is where learning happens)
        self.policy.record_outcome(won, info)

        return {
            'won': won,
            'winner': state.winner,
            'tick': state.tick,
            'p0_units': len(state.game_map.get_player_units(0)),
            'p1_units': len(state.game_map.get_player_units(1)),
            'p0_resources': state.player_resources[0],
            'p1_resources': state.player_resources[1],
        }

    def train(self, total_episodes: int = 500,
              opponent: str = 'rush',
              log_interval: int = 10,
              save_path: str = None,
              max_ticks: int = 2000) -> Dict:
        """
        Main training loop. Plays games and learns from outcomes.

        Args:
            total_episodes: Number of games to play
            opponent: Name of opponent AI
            log_interval: Print stats every N episodes
            save_path: Where to save checkpoints
            max_ticks: Max ticks per game
        """
        start_time = time.time()

        print(f"Starting pattern-based training: {total_episodes} episodes")
        print(f"  Opponent: {opponent}")
        print(f"  Map: {self.map_height}x{self.map_width}")
        print(f"  Initial exploration rate: "
              f"{self.policy.explorer.exploration_rate:.1%}")
        print()

        for ep in range(1, total_episodes + 1):
            result = self.play_episode(opponent, max_ticks)

            if ep % log_interval == 0:
                recent_wins = self.win_history[-100:]
                win_rate = np.mean(recent_wins) if recent_wins else 0
                recent_ticks = self.episode_ticks[-100:]
                avg_ticks = np.mean(recent_ticks) if recent_ticks else 0
                stats = self.policy.get_stats()

                print(f"Episode {ep}/{total_episodes} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"Avg Ticks: {avg_ticks:.0f} | "
                      f"Patterns: {stats['patterns_discovered']} | "
                      f"Bindings: {stats['strategy_bindings']} | "
                      f"Explore: {stats['exploration_rate']:.1%}")

            # Save periodically
            if save_path and ep % (log_interval * 10) == 0:
                self.save(save_path)

        elapsed = time.time() - start_time

        # Final save
        if save_path:
            self.save(save_path)

        final_stats = self.policy.get_stats()
        final_win_rate = (np.mean(self.win_history[-100:])
                          if self.win_history else 0)

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Episodes:         {self.episodes_completed}")
        print(f"  Final win rate:   {final_win_rate:.1%}")
        print(f"  Patterns learned: {final_stats['patterns_discovered']}")
        print(f"  Strategy bindings:{final_stats['strategy_bindings']}")
        print(f"  Exploration rate: {final_stats['exploration_rate']:.1%}")
        print(f"  Elapsed time:     {elapsed:.1f}s")
        print(f"\n  Strategy usage:")
        for name, count in final_stats['strategy_usage'].items():
            print(f"    {name}: {count}")

        if save_path:
            print(f"\n  Checkpoint saved to: {save_path}")

        return {
            'episodes': self.episodes_completed,
            'final_win_rate': final_win_rate,
            'patterns_learned': final_stats['patterns_discovered'],
            'elapsed_time': elapsed,
        }

    def train_parallel(self, total_episodes: int = 500,
                       opponent: str = 'rush',
                       workers: int = None,
                       batch_size: int = None,
                       log_interval: int = 10,
                       save_path: str = None,
                       max_ticks: int = 2000,
                       warmup: int = None) -> Dict:
        """
        Parallel training loop using a rolling window of concurrent games.

        Architecture:
          1. WARM-UP PHASE: Runs a small number of sequential episodes first
             to establish initial strategy preferences. Without this, parallel
             workers all start from uninformed priors (confidence=0.5 for all
             strategies) and make diverse random choices, diluting the learning
             signal. The warm-up builds enough asymmetry in the strategy bindings
             that parallel workers can exploit known-good strategies.

          2. PARALLEL PHASE: Keeps `batch_size` games running in parallel at
             all times. As each game finishes, immediately applies its feedback
             and launches a new game with the UPDATED policy snapshot.

        Args:
            total_episodes: Number of games to play
            opponent: Name of opponent AI
            workers: Number of worker processes (default: cpu_count)
            batch_size: Concurrent games in flight (default: workers)
            log_interval: Print stats every N episodes
            save_path: Where to save checkpoints
            max_ticks: Max ticks per game
            warmup: Sequential episodes before parallelizing (default: 0).
                    Can help with bootstrapping in some scenarios but
                    generally not needed.
        """
        import multiprocessing
        from rts_ai.parallel_worker import play_episode_worker

        if workers is None:
            workers = multiprocessing.cpu_count()
        if batch_size is None:
            batch_size = workers

        # Default: no warm-up (parallel converges well on its own)
        if warmup is None:
            warmup = 0
        warmup = min(warmup, total_episodes)

        start_time = time.time()

        print(f"Starting PARALLEL pattern-based training: "
              f"{total_episodes} episodes")
        print(f"  Opponent: {opponent}")
        print(f"  Map: {self.map_height}x{self.map_width}")
        print(f"  Workers: {workers} processes")
        print(f"  Concurrent games: {batch_size}")
        if warmup > 0:
            print(f"  Warm-up (sequential): {warmup} episodes")
        print(f"  Initial exploration rate: "
              f"{self.policy.explorer.exploration_rate:.1%}")
        print()

        # ── Phase 1: Sequential warm-up ──────────────────────────────
        # Build initial strategy preferences so parallel workers know
        # which strategies to exploit instead of choosing randomly.
        if warmup > 0:
            print(f"Phase 1: Sequential warm-up ({warmup} episodes)...")
            for ep in range(1, warmup + 1):
                result = self.play_episode(opponent, max_ticks)
                if ep % log_interval == 0:
                    recent_wins = self.win_history[-100:]
                    win_rate = np.mean(recent_wins) if recent_wins else 0
                    stats = self.policy.get_stats()
                    print(f"  Warmup {ep}/{warmup} | "
                          f"Win Rate: {win_rate:.1%} | "
                          f"Patterns: {stats['patterns_discovered']} | "
                          f"Explore: {stats['exploration_rate']:.1%}")

            print(f"  Warm-up complete. "
                  f"Exploration rate: "
                  f"{self.policy.explorer.exploration_rate:.1%}")
            print()

        remaining_episodes = total_episodes - warmup
        if remaining_episodes <= 0:
            # All episodes consumed by warm-up
            elapsed = time.time() - start_time
            if save_path:
                self.save(save_path)
            return self._parallel_summary(elapsed, save_path)

        print(f"Phase 2: Parallel training ({remaining_episodes} episodes "
              f"across {workers} workers)...")

        episodes_done = 0
        episodes_dispatched = 0
        checkpoint_interval = log_interval * 10

        with multiprocessing.Pool(processes=workers) as pool:
            # Dispatch initial batch of concurrent games
            snapshot = self.policy.snapshot_for_workers()
            pending = []
            initial_dispatch = min(batch_size, remaining_episodes)
            for _ in range(initial_dispatch):
                args = (snapshot, opponent, max_ticks, self.map_height)
                future = pool.apply_async(play_episode_worker, (args,))
                pending.append(future)
                episodes_dispatched += 1

            # Process results as they arrive, launching replacements
            while episodes_done < remaining_episodes:
                # Poll for completed results
                ready = []
                still_pending = []
                for future in pending:
                    if future.ready():
                        ready.append(future)
                    else:
                        still_pending.append(future)

                if not ready:
                    # Nothing ready yet — wait a bit for the next one
                    try:
                        pending[0].get(timeout=0.1)
                        ready.append(pending[0])
                        still_pending = pending[1:]
                    except multiprocessing.TimeoutError:
                        continue
                    except Exception:
                        still_pending = pending[1:]

                pending = still_pending

                # Process each completed result
                for future in ready:
                    try:
                        result = future.get()
                    except Exception:
                        continue

                    self.episodes_completed += 1
                    episodes_done += 1
                    self.win_history.append(1.0 if result['won'] else 0.0)
                    self.episode_ticks.append(result['tick'])

                    # Apply feedback immediately
                    self.policy.apply_episode_result(
                        result['decisions'],
                        result['won'],
                        result['game_info'],
                        result.get('discovered_patterns'),
                    )

                    # Refresh snapshot so the next launched game
                    # sees the latest policy state
                    snapshot = self.policy.snapshot_for_workers()

                    # Launch replacement game if we have more to do
                    if episodes_dispatched < remaining_episodes:
                        args = (snapshot, opponent, max_ticks,
                                self.map_height)
                        future = pool.apply_async(
                            play_episode_worker, (args,))
                        pending.append(future)
                        episodes_dispatched += 1

                    # Log progress
                    total_done = warmup + episodes_done
                    if (episodes_done % log_interval == 0
                            or episodes_done == remaining_episodes):
                        recent_wins = self.win_history[-100:]
                        win_rate = (np.mean(recent_wins)
                                    if recent_wins else 0)
                        recent_ticks = self.episode_ticks[-100:]
                        avg_ticks = (np.mean(recent_ticks)
                                     if recent_ticks else 0)
                        stats = self.policy.get_stats()
                        elapsed = time.time() - start_time
                        eps_per_sec = episodes_done / max(elapsed, 0.1)

                        print(
                            f"Episode {total_done}/{total_episodes} | "
                            f"Win Rate: {win_rate:.1%} | "
                            f"Avg Ticks: {avg_ticks:.0f} | "
                            f"Patterns: "
                            f"{stats['patterns_discovered']} | "
                            f"Bindings: "
                            f"{stats['strategy_bindings']} | "
                            f"Explore: "
                            f"{stats['exploration_rate']:.1%} | "
                            f"{eps_per_sec:.1f} ep/s"
                        )

                    # Checkpoint periodically
                    if (save_path
                            and episodes_done % checkpoint_interval == 0
                            and episodes_done < remaining_episodes):
                        self.save(save_path)

        elapsed = time.time() - start_time

        # Final save
        if save_path:
            self.save(save_path)

        return self._parallel_summary(elapsed, save_path, warmup)

    def _parallel_summary(self, elapsed: float, save_path: str = None,
                          warmup: int = 0) -> Dict:
        """Print training summary and return results dict."""
        final_stats = self.policy.get_stats()
        final_win_rate = (np.mean(self.win_history[-100:])
                          if self.win_history else 0)

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE (parallel)")
        print(f"{'=' * 60}")
        print(f"  Episodes:         {self.episodes_completed}"
              + (f" ({warmup} warmup + "
                 f"{self.episodes_completed - warmup} parallel)"
                 if warmup > 0 else ""))
        print(f"  Final win rate:   {final_win_rate:.1%}")
        print(f"  Patterns learned: {final_stats['patterns_discovered']}")
        print(f"  Strategy bindings:{final_stats['strategy_bindings']}")
        print(f"  Exploration rate: {final_stats['exploration_rate']:.1%}")
        print(f"  Elapsed time:     {elapsed:.1f}s")
        print(f"  Throughput:       "
              f"{self.episodes_completed / max(elapsed, 0.1):.1f} episodes/sec")
        print(f"\n  Strategy usage:")
        for name, count in final_stats['strategy_usage'].items():
            print(f"    {name}: {count}")

        if save_path:
            print(f"\n  Checkpoint saved to: {save_path}")

        return {
            'episodes': self.episodes_completed,
            'final_win_rate': final_win_rate,
            'patterns_learned': final_stats['patterns_discovered'],
            'elapsed_time': elapsed,
        }

    def save(self, path: str):
        """Save agent state."""
        os.makedirs(path, exist_ok=True)

        # Save policy (patterns + bindings)
        self.policy.save(path)

        # Save training stats
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
            'episodes_completed': self.episodes_completed,
            'win_history': self.win_history[-1000:],
            'episode_ticks': self.episode_ticks[-1000:],
        })
        with open(os.path.join(path, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load agent state."""
        self.policy.load(path)

        stats_path = os.path.join(path, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.episodes_completed = stats.get('episodes_completed', 0)
            self.win_history = stats.get('win_history', [])
            self.episode_ticks = stats.get('episode_ticks', [])
