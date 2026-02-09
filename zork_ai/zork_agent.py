"""
Zork Pattern Agent — Training loop for the pattern-based Zork AI.

Runs episodes of Zork I, using the ZorkPatternPolicy to make decisions
each turn, then feeds score outcomes back to refine patterns.

No neural network. No gradient descent. Learning happens through the
four pillars: feedback, approximability, composability, exploration.

Usage:
    python -m zork_ai.train_zork --game-file zork1.z5 --episodes 500
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Optional

from zork_ai.zork_policy import ZorkPatternPolicy, ZorkStrategy
from zork_ai.game_interface import FrotzInterface, ZorkState
from zork_ai.text_parser import (
    parse_exits,
    detect_death,
    is_error_response,
    is_blocked_movement,
)


class ZorkPatternAgent:
    """
    Training coordinator for the pattern-based Zork AI.

    Runs full Zork episodes, feeds outcomes back to the ZorkPatternPolicy,
    and tracks learning progress.
    """

    def __init__(self, game_file: str, frotz_path: str = 'dfrotz'):
        self.game_file = game_file
        self.frotz_path = frotz_path
        self.policy = ZorkPatternPolicy()

        # Training stats
        self.episodes_completed = 0
        self.score_history: List[int] = []
        self.best_score: int = 0
        self.rooms_visited_history: List[int] = []

    def play_episode(self, max_moves: int = 400,
                     verbose: bool = False) -> Dict:
        """
        Play one full Zork episode.

        Args:
            max_moves: Max commands before ending the episode
            verbose: Print each command/response (for debugging)

        Returns:
            Dict with episode outcome info.
        """
        fi = FrotzInterface(self.game_file, self.frotz_path)

        try:
            opening = fi.start()
            if verbose:
                print(f"[OPENING] {opening[:200]}...")

            self.policy.begin_episode()

            moves = 0
            prev_score = 0
            moves_since_score = 0
            max_stale_moves = 100  # End episode if stuck
            last_inv_check = 0     # Track when we last checked inventory
            INV_CHECK_INTERVAL = 10  # Check inventory every N moves

            while moves < max_moves:
                state = fi.state

                # Check termination conditions
                if state.is_dead or state.is_won:
                    break
                if not fi.is_running:
                    break

                # Periodically refresh inventory so strategies know what we have
                if moves - last_inv_check >= INV_CHECK_INTERVAL:
                    try:
                        fi.get_inventory()
                    except Exception:
                        pass
                    last_inv_check = moves

                # Parse available exits from current description
                available_exits = parse_exits(state.description)

                # Get command from policy
                command = self.policy.get_command(state, available_exits)

                if verbose:
                    print(f"[{moves}] {state.location} > {command}")

                # Send command
                response = fi.send_command(command)

                if verbose and len(response) < 300:
                    print(f"  {response.strip()[:200]}")

                # Track room visits
                if state.location:
                    self.policy.room_memory.visit_room(state.location)

                # After a successful "take" command, refresh inventory
                if (command.lower().startswith('take')
                        and 'taken' in response.lower()):
                    try:
                        fi.get_inventory()
                        last_inv_check = moves
                    except Exception:
                        pass

                # Track score deltas for credit assignment
                score_delta = fi.score_delta
                if score_delta > 0:
                    # Credit the strategy that was just used
                    if self.policy._episode_decisions:
                        last_strat = self.policy._episode_decisions[-1][1]
                        self.policy.record_score_delta(last_strat, score_delta)
                    moves_since_score = 0
                else:
                    moves_since_score += 1

                # Staleness check: end episode if no progress
                if moves_since_score >= max_stale_moves:
                    if verbose:
                        print(f"  [STALE] No score change in "
                              f"{max_stale_moves} moves, ending episode")
                    break

                moves += 1

                # Check for death after command
                if detect_death(response):
                    break

            # Get final state
            final_score = fi.state.score
            rooms_visited = len(fi.state.visited_rooms)

            # Update stats
            self.episodes_completed += 1
            self.score_history.append(final_score)
            self.rooms_visited_history.append(rooms_visited)
            if final_score > self.best_score:
                self.best_score = final_score

            # Game info for feedback
            game_info = {
                'rooms_visited': rooms_visited,
                'items_collected': len(fi.state.items_collected),
                'moves': moves,
                'is_dead': fi.state.is_dead,
                'is_won': fi.state.is_won,
            }

            # Feed outcome back to policy (this is where learning happens)
            self.policy.record_outcome(final_score, game_info=game_info)

            return {
                'score': final_score,
                'rooms_visited': rooms_visited,
                'items_collected': len(fi.state.items_collected),
                'moves': moves,
                'is_dead': fi.state.is_dead,
                'is_won': fi.state.is_won,
            }

        finally:
            fi.close()

    def play_visual(self, max_moves: int = 200) -> Dict:
        """
        Play one episode with richly formatted visual output.

        Shows the agent's decision-making in real-time:
        - Room name and description
        - Strategy chosen and why
        - Score changes highlighted
        - Inventory and room tracking

        Use with: python -m zork_ai.train_zork --game-file zork1.z5 --play
        """
        fi = FrotzInterface(self.game_file, self.frotz_path)
        BOLD = '\033[1m'
        DIM = '\033[2m'
        GREEN = '\033[32m'
        RED = '\033[31m'
        YELLOW = '\033[33m'
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        RESET = '\033[0m'
        BG_GREEN = '\033[42m'

        strat_colors = {
            ZorkStrategy.EXPLORE_NEW: CYAN,
            ZorkStrategy.EXPLORE_KNOWN: CYAN,
            ZorkStrategy.COLLECT_ITEMS: GREEN,
            ZorkStrategy.USE_ITEM: MAGENTA,
            ZorkStrategy.DEPOSIT_TROPHY: YELLOW,
            ZorkStrategy.FIGHT: RED,
            ZorkStrategy.MANAGE_LIGHT: YELLOW,
            ZorkStrategy.INTERACT: MAGENTA,
        }

        try:
            opening = fi.start()

            print()
            print(f"{BOLD}{'=' * 70}{RESET}")
            print(f"{BOLD}  ZORK I — AI Agent Playthrough{RESET}")
            print(f"{BOLD}{'=' * 70}{RESET}")
            print()

            # Show opening text (trimmed)
            lines = opening.split('\n')
            for line in lines:
                s = line.strip()
                if s and 'Score:' not in s and s != '>':
                    print(f"  {DIM}{s}{RESET}")
            print()

            self.policy.begin_episode()
            moves = 0
            moves_since_score = 0
            max_stale_moves = 100
            last_location = ''

            while moves < max_moves:
                state = fi.state
                if state.is_dead or state.is_won or not fi.is_running:
                    break

                available_exits = parse_exits(state.description)

                # Show location change
                if state.location != last_location:
                    print(f"{BOLD}{'-' * 50}{RESET}")
                    print(f"  {BOLD}{CYAN}{state.location}{RESET}"
                          f"  {DIM}(Score: {state.score}  "
                          f"Moves: {moves}){RESET}")

                    # Show description (first 2 meaningful lines)
                    desc_lines = [l.strip() for l in
                                  state.description.split('\n')
                                  if l.strip() and 'Score:' not in l
                                  and l.strip() != '>']
                    for dl in desc_lines[:3]:
                        print(f"  {DIM}{dl[:80]}{RESET}")

                    if state.is_dark:
                        print(f"  {RED}{BOLD}[DARK!]{RESET}")
                    if state.enemies:
                        print(f"  {RED}{BOLD}[ENEMIES: "
                              f"{', '.join(state.enemies)}]{RESET}")
                    if state.visible_items:
                        print(f"  {GREEN}[Items: "
                              f"{', '.join(state.visible_items[:3])}]{RESET}")
                    print()
                    last_location = state.location

                # Get strategy selection
                strategy, sit_key = self.policy.select_strategy(
                    state, available_exits)
                strat_name = ZorkStrategy(strategy).name
                color = strat_colors.get(strategy, RESET)

                # Execute strategy to get command
                from zork_ai.zork_policy import STRATEGY_EXECUTORS
                executor = STRATEGY_EXECUTORS.get(
                    strategy,
                    STRATEGY_EXECUTORS[ZorkStrategy.INTERACT])
                command = executor(
                    state, self.policy.room_memory,
                    available_exits or [])

                # Track direction
                cmd_lower = command.lower().strip()
                from zork_ai.text_parser import DIRECTIONS, DIR_ABBREVS
                if cmd_lower in DIRECTIONS or cmd_lower in DIR_ABBREVS:
                    d = DIR_ABBREVS.get(cmd_lower, cmd_lower)
                    if state.location:
                        self.policy.room_memory.try_exit(state.location, d)

                # Show the decision
                print(f"  {DIM}[{strat_name}]{RESET} "
                      f"{color}> {command}{RESET}")

                # Send command
                response = fi.send_command(command)

                # Show score change
                if fi.score_delta > 0:
                    print(f"  {BG_GREEN}{BOLD} +{fi.score_delta} POINTS! "
                          f"(Total: {fi.state.score}/350) {RESET}")
                    if self.policy._episode_decisions:
                        last_strat = self.policy._episode_decisions[-1][1]
                        self.policy.record_score_delta(
                            last_strat, fi.score_delta)
                    moves_since_score = 0
                elif fi.score_delta < 0:
                    print(f"  {RED}{fi.score_delta} points "
                          f"(Total: {fi.state.score}/350){RESET}")
                    moves_since_score = 0
                else:
                    moves_since_score += 1

                # Show interesting responses
                resp_lines = [l.strip() for l in response.split('\n')
                              if l.strip() and 'Score:' not in l
                              and l.strip() != '>' and l.strip() != command]
                for rl in resp_lines[:2]:
                    if rl and rl != state.location and len(rl) > 5:
                        # Skip room name repetitions
                        if rl.lower() == state.location.lower():
                            continue
                        print(f"  {DIM}  {rl[:70]}{RESET}")

                if state.is_dead:
                    print(f"\n  {RED}{BOLD}*** YOU HAVE DIED ***{RESET}")
                    break

                if moves_since_score >= max_stale_moves:
                    print(f"\n  {DIM}(No progress in {max_stale_moves} "
                          f"moves — ending episode){RESET}")
                    break

                moves += 1

                if detect_death(response):
                    print(f"\n  {RED}{BOLD}*** YOU HAVE DIED ***{RESET}")
                    break

            # Final summary
            final_score = fi.state.score
            rooms_visited = len(fi.state.visited_rooms)

            print()
            print(f"{BOLD}{'=' * 70}{RESET}")
            print(f"{BOLD}  EPISODE SUMMARY{RESET}")
            print(f"{BOLD}{'=' * 70}{RESET}")
            print(f"  Final Score:    {BOLD}{final_score}{RESET} / 350")
            print(f"  Rooms Visited:  {rooms_visited}")
            print(f"  Moves Used:     {moves} / {max_moves}")
            print(f"  Items Found:    {len(fi.state.items_collected)}")
            if fi.state.is_dead:
                print(f"  Outcome:        {RED}DIED{RESET}")
            elif fi.state.is_won:
                print(f"  Outcome:        {GREEN}{BOLD}WON!{RESET}")
            else:
                print(f"  Outcome:        Survived")

            if fi.state.inventory:
                print(f"  Inventory:      {', '.join(fi.state.inventory)}")

            # Strategy breakdown
            strat_counts = {}
            for _, s in self.policy._episode_decisions:
                name = ZorkStrategy(s).name
                strat_counts[name] = strat_counts.get(name, 0) + 1
            if strat_counts:
                print(f"\n  Strategy Usage:")
                for name, count in sorted(strat_counts.items(),
                                          key=lambda x: -x[1]):
                    bar = '#' * min(count, 30)
                    print(f"    {name:20s} {bar} ({count})")

            print(f"{'=' * 70}")
            print()

            # Record outcome
            self.episodes_completed += 1
            self.score_history.append(final_score)
            self.rooms_visited_history.append(rooms_visited)
            if final_score > self.best_score:
                self.best_score = final_score

            game_info = {
                'rooms_visited': rooms_visited,
                'items_collected': len(fi.state.items_collected),
                'moves': moves,
                'is_dead': fi.state.is_dead,
                'is_won': fi.state.is_won,
            }
            self.policy.record_outcome(final_score, game_info=game_info)

            return {
                'score': final_score,
                'rooms_visited': rooms_visited,
                'moves': moves,
            }

        finally:
            fi.close()

    def train(self, total_episodes: int = 500,
              log_interval: int = 10,
              save_path: str = None,
              max_moves: int = 400,
              verbose: bool = False) -> Dict:
        """
        Main training loop. Plays Zork episodes and learns from outcomes.
        """
        start_time = time.time()

        print(f"Starting Zork pattern-based training: "
              f"{total_episodes} episodes")
        print(f"  Game file: {self.game_file}")
        print(f"  Max moves per episode: {max_moves}")
        print(f"  Initial exploration rate: "
              f"{self.policy.explorer.exploration_rate:.1%}")
        print()

        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10  # Abort if 10 episodes fail in a row

        for ep in range(1, total_episodes + 1):
            try:
                result = self.play_episode(max_moves,
                                           verbose=(verbose and ep <= 3))
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                # Episode crashed (dfrotz hang, broken pipe, etc.)
                # Log it, record a zero-score episode, and continue
                consecutive_failures += 1
                self.episodes_completed += 1
                self.score_history.append(0)
                self.rooms_visited_history.append(0)
                self.policy.begin_episode()  # Reset policy state

                if verbose or consecutive_failures <= 3:
                    print(f"  [Episode {ep} failed: {type(e).__name__}: "
                          f"{str(e)[:80]}]")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  ERROR: {MAX_CONSECUTIVE_FAILURES} consecutive "
                          f"episode failures — aborting training.")
                    print(f"  Last error: {e}")
                    break

                continue

            if ep % log_interval == 0:
                recent_scores = self.score_history[-100:]
                avg_score = np.mean(recent_scores) if recent_scores else 0
                max_recent = max(recent_scores) if recent_scores else 0
                recent_rooms = self.rooms_visited_history[-100:]
                avg_rooms = np.mean(recent_rooms) if recent_rooms else 0
                stats = self.policy.get_stats()

                print(
                    f"Episode {ep}/{total_episodes} | "
                    f"Avg Score: {avg_score:.1f} | "
                    f"Best: {self.best_score} | "
                    f"Rooms: {avg_rooms:.1f} | "
                    f"Patterns: {stats['patterns_discovered']} | "
                    f"Bindings: {stats['strategy_bindings']} | "
                    f"Explore: {stats['exploration_rate']:.1%}"
                )

            # Save every 50 episodes (more frequent to prevent data loss)
            if save_path and ep % 50 == 0:
                self.save(save_path)

        elapsed = time.time() - start_time

        # Final save
        if save_path:
            self.save(save_path)

        final_stats = self.policy.get_stats()
        recent_scores = self.score_history[-100:]
        avg_score = np.mean(recent_scores) if recent_scores else 0

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Episodes:         {self.episodes_completed}")
        print(f"  Average score:    {avg_score:.1f} / 350")
        print(f"  Best score:       {self.best_score} / 350")
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
            'avg_score': float(avg_score),
            'best_score': self.best_score,
            'patterns_learned': final_stats['patterns_discovered'],
            'elapsed_time': elapsed,
        }

    def train_parallel(self, total_episodes: int = 500,
                       workers: int = None,
                       batch_size: int = None,
                       log_interval: int = 10,
                       save_path: str = None,
                       max_moves: int = 400,
                       warmup: int = None) -> Dict:
        """
        Parallel training loop using concurrent dfrotz processes.

        Each worker spawns its own dfrotz subprocess and runs an episode
        independently, then the main process merges the results.
        """
        import multiprocessing
        from zork_ai.parallel_worker import play_episode_worker

        if workers is None:
            workers = min(multiprocessing.cpu_count(), 8)  # cap at 8
        if batch_size is None:
            batch_size = workers

        # Default: no warm-up
        if warmup is None:
            warmup = 0
        warmup = min(warmup, total_episodes)

        start_time = time.time()

        print(f"Starting PARALLEL Zork training: "
              f"{total_episodes} episodes")
        print(f"  Game file: {self.game_file}")
        print(f"  Workers: {workers} processes")
        print(f"  Concurrent games: {batch_size}")
        if warmup > 0:
            print(f"  Warm-up (sequential): {warmup} episodes")
        print(f"  Initial exploration rate: "
              f"{self.policy.explorer.exploration_rate:.1%}")
        print()

        # ── Phase 1: Sequential warm-up ──
        if warmup > 0:
            print(f"Phase 1: Sequential warm-up ({warmup} episodes)...")
            for ep in range(1, warmup + 1):
                result = self.play_episode(max_moves)
                if ep % log_interval == 0:
                    recent_scores = self.score_history[-100:]
                    avg_score = (np.mean(recent_scores)
                                 if recent_scores else 0)
                    stats = self.policy.get_stats()
                    print(f"  Warmup {ep}/{warmup} | "
                          f"Avg Score: {avg_score:.1f} | "
                          f"Best: {self.best_score} | "
                          f"Explore: {stats['exploration_rate']:.1%}")
            print()

        remaining_episodes = total_episodes - warmup
        if remaining_episodes <= 0:
            elapsed = time.time() - start_time
            if save_path:
                self.save(save_path)
            return self._parallel_summary(elapsed, save_path, warmup)

        print(f"Phase 2: Parallel training ({remaining_episodes} episodes "
              f"across {workers} workers)...")

        episodes_done = 0
        episodes_dispatched = 0
        checkpoint_interval = log_interval * 10

        with multiprocessing.Pool(processes=workers) as pool:
            # Dispatch initial batch
            snapshot = self.policy.snapshot_for_workers()
            pending = []
            initial_dispatch = min(batch_size, remaining_episodes)
            for _ in range(initial_dispatch):
                args = (snapshot, self.game_file, self.frotz_path,
                        max_moves)
                future = pool.apply_async(play_episode_worker, (args,))
                pending.append(future)
                episodes_dispatched += 1

            # Process results as they arrive
            while episodes_done < remaining_episodes:
                ready = []
                still_pending = []
                for future in pending:
                    if future.ready():
                        ready.append(future)
                    else:
                        still_pending.append(future)

                if not ready:
                    try:
                        pending[0].get(timeout=0.5)
                        ready.append(pending[0])
                        still_pending = pending[1:]
                    except Exception:
                        # Timeout or error — check again
                        still_pending = pending[:]
                        ready = []
                        # Check for dead futures
                        new_pending = []
                        for f in still_pending:
                            try:
                                f.get(timeout=0.01)
                                ready.append(f)
                            except Exception as e:
                                if 'TimeoutError' in type(e).__name__:
                                    new_pending.append(f)
                                else:
                                    # Failed episode — count it
                                    episodes_done += 1
                                    self.episodes_completed += 1
                                    self.score_history.append(0)
                                    self.rooms_visited_history.append(0)
                        still_pending = new_pending

                pending = still_pending

                for future in ready:
                    try:
                        result = future.get()
                    except Exception:
                        # Failed episode
                        episodes_done += 1
                        self.episodes_completed += 1
                        self.score_history.append(0)
                        self.rooms_visited_history.append(0)
                        continue

                    self.episodes_completed += 1
                    episodes_done += 1
                    score = result.get('score', 0)
                    self.score_history.append(score)
                    rooms = result.get('rooms_visited', 0)
                    self.rooms_visited_history.append(rooms)
                    if score > self.best_score:
                        self.best_score = score

                    # Apply feedback
                    self.policy.apply_episode_result(
                        result.get('decisions', []),
                        score,
                        result.get('game_info', {}),
                        result.get('discovered_patterns'),
                        result.get('score_by_strategy'),
                    )

                    # Refresh snapshot
                    snapshot = self.policy.snapshot_for_workers()

                    # Launch replacement
                    if episodes_dispatched < remaining_episodes:
                        args = (snapshot, self.game_file,
                                self.frotz_path, max_moves)
                        future = pool.apply_async(
                            play_episode_worker, (args,))
                        pending.append(future)
                        episodes_dispatched += 1

                    # Log progress
                    total_done = warmup + episodes_done
                    if (episodes_done % log_interval == 0
                            or episodes_done == remaining_episodes):
                        recent_scores = self.score_history[-100:]
                        avg_score = (np.mean(recent_scores)
                                     if recent_scores else 0)
                        stats = self.policy.get_stats()
                        elapsed = time.time() - start_time
                        eps_per_sec = episodes_done / max(elapsed, 0.1)

                        print(
                            f"Episode {total_done}/{total_episodes} | "
                            f"Avg Score: {avg_score:.1f} | "
                            f"Best: {self.best_score} | "
                            f"Rooms: {np.mean(self.rooms_visited_history[-100:]):.1f} | "
                            f"Patterns: "
                            f"{stats['patterns_discovered']} | "
                            f"Explore: "
                            f"{stats['exploration_rate']:.1%} | "
                            f"{eps_per_sec:.1f} ep/s"
                        )

                    # Checkpoint
                    if (save_path
                            and episodes_done % checkpoint_interval == 0
                            and episodes_done < remaining_episodes):
                        self.save(save_path)

        elapsed = time.time() - start_time

        if save_path:
            self.save(save_path)

        return self._parallel_summary(elapsed, save_path, warmup)

    def _parallel_summary(self, elapsed: float, save_path: str = None,
                          warmup: int = 0) -> Dict:
        """Print training summary and return results dict."""
        final_stats = self.policy.get_stats()
        recent_scores = self.score_history[-100:]
        avg_score = np.mean(recent_scores) if recent_scores else 0

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE (parallel)")
        print(f"{'=' * 60}")
        print(f"  Episodes:         {self.episodes_completed}"
              + (f" ({warmup} warmup + "
                 f"{self.episodes_completed - warmup} parallel)"
                 if warmup > 0 else ""))
        print(f"  Average score:    {avg_score:.1f} / 350")
        print(f"  Best score:       {self.best_score} / 350")
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
            'avg_score': float(avg_score),
            'best_score': self.best_score,
            'patterns_learned': final_stats['patterns_discovered'],
            'elapsed_time': elapsed,
        }

    def save(self, path: str):
        """Save agent state."""
        os.makedirs(path, exist_ok=True)
        self.policy.save(path)

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
            'score_history': self.score_history[-1000:],
            'rooms_visited_history': self.rooms_visited_history[-1000:],
            'best_score': self.best_score,
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
            self.score_history = stats.get('score_history', [])
            self.rooms_visited_history = stats.get(
                'rooms_visited_history', [])
            self.best_score = stats.get('best_score', 0)
