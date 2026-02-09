"""
Training Script - Train the AI agent to play the MicroRTS-style game.

Usage:
    python -m rts_ai.train                           # Train with defaults
    python -m rts_ai.train --timesteps 500000        # Custom timesteps
    python -m rts_ai.train --opponent rush           # Specific opponent
    python -m rts_ai.train --map-size 16             # 16x16 map
    python -m rts_ai.train --save-path checkpoints/  # Custom save path
    python -m rts_ai.train --resume checkpoints/     # Resume training
"""

import argparse
import os
import sys
import json
import time
import numpy as np

from game.engine import VecGameEnv
from game.ai_opponents import RandomAI, RushAI, EconomyAI, DefensiveAI
from game.renderer import GameRenderer
from rts_ai.agent import PPOAgent
from rts_ai.knowledge_store import KnowledgeStore
from rts_ai.transfer import TransferBridge


OPPONENTS = {
    'random': RandomAI,
    'rush': RushAI,
    'economy': EconomyAI,
    'defensive': DefensiveAI,
}


def create_curriculum_opponent(step: int, total_steps: int):
    """
    Curriculum learning: start with easy opponents, increase difficulty.

    Phase 1 (0-25%): Random AI
    Phase 2 (25-50%): Rush AI
    Phase 3 (50-75%): Economy AI
    Phase 4 (75-100%): Defensive AI
    """
    progress = step / total_steps
    if progress < 0.25:
        return RandomAI()
    elif progress < 0.50:
        return RushAI()
    elif progress < 0.75:
        return EconomyAI()
    else:
        return DefensiveAI()


def train(args):
    """Main training function."""
    print("=" * 70)
    print("RECURSIVE LEARNING AI - MicroRTS Training")
    print("=" * 70)

    # Select opponent
    if args.opponent == 'curriculum':
        opponent = RandomAI()  # Start with random, will be updated
        use_curriculum = True
    else:
        opponent = OPPONENTS[args.opponent]()
        use_curriculum = False

    # Create vectorized environment
    env = VecGameEnv(
        num_envs=args.num_envs,
        map_size=args.map_size,
        max_ticks=args.max_ticks,
        opponent_ai=opponent,
    )

    print(f"\nEnvironment: {args.map_size}x{args.map_size} map, "
          f"{args.num_envs} parallel envs")
    print(f"Opponent: {args.opponent}")
    print(f"Total timesteps: {args.timesteps}")

    # Create agent
    config = {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'lr': 2.5e-4,
        'n_steps': args.n_steps,
        'n_minibatches': 4,
        'n_epochs': 4,
        'hidden_dim': 64,
        'total_timesteps': args.timesteps,
    }

    agent = PPOAgent(
        map_height=args.map_size,
        map_width=args.map_size,
        config=config,
    )

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        agent.load(args.resume)
        print(f"  Loaded: {agent.total_steps} steps, "
              f"{agent.episodes_completed} episodes")

    # Knowledge store for persistent learning
    knowledge_store = KnowledgeStore(
        store_path=os.path.join(args.save_path, 'knowledge')
    )
    transfer_bridge = TransferBridge(
        knowledge_store,
        concepts_path=os.path.join(args.save_path, 'concepts')
    )

    # Training callback for curriculum and knowledge extraction
    def training_callback(agent_ref, update_num):
        # Curriculum opponent switching
        if use_curriculum:
            new_opponent = create_curriculum_opponent(
                agent_ref.total_steps, args.timesteps
            )
            env.opponent_ai = new_opponent

        # Periodically extract knowledge
        if update_num % 50 == 0:
            new_knowledge = knowledge_store.extract_from_patterns(
                agent_ref.pattern_memory
            )
            if new_knowledge > 0:
                print(f"  [Knowledge] Extracted {new_knowledge} new strategic entries")
                knowledge_store.save()

    # Show initial game state
    print("\nInitial game state:")
    state = env.engines[0].reset()
    print(GameRenderer.render(state))
    env.reset()  # Re-reset all envs

    print(f"\nTraining started...")
    start_time = time.time()

    # Train
    results = agent.train(
        env=env,
        total_timesteps=args.timesteps,
        log_interval=args.log_interval,
        save_path=args.save_path,
        callback=training_callback,
    )

    elapsed = time.time() - start_time

    # Final knowledge extraction
    knowledge_store.extract_from_patterns(agent.pattern_memory)
    concepts = transfer_bridge.extract_concepts_from_training(
        agent.pattern_memory, game_name="micrortsai"
    )
    knowledge_store.save()
    transfer_bridge.save()

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps:      {results['total_steps']}")
    print(f"  Episodes:         {results['episodes']}")
    print(f"  Final win rate:   {results['final_win_rate']:.1%}")
    print(f"  Final avg reward: {results['final_avg_reward']:.2f}")
    print(f"  Patterns learned: {results['patterns_learned']}")
    print(f"  Elapsed time:     {elapsed:.1f}s")

    print(f"\n  Knowledge store:")
    ks_stats = knowledge_store.get_stats()
    print(f"    Total entries:      {ks_stats['total_knowledge']}")
    print(f"    Categories:         {ks_stats['categories']}")
    print(f"    Avg effectiveness:  {ks_stats['avg_effectiveness']:.2f}")

    print(f"\n  Transfer concepts:")
    tb_stats = transfer_bridge.get_stats()
    print(f"    Universal:       {tb_stats['universal']}")
    print(f"    Genre-specific:  {tb_stats['genre_specific']}")
    print(f"    Game-specific:   {tb_stats['game_specific']}")

    # Show transfer recommendations for Age of Empires
    print(f"\n  Strategy recommendations for Age of Empires:")
    aoe_recs = transfer_bridge.get_recommendations(
        "age_of_empires",
        current_conditions={"game_phase": "early"},
        top_k=3,
    )
    for i, rec in enumerate(aoe_recs, 1):
        print(f"    {i}. {rec['concept']} (score: {rec['score']:.2f})")
        print(f"       {rec['description'][:80]}...")

    # Save final results
    with open(os.path.join(args.save_path, 'final_results.json'), 'w') as f:
        json.dump({
            'results': results,
            'config': config,
            'elapsed_seconds': elapsed,
            'knowledge_stats': ks_stats,
            'transfer_stats': tb_stats,
        }, f, indent=2)

    print(f"\nCheckpoint saved to: {args.save_path}")
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train AI agent to play MicroRTS-style game'
    )
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps (default: 100000)')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--map-size', type=int, default=8,
                        help='Map size (8 or 16, default: 8)')
    parser.add_argument('--max-ticks', type=int, default=2000,
                        help='Max game ticks per episode (default: 2000)')
    parser.add_argument('--opponent', type=str, default='rush',
                        choices=['random', 'rush', 'economy', 'defensive', 'curriculum'],
                        help='Opponent AI (default: rush)')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Steps per rollout (default: 128)')
    parser.add_argument('--save-path', type=str, default='checkpoints',
                        help='Save path for checkpoints (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='Log every N updates (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
