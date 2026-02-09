"""
Interactive Play Script - Watch the AI play or play against it.

Usage:
    python -m rts_ai.play                        # Watch AI vs Rush bot
    python -m rts_ai.play --mode watch           # Watch trained AI play
    python -m rts_ai.play --mode demo            # Quick demo game
    python -m rts_ai.play --mode evaluate        # Evaluate against all bots
    python -m rts_ai.play --mode transfer-demo   # Demo transfer learning
    python -m rts_ai.play --checkpoint path/     # Load trained model
"""

import argparse
import os
import sys
import time
import numpy as np

from game.engine import GameEngine, VecGameEnv
from game.game_state import GameState
from game.ai_opponents import RandomAI, RushAI, EconomyAI, DefensiveAI
from game.renderer import GameRenderer
from game.actions import DIMS_PER_CELL
from rts_ai.agent import PPOAgent
from rts_ai.encoder import GameStateEncoder
from rts_ai.knowledge_store import KnowledgeStore
from rts_ai.transfer import TransferBridge


def demo_game(args):
    """Run a quick demo game between two scripted AIs."""
    print("=" * 60)
    print("DEMO: Rush AI vs Defensive AI")
    print("=" * 60)

    engine = GameEngine(map_size=args.map_size, max_ticks=500)
    state = engine.reset()

    p0_ai = RushAI()
    p1_ai = DefensiveAI()

    print(f"\nInitial state:")
    print(GameRenderer.render(state))
    time.sleep(0.5)

    tick_display = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500]
    display_idx = 0

    while not state.done:
        p0_action = p0_ai.get_action(state, player=0)
        p1_action = p1_ai.get_action(state, player=1)
        state, info = engine.step(p0_action, p1_action)

        if display_idx < len(tick_display) and state.tick >= tick_display[display_idx]:
            print(f"\n--- Tick {state.tick} ---")
            print(GameRenderer.render(state))
            display_idx += 1

    print(f"\n{'=' * 60}")
    print(f"GAME OVER at tick {state.tick}")
    print(GameRenderer.render(state))

    if state.winner == 0:
        print("Rush AI (Player 0) wins!")
    elif state.winner == 1:
        print("Defensive AI (Player 1) wins!")
    else:
        print("Draw!")


def watch_ai(args):
    """Watch a trained AI play against a bot."""
    print("=" * 60)
    print("WATCH: Trained AI vs Bot")
    print("=" * 60)

    # Load agent
    agent = PPOAgent(map_height=args.map_size, map_width=args.map_size)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        print("No checkpoint loaded - using untrained agent")

    # Select opponent
    opponents = {
        'random': RandomAI(),
        'rush': RushAI(),
        'economy': EconomyAI(),
        'defensive': DefensiveAI(),
    }
    opponent = opponents.get(args.opponent, RushAI())
    print(f"Opponent: {args.opponent}")

    engine = GameEngine(map_size=args.map_size, max_ticks=500)
    state = engine.reset()
    h, w = state.game_map.height, state.game_map.width

    print(f"\nInitial state:")
    print(GameRenderer.render(state))

    step = 0
    while not state.done:
        # Get AI action
        obs = state.get_observation(player=0)
        action_mask = state.get_action_mask(player=0)

        obs_batch = obs[np.newaxis, ...]
        mask_batch = action_mask[np.newaxis, ...]

        actions, _, _ = agent.policy.get_action(
            obs_batch, mask_batch, deterministic=True
        )
        p0_action = actions[0].reshape(-1)

        # Get opponent action
        p1_action = opponent.get_action(state, player=1)

        state, info = engine.step(p0_action, p1_action)
        step += 1

        # Display at intervals
        if step % 25 == 0 or state.done:
            print(f"\n--- Step {step} (Tick {state.tick}) ---")
            print(GameRenderer.render(state))

    print(f"\n{'=' * 60}")
    print(f"GAME OVER")
    if state.winner == 0:
        print("AI (Player 0) wins!")
    elif state.winner == 1:
        print(f"{args.opponent.title()} AI (Player 1) wins!")
    else:
        print("Draw!")


def evaluate(args):
    """Evaluate the trained agent against all bot types."""
    print("=" * 60)
    print("EVALUATION: Trained AI vs All Bots")
    print("=" * 60)

    agent = PPOAgent(map_height=args.map_size, map_width=args.map_size)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        print("No checkpoint - using untrained agent")

    opponents = {
        'Random': RandomAI(),
        'Rush': RushAI(),
        'Economy': EconomyAI(),
        'Defensive': DefensiveAI(),
    }

    num_games = args.num_games
    results = {}

    for opp_name, opp_ai in opponents.items():
        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            engine = GameEngine(map_size=args.map_size, max_ticks=2000)
            state = engine.reset()
            h, w = state.game_map.height, state.game_map.width

            while not state.done:
                obs = state.get_observation(player=0)
                action_mask = state.get_action_mask(player=0)
                obs_batch = obs[np.newaxis, ...]
                mask_batch = action_mask[np.newaxis, ...]

                actions, _, _ = agent.policy.get_action(
                    obs_batch, mask_batch, deterministic=True
                )
                p0_action = actions[0].reshape(-1)
                p1_action = opp_ai.get_action(state, player=1)
                state, _ = engine.step(p0_action, p1_action)

            if state.winner == 0:
                wins += 1
            elif state.winner == 1:
                losses += 1
            else:
                draws += 1

        win_rate = wins / num_games
        results[opp_name] = {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': win_rate
        }
        print(f"  vs {opp_name:12s}: {wins}W / {losses}L / {draws}D  "
              f"(Win rate: {win_rate:.1%})")

    print(f"\nOverall win rate: "
          f"{sum(r['wins'] for r in results.values()) / (num_games * len(opponents)):.1%}")
    return results


def transfer_demo(args):
    """Demo the transfer learning system."""
    print("=" * 60)
    print("TRANSFER LEARNING DEMO")
    print("Showing how MicroRTS learnings transfer to other games")
    print("=" * 60)

    # Load knowledge
    knowledge_store = KnowledgeStore(
        store_path=os.path.join(args.checkpoint or 'checkpoints', 'knowledge')
    )
    transfer_bridge = TransferBridge(
        knowledge_store,
        concepts_path=os.path.join(args.checkpoint or 'checkpoints', 'concepts')
    )

    # If we have a trained agent, extract fresh knowledge
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent = PPOAgent(map_height=args.map_size, map_width=args.map_size)
        agent.load(args.checkpoint)
        knowledge_store.extract_from_patterns(agent.pattern_memory)
        transfer_bridge.extract_concepts_from_training(
            agent.pattern_memory, "micrortsai"
        )

    print(f"\nKnowledge Store Stats:")
    ks_stats = knowledge_store.get_stats()
    for key, val in ks_stats.items():
        print(f"  {key}: {val}")

    print(f"\nTransfer Bridge Stats:")
    tb_stats = transfer_bridge.get_stats()
    for key, val in tb_stats.items():
        print(f"  {key}: {val}")

    # Demo: Transfer to Age of Empires
    print("\n" + "=" * 60)
    print("TRANSFER TO: Age of Empires II")
    print("=" * 60)

    aoe_info = {
        'unit_types': ['Villager', 'Militia', 'Knight', 'Crossbowman',
                       'Siege Ram', 'Trebuchet', 'Scout Cavalry'],
        'resources': ['Food', 'Wood', 'Gold', 'Stone'],
        'buildings': ['Town Center', 'Barracks', 'Archery Range',
                      'Stable', 'Siege Workshop', 'Castle'],
        'map_features': ['forests', 'gold_mines', 'stone_mines',
                         'berries', 'fish', 'relics'],
    }

    adapter = transfer_bridge.generate_game_adapter(
        "micrortsai", "age_of_empires", aoe_info
    )

    print(f"\nUnit Mappings (MicroRTS -> AoE):")
    for role, unit in adapter['unit_mappings'].items():
        print(f"  {role:20s} -> {unit}")

    print(f"\nStrategy Recommendations:")
    for i, rec in enumerate(adapter['recommendations'], 1):
        print(f"\n  {i}. {rec['concept']}")
        print(f"     Score: {rec['score']:.2f}")
        print(f"     Level: {rec['abstraction_level']}")
        print(f"     {rec['description'][:100]}")

    print(f"\nTransferable Concepts:")
    for cid, mapping in adapter['concept_mappings'].items():
        if mapping['applicable']:
            print(f"  [{cid}] {mapping['name']} "
                  f"(effectiveness: {mapping['source_effectiveness']:.2f})")
            print(f"    Adaptation: {mapping['adaptation_notes']}")

    # Demo: Transfer to StarCraft-like game
    print("\n" + "=" * 60)
    print("TRANSFER TO: StarCraft-style Game")
    print("=" * 60)

    sc_recs = transfer_bridge.get_recommendations(
        "starcraft",
        current_conditions={
            "game_phase": "early",
            "enemy_aggression": "unknown",
        },
        top_k=3,
    )

    for i, rec in enumerate(sc_recs, 1):
        print(f"\n  {i}. {rec['concept']} (score: {rec['score']:.2f})")
        print(f"     {rec['description'][:100]}")
        if rec['expected_outcomes']:
            print(f"     Expected: {rec['expected_outcomes']}")


def main():
    parser = argparse.ArgumentParser(
        description='Play and evaluate the MicroRTS AI'
    )
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'watch', 'evaluate', 'transfer-demo'],
                        help='Play mode (default: demo)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--opponent', type=str, default='rush',
                        choices=['random', 'rush', 'economy', 'defensive'],
                        help='Opponent for watch mode (default: rush)')
    parser.add_argument('--map-size', type=int, default=8,
                        help='Map size (default: 8)')
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of evaluation games (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.mode == 'demo':
        demo_game(args)
    elif args.mode == 'watch':
        watch_ai(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'transfer-demo':
        transfer_demo(args)


if __name__ == '__main__':
    main()
