"""
Visualizer Backend Server

Runs MicroRTS games and serves game state as JSON over HTTP.
The frontend (visualizer.html) connects and streams the game state
frame-by-frame for rich canvas-based rendering.

Usage:
    python -m game.visualizer_server                       # Default: Rush vs Defensive, 8x8
    python -m game.visualizer_server --p0 rush --p1 economy
    python -m game.visualizer_server --map-size 16
    python -m game.visualizer_server --port 8765
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List, Optional

import numpy as np

from game.engine import GameEngine
from game.game_state import GameState
from game.ai_opponents import (
    RandomAI, RushAI, EconomyAI, DefensiveAI,
    PPOAgentWrapper, PatternAgentWrapper,
)
from game.units import UnitType, ActionState, UNIT_STATS

AI_REGISTRY = {
    'random': RandomAI,
    'rush': RushAI,
    'economy': EconomyAI,
    'defensive': DefensiveAI,
}

VALID_AI_NAMES = list(AI_REGISTRY.keys()) + ['ppo', 'pattern']


def serialize_unit(unit) -> Dict:
    """Serialize a single unit to JSON-friendly dict."""
    return {
        'id': unit.unit_id,
        'type': unit.unit_type.name.lower(),
        'type_id': int(unit.unit_type),
        'player': unit.player,
        'x': unit.x,
        'y': unit.y,
        'hp': unit.hp,
        'max_hp': unit.stats.hp,
        'resources_carried': unit.resources_carried,
        'action': unit.action_state.name.lower(),
        'action_id': int(unit.action_state),
        'ticks_remaining': unit.action_ticks_remaining,
        'target': list(unit.action_target) if unit.action_target else None,
        'is_structure': unit.stats.is_structure,
        'damage': unit.stats.damage,
        'attack_range': unit.stats.attack_range,
    }


def serialize_state(state: GameState) -> Dict:
    """Serialize complete game state to JSON-friendly dict."""
    gm = state.game_map
    units = [serialize_unit(u) for u in gm.units.values() if u.is_alive]

    p0_units = gm.get_player_units(0)
    p1_units = gm.get_player_units(1)

    def count_type(units_list, utype):
        return sum(1 for u in units_list if u.unit_type == utype)

    return {
        'tick': state.tick,
        'max_ticks': state.max_ticks,
        'done': state.done,
        'winner': state.winner,
        'map_width': gm.width,
        'map_height': gm.height,
        'units': units,
        'terrain': [[int(gm.terrain[y][x]) for x in range(gm.width)]
                     for y in range(gm.height)],
        'players': {
            '0': {
                'resources': state.player_resources[0],
                'total_units': len(p0_units),
                'workers': count_type(p0_units, UnitType.WORKER),
                'light': count_type(p0_units, UnitType.LIGHT),
                'heavy': count_type(p0_units, UnitType.HEAVY),
                'ranged': count_type(p0_units, UnitType.RANGED),
                'bases': count_type(p0_units, UnitType.BASE),
                'barracks': count_type(p0_units, UnitType.BARRACKS),
                'total_hp': sum(u.hp for u in p0_units),
            },
            '1': {
                'resources': state.player_resources[1],
                'total_units': len(p1_units),
                'workers': count_type(p1_units, UnitType.WORKER),
                'light': count_type(p1_units, UnitType.LIGHT),
                'heavy': count_type(p1_units, UnitType.HEAVY),
                'ranged': count_type(p1_units, UnitType.RANGED),
                'bases': count_type(p1_units, UnitType.BASE),
                'barracks': count_type(p1_units, UnitType.BARRACKS),
                'total_hp': sum(u.hp for u in p1_units),
            },
        },
    }


class GameSimulator:
    """Runs a game and stores all frames for playback."""

    def __init__(self, p0_ai_name: str = 'rush', p1_ai_name: str = 'defensive',
                 map_size: int = 8, max_ticks: int = 1000, speed: int = 1,
                 checkpoint_path: str = 'checkpoints/'):
        self.p0_ai_name = p0_ai_name
        self.p1_ai_name = p1_ai_name
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.speed = speed
        self.checkpoint_path = checkpoint_path
        self.frames: List[Dict] = []
        self.events: List[Dict] = []
        self.is_running = False
        self.is_complete = False

    def _create_ai(self, ai_name: str):
        """Create an AI instance, handling PPO and pattern specially."""
        if ai_name == 'ppo':
            return PPOAgentWrapper(self.checkpoint_path, map_size=self.map_size)
        if ai_name == 'pattern':
            return PatternAgentWrapper(self.checkpoint_path, map_size=self.map_size)
        return AI_REGISTRY[ai_name]()

    def run(self):
        """Run the full game simulation, recording every frame."""
        self.frames = []
        self.events = []
        self.is_running = True
        self.is_complete = False

        engine = GameEngine(map_size=self.map_size, max_ticks=self.max_ticks)
        state = engine.reset()

        p0_ai = self._create_ai(self.p0_ai_name)
        p1_ai = self._create_ai(self.p1_ai_name)

        # Record initial frame
        self.frames.append(serialize_state(state))

        prev_units_p0 = {u.unit_id for u in state.game_map.get_player_units(0)}
        prev_units_p1 = {u.unit_id for u in state.game_map.get_player_units(1)}

        while not state.done:
            p0_action = p0_ai.get_action(state, player=0)
            p1_action = p1_ai.get_action(state, player=1)
            state, info = engine.step(p0_action, p1_action)

            # Detect events
            curr_units_p0 = {u.unit_id for u in state.game_map.get_player_units(0)}
            curr_units_p1 = {u.unit_id for u in state.game_map.get_player_units(1)}

            for uid in prev_units_p0 - curr_units_p0:
                self.events.append({'tick': state.tick, 'type': 'unit_destroyed',
                                    'player': 0, 'unit_id': uid})
            for uid in curr_units_p0 - prev_units_p0:
                u = state.game_map.units.get(uid)
                if u:
                    self.events.append({'tick': state.tick, 'type': 'unit_created',
                                        'player': 0, 'unit_type': u.unit_type.name,
                                        'x': u.x, 'y': u.y})
            for uid in prev_units_p1 - curr_units_p1:
                self.events.append({'tick': state.tick, 'type': 'unit_destroyed',
                                    'player': 1, 'unit_id': uid})
            for uid in curr_units_p1 - prev_units_p1:
                u = state.game_map.units.get(uid)
                if u:
                    self.events.append({'tick': state.tick, 'type': 'unit_created',
                                        'player': 1, 'unit_type': u.unit_type.name,
                                        'x': u.x, 'y': u.y})

            prev_units_p0 = curr_units_p0
            prev_units_p1 = curr_units_p1

            self.frames.append(serialize_state(state))

        self.is_running = False
        self.is_complete = True


# ── Global state ──────────────────────────────────────────────
simulator: Optional[GameSimulator] = None
html_content: str = ""  # loaded once at startup
checkpoint_path: str = "checkpoints/"  # path to PPO agent checkpoint


class GameHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the visualizer."""

    def log_message(self, fmt, *args):
        # Suppress noisy request logging
        pass

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self._cors()
        self.end_headers()
        self.wfile.write(content.encode())

    # ── Routes ───────────────────────────────────────────────
    def do_GET(self):
        global simulator

        if self.path == '/' or self.path == '/index.html':
            self._html(html_content)

        elif self.path == '/api/status':
            if simulator is None:
                self._json({'status': 'no_game'})
            else:
                self._json({
                    'status': 'complete' if simulator.is_complete else 'running',
                    'total_frames': len(simulator.frames),
                    'p0': simulator.p0_ai_name,
                    'p1': simulator.p1_ai_name,
                    'map_size': simulator.map_size,
                })

        elif self.path.startswith('/api/frame/'):
            idx_str = self.path.split('/')[-1]
            try:
                idx = int(idx_str)
            except ValueError:
                self._json({'error': 'invalid frame index'}, 400)
                return
            if simulator is None or idx < 0 or idx >= len(simulator.frames):
                self._json({'error': 'frame not found'}, 404)
                return
            self._json(simulator.frames[idx])

        elif self.path == '/api/frames/all':
            if simulator is None:
                self._json({'error': 'no game'}, 404)
                return
            self._json({
                'frames': simulator.frames,
                'events': simulator.events,
                'p0': simulator.p0_ai_name,
                'p1': simulator.p1_ai_name,
                'map_size': simulator.map_size,
            })

        elif self.path.startswith('/api/new_game'):
            # Parse query params
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            p0 = params.get('p0', [simulator.p0_ai_name if simulator else 'rush'])[0]
            p1 = params.get('p1', [simulator.p1_ai_name if simulator else 'defensive'])[0]
            size = int(params.get('size', [simulator.map_size if simulator else 8])[0])
            ticks = int(params.get('ticks', ['1000'])[0])
            checkpoint = params.get('checkpoint', [checkpoint_path])[0]

            if p0 not in VALID_AI_NAMES or p1 not in VALID_AI_NAMES:
                self._json({'error': f'unknown AI. Options: {VALID_AI_NAMES}'}, 400)
                return

            simulator = GameSimulator(p0, p1, size, ticks, checkpoint_path=checkpoint)
            thread = threading.Thread(target=simulator.run, daemon=True)
            thread.start()
            self._json({'status': 'started', 'p0': p0, 'p1': p1, 'map_size': size})

        else:
            self._json({'error': 'not found'}, 404)


def main():
    global simulator, html_content, checkpoint_path

    parser = argparse.ArgumentParser(description='MicroRTS Game Visualizer')
    parser.add_argument('--p0', default='rush', choices=VALID_AI_NAMES,
                        help='Player 0 AI (default: rush)')
    parser.add_argument('--p1', default='defensive', choices=VALID_AI_NAMES,
                        help='Player 1 AI (default: defensive)')
    parser.add_argument('--map-size', type=int, default=8)
    parser.add_argument('--max-ticks', type=int, default=1000)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--checkpoint', default='checkpoints/',
                        help='Path to trained PPO agent checkpoint (default: checkpoints/)')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    # Validate checkpoints exist if trained AIs are selected
    if 'ppo' in (args.p0, args.p1):
        policy_file = os.path.join(args.checkpoint, 'policy.npz')
        if not os.path.exists(policy_file):
            print(f"ERROR: No trained PPO agent found at {args.checkpoint}")
            print(f"       Expected file: {policy_file}")
            print(f"")
            print(f"  Train an agent first:")
            print(f"    python -m rts_ai.train --timesteps 50000 --save-path {args.checkpoint}")
            print(f"")
            print(f"  Or use scripted AIs:")
            print(f"    python visualize.py rush defensive")
            sys.exit(1)

    if 'pattern' in (args.p0, args.p1):
        bindings_file = os.path.join(args.checkpoint, 'strategy_bindings.json')
        if not os.path.exists(bindings_file):
            print(f"ERROR: No trained pattern agent found at {args.checkpoint}")
            print(f"       Expected file: {bindings_file}")
            print(f"")
            print(f"  Train a pattern agent first:")
            print(f"    python -m rts_ai.train_pattern --episodes 500 --save-path {args.checkpoint}")
            print(f"")
            print(f"  Or use scripted AIs:")
            print(f"    python visualize.py rush defensive")
            sys.exit(1)

    # Load the HTML frontend
    html_path = os.path.join(os.path.dirname(__file__), 'visualizer.html')
    if not os.path.exists(html_path):
        print(f"ERROR: {html_path} not found")
        sys.exit(1)
    with open(html_path, 'r') as f:
        html_content = f.read()

    # Run initial game
    p0_label = "YOUR AI (PPO)" if args.p0 == 'ppo' else args.p0.upper()
    p1_label = "YOUR AI (PPO)" if args.p1 == 'ppo' else args.p1.upper()
    print(f"Simulating: {p0_label} vs {p1_label} on {args.map_size}x{args.map_size}...")
    simulator = GameSimulator(args.p0, args.p1, args.map_size, args.max_ticks,
                              checkpoint_path=args.checkpoint)
    simulator.run()
    print(f"Game complete: {len(simulator.frames)} frames recorded.")

    winner_label = {0: 'Player 0', 1: 'Player 1', 2: 'Draw', -1: 'Ongoing'}
    last = simulator.frames[-1]
    print(f"Winner: {winner_label.get(last['winner'], '?')}")

    # Start HTTP server
    server = HTTPServer(('127.0.0.1', args.port), GameHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n{'='*60}")
    print(f"  Visualizer running at: {url}")
    print(f"  Open this URL in your browser")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == '__main__':
    main()
