"""
Pattern Learning Visualizer — Backend Server

Reads training checkpoint data from ANY domain (Chess, RTS, or future games)
and serves it as JSON over HTTP. The frontend (visualizer.html) renders
interactive charts and dashboards.

The checkpoint format is universal across all game domains — the four pillars
of learning (Feedback, Approximability, Composability, Exploration) produce
identical file structures regardless of the game being played.

Usage:
    python -m chess_ai.visualizer_server                              # Default
    python -m chess_ai.visualizer_server --checkpoint checkpoints_pattern  # RTS
    python -m chess_ai.visualizer_server --port 8888                  # Custom port
"""

import argparse
import json
import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


# Strategy names per detected domain
DOMAIN_STRATEGIES = {
    'chess': [
        'DEVELOP', 'CONTROL_CENTER', 'ATTACK_KING',
        'TRADE_PIECES', 'PUSH_PAWNS', 'DEFEND',
        'CASTLE', 'ENDGAME_PUSH'
    ],
    'rts': [
        'HARVEST', 'BUILD_BARRACKS', 'PRODUCE_LIGHT',
        'PRODUCE_HEAVY', 'PRODUCE_RANGED', 'ATTACK'
    ],
    'zork': [
        'EXPLORE_NEW', 'EXPLORE_KNOWN', 'COLLECT_ITEMS',
        'USE_ITEM', 'DEPOSIT_TROPHY', 'FIGHT',
        'MANAGE_LIGHT', 'INTERACT'
    ],
}


def detect_domain(situation_keys):
    """Auto-detect the game domain from situation key format.

    Chess keys look like:  mat+_mid_saf_hi_ceq
    RTS keys look like:    w1_b0_c0_r1_e
    Zork keys look like:   loc_underground_s2_i1_d1_m
    """
    if not situation_keys:
        return 'generic'
    sample = situation_keys[0]
    if re.match(r'^mat[+\-=]+_', sample):
        return 'chess'
    if re.match(r'^w\d+_b\d+_', sample):
        return 'rts'
    if re.match(r'^loc_\w+_s\d+_', sample):
        return 'zork'
    return 'generic'


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves checkpoint data as JSON + the HTML dashboard."""

    checkpoint_path = 'checkpoints_chess'
    html_content = None  # Loaded once at startup

    def log_message(self, format, *args):
        """Suppress default request logging for cleaner output."""
        pass

    def _json(self, data, status=200):
        """Send a JSON response."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content):
        """Send an HTML response."""
        body = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _load_json_file(self, filename):
        """Load a JSON file from the checkpoint directory."""
        path = os.path.join(self.checkpoint_path, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '':
            if self.html_content:
                self._html(self.html_content)
            else:
                self._json({'error': 'HTML file not loaded'}, 500)

        elif path == '/api/data':
            # Return all data in a single blob
            bindings = self._load_json_file('strategy_bindings.json') or {}
            patterns = self._load_json_file('patterns.json') or {}
            stats = self._load_json_file('training_stats.json') or {}
            exploration = self._load_json_file('exploration.json') or {}

            # Auto-detect domain from situation key format
            situation_keys = list(bindings.keys())
            domain = detect_domain(situation_keys)

            # Get strategy names for this domain
            strategy_names = DOMAIN_STRATEGIES.get(domain, [])
            if not strategy_names and bindings:
                # Generic: infer count from first binding entry
                first = next(iter(bindings.values()), {})
                n = len(first)
                strategy_names = [f'STRATEGY_{i}' for i in range(n)]

            # Count total bindings
            num_bindings = sum(
                len(strats) for strats in bindings.values()
            )

            data = {
                'strategy_bindings': bindings,
                'patterns': patterns,
                'training_stats': stats,
                'exploration_rate': exploration.get('exploration_rate', 0.0),
                'strategy_names': strategy_names,
                'detected_domain': domain,
                'meta': {
                    'checkpoint_path': self.checkpoint_path,
                    'num_situations': len(bindings),
                    'num_patterns': len(patterns),
                    'num_bindings': num_bindings,
                    'episodes_completed': stats.get('episodes_completed', 0),
                }
            }
            self._json(data)

        elif path == '/api/bindings':
            data = self._load_json_file('strategy_bindings.json')
            self._json(data or {})

        elif path == '/api/patterns':
            data = self._load_json_file('patterns.json')
            self._json(data or {})

        elif path == '/api/stats':
            data = self._load_json_file('training_stats.json')
            self._json(data or {})

        else:
            self._json({'error': 'Not found'}, 404)


def main():
    parser = argparse.ArgumentParser(
        description='Pattern Learning Visualizer (any game domain)'
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints_chess',
        help='Path to checkpoint directory (default: checkpoints_chess)'
    )
    parser.add_argument(
        '--port', type=int, default=8877,
        help='HTTP server port (default: 8877)'
    )
    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.isdir(args.checkpoint):
        print(f'Error: checkpoint directory not found: {args.checkpoint}')
        sys.exit(1)

    # Load HTML file
    html_path = os.path.join(os.path.dirname(__file__), 'visualizer.html')
    if not os.path.exists(html_path):
        print(f'Error: visualizer.html not found at {html_path}')
        sys.exit(1)

    with open(html_path, 'r') as f:
        VisualizerHandler.html_content = f.read()

    VisualizerHandler.checkpoint_path = args.checkpoint

    server = HTTPServer(('', args.port), VisualizerHandler)
    print('=' * 60)
    print('  Recursive Learning AI — Pattern Visualizer')
    print('=' * 60)
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Server:     http://localhost:{args.port}')
    print(f'  Press Ctrl+C to stop')
    print('=' * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
        server.server_close()


if __name__ == '__main__':
    main()
