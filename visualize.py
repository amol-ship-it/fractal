#!/usr/bin/env python3
"""
Quick-start visualizer launcher.

Usage:
    python visualize.py                              # Rush vs Defensive
    python visualize.py rush economy                 # Rush vs Economy
    python visualize.py random random --size 16      # Random vs Random on 16x16
    python visualize.py pattern rush                 # Pattern AI vs Rush
    python visualize.py ppo rush                     # PPO AI vs Rush
    python visualize.py economy pattern              # Economy vs Pattern AI
"""

import argparse
import subprocess
import sys
import os
import webbrowser
import time

AI_CHOICES = ['rush', 'economy', 'defensive', 'random', 'ppo', 'pattern']


def _ai_label(name):
    """Human-readable label for an AI name."""
    if name == 'ppo':
        return "YOUR AI (PPO)"
    if name == 'pattern':
        return "YOUR AI (PATTERN)"
    return name.upper()


def main():
    parser = argparse.ArgumentParser(description='Launch the MicroRTS game visualizer')
    parser.add_argument('p0', nargs='?', default='rush',
                        choices=AI_CHOICES,
                        help='Player 0 AI (default: rush)')
    parser.add_argument('p1', nargs='?', default='defensive',
                        choices=AI_CHOICES,
                        help='Player 1 AI (default: defensive)')
    parser.add_argument('--size', type=int, default=8, choices=[8, 12, 16],
                        help='Map size (default: 8)')
    parser.add_argument('--ticks', type=int, default=1000,
                        help='Max game ticks (default: 1000)')
    parser.add_argument('--port', type=int, default=8765,
                        help='HTTP port (default: 8765)')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to trained agent checkpoint '
                             '(default: checkpoints/ for ppo, '
                             'checkpoints_pattern/ for pattern)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open browser')
    args = parser.parse_args()

    # Auto-select checkpoint path based on AI type
    if args.checkpoint is None:
        if 'pattern' in (args.p0, args.p1):
            args.checkpoint = 'checkpoints_pattern/'
        else:
            args.checkpoint = 'checkpoints/'

    project_root = os.path.dirname(os.path.abspath(__file__))

    url = f"http://127.0.0.1:{args.port}"

    print(f"Starting MicroRTS Visualizer...")
    print(f"  Match: {_ai_label(args.p0)} vs {_ai_label(args.p1)}")
    print(f"  Map:   {args.size}x{args.size}")
    print(f"  URL:   {url}")

    cmd = [
        sys.executable, '-m', 'game.visualizer_server',
        '--p0', args.p0,
        '--p1', args.p1,
        '--map-size', str(args.size),
        '--max-ticks', str(args.ticks),
        '--port', str(args.port),
        '--checkpoint', args.checkpoint,
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    if not args.no_browser:
        # Open browser after a small delay to let server start
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(url)
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
