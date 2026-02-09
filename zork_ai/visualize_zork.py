"""
Quick-start launcher for the Zork Pattern Learning Visualizer.

Opens an interactive browser dashboard showing strategy heatmaps,
training curves, pattern discovery, and strategy usage profiles
from Zork training checkpoints.

Usage:
    python -m zork_ai.visualize_zork
    python -m zork_ai.visualize_zork --checkpoint my_checkpoint/
    python -m zork_ai.visualize_zork --port 8888
"""

import argparse
import importlib.util
import os
import sys
import threading
import time
import webbrowser


def _import_visualizer_server():
    """Import visualizer_server.py directly by file path.

    This avoids importing the chess_ai package (which requires the
    `chess` library). The visualizer_server.py itself has no chess
    dependencies — it's a generic JSON+HTML server.
    """
    server_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'chess_ai', 'visualizer_server.py'
    )
    spec = importlib.util.spec_from_file_location(
        'visualizer_server', server_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(
        description='Zork AI — Pattern Learning Visualizer'
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints_zork',
        help='Path to checkpoint directory (default: checkpoints_zork)'
    )
    parser.add_argument(
        '--port', type=int, default=8877,
        help='HTTP server port (default: 8877)'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not open browser automatically'
    )
    args = parser.parse_args()

    # Import the server module directly (no chess dependency)
    server_mod = _import_visualizer_server()

    # Override sys.argv for the server's argparse
    sys.argv = [
        'visualizer_server',
        '--checkpoint', args.checkpoint,
        '--port', str(args.port),
    ]

    # Open browser after short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(0.8)
            webbrowser.open(f'http://localhost:{args.port}')
        threading.Thread(target=open_browser, daemon=True).start()

    server_mod.main()


if __name__ == '__main__':
    main()
