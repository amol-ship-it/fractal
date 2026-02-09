"""
Quick-start launcher for the Pattern Learning Visualizer.

Works with checkpoints from any game domain (Chess, RTS, or future games).
The visualizer auto-detects the domain from the checkpoint data format.

Usage:
    python -m chess_ai.visualize_chess                                 # Chess (default)
    python -m chess_ai.visualize_chess --checkpoint checkpoints_pattern # RTS
    python -m chess_ai.visualize_chess --port 8888                     # Custom port
"""

import argparse
import sys
import threading
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(
        description='Recursive Learning AI — Pattern Visualizer'
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints_chess',
        help='Path to checkpoint directory (default: checkpoints_chess)'
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

    # Import and run the server
    from chess_ai.visualizer_server import main as server_main

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

    server_main()


if __name__ == '__main__':
    main()
