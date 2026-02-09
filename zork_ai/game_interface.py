"""
Game Interface — Wraps dfrotz subprocess for programmatic Zork I play.

Provides FrotzInterface (subprocess management) and ZorkState (game snapshot).
Each interface instance owns one dfrotz process; multiple instances can run
concurrently in separate processes for parallel training.
"""

import os
import re
import select
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

from zork_ai.text_parser import (
    parse_visible_items,
    parse_enemies,
    parse_inventory,
    extract_room_name,
    extract_score_and_moves,
    detect_death,
    detect_darkness,
    categorize_room,
)


@dataclass
class ZorkState:
    """
    Snapshot of the current Zork game state.

    All fields are updated after every command sent to the interpreter.
    """
    # Current location
    location: str = ''
    description: str = ''

    # Score and progress
    score: int = 0
    moves: int = 0

    # Inventory
    inventory: List[str] = field(default_factory=list)

    # Exploration tracking
    visited_rooms: Set[str] = field(default_factory=set)
    items_collected: Set[str] = field(default_factory=set)

    # Danger indicators
    is_dark: bool = False
    is_dead: bool = False
    is_won: bool = False

    # Last interaction
    last_command: str = ''
    last_response: str = ''

    # Enemies present
    enemies: List[str] = field(default_factory=list)

    # Visible items in current room
    visible_items: List[str] = field(default_factory=list)


class FrotzInterface:
    """
    Manages a dfrotz subprocess for programmatic text adventure play.

    Usage:
        fi = FrotzInterface('/path/to/zork1.z5')
        opening = fi.start()
        response = fi.send_command('open mailbox')
        ...
        fi.close()

    Or as a context manager:
        with FrotzInterface('/path/to/zork1.z5') as fi:
            fi.start()
            fi.send_command('look')
    """

    def __init__(self, game_file: str, frotz_path: str = 'dfrotz',
                 read_timeout: float = 3.0):
        self.game_file = os.path.abspath(game_file)
        self.frotz_path = frotz_path
        self.read_timeout = read_timeout

        self._process: Optional[subprocess.Popen] = None
        self.state = ZorkState()

        # Track score changes for credit assignment
        self._prev_score: int = 0
        self._score_delta_this_turn: int = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()

    def start(self) -> str:
        """Launch dfrotz subprocess and return the opening text."""
        if self._process is not None:
            self.close()

        # -p flag: plain output (no bold/color escape codes)
        self._process = subprocess.Popen(
            [self.frotz_path, '-p', self.game_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Read opening text
        opening = self._read_until_prompt()

        # Parse initial state
        self._update_state('', opening)

        return opening

    def send_command(self, command: str) -> str:
        """
        Send a command to the running game and return the response.

        Raises RuntimeError if the process isn't running.
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("dfrotz process not running")

        command = command.strip()

        # Write command
        try:
            self._process.stdin.write((command + '\n').encode('utf-8'))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.state.is_dead = True
            return ''

        # Read response
        response = self._read_until_prompt()

        # Update state
        self._update_state(command, response)

        return response

    def get_score(self) -> int:
        """Get current score by sending the 'score' command."""
        response = self.send_command('score')
        # Parse "Your score is X (total of 350 points), in Y moves."
        match = re.search(r'your score is (\d+)', response, re.I)
        if match:
            return int(match.group(1))
        return self.state.score

    def get_location(self) -> str:
        """Get current location name."""
        return self.state.location

    def get_inventory(self) -> List[str]:
        """Get current inventory by sending 'inventory' command."""
        response = self.send_command('inventory')
        items = parse_inventory(response)
        self.state.inventory = items
        return items

    def reset(self) -> str:
        """Kill process and restart for a new episode."""
        self.close()
        self.state = ZorkState()
        self._prev_score = 0
        self._score_delta_this_turn = 0
        return self.start()

    def close(self):
        """Terminate the dfrotz subprocess."""
        if self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @property
    def is_running(self) -> bool:
        """Check if the dfrotz process is still alive."""
        return (self._process is not None
                and self._process.poll() is None)

    @property
    def score_delta(self) -> int:
        """Score change from the last command."""
        return self._score_delta_this_turn

    def _read_until_prompt(self) -> str:
        """
        Read from dfrotz stdout until we see the '>' prompt.

        Uses select() for non-blocking reads with a proper timeout.
        The previous implementation used blocking read(1) which could
        hang forever if dfrotz stalled without closing stdout.
        """
        output = []
        start = time.time()
        fd = self._process.stdout.fileno()

        while time.time() - start < self.read_timeout:
            # Wait up to 0.5s for data to be available
            remaining = self.read_timeout - (time.time() - start)
            if remaining <= 0:
                break
            wait = min(remaining, 0.5)

            try:
                ready, _, _ = select.select([fd], [], [], wait)
            except (ValueError, OSError):
                # fd closed or invalid
                break

            if not ready:
                # No data available within wait period — check if
                # process is still alive
                if self._process.poll() is not None:
                    break
                continue

            try:
                byte = self._process.stdout.read(1)
                if not byte:
                    # Process terminated / EOF
                    break

                char = byte.decode('utf-8', errors='replace')
                output.append(char)

                # Check for prompt
                text = ''.join(output)
                if text.rstrip().endswith('>'):
                    break

            except Exception:
                break

        return ''.join(output)

    def _update_state(self, command: str, response: str):
        """Update ZorkState from the latest command/response pair."""
        self.state.last_command = command
        self.state.last_response = response

        # Extract room name from status line
        room = extract_room_name(response)
        if room:
            self.state.location = room
            self.state.visited_rooms.add(room.lower())

        # Extract score and moves from status line
        score, moves = extract_score_and_moves(response)
        if score is not None:
            self._score_delta_this_turn = score - self._prev_score
            self._prev_score = score
            self.state.score = score
        else:
            self._score_delta_this_turn = 0

        if moves is not None:
            self.state.moves = moves

        # Check for death
        if detect_death(response):
            self.state.is_dead = True

        # Check for darkness
        self.state.is_dark = detect_darkness(response)

        # Parse enemies
        self.state.enemies = parse_enemies(response)

        # Parse visible items
        self.state.visible_items = parse_visible_items(response)

        # Update description (everything between status line and prompt)
        lines = response.split('\n')
        desc_lines = []
        skip_status = True
        for line in lines:
            if skip_status and re.match(
                    r'^.+\s+Score:\s*\d+\s+Moves:\s*\d+', line):
                skip_status = False
                continue
            if not skip_status and line.strip() != '>':
                desc_lines.append(line)
        self.state.description = '\n'.join(desc_lines).strip()

        # Track items collected (from inventory updates after 'take' commands)
        if command.lower().startswith('take') and \
           'taken' in response.lower():
            # Extract what was taken
            item = command[5:].strip() if len(command) > 5 else ''
            if item:
                self.state.items_collected.add(item.lower())

        # Check for win condition (score = 350 in Zork I)
        if self.state.score >= 350:
            self.state.is_won = True

        # Process died
        if self._process is not None and self._process.poll() is not None:
            self.state.is_dead = True
