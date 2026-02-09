"""
Zork Pattern Policy — Uses the four pillars of learning to play Zork I.

Instead of a neural network, this policy:
1. Encodes the game state as a feature vector
2. Quantizes it into a situation key (e.g. "loc_underground_s2_i1_d1_mid")
3. Looks up the best strategy for that situation from learned bindings
4. Executes the strategy by selecting a concrete text command
5. Learns from score changes (feedback refines which situation->strategy pairs work)

The four pillars drive every decision:
- Feedback Loops: Score changes refine which situation->strategy pairs work
- Approximability: Pattern signatures improve with each episode
- Composability: Simple patterns compose into higher-level strategic concepts
- Exploration: Agent occasionally tries novel strategies to discover better ones

Follows the same architecture as rts_ai/pattern_policy.py but adapted for
turn-based text adventure (no strategy persistence, like chess_ai).
"""

import random
import json
import os
from typing import Dict, List, Tuple, Optional, Set
from enum import IntEnum

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory
from core.feedback import (
    FeedbackLoop, FeedbackType, FeedbackSignal, ExplorationStrategy
)

from zork_ai.text_encoder import ZorkStateEncoder
from zork_ai.game_interface import ZorkState
from zork_ai.text_parser import (
    categorize_room,
    identify_treasures,
    parse_exits,
    parse_visible_items,
    KNOWN_ITEMS,
    DIRECTIONS,
    DIR_ABBREVS,
)


# ── Strategies ──────────────────────────────────────────────────────────────

class ZorkStrategy(IntEnum):
    """
    The action vocabulary. The agent learns WHEN to use each strategy;
    the strategies define WHAT command to issue.
    """
    EXPLORE_NEW = 0       # Move to unvisited rooms
    EXPLORE_KNOWN = 1     # Revisit rooms with untried actions
    COLLECT_ITEMS = 2     # Pick up visible items
    USE_ITEM = 3          # Try items on objects (puzzle solving)
    DEPOSIT_TROPHY = 4    # Bring treasures to trophy case
    FIGHT = 5             # Attack enemies with weapons
    MANAGE_LIGHT = 6      # Handle lamp in dark areas
    INTERACT = 7          # Examine/open/read objects


STRATEGY_NAMES = [s.name for s in ZorkStrategy]
NUM_STRATEGIES = len(ZorkStrategy)


# ── Strategy Binding ────────────────────────────────────────────────────────

class StrategyBinding:
    """
    Links a situation key to a strategy, tracking how well it performs.
    This is what the agent LEARNS — which situation calls for which strategy.

    Uses Bayesian smoothing (same as RTS) to prevent premature lock-in.
    """

    def __init__(self, strategy: int, pattern_id: str):
        self.strategy = strategy
        self.pattern_id = pattern_id  # actually the situation key
        self.wins = 0.0   # float to support fractional credit
        self.losses = 0.0
        self.times_used = 0

    @property
    def confidence(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.5  # Uninformed prior
        # Bayesian smoothing with pseudo_count=2
        pseudo_count = 2
        return (self.wins + pseudo_count * 0.5) / (total + pseudo_count)

    def to_dict(self) -> dict:
        return {
            'strategy': self.strategy,
            'pattern_id': self.pattern_id,
            'wins': self.wins,
            'losses': self.losses,
            'times_used': self.times_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyBinding':
        b = cls(data['strategy'], data['pattern_id'])
        b.wins = data.get('wins', 0)
        b.losses = data.get('losses', 0)
        b.times_used = data.get('times_used', 0)
        return b


# ── Situation Key ───────────────────────────────────────────────────────────

def _situation_key_from_state(state: ZorkState,
                              inventory: List[str] = None) -> str:
    """
    Quantize the game state into a discrete situation key.

    This is the APPROXIMABILITY pillar: similar game states map to the
    same key, so the agent can generalize across similar situations.

    Format: loc_{category}_s{score_bin}_i{inv_bin}_d{danger}_p{phase}

    Location: 8 categories (house, forest, cave_entrance, underground,
              river, maze, treasure_room, other)
    Score bin: 0-6 (each ~50 points)
    Inventory bin: 0-3 (0, 1-2, 3-5, 6+)
    Danger: 0 or 1 (dark or enemy present)
    Phase: early/mid/late
    """
    # Location category
    loc = categorize_room(state.location)

    # Score bin (0-350, binned by 50)
    s_bin = min(state.score // 50, 6)

    # Inventory bin
    inv = inventory if inventory is not None else state.inventory
    n_items = len(inv)
    if n_items == 0:
        i_bin = 0
    elif n_items <= 2:
        i_bin = 1
    elif n_items <= 5:
        i_bin = 2
    else:
        i_bin = 3

    # Danger indicator
    danger = 1 if (state.is_dark or state.enemies) else 0

    # Phase: early (<50), mid (50-200), late (200+)
    if state.score < 50:
        phase = 'e'
    elif state.score < 200:
        phase = 'm'
    else:
        phase = 'l'

    return f"loc_{loc}_s{s_bin}_i{i_bin}_d{danger}_{phase}"


# ── Strategy Executors ──────────────────────────────────────────────────────

class RoomMemory:
    """Per-episode memory of rooms visited and actions tried."""

    def __init__(self):
        self.room_visits: Dict[str, int] = {}           # room -> visit count
        self.room_exits_tried: Dict[str, Set[str]] = {} # room -> set of dirs
        self.room_items_tried: Dict[str, Set[str]] = {} # room -> items tried
        self.room_objects_tried: Dict[str, Set[str]] = {} # room -> objects
        self.recent_rooms: List[str] = []                # last N rooms
        self.failed_commands: Set[str] = set()           # commands that failed

    def visit_room(self, room: str):
        room_lower = room.lower()
        self.room_visits[room_lower] = self.room_visits.get(room_lower, 0) + 1
        self.recent_rooms.append(room_lower)
        if len(self.recent_rooms) > 20:
            self.recent_rooms.pop(0)

    def try_exit(self, room: str, direction: str):
        room_lower = room.lower()
        if room_lower not in self.room_exits_tried:
            self.room_exits_tried[room_lower] = set()
        self.room_exits_tried[room_lower].add(direction.lower())

    def get_untried_exits(self, room: str,
                          available_exits: List[str]) -> List[str]:
        room_lower = room.lower()
        tried = self.room_exits_tried.get(room_lower, set())
        return [d for d in available_exits if d.lower() not in tried]

    def is_new_room(self, room: str) -> bool:
        return room.lower() not in self.room_visits


def _all_possible_exits(available_exits: List[str]) -> List[str]:
    """Combine parsed exits with standard cardinal directions.

    The text parser can only find exits mentioned in the description.
    In Zork, many exits exist without being explicitly stated (e.g. you
    can always try north/south/east/west). Add cardinal directions so
    the agent explores them even when not mentioned.
    """
    standard = ['north', 'south', 'east', 'west', 'up', 'down',
                'northeast', 'northwest', 'southeast', 'southwest',
                'enter', 'in']
    combined = set(available_exits)
    combined.update(standard)
    return sorted(combined)


def execute_explore_new(state: ZorkState, room_memory: RoomMemory,
                        available_exits: List[str]) -> str:
    """EXPLORE_NEW: Move to unvisited rooms by trying untried exits.

    Note: _get_exploration_unlock() and _get_early_game_navigation() are
    now called in get_command() BEFORE strategy executors, so they fire
    from ALL strategies, not just EXPLORE_NEW.
    """
    all_exits = _all_possible_exits(available_exits)

    # Prefer untried exits
    untried = room_memory.get_untried_exits(state.location, all_exits)

    if untried:
        # Prioritize parsed exits (more likely to work) over cardinal guesses
        parsed_untried = [d for d in untried if d in available_exits]
        if parsed_untried:
            direction = random.choice(parsed_untried)
        else:
            direction = random.choice(untried)
    elif available_exits:
        direction = random.choice(available_exits)
    else:
        direction = random.choice(['north', 'south', 'east', 'west'])

    return direction


def _get_early_game_navigation(state: ZorkState,
                                room_memory: RoomMemory) -> Optional[str]:
    """
    Deterministic navigation for the EARLY GAME — guide the agent from the
    starting location (West of House) through the house circuit to get inside.

    The critical path in Zork I:
      West of House → (north) → North of House → (east) → Behind House
      → open window → enter → Kitchen (inside the house, where the game opens up)

    Without this, the agent randomly wanders into the forest loop ~70% of the
    time and never scores. This function fires BEFORE strategy executors and
    provides a deterministic step along the critical path.

    Only fires in the EARLY GAME (score == 0 and visited_rooms < 8).
    Once the agent has entered the house and scored, this defers to learned
    strategies.
    """
    # Only guide early game — once agent is scoring or has explored enough,
    # let learned strategies take over
    if state.score > 0:
        return None
    if len(room_memory.room_visits) > 8:
        return None

    loc_lower = state.location.lower() if state.location else ''

    # ── HOUSE CIRCUIT: deterministic path to get inside ──

    # West of House (starting location): go north to reach North of House
    if 'west of house' in loc_lower:
        # First visit: open mailbox (quick interaction, gets leaflet)
        desc_lower = state.description.lower()
        tried = room_memory.room_objects_tried.get(loc_lower, set())
        if 'mailbox' in desc_lower and 'open_mailbox' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('open_mailbox')
            return 'open mailbox'
        # Then head north toward Behind House
        return 'north'

    # North of House: go east to reach Behind House
    if 'north of house' in loc_lower:
        return 'east'

    # Behind House / East of House: open window and enter
    if 'behind house' in loc_lower or 'east of house' in loc_lower:
        desc_lower = state.description.lower()
        tried = room_memory.room_objects_tried.get(loc_lower, set())
        if 'window' in desc_lower:
            # Need to open window if closed
            if 'closed' in desc_lower or ('open' not in desc_lower
                                           and 'window is slightly' not in desc_lower):
                if 'open_window' not in tried:
                    if loc_lower not in room_memory.room_objects_tried:
                        room_memory.room_objects_tried[loc_lower] = set()
                    room_memory.room_objects_tried[loc_lower].add('open_window')
                    return 'open window'
            # Window is open — enter
            if 'enter_window' not in tried:
                if loc_lower not in room_memory.room_objects_tried:
                    room_memory.room_objects_tried[loc_lower] = set()
                room_memory.room_objects_tried[loc_lower].add('enter_window')
                return 'enter'
        # Fallback: try entering anyway
        return 'enter'

    # South of House: go east or north to get back on track
    if 'south of house' in loc_lower:
        return 'east'  # Goes to Behind House in Zork I

    # Forest rooms: head back toward the house
    if 'forest' in loc_lower:
        # Most forest rooms have paths back to the house perimeter
        # Priority: west (often leads back to house area)
        return 'west'

    # Clearing: go south or west
    if 'clearing' in loc_lower:
        return 'south'

    return None


def _get_exploration_unlock(state: ZorkState,
                            room_memory: RoomMemory) -> Optional[str]:
    """
    Return a command that unlocks new areas if we're at a known gate-point.

    Zork has several key chokepoints where you MUST do a specific action
    to access the rest of the game. Without these, the agent loops forever
    in the starting area.

    This fires AFTER early-game navigation and handles mid/late-game
    chokepoints like the trap door, cellar access, etc.
    """
    loc_lower = state.location.lower() if state.location else ''
    desc_lower = state.description.lower()
    tried = room_memory.room_objects_tried.get(loc_lower, set())

    # Behind House: open window + enter (THE critical entry to most of the game)
    if 'behind house' in loc_lower or 'east of house' in loc_lower:
        if 'window' in desc_lower:
            if 'closed' in desc_lower or ('open' not in desc_lower
                                           and 'window is slightly' not in desc_lower):
                if 'open_window' not in tried:
                    if loc_lower not in room_memory.room_objects_tried:
                        room_memory.room_objects_tried[loc_lower] = set()
                    room_memory.room_objects_tried[loc_lower].add('open_window')
                    return 'open window'
            if 'enter_window' not in tried:
                if loc_lower not in room_memory.room_objects_tried:
                    room_memory.room_objects_tried[loc_lower] = set()
                room_memory.room_objects_tried[loc_lower].add('enter_window')
                return 'enter'

    # West of House: open mailbox (gets leaflet, first interaction)
    if 'west of house' in loc_lower:
        if 'mailbox' in desc_lower and 'open_mailbox' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('open_mailbox')
            return 'open mailbox'

    # Living Room: move rug, open trap door (access underground)
    if 'living room' in loc_lower:
        if 'rug' in desc_lower and 'move_rug' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('move_rug')
            return 'move rug'
        if ('trap door' in desc_lower or 'trapdoor' in desc_lower):
            if 'open_trapdoor' not in tried:
                if loc_lower not in room_memory.room_objects_tried:
                    room_memory.room_objects_tried[loc_lower] = set()
                room_memory.room_objects_tried[loc_lower].add('open_trapdoor')
                return 'open trap door'

    # Kitchen: take items, then go west to Living Room (for trophy case & underground)
    if 'kitchen' in loc_lower:
        # Take the sack and bottle in kitchen (important items)
        if 'take_kitchen_items' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('take_kitchen_items')
            return 'take all'
        # Go west to Living Room (NOT down — "down" doesn't work in Kitchen)
        if 'go_living_room' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('go_living_room')
            return 'west'

    # Attic: take knife and rope (useful items), then go back down
    if 'attic' in loc_lower:
        if 'take_attic_items' not in tried:
            if loc_lower not in room_memory.room_objects_tried:
                room_memory.room_objects_tried[loc_lower] = set()
            room_memory.room_objects_tried[loc_lower].add('take_attic_items')
            return 'take all'

    return None


def execute_explore_known(state: ZorkState, room_memory: RoomMemory,
                          available_exits: List[str]) -> str:
    """EXPLORE_KNOWN: Revisit rooms with untried paths or objects."""
    all_exits = _all_possible_exits(available_exits)

    # Prefer exits we haven't tried from THIS room
    untried = room_memory.get_untried_exits(state.location, all_exits)
    if untried:
        # Prioritize parsed exits first
        parsed_untried = [d for d in untried if d in available_exits]
        if parsed_untried:
            return random.choice(parsed_untried)
        return random.choice(untried)

    # All exits tried — go somewhere we've been less
    if available_exits:
        return random.choice(available_exits)

    return random.choice(['north', 'south', 'east', 'west'])


def execute_collect_items(state: ZorkState, room_memory: RoomMemory,
                          available_exits: List[str]) -> str:
    """COLLECT_ITEMS: Pick up visible items, then move to find more."""
    room_lower = state.location.lower() if state.location else ''

    # Pick up specifically named visible items first (more reliable than take all)
    if state.visible_items:
        items_tried = room_memory.room_items_tried.get(room_lower, set())
        for item in state.visible_items:
            if item.lower() not in items_tried:
                if room_lower not in room_memory.room_items_tried:
                    room_memory.room_items_tried[room_lower] = set()
                room_memory.room_items_tried[room_lower].add(item.lower())
                return f"take {item}"

    # Try 'take all' once per room as a sweep
    items_tried = room_memory.room_items_tried.get(room_lower, set())
    if 'take_all' not in items_tried:
        if room_lower not in room_memory.room_items_tried:
            room_memory.room_items_tried[room_lower] = set()
        room_memory.room_items_tried[room_lower].add('take_all')
        return "take all"

    # Nothing left to collect here — move to a new room to find more
    all_exits = _all_possible_exits(available_exits)
    untried = room_memory.get_untried_exits(state.location, all_exits)
    if untried:
        parsed_untried = [d for d in untried if d in available_exits]
        if parsed_untried:
            return random.choice(parsed_untried)
        return random.choice(untried)
    if available_exits:
        return random.choice(available_exits)
    return random.choice(['north', 'south', 'east', 'west'])


def execute_use_item(state: ZorkState, room_memory: RoomMemory,
                     available_exits: List[str]) -> str:
    """USE_ITEM: Try items on objects (puzzle solving)."""
    inventory = state.inventory

    if not inventory:
        # No inventory — explore instead of wasting a turn on "look"
        if available_exits:
            return random.choice(available_exits)
        return random.choice(['north', 'south', 'east', 'west'])

    # Context-sensitive item usage
    desc_lower = state.description.lower()
    loc_lower = state.location.lower()

    # If we're at the trophy case and have treasures
    if 'living room' in loc_lower or 'trophy case' in loc_lower:
        treasures = identify_treasures(inventory)
        if treasures:
            for item in inventory:
                item_lower = item.lower()
                for t in treasures:
                    if t in item_lower:
                        return f"put {item} in trophy case"

    # Specific puzzle interactions based on known Zork I solutions
    puzzle_actions = _get_puzzle_actions(state, inventory)
    if puzzle_actions:
        return random.choice(puzzle_actions)

    # Generic: try using a random item on a visible object
    if state.visible_items:
        item = random.choice(inventory)
        obj = random.choice(state.visible_items)
        actions = [
            f"put {item} in {obj}",
            f"unlock {obj} with {item}",
            f"open {obj} with {item}",
        ]
        return random.choice(actions)

    # Try examining something
    if inventory:
        return f"examine {random.choice(inventory)}"

    return "look"


def _get_puzzle_actions(state: ZorkState,
                        inventory: List[str]) -> List[str]:
    """Generate context-sensitive puzzle actions for known Zork I scenarios."""
    actions = []
    desc_lower = state.description.lower()
    loc_lower = state.location.lower()
    inv_lower = [i.lower() for i in inventory]

    # Mailbox at West of House
    if 'mailbox' in desc_lower:
        actions.extend(['open mailbox', 'take leaflet', 'read leaflet'])

    # Window at Behind House (key entry point!)
    if 'window' in desc_lower:
        if 'open' in desc_lower and 'window' in desc_lower:
            actions.append('enter')
        else:
            actions.extend(['open window', 'enter'])

    # Kitchen with sack and bottle
    if 'kitchen' in loc_lower:
        actions.extend(['open sack', 'take all'])

    # Troll Room: fight with sword
    if 'troll' in desc_lower and any('sword' in i for i in inv_lower):
        actions.append('attack troll with sword')

    # Loud Room: echo
    if 'loud room' in loc_lower:
        actions.append('echo')

    # Dam: turn bolt with wrench
    if 'dam' in loc_lower:
        if any('wrench' in i for i in inv_lower):
            actions.append('turn bolt with wrench')
        if any('screwdriver' in i for i in inv_lower):
            actions.append('turn yellow button')

    # Inflatable boat
    if any('pump' in i for i in inv_lower) and \
       any('boat' in i or 'pile' in i for i in inv_lower):
        actions.append('inflate boat with pump')

    # Bell/book/candles for Hades
    if 'hades' in loc_lower or 'entrance to hades' in loc_lower:
        if any('bell' in i for i in inv_lower):
            actions.append('ring bell')
        if any('candles' in i for i in inv_lower):
            actions.append('light candles with match')
        if any('book' in i for i in inv_lower):
            actions.append('read book')

    # Cyclops: give lunch
    if 'cyclops' in desc_lower:
        if any('lunch' in i or 'food' in i for i in inv_lower):
            actions.append('give lunch to cyclops')
        actions.append('odysseus')  # Magic word to scare cyclops

    # Dark places: turn on lamp
    if state.is_dark and any('lamp' in i or 'lantern' in i for i in inv_lower):
        actions.append('turn on lamp')

    # End of Rainbow: wave sceptre
    if 'rainbow' in loc_lower or 'end of rainbow' in loc_lower:
        if any('sceptre' in i for i in inv_lower):
            actions.append('wave sceptre')

    # Coal mine / shaft: put coal in machine
    if 'machine' in desc_lower:
        if any('coal' in i for i in inv_lower):
            actions.extend(['open lid', 'put coal in machine',
                            'close lid', 'turn switch'])

    return actions


def execute_deposit_trophy(state: ZorkState, room_memory: RoomMemory,
                           available_exits: List[str]) -> str:
    """DEPOSIT_TROPHY: Navigate to living room and deposit treasures."""
    loc_lower = state.location.lower()

    # If we're in the living room, deposit treasures
    if 'living room' in loc_lower:
        treasures = identify_treasures(state.inventory)
        if treasures:
            for item in state.inventory:
                item_lower = item.lower()
                for t in treasures:
                    if t in item_lower:
                        return f"put {item} in trophy case"

        # No treasures to deposit — go explore
        return random.choice(available_exits) if available_exits else 'west'

    # Navigate toward the living room
    # Living room is accessible from the house area
    if 'kitchen' in loc_lower:
        return 'west'  # Kitchen → Living Room
    if 'west of house' in loc_lower:
        return 'east'  # Try to enter via back
    if 'behind house' in loc_lower or 'east of house' in loc_lower:
        return 'enter'
    if 'attic' in loc_lower:
        return 'down'

    # Generic: try to head toward the house
    if 'north' in [e.lower() for e in available_exits]:
        return 'north'
    if available_exits:
        return random.choice(available_exits)

    return 'up'  # Underground? Go up


def execute_fight(state: ZorkState, room_memory: RoomMemory,
                  available_exits: List[str]) -> str:
    """FIGHT: Attack enemies with weapons."""
    if not state.enemies:
        # No enemy here — explore instead of wasting a turn
        if available_exits:
            return random.choice(available_exits)
        return random.choice(['north', 'south', 'east', 'west'])

    enemy = state.enemies[0]
    inv_lower = [i.lower() for i in state.inventory]

    # Use the best weapon available
    if any('sword' in i for i in inv_lower):
        return f"attack {enemy} with sword"
    if any('knife' in i for i in inv_lower):
        return f"attack {enemy} with knife"

    # No weapon — try to flee
    if available_exits:
        return random.choice(available_exits)

    # Last resort: kick
    return f"attack {enemy}"


def execute_manage_light(state: ZorkState, room_memory: RoomMemory,
                         available_exits: List[str]) -> str:
    """MANAGE_LIGHT: Handle lamp in dark areas."""
    inv_lower = [i.lower() for i in state.inventory]

    # Turn on lamp if we have one and it's dark
    if any('lamp' in i or 'lantern' in i for i in inv_lower):
        if state.is_dark:
            return "turn on lamp"
        # Lamp is already on and it's not dark — move on
        if available_exits:
            return random.choice(available_exits)
        return random.choice(['north', 'south', 'east', 'west'])

    # No lamp — avoid dark areas by going back
    if state.is_dark and available_exits:
        return random.choice(available_exits)

    # Try to find a lamp
    if state.visible_items:
        for item in state.visible_items:
            if 'lamp' in item.lower() or 'lantern' in item.lower():
                return f"take {item}"

    # Can't do anything light-related — explore
    if available_exits:
        return random.choice(available_exits)
    return random.choice(['north', 'south', 'east', 'west'])


def execute_interact(state: ZorkState, room_memory: RoomMemory,
                     available_exits: List[str]) -> str:
    """INTERACT: Examine/open/read objects in the room."""
    desc_lower = state.description.lower()
    room_lower = state.location.lower()

    # Track what we've already tried in this room
    if room_lower not in room_memory.room_objects_tried:
        room_memory.room_objects_tried[room_lower] = set()
    tried = room_memory.room_objects_tried[room_lower]

    # Try interacting with visible items/objects
    interactable_objects = []

    # Check for known interactable objects
    objects = [
        'mailbox', 'door', 'window', 'trapdoor', 'trap door',
        'grating', 'gate', 'rug', 'carpet', 'leaves',
        'table', 'desk', 'pedestal', 'altar', 'basket',
        'case', 'trophy case', 'chest', 'nest',
        'dam', 'button', 'switch', 'lever', 'mirror',
        'machine', 'lid', 'bolt', 'boat', 'tree',
    ]

    for obj in objects:
        if obj in desc_lower and obj not in tried:
            interactable_objects.append(obj)

    if interactable_objects:
        obj = random.choice(interactable_objects)
        tried.add(obj)
        actions = [
            f"examine {obj}",
            f"open {obj}",
            f"look in {obj}",
            f"move {obj}",
        ]
        return random.choice(actions)

    # Try examining visible items we haven't examined
    if state.visible_items:
        for item in state.visible_items:
            if item not in tried:
                tried.add(item)
                return f"examine {item}"

    # Examine inventory items
    if state.inventory:
        item = random.choice(state.inventory)
        return f"examine {item}"

    # Nothing left to interact with — move on
    if available_exits:
        untried = room_memory.get_untried_exits(state.location, available_exits)
        if untried:
            return random.choice(untried)
        return random.choice(available_exits)

    return "look"


# ── Strategy Dispatch Table ─────────────────────────────────────────────────

# Each executor takes (state, room_memory, available_exits) -> command string
STRATEGY_EXECUTORS = {
    ZorkStrategy.EXPLORE_NEW: execute_explore_new,
    ZorkStrategy.EXPLORE_KNOWN: execute_explore_known,
    ZorkStrategy.COLLECT_ITEMS: execute_collect_items,
    ZorkStrategy.USE_ITEM: execute_use_item,
    ZorkStrategy.DEPOSIT_TROPHY: execute_deposit_trophy,
    ZorkStrategy.FIGHT: execute_fight,
    ZorkStrategy.MANAGE_LIGHT: execute_manage_light,
    ZorkStrategy.INTERACT: execute_interact,
}


# ── Zork Pattern Policy ────────────────────────────────────────────────────

class ZorkPatternPolicy:
    """
    The core decision-maker for Zork. Uses the four pillars of learning to
    select strategies based on pattern-matched game situations.

    Each turn (no strategy persistence — like chess, each command is fresh):
    1. Quantize the game into a situation key
    2. Feed features to PatternEngine (composability + pattern discovery)
    3. Look up best strategy for this situation from learned bindings
    4. Execute the strategy to pick a concrete text command
    5. After episode ends, feed score delta back (feedback + approximability)
    """

    def __init__(self):
        # Core AI components (the four pillars)
        self.memory = DualMemory(max_patterns=5000, max_state=500)
        self.engine = PatternEngine(self.memory)
        self.feedback_loop = FeedbackLoop(
            learning_rate=0.1, discount_factor=0.99)
        # Higher initial exploration than RTS/Chess — Zork needs more.
        # Minimum 15% exploration to prevent lock-in on bad strategies.
        self.explorer = ExplorationStrategy(exploration_rate=0.4)
        self._decay_rate = 0.999  # Very slow decay — Zork needs lots of exploration
        self._min_exploration = 0.15  # Never go below 15%

        # Zork-specific
        self.encoder = ZorkStateEncoder()

        # Strategy bindings: situation_key -> {strategy_idx -> StrategyBinding}
        self.strategy_bindings: Dict[str, Dict[int, StrategyBinding]] = {}

        # Per-episode tracking
        self._episode_decisions: List[Tuple[str, int]] = []
        self._episode_score_by_strategy: Dict[int, float] = {}
        self.room_memory = RoomMemory()

    def select_strategy(self, state: ZorkState,
                        available_exits: List[str] = None
                        ) -> Tuple[int, str]:
        """
        Select a strategy using situation key lookup + exploration.

        No persistence needed — each Zork command is a fresh decision
        (like chess, unlike the RTS tick-based policy).

        Uses CONTEXT-AWARE FILTERING: strategies that cannot do anything
        useful in the current state are excluded from selection. This
        prevents wasting moves on "look" fallbacks.
        """
        sit_key = _situation_key_from_state(state)

        # Feed features to PatternEngine for discovery/composition
        tried_here = len(self.room_memory.room_exits_tried.get(
            state.location.lower(), set())) if state.location else 0
        features = self.encoder.encode(state, available_exits or [],
                                       tried_here)
        self.engine.process(features, domain="zork_situation")

        # ── Context-aware strategy filtering ──
        # Only consider strategies that can do something useful right now.
        # This is the key fix: prevents COLLECT_ITEMS from dominating when
        # there's nothing to collect, FIGHT when no enemies, etc.
        viable_strategies = self._get_viable_strategies(
            state, available_exits or [])

        # Look up best strategy for this situation FROM VIABLE SET
        # Deterministic tie-breaking: when confidences are equal,
        # always pick the lowest strategy index.
        best_strategy = None
        best_confidence = -1.0

        if sit_key in self.strategy_bindings:
            for strat_idx in sorted(self.strategy_bindings[sit_key].keys()):
                if strat_idx not in viable_strategies:
                    continue
                binding = self.strategy_bindings[sit_key][strat_idx]
                if (binding.confidence > best_confidence
                        or (binding.confidence == best_confidence
                            and best_strategy is not None
                            and strat_idx < best_strategy)):
                    best_confidence = binding.confidence
                    best_strategy = strat_idx

        # Exploration: sometimes try a random viable strategy
        if self.explorer.should_explore() or best_strategy is None:
            best_strategy = random.choice(list(viable_strategies))

        # Contextual override: if it's dark and we have a lamp, always
        # prioritize light management (safety > exploration)
        if state.is_dark and best_strategy != ZorkStrategy.MANAGE_LIGHT:
            inv_lower = [i.lower() for i in state.inventory]
            if any('lamp' in i or 'lantern' in i for i in inv_lower):
                # 70% chance to override with MANAGE_LIGHT
                if random.random() < 0.7:
                    best_strategy = ZorkStrategy.MANAGE_LIGHT

        # If enemies present, high chance to fight or flee
        if state.enemies and best_strategy not in (
                ZorkStrategy.FIGHT, ZorkStrategy.EXPLORE_NEW):
            if random.random() < 0.6:
                best_strategy = ZorkStrategy.FIGHT

        # Ensure binding exists
        if sit_key not in self.strategy_bindings:
            self.strategy_bindings[sit_key] = {}
        if best_strategy not in self.strategy_bindings[sit_key]:
            self.strategy_bindings[sit_key][best_strategy] = \
                StrategyBinding(best_strategy, sit_key)

        # Track decision
        self._episode_decisions.append((sit_key, best_strategy))
        self.strategy_bindings[sit_key][best_strategy].times_used += 1

        return best_strategy, sit_key

    def _get_viable_strategies(self, state: ZorkState,
                               available_exits: List[str]) -> Set[int]:
        """
        Return the set of strategies that can do something useful in the
        current game state. This prevents the agent from wasting moves.

        EXPLORE_NEW and EXPLORE_KNOWN are ALWAYS viable (can always move).
        Others depend on context: items visible, inventory, enemies, etc.
        """
        viable = {ZorkStrategy.EXPLORE_NEW, ZorkStrategy.EXPLORE_KNOWN}

        room_lower = state.location.lower() if state.location else ''

        # COLLECT_ITEMS: only if items visible OR we haven't tried take_all here
        items_tried = self.room_memory.room_items_tried.get(room_lower, set())
        if state.visible_items or 'take_all' not in items_tried:
            viable.add(ZorkStrategy.COLLECT_ITEMS)

        # USE_ITEM: only if we have inventory items
        if state.inventory:
            viable.add(ZorkStrategy.USE_ITEM)

        # DEPOSIT_TROPHY: only if we have treasures in inventory
        if state.inventory:
            from zork_ai.text_parser import identify_treasures
            if identify_treasures(state.inventory):
                viable.add(ZorkStrategy.DEPOSIT_TROPHY)

        # FIGHT: only if enemies present
        if state.enemies:
            viable.add(ZorkStrategy.FIGHT)

        # MANAGE_LIGHT: only if dark OR we have/see a lamp
        if state.is_dark:
            viable.add(ZorkStrategy.MANAGE_LIGHT)
        else:
            inv_lower = [i.lower() for i in state.inventory]
            if any('lamp' in i or 'lantern' in i for i in inv_lower):
                # Can turn off lamp to conserve battery
                pass  # not useful unless dark
            for item in state.visible_items:
                if 'lamp' in item.lower() or 'lantern' in item.lower():
                    viable.add(ZorkStrategy.MANAGE_LIGHT)

        # INTERACT: if there are objects/items to examine that we haven't tried
        objects_tried = self.room_memory.room_objects_tried.get(room_lower, set())
        desc_lower = state.description.lower()
        interactable_objects = [
            'mailbox', 'door', 'window', 'trapdoor', 'trap door',
            'grating', 'gate', 'rug', 'carpet', 'leaves',
            'table', 'desk', 'pedestal', 'altar', 'basket',
            'case', 'trophy case', 'chest', 'nest',
            'dam', 'button', 'switch', 'lever', 'mirror',
            'machine', 'lid', 'bolt', 'boat', 'tree',
        ]
        has_untried_objects = any(
            obj in desc_lower and obj not in objects_tried
            for obj in interactable_objects
        )
        has_untried_visible = any(
            item not in objects_tried for item in state.visible_items
        )
        if has_untried_objects or has_untried_visible or state.inventory:
            viable.add(ZorkStrategy.INTERACT)

        return viable

    def get_command(self, state: ZorkState,
                    available_exits: List[str] = None) -> str:
        """
        Full decision pipeline: situation -> strategy -> command.
        This is the main entry point called each turn.

        Priority order:
        1. Early-game navigation (deterministic path to house entry)
        2. Exploration unlocks (key chokepoint actions like open window)
        3. Strategy-based decisions (learned from training)
        """
        # ── Priority 1: Early-game navigation ──
        # In the first few moves, guide the agent deterministically from
        # West of House → North of House → Behind House → enter
        early_nav = _get_early_game_navigation(state, self.room_memory)
        if early_nav:
            # Still track as EXPLORE_NEW for credit assignment
            sit_key = _situation_key_from_state(state)
            self._episode_decisions.append((sit_key, ZorkStrategy.EXPLORE_NEW))
            if sit_key not in self.strategy_bindings:
                self.strategy_bindings[sit_key] = {}
            if ZorkStrategy.EXPLORE_NEW not in self.strategy_bindings[sit_key]:
                self.strategy_bindings[sit_key][ZorkStrategy.EXPLORE_NEW] = \
                    StrategyBinding(ZorkStrategy.EXPLORE_NEW, sit_key)
            self.strategy_bindings[sit_key][ZorkStrategy.EXPLORE_NEW].times_used += 1
            self._track_direction(early_nav, state)
            return early_nav

        # ── Priority 2: Exploration unlocks (chokepoints) ──
        # Fire from ALL strategies, not just EXPLORE_NEW.
        # These are one-time actions at key gates (open window, move rug, etc.)
        unlock = _get_exploration_unlock(state, self.room_memory)
        if unlock:
            sit_key = _situation_key_from_state(state)
            self._episode_decisions.append((sit_key, ZorkStrategy.INTERACT))
            if sit_key not in self.strategy_bindings:
                self.strategy_bindings[sit_key] = {}
            if ZorkStrategy.INTERACT not in self.strategy_bindings[sit_key]:
                self.strategy_bindings[sit_key][ZorkStrategy.INTERACT] = \
                    StrategyBinding(ZorkStrategy.INTERACT, sit_key)
            self.strategy_bindings[sit_key][ZorkStrategy.INTERACT].times_used += 1
            self._track_direction(unlock, state)
            return unlock

        # ── Priority 3: Strategy-based decisions (the learning system) ──
        strategy, sit_key = self.select_strategy(state, available_exits)

        # Execute the strategy
        executor = STRATEGY_EXECUTORS.get(
            strategy, STRATEGY_EXECUTORS[ZorkStrategy.INTERACT])
        command = executor(state, self.room_memory,
                           available_exits or [])

        self._track_direction(command, state)
        return command

    def _track_direction(self, command: str, state: ZorkState):
        """Track directional commands for room memory."""
        cmd_lower = command.lower().strip()
        if cmd_lower in DIRECTIONS or cmd_lower in DIR_ABBREVS:
            direction = DIR_ABBREVS.get(cmd_lower, cmd_lower)
            if state.location:
                self.room_memory.try_exit(state.location, direction)

    def begin_episode(self):
        """Call at the start of each training episode."""
        self._episode_decisions = []
        self._episode_score_by_strategy = {}
        self.room_memory = RoomMemory()
        self.encoder.reset()

    def record_score_delta(self, strategy: int, delta: int):
        """Track score changes attributed to each strategy during gameplay."""
        if delta > 0:
            self._episode_score_by_strategy[strategy] = \
                self._episode_score_by_strategy.get(strategy, 0) + delta

    def record_outcome(self, final_score: int, max_score: int = 350,
                       game_info: Dict = None):
        """
        After an episode ends, apply feedback to all situations and strategies
        used during this episode. This is the FEEDBACK pillar in action.

        Unlike RTS (binary win/loss), Zork uses continuous score-based feedback.
        A score of 200/350 is much better than 10/350.

        CREDIT ASSIGNMENT RULES:
        1. Strategies that DIRECTLY caused score increases always get wins
        2. In episodes with any score, the dominant strategy per-situation
           gets fractional credit proportional to final score
        3. In zero-score episodes, we use ROOMS VISITED as a proxy signal:
           exploration strategies get small credit if they found new rooms
        4. We NO LONGER penalize all strategies equally on zero-score episodes
           (that was making everything look equally bad)
        """
        game_info = game_info or {}

        score_ratio = min(final_score / max(max_score, 1), 1.0)
        rooms_visited = game_info.get('rooms_visited', 0)

        # Count usage per situation per strategy
        sit_strat_counts: Dict[str, Dict[int, int]] = {}
        for sit_key, strat in self._episode_decisions:
            if sit_key not in sit_strat_counts:
                sit_strat_counts[sit_key] = {}
            sit_strat_counts[sit_key][strat] = \
                sit_strat_counts[sit_key].get(strat, 0) + 1

        # ── RULE 1: Direct score credit (strongest signal) ──
        # Strategies that directly caused score increases ALWAYS get wins.
        # This is the most reliable feedback — it happened THIS turn.
        for strat, score_gained in self._episode_score_by_strategy.items():
            for sit_key, strat_counts in sit_strat_counts.items():
                if strat in strat_counts:
                    if sit_key not in self.strategy_bindings:
                        self.strategy_bindings[sit_key] = {}
                    if strat not in self.strategy_bindings[sit_key]:
                        self.strategy_bindings[sit_key][strat] = \
                            StrategyBinding(strat, sit_key)
                    binding = self.strategy_bindings[sit_key][strat]
                    # Strong credit: score gained / 25 (so +5 pts = 0.2 wins)
                    binding.wins += min(score_gained / 25.0, 2.0)

        # ── RULE 2: Episode-level credit for scoring episodes ──
        if final_score > 0:
            for sit_key, strat_counts in sit_strat_counts.items():
                dominant_strat = max(strat_counts, key=strat_counts.get)

                if sit_key not in self.strategy_bindings:
                    self.strategy_bindings[sit_key] = {}
                if dominant_strat not in self.strategy_bindings[sit_key]:
                    self.strategy_bindings[sit_key][dominant_strat] = \
                        StrategyBinding(dominant_strat, sit_key)

                binding = self.strategy_bindings[sit_key][dominant_strat]
                # Moderate credit proportional to score
                binding.wins += score_ratio * 0.5

        # ── RULE 3: Exploration proxy for zero-score episodes ──
        # When score is 0, use rooms_visited as a weak learning signal.
        # Exploration strategies that led to visiting many rooms get
        # small credit. Strategies that produced no movement get small penalty.
        if final_score == 0 and rooms_visited > 0:
            exploration_strats = {
                ZorkStrategy.EXPLORE_NEW, ZorkStrategy.EXPLORE_KNOWN}
            room_ratio = min(rooms_visited / 15.0, 1.0)  # 15 rooms = full credit

            for sit_key, strat_counts in sit_strat_counts.items():
                dominant_strat = max(strat_counts, key=strat_counts.get)

                if sit_key not in self.strategy_bindings:
                    self.strategy_bindings[sit_key] = {}
                if dominant_strat not in self.strategy_bindings[sit_key]:
                    self.strategy_bindings[sit_key][dominant_strat] = \
                        StrategyBinding(dominant_strat, sit_key)

                binding = self.strategy_bindings[sit_key][dominant_strat]

                if dominant_strat in exploration_strats:
                    # Weak credit for exploring (more rooms = more credit)
                    binding.wins += room_ratio * 0.2
                else:
                    # Very small penalty for non-exploration in zero-score
                    binding.losses += 0.1

        # Refine patterns in memory (APPROXIMABILITY pillar)
        # Score-based feedback value: map [0, max_score] to [0.2, 0.9]
        feedback_value = 0.2 + 0.7 * score_ratio
        all_pattern_ids = list(self.memory.patterns.patterns.keys())
        for pid in all_pattern_ids[-20:]:
            pattern = self.memory.patterns.patterns.get(pid)
            if pattern:
                pattern.refine(feedback=feedback_value)

        # Apply feedback through FeedbackLoop
        if all_pattern_ids:
            fv = score_ratio * 2 - 1  # Map [0,1] to [-1,1]
            self.feedback_loop.apply_feedback(
                FeedbackSignal(
                    signal_type=FeedbackType.EXTRINSIC,
                    value=fv,
                    target_pattern_ids=all_pattern_ids[-10:],
                    context={'score': final_score, 'ratio': score_ratio}
                ),
                self.memory.patterns.patterns
            )

        # Process outcome as composite pattern (COMPOSABILITY)
        outcome_features = [
            score_ratio,
            game_info.get('rooms_visited', 0) / 110.0,
            game_info.get('items_collected', 0) / 20.0,
            game_info.get('moves', 0) / 400.0,
        ]
        self.engine.process(outcome_features, domain="zork_outcome")

        # Try composing novel patterns (COMPOSABILITY + EXPLORATION)
        all_patterns = list(self.memory.patterns.patterns.values())
        if len(all_patterns) >= 2:
            combo = self.explorer.suggest_combination(all_patterns)
            if combo and len(combo) >= 2:
                composite = Pattern.create_composite(combo, "zork_situation")
                self.memory.patterns.store(composite)

        # Decay exploration rate (with minimum floor)
        self.explorer.decay_exploration(self._decay_rate)
        if self.explorer.exploration_rate < self._min_exploration:
            self.explorer.exploration_rate = self._min_exploration

        # Clear episode tracking
        self._episode_decisions = []
        self._episode_score_by_strategy = {}

    def save(self, path: str):
        """Save policy state to disk."""
        os.makedirs(path, exist_ok=True)

        # Save patterns
        patterns_data = {
            pid: p.to_dict()
            for pid, p in self.memory.patterns.patterns.items()
        }
        with open(os.path.join(path, 'patterns.json'), 'w') as f:
            json.dump(patterns_data, f, indent=2)

        # Save strategy bindings
        bindings_data = {}
        for sit_key, strat_map in self.strategy_bindings.items():
            bindings_data[sit_key] = {
                str(strat_idx): binding.to_dict()
                for strat_idx, binding in strat_map.items()
            }
        with open(os.path.join(path, 'strategy_bindings.json'), 'w') as f:
            json.dump(bindings_data, f, indent=2)

        # Save exploration state
        with open(os.path.join(path, 'exploration.json'), 'w') as f:
            json.dump({
                'exploration_rate': self.explorer.exploration_rate,
            }, f, indent=2)

    def load(self, path: str):
        """Load policy state from disk."""
        # Load patterns
        patterns_path = os.path.join(path, 'patterns.json')
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                patterns_data = json.load(f)
            for pid, pdata in patterns_data.items():
                pattern = Pattern.from_dict(pdata)
                self.memory.patterns.store(pattern)

        # Load strategy bindings
        bindings_path = os.path.join(path, 'strategy_bindings.json')
        if os.path.exists(bindings_path):
            with open(bindings_path, 'r') as f:
                bindings_data = json.load(f)
            for sit_key, strat_map in bindings_data.items():
                self.strategy_bindings[sit_key] = {}
                for strat_idx_str, bdata in strat_map.items():
                    strat_idx = int(strat_idx_str)
                    self.strategy_bindings[sit_key][strat_idx] = \
                        StrategyBinding.from_dict(bdata)

        # Load exploration state
        explore_path = os.path.join(path, 'exploration.json')
        if os.path.exists(explore_path):
            with open(explore_path, 'r') as f:
                explore_data = json.load(f)
            self.explorer.exploration_rate = explore_data.get(
                'exploration_rate', 0.4)

    def get_stats(self) -> Dict:
        """Get current learning statistics."""
        total_bindings = sum(
            len(smap) for smap in self.strategy_bindings.values()
        )
        strategy_counts = {s.name: 0 for s in ZorkStrategy}
        for sit_key, smap in self.strategy_bindings.items():
            for strat_idx, binding in smap.items():
                name = ZorkStrategy(strat_idx).name
                strategy_counts[name] += binding.times_used

        return {
            'patterns_discovered': len(self.memory.patterns.patterns),
            'strategy_bindings': total_bindings,
            'exploration_rate': self.explorer.exploration_rate,
            'strategy_usage': strategy_counts,
        }

    # ── Parallel training helpers ────────────────────────────────────────

    def snapshot_for_workers(self) -> dict:
        """Return minimal policy state for parallel workers."""
        bindings_snapshot = {}
        for sit_key, strat_map in self.strategy_bindings.items():
            bindings_snapshot[sit_key] = {
                str(strat_idx): binding.to_dict()
                for strat_idx, binding in strat_map.items()
            }
        return {
            'strategy_bindings': bindings_snapshot,
            'exploration_rate': self.explorer.exploration_rate,
        }

    def apply_episode_result(self, decisions: list, final_score: int,
                             game_info: dict,
                             discovered_patterns: list = None,
                             score_by_strategy: dict = None):
        """Apply a single episode's outcome to the shared policy state.

        Used by parallel training to merge worker results.
        """
        # Merge worker-discovered patterns into the main memory
        if discovered_patterns:
            for pdata in discovered_patterns:
                pattern = Pattern.from_dict(pdata)
                existing = self.memory.patterns.patterns.get(pattern.id)
                if existing:
                    if pattern.activation_count > existing.activation_count:
                        self.memory.patterns.store(pattern)
                else:
                    self.memory.patterns.store(pattern)

        # Count usage per (sit_key, strategy) from worker decisions
        usage_counts = {}
        for sit_key, strat in decisions:
            key = (sit_key, strat)
            usage_counts[key] = usage_counts.get(key, 0) + 1

        # Ensure all bindings exist and merge times_used
        for (sit_key, strategy), count in usage_counts.items():
            if sit_key not in self.strategy_bindings:
                self.strategy_bindings[sit_key] = {}
            if strategy not in self.strategy_bindings[sit_key]:
                self.strategy_bindings[sit_key][strategy] = \
                    StrategyBinding(strategy, sit_key)
            self.strategy_bindings[sit_key][strategy].times_used += count

        # Apply feedback using the episode decisions
        self._episode_decisions = decisions
        self._episode_score_by_strategy = score_by_strategy or {}
        self.record_outcome(final_score, game_info=game_info)
