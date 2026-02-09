"""
Text Parser — Utilities for parsing Zork I text output.

Pure utility functions with no dependencies on core/. Handles extraction
of exits, items, enemies, score, and room identification from dfrotz output.

Zork I has ~110 rooms, ~40 takeable items, and a handful of enemies (troll,
thief, spirits). This module uses hardcoded vocabulary for reliable parsing.
"""

import re
from typing import List, Optional, Tuple


# ── Direction / Exit Parsing ────────────────────────────────────────────────

# All canonical directions Zork understands
DIRECTIONS = [
    'north', 'south', 'east', 'west',
    'northeast', 'northwest', 'southeast', 'southwest',
    'up', 'down',
    'in', 'out',
    'enter', 'exit',
    'climb',
]

# Abbreviations the game accepts
DIR_ABBREVS = {
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest',
    'u': 'up', 'd': 'down',
}

# Patterns that indicate available exits in room descriptions
_EXIT_PATTERNS = [
    # "To the north is ..."
    re.compile(r'\bto the (\w+)\b', re.I),
    # "passage leads north"
    re.compile(r'\bpassage\w* leads? (\w+)\b', re.I),
    # "A path leads east"
    re.compile(r'\bpath\w* leads? (\w+)\b', re.I),
    # "north of ..."
    re.compile(r'\b(north|south|east|west|up|down)\s+of\b', re.I),
    # Direct directions embedded in text
    re.compile(
        r'\b(north|south|east|west|northeast|northwest|'
        r'southeast|southwest|up|down|enter|in)\b', re.I
    ),
]


def parse_exits(description: str) -> List[str]:
    """
    Extract available movement directions from a room description.

    Returns a deduplicated list of canonical direction strings.
    """
    description_lower = description.lower()
    exits = set()

    for pat in _EXIT_PATTERNS:
        for match in pat.finditer(description_lower):
            word = match.group(1) if match.lastindex else match.group(0)
            word = word.strip()
            if word in DIRECTIONS:
                exits.add(word)
            elif word in DIR_ABBREVS:
                exits.add(DIR_ABBREVS[word])

    return sorted(exits)


# ── Item Parsing ────────────────────────────────────────────────────────────

# Known takeable items in Zork I
KNOWN_ITEMS = [
    'sword', 'lamp', 'lantern', 'leaflet', 'egg', 'jeweled egg',
    'clockwork canary', 'canary', 'gold coffin', 'coffin',
    'sceptre', 'trident', 'crystal trident',
    'knife', 'rusty knife', 'stiletto',
    'torch', 'ivory torch', 'scarab', 'jewel-encrusted scarab',
    'pot of gold', 'pot', 'coins', 'bag of coins',
    'platinum bar', 'diamond', 'huge diamond',
    'jade', 'jade figurine', 'figurine',
    'sapphire', 'sapphire bracelet', 'bracelet',
    'coal', 'trunk of jewels', 'trunk', 'jewels',
    'painting', 'beautiful painting',
    'chalice', 'silver chalice',
    'shovel', 'nasty knife',
    'rope', 'coil of rope',
    'bell', 'brass bell', 'book', 'prayer book',
    'candles', 'pair of candles', 'matchbook', 'match',
    'garlic', 'clove of garlic',
    'skull', 'crystal skull',
    'buoy', 'red buoy',
    'timber', 'pile of timber', 'lumber',
    'screwdriver', 'wrench',
    'key', 'skeleton key', 'golden key',
    'map', 'guidebook',
    'tube', 'viscous material',
    'inflatable boat', 'boat', 'pump', 'hand pump', 'air pump',
    'sack', 'brown sack', 'lunch', 'bottle', 'water',
    'gunk', 'black book',
    'bauble',
]

# Phrases that indicate visible/takeable items
_ITEM_INDICATOR_PATTERNS = [
    re.compile(r'there is (?:a |an |the )?(.+?) here', re.I),
    re.compile(r'(?:a |an |the )(.+?) is (?:sitting |lying |resting )?here', re.I),
    re.compile(r'on the (?:ground|floor|table) is (?:a |an |the )?(.+?)\.', re.I),
    re.compile(r'you can see (?:a |an |the )?(.+?) here', re.I),
]


def parse_visible_items(description: str) -> List[str]:
    """
    Find takeable items mentioned in a room description.

    Matches against the known item list and also uses indicator patterns.
    """
    desc_lower = description.lower()
    found = set()

    # Check for known items mentioned in text
    for item in KNOWN_ITEMS:
        if item in desc_lower:
            found.add(item)

    # Check indicator patterns
    for pat in _ITEM_INDICATOR_PATTERNS:
        for match in pat.finditer(desc_lower):
            item_text = match.group(1).strip().rstrip('.')
            # Check if the matched text contains a known item
            for item in KNOWN_ITEMS:
                if item in item_text:
                    found.add(item)
                    break
            else:
                # Add it as-is even if not in the known list
                if len(item_text) < 40:
                    found.add(item_text)

    return sorted(found)


def parse_room_objects(description: str) -> List[str]:
    """
    Find interactable objects in a room (doors, containers, etc.).

    These are objects you can examine, open, or interact with but not
    necessarily take.
    """
    desc_lower = description.lower()
    objects = set()

    interactables = [
        'mailbox', 'door', 'window', 'trapdoor', 'trap door',
        'grating', 'gate', 'rug', 'carpet',
        'table', 'desk', 'pedestal', 'altar',
        'case', 'trophy case', 'basket', 'chest',
        'dam', 'button', 'switch', 'lever',
        'mirror', 'machine', 'lid', 'bolt',
        'pile of leaves', 'leaves',
        'nest', "bird's nest", 'tree',
        'rainbow', 'railing',
        'boat', 'river', 'stream', 'lake',
    ]

    for obj in interactables:
        if obj in desc_lower:
            objects.add(obj)

    return sorted(objects)


# ── Enemy Detection ─────────────────────────────────────────────────────────

KNOWN_ENEMIES = {
    'troll': ['troll', 'nasty-looking troll', 'a troll'],
    'thief': ['thief', 'lean and hungry gentleman', 'a seedy-looking individual'],
    'cyclops': ['cyclops', 'a cyclops'],
    'spirits': ['spirits', 'evil spirits', 'the spirits'],
    'bat': ['vampire bat', 'a vampire bat'],
}


def parse_enemies(description: str) -> List[str]:
    """
    Detect known enemies in a room description.

    Returns list of enemy type names (e.g. ['troll', 'thief']).
    """
    desc_lower = description.lower()
    enemies = []

    for enemy_type, keywords in KNOWN_ENEMIES.items():
        for keyword in keywords:
            if keyword in desc_lower:
                enemies.append(enemy_type)
                break

    return enemies


# ── Treasure Identification ─────────────────────────────────────────────────

# Items that can be placed in the trophy case for points
TREASURES = [
    'egg', 'jeweled egg',
    'clockwork canary', 'canary',
    'painting', 'beautiful painting',
    'platinum bar',
    'ivory torch', 'torch',
    'gold coffin', 'coffin',
    'sceptre',
    'trident', 'crystal trident',
    'pot of gold', 'pot',
    'coins', 'bag of coins',
    'diamond', 'huge diamond',
    'jade figurine', 'figurine', 'jade',
    'sapphire bracelet', 'bracelet', 'sapphire',
    'trunk of jewels', 'trunk', 'jewels',
    'chalice', 'silver chalice',
    'scarab', 'jewel-encrusted scarab',
    'coal',
    'skull', 'crystal skull',
    'bauble',
]

# Canonical treasure names (deduplicated by shortest unique name)
TREASURE_CANONICAL = {
    'egg': 'egg',
    'jeweled egg': 'egg',
    'clockwork canary': 'canary',
    'canary': 'canary',
    'painting': 'painting',
    'beautiful painting': 'painting',
    'platinum bar': 'bar',
    'ivory torch': 'torch',
    'torch': 'torch',
    'gold coffin': 'coffin',
    'coffin': 'coffin',
    'sceptre': 'sceptre',
    'trident': 'trident',
    'crystal trident': 'trident',
    'pot of gold': 'pot',
    'pot': 'pot',
    'coins': 'coins',
    'bag of coins': 'coins',
    'diamond': 'diamond',
    'huge diamond': 'diamond',
    'jade figurine': 'figurine',
    'figurine': 'figurine',
    'jade': 'figurine',
    'sapphire bracelet': 'bracelet',
    'bracelet': 'bracelet',
    'sapphire': 'bracelet',
    'trunk of jewels': 'trunk',
    'trunk': 'trunk',
    'jewels': 'trunk',
    'chalice': 'chalice',
    'silver chalice': 'chalice',
    'scarab': 'scarab',
    'jewel-encrusted scarab': 'scarab',
    'coal': 'coal',
    'skull': 'skull',
    'crystal skull': 'skull',
    'bauble': 'bauble',
}


def identify_treasures(inventory_items: List[str]) -> List[str]:
    """
    From an inventory list, return items that are treasures
    worth depositing in the trophy case.
    """
    found = set()
    for item in inventory_items:
        item_lower = item.lower().strip()
        for treasure in TREASURES:
            if treasure in item_lower:
                found.add(TREASURE_CANONICAL.get(treasure, treasure))
                break
    return sorted(found)


# ── Game State Detection ────────────────────────────────────────────────────

_DEATH_PATTERNS = [
    re.compile(r'\*+\s*you have died\s*\*+', re.I),
    re.compile(r'you have been (?:killed|slain|eaten)', re.I),
    re.compile(r'you are dead', re.I),
    re.compile(r'your score is .+ in \d+ moves?\.\nthis gives you the rank',
               re.I),
    re.compile(r'you have been killed', re.I),
    re.compile(r'it appears that .+ was too much for you', re.I),
    re.compile(r'the (?:troll|thief|cyclops) .+ (?:kills|stabs|strikes) you',
               re.I),
    re.compile(r'you are devoured', re.I),
    re.compile(r'you stumble .+ and .+ (?:fall|plunge)', re.I),
    re.compile(r'you have starved', re.I),
]


def detect_death(response: str) -> bool:
    """Check if the response indicates the player has died."""
    for pat in _DEATH_PATTERNS:
        if pat.search(response):
            return True

    # Also check for "****  You have died  ****" style
    if '****' in response and 'died' in response.lower():
        return True

    return False


_DARKNESS_PATTERNS = [
    re.compile(r'it is pitch (?:black|dark)', re.I),
    re.compile(r'you are likely to be eaten by a grue', re.I),
    re.compile(r'it is too dark to see', re.I),
    re.compile(r'(?:total |pitch )?darkness', re.I),
]


def detect_darkness(response: str) -> bool:
    """Check if the response indicates the player is in darkness."""
    for pat in _DARKNESS_PATTERNS:
        if pat.search(response):
            return True
    return False


_SCORE_PATTERN = re.compile(
    r'your score is (\d+) \(total of (\d+) points?\)', re.I
)
_SCORE_PATTERN_ALT = re.compile(
    r'score:\s*(\d+)', re.I
)


def extract_score(response: str) -> Optional[int]:
    """
    Extract the current score from a response or status line.

    Returns None if no score found.
    """
    match = _SCORE_PATTERN.search(response)
    if match:
        return int(match.group(1))

    match = _SCORE_PATTERN_ALT.search(response)
    if match:
        return int(match.group(1))

    return None


def extract_score_and_moves(response: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract score and move count from dfrotz status line.

    dfrotz status lines look like:
      "Room Name                Score: 10       Moves: 42"
    """
    score = None
    moves = None

    score_match = re.search(r'Score:\s*(\d+)', response)
    if score_match:
        score = int(score_match.group(1))

    moves_match = re.search(r'Moves:\s*(\d+)', response)
    if moves_match:
        moves = int(moves_match.group(1))

    return score, moves


# ── Room Identification ─────────────────────────────────────────────────────

# Known rooms in Zork I, categorized for situation key generation
ROOM_CATEGORIES = {
    'house': [
        'west of house', 'north of house', 'south of house',
        'east of house', 'behind house',
        'kitchen', 'living room', 'attic',
    ],
    'forest': [
        'forest', 'forest path', 'clearing',
        'canyon view', 'rocky ledge', 'up a tree',
    ],
    'cave_entrance': [
        'cellar', 'east of chasm', 'gallery',
        'studio', 'troll room',
    ],
    'underground': [
        'maze', 'round room', 'narrow passage',
        'loud room', 'damp cave', 'white cliffs',
        'engravings cave', 'dome room', 'torch room',
        'north-south passage', 'chasm',
        'egypt room', 'egyptian room',
        'land of the dead', 'entrance to hades',
        'altar', 'temple',
        'mine', 'coal mine', 'shaft room',
        'machine room', 'drafty room',
        'smelly room', 'gas room',
        'slide room', 'cellar',
    ],
    'river': [
        'frigid river', 'dam', 'dam base', 'dam lobby',
        'reservoir', 'reservoir south', 'reservoir north',
        'stream', 'stream view', 'sandy beach', 'shore',
        'aragain falls', 'end of rainbow',
    ],
    'maze': [
        'maze', 'twisty passage', 'twisting passage',
        'dead end', 'grail diary',
        'grating room', 'cyclops room',
    ],
    'treasure_room': [
        'treasure room', 'strange passage',
        'mirror room', 'small cave',
    ],
}

# Flatten for lookup
_ROOM_TO_CATEGORY = {}
for category, rooms in ROOM_CATEGORIES.items():
    for room in rooms:
        _ROOM_TO_CATEGORY[room] = category


def categorize_room(room_name: str) -> str:
    """
    Categorize a room name into one of the major areas.

    Returns one of: house, forest, cave_entrance, underground,
    river, maze, treasure_room, or 'other'.
    """
    if not room_name:
        return 'other'

    room_lower = room_name.lower().strip()

    # Direct match
    if room_lower in _ROOM_TO_CATEGORY:
        return _ROOM_TO_CATEGORY[room_lower]

    # Substring match
    for room_key, category in _ROOM_TO_CATEGORY.items():
        if room_key in room_lower:
            return category

    # Keyword-based fallback
    if 'maze' in room_lower or 'twisty' in room_lower:
        return 'maze'
    if 'river' in room_lower or 'dam' in room_lower or 'beach' in room_lower:
        return 'river'
    if 'forest' in room_lower or 'clearing' in room_lower:
        return 'forest'
    if 'house' in room_lower or 'kitchen' in room_lower:
        return 'house'

    return 'other'


# ── Inventory Parsing ───────────────────────────────────────────────────────

def parse_inventory(response: str) -> List[str]:
    """
    Parse the response to an 'inventory' command.

    Typical format:
        You are carrying:
          A sword
          A brass lantern
          A leaflet
    """
    items = []
    lines = response.strip().split('\n')

    in_inventory = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith('you are carrying') or \
           stripped.lower().startswith('you have:'):
            in_inventory = True
            continue

        if in_inventory:
            if not stripped:
                # End of inventory list
                break
            # Remove article prefix
            item = re.sub(r'^(?:a |an |the |some )', '', stripped, flags=re.I)
            item = item.strip().rstrip('.')
            if item:
                items.append(item)

    return items


# ── Room Name Extraction ────────────────────────────────────────────────────

def extract_room_name(response: str) -> Optional[str]:
    """
    Extract the room name from dfrotz output.

    dfrotz with -p flag shows status lines like:
      "Room Name                Score: 10       Moves: 42"

    The room name is everything before "Score:" on such lines.
    """
    for line in response.split('\n'):
        # Look for the status line pattern
        match = re.match(r'^(.+?)\s+Score:\s*\d+\s+Moves:\s*\d+', line.strip())
        if match:
            room_name = match.group(1).strip()
            if room_name:
                return room_name

    return None


# ── Response Classification ─────────────────────────────────────────────────

def is_error_response(response: str) -> bool:
    """Check if the response indicates the command wasn't understood."""
    resp_lower = response.lower().strip()

    error_phrases = [
        "i don't understand",
        "i don't know the word",
        "that sentence isn't one i recognize",
        "you can't see any",
        "you can't do that",
        "there is nothing here",
        "you aren't carrying",
        "you don't have",
        "it's too dark to see",
        "that's not a verb",
    ]

    for phrase in error_phrases:
        if phrase in resp_lower:
            return True

    return False


def is_blocked_movement(response: str) -> bool:
    """Check if movement was blocked."""
    resp_lower = response.lower()

    blocked_phrases = [
        "you can't go that way",
        "there is no way to go",
        "the door is closed",
        "the door is locked",
        "the gate is closed",
        "the grating is closed",
        "you can't enter",
        "the way is blocked",
    ]

    for phrase in blocked_phrases:
        if phrase in resp_lower:
            return True

    return False
