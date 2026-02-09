"""
Game Renderer - ASCII visualization of game state.

Renders the game map as text for debugging and monitoring training.
"""

from typing import Optional
from game.game_state import GameState
from game.units import UnitType, ActionState


# Unit type symbols
UNIT_SYMBOLS = {
    UnitType.RESOURCE: '$',
    UnitType.BASE: 'B',
    UnitType.BARRACKS: 'K',
    UnitType.WORKER: 'w',
    UnitType.LIGHT: 'l',
    UnitType.HEAVY: 'h',
    UnitType.RANGED: 'r',
}


class GameRenderer:
    """ASCII renderer for game state visualization."""

    @staticmethod
    def render(state: GameState, show_info: bool = True) -> str:
        """Render game state as ASCII string."""
        gm = state.game_map
        h, w = gm.height, gm.width
        lines = []

        if show_info:
            lines.append(f"Tick: {state.tick}/{state.max_ticks}  "
                         f"P0 res: {state.player_resources[0]}  "
                         f"P1 res: {state.player_resources[1]}")
            p0_units = gm.get_player_units(0)
            p1_units = gm.get_player_units(1)
            lines.append(f"P0 units: {len(p0_units)}  P1 units: {len(p1_units)}")
            lines.append("")

        # Top border
        lines.append("  " + "".join(f"{x % 10}" for x in range(w)))
        lines.append("  " + "-" * w)

        for y in range(h):
            row = f"{y % 10}|"
            for x in range(w):
                unit = gm.get_unit_at(x, y)
                if unit is None:
                    row += "."
                else:
                    sym = UNIT_SYMBOLS.get(unit.unit_type, "?")
                    # Uppercase for player 0, lowercase for player 1
                    if unit.player == 0:
                        sym = sym.upper()
                    elif unit.player == 1:
                        sym = sym.lower()
                    # Resources are always $
                    if unit.unit_type == UnitType.RESOURCE:
                        sym = "$"
                    row += sym
            row += f"|{y % 10}"
            lines.append(row)

        lines.append("  " + "-" * w)
        lines.append("  " + "".join(f"{x % 10}" for x in range(w)))

        if show_info:
            lines.append("")
            lines.append("Legend: B=Base K=Barracks W=Worker L=Light H=Heavy R=Ranged $=Resource")
            lines.append("        UPPER=P0  lower=P1")

            if state.done:
                if state.winner == 0:
                    lines.append("\n*** PLAYER 0 WINS! ***")
                elif state.winner == 1:
                    lines.append("\n*** PLAYER 1 WINS! ***")
                else:
                    lines.append("\n*** DRAW ***")

        return "\n".join(lines)

    @staticmethod
    def render_compact(state: GameState) -> str:
        """Compact single-line rendering for logging."""
        gm = state.game_map
        p0 = gm.get_player_units(0)
        p1 = gm.get_player_units(1)
        return (f"T{state.tick:04d} "
                f"P0[u={len(p0)} r={state.player_resources[0]}] "
                f"P1[u={len(p1)} r={state.player_resources[1]}]")

    @staticmethod
    def render_unit_details(state: GameState, player: int) -> str:
        """Render detailed unit info for a player."""
        lines = [f"Player {player} units:"]
        for unit in state.game_map.get_player_units(player):
            status = unit.action_state.name
            extra = ""
            if unit.resources_carried > 0:
                extra = f" [carrying {unit.resources_carried}]"
            if unit.action_ticks_remaining > 0:
                extra += f" [{unit.action_ticks_remaining} ticks left]"
            lines.append(
                f"  {UNIT_SYMBOLS.get(unit.unit_type, '?')} "
                f"({unit.x},{unit.y}) "
                f"HP={unit.hp}/{unit.stats.hp} "
                f"{status}{extra}"
            )
        return "\n".join(lines)
