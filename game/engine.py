"""
Game Engine - Core game loop that processes actions and advances the game.

Handles:
- Action execution (move, harvest, return, produce, attack)
- Durative action ticking (actions take multiple ticks)
- Combat resolution
- Unit production and building
- Resource management
- Win/loss detection
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from game.game_map import GameMap
from game.game_state import GameState
from game.units import (
    Unit, UnitType, ActionState, UNIT_STATS, PRODUCTION_TABLE, BUILD_TABLE,
)
from game.actions import (
    Action, ActionType, Direction, DIR_OFFSETS,
    decode_cell_action, get_action_space_dims, DIMS_PER_CELL,
)


class GameEngine:
    """
    The core game engine that processes actions and advances game state.
    Provides a Gym-like step interface for RL agent interaction.
    """

    def __init__(self, map_size: int = 8, max_ticks: int = 2000):
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.state: Optional[GameState] = None

    def reset(self, map_type: str = "standard") -> GameState:
        """Reset the game to initial state."""
        if map_type == "16x16":
            game_map = GameMap.create_16x16_map()
            self.map_size = 16
        else:
            game_map = GameMap.create_standard_map(self.map_size)

        self.state = GameState(game_map)
        self.state.max_ticks = self.max_ticks

        # Starting resources
        self.state.player_resources = {0: 5, 1: 5}

        return self.state

    def step(self, p0_actions: np.ndarray, p1_actions: np.ndarray) -> Tuple[GameState, Dict]:
        """
        Process one game tick with actions from both players.

        p0_actions, p1_actions: (H*W, DIMS_PER_CELL) or (H*W*DIMS_PER_CELL,) arrays
        Returns: (state, info_dict)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        h, w = self.state.game_map.height, self.state.game_map.width
        dims_per_cell = DIMS_PER_CELL

        # Reshape if flat
        if p0_actions.ndim == 1:
            p0_actions = p0_actions.reshape(h * w, dims_per_cell)
        if p1_actions.ndim == 1:
            p1_actions = p1_actions.reshape(h * w, dims_per_cell)

        # Decode actions for each player's units
        actions_p0 = self._decode_player_actions(p0_actions, player=0)
        actions_p1 = self._decode_player_actions(p1_actions, player=1)

        all_actions = actions_p0 + actions_p1

        # Assign new actions to idle units
        self._assign_actions(all_actions)

        # Tick all active units
        self._tick_all_units()

        # Remove dead units
        self._remove_dead_units()

        # Advance tick
        self.state.tick += 1

        # Check game over
        done, winner = self.state.check_game_over()
        if done:
            self.state.done = True
            self.state.winner = winner

        info = {
            'tick': self.state.tick,
            'done': self.state.done,
            'winner': self.state.winner,
            'p0_resources': self.state.player_resources[0],
            'p1_resources': self.state.player_resources[1],
            'p0_units': len(self.state.game_map.get_player_units(0)),
            'p1_units': len(self.state.game_map.get_player_units(1)),
        }

        return self.state, info

    def _decode_player_actions(self, action_grid: np.ndarray,
                               player: int) -> List[Action]:
        """Decode action grid into Action objects for a player's units."""
        h, w = self.state.game_map.height, self.state.game_map.width
        actions = []

        for y in range(h):
            for x in range(w):
                cell_idx = y * w + x
                unit = self.state.game_map.get_unit_at(x, y)
                if unit is None or unit.player != player:
                    continue
                if not unit.is_idle:
                    continue

                cell_action = action_grid[cell_idx].astype(int).tolist()
                action = decode_cell_action(cell_action, unit, x, y)
                if action:
                    actions.append(action)

        return actions

    def _assign_actions(self, actions: List[Action]):
        """Assign actions to idle units, validating each action."""
        for action in actions:
            unit = self.state.game_map.units.get(action.unit_id)
            if unit is None or not unit.is_alive or not unit.is_idle:
                continue

            if action.action_type == ActionType.NOOP:
                continue

            if action.action_type == ActionType.MOVE:
                self._start_move(unit, action)
            elif action.action_type == ActionType.HARVEST:
                self._start_harvest(unit, action)
            elif action.action_type == ActionType.RETURN:
                self._start_return(unit, action)
            elif action.action_type == ActionType.PRODUCE:
                self._start_produce(unit, action)
            elif action.action_type == ActionType.ATTACK:
                self._start_attack(unit, action)

    def _start_move(self, unit: Unit, action: Action):
        """Start a move action."""
        if not unit.can_move or action.direction is None:
            return
        dx, dy = DIR_OFFSETS[action.direction]
        nx, ny = unit.x + dx, unit.y + dy
        if not self.state.game_map.is_empty(nx, ny):
            return
        unit.action_state = ActionState.MOVING
        unit.action_target = (nx, ny)
        unit.action_ticks_remaining = unit.stats.move_time

    def _start_harvest(self, unit: Unit, action: Action):
        """Start a harvest action."""
        if not unit.can_harvest or action.direction is None:
            return
        dx, dy = DIR_OFFSETS[action.direction]
        nx, ny = unit.x + dx, unit.y + dy
        target = self.state.game_map.get_unit_at(nx, ny)
        if not target or target.unit_type != UnitType.RESOURCE:
            return
        unit.action_state = ActionState.HARVESTING
        unit.action_target = (nx, ny)
        unit.action_ticks_remaining = unit.stats.harvest_time

    def _start_return(self, unit: Unit, action: Action):
        """Start a return-resource action."""
        if not unit.can_return or action.direction is None:
            return
        dx, dy = DIR_OFFSETS[action.direction]
        nx, ny = unit.x + dx, unit.y + dy
        target = self.state.game_map.get_unit_at(nx, ny)
        if not (target and target.unit_type == UnitType.BASE
                and target.player == unit.player):
            return
        unit.action_state = ActionState.RETURNING
        unit.action_target = (nx, ny)
        unit.action_ticks_remaining = unit.stats.return_time

    def _start_produce(self, unit: Unit, action: Action):
        """Start a produce/build action."""
        if action.direction is None or action.produce_type is None:
            return

        produce_type = action.produce_type
        cost = UNIT_STATS[produce_type].cost

        # Validate
        if unit.unit_type in PRODUCTION_TABLE:
            if produce_type not in PRODUCTION_TABLE[unit.unit_type]:
                return
            if not unit.can_produce:
                return
        elif unit.stats.can_build:
            if produce_type not in BUILD_TABLE:
                return
            if not unit.can_build:
                return
        else:
            return

        if self.state.player_resources[unit.player] < cost:
            return

        dx, dy = DIR_OFFSETS[action.direction]
        nx, ny = unit.x + dx, unit.y + dy
        if not self.state.game_map.is_empty(nx, ny):
            return

        # Deduct cost
        self.state.player_resources[unit.player] -= cost

        unit.action_state = ActionState.PRODUCING
        unit.action_target = (nx, ny)
        unit.producing_type = produce_type
        unit.action_ticks_remaining = UNIT_STATS[produce_type].build_time

    def _start_attack(self, unit: Unit, action: Action):
        """Start an attack action."""
        if not unit.can_attack:
            return
        tx, ty = action.target_x, action.target_y
        if tx is None or ty is None:
            return
        if not unit.in_attack_range(tx, ty):
            return
        target = self.state.game_map.get_unit_at(tx, ty)
        if not target or target.player == unit.player:
            return

        unit.action_state = ActionState.ATTACKING
        unit.action_target = (tx, ty)
        unit.action_ticks_remaining = unit.stats.attack_time

    def _tick_all_units(self):
        """Advance all units by one tick."""
        for unit in list(self.state.game_map.units.values()):
            if not unit.is_alive or unit.is_idle:
                continue

            unit.action_ticks_remaining -= 1

            if unit.action_ticks_remaining <= 0:
                self._complete_action(unit)

    def _complete_action(self, unit: Unit):
        """Complete a unit's current action."""
        if unit.action_state == ActionState.MOVING:
            tx, ty = unit.action_target
            if self.state.game_map.is_empty(tx, ty):
                self.state.game_map.move_unit(unit.unit_id, tx, ty)

        elif unit.action_state == ActionState.HARVESTING:
            tx, ty = unit.action_target
            resource = self.state.game_map.get_unit_at(tx, ty)
            if resource and resource.unit_type == UnitType.RESOURCE:
                unit.resources_carried = 1
                resource.hp -= 1
                if resource.hp <= 0:
                    self.state.game_map.remove_unit(resource.unit_id)

        elif unit.action_state == ActionState.RETURNING:
            tx, ty = unit.action_target
            base = self.state.game_map.get_unit_at(tx, ty)
            if (base and base.unit_type == UnitType.BASE
                    and base.player == unit.player):
                self.state.player_resources[unit.player] += unit.resources_carried
                unit.resources_carried = 0

        elif unit.action_state == ActionState.PRODUCING:
            tx, ty = unit.action_target
            if self.state.game_map.is_empty(tx, ty) and unit.producing_type is not None:
                self.state.game_map.add_unit(
                    unit.producing_type, unit.player, tx, ty
                )

        elif unit.action_state == ActionState.ATTACKING:
            tx, ty = unit.action_target
            target = self.state.game_map.get_unit_at(tx, ty)
            if target and target.is_alive and target.player != unit.player:
                target.take_damage(unit.stats.damage)

        # Reset to idle
        unit.action_state = ActionState.IDLE
        unit.action_target = None
        unit.action_ticks_remaining = 0
        unit.producing_type = None

    def _remove_dead_units(self):
        """Remove all dead units from the map."""
        dead_ids = [uid for uid, u in self.state.game_map.units.items()
                    if not u.is_alive and u.unit_type != UnitType.RESOURCE]
        # Also remove depleted resources
        dead_ids += [uid for uid, u in self.state.game_map.units.items()
                     if u.unit_type == UnitType.RESOURCE and not u.is_alive]
        for uid in dead_ids:
            self.state.game_map.remove_unit(uid)


class VecGameEnv:
    """
    Vectorized game environment for training RL agents.
    Runs multiple games in parallel.

    Interface matches common RL conventions:
    - reset() -> observations
    - step(actions) -> observations, rewards, dones, infos
    - get_action_mask() -> masks
    """

    def __init__(self, num_envs: int = 4, map_size: int = 8,
                 max_ticks: int = 2000, opponent_ai=None):
        self.num_envs = num_envs
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.engines = [GameEngine(map_size, max_ticks) for _ in range(num_envs)]
        self.opponent_ai = opponent_ai

        # Dimensions
        self.height = map_size
        self.width = map_size
        self.action_dims = get_action_space_dims()
        self.dims_per_cell = DIMS_PER_CELL

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns stacked observations."""
        obs_list = []
        for engine in self.engines:
            state = engine.reset()
            obs_list.append(state.get_observation(player=0))
        return np.stack(obs_list)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, List[Dict]]:
        """
        Step all environments.

        actions: (num_envs, H*W*DIMS_PER_CELL) or (num_envs, H*W, DIMS_PER_CELL)
        Returns: (obs, rewards, dones, infos)
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for i, engine in enumerate(self.engines):
            state = engine.state

            # Get pre-step info for reward
            prev_info = state.get_state_info(player=0)

            # Player 0 actions (from agent)
            p0_action = actions[i]

            # Player 1 actions (from opponent AI)
            if self.opponent_ai is not None:
                p1_action = self.opponent_ai.get_action(state, player=1)
            else:
                # Default: all NOOPs
                p1_action = np.zeros(
                    self.height * self.width * self.dims_per_cell,
                    dtype=np.int32
                )

            state, info = engine.step(p0_action, p1_action)

            # Compute reward
            reward = state.compute_reward(player=0, prev_state_info=prev_info)

            obs = state.get_observation(player=0)

            # Auto-reset on done
            if state.done:
                state = engine.reset()
                obs = state.get_observation(player=0)

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(info['done'])
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.bool_),
            info_list,
        )

    def get_action_mask(self) -> np.ndarray:
        """Get action masks for all environments. (num_envs, H*W, sum(action_dims))"""
        masks = []
        for engine in self.engines:
            mask = engine.state.get_action_mask(player=0)
            masks.append(mask)
        return np.stack(masks)

    def close(self):
        """Cleanup."""
        pass
