"""
Tests for the MicroRTS-style game environment.

Tests cover:
- Unit creation and stats
- Map operations
- Game state encoding
- Action masking
- Game engine mechanics
- AI opponents
- Full game simulation
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from game.units import Unit, UnitType, UnitStats, UNIT_STATS, ActionState
from game.game_map import GameMap, Terrain
from game.game_state import GameState, NUM_FEATURE_PLANES
from game.actions import (
    ActionType, Direction, DIR_OFFSETS, Action,
    get_action_space_dims, get_valid_actions_mask, DIMS_PER_CELL,
    decode_cell_action, MAX_ATTACK_RANGE,
)
from game.engine import GameEngine, VecGameEnv
from game.ai_opponents import RandomAI, RushAI, EconomyAI, DefensiveAI
from game.renderer import GameRenderer


class TestUnits:
    def test_unit_types_exist(self):
        for ut in UnitType:
            assert ut in UNIT_STATS

    def test_unit_creation(self):
        u = Unit(unit_id=0, unit_type=UnitType.WORKER, player=0, x=3, y=4)
        assert u.hp == 1  # Worker has 1 HP
        assert u.x == 3
        assert u.y == 4
        assert u.player == 0
        assert u.is_idle
        assert u.is_alive

    def test_unit_stats(self):
        worker_stats = UNIT_STATS[UnitType.WORKER]
        assert worker_stats.hp == 1
        assert worker_stats.cost == 1
        assert worker_stats.can_build is True
        assert worker_stats.damage == 1

        base_stats = UNIT_STATS[UnitType.BASE]
        assert base_stats.hp == 10
        assert base_stats.is_structure is True
        assert base_stats.can_produce is True
        assert base_stats.move_time == 0

    def test_unit_damage(self):
        u = Unit(unit_id=0, unit_type=UnitType.LIGHT, player=0, x=0, y=0)
        assert u.hp == 4
        u.take_damage(2)
        assert u.hp == 2
        assert u.is_alive
        u.take_damage(3)
        assert u.hp == 0
        assert not u.is_alive

    def test_unit_distance(self):
        u = Unit(unit_id=0, unit_type=UnitType.WORKER, player=0, x=2, y=3)
        assert u.distance_to(2, 3) == 0
        assert u.distance_to(3, 3) == 1
        assert u.distance_to(5, 6) == 6

    def test_attack_range(self):
        ranged = Unit(unit_id=0, unit_type=UnitType.RANGED, player=0, x=5, y=5)
        assert ranged.in_attack_range(6, 5)  # 1 cell
        assert ranged.in_attack_range(5, 8)  # 3 cells
        assert not ranged.in_attack_range(5, 9)  # 4 cells, out of range


class TestGameMap:
    def test_create_map(self):
        gm = GameMap(8, 8)
        assert gm.width == 8
        assert gm.height == 8

    def test_add_and_get_unit(self):
        gm = GameMap(8, 8)
        unit = gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        assert unit is not None
        assert unit.unit_type == UnitType.WORKER
        retrieved = gm.get_unit_at(3, 3)
        assert retrieved.unit_id == unit.unit_id

    def test_cannot_add_to_occupied(self):
        gm = GameMap(8, 8)
        gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        result = gm.add_unit(UnitType.WORKER, player=1, x=3, y=3)
        assert result is None

    def test_move_unit(self):
        gm = GameMap(8, 8)
        unit = gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        success = gm.move_unit(unit.unit_id, 4, 3)
        assert success
        assert unit.x == 4
        assert gm.get_unit_at(3, 3) is None
        assert gm.get_unit_at(4, 3) is not None

    def test_remove_unit(self):
        gm = GameMap(8, 8)
        unit = gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        gm.remove_unit(unit.unit_id)
        assert gm.get_unit_at(3, 3) is None
        assert unit.unit_id not in gm.units

    def test_standard_map(self):
        gm = GameMap.create_standard_map(8)
        p0_units = gm.get_player_units(0)
        p1_units = gm.get_player_units(1)
        assert len(p0_units) >= 2  # base + worker
        assert len(p1_units) >= 2
        resources = gm.get_resources()
        assert len(resources) > 0

    def test_bounds_checking(self):
        gm = GameMap(8, 8)
        assert gm.in_bounds(0, 0)
        assert gm.in_bounds(7, 7)
        assert not gm.in_bounds(-1, 0)
        assert not gm.in_bounds(8, 0)

    def test_adjacent_positions(self):
        gm = GameMap(8, 8)
        adj = gm.adjacent_positions(3, 3)
        assert len(adj) == 4
        # Corner has only 2
        adj_corner = gm.adjacent_positions(0, 0)
        assert len(adj_corner) == 2

    def test_wall_terrain(self):
        gm = GameMap(8, 8)
        gm.set_wall(4, 4)
        assert not gm.is_passable(4, 4)
        assert gm.is_passable(3, 3)

    def test_get_player_units(self):
        gm = GameMap(8, 8)
        gm.add_unit(UnitType.WORKER, player=0, x=0, y=0)
        gm.add_unit(UnitType.WORKER, player=0, x=1, y=0)
        gm.add_unit(UnitType.WORKER, player=1, x=7, y=7)
        assert len(gm.get_player_units(0)) == 2
        assert len(gm.get_player_units(1)) == 1


class TestGameState:
    def test_observation_shape(self):
        gm = GameMap.create_standard_map(8)
        state = GameState(gm)
        obs = state.get_observation(player=0)
        assert obs.shape == (8, 8, NUM_FEATURE_PLANES)
        assert obs.dtype == np.float32

    def test_observation_values(self):
        gm = GameMap.create_standard_map(8)
        state = GameState(gm)
        obs = state.get_observation(player=0)
        # All values should be 0 or 1 (binary feature planes)
        assert np.all((obs == 0) | (obs == 1))

    def test_action_mask_shape(self):
        gm = GameMap.create_standard_map(8)
        state = GameState(gm)
        mask = state.get_action_mask(player=0)
        expected_cols = sum(get_action_space_dims())
        assert mask.shape == (64, expected_cols)

    def test_action_mask_noop_always_valid(self):
        gm = GameMap.create_standard_map(8)
        state = GameState(gm)
        mask = state.get_action_mask(player=0)
        # NOOP (index 0) should be valid for every cell
        assert np.all(mask[:, 0] == 1.0)

    def test_game_over_no_units(self):
        gm = GameMap(8, 8)
        state = GameState(gm)
        # No units -> draw
        done, winner = state.check_game_over()
        assert done
        assert winner == 2

    def test_game_over_one_player_eliminated(self):
        gm = GameMap(8, 8)
        gm.add_unit(UnitType.BASE, player=0, x=0, y=0)
        state = GameState(gm)
        done, winner = state.check_game_over()
        assert done
        assert winner == 0

    def test_reward_win(self):
        gm = GameMap(8, 8)
        gm.add_unit(UnitType.BASE, player=0, x=0, y=0)
        state = GameState(gm)
        prev_info = {'resources': 0, 'unit_count': 1, 'combat_count': 0, 'enemy_hp': 0}
        reward = state.compute_reward(0, prev_info)
        assert reward >= 10.0  # Win bonus


class TestActions:
    def test_action_space_dims(self):
        dims = get_action_space_dims()
        assert len(dims) == DIMS_PER_CELL
        assert dims[0] == 6  # action types

    def test_direction_offsets(self):
        assert DIR_OFFSETS[Direction.NORTH] == (0, -1)
        assert DIR_OFFSETS[Direction.EAST] == (1, 0)
        assert DIR_OFFSETS[Direction.SOUTH] == (0, 1)
        assert DIR_OFFSETS[Direction.WEST] == (-1, 0)

    def test_valid_actions_idle_worker(self):
        gm = GameMap(8, 8)
        worker = gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        # Place a resource adjacent
        gm.add_unit(UnitType.RESOURCE, player=-1, x=4, y=3)
        # Place own base adjacent
        gm.add_unit(UnitType.BASE, player=0, x=3, y=2)

        mask = get_valid_actions_mask(worker, gm, player_resources=5)
        # NOOP valid
        assert mask[0][ActionType.NOOP] == 1
        # MOVE valid (some directions)
        assert mask[0][ActionType.MOVE] == 1
        # HARVEST valid (resource to east)
        assert mask[0][ActionType.HARVEST] == 1
        assert mask[2][Direction.EAST] == 1

    def test_valid_actions_busy_unit(self):
        gm = GameMap(8, 8)
        worker = gm.add_unit(UnitType.WORKER, player=0, x=3, y=3)
        worker.action_state = ActionState.MOVING
        worker.action_ticks_remaining = 5

        mask = get_valid_actions_mask(worker, gm, player_resources=5)
        # Only NOOP valid when busy
        assert mask[0][ActionType.NOOP] == 1
        assert sum(mask[0]) == 1

    def test_decode_cell_action(self):
        worker = Unit(unit_id=0, unit_type=UnitType.WORKER, player=0, x=3, y=3)
        cell_action = [int(ActionType.MOVE), int(Direction.EAST), 0, 0, 0, 0, 0]
        action = decode_cell_action(cell_action, worker, 3, 3)
        assert action is not None
        assert action.action_type == ActionType.MOVE
        assert action.direction == Direction.EAST


class TestGameEngine:
    def test_reset(self):
        engine = GameEngine(map_size=8, max_ticks=100)
        state = engine.reset()
        assert state is not None
        assert state.tick == 0
        assert not state.done

    def test_step_noop(self):
        engine = GameEngine(map_size=8, max_ticks=100)
        state = engine.reset()
        h, w = 8, 8
        p0 = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        p1 = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        state, info = engine.step(p0, p1)
        assert state.tick == 1
        assert info['tick'] == 1

    def test_game_runs_to_completion(self):
        engine = GameEngine(map_size=8, max_ticks=100)
        state = engine.reset()
        p0_ai = RushAI()
        p1_ai = RandomAI()

        while not state.done:
            p0 = p0_ai.get_action(state, player=0)
            p1 = p1_ai.get_action(state, player=1)
            state, info = engine.step(p0, p1)

        assert state.done
        assert state.winner in (0, 1, 2)

    def test_resource_harvesting(self):
        gm = GameMap(8, 8)
        base = gm.add_unit(UnitType.BASE, player=0, x=3, y=3)
        worker = gm.add_unit(UnitType.WORKER, player=0, x=4, y=3)
        resource = gm.add_unit(UnitType.RESOURCE, player=-1, x=5, y=3)

        state = GameState(gm)
        state.player_resources = {0: 0, 1: 0}
        engine = GameEngine(map_size=8)
        engine.state = state

        # Verify initial state
        assert worker.resources_carried == 0
        assert worker.is_idle


class TestVecEnv:
    def test_vec_env_creation(self):
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        obs = env.reset()
        assert obs.shape == (2, 8, 8, NUM_FEATURE_PLANES)

    def test_vec_env_step(self):
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        obs = env.reset()
        actions = np.zeros((2, 8 * 8 * DIMS_PER_CELL), dtype=np.int32)
        next_obs, rewards, dones, infos = env.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2

    def test_vec_env_action_mask(self):
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        env.reset()
        masks = env.get_action_mask()
        expected_cols = sum(get_action_space_dims())
        assert masks.shape == (2, 64, expected_cols)


class TestAIOpponents:
    def test_random_ai_produces_valid_actions(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        ai = RandomAI()
        action = ai.get_action(state, player=0)
        assert action.shape == (8 * 8 * DIMS_PER_CELL,)

    def test_rush_ai(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        ai = RushAI()
        action = ai.get_action(state, player=0)
        assert action.shape == (8 * 8 * DIMS_PER_CELL,)

    def test_economy_ai(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        ai = EconomyAI()
        action = ai.get_action(state, player=1)
        assert action.shape == (8 * 8 * DIMS_PER_CELL,)

    def test_defensive_ai(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        ai = DefensiveAI()
        action = ai.get_action(state, player=1)
        assert action.shape == (8 * 8 * DIMS_PER_CELL,)

    def test_ai_vs_ai_game(self):
        """Run a full game between two AIs."""
        engine = GameEngine(map_size=8, max_ticks=200)
        state = engine.reset()
        rush = RushAI()
        eco = EconomyAI()

        steps = 0
        while not state.done:
            p0 = rush.get_action(state, player=0)
            p1 = eco.get_action(state, player=1)
            state, _ = engine.step(p0, p1)
            steps += 1

        assert state.done
        assert steps > 0


class TestRenderer:
    def test_render(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        output = GameRenderer.render(state)
        assert "Tick:" in output
        assert "P0" in output
        assert "P1" in output

    def test_render_compact(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        output = GameRenderer.render_compact(state)
        assert "T0000" in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
