"""
Tests for the RTS AI agent, encoder, policy, knowledge store, and transfer bridge.
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from game.engine import GameEngine, VecGameEnv
from game.game_state import GameState, NUM_FEATURE_PLANES
from game.ai_opponents import RandomAI, RushAI
from game.actions import get_action_space_dims, DIMS_PER_CELL

from rts_ai.encoder import GameStateEncoder
from rts_ai.policy import GridNetPolicy
from rts_ai.agent import PPOAgent, RolloutBuffer
from rts_ai.knowledge_store import KnowledgeStore, StrategicKnowledge
from rts_ai.transfer import TransferBridge, StrategicConcept, AbstractionLevel


class TestGameStateEncoder:
    def test_encode_observation(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        encoder = GameStateEncoder(8, 8)
        obs = encoder.encode_observation(state, player=0)
        assert obs.shape == (8, 8, NUM_FEATURE_PLANES)

    def test_encode_for_pattern_engine(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        encoder = GameStateEncoder(8, 8)
        features = encoder.encode_for_pattern_engine(state, player=0)
        assert isinstance(features, list)
        assert len(features) > 10
        assert all(isinstance(f, float) for f in features)
        # All features should be normalized roughly [0, 1]
        assert all(-0.5 <= f <= 1.5 for f in features)

    def test_strategic_summary(self):
        engine = GameEngine(map_size=8)
        state = engine.reset()
        encoder = GameStateEncoder(8, 8)
        summary = encoder.extract_strategic_summary(state, player=0)
        assert 'economy_ratio' in summary
        assert 'military_ratio' in summary
        assert 'game_phase' in summary
        assert 0 <= summary['economy_ratio'] <= 1
        assert 0 <= summary['military_ratio'] <= 1


class TestGridNetPolicy:
    def test_creation(self):
        policy = GridNetPolicy(map_height=8, map_width=8, hidden_dim=32)
        assert policy.map_height == 8
        assert policy.total_action_dim == sum(get_action_space_dims())

    def test_forward(self):
        policy = GridNetPolicy(map_height=8, map_width=8, hidden_dim=32)
        batch = 2
        obs = np.random.randn(batch, 8, 8, NUM_FEATURE_PLANES).astype(np.float32)
        total_dim = sum(get_action_space_dims())
        mask = np.ones((batch, 64, total_dim), dtype=np.float32)

        logits, values, hidden = policy.forward(obs, mask)
        assert logits.shape == (batch, 64, total_dim)
        assert values.shape == (batch,)
        assert hidden.shape == (batch, 8, 8, 32)

    def test_get_action(self):
        policy = GridNetPolicy(map_height=8, map_width=8, hidden_dim=32)
        obs = np.random.randn(1, 8, 8, NUM_FEATURE_PLANES).astype(np.float32)
        total_dim = sum(get_action_space_dims())
        mask = np.ones((1, 64, total_dim), dtype=np.float32)

        actions, log_probs, values = policy.get_action(obs, mask)
        assert actions.shape == (1, 64, DIMS_PER_CELL)
        assert log_probs.shape == (1,)
        assert values.shape == (1,)

    def test_save_load(self):
        policy = GridNetPolicy(map_height=8, map_width=8, hidden_dim=32)
        original_params = policy.get_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_policy')
            policy.save(path)

            # Create new policy and load
            policy2 = GridNetPolicy(map_height=8, map_width=8, hidden_dim=32)
            policy2.load(path + '.npz')

            loaded_params = policy2.get_params()
            for key in original_params:
                np.testing.assert_array_almost_equal(
                    original_params[key], loaded_params[key]
                )


class TestPPOAgent:
    def test_creation(self):
        agent = PPOAgent(map_height=8, map_width=8)
        assert agent.map_height == 8
        assert agent.total_steps == 0

    def test_collect_rollouts(self):
        agent = PPOAgent(map_height=8, map_width=8,
                         config={'n_steps': 4})
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        info = agent.collect_rollouts(env)
        assert agent.buffer.size == 4
        assert 'final_values' in info

    def test_compute_advantages(self):
        agent = PPOAgent(map_height=8, map_width=8,
                         config={'n_steps': 4})
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        info = agent.collect_rollouts(env)
        advantages, returns = agent.compute_advantages(info['final_values'])
        assert advantages.shape == (4, 2)
        assert returns.shape == (4, 2)

    def test_save_load(self):
        agent = PPOAgent(map_height=8, map_width=8)
        agent.total_steps = 1000
        agent.episodes_completed = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            agent2 = PPOAgent(map_height=8, map_width=8)
            agent2.load(tmpdir)
            assert agent2.total_steps == 1000
            assert agent2.episodes_completed == 50

    def test_short_training_run(self):
        """Run a very short training to verify the pipeline works end-to-end."""
        agent = PPOAgent(map_height=8, map_width=8, config={
            'n_steps': 8,
            'n_minibatches': 2,
            'n_epochs': 2,
            'total_timesteps': 64,
            'hidden_dim': 16,
        })
        env = VecGameEnv(num_envs=2, map_size=8, opponent_ai=RandomAI())
        results = agent.train(env, total_timesteps=64, log_interval=100)
        assert results['total_steps'] >= 64
        env.close()


class TestKnowledgeStore:
    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=tmpdir)
            assert len(ks.knowledge) == 0

    def test_add_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=tmpdir)

            knowledge = StrategicKnowledge(
                knowledge_id="test_1",
                category="military",
                description="Rush with light units",
                conditions={"game_phase": 0.2, "resource_level": 0.5},
                actions={"aggression": 0.9, "economy_focus": 0.1},
                effectiveness=0.8,
                confidence=0.7,
                game_source="micrortsai",
            )
            ks.add_knowledge(knowledge)

            results = ks.query_knowledge(category="military")
            assert len(results) == 1
            assert results[0].knowledge_id == "test_1"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=tmpdir)
            knowledge = StrategicKnowledge(
                knowledge_id="test_2",
                category="economy",
                description="Build workers first",
                conditions={"game_phase": 0.1},
                actions={"economy_focus": 0.9},
                effectiveness=0.7,
                confidence=0.6,
                game_source="micrortsai",
            )
            ks.add_knowledge(knowledge)
            ks.save()

            # Reload
            ks2 = KnowledgeStore(store_path=tmpdir)
            assert len(ks2.knowledge) == 1
            assert "test_2" in ks2.knowledge

    def test_update_effectiveness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=tmpdir)
            knowledge = StrategicKnowledge(
                knowledge_id="test_3",
                category="timing",
                description="Attack at tick 100",
                conditions={},
                actions={},
                effectiveness=0.5,
                confidence=0.5,
                game_source="micrortsai",
            )
            ks.add_knowledge(knowledge)
            ks.update_effectiveness("test_3", success=True)
            assert ks.knowledge["test_3"].times_applied == 1
            assert ks.knowledge["test_3"].times_successful == 1

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=tmpdir)
            stats = ks.get_stats()
            assert stats['total_knowledge'] == 0


class TestTransferBridge:
    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=os.path.join(tmpdir, 'knowledge'))
            tb = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))
            assert len(tb.concepts) > 0  # Should have universal concepts

    def test_universal_concepts_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=os.path.join(tmpdir, 'knowledge'))
            tb = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))
            assert "rush" in tb.concepts
            assert "boom" in tb.concepts
            assert "timing_attack" in tb.concepts

    def test_recommendations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=os.path.join(tmpdir, 'knowledge'))
            tb = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))
            recs = tb.get_recommendations(
                "age_of_empires",
                current_conditions={"game_phase": "early"},
                top_k=3,
            )
            assert len(recs) == 3
            assert 'concept' in recs[0]
            assert 'score' in recs[0]

    def test_game_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=os.path.join(tmpdir, 'knowledge'))
            tb = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))

            aoe_info = {
                'unit_types': ['Villager', 'Knight', 'Crossbowman'],
                'resources': ['Food', 'Gold'],
                'buildings': ['Town Center', 'Barracks'],
            }
            adapter = tb.generate_game_adapter("micrortsai", "aoe", aoe_info)
            assert adapter['source_game'] == "micrortsai"
            assert adapter['target_game'] == "aoe"
            assert 'unit_mappings' in adapter
            assert 'recommendations' in adapter

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ks = KnowledgeStore(store_path=os.path.join(tmpdir, 'knowledge'))
            tb = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))
            tb.save()

            # Reload
            tb2 = TransferBridge(ks, concepts_path=os.path.join(tmpdir, 'concepts'))
            assert len(tb2.concepts) == len(tb.concepts)


class TestRolloutBuffer:
    def test_buffer(self):
        buf = RolloutBuffer()
        assert buf.size == 0
        buf.observations.append(np.zeros((2, 8, 8, 27)))
        assert buf.size == 1
        buf.clear()
        assert buf.size == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
