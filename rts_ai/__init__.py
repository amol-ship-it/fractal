"""
RTS AI Agent - Reinforcement learning agent for the MicroRTS-style game.

Integrates PPO-based policy learning with the Recursive Learning AI's
pattern recognition system. Learned strategic patterns are stored
persistently for transfer to more complex games (e.g., Age of Empires).

Architecture:
- GridNet CNN encoder/decoder for spatial action selection
- PPO with invalid action masking
- Pattern memory bridge to Recursive Learning AI core
- Persistent knowledge store for cross-game transfer
"""

from rts_ai.encoder import GameStateEncoder
from rts_ai.policy import GridNetPolicy
from rts_ai.agent import PPOAgent
from rts_ai.knowledge_store import KnowledgeStore
from rts_ai.transfer import TransferBridge, StrategicConcept

__all__ = [
    "GameStateEncoder",
    "GridNetPolicy",
    "PPOAgent",
    "KnowledgeStore",
    "TransferBridge",
    "StrategicConcept",
]
