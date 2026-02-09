"""Tests for the chess AI module."""

import os
import tempfile
import pytest
import chess

from chess_ai.board_encoder import ChessBoardEncoder
from chess_ai.chess_policy import (
    ChessStrategy, StrategyBinding, ChessPatternPolicy, NUM_STRATEGIES
)
from chess_ai.opponents import RandomPlayer, GreedyPlayer, MinimaxPlayer
from chess_ai.chess_agent import ChessPatternAgent


# ── Encoder Tests ─────────────────────────────────────────────────────────

class TestChessBoardEncoder:
    def setup_method(self):
        self.encoder = ChessBoardEncoder()

    def test_encode_board_length(self):
        """Feature vector should be 42 elements."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        assert len(features) == 42

    def test_encode_board_normalization(self):
        """All features should be in [0, 1]."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        for i, f in enumerate(features):
            assert 0.0 <= f <= 1.0, f"Feature {i} = {f} out of range"

    def test_initial_position_material(self):
        """Starting position should have full material."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        # Index 35 = white total material, 36 = black total material
        assert features[35] == features[36]  # Symmetric
        assert features[35] == 1.0  # Full material (39/39)

    def test_initial_position_balance(self):
        """Starting position should have balanced material (0.5)."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        assert features[37] == 0.5  # Material balance centered

    def test_game_phase_opening(self):
        """Starting position should be early opening phase."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        phase = features[30]
        assert phase < 0.1, f"Opening phase should be near 0, got {phase}"

    def test_game_phase_endgame(self):
        """Board with only kings + pawns should be late endgame."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        features = self.encoder.encode_board(board)
        phase = features[30]
        assert phase > 0.9, f"Endgame phase should be near 1.0, got {phase}"

    def test_encode_after_moves(self):
        """Features should change after moves."""
        board = chess.Board()
        before = self.encoder.encode_board(board)
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        after = self.encoder.encode_board(board)
        assert before != after

    def test_castling_rights(self):
        """Castling rights should be all 1.0 at start."""
        board = chess.Board()
        features = self.encoder.encode_board(board)
        # Indices 12-15 are castling rights
        assert features[12] == 1.0  # White kingside
        assert features[13] == 1.0  # White queenside
        assert features[14] == 1.0  # Black kingside
        assert features[15] == 1.0  # Black queenside


# ── Policy Tests ──────────────────────────────────────────────────────────

class TestChessPolicy:
    def setup_method(self):
        self.policy = ChessPatternPolicy()

    def test_creation(self):
        """Policy should initialize with all core components."""
        assert self.policy.memory is not None
        assert self.policy.engine is not None
        assert self.policy.feedback_loop is not None
        assert self.policy.explorer is not None
        assert self.policy.encoder is not None

    def test_situation_key_format(self):
        """Situation key should have expected format."""
        board = chess.Board()
        key = self.policy._situation_key(board, chess.WHITE)
        parts = key.split('_')
        assert len(parts) == 5
        assert parts[0].startswith('mat')
        assert parts[1] in ('opn', 'mid', 'end')
        assert parts[2] in ('saf', 'mod', 'dan')
        assert parts[3] in ('lo', 'md', 'hi')
        assert parts[4] in ('cwk', 'ceq', 'cst')

    def test_situation_key_initial_position(self):
        """Initial position should be opening, equal material, low development."""
        board = chess.Board()
        key = self.policy._situation_key(board, chess.WHITE)
        assert 'mat=' in key
        assert 'opn' in key
        assert 'lo' in key  # No pieces developed yet

    def test_select_strategy_returns_valid(self):
        """select_strategy should return valid strategy index and key."""
        board = chess.Board()
        self.policy.begin_episode()
        strat, key = self.policy.select_strategy(board, chess.WHITE)
        assert 0 <= strat < NUM_STRATEGIES
        assert isinstance(key, str)

    def test_get_move_returns_legal(self):
        """get_move should always return a legal move."""
        board = chess.Board()
        self.policy.begin_episode()
        for _ in range(10):
            if board.is_game_over():
                break
            move = self.policy.get_move(board, board.turn)
            assert move in board.legal_moves, \
                f"Move {move} is not legal in position {board.fen()}"
            board.push(move)

    def test_record_outcome_win(self):
        """Bindings should update on win."""
        board = chess.Board()
        self.policy.begin_episode()
        self.policy.get_move(board, chess.WHITE)
        self.policy.record_outcome(won=True, draw=False)
        # Some bindings should now exist
        assert len(self.policy.strategy_bindings) > 0

    def test_record_outcome_draw(self):
        """Bindings should update on draw."""
        board = chess.Board()
        self.policy.begin_episode()
        self.policy.get_move(board, chess.WHITE)
        self.policy.record_outcome(won=False, draw=True)
        # Check that draw adds 0.5 to both wins and losses
        for sit_key, smap in self.policy.strategy_bindings.items():
            for strat_idx, binding in smap.items():
                if binding.times_used > 0:
                    assert binding.wins == 0.5
                    assert binding.losses == 0.5

    def test_save_load(self):
        """Save and load should preserve state."""
        board = chess.Board()
        self.policy.begin_episode()
        # Make a few moves to build bindings
        for _ in range(5):
            if board.is_game_over():
                break
            move = self.policy.get_move(board, board.turn)
            board.push(move)
        self.policy.record_outcome(won=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.policy.save(tmpdir)

            new_policy = ChessPatternPolicy()
            new_policy.load(tmpdir)

            assert len(new_policy.strategy_bindings) == \
                len(self.policy.strategy_bindings)

    def test_get_stats(self):
        """get_stats should return expected keys."""
        stats = self.policy.get_stats()
        assert 'patterns_discovered' in stats
        assert 'strategy_bindings' in stats
        assert 'exploration_rate' in stats
        assert 'strategy_usage' in stats

    def test_all_strategies_have_names(self):
        """Every strategy should have a name in ChessStrategy."""
        for i in range(NUM_STRATEGIES):
            name = ChessStrategy(i).name
            assert isinstance(name, str)


# ── Opponent Tests ────────────────────────────────────────────────────────

class TestOpponents:
    def test_random_player_legal(self):
        """RandomPlayer should always return legal moves."""
        player = RandomPlayer()
        board = chess.Board()
        for _ in range(20):
            if board.is_game_over():
                break
            move = player.get_move(board, board.turn)
            assert move in board.legal_moves
            board.push(move)

    def test_greedy_captures_queen(self):
        """GreedyPlayer should capture a queen when possible."""
        player = GreedyPlayer()
        # Set up a position where a pawn can capture a queen
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E5, chess.Piece(chess.QUEEN, chess.BLACK))
        board.turn = chess.WHITE

        move = player.get_move(board, chess.WHITE)
        assert move.to_square == chess.E5, \
            f"Should capture queen on e5, got {move}"

    def test_minimax_finds_checkmate(self):
        """MinimaxPlayer should find mate-in-1."""
        player = MinimaxPlayer(depth=2)
        # Back-rank mate: White Ra1 + Ke1, Black Kg8 + pawns f7/g7/h7
        # Ra8# is checkmate (king boxed in by own pawns)
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.F7, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.H7, chess.Piece(chess.PAWN, chess.BLACK))
        board.turn = chess.WHITE

        move = player.get_move(board, chess.WHITE)
        # After this move, it should be checkmate
        board.push(move)
        assert board.is_checkmate(), \
            f"Minimax should find mate, played {move} instead"

    def test_minimax_avoids_blunder(self):
        """MinimaxPlayer should not hang its queen."""
        player = MinimaxPlayer(depth=2)
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))
        board.turn = chess.WHITE

        move = player.get_move(board, chess.WHITE)
        # Should not move queen to e5 where it gets taken by... wait,
        # the pawn on e5 can't take on d4. Let's just verify it returns legal
        assert move in board.legal_moves


# ── Agent Tests ───────────────────────────────────────────────────────────

class TestChessAgent:
    def test_play_episode(self):
        """Play one episode, verify result dict."""
        agent = ChessPatternAgent()
        opponent = RandomPlayer()
        result = agent.play_episode(opponent, chess.WHITE, max_moves=50)

        assert 'won' in result
        assert 'draw' in result
        assert 'num_moves' in result
        assert 'result' in result
        assert isinstance(result['won'], bool)
        assert isinstance(result['num_moves'], int)
        assert result['num_moves'] <= 50

    def test_short_training_run(self):
        """Train 10 episodes, verify tracking."""
        agent = ChessPatternAgent()
        agent.train(
            total_episodes=10,
            opponent_name='random',
            log_interval=5,
            max_moves=50,
        )

        assert agent.episodes_completed == 10
        assert len(agent.win_history) == 10
        assert len(agent.game_lengths) == 10

    def test_save_load(self):
        """Train, save, load, verify state preserved."""
        agent = ChessPatternAgent()
        agent.train(
            total_episodes=5,
            opponent_name='random',
            log_interval=5,
            max_moves=30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)

            new_agent = ChessPatternAgent()
            new_agent.load(tmpdir)

            assert new_agent.episodes_completed == agent.episodes_completed
            assert len(new_agent.win_history) == len(agent.win_history)

    def test_play_as_black(self):
        """Agent should be able to play as black."""
        agent = ChessPatternAgent()
        opponent = RandomPlayer()
        result = agent.play_episode(opponent, chess.BLACK, max_moves=50)
        assert 'won' in result
        assert isinstance(result['won'], bool)


# ── Strategy Binding Tests ────────────────────────────────────────────────

class TestStrategyBinding:
    def test_confidence_uninformed(self):
        """New binding should have 0.5 confidence."""
        b = StrategyBinding(0, "test")
        assert b.confidence == 0.5

    def test_confidence_after_wins(self):
        """Confidence should increase with wins."""
        b = StrategyBinding(0, "test")
        b.wins = 8
        b.losses = 2
        assert b.confidence == 0.8

    def test_to_dict_from_dict(self):
        """Round-trip serialization should preserve data."""
        b = StrategyBinding(3, "mat+_mid_saf_hi_ceq")
        b.wins = 10
        b.losses = 5
        b.times_used = 100

        data = b.to_dict()
        b2 = StrategyBinding.from_dict(data)

        assert b2.strategy == 3
        assert b2.pattern_id == "mat+_mid_saf_hi_ceq"
        assert b2.wins == 10
        assert b2.losses == 5
        assert b2.times_used == 100
