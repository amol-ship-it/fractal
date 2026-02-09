"""
Chess Pattern Policy — Uses the four pillars of learning to play chess.

Instead of a neural network, this policy:
1. Encodes the board position as a feature vector
2. Quantizes it into a situation key (e.g. "mat+_mid_saf_hi_ceq")
3. Looks up the best strategy for that situation from learned bindings
4. Executes the strategy by picking a concrete legal move
5. Learns from game outcomes (wins refine which situation→strategy pairs work)

The four pillars drive every decision:
- Feedback Loops: Win/loss/draw outcomes refine strategy bindings
- Approximability: Pattern signatures improve with each game
- Composability: Atomic patterns compose into strategic concepts
- Exploration: Agent occasionally tries novel strategies
"""

import random
import json
import os
import chess
from typing import Dict, List, Tuple, Optional
from enum import IntEnum

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory
from core.feedback import (
    FeedbackLoop, FeedbackType, FeedbackSignal, ExplorationStrategy
)
from chess_ai.board_encoder import ChessBoardEncoder, PIECE_VALUES


# ── Chess Strategies ──────────────────────────────────────────────────────

class ChessStrategy(IntEnum):
    DEVELOP = 0         # Develop minor pieces, prepare castling
    CONTROL_CENTER = 1  # Maximize center influence
    ATTACK_KING = 2     # Direct pieces toward opponent king
    TRADE_PIECES = 3    # Exchange pieces (when ahead in material)
    PUSH_PAWNS = 4      # Advance pawns (especially passed pawns)
    DEFEND = 5          # Defend attacked pieces
    CASTLE = 6          # Prioritize castling
    ENDGAME_PUSH = 7    # Centralize king + advance passed pawns

NUM_STRATEGIES = len(ChessStrategy)


# ── Strategy Binding ──────────────────────────────────────────────────────

class StrategyBinding:
    """
    Links a situation key to a strategy, tracking win/loss performance.
    This is what the agent LEARNS — which situation calls for which strategy.
    """

    def __init__(self, strategy: int, pattern_id: str):
        self.strategy = strategy
        self.pattern_id = pattern_id  # actually the situation key
        self.wins = 0.0
        self.losses = 0.0
        self.times_used = 0

    @property
    def confidence(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.5  # Uninformed prior
        return self.wins / total

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


# ── Helpers ───────────────────────────────────────────────────────────────

# Piece values for strategy scoring (centipawns)
_PV = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}

# Center and extended center squares
_CENTER = {chess.E4, chess.D4, chess.E5, chess.D5}
_BACK_RANK_W = set(range(chess.A1, chess.H1 + 1))
_BACK_RANK_B = set(range(chess.A8, chess.H8 + 1))

# King zone: squares around the king (including king square)
def _king_zone(board: chess.Board, color: chess.Color) -> List[int]:
    """Get squares around the opponent's king."""
    king_sq = board.king(not color)
    if king_sq is None:
        return []
    zone = [king_sq]
    kr, kf = chess.square_rank(king_sq), chess.square_file(king_sq)
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            r, f = kr + dr, kf + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                sq = chess.square(f, r)
                if sq != king_sq:
                    zone.append(sq)
    return zone


def _score_center_control(board: chess.Board, color: chess.Color) -> int:
    """Count our attackers on center squares."""
    score = 0
    for sq in _CENTER:
        score += len(board.attackers(color, sq))
    return score


def _is_passed_pawn(board: chess.Board, sq: int, color: chess.Color) -> bool:
    """Check if a pawn is passed."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    opp = not color

    for opp_sq in board.pieces(chess.PAWN, opp):
        of = chess.square_file(opp_sq)
        orank = chess.square_rank(opp_sq)
        if abs(of - f) <= 1:
            if color == chess.WHITE and orank > r:
                return False
            elif color == chess.BLACK and orank < r:
                return False
    return True


# ── Strategy Executors ────────────────────────────────────────────────────

def _execute_develop(board: chess.Board,
                     color: chess.Color) -> Optional[chess.Move]:
    """DEVELOP: Move undeveloped minor pieces toward center, or castle."""
    moves = list(board.legal_moves)
    if not moves:
        return None

    back_rank = _BACK_RANK_W if color == chess.WHITE else _BACK_RANK_B
    best_move = None
    best_score = -999

    for move in moves:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.color != color:
            continue

        score = 0

        # Castling is great for development
        if board.is_castling(move):
            score = 50
        # Move minor pieces off back rank
        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if move.from_square in back_rank:
                score = 20
                # Bonus for center-adjacent destination
                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)
                center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
                score += max(0, 10 - int(center_dist * 3))
            else:
                # Already developed, small bonus for improving position
                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)
                center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
                score = max(0, 5 - int(center_dist * 2))
        # Pawn moves that support development (e4, d4, etc.)
        elif piece.piece_type == chess.PAWN:
            to_sq = move.to_square
            if to_sq in _CENTER:
                score = 10

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score > 0 else None


def _execute_control_center(board: chess.Board,
                            color: chess.Color) -> Optional[chess.Move]:
    """CONTROL_CENTER: Pick move that most increases center control."""
    moves = list(board.legal_moves)
    if not moves:
        return None

    base_control = _score_center_control(board, color)
    best_move = None
    best_improvement = -999

    for move in moves:
        board.push(move)
        new_control = _score_center_control(board, color)
        improvement = new_control - base_control

        # Small bonus for not losing material
        if board.is_check():
            improvement += 2

        board.pop()

        if improvement > best_improvement:
            best_improvement = improvement
            best_move = move

    return best_move if best_improvement > 0 else None


def _execute_attack_king(board: chess.Board,
                         color: chess.Color) -> Optional[chess.Move]:
    """ATTACK_KING: Direct pieces toward opponent king zone, prefer checks."""
    moves = list(board.legal_moves)
    if not moves:
        return None

    zone = _king_zone(board, color)
    if not zone:
        return None

    best_move = None
    best_score = -999

    for move in moves:
        score = 0
        board.push(move)

        # Big bonus for checks
        if board.is_check():
            score += 15

        # Count our attacks on king zone
        for sq in zone:
            score += len(board.attackers(color, sq))

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score > 2 else None


def _execute_trade_pieces(board: chess.Board,
                          color: chess.Color) -> Optional[chess.Move]:
    """TRADE_PIECES: Capture highest-value piece (MVV-LVA)."""
    best_move = None
    best_score = -999

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        captured = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if captured is None:
            # En passant
            victim_val = _PV[chess.PAWN]
        else:
            victim_val = _PV.get(captured.piece_type, 0)

        attacker_val = _PV.get(attacker.piece_type, 0) if attacker else 0

        # MVV-LVA: capture most valuable with least valuable
        score = victim_val * 10 - attacker_val

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def _execute_push_pawns(board: chess.Board,
                        color: chess.Color) -> Optional[chess.Move]:
    """PUSH_PAWNS: Advance pawns, prioritize passed pawns and promotions."""
    best_move = None
    best_score = -999

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            continue
        if piece.color != color:
            continue

        to_rank = chess.square_rank(move.to_square)
        if color == chess.WHITE:
            advancement = to_rank  # 0-7
        else:
            advancement = 7 - to_rank

        score = advancement

        # Promotion is huge
        if move.promotion:
            score += 20

        # Passed pawn bonus
        if _is_passed_pawn(board, move.from_square, color):
            score += 8

        # Center pawn push bonus
        if move.to_square in _CENTER:
            score += 3

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score > 0 else None


def _execute_defend(board: chess.Board,
                    color: chess.Color) -> Optional[chess.Move]:
    """DEFEND: Move attacked pieces to safety or add defenders."""
    opp = not color
    best_move = None
    best_score = -999

    # Find our pieces under attack by opponent
    threatened = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            if board.is_attacked_by(opp, sq):
                threatened.append((sq, _PV.get(piece.piece_type, 0)))

    if not threatened:
        return None

    # Sort by value (most valuable first)
    threatened.sort(key=lambda x: x[1], reverse=True)

    for move in board.legal_moves:
        score = 0
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue

        # Moving a threatened piece away
        for sq, val in threatened:
            if move.from_square == sq:
                # Check if destination is safe
                board.push(move)
                if not board.is_attacked_by(opp, move.to_square):
                    score = val // 10  # Scale down
                board.pop()
                break

        # Adding a defender to a threatened piece
        if score == 0:
            board.push(move)
            for sq, val in threatened[:3]:  # Check top 3 threatened
                if board.is_attacked_by(color, sq):
                    # We defend it — but did we already?
                    board.pop()
                    board_before = board.copy()
                    defenders_before = len(board_before.attackers(color, sq))
                    board.push(move)
                    defenders_after = len(board.attackers(color, sq))
                    if defenders_after > defenders_before:
                        score = val // 20
                    break
            else:
                board.pop()
                continue
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score > 0 else None


def _execute_castle(board: chess.Board,
                    color: chess.Color) -> Optional[chess.Move]:
    """CASTLE: Castle if legal, else try to unblock castling path."""
    # Direct castling
    for move in board.legal_moves:
        if board.is_castling(move):
            return move

    # If castling rights exist but can't castle yet, try to clear the path
    can_kingside = board.has_kingside_castling_rights(color)
    can_queenside = board.has_queenside_castling_rights(color)

    if not can_kingside and not can_queenside:
        return None

    # Try to move blocking pieces
    if color == chess.WHITE:
        kingside_between = [chess.F1, chess.G1]
        queenside_between = [chess.D1, chess.C1, chess.B1]
    else:
        kingside_between = [chess.F8, chess.G8]
        queenside_between = [chess.D8, chess.C8, chess.B8]

    blocking_squares = []
    if can_kingside:
        blocking_squares.extend(kingside_between)
    if can_queenside:
        blocking_squares.extend(queenside_between)

    for move in board.legal_moves:
        if move.from_square in blocking_squares:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == color:
                return move

    return None


def _execute_endgame_push(board: chess.Board,
                          color: chess.Color) -> Optional[chess.Move]:
    """ENDGAME_PUSH: Centralize king + advance passed pawns."""
    best_move = None
    best_score = -999

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.color != color:
            continue

        score = 0

        if piece.piece_type == chess.KING:
            # Move king toward center
            to_f = chess.square_file(move.to_square)
            to_r = chess.square_rank(move.to_square)
            from_f = chess.square_file(move.from_square)
            from_r = chess.square_rank(move.from_square)

            old_dist = abs(from_f - 3.5) + abs(from_r - 3.5)
            new_dist = abs(to_f - 3.5) + abs(to_r - 3.5)
            if new_dist < old_dist:
                score = 10

        elif piece.piece_type == chess.PAWN:
            # Push passed pawns
            if _is_passed_pawn(board, move.from_square, color):
                to_rank = chess.square_rank(move.to_square)
                advancement = to_rank if color == chess.WHITE else (7 - to_rank)
                score = 5 + advancement
                if move.promotion:
                    score += 20

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score > 0 else None


# ── Strategy Dispatch ─────────────────────────────────────────────────────

STRATEGY_EXECUTORS = {
    ChessStrategy.DEVELOP: _execute_develop,
    ChessStrategy.CONTROL_CENTER: _execute_control_center,
    ChessStrategy.ATTACK_KING: _execute_attack_king,
    ChessStrategy.TRADE_PIECES: _execute_trade_pieces,
    ChessStrategy.PUSH_PAWNS: _execute_push_pawns,
    ChessStrategy.DEFEND: _execute_defend,
    ChessStrategy.CASTLE: _execute_castle,
    ChessStrategy.ENDGAME_PUSH: _execute_endgame_push,
}


# ── Chess Pattern Policy ─────────────────────────────────────────────────

class ChessPatternPolicy:
    """
    Pattern-based chess policy driven by the four pillars of learning.

    Each turn:
    1. Quantize the board into a situation key
    2. Feed features to PatternEngine (composability + pattern discovery)
    3. Look up best strategy for this situation from learned bindings
    4. Execute the strategy to pick a concrete legal move
    5. After game ends, feed outcome back (feedback + approximability)
    """

    def __init__(self):
        # Core AI components (the four pillars)
        self.memory = DualMemory(max_patterns=5000, max_state=500)
        self.engine = PatternEngine(self.memory)
        self.feedback_loop = FeedbackLoop(
            learning_rate=0.1, discount_factor=0.99)
        self.explorer = ExplorationStrategy(exploration_rate=0.3)
        self._decay_rate = 0.995

        # Chess-specific
        self.encoder = ChessBoardEncoder()

        # Strategy bindings: situation_key -> {strategy_idx -> StrategyBinding}
        self.strategy_bindings: Dict[str, Dict[int, StrategyBinding]] = {}

        # Per-game tracking
        self._episode_decisions: List[Tuple[str, int]] = []

    def _situation_key(self, board: chess.Board,
                       color: chess.Color) -> str:
        """
        Quantize the board into a discrete situation key.

        This is the APPROXIMABILITY pillar: similar positions map to the
        same key, so the agent generalizes across similar games.
        """
        # 1. Material advantage
        my_mat = sum(len(board.pieces(pt, color)) * v
                     for pt, v in PIECE_VALUES.items())
        opp_mat = sum(len(board.pieces(pt, not color)) * v
                      for pt, v in PIECE_VALUES.items())
        diff = my_mat - opp_mat
        if diff <= -5:
            mat = "--"
        elif diff <= -1:
            mat = "-"
        elif diff == 0:
            mat = "="
        elif diff <= 4:
            mat = "+"
        else:
            mat = "++"

        # 2. Game phase
        phase_val = self.encoder._game_phase_feature(board)[0]
        if phase_val < 0.25:
            phase = "opn"
        elif phase_val < 0.65:
            phase = "mid"
        else:
            phase = "end"

        # 3. King safety
        shield = self.encoder._pawn_shield_score(board, color)
        king_sq = board.king(color)
        in_check = board.is_check() if board.turn == color else False

        if in_check:
            safety = "dan"
        elif shield >= 0.66:
            safety = "saf"
        elif shield >= 0.33:
            safety = "mod"
        else:
            # King in center with no shield = danger in opening/middlegame
            if king_sq is not None and phase != "end":
                kf = chess.square_file(king_sq)
                if 2 <= kf <= 5:  # King still in center files
                    safety = "dan"
                else:
                    safety = "mod"
            else:
                safety = "mod"

        # 4. Development
        back_rank = _BACK_RANK_W if color == chess.WHITE else _BACK_RANK_B
        developed = 0
        for pt in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(pt, color):
                if sq not in back_rank:
                    developed += 1
        if developed < 2:
            dev = "lo"
        elif developed < 4:
            dev = "md"
        else:
            dev = "hi"

        # 5. Center control
        my_center = _score_center_control(board, color)
        opp_center = _score_center_control(board, not color)
        center_diff = my_center - opp_center
        if center_diff < -2:
            center = "cwk"
        elif center_diff <= 2:
            center = "ceq"
        else:
            center = "cst"

        return f"mat{mat}_{phase}_{safety}_{dev}_{center}"

    def select_strategy(self, board: chess.Board,
                        color: chess.Color) -> Tuple[int, str]:
        """
        Select a strategy using situation key lookup + exploration.

        No persistence needed — chess is turn-based, each move is fresh.
        """
        sit_key = self._situation_key(board, color)

        # Feed features to PatternEngine for discovery/composition
        features = self.encoder.encode_board(board)
        self.engine.process(features, domain="chess_situation")

        # Look up best strategy for this situation
        best_strategy = None
        best_confidence = -1.0

        if sit_key in self.strategy_bindings:
            for strat_idx, binding in self.strategy_bindings[sit_key].items():
                if binding.confidence > best_confidence:
                    best_confidence = binding.confidence
                    best_strategy = strat_idx

        # Exploration: sometimes try a random strategy
        if self.explorer.should_explore() or best_strategy is None:
            best_strategy = random.randint(0, NUM_STRATEGIES - 1)

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

    def execute_strategy(self, strategy: int, board: chess.Board,
                         color: chess.Color) -> chess.Move:
        """
        Execute a strategy, returning a concrete chess move.
        Falls back to random legal move if executor returns None.
        """
        executor = STRATEGY_EXECUTORS.get(strategy, _execute_develop)
        move = executor(board, color)

        if move is None or move not in board.legal_moves:
            # Fallback to random legal move
            moves = list(board.legal_moves)
            move = random.choice(moves) if moves else None

        return move

    def get_move(self, board: chess.Board,
                 color: chess.Color) -> chess.Move:
        """
        Full decision pipeline: situation → strategy → move.
        This is the main entry point called each turn.
        """
        strategy, sit_key = self.select_strategy(board, color)
        return self.execute_strategy(strategy, board, color)

    def begin_episode(self):
        """Call at the start of each training game."""
        self._episode_decisions = []

    def record_outcome(self, won: bool, draw: bool = False,
                       game_info: Dict = None):
        """
        After a game ends, apply feedback to all situations and strategies
        used during this game. This is the FEEDBACK pillar in action.
        """
        game_info = game_info or {}

        # Feedback value: win=0.8, draw=0.5, loss=0.2
        if draw:
            feedback_value = 0.5
        elif won:
            feedback_value = 0.8
        else:
            feedback_value = 0.2

        # Deduplicate decisions
        seen = set()
        unique_decisions = []
        for sit_key, strat in self._episode_decisions:
            key = (sit_key, strat)
            if key not in seen:
                seen.add(key)
                unique_decisions.append(key)

        # Update strategy bindings (FEEDBACK pillar)
        for sit_key, strategy in unique_decisions:
            if sit_key in self.strategy_bindings:
                if strategy in self.strategy_bindings[sit_key]:
                    binding = self.strategy_bindings[sit_key][strategy]
                    if won:
                        binding.wins += 1
                    elif draw:
                        binding.wins += 0.5
                        binding.losses += 0.5
                    else:
                        binding.losses += 1

        # Refine patterns in memory (APPROXIMABILITY pillar)
        all_pattern_ids = list(self.memory.patterns.patterns.keys())
        for pid in all_pattern_ids[-20:]:
            pattern = self.memory.patterns.patterns.get(pid)
            if pattern:
                pattern.refine(feedback=feedback_value)

        # Apply feedback through FeedbackLoop
        if all_pattern_ids:
            fv = 1.0 if won else (-0.5 if not draw else 0.0)
            self.feedback_loop.apply_feedback(
                FeedbackSignal(
                    signal_type=FeedbackType.EXTRINSIC,
                    value=fv,
                    target_pattern_ids=all_pattern_ids[-10:],
                    context={'won': won, 'draw': draw}
                ),
                self.memory.patterns.patterns
            )

        # Process outcome as composite pattern (COMPOSABILITY)
        outcome_features = [
            float(won),
            float(draw),
            game_info.get('num_moves', 0) / 200.0,
            game_info.get('final_material_balance', 0.5),
        ]
        self.engine.process(outcome_features, domain="chess_outcome")

        # Try composing novel patterns (COMPOSABILITY + EXPLORATION)
        all_patterns = list(self.memory.patterns.patterns.values())
        if len(all_patterns) >= 2:
            combo = self.explorer.suggest_combination(all_patterns)
            if combo and len(combo) >= 2:
                composite = Pattern.create_composite(combo, "chess_situation")
                self.memory.patterns.store(composite)

        # Decay exploration rate
        self.explorer.decay_exploration(self._decay_rate)

        # Clear episode tracking
        self._episode_decisions = []

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
                'exploration_rate', 0.3)

    def get_stats(self) -> Dict:
        """Get current learning statistics."""
        total_bindings = sum(
            len(smap) for smap in self.strategy_bindings.values()
        )
        strategy_counts = {s.name: 0 for s in ChessStrategy}
        for sit_key, smap in self.strategy_bindings.items():
            for strat_idx, binding in smap.items():
                name = ChessStrategy(strat_idx).name
                strategy_counts[name] += binding.times_used

        return {
            'patterns_discovered': len(self.memory.patterns.patterns),
            'strategy_bindings': total_bindings,
            'exploration_rate': self.explorer.exploration_rate,
            'strategy_usage': strategy_counts,
        }
