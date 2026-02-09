"""
Chess Board Encoder — Converts a chess.Board into a feature vector
for the pattern recognition engine.

Produces a 42-element normalized feature vector covering:
- Material counts (12)
- Castling rights (4)
- Pawn shield (2)
- Center control (2)
- Development (4)
- Pawn structure (6)
- Game phase (1)
- Mobility (2)
- Space (2)
- Material totals (3)
- Threats (4)
"""

import chess
from typing import List


# Standard piece values (in pawns)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# Center squares
CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

# Back ranks
WHITE_BACK_RANK = [chess.A1, chess.B1, chess.C1, chess.D1,
                   chess.E1, chess.F1, chess.G1, chess.H1]
BLACK_BACK_RANK = [chess.A8, chess.B8, chess.C8, chess.D8,
                   chess.E8, chess.F8, chess.G8, chess.H8]


class ChessBoardEncoder:
    """Encodes a chess board position into a normalized feature vector."""

    def encode_board(self, board: chess.Board) -> List[float]:
        """
        Encode a chess.Board as a 42-element normalized feature vector.

        All values are in [0.0, 1.0].
        """
        features = []
        features.extend(self._material_features(board))        # 12
        features.extend(self._king_safety_features(board))      # 6
        features.extend(self._center_control_features(board))   # 2
        features.extend(self._development_features(board))      # 4
        features.extend(self._pawn_structure_features(board))   # 6
        features.extend(self._game_phase_feature(board))        # 1
        features.extend(self._mobility_features(board))         # 2
        features.extend(self._space_features(board))            # 2
        features.extend(self._material_balance_features(board)) # 3
        features.extend(self._threat_features(board))           # 4
        return features  # 42 total

    def _material_features(self, board: chess.Board) -> List[float]:
        """12 floats: count of each piece type per color, normalized."""
        features = []
        max_counts = {
            chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
            chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1,
        }
        for color in [chess.WHITE, chess.BLACK]:
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING]:
                count = len(board.pieces(pt, color))
                features.append(min(count / max_counts[pt], 1.0))
        return features

    def _king_safety_features(self, board: chess.Board) -> List[float]:
        """6 floats: castling rights (4) + pawn shield scores (2)."""
        features = [
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        ]
        # Pawn shield for each side
        features.append(self._pawn_shield_score(board, chess.WHITE))
        features.append(self._pawn_shield_score(board, chess.BLACK))
        return features

    def _pawn_shield_score(self, board: chess.Board,
                           color: chess.Color) -> float:
        """Score pawn shield around the king (0.0 to 1.0)."""
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0

        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Determine shield squares (one rank in front of king)
        if color == chess.WHITE:
            shield_rank = king_rank + 1
        else:
            shield_rank = king_rank - 1

        if shield_rank < 0 or shield_rank > 7:
            return 0.0

        # Check pawns on files adjacent to and including king's file
        shield_count = 0
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            sq = chess.square(f, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield_count += 1

        return min(shield_count / 3.0, 1.0)

    def _center_control_features(self, board: chess.Board) -> List[float]:
        """2 floats: white and black center control scores."""
        white_control = 0
        black_control = 0
        for sq in CENTER_SQUARES:
            white_control += len(board.attackers(chess.WHITE, sq))
            black_control += len(board.attackers(chess.BLACK, sq))
        return [
            min(white_control / 16.0, 1.0),
            min(black_control / 16.0, 1.0),
        ]

    def _development_features(self, board: chess.Board) -> List[float]:
        """4 floats: minor pieces developed (2) + rooks connected (2)."""
        features = []
        for color in [chess.WHITE, chess.BLACK]:
            back_rank = WHITE_BACK_RANK if color == chess.WHITE else BLACK_BACK_RANK
            # Count minor pieces NOT on back rank
            developed = 0
            total_minors = 0
            for pt in [chess.KNIGHT, chess.BISHOP]:
                for sq in board.pieces(pt, color):
                    total_minors += 1
                    if sq not in back_rank:
                        developed += 1
            features.append(min(developed / 4.0, 1.0))

        # Rooks connected
        for color in [chess.WHITE, chess.BLACK]:
            rooks = list(board.pieces(chess.ROOK, color))
            if len(rooks) >= 2:
                r1, r2 = rooks[0], rooks[1]
                # Connected if on same rank/file with no pieces between
                connected = self._rooks_connected(board, r1, r2)
                features.append(1.0 if connected else 0.0)
            else:
                features.append(0.0)
        return features

    def _rooks_connected(self, board: chess.Board,
                         sq1: int, sq2: int) -> bool:
        """Check if two rooks can see each other (same rank/file, clear path)."""
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)

        if f1 == f2:
            # Same file
            low, high = min(r1, r2), max(r1, r2)
            for r in range(low + 1, high):
                if board.piece_at(chess.square(f1, r)) is not None:
                    return False
            return True
        elif r1 == r2:
            # Same rank
            low, high = min(f1, f2), max(f1, f2)
            for f in range(low + 1, high):
                if board.piece_at(chess.square(f, r1)) is not None:
                    return False
            return True
        return False

    def _pawn_structure_features(self, board: chess.Board) -> List[float]:
        """6 floats: doubled (2), isolated (2), passed (2) pawns."""
        features = []
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            pawn_files = [chess.square_file(sq) for sq in pawns]

            # Doubled: multiple pawns on same file
            doubled = sum(1 for f in range(8) if pawn_files.count(f) > 1)
            features.append(min(doubled / 8.0, 1.0))

        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            pawn_files = set(chess.square_file(sq) for sq in pawns)

            # Isolated: no friendly pawns on adjacent files
            isolated = 0
            for f in pawn_files:
                has_neighbor = False
                if f > 0 and (f - 1) in pawn_files:
                    has_neighbor = True
                if f < 7 and (f + 1) in pawn_files:
                    has_neighbor = True
                if not has_neighbor:
                    isolated += 1
            features.append(min(isolated / 8.0, 1.0))

        for color in [chess.WHITE, chess.BLACK]:
            passed = self._count_passed_pawns(board, color)
            features.append(min(passed / 8.0, 1.0))
        return features

    def _count_passed_pawns(self, board: chess.Board,
                            color: chess.Color) -> int:
        """Count passed pawns (no opponent pawns blocking or on adjacent files ahead)."""
        opp_color = not color
        opp_pawns = board.pieces(chess.PAWN, opp_color)
        opp_pawn_positions = [(chess.square_file(sq), chess.square_rank(sq))
                              for sq in opp_pawns]

        passed = 0
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            is_passed = True
            for of, orank in opp_pawn_positions:
                if abs(of - f) <= 1:
                    if color == chess.WHITE and orank > r:
                        is_passed = False
                        break
                    elif color == chess.BLACK and orank < r:
                        is_passed = False
                        break
            if is_passed:
                passed += 1
        return passed

    def _game_phase_feature(self, board: chess.Board) -> List[float]:
        """1 float: 0.0=opening → 1.0=endgame."""
        piece_values = {
            chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9,
        }
        total = 0
        for pt, val in piece_values.items():
            total += len(board.pieces(pt, chess.WHITE)) * val
            total += len(board.pieces(pt, chess.BLACK)) * val
        max_material = 62.0  # 2*(3+3+3+3+5+5+9) = 62
        phase = 1.0 - min(total / max_material, 1.0)
        return [phase]

    def _mobility_features(self, board: chess.Board) -> List[float]:
        """2 floats: legal move count for current side and opponent."""
        current_moves = len(list(board.legal_moves))

        # Estimate opponent mobility by temporarily switching sides
        board_copy = board.copy()
        board_copy.turn = not board.turn
        # Clear en passant to avoid invalid state
        board_copy.ep_square = None
        opp_moves = len(list(board_copy.legal_moves))

        if board.turn == chess.WHITE:
            return [
                min(current_moves / 100.0, 1.0),
                min(opp_moves / 100.0, 1.0),
            ]
        else:
            return [
                min(opp_moves / 100.0, 1.0),
                min(current_moves / 100.0, 1.0),
            ]

    def _space_features(self, board: chess.Board) -> List[float]:
        """2 floats: pieces in opponent's half per side."""
        white_space = 0
        black_space = 0

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            rank = chess.square_rank(sq)
            if piece.color == chess.WHITE and rank >= 4:  # ranks 4-7
                white_space += 1
            elif piece.color == chess.BLACK and rank <= 3:  # ranks 0-3
                black_space += 1

        return [
            min(white_space / 16.0, 1.0),
            min(black_space / 16.0, 1.0),
        ]

    def _material_balance_features(self, board: chess.Board) -> List[float]:
        """3 floats: white total, black total, balance."""
        white_mat = sum(
            len(board.pieces(pt, chess.WHITE)) * val
            for pt, val in PIECE_VALUES.items()
        )
        black_mat = sum(
            len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in PIECE_VALUES.items()
        )
        max_material = 39.0  # Q=9 + 2R=10 + 2B=6 + 2N=6 + 8P=8 = 39
        balance = (white_mat - black_mat + max_material) / (2 * max_material)
        return [
            min(white_mat / max_material, 1.0),
            min(black_mat / max_material, 1.0),
            max(0.0, min(balance, 1.0)),  # centered at 0.5
        ]

    def _threat_features(self, board: chess.Board) -> List[float]:
        """4 floats: in check (2), pieces under attack (2)."""
        features = []

        # Check status
        if board.turn == chess.WHITE:
            features.append(1.0 if board.is_check() else 0.0)
            # For black check, temporarily switch
            board_copy = board.copy()
            board_copy.turn = chess.BLACK
            board_copy.ep_square = None
            features.append(1.0 if board_copy.is_check() else 0.0)
        else:
            board_copy = board.copy()
            board_copy.turn = chess.WHITE
            board_copy.ep_square = None
            features.append(1.0 if board_copy.is_check() else 0.0)
            features.append(1.0 if board.is_check() else 0.0)

        # Pieces under attack
        white_attacked = 0
        black_attacked = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            if piece.color == chess.WHITE:
                if board.is_attacked_by(chess.BLACK, sq):
                    white_attacked += 1
            else:
                if board.is_attacked_by(chess.WHITE, sq):
                    black_attacked += 1

        features.append(min(white_attacked / 16.0, 1.0))
        features.append(min(black_attacked / 16.0, 1.0))
        return features
