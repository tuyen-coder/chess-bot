import threading
import time

import chess

from app.engine import chess_engine
from app.web import db as web_db
from app.web.config import AI_DELAY_MS
from app.web.utils import board_history_from_moves, describe_piece_code, terminal_result_key


class OnlineMatch:
    def __init__(self, match_id, white_user_id, black_user_id):
        self.id = match_id
        self.game_state = chess_engine.GameState()
        self.players = {
            chess.WHITE: white_user_id,
            chess.BLACK: black_user_id,
        }
        self.selected = {
            white_user_id: None,
            black_user_id: None,
        }
        self.legal_moves = {
            white_user_id: [],
            black_user_id: [],
        }
        self.result_recorded = False
        self.result_override = None
        self.result_label = None
        self.termination = "normal"
        self.history_id = None
        self.elo_change = None
        self.draw_offered_by = None
        self.rematch_requests = set()
        self.created_at = time.time()
        self.lock = threading.Lock()

    def user_color(self, user_id):
        if self.players[chess.WHITE] == user_id:
            return chess.WHITE
        if self.players[chess.BLACK] == user_id:
            return chess.BLACK
        return None

    def opponent_id(self, user_id):
        color = self.user_color(user_id)
        if color is None:
            return None
        return self.players[not color]

    def color_name(self, color):
        return "White" if color == chess.WHITE else "Black"

    def can_interact(self, user_id):
        color = self.user_color(user_id)
        return (
            color is not None
            and not self.is_over()
            and self.game_state.board.turn == color
        )

    def selection_payload(self, user_id):
        selected = self.selected.get(user_id)
        legal_moves = self.legal_moves.get(user_id, [])
        if selected is None:
            return {
                "square": "No piece selected",
                "piece": "",
                "legalMovesText": "Click one of your pieces to preview legal moves.",
            }

        piece_code = self.game_state.board_to_array()[selected[0]][selected[1]]
        moves = [self.game_state.rc_to_algebraic(move) for move in legal_moves]
        return {
            "square": self.game_state.rc_to_algebraic(selected),
            "piece": describe_piece_code(piece_code),
            "legalMovesText": ", ".join(moves) if moves else "No legal moves available from this square.",
        }

    def describe_status(self):
        if self.result_label:
            return self.result_label
        if self.game_state.is_checkmate():
            return self.game_state.game_result()
        if self.game_state.is_stalemate():
            return "Stalemate"
        if self.game_state.is_check():
            return "Check"
        return "In progress"

    def is_over(self):
        return self.result_override is not None or self.game_state.is_game_over()

    def result_key(self):
        if self.result_override is not None:
            return self.result_override
        if not self.game_state.is_game_over():
            return None
        return terminal_result_key(self.game_state.board)

    def default_result_label(self, result):
        white = web_db.get_user(self.players[chess.WHITE])
        black = web_db.get_user(self.players[chess.BLACK])
        white_name = white["username"] if white else "White"
        black_name = black["username"] if black else "Black"
        if result == "white":
            return f"{white_name} won"
        if result == "black":
            return f"{black_name} won"
        return "Draw"

    def maybe_record_result(self):
        result = self.result_key()
        if self.result_recorded or result is None:
            return
        self.elo_change = web_db.record_multiplayer_result(
            self.players[chess.WHITE],
            self.players[chess.BLACK],
            result,
        )
        white = web_db.get_user(self.players[chess.WHITE])
        black = web_db.get_user(self.players[chess.BLACK])
        white_name = white["username"] if white else "White"
        black_name = black["username"] if black else "Black"
        self.result_label = self.result_label or self.default_result_label(result)
        self.history_id = web_db.save_game_history(
            mode="online",
            white_user_id=self.players[chess.WHITE],
            black_user_id=self.players[chess.BLACK],
            white_username=white_name,
            black_username=black_name,
            result=result,
            result_label=self.result_label,
            termination=self.termination,
            moves=self.game_state.moveLog,
            elo_change=self.elo_change,
        )
        self.result_recorded = True

    def finish_by_action(self, result, label, termination):
        if self.is_over():
            return
        self.result_override = result
        self.result_label = label
        self.termination = termination
        for player_id in self.selected:
            self.selected[player_id] = None
            self.legal_moves[player_id] = []
        self.maybe_record_result()

    def resign(self, user_id):
        color = self.user_color(user_id)
        if color is None or self.is_over():
            return
        winner = "black" if color == chess.WHITE else "white"
        loser = web_db.get_user(user_id)
        loser_name = loser["username"] if loser else self.color_name(color)
        self.finish_by_action(winner, f"{loser_name} resigned", "resignation")

    def offer_draw(self, user_id):
        if self.user_color(user_id) is not None and not self.is_over():
            self.draw_offered_by = user_id

    def respond_draw(self, user_id, accept):
        if self.draw_offered_by is None or self.draw_offered_by == user_id or self.is_over():
            return
        if accept:
            self.finish_by_action("draw", "Draw agreed", "draw_agreement")
        else:
            self.draw_offered_by = None

    def request_rematch(self, user_id):
        if self.user_color(user_id) is None or not self.is_over():
            return False
        self.rematch_requests.add(user_id)
        if set(self.players.values()).issubset(self.rematch_requests):
            self.reset_for_rematch()
            return True
        return False

    def reset_for_rematch(self):
        self.game_state = chess_engine.GameState()
        self.selected = {player_id: None for player_id in self.selected}
        self.legal_moves = {player_id: [] for player_id in self.legal_moves}
        self.result_recorded = False
        self.result_override = None
        self.result_label = None
        self.termination = "normal"
        self.history_id = None
        self.elo_change = None
        self.draw_offered_by = None
        self.rematch_requests = set()

    def piece_belongs_to_user(self, row, col, user_id):
        color = self.user_color(user_id)
        if color is None:
            return False
        square = chess.square(col, 7 - row)
        piece = self.game_state.board.piece_at(square)
        return piece is not None and piece.color == color

    def click_square(self, user_id, row, col):
        if not self.can_interact(user_id):
            return

        selected = self.selected[user_id]
        if selected is None:
            if self.piece_belongs_to_user(row, col, user_id):
                moves = self.game_state.legal_moves_from(row, col)
                self.selected[user_id] = (row, col) if moves else None
                self.legal_moves[user_id] = moves
            return

        moved = self.game_state.make_move(selected, (row, col))
        if moved:
            for player_id in self.selected:
                self.selected[player_id] = None
                self.legal_moves[player_id] = []
            self.draw_offered_by = None
            self.maybe_record_result()
            return

        if self.piece_belongs_to_user(row, col, user_id):
            moves = self.game_state.legal_moves_from(row, col)
            self.selected[user_id] = (row, col) if moves else None
            self.legal_moves[user_id] = moves
        else:
            self.selected[user_id] = None
            self.legal_moves[user_id] = []

    def state(self, user_id):
        color = self.user_color(user_id)
        opponent = web_db.get_user(self.opponent_id(user_id))
        user = web_db.get_user(user_id)
        check_square = self.game_state.king_in_check_rc()
        selection = self.selection_payload(user_id)
        self.maybe_record_result()
        history_boards, history_checks = board_history_from_moves(self.game_state.moveLog)
        draw_offer = None
        if self.draw_offered_by is not None:
            draw_offer_user = web_db.get_user(self.draw_offered_by)
            draw_offer = {
                "fromUserId": self.draw_offered_by,
                "fromUsername": draw_offer_user["username"] if draw_offer_user else "Opponent",
                "canRespond": self.draw_offered_by != user_id,
            }
        return {
            "inMenu": False,
            "online": True,
            "matchId": self.id,
            "user": user,
            "opponent": opponent,
            "stats": web_db.get_user_stats(user_id),
            "leaderboard": web_db.get_leaderboard(),
            "mode": "online",
            "modeLabel": "Online Match",
            "playerColor": self.color_name(color) if color is not None else None,
            "board": self.game_state.board_to_array(),
            "selected": list(self.selected.get(user_id)) if self.selected.get(user_id) else None,
            "legalMoves": [list(move) for move in self.legal_moves.get(user_id, [])],
            "checkSquare": list(check_square) if check_square else None,
            "turn": "White" if self.game_state.board.turn == chess.WHITE else "Black",
            "status": self.describe_status(),
            "lastMove": self.game_state.moveLog[-1] if self.game_state.moveLog else "No moves yet",
            "moveCount": len(self.game_state.moveLog),
            "moveLog": self.game_state.moveLog,
            "historyBoards": history_boards,
            "historyCheckSquares": history_checks,
            "historyId": self.history_id,
            "selection": selection,
            "selectionDisplay": f"{selection['square']} ({selection['piece']})" if selection["piece"] else selection["square"],
            "gameOver": self.is_over(),
            "gameResult": self.describe_status() if self.is_over() else None,
            "canInteract": self.can_interact(user_id),
            "aiWaiting": False,
            "aiDelayMs": AI_DELAY_MS,
            "eloChange": self.elo_change,
            "drawOffer": draw_offer,
            "rematchRequested": user_id in self.rematch_requests,
            "opponentRematchRequested": self.opponent_id(user_id) in self.rematch_requests,
        }
