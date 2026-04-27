import chess

from app.engine import chess_engine
from app.engine import minimax as solver
from app.ml import model as ml_solver
from app.web import db as web_db
from app.web.config import (
    AI_DELAY_MS,
    BOT_LABELS,
    CHESS_TURN,
    MINIMAX_DEPTH,
    MODE_LABELS,
    PLAYER_MODES,
)
from app.web.utils import board_history_from_moves, describe_piece_code


class WebChessController:
    def __init__(self):
        self.ml_model = None
        self.go_to_menu()

    def go_to_menu(self):
        self.mode = None
        self.game_state = chess_engine.GameState()
        self.selected = None
        self.legal_moves = []
        self.result_recorded = False
        self.history_id = None

    def start_game(self, mode):
        self.mode = mode
        self.game_state = chess_engine.GameState()
        self.selected = None
        self.legal_moves = []
        self.result_recorded = False
        self.history_id = None

        if mode in ["pvl", "mlr"] and self.ml_model is None:
            print("Loading or training ML model...", flush=True)
            self.ml_model = ml_solver.train_model()

    def player_can_interact(self):
        if self.mode is None or self.game_state.is_game_over():
            return False

        turn = self.game_state.board.turn == CHESS_TURN

        if self.mode in ["mvm", "rvm"]:
            return False
        if self.mode == "pvr" and not turn:
            return False
        if self.mode == "pvm" and not turn:
            return False
        if self.mode == "pvl" and not turn:
            return False

        return True

    def current_ai_type(self):
        if self.mode is None or self.game_state.is_game_over():
            return None

        turn = self.game_state.board.turn == CHESS_TURN

        if self.mode == "pvr":
            return "random" if not turn else None
        if self.mode == "pvm":
            return "minimax" if not turn else None
        if self.mode == "rvm":
            return "random" if turn else "minimax"
        if self.mode == "mvm":
            return "minimax"
        if self.mode == "pvl":
            return "ml" if not turn else None
        if self.mode == "mlr":
            return "ml" if turn else "random"
        return None

    def click_square(self, row, col):
        if not self.player_can_interact():
            return

        if self.selected is None:
            moves = self.game_state.legal_moves_from(row, col)
            if moves:
                self.selected = (row, col)
                self.legal_moves = moves
            else:
                self.selected = None
                self.legal_moves = []
            return

        moved = self.game_state.make_move(self.selected, (row, col))
        if moved:
            self.selected = None
            self.legal_moves = []
            return

        moves = self.game_state.legal_moves_from(row, col)
        if moves:
            self.selected = (row, col)
            self.legal_moves = moves
        else:
            self.selected = None
            self.legal_moves = []

    def ai_step(self):
        ai_type = self.current_ai_type()
        if ai_type is None:
            return False

        if ai_type == "random":
            move = solver.random_move(self.game_state.board)
        elif ai_type == "minimax":
            move = solver.find_best_move(self.game_state.board, MINIMAX_DEPTH)
        else:
            move = ml_solver.ml_move(self.game_state.board, self.ml_model)

        if move:
            self.game_state.apply_move_obj(move)
            self.selected = None
            self.legal_moves = []
            return True

        return False

    def maybe_record_result(self, user_id):
        if self.result_recorded or user_id is None:
            return
        if self.mode not in PLAYER_MODES or not self.game_state.is_game_over():
            return

        if self.game_state.board.is_checkmate():
            result = "win" if not self.game_state.board.turn else "loss"
        else:
            result = "draw"

        web_db.record_result(user_id, PLAYER_MODES[self.mode], result)
        self.record_history(user_id, result)
        self.result_recorded = True

    def record_history(self, user_id, result):
        user = web_db.get_user(user_id)
        if user is None:
            return
        opponent = PLAYER_MODES.get(self.mode)
        if opponent is None:
            return
        if result == "win":
            result_key = "white"
            result_label = f"{user['username']} won"
        elif result == "loss":
            result_key = "black"
            result_label = f"{BOT_LABELS[opponent]} won"
        else:
            result_key = "draw"
            result_label = "Draw"
        self.history_id = web_db.save_game_history(
            mode=self.mode,
            white_user_id=user_id,
            black_user_id=None,
            white_username=user["username"],
            black_username=BOT_LABELS[opponent],
            result=result_key,
            result_label=result_label,
            termination="normal",
            moves=self.game_state.moveLog,
        )

    def selection_payload(self):
        if self.selected is None:
            return {
                "square": "No piece selected",
                "piece": "",
                "legalMovesText": "Click a piece to preview its legal moves.",
            }

        piece_code = self.game_state.board_to_array()[self.selected[0]][self.selected[1]]
        legal_moves = [self.game_state.rc_to_algebraic(move) for move in self.legal_moves]
        return {
            "square": self.game_state.rc_to_algebraic(self.selected),
            "piece": describe_piece_code(piece_code),
            "legalMovesText": ", ".join(legal_moves) if legal_moves else "No legal moves available from this square.",
        }

    def describe_status(self):
        if self.mode is None:
            return "Select a game mode to begin."
        if self.game_state.is_checkmate():
            return self.game_state.game_result()
        if self.game_state.is_stalemate():
            return "Stalemate"
        if self.game_state.is_check():
            return "Check"
        return "In progress"

    def state(self, user_id=None):
        check_square = self.game_state.king_in_check_rc()
        selection = self.selection_payload()
        user = web_db.get_user(user_id) if user_id is not None else None
        stats = web_db.get_user_stats(user_id) if user_id is not None else None
        history_boards, history_checks = board_history_from_moves(self.game_state.moveLog)
        return {
            "inMenu": self.mode is None,
            "user": user,
            "stats": stats,
            "leaderboard": web_db.get_leaderboard(),
            "mode": self.mode,
            "modeLabel": MODE_LABELS.get(self.mode, "Not selected"),
            "board": self.game_state.board_to_array(),
            "selected": list(self.selected) if self.selected else None,
            "legalMoves": [list(move) for move in self.legal_moves],
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
            "gameOver": self.game_state.is_game_over(),
            "gameResult": self.game_state.game_result() if self.game_state.is_game_over() else None,
            "canInteract": self.player_can_interact(),
            "aiWaiting": self.current_ai_type() is not None,
            "aiDelayMs": AI_DELAY_MS,
        }
