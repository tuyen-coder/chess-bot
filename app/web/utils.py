import math
import threading
import time
from collections import defaultdict, deque

import chess


def describe_piece_code(piece_code):
    if piece_code == "--":
        return "empty"

    color = "White" if piece_code[0] == "w" else "Black"
    names = {
        "p": "pawn",
        "R": "rook",
        "N": "knight",
        "B": "bishop",
        "Q": "queen",
        "K": "king",
    }
    return f"{color} {names.get(piece_code[1], 'piece')}"


def board_to_array(board):
    arr = [["--" for _ in range(8)] for _ in range(8)]
    for row in range(8):
        rank = 7 - row
        for col in range(8):
            square = chess.square(col, rank)
            piece = board.piece_at(square)
            if piece is None:
                continue
            color = "w" if piece.color == chess.WHITE else "b"
            symbol = piece.symbol().upper()
            arr[row][col] = f"{color}p" if symbol == "P" else f"{color}{symbol}"
    return arr


def check_square_for_board(board):
    if not board.is_check():
        return None
    square = board.king(board.turn)
    if square is None:
        return None
    return [7 - chess.square_rank(square), chess.square_file(square)]


def board_history_from_moves(moves):
    board = chess.Board()
    boards = [board_to_array(board)]
    checks = [check_square_for_board(board)]
    for move_text in moves:
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
        boards.append(board_to_array(board))
        checks.append(check_square_for_board(board))
    return boards, checks


def terminal_result_key(board):
    if board.is_checkmate():
        return "black" if board.turn == chess.WHITE else "white"
    return "draw"


class RateLimiter:
    def __init__(self, limit, window_seconds):
        self.limit = limit
        self.window_seconds = window_seconds
        self.events = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.time()
        cutoff = now - self.window_seconds
        with self.lock:
            queue = self.events[key]
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= self.limit:
                return False
            queue.append(now)
            return True
