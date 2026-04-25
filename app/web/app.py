import json
import threading
import time
from collections import defaultdict, deque
from http import cookies
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import chess

from app.engine import chess_engine
from app.engine import minimax as solver
from app.ml import model as ml_solver
from app.web import db as web_db

HOST = "127.0.0.1"
PORT = 8000
AI_DELAY_MS = 800
MINIMAX_DEPTH = 3
CHESS_TURN = chess.WHITE
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
MAX_JSON_BYTES = 8 * 1024
SESSION_COOKIE_NAME = "chess_session"
SESSION_MAX_AGE = 60 * 60 * 12

MODE_LABELS = {
    "pvp": "Player vs Player",
    "pvr": "Player vs Random AI",
    "pvm": "Player vs Minimax AI",
    "rvm": "Random AI vs Minimax AI",
    "mvm": "Minimax AI vs Minimax AI",
    "pvl": "Player vs ML AI",
    "mlr": "ML AI vs Random AI",
}

PLAYER_MODES = {
    "pvr": "random",
    "pvm": "minimax",
    "pvl": "ml",
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
}


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


AUTH_RATE_LIMITER = RateLimiter(limit=10, window_seconds=5 * 60)
PASSWORD_RATE_LIMITER = RateLimiter(limit=8, window_seconds=10 * 60)


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

    def start_game(self, mode):
        self.mode = mode
        self.game_state = chess_engine.GameState()
        self.selected = None
        self.legal_moves = []
        self.result_recorded = False

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
        self.result_recorded = True

    def describe_piece(self, piece_code):
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
            "piece": self.describe_piece(piece_code),
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
        return {
            "inMenu": self.mode is None,
            "user": user,
            "stats": stats,
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
            "selection": selection,
            "selectionDisplay": f"{selection['square']} ({selection['piece']})" if selection["piece"] else selection["square"],
            "gameOver": self.game_state.is_game_over(),
            "gameResult": self.game_state.game_result() if self.game_state.is_game_over() else None,
            "canInteract": self.player_can_interact(),
            "aiWaiting": self.current_ai_type() is not None,
            "aiDelayMs": AI_DELAY_MS,
        }

web_db.init_db()
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


class SessionState:
    def __init__(self):
        self.controller = WebChessController()
        self.user_id = None


def create_session():
    token = uuid4().hex
    with SESSIONS_LOCK:
        SESSIONS[token] = SessionState()
    return token, SESSIONS[token]


def rotate_session(previous_token=None, session=None):
    token = uuid4().hex
    with SESSIONS_LOCK:
        if previous_token:
            SESSIONS.pop(previous_token, None)
        if session is None:
            session = SessionState()
        SESSIONS[token] = session
    return token, session


class ChessRequestHandler(BaseHTTPRequestHandler):
    server_version = "ChessWeb/1.0"

    def get_or_create_session(self):
        self.pending_session_token = None
        self.clear_session_cookie = False
        session_token = None

        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            try:
                jar = cookies.SimpleCookie()
                jar.load(cookie_header)
                morsel = jar.get(SESSION_COOKIE_NAME)
                session_token = morsel.value if morsel else None
            except cookies.CookieError:
                session_token = None

        with SESSIONS_LOCK:
            if session_token and session_token in SESSIONS:
                return session_token, SESSIONS[session_token]

        token, session = create_session()
        self.pending_session_token = token
        return token, session

    def send_security_headers(self):
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)

    def send_no_store_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")

    def request_origin_allowed(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        allowed = {
            f"http://{self.headers.get('Host')}",
            f"http://{HOST}:{PORT}",
            f"http://localhost:{PORT}",
            f"http://127.0.0.1:{PORT}",
        }
        return origin in allowed

    def rotate_authenticated_session(self, current_token, session, user_id=None):
        if user_id is not None:
            session.user_id = user_id
        new_token, _ = rotate_session(previous_token=current_token, session=session)
        self.pending_session_token = new_token

    def destroy_session(self, current_token):
        with SESSIONS_LOCK:
            SESSIONS.pop(current_token, None)
        self.clear_session_cookie = True

    def limited(self, limiter, bucket):
        client_ip = self.client_address[0] if self.client_address else "unknown"
        return not limiter.allow(f"{bucket}:{client_ip}")

    def do_GET(self):
        path = urlparse(self.path).path
        _, session = self.get_or_create_session()

        if path in ["/", "/index.html"]:
            return self.serve_static("index.html", "text/html; charset=utf-8")
        if path == "/static/css/styles.css":
            return self.serve_static("css/styles.css", "text/css; charset=utf-8")
        if path == "/static/js/app.js":
            return self.serve_static("js/app.js", "application/javascript; charset=utf-8")
        if path == "/api/state":
            return self.send_json(session.controller.state(session.user_id))

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if not self.request_origin_allowed():
            return self.send_json({"error": "Invalid request origin."}, status=HTTPStatus.FORBIDDEN)

        path = urlparse(self.path).path
        try:
            payload = self.read_json()
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        session_token, session = self.get_or_create_session()

        if path == "/api/register":
            if self.limited(AUTH_RATE_LIMITER, "register"):
                return self.send_json({"error": "Too many registration attempts. Please wait and try again."}, status=HTTPStatus.TOO_MANY_REQUESTS)
            try:
                user = web_db.create_user(payload.get("username", "").strip(), payload.get("password", ""))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            self.rotate_authenticated_session(session_token, session, user_id=user["id"])
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/login":
            if self.limited(AUTH_RATE_LIMITER, "login"):
                return self.send_json({"error": "Too many login attempts. Please wait and try again."}, status=HTTPStatus.TOO_MANY_REQUESTS)
            try:
                user = web_db.verify_user(payload.get("username", "").strip(), payload.get("password", ""))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            self.rotate_authenticated_session(session_token, session, user_id=user["id"])
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/logout":
            self.destroy_session(session_token)
            new_token, session = create_session()
            self.pending_session_token = new_token
            return self.send_json(session.controller.state())

        if path == "/api/profile/username":
            if session.user_id is None:
                return self.send_json({"error": "Please log in first."}, status=HTTPStatus.UNAUTHORIZED)
            try:
                web_db.update_username(session.user_id, payload.get("username", ""))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/profile/password":
            if session.user_id is None:
                return self.send_json({"error": "Please log in first."}, status=HTTPStatus.UNAUTHORIZED)
            if self.limited(PASSWORD_RATE_LIMITER, "profile-password"):
                return self.send_json({"error": "Too many password change attempts. Please wait and try again."}, status=HTTPStatus.TOO_MANY_REQUESTS)
            try:
                web_db.update_password(
                    session.user_id,
                    payload.get("currentPassword", ""),
                    payload.get("newPassword", ""),
                )
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self.send_json(session.controller.state(session.user_id))

        if session.user_id is None:
            return self.send_json({"error": "Please log in first."}, status=HTTPStatus.UNAUTHORIZED)

        if path == "/api/start":
            mode = payload.get("mode")
            if mode not in MODE_LABELS:
                return self.send_json({"error": "Invalid mode"}, status=HTTPStatus.BAD_REQUEST)
            session.controller.start_game(mode)
            session.controller.maybe_record_result(session.user_id)
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/menu":
            session.controller.go_to_menu()
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/click":
            row = payload.get("row")
            col = payload.get("col")
            if not isinstance(row, int) or not isinstance(col, int):
                return self.send_json({"error": "Invalid square"}, status=HTTPStatus.BAD_REQUEST)
            if not (0 <= row < 8 and 0 <= col < 8):
                return self.send_json({"error": "Square is out of range."}, status=HTTPStatus.BAD_REQUEST)
            session.controller.click_square(row, col)
            session.controller.maybe_record_result(session.user_id)
            return self.send_json(session.controller.state(session.user_id))

        if path == "/api/ai-step":
            session.controller.ai_step()
            session.controller.maybe_record_result(session.user_id)
            return self.send_json(session.controller.state(session.user_id))

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def read_json(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError("Content-Type must be application/json.")

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_JSON_BYTES:
            raise ValueError("Request body is too large.")
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON body.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def serve_static(self, filename, content_type):
        if filename == "index.html":
            file_path = TEMPLATE_DIR / filename
        else:
            file_path = STATIC_DIR / filename
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if filename == "index.html":
            self.send_no_store_headers()
        self.maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_security_headers()
        self.send_no_store_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(body)

    def maybe_set_session_cookie(self):
        if getattr(self, "clear_session_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0",
            )
        if getattr(self, "pending_session_token", None):
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}={self.pending_session_token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_MAX_AGE}",
            )

    def log_message(self, format_str, *args):
        print(f"{self.address_string()} - {format_str % args}", flush=True)


def run():
    server = ThreadingHTTPServer((HOST, PORT), ChessRequestHandler)
    print(f"Web UI running at http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
