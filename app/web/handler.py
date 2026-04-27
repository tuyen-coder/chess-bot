import json
from http import cookies
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from app.web import db as web_db
from app.web.config import (
    HOST,
    MAX_JSON_BYTES,
    MODE_LABELS,
    PORT,
    SECURITY_HEADERS,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
    STATIC_DIR,
    TEMPLATE_DIR,
)
from app.web.sessions import (
    MATCH_LOCK,
    MATCH_QUEUE,
    SESSIONS,
    SESSIONS_LOCK,
    create_session,
    find_user_match,
    get_active_match,
    history_detail_payload,
    join_matchmaking,
    leave_matchmaking,
    rotate_session,
    waiting_state,
)
from app.web.utils import RateLimiter

AUTH_RATE_LIMITER = RateLimiter(limit=10, window_seconds=5 * 60)
PASSWORD_RATE_LIMITER = RateLimiter(limit=8, window_seconds=10 * 60)

STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/css/styles.css": ("css/styles.css", "text/css; charset=utf-8"),
    "/static/js/app.js": ("js/app.js", "application/javascript; charset=utf-8"),
}


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

        if path in STATIC_ROUTES:
            filename, content_type = STATIC_ROUTES[path]
            return self.serve_static(filename, content_type)
        if path == "/api/state":
            return self.route_state(session)
        if path == "/api/leaderboard":
            return self.send_json({"leaderboard": web_db.get_leaderboard()})

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

        public_handler = {
            "/api/register": self.route_register,
            "/api/login": self.route_login,
            "/api/logout": self.route_logout,
        }.get(path)
        if public_handler is not None:
            return public_handler(payload, session_token, session)

        if session.user_id is None:
            return self.send_json({"error": "Please log in first."}, status=HTTPStatus.UNAUTHORIZED)

        protected_handler = {
            "/api/profile/username": self.route_profile_username,
            "/api/profile/password": self.route_profile_password,
            "/api/history/list": self.route_history_list,
            "/api/history/detail": self.route_history_detail,
            "/api/start": self.route_start,
            "/api/menu": self.route_menu,
            "/api/matchmaking/join": self.route_matchmaking_join,
            "/api/matchmaking/leave": self.route_matchmaking_leave,
            "/api/matchmaking/state": self.route_matchmaking_state,
            "/api/matchmaking/resign": self.route_matchmaking_action,
            "/api/matchmaking/draw-offer": self.route_matchmaking_action,
            "/api/matchmaking/draw-respond": self.route_matchmaking_action,
            "/api/matchmaking/rematch": self.route_matchmaking_action,
            "/api/click": self.route_click,
            "/api/matchmaking/click": self.route_matchmaking_click,
            "/api/ai-step": self.route_ai_step,
        }.get(path)
        if protected_handler is not None:
            return protected_handler(path, payload, session)

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def route_state(self, session):
        if session.user_id is not None:
            match = get_active_match(session.online_match_id, session.user_id) or find_user_match(session.user_id)
            if match:
                session.online_match_id = match.id
                with match.lock:
                    return self.send_json(match.state(session.user_id))
        return self.send_json(session.controller.state(session.user_id))

    def route_register(self, payload, session_token, session):
        if self.limited(AUTH_RATE_LIMITER, "register"):
            return self.send_json({"error": "Too many registration attempts. Please wait and try again."}, status=HTTPStatus.TOO_MANY_REQUESTS)
        try:
            user = web_db.create_user(payload.get("username", "").strip(), payload.get("password", ""))
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        self.rotate_authenticated_session(session_token, session, user_id=user["id"])
        return self.send_json(session.controller.state(session.user_id))

    def route_login(self, payload, session_token, session):
        if self.limited(AUTH_RATE_LIMITER, "login"):
            return self.send_json({"error": "Too many login attempts. Please wait and try again."}, status=HTTPStatus.TOO_MANY_REQUESTS)
        try:
            user = web_db.verify_user(payload.get("username", "").strip(), payload.get("password", ""))
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        self.rotate_authenticated_session(session_token, session, user_id=user["id"])
        return self.send_json(session.controller.state(session.user_id))

    def route_logout(self, payload, session_token, session):
        leave_matchmaking(session.user_id)
        self.destroy_session(session_token)
        new_token, session = create_session()
        self.pending_session_token = new_token
        return self.send_json(session.controller.state())

    def route_profile_username(self, path, payload, session):
        try:
            web_db.update_username(session.user_id, payload.get("username", ""))
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        return self.send_json(session.controller.state(session.user_id))

    def route_profile_password(self, path, payload, session):
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

    def route_history_list(self, path, payload, session):
        return self.send_json({"history": web_db.get_game_history(session.user_id)})

    def route_history_detail(self, path, payload, session):
        history_id = payload.get("id")
        if not isinstance(history_id, int):
            return self.send_json({"error": "Invalid game history id."}, status=HTTPStatus.BAD_REQUEST)
        record = history_detail_payload(session.user_id, history_id)
        if record is None:
            return self.send_json({"error": "Game not found."}, status=HTTPStatus.NOT_FOUND)
        return self.send_json({"game": record})

    def route_start(self, path, payload, session):
        mode = payload.get("mode")
        if mode not in MODE_LABELS:
            return self.send_json({"error": "Invalid mode"}, status=HTTPStatus.BAD_REQUEST)
        leave_matchmaking(session.user_id)
        session.online_match_id = None
        session.controller.start_game(mode)
        session.controller.maybe_record_result(session.user_id)
        return self.send_json(session.controller.state(session.user_id))

    def route_menu(self, path, payload, session):
        leave_matchmaking(session.user_id)
        session.online_match_id = None
        session.controller.go_to_menu()
        return self.send_json(session.controller.state(session.user_id))

    def route_matchmaking_join(self, path, payload, session):
        match = join_matchmaking(session.user_id)
        if match is None:
            session.online_match_id = None
            return self.send_json(waiting_state(session.user_id))
        session.online_match_id = match.id
        with match.lock:
            return self.send_json(match.state(session.user_id))

    def route_matchmaking_leave(self, path, payload, session):
        leave_matchmaking(session.user_id)
        session.online_match_id = None
        session.controller.go_to_menu()
        return self.send_json(session.controller.state(session.user_id))

    def route_matchmaking_state(self, path, payload, session):
        match = get_active_match(session.online_match_id, session.user_id) or find_user_match(session.user_id)
        if match:
            session.online_match_id = match.id
            with match.lock:
                return self.send_json(match.state(session.user_id))
        with MATCH_LOCK:
            waiting = session.user_id in MATCH_QUEUE
        if waiting:
            return self.send_json(waiting_state(session.user_id))
        return self.send_json(session.controller.state(session.user_id))

    def route_matchmaking_action(self, path, payload, session):
        match = get_active_match(session.online_match_id, session.user_id)
        if match is None:
            return self.send_json({"error": "No active online match."}, status=HTTPStatus.BAD_REQUEST)
        with match.lock:
            if path == "/api/matchmaking/resign":
                match.resign(session.user_id)
            elif path == "/api/matchmaking/draw-offer":
                match.offer_draw(session.user_id)
            elif path == "/api/matchmaking/draw-respond":
                match.respond_draw(session.user_id, bool(payload.get("accept")))
            else:
                match.request_rematch(session.user_id)
            return self.send_json(match.state(session.user_id))

    def route_click(self, path, payload, session):
        row, col, error_response = self.valid_square_from_payload(payload)
        if error_response is not None:
            return error_response
        session.controller.click_square(row, col)
        session.controller.maybe_record_result(session.user_id)
        return self.send_json(session.controller.state(session.user_id))

    def route_matchmaking_click(self, path, payload, session):
        match = get_active_match(session.online_match_id, session.user_id)
        if match is None:
            return self.send_json({"error": "No active online match."}, status=HTTPStatus.BAD_REQUEST)
        row, col, error_response = self.valid_square_from_payload(payload)
        if error_response is not None:
            return error_response
        with match.lock:
            match.click_square(session.user_id, row, col)
            return self.send_json(match.state(session.user_id))

    def route_ai_step(self, path, payload, session):
        session.controller.ai_step()
        session.controller.maybe_record_result(session.user_id)
        return self.send_json(session.controller.state(session.user_id))

    def valid_square_from_payload(self, payload):
        row = payload.get("row")
        col = payload.get("col")
        if not isinstance(row, int) or not isinstance(col, int):
            return None, None, self.send_json({"error": "Invalid square"}, status=HTTPStatus.BAD_REQUEST)
        if not (0 <= row < 8 and 0 <= col < 8):
            return None, None, self.send_json({"error": "Square is out of range."}, status=HTTPStatus.BAD_REQUEST)
        return row, col, None

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
