import threading
from uuid import uuid4

from app.web import db as web_db
from app.web.controllers import WebChessController
from app.web.online import OnlineMatch
from app.web.utils import board_history_from_moves

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
MATCHES = {}
MATCH_QUEUE = []
MATCH_LOCK = threading.Lock()


class SessionState:
    def __init__(self):
        self.controller = WebChessController()
        self.user_id = None
        self.online_match_id = None


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


def waiting_state(user_id):
    return {
        "inMenu": False,
        "online": True,
        "waitingForOpponent": True,
        "user": web_db.get_user(user_id),
        "stats": web_db.get_user_stats(user_id),
        "leaderboard": web_db.get_leaderboard(),
        "mode": "online",
        "modeLabel": "Online Match",
        "status": "Waiting for another player",
        "gameOver": False,
        "canInteract": False,
        "aiWaiting": False,
    }


def leave_matchmaking(user_id):
    if user_id is None:
        return
    with MATCH_LOCK:
        while user_id in MATCH_QUEUE:
            MATCH_QUEUE.remove(user_id)


def join_matchmaking(user_id):
    with MATCH_LOCK:
        for match in MATCHES.values():
            if user_id in match.players.values() and not match.game_state.is_game_over():
                return match

        while user_id in MATCH_QUEUE:
            MATCH_QUEUE.remove(user_id)

        if MATCH_QUEUE:
            opponent_id = MATCH_QUEUE.pop(0)
            match_id = uuid4().hex
            match = OnlineMatch(match_id, opponent_id, user_id)
            MATCHES[match_id] = match
            return match

        MATCH_QUEUE.append(user_id)
        return None


def get_active_match(match_id, user_id):
    if not match_id:
        return None
    with MATCH_LOCK:
        match = MATCHES.get(match_id)
        if match and user_id in match.players.values():
            return match
    return None


def find_user_match(user_id):
    with MATCH_LOCK:
        for match in MATCHES.values():
            if user_id in match.players.values() and not match.game_state.is_game_over():
                return match
    return None


def history_detail_payload(user_id, history_id):
    record = web_db.get_game_history_detail(user_id, history_id)
    if record is None:
        return None
    boards, checks = board_history_from_moves(record.get("moves", []))
    record["historyBoards"] = boards
    record["historyCheckSquares"] = checks
    return record
