import os
from pathlib import Path

import chess

HOST = os.getenv("WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_PORT", "8000"))
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
    "online": "Online Match",
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

BOT_LABELS = {
    "random": "Random AI",
    "minimax": "Minimax AI",
    "ml": "ML AI",
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
}
