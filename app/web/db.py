import hashlib
import hmac
import os
import re
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chess_web.db"
DB_LOCK = threading.Lock()
OPPONENTS = ["random", "minimax", "ml"]
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,24}$")
MAX_PASSWORD_LENGTH = 128
STARTING_ELO = 1200


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                elo INTEGER NOT NULL DEFAULT {STARTING_ELO},
                multiplayer_wins INTEGER NOT NULL DEFAULT 0,
                multiplayer_losses INTEGER NOT NULL DEFAULT 0,
                multiplayer_draws INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ensure_user_columns(cursor)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER NOT NULL,
                opponent TEXT NOT NULL,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                draws INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, opponent),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        connection.commit()
        connection.close()


def ensure_user_columns(cursor):
    cursor.execute("PRAGMA table_info(users)")
    columns = {row["name"] for row in cursor.fetchall()}
    missing_columns = {
        "elo": f"INTEGER NOT NULL DEFAULT {STARTING_ELO}",
        "multiplayer_wins": "INTEGER NOT NULL DEFAULT 0",
        "multiplayer_losses": "INTEGER NOT NULL DEFAULT 0",
        "multiplayer_draws": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in missing_columns.items():
        if column not in columns:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")


def hash_password(password, salt=None):
    salt_bytes = salt or os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        200000,
    )
    return password_hash.hex(), salt_bytes.hex()


def normalize_username(username):
    if not isinstance(username, str):
        raise ValueError("Username is required.")

    normalized = username.strip()
    if not USERNAME_RE.fullmatch(normalized):
        raise ValueError("Username must be 3-24 characters using letters, numbers, _ or -.")
    return normalized


def validate_password(password):
    if not isinstance(password, str) or not password:
        raise ValueError("Password is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password is too long.")
    return password


def verify_password(password, password_hash, password_salt):
    candidate_hash, _ = hash_password(password, salt=bytes.fromhex(password_salt))
    return hmac.compare_digest(candidate_hash, password_hash)


def create_user(username, password):
    username = normalize_username(username)
    password = validate_password(password)

    password_hash, password_salt = hash_password(password)

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, password_salt)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, password_salt),
            )
            user_id = cursor.lastrowid

            for opponent in OPPONENTS:
                cursor.execute(
                    """
                    INSERT INTO user_stats (user_id, opponent, wins, losses, draws)
                    VALUES (?, ?, 0, 0, 0)
                    """,
                    (user_id, opponent),
                )

            connection.commit()
            return {
                "id": user_id,
                "username": username,
            }
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists.") from exc
        finally:
            connection.close()


def verify_user(username, password):
    username = normalize_username(username)
    validate_password(password)

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, username, password_hash, password_salt
            FROM users
            WHERE username = ?
            """,
            (username,),
        )
        row = cursor.fetchone()
        connection.close()

    if row is None:
        raise ValueError("Invalid username or password.")

    if not verify_password(password, row["password_hash"], row["password_salt"]):
        raise ValueError("Invalid username or password.")

    return {"id": row["id"], "username": row["username"]}


def get_user_auth(user_id):
    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                id,
                username,
                password_hash,
                password_salt,
                elo,
                multiplayer_wins,
                multiplayer_losses,
                multiplayer_draws,
                created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        connection.close()

    return row


def get_user(user_id):
    row = get_user_auth(user_id)

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "elo": row["elo"],
        "multiplayer_wins": row["multiplayer_wins"],
        "multiplayer_losses": row["multiplayer_losses"],
        "multiplayer_draws": row["multiplayer_draws"],
        "created_at": row["created_at"],
    }


def update_username(user_id, new_username):
    new_username = normalize_username(new_username)

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                UPDATE users
                SET username = ?
                WHERE id = ?
                """,
                (new_username, user_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("User not found.")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists.") from exc
        finally:
            connection.close()

    return get_user(user_id)


def update_password(user_id, current_password, new_password):
    validate_password(current_password)
    validate_password(new_password)

    row = get_user_auth(user_id)
    if row is None:
        raise ValueError("User not found.")
    if not verify_password(current_password, row["password_hash"], row["password_salt"]):
        raise ValueError("Current password is incorrect.")

    new_hash, new_salt = hash_password(new_password)

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?, password_salt = ?
            WHERE id = ?
            """,
            (new_hash, new_salt, user_id),
        )
        connection.commit()
        connection.close()


def get_user_stats(user_id):
    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT opponent, wins, losses, draws
            FROM user_stats
            WHERE user_id = ?
            ORDER BY opponent
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        connection.close()

    stats = {}
    for row in rows:
        total = row["wins"] + row["losses"] + row["draws"]
        win_rate = (row["wins"] / total) if total else 0.0
        stats[row["opponent"]] = {
            "wins": row["wins"],
            "losses": row["losses"],
            "draws": row["draws"],
            "total": total,
            "winRate": round(win_rate * 100, 1),
        }

    for opponent in OPPONENTS:
        stats.setdefault(
            opponent,
            {"wins": 0, "losses": 0, "draws": 0, "total": 0, "winRate": 0.0},
        )

    return stats


def get_leaderboard(limit=10):
    safe_limit = max(1, min(int(limit), 50))
    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, username, elo, multiplayer_wins, multiplayer_losses, multiplayer_draws
            FROM users
            ORDER BY elo DESC, multiplayer_wins DESC, username ASC
            LIMIT ?
            """,
            (safe_limit,),
        )
        rows = cursor.fetchall()
        connection.close()

    leaderboard = []
    for index, row in enumerate(rows, start=1):
        wins = row["multiplayer_wins"]
        losses = row["multiplayer_losses"]
        draws = row["multiplayer_draws"]
        total = wins + losses + draws
        leaderboard.append(
            {
                "rank": index,
                "id": row["id"],
                "username": row["username"],
                "elo": row["elo"],
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": total,
            }
        )
    return leaderboard


def expected_score(player_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))


def record_multiplayer_result(white_user_id, black_user_id, result):
    if result not in {"white", "black", "draw"} or white_user_id == black_user_id:
        return None

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, elo FROM users WHERE id IN (?, ?)",
            (white_user_id, black_user_id),
        )
        rows = {row["id"]: row for row in cursor.fetchall()}
        if white_user_id not in rows or black_user_id not in rows:
            connection.close()
            return None

        white_elo = rows[white_user_id]["elo"]
        black_elo = rows[black_user_id]["elo"]
        if result == "white":
            white_score = 1.0
            black_score = 0.0
            white_column = "multiplayer_wins"
            black_column = "multiplayer_losses"
        elif result == "black":
            white_score = 0.0
            black_score = 1.0
            white_column = "multiplayer_losses"
            black_column = "multiplayer_wins"
        else:
            white_score = 0.5
            black_score = 0.5
            white_column = "multiplayer_draws"
            black_column = "multiplayer_draws"

        k_factor = 32
        white_new = round(white_elo + k_factor * (white_score - expected_score(white_elo, black_elo)))
        black_new = round(black_elo + k_factor * (black_score - expected_score(black_elo, white_elo)))

        cursor.execute(
            f"UPDATE users SET elo = ?, {white_column} = {white_column} + 1 WHERE id = ?",
            (white_new, white_user_id),
        )
        cursor.execute(
            f"UPDATE users SET elo = ?, {black_column} = {black_column} + 1 WHERE id = ?",
            (black_new, black_user_id),
        )
        connection.commit()
        connection.close()

    return {
        "white": {"old": white_elo, "new": white_new, "change": white_new - white_elo},
        "black": {"old": black_elo, "new": black_new, "change": black_new - black_elo},
    }


def record_result(user_id, opponent, result):
    if opponent not in OPPONENTS:
        return

    if result not in {"win", "loss", "draw"}:
        return

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        if result == "win":
            cursor.execute(
                """
                UPDATE user_stats
                SET wins = wins + 1
                WHERE user_id = ? AND opponent = ?
                """,
                (user_id, opponent),
            )
        elif result == "loss":
            cursor.execute(
                """
                UPDATE user_stats
                SET losses = losses + 1
                WHERE user_id = ? AND opponent = ?
                """,
                (user_id, opponent),
            )
        else:
            cursor.execute(
                """
                UPDATE user_stats
                SET draws = draws + 1
                WHERE user_id = ? AND opponent = ?
                """,
                (user_id, opponent),
            )
        connection.commit()
        connection.close()
