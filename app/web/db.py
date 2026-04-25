import hashlib
import os
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chess_web.db"
DB_LOCK = threading.Lock()
OPPONENTS = ["random", "minimax", "ml"]


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
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
    normalized = (username or "").strip()
    if len(normalized) < 3:
        raise ValueError("Username must be at least 3 characters.")
    return normalized


def validate_password(password):
    if not password:
        raise ValueError("Password is required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    return password


def verify_password(password, password_hash, password_salt):
    candidate_hash, _ = hash_password(password, salt=bytes.fromhex(password_salt))
    return candidate_hash == password_hash


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
            SELECT id, username, password_hash, password_salt, created_at
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


def record_result(user_id, opponent, result):
    if opponent not in OPPONENTS:
        return

    column = {
        "win": "wins",
        "loss": "losses",
        "draw": "draws",
    }.get(result)

    if column is None:
        return

    with DB_LOCK:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            f"""
            UPDATE user_stats
            SET {column} = {column} + 1
            WHERE user_id = ? AND opponent = ?
            """,
            (user_id, opponent),
        )
        connection.commit()
        connection.close()
