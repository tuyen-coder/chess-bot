import json
from dataclasses import dataclass

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse

from app.web import db as web_db
from app.web.config import (
    HOST,
    MAX_JSON_BYTES,
    MODE_LABELS,
    PORT,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
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

router = APIRouter()
AUTH_RATE_LIMITER = RateLimiter(limit=10, window_seconds=5 * 60)
PASSWORD_RATE_LIMITER = RateLimiter(limit=8, window_seconds=10 * 60)


class ApiError(Exception):
    def __init__(self, message, status_code=status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class RequestContext:
    request: Request
    response: Response
    session_token: str
    session: object


def api_error_handler(request, exc):
    return JSONResponse({"error": exc.message}, status_code=exc.status_code)


def set_session_cookie(response, session_token):
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )


def session_from_request(request):
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    with SESSIONS_LOCK:
        if session_token and session_token in SESSIONS:
            return session_token, SESSIONS[session_token], False
    token, session = create_session()
    return token, session, True


def get_context(request: Request, response: Response):
    session_token, session, created = session_from_request(request)
    if created:
        set_session_cookie(response, session_token)
    return RequestContext(request, response, session_token, session)


def require_user(context: RequestContext = Depends(get_context)):
    if context.session.user_id is None:
        raise ApiError("Please log in first.", status.HTTP_401_UNAUTHORIZED)
    return context


def request_origin_allowed(request):
    origin = request.headers.get("Origin")
    if not origin:
        return True
    allowed = {
        f"http://{request.headers.get('Host')}",
        f"http://{HOST}:{PORT}",
        f"http://localhost:{PORT}",
        f"http://127.0.0.1:{PORT}",
    }
    return origin in allowed


async def read_json_payload(request: Request):
    if not request_origin_allowed(request):
        raise ApiError("Invalid request origin.", status.HTTP_403_FORBIDDEN)

    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise ApiError("Content-Type must be application/json.")

    try:
        content_length = int(request.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ApiError("Invalid Content-Length header.") from exc

    if content_length > MAX_JSON_BYTES:
        raise ApiError("Request body is too large.")
    if content_length == 0:
        return {}

    raw_body = await request.body()
    if not raw_body:
        return {}
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError("Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise ApiError("JSON body must be an object.")
    return payload


def limited(limiter, bucket, request):
    client_ip = request.client.host if request.client else "unknown"
    return not limiter.allow(f"{bucket}:{client_ip}")


def rotate_authenticated_session(context, user_id=None):
    if user_id is not None:
        context.session.user_id = user_id
    new_token, _ = rotate_session(previous_token=context.session_token, session=context.session)
    context.session_token = new_token
    set_session_cookie(context.response, new_token)


def destroy_session(context):
    with SESSIONS_LOCK:
        SESSIONS.pop(context.session_token, None)
    context.response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def state_payload(session):
    if session.user_id is not None:
        match = get_active_match(session.online_match_id, session.user_id) or find_user_match(session.user_id)
        if match:
            session.online_match_id = match.id
            with match.lock:
                return match.state(session.user_id)
    return session.controller.state(session.user_id)


def valid_square_from_payload(payload):
    row = payload.get("row")
    col = payload.get("col")
    if not isinstance(row, int) or not isinstance(col, int):
        raise ApiError("Invalid square.")
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ApiError("Square is out of range.")
    return row, col


@router.get("/")
@router.get("/index.html")
def index(request: Request):
    session_token, _, created = session_from_request(request)
    response = FileResponse(TEMPLATE_DIR / "index.html", media_type="text/html; charset=utf-8")
    if created:
        set_session_cookie(response, session_token)
    return response


@router.get("/api/state")
def api_state(context: RequestContext = Depends(get_context)):
    return state_payload(context.session)


@router.get("/api/leaderboard")
def api_leaderboard(context: RequestContext = Depends(get_context)):
    return {"leaderboard": web_db.get_leaderboard()}


@router.post("/api/register")
async def api_register(
    request: Request,
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(get_context),
):
    if limited(AUTH_RATE_LIMITER, "register", request):
        raise ApiError("Too many registration attempts. Please wait and try again.", status.HTTP_429_TOO_MANY_REQUESTS)
    try:
        user = web_db.create_user(payload.get("username", "").strip(), payload.get("password", ""))
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    rotate_authenticated_session(context, user_id=user["id"])
    return context.session.controller.state(context.session.user_id)


@router.post("/api/login")
async def api_login(
    request: Request,
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(get_context),
):
    if limited(AUTH_RATE_LIMITER, "login", request):
        raise ApiError("Too many login attempts. Please wait and try again.", status.HTTP_429_TOO_MANY_REQUESTS)
    try:
        user = web_db.verify_user(payload.get("username", "").strip(), payload.get("password", ""))
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    rotate_authenticated_session(context, user_id=user["id"])
    return context.session.controller.state(context.session.user_id)


@router.post("/api/logout")
async def api_logout(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(get_context),
):
    leave_matchmaking(context.session.user_id)
    destroy_session(context)
    new_token, new_session = create_session()
    set_session_cookie(context.response, new_token)
    return new_session.controller.state()


@router.post("/api/profile/username")
async def api_profile_username(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    try:
        web_db.update_username(context.session.user_id, payload.get("username", ""))
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    return context.session.controller.state(context.session.user_id)


@router.post("/api/profile/password")
async def api_profile_password(
    request: Request,
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    if limited(PASSWORD_RATE_LIMITER, "profile-password", request):
        raise ApiError("Too many password change attempts. Please wait and try again.", status.HTTP_429_TOO_MANY_REQUESTS)
    try:
        web_db.update_password(
            context.session.user_id,
            payload.get("currentPassword", ""),
            payload.get("newPassword", ""),
        )
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    return context.session.controller.state(context.session.user_id)


@router.post("/api/history/list")
async def api_history_list(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    return {"history": web_db.get_game_history(context.session.user_id)}


@router.post("/api/history/detail")
async def api_history_detail(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    history_id = payload.get("id")
    if not isinstance(history_id, int):
        raise ApiError("Invalid game history id.")
    record = history_detail_payload(context.session.user_id, history_id)
    if record is None:
        raise ApiError("Game not found.", status.HTTP_404_NOT_FOUND)
    return {"game": record}


@router.post("/api/start")
async def api_start(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    mode = payload.get("mode")
    if mode not in MODE_LABELS:
        raise ApiError("Invalid mode.")
    leave_matchmaking(context.session.user_id)
    context.session.online_match_id = None
    context.session.controller.start_game(mode)
    context.session.controller.maybe_record_result(context.session.user_id)
    return context.session.controller.state(context.session.user_id)


@router.post("/api/menu")
async def api_menu(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    leave_matchmaking(context.session.user_id)
    context.session.online_match_id = None
    context.session.controller.go_to_menu()
    return context.session.controller.state(context.session.user_id)


@router.post("/api/matchmaking/join")
async def api_matchmaking_join(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = join_matchmaking(context.session.user_id)
    if match is None:
        context.session.online_match_id = None
        return waiting_state(context.session.user_id)
    context.session.online_match_id = match.id
    with match.lock:
        return match.state(context.session.user_id)


@router.post("/api/matchmaking/leave")
async def api_matchmaking_leave(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    leave_matchmaking(context.session.user_id)
    context.session.online_match_id = None
    context.session.controller.go_to_menu()
    return context.session.controller.state(context.session.user_id)


@router.post("/api/matchmaking/state")
async def api_matchmaking_state(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = get_active_match(context.session.online_match_id, context.session.user_id) or find_user_match(context.session.user_id)
    if match:
        context.session.online_match_id = match.id
        with match.lock:
            return match.state(context.session.user_id)
    with MATCH_LOCK:
        waiting = context.session.user_id in MATCH_QUEUE
    if waiting:
        return waiting_state(context.session.user_id)
    return context.session.controller.state(context.session.user_id)


def active_match_for_context(context):
    match = get_active_match(context.session.online_match_id, context.session.user_id)
    if match is None:
        raise ApiError("No active online match.")
    return match


@router.post("/api/matchmaking/resign")
async def api_matchmaking_resign(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = active_match_for_context(context)
    with match.lock:
        match.resign(context.session.user_id)
        return match.state(context.session.user_id)


@router.post("/api/matchmaking/draw-offer")
async def api_matchmaking_draw_offer(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = active_match_for_context(context)
    with match.lock:
        match.offer_draw(context.session.user_id)
        return match.state(context.session.user_id)


@router.post("/api/matchmaking/draw-respond")
async def api_matchmaking_draw_respond(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = active_match_for_context(context)
    with match.lock:
        match.respond_draw(context.session.user_id, bool(payload.get("accept")))
        return match.state(context.session.user_id)


@router.post("/api/matchmaking/rematch")
async def api_matchmaking_rematch(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = active_match_for_context(context)
    with match.lock:
        match.request_rematch(context.session.user_id)
        return match.state(context.session.user_id)


@router.post("/api/click")
async def api_click(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    row, col = valid_square_from_payload(payload)
    context.session.controller.click_square(row, col)
    context.session.controller.maybe_record_result(context.session.user_id)
    return context.session.controller.state(context.session.user_id)


@router.post("/api/matchmaking/click")
async def api_matchmaking_click(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    match = active_match_for_context(context)
    row, col = valid_square_from_payload(payload)
    with match.lock:
        match.click_square(context.session.user_id, row, col)
        return match.state(context.session.user_id)


@router.post("/api/ai-step")
async def api_ai_step(
    payload: dict = Depends(read_json_payload),
    context: RequestContext = Depends(require_user),
):
    context.session.controller.ai_step()
    context.session.controller.maybe_record_result(context.session.user_id)
    return context.session.controller.state(context.session.user_id)
