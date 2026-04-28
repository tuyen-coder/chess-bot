from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.web import db as web_db
from app.web.config import HOST, PORT, SECURITY_HEADERS, STATIC_DIR
from app.web.handler import ApiError, api_error_handler, router


@asynccontextmanager
async def lifespan(app):
    web_db.init_db()
    yield


def create_app():
    fastapi_app = FastAPI(title="Chess Studio", lifespan=lifespan)
    fastapi_app.add_exception_handler(ApiError, api_error_handler)

    @fastapi_app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        if request.url.path in {"/", "/index.html"} or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    fastapi_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    fastapi_app.include_router(router)
    return fastapi_app


app = create_app()


def run():
    print(f"Web UI running at http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
