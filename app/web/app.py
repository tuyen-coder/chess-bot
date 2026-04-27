from http.server import ThreadingHTTPServer

from app.web import db as web_db
from app.web.config import HOST, PORT
from app.web.handler import ChessRequestHandler


def run():
    web_db.init_db()
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
