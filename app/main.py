import sys
from pathlib import Path

sys.pycache_prefix = str(Path(__file__).resolve().parents[1] / ".pycache")

from app.web.app import run


def main():
    run()


if __name__ == "__main__":
    main()
