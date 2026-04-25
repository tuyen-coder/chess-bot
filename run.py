import sys
from pathlib import Path

sys.pycache_prefix = str(Path(__file__).resolve().parent / ".pycache")

from app.main import main


if __name__ == "__main__":
    main()
