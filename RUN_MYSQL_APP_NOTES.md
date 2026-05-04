# Run MySQL And Chess App

The web app runs on FastAPI with Uvicorn. MySQL stores accounts, Elo, leaderboard data, and game history.

The app automatically loads `.env` before connecting to MySQL.

## One-Time Setup

From the project folder:

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main
pip install -r requirements.txt
```

If `.env` is missing, create it from the example:

```bash
cp .env.example .env
```

The default `.env.example` is set up for Docker MySQL exposed on port `3307`:

```env
WEB_HOST=127.0.0.1
WEB_PORT=8000
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_ROOT_PASSWORD=change_me_root_password
MYSQL_USER=chess_app
MYSQL_PASSWORD=change_me_app_password
MYSQL_DATABASE=chess_web
```

If you already have an old Docker MySQL volume and get `Access denied`, either set `.env` to the same password values that were used when that volume was first created, or reset the Docker database volume.

## Option 1: Docker Runs Everything

Use this if you want Docker to run both MySQL and the web app:

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main
docker compose up --build
```

Train or continue training the ML chess model:

```bash
docker compose --profile trainer run --rm trainer
```

Optional training settings:

```bash
TRAIN_GAMES=24 \
TRAIN_SIMULATIONS=80 \
TRAIN_EPOCHS=2 \
docker compose --profile trainer run --rm trainer
```

The trainer writes checkpoints to `app/data/models`, which is mounted into the web container too.

Open:

```text
http://127.0.0.1:8000
```

Stop with `Ctrl+C`, or from another terminal:

```bash
docker compose down
```

Reset the Docker database volume:

```bash
docker compose down -v
```

Only reset the volume if you do not need the existing accounts or game history.

## Option 2: Docker MySQL + Local FastAPI

Use this if you want to run the Python app yourself with `python3 run.py`.

Terminal 1:

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main
docker compose up -d db
```

Terminal 2:

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main
python3 run.py
```

Open:

```text
http://127.0.0.1:8000
```

Do not run `docker compose up --build` and `python3 run.py` at the same time, because both try to use port `8000`.

Stop the local app with `Ctrl+C`.

Stop Docker MySQL with:

```bash
docker compose down
```

## Option 3: Your Own MySQL Server

Use this if you already have MySQL installed outside Docker.

Edit `.env` so it points to your local MySQL server:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=chess_app
MYSQL_PASSWORD=change_me_app_password
MYSQL_DATABASE=chess_web
```

Create the database and app user once from a MySQL admin account. Replace `change_me_app_password` with the same password used in `.env`:

```bash
mysql -u root -p
```

```sql
CREATE DATABASE IF NOT EXISTS chess_web
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'chess_app'@'localhost'
  IDENTIFIED BY 'change_me_app_password';

GRANT ALL PRIVILEGES ON chess_web.* TO 'chess_app'@'localhost';
FLUSH PRIVILEGES;
```

Then start the app:

```bash
python3 run.py
```

## Common Errors

If startup says `Access denied`, the MySQL user/password in `.env` does not match the actual MySQL account.

If startup says it cannot connect to MySQL on port `3307`, start Docker MySQL first:

```bash
docker compose up -d db
```

If port `8000` is already in use, stop the other app/container or change `WEB_PORT` in `.env`.
