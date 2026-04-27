# Run MySQL And Chess App

## Docker: Start Everything

If Docker Desktop is running, this is the easiest way:

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000
```

Stop with `Ctrl+C`, or run:

```bash
docker compose down
```

Reset the Docker database:

```bash
docker compose down -v
```

## Manual: Start Everything

## Terminal 1: Start MySQL

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main

/usr/local/mysql/bin/mysqld \
  --basedir=/usr/local/mysql \
  --datadir=/Users/lannguyen/Downloads/AI-Assignment-2-main/app/data/mysql_data \
  --port=3307 \
  --socket=/tmp/chess_mysql.sock \
  --pid-file=/tmp/chess_mysql.pid \
  --mysqlx=0
```

Leave this terminal open.

## Terminal 2: Start The Chess App

```bash
cd /Users/lannguyen/Downloads/AI-Assignment-2-main

MYSQL_HOST=127.0.0.1 \
MYSQL_PORT=3307 \
MYSQL_USER=chess_app \
MYSQL_PASSWORD='chesspass123' \
MYSQL_DATABASE=chess_web \
python3 run.py
```

Use the same MySQL values as `docker-compose.yml`. The password is
`chesspass123`, not the placeholder `your_mysql_password`.

Open:

```text
http://127.0.0.1:8000
```

## Stop The App

Press `Ctrl+C` in Terminal 2.

## Stop MySQL

Run this in another terminal:

```bash
/usr/local/mysql/bin/mysqladmin --protocol=TCP -h 127.0.0.1 -P 3307 -u root shutdown
```
