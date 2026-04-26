# AI Chess Assignment

This project is a chess game built with `tkinter`, `python-chess`, and two AI approaches:

- a classic minimax agent with alpha-beta pruning
- a neural-network-based agent trained from self-play with MCTS

## Features

- Play chess in a desktop GUI
- Highlight legal moves and checks
- Play in multiple modes:
  - Player vs Player
  - Player vs Random AI
  - Player vs Minimax AI
  - Random AI vs Minimax AI
  - Minimax AI vs Minimax AI
  - Player vs ML AI
  - ML AI vs Random AI

## Project Structure

- `run.py`: main launcher for the web app
- `app/main.py`: app entrypoint
- `app/engine/chess_engine.py`: board state wrapper around `python-chess`
- `app/engine/minimax.py`: random agent and minimax agent
- `app/ml/model.py`: neural network, MCTS, self-play training, checkpointing
- `app/web/app.py`: web server and API
- `app/web/db.py`: MySQL user database
- `app/web/templates/index.html`: main web page
- `app/static/css/styles.css`: browser styling
- `app/static/js/app.js`: browser interactions
- `app/ui/desktop_ui.py`: Tkinter desktop UI
- `app/data/models/ml_model.pt`: saved ML checkpoint

## Requirements

Install Python 3 and these packages:

```bash
pip install -r requirements.txt
```

`tkinter` is included with most Python desktop installs. On some systems you may need to install it separately.

## Database Setup

The web app uses MySQL for accounts, bot stats, Elo, and leaderboard data.

Create a MySQL user or use an existing local user, then configure the connection with environment variables:

```bash
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DATABASE=chess_web
```

The app creates the configured database and tables if they do not exist.

## How To Run

Start the web app with:

```bash
python3 run.py
```

Then open `http://127.0.0.1:8000`.

If you want the desktop UI, run:

```bash
python3 -m app.ui.desktop_ui
```

## How The AI Works

### 1. Random AI

The random agent selects any legal move uniformly at random.

### 2. Minimax AI

The minimax agent in `app/engine/minimax.py` uses:

- material-based evaluation
- alpha-beta pruning
- configurable search depth

### 3. ML AI

The ML agent in `app/ml/model.py` uses:

- board encoding with piece planes and game-state planes
- a neural network with:
  - a value head to estimate position strength
  - a policy head to predict move preferences
- Monte Carlo Tree Search (MCTS)
- self-play training
- policy targets generated from root visit counts

## Checkpoints

The ML model is saved as:

```text
app/data/models/ml_model.pt
```

Behavior:

- If `app/data/models/ml_model.pt` already exists, the program loads it and skips retraining.
- If the checkpoint does not exist, the ML solver trains a model and saves it.
- If the checkpoint is outdated or incompatible with the current network architecture, loading will fail safely and the model will be retrained.

This means you do not need to train beforehand if a valid checkpoint is already present.

## Training Notes

The current ML pipeline:

- auto-detects `cuda`, then `mps`, then falls back to `cpu`
- batches self-play evaluations to make GPU use more effective
- trains on both:
  - value targets from final game results
  - policy targets from MCTS visit distributions

The first ML run can take time, especially without a GPU.

## Notes For Submission

If this is for an assignment, useful talking points are:

- comparison between classical search and learning-based search
- why `python-chess` was used for move legality and game rules
- how MCTS improves move selection compared with direct network output
- why checkpointing makes repeated runs much faster

## Possible Future Improvements

- stronger policy architecture
- replay buffer across training runs
- opening book support
- endgame tablebase integration
- stronger board evaluation features
- configurable training settings from the UI
