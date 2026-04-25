import math
import os

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Device setup


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = detect_device()
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "models",
    "ml_model.pt",
)

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

BOARD_PLANES = 12
EXTRA_PLANES = 8
INPUT_PLANES = BOARD_PLANES + EXTRA_PLANES
PROMOTION_TYPES = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PROMOTION_TO_INDEX = {
    promotion: index for index, promotion in enumerate(PROMOTION_TYPES)
}
ACTION_SIZE = 64 * 64 * len(PROMOTION_TYPES)


def default_self_play_batch_size():
    if device.type == "cuda":
        return 16
    if device.type == "mps":
        return 8
    return 1

# 1. Board encoding


def encode_board_array(board):
    tensor = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - square // 8
        col = square % 8
        offset = 0 if piece.color == chess.WHITE else 6
        tensor[piece.piece_type - 1 + offset][row][col] = 1.0

    extra = BOARD_PLANES

    if board.turn == chess.WHITE:
        tensor[extra, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[extra + 1, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[extra + 2, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[extra + 3, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[extra + 4, :, :] = 1.0

    if board.ep_square is not None:
        row = 7 - board.ep_square // 8
        col = board.ep_square % 8
        tensor[extra + 5, row, col] = 1.0

    tensor[extra + 6, :, :] = min(board.halfmove_clock, 100) / 100.0

    if board.can_claim_threefold_repetition():
        tensor[extra + 7, :, :] = 1.0

    return tensor


def encode_board(board):
    return torch.from_numpy(encode_board_array(board))


def encode_boards(boards):
    if not boards:
        return torch.empty((0, INPUT_PLANES, 8, 8), dtype=torch.float32)
    stacked = np.stack([encode_board_array(board) for board in boards])
    return torch.from_numpy(stacked)


# 2. Value network


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, ACTION_SIZE),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.value_head(x), self.policy_head(x)


def create_model():
    return ValueNet().to(device)


def move_to_index(move):
    promotion_index = PROMOTION_TO_INDEX.get(move.promotion, 0)
    return ((move.from_square * 64) + move.to_square) * len(PROMOTION_TYPES) + promotion_index


def index_to_move(index):
    promotion_index = index % len(PROMOTION_TYPES)
    square_index = index // len(PROMOTION_TYPES)
    from_square = square_index // 64
    to_square = square_index % 64
    return chess.Move(from_square, to_square, promotion=PROMOTION_TYPES[promotion_index])


def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    if not os.path.exists(checkpoint_path):
        return None

    model = create_model()

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded ML checkpoint: {checkpoint_path}", flush=True)
        return model
    except Exception as exc:
        print(f"Checkpoint load failed ({checkpoint_path}): {exc}", flush=True)
        return None


def save_model(model, checkpoint_path=DEFAULT_CHECKPOINT, metadata=None):
    payload = {"model_state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, checkpoint_path)
    print(f"Saved ML checkpoint: {checkpoint_path}", flush=True)


# 3. MCTS node


class Node:
    def __init__(self, board, parent=None, prior=1.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expanded(self):
        return len(self.children) > 0


# 4. PUCT parameters


C_PUCT = 1.4
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


def puct(parent, child):
    # child.value() is stored from the child's side-to-move perspective.
    # Negate it so selection stays in the parent's perspective.
    q = -child.value()
    u = (
        C_PUCT
        * child.prior
        * math.sqrt(parent.visit_count + 1)
        / (1 + child.visit_count)
    )
    return q + u


# 5. Search helpers


def terminal_value(board):
    if board.is_checkmate():
        return -1.0
    return 0.0


def evaluate_board(board, model):
    if board.is_game_over(claim_draw=True):
        return terminal_value(board), None

    state_tensor = encode_board(board).unsqueeze(0).to(device)
    with torch.inference_mode():
        value, policy_logits = model(state_tensor)
    return value.item(), policy_logits.squeeze(0).detach().cpu()


def evaluate_boards(boards, model):
    values = [0.0] * len(boards)
    policies = [None] * len(boards)
    pending_indices = []
    pending_boards = []

    for index, board in enumerate(boards):
        if board.is_game_over(claim_draw=True):
            values[index] = terminal_value(board)
        else:
            pending_indices.append(index)
            pending_boards.append(board)

    if pending_boards:
        state_batch = encode_boards(pending_boards).to(device)
        with torch.inference_mode():
            value_preds, policy_logits = model(state_batch)
            value_preds = value_preds.squeeze(-1).detach().cpu().tolist()
            policy_logits = policy_logits.detach().cpu()

        for batch_offset, (index, pred) in enumerate(zip(pending_indices, value_preds)):
            values[index] = float(pred)
            policies[index] = policy_logits[batch_offset]

    return values, policies


def move_priors(board, legal_moves, policy_logits=None):
    if not legal_moves:
        return np.array([], dtype=np.float32)

    if policy_logits is None:
        return np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)

    logits = np.array(
        [float(policy_logits[move_to_index(move)].item()) for move in legal_moves],
        dtype=np.float32,
    )
    logits -= logits.max()
    exp_logits = np.exp(logits)
    total = exp_logits.sum()

    if not np.isfinite(total) or total <= 0:
        return np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)

    return exp_logits / total


def expand_node(node, policy_logits=None, add_noise=False):
    legal_moves = list(node.board.legal_moves)
    priors = move_priors(node.board, legal_moves, policy_logits=policy_logits)

    if add_noise and len(priors) > 1:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(priors))
        priors = (1 - DIRICHLET_EPS) * priors + DIRICHLET_EPS * noise

    for move, prior in zip(legal_moves, priors):
        node.board.push(move)
        node.children[move] = Node(
            node.board.copy(),
            node,
            prior=float(prior),
        )
        node.board.pop()


def select_leaf(root):
    node = root
    path = [node]

    while node.expanded():
        parent = path[-1]
        node = max(node.children.values(), key=lambda child: puct(parent, child))
        path.append(node)

    return node, path


def run_simulations(roots, model, simulations):
    for _ in range(simulations):
        selections = [select_leaf(root) for root in roots]
        leaf_boards = [node.board for node, _ in selections]
        values, policies = evaluate_boards(leaf_boards, model)

        for (node, path), value, policy_logits in zip(selections, values, policies):
            if not node.board.is_game_over(claim_draw=True):
                expand_node(node, policy_logits=policy_logits, add_noise=False)

            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value


def pick_root_move(root, training):
    moves = list(root.children.keys())
    if not moves:
        return None

    visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float32)

    if visits.sum() <= 0:
        priors = np.array([root.children[move].prior for move in moves], dtype=np.float32)
        priors /= priors.sum()
        return np.random.choice(moves, p=priors) if training else moves[int(np.argmax(priors))]

    if training:
        probs = visits / visits.sum()
        return np.random.choice(moves, p=probs)

    return moves[int(np.argmax(visits))]


def root_policy_target(root):
    target = np.zeros(ACTION_SIZE, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())

    if total_visits <= 0:
        legal_moves = list(root.children.keys())
        if not legal_moves:
            return target
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            target[move_to_index(move)] = uniform
        return target

    for move, child in root.children.items():
        target[move_to_index(move)] = child.visit_count / total_visits

    return target


# 6. MCTS with policy priors


def mcts_move(board, model, simulations=60, training=True):
    if board.is_game_over(claim_draw=True):
        return None, terminal_value(board), np.zeros(ACTION_SIZE, dtype=np.float32)

    model.eval()
    root = Node(board.copy())
    _, root_policy_logits = evaluate_board(root.board, model)
    expand_node(root, policy_logits=root_policy_logits, add_noise=training)
    run_simulations([root], model, simulations)
    return pick_root_move(root, training), root.value(), root_policy_target(root)


# 7. Self-play data generation


def white_game_result(board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def initialize_root(board):
    root = Node(board.copy())
    return root


def finalize_game(game_state):
    result = white_game_result(game_state["board"])
    samples = []

    for state, turn, policy_target in game_state["positions"]:
        target = result if turn == chess.WHITE else -result
        samples.append((state, target, policy_target))

    print(
        f"Finished game {game_state['id']} | Result: {game_state['board'].result(claim_draw=True)}",
        flush=True,
    )
    return samples


def generate_self_play_data(model, games, simulations, batch_size):
    if games <= 0:
        return []

    model.eval()
    batch_size = max(1, batch_size)
    next_game_id = 1
    active_games = []
    all_samples = []

    def start_game(game_id):
        print(f"Starting game {game_id}", flush=True)
        active_games.append(
            {
                "id": game_id,
                "board": chess.Board(),
                "positions": [],
            }
        )

    while next_game_id <= games and len(active_games) < batch_size:
        start_game(next_game_id)
        next_game_id += 1

    while active_games:
        roots = []

        for game_state in active_games:
            roots.append(initialize_root(game_state["board"]))

        root_boards = [root.board for root in roots]
        _, root_policies = evaluate_boards(root_boards, model)

        for root, policy_logits in zip(roots, root_policies):
            expand_node(root, policy_logits=policy_logits, add_noise=True)

        run_simulations(roots, model, simulations)

        survivors = []

        for game_state, root in zip(active_games, roots):
            game_state["positions"].append(
                (
                    encode_board_array(game_state["board"]),
                    game_state["board"].turn,
                    root_policy_target(root),
                )
            )
            move = pick_root_move(root, training=True)

            if move is not None:
                game_state["board"].push(move)

            if game_state["board"].is_game_over(claim_draw=True):
                all_samples.extend(finalize_game(game_state))
            else:
                survivors.append(game_state)

        active_games = survivors

        while next_game_id <= games and len(active_games) < batch_size:
            start_game(next_game_id)
            next_game_id += 1

    return all_samples


# 8. Train / load model


def train_model(
    games=24,
    simulations=80,
    epochs=4,
    checkpoint_path=DEFAULT_CHECKPOINT,
    force_retrain=False,
    self_play_batch_size=None,
):
    existing_model = None if force_retrain else load_model(checkpoint_path)
    if existing_model is not None:
        return existing_model

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    value_loss_fn = nn.MSELoss()

    if self_play_batch_size is None:
        self_play_batch_size = default_self_play_batch_size()

    print(
        f"Using device {device.type} with self-play batch size {self_play_batch_size}",
        flush=True,
    )
    print("Generating self-play data...", flush=True)

    samples = generate_self_play_data(
        model,
        games=games,
        simulations=simulations,
        batch_size=self_play_batch_size,
    )

    X = [state for state, _, _ in samples]
    y_value = [target for _, target, _ in samples]
    y_policy = [policy for _, _, policy in samples]

    if not X:
        print("No self-play samples were generated; returning untrained model.", flush=True)
        model.eval()
        return model

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_value = torch.tensor(np.array(y_value), dtype=torch.float32).unsqueeze(1)
    y_policy = torch.tensor(np.array(y_policy), dtype=torch.float32)

    shuffled = torch.randperm(X.size(0))
    X = X[shuffled]
    y_value = y_value[shuffled]
    y_policy = y_policy[shuffled]

    val_size = max(1, X.size(0) // 10) if X.size(0) >= 10 else 0

    if val_size > 0:
        X_val = X[:val_size]
        y_value_val = y_value[:val_size]
        y_policy_val = y_policy[:val_size]
        X_train = X[val_size:]
        y_value_train = y_value[val_size:]
        y_policy_train = y_policy[val_size:]
    else:
        X_val = None
        y_value_val = None
        y_policy_val = None
        X_train = X
        y_value_train = y_value
        y_policy_train = y_policy

    print(
        f"Training network on {X_train.size(0)} samples"
        + (f" with {X_val.size(0)} validation samples..." if X_val is not None else "..."),
        flush=True,
    )

    batch_size = 128

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0.0

        for index in range(0, X_train.size(0), batch_size):
            indices = permutation[index:index + batch_size]
            batch_x = X_train[indices].to(device)
            batch_y_value = y_value_train[indices].to(device)
            batch_y_policy = y_policy_train[indices].to(device)

            value_preds, policy_logits = model(batch_x)
            value_loss = value_loss_fn(value_preds, batch_y_value)
            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(batch_y_policy * policy_log_probs).sum(dim=1).mean()
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        message = f"Epoch {epoch + 1}/{epochs} train_loss: {total_loss:.4f}"

        if X_val is not None and X_val.size(0) > 0:
            with torch.inference_mode():
                val_value_preds, val_policy_logits = model(X_val.to(device))
                val_value_loss = value_loss_fn(val_value_preds, y_value_val.to(device)).item()
                val_policy_log_probs = torch.log_softmax(val_policy_logits, dim=1)
                val_policy_loss = -(
                    y_policy_val.to(device) * val_policy_log_probs
                ).sum(dim=1).mean().item()
                val_loss = val_value_loss + val_policy_loss
            message += (
                f" | val_loss: {val_loss:.4f}"
                f" (value={val_value_loss:.4f}, policy={val_policy_loss:.4f})"
            )

        print(message, flush=True)

    save_model(
        model,
        checkpoint_path=checkpoint_path,
        metadata={
            "games": games,
            "simulations": simulations,
            "epochs": epochs,
            "input_planes": INPUT_PLANES,
            "action_size": ACTION_SIZE,
        },
    )

    print("Training complete.", flush=True)
    model.eval()
    return model


# 9. Move function for ChessMain


def ml_move(board, model, simulations=120):
    if model is None:
        model = load_model()
        if model is None:
            model = train_model()

    move, _, _ = mcts_move(board, model, simulations=simulations, training=False)
    return move
