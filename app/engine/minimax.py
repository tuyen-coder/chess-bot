import random
import chess
import math


# Random move agent ----------------------------------------------------
def random_move(board):
    moves = list(board.legal_moves)
    if len(moves) == 0:
        return None
    return random.choice(moves)


# Minimax alpha-beta pruning agent -------------------------------------
pieceScore = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


# white -> maximize
# black -> minimize
def evaluate(board):
    if board.is_checkmate():
        if board.turn:   # white turn -> white gets checked -> white lost
            return -9999
        else:            # black turn -> black gets checked -> black lost
            return 9999
    if board.is_stalemate():
        return 0
    
    whiteScore = 0
    blackScore = 0

    for piece in pieceScore:
        value = pieceScore[piece]
        whiteScore += len(board.pieces(piece, chess.WHITE))*value
        blackScore += len(board.pieces(piece, chess.BLACK))*value

    score = whiteScore - blackScore
    score += 0.1*len(list(board.legal_moves))

    # if board.is_check():
    #     if board.turn == chess.BLACK:
    #         score += 0.1
    #     else:
    #         score -= 0.1

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    # base case
    if board.is_game_over() or depth == 0:
        return evaluate(board)
    moves = list(board.legal_moves)

    # white turn (maximize)
    if maximizingPlayer:
        maxEval = -math.inf
        moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            maxEval = max(eval, maxEval)
            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break
            if maximizingPlayer and maxEval == 9999:
                break
        return maxEval
        
    # black turn (minimize)
    else:
        minEval = math.inf
        moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            minEval = min(eval, minEval)
            beta = min(beta, minEval)
            if beta <= alpha:
                break
            if not maximizingPlayer and minEval == -9999:
                break
        return minEval
        

def find_best_move(board, depth):
    best_move = None
    maximizingPlayer = (board.turn == chess.WHITE) # if agent is white
    if maximizingPlayer:
        best_value = -math.inf
    else:
        best_value = math.inf

    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -100000, 100000, not maximizingPlayer)
        board.pop()

        if maximizingPlayer:
            if value > best_value:
                best_value = value
                best_move = move
        else:
            if value < best_value:
                best_value = value
                best_move = move

    return best_move

