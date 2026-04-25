import chess


class GameState:

    def __init__(self):
        self.board = chess.Board()
        self.moveLog = []


    def board_to_array(self):
        # return 8x8 array for UI
        # row 0 is the top of the screen (rank 8), row 7 is the bottom (rank 1)
        # chess.Board() use file/rank, UI use row/col
        arr = [["--" for _ in range(8)] for _ in range(8)]
        for r in range(8):
            rank = 7 - r  # convert row -> chess rank index (0 = rank1)
            for c in range(8):
                sq = chess.square(c, rank)
                piece = self.board.piece_at(sq)
                if piece is None:
                    arr[r][c] = "--"
                else:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    symbol = piece.symbol().upper()

                    if symbol == 'P':
                        arr[r][c] = f"{color}p"
                    else:
                        arr[r][c] = f"{color}{symbol}"
        return arr


    def rc_to_uci(self, start, end):
        # convert [(row,col),(row,col)] to UCI string (ex:'e2e4')
        # convert [(row,col),(row,col)] to UCI string (ex:'e2e4')
        # chess lib use uci, UI use rc

        def rc_to_sq(row, col):
            file = chr(ord('a') + col)
            rank = str(8 - row)
            return file + rank
        return rc_to_sq(start[0], start[1]) + rc_to_sq(end[0], end[1])


    def rc_to_algebraic(self, rc):
        # convert (row,col) to UCI string (ex:'e2')
        row, col = rc
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank


    def make_move(self, start, end):
        # move from start(row,col) to end(row,col)
        # return true if move was legal and made, otherwise false
        uci = self.rc_to_uci(start, end)

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False

        if move not in self.board.legal_moves:
            # try promotion to queen
            promo_move = chess.Move.from_uci(uci + "q")
            if promo_move in self.board.legal_moves:
                move = promo_move
            else:
                return False

        self.board.push(move)
        self.moveLog.append(move.uci())
        return True


    def apply_move_obj(self, move_obj):
        # apply move obj from solver if legal
        if move_obj is None:
            return False
        if move_obj in self.board.legal_moves:
            self.board.push(move_obj)
            self.moveLog.append(move_obj.uci())
            return True
        return False


    def legal_moves_from(self, row, col):
        # return list of (row,col) squares that the piece at (row,col) can move to
        targets = []
        file = col
        rank = 7 - row
        sq = chess.square(file, rank)
        piece = self.board.piece_at(sq)
        if piece is None:
            return targets
        for move in self.board.legal_moves:
            if move.from_square == sq:
                to_sq = move.to_square
                to_file = chess.square_file(to_sq)
                to_rank = chess.square_rank(to_sq)
                to_row = 7 - to_rank
                to_col = to_file
                targets.append((to_row, to_col))
        return targets


    def is_check(self):
        return self.board.is_check()


    def is_checkmate(self):
        return self.board.is_checkmate()


    def is_stalemate(self):
        return self.board.is_stalemate()


    def is_game_over(self):
        return (
            self.board.is_game_over()
            or self.board.can_claim_threefold_repetition()
        )

    def game_result(self):
        if self.board.is_checkmate():
            winner = 'White' if not self.board.turn else 'Black'
            return f"Checkmate - {winner} wins"
        if self.board.is_stalemate():
            return "Stalemate - Draw"
        if self.board.is_insufficient_material():
            return "Draw - Insufficient material"
        if self.board.can_claim_fifty_moves():
            return "Draw - Fifty-move rule"
        if self.board.can_claim_threefold_repetition():
            return "Draw - Threefold repetition"

        return "Game over"


    def king_in_check_rc(self):
        # return (row,col) of that king if board gets check
        if not self.board.is_check():
            return None
        sq = self.board.king(self.board.turn)
        if sq is None:
            return None
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        row = 7 - rank
        col = file
        return (row, col)
