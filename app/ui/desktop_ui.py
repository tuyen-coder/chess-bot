import sys
from pathlib import Path
import tkinter as tk
import tkinter.messagebox as messagebox

import chess

sys.pycache_prefix = str(Path(__file__).resolve().parents[2] / ".pycache")

from app.engine import chess_engine
from app.engine import minimax as solver
from app.ml import model as ml_solver

AI_DELAY = 800
MINIMAX_DEPTH = 3
CHESS_TURN = chess.WHITE
BOARD_SQUARE_SIZE = 78
BOARD_FONT = ("Times New Roman", 36, "bold")
TITLE_FONT = ("Georgia", 24, "bold")
SUBTITLE_FONT = ("Georgia", 11)
CARD_TITLE_FONT = ("Georgia", 12, "bold")
TEXT_FONT = ("Helvetica", 11)
SMALL_FONT = ("Helvetica", 10)
COORD_FONT = ("Helvetica", 10, "bold")

COLORS = {
    "app_bg": "#f4efe6",
    "panel_bg": "#e9dfcf",
    "card_bg": "#fffaf2",
    "light_square": "#ead7b5",
    "dark_square": "#8f5f3b",
    "selected": "#f4c95d",
    "legal_move": "#6fb6d8",
    "check": "#d95d39",
    "text": "#2f241f",
    "muted": "#6e6258",
    "accent": "#3d2c1d",
    "button": "#f0dfc3",
    "button_text": "#1b1511",
    "button_border": "#3d2c1d",
    "move_log_text": "#1f1813",
    "piece_dark": "#12100d",
    "piece_light": "#12100d",
    "info_bar_bg": "#f8f1e6",
}

PIECE_UNICODE = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}

MODE_LABELS = {
    "pvp": "Player vs Player",
    "pvr": "Player vs Random AI",
    "pvm": "Player vs Minimax AI",
    "rvm": "Random AI vs Minimax AI",
    "mvm": "Minimax AI vs Minimax AI",
    "pvl": "Player vs ML AI",
    "mlr": "ML AI vs Random AI",
}


class ChessUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        self.root.configure(bg=COLORS["app_bg"])
        self.root.minsize(1120, 760)

        self.game_state = chess_engine.GameState()
        self.buttons = [[None for _ in range(8)] for _ in range(8)]
        self.selected = None
        self.legal_moves = []
        self.mode = None
        self.ml_model = None
        self.ai_after_id = None
        self.status_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.last_move_var = tk.StringVar()
        self.move_count_var = tk.StringVar()
        self.selection_var = tk.StringVar(value="No piece selected")
        self.legal_moves_var = tk.StringVar(value="Click a piece to preview its legal moves.")
        self.create_menu()

    # Menu
    def create_menu(self):
        self.cancel_ai_loop()
        self.clear_screen()

        shell = tk.Frame(self.root, bg=COLORS["app_bg"], padx=40, pady=36)
        shell.pack(fill="both", expand=True)

        hero = tk.Frame(shell, bg=COLORS["panel_bg"], bd=0, padx=36, pady=28)
        hero.pack(fill="x", pady=(0, 24))

        tk.Label(
            hero,
            text="Chess Studio",
            font=TITLE_FONT,
            bg=COLORS["panel_bg"],
            fg=COLORS["accent"],
        ).pack(anchor="w")

        tk.Label(
            hero,
            text="Choose a mode to play, test search agents, or watch the AIs battle it out.",
            font=SUBTITLE_FONT,
            bg=COLORS["panel_bg"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(8, 0))

        modes_frame = tk.Frame(shell, bg=COLORS["app_bg"])
        modes_frame.pack(fill="both", expand=True)

        modes = [
            ("1. Player vs Player", "pvp"),
            ("2. Player vs Random AI", "pvr"),
            ("3. Player vs Minimax AI", "pvm"),
            ("4. Random vs Minimax", "rvm"),
            ("5. Minimax vs Minimax", "mvm"),
            ("6. Player vs ML AI", "pvl"),
            ("7. ML vs Random", "mlr"),
        ]

        for index, (text, mode) in enumerate(modes):
            card = tk.Frame(
                modes_frame,
                bg=COLORS["card_bg"],
                padx=18,
                pady=16,
                highlightbackground="#d4c3ae",
                highlightthickness=1,
            )
            card.grid(
                row=index // 2,
                column=index % 2,
                padx=10,
                pady=10,
                sticky="nsew",
            )

            tk.Label(
                card,
                text=text,
                font=("Georgia", 14, "bold"),
                bg=COLORS["card_bg"],
                fg=COLORS["text"],
            ).pack(anchor="w")

            tk.Button(
                card,
                text="Start",
                command=lambda m=mode: self.start_game(m),
                font=TEXT_FONT,
                bg=COLORS["button"],
                fg=COLORS["button_text"],
                activebackground="#dfc79f",
                activeforeground=COLORS["button_text"],
                relief="solid",
                bd=1,
                highlightbackground=COLORS["button_border"],
                padx=18,
                pady=8,
                cursor="hand2",
            ).pack(anchor="w", pady=(14, 0))

        for column in range(2):
            modes_frame.grid_columnconfigure(column, weight=1)

    def start_game(self, mode):
        self.mode = mode
        self.game_state = chess_engine.GameState()
        self.selected = None
        self.legal_moves = []

        if mode in ["pvl", "mlr"] and self.ml_model is None:
            print("Loading or training ML model...")
            self.ml_model = ml_solver.train_model()

        self.clear_screen()
        self.create_board_screen()
        self.update_board()
        self.schedule_ai_loop()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def cancel_ai_loop(self):
        if self.ai_after_id is not None:
            try:
                self.root.after_cancel(self.ai_after_id)
            except tk.TclError:
                pass
            self.ai_after_id = None

    def schedule_ai_loop(self):
        self.ai_after_id = self.root.after(AI_DELAY, self.ai_loop)

    # Board
    def create_board_screen(self):
        self.root.configure(bg=COLORS["app_bg"])

        page = tk.Frame(self.root, bg=COLORS["app_bg"], padx=28, pady=24)
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=0)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        board_panel = tk.Frame(page, bg=COLORS["panel_bg"], padx=18, pady=18)
        board_panel.grid(row=0, column=0, sticky="n")

        sidebar = tk.Frame(page, bg=COLORS["app_bg"], padx=18)
        sidebar.grid(row=0, column=1, sticky="nsew")

        top_bar = tk.Frame(board_panel, bg=COLORS["panel_bg"])
        top_bar.pack(fill="x", pady=(0, 10))

        tk.Label(
            top_bar,
            text="Match Board",
            font=("Georgia", 18, "bold"),
            bg=COLORS["panel_bg"],
            fg=COLORS["accent"],
        ).pack(side="left")

        tk.Button(
            top_bar,
            text="Back to Menu",
            command=self.create_menu,
            font=SMALL_FONT,
            bg=COLORS["button"],
            fg=COLORS["button_text"],
            activebackground="#dfc79f",
            activeforeground=COLORS["button_text"],
            relief="solid",
            bd=1,
            highlightbackground=COLORS["button_border"],
            padx=14,
            pady=6,
            cursor="hand2",
        ).pack(side="right")

        self.create_board(board_panel)
        self.create_board_info_bar(board_panel)
        self.create_sidebar(sidebar)

    def create_board(self, parent):
        container = tk.Frame(parent, bg=COLORS["panel_bg"])
        container.pack()

        for c, file_label in enumerate("abcdefgh"):
            tk.Label(
                container,
                text=file_label,
                font=COORD_FONT,
                bg=COLORS["panel_bg"],
                fg=COLORS["muted"],
                anchor="center",
            ).grid(row=0, column=c + 1, pady=(0, 6), sticky="nsew")

        for r in range(8):
            rank_text = str(8 - r)
            tk.Label(
                container,
                text=rank_text,
                font=COORD_FONT,
                bg=COLORS["panel_bg"],
                fg=COLORS["muted"],
                anchor="center",
            ).grid(row=r + 1, column=0, padx=(0, 8), sticky="nsew")

            tk.Label(
                container,
                text=rank_text,
                font=COORD_FONT,
                bg=COLORS["panel_bg"],
                fg=COLORS["muted"],
                anchor="center",
            ).grid(row=r + 1, column=9, padx=(8, 0), sticky="nsew")

        for c, file_label in enumerate("abcdefgh"):
            tk.Label(
                container,
                text=file_label,
                font=COORD_FONT,
                bg=COLORS["panel_bg"],
                fg=COLORS["muted"],
                anchor="center",
            ).grid(row=9, column=c + 1, pady=(6, 0), sticky="nsew")

        for r in range(8):
            for c in range(8):
                cell = tk.Frame(
                    container,
                    width=BOARD_SQUARE_SIZE,
                    height=BOARD_SQUARE_SIZE,
                    bg=COLORS["light_square"] if (r + c) % 2 == 0 else COLORS["dark_square"],
                    highlightbackground="#42352c",
                    highlightthickness=1,
                )
                cell.grid(row=r + 1, column=c + 1, sticky="nsew")
                cell.grid_propagate(False)

                square = tk.Label(
                    cell,
                    text="",
                    font=BOARD_FONT,
                    bg=cell["bg"],
                    fg=COLORS["piece_dark"],
                    anchor="center",
                    cursor="hand2",
                )
                square.pack(fill="both", expand=True)
                square.bind("<Button-1>", lambda event, r=r, c=c: self.on_click(r, c))
                cell.bind("<Button-1>", lambda event, r=r, c=c: self.on_click(r, c))
                self.buttons[r][c] = square

        for c in range(8):
            container.grid_columnconfigure(c + 1, minsize=BOARD_SQUARE_SIZE)
        for r in range(8):
            container.grid_rowconfigure(r + 1, minsize=BOARD_SQUARE_SIZE)

        self.frame = container

    def create_board_info_bar(self, parent):
        info_bar = tk.Frame(
            parent,
            bg=COLORS["info_bar_bg"],
            padx=14,
            pady=12,
            highlightbackground="#d4c3ae",
            highlightthickness=1,
        )
        info_bar.pack(fill="x", pady=(14, 0))

        top_row = tk.Frame(info_bar, bg=COLORS["info_bar_bg"])
        top_row.pack(fill="x")

        tk.Label(
            top_row,
            text="Selected:",
            font=CARD_TITLE_FONT,
            bg=COLORS["info_bar_bg"],
            fg=COLORS["accent"],
        ).pack(side="left")

        tk.Label(
            top_row,
            textvariable=self.selection_var,
            font=TEXT_FONT,
            bg=COLORS["info_bar_bg"],
            fg=COLORS["text"],
        ).pack(side="left", padx=(8, 18))

        tk.Label(
            top_row,
            text="Legal moves:",
            font=CARD_TITLE_FONT,
            bg=COLORS["info_bar_bg"],
            fg=COLORS["accent"],
        ).pack(side="left")

        tk.Label(
            top_row,
            textvariable=self.legal_moves_var,
            font=TEXT_FONT,
            bg=COLORS["info_bar_bg"],
            fg=COLORS["text"],
            anchor="w",
            justify="left",
        ).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def create_sidebar(self, parent):
        header = tk.Frame(parent, bg=COLORS["app_bg"])
        header.pack(fill="x", pady=(0, 18))

        tk.Label(
            header,
            text="Game Details",
            font=TITLE_FONT,
            bg=COLORS["app_bg"],
            fg=COLORS["accent"],
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Track turns, moves, and the current engine matchup.",
            font=SUBTITLE_FONT,
            bg=COLORS["app_bg"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(6, 0))

        self.mode_var.set("")
        self.turn_var.set("")
        self.status_var.set("")
        self.last_move_var.set("")
        self.move_count_var.set("")
        info_card = self.create_card(parent, "Overview")
        self.create_info_row(info_card, "Mode", self.mode_var)
        self.create_info_row(info_card, "Turn", self.turn_var)
        self.create_info_row(info_card, "Status", self.status_var)
        self.create_info_row(info_card, "Last move", self.last_move_var)
        self.create_info_row(info_card, "Move count", self.move_count_var)

        selection_card = self.create_card(parent, "Selected Piece")
        self.create_info_row(selection_card, "Square", self.selection_var)

        tk.Label(
            selection_card,
            text="Legal moves",
            font=TEXT_FONT,
            bg=COLORS["card_bg"],
            fg=COLORS["muted"],
            anchor="w",
        ).pack(fill="x", pady=(8, 2))

        tk.Label(
            selection_card,
            textvariable=self.legal_moves_var,
            font=TEXT_FONT,
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            anchor="w",
            justify="left",
            wraplength=300,
        ).pack(fill="x")

        tips_card = self.create_card(parent, "How to Play")
        tips = [
            "Click a piece to see legal destinations.",
            "Yellow shows your selected square.",
            "Blue highlights legal moves.",
            "Red marks the king in check.",
        ]

        for tip in tips:
            tk.Label(
                tips_card,
                text=tip,
                font=TEXT_FONT,
                bg=COLORS["card_bg"],
                fg=COLORS["text"],
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=2)

        log_card = self.create_card(parent, "Move Log")

        self.move_log = tk.Text(
            log_card,
            height=18,
            width=34,
            wrap="word",
            font=TEXT_FONT,
            bg="#fffdf9",
            fg=COLORS["move_log_text"],
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
        )
        self.move_log.pack(fill="both", expand=True)

    def create_card(self, parent, title):
        card = tk.Frame(
            parent,
            bg=COLORS["card_bg"],
            padx=16,
            pady=14,
            highlightbackground="#d4c3ae",
            highlightthickness=1,
        )
        card.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text=title,
            font=CARD_TITLE_FONT,
            bg=COLORS["card_bg"],
            fg=COLORS["accent"],
        ).pack(anchor="w", pady=(0, 10))

        return card

    def create_info_row(self, parent, label, variable):
        row = tk.Frame(parent, bg=COLORS["card_bg"])
        row.pack(fill="x", pady=3)

        tk.Label(
            row,
            text=label,
            font=TEXT_FONT,
            bg=COLORS["card_bg"],
            fg=COLORS["muted"],
            width=10,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            row,
            textvariable=variable,
            font=TEXT_FONT,
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            anchor="w",
            justify="left",
        ).pack(side="left", fill="x", expand=True)

    def update_board(self):
        if (
            not self.root.winfo_exists()
            or self.buttons[0][0] is None
            or not self.buttons[0][0].winfo_exists()
        ):
            return

        board = self.game_state.board_to_array()

        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                text = ""

                if piece != "--":
                    color = piece[0]
                    symbol = piece[1]
                    display_symbol = "P" if symbol == "p" else symbol
                    if color == "b":
                        display_symbol = display_symbol.lower()
                    text = PIECE_UNICODE[display_symbol]

                bg = COLORS["light_square"] if (r + c) % 2 == 0 else COLORS["dark_square"]
                fg = COLORS["piece_light"] if piece.startswith("b") else COLORS["piece_dark"]
                if piece == "--":
                    fg = COLORS["piece_dark"]

                self.buttons[r][c].config(
                    text=text,
                    font=BOARD_FONT,
                    bg=bg,
                    fg=fg,
                )
                self.buttons[r][c].master.config(bg=bg)

        if self.selected:
            r, c = self.selected
            self.buttons[r][c].config(
                bg=COLORS["selected"],
                fg=COLORS["piece_dark"],
            )
            self.buttons[r][c].master.config(bg=COLORS["selected"])

        for r, c in self.legal_moves:
            self.buttons[r][c].config(
                bg=COLORS["legal_move"],
                fg=COLORS["piece_dark"],
            )
            self.buttons[r][c].master.config(bg=COLORS["legal_move"])

        ck = self.game_state.king_in_check_rc()
        if ck:
            r, c = ck
            self.buttons[r][c].config(
                bg=COLORS["check"],
                fg="white",
            )
            self.buttons[r][c].master.config(bg=COLORS["check"])

        self.update_sidebar()

    def update_sidebar(self):
        current_turn = "White" if self.game_state.board.turn == chess.WHITE else "Black"
        self.mode_var.set(MODE_LABELS.get(self.mode, "Not selected"))
        self.turn_var.set(current_turn)
        self.status_var.set(self.describe_status())
        self.last_move_var.set(self.game_state.moveLog[-1] if self.game_state.moveLog else "No moves yet")
        self.move_count_var.set(str(len(self.game_state.moveLog)))
        self.update_selection_panel()
        self.refresh_move_log()

    def update_selection_panel(self):
        if self.selected is None:
            self.selection_var.set("No piece selected")
            self.legal_moves_var.set("Click a piece to preview its legal moves.")
            return

        square_name = self.game_state.rc_to_algebraic(self.selected)
        piece = self.game_state.board_to_array()[self.selected[0]][self.selected[1]]
        piece_name = self.describe_piece(piece)
        self.selection_var.set(f"{square_name} ({piece_name})")

        if not self.legal_moves:
            self.legal_moves_var.set("No legal moves available from this square.")
            return

        move_names = [self.game_state.rc_to_algebraic(move) for move in self.legal_moves]
        self.legal_moves_var.set(", ".join(move_names))

    def describe_piece(self, piece_code):
        if piece_code == "--":
            return "empty"

        color = "White" if piece_code[0] == "w" else "Black"
        names = {
            "p": "pawn",
            "R": "rook",
            "N": "knight",
            "B": "bishop",
            "Q": "queen",
            "K": "king",
        }
        return f"{color} {names.get(piece_code[1], 'piece')}"

    def refresh_move_log(self):
        if not hasattr(self, "move_log"):
            return

        self.move_log.config(state="normal")
        self.move_log.delete("1.0", tk.END)

        moves = self.game_state.moveLog
        if not moves:
            self.move_log.insert(tk.END, "The move log will appear here once the game begins.")
        else:
            for move_index in range(0, len(moves), 2):
                turn_number = (move_index // 2) + 1
                white_move = moves[move_index]
                black_move = moves[move_index + 1] if move_index + 1 < len(moves) else ""
                line = f"{turn_number}. {white_move}"
                if black_move:
                    line += f"   {black_move}"
                self.move_log.insert(tk.END, line + "\n")

        self.move_log.config(state="disabled")

    def describe_status(self):
        if self.game_state.is_checkmate():
            return self.game_state.game_result()
        if self.game_state.is_stalemate():
            return "Stalemate"
        if self.game_state.is_check():
            return "Check"
        return "In progress"

    # Player input
    def on_click(self, r, c):
        if self.game_state.is_game_over():
            self.show_game_over()
            return

        turn = self.game_state.board.turn == CHESS_TURN

        if self.mode in ["mvm", "rvm"]:
            return
        if self.mode == "pvr" and not turn:
            return
        if self.mode == "pvm" and not turn:
            return
        if self.mode == "pvl" and not turn:
            return

        if self.selected is None:
            moves = self.game_state.legal_moves_from(r, c)
            if moves:
                self.selected = (r, c)
                self.legal_moves = moves
            else:
                self.selected = None
                self.legal_moves = []
        else:
            moved = self.game_state.make_move(self.selected, (r, c))
            if moved:
                print("Player move:", self.game_state.moveLog[-1])
            if moved:
                self.selected = None
                self.legal_moves = []
            else:
                moves = self.game_state.legal_moves_from(r, c)
                if moves:
                    self.selected = (r, c)
                    self.legal_moves = moves
                else:
                    self.selected = None
                    self.legal_moves = []

        self.update_board()

    # Game over
    def show_game_over(self):
        result = self.game_state.game_result()
        messagebox.showinfo("Game Over", result)

    # AI loop
    def ai_loop(self):
        if self.game_state.is_game_over():
            self.show_game_over()
            return

        turn = self.game_state.board.turn == CHESS_TURN
        ai_type = None

        if self.mode == "pvr":
            ai_type = "random" if not turn else None
        elif self.mode == "pvm":
            ai_type = "minimax" if not turn else None
        elif self.mode == "rvm":
            ai_type = "random" if turn else "minimax"
        elif self.mode == "mvm":
            ai_type = "minimax"
        elif self.mode == "pvl":
            ai_type = "ml" if not turn else None
        elif self.mode == "mlr":
            ai_type = "ml" if turn else "random"

        if ai_type:
            if ai_type == "random":
                mv = solver.random_move(self.game_state.board)
            elif ai_type == "minimax":
                mv = solver.find_best_move(self.game_state.board, MINIMAX_DEPTH)
            elif ai_type == "ml":
                mv = ml_solver.ml_move(self.game_state.board, self.ml_model)

            if mv:
                self.game_state.apply_move_obj(mv)
                print(f"AI ({ai_type}):", mv.uci())

        self.update_board()
        self.schedule_ai_loop()


def main():
    root = tk.Tk()
    ChessUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
