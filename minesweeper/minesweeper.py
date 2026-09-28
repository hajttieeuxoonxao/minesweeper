"""
Minesweeper Game with AI Solver
Course: CO3061 - Introduction to Artificial Intelligence
HCMUT - VNU HCM

Features:
- GUI using tkinter
- Grid sizes: 5x5 (default), 9x9
- BFS for expanding blank cells
- AI Solver using heuristic-based approach (single-cell constraint + probability)
"""

import tkinter as tk
from tkinter import messagebox
import random
from collections import deque


# =============================================================================
# GAME LOGIC
# =============================================================================

class MinesweeperGame:
    """Core game logic, independent of GUI."""

    def __init__(self, rows, cols, num_mines):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.board = [[0] * cols for _ in range(rows)]       # number hints
        self.mines = [[False] * cols for _ in range(rows)]   # mine locations
        self.revealed = [[False] * cols for _ in range(rows)]
        self.flagged = [[False] * cols for _ in range(rows)]
        self.game_over = False
        self.win = False
        self.first_click = True
        self.mines_placed = False

    # ---- Setup ----

    def place_mines(self, safe_row, safe_col):
        """Place mines randomly, ensuring first click cell and its neighbors are safe."""
        safe_zone = set()
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = safe_row + dr, safe_col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    safe_zone.add((r, c))

        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in safe_zone:
                    candidates.append((r, c))

        # If not enough candidates, allow some safe zone cells (except the click itself)
        if len(candidates) < self.num_mines:
            extra = [(r, c) for (r, c) in safe_zone if (r, c) != (safe_row, safe_col)]
            candidates.extend(extra)

        mine_positions = random.sample(candidates, min(self.num_mines, len(candidates)))
        for r, c in mine_positions:
            self.mines[r][c] = True

        self._calculate_numbers()
        self.mines_placed = True

    def _calculate_numbers(self):
        """Calculate the number hints for each cell (count of neighboring mines)."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mines[r][c]:
                    self.board[r][c] = -1
                else:
                    count = 0
                    for nr, nc in self.get_neighbors(r, c):
                        if self.mines[nr][nc]:
                            count += 1
                    self.board[r][c] = count

    # ---- Helpers ----

    def get_neighbors(self, r, c):
        """Return list of valid neighbor coordinates (8 directions)."""
        neighbors = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    # ---- Actions ----

    def reveal(self, r, c):
        """
        Reveal a cell. If it's a blank (0), use BFS to expand all connected blank cells.
        Returns list of newly revealed cells.
        """
        if self.game_over or self.revealed[r][c] or self.flagged[r][c]:
            return []

        if self.first_click:
            self.place_mines(r, c)
            self.first_click = False

        if self.mines[r][c]:
            self.game_over = True
            self.win = False
            self.revealed[r][c] = True
            return [(r, c)]

        # BFS to expand blank cells
        newly_revealed = []
        queue = deque()
        queue.append((r, c))
        self.revealed[r][c] = True
        newly_revealed.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            if self.board[cr][cc] == 0:
                # Expand neighbors for blank cell
                for nr, nc in self.get_neighbors(cr, cc):
                    if not self.revealed[nr][nc] and not self.mines[nr][nc]:
                        self.revealed[nr][nc] = True
                        newly_revealed.append((nr, nc))
                        if self.board[nr][nc] == 0:
                            queue.append((nr, nc))

        self._check_win()
        return newly_revealed

    def toggle_flag(self, r, c):
        """Toggle flag on a cell."""
        if self.game_over or self.revealed[r][c]:
            return
        self.flagged[r][c] = not self.flagged[r][c]

    def _check_win(self):
        """Check if all non-mine cells are revealed."""
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.mines[r][c] and not self.revealed[r][c]:
                    return
        self.game_over = True
        self.win = True

    # ---- State queries for AI ----

    def get_unrevealed_neighbors(self, r, c):
        """Return unrevealed, unflagged neighbors of (r, c)."""
        return [(nr, nc) for nr, nc in self.get_neighbors(r, c)
                if not self.revealed[nr][nc] and not self.flagged[nr][nc]]

    def get_flagged_neighbor_count(self, r, c):
        """Count flagged neighbors of (r, c)."""
        return sum(1 for nr, nc in self.get_neighbors(r, c) if self.flagged[nr][nc])

    def count_remaining_mines(self):
        """Return number of mines not yet flagged."""
        flagged = sum(self.flagged[r][c] for r in range(self.rows) for c in range(self.cols))
        return self.num_mines - flagged


# =============================================================================
# AI SOLVER (Heuristic-based using constraint propagation)
# =============================================================================

class MinesweeperAI:
    """
    AI Solver using heuristic strategies:

    Strategy 1 - Certain Safe: If a revealed number cell has exactly enough flags
                 around it, all remaining unrevealed neighbors are safe.

    Strategy 2 - Certain Mine: If a revealed number cell has (number - flagged) ==
                 unrevealed neighbor count, all unrevealed neighbors are mines.

    Strategy 3 - Probability Heuristic: When no certain move exists, pick the cell
                 with the lowest probability of being a mine, estimated from
                 neighboring constraints.

    This uses BFS/DFS-style iteration over revealed cells to propagate constraints.
    """

    def __init__(self, game):
        self.game = game

    def get_next_move(self):
        """
        Returns (action, r, c) where action is 'reveal' or 'flag'.
        Returns None if no move can be determined.
        """
        # Strategy 1 & 2: Constraint-based certain moves
        move = self._find_certain_move()
        if move:
            return move

        # Strategy 3: Probability heuristic
        move = self._find_heuristic_move()
        if move:
            return move

        return None

    def _find_certain_move(self):
        """
        Scan all revealed numbered cells using BFS order.
        Check if any cell gives certain information about its neighbors.
        """
        game = self.game

        # BFS over all revealed cells to find constraints
        for r in range(game.rows):
            for c in range(game.cols):
                if not game.revealed[r][c] or game.board[r][c] <= 0:
                    continue

                number = game.board[r][c]
                flagged = game.get_flagged_neighbor_count(r, c)
                unrevealed = game.get_unrevealed_neighbors(r, c)

                if not unrevealed:
                    continue

                remaining_mines = number - flagged

                # Strategy 2: All unrevealed neighbors must be mines
                if remaining_mines == len(unrevealed):
                    for nr, nc in unrevealed:
                        return ('flag', nr, nc)

                # Strategy 1: All mines found, remaining are safe
                if remaining_mines == 0:
                    for nr, nc in unrevealed:
                        return ('reveal', nr, nc)

        return None

    def _find_heuristic_move(self):
        """
        Probability-based heuristic:
        For each unrevealed cell adjacent to a revealed number, estimate
        the probability of being a mine based on neighboring constraints.

        h(cell) = max over all adjacent revealed numbers of:
                  (number - flagged_count) / unrevealed_count

        Pick the cell with the LOWEST mine probability to reveal.
        This is our heuristic function - it estimates the "danger" of each cell.
        """
        game = self.game
        mine_prob = {}  # (r, c) -> estimated mine probability

        for r in range(game.rows):
            for c in range(game.cols):
                if not game.revealed[r][c] or game.board[r][c] <= 0:
                    continue

                number = game.board[r][c]
                flagged = game.get_flagged_neighbor_count(r, c)
                unrevealed = game.get_unrevealed_neighbors(r, c)

                if not unrevealed:
                    continue

                remaining_mines = number - flagged
                if remaining_mines < 0:
                    continue

                prob = remaining_mines / len(unrevealed)

                for nr, nc in unrevealed:
                    # Take the maximum probability from all constraints
                    if (nr, nc) not in mine_prob:
                        mine_prob[(nr, nc)] = prob
                    else:
                        mine_prob[(nr, nc)] = max(mine_prob[(nr, nc)], prob)

        if mine_prob:
            # Choose cell with lowest mine probability (safest choice)
            best_cell = min(mine_prob, key=mine_prob.get)
            return ('reveal', best_cell[0], best_cell[1])

        # No border cells found - pick any unrevealed cell (random)
        unrevealed_cells = []
        for r in range(game.rows):
            for c in range(game.cols):
                if not game.revealed[r][c] and not game.flagged[r][c]:
                    unrevealed_cells.append((r, c))

        if unrevealed_cells:
            cell = random.choice(unrevealed_cells)
            return ('reveal', cell[0], cell[1])

        return None


# =============================================================================
# GUI
# =============================================================================

# Color scheme for number cells
NUMBER_COLORS = {
    1: '#0000FF',  # Blue
    2: '#008000',  # Green
    3: '#FF0000',  # Red
    4: '#000080',  # Dark Blue
    5: '#800000',  # Maroon
    6: '#008080',  # Teal
    7: '#000000',  # Black
    8: '#808080',  # Gray
}

CELL_SIZE = 50
PRESETS = {
    '5x5': (5, 5, 4),
    '9x9': (9, 9, 10),
}


class MinesweeperGUI:
    """Tkinter GUI for Minesweeper with AI integration."""

    def __init__(self, root):
        self.root = root
        self.root.title("Minesweeper - CO3061 AI")
        self.root.resizable(False, False)

        self.rows, self.cols, self.num_mines = PRESETS['5x5']
        self.game = None
        self.ai = None
        self.buttons = []
        self.ai_running = False
        self.ai_step_delay = 500  # ms between AI steps

        self._build_menu()
        self._build_info_bar()
        self._build_grid()
        self._build_controls()
        self.new_game()

    def _build_menu(self):
        """Build the size selection menu."""
        menu_frame = tk.Frame(self.root, bg='#2c3e50', padx=5, pady=5)
        menu_frame.pack(fill=tk.X)

        tk.Label(menu_frame, text="Grid Size:", bg='#2c3e50', fg='white',
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)

        self.size_var = tk.StringVar(value='5x5')
        for name in PRESETS:
            tk.Radiobutton(menu_frame, text=name, variable=self.size_var,
                           value=name, command=self._on_size_change,
                           bg='#2c3e50', fg='white', selectcolor='#34495e',
                           activebackground='#34495e', activeforeground='white',
                           font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

    def _build_info_bar(self):
        """Build mine counter and status display."""
        info_frame = tk.Frame(self.root, bg='#ecf0f1', padx=10, pady=5)
        info_frame.pack(fill=tk.X)

        self.mine_label = tk.Label(info_frame, text="Mines: 0", bg='#ecf0f1',
                                   font=('Arial', 12, 'bold'), fg='#e74c3c')
        self.mine_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(info_frame, text="Click to start!",
                                     bg='#ecf0f1', font=('Arial', 12), fg='#2c3e50')
        self.status_label.pack(side=tk.RIGHT)

    def _build_grid(self):
        """Build the button grid."""
        self.grid_frame = tk.Frame(self.root, bg='#bdc3c7', padx=2, pady=2)
        self.grid_frame.pack(padx=10, pady=5)

    def _build_controls(self):
        """Build control buttons."""
        ctrl_frame = tk.Frame(self.root, bg='#ecf0f1', padx=10, pady=8)
        ctrl_frame.pack(fill=tk.X)

        btn_style = {'font': ('Arial', 10, 'bold'), 'relief': tk.RAISED, 'bd': 2}

        tk.Button(ctrl_frame, text="New Game", command=self.new_game,
                  bg='#3498db', fg='white', **btn_style).pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_frame, text="AI: One Step", command=self.ai_step,
                  bg='#e67e22', fg='white', **btn_style).pack(side=tk.LEFT, padx=5)

        self.ai_auto_btn = tk.Button(ctrl_frame, text="AI: Auto Solve",
                                      command=self.ai_auto_toggle,
                                      bg='#27ae60', fg='white', **btn_style)
        self.ai_auto_btn.pack(side=tk.LEFT, padx=5)

        # Legend
        legend_frame = tk.Frame(self.root, bg='#ecf0f1', padx=10, pady=3)
        legend_frame.pack(fill=tk.X)
        tk.Label(legend_frame, text="Left click: Reveal | Right click: Flag/Unflag",
                 bg='#ecf0f1', font=('Arial', 9), fg='#7f8c8d').pack()

    # ---- Game management ----

    def new_game(self):
        """Start a new game."""
        self.ai_running = False
        preset = PRESETS[self.size_var.get()]
        self.rows, self.cols, self.num_mines = preset

        self.game = MinesweeperGame(self.rows, self.cols, self.num_mines)
        self.ai = MinesweeperAI(self.game)

        self._rebuild_grid()
        self._update_info()
        self.status_label.config(text="Click to start!", fg='#2c3e50')

    def _on_size_change(self):
        self.new_game()

    def _rebuild_grid(self):
        """Destroy and recreate the grid buttons."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.buttons = []
        for r in range(self.rows):
            row_btns = []
            for c in range(self.cols):
                btn = tk.Button(self.grid_frame, width=3, height=1,
                                font=('Consolas', 14, 'bold'),
                                bg='#95a5a6', fg='#2c3e50',
                                activebackground='#7f8c8d',
                                relief=tk.RAISED, bd=2)
                btn.grid(row=r, column=c, padx=1, pady=1)
                btn.bind('<Button-1>', lambda e, row=r, col=c: self._on_left_click(row, col))
                btn.bind('<Button-3>', lambda e, row=r, col=c: self._on_right_click(row, col))
                row_btns.append(btn)
            self.buttons.append(row_btns)

        # Resize window
        self.root.update_idletasks()

    # ---- Click handlers ----

    def _on_left_click(self, r, c):
        if self.game.game_over:
            return
        newly = self.game.reveal(r, c)
        self._update_cells(newly)
        self._check_game_end()

    def _on_right_click(self, r, c):
        if self.game.game_over:
            return
        self.game.toggle_flag(r, c)
        self._update_cell(r, c)
        self._update_info()

    # ---- Display updates ----

    def _update_cells(self, cells):
        for r, c in cells:
            self._update_cell(r, c)
        self._update_info()

    def _update_cell(self, r, c):
        btn = self.buttons[r][c]
        game = self.game

        if game.revealed[r][c]:
            if game.mines[r][c]:
                btn.config(text='*', bg='#e74c3c', fg='white',
                           relief=tk.SUNKEN, state=tk.DISABLED)
            elif game.board[r][c] == 0:
                btn.config(text='', bg='#ecf0f1', relief=tk.SUNKEN,
                           state=tk.DISABLED)
            else:
                num = game.board[r][c]
                color = NUMBER_COLORS.get(num, '#000000')
                btn.config(text=str(num), bg='#ecf0f1', fg=color,
                           relief=tk.SUNKEN, state=tk.DISABLED)
        elif game.flagged[r][c]:
            btn.config(text='F', bg='#f39c12', fg='white',
                       relief=tk.RAISED, state=tk.NORMAL)
        else:
            btn.config(text='', bg='#95a5a6', fg='#2c3e50',
                       relief=tk.RAISED, state=tk.NORMAL)

    def _update_info(self):
        remaining = self.game.count_remaining_mines()
        self.mine_label.config(text=f"Mines: {remaining}")

    def _check_game_end(self):
        if self.game.game_over:
            self.ai_running = False
            if self.game.win:
                self.status_label.config(text="YOU WIN!", fg='#27ae60')
                self._reveal_all()
                messagebox.showinfo("Minesweeper", "Congratulations! You win!")
            else:
                self.status_label.config(text="GAME OVER!", fg='#e74c3c')
                self._reveal_all()
                messagebox.showinfo("Minesweeper", "Boom! Game Over!")

    def _reveal_all(self):
        """Show all cells at game end."""
        for r in range(self.rows):
            for c in range(self.cols):
                btn = self.buttons[r][c]
                if self.game.mines[r][c]:
                    if self.game.flagged[r][c]:
                        btn.config(text='F', bg='#27ae60', fg='white')
                    else:
                        btn.config(text='*', bg='#e74c3c', fg='white',
                                   relief=tk.SUNKEN)
                elif self.game.flagged[r][c]:
                    # Wrong flag
                    btn.config(text='X', bg='#e67e22', fg='white')
                elif not self.game.revealed[r][c]:
                    if self.game.board[r][c] == 0:
                        btn.config(text='', bg='#ecf0f1', relief=tk.SUNKEN)
                    else:
                        num = self.game.board[r][c]
                        color = NUMBER_COLORS.get(num, '#000000')
                        btn.config(text=str(num), bg='#ecf0f1', fg=color,
                                   relief=tk.SUNKEN)

    # ---- AI Controls ----

    def ai_step(self):
        """Execute one AI move."""
        if self.game.game_over:
            return

        if self.game.first_click:
            # AI makes first click at center
            r, c = self.rows // 2, self.cols // 2
            newly = self.game.reveal(r, c)
            self._update_cells(newly)
            self._check_game_end()
            self.status_label.config(text="AI: revealed center", fg='#e67e22')
            return

        move = self.ai.get_next_move()
        if move is None:
            self.status_label.config(text="AI: no move found", fg='#e67e22')
            return

        action, r, c = move
        if action == 'flag':
            self.game.toggle_flag(r, c)
            self._update_cell(r, c)
            self._update_info()
            self.status_label.config(text=f"AI: flag ({r},{c})", fg='#e67e22')
        elif action == 'reveal':
            newly = self.game.reveal(r, c)
            self._update_cells(newly)
            self._check_game_end()
            self.status_label.config(text=f"AI: reveal ({r},{c})", fg='#e67e22')

    def ai_auto_toggle(self):
        """Toggle AI auto-solve mode."""
        if self.ai_running:
            self.ai_running = False
            self.ai_auto_btn.config(text="AI: Auto Solve", bg='#27ae60')
            self.status_label.config(text="AI stopped", fg='#2c3e50')
        else:
            self.ai_running = True
            self.ai_auto_btn.config(text="Stop AI", bg='#c0392b')
            self._ai_auto_step()

    def _ai_auto_step(self):
        """Auto step: perform one AI move, then schedule next."""
        if not self.ai_running or self.game.game_over:
            self.ai_running = False
            self.ai_auto_btn.config(text="AI: Auto Solve", bg='#27ae60')
            return

        self.ai_step()

        if not self.game.game_over and self.ai_running:
            self.root.after(self.ai_step_delay, self._ai_auto_step)
        else:
            self.ai_running = False
            self.ai_auto_btn.config(text="AI: Auto Solve", bg='#27ae60')


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    MinesweeperGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
