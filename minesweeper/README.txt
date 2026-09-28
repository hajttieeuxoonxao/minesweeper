MINESWEEPER - CO3061 Introduction to Artificial Intelligence
=============================================================

HOW TO RUN:
    python minesweeper.py

REQUIREMENTS:
    - Python 3.x
    - tkinter (included with Python by default)

CONTROLS:
    - Left click:  Reveal a cell
    - Right click: Flag/unflag a cell (mark suspected mine)
    - "New Game":  Start a new game
    - "AI: One Step": Let AI make one move
    - "AI: Auto Solve": Let AI solve automatically (step by step)

GRID SIZES:
    - 5x5 with 4 mines (default)
    - 9x9 with 10 mines

ALGORITHMS USED:
    1. BFS (Breadth-First Search): Used to expand blank cells (cells with 0
       neighboring mines). When a blank cell is revealed, BFS explores all
       connected blank cells and their numbered borders.

    2. AI Solver with Heuristic:
       - Strategy 1 (Certain Safe): If a number cell already has enough flags,
         remaining unrevealed neighbors are safe -> reveal them.
       - Strategy 2 (Certain Mine): If a number cell's remaining mine count
         equals its unrevealed neighbor count -> flag them all.
       - Strategy 3 (Probability Heuristic): When no certain move exists,
         estimate mine probability for each border cell using:
           h(cell) = max( (number - flagged) / unrevealed_count )
         Pick the cell with the LOWEST probability to reveal (safest choice).
