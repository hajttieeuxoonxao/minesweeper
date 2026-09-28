/* ==========================================================================
   Minesweeper Game with AI Solver - Web version
   Course: CO3061 - Introduction to Artificial Intelligence
   HCMUT - VNU HCM

   This file is a direct port of minesweeper.py:
       MinesweeperGame  -> class MinesweeperGame   (game rules, BFS reveal)
       MinesweeperAI    -> class MinesweeperAI     (heuristic solver)
       MinesweeperGUI   -> the UI section at the bottom of this file

   Features:
   - Grid sizes: 5x5 (default), 9x9, 16x16 and a Custom board
     (rows 5-16, columns 5-16, mines 4-40)
   - Game timer (starts on the first reveal, stops on win/lose)
   - BFS for expanding blank cells
   - AI Solver: global mine-count check, single-cell constraints
     (Strategy 1/2), bounded local-group solving (Strategy 2.5) and a
     probability heuristic (Strategy 3)
   - No-guess board generation (headless logic simulation on first click)
   ========================================================================== */

'use strict';

// =============================================================================
// CONFIGURATION
// =============================================================================

// Fixed board presets: name -> [rows, cols, mines]
const PRESETS = {
    '5x5':   [5, 5, 4],
    '9x9':   [9, 9, 10],
    '16x16': [16, 16, 40],
};

const CUSTOM_LABEL = 'Custom';

// Allowed ranges for the Custom board (per assignment specification)
const MIN_ROWS = 5,  MAX_ROWS = 16;
const MIN_COLS = 5,  MAX_COLS = 16;
const MIN_MINES = 4, MAX_MINES = 40;

// The first click and its 8 neighbours are always safe, so a board must keep
// at least this many mine-free cells available.
const SAFE_ZONE_CELLS = 9;

const DEFAULT_CUSTOM = [10, 10, 20];

// ---- AI solver settings ----
// Strategy 2.5 only enumerates a frontier component when it has at most this
// many unrevealed cells; larger components fall through to Strategy 3.
const MAX_GROUP_SIZE = 20;

// ---- No-guess board generation (JavaScript budget) ----
// Generation stops at whichever limit is reached first. The Python version
// uses its own, larger time budget and fewer attempts (CPython is slower).
const NO_GUESS_MAX_ATTEMPTS = 100;
const NO_GUESS_TIME_BUDGET_MS = 30;

// Two probabilities closer than this are treated as equal (tie-breaking).
const PROB_EPSILON = 1e-9;

/**
 * Largest legal mine count for a board of this size.
 * Capped by MAX_MINES and by the need to keep the first-click safe zone
 * (the clicked cell plus its 8 neighbours) free of mines.
 */
function maxMinesFor(rows, cols) {
    return Math.max(MIN_MINES, Math.min(MAX_MINES, rows * cols - SAFE_ZONE_CELLS));
}

/** Create a rows x cols 2D array filled with `value`. */
function makeGrid(rows, cols, value) {
    return Array.from({ length: rows }, () => new Array(cols).fill(value));
}

/** Pick `k` distinct random items from `items` (equivalent of random.sample). */
function randomSample(items, k) {
    const pool = items.slice();
    for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
}

// Neighbour lists are the same for every board of a given size, so they are
// computed once and shared (the no-guess generator builds many boards).
const neighborTables = new Map();

/** Return a rows x cols grid whose entries are the [r, c] neighbour lists. */
function neighborTable(rows, cols) {
    const key = `${rows}x${cols}`;
    let table = neighborTables.get(key);
    if (!table) {
        table = makeGrid(rows, cols, null);
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const list = [];
                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        if (dr === 0 && dc === 0) continue;
                        const nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) list.push([nr, nc]);
                    }
                }
                table[r][c] = list;
            }
        }
        neighborTables.set(key, table);
    }
    return table;
}


// =============================================================================
// GAME LOGIC
// =============================================================================

class MinesweeperGame {
    /** Core game logic, independent of the interface. */

    constructor(rows, cols, numMines) {
        this.rows = rows;
        this.cols = cols;
        this.numMines = numMines;
        this.board = makeGrid(rows, cols, 0);          // number hints
        this.mines = makeGrid(rows, cols, false);      // mine locations
        this.revealed = makeGrid(rows, cols, false);
        this.flagged = makeGrid(rows, cols, false);
        this.gameOver = false;
        this.win = false;
        this.firstClick = true;
        this.minesPlaced = false;
        this.neighbors = neighborTable(rows, cols);
        // Set by placeMines: true = verified no-guess board, false = fallback
        // (best-progress) board, null = mines not placed yet.
        this.guessFree = null;
        this.generationAttempts = 0;
    }

    // ---- Setup ----

    /**
     * Place mines, ensuring the first click cell and its neighbours are safe.
     *
     * No-guess generation: each random candidate layout is checked by a
     * headless logic simulation (simulateLogicSolve). The first candidate that
     * pure logic can clear completely is accepted. If none is found within
     * NO_GUESS_MAX_ATTEMPTS / NO_GUESS_TIME_BUDGET_MS, the candidate that let
     * logic reveal the most safe cells is used and guessFree is set to false.
     */
    placeMines(safeRow, safeCol) {
        const safeZone = new Set();
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                const r = safeRow + dr, c = safeCol + dc;
                if (r >= 0 && r < this.rows && c >= 0 && c < this.cols) {
                    safeZone.add(`${r},${c}`);
                }
            }
        }

        const candidates = [];
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (!safeZone.has(`${r},${c}`)) candidates.push([r, c]);
            }
        }

        // If not enough candidates, allow some safe zone cells (except the click itself)
        if (candidates.length < this.numMines) {
            for (const key of safeZone) {
                const [r, c] = key.split(',').map(Number);
                if (r !== safeRow || c !== safeCol) candidates.push([r, c]);
            }
        }

        const mineCount = Math.min(this.numMines, candidates.length);
        const startTime = performance.now();
        let bestPositions = null, bestProgress = -1, solved = false, attempts = 0;

        while (attempts < NO_GUESS_MAX_ATTEMPTS) {
            attempts++;
            const minePositions = randomSample(candidates, mineCount);
            const result = simulateLogicSolve(this.rows, this.cols, mineCount,
                                              minePositions, safeRow, safeCol);
            if (result.solved) {
                bestPositions = minePositions;
                solved = true;
                break;
            }
            // Remember the candidate on which logic made the most progress.
            if (result.progress > bestProgress) {
                bestProgress = result.progress;
                bestPositions = minePositions;
            }
            if (performance.now() - startTime >= NO_GUESS_TIME_BUDGET_MS) break;
        }

        for (const [r, c] of bestPositions) this.mines[r][c] = true;

        this.calculateNumbers();
        this.minesPlaced = true;
        this.guessFree = solved;
        this.generationAttempts = attempts;
    }

    /** Calculate the number hints for each cell (count of neighbouring mines). */
    calculateNumbers() {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (this.mines[r][c]) {
                    this.board[r][c] = -1;
                } else {
                    let count = 0;
                    for (const [nr, nc] of this.getNeighbors(r, c)) {
                        if (this.mines[nr][nc]) count++;
                    }
                    this.board[r][c] = count;
                }
            }
        }
    }

    // ---- Helpers ----

    /**
     * Return list of valid neighbour coordinates (8 directions).
     * The list comes from the shared neighbour table: read it, don't modify it.
     */
    getNeighbors(r, c) {
        return this.neighbors[r][c];
    }

    // ---- Actions ----

    /**
     * Reveal a cell. If it is a blank (0), use BFS to expand all connected
     * blank cells. Returns the list of newly revealed cells.
     */
    reveal(r, c) {
        if (this.gameOver || this.revealed[r][c] || this.flagged[r][c]) return [];

        if (this.firstClick) {
            this.placeMines(r, c);
            this.firstClick = false;
        }

        if (this.mines[r][c]) {
            this.gameOver = true;
            this.win = false;
            this.revealed[r][c] = true;
            return [[r, c]];
        }

        // BFS to expand blank cells
        const newlyRevealed = [];
        const queue = [];               // used as a FIFO queue (push / shift)
        let head = 0;                   // read index, avoids shift() cost

        queue.push([r, c]);
        this.revealed[r][c] = true;
        newlyRevealed.push([r, c]);

        while (head < queue.length) {
            const [cr, cc] = queue[head++];
            if (this.board[cr][cc] === 0) {
                // Expand neighbours for a blank cell
                for (const [nr, nc] of this.getNeighbors(cr, cc)) {
                    if (!this.revealed[nr][nc] && !this.mines[nr][nc]) {
                        this.revealed[nr][nc] = true;
                        newlyRevealed.push([nr, nc]);
                        if (this.board[nr][nc] === 0) queue.push([nr, nc]);
                    }
                }
            }
        }

        this.checkWin();
        return newlyRevealed;
    }

    /** Toggle flag on a cell. */
    toggleFlag(r, c) {
        if (this.gameOver || this.revealed[r][c]) return;
        this.flagged[r][c] = !this.flagged[r][c];
    }

    /** Check if all non-mine cells are revealed. */
    checkWin() {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (!this.mines[r][c] && !this.revealed[r][c]) return;
            }
        }
        this.gameOver = true;
        this.win = true;
    }

    // ---- State queries for AI ----

    /** Return unrevealed, unflagged neighbours of (r, c). */
    getUnrevealedNeighbors(r, c) {
        return this.getNeighbors(r, c)
            .filter(([nr, nc]) => !this.revealed[nr][nc] && !this.flagged[nr][nc]);
    }

    /** Count flagged neighbours of (r, c). */
    getFlaggedNeighborCount(r, c) {
        return this.getNeighbors(r, c).filter(([nr, nc]) => this.flagged[nr][nc]).length;
    }

    /** Return number of mines not yet flagged. */
    countRemainingMines() {
        let flagged = 0;
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (this.flagged[r][c]) flagged++;
            }
        }
        return this.numMines - flagged;
    }
}


// =============================================================================
// AI SOLVER (Heuristic-based using constraint propagation)
// =============================================================================

class MinesweeperAI {
    /**
     * AI Solver. Every turn the strategies below are tried in order, and the
     * first one that produces a move wins:
     *
     * Global check - Mine count: if no unflagged mines remain, every
     *              unrevealed cell is safe; if the remaining mines equal the
     *              number of unrevealed cells, every unrevealed cell is a mine.
     *
     * Strategy 1 - Certain Safe: If a revealed number cell has exactly enough
     *              flags around it, all remaining unrevealed neighbours are safe.
     *
     * Strategy 2 - Certain Mine: If a revealed number cell has
     *              (number - flagged) == unrevealed neighbour count, all
     *              unrevealed neighbours are mines.
     *
     * Strategy 2.5 - Bounded Local-Group Solving: group the frontier into
     *              connected components (numbers that share unrevealed
     *              neighbours). For each component with at most MAX_GROUP_SIZE
     *              cells, enumerate every mine assignment that satisfies all of
     *              its numbers. A cell that is safe (or a mine) in every valid
     *              assignment is certain. Only one component is solved at a
     *              time - this is NOT a full-board CSP.
     *
     * Strategy 3 - Probability Heuristic: When no certain move exists, pick the
     *              cell with the lowest probability of being a mine (exact
     *              probability from Strategy 2.5 where available, otherwise an
     *              estimate).
     */

    constructor(game) {
        this.game = game;
        // Certain moves proven by the last Strategy 2.5 pass, not played yet.
        this.pendingMoves = [];
    }

    /**
     * Returns [action, r, c] where action is 'reveal' or 'flag'.
     * Returns null if no move can be determined.
     */
    getNextMove() {
        // Finish the moves already proven by the last Strategy 2.5 pass.
        // (Re-deriving them later could fail: a new number may merge two
        // components into one larger than MAX_GROUP_SIZE.)
        while (this.pendingMoves.length > 0) {
            const move = this.pendingMoves.shift();
            const [, r, c] = move;
            if (!this.game.revealed[r][c] && !this.game.flagged[r][c]) return move;
        }

        // Global mine-count check
        const globalMoves = this.findGlobalMoves();
        if (globalMoves.length > 0) return globalMoves[0];

        // Strategy 1 & 2: Constraint-based certain moves
        const move = this.findCertainMove();
        if (move) return move;

        // Strategy 2.5: Bounded local-group solving. All certain moves of the
        // pass are deduced together; one is played now, the rest are queued
        // and played one per AI step.
        const analysis = this.analyzeGroups();
        if (analysis.moves.length > 0) {
            this.pendingMoves = analysis.moves.slice(1);
            return analysis.moves[0];
        }

        // Strategy 3: Probability heuristic (last resort - a real guess)
        return this.findHeuristicMove(analysis);
    }

    /** List of unrevealed, unflagged cells on the whole board. */
    getUnknownCells() {
        const game = this.game;
        const cells = [];
        for (let r = 0; r < game.rows; r++) {
            for (let c = 0; c < game.cols; c++) {
                if (!game.revealed[r][c] && !game.flagged[r][c]) cells.push([r, c]);
            }
        }
        return cells;
    }

    /**
     * Global mine-count verification.
     *   remaining mines == 0                -> reveal every unknown cell
     *   remaining mines == unknown cells    -> flag every unknown cell
     * Returns a (possibly empty) list of moves.
     */
    findGlobalMoves() {
        const unknown = this.getUnknownCells();
        if (unknown.length === 0) return [];

        const remainingMines = this.game.countRemainingMines();
        if (remainingMines === 0) {
            return unknown.map(([r, c]) => ['reveal', r, c]);
        }
        if (remainingMines === unknown.length) {
            return unknown.map(([r, c]) => ['flag', r, c]);
        }
        return [];
    }

    /**
     * Scan all revealed numbered cells.
     * Check if any cell gives certain information about its neighbours.
     * Returns the first certain move, or null.
     */
    findCertainMove() {
        const moves = this.findCertainMoves();
        return moves.length > 0 ? moves[0] : null;
    }

    /** Strategy 1 & 2 over the whole board: list every single-cell certain move. */
    findCertainMoves() {
        const game = this.game;
        const moves = [];

        for (let r = 0; r < game.rows; r++) {
            for (let c = 0; c < game.cols; c++) {
                if (!game.revealed[r][c] || game.board[r][c] <= 0) continue;

                const number = game.board[r][c];
                const flagged = game.getFlaggedNeighborCount(r, c);
                const unrevealed = game.getUnrevealedNeighbors(r, c);

                if (unrevealed.length === 0) continue;

                const remainingMines = number - flagged;

                // Strategy 2: All unrevealed neighbours must be mines
                if (remainingMines === unrevealed.length) {
                    for (const [nr, nc] of unrevealed) moves.push(['flag', nr, nc]);
                }
                // Strategy 1: All mines found, remaining are safe
                else if (remainingMines === 0) {
                    for (const [nr, nc] of unrevealed) moves.push(['reveal', nr, nc]);
                }
            }
        }

        return moves;
    }

    /**
     * Strategy 2.5 - Bounded local-group solving.
     *
     * 1. Every revealed number with unrevealed neighbours gives a constraint:
     *        (sum of mines over its unrevealed neighbours) == number - flagged
     * 2. Constraints that share an unrevealed cell are joined (union-find) into
     *    frontier components.
     * 3. Each component with <= MAX_GROUP_SIZE cells is enumerated on its own.
     *    Larger components are skipped and handled by Strategy 3.
     *
     * Returns an analysis object:
     *   moves          - certain moves found (reveal safe cells, flag mines)
     *   exactProb      - Map cellIndex -> exact mine probability (enumerated
     *                    components only)
     *   oversized      - Set of cellIndex in components that were not enumerated
     *   confirmedMines - number of cells proven to be mines by this pass
     * A cell index is r * cols + c.
     */
    analyzeGroups() {
        const game = this.game;
        const cols = game.cols;
        const analysis = { moves: [], exactProb: new Map(), oversized: new Set(), confirmedMines: 0 };

        // 1. Collect constraints from revealed numbered cells
        const constraints = [];
        for (let r = 0; r < game.rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (!game.revealed[r][c] || game.board[r][c] <= 0) continue;
                const unrevealed = game.getUnrevealedNeighbors(r, c);
                if (unrevealed.length === 0) continue;
                const need = game.board[r][c] - game.getFlaggedNeighborCount(r, c);
                // Skip contradictory constraints (possible only after wrong manual flags)
                if (need < 0 || need > unrevealed.length) continue;
                constraints.push({ cells: unrevealed.map(([nr, nc]) => nr * cols + nc), need });
            }
        }
        if (constraints.length === 0) return analysis;

        // 2. Union-find: cells of the same constraint belong to the same component
        const parent = new Map();
        const find = (x) => {
            while (parent.get(x) !== x) {
                parent.set(x, parent.get(parent.get(x)));   // path halving
                x = parent.get(x);
            }
            return x;
        };
        for (const con of constraints) {
            for (const cell of con.cells) if (!parent.has(cell)) parent.set(cell, cell);
            const root = find(con.cells[0]);
            for (const cell of con.cells) {
                const other = find(cell);
                if (other !== root) parent.set(other, root);
            }
        }

        const components = new Map();   // root -> { cells, constraints }
        for (const con of constraints) {
            const root = find(con.cells[0]);
            if (!components.has(root)) components.set(root, { cells: [], constraints: [] });
            components.get(root).constraints.push(con);
        }
        for (const cell of parent.keys()) components.get(find(cell)).cells.push(cell);

        // 3. Solve each component independently
        for (const comp of components.values()) {
            if (comp.cells.length > MAX_GROUP_SIZE) {
                for (const cell of comp.cells) analysis.oversized.add(cell);
                continue;
            }

            const { total, mineCounts } = this.enumerateComponent(comp);
            if (total === 0) {
                // No consistent assignment (wrong manual flags): use the fallback estimate.
                for (const cell of comp.cells) analysis.oversized.add(cell);
                continue;
            }

            comp.cells.forEach((cell, i) => {
                const r = Math.floor(cell / cols), c = cell % cols;
                if (mineCounts[i] === 0) {
                    analysis.moves.push(['reveal', r, c]);           // safe in every assignment
                } else if (mineCounts[i] === total) {
                    analysis.moves.push(['flag', r, c]);             // mine in every assignment
                    analysis.confirmedMines++;
                }
                analysis.exactProb.set(cell, mineCounts[i] / total);
            });
        }

        return analysis;
    }

    /**
     * Enumerate every 0/1 mine assignment of ONE component that satisfies all
     * of its constraints.
     *
     * Cells are assigned one at a time; a partial assignment is abandoned as
     * soon as some constraint can no longer be met (too many mines, or not
     * enough cells left to reach its number). This gives exactly the same set
     * of valid assignments as trying all 2^n combinations, only faster.
     *
     * Returns { total, mineCounts } where total is the number of valid
     * assignments and mineCounts[i] is how many of them put a mine on
     * comp.cells[i].
     */
    enumerateComponent(comp) {
        const n = comp.cells.length;
        const indexOf = new Map(comp.cells.map((cell, i) => [cell, i]));

        // Per constraint: required mines, mines placed so far, cells not yet assigned
        const need = comp.constraints.map((con) => con.need);
        const placed = new Array(comp.constraints.length).fill(0);
        const left = comp.constraints.map((con) => con.cells.length);

        // For each cell, the constraints it appears in
        const cellConstraints = Array.from({ length: n }, () => []);
        comp.constraints.forEach((con, k) => {
            for (const cell of con.cells) cellConstraints[indexOf.get(cell)].push(k);
        });

        // Assign cells constraint by constraint, so each constraint is
        // completed (and checked) as early as possible.
        const order = [];
        const seen = new Set();
        for (const con of comp.constraints) {
            for (const cell of con.cells) {
                if (!seen.has(cell)) { seen.add(cell); order.push(indexOf.get(cell)); }
            }
        }

        const value = new Array(n).fill(0);
        const mineCounts = new Array(n).fill(0);
        let total = 0;

        const assign = (depth) => {
            if (depth === n) {
                // Every constraint is fully assigned and satisfied here.
                total++;
                for (let i = 0; i < n; i++) mineCounts[i] += value[i];
                return;
            }
            const v = order[depth];
            for (let bit = 0; bit <= 1; bit++) {
                let ok = true;
                for (const k of cellConstraints[v]) { placed[k] += bit; left[k]--; }
                for (const k of cellConstraints[v]) {
                    if (placed[k] > need[k] || placed[k] + left[k] < need[k]) { ok = false; break; }
                }
                if (ok) {
                    value[v] = bit;
                    assign(depth + 1);
                }
                for (const k of cellConstraints[v]) { placed[k] -= bit; left[k]++; }
            }
            value[v] = 0;
        };
        assign(0);

        return { total, mineCounts };
    }

    /**
     * Strategy 3 - Probability heuristic. Every unknown cell gets a mine
     * probability from one of three sources:
     *
     *   (a) Enumerated component (<= MAX_GROUP_SIZE): exact probability
     *         P = (#valid assignments with a mine on the cell) / (#valid assignments)
     *   (b) Oversized component: local-ratio estimate
     *         P = max over adjacent revealed numbers of
     *             (number - flagged_count) / unrevealed_count
     *   (c) Outer cell (unrevealed, not a neighbour of any revealed number):
     *         P = unaccounted_mines / total_outer_cells
     *         unaccounted_mines = total_mines - flagged - mines confirmed by 2.5
     *
     * The cell with the LOWEST probability is revealed. Ties are broken by the
     * number of unrevealed neighbours (more = bigger chance of a BFS cascade).
     */
    findHeuristicMove(analysis = this.analyzeGroups()) {
        const game = this.game;
        const cols = game.cols;
        const mineProb = new Map();   // cellIndex -> mine probability

        // (a) Exact probabilities from Strategy 2.5
        for (const [cell, prob] of analysis.exactProb) mineProb.set(cell, prob);

        // (b) Local-ratio estimate for cells of oversized components
        if (analysis.oversized.size > 0) {
            for (let r = 0; r < game.rows; r++) {
                for (let c = 0; c < cols; c++) {
                    if (!game.revealed[r][c] || game.board[r][c] <= 0) continue;

                    const unrevealed = game.getUnrevealedNeighbors(r, c);
                    if (unrevealed.length === 0) continue;
                    const remainingMines = game.board[r][c] - game.getFlaggedNeighborCount(r, c);
                    if (remainingMines < 0) continue;

                    const prob = remainingMines / unrevealed.length;
                    for (const [nr, nc] of unrevealed) {
                        const cell = nr * cols + nc;
                        if (!analysis.oversized.has(cell)) continue;
                        // Take the maximum probability from all constraints
                        mineProb.set(cell, mineProb.has(cell) ? Math.max(mineProb.get(cell), prob) : prob);
                    }
                }
            }
        }

        // (c) Outer cells: not a neighbour of any revealed numbered cell
        const outer = this.getUnknownCells().filter(([r, c]) =>
            !game.getNeighbors(r, c).some(([nr, nc]) => game.revealed[nr][nc] && game.board[nr][nc] > 0));
        if (outer.length > 0) {
            const unaccounted = game.countRemainingMines() - analysis.confirmedMines;
            const prob = Math.min(1, Math.max(0, unaccounted / outer.length));
            for (const [r, c] of outer) mineProb.set(r * cols + c, prob);
        }

        // Choose the lowest probability; tie -> most unrevealed neighbours
        let best = null, bestProb = Infinity, bestOpen = -1;
        for (let r = 0; r < game.rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cell = r * cols + c;
                if (!mineProb.has(cell)) continue;
                const prob = mineProb.get(cell);
                const open = game.getUnrevealedNeighbors(r, c).length;
                if (prob < bestProb - PROB_EPSILON ||
                    (Math.abs(prob - bestProb) <= PROB_EPSILON && open > bestOpen)) {
                    best = [r, c];
                    bestProb = prob;
                    bestOpen = open;
                }
            }
        }
        if (best) return ['reveal', best[0], best[1]];

        // No probability available - pick any unrevealed cell (random)
        const unrevealedCells = this.getUnknownCells();
        if (unrevealedCells.length > 0) {
            const [r, c] = unrevealedCells[Math.floor(Math.random() * unrevealedCells.length)];
            return ['reveal', r, c];
        }

        return null;
    }
}


// =============================================================================
// NO-GUESS BOARD GENERATION (headless simulation)
// =============================================================================

/**
 * Play a candidate board in memory using ONLY logic:
 * global check + Strategy 1 + Strategy 2 + Strategy 2.5.
 * Strategy 3 is never used: needing it means the board requires a guess.
 *
 * Works on plain arrays through a MinesweeperGame instance - no DOM, no
 * timers, no animation. All certain moves of a pass are applied at once.
 *
 * Returns { solved, progress }:
 *   solved   - true if logic alone revealed every safe cell
 *   progress - number of safe cells revealed before stopping
 */
function simulateLogicSolve(rows, cols, numMines, minePositions, startRow, startCol) {
    const sim = new MinesweeperGame(rows, cols, numMines);
    for (const [r, c] of minePositions) sim.mines[r][c] = true;
    sim.calculateNumbers();
    sim.firstClick = false;          // mines are already placed
    sim.minesPlaced = true;
    sim.reveal(startRow, startCol);

    const ai = new MinesweeperAI(sim);
    while (!sim.gameOver) {
        let moves = ai.findGlobalMoves();
        if (moves.length === 0) moves = ai.findCertainMoves();
        if (moves.length === 0) moves = ai.analyzeGroups().moves;
        if (moves.length === 0) break;              // deadlock: a guess would be needed

        for (const [action, r, c] of moves) {
            if (sim.revealed[r][c] || sim.flagged[r][c]) continue;   // already handled
            if (action === 'flag') sim.flagged[r][c] = true;
            else sim.reveal(r, c);
            if (sim.gameOver) break;
        }
    }

    let progress = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (sim.revealed[r][c] && !sim.mines[r][c]) progress++;
        }
    }
    return { solved: sim.win, progress };
}


// =============================================================================
// USER INTERFACE
// =============================================================================

const UI = {
    // Board state
    rows: 5, cols: 5, numMines: 4,
    game: null,
    ai: null,
    cells: [],                       // 2D array of <button> elements

    // Size selection
    sizeChoice: '5x5',
    previousSizeChoice: '5x5',
    customConfig: DEFAULT_CUSTOM.slice(),

    // AI
    aiRunning: false,
    aiTimeoutId: null,
    aiStepDelay: 400,                // ms between AI steps

    // Timer
    elapsedSeconds: 0,
    timerRunning: false,
    timerIntervalId: null,
};

// ---- Element references -----------------------------------------------------

const el = {
    board:        document.getElementById('board'),
    boardWrap:    document.getElementById('board-wrap'),
    mineCount:    document.getElementById('mine-count'),
    timer:        document.getElementById('timer'),
    status:       document.getElementById('status'),
    sizeButtons:  Array.from(document.querySelectorAll('.size-btn')),
    newGameBtn:   document.getElementById('new-game-btn'),
    aiStepBtn:    document.getElementById('ai-step-btn'),
    aiAutoBtn:    document.getElementById('ai-auto-btn'),

    customOverlay: document.getElementById('custom-overlay'),
    inputRows:     document.getElementById('input-rows'),
    inputCols:     document.getElementById('input-cols'),
    inputMines:    document.getElementById('input-mines'),
    customError:   document.getElementById('custom-error'),
    customStart:   document.getElementById('custom-start'),
    customCancel:  document.getElementById('custom-cancel'),

    resultOverlay: document.getElementById('result-overlay'),
    resultTitle:   document.getElementById('result-title'),
    resultText:    document.getElementById('result-text'),
    resultTime:    document.getElementById('result-time'),
    resultClose:   document.getElementById('result-close'),
    resultAgain:   document.getElementById('result-again'),
};

// ---- Game management --------------------------------------------------------

/** Start a new game using the currently selected board size. */
function newGame() {
    stopAI();

    if (UI.sizeChoice === CUSTOM_LABEL) {
        [UI.rows, UI.cols, UI.numMines] = UI.customConfig;
    } else {
        [UI.rows, UI.cols, UI.numMines] = PRESETS[UI.sizeChoice];
    }

    UI.game = new MinesweeperGame(UI.rows, UI.cols, UI.numMines);
    UI.ai = new MinesweeperAI(UI.game);

    resetTimer();
    rebuildGrid();
    updateInfo();
    setStatus('Click to start!', 'dim');
    el.resultOverlay.hidden = true;
}

/** Handle a click on one of the grid-size buttons. */
function onSizeChange(choice) {
    if (choice === CUSTOM_LABEL) {
        openCustomDialog();
        return;
    }
    UI.sizeChoice = choice;
    UI.previousSizeChoice = choice;
    markSelectedSizeButton();
    newGame();
}

function markSelectedSizeButton() {
    for (const btn of el.sizeButtons) {
        btn.setAttribute('aria-pressed', String(btn.dataset.size === UI.sizeChoice));
    }
}

/** Destroy and recreate the grid buttons. */
function rebuildGrid() {
    el.board.textContent = '';
    el.board.style.setProperty('--cols', UI.cols);
    sizeBoard();

    UI.cells = [];
    for (let r = 0; r < UI.rows; r++) {
        const rowCells = [];
        for (let c = 0; c < UI.cols; c++) {
            const btn = document.createElement('button');
            btn.className = 'cell';
            btn.type = 'button';
            btn.setAttribute('aria-label', `Row ${r + 1}, column ${c + 1}`);

            // Left click reveals the cell
            btn.addEventListener('click', () => onLeftClick(r, c));

            // Right click places / removes a flag
            btn.addEventListener('contextmenu', (event) => {
                event.preventDefault();
                onRightClick(r, c);
            });

            el.board.appendChild(btn);
            rowCells.push(btn);
        }
        UI.cells.push(rowCells);
    }
}

/** Choose a cell size that keeps the whole board on screen. */
function sizeBoard() {
    const maxCell = UI.cols <= 9 ? 42 : UI.cols <= 12 ? 34 : 28;
    const available = el.boardWrap.clientWidth || 640;
    const gaps = (UI.cols - 1) * 2 + 18;            // gaps + board padding
    const fitted = Math.floor((available - gaps) / UI.cols);
    const size = Math.max(18, Math.min(maxCell, fitted));
    el.board.style.setProperty('--cell-size', `${size}px`);
}

window.addEventListener('resize', () => {
    if (UI.game) sizeBoard();
});

// ---- Click handlers ---------------------------------------------------------

function onLeftClick(r, c) {
    if (UI.game.gameOver) return;
    const wasFirstClick = UI.game.firstClick;
    const newly = UI.game.reveal(r, c);
    // The clock starts once the board is actually generated (first reveal).
    if (!UI.game.firstClick && !UI.game.gameOver) startTimer();
    updateCells(newly);
    checkGameEnd();
    if (wasFirstClick && !UI.game.gameOver && UI.game.guessFree === false) {
        setStatus('Board not fully guess-free', 'ai');
    }
}

function onRightClick(r, c) {
    if (UI.game.gameOver) return;
    UI.game.toggleFlag(r, c);
    updateCell(r, c);
    updateInfo();
}

// ---- Display updates --------------------------------------------------------

function updateCells(cells) {
    for (const [r, c] of cells) updateCell(r, c);
    updateInfo();
}

function updateCell(r, c) {
    const btn = UI.cells[r][c];
    const game = UI.game;

    btn.className = 'cell';
    btn.textContent = '';

    if (game.revealed[r][c]) {
        btn.classList.add('revealed');
        if (game.mines[r][c]) {
            btn.classList.add('mine');
            btn.textContent = '*';
        } else if (game.board[r][c] > 0) {
            btn.classList.add('n' + game.board[r][c]);
            btn.textContent = String(game.board[r][c]);
        }
    } else if (game.flagged[r][c]) {
        btn.classList.add('flagged');
        btn.textContent = 'F';
    }
}

function updateInfo() {
    el.mineCount.textContent = String(UI.game.countRemainingMines()).padStart(3, '0');
}

function setStatus(text, tone) {
    el.status.textContent = text;
    el.status.style.color = tone === 'win' ? 'var(--ok)'
        : tone === 'lose' ? 'var(--danger)'
        : tone === 'ai' ? 'var(--flag)'
        : 'var(--text-dim)';
}

function checkGameEnd() {
    if (!UI.game.gameOver) return;

    stopAI();
    stopTimer();
    const elapsed = formatTime();
    revealAll();

    if (UI.game.win) {
        setStatus(`YOU WIN! (${elapsed})`, 'win');
        el.timer.style.color = 'var(--ok)';
        showResult(true, elapsed);
    } else {
        setStatus(`GAME OVER! (${elapsed})`, 'lose');
        el.timer.style.color = 'var(--danger)';
        showResult(false, elapsed);
    }
}

/** Show all cells at game end. */
function revealAll() {
    const game = UI.game;
    for (let r = 0; r < UI.rows; r++) {
        for (let c = 0; c < UI.cols; c++) {
            const btn = UI.cells[r][c];
            btn.className = 'cell';
            btn.textContent = '';

            if (game.mines[r][c]) {
                if (game.flagged[r][c]) {
                    btn.classList.add('flag-correct');      // mine correctly flagged
                    btn.textContent = 'F';
                } else {
                    btn.classList.add('revealed', 'mine');
                    btn.textContent = '*';
                }
            } else if (game.flagged[r][c]) {
                btn.classList.add('flag-wrong');            // wrong flag
                btn.textContent = 'X';
            } else {
                btn.classList.add('revealed');
                if (game.board[r][c] > 0) {
                    btn.classList.add('n' + game.board[r][c]);
                    btn.textContent = String(game.board[r][c]);
                }
            }
        }
    }
}

// ---- Timer ------------------------------------------------------------------

/** Stop any running timer and reset the display to zero. */
function resetTimer() {
    stopTimer();
    UI.elapsedSeconds = 0;
    el.timer.textContent = '000';
    el.timer.style.color = 'var(--accent)';
}

/** Start counting from the current elapsed value (no-op if already running). */
function startTimer() {
    if (UI.timerRunning) return;
    UI.timerRunning = true;
    UI.timerIntervalId = setInterval(tick, 1000);
}

/** Advance the timer by one second. */
function tick() {
    UI.elapsedSeconds++;
    el.timer.textContent = String(UI.elapsedSeconds).padStart(3, '0');
}

/** Freeze the timer at its current value. */
function stopTimer() {
    UI.timerRunning = false;
    if (UI.timerIntervalId !== null) {
        clearInterval(UI.timerIntervalId);
        UI.timerIntervalId = null;
    }
}

/** Elapsed time as a readable "m min s s" / "s s" string. */
function formatTime() {
    const minutes = Math.floor(UI.elapsedSeconds / 60);
    const seconds = UI.elapsedSeconds % 60;
    return minutes ? `${minutes} min ${seconds} s` : `${seconds} s`;
}

// ---- AI controls ------------------------------------------------------------

/**
 * Show an AI status message. On a fallback board (no guess-free layout was
 * found) a reminder is appended, because the AI may still have to guess.
 */
function setAIStatus(text) {
    const note = UI.game.guessFree === false ? ' (not guess-free)' : '';
    setStatus(text + note, 'ai');
}

/** Execute one AI move. */
function aiStep() {
    if (UI.game.gameOver) return;

    if (UI.game.firstClick) {
        // AI makes first click at center
        const r = Math.floor(UI.rows / 2), c = Math.floor(UI.cols / 2);
        const newly = UI.game.reveal(r, c);
        startTimer();
        updateCells(newly);
        checkGameEnd();
        if (!UI.game.gameOver && UI.game.guessFree === false) {
            setStatus('AI: revealed center - Board not fully guess-free', 'ai');
        } else {
            setStatus('AI: revealed center', 'ai');
        }
        return;
    }

    const move = UI.ai.getNextMove();
    if (move === null) {
        setAIStatus('AI: no move found');
        return;
    }

    const [action, r, c] = move;
    if (action === 'flag') {
        UI.game.toggleFlag(r, c);
        updateCell(r, c);
        updateInfo();
        setAIStatus(`AI: flag (${r},${c})`);
    } else if (action === 'reveal') {
        const newly = UI.game.reveal(r, c);
        if (!UI.game.gameOver) startTimer();
        updateCells(newly);
        checkGameEnd();
        setAIStatus(`AI: reveal (${r},${c})`);
    }
}

/** Toggle AI auto-solve mode. */
function aiAutoToggle() {
    if (UI.aiRunning) {
        stopAI();
        setStatus('AI stopped', 'dim');
    } else {
        UI.aiRunning = true;
        el.aiAutoBtn.textContent = 'Stop AI';
        el.aiAutoBtn.classList.remove('solve');
        el.aiAutoBtn.classList.add('stop');
        aiAutoStep();
    }
}

/** Auto step: perform one AI move, then schedule the next. */
function aiAutoStep() {
    if (!UI.aiRunning || UI.game.gameOver) { stopAI(); return; }

    aiStep();

    if (!UI.game.gameOver && UI.aiRunning) {
        UI.aiTimeoutId = setTimeout(aiAutoStep, UI.aiStepDelay);
    } else {
        stopAI();
    }
}

function stopAI() {
    UI.aiRunning = false;
    if (UI.aiTimeoutId !== null) {
        clearTimeout(UI.aiTimeoutId);
        UI.aiTimeoutId = null;
    }
    el.aiAutoBtn.textContent = 'AI: Auto Solve';
    el.aiAutoBtn.classList.remove('stop');
    el.aiAutoBtn.classList.add('solve');
}

// ---- Custom size dialog -----------------------------------------------------

function openCustomDialog() {
    const [rows, cols, mines] = UI.customConfig;
    el.inputRows.value = rows;
    el.inputCols.value = cols;
    el.inputMines.value = mines;
    el.customError.textContent = '';
    el.customOverlay.hidden = false;
    el.inputRows.focus();
}

function closeCustomDialog() {
    el.customOverlay.hidden = true;
    // Cancel: keep the previous selection and the current game.
    UI.sizeChoice = UI.previousSizeChoice;
    markSelectedSizeButton();
}

/**
 * Read one field of the custom dialog.
 * Returns the number, or throws an Error with a readable message.
 */
function readInt(input, label, lo, hi) {
    const text = input.value.trim();
    if (!/^-?\d+$/.test(text)) throw new Error(`${label} must be a whole number.`);
    const value = Number(text);
    if (value < lo || value > hi) throw new Error(`${label} must be between ${lo} and ${hi}.`);
    return value;
}

function confirmCustomDialog() {
    let rows, cols, mines;
    try {
        rows = readInt(el.inputRows, 'Rows', MIN_ROWS, MAX_ROWS);
        cols = readInt(el.inputCols, 'Columns', MIN_COLS, MAX_COLS);
        mines = readInt(el.inputMines, 'Mines', MIN_MINES, MAX_MINES);

        const allowed = maxMinesFor(rows, cols);
        if (mines > allowed) {
            throw new Error(
                `A ${rows}x${cols} board can hold at most ${allowed} mines ` +
                `(the first click and its 8 neighbours stay safe).`
            );
        }
    } catch (err) {
        el.customError.textContent = err.message;
        return;
    }

    UI.customConfig = [rows, cols, mines];
    UI.sizeChoice = CUSTOM_LABEL;
    UI.previousSizeChoice = CUSTOM_LABEL;
    el.customOverlay.hidden = true;
    markSelectedSizeButton();
    newGame();
}

// ---- End of game dialog -----------------------------------------------------

function showResult(win, elapsed) {
    el.resultTitle.textContent = win ? 'You win!' : 'Boom! Game over';
    el.resultTitle.className = win ? 'win' : 'lose';
    el.resultText.textContent = win
        ? 'Every safe cell is uncovered.'
        : 'You clicked on a mine.';
    el.resultTime.textContent = `Time: ${elapsed}`;
    el.resultOverlay.hidden = false;
}

// ---- Wiring -----------------------------------------------------------------

for (const btn of el.sizeButtons) {
    btn.addEventListener('click', () => onSizeChange(btn.dataset.size));
}

el.newGameBtn.addEventListener('click', newGame);
el.aiStepBtn.addEventListener('click', aiStep);
el.aiAutoBtn.addEventListener('click', aiAutoToggle);

el.customStart.addEventListener('click', confirmCustomDialog);
el.customCancel.addEventListener('click', closeCustomDialog);
el.customOverlay.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') confirmCustomDialog();
    if (event.key === 'Escape') closeCustomDialog();
});

el.resultClose.addEventListener('click', () => { el.resultOverlay.hidden = true; });
el.resultAgain.addEventListener('click', newGame);

// Stop the browser menu appearing when right-clicking the board padding.
el.board.addEventListener('contextmenu', (event) => event.preventDefault());

// Start the first game.
markSelectedSizeButton();
newGame();
