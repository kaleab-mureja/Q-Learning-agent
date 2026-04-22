# Matrix Assassin: Singularity Game

A 4x4 matrix-based game where a **Q-Learning RL Agent** competes against a **Mathematical Heuristic** to reach a singular state ($det(A) = 0$).

## Quick Start
1. **Requirement:** Python 3.x (No external libraries required).
2. **Run:** `python matrix_singularity_game.py`

## How to Play
- The matrix starts 10% pre-filled.
- Agents take turns placing numbers (1–9).
- The game ends when the matrix is singular or the board is full.
- The agent with the highest reward wins.

---

## Technical Description

### 1. Reinforcement Learning (Q-Learning)
We implemented a **model-free, value-based** RL algorithm. The agent learns the "Quality" (Q-value) of actions through experience.

* **State:** Canonical matrix configuration.
* **Action:** Placement of a number (1-9) in an empty cell.
* **Update Rule:** Bellman Equation 
    $Q(s,a) \leftarrow Q(s,a) + \alpha[ r + \gamma\max Q(s',a') - Q(s,a)]$

### 2. The "Elegant Method": Symmetry Mapping
To handle the massive state space, we use **Canonical State Representation**. By sorting the matrix rows before storing them in the Q-table, we treat rotated or row-swapped matrices as the same state. This allows the agent to converge on a winning strategy within just 3,000 games.

### 3. Heuristic Strategy
The Heuristic agent uses a **Greedy Look-Ahead** strategy. It utilizes the game's **Traceback** feature to simulate moves and choose the one that maximizes the absolute determinant, acting as a mathematical blocker against the RL agent.

### 4. Constraints & Metrics
- **Metric:** Singularity is defined as $det(A) = 0$.
- **Penalties:** Includes a -100 reward penalty for rank deficiency (zero-equivalent rows).
- **History:** A stack-based system enables full traceback for move simulation.