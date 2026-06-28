# Matrix Assassin: Singularity Game

A 4×4 matrix-based game where a **Q-Learning Reinforcement Learning (RL) Agent** competes against a **Mathematical Heuristic Agent** to reach a singular state:

\[
\det(A) = 0
\]

---

## Quick Start

You can run the game directly using Docker without installing Python locally.

### 1. Pull the Docker image

```bash
docker pull kaleab1993/q-learning-agent:latest
```

### 2. Run the game

```bash
docker run -it --rm kaleab1993/q-learning-agent:latest
```

Alternatively, if running locally:

```bash
python matrix_singularity_game.py
```

---

## How to Play

- The 4×4 matrix starts **10% pre-filled**.
- Players take turns placing numbers **1–9** into empty cells.
- The game ends when:
  - the matrix becomes **singular**, or
  - the board is completely filled.
- The player with the **highest cumulative reward** wins.

---

# Technical Description

## 1. Reinforcement Learning (Q-Learning)

The RL agent uses a **model-free, value-based Q-Learning algorithm** to learn an optimal strategy through repeated self-play.

### State

The canonical representation of the current matrix configuration.

### Action

Place a value from **1–9** into an empty matrix cell.

### Q-Learning Update Rule

\[
Q(s,a) \leftarrow Q(s,a)
+
\alpha
\left[
r
+
\gamma
\max_{a'}Q(s',a')
-
Q(s,a)
\right]
\]

where:

- \(Q(s,a)\) = quality of taking action \(a\) in state \(s\)
- \(\alpha\) = learning rate
- \(\gamma\) = discount factor
- \(r\) = immediate reward
- \(s'\) = next state

---

## 2. Elegant Method: Symmetry Mapping

The game's state space is extremely large.

To improve learning efficiency, the agent stores **Canonical State Representations** instead of raw matrices.

Before inserting a state into the Q-table:

- Matrix rows are sorted into a canonical ordering.
- Symmetric matrices (e.g., row permutations) map to the same state.

This dramatically reduces the number of unique states the agent must learn, allowing convergence to a competitive strategy in approximately **3,000 training games**.

---

## 3. Heuristic Strategy

The opposing agent follows a **Greedy Look-Ahead heuristic**.

For every possible move, it:

1. Uses the game's **Traceback** mechanism to simulate the move.
2. Computes the resulting determinant.
3. Chooses the move that maximizes the **absolute determinant**.

By maximizing \(|\det(A)|\), the heuristic attempts to keep the matrix far from singularity, acting as a mathematical blocker against the RL agent.

---

## 4. Constraints & Reward Metrics

### Singularity Condition

A matrix is considered singular when

\[
\det(A)=0
\]

### Penalties

The reward function includes a significant penalty:

- **−100 reward** for creating rank-deficient structures (e.g., zero-equivalent rows).

### Traceback System

A stack-based history mechanism enables:

- Full move undo
- Efficient simulation
- Heuristic look-ahead without modifying the actual game state

---

## Summary

| Component | Description |
|-----------|-------------|
| **Game Board** | 4×4 matrix |
| **RL Algorithm** | Q-Learning |
| **State Compression** | Canonical symmetry mapping |
| **Opponent** | Greedy mathematical heuristic |
| **Objective** | Reach a singular matrix (\(\det(A)=0\)) |
| **Reward Penalty** | −100 for rank-deficient rows |
| **Training Convergence** | ~3,000 games |
| **Simulation** | Stack-based traceback system |