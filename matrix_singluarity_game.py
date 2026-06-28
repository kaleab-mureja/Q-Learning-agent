import random
import time
import os

# Global Visuals
CYAN = "\033[96m"; YELLOW = "\033[93m"; GREEN = "\033[92m"; 
BLUE = "\033[94m"; RED = "\033[91m"; RESET = "\033[0m"
RL_BG = "\033[42;30m"; HEU_BG = "\033[44;37m"

if os.name == 'nt':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

class MatrixGame:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.history = []
        self._initialize_board()

    def _initialize_board(self):
        # 10% Initialization (2/16 cells)
        count = 0
        while count < 2:
            r, c = random.randint(0, 3), random.randint(0, 3)
            if self.board[r][c] is None:
                self.board[r][c] = random.choice(self.numbers)
                count += 1

    def get_determinant(self, matrix):
        m = [[(cell if cell is not None else 0) for cell in row] for row in matrix]
        return self._calc_det(m)

    def _calc_det(self, m):
        size = len(m)
        if size == 1: return m[0][0]
        if size == 2: return m[0][0]*m[1][1] - m[0][1]*m[1][0]
        det = 0
        for c in range(size):
            minor = [row[:c] + row[c+1:] for row in m[1:]]
            det += ((-1)**c) * m[0][c] * self._calc_det(minor)
        return det

    def is_full(self):
        return all(cell is not None for row in self.board for cell in row)

    def make_move(self, r, c, val):
        # Traceback capability
        self.history.append([row[:] for row in self.board])
        self.board[r][c] = val
        
    def traceback(self):
        if self.history: self.board = self.history.pop()

    def get_state(self):
        # Canonical symmetry mapping
        rows = []
        for row in self.board:
            rows.append(tuple(cell if cell else 0 for cell in row))
        return str(tuple(sorted(rows)))

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.3
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.9992 # Epsilon-greedy decay
        self.min_epsilon = 0.01

    def choose_action(self, game, exploit_only=False):
        state = game.get_state()
        available = [(r, c, v) for r in range(4) for c in range(4) 
                     if game.board[r][c] is None for v in game.numbers]
        if not available: return None
        # Epsilon-greedy approach: Exploration vs Exploitation
        if (exploit_only or random.random() > self.epsilon) and state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return random.choice(available)

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = {}
        if action not in self.q_table[state]: self.q_table[state][action] = 0
        old_val = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {None: 0}).values(), default=0)
        self.q_table[state][action] = old_val + self.lr * (reward + self.gamma * next_max - old_val)
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

class HeuristicAgent:
    def choose_action(self, game):
        available = [(r, c) for r in range(4) for c in range(4) if game.board[r][c] is None]
        best_move = None
        max_abs_det = -1
        for r, c in random.sample(available, min(len(available), 4)):
            for v in [1, 9]: 
                game.make_move(r, c, v)
                d = abs(game.get_determinant(game.board))
                if d > max_abs_det:
                    max_abs_det = d
                    best_move = (r, c, v)
                game.traceback()
        return best_move or (*available[0], random.choice(game.numbers))

def render_board(game, turn, det, last_move, scores):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(YELLOW + "╔════════════════════════════════════════════════════════════╗" + RESET)
    print(f"║  {CYAN}TURN:{RESET} {turn:02}  ║  {CYAN}DET:{RESET} {det:<10}  ║  {GREEN}RL:{RESET} {scores['RL']:<5} ║  {BLUE}HEU:{RESET} {scores['Heuristic']:<5} ║")
    print(YELLOW + "╚════════════════════════════════════════════════════════════╝" + RESET)
    
    print("\n      0      1      2      3")
    print("   +------+------+------+------+")
    for r in range(4):
        row_str = f" {r} |"
        for c in range(4):
            val = game.board[r][c]
            char = str(val) if val else " "
            if last_move and (r, c) == (last_move[0], last_move[1]):
                color = RL_BG if last_move[2] == "RL" else HEU_BG
                row_str += color + f"  {char}   " + RESET + "|"
            else:
                row_str += f"  {char}   |"
        print(row_str)
        print("   +------+------+------+------+")
    p_color = GREEN if last_move[2] == "RL" else BLUE
    print(f"\n >>> ACTION: {p_color}{last_move[2]}{RESET} placed {last_move[3]} at ({last_move[0]}, {last_move[1]})")

def play(rl, heu, training=False):
    game = MatrixGame()
    players = ["RL", "Heuristic"]
    random.shuffle(players)
    turn, scores = 1, {"RL": 0, "Heuristic": 0}

    while not game.is_full():
        p_name = players[turn % 2]
        state_before = game.get_state()
        det_before = abs(game.get_determinant(game.board))
        
        action = rl.choose_action(game, exploit_only=not training) if p_name == "RL" else heu.choose_action(game)
        if not action: break
        
        game.make_move(action[0], action[1], action[2])
        det_after = abs(game.get_determinant(game.board))
        is_singular = (det_after == 0 and turn > 3)
        
        # Reward Logic
        reward = 2000 if is_singular else (20 if det_after < det_before else -10)
        if game.is_full() and not is_singular: reward = -500
        # Rank deficiency penalty
        if all(cell is None or cell == 0 for cell in game.board[action[0]]):
            reward -= 100

        scores[p_name] += int(reward)
        if p_name == "RL": rl.update_q(state_before, action, reward, game.get_state())

        if not training:
            render_board(game, turn, int(det_after), (action[0], action[1], p_name, action[2]), scores)
            if not is_singular:
                time.sleep(0.3)

        if is_singular:
            if not training: print(f"\n {GREEN}★ SINGULARITY REACHED ★{RESET}")
            break
        turn += 1
    
    return ("RL" if scores["RL"] > scores["Heuristic"] else "Heuristic"), scores

if __name__ == "__main__":
    rl_agent, heu_agent = QLearningAgent(), HeuristicAgent()
    print(f"\n{YELLOW}[!] TRAINING RL AGENT (3000 SESSIONS)...{RESET}")
    for i in range(3001):
        play(rl_agent, heu_agent, training=True)
        if i % 1000 == 0: print(f" Progress: {i}/3000")

    print(f"\n{YELLOW}--- LIVE SHOWDOWN ---{RESET}")
    time.sleep(1)
    winner, final_scores = play(rl_agent, heu_agent, training=False)
    
    # Final Result stays on terminal
    print(f"\n{GREEN if winner == 'RL' else RED}WINNER: {winner.upper()}{RESET}")
    print(f"Final Totals -> RL: {final_scores['RL']} | Heuristic: {final_scores['Heuristic']}\n")