import random
import time
import os
import numpy as np

# Terminal Color Support
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
        # Initial entropy to prevent repetitive game states
        for _ in range(2):
            r, c = random.randint(0, 3), random.randint(0, 3)
            if self.board[r][c] is None:
                self.board[r][c] = random.choice(self.numbers)

    def get_determinant(self, matrix):
        # Convert None to 0 for calculation
        mat = np.array([[ (cell if cell is not None else 0) for cell in row] for row in matrix])
        return int(round(np.linalg.det(mat)))

    def is_full(self):
        return all(cell is not None for row in self.board for cell in row)

    def make_move(self, r, c, val):
        self.history.append([row[:] for row in self.board])
        self.board[r][c] = val
        
    def traceback(self):
        if self.history: self.board = self.history.pop()

    def get_state_bits(self):
        return "".join(str(cell if cell else 0) for row in self.board for cell in row)

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.min_epsilon = 0.05

    def choose_action(self, game, exploit_only=False):
        state = game.get_state_bits()
        available = [(r, c, v) for r in range(4) for c in range(4) 
                     if game.board[r][c] is None for v in game.numbers]
        
        if (exploit_only or random.random() > self.epsilon) and state in self.q_table:
            # Pick best move from known state
            actions = self.q_table[state]
            return max(actions, key=actions.get)
        
        return random.choice(available) if available else None

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = {}
        if action not in self.q_table[state]: self.q_table[state][action] = 0
        
        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {None: 0}).values(), default=0)
        
        # Q-Learning Formula
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

class HeuristicAgent:
    def choose_action(self, game):
        # The Heuristic tries to maximize the absolute determinant to stay non-singular
        available = [(r, c) for r in range(4) for c in range(4) if game.board[r][c] is None]
        best_move = None
        max_abs_det = -1
        
        # Sample moves to keep it performant
        sample_spots = random.sample(available, min(len(available), 4))
        for r, c in sample_spots:
            for v in [1, 5, 9]: # Testing extremes
                game.make_move(r, c, v)
                d = abs(game.get_determinant(game.board))
                if d > max_abs_det:
                    max_abs_det = d
                    best_move = (r, c, v)
                game.traceback()
        return best_move or (*available[0], random.choice(game.numbers))

def render_board(game, turn, det, last_move=None):
    CYAN = "\033[96m"; MAGENTA = "\033[95m"; YELLOW = "\033[93m"; RESET = "\033[0m"
    RL_COLOR = "\033[41;37m"; HEU_COLOR = "\033[44;37m"
    
    print(f"\n{YELLOW}Turn {turn}{RESET} | {CYAN}Current Determinant: {det}{RESET}")
    print("    0   1   2   3")
    print("  +---+---+---+---+")
    for r in range(4):
        row_str = f"{r} |"
        for c in range(4):
            val = game.board[r][c]
            char = str(val) if val else " "
            if last_move and (r, c) == (last_move[0], last_move[1]):
                color = RL_COLOR if last_move[2] == "RL" else HEU_COLOR
                row_str += f"{color} {char} {RESET}|"
            else:
                row_str += f" {char} |"
        print(row_str)
        print("  +---+---+---+---+")

def play(rl, heu, training=False):
    game = MatrixGame()
    players = ["RL", "Heuristic"]
    random.shuffle(players)
    turn = 1
    rl_reward_accum = 0

    while not game.is_full():
        p_name = players[turn % 2]
        state_before = game.get_state_bits()
        
        if p_name == "RL":
            action = rl.choose_action(game, exploit_only=not training)
        else:
            action = heu.choose_action(game)
            
        if not action: break
        game.make_move(action[0], action[1], action[2])
        det = game.get_determinant(game.board)
        
        # Reward Logic
        is_singular = (det == 0 and turn > 4)
        reward = 0
        if p_name == "RL":
            if is_singular: reward = 1000
            elif game.is_full(): reward = -500
            else: reward = 2 / (abs(det) + 1) # Shaping: Reward getting closer to zero
            
            rl.update_q(state_before, action, reward, game.get_state_bits())
            rl_reward_accum += reward

        if not training:
            render_board(game, turn, det, (action[0], action[1], p_name))
            time.sleep(0.3)

        if is_singular:
            if not training: print(f"\033[92mSINGULARITY REACHED! RL WINS!\033[0m")
            return "RL", rl_reward_accum
        
        turn += 1
    
    if not training: print(f"\033[91mMATRIX IS NON-SINGULAR. Heuristic Wins.\033[0m")
    return "Heuristic", rl_reward_accum

if __name__ == "__main__":
    rl_agent = QLearningAgent()
    heu_agent = HeuristicAgent()
    
    print(f"\n{'='*40}\n  PHASE 1: TRAINING RL (3000 GAMES)\n{'='*40}")
    for i in range(3001):
        play(rl_agent, heu_agent, training=True)
        if i % 1000 == 0: print(f"Progress: {i}/3000 games...")

    print(f"\n{'='*40}\n  PHASE 2: LIVE COMPETITIVE MATCH\n{'='*40}")
    winner, score = play(rl_agent, heu_agent, training=False)
    
    print(f"\nFinal RL Strategy Score: {int(score)}")
    print(f"Tournament Result: {winner} is the victor.")