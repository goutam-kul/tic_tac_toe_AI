import numpy as np
from collections import defaultdict
import pickle
import os 

class QLearningAgent:
    def __init__(self, player_symbol, epsilon=0.1, alpha=0.1, gamma=0.9):
        """Initialize Q-learning agent with improved state representation
        
        Args:
            player_symbol (int): 1 for X, -1 for O
            epsilon (float): Exploration rate
            alpha (float): Learning rate
            gamma (float): Discount factor
        """
        self.player_symbol = player_symbol
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.state_history = []

    def get_state_key(self, board):
        """Convert board to a unique state key considering symmetricies
        Args:
            board (np.array): Current game board
        Returns:
            str: Canonical state representation
        """
        # Get all symmetrical board positions 
        rotations = [np.rot90(board, k=k) for k in range(4)]
        flipped = [np.fliplr(board) for board in rotations]
        all_symmetricies = rotations + flipped

        # Convert each to string and return the lexicographically smallest 
        state_strings = [''.join(map(str, b.flatten())) for b in all_symmetricies]
        return min(state_strings)
    
    def get_legal_actions(self, board):
        """Get all the leagal moves on the board"""
        return [(i // 3, i % 3) for i in range(9) if board[i // 3, i % 3] == 0]
    
    def get_action(self, board):
        """
        Choose action using epsilon-greedy startegy with improved selection
        Args:
            board (np.array): Current game board
        Return:
            tuple: Selected position (row, col)
        """
        legal_actions = self.get_legal_actions(board=board)
        if not legal_actions:
            return None
        
        state = self.get_state_key(board)

        # Exploration
        if np.random.random() < self.epsilon:
            return legal_actions[np.random.randint(len(legal_actions))]
        
        # Exploitation with action making
        q_values = self.q_table[state].copy()
        # Mask illegal actions
        for i in range(9):
            if (i//3, i%3) not in legal_actions:
                q_values[i] = -np.inf

        return legal_actions[np.argmax([q_values[a[0]*3 + a[1]] for a in legal_actions])]
    
    def learn(self, state, action, reward, next_state, done):
        """Update the Q-values using Q-learning update rule with eligibility
        Args:
            state (str): current state key
            action (tuple): Taken action (row, col)
            rewards (float): Received reward
            next_state (str): Next state key
            done (bool): whether episode is done
        """
        action_idx = action[0] * 3 + action[1]

        # Get current Q-value
        current_q = self.q_table[state][action_idx]

        # Get next max Q-value (0 if terminal state)
        next_max_q = 0 if done else np.max(self.q_table[next_state])

        # Q-learing update
        new_q = current_q + self.alpha *(reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action_idx] = new_q

        # Add to state history for experience replay
        self.state_history.append((state, action, reward, next_state, done))
        
    def save_model(self, filename):
        """Save Q-table and agent parameters"""
        save_dict = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'player_symbol': self.player_symbol
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load_model(self, filename):
        """Load Q-table and agent parameters"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                save_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(9), save_dict['q_table'])
                self.epsilon = save_dict['epsilon']
                self.alpha = save_dict['alpha']
                self.gamma = save_dict['gamma']
                self.player_symbol = save_dict['player_symbol']
            return True
        return False