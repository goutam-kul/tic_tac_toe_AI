import numpy as np
from agent import QLearningAgent
import time
from tqdm import tqdm
import os
from datetime import datetime

class TicTacToeEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the game board and return initial state"""
        self.board = np.zeros((3, 3))
        # Randomly choose starting player
        self.current_player = np.random.choice([1, -1])
        return self.board.copy()
    
    def get_reward(self, player):
        """Calculate reward for the current board state with enhanced strategic rewards"""
        reward = 0
        
        # Major rewards/penalties with balanced values
        if self.check_winner(player):
            return 20  # Winning
        elif self.check_winner(-player):
            return -20  # Losing
        elif np.all(self.board != 0):
            return 10  # Draw
            
        # 1. Early Game Strategy
        num_moves = np.count_nonzero(self.board)
        if num_moves <= 2:
            # Reward for taking center in early game
            if self.board[1,1] == player:
                reward += 3
            # Reward for taking corners in early game
            corners = [(0,0), (0,2), (2,0), (2,2)]
            for i, j in corners:
                if self.board[i,j] == player:
                    reward += 2
        
        # 2. Fork Creation (Multiple Winning Paths)
        fork_reward = self.check_fork_opportunities(player)
        reward += fork_reward * 5  # High reward for creating forks
        
        # 3. Fork Defense
        opponent_fork_threat = self.check_fork_opportunities(-player)
        if opponent_fork_threat > 0:
            reward += 4  # Reward for blocking potential forks
            
        # 4. Center and Corner Control (Mid-game)
        if num_moves > 2:
            # Center control becomes more valuable mid-game
            if self.board[1,1] == player:
                reward += 2
            
            # Corner control with adjacent edge analysis
            corners = [(0,0), (0,2), (2,0), (2,2)]
            for i, j in corners:
                if self.board[i,j] == player:
                    # Check adjacent edges for enhanced corner value
                    adj_edges = self.get_adjacent_edges(i, j)
                    for edge_i, edge_j in adj_edges:
                        if self.board[edge_i, edge_j] == player:
                            reward += 2  # Bonus for connected corner-edge patterns
                
        # 5. Blocking opponent's winning moves (with priority)
        block_reward = self.check_blocking_opportunities(player)
        reward += block_reward * 4
        
        # 6. Creating winning opportunities (with pattern recognition)
        win_ops = self.check_winning_opportunities(player)
        reward += win_ops * 3
        
        # 7. Board Control Ratio
        player_squares = np.sum(self.board == player)
        opponent_squares = np.sum(self.board == -player)
        if player_squares > opponent_squares:
            reward += 1  # Small reward for controlling more squares
            
        # 8. Pattern-based penalties
        if self.is_trapped(player):
            reward -= 3  # Penalty for getting trapped
            
        return reward
        
    def check_fork_opportunities(self, player):
        """Count number of potential fork opportunities with accurate position checking"""
        fork_count = 0
        empty_positions = [(i, j) for i in range(3) for j in range(3) if self.board[i,j] == 0]
        
        for i, j in empty_positions:
            # Temporarily place player's symbol
            self.board[i,j] = player
            
            # Count potential winning moves after this move
            winning_moves = 0
            
            # Check all empty positions for potential wins
            for test_i, test_j in [(x, y) for x in range(3) for y in range(3) if self.board[x,y] == 0]:
                # Temporarily make the test move
                self.board[test_i, test_j] = player
                
                # Check if this creates a win
                if self.check_winner(player):
                    winning_moves += 1
                    
                # Undo test move
                self.board[test_i, test_j] = 0
            
            # It's a fork if placing piece here creates two or more winning moves
            if winning_moves >= 2:
                fork_count += 1
            
            # Remove the original test move
            self.board[i,j] = 0
                
        return fork_count
        
    def get_lines_through_position(self, i, j):
        """Get all lines (row, column, diagonals) that pass through position (i,j)"""
        lines = []
        # Add row and column
        lines.append(self.board[i,:])
        lines.append(self.board[:,j])
        
        # Check if position is on diagonals
        if i == j:
            lines.append(np.diag(self.board))
        if i + j == 2:
            lines.append(np.diag(np.fliplr(self.board)))
            
        return lines
        
    def get_adjacent_edges(self, i, j):
        """Get adjacent edge positions for a corner"""
        if i == 0 and j == 0:  # Top-left corner
            return [(0,1), (1,0)]
        elif i == 0 and j == 2:  # Top-right corner
            return [(0,1), (1,2)]
        elif i == 2 and j == 0:  # Bottom-left corner
            return [(2,1), (1,0)]
        else:  # Bottom-right corner
            return [(2,1), (1,2)]
            
    def check_blocking_opportunities(self, player):
        """Check and prioritize blocking opponent's winning moves"""
        block_count = 0
        for i in range(3):
            # Check rows and columns
            if sum(self.board[i,:] == -player) == 2 and sum(self.board[i,:] == player) == 1:
                block_count += 1
            if sum(self.board[:,i] == -player) == 2 and sum(self.board[:,i] == player) == 1:
                block_count += 1
                
        # Check diagonals
        for diag in [np.diag(self.board), np.diag(np.fliplr(self.board))]:
            if sum(diag == -player) == 2 and sum(diag == player) == 1:
                block_count += 1
                
        return block_count
        
    def check_winning_opportunities(self, player):
        """Check for potential winning moves with pattern recognition"""
        win_ops = 0
        for i in range(3):
            # Check rows and columns
            if sum(self.board[i,:] == player) == 2 and sum(self.board[i,:] == 0) == 1:
                win_ops += 1
            if sum(self.board[:,i] == player) == 2 and sum(self.board[:,i] == 0) == 1:
                win_ops += 1
                
        # Check diagonals
        for diag in [np.diag(self.board), np.diag(np.fliplr(self.board))]:
            if sum(diag == player) == 2 and sum(diag == 0) == 1:
                win_ops += 1
                
        return win_ops
        
    def is_trapped(self, player):
        """Check if player is in a disadvantageous position"""
        # Example trap: opponent controls center and opposite corners
        if self.board[1,1] == -player:  # Opponent has center
            corners = [(0,0), (0,2), (2,0), (2,2)]
            opposite_corners_controlled = 0
            for i, j in corners:
                if self.board[i,j] == -player:
                    opposite_corners_controlled += 1
            if opposite_corners_controlled >= 2:
                return True
        return False

    def step(self, action):
        """Execute one step in the environment"""
        if action is None:
            return self.board.copy(), -15, True
            
        row, col = action
        
        # Invalid move
        if self.board[row, col] != 0:
            return self.board.copy(), -15, True
        
        # Make move
        self.board[row, col] = self.current_player
        
        # Get reward and check if game is done
        reward = self.get_reward(self.current_player)
        done = self.check_winner(self.current_player) or \
               self.check_winner(-self.current_player) or \
               np.all(self.board != 0)
        
        # Switch player
        self.current_player *= -1
        
        return self.board.copy(), reward, done

def train_agents(episodes=100000, save_interval=10000):
    """Train both agents with balanced learning opportunities"""
    env = TicTacToeEnv()
    
    # Initialize agents with identical parameters
    initial_epsilon = 0.4
    final_epsilon = 0.1
    decay_rate = (initial_epsilon - final_epsilon) / (episodes * 0.7)
    
    agent_x = QLearningAgent(player_symbol=1, epsilon=initial_epsilon, alpha=0.3)
    agent_o = QLearningAgent(player_symbol=-1, epsilon=initial_epsilon, alpha=0.3)
    
    # Load existing models if available
    agent_x.load_model('models/agent_x_final.pkl')
    agent_o.load_model('models/agent_o_final.pkl')
    
    # Statistics tracking
    stats = {'x_wins': 0, 'o_wins': 0, 'draws': 0}
    best_rates = {'x': 0, 'o': 0}
    window_size = 1000  # Track rolling performance
    recent_results = []
    
    pbar = tqdm(total=episodes, desc="Training Progress")
    
    for episode in range(episodes):
        # Decay epsilon
        if episode < episodes * 0.7:
            current_epsilon = initial_epsilon - (decay_rate * episode)
            agent_x.epsilon = current_epsilon
            agent_o.epsilon = current_epsilon
        
        state = env.reset()
        game_over = False
        
        while not game_over:
            current_agent = agent_x if env.current_player == 1 else agent_o
            
            # Get action for current player
            action = current_agent.get_action(state)
            if action is None:
                break
                
            next_state, reward, game_over = env.step(action)
            
            # Learn from this move
            current_agent.learn(
                current_agent.get_state_key(state),
                action,
                reward,
                current_agent.get_state_key(next_state),
                game_over
            )
            
            if game_over:
                # Update statistics
                if reward == 20:  # Win
                    if env.current_player == 1:
                        stats['x_wins'] += 1
                    else:
                        stats['o_wins'] += 1
                elif reward == 10:  # Draw
                    stats['draws'] += 1
                
                recent_results.append((env.current_player, reward))
                if len(recent_results) > window_size:
                    recent_results.pop(0)
                    
                break
                
            state = next_state
        
        if (episode + 1) % save_interval == 0:
            # Calculate rolling statistics
            total_games = len(recent_results)
            if total_games > 0:
                x_win_rate = sum(1 for p, r in recent_results if p == 1 and r == 20) / total_games * 100
                o_win_rate = sum(1 for p, r in recent_results if p == -1 and r == 20) / total_games * 100
                draw_rate = sum(1 for _, r in recent_results if r == 10) / total_games * 100
                
                pbar.set_description(
                    f"ε={current_epsilon:.2f} X: {x_win_rate:.1f}% O: {o_win_rate:.1f}% D: {draw_rate:.1f}%"
                )
                
                # Save best models
                if x_win_rate > best_rates['x']:
                    best_rates['x'] = x_win_rate
                    agent_x.save_model('models/best_agent_x.pkl')
                
                if o_win_rate > best_rates['o']:
                    best_rates['o'] = o_win_rate
                    agent_o.save_model('models/best_agent_o.pkl')
            
            # Regular checkpoints
            agent_x.save_model('models/agent_x_final.pkl')
            agent_o.save_model('models/agent_o_final.pkl')
        
        pbar.update(1)
    
    pbar.close()
    
    # Final statistics
    total_games = sum(stats.values())
    print(f"\nTraining completed!")
    print(f"Total games: {total_games}")
    print(f"X wins: {stats['x_wins']/total_games*100:.1f}%")
    print(f"O wins: {stats['o_wins']/total_games*100:.1f}%")
    print(f"Draws: {stats['draws']/total_games*100:.1f}%")
    
    return agent_x, agent_o

if __name__ == "__main__":
    print("Training agents...")
    os.makedirs('models', exist_ok=True)
    agent_x, agent_o = train_agents(episodes=2000000)  # 2 million episodes