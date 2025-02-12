import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from agent import QLearningAgent

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe vs Learning AI")
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize AI agent with learning parameters
        self.ai_agent = QLearningAgent(
            player_symbol=1,  # AI plays as X
            epsilon=0.1,      # Some exploration to try new moves
            alpha=0.1         # Learning rate during gameplay
        )
        self.load_best_model()
        
        # Learning history
        self.game_history = []
        self.games_played = 0
        self.training_episodes = 100  # Number of games to learn from
        
        # Game state
        self.board = np.zeros((3, 3))
        self.buttons = []
        self.game_over = False
        self.last_state = None
        self.last_action = None
        
        # GUI Setup
        self.setup_gui()
        
        # Initialize game stats
        self.stats = {'wins': 0, 'losses': 0, 'draws': 0}
        self.update_stats_display()
        
        # AI moves first
        self.make_ai_move()

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.create_board_gui()
        self.create_status_label()
        self.create_control_buttons()
        self.create_learning_display()
        
        # Configure button styles
        self.style.configure('Game.TButton', font=('Arial', 20, 'bold'), padding=20)
        self.style.configure('Control.TButton', font=('Arial', 12))

    def create_learning_display(self):
        """Create display for learning progress"""
        self.learning_label = ttk.Label(
            self.main_frame,
            text=f"Learning Progress: 0/{self.training_episodes} games",
            font=('Arial', 12),
            padding=5
        )
        self.learning_label.grid(row=3, column=0, columnspan=2)

    def load_best_model(self):
        """Load the X agent model"""
        try:
            if self.ai_agent.load_model('models/agent_x_final.pkl'):
                print("Loaded X agent model")
            else:
                messagebox.showerror("Error", "Could not load AI model")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", "Failed to load AI model")

    def create_board_gui(self):
        """Create the game board buttons"""
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        for i in range(3):
            row_buttons = []
            for j in range(3):
                button = ttk.Button(
                    self.board_frame,
                    text="",
                    style='Game.TButton',
                    command=lambda row=i, col=j: self.make_move(row, col)
                )
                button.grid(row=i, column=j, padx=2, pady=2)
                row_buttons.append(button)
            self.buttons.append(row_buttons)

    def create_status_label(self):
        """Create the game status display"""
        self.status_label = ttk.Label(
            self.main_frame,
            text="AI's turn (X)",
            font=('Arial', 14),
            padding=10
        )
        self.status_label.grid(row=1, column=0, columnspan=2)

    def create_control_buttons(self):
        """Create control buttons"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.reset_button = ttk.Button(
            control_frame,
            text="New Game",
            style='Control.TButton',
            command=self.reset_game
        )
        self.reset_button.grid(row=0, column=0, padx=5)
        
        self.stats_button = ttk.Button(
            control_frame,
            text="View Stats",
            style='Control.TButton',
            command=self.show_stats
        )
        self.stats_button.grid(row=0, column=1, padx=5)

    def make_move(self, row, col):
        """Handle player move (O) and AI response with learning"""
        if self.game_over or self.board[row, col] != 0:
            return

        # Player move (O)
        self.board[row, col] = -1
        self.buttons[row][col].configure(text="O")
        
        # Check game state after player move
        if self.check_winner(-1):
            # AI learns from loss
            if self.last_state and self.last_action:
                self.ai_agent.learn(
                    self.last_state,
                    self.last_action,
                    -20,  # Negative reward for losing
                    self.ai_agent.get_state_key(self.board),
                    True
                )
            self.end_game("You win!")
            self.stats['wins'] += 1
            return
        elif self.is_board_full():
            # AI learns from draw
            if self.last_state and self.last_action:
                self.ai_agent.learn(
                    self.last_state,
                    self.last_action,
                    5,  # Small positive reward for draw
                    self.ai_agent.get_state_key(self.board),
                    True
                )
            self.end_game("It's a draw!")
            self.stats['draws'] += 1
            return

        # AI's turn
        self.make_ai_move()

    def make_ai_move(self):
        """Handle AI move (X) with learning"""
        if self.game_over:
            return

        self.status_label.configure(text="AI thinking...")
        self.root.update()
        
        # Store current state for learning
        current_state = self.ai_agent.get_state_key(self.board)
        
        # Get AI's move with some randomness
        if np.count_nonzero(self.board) == 0:  # First move
            if np.random.random() < 0.3:  # 30% chance of random first move
                possible_moves = [(0,0), (0,2), (1,1), (2,0), (2,2)]
                ai_action = possible_moves[np.random.randint(len(possible_moves))]
            else:
                ai_action = self.get_best_ai_move()
        else:
            ai_action = self.get_best_ai_move()

        if ai_action:
            ai_row, ai_col = ai_action
            self.last_state = current_state
            self.last_action = ai_action
            
            self.board[ai_row, ai_col] = 1
            self.buttons[ai_row][ai_col].configure(text="X")
            
            # Check game state after AI move
            if self.check_winner(1):
                # AI learns from win
                self.ai_agent.learn(
                    current_state,
                    ai_action,
                    25,  # High reward for winning
                    self.ai_agent.get_state_key(self.board),
                    True
                )
                self.end_game("AI wins!")
                self.stats['losses'] += 1
            elif self.is_board_full():
                # AI learns from draw
                self.ai_agent.learn(
                    current_state,
                    ai_action,
                    5,  # Small positive reward for draw
                    self.ai_agent.get_state_key(self.board),
                    True
                )
                self.end_game("It's a draw!")
                self.stats['draws'] += 1
            else:
                self.status_label.configure(text="Your turn (O)")

    def get_best_ai_move(self):
        """Get the best move for AI with multiple attempts"""
        best_action = None
        best_value = float('-inf')
        
        # Try multiple times to get the best move
        for _ in range(3):
            action = self.ai_agent.get_action(self.board)
            if action:
                state_key = self.ai_agent.get_state_key(self.board)
                value = self.ai_agent.q_table[state_key][action[0] * 3 + action[1]]
                if value > best_value:
                    best_value = value
                    best_action = action
        
        return best_action

    def check_winner(self, player):
        """Check if the specified player has won"""
        # Check rows and columns
        for i in range(3):
            if all(self.board[i,:] == player) or all(self.board[:,i] == player):
                return True
        # Check diagonals
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_board_full(self):
        """Check if the board is full (draw)"""
        return np.all(self.board != 0)

    def end_game(self, message):
        """Handle game end and learning updates"""
        self.game_over = True
        self.status_label.configure(text=message)
        self.games_played += 1
        
        # Update learning progress
        self.learning_label.configure(
            text=f"Learning Progress: {self.games_played}/{self.training_episodes} games"
        )
        
        # Save model periodically
        if self.games_played % 10 == 0:  # Save every 10 games
            self.ai_agent.save_model('models/agent_x_learning.pkl')
        
        # Check if training is complete
        if self.games_played >= self.training_episodes:
            messagebox.showinfo(
                "Training Complete",
                f"AI has learned from {self.training_episodes} games!\n"
                f"Win Rate: {(self.stats['losses']/self.training_episodes)*100:.1f}%\n"
                f"Final model saved as 'agent_x_learning.pkl'"
            )
        
        self.update_stats_display()

    def reset_game(self):
        """Reset the game board and state"""
        self.board = np.zeros((3, 3))
        self.game_over = False
        self.last_state = None
        self.last_action = None
        
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].configure(text="")
        
        self.status_label.configure(text="AI's turn (X)")
        # AI moves first
        self.make_ai_move()

    def update_stats_display(self):
        """Update the statistics display"""
        total_games = sum(self.stats.values())
        if total_games > 0:
            win_rate = (self.stats['wins'] / total_games) * 100
            self.stats_text = f"Games: {total_games}, Win Rate: {win_rate:.1f}%"
        else:
            self.stats_text = "No games played yet"

    def show_stats(self):
        """Display game statistics"""
        messagebox.showinfo("Game Statistics", 
            f"Games Played: {sum(self.stats.values())}\n"
            f"Wins: {self.stats['wins']}\n"
            f"Losses: {self.stats['losses']}\n"
            f"Draws: {self.stats['draws']}\n\n"
            f"{self.stats_text}")

def main():
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()