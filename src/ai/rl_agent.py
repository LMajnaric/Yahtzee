# ai/rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import pickle

from src.models import Category, GameState
from src.yahtzee_game import YahtzeeGame
from src.scoring import ScoreCalculator

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class StateEncoder:
    """Encodes game state into neural network input"""
    
    def __init__(self, num_dice=6):
        self.num_dice = num_dice
        self.categories = list(Category)
        self.state_size = self._calculate_state_size()
    
    def _calculate_state_size(self):
        """Calculate total state vector size"""
        size = 0
        size += self.num_dice * 7  # Dice one-hot encoding (0-6, 0 means no die)
        size += 1  # Roll number (normalized 0-1)
        size += 4 * len(self.categories)  # Filled categories (binary for each column)
        size += 4 * len(self.categories)  # Scores (normalized)
        size += 4  # Column bonuses
        size += 4  # Special scores
        size += 4  # Available categories count per column
        size += 1  # Turn number (normalized)
        return size
    
    def encode_state(self, game_state: GameState, current_dice: List[int], 
                    current_roll: int, available_categories: Dict[int, List[Category]]) -> np.ndarray:
        """Encode game state into feature vector"""
        features = []
        
        # Encode current dice (one-hot for each die position)
        for i in range(self.num_dice):
            die_encoding = [0] * 7  # 0-6
            if i < len(current_dice):
                die_encoding[current_dice[i]] = 1
            else:
                die_encoding[0] = 1  # No die
            features.extend(die_encoding)
        
        # Current roll number (normalized)
        features.append(current_roll / 3.0)
        
        # Filled categories for each column (binary)
        for col in range(1, 5):
            for category in self.categories:
                features.append(1 if category in game_state.filled_categories[col] else 0)
        
        # Scores for each column (normalized by max possible score ~500)
        for col in range(1, 5):
            for category in self.categories:
                score = game_state.scores[col].get(category, 0)
                features.append(score / 500.0)
        
        # Column bonuses
        for col in range(1, 5):
            features.append(game_state.column_bonuses[col] / 30.0)
        
        # Special scores  
        for col in range(1, 5):
            features.append(game_state.special_scores[col] / 100.0)
        
        # Available categories count per column
        for col in range(1, 5):
            features.append(len(available_categories.get(col, [])) / len(self.categories))
        
        # Turn number (normalized by max turns = 48)
        features.append(game_state.current_turn / 48.0)
        
        return np.array(features, dtype=np.float32)

class ActionEncoder:
    """Encodes and decodes actions"""
    
    def __init__(self, num_dice=6):
        self.num_dice = num_dice
        self.categories = list(Category)
        
        # Action types: dice_selection, scoring_decision
        self.dice_actions = 2 ** num_dice  # All possible dice keep combinations
        self.scoring_actions = 4 * len(self.categories)  # Column x Category combinations
        self.total_actions = self.dice_actions + self.scoring_actions
    
    def encode_dice_action(self, keep_indices: List[int]) -> int:
        """Encode dice keeping decision as integer"""
        action = 0
        for i in keep_indices:
            action |= (1 << i)
        return action
    
    def decode_dice_action(self, action: int) -> List[int]:
        """Decode integer back to dice indices"""
        keep_indices = []
        for i in range(self.num_dice):
            if action & (1 << i):
                keep_indices.append(i)
        return keep_indices
    
    def encode_scoring_action(self, column: int, category: Category) -> int:
        """Encode scoring decision"""
        cat_idx = self.categories.index(category)
        return self.dice_actions + (column - 1) * len(self.categories) + cat_idx
    
    def decode_scoring_action(self, action: int) -> Tuple[int, Category]:
        """Decode scoring action"""
        action -= self.dice_actions
        column = (action // len(self.categories)) + 1
        category = self.categories[action % len(self.categories)]
        return column, category

class DQN(nn.Module):
    """Deep Q-Network for Yahtzee"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent for Yahtzee"""
    
    def __init__(self, num_dice: int = 6, learning_rate: float = 0.001, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, gamma: float = 0.95,
                 batch_size: int = 32, update_target_freq: int = 1000):
        
        self.num_dice = num_dice
        self.state_encoder = StateEncoder(num_dice)
        self.action_encoder = ActionEncoder(num_dice)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.state_encoder.state_size, 
                            self.action_encoder.total_actions).to(self.device)
        self.target_network = DQN(self.state_encoder.state_size, 
                                 self.action_encoder.total_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Training metrics
        self.episode_scores = []
        self.episode_rewards = []
        self.losses = []
        self.step_count = 0
    
    def get_valid_actions(self, game_state: GameState, current_dice: List[int], 
                         current_roll: int, phase: str) -> List[int]:
        """Get valid actions for current state"""
        valid_actions = []
        
        if phase == "dice":
            # All dice combinations are valid
            for action in range(self.action_encoder.dice_actions):
                valid_actions.append(action)
        
        elif phase == "scoring":
            # Only valid scoring actions
            for col in range(1, 5):
                available_cats = self._get_available_categories(game_state, col)
                for category in available_cats:
                    action = self.action_encoder.encode_scoring_action(col, category)
                    valid_actions.append(action)
        
        return valid_actions
    
    def _get_available_categories(self, game_state: GameState, column: int) -> List[Category]:
        """Get available categories for a column (simplified)"""
        from src.game_logic import ColumnConstraints
        return ColumnConstraints.get_available_categories(
            column, game_state.filled_categories[column]
        )
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            mask = torch.full_like(q_values, float('-inf'))
            mask[0, valid_actions] = 0
            masked_q_values += mask
            
            return masked_q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def play_episode(self, training: bool = True) -> Tuple[int, float]:
        """Play one complete episode"""
        game = YahtzeeGame(self.num_dice)
        game.start_new_game()
        
        total_reward = 0
        episode_experiences = []
        
        while not game.get_game_state().game_over:
            # Get current state
            game_state = game.get_game_state()
            available_cats = {col: game.get_available_categories(col) 
                             for col in range(1, 5)}
            
            # Dice rolling phase
            current_dice = []
            current_roll = 0
            
            for roll_num in range(3):  # Maximum 3 rolls
                current_roll = roll_num + 1
                
                if roll_num == 0:
                    # First roll - roll all dice
                    dice_result = game.roll_dice()
                    current_dice = dice_result.dice
                else:
                    # Subsequent rolls - choose which dice to keep
                    state = self.state_encoder.encode_state(
                        game_state, current_dice, current_roll, available_cats
                    )
                    
                    valid_dice_actions = self.get_valid_actions(
                        game_state, current_dice, current_roll, "dice"
                    )
                    
                    dice_action = self.choose_action(state, valid_dice_actions, training)
                    keep_indices = self.action_encoder.decode_dice_action(dice_action)
                    
                    dice_result = game.roll_dice(keep_indices)
                    current_dice = dice_result.dice
                
                # Check if we want to stop rolling (simplified logic)
                if self._should_stop_rolling(current_dice, current_roll):
                    break
            
            # Scoring phase
            state = self.state_encoder.encode_state(
                game_state, current_dice, current_roll, available_cats
            )
            
            valid_scoring_actions = self.get_valid_actions(
                game_state, current_dice, current_roll, "scoring"
            )
            
            scoring_action = self.choose_action(state, valid_scoring_actions, training)
            column, category = self.action_encoder.decode_scoring_action(scoring_action)
            
            # Handle column 4 announcement if needed
            if column == 4:
                game.announce_category(column, category)
            
            # Execute scoring
            old_score = game.logic.get_total_score()
            success = game.score_category(column, category)
            new_score = game.logic.get_total_score()
            
            # Calculate reward
            reward = self._calculate_reward(old_score, new_score, success, game_state)
            total_reward += reward
            
            # Store experience for training
            if training:
                next_state = self.state_encoder.encode_state(
                    game.get_game_state(), [], 0, 
                    {col: game.get_available_categories(col) for col in range(1, 5)}
                )
                
                self.store_experience(
                    state, scoring_action, reward, next_state, 
                    game.get_game_state().game_over
                )
                
                # Train periodically
                if len(self.replay_buffer) > self.batch_size:
                    self.train_step()
        
        final_score = game.logic.get_total_score()
        
        if training:
            self.episode_scores.append(final_score)
            self.episode_rewards.append(total_reward)
        
        return final_score, total_reward
    
    def _should_stop_rolling(self, dice: List[int], roll_number: int) -> bool:
        """Simple heuristic for when to stop rolling"""
        # This is a simplified heuristic - could be improved
        dice_counts = {}
        for die in dice:
            dice_counts[die] = dice_counts.get(die, 0) + 1
        
        max_count = max(dice_counts.values())
        
        # Stop if we have 4+ of a kind, or on final roll
        return max_count >= 4 or roll_number >= 3
    
    def _calculate_reward(self, old_score: int, new_score: int, success: bool, 
                         game_state: GameState) -> float:
        """Calculate reward for the action"""
        if not success:
            return -10  # Penalty for invalid action
        
        score_gain = new_score - old_score
        
        # Normalize score gain and add small completion bonus
        reward = score_gain / 100.0
        
        # Bonus for completing categories efficiently
        if game_state.game_over:
            reward += 10  # Game completion bonus
        
        return reward
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_scores': self.episode_scores,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'batch_size': self.batch_size
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_scores = checkpoint['episode_scores']
    
    def get_average_score(self, last_n: int = 100) -> float:
        """Get average score over last N episodes"""
        if len(self.episode_scores) == 0:
            return 0
        return np.mean(self.episode_scores[-last_n:])

# Training script
class YahtzeeTrainer:
    """Manages DQN training process"""
    
    def __init__(self, agent: DQNAgent, save_interval: int = 1000):
        self.agent = agent
        self.save_interval = save_interval
    
    def train(self, episodes: int, verbose: bool = True):
        """Train the agent for specified episodes"""
        best_avg_score = -float('inf')
        
        for episode in range(episodes):
            score, reward = self.agent.play_episode(training=True)
            
            if verbose and episode % 100 == 0:
                avg_score = self.agent.get_average_score()
                print(f"Episode {episode}, Score: {score}, Avg Score: {avg_score:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Save best model
            if episode % self.save_interval == 0:
                avg_score = self.agent.get_average_score()
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    self.agent.save_model(f"yahtzee_dqn_best_{episode}.pth")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self.agent.save_model(f"yahtzee_dqn_checkpoint_{episode}.pth")

# Example usage
if __name__ == "__main__":
    # Create and train agent
    agent = DQNAgent(num_dice=6)
    trainer = YahtzeeTrainer(agent)
    
    print("Starting training...")
    trainer.train(episodes=10000)
    
    # Test trained agent
    print("\nTesting trained agent...")
    total_score = 0
    num_tests = 100
    
    for _ in range(num_tests):
        score, _ = agent.play_episode(training=False)
        total_score += score
    
    print(f"Average test score: {total_score / num_tests:.2f}")