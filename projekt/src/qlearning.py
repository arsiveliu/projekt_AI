"""
Q-Learning Agent for Campus Navigation
Implements Q-Learning reinforcement learning for adaptive path planning in dynamic environments.

Reference implementations:
- tayfunkscu/QLearning-path-planning
- zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation
"""
import numpy as np
import random
import pickle
from typing import Tuple, List, Optional
from campus_environment import CampusEnvironment


class QLearningAgent:
    """
    Q-Learning agent for navigation with dynamic obstacles.
    
    Q-Learning update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    
    Where:
    - s: current state
    - a: action taken
    - r: reward received
    - s': next state
    - α: learning rate
    - γ: discount factor
    """
    
    # Action space: [up, down, left, right]
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
    
    def __init__(self, environment: CampusEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            environment: Campus environment
            learning_rate (α): Learning rate for Q-value updates
            discount_factor (γ): Discount factor for future rewards
            epsilon (ε): Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table: Q[state][action]
        num_states = self.env.get_num_states()
        num_actions = len(self.ACTIONS)
        self.q_table = np.zeros((num_states, num_actions))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def get_state(self, position: Tuple[int, int]) -> int:
        """Convert position to state index."""
        return self.env.get_state_representation(position)
    
    def get_position(self, state: int) -> Tuple[int, int]:
        """Convert state index to position."""
        return self.env.get_position_from_state(state)
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, len(self.ACTIONS) - 1)
        else:
            # Exploitation: best known action
            return np.argmax(self.q_table[state])
    
    def get_valid_actions(self, position: Tuple[int, int]) -> List[int]:
        """Get list of valid action indices from current position."""
        valid_actions = []
        for i, action in enumerate(self.ACTIONS):
            next_pos = (position[0] + action[0], position[1] + action[1])
            # Allow congestion during training (agent learns to avoid it)
            if self.env.is_walkable(next_pos, allow_congestion=True):
                valid_actions.append(i)
        return valid_actions if valid_actions else [0]  # Default to first action if stuck
    
    def take_action(self, position: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action and return next state, reward, and done flag.
        
        Args:
            position: Current position
            action_idx: Index of action to take
            
        Returns:
            Tuple of (next_position, reward, done)
        """
        action = self.ACTIONS[action_idx]
        next_pos = (position[0] + action[0], position[1] + action[1])
        
        # Check if action is valid
        if not self.env.is_valid_position(next_pos):
            # Hit boundary - stay in place with penalty
            return position, -10.0, False
        
        cell_value = self.env.grid[next_pos[0]][next_pos[1]]
        
        # Calculate reward based on cell type
        if cell_value == 1:  # Static obstacle (building)
            # Can't move here - stay in place with large penalty
            return position, -50.0, False
        
        elif cell_value == 2:  # Dynamic obstacle (congestion)
            # Can move but with penalty
            reward = -5.0
            done = False
        
        elif next_pos == self.env.goal:
            # Reached goal - large positive reward
            reward = 100.0
            done = True
        
        else:
            # Normal movement - small negative reward (encourages shorter paths)
            distance_to_goal = self.env.euclidean_distance(next_pos, self.env.goal)
            old_distance = self.env.euclidean_distance(position, self.env.goal)
            
            # Reward for moving closer to goal
            if distance_to_goal < old_distance:
                reward = -0.5  # Small penalty but moving right direction
            else:
                reward = -1.0  # Penalty for moving away
            
            done = False
        
        return next_pos, reward, done
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Track loss for monitoring
        loss = abs(reward + self.gamma * max_next_q - current_q)
        return loss
    
    def train_episode(self, max_steps: int = 200) -> Tuple[float, int, bool]:
        """
        Train agent for one episode.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total_reward, steps_taken, success)
        """
        # Reset environment
        position = self.env.reset()
        
        # Randomly add dynamic obstacles
        self.env.add_dynamic_obstacles(count=random.randint(3, 8))
        
        total_reward = 0.0
        episode_loss = 0.0
        
        for step in range(max_steps):
            # Get current state
            state = self.get_state(position)
            
            # Select action
            action = self.select_action(state, training=True)
            
            # Take action
            next_position, reward, done = self.take_action(position, action)
            next_state = self.get_state(next_position)
            
            # Update Q-table
            loss = self.update_q_value(state, action, reward, next_state)
            episode_loss += loss
            
            # Update statistics
            total_reward += reward
            position = next_position
            
            if done:
                # Reached goal
                return total_reward, step + 1, True
        
        # Max steps reached without finding goal
        self.training_losses.append(episode_loss / max_steps)
        return total_reward, max_steps, False
    
    def train(self, num_episodes: int = 1000, max_steps: int = 200, verbose: bool = True):
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print training progress
        """
        success_count = 0
        
        for episode in range(num_episodes):
            reward, steps, success = self.train_episode(max_steps)
            
            # Store statistics
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            
            if success:
                success_count += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                recent_success_rate = sum([1 for i in range(max(0, episode-99), episode+1) 
                                          if self.episode_lengths[i] < max_steps]) / min(100, episode+1)
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_lengths[-100:])
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Success Rate: {recent_success_rate:.2%} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        final_success_rate = success_count / num_episodes
        print(f"\nTraining completed!")
        print(f"Overall success rate: {final_success_rate:.2%}")
        print(f"Final epsilon: {self.epsilon:.3f}")
    
    def get_policy_path(self, start: Optional[Tuple[int, int]] = None,
                       goal: Optional[Tuple[int, int]] = None,
                       max_steps: int = 200) -> Tuple[List[Tuple[int, int]], float]:
        """
        Get path using learned Q-table (greedy policy).
        
        Args:
            start: Starting position
            goal: Goal position
            max_steps: Maximum steps
            
        Returns:
            Tuple of (path, total_cost)
        """
        position = start or self.env.start
        goal = goal or self.env.goal
        
        path = [position]
        total_cost = 0.0
        
        for _ in range(max_steps):
            if position == goal:
                break
            
            state = self.get_state(position)
            action = self.select_action(state, training=False)  # Greedy
            
            next_position, cost, done = self.take_action(position, action)
            
            path.append(next_position)
            total_cost += abs(cost)
            position = next_position
            
            if done:
                break
        
        return path, total_cost
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters to file."""
        model_data = {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_lengths = model_data.get('episode_lengths', [])
        self.training_losses = model_data.get('training_losses', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Trained episodes: {len(self.episode_rewards)}")


# Example usage
if __name__ == "__main__":
    # Create environment
    env = CampusEnvironment(width=10, height=10, obstacle_prob=0.15)
    print("Campus Environment:")
    print(env)
    
    # Create Q-Learning agent
    agent = QLearningAgent(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\nTraining Q-Learning agent...")
    agent.train(num_episodes=500, max_steps=100, verbose=True)
    
    # Test learned policy
    print("\n--- Testing Learned Policy ---")
    env.add_dynamic_obstacles(count=5)
    path, cost = agent.get_policy_path()
    
    print(f"Path length: {len(path)}")
    print(f"Total cost: {cost:.2f}")
    print(f"Path: {' -> '.join([str(p) for p in path[:10]])}...")
