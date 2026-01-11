"""
Training Script for Campus Navigation Agent
------------------------------------------
This script trains a Q-Learning reinforcement learning agent to navigate
a campus-like grid environment and compares its performance with A* search.

EDUCATIONAL NOTE:
Q-Learning is a model-free reinforcement learning algorithm that learns
an optimal policy through trial and error. The agent learns by:
1. Interacting with the environment (taking actions)
2. Receiving rewards (positive for progress, negative for bad moves)
3. Updating its Q-table (state-action value function)

The grid abstraction makes the state space finite and manageable,
allowing the Q-Learning algorithm to converge to an optimal policy.

Run with: python train_agent.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from campus_environment import CampusEnvironment
from qlearning import QLearningAgent
from astar import AStarPlanner



def plot_training_curves(agent, save_path):
    """
    Generate and save training performance visualizations.
    
    This function creates four plots to analyze the learning process:
    1. Episode rewards - shows how rewards improve over training
    2. Steps to goal - tracks path efficiency improvements
    3. Success rate - percentage of episodes that reached the goal
    4. Distribution - overall distribution of episode lengths
    
    These visualizations help verify that the agent is learning effectively.
    
    Parameters:
    - agent: Trained QLearningAgent with episode history data
    - save_path: File path where the figure will be saved
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Episode rewards over time
    axes[0, 0].plot(agent.episode_rewards, alpha=0.6, color='blue')
    axes[0, 0].set_title("Episode Reward Over Time", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # 2️⃣ Episode length
    axes[0, 1].plot(agent.episode_lengths, alpha=0.6, color='green')
    axes[0, 1].set_title("Steps to Goal", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True, alpha=0.3)

    # 3️⃣ Success rate
    window = 100
    success_rates = []
    for i in range(len(agent.episode_lengths)):
        recent = agent.episode_lengths[max(0, i - window):i + 1]
        success_rates.append(
            sum(1 for x in recent if x < 200) / len(recent) * 100
        )

    axes[1, 0].plot(success_rates, color='purple')
    axes[1, 0].set_title("Success Rate (%)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)

    # 4️⃣ Distribution
    axes[1, 1].hist(agent.episode_lengths, bins=40, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Distribution of Episode Lengths", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"SUCCESS: Training curves saved to: {save_path}")
    plt.close()


def train_and_evaluate(grid_size=15, obstacle_density=0.2, num_episodes=10000):
    """
    Main training pipeline for Q-Learning agent.
    
    This function orchestrates the complete training process:
    1. Creates the environment
    2. Initializes the Q-Learning agent with hyperparameters
    3. Trains the agent for multiple episodes
    4. Saves the trained model and environment configuration
    5. Generates training visualizations
    6. Compares performance with A* search algorithm
    
    Parameters:
    - grid_size: Size of the grid (width and height)
    - obstacle_density: Probability of obstacles (0.0 to 1.0)
    - num_episodes: Number of training episodes
    
    This is the primary entry point for training the agent.
    """

    print("=" * 60)
    print("TRAINING Q-LEARNING AGENT FOR CAMPUS NAVIGATION")
    print("=" * 60)

    # Create environment with specified parameters
    # Using consistent parameters allows the agent to learn the specific environment
    env = CampusEnvironment(width=grid_size, height=grid_size, obstacle_prob=obstacle_density)

    # Check if a pre-trained model exists for this configuration
    model_filename = f"models/qlearning_agent_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    config_filename = f"models/environment_config_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    
    if os.path.exists(model_filename) and os.path.exists(config_filename):
        print("\nFOUND EXISTING MODEL - Continuing training from previous Q-table...")
        print("This allows the agent to improve further by building on prior learning.")
        
        # Load existing environment configuration
        with open(config_filename, "rb") as f:
            env_config = pickle.load(f)
        
        # Recreate the exact same environment
        env = CampusEnvironment(width=env_config["width"], height=env_config["height"])
        env.start = env_config["start"]
        env.goal = env_config["goal"]
        env.static_grid = env_config["static_grid"]
        env.grid = env_config["static_grid"].copy()
        
        # Load existing model data
        with open(model_filename, "rb") as f:
            saved_data = pickle.load(f)
        
        # Initialize agent with existing Q-table
        agent = QLearningAgent(
            environment=env,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=saved_data.get("epsilon", 0.1),  # Resume with previous epsilon
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )
        agent.q_table = saved_data["q_table"]  # Continue from existing knowledge
        
        print(f"Loaded Q-table with shape: {agent.q_table.shape}")
        print(f"Resuming with epsilon: {agent.epsilon:.4f}")
    else:
        print("\nNO EXISTING MODEL - Training from scratch...")
        
        # Initialize Q-Learning agent with carefully tuned hyperparameters
        agent = QLearningAgent(
            environment=env,
            learning_rate=0.1,        # Alpha: How much new info overrides old (0.1 = gradual learning)
            discount_factor=0.99,     # Gamma: Importance of future rewards (0.99 = values long-term planning)
            epsilon=1.0,              # Start with 100% exploration (random actions)
            epsilon_decay=0.9995,     # Slow decay allows more exploration time
            epsilon_min=0.01          # Always keep 1% exploration to avoid local optima
        )

    # Training configuration
    # More episodes = more learning opportunities for the agent
    print(f"Training for {num_episodes} episodes...")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Obstacle density: {obstacle_density:.2f}")
    print(f"Start: {env.start}, Goal: {env.goal}")

    # Execute training loop
    # max_steps increased to 300 to allow agent time to explore 15x15 grid
    agent.train(
        num_episodes=num_episodes,
        max_steps=300,
        verbose=True
    )

    # Save the trained model to disk for later use in the Streamlit app
    # Use descriptive filename that includes training parameters
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/qlearning_agent_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    agent.save_model(model_filename)
    print(f"Model saved to: {model_filename}")

    # Save environment configuration with same naming convention
    config_filename = f"models/environment_config_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "width": env.width,
            "height": env.height,
            "start": env.start,
            "goal": env.goal,
            "static_grid": env.static_grid
        }, f)

    # Plot curves
    os.makedirs("data", exist_ok=True)
    plot_training_curves(agent, "data/training_curves.png")

    # Benchmark against A* search algorithm for comparison
    # A* provides an optimal baseline to evaluate Q-Learning performance
    planner = AStarPlanner(env)
    path, cost, stats = planner.search()

    print("\nTraining completed successfully.")
    print(f"Final epsilon: {agent.epsilon:.4f}")  # Should be close to epsilon_min
    
    # Print summary statistics from training
    if len(agent.episode_lengths) > 0:
        avg_last_100 = np.mean(agent.episode_lengths[-100:])
        min_steps = np.min(agent.episode_lengths)
        max_steps = np.max(agent.episode_lengths)
        
        print(f"Average Steps (last 100 episodes): {avg_last_100:.2f}")
        print(f"Minimum Steps Achieved: {min_steps}")
        print(f"Maximum Steps: {max_steps}")
    
    # Compare Q-Learning with A* algorithm
    if path:
        print(f"\nSUCCESS: A* found path with {len(path)} steps")
        print(f"   Nodes expanded: {stats['nodes_expanded']}")
        
        # Test the learned Q-Learning policy
        q_path, q_cost = agent.get_policy_path(env.start)
        if q_path and len(q_path) > 0:
            reached_goal = q_path[-1] == env.goal
            status = "REACHED GOAL!" if reached_goal else "did not reach goal"
            print(f"SUCCESS: Q-Learning path: {len(q_path)} steps ({status})")
            if reached_goal:
                diff = len(q_path) - len(path)
                efficiency = (len(path) / len(q_path) * 100)
                print(f"   Difference: {diff:+d} steps")
                print(f"   Efficiency: {efficiency:.1f}%")
        else:
            print("ERROR: Q-Learning failed to find path in test")
    
    # Calculate overall training success rate
    success_count = sum(1 for length in agent.episode_lengths if length < 300)
    success_rate = (success_count / len(agent.episode_lengths)) * 100
    print(f"\nTraining Success Rate: {success_rate:.1f}%")
    print(f"   Episodes that reached goal: {success_count}/{len(agent.episode_lengths)}")
    
    print("\n" + "=" * 60)
    print("ALL DONE! You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command-line arguments for flexible training
    parser = argparse.ArgumentParser(description="Train Q-Learning agent for campus navigation")
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size (width and height)')
    parser.add_argument('--obstacle-density', type=float, default=0.2, help='Obstacle probability (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.grid_size < 5 or args.grid_size > 50:
        print("ERROR: Grid size must be between 5 and 50")
        sys.exit(1)
    if args.obstacle_density < 0.0 or args.obstacle_density > 0.8:
        print("ERROR: Obstacle density must be between 0.0 and 0.8")
        sys.exit(1)
    
    train_and_evaluate(
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        num_episodes=args.episodes
    )