"""
Training Script for Campus Navigation Agent
Train Q-Learning agent and compare with A* search baseline.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from campus_environment import CampusEnvironment
from qlearning import QLearningAgent
from astar import AStarPlanner


def plot_training_curves(agent: QLearningAgent, save_path: str = None):
    """Plot training statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards over episodes
    axes[0, 0].plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
    # Moving average
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.episode_rewards)), 
                       moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(agent.episode_lengths, alpha=0.6, label='Episode Length')
    if len(agent.episode_lengths) >= window:
        moving_avg = np.convolve(agent.episode_lengths, 
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(agent.episode_lengths)), 
                       moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate over time
    window_size = 100
    success_rate = []
    for i in range(len(agent.episode_lengths)):
        start_idx = max(0, i - window_size + 1)
        window_episodes = agent.episode_lengths[start_idx:i+1]
        successes = sum(1 for length in window_episodes if length < 200)
        success_rate.append(successes / len(window_episodes) * 100)
    
    axes[1, 0].plot(success_rate, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title(f'Success Rate (Rolling {window_size}-Episode Window)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    # Distribution of episode lengths
    axes[1, 1].hist(agent.episode_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Episode Length (Steps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Episode Lengths')
    axes[1, 1].axvline(x=np.median(agent.episode_lengths), color='r', 
                      linestyle='--', linewidth=2, label=f'Median: {np.median(agent.episode_lengths):.1f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def compare_algorithms(env: CampusEnvironment, agent: QLearningAgent, 
                       num_tests: int = 50):
    """Compare A* and Q-Learning performance."""
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    astar_planner = AStarPlanner(env, heuristic='euclidean')
    
    astar_successes = 0
    astar_costs = []
    astar_times = []
    
    ql_successes = 0
    ql_costs = []
    ql_path_lengths = []
    
    for test in range(num_tests):
        # Reset environment with random obstacles
        env.reset()
        env.add_dynamic_obstacles(count=np.random.randint(3, 8))
        
        # Test A* (without allowing congestion)
        path, cost, stats = astar_planner.search(allow_diagonal=False, allow_congestion=False)
        if stats['success']:
            astar_successes += 1
            astar_costs.append(cost)
            astar_times.append(stats['nodes_expanded'])
        
        # Test Q-Learning
        path, cost = agent.get_policy_path(max_steps=200)
        if path[-1] == env.goal:
            ql_successes += 1
            ql_costs.append(cost)
            ql_path_lengths.append(len(path))
    
    print(f"\nTests performed: {num_tests}")
    print(f"\n{'Algorithm':<20} {'Success Rate':<15} {'Avg Cost':<15} {'Avg Metric'}")
    print("-" * 70)
    
    astar_success_rate = astar_successes / num_tests * 100
    astar_avg_cost = np.mean(astar_costs) if astar_costs else float('inf')
    astar_avg_nodes = np.mean(astar_times) if astar_times else 0
    
    print(f"{'A* Search':<20} {astar_success_rate:>6.1f}%        "
          f"{astar_avg_cost:>8.2f}        {astar_avg_nodes:.1f} nodes expanded")
    
    ql_success_rate = ql_successes / num_tests * 100
    ql_avg_cost = np.mean(ql_costs) if ql_costs else float('inf')
    ql_avg_length = np.mean(ql_path_lengths) if ql_path_lengths else 0
    
    print(f"{'Q-Learning':<20} {ql_success_rate:>6.1f}%        "
          f"{ql_avg_cost:>8.2f}        {ql_avg_length:.1f} steps")
    
    print("\n" + "="*70)
    
    return {
        'astar_success_rate': astar_success_rate,
        'astar_avg_cost': astar_avg_cost,
        'ql_success_rate': ql_success_rate,
        'ql_avg_cost': ql_avg_cost
    }


def train_and_evaluate():
    """Main training and evaluation function."""
    print("="*70)
    print("CAMPUS NAVIGATION AGENT - TRAINING")
    print("="*70)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create environment
    print("\n[1/5] Creating campus environment...")
    env = CampusEnvironment(width=15, height=15, obstacle_prob=0.20)
    print(f"   ✓ Environment created: {env.width}x{env.height} grid")
    print(f"   ✓ Start: {env.start}, Goal: {env.goal}")
    
    # Create Q-Learning agent
    print("\n[2/5] Initializing Q-Learning agent...")
    agent = QLearningAgent(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    print(f"   ✓ Agent initialized")
    print(f"   ✓ Learning rate (α): {agent.alpha}")
    print(f"   ✓ Discount factor (γ): {agent.gamma}")
    print(f"   ✓ Initial exploration (ε): {agent.epsilon}")
    
    # Train agent
    print("\n[3/5] Training Q-Learning agent...")
    print("-" * 70)
    num_episodes = 2000
    agent.train(num_episodes=num_episodes, max_steps=200, verbose=True)
    
    # Save model
    print("\n[4/5] Saving trained model...")
    os.makedirs('models', exist_ok=True)
    agent.save_model('models/qlearning_agent.pkl')
    
    # Save environment configuration
    env_config = {
        'width': env.width,
        'height': env.height,
        'start': env.start,
        'goal': env.goal,
        'static_grid': env.static_grid
    }
    import pickle
    with open('models/environment_config.pkl', 'wb') as f:
        pickle.dump(env_config, f)
    print("   ✓ Environment configuration saved")
    
    # Plot training curves
    print("\n[5/5] Generating training visualizations...")
    os.makedirs('data', exist_ok=True)
    plot_training_curves(agent, save_path='data/training_curves.png')
    
    # Compare algorithms
    print("\nEvaluating and comparing algorithms...")
    comparison_results = compare_algorithms(env, agent, num_tests=50)
    
    # Final statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total episodes trained: {num_episodes}")
    print(f"Final exploration rate: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Average steps (last 100 episodes): {np.mean(agent.episode_lengths[-100:]):.1f}")
    
    success_in_last_100 = sum(1 for length in agent.episode_lengths[-100:] if length < 200)
    print(f"Success rate (last 100 episodes): {success_in_last_100}%")
    
    print("\n" + "="*70)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\n✓ All files saved:")
    print("  - models/qlearning_agent.pkl (Trained Q-Learning agent)")
    print("  - models/environment_config.pkl (Environment configuration)")
    print("  - data/training_curves.png (Training visualization)")
    
    print("\nNext steps:")
    print("  Run 'streamlit run app.py' to launch the interactive web application")


if __name__ == "__main__":
    train_and_evaluate()
