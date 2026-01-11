"""
app.py - Campus Navigation Intelligent Agent - Streamlit Application

This application provides a visual and interactive interface
to compare different navigation algorithms on a campus-like map.

Algorithms demonstrated:
1. A* Search (classical search-based planning)
2. Q-Learning (reinforcement learning based navigation)

IMPORTANT NOTE ABOUT DATASET:

Although the agent operates in a grid-based environment, the grid
is a DISCRETIZED ABSTRACTION of a real campus navigation dataset
(Global Navigation Dataset - GND). This abstraction allows
reinforcement learning algorithms to be applied efficiently while
preserving real-world spatial constraints.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

# Import project modules
from campus_environment import CampusEnvironment
from qlearning import QLearningAgent
from astar import AStarPlanner


# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="Campus Navigation AI Agent",
    layout="wide"
)

st.title("Intelligent Campus Navigation Agent")
st.markdown("""
This project demonstrates an **intelligent agent** navigating a campus-like
environment using **Artificial Intelligence techniques** studied in CEN 352.

The agent compares:
- **A\*** search (optimal path planning)
- **Q-Learning** (reinforcement learning)
""")


# -------------------------------------------------------
# SIDEBAR - USER CONTROLS
# This section creates the left sidebar where users can adjust parameters
# like grid size and obstacle density for testing different scenarios
# -------------------------------------------------------

st.sidebar.header("Environment Settings")

# Grid size slider: determines the dimensions of the navigation grid (e.g., 15x15)
# Larger grids create more complex navigation problems for the agent to solve
grid_size = st.sidebar.slider(
    "Grid Size",
    min_value=10,
    max_value=25,
    value=15
)

# Obstacle density: controls what percentage of grid cells are blocked
# Higher density = more obstacles = harder navigation problem
obstacle_density = st.sidebar.slider(
    "Obstacle Density",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

# View mode selection: allows comparing different algorithm approaches
# - A* Only: shows classical search algorithm
# - Q-Learning Only: shows reinforcement learning approach
# - Side-by-Side: compares both algorithms on the same environment
view_mode = st.sidebar.radio(
    "Choose View Mode",
    ["A* Only", "Q-Learning Only", "Side-by-Side Comparison"]
)

# Generate button: triggers the navigation algorithm execution
generate_btn = st.sidebar.button("Generate Navigation", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Instructions
1. Adjust grid size and obstacle density
2. Select view mode
3. Click **Generate Navigation**

**Note:** You must train Q-Learning models manually first.

**Training Commands:**
```bash
# Default: 15x15 grid, 0.2 obstacle density
py train_agent.py

# Custom grid size and density
py train_agent.py --grid-size 20 --obstacle-density 0.3

# More episodes for better learning
py train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 15000
```

**Tip:** Train multiple times with more episodes to improve performance!
""")


# -------------------------------------------------------
# HELPER FUNCTIONS FOR MODEL MANAGEMENT
# Functions to generate filenames and check for trained models
# -------------------------------------------------------

def get_model_paths(grid_size, obstacle_density):
    """
    Generate model file paths based on training parameters.
    
    Parameters:
    - grid_size: Size of the grid
    - obstacle_density: Obstacle probability
    
    Returns:
    - tuple: (model_path, env_config_path)
    """
    model_path = f"models/qlearning_agent_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    env_config_path = f"models/environment_config_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    return model_path, env_config_path


# -------------------------------------------------------
# LOAD TRAINED MODEL (IF EXISTS)
# This section handles loading the pre-trained Q-Learning agent from disk.
# Models are trained on-demand for specific grid configurations.
# -------------------------------------------------------

q_agent = None

def load_qlearning_agent(grid_size=15, obstacle_density=0.2):
    """
    Load the trained Q-Learning agent from a saved file.
    
    This function reconstructs the agent by:
    1. Loading the environment configuration (grid layout, start, goal)
    2. Loading the learned Q-table (action-value function)
    3. Recreating the agent object with all saved parameters
    
    Parameters:
    - grid_size: Size of the grid the model was trained on
    - obstacle_density: Obstacle density the model was trained on
    
    Returns:
        QLearningAgent object if successful, None otherwise
    """
    MODEL_PATH, ENV_CONFIG_PATH = get_model_paths(grid_size, obstacle_density)
    
    if os.path.exists(MODEL_PATH) and os.path.exists(ENV_CONFIG_PATH):
        try:
            # Step 1: Load the environment configuration that was saved during training
            # This includes grid dimensions, obstacle positions, start and goal locations
            with open(ENV_CONFIG_PATH, "rb") as f:
                env_config = pickle.load(f)
            
            # Step 2: Recreate the exact same environment the agent was trained on
            # Q-Learning is environment-specific - it learns optimal actions for THIS grid
            env = CampusEnvironment(
                width=env_config["width"],
                height=env_config["height"]
            )
            env.start = env_config["start"]
            env.goal = env_config["goal"]
            env.static_grid = env_config["static_grid"]
            env.grid = env_config["static_grid"].copy()
            
            # Step 3: Load the trained model data containing the Q-table
            # The Q-table maps (state, action) pairs to expected rewards
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            
            # Step 4: Reconstruct the agent with the trained Q-table and learning parameters
            agent = QLearningAgent(
                environment=env,
                learning_rate=model_data['alpha'],      # Learning rate (alpha) used during training
                discount_factor=model_data['gamma'],    # Discount factor (gamma) for future rewards
                epsilon=model_data['epsilon']           # Exploration rate after training
            )
            # Restore the learned Q-table - this is the "brain" of the agent
            agent.q_table = model_data['q_table']
            # Restore training history for analysis and visualization
            agent.episode_rewards = model_data['episode_rewards']
            agent.episode_lengths = model_data['episode_lengths']
            agent.training_losses = model_data['training_losses']
            
            return agent
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


# -------------------------------------------------------
# ENVIRONMENT CREATION
# -------------------------------------------------------

# IMPORTANT NOTE FOR STUDENTS:
# The grid environment used in this project represents a discretized abstraction
# of a real campus navigation dataset (Global Navigation Dataset - GND).
# 
# Why use a grid instead of real coordinates?
# - Grids make the state space finite and manageable for Q-Learning
# - Each grid cell represents a discrete state the agent can be in
# - This simplification allows reinforcement learning algorithms to converge
# - Real-world continuous spaces would require more advanced techniques (deep RL)


# -------------------------------------------------------
# VISUALIZATION FUNCTIONS
# These functions create visual representations of the environment and paths
# to help understand how the algorithms navigate through the grid
# -------------------------------------------------------

def visualize_grid(environment, path=None, title="Campus Navigation", show_stats=True):
    """
    Visualize the grid world environment with color-coded cells and path overlay.
    
    This function creates a matplotlib plot showing:
    - Gray cells: free space where the agent can move
    - Dark cells: obstacles (buildings) that block movement
    - Green circle: starting position
    - Red star: goal position
    - Blue gradient: the path taken by the agent (lighter to darker shows progression)

    Parameters:
    - environment: CampusEnvironment object containing the grid and positions
    - path: Optional list of (x, y) coordinates representing the agent's path
    - title: Title text displayed above the visualization
    - show_stats: Whether to display statistics (currently unused)
    
    Returns:
    - fig: matplotlib Figure object that can be displayed in Streamlit
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a 3D array to store RGB color values for each grid cell
    # Shape: (height, width, 3) where 3 represents RGB color channels
    grid_visual = np.zeros((environment.height, environment.width, 3))
    
    # Color each cell based on its type (obstacle vs. free space)
    for y in range(environment.height):
        for x in range(environment.width):
            if environment.grid[y][x] == 1:  # Cell contains an obstacle
                grid_visual[y, x] = [0.2, 0.2, 0.2]  # Dark gray RGB
            else:  # Cell is free space
                grid_visual[y, x] = [0.95, 0.95, 0.95]  # Light gray RGB
    
    # Overlay the path if one was found by the algorithm
    if path is not None and len(path) > 0:
        for i, (x, y) in enumerate(path):
            if 0 <= x < environment.width and 0 <= y < environment.height:
                # Create a blue gradient effect: cells later in path are darker blue
                # This helps visualize the direction and progression of movement
                intensity = 0.3 + (0.7 * i / len(path))
                grid_visual[y, x] = [0.3, 0.5 + (0.4 * (1 - intensity)), 1.0]
    
    # Display the colored grid as an image
    ax.imshow(grid_visual)
    
    # Mark the start and goal positions with distinct symbols
    sx, sy = environment.start
    gx, gy = environment.goal
    
    # Start position: green circle to indicate where agent begins
    ax.scatter(sx, sy, c="lime", s=300, marker="o", 
               edgecolors="darkgreen", linewidths=2, label="Start", zorder=5)
    # Goal position: red star to indicate target destination
    ax.scatter(gx, gy, c="red", s=300, marker="*", 
               edgecolors="darkred", linewidths=2, label="Goal", zorder=5)
    
    # Add subtle grid lines between cells for better visualization
    # Lines are placed at cell boundaries (offset by 0.5)
    ax.set_xticks(np.arange(-0.5, environment.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, environment.height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Remove numeric labels on axes (not needed for grid visualization)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create a legend to explain what each color represents
    legend_elements = [
        mpatches.Patch(facecolor='lime', edgecolor='darkgreen', label='Start Position'),
        mpatches.Patch(facecolor='red', edgecolor='darkred', label='Goal Position'),
        mpatches.Patch(facecolor=[0.3, 0.7, 1.0], label='Path Taken'),
        mpatches.Patch(facecolor=[0.2, 0.2, 0.2], label='Obstacles'),
        mpatches.Patch(facecolor=[0.95, 0.95, 0.95], label='Free Space')
    ]
    
    # Place legend outside the plot on the right side
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.02, 1), fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def run_astar(env):
    """
    Execute the A* search algorithm on the given environment.
    
    A* is a best-first search algorithm that uses a heuristic function to
    efficiently find the shortest path from start to goal.
    
    Returns: (path, cost, stats) tuple containing the solution
    """
    planner = AStarPlanner(env)
    path, cost, stats = planner.search()
    return path, cost, stats

def run_qlearning(env, agent):
    """
    Execute the trained Q-Learning agent to find a path.
    
    IMPORTANT: Q-Learning is environment-specific!
    The agent can only navigate the exact grid it was trained on because:
    - The Q-table maps specific states (grid positions) to actions
    - Different obstacle layouts = different states = Q-table doesn't apply
    
    This is why we use agent.env instead of the passed env parameter.
    
    Returns: (path, cost, stats) tuple containing the solution
    """
    if agent is None:
        return None, None, None
    
    # Get the path using the agent's learned policy (Q-table)
    # The agent follows the highest Q-value action at each state
    path, total_cost = agent.get_policy_path(agent.env.start)
    cost = len(path) if path else 0
    stats = {"path_length": len(path), "total_cost": total_cost}
    return path, cost, stats

# -------------------------------------------------------
# MAIN APPLICATION LOGIC
# This section controls the flow of the application based on user interactions
# -------------------------------------------------------

# Initialize Streamlit session state to persist data between reruns
# Session state maintains data across user interactions in the web interface
if 'env_astar' not in st.session_state:
    st.session_state.env_astar = None
if 'env_qlearning' not in st.session_state:
    st.session_state.env_qlearning = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'astar_results' not in st.session_state:
    st.session_state.astar_results = None
if 'qlearning_results' not in st.session_state:
    st.session_state.qlearning_results = None

# Handle environment generation when user clicks the button
if generate_btn:
    # CRITICAL CONCEPT FOR STUDENTS:
    # Q-Learning learns a Q-table that is SPECIFIC to one environment layout.
    # Unlike A* which can work on any graph, Q-Learning's learned policy only
    # applies to the exact grid configuration it was trained on.
    
    # For A* mode: always generate a new random environment
    if view_mode == "A* Only":
        st.session_state.env_astar = CampusEnvironment(
            width=grid_size,
            height=grid_size,
            obstacle_prob=obstacle_density
        )
        st.session_state.results_ready = True
    
    # For Q-Learning mode: load existing trained model
    elif view_mode == "Q-Learning Only":
        # Check if model exists for current settings
        MODEL_PATH, ENV_CONFIG_PATH = get_model_paths(grid_size, obstacle_density)
        
        if not os.path.exists(MODEL_PATH):
            # No model exists - show training instructions
            st.error(f"No trained model found for grid={grid_size}x{grid_size}, density={obstacle_density:.2f}")
            st.info("""
            **Please train the model first using:**
            ```bash
            py train_agent.py --grid-size {} --obstacle-density {} --episodes 10000
            ```
            
            **Tip:** Train multiple times with more episodes to improve Q-Learning performance!
            """.format(grid_size, obstacle_density))
            st.session_state.results_ready = False
        else:
            # Model exists - load it
            q_agent_temp = load_qlearning_agent(grid_size, obstacle_density)
            
            if q_agent_temp is not None:
                # Use the exact environment the agent was trained on
                st.session_state.env_qlearning = q_agent_temp.env
                st.success(f"✅ Loaded trained model for grid={grid_size}x{grid_size}, density={obstacle_density:.2f}")
                st.session_state.results_ready = True
            else:
                st.error("ERROR: Failed to load Q-Learning model")
                st.session_state.results_ready = False
    
    # For comparison mode: show different environments to highlight each algorithm's strength
    else:  # Side-by-Side Comparison
        # Check if model exists for current settings
        MODEL_PATH, ENV_CONFIG_PATH = get_model_paths(grid_size, obstacle_density)
        
        if not os.path.exists(MODEL_PATH):
            # No model exists - show training instructions
            st.error(f"No trained model found for grid={grid_size}x{grid_size}, density={obstacle_density:.2f}")
            st.info("""
            **Please train the model first using:**
            ```bash
            py train_agent.py --grid-size {} --obstacle-density {} --episodes 10000
            ```
            
            **For better Q-Learning results, train with more episodes:**
            ```bash
            py train_agent.py --grid-size {} --obstacle-density {} --episodes 20000
            ```
            """.format(grid_size, obstacle_density, grid_size, obstacle_density))
            st.session_state.results_ready = False
        else:
            # Model exists - load it
            q_agent_temp = load_qlearning_agent(grid_size, obstacle_density)
            
            if q_agent_temp is not None:
                # SAME environment for fair comparison:
                # Both algorithms solve the exact same grid to directly compare performance
                st.session_state.env_qlearning = q_agent_temp.env
                st.session_state.env_astar = q_agent_temp.env  # Same environment for fair comparison
                st.success(f"✅ Loaded trained model (grid={grid_size}x{grid_size}, density={obstacle_density:.2f})")
                st.session_state.results_ready = True
            else:
                st.error("ERROR: Failed to load Q-Learning model")
                st.session_state.results_ready = False
    
    # Clear previous results to force fresh computation
    st.session_state.astar_results = None
    st.session_state.qlearning_results = None

# Display results section - executes after environment is generated
if st.session_state.results_ready:
    
    # Mode 1: A* Search Only
    if view_mode == "A* Only":
        if st.session_state.env_astar is not None:
            st.subheader("A* Search Algorithm")
            st.markdown("""
            **A\*** is a classical informed search algorithm that uses a heuristic function
            to efficiently find the shortest path. It guarantees optimality if the heuristic
            is admissible (never overestimates the true cost to reach the goal).
            """)
            
            # Run A* fresh each time generate is clicked
            with st.spinner("Running A* algorithm..."):
                path, cost, stats = run_astar(st.session_state.env_astar)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = visualize_grid(st.session_state.env_astar, path, "A* Path Planning")
                st.pyplot(fig)
            
            with col2:
                st.metric("Path Length", len(path) if path else 0)
                st.metric("Path Cost", cost)
                st.metric("Nodes Expanded", stats["nodes_expanded"])
                
                if path:
                    st.success("SUCCESS: Path found successfully!")
                else:
                    st.error("ERROR: No path found")
    
    # Mode 2: Q-Learning Only
    elif view_mode == "Q-Learning Only":
        if st.session_state.env_qlearning is not None:
            st.subheader("Q-Learning Algorithm")
            st.markdown("""
            **Q-Learning** is a model-free reinforcement learning algorithm that learns
            an optimal action-selection policy. The agent learns by interacting with the
            environment and updating Q-values based on received rewards.
            """)
            
            q_agent = load_qlearning_agent(grid_size, obstacle_density)
            
            if q_agent is None:
                st.warning(f"WARNING: No trained Q-Learning model found for these settings. Training will start when you click Generate.")
            else:
                # Run Q-Learning fresh each time generate is clicked
                with st.spinner("Running Q-Learning agent..."):
                    path, cost, stats = run_qlearning(st.session_state.env_qlearning, q_agent)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Use the agent's environment for visualization
                    fig = visualize_grid(q_agent.env, path, "Q-Learning Navigation")
                    st.pyplot(fig)
                
                with col2:
                    st.metric("Steps Taken", len(path) if path else 0)
                    st.metric("Policy Type", "Learned Q-Table")
                    # Epsilon shows exploration rate (low value = more exploitation)
                    st.metric("Epsilon", f"{q_agent.epsilon:.4f}")
                    
                    if path and len(path) > 0:
                        st.success("SUCCESS: Goal reached!")
                    else:
                        st.error("ERROR: Failed to reach goal")
    
    # Mode 3: Side-by-Side Comparison of both algorithms
    else:
        st.subheader("Algorithm Comparison")
        st.markdown("""
        **Fair Performance Comparison:**
        Both algorithms solve the **same grid** to directly compare their performance.
        - **A\***: Classical search algorithm - always finds optimal path
        - **Q-Learning**: Learned policy - performance depends on training quality
        """)
        
        q_agent = load_qlearning_agent(grid_size, obstacle_density)
        
        if q_agent is None:
            st.warning("WARNING: No trained Q-Learning model found. Showing A* only.")
            # Fallback: show only A* if Q-Learning isn't available
            if st.session_state.env_astar is not None:
                with st.spinner("Running A*..."):
                    astar_path, astar_cost, astar_stats = run_astar(st.session_state.env_astar)
                
                fig1 = visualize_grid(st.session_state.env_astar, astar_path, "A* Path Planning")
                st.pyplot(fig1)
                st.metric("Path Length", len(astar_path) if astar_path else 0)
        else:
            # Both algorithms available - run on SAME environment for fair comparison
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### A* Search")
                st.caption("Classical informed search - guarantees optimal path")
                with st.spinner("Running A*..."):
                    # Run A* on the same environment Q-Learning was trained on
                    astar_path, astar_cost, astar_stats = run_astar(st.session_state.env_astar)
                
                fig1 = visualize_grid(st.session_state.env_astar, astar_path, "A* Path Planning")
                st.pyplot(fig1)
                
                st.metric("Path Length", len(astar_path) if astar_path else 0)
                st.metric("Path Cost", astar_cost)
                st.metric("Nodes Expanded", astar_stats["nodes_expanded"])
                
                if astar_path:
                    st.success("SUCCESS: Optimal path found!")
                else:
                    st.error("ERROR: No path exists")
            
            with col2:
                st.markdown("### Q-Learning")
                st.caption("Reinforcement learning - learned policy")
                with st.spinner("Running Q-Learning..."):
                    q_path, q_cost, q_stats = run_qlearning(st.session_state.env_qlearning, q_agent)
                
                fig2 = visualize_grid(st.session_state.env_qlearning, q_path, "Q-Learning Navigation")
                st.pyplot(fig2)
                
                st.metric("Steps Taken", len(q_path) if q_path else 0)
                st.metric("Policy Type", "Learned Q-Table")
                st.metric("Epsilon", f"{q_agent.epsilon:.4f}")
                
                if q_path and len(q_path) > 0:
                    st.success("SUCCESS: Goal reached!")
                else:
                    st.error("ERROR: Failed to reach goal")
            
            # Performance comparison metrics on same environment
            if astar_path and q_path:
                st.markdown("---")
                st.subheader("Performance Comparison")
                st.caption("Both algorithms solved the same grid - direct performance comparison")
                
                # Display three key metrics comparing the algorithms
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    # Difference in path lengths (negative = Q-Learning is shorter)
                    diff = len(q_path) - len(astar_path)
                    st.metric("Path Length Difference", 
                             f"{diff:+d} steps",
                             delta_color="inverse")
                
                with comp_col2:
                    # Efficiency: how close Q-Learning gets to A*'s optimal path
                    efficiency = (len(astar_path) / len(q_path) * 100) if len(q_path) > 0 else 0
                    st.metric("Q-Learning Efficiency", f"{efficiency:.1f}%")
                
                with comp_col3:
                    # Determine which algorithm found the shorter path
                    winner = "Tie" if len(astar_path) == len(q_path) else ("A*" if len(astar_path) < len(q_path) else "Q-Learning")
                    st.metric("Shorter Path", winner)
                
                # Educational explanation
                st.markdown("""
                **Understanding the Results:**
                - **A\*** always finds the optimal (shortest) path - it's the gold standard
                - **Q-Learning** uses its learned policy - efficiency shows how close it gets to optimal
                - 100% efficiency = Q-Learning matched A*'s optimal solution
                - Both algorithms train/run on the same grid configuration for fair comparison
                """)

else:
    st.info("Configure settings in the sidebar and click Generate Navigation to start!")


# -------------------------------------------------------
# PROJECT INFORMATION
# Educational context and references for the project
# -------------------------------------------------------

st.subheader("Project Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Course:** CEN 352 – Artificial Intelligence  
    **Techniques Used:**  
    - Search Algorithms: A* (informed search with heuristics)  
    - Reinforcement Learning: Q-Learning (model-free value-based learning)  
    """)

with col2:
    st.markdown("""
    **Dataset Reference:**  
    Global Navigation Dataset (GND) – GMU  
    
    **Note:** Grid is a discretized abstraction of real campus maps.
    """)

st.info("""
**Ethical Consideration:**  
Autonomous navigation systems must be designed carefully to avoid
unintended bias or unsafe routing decisions in real-world deployments.
""")