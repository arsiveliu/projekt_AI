"""
Streamlit Web Application for Campus Navigation
Interactive visualization of A* Search and Q-Learning agents.
"""
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from campus_environment import CampusEnvironment
from astar import AStarPlanner
from qlearning import QLearningAgent

# Page configuration
st.set_page_config(
    page_title="Campus Navigation AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .algorithm-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_agent():
    """Load trained Q-Learning agent and environment."""
    try:
        # Load agent
        env_temp = CampusEnvironment(width=15, height=15)
        agent = QLearningAgent(environment=env_temp)
        agent.load_model('models/qlearning_agent.pkl')
        
        # Load environment config
        with open('models/environment_config.pkl', 'rb') as f:
            env_config = pickle.load(f)
        
        # Reconstruct environment
        env = CampusEnvironment(
            width=env_config['width'],
            height=env_config['height']
        )
        env.start = env_config['start']
        env.goal = env_config['goal']
        env.static_grid = env_config['static_grid']
        env.grid = env_config['static_grid'].copy()
        
        agent.env = env
        
        return agent, env, True
    except FileNotFoundError:
        return None, None, False


def create_grid_visualization(grid: np.ndarray, path: List[Tuple[int, int]] = None,
                              title: str = "Campus Map") -> go.Figure:
    """Create interactive heatmap visualization of campus grid."""
    
    # Create display grid
    display_grid = grid.copy().astype(float)
    
    # Mark path if provided
    if path:
        for i, pos in enumerate(path):
            if display_grid[pos[0]][pos[1]] not in [3, 4]:  # Don't overwrite start/goal
                display_grid[pos[0]][pos[1]] = 5 + (i / len(path)) * 0.5
    
    # Create color scale
    colorscale = [
        [0.0, '#90EE90'],   # Walkable - light green
        [0.2, '#FFB6C1'],   # Building - light red
        [0.4, '#FFD700'],   # Congestion - gold
        [0.6, '#00FF00'],   # Start - bright green
        [0.8, '#FF0000'],   # Goal - bright red
        [1.0, '#1E90FF']    # Path - blue
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>'
    ))
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', dtick=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', dtick=1)
    
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        height=600,
        yaxis=dict(autorange='reversed'),  # Flip y-axis
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_comparison_chart(astar_stats: dict, ql_stats: dict) -> go.Figure:
    """Create comparison chart for A* and Q-Learning."""
    
    metrics = ['Success', 'Path Length', 'Cost']
    astar_values = [
        100 if astar_stats['success'] else 0,
        astar_stats.get('path_length', 0),
        astar_stats.get('path_cost', 0) if astar_stats.get('path_cost', float('inf')) != float('inf') else 0
    ]
    ql_values = [
        100 if ql_stats['success'] else 0,
        ql_stats.get('path_length', 0),
        ql_stats.get('cost', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='A* Search', x=metrics, y=astar_values, marker_color='#1f77b4'),
        go.Bar(name='Q-Learning', x=metrics, y=ql_values, marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        title='Algorithm Comparison',
        barmode='group',
        height=400,
        yaxis_title='Value',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🗺️ Intelligent Campus Navigation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dynamic Path Planning using A* Search & Q-Learning</p>', unsafe_allow_html=True)
    
    # Load trained agent
    agent, trained_env, model_loaded = load_trained_agent()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Environment settings
    st.sidebar.subheader("Environment Settings")
    grid_size = st.sidebar.slider("Grid Size", min_value=10, max_value=25, value=15, step=1)
    obstacle_prob = st.sidebar.slider("Building Density", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    num_congestion = st.sidebar.slider("Dynamic Obstacles", min_value=0, max_value=15, value=5, step=1)
    
    # Algorithm settings
    st.sidebar.subheader("Algorithm Settings")
    heuristic = st.sidebar.selectbox("A* Heuristic", ["euclidean", "manhattan", "diagonal"])
    allow_diagonal = st.sidebar.checkbox("Allow Diagonal Movement", value=False)
    
    # Model status
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Status")
    if model_loaded:
        st.sidebar.success("✅ Q-Learning Model Loaded")
        st.sidebar.info(f"Episodes Trained: {len(agent.episode_rewards)}")
        st.sidebar.metric("Final ε", f"{agent.epsilon:.4f}")
    else:
        st.sidebar.warning("⚠️ No trained model found")
        st.sidebar.info("Run 'python train_agent.py' first")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Navigation Demo", "📊 Performance Analysis", "📚 Documentation", "ℹ️ About"])
    
    with tab1:
        st.header("Interactive Navigation Demo")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Campus Environment")
            
            # Create or use loaded environment
            if st.button("🔄 Generate New Map", type="primary"):
                env = CampusEnvironment(width=grid_size, height=grid_size, obstacle_prob=obstacle_prob)
                st.session_state['env'] = env
            
            if 'env' not in st.session_state:
                if trained_env:
                    env = trained_env
                else:
                    env = CampusEnvironment(width=grid_size, height=grid_size, obstacle_prob=obstacle_prob)
                st.session_state['env'] = env
            else:
                env = st.session_state['env']
            
            # Add dynamic obstacles
            env.reset()
            env.add_dynamic_obstacles(count=num_congestion)
            
            # Display legend
            st.markdown("""
            **Legend:**
            - 🟢 Green: Walkable paths
            - 🔴 Red: Buildings (static obstacles)
            - 🟡 Yellow: Congestion (dynamic obstacles)
            - 🎯 Start and Goal positions
            """)
            
            # Display environment info
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Grid Size", f"{env.width}x{env.height}")
            col_b.metric("Start", f"{env.start}")
            col_c.metric("Goal", f"{env.goal}")
        
        with col2:
            st.subheader("Algorithm Selection")
            algorithm = st.radio(
                "Choose Algorithm:",
                ["A* Search", "Q-Learning", "Both (Compare)"],
                help="Select which pathfinding algorithm to use"
            )
        
        # Execute navigation
        if st.button("🚀 Find Path", use_container_width=True):
            with st.spinner("Computing paths..."):
                
                # A* Search
                astar_path = []
                astar_stats = {}
                if algorithm in ["A* Search", "Both (Compare)"]:
                    planner = AStarPlanner(env, heuristic=heuristic)
                    astar_path, astar_cost, astar_stats = planner.search(
                        allow_diagonal=allow_diagonal,
                        allow_congestion=False
                    )
                
                # Q-Learning
                ql_path = []
                ql_stats = {}
                if algorithm in ["Q-Learning", "Both (Compare)"] and model_loaded:
                    agent.env = env
                    ql_path, ql_cost = agent.get_policy_path(max_steps=200)
                    ql_stats = {
                        'success': ql_path[-1] == env.goal if ql_path else False,
                        'path_length': len(ql_path),
                        'cost': ql_cost
                    }
                
                # Display results
                st.markdown("---")
                st.subheader("Results")
                
                if algorithm == "Both (Compare)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### A* Search")
                        if astar_stats.get('success'):
                            st.success("✅ Path Found")
                            fig = create_grid_visualization(env.grid, astar_path, "A* Search Path")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.metric("Path Length", f"{astar_stats['path_length']} steps")
                            st.metric("Path Cost", f"{astar_stats['path_cost']:.2f}")
                            st.metric("Nodes Expanded", astar_stats['nodes_expanded'])
                        else:
                            st.error("❌ No path found (obstacles blocking)")
                    
                    with col2:
                        st.markdown("### Q-Learning")
                        if ql_stats.get('success'):
                            st.success("✅ Path Found")
                            fig = create_grid_visualization(env.grid, ql_path, "Q-Learning Path")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.metric("Path Length", f"{ql_stats['path_length']} steps")
                            st.metric("Path Cost", f"{ql_stats['cost']:.2f}")
                        else:
                            st.error("❌ No path found")
                    
                    # Comparison chart
                    st.markdown("---")
                    if astar_stats.get('success') and ql_stats.get('success'):
                        fig = create_comparison_chart(astar_stats, ql_stats)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif algorithm == "A* Search":
                    if astar_stats.get('success'):
                        st.success("✅ Path Found!")
                        fig = create_grid_visualization(env.grid, astar_path, "A* Search Path")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Path Length", f"{astar_stats['path_length']} steps")
                        col2.metric("Path Cost", f"{astar_stats['path_cost']:.2f}")
                        col3.metric("Nodes Expanded", astar_stats['nodes_expanded'])
                    else:
                        st.error("❌ No path found (obstacles blocking all routes)")
                
                elif algorithm == "Q-Learning":
                    if model_loaded:
                        if ql_stats.get('success'):
                            st.success("✅ Path Found!")
                            fig = create_grid_visualization(env.grid, ql_path, "Q-Learning Path")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Path Length", f"{ql_stats['path_length']} steps")
                            col2.metric("Path Cost", f"{ql_stats['cost']:.2f}")
                        else:
                            st.error("❌ Goal not reached within step limit")
                    else:
                        st.warning("⚠️ Please train the Q-Learning model first")
    
    with tab2:
        st.header("Performance Analysis")
        
        if model_loaded:
            st.subheader("Training History")
            
            # Training curves
            if os.path.exists('data/training_curves.png'):
                st.image('data/training_curves.png', caption='Training Performance Over Time')
            else:
                st.info("Training curves not available. Run 'python train_agent.py' to generate.")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Episodes", len(agent.episode_rewards))
            col2.metric("Avg Reward (Last 100)", f"{np.mean(agent.episode_rewards[-100:]):.2f}")
            col3.metric("Avg Steps (Last 100)", f"{np.mean(agent.episode_lengths[-100:]):.1f}")
            
            success_rate = sum(1 for x in agent.episode_lengths[-100:] if x < 200) / min(100, len(agent.episode_lengths)) * 100
            col4.metric("Success Rate (Last 100)", f"{success_rate:.1f}%")
            
            # Detailed statistics
            st.subheader("Detailed Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Best Episode Reward', 'Worst Episode Reward', 
                          'Median Episode Length', 'Shortest Path Found'],
                'Value': [
                    f"{max(agent.episode_rewards):.2f}",
                    f"{min(agent.episode_rewards):.2f}",
                    f"{np.median(agent.episode_lengths):.1f} steps",
                    f"{min(agent.episode_lengths)} steps"
                ]
            })
            st.table(stats_df)
        else:
            st.warning("⚠️ No trained model available. Run training script first.")
    
    with tab3:
        st.header("Algorithm Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 A* Search Algorithm
            
            **Purpose:** Compute optimal initial route using static map data.
            
            **Properties:**
            - **Complete:** Yes (finds solution if one exists)
            - **Optimal:** Yes (with admissible heuristic)
            - **Time Complexity:** O(b^d)
            - **Space Complexity:** O(b^d)
            
            **How it works:**
            1. Uses priority queue with f(n) = g(n) + h(n)
            2. g(n) = actual cost from start
            3. h(n) = heuristic estimate to goal
            4. Expands nodes with lowest f(n) first
            
            **Heuristics:**
            - **Euclidean:** Straight-line distance
            - **Manhattan:** Grid distance (4-directional)
            - **Diagonal:** Optimal for 8-directional movement
            
            All heuristics are admissible (never overestimate).
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 Q-Learning Algorithm
            
            **Purpose:** Learn adaptive navigation through experience in dynamic conditions.
            
            **Update Rule:**
            ```
            Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            ```
            
            **Parameters:**
            - **α (alpha):** Learning rate (0.1)
            - **γ (gamma):** Discount factor (0.95)
            - **ε (epsilon):** Exploration rate (decays over time)
            
            **How it works:**
            1. Agent explores environment using ε-greedy policy
            2. Receives rewards for actions taken
            3. Updates Q-values based on received rewards
            4. Gradually learns optimal policy over many episodes
            
            **Rewards:**
            - Reach goal: +100
            - Move through congestion: -5
            - Hit obstacle: -50
            - Normal move: -1 (encourages shorter paths)
            """)
        
        st.markdown("---")
        st.subheader("PEAS Framework")
        
        peas_df = pd.DataFrame({
            'Component': ['Performance Measure', 'Environment', 'Actuators', 'Sensors'],
            'Description': [
                'Minimize travel cost, reach destination reliably',
                'Grid-based campus with static buildings and dynamic congestion',
                'Move up, down, left, right (optionally diagonal)',
                'Current position, adjacent cells, obstacle detection'
            ]
        })
        st.table(peas_df)
    
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎓 CEN 352 Term Project
        ### Intelligent Agent for Dynamic Campus Navigation
        
        #### 📋 Problem Definition
        This project focuses on designing an intelligent agent for dynamic campus navigation. 
        The agent must find efficient routes between locations on a campus map that may change 
        over time due to obstacles or congestion, requiring adaptive decision-making rather than 
        just following a static map plan.
        
        #### 🤖 AI Techniques
        
        **1. Heuristic Search (A*)**
        - Computes initial optimal route based on static map data
        - Guides agent efficiently toward goal with admissible heuristics
        - Guarantees optimal path when solution exists
        
        **2. Reinforcement Learning (Q-Learning)**
        - Adapts agent's decisions during navigation when environment changes
        - Over many trials, learns to prefer actions that reduce travel cost
        - Handles dynamic conditions like congested paths
        
        #### 📊 Dataset
        **Global Navigation Dataset (GND)**
        - Contains detailed campus outdoor spatial data
        - Navigational information (maps, traversability, environmental structure)
        - Multi-modal perception from multiple university campuses
        - Source: https://cs.gmu.edu/~xiao/Research/GND
        
        #### 🔗 References
        This implementation was inspired by:
        - tayfunkscu/QLearning-path-planning (GitHub)
        - zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation (GitHub)
        - Reinforcement-Learning-F22/Dynamic-Routing-for-Navigation (GitHub)
        - alirezanobakht13/Maze_with_sarsa_and_Qlearning (GitHub)
        - AlinaBaber/Robotic-Path-Tracking-with-Q-Learning-and-SARSA (GitHub)
        - omron-sinicx/neural-astar (GitHub)
        
        #### 🛠️ Technologies
        - **Python 3.8+**
        - **Streamlit** - Web interface
        - **NumPy** - Numerical computations
        - **Plotly** - Interactive visualizations
        - **Matplotlib** - Training plots
        
        #### ⚠️ Ethical Considerations
        **Potential Societal Impact:**
        
        Campus navigation systems must consider:
        - **Accessibility:** Ensure routes accommodate all users including those with disabilities
        - **Privacy:** Location tracking and movement data must be protected
        - **Safety:** Routes should avoid hazardous areas and consider emergency scenarios
        - **Fairness:** Algorithm should not discriminate based on user characteristics
        
        #### 📝 Future Improvements
        - Integration with real campus map data (GND dataset)
        - Multi-agent coordination for crowded scenarios
        - Deep Q-Networks (DQN) for larger state spaces
        - Real-time obstacle detection and path replanning
        - Mobile app integration for actual campus use
        
        ---
        
        **Project Type:** Model-based Reflex Agent with Learning
        
        **Environment Properties:**
        - Partially Observable
        - Stochastic (dynamic obstacles)
        - Sequential
        - Static during A* planning, Dynamic during execution
        - Discrete
        - Single-agent
        """)


if __name__ == "__main__":
    main()
