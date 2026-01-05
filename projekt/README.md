# 🗺️ Intelligent Campus Navigation System
## CEN 352 Term Project - Dynamic Path Planning with A* Search & Q-Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent agent for dynamic campus navigation that combines **A* heuristic search** for initial path planning with **Q-Learning reinforcement learning** for adaptive decision-making in environments with dynamic obstacles and congestion.

## 📋 Table of Contents
- [Problem Definition](#problem-definition)
- [AI Techniques](#ai-techniques)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [PEAS Framework](#peas-framework)
- [Dataset](#dataset)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [References](#references)
- [Contributors](#contributors)

## 🎯 Problem Definition

This project addresses the challenge of **dynamic campus navigation** where an intelligent agent must:
- Find efficient routes between campus locations
- Adapt to changing conditions (obstacles, congestion, path closures)
- Minimize travel cost while ensuring reliable destination arrival
- Handle partially observable and stochastic environments

Unlike static pathfinding, this system requires **adaptive decision-making** to respond to real-world dynamics rather than following a pre-computed static path.

## 🤖 AI Techniques

### 1. A* Heuristic Search
**Purpose:** Compute initial optimal route based on static campus map.

- **Complete:** Yes (finds solution if one exists)
- **Optimal:** Yes (with admissible heuristic)
- **Time Complexity:** O(b^d) where b is branching factor, d is depth
- **Space Complexity:** O(b^d)

**Implementation:**
- Uses Euclidean/Manhattan/Diagonal distance as admissible heuristic
- Priority queue-based exploration with f(n) = g(n) + h(n)
- Guarantees shortest path on static map

### 2. Q-Learning (Reinforcement Learning)
**Purpose:** Learn adaptive navigation policy through experience in dynamic conditions.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Parameters:**
- α (learning rate): 0.1
- γ (discount factor): 0.95
- ε (exploration rate): 1.0 → 0.01 (decays over training)

**Reward Structure:**
- Reach goal: +100
- Normal move: -1 (encourages efficiency)
- Move through congestion: -5
- Hit obstacle: -50

## ✨ Features

- 🗺️ **Grid-based Campus Environment** with buildings and dynamic obstacles
- 🔍 **A* Search Implementation** with multiple heuristic options
- 🤖 **Q-Learning Agent** with epsilon-greedy exploration
- 📊 **Interactive Streamlit Web App** for visualization
- 📈 **Training Analytics** with performance curves and statistics
- ⚖️ **Algorithm Comparison** side-by-side evaluation
- 🎨 **Beautiful Visualizations** using Plotly
- 💾 **Model Persistence** save and load trained agents

## 🏗️ Project Structure

```
projekt/
├── src/
│   ├── campus_environment.py    # Grid-based campus simulation
│   ├── astar.py                 # A* search algorithm implementation
│   └── qlearning.py             # Q-Learning agent implementation
├── models/                      # Saved trained models
│   ├── qlearning_agent.pkl
│   └── environment_config.pkl
├── data/                        # Training visualizations and logs
│   └── training_curves.png
├── app.py                       # Streamlit web application
├── train_agent.py               # Training script for Q-Learning
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Navigate to project directory:**
   ```bash
   cd projekt
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Training the Q-Learning Agent

Before using the web application, train the reinforcement learning agent:

```bash
python train_agent.py
```

**Expected output:**
- Training progress every 100 episodes
- Final success rate and performance metrics
- Saved model files in `models/` directory
- Training visualization in `data/training_curves.png`

**Training typically takes 5-10 minutes for 2000 episodes.**

### Running the Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Application Features

#### 🎯 Navigation Demo Tab
- Generate random campus maps
- Configure environment (grid size, obstacles, congestion)
- Run A* Search, Q-Learning, or both algorithms
- Compare paths and performance metrics
- Interactive grid visualization

#### 📊 Performance Analysis Tab
- View training history curves
- Analyze reward progression
- Success rate evolution
- Episode length statistics

#### 📚 Documentation Tab
- Algorithm explanations
- Complexity analysis
- PEAS framework description
- Heuristic comparisons

#### ℹ️ About Tab
- Project overview
- Problem definition
- Ethical considerations
- References and citations

## 🔬 Algorithm Details

### A* Search Implementation

**File:** `src/astar.py`

```python
# Core A* search with admissible heuristic
def search(self, start, goal, allow_diagonal=False, allow_congestion=False):
    # f(n) = g(n) + h(n)
    # Uses priority queue for optimal node expansion
    # Returns: path, cost, statistics
```

**Heuristics Available:**
- **Euclidean:** √[(x₂-x₁)² + (y₂-y₁)²] - Best for any movement
- **Manhattan:** |x₂-x₁| + |y₂-y₁| - Optimal for 4-directional
- **Diagonal:** Optimized for 8-directional movement

### Q-Learning Implementation

**File:** `src/qlearning.py`

```python
# Q-Learning update
def update_q_value(self, state, action, reward, next_state):
    current_q = self.q_table[state][action]
    max_next_q = np.max(self.q_table[next_state])
    new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    self.q_table[state][action] = new_q
```

**Action Space:**
- Up, Down, Left, Right (4 actions)
- Can be extended to include diagonal movements

**State Space:**
- Grid position encoded as single integer
- State = row × width + column
- Total states = width × height

## 📐 PEAS Framework

| Component | Description |
|-----------|-------------|
| **Performance Measure** | Minimize travel cost, reach destination reliably, adapt to dynamic changes |
| **Environment** | Grid-based campus with static buildings and dynamic congestion; partially observable, stochastic, sequential |
| **Actuators** | Move up, down, left, right (optionally diagonal) |
| **Sensors** | Current position, adjacent cell information, obstacle detection |

**Agent Type:** Model-based Reflex Agent with Learning

**Environment Properties:**
- ⚠️ Partially Observable (limited sensor range)
- 🎲 Stochastic (dynamic obstacles appear/disappear)
- 📋 Sequential (actions affect future states)
- 🔄 Dynamic (environment changes during execution)
- 📊 Discrete (grid-based state and action spaces)
- 👤 Single-agent

## 📊 Dataset

### Global Navigation Dataset (GND)

**Source:** https://cs.gmu.edu/~xiao/Research/GND

**Description:**
- Detailed campus outdoor spatial data
- Navigational information (maps, traversability categories)
- Multi-modal perception and navigability maps
- Data from multiple university campuses
- Suitable for realistic path planning tasks

**Current Implementation:**
- Simulated grid-based environment
- Random obstacle generation
- Dynamic congestion modeling
- Can be extended to use real GND data

## 📈 Results

### Typical Performance Metrics

**A* Search:**
- Success Rate: ~70-90% (depends on obstacle density)
- Path Optimality: Guaranteed (when path exists)
- Computational Cost: Low for small grids (< 1 second)
- Limitation: Cannot handle dynamic obstacles

**Q-Learning (after 2000 episodes):**
- Success Rate: ~85-95%
- Adaptation: Learns to navigate around congestion
- Training Time: 5-10 minutes
- Inference Speed: Fast (< 0.1 seconds)

### Comparison

| Metric | A* Search | Q-Learning |
|--------|-----------|------------|
| **Optimal Path** | ✅ Yes (static) | ⚠️ Near-optimal |
| **Dynamic Adaptation** | ❌ No | ✅ Yes |
| **Training Required** | ❌ No | ✅ Yes |
| **Completeness** | ✅ Yes | ⚠️ Probabilistic |
| **Computational Cost** | Low | Medium (training), Low (inference) |

**Key Insight:** A* provides optimal initial planning, while Q-Learning enables adaptive behavior in dynamic conditions. The hybrid approach combines strengths of both.

## ⚖️ Ethical Considerations

### Potential Societal Impacts

1. **Accessibility**
   - Routes must accommodate users with disabilities
   - Consider wheelchair accessibility, ramps, elevators
   - Avoid stairs-only paths for mobility-impaired users

2. **Privacy Concerns**
   - Location tracking raises privacy issues
   - Movement data must be anonymized and protected
   - Comply with data protection regulations (GDPR, etc.)

3. **Safety & Security**
   - Routes should avoid hazardous areas
   - Consider emergency evacuation scenarios
   - Real-time updates for dangerous conditions

4. **Fairness & Bias**
   - Algorithm must not discriminate based on user characteristics
   - Equal access to efficient routes for all users
   - Avoid optimizing for specific groups at expense of others

5. **Transparency**
   - Users should understand how routes are determined
   - Explain why certain paths are suggested
   - Allow user override of automated decisions

### Recommendations

- Implement user preferences and constraints
- Regular audits for algorithmic bias
- Privacy-preserving location aggregation
- Emergency override capabilities
- User education about system limitations

## 🔗 References

### Academic Background
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)

### Dataset
- **Global Navigation Dataset (GND)**
  - Source: https://cs.gmu.edu/~xiao/Research/GND
  - GMU Autonomous Robotics Lab

### Implementation References

This project was inspired by the following open-source implementations:

1. **tayfunkscu/QLearning-path-planning**
   - Q-learning path planner with rewards and obstacles
   - GitHub: https://github.com/tayfunkscu/QLearning-path-planning

2. **zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation**
   - Comparison of RL algorithms for navigation
   - GitHub: https://github.com/zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation

3. **Reinforcement-Learning-F22/Dynamic-Routing-for-Navigation-in-Changing-Unknown-Maps**
   - Grid-based RL agent for dynamic routing
   - GitHub: https://github.com/Reinforcement-Learning-F22/Dynamic-Routing-for-Navigation-in-Changing-Unknown-Maps

4. **alirezanobakht13/Maze_with_sarsa_and_Qlearning**
   - Maze solving using Q-learning and SARSA
   - GitHub: https://github.com/alirezanobakht13/Maze_with_sarsa_and_Qlearning

5. **AlinaBaber/Robotic-Path-Tracking-with-Q-Learning-and-SARSA**
   - Robotic path tracking with RL algorithms
   - GitHub: https://github.com/AlinaBaber/Robotic-Path-Tracking-with-Q-Learning-and-SARSA

6. **omron-sinicx/neural-astar**
   - Learned A* search planner
   - GitHub: https://github.com/omron-sinicx/neural-astar

## 👥 Contributors

**Project Team:**
- [Your Name] - [Your Role]
- [Partner Name] - [Partner Role]

**Course:** CEN 352 - Artificial Intelligence  
**Institution:** [Your University]  
**Semester:** Fall 2025 / Spring 2026  
**Instructor:** [Instructor Name]

## 📝 License

This project is for educational purposes as part of CEN 352 coursework.

## 🚀 Future Work

### Planned Enhancements

1. **Real Data Integration**
   - Import actual GND dataset maps
   - Real campus topology and traversability

2. **Advanced RL Techniques**
   - Deep Q-Networks (DQN) for larger state spaces
   - Actor-Critic methods
   - Multi-agent coordination

3. **Enhanced Environment**
   - Time-dependent congestion patterns
   - Weather effects on paths
   - Building schedules (open/closed)

4. **User Features**
   - Personalized route preferences
   - Real-time replanning
   - Mobile application integration

5. **Performance Optimization**
   - Faster training with GPU acceleration
   - Hierarchical pathfinding for large maps
   - Learned heuristics for A*

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure you're in the projekt directory
cd projekt
python train_agent.py
```

**2. Model Not Found**
```
⚠️ No trained model found
# Solution: Run training first
python train_agent.py
```

**3. Streamlit Port Already in Use**
```bash
# Solution: Specify different port
streamlit run app.py --server.port 8502
```

**4. Memory Issues During Training**
```python
# Solution: Reduce grid size or episode count
# In train_agent.py, change:
env = CampusEnvironment(width=10, height=10)  # Smaller grid
agent.train(num_episodes=1000)  # Fewer episodes
```

## 📞 Support

For questions or issues:
1. Check documentation in the app's "Documentation" tab
2. Review code comments in source files
3. Consult referenced GitHub repositories
4. Contact project team members

## 🙏 Acknowledgments

- **CEN 352 Course Staff** for project guidance
- **Open Source Community** for reference implementations
- **GMU Autonomous Robotics Lab** for GND dataset
- **Streamlit Team** for excellent web framework

---

**Last Updated:** January 2026  
**Version:** 1.0.0

**Happy Navigating! 🗺️**
