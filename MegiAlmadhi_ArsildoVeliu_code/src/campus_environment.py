"""
Campus Navigation Environment

Represents a discretized grid-based campus map with buildings, walkable paths,
and dynamic obstacles (like crowds or temporary blockages).

This environment serves as the state space for both A* search and Q-Learning algorithms.
The grid abstraction makes the problem tractable for reinforcement learning while
still representing realistic spatial navigation constraints.
"""
import numpy as np
import random
from typing import Tuple, List, Set

class CampusEnvironment:
    """
    Simulates a campus environment for navigation with dynamic obstacles.
    
    The environment is represented as a 2D grid where each cell can be:
    - 0: Walkable path (free space)
    - 1: Building/Static obstacle (permanent blockage)
    - 2: Dynamic obstacle (temporary congestion or crowd)
    - 3: Start position (where agent begins)
    - 4: Goal position (where agent wants to reach)
    
    This discrete representation allows both:
    1. A* search (which needs a graph structure)
    2. Q-Learning (which needs a finite state space)
    """
    
    def __init__(self, width: int = 20, height: int = 20, obstacle_prob: float = 0.2):
        """
        Initialize a campus environment with random obstacle placement.
        
        Args:
            width: Width of the grid (number of columns)
            height: Height of the grid (number of rows)
            obstacle_prob: Probability (0-1) that any given cell is a static obstacle
                          - 0.2 means approximately 20% of cells will be buildings
                          - Higher values create more challenging navigation problems
        """
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        
        # Create base grid: 2D array of integers representing cell types
        self.grid = np.zeros((height, width), dtype=int)
        self.static_grid = None  # Stores original grid (without dynamic obstacles)
        
        # Set to track positions of dynamic obstacles (crowds, temporary blocks)
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.congestion_prob = 0.1  # Probability of congestion appearing
        
        # Start and goal positions (set during grid initialization)
        self.start = None
        self.goal = None
        
        # Movement costs for different cell types
        self.base_cost = 1.0              # Cost of moving to free space
        self.obstacle_cost = float('inf') # Cannot move through static obstacles
        self.congestion_cost = 3.0        # High cost but passable (agent should avoid)
        
        self._initialize_grid()
    
    def _initialize_grid(self):
        """
        Initialize the campus grid with random building placement.
        
        Process:
        1. Randomly place static obstacles (buildings) based on obstacle_prob
        2. Find valid free positions for start and goal
        3. Ensure start and goal are not blocked
        4. Save a copy of the static grid (used for resetting)
        """
        # Randomly place buildings (static obstacles) throughout the grid
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.obstacle_prob:
                    self.grid[i][j] = 1  # Mark as static obstacle
        
        # Find free positions for start and goal (cannot be on obstacles)
        self.start = self._find_free_position()
        self.goal = self._find_free_position(exclude=[self.start])
        
        # Ensure start and goal cells are marked as free space
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        
        # Save a copy of the static grid (before adding dynamic obstacles)
        # This allows resetting the environment without recreating obstacles
        self.static_grid = self.grid.copy()
    
    def _find_free_position(self, exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Find a random free position on the grid.
        
        Searches for a grid cell that is walkable (value = 0) and not in
        the exclusion list. Continues until a valid position is found.
        
        Args:
            exclude: List of positions to avoid (e.g., don't place goal on start)
            
        Returns:
            A valid (row, col) position
        """
        exclude = exclude or []
        while True:
            pos = (random.randint(0, self.height - 1), 
                   random.randint(0, self.width - 1))
            if self.grid[pos[0]][pos[1]] == 0 and pos not in exclude:
                return pos
    
    def reset(self):
        """Reset environment to initial state."""
        self.grid = self.static_grid.copy()
        self.dynamic_obstacles.clear()
        return self.start
    
    def add_dynamic_obstacles(self, count: int = None):
        """
        Add random dynamic obstacles (congestion/crowds) to the grid.
        
        Dynamic obstacles represent temporary blockages like:
        - Crowds of people
        - Temporary construction
        - Moving obstacles
        
        These are different from static obstacles (buildings) because:
        - They can appear and disappear
        - They have a penalty but can be traversed
        - Agent learns to avoid them when possible
        
        Args:
            count: Number of obstacles to add (randomly determined if None)
        """
        if count is None:
            count = random.randint(1, max(1, int(self.width * self.height * 0.05)))
        
        self.dynamic_obstacles.clear()
        for _ in range(count):
            pos = self._find_free_position(exclude=[self.start, self.goal])
            self.dynamic_obstacles.add(pos)
            self.grid[pos[0]][pos[1]] = 2
    
    def remove_dynamic_obstacle(self, position: Tuple[int, int]):
        """Remove a dynamic obstacle from the grid."""
        if position in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(position)
            if self.static_grid[position[0]][position[1]] == 0:
                self.grid[position[0]][position[1]] = 0
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_walkable(self, position: Tuple[int, int], allow_congestion: bool = False) -> bool:
        """
        Check if a position is walkable (agent can move there).
        
        A position is walkable if:
        1. It's within grid boundaries
        2. It's not a static obstacle (building)
        3. Either allow_congestion=True OR it's not a dynamic obstacle
        
        Args:
            position: (row, col) position to check
            allow_congestion: If True, congested areas count as walkable
                            (agent can pass through but with penalty)
                            If False, congested areas are treated as blocked
        
        Returns:
            True if position is walkable, False otherwise
        """
        if not self.is_valid_position(position):
            return False
        
        row, col = position
        cell_value = self.grid[row][col]
        
        if cell_value == 1:  # Static obstacle
            return False
        if cell_value == 2 and not allow_congestion:  # Dynamic obstacle
            return False
        
        return True
    
    def get_neighbors(self, position: Tuple[int, int], 
                     allow_diagonal: bool = False,
                     allow_congestion: bool = False) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions from current position.
        
        Returns positions the agent can move to from the current position.
        Used by both A* (for graph expansion) and Q-Learning (for valid actions).
        
        Args:
            position: Current (row, col) position
            allow_diagonal: If True, includes diagonal moves (8 neighbors)
                          If False, only cardinal directions (4 neighbors)
            allow_congestion: If True, includes congested positions as valid
        
        Returns:
            List of valid neighboring positions
        """
        row, col = position
        neighbors = []
        
        # Four main directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Add diagonal directions if allowed
        if allow_diagonal:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self.is_walkable(new_pos, allow_congestion=allow_congestion):
                neighbors.append(new_pos)
        
        return neighbors
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate the cost of moving from one position to another.
        
        Cost structure:
        - Static obstacle (building): infinite (cannot move)
        - Dynamic obstacle (congestion): 3.0 (high penalty but passable)
        - Free space: 1.0 * distance (diagonal moves cost sqrt(2) ≈ 1.41)
        
        Used by A* algorithm to calculate path costs.
        
        Args:
            from_pos: Starting (row, col) position
            to_pos: Target (row, col) position
        
        Returns:
            Movement cost (float), or infinity if move is invalid
        """
        if not self.is_valid_position(to_pos):
            return self.obstacle_cost
        
        cell_value = self.grid[to_pos[0]][to_pos[1]]
        
        if cell_value == 1:  # Static obstacle
            return self.obstacle_cost
        elif cell_value == 2:  # Dynamic obstacle (congestion)
            return self.congestion_cost
        else:
            # Calculate Euclidean distance for diagonal moves
            distance = np.sqrt((from_pos[0] - to_pos[0])**2 + 
                             (from_pos[1] - to_pos[1])**2)
            return self.base_cost * distance
    
    def get_state_representation(self, position: Tuple[int, int]) -> int:
        """Convert 2D position to 1D state index."""
        return position[0] * self.width + position[1]
    
    def get_position_from_state(self, state: int) -> Tuple[int, int]:
        """Convert 1D state index to 2D position."""
        row = state // self.width
        col = state % self.width
        return (row, col)
    
    def get_num_states(self) -> int:
        """Get total number of states."""
        return self.width * self.height
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_grid_for_display(self) -> np.ndarray:
        """Get grid with start and goal marked for display."""
        display_grid = self.grid.copy()
        display_grid[self.start[0]][self.start[1]] = 3  # Start
        display_grid[self.goal[0]][self.goal[1]] = 4    # Goal
        return display_grid
    
    def __repr__(self):
        """String representation of the environment."""
        return f"CampusEnvironment({self.width}x{self.height}, " \
               f"Start: {self.start}, Goal: {self.goal}, " \
               f"Dynamic Obstacles: {len(self.dynamic_obstacles)})"
