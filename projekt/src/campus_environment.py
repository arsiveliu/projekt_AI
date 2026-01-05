"""
Campus Navigation Environment
Represents a grid-based campus map with buildings, paths, and dynamic obstacles.
"""
import numpy as np
import random
from typing import Tuple, List, Set

class CampusEnvironment:
    """
    Simulates a campus environment for navigation with dynamic obstacles.
    
    Grid values:
    - 0: Walkable path
    - 1: Building/Static obstacle
    - 2: Dynamic obstacle (congestion/temporary blockage)
    - 3: Start position
    - 4: Goal position
    """
    
    def __init__(self, width: int = 20, height: int = 20, obstacle_prob: float = 0.2):
        """
        Initialize campus environment.
        
        Args:
            width: Width of the campus grid
            height: Height of the campus grid
            obstacle_prob: Probability of static obstacles (buildings)
        """
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        
        # Create base grid
        self.grid = np.zeros((height, width), dtype=int)
        self.static_grid = None  # Store original grid without dynamic obstacles
        
        # Dynamic obstacles
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.congestion_prob = 0.1  # Probability of congestion appearing
        
        # Start and goal positions
        self.start = None
        self.goal = None
        
        # Movement costs
        self.base_cost = 1.0
        self.obstacle_cost = float('inf')
        self.congestion_cost = 3.0  # High cost but not impossible
        
        self._initialize_grid()
    
    def _initialize_grid(self):
        """Initialize the campus grid with buildings and paths."""
        # Create buildings (static obstacles)
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.obstacle_prob:
                    self.grid[i][j] = 1
        
        # Ensure start and goal are not on obstacles
        self.start = self._find_free_position()
        self.goal = self._find_free_position(exclude=[self.start])
        
        # Mark start and goal
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        
        # Save static grid
        self.static_grid = self.grid.copy()
    
    def _find_free_position(self, exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Find a random free position on the grid."""
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
        Add random dynamic obstacles (congestion) to the grid.
        
        Args:
            count: Number of obstacles to add (random if None)
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
        Check if position is walkable.
        
        Args:
            position: Position to check
            allow_congestion: If True, congested areas are considered walkable (with high cost)
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
        Get valid neighboring positions.
        
        Args:
            position: Current position
            allow_diagonal: Allow diagonal movement
            allow_congestion: Allow movement through congested areas
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
        Get the cost of moving from one position to another.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
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
