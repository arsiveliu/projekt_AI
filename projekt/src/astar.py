"""
A* Search Algorithm for Campus Navigation
Implements A* pathfinding with admissible heuristics for optimal route planning.
"""
import heapq
from typing import List, Tuple, Optional, Callable
import numpy as np
from campus_environment import CampusEnvironment


class AStarPlanner:
    """
    A* search algorithm for finding optimal paths in campus environment.
    
    Properties:
    - Complete: Yes (if solution exists)
    - Optimal: Yes (with admissible heuristic)
    - Time Complexity: O(b^d) where b is branching factor, d is depth
    - Space Complexity: O(b^d) - stores all generated nodes
    """
    
    def __init__(self, environment: CampusEnvironment, heuristic: str = 'euclidean'):
        """
        Initialize A* planner.
        
        Args:
            environment: Campus environment to navigate
            heuristic: Heuristic function to use ('manhattan', 'euclidean', 'diagonal')
        """
        self.env = environment
        self.heuristic_type = heuristic
        
        # Statistics
        self.nodes_expanded = 0
        self.path_cost = 0.0
        
    def heuristic(self, position: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic value (estimated cost to goal).
        Must be admissible (never overestimate) for optimal solution.
        
        Args:
            position: Current position
            goal: Goal position
            
        Returns:
            Estimated cost to reach goal
        """
        if self.heuristic_type == 'manhattan':
            # Manhattan distance - admissible for 4-directional movement
            return self.env.manhattan_distance(position, goal)
        
        elif self.heuristic_type == 'euclidean':
            # Euclidean distance - admissible for any movement
            return self.env.euclidean_distance(position, goal)
        
        elif self.heuristic_type == 'diagonal':
            # Diagonal distance - for 8-directional movement
            dx = abs(position[0] - goal[0])
            dy = abs(position[1] - goal[1])
            # Cost of diagonal move = sqrt(2), straight move = 1
            D = 1.0
            D2 = np.sqrt(2)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        
        else:
            # Default to Euclidean
            return self.env.euclidean_distance(position, goal)
    
    def search(self, start: Optional[Tuple[int, int]] = None,
              goal: Optional[Tuple[int, int]] = None,
              allow_diagonal: bool = False,
              allow_congestion: bool = False) -> Tuple[List[Tuple[int, int]], float, dict]:
        """
        Perform A* search to find optimal path from start to goal.
        
        Args:
            start: Starting position (uses env.start if None)
            goal: Goal position (uses env.goal if None)
            allow_diagonal: Allow diagonal movement
            allow_congestion: Allow movement through congested areas (with high cost)
            
        Returns:
            Tuple of (path, cost, statistics)
            - path: List of positions from start to goal (empty if no path found)
            - cost: Total path cost
            - statistics: Dict with search statistics
        """
        start = start or self.env.start
        goal = goal or self.env.goal
        
        # Reset statistics
        self.nodes_expanded = 0
        self.path_cost = 0.0
        
        # Priority queue: (f_score, counter, position, g_score, path)
        # f_score = g_score + h_score
        counter = 0  # Tie-breaker for equal f_scores
        open_set = [(0.0, counter, start, 0.0, [start])]
        
        # Keep track of visited nodes and their best g_scores
        visited = {}
        visited[start] = 0.0
        
        while open_set:
            f_score, _, current_pos, g_score, path = heapq.heappop(open_set)
            
            self.nodes_expanded += 1
            
            # Check if we reached the goal
            if current_pos == goal:
                self.path_cost = g_score
                statistics = {
                    'nodes_expanded': self.nodes_expanded,
                    'path_length': len(path),
                    'path_cost': self.path_cost,
                    'success': True
                }
                return path, self.path_cost, statistics
            
            # Skip if we've found a better path to this position
            if current_pos in visited and visited[current_pos] < g_score:
                continue
            
            # Explore neighbors
            neighbors = self.env.get_neighbors(
                current_pos, 
                allow_diagonal=allow_diagonal,
                allow_congestion=allow_congestion
            )
            
            for neighbor in neighbors:
                # Calculate movement cost
                move_cost = self.env.get_movement_cost(current_pos, neighbor)
                new_g_score = g_score + move_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor in visited and visited[neighbor] <= new_g_score:
                    continue
                
                # Calculate f_score = g_score + h_score
                h_score = self.heuristic(neighbor, goal)
                new_f_score = new_g_score + h_score
                
                # Add to open set
                counter += 1
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f_score, counter, neighbor, new_g_score, new_path))
                
                # Update visited
                visited[neighbor] = new_g_score
        
        # No path found
        statistics = {
            'nodes_expanded': self.nodes_expanded,
            'path_length': 0,
            'path_cost': float('inf'),
            'success': False
        }
        return [], float('inf'), statistics
    
    def get_algorithm_properties(self) -> dict:
        """Get theoretical properties of A* algorithm."""
        return {
            'name': 'A* Search',
            'complete': 'Yes (if solution exists)',
            'optimal': 'Yes (with admissible heuristic)',
            'time_complexity': 'O(b^d)',
            'space_complexity': 'O(b^d)',
            'heuristic': self.heuristic_type,
            'admissible': 'Yes' if self.heuristic_type in ['manhattan', 'euclidean', 'diagonal'] else 'Unknown'
        }


def visualize_path_on_grid(grid: np.ndarray, path: List[Tuple[int, int]]) -> np.ndarray:
    """
    Mark the path on a grid for visualization.
    
    Args:
        grid: Campus grid
        path: List of positions in the path
        
    Returns:
        Grid with path marked (value 5)
    """
    visual_grid = grid.copy()
    for pos in path[1:-1]:  # Exclude start and goal
        if visual_grid[pos[0]][pos[1]] not in [3, 4]:  # Don't overwrite start/goal
            visual_grid[pos[0]][pos[1]] = 5
    return visual_grid


# Example usage
if __name__ == "__main__":
    # Create environment
    env = CampusEnvironment(width=15, height=15, obstacle_prob=0.2)
    env.add_dynamic_obstacles(count=5)
    
    print("Campus Environment:")
    print(env)
    print("\nGrid (0=path, 1=building, 2=congestion):")
    print(env.grid)
    
    # Create A* planner
    planner = AStarPlanner(env, heuristic='euclidean')
    
    print("\nA* Algorithm Properties:")
    for key, value in planner.get_algorithm_properties().items():
        print(f"  {key}: {value}")
    
    # Find path without allowing congestion
    print("\n--- Searching without congestion ---")
    path, cost, stats = planner.search(allow_diagonal=False, allow_congestion=False)
    
    if stats['success']:
        print(f"✓ Path found!")
        print(f"  Path length: {stats['path_length']} steps")
        print(f"  Path cost: {stats['path_cost']:.2f}")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Path: {' -> '.join([str(p) for p in path])}")
    else:
        print("✗ No path found (obstacles blocking)")
    
    # Find path allowing congestion
    print("\n--- Searching with congestion allowed ---")
    path, cost, stats = planner.search(allow_diagonal=False, allow_congestion=True)
    
    if stats['success']:
        print(f"✓ Path found!")
        print(f"  Path length: {stats['path_length']} steps")
        print(f"  Path cost: {stats['path_cost']:.2f}")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
