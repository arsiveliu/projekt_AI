"""
A* Search Algorithm for Campus Navigation

Implements the A* (A-star) pathfinding algorithm with admissible heuristics.
A* is a best-first search algorithm that finds the optimal path from start to goal
by using both:
1. g(n): actual cost from start to current node
2. h(n): heuristic estimate of cost from current node to goal

f(n) = g(n) + h(n) determines node exploration order
"""
import heapq
from typing import List, Tuple, Optional, Callable
import numpy as np
from campus_environment import CampusEnvironment


class AStarPlanner:
    """
    A* search algorithm for finding optimal paths in the campus environment.
    
    Algorithm Properties:
    - Complete: Yes (guaranteed to find a path if one exists)
    - Optimal: Yes (finds shortest path with admissible heuristic)
    - Time Complexity: O(b^d) where b=branching factor, d=depth
    - Space Complexity: O(b^d) stores all generated nodes in memory
    
    Key Concept:
    A* is optimal because the heuristic function h(n) is admissible
    (never overestimates actual cost to goal). This ensures the first
    path found to the goal is always the shortest path.
    """
    
    def __init__(self, environment: CampusEnvironment, heuristic: str = 'euclidean'):
        """
        Initialize A* planner with environment and heuristic choice.
        
        Args:
            environment: CampusEnvironment object to navigate
            heuristic: Type of heuristic function to use:
                      - 'manhattan': sum of absolute differences (4-dir movement)
                      - 'euclidean': straight-line distance (any movement)
                      - 'diagonal': considers diagonal moves (8-dir movement)
        """
        self.env = environment
        self.heuristic_type = heuristic
        
        # Statistics
        self.nodes_expanded = 0
        self.path_cost = 0.0
        
    def heuristic(self, position: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic value (estimated cost from position to goal).
        
        CRITICAL REQUIREMENT FOR OPTIMALITY:
        The heuristic MUST be admissible (never overestimate actual cost).
        All three heuristic options below are admissible.
        
        Admissibility ensures:
        - If heuristic overestimated, A* might skip the optimal path
        - If heuristic underestimates (or exact), A* finds optimal path
        
        Args:
            position: Current (row, col) position
            goal: Goal (row, col) position
            
        Returns:
            Estimated cost to reach goal (always ≤ actual cost)
        """
        if self.heuristic_type == 'manhattan':
            # Manhattan distance: sum of horizontal and vertical distances
            # Admissible for 4-directional movement (no diagonals)
            # Example: from (0,0) to (3,4) = |3-0| + |4-0| = 7
            return self.env.manhattan_distance(position, goal)
        
        elif self.heuristic_type == 'euclidean':
            # Euclidean distance: straight-line distance
            # Admissible for any type of movement
            # Example: from (0,0) to (3,4) = sqrt(3² + 4²) = 5.0
            return self.env.euclidean_distance(position, goal)
        
        elif self.heuristic_type == 'diagonal':
            # Diagonal distance: optimized for 8-directional movement
            # Accounts for diagonal moves costing sqrt(2) instead of 2
            dx = abs(position[0] - goal[0])
            dy = abs(position[1] - goal[1])
            D = 1.0      # Cost of straight move
            D2 = np.sqrt(2)  # Cost of diagonal move
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        
        else:
            # Default to Euclidean
            return self.env.euclidean_distance(position, goal)
    
    def search(self, start: Optional[Tuple[int, int]] = None,
              goal: Optional[Tuple[int, int]] = None,
              allow_diagonal: bool = False,
              allow_congestion: bool = False) -> Tuple[List[Tuple[int, int]], float, dict]:
        """
        Execute A* search to find the optimal path from start to goal.
        
        Algorithm:
        1. Initialize priority queue with start node (f_score = 0 + h(start))
        2. Pop node with lowest f_score from queue
        3. If node is goal, reconstruct and return path
        4. Otherwise, expand node's neighbors
        5. For each neighbor, calculate g_score and f_score
        6. Add to queue if this is a better path than previously found
        7. Repeat until goal found or queue empty
        
        Args:
            start: Starting position (uses env.start if None)
            goal: Goal position (uses env.goal if None)
            allow_diagonal: If True, agent can move diagonally (8 directions)
            allow_congestion: If True, can move through congested areas (with cost penalty)
            
        Returns:
            Tuple of (path, cost, statistics):
            - path: List of positions from start to goal (empty if no path exists)
            - cost: Total cost of the path
            - statistics: Dict containing nodes_expanded, path_length, success
        """
        start = start or self.env.start
        goal = goal or self.env.goal
        
        # Reset statistics for this search
        self.nodes_expanded = 0
        self.path_cost = 0.0
        
        # Priority queue: (f_score, counter, position, g_score, path)
        # - f_score: total estimated cost (g + h), used for priority
        # - counter: tie-breaker for equal f_scores (FIFO order)
        # - position: current (row, col) position
        # - g_score: actual cost from start to this position
        # - path: list of positions from start to current
        counter = 0
        open_set = [(0.0, counter, start, 0.0, [start])]
        
        # Track visited nodes with their best known g_scores
        # This prevents re-exploring nodes we've already found better paths to
        visited = {}
        visited[start] = 0.0
        
        while open_set:
            # Pop node with lowest f_score (most promising path)
            f_score, _, current_pos, g_score, path = heapq.heappop(open_set)
            
            # Count how many nodes we've explored (for performance analysis)
            self.nodes_expanded += 1
            
            # SUCCESS: Check if we've reached the goal
            if current_pos == goal:
                self.path_cost = g_score
                statistics = {
                    'nodes_expanded': self.nodes_expanded,
                    'path_length': len(path),
                    'path_cost': self.path_cost,
                    'success': True
                }
                return path, self.path_cost, statistics
            
            # Skip this node if we've already found a better path to it
            # This prevents redundant exploration
            if current_pos in visited and visited[current_pos] < g_score:
                continue
            
            # Expand current node: explore all reachable neighbors
            neighbors = self.env.get_neighbors(
                current_pos, 
                allow_diagonal=allow_diagonal,
                allow_congestion=allow_congestion
            )
            
            for neighbor in neighbors:
                # Calculate cost to reach this neighbor
                move_cost = self.env.get_movement_cost(current_pos, neighbor)
                new_g_score = g_score + move_cost
                
                # Skip if we've already found a better or equal path to this neighbor
                if neighbor in visited and visited[neighbor] <= new_g_score:
                    continue
                
                # Calculate f_score = g_score + h_score
                # g_score: actual cost from start to neighbor
                # h_score: estimated cost from neighbor to goal (heuristic)
                h_score = self.heuristic(neighbor, goal)
                new_f_score = new_g_score + h_score
                
                # Add neighbor to priority queue for future exploration
                counter += 1
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f_score, counter, neighbor, new_g_score, new_path))
                
                # Update best known g_score for this neighbor
                visited[neighbor] = new_g_score
        
        # No path found - open_set is empty and goal never reached
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
        print(f"SUCCESS: Path found!")
        print(f"  Path length: {stats['path_length']} steps")
        print(f"  Path cost: {stats['path_cost']:.2f}")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Path: {' -> '.join([str(p) for p in path])}")
    else:
        print("FAIL: No path found (obstacles blocking)")
    
    # Find path allowing congestion
    print("\n--- Searching with congestion allowed ---")
    path, cost, stats = planner.search(allow_diagonal=False, allow_congestion=True)
    
    if stats['success']:
        print(f"SUCCESS: Path found!")
        print(f"  Path length: {stats['path_length']} steps")
        print(f"  Path cost: {stats['path_cost']:.2f}")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
