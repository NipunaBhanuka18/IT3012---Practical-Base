# agent.py
import random
from collections import deque
import heapq

class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept.get('agent_pos', [0, 0])
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """A simple reflex agent that reacts only to the current percept."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'MoveForward'
        if percept.get('wall_ahead', False):
            return 'TurnLeft'
        return 'MoveForward'


class ModelBasedAgent:
    """A model-based agent that remembers prior percepts and avoids repeating the same failed action."""

    def __init__(self):
        self.last_percept = None
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            action = 'MoveForward'
        elif percept.get('wall_ahead', False):
            # If we hit a wall and our last percept was also a wall, turning left didn't help, so turn right.
            if self.last_percept is not None and frozenset(self.last_percept.items()) == frozenset(percept.items()):
                action = 'TurnRight'
            else:
                action = 'TurnLeft'
        else:
            action = 'MoveForward'

        self.last_percept = percept
        self.last_action = action
        return action


class SearchAgent:
    """An agent that uses search algorithms (BFS, DFS, UCS) to navigate the grid environment."""

    def __init__(self, active_algo: str = 'BFS'):
        self.plan = []  # List of actions to execute
        self.active_algo = active_algo  # 'BFS', 'DFS', or 'UCS'

    def _get_neighbors(self, state: tuple, grid_size: tuple, walls: set):
        """Generates valid next states and the corresponding action."""
        x, y = state
        width, height = grid_size
        moves = [
            ('Up', (x, y + 1)),
            ('Down', (x, y - 1)),
            ('Left', (x - 1, y)),
            ('Right', (x + 1, y))
        ]
        neighbors = []
        for action, (nx, ny) in moves:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                neighbors.append((action, (nx, ny)))
        return neighbors

    def bfs_search(self, start: tuple, goals: set, grid_size: tuple, walls: set) -> list:
        """Breadth-First Search using a FIFO queue (deque.popleft())."""
        if start in goals:
            return []

        frontier = deque([(start, [])])  # (state, path_of_actions)
        reached = {start}

        while frontier:
            state, path = frontier.popleft()

            if state in goals:
                return path

            for action, neighbor in self._get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))

        return []

    def dfs_search(self, start: tuple, goals: set, grid_size: tuple, walls: set) -> list:
        """Depth-First Search using a LIFO stack (list.pop())."""
        if start in goals:
            return []

        frontier = [(start, [])]  # (state, path_of_actions)
        reached = {start}

        while frontier:
            state, path = frontier.pop()

            if state in goals:
                return path

            for action, neighbor in self._get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))

        return []

    def ucs_search(self, start: tuple, goals: set, grid_size: tuple, walls: set, step_cost: int = 1) -> list:
        """Uniform-Cost Search using a Priority Queue (heapq.heappop()) ordered by path cost g(n)."""
        if start in goals:
            return []

        # (cost, counter, state, path_of_actions)
        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state in goals:
                return path

            # If we found a higher cost path than already reached, skip
            if cost > reached.get(state, float('inf')):
                continue

            for action, neighbor in self._get_neighbors(state, grid_size, walls):
                new_cost = cost + step_cost
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [action]))

        return []

    def plan_path(self, start: tuple, goals: set, grid_size: tuple, walls: set) -> list:
        """Plans a path to the goal food using the configured search strategy."""
        algo = self.active_algo.upper()
        if algo == 'BFS':
            return self.bfs_search(start, goals, grid_size, walls)
        elif algo == 'DFS':
            return self.dfs_search(start, goals, grid_size, walls)
        elif algo == 'UCS':
            return self.ucs_search(start, goals, grid_size, walls)
        else:
            raise ValueError(f"Unknown search strategy: {self.active_algo}")

    def sense_and_act(self, percept: dict) -> str:
        # Check if plan is empty
        if not self.plan:
            # Extract environment details from percept
            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            all_food = percept.get('all_food', [])
            grid_size = tuple(percept.get('grid_size', (10, 10)))
            walls = set(tuple(w) for w in percept.get('walls', []))

            if not all_food:
                return 'Stay'

            # Find the closest food pellet using Manhattan distance
            closest_food = min(
                all_food,
                key=lambda f: abs(agent_pos[0] - f[0]) + abs(agent_pos[1] - f[1])
            )
            goal_set = {tuple(closest_food)}

            # Execute search method matching active_algo and store in self.plan
            self.plan = self.plan_path(agent_pos, goal_set, grid_size, walls)

        # Return the first action from the plan
        if self.plan:
            return self.plan.pop(0)
        return random.choice(['Up', 'Down', 'Left', 'Right'])