# agent.py
import random
import math
from collections import deque
import heapq
from logic_engine import KnowledgeBase


class GreedyGridAgent:
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos', [0, 0])
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'MoveForward'
        if percept.get('wall_ahead', False):
            return 'TurnLeft'
        return 'MoveForward'


class ModelBasedAgent:
    def __init__(self):
        self.last_percept = None
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            action = 'MoveForward'
        elif percept.get('wall_ahead', False):
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
    def __init__(self, active_algo: str = 'BFS'):
        self.plan = []
        self.active_algo = active_algo
        
        # Step 3.1: Defining the Game Constraints
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def _get_neighbors(self, state: tuple, grid_size: tuple, walls: set):
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
        if start in goals:
            return []

        frontier = deque([(start, [])])
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
        if start in goals:
            return []

        frontier = [(start, [])]
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
        if start in goals:
            return []

        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state in goals:
                return path

            if cost > reached.get(state, float('inf')):
                continue

            for action, neighbor in self._get_neighbors(state, grid_size, walls):
                new_cost = cost + step_cost
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [action]))

        return []

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', tile_percepts=None):
        import itertools
        counter = itertools.count()
        frontier = []
        reached_states = set()
        
        if heuristic_type == 'manhattan':
            h = self.manhattan_distance(start_pos, goal_pos)
        else:
            h = self.euclidean_distance(start_pos, goal_pos)
            
        heapq.heappush(frontier, (0 + h, next(counter), 0, start_pos, []))
        
        while frontier:
            f_cost, _, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            
            if current_pos == goal_pos:
                return path_taken
                
            if current_pos not in reached_states:
                reached_states.add(current_pos)
                
                for action, neighbor in self._get_neighbors(current_pos, grid_size, walls):
                    if neighbor not in reached_states:
                        # Step 3.2: Consult Knowledge Base before adding neighbor to open list
                        self.kb.clear_facts()
                        if tile_percepts and neighbor in tile_percepts:
                            for fact in tile_percepts[neighbor]:
                                self.kb.tell_fact(fact)
                        
                        self.kb.forward_chain()
                        
                        # If 'Retreat' is deduced, mark tile as Infeasible and skip it
                        if 'Retreat' in self.kb.facts:
                            continue

                        g_new = g_cost + 1
                        if heuristic_type == 'manhattan':
                            h_new = self.manhattan_distance(neighbor, goal_pos)
                        else:
                            h_new = self.euclidean_distance(neighbor, goal_pos)
                        f_new = g_new + h_new
                        heapq.heappush(frontier, (f_new, next(counter), g_new, neighbor, path_taken + [action]))
                        
        return []

    def plan_path(self, start: tuple, goals: set, grid_size: tuple, walls: set, tile_percepts: dict = None) -> list:
        algo = self.active_algo.upper()
        if algo == 'BFS':
            return self.bfs_search(start, goals, grid_size, walls)
        elif algo == 'DFS':
            return self.dfs_search(start, goals, grid_size, walls)
        elif algo == 'UCS':
            return self.ucs_search(start, goals, grid_size, walls)
        elif algo == 'ASTAR':
            return self.astar_search(start, list(goals)[0], walls, grid_size, tile_percepts=tile_percepts)
        else:
            raise ValueError(f'Unknown search strategy: {self.active_algo}')

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            all_food = percept.get('all_food', [])
            grid_size = tuple(percept.get('grid_size', (10, 10)))
            walls = set(tuple(w) for w in percept.get('walls', []))
            tile_percepts = percept.get('tile_percepts', None)

            if not all_food:
                return 'Stay'

            closest_food = min(
                all_food,
                key=lambda f: abs(agent_pos[0] - f[0]) + abs(agent_pos[1] - f[1])
            )
            goal_set = {tuple(closest_food)}

            self.plan = self.plan_path(agent_pos, goal_set, grid_size, walls, tile_percepts=tile_percepts)

        if self.plan:
            return self.plan.pop(0)
        return random.choice(['Up', 'Down', 'Left', 'Right'])
