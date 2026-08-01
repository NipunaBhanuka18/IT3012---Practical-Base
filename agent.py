# agent.py
import random

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