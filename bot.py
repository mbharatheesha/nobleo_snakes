#!/usr/bin/env python3

from random import choice
from typing import List, Tuple
import heapq


import numpy as np

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake


def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def collides(pos: np.array, snakes: List[Snake]) -> bool:
    """
    Check if a position is occupied by any of the snakes
    """
    for snake in snakes:
        if snake.collides(pos):
            return True
    return False


class Snakunamatata(Bot):
    """
    Moves randomly, but makes sure it doesn't collide with other snakes
    """

    @property
    def name(self):
        return 'Snakunamatata'

    @property
    def contributor(self):
        return 'Mukunda'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        moves = self._determine_possible_moves(snake, other_snakes[0], candies)
        return self.choose_move(moves)

    def _determine_possible_moves(self, snake, other_snake, candies) -> List[Move]:
        """
        Return a list with all moves that we want to do. Later we'll choose one from this list randomly. This method
        will be used during unit-testing
        """
        # highest priority, a move that is on the grid and towards the candy
        # Find nearest candy
        nearest_candy = candies[np.argmin(np.sum(np.abs(snake[0]-candies),axis=1))]
        _, path_to_candy, directions_to_candy = self.astar_manhattan(snake[0], nearest_candy)


        if (len(snake) > 2*len(other_snake)):
            for key, value in MOVE_VALUE_TO_DIRECTION.items():
                if np.all(value == np.array([snake[1] - snake[0]])):
                    return [key]


        if (len(path_to_candy) > 1 and not collides(snake[0] + MOVE_VALUE_TO_DIRECTION[directions_to_candy[0]], [snake, other_snake])):
            return [directions_to_candy[0]]


        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return list(Move)

        # then avoid collisions with other snakes
        collision_free = [move for move in on_grid
                          if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)
                          and not collides(snake[0] + MOVE_VALUE_TO_DIRECTION[move], [snake, other_snake])]
        if collision_free:
            return collision_free
        else:
            return on_grid

    def astar_manhattan(self, start, end):
        # Get the dimensions of the grid
        rows, cols = self.grid_size[0], self.grid_size[1]
        # Initialize the distance and visited arrays
        distance = [[float('inf')] * cols for _ in range(rows)]
        distance[start[0]][start[1]] = 0
        visited = [[False] * cols for _ in range(rows)]

        # Priority queue to keep track of the cells to visit
        min_heap = [(0, start)]

        # Dictionary to store the parent of each cell
        came_from = {}

        while min_heap:
            _, current = heapq.heappop(min_heap)
            x, y = current

            # Mark the current cell as visited
            visited[x][y] = True

            # Check if we have reached the end point
            if np.all(current == end):
                # Reconstruct the path from end to start
                path = []
                directions = []
                while np.all(current != start):
                    _, direction = came_from[current]
                    path.insert(0, current)
                    directions.insert(0, direction)
                    current = tuple(np.subtract(current, MOVE_VALUE_TO_DIRECTION[direction]))
                path.insert(0, start)
                return [], path, directions

            # Explore all possible directions
            for move in MOVE_VALUE_TO_DIRECTION:
                nx, ny = tuple(np.add(current, MOVE_VALUE_TO_DIRECTION[move]))

                # Check if the next cell is within the grid boundaries
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny]:
                    # Calculate the new distance using Manhattan distance
                    new_dist = distance[x][y] + (abs(current[0] - nx) + abs(current[1] - ny))
                    # Update the distance if it's shorter
                    if new_dist < distance[nx][ny]:
                        distance[nx][ny] = new_dist
                        # Calculate the heuristic (Manhattan distance to the end)
                        heuristic = abs(nx - end[0]) + abs(ny - end[1])
                        # Use the sum of the distance and heuristic as the priority
                        priority = new_dist + heuristic
                        heapq.heappush(min_heap, (priority, (nx, ny)))
                        # Update the parent of the next cell
                        came_from[(nx, ny)] = (tuple(current), move)
                        # Greedy strategy. Stop once we hit the goal. With Manhattan this is the best anyway.
                        if np.all((nx,ny) == end):
                            current = (nx,ny)
                            path = []
                            directions = []
                            while not(current[0] == start[0] and current[1] == start[1]):
                                parent, direction = came_from[current]
                                path.insert(0, current)
                                directions.insert(0, direction)
                                current = parent
                            path.insert(0, start)
                            return [], path, directions


        # If no path is found
        return -1, [], []


    def choose_move(self, moves: List[Move]) -> Move:
        """
        Randomly pick a move
        """
        return choice(moves)
