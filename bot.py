#!/usr/bin/env python3

from random import choice
from typing import List, Tuple
from copy import deepcopy
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
        if (len(snake) > 2*len(other_snake)):
            for key, value in MOVE_VALUE_TO_DIRECTION.items():
                if np.all(value == np.array([snake[1] - snake[0]])):
                    return [key]

        candy_distances = np.sum(np.abs(snake[0] - candies), axis = 1)
        candy_distances_other = np.sum(np.abs(other_snake[0] - candies), axis = 1)
        sorted_ids = np.argsort(candy_distances)
        sorted_candies = []
        for idx in sorted_ids:
            sorted_candies.append(candies[idx])

        nearest_candy = candies[np.argmin(np.sum(np.abs(snake[0]-candies),axis=1))]

        for pos in snake.positions:
            if np.array_equal(nearest_candy, pos):
                nearest_candy = sorted_candies[1]

        dist_other_to_nearest = np.sum(np.abs(other_snake[0] - nearest_candy), axis=0)
        dist_me_to_nearest = np.sum(np.abs(snake[0] - nearest_candy), axis=0)
        if dist_me_to_nearest >= dist_other_to_nearest:
            nearest_candy = candies[np.argmax(np.sum(np.abs(other_snake[0]-candies),axis=1))]

        corners = [np.array([0,0]), np.array([0,15]), np.array([15,0]), np.array([15,15]), np.array([7,7])]

        if len(snake) > 6 and np.all(candy_distances < candy_distances_other):
            _, path_to_candy, directions_to_candy = self.astar_manhattan(snake[0], other_snake[0], snake, other_snake, wiggle_factor=5)
        else:
            _, path_to_candy, directions_to_candy = self.astar_manhattan(snake[0], nearest_candy, snake, other_snake)

        head_snake = snake[0]
        head_previous = snake[1]
        head_other = other_snake[0]
        head_previous_other = other_snake[1]
        if(head_snake[0] == 1 and head_other[0] == 0 and head_previous[0] == 1 and head_previous_other[0] == 0):
            if(head_previous[1] > head_snake[1]):
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.LEFT], [snake, other_snake]):
                    return [Move.LEFT]
            else:
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.LEFT], [snake, other_snake]):
                    return [Move.LEFT]
        elif (head_snake[0]==14 and head_other[0] == 15 and head_previous[0] == 14 and head_previous_other[0] == 15):
            if(head_previous[1] > head_snake[1]):
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.LEFT], [snake, other_snake]):
                    return [Move.LEFT]
            else:
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.RIGHT], [snake, other_snake]):
                    return [Move.RIGHT]
        elif (head_snake[1] == 1 and head_other[1] == 0 and head_previous[1] == 1 and head_previous_other[1] == 0):
            if(head_previous[0] > head_snake[0]):
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.DOWN], [snake, other_snake]):
                    return [Move.DOWN]
            else:
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.DOWN], [snake, other_snake]):
                    return [Move.DOWN]
        elif (head_snake[1] == 14 and head_other[1] == 15 and head_previous[1] == 14 and head_previous_other[1] == 15):
            if(head_previous[0] > head_snake[0]):
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.UP], [snake, other_snake]):
                    return [Move.UP]
            else:
                if not collides(snake[0]+MOVE_VALUE_TO_DIRECTION[Move.UP], [snake, other_snake]):
                    return [Move.UP]
        if len(path_to_candy) > 1:
            next_snake = deepcopy(snake)
            for dir in directions_to_candy:
                next_snake.move(MOVE_VALUE_TO_DIRECTION[dir], grow = True)
            nearest_candy = corners[np.argmax(np.sum(np.abs(next_snake[0]-corners),axis=1))]
            _, free_space, _ = self.astar_manhattan(next_snake[0], nearest_candy, snake, other_snake, wiggle_factor=15)
            if free_space == []:
                new_path = []
                new_directions = []
                for candy in sorted_candies:
                    _, path_to_candy, directions_to_candy = self.astar_manhattan(snake[0], candy, snake, other_snake, wiggle_factor=15)
                    if len(path_to_candy) > len(new_path):
                        new_directions = directions_to_candy
                        new_path = path_to_candy
                path_to_candy = new_path
                directions_to_candy = new_directions
        else:
            new_path = []
            new_directions = []
            for candy in sorted_candies:
                _, path_to_candy, directions_to_candy = self.astar_manhattan(snake[0], candy, snake, other_snake, wiggle_factor=5)
                if len(path_to_candy) > len(new_path):
                    new_directions = directions_to_candy
                    new_path = path_to_candy
            path_to_candy = new_path
            directions_to_candy = new_directions

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


    def astar_manhattan(self, start, end, snake, other_snake, wiggle_factor=np.Inf):
        # Get the dimensions of the grid
        rows, cols = self.grid_size[0], self.grid_size[1]
        # print(f"current goal: {end}")
        # Initialize the distance and visited arrays
        distance = [[float('inf')] * cols for _ in range(rows)]
        distance[start[0]][start[1]] = 0
        visited = [[False] * cols for _ in range(rows)]
        wiggle_yeah = wiggle_factor

        # Priority queue to keep track of the cells to visit
        min_heap = [(0, start)]

        # Dictionary to store the parent of each cell
        came_from = {}

        while min_heap:
            _, current = heapq.heappop(min_heap)
            x, y = current

            # Mark the current cell as visited
            visited[x][y] = True
            for cell in snake.positions:
                visited[cell[0]][cell[1]] = True
            for cell in other_snake.positions:
                visited[cell[0]][cell[1]] = True

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
                if collides(np.array([nx, ny]), [snake,other_snake]):
                    visited[nx][ny] = True

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
                        # if collides(np.array([nx,ny]),[snake, other_snake]):
                        #     priority = 1e6
                        heapq.heappush(min_heap, (priority, (nx, ny)))
                        # Update the parent of the next cell
                        came_from[(nx, ny)] = (tuple(current), move)
                        # Greedy strategy. Stop once we hit the goal. With Manhattan this is the best anyway.
                        if np.all((nx,ny) == end) or len(min_heap) == wiggle_yeah:
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
