"""Run the original random-obstacle evaluator with extra spawn/goal capacity in Z."""

import numpy as np

import evaluate
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


original_generator = Scenario_o_base.generate_pos_obst_map_2


def capacity_in_height(self, num_agents):
    if num_agents <= len(self.free_space):
        return original_generator(self, num_agents)

    width = self.obstacle_map.shape[0]
    free_cells = np.asarray(self.free_space)
    xy = np.asarray(self.cell_centers)[free_cells[:, 0] + width * free_cells[:, 1]]
    levels = np.tile(np.array([1.15, 2.0, 2.85]), (len(xy), 1))
    # Each free cell is first covered once. Further assignments use unused
    # levels in distinct cells, with randomized levels within each column.
    orders = np.argsort(np.random.random(levels.shape), axis=1)
    heights = np.take_along_axis(levels, orders, axis=1)
    points = np.concatenate((np.broadcast_to(xy[:, None], (*heights.shape, 2)),
                             heights[..., None]), axis=-1).transpose(1, 0, 2)
    if num_agents > points.shape[0] * points.shape[1]:
        raise ValueError(f"Random map has too few free cells for {num_agents} agents")
    points = np.concatenate([np.random.permutation(layer) for layer in points])[:num_agents]
    points[:, 2] += np.random.uniform(-.08, .08, size=num_agents)
    points = np.random.permutation(points)
    return points


Scenario_o_base.generate_pos_obst_map_2 = capacity_in_height

if __name__ == "__main__":
    evaluate.main()
