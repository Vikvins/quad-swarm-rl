"""Fixed individual starts/goals for deterministic evaluation layouts."""

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_layout(QuadrotorScenario):
    def reset(self, layout_plan=None, **kwargs):
        if layout_plan is None:
            raise ValueError('This scenario requires an explicit layout plan')
        self.spawn_points = layout_plan.starts.copy()
        self.goals = layout_plan.goals.copy()
        self.formation_size = 1.
        self.approch_goal_metric = .3

    def step(self):
        pass


class Scenario_o_corridor(Scenario_o_layout):
    pass


class Scenario_o_dead_end(Scenario_o_layout):
    pass
