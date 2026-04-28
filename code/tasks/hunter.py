"""
HunterTask - Reward function for STAGE 2: Combat Optimization.

Same metric-tracking design as MoveForwardTask: won/died are read from
self.status AFTER doEpisodes() returns.
"""

import numpy as np
import marioai


ENEMY_CODES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}


def count_enemies(scene):
    if scene is None:
        return 0
    count = 0
    for code in ENEMY_CODES:
        count += int(np.sum(scene == code))
    return count


class HunterTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Hunter"
        self._reset_metrics()

    def _reset_metrics(self):
        self.metric_distance = 0.0
        self.metric_kills = 0
        self.metric_steps = 0

    def reset(self):
        super().reset()
        self._reset_metrics()

    def get_metrics(self):
        """Read won/died from task.status (only valid after the episode ends)."""
        return {
            "distance": self.metric_distance,
            "kills": self.metric_kills,
            "won": (self.status == 1),
            "died": (self.status == 2),
            "steps": self.metric_steps,
            "final_status": self.status,
        }

    def compute_reward(self, current_obs, last_obs):
        self.metric_steps += 1

        if last_obs is None:
            return 0.0

        reward = 0.0

        # 1) Smaller forward-progress reward (still need movement)
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            reward += 0.5 * dx
            if current_obs.mario_pos[0] > self.metric_distance:
                self.metric_distance = current_obs.mario_pos[0]

        # 2) Big reward for kills (the focus of Stage 2)
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            reward += 25.0 * killed
            self.metric_kills += killed

        # 3) Tick penalty
        reward -= 0.05

        # 4) Termination handling
        if current_obs.status == 1:
            reward += 500.0
        elif current_obs.status == 2:
            reward -= 50.0

        return reward