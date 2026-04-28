"""
MoveForwardTask - Reward function for STAGE 1: General Completion.

The level status (win=1 / death=2 / timeout=other) is NOT available during
the steps - it is only known at the end of the episode. So we set the
metric flags inside compute_reward() but the OFFICIAL final value of
won/died should be read from `task.status` AFTER doEpisodes() returns.

Metrics tracked:
  - distance      : how far Mario walked (max x reached)
  - kills         : enemies eliminated during the episode
  - won_level     : True if Mario reached the flag (status == 1) at the end
  - died          : True if Mario died (status == 2) at the end
  - steps         : number of game ticks the episode lasted
"""

import numpy as np
import marioai


# Enemy codes inside the level_scene grid
ENEMY_CODES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}


def count_enemies(scene):
    """Count how many cells of the 22x22 grid contain an enemy code."""
    if scene is None:
        return 0
    count = 0
    for code in ENEMY_CODES:
        count += int(np.sum(scene == code))
    return count


class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"
        self._reset_metrics()

    # -----------------------------------------------------------------------
    # Helpers to reset and read metrics
    # -----------------------------------------------------------------------
    def _reset_metrics(self):
        """Reset all per-episode statistics."""
        self.metric_distance = 0.0
        self.metric_kills = 0
        self.metric_steps = 0

    def reset(self):
        """Called by the engine at the start of every new episode."""
        super().reset()
        self._reset_metrics()

    def get_metrics(self):
        """
        Return a dictionary with all the metrics of the last episode.
        IMPORTANT: won/died are read from self.status, which is set by the
        engine ONLY after the episode ends. So call get_metrics() AFTER
        doEpisodes() returns.
        """
        return {
            "distance": self.metric_distance,
            "kills": self.metric_kills,
            "won": (self.status == 1),
            "died": (self.status == 2),
            "steps": self.metric_steps,
            "final_status": self.status,
        }

    # -----------------------------------------------------------------------
    # Reward function
    # -----------------------------------------------------------------------
    def compute_reward(self, current_obs, last_obs):
        """Compute the per-step reward AND update per-episode metrics."""
        self.metric_steps += 1

        if last_obs is None:
            return 0.0

        reward = 0.0

        # 1) Forward progress reward + distance metric
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            reward += dx
            if current_obs.mario_pos[0] > self.metric_distance:
                self.metric_distance = current_obs.mario_pos[0]

        # 2) Tick penalty so being idle is discouraged
        reward -= 0.05

        # 3) Kill detection: compare enemy counts between steps
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed

        # 4) Termination handling - status here is mostly 0 during the run,
        #    but we still keep these checks for safety. The OFFICIAL win/death
        #    is read from task.status after the episode ends (see get_metrics).
        if current_obs.status == 1:
            reward += 1000.0
        elif current_obs.status == 2:
            reward -= 50.0

        return reward