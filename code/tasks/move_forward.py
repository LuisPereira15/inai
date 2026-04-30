"""
MoveForwardTask - Reward function for STAGE 1: General Completion.
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
        
        # NOVO: Contador para saber há quanto tempo o Mario não avança
        self.stuck_counter = 0 

    def reset(self):
        """Called by the engine at the start of every new episode."""
        super().reset()
        self._reset_metrics()

    def get_metrics(self):
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

            # NOVO: Lógica de penalização por ficar preso nas paredes/obstáculos
            # Se o avanço for quase nulo ou negativo (andar para trás ou bater na parede)
            if dx <= 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0 # Reset ao contador mal ele consiga avançar

            # Se o Mario estiver "encravado" há mais de 10 frames, aplicamos uma penalização
            # Isto vai forçá-lo a explorar o salto para parar de sofrer esta penalização
            if self.stuck_counter > 10:
                reward -= 0.5

        # 2) Tick penalty so being idle is discouraged
        reward -= 0.05

        # 3) Kill detection: compare enemy counts between steps
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed
            # NOVO: Faltava adicionar a recompensa por matar os inimigos!
            reward += killed * 10.0 

        # 4) Termination handling
        if current_obs.status == 1:
            reward += 1000.0
        elif current_obs.status == 2:
            reward -= 50.0

        return reward