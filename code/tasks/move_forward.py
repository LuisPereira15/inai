"""
MoveForwardTask - Reward function for STAGE 1: General Completion.

ALTERAÇÕES v2 (relativamente à versão original do professor):
  - Penalização por morte aumentada de -50 para -500 (torna a sobrevivência crítica)
  - Bónus de win mantido em +1000
  - Stuck threshold reduzido de 10 para 15 frames (menos falsos positivos em saltos)
  - Penalização de stuck reduzida de -0.5 para -0.3 (menos agressiva)
  - Tick penalty reduzida de -0.05 para -0.02 (não penaliza tanto a exploração lenta)
  - Adicionada recompensa por progresso máximo (evita regressão)
  - Kill reward mantido em +10 por inimigo
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

    def _reset_metrics(self):
        """Reset all per-episode statistics."""
        self.metric_distance = 0.0
        self.metric_kills = 0
        self.metric_steps = 0
        self.stuck_counter = 0
        self.max_x_reached = 0.0  # NOVO: progresso máximo alcançado no episódio

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

    def compute_reward(self, current_obs, last_obs):
        """Compute the per-step reward AND update per-episode metrics."""
        self.metric_steps += 1

        if last_obs is None:
            return 0.0

        reward = 0.0

        # 1) Forward progress reward
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            reward += dx  # recompensa direta pelo avanço

            # Atualizar distância máxima
            current_x = current_obs.mario_pos[0]
            if current_x > self.max_x_reached:
                # Bónus extra por atingir novo território (incentiva exploração)
                reward += (current_x - self.max_x_reached) * 0.5
                self.max_x_reached = current_x
                self.metric_distance = current_x

            # Stuck detection: penaliza ficar encravado
            # Threshold aumentado de 10 para 15 (menos falsos positivos em saltos)
            if dx <= 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            if self.stuck_counter > 15:
                reward -= 0.3  # penalização mais suave (era -0.5)

        # 2) Tick penalty reduzida (não penaliza tanto a exploração cuidadosa)
        reward -= 0.02  # era -0.05

        # 3) Kill detection
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed
            reward += killed * 10.0

        # 4) Termination handling
        if current_obs.status == 1:  # WIN
            reward += 1000.0
        elif current_obs.status == 2:  # DEATH
            # Penalização de morte muito mais severa (era -50)
            # Torna a sobrevivência uma prioridade evolutiva clara
            reward -= 500.0

        return reward
