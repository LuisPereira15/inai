"""
MoveForwardTask - Reward function for STAGE 1: General Completion.

ALTERAÇÕES v4 (baseadas na análise dos resultados v3):

  PROBLEMA 1 — dx incremental incentivava oscilação:
    - Removida recompensa de dx por step
    - Substituída por recompensa APENAS de progresso máximo novo (+2.0 × novo_território)
    - O agente só ganha pontos se chegar a um ponto nunca antes atingido

  PROBLEMA 2 — Timeout de 750 steps impossibilitava win:
    - Timeout controlado em evaluation.py (MAX_STEPS → 2000)
    - Tick penalty ajustada para -0.01 (horizonte mais longo)

  PROBLEMA 3 — Stuck penalty evitava saltos:
    - Removida a penalização de stuck completamente
    - Substituída por penalização de REGRESSÃO (-0.5 se andar para trás)
    - Recompensa vertical: subir também é progresso (+0.5 × novo_y)

  PROBLEMA 4 — Win bonus insuficiente:
    - Win bonus aumentado de +1000 para +5000
    - Deve ser claramente o objetivo mais valioso

  MANTIDO:
    - Kill reward em +10 por inimigo (funciona bem)
    - Death penalty em -500
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


class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"
        self._reset_metrics()

    def _reset_metrics(self):
        self.metric_distance = 0.0
        self.metric_kills    = 0
        self.metric_steps    = 0
        self.max_x_reached   = 0.0
        self.max_y_reached   = 0.0

    def reset(self):
        super().reset()
        self._reset_metrics()

    def get_metrics(self):
        return {
            "distance":     self.metric_distance,
            "kills":        self.metric_kills,
            "won":          (self.status == 1),
            "died":         (self.status == 2),
            "steps":        self.metric_steps,
            "final_status": self.status,
        }

    def compute_reward(self, current_obs, last_obs):
        self.metric_steps += 1

        if last_obs is None:
            return 0.0

        reward = 0.0

        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            current_x = current_obs.mario_pos[0]
            current_y = current_obs.mario_pos[1]
            last_x    = last_obs.mario_pos[0]

            # 1) Recompensa APENAS por novo território horizontal
            if current_x > self.max_x_reached:
                reward += (current_x - self.max_x_reached) * 2.0
                self.max_x_reached = current_x
                self.metric_distance = current_x

            # 2) Penalização por regressão (andar para trás)
            elif current_x < last_x - 1.0:
                reward -= 0.5

            # 3) Recompensa por progresso vertical (saltos sobre obstáculos)
            if current_y > self.max_y_reached:
                reward += (current_y - self.max_y_reached) * 0.5
                self.max_y_reached = current_y

        # 4) Tick penalty pequena (horizonte de 2000 steps)
        reward -= 0.01

        # 5) Kill detection
        enemies_now    = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed
            reward += killed * 10.0

        # 6) Termination
        if current_obs.status == 1:    # WIN
            reward += 5000.0
        elif current_obs.status == 2:  # DEATH
            reward -= 500.0

        return reward
