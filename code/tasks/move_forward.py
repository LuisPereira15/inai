"""
MoveForwardTask — Fitness para Stage 1: General Completion.

Versão 3 — penalização de imobilidade:
  Agente que fica parado encostado a obstáculos acumula penalização
  crescente → forçado a tentar escapar (recuar, saltar, etc.).
  Resolve o problema de agentes que ficam presos indefinidamente.

Componentes da reward por step:
  +1.0 * dx              — progresso forward
  +2.0 * Δmax_x          — bónus ao atingir novo máximo X
  -0.05 per step         — tick penalty base (encoraja velocidade)
  -stuck_penalty         — penalização crescente por imobilidade
  +3000 se WIN           — dominante: terminar vale mais que qualquer distância
  -200  se DEATH         — penalização terminal

Stuck penalty:
  Se |vel_x| < STUCK_VEL_THRESHOLD durante STUCK_PATIENCE steps
  consecutivos → penalização de STUCK_BASE * stuck_count por step.
  Reset quando o agente volta a mover-se.
"""

import numpy as np
import marioai


class MoveForwardTask(marioai.Task):

    # Limiar de velocidade para considerar "parado"
    STUCK_VEL_THRESHOLD = 0.5   # tiles/step (após normalização ×16)
    STUCK_PATIENCE      = 20    # steps antes de começar a penalizar
    STUCK_BASE          = 0.5   # penalização por step quando preso
    STUCK_MAX_PENALTY   = 2.0   # cap: penalização máxima por step (não cresce infinitamente)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"
        self._reset_metrics()

    def _reset_metrics(self):
        self.metric_distance = 0.0
        self.metric_kills    = 0
        self.metric_steps    = 0
        self._max_x          = 0.0
        self._prev_x         = None
        self._stuck_count    = 0    # steps consecutivos parado

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
            self._prev_x = (current_obs.mario_pos[0]
                            if current_obs.mario_pos is not None else None)
            return 0.0

        reward = 0.0

        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            curr_x = current_obs.mario_pos[0]
            prev_x = last_obs.mario_pos[0]
            dx     = curr_x - prev_x

            # 1) Progresso forward
            reward += 1.0 * dx

            # 2) Bónus de novo território
            if curr_x > self._max_x:
                reward += 2.0 * (curr_x - self._max_x)
                self._max_x          = curr_x
                self.metric_distance = curr_x

            # 3) Stuck detection — velocidade X em tiles/step (×16 px/tile)
            vel_x_tiles = abs(dx) / 16.0
            if vel_x_tiles < self.STUCK_VEL_THRESHOLD:
                self._stuck_count += 1
            else:
                self._stuck_count = 0   # voltou a mover-se → reset

            # 4) Stuck penalty flat com cap — não cresce quadraticamente
            if self._stuck_count > self.STUCK_PATIENCE:
                reward -= min(self.STUCK_BASE, self.STUCK_MAX_PENALTY)

        # 5) Tick penalty base
        reward -= 0.05

        # 6) Terminação
        if current_obs.status == 1:    # WIN
            reward += 3000.0
        elif current_obs.status == 2:  # DEATH
            reward -= 200.0

        return reward
