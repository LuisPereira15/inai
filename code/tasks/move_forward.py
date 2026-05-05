"""
MoveForwardTask - Reward function for STAGE 1: General Completion.

Filosofia: DENSA EQUILIBRADA com 3 termos puros.

    R(t) = ProgressDelta(t)  +  TickPenalty  +  Terminal(t)

Princípios:
  1. Progresso = MAX_X alcançado (impede reward por ir-e-voltar).
  2. Hierarquia: chegar ao fim >> avançar muito >> avançar e morrer >> parar.
  3. Sem heurísticas prescritivas (sem stuck_counter, sem anti-canguru).
     A evolução descobre POR SI SÓ como saltar canos, evitar inimigos, etc.
  4. Escalas calibradas: terminal domina o denso acumulado.
"""

import numpy as np
import marioai


# Códigos de inimigo na grelha 22x22 do level_scene
ENEMY_CODES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}


def count_enemies(scene):
    """Conta células da grelha que contêm um inimigo (apenas para métricas)."""
    if scene is None:
        return 0
    return sum(int(np.sum(scene == code)) for code in ENEMY_CODES)


# ---------------------------------------------------------------------------
# Constantes da função de recompensa  (calibradas — ver README do relatório)
# ---------------------------------------------------------------------------
PROGRESS_WEIGHT = 1.0    # 1 unidade de mundo = 1 ponto de fitness
TICK_PENALTY    = 0.03   # custo por passo (incentiva eficiência)
WIN_BONUS       = 500.0  # > qualquer fitness alcançável só com progresso
DEATH_PENALTY   = 100.0  # forte mas não esmagadora


class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"
        self._reset_metrics()

    # ----------------------------------------------------------------- metrics
    def _reset_metrics(self):
        self.metric_distance = 0.0   # progresso máximo (= max_x_reached)
        self.metric_kills    = 0     # apenas tracking, NÃO entra no reward
        self.metric_steps    = 0
        self.max_x_reached   = 0.0   # estado interno do progresso

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

    # ------------------------------------------------------------------ reward
    def compute_reward(self, current_obs, last_obs):
        """
        Per-step reward. Soma-se ao longo do episódio em task.cum_reward,
        que é o que o algoritmo evolutivo usa como FITNESS.
        """
        self.metric_steps += 1

        # Primeiro passo do episódio: sem delta possível
        if last_obs is None:
            return 0.0

        reward = 0.0

        # ------------------------------------------------------------------
        # 1) PROGRESSO  — só conta avançar para território NOVO
        # ------------------------------------------------------------------
        # Esta é a versão "potential-based" do progresso:
        #   só ganhamos reward quando max_x aumenta, nunca por dx > 0
        #   sobre território já visitado. Isto remove o incentivo a
        #   ir-e-voltar para acumular reward, e é equivalente a uma
        #   função potencial Φ(s) = max_x_reached(s).
        # ------------------------------------------------------------------
        if current_obs.mario_pos is not None:
            current_x = current_obs.mario_pos[0]
            if current_x > self.max_x_reached:
                reward += PROGRESS_WEIGHT * (current_x - self.max_x_reached)
                self.max_x_reached = current_x
                self.metric_distance = current_x

        # ------------------------------------------------------------------
        # 2) TICK PENALTY — pressiona contra inércia e episódios eternos
        # ------------------------------------------------------------------
        reward -= TICK_PENALTY

        # ------------------------------------------------------------------
        # 3) TERMINAL — domina a fitness, alinhada com o objectivo real
        # ------------------------------------------------------------------
        if current_obs.status == 1:        # WIN — chegou à bandeira
            reward += WIN_BONUS
        elif current_obs.status == 2:      # DEATH
            reward -= DEATH_PENALTY

        # ------------------------------------------------------------------
        # 4) MÉTRICAS (kills) — apenas tracking, NÃO afecta reward na Stage 1
        # ------------------------------------------------------------------
        enemies_now    = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            self.metric_kills += (enemies_before - enemies_now)

        return reward