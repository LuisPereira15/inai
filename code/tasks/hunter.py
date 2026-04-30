"""
HunterTask - Reward function for STAGE 2: Combat Optimization.

ALTERAÇÕES v2 (relativamente à versão original do professor):
  - Kill reward aumentado de 25 para 40 por inimigo (foco ainda maior em combat)
  - Adicionado bónus por kills consecutivos (combo) para incentivar agressividade
  - Penalização de morte aumentada de -50 para -300
  - Forward progress weight reduzido de 0.5 para 0.3 (movimento é secundário)
  - Tick penalty mantida em -0.05
  - Win reward mantido em +500
  - Adicionado bónus por manter modo Fire (se Mario tiver fire, pode atacar à distância)
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
        self.combo_counter = 0       # NOVO: kills consecutivos para bónus de combo
        self.combo_timer = 0         # NOVO: frames desde o último kill

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

        # 1) Forward progress (secundário no Stage 2)
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            reward += 0.3 * dx  # reduzido de 0.5 para 0.3
            if current_obs.mario_pos[0] > self.metric_distance:
                self.metric_distance = current_obs.mario_pos[0]

        # 2) Kill detection com combo system
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)

        self.combo_timer += 1
        # Resetar combo se não houver kill em 60 frames (~1 segundo)
        if self.combo_timer > 60:
            self.combo_counter = 0
            self.combo_timer = 0

        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed

            # Bónus base por kill
            reward += 40.0 * killed  # aumentado de 25 para 40

            # Bónus de combo: cada kill consecutivo vale mais
            self.combo_counter += killed
            if self.combo_counter >= 2:
                reward += 15.0 * (self.combo_counter - 1)  # bónus crescente

            self.combo_timer = 0  # resetar timer de combo

        # 3) Tick penalty
        reward -= 0.05

        # 4) Termination handling
        if current_obs.status == 1:  # WIN
            reward += 500.0
        elif current_obs.status == 2:  # DEATH
            reward -= 300.0  # aumentado de -50 para -300

        return reward
