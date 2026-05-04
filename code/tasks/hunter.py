"""
HunterTask - Reward function for STAGE 2: Combat Optimization.

O Foco Absoluto: Eliminação de inimigos, multi-kills, e recolha de Power-Ups.
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

class HunterTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Hunter"
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset all per-episode statistics."""
        self.metric_distance = 0.0
        self.metric_kills = 0
        self.metric_steps = 0
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

    def compute_reward(self, current_obs, last_obs):
        """Compute the per-step reward AND update per-episode metrics."""
        self.metric_steps += 1

        if last_obs is None:
            return 0.0

        reward = 0.0

        # -------------------------------------------------------------
        # 1. MOVIMENTO (Baixa Prioridade - Mantém o Mário a avançar devagar)
        # -------------------------------------------------------------
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            dy = last_obs.mario_pos[1] - current_obs.mario_pos[1]
            
            # Recompensa muito menor (0.5). O objetivo já não é o speedrun!
            reward += 0.5 * dx 
            if current_obs.mario_pos[0] > self.metric_distance:
                self.metric_distance = current_obs.mario_pos[0]

            # Fuga de bloqueios básica (herdada e simplificada da Stage 1)
            if dx <= 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            if self.stuck_counter > 20:
                reward -= 1.0  # Pressão para não ficar acampado
                if dy > 0 and dx > 0:
                    reward += 5.0
                    self.stuck_counter -= 10

        # -------------------------------------------------------------
        # 2. INSTINTO PREDADOR (O Coração da Stage 2)
        # -------------------------------------------------------------
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed
            
            # Recompensa Base por Matar
            base_kill_reward = 75.0 * killed
            
            # MULTI-KILL BONUS: Se matar > 1 de uma vez (Carapaça ou Super Estrela)
            # Ganha um extra substancial para incentivar combos!
            multi_kill_bonus = 50.0 * (killed - 1) if killed > 1 else 0.0
            
            reward += base_kill_reward + multi_kill_bonus

        # -------------------------------------------------------------
        # 3. GESTÃO DE ARMAMENTO (Power-ups: O segredo para caçar)
        # -------------------------------------------------------------
        # mario_mode: 0 = Small, 1 = Big, 2 = Fire
        mode_now = getattr(current_obs, 'mario_mode', 0)
        mode_before = getattr(last_obs, 'mario_mode', 0)
        
        if mode_now > mode_before:
            # Recompensa massiva por evoluir para Big ou Fire!
            # Ensina-o a bater nos blocos ? e a apanhar os itens.
            reward += 100.0  
        elif mode_now < mode_before:
            # PUNIÇÃO AGRESSIVA por levar dano! 
            # Ensina-o a usar a Flor de Fogo à distância em vez de se atirar para o perigo.
            reward -= 30.0

        # -------------------------------------------------------------
        # 4. Tick Penalty & Término
        # -------------------------------------------------------------
        # Penalização por frame continua a existir para forçar a exploração
        reward -= 0.05

        if current_obs.status == 1:  # WIN
            reward += 500.0  # Chegar ao fim ainda é o prémio máximo
        elif current_obs.status == 2:  # DEATH
            reward -= 100.0  # A morte é má, mas menos penalizadora que na Stage 1 para o encorajar a correr riscos de combate

        return reward