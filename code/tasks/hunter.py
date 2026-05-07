"""
HunterTask - Reward function for STAGE 2: Combat Optimization.
Foco: Eliminação de inimigos, multi-kills, recolha de Power-Ups 
e navegação tática de obstáculos com balanço consistente.
"""

import numpy as np
import marioai

# Categorização dos Inimigos por Risco
TIER_1_ENEMIES = {2}           # Goomba
TIER_2_ENEMIES = {3, 4, 5, 6}  # Koopas
TIER_3_ENEMIES = {7, 8, 9, 10, 12, 13} # Spikys, Piranhas

def count_enemies_by_tier(scene):
    if scene is None:
        return {1: 0, 2: 0, 3: 0}
    
    counts = {1: 0, 2: 0, 3: 0}
    for code in TIER_1_ENEMIES: counts[1] += int(np.sum(scene == code))
    for code in TIER_2_ENEMIES: counts[2] += int(np.sum(scene == code))
    for code in TIER_3_ENEMIES: counts[3] += int(np.sum(scene == code))
    return counts

class HunterTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Hunter"
        self._reset_metrics()

    def _reset_metrics(self):
        self.metric_distance = 0.0
        self.metric_kills = 0
        self.metric_steps = 0
        self.stuck_counter = 0
        self.max_x_reached = 0.0
        self.momentum_authorized = False

    def reset(self):
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
        self.metric_steps += 1

        
        if last_obs is None or current_obs.mario_pos is None or last_obs.mario_pos is None:
            return 0.0

        reward = 0.0

        # -------------------------------------------------------------
        # 1. MOVIMENTO (Lógica de Compromisso de Balanço)
        # -------------------------------------------------------------
        dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
        current_x = current_obs.mario_pos[0]
        
        # Recompensa de Desbravador: O prémio por bater o recorde de distância no nível
        if current_x > self.max_x_reached:
            reward += (current_x - self.max_x_reached) * 5.0 # Peso aumentado para encorajar progresso
            self.max_x_reached = current_x
            self.metric_distance = current_x

        # SISTEMA ANTI-INDECISÃO
        if abs(dx) <= 0.1:
            self.stuck_counter += 1
        else:
            if self.momentum_authorized and dx < -0.1:
                reward += abs(dx) * 2.5 # Torna o recuo "lucrativo" temporariamente
            
            # Se ele já está a correr para a frente com velocidade após o balanço
            if dx > 0.5:
                self.momentum_authorized = False # Objetivo de balanço cumprido
            
            self.stuck_counter = 0

        # Ativação do Balanço: Se falhou o salto durante 15 frames, tem de recuar
        if self.stuck_counter > 15:
            self.momentum_authorized = True
            reward += 10.0 # Bónus por "perceber" a estratégia

        # Recompensa normal de avanço (só se não estiver em "modo recuo")
        if not self.momentum_authorized:
            reward += 1.3 * dx
            if dx < -0.1: # Penalização por recuar sem ter tentado saltar primeiro
                reward -= 5.0 

        # Penalização por inércia total (ficar a beijar o cano sem se mexer)
        if self.stuck_counter > 40:
            reward -= 2.0

        # -------------------------------------------------------------
        # 2. INSTINTO PREDADOR (Kills por Tier e Multi-Kills)
        # -------------------------------------------------------------
        enemies_now = count_enemies_by_tier(current_obs.level_scene)
        enemies_before = count_enemies_by_tier(last_obs.level_scene)
        
        total_killed = 0
        # Recompensas atualizadas conforme sugerido para o relatório [cite: 1]
        pts_map = {1: 300.0, 2: 700.0, 3: 2000.0} 

        for tier in [1, 2, 3]:
            if enemies_before[tier] > enemies_now[tier]:
                killed = enemies_before[tier] - enemies_now[tier]
                total_killed += killed
                reward += pts_map[tier] * killed

        if total_killed > 0:
            self.metric_kills += total_killed
            # BÓNUS DE MULTI-KILL: Bónus massivo por matar vários de uma vez (ex: carapaça)
            if total_killed > 1:
                reward += 60000.0 * (total_killed - 1)

        # -------------------------------------------------------------
        # 3. GESTÃO DE POWER-UPS
        # -------------------------------------------------------------
        mode_now = getattr(current_obs, 'mario_mode', 0)
        mode_before = getattr(last_obs, 'mario_mode', 0)
        
        if mode_now > mode_before:
            reward += 15000.0  
        elif mode_now < mode_before:
            reward -= 500.0

        # -------------------------------------------------------------
        # 4. PENALIZAÇÕES E VITÓRIA
        # -------------------------------------------------------------
        reward -= 0.05 # Tick penalty

        if current_obs.status == 1:  # WIN
            reward += 15000.0  
        elif current_obs.status == 2:  # DEATH
            reward -= 500.0  

        return reward