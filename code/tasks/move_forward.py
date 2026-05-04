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
        self.stuck_counter = 0
        self.max_x_reached = 0.0  # Progresso máximo alcançado no episódio

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

        # 1) Leitura de Movimento (Eixo X e Eixo Y)
        if current_obs.mario_pos is not None and last_obs.mario_pos is not None:
            dx = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            # No MarioAI, o Y diminui quando se salta para cima:
            dy = last_obs.mario_pos[1] - current_obs.mario_pos[1] 
            
            # Recompensa base por avançar (O Foco Absoluto da Stage 1)
            reward += dx * 1.5 

            # Bónus extra de Desbravador (Atingir novo território)
            current_x = current_obs.mario_pos[0]
            if current_x > self.max_x_reached:
                reward += (current_x - self.max_x_reached) * 1.5
                self.max_x_reached = current_x
                self.metric_distance = current_x

            # -------------------------------------------------------------
            # A "TAXA DE SALTO" (Evita o Mário "Canguru")
            # -------------------------------------------------------------
            # Se está a correr rápido e livre (dx > 0.5) e salta sem razão:
            if dy > 0 and dx > 0.5:
                reward -= 0.2

            # -------------------------------------------------------------
            # MECANISMO AGRESSIVO DE FUGA DE OBSTÁCULOS
            # -------------------------------------------------------------
            # Memorizamos se ele estava encravado antes de atualizar o contador
            estava_encravado = self.stuck_counter > 10

            if dx <= 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0 # Reset ao contador mal ele consiga avançar

            # CASO 1: Ele continua encravado a bater na parede
            if self.stuck_counter > 10:
                reward -= 2.0  # Dor forte para o obrigar a mudar de estratégia

                if dy > 0:  # Se ele tentar saltar
                    if dx > 0:
                        # Bónus GIGANTE: Saltou e está a raspar/avançar no cano
                        reward += 20.0  
                    else:
                        # Bónus NORMAL: Saltou de forma puramente vertical (dx = 0)
                        reward += 10.0  
                    
                    self.stuck_counter -= 5 # Dá-lhe tempo no ar para passar o cano

            # CASO 2: O MOMENTO DE GLÓRIA!
            # Ele estava encravado, deu um grande salto e conseguiu avançar livremente!
            elif estava_encravado and dy > 0 and dx > 0.1:
                reward += 20.0  # Recompensa máxima por se libertar de vez!

        # 2) Tick penalty (O relógio não perdoa inércia, força a velocidade)
        reward -= 0.02

        # 3) Kill detection (Apenas para métricas, SEM recompensa na Stage 1)
        enemies_now = count_enemies(current_obs.level_scene)
        enemies_before = count_enemies(last_obs.level_scene)
        if enemies_before > enemies_now:
            killed = enemies_before - enemies_now
            self.metric_kills += killed

        # 4) Termination handling (Os Pilares do Projeto)
        if current_obs.status == 1:  # WIN
            reward += 1000.0
        elif current_obs.status == 2:  # DEATH
            reward -= 500.0

        return reward