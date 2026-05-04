"""
MLPAgent v2 - Janela alargada 13x13 + rede maior

ALTERAÇÕES v2 (para resolver o problema de wins=0):

  PROBLEMA: janela 7x7 só vê 3 células à frente de Mario
    - A 1.7px/step, o agente vê obstáculos com ~2 frames de antecedência
    - Insuficiente para planear saltos
    - O agente aprende a parar antes dos obstáculos em vez de saltar

  SOLUÇÃO: janela 13x13 centrada 4 células à frente e 1 abaixo de Mario
    - Vê 6 células à frente → ~4x mais tempo para reagir
    - Vê mais do ambiente acima (para saltos com altura)
    - Input: 13*13*2 + 4 = 342 inputs (era 102)

  REDE: aumentada de 102→32→16→5 para 342→64→32→5
    - Mais capacidade para processar o campo visual maior
    - Parâmetros: 3909 → 23941 (6x mais, mas ainda tratável pelo EA)

  MANTIDO:
    - Separação landscape/enemies (funciona bem)
    - Normalização dos inputs
    - Threshold 0.5 para acções binárias
    - Flags: can_jump, on_ground, mario_y, bias
"""

import torch
import torch.nn as nn
import numpy as np
import marioai

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ENEMY_CODES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}

# ALTERAÇÃO v2: janela aumentada de 7 para 13
WINDOW_SIZE = 13
WINDOW_HALF = WINDOW_SIZE // 2   # = 6

MARIO_ROW = 11
MARIO_COL = 11


# ---------------------------------------------------------------------------
# Rede Neural — aumentada para processar campo visual maior
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    342 inputs → 64 → 32 → 5 outputs
    Maior que a v1 (102→32→16→5) para processar janela 13x13
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),   # ALTERADO: era 32
            nn.Tanh(),
            nn.Linear(64, 32),          # ALTERADO: era 16
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
class MLPAgent(marioai.Agent):

    def __init__(self):
        super(MLPAgent, self).__init__()

        # Input: 13*13 landscape + 13*13 enemies + 4 flags = 342
        self.input_dim  = WINDOW_SIZE * WINDOW_SIZE * 2 + 4
        self.output_dim = 5

        self.mlp = MLP(self.input_dim, self.output_dim)
        self.threshold = 0.5

    # -----------------------------------------------------------------------
    # Extração de features — janela 13x13 deslocada à frente de Mario
    # -----------------------------------------------------------------------
    def _extract_window(self, scene):
        """
        Janela 13x13 centrada 4 células à frente e 1 abaixo de Mario.
        Versus v1: 7x7 centrada 2 à frente e 1 abaixo.
        O agente vê agora 6 células à frente (era 3) — tempo de reação 2x maior.
        """
        # ALTERAÇÃO v2: deslocamento maior para ver mais à frente
        center_row = MARIO_ROW + 1   # 1 célula abaixo (igual à v1)
        center_col = MARIO_COL + 4   # ALTERADO: era +2, agora +4 (mais à frente)

        r_start = max(0, center_row - WINDOW_HALF)
        r_end   = min(22, center_row + WINDOW_HALF + 1)
        c_start = max(0, center_col - WINDOW_HALF)
        c_end   = min(22, center_col + WINDOW_HALF + 1)

        window = scene[r_start:r_end, c_start:c_end]

        if window.shape != (WINDOW_SIZE, WINDOW_SIZE):
            padded = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.int32)
            padded[:window.shape[0], :window.shape[1]] = window
            window = padded

        return window

    def _split_landscape_and_enemies(self, window):
        landscape_map = np.zeros(window.shape, dtype=np.float32)
        enemy_map     = np.zeros(window.shape, dtype=np.float32)

        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                v = int(window[i, j])
                if v in ENEMY_CODES:
                    enemy_map[i, j] = 1.0
                elif v == -10:
                    landscape_map[i, j] = -1.0
                elif v == -11:
                    landscape_map[i, j] = -0.5
                elif v in (16, 20, 21):
                    landscape_map[i, j] = 1.0

        return landscape_map, enemy_map

    # -----------------------------------------------------------------------
    # Sense e Act
    # -----------------------------------------------------------------------
    def sense(self, obs):
        super(MLPAgent, self).sense(obs)

    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0]

        window = self._extract_window(self.level_scene)
        land_map, enemy_map = self._split_landscape_and_enemies(window)

        flag_can_jump  = float(self.can_jump)   if self.can_jump   is not None else 0.0
        flag_on_ground = float(self.on_ground)  if self.on_ground  is not None else 0.0
        mario_y        = self.mario_floats[1] / 240.0 if self.mario_floats else 0.0
        bias           = 1.0

        inputs = np.concatenate([
            land_map.flatten(),
            enemy_map.flatten(),
            np.array([flag_can_jump, flag_on_ground, mario_y, bias],
                     dtype=np.float32)
        ]).astype(np.float32)

        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        action_probs = output_tensor.numpy()
        action = (action_probs > self.threshold).astype(int).tolist()
        return action

    # -----------------------------------------------------------------------
    # Genotype helpers
    # -----------------------------------------------------------------------
    def get_param_vector(self):
        params = []
        for param in self.mlp.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_param_vector(self, vector):
        offset = 0
        for param in self.mlp.parameters():
            shape  = param.shape
            size   = int(np.prod(shape))
            chunk  = vector[offset:offset + size].reshape(shape)
            param.data = torch.tensor(chunk, dtype=torch.float32)
            offset += size
