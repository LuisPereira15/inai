import torch
import torch.nn as nn
import numpy as np
import marioai


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MLPAgent(marioai.Agent):
    """
    Input features (total = 105):
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Landscape 7x7  (49)  — blocos/obstáculos à volta        │
    │ 2. Enemy grid 7x7 (49)  — inimigos mapeados na mesma grelha│
    │ 3. Hole flags      (4)  — buracos à frente/atrás (1-2 cols)│
    │ 4. Boolean flags   (2)  — can_jump, on_ground              │
    │ 5. Mario mode      (1)  — 0=small,1=large,2=fire (norm.)   │
    │                   ────                                      │
    │                   105  total                                │
    └─────────────────────────────────────────────────────────────┘

    Porquê estas features?
    - Landscape 7x7: visão local de obstáculos (paredes, blocos, etc.)
    - Enemy grid 7x7: mesma grelha mas com 1 onde há inimigo → MLP vê
      simultaneamente obstáculo E inimigo na mesma posição relativa
    - Hole flags: colunas à frente/atrás sem chão → sinal direto de buraco
    - can_jump / on_ground: estado físico de Mário
    - mario_mode: sabe se pode disparar (2) ou se está vulnerável (0)
    """

    HALF   = 2    # raio da janela → 7x7
    CENTER = 11   # posição de Mário na grelha 22x22

    LANDSCAPE_DIM = (2 * HALF + 1) ** 2   # 49
    ENEMY_DIM     = (2 * HALF + 1) ** 2   # 49
    HOLE_DIM      = 4
    FLAGS_DIM     = 2
    MODE_DIM      = 1

    INPUT_DIM  = LANDSCAPE_DIM + ENEMY_DIM + HOLE_DIM + FLAGS_DIM + MODE_DIM  # 105
    OUTPUT_DIM = 5   # [backward, forward, crouch, jump, speed/bombs]

    def __init__(self):
        super(MLPAgent, self).__init__()
        self.input_dim   = self.INPUT_DIM
        self.output_dim  = self.OUTPUT_DIM
        self.mlp         = MLP(self.input_dim, self.output_dim)
        self.threshold   = 0.5
        self._mario_mode = 0

    # ------------------------------------------------------------------
    # sense(): guarda mario_mode que não está no Agent base
    # ------------------------------------------------------------------
    def sense(self, obs):
        super(MLPAgent, self).sense(obs)
        if not self.episode_over:
            self._mario_mode = getattr(obs, 'mario_mode', 0)

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------
    def _landscape_features(self):
        """Janela 7x7 do landscape normalizada para [-1, 1]."""
        c, h = self.CENTER, self.HALF
        window = self.level_scene[c - h: c + h + 1,
                                  c - h: c + h + 1].astype(np.float32)
        return (window / 21.0).flatten()

    def _enemy_grid_features(self):
        """
        Mapeia inimigos (enemies_floats) numa grelha 7x7 relativa a Mário.
        Cada célula vale 1.0 se houver inimigo, 0.0 caso contrário.
        enemies_floats: lista de tuplos (x, y, type) em coordenadas do mundo.
        mario_floats: (mario_x, mario_y) em coordenadas do mundo.
        Cada célula do mundo ≈ 16 píxeis.
        """
        size = 2 * self.HALF + 1
        grid = np.zeros((size, size), dtype=np.float32)

        if self.enemies_floats and self.mario_floats:
            mario_x, mario_y = self.mario_floats
            for enemy in self.enemies_floats:
                ex, ey = enemy[0], enemy[1]
                dx = int(round((ex - mario_x) / 16.0))
                dy = int(round((ey - mario_y) / 16.0))
                row = self.HALF + dy
                col = self.HALF + dx
                if 0 <= row < size and 0 <= col < size:
                    grid[row, col] = 1.0

        return grid.flatten()

    def _hole_features(self):
        """
        Deteta buracos nas 2 colunas à frente e 2 atrás de Mário.
        Buraco = todas as células abaixo de Mário nessa coluna são 0.
        Retorna [atrás2, atrás1, frente1, frente2].
        """
        c = self.CENTER
        holes = []
        for col in [c - 2, c - 1, c + 1, c + 2]:
            if 0 <= col < 22:
                below = self.level_scene[c + 1:, col]
                is_hole = float(np.all(below == 0))
            else:
                is_hole = 0.0
            holes.append(is_hole)
        return np.array(holes, dtype=np.float32)

    def _flags_features(self):
        return np.array([float(self.can_jump), float(self.on_ground)],
                        dtype=np.float32)

    def _mode_features(self):
        return np.array([self._mario_mode / 2.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------
    def act(self):
        if self.level_scene is None:
            return [0, 0, 0, 0, 0]

        landscape = self._landscape_features()   # 49
        enemies   = self._enemy_grid_features()  # 49
        holes     = self._hole_features()        # 4
        flags     = self._flags_features()       # 2
        mode      = self._mode_features()        # 1

        inputs = np.concatenate([landscape, enemies, holes, flags, mode])  # 105

        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        action_probs = output_tensor.numpy()
        action = (action_probs > self.threshold).astype(int).tolist()
        return action

    # ------------------------------------------------------------------
    # Param vector (para o algoritmo evolutivo)
    # ------------------------------------------------------------------
    def get_param_vector(self):
        params = []
        for param in self.mlp.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_param_vector(self, vector):
        offset = 0
        for param in self.mlp.parameters():
            shape = param.shape
            size  = np.prod(shape)
            param.data = torch.tensor(
                vector[offset:offset + size].reshape(shape), dtype=torch.float32
            )
            offset += size