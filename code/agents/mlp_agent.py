import torch
import torch.nn as nn
import numpy as np
import marioai


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MLPAgent(marioai.Agent):
    """
    Input features (total = 107):
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Landscape 7x7  (49)  — blocos/obstáculos à volta        │
    │ 2. Enemy grid 7x7 (49)  — inimigos mapeados na mesma grelha│
    │ 3. Hole flags      (4)  — buracos à frente/atrás (1-2 cols)│
    │ 4. Boolean flags   (2)  — can_jump, on_ground              │
    │ 5. Mario mode      (1)  — 0=small,1=large,2=fire (norm.)   │
    │ 6. Velocity        (2)  — vel_x, vel_y (delta de posição)  │
    │                   ────                                      │
    │                   107  total                                │
    └─────────────────────────────────────────────────────────────┘

    Velocity como delta de posição:
      vel_x = mario_x_atual - mario_x_anterior  (normalizado por 16px/tile)
      vel_y = mario_y_atual - mario_y_anterior  (normalizado por 16px/tile)

    Porquê velocidade?
      Sem vel, o MLP recebe input idêntico quando parado encostado a uma
      parede e quando a andar — não consegue distinguir os dois estados
      e nunca aprende a recuar para ganhar balanço.
      Com vel_x < 0, o agente sabe que está a recuar.
      Com vel_y > 0, o agente sabe que está a subir (saltou).
    """

    HALF   = 3
    CENTER = 11

    LANDSCAPE_DIM = (2 * HALF + 1) ** 2   # 49
    ENEMY_DIM     = (2 * HALF + 1) ** 2   # 49
    HOLE_DIM      = 4
    FLAGS_DIM     = 2
    MODE_DIM      = 1
    VEL_DIM       = 2   # novo: vel_x, vel_y

    INPUT_DIM  = LANDSCAPE_DIM + ENEMY_DIM + HOLE_DIM + FLAGS_DIM + MODE_DIM + VEL_DIM  # 107
    OUTPUT_DIM = 5   # [backward, forward, crouch, jump, speed/bombs]

    # Normalização da velocidade: clip a [-VEL_CLIP, VEL_CLIP] tiles/step
    VEL_CLIP = 3.0

    def __init__(self):
        super(MLPAgent, self).__init__()
        self.input_dim   = self.INPUT_DIM
        self.output_dim  = self.OUTPUT_DIM
        self.mlp         = MLP(self.input_dim, self.output_dim)
        self.threshold   = 0.5
        self.stochastic  = True   # True durante treino → amostragem probabilística
                                  # False no eval → determinístico e reproduzível
        self._mario_mode = 0
        self._prev_pos   = None   # posição no frame anterior (para delta)
        self._vel_x      = 0.0   # velocidade calculada por delta
        self._vel_y      = 0.0

    # ------------------------------------------------------------------
    # sense(): actualiza velocidade por delta de posição
    # ------------------------------------------------------------------
    def sense(self, obs):
        super(MLPAgent, self).sense(obs)
        if not self.episode_over:
            self._mario_mode = getattr(obs, 'mario_mode', 0)

            # Delta de posição → velocidade
            current_pos = self.mario_floats  # (x, y) ou None
            if current_pos is not None and self._prev_pos is not None:
                # Normalizar por 16px (tamanho de 1 tile) e clipar
                self._vel_x = float(np.clip(
                    (current_pos[0] - self._prev_pos[0]) / 16.0,
                    -self.VEL_CLIP, self.VEL_CLIP
                ))
                self._vel_y = float(np.clip(
                    (current_pos[1] - self._prev_pos[1]) / 16.0,
                    -self.VEL_CLIP, self.VEL_CLIP
                ))
            else:
                self._vel_x = 0.0
                self._vel_y = 0.0

            self._prev_pos = current_pos

    def new_episode(self):
        """Reset estado interno no início de cada episódio."""
        super(MLPAgent, self).new_episode()
        self._prev_pos = None
        self._vel_x    = 0.0
        self._vel_y    = 0.0

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
        Mapeia inimigos numa grelha 7x7 relativa a Mário.
        Cada célula vale 1.0 se houver inimigo, 0.0 caso contrário.
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

    def _velocity_features(self):
        """
        vel_x e vel_y normalizados para [-1, 1] via clip de VEL_CLIP tiles/step.
        vel_x > 0 → a avançar | vel_x < 0 → a recuar
        vel_y > 0 → a descer  | vel_y < 0 → a subir (convenção y invertido)
        """
        return np.array([
            self._vel_x / self.VEL_CLIP,
            self._vel_y / self.VEL_CLIP
        ], dtype=np.float32)

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
        velocity  = self._velocity_features()    # 2  ← novo

        inputs = np.concatenate([landscape, enemies, holes, flags, mode, velocity])  # 107

        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        action_probs = output_tensor.numpy()

        if self.stochastic:
            # Durante o treino: amostrar de Bernoulli(p)
            # Permite explorar ações com baixa probabilidade (ex: jump=0.08)
            # A evolução reforça as que resultam em progresso
            action = (np.random.rand(len(action_probs)) < action_probs).astype(int).tolist()
        else:
            # No eval: determinístico e reproduzível
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
