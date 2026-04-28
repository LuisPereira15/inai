"""
MLPAgent - Neuroevolution controller for Super Mario.

Architecture (improvements from project guide, Appendix C):
  1. Curse of dimensionality (Challenge 3):
       Instead of feeding the whole 22x22 level scene (484 inputs),
       we crop a small 7x7 window placed in FRONT of and BELOW Mario.
       Mario does not need to see what is behind him or in the sky.
  2. Sight for enemies (Challenge 2):
       The level_scene already encodes enemies as integer codes (2-13).
       We build TWO separate 7x7 windows:
         - landscape window: only obstacle codes (-11, -10, 16, 20, 21)
         - enemies window:   only enemy codes (2..13)
       Both are normalized to [-1, 1].
  3. Elitism is handled by the evolutionary algorithm (see mario_evolution_mlp.py).

Resulting input dimension:
  7*7 (landscape) + 7*7 (enemies) + 4 (flags + mario_y) = 102 inputs
"""

import torch
import torch.nn as nn
import numpy as np
import marioai


# ---------------------------------------------------------------------------
# 1. Constants - which integer codes mean what in the level_scene grid
# ---------------------------------------------------------------------------
# Enemy codes (Goomba, Koopa, Bullet Bill, Spiky, Flower, Shell ...)
ENEMY_CODES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}

# Window size: how many cells we look at around Mario (must be odd)
WINDOW_SIZE = 7
WINDOW_HALF = WINDOW_SIZE // 2   # = 3

# Mario is fixed at row=11, col=11 inside the 22x22 grid
MARIO_ROW = 11
MARIO_COL = 11


# ---------------------------------------------------------------------------
# 2. The Neural Network (Multi-Layer Perceptron)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Small fully-connected network. Two hidden layers with tanh activations.
    Output uses sigmoid so each of the 5 controller buttons gets a probability
    in [0, 1], which is later thresholded to 0 or 1.
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),                # tanh works better with normalized inputs
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()              # output in [0, 1] -> threshold to action
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 3. The Agent that uses the MLP to decide actions
# ---------------------------------------------------------------------------
class MLPAgent(marioai.Agent):
    """
    Agent whose decision-making is a neural network.
    The weights of the network are the genotype of the evolutionary algorithm.
    """

    def __init__(self):
        super(MLPAgent, self).__init__()

        # Input dimension breakdown:
        #   landscape window: 7 * 7 = 49
        #   enemies   window: 7 * 7 = 49
        #   flags         : can_jump + on_ground = 2
        #   mario state   : mario_y (height) + a "time" placeholder = 2
        # Total = 102
        self.input_dim = WINDOW_SIZE * WINDOW_SIZE * 2 + 4
        self.output_dim = 5  # [backward, forward, crouch, jump, speed]

        # Build the MLP brain
        self.mlp = MLP(self.input_dim, self.output_dim)

        # Threshold to decide if a button is pressed (output > 0.5 => press)
        self.threshold = 0.5

    # -----------------------------------------------------------------------
    # 3.1 Helpers to extract features from the level_scene
    # -----------------------------------------------------------------------
    def _extract_window(self, scene):
        """
        Crop a 7x7 window from the 22x22 level_scene placed in FRONT of Mario
        and slightly BELOW him.

        We shift the center of the window so Mario sees mostly what is to
        his right (forward direction) and a couple of cells below his feet.
        """
        # Center of the window: 2 cells to the right, 1 cell below Mario
        center_row = MARIO_ROW + 1
        center_col = MARIO_COL + 2

        # Compute the slice bounds, clipping to grid limits
        r_start = max(0, center_row - WINDOW_HALF)
        r_end   = min(22, center_row + WINDOW_HALF + 1)
        c_start = max(0, center_col - WINDOW_HALF)
        c_end   = min(22, center_col + WINDOW_HALF + 1)

        # Cut the window from the scene
        window = scene[r_start:r_end, c_start:c_end]

        # If we hit a border, pad with zeros to keep size constant
        if window.shape != (WINDOW_SIZE, WINDOW_SIZE):
            padded = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.int32)
            padded[:window.shape[0], :window.shape[1]] = window
            window = padded

        return window

    def _split_landscape_and_enemies(self, window):
        """
        Build two binary maps (same shape as window):
          - landscape_map: 1 if the cell is an obstacle, 0 otherwise
          - enemy_map   : 1 if the cell contains an enemy, 0 otherwise
        Then we make landscape_map signed (-1 for hard obstacle, +1 for blocks)
        so the network can distinguish them.
        """
        landscape_map = np.zeros(window.shape, dtype=np.float32)
        enemy_map = np.zeros(window.shape, dtype=np.float32)

        # Iterate cell by cell
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                v = int(window[i, j])
                if v in ENEMY_CODES:
                    enemy_map[i, j] = 1.0
                elif v == -10:                     # hard obstacle
                    landscape_map[i, j] = -1.0
                elif v == -11:                     # soft obstacle
                    landscape_map[i, j] = -0.5
                elif v in (16, 20, 21):            # bricks / pots / question
                    landscape_map[i, j] = 1.0

        return landscape_map, enemy_map

    # -----------------------------------------------------------------------
    # 3.2 Sense and act
    # -----------------------------------------------------------------------
    def sense(self, obs):
        # Reuse the base class sense() to fill self.level_scene, etc.
        super(MLPAgent, self).sense(obs)

    def act(self):
        """
        Build the input vector, run the MLP forward, threshold to a binary
        action vector and return it.
        """
        # Safety check - first frame may not have observation yet
        if self.level_scene is None:
            return [0, 0, 0, 0, 0]

        # 1) Crop the window in front of Mario
        window = self._extract_window(self.level_scene)

        # 2) Split it into landscape and enemy maps
        land_map, enemy_map = self._split_landscape_and_enemies(window)

        # 3) Boolean / numerical flags
        flag_can_jump = float(self.can_jump) if self.can_jump is not None else 0.0
        flag_on_ground = float(self.on_ground) if self.on_ground is not None else 0.0
        # Mario's height in the level (normalized roughly to [0,1])
        mario_y = self.mario_floats[1] / 240.0 if self.mario_floats else 0.0
        # Constant bias input
        bias = 1.0

        # 4) Concatenate everything into a single flat vector
        inputs = np.concatenate([
            land_map.flatten(),
            enemy_map.flatten(),
            np.array([flag_can_jump, flag_on_ground, mario_y, bias],
                     dtype=np.float32)
        ]).astype(np.float32)

        # 5) Forward pass through the network (no gradient needed)
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output_tensor = self.mlp(input_tensor)

        # 6) Threshold the 5 outputs to get a binary action vector
        action_probs = output_tensor.numpy()
        action = (action_probs > self.threshold).astype(int).tolist()

        return action

    # -----------------------------------------------------------------------
    # 3.3 Genotype helpers - flatten/unflatten weights as a 1D vector
    # -----------------------------------------------------------------------
    def get_param_vector(self):
        """Flatten ALL network weights into a single numpy 1D array."""
        params = []
        for param in self.mlp.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_param_vector(self, vector):
        """Inverse of get_param_vector: load weights from a flat numpy array."""
        offset = 0
        for param in self.mlp.parameters():
            shape = param.shape
            size = int(np.prod(shape))
            chunk = vector[offset:offset + size].reshape(shape)
            param.data = torch.tensor(chunk, dtype=torch.float32)
            offset += size