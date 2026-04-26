import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from engine import BlockBlastLogic

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Dict observation space.
    Routes board through CNN, other inputs through linear layers, then concatenates.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract shapes from observation_space
        board_shape = observation_space["board"].shape  # (1, 8, 8)
        hand_shape = observation_space["hand"].shape    # (3, 3, 3)
        
        # CNN for board: input (1, 8, 8)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size: after 2 conv layers with same padding, size stays 8x8
        # 64 filters * 8 * 8
        cnn_output_size = 64 * 8 * 8
        
        # Linear layer for hand (3, 3, 3)
        hand_flat_size = np.prod(hand_shape)
        self.hand_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hand_flat_size, 64),
            nn.ReLU(),
        )
        
        # Linear for available (3 binary values)
        self.available_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(),
        )
        
        # Final linear layer to combine all features
        combined_size = cnn_output_size + 64 + 16
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        # Extract individual components
        board = observations["board"].float()  # Shape: (batch, 1, 8, 8)
        hand = observations["hand"].float()    # Shape: (batch, 3, 3, 3)
        available = observations["available"].float()  # Shape: (batch, 3)
        
        # Process board through CNN
        board_features = self.cnn(board)  # Shape: (batch, 4096)
        
        # Process hand through linear
        hand_features = self.hand_fc(hand)  # Shape: (batch, 64)
        
        # Process available through linear
        available_features = self.available_fc(available)  # Shape: (batch, 16)
        
        # Concatenate all features
        combined = torch.cat([board_features, hand_features, available_features], dim=1)
        
        # Final processing
        output = self.fc_combined(combined)  # Shape: (batch, features_dim)
        
        return output


class BlockBlastEnv(gym.Env):
    def __init__(self, reward_config=None, apply_hole_penalty=False):
        super(BlockBlastEnv, self).__init__()
        self.game = BlockBlastLogic()
        self.action_space = spaces.Discrete(192)
        self.apply_hole_penalty = apply_hole_penalty
        
        # Observation space with channel dimension for CNN compatibility
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(1, 8, 8), dtype=np.int32),
            "hand": spaces.Box(low=0, high=1, shape=(3, 3, 3), dtype=np.int32),
            "available": spaces.MultiBinary(3)
        })
        
        self.reward_config = {
            "placement_reward": 0.0,
            "line_clear_scale": 10.0,
            "line_clear_bonus": 0.0,
            "stage_complete_reward": 10.0,
            "no_line_penalty": 0.0,
            "game_over_penalty": 200.0,
            "game_over_early_weight": 3.0,
            "reward_scale": 1.0,
        }
        if reward_config:
            self.reward_config.update(reward_config)

    def valid_action_mask(self):
        """Return valid action mask for 192 possible placements (3 pieces * 8x8 grid)."""
        mask = np.zeros(192, dtype=np.int8)
        for hand_idx in range(3):
            if not self.game.available[hand_idx]:
                continue
            block = self.game.hand[hand_idx]
            for r in range(8):
                for c in range(8):
                    if self.game.can_place(block, r, c):
                        action_idx = (hand_idx * 64) + (r * 8) + c
                        mask[action_idx] = 1
        return mask

    def _calculate_holes_penalty(self, grid):
        """
        Calculate penalty for isolated empty regions (holes) using flood fill (BFS).
        
        An isolated hole is an empty region (0-valued cells) that is "small" 
        (e.g., size <= 3). Larger empty regions are acceptable playable space.
        
        Returns: penalty value (negative, to be added to reward)
        """
        visited = np.zeros_like(grid, dtype=bool)
        hole_penalty = 0.0
        
        def bfs_flood_fill(start_r, start_c):
            """BFS to find all connected empty cells."""
            if visited[start_r, start_c] or grid[start_r, start_c] != 0:
                return 0
            
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            region_size = 1
            
            while queue:
                r, c = queue.popleft()
                # Check 4 neighbors (orthogonal connectivity only)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
                        region_size += 1
            
            return region_size
        
        # Find all empty regions and penalize small ones
        for r in range(8):
            for c in range(8):
                if grid[r, c] == 0 and not visited[r, c]:
                    region_size = bfs_flood_fill(r, c)
                    # Penalize small isolated holes (size 1-3)
                    if 1 <= region_size <= 3:
                        hole_penalty += region_size * 0.5  # penalty = 0.5 to 1.5 per hole
        
        return -hole_penalty  # Return negative (penalty)

    def _get_obs(self):
        """Construct observation dict with board reshaped to (1, 8, 8)."""
        padded_hand = np.zeros((3, 3, 3), dtype=np.int32)
        for i, block in enumerate(self.game.hand):
            h, w = block.shape
            padded_hand[i, :h, :w] = block
        
        # Reshape board to (1, 8, 8) for CNN
        board_reshaped = self.game.grid.copy().reshape(1, 8, 8).astype(np.int32)
        
        return {
            "board": board_reshaped,
            "hand": padded_hand,
            "available": np.array(self.game.available, dtype=np.int8)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = BlockBlastLogic()
        return self._get_obs(), {}

    def step(self, action):
        hand_index = action // 64
        remainder = action % 64
        row = remainder // 8
        col = remainder % 8
        
        is_valid, is_game_over, lines_cleared, f_rows, f_cols, stage_completed = self.game.step(hand_index, row, col)
        
        cfg = self.reward_config
        reward = 0.0
        
        if is_valid:
            if lines_cleared > 0:
                reward += (lines_cleared ** 2) * cfg["line_clear_scale"]
                reward += lines_cleared * cfg["line_clear_bonus"]
            else:
                reward += cfg["placement_reward"]
                reward -= cfg["no_line_penalty"]
            
            if stage_completed:
                reward += cfg["stage_complete_reward"]
            
            # Apply hole penalty if enabled
            if self.apply_hole_penalty:
                hole_penalty = self._calculate_holes_penalty(self.game.grid)
                reward += hole_penalty
        
        if is_game_over:
            early_loss_multiplier = 1.0 + (cfg["game_over_early_weight"] / (self.game.stages_passed + 1))
            reward -= cfg["game_over_penalty"] * early_loss_multiplier
        
        reward *= cfg["reward_scale"]
        
        info = {
            "anim_rows": f_rows,
            "anim_cols": f_cols
        }
        
        if is_game_over:
            info["game/etap_max"] = self.game.stages_passed
            info["game/linii_distruse"] = self.game.lines_destroyed
            info["game/blocuri_puse"] = self.game.blocks_placed
        
        return self._get_obs(), reward, is_game_over, False, info
