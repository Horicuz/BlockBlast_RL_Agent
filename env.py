import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
from engine import BlockBlastLogic

import torch
import torch.nn as nn
from blocks import TRAINING_POOLS
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Dict observation space.
    Routes board through CNN, other inputs through linear layers, then concatenates.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        board_shape = observation_space["board"].shape  # (1, 8, 8)
        legal_shape = observation_space["valid_actions"].shape  # (3, 8, 8)
        hand_shape = observation_space["hand"].shape    # (3, 3, 3)
        
        spatial_channels = board_shape[0] + legal_shape[0]
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        spatial_output_size = 64 * 8 * 8
        
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
        combined_size = spatial_output_size + 64 + 16
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        # Extract individual components
        board = observations["board"].float()  # Shape: (batch, 1, 8, 8)
        valid_actions = observations["valid_actions"].float()  # Shape: (batch, 3, 8, 8)
        hand = observations["hand"].float()    # Shape: (batch, 3, 3, 3)
        available = observations["available"].float()  # Shape: (batch, 3)
        
        spatial = torch.cat([board, valid_actions], dim=1)
        spatial_features = self.spatial_cnn(spatial)  # Shape: (batch, 4096)
        
        # Process hand through linear
        hand_features = self.hand_fc(hand)  # Shape: (batch, 64)
        
        # Process available through linear
        available_features = self.available_fc(available)  # Shape: (batch, 16)
        
        # Concatenate all features
        combined = torch.cat([spatial_features, hand_features, available_features], dim=1)
        
        # Final processing
        output = self.fc_combined(combined)  # Shape: (batch, features_dim)
        
        return output


class BlockBlastEnv(gym.Env):
    def __init__(
        self,
        reward_config=None,
        apply_hole_penalty=False,
        fixed_game_seed=None,
        shape_pool="all",
        hand_generator="solvable",
    ):
        super(BlockBlastEnv, self).__init__()
        self.fixed_game_seed = fixed_game_seed
        self.shape_pool = shape_pool
        self.hand_generator = hand_generator
        self.shape_keys = TRAINING_POOLS[shape_pool]
        self.game = BlockBlastLogic(
            rng=self._new_game_rng(),
            shape_keys=self.shape_keys,
            hand_generator=self.hand_generator,
        )
        self.action_space = spaces.Discrete(192)
        self.apply_hole_penalty = apply_hole_penalty
        
        # Observation space with channel dimension for CNN compatibility
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(1, 8, 8), dtype=np.int32),
            "valid_actions": spaces.MultiBinary((3, 8, 8)),
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
            "hole_penalty_weight": 0.25,
            "created_hole_penalty_weight": 1.0,
        }
        if reward_config:
            self.reward_config.update(reward_config)

    def _new_game_rng(self):
        if self.fixed_game_seed is not None:
            return random.Random(self.fixed_game_seed)
        seed = int(self.np_random.integers(0, 2**32 - 1))
        return random.Random(seed)

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

    def _count_small_hole_cells(self, grid):
        """
        Count cells that belong to small isolated empty regions using flood fill (BFS).
        
        An isolated hole is an empty region (0-valued cells) that is "small" 
        (e.g., size <= 3). Larger empty regions are acceptable playable space.
        """
        visited = np.zeros_like(grid, dtype=bool)
        small_hole_cells = 0
        
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
                    if 1 <= region_size <= 3:
                        small_hole_cells += region_size
        
        return small_hole_cells

    def _calculate_holes_penalty(self, grid, previous_grid=None):
        """
        Penalize both current small holes and newly-created small holes.

        The created-hole component matters most for learning because it connects the
        penalty to the action that worsened board shape.
        """
        cfg = self.reward_config
        current_hole_cells = self._count_small_hole_cells(grid)
        penalty = current_hole_cells * cfg["hole_penalty_weight"]
        previous_hole_cells = None
        created_hole_cells = 0

        if previous_grid is not None:
            previous_hole_cells = self._count_small_hole_cells(previous_grid)
            created_hole_cells = max(current_hole_cells - previous_hole_cells, 0)
            penalty += created_hole_cells * cfg["created_hole_penalty_weight"]

        return {
            "penalty": -penalty,
            "current_hole_cells": current_hole_cells,
            "previous_hole_cells": previous_hole_cells,
            "created_hole_cells": created_hole_cells,
        }

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
            "valid_actions": self.valid_action_mask().reshape(3, 8, 8).astype(np.int8),
            "hand": padded_hand,
            "available": np.array(self.game.available, dtype=np.int8)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = BlockBlastLogic(
            rng=self._new_game_rng(),
            shape_keys=self.shape_keys,
            hand_generator=self.hand_generator,
        )
        return self._get_obs(), {}

    def step(self, action):
        hand_index = action // 64
        remainder = action % 64
        row = remainder // 8
        col = remainder % 8
        previous_grid = self.game.grid.copy()
        
        is_valid, is_game_over, lines_cleared, f_rows, f_cols, stage_completed = self.game.step(hand_index, row, col)
        
        cfg = self.reward_config
        reward = 0.0
        reward_line = 0.0
        reward_stage = 0.0
        reward_holes = 0.0
        reward_game_over = 0.0
        hole_stats = {
            "current_hole_cells": self._count_small_hole_cells(self.game.grid),
            "previous_hole_cells": None,
            "created_hole_cells": 0,
        }
        
        if is_valid:
            if lines_cleared > 0:
                reward_line = (lines_cleared ** 2) * cfg["line_clear_scale"]
                reward_line += lines_cleared * cfg["line_clear_bonus"]
                reward += reward_line
            else:
                reward += cfg["placement_reward"]
                reward -= cfg["no_line_penalty"]
            
            if stage_completed:
                reward_stage = cfg["stage_complete_reward"]
                reward += reward_stage
            
            if self.apply_hole_penalty:
                hole_stats = self._calculate_holes_penalty(self.game.grid, previous_grid)
                reward_holes = hole_stats["penalty"]
                reward += reward_holes
        
        if is_game_over:
            early_loss_multiplier = 1.0 + (cfg["game_over_early_weight"] / (self.game.stages_passed + 1))
            reward_game_over = -cfg["game_over_penalty"] * early_loss_multiplier
            reward += reward_game_over
        
        reward *= cfg["reward_scale"]
        
        info = {
            "anim_rows": f_rows,
            "anim_cols": f_cols,
            "reward/line": reward_line * cfg["reward_scale"],
            "reward/stage": reward_stage * cfg["reward_scale"],
            "reward/holes": reward_holes * cfg["reward_scale"],
            "reward/game_over": reward_game_over * cfg["reward_scale"],
            "holes/current_cells": hole_stats["current_hole_cells"],
            "holes/created_cells": hole_stats["created_hole_cells"],
            "game/valid_actions": int(self.valid_action_mask().sum()),
        }
        
        if is_game_over:
            info["game/etap_max"] = self.game.stages_passed
            info["game/linii_distruse"] = self.game.lines_destroyed
            info["game/blocuri_puse"] = self.game.blocks_placed
        
        return self._get_obs(), reward, is_game_over, False, info
