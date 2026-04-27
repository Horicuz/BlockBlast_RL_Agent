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


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class CustomCNNExtractorV2(BaseFeaturesExtractor):
    """
    Deeper CNN feature extractor for the same Dict observation space.
    Keeps full 8x8 resolution longer, then mixes board/action geometry with
    hand metadata through a larger feature head.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        board_shape = observation_space["board"].shape
        legal_shape = observation_space["valid_actions"].shape
        hand_shape = observation_space["hand"].shape

        spatial_channels = board_shape[0] + legal_shape[0]
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            ResidualConvBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResidualConvBlock(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
            ResidualConvBlock(96),
            nn.Flatten(),
        )

        spatial_output_size = 96 * 8 * 8
        hand_flat_size = np.prod(hand_shape)
        self.hand_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hand_flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.available_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        combined_size = spatial_output_size + 64 + 16
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        board = observations["board"].float()
        valid_actions = observations["valid_actions"].float()
        hand = observations["hand"].float()
        available = observations["available"].float()

        spatial = torch.cat([board, valid_actions], dim=1)
        spatial_features = self.spatial_cnn(spatial)
        hand_features = self.hand_fc(hand)
        available_features = self.available_fc(available)
        combined = torch.cat([spatial_features, hand_features, available_features], dim=1)
        return self.fc_combined(combined)


class ActionAwareCNNExtractor(BaseFeaturesExtractor):
    """
    Action-aware extractor for the 8x8 board.

    The older extractors mix all valid-action maps together early. This one keeps
    each hand slot separate for longer: for each slot it sees board + that slot's
    legal-placement map + an embedding of that slot's shape, then a shared CNN
    builds slot-specific features. That is closer to how the heuristic thinks:
    score concrete placements for a concrete piece.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 768):
        super().__init__(observation_space, features_dim)

        hand_shape = observation_space["hand"].shape
        hand_flat_size = int(np.prod(hand_shape[1:]))

        coord_y = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        coord_x = torch.linspace(-1.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coord_channels", torch.cat([coord_y, coord_x], dim=1), persistent=False)

        self.slot_hand_encoder = nn.Sequential(
            nn.Linear(hand_flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
        )

        slot_channels = 1 + 1 + 1 + 2 + 24
        self.slot_cnn = nn.Sequential(
            nn.Conv2d(slot_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResidualConvBlock(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
            ResidualConvBlock(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )

        self.global_cnn = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResidualConvBlock(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 192),
            nn.ReLU(),
        )

        self.hand_global = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(hand_shape)), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.available_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 32),
            nn.ReLU(),
        )

        combined_size = (3 * 256) + 192 + 64 + 32
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        board = observations["board"].float()
        valid_actions = observations["valid_actions"].float()
        hand = observations["hand"].float()
        available = observations["available"].float()

        batch_size = board.shape[0]
        slot_features = []
        coords = self.coord_channels.expand(batch_size, -1, -1, -1)

        for slot_idx in range(3):
            slot_hand = hand[:, slot_idx].reshape(batch_size, -1)
            slot_embedding = self.slot_hand_encoder(slot_hand).view(batch_size, 24, 1, 1).expand(-1, -1, 8, 8)
            slot_available = available[:, slot_idx].view(batch_size, 1, 1, 1).expand(-1, 1, 8, 8)
            slot_valid = valid_actions[:, slot_idx:slot_idx + 1]

            slot_input = torch.cat([board, slot_valid, slot_available, coords, slot_embedding], dim=1)
            slot_features.append(self.slot_cnn(slot_input))

        all_valid = valid_actions
        global_input = torch.cat([board, all_valid], dim=1)
        global_features = self.global_cnn(global_input)
        hand_features = self.hand_global(hand)
        available_features = self.available_fc(available)

        combined = torch.cat([*slot_features, global_features, hand_features, available_features], dim=1)
        return self.fc_combined(combined)


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
            complexity_weights=self._complexity_weights_from_reward_config(reward_config),
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
            "contact_reward_scale": 12.0,
            "contact_reward_power": 1.25,
            "contact_reward_threshold": 0.0,
            "contact_penalty_scale": 0.0,
            "complexity_simple_prob": 0.78,
            "complexity_medium_prob": 0.18,
            "complexity_hard_prob": 0.04,
        }
        if reward_config:
            self.reward_config.update(reward_config)

    @staticmethod
    def _complexity_weights_from_reward_config(reward_config):
        if not reward_config:
            return None
        return {
            "simple": reward_config.get("complexity_simple_prob", 0.78),
            "medium": reward_config.get("complexity_medium_prob", 0.18),
            "hard": reward_config.get("complexity_hard_prob", 0.04),
        }

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

    def _calculate_contact_stats(self, previous_grid, block, row, col):
        """
        Compute how much of the placed piece perimeter touches either board edges
        or already-filled board cells. Internal piece edges are ignored.
        """
        block_h, block_w = block.shape
        touching_edges = 0
        external_edges = 0

        for br in range(block_h):
            for bc in range(block_w):
                if block[br, bc] == 0:
                    continue

                board_r = row + br
                board_c = col + bc

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nbr_br = br + dr
                    nbr_bc = bc + dc

                    if 0 <= nbr_br < block_h and 0 <= nbr_bc < block_w and block[nbr_br, nbr_bc] == 1:
                        continue

                    external_edges += 1
                    nbr_r = board_r + dr
                    nbr_c = board_c + dc
                    if nbr_r < 0 or nbr_r >= 8 or nbr_c < 0 or nbr_c >= 8:
                        touching_edges += 1
                        continue
                    if previous_grid[nbr_r, nbr_c] == 1:
                        touching_edges += 1

        ratio = 0.0 if external_edges == 0 else (touching_edges / external_edges)
        return {
            "touching_edges": touching_edges,
            "external_edges": external_edges,
            "contact_ratio": ratio,
        }

    def _calculate_contact_reward(self, contact_ratio):
        cfg = self.reward_config
        threshold = max(0.0, min(float(cfg["contact_reward_threshold"]), 0.99))
        power = cfg["contact_reward_power"]

        if threshold <= 0.0:
            reward = (contact_ratio ** power) * cfg["contact_reward_scale"]
            return reward, contact_ratio

        if contact_ratio < threshold:
            miss_ratio = (threshold - contact_ratio) / max(threshold, 1e-6)
            reward = -((miss_ratio ** power) * cfg["contact_penalty_scale"])
            return reward, -miss_ratio

        surplus_ratio = (contact_ratio - threshold) / max(1.0 - threshold, 1e-6)
        reward = (surplus_ratio ** power) * cfg["contact_reward_scale"]
        return reward, surplus_ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = BlockBlastLogic(
            rng=self._new_game_rng(),
            shape_keys=self.shape_keys,
            hand_generator=self.hand_generator,
            complexity_weights=self._complexity_weights_from_reward_config(self.reward_config),
        )
        return self._get_obs(), {}

    def step(self, action):
        hand_index = action // 64
        remainder = action % 64
        row = remainder // 8
        col = remainder % 8
        previous_grid = self.game.grid.copy()
        placed_block = None
        if 0 <= hand_index < len(self.game.hand):
            placed_block = self.game.hand[hand_index].copy()
        
        is_valid, is_game_over, lines_cleared, f_rows, f_cols, stage_completed = self.game.step(hand_index, row, col)
        
        cfg = self.reward_config
        reward = 0.0
        reward_line = 0.0
        reward_stage = 0.0
        reward_contact = 0.0
        reward_holes = 0.0
        reward_game_over = 0.0
        hole_stats = {
            "current_hole_cells": 0,
            "previous_hole_cells": None,
            "created_hole_cells": 0,
        }
        contact_stats = {
            "touching_edges": 0,
            "external_edges": 0,
            "contact_ratio": 0.0,
        }
        contact_score = 0.0
        
        if is_valid:
            if placed_block is not None:
                contact_stats = self._calculate_contact_stats(previous_grid, placed_block, row, col)
            reward_contact, contact_score = self._calculate_contact_reward(contact_stats["contact_ratio"])
            reward += reward_contact

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
            "reward/contact": reward_contact * cfg["reward_scale"],
            "reward/holes": reward_holes * cfg["reward_scale"],
            "reward/game_over": reward_game_over * cfg["reward_scale"],
            "placement/contact_ratio": contact_stats["contact_ratio"],
            "placement/contact_score": contact_score,
            "placement/contact_threshold": cfg["contact_reward_threshold"],
            "placement/touching_edges": contact_stats["touching_edges"],
            "holes/current_cells": hole_stats["current_hole_cells"],
            "holes/created_cells": hole_stats["created_hole_cells"],
            "game/valid_actions": int(self.valid_action_mask().sum()),
        }
        
        if is_game_over:
            info["game/etap_max"] = self.game.stages_passed
            info["game/linii_distruse"] = self.game.lines_destroyed
            info["game/blocuri_puse"] = self.game.blocks_placed
        
        return self._get_obs(), reward, is_game_over, False, info
