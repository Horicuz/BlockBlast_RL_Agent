import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine import BlockBlastLogic

class BlockBlastEnv(gym.Env):
    def __init__(self, reward_config=None):
        super(BlockBlastEnv, self).__init__()
        self.game = BlockBlastLogic()
        self.action_space = spaces.Discrete(192)

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
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.int32),
            "hand": spaces.Box(low=0, high=1, shape=(3, 3, 3), dtype=np.int32),
            "available": spaces.MultiBinary(3)
        })

    def valid_action_mask(self):
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

    def _get_obs(self):
        padded_hand = np.zeros((3, 3, 3), dtype=np.int32)
        for i, block in enumerate(self.game.hand):
            h, w = block.shape
            padded_hand[i, :h, :w] = block
        return {
            "board": self.game.grid.copy(),
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
