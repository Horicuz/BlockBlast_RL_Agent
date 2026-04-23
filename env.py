import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine import BlockBlastLogic

class BlockBlastEnv(gym.Env):
    def __init__(self):
        super(BlockBlastEnv, self).__init__()
        self.game = BlockBlastLogic()
        self.action_space = spaces.Discrete(192)
        
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

    def _calculate_fragmentation_penalty(self):
        visited = np.zeros((8, 8), dtype=bool)
        penalty = 0
        for r in range(8):
            for c in range(8):
                if self.game.grid[r, c] == 0 and not visited[r, c]:
                    region_size = 0
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if curr_r < 0 or curr_r >= 8 or curr_c < 0 or curr_c >= 8:
                            continue
                        if visited[curr_r, curr_c] or self.game.grid[curr_r, curr_c] == 1:
                            continue
                        visited[curr_r, curr_c] = True
                        region_size += 1
                        stack.extend([
                            (curr_r - 1, curr_c), (curr_r + 1, curr_c), 
                            (curr_r, curr_c - 1), (curr_r, curr_c + 1)
                        ])
                    if region_size < 9:
                        penalty += (9 - region_size) 
        return penalty

    def step(self, action):
        hand_index = action // 64
        remainder = action % 64
        row = remainder // 8
        col = remainder % 8
        
        old_penalty = self._calculate_fragmentation_penalty()
        
        # PRELUĂM LINIILE DISTRUSE AICI
        is_valid, is_game_over, lines_cleared, f_rows, f_cols = self.game.step(hand_index, row, col)
        
        reward = 0
        if is_valid:
            if lines_cleared > 0:
                reward += (lines_cleared ** 2) * 10
            else:
                reward += 1 
            new_penalty = self._calculate_fragmentation_penalty()
            reward -= (new_penalty - old_penalty) * 1.5 

        if is_game_over:
            reward -= 100

        # TRIMITEM DATELE LA WATCH.PY
        info = {
            "anim_rows": f_rows,
            "anim_cols": f_cols
        }
        
        if is_game_over:
            info["game/etap_max"] = self.game.stages_passed
            info["game/linii_distruse"] = self.game.lines_destroyed
            info["game/blocuri_puse"] = self.game.blocks_placed

        return self._get_obs(), reward, is_game_over, False, info