import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .fast_engine import ACTION_COUNT, BOARD_CELLS, FastBlockBlastLogic


class FastBlockBlastEnv(gym.Env):
    def __init__(self, reward_config=None):
        super().__init__()
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.MultiBinary(BOARD_CELLS),
                "hand": spaces.Box(low=0, high=1, shape=(3, 9), dtype=np.int8),
                "available": spaces.MultiBinary(3),
                "valid_actions": spaces.MultiBinary(ACTION_COUNT),
                "stats": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
            }
        )
        self.reward_config = {
            "line_clear_scale": 8.0,
            "stage_complete_reward": 3.0,
            "game_over_penalty": 60.0,
            "valid_action_delta_weight": 0.04,
            "filled_cell_penalty": 0.015,
            "pocket_penalty_weight": 0.15,
            "created_pocket_penalty_weight": 0.75,
            "line_potential_weight": 0.0,
            "line_potential_delta_weight": 0.0,
            "reward_scale": 1.0,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        self.game = FastBlockBlastLogic(rng=self._new_game_rng())

    def _new_game_rng(self):
        seed = int(self.np_random.integers(0, 2**32 - 1))
        return random.Random(seed)

    def valid_action_mask(self):
        return self.game.valid_action_mask()

    def _get_obs(self):
        board_flat = self.game.board_array().reshape(-1).astype(np.int8)
        hand = np.zeros((3, 9), dtype=np.int8)
        for hand_index, shape in enumerate(self.game.hand_arrays()):
            padded_shape = np.zeros((3, 3), dtype=np.int8)
            height, width = shape.shape
            padded_shape[:height, :width] = shape
            hand[hand_index] = padded_shape.reshape(-1)

        valid_actions = self.valid_action_mask()
        filled_ratio = self.game.filled_cells() / BOARD_CELLS
        valid_ratio = float(valid_actions.sum()) / ACTION_COUNT
        remaining_ratio = sum(self.game.available) / 3
        pocket_ratio = self.game.single_cell_pockets() / BOARD_CELLS
        line_ratio = min(self.game.lines_destroyed / 100.0, 1.0)
        stage_ratio = min(self.game.stages_passed / 100.0, 1.0)
        blocks_ratio = min(self.game.blocks_placed / 300.0, 1.0)
        empty_ratio = 1.0 - filled_ratio
        stats = np.array(
            [
                filled_ratio,
                empty_ratio,
                valid_ratio,
                remaining_ratio,
                pocket_ratio,
                line_ratio,
                stage_ratio,
                blocks_ratio,
            ],
            dtype=np.float32,
        )

        return {
            "board": board_flat,
            "hand": hand,
            "available": np.array(self.game.available, dtype=np.int8),
            "valid_actions": valid_actions.astype(np.int8),
            "stats": stats,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = FastBlockBlastLogic(rng=self._new_game_rng())
        return self._get_obs(), {}

    def step(self, action):
        cfg = self.reward_config
        previous_valid_actions = self.game.valid_action_count()
        previous_pockets = self.game.single_cell_pockets()
        previous_line_potential = self.game.line_potential()

        is_valid, is_game_over, lines_cleared, full_rows, full_cols, stage_completed = self.game.step(int(action))

        reward_line = 0.0
        reward_line_potential = 0.0
        reward_stage = 0.0
        reward_shape = 0.0
        reward_pockets = 0.0
        reward_game_over = 0.0

        current_valid_actions = self.game.valid_action_count()
        current_pockets = self.game.single_cell_pockets()
        current_line_potential = self.game.line_potential()
        created_pockets = max(current_pockets - previous_pockets, 0)

        if is_valid:
            reward_line = (lines_cleared**2) * cfg["line_clear_scale"]
            if lines_cleared == 0:
                reward_line_potential = current_line_potential * cfg["line_potential_weight"]
                reward_line_potential += (current_line_potential - previous_line_potential) * cfg["line_potential_delta_weight"]
            reward_stage = cfg["stage_complete_reward"] if stage_completed else 0.0
            reward_shape = (current_valid_actions - previous_valid_actions) * cfg["valid_action_delta_weight"]
            reward_shape -= self.game.filled_cells() * cfg["filled_cell_penalty"]
            reward_pockets = -(current_pockets * cfg["pocket_penalty_weight"])
            reward_pockets -= created_pockets * cfg["created_pocket_penalty_weight"]

        if is_game_over:
            reward_game_over = -cfg["game_over_penalty"]

        reward = (reward_line + reward_line_potential + reward_stage + reward_shape + reward_pockets + reward_game_over) * cfg["reward_scale"]

        info = {
            "anim_rows": full_rows,
            "anim_cols": full_cols,
            "reward/line": reward_line * cfg["reward_scale"],
            "reward/line_potential": reward_line_potential * cfg["reward_scale"],
            "reward/stage": reward_stage * cfg["reward_scale"],
            "reward/shape": reward_shape * cfg["reward_scale"],
            "reward/pockets": reward_pockets * cfg["reward_scale"],
            "reward/game_over": reward_game_over * cfg["reward_scale"],
            "board/filled_cells": self.game.filled_cells(),
            "board/pockets": current_pockets,
            "board/created_pockets": created_pockets,
            "board/line_potential": current_line_potential,
            "game/valid_actions": current_valid_actions,
        }

        if is_game_over:
            info["game/etap_max"] = self.game.stages_passed
            info["game/linii_distruse"] = self.game.lines_destroyed
            info["game/blocuri_puse"] = self.game.blocks_placed

        return self._get_obs(), reward, is_game_over, False, info
