import numpy as np
import random
from blocks import COMPLEXITY_BY_SHAPE, SHAPE_LIBRARY, SIZE_BY_SHAPE

BOARD_SIZE = 4
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

class BlockBlastLogic:
    def __init__(
        self,
        rng=None,
        shape_keys=None,
        hand_generator="solvable",
        max_hand_attempts=50,
        complexity_weights=None,
        board_size=BOARD_SIZE,
    ):
        self.rng = rng or random.Random()
        self.shape_keys = list(shape_keys or SHAPE_LIBRARY.keys())
        self.hand_generator = hand_generator
        self.max_hand_attempts = max_hand_attempts
        self.board_size = int(board_size)
        self.board_cells = self.board_size * self.board_size
        self.complexity_weights = self._normalize_complexity_weights(complexity_weights)
        self.grid = np.zeros((self.board_size, self.board_size), dtype=int)
        self.stages_passed = 0
        self.lines_destroyed = 0
        self.blocks_placed = 0 
        
        self.hand = []
        self.available = []
        if not self.get_new_hand():
            raise RuntimeError("Nu s-a putut genera o mana initiala solvabila.")

    @staticmethod
    def _normalize_complexity_weights(weights):
        default_weights = {
            "simple": 0.78,
            "medium": 0.18,
            "hard": 0.04,
        }
        if not weights:
            return default_weights

        merged = default_weights.copy()
        for key in default_weights:
            value = float(weights.get(key, merged[key]))
            merged[key] = max(value, 0.0)

        total = sum(merged.values())
        if total <= 1e-9:
            return default_weights

        return {key: value / total for key, value in merged.items()}

    def _shuffled_keys(self):
        keys = list(self.shape_keys)
        self.rng.shuffle(keys)
        return keys

    def _can_place_on_grid(self, grid, block, row, col):
        block_h, block_w = block.shape
        if row + block_h > self.board_size or col + block_w > self.board_size:
            return False
        target_area = grid[row:row+block_h, col:col+block_w]
        return not np.any(np.logical_and(target_area, block))

    def _get_legal_moves_on_grid(self, grid, block):
        block_h, block_w = block.shape
        legal_moves = []
        for row in range(self.board_size - block_h + 1):
            for col in range(self.board_size - block_w + 1):
                if self._can_place_on_grid(grid, block, row, col):
                    legal_moves.append((row, col))
        return legal_moves

    @staticmethod
    def _apply_move_on_grid(grid, block, row, col):
        next_grid = grid.copy()
        block_h, block_w = block.shape
        next_grid[row:row+block_h, col:col+block_w] += block

        full_rows = list(np.where(np.all(next_grid == 1, axis=1))[0])
        full_cols = list(np.where(np.all(next_grid == 1, axis=0))[0])
        if full_rows:
            next_grid[full_rows, :] = 0
        if full_cols:
            next_grid[:, full_cols] = 0

        return next_grid

    def _find_solvable_hand_sequence(self, grid, depth):
        if depth == 0:
            return []

        for key in self._shuffled_keys():
            block = SHAPE_LIBRARY[key]
            legal_moves = self._get_legal_moves_on_grid(grid, block)
            if not legal_moves:
                continue

            self.rng.shuffle(legal_moves)
            for row, col in legal_moves:
                next_grid = self._apply_move_on_grid(grid, block, row, col)
                suffix = self._find_solvable_hand_sequence(next_grid, depth - 1)
                if suffix is not None:
                    return [key] + suffix

        return None

    def _random_hand_keys(self):
        return [self.rng.choice(self.shape_keys) for _ in range(3)]

    def _sample_random_shape_key(self, candidate_keys=None):
        return self.rng.choice(list(candidate_keys or self.shape_keys))

    def _difficulty_progress(self):
        stage_progress = min(self.stages_passed / 35.0, 1.0)
        placement_progress = min(self.blocks_placed / 105.0, 1.0)
        return max(stage_progress, placement_progress)

    def _shape_size_weights(self):
        free_ratio = float(np.count_nonzero(self.grid == 0)) / float(self.board_cells)
        progress = self._difficulty_progress()

        if free_ratio >= 0.82:
            return {
                "small": 0.22 - (0.07 * progress),
                "medium": 0.33 + (0.03 * progress),
                "large": 0.45 + (0.04 * progress),
            }
        if free_ratio >= 0.68:
            return {
                "small": 0.30 - (0.08 * progress),
                "medium": 0.42 + (0.04 * progress),
                "large": 0.28 + (0.04 * progress),
            }
        if free_ratio >= 0.52:
            return {
                "small": 0.46 - (0.12 * progress),
                "medium": 0.44 + (0.08 * progress),
                "large": 0.10 + (0.04 * progress),
            }
        if free_ratio >= 0.38:
            return {
                "small": 0.68 - (0.16 * progress),
                "medium": 0.31 + (0.155 * progress),
                "large": 0.01 + (0.005 * progress),
            }
        if free_ratio >= 0.25:
            return {
                "small": 0.84 - (0.12 * progress),
                "medium": 0.159 + (0.119 * progress),
                "large": 0.001,
            }
        return {"small": 0.94, "medium": 0.06, "large": 0.0}

    def _shape_preference_weight(self, key):
        progress = self._difficulty_progress()
        if key.startswith("line_"):
            return 2.55 - (0.35 * progress)
        if key == "square_2x2":
            return 2.15
        if key.startswith("rect_"):
            return 1.90 + (0.20 * progress)
        if key == "square_3x3":
            return 0.42 + (0.45 * progress)
        if key.startswith("corner_3"):
            return 1.15
        if key.startswith(("L_4", "J_4")):
            return 1.00 + (0.35 * progress)
        if key.startswith("big_L"):
            return 0.52 + (0.30 * progress)
        if key.startswith(("T_4", "big_T")):
            return 0.62 + (0.75 * progress)
        if key.startswith(("S_4", "Z_4")):
            return 0.56 + (0.68 * progress)
        if key.startswith("diag_"):
            return 0.10 + (0.06 * progress)
        return 1.0

    def _board_density_shape_multiplier(self, key, free_ratio):
        progress = self._difficulty_progress()
        cell_count = int(np.count_nonzero(SHAPE_LIBRARY[key]))
        footprint_cells = int(SHAPE_LIBRARY[key].shape[0] * SHAPE_LIBRARY[key].shape[1])

        if cell_count > 4:
            if free_ratio < 0.25:
                return 0.0
            if free_ratio < 0.38:
                return 0.01
            if free_ratio < 0.52:
                return 0.08 + (0.04 * progress)
            if free_ratio < 0.68:
                return 0.45 + (0.10 * progress)
            return 1.0 + (0.15 * progress)

        if footprint_cells >= 9 and free_ratio < 0.45:
            return 0.35

        return 1.0

    def _complexity_progress_multiplier(self, shape_complexity):
        progress = self._difficulty_progress()
        if shape_complexity == "simple":
            return 1.0 - (0.24 * progress)
        if shape_complexity == "medium":
            return 1.0 + (0.95 * progress)
        if shape_complexity == "hard":
            return 1.0 + (3.40 * progress)
        return 1.0

    def _adaptive_shape_weights(self, candidate_keys=None):
        free_ratio = float(np.count_nonzero(self.grid == 0)) / float(self.board_cells)
        size_weights = self._shape_size_weights()
        keys = list(candidate_keys or self.shape_keys)
        weights = {}

        for key in keys:
            shape_size = SIZE_BY_SHAPE.get(key, "medium")
            shape_complexity = COMPLEXITY_BY_SHAPE.get(key, "medium")
            weight = (
                size_weights.get(shape_size, 0.0)
                * self.complexity_weights.get(shape_complexity, 0.0)
                * self._complexity_progress_multiplier(shape_complexity)
                * self._shape_preference_weight(key)
                * self._board_density_shape_multiplier(key, free_ratio)
            )
            weights[key] = max(weight, 0.0)

        total = sum(weights.values())
        if total <= 1e-9 and keys:
            fallback = 1.0 / len(keys)
            return {key: fallback for key in keys}

        return weights

    def _sample_weighted_shape_key(self, candidate_keys=None):
        weights = self._adaptive_shape_weights(candidate_keys)
        total = sum(weights.values())
        if total <= 1e-9:
            keys = list(candidate_keys or self.shape_keys)
            return self.rng.choice(keys)

        cumulative = 0.0
        target = self.rng.random() * total
        for key, weight in weights.items():
            if weight <= 0:
                continue
            cumulative += weight
            if cumulative >= target:
                return key
        return next(reversed(weights))

    def _has_any_legal_move_for_keys(self, hand_keys):
        for key in hand_keys:
            if self._get_legal_moves_on_grid(self.grid, SHAPE_LIBRARY[key]):
                return True
        return False

    def _get_hand_with_playable_piece(self, sample_key):
        for _ in range(self.max_hand_attempts):
            hand_keys = [sample_key() for _ in range(3)]
            if self._has_any_legal_move_for_keys(hand_keys):
                return hand_keys

        fitting_keys = [
            key for key in self.shape_keys
            if self._get_legal_moves_on_grid(self.grid, SHAPE_LIBRARY[key])
        ]
        if not fitting_keys:
            return None

        hand_keys = [sample_key(fitting_keys)]
        hand_keys.extend(sample_key() for _ in range(2))
        self.rng.shuffle(hand_keys)
        return hand_keys

    def get_new_hand(self):
        if self.hand_generator == "random":
            hand_keys = self._random_hand_keys()
        elif self.hand_generator == "playable":
            hand_keys = self._get_hand_with_playable_piece(self._sample_random_shape_key)
        elif self.hand_generator == "adaptive_playable":
            hand_keys = self._get_hand_with_playable_piece(self._sample_weighted_shape_key)
        elif self.hand_generator == "solvable":
            hand_keys = self._find_solvable_hand_sequence(self.grid.copy(), depth=3)
        else:
            raise ValueError(f"Generator de mana necunoscut: {self.hand_generator}")

        if hand_keys is None:
            self.available = [False, False, False]
            return False

        if self.hand_generator == "solvable":
            self.rng.shuffle(hand_keys)

        self.hand = [SHAPE_LIBRARY[key] for key in hand_keys]
        self.available = [True, True, True]
        return True

    def can_place(self, block, row, col):
        return self._can_place_on_grid(self.grid, block, row, col)

    def clear_lines(self):
        full_rows = list(np.where(np.all(self.grid == 1, axis=1))[0])
        full_cols = list(np.where(np.all(self.grid == 1, axis=0))[0])
        
        lines_cleared = len(full_rows) + len(full_cols)
        
        if lines_cleared > 0:
            self.grid[full_rows, :] = 0
            self.grid[:, full_cols] = 0
            self.lines_destroyed += lines_cleared
            
        return lines_cleared, full_rows, full_cols

    def step(self, hand_index, row, col):
        if not self.available[hand_index]:
            return False, False, 0, [], [], False
            
        block = self.hand[hand_index]
        if not self.can_place(block, row, col):
            return False, False, 0, [], [], False
            
        block_h, block_w = block.shape
        self.grid[row:row+block_h, col:col+block_w] += block
        self.available[hand_index] = False
        self.blocks_placed += 1 
        
        lines_cleared, f_rows, f_cols = self.clear_lines()
        stage_completed = False
        
        if not any(self.available):
            if self.get_new_hand():
                self.stages_passed += 1
                stage_completed = True
            
        is_game_over = self.check_game_over()
        
        return True, is_game_over, lines_cleared, f_rows, f_cols, stage_completed

    def check_game_over(self):
        for i, is_avail in enumerate(self.available):
            if is_avail:
                block = self.hand[i]
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        if self.can_place(block, r, c):
                            return False
        return True
