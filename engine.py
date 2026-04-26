import numpy as np
import random
from blocks import COMPLEXITY_BY_SHAPE, SHAPES, SIZE_BY_SHAPE

class BlockBlastLogic:
    def __init__(
        self,
        rng=None,
        shape_keys=None,
        hand_generator="solvable",
        max_hand_attempts=50,
        complexity_weights=None,
    ):
        self.rng = rng or random.Random()
        self.shape_keys = list(shape_keys or SHAPES.keys())
        self.hand_generator = hand_generator
        self.max_hand_attempts = max_hand_attempts
        self.complexity_weights = self._normalize_complexity_weights(complexity_weights)
        self.grid = np.zeros((8, 8), dtype=int)
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
            "simple": 0.62,
            "medium": 0.28,
            "hard": 0.10,
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

    @staticmethod
    def _can_place_on_grid(grid, block, row, col):
        block_h, block_w = block.shape
        if row + block_h > 8 or col + block_w > 8:
            return False
        target_area = grid[row:row+block_h, col:col+block_w]
        return not np.any(np.logical_and(target_area, block))

    def _get_legal_moves_on_grid(self, grid, block):
        block_h, block_w = block.shape
        legal_moves = []
        for row in range(8 - block_h + 1):
            for col in range(8 - block_w + 1):
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
            block = SHAPES[key]
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

    def _shape_size_weights(self):
        """
        Adapt shape-size probabilities to board occupancy.
        More free space => bias towards large pieces.
        Less free space => bias towards small pieces.
        """
        free_ratio = float(np.count_nonzero(self.grid == 0)) / 64.0
        if free_ratio >= 0.65:
            return {"small": 0.20, "medium": 0.35, "large": 0.45}
        if free_ratio >= 0.35:
            return {"small": 0.35, "medium": 0.45, "large": 0.20}
        return {"small": 0.65, "medium": 0.30, "large": 0.05}

    def _adaptive_shape_weights(self):
        size_weights = self._shape_size_weights()
        shape_weights = {}

        for key in self.shape_keys:
            shape_size = SIZE_BY_SHAPE.get(key, "medium")
            shape_complexity = COMPLEXITY_BY_SHAPE.get(key, "medium")
            size_weight = size_weights.get(shape_size, 0.0)
            complexity_weight = self.complexity_weights.get(shape_complexity, 0.0)
            shape_weights[key] = max(size_weight * complexity_weight, 0.0)

        total = sum(shape_weights.values())
        if total <= 1e-9:
            fallback = 1.0 / max(len(self.shape_keys), 1)
            return {key: fallback for key in self.shape_keys}

        return {key: value / total for key, value in shape_weights.items()}

    def _sample_weighted_shape_key(self, normalized_weights):
        cumulative = 0.0
        target = self.rng.random()
        for key, weight in normalized_weights.items():
            cumulative += weight
            if cumulative >= target:
                return key
        return self.shape_keys[-1]

    def _adaptive_hand_keys(self):
        normalized_weights = self._adaptive_shape_weights()
        return [self._sample_weighted_shape_key(normalized_weights) for _ in range(3)]

    def _has_any_legal_move_for_keys(self, hand_keys):
        for key in hand_keys:
            if self._get_legal_moves_on_grid(self.grid, SHAPES[key]):
                return True
        return False

    def _get_random_hand_with_playable_piece(self):
        """
        Generate a random hand, but guarantee at least one piece can be placed.

        This is much cheaper than solving the full 3-piece sequence and closer to
        a gentle game generator: the player gets a chance, but not a guaranteed
        perfect route through the whole hand.
        """
        for _ in range(self.max_hand_attempts):
            hand_keys = self._random_hand_keys()
            if self._has_any_legal_move_for_keys(hand_keys):
                return hand_keys

        fitting_keys = [
            key for key in self.shape_keys
            if self._get_legal_moves_on_grid(self.grid, SHAPES[key])
        ]
        if not fitting_keys:
            return None

        hand_keys = [self.rng.choice(fitting_keys)]
        hand_keys.extend(self.rng.choice(self.shape_keys) for _ in range(2))
        self.rng.shuffle(hand_keys)
        return hand_keys

    def _get_adaptive_hand_with_playable_piece(self):
        """
        Generate a hand from adaptive weighted probabilities, while preserving
        the "at least one legal move exists" safety.
        """
        normalized_weights = self._adaptive_shape_weights()
        for _ in range(self.max_hand_attempts):
            hand_keys = [self._sample_weighted_shape_key(normalized_weights) for _ in range(3)]
            if self._has_any_legal_move_for_keys(hand_keys):
                return hand_keys

        fitting_keys = [
            key for key in self.shape_keys
            if self._get_legal_moves_on_grid(self.grid, SHAPES[key])
        ]
        if not fitting_keys:
            return None

        fitting_weights = {key: normalized_weights.get(key, 0.0) for key in fitting_keys}
        total = sum(fitting_weights.values())
        if total <= 1e-9:
            normalized_fitting_weights = {key: 1.0 / len(fitting_keys) for key in fitting_keys}
        else:
            normalized_fitting_weights = {key: value / total for key, value in fitting_weights.items()}

        hand_keys = [self._sample_weighted_shape_key(normalized_fitting_weights)]
        hand_keys.extend(self._sample_weighted_shape_key(normalized_weights) for _ in range(2))
        self.rng.shuffle(hand_keys)
        return hand_keys

    def get_new_hand(self):
        """Generează o mână nouă folosind strategia configurată."""
        if self.hand_generator == "random":
            hand_keys = self._random_hand_keys()
        elif self.hand_generator == "playable":
            hand_keys = self._get_random_hand_with_playable_piece()
        elif self.hand_generator == "adaptive_playable":
            hand_keys = self._get_adaptive_hand_with_playable_piece()
        elif self.hand_generator == "solvable":
            hand_keys = self._find_solvable_hand_sequence(self.grid.copy(), depth=3)
        else:
            raise ValueError(f"Generator de mana necunoscut: {self.hand_generator}")

        if hand_keys is None:
            self.available = [False, False, False]
            return False

        if self.hand_generator == "solvable":
            self.rng.shuffle(hand_keys)

        self.hand = [SHAPES[key] for key in hand_keys]
        self.available = [True, True, True]
        return True

    def get_new_solvable_hand(self):
        """Backward-compatible alias for older experiments."""
        hand_keys = self._find_solvable_hand_sequence(self.grid.copy(), depth=3)

        if hand_keys is None:
            self.available = [False, False, False]
            return False

        self.rng.shuffle(hand_keys)
        self.hand = [SHAPES[key] for key in hand_keys]
        self.available = [True, True, True]
        return True

    def _can_fit_anywhere(self, block):
        bh, bw = block.shape
        for r in range(8 - bh + 1):
            for c in range(8 - bw + 1):
                if not np.any(np.logical_and(self.grid[r:r+bh, c:c+bw], block)):
                    return True
        return False

    def can_place(self, block, row, col):
        block_h, block_w = block.shape
        if row + block_h > 8 or col + block_w > 8:
            return False
        target_area = self.grid[row:row+block_h, col:col+block_w]
        if np.any(np.logical_and(target_area, block)):
            return False
        return True

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
                for r in range(8):
                    for c in range(8):
                        if self.can_place(block, r, c):
                            return False
        return True
