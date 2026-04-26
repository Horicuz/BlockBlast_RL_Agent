import numpy as np
import random
from blocks import SHAPES

class BlockBlastLogic:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.grid = np.zeros((8, 8), dtype=int)
        self.stages_passed = 0
        self.lines_destroyed = 0
        self.blocks_placed = 0 
        
        self.hand = []
        self.available = []
        if not self.get_new_hand():
            raise RuntimeError("Nu s-a putut genera o mana initiala solvabila.")

    def _shuffled_keys(self):
        keys = list(SHAPES.keys())
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

    def get_new_hand(self):
        """Generează o mână complet jucabilă, inclusiv secvențe care cer clear între piese."""
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
