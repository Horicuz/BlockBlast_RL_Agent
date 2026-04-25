import numpy as np
import random
# IMPORTĂM NOILE CATALOGOACE
from blocks import SHAPES, SIZE_CATEGORIES, COMPLEXITY_CATEGORIES

class BlockBlastLogic:
    def __init__(self):
        self.grid = np.zeros((8, 8), dtype=int)
        self.stages_passed = 0
        self.lines_destroyed = 0
        self.blocks_placed = 0 
        
        self.hand = []
        self.available = []
        self.get_new_hand()

    def _get_piece_info(self, key):
        """Caută direct în dicționarele statice din blocks.py"""
        size = 'small' if key in SIZE_CATEGORIES['small'] else 'medium' if key in SIZE_CATEGORIES['medium'] else 'large'
        comp = 'simple' if key in COMPLEXITY_CATEGORIES['simple'] else 'medium' if key in COMPLEXITY_CATEGORIES['medium'] else 'hard'
        return size, comp

    def get_new_hand(self):
        """Generare corectă folosind clasificarea din blocks.py"""
        empty_spaces = 64 - np.sum(self.grid)
        empty_ratio = empty_spaces / 64.0
        
        # Ponderi de Mărime (Se adaptează)
        m_large = (empty_ratio ** 2) * 2.5 
        m_medium = 1.0
        m_small = 1.0 + ((1.0 - empty_ratio) * 3.5)

        # Ponderi de Complexitate (Fixe)
        base_comp_weights = {'simple': 10.0, 'medium': 5.0, 'hard': 3.0}

        keys = list(SHAPES.keys())
        weights = []
        
        # Calculăm greutatea finală instantaneu
        for k in keys:
            size, comp = self._get_piece_info(k)
            w = base_comp_weights[comp]
            
            if size == 'small': w *= m_small
            elif size == 'medium': w *= m_medium
            elif size == 'large': w *= m_large
            weights.append(w)

        # Oracolul de siguranță
        for attempt in range(30):
            temp_keys = random.choices(keys, weights=weights, k=3)
            temp_hand = [SHAPES[k] for k in temp_keys]
            
            if any(self._can_fit_anywhere(b) for b in temp_hand):
                self.hand = temp_hand
                self.available = [True, True, True]
                return 
        
        self.hand = temp_hand
        self.available = [True, True, True]

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
            return False, False, 0, [], []
            
        block = self.hand[hand_index]
        if not self.can_place(block, row, col):
            return False, False, 0, [], []
            
        block_h, block_w = block.shape
        self.grid[row:row+block_h, col:col+block_w] += block
        self.available[hand_index] = False
        self.blocks_placed += 1 
        
        lines_cleared, f_rows, f_cols = self.clear_lines()
        
        if not any(self.available):
            self.get_new_hand()
            self.stages_passed += 1 
            
        is_game_over = self.check_game_over()
        
        return True, is_game_over, lines_cleared, f_rows, f_cols

    def check_game_over(self):
        for i, is_avail in enumerate(self.available):
            if is_avail:
                block = self.hand[i]
                for r in range(8):
                    for c in range(8):
                        if self.can_place(block, r, c):
                            return False
        return True