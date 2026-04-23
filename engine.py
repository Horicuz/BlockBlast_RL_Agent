import numpy as np
import random
from blocks import SHAPES

class BlockBlastLogic:
    def __init__(self):
        self.grid = np.zeros((8, 8), dtype=int)
        
        self.stages_passed = 0
        self.lines_destroyed = 0
        self.blocks_placed = 0 
        
        self.hand = []
        self.available = []
        self.get_new_hand()

    def get_new_hand(self):
        shape_keys = list(SHAPES.keys())
        self.hand = [SHAPES[random.choice(shape_keys)] for _ in range(3)]
        self.available = [True, True, True]

    def can_place(self, block, row, col):
        block_h, block_w = block.shape
        if row + block_h > 8 or col + block_w > 8:
            return False
        target_area = self.grid[row:row+block_h, col:col+block_w]
        if np.any(np.logical_and(target_area, block)):
            return False
        return True

    def clear_lines(self):
        # Aflăm exact INDEXUL rândurilor și coloanelor pline
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
        
        # Salvăm rândurile/coloanele ca să le trimitem la animație
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