import random

import numpy as np

from blocks import SHAPES


GRID_SIZE = 8
ACTION_COUNT = 3 * GRID_SIZE * GRID_SIZE
BOARD_CELLS = GRID_SIZE * GRID_SIZE
BOARD_MASK = (1 << BOARD_CELLS) - 1
ROW_MASKS = [sum(1 << (row * GRID_SIZE + col) for col in range(GRID_SIZE)) for row in range(GRID_SIZE)]
COL_MASKS = [sum(1 << (row * GRID_SIZE + col) for row in range(GRID_SIZE)) for col in range(GRID_SIZE)]
SHAPE_KEYS = tuple(SHAPES.keys())


def _shape_placement_masks(shape):
    masks = [0] * BOARD_CELLS
    height, width = shape.shape
    for row in range(GRID_SIZE - height + 1):
        for col in range(GRID_SIZE - width + 1):
            mask = 0
            for shape_row in range(height):
                for shape_col in range(width):
                    if shape[shape_row, shape_col]:
                        bit_index = (row + shape_row) * GRID_SIZE + col + shape_col
                        mask |= 1 << bit_index
            masks[row * GRID_SIZE + col] = mask
    return tuple(masks)


PLACEMENT_MASKS = {key: _shape_placement_masks(shape) for key, shape in SHAPES.items()}
SHAPE_CELL_COUNTS = {key: int(shape.sum()) for key, shape in SHAPES.items()}


class FastBlockBlastLogic:
    def __init__(self, rng=None, hand_size=3, shape_keys=None):
        self.rng = rng or random.Random()
        self.hand_size = hand_size
        self.shape_keys = tuple(shape_keys or SHAPE_KEYS)
        self.board_mask = 0
        self.stages_passed = 0
        self.lines_destroyed = 0
        self.blocks_placed = 0
        self.hand = []
        self.available = []
        self.get_new_hand()

    def get_new_hand(self):
        self.hand = [self.rng.choice(self.shape_keys) for _ in range(self.hand_size)]
        self.available = [True] * self.hand_size
        return True

    def can_place(self, hand_index, position):
        if not self.available[hand_index]:
            return False
        placement = PLACEMENT_MASKS[self.hand[hand_index]][position]
        return placement != 0 and (self.board_mask & placement) == 0

    def valid_action_mask(self):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        for hand_index, is_available in enumerate(self.available):
            if not is_available:
                continue
            placements = PLACEMENT_MASKS[self.hand[hand_index]]
            offset = hand_index * BOARD_CELLS
            for position, placement in enumerate(placements):
                if placement and (self.board_mask & placement) == 0:
                    mask[offset + position] = 1
        return mask

    def valid_action_count(self):
        count = 0
        for hand_index, is_available in enumerate(self.available):
            if not is_available:
                continue
            for placement in PLACEMENT_MASKS[self.hand[hand_index]]:
                if placement and (self.board_mask & placement) == 0:
                    count += 1
        return count

    def check_game_over(self):
        return self.valid_action_count() == 0

    def step(self, action):
        hand_index = action // BOARD_CELLS
        position = action % BOARD_CELLS

        if hand_index < 0 or hand_index >= self.hand_size or not self.can_place(hand_index, position):
            return False, False, 0, [], [], False

        shape_key = self.hand[hand_index]
        self.board_mask |= PLACEMENT_MASKS[shape_key][position]
        self.available[hand_index] = False
        self.blocks_placed += 1

        lines_cleared, full_rows, full_cols = self.clear_lines()
        stage_completed = False

        if not any(self.available):
            self.stages_passed += 1
            stage_completed = True
            self.get_new_hand()

        is_game_over = self.check_game_over()
        return True, is_game_over, lines_cleared, full_rows, full_cols, stage_completed

    def clear_lines(self):
        full_rows = [row for row, row_mask in enumerate(ROW_MASKS) if (self.board_mask & row_mask) == row_mask]
        full_cols = [col for col, col_mask in enumerate(COL_MASKS) if (self.board_mask & col_mask) == col_mask]
        clear_mask = 0
        for row in full_rows:
            clear_mask |= ROW_MASKS[row]
        for col in full_cols:
            clear_mask |= COL_MASKS[col]
        if clear_mask:
            self.board_mask &= BOARD_MASK ^ clear_mask
            self.lines_destroyed += len(full_rows) + len(full_cols)
        return len(full_rows) + len(full_cols), full_rows, full_cols

    def board_array(self):
        data = np.fromiter(((self.board_mask >> index) & 1 for index in range(BOARD_CELLS)), dtype=np.int8)
        return data.reshape(GRID_SIZE, GRID_SIZE)

    def hand_arrays(self):
        return [SHAPES[key] for key in self.hand]

    def filled_cells(self):
        return self.board_mask.bit_count()

    def line_potential(self):
        score = 0.0
        for mask in ROW_MASKS:
            filled = (self.board_mask & mask).bit_count()
            if filled >= 3:
                score += (filled / GRID_SIZE) ** 3
        for mask in COL_MASKS:
            filled = (self.board_mask & mask).bit_count()
            if filled >= 3:
                score += (filled / GRID_SIZE) ** 3
        return score

    def single_cell_pockets(self):
        pockets = 0
        empty_mask = BOARD_MASK ^ self.board_mask
        for position in range(BOARD_CELLS):
            bit = 1 << position
            if not (empty_mask & bit):
                continue
            row = position // GRID_SIZE
            col = position % GRID_SIZE
            has_empty_neighbor = False
            if row > 0 and (empty_mask & (1 << (position - GRID_SIZE))):
                has_empty_neighbor = True
            if row < GRID_SIZE - 1 and (empty_mask & (1 << (position + GRID_SIZE))):
                has_empty_neighbor = True
            if col > 0 and (empty_mask & (1 << (position - 1))):
                has_empty_neighbor = True
            if col < GRID_SIZE - 1 and (empty_mask & (1 << (position + 1))):
                has_empty_neighbor = True
            if not has_empty_neighbor:
                pockets += 1
        return pockets
