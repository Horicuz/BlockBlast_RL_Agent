import numpy as np
from blocks import SHAPES
from engine import BlockBlastLogic

# 1. Initialize our game logic
game = BlockBlastLogic()

# 2. Recreate the exact grid from your screenshot (1 = blue, 0 = empty)
# Carefully transcribed row by row from top to bottom
game.grid = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0]
])

# 3. Get the blocks from the bottom of your screenshot
block_1 = SHAPES["square_3x3"]
block_2 = SHAPES["line_1x2"]
block_3 = SHAPES["square_3x3"]

print("--- Testing the 3x3 Square ---")
# Let's see if there is anywhere we can put the 3x3 square right now
possible_3x3_moves = 0
for r in range(8):
    for c in range(8):
        if game.can_place(block_1, r, c):
            print(f"Can place 3x3 at Row {r}, Col {c}")
            possible_3x3_moves += 1

if possible_3x3_moves == 0:
    print("Uh oh! The 3x3 block cannot be placed anywhere on this board.")
    print("This means placing the 1x2 block FIRST to clear a line is absolutely mandatory to survive.")

print("\n--- Testing the 1x2 Line ---")
# Let's see where we can put the 1x2 line
for r in range(8):
    for c in range(8):
        if game.can_place(block_2, r, c):
            print(f"Can place 1x2 at Row {r}, Col {c}")