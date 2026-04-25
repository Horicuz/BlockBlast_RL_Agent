import numpy as np

SHAPES = {
    # ==========================================
    # 2 BLOCKS
    # ==========================================
    # Diagonals
    "diag_2_1": np.array([[1, 0], 
                          [0, 1]], dtype=int),
    "diag_2_2": np.array([[0, 1], 
                          [1, 0]], dtype=int),
    # Lines
    "line_1x2": np.array([[1, 1]], dtype=int),
    "line_2x1": np.array([[1], 
                          [1]], dtype=int),

    # ==========================================
    # 3 BLOCKS
    # ==========================================
    # Lines
    "line_1x3": np.array([[1, 1, 1]], dtype=int),
    "line_3x1": np.array([[1], 
                          [1], 
                          [1]], dtype=int),
    # Small Corners
    "corner_3_ul": np.array([[1, 1], 
                             [1, 0]], dtype=int),
    "corner_3_ur": np.array([[1, 1], 
                             [0, 1]], dtype=int),
    "corner_3_dl": np.array([[1, 0], 
                             [1, 1]], dtype=int),
    "corner_3_dr": np.array([[0, 1], 
                             [1, 1]], dtype=int),

    # ==========================================
    # 4 BLOCKS
    # ==========================================
    # 2x2 Cube
    "square_2x2": np.array([[1, 1], 
                            [1, 1]], dtype=int),
    # T-Shapes
    "T_4_up": np.array([[0, 1, 0], 
                        [1, 1, 1]], dtype=int),
    "T_4_down": np.array([[1, 1, 1], 
                          [0, 1, 0]], dtype=int),
    "T_4_left": np.array([[0, 1], 
                          [1, 1], 
                          [0, 1]], dtype=int),
    "T_4_right": np.array([[1, 0], 
                           [1, 1], 
                           [1, 0]], dtype=int),
    # S-Shapes (and Z-Shapes)
    "S_4_h": np.array([[0, 1, 1], 
                       [1, 1, 0]], dtype=int),
    "S_4_v": np.array([[1, 0], 
                       [1, 1], 
                       [0, 1]], dtype=int),
    "Z_4_h": np.array([[1, 1, 0], 
                       [0, 1, 1]], dtype=int),
    "Z_4_v": np.array([[0, 1], 
                       [1, 1], 
                       [1, 0]], dtype=int),
    # L-Shapes (4 standard, 4 reversed/J-shapes)
    "L_4_1": np.array([[1, 0], 
                       [1, 0], 
                       [1, 1]], dtype=int),
    "L_4_2": np.array([[1, 1, 1], 
                       [1, 0, 0]], dtype=int),
    "L_4_3": np.array([[1, 1], 
                       [0, 1], 
                       [0, 1]], dtype=int),
    "L_4_4": np.array([[0, 0, 1], 
                       [1, 1, 1]], dtype=int),
    "J_4_1": np.array([[0, 1], 
                       [0, 1], 
                       [1, 1]], dtype=int),
    "J_4_2": np.array([[1, 0, 0], 
                       [1, 1, 1]], dtype=int),
    "J_4_3": np.array([[1, 1], 
                       [1, 0], 
                       [1, 0]], dtype=int),
    "J_4_4": np.array([[1, 1, 1], 
                       [0, 0, 1]], dtype=int),

    # ==========================================
    # 5 BLOCKS
    # ==========================================
    # Big L-Corners (3x3 footprint)
    "big_L_ul": np.array([[1, 1, 1], 
                          [1, 0, 0], 
                          [1, 0, 0]], dtype=int),
    "big_L_ur": np.array([[1, 1, 1], 
                          [0, 0, 1], 
                          [0, 0, 1]], dtype=int),
    "big_L_dl": np.array([[1, 0, 0], 
                          [1, 0, 0], 
                          [1, 1, 1]], dtype=int),
    "big_L_dr": np.array([[0, 0, 1], 
                          [0, 0, 1], 
                          [1, 1, 1]], dtype=int),
    # Big T-Shapes (3x3 footprint)
    "big_T_up": np.array([[1, 1, 1], 
                          [0, 1, 0], 
                          [0, 1, 0]], dtype=int),
    "big_T_down": np.array([[0, 1, 0], 
                            [0, 1, 0], 
                            [1, 1, 1]], dtype=int),
    "big_T_left": np.array([[1, 0, 0], 
                            [1, 1, 1], 
                            [1, 0, 0]], dtype=int),
    "big_T_right": np.array([[0, 0, 1], 
                             [1, 1, 1], 
                             [0, 0, 1]], dtype=int),

    # ==========================================
    # 6 BLOCKS
    # ==========================================
    # 2x3 and 3x2 Rectangles
    "rect_2x3": np.array([[1, 1, 1], 
                          [1, 1, 1]], dtype=int),
    "rect_3x2": np.array([[1, 1], 
                          [1, 1], 
                          [1, 1]], dtype=int),

    # ==========================================
    # 9 BLOCKS
    # ==========================================
    # 3x3 Cube
    "square_3x3": np.array([[1, 1, 1], 
                            [1, 1, 1], 
                            [1, 1, 1]], dtype=int)
}

SIZE_CATEGORIES = {
    "small": [
        "diag_2_1", "diag_2_2", 
        "line_1x2", "line_2x1", "line_1x3", "line_3x1", 
        "corner_3_ul", "corner_3_ur", "corner_3_dl", "corner_3_dr"
    ],
    "medium": [
        "square_2x2", 
        "T_4_up", "T_4_down", "T_4_left", "T_4_right", 
        "S_4_h", "S_4_v", "Z_4_h", "Z_4_v", 
        "L_4_1", "L_4_2", "L_4_3", "L_4_4", "J_4_1", "J_4_2", "J_4_3", "J_4_4"
    ],
    "large": [
        "big_L_ul", "big_L_ur", "big_L_dl", "big_L_dr", 
        "big_T_up", "big_T_down", "big_T_left", "big_T_right", 
        "rect_2x3", "rect_3x2", "square_3x3"
    ]
}

# COMPLEXITATE (Complexity)
COMPLEXITY_CATEGORIES = {
    "simple": [ # Linii și dreptunghiuri perfecte
        "line_1x2", "line_2x1", "line_1x3", "line_3x1", 
        "square_2x2", "rect_2x3", "rect_3x2", "square_3x3"
    ],
    "medium": [ # Colțuri și L-uri
        "corner_3_ul", "corner_3_ur", "corner_3_dl", "corner_3_dr", 
        "L_4_1", "L_4_2", "L_4_3", "L_4_4", "J_4_1", "J_4_2", "J_4_3", "J_4_4",
        "big_L_ul", "big_L_ur", "big_L_dl", "big_L_dr"
    ],
    "hard": [ # S, Z, T, Diagonale
        "diag_2_1", "diag_2_2", 
        "T_4_up", "T_4_down", "T_4_left", "T_4_right", 
        "S_4_h", "S_4_v", "Z_4_h", "Z_4_v", 
        "big_T_up", "big_T_down", "big_T_left", "big_T_right"
    ]
}