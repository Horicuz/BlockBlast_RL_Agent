import numpy as np

SHAPES = {
    # 2 cells: diagonals and short lines.
    "diag_2_1": np.array([[1, 0], 
                          [0, 1]], dtype=int),
    "diag_2_2": np.array([[0, 1], 
                          [1, 0]], dtype=int),
    "line_1x2": np.array([[1, 1]], dtype=int),
    "line_2x1": np.array([[1], 
                          [1]], dtype=int),

    # 3 cells: lines and small corners.
    "line_1x3": np.array([[1, 1, 1]], dtype=int),
    "line_3x1": np.array([[1], 
                          [1], 
                          [1]], dtype=int),
    "corner_3_ul": np.array([[1, 1], 
                             [1, 0]], dtype=int),
    "corner_3_ur": np.array([[1, 1], 
                             [0, 1]], dtype=int),
    "corner_3_dl": np.array([[1, 0], 
                             [1, 1]], dtype=int),
    "corner_3_dr": np.array([[0, 1], 
                             [1, 1]], dtype=int),

    # 4 cells: square, T, S/Z and L/J shapes.
    "square_2x2": np.array([[1, 1], 
                            [1, 1]], dtype=int),
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

    # 5 cells: large L and T variants.
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

    # 6 and 9 cells: rectangles and 3x3 square.
    "rect_2x3": np.array([[1, 1, 1], 
                          [1, 1, 1]], dtype=int),
    "rect_3x2": np.array([[1, 1], 
                          [1, 1], 
                          [1, 1]], dtype=int),
    "square_3x3": np.array([[1, 1, 1], 
                            [1, 1, 1], 
                            [1, 1, 1]], dtype=int)
}

SIZE_CATEGORIES = {
    "small": [
        "dot",
        "diag_2_1", "diag_2_2",
        "line_1x2", "line_2x1",
        "line_1x3", "line_3x1",
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
        "rect_2x3", "rect_3x2",
        "square_3x3"
    ]
}

COMPLEXITY_CATEGORIES = {
    "simple": [
        "dot", "line_1x2", "line_2x1", "line_1x3", "line_3x1", 
        "square_2x2", "square_3x3", "rect_2x3", "rect_3x2"
    ],
    "medium": [
        "corner_3_ul", "corner_3_ur", "corner_3_dl", "corner_3_dr", 
        "L_4_1", "L_4_2", "L_4_3", "L_4_4", "J_4_1", "J_4_2", "J_4_3", "J_4_4",
        "big_L_ul", "big_L_ur", "big_L_dl", "big_L_dr"
    ],
    "hard": [
        "diag_2_1", "diag_2_2",
        "T_4_up", "T_4_down", "T_4_left", "T_4_right", 
        "S_4_h", "S_4_v", "Z_4_h", "Z_4_v", 
        "big_T_up", "big_T_down", "big_T_left", "big_T_right"
    ]
}

TRAINING_POOLS = {
    "all": list(SHAPES.keys()),
    "simple": COMPLEXITY_CATEGORIES["simple"],
    "simple_small": [
        "line_1x2", "line_2x1", "line_1x3", "line_3x1",
        "square_2x2",
    ],
    "small": SIZE_CATEGORIES["small"],
    "medium_size": SIZE_CATEGORIES["medium"],
    "large": SIZE_CATEGORIES["large"],
    "medium_complexity": COMPLEXITY_CATEGORIES["medium"],
    "hard": COMPLEXITY_CATEGORIES["hard"],
}

SIZE_BY_SHAPE = {}
for _size_name, _shape_keys in SIZE_CATEGORIES.items():
    for _shape_key in _shape_keys:
        SIZE_BY_SHAPE[_shape_key] = _size_name

COMPLEXITY_BY_SHAPE = {}
for _complexity_name, _shape_keys in COMPLEXITY_CATEGORIES.items():
    for _shape_key in _shape_keys:
        COMPLEXITY_BY_SHAPE[_shape_key] = _complexity_name

for _shape_key in SHAPES:
    SIZE_BY_SHAPE.setdefault(_shape_key, "medium")
    COMPLEXITY_BY_SHAPE.setdefault(_shape_key, "medium")


MINI_SHAPES = {
    "dot": np.array([[1]], dtype=int),
    "line_1x2": np.array([[1, 1]], dtype=int),
    "line_2x1": np.array([[1], [1]], dtype=int),
    "line_1x3": np.array([[1, 1, 1]], dtype=int),
    "line_3x1": np.array([[1], [1], [1]], dtype=int),
    "diag_ul": np.array([[1, 0], [0, 1]], dtype=int),
    "diag_ur": np.array([[0, 1], [1, 0]], dtype=int),
    "corner_ul": np.array([[1, 1], [1, 0]], dtype=int),
    "corner_ur": np.array([[1, 1], [0, 1]], dtype=int),
    "corner_dl": np.array([[1, 0], [1, 1]], dtype=int),
    "corner_dr": np.array([[0, 1], [1, 1]], dtype=int),
    "square_2x2": np.array([[1, 1], [1, 1]], dtype=int),
    "square_3x3": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    "L_4_1": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    "L_4_2": np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    "L_4_3": np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
    "L_4_4": np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    "J_4_1": np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    "J_4_2": np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    "J_4_3": np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    "J_4_4": np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
}

for _shape_key in MINI_SHAPES:
    _cell_count = int(np.count_nonzero(MINI_SHAPES[_shape_key]))
    if _cell_count <= 3:
        SIZE_BY_SHAPE[_shape_key] = "small"
    elif _cell_count == 4:
        SIZE_BY_SHAPE[_shape_key] = "medium"
    else:
        SIZE_BY_SHAPE[_shape_key] = "large"

    if _shape_key in {"diag_ul", "diag_ur"}:
        COMPLEXITY_BY_SHAPE[_shape_key] = "hard"
    elif _shape_key in {
        "corner_ul", "corner_ur", "corner_dl", "corner_dr",
        "L_4_1", "L_4_2", "L_4_3", "L_4_4", "J_4_1", "J_4_2", "J_4_3", "J_4_4",
    }:
        COMPLEXITY_BY_SHAPE[_shape_key] = "medium"
    else:
        COMPLEXITY_BY_SHAPE[_shape_key] = "simple"

TRAINING_POOLS["mini_small"] = [
    "dot", "line_1x2", "line_2x1", "line_1x3", "line_3x1",
    "diag_ul", "diag_ur", "corner_ul", "corner_ur", "corner_dl", "corner_dr",
]
TRAINING_POOLS["mini_large"] = [
    "square_2x2", "square_3x3",
    "L_4_1", "L_4_2", "L_4_3", "L_4_4", "J_4_1", "J_4_2", "J_4_3", "J_4_4",
]
TRAINING_POOLS["mini"] = list(MINI_SHAPES.keys())

SHAPE_LIBRARY = {**SHAPES, **MINI_SHAPES}
