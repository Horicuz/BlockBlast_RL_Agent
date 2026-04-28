import argparse
import os
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from blocks import SHAPE_LIBRARY, TRAINING_POOLS
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

DEFAULT_BOARD_SIZE = 8
GRID_SIZE = DEFAULT_BOARD_SIZE
HAND_SIZE = 3
BOARD_CELLS = GRID_SIZE * GRID_SIZE

WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 930

FPS = 60
AI_STEP_INTERVAL = 0.35

SINGLE_CELL = 60
SINGLE_BOARD_X = 88
SINGLE_BOARD_Y = 130
SINGLE_BOARD_SIZE = GRID_SIZE * SINGLE_CELL

DUAL_CELL = 52
DUAL_BOARD_X_LEFT = 86
DUAL_BOARD_X_RIGHT = 748
DUAL_BOARD_Y = 126
DUAL_BOARD_SIZE = GRID_SIZE * DUAL_CELL

HAND_SLOT_W = 130
HAND_SLOT_H = 130
HAND_SLOT_GAP = 20

MINI_CELL = 22
HAND_BLOCK_OFFSET_X = 24
HAND_BLOCK_OFFSET_Y = 24

MENU_BTN_W = 440
MENU_BTN_H = 78
MENU_BTN_GAP = 18

BG = (11, 18, 34)
PANEL = (22, 34, 62)
PANEL_2 = (30, 45, 79)
BOARD_BG = (18, 29, 52)
CELL_BG = (40, 54, 85)
CELL_FILL = (98, 205, 255)
CELL_SHADOW = (42, 130, 188)
TEXT = (236, 243, 255)
MUTED = (162, 178, 204)
DIM = (97, 110, 142)
WHITE = (255, 255, 255)
HUMAN = (255, 187, 85)
AI = (112, 218, 164)
WARN = (255, 203, 100)
BAD = (255, 111, 133)
GOOD = (88, 231, 157)
GHOST_VALID = (106, 220, 149)
GHOST_INVALID = (233, 104, 126)

WATCH_DEFAULTS = {
    4: {
        "model_path": "checkpoints/4x4_best8x8/4x4_best8x8_basecnn_g03_lr2e3/block_blast_4x4_best8x8_basecnn_v1.zip",
        "shape_pool": "mini",
        "hand_generator": "adaptive_playable",
    },
    8: {
        "model_path": "checkpoints/cnn_contact_threshold/basecnn_contact_threshold_gamma06_lr8e4_lines28/block_blast_basecnn_contact_threshold_gamma06_v1.zip",
        "shape_pool": "all",
        "hand_generator": "adaptive_playable",
    },
}


def configure_board_mode(board_size):
    global GRID_SIZE, BOARD_CELLS, SINGLE_BOARD_SIZE, DUAL_BOARD_SIZE
    GRID_SIZE = int(board_size)
    BOARD_CELLS = GRID_SIZE * GRID_SIZE
    SINGLE_BOARD_SIZE = GRID_SIZE * SINGLE_CELL
    DUAL_BOARD_SIZE = GRID_SIZE * DUAL_CELL


def resolve_watch_defaults(args):
    defaults = WATCH_DEFAULTS.get(int(args.board_size), WATCH_DEFAULTS[DEFAULT_BOARD_SIZE])
    if getattr(args, "model_path", None) is None:
        args.model_path = defaults["model_path"]
    if getattr(args, "shape_pool", None) is None:
        args.shape_pool = defaults["shape_pool"]
    if getattr(args, "hand_generator", None) is None:
        args.hand_generator = defaults["hand_generator"]
    return args

SHAPE_KEYS = list(SHAPE_LIBRARY.keys())

WATCH_REWARD_CONFIG = {
    "placement_reward": 0.0,
    "line_clear_scale": 28.0,
    "line_clear_bonus": 1.5,
    "stage_complete_reward": 0.0,
    "no_line_penalty": 0.0,
    "game_over_penalty": 90.0,
    "game_over_early_weight": 0.0,
    "contact_reward_scale": 24.0,
    "contact_reward_power": 1.15,
    "contact_reward_threshold": 0.40,
    "contact_penalty_scale": 8.0,
    "complexity_simple_prob": 0.78,
    "complexity_medium_prob": 0.18,
    "complexity_hard_prob": 0.04,
}

MODEL_METRICS = {
    "checkpoints/4x4_best8x8/4x4_best8x8_basecnn_g03_lr2e3/block_blast_4x4_best8x8_basecnn_v1.zip": ("2.5", "11"),
    "checkpoints/cnn_contact_threshold/basecnn_contact_threshold_gamma06_lr8e4_lines28/block_blast_basecnn_contact_threshold_gamma06_v1.zip": ("20.2", "51"),
    "checkpoints/cnn_contact_threshold/short_g03_lr2e3/block_blast_basecnn_contact_threshold_short_v1.zip": ("19.9", "60"),
    "checkpoints/cnn_contact_threshold/short_g05_lr2e2/block_blast_basecnn_contact_threshold_short_v1.zip": ("13.8", "n/a"),
    "checkpoints/cnn_holes/block_blast_cnn_holes_v1.zip": ("n/a", "n/a"),
    "checkpoints/cnn_immediate_contact/final_softgen_gamma04_lr2e4_lines42_contact18/block_blast_cnn_immediate_contact_gamma04_lines_v1.zip": ("n/a", "n/a"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Block Blast visual arena (PySide6)")
    parser.add_argument("--board-size", type=int, choices=[4, 8], default=DEFAULT_BOARD_SIZE, help="Select the board size to run")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model path with or without .zip. If omitted, a default for the selected board size is used.",
    )
    parser.add_argument("--no-model", action="store_true", help="Start without loading a PPO model; heuristic mode still works")
    parser.add_argument("--ai-interval", type=float, default=AI_STEP_INTERVAL)
    parser.add_argument("--shape-pool", choices=sorted(TRAINING_POOLS.keys()), default=None)
    parser.add_argument("--hand-generator", choices=["random", "playable", "adaptive_playable", "solvable"], default=None)
    parser.add_argument("--fixed-game-seed", type=int, default=None)
    return parser.parse_args()


def model_display_name(path):
    parts = path.split(os.sep)
    if len(parts) >= 3 and parts[-2] != "checkpoints":
        return parts[-2]
    return os.path.splitext(os.path.basename(path))[0]


def infer_model_config(path):
    normalized = path.replace("\\", "/")
    board_size = 4 if ("4x4" in normalized or "/bc/" in normalized) else 8
    shape_pool = "mini" if board_size == 4 else "all"
    return {
        "path": path,
        "board_size": board_size,
        "shape_pool": shape_pool,
        "hand_generator": "adaptive_playable",
    }


def discover_models():
    models = []
    for root, _dirs, files in os.walk("checkpoints"):
        for file_name in files:
            if file_name.endswith(".zip"):
                path = os.path.join(root, file_name)
                config = infer_model_config(path)
                mean_stage, max_stage = MODEL_METRICS.get(path.replace("\\", "/"), ("n/a", "n/a"))
                config["mean_stage"] = mean_stage
                config["max_stage"] = max_stage
                config["name"] = model_display_name(path)
                models.append(config)
    models.sort(key=lambda item: (item["board_size"], item["name"], item["path"]))
    return models


def mask_fn(env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()

    raise AttributeError("valid_action_mask() is not available on the current environment")


def decode_action(action):
    actions_per_hand = GRID_SIZE * GRID_SIZE
    hand_idx = action // actions_per_hand
    remainder = action % actions_per_hand
    row = remainder // GRID_SIZE
    col = remainder % GRID_SIZE
    return hand_idx, row, col


def action_from_selection(hand_index, row, col):
    return (hand_index * (GRID_SIZE * GRID_SIZE)) + (row * GRID_SIZE) + col


def shape_key_from_matrix(matrix):
    for key in SHAPE_KEYS:
        shape = SHAPE_LIBRARY[key]
        if shape.shape == matrix.shape and np.array_equal(shape, matrix):
            return key
    return SHAPE_KEYS[0]


def capture_game_state(game):
    return {
        "grid": game.grid.copy(),
        "hand": [block.copy() for block in game.hand],
        "available": list(game.available),
        "stages_passed": game.stages_passed,
        "lines_destroyed": game.lines_destroyed,
        "blocks_placed": game.blocks_placed,
        "rng_state": game.rng.getstate(),
    }


def restore_game_state(game, game_state):
    game.grid = game_state["grid"].copy()
    game.hand = [block.copy() for block in game_state["hand"]]
    game.available = list(game_state["available"])
    game.stages_passed = game_state["stages_passed"]
    game.lines_destroyed = game_state["lines_destroyed"]
    game.blocks_placed = game_state["blocks_placed"]
    if "rng_state" in game_state:
        game.rng.setstate(game_state["rng_state"])


def create_env_bundle(args=None):
    board_size = getattr(args, "board_size", DEFAULT_BOARD_SIZE)
    raw_env = BlockBlastEnv(
        reward_config=WATCH_REWARD_CONFIG,
        apply_hole_penalty=False,
        fixed_game_seed=getattr(args, "fixed_game_seed", None),
        shape_pool=getattr(args, "shape_pool", WATCH_DEFAULTS[board_size]["shape_pool"]),
        hand_generator=getattr(args, "hand_generator", WATCH_DEFAULTS[board_size]["hand_generator"]),
        board_size=board_size,
    )
    env = ActionMasker(raw_env, mask_fn)
    obs, _ = env.reset()
    return raw_env, env, obs


def load_model(env, model_path):
    normalized_path = model_path[:-4] if model_path.endswith(".zip") else model_path
    if not os.path.exists(normalized_path + ".zip"):
        raise FileNotFoundError(normalized_path + ".zip")

    try:
        return MaskablePPO.load(normalized_path, env=env)
    except ValueError as exc:
        raise RuntimeError(
            "Failed to load model due to env/model mismatch. "
            "Use a CNN checkpoint like checkpoints/cnn_noholes/block_blast_cnn_noholes_v1. "
            f"Original error: {exc}"
        ) from exc


def to_color(rgb, alpha=255):
    return QColor(rgb[0], rgb[1], rgb[2], alpha)


def choose_font_family():
    preferred = [
        "Avenir Next",
        "SF Pro Text",
        "Helvetica Neue",
        "Noto Sans",
        "Segoe UI",
        "Ubuntu",
    ]
    families = set(QFontDatabase.families())
    for family in preferred:
        if family in families:
            return family

    app = QApplication.instance()
    if app is not None:
        return app.font().family()
    return "Sans Serif"


def point_to_cell(mouse_pos, board_x, board_y, cell_size):
    mx, my = mouse_pos
    if board_x <= mx < board_x + GRID_SIZE * cell_size and board_y <= my < board_y + GRID_SIZE * cell_size:
        col = int((mx - board_x) // cell_size)
        row = int((my - board_y) // cell_size)
        return row, col
    return None


def compute_block_cells(block_shape, anchor_row, anchor_col):
    cells = []
    for r in range(block_shape.shape[0]):
        for c in range(block_shape.shape[1]):
            if block_shape[r, c] == 1:
                cells.append((anchor_row + r, anchor_col + c))
    return cells


def get_hand_slot_rects(start_x, start_y):
    rects = []
    for i in range(HAND_SIZE):
        rects.append(QRect(start_x + i * (HAND_SLOT_W + HAND_SLOT_GAP), start_y, HAND_SLOT_W, HAND_SLOT_H))
    return rects


def make_button(rect, label, accent, key=None):
    button = {"rect": rect, "label": label, "accent": accent}
    if key is not None:
        button["key"] = key
    return button


def ai_step(model, obs, raw_env, env):
    if model is None:
        return obs, True, "No PPO model is loaded. Use heuristic mode or start without --no-model."

    action_masks = mask_fn(raw_env)
    if not np.any(action_masks):
        return obs, True, "No valid AI moves available."

    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
    action = int(action)
    hand_idx, row, col = decode_action(action)

    next_obs, reward, done, _, _ = env.step(action)
    msg = f"AI placed slot {hand_idx + 1} at R{row} C{col} | reward {reward:.1f}"
    return next_obs, done, msg


def can_place_on_grid(grid, block, row, col):
    block_h, block_w = block.shape
    if row + block_h > GRID_SIZE or col + block_w > GRID_SIZE:
        return False
    target = grid[row:row + block_h, col:col + block_w]
    return not np.any(np.logical_and(target, block))


def valid_actions_for_state(grid, hand, available):
    actions = []
    for hand_idx, is_available in enumerate(available):
        if not is_available:
            continue
        block = hand[hand_idx]
        block_h, block_w = block.shape
        for row in range(GRID_SIZE - block_h + 1):
            for col in range(GRID_SIZE - block_w + 1):
                if can_place_on_grid(grid, block, row, col):
                    actions.append(action_from_selection(hand_idx, row, col))
    return actions


def apply_action_to_grid(grid, block, row, col):
    next_grid = grid.copy()
    block_h, block_w = block.shape
    next_grid[row:row + block_h, col:col + block_w] += block

    full_rows = list(np.where(np.all(next_grid == 1, axis=1))[0])
    full_cols = list(np.where(np.all(next_grid == 1, axis=0))[0])
    if full_rows:
        next_grid[full_rows, :] = 0
    if full_cols:
        next_grid[:, full_cols] = 0

    return next_grid, len(full_rows) + len(full_cols)


def contact_ratio_on_grid(previous_grid, block, row, col):
    touching_edges = 0
    external_edges = 0
    block_h, block_w = block.shape

    for block_row in range(block_h):
        for block_col in range(block_w):
            if block[block_row, block_col] == 0:
                continue

            board_row = row + block_row
            board_col = col + block_col
            for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_block_row = block_row + d_row
                neighbor_block_col = block_col + d_col
                if (
                    0 <= neighbor_block_row < block_h
                    and 0 <= neighbor_block_col < block_w
                    and block[neighbor_block_row, neighbor_block_col] == 1
                ):
                    continue

                external_edges += 1
                neighbor_row = board_row + d_row
                neighbor_col = board_col + d_col
                if (
                    neighbor_row < 0
                    or neighbor_row >= GRID_SIZE
                    or neighbor_col < 0
                    or neighbor_col >= GRID_SIZE
                    or previous_grid[neighbor_row, neighbor_col] == 1
                ):
                    touching_edges += 1

    return 0.0 if external_edges == 0 else touching_edges / external_edges


def empty_region_stats(grid):
    visited = np.zeros_like(grid, dtype=bool)
    largest_region = 0
    region_count = 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row, col] != 0 or visited[row, col]:
                continue

            queue = [(row, col)]
            visited[row, col] = True
            region_size = 0
            while queue:
                current_row, current_col = queue.pop()
                region_size += 1
                for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if (
                        0 <= next_row < GRID_SIZE
                        and 0 <= next_col < GRID_SIZE
                        and not visited[next_row, next_col]
                        and grid[next_row, next_col] == 0
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))

            region_count += 1
            largest_region = max(largest_region, region_size)

    return largest_region, region_count


def heuristic_board_score(grid, hand, available):
    valid_actions_after = len(valid_actions_for_state(grid, hand, available))
    rows = grid.sum(axis=1)
    cols = grid.sum(axis=0)
    line_potential = float(np.sum((rows / GRID_SIZE) ** 3) + np.sum((cols / GRID_SIZE) ** 3))
    largest_empty_region, empty_regions = empty_region_stats(grid)
    filled_cells = int(grid.sum())

    return (
        valid_actions_after * 0.65
        + line_potential * 5.0
        + largest_empty_region * 0.25
        - empty_regions * 0.5
        - max(filled_cells - int(BOARD_CELLS * 0.6875), 0) * 0.4
    )


def heuristic_evaluate_action(grid, hand, available, action):
    hand_idx, row, col = decode_action(action)
    block = hand[hand_idx]
    next_grid, lines_cleared = apply_action_to_grid(grid, block, row, col)
    next_available = list(available)
    next_available[hand_idx] = False
    stage_completed = not any(next_available)

    contact_ratio = contact_ratio_on_grid(grid, block, row, col)
    contact_bonus = max(contact_ratio - 0.40, 0.0) * 28.0
    contact_penalty = max(0.40 - contact_ratio, 0.0) * 18.0
    valid_actions_after = len(valid_actions_for_state(next_grid, hand, next_available))
    no_move_penalty = 160.0 if (not stage_completed and valid_actions_after == 0) else 0.0

    score = (
        lines_cleared * 60.0
        + (lines_cleared ** 2) * 25.0
        + contact_bonus
        - contact_penalty
        + heuristic_board_score(next_grid, hand, next_available)
        + (8.0 if stage_completed else 0.0)
        - no_move_penalty
    )

    return {
        "score": score,
        "lines_cleared": lines_cleared,
        "contact_ratio": contact_ratio,
        "valid_actions_after": valid_actions_after,
    }


def choose_heuristic_action(raw_env):
    grid = raw_env.game.grid
    hand = raw_env.game.hand
    available = raw_env.game.available
    actions = valid_actions_for_state(grid, hand, available)
    if not actions:
        return None, None

    ranked = []
    for action in actions:
        result = heuristic_evaluate_action(grid, hand, available, action)
        ranked.append((result["score"], action, result))

    ranked.sort(reverse=True, key=lambda item: item[0])
    best_score, best_action, best_result = ranked[0]
    best_result["score"] = best_score
    return best_action, best_result


def heuristic_step(obs, raw_env, env):
    action, decision = choose_heuristic_action(raw_env)
    if action is None:
        return obs, True, "No valid heuristic moves available."

    hand_idx, row, col = decode_action(action)
    next_obs, reward, done, _, _ = env.step(action)
    msg = (
        f"Heuristic placed slot {hand_idx + 1} at R{row} C{col} | "
        f"score {decision['score']:.1f} | contact {decision['contact_ratio']:.2f} | reward {reward:.1f}"
    )
    return next_obs, done, msg


class ArenaWidget(QWidget):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        self.setWindowTitle("Block Blast Arena - PySide6")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMouseTracking(True)

        font_family = choose_font_family()
        self.title_font = QFont(font_family, 30, QFont.Bold)
        self.subtitle_font = QFont(font_family, 20, QFont.Bold)
        self.body_font = QFont(font_family, 14)
        self.button_font = QFont(font_family, 16, QFont.Bold)

        self.scene = "menu"
        self.mouse_pos = (0, 0)
        self.reset_drag_state()

        self.last_error = ""
        self.model_catalog = discover_models()
        self.model_status = "Model loaded: " + model_display_name(args.model_path) if model is not None else "No PPO model loaded."
        self.model_combo = QComboBox(self)
        self.model_combo.setFont(QFont(font_family, 12))
        self.model_combo.setGeometry(840, 646, 700, 38)
        self.model_combo.currentIndexChanged.connect(self.handle_model_selection)
        self.populate_model_combo()

        self.init_menu_scene()

        self.loop_timer = QTimer(self)
        self.loop_timer.setInterval(int(1000 / FPS))
        self.loop_timer.timeout.connect(self.on_tick)
        self.loop_timer.start()

    def populate_model_combo(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("No PPO model (heuristic/manual only)", None)
        current_path = os.path.normpath(self.args.model_path) if getattr(self.args, "model_path", None) else ""
        selected_index = 0

        for model_info in self.model_catalog:
            label = (
                f"{model_info['name']} | {model_info['board_size']}x{model_info['board_size']} | "
                f"mean {model_info['mean_stage']} | max {model_info['max_stage']}"
            )
            self.model_combo.addItem(label, model_info)
            if os.path.normpath(model_info["path"]) == current_path:
                selected_index = self.model_combo.count() - 1

        self.model_combo.setCurrentIndex(selected_index)
        self.model_combo.blockSignals(False)

    def handle_model_selection(self, index):
        if index < 0:
            return

        model_info = self.model_combo.itemData(index)
        if model_info is None:
            self.model = None
            self.args.model_path = None
            self.model_status = "No PPO model loaded. Heuristic and manual modes are available."
            self.init_menu_scene()
            self.update()
            return

        self.args.model_path = model_info["path"]
        self.args.board_size = model_info["board_size"]
        self.args.shape_pool = model_info["shape_pool"]
        self.args.hand_generator = model_info["hand_generator"]
        configure_board_mode(self.args.board_size)

        try:
            _raw, env, _obs = create_env_bundle(self.args)
            self.model = load_model(env, self.args.model_path)
            self.last_error = ""
            self.model_status = (
                f"Loaded {model_info['name']} | board {self.args.board_size}x{self.args.board_size} | "
                f"mean {model_info['mean_stage']} | max {model_info['max_stage']}"
            )
        except Exception as exc:
            self.model = None
            self.last_error = f"Could not load selected model: {exc}"
            self.model_status = "Selected model failed to load. Heuristic/manual modes still work."
            traceback.print_exc()

        self.init_menu_scene()
        self.update()

    def init_menu_scene(self):
        self.scene = "menu"
        base_x = 118
        base_y = 232
        gap = MENU_BTN_H + MENU_BTN_GAP
        self.menu_buttons = [
            make_button(QRect(base_x, base_y + index * gap, MENU_BTN_W, MENU_BTN_H), label, accent, key)
            for index, (key, label, accent) in enumerate(
                [
                    ("solver", "Let AI solve your position", HUMAN),
                    ("watch", "Watch AI play a game", AI),
                    ("heuristic", "Watch heuristic play", WARN),
                    ("versus", "You vs the AI", (175, 121, 255)),
                ]
            )
        ]
        self.menu_status = "Choose AI, heuristic, or a manual comparison mode."

    def init_solver_scene(self):
        self.scene = "solver"
        self.solver_raw, self.solver_env, self.solver_obs = create_env_bundle(self.args)
        self.solver_edit_grid = self.solver_raw.game.grid.copy()
        self.solver_edit_hand_keys = [shape_key_from_matrix(block) for block in self.solver_raw.game.hand]
        self.solver_solving = False
        self.solver_done = False
        self.solver_status = "Edit board + hand, then click Start AI Solve."
        self.solver_last_step_time = time.time()
        self.solver_buttons = {
            "start": QRect(644, 240, 250, 52),
            "reset": QRect(916, 240, 250, 52),
            "back": QRect(1188, 240, 250, 52),
        }

    def init_watch_scene(self, policy="ai"):
        self.scene = "watch"
        self.watch_policy = policy
        self.watch_raw, self.watch_env, self.watch_obs = create_env_bundle(self.args)
        self.watch_done = False
        self.watch_status = "Heuristic is playing this random run." if policy == "heuristic" else "AI is playing this random run."
        self.watch_last_step_time = 0.0
        self.watch_buttons = {
            "new": QRect(642, 182, 270, 50),
            "back": QRect(642, 246, 270, 50),
        }

    def init_versus_scene(self):
        self.scene = "versus"
        self.vs_h_raw, self.vs_h_env, self.vs_h_obs = create_env_bundle(self.args)
        self.vs_a_raw, self.vs_a_env, self.vs_a_obs = create_env_bundle(self.args)

        initial_state = capture_game_state(self.vs_h_raw.game)
        restore_game_state(self.vs_a_raw.game, initial_state)
        self.vs_a_obs = self.vs_a_raw._get_obs()

        self.vs_h_done = False
        self.vs_a_done = False
        self.vs_locked = False
        self.vs_winner = None
        self.vs_status = "Drag a piece from your hand and drop it on your board."

        self.reset_drag_state()

        self.vs_buttons = {
            "new": QRect(1220, 182, 340, 50),
            "back": QRect(1220, 246, 340, 50),
        }

    def on_tick(self):
        try:
            now = time.time()

            if self.drag_active:
                vx, vy = self.drag_visual_pos
                tx, ty = self.drag_target_pos
                self.drag_visual_pos = (vx + (tx - vx) * 0.42, vy + (ty - vy) * 0.42)

            if self.scene == "solver" and self.solver_solving and (not self.solver_done):
                if now - self.solver_last_step_time >= self.args.ai_interval:
                    self.solver_obs, self.solver_done, msg = ai_step(
                        self.model,
                        self.solver_obs,
                        self.solver_raw,
                        self.solver_env,
                    )
                    self.solver_status = msg
                    self.solver_last_step_time = now
                    if self.solver_done:
                        self.solver_status = f"AI solve finished at stage {self.solver_raw.game.stages_passed}."

            if self.scene == "watch" and (not self.watch_done):
                if now - self.watch_last_step_time >= self.args.ai_interval:
                    if self.watch_policy == "heuristic":
                        self.watch_obs, self.watch_done, msg = heuristic_step(
                            self.watch_obs,
                            self.watch_raw,
                            self.watch_env,
                        )
                    else:
                        self.watch_obs, self.watch_done, msg = ai_step(
                            self.model,
                            self.watch_obs,
                            self.watch_raw,
                            self.watch_env,
                        )
                    self.watch_status = msg
                    self.watch_last_step_time = now
                    if self.watch_done:
                        actor = "Heuristic" if self.watch_policy == "heuristic" else "AI"
                        self.watch_status = f"{actor} game ended at stage {self.watch_raw.game.stages_passed}."

        except Exception as exc:
            self.last_error = f"Runtime error: {exc}"
            traceback.print_exc()
            if self.scene == "solver":
                self.solver_solving = False
                self.solver_done = True
                self.solver_status = self.last_error
            elif self.scene == "watch":
                self.watch_done = True
                self.watch_status = self.last_error
            elif self.scene == "versus":
                self.vs_locked = True
                self.vs_status = self.last_error

        self.update()

    def reset_drag_state(self):
        self.drag_active = False
        self.drag_slot = None
        self.drag_offset_cell = (0, 0)
        self.drag_visual_pos = (0.0, 0.0)
        self.drag_target_pos = (0.0, 0.0)

    def paintEvent(self, _event):
        self.model_combo.setVisible(self.scene == "menu")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_background(painter)

        if self.scene == "menu":
            self.draw_menu_scene(painter)
        elif self.scene == "solver":
            self.draw_solver_scene(painter)
        elif self.scene == "watch":
            self.draw_watch_scene(painter)
        elif self.scene == "versus":
            self.draw_versus_scene(painter)

        painter.end()

    def mouseMoveEvent(self, event):
        self.mouse_pos = (event.position().x(), event.position().y())
        if self.drag_active:
            self.drag_target_pos = self.mouse_pos

    def mousePressEvent(self, event):
        pos = (event.position().x(), event.position().y())

        try:
            if self.scene == "menu":
                self.handle_menu_click(pos)
                return

            if self.scene == "solver":
                self.handle_solver_click(pos, event.button())
                return

            if self.scene == "watch":
                self.handle_watch_click(pos, event.button())
                return

            if self.scene == "versus":
                self.handle_versus_mouse_press(pos, event.button())
                return
        except Exception as exc:
            self.last_error = f"Input error: {exc}"
            traceback.print_exc()
            if self.scene == "versus":
                self.vs_status = self.last_error

    def mouseReleaseEvent(self, event):
        if self.scene != "versus":
            return

        if event.button() != Qt.LeftButton:
            return

        if not self.drag_active or self.drag_slot is None:
            return

        try:
            mouse_pos = (event.position().x(), event.position().y())
            slot = self.drag_slot
            block = self.vs_h_raw.game.hand[slot]
            drop = point_to_cell(mouse_pos, DUAL_BOARD_X_LEFT, DUAL_BOARD_Y, DUAL_CELL)

            if drop is not None:
                drop_row, drop_col = drop
                anchor_row = drop_row - self.drag_offset_cell[0]
                anchor_col = drop_col - self.drag_offset_cell[1]
                self.try_human_drop(slot, anchor_row, anchor_col)
        finally:
            self.reset_drag_state()

    def handle_menu_click(self, pos):
        p = QPoint(int(pos[0]), int(pos[1]))
        for btn in self.menu_buttons:
            if btn["rect"].contains(p):
                if btn["key"] == "solver":
                    self.init_solver_scene()
                elif btn["key"] == "watch":
                    self.init_watch_scene()
                elif btn["key"] == "heuristic":
                    self.init_watch_scene(policy="heuristic")
                elif btn["key"] == "versus":
                    self.init_versus_scene()
                return

    def handle_solver_click(self, pos, button):
        p = QPoint(int(pos[0]), int(pos[1]))

        if button == Qt.LeftButton:
            if self.solver_buttons["start"].contains(p):
                self.apply_solver_setup()
                return
            if self.solver_buttons["reset"].contains(p):
                self.init_solver_scene()
                return
            if self.solver_buttons["back"].contains(p):
                self.init_menu_scene()
                return

        if self.solver_solving:
            return

        if button == Qt.LeftButton:
            cell = point_to_cell(pos, SINGLE_BOARD_X, SINGLE_BOARD_Y, SINGLE_CELL)
            if cell is not None:
                r, c = cell
                self.solver_edit_grid[r, c] = 0 if self.solver_edit_grid[r, c] == 1 else 1
                return

        slots = get_hand_slot_rects(SINGLE_BOARD_X + 4, SINGLE_BOARD_Y + SINGLE_BOARD_SIZE + 56)
        for i, rect in enumerate(slots):
            if rect.contains(p):
                current = self.solver_edit_hand_keys[i]
                idx = SHAPE_KEYS.index(current)
                if button == Qt.LeftButton:
                    self.solver_edit_hand_keys[i] = SHAPE_KEYS[(idx + 1) % len(SHAPE_KEYS)]
                elif button == Qt.RightButton:
                    self.solver_edit_hand_keys[i] = SHAPE_KEYS[(idx - 1) % len(SHAPE_KEYS)]
                return

    def handle_watch_click(self, pos, button):
        if button != Qt.LeftButton:
            return

        p = QPoint(int(pos[0]), int(pos[1]))
        if self.watch_buttons["new"].contains(p):
            self.init_watch_scene(policy=self.watch_policy)
            return
        if self.watch_buttons["back"].contains(p):
            self.init_menu_scene()
            return

    def handle_versus_mouse_press(self, pos, button):
        if button != Qt.LeftButton:
            return

        p = QPoint(int(pos[0]), int(pos[1]))

        if self.vs_buttons["new"].contains(p):
            self.init_versus_scene()
            return
        if self.vs_buttons["back"].contains(p):
            self.init_menu_scene()
            return

        if self.vs_locked or self.vs_h_done:
            return

        slots = get_hand_slot_rects(DUAL_BOARD_X_LEFT, DUAL_BOARD_Y + DUAL_BOARD_SIZE + 50)
        for idx, rect in enumerate(slots):
            if rect.contains(p) and self.vs_h_raw.game.available[idx]:
                block = self.vs_h_raw.game.hand[idx]
                offset = self.drag_offset_from_click(rect, block, pos)
                if offset is None:
                    self.vs_status = "Grab one highlighted cell of the shape to drag it."
                    return

                self.drag_active = True
                self.drag_slot = idx
                self.drag_offset_cell = offset
                self.drag_target_pos = pos
                self.drag_visual_pos = pos
                return

    def drag_offset_from_click(self, slot_rect, block, pos):
        lx = pos[0] - (slot_rect.x() + HAND_BLOCK_OFFSET_X)
        ly = pos[1] - (slot_rect.y() + HAND_BLOCK_OFFSET_Y)

        fx = lx / max(1.0, float(MINI_CELL))
        fy = ly / max(1.0, float(MINI_CELL))

        occupied = []
        for r in range(block.shape[0]):
            for c in range(block.shape[1]):
                if block[r, c] == 1:
                    occupied.append((r, c))

        if not occupied:
            return None

        best = None
        best_dist = None
        for r, c in occupied:
            dist = (fy - (r + 0.5)) ** 2 + (fx - (c + 0.5)) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (r, c)
        return best

    def apply_solver_setup(self):
        self.solver_raw.game.grid = self.solver_edit_grid.copy()
        self.solver_raw.game.hand = [SHAPE_LIBRARY[key].copy() for key in self.solver_edit_hand_keys]
        self.solver_raw.game.available = [True, True, True]
        self.solver_raw.game.stages_passed = 0
        self.solver_raw.game.lines_destroyed = 0
        self.solver_raw.game.blocks_placed = 0

        self.solver_obs = self.solver_raw._get_obs()
        self.solver_done = self.solver_raw.game.check_game_over()
        self.solver_solving = True
        self.solver_last_step_time = time.time()
        if self.solver_done:
            self.solver_status = "Setup has no legal moves."
        else:
            self.solver_status = "AI started solving your setup."

    def try_human_drop(self, slot_idx, anchor_row, anchor_col):
        game = self.vs_h_raw.game
        if not game.available[slot_idx]:
            self.vs_status = f"Slot {slot_idx + 1} is not available."
            return

        block = game.hand[slot_idx]
        if not game.can_place(block, anchor_row, anchor_col):
            self.vs_status = "Invalid placement."
            return

        action = action_from_selection(slot_idx, anchor_row, anchor_col)
        self.vs_h_obs, reward, self.vs_h_done, _, _ = self.vs_h_env.step(action)
        self.vs_status = f"You placed slot {slot_idx + 1} at R{anchor_row} C{anchor_col} | reward {reward:.1f}"

        self.ai_reply_in_versus()
        self.sync_vs_hands_when_stage_refreshes()
        self.resolve_versus_winner()

    def ai_reply_in_versus(self):
        if self.vs_a_done:
            return
        self.vs_a_obs, self.vs_a_done, self.vs_status = ai_step(
            self.model,
            self.vs_a_obs,
            self.vs_a_raw,
            self.vs_a_env,
        )

    def sync_vs_hands_when_stage_refreshes(self):
        h_game = self.vs_h_raw.game
        a_game = self.vs_a_raw.game

        if all(h_game.available) and all(a_game.available):
            a_game.hand = [block.copy() for block in h_game.hand]
            a_game.available = [True, True, True]
            self.vs_a_obs = self.vs_a_raw._get_obs()

    def resolve_versus_winner(self):
        if self.vs_winner is not None:
            return

        if (not self.vs_h_done) and (not self.vs_a_done):
            return

        if self.vs_h_done and (not self.vs_a_done):
            self.vs_winner = "AI wins (you ran out of moves first)."
        elif self.vs_a_done and (not self.vs_h_done):
            self.vs_winner = "You win (AI ran out of moves first)."
        else:
            h_stage = self.vs_h_raw.game.stages_passed
            a_stage = self.vs_a_raw.game.stages_passed
            if h_stage > a_stage:
                self.vs_winner = "You win on stages."
            elif a_stage > h_stage:
                self.vs_winner = "AI wins on stages."
            else:
                h_lines = self.vs_h_raw.game.lines_destroyed
                a_lines = self.vs_a_raw.game.lines_destroyed
                if h_lines > a_lines:
                    self.vs_winner = "You win on cleared lines."
                elif a_lines > h_lines:
                    self.vs_winner = "AI wins on cleared lines."
                else:
                    self.vs_winner = "Draw."

        self.vs_locked = True
        self.vs_status = self.vs_winner

    def draw_background(self, painter):
        painter.fillRect(self.rect(), to_color(BG))

        painter.setPen(Qt.NoPen)
        painter.setBrush(to_color((61, 112, 202), 84))
        painter.drawEllipse(60, -120, 520, 420)
        painter.setBrush(to_color((127, 228, 171), 62))
        painter.drawEllipse(1200, -80, 430, 360)
        painter.setBrush(to_color((255, 183, 102), 48))
        painter.drawEllipse(1280, 660, 340, 260)

    def draw_rounded_block(self, painter, rect, fill, radius=16, border=None, border_width=1):
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, radius, radius)
        if border is not None:
            pen = QPen(border)
            pen.setWidth(border_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)

    def draw_label(self, painter, text, x, y, color, font):
        painter.setPen(color)
        painter.setFont(font)
        painter.drawText(x, y, text)

    def draw_metric_card(self, painter, rect, title, value, accent):
        self.draw_rounded_block(
            painter,
            rect,
            to_color(PANEL_2),
            radius=14,
            border=to_color(accent),
            border_width=1,
        )
        self.draw_label(painter, title, rect.x() + 12, rect.y() + 24, to_color(MUTED), self.body_font)
        self.draw_label(painter, value, rect.x() + 12, rect.y() + 56, to_color(TEXT), self.subtitle_font)

    def draw_game_metrics(self, painter, game, x, y, done=None):
        cards = [
            ("Stages", str(game.stages_passed), HUMAN),
            ("Lines", str(game.lines_destroyed), AI),
            ("Blocks", str(game.blocks_placed), WARN),
        ]
        if done is not None:
            cards.append(("State", "DONE" if done else "LIVE", BAD if done else GOOD))

        for index, (title, value, accent) in enumerate(cards):
            self.draw_metric_card(painter, QRect(x + index * 202, y, 188, 74), title, value, accent)

    def draw_button(self, painter, button):
        hovered = button["rect"].contains(QPoint(int(self.mouse_pos[0]), int(self.mouse_pos[1])))
        fill = to_color(button["accent"] if hovered else PANEL_2)
        border = to_color(WHITE if hovered else (74, 92, 126))
        self.draw_rounded_block(painter, button["rect"], fill, radius=16, border=border, border_width=1)

        painter.setPen(to_color(WHITE))
        painter.setFont(self.button_font)
        painter.drawText(button["rect"], Qt.AlignCenter, button["label"])

    def draw_board(self, painter, grid, board_x, board_y, cell_size, title, accent, ghost_cells=None, ghost_valid=True):
        outer = QRect(board_x - 18, board_y - 18, GRID_SIZE * cell_size + 36, GRID_SIZE * cell_size + 36)
        inner = QRect(board_x - 6, board_y - 6, GRID_SIZE * cell_size + 12, GRID_SIZE * cell_size + 12)
        self.draw_rounded_block(
            painter,
            outer,
            to_color(PANEL),
            radius=22,
            border=to_color((74, 92, 126)),
            border_width=1,
        )
        self.draw_rounded_block(painter, inner, to_color(BOARD_BG), radius=18)

        ghost_set = set(ghost_cells or [])
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell = QRect(
                    int(board_x + col * cell_size),
                    int(board_y + row * cell_size),
                    int(cell_size - 3),
                    int(cell_size - 3),
                )
                occupied = grid[row, col] == 1
                self.draw_rounded_block(
                    painter,
                    cell,
                    to_color(CELL_FILL if occupied else CELL_BG),
                    radius=8,
                )

                if occupied:
                    shadow = QRect(cell.x(), cell.y() + cell.height() - 5, cell.width(), 4)
                    self.draw_rounded_block(painter, shadow, to_color(CELL_SHADOW), radius=5)

                if (row, col) in ghost_set:
                    ghost_color = to_color(GHOST_VALID if ghost_valid else GHOST_INVALID, 150)
                    self.draw_rounded_block(painter, cell, ghost_color, radius=8)

        self.draw_label(painter, title, board_x, board_y - 44, to_color(accent), self.subtitle_font)
        self.draw_label(painter, f"{GRID_SIZE} x {GRID_SIZE}", board_x + 180, board_y - 44, to_color(MUTED), self.body_font)

    def draw_hand(self, painter, hand, available, slot_rects, title, accent, selected_slot=None, key_labels=None):
        self.draw_label(painter, title, slot_rects[0].x(), slot_rects[0].y() - 36, to_color(TEXT), self.subtitle_font)

        for idx, rect in enumerate(slot_rects):
            selected = selected_slot == idx
            border_color = to_color(accent if selected else (75, 92, 123))

            self.draw_rounded_block(
                painter,
                rect.adjusted(-6, -6, 6, 6),
                to_color(PANEL_2),
                radius=16,
                border=border_color,
                border_width=2 if selected else 1,
            )
            self.draw_rounded_block(painter, rect, to_color((26, 39, 68)), radius=12)

            block = hand[idx]
            for r in range(block.shape[0]):
                for c in range(block.shape[1]):
                    if block[r, c] != 1:
                        continue
                    bx = rect.x() + HAND_BLOCK_OFFSET_X + c * MINI_CELL
                    by = rect.y() + HAND_BLOCK_OFFSET_Y + r * MINI_CELL
                    brect = QRect(int(bx), int(by), MINI_CELL - 2, MINI_CELL - 2)
                    fill = to_color(CELL_FILL if available[idx] else DIM)
                    sh = to_color(CELL_SHADOW if available[idx] else (65, 73, 96))
                    self.draw_rounded_block(painter, brect, fill, radius=6)
                    shadow = QRect(brect.x(), brect.y() + brect.height() - 4, brect.width(), 3)
                    self.draw_rounded_block(painter, shadow, sh, radius=4)

            label = key_labels[idx] if key_labels is not None else f"Slot {idx + 1}"
            self.draw_label(
                painter,
                label,
                rect.x() + 8,
                rect.y() - 12,
                to_color(accent if selected else MUTED),
                self.body_font,
            )

    def draw_menu_scene(self, painter):
        self.draw_label(painter, "Block Blast Arena", 86, 72, to_color(TEXT), self.title_font)
        self.draw_label(painter, "Choose a model, then choose one mode to start", 90, 110, to_color(MUTED), self.subtitle_font)

        left_panel = QRect(80, 150, 700, 690)
        right_panel = QRect(812, 150, 790, 690)

        self.draw_rounded_block(
            painter,
            left_panel,
            to_color(PANEL),
            radius=24,
            border=to_color((78, 96, 128)),
            border_width=1,
        )
        self.draw_rounded_block(
            painter,
            right_panel,
            to_color(PANEL),
            radius=24,
            border=to_color((78, 96, 128)),
            border_width=1,
        )

        for button in self.menu_buttons:
            self.draw_button(painter, button)

        self.draw_label(painter, self.menu_status, 96, 650, to_color(MUTED), self.body_font)
        self.draw_label(painter, self.model_status, 96, 690, to_color(AI if self.model is not None else WARN), self.body_font)
        self.draw_label(painter, f"Current board: {self.args.board_size} x {self.args.board_size}", 96, 724, to_color(MUTED), self.body_font)
        self.draw_label(painter, f"Generator: {self.args.hand_generator} | pool: {self.args.shape_pool}", 96, 758, to_color(MUTED), self.body_font)

        self.draw_label(painter, "Modes", 840, 202, to_color(TEXT), self.subtitle_font)
        mode_lines = [
            ("1. Let AI solve your position", "Build a board and hand, then let the loaded model continue.", HUMAN),
            ("2. Watch AI play a game", "Autoplay with the selected PPO model.", AI),
            ("3. Watch heuristic play", "Scores every legal move and always picks the best one.", WARN),
            ("4. You vs the AI", "Split boards. Your valid drop triggers one AI move.", (175, 121, 255)),
        ]
        for index, (title, detail, accent) in enumerate(mode_lines):
            y = 250 + index * 84
            self.draw_label(painter, title, 840, y, to_color(accent), self.subtitle_font)
            self.draw_label(painter, detail, 840, y + 30, to_color(MUTED), self.body_font)

        self.draw_label(painter, "Model selector", 840, 612, to_color(TEXT), self.subtitle_font)
        self.draw_label(painter, "Metrics are shown as mean/max stages when local data is known.", 840, 714, to_color(MUTED), self.body_font)

        if self.last_error:
            self.draw_label(painter, self.last_error[:90], 840, 760, to_color(BAD), self.body_font)

    def draw_solver_scene(self, painter):
        right_panel = QRect(622, 130, 968, 692)
        self.draw_rounded_block(
            painter,
            right_panel,
            to_color(PANEL),
            radius=24,
            border=to_color((78, 96, 128)),
            border_width=1,
        )

        board_grid = self.solver_raw.game.grid if self.solver_solving else self.solver_edit_grid
        self.draw_board(
            painter,
            board_grid,
            SINGLE_BOARD_X,
            SINGLE_BOARD_Y,
            SINGLE_CELL,
            "Custom Position",
            HUMAN,
        )

        slots = get_hand_slot_rects(SINGLE_BOARD_X + 6, SINGLE_BOARD_Y + SINGLE_BOARD_SIZE + 56)
        if self.solver_solving:
            hand = self.solver_raw.game.hand
            available = self.solver_raw.game.available
            labels = [f"Slot {i + 1}" for i in range(HAND_SIZE)]
        else:
            hand = [SHAPE_LIBRARY[k] for k in self.solver_edit_hand_keys]
            available = [True, True, True]
            labels = self.solver_edit_hand_keys

        self.draw_hand(
            painter,
            hand,
            available,
            slots,
            "Hand",
            HUMAN,
            selected_slot=None,
            key_labels=labels,
        )

        self.draw_label(painter, "Let AI Solve Your Position", 644, 168, to_color(TEXT), self.title_font)
        self.draw_label(painter, self.solver_status, 644, 208, to_color(MUTED), self.body_font)

        button_specs = {
            "start": ("Start AI Solve", AI),
            "reset": ("Reset Editor", WARN),
            "back": ("Back To Menu", BAD),
        }
        for key, rect in self.solver_buttons.items():
            label, accent = button_specs[key]
            self.draw_button(painter, make_button(rect, label, accent))

        self.draw_game_metrics(painter, self.solver_raw.game, 644, 334, done=self.solver_done)

        if not self.solver_solving:
            self.draw_label(painter, "Editor controls:", 644, 494, to_color(TEXT), self.subtitle_font)
            self.draw_label(painter, "- Left click board cell: toggle filled / empty", 644, 528, to_color(MUTED), self.body_font)
            self.draw_label(painter, "- Left click hand slot: next shape", 644, 556, to_color(MUTED), self.body_font)
            self.draw_label(painter, "- Right click hand slot: previous shape", 644, 584, to_color(MUTED), self.body_font)

    def draw_watch_scene(self, painter):
        right_panel = QRect(622, 130, 968, 692)
        self.draw_rounded_block(
            painter,
            right_panel,
            to_color(PANEL),
            radius=24,
            border=to_color((78, 96, 128)),
            border_width=1,
        )

        title = "Heuristic Board" if self.watch_policy == "heuristic" else "AI Board"
        accent = WARN if self.watch_policy == "heuristic" else AI
        self.draw_board(
            painter,
            self.watch_raw.game.grid,
            SINGLE_BOARD_X,
            SINGLE_BOARD_Y,
            SINGLE_CELL,
            title,
            accent,
        )

        slots = get_hand_slot_rects(SINGLE_BOARD_X + 6, SINGLE_BOARD_Y + SINGLE_BOARD_SIZE + 56)
        self.draw_hand(
            painter,
            self.watch_raw.game.hand,
            self.watch_raw.game.available,
            slots,
            "Heuristic Hand" if self.watch_policy == "heuristic" else "AI Hand",
            accent,
        )

        self.draw_label(painter, "Watch Heuristic Play" if self.watch_policy == "heuristic" else "Watch AI Play", 644, 168, to_color(TEXT), self.title_font)
        self.draw_label(painter, self.watch_status, 644, 208, to_color(MUTED), self.body_font)

        self.draw_button(painter, make_button(self.watch_buttons["new"], "New Random Game", WARN))
        self.draw_button(painter, make_button(self.watch_buttons["back"], "Back To Menu", BAD))
        self.draw_game_metrics(painter, self.watch_raw.game, 644, 322, done=self.watch_done)

    def current_drag_anchor(self):
        if (not self.drag_active) or self.drag_slot is None:
            return None

        block = self.vs_h_raw.game.hand[self.drag_slot]
        cell = point_to_cell(self.mouse_pos, DUAL_BOARD_X_LEFT, DUAL_BOARD_Y, DUAL_CELL)
        if cell is None:
            return None

        row, col = cell
        anchor_row = row - self.drag_offset_cell[0]
        anchor_col = col - self.drag_offset_cell[1]
        valid = self.vs_h_raw.game.can_place(block, anchor_row, anchor_col)
        return (anchor_row, anchor_col, valid)

    def draw_drag_piece(self, painter):
        if (not self.drag_active) or self.drag_slot is None:
            return

        block = self.vs_h_raw.game.hand[self.drag_slot]
        anchor_preview = self.current_drag_anchor()
        valid = anchor_preview is not None and anchor_preview[2]

        if anchor_preview is not None:
            anchor_row, anchor_col, _ = anchor_preview
            top_left_x = int(DUAL_BOARD_X_LEFT + anchor_col * DUAL_CELL)
            top_left_y = int(DUAL_BOARD_Y + anchor_row * DUAL_CELL)
        else:
            top_left_x = int(self.drag_visual_pos[0] - (self.drag_offset_cell[1] * DUAL_CELL))
            top_left_y = int(self.drag_visual_pos[1] - (self.drag_offset_cell[0] * DUAL_CELL))

        fill = to_color(GHOST_VALID if valid else GHOST_INVALID, 160)
        for r in range(block.shape[0]):
            for c in range(block.shape[1]):
                if block[r, c] != 1:
                    continue
                rect = QRect(
                    top_left_x + c * DUAL_CELL + 1,
                    top_left_y + r * DUAL_CELL + 1,
                    DUAL_CELL - 3,
                    DUAL_CELL - 3,
                )
                self.draw_rounded_block(painter, rect, fill, radius=8)

    def draw_versus_scene(self, painter):
        side_panel = QRect(1204, 130, 396, 692)
        self.draw_rounded_block(
            painter,
            side_panel,
            to_color(PANEL),
            radius=24,
            border=to_color((78, 96, 128)),
            border_width=1,
        )

        ghost_cells = None
        ghost_valid = False
        anchor = self.current_drag_anchor()
        if anchor is not None and self.drag_slot is not None:
            ar, ac, ghost_valid = anchor
            block = self.vs_h_raw.game.hand[self.drag_slot]
            ghost_cells = compute_block_cells(block, ar, ac)

        self.draw_board(
            painter,
            self.vs_h_raw.game.grid,
            DUAL_BOARD_X_LEFT,
            DUAL_BOARD_Y,
            DUAL_CELL,
            "Your Board",
            HUMAN,
            ghost_cells=ghost_cells,
            ghost_valid=ghost_valid,
        )

        self.draw_board(
            painter,
            self.vs_a_raw.game.grid,
            DUAL_BOARD_X_RIGHT,
            DUAL_BOARD_Y,
            DUAL_CELL,
            "AI Board",
            AI,
        )

        human_slots = get_hand_slot_rects(DUAL_BOARD_X_LEFT, DUAL_BOARD_Y + DUAL_BOARD_SIZE + 50)
        ai_slots = get_hand_slot_rects(DUAL_BOARD_X_RIGHT, DUAL_BOARD_Y + DUAL_BOARD_SIZE + 50)

        self.draw_hand(
            painter,
            self.vs_h_raw.game.hand,
            self.vs_h_raw.game.available,
            human_slots,
            "Your Hand (drag and drop)",
            HUMAN,
            selected_slot=self.drag_slot if self.drag_active else None,
        )
        self.draw_hand(
            painter,
            self.vs_a_raw.game.hand,
            self.vs_a_raw.game.available,
            ai_slots,
            "AI Hand",
            AI,
        )

        self.draw_label(painter, "You vs the AI", 1222, 168, to_color(TEXT), self.title_font)
        self.draw_label(painter, self.vs_status, 1222, 208, to_color(MUTED), self.body_font)

        self.draw_button(painter, make_button(self.vs_buttons["new"], "New Match", WARN))
        self.draw_button(painter, make_button(self.vs_buttons["back"], "Back To Menu", BAD))

        self.draw_label(painter, "You", 1222, 340, to_color(HUMAN), self.subtitle_font)
        self.draw_metric_card(painter, QRect(1222, 356, 176, 68), "Stages", str(self.vs_h_raw.game.stages_passed), HUMAN)
        self.draw_metric_card(painter, QRect(1408, 356, 176, 68), "Lines", str(self.vs_h_raw.game.lines_destroyed), HUMAN)
        self.draw_metric_card(painter, QRect(1222, 432, 176, 68), "Blocks", str(self.vs_h_raw.game.blocks_placed), HUMAN)

        self.draw_label(painter, "AI", 1222, 544, to_color(AI), self.subtitle_font)
        self.draw_metric_card(painter, QRect(1222, 560, 176, 68), "Stages", str(self.vs_a_raw.game.stages_passed), AI)
        self.draw_metric_card(painter, QRect(1408, 560, 176, 68), "Lines", str(self.vs_a_raw.game.lines_destroyed), AI)
        self.draw_metric_card(painter, QRect(1222, 636, 176, 68), "Blocks", str(self.vs_a_raw.game.blocks_placed), AI)

        if self.vs_winner is not None:
            self.draw_label(painter, f"Result: {self.vs_winner}", 1222, 736, to_color(GOOD), self.subtitle_font)

        self.draw_drag_piece(painter)


def main():
    args = parse_args()
    args = resolve_watch_defaults(args)
    configure_board_mode(args.board_size)

    model = None
    if not args.no_model:
        bootstrap_raw, bootstrap_env, _ = create_env_bundle(args)
        try:
            model = load_model(bootstrap_env, args.model_path)
        except FileNotFoundError:
            print(f"Model not found: {args.model_path}.zip")
            raise SystemExit(1)
        except RuntimeError as exc:
            print(str(exc))
            raise SystemExit(1)

        del bootstrap_raw

    app = QApplication(sys.argv)
    widget = ArenaWidget(model, args)
    widget.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
