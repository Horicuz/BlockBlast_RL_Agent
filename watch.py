import argparse
import os
import random
import time

import numpy as np
import pygame
from blocks import SHAPES
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

GRID_SIZE = 8
HAND_SIZE = 3
AUTO_MOVE_INTERVAL = 0.7

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860

BOARD_X = 64
BOARD_Y = 110
CELL_SIZE = 56
CELL_GAP = 4
BOARD_SIZE = GRID_SIZE * CELL_SIZE

PANEL_X = BOARD_X + BOARD_SIZE + 44
PANEL_W = WINDOW_WIDTH - PANEL_X - 64

HAND_Y = 700
HAND_SLOT_W = 112
HAND_SLOT_H = 112
HAND_SLOT_GAP = 20

BG = (11, 18, 33)
BG_2 = (16, 26, 47)
PANEL = (19, 28, 48)
PANEL_2 = (26, 38, 63)
BOARD_BG = (23, 32, 54)
BOARD_CELL = (40, 51, 79)
BOARD_CELL_HOT = (55, 68, 104)
BLOCK = (89, 198, 255)
BLOCK_SHADOW = (34, 128, 193)
HUMAN = (255, 184, 77)
AI = (113, 214, 161)
ROUNDSAFE = (175, 121, 255)
WARN = (255, 201, 92)
LAST_MOVE = WARN
LAST_MOVE_SHADOW = (176, 132, 34)
TEXT = (233, 240, 255)
MUTED = (160, 174, 198)
DIM = (84, 96, 120)
GOOD = (70, 226, 152)
BAD = (255, 110, 122)
WHITE = (255, 255, 255)

SHAPE_KEYS = list(SHAPES.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Play and compare Block Blast runs against the AI")
    parser.add_argument("--model-path", default="checkpoints/roundsafe/block_blast_roundsafe_v1", help="Model path with or without .zip")
    parser.add_argument("--auto-interval", type=float, default=AUTO_MOVE_INTERVAL)
    return parser.parse_args()


def mask_fn(env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()

    raise AttributeError("valid_action_mask() is not available on the current environment")


def shape_key_from_matrix(matrix):
    for key in SHAPE_KEYS:
        shape = SHAPES[key]
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
    }


def restore_game_state(game, game_state):
    game.grid = game_state["grid"].copy()
    game.hand = [block.copy() for block in game_state["hand"]]
    game.available = list(game_state["available"])
    game.stages_passed = game_state["stages_passed"]
    game.lines_destroyed = game_state["lines_destroyed"]
    game.blocks_placed = game_state["blocks_placed"]


def capture_snapshot(raw_env, runtime_state):
    return {
        "game_state": capture_game_state(raw_env.game),
        "random_state": random.getstate(),
        "done": runtime_state["done"],
        "mode": runtime_state["mode"],
        "last_action_text": runtime_state["last_action_text"],
        "last_placed_cells": list(runtime_state["last_placed_cells"]),
    }


def apply_snapshot(raw_env, runtime_state, snapshot):
    restore_game_state(raw_env.game, snapshot["game_state"])
    random.setstate(snapshot["random_state"])

    runtime_state["obs"] = raw_env._get_obs()
    runtime_state["done"] = snapshot["done"]
    runtime_state["mode"] = snapshot.get("mode", runtime_state["mode"])
    runtime_state["last_action_text"] = snapshot["last_action_text"]
    runtime_state["last_placed_cells"] = list(snapshot["last_placed_cells"])
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []
    runtime_state["step_requested"] = False
    runtime_state["auto_enabled"] = False


def push_history(raw_env, runtime_state):
    history = runtime_state["history"]
    history_index = runtime_state["history_index"]

    if history_index < len(history) - 1:
        runtime_state["history"] = history[:history_index + 1]

    runtime_state["history"].append(capture_snapshot(raw_env, runtime_state))
    runtime_state["history_index"] += 1


def decode_action(action):
    hand_idx = action // 64
    remainder = action % 64
    row = remainder // 8
    col = remainder % 8
    return hand_idx, row, col


def action_from_selection(hand_index, row, col):
    return (hand_index * 64) + (row * 8) + col


def compute_placed_cells(block_matrix, row, col):
    cells = []
    for block_row in range(block_matrix.shape[0]):
        for block_col in range(block_matrix.shape[1]):
            if block_matrix[block_row, block_col] == 1:
                cells.append((row + block_row, col + block_col))
    return cells


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def draw_rounded_rect(screen, color, rect, radius=16, border_color=None, border_width=0):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    if border_color is not None and border_width > 0:
        pygame.draw.rect(screen, border_color, rect, width=border_width, border_radius=radius)


def draw_gradient_background(screen):
    screen.fill(BG)
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(overlay, (55, 98, 180, 80), (140, 110), 260)
    pygame.draw.circle(overlay, (168, 119, 255, 70), (1110, 150), 220)
    pygame.draw.circle(overlay, (80, 232, 172, 40), (1000, 760), 220)
    pygame.draw.rect(overlay, (255, 255, 255, 12), (0, 0, WINDOW_WIDTH, 80))
    screen.blit(overlay, (0, 0))


def draw_text(screen, font, text, color, pos):
    surface = font.render(text, True, color)
    screen.blit(surface, pos)
    return surface.get_rect(topleft=pos)


def draw_chip(screen, font, label, value, pos, accent):
    rect = pygame.Rect(pos[0], pos[1], 180, 64)
    draw_rounded_rect(screen, PANEL_2, rect, radius=18, border_color=accent, border_width=1)
    draw_text(screen, font, label, MUTED, (rect.x + 14, rect.y + 10))
    draw_text(screen, font, str(value), TEXT, (rect.x + 14, rect.y + 32))


def draw_button(screen, font, rect, label, active=False, accent=BLOCK):
    bg = accent if active else PANEL_2
    border = WHITE if active else (70, 86, 116)
    draw_rounded_rect(screen, bg, rect, radius=16, border_color=border, border_width=1)
    text_surface = font.render(label, True, WHITE)
    screen.blit(text_surface, text_surface.get_rect(center=rect.center))


def build_buttons():
    buttons = {}
    x = PANEL_X
    y = 370
    w = 172
    h = 42
    gap = 12

    labels = [
        ("human", "Human Mode"),
        ("ai_replay", "AI Replay"),
        ("auto", "Auto AI"),
        ("step", "Step AI"),
        ("new", "New Round"),
        ("undo", "Undo"),
        ("redo", "Redo"),
        ("edit", "Edit Setup"),
        ("apply", "Apply Setup"),
        ("same_seed", "Reset Same Seed"),
    ]

    for index, (key, label) in enumerate(labels):
        row = index // 2
        col = index % 2
        rect = pygame.Rect(x + col * (w + gap), y + row * (h + gap), w, h)
        buttons[key] = (rect, label)

    return buttons


def get_hand_slot_rects():
    rects = []
    start_x = BOARD_X + 8
    for i in range(HAND_SIZE):
        x = start_x + i * (HAND_SLOT_W + HAND_SLOT_GAP)
        rects.append(pygame.Rect(x, HAND_Y, HAND_SLOT_W, HAND_SLOT_H))
    return rects


def draw_mini_block(screen, block_matrix, start_x, start_y, color, shadow, available=True):
    mini = 22
    rows, cols = block_matrix.shape
    for r in range(rows):
        for c in range(cols):
            if block_matrix[r, c] == 1:
                rect = pygame.Rect(start_x + c * mini, start_y + r * mini, mini - 2, mini - 2)
                draw_color = color if available else DIM
                draw_shadow = shadow if available else (52, 58, 76)
                pygame.draw.rect(screen, draw_color, rect, border_radius=6)
                shadow_rect = pygame.Rect(rect.x, rect.y + mini - 6, rect.w, 4)
                pygame.draw.rect(screen, draw_shadow, shadow_rect, border_radius=6)


def draw_board(screen, board_font, raw_env, runtime_state, selected_slot, hand_slot_rects, edit_mode, edit_grid, edit_hand_keys):
    board_rect = pygame.Rect(BOARD_X - 18, BOARD_Y - 18, BOARD_SIZE + 36, BOARD_SIZE + 36)
    draw_rounded_rect(screen, PANEL, board_rect, radius=28, border_color=(68, 84, 120), border_width=1)

    inner_rect = pygame.Rect(BOARD_X - 6, BOARD_Y - 6, BOARD_SIZE + 12, BOARD_SIZE + 12)
    draw_rounded_rect(screen, BOARD_BG, inner_rect, radius=22)

    grid = edit_grid if edit_mode else raw_env.game.grid

    selected_block = None
    if edit_mode:
        selected_block = SHAPES[edit_hand_keys[selected_slot]]
    elif runtime_state["mode"] == "human" and not runtime_state["done"] and raw_env.game.available[selected_slot]:
        selected_block = raw_env.game.hand[selected_slot]

    legal_cells = set()
    if selected_block is not None and not edit_mode and runtime_state["mode"] == "human":
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if raw_env.game.can_place(selected_block, row, col):
                    legal_cells.add((row, col))

    hover_row, hover_col = runtime_state["hover_cell"] if runtime_state["hover_cell"] is not None else (None, None)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_rect = pygame.Rect(
                BOARD_X + col * CELL_SIZE,
                BOARD_Y + row * CELL_SIZE,
                CELL_SIZE - CELL_GAP,
                CELL_SIZE - CELL_GAP,
            )

            occupied = grid[row, col] == 1
            animating = (not edit_mode) and runtime_state["anim_frames"] > 0 and (row in runtime_state["anim_rows"] or col in runtime_state["anim_cols"])
            last_move = (row, col) in runtime_state["last_placed_cells"] and not edit_mode
            legal = (row, col) in legal_cells
            hovered = (row == hover_row and col == hover_col)

            cell_color = BOARD_CELL_HOT if hovered or legal else BOARD_CELL
            if occupied or last_move:
                cell_color = LAST_MOVE if last_move else BLOCK
            if animating:
                cell_color = WHITE

            pygame.draw.rect(screen, cell_color, cell_rect, border_radius=10)

            if occupied or last_move:
                shadow_color = LAST_MOVE_SHADOW if last_move else BLOCK_SHADOW
                shadow_rect = pygame.Rect(cell_rect.x, cell_rect.y + CELL_SIZE - 14, cell_rect.w, 5)
                pygame.draw.rect(screen, shadow_color, shadow_rect, border_radius=8)
            elif legal:
                outline = pygame.Surface((cell_rect.w, cell_rect.h), pygame.SRCALPHA)
                pygame.draw.rect(outline, (113, 214, 161, 70), outline.get_rect(), width=2, border_radius=10)
                screen.blit(outline, (cell_rect.x, cell_rect.y))

    title_font = board_font[0]
    subtitle_font = board_font[1]
    draw_text(screen, title_font, "BLOCK BLAST ARENA", TEXT, (PANEL_X, 52))
    draw_text(screen, subtitle_font, f"Mode: {runtime_state['mode'].upper()}", ROUNDSAFE if runtime_state["mode"] == "ai" else HUMAN, (PANEL_X, 88))

    return selected_block


def draw_hand(screen, font, raw_env, runtime_state, selected_slot, hand_slot_rects, edit_mode, edit_hand_keys):
    hand_title_font = font[0]
    small_font = font[1]

    draw_text(screen, hand_title_font, "Hand", TEXT, (BOARD_X, 640))

    for index, slot_rect in enumerate(hand_slot_rects):
        active = index == selected_slot
        border = HUMAN if runtime_state["mode"] == "human" else ROUNDSAFE
        if active:
            draw_rounded_rect(screen, (30, 42, 70), slot_rect.inflate(14, 14), radius=20, border_color=border, border_width=2)
        else:
            draw_rounded_rect(screen, (22, 32, 55), slot_rect.inflate(14, 14), radius=20, border_color=(68, 84, 120), border_width=1)

        if edit_mode:
            block = SHAPES[edit_hand_keys[index]]
            available = True
            label = edit_hand_keys[index]
        else:
            block = raw_env.game.hand[index]
            available = raw_env.game.available[index]
            label = f"Slot {index + 1}"

        draw_mini_block(screen, block, slot_rect.x + 26, slot_rect.y + 20, BLOCK, BLOCK_SHADOW, available=available)
        label_color = HUMAN if active else MUTED
        draw_text(screen, small_font, label, label_color, (slot_rect.x + 12, slot_rect.y - 28))


def draw_sidebar(screen, fonts, raw_env, runtime_state, buttons):
    title_font, big_font, small_font, tiny_font = fonts

    panel_rect = pygame.Rect(PANEL_X - 18, 24, PANEL_W, WINDOW_HEIGHT - 48)
    draw_rounded_rect(screen, PANEL, panel_rect, radius=26, border_color=(68, 84, 120), border_width=1)

    draw_text(screen, title_font, "Session Control", TEXT, (PANEL_X + 4, 34))
    draw_text(screen, small_font, runtime_state["last_action_text"], MUTED, (PANEL_X + 4, 68))

    stage_color = HUMAN if runtime_state["mode"] == "human" else ROUNDSAFE
    draw_chip(screen, big_font, "Current stages", raw_env.game.stages_passed, (PANEL_X, 112), stage_color)
    draw_chip(screen, big_font, "Lines cleared", raw_env.game.lines_destroyed, (PANEL_X + 192, 112), AI)
    draw_chip(screen, big_font, "Blocks placed", raw_env.game.blocks_placed, (PANEL_X, 186), BLOCK)
    draw_chip(screen, big_font, "Round state", "DONE" if runtime_state["done"] else "LIVE", (PANEL_X + 192, 186), BAD if runtime_state["done"] else GOOD)

    stats_top = 264
    draw_rounded_rect(screen, PANEL_2, pygame.Rect(PANEL_X, stats_top, PANEL_W - 8, 104), radius=20, border_color=(68, 84, 120), border_width=1)
    draw_text(screen, small_font, "Comparison", TEXT, (PANEL_X + 14, stats_top + 12))
    draw_text(
        screen,
        tiny_font,
        f"You: last {runtime_state['human_last']} | best {runtime_state['human_best']} | runs {runtime_state['human_runs']}",
        HUMAN,
        (PANEL_X + 14, stats_top + 38),
    )
    draw_text(
        screen,
        tiny_font,
        f"AI:  last {runtime_state['ai_last']} | best {runtime_state['ai_best']} | runs {runtime_state['ai_runs']}",
        ROUNDSAFE,
        (PANEL_X + 14, stats_top + 60),
    )

    button_font = fonts[3]
    for index, (key, payload) in enumerate(buttons.items()):
        rect, label = payload
        row = index // 2
        active = False
        if key == "human":
            active = runtime_state["mode"] == "human"
        elif key == "auto":
            active = runtime_state["mode"] == "ai" and runtime_state["auto_enabled"]
        elif key == "edit":
            active = runtime_state["edit_mode"]
        elif key == "ai_replay":
            active = runtime_state["mode"] == "ai"

        accent = ROUNDSAFE if key in {"ai_replay", "auto", "step"} else HUMAN
        if key in {"undo", "redo", "new", "edit", "apply", "same_seed"}:
            accent = BLOCK

        draw_button(screen, button_font, rect, label, active=active, accent=accent)

    help_y = 610
    draw_rounded_rect(screen, PANEL_2, pygame.Rect(PANEL_X, help_y, PANEL_W - 8, 180), radius=20, border_color=(68, 84, 120), border_width=1)
    draw_text(screen, small_font, "How to play", TEXT, (PANEL_X + 14, help_y + 12))
    draw_text(screen, tiny_font, "1. Switch to Human Mode and click a hand slot.", MUTED, (PANEL_X + 14, help_y + 42))
    draw_text(screen, tiny_font, "2. Click a board cell to place the piece.", MUTED, (PANEL_X + 14, help_y + 66))
    draw_text(screen, tiny_font, "3. After your run, press AI Replay to compare the same start.", MUTED, (PANEL_X + 14, help_y + 90))
    draw_text(screen, tiny_font, "4. In AI mode, use Auto AI or Step AI.", MUTED, (PANEL_X + 14, help_y + 114))


def load_model(env, model_path):
    normalized_path = model_path[:-4] if model_path.endswith(".zip") else model_path
    if not os.path.exists(normalized_path + ".zip"):
        raise FileNotFoundError(normalized_path + ".zip")
    return MaskablePPO.load(normalized_path, env=env)


def initialize_runtime(env, raw_env):
    obs, _ = env.reset()
    runtime_state = {
        "obs": obs,
        "done": False,
        "mode": "human",
        "auto_enabled": False,
        "step_requested": False,
        "last_action_text": "New round ready. Select Human Mode or AI Replay.",
        "last_placed_cells": [],
        "anim_frames": 0,
        "anim_rows": [],
        "anim_cols": [],
        "last_auto_move": time.time(),
        "history": [],
        "history_index": 0,
        "hover_cell": None,
        "edit_mode": False,
        "selected_slot": 0,
        "human_runs": 0,
        "ai_runs": 0,
        "human_best": 0,
        "ai_best": 0,
        "human_last": 0,
        "ai_last": 0,
        "finalized_mode": None,
        "challenge_snapshot": None,
    }
    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["challenge_snapshot"] = runtime_state["history"][0]
    return runtime_state


def reset_round(env, raw_env, runtime_state, mode="human"):
    obs, _ = env.reset()
    runtime_state["obs"] = obs
    runtime_state["done"] = False
    runtime_state["mode"] = mode
    runtime_state["auto_enabled"] = False
    runtime_state["step_requested"] = False
    runtime_state["last_action_text"] = "New round ready."
    runtime_state["last_placed_cells"] = []
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []
    runtime_state["last_auto_move"] = time.time()
    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["history_index"] = 0
    runtime_state["finalized_mode"] = None
    runtime_state["challenge_snapshot"] = runtime_state["history"][0]


def push_run_stats(runtime_state, mode, stages):
    if mode == "human":
        runtime_state["human_runs"] += 1
        runtime_state["human_last"] = stages
        runtime_state["human_best"] = max(runtime_state["human_best"], stages)
    else:
        runtime_state["ai_runs"] += 1
        runtime_state["ai_last"] = stages
        runtime_state["ai_best"] = max(runtime_state["ai_best"], stages)


def finalize_if_needed(raw_env, runtime_state):
    if runtime_state["done"] and runtime_state["finalized_mode"] != runtime_state["mode"]:
        push_run_stats(runtime_state, runtime_state["mode"], raw_env.game.stages_passed)
        runtime_state["finalized_mode"] = runtime_state["mode"]
        runtime_state["last_action_text"] = f"{runtime_state['mode'].title()} run finished at {raw_env.game.stages_passed} stages."


def cycle_piece_key(edit_hand_keys, selected_slot, delta):
    current = edit_hand_keys[selected_slot]
    idx = SHAPE_KEYS.index(current)
    edit_hand_keys[selected_slot] = SHAPE_KEYS[(idx + delta) % len(SHAPE_KEYS)]


def apply_editor_setup(raw_env, runtime_state, edit_grid, edit_hand_keys):
    raw_env.game.grid = edit_grid.copy()
    raw_env.game.hand = [SHAPES[key].copy() for key in edit_hand_keys]
    raw_env.game.available = [True, True, True]
    raw_env.game.stages_passed = 0
    raw_env.game.lines_destroyed = 0
    raw_env.game.blocks_placed = 0

    runtime_state["obs"] = raw_env._get_obs()
    runtime_state["done"] = raw_env.game.check_game_over()
    runtime_state["mode"] = "human"
    runtime_state["auto_enabled"] = False
    runtime_state["last_action_text"] = "Setup applied. You can play from this custom position."
    runtime_state["last_placed_cells"] = []
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []
    runtime_state["last_auto_move"] = time.time()
    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["history_index"] = 0
    runtime_state["finalized_mode"] = None
    runtime_state["challenge_snapshot"] = runtime_state["history"][0]


def apply_ai_move(model, env, raw_env, runtime_state):
    if runtime_state["done"]:
        return

    action_masks = mask_fn(raw_env)
    if not np.any(action_masks):
        runtime_state["done"] = True
        runtime_state["last_action_text"] = "No valid AI moves available."
        return

    action, _ = model.predict(runtime_state["obs"], action_masks=action_masks, deterministic=True)
    action = int(action)

    hand_idx, row, col = decode_action(action)
    block_matrix = raw_env.game.hand[hand_idx]
    runtime_state["last_placed_cells"] = compute_placed_cells(block_matrix, row, col)

    runtime_state["obs"], reward, runtime_state["done"], _, info = env.step(action)
    runtime_state["last_action_text"] = f"AI: slot {hand_idx + 1} at R{row} C{col} | reward {reward:.1f}"
    runtime_state["anim_rows"] = info.get("anim_rows", [])
    runtime_state["anim_cols"] = info.get("anim_cols", [])
    runtime_state["anim_frames"] = 15 if (runtime_state["anim_rows"] or runtime_state["anim_cols"]) else 0
    runtime_state["last_auto_move"] = time.time()

    push_history(raw_env, runtime_state)


def apply_human_move(env, raw_env, runtime_state, selected_slot, board_cell):
    if runtime_state["done"] or runtime_state["mode"] != "human" or runtime_state["edit_mode"]:
        return

    if not raw_env.game.available[selected_slot]:
        runtime_state["last_action_text"] = f"Slot {selected_slot + 1} is not available."
        return

    row, col = board_cell
    block_matrix = raw_env.game.hand[selected_slot]
    if not raw_env.game.can_place(block_matrix, row, col):
        runtime_state["last_action_text"] = f"Invalid placement for slot {selected_slot + 1}."
        return

    action = action_from_selection(selected_slot, row, col)
    runtime_state["last_placed_cells"] = compute_placed_cells(block_matrix, row, col)
    runtime_state["obs"], reward, runtime_state["done"], _, info = env.step(action)
    runtime_state["last_action_text"] = f"You placed slot {selected_slot + 1} at R{row} C{col} | reward {reward:.1f}"
    runtime_state["anim_rows"] = info.get("anim_rows", [])
    runtime_state["anim_cols"] = info.get("anim_cols", [])
    runtime_state["anim_frames"] = 15 if (runtime_state["anim_rows"] or runtime_state["anim_cols"]) else 0
    runtime_state["last_auto_move"] = time.time()
    push_history(raw_env, runtime_state)


def main():
    args = parse_args()

    pygame.init()
    pygame.display.set_caption("Block Blast Arena")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    big_font = pygame.font.SysFont("DejaVu Sans", 22, bold=True)
    small_font = pygame.font.SysFont("DejaVu Sans", 18, bold=True)
    tiny_font = pygame.font.SysFont("DejaVu Sans", 15, bold=False)
    button_font = pygame.font.SysFont("DejaVu Sans", 16, bold=True)

    raw_env = BlockBlastEnv()
    env = ActionMasker(raw_env, mask_fn)

    try:
        model = load_model(env, args.model_path)
    except FileNotFoundError:
        print(f"Model not found: {args.model_path}.zip")
        pygame.quit()
        raise SystemExit(1)

    runtime_state = initialize_runtime(env, raw_env)
    buttons = build_buttons()
    hand_slot_rects = get_hand_slot_rects()

    edit_mode = False
    edit_grid = raw_env.game.grid.copy()
    edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]
    selected_slot = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    selected_slot = event.key - pygame.K_1
                elif event.key == pygame.K_e:
                    edit_mode = not edit_mode
                    runtime_state["edit_mode"] = edit_mode
                    runtime_state["auto_enabled"] = False
                    runtime_state["mode"] = "human"
                elif event.key == pygame.K_a and runtime_state["mode"] == "ai":
                    runtime_state["auto_enabled"] = not runtime_state["auto_enabled"]
                elif event.key == pygame.K_SPACE and runtime_state["mode"] == "ai":
                    runtime_state["step_requested"] = True
                elif edit_mode and event.key == pygame.K_LEFT:
                    cycle_piece_key(edit_hand_keys, selected_slot, -1)
                elif edit_mode and event.key == pygame.K_RIGHT:
                    cycle_piece_key(edit_hand_keys, selected_slot, 1)

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                if BOARD_X <= mouse_x < BOARD_X + BOARD_SIZE and BOARD_Y <= mouse_y < BOARD_Y + BOARD_SIZE:
                    runtime_state["hover_cell"] = ((mouse_y - BOARD_Y) // CELL_SIZE, (mouse_x - BOARD_X) // CELL_SIZE)
                else:
                    runtime_state["hover_cell"] = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                clicked_key = None

                for key, (rect, _) in buttons.items():
                    if rect.collidepoint(mouse_pos):
                        clicked_key = key
                        break

                if clicked_key == "human":
                    runtime_state["mode"] = "human"
                    runtime_state["auto_enabled"] = False
                    edit_mode = False

                elif clicked_key == "ai_replay":
                    if runtime_state["challenge_snapshot"] is not None:
                        apply_snapshot(raw_env, runtime_state, runtime_state["challenge_snapshot"])
                        runtime_state["mode"] = "ai"
                        runtime_state["auto_enabled"] = False
                        edit_mode = False

                elif clicked_key == "auto" and runtime_state["mode"] == "ai":
                    runtime_state["auto_enabled"] = not runtime_state["auto_enabled"]

                elif clicked_key == "step" and runtime_state["mode"] == "ai":
                    runtime_state["step_requested"] = True

                elif clicked_key == "new":
                    reset_round(env, raw_env, runtime_state, mode="human")
                    edit_mode = False
                    edit_grid = raw_env.game.grid.copy()
                    edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]
                    selected_slot = 0

                elif clicked_key == "undo":
                    runtime_state["auto_enabled"] = False
                    if runtime_state["history_index"] > 0:
                        runtime_state["history_index"] -= 1
                        apply_snapshot(raw_env, runtime_state, runtime_state["history"][runtime_state["history_index"]])

                elif clicked_key == "redo":
                    runtime_state["auto_enabled"] = False
                    if runtime_state["history_index"] < len(runtime_state["history"]) - 1:
                        runtime_state["history_index"] += 1
                        apply_snapshot(raw_env, runtime_state, runtime_state["history"][runtime_state["history_index"]])

                elif clicked_key == "edit":
                    edit_mode = not edit_mode
                    runtime_state["edit_mode"] = edit_mode
                    runtime_state["auto_enabled"] = False
                    runtime_state["mode"] = "human"
                    if edit_mode:
                        edit_grid = raw_env.game.grid.copy()
                        edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]

                elif clicked_key == "apply":
                    apply_editor_setup(raw_env, runtime_state, edit_grid, edit_hand_keys)
                    edit_mode = False
                    runtime_state["edit_mode"] = False

                elif clicked_key == "same_seed":
                    if runtime_state["challenge_snapshot"] is not None:
                        apply_snapshot(raw_env, runtime_state, runtime_state["challenge_snapshot"])
                        runtime_state["mode"] = "human"
                        runtime_state["auto_enabled"] = False
                        edit_mode = False

                else:
                    if edit_mode:
                        for idx, slot_rect in enumerate(hand_slot_rects):
                            if slot_rect.collidepoint(mouse_pos):
                                selected_slot = idx

                        board_x, board_y = mouse_pos
                        if BOARD_X <= board_x < BOARD_X + BOARD_SIZE and BOARD_Y <= board_y < BOARD_Y + BOARD_SIZE:
                            col = (board_x - BOARD_X) // CELL_SIZE
                            row = (board_y - BOARD_Y) // CELL_SIZE
                            edit_grid[row, col] = 0 if edit_grid[row, col] == 1 else 1
                    else:
                        for idx, slot_rect in enumerate(hand_slot_rects):
                            if slot_rect.collidepoint(mouse_pos):
                                selected_slot = idx
                                runtime_state["selected_slot"] = selected_slot

                        if runtime_state["mode"] == "human" and not runtime_state["done"]:
                            board_x, board_y = mouse_pos
                            if BOARD_X <= board_x < BOARD_X + BOARD_SIZE and BOARD_Y <= board_y < BOARD_Y + BOARD_SIZE:
                                col = (board_x - BOARD_X) // CELL_SIZE
                                row = (board_y - BOARD_Y) // CELL_SIZE
                                apply_human_move(env, raw_env, runtime_state, selected_slot, (row, col))

        draw_gradient_background(screen)

        selected_block = draw_board(
            screen,
            (title_font, small_font),
            raw_env,
            runtime_state,
            selected_slot,
            hand_slot_rects,
            edit_mode,
            edit_grid,
            edit_hand_keys,
        )

        draw_hand(screen, (big_font, tiny_font), raw_env, runtime_state, selected_slot, hand_slot_rects, edit_mode, edit_hand_keys)
        draw_sidebar(screen, (title_font, big_font, small_font, tiny_font), raw_env, runtime_state, buttons)

        if runtime_state["done"]:
            finalize_if_needed(raw_env, runtime_state)

        if runtime_state["anim_frames"] > 0:
            runtime_state["anim_frames"] -= 1
            if runtime_state["anim_frames"] == 0:
                runtime_state["last_placed_cells"] = [
                    cell
                    for cell in runtime_state["last_placed_cells"]
                    if cell[0] not in runtime_state["anim_rows"] and cell[1] not in runtime_state["anim_cols"]
                ]
            pygame.display.flip()
            clock.tick(30)
            continue

        if runtime_state["mode"] == "ai" and not runtime_state["done"]:
            if runtime_state["auto_enabled"] and (time.time() - runtime_state["last_auto_move"] > args.auto_interval):
                apply_ai_move(model, env, raw_env, runtime_state)
            elif runtime_state["step_requested"]:
                apply_ai_move(model, env, raw_env, runtime_state)
                runtime_state["step_requested"] = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
