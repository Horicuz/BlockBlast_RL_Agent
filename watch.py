import argparse
import random
import time

import numpy as np
import pygame
from blocks import SHAPES
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- SETARI GRAFICE ---
CELL_SIZE = 45
MARGIN = 4
GRID_OFFSET_X = 40
GRID_OFFSET_Y = 40

WINDOW_WIDTH = 8 * CELL_SIZE + 400
WINDOW_HEIGHT = 8 * CELL_SIZE + 300
AUTO_MOVE_INTERVAL = 0.8

# Culori
BG_COLOR = (20, 20, 24)
GRID_BG_COLOR = (30, 30, 36)
EMPTY_COLOR = (42, 42, 50)
BLOCK_COLOR = (52, 152, 219)
BLOCK_SHADOW = (41, 128, 185)
LAST_MOVE_COLOR = (241, 196, 15)
LAST_MOVE_SHADOW = (211, 84, 0)
ANIM_COLOR = (255, 255, 255)
DIM_COLOR = (60, 60, 60)
TEXT_COLOR = (240, 240, 240)
BUTTON_BG = (52, 73, 94)
BUTTON_BG_ACTIVE = (41, 128, 185)
BUTTON_TEXT = (245, 245, 245)
EDIT_HINT_COLOR = (241, 196, 15)

SHAPE_KEYS = list(SHAPES.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Watch and inspect a trained Block Blast model")
    parser.add_argument("--model-path", default="block_blast_ai_v4", help="Model path, cu sau fara .zip")
    return parser.parse_args()


def mask_fn(env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()

    raise AttributeError("valid_action_mask() nu este disponibil pe mediul curent")


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
        "last_action_text": runtime_state["last_action_text"],
        "last_placed_cells": list(runtime_state["last_placed_cells"]),
    }


def apply_snapshot(raw_env, runtime_state, snapshot):
    restore_game_state(raw_env.game, snapshot["game_state"])
    random.setstate(snapshot["random_state"])

    runtime_state["obs"] = raw_env._get_obs()
    runtime_state["done"] = snapshot["done"]
    runtime_state["last_action_text"] = snapshot["last_action_text"]
    runtime_state["last_placed_cells"] = list(snapshot["last_placed_cells"])
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []


def push_history(raw_env, runtime_state):
    history_index = runtime_state["history_index"]
    history = runtime_state["history"]

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


def compute_placed_cells(block_matrix, row, col):
    cells = []
    for b_r in range(block_matrix.shape[0]):
        for b_c in range(block_matrix.shape[1]):
            if block_matrix[b_r, b_c] == 1:
                cells.append((row + b_r, col + b_c))
    return cells


def execute_model_move(model, env, raw_env, runtime_state):
    if runtime_state["done"]:
        return

    action_masks = mask_fn(raw_env)
    if not np.any(action_masks):
        runtime_state["done"] = True
        runtime_state["last_action_text"] = "Nicio mutare valida disponibila."
        return

    action, _ = model.predict(runtime_state["obs"], action_masks=action_masks, deterministic=True)
    action = int(action)

    hand_idx, row, col = decode_action(action)
    block_matrix = raw_env.game.hand[hand_idx]
    runtime_state["last_placed_cells"] = compute_placed_cells(block_matrix, row, col)

    runtime_state["obs"], reward, runtime_state["done"], _, info = env.step(action)
    runtime_state["last_action_text"] = f"> Piesa {hand_idx + 1} la R:{row} C:{col} | Reward: {reward:.1f}"

    runtime_state["anim_rows"] = info.get("anim_rows", [])
    runtime_state["anim_cols"] = info.get("anim_cols", [])
    runtime_state["anim_frames"] = 15 if (runtime_state["anim_rows"] or runtime_state["anim_cols"]) else 0
    runtime_state["last_auto_move"] = time.time()

    push_history(raw_env, runtime_state)


def get_hand_slot_rects():
    hand_y = GRID_OFFSET_Y + 8 * CELL_SIZE + 30
    rects = []
    for i in range(3):
        x = GRID_OFFSET_X + (i * 120) - 8
        rects.append(pygame.Rect(x, hand_y + 28, 92, 92))
    return rects


def build_buttons():
    x = GRID_OFFSET_X + 8 * CELL_SIZE + 40
    y = 290
    w = 150
    h = 36
    gap_x = 12
    gap_y = 10

    keys = [
        "auto", "step",
        "back", "forward",
        "new", "solve",
        "edit", "apply",
    ]

    buttons = {}
    for idx, key in enumerate(keys):
        row = idx // 2
        col = idx % 2
        rect = pygame.Rect(x + col * (w + gap_x), y + row * (h + gap_y), w, h)
        buttons[key] = rect
    return buttons


def draw_button(screen, font, rect, label, active=False):
    bg = BUTTON_BG_ACTIVE if active else BUTTON_BG
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, (180, 180, 180), rect, width=1, border_radius=8)
    txt = font.render(label, True, BUTTON_TEXT)
    txt_rect = txt.get_rect(center=rect.center)
    screen.blit(txt, txt_rect)


def draw_mini_block(screen, block_matrix, start_x, start_y, is_available):
    mini_cell = 22
    rows, cols = block_matrix.shape
    color = BLOCK_COLOR if is_available else DIM_COLOR
    shadow = BLOCK_SHADOW if is_available else (40, 40, 40)

    for r in range(rows):
        for c in range(cols):
            if block_matrix[r, c] == 1:
                rect = pygame.Rect(start_x + c * mini_cell, start_y + r * mini_cell, mini_cell - 2, mini_cell - 2)
                pygame.draw.rect(screen, color, rect, border_radius=4)
                edge = pygame.Rect(start_x + c * mini_cell, start_y + r * mini_cell + mini_cell - 6, mini_cell - 2, 4)
                pygame.draw.rect(screen, shadow, edge, border_radius=4)


def draw_window(
    screen,
    font,
    small_font,
    button_font,
    env_logic,
    runtime_state,
    buttons,
    edit_mode,
    edit_grid,
    edit_hand_keys,
    selected_slot,
    hand_slot_rects,
):
    screen.fill(BG_COLOR)

    grid_bg = pygame.Rect(GRID_OFFSET_X - MARGIN, GRID_OFFSET_Y - MARGIN, 8 * CELL_SIZE + MARGIN, 8 * CELL_SIZE + MARGIN)
    pygame.draw.rect(screen, GRID_BG_COLOR, grid_bg, border_radius=10)

    grid_to_draw = edit_grid if edit_mode else env_logic.grid

    for r in range(8):
        for c in range(8):
            rect = pygame.Rect(GRID_OFFSET_X + c * CELL_SIZE, GRID_OFFSET_Y + r * CELL_SIZE, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)

            is_animating = (not edit_mode) and (runtime_state["anim_frames"] > 0) and (
                r in runtime_state["anim_rows"] or c in runtime_state["anim_cols"]
            )

            if is_animating:
                pygame.draw.rect(screen, ANIM_COLOR, rect, border_radius=6)
            elif grid_to_draw[r, c] == 1 or ((r, c) in runtime_state["last_placed_cells"] and not edit_mode):
                is_last_cell = (r, c) in runtime_state["last_placed_cells"] and not edit_mode
                color = LAST_MOVE_COLOR if is_last_cell else BLOCK_COLOR
                shadow_color = LAST_MOVE_SHADOW if is_last_cell else BLOCK_SHADOW

                pygame.draw.rect(screen, color, rect, border_radius=6)
                shadow = pygame.Rect(
                    GRID_OFFSET_X + c * CELL_SIZE,
                    GRID_OFFSET_Y + r * CELL_SIZE + CELL_SIZE - 9,
                    CELL_SIZE - MARGIN,
                    5,
                )
                pygame.draw.rect(screen, shadow_color, shadow, border_radius=6)
            else:
                pygame.draw.rect(screen, EMPTY_COLOR, rect, border_radius=6)

    x_text = GRID_OFFSET_X + 8 * CELL_SIZE + 40

    title = font.render("BLOCK BLAST AI", True, BLOCK_COLOR)
    screen.blit(title, (x_text, 40))

    stages_text = small_font.render(f"Etape (Maini): {env_logic.stages_passed}", True, TEXT_COLOR)
    screen.blit(stages_text, (x_text, 100))
    lines_text = small_font.render(f"Linii Distruse: {env_logic.lines_destroyed}", True, TEXT_COLOR)
    screen.blit(lines_text, (x_text, 140))
    blocks_text = small_font.render(f"Piese Asezate: {env_logic.blocks_placed}", True, TEXT_COLOR)
    screen.blit(blocks_text, (x_text, 180))

    info_text = small_font.render(runtime_state["last_action_text"], True, LAST_MOVE_COLOR)
    screen.blit(info_text, (x_text, 240))

    if runtime_state["is_auto"]:
        status_msg = small_font.render("AUTO: pornit", True, (46, 204, 113))
    else:
        status_msg = small_font.render("AUTO: oprit", True, (241, 196, 15))
    screen.blit(status_msg, (x_text, 260))

    for key, rect in buttons.items():
        if key == "auto":
            label = "Auto ON/OFF"
            active = runtime_state["is_auto"]
        elif key == "step":
            label = "Step"
            active = False
        elif key == "back":
            label = "Back"
            active = False
        elif key == "forward":
            label = "Forward"
            active = False
        elif key == "new":
            label = "New Game"
            active = False
        elif key == "solve":
            label = "Solve Move"
            active = False
        elif key == "edit":
            label = "Edit Mode"
            active = edit_mode
        else:
            label = "Apply Setup"
            active = False
        draw_button(screen, button_font, rect, label, active=active)

    if edit_mode:
        hint_1 = small_font.render("EDIT MODE: click pe tabla ca sa pui/scoti blocuri.", True, EDIT_HINT_COLOR)
        hint_2 = small_font.render("Taste 1/2/3 select slot, stanga/dreapta schimba piesa.", True, EDIT_HINT_COLOR)
        screen.blit(hint_1, (x_text, 470))
        screen.blit(hint_2, (x_text, 495))

        for i, key in enumerate(edit_hand_keys):
            color = LAST_MOVE_COLOR if i == selected_slot else TEXT_COLOR
            piece_label = small_font.render(f"Slot {i+1}: {key}", True, color)
            screen.blit(piece_label, (x_text, 525 + i * 24))

    if runtime_state["done"]:
        game_over_text = font.render("GAME OVER!", True, (231, 76, 60))
        screen.blit(game_over_text, (x_text, 560))

    hand_y = GRID_OFFSET_Y + 8 * CELL_SIZE + 30
    for i in range(3):
        block_x = GRID_OFFSET_X + (i * 120)
        block_y = hand_y + 35

        if edit_mode:
            block = SHAPES[edit_hand_keys[i]]
            is_available = True
        else:
            block = env_logic.hand[i]
            is_available = env_logic.available[i]

        draw_mini_block(screen, block, block_x, block_y, is_available)

        if edit_mode:
            border_color = LAST_MOVE_COLOR if i == selected_slot else (130, 130, 130)
            pygame.draw.rect(screen, border_color, hand_slot_rects[i], width=2, border_radius=8)

    pygame.display.flip()


def reset_runtime(env, raw_env, runtime_state):
    runtime_state["obs"], _ = env.reset()
    runtime_state["done"] = False
    runtime_state["is_auto"] = False
    runtime_state["step_requested"] = False
    runtime_state["solve_requested"] = False
    runtime_state["last_action_text"] = "Astept start..."
    runtime_state["last_placed_cells"] = []
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []
    runtime_state["last_auto_move"] = time.time()
    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["history_index"] = 0


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
    runtime_state["is_auto"] = False
    runtime_state["last_action_text"] = "Setup aplicat. Poti folosi Solve/Step/Auto."
    runtime_state["last_placed_cells"] = []
    runtime_state["anim_frames"] = 0
    runtime_state["anim_rows"] = []
    runtime_state["anim_cols"] = []
    runtime_state["last_auto_move"] = time.time()
    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["history_index"] = 0


if __name__ == "__main__":
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Block Blast AI - Test Visualizer")

    font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    small_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
    button_font = pygame.font.SysFont("Segoe UI", 16, bold=True)

    raw_env = BlockBlastEnv()
    env = ActionMasker(raw_env, mask_fn)

    model_path = args.model_path
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    try:
        print(f"Incarcam modelul: {model_path}.zip")
        model = MaskablePPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Modelul {model_path}.zip nu exista. Ruleaza train.py mai intai.")
        pygame.quit()
        raise SystemExit(1)

    obs, _ = env.reset()
    runtime_state = {
        "obs": obs,
        "done": False,
        "is_auto": False,
        "step_requested": False,
        "solve_requested": False,
        "last_action_text": "Astept start...",
        "last_placed_cells": [],
        "anim_frames": 0,
        "anim_rows": [],
        "anim_cols": [],
        "last_auto_move": time.time(),
        "history": [],
        "history_index": -1,
    }

    runtime_state["history"] = [capture_snapshot(raw_env, runtime_state)]
    runtime_state["history_index"] = 0

    buttons = build_buttons()
    hand_slot_rects = get_hand_slot_rects()

    edit_mode = False
    edit_grid = raw_env.game.grid.copy()
    edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]
    selected_slot = 0

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    runtime_state["is_auto"] = not runtime_state["is_auto"]
                    if runtime_state["is_auto"]:
                        edit_mode = False

                elif event.key == pygame.K_SPACE:
                    runtime_state["step_requested"] = True

                elif edit_mode:
                    if event.key == pygame.K_1:
                        selected_slot = 0
                    elif event.key == pygame.K_2:
                        selected_slot = 1
                    elif event.key == pygame.K_3:
                        selected_slot = 2
                    elif event.key == pygame.K_LEFT:
                        cycle_piece_key(edit_hand_keys, selected_slot, -1)
                    elif event.key == pygame.K_RIGHT:
                        cycle_piece_key(edit_hand_keys, selected_slot, 1)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                button_clicked = None

                for key, rect in buttons.items():
                    if rect.collidepoint(mouse_pos):
                        button_clicked = key
                        break

                if button_clicked == "auto":
                    runtime_state["is_auto"] = not runtime_state["is_auto"]
                    if runtime_state["is_auto"]:
                        edit_mode = False

                elif button_clicked == "step":
                    runtime_state["step_requested"] = True

                elif button_clicked == "back":
                    runtime_state["is_auto"] = False
                    if runtime_state["history_index"] > 0:
                        runtime_state["history_index"] -= 1
                        apply_snapshot(raw_env, runtime_state, runtime_state["history"][runtime_state["history_index"]])

                elif button_clicked == "forward":
                    runtime_state["is_auto"] = False
                    if runtime_state["history_index"] < len(runtime_state["history"]) - 1:
                        runtime_state["history_index"] += 1
                        apply_snapshot(raw_env, runtime_state, runtime_state["history"][runtime_state["history_index"]])

                elif button_clicked == "new":
                    reset_runtime(env, raw_env, runtime_state)
                    edit_grid = raw_env.game.grid.copy()
                    edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]

                elif button_clicked == "solve":
                    runtime_state["solve_requested"] = True

                elif button_clicked == "edit":
                    runtime_state["is_auto"] = False
                    edit_mode = not edit_mode
                    if edit_mode:
                        edit_grid = raw_env.game.grid.copy()
                        edit_hand_keys = [shape_key_from_matrix(block) for block in raw_env.game.hand]

                elif button_clicked == "apply":
                    apply_editor_setup(raw_env, runtime_state, edit_grid, edit_hand_keys)
                    edit_mode = False

                elif edit_mode:
                    board_x, board_y = mouse_pos
                    if (
                        GRID_OFFSET_X <= board_x < GRID_OFFSET_X + 8 * CELL_SIZE
                        and GRID_OFFSET_Y <= board_y < GRID_OFFSET_Y + 8 * CELL_SIZE
                    ):
                        c = (board_x - GRID_OFFSET_X) // CELL_SIZE
                        r = (board_y - GRID_OFFSET_Y) // CELL_SIZE
                        edit_grid[r, c] = 0 if edit_grid[r, c] == 1 else 1

                    for idx, slot_rect in enumerate(hand_slot_rects):
                        if slot_rect.collidepoint(mouse_pos):
                            selected_slot = idx

        draw_window(
            screen,
            font,
            small_font,
            button_font,
            raw_env.game,
            runtime_state,
            buttons,
            edit_mode,
            edit_grid,
            edit_hand_keys,
            selected_slot,
            hand_slot_rects,
        )

        if runtime_state["anim_frames"] > 0:
            runtime_state["anim_frames"] -= 1
            if runtime_state["anim_frames"] == 0:
                runtime_state["last_placed_cells"] = [
                    cell
                    for cell in runtime_state["last_placed_cells"]
                    if cell[0] not in runtime_state["anim_rows"] and cell[1] not in runtime_state["anim_cols"]
                ]
            clock.tick(30)
            continue

        time_to_move = runtime_state["is_auto"] and (time.time() - runtime_state["last_auto_move"] > AUTO_MOVE_INTERVAL)
        if (time_to_move or runtime_state["step_requested"] or runtime_state["solve_requested"]) and not edit_mode:
            execute_model_move(model, env, raw_env, runtime_state)
            runtime_state["step_requested"] = False
            runtime_state["solve_requested"] = False

        clock.tick(30)

    pygame.quit()
