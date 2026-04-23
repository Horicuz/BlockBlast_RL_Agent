import pygame
import time
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- SETĂRI GRAFICE MODERNE ---
CELL_SIZE = 45
MARGIN = 4
GRID_OFFSET_X = 40
GRID_OFFSET_Y = 40

WINDOW_WIDTH = 8 * CELL_SIZE + 400 
WINDOW_HEIGHT = 8 * CELL_SIZE + 220 

# Paletă de Culori
BG_COLOR = (20, 20, 24)
GRID_BG_COLOR = (30, 30, 36)
EMPTY_COLOR = (42, 42, 50)
BLOCK_COLOR = (52, 152, 219) 
BLOCK_SHADOW = (41, 128, 185) 
LAST_MOVE_COLOR = (241, 196, 15) # GALBEN PENTRU ULTIMA PIESĂ
LAST_MOVE_SHADOW = (211, 84, 0)
ANIM_COLOR = (255, 255, 255) # ALB PENTRU LINII DISTRUSE
DIM_COLOR = (60, 60, 60) 
TEXT_COLOR = (240, 240, 240)

def mask_fn(env):
    return env.valid_action_mask()

def draw_mini_block(screen, block_matrix, start_x, start_y, is_available):
    mini_cell = 22
    rows, cols = block_matrix.shape
    color = BLOCK_COLOR if is_available else DIM_COLOR
    shadow = BLOCK_SHADOW if is_available else (40, 40, 40)
    
    for r in range(rows):
        for c in range(cols):
            if block_matrix[r, c] == 1:
                rect = pygame.Rect(start_x + c*mini_cell, start_y + r*mini_cell, mini_cell-2, mini_cell-2)
                pygame.draw.rect(screen, color, rect, border_radius=4)
                edge = pygame.Rect(start_x + c*mini_cell, start_y + r*mini_cell + mini_cell - 6, mini_cell-2, 4)
                pygame.draw.rect(screen, shadow, edge, border_radius=4)

def draw_window(screen, font, small_font, env_logic, is_auto, last_action_text, last_placed_cells, anim_frames, anim_rows, anim_cols):
    screen.fill(BG_COLOR)
    
    grid_bg = pygame.Rect(GRID_OFFSET_X-MARGIN, GRID_OFFSET_Y-MARGIN, 8*CELL_SIZE+MARGIN, 8*CELL_SIZE+MARGIN)
    pygame.draw.rect(screen, GRID_BG_COLOR, grid_bg, border_radius=10)
    
    for r in range(8):
        for c in range(8):
            rect = pygame.Rect(GRID_OFFSET_X + c*CELL_SIZE, GRID_OFFSET_Y + r*CELL_SIZE, CELL_SIZE-MARGIN, CELL_SIZE-MARGIN)
            
            # Verificăm dacă blocul face parte din animația de distrugere
            is_animating = (anim_frames > 0) and (r in anim_rows or c in anim_cols)
            
            if is_animating:
                # Desenăm Alb Strălucitor
                pygame.draw.rect(screen, ANIM_COLOR, rect, border_radius=6)
            elif env_logic.grid[r, c] == 1 or (r, c) in last_placed_cells:
                # Alegem culoarea normală sau pe cea evidențiată (Galben)
                color = LAST_MOVE_COLOR if (r, c) in last_placed_cells else BLOCK_COLOR
                shadow_color = LAST_MOVE_SHADOW if (r, c) in last_placed_cells else BLOCK_SHADOW
                
                pygame.draw.rect(screen, color, rect, border_radius=6)
                shadow = pygame.Rect(GRID_OFFSET_X + c*CELL_SIZE, GRID_OFFSET_Y + r*CELL_SIZE + CELL_SIZE - 9, CELL_SIZE-MARGIN, 5)
                pygame.draw.rect(screen, shadow_color, shadow, border_radius=6)
            else:
                pygame.draw.rect(screen, EMPTY_COLOR, rect, border_radius=6)
                
    # --- PANOU SCOR ---
    x_text = GRID_OFFSET_X + 8 * CELL_SIZE + 40
    
    title = font.render("BLOCK BLAST AI", True, BLOCK_COLOR)
    screen.blit(title, (x_text, 40))
    
    stages_text = small_font.render(f"Etape (Mâini): {env_logic.stages_passed}", True, TEXT_COLOR)
    screen.blit(stages_text, (x_text, 100))
    lines_text = small_font.render(f"Linii Distruse: {env_logic.lines_destroyed}", True, TEXT_COLOR)
    screen.blit(lines_text, (x_text, 140))
    blocks_text = small_font.render(f"Piese Așezate: {env_logic.blocks_placed}", True, TEXT_COLOR)
    screen.blit(blocks_text, (x_text, 180))
    
    info_text = small_font.render(last_action_text, True, LAST_MOVE_COLOR)
    screen.blit(info_text, (x_text, 240))

    # --- STATUS CONTROL ---
    status_y = GRID_OFFSET_Y + 8 * CELL_SIZE - 60
    if is_auto:
        status_msg = small_font.render("▶ AUTO PORNIT (Apasă 'A' să oprești)", True, (46, 204, 113))
    else:
        status_msg = small_font.render("⏸ MANUAL (Apasă SPACE o dată)", True, (241, 196, 15))
    screen.blit(status_msg, (x_text, status_y))

    # --- PIESELE DIN MÂNĂ ---
    hand_y = GRID_OFFSET_Y + 8 * CELL_SIZE + 30
    for i, block in enumerate(env_logic.hand):
        block_x = GRID_OFFSET_X + (i * 120)
        block_y = hand_y + 35
        draw_mini_block(screen, block, block_x, block_y, env_logic.available[i])

    pygame.display.flip()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Block Blast AI - Test Visualizer")
    
    font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    small_font = pygame.font.SysFont("Segoe UI", 20, bold=True)

    raw_env = BlockBlastEnv()
    env = ActionMasker(raw_env, mask_fn)
    
    try:
        model = MaskablePPO.load("block_blast_ai_v1", env=env)
    except FileNotFoundError:
        print("Model nesalvat! Rulează train.py mai întâi.")
        pygame.quit()
        exit()

    obs, _ = env.reset()
    done = False
    
    is_auto = False
    step_requested = False # Pentru a declanșa mutarea manuală cu Space
    
    last_action_text = "Aștept start..."
    last_placed_cells = []
    
    # Variabile pentru Animație
    anim_frames = 0
    anim_rows = []
    anim_cols = []
    
    clock = pygame.time.Clock()
    last_auto_move = time.time()

    running = True
    while running:
        # 1. VERIFICĂM BUTOANELE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    is_auto = not is_auto # Toggle Auto Mode
                if event.key == pygame.K_SPACE and not is_auto:
                    step_requested = True # Face o mutare la apăsare
        
        # 2. DESENĂM ECRANUL
        draw_window(screen, font, small_font, raw_env.game, is_auto, last_action_text, last_placed_cells, anim_frames, anim_rows, anim_cols)

        # Scădem frame-urile de animație ca să dispară blițul alb
        if anim_frames > 0:
            anim_frames -= 1
            # Dacă animația tocmai s-a terminat, golim piesele marcate ca să dispară linia distrusă de pe ecran
            if anim_frames == 0:
                last_placed_cells = [cell for cell in last_placed_cells if cell[0] not in anim_rows and cell[1] not in anim_cols]
            clock.tick(30)
            continue # Blocăm jocul să nu mute cât timp e animație

        # 3. VERIFICĂM DACĂ TREBUIE SĂ FACEM MUTAREA
        time_to_move = is_auto and (time.time() - last_auto_move > 1.0)
        
        if (time_to_move or step_requested) and not done:
            action_masks = mask_fn(raw_env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Calculăm UNDE se pune piesa pentru a o face Galbenă
            hand_idx = action // 64
            r = (action % 64) // 8
            c = (action % 64) % 8
            block_matrix = raw_env.game.hand[hand_idx]
            
            last_placed_cells = []
            for b_r in range(block_matrix.shape[0]):
                for b_c in range(block_matrix.shape[1]):
                    if block_matrix[b_r, b_c] == 1:
                        last_placed_cells.append((r + b_r, c + b_c))
                        
            last_action_text = f"► A pus Piesa {hand_idx + 1} la R:{r} C:{c}"
            
            # FACEM MUTAREA EFECTIV
            obs, reward, done, truncated, info = env.step(action)
            
            # Dacă s-au distrus linii, pornim animația albă!
            if info.get("anim_rows") or info.get("anim_cols"):
                anim_rows = info["anim_rows"]
                anim_cols = info["anim_cols"]
                anim_frames = 15 # Durează 15 frame-uri (jumătate de secundă)
            
            step_requested = False
            last_auto_move = time.time()
            
        if done:
            game_over_text = font.render("GAME OVER!", True, (231, 76, 60))
            screen.blit(game_over_text, (GRID_OFFSET_X + 8 * CELL_SIZE + 40, 300))
            pygame.display.flip()
            
        clock.tick(30)

    pygame.quit()