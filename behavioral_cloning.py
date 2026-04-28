import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from env import ActionAwareCNNExtractor, BlockBlastEnv, CustomCNNExtractor, CustomCNNExtractorV2
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
GRID_SIZE = 4
DEFAULT_REWARD_CONFIG = {'placement_reward': 0.0, 'line_clear_scale': 28.0, 'line_clear_bonus': 1.5, 'stage_complete_reward': 0.0, 'no_line_penalty': 0.0, 'game_over_penalty': 90.0, 'game_over_early_weight': 0.0, 'contact_reward_scale': 24.0, 'contact_reward_power': 1.15, 'contact_reward_threshold': 0.4, 'contact_penalty_scale': 8.0, 'complexity_simple_prob': 0.78, 'complexity_medium_prob': 0.18, 'complexity_hard_prob': 0.04}

def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def normalize_model_path(path):
    if path.endswith('.zip'):
        return path[:-4]
    return path

def mask_fn(env):
    if hasattr(env, 'valid_action_mask'):
        return env.valid_action_mask()
    unwrapped = getattr(env, 'unwrapped', None)
    if unwrapped is not None and hasattr(unwrapped, 'valid_action_mask'):
        return unwrapped.valid_action_mask()
    raise AttributeError('valid_action_mask() is not available on the current environment')

def decode_action(action):
    hand_idx = action // 16
    remainder = action % 16
    row = remainder // 4
    col = remainder % 4
    return (hand_idx, row, col)

def action_from_selection(hand_index, row, col):
    return hand_index * 16 + row * 4 + col

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
    return (next_grid, len(full_rows) + len(full_cols))

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
                if 0 <= neighbor_block_row < block_h and 0 <= neighbor_block_col < block_w and (block[neighbor_block_row, neighbor_block_col] == 1):
                    continue
                external_edges += 1
                neighbor_row = board_row + d_row
                neighbor_col = board_col + d_col
                if neighbor_row < 0 or neighbor_row >= GRID_SIZE or neighbor_col < 0 or (neighbor_col >= GRID_SIZE) or (previous_grid[neighbor_row, neighbor_col] == 1):
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
                    if 0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE and (not visited[next_row, next_col]) and (grid[next_row, next_col] == 0):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            region_count += 1
            largest_region = max(largest_region, region_size)
    return (largest_region, region_count)

def heuristic_board_score(grid, hand, available):
    valid_actions_after = len(valid_actions_for_state(grid, hand, available))
    rows = grid.sum(axis=1)
    cols = grid.sum(axis=0)
    line_potential = float(np.sum((rows / GRID_SIZE) ** 3) + np.sum((cols / GRID_SIZE) ** 3))
    largest_empty_region, empty_regions = empty_region_stats(grid)
    filled_cells = int(grid.sum())
    return valid_actions_after * 0.65 + line_potential * 5.0 + largest_empty_region * 0.25 - empty_regions * 0.5 - max(filled_cells - 44, 0) * 0.4

def heuristic_evaluate_action(grid, hand, available, action):
    hand_idx, row, col = decode_action(action)
    block = hand[hand_idx]
    next_grid, lines_cleared = apply_action_to_grid(grid, block, row, col)
    next_available = list(available)
    next_available[hand_idx] = False
    stage_completed = not any(next_available)
    contact_ratio = contact_ratio_on_grid(grid, block, row, col)
    contact_bonus = max(contact_ratio - 0.4, 0.0) * 28.0
    contact_penalty = max(0.4 - contact_ratio, 0.0) * 18.0
    valid_actions_after = len(valid_actions_for_state(next_grid, hand, next_available))
    no_move_penalty = 160.0 if not stage_completed and valid_actions_after == 0 else 0.0
    score = lines_cleared * 60.0 + lines_cleared ** 2 * 25.0 + contact_bonus - contact_penalty + heuristic_board_score(next_grid, hand, next_available) + (8.0 if stage_completed else 0.0) - no_move_penalty
    return {'score': float(score), 'lines_cleared': float(lines_cleared), 'contact_ratio': float(contact_ratio), 'valid_actions_after': float(valid_actions_after)}

def choose_heuristic_action(raw_env):
    grid = raw_env.game.grid
    hand = raw_env.game.hand
    available = raw_env.game.available
    actions = valid_actions_for_state(grid, hand, available)
    if not actions:
        return (None, None)
    ranked = []
    for action in actions:
        result = heuristic_evaluate_action(grid, hand, available, action)
        ranked.append((result['score'], action, result))
    ranked.sort(reverse=True, key=lambda item: item[0])
    best_score, best_action, best_result = ranked[0]
    best_result['score'] = best_score
    return (best_action, best_result)

def build_reward_config(reward_config=None, shape_pool='all', hand_generator='adaptive_playable'):
    merged = dict(DEFAULT_REWARD_CONFIG)
    if reward_config:
        merged.update(reward_config)
    merged['shape_pool'] = shape_pool
    merged['hand_generator'] = hand_generator
    return merged

def make_env(reward_config=None, shape_pool='all', hand_generator='adaptive_playable', fixed_game_seed=None):
    resolved_reward_config = build_reward_config(reward_config, shape_pool=shape_pool, hand_generator=hand_generator)
    env = BlockBlastEnv(reward_config=resolved_reward_config, apply_hole_penalty=bool(resolved_reward_config.get('apply_hole_penalty', False)), fixed_game_seed=fixed_game_seed, shape_pool=shape_pool, hand_generator=hand_generator)
    return ActionMasker(env, mask_fn)

def collect_expert_data(output_path, num_episodes, reward_config=None, shape_pool='all', hand_generator='adaptive_playable', fixed_game_seed=None, fixed_game_seeds=None, max_steps_per_episode=5000, base_reset_seed=10000):
    archive_path = output_path if output_path.endswith('.npz') else f'{output_path}.npz'
    print('[BC] collect_expert_data')
    print(f'[BC] reading environment state from BlockBlastEnv with shape_pool={shape_pool}, hand_generator={hand_generator}')
    print(f'[BC] expert actions come from the built-in heuristic in watch.py')
    print(f'[BC] saving expert dataset to: {archive_path}')
    print(f'[BC] planned episodes: {num_episodes}, max_steps_per_episode: {max_steps_per_episode}')
    raw_env = BlockBlastEnv(reward_config=build_reward_config(reward_config, shape_pool=shape_pool, hand_generator=hand_generator), apply_hole_penalty=bool((reward_config or {}).get('apply_hole_penalty', False)), fixed_game_seed=fixed_game_seed, shape_pool=shape_pool, hand_generator=hand_generator)
    env = ActionMasker(raw_env, mask_fn)
    boards = []
    valid_actions_list = []
    hands = []
    available_list = []
    actions = []
    collection_start = time.perf_counter()
    progress_every = max(1, num_episodes // 20)
    try:
        for episode_idx in range(num_episodes):
            episode_start = time.perf_counter()
            if fixed_game_seeds:
                raw_env.fixed_game_seed = fixed_game_seeds[episode_idx % len(fixed_game_seeds)]
            elif fixed_game_seed is not None:
                raw_env.fixed_game_seed = fixed_game_seed
            obs, _ = env.reset(seed=base_reset_seed + episode_idx)
            done = False
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                action, _ = choose_heuristic_action(raw_env)
                if action is None:
                    break
                boards.append(np.asarray(obs['board'], dtype=np.uint8).copy())
                valid_actions_list.append(np.asarray(obs['valid_actions'], dtype=np.uint8).copy())
                hands.append(np.asarray(obs['hand'], dtype=np.uint8).copy())
                available_list.append(np.asarray(obs['available'], dtype=np.uint8).copy())
                actions.append(int(action))
                obs, _, done, _, _ = env.step(int(action))
                step_count += 1
            if (episode_idx + 1) % progress_every == 0 or episode_idx + 1 == num_episodes:
                elapsed = time.perf_counter() - collection_start
                episodes_done = episode_idx + 1
                episodes_per_sec = episodes_done / max(elapsed, 1e-06)
                samples_per_sec = len(actions) / max(elapsed, 1e-06)
                estimated_total = elapsed / episodes_done * num_episodes
                remaining = max(estimated_total - elapsed, 0.0)
                print(f'[BC] collect {episodes_done}/{num_episodes} | last_episode_steps={step_count} | last_stages={raw_env.game.stages_passed} | samples={len(actions)} | eps/s={episodes_per_sec:.2f} | samples/s={samples_per_sec:.1f} | elapsed={elapsed / 60.0:.1f}m | eta={remaining / 60.0:.1f}m')
    finally:
        env.close()
    if not actions:
        raise RuntimeError('No expert samples were collected. Check the heuristic or environment configuration.')
    archive_path = output_path if output_path.endswith('.npz') else f'{output_path}.npz'
    ensure_parent_dir(archive_path)
    np.savez_compressed(archive_path, board=np.asarray(boards, dtype=np.uint8), valid_actions=np.asarray(valid_actions_list, dtype=np.uint8), hand=np.asarray(hands, dtype=np.uint8), available=np.asarray(available_list, dtype=np.uint8), actions=np.asarray(actions, dtype=np.int16), num_episodes=np.asarray(num_episodes, dtype=np.int32), num_samples=np.asarray(len(actions), dtype=np.int32))
    collection_elapsed = time.perf_counter() - collection_start
    print(f'[BC] expert dataset written to {archive_path}')
    print(f'[BC] collection finished in {collection_elapsed / 60.0:.1f}m with {len(actions)} samples')
    return archive_path

class ExpertDataset(Dataset):

    def __init__(self, archive_path):
        with np.load(archive_path, allow_pickle=False) as data:
            self.boards = data['board'].copy()
            self.valid_actions = data['valid_actions'].copy()
            self.hands = data['hand'].copy()
            self.available = data['available'].copy()
            self.actions = data['actions'].astype(np.int64).copy()
        lengths = {len(self.boards), len(self.valid_actions), len(self.hands), len(self.available), len(self.actions)}
        if len(lengths) != 1:
            raise ValueError(f'Expert archive contains mismatched lengths: {lengths}')

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        observation = {'board': torch.from_numpy(self.boards[index]).float(), 'valid_actions': torch.from_numpy(self.valid_actions[index]).float(), 'hand': torch.from_numpy(self.hands[index]).float(), 'available': torch.from_numpy(self.available[index]).float()}
        action = torch.tensor(self.actions[index], dtype=torch.long)
        return (observation, action)

def split_expert_dataset(dataset, validation_split=0.1, seed=0):
    if validation_split <= 0.0 or len(dataset) < 2:
        return (Subset(dataset, list(range(len(dataset)))), None)
    validation_size = int(round(len(dataset) * validation_split))
    validation_size = max(1, min(validation_size, len(dataset) - 1))
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    validation_indices = indices[:validation_size].tolist()
    train_indices = indices[validation_size:].tolist()
    return (Subset(dataset, train_indices), Subset(dataset, validation_indices))

def evaluate_behavioral_cloning(policy, loader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    policy.eval()
    with torch.no_grad():
        for observations, expert_actions in loader:
            observations = move_batch_to_device(observations, device)
            expert_actions = expert_actions.to(device)
            logits = policy_logits(policy, observations)
            loss = criterion(logits, expert_actions)
            predictions = torch.argmax(logits, dim=1)
            batch_size = int(expert_actions.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == expert_actions).sum().item())
            total_examples += batch_size
    mean_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return (mean_loss, accuracy)

def parse_net_arch(net_arch):
    if isinstance(net_arch, str):
        return [int(value.strip()) for value in net_arch.split(',') if value.strip()]
    return [int(value) for value in net_arch]

def build_policy_kwargs(cnn_arch, features_dim, net_arch):
    extractor_class = {'base': CustomCNNExtractor, 'v2': CustomCNNExtractorV2, 'actionaware': ActionAwareCNNExtractor}[cnn_arch]
    parsed_net_arch = parse_net_arch(net_arch)
    return dict(features_extractor_class=extractor_class, features_extractor_kwargs=dict(features_dim=features_dim), net_arch=dict(pi=parsed_net_arch, vf=parsed_net_arch), activation_fn=nn.ReLU)

def create_bc_model(device='auto', reward_config=None, shape_pool='all', hand_generator='adaptive_playable', fixed_game_seed=None, cnn_arch='actionaware', features_dim=256, net_arch=(256, 256), learning_rate=0.0003, gamma=0.99, ent_coef=0.0, n_steps=1024, batch_size=1024, n_epochs=4):
    env = make_env(reward_config=reward_config, shape_pool=shape_pool, hand_generator=hand_generator, fixed_game_seed=fixed_game_seed)
    model = MaskablePPO('MultiInputPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, device=device, policy_kwargs=build_policy_kwargs(cnn_arch, features_dim, net_arch))
    return (model, env)

def policy_logits(policy, observations):
    features = policy.extract_features(observations)
    latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)

def move_batch_to_device(observations, device):
    return {key: value.to(device) for key, value in observations.items()}

def pretrain_behavioral_cloning(data_path, model_path, reward_config=None, device='auto', shape_pool='all', hand_generator='adaptive_playable', fixed_game_seed=None, cnn_arch='actionaware', features_dim=256, net_arch=(256, 256), learning_rate=0.0003, batch_size=256, epochs=50, validation_split=0.1, target_accuracy=0.95, min_epochs=5, seed=0, shuffle=True, num_workers=0, pin_memory=False, log_dir='./tensorboard_sweeps/bc_logs', tb_log_name='PPO_BlockBlast_BC'):
    print('[BC] pretrain_behavioral_cloning')
    print(f'[BC] loading expert dataset from: {data_path}')
    print(f'[BC] model checkpoint will be saved to: {normalize_model_path(model_path)}.zip')
    print(f'[BC] using cnn_arch={cnn_arch}, features_dim={features_dim}, net_arch={net_arch}')
    print(f'[BC] target validation accuracy={target_accuracy:.2f}, validation_split={validation_split:.2f}, min_epochs={min_epochs}')
    tb_run_dir = os.path.join(log_dir, tb_log_name)
    print(f'[BC] TensorBoard logs will be written to: {tb_run_dir}')
    dataset = ExpertDataset(data_path)
    train_dataset, validation_dataset = split_expert_dataset(dataset, validation_split=validation_split, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = None
    if validation_dataset is not None and len(validation_dataset) > 0:
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    model, env = create_bc_model(device=device, reward_config=reward_config, shape_pool=shape_pool, hand_generator=hand_generator, fixed_game_seed=fixed_game_seed, cnn_arch=cnn_arch, features_dim=features_dim, net_arch=net_arch, learning_rate=learning_rate)
    torch_device = torch.device(device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.policy.to(torch_device)
    model.policy.train()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_state_dict = None
    train_start = time.perf_counter()
    writer = SummaryWriter(log_dir=tb_run_dir)
    try:
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            model.policy.train()
            for observations, expert_actions in train_loader:
                observations = move_batch_to_device(observations, torch_device)
                expert_actions = expert_actions.to(torch_device)
                logits = policy_logits(model.policy, observations)
                loss = criterion(logits, expert_actions)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    epoch_correct += int((predictions == expert_actions).sum().item())
                    epoch_total += int(expert_actions.shape[0])
                    epoch_loss += float(loss.item()) * int(expert_actions.shape[0])
            mean_loss = epoch_loss / max(epoch_total, 1)
            accuracy = epoch_correct / max(epoch_total, 1)
            validation_loss = float('nan')
            validation_accuracy = float('nan')
            if validation_loader is not None:
                validation_loss, validation_accuracy = evaluate_behavioral_cloning(model.policy, validation_loader, torch_device)
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
            print(f'[BC] epoch {epoch + 1}/{epochs} | train_loss={mean_loss:.4f} | train_acc={accuracy:.4f} | val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f} | samples={epoch_total} | epoch_time={time.perf_counter() - epoch_start:.1f}s | elapsed={(time.perf_counter() - train_start) / 60.0:.1f}m')
            writer.add_scalar('Loss/train', mean_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', accuracy, epoch + 1)
            writer.add_scalar('Loss/val', validation_loss, epoch + 1)
            writer.add_scalar('Accuracy/val', validation_accuracy, epoch + 1)
            writer.flush()
            if validation_loader is not None and epoch + 1 >= min_epochs and (validation_accuracy >= target_accuracy):
                print(f'[BC] target validation accuracy reached: {validation_accuracy:.4f} >= {target_accuracy:.4f}. Stopping early and saving the best policy.')
                break
        if best_state_dict is not None:
            model.policy.load_state_dict(best_state_dict)
        ensure_parent_dir(model_path)
        model.save(normalize_model_path(model_path))
        total_elapsed = time.perf_counter() - train_start
        print(f'[BC] pre-trained SB3 model saved to {normalize_model_path(model_path)}.zip')
        print(f'[BC] pretraining finished in {total_elapsed / 60.0:.1f}m')
    finally:
        writer.close()
        env.close()
    return model

def collect_and_pretrain(data_path, model_path, num_episodes, reward_config=None, device='auto', shape_pool='all', hand_generator='adaptive_playable', fixed_game_seed=None, fixed_game_seeds=None, max_steps_per_episode=5000, cnn_arch='actionaware', features_dim=256, net_arch=(256, 256), learning_rate=0.0003, batch_size=256, epochs=50, validation_split=0.1, target_accuracy=0.95, min_epochs=5, seed=0, log_dir='./tensorboard_sweeps/bc_logs', tb_log_name='PPO_BlockBlast_BC'):
    archive_path = collect_expert_data(output_path=data_path, num_episodes=num_episodes, reward_config=reward_config, shape_pool=shape_pool, hand_generator=hand_generator, fixed_game_seed=fixed_game_seed, fixed_game_seeds=fixed_game_seeds, max_steps_per_episode=max_steps_per_episode)
    return pretrain_behavioral_cloning(data_path=archive_path, model_path=model_path, reward_config=reward_config, device=device, shape_pool=shape_pool, hand_generator=hand_generator, fixed_game_seed=fixed_game_seed, cnn_arch=cnn_arch, features_dim=features_dim, net_arch=net_arch, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, validation_split=validation_split, target_accuracy=target_accuracy, min_epochs=min_epochs, seed=seed, log_dir=log_dir, tb_log_name=tb_log_name)
