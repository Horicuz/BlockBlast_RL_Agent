import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .fast_env import FastBlockBlastEnv


def mask_fn(env: gym.Env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()
    raise AttributeError("valid_action_mask() is not available")


class FastStatsCallback(BaseCallback):
    def __init__(self, enabled=True, verbose=0):
        super().__init__(verbose)
        self.enabled = enabled

    def _on_step(self):
        if not self.enabled:
            return True
        infos = self.locals.get("infos", [])
        scalar_keys = [
            "reward/line",
            "reward/line_potential",
            "reward/stage",
            "reward/shape",
            "reward/pockets",
            "reward/game_over",
            "board/filled_cells",
            "board/pockets",
            "board/created_pockets",
            "board/line_potential",
            "game/valid_actions",
        ]
        for key in scalar_keys:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record(f"fast_stats/{key.replace('/', '_')}", float(np.mean(values)))
        for info in infos:
            if "game/etap_max" in info:
                self.logger.record("game_stats/Etape_Supravietuite", info["game/etap_max"])
                self.logger.record("game_stats/Linii_Distruse", info["game/linii_distruse"])
                self.logger.record("game_stats/Blocuri_Puse", info["game/blocuri_puse"])
        return True


class FastEvalCallback(BaseCallback):
    def __init__(self, reward_config, eval_freq=0, n_eval_episodes=100, max_eval_steps=5000, verbose=0):
        super().__init__(verbose)
        self.reward_config = reward_config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_eval_steps = max_eval_steps
        self.last_eval_step = 0

    def _on_step(self):
        if self.eval_freq <= 0 or self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        stages, lines, blocks = self._run_eval()
        self.logger.record("eval_stages/mean", float(np.mean(stages)))
        self.logger.record("eval_stages/median", float(np.median(stages)))
        self.logger.record("eval_stages/p90", float(np.percentile(stages, 90)))
        self.logger.record("eval_stages/max", float(np.max(stages)))
        self.logger.record("eval_stats/mean_lines", float(np.mean(lines)))
        self.logger.record("eval_stats/mean_blocks", float(np.mean(blocks)))
        if self.verbose:
            print(f"Eval @ {self.num_timesteps}: mean={np.mean(stages):.2f}, p90={np.percentile(stages, 90):.1f}, max={np.max(stages)}")
        return True

    def _run_eval(self):
        eval_env = FastBlockBlastEnv(reward_config=self.reward_config)
        stages = []
        lines = []
        blocks = []
        try:
            for episode_index in range(self.n_eval_episodes):
                obs, _ = eval_env.reset(seed=20_000 + episode_index)
                done = False
                step_count = 0
                while not done and step_count < self.max_eval_steps:
                    mask = eval_env.valid_action_mask()
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                    obs, _, done, _, _ = eval_env.step(int(action))
                    step_count += 1
                stages.append(eval_env.game.stages_passed)
                lines.append(eval_env.game.lines_destroyed)
                blocks.append(eval_env.game.blocks_placed)
        finally:
            eval_env.close()
        return np.array(stages), np.array(lines), np.array(blocks)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fast Block Blast PPO baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--subproc-start-method", choices=["fork", "forkserver", "spawn"], default="fork")
    parser.add_argument("--num-cpu", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--total-timesteps", type=int, default=25_000_000)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=60_000)
    parser.add_argument("--model-path", default="./checkpoints/fast/block_blast_fast_mlp_v1")
    parser.add_argument("--tb-log-name", default="PPO_BlockBlast_Fast_MLP_V1")
    parser.add_argument("--log-dir", default="./tensorboard_logs/")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/fast/")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--reward-line-scale", type=float, default=8.0)
    parser.add_argument("--reward-stage-complete", type=float, default=3.0)
    parser.add_argument("--reward-game-over", type=float, default=60.0)
    parser.add_argument("--valid-action-delta-weight", type=float, default=0.04)
    parser.add_argument("--filled-cell-penalty", type=float, default=0.015)
    parser.add_argument("--pocket-penalty-weight", type=float, default=0.15)
    parser.add_argument("--created-pocket-penalty-weight", type=float, default=0.75)
    parser.add_argument("--line-potential-weight", type=float, default=0.0)
    parser.add_argument("--line-potential-delta-weight", type=float, default=0.0)
    return parser.parse_args()


def build_reward_config(args):
    return {
        "line_clear_scale": args.reward_line_scale,
        "stage_complete_reward": args.reward_stage_complete,
        "game_over_penalty": args.reward_game_over,
        "valid_action_delta_weight": args.valid_action_delta_weight,
        "filled_cell_penalty": args.filled_cell_penalty,
        "pocket_penalty_weight": args.pocket_penalty_weight,
        "created_pocket_penalty_weight": args.created_pocket_penalty_weight,
        "line_potential_weight": args.line_potential_weight,
        "line_potential_delta_weight": args.line_potential_delta_weight,
    }


def make_env(reward_config):
    def _init():
        env = FastBlockBlastEnv(reward_config=reward_config)
        env = Monitor(env)
        return ActionMasker(env, mask_fn)
    return _init


def build_vec_env(args, reward_config):
    env_fns = [make_env(reward_config) for _ in range(args.num_cpu)]
    if args.vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=args.subproc_start_method)


def resolve_device(choice):
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_threads(thread_count):
    if thread_count > 0:
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(max(1, min(thread_count, 4)))
        except RuntimeError:
            pass


def build_callbacks(args, reward_config):
    callbacks = [FastStatsCallback(enabled=True)]
    if args.eval_freq > 0:
        callbacks.append(FastEvalCallback(reward_config, args.eval_freq, args.eval_episodes, verbose=1))
    if args.checkpoint_freq > 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.num_cpu, 1), 1),
                save_path=args.checkpoint_dir,
                name_prefix=os.path.basename(args.model_path),
            )
        )
    return CallbackList(callbacks)


def build_model(args, env):
    device = resolve_device(args.device)
    model_exists = os.path.exists(args.model_path + ".zip")
    if args.resume and model_exists:
        return MaskablePPO.load(args.model_path, env=env, tensorboard_log=args.log_dir, device=device), True

    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=args.log_dir,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    return model, False


def run_benchmark(args, reward_config):
    candidates = [
        {"num_cpu": 4, "vec_env": "subproc", "torch_threads": 2},
        {"num_cpu": 6, "vec_env": "subproc", "torch_threads": 2},
        {"num_cpu": 8, "vec_env": "subproc", "torch_threads": 2},
        {"num_cpu": 12, "vec_env": "subproc", "torch_threads": 2},
    ]
    results = []
    original_num_cpu = args.num_cpu
    original_vec_env = args.vec_env
    original_torch_threads = args.torch_threads
    try:
        for candidate in candidates:
            args.num_cpu = candidate["num_cpu"]
            args.vec_env = candidate["vec_env"]
            configure_torch_threads(candidate["torch_threads"])
            env = build_vec_env(args, reward_config)
            model, _ = build_model(args, env)
            start_time = time.perf_counter()
            model.learn(total_timesteps=args.benchmark_steps, reset_num_timesteps=True, tb_log_name="FAST_BENCHMARK")
            elapsed = time.perf_counter() - start_time
            fps = model.num_timesteps / elapsed if elapsed > 0 else 0.0
            print(f"fps={fps:.1f} num_cpu={args.num_cpu} vec_env={args.vec_env} torch_threads={candidate['torch_threads']}")
            results.append((fps, candidate.copy()))
            env.close()
    finally:
        args.num_cpu = original_num_cpu
        args.vec_env = original_vec_env
        args.torch_threads = original_torch_threads
    best = max(results, key=lambda item: item[0])
    print(f"best={best[0]:.1f} fps config={best[1]}")


def main():
    args = parse_args()
    configure_torch_threads(args.torch_threads)
    reward_config = build_reward_config(args)
    if args.benchmark:
        run_benchmark(args, reward_config)
        return

    env = build_vec_env(args, reward_config)
    model, resumed = build_model(args, env)
    callbacks = build_callbacks(args, reward_config)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=args.tb_log_name,
            reset_num_timesteps=not resumed,
            callback=callbacks,
        )
    finally:
        model.save(args.model_path)
        env.close()
        print(f"Saved model to {args.model_path}.zip")


if __name__ == "__main__":
    main()
