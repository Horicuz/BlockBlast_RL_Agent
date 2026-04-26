import argparse
import os
import shutil
import torch
from env import BlockBlastEnv, CustomCNNExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np


CURRICULUM_PRESETS = {
    "phase1": {
        "model_path": "block_blast_phase1_v1",
        "tb_log_name": "PPO_BlockBlast_Phase1_V1",
        "best_model_dir": "./checkpoints/best_phase1_v1/",
        "total_timesteps": 15_000_000,
        "reward_stage_complete": 15.0,
        "reward_game_over": 180.0,
        "reward_game_over_early_weight": 2.5,
        "apply_hole_penalty": False,
    },
    "phase2": {
        "model_path": "block_blast_phase2_v1",
        "tb_log_name": "PPO_BlockBlast_Phase2_V1",
        "best_model_dir": "./checkpoints/best_phase2_v1/",
        "total_timesteps": 20_000_000,
        "reward_stage_complete": 8.0,
        "reward_game_over": 260.0,
        "reward_game_over_early_weight": 3.0,
        "apply_hole_penalty": False,
    },
    "phase3": {
        "model_path": "block_blast_phase3_v1",
        "tb_log_name": "PPO_BlockBlast_Phase3_V1",
        "best_model_dir": "./checkpoints/best_phase3_v1/",
        "total_timesteps": 25_000_000,
        "reward_stage_complete": 3.0,
        "reward_game_over": 350.0,
        "reward_game_over_early_weight": 3.0,
        "apply_hole_penalty": False,
    },
}


def mask_fn(env: gym.Env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()
    raise AttributeError("valid_action_mask() not available on current environment")


class GameStatsCallback(BaseCallback):
    """Log game statistics to tensorboard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "game/etap_max" in info:
                self.logger.record("game_stats/Etape_Supravietuite", info["game/etap_max"])
                self.logger.record("game_stats/Linii_Distruse", info["game/linii_distruse"])
                self.logger.record("game_stats/Blocuri_Puse", info["game/blocuri_puse"])
        return True


def make_env(reward_config, apply_hole_penalty=False):
    def _init():
        env = BlockBlastEnv(reward_config=reward_config, apply_hole_penalty=apply_hole_penalty)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train Block Blast PPO agent with CNN feature extractor")
    parser.add_argument("--curriculum-phase", choices=["phase1", "phase2", "phase3", "custom"], default="phase1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-cpu", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    parser.add_argument("--lr-final-ratio", type=float, default=0.2)
    parser.add_argument("--total-timesteps", type=int, default=25_000_000)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-dir", default="./tensorboard_logs/")
    parser.add_argument("--best-model-dir", default="./checkpoints/best_phase1_v1/")
    parser.add_argument("--model-path", default="block_blast_phase1_v1")
    parser.add_argument("--tb-log-name", default="PPO_BlockBlast_Phase1_V1")
    
    parser.add_argument("--reward-placement", type=float, default=0.0)
    parser.add_argument("--reward-line-scale", type=float, default=10.0)
    parser.add_argument("--reward-line-bonus", type=float, default=0.0)
    parser.add_argument("--reward-stage-complete", type=float, default=10.0)
    parser.add_argument("--reward-no-line", type=float, default=0.0)
    parser.add_argument("--reward-game-over", type=float, default=200.0)
    parser.add_argument("--reward-game-over-early-weight", type=float, default=3.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    
    parser.add_argument("--apply-hole-penalty", action=argparse.BooleanOptionalAction, default=False)
    
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        print("🚀 Using NVIDIA CUDA")
        return "cuda"
    if torch.backends.mps.is_available():
        print("⚠️ MPS available but CPU preferred. Use --device mps to test.")
    print("⚠️ Running on CPU.")
    return "cpu"


def build_learning_rate(args):
    if args.lr_schedule == "constant":
        return args.learning_rate
    initial_lr = args.learning_rate
    final_lr = max(initial_lr * args.lr_final_ratio, 1e-8)
    def linear_decay(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return linear_decay


def build_reward_config(args):
    return {
        "placement_reward": args.reward_placement,
        "line_clear_scale": args.reward_line_scale,
        "line_clear_bonus": args.reward_line_bonus,
        "stage_complete_reward": args.reward_stage_complete,
        "no_line_penalty": args.reward_no_line,
        "game_over_penalty": args.reward_game_over,
        "game_over_early_weight": args.reward_game_over_early_weight,
        "reward_scale": args.reward_scale,
    }


def apply_curriculum_preset(args):
    if args.curriculum_phase == "custom":
        return
    preset = CURRICULUM_PRESETS[args.curriculum_phase]
    args.model_path = preset["model_path"]
    args.tb_log_name = preset["tb_log_name"]
    args.best_model_dir = preset["best_model_dir"]
    args.total_timesteps = preset["total_timesteps"]
    args.reward_stage_complete = preset["reward_stage_complete"]
    args.reward_game_over = preset["reward_game_over"]
    args.reward_game_over_early_weight = preset["reward_game_over_early_weight"]


def build_vec_env(num_cpu, reward_config, apply_hole_penalty):
    env_fns = [make_env(reward_config, apply_hole_penalty) for _ in range(num_cpu)]
    return SubprocVecEnv(env_fns)


def build_model(env, device, args, resumed):
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=build_learning_rate(args),
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=args.log_dir,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    return model


def build_training_callbacks(args, eval_env):
    callbacks = []
    callbacks.append(GameStatsCallback(verbose=1))
    
    if args.eval_freq > 0 and args.eval_episodes > 0:
        os.makedirs(args.best_model_dir, exist_ok=True)
        callbacks.append(
            MaskableEvalCallback(
                eval_env=eval_env,
                best_model_save_path=args.best_model_dir,
                log_path=args.best_model_dir,
                eval_freq=max(args.eval_freq // max(args.num_cpu, 1), 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )
    
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)


def export_best_model(args):
    source_best_model = os.path.join(args.best_model_dir, "best_model.zip")
    target_best_model = args.model_path + "_best.zip"
    if os.path.exists(source_best_model):
        shutil.copyfile(source_best_model, target_best_model)
        print(f"🏆 Best model exported: {target_best_model}")


def main():
    args = parse_args()
    apply_curriculum_preset(args)
    
    device = resolve_device(args.device)
    print(f"🚀 Training on: {device.upper()}")
    print(f"🧭 Curriculum phase: {args.curriculum_phase}")
    print(f"⚡ Launching {args.num_cpu} parallel game instances...")
    print(f"📦 Model path: {args.model_path}.zip")
    print(f"🔧 Hole penalty: {'ENABLED' if args.apply_hole_penalty else 'DISABLED'}")
    
    reward_config = build_reward_config(args)
    env = build_vec_env(args.num_cpu, reward_config, args.apply_hole_penalty)
    eval_env = build_vec_env(1, reward_config, args.apply_hole_penalty)
    
    # Check if model exists for resume
    model_exists = os.path.exists(args.model_path + ".zip")
    should_resume = args.resume and model_exists
    
    if should_resume:
        print("✅ FOUND! Resuming training from existing model...")
        model = MaskablePPO.load(args.model_path, env=env, tensorboard_log=args.log_dir, device=device)
    else:
        print("🧠 Initializing new model from scratch...")
        model = build_model(env, device, args, False)
    
    print("🏁 Starting training...")
    callback = build_training_callbacks(args, eval_env)
    interrupted = False
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=args.tb_log_name,
            reset_num_timesteps=not should_resume,
            callback=callback,
        )
        print("✅ Training complete! Saving model...")
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️ Training interrupted. Saving progress...")
    finally:
        model.save(args.model_path)
        print(f"💾 Model saved: {args.model_path}.zip")
        export_best_model(args)
        
        try:
            env.close()
            eval_env.close()
        except Exception:
            pass
    
    if interrupted:
        print("ℹ️ Resume training anytime with --resume flag.")


if __name__ == "__main__":
    main()

