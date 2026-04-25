import argparse
import os
import time

import gymnasium as gym
import torch
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def mask_fn(env: gym.Env):
    if hasattr(env, "valid_action_mask"):
        return env.valid_action_mask()

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "valid_action_mask"):
        return unwrapped.valid_action_mask()

    raise AttributeError("valid_action_mask() nu este disponibil pe mediul curent")

class TensorboardStatsCallback(BaseCallback):
    def __init__(self, enabled=True, verbose=0):
        super().__init__(verbose)
        self.enabled = enabled

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        for info in self.locals.get("infos", []):
            if "game/etap_max" in info:
                self.logger.record("game_stats/Etape_Supravietuite", info["game/etap_max"])
                self.logger.record("game_stats/Linii_Distruse", info["game/linii_distruse"])
                self.logger.record("game_stats/Blocuri_Puse", info["game/blocuri_puse"])
        return True


def make_env(reward_config):
    def _init():
        env = BlockBlastEnv(reward_config=reward_config)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train or benchmark the Block Blast PPO agent")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--num-cpu", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    parser.add_argument("--lr-final-ratio", type=float, default=0.2)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=30_000)
    parser.add_argument("--model-path", default="block_blast_ai_v4")
    parser.add_argument("--tb-log-name", default="PPO_BlockBlast_V4")
    parser.add_argument("--log-dir", default="./tensorboard_logs/")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--best-model-dir", default="./checkpoints/best/")
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-stats", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reward-placement", type=float, default=0.0)
    parser.add_argument("--reward-line-scale", type=float, default=10.0)
    parser.add_argument("--reward-line-bonus", type=float, default=0.0)
    parser.add_argument("--reward-no-line", type=float, default=0.0)
    parser.add_argument("--reward-game-over", type=float, default=100.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)

    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice

    if torch.cuda.is_available():
        print("🚀 Dispozitiv procesare Neural Network: NVIDIA CUDA")
        return "cuda"

    if torch.backends.mps.is_available():
        print("⚠️ MPS este disponibil, dar pentru acest proiect CPU tinde să fie mai rapid. Folosește --device mps doar dacă vrei să-l testezi explicit.")

    print("⚠️ Atenție: rulăm pe CPU.")
    return "cpu"


def build_vec_env(vec_env_kind: str, num_cpu: int, reward_config):
    env_fns = [make_env(reward_config) for _ in range(num_cpu)]
    if vec_env_kind == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def configure_torch_threads(thread_count: int):
    if thread_count > 0:
        torch.set_num_threads(thread_count)


def build_reward_config(args):
    return {
        "placement_reward": args.reward_placement,
        "line_clear_scale": args.reward_line_scale,
        "line_clear_bonus": args.reward_line_bonus,
        "no_line_penalty": args.reward_no_line,
        "game_over_penalty": args.reward_game_over,
        "reward_scale": args.reward_scale,
    }


def build_learning_rate(args):
    if args.lr_schedule == "constant":
        return args.learning_rate

    initial_lr = args.learning_rate
    final_lr = max(initial_lr * args.lr_final_ratio, 1e-8)

    def linear_decay(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining

    return linear_decay


def build_model(env, device: str, args):
    model_exists = os.path.exists(args.model_path + ".zip")
    should_resume = args.resume and model_exists

    if should_resume:
        print("✅ GĂSIT! Continuăm antrenamentul modelului existent...")
        model = MaskablePPO.load(args.model_path, env=env, tensorboard_log=args.log_dir, device=device)
        print("ℹ️ Pentru modelul încărcat, hiperparametrii salvați rămân activi. Dacă vrei setări complet noi, folosește --no-resume.")
        return model, True

    print("🧠 Inițializăm un model nou de la zero...")
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
    )
    return model, False


def build_training_callbacks(args, eval_env):
    callbacks = []

    if args.log_stats:
        callbacks.append(TensorboardStatsCallback(enabled=True))

    if args.checkpoint_freq > 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.num_cpu, 1), 1),
                save_path=args.checkpoint_dir,
                name_prefix=os.path.basename(args.model_path),
            )
        )

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

    if not callbacks:
        return None

    if len(callbacks) == 1:
        return callbacks[0]

    return CallbackList(callbacks)


def run_benchmark(args, reward_config):
    candidate_configs = [
        {"device": "cpu", "num_cpu": 8, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 1, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "dummy"},
        {"device": "cpu", "num_cpu": 8, "n_steps": 4096, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 1, "n_steps": 4096, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "dummy"},
        {"device": "cpu", "num_cpu": 6, "n_steps": 4096, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "subproc"},
    ]

    if torch.backends.mps.is_available():
        candidate_configs.append(
            {"device": "mps", "num_cpu": 8, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "subproc"}
        )

    results = []
    print("🏁 Pornim benchmark-ul scurt pentru FPS...")

    for index, candidate in enumerate(candidate_configs, start=1):
        print(
            f"\n[{index}/{len(candidate_configs)}] device={candidate['device']} vec_env={candidate['vec_env']} "
            f"num_cpu={candidate['num_cpu']} n_steps={candidate['n_steps']} batch_size={candidate['batch_size']} "
            f"n_epochs={candidate['n_epochs']}"
        )

        torch.set_num_threads(candidate["torch_threads"])
        env = build_vec_env(candidate["vec_env"], candidate["num_cpu"], reward_config)
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            n_steps=candidate["n_steps"],
            batch_size=candidate["batch_size"],
            n_epochs=candidate["n_epochs"],
            device=candidate["device"],
        )

        start_time = time.perf_counter()
        model.learn(
            total_timesteps=args.benchmark_steps,
            reset_num_timesteps=True,
            tb_log_name="BENCHMARK",
            callback=None,
        )
        elapsed = time.perf_counter() - start_time
        fps = args.benchmark_steps / elapsed if elapsed > 0 else 0.0
        print(f"    -> elapsed={elapsed:.2f}s, fps={fps:.1f}")
        results.append((fps, candidate))

        env.close()

    results.sort(key=lambda item: item[0], reverse=True)
    best_fps, best_candidate = results[0]

    print("\n📊 Rezultate benchmark:")
    for fps, candidate in results:
        print(
            f"  fps={fps:.1f} | device={candidate['device']} vec_env={candidate['vec_env']} "
            f"num_cpu={candidate['num_cpu']} n_steps={candidate['n_steps']} batch_size={candidate['batch_size']} "
            f"n_epochs={candidate['n_epochs']}"
        )

    print(
        f"\n✅ Cea mai rapidă variantă: device={best_candidate['device']} vec_env={best_candidate['vec_env']} "
        f"num_cpu={best_candidate['num_cpu']} n_steps={best_candidate['n_steps']} "
        f"batch_size={best_candidate['batch_size']} n_epochs={best_candidate['n_epochs']} "
        f"cu ~{best_fps:.1f} FPS"
    )


def main():
    args = parse_args()

    if args.torch_threads <= 0:
        args.torch_threads = max(1, min(4, (os.cpu_count() or 1) // 2))

    configure_torch_threads(args.torch_threads)
    reward_config = build_reward_config(args)

    if args.benchmark:
        run_benchmark(args, reward_config)
        return

    device = resolve_device(args.device)
    print(f"🚀 Antrenament pe: {device.upper()}")
    print(f"⚡ Se pornesc {args.num_cpu} instanțe de joc în paralel...")
    print(f"📦 Model path: {args.model_path}.zip")

    env = build_vec_env(args.vec_env, args.num_cpu, reward_config)
    eval_env = build_vec_env(args.vec_env, 1, reward_config)
    model, resumed = build_model(env, device, args)

    print("🏁 Începem antrenamentul...")
    callback = build_training_callbacks(args, eval_env)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=args.tb_log_name,
            reset_num_timesteps=not resumed,
            callback=callback,
        )

        print("Antrenament complet! Salvăm modelul final...")
        model.save(args.model_path)
    finally:
        for vec_env in (env, eval_env):
            try:
                vec_env.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()