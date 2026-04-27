import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
from blocks import TRAINING_POOLS
from env import BlockBlastEnv, ActionAwareCNNExtractor, CustomCNNExtractor, CustomCNNExtractorV2
from sb3_contrib import MaskablePPO
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

        infos = self.locals.get("infos", [])
        scalar_keys = [
            "reward/line",
            "reward/stage",
            "reward/contact",
            "reward/holes",
            "reward/game_over",
            "placement/contact_ratio",
            "placement/contact_score",
            "placement/contact_threshold",
            "holes/current_cells",
            "holes/created_cells",
            "game/valid_actions",
        ]

        for key in scalar_keys:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record(f"step_stats/{key.replace('/', '_')}", float(np.mean(values)))

        for info in infos:
            if "game/etap_max" in info:
                self.logger.record("game_stats/Etape_Supravietuite", info["game/etap_max"])
                self.logger.record("game_stats/Linii_Distruse", info["game/linii_distruse"])
                self.logger.record("game_stats/Blocuri_Puse", info["game/blocuri_puse"])
        return True


class StageEvalCallback(BaseCallback):
    def __init__(
        self,
        reward_config,
        fixed_game_seed=None,
        fixed_game_seeds=None,
        eval_freq=0,
        n_eval_episodes=50,
        max_eval_steps=5000,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self.reward_config = reward_config
        self.fixed_game_seed = fixed_game_seed
        self.fixed_game_seeds = fixed_game_seeds
        self.shape_pool = reward_config["shape_pool"]
        self.hand_generator = reward_config["hand_generator"]
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_eval_steps = max_eval_steps
        self.deterministic = deterministic
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
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
            print(
                f"Eval @ {self.num_timesteps}: mean_stages={np.mean(stages):.2f}, "
                f"p90={np.percentile(stages, 90):.1f}, max={np.max(stages)}"
            )

        return True

    def _run_eval(self):
        stages = []
        lines = []
        blocks = []
        eval_env = BlockBlastEnv(
            reward_config=self.reward_config,
            apply_hole_penalty=self.reward_config["apply_hole_penalty"],
            fixed_game_seed=self.fixed_game_seed,
            shape_pool=self.shape_pool,
            hand_generator=self.hand_generator,
        )

        try:
            for episode_idx in range(self.n_eval_episodes):
                if self.fixed_game_seeds:
                    eval_env.fixed_game_seed = self.fixed_game_seeds[episode_idx % len(self.fixed_game_seeds)]
                obs, _ = eval_env.reset(seed=10_000 + episode_idx)
                done = False
                step_count = 0

                while not done and step_count < self.max_eval_steps:
                    action_mask = eval_env.valid_action_mask()
                    action, _ = self.model.predict(
                        obs,
                        action_masks=action_mask,
                        deterministic=self.deterministic,
                    )
                    obs, _, done, _, _ = eval_env.step(int(action))
                    step_count += 1

                stages.append(eval_env.game.stages_passed)
                lines.append(eval_env.game.lines_destroyed)
                blocks.append(eval_env.game.blocks_placed)
        finally:
            eval_env.close()

        return np.array(stages), np.array(lines), np.array(blocks)


def make_env(reward_config, fixed_game_seed=None):
    def _init():
        env = BlockBlastEnv(
            reward_config=reward_config,
            apply_hole_penalty=reward_config["apply_hole_penalty"],
            fixed_game_seed=fixed_game_seed,
            shape_pool=reward_config["shape_pool"],
            hand_generator=reward_config["hand_generator"],
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train or benchmark the Block Blast PPO agent")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--subproc-start-method", choices=["fork", "forkserver", "spawn"], default="fork")
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
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=30_000)
    parser.add_argument("--model-path", default="./checkpoints/cnn/block_blast_cnn_v1")
    parser.add_argument("--tb-log-name", default="PPO_BlockBlast_CNN_V1")
    parser.add_argument("--log-dir", default="./tensorboard_logs/")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/cnn/")
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-stats", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reward-placement", type=float, default=0.0)
    parser.add_argument("--reward-line-scale", type=float, default=10.0)
    parser.add_argument("--reward-line-bonus", type=float, default=0.0)
    parser.add_argument("--reward-stage-complete", type=float, default=10.0)
    parser.add_argument("--reward-no-line", type=float, default=0.0)
    parser.add_argument("--reward-game-over", type=float, default=200.0)
    parser.add_argument("--reward-game-over-early-weight", type=float, default=3.0)
    parser.add_argument("--reward-contact-scale", type=float, default=12.0)
    parser.add_argument("--reward-contact-power", type=float, default=1.25)
    parser.add_argument("--reward-contact-threshold", type=float, default=0.0)
    parser.add_argument("--reward-contact-penalty-scale", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--apply-hole-penalty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hole-penalty-weight", type=float, default=0.25)
    parser.add_argument("--created-hole-penalty-weight", type=float, default=1.0)
    parser.add_argument("--complexity-simple-prob", type=float, default=0.78)
    parser.add_argument("--complexity-medium-prob", type=float, default=0.18)
    parser.add_argument("--complexity-hard-prob", type=float, default=0.04)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-eval-steps", type=int, default=5000)
    parser.add_argument("--fixed-game-seed", type=int, default=None)
    parser.add_argument("--fixed-game-seeds", default=None, help="Comma-separated seeds, assigned across envs and eval episodes.")
    parser.add_argument("--fixed-game-seed-count", type=int, default=0, help="Use fixed-game-seed as the first seed and generate this many consecutive seeds.")
    parser.add_argument("--init-from-model", default=None, help="Initialize policy weights from this model path, but keep current hyperparameters.")
    parser.add_argument("--shape-pool", choices=sorted(TRAINING_POOLS.keys()), default="all")
    parser.add_argument("--hand-generator", choices=["solvable", "playable", "adaptive_playable", "random"], default="solvable")
    parser.add_argument("--cnn-arch", choices=["base", "v2", "actionaware"], default="base")
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--net-arch", default="256,256", help="Comma-separated hidden sizes for policy/value heads.")

    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


def normalize_model_path(path: str) -> str:
    if path.endswith(".zip"):
        return path[:-4]
    return path


def parse_fixed_game_seeds(args):
    if args.fixed_game_seeds:
        return [int(value.strip()) for value in args.fixed_game_seeds.split(",") if value.strip()]
    if args.fixed_game_seed is not None and args.fixed_game_seed_count > 0:
        return [args.fixed_game_seed + offset for offset in range(args.fixed_game_seed_count)]
    if args.fixed_game_seed is not None:
        return [args.fixed_game_seed]
    return None


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


def build_vec_env(vec_env_kind: str, num_cpu: int, reward_config, start_method: str = "fork", fixed_game_seed=None, fixed_game_seeds=None):
    env_fns = []
    for env_index in range(num_cpu):
        env_seed = fixed_game_seed
        if fixed_game_seeds:
            env_seed = fixed_game_seeds[env_index % len(fixed_game_seeds)]
        env_fns.append(make_env(reward_config, fixed_game_seed=env_seed))
    if vec_env_kind == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)


def configure_torch_threads(thread_count: int):
    if thread_count > 0:
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(max(1, min(thread_count, 4)))
        except RuntimeError:
            pass


def build_reward_config(args):
    return {
        "placement_reward": args.reward_placement,
        "line_clear_scale": args.reward_line_scale,
        "line_clear_bonus": args.reward_line_bonus,
        "stage_complete_reward": args.reward_stage_complete,
        "no_line_penalty": args.reward_no_line,
        "game_over_penalty": args.reward_game_over,
        "game_over_early_weight": args.reward_game_over_early_weight,
        "contact_reward_scale": args.reward_contact_scale,
        "contact_reward_power": args.reward_contact_power,
        "contact_reward_threshold": args.reward_contact_threshold,
        "contact_penalty_scale": args.reward_contact_penalty_scale,
        "reward_scale": args.reward_scale,
        "apply_hole_penalty": args.apply_hole_penalty,
        "hole_penalty_weight": args.hole_penalty_weight,
        "created_hole_penalty_weight": args.created_hole_penalty_weight,
        "complexity_simple_prob": args.complexity_simple_prob,
        "complexity_medium_prob": args.complexity_medium_prob,
        "complexity_hard_prob": args.complexity_hard_prob,
        "shape_pool": args.shape_pool,
        "hand_generator": args.hand_generator,
    }


def build_policy_kwargs(args):
    extractor_class = {
        "base": CustomCNNExtractor,
        "v2": CustomCNNExtractorV2,
        "actionaware": ActionAwareCNNExtractor,
    }[args.cnn_arch]
    net_arch = [int(value.strip()) for value in args.net_arch.split(",") if value.strip()]
    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ReLU,
    )


def build_learning_rate(args):
    if args.lr_schedule == "constant":
        return args.learning_rate

    initial_lr = args.learning_rate
    final_lr = max(initial_lr * args.lr_final_ratio, 1e-8)

    def linear_decay(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining

    return linear_decay


def ensure_parent_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def build_model(env, device: str, args):
    model_exists = os.path.exists(args.model_path + ".zip")
    should_resume = args.resume and model_exists
    init_from_model = normalize_model_path(args.init_from_model) if args.init_from_model else None

    policy_kwargs = build_policy_kwargs(args)

    if should_resume:
        print("✅ GĂSIT! Continuăm antrenamentul modelului existent...")
        model = MaskablePPO.load(args.model_path, env=env, tensorboard_log=args.log_dir, device=device)
        print("ℹ️ Pentru modelul încărcat, hiperparametrii salvați rămân activi. Dacă vrei setări complet noi, folosește --no-resume.")
        return model, True

    if init_from_model:
        print("🧠 Inițializăm model nou cu hiperparametrii curenți și greutăți preluate din modelul indicat...")
    else:
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
        policy_kwargs=policy_kwargs,
    )

    if init_from_model:
        if not os.path.exists(init_from_model + ".zip"):
            raise FileNotFoundError(f"Nu există modelul pentru init-from-model: {init_from_model}.zip")
        source_model = MaskablePPO.load(init_from_model, device=device)
        model.policy.load_state_dict(source_model.policy.state_dict())
        print(f"✅ Greutăți încărcate din: {init_from_model}.zip")

    return model, False


def build_training_callbacks(args, reward_config):
    callbacks = []

    if args.log_stats:
        callbacks.append(TensorboardStatsCallback(enabled=True))

    if args.eval_freq > 0:
        callbacks.append(
            StageEvalCallback(
                reward_config=reward_config,
                fixed_game_seed=args.fixed_game_seed,
                fixed_game_seeds=args.resolved_fixed_game_seeds,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                max_eval_steps=args.max_eval_steps,
                deterministic=True,
                verbose=1,
            )
        )

    if args.checkpoint_freq > 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.num_cpu, 1), 1),
                save_path=args.checkpoint_dir,
                name_prefix=os.path.basename(args.model_path),
            )
        )

    if not callbacks:
        return None

    if len(callbacks) == 1:
        return callbacks[0]

    return CallbackList(callbacks)


def run_benchmark(args, reward_config):
    candidate_configs = [
        {"device": "cpu", "num_cpu": 8, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 1, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 12, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 1, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 16, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 1, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 12, "n_steps": 1024, "batch_size": 1024, "n_epochs": 8, "torch_threads": 1, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 12, "n_steps": 2048, "batch_size": 2048, "n_epochs": 5, "torch_threads": 1, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 8, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "subproc"},
        {"device": "cpu", "num_cpu": 1, "n_steps": 2048, "batch_size": 1024, "n_epochs": 8, "torch_threads": 4, "vec_env": "dummy"},
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

        configure_torch_threads(candidate["torch_threads"])
        env = build_vec_env(candidate["vec_env"], candidate["num_cpu"], reward_config, args.subproc_start_method)
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
            policy_kwargs=dict(
                features_extractor_class=(
                    CustomCNNExtractor
                    if args.cnn_arch == "base"
                    else CustomCNNExtractorV2
                    if args.cnn_arch == "v2"
                    else ActionAwareCNNExtractor
                ),
                features_extractor_kwargs=dict(features_dim=args.features_dim),
                net_arch=dict(
                    pi=[int(value.strip()) for value in args.net_arch.split(",") if value.strip()],
                    vf=[int(value.strip()) for value in args.net_arch.split(",") if value.strip()],
                ),
                activation_fn=torch.nn.ReLU,
            ),
        )

        start_time = time.perf_counter()
        model.learn(
            total_timesteps=args.benchmark_steps,
            reset_num_timesteps=True,
            tb_log_name="BENCHMARK",
            callback=None,
        )
        elapsed = time.perf_counter() - start_time
        actual_steps = model.num_timesteps
        fps = actual_steps / elapsed if elapsed > 0 else 0.0
        print(f"    -> elapsed={elapsed:.2f}s, steps={actual_steps}, fps={fps:.1f}")
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
    args.resolved_fixed_game_seeds = parse_fixed_game_seeds(args)

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
    print(f"🧱 Checkpoint dir: {args.checkpoint_dir}")
    print(f"🕳️ Hole penalty: {'ENABLED' if args.apply_hole_penalty else 'DISABLED'}")
    print(f"🧩 Shape pool: {args.shape_pool} ({len(TRAINING_POOLS[args.shape_pool])} piese)")
    print(f"🎲 Hand generator: {args.hand_generator}")
    print(f"🧠 CNN arch: {args.cnn_arch}, features_dim={args.features_dim}")
    print(
        f"🤝 Contact reward: threshold={args.reward_contact_threshold}, "
        f"plus_scale={args.reward_contact_scale}, minus_scale={args.reward_contact_penalty_scale}, "
        f"power={args.reward_contact_power}"
    )
    print(
        f"🔁 PPO rollout: n_steps={args.n_steps}, envs={args.num_cpu}, "
        f"batch={args.batch_size}, epochs={args.n_epochs}, gamma={args.gamma}, lr={args.learning_rate}"
    )
    if args.resolved_fixed_game_seeds:
        preview = ", ".join(str(seed) for seed in args.resolved_fixed_game_seeds[:12])
        suffix = "" if len(args.resolved_fixed_game_seeds) <= 12 else ", ..."
        print(f"🎯 Fixed game seeds: {preview}{suffix} ({len(args.resolved_fixed_game_seeds)} seed-uri)")

    env = build_vec_env(
        args.vec_env,
        args.num_cpu,
        reward_config,
        args.subproc_start_method,
        fixed_game_seed=args.fixed_game_seed,
        fixed_game_seeds=args.resolved_fixed_game_seeds,
    )
    model, resumed = build_model(env, device, args)

    print("🏁 Începem antrenamentul...")
    callback = build_training_callbacks(args, reward_config)
    interrupted = False

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=args.tb_log_name,
            reset_num_timesteps=not resumed,
            callback=callback,
        )

        print("Antrenament complet! Salvăm modelul final...")
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️ Antrenament oprit manual. Salvăm progresul curent...")
    finally:
        ensure_parent_dir(args.model_path)
        model.save(args.model_path)
        print(f"💾 Model salvat în: {args.model_path}.zip")

        try:
            env.close()
        except Exception:
            pass

    if interrupted:
        print("ℹ️ Poți relua ulterior cu --resume dacă vrei să continui din acest punct.")

if __name__ == "__main__":
    main()
