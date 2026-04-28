#!/usr/bin/env python3

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from behavioral_cloning import collect_expert_data, pretrain_behavioral_cloning
from blocks import TRAINING_POOLS


def parse_fixed_game_seeds(raw_value):
    if not raw_value:
        return None
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Block Blast behavioral cloning pretraining")
    parser.add_argument("--mode", choices=["collect", "train", "all"], default="all")
    parser.add_argument("--data-path", default="./datasets/bc_expert/block_blast_bc_expert.npz")
    parser.add_argument("--model-path", default="./checkpoints/bc/block_blast_bc_policy")
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=5000)
    parser.add_argument("--shape-pool", choices=sorted(TRAINING_POOLS.keys()), default="all")
    parser.add_argument("--hand-generator", choices=["random", "playable", "adaptive_playable", "solvable"], default="adaptive_playable")
    parser.add_argument("--fixed-game-seed", type=int, default=None)
    parser.add_argument("--fixed-game-seeds", default=None, help="Comma-separated fixed seeds for data collection")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--cnn-arch", choices=["base", "v2", "actionaware"], default="actionaware")
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--net-arch", default="256,256")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="./tensorboard_sweeps/bc_logs")
    parser.add_argument("--tb-log-name", default="PPO_BlockBlast_BC")
    return parser.parse_args()


def main():
    args = parse_args()
    fixed_game_seeds = parse_fixed_game_seeds(args.fixed_game_seeds)

    print("[BC] launch summary")
    print(f"[BC] dataset file: {args.data_path}")
    print(f"[BC] model checkpoint: {args.model_path}.zip")
    print(f"[BC] mode: {args.mode}")
    print(f"[BC] target accuracy: {args.target_accuracy:.2f}")
    print(f"[BC] num episodes: {args.num_episodes}")
    print(f"[BC] TensorBoard log dir: {os.path.join(args.log_dir, args.tb_log_name)}")

    if args.mode == "collect":
        archive_path = collect_expert_data(
            output_path=args.data_path,
            num_episodes=args.num_episodes,
            shape_pool=args.shape_pool,
            hand_generator=args.hand_generator,
            fixed_game_seed=args.fixed_game_seed,
            fixed_game_seeds=fixed_game_seeds,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        print(f"[BC] expert dataset saved to {archive_path}")
        return

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Expert dataset not found: {args.data_path}")

    if args.mode == "train":
        print(f"[BC] loading existing dataset only: {args.data_path}")
        pretrain_behavioral_cloning(
            data_path=args.data_path,
            model_path=args.model_path,
            device=args.device,
            shape_pool=args.shape_pool,
            hand_generator=args.hand_generator,
            fixed_game_seed=args.fixed_game_seed,
            cnn_arch=args.cnn_arch,
            features_dim=args.features_dim,
            net_arch=args.net_arch,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            target_accuracy=args.target_accuracy,
            min_epochs=args.min_epochs,
            seed=args.seed,
            log_dir=args.log_dir,
            tb_log_name=args.tb_log_name,
        )
        return

    archive_path = collect_expert_data(
        output_path=args.data_path,
        num_episodes=args.num_episodes,
        shape_pool=args.shape_pool,
        hand_generator=args.hand_generator,
        fixed_game_seed=args.fixed_game_seed,
        fixed_game_seeds=fixed_game_seeds,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    pretrain_behavioral_cloning(
        data_path=archive_path,
        model_path=args.model_path,
        device=args.device,
        shape_pool=args.shape_pool,
        hand_generator=args.hand_generator,
        fixed_game_seed=args.fixed_game_seed,
        cnn_arch=args.cnn_arch,
        features_dim=args.features_dim,
        net_arch=args.net_arch,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        target_accuracy=args.target_accuracy,
        min_epochs=args.min_epochs,
        seed=args.seed,
        log_dir=args.log_dir,
        tb_log_name=args.tb_log_name,
    )
    print(f"[BC] expert dataset saved to {archive_path}")
    print(f"[BC] pre-trained SB3 model saved to {args.model_path}.zip")


if __name__ == "__main__":
    main()
