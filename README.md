# BlockBlast RL Agent

A clean, practical reinforcement learning project where an agent learns to play **BlockBlast-style puzzle gameplay** through trial and error.

## Project Overview

This repository demonstrates how to:

1. Build a custom game environment suitable for RL training.
2. Train an agent with reward shaping and episodic learning.
3. Evaluate and visualize performance over time.

The goal is to create an agent that discovers strategies for maximizing score and surviving longer without hard-coded rules.

## Features

- Custom BlockBlast-like environment interface
- Training loop for RL agents (policy/value-based ready)
- Reward design for score, board control, and survival
- Logging of episode reward, score, and learning progress
- Evaluation mode to benchmark trained checkpoints

## Suggested Tech Stack

- **Language:** Python 3.10+
- **Core:** NumPy, PyTorch
- **RL tooling:** Gymnasium-compatible API (or custom wrapper)
- **Visualization:** Matplotlib / TensorBoard

## Repository Structure

```text
.
├── env/              # Game logic and environment wrappers
├── agent/            # Model, policy, replay buffer, update rules
├── train.py          # Main training entrypoint
├── evaluate.py       # Runs trained models on evaluation episodes
├── checkpoints/      # Saved models
└── README.md
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd BlockBlast_RL_Agent
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run training with:

```bash
python train.py
```

Typical configurable hyperparameters include:

- learning rate
- discount factor (`gamma`)
- exploration schedule
- batch size
- target update frequency (if using DQN-style methods)

## Evaluation

Evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/latest.pt
```

Track:

- mean score
- best score
- episode length
- consistency across random seeds

## Roadmap

- Add curriculum learning for harder board states
- Compare multiple RL algorithms (DQN, PPO, A2C)
- Export gameplay rollouts as GIF/video
- Add automated hyperparameter sweeps

## Why this project is nice

It is small enough to understand end-to-end, but rich enough to showcase core RL concepts: exploration, reward design, stability, and generalization.

## License

MIT (or your preferred open-source license).
