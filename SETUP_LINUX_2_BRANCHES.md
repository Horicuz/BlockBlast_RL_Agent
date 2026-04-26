# Linux Setup: Single CNN Model Workflow

## Overview
Codul pe `main` folosește acum un singur model CNN și nu mai are faze curriculum. Ai doar două configurări de antrenare:
- **CNN only**: fără hole penalty
- **CNN + holes**: cu reward shaping pentru găuri mici

## Setup pe Linux

### 1. Clone și intră pe `main`

```bash
git clone https://github.com/Horicuz/BlockBlast_RL_Agent.git
cd BlockBlast_RL_Agent
git checkout main
```

### 2. Verifică mediul

```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements-torch-cpu.txt
pip install -r requirements.txt
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 3. Rulează varianta fără holes

Folosește un model path separat ca să nu suprascrii experimentul cu holes:

```bash
python3 train.py \
  --resume \
  --no-apply-hole-penalty \
  --model-path ./checkpoints/cnn_noholes/block_blast_cnn_noholes_v1 \
  --tb-log-name PPO_BlockBlast_CNN_NoHoles_V1 \
  --checkpoint-dir ./checkpoints/cnn_noholes/ \
  --num-cpu 8 \
  --torch-threads 4 \
  --subproc-start-method fork \
  --device auto
```

### 4. Rulează varianta cu holes

```bash
python3 train.py \
  --no-resume \
  --apply-hole-penalty \
  --hole-penalty-weight 0.25 \
  --created-hole-penalty-weight 1.0 \
  --model-path ./checkpoints/cnn_holes/block_blast_cnn_holes_v1 \
  --tb-log-name PPO_BlockBlast_CNN_Holes_V1 \
  --checkpoint-dir ./checkpoints/cnn_holes/ \
  --checkpoint-freq 1000000 \
  --eval-freq 500000 \
  --eval-episodes 100 \
  --num-cpu 8 \
  --torch-threads 4 \
  --subproc-start-method fork \
  --device auto
```

### 5. Dacă ai NVIDIA GPU pe Linux

```bash
python3 train.py \
  --resume \
  --apply-hole-penalty \
  --device cuda \
  --num-cpu 8 \
  --model-path ./checkpoints/cnn_holes/block_blast_cnn_holes_v1 \
  --tb-log-name PPO_BlockBlast_CNN_Holes_V1
```

### 6. Monitorizează în TensorBoard

```bash
tensorboard --logdir=./tensorboard_logs/
```

## Key Code Components

### CustomCNNExtractor (in env.py)
- Primește `Dict` observation space
- Board `(1, 8, 8)` → CNN → features spațiale
- Hand `(3, 3, 3)` → Linear
- Available `(3,)` → Linear
- Feature vector final → `features_dim=256`

### Hole Penalty (in env.py)
- BFS flood fill pe celule vide
- Connectivity ortogonală: sus, jos, stânga, dreapta
- Penalizează regiuni mici și adaugă penalizare separată pentru găuri nou create

### Legal Action Maps (in env.py)
- Observația include `valid_actions` cu shape `(3, 8, 8)`
- CNN-ul primește board-ul plus hărțile pozițiilor valide ca input spațial
- Action masking rămâne activ, dar rețeaua vede explicit unde poate juca fiecare piesă

### Training Integration (in train.py)
- `policy_kwargs` injectează `CustomCNNExtractor`
- `--apply-hole-penalty` controlează reward shaping-ul
- Nu mai există `phase1/phase2/phase3`

## Note
- Dacă vrei comparatie curată, folosește două `model-path` diferite.
- Dacă vrei să continui același experiment, folosește același `model-path` și `--resume`.
