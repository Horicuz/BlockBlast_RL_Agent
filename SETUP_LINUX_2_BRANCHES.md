# Linux Setup: CNN Experiments (2 Branches)

## Overview
Codul pe `main` conține implementarea complete cu:
- **CustomCNNExtractor**: Feature extractor cu CNN pentru spatial understanding
- **Hole Penalty Function**: Topological reward shaping cu flood-fill BFS
- **Flag `--apply-hole-penalty`**: Toggle pentru a activa/dezactiva penalitatea

## Setup pe Linux

### 1. Clone și verifica codul

```bash
git clone https://github.com/Horicuz/BlockBlast_RL_Agent.git
cd BlockBlast_RL_Agent
git checkout main
```

### 2. Creeaza 2 branch-uri experimentale

#### Branch 1: CNN Only (fara hole penalty)
```bash
git checkout -b experiment-cnn-only
# Train cu CNN, fara reward shaping:
python3 train.py --curriculum-phase phase3 --resume --apply-hole-penalty false
```

#### Branch 2: CNN + Hole Penalty (cu reward shaping)
```bash
git checkout main
git checkout -b experiment-cnn-holes
# Train cu CNN si hole penalty:
python3 train.py --curriculum-phase phase3 --resume --apply-hole-penalty true
```

### 3. Comanda de training pentru fiecare branch

**Branch experiment-cnn-only:**
```bash
git checkout experiment-cnn-only
python3 train.py \
  --curriculum-phase phase3 \
  --resume \
  --apply-hole-penalty false \
  --num-cpu 8 \
  --device auto
```

**Branch experiment-cnn-holes:**
```bash
git checkout experiment-cnn-holes
python3 train.py \
  --curriculum-phase phase3 \
  --resume \
  --apply-hole-penalty true \
  --num-cpu 8 \
  --device auto
```

### 4. Configurare GPU (optional pe Linux)

Daca ai GPU NVIDIA:
```bash
# Verifica CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Train pe GPU:
python3 train.py \
  --curriculum-phase phase3 \
  --resume \
  --apply-hole-penalty [true|false] \
  --device cuda \
  --num-cpu 8
```

### 5. Monitorizeaza training

In alt terminal:
```bash
tensorboard --logdir=./tensorboard_logs/
# Deschide http://localhost:6006
```

### 6. Schimba intre branch-uri in timpul training-ului

```bash
# Save current progress (training va fi oprit cu Ctrl+C mai intai)
git stash

# Switch branch
git checkout experiment-cnn-holes

# Resume training pe noua configuratie
python3 train.py --curriculum-phase phase3 --resume --apply-hole-penalty true
```

## Key Code Components

### CustomCNNExtractor (in env.py)
- Primeste Dict observation space
- Board (1,8,8) → CNN (2 layers, 32→64 filters) → 4096 features
- Hand (3,3,3) → Linear → 64 features  
- Available (3,) → Linear → 16 features
- Combined: 4096 + 64 + 16 = 4176 → 256 dims

### Hole Penalty Function (in env.py - _calculate_holes_penalty)
- BFS flood-fill pentru a detecta regiuni vide conexe
- Connectivity: orthogonal (sus, jos, stanga, dreapta)
- Penalizeaza doar holes mici (1-3 celule)
- Penalty = -0.5 * region_size

### Training Integration (in train.py)
- `apply_hole_penalty` parameter controleaza activarea
- policy_kwargs cu CustomCNNExtractor este setat automat
- Curriculum phases raman aceleasi (phase1, phase2, phase3)

## Rezultate Asteptate

### Experiment 1: CNN Only (experiment-cnn-only)
- Ar trebui sa invete spatial patterns
- Posibil sa creeze mai multe holes (fara penalitate)
- FPS ar trebui sa fie mai inalt (computatie mai simplă)

### Experiment 2: CNN + Holes (experiment-cnn-holes)
- CNN invata cu guidance explicit ca holes sunt rele
- Ar trebui sa creeze mai putine holes
- Posibil FPS ceva mai mic din cauza penalty calculation

## Notes
- Ambele configuri au CNN-ul, diferenta este flag-ul `--apply-hole-penalty`
- Modelele salvate in branch-uri diferite vor fi in: `block_blast_phase3_v1.zip`
- Best models: `checkpoints/best_phase3_v1/best_model.zip`
- Pentru a compara corect, ruleaza amandoua pe same hardware si timp (de ex, 10M steps fiecare)
