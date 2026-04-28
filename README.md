# BlockBlast RL Agent

Proiectul implementeaza un agent inteligent pentru un joc de tip Block Blast. Agentul este antrenat prin invatare prin recompensa, folosind un mediu compatibil Gymnasium, algoritmul MaskablePPO si extractoare de caracteristici bazate pe CNN.

Scopul principal este testarea mai multor variante de generare a pieselor, recompense, arhitecturi neuronale si metode de evaluare pentru a vedea ce configuratii pot invata o strategie stabila pe tabla de joc.

## Structura proiectului

```text
.
|-- blocks.py                  # biblioteca de piese si categorii de forme
|-- engine.py                  # motorul jocului: plasare, validare, linii, generator
|-- env.py                     # mediul Gymnasium, reward-uri, action mask, CNN-uri
|-- train.py                   # antrenare PPO, evaluare periodica, checkpoint-uri
|-- watch.py                   # interfata vizuala pentru model, euristica si testare manuala
|-- behavioral_cloning.py      # colectare date euristice si pre-antrenare supervizata
|-- scripts/                   # scripturi de pornire pentru experimente concrete
|-- docs/latex/                # documentatia proiectului in LaTeX si PDF
|-- requirements.txt           # dependinte principale, fara PyTorch
|-- requirements-torch-cpu.txt # PyTorch pentru instalare CPU
`-- README.md
```

Folderele de rezultate locale nu sunt necesare pentru rularea codului sursa si nu ar trebui urcate in repository:

```text
checkpoints/
tensorboard_logs/
tensorboard_sweeps/
run_logs/
datasets/
__pycache__/
venv/
```

## Componente principale

### `blocks.py`

Defineste toate piesele folosite in joc. Fiecare piesa este o matrice binara, iar formele sunt grupate dupa dimensiune si complexitate. Aceste categorii sunt folosite de generatorul adaptiv din motorul jocului.

### `engine.py`

Contine logica determinista a jocului:

- initializarea tablei;
- verificarea unei plasari;
- aplicarea piesei pe tabla;
- stergerea liniilor si coloanelor complete;
- detectarea finalului de joc;
- generarea mainii urmatoare.

Motorul nu contine logica de invatare. El poate fi folosit atat de mediul de antrenare, cat si de interfata vizuala.

### `env.py`

Transforma motorul jocului intr-un mediu Gymnasium. Aici sunt definite:

- spatiul de observatii;
- spatiul de actiuni;
- masca actiunilor valide;
- reward-ul pentru linii;
- reward-ul de contact;
- optional, penalizarea pentru gauri;
- extractoarele CNN folosite de PPO.

Actiunea are forma:

```text
actiune = index_piesa * board_size * board_size + rand * board_size + coloana
```

Pentru tabla 4x4 exista 48 actiuni posibile, iar pentru 8x8 exista 192 actiuni posibile.

### `train.py`

Este punctul principal de pornire pentru antrenare. Scriptul creeaza mediile paralele, configureaza MaskablePPO, porneste antrenarea, salveaza modelul final si poate salva checkpoint-uri intermediare.

Optiunile importante sunt:

- `--cnn-arch base|v2|actionaware`
- `--hand-generator solvable|playable|adaptive_playable|random`
- `--shape-pool ...`
- `--gamma`
- `--learning-rate`
- `--n-steps`
- `--batch-size`
- `--eval-freq`
- `--checkpoint-freq`
- `--resume`
- `--init-from-model`

### `watch.py`

Interfata vizuala folosita pentru inspectarea comportamentului. Poate rula:

- modelul antrenat;
- euristica de referinta;
- joc manual;
- comparatie intre jucator si AI;
- editor/solver pentru situatii construite manual.

Aceasta interfata este importanta deoarece metricile din TensorBoard nu arata mereu de ce modelul ia o anumita decizie.

### `behavioral_cloning.py`

Modul experimental pentru invatare supervizata din euristica. El poate colecta exemple generate de euristica si poate pre-antrena o politica pe aceste exemple. Rezultatul poate fi apoi folosit in `train.py` cu `--init-from-model`.

### `scripts/`

Au ramas doar scripturile utile pentru rularile reprezentative:

- `run_basecnn_contact_threshold_gamma06.sh` - antrenare 8x8 cu CNN de baza si contact threshold;
- `run_immediate_contact_gamma04_lines.sh` - varianta finala soft generator/contact/line reward;
- `run_4x4_best8x8_basecnn.sh` - testul 4x4 cu parametri apropiati de modelul 8x8;
- `run_behavioral_cloning.py` - colectare date euristice si pre-antrenare supervizata.

Scripturile de cloud, smoke test si sweep-uri scurte au fost eliminate din varianta curata a proiectului.

## Instalare

Se recomanda un mediu virtual separat:

```bash
python -m venv venv
source venv/bin/activate
```

Instalare PyTorch pe CPU:

```bash
pip install -r requirements-torch-cpu.txt
```

Instalare dependinte principale:

```bash
pip install -r requirements.txt
```

Pentru CUDA sau alte variante de PyTorch, instalarea trebuie adaptata dupa platforma folosita.

## Antrenare rapida

Exemplu de rulare PPO cu generator adaptiv si CNN de baza:

```bash
python train.py \
  --shape-pool mini \
  --hand-generator adaptive_playable \
  --cnn-arch base \
  --gamma 0.3 \
  --learning-rate 0.002 \
  --n-steps 1024 \
  --batch-size 1024 \
  --total-timesteps 1500000 \
  --model-path checkpoints/cnn/demo_model \
  --tb-log-name demo_model \
  --log-dir tensorboard_logs \
  --checkpoint-dir checkpoints/cnn/demo_model \
  --checkpoint-freq 250000
```

Reluarea unui model existent:

```bash
python train.py \
  --resume \
  --model-path checkpoints/cnn/demo_model
```

Initializarea unui model nou cu greutati dintr-un model existent:

```bash
python train.py \
  --no-resume \
  --init-from-model checkpoints/cnn/demo_model \
  --model-path checkpoints/cnn/new_finetune_model
```

## TensorBoard

Pornire TensorBoard:

```bash
tensorboard --logdir tensorboard_logs
```

Metrici importante:

- `rollout/ep_len_mean` - durata medie a episoadelor in timpul antrenarii;
- `rollout/ep_rew_mean` - recompensa medie;
- `eval_stages/mean` - performanta medie in evaluari separate;
- `eval_stages/max` - cel mai bun episod din evaluare;
- `step_stats/reward_contact` - contributia reward-ului de contact;
- `step_stats/reward_line` - contributia reward-ului pentru linii;
- `time/fps` - viteza de rulare.

## Testare vizuala

Interfata se porneste cu:

```bash
python watch.py
```

In functie de configuratia din `watch.py`, se poate incarca un model salvat, se poate rula euristica sau se poate testa manual generatorul de piese.

Daca nu exista local checkpoint-ul default, interfata poate fi pornita fara model PPO:

```bash
python watch.py --no-model
```

## Benchmark FPS

Pentru a vedea configuratia locala mai rapida:

```bash
python train.py --benchmark --benchmark-steps 30000
```

Benchmark-ul compara mai multe combinatii de medii paralele, device si thread-uri PyTorch.

## Documentatie

Documentatia finala este in:

```text
docs/latex/main.tex
docs/latex/main.pdf
```

Compilare recomandata:

```bash
cd docs/latex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error main.tex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Se foloseste XeLaTeX pentru Times New Roman. `pdflatex` nu este recomandat pentru aceasta documentatie.

## Workflow general

1. `blocks.py` defineste piesele.
2. `engine.py` genereaza o mana si aplica regulile jocului.
3. `env.py` converteste starea jocului in observatii pentru model.
4. `train.py` creeaza mai multe medii paralele si antreneaza PPO.
5. TensorBoard salveaza metricile din timpul rularii.
6. `watch.py` permite inspectarea modelului dupa antrenare.

## Fisiere care merita pastrate in repository

Pentru o varianta curata de push, merita pastrate:

- fisierele sursa principale: `blocks.py`, `engine.py`, `env.py`, `train.py`, `watch.py`;
- `behavioral_cloning.py`, daca vrei sa pastrezi experimentul cu teacher heuristic;
- `scripts/` doar pentru scripturile de rulare inca relevante;
- `docs/latex/`, pentru documentatia proiectului;
- `requirements.txt` si `requirements-torch-cpu.txt`;
- `.gitignore` si `README.md`.

Se recomanda excluderea output-urilor mari si regenerate automat: modele, loguri, cache-uri, medii virtuale si dataset-uri generate local.
