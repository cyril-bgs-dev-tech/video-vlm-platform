# Guide d'installation — Video VLM Platform

## Contexte

Certains packages ML ne se compilent pas nativement sur Windows
(`flash-attn`, `bitsandbytes`, `decord`, `faiss-gpu`).
**L'approche recommandée est Docker**, mais si tu veux un environnement
local pour développer, voici les deux chemins.

---

## Option A — Docker (recommandé, Windows/Linux/Mac)

```bash
# Prérequis : Docker Desktop + NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Lancer toute la stack
cp .env.example .env
make up

# Accès
# Streamlit  → http://localhost:8501
# API docs   → http://localhost:8000/docs
# Flower     → http://localhost:5555
```

---

## Option B — Environnement local (développement)

### Prérequis

- Python 3.11.x (via conda ou pyenv)
- CUDA 12.4 + drivers NVIDIA à jour
- Sur Windows : **WSL2** fortement recommandé pour les packages GPU

### Étape 1 — Créer l'environnement conda

```bash
conda create -n vlm-platform python=3.11.9
conda activate vlm-platform
```

### Étape 2 — PyTorch avec CUDA (OBLIGATOIRE EN PREMIER)

```bash
# RTX 5090 → CUDA 12.4
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Vérifier
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Étape 3 — Flash-Attention (Linux / WSL2 uniquement)

```bash
# Nécessite torch installé ET un compilateur C++ (gcc)
# Sur Windows natif : SAUTER cette étape
pip install flash-attn --no-build-isolation
```

> **Windows natif** : flash-attn n'a pas de wheel Windows.
> Le code fonctionne sans (les VLM utilisent `attn_implementation="eager"`
> comme fallback automatique).

### Étape 4 — Dépendances API

```bash
pip install -r api/requirements.txt
```

### Étape 5 — PaddleOCR (selon plateforme)

```bash
# Linux GPU
pip install paddlepaddle-gpu==2.6.2 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Windows ou CPU uniquement
pip install paddlepaddle==2.6.2

# Puis dans les deux cas
pip install paddleocr>=2.8.0
```

### Étape 6 — FAISS (selon plateforme)

```bash
# Linux avec GPU
pip install faiss-gpu

# Windows ou sans GPU
pip install faiss-cpu
```

### Étape 7 — SAM2 (depuis GitHub)

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Étape 8 — GroundingDINO (depuis GitHub)

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Étape 9 — Streamlit uniquement

```bash
pip install -r streamlit_app/requirements.txt
```

### Étape 10 — Tests

```bash
pip install -r tests/requirements-test.txt
pytest tests/ -v
```

---

## Packages non disponibles sur Windows natif

| Package | Raison | Solution |
|---------|--------|----------|
| `flash-attn` | Nécessite gcc/nvcc | Sauter — fallback automatique |
| `bitsandbytes` | Pas de wheel Windows officiel | Sauter ou utiliser WSL2 |
| `decord` | Pas de wheel Windows | Remplacé par PyAV (`av`) |
| `faiss-gpu` | Linux uniquement | Utiliser `faiss-cpu` |
| `paddlepaddle-gpu` | Linux uniquement | Utiliser `paddlepaddle` (CPU) |

> Toutes ces limitations disparaissent avec **Docker + WSL2** ou
> en développant directement sur Linux.

---

## Vérification rapide de l'installation

```bash
python -c "
import torch, transformers, streamlit, fastapi, celery
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('streamlit:', streamlit.__version__)
print('fastapi:', fastapi.__version__)
print('celery:', celery.__version__)
"
```
