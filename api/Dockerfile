# ═══════════════════════════════════════════════════════════════
# Dockerfile — Video VLM Platform
# Python 3.12 + PyTorch ≥2.7 cu128 + SAM 3.1 + GroundingDINO HF
# RTX 5090 (sm_120 Blackwell) — CUDA 12.8
# ═══════════════════════════════════════════════════════════════
FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Dépendances système ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 \
    gcc g++ ninja-build \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch cu128 (RTX 5090 sm_120 Blackwell) ─────────────────
# Doit être installé EN PREMIER avant tous les autres packages.
# Versions stables épinglées : l'index nightly purge ses wheels, ce qui
# rendait le build irreproductible (torchvision nightly sans torch assorti).
RUN pip install --upgrade pip && \
    pip install \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ── SAM 3.1 (Meta, mars 2026) ─────────────────────────────────
# Requiert Python 3.12+ et PyTorch 2.7+ (déjà installés ci-dessus)
# Accès HuggingFace requis : huggingface-cli login avant d'utiliser les checkpoints
RUN pip install "git+https://github.com/facebookresearch/sam3.git"

# ── flash-attn-3 (optimisation attention pour SAM 3.1) ─────────
# Optionnel mais recommandé pour les performances
RUN pip install einops ninja && \
    pip install flash-attn --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 || \
    echo "flash-attn non disponible — skip"

# ── Dépendances Python ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
