#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# download_models.sh — Téléchargement automatique des poids
# SAM 3.1 nécessite une authentification HuggingFace
# ═══════════════════════════════════════════════════════════════
set -e

MODELS_DIR="${MODELS_PATH:-/models}"
HF_TOKEN="${HF_TOKEN:-}"
echo "📦 Vérification des modèles dans $MODELS_DIR..."

# ── SAM 3.1 via HuggingFace (accès restreint) ──────────────────
SAM3_DIR="$MODELS_DIR/sam3"
SAM3_LARGE="$SAM3_DIR/sam3.1_large.pt"

if [ ! -f "$SAM3_LARGE" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "⚠️  SAM 3.1 : HF_TOKEN non défini."
        echo "   1. Demandez l'accès sur : https://huggingface.co/facebook/sam3"
        echo "   2. Créez un token sur   : https://huggingface.co/settings/tokens"
        echo "   3. Ajoutez HF_TOKEN=hf_xxx dans votre .env"
        echo "   → Fallback sur SAM 2.1 pour l'instant"
        USE_SAM21=true
    else
        echo "⬇️  SAM 3.1 large via HuggingFace..."
        mkdir -p "$SAM3_DIR"
        python3 -c "
from huggingface_hub import hf_hub_download
import os
token = os.environ['HF_TOKEN']
# Checkpoints SAM 3.1 sur HuggingFace facebook/sam3
files = ['sam3.1_large.pt', 'sam3.1_base_plus.pt']
for f in files:
    try:
        path = hf_hub_download(
            repo_id='facebook/sam3',
            filename=f,
            local_dir='$SAM3_DIR',
            token=token
        )
        print(f'✅ Téléchargé : {f}')
    except Exception as e:
        print(f'⚠️  {f} non disponible : {e}')
" || USE_SAM21=true
    fi
else
    echo "✅ SAM 3.1 présent : $SAM3_LARGE"
fi

# ── SAM 2.1 (fallback si SAM 3.1 non disponible) ──────────────
SAM21_DIR="$MODELS_DIR/sam2"
SAM21_FILE="$SAM21_DIR/sam2.1_hiera_large.pt"

if [ "${USE_SAM21:-false}" = "true" ] || [ ! -f "$SAM3_LARGE" ]; then
    if [ ! -f "$SAM21_FILE" ]; then
        echo "⬇️  SAM 2.1 large (fallback, 856 Mo)..."
        mkdir -p "$SAM21_DIR"
        wget -q --show-progress \
            -O "$SAM21_FILE" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        echo "✅ SAM 2.1 téléchargé"
    else
        echo "✅ SAM 2.1 présent (fallback) : $SAM21_FILE"
    fi
fi

# ── YOLO-World v2 X ────────────────────────────────────────────
YOLO_FILE="$MODELS_DIR/yolo_world_v2_x.pt"
YOLO_SIZE=0
if [ -f "$YOLO_FILE" ]; then
    YOLO_SIZE=$(stat -c%s "$YOLO_FILE" 2>/dev/null || echo 0)
fi
if [ ! -f "$YOLO_FILE" ] || [ "${YOLO_SIZE:-0}" -lt 1000000 ]; then
    echo "⬇️  YOLO-World / YOLOv8-World (~140–300 Mo)..."
    rm -f "$YOLO_FILE"
    # Ultralytics official world weights (compatible YOLOWorld)
    wget -q --show-progress \
        -O "$YOLO_FILE" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt" \
     || wget -q --show-progress \
        -O "$YOLO_FILE" \
        "https://github.com/AILab-CVC/YOLO-World/releases/download/v2.0/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain.pt"
    echo "✅ YOLO-World téléchargé ($(stat -c%s "$YOLO_FILE" 2>/dev/null || echo ?) bytes)"
else
    echo "✅ YOLO-World présent : $YOLO_FILE ($YOLO_SIZE bytes)"
fi

# ── GroundingDINO HF (cache automatique via transformers) ──────
echo "✅ GroundingDINO HF : cache dans $MODELS_DIR/hf_cache au premier appel"

echo ""
echo "✅ Vérification modèles terminée — démarrage worker..."
