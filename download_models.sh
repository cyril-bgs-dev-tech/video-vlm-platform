#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# download_models.sh — Téléchargement des poids modèles
# Usage : chmod +x download_models.sh && ./download_models.sh
# ═══════════════════════════════════════════════════════════════
set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "══════════════════════════════════════════"
echo "  Video VLM Platform — Téléchargement modèles"
echo "══════════════════════════════════════════"

# ── SAM2 (Facebook Research) ──────────────────────────────────
echo ""
echo "→ SAM2 Hiera Large..."
mkdir -p "$MODELS_DIR/sam2"
wget -q --show-progress -P "$MODELS_DIR/sam2/" \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# ── GroundingDINO ────────────────────────────────────────────
echo ""
echo "→ GroundingDINO SwinB..."
mkdir -p "$MODELS_DIR/GroundingDINO"
wget -q --show-progress -P "$MODELS_DIR/GroundingDINO/" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
wget -q --show-progress -P "$MODELS_DIR/GroundingDINO/" \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py

# ── YOLO-World v2 ────────────────────────────────────────────
echo ""
echo "→ YOLO-World v2-X..."
wget -q --show-progress -P "$MODELS_DIR/" \
  https://github.com/AILab-CVC/YOLO-World/releases/download/v2.0/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain.pt
mv "$MODELS_DIR/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain.pt" \
   "$MODELS_DIR/yolo_world_v2_x.pt"

# ── OSNet Re-ID (pour BoxMOT) ────────────────────────────────
echo ""
echo "→ OSNet-x0.25 (Re-ID)..."
wget -q --show-progress -P "$MODELS_DIR/" \
  https://drive.google.com/uc?id=1N2nnM-3HlXCEBB5OFKbJGkpFbvBQgfhJ
# Alternative via HuggingFace :
# huggingface-cli download mikel-brostrom/boxmot osnet_x0_25_market1501.pt --local-dir "$MODELS_DIR/"

# ── LLaVA-NeXT-Video 7B (HuggingFace) ───────────────────────
echo ""
echo "→ LLaVA-NeXT-Video-7B (HuggingFace)..."
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download lmms-lab/LLaVA-NeXT-Video-7B \
    --local-dir "$MODELS_DIR/llava-next-video-7b" \
    --exclude "*.bin"  # préférer safetensors
else
  echo "  ⚠️  huggingface-cli non installé. Installer : pip install huggingface_hub[cli]"
  echo "  Puis : huggingface-cli download lmms-lab/LLaVA-NeXT-Video-7B --local-dir $MODELS_DIR/llava-next-video-7b"
fi

# ── CLIP ViT-L/14 (via open_clip — téléchargement automatique au premier run) ──
echo ""
echo "→ CLIP ViT-L/14 (sera téléchargé automatiquement au premier run via open_clip)"

# ── VLM Grounding Models (RTX 5090 32GB) ─────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  VLM Grounding Models (Nouveau!)"
echo "══════════════════════════════════════════"

# ── LocateAnything-3B (NVIDIA) ─────────────────────────────────────────────
echo ""
echo "→ LocateAnything-3B (NVIDIA) — Parallel Box Decoding..."
echo "  ~12GB VRAM — Optimisé détection multi-objets"
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download nvidia/LocateAnything-3B \
    --local-dir "$MODELS_DIR/locate_anything_3b" \
    --exclude "*.bin"
else
  echo "  ⚠️  huggingface-cli non installé. Installer : pip install huggingface_hub[cli]"
  echo "  Puis : huggingface-cli download nvidia/LocateAnything-3B --local-dir $MODELS_DIR/locate_anything_3b"
fi

# ── Mini-InternVL-4B ───────────────────────────────────────────────────────
echo ""
echo "→ Mini-InternVL-4B — 5% params, 90% performance..."
echo "  ~8-10GB VRAM — Ultra efficace"
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 \
    --local-dir "$MODELS_DIR/mini_internvl_4b" \
    --exclude "*.bin"
else
  echo "  ⚠️  huggingface-cli non installé. Installer : pip install huggingface_hub[cli]"
  echo "  Puis : huggingface-cli download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir $MODELS_DIR/mini_internvl_4b"
fi

# ── InternVL3.5-8B ─────────────────────────────────────────────────────────
echo ""
echo "→ InternVL3.5-8B — SOTA open-source..."
echo "  ~14-16GB VRAM — Variable Visual Position Encoding"
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download OpenGVLab/InternVL3_5-8B \
    --local-dir "$MODELS_DIR/internvl35_8b" \
    --exclude "*.bin"
else
  echo "  ⚠️  huggingface-cli non installé. Installer : pip install huggingface_hub[cli]"
  echo "  Puis : huggingface-cli download OpenGVLab/InternVL3_5-8B --local-dir $MODELS_DIR/internvl35_8b"
fi

echo ""
echo "══════════════════════════════════════════"
echo "✅ Téléchargements terminés dans : $MODELS_DIR"
echo ""
echo "Contenu :"
find "$MODELS_DIR" -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" 2>/dev/null | head -20
echo "══════════════════════════════════════════"
