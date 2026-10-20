# 🎬 Video VLM Platform

Stack d'analyse vidéo temps réel sur **RTX 5090**, orchestrée via Docker Compose.

## Architecture

```
Streamlit (8501) ──► FastAPI (8000) ──► Celery Workers (GPU)
                           │
                     Redis / Qdrant / PostgreSQL / MinIO
```

## Interface (5 pages + accueil)

| Fichier | Page | Contenu |
|---------|------|---------|
| `app.py` | 🎯 Accueil | Health check API/workers, catalogue des 10 vidéos de démo |
| `pages/1_demo_detection.py` | 🎬 Démo détection — live | Upload ou vidéo de démo, presets de requêtes, flux WebSocket temps réel (bbox + labels + scores), **démo rapide multi-modèles** : évaluation séquentielle, métriques par modèle (détections, confiance, ms/frame, P95, FPS), comparatif et replay annoté |
| `pages/2_analyses_detail.py` | 📊 Analyses détaillées | Vues par analyse (détection, OCR, clustering, actions) |
| `pages/3_rapport.py` | 📋 Rapport intelligent | Synthèse générée à partir des analyses |
| `pages/4_benchmark_kpis.py` | 📐 Benchmark & KPIs DataOps | 20 datasets multi-domaines (COCO, MOT20, VisDrone, BDD100K, DOTA, ExDark, FLIR…), sélection automatique du modèle par dataset, matrice dataset → modèle |
| `pages/5_compositions_vlm.py` | 🧬 Compositions VLM | Compositions hiérarchique / ensembling / itérative, **en cours d'entretien** : toutes reposent sur LocateAnything-3B |

## État réel des modèles de détection

Vérifié en end-to-end (upload → WebSocket) sur RTX 5090, vidéo 1280×720. Le champ
`mock` du message `started` distingue l'inférence réelle de la simulation ; l'UI
affiche « Mode mock » le cas échéant. Les latences sont des moyennes sur une passe
courte (2 à 12 frames) et **incluent le chargement du modèle** — elles surestiment
donc le régime établi, d'autant plus que la passe est courte.

| Modèle | État | Latence moyenne mesurée | Notes |
|--------|------|-------------------------|-------|
| `yolo_visdrone` | ✅ Réel | 17–64 ms/frame | YOLOv11s fine-tuné VisDrone2019, 10 classes fixes, vues aériennes |
| `grounding_dino` | ✅ Réel | ~175 ms/frame | fp32 + autocast fp16 (le fp16 intégral n'est pas supporté) |
| `yolo_world_v2` | ✅ Réel | ~210 ms/frame sur 12 frames | Open-vocabulaire ; descend à ~25 ms/frame une fois chargé |
| `mini_internvl_4b` | ✅ Réel | ~15 s/frame | Grounding via `model.chat`, tuiles 448 |
| `internvl35_8b` | ✅ Réel | ~14 s/frame | Un seul gros VLM en VRAM à la fois (éviction automatique) |
| `locate_anything_3b` | 🔧 En cours d'entretien | — | Le code remote (Parallel Box Decoding) est incompatible transformers v5 : le modèle charge et génère, mais la sortie est dégénérée et ne produit aucune boîte. Son chargement est court-circuité (`LOCATE_ANYTHING_DEGRADED=0` pour le réactiver). Dans l'interface, il est affiché « en cours d'entretien » et ne peut pas être lancé |

`ALLOW_MOCK=0` désactive le repli sur simulation : un modèle indisponible renvoie alors une erreur
explicite au lieu de résultats factices.

## Démarrage rapide

### 1. Prérequis

- Docker 24+ avec NVIDIA Container Toolkit
- RTX 5090 (ou toute GPU CUDA 12.x)
- Drivers CUDA 12.8 (wheels PyTorch cu128)

```bash
# Vérifier NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 2. Cloner et configurer

```bash
git clone https://github.com/cyril-bgs-dev-tech/video-vlm-platform.git
# miroir : https://gitlab.com/cyril.bgs.dev/video-vlm-platform.git
cd video-vlm-platform
cp .env.example .env
```

### 3. Télécharger les modèles

```bash
# Créer le volume modèles
mkdir -p models

# SAM2 (Facebook Research)
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# GroundingDINO
mkdir -p models/GroundingDINO
wget -P models/GroundingDINO/ \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# YOLO-World v2
wget -P models/ https://github.com/AILab-CVC/YOLO-World/releases/download/v2.0/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain.pt

# LLaVA-NeXT-Video (HuggingFace)
huggingface-cli download lmms-lab/LLaVA-NeXT-Video-7B --local-dir models/llava-next-video-7b

# VLM Grounding Models (Nouveau! — RTX 5090 32GB)
# LocateAnything-3B (NVIDIA) — Parallel Box Decoding
huggingface-cli download nvidia/LocateAnything-3B --local-dir models/locate_anything_3b

# Mini-InternVL-4B — 5% params, 90% performance
huggingface-cli download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir models/mini_internvl_4b

# InternVL3.5-8B — SOTA open-source
huggingface-cli download OpenGVLab/InternVL3_5-8B --local-dir models/internvl35_8b
```

### 4. Lancer la stack

```bash
docker compose up -d

# Suivre les logs
docker compose logs -f streamlit api worker_vlm

# Accès
open http://localhost:8501   # Streamlit
open http://localhost:8000/docs  # FastAPI Swagger
open http://localhost:5555   # Flower (Celery monitor)
open http://localhost:9001   # MinIO Console
```

### 5. Vérifier les workers GPU

```bash
# Status workers Celery
docker compose exec api celery -A workers.celery_app inspect active

# GPU utilisation
docker compose exec worker_vlm nvidia-smi
```

## Variables d'environnement (.env)

```env
# GPU
NVIDIA_VISIBLE_DEVICES=0

# Modèles par défaut
DEFAULT_VLM=llava-next-video-7b
DEFAULT_CLIP=ViT-L/14
DEFAULT_DETECTOR=grounding_dino_v2
DEFAULT_TRACKER=botsort

# Limites
MAX_VIDEO_SIZE_MB=2048
MAX_VIDEO_DURATION_SEC=3600

# Sécurité (optionnel)
API_KEY=changeme
```

## Optimisations RTX 5090

- **BF16** natif (Ampere+) via `torch.bfloat16`
- **FlashAttention-2** pour les VLM (attention 4× plus rapide)
- **TensorRT** : `torch.compile()` sur les workers detection
- **Batch inference** configurable par page
- **Multi-worker** : VLM / Detection / Clustering sur 3 workers séparés
- **fp16** GroundingDINO + YOLO-World
- **Decord** pour décodage vidéo GPU (bypass OpenCV CPU)

## Modèles VLM Grounding (Détection Multi-Objets)

La plateforme intègre 3 modèles VLM de dernière génération pour la détection d'objets multiples dans les flux vidéo, optimisés pour RTX 5090 (32GB VRAM):

### LocateAnything-3B (NVIDIA) — Recommandé
- **Paramètres**: 3B (4B avec vision encoder)
- **VRAM**: ~12GB en BF16
- **Innovation**: Parallel Box Decoding (PBD) — prédit les bounding boxes en parallèle
- **Performance**: Jusqu'à 2.5× plus rapide que les approches autorégressives
- **Cas d'usage**: Détection multi-objets, scènes denses, open-vocabulary
- **Licence**: NVIDIA (recherche/non-commercial)
- **Paper**: [arXiv:2605.27365](https://arxiv.org/abs/2605.27365)

### Mini-InternVL-4B
- **Paramètres**: 4B (5% d'InternVL standard)
- **VRAM**: ~8-10GB en BF16
- **Performance**: 90% des performances avec 5% des paramètres
- **Cas d'usage**: Edge devices, GPUs consumer-grade, déploiement léger
- **Paper**: [arXiv:2410.16261](https://arxiv.org/abs/2410.16261)

### InternVL3.5-8B
- **Paramètres**: 8B
- **VRAM**: ~14-16GB en BF16
- **Features**: Variable Visual Position Encoding, Native Multimodal Pre-Training
- **Performance**: SOTA parmi les modèles open-source
- **Cas d'usage**: Détection haute précision, raisonnement visuel complexe
- **Paper**: [arXiv:2508.18265](https://arxiv.org/abs/2508.18265)

### Comparaison des modèles

| Modèle | VRAM | Vitesse | Précision | Cas d'usage |
|--------|------|---------|-----------|-------------|
| LocateAnything-3B | ~12GB | ⚡⚡⚡ Très rapide | ⭐⭐⭐⭐ | Multi-objets, temps réel |
| Mini-InternVL-4B | ~8-10GB | ⚡⚡ Rapide | ⭐⭐⭐⭐ | Edge, ressources limitées |
| InternVL3.5-8B | ~14-16GB | ⚡ Moyen | ⭐⭐⭐⭐⭐ | Haute précision, reasoning |
| GroundingDINO | ~2GB | ⚡⚡⚡ Très rapide | ⭐⭐⭐ | Détection classique |
| YOLO-World v2 | ~4GB | ⚡⚡⚡⚡ Ultra rapide | ⭐⭐⭐ | Open-vocabulaire rapide |

## Structure des queues Celery

| Queue | Worker | Modèles |
|-------|--------|---------|
| `vlm_queue` | `worker_vlm` | SAM2, LLaVA, VideoMAE, Action Rec |
| `detection_queue` | `worker_detection` | DINO, YOLO, OCR, Re-ID, ByteTrack |
| `clustering_queue` | `worker_clustering` | CLIP, UMAP, HDBSCAN, Qdrant |

## Ajout de modèles

1. Ajouter le modèle dans `api/workers/vlm_worker.py` dans `_loaded`
2. Exposer une route FastAPI dans `api/main.py`
3. Ajouter le choix dans la page Streamlit correspondante

## Troubleshooting

```bash
# OOM GPU — réduire batch_size ou activer 8-bit quantization
docker compose exec worker_vlm nvidia-smi

# Redis queue bloquée
docker compose exec redis redis-cli FLUSHALL

# Reconstruire sans cache
docker compose build --no-cache api
```
