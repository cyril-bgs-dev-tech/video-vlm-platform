# Intégration VLM Grounding — Guide Rapide

## Modèles intégrés

3 modèles VLM de dernière génération ont été ajoutés pour la détection d'objets multiples sur flux vidéo:

### 1. LocateAnything-3B (NVIDIA) ⭐ Recommandé
- **VRAM**: ~12GB
- **Spécialité**: Parallel Box Decoding — détection multi-objets ultra-rapide
- **Performance**: 2.5× plus rapide que les approches classiques
- **License**: NVIDIA (recherche/non-commercial)

### 2. Mini-InternVL-4B
- **VRAM**: ~8-10GB
- **Spécialité**: Efficacité maximale — 90% performance avec 5% paramètres
- **Performance**: Idéal pour ressources limitées

### 3. InternVL3.5-8B
- **VRAM**: ~14-16GB
- **Spécialité**: SOTA open-source — haute précision
- **Performance**: Meilleure précision pour scènes complexes

## Installation

### 1. Télécharger les modèles

```bash
cd /mnt/d/projet\ vlm/video-vlm-platform

# Option A: Script automatique (tous les modèles)
chmod +x download_models.sh
./download_models.sh

# Option B: Télécharger individuellement
# LocateAnything-3B
huggingface-cli download nvidia/LocateAnything-3B \
  --local-dir models/locate_anything_3b

# Mini-InternVL-4B
huggingface-cli download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 \
  --local-dir models/mini_internvl_4b

# InternVL3.5-8B
huggingface-cli download OpenGVLab/InternVL3_5-8B \
  --local-dir models/internvl35_8b
```

### 2. Lancer la plateforme

```bash
# Docker (recommandé)
docker compose up -d

# Ou en local (Windows)
./start.bat

# Ou en local (Linux/Mac)
make start
```

### 3. Accéder à l'interface

```
Streamlit: http://localhost:8501
API Docs:  http://localhost:8000/docs
```

## Utilisation

### Via l'interface Streamlit

1. Aller sur **Page 1 — Démo détection VLM**
2. Uploader une vidéo
3. Sélectionner les objets à détecter (presets ou personnalisé)
4. **Choisir le modèle VLM** dans le sélecteur:
   - LocateAnything-3B (recommandé pour multi-objets)
   - Mini-InternVL-4B (ressources limitées)
   - InternVL3.5-8B (haute précision)
5. Lancer la détection
6. Visualiser les résultats avec tracking

### Via l'API

```bash
# Lancer une détection avec LocateAnything-3B
curl -X POST http://localhost:8000/detection/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "votre_video_id",
    "model": "locate_anything_3b",
    "queries": ["person", "car", "bicycle"],
    "confidence": 0.35,
    "track": true
  }'

# Avec Mini-InternVL-4B
curl -X POST http://localhost:8000/detection/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "votre_video_id",
    "model": "mini_internvl_4b",
    "queries": ["person wearing red"],
    "confidence": 0.4,
    "track": true
  }'

# Avec InternVL3.5-8B
curl -X POST http://localhost:8000/detection/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "votre_video_id",
    "model": "internvl35_8b",
    "queries": ["traffic sign", "stop sign"],
    "confidence": 0.5,
    "track": true
  }'
```

## Monitoring GPU

```bash
# Vérifier l'utilisation GPU
docker compose exec worker_detection nvidia-smi

# Ou en continu
watch -n 1 nvidia-smi
```

## Comparaison des performances

| Modèle | VRAM | Vitesse | Précision | Cas d'usage |
|--------|------|---------|-----------|-------------|
| LocateAnything-3B | ~12GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Multi-objets, temps réel |
| Mini-InternVL-4B | ~8-10GB | ⚡⚡ | ⭐⭐⭐⭐ | Edge, ressources limitées |
| InternVL3.5-8B | ~14-16GB | ⚡ | ⭐⭐⭐⭐⭐ | Haute précision |
| GroundingDINO | ~2GB | ⚡⚡⚡ | ⭐⭐⭐ | Détection classique |
| YOLO-World v2 | ~4GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Open-vocabulaire rapide |

## Dépannage

### OOM GPU (Out of Memory)
```bash
# Réduire le batch_size dans la requête
# Ou utiliser un modèle plus petit (Mini-InternVL-4B)
# Ou activer la quantification 8-bit
```

### Modèle non trouvé
```bash
# Vérifier que les modèles sont téléchargés
ls models/locate_anything_3b/
ls models/mini_internvl_4b/
ls models/internvl35_8b/

# Re-télécharger si nécessaire
huggingface-cli download nvidia/LocateAnything-3B --local-dir models/locate_anything_3b
```

### Erreur de chargement
```bash
# Vérifier les logs
docker compose logs worker_detection

# Redémarrer le worker
docker compose restart worker_detection
```

## Architecture technique

```
Streamlit (UI)
    ↓
FastAPI (API Gateway)
    ↓
Celery Workers (GPU)
    ├── worker_detection
    │   ├── LocateAnything-3B
    │   ├── Mini-InternVL-4B
    │   ├── InternVL3.5-8B
    │   ├── GroundingDINO
    │   └── YOLO-World v2
    ├── worker_vlm
    └── worker_clustering
```

## Fichiers modifiés

- `api/workers/detection_worker.py` — Ajout des loaders et inférence VLM
- `streamlit_app/pages/1_demo_detection.py` — Sélecteur de modèle VLM
- `download_models.sh` — Script de téléchargement
- `README.md` — Documentation

## Références

- LocateAnything: https://arxiv.org/abs/2605.27365
- Mini-InternVL: https://arxiv.org/abs/2410.16261
- InternVL3.5: https://arxiv.org/abs/2508.18265
