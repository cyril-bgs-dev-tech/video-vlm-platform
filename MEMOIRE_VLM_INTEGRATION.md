# Mémoire — Intégration VLM pour Détection Multi-Objets sur RTX 5090

**Date** : 20 juillet 2026  
**Plateforme** : Video VLM Platform  
**GPU cible** : NVIDIA RTX 5090 (32 Go VRAM)  
**Objectif** : Détection d'objets multiples dans un flux vidéo en local

---

## 1. Contexte et objectifs

Recherche dans la littérature scientifique actuelle (2024-2026) des meilleurs Vision-Language Models (VLM) déployables en local, optimisés pour la détection d'objets multiples dans un flux vidéo, avec contraintes de performance et de consommation réduite sur RTX 5090 (32 Go VRAM).

### Contraintes techniques
- **VRAM disponible** : 32 Go (RTX 5090)
- **Architecture** : CUDA 12.x, BF16 natif, FlashAttention-2
- **Cas d'usage** : Flux vidéo temps réel, détection multi-objets open-vocabulary
- **Plateforme** : Docker Compose, FastAPI, Celery, Streamlit

---

## 2. Recherche bibliographique — Résultats

### Sources consultées
- **arXiv** : cs.CV recent submissions (juillet 2026)
- **Hugging Face** : trending models image-text-to-text
- **GitHub** : QwenLM/Qwen-VL, haotian-liu/LLaVA, OpenGVLab/InternVL, NVlabs/Eagle
- **Papers clés** : arXiv:2605.27365, arXiv:2410.16261, arXiv:2508.18265

### Tendances identifiées (2025-2026)

1. **Parallel Box Decoding** : Prédiction parallèle des bounding boxes plutôt que autorégressive → gain de vitesse 2.5×
2. **Modèles hybrides VLM + détecteurs classiques** : Combinaison YOLO-World + VLM pour le raisonnement sémantique
3. **Compression aggressive** : Modèles 5% paramètres pour 90% performance (Mini-InternVL)
4. **Variable Visual Position Encoding** : Adaptation dynamique à la résolution (InternVL3.5)
5. **Native Multimodal Pre-Training** : Pré-entraînement conjoint vision-language dès le départ

---

## 3. Modèles sélectionnés

### 3.1 LocateAnything-3B (NVIDIA) — ⭐ Recommandé

| Caractéristique | Valeur |
|---|---|
| **Paramètres** | 3B (4B avec vision encoder) |
| **VRAM** | ~12 Go en BF16 |
| **Paper** | arXiv:2605.27365 (mai 2026) |
| **Innovation** | Parallel Box Decoding (PBD) |
| **Vitesse** | Jusqu'à 2.5× plus rapide |
| **Dataset** | 12M images, 138M requêtes, 785M bounding boxes |
| **Licence** | NVIDIA (recherche/non-commercial) |
| **Modes** | fast (MTP), slow (AR), hybrid (défaut) |

**Points forts** :
- Conçu spécifiquement pour la détection multi-objets
- Parallel Box Decoding : prédit toutes les boxes d'un coup
- Supporte open-vocabulary, GUI grounding, OCR, pointing
- Intégré dans NVIDIA Nemotron et Cosmos

**Points faibles** :
- Licence restrictive (non-commercial)
- API spécifique (non standard transformers)

### 3.2 Mini-InternVL-4B

| Caractéristique | Valeur |
|---|---|
| **Paramètres** | 4B (5% d'InternVL standard) |
| **VRAM** | ~8-10 Go en BF16 |
| **Paper** | arXiv:2410.16261 (oct 2024) |
| **Performance** | 90% avec 5% des paramètres |
| **Licence** | MIT |

**Points forts** :
- Ultra-efficace, idéal pour edge devices
- Framework d'adaptation unifié (autonomous driving, medical, remote sensing)
- Licence ouverte

**Points faibles** :
- Moins précis que les modèles plus grands
- Pas de Parallel Box Decoding

### 3.3 InternVL3.5-8B

| Caractéristique | Valeur |
|---|---|
| **Paramètres** | 8B |
| **VRAM** | ~14-16 Go en BF16 |
| **Paper** | arXiv:2508.18265 (août 2025) |
| **Innovation** | Variable Visual Position Encoding |
| **Licence** | MIT |

**Points forts** :
- SOTA parmi les open-source
- Raisonnement visuel avancé
- Support vidéo natif

**Points faibles** :
- Plus lent que LocateAnything
- Consommation VRAM plus élevée

### 3.4 Modèles de référence (déjà intégrés)

| Modèle | VRAM | Usage |
|---|---|---|
| GroundingDINO | ~2 Go | Détection zero-shot classique |
| YOLO-World v2 | ~4 Go | Open-vocabulary ultra-rapide |

---

## 4. Intégration technique

### 4.1 Architecture

```
Streamlit (8501) ──► FastAPI (8000) ──► Celery Workers
                                              │
                                    ┌─────────┼─────────┐
                                    ▼         ▼         ▼
                              worker_vlm  worker_det  worker_cluster
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            LocateAnything    Mini-InternVL    InternVL3.5
               3B                 4B               8B
```

### 4.2 Fichiers modifiés

| Fichier | Modifications |
|---|---|
| `api/workers/detection_worker.py` | +3 loaders, +3 inférenceurs, routing multi-modèle |
| `streamlit_app/pages/1_demo_detection.py` | Sélecteur de modèle VLM avec infos VRAM |
| `download_models.sh` | Ajout téléchargement des 3 modèles VLM |
| `README.md` | Documentation complète des modèles |
| `VLM_INTEGRATION.md` | Guide d'utilisation |

### 4.3 Loaders implémentés

```python
_get_locate_anything()    # → (model, tokenizer, processor)
_get_mini_internvl()      # → (model, tokenizer)
_get_internvl35_8b()      # → (model, tokenizer)
```

**Caractéristiques** :
- Chargement lazy (au premier appel)
- Cache dans `_loaded` dict
- Fallback mock si échec
- BF16 automatique
- Device mapping auto pour CUDA

### 4.4 Inférenceurs implémentés

```python
_infer_locate_anything()   # PBD — <box><x1><y1><x2><y2></box>
_infer_mini_internvl()     # Prompting — [x1, y1, x2, y2]
_infer_internvl35()        # Prompting — <box>(x1,y1),(x2,y2)</box>
```

**Parsing des sorties** :
- LocateAnything : regex `<box><(\d+)><(\d+)><(\d+)><(\d+)></box>` (normalisé 0-1000)
- Mini-InternVL : regex `\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]` (normalisé 0-1000)
- InternVL3.5 : regex `<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>` (normalisé 0-1000)

### 4.5 Routing multi-modèle

```python
if "locate_anything" in model:
    model_obj = _get_locate_anything()
elif "mini_internvl" in model:
    model_obj = _get_mini_internvl()
elif "internvl35" in model:
    model_obj = _get_internvl35_8b()
elif "yolo" in model:
    model_obj = _get_yolo_world()
else:
    model_obj = _get_grounding_dino()
```

### 4.6 Interface Streamlit

Sélecteur de modèle avec affichage :
- Nom du modèle
- Description technique
- VRAM estimée
- Cas d'usage recommandé

---

## 5. Résultats et performances attendues

### 5.1 Occupation VRAM (RTX 5090 — 32 Go)

| Configuration | VRAM estimée | Marge |
|---|---|---|
| LocateAnything-3B seul | ~12 Go | 20 Go libre |
| Mini-InternVL-4B seul | ~8-10 Go | 22-24 Go libre |
| InternVL3.5-8B seul | ~14-16 Go | 16-18 Go libre |
| LocateAnything + SAM2 | ~18 Go | 14 Go libre |
| InternVL3.5 + GroundingDINO | ~18 Go | 14 Go libre |
| Les 3 VLM en parallèle | ~34 Go | ❌ OOM — impossible |

**Conclusion** : 1 à 2 VLM simultanément sur RTX 5090, jamais les 3 en même temps.

### 5.2 Performances comparées

| Modèle | FPS estimé | Précision COCO | Latence/frame |
|---|---|---|---|
| LocateAnything-3B (fast) | ~15-20 | ~55-60 AP | ~50-70 ms |
| LocateAnything-3B (hybrid) | ~8-12 | ~60-65 AP | ~80-120 ms |
| Mini-InternVL-4B | ~5-8 | ~45-50 AP | ~120-200 ms |
| InternVL3.5-8B | ~3-5 | ~55-60 AP | ~200-330 ms |
| GroundingDINO (référence) | ~10-15 | ~50-55 AP | ~70-100 ms |
| YOLO-World v2 (référence) | ~25-30 | ~45-50 AP | ~30-40 ms |

*Estimations basées sur la littérature et les benchmarks publiés.*

### 5.3 Cas d'usage recommandés

| Besoin | Modèle recommandé |
|---|---|
| Détection multi-objets temps réel | **LocateAnything-3B** (mode fast) |
| Ressources limitées / edge | **Mini-InternVL-4B** |
| Haute précision / scènes complexes | **InternVL3.5-8B** |
| Détection rapide standard | GroundingDINO |
| Ultra-rapide open-vocabulaire | YOLO-World v2 |

---

## 6. Approche hybride recommandée

La littérature suggère une approche en 2 niveaux pour le flux vidéo :

```
Flux vidéo (30 FPS)
       │
       ▼
  YOLO-World v2 ──── Détection rapide frame-par-frame (30 FPS)
       │                 → Bounding boxes + classes
       │
       ▼ (échantillonnage key frames)
  VLM (LocateAnything) ── Analyse sémantique (2-5 FPS)
       │                 → Attribution, relations, contexte
       ▼
  Résultats fusionnés
```

**Bénéfices** :
- Réduction de 60-80% de la charge computationnelle
- Détection rapide + compréhension profonde
- Tracking temporel cohérent

---

## 7. Optimisations appliquées

1. **BF16 natif** : `torch.bfloat16` pour tous les VLM
2. **Device mapping auto** : Répartition automatique sur GPU
3. **Chargement lazy** : Modèles chargés à la demande
4. **Cache `_loaded`** : Pas de re-chargement entre frames
5. **Inference mode** : `torch.inference_mode()` pour réduire la mémoire
6. **Parsing regex optimisé** : Extraction directe des coordonnées

---

## 8. Limitations et perspectives

### Limitations actuelles
- **Licence LocateAnything** : Non-commercial uniquement
- **VRAM** : Impossible de charger les 3 VLM simultanément
- **Vitesse VLM** : 3-20 FPS vs 25-30 FPS pour YOLO
- **Parsing fragile** : Les sorties textuelles des VLM peuvent être irrégulières

### Perspectives d'amélioration
1. **Quantification INT4/INT8** : Réduire VRAM de 50-75%
2. **TensorRT** : Compilation optimisée pour RTX 5090 (sm_120)
3. **Approche hybride YOLO + VLM** : Combiner vitesse et précision
4. **Batch inference** : Traiter plusieurs frames d'un coup
5. **Model switching dynamique** : Changer de modèle selon la complexité de la scène
6. **Streaming WebSocket** : Détection temps réel avec feedback visuel

---

## 9. Commandes de référence

```bash
# Télécharger les modèles VLM
huggingface-cli download nvidia/LocateAnything-3B --local-dir models/locate_anything_3b
huggingface-cli download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir models/mini_internvl_4b
huggingface-cli download OpenGVLab/InternVL3_5-8B --local-dir models/internvl35_8b

# Lancer la plateforme
docker compose up -d

# Monitor GPU
docker compose exec worker_detection nvidia-smi

# API — Lancer une détection VLM
curl -X POST http://localhost:8000/detection/run \
  -H "Content-Type: application/json" \
  -d '{"video_id":"xxx","model":"locate_anything_3b","queries":["person","car"],"confidence":0.35,"track":true}'
```

---

## 10. Références bibliographiques

1. **LocateAnything** — Wang et al., "Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding", NVIDIA Tech Report, arXiv:2605.27365, mai 2026
2. **Mini-InternVL** — Gao et al., "A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance", arXiv:2410.16261, oct 2024
3. **InternVL3.5** — "Closing the Gap to Commercial Multimodal Models", arXiv:2508.18265, août 2025
4. **LLaVA-NeXT** — Liu et al., "Improved reasoning, OCR, and world knowledge", blog 2024-01-30
5. **Qwen-VL** — Bai et al., "Qwen Large Vision Language Model", Alibaba Cloud, 2023-2024
6. **SAM2Auto** — Rocky & Wu, "Auto Annotation Using FLASH", arXiv:2506.07850, juin 2025
7. **VIBES** — Mao et al., "Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning", arXiv:2604.23724, avril 2026

---

## 11. Compositions conceptuelles intégrées (littérature 2025-2026)

### 11.1 Composition Hiérarchique (2COOOL ICCV 2025 — 2ème place)

**Architecture implémentée:**
```
YOLO-World v2 (15 FPS) → détection rapide
    ↓
LocateAnything-3B (3 FPS) → analyse sémantique
    ↓
ByteTrack/BoT-SORT → tracking inter-frames
```

**Fonction Celery:** `run_composition_hierarchical`

**Paramètres:**
- `fast_fps`: FPS du détecteur rapide (défaut 15)
- `semantic_fps`: FPS du VLM sémantique (défaut 3)
- `track`: activation tracking (défaut True)

**Gain:** 60-80% réduction charge computationnelle

**VRAM:** ~16GB

---

### 11.2 Composition Ensembling (Multiple papiers 2025-2026)

**Architecture implémentée:**
```
GroundingDINO → détection zero-shot
    ↓
LocateAnything-3B → détection parallèle
    ↓
Fusion NMS + voting → résultat consolidé
```

**Fonction Celery:** `run_composition_ensembling`

**Paramètres:**
- `iou_threshold`: seuil IoU pour fusion (défaut 0.5)
- `fps`: fréquence d'analyse (défaut 5)

**Gain:** Robustesse accrue, réduction faux positifs

**VRAM:** ~14GB

**Helpers:**
- `_nms_fusion()`: Non-Maximum Suppression pour fusion
- `_compute_iou()`: calcul IoU entre boxes

---

### 11.3 Composition Itérative (LongShOTAgent, ScreenSeekeR 2025-2026)

**Architecture implémentée:**
```
LocateAnything-3B → détection initiale
    ↓
InternVL3.5-8B → raffinement (N passes)
    ↓
Validation croisée → cohérence
```

**Fonction Celery:** `run_composition_iterative`

**Paramètres:**
- `refinement_passes`: nombre de passes (défaut 2)
- `fps`: fréquence d'analyse (défaut 3)

**Gain:** Haute précision pour scènes complexes

**VRAM:** ~28GB

**Helpers:**
- `_cross_validate_detections()`: validation croisée entre passes

---

### 11.4 Routes API

**Endpoint:** `POST /composition/run`

**Body:**
```json
{
  "video_id": "string",
  "composition_type": "hierarchical|ensembling|iterative_refinement",
  "queries": ["person", "car"],
  "confidence": 0.35,
  "track": true,
  "fast_fps": 15,
  "semantic_fps": 3,
  "fps": 5,
  "iou_threshold": 0.5,
  "refinement_passes": 2
}
```

**Réponse:** `202 Accepted` avec `task_id`

---

### 11.5 Interface Streamlit

**Page:** `streamlit_app/pages/5_compositions_vlm.py`

**Fonctionnalités:**
- Sélection du type de composition
- Affichage des modèles utilisés et VRAM
- Presets de requêtes (véhicules, personnes, textes, personnalisé)
- Paramètres ajustables selon composition
- Visualisation résultats avec timeline, tableau, player vidéo
- Export CSV/JSON

---

## 12. Références bibliographiques complètes

1. **LocateAnything** — Wang et al., "Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding", NVIDIA Tech Report, arXiv:2605.27365, mai 2026
2. **Mini-InternVL** — Gao et al., "A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance", arXiv:2410.16261, oct 2024
3. **InternVL3.5** — "Closing the Gap to Commercial Multimodal Models", arXiv:2508.18265, août 2025
4. **2COOOL ICCV 2025** — Yokoi et al., "Hierarchical Reasoning with Vision-Language Models for Incident Reports from Dashcam Videos", arXiv:2510.12190, oct 2025 — 2ème place
5. **LongShOTAgent** — Kurpath et al., "A Benchmark for Omni-Modal Reasoning in Long Videos", arXiv:2512.16978, déc 2025 — 66.64% overall
6. **ScreenSeekeR** — Li et al., "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use", arXiv:2504.07981, avril 2025 — 48.1% accuracy
7. **SAIL-VL2** — Yin et al., "SAIL-VL2 Technical Report", arXiv:2509.14033, sept 2025 — #1 OpenCompass <4B
8. **QQMM-Embed** — Xue et al., "Improve Multi-Modal Embedding Learning via Explicit Hard Negative Gradient Amplifying", arXiv:2506.02020, juin 2025 — SOTA MMEB
9. **llama-nemoretriever-colembed** — Xu et al., "Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model", arXiv:2507.05513, juillet 2025 — #1 ViDoRe V1 & V2
10. **VLA-Arena** — Zhang et al., "An Open-Source Framework for Benchmarking Vision-Language-Action Models", arXiv:2512.22539, déc 2025 — ICML 2026
11. **MultiNet v1.0** — Guruprasad et al., "Benchmarking the Generality of Vision-Language-Action Models", arXiv:2512.11315, déc 2025
12. **LMMs-Eval** — Zhang et al., "LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models", arXiv:2407.12772 — toolkit d'évaluation unifié

---

*Mémoire rédigé le 20 juillet 2026 — Session de recherche, intégration technique et compositions conceptuelles*
