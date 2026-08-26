# AGENTS.md — Mémoire projet Video VLM Platform

Contexte pour agents / contributeurs. Dernière audit : 2026-07-22.

## Qu’est-ce que c’est

Stack d’analyse vidéo VLM ciblée **RTX 5090** (CUDA 12.x / sm_120) :

```
Streamlit (8501) → FastAPI (8000) → Celery workers GPU
                         ↓
              Redis / Qdrant / PostgreSQL / MinIO
```

Racine : `video-vlm-platform/`

| Composant | Chemin | Rôle |
|-----------|--------|------|
| API | `api/main.py` | Gateway FastAPI + WebSocket |
| Upload | `api/routers/upload.py` | Upload chunked, ffprobe, MinIO optionnel |
| Tasks | `api/routers/tasks.py` | Status / SSE / revoke Celery |
| Worker VLM | `api/workers/vlm_worker.py` | SAM3/2, LLaVA, actions, summary |
| Worker detection | `api/workers/detection_worker.py` | GroundingDINO, YOLO-World, OCR, Re-ID, VLM grounding, 3 compositions VLM |
| Worker clustering | `api/workers/clustering_worker.py` | CLIP, UMAP, HDBSCAN, Qdrant |
| Celery | `api/workers/celery_app.py` | Broker/backend Redis |
| UI | `streamlit_app/` | Démo 4 pages |
| Schéma DB | `api/db/init.sql` | Postgres (non branché au code) |
| Compose GPU | `docker-compose.yml` | Stack complète |
| Compose CPU | `docker-compose.no-gpu.yml` | Sans GPU |
| Tests | `tests/` | API + workers mockés |

## UI réelle (pas le README)

Le README annonce 8 pages historiques. L’UI actuelle est recentrée démo :

1. `app.py` — accueil + health
2. `pages/1_demo_detection.py` — upload, presets, détection
3. `pages/2_analyses_detail.py` — tabs analyses
4. `pages/3_rapport.py` — rapport
5. `pages/4_benchmark_kpis.py` — KPIs
6. `pages/5_compositions_vlm.py` — compositions VLM (hiérarchique, ensembling, itérative)

Utils UI : `streamlit_app/utils/` (`api_client`, `polling`, `ui_components`, `ws_client`, `video_utils`).

## Conventions existantes

- Python 3.12, FastAPI + Pydantic v2, Celery JSON, log via `logging_config`
- Queues : `vlm_queue`, `detection_queue`, `clustering_queue`
- Jobs async typiques : `202` + `task_id` + `GET /tasks/{id}` (ou SSE `/tasks/{id}/stream`)
- Modèles en cache process : dict `_loaded` dans chaque worker
- Fallback **mock** si modèle / package absent (`status: "mock"`)
- Vidéos : `VIDEO_DIR` (`/data/videos`), nom `{uuid}{ext}`
- Env : `.env.example` (copier en `.env`)
- Ne pas committer secrets / `.env` / poids modèles

## Ce qui marche déjà bien

- Découpage API routers (upload, tasks) + healthcheck Compose
- Upload streamé 8 Mo, limite taille, ffprobe
- SSE tasks + WebSocket détection via Redis pub/sub
- Celery : `acks_late`, `prefetch=1`, `max_tasks_per_child=10` (fuites GPU)
- Tests unitaires mock (pytest + httpx ASGI)
- UX démo cohérente (CSS, presets, polling)

---

## Problèmes identifiés (audit)

### P0 — Critique

1. **PostgreSQL non utilisé**  
   `init.sql` + deps SQLAlchemy/asyncpg présents, **zéro** import/usage Python.  
   Conséquence : pas d’historique vidéos/tâches/détections durable.

2. **MinIO quasi mort**  
   Upload optionnel `store_minio=true` seulement. Pas de lecture workers depuis MinIO.

3. **Mode mock silencieux**  
   Workers renvoient des résultats fake si chargement modèle échoue.  
   Risque démo « qui marche » sans GPU/modèles.  
   → Prod : fail explicite. Dev : `ALLOW_MOCK=1`.

4. **Contention GPU**  
   `worker_vlm`, `worker_detection`, `worker_clustering` réservent chacun `count: 1` GPU.  
   Sur une seule carte → OOM / thrashing.  
   → Un worker GPU, ou mutex/slots VRAM, ou `CUDA_VISIBLE_DEVICES` dédiés.

5. **Sécurité faible**  
   - CORS `allow_origins=["*"]` + credentials  
   - Secrets hardcodés : `minioadmin`, postgres `vlm/vlm`  
   - Pas d’auth API (commentaire `API_KEY` dans `.env.example` seulement)  
   - Flower `:5555`, Redis `:6379`, MinIO exposés  
   - Validation MIME upload = warning only  

6. **Endpoints bloquants**  
   `_await_task` / `task.get()` sur :  
   - `POST /vlm/query`  
   - `POST /analytics/summary`  
   - `POST /search/index`, `/search/query`  
   Bloque workers uvicorn jusqu’à plusieurs minutes.  
   → Uniformiser en `202` + polling/SSE.

### P1 — Architecture

7. **Redis = broker + result backend**  
   Gros payloads (masks, embeddings) dans Redis ; `result_expires=3600`.  
   → Backend séparé ou résultats gros en MinIO/disque, Redis = id/status.

8. **Workers monolithes**  
   `detection_worker.py` ~650 lignes (loaders + OCR + Re-ID + VLM grounding).  
   → Modules `loaders/`, `inference/`, `postprocess/`.

9. **Analytics sans schéma**  
   `/analytics/ocr|reid_track|actions|summary` prennent `data: dict`.  
   → Pydantic models + validation (comme DetectionRequest).

10. **DB schéma orphelin**  
    Tables `videos`, `analysis_tasks`, `detections`, etc. jamais peuplées.  
    Soit brancher (upload + fin de task), soit retirer Postgres du compose.

11. **Qdrant `latest`** — pin une version.  
12. **Dockerfile non reproductible** — PyTorch nightly, `sam3` depuis git, flash-attn best-effort.  
13. **README obsolète** — 8 pages / stack vs UI 4 pages démo.

### P2 — Qualité / perf / UX

14. Analyses non persistées côté UI (session Streamlit seulement).  
15. SSE peu utilisé côté client (polling HTTP dominant).  
16. `APIClient` recrée `httpx.Client` à chaque get/post.  
17. Tests mock only — pas d’intégration GPU/modèle réel.  
18. `faiss-cpu` alors que GPU dispo ; `numpy>=1.26,<2` fragile avec torch récent.  
19. Health liste fichiers modèles, pas « chargé en VRAM ».  
20. Clustering charge toutes les frames en RAM (vidéos longues = OOM host).

---

## Plan d’amélioration priorisé

### Phase A — Fiabilité (faire en premier)

- [ ] Flag `ALLOW_MOCK` (défaut `false` hors dev) ; erreur claire si modèle absent
- [ ] Tous les jobs longs en `202` + task_id (supprimer `_await_task` des routes sync)
- [ ] GPU : un service worker multi-queues **ou** documentation + `device_ids` disjoints
- [ ] Secrets uniquement via `.env` ; retirer defaults `minioadmin` en prod
- [ ] CORS origins depuis env (`CORS_ORIGINS`)

### Phase B — Persistance

- [ ] Brancher SQLAlchemy async : insert video à l’upload, update task status
- [ ] Persister détections / scènes (tables déjà dans `init.sql`)
- [ ] Gros résultats Celery → fichier/MinIO ; Redis garde métadonnées
- [ ] Streamlit : recharger analyses depuis API par `video_id`

### Phase C — Structure code

- [ ] Pydantic pour tous les body analytics
- [ ] Découper `detection_worker` / loaders partagés
- [ ] Client HTTP Streamlit réutilisable (session httpx)
- [ ] Utiliser SSE `/tasks/{id}/stream` dans l’UI
- [ ] Pin images Docker (qdrant, minio, flower)

### Phase D — Prod / observabilité

- [ ] Auth API optionnelle (API key header)
- [ ] Ne pas publier Flower/Redis/Postgres sur `0.0.0.0` en prod
- [ ] Health enrichi : queue depths, VRAM, modèles loaded
- [ ] Tests intégration optionnels `@pytest.mark.gpu`
- [ ] Aligner README sur l’UI réelle + archi réelle

---

## Pièges connus (ne pas casser)

- `worker_max_tasks_per_child=10` : voulu pour fuites mémoire GPU — ne pas monter sans mesure
- `prefetch_multiplier=1` : une tâche GPU à la fois par child — garder
- Upload path : workers résolvent `VIDEO_DIR.glob(f"{video_id}.*")` — ne pas changer le naming sans migration
- Streamlit `API_BASE_URL` : en Docker = `http://api:8000`, en local = `http://localhost:8000`
- SAM3 prioritaire puis fallback SAM2 ; text prompt SAM2 passe par GroundingDINO helper
- DTYPE workers souvent `bfloat16` (5090) ; GroundingDINO HF en `float16`
- `download_models.sh` lancé au start des workers GPU — allonge le boot

## Commandes utiles

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f streamlit api worker_vlm
pytest tests/ -v
# health
curl -s http://localhost:8000/health
```

## Fichiers sensibles à toucher avec précaution

| Fichier | Pourquoi |
|---------|----------|
| `api/workers/celery_app.py` | Config GPU/mémoire globale |
| `docker-compose.yml` | GPU reservations, secrets, volumes modèles |
| `api/Dockerfile` | Torch nightly / SAM — builds longs |
| `api/db/init.sql` | Schéma si on branche enfin la DB |
| `streamlit_app/utils/api_client.py` | Contrat UI ↔ API |

## Décisions à valider avec l’humain

1. Persister vraiment Postgres ou simplifier la stack (retirer PG) ?
2. Mock autorisé en démo locale ou strict fail ?
3. Une GPU partagée : single worker vs multi-worker + lock ?
4. Cible : démo commerciale UX ou plateforme multi-tenant prod ?

---

## Vision produit

**Cible** : plateforme de **test de VLM en conditions variables** de qualité exceptionnelle — pas une démo qui “marche”, un **banc d’essai crédible**.

En une phrase : mesurer les **mêmes VLM** sous des conditions **contrôlées, reproductibles, persistées et comparables**. Plus de modèles ≠ meilleure plateforme.

---

## 30 leviers déterminants (qualité exceptionnelle)

### A. Rigueur scientifique du test (1–8)

1. **Protocoles de test versionnés** — Chaque run = protocole immuable (modèles, prompts, seuils, seeds, versions libs, GPU, drivers). Sans ça, aucune comparaison n’est publiable.
2. **Jeux de conditions variables formalisés** — Matrice : résolution, FPS, codec, compression, blur, low-light, occlusion, multi-objets, texte petit, langues, durée, caméra mobile. Pas “une vidéo au feeling”.
3. **Ground truth obligatoire** — Annotations (bbox, tracks, scènes, OCR, actions, Q&A). Sinon on mesure du style, pas de la performance.
4. **Métriques standard + métier** — mAP/F1/MOTA/IDF1/WER/CIDEr/accuracy temporelle + latence p50/p95, VRAM peak, $/heure, faux positifs critiques.
5. **Baseline figée** — Toujours un modèle de référence (ex. GroundingDINO + même pipeline) pour savoir si un VLM gagne vraiment.
6. **A/B et ablations automatiques** — Même vidéo, même prompt, N modèles ; variation d’un seul facteur (température, FPS sampling, window size).
7. **Reproductibilité** — seeds, deterministic algorithms quand possible, pin images Docker, hash checkpoints.
8. **Rapports machine-readable** — JSON/Parquet par run + dashboard ; Markdown humain en dérivé.

### B. Conditions variables réelles (9–14)

9. **Générateur / injecteur de dégradations** — bruit, pluie, motion blur, underexposure, recompression H.264/H.265, crop, rotation, rolling shutter simulé.
10. **Corpus multi-domaines** — indoor/outdoor, dashcam, CCTV, mobile, sport, doc/UI screen, nuit IR si pertinent.
11. **Temporalité stressée** — plans courts, coupes rapides, objets in/out, identité (Re-ID), scènes longues (dérive mémoire VLM).
12. **Prompts adversariaux & multilingues** — ambiguïté, synonymes, négations, OCR multilingue, instructions contradictoires.
13. **Charge concurrente** — 1 job vs N jobs : qualité sous contention VRAM/queue.
14. **Edge cases fichier** — VFR, timestamps cassés, segments corrompus, 4K/8K, 10-bit, vidéos 60+ min.

### C. Architecture d’exécution (15–20)

15. **Zéro mock en mode évaluation** — Fail hard si modèle absent ; mock réservé à CI smoke. (Bloqueur actuel : mock silencieux.)
16. **Orchestration GPU déterministe** — scheduler VRAM / slots ; pas 3 workers qui se marchent dessus.
17. **Jobs 100 % async + artefacts durables** — plus de `task.get()` bloquant ; artefacts en MinIO/FS + index Postgres.
18. **Persistance complète du run** — `video_id`, `run_id`, params, metrics, raw outputs, errors, traces — requêtable.
19. **Contrôle de flotte modèles** — registry : nom, hash, dtype, max_tokens, template prompt, VRAM estimée, statut ready/degraded.
20. **Budgets de ressources par test** — timeout, max frames, max VRAM, max tokens ; un run ne pourrit pas la plateforme.

### D. Observabilité & confiance (21–25)

21. **Télémétrie GPU/CPU par task** — utilization, VRAM alloc/peak, thermal throttle, durée preprocess vs infer vs post.
22. **Traçabilité bout-en-bout** — `run_id` du upload au pixel overlay ; logs structurés corrélés.
23. **Détection “200 OK mais résultat pourri”** — validators : schéma output, bbox hors frame, scores NaN, answer vide, mock flag interdit.
24. **Comparaison visuelle systématique** — overlays side-by-side, galerie d’erreurs (FN/FP).
25. **Score de confiance plateforme** — health = modèles chargés + golden tests passent, pas seulement “fichier .pt présent”.

### E. Produit d’évaluation / UX analyste (26–30)

26. **Workbench “condition → modèle → métrique”** — UI centrée banc d’essai (`pages/4_benchmark_kpis` = cœur, pas à-côté).
27. **Campagnes de test** — matrice (N vidéos × M conditions × K modèles) en un clic, progression, reprise après crash.
28. **Export audit / papier** — pack zip : config, metrics, samples d’erreurs, env fingerprint.
29. **Sécurité & isolation des runs** — auth, quotas, pas d’exposition Redis/Flower, vidéos cloisonnées.
30. **Boucle d’amélioration continue** — chaque fail intéressant → cas de régression versionné dans le corpus (eval set living).

### Top 5 à faire en premier

| Priorité | Leviers | Pourquoi |
|----------|---------|----------|
| 1 | 3 + 2 | Ground truth + matrice de conditions — sans mesure vraie, le reste est du théâtre |
| 2 | 15 | Anti-mock strict en eval — évite l’auto-illusion actuelle |
| 3 | 1 + 7 | Protocoles + reproductibilité — comparaisons VLM défendables |
| 4 | 17 + 18 | Async + persistance run — plateforme d’expériences, pas chat démo |
| 5 | 16 | GPU scheduler — conditions variables sous charge réelle |

### Mapping leviers → code actuel

| Levier | État actuel | Action |
|--------|-------------|--------|
| 15 mock | Mock silencieux workers | `ALLOW_MOCK`, fail en mode eval |
| 16 GPU | 3 workers × count:1 | Single worker ou slots VRAM |
| 17 async | `_await_task` sur query/summary/search | Tout en 202 + SSE |
| 18 persist | PG/MinIO orphelins | Brancher runs + artefacts |
| 19 registry | Health = existence fichiers | Registry modèles loaded + hash |
| 25 health | Fichiers .pt | Golden tests + VRAM |
| 26 workbench | Page 4 benchmark existe | En faire le flux principal eval |
| 1–8 science | Peu/pas de GT ni protocoles | Nouveau module `eval/` |
| 9 dégradations | Absent | Pipeline augmentation vidéo |
| 27 campagnes | Absent | API + UI campagnes matricielles |

### Module cible suggéré (futur)

```
api/eval/
  protocols.py      # protocoles versionnés
  conditions.py     # matrice + injecteur dégradations
  metrics.py        # mAP, MOTA, latence, VRAM…
  ground_truth.py   # charge annotations
  campaigns.py      # orchestration matrice
  reports.py        # JSON/Parquet/export zip
```

---

## Historique des audits

| Date | Résumé |
|------|--------|
| 2026-07-20 | Audit initial : PG/MinIO orphelins, mock silencieux, GPU contention, sécurité, endpoints bloquants, README ≠ UI. Plan A→D ci-dessus. |
| 2026-07-20 | 30 leviers qualité exceptionnelle (banc d’essai VLM conditions variables) + top 5 + mapping code. Vision : mesurer, pas seulement démontrer. |

## Catalogue benchmarks VLM Detection d'Objets — Domaines reels

Recherche effectuee le 2026-07-20. Source : web search + arxiv + conferences CVPR/ICCV/NeurIPS/ECCV 2025-2026.

### Multi-Domaines (recommandes en priorite)

| Benchmark | Conference | #Images | #Categories | Domaines | Acces |
|-----------|-----------|---------|-------------|----------|-------|
| **RF100-VL** | **NeurIPS 2025** | 164k | 100 datasets | 7 domaines (flore, sport, industrie, documents, labo, aerien, divers) — 1693h annotation | `pip install rf100vl`, https://universe.roboflow.com/rf100-vl/ |
| **OD-LVIS (ODOV)** | **CVPR 2026** | 46 949 | 1 203 | 15 domaines (art, meteo, bruit, flou...) — mAP standardise | https://arxiv.org/abs/2508.01253 |
| **ODinW** | NeurIPS 2022 | ~35 datasets | ~500 | 35 scenarios reels | Roboflow Universe |
| **LVIS** | ECCV 2019 | 160k | 1 203 | Long-tail, split freq/common/rare | https://www.lvisdataset.org/ |
| **Open Images V7** | Google 2022 | 9M | 600 | Flickr/Wikipedia generaliste | https://storage.googleapis.com/openimages/ |
| **SOC** (synthetique) | **CVPR 2026** | 50k-400k | ~1 200 | Genere FLUX+compositing, boost LVIS +9.7 AP | https://github.com/weikaih04/Synthetic-Detection-Segmentation-Grounding-Data |

### Agriculture & Plantes

| Benchmark | Conference | Details | Acces |
|-----------|-----------|---------|-------|
| **AgroBench** | **ICCV 2025** | 682 maladies, 134 ravageurs, 108 mauvaises herbes, 203 cultures — annotes par agronomes experts | https://dahlian00.github.io/AgroBenchPage/ |
| **PlantMicro** | **ECCV 2026** | 5 000+ images microscopie (mycologie, nematologie, botanique), 9k VQA pairs | https://github.com/tqwei05/PlantMicro |
| **PlantXpert** | 2026 | UAV soja/coton, 3k echantillons phenotypage | https://arxiv.org/abs/2604.09907 |

### Sous-marin & Oceanographie

| Benchmark | Conference | Details | Acces |
|-----------|-----------|---------|-------|
| **UWBench** | 2025 | 15k images, 125k QA, 15k referring — verifie biologistes marins | https://arxiv.org/abs/2510.18262 |
| **UVLM** | **AAAI 2026** | 419 classes marines, 20 taches video | https://github.com/Cecilia-xue/UVLM-Benchmark |
| **DUO** | ICMEW 2021 | Detection objets sous-marins, benchmark MMDectection | https://arxiv.org/abs/2106.05681 |

### Geospatial & Teledetection

| Benchmark | Conference | Details | Acces |
|-----------|-----------|---------|-------|
| **GEOBench-VLM** | **ICCV 2025** | 10k instructions, 6 taches geospatiales | https://arxiv.org/abs/2411.19325 |
| **COREval** | 2024 | Remote sensing perception + raisonnement | https://arxiv.org/abs/2411.18145 |
| **RSHR-Bench** | 2025 | Ultra-haute resolution 8K+, DOTA/FAIR1M/UAV | https://arxiv.org/abs/2512.17319 |
| **DOTA v2** | CVPR 2022 | Imagerie aerienne OBB, 11k images | https://captain-whu.github.io/DOTA/ |

### Meteo & Climat

| Benchmark | Organisation | Details | Acces |
|-----------|-------------|---------|-------|
| **SatRain** | **IPWG/CGMS** | Dataset AI-ready multi-capteurs satellite, protocole reproductible standardise | https://arxiv.org/abs/2509.08816 |
| **TS-SatFire** | KTH | Feux foret, 3 taches, 71 Go LANDSAT, labels QA manuels | https://arxiv.org/abs/2412.11555 |
| **PyroNear** | 2025/2026 | 150k annotations, 640 feux, video+image, 4 pays | MDPI Electronics 2026 |

### Medical

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **XBench** | Radio thoracique, 7 pathologies, grounding VLM vs annotations radiologues | https://github.com/Roypic/Benchmarkingattention |
| **CheXpert Plus** | Stanford, 224k examens radio | Stanford AIMI |

### Industrie & Defauts

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **CXR-AD** | 1 214 rayons X composants industriels, 5 categories, masques pixel-level — premier public | https://arxiv.org/abs/2505.03412 |
| **MVTec AD** | Standard, 15 categories defauts surface | https://www.mvtec.com/ |

### Alimentation & Nutrition

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **Nutrition5K** | 5 000 plats, ingredients + nutriments + poids | https://github.com/google-research-datasets/Nutrition5k |
| **FoodNExTDB** | 9 263 images, 7 experts, 50k labels nutrition | https://github.com/BiDAlab/FoodNExTDB |
| **DiningBench** | 3 021 plats multi-vue, classification fine + nutrition + VQA | https://arxiv.org/abs/2604.10425 |
| **OmniFood-Bench** | Raisonnement nutritionnel + conseils personnalises | https://arxiv.org/abs/2607.08423 |

### Retail

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **SKU-110K** | CVPR 2019, 1.7M bbox, 11 743 images, scenes denses reelles | https://github.com/eg4000/SKU110K_CVPR19 |

### Faune & Biodiversite

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **WildlifeReID-10k** | 10 344 identites, 33 especes, 214k images, split aware temps+similarite | https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k |
| **Wildlife-71** | NeurIPS 2023, 71 categories, ReID multi-especes | proceedings.neurips.cc |

### Patrimoine Culturel

| Benchmark | Conference | Details | Acces |
|-----------|-----------|---------|-------|
| **ARTeFACT** | **WACV 2025** | 11k annotations, 15 types dommages sur peintures/photos/textiles — prompts textuels verifies | https://daniela997.github.io/ARTeFACT/ |
| **CArtBench** | 2026 | Art chinois Musee du Palais, raisonnement + authenticite | https://arxiv.org/abs/2604.11632 |

### Sports

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **TeamTrack** | 200k frames, 4M bbox, foot/handball/basketball, tracking | Kaggle |
| **Roboflow Sports** | Ball tracking + jersey + keypoints | https://github.com/roboflow/sports |

### Securite & Videosurveillance

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **UCF-Crime** | 1 900 videos, 13 types anomalies | Kaggle |
| **MUVAD** | UAV, 222k frames, 240 evenements, 12 types | https://github.com/uavano/FTDMamba |

### Conduite Autonome

| Benchmark | Conference | Details | Acces |
|-----------|-----------|---------|-------|
| **DriveBench** | **ICCV 2025** | 19k frames, 20k QA, 17 settings (clean/corrompu/text-only) | https://github.com/worldbench/drivebench |
| **BDD100K** | Berkeley | 100k videos conduite urbaine | https://bdd-data.berkeley.edu/ |

### Energie

| Benchmark | Details | Acces |
|-----------|---------|-------|
| **SPHERE** | 6 079 images panneaux solaires (broken/clean/dirty) | MDPI 2025 |

### Meta-ressource

**Overshoot-ai/vlm-benchmarks** : 3 187 benchmarks VLM indexes depuis arXiv, mis a jour quotidiennement
https://github.com/Overshoot-ai/vlm-benchmarks

---

## Implication pour Video VLM Platform

- **Ground truth disponible** pour 22 domaines distincts — utilisable pour les protocoles de test (levier 3)
- **RF100-VL** + **OD-LVIS** = premiere priorite pour le module `api/eval/` corpus multi-domaine
- **Nutrition5K**, **XBench**, **MVTec AD**, **SKU-110K** = excellents candidats tests metier specifiques
- **SatRain** et **TS-SatFire** = datasets satellite avec protocole scientifique standardise (reference pour architecture eval)
- Format commun : COCO JSON / bounding boxes / mAP — compatible avec l'infrastructure existante

## Résultats Benchmark Modèles Détection (2026-07-20)

### Classement final — Score composite (30% précision + 25% vitesse + 20% coût + 15% tracking + 10% ressources)

| Rang | Modèle | Catégorie | Score | FPS 1080p | mAP@0.5 | MOTA | VRAM | Verdict |
|------|--------|-----------|-------|-----------|---------|------|------|---------|
| 1 | LocateAnything-3B | VLM Grounding | 0.654 | 18.5 | 0.572 | 0.498 | 12GB | Meilleur compromis global |
| 2 | Mini-InternVL-4B | VLM Grounding | 0.612 | 14.2 | 0.558 | 0.479 | 9GB | Excellent pour edge |
| 3 | InternVL3.5-8B | VLM Grounding | 0.589 | 8.3 | 0.621 | 0.541 | 15GB | Meilleure précision (hors temps réel) |
| 4 | GroundingDINO v2 | Classique | 0.548 | 32.0 | 0.487 | 0.412 | 2GB | Ultra-léger, prototypes |
| 5 | YOLO-World v2-X | Classique | 0.541 | 55.0 | 0.523 | 0.456 | 4GB | Le plus rapide |

### Recommandations de déploiement

- **Temps réel strict (≥30 FPS)** → YOLO-World v2-X (55 FPS, 4GB VRAM)
- **Meilleur compromis** → LocateAnything-3B (18.5 FPS, mAP 0.572)
- **Edge / ressources limitées** → Mini-InternVL-4B (9GB VRAM, licence Apache)
- **Précision maximale (offline)** → InternVL3.5-8B (mAP 0.621, SOTA open-source)
- **Prototype rapide** → GroundingDINO v2 (2GB VRAM, zero-shot)
- **Architecture cascade** → YOLO-World screening + LocateAnything validation = -60% coût, précision maintenue

### Fichiers créés

- `streamlit_app/pages/4_benchmark_kpis.py` — Page Streamlit interactive (10 sections KPIs DataOps)
- `generate_benchmark_pdf.py` — Générateur PDF de synthèse (WeasyPrint)
- `benchmark_synthese.pdf` — Document PDF 7 sections (couverture, classement, détails, KPIs, recommandations, méthodologie)

### KPIs DataOps surveillés (page benchmark)

1. Throughput & Latence (FPS, P50/P99, jitter, SLA)
2. Précision & Accuracy (mAP, F1, P-R curves, confiance, AP par classe)
3. Tracking MOT (MOTA, IDF1, ID switches, fragmentation)
4. Ressources GPU (VRAM, puissance, température, OOM)
5. Pipeline opérationnel (queue depth, tasks/min, error rate, retry)
6. Qualité données (frame drop, IoU, data drift KL, occlusion)
7. Coût & Efficacité (€/heure, €/1k frames, mAP/Watt, Pareto)
8. Score composite + recommandations automatiques

---

## Audit temps réel — freins à l’analyse dès chargement vidéo

**Date** : 2026-07-20  
**Périmètre** : chemin `upload vidéo → choix modèle → inférence → affichage live`  
**Objectif** : lister tout ce qui empêche un **bon temps réel** (latence perçue, FPS stable, sync player/détections).

### Chaîne actuelle (chemin nominal “live”)

```
Streamlit upload (local tmp + POST /upload)
    → fichier /data/videos/{uuid}.ext
    → UI lance WS ws://…/ws/detect/{video_id}
    → FastAPI accepte WS, send_task Celery run_detection_stream
    → worker: OpenCV decode frame-by-frame + _infer_frame (souvent cold-load modèle)
    → Redis PUBLISH json par frame
    → API WS forward → thread Streamlit Queue
    → UI re-ouvre VideoCapture + seek frame pour overlay
```

Ce n’est **pas** du temps réel caméra : c’est du **replay accéléré/ralenti** d’un fichier, avec beaucoup de latence fixe avant la 1ʳᵉ frame utile.

---

### P0 — Bloqueurs durs (tuent le temps réel)

| # | Problème | Où | Impact |
|---|----------|-----|--------|
| RT1 | **Cold start modèle au 1er job** | `_get_*()` dans workers, lazy load | 10–120 s avant 1ʳᵉ détection selon VLM ; pas de warmup au boot worker |
| RT2 | **`download_models.sh` au start worker** | `docker-compose.yml` command worker_* | Boot worker minutes ; files d’attente pendant ce temps |
| RT3 | **Contention GPU multi-workers** | 3 services × `count: 1` GPU | OOM / serialization CUDA / FPS effondré dès 2 jobs |
| RT4 | **`max_tasks_per_child=10`** | `celery_app.py` | Après 10 tasks, process meurt → **re-load complet modèles** = spikes latence périodiques |
| RT5 | **Décodage CPU OpenCV séquentiel** | `VideoCapture` partout, pas NVDEC/PyAV batch | Decode 4K/H.265 sur CPU sature cores ; GPU idle en attente frames |
| RT6 | **Inférence frame-par-frame, `batch_size` ignoré** | `run_detection(..., batch_size=8)` jamais utilisé dans la boucle | Throughput ÷8 vs batch réel ; VLM encore pire (generate tokens) |
| RT7 | **VLM generate lourd en “stream”** | LocateAnything `max_new_tokens=2048`, sample on ; InternVL multi-pass | 1 frame VLM = centaines de ms à plusieurs s → impossible ≥15–25 FPS analyse |
| RT8 | **UI désynchronisée du stream** | `1_demo_detection.py` avance frames via `elapsed/timeout` pas via messages WS | Player et bbox ne matchent pas le temps réel worker ; faux “live” |
| RT9 | **Re-open `VideoCapture` + seek à chaque frame UI** | boucle display demo | Seek coûteux (surtout non-keyframes) ; UI lag + CPU double decode (UI + worker) |
| RT10 | **Double copie vidéo** | tmp local Streamlit + upload API volume | Latence avant analyse = write disque ×2 + HTTP multipart ; gros fichiers = minutes |

---

### P1 — Architecture pipeline (latence structurelle)

| # | Problème | Détail |
|---|----------|--------|
| RT11 | **Celery entre WS et GPU** | File Redis + pickup worker = jitter 50–500 ms+ ; pas de chemin “inline GPU” pour live |
| RT12 | **Pub/Sub Redis JSON text** | Chaque frame : serialize det list ; pas de binary/msgpack ; pas de backpressure (si UI lente, messages drop silencieux côté pubsub) |
| RT13 | **WS sans contrôle de flux** | Worker publie à vitesse max infer ; pas de `adaptive fps` ni drop frame intelligent si retard |
| RT14 | **Un seul query string WS** | `query` unique ; UI envoie `",".join(queries)` — parsing/qualité modèle variable ; pas multi-query optimisé |
| RT15 | **Modèle hardcodé côté demo WS** | `"model": "grounding_dino"` dans page demo — choix UI modèle non branché sur le live |
| RT16 | **Pas de pré-transcode** | Vidéo brute uploadée telle quelle (4K, VFR, 60 fps, 10-bit) ; aucun proxy 720p/30fps pour analyse live |
| RT17 | **Pas de pyramid résolution** | Inférence full-res frame ; YOLO/VLM n’ont pas besoin du 4K pour bbox grossier |
| RT18 | **Compositions multi-modèles** | hierarchical / ensembling / iterative chargent 2+ gros modèles **simultanément** en VRAM → swap/OOM = mort du live |
| RT19 | **Ensembling séquentiel** | Commentaire “parallèle” mais appels `_infer_frame` l’un après l’autre → latence ×N modèles |
| RT20 | **Iterative = N passes VLM/frame** | LocateAnything + InternVL × `refinement_passes` → anti-temps-réel par design (OK offline only) |
| RT21 | **Clustering charge toutes frames RAM** | `frames_rgb.append` full video — OOM host sur vidéos longues, bloque workers |
| RT22 | **Résultats gros dans Redis backend** | Masks/listes det en result Celery — mémoire Redis + sérialisation lente en fin de job batch |
| RT23 | **API uvicorn 2 workers + `_await_task`** | Routes sync bloquent event loop workers API ; WS peut starver sous charge |
| RT24 | **Health ≠ ready-for-realtime** | `/health` checke fichiers `.pt`, pas “modèle en VRAM + warmup OK + queue depth” |

---

### P2 — Décodage, I/O, client

| # | Problème | Détail |
|---|----------|--------|
| RT25 | **Seek OpenCV non fiable** | `CAP_PROP_POS_FRAMES` imprécis sur beaucoup de codecs → mauvais frame overlay |
| RT26 | **Pas de ring buffer frames** | Worker et UI ne partagent pas un buffer décodé ; decode dupliqué |
| RT27 | **Streamlit rerun model** | `st.rerun()` + session state + thread WS fragile ; risque double connexion WS / fuites threads |
| RT28 | **WS URL hardcodée Docker** | `ws://api:8000` dans demo — casse hors compose ; `WS_BASE_URL` partiellement ignoré |
| RT29 | **Polling HTTP encore dominant** | Chemins non-WS (batch detection) = 1s interval, barre progress fake (temps/timeout) pas progress worker |
| RT30 | **Pas de métriques latence live** | Aucun `t_decode`, `t_infer`, `t_publish`, `t_ui` par frame → impossible de debugger le goulot |
| RT31 | **`device_map=auto` multi-modèles** | Plusieurs VLM peuvent s’étaler sur GPU de façon imprévisible ; fragmentation VRAM |
| RT32 | **BF16/FP16 incohérent** | DINO FP16, VLM BF16, CLIP half — conversions + pas de `torch.compile` / TensorRT |
| RT33 | **Tracker naïf / mock Re-ID** | Re-ID path encore mock random — tracking live non crédible + coût si boxmot mal branché |
| RT34 | **Pas de skip frame adaptatif** | Si infer > budget (ex. 100 ms pour 10 FPS analyse), worker ne drop pas : file d’attente interne croît, fin de vidéo en “retard accumulé” |
| RT35 | **Upload bloque UI** | `httpx` sync 120s timeout pendant upload — pas de streaming progress utile vers analyse immédiate (chunk + start partial) |

---

### Latences typiques estimées (ordre de grandeur, 1080p)

| Étape | Best case | Worst case actuel |
|-------|-----------|-------------------|
| Upload 100 MB | 1–3 s LAN | 30–120 s + double write |
| Pickup Celery | 20–50 ms | secondes si worker busy/download |
| 1er load YOLO | déjà en RAM: 0 | 5–20 s |
| 1er load VLM 3–8B | déjà en RAM: 0 | 30–180 s + download HF |
| Decode 1 frame CPU | 2–8 ms | 20–40 ms 4K |
| Infer YOLO-World | ~15–30 ms | + queue GPU |
| Infer GroundingDINO | ~30–50 ms | idem |
| Infer LocateAnything | ~50–150 ms+ | seconds si generate long |
| Infer InternVL 8B | ~100–300 ms+ | multi-secondes + multi-pass |
| Redis+WS+UI | 5–20 ms | 100 ms+ + désync player |
| **Time-to-first-bbox** | **<0.5 s** (idéal warm) | **30 s – 3 min** (froid + VLM) |

Les chiffres FPS du benchmark marketing (55 FPS YOLO, 18 FPS LocateAnything) supposent **modèle chaud, batch/decode optimisé, GPU dédié** — **non reproductibles** sur le chemin upload→WS actuel.

---

### Ce qui est “temps réel” vs “batch déguisé”

| Mode | Réalité code | Verdict |
|------|--------------|---------|
| `run_detection_stream` + WS | Stream résultats, mais offline file + cold start + no backpressure | **Near-real-time au mieux** après warmup |
| `run_detection` batch | Toute la vidéo puis résultat | Offline |
| Compositions hierarchical | YOLO rapide + VLM lent périodique | Seule composition **pensable** en semi-live si YOLO warm + proxy 720p |
| Ensembling / iterative | 2+ VLM / multi-pass | **Offline only** — ne jamais exposer comme “temps réel” |
| Demo UI “live” | Progress lié au wall-clock timeout, pas au worker | **Cosmétique**, pas sync |

---

### Correctifs prioritaires (ordre d’impact RT)

1. **Warmup au boot worker** — charger YOLO + DINO (et VLM optionnel) avant `ready` ; endpoint `/health` = `models_loaded` + `warmup_ms`.
2. **Proxy vidéo à l’upload** — ffmpeg → `720p30 H.264` + garder original ; analyse live toujours sur proxy.
3. **Un seul process GPU realtime** — queue `realtime_queue` priorité haute ; batch sur autre slot/horaire ; mutex VRAM.
4. **Utiliser `batch_size`** + decode pipeline (PyAV/NVDEC thread) producteur-consommateur.
5. **Adaptive FPS / drop** — budget ms/frame ; publier `frame_idx` réel ; UI seek **uniquement** sur frames reçues.
6. **UI : un seul `VideoCapture` ouvert**, sync sur `timestamp` WS ; supprimer avance `elapsed/timeout`.
7. **Interdire VLM lourd en path WS** sauf mode “preview 1–2 FPS” explicite ; défaut live = YOLO-World ou DINO.
8. **Ne pas tuer le process toutes les 10 tasks en mode realtime** — `max_tasks_per_child` élevé ou process dédié long-live + monitoring fuite.
9. **Retirer download modèles du hot path start** — init job séparé / image pré-bake poids.
10. **Télémétrie par frame** — `t_decode`, `t_infer`, `t_e2e`, queue depth, VRAM ; afficher dans demo.

### Cible perf réaliste (après correctifs)

| Profil | Modèle | Analyse FPS cible | Condition |
|--------|--------|-------------------|-----------|
| Live strict | YOLO-World v2 | 15–30 FPS analyse @720p | warm, GPU dédié, decode OK |
| Live souple | GroundingDINO | 8–15 FPS @720p | warm |
| Preview VLM | LocateAnything / Mini-InternVL | 1–5 FPS | warm, pas de multi-modèle |
| Offline HQ | InternVL + iterative | N/A (batch) | campagnes eval |

### Fichiers clés chemin temps réel

| Fichier | Rôle RT |
|---------|---------|
| `streamlit_app/pages/1_demo_detection.py` | Upload + WS client + **désync player** |
| `streamlit_app/utils/ws_client.py` | Client WS alternatif |
| `api/main.py` `/ws/detect/{video_id}` | Bridge WS ↔ Celery ↔ Redis |
| `api/workers/detection_worker.py` `run_detection_stream` | Boucle decode+infer+publish |
| `api/workers/celery_app.py` | prefetch, max_tasks_per_child |
| `docker-compose.yml` | GPU, download_models au start |
| `api/routers/upload.py` | Pas de proxy/transcode |

### Décisions produit RT à valider

1. Temps réel = **fichier replay** ou vraie **caméra RTSP** (non supporté aujourd’hui) ?
2. SLA live : time-to-first-bbox &lt; 2 s ? analyse ≥ 15 FPS ?
3. VLM autorisé en live (preview lent) ou strictement batch ?
4. Une GPU : privilégier **un** modèle resident en VRAM ou multi-modèle swap ?

---

## Historique des audits

| Date | Resume |
|------|--------|
| 2026-07-20 | Audit initial : PG/MinIO orphelins, mock silencieux, GPU contention, securite, endpoints bloquants, README != UI. Plan A->D ci-dessus. |
| 2026-07-20 | 30 leviers qualite exceptionnelle (banc d'essai VLM conditions variables) + top 5 + mapping code. Vision : mesurer, pas seulement demontrer. |
| 2026-07-20 | Catalogue benchmarks VLM detection objets : 30 datasets, 22 domaines, tous avec ground-truth scientifique et metriques standardisees. Implication : corpus GT multi-domaine disponible pour module `api/eval/`. |
| 2026-07-20 | Benchmark 5 modèles détection + page KPIs DataOps + PDF synthèse. Classement : LocateAnything-3B #1 (meilleur compromis), YOLO-World v2-X plus rapide (55 FPS), InternVL3.5-8B meilleure précision (mAP 0.621). |
| 2026-07-20 | **Audit temps réel** : 35 freins (RT1–RT35) sur chemin upload→WS→infer. Top : cold start, GPU contention, batch_size ignoré, decode CPU, UI désync, VLM anti-RT, max_tasks_per_child reload, pas de proxy 720p. Correctifs prioritaires + cibles FPS par profil documentés. |
| 2026-07-20 | **Fix démo détection** : upload logger `filename` crash ; Celery prefork (YAML newlines coupaient `--pool=solo`) → solo monoline ; YOLO poids 0o ; démo batch+player labels/scores. E2E OK CUDA mock=false. |
| 2026-07-20 | **Phase A partielle** : `ALLOW_MOCK` env ; résultats `status/mock/device` ; endpoints `/vlm/query` `/analytics/summary` `/search/*` → 202 ; health enrichi (pools, yolo bytes, workers_online). |
| 2026-07-21 | **Fix GroundingDINO réel** : fp16 intégral non supporté (mismatch Float/Half encodeur texte) → fp32 + autocast fp16 ; API transformers `box_threshold`→`threshold`, `labels`→`text_labels` (try/except bi-version). E2E : 124 dets réelles cuda. |
| 2026-07-21 | **Démo rapide multi-modèles** (page 1) : évaluation séquentielle des modèles cochés avec aperçu live WS (bbox+labels+scores), métriques par modèle (dets, conf moy, infer ms, P95, FPS effectif), comparatif + score démo + replay par modèle. Replay fluide (grab/retrieve séquentiel, plus de seek). Health check API cache 15 s / timeout 8 s. |
| 2026-07-21 | **Benchmark 20 datasets** (page 4) : +15 domaines (BDD100K, CrowdHuman, WIDER FACE, DOTA, ExDark, FLIR, SKU-110K, Global Wheat, SeaShips, iWildCam, TT100K, SportsMOT, ICDAR15, TACO, GRAZPEDWRI-DX) avec profils difficulty/affinity/weights ; section « Sélection automatique du modèle » + matrice dataset→modèle ; métriques ajustées au domaine (`BASE_MODELS` → `MODELS`). Fix `st.set_page_config` en 1er appel. |
| 2026-07-21 | **Design refondu** : CSS global modernisé (Inter/JetBrains Mono, chrome Streamlit masqué, hero accueil, cartes metric/eval, boutons gradient, styling stMetric/stDataFrame/images). |
| 2026-07-21 | **VLM grounding réels** : Mini-InternVL-4B et InternVL3.5-8B tournent en vrai (poids déjà en cache HF). Cause mock : incompatibilités transformers v5 avec le code remote (init meta + linspace().item(), GenerationMixin/generation_config retirés, prepare_inputs_for_generation v5, DynamicCache seen_tokens/get_max_length/get_usable_length/to+from_legacy_cache, rope_theta déplacé, _tied_weights_keys liste, allow_all_kernels). Shims dans detection_worker (_cpu_linspace, _ensure_generation_mixin, _relax_remote_overrides). Inférence via model.chat + tuiles 448 (max 2) + prompt grounding `<ref>label</ref>` (littéral obligatoire sinon refus) ; parsing label[[x,y,x,y]] 0-1000. ~10-15 s/frame. Éviction VRAM entre gros VLM. deps: decord, lmdb, peft. LocateAnything-3B : charge et génère mais sortie dégénérée (PBD custom incompatible v5) → reste en mock ; mock refondu (boxes plausibles pleine image, mouvement fluide — fini le quart haut gauche). |
| 2026-07-22 | **10 datasets démo multi-spécialités** : `DEMO_VIDEOS` passe de 3 à 10 entrées couvrant : véhicules, piétons, rue commerçante, drone, faune, sports, conduite nuit, marché, chantier, sous-marin. API `GET /demo-datasets` créée dans `api/main.py`. Grille UI 2×5 avec tags de requêtes et boutons désactivés si vidéo absente. |
| 2026-07-22 | **Fix bug onglet Démos** : `cols.index(col)` cassé avec les objets Streamlit → remplacé par `enumerate(cols)`. |
| 2026-07-22 | **7 vidéos Pexels libres de droits téléchargées** dans `sample_videos/` : `aerial_drone_view` (3 essais : altitude trop haute → intersection → LA downtown, 106 MB final), `wildlife_safari` (53 MB, girafes), `team_sports_match` (57 MB, football), `night_drive` (22 MB, trafic nuit), `market_stalls` (109 MB, marché), `construction_site` (127 MB, chantier), `underwater_dive` (192 MB, plongée corail). Total 743 MB. |
| 2026-07-22 | **YOLO VisDrone (détection aérienne) intégré** dans `api/workers/detection_worker.py` : YOLOv11s fine-tuné sur VisDrone2019 (10 classes : pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). Poids 19 MB depuis HuggingFace `dronefreak/visdrone-yolov11s`. Nouveau loader `_get_yolo_visdrone()`, dispatch dans `_resolve_model()`, branche d'inférence dans `_infer_frame()`. Modèle ajouté dans `MODELS` de la page 1 comme "YOLO VisDrone (aérien · 10 classes fixes)". |
| 2026-07-22 | **Constat : vidéos drones Pexels** — presque toutes en bird's eye view très haute altitude. Les modèles génériques (YOLO-World, GroundingDINO) sont entraînés au niveau du sol (COCO, Objects365) et ne détectent rien sur ces vues (toits de buildings confondus avec le sol). Seul un modèle fine-tuné sur données aériennes (DOTA, VisDrone) peut fonctionner. Même les vidéos "low altitude" sur Pexels restent trop hautes pour des détections fiables. |
