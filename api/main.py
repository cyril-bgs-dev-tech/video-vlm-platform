"""
FastAPI Gateway — Video VLM Platform
Routes complètes : upload, SAM3/SAM2, VLM, détection, clustering,
                   analytics, search, WebSocket streaming
"""
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import aiofiles
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from workers.celery_app import celery_app
from logging_config import configure_logging, get_logger
from routers.upload import router as upload_router
from routers.tasks import router as tasks_router
from fastapi.responses import HTMLResponse
from pathlib import Path

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Video VLM Platform API",
    version="2.0.0",
    description="SAM3.1 · GroundingDINO · YOLO-World · CLIP · LLaVA — RTX 5090",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(tasks_router)

# Servir la page HTML temps réel
REALTIME_HTML = Path("/app/realtime.html")

DEMO_DATASETS = [
    {
        "name": "Trafic urbain dense",
        "file": "urban_traffic.mp4",
        "queries": ["car", "truck", "bus", "person", "traffic light", "motorcycle", "sign"],
        "icon": "🚗",
        "specialty": "véhicules & transport",
        "duration_sec": 60,
        "description": "Carrefour urbain très fréquenté — voitures, bus, deux-roues, piétons et feux tricolores"
    },
    {
        "name": "Piétons & foule",
        "file": "street_pedestrians.mp4",
        "queries": ["person", "backpack", "handbag", "umbrella", "stroller", "dog"],
        "icon": "🚶",
        "specialty": "personnes & actions",
        "duration_sec": 45,
        "description": "Passage piéton animé — personnes seules, groupes, accessoires et animaux"
    },
    {
        "name": "Rue commerçante",
        "file": "bustling_street.mp4",
        "queries": ["person", "car", "bicycle", "bus", "traffic light", "sign", "storefront"],
        "icon": "🏙️",
        "specialty": "scène urbaine complète",
        "duration_sec": 90,
        "description": "Avenue commerçante — piétons, vitrines, véhicules, panneaux et mobilité douce"
    },
    {
        "name": "Vue aérienne drone",
        "file": "aerial_drone_view.mp4",
        "queries": ["car", "vehicle", "person", "building", "traffic light", "bus", "motorcycle"],
        "icon": "🛩️",
        "specialty": "géospatial & drone",
        "duration_sec": 34,
        "description": "Centre-ville vu du drone — voitures, piétons, buildings et traffic entre les gratte-ciel"
    },
    {
        "name": "Faune & animaux",
        "file": "wildlife_safari.mp4",
        "queries": ["elephant", "giraffe", "zebra", "bird", "tree", "grass", "animal"],
        "icon": "🦁",
        "specialty": "animaux & biodiversité",
        "duration_sec": 70,
        "description": "Savane africaine — faune sauvage dans son habitat naturel, grands mammifères et oiseaux"
    },
    {
        "name": "Sports collectifs",
        "file": "team_sports_match.mp4",
        "queries": ["person", "ball", "player", "goal", "net", "crowd", "uniform"],
        "icon": "⚽",
        "specialty": "sports & action",
        "duration_sec": 80,
        "description": "Match de sport collectif — joueurs en mouvement, ballon, but et public"
    },
    {
        "name": "Conduite de nuit",
        "file": "night_drive.mp4",
        "queries": ["car", "traffic light", "sign", "person", "street light", "building", "headlight"],
        "icon": "🌙",
        "specialty": "conduite & faible luminosité",
        "duration_sec": 65,
        "description": "Trafic nocturne — véhicules, éclairage public, panneaux et piétons sous faible lumière"
    },
    {
        "name": "Marché & étals",
        "file": "market_stalls.mp4",
        "queries": ["person", "fruit", "vegetable", "box", "table", "bottle", "canopy"],
        "icon": "🧺",
        "specialty": "retail & marchés",
        "duration_sec": 55,
        "description": "Marché de plein air — étals, produits frais, clients et marchandises diverses"
    },
    {
        "name": "Chantier BTP",
        "file": "construction_site.mp4",
        "queries": ["person", "truck", "crane", "hard hat", "tool", "concrete", "excavator"],
        "icon": "🏗️",
        "specialty": "industrie & construction",
        "duration_sec": 70,
        "description": "Chantier de construction — engins, ouvriers, équipements de sécurité et matériaux"
    },
    {
        "name": "Sous-marin & plongée",
        "file": "underwater_dive.mp4",
        "queries": ["fish", "coral", "diver", "turtle", "jellyfish", "reef", "sea"],
        "icon": "🐠",
        "specialty": "sous-marin & océan",
        "duration_sec": 60,
        "description": "Fond marin tropical — poissons, coraux, plongeur et vie aquatique colorée"
    },
]

@app.get("/demo-datasets")
async def list_demo_datasets():
    """Retourne la liste des jeux de données de démonstration disponibles."""
    return {"datasets": DEMO_DATASETS, "total": len(DEMO_DATASETS)}

@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page():
    """Page HTML temps réel pour détection WebSocket"""
    if REALTIME_HTML.exists():
        return HTMLResponse(content=REALTIME_HTML.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Page temps réel non disponible</h1>", status_code=404)

# Servir les vidéos pour la page HTML
from fastapi.responses import FileResponse

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Retourne le fichier vidéo pour lecture dans le navigateur"""
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if video_files:
        return FileResponse(video_files[0], media_type="video/mp4")
    return JSONResponse(content={"error": "Video not found"}, status_code=404)

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/data/videos"))
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


# ─────────────────────────────────────────────────────────────────────────────
# Helper : await task sans bloquer asyncio
# ─────────────────────────────────────────────────────────────────────────────
async def _await_task(task, timeout: float = 300) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: task.get(timeout=timeout))


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class SAM3TextRequest(BaseModel):
    video_id:    str
    text_prompt: str
    propagate:   bool = True

class SAM2Request(BaseModel):
    video_id:  str
    frame:     int
    prompt:    dict
    propagate: bool = False

class VLMQueryRequest(BaseModel):
    video_id:   str
    model:      str
    question:   str
    start_time: float = 0.0
    end_time:   float = 10.0

class ClusteringRequest(BaseModel):
    video_id:           str
    algorithm:          str   = "hdbscan"
    n_clusters:         int   = 8
    clip_model:         str   = "ViT-L/14"
    frame_sampling:     str   = "1 fps"
    min_scene_duration: float = 3.0
    umap_neighbors:     int   = 15
    detect_transitions: bool  = True

class DetectionRequest(BaseModel):
    video_id:     str
    model:        str = "grounding_dino_swinb"
    queries:      list[str]
    confidence:   float = 0.35
    nms_thresh:   float = 0.45
    batch_size:   int = 8
    use_half:     bool = True
    track:        bool = True
    temporal_nms: bool = True

class CompositionRequest(BaseModel):
    video_id:           str
    composition_type:   str = "hierarchical"
    queries:            list[str]
    confidence:         float = 0.35
    track:              bool = True
    fast_fps:           int = 15
    semantic_fps:       int = 3
    fps:                int = 5
    iou_threshold:      float = 0.5
    refinement_passes:  int = 2

class SearchIndexRequest(BaseModel):
    video_id:  str
    clip_model: str = "ViT-L/14"
    fps:        int = 2

class SearchQueryRequest(BaseModel):
    video_id:    str
    query_text:  Optional[str]  = None
    query_image: Optional[str]  = None
    query_texts: Optional[list[str]] = None
    top_n:       int   = 10
    threshold:   float = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    models_path = Path(os.getenv("MODELS_PATH", "/models"))
    yolo_pt = models_path / "yolo_world_v2_x.pt"
    yolo_ok = yolo_pt.exists() and yolo_pt.stat().st_size > 1_000_000
    try:
        insp = celery_app.control.inspect(timeout=2)
        workers = insp.active() or {}
        ping = insp.ping() or {}
        stats = insp.stats() or {}
    except Exception:
        workers, ping, stats = {}, {}, {}

    n_workers = len(ping)
    return {
        "status": "ok" if n_workers else "degraded",
        "version": "2.0.0",
        "allow_mock": os.getenv("ALLOW_MOCK", "1"),
        "workers": workers,
        "workers_online": list(ping.keys()),
        "workers_count": n_workers,
        "worker_pools": {
            name: (s.get("pool") or {}).get("implementation")
            for name, s in stats.items()
        },
        "models": {
            "sam3": (models_path / "sam3" / "sam3.1_large.pt").exists()
                    or (models_path / "sam3" / "sam3.1.pt").exists(),
            "sam2": any(models_path.glob("sam2/*.pt")),
            "grounding_dino_hf": True,  # cache HF lazy
            "yolo_world": yolo_ok,
            "yolo_world_bytes": yolo_pt.stat().st_size if yolo_pt.exists() else 0,
        },
    }


# ── Upload (via routers/upload.py) ────────────────────────────────────────────


# ── SAM3 (texte natif) ────────────────────────────────────────────────────────
@app.post("/vlm/sam3", status_code=202)
async def sam3_text(req: SAM3TextRequest):
    """SAM3.1 : segmente tous les objets correspondant au texte dans la vidéo."""
    task = celery_app.send_task(
        "workers.vlm_worker.run_sam3_text",
        kwargs=req.model_dump(),
        queue="vlm_queue",
    )
    return {"task_id": task.id, "model": "sam3.1"}


# ── SAM2 (prompt visuel + texte via GroundingDINO) ────────────────────────────
@app.post("/vlm/sam2", status_code=202)
async def sam2_segment(req: SAM2Request):
    """SAM2.1 : segmentation par point/box/texte sur une frame."""
    task = celery_app.send_task(
        "workers.vlm_worker.run_sam2",
        kwargs=req.model_dump(),
        queue="vlm_queue",
    )
    return {"task_id": task.id, "model": "sam2.1"}


# ── VLM Query ─────────────────────────────────────────────────────────────────
@app.post("/vlm/query", status_code=202)
async def vlm_query(req: VLMQueryRequest):
    task = celery_app.send_task(
        "workers.vlm_worker.run_vlm_query",
        kwargs=req.model_dump(),
        queue="vlm_queue",
    )
    return {"task_id": task.id}


# ── Task polling (via routers/tasks.py) ───────────────────────────────────────


# ── Clustering ────────────────────────────────────────────────────────────────
@app.post("/clustering/scenes", status_code=202)
async def cluster_scenes(req: ClusteringRequest):
    task = celery_app.send_task(
        "workers.clustering_worker.run_scene_clustering",
        kwargs=req.model_dump(),
        queue="clustering_queue",
    )
    return {"task_id": task.id}


# ── Detection ─────────────────────────────────────────────────────────────────
@app.post("/detection/run", status_code=202)
async def run_detection(req: DetectionRequest):
    task = celery_app.send_task(
        "workers.detection_worker.run_detection",
        kwargs=req.model_dump(),
        queue="detection_queue",
    )
    return {"task_id": task.id}


# ── Compositions VLM (littérature 2025-2026) ──────────────────────────────────
@app.post("/composition/run", status_code=202)
async def run_composition(req: CompositionRequest):
    """
    Lancer une composition conceptuelle de modèles VLM.
    
    Types disponibles:
    - hierarchical: YOLO-World → LocateAnything → ByteTrack (60-80% réduction charge)
    - ensembling: GroundingDINO + LocateAnything → fusion NMS (robustesse accrue)
    - iterative_refinement: LocateAnything → InternVL3.5 → validation (haute précision)
    """
    composition_map = {
        "hierarchical": "workers.detection_worker.run_composition_hierarchical",
        "ensembling": "workers.detection_worker.run_composition_ensembling",
        "iterative_refinement": "workers.detection_worker.run_composition_iterative",
    }
    
    task_name = composition_map.get(req.composition_type)
    if not task_name:
        return JSONResponse(
            status_code=400,
            content={"error": f"Composition type '{req.composition_type}' not supported. "
                            f"Available: {list(composition_map.keys())}"}
        )
    
    task = celery_app.send_task(
        task_name,
        kwargs=req.model_dump(),
        queue="detection_queue",
    )
    return {"task_id": task.id, "composition": req.composition_type}


# ── Analytics ─────────────────────────────────────────────────────────────────
@app.post("/analytics/ocr", status_code=202)
async def run_ocr(data: dict):
    task = celery_app.send_task(
        "workers.detection_worker.run_ocr",
        kwargs=data, queue="detection_queue"
    )
    return {"task_id": task.id}

@app.post("/analytics/reid_track", status_code=202)
async def run_reid(data: dict):
    task = celery_app.send_task(
        "workers.detection_worker.run_reid_tracking",
        kwargs=data, queue="detection_queue"
    )
    return {"task_id": task.id}

@app.post("/analytics/actions", status_code=202)
async def run_actions(data: dict):
    task = celery_app.send_task(
        "workers.vlm_worker.run_action_recognition",
        kwargs=data, queue="vlm_queue"
    )
    return {"task_id": task.id}

@app.post("/analytics/summary", status_code=202)
async def run_summary(data: dict):
    task = celery_app.send_task(
        "workers.vlm_worker.run_summary",
        kwargs=data, queue="vlm_queue"
    )
    return {"task_id": task.id}


# ── Semantic Search ───────────────────────────────────────────────────────────
@app.post("/search/index", status_code=202)
async def index_video(req: SearchIndexRequest):
    task = celery_app.send_task(
        "workers.clustering_worker.index_video_embeddings",
        kwargs=req.model_dump(), queue="clustering_queue"
    )
    return {"task_id": task.id}

@app.post("/search/query", status_code=202)
async def search_query(req: SearchQueryRequest):
    task = celery_app.send_task(
        "workers.clustering_worker.search_similar_frames",
        kwargs=req.model_dump(), queue="clustering_queue"
    )
    return {"task_id": task.id}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Stream de détection temps réel
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/detect/{video_id}")
async def ws_detect(websocket: WebSocket, video_id: str):
    await websocket.accept()
    redis_conn = None
    try:
        config = await websocket.receive_json()
        model = config.get("model", "yolo_world_v2")
        query = config.get("query", "person")
        queries = config.get("queries")
        fps = int(config.get("fps", 5))
        confidence = float(config.get("confidence", 0.35))
        send_jpeg = bool(config.get("send_jpeg", True))
        max_width = int(config.get("max_width", 960))
        jpeg_quality = int(config.get("jpeg_quality", 70))
        pace = str(config.get("pace", "realtime"))
        channel = f"ws:detect:{video_id}:{uuid.uuid4().hex[:8]}"

        # Subscribe AVANT d'envoyer la task (évite perte des 1ers messages)
        import redis.asyncio as aioredis
        redis_conn = aioredis.from_url(REDIS_URL, decode_responses=True)
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(channel)

        celery_app.send_task(
            "workers.detection_worker.run_detection_stream",
            kwargs={
                "video_id": video_id,
                "model": model,
                "query": query,
                "queries": queries,
                "fps": fps,
                "confidence": confidence,
                "ws_channel": channel,
                "send_jpeg": send_jpeg,
                "max_width": max_width,
                "jpeg_quality": jpeg_quality,
                "pace": pace,
            },
            queue="detection_queue",
        )

        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
                if data.get("type") in ("done", "error"):
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if redis_conn:
            try:
                await redis_conn.aclose()
            except Exception:
                pass
